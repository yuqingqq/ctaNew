"""Full-history backtest with a PIT UNIVERSE (fixes the look-ahead fixed exclude_high_vol list). At each bar the
tradeable book = symbols that are (a) MATURE (listed >= MIN_DAYS before the bar) AND (b) LOW-VOL by TRAILING rvol_7d
(<= cross-sectional median among mature names that bar). Replaces the recent-dated full-sample exclude list. Re-runs
the v2 replay + per-year honest Sharpe, and re-checks the opportunity regime gate on the corrected result.
"""
import os, sys, subprocess; from pathlib import Path
import numpy as np, pandas as pd
REPO=Path("/home/yuqing/ctaNew"); FH=REPO/"live/state/v3loop/fullhist"; OUT=REPO/"live/state/v3loop/fullhist_pit"
OUT.mkdir(parents=True,exist_ok=True); ANN=np.sqrt(365); MIN_DAYS=180
import warnings; warnings.filterwarnings("ignore")

# PIT vol + maturity from the panel. Match production rvol_window_days=30: trailing-30d (180-bar) realized vol
# of REALIZED 4h returns (= return_pct.shift(1), already-settled), shifted => strictly PIT.
pan=pd.read_parquet(REPO/"outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","return_pct"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True)
pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
first_seen=pan[pan.return_pct.notna()].groupby("symbol")["open_time"].min()
pan["mature"]=pan["open_time"] >= pan["symbol"].map(first_seen)+pd.Timedelta(days=MIN_DAYS)
pan["vol30"]=pan.groupby("symbol")["return_pct"].transform(lambda s: s.shift(1).rolling(180,min_periods=60).std())
pan["rvol_7d"]=pan["vol30"]   # reuse var name below
m=pan[pan["mature"] & pan["rvol_7d"].notna()].copy()
med=m.groupby("open_time")["rvol_7d"].transform("median")
m["lowvol"]=m["rvol_7d"]<=med
mask=m[m["lowvol"]][["symbol","open_time"]]    # PIT-eligible (mature+lowvol) rows
print(f"PIT-eligible rows: {len(mask)}; per-year eligible-distinct-syms:")
mm=mask.copy(); mm["yr"]=mm.open_time.dt.year
print("  ",mm.groupby("yr")["symbol"].nunique().to_dict(),flush=True)

# filter the full (un-excluded) preds to the PIT mask
for src,dst in [("short_full.parquet","base.parquet"),("long_full.parquet","long.parquet")]:
    d=pd.read_parquet(FH/src); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d.merge(mask,on=["symbol","open_time"],how="inner")
    d.to_parquet(OUT/dst);
print(f"PIT preds written ({pd.read_parquet(OUT/'base.parquet').shape[0]} base rows)",flush=True)

# replay
PROD=dict(COST_BPS_LEG="4.5",STRAT_K="3",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")
env=dict(os.environ); env.update(PROD); env.update(PYTHONPATH=str(REPO),CONVEXITY_STATE=str(OUT),
    CONVEXITY_PREDS_PATH=str(OUT/"base.parquet"),CONVEXITY_PREDS_LONG=str(OUT/"long.parquet")); env.pop("CONVEXITY_UNIVERSE_META",None)
r=subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=str(REPO),
                 stdout=open(OUT/"run.log","w"),stderr=subprocess.STDOUT)
print(f"replay rc {r.returncode}",flush=True)
c=pd.read_csv(OUT/"cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c=c.sort_values("open_time").reset_index(drop=True)
rr=c["pnl_bps"].fillna(0)/1e4
def dsh(x): d=pd.Series(np.asarray(x),index=c["open_time"]).resample("1D").sum(); return d.mean()/d.std()*ANN if d.std()>0 else np.nan
print(f"\n=== PIT-UNIVERSE FULL-HISTORY honest daily Sharpe ===")
print(f"  OVERALL {c.open_time.min().date()}->{c.open_time.max().date()}: dSharpe {dsh(rr):+.2f}  totPnL {c['pnl_bps'].sum():+.0f}  cycles {len(c)}")
for yr,g in c.groupby(c["open_time"].dt.year):
    gd=pd.Series(g["pnl_bps"].fillna(0).values/1e4,index=g["open_time"]).resample("1D").sum()
    print(f"  {yr}: dSharpe {gd.mean()/gd.std()*ANN if gd.std()>0 else float('nan'):+.2f}  totPnL {g['pnl_bps'].sum():+7.0f}  cycles {len(g)}  gross {g['long_ret_bps'].sum()+g['short_ret_bps'].sum():+.0f}")
# regime gate on PIT result
c["gs"]=c["long_ret_bps"]+c["short_ret_bps"]
s_eq=rr.shift(1).rolling(360).sum(); s_sp=c["gs"].shift(1).rolling(180).mean()
t_eq=s_eq.shift(1).rolling(360,min_periods=90).quantile(.5); t_sp=s_sp.shift(1).rolling(360,min_periods=90).quantile(.5)
on=((s_eq>t_eq)&(s_sp>t_sp)).fillna(False)
print(f"\n  + opportunity gate on PIT: dSharpe {dsh(rr.where(on,0.0)):+.2f} (trades {100*on.mean():.0f}%)")
print("DONE fullhist_pit",flush=True)
