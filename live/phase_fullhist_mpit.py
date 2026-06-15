"""FAITHFUL full-history backtest: apply PRODUCTION's exact monthly-PIT universe rule (exp_xs94_monthlypit) extended
to 2022 — exclude the high-vol cohort by trailing-30d-mean rvol_7d, re-ranked AS-OF each monthly fold (PIT). Production
excludes a fixed top-80 (universe ~153); for full history with a growing universe we exclude the top FRAC by trailing
rvol per fold (FRAC=0.52 ~ 80/153) so the low-vol cohort is consistent across years. Same preds as production
(corr +0.98). Reports honest daily Sharpe per year + the recent window for direct reconciliation with +3.68.
"""
import os, sys, subprocess; from pathlib import Path
import numpy as np, pandas as pd
REPO=Path("/home/yuqing/ctaNew"); FH=REPO/"live/state/v3loop/fullhist"; OUT=REPO/"live/state/v3loop/fullhist_mpit"
OUT.mkdir(parents=True,exist_ok=True); ANN=np.sqrt(365); FRAC=0.52
import warnings; warnings.filterwarnings("ignore")
import live.train_twobook_models as tt
CUTS=[pd.Timestamp(t,tz="UTC") for t in pd.date_range("2022-01-01","2026-06-01",freq="MS")]+[pd.Timestamp("2026-06-30",tz="UTC")]
pan=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","rvol_7d"]); pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True)
def excl_for(c0):                       # production rule: high-vol by trailing-30d-mean rvol_7d, as-of c0
    lo=c0-pd.Timedelta(days=30); rv=pan[(pan.open_time>=lo)&(pan.open_time<c0)].groupby("symbol")["rvol_7d"].mean().dropna()
    k=int(round(FRAC*len(rv)))
    return set(rv.sort_values(ascending=False).index[:k])
for book,full in [("base","short_full"),("long","long_full")]:
    d=pd.read_parquet(FH/f"{full}.parquet"); d["open_time"]=pd.to_datetime(d["open_time"],utc=True); keep=[]
    for i in range(len(CUTS)-1):
        ex=excl_for(CUTS[i]); w=d[(d.open_time>=CUTS[i])&(d.open_time<CUTS[i+1])]; keep.append(w[~w["symbol"].isin(ex)])
    pd.concat(keep,ignore_index=True).to_parquet(OUT/f"{book}.parquet")
print(f"mpit-rule preds written ({pd.read_parquet(OUT/'base.parquet').shape[0]} base rows)",flush=True)
PROD=dict(COST_BPS_LEG="4.5",STRAT_K="3",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")
env=dict(os.environ); env.update(PROD); env.update(PYTHONPATH=str(REPO),CONVEXITY_STATE=str(OUT),
    CONVEXITY_PREDS_PATH=str(OUT/"base.parquet"),CONVEXITY_PREDS_LONG=str(OUT/"long.parquet")); env.pop("CONVEXITY_UNIVERSE_META",None)
r=subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=str(REPO),
                 stdout=open(OUT/"run.log","w"),stderr=subprocess.STDOUT); print(f"replay rc {r.returncode}",flush=True)
c=pd.read_csv(OUT/"cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c=c.sort_values("open_time").set_index("open_time")
def dsh(x): d=(x.fillna(0)/1e4).resample("1D").sum(); return d.mean()/d.std()*ANN if d.std()>0 else float('nan')
print(f"\n=== FAITHFUL (production mpit-rule) FULL-HISTORY honest daily Sharpe ===")
print(f"  OVERALL: {dsh(c['pnl_bps']):+.2f}  totPnL {c['pnl_bps'].sum():+.0f}  cycles {len(c)}")
for yr,g in c.groupby(c.index.year): print(f"  {yr}: {dsh(g['pnl_bps']):+.2f}  PnL {g['pnl_bps'].sum():+7.0f}")
print(f"  recon Oct2025-Jun2026 (vs production +3.68): {dsh(c.loc['2025-10-04':'2026-06-30','pnl_bps']):+.2f}")
print("DONE fullhist_mpit",flush=True)
