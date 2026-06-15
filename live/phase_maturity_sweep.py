"""Universe-quality loop: does a TIGHTER maturity gate (exclude young/fragile names) help the faithful full history,
especially the Apr-Aug 2025 fragile-name tail? Filter the faithful mpit preds to listing-age >= MIN_AGE before replay.
Baseline production MIN_HISTORY_DAYS=180. Test 180/270/365/540. Honest: per-year + Apr-Aug 2025 + tail, daily Sharpe.
"""
import os, sys, subprocess; from pathlib import Path
import numpy as np, pandas as pd
REPO=Path("/home/yuqing/ctaNew"); FH=REPO/"live/state/v3loop/fullhist_mpit"; ANN=np.sqrt(365)
import warnings; warnings.filterwarnings("ignore")
pan=pd.read_parquet(REPO/"outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","return_pct"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)]
first_seen=pan[pan.return_pct.notna()].groupby("symbol")["open_time"].min()
PROD=dict(COST_BPS_LEG="4.5",STRAT_K="3",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")
def dsh(s): d=(s.fillna(0)/1e4).resample("1D").sum(); return d.mean()/d.std()*ANN if d.std()>0 else float('nan')
def run(MIN_AGE):
    out=REPO/f"live/state/v3loop/mat_{MIN_AGE}"; out.mkdir(parents=True,exist_ok=True)
    for book in ["base","long"]:
        d=pd.read_parquet(FH/f"{book}.parquet"); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
        d["age"]=(d["open_time"]-d["symbol"].map(first_seen)).dt.days
        d[d["age"]>=MIN_AGE].drop(columns=["age"]).to_parquet(out/f"{book}.parquet")
    env=dict(os.environ); env.update(PROD); env.update(PYTHONPATH=str(REPO),CONVEXITY_STATE=str(out),
        CONVEXITY_PREDS_PATH=str(out/"base.parquet"),CONVEXITY_PREDS_LONG=str(out/"long.parquet")); env.pop("CONVEXITY_UNIVERSE_META",None)
    r=subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=str(REPO),
                     stdout=open(out/"run.log","w"),stderr=subprocess.STDOUT)
    c=pd.read_csv(out/"cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c=c.sort_values("open_time").set_index("open_time")
    yr={y:dsh(g["pnl_bps"]) for y,g in c.groupby(c.index.year)}
    aug=dsh(c.loc["2025-04-01":"2025-08-31","pnl_bps"])
    eq=c["pnl_bps"].fillna(0).cumsum(); mdd=float((eq-eq.cummax()).min())
    return dict(MIN_AGE=MIN_AGE, overall=round(dsh(c["pnl_bps"]),3), aprAug2025=round(aug,2),
                maxdd=round(mdd), totpnl=round(c["pnl_bps"].sum()), per_year={y:round(v,2) for y,v in yr.items()})
import json
for a in [180,270,365,540]:
    print(json.dumps(run(a)), flush=True)
print("DONE maturity_sweep")
