"""K-breadth sweep on the FAITHFUL full history. Motivation: broad XS reversion stayed intact in the bad windows
(autocorr -0.04) but the EXTREME K=3 picks failed (idiosyncratic trends in tail names). Higher K = less extreme +
more diversified -> should harvest the robust broad reversion and dilute single-name tail failures. Test K=3..8,
per-year + Apr-Aug 2025 + tail, honest daily Sharpe."""
import os, sys, subprocess; from pathlib import Path
import numpy as np, pandas as pd, json
REPO=Path("/home/yuqing/ctaNew"); FH=REPO/"live/state/v3loop/fullhist_mpit"; ANN=np.sqrt(365)
import warnings; warnings.filterwarnings("ignore")
PROD=dict(COST_BPS_LEG="4.5",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")
def dsh(s): d=(s.fillna(0)/1e4).resample("1D").sum(); return d.mean()/d.std()*ANN if d.std()>0 else float('nan')
def run(K):
    out=REPO/f"live/state/v3loop/kb_{K}"; out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ); env.update(PROD); env.update(STRAT_K=str(K),PYTHONPATH=str(REPO),CONVEXITY_STATE=str(out),
        CONVEXITY_PREDS_PATH=str(FH/"base.parquet"),CONVEXITY_PREDS_LONG=str(FH/"long.parquet")); env.pop("CONVEXITY_UNIVERSE_META",None)
    r=subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=str(REPO),
                     stdout=open(out/"run.log","w"),stderr=subprocess.STDOUT)
    c=pd.read_csv(out/"cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c=c.sort_values("open_time").set_index("open_time")
    eq=c["pnl_bps"].fillna(0).cumsum(); mdd=float((eq-eq.cummax()).min())
    return dict(K=K, overall=round(dsh(c["pnl_bps"]),3), aprAug2025=round(dsh(c.loc["2025-04-01":"2025-08-31","pnl_bps"]),2),
                maxdd=round(mdd), totpnl=round(c["pnl_bps"].sum()),
                per_year={int(y):round(dsh(g["pnl_bps"]),2) for y,g in c.groupby(c.index.year)})
for K in [3,4,5,6,8]:
    print(json.dumps(run(K)), flush=True)
print("DONE kbreadth")
