"""Cost sensitivity on the FAITHFUL full history — the agent's dominant drag (cost -0.47, ~100% of gross in 2022/23).
Deployable cost lever = maker execution (~1bps) vs current taker (4.5). Sweep COST_BPS_LEG {1,2,3,4.5,6}. If the
thin years (2022/23) flip strongly positive at low cost, the +0.91 is cost-suppressed and maker execution is the
single biggest deployable lever. per-year + overall daily Sharpe."""
import os, sys, subprocess; from pathlib import Path
import numpy as np, pandas as pd, json
REPO=Path("/home/yuqing/ctaNew"); FH=REPO/"live/state/v3loop/fullhist_mpit"; ANN=np.sqrt(365)
import warnings; warnings.filterwarnings("ignore")
BASE=dict(STRAT_K="3",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")
def dsh(s): d=(s.fillna(0)/1e4).resample("1D").sum(); return d.mean()/d.std()*ANN if d.std()>0 else float('nan')
def run(cost):
    out=REPO/f"live/state/v3loop/cs_{cost}"; out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ); env.update(BASE); env.update(COST_BPS_LEG=str(cost),PYTHONPATH=str(REPO),CONVEXITY_STATE=str(out),
        CONVEXITY_PREDS_PATH=str(FH/"base.parquet"),CONVEXITY_PREDS_LONG=str(FH/"long.parquet")); env.pop("CONVEXITY_UNIVERSE_META",None)
    r=subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=str(REPO),
                     stdout=open(out/"run.log","w"),stderr=subprocess.STDOUT)
    c=pd.read_csv(out/"cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c=c.sort_values("open_time").set_index("open_time")
    return dict(cost_bps=cost, overall=round(dsh(c["pnl_bps"]),3), totpnl=round(c["pnl_bps"].sum()),
                per_year={int(y):round(dsh(g["pnl_bps"]),2) for y,g in c.groupby(c.index.year)})
for cst in [1,2,3,4.5,6]:
    print(json.dumps(run(cst)), flush=True)
print("DONE costsens")
