"""Test the research agent's #1 lever: beta-neutralization on the FAITHFUL full history (where beta bleeds -2572bps,
-1516 in 2025). Earlier SIDE_BETA_NEUT=1 was rejected on the 8-MONTH window (beta wasn't the drag there); re-test on
full history. baseline=SIDE_BETA_NEUT=0 (+0.918). Question: does the bot's PIT per-name beta-neutral recover the
agent's ideal +0.18, or do noisy 4h betas kill it again?"""
import os, sys, subprocess; from pathlib import Path
import numpy as np, pandas as pd, json
REPO=Path("/home/yuqing/ctaNew"); FH=REPO/"live/state/v3loop/fullhist_mpit"; ANN=np.sqrt(365)
import warnings; warnings.filterwarnings("ignore")
BASE=dict(COST_BPS_LEG="4.5",STRAT_K="3",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")
def dsh(s): d=(s.fillna(0)/1e4).resample("1D").sum(); return d.mean()/d.std()*ANN if d.std()>0 else float('nan')
def run(label,extra):
    out=REPO/f"live/state/v3loop/bn_{label}"; out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ); env.update(BASE); env.update(extra); env.update(PYTHONPATH=str(REPO),CONVEXITY_STATE=str(out),
        CONVEXITY_PREDS_PATH=str(FH/"base.parquet"),CONVEXITY_PREDS_LONG=str(FH/"long.parquet")); env.pop("CONVEXITY_UNIVERSE_META",None)
    r=subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=str(REPO),
                     stdout=open(out/"run.log","w"),stderr=subprocess.STDOUT)
    if r.returncode!=0 or not (out/"cycles.csv").exists(): return dict(label=label, error=open(out/"run.log").read()[-300:])
    c=pd.read_csv(out/"cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c=c.sort_values("open_time").set_index("open_time")
    eq=c["pnl_bps"].fillna(0).cumsum(); mdd=float((eq-eq.cummax()).min())
    return dict(label=label, overall=round(dsh(c["pnl_bps"]),3), maxdd=round(mdd), totpnl=round(c["pnl_bps"].sum()),
                per_year={int(y):round(dsh(g["pnl_bps"]),2) for y,g in c.groupby(c.index.year)})
print(json.dumps(run("base_neut0",{"SIDE_BETA_NEUT":"0"})), flush=True)
print(json.dumps(run("betaneut1",{"SIDE_BETA_NEUT":"1"})), flush=True)
print("DONE betaneut")
