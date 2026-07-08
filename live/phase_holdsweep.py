"""Hold-DOWN sweep on faithful full history. Finding: 24h-hold (6 sleeves) AMPLIFIES the tail (kurt 22->80.6, squeeze
compounds across sleeves). Prior sweep only tested LONGER holds (7/8/9, all worse); SHORTER (3/4/5) untested. Shorter
hold cuts squeeze-compounding (lower tail) but raises turnover/cost — net effect on faithful (bot counts the cost)?"""
import os, sys, subprocess; from pathlib import Path
import numpy as np, pandas as pd, json
REPO=Path("/home/yuqing/ctaNew"); FH=REPO/"live/state/v3loop/fullhist_mpit"; ANN=np.sqrt(365)
import warnings; warnings.filterwarnings("ignore")
BASE=dict(COST_BPS_LEG="4.5",STRAT_K="3",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")
def stats(c):
    c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c=c.sort_values("open_time").set_index("open_time")
    r=c["pnl_bps"].fillna(0)/1e4; d=r.resample("1D").sum(); sh=d.mean()/d.std()*ANN if d.std()>0 else np.nan
    eq=c["pnl_bps"].fillna(0).cumsum(); mdd=float((eq-eq.cummax()).min())
    x=c["pnl_bps"].dropna()
    return dict(overall=round(sh,3),kurt=round(float(x.kurtosis()),1),maxdd=round(mdd),totpnl=round(c["pnl_bps"].sum()),
                cost=round(c["cost_bps"].sum()),per_year={int(y):round((g["pnl_bps"].fillna(0)/1e4).resample("1D").sum().pipe(lambda d:d.mean()/d.std()*ANN if d.std()>0 else np.nan),2) for y,g in c.groupby(c.index.year)})
def run(H):
    out=REPO/f"live/state/v3loop/hold_{H}"; out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ); env.update(BASE); env.update(STRAT_HOLD=str(H),PYTHONPATH=str(REPO),CONVEXITY_STATE=str(out),
        CONVEXITY_PREDS_PATH=str(FH/"base.parquet"),CONVEXITY_PREDS_LONG=str(FH/"long.parquet")); env.pop("CONVEXITY_UNIVERSE_META",None)
    r=subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=str(REPO),
                     stdout=open(out/"run.log","w"),stderr=subprocess.STDOUT)
    if r.returncode!=0 or not (out/"cycles.csv").exists(): return dict(HOLD=H,error=open(out/"run.log").read()[-200:])
    return dict(HOLD=H, **stats(pd.read_csv(out/"cycles.csv")))
for H in [3,4,5,6]:
    print(json.dumps(run(H)), flush=True)
print("DONE holdsweep")
