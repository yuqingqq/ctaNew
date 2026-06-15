"""Drawback: concentration / universe fragility (top-10 names = 77% of PnL). Root-cause test: how sensitive is the
edge to symbol composition? Drop K random symbols from the preds, replay, measure honest daily Sharpe distribution.
If dropping a few names collapses it / wide spread -> the edge is symbol-overfit (forward risk widens). If robust
-> composition-insensitive. Parallel replays; honest daily metric.
  python live/phase_universe_stress.py            # K in {3,6,10}, 20 seeds each
"""
import os, sys, subprocess, json, shutil
import concurrent.futures as cf
import numpy as np, pandas as pd
ROOT="/home/yuqing/ctaNew"; BASE=f"{ROOT}/live/state/exp_xs94/baseline"; PY=sys.executable
PROD=dict(COST_BPS_LEG="4.5",STRAT_K="3",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")
ANN=np.sqrt(365)
KS=[3,6,10]; NSEED=20; WORKERS=10
b0=pd.read_parquet(f"{BASE}/base_mpit.parquet"); l0=pd.read_parquet(f"{BASE}/long_mpit.parquet")
syms=sorted(b0["symbol"].unique()); print(f"universe size {len(syms)}", flush=True)

def dsh_from(cc):
    c=pd.read_csv(cc); c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c=c.sort_values("open_time")
    s=c.set_index("open_time")["pnl_bps"].fillna(0)/1e4; d=s.resample("1D").sum()
    eq=s.cumsum(); return float(d.mean()/d.std()*ANN), float((eq-eq.cummax()).min()), float(c["pnl_bps"].sum())

def run_draw(args):
    K, seed = args
    rng=np.random.default_rng(1000*K+seed); drop=set(rng.choice(syms, K, replace=False))
    keep=[s for s in syms if s not in drop]
    out=f"{ROOT}/live/state/v3loop/us_K{K}_s{seed}"; os.makedirs(out, exist_ok=True)
    b0[b0["symbol"].isin(keep)].to_parquet(f"{out}/base.parquet")
    l0[l0["symbol"].isin(keep)].to_parquet(f"{out}/long.parquet")
    env=dict(os.environ); env.update(PROD)
    env.update(PYTHONPATH=ROOT,CONVEXITY_STATE=out,CONVEXITY_PREDS_PATH=f"{out}/base.parquet",
               CONVEXITY_PREDS_LONG=f"{out}/long.parquet"); env.pop("CONVEXITY_UNIVERSE_META",None)
    r=subprocess.run([PY,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=ROOT,
                     stdout=open(f"{out}/run.log","w"),stderr=subprocess.STDOUT)
    cc=f"{out}/cycles.csv"
    res=(None if r.returncode!=0 or not os.path.exists(cc) else dsh_from(cc))
    shutil.rmtree(out, ignore_errors=True)   # cleanup temp preds (gitignored anyway)
    return K, seed, res

BASE_SH=3.679
jobs=[(K,s) for K in KS for s in range(NSEED)]
by_k={K:[] for K in KS}
with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for K,seed,res in ex.map(run_draw, jobs):
        if res: by_k[K].append(res[0])
        print(f"K{K} s{seed}: {'fail' if not res else round(res[0],3)}", flush=True)
print("\n=== UNIVERSE FRAGILITY (honest daily Sharpe, baseline +3.679) ===")
for K in KS:
    a=np.array(by_k[K])
    if len(a)==0: print(f"K={K}: all failed"); continue
    print(f"K={K:2d} drop ({len(syms)-K} syms): mean {a.mean():+.2f}  std {a.std():.2f}  min {a.min():+.2f}  max {a.max():+.2f}  %>=baseline {100*(a>=BASE_SH).mean():.0f}%")
