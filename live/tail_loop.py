"""Tail-risk optimization loop driver (2026-06-14). Targets the ONE identified drawback: the bear-regime
short-SQUEEZE correlated-blowup tail (worst 1% of cycles = -46% of PnL). Honest metrics throughout:
  - Sharpe = DAILY-resampled (autocorr-correct, not 4h-naive)
  - tail   = worst-1% cycle sum (CVaR-ish), p01, kurtosis, maxDD
  - funding= optional per-variant carry (set FUND=1; slow)
Each variant runs the real bot --replay-all with PROD_ENV + override, then we post-process its cycles.csv.

  python live/tail_loop.py <phaseA|...>      # runs the variant set for that phase, prints one JSON/line
"""
import sys, os, subprocess, json, time
import numpy as np, pandas as pd

ROOT="/home/yuqing/ctaNew"; BASE=f"{ROOT}/live/state/exp_xs94/baseline"
PY=sys.executable
PROD_ENV=dict(COST_BPS_LEG="4.5", STRAT_K="3", SIDE_MODE="default", XS_LEAN="1", CONVEXITY_PIT_DVOL="1",
              BEAR_MODE="equal", STOP_SKIP_REGIMES="bear", SIDE_BETA_NEUT="0", BEAR_K="2",
              SIZING_MODE="inv_vol", LONG_MAX_RET3D="0.20")

PHASES = {
  # bear de-gross frontier — the deployment risk/return frontier, measured HONESTLY
  "phaseA": [
     ("baseline",      {}),
     ("bear_flat",     {"BEAR_MODE":"flat"}),
     ("bear_bg0.3",    {"BEAR_GROSS_MULT":"0.3"}),
     ("bear_bg0.5",    {"BEAR_GROSS_MULT":"0.5"}),
     ("bear_bg0.7",    {"BEAR_GROSS_MULT":"0.7"}),
  ],
}

def run_variant(label, ov):
    out=f"{ROOT}/live/state/v3loop/tl_{label}"; os.makedirs(out, exist_ok=True)
    env=dict(os.environ); env.update(PROD_ENV); env.update(ov)
    env.update(PYTHONPATH=ROOT, CONVEXITY_STATE=out,
               CONVEXITY_PREDS_PATH=f"{BASE}/base_mpit.parquet", CONVEXITY_PREDS_LONG=f"{BASE}/long_mpit.parquet")
    env.pop("CONVEXITY_UNIVERSE_META", None)
    r=subprocess.run([PY,"-m","live.convexity_paper_bot","--replay-all"], env=env, cwd=ROOT,
                     stdout=open(f"{out}/run.log","w"), stderr=subprocess.STDOUT)
    cc=f"{out}/cycles.csv"
    if r.returncode!=0 or not os.path.exists(cc): return None
    return cc

def honest(cc):
    c=pd.read_csv(cc); c["open_time"]=pd.to_datetime(c["open_time"],utc=True)
    c=c.sort_values("open_time").set_index("open_time")
    r=c["pnl_bps"].fillna(0)/1e4
    d=r.resample("1D").sum()
    sh_naive=r.mean()/r.std()*np.sqrt(6*365)
    sh_daily=d.mean()/d.std()*np.sqrt(365)
    eq=c["pnl_bps"].fillna(0).cumsum(); dd=float((eq-eq.cummax()).min())
    x=c["pnl_bps"].dropna()
    w1=float(x.nsmallest(max(1,len(x)//100)).sum())
    return dict(sharpe_daily=round(sh_daily,3), sharpe_naive=round(sh_naive,3),
                totpnl=round(float(c["pnl_bps"].sum())), maxdd=round(dd),
                worst1pct=round(w1), p01=round(float(x.quantile(.01))), kurt=round(float(x.kurtosis()),1),
                n_bear_degrossed=int((c["regime"]=="bear").sum()))

def main():
    phase=sys.argv[1]
    base=None
    for label, ov in PHASES[phase]:
        t=time.time(); cc=run_variant(label, ov)
        if cc is None: print(json.dumps(dict(label=label, error="replay failed"))); continue
        h=honest(cc);
        if label=="baseline": base=h
        rec=dict(label=label, overrides=ov, **h, elapsed_s=round(time.time()-t))
        if base: rec["d_sharpe_daily"]=round(h["sharpe_daily"]-base["sharpe_daily"],3); \
                 rec["maxdd_pct"]=round((h["maxdd"]-base["maxdd"])/abs(base["maxdd"])*100); \
                 rec["worst1_pct_change"]=round((h["worst1pct"]-base["worst1pct"])/abs(base["worst1pct"])*100)
        print(json.dumps(rec), flush=True)

if __name__=="__main__": main()
