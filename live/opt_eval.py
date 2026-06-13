"""Standardized one-shot evaluator for the optimization workflow. Runs the v2 bot replay with the fixed
production env + caller-supplied env OVERRIDES, then prints a single JSON line with Sharpe/maxDD/totPnL/per-fold
and folds_positive vs the +4.22 baseline. Agents call this so every test is apples-to-apples.

  usage: python live/opt_eval.py <label> "ENV1=v1 ENV2=v2 ..."     (override string may be empty)
  e.g.   python live/opt_eval.py betaneut "SIDE_BETA_NEUT=1"
"""
import sys, os, subprocess, json
import numpy as np, pandas as pd

ROOT = "/home/yuqing/ctaNew"
BASE = f"{ROOT}/live/state/exp_xs94/baseline"
ANN = np.sqrt(6*365)
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27","2026-06-30"]]
PROD_ENV = dict(COST_BPS_LEG="4.5", STRAT_K="3", SIDE_MODE="default", XS_LEAN="1", CONVEXITY_PIT_DVOL="1",
                BEAR_MODE="equal", STOP_SKIP_REGIMES="bear", SIDE_BETA_NEUT="0", BEAR_K="2",
                SIZING_MODE="inv_vol", LONG_MAX_RET3D="0.20")

def fold(t):
    for i in range(len(CUTS)-1):
        if CUTS[i] <= t < CUTS[i+1]: return i
    return -1

def perfold(cyc):
    cyc = cyc.copy(); cyc["f"] = cyc["open_time"].map(fold)
    return cyc.groupby("f")["pnl_bps"].sum()

def stats(cycles_csv):
    c = pd.read_csv(cycles_csv); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    p = c["pnl_bps"].dropna()/1e4
    eq = c["pnl_bps"].fillna(0).cumsum()
    return dict(sharpe=float(p.mean()/p.std()*ANN), totpnl=float(c["pnl_bps"].sum()),
                maxdd=float((eq-eq.cummax()).min())), perfold(c)

def main():
    label = sys.argv[1]
    ov = sys.argv[2] if len(sys.argv) > 2 else ""
    overrides = dict(kv.split("=",1) for kv in ov.split()) if ov.strip() else {}
    out = f"{ROOT}/live/state/v3loop/wf_{label}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ); env.update(PROD_ENV); env.update(overrides)
    env.update(PYTHONPATH=ROOT, CONVEXITY_STATE=out,
               CONVEXITY_PREDS_PATH=f"{BASE}/base_mpit.parquet",
               CONVEXITY_PREDS_LONG=f"{BASE}/long_mpit.parquet")
    env.pop("CONVEXITY_UNIVERSE_META", None)   # default PANEL = true listing dates
    r = subprocess.run([sys.executable, "-m", "live.convexity_paper_bot", "--replay-all"],
                       env=env, cwd=ROOT, stdout=open(f"{out}/run.log","w"), stderr=subprocess.STDOUT)
    cc = f"{out}/cycles.csv"
    if r.returncode != 0 or not os.path.exists(cc):
        print(json.dumps(dict(label=label, error="replay failed", tail=open(f"{out}/run.log").read()[-600:]))); return
    s, fp = stats(cc)
    bs, bfp = stats(f"{ROOT}/live/state/v3loop/iter5_tilt0/cycles.csv")
    d = (fp - bfp).reindex(range(9)).fillna(0)
    print(json.dumps(dict(
        label=label, overrides=overrides,
        sharpe=round(s["sharpe"],3), maxdd=round(s["maxdd"],0), totpnl=round(s["totpnl"],0),
        baseline_sharpe=round(bs["sharpe"],3), baseline_maxdd=round(bs["maxdd"],0),
        lift=round(s["sharpe"]-bs["sharpe"],3), maxdd_change_pct=round((s["maxdd"]-bs["maxdd"])/abs(bs["maxdd"])*100,0),
        folds_positive=int((d>0).sum()), per_fold_delta=[round(x) for x in d.values])))

if __name__ == "__main__":
    main()
