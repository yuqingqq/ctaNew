"""Emit one JSON line of metrics for a convexity replay state dir. Used by the dd-optimization loop.
Reports AGGREGATE + PER-REGIME + PER-MONTH so a candidate is never judged on the aggregate alone
(the 06-18→06-22 live drawdown was bear-dominated; aggregate dilutes a bear-specific win).

usage: python3 live/ddloop_eval.py <state_dir> [tag]
   state_dir contains state/cycles.csv (or cycles.csv directly)
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
ANN = np.sqrt(365)

def dsh(s):
    d = s.fillna(0).resample("1D").sum()
    return float(d.mean()/d.std()*ANN) if d.std() > 0 else float("nan")
def mdd(s):
    eq = s.fillna(0).cumsum(); return float((eq-eq.cummax()).min())
def cvar(s, q=0.05):
    t = s.quantile(q); return float(s[s <= t].mean())

def main():
    d = Path(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else d.name
    f = d/"state"/"cycles.csv"
    if not f.exists(): f = d/"cycles.csv"
    c = pd.read_csv(f); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    c = c.sort_values("open_time").set_index("open_time")
    p = c["pnl_bps"]
    out = {"tag": tag, "cycles": int(len(c)),
           "totPnL": round(float(p.sum()), 1), "Sharpe": round(dsh(p), 3),
           "maxDD": round(mdd(p), 1), "worstcyc": round(float(p.min()), 1),
           "CVaR5": round(cvar(p), 1), "meanbps": round(float(p.mean()), 2)}
    # per-regime (daily Sharpe within the regime's cycles, regime totPnL/maxDD)
    for r, g in c.groupby("regime"):
        gp = g["pnl_bps"]
        out[f"{r}_n"] = int(len(g)); out[f"{r}_pnl"] = round(float(gp.sum()), 1)
        out[f"{r}_Sh"] = round(dsh(gp), 2); out[f"{r}_mDD"] = round(mdd(gp), 1)
    # per-month daily Sharpe (fold proxy) -> list for nested/fold checks
    dd = p.resample("1D").sum()
    msh = dd.groupby(dd.index.to_period("M")).apply(lambda x: x.mean()/x.std()*ANN if x.std() > 0 else np.nan)
    out["months"] = {str(k): (round(float(v), 2) if v == v else None) for k, v in msh.items()}
    # Dec mid-bleed window (worst historical multi-week DD) + a recent-tail proxy
    out["DecBleed"] = round(float(c.loc["2025-12-12":"2026-01-13", "pnl_bps"].sum()), 1)
    out["May26"] = round(float(c.loc["2026-05-01":"2026-06-04", "pnl_bps"].sum()), 1)
    print(json.dumps(out))

if __name__ == "__main__":
    main()
