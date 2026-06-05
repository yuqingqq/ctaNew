"""Nested-OOS + per-fold validation of the trend-book allocation (the loop's best lead).
The momentum weight is a tuned param -> for each month, pick w* that maximized combined Sharpe on PRIOR
months, apply to the current month, accumulate. If nested Sharpe ~= static-best, the weight generalizes.
Pure pandas on existing equity series (mean-rev two-book + momentum single-book) — no new backtests.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew"); ANN = np.sqrt(6*365)
GRID = [0.0, 0.15, 0.25, 0.35, 0.50]
def sh(x): return x.mean()/x.std()*ANN if x.std()>0 else 0.0

mom = pd.read_csv(REPO/"live/state/single_momentum/cycles.csv")[["open_time","pnl_bps"]].rename(columns={"pnl_bps":"mom"})
mom["open_time"] = pd.to_datetime(mom["open_time"], utc=True)
for pol in ("monthly","daily"):
    mr = pd.read_csv(REPO/f"live/state/ab_wfund175/{pol}/combine/twobook_equity.csv")[["open_time","pnl_bps_combined_fill0"]].rename(columns={"pnl_bps_combined_fill0":"mr"})
    mr["open_time"] = pd.to_datetime(mr["open_time"], utc=True)
    m = mr.merge(mom, on="open_time", how="inner").fillna(0)
    m["mn"] = m["open_time"].dt.to_period("M")
    months = sorted(m["mn"].unique())
    # per-fold: does static w=0.25 beat baseline each month?
    base_sh = sh(m["mr"]); stat = (1-0.25)*m["mr"] + 0.25*m["mom"]
    wins = sum(1 for mo in months if sh((0.75*m[m.mn==mo]["mr"]+0.25*m[m.mn==mo]["mom"])) >= sh(m[m.mn==mo]["mr"]))
    # nested: pick w from prior months, apply to current
    nested = []
    for i, mo in enumerate(months):
        if i == 0:                       # no history -> default w=0.25
            wstar = 0.25
        else:
            past = m[m.mn.isin(months[:i])]
            wstar = max(GRID, key=lambda w: sh((1-w)*past["mr"] + w*past["mom"]))
        cur = m[m.mn==mo]
        nested.append((1-wstar)*cur["mr"] + wstar*cur["mom"])
    nested = pd.concat(nested)
    print(f"[{pol:7s}] baseline(w=0) {base_sh:+.3f} | static w=0.25 {sh(stat):+.3f} | NESTED-OOS {sh(nested):+.3f} "
          f"| fold-wins(w=.25) {wins}/{len(months)}")
    chosen = [ (months[i], max(GRID, key=lambda w: sh((1-w)*m[m.mn.isin(months[:i])]["mr"]+w*m[m.mn.isin(months[:i])]["mom"])) if i>0 else 0.25) for i in range(len(months))]
    print(f"          nested w path: {[(str(mo),w) for mo,w in chosen]}")
