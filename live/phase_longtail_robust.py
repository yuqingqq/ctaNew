"""Robustness probe for squeeze survivors: is the Sharpe gain BROAD or one-episode (May)?
For each candidate state dir vs baseline:
  * per-MONTH folds-beat (pnl_bps sum) — majority => broad, not one lucky month
  * BULL split: profitable-bull vs squeeze-bull (by month) — the candidate must not kill the good bull alpha
  * monthly pnl delta table so we can see WHERE the gain comes from
Usage: python3 -m live.phase_longtail_robust <tag1> <tag2> ...
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
ROOT = Path("/home/yuqing/ctaNew/live/state/longtail")
ANN = np.sqrt(365)

def load(tag):
    f = (ROOT / tag / "state" / "cycles.csv") if tag != "baseline" else (ROOT / "baseline" / "state" / "cycles.csv")
    c = pd.read_csv(f); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    c = c.sort_values("open_time").set_index("open_time")
    c["month"] = c.index.to_period("M").astype(str)
    return c
def dsh(s):
    d = (s.fillna(0) / 1e4).resample("1D").sum(); return float(d.mean()/d.std()*ANN) if d.std()>0 else np.nan

base = load("baseline")
bm = base.groupby("month")["pnl_bps"].sum()
# bull months
bull_months = base[base["regime"] == "bull"].groupby("month")["pnl_bps"].sum()
print("BASELINE bull PnL by month:")
print("  " + "  ".join(f"{m}:{v:+.0f}" for m, v in bull_months.items()))
print()

for tag in sys.argv[1:]:
    c = load(tag)
    cm = c.groupby("month")["pnl_bps"].sum()
    common = bm.index.intersection(cm.index)
    beat = int((cm[common] > bm[common]).sum())
    d = (cm[common] - bm[common])
    # bull-only monthly delta
    cbull = c[c["regime"] == "bull"].groupby("month")["pnl_bps"].sum()
    bbull = base[base["regime"] == "bull"].groupby("month")["pnl_bps"].sum()
    bc = bbull.index.intersection(cbull.index)
    dbull = (cbull.reindex(bc).fillna(0) - bbull.reindex(bc).fillna(0))
    print(f"=== {tag} ===")
    print(f"  folds_beat (all-cycle monthly): {beat}/{len(common)}")
    worst = d.nsmallest(3); best = d.nlargest(3)
    print(f"  months HURT most: " + "  ".join(f"{m}:{v:+.0f}" for m, v in worst.items()))
    print(f"  months HELP most: " + "  ".join(f"{m}:{v:+.0f}" for m, v in best.items()))
    print(f"  BULL Δ by month:  " + "  ".join(f"{m}:{v:+.0f}" for m, v in dbull.items() if abs(v) > 30))
    # concentration: does one month own >60% of the total gain?
    tot_d = d.sum(); pos = d[d > 0]
    top1 = d.max()
    print(f"  total Δpnl {tot_d:+.0f}; single best month = {top1:+.0f} ({100*top1/tot_d:.0f}% of Δ)" if tot_d != 0 else "  no Δ")
    print()
