"""OOS validation of the longtail conclusions on the fullhist (2022-2026) pred set.

The loop derived D0=0.15 and the bull-irreducibility conclusion on 2025-10 -> 2026-05 only.
fullhist_mpit preds cover 2022-01 -> 2026-06. Everything BEFORE 2025-10-04 is genuine
hold-out (the loop never saw it). This tests whether:
  (1) the bear-grind depth-ramp fix (D0 0.10 -> 0.15) REPLICATES on OOS grind episodes;
  (2) bull still owns the maxDD / squeeze stays a loss center on OOS bull episodes.
Different model artifact => absolute Sharpe won't match hl_lean175; this is a MECHANISM test.
"""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/home/yuqing/ctaNew/live/state/longtail")
ANN = np.sqrt(365)
SPLIT = pd.Timestamp("2025-10-04", tz="UTC")   # loop's in-sample start
GRIND_THR = -0.25

def load(tag):
    c = pd.read_csv(ROOT / tag / "state" / "cycles.csv")
    c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    return c.sort_values("open_time").set_index("open_time")
def dsh(s):
    d = (s.fillna(0)/1e4).resample("1D").sum(); return float(d.mean()/d.std()*ANN) if d.std()>0 else np.nan
def maxdd(s):
    eq = s.fillna(0).cumsum(); return float((eq-eq.cummax()).min())

def grind(c): return c[(c["regime"]=="bear") & (c["btc_ret_30d"]>=GRIND_THR)]
def deep(c):  return c[(c["regime"]=="bear") & (c["btc_ret_30d"]<GRIND_THR)]
def bull(c):  return c[c["regime"]=="bull"]

b10 = load("oos_d0_0.10"); b15 = load("oos_d0_0.15")
print(f"fullhist window: {b10.index.min().date()} -> {b10.index.max().date()}  ({b10.index.to_period('M').nunique()} months, {len(b10)} cycles)")
print(f"OOS hold-out = before {SPLIT.date()}\n")

for name, lo, hi in [("OOS  (2022-01..2025-09, HOLD-OUT)", None, SPLIT),
                     ("IS   (2025-10..2026-05, loop win)", SPLIT, None)]:
    def sl(c):
        m = pd.Series(True, index=c.index)
        if lo is not None: m &= c.index >= lo
        if hi is not None: m &= c.index < hi
        return c[m]
    c10, c15 = sl(b10), sl(b15)
    if len(c10) == 0: continue
    print(f"===== {name}  n={len(c10)} =====")
    # overall
    print(f"  ALL    D0.10 sh {dsh(c10['pnl_bps']):+.3f} tot {c10['pnl_bps'].sum():+8.0f} dd {maxdd(c10['pnl_bps']):+8.0f}  ||"
          f"  D0.15 sh {dsh(c15['pnl_bps']):+.3f} tot {c15['pnl_bps'].sum():+8.0f} dd {maxdd(c15['pnl_bps']):+8.0f}")
    # grind — the adopted fix's target
    g10, g15 = grind(c10)["pnl_bps"], grind(c15)["pnl_bps"]
    print(f"  GRIND  D0.10 sh {dsh(g10):+.3f} tot {g10.sum():+8.0f} (n={len(g10)})  ->"
          f"  D0.15 sh {dsh(g15):+.3f} tot {g15.sum():+8.0f}   Δtot {g15.sum()-g10.sum():+.0f}")
    # per-grind-episode (month) improvement
    gm10 = grind(c10).groupby(grind(c10).index.to_period("M"))["pnl_bps"].sum()
    gm15 = grind(c15).groupby(grind(c15).index.to_period("M"))["pnl_bps"].sum()
    common = gm10.index.intersection(gm15.index)
    d = (gm15[common]-gm10[common])
    imp = int((d > 5).sum()); wor = int((d < -5).sum())
    print(f"    grind episodes: {imp} improved / {wor} worsened / {len(common)} total")
    # deep — must not be hurt
    dd10, dd15 = deep(c10)["pnl_bps"], deep(c15)["pnl_bps"]
    print(f"  DEEP   D0.10 sh {dsh(dd10):+.3f} tot {dd10.sum():+8.0f}  ->  D0.15 sh {dsh(dd15):+.3f} tot {dd15.sum():+8.0f}   Δtot {dd15.sum()-dd10.sum():+.0f}")
    # bull — does it own the DD?
    bl10 = bull(c10)["pnl_bps"]
    bull_dd = maxdd(bl10); all_dd = maxdd(c10["pnl_bps"])
    print(f"  BULL   tot {bl10.sum():+.0f} sh {dsh(bl10):+.3f} maxDD {bull_dd:+.0f}   (ALL maxDD {all_dd:+.0f}; bull owns {'YES' if abs(bull_dd) >= 0.9*abs(all_dd) else 'no'})")
    print()
