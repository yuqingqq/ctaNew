"""iter4 (H-INTERACT): does a signal's forward XS-IC come ALIVE inside a bucket of a conditioner?

The last untested emergent-RETURN hypothesis: a marginal IC ~0 can hide a conditional edge if the
signal only works in a sub-regime defined by ANOTHER feature (the "conditional subspace, done right").
Mechanism-motivated pairs only (absorption/replenishment, flow-intensity), not an exhaustive scan.

Buckets = the conditioner's PER-BAR cross-sectional rank terciles (era-locked & breadth-robust by
construction). Within each tercile, cross-sectional rank-IC of the signal vs forward return, both eras,
day-clustered CI. Interaction = top-tercile IC vs bottom-tercile IC, both-era consistent? Bonferroni
across the primary contrasts.

Run:  python3 -m live.emergent_iter4_interact
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.flow_harness import CUT, ci, load_panel, xsic

# (signal, conditioner, horizon)  — conditioner is a column added below
PAIRS = [
    ("imb1", "repl", "fwd_5m"),                 # absorption -> imbalance continuation?
    ("imb1", "repl", "fwd_30m"),
    ("signed_pressure_5min", "repl", "fwd_5m"), # absorbed aggressive flow -> continuation?
    ("return_5min", "repl", "fwd_5m"),          # reversal conditional on replenishment/liquidity
    ("imb_change_5min", "aggr", "fwd_5m"),       # continuation conditional on flow intensity
    ("return_5min", "aggr", "fwd_5m"),          # reversal conditional on flow intensity
]
NPRIM = len(PAIRS)  # Bonferroni denominator for the top-vs-bottom both-era claim


def per_bar_tercile(D: pd.DataFrame, col: str) -> np.ndarray:
    codes, _ = pd.factorize(D["bar_time"].to_numpy("datetime64[ns]"), sort=True)
    r = pd.Series(D[col].to_numpy()).groupby(codes).rank(pct=True).to_numpy()
    t = np.full(len(r), -1, dtype=np.int8)
    t[r < 1 / 3] = 0
    t[(r >= 1 / 3) & (r < 2 / 3)] = 1
    t[r >= 2 / 3] = 2
    return t


def main():
    cols = ["symbol", "bar_time", "imb1", "signed_pressure_5min", "return_5min",
            "imb_change_5min", "ask_depth_residual_5min", "bid_depth_residual_5min",
            "buy_to_ask_5min", "sell_to_bid_5min", "fwd_5m", "fwd_30m"]
    D = load_panel(cols)
    D["bar_time"] = pd.to_datetime(D["bar_time"], utc=True)
    D["repl"] = D["ask_depth_residual_5min"] + D["bid_depth_residual_5min"]   # replenishment (absorption)
    D["aggr"] = D["buy_to_ask_5min"] + D["sell_to_bid_5min"]                  # aggressive-flow intensity
    print(f"panel {len(D):,} rows | {D.symbol.nunique()} syms", flush=True)
    m_oos = (D["bar_time"] < CUT).to_numpy()
    m_rec = (D["bar_time"] >= CUT).to_numpy()

    tercache = {}
    print(f"\nPrimary contrasts: {NPRIM} | Bonferroni α=0.05 → per-test {0.05/NPRIM:.4f} "
          "(use ~99.2% CIs; here reporting 95% CIs, flag survivors conservatively)\n", flush=True)
    print(f"{'signal':<22}{'cond':<6}{'h':<7}{'era':<5}"
          f"{'IC bottom-tercile':<26}{'IC top-tercile':<26}{'top−bottom'}", flush=True)
    for sig, cond, h in PAIRS:
        if cond not in tercache:
            tercache[cond] = per_bar_tercile(D, cond)
        ter = tercache[cond]
        for era, emask in (("OOS", m_oos), ("REC", m_rec)):
            botmask = (ter == 0) & emask
            topmask = (ter == 2) & emask
            ic_b = xsic(D, sig, h, row_mask=botmask)
            ic_t = xsic(D, sig, h, row_mask=topmask)
            ab, lb, ub = ci(ic_b)
            at, lt, ut = ci(ic_t)
            diff = at - ab
            sb = "*" if (lb > 0 or ub < 0) else " "
            st = "*" if (lt > 0 or ut < 0) else " "
            print(f"{sig:<22}{cond:<6}{h:<7}{era:<5}"
                  f"{f'{ab:+.4f}[{lb:+.4f},{ub:+.4f}]{sb}':<26}"
                  f"{f'{at:+.4f}[{lt:+.4f},{ut:+.4f}]{st}':<26}"
                  f"{diff:+.4f}", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
