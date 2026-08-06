"""iter8: full MACRO-SIGNAL inventory — what does each micro atom predict, and is it one of a few
distinct signals? Cross-sectional rank-IC vs fwd (5m/30m/1h/4h), both eras, grouped by macro signal.

Purpose: enumerate the general signals the micro features generate (beyond tfi/reversal) and give the
raw predictive map to classify each (alpha-residual vs microstructure) alongside the control tests
already run (vol control iter2/5b, partial-IC iter5, cost iter4c/6/7).

Run:  python3 -m live.emergent_iter8_inventory
"""
from __future__ import annotations

import numpy as np

from live.flow_harness import CUT, xsic
from live.emergent_iter5_richatoms import load_ext

GROUPS = {
    "PRICE reversal":      ["tr_5m"],
    "FLOW / order-flow":   ["tfi", "signed_volume_z", "signed_pressure_5min", "buy_to_ask_5min"],
    "BOOK imbalance":      ["imb1", "imb_change_5min", "imb02"],
    "IMPACT / illiquidity": ["kyle_lambda", "impact_bps_per_pressure_5min", "vpin"],
    "DEPTH / absorption":  ["ask_depth_residual_5min", "bid_depth_residual_5min", "avg_trade_size"],
}
HZ = ["fwd_5m", "fwd_1h"]   # 5m = where signals live; 1h = persistence check (2 horizons for speed)


def main():
    feats = [f for g in GROUPS.values() for f in g]
    D = load_ext(["symbol", "bar_time", *feats, *HZ])
    m = {"OOS": (D["bar_time"] < CUT).to_numpy(), "REC": (D["bar_time"] >= CUT).to_numpy()}
    print(f"panel {len(D):,} rows | rank-IC (mean over ~350k bars; CIs ~±0.001)\n", flush=True)
    print("* = both-era same sign AND min|IC|>0.003 (given tiny CIs, ~significant)\n", flush=True)
    hdr = "".join(f"{h.replace('fwd_',''):<15}" for h in HZ)
    print(f"{'feature':<28}{hdr}", flush=True)
    for gname, gfeats in GROUPS.items():
        print(f"— {gname} —", flush=True)
        for f in gfeats:
            cells = []
            for h in HZ:
                ao = float(xsic(D, f, h, row_mask=m["OOS"]).mean())
                ar = float(xsic(D, f, h, row_mask=m["REC"]).mean())
                both = np.sign(ao) == np.sign(ar) and min(abs(ao), abs(ar)) > 0.003
                cells.append(f"{ao:+.3f}/{ar:+.3f}{'*' if both else ' '}")
            print(f"  {f:<26}" + "".join(f"{c:<15}" for c in cells), flush=True)
    print("\n(cell = OOS/REC mean rank-IC. imb02 recent-only ~53% cov. Sign: + continuation, − reversal.)",
          flush=True)


if __name__ == "__main__":
    main()
