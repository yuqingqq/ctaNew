"""iter4b: is the iter4 reversal-modulation OB-specific, or just VOL again?

iter4 found return_5min reversal is stronger in low-replenishment / high-flow-intensity books,
both-era. But the whole program says these features proxy vol, and reversal strength is a known
vol effect. Decisive control: DOUBLE-SORT by a vol/activity proxy (|tr_1h| per-bar rank tercile)
AND the OB conditioner. If the OB conditioner still modulates reversal WITHIN each vol tercile,
both-era, it is OB-specific; if the modulation vanishes within vol terciles, it was vol.

Run:  python3 -m live.emergent_iter4b_volctrl
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.flow_harness import CUT, ci, load_panel, xsic
from live.emergent_iter4_interact import per_bar_tercile


def main():
    cols = ["symbol", "bar_time", "return_5min", "ask_depth_residual_5min",
            "bid_depth_residual_5min", "buy_to_ask_5min", "sell_to_bid_5min",
            "tr_1h", "fwd_5m"]
    D = load_panel(cols)
    D["bar_time"] = pd.to_datetime(D["bar_time"], utc=True)
    D["repl"] = D["ask_depth_residual_5min"] + D["bid_depth_residual_5min"]
    D["aggr"] = D["buy_to_ask_5min"] + D["sell_to_bid_5min"]
    D["absvol"] = D["tr_1h"].abs()
    print(f"panel {len(D):,} rows", flush=True)
    m = {"OOS": (D["bar_time"] < CUT).to_numpy(), "REC": (D["bar_time"] >= CUT).to_numpy()}

    vt = per_bar_tercile(D, "absvol")
    for cond in ("repl", "aggr"):
        ct = per_bar_tercile(D, cond)
        print(f"\n=== reversal (return_5min→fwd_5m) IC by [vol tercile]×[{cond} tercile] ===",
              flush=True)
        print(f"  interaction of interest: (low-{cond} IC) − (high-{cond} IC) within each vol "
              "tercile, both-era", flush=True)
        for vlab in (0, 1, 2):
            row = {"OOS": {}, "REC": {}}
            for era in ("OOS", "REC"):
                for clab in (0, 2):
                    mask = (vt == vlab) & (ct == clab) & m[era]
                    a, lo, hi = ci(xsic(D, "return_5min", "fwd_5m", row_mask=mask))
                    row[era][clab] = a
                d = row[era][0] - row[era][2]   # low-cond minus high-cond reversal IC
                row[era]["diff"] = d
            print(f"  vol T{vlab}: OOS low {row['OOS'][0]:+.4f} / high {row['OOS'][2]:+.4f} "
                  f"→ diff {row['OOS']['diff']:+.4f} | "
                  f"REC low {row['REC'][0]:+.4f} / high {row['REC'][2]:+.4f} "
                  f"→ diff {row['REC']['diff']:+.4f} | "
                  f"both-era {'YES' if np.sign(row['OOS']['diff'])==np.sign(row['REC']['diff']) else 'no'}",
                  flush=True)


if __name__ == "__main__":
    main()
