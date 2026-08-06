"""iter5c: economic test of the one surviving orthogonal signal — tfi (flow continuation).

tfi survives price+book/flow+vol controls (iter5b), both-era, +0.029@5m→+0.012@30m. Is it usable net of
cost, or the same sub-cost HFT lead? Continuation L/S: long high-tfi / short low-tfi quintiles, rebalanced
every h, non-overlapping, NET of 24 bps hedged RT/rebalance. h=5m/30m/1h, both eras.

Run:  python3 -m live.emergent_iter5c_tfi_econ
"""
from __future__ import annotations

import numpy as np

from live.flow_harness import CUT
from live.emergent_harness import EXT
from live.emergent_iter4c_econ import HBARS, PYR, COST, spread_series, stats, CUT_NS
from live.emergent_iter5_richatoms import load_ext

TGT = {"5m": "fwd_5m", "30m": "fwd_30m", "1h": "fwd_1h"}


def main():
    D = load_ext(["symbol", "bar_time", "tfi", "fwd_5m", "fwd_30m", "fwd_1h"])
    bt = D["bar_time"].to_numpy("datetime64[ns]")
    tfi = D["tfi"].to_numpy()
    fwds = {h: D[TGT[h]].to_numpy() for h in HBARS}
    del D
    print(f"panel {len(bt):,} rows | cost/period {COST*1e4:.0f} bps | signal=tfi (long high / short low)\n",
          flush=True)
    uniq_all = np.unique(bt)
    for h in ("5m", "30m", "1h"):
        hb = HBARS[h]; fwd = fwds[h]
        rb = uniq_all[::hb]
        base = np.where(np.isin(bt, rb) & np.isfinite(tfi) & np.isfinite(fwd))[0]
        # long high tfi / short low tfi  => pass sig = -tfi to spread_series (which longs bottom-20%)
        days, sp = spread_series(bt, (-tfi), fwd, base)
        print(f"===== horizon {h} =====", flush=True)
        for era in ("OOS", "REC"):
            st = stats(days, sp, h, era)
            if st:
                g, n, shr, gshr, npr = st
                print(f"  tfi L/S {era}: gross {g:+6.2f}bps | net {n:+7.2f}bps | "
                      f"netSharpe {shr:+5.2f} | grossSharpe {gshr:+5.2f} | nper {npr}", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
