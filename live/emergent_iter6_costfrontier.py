"""iter6: cost/turnover FRONTIER for the real signals (tfi continuation, reversal).

iter4c/5c assumed FULL turnover every rebalance @24 bps -> deeply sub-cost. But order flow has long
memory, so bucket membership may persist (turnover <100%), improving the economics. Measure REALIZED
turnover (same-symbol bucket retention across rebalances), the BREAK-EVEN cost (gross/turnover), and net
Sharpe across a maker/rebate cost grid. h=5m/30m/1h, both eras.

Decisive number: break-even RT cost. If below achievable maker cost (~1-2 bps, or rebate) -> monetizable
by a maker/HFT vehicle; if below ~0.5 bp -> effectively not.
Run:  python3 -m live.emergent_iter6_costfrontier
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.flow_harness import load_panel
from live.emergent_harness import EXT
from live.emergent_iter4c_econ import HBARS, PYR, spread_series, CUT_NS
from live.emergent_iter5_richatoms import load_ext

COST_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0]   # bps RT per full turnover


def turnover(bt, sym_code, sig, idx):
    """Fraction of the long (and short) book that EXITS per rebalance (same-symbol retention)."""
    b = bt[idx]; s = sym_code[idx]
    codes, _ = pd.factorize(b, sort=True)
    r = pd.Series(sig[idx]).groupby(codes).rank(pct=True).to_numpy()
    longind = r <= 0.2; shortind = r >= 0.8
    order = np.lexsort((b, s))            # sort by symbol, then time
    so = s[order]; same = so[1:] == so[:-1]
    out = {}
    for name, ind in (("long", longind), ("short", shortind)):
        io = ind[order]
        prev = io[:-1] & same
        stay = prev & io[1:]
        ret = stay.sum() / max(prev.sum(), 1)
        out[name] = 1.0 - ret
    return (out["long"] + out["short"]) / 2.0


def report(bt, sc, sigs, fwds, tag):
    uniq = np.unique(bt)
    print(f"\n===== {tag} =====", flush=True)
    for h in ("5m", "30m", "1h"):
        hb = HBARS[h]; fw = fwds[h]; sig = sigs[h]
        idx = np.where(np.isin(bt, uniq[::hb]) & np.isfinite(sig) & np.isfinite(fw))[0]
        days, sp = spread_series(bt, sig, fw, idx)
        f = turnover(bt, sc, sig, idx)
        for era in ("OOS", "REC"):
            m = (days < CUT_NS) if era == "OOS" else (days >= CUT_NS)
            s = sp[m]
            if len(s) < 20:
                continue
            g = float(np.mean(s)); sd = float(np.std(s)); gbps = g * 1e4
            be = gbps / max(f, 1e-9)
            grid = "  ".join(f"{c}:{(g - f*c/1e4)/sd*np.sqrt(PYR[h]):+.1f}" for c in COST_GRID)
            print(f"  {h:<4}{era}: gross {gbps:+5.2f}bps | turnover {f:4.2f} | "
                  f"break-even {be:5.2f}bps | netSharpe@RTcost(bps) {grid}", flush=True)


def main():
    # reversal from slim (tr_h); tfi from ext
    S = load_panel(["symbol", "bar_time", "tr_5m", "tr_30m", "tr_1h", "fwd_5m", "fwd_30m", "fwd_1h"])
    bt = S["bar_time"].to_numpy("datetime64[ns]")
    sc = pd.factorize(S["symbol"])[0]
    trl = {"5m": S["tr_5m"].to_numpy(), "30m": S["tr_30m"].to_numpy(), "1h": S["tr_1h"].to_numpy()}
    fwd = {"5m": S["fwd_5m"].to_numpy(), "30m": S["fwd_30m"].to_numpy(), "1h": S["fwd_1h"].to_numpy()}
    del S
    print(f"panel {len(bt):,} rows | cost grid (bps RT): {COST_GRID}", flush=True)
    # reversal: long recent losers => sig = tr_h (spread_series longs bottom-20% = biggest losers)
    report(bt, sc, trl, fwd, "REVERSAL (long recent losers / short winners)")
    del trl, fwd

    E = load_ext(["symbol", "bar_time", "tfi", "fwd_5m", "fwd_30m", "fwd_1h"])
    bt2 = E["bar_time"].to_numpy("datetime64[ns]")
    sc2 = pd.factorize(E["symbol"])[0]
    tfi = (-E["tfi"]).to_numpy()   # long high tfi -> negate (spread_series longs bottom-20%)
    fwd2 = {"5m": E["fwd_5m"].to_numpy(), "30m": E["fwd_30m"].to_numpy(), "1h": E["fwd_1h"].to_numpy()}
    del E
    report(bt2, sc2, {"5m": tfi, "30m": tfi, "1h": tfi}, fwd2,
           "tfi CONTINUATION (long high tfi / short low)")


if __name__ == "__main__":
    main()
