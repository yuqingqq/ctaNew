"""iter4c: ECONOMIC decider — does OB-state conditioning make the reversal signal usable net of cost?

Reversal quintile L/S: long the bottom-20% by trailing return over h (recent losers), short the top-20%
(recent winners), rebalanced every h (non-overlapping). Realized period return = mean(fwd_h|long) −
mean(fwd_h|short). Compare:
  (A) unconditioned (all names)
  (B) OB-concentrated: restrict to the low-aggr half per bar (strong-reversal regime)
  (C) OB-concentrated: restrict to the low-repl half per bar
at h=5m/30m/1h, both eras, NET of a conservative full-turnover cost (24 bps hedged RT per rebalance).

The decisive number: gross spread per period vs 24 bps. If gross < cost, net<0 => sub-cost (program pattern).
Memory-frugal: numpy arrays + index masks, no DataFrame copies (28M rows under the per-proc cap).
Run:  python3 -m live.emergent_iter4c_econ
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.flow_harness import CUT, load_panel

HBARS = {"5m": 1, "30m": 6, "1h": 12}
TGT = {"5m": "fwd_5m", "30m": "fwd_30m", "1h": "fwd_1h"}
TRL = {"5m": "tr_5m", "30m": "tr_30m", "1h": "tr_1h"}
PYR = {"5m": 105120.0, "30m": 17520.0, "1h": 8760.0}
COST = 0.0024
MIN_LEG = 3
CUT_NS = np.datetime64(CUT.tz_convert(None))


def bar_rank(bt_sub, x_sub):
    codes, uniq = pd.factorize(bt_sub, sort=True)
    r = pd.Series(x_sub).groupby(codes).rank(pct=True).to_numpy()
    return codes, uniq, r


def spread_series(bt, sig, fwd, idx):
    """Per-bar reversal quintile L/S spread over the rows selected by `idx`."""
    bts = bt[idx]
    codes, uniq, r = bar_rank(bts, sig[idx])
    f = fwd[idx]; k = len(uniq)
    long = (r <= 0.2) & np.isfinite(f)
    short = (r >= 0.8) & np.isfinite(f)
    sl = np.bincount(codes, weights=np.where(long, f, 0.0), minlength=k)
    nl = np.bincount(codes, weights=long.astype(float), minlength=k)
    ss = np.bincount(codes, weights=np.where(short, f, 0.0), minlength=k)
    ns = np.bincount(codes, weights=short.astype(float), minlength=k)
    ok = (nl >= MIN_LEG) & (ns >= MIN_LEG)
    spread = np.where(ok, sl / np.maximum(nl, 1) - ss / np.maximum(ns, 1), np.nan)
    return uniq[ok], spread[ok]


def stats(days, sp, h, era):
    m = (days < CUT_NS) if era == "OOS" else (days >= CUT_NS)
    s = sp[m]
    if len(s) < 20:
        return None
    gross = float(np.mean(s)); sd = float(np.std(s)); net = gross - COST
    shr = net / sd * np.sqrt(PYR[h]) if sd > 0 else np.nan
    gshr = gross / sd * np.sqrt(PYR[h]) if sd > 0 else np.nan
    return gross * 1e4, net * 1e4, shr, gshr, len(s)


def main():
    cols = (["bar_time", "ask_depth_residual_5min", "bid_depth_residual_5min",
             "buy_to_ask_5min", "sell_to_bid_5min"]
            + sorted(set(TRL.values())) + sorted(set(TGT.values())))
    D = load_panel(cols)
    bt = pd.to_datetime(D["bar_time"], utc=True).to_numpy("datetime64[ns]")
    repl = (D["ask_depth_residual_5min"] + D["bid_depth_residual_5min"]).to_numpy()
    aggr = (D["buy_to_ask_5min"] + D["sell_to_bid_5min"]).to_numpy()
    arr = {c: D[c].to_numpy() for c in set(TRL.values()) | set(TGT.values())}
    del D
    print(f"panel {len(bt):,} rows | cost/period {COST*1e4:.0f} bps (full-turnover)\n", flush=True)
    uniq_all = np.unique(bt)

    for h in ("5m", "30m", "1h"):
        hb = HBARS[h]; sig = arr[TRL[h]]; fwd = arr[TGT[h]]
        rb_times = uniq_all[::hb]
        rb_set = np.isin(bt, rb_times)
        base = np.where(rb_set & np.isfinite(sig) & np.isfinite(fwd))[0]
        _, _, aggr_r = bar_rank(bt[base], aggr[base])
        _, _, repl_r = bar_rank(bt[base], repl[base])
        variants = {
            "(A) all names     ": base,
            "(B) low-aggr half ": base[aggr_r <= 0.5],
            "(C) low-repl half ": base[repl_r <= 0.5],
        }
        print(f"===== horizon {h} =====", flush=True)
        for name, idx in variants.items():
            days, sp = spread_series(bt, sig, fwd, idx)
            for era in ("OOS", "REC"):
                st = stats(days, sp, h, era)
                if st:
                    g, n, shr, gshr, npr = st
                    print(f"  {name} {era}: gross {g:+6.2f}bps | net {n:+7.2f}bps | "
                          f"netSharpe {shr:+5.2f} | grossSharpe {gshr:+5.2f} | nper {npr}", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
