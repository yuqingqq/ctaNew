"""iter7: does SIGNAL CONCENTRATION lift the break-even cost above the maker frontier?

iter6: reversal/tfi break-even RT cost ~1-4 bps at quintile L/S. Tighter buckets = stronger per-name
signal (higher gross) but fewer names (thinner) and likely higher turnover (extreme membership less
persistent). Break-even = gross/turnover is the clean monetizability metric. Sweep q = 20/10/5/2.5%
per leg, for reversal and tfi, h=5m/30m, both eras.

If break-even does NOT rise materially above ~2 bps, ~1-4 bps is the ceiling => converged.
Run:  python3 -m live.emergent_iter7_concentration
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.flow_harness import load_panel
from live.emergent_iter4c_econ import HBARS, PYR, CUT_NS
from live.emergent_iter5_richatoms import load_ext

COST_GRID = [0.5, 1.0, 2.0, 4.0]
QS = [0.20, 0.10, 0.05, 0.025]
MIN_LEG = 3


def spread_q(bt, sig, fwd, idx, q):
    bts = bt[idx]
    codes, uniq = pd.factorize(bts, sort=True)
    r = pd.Series(sig[idx]).groupby(codes).rank(pct=True).to_numpy()
    f = fwd[idx]; k = len(uniq)
    long = (r <= q) & np.isfinite(f)
    short = (r >= 1 - q) & np.isfinite(f)
    sl = np.bincount(codes, weights=np.where(long, f, 0.0), minlength=k)
    nl = np.bincount(codes, weights=long.astype(float), minlength=k)
    ss = np.bincount(codes, weights=np.where(short, f, 0.0), minlength=k)
    ns = np.bincount(codes, weights=short.astype(float), minlength=k)
    ok = (nl >= MIN_LEG) & (ns >= MIN_LEG)
    spread = np.where(ok, sl / np.maximum(nl, 1) - ss / np.maximum(ns, 1), np.nan)
    leg = (nl[ok].mean() + ns[ok].mean()) / 2 if ok.any() else 0.0
    return uniq[ok], spread[ok], leg


def turnover_q(bt, sc, sig, idx, q):
    b = bt[idx]; s = sc[idx]
    codes, _ = pd.factorize(b, sort=True)
    r = pd.Series(sig[idx]).groupby(codes).rank(pct=True).to_numpy()
    order = np.lexsort((b, s)); so = s[order]; same = so[1:] == so[:-1]
    ts = []
    for ind in ((r <= q), (r >= 1 - q)):
        io = ind[order]; prev = io[:-1] & same
        ts.append(1.0 - (prev & io[1:]).sum() / max(prev.sum(), 1))
    return float(np.mean(ts))


def report(bt, sc, sig_of_h, fwd_of_h, tag):
    uniq = np.unique(bt)
    print(f"\n===== {tag} =====", flush=True)
    for h in ("5m", "30m"):
        hb = HBARS[h]; sig = sig_of_h[h]; fwd = fwd_of_h[h]
        idx0 = np.where(np.isin(bt, uniq[::hb]) & np.isfinite(sig) & np.isfinite(fwd))[0]
        for q in QS:
            days, sp, leg = spread_q(bt, sig, fwd, idx0, q)
            f = turnover_q(bt, sc, sig, idx0, q)
            for era in ("OOS", "REC"):
                m = (days < CUT_NS) if era == "OOS" else (days >= CUT_NS)
                s = sp[m]
                if len(s) < 20:
                    continue
                g = float(np.mean(s)); sd = float(np.std(s)); gbps = g * 1e4
                be = gbps / max(f, 1e-9)
                grid = " ".join(f"{c}:{(g - f*c/1e4)/sd*np.sqrt(PYR[h]):+.0f}" for c in COST_GRID)
                print(f"  {h:<4} q={q:<5} {era}: gross {gbps:+5.2f}bps | turn {f:4.2f} | "
                      f"break-even {be:5.2f}bps | leg~{leg:4.1f} | netSh@cost {grid}", flush=True)
        print("", flush=True)


def main():
    S = load_panel(["symbol", "bar_time", "tr_5m", "tr_30m", "fwd_5m", "fwd_30m"])
    bt = S["bar_time"].to_numpy("datetime64[ns]"); sc = pd.factorize(S["symbol"])[0]
    rev_sig = {"5m": S["tr_5m"].to_numpy(), "30m": S["tr_30m"].to_numpy()}
    rev_fwd = {"5m": S["fwd_5m"].to_numpy(), "30m": S["fwd_30m"].to_numpy()}
    del S
    print(f"panel {len(bt):,} rows | q sweep {QS} | cost grid {COST_GRID} bps RT", flush=True)
    report(bt, sc, rev_sig, rev_fwd, "REVERSAL (long recent losers)")
    del rev_sig, rev_fwd

    E = load_ext(["symbol", "bar_time", "tfi", "fwd_5m", "fwd_30m"])
    bt2 = E["bar_time"].to_numpy("datetime64[ns]"); sc2 = pd.factorize(E["symbol"])[0]
    tfi = (-E["tfi"]).to_numpy()
    tfi_fwd = {"5m": E["fwd_5m"].to_numpy(), "30m": E["fwd_30m"].to_numpy()}
    del E
    report(bt2, sc2, {"5m": tfi, "30m": tfi}, tfi_fwd, "tfi CONTINUATION (long high tfi)")


if __name__ == "__main__":
    main()
