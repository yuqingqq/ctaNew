"""NET edge: take the live per-symbol Ridge L/S book, measure REALIZED turnover, and net out cost at a
range of fee levels — both eras. Turns the gross ~3.3 bps/bar into an actual net figure.

Portfolio: quintile L/S on raw forward returns (return_pct), rebalanced every 4h (native cadence).
Turnover = fraction of each leg that exits per rebalance (same-symbol retention). Net = gross − turnover×cost.
Run: python3 -u -m live.build_net_edge
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

COST_GRID = [24.0, 12.0, 6.0, 3.0, 1.5]   # bps round-trip (retail VIP0 hedged ... deep maker)
PYR = 6 * 365.0                            # 4h bars per year


def leg_turnover(bt, sym, pos):
    """avg fraction of the (long and short) book that exits per rebalance, same-symbol retention."""
    order = np.lexsort((bt, sym))
    so, po = sym[order], pos[order]
    same = so[1:] == so[:-1]
    out = []
    for side in (1, -1):
        cur = (po == side)
        prev = cur[:-1] & same
        stay = prev & cur[1:]
        out.append(1.0 - stay.sum() / max(prev.sum(), 1))
    return float(np.mean(out))


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    print(f"NET edge (quintile L/S, 4h rebalance, {PYR:.0f} bars/yr)\n", flush=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna()
        bt = d["open_time"].to_numpy("datetime64[ns]")
        codes, uniq = pd.factorize(bt, sort=True)
        r = pd.Series(d["pred"].to_numpy()).groupby(codes).rank(pct=True).to_numpy()
        rp = d["return_pct"].to_numpy()
        pos = np.where(r >= 0.8, 1, np.where(r <= 0.2, -1, 0))
        # per-bar gross L/S (long top / short bottom)
        k = len(uniq)
        wl = np.where(pos == 1, rp, 0.0); nl = np.bincount(codes, (pos == 1).astype(float), k)
        ws = np.where(pos == -1, rp, 0.0); ns = np.bincount(codes, (pos == -1).astype(float), k)
        sl = np.bincount(codes, wl, k); ss = np.bincount(codes, ws, k)
        ok = (nl >= 3) & (ns >= 3)
        gross = (sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1))  # per-bar L/S return
        turn = leg_turnover(bt, d["symbol"].to_numpy(), pos)
        gm, gsd = gross.mean(), gross.std()
        print(f"===== {era} =====", flush=True)
        print(f"    gross L/S {gm*1e4:+.2f} bps/bar | turnover {turn:.2f}/rebalance | "
              f"gross Sharpe {gm/gsd*np.sqrt(PYR):+.2f}", flush=True)
        for c in COST_GRID:
            net = gm - turn * c / 1e4
            nsh = net / gsd * np.sqrt(PYR)
            print(f"      cost {c:>5.1f} bps RT → net {net*1e4:+6.2f} bps/bar | net Sharpe {nsh:+6.2f}",
                  flush=True)
        print("", flush=True)
    print("NETDONE", flush=True)


if __name__ == "__main__":
    main()
