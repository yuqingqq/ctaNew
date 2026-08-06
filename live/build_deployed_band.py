"""SOLID test: does a wider no-trade band improve NET on the DEPLOYED-style construction (top-K=3 L/S with
K+M hysteresis), i.e. on top of the hysteresis the strategy already uses?

Enter long = pred in top-K; hold long until it exits top-(K+M); symmetric for short (bottom-K / K+M).
M=0 = full rebalance (no band). Sweep M. Turnover / gross / NET at cost grid, both eras.
Conclusion: is the deployed band already ~optimal (idea already captured) or is wider better (deployable gain)?
Run: python3 -u -m live.build_deployed_band
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

COST_GRID = [24.0, 12.0, 6.0, 3.0]
PYR = 6 * 365.0
K = 3
M_SWEEP = [0, 2, 4, 8, 12]


def band_topk(rhi, rlo, K, M):
    n = len(rhi); pos = np.zeros(n, np.int8); p = 0
    for i in range(n):
        if rhi[i] <= K: p = 1
        elif rlo[i] <= K: p = -1
        elif p == 1 and rhi[i] <= K + M: p = 1
        elif p == -1 and rlo[i] <= K + M: p = -1
        else: p = 0
        pos[i] = p
    return pos


def turnover(bt, sym, pos):
    order = np.lexsort((bt, sym)); so, po = sym[order], pos[order]; same = so[1:] == so[:-1]
    out = []
    for side in (1, -1):
        cur = (po == side); prev = cur[:-1] & same
        out.append(1.0 - (prev & cur[1:]).sum() / max(prev.sum(), 1))
    return float(np.mean(out))


def pf(d, pos, min_leg=2):
    bt = d["open_time"].to_numpy("datetime64[ns]"); rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(bt, sort=True); k = len(uniq)
    nl = np.bincount(codes, (pos == 1).astype(float), k); ns = np.bincount(codes, (pos == -1).astype(float), k)
    sl = np.bincount(codes, np.where(pos == 1, rp, 0.0), k); ss = np.bincount(codes, np.where(pos == -1, rp, 0.0), k)
    ok = (nl >= min_leg) & (ns >= min_leg)
    return sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1)


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    print(f"DEPLOYED-style top-K={K} L/S + K+M hysteresis band (M=0 is full rebalance)\n", flush=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna().sort_values(["symbol", "open_time"])
        d["rhi"] = d.groupby("open_time")["pred"].rank(ascending=False, method="first")
        d["n"] = d.groupby("open_time")["pred"].transform("size")
        d["rlo"] = d["n"] + 1 - d["rhi"]
        print(f"===== {era} =====", flush=True)
        for M in M_SWEEP:
            pos = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M)
                                  for _, g in d.groupby("symbol", sort=False)])
            g = pf(d, pos); turn = turnover(d["open_time"].to_numpy("datetime64[ns]"),
                                            d["symbol"].to_numpy(), pos)
            gm, gsd = g.mean(), g.std()
            be = gm * 1e4 / max(turn, 1e-9)
            nets = "  ".join(f"{c:g}:{(gm - turn*c/1e4)/gsd*np.sqrt(PYR):+.2f}" for c in COST_GRID)
            tag = "(full rebal)" if M == 0 else f"(band K+{M})"
            print(f"  M={M:<2} {tag:<14} gross {gm*1e4:+.2f}bps | turn {turn:.2f} | "
                  f"break-even {be:5.1f}bps | netSharpe@cost {nets}", flush=True)
        print("", flush=True)
    print("DEPBANDDONE", flush=True)


if __name__ == "__main__":
    main()
