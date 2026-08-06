"""Do the two validated improvements STACK? Test combined on deployed top-K=3:
  BASELINE  = full rebalance (M=0), unhedged.
  COMBINED  = no-trade band (K+M) + beta-neutralization (era-locked hedge).
Report gross, turnover, NET at cost grid, and era-stability — both eras. Honest: is it "much better" or
just more robust/higher-break-even but still cost-gated?
Run: python3 -u -m live.build_combined
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import band_topk, turnover

COST_GRID = [24.0, 12.0, 6.0, 3.0]
PYR = 6 * 365.0
K, M = 3, 8


def ls_mkt(d, pos, min_leg=2):
    bt = d["open_time"].to_numpy("datetime64[ns]"); rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(bt, sort=True); k = len(uniq)
    nl = np.bincount(codes, (pos == 1).astype(float), k); ns = np.bincount(codes, (pos == -1).astype(float), k)
    sl = np.bincount(codes, np.where(pos == 1, rp, 0.0), k); ss = np.bincount(codes, np.where(pos == -1, rp, 0.0), k)
    na = np.bincount(codes, minlength=k); sa = np.bincount(codes, rp, k)
    ok = (nl >= min_leg) & (ns >= min_leg)
    return sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1), sa[ok] / np.maximum(na[ok], 1)


def report(name, ls, turn, cost_hedge=False):
    gm, gsd = ls.mean(), ls.std()
    be = gm * 1e4 / max(turn, 1e-9)
    grid = "  ".join(f"{c:g}:{(gm - turn*c/1e4)/gsd*np.sqrt(PYR):+.2f}" for c in COST_GRID)
    print(f"    {name:<26} gross {gm*1e4:+.2f}bps | turn {turn:.2f} | break-even {be:5.1f}bps | "
          f"netSharpe@cost {grid}", flush=True)


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    store = {}
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna().sort_values(["symbol", "open_time"])
        d["rhi"] = d.groupby("open_time")["pred"].rank(ascending=False, method="first")
        d["n"] = d.groupby("open_time")["pred"].transform("size"); d["rlo"] = d["n"] + 1 - d["rhi"]
        sym = d["symbol"].to_numpy(); bt = d["open_time"].to_numpy("datetime64[ns]")
        pos0 = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, 0) for _, g in d.groupby("symbol", sort=False)])
        posB = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M) for _, g in d.groupby("symbol", sort=False)])
        ls0, mkt0 = ls_mkt(d, pos0); lsB, mktB = ls_mkt(d, posB)
        store[era] = dict(ls0=ls0, t0=turnover(bt, sym, pos0), lsB=lsB, mktB=mktB, tB=turnover(bt, sym, posB))
    betaB = {e: np.polyfit(store[e]["mktB"], store[e]["lsB"], 1)[0] for e in store}
    other = {"RECENT": "OOS", "OOS": "RECENT"}
    print(f"deployed top-K={K}; band M={M}; era-locked beta-hedge\n", flush=True)
    net_hedged = {}
    for era in ("RECENT", "OOS"):
        s = store[era]
        lsH = s["lsB"] - betaB[other[era]] * s["mktB"]      # banded + beta-hedged
        net_hedged[era] = lsH
        print(f"===== {era} =====", flush=True)
        report("baseline (M=0, unhedged)", s["ls0"], s["t0"])
        report("band only (M=8)", s["lsB"], s["tB"])
        report("band + beta-hedge (COMBINED)", lsH, s["tB"])
        print("", flush=True)
    g0 = abs(store["RECENT"]["ls0"].mean() - store["OOS"]["ls0"].mean()) * 1e4
    gC = abs(net_hedged["RECENT"].mean() - net_hedged["OOS"].mean()) * 1e4
    print(f"era gap in gross mean: baseline {g0:.2f}bps  vs  COMBINED {gC:.2f}bps (smaller = more stable)",
          flush=True)
    print("\nCOMBODONE", flush=True)


if __name__ == "__main__":
    main()
