"""Robustness check on the top-K=3 gross (esp. RECENT +27 bps): is it a few fat-tail bars in 3-name legs,
or a real broad edge? Report mean vs median vs winsorized-mean, n_bars, and % of total from the 5 biggest bars.
Run: python3 -u -m live.build_topk_robust
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import band_topk, pf

K = 3


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna().sort_values(["symbol", "open_time"])
        d["rhi"] = d.groupby("open_time")["pred"].rank(ascending=False, method="first")
        d["n"] = d.groupby("open_time")["pred"].transform("size")
        d["rlo"] = d["n"] + 1 - d["rhi"]
        pos = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, 0)
                              for _, g in d.groupby("symbol", sort=False)])
        s = pf(d, pos) * 1e4  # per-bar spread in bps
        s = s[np.isfinite(s)]
        top5 = np.sort(np.abs(s))[-5:].sum()
        wlo, whi = np.percentile(s, [1, 99])
        sw = np.clip(s, wlo, whi)
        print(f"===== {era} (top-{K}, M=0) =====", flush=True)
        print(f"  n_bars {len(s)} | mean {s.mean():+.2f} | median {np.median(s):+.2f} | "
              f"winsorized(1%) mean {sw.mean():+.2f} bps", flush=True)
        print(f"  |5 biggest bars| = {top5:.0f} bps = {top5/np.abs(s).sum()*100:.0f}% of total abs P&L "
              f"(high % = fat-tail-driven / not robust)", flush=True)
        print(f"  std {s.std():.1f} | mean/median ratio {s.mean()/ (np.median(s) if abs(np.median(s))>1e-9 else np.nan):.1f}\n",
              flush=True)
    print("ROBUSTDONE", flush=True)


if __name__ == "__main__":
    main()
