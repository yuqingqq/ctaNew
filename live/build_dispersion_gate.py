"""Research cycle: is our crypto reversal+low-vol edge DISPERSION-conditional? (factor-timing literature;
crypto momentum breaks down in high dispersion — SSRN 6648082 — so reversal may strengthen there.)

Per bar: model rank-IC (pred vs fwd alpha) + cross-sectional dispersion (std of trailing return_1d, PIT).
Bucket by era-locked dispersion terciles; is the edge stronger in low or high dispersion, BOTH eras?
If a stable pattern -> a PIT dispersion gate (trade the high-edge regime) improves edge/bar & cuts cost.
Run: python3 -u -m live.build_dispersion_gate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred

CUT = pd.Timestamp("2025-10-01", tz="UTC")


def main():
    PAN = build_panel()
    key = PAN[["symbol", "open_time", "alpha_vs_btc_realized", "return_1d"]].copy()
    frames = []
    for era, cuts in (("REC", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(key, on=["symbol", "open_time"], how="inner").dropna()
        per = d.groupby("open_time").apply(lambda g: pd.Series({
            "ic": spearmanr(g["pred"], g["alpha_vs_btc_realized"]).correlation if len(g) >= 8 else np.nan,
            "disp": g["return_1d"].std()}), include_groups=False).dropna()
        per["era"] = era
        frames.append(per)
    A = pd.concat(frames)
    # era-locked dispersion terciles from OOS
    q = np.nanquantile(A.loc[A.era == "OOS", "disp"], [1/3, 2/3])
    A["db"] = np.digitize(A["disp"].to_numpy(), q)
    print("model rank-IC by cross-sectional DISPERSION tercile (era-locked), both eras:", flush=True)
    print(f"  {'regime':<12}{'OOS mean IC':<14}{'REC mean IC':<14}{'OOS nbars':<10}", flush=True)
    for b, name in [(0, "disp LOW"), (1, "disp MID"), (2, "disp HIGH")]:
        o = A[(A.db == b) & (A.era == "OOS")]; r = A[(A.db == b) & (A.era == "REC")]
        print(f"  {name:<12}{o.ic.mean():<+14.4f}{r.ic.mean():<+14.4f}{len(o):<10}", flush=True)
    print("\n  correlation(dispersion, IC):", flush=True)
    for era in ("OOS", "REC"):
        a = A[A.era == era]
        print(f"    {era}: spearman(disp, IC) = {spearmanr(a.disp, a.ic).correlation:+.2f}", flush=True)
    print("  (+ = edge stronger in HIGH dispersion; − = stronger in LOW dispersion)", flush=True)
    print("\nDISPDONE", flush=True)


if __name__ == "__main__":
    main()
