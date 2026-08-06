"""Confirmation for the intermediate-momentum lead: does skip-recent 14d momentum add INCREMENTAL rank-IC to V0
through the REAL per-symbol RidgeCV pipeline, both eras (paired day-clustered CI)? Unlike the cross-sectional
positioning signals (G2 null), mom_14_3 is a per-symbol feature the model CAN use. Run: python3 -u -m live.build_mom_pipeline
"""
from __future__ import annotations

import numpy as np

from live.v0_feature_ablation import build_panel, gen, perbar_ic, paired_ci, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import FULL
import pandas as pd


def main():
    PAN = build_panel()
    ex = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
    PAN = PAN.merge(ex, on=["symbol", "open_time"], how="left").sort_values(["symbol", "open_time"])
    PAN["mom_14_3"] = PAN.groupby("symbol")["return_pct"].transform(lambda s: s.shift(18).rolling(66).sum())
    PAN["mom_14_3"] = PAN.groupby("open_time")["mom_14_3"].transform(lambda s: s.fillna(s.median())).fillna(0.0)
    PAN = PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        ib = perbar_ic(gen(PAN, list(V0), cuts))
        iv = perbar_ic(gen(PAN, list(V0) + ["mom_14_3"], cuts))
        d, lo, up = paired_ci(ib, iv)
        tag = "ADDS (CI>0)" if (np.isfinite(lo) and lo > 0) else (
              "hurts (CI<0)" if (np.isfinite(up) and up < 0) else "within noise")
        print(f"  {era}: baseline {ib.mean():+.4f} | +mom_14_3 Δ {d:+.4f} [{lo:+.4f},{up:+.4f}]  {tag}", flush=True)
    print("MOMPIPEDONE", flush=True)


if __name__ == "__main__":
    main()
