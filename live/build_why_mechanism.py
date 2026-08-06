"""CHECK the mechanism behind the low-vol edge (currently just my inference: "high-vol = overbid hype
that underperforms residually").

Testable predictions if hype-fading / low-vol anomaly is right:
  1. ASYMMETRY: forward residual return by vol quintile is driven by the HIGH-vol tail crashing
     (short side), not a symmetric low-vol-rises effect.
  2. FROTH: within high vol, the recently-PUMPED names (high run-up) underperform MOST (the hype fade).
If instead it's monotone & symmetric and not froth-concentrated -> "hype-fading" is wrong; it's a
generic low-vol tilt (or something else).

Market-neutral: forward residual = alpha_vs_btc_realized demeaned per bar. Both eras.
Run: python3 -u -m live.build_why_mechanism
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel

CUT = pd.Timestamp("2025-10-01", tz="UTC")


def qcut_bar(df, col, n):
    r = df.groupby("open_time")[col].rank(pct=True)
    return np.clip((r * n).astype(int), 0, n - 1)


def main():
    PAN = build_panel()
    PAN["fwd"] = (PAN["alpha_vs_btc_realized"]
                  - PAN.groupby("open_time")["alpha_vs_btc_realized"].transform("mean")) * 1e4  # bps, mkt-neutral
    PAN = PAN.dropna(subset=["fwd", "idio_vol_to_btc_1d", "ret_3d"])
    PAN["volq"] = qcut_bar(PAN, "idio_vol_to_btc_1d", 5)   # Q0=low vol ... Q4=high vol
    PAN["ruq"] = qcut_bar(PAN, "ret_3d", 3)                # T0=recent loser ... T2=recent winner
    PAN["era"] = np.where(PAN["open_time"] < CUT, "OOS", "REC")

    print("=== (1) forward residual return (bps) by VOL quintile — is it asymmetric (high-vol crash)? ===",
          flush=True)
    print(f"  {'':<10}{'Q0 lowvol':<11}{'Q1':<9}{'Q2':<9}{'Q3':<9}{'Q4 hivol':<11}{'lowvol−hivol':<14}", flush=True)
    for era in ("OOS", "REC"):
        g = PAN[PAN.era == era].groupby("volq")["fwd"].mean()
        mid = g.get(2, np.nan)
        print(f"  {era:<10}" + "".join(f"{g.get(k,np.nan):<+9.1f} " for k in range(5))
              + f"  {g.get(0,np.nan)-g.get(4,np.nan):<+.1f}", flush=True)
        print(f"  {'  asym→':<10}low-side (Q0−mid) {g.get(0,np.nan)-mid:+.1f} | "
              f"high-side (Q4−mid) {g.get(4,np.nan)-mid:+.1f}  "
              f"(|high|>|low| = crash-driven/short-side)", flush=True)

    print("\n=== (2) FROTH: forward residual (bps) by VOL quintile × recent run-up — "
          "is the pumped high-vol cell the worst? ===", flush=True)
    for era in ("OOS", "REC"):
        print(f"  [{era}]  rows=vol quintile, cols=run-up (loser/mid/winner):", flush=True)
        t = PAN[PAN.era == era].groupby(["volq", "ruq"])["fwd"].mean().unstack()
        for q in range(5):
            lab = "Q0 lowvol" if q == 0 else ("Q4 hivol" if q == 4 else f"Q{q}")
            print(f"    {lab:<10}" + "".join(f"{t.loc[q].get(c,np.nan):<+9.1f} " for c in range(3)), flush=True)
    print("\nMECHDONE", flush=True)


if __name__ == "__main__":
    main()
