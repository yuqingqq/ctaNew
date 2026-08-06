"""DERIVED FROM THE LATENT MAP (not a guessed test):
The map: the edge = a low-VOL factor (idio_vol) + a MOMENTUM/REVERSAL factor (return_1d); which one
carries the edge rotates by era (ablation: idio_vol top OOS, return_1d top RECENT).

Pre-registered prediction from the low-vol-anomaly + short-reversal literature: the two factors are
REGIME-conditional (vol factor stronger in high-vol/risk-off; reversal stronger in high-vol/choppy;
momentum can dominate in strong low-vol trends). If so, the era-instability = a STABLE regime rotation,
not random drift — understandable and potentially conditionable.

Test: per-bar cross-sectional IC of each factor vs forward alpha, related to PIT regime (BTC vol,
market trend), era-locked terciles, both eras. Disciplined: pre-registered, era-locked, both-era.
Run: python3 -u -m live.build_regime_derive
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.v0_feature_ablation import build_panel

CUT = pd.Timestamp("2025-10-01", tz="UTC")
TGT = "alpha_vs_btc_realized"


def perbar_ic(PAN, feat):
    return (PAN.groupby("open_time")
            .apply(lambda g: spearmanr(g[feat], g[TGT]).correlation if g[feat].notna().sum() >= 8 else np.nan)
            .dropna())


def main():
    PAN = build_panel()
    print(f"panel {len(PAN):,} rows | {PAN.symbol.nunique()} syms\n", flush=True)
    reg = PAN.groupby("open_time").agg(volreg=("btc_rvol_7d", "mean"), trend=("return_1d", "mean"))
    ic_vol = perbar_ic(PAN, "idio_vol_to_btc_1d")   # low-vol anomaly -> expect NEGATIVE (low-vol wins)
    ic_mom = perbar_ic(PAN, "return_1d")            # short reversal -> expect NEGATIVE (winners revert)
    df = pd.DataFrame({"ic_vol": ic_vol, "ic_mom": ic_mom})
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.join(reg).dropna()
    df["era"] = np.where(df.index < CUT, "OOS", "REC")
    print(f"bars with IC: OOS {int((df.era=='OOS').sum())} / REC {int((df.era=='REC').sum())}", flush=True)
    print(f"overall factor IC: idio_vol OOS {df[df.era=='OOS'].ic_vol.mean():+.4f} / "
          f"REC {df[df.era=='REC'].ic_vol.mean():+.4f} | "
          f"return_1d OOS {df[df.era=='OOS'].ic_mom.mean():+.4f} / REC {df[df.era=='REC'].ic_mom.mean():+.4f}",
          flush=True)
    print("  (negative = the factor 'works': low-vol wins / winners revert)\n", flush=True)

    q = np.nanquantile(df.loc[df.era == "OOS", "volreg"], [1/3, 2/3])  # era-locked from OOS
    df["vb"] = np.digitize(df["volreg"].to_numpy(), q)
    print("=== factor IC by BTC-vol regime (era-locked terciles) — DERIVED PREDICTION: "
          "factors stronger (more negative) in high-vol ===", flush=True)
    print(f"  {'regime':<9}{'OOS ic_vol':<12}{'OOS ic_mom':<12}{'REC ic_vol':<12}{'REC ic_mom':<12}", flush=True)
    for b, name in [(0, "vol LOW"), (1, "vol MID"), (2, "vol HIGH")]:
        o = df[(df.vb == b) & (df.era == "OOS")]; r = df[(df.vb == b) & (df.era == "REC")]
        print(f"  {name:<9}{o.ic_vol.mean():<+12.4f}{o.ic_mom.mean():<+12.4f}"
              f"{r.ic_vol.mean():<+12.4f}{r.ic_mom.mean():<+12.4f}", flush=True)

    print("\n=== corr(factor-IC, regime) both eras (stable sign = regime-conditional edge) ===", flush=True)
    for era in ["OOS", "REC"]:
        d = df[df.era == era]
        print(f"  {era}: corr(ic_vol,volreg) {d.ic_vol.corr(d.volreg):+.2f} | "
              f"corr(ic_mom,volreg) {d.ic_mom.corr(d.volreg):+.2f} | "
              f"corr(ic_vol,trend) {d.ic_vol.corr(d.trend):+.2f} | "
              f"corr(ic_mom,trend) {d.ic_mom.corr(d.trend):+.2f}", flush=True)
    print("\nREGIMEDONE", flush=True)


if __name__ == "__main__":
    main()
