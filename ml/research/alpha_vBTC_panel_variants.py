"""Build 3 basket-construction variants on the 51-name panel.

Variant A (control): equal-weight mean basket, β=1 residual.
Variant B: inverse-volatility weighted basket (1/atr_pct), β=1 residual.
Variant C: trimmed equal-weight (drop top/bottom 10% per cycle), β=1 residual.

All 3 use β=1 (no β-adjustment) for apples-to-apples comparison of basket
construction effect alone. The original panel has β-adjusted alpha; this
script builds simpler residuals.

Output: panel_variants.parquet with new columns
  basket_A_fwd, basket_B_fwd, basket_C_fwd
  alpha_A, alpha_B, alpha_C
  target_A, target_B, target_C   (z-scored per symbol via expanding mean,
                                  rolling 7d std — matches v6_clean pattern)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PANEL_IN = REPO / "outputs/vBTC_features/panel_with_btc_features.parquet"
PANEL_OUT = REPO / "outputs/vBTC_features/panel_variants.parquet"
HORIZON = 48


def build_baskets(g: pd.DataFrame) -> pd.Series:
    rets = g["return_pct"].to_numpy()
    atr = g["atr_pct"].to_numpy()
    valid = ~(np.isnan(rets) | np.isnan(atr) | (atr <= 0))
    if valid.sum() < 5:
        return pd.Series({"basket_A_fwd": np.nan, "basket_B_fwd": np.nan,
                          "basket_C_fwd": np.nan})
    rets_v = rets[valid]
    atr_v = atr[valid]
    # A: equal-weight mean
    A = rets_v.mean()
    # B: inverse-vol weighted
    w = 1.0 / np.clip(atr_v, 1e-4, None)
    B = float((rets_v * w).sum() / w.sum())
    # C: trimmed mean (drop top/bottom 10%)
    n = len(rets_v)
    n_drop = max(1, n // 10)
    sorted_rets = np.sort(rets_v)
    C = float(sorted_rets[n_drop:-n_drop].mean()) if (n - 2 * n_drop) >= 1 else A
    return pd.Series({"basket_A_fwd": A, "basket_B_fwd": B, "basket_C_fwd": C})


def add_targets(panel: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Add alpha_X and target_X columns for a basket variant."""
    bk_col = f"basket_{variant}_fwd"
    alpha_col = f"alpha_{variant}"
    target_col = f"target_{variant}"
    panel[alpha_col] = panel["return_pct"] - panel[bk_col]
    # Per-symbol z-score using expanding mean (shifted by horizon to avoid lookahead)
    # and rolling 7d std (288*7 bars).
    out = []
    for s, g in panel.groupby("symbol", sort=False):
        a = g[alpha_col]
        rmean = a.expanding(min_periods=288).mean().shift(HORIZON)
        rstd = a.rolling(288 * 7, min_periods=288).std().shift(HORIZON)
        z = (a - rmean) / rstd.replace(0, np.nan)
        out.append(z.rename(target_col))
    panel[target_col] = pd.concat(out).sort_index()
    return panel


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_IN)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    print(f"Computing basket variants per timestamp...", flush=True)
    t0 = time.time()
    baskets = panel.groupby("open_time", sort=False).apply(build_baskets)
    baskets = baskets.reset_index()
    print(f"  built {len(baskets):,} timestamp baskets in {time.time()-t0:.0f}s", flush=True)

    print(f"Merging baskets back into panel...", flush=True)
    panel = panel.merge(baskets, on="open_time", how="left")

    for v in ("A", "B", "C"):
        print(f"Computing alpha_{v} + target_{v}...", flush=True)
        t0 = time.time()
        panel = add_targets(panel, v)
        ar = panel[f"alpha_{v}"].dropna()
        tg = panel[f"target_{v}"].dropna()
        print(f"  alpha_{v}: mean={ar.mean():+.6f}, std={ar.std():.6f}", flush=True)
        print(f"  target_{v}: mean={tg.mean():+.4f}, std={tg.std():.4f}, n={len(tg):,}", flush=True)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    print(f"Saving panel...", flush=True)
    panel.to_parquet(PANEL_OUT, index=False)
    print(f"  saved → {PANEL_OUT}", flush=True)


if __name__ == "__main__":
    main()
