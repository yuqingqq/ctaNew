"""Cross-asset features: relative momentum, beta, correlation, log-spread.

Adds features that reference another symbol's price/return series. For each
target symbol, a reference symbol is specified (e.g. BTC's reference is ETH;
ETH and SOL reference BTC). All features computed on synchronized timestamps
(5-min bars from Binance Vision share the same grid).

Features added:
    excess_ret_N_vs_ref   N-bar return minus reference's N-bar return (N ∈ {3, 12, 48})
    beta_ref_1d           rolling beta of my returns to ref returns (288 bars)
    corr_ref_1d           rolling Pearson correlation
    spread_log_vs_ref     log(my_close / ref_close) — current dominance level
    spread_zscore_1d      z-score of spread over last 1d (mean reversion in spread)
    spread_zscore_7d      z-score of spread over last 7d (longer dominance regime)

All features are point-in-time (rolling stats use trailing windows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_cross_asset_features(
    target_feats: pd.DataFrame, ref_feats: pd.DataFrame, *, ref_label: str = "ref",
) -> pd.DataFrame:
    """Add cross-asset features to `target_feats` using `ref_feats` as reference.

    Both DataFrames must be indexed by `open_time` (UTC) and have a `close`
    column at minimum. Returns target_feats with new columns added.

    Parameters
    ----------
    target_feats : DataFrame indexed by `open_time` for the symbol we're modeling.
    ref_feats   : DataFrame indexed by `open_time` for the reference symbol.
    ref_label   : suffix for the new column names (default "ref"); use the
                   actual symbol abbreviation (e.g. "btc") for clarity.
    """
    out = target_feats.copy()
    # Align reference by index
    ref_close = ref_feats["close"].reindex(out.index)
    my_close = out["close"]
    my_ret = my_close.pct_change()
    ref_ret = ref_close.pct_change()

    # === Excess (relative) returns at multiple horizons ===
    for h in (3, 12, 48):
        out[f"excess_ret_{h}_vs_{ref_label}"] = (
            my_close.pct_change(h) - ref_close.pct_change(h)
        )

    # === Rolling beta and correlation (1 day) ===
    win = 288  # 1 day at 5m
    cov = (my_ret * ref_ret).rolling(win).mean() - my_ret.rolling(win).mean() * ref_ret.rolling(win).mean()
    var = ref_ret.rolling(win).var().replace(0, np.nan)
    out[f"beta_{ref_label}_1d"] = (cov / var).clip(-5, 5)

    # Pearson correlation: cov / (std_my * std_ref)
    std_my = my_ret.rolling(win).std()
    std_ref = ref_ret.rolling(win).std()
    out[f"corr_{ref_label}_1d"] = (cov / (std_my * std_ref).replace(0, np.nan)).clip(-1, 1)

    # === Log spread (current dominance level) ===
    spread = np.log(my_close / ref_close)
    out[f"spread_log_vs_{ref_label}"] = spread

    # Z-score of spread (mean reversion in dominance)
    for w, name in ((288, "1d"), (2016, "7d")):
        rmean = spread.rolling(w, min_periods=max(48, w // 4)).mean()
        rstd = spread.rolling(w, min_periods=max(48, w // 4)).std().replace(0, np.nan)
        out[f"spread_zscore_{name}_vs_{ref_label}"] = ((spread - rmean) / rstd).clip(-5, 5)

    return out
