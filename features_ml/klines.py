"""Kline-based feature wrapper.

Thin wrapper over the existing `HFFeatureEngine` plus a small set of ML-specific
extras (longer-horizon returns, realized vol, ATR%). Output is a DataFrame
indexed by bar `open_time` with strictly point-in-time features.

Design notes:
- All features are computed from bars at-or-before `t`; nothing leaks from
  bar `t+1` or later. The underlying `HFFeatureEngine` already enforces this
  via rolling/ewm.
- Warmup rows (those with NaN in any lookback feature) are flagged but not
  auto-dropped — labelers / models decide their own warmup tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hf_features import HFFeatureEngine


def compute_kline_features(klines: pd.DataFrame) -> pd.DataFrame:
    """Compute the full kline feature set.

    Parameters
    ----------
    klines : DataFrame indexed by `open_time` (UTC) with columns
        open, high, low, close, volume.

    Returns
    -------
    DataFrame with the same index plus 160+ feature columns.
    """
    if not isinstance(klines.index, pd.DatetimeIndex):
        raise TypeError("klines must be indexed by DatetimeIndex (open_time)")
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(klines.columns)
    if missing:
        raise ValueError(f"klines missing columns: {missing}")

    feats = HFFeatureEngine.calculate_features(klines.copy())

    # ATR% is more useful than raw ATR for cross-symbol / cross-regime work.
    if "atr_14" in feats.columns and "atr_pct" not in feats.columns:
        feats["atr_pct"] = feats["atr_14"] / feats["close"]

    # Longer-horizon returns (1h=12, 4h=48, 1d=288 at 5m bars).
    for horizon, name in [(12, "1h"), (48, "4h"), (288, "1d")]:
        col = f"return_{name}"
        if col not in feats.columns:
            feats[col] = feats["close"].pct_change(horizon)

    # Realized volatility over multiple windows (annualized later if needed).
    log_ret = np.log(feats["close"]).diff()
    for window, name in [(12, "1h"), (48, "4h"), (288, "1d")]:
        feats[f"realized_vol_{name}"] = log_ret.rolling(window).std()

    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    """Return only feature columns (excludes raw OHLCV passthrough)."""
    raw = {"open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"}
    return [c for c in feats.columns if c not in raw]
