"""Funding-rate feature engineering for cross-sectional alpha v7.

Anti-leakage: every feature at bar open_time t uses only funding publications
with calc_time <= t. Aligned via `align_funding_to_bars` (merge_asof backward).

Features:
  - funding_rate          : last published rate (decimal, e.g. 0.0001 = 1bp/8h)
  - funding_rate_z_7d     : 7d rolling z-score (window = 21 publications ≈ 7d
                              of 8h-period rates, but here computed at bar grid
                              with .rolling(2016) on bar index)
  - funding_rate_1d_change: rate now minus rate 1d ago (3 publications back)
  - funding_streak_pos    : consecutive bars where funding_rate > 0 (capped 21)
  - funding_streak_neg    : consecutive bars where funding_rate < 0 (capped 21)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data_collectors.funding_rate_loader import align_funding_to_bars, load_funding_rate

log = logging.getLogger(__name__)


def add_funding_features(feats: pd.DataFrame, symbol: str,
                          start_month: str = "2025-01",
                          end_month: str = "2026-03") -> pd.DataFrame:
    """Add funding-rate-derived features to a per-symbol kline frame.

    `feats` should be indexed by open_time (UTC). All new features are
    point-in-time at the bar's open_time.
    """
    out = feats.copy()
    funding_df = load_funding_rate(symbol, start_month=start_month, end_month=end_month)
    if funding_df.empty:
        log.warning("[%s] no funding data; features will be NaN", symbol)
        for c in ["funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
                   "funding_streak_pos", "funding_streak_neg"]:
            out[c] = np.nan
        return out

    bar_index = out.index
    if bar_index.tz is None:
        bar_index = bar_index.tz_localize("UTC")

    # Aligned current rate (PIT via merge_asof backward — only past funding visible at t)
    out["funding_rate"] = align_funding_to_bars(funding_df, bar_index).to_numpy()

    # 7d rolling z-score of funding_rate (288 bars/day × 7 = 2016, min 288)
    rmean = out["funding_rate"].rolling(2016, min_periods=288).mean()
    rstd = out["funding_rate"].rolling(2016, min_periods=288).std().replace(0, np.nan)
    out["funding_rate_z_7d"] = ((out["funding_rate"] - rmean) / rstd).clip(-5, 5)

    # 1d change in funding rate (288 bars back ≈ 3 settlement intervals at 8h)
    out["funding_rate_1d_change"] = out["funding_rate"] - out["funding_rate"].shift(288)

    # Consecutive-same-sign streak (capped at 21 = 1 week of settlements)
    sign = np.sign(out["funding_rate"]).fillna(0)
    pos_streak = sign.where(sign > 0).fillna(0).abs()
    neg_streak = sign.where(sign < 0).fillna(0).abs()

    def _streak(s: pd.Series, max_cap: int = 21) -> pd.Series:
        # Counts consecutive 1s in s (assumes s ∈ {0, 1})
        grp = (s != s.shift()).cumsum()
        return s.groupby(grp).cumsum().clip(upper=max_cap)

    out["funding_streak_pos"] = _streak(pos_streak)
    out["funding_streak_neg"] = _streak(neg_streak)

    return out


FUNDING_FEATURES = [
    "funding_rate",
    "funding_rate_z_7d",
    "funding_rate_1d_change",
    "funding_streak_pos",
    "funding_streak_neg",
]
