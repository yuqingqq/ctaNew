"""Regime / drawdown / drift-residual features.

Built from `compute_kline_features` output. All point-in-time (use trailing
windows shifted by 1 bar where appropriate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_regime_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Add curated regime / drawdown / residual features to a feats DataFrame.

    Required input columns: `close`, `atr_pct`.
    """
    out = feats.copy()
    close = out["close"]
    atr = out["atr_pct"]
    ret = close.pct_change()

    # === volatility regime ===
    # ATR z-score over trailing 1d (288 bars). Captures "is current vol unusual".
    out["atr_zscore_1d"] = (atr - atr.rolling(288).mean()) / atr.rolling(288).std()
    # 30d percentile of ATR_pct, shifted to use only past data.
    out["atr_pctile_30d"] = atr.rolling(8640, min_periods=288).rank(pct=True).shift(1)

    # === drift residuals (return minus rolling mean) ===
    out["detrend_resid_1h"] = ret - ret.rolling(12).mean()
    out["detrend_resid_1d"] = ret - ret.rolling(288).mean()

    # === drawdown / time-since-high ===
    high_1d = close.rolling(288).max()
    out["dd_from_1d_high"] = close / high_1d - 1
    is_new_high = (close == high_1d).astype(int)
    out["bars_since_high"] = (
        (1 - is_new_high)
        .groupby(is_new_high.cumsum())
        .cumcount()
    )

    return out
