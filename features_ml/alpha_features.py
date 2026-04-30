"""Alpha-residual targeted features.

These features are designed specifically to predict alpha = my_fwd - β × ref_fwd,
not raw forward return. They emphasize what's UNIQUE to the target symbol
relative to the reference, rather than what they have in common.

Five feature families:

  1. Dominance dynamics
     log(my/ref) level, change at multiple horizons, z-scored deviation.
     Captures relative-strength regimes (e.g. ETH outperforming BTC).

  2. Reference symbol's recent state
     Reference's own multi-horizon returns and EMA slope. Lead-lag relationships
     where the reference moves first and target follows in alpha.

  3. Beta dynamics
     Beta z-score (current short-window beta vs long baseline). When β shifts
     away from its norm, the residual ALSO shifts.

  4. Idiosyncratic returns / volatility
     idio[t-h:t] = my_ret - β × ref_ret over past windows. Past idiosyncratic
     moves and their volatility regime — most directly tied to forward alpha
     by construction.

  5. Order-flow divergence
     my_tfi - ref_tfi (or signed-volume z-score difference). When the symbol
     has aggressor flow that the reference lacks, alpha tends to follow.

All features are point-in-time (rolling stats use trailing windows; β estimates
are shifted by one bar to avoid using current information).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BETA_WINDOW_SHORT = 288    # 1 day at 5m
BETA_WINDOW_LONG = 2016    # 7 days at 5m


def add_alpha_features(
    target_feats: pd.DataFrame,
    ref_feats: pd.DataFrame,
    *,
    ref_label: str = "ref",
) -> pd.DataFrame:
    """Add alpha-targeted features to `target_feats` using `ref_feats` as reference.

    Both DataFrames must be indexed by `open_time` (UTC). `ref_feats` must have
    `close` (and ideally `tfi_smooth`, `signed_volume` for the order-flow family).

    All new column names end with `_vs_{ref_label}` so they're unambiguous when
    pooling across symbols.
    """
    out = target_feats.copy()
    my_close = out["close"]
    ref_close = ref_feats["close"].reindex(out.index).ffill()
    my_ret = my_close.pct_change()
    ref_ret = ref_close.pct_change()

    rl = ref_label

    # =========================================================================
    # 1. Dominance dynamics — log(my/ref)
    # =========================================================================
    spread = np.log(my_close / ref_close)
    out[f"dom_level_vs_{rl}"] = spread

    # Multi-horizon dominance change (relative momentum)
    for h in (12, 48, 288):  # 1h, 4h, 1d
        out[f"dom_change_{h}b_vs_{rl}"] = spread - spread.shift(h)

    # Z-scored dominance over 1d / 7d
    for w, name in ((288, "1d"), (2016, "7d")):
        rmean = spread.rolling(w, min_periods=max(48, w // 4)).mean()
        rstd = spread.rolling(w, min_periods=max(48, w // 4)).std().replace(0, np.nan)
        out[f"dom_z_{name}_vs_{rl}"] = ((spread - rmean) / rstd).clip(-5, 5)

    # =========================================================================
    # 2. Reference symbol's recent state (lead-lag)
    # =========================================================================
    for h in (12, 48):
        out[f"{rl}_ret_{h}b"] = ref_close.pct_change(h)

    # Reference EMA slope at 4h horizon
    ref_ema_short = ref_close.ewm(span=12, adjust=False).mean()
    ref_ema_long = ref_close.ewm(span=48, adjust=False).mean()
    out[f"{rl}_ema_slope_4h"] = (ref_ema_long - ref_ema_long.shift(12)) / ref_close.replace(0, np.nan)
    out[f"{rl}_ema_diff_short_long"] = (ref_ema_short - ref_ema_long) / ref_close.replace(0, np.nan)

    # Reference realized vol (1h) — symbol-symbol vol contagion
    out[f"{rl}_realized_vol_1h"] = ref_ret.rolling(12).std()

    # =========================================================================
    # 3. Beta dynamics
    # =========================================================================
    # Short-window beta (1d)
    cov_s = (my_ret * ref_ret).rolling(BETA_WINDOW_SHORT).mean() - \
            my_ret.rolling(BETA_WINDOW_SHORT).mean() * ref_ret.rolling(BETA_WINDOW_SHORT).mean()
    var_s = ref_ret.rolling(BETA_WINDOW_SHORT).var().replace(0, np.nan)
    beta_short = (cov_s / var_s).clip(-5, 5)

    # Long-window beta (7d) — baseline
    cov_l = (my_ret * ref_ret).rolling(BETA_WINDOW_LONG).mean() - \
            my_ret.rolling(BETA_WINDOW_LONG).mean() * ref_ret.rolling(BETA_WINDOW_LONG).mean()
    var_l = ref_ret.rolling(BETA_WINDOW_LONG).var().replace(0, np.nan)
    beta_long = (cov_l / var_l).clip(-5, 5)

    # Beta deviation: short - long, normalized by beta-history std
    beta_std = beta_short.rolling(BETA_WINDOW_LONG).std().replace(0, np.nan)
    out[f"beta_zscore_vs_{rl}"] = ((beta_short - beta_long) / beta_std).clip(-5, 5).shift(1)
    out[f"beta_short_vs_{rl}"] = beta_short.shift(1)

    # =========================================================================
    # 4. Idiosyncratic returns and volatility (orthogonal to ref)
    # =========================================================================
    # 1-bar idio: my_ret - β × ref_ret (β is point-in-time = shifted short beta)
    beta_pit = beta_short.shift(1)
    idio_1bar = my_ret - beta_pit * ref_ret

    # Past h-bar idio returns (short cumulative idio)
    for h in (12, 48):
        my_ret_h = my_close.pct_change(h)
        ref_ret_h = ref_close.pct_change(h)
        out[f"idio_ret_{h}b_vs_{rl}"] = my_ret_h - beta_pit * ref_ret_h

    # Idio volatility regime
    idio_vol_1h = idio_1bar.rolling(12).std()
    idio_vol_1d = idio_1bar.rolling(288).std()
    out[f"idio_vol_1h_vs_{rl}"] = idio_vol_1h
    out[f"idio_vol_1d_vs_{rl}"] = idio_vol_1d
    # Ratio: is idio vol elevated NOW vs typical?
    out[f"idio_vol_ratio_vs_{rl}"] = (idio_vol_1h / idio_vol_1d.replace(0, np.nan)).clip(0, 5)

    # =========================================================================
    # 5. Correlation regime
    # =========================================================================
    std_my = my_ret.rolling(BETA_WINDOW_SHORT).std()
    std_ref = ref_ret.rolling(BETA_WINDOW_SHORT).std()
    corr_1d = (cov_s / (std_my * std_ref).replace(0, np.nan)).clip(-1, 1)
    out[f"corr_1d_vs_{rl}"] = corr_1d
    # 3d change in correlation (regime shift)
    out[f"corr_change_3d_vs_{rl}"] = corr_1d - corr_1d.shift(864)

    # =========================================================================
    # 6. Order-flow divergence (if available)
    # =========================================================================
    if "tfi_smooth" in out.columns and "tfi_smooth" in ref_feats.columns:
        ref_tfi = ref_feats["tfi_smooth"].reindex(out.index).ffill()
        out[f"tfi_diff_vs_{rl}"] = out["tfi_smooth"] - ref_tfi

    if "signed_volume" in out.columns and "signed_volume" in ref_feats.columns:
        # Standardize each by its own rolling std before differencing — avoids
        # SOL's much larger absolute signed_volume from dominating.
        my_sv = out["signed_volume"]
        ref_sv = ref_feats["signed_volume"].reindex(out.index).ffill()
        my_sv_z = my_sv / my_sv.rolling(288).std().replace(0, np.nan)
        ref_sv_z = ref_sv / ref_sv.rolling(288).std().replace(0, np.nan)
        out[f"signed_vol_z_diff_vs_{rl}"] = (my_sv_z - ref_sv_z).clip(-5, 5)

    return out


ALPHA_FEATURE_NAMES = [
    # 1. Dominance
    "dom_level_vs_{rl}",
    "dom_change_12b_vs_{rl}", "dom_change_48b_vs_{rl}", "dom_change_288b_vs_{rl}",
    "dom_z_1d_vs_{rl}", "dom_z_7d_vs_{rl}",
    # 2. Reference state
    "{rl}_ret_12b", "{rl}_ret_48b",
    "{rl}_ema_slope_4h", "{rl}_ema_diff_short_long",
    "{rl}_realized_vol_1h",
    # 3. Beta
    "beta_zscore_vs_{rl}", "beta_short_vs_{rl}",
    # 4. Idio
    "idio_ret_12b_vs_{rl}", "idio_ret_48b_vs_{rl}",
    "idio_vol_1h_vs_{rl}", "idio_vol_1d_vs_{rl}", "idio_vol_ratio_vs_{rl}",
    # 5. Correlation
    "corr_1d_vs_{rl}", "corr_change_3d_vs_{rl}",
    # 6. Flow divergence
    "tfi_diff_vs_{rl}", "signed_vol_z_diff_vs_{rl}",
]


def alpha_feature_columns(ref_label: str) -> list[str]:
    return [c.format(rl=ref_label) for c in ALPHA_FEATURE_NAMES]
