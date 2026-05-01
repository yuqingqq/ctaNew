"""Cross-sectional alpha pipeline.

Builds a multi-symbol universe from kline parquets and produces a cross-sectional
residual target plus basket-relative features.

Design:

  At each bar t, with N symbols in the universe:
    - basket_ret[t] = mean_s(my_ret_s[t])              (equal-weight basket)
    - basket_close[t] = product over time of (1 + basket_ret) (basket index)

  For each symbol s:
    - alpha_s[t] = my_fwd_s[t] - basket_fwd[t]         (cross-sectional residual)
    - or β-adjusted: alpha_s[t] = my_fwd_s[t] - β_s[t] × basket_fwd[t]

  Features per (s, t):
    - All base kline + regime features (existing pipeline)
    - dom_level_vs_basket = log(my_close / basket_close)
    - dom_change_{12,48,288}b_vs_basket
    - dom_z_{1d,7d}_vs_basket
    - basket_ret_{12,48}b
    - basket_ema_slope_4h
    - idio_ret_{12,48}b_vs_basket
    - idio_vol_{1h,1d}_vs_basket
    - corr_1d_vs_basket, corr_change_3d_vs_basket
    - beta_short_vs_basket
    - sym_id (categorical)

  Cross-sectional ranking at inference: predict alpha_s, rank, long top-K / short bottom-K.

The basket index is computed POINT-IN-TIME using only past returns — no
look-ahead bias. Cross-sectional stats (rank, decile assignment) are also
strictly per-bar so they don't leak future information.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from features_ml.klines import compute_kline_features
from features_ml.regime_features import add_regime_features

log = logging.getLogger(__name__)

DATA_DIR = Path("data/ml/test/parquet")
CACHE_DIR = Path("data/ml/cache")

BETA_WINDOW = 288


def list_universe(min_days: int = 200) -> list[str]:
    """Return all symbols in the kline directory with at least `min_days` daily files."""
    out = []
    for sym_dir in sorted((DATA_DIR / "klines").iterdir()):
        if not sym_dir.is_dir():
            continue
        files = sorted(sym_dir.glob("5m/*.parquet"))
        if len(files) >= min_days:
            out.append(sym_dir.name)
    return out


def build_kline_features(symbol: str, *, force_rebuild: bool = False) -> pd.DataFrame:
    """Compute kline features for one symbol with disk caching (xs_feats_<symbol>)."""
    cache = CACHE_DIR / f"xs_feats_{symbol}.parquet"
    if not force_rebuild and cache.exists():
        return pd.read_parquet(cache)

    paths = sorted((DATA_DIR / f"klines/{symbol}/5m").glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    klines = pd.concat([pd.read_parquet(p) for p in paths]).sort_values("open_time").set_index("open_time")
    feats = compute_kline_features(klines)
    feats = add_regime_features(feats)

    # Autocorrelation regime flag (consistent with the existing v3 pipeline)
    ret = feats["close"].pct_change()
    feats["autocorr_1h"] = ret.rolling(36).apply(lambda s: s.autocorr(lag=1) if s.std() > 0 else 0.0)
    feats["autocorr_pctile_7d"] = (
        feats["autocorr_1h"].rolling(2016, min_periods=288).rank(pct=True).shift(1)
    )

    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cache, compression="zstd")
    return feats


def build_basket(closes: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Equal-weight basket: returns and price-index series.

    Parameters
    ----------
    closes : DataFrame, columns = symbols, index = open_time. NaN where a symbol
             didn't yet exist or has missing data — basket return for that bar
             is the mean of the available symbols.

    Returns
    -------
    basket_ret : Series of mean per-bar returns
    basket_close : Series of cumulative-product basket index
    """
    rets = closes.pct_change()
    basket_ret = rets.mean(axis=1, skipna=True)
    basket_close = (1.0 + basket_ret.fillna(0.0)).cumprod()
    return basket_ret, basket_close


def add_basket_features(
    feats: pd.DataFrame, basket_close: pd.Series, basket_ret: pd.Series,
) -> pd.DataFrame:
    """Add basket-relative features to a single symbol's feature frame.

    All names use the suffix `_vs_bk`."""
    out = feats.copy()
    my_close = out["close"]
    bk_close = basket_close.reindex(out.index).ffill()
    bk_ret = basket_ret.reindex(out.index)
    my_ret = my_close.pct_change()

    # 1. Dominance vs basket
    spread = np.log(my_close / bk_close)
    out["dom_level_vs_bk"] = spread
    for h in (12, 48, 288):
        out[f"dom_change_{h}b_vs_bk"] = spread - spread.shift(h)
    for w, name in ((288, "1d"), (2016, "7d")):
        rmean = spread.rolling(w, min_periods=max(48, w // 4)).mean()
        rstd = spread.rolling(w, min_periods=max(48, w // 4)).std().replace(0, np.nan)
        out[f"dom_z_{name}_vs_bk"] = ((spread - rmean) / rstd).clip(-5, 5)

    # 2. Basket recent state
    for h in (12, 48):
        out[f"bk_ret_{h}b"] = bk_close.pct_change(h)
    bk_ema_long = bk_close.ewm(span=48, adjust=False).mean()
    out["bk_ema_slope_4h"] = (bk_ema_long - bk_ema_long.shift(12)) / bk_close.replace(0, np.nan)
    out["bk_realized_vol_1h"] = bk_ret.rolling(12).std()

    # 3. Beta and correlation (vs basket)
    cov = (my_ret * bk_ret).rolling(BETA_WINDOW).mean() - \
          my_ret.rolling(BETA_WINDOW).mean() * bk_ret.rolling(BETA_WINDOW).mean()
    var = bk_ret.rolling(BETA_WINDOW).var().replace(0, np.nan)
    beta_short = (cov / var).clip(-5, 5)
    out["beta_short_vs_bk"] = beta_short.shift(1)
    std_my = my_ret.rolling(BETA_WINDOW).std()
    std_bk = bk_ret.rolling(BETA_WINDOW).std()
    corr = (cov / (std_my * std_bk).replace(0, np.nan)).clip(-1, 1)
    out["corr_1d_vs_bk"] = corr
    out["corr_change_3d_vs_bk"] = corr - corr.shift(864)

    # 4. Idiosyncratic returns / vol
    beta_pit = beta_short.shift(1)
    idio_1bar = my_ret - beta_pit * bk_ret
    for h in (12, 48):
        out[f"idio_ret_{h}b_vs_bk"] = my_close.pct_change(h) - beta_pit * bk_close.pct_change(h)
    out["idio_vol_1h_vs_bk"] = idio_1bar.rolling(12).std()
    out["idio_vol_1d_vs_bk"] = idio_1bar.rolling(288).std()
    out["idio_vol_ratio_vs_bk"] = (out["idio_vol_1h_vs_bk"] / out["idio_vol_1d_vs_bk"].replace(0, np.nan)).clip(0, 5)

    return out


XS_BASE_FEATURES = [
    "return_1d", "ema_slope_20_1h", "bars_since_high",
    "atr_pct", "volume_ma_50",
    "hour_cos", "hour_sin",
]

XS_CROSS_FEATURES = [
    "dom_level_vs_bk",
    "dom_z_7d_vs_bk",
    "dom_change_288b_vs_bk",
    "bk_ret_48b",
    "bk_ema_slope_4h",
    "idio_vol_1d_vs_bk",
    "idio_ret_48b_vs_bk",
    "corr_change_3d_vs_bk",
    "beta_short_vs_bk",
]

# v5 flow features. Selected by alpha_v4_flow_audit.py: each passed
# the gates |IS IC| ≥ 0.015, sign-consistent ≥80% of symbols, and
# IS-OOS sign match ≥60%. `obv_z_1d` is engineered (rolling 1d z-score
# of OBV); the rest exist in xs_feats caches from compute_kline_features.
XS_FLOW_FEATURES = [
    "obv_z_1d",            # engineered, OOS |IC| 0.066
    "vwap_slope_96",       # OOS |IC| 0.043
    "vwap_zscore",         # OOS |IC| 0.044
    "price_volume_corr_20",  # OOS |IC| 0.032
    "obv_signal",          # OOS |IC| 0.034
    "mfi",                 # OOS |IC| 0.027
    "price_volume_corr_10",  # OOS |IC| 0.025
]

XS_FEATURE_COLS = XS_BASE_FEATURES + XS_CROSS_FEATURES + ["sym_id"]
XS_FEATURE_COLS_V5 = XS_BASE_FEATURES + XS_CROSS_FEATURES + XS_FLOW_FEATURES + ["sym_id"]

# v5_lean: drop the 5 flow features with OOS LGBM gain < 0.5% (kept v5 audit
# IS-OOS gates but the LGBM didn't pick them up) and 2 base features that
# are useless at h=288 (hour_cos/sin had OOS |IC| ~0). Survivors are the
# 2 productive flow features + 15 base/cross + sym_id = 18.
XS_FLOW_FEATURES_LEAN = [
    "obv_z_1d",       # gain 1.96%
    "vwap_slope_96",  # gain 0.77%
    "vwap_zscore",    # gain 0.30% (kept — borderline, has 92% sign consistency)
]
XS_BASE_FEATURES_LEAN = [
    "return_1d", "ema_slope_20_1h", "bars_since_high",
    "atr_pct", "volume_ma_50",
    # dropped: hour_cos, hour_sin (useless at 1d horizon)
]
XS_FEATURE_COLS_V5_LEAN = (XS_BASE_FEATURES_LEAN + XS_CROSS_FEATURES
                            + XS_FLOW_FEATURES_LEAN + ["sym_id"])

# v6: v5 + per-bar cross-sectional pctile ranks of select base/flow features.
# Hypothesis: addresses per-symbol IC heterogeneity (+0.18 to -0.07 across
# symbols). Per-bar rank is point-in-time by construction (uses only bar-t
# universe values).
XS_RANK_FEATURES = [
    "return_1d_xs_rank",
    "atr_pct_xs_rank",
    "volume_ma_50_xs_rank",
    "bars_since_high_xs_rank",
    "ema_slope_20_1h_xs_rank",
    "idio_vol_1d_vs_bk_xs_rank",
    "obv_z_1d_xs_rank",
    "vwap_zscore_xs_rank",
]
# Source feature → rank-feature name mapping
XS_RANK_SOURCES = {
    "return_1d": "return_1d_xs_rank",
    "atr_pct": "atr_pct_xs_rank",
    "volume_ma_50": "volume_ma_50_xs_rank",
    "bars_since_high": "bars_since_high_xs_rank",
    "ema_slope_20_1h": "ema_slope_20_1h_xs_rank",
    "idio_vol_1d_vs_bk": "idio_vol_1d_vs_bk_xs_rank",
    "obv_z_1d": "obv_z_1d_xs_rank",
    "vwap_zscore": "vwap_zscore_xs_rank",
}
XS_FEATURE_COLS_V6 = (XS_BASE_FEATURES + XS_CROSS_FEATURES
                       + XS_FLOW_FEATURES + XS_RANK_FEATURES + ["sym_id"])

# v6_clean: drop 4 features confirmed harmful by permutation-importance audit
# (alpha_v6_permutation_lean.py, OOS holdout):
#   - beta_short_vs_bk      gain 9.06%, perm_drop -0.0014 (active OOS drag)
#   - idio_vol_1d_vs_bk     gain 7.37%, perm_drop -0.0001 (capacity wasted)
#   - bars_since_high       gain 3.70%, perm_drop -0.0008 (small drag)
#   - volume_ma_50_xs_rank  gain 4.19%, perm_drop -0.0007 (small drag)
# `idio_vol_1d_vs_bk` and `bars_since_high` are kept in the panel (computed
# upstream) as sources for their xs_rank derivatives, just not exposed as
# training features. `beta_short_vs_bk` is also still in the panel because
# β-neutral execution reads it.
XS_BASE_FEATURES_CLEAN = [c for c in XS_BASE_FEATURES if c != "bars_since_high"]
XS_CROSS_FEATURES_CLEAN = [c for c in XS_CROSS_FEATURES
                            if c not in ("beta_short_vs_bk", "idio_vol_1d_vs_bk")]
XS_RANK_FEATURES_CLEAN = [c for c in XS_RANK_FEATURES if c != "volume_ma_50_xs_rank"]
XS_FEATURE_COLS_V6_CLEAN = (XS_BASE_FEATURES_CLEAN + XS_CROSS_FEATURES_CLEAN
                             + XS_FLOW_FEATURES + XS_RANK_FEATURES_CLEAN + ["sym_id"])

# v7: v6 + 3 funding-rate features. From alpha_v7_funding_audit.py:
# all 3 passed |IS IC|≥0.015, sign-consistent ≥80%, IS-OOS match ≥60%.
# OOS |IC| range 0.068-0.084 — strongest single features in the entire set.
XS_FUNDING_FEATURES = [
    "funding_rate",         # OOS |IC| 0.068
    "funding_rate_z_7d",    # OOS |IC| 0.080
    "funding_streak_pos",   # OOS |IC| 0.084
]
XS_FEATURE_COLS_V7 = (XS_BASE_FEATURES + XS_CROSS_FEATURES + XS_FLOW_FEATURES
                       + XS_RANK_FEATURES + XS_FUNDING_FEATURES + ["sym_id"])

# v7_lean: v6 + only the 2 substantively-used funding features (drop
# funding_streak_pos which had only 0.15% gain in v7).
XS_FUNDING_FEATURES_LEAN = ["funding_rate", "funding_rate_z_7d"]
XS_FEATURE_COLS_V7_LEAN = (XS_BASE_FEATURES + XS_CROSS_FEATURES + XS_FLOW_FEATURES
                            + XS_RANK_FEATURES + XS_FUNDING_FEATURES_LEAN + ["sym_id"])


def add_xs_rank_features(panel: pd.DataFrame, sources: dict = None) -> pd.DataFrame:
    """Add per-bar cross-sectional pctile ranks of selected features.

    Anti-leakage: per-bar rank uses only that bar's universe values, which
    are themselves point-in-time. Output is in [0, 1] (NaN preserved when
    the source value is NaN — the symbol just isn't ranked at that bar).

    Parameters
    ----------
    panel : DataFrame with `open_time` and `symbol` columns plus source feature cols.
    sources : dict {source_col: rank_col_name}. Defaults to XS_RANK_SOURCES.
    """
    if sources is None:
        sources = XS_RANK_SOURCES
    out = panel.copy()
    for src, dst in sources.items():
        if src not in out.columns:
            log.warning("xs_rank: source col %s missing; skipping", src)
            continue
        out[dst] = out.groupby("open_time")[src].rank(pct=True)
    return out


def add_engineered_flow_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Add features that don't already exist in xs_feats caches but are
    cheap to compute on-the-fly. Currently: obv_z_1d (rolling 1d z-score
    of OBV).

    Anti-leakage: the rolling z-score uses left-aligned window — at bar t
    it includes bar-t OBV value but only past values otherwise. Same
    pattern as existing dom_z_*_vs_bk features. Not formal leakage.
    """
    out = feats.copy()
    if "obv" in out.columns and "obv_z_1d" not in out.columns:
        rolling_mean = out["obv"].rolling(288, min_periods=48).mean()
        rolling_std = out["obv"].rolling(288, min_periods=48).std().replace(0, np.nan)
        out["obv_z_1d"] = (out["obv"] - rolling_mean) / rolling_std
    return out


def assemble_universe(symbols: list[str], horizon: int = 48) -> dict:
    """Build the full cross-sectional dataset:
       - per-symbol features enriched with basket-relative features
       - cross-sectional alpha target (alpha = my_fwd - basket_fwd)

    Returns dict with:
       'symbols': list, 'sym_id': {sym: id}, 'feats_by_sym': DataFrame per symbol,
       'basket_close': basket index series, 'horizon': horizon used
    """
    log.info("Building universe of %d symbols", len(symbols))

    # First pass: load each symbol's features and collect closes for basket
    feats_by_sym = {}
    for s in symbols:
        f = build_kline_features(s)
        if not f.empty:
            feats_by_sym[s] = f
    if len(feats_by_sym) < 5:
        raise RuntimeError(f"need ≥5 symbols, have {len(feats_by_sym)}")

    # Aligned closes for basket
    closes = pd.DataFrame({s: f["close"] for s, f in feats_by_sym.items()})
    closes = closes.sort_index()
    basket_ret, basket_close = build_basket(closes)

    # Second pass: enrich each symbol's features with basket-relative + flow + funding
    sym_to_id = {s: i for i, s in enumerate(sorted(feats_by_sym.keys()))}
    # Lazy import: only loaded when funding features are requested
    try:
        from features_ml.funding_features import add_funding_features
    except ImportError:
        add_funding_features = None
    enriched = {}
    for s, f in feats_by_sym.items():
        # Reindex to full universe time grid (handles symbols that started later)
        f = f.reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        if add_funding_features is not None:
            try:
                f = add_funding_features(f, s)
            except Exception as e:
                log.warning("[%s] funding features failed: %s", s, e)
        f["sym_id"] = sym_to_id[s]
        enriched[s] = f

    return {
        "symbols": sorted(feats_by_sym.keys()),
        "sym_id": sym_to_id,
        "feats_by_sym": enriched,
        "basket_close": basket_close,
        "basket_ret": basket_ret,
        "horizon": horizon,
    }


def make_xs_alpha_labels(
    feats_by_sym: dict, basket_close: pd.Series, horizon: int,
) -> dict:
    """Build per-symbol cross-sectional alpha labels.

    alpha_s[t] = my_fwd_s[t] - β_s × basket_fwd[t]

    where β_s is rolling 1d beta of my_ret to basket_ret (point-in-time).
    """
    bk_fwd = basket_close.pct_change(horizon).shift(-horizon)

    labels_by_sym = {}
    for s, f in feats_by_sym.items():
        my_close = f["close"]
        my_fwd = my_close.pct_change(horizon).shift(-horizon)
        # β already in features as 'beta_short_vs_bk' (already shifted by 1)
        beta = f["beta_short_vs_bk"]
        alpha = my_fwd - beta * bk_fwd
        rmean = alpha.expanding(min_periods=288).mean().shift(horizon)
        rstd = alpha.rolling(288 * 7, min_periods=288).std().shift(horizon)
        target = (alpha - rmean) / rstd.replace(0, np.nan)
        exit_time = my_close.index.to_series().shift(-horizon)
        labels_by_sym[s] = pd.DataFrame({
            "return_pct": my_fwd, "basket_fwd": bk_fwd,
            "alpha_realized": alpha, "demeaned_target": target,
            "exit_time": exit_time,
        })
    return labels_by_sym
