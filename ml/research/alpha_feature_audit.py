"""Feature audit for alpha-residual prediction.

We've migrated the objective from RAW forward return to ALPHA RESIDUAL:
    alpha[t] = my_fwd[t] - β[t] × ref_fwd[t]

The current 18-feature set was chosen for raw-return prediction. Some features
that work for raw return may be redundant or noisy when the market component
is removed. Other features that capture symbol-specific information (cross-asset,
spread, idiosyncratic vol) may help more.

This audit measures, per symbol and per alpha definition:

  1. IC(feature, alpha_target)            for every feature in the candidate set
  2. IC(feature, raw_return_target)       for the same features (baseline)
  3. The "alpha lift" = IC_alpha - IC_raw — features that specifically help
     predict residual (not just direction of market).

Three alpha definitions are tested:
  - β window = 288   bars (1 day)   — current default
  - β window = 864   bars (3 days)
  - β window = 2016  bars (7 days)

Cross-asset feature additions are also included as candidates.

Output: a ranked feature table + recommended target / feature set.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ml.research.trend_pooled_v2 import _build_symbol_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HORIZON = 48
VOL_WIN = 288
HOLDOUT_DAYS = 90

# Candidate feature set — current 18 + a few extras worth testing.
BASE_FEATURES = [
    "atr_zscore_1d", "return_1d", "efficiency_96", "adx_15m",
    "bars_since_high", "atr_pct", "realized_vol_1h",
    "dist_resistance_20", "dist_resistance_50",
    "volume_ma_20", "volume_ma_50",
    "ema_slope_20_1h", "bb_squeeze_20", "vpin",
    "hour_cos", "hour_sin",
    "tfi_smooth", "signed_volume",
]

# Reference symbol per target (for alpha residual).
REF_OF = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT", "SOLUSDT": "BTCUSDT"}

BETA_WINDOWS = {"1d": 288, "3d": 864, "7d": 2016}


def _build_alpha_target(my_close: pd.Series, ref_close: pd.Series,
                         horizon: int, beta_window: int) -> dict:
    """Return raw return target, alpha residual, and the components.
    All series have the same index as my_close; demeaning uses the
    look-ahead-fixed shift(horizon)."""
    ref_close = ref_close.reindex(my_close.index).ffill()
    my_fwd = my_close.pct_change(horizon).shift(-horizon)
    ref_fwd = ref_close.pct_change(horizon).shift(-horizon)
    my_ret = my_close.pct_change()
    ref_ret = ref_close.pct_change()
    cov = (my_ret * ref_ret).rolling(beta_window).mean() - \
          my_ret.rolling(beta_window).mean() * ref_ret.rolling(beta_window).mean()
    var = ref_ret.rolling(beta_window).var().replace(0, np.nan)
    beta = (cov / var).clip(-3, 3).shift(1)
    alpha = my_fwd - beta * ref_fwd

    def _demean(signal):
        rmean = signal.expanding(min_periods=288).mean().shift(horizon)
        rstd = signal.rolling(VOL_WIN * 7, min_periods=VOL_WIN).std().shift(horizon)
        return (signal - rmean) / rstd.replace(0, np.nan)

    return {
        "raw_target": _demean(my_fwd),
        "alpha_target": _demean(alpha),
        "alpha_realized": alpha,
        "my_fwd": my_fwd,
        "ref_fwd": ref_fwd,
        "beta": beta,
    }


def _ic(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Pearson IC and Spearman IC, on overlapping non-NaN rows."""
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 100:
        return np.nan, np.nan
    pearson = df.iloc[:, 0].corr(df.iloc[:, 1])
    spearman = df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank())
    return pearson, spearman


def _audit_symbol(symbol: str, holdout_start: pd.Timestamp) -> pd.DataFrame:
    """For one symbol, compute IC of every feature vs (raw, alpha at 3 beta windows).
    Returns a long-format DataFrame: feature × target × {pearson,spearman}.

    Uses ONLY in-sample data (before holdout) so we don't leak holdout statistics.
    """
    feats, _ = _build_symbol_features(symbol)
    ref_feats, _ = _build_symbol_features(REF_OF[symbol])
    is_feats = feats[feats.index < holdout_start]

    targets = {}
    for label, bw in BETA_WINDOWS.items():
        comps = _build_alpha_target(is_feats["close"],
                                     ref_feats["close"][ref_feats.index < holdout_start],
                                     HORIZON, bw)
        if label == "1d":
            targets["raw"] = comps["raw_target"]
        targets[f"alpha_{label}"] = comps["alpha_target"]

    avail_feats = [c for c in BASE_FEATURES if c in is_feats.columns]
    rows = []
    for feat in avail_feats:
        x = is_feats[feat]
        for tgt_name, y in targets.items():
            p, s = _ic(x, y)
            rows.append({"symbol": symbol, "feature": feat,
                          "target": tgt_name, "pearson": p, "spearman": s})
    return pd.DataFrame(rows)


def _alpha_target_audit(symbol: str, holdout_start: pd.Timestamp) -> pd.DataFrame:
    """Audit the ALPHA TARGET itself (not features) at each beta window.
    Compares: target std (smaller = more concentrated alpha), correlation
    with raw return (lower = more orthogonal to market), serial autocorrelation
    (alpha persistence)."""
    feats, _ = _build_symbol_features(symbol)
    ref_feats, _ = _build_symbol_features(REF_OF[symbol])
    is_feats = feats[feats.index < holdout_start]

    rows = []
    for label, bw in BETA_WINDOWS.items():
        comps = _build_alpha_target(is_feats["close"],
                                     ref_feats["close"][ref_feats.index < holdout_start],
                                     HORIZON, bw)
        my_fwd = comps["my_fwd"].dropna()
        alpha = comps["alpha_realized"].dropna()
        beta = comps["beta"].dropna()
        # Variance reduction
        var_ratio = alpha.std() / my_fwd.std()
        # Correlation with raw return
        corr_raw = alpha.corr(my_fwd)
        # 1-bar autocorrelation of alpha (signal persistence)
        ac1 = alpha.autocorr(lag=1)
        # 12-bar autocorrelation
        ac12 = alpha.autocorr(lag=12)
        rows.append({
            "symbol": symbol, "beta_window": label, "beta_bars": bw,
            "my_fwd_std_bps": my_fwd.std() * 1e4,
            "alpha_std_bps": alpha.std() * 1e4,
            "var_ratio": var_ratio,
            "corr_alpha_vs_raw": corr_raw,
            "beta_mean": beta.mean(),
            "beta_std": beta.std(),
            "alpha_autocorr_1": ac1,
            "alpha_autocorr_12": ac12,
        })
    return pd.DataFrame(rows)


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    feats0, _ = _build_symbol_features(symbols[0])
    data_end = feats0.index.max()
    holdout_start = data_end - pd.Timedelta(days=HOLDOUT_DAYS)
    log.info("data_end=%s, holdout_start=%s", data_end, holdout_start)

    print("=" * 70)
    print("ALPHA TARGET AUDIT — comparing β windows 1d / 3d / 7d")
    print("=" * 70)
    target_audits = []
    for s in symbols:
        df = _alpha_target_audit(s, holdout_start)
        target_audits.append(df)
    target_df = pd.concat(target_audits, ignore_index=True)
    print(target_df.round(4).to_string(index=False))

    print("\n--- Recommended β window: ---")
    # Pick window minimizing var_ratio (most variance removed) and
    # |corr_alpha_vs_raw| (most orthogonal to market).
    score = target_df.groupby("beta_window").agg(
        avg_var_ratio=("var_ratio", "mean"),
        avg_abs_corr_raw=("corr_alpha_vs_raw", lambda x: x.abs().mean()),
        avg_alpha_ac1=("alpha_autocorr_1", "mean"),
    )
    print(score.round(4).to_string())
    best = score["avg_var_ratio"].idxmin()
    print(f"\nLowest variance ratio (most market-stripped): β={best}")

    print("\n" + "=" * 70)
    print("FEATURE IC AUDIT (in-sample only, before holdout)")
    print("=" * 70)
    feat_audits = []
    for s in symbols:
        df = _audit_symbol(s, holdout_start)
        feat_audits.append(df)
    feat_df = pd.concat(feat_audits, ignore_index=True)

    # Pivot: rows = (symbol, feature), cols = target, values = spearman
    pivot = feat_df.pivot_table(index=["symbol", "feature"],
                                  columns="target", values="spearman").reset_index()

    # Compute lift: alpha_1d - raw, alpha_3d - raw, alpha_7d - raw
    for k in BETA_WINDOWS:
        col = f"alpha_{k}"
        if col in pivot.columns and "raw" in pivot.columns:
            pivot[f"lift_{k}"] = pivot[col] - pivot["raw"]

    print("\n--- IC vs alpha target (per symbol, Spearman) ---")
    for s in symbols:
        sub = pivot[pivot["symbol"] == s].set_index("feature").drop(columns="symbol")
        # Sort by alpha_1d magnitude
        sub = sub.reindex(sub["alpha_1d"].abs().sort_values(ascending=False).index)
        print(f"\n{s}:")
        print(sub.round(4).to_string())

    print("\n--- Top alpha-specific features (avg |lift| across symbols, β=1d) ---")
    avg_lift = pivot.groupby("feature").agg(
        alpha_1d_mean=("alpha_1d", "mean"),
        raw_mean=("raw", "mean"),
        lift_1d_mean=("lift_1d", "mean"),
        lift_1d_abs_mean=("lift_1d", lambda x: x.abs().mean()),
    ).round(4).sort_values("lift_1d_abs_mean", ascending=False)
    print(avg_lift.to_string())

    # Save artifacts
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    target_df.to_csv(out_dir / "alpha_target_audit.csv", index=False)
    feat_df.to_csv(out_dir / "alpha_feature_audit.csv", index=False)
    pivot.to_csv(out_dir / "alpha_feature_audit_pivot.csv", index=False)
    print(f"\nSaved audit CSVs to {out_dir}/")


if __name__ == "__main__":
    main()
