"""Step 2: Prepare features for the linear (Ridge) model.

WINNER_17 minus sym_id = 16 numeric features. sym_id one-hot encoded to 50
columns (drop reference). Total = 66 input features for Ridge.

Preprocessing per feature (computed from fold-0 training data, then applied
to all data):
  1. Winsorize at fold-0 [1, 99] percentile bounds.
  2. Z-score using fold-0 train mean/std (so coefs are scale-comparable).
  3. NaN → 0 after z-scoring (= mean substitution; Ridge can't handle NaN).

PIT discipline:
  - All winsorize bounds + z-score params are from fold-0 TRAINING rows only.
  - Same bounds applied to cal/test rows — they may exceed bounds but get clipped.

Output:
  data/features.parquet         processed X matrix + symbol + open_time
  data/feature_stats.csv        per-feature mean/std/winsorize bounds for inspection
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

PANEL_BASE = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
TARGETS    = REPO / "linear_model/data/targets.parquet"
OUT_DIR    = REPO / "linear_model/data"

# WINNER_17 reproduction
ALL_DROPS = [
    "return_1d_xs_rank", "bk_ret_48b", "volume_ma_50",
    "ema_slope_20_1h", "ema_slope_20_1h_xs_rank",
    "vwap_zscore_xs_rank", "vwap_zscore",
    "atr_pct_xs_rank", "dom_z_7d_vs_bk", "obv_z_1d_xs_rank",
    "obv_signal", "price_volume_corr_10",
    "hour_cos", "hour_sin",
]
FUNDING_LEAN = ["funding_rate", "funding_rate_z_7d"]
ADD_CROSS_BTC = ["corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]
ADD_MORE_FUNDING = ["funding_rate_1d_change", "funding_streak_pos"]
WINNER_21 = ([f for f in XS_FEATURE_COLS_V6_CLEAN if f not in ALL_DROPS]
             + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING)
DEAD_WEIGHT = {"mfi", "price_volume_corr_20", "idio_ret_48b_vs_bk", "funding_streak_pos"}
WINNER_17 = [f for f in WINNER_21 if f not in DEAD_WEIGHT]

# sym_id is categorical — drop from numeric set, will one-hot separately
NUMERIC_FEATS = [f for f in WINNER_17 if f != "sym_id"]
print(f"Numeric features ({len(NUMERIC_FEATS)}): {NUMERIC_FEATS}")


def main():
    print("\n=== Step 2: Build standardized feature matrix ===\n", flush=True)
    t0 = time.time()

    # Load targets to inherit row alignment (BTC already dropped)
    print("  Loading target file (BTC already dropped)...", flush=True)
    targets = pd.read_parquet(TARGETS, columns=["symbol", "open_time"])
    targets["open_time"] = pd.to_datetime(targets["open_time"], utc=True)
    print(f"    target rows: {len(targets):,} × {targets.symbol.nunique()} symbols",
          flush=True)

    # Load base panel features
    print("\n  Loading base panel features...", flush=True)
    cols = ["symbol", "open_time"] + WINNER_17
    base = pd.read_parquet(PANEL_BASE, columns=cols)
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    # Drop BTC (matches target alignment)
    base = base[base["symbol"] != "BTCUSDT"].copy()
    print(f"    base rows: {len(base):,}", flush=True)

    # Inner-join to targets to ensure exact row alignment
    panel = targets.merge(base, on=["symbol", "open_time"], how="left")
    print(f"    after merge: {len(panel):,} rows × {len(WINNER_17)+2} cols", flush=True)
    if panel[NUMERIC_FEATS].isna().sum().sum() > 0:
        nan_per_feat = panel[NUMERIC_FEATS].isna().sum()
        print(f"    NaN counts per feature:", flush=True)
        for f, n in nan_per_feat.items():
            if n > 0:
                print(f"      {f:<32}: {n:,} ({n/len(panel)*100:.1f}%)", flush=True)

    # ----- Fold-0 training stats -----
    print("\n  Computing fold-0 training stats for winsorize + z-score...", flush=True)
    folds_all = _multi_oos_splits(panel)
    train0, _, _ = _slice(panel, folds_all[0])
    print(f"    fold-0 train: {len(train0):,} rows "
          f"({train0.open_time.min()} → {train0.open_time.max()})", flush=True)

    stats_rows = []
    for f in NUMERIC_FEATS:
        s = train0[f].dropna()
        p1 = float(s.quantile(0.01))
        p99 = float(s.quantile(0.99))
        # Stats post-winsorize (so z-score uses cleaned data)
        s_w = s.clip(lower=p1, upper=p99)
        mu = float(s_w.mean())
        sd = float(s_w.std())
        stats_rows.append({"feature": f, "p1": p1, "p99": p99,
                           "mean_post_winsor": mu, "std_post_winsor": sd,
                           "raw_min": float(s.min()), "raw_max": float(s.max()),
                           "n_train": len(s)})
    feat_stats = pd.DataFrame(stats_rows).set_index("feature")
    feat_stats.to_csv(OUT_DIR / "feature_stats.csv")
    print(f"    fold-0 stats saved: {OUT_DIR / 'feature_stats.csv'}", flush=True)
    print(f"\n  per-feature winsorize bounds + std (sample):", flush=True)
    print(f"    {'feature':<32} {'p1':>10} {'p99':>10} {'std_pw':>10}", flush=True)
    for f, row in feat_stats.iterrows():
        print(f"    {f:<32} {row['p1']:>+10.4f} {row['p99']:>+10.4f} "
              f"{row['std_post_winsor']:>10.4f}", flush=True)

    # ----- Apply winsorize + z-score to ALL rows -----
    print("\n  Applying winsorize + z-score to full panel...", flush=True)
    X = pd.DataFrame({"symbol": panel["symbol"], "open_time": panel["open_time"]})
    for f in NUMERIC_FEATS:
        p1 = feat_stats.loc[f, "p1"]
        p99 = feat_stats.loc[f, "p99"]
        mu = feat_stats.loc[f, "mean_post_winsor"]
        sd = feat_stats.loc[f, "std_post_winsor"]
        v = panel[f].clip(lower=p1, upper=p99)
        v = (v - mu) / (sd if sd > 0 else 1.0)
        X[f] = v.astype("float32")

    # ----- One-hot encode sym_id -----
    print("\n  One-hot encoding sym_id (50 dummies, drop AAVEUSDT as reference)...",
          flush=True)
    sym_dummies = pd.get_dummies(panel["symbol"], prefix="sym", drop_first=True,
                                  dtype="float32")
    print(f"    sym_dummies shape: {sym_dummies.shape}", flush=True)
    X = pd.concat([X, sym_dummies], axis=1)

    # ----- NaN → 0 (mean substitution, since features are z-scored) -----
    feat_cols = [c for c in X.columns if c not in ("symbol", "open_time")]
    nan_before = X[feat_cols].isna().sum().sum()
    X[feat_cols] = X[feat_cols].fillna(0)
    nan_after = X[feat_cols].isna().sum().sum()
    print(f"\n  NaN handling: {nan_before:,} → {nan_after} (mean-imputed via fill 0)",
          flush=True)

    # ----- Post-transform sanity check on training data -----
    print("\n  Post-transform sanity check on fold-0 training rows:", flush=True)
    train_mask = (panel["open_time"] >= train0.open_time.min()) & \
                 (panel["open_time"] <= train0.open_time.max())
    X_train = X.loc[train_mask, NUMERIC_FEATS]
    print(f"    {'feature':<32} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}", flush=True)
    for f in NUMERIC_FEATS[:10]:
        col = X_train[f]
        print(f"    {f:<32} {col.mean():>+8.4f} {col.std():>+8.4f} "
              f"{col.min():>+8.2f} {col.max():>+8.2f}", flush=True)
    print(f"    ... ({len(NUMERIC_FEATS)-10} more features)", flush=True)
    print(f"\n  Overall: mean of feature means = {X_train.mean().mean():+.6f} (should be ~0)",
          flush=True)
    print(f"           mean of feature stds  = {X_train.std().mean():.4f} (should be ~1)",
          flush=True)

    # Save
    X.to_parquet(OUT_DIR / "features.parquet", index=False)
    print(f"\n  Saved: {OUT_DIR / 'features.parquet'}", flush=True)
    print(f"    shape: {len(X):,} rows × {len(X.columns)} cols "
          f"({len(NUMERIC_FEATS)} numeric + {len(sym_dummies.columns)} sym dummies)",
          flush=True)
    print(f"\n  Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
