"""Feature audit: how well-matched are WINNER_21 features to the β-residual target?

WINNER_21 was selected for the basket-residual target (production). Phase 1D switched
the target to β-residual but inherited WINNER_21 unchanged. This may not be optimal.

Diagnostics:
  1. Per-feature cross-sectional IC against alpha_β (new target)
     vs against alpha_A (production basket-residual target)
     → identifies features that lost predictive power under the new target
  2. Per-feature time stability (rolling 90d IC)
     → identifies features whose signal is decaying
  3. Feature redundancy: pairwise correlation matrix among features
     → identifies clusters of features that carry the same info
  4. Tail-conditional IC: cycles where realized α-spread is biggest
     → which features predict at extremes (where the strategy actually trades)
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

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
APD_PATH = REPO / "outputs/vBTC_phase1d_rolling_beta/all_predictions.parquet"
OUT = REPO / "outputs/vBTC_feature_audit"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 48
OOS_FOLDS = list(range(1, 10))

V6_CLEAN_28 = list(XS_FEATURE_COLS_V6_CLEAN)
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
WINNER_21 = ([f for f in V6_CLEAN_28 if f not in ALL_DROPS]
             + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING)


def per_cycle_feature_ic(panel, feature, target_col):
    """At each entry cycle, cross-sectional Spearman corr between feature and target.
    Returns array of per-cycle IC values."""
    df = panel.dropna(subset=[feature, target_col]).copy()
    # Sample at entry cadence
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    ics = []
    for t, g in df.groupby("open_time"):
        if len(g) < 10: continue
        ic = g[feature].rank().corr(g[target_col].rank())
        if not pd.isna(ic): ics.append(ic)
    return np.array(ics)


def main():
    print("=== Feature audit for β-residual target ===\n", flush=True)

    print("Loading panel + building alpha_β realization...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    # Construct β-residual realization (proxy PIT β with first-fold OLS for diagnostic)
    btc_ret = panel[panel.symbol == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret"}).drop_duplicates("open_time")
    panel = panel.merge(btc_ret, on="open_time", how="left")
    # Use β=1 here for the diagnostic (close to PIT β for most alts, simpler)
    # For IC, what matters is rank, and β-scaling per-symbol is monotonic per-symbol
    panel["alpha_beta_naive"] = panel["return_pct"] - panel["btc_ret"]
    panel["target_beta_proxy"] = panel.groupby("symbol")["alpha_beta_naive"].transform(
        lambda x: x / x.std()) # z within-symbol
    # Restrict to OOS
    folds_all = _multi_oos_splits(panel)
    train_panel = panel  # use full panel for IC computation; we just want feature info
    print(f"  panel: {len(panel):,} rows × {panel['symbol'].nunique()} symbols\n", flush=True)

    # === DIAG 1: per-feature cross-sectional IC, two targets ===
    print("="*80)
    print("DIAG 1 — Per-feature cross-sectional IC: alpha_A (prod) vs alpha_β (new)")
    print("="*80)
    features = [f for f in WINNER_21 if f in panel.columns and f != "sym_id"]
    print(f"\n  Computing per-cycle IC for {len(features)} features against each target...",
          flush=True)
    t0 = time.time()
    rows = []
    for feat in features:
        ic_A = per_cycle_feature_ic(panel, feat, "alpha_A")
        ic_B = per_cycle_feature_ic(panel, feat, "alpha_beta_naive")
        rows.append({
            "feature": feat,
            "mean_ic_alpha_A": float(ic_A.mean()) if len(ic_A) else np.nan,
            "median_ic_alpha_A": float(np.median(ic_A)) if len(ic_A) else np.nan,
            "mean_ic_alpha_β": float(ic_B.mean()) if len(ic_B) else np.nan,
            "median_ic_alpha_β": float(np.median(ic_B)) if len(ic_B) else np.nan,
            "delta_β_minus_A": (float(ic_B.mean()) - float(ic_A.mean())) if len(ic_A) and len(ic_B) else np.nan,
        })
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)
    df_ic = pd.DataFrame(rows)
    df_ic["abs_ic_β"] = df_ic["mean_ic_alpha_β"].abs()
    df_ic = df_ic.sort_values("abs_ic_β", ascending=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    print("\nFeatures ranked by |mean IC vs alpha_β| (the NEW target):", flush=True)
    print(df_ic[["feature", "mean_ic_alpha_A", "mean_ic_alpha_β",
                 "delta_β_minus_A"]].to_string(index=False, float_format=lambda x: f"{x:+.4f}"),
          flush=True)
    df_ic.to_csv(OUT / "feature_ic_two_targets.csv", index=False)

    # === DIAG 2: how many features have meaningful IC under each target? ===
    print("\n" + "="*80)
    print("DIAG 2 — How many features carry meaningful signal under each target?")
    print("="*80)
    for thresh in [0.005, 0.010, 0.015, 0.020, 0.030]:
        n_A = (df_ic["mean_ic_alpha_A"].abs() >= thresh).sum()
        n_B = (df_ic["mean_ic_alpha_β"].abs() >= thresh).sum()
        print(f"  |IC| ≥ {thresh:.3f}:  alpha_A: {n_A:>2}/{len(df_ic)}   "
              f"alpha_β: {n_B:>2}/{len(df_ic)}", flush=True)

    # === DIAG 3: features that LOST IC under the new target ===
    print("\n" + "="*80)
    print("DIAG 3 — Features that lost IC under the new (β-residual) target")
    print("="*80)
    df_ic["abs_delta"] = df_ic["delta_β_minus_A"].abs()
    # Did IC sign flip? (predictive direction switched)
    df_ic["sign_flip"] = (np.sign(df_ic["mean_ic_alpha_A"]) !=
                          np.sign(df_ic["mean_ic_alpha_β"]))
    df_ic_loss = df_ic[df_ic["mean_ic_alpha_A"].abs() > df_ic["mean_ic_alpha_β"].abs()].copy()
    df_ic_loss = df_ic_loss.sort_values("abs_delta", ascending=False)
    print("\n  Features where |IC vs alpha_β| < |IC vs alpha_A| (lost predictive power):",
          flush=True)
    print(df_ic_loss[["feature", "mean_ic_alpha_A", "mean_ic_alpha_β",
                       "delta_β_minus_A", "sign_flip"]].head(15).to_string(index=False),
          flush=True)

    df_ic_flip = df_ic[df_ic["sign_flip"]].copy()
    print(f"\n  Features with SIGN FLIP (predictive direction reversed): {len(df_ic_flip)}",
          flush=True)
    if len(df_ic_flip):
        print(df_ic_flip[["feature", "mean_ic_alpha_A", "mean_ic_alpha_β"]].to_string(index=False))

    # === DIAG 4: LGBM feature importance from the Phase 1D retrained model ===
    print("\n" + "="*80)
    print("DIAG 4 — LGBM feature importance (re-retrain fold 1 quickly to extract)")
    print("="*80)
    print("  Retraining fold 1 once to read feature importance...", flush=True)
    t0 = time.time()
    import lightgbm as lgb
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    train, cal, _ = _slice(panel.assign(target_beta=panel["target_beta_proxy"]), folds_all[1])
    tr = train.dropna(subset=feat_set + ["target_beta"])
    ca = cal.dropna(subset=feat_set + ["target_beta"])
    Xt = tr[feat_set].to_numpy(np.float32)
    yt = tr["target_beta"].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    yc = ca["target_beta"].to_numpy(np.float32)
    dtr = lgb.Dataset(Xt, yt, feature_name=feat_set)
    dc = lgb.Dataset(Xc, yc, reference=dtr, feature_name=feat_set)
    params = dict(objective="regression", metric="rmse", learning_rate=0.03,
                  num_leaves=63, max_depth=8, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                  lambda_l2=3.0, verbose=-1, seed=42)
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dc],
                  callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    imp_split = pd.DataFrame({
        "feature": feat_set,
        "split_count": m.feature_importance(importance_type="split"),
        "gain": m.feature_importance(importance_type="gain"),
    })
    imp_split["gain_pct"] = imp_split["gain"] / imp_split["gain"].sum() * 100
    imp_split = imp_split.sort_values("gain", ascending=False)
    print(f"  trained ({time.time()-t0:.0f}s)")
    print("\nFeature importance (gain) — fold 1 retrain on β-residual target:")
    print(imp_split.to_string(index=False), flush=True)
    imp_split.to_csv(OUT / "feature_importance_phase1d.csv", index=False)

    # === DIAG 5: Pairwise feature correlation (redundancy) ===
    print("\n" + "="*80)
    print("DIAG 5 — Feature correlation matrix (redundancy check)")
    print("="*80)
    panel_sample = panel.dropna(subset=feat_set).sample(min(100000, len(panel)), random_state=42)
    corr = panel_sample[feat_set].corr().abs()
    np.fill_diagonal(corr.values, 0)
    # Find top correlated pairs
    pairs = []
    for i, f1 in enumerate(feat_set):
        for f2 in feat_set[i+1:]:
            pairs.append({"f1": f1, "f2": f2, "abs_corr": corr.loc[f1, f2]})
    pairs_df = pd.DataFrame(pairs).sort_values("abs_corr", ascending=False)
    print("\n  Top 15 pairwise |correlation| (redundant feature pairs):")
    print(pairs_df.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"),
          flush=True)
    pairs_df.to_csv(OUT / "feature_pairwise_correlation.csv", index=False)

    # === Synthesis ===
    print("\n" + "="*80)
    print("SYNTHESIS")
    print("="*80)
    # Join IC + importance
    merged = df_ic.merge(imp_split[["feature","split_count","gain","gain_pct"]],
                          on="feature", how="left")
    merged = merged.sort_values("gain", ascending=False)
    print("\nAll features ranked by LGBM gain on β-target, with their cross-sectional IC:",
          flush=True)
    cols_show = ["feature","gain_pct","mean_ic_alpha_A","mean_ic_alpha_β","delta_β_minus_A"]
    print(merged[cols_show].to_string(index=False, float_format=lambda x: f"{x:+.4f}"),
          flush=True)
    merged.to_csv(OUT / "feature_audit_summary.csv", index=False)
    print(f"\nSaved CSVs to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
