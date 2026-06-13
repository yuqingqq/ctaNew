"""Train V0 (WINNER_17), V1 (v3_5m augment), and V4 (target-horizon augment).
Extract LGBM feature importance to answer: do the new features actually
contribute predictive signal to the model?

V4 uses target-horizon features from the base panel that WINNER_17 doesn't use:
  idio_ret_to_btc_48b   (4h backward β-residual return — DIRECTLY matches target)
  idio_ret_to_btc_12b   (1h backward residual)
  btc_ret_48b           (4h BTC momentum)
  xs_alpha_dispersion_48b (4h cross-sectional dispersion)
  xs_alpha_mean_48b     (4h cross-sectional alpha mean)
  idio_vol_to_btc_1d    (1d idio vol — cleaner than W17's 1h version)
  xs_alpha_iqr_12b      (1h dispersion robustness)

These are at the SAME time scale as the prediction target (4h forward).
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

PANEL_BASE = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
PANEL_V3   = REPO / "outputs/vBTC_features_btc_v3/panel_v3_5m.parquet"
OUT_DIR    = REPO / "outputs/vBTC_v3_feature_importance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAR_PER_DAY = 288
BETA_WIN_PIT_DAYS = 90
RC = 0.50
THRESHOLD = 1 - RC

# WINNER_17
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

# v3_5m features (full set)
V3_FULL_19 = [
    "log_dollar_volume_7d", "volume_stability_30d", "amihud_illiq_30d",
    "beta_btc_30d", "beta_btc_90d", "corr_btc_30d", "corr_breakdown",
    "resid_vol_30d", "resid_vol_90d",
    "resid_skew_30d", "resid_kurt_30d",
    "resid_jump_count_30d", "resid_trend_score_30d",
    "dist_from_30d_high", "dist_from_365d_high", "multi_horizon_trend_score",
    "funding_mean_30d",
    "idio_skew_1d", "idio_max_abs_12b",
]

# NEW: target-horizon features from base panel (unused by WINNER_17)
V4_TARGET_HORIZON_7 = [
    "idio_ret_to_btc_48b",     # 4h backward β-residual — MATCHES target exactly
    "idio_ret_to_btc_12b",     # 1h backward residual
    "btc_ret_48b",             # 4h BTC momentum
    "xs_alpha_dispersion_48b", # 4h cross-sectional dispersion
    "xs_alpha_mean_48b",       # 4h cross-sectional mean alpha
    "idio_vol_to_btc_1d",      # 1d idio vol (cleaner than W17's 1h)
    "xs_alpha_iqr_12b",        # 1h IQR
]


def compute_pit_beta(panel, beta_win_days=90):
    btc_ret = panel[panel.symbol == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret"}).drop_duplicates("open_time")
    bar_window = beta_win_days * 288
    out = []
    for sym, g in panel.groupby("symbol"):
        gg = g[["open_time", "return_pct"]].merge(btc_ret, on="open_time", how="left")
        gg = gg.sort_values("open_time").reset_index(drop=True)
        if sym == "BTCUSDT":
            gg["beta_pit"] = 1.0
        else:
            y = gg["return_pct"]; x = gg["btc_ret"]
            cov = y.rolling(bar_window, min_periods=1000).cov(x)
            var = x.rolling(bar_window, min_periods=1000).var()
            gg["beta_pit"] = (cov / var.replace(0, np.nan)).shift(1)
        gg["symbol"] = sym
        out.append(gg)
    return pd.concat(out, ignore_index=True)[["symbol", "open_time", "beta_pit"]]


def train_fold_with_importance(train_data, cal_data, feat_set, seed=42):
    Xt = train_data[feat_set].to_numpy(np.float32)
    Xc = cal_data[feat_set].to_numpy(np.float32)
    yt = train_data["target_beta"].to_numpy(np.float32)
    yc = cal_data["target_beta"].to_numpy(np.float32)
    mt = ~np.isnan(yt); mc = ~np.isnan(yc)
    params = dict(
        objective="regression", metric="rmse", learning_rate=0.03,
        num_leaves=63, max_depth=8, min_data_in_leaf=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=3.0, verbose=-1,
        seed=seed, feature_fraction_seed=seed, bagging_seed=seed,
        data_random_seed=seed,
    )
    dtr = lgb.Dataset(Xt[mt], yt[mt], free_raw_data=False, feature_name=feat_set)
    dc  = lgb.Dataset(Xc[mc], yc[mc], reference=dtr, free_raw_data=False, feature_name=feat_set)
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dc],
                  callbacks=[lgb.early_stopping(stopping_rounds=80),
                             lgb.log_evaluation(period=0)])
    return m


def get_importance(model, feat_set):
    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    return pd.DataFrame({"feature": feat_set, "gain": gain, "split": split})


def prepare_panel(want_v3_feats, want_v4_feats):
    print("Loading base panel...", flush=True)
    panel = pd.read_parquet(PANEL_BASE)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    folds_all = _multi_oos_splits(panel)
    # Verify v4 features already in base
    missing_v4 = [f for f in want_v4_feats if f not in panel.columns]
    if missing_v4:
        print(f"  WARN: V4 features missing from base: {missing_v4}", flush=True)
    print(f"  base: {len(panel):,} rows × {len(panel.columns)} cols", flush=True)
    # Merge v3 features
    new_v3 = [f for f in want_v3_feats if f not in panel.columns]
    p3 = pd.read_parquet(PANEL_V3, columns=["symbol","open_time"] + new_v3)
    p3["open_time"] = pd.to_datetime(p3["open_time"], utc=True)
    panel = panel.merge(p3, on=["symbol","open_time"], how="left")
    print(f"  + v3 features merged: {len(panel.columns)} cols", flush=True)

    # β-residual
    t0 = time.time()
    print("  Computing PIT β...", flush=True)
    pit = compute_pit_beta(panel)
    panel = panel.merge(pit, on=["symbol","open_time"], how="left")
    btc = panel[panel.symbol=="BTCUSDT"][["open_time","return_pct"]].rename(
        columns={"return_pct":"btc_ret_t"}).drop_duplicates("open_time")
    panel = panel.merge(btc, on="open_time", how="left")
    panel["alpha_beta"] = panel["return_pct"] - panel["beta_pit"] * panel["btc_ret_t"]
    train0, _, _ = _slice(panel, folds_all[0])
    sigma = train0.groupby("symbol")["alpha_beta"].std().to_dict()
    fallback = panel["alpha_beta"].std()
    panel["sigma_idio_ref"] = panel["symbol"].map(sigma).fillna(fallback).clip(lower=1e-6)
    panel["target_beta"] = panel["alpha_beta"] / panel["sigma_idio_ref"]
    print(f"  done {time.time()-t0:.0f}s", flush=True)
    return panel, folds_all


def main():
    print("=== v3 feature importance + V4 target-horizon test ===\n", flush=True)
    panel, folds_all = prepare_panel(V3_FULL_19, V4_TARGET_HORIZON_7)

    variants = {
        "V0_WINNER_17":     WINNER_17,
        "V1_W17_plus_v3":   WINNER_17 + V3_FULL_19,
        "V4_W17_plus_targethorizon": WINNER_17 + V4_TARGET_HORIZON_7,
    }

    # Use fold 4 (the lift-driving fold per LOFO) for in-fold importance
    FOLD_IMPORTANCE = 4
    print(f"\nUsing fold {FOLD_IMPORTANCE} for importance extraction "
          f"(LOFO showed this fold drives V1 lift)\n", flush=True)
    train, cal, test = _slice(panel, folds_all[FOLD_IMPORTANCE])
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    print(f"  train: {len(tr):,}, cal: {len(ca):,}", flush=True)

    importance_results = {}
    for label, feat_set in variants.items():
        missing = [f for f in feat_set if f not in panel.columns]
        if missing:
            print(f"\n!! {label}: missing {missing}", flush=True)
            continue
        print(f"\n--- {label} ({len(feat_set)} features) ---", flush=True)
        t0 = time.time()
        model = train_fold_with_importance(tr, ca, feat_set, seed=42)
        imp = get_importance(model, feat_set)
        imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
        imp["gain_pct"] = imp["gain"] / imp["gain"].sum() * 100
        imp["cumulative_gain_pct"] = imp["gain_pct"].cumsum()
        importance_results[label] = imp
        print(f"  best_iter: {model.best_iteration}, train time: {time.time()-t0:.0f}s",
              flush=True)
        # Top 10
        print(f"  Top 10 features by gain:", flush=True)
        for _, row in imp.head(10).iterrows():
            tag = ""
            if row["feature"] in V3_FULL_19 and label != "V0_WINNER_17":
                tag = "  [v3_5m]"
            elif row["feature"] in V4_TARGET_HORIZON_7 and label != "V0_WINNER_17":
                tag = "  [v4_target]"
            print(f"    {row['feature']:<35} gain={row['gain']:>12.0f} "
                  f"({row['gain_pct']:>5.1f}%) splits={row['split']}{tag}", flush=True)
        imp.to_csv(OUT_DIR / f"{label}_importance.csv", index=False)

    # ======= Cross-variant comparison =======
    print(f"\n{'='*100}", flush=True)
    print(f"  v3 features TOTAL contribution in V1 (fold {FOLD_IMPORTANCE})", flush=True)
    print(f"{'='*100}", flush=True)
    if "V1_W17_plus_v3" in importance_results:
        v1 = importance_results["V1_W17_plus_v3"]
        v3_only = v1[v1["feature"].isin(V3_FULL_19)]
        w17_in_v1 = v1[~v1["feature"].isin(V3_FULL_19)]
        v3_total_gain_pct = v3_only["gain_pct"].sum()
        w17_total_gain_pct = w17_in_v1["gain_pct"].sum()
        print(f"  v3_5m features (19): total gain share = {v3_total_gain_pct:.1f}%", flush=True)
        print(f"  WINNER_17 features (17): total gain share = {w17_total_gain_pct:.1f}%", flush=True)
        print(f"  v3 average gain per feature: {v3_only['gain'].mean():,.0f}", flush=True)
        print(f"  w17 average gain per feature: {w17_in_v1['gain'].mean():,.0f}", flush=True)
        v3_used = (v3_only["split"] > 0).sum()
        print(f"  v3 features actually used (split>0): {v3_used}/{len(V3_FULL_19)}", flush=True)
    print(f"\n  v4 features TOTAL contribution in V4 (fold {FOLD_IMPORTANCE})", flush=True)
    if "V4_W17_plus_targethorizon" in importance_results:
        v4 = importance_results["V4_W17_plus_targethorizon"]
        v4_only = v4[v4["feature"].isin(V4_TARGET_HORIZON_7)]
        w17_in_v4 = v4[~v4["feature"].isin(V4_TARGET_HORIZON_7)]
        print(f"  v4_target features (7): total gain share = {v4_only['gain_pct'].sum():.1f}%",
              flush=True)
        print(f"  WINNER_17 features in V4 (17): total gain share = "
              f"{w17_in_v4['gain_pct'].sum():.1f}%", flush=True)
        print(f"  v4 average gain per feature: {v4_only['gain'].mean():,.0f}", flush=True)
        v4_used = (v4_only["split"] > 0).sum()
        print(f"  v4 features actually used (split>0): {v4_used}/{len(V4_TARGET_HORIZON_7)}",
              flush=True)


if __name__ == "__main__":
    main()
