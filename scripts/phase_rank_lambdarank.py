"""Phase RANK: replace MSE with LambdaRank loss.

Train LGBM with `objective="lambdarank"` where:
  - group = cross-section per timestamp (all symbols at one open_time)
  - label = quintile of target_A within that cross-section (0-4)
  - lambdarank_truncation_level = 3 (matches K=3 production picks)

Same WINNER_21 features, same other hyperparameters, same 5-seed ensemble.

Then rebuild sleeves and run V3.1 + 6-gate validation against current V3.1.

Why this might lift:
  - MSE rewards predicting target_A magnitude → spends model capacity on
    matching scale across heterogeneous symbols
  - LambdaRank rewards within-cycle ordering, especially of top-3 positions
  - Strategy uses ranks only → objective alignment should help
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

OUT = REPO / "outputs/vBTC_phase_RANK"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 48
SEEDS = (42, 1337, 7, 19, 2718)
TARGET_COL = "target_A"
MIN_HISTORY_DAYS = 60
RC = 0.50
THRESHOLD = 1 - RC
N_QUANTILES = 5  # quintile labels (0-4)
TRUNC_LEVEL = 3  # NDCG@3 — match production K=3

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
WINNER_21 = [f for f in V6_CLEAN_28 if f not in ALL_DROPS] + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING


def _train_lambdarank(X_train, y_train, group_train, X_cal, y_cal, group_cal,
                          *, seed):
    """Train LGBM with LambdaRank objective.

    y_*: integer relevance labels (0 to N_QUANTILES-1)
    group_*: array of group sizes (rows per cross-section)
    """
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[TRUNC_LEVEL],
        learning_rate=0.03,
        num_leaves=63, max_depth=8, min_data_in_leaf=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=3.0, verbose=-1,
        lambdarank_truncation_level=TRUNC_LEVEL,
        seed=seed, feature_fraction_seed=seed, bagging_seed=seed,
        data_random_seed=seed,
    )
    dtr = lgb.Dataset(X_train, label=y_train, group=group_train,
                       free_raw_data=False)
    dc = lgb.Dataset(X_cal, label=y_cal, group=group_cal,
                       reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dc],
                       callbacks=[lgb.early_stopping(stopping_rounds=80),
                                    lgb.log_evaluation(period=0)])


def prepare_group_data(df, feat_set):
    """Sort by open_time, compute group sizes, and convert target_A → quintile labels.

    Returns: X, y, group, valid_mask, df_sorted.
    """
    df = df.copy().sort_values(["open_time", "symbol"]).reset_index(drop=True)
    # quintile labels within each cross-section
    df["y_lr"] = df.groupby("open_time")[TARGET_COL].transform(
        lambda x: pd.qcut(x, N_QUANTILES, labels=False, duplicates="drop")
                    if x.notna().sum() >= N_QUANTILES else np.nan
    )
    # drop rows with NaN target or feature
    valid_mask = df["y_lr"].notna()
    for f in feat_set:
        valid_mask &= df[f].notna()
    df = df[valid_mask].reset_index(drop=True)
    # Re-derive groups after filtering
    group_sizes = df.groupby("open_time").size().to_numpy()
    X = df[feat_set].to_numpy(np.float32)
    y = df["y_lr"].to_numpy(np.int32)
    return X, y, group_sizes, df


def get_listings():
    listings = {}
    klines_dir = REPO / "data/ml/test/parquet/klines"
    for sym_dir in klines_dir.iterdir():
        if not sym_dir.is_dir(): continue
        m5 = sym_dir / "5m"
        if not m5.exists(): continue
        files = sorted(m5.glob("*.parquet"))
        if not files: continue
        try:
            ts = pd.Timestamp(files[0].stem, tz="UTC")
            listings[sym_dir.name] = ts
        except Exception:
            continue
    return listings


def train_fold_lambdarank(panel, fold, feat_set, eligible_syms):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) &
                 (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) &
                (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100:
        return None

    Xt, yt, gt, _ = prepare_group_data(tr, feat_set)
    Xc, yc, gc, _ = prepare_group_data(ca, feat_set)

    preds_all = []
    test_r_sorted = test_r.sort_values(["open_time", "symbol"]).reset_index(drop=True)
    Xtest = test_r_sorted[feat_set].to_numpy(np.float32)

    for s in SEEDS:
        m = _train_lambdarank(Xt, yt, gt, Xc, yc, gc, seed=s)
        preds_all.append(m.predict(Xtest, num_iteration=m.best_iteration))
    pred_cols = ["symbol", "open_time", "alpha_A"]
    if "exit_time" in test_r_sorted.columns:
        pred_cols.append("exit_time")
    df_f = test_r_sorted[pred_cols].copy()
    df_f["pred"] = np.mean(preds_all, axis=0)
    df_f["fold"] = fold["fid"]
    return df_f


def main():
    print("=== Phase RANK: LambdaRank retrain ===\n", flush=True)
    print(f"  Pre-registered: objective=lambdarank, NDCG@{TRUNC_LEVEL}, "
          f"labels = quintile within cross-section", flush=True)
    print(f"  Same WINNER_21 features, same hyperparameters otherwise.\n", flush=True)

    # Load panel
    print("  loading panel...", flush=True)
    t0 = time.time()
    panel = pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    print(f"  panel {len(panel):,} rows ({time.time()-t0:.0f}s)", flush=True)

    feat_set = [f for f in WINNER_21 if f in panel.columns]
    print(f"  features: {len(feat_set)}", flush=True)

    listings = get_listings()
    panel_first_obs = panel.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            ts = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[sym] = ts

    folds = _multi_oos_splits(panel)
    print(f"  total folds: {len(folds)}\n", flush=True)

    all_preds = []
    t0_total = time.time()
    for fold in folds:
        t0 = time.time()
        cutoff = fold["cal_start"] - pd.Timedelta(days=MIN_HISTORY_DAYS)
        eligible = {s for s in panel["symbol"].unique()
                      if listings.get(s) and listings[s] <= cutoff}
        df_f = train_fold_lambdarank(panel, fold, feat_set, eligible)
        if df_f is None:
            print(f"  fold {fold['fid']}: SKIP", flush=True)
            continue
        all_preds.append(df_f)
        print(f"  fold {fold['fid']}: {len(df_f):,} preds  "
              f"(elapsed={time.time()-t0:.0f}s, total={time.time()-t0_total:.0f}s)",
              flush=True)

    all_pred = pd.concat(all_preds, ignore_index=True)
    if "exit_time" not in all_pred.columns:
        all_pred["exit_time"] = all_pred["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
    out_path = OUT / "all_predictions_lambdarank.parquet"
    all_pred.to_parquet(out_path, index=False)
    print(f"\n  saved {len(all_pred):,} predictions to {out_path}", flush=True)
    print(f"  total time: {time.time()-t0_total:.0f}s", flush=True)

    # Quick IC diagnostic
    print(f"\n  Quick per-cycle IC check (LambdaRank vs WINNER_21 MSE baseline):",
          flush=True)
    base = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet")
    base = base.dropna(subset=["pred", "alpha_A"])
    base = base[base["fold"].isin(range(1, 10))]
    all_pred_oos = all_pred[all_pred["fold"].isin(range(1, 10))].dropna(
        subset=["pred", "alpha_A"])

    base_ics = []
    for t, g in base.groupby("open_time"):
        if len(g) < 10: continue
        ic = g["pred"].rank().corr(g["alpha_A"].rank())
        if not pd.isna(ic): base_ics.append(ic)
    lr_ics = []
    for t, g in all_pred_oos.groupby("open_time"):
        if len(g) < 10: continue
        ic = g["pred"].rank().corr(g["alpha_A"].rank())
        if not pd.isna(ic): lr_ics.append(ic)

    base_ics = np.array(base_ics); lr_ics = np.array(lr_ics)
    print(f"    Baseline (MSE):    mean IC = {base_ics.mean():+.4f}  "
          f"median = {np.median(base_ics):+.4f}  pct_pos = {(base_ics>0).mean()*100:.1f}%",
          flush=True)
    print(f"    LambdaRank:        mean IC = {lr_ics.mean():+.4f}  "
          f"median = {np.median(lr_ics):+.4f}  pct_pos = {(lr_ics>0).mean()*100:.1f}%",
          flush=True)
    print(f"    Δ mean IC: {lr_ics.mean() - base_ics.mean():+.4f}", flush=True)


if __name__ == "__main__":
    main()
