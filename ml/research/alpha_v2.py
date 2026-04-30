"""Alpha-residual model v2: optimized feature set + symbol indicator.

Changes from v1 (trend_alpha_residual.py):

Drop weak features (|IC vs alpha| < 0.01 across all 3 symbols):
  - tfi_smooth, signed_volume, efficiency_96, adx_15m

Keep base features:
  - atr_zscore_1d, return_1d, atr_pct, realized_vol_1h
  - bars_since_high, dist_resistance_20, dist_resistance_50
  - volume_ma_20, volume_ma_50
  - ema_slope_20_1h, bb_squeeze_20, vpin
  - hour_cos, hour_sin

Add cross-asset features (top alpha-IC + lift from audit):
  - spread_log_vs_ref         (-0.07 to -0.10 IC vs alpha across all 3 symbols)
  - beta_ref_1d               (+lift on ETH and SOL)
  - ref_ema_slope_20_1h       (+0.05 IC for ETH, +0.02 for SOL — strong lift)
  - ref_return_1d             (similar to above)

Add symbol indicator:
  - sym_id ∈ {0, 1, 2} for BTC, ETH, SOL — lets the tree split per-symbol when
    feature signs differ (audit showed ema_slope_20_1h is +0.03 IC on ETH alpha
    but -0.08 on BTC alpha — a single global tree without symbol context can't
    capture both).

Two head-to-head probes:
  A. v1 baseline      = current 18 features, pooled, alpha target
  B. v2 optimized     = refined feature set + cross-asset + symbol indicator

For each: walk-forward and OOS holdout, with per-symbol decomposition.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features_ml.cross_asset import add_cross_asset_features
from ml.cost_model import CostConfig
from ml.cv import FoldSpec, make_walk_forward_folds, split_features_by_fold
from ml.research.trend_pooled_v2 import _build_symbol_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# === Feature sets ===
V1_FEATURES = [
    "atr_zscore_1d", "return_1d", "efficiency_96", "adx_15m",
    "bars_since_high", "atr_pct", "realized_vol_1h",
    "dist_resistance_20", "dist_resistance_50",
    "volume_ma_20", "volume_ma_50",
    "ema_slope_20_1h", "bb_squeeze_20", "vpin",
    "hour_cos", "hour_sin",
    "tfi_smooth", "signed_volume",
]

V2_BASE = [
    "atr_zscore_1d", "return_1d", "atr_pct", "realized_vol_1h",
    "bars_since_high", "dist_resistance_20", "dist_resistance_50",
    "volume_ma_20", "volume_ma_50",
    "ema_slope_20_1h", "bb_squeeze_20", "vpin",
    "hour_cos", "hour_sin",
]

# Cross-asset additions added per symbol with the symbol's reference label.
# Generic names below; we'll resolve per-symbol below using REF_OF.
V2_CROSS_TEMPLATES = [
    "spread_log_vs_{ref}",
    "beta_{ref}_1d",
    "{ref}_ema_slope_20_1h",
    "{ref}_return_1d",
]
SYM_INDICATOR = "sym_id"

REF_OF = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT", "SOLUSDT": "BTCUSDT"}
SYM_TO_ID = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
REF_BORROW_FEATS = ["ema_slope_20_1h", "return_1d"]

ENSEMBLE_SEEDS = (42, 7, 123, 99, 314)
THRESHOLD_Q = 0.95
HORIZON = 48
REGIME_CUTOFF = 0.33
VOL_WIN = 288
HOLDOUT_DAYS = 90
BETA_WINDOW = 288


def _make_alpha_label(my_feats, ref_feats, horizon):
    my_close = my_feats["close"]
    ref_close = ref_feats["close"].reindex(my_close.index).ffill()
    my_fwd = my_close.pct_change(horizon).shift(-horizon)
    ref_fwd = ref_close.pct_change(horizon).shift(-horizon)
    exit_time = my_close.index.to_series().shift(-horizon)

    my_ret = my_close.pct_change()
    ref_ret = ref_close.pct_change()
    cov = (my_ret * ref_ret).rolling(BETA_WINDOW).mean() - \
          my_ret.rolling(BETA_WINDOW).mean() * ref_ret.rolling(BETA_WINDOW).mean()
    var = ref_ret.rolling(BETA_WINDOW).var().replace(0, np.nan)
    beta = (cov / var).clip(-3, 3).shift(1)
    alpha = my_fwd - beta * ref_fwd

    rmean = alpha.expanding(min_periods=288).mean().shift(horizon)
    rstd = alpha.rolling(VOL_WIN * 7, min_periods=VOL_WIN).std().shift(horizon)
    target = (alpha - rmean) / rstd.replace(0, np.nan)

    return pd.DataFrame({
        "return_pct": my_fwd, "ref_fwd": ref_fwd, "beta": beta,
        "alpha_realized": alpha, "demeaned_target": target, "exit_time": exit_time,
    }).dropna(subset=["demeaned_target", "return_pct", "exit_time"])


def _enrich_features(symbol: str) -> tuple[pd.DataFrame, pd.Series, str]:
    """Build per-symbol features with cross-asset extensions and a symbol indicator."""
    feats, spread = _build_symbol_features(symbol)
    ref_symbol = REF_OF[symbol]
    ref_feats, _ = _build_symbol_features(ref_symbol)
    ref_label = ref_symbol[:3].lower()
    enriched = add_cross_asset_features(feats, ref_feats, ref_label=ref_label)
    # Borrow ref features
    borrowed = ref_feats[REF_BORROW_FEATS].reindex(enriched.index)
    borrowed.columns = [f"{ref_label}_{c}" for c in borrowed.columns]
    enriched = enriched.join(borrowed)
    enriched[SYM_INDICATOR] = SYM_TO_ID[symbol]
    return enriched, spread, ref_label


def _v2_columns(symbol: str) -> list[str]:
    """Resolve the per-symbol v2 feature column names."""
    ref_label = REF_OF[symbol][:3].lower()
    cross_cols = [tpl.format(ref=ref_label) for tpl in V2_CROSS_TEMPLATES]
    return V2_BASE + cross_cols + [SYM_INDICATOR]


def _train(X_train, y_train, X_cal, y_cal, *, seed):
    params = dict(
        objective="regression", metric="rmse", learning_rate=0.03,
        num_leaves=63, max_depth=8, min_data_in_leaf=50,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=3.0, verbose=-1,
        seed=seed, feature_fraction_seed=seed, bagging_seed=seed,
        data_random_seed=seed,
    )
    dtr = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dc = lgb.Dataset(X_cal, y_cal, reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dc],
                     callbacks=[lgb.early_stopping(stopping_rounds=50),
                                lgb.log_evaluation(period=0)])


def _expanding_train(feats, labels, fold: FoldSpec):
    base = split_features_by_fold(feats, labels, fold)
    test_left = fold.test_start - fold.embargo
    joined = feats.join(labels, how="inner")
    train = joined[joined.index < fold.cal_start]
    if "exit_time" in train.columns:
        overlap = (train["exit_time"] >= test_left) & (train.index < fold.test_end + fold.embargo)
        train = train.loc[~overlap]
    base["train"] = train
    return base


def _harmonized_columns(symbols: list, version: str) -> list[str]:
    """Each symbol's enriched features include `*_vs_eth` (for BTC) or
    `*_vs_btc` (for ETH/SOL). To pool, we rename per-symbol cross columns
    to canonical names: e.g. `spread_log_vs_btc` and `spread_log_vs_eth`
    both become `spread_log_vs_ref`. Returns the canonical column list."""
    if version == "v1":
        return V1_FEATURES
    elif version == "v2":
        return V2_BASE + [tpl.format(ref="ref") for tpl in V2_CROSS_TEMPLATES] + [SYM_INDICATOR]
    raise ValueError(version)


def _canonicalize(feats: pd.DataFrame, symbol: str, version: str) -> pd.DataFrame:
    """Rename per-symbol cross columns to `*_vs_ref` and `ref_*` so all symbols
    share the same column names for pooling."""
    if version == "v1":
        return feats
    ref_label = REF_OF[symbol][:3].lower()
    rename = {}
    for tpl in V2_CROSS_TEMPLATES:
        src = tpl.format(ref=ref_label)
        dst = tpl.format(ref="ref")
        rename[src] = dst
    return feats.rename(columns=rename)


def _run(symbols: list, version: str, mode: str):
    """version = 'v1' or 'v2'; mode = 'walkforward' or 'oos_holdout'."""
    cost = CostConfig(flat_slippage_bps=1.0)
    fee = cost.fee_taker * 2; slip = 2 * cost.flat_slippage_bps / 1e4

    sym_data = {}
    spread_data = {}
    for s in symbols:
        if version == "v1":
            feats, sp = _build_symbol_features(s)
            feats = feats.copy()  # no enrichment
        else:
            feats, sp, _ = _enrich_features(s)
        feats = _canonicalize(feats, s, version)
        ref_feats, _ = _build_symbol_features(REF_OF[s])
        labels = _make_alpha_label(feats, ref_feats, HORIZON)
        sym_data[s] = (feats, labels)
        spread_data[s] = sp

    cols = _harmonized_columns(symbols, version)

    feats0 = sym_data[symbols[0]][0]
    if mode == "walkforward":
        folds = make_walk_forward_folds(
            data_start=feats0.index.min(), data_end=feats0.index.max(),
            n_folds=5, train_days=50, cal_days=10, test_days=20, embargo_days=1.0,
        )
    elif mode == "oos_holdout":
        data_end = feats0.index.max()
        holdout_start = data_end - pd.Timedelta(days=HOLDOUT_DAYS)
        cal_start = holdout_start - pd.Timedelta(days=11)
        cal_end = cal_start + pd.Timedelta(days=10)
        folds = [FoldSpec(
            train_start=feats0.index.min(), train_end=cal_start,
            cal_start=cal_start, cal_end=cal_end,
            test_start=holdout_start, test_end=data_end,
            embargo=pd.Timedelta(days=1), fold_id=0,
        )]
    else:
        raise ValueError(mode)

    fold_rows = []
    for fold in folds:
        train_dfs, cal_dfs = [], []
        for s in symbols:
            feats_s, labels_s = sym_data[s]
            splits = _expanding_train(feats_s, labels_s, fold)
            train_dfs.append(splits["train"])
            cal_dfs.append(splits["cal"])
        train = pd.concat(train_dfs, ignore_index=True).dropna(
            subset=cols + ["demeaned_target", "return_pct"])
        cal = pd.concat(cal_dfs, ignore_index=True).dropna(
            subset=cols + ["demeaned_target", "return_pct"])
        train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
        cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
        if len(train_f) < 200 or len(cal_f) < 50: continue

        cal_preds, models = [], []
        for seed in ENSEMBLE_SEEDS:
            m = _train(train_f[cols].to_numpy(),
                        train_f["demeaned_target"].to_numpy(),
                        cal_f[cols].to_numpy(),
                        cal_f["demeaned_target"].to_numpy(), seed=seed)
            models.append(m)
            cal_preds.append(m.predict(cal_f[cols].to_numpy(), num_iteration=m.best_iteration))
        yc = np.mean(cal_preds, axis=0)
        thr = float(np.quantile(np.abs(yc), THRESHOLD_Q))

        for s in symbols:
            feats_s, labels_s = sym_data[s]
            splits = _expanding_train(feats_s, labels_s, fold)
            test = splits["test"].dropna(
                subset=cols + ["demeaned_target", "return_pct", "atr_pct"])
            test_f = test[test["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
            if len(test_f) < 20: continue

            preds = [m.predict(test_f[cols].to_numpy(), num_iteration=m.best_iteration) for m in models]
            yt = np.mean(preds, axis=0)
            triggered = np.abs(yt) >= thr
            n = int(triggered.sum())
            if n == 0: continue

            side = np.sign(yt[triggered]); side[side == 0] = 1
            atr = np.clip(test_f["atr_pct"].to_numpy()[triggered], 1e-4, 1e-1)
            inv_vol = np.clip((1.0 / atr) / (1.0 / atr).mean(), 0.3, 3.0)
            gross = side * test_f["return_pct"].to_numpy()[triggered] * inv_vol
            alpha_t = test_f["alpha_realized"].to_numpy()[triggered]
            alpha_pnl = side * alpha_t * inv_vol
            ref_fwd_t = test_f["ref_fwd"].to_numpy()[triggered]
            beta_t = test_f["beta"].to_numpy()[triggered]
            market_pnl = side * beta_t * ref_fwd_t * inv_vol

            sp = spread_data[s]
            idx = test_f.index[triggered]
            sp_e = sp.reindex(idx).fillna(0.0).to_numpy()
            sp_x = sp.reindex(test_f["exit_time"].iloc[triggered]).fillna(0.0).to_numpy()
            spread_term = 0.5 * (sp_e + sp_x) / 1e4
            cost_per_trade = (fee + slip + spread_term) * inv_vol
            net = gross - cost_per_trade

            fold_rows.append({
                "fold": fold.fold_id, "symbol": s, "n": n,
                "win_rate": float((net > 0).mean()),
                "alpha_pnl_bps": float(alpha_pnl.mean() * 1e4),
                "market_pnl_bps": float(market_pnl.mean() * 1e4),
                "gross_bps": float(gross.mean() * 1e4),
                "net_bps": float(net.mean() * 1e4),
            })

    df = pd.DataFrame(fold_rows)
    return df, models, cols


def _summary(df, label):
    print(f"\n=== {label} ===")
    if df.empty:
        print("  EMPTY"); return
    print(f"  total: n={int(df['n'].sum())}, "
          f"alpha={df['alpha_pnl_bps'].mean():+.2f}, "
          f"market={df['market_pnl_bps'].mean():+.2f}, "
          f"net={df['net_bps'].mean():+.2f} bps, "
          f"folds_pos={int((df['net_bps']>0).sum())}/{len(df)}")
    if df["symbol"].nunique() > 1:
        for s, sub in df.groupby("symbol"):
            print(f"    {s}: n={int(sub['n'].sum())}, "
                  f"alpha={sub['alpha_pnl_bps'].mean():+.2f}, "
                  f"market={sub['market_pnl_bps'].mean():+.2f}, "
                  f"net={sub['net_bps'].mean():+.2f}, "
                  f"folds_pos={int((sub['net_bps']>0).sum())}/{len(sub)}")


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    print("=" * 70)
    print("ALPHA-RESIDUAL HEAD-TO-HEAD: v1 (18 features) vs v2 (refined + cross-asset)")
    print("=" * 70)

    print("\n--- WALK-FORWARD ---")
    df_v1_wf, _, _ = _run(symbols, version="v1", mode="walkforward")
    _summary(df_v1_wf, "WF, v1 baseline")
    df_v2_wf, _, _ = _run(symbols, version="v2", mode="walkforward")
    _summary(df_v2_wf, "WF, v2 optimized")

    print("\n--- OOS HOLDOUT ---")
    df_v1_oos, _, _ = _run(symbols, version="v1", mode="oos_holdout")
    _summary(df_v1_oos, "OOS, v1 baseline")
    df_v2_oos, models_v2, cols_v2 = _run(symbols, version="v2", mode="oos_holdout")
    _summary(df_v2_oos, "OOS, v2 optimized")

    # v2 feature importance on OOS
    if models_v2:
        gains = np.mean([m.feature_importance(importance_type="gain") for m in models_v2], axis=0)
        gain_share = gains / gains.sum()
        imp = pd.Series(dict(zip(cols_v2, gain_share))).sort_values(ascending=False)
        print("\n--- v2 feature importance (OOS gain share) ---")
        for f, v in imp.items():
            print(f"  {f:<26}: {v*100:5.2f}%")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    rows = []
    for label, df in [("WF v1", df_v1_wf), ("WF v2", df_v2_wf),
                       ("OOS v1", df_v1_oos), ("OOS v2", df_v2_oos)]:
        if df.empty:
            rows.append({"config": label, "n": 0,
                          "alpha_bps": np.nan, "market_bps": np.nan, "net_bps": np.nan,
                          "folds_pos": 0, "n_folds": 0})
        else:
            rows.append({
                "config": label, "n": int(df["n"].sum()),
                "alpha_bps": df["alpha_pnl_bps"].mean(),
                "market_bps": df["market_pnl_bps"].mean(),
                "net_bps": df["net_bps"].mean(),
                "folds_pos": int((df["net_bps"] > 0).sum()),
                "n_folds": len(df),
            })
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
