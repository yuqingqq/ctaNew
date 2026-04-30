"""Trend strategy v2: pooled training + Sharpe target + CROSS-ASSET features
+ disk-cached features (kills the 60-min rebuild cost).

Builds on the validated config D from `trend_pooled_sharpe.py`:
    - Pooled training (BTC + ETH + SOL)
    - Sharpe-like target (vol-normalized)
    - 5-seed ensemble
    - autocorr_1h regime gate (cutoff 0.33), q=0.95 trigger, h=48 hold
    - vol-scaled position sizing

NEW:
    - Cross-asset features (excess returns, beta, correlation, dominance spread)
    - Per-symbol feature cache to data/ml/cache/  (avoids rebuilding from raw aggTrades)

Two configs compared:
    D (baseline): config from trend_pooled_sharpe.py (pooled + Sharpe, no cross-asset)
    E (with cross-asset): D + cross-asset features
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features_ml.cross_asset import add_cross_asset_features
from features_ml.klines import compute_kline_features
from features_ml.regime_features import add_regime_features
from features_ml.trade_flow import TradeFlowConfig, aggregate_trades_streaming
from ml.cost_model import CostConfig, effective_spread_roll
from ml.cv import FoldSpec, fold_iter, make_walk_forward_folds, split_features_by_fold

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data/ml/test/parquet")
CACHE_DIR = Path("data/ml/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_TR_FEATURES = [
    "atr_zscore_1d", "return_1d", "efficiency_96", "adx_15m",
    "bars_since_high", "atr_pct", "realized_vol_1h",
    "dist_resistance_20", "dist_resistance_50",
    "volume_ma_20", "volume_ma_50",
    "ema_slope_20_1h", "bb_squeeze_20", "vpin",
    "hour_cos", "hour_sin",
    "tfi_smooth", "signed_volume",
]
ENSEMBLE_SEEDS = (42, 7, 123, 99, 314)
THRESHOLD_Q = 0.95
HORIZON = 48
REGIME_CUTOFF = 0.33
VOL_NORMALIZATION_WINDOW = 288


def _build_symbol_features(symbol: str, force_rebuild: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Build features + spread for one symbol with disk caching."""
    cache_feat = CACHE_DIR / f"feats_{symbol}.parquet"
    cache_spread = CACHE_DIR / f"spread_{symbol}.parquet"
    if not force_rebuild and cache_feat.exists() and cache_spread.exists():
        log.info("[%s] loading from cache", symbol)
        feats = pd.read_parquet(cache_feat)
        spread = pd.read_parquet(cache_spread)["spread_bps"]
        return feats, spread

    log.info("[%s] building features (no cache)", symbol)
    paths = sorted((DATA_DIR / f"klines/{symbol}/5m").glob("*.parquet"))
    klines = pd.concat([pd.read_parquet(p) for p in paths]).sort_values("open_time").set_index("open_time")
    feats = compute_kline_features(klines)
    flow = aggregate_trades_streaming(
        sorted((DATA_DIR / f"aggTrades/{symbol}").glob("*.parquet")),
        TradeFlowConfig(bar_interval="5min", compute_kyle_lambda=False),
    ).drop(columns=["vwap", "last_price"], errors="ignore")
    feats = feats.join(flow, how="inner")
    feats = add_regime_features(feats)

    ret = feats["close"].pct_change()
    feats["autocorr_1h"] = ret.rolling(36).apply(lambda s: s.autocorr(lag=1) if s.std() > 0 else 0.0)
    feats["autocorr_pctile_7d"] = (
        feats["autocorr_1h"].rolling(2016, min_periods=288).rank(pct=True).shift(1)
    )

    chunks = []
    for path in sorted((DATA_DIR / f"aggTrades/{symbol}").glob("*.parquet")):
        day = pd.read_parquet(path)
        prices = day.set_index("transact_time")["price"]
        chunks.append(effective_spread_roll(prices, bar_interval="5min"))
        del day
    sp = pd.concat(chunks).sort_index()
    sp = sp[~sp.index.duplicated(keep="last")]
    aligned = pd.DataFrame({"sp": sp, "close": feats["close"]}).dropna()
    spread_bps = (1e4 * aligned["sp"] / aligned["close"]).rename("spread_bps")

    feats.to_parquet(cache_feat, compression="zstd")
    spread_bps.to_frame().to_parquet(cache_spread, compression="zstd")
    log.info("[%s] cached features + spread", symbol)
    return feats, spread_bps


def _make_labels_sharpe(feats, horizon):
    fwd = feats["close"].pct_change(horizon).shift(-horizon)
    exit_time = feats.index.to_series().shift(-horizon)
    rolling_mean = fwd.expanding(min_periods=288).mean().shift(1)
    rolling_std = fwd.rolling(VOL_NORMALIZATION_WINDOW * 7,
                              min_periods=VOL_NORMALIZATION_WINDOW).std().shift(1)
    target = (fwd - rolling_mean) / rolling_std.replace(0, np.nan)
    return pd.DataFrame({
        "return_pct": fwd, "demeaned_target": target, "exit_time": exit_time,
    }).dropna()


def _train_lgbm(X_train, y_train, X_cal, y_cal, *, seed):
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


def _run(symbols: list, with_cross_asset: bool):
    """Run pooled+Sharpe trend strategy, optionally with cross-asset features."""
    cost = CostConfig(flat_slippage_bps=1.0)
    fee = cost.fee_taker * 2; slip = 2 * cost.flat_slippage_bps / 1e4

    sym_data = {s: _build_symbol_features(s) for s in symbols}

    # Add cross-asset features
    feature_cols = list(BASE_TR_FEATURES)
    if with_cross_asset:
        # Pair: each symbol references one other. BTC ↔ ETH, SOL → BTC.
        cross_refs = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT", "SOLUSDT": "BTCUSDT"}
        for s in symbols:
            ref = cross_refs.get(s)
            if ref is None or ref not in sym_data: continue
            feats_s, _ = sym_data[s]
            feats_ref, _ = sym_data[ref]
            ref_label = ref[:3].lower()
            updated = add_cross_asset_features(feats_s, feats_ref, ref_label=ref_label)
            sym_data[s] = (updated, sym_data[s][1])
        # New columns added (use first symbol to discover them)
        first_sym = symbols[0]
        ref_first = cross_refs.get(first_sym, "ETHUSDT")[:3].lower()
        cross_cols = [
            f"excess_ret_3_vs_{ref_first}", f"excess_ret_12_vs_{ref_first}", f"excess_ret_48_vs_{ref_first}",
            f"beta_{ref_first}_1d", f"corr_{ref_first}_1d",
            f"spread_log_vs_{ref_first}",
            f"spread_zscore_1d_vs_{ref_first}", f"spread_zscore_7d_vs_{ref_first}",
        ]
        # Note: each symbol may have a different ref suffix. We add all possible cols
        # to feature_cols and let dropna handle missing per-symbol.
        # But for the LGBM train, we need a unified column set. Easier approach:
        # rename per-symbol cross features to a generic name (`*_vs_ref`) before pooling.
        for s in symbols:
            feats_s, sp = sym_data[s]
            ref = cross_refs.get(s); ref_label = ref[:3].lower() if ref else None
            if ref_label is None: continue
            rename_map = {}
            for col in [f"excess_ret_3_vs_{ref_label}", f"excess_ret_12_vs_{ref_label}",
                         f"excess_ret_48_vs_{ref_label}",
                         f"beta_{ref_label}_1d", f"corr_{ref_label}_1d",
                         f"spread_log_vs_{ref_label}",
                         f"spread_zscore_1d_vs_{ref_label}", f"spread_zscore_7d_vs_{ref_label}"]:
                rename_map[col] = col.replace(f"_vs_{ref_label}", "_vs_ref").replace(f"_{ref_label}_", "_ref_")
            feats_s = feats_s.rename(columns=rename_map)
            sym_data[s] = (feats_s, sp)
        cross_cols_generic = [
            "excess_ret_3_vs_ref", "excess_ret_12_vs_ref", "excess_ret_48_vs_ref",
            "beta_ref_1d", "corr_ref_1d",
            "spread_log_vs_ref",
            "spread_zscore_1d_vs_ref", "spread_zscore_7d_vs_ref",
        ]
        feature_cols.extend(cross_cols_generic)

    log.info("feature_cols (%d): %s", len(feature_cols), feature_cols[-8:] if with_cross_asset else "[base only]")

    # Folds based on first symbol's date range
    feats0 = sym_data[symbols[0]][0]
    folds = make_walk_forward_folds(
        data_start=feats0.index.min(), data_end=feats0.index.max(),
        n_folds=5, train_days=50, cal_days=10, test_days=20, embargo_days=1.0,
    )
    cols_avail = [c for c in feature_cols if c in feats0.columns]

    # Pre-compute labels per symbol
    labels_by_sym = {s: _make_labels_sharpe(sym_data[s][0], HORIZON) for s in symbols}

    fold_rows = []
    for fold in folds:
        # Pool train+cal
        train_dfs, cal_dfs = [], []
        for s in symbols:
            feats_s, _ = sym_data[s]
            splits = _expanding_train(feats_s, labels_by_sym[s], fold)
            train_dfs.append(splits["train"])
            cal_dfs.append(splits["cal"])
        train = pd.concat(train_dfs, ignore_index=True)
        cal = pd.concat(cal_dfs, ignore_index=True)
        train = train.dropna(subset=cols_avail + ["demeaned_target", "return_pct"])
        cal = cal.dropna(subset=cols_avail + ["demeaned_target", "return_pct"])
        train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
        cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
        if len(train_f) < 200 or len(cal_f) < 50: continue

        cal_preds, models = [], []
        for seed in ENSEMBLE_SEEDS:
            m = _train_lgbm(train_f[cols_avail].to_numpy(),
                             train_f["demeaned_target"].to_numpy(),
                             cal_f[cols_avail].to_numpy(),
                             cal_f["demeaned_target"].to_numpy(), seed=seed)
            models.append(m)
            cal_preds.append(m.predict(cal_f[cols_avail].to_numpy(), num_iteration=m.best_iteration))
        yc = np.mean(cal_preds, axis=0)
        thr = float(np.quantile(np.abs(yc), THRESHOLD_Q))

        for s in symbols:
            feats_s, spread_bps = sym_data[s]
            splits = _expanding_train(feats_s, labels_by_sym[s], fold)
            test = splits["test"].dropna(subset=cols_avail + ["demeaned_target", "return_pct", "atr_pct"])
            test_f = test[test["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
            if len(test_f) < 20: continue
            preds = [m.predict(test_f[cols_avail].to_numpy(), num_iteration=m.best_iteration) for m in models]
            yt = np.mean(preds, axis=0)
            triggered = np.abs(yt) >= thr
            n = int(triggered.sum())
            if n == 0: continue

            side = np.sign(yt[triggered]); side[side == 0] = 1
            atr = np.clip(test_f["atr_pct"].to_numpy()[triggered], 1e-4, 1e-1)
            inv_vol = np.clip((1.0 / atr) / (1.0 / atr).mean(), 0.3, 3.0)
            gross_raw = side * test_f["return_pct"].to_numpy()[triggered]
            gross = gross_raw * inv_vol
            idx = test_f.index[triggered]
            sp_e = spread_bps.reindex(idx).fillna(0.0).to_numpy()
            sp_x = spread_bps.reindex(test_f["exit_time"].iloc[triggered]).fillna(0.0).to_numpy()
            spread_term = 0.5 * (sp_e + sp_x) / 1e4
            cost_per_trade = (fee + slip + spread_term) * inv_vol
            net = gross - cost_per_trade
            fold_rows.append({
                "fold": fold.fold_id, "symbol": s, "n": n,
                "win_rate": float((net > 0).mean()),
                "gross_bps": float(gross.mean() * 1e4),
                "net_bps": float(net.mean() * 1e4),
            })
    return pd.DataFrame(fold_rows)


def _summary(df, label):
    if df.empty:
        print(f"\n=== {label} === EMPTY"); return None
    print(f"\n=== {label} ===")
    print(df.round(2).to_string(index=False))
    print(f"\n  total: n={int(df['n'].sum())}, mean_net={df['net_bps'].mean():+.2f} bps, "
          f"std={df['net_bps'].std():.2f}, folds_pos={int((df['net_bps']>0).sum())}/{len(df)}")
    if df["symbol"].nunique() > 1:
        print("\n  by symbol:")
        for s, sub in df.groupby("symbol"):
            print(f"    {s}: n={int(sub['n'].sum())}, mean_net={sub['net_bps'].mean():+.2f}, "
                  f"folds_pos={int((sub['net_bps']>0).sum())}/{len(sub)}")
    return df


def main() -> None:
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    print("\n" + "=" * 70)
    print("CONFIG D (baseline): pooled + Sharpe target, NO cross-asset features")
    print("=" * 70)
    df_d = _run(symbols, with_cross_asset=False)
    _summary(df_d, "D baseline")

    print("\n" + "=" * 70)
    print("CONFIG E: D + cross-asset features (excess returns, beta, dominance)")
    print("=" * 70)
    df_e = _run(symbols, with_cross_asset=True)
    _summary(df_e, "E with cross-asset")

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD")
    print("=" * 70)
    rows = []
    for label, df in [("D: pooled/sharpe", df_d), ("E: D + cross-asset", df_e)]:
        if df.empty: continue
        rows.append({
            "config": label,
            "total_trades": int(df["n"].sum()),
            "mean_net_bps": df["net_bps"].mean(),
            "std_fold_net": df["net_bps"].std(),
            "folds_pos": int((df["net_bps"] > 0).sum()),
            "n_folds": len(df),
        })
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # Per-symbol detail
    print("\n--- per symbol mean net bps ---")
    for label, df in [("D", df_d), ("E", df_e)]:
        if df.empty: continue
        by_sym = df.groupby("symbol").agg(
            mean_net=("net_bps", "mean"), folds_pos=("net_bps", lambda x: (x > 0).sum()),
            n_total=("n", "sum"),
        ).round(2)
        print(f"\n  Config {label}:")
        print(by_sym.to_string())


if __name__ == "__main__":
    main()
