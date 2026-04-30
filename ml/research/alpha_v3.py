"""Alpha-residual model v3: alpha-tailored features.

Curated feature set chosen by the alpha-targeted IC audit (alpha_v3_audit):

Cross-asset alpha-tailored (9):
  - dom_level_vs_ref           — log dominance, sign-consistent across all 3 symbols (avg |IC| 0.065)
  - dom_z_7d_vs_ref            — long-term dominance z-score
  - dom_change_288b_vs_ref     — 1-day dominance momentum
  - ref_ret_48b                — reference's 4h return (lead-lag)
  - ref_ema_slope_4h           — reference's 4h EMA slope
  - idio_vol_1d_vs_ref         — idiosyncratic vol regime
  - idio_ret_48b_vs_ref        — past 4h idiosyncratic return (sign-consistent mean-reversion)
  - corr_change_3d_vs_ref      — correlation regime shift
  - beta_short_vs_ref          — current beta level (point-in-time)

Base symbol-own (8 — top IC vs alpha from earlier audit):
  - return_1d, ema_slope_20_1h          (strong for BTC alpha)
  - bars_since_high                      (consistent direction across symbols)
  - atr_pct, volume_ma_50, vpin
  - hour_cos, hour_sin

Symbol indicator (1):
  - sym_id

Total: 18 input columns. Same count as v1, but every feature was selected
by an audit on the alpha target rather than the raw-return target.

Drops vs v1: efficiency_96, adx_15m, tfi_smooth, signed_volume,
              dist_resistance_20, dist_resistance_50, atr_zscore_1d,
              bb_squeeze_20, realized_vol_1h, volume_ma_20

Drops vs v2: many overlapping spread/cross features replaced with the
              audit-validated alpha-tailored ones.

Compares: v1 (baseline), v2 (audit-informed cross-asset), v3 (alpha-tailored).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features_ml.alpha_features import add_alpha_features
from features_ml.cross_asset import add_cross_asset_features
from ml.cost_model import CostConfig
from ml.cv import FoldSpec, make_walk_forward_folds, split_features_by_fold
from ml.research.trend_pooled_v2 import _build_symbol_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# === Feature templates (using {ref} placeholder for the per-symbol reference) ===
V1_FEATURES = [
    "atr_zscore_1d", "return_1d", "efficiency_96", "adx_15m",
    "bars_since_high", "atr_pct", "realized_vol_1h",
    "dist_resistance_20", "dist_resistance_50",
    "volume_ma_20", "volume_ma_50",
    "ema_slope_20_1h", "bb_squeeze_20", "vpin",
    "hour_cos", "hour_sin",
    "tfi_smooth", "signed_volume",
]

V3_BASE = [
    "return_1d", "ema_slope_20_1h", "bars_since_high",
    "atr_pct", "volume_ma_50", "vpin",
    "hour_cos", "hour_sin",
]
V3_CROSS_TEMPLATES = [
    "dom_level_vs_{ref}",
    "dom_z_7d_vs_{ref}",
    "dom_change_288b_vs_{ref}",
    "{ref}_ret_48b",
    "{ref}_ema_slope_4h",
    "idio_vol_1d_vs_{ref}",
    "idio_ret_48b_vs_{ref}",
    "corr_change_3d_vs_{ref}",
    "beta_short_vs_{ref}",
]
SYM_INDICATOR = "sym_id"

REF_OF = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT", "SOLUSDT": "BTCUSDT"}
SYM_TO_ID = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}

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


def _enrich(symbol: str, version: str) -> tuple[pd.DataFrame, pd.Series]:
    feats, sp = _build_symbol_features(symbol)
    if version == "v1":
        return feats, sp
    ref_symbol = REF_OF[symbol]
    ref_feats, _ = _build_symbol_features(ref_symbol)
    ref_label = ref_symbol[:3].lower()
    if version == "v2":
        enriched = add_cross_asset_features(feats, ref_feats, ref_label=ref_label)
        # Borrow ref features
        for col in ("ema_slope_20_1h", "return_1d"):
            enriched[f"{ref_label}_{col}"] = ref_feats[col].reindex(enriched.index)
    elif version == "v3":
        enriched = add_alpha_features(feats, ref_feats, ref_label=ref_label)
    else:
        raise ValueError(version)
    enriched[SYM_INDICATOR] = SYM_TO_ID[symbol]
    return enriched, sp


def _canonicalize(feats: pd.DataFrame, symbol: str, version: str) -> pd.DataFrame:
    """Rename per-symbol cross columns (with concrete ref label) to canonical
    `{ref}` placeholders so all symbols share column names for pooling."""
    if version == "v1":
        return feats
    ref_label = REF_OF[symbol][:3].lower()
    rename = {}
    if version == "v2":
        for tpl in ["spread_log_vs_{ref}", "beta_{ref}_1d",
                     "{ref}_ema_slope_20_1h", "{ref}_return_1d"]:
            src = tpl.format(ref=ref_label); dst = tpl.format(ref="ref")
            rename[src] = dst
    elif version == "v3":
        for tpl in V3_CROSS_TEMPLATES:
            src = tpl.format(ref=ref_label); dst = tpl.format(ref="ref")
            rename[src] = dst
    return feats.rename(columns=rename)


def _harmonized_columns(version: str) -> list[str]:
    if version == "v1":
        return V1_FEATURES
    elif version == "v2":
        v2_base = [
            "atr_zscore_1d", "return_1d", "atr_pct", "realized_vol_1h",
            "bars_since_high", "dist_resistance_20", "dist_resistance_50",
            "volume_ma_20", "volume_ma_50",
            "ema_slope_20_1h", "bb_squeeze_20", "vpin",
            "hour_cos", "hour_sin",
        ]
        cross = ["spread_log_vs_ref", "beta_ref_1d", "ref_ema_slope_20_1h", "ref_return_1d"]
        return v2_base + cross + [SYM_INDICATOR]
    elif version == "v3":
        cross = [t.format(ref="ref") for t in V3_CROSS_TEMPLATES]
        return V3_BASE + cross + [SYM_INDICATOR]
    raise ValueError(version)


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


def _run(symbols: list, version: str, mode: str):
    cost = CostConfig(flat_slippage_bps=1.0)
    fee = cost.fee_taker * 2; slip = 2 * cost.flat_slippage_bps / 1e4

    sym_data = {}
    spread_data = {}
    for s in symbols:
        feats, sp = _enrich(s, version)
        feats = _canonicalize(feats, s, version)
        ref_feats, _ = _build_symbol_features(REF_OF[s])
        labels = _make_alpha_label(feats, ref_feats, HORIZON)
        sym_data[s] = (feats, labels)
        spread_data[s] = sp

    cols = _harmonized_columns(version)
    feats0 = sym_data[symbols[0]][0]

    if mode == "walkforward":
        folds = make_walk_forward_folds(
            data_start=feats0.index.min(), data_end=feats0.index.max(),
            n_folds=5, train_days=50, cal_days=10, test_days=20, embargo_days=1.0,
        )
    else:
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

    fold_rows = []; last_models = []; last_cols = cols
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
        last_models = models
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
                "ic_pred_alpha": float(np.corrcoef(yt, test_f["alpha_realized"].to_numpy())[0, 1]),
            })

    df = pd.DataFrame(fold_rows)
    return df, last_models, last_cols


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
                  f"ic={sub['ic_pred_alpha'].mean():+.4f}, "
                  f"folds_pos={int((sub['net_bps']>0).sum())}/{len(sub)}")


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    print("=" * 70)
    print("ALPHA-RESIDUAL HEAD-TO-HEAD: v1 / v2 / v3 (alpha-tailored)")
    print("=" * 70)

    print("\n--- WALK-FORWARD ---")
    df_v1_wf, _, _ = _run(symbols, version="v1", mode="walkforward")
    _summary(df_v1_wf, "WF, v1 baseline")
    df_v2_wf, _, _ = _run(symbols, version="v2", mode="walkforward")
    _summary(df_v2_wf, "WF, v2 cross-asset")
    df_v3_wf, _, _ = _run(symbols, version="v3", mode="walkforward")
    _summary(df_v3_wf, "WF, v3 alpha-tailored")

    print("\n--- OOS HOLDOUT ---")
    df_v1_oos, _, _ = _run(symbols, version="v1", mode="oos_holdout")
    _summary(df_v1_oos, "OOS, v1")
    df_v2_oos, _, _ = _run(symbols, version="v2", mode="oos_holdout")
    _summary(df_v2_oos, "OOS, v2")
    df_v3_oos, models_v3, cols_v3 = _run(symbols, version="v3", mode="oos_holdout")
    _summary(df_v3_oos, "OOS, v3")

    if models_v3:
        gains = np.mean([m.feature_importance(importance_type="gain") for m in models_v3], axis=0)
        gain_share = gains / gains.sum()
        imp = pd.Series(dict(zip(cols_v3, gain_share))).sort_values(ascending=False)
        print("\n--- v3 feature importance (OOS gain share) ---")
        for f, v in imp.items():
            print(f"  {f:<28}: {v*100:5.2f}%")

    print("\n" + "=" * 70)
    print("SUMMARY (per-trade bps; positive = profit)")
    print("=" * 70)
    rows = []
    for label, df in [("WF v1", df_v1_wf), ("WF v2", df_v2_wf), ("WF v3", df_v3_wf),
                       ("OOS v1", df_v1_oos), ("OOS v2", df_v2_oos), ("OOS v3", df_v3_oos)]:
        if df.empty:
            rows.append({"config": label, "n": 0,
                          "alpha": np.nan, "market": np.nan, "net": np.nan, "wins": 0, "folds": 0})
        else:
            rows.append({
                "config": label, "n": int(df["n"].sum()),
                "alpha": df["alpha_pnl_bps"].mean(),
                "market": df["market_pnl_bps"].mean(),
                "net": df["net_bps"].mean(),
                "wins": int((df["net_bps"] > 0).sum()),
                "folds": len(df),
            })
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
