"""alpha_v3 with proper trigger calibration.

Diagnosis from alpha_v3.py:
  - v3 has the BEST per-symbol IC (BTC +0.080, ETH +0.064, SOL +0.035)
  - But trigger rate ballooned to ~26% OOS (vs ~5% target from q=0.95 on cal)
  - Result: alpha edge gets diluted by costs on too many trades

Hypothesis: prediction distributions differ across symbols. Pooling predictions
to find a single quantile threshold underweights some symbols. SOL's predictions
in v3 are systematically larger than ETH's, so 5% of the pooled cal distribution
maps to >5% of SOL's OOS distribution.

Fix: compute the threshold PER SYMBOL on cal — each symbol's q=0.95 of its
own cal predictions. Same logic, just stratified by symbol.

Also test:
  - Tighter quantile (q=0.97 → top 3% per symbol) for higher conviction.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features_ml.alpha_features import add_alpha_features
from ml.cost_model import CostConfig
from ml.cv import FoldSpec, make_walk_forward_folds, split_features_by_fold
from ml.research.alpha_v3 import (
    V3_BASE, V3_CROSS_TEMPLATES, SYM_INDICATOR, REF_OF, SYM_TO_ID,
    HORIZON, REGIME_CUTOFF, VOL_WIN, HOLDOUT_DAYS, BETA_WINDOW,
    _make_alpha_label, _enrich, _canonicalize, _harmonized_columns,
    _train, _expanding_train,
)
from ml.research.trend_pooled_v2 import _build_symbol_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ENSEMBLE_SEEDS = (42, 7, 123, 99, 314)


def _run(symbols: list, *, version: str, mode: str, threshold_q: float,
          per_symbol_threshold: bool):
    """version uses alpha_v3.py's machinery; per_symbol_threshold splits cal
    predictions by symbol before computing the trigger quantile."""
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

    fold_rows = []; last_models = []
    for fold in folds:
        # Train pooled
        train_dfs, cal_dfs_per_sym = [], {}
        for s in symbols:
            feats_s, labels_s = sym_data[s]
            splits = _expanding_train(feats_s, labels_s, fold)
            train_dfs.append(splits["train"])
            cal_dfs_per_sym[s] = splits["cal"]
        train = pd.concat(train_dfs, ignore_index=True).dropna(
            subset=cols + ["demeaned_target", "return_pct"])
        train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]

        # Pool cal for model training but keep per-symbol cal for threshold
        cal_pool = pd.concat(list(cal_dfs_per_sym.values()), ignore_index=True).dropna(
            subset=cols + ["demeaned_target", "return_pct"])
        cal_pool_f = cal_pool[cal_pool["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
        if len(train_f) < 200 or len(cal_pool_f) < 50: continue

        models = []
        for seed in ENSEMBLE_SEEDS:
            m = _train(train_f[cols].to_numpy(),
                        train_f["demeaned_target"].to_numpy(),
                        cal_pool_f[cols].to_numpy(),
                        cal_pool_f["demeaned_target"].to_numpy(), seed=seed)
            models.append(m)
        last_models = models

        # Threshold: pooled (single threshold for all) OR per-symbol
        per_sym_thr = {}
        if per_symbol_threshold:
            for s in symbols:
                cal_s = cal_dfs_per_sym[s].dropna(subset=cols + ["demeaned_target"])
                cal_s = cal_s[cal_s["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
                if len(cal_s) < 30:
                    per_sym_thr[s] = np.nan
                    continue
                preds_s = np.mean([m.predict(cal_s[cols].to_numpy(),
                                                num_iteration=m.best_iteration)
                                    for m in models], axis=0)
                per_sym_thr[s] = float(np.quantile(np.abs(preds_s), threshold_q))
        else:
            preds_pool = np.mean([m.predict(cal_pool_f[cols].to_numpy(),
                                              num_iteration=m.best_iteration)
                                    for m in models], axis=0)
            single_thr = float(np.quantile(np.abs(preds_pool), threshold_q))

        for s in symbols:
            feats_s, labels_s = sym_data[s]
            splits = _expanding_train(feats_s, labels_s, fold)
            test = splits["test"].dropna(
                subset=cols + ["demeaned_target", "return_pct", "atr_pct"])
            test_f = test[test["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
            if len(test_f) < 20: continue

            preds = [m.predict(test_f[cols].to_numpy(),
                                num_iteration=m.best_iteration) for m in models]
            yt = np.mean(preds, axis=0)
            thr = per_sym_thr[s] if per_symbol_threshold else single_thr
            if not np.isfinite(thr): continue
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
                "trigger_rate": float(triggered.mean()),
                "win_rate": float((net > 0).mean()),
                "alpha_pnl_bps": float(alpha_pnl.mean() * 1e4),
                "market_pnl_bps": float(market_pnl.mean() * 1e4),
                "gross_bps": float(gross.mean() * 1e4),
                "net_bps": float(net.mean() * 1e4),
                "ic_pred_alpha": float(np.corrcoef(yt, test_f["alpha_realized"].to_numpy())[0, 1]),
            })
    return pd.DataFrame(fold_rows), last_models


def _summary(df, label):
    print(f"\n=== {label} ===")
    if df.empty:
        print("  EMPTY"); return
    print(f"  total: n={int(df['n'].sum())}, "
          f"trig={df['trigger_rate'].mean()*100:.1f}%, "
          f"alpha={df['alpha_pnl_bps'].mean():+.2f}, "
          f"market={df['market_pnl_bps'].mean():+.2f}, "
          f"net={df['net_bps'].mean():+.2f} bps, "
          f"folds_pos={int((df['net_bps']>0).sum())}/{len(df)}")
    if df["symbol"].nunique() > 1:
        for s, sub in df.groupby("symbol"):
            print(f"    {s}: n={int(sub['n'].sum())}, "
                  f"trig={sub['trigger_rate'].mean()*100:.1f}%, "
                  f"alpha={sub['alpha_pnl_bps'].mean():+.2f}, "
                  f"market={sub['market_pnl_bps'].mean():+.2f}, "
                  f"net={sub['net_bps'].mean():+.2f}, "
                  f"ic={sub['ic_pred_alpha'].mean():+.4f}, "
                  f"folds_pos={int((sub['net_bps']>0).sum())}/{len(sub)}")


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    print("=" * 70)
    print("v3 WITH PER-SYMBOL THRESHOLDS (q=0.95) and TIGHTER (q=0.97)")
    print("=" * 70)

    configs = [
        ("v3 pooled thr q=0.95",     "v3", 0.95, False),
        ("v3 per-sym thr q=0.95",    "v3", 0.95, True),
        ("v3 per-sym thr q=0.97",    "v3", 0.97, True),
    ]

    print("\n--- WALK-FORWARD ---")
    for label, ver, q, ps in configs:
        df, _ = _run(symbols, version=ver, mode="walkforward", threshold_q=q,
                       per_symbol_threshold=ps)
        _summary(df, f"WF, {label}")

    print("\n--- OOS HOLDOUT ---")
    summary_rows = []
    for label, ver, q, ps in configs:
        df, models = _run(symbols, version=ver, mode="oos_holdout",
                            threshold_q=q, per_symbol_threshold=ps)
        _summary(df, f"OOS, {label}")
        if not df.empty:
            summary_rows.append({
                "config": label,
                "n": int(df["n"].sum()),
                "trig_pct": df["trigger_rate"].mean() * 100,
                "alpha": df["alpha_pnl_bps"].mean(),
                "net": df["net_bps"].mean(),
                "ic_avg": df["ic_pred_alpha"].mean(),
                "folds_pos": int((df["net_bps"] > 0).sum()),
            })

    print("\n" + "=" * 70)
    print("OOS HEAD-TO-HEAD")
    print("=" * 70)
    print(pd.DataFrame(summary_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
