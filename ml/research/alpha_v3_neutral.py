"""alpha_v3 with MARKET-NEUTRAL execution.

Diagnosis from alpha_v3_btceth.py:
  - We predict alpha residual but trade my_symbol UNHEDGED.
  - Realized P&L = side × my_fwd = alpha + β × ref_fwd.
  - The β × ref_fwd component is uncorrelated noise — sometimes large.
  - In OOS BTC: alpha=+9 bps, market_pnl=-30 bps → net=-32 (market eats alpha).

If we predict alpha and want to capture alpha, we should HEDGE the market
component: LONG 1 unit of my_symbol AND SHORT β units of ref_symbol.

Realized hedged P&L per trade =
    side × my_fwd - side × β × ref_fwd
  = side × (my_fwd - β × ref_fwd)
  = side × alpha_realized                    ← exactly alpha capture!

Cost: now 2 round trips (my and ref). Same fee structure on each leg, plus
spread on each. ~2× cost vs naked.

This probe: same v3 model, trade hedged, recompute net P&L.
Tests whether the alpha edge survives 2× cost.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.cost_model import CostConfig
from ml.cv import FoldSpec, make_walk_forward_folds
from ml.research.alpha_v3 import (
    REF_OF, HORIZON, REGIME_CUTOFF, VOL_WIN, HOLDOUT_DAYS, BETA_WINDOW,
    _make_alpha_label, _enrich, _canonicalize, _harmonized_columns,
    _train, _expanding_train,
)
from ml.research.trend_pooled_v2 import _build_symbol_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ENSEMBLE_SEEDS = (42, 7, 123, 99, 314)


def _run_neutral(symbols: list, *, mode: str, threshold_q: float, hedge: bool):
    cost = CostConfig(flat_slippage_bps=1.0)
    fee = cost.fee_taker * 2; slip = 2 * cost.flat_slippage_bps / 1e4

    sym_data = {}; spread_data = {}; ref_spread = {}
    for s in symbols:
        feats, sp = _enrich(s, "v3")
        feats = _canonicalize(feats, s, "v3")
        ref_feats, ref_sp = _build_symbol_features(REF_OF[s])
        labels = _make_alpha_label(feats, ref_feats, HORIZON)
        sym_data[s] = (feats, labels)
        spread_data[s] = sp
        ref_spread[s] = ref_sp

    cols = _harmonized_columns("v3")
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

    fold_rows = []
    for fold in folds:
        train_dfs, cal_per_sym = [], {}
        for s in symbols:
            feats_s, labels_s = sym_data[s]
            splits = _expanding_train(feats_s, labels_s, fold)
            train_dfs.append(splits["train"])
            cal_per_sym[s] = splits["cal"]
        train = pd.concat(train_dfs, ignore_index=True).dropna(
            subset=cols + ["demeaned_target", "return_pct"])
        train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
        cal_pool = pd.concat(list(cal_per_sym.values()), ignore_index=True).dropna(
            subset=cols + ["demeaned_target"])
        cal_pool_f = cal_pool[cal_pool["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
        if len(train_f) < 200 or len(cal_pool_f) < 50: continue

        models = []
        for seed in ENSEMBLE_SEEDS:
            m = _train(train_f[cols].to_numpy(),
                        train_f["demeaned_target"].to_numpy(),
                        cal_pool_f[cols].to_numpy(),
                        cal_pool_f["demeaned_target"].to_numpy(), seed=seed)
            models.append(m)

        # Per-symbol threshold on its own cal predictions
        per_sym_thr = {}
        for s in symbols:
            cal_s = cal_per_sym[s].dropna(subset=cols + ["demeaned_target"])
            cal_s = cal_s[cal_s["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
            if len(cal_s) < 30:
                per_sym_thr[s] = np.nan; continue
            preds_s = np.mean([m.predict(cal_s[cols].to_numpy(),
                                            num_iteration=m.best_iteration)
                                for m in models], axis=0)
            per_sym_thr[s] = float(np.quantile(np.abs(preds_s), threshold_q))

        for s in symbols:
            feats_s, labels_s = sym_data[s]
            splits = _expanding_train(feats_s, labels_s, fold)
            test = splits["test"].dropna(
                subset=cols + ["demeaned_target", "return_pct", "atr_pct",
                               "alpha_realized", "beta", "ref_fwd", "exit_time"])
            test_f = test[test["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
            if len(test_f) < 20: continue

            preds = [m.predict(test_f[cols].to_numpy(),
                                num_iteration=m.best_iteration) for m in models]
            yt = np.mean(preds, axis=0)
            thr = per_sym_thr[s]
            if not np.isfinite(thr): continue
            triggered = np.abs(yt) >= thr
            n = int(triggered.sum())
            if n == 0: continue

            side = np.sign(yt[triggered]); side[side == 0] = 1
            atr = np.clip(test_f["atr_pct"].to_numpy()[triggered], 1e-4, 1e-1)
            inv_vol = np.clip((1.0 / atr) / (1.0 / atr).mean(), 0.3, 3.0)

            my_fwd = test_f["return_pct"].to_numpy()[triggered]
            ref_fwd = test_f["ref_fwd"].to_numpy()[triggered]
            beta_t = test_f["beta"].to_numpy()[triggered]
            alpha_t = test_f["alpha_realized"].to_numpy()[triggered]

            # Gross PnL: hedged = side × alpha; naked = side × my_fwd
            if hedge:
                gross = side * alpha_t * inv_vol
                # Cost: my-leg + ref-leg
                idx = test_f.index[triggered]
                exit_idx = test_f["exit_time"].iloc[triggered]
                my_sp_e = spread_data[s].reindex(idx).fillna(0.0).to_numpy()
                my_sp_x = spread_data[s].reindex(exit_idx).fillna(0.0).to_numpy()
                ref_sp_e = ref_spread[s].reindex(idx).fillna(0.0).to_numpy()
                ref_sp_x = ref_spread[s].reindex(exit_idx).fillna(0.0).to_numpy()
                # Hedge size = |beta| (in units of ref vs my). Cost scales with each leg's size.
                my_spread_term = 0.5 * (my_sp_e + my_sp_x) / 1e4
                ref_spread_term = 0.5 * (ref_sp_e + ref_sp_x) / 1e4 * np.abs(beta_t)
                # Per-leg fee+slip; ref leg sized at |beta|
                cost_per_trade = (
                    (fee + slip + my_spread_term)
                    + np.abs(beta_t) * (fee + slip)
                    + ref_spread_term
                ) * inv_vol
            else:
                gross = side * my_fwd * inv_vol
                idx = test_f.index[triggered]
                exit_idx = test_f["exit_time"].iloc[triggered]
                sp_e = spread_data[s].reindex(idx).fillna(0.0).to_numpy()
                sp_x = spread_data[s].reindex(exit_idx).fillna(0.0).to_numpy()
                spread_term = 0.5 * (sp_e + sp_x) / 1e4
                cost_per_trade = (fee + slip + spread_term) * inv_vol

            net = gross - cost_per_trade

            fold_rows.append({
                "fold": fold.fold_id, "symbol": s, "n": n, "hedge": hedge,
                "trigger_rate": float(triggered.mean()),
                "win_rate": float((net > 0).mean()),
                "alpha_pnl_bps": float((side * alpha_t * inv_vol).mean() * 1e4),
                "market_pnl_bps": float((side * beta_t * ref_fwd * inv_vol).mean() * 1e4),
                "gross_bps": float(gross.mean() * 1e4),
                "cost_bps": float(cost_per_trade.mean() * 1e4),
                "net_bps": float(net.mean() * 1e4),
                "ic_pred_alpha": float(np.corrcoef(yt, test_f["alpha_realized"].to_numpy())[0, 1]),
            })
    return pd.DataFrame(fold_rows)


def _summary(df, label):
    print(f"\n=== {label} ===")
    if df.empty:
        print("  EMPTY"); return
    print(f"  total: n={int(df['n'].sum())}, "
          f"trig={df['trigger_rate'].mean()*100:.1f}%, "
          f"alpha={df['alpha_pnl_bps'].mean():+.2f}, "
          f"gross={df['gross_bps'].mean():+.2f}, "
          f"cost={df['cost_bps'].mean():.2f}, "
          f"net={df['net_bps'].mean():+.2f} bps, "
          f"folds_pos={int((df['net_bps']>0).sum())}/{len(df)}")
    if df["symbol"].nunique() > 1:
        for s, sub in df.groupby("symbol"):
            print(f"    {s}: n={int(sub['n'].sum())}, "
                  f"trig={sub['trigger_rate'].mean()*100:.1f}%, "
                  f"alpha={sub['alpha_pnl_bps'].mean():+.2f}, "
                  f"gross={sub['gross_bps'].mean():+.2f}, "
                  f"cost={sub['cost_bps'].mean():.2f}, "
                  f"net={sub['net_bps'].mean():+.2f}, "
                  f"folds_pos={int((sub['net_bps']>0).sum())}/{len(sub)}")


def main():
    print("=" * 70)
    print("v3 NEUTRAL HEDGE vs NAKED — does β-hedging let alpha edge survive?")
    print("=" * 70)

    for syms_label, syms in [("BTC+ETH+SOL", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
                                ("BTC+ETH",     ["BTCUSDT", "ETHUSDT"])]:
        print(f"\n{'#' * 70}\n# Symbols: {syms_label}\n{'#' * 70}")

        for q in (0.95, 0.97, 0.99):
            print(f"\n--- threshold q={q} ---")
            print("\n=== NAKED (trade my_symbol only) ===")
            for mode in ("walkforward", "oos_holdout"):
                df = _run_neutral(syms, mode=mode, threshold_q=q, hedge=False)
                _summary(df, f"  {mode}")
            print("\n=== HEDGED (long my, short β×ref) ===")
            for mode in ("walkforward", "oos_holdout"):
                df = _run_neutral(syms, mode=mode, threshold_q=q, hedge=True)
                _summary(df, f"  {mode}")


if __name__ == "__main__":
    main()
