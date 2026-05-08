"""Reliability tests for the +2.67 multi-horizon ensemble.

Three tests:
  1. Hard split (train 2013-2019 frozen, test 2020-2026)
     Same test that killed the original baseline (Sharpe -0.28)
     If ensemble survives, optimizations are robust to retraining absence.

  2. Hyperparameter perturbation
     Run with M=1 and M=3 (instead of M=2) — should be ~similar Sharpe if robust.

  3. Single horizon vs ensemble at hard split
     Does ensemble beat single h=5 in OOS? If yes, ensemble adds genuine value.
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_collectors.sp100_loader import load_universe
from ml.research.alpha_v7_multi import (
    LGB_PARAMS, SEEDS, add_returns_and_basket, add_features_A, load_anchors,
)
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100
from ml.research.alpha_v7_freq_sweep import (
    add_residual_and_label, metrics_freq, annualized_unconditional,
)
from ml.research.alpha_v7_pead_fixed import add_features_B_fixed
from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_daily_optimized import (
    daily_portfolio_hysteresis, make_folds, metrics_for, boot_ci,
)
from ml.research.alpha_v7_daily_v2 import run_walk_multihorizon

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOP_K = 5
COST_BPS_SIDE = 1.5
GATE_PCTILE = 0.6
GATE_WINDOW = 252


def hard_split_predict(panel: pd.DataFrame, feats: list[str], train_labels: list[str],
                        train_end: pd.Timestamp, test_start: pd.Timestamp,
                        test_end: pd.Timestamp) -> pd.DataFrame:
    """Train ONCE on data ≤ train_end, predict on test window. No retraining."""
    train = panel[panel["ts"] <= train_end].copy()
    test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
    log.info("  hard split: train n=%d (≤%s)  test n=%d",
             len(train), train_end.strftime("%Y-%m-%d"), len(test))
    ensemble_preds = []
    sub_test = None
    for label in train_labels:
        train_ = train.dropna(subset=feats + [label])
        if len(train_) < 1000:
            continue
        sub = test.dropna(subset=feats).copy()
        if sub_test is None:
            sub_test = sub.copy()
        for seed in SEEDS:
            m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
            m.fit(train_[feats], train_[label])
            ensemble_preds.append(m.predict(sub[feats]))
    if not ensemble_preds or sub_test is None:
        return pd.DataFrame()
    sub_test["pred"] = np.mean(ensemble_preds, axis=0)
    return sub_test


def main() -> None:
    log.info("loading panel...")
    panel, earnings, _ = load_universe()
    if panel.empty: return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    for h in (1, 3, 5, 10):
        panel = add_residual_and_label(panel, h)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    feats = feats_A + feats_B + ["sym_id"]
    allowed = set(XYZ_IN_SP100)

    # ---- Test 1: Hard split, ensemble (3,5,10) -----------------------
    log.info("\n=== TEST 1: Hard split (train 2013-2019, test 2020-2026 frozen) ===")
    log.info(">>> Multi-horizon ensemble (3,5,10), M=2")
    test_pred_e = hard_split_predict(
        panel, feats, ["fwd_resid_3d", "fwd_resid_5d", "fwd_resid_10d"],
        pd.Timestamp("2019-12-31", tz="UTC"),
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2026-05-05", tz="UTC"),
    )
    if not test_pred_e.empty:
        pnl_e = daily_portfolio_hysteresis(test_pred_e, "pred", "fwd_resid_1d",
                                             allowed, TOP_K, 2, COST_BPS_SIDE)
        pnl_e_g = gate_rolling(pnl_e, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
        m = metrics_for(pnl_e_g, 1)
        lo, hi = boot_ci(pnl_e_g, 1)
        log.info("  ENSEMBLE FROZEN: n=%d  active_Sh=%+.2f  [%+.2f,%+.2f]  uncond=%+.2f  ann=%+.2f%%",
                 m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["uncond_sharpe"], m["annual_return_pct"])

    log.info(">>> Single horizon h=5, M=2 (for comparison)")
    test_pred_s = hard_split_predict(
        panel, feats, ["fwd_resid_5d"],
        pd.Timestamp("2019-12-31", tz="UTC"),
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2026-05-05", tz="UTC"),
    )
    if not test_pred_s.empty:
        pnl_s = daily_portfolio_hysteresis(test_pred_s, "pred", "fwd_resid_1d",
                                             allowed, TOP_K, 2, COST_BPS_SIDE)
        pnl_s_g = gate_rolling(pnl_s, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
        m = metrics_for(pnl_s_g, 1)
        lo, hi = boot_ci(pnl_s_g, 1)
        log.info("  SINGLE h=5 FROZEN: n=%d  active_Sh=%+.2f  [%+.2f,%+.2f]  uncond=%+.2f  ann=%+.2f%%",
                 m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["uncond_sharpe"], m["annual_return_pct"])

    # Per-year breakdown for hard-split ensemble
    if not test_pred_e.empty and not pnl_e_g.empty:
        pnl_e_g["year"] = pnl_e_g["ts"].dt.year
        log.info("\n  Per-year for ensemble FROZEN at 2019-12-31:")
        for y, g in pnl_e_g.groupby("year"):
            if len(g) < 3:
                continue
            ym = metrics_for(g, 1)
            log.info("    %d  n=%d  net/d=%+.2fbps  active_Sh=%+.2f",
                     y, ym["n_rebal"], ym["net_bps_per_rebal"], ym["active_sharpe"])

    # ---- Test 2: Walk-forward with M perturbations -----------------------
    log.info("\n=== TEST 2: Hyperparameter robustness (walk-forward, M perturbed) ===")
    folds = make_folds(panel)
    train_labels = ["fwd_resid_3d", "fwd_resid_5d", "fwd_resid_10d"]
    log.info("  %-10s  %5s  %10s  %16s",
             "M", "n_reb", "active_Sh", "95% CI")
    for M in (1, 2, 3):
        pnl_pre = run_walk_multihorizon(
            panel, feats, train_labels, folds,
            daily_portfolio_hysteresis,
            {"pnl_label": "fwd_resid_1d", "allowed": allowed,
             "top_k": TOP_K, "exit_buffer": M, "cost_bps_side": COST_BPS_SIDE},
        )
        if pnl_pre.empty:
            continue
        pnl_g = gate_rolling(pnl_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
        m = metrics_for(pnl_g, 1)
        lo, hi = boot_ci(pnl_g, 1)
        log.info("  M=%-8d  %5d  %+8.2f  [%+5.2f,%+5.2f]",
                 M, m["n_rebal"], m["active_sharpe"], lo, hi)

    # ---- Test 3: Sub-period stability (split walk-forward results) ----
    log.info("\n=== TEST 3: Walk-forward + sub-period split ===")
    pnl_wf_pre = run_walk_multihorizon(
        panel, feats, train_labels, folds,
        daily_portfolio_hysteresis,
        {"pnl_label": "fwd_resid_1d", "allowed": allowed,
         "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": COST_BPS_SIDE},
    )
    pnl_wf_g = gate_rolling(pnl_wf_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    log.info("  %-30s  %5s  %10s  %16s  %10s",
             "subset", "n", "active_Sh", "95% CI", "uncond_Sh")
    for name, sub in [
        ("Full sample (walk-forward)", pnl_wf_g),
        ("First half (≤2018)", pnl_wf_g[pnl_wf_g["ts"].dt.year <= 2018]),
        ("Mid (2019-2022)", pnl_wf_g[(pnl_wf_g["ts"].dt.year >= 2019) & (pnl_wf_g["ts"].dt.year <= 2022)]),
        ("Recent (2023+)", pnl_wf_g[pnl_wf_g["ts"].dt.year >= 2023]),
    ]:
        if sub.empty: continue
        m = metrics_for(sub, 1)
        lo, hi = boot_ci(sub, 1)
        log.info("  %-30s  %5d  %+8.2f  [%+5.2f,%+5.2f]  %+8.2f",
                 name, m["n_rebal"], m["active_sharpe"], lo, hi, m["uncond_sharpe"])


if __name__ == "__main__":
    main()
