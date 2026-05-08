"""How often does the strategy need to retrain to maintain edge?

Tests retrain cadences 1mo, 3mo, 6mo, 12mo, 24mo by varying the test_days
parameter in walk-forward. Each fold: expanding-window train, then test
for `cadence` months without re-training.

Hard split (train 2013-2019, test 2020-2026 frozen) gave Sharpe -0.28.
Walk-forward with 12-month test gave +1.5. Goal: find the cadence-Sharpe
curve to identify optimal retrain frequency.

If quarterly (3mo) gives a meaningful lift over annual (12mo), that's
the answer. If gains plateau by 12mo, no need to retrain more often.
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_collectors.sp100_loader import load_universe
from ml.research.alpha_v7_multi import (
    add_returns_and_basket, add_features_A, load_anchors,
)
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz import construct_portfolio_subset
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100
from ml.research.alpha_v7_freq_sweep import (
    add_residual_and_label, metrics_freq, annualized_unconditional,
)
from ml.research.alpha_v7_pead_fixed import add_features_B_fixed
from ml.research.alpha_v7_honest import (
    fit_predict, gate_rolling, bootstrap_active_sharpe_ci,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HOLD_DAYS = 3
COST_BPS_SIDE = 2.5
TOP_K = 5


def make_folds_cadence(panel: pd.DataFrame, train_min_days: int, test_days: int,
                        embargo_days: int = 5) -> list[tuple]:
    """Build walk-forward folds with given test_days length per fold."""
    panel = panel.sort_values("ts")
    t0 = panel["ts"].min().normalize()
    t_max = panel["ts"].max()
    folds = []
    days = train_min_days
    while True:
        train_end = t0 + timedelta(days=days)
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_days)
        if test_start >= t_max:
            break
        if test_end > t_max:
            test_end = t_max
        folds.append((train_end, test_start, test_end))
        days += test_days
    return folds


def run_at_cadence(panel: pd.DataFrame, anchors: pd.DataFrame, regime: pd.DataFrame,
                    feats: list[str], label: str, test_days: int) -> tuple:
    """Run walk-forward at given cadence (test_days = retrain frequency).
    First fold uses 3y train minimum."""
    folds = make_folds_cadence(panel, train_min_days=365 * 3, test_days=test_days)
    log.info("  cadence=%dd, %d folds total", test_days, len(folds))
    all_pnls = []
    for fold in folds:
        train_end, test_start, test_end = fold
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        test_pred = fit_predict(train, test, feats, label)
        if test_pred.empty:
            continue
        lp = construct_portfolio_subset(
            test_pred, "pred", label, allowed_symbols=set(XYZ_IN_SP100),
            top_k=TOP_K, cost_bps=COST_BPS_SIDE * 2, hold_days=HOLD_DAYS,
        )
        if not lp.empty:
            all_pnls.append(lp)
    if not all_pnls:
        return pd.DataFrame(), {}, {}
    pnl = pd.concat(all_pnls, ignore_index=True)
    pnl_g = gate_rolling(pnl, regime, pctile=0.6, window_days=252)
    return pnl_g, metrics_freq(pnl_g, HOLD_DAYS), annualized_unconditional(pnl_g, HOLD_DAYS)


def main() -> None:
    log.info("loading universe + earnings + anchors...")
    panel, earnings, _ = load_universe()
    if panel.empty:
        return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    panel = add_residual_and_label(panel, HOLD_DAYS)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    label = f"fwd_resid_{HOLD_DAYS}d"
    feats = feats_A + feats_B + ["sym_id"]

    log.info("\n=== Retrain cadence sensitivity (PIT rolling gate, A+B fixed PEAD) ===")
    log.info("  %-12s  %-5s  %-5s  %-12s  %-10s  %-16s  %-10s",
             "cadence", "folds", "n_reb", "net/d", "active_Sh", "95% CI", "uncond_Sh")

    cadence_days = {
        "1 month": 30,
        "3 months": 90,
        "6 months": 180,
        "12 months": 365,
        "24 months": 730,
    }
    results = {}
    for name, td in cadence_days.items():
        pnl, m, ann = run_at_cadence(panel, anchors, regime, feats, label, td)
        if not m:
            continue
        lo, hi = bootstrap_active_sharpe_ci(pnl, HOLD_DAYS)
        results[name] = (m, ann, lo, hi)
        log.info("  %-12s  %5d  %5d  %+10.2fbps  %+8.2f  [%+.2f,%+.2f]  %+8.2f",
                 name,
                 (730 // td) if td <= 730 else 0,  # rough fold count
                 m["n_rebal"], m["net_bps_per_day"],
                 m["active_sharpe_annu"], lo, hi,
                 ann["unconditional_sharpe"])

    # Per-year for the BEST cadence
    if results:
        best_name = max(results.keys(), key=lambda k: results[k][0]["active_sharpe_annu"])
        log.info("\n=== Per-year for best cadence: %s ===", best_name)
        # Re-run to get per-year breakdown
        td = cadence_days[best_name]
        pnl, _, _ = run_at_cadence(panel, anchors, regime, feats, label, td)
        if not pnl.empty:
            pnl["year"] = pnl["ts"].dt.year
            log.info("  %-6s %5s %12s %12s %10s",
                     "year", "n_reb", "gross/d", "net/d", "active_Sh")
            for y, g in pnl.groupby("year"):
                qm = metrics_freq(g, HOLD_DAYS)
                log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+10.2f",
                         y, qm["n_rebal"], qm["gross_bps_per_day"],
                         qm["net_bps_per_day"], qm["active_sharpe_annu"])


if __name__ == "__main__":
    main()
