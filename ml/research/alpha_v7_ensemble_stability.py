"""Per-year + regime split verification for multi-horizon ensemble.

The headline +2.67 active Sharpe needs to hold across regimes, not be a
2023/2026 windfall. This script:
  1. Re-runs the ensemble config (h=3,5,10 ensemble, M=2 hysteresis, gated)
  2. Per-year breakdown with bootstrap CI per year
  3. 2013-2019 vs 2020-2026 split (regime halves)
  4. Compute hit-rate per year (consistency check)
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

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
from ml.research.alpha_v7_honest import gate_rolling, bootstrap_active_sharpe_ci
from ml.research.alpha_v7_daily_optimized import (
    daily_portfolio_hysteresis, make_folds, metrics_for, boot_ci,
)
from ml.research.alpha_v7_daily_v2 import run_walk_multihorizon

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOP_K = 5
COST_BPS_SIDE = 1.5
EXIT_BUFFER = 2
GATE_PCTILE = 0.6
GATE_WINDOW = 252
HORIZONS = [3, 5, 10]


def main() -> None:
    log.info("loading panel...")
    panel, earnings, _ = load_universe()
    if panel.empty: return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    for h in HORIZONS + [1]:
        panel = add_residual_and_label(panel, h)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    feats = feats_A + feats_B + ["sym_id"]
    allowed = set(XYZ_IN_SP100)
    folds = make_folds(panel)
    train_labels = [f"fwd_resid_{h}d" for h in HORIZONS]

    log.info("\nRunning multi-horizon ensemble (h=%s, M=%d)...", HORIZONS, EXIT_BUFFER)
    pnl_pre = run_walk_multihorizon(
        panel, feats, train_labels, folds,
        daily_portfolio_hysteresis,
        {"pnl_label": "fwd_resid_1d", "allowed": allowed,
         "top_k": TOP_K, "exit_buffer": EXIT_BUFFER, "cost_bps_side": COST_BPS_SIDE},
    )
    pnl = gate_rolling(pnl_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    if pnl.empty:
        log.error("empty pnl"); return

    # Headline
    m = metrics_for(pnl, 1)
    lo, hi = boot_ci(pnl, 1)
    log.info("\n=== HEADLINE ===")
    log.info("  n_rebal=%d  active_Sh=%+.2f  [%+.2f, %+.2f]  uncond_Sh=%+.2f  ann_ret=%+.2f%%",
             m["n_rebal"], m["active_sharpe"], lo, hi,
             m["uncond_sharpe"], m["annual_return_pct"])

    # Per-year breakdown
    pnl["year"] = pnl["ts"].dt.year
    log.info("\n=== Per-year breakdown ===")
    log.info("  %-6s %5s %12s %12s %10s %10s %8s",
             "year", "n_reb", "gross/d", "net/d", "active_Sh", "uncond_Sh", "hit%")
    yr_rows = []
    for y, g in pnl.groupby("year"):
        if len(g) < 5:
            log.info("  %-6d %5d  too few trades", y, len(g))
            continue
        ym = metrics_for(g, 1)
        gross = g["spread_alpha"].mean() * 1e4
        hit = (g["spread_alpha"] > 0).mean()
        log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+10.2f %+10.2f %7.0f%%",
                 y, ym["n_rebal"], gross, ym["net_bps_per_rebal"],
                 ym["active_sharpe"], ym["uncond_sharpe"], 100 * hit)
        yr_rows.append({"year": y, "n": ym["n_rebal"], "active_Sh": ym["active_sharpe"],
                        "hit": hit, "annual_ret": ym["annual_return_pct"]})

    # Regime split: 2013-2019 vs 2020-2026
    log.info("\n=== Regime split ===")
    early = pnl[pnl["ts"].dt.year <= 2019]
    late = pnl[pnl["ts"].dt.year >= 2020]
    for name, sub in [("2013-2019 (pre-COVID)", early), ("2020-2026 (post-COVID)", late)]:
        if sub.empty:
            continue
        sm = metrics_for(sub, 1)
        slo, shi = boot_ci(sub, 1)
        log.info("  %-30s  n=%4d  active_Sh=%+.2f  [%+.2f,%+.2f]  uncond=%+.2f  ann=%+.2f%%",
                 name, sm["n_rebal"], sm["active_sharpe"], slo, shi,
                 sm["uncond_sharpe"], sm["annual_return_pct"])

    # Stability summary
    log.info("\n=== Year-by-year stability ===")
    if yr_rows:
        ys = pd.DataFrame(yr_rows)
        n_pos = (ys["active_Sh"] > 0).sum()
        n_total = len(ys)
        avg_hit = ys["hit"].mean()
        med_sh = ys["active_Sh"].median()
        log.info("  positive Sharpe years: %d / %d (%.0f%%)", n_pos, n_total, 100 * n_pos / n_total)
        log.info("  median per-year Sharpe: %+.2f", med_sh)
        log.info("  worst year:  %s (Sharpe %+.2f)",
                 ys.loc[ys["active_Sh"].idxmin(), "year"], ys["active_Sh"].min())
        log.info("  best year:   %s (Sharpe %+.2f)",
                 ys.loc[ys["active_Sh"].idxmax(), "year"], ys["active_Sh"].max())
        log.info("  std of yearly Sharpe: %.2f", ys["active_Sh"].std())

    # Drawdown analysis
    log.info("\n=== Drawdown analysis (cumulative net P&L) ===")
    pnl_sorted = pnl.sort_values("ts").reset_index(drop=True)
    cum = pnl_sorted["net_alpha"].cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = drawdown.min() * 1e4  # bps
    # In percent of cumulative gain
    final_cum = cum.iloc[-1]
    # Convert to annualized return basis
    n_years = (pnl_sorted["ts"].max() - pnl_sorted["ts"].min()).days / 365.25
    annualized_return_pct = (final_cum / n_years) * 100
    max_dd_pct = (drawdown.min() / max(running_max.max(), 0.01)) * 100
    # Find drawdown duration
    in_dd = drawdown < -0.001
    dd_runs = []
    cur_run = 0
    for in_drawdown in in_dd:
        if in_drawdown:
            cur_run += 1
        else:
            if cur_run > 0:
                dd_runs.append(cur_run)
            cur_run = 0
    if cur_run > 0:
        dd_runs.append(cur_run)
    max_dd_days = max(dd_runs) if dd_runs else 0
    log.info("  cumulative net return:   %+.2f%% (over %.1f years = %+.2f%%/yr)",
             final_cum * 100, n_years, annualized_return_pct)
    log.info("  max drawdown:            %.2f bps cumulative (%.1f%% of peak)", max_dd, max_dd_pct)
    log.info("  longest drawdown spell:  %d trading days", max_dd_days)


if __name__ == "__main__":
    main()
