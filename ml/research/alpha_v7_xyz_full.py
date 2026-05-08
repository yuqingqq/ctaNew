"""A+B + dispersion gating on FULL xyz US equity universe (33 names).

Extends alpha_v7_xyz.py:
  - Loads S&P 100 (100 names) + 18 xyz-only US equities
  - Combined training universe: ~118 names
  - Trade only the 33 xyz-tradeable names
  - 4 of the 18 xyz-extras have <2y history (CRCL, CRWV, SNDK, USAR) and
    are auto-excluded from early folds; they contribute in 2024-2026 folds.
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_collectors.sp100_loader import (
    SP100, fetch_daily, fetch_earnings, load_universe,
)
from ml.research.alpha_v7_multi import (
    BETA_WINDOW, LGB_PARAMS, SEEDS, TOP_K,
    load_anchors, add_returns_and_basket,
    add_features_A, add_features_B,
)
from ml.research.alpha_v7_weekly import (
    HOLD_DAYS, COST_PER_TRADE_BPS,
    add_residual_5d, fit_predict, construct_portfolio_weekly,
    metrics_weekly, bootstrap_ci_weekly, make_folds,
)
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz import construct_portfolio_subset, gate_by_dispersion

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

XYZ_EXTRAS = [
    "BIRD", "BX", "COIN", "CRCL", "CRWV", "DKNG", "EBAY", "GME",
    "HIMS", "HOOD", "LITE", "MRVL", "MSTR", "RIVN", "RKLB", "SNDK",
    "USAR", "ZM",
]
ALL_XYZ_US_EQUITY = [
    # 15 in S&P 100 already
    "AAPL", "AMD", "AMZN", "COST", "GOOGL", "INTC", "LLY", "META",
    "MSFT", "MU", "NFLX", "NVDA", "ORCL", "PLTR", "TSLA",
    # 18 not in S&P 100 (4 may be filtered for short history)
    "BIRD", "BX", "COIN", "CRCL", "CRWV", "DKNG", "EBAY", "GME",
    "HIMS", "HOOD", "LITE", "MRVL", "MSTR", "RIVN", "RKLB", "SNDK",
    "USAR", "ZM",
]
TOP_K_XYZ_FULL = 5  # 5 long, 5 short out of valid xyz universe at each ts


def load_combined_universe(min_history_days: int = 365 * 2):
    """S&P 100 + xyz-only extras, with relaxed history filter."""
    rows = []
    earnings_rows = []
    surviving = []
    all_names = sorted(set(SP100) | set(XYZ_EXTRAS))
    for sym in all_names:
        d = fetch_daily(sym, start="2013-01-01")
        if d.empty or len(d) < min_history_days:
            continue
        e = fetch_earnings(sym)
        if e.empty or len(e) < 4:
            continue
        rows.append(d)
        earnings_rows.append(e)
        surviving.append(sym)
    panel = pd.concat(rows, ignore_index=True)
    earnings = pd.concat(earnings_rows, ignore_index=True)
    log.info("combined universe: %d names (%d S&P 100 + %d xyz extras with sufficient history)",
             len(surviving),
             sum(1 for s in surviving if s in SP100),
             sum(1 for s in surviving if s in XYZ_EXTRAS and s not in SP100))
    return panel, earnings, surviving


def main() -> None:
    log.info("loading combined universe (S&P 100 + xyz extras)...")
    panel, earnings, surv = load_combined_universe(min_history_days=365 * 2)
    if panel.empty:
        return
    anchors = load_anchors()

    xyz_tradeable = [s for s in ALL_XYZ_US_EQUITY if s in surv]
    log.info("xyz-tradeable subset: %d / %d", len(xyz_tradeable), len(ALL_XYZ_US_EQUITY))
    log.info("  tradeable: %s", xyz_tradeable)
    not_tradeable = [s for s in ALL_XYZ_US_EQUITY if s not in surv]
    if not_tradeable:
        log.info("  excluded (insufficient history): %s", not_tradeable)

    panel = add_returns_and_basket(panel)
    panel = add_residual_5d(panel)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    log.info("computing regime indicators...")
    regime = compute_regime_indicators(panel, anchors)

    label = "fwd_resid_5d"
    pnl_label = "fwd_resid_5d"
    folds = make_folds(panel, train_min_days=365 * 3, test_days=365)
    feats = feats_A + feats_B + ["sym_id"]
    log.info("running A+B walk-forward (%d folds)...", len(folds))

    # Single config: train on ALL combined universe, trade on xyz tradeable
    log.info("\n>>> Train on combined (%d names), trade on xyz (%d names)",
             len(surv), len(xyz_tradeable))
    all_pnls = []
    for fold in folds:
        train_end, test_start, test_end = fold
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        test_pred = fit_predict(train, test, feats, label)
        if test_pred.empty:
            continue
        lp = construct_portfolio_subset(
            test_pred, "pred", pnl_label,
            allowed_symbols=set(xyz_tradeable),
            top_k=TOP_K_XYZ_FULL, cost_bps=COST_PER_TRADE_BPS,
            hold_days=HOLD_DAYS,
        )
        if not lp.empty:
            all_pnls.append(lp)
    pnl = pd.concat(all_pnls, ignore_index=True) if all_pnls else pd.DataFrame()

    log.info("\n=== UNGATED ===")
    if pnl.empty:
        log.info("  no results"); return
    m = metrics_weekly(pnl)
    lo, hi = bootstrap_ci_weekly(pnl)
    log.info("  n=%d  gross=%+.2f bps/5d  net=%+.2f  net_Sh=%+.2f  [%+.2f, %+.2f]  hit=%.0f%%",
             m["n_rebal"], m["gross_bps_per_5d"], m["net_bps_per_5d"],
             m["net_sharpe_annu"], lo, hi, 100 * m["hit_rate"])

    log.info("\n=== DISPERSION-GATED (top 40%) ===")
    pnl_g = gate_by_dispersion(pnl, regime, threshold_pctile=0.6)
    if pnl_g.empty:
        log.info("  no gated results"); return
    m = metrics_weekly(pnl_g)
    lo, hi = bootstrap_ci_weekly(pnl_g)
    log.info("  n=%d  gross=%+.2f bps/5d  net=%+.2f  net_Sh=%+.2f  [%+.2f, %+.2f]  hit=%.0f%%",
             m["n_rebal"], m["gross_bps_per_5d"], m["net_bps_per_5d"],
             m["net_sharpe_annu"], lo, hi, 100 * m["hit_rate"])

    # Compare: gating threshold sensitivity
    log.info("\n=== Gating threshold sensitivity ===")
    log.info("  %-12s %5s %12s %12s %10s %18s",
             "pctile", "n", "gross/5d", "net/5d", "net_Sh", "95% CI")
    for p in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8):
        sub = gate_by_dispersion(pnl, regime, threshold_pctile=p)
        if sub.empty:
            continue
        m = metrics_weekly(sub)
        lo, hi = bootstrap_ci_weekly(sub)
        log.info("  %-12.2f %5d %+10.2fbps %+10.2fbps %+8.2f  [%+.2f, %+.2f]",
                 p, m["n_rebal"], m["gross_bps_per_5d"],
                 m["net_bps_per_5d"], m["net_sharpe_annu"], lo, hi)

    # Per-year for gated
    log.info("\n=== Per-year (gated, top 40%) ===")
    pnl_g["year"] = pnl_g["ts"].dt.year
    log.info("  %-6s %5s %12s %12s %10s %8s",
             "year", "n_reb", "gross/5d", "net/5d", "net_Sh", "hit")
    for y, g in pnl_g.groupby("year"):
        qm = metrics_weekly(g)
        log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+8.2f %7.0f%%",
                 y, qm["n_rebal"], qm["gross_bps_per_5d"],
                 qm["net_bps_per_5d"], qm["net_sharpe_annu"],
                 100 * qm["hit_rate"])


if __name__ == "__main__":
    main()
