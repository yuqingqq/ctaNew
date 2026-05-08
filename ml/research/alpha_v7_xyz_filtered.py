"""A+B + dispersion gating with filtered xyz extras.

Tests which xyz-only names degrade vs improve the strategy by adding them
selectively to BOTH training and trade universe.

Filter levels:
  - STRICT (15 names): pure S&P 100 overlap (current best, +1.22 Sharpe)
  - CONSERVATIVE (+4 = 19 names): add cleanest mature operating-business
                                   names: BX, EBAY, LITE, MRVL
  - MODERATE (+8 = 23 names): conservative + DKNG, HIMS, HOOD, ZM
  - LIBERAL (+10 = 25 names): moderate + RKLB, RIVN
  - FULL (+14 = 29 names): all xyz extras (already tested, +0.41-0.78)

Names always EXCLUDED (BTC-correlated / meme / bubble): MSTR, COIN, GME, BIRD
plus optionally RIVN, RKLB.
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_collectors.sp100_loader import SP100, fetch_daily, fetch_earnings
from ml.research.alpha_v7_multi import (
    LGB_PARAMS, SEEDS, TOP_K,
    load_anchors, add_returns_and_basket,
    add_features_A, add_features_B,
)
from ml.research.alpha_v7_weekly import (
    HOLD_DAYS, COST_PER_TRADE_BPS,
    add_residual_5d, fit_predict, metrics_weekly, bootstrap_ci_weekly, make_folds,
)
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz import construct_portfolio_subset, gate_by_dispersion

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# 15 names overlapping with S&P 100 (always tradeable, always in training as
# part of S&P 100)
XYZ_IN_SP100 = ["AAPL", "AMD", "AMZN", "COST", "GOOGL", "INTC", "LLY", "META",
                "MSFT", "MU", "NFLX", "NVDA", "ORCL", "PLTR", "TSLA"]

# Clean operating businesses with mature earnings dynamics. ADD these.
CLEAN_EXTRAS = ["BX", "EBAY", "LITE", "MRVL"]
# Mid-cap operating businesses, real earnings cycle (some COVID distortion).
# Include in moderate.
OPERATING_EXTRAS = ["DKNG", "HIMS", "HOOD", "ZM"]
# Speculative / narrative-driven (less earnings-driven). Include in liberal.
SPECULATIVE_EXTRAS = ["RKLB", "RIVN"]
# Always-excluded (BTC-correlated, meme, penny). Bad signal pollution.
DROPPED = ["MSTR", "COIN", "GME", "BIRD"]


def load_filtered_universe(extras: list[str], min_history_days: int = 365 * 2):
    """S&P 100 + only the specified xyz extras (filtered)."""
    rows, earnings_rows, surviving = [], [], []
    all_names = sorted(set(SP100) | set(extras))
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
    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    earn = pd.concat(earnings_rows, ignore_index=True) if earnings_rows else pd.DataFrame()
    return panel, earn, surviving


def run_one_config(name: str, extras: list[str], log_results: dict) -> None:
    log.info("\n" + "=" * 78)
    log.info("CONFIG: %s  (extras: %s)", name, extras)
    log.info("=" * 78)

    panel, earnings, surv = load_filtered_universe(extras, min_history_days=365 * 2)
    if panel.empty:
        log.error("  no data"); return

    trade_universe = [s for s in (XYZ_IN_SP100 + extras) if s in surv]
    log.info("  training universe: %d names  trade universe: %d names",
             len(surv), len(trade_universe))

    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    panel = add_residual_5d(panel)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    label = "fwd_resid_5d"
    pnl_label = "fwd_resid_5d"
    folds = make_folds(panel, train_min_days=365 * 3, test_days=365)
    feats = feats_A + feats_B + ["sym_id"]

    # K scales with universe size: K=5 for 15-19, K=6 for 20-25, K=7 for 26+
    top_k = 5 if len(trade_universe) <= 19 else (6 if len(trade_universe) <= 25 else 7)

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
            allowed_symbols=set(trade_universe), top_k=top_k,
            cost_bps=COST_PER_TRADE_BPS, hold_days=HOLD_DAYS,
        )
        if not lp.empty:
            all_pnls.append(lp)

    if not all_pnls:
        log.warning("  no rebalances"); return
    pnl = pd.concat(all_pnls, ignore_index=True)

    log.info("\n  UNGATED:")
    m = metrics_weekly(pnl)
    lo, hi = bootstrap_ci_weekly(pnl)
    log.info("    n=%d top_k=%d  net=%+.2f bps/5d  net_Sh=%+.2f  [%+.2f, %+.2f]",
             m["n_rebal"], top_k, m["net_bps_per_5d"],
             m["net_sharpe_annu"], lo, hi)

    gated = gate_by_dispersion(pnl, regime, threshold_pctile=0.6)
    if gated.empty:
        log.warning("  gated empty"); return
    log.info("\n  GATED (top 40% dispersion):")
    m = metrics_weekly(gated)
    lo, hi = bootstrap_ci_weekly(gated)
    log.info("    n=%d  gross=%+.2f bps/5d  net=%+.2f  net_Sh=%+.2f  [%+.2f, %+.2f]  hit=%.0f%%",
             m["n_rebal"], m["gross_bps_per_5d"], m["net_bps_per_5d"],
             m["net_sharpe_annu"], lo, hi, 100 * m["hit_rate"])
    log_results[name] = (m, lo, hi, gated, len(trade_universe), top_k)


def main() -> None:
    results: dict = {}
    configs = [
        ("STRICT (15)", []),
        ("CONSERVATIVE (+4 clean)", CLEAN_EXTRAS),
        ("MODERATE (+8)", CLEAN_EXTRAS + OPERATING_EXTRAS),
        ("LIBERAL (+10)", CLEAN_EXTRAS + OPERATING_EXTRAS + SPECULATIVE_EXTRAS),
        # FULL already tested; skip
    ]
    for name, extras in configs:
        run_one_config(name, extras, results)

    log.info("\n\n" + "=" * 78)
    log.info("SUMMARY (all gated, top 40% dispersion)")
    log.info("=" * 78)
    log.info("  %-28s %5s %5s %12s %10s %18s",
             "config", "n_uni", "n_reb", "net/5d", "net_Sh", "95% CI")
    for name, (m, lo, hi, _, n_uni, k) in results.items():
        log.info("  %-28s %5d %5d %+10.2fbps %+8.2f  [%+.2f, %+.2f]",
                 name, n_uni, m["n_rebal"], m["net_bps_per_5d"],
                 m["net_sharpe_annu"], lo, hi)

    # Per-year for best
    if results:
        best_name = max(results.keys(), key=lambda k: results[k][0].get("net_sharpe_annu", -10))
        log.info("\n=== Per-year for best: %s ===", best_name)
        gated = results[best_name][3].copy()
        gated["year"] = gated["ts"].dt.year
        log.info("  %-6s %5s %12s %12s %10s %8s",
                 "year", "n_reb", "gross/5d", "net/5d", "net_Sh", "hit")
        for y, g in gated.groupby("year"):
            qm = metrics_weekly(g)
            log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+8.2f %7.0f%%",
                     y, qm["n_rebal"], qm["gross_bps_per_5d"],
                     qm["net_bps_per_5d"], qm["net_sharpe_annu"],
                     100 * qm["hit_rate"])


if __name__ == "__main__":
    main()
