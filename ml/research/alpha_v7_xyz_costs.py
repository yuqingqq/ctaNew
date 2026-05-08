"""Cost-sensitivity analysis for STRICT 15-name xyz strategy.

Investigates two questions:
  1. Is my 5 bps/trade-side assumption realistic?
  2. How does Sharpe degrade at realistic costs?

Cost stack on xyz perp execution (per single name leg):
  - HL taker fee: 3.5 bps
  - HL maker fee: 1 bps (if patient execution)
  - Slippage in $30M/d book at $1M trade: 5-10 bps
  - Funding (5-day hold): 0.5-1 bps/8h × 15 = 7.5-15 bps  (long leg pays, short receives)

Realistic per-trade-side cost:
  - Aggressive taker: ~10 bps (fee + slippage)
  - Patient maker: ~3-4 bps (rebate captures part of spread)
  - Avg expected: ~7 bps

Total per rebalance (turnover-weighted):
  cost_per_rebal = turnover * cost_per_trade_side
  e.g. 25% turnover × 7 bps = 1.75 bps per rebalance
       (or × 14 bps for full round-trip = 3.5 bps)

I'll redo the analysis at multiple cost levels and also report
turnover and annualized cost burn.
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
    add_returns_and_basket, add_features_A, add_features_B,
    load_anchors,
)
from ml.research.alpha_v7_weekly import (
    HOLD_DAYS, add_residual_5d, fit_predict, make_folds,
    metrics_weekly, bootstrap_ci_weekly,
)
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz import construct_portfolio_subset, gate_by_dispersion
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    log.info("loading S&P 100 + earnings + anchors...")
    panel, earnings, surv = load_universe()
    if panel.empty:
        return
    anchors = load_anchors()

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

    # === Build P&L at zero cost; then apply different cost levels post-hoc ===
    # We re-run with cost_bps=0 to get raw turnover and gross alpha, then
    # compute net at multiple cost levels.
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
            allowed_symbols=set(XYZ_IN_SP100),
            top_k=5, cost_bps=0, hold_days=HOLD_DAYS,
        )
        if not lp.empty:
            all_pnls.append(lp)
    pnl = pd.concat(all_pnls, ignore_index=True)

    # Diagnostics: turnover, hit rate, distribution
    log.info("\n=== Strategy diagnostics (15-name STRICT, weekly, ungated) ===")
    log.info("  n_rebalances:           %d", len(pnl))
    log.info("  avg turnover/rebal:     %.1f%%", pnl["turnover"].mean() * 100)
    log.info("  median turnover:        %.1f%%", pnl["turnover"].median() * 100)
    log.info("  max turnover:           %.1f%%", pnl["turnover"].max() * 100)
    log.info("  rebalances/year (if always-on): ~52")
    log.info("  rebalances/year (gated 40%%):    ~21")
    log.info("  gross alpha mean:       %+.2f bps/5d", pnl["spread_alpha"].mean() * 1e4)
    log.info("  gross alpha std:        %+.2f bps/5d", pnl["spread_alpha"].std() * 1e4)

    # Apply gating
    gated = gate_by_dispersion(pnl, regime, threshold_pctile=0.6)
    log.info("\n=== GATED diagnostics ===")
    log.info("  n_rebalances:           %d", len(gated))
    log.info("  avg turnover/rebal:     %.1f%%", gated["turnover"].mean() * 100)
    log.info("  gross alpha mean:       %+.2f bps/5d", gated["spread_alpha"].mean() * 1e4)
    log.info("  gross alpha std:        %+.2f bps/5d", gated["spread_alpha"].std() * 1e4)

    # === Cost sensitivity: ungated and gated ===
    def evaluate_at_cost(pnl_in: pd.DataFrame, cost_per_trade_side_bps: float,
                         round_trip: bool = True) -> dict:
        """Re-compute net P&L at given cost.  round_trip=True means we charge
        2 × cost_per_trade × turnover (close + open per rebalanced name)."""
        pnl_e = pnl_in.copy()
        cost_mult = 2.0 if round_trip else 1.0
        pnl_e["cost"] = pnl_e["turnover"] * cost_per_trade_side_bps * cost_mult / 1e4
        pnl_e["net_alpha"] = pnl_e["spread_alpha"] - pnl_e["cost"]
        m = metrics_weekly(pnl_e)
        # annualized cost burn (depends on rebal frequency)
        rebals_per_year = (52 if len(pnl_in) >= 100 else 0)  # rough estimate
        ann_cost = pnl_e["cost"].mean() * 1e4 * 52  # assume weekly
        return {
            "cost_per_side_bps": cost_per_trade_side_bps,
            "round_trip": round_trip,
            "ann_cost_bps": ann_cost,
            **m,
        }

    log.info("\n=== UNGATED cost sensitivity (round-trip cost = 2 × per-side × turnover) ===")
    log.info("  %-15s %-12s %12s %10s %18s",
             "per-side bps", "round-trip", "net/5d", "net_Sh", "95% CI")
    for c in (0, 3, 5, 7, 10, 15, 20, 25):
        eg = evaluate_at_cost(pnl, c, round_trip=True)
        # bootstrap CI manually
        pnl_c = pnl.copy()
        pnl_c["net_alpha"] = pnl_c["spread_alpha"] - pnl_c["turnover"] * c * 2 / 1e4
        lo, hi = bootstrap_ci_weekly(pnl_c)
        log.info("  %-15.0f %-12s %+10.2fbps %+8.2f  [%+.2f, %+.2f]",
                 c, "RT" if True else "1-side",
                 eg["net_bps_per_5d"], eg["net_sharpe_annu"], lo, hi)

    log.info("\n=== GATED cost sensitivity (round-trip cost) ===")
    log.info("  %-15s %12s %10s %18s",
             "per-side bps", "net/5d", "net_Sh", "95% CI")
    for c in (0, 3, 5, 7, 10, 15, 20, 25):
        pnl_c = gated.copy()
        pnl_c["net_alpha"] = pnl_c["spread_alpha"] - pnl_c["turnover"] * c * 2 / 1e4
        m = metrics_weekly(pnl_c)
        lo, hi = bootstrap_ci_weekly(pnl_c)
        log.info("  %-15.0f %+10.2fbps %+8.2f  [%+.2f, %+.2f]",
                 c, m["net_bps_per_5d"], m["net_sharpe_annu"], lo, hi)

    # Realistic cost stacks
    log.info("\n=== REALISTIC COST STACKS ===")
    log.info("Stack 1 — HL xyz, aggressive taker:")
    log.info("  fee  3.5 bps + slippage 7 bps + funding 2 bps = 12.5 bps/trade-side")
    log.info("  RT  = 25 bps round-trip per name swap")
    pnl_c = gated.copy()
    pnl_c["net_alpha"] = pnl_c["spread_alpha"] - pnl_c["turnover"] * 25 / 1e4
    m = metrics_weekly(pnl_c)
    lo, hi = bootstrap_ci_weekly(pnl_c)
    log.info("  → GATED net=%+.2f bps/5d  net_Sh=%+.2f  [%+.2f, %+.2f]",
             m["net_bps_per_5d"], m["net_sharpe_annu"], lo, hi)

    log.info("\nStack 2 — HL xyz, patient maker (limit orders, rebates):")
    log.info("  fee  -1 bps (rebate) + slippage 3 bps + funding 2 bps = 4 bps/trade-side")
    log.info("  RT  = 8 bps round-trip per name swap")
    pnl_c = gated.copy()
    pnl_c["net_alpha"] = pnl_c["spread_alpha"] - pnl_c["turnover"] * 8 / 1e4
    m = metrics_weekly(pnl_c)
    lo, hi = bootstrap_ci_weekly(pnl_c)
    log.info("  → GATED net=%+.2f bps/5d  net_Sh=%+.2f  [%+.2f, %+.2f]",
             m["net_bps_per_5d"], m["net_sharpe_annu"], lo, hi)

    log.info("\nStack 3 — Cash equity broker (IBKR Pro, no shorts via perp):")
    log.info("  fee  0.35 bps + slippage 1 bps (mega-cap) + borrow 0.5 bps = 1.85/trade-side")
    log.info("  RT  = 3.7 bps round-trip per name swap")
    pnl_c = gated.copy()
    pnl_c["net_alpha"] = pnl_c["spread_alpha"] - pnl_c["turnover"] * 3.7 / 1e4
    m = metrics_weekly(pnl_c)
    lo, hi = bootstrap_ci_weekly(pnl_c)
    log.info("  → GATED net=%+.2f bps/5d  net_Sh=%+.2f  [%+.2f, %+.2f]",
             m["net_bps_per_5d"], m["net_sharpe_annu"], lo, hi)


if __name__ == "__main__":
    main()
