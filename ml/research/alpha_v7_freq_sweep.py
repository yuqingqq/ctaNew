"""Sweep hold-period / rebalance-frequency on A+B + dispersion gating.

Tests {1, 2, 3, 5} day hold periods. For each:
  - Compute fwd_resid_<h>d label
  - Train A+B model on it
  - Construct top-K=5 long-short portfolio with hold = h days
  - Apply dispersion gating
  - Use realistic xyz cost (5 bps round-trip per name swap)
  - Report Sharpe (per-rebalance), trades/year, annualized APR

Hypothesis: shorter holds amplify signal but inflate cost. With xyz's low
fees (~2 bps/side), daily rebalance might survive cost. Sweet spot likely
2-3 day hold.
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
    BETA_WINDOW, LGB_PARAMS, SEEDS,
    load_anchors, add_returns_and_basket,
    add_features_A, add_features_B,
)
from ml.research.alpha_v7_weekly import (
    fit_predict, metrics_weekly, bootstrap_ci_weekly, make_folds,
)
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz import construct_portfolio_subset, gate_by_dispersion
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def add_residual_and_label(panel: pd.DataFrame, fwd_days: int) -> pd.DataFrame:
    def _beta(g):
        cov = (g["ret"] * g["bk_ret"]).rolling(BETA_WINDOW).mean() - \
              g["ret"].rolling(BETA_WINDOW).mean() * g["bk_ret"].rolling(BETA_WINDOW).mean()
        var = g["bk_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
        return (cov / var).clip(-5, 5).shift(1)
    panel["beta"] = panel.groupby("symbol", group_keys=False).apply(_beta).values
    panel["resid"] = panel["ret"] - panel["beta"] * panel["bk_ret"]
    panel[f"fwd_resid_{fwd_days}d"] = (panel.groupby("symbol", group_keys=False)["resid"]
                                       .apply(lambda s: s.rolling(fwd_days).sum().shift(-fwd_days))
                                       .values)
    return panel


def metrics_freq(pnl: pd.DataFrame, hold_days: int) -> dict:
    """Per-rebalance metrics, annualized by sqrt(rebals_per_year)."""
    if pnl.empty:
        return {"n": 0}
    n = len(pnl)
    # rebalances per year if always-on at this hold period: 252 / hold_days
    rebals_per_year_max = 252 / hold_days
    g_sh_active = (pnl["spread_alpha"].mean() / pnl["spread_alpha"].std()
                   * np.sqrt(rebals_per_year_max)) if pnl["spread_alpha"].std() > 0 else 0
    n_sh_active = (pnl["net_alpha"].mean() / pnl["net_alpha"].std()
                   * np.sqrt(rebals_per_year_max)) if pnl["net_alpha"].std() > 0 else 0
    return {
        "n_rebal": n,
        "hold_days": hold_days,
        "gross_bps_per_rebal": pnl["spread_alpha"].mean() * 1e4,
        "net_bps_per_rebal": pnl["net_alpha"].mean() * 1e4,
        "gross_bps_per_day": pnl["spread_alpha"].mean() * 1e4 / hold_days,
        "net_bps_per_day": pnl["net_alpha"].mean() * 1e4 / hold_days,
        "active_sharpe_annu": n_sh_active,  # "in-market" sharpe
        "hit_rate": float((pnl["spread_alpha"] > 0).mean()),
        "avg_turnover": pnl["turnover"].mean() * 100,
    }


def annualized_unconditional(pnl: pd.DataFrame, hold_days: int,
                              ungated_rebals_per_year: float = 250) -> dict:
    """Compute true annualized return + vol accounting for idle (gated-out)
    weeks. ungated_rebals_per_year = 252/hold_days × (active fraction)."""
    if pnl.empty:
        return {}
    # active fraction estimated from data
    # rebalances/year actual = n_rebal / years
    if "year" not in pnl.columns:
        pnl_y = pnl.copy()
        pnl_y["year"] = pnl_y["ts"].dt.year
    else:
        pnl_y = pnl
    n_years = pnl_y["year"].nunique()
    rebals_per_year = len(pnl) / max(n_years, 1)
    mean_rebal = pnl["net_alpha"].mean()
    std_rebal = pnl["net_alpha"].std()
    annual_mean = rebals_per_year * mean_rebal
    annual_std = std_rebal * np.sqrt(rebals_per_year)
    annual_sharpe = annual_mean / annual_std if annual_std > 0 else 0
    return {
        "rebals_per_year": rebals_per_year,
        "annual_return_pct": annual_mean * 100,
        "annual_vol_pct": annual_std * 100,
        "unconditional_sharpe": annual_sharpe,
    }


def run_freq(panel_orig: pd.DataFrame, earnings: pd.DataFrame, anchors: pd.DataFrame,
             hold_days: int, cost_per_side_bps: float = 2.5,
             top_k: int = 5) -> tuple:
    """Build label, train, construct portfolio, gate, return pnl + metrics."""
    panel = panel_orig.copy()
    panel = add_residual_and_label(panel, hold_days)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    label = f"fwd_resid_{hold_days}d"
    feats = feats_A + feats_B + ["sym_id"]
    folds = make_folds(panel, train_min_days=365 * 3, test_days=365)

    all_pnls = []
    for fold in folds:
        train_end, test_start, test_end = fold
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        test_pred = fit_predict(train, test, feats, label)
        if test_pred.empty:
            continue
        # cost = turnover * cost_per_side * 2 (round-trip)
        lp = construct_portfolio_subset(
            test_pred, "pred", label,
            allowed_symbols=set(XYZ_IN_SP100),
            top_k=top_k, cost_bps=cost_per_side_bps * 2,
            hold_days=hold_days,
        )
        if not lp.empty:
            all_pnls.append(lp)

    if not all_pnls:
        return pd.DataFrame(), {}, {}
    pnl = pd.concat(all_pnls, ignore_index=True)
    pnl_g = gate_by_dispersion(pnl, regime, threshold_pctile=0.6)
    return pnl_g, metrics_freq(pnl_g, hold_days), annualized_unconditional(pnl_g, hold_days)


def main() -> None:
    log.info("loading universe + earnings + anchors...")
    panel, earnings, surv = load_universe()
    if panel.empty:
        return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)

    log.info("\nSweeping hold periods at 2.5 bps/side cost (xyz growth-mode est.)")
    log.info("\n=== GATED RESULTS (top 40% dispersion) ===")
    log.info("  %-6s %5s %12s %12s %10s %12s %14s %12s",
             "hold", "n_reb", "gross/d", "net/d", "active_Sh", "rebals/yr",
             "ann_ret%", "uncond_Sh")

    summaries = {}
    for hold in (1, 2, 3, 5):
        pnl, m, ann = run_freq(panel, earnings, anchors, hold_days=hold,
                                cost_per_side_bps=2.5)
        if not m:
            continue
        summaries[hold] = (pnl, m, ann)
        log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+10.2f %12.1f %+12.2f%% %12.2f",
                 hold, m["n_rebal"],
                 m["gross_bps_per_day"], m["net_bps_per_day"],
                 m["active_sharpe_annu"],
                 ann["rebals_per_year"],
                 ann["annual_return_pct"],
                 ann["unconditional_sharpe"])

    # Cost sensitivity for each hold period
    log.info("\n=== COST SENSITIVITY ACROSS HOLD PERIODS ===")
    log.info("  %-8s %-6s %-12s %-12s %-12s %-12s",
             "hold", "n_reb", "cost@2bps", "cost@5bps", "cost@10bps", "cost@15bps")
    for hold in (1, 2, 3, 5):
        if hold not in summaries:
            continue
        pnl_raw, _, _ = summaries[hold]
        results = []
        for cost_per_side in (2, 5, 10, 15):
            pnl_c = pnl_raw.copy()
            # rebuild net at this cost
            pnl_c["net_alpha"] = pnl_c["spread_alpha"] - pnl_c["turnover"] * cost_per_side * 2 / 1e4
            m = metrics_freq(pnl_c, hold)
            ann = annualized_unconditional(pnl_c, hold)
            results.append((m["active_sharpe_annu"], ann.get("unconditional_sharpe", 0)))
        log.info("  %-8d %-6d  Act/Unc:%5.2f/%4.2f  %5.2f/%4.2f  %5.2f/%4.2f  %5.2f/%4.2f",
                 hold, summaries[hold][1]["n_rebal"],
                 results[0][0], results[0][1],
                 results[1][0], results[1][1],
                 results[2][0], results[2][1],
                 results[3][0], results[3][1])

    # Per-year for best
    if summaries:
        best_hold = max(summaries.keys(),
                        key=lambda h: summaries[h][2].get("unconditional_sharpe", -10))
        log.info("\n=== Per-year breakdown for hold=%dd (best uncond Sharpe) ===", best_hold)
        pnl, _, _ = summaries[best_hold]
        pnl_y = pnl.copy()
        pnl_y["year"] = pnl_y["ts"].dt.year
        log.info("  %-6s %5s %12s %12s %10s",
                 "year", "n_reb", "gross/d", "net/d", "active_Sh")
        for y, g in pnl_y.groupby("year"):
            qm = metrics_freq(g, best_hold)
            log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+10.2f",
                     y, qm["n_rebal"], qm["gross_bps_per_day"],
                     qm["net_bps_per_day"], qm["active_sharpe_annu"])


if __name__ == "__main__":
    main()
