"""Production-ready xyz strategy: A + B (fixed PEAD) + E (extended hours)
on full 100-name S&P 100 universe with all 100 having Polygon 5m cache.

Configuration:
  - Universe: S&P 100 (training), trade only XYZ_IN_SP100 (15 names, xyz tradeable)
  - Features:
    A — price-pattern (10 features)
    B — fixed PEAD with AMC→next-BDay timing (4 features)
    E — extended hours (5 features) with full-history coverage (after Polygon
        5m fetch for all 100 names)
  - Model: LGBM, 5-seed ensemble, NaN-tolerant fit (E features available
    only for trading days 2024-06+, model handles via internal split-on-NaN)
  - Hold: 3 days. Rebalance every 3 trading days when gate is open.
  - Gate: trade only if 22d cross-sectional dispersion ≥ 60th percentile.
  - Cost: 5 bps round-trip per name swap (xyz growth-mode estimate).
  - Walk-forward: 11 expanding folds across 2013-2026.

Compares two configs:
  C1 — A + B (production baseline, +1.63 active Sharpe established)
  C2 — A + B + E (with full-history E features now possible)
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
    LGB_PARAMS, SEEDS,
    add_returns_and_basket, add_features_A, load_anchors,
)
from ml.research.alpha_v7_weekly import (
    metrics_weekly, bootstrap_ci_weekly, make_folds,
)
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz import construct_portfolio_subset, gate_by_dispersion
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100
from ml.research.alpha_v7_freq_sweep import (
    add_residual_and_label, metrics_freq, annualized_unconditional,
)
from ml.research.alpha_v7_pead_fixed import add_features_B_fixed
from ml.research.alpha_v7_extended_hours import add_features_E

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HOLD_DAYS = 3
COST_BPS_SIDE = 2.5  # xyz growth-mode estimate
TOP_K = 5


def fit_predict_strict(train: pd.DataFrame, test: pd.DataFrame,
                        features: list[str], label: str,
                        require_features: list[str]) -> pd.DataFrame:
    """Strict dropna on `require_features` + label. Optional features
    (in `features` but not `require_features`) may be NaN — LGBM handles."""
    train_ = train.dropna(subset=require_features + [label])
    if len(train_) < 1000:
        return pd.DataFrame()
    sub = test.dropna(subset=require_features).copy()
    if sub.empty:
        return pd.DataFrame()
    preds = []
    for seed in SEEDS:
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(train_[features], train_[label])
        preds.append(m.predict(sub[features]))
    sub["pred"] = np.mean(preds, axis=0)
    return sub


def run_config(panel: pd.DataFrame, anchors: pd.DataFrame,
                feats: list[str], require: list[str], label: str,
                folds: list, name: str) -> pd.DataFrame:
    log.info("\n>>> %s  (n_features=%d, required=%d)", name, len(feats), len(require))
    all_pnls = []
    for fold in folds:
        train_end, test_start, test_end = fold
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        test_pred = fit_predict_strict(train, test, feats, label, require)
        if test_pred.empty:
            continue
        lp = construct_portfolio_subset(
            test_pred, "pred", label,
            allowed_symbols=set(XYZ_IN_SP100),
            top_k=TOP_K, cost_bps=COST_BPS_SIDE * 2,
            hold_days=HOLD_DAYS,
        )
        if not lp.empty:
            all_pnls.append(lp)
    if not all_pnls:
        return pd.DataFrame()
    return pd.concat(all_pnls, ignore_index=True)


def report(name: str, pnl: pd.DataFrame, regime: pd.DataFrame) -> tuple:
    if pnl.empty:
        log.info("  %s: NO RESULTS", name); return None, None
    pnl_g = gate_by_dispersion(pnl, regime, threshold_pctile=0.6)
    if pnl_g.empty:
        log.info("  %s gated: NO RESULTS", name); return None, None
    m = metrics_freq(pnl_g, HOLD_DAYS)
    ann = annualized_unconditional(pnl_g, HOLD_DAYS)
    log.info("  %-30s  n=%4d  net/d=%+.2fbps  active_Sh=%+.2f  uncond_Sh=%+.2f  ann_ret=%+.2f%%",
             name, m["n_rebal"], m["net_bps_per_day"],
             m["active_sharpe_annu"], ann["unconditional_sharpe"],
             ann["annual_return_pct"])
    # Per-year
    pnl_y = pnl_g.copy()
    pnl_y["year"] = pnl_y["ts"].dt.year
    return m, ann


def main() -> None:
    log.info("loading universe + earnings + anchors...")
    panel, earnings, surv = load_universe()
    if panel.empty:
        return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    panel = add_residual_and_label(panel, HOLD_DAYS)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    log.info("Computing extended-hours features (now uses %d cached names)...",
             len([p for p in Path('/home/yuqing/ctaNew/data/ml/cache').glob('poly_*_5m.parquet')]))
    panel, feats_E = add_features_E(panel)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes

    # Diagnostic: how many panel rows have non-NaN E features?
    n_e_valid = panel[feats_E[0]].notna().sum() if feats_E else 0
    log.info("E features non-NaN: %d / %d panel rows (%.1f%%)",
             n_e_valid, len(panel), 100 * n_e_valid / max(len(panel), 1))

    regime = compute_regime_indicators(panel, anchors)
    folds = make_folds(panel, train_min_days=365 * 3, test_days=365)
    label = f"fwd_resid_{HOLD_DAYS}d"

    log.info("\n=== Production strategy ablation (cost=%.1f bps/side, K=%d, hold=%dd) ===",
             COST_BPS_SIDE, TOP_K, HOLD_DAYS)

    # C1: strict A + B (baseline +1.63)
    feats1 = feats_A + feats_B + ["sym_id"]
    require1 = feats_A + feats_B + ["sym_id"]
    pnl1 = run_config(panel, anchors, feats1, require1, label, folds, "C1: A+B fixed PEAD strict")
    m1, ann1 = report("C1 strict A+B fixed PEAD", pnl1, regime)

    # C2: A + B + E (E optional via NaN-tolerant)
    feats2 = feats_A + feats_B + feats_E + ["sym_id"]
    require2 = feats_A + feats_B + ["sym_id"]
    pnl2 = run_config(panel, anchors, feats2, require2, label, folds, "C2: A+B+E (E optional)")
    m2, ann2 = report("C2 A+B+E (E optional)", pnl2, regime)

    # C3: A + B + E (E required — only 2024+ trains)
    feats3 = feats_A + feats_B + feats_E + ["sym_id"]
    require3 = feats_A + feats_B + feats_E + ["sym_id"]
    pnl3 = run_config(panel, anchors, feats3, require3, label, folds, "C3: A+B+E (E required)")
    m3, ann3 = report("C3 A+B+E (E required, 2024+ only)", pnl3, regime)

    # Summary
    log.info("\n=== SUMMARY ===")
    log.info("  %-32s %5s %12s %10s %12s",
             "config", "n_reb", "net/d", "act_Sh", "uncond_Sh")
    for n, m, ann in [("C1 strict A+B fixed PEAD", m1, ann1),
                       ("C2 A+B+E (E optional)", m2, ann2),
                       ("C3 A+B+E (E required, 2024+)", m3, ann3)]:
        if m is None: continue
        log.info("  %-32s %5d %+10.2fbps %+8.2f %+10.2f",
                 n, m["n_rebal"], m["net_bps_per_day"],
                 m["active_sharpe_annu"], ann["unconditional_sharpe"])

    # Per-year for best config
    best_pnl, best_name = max(
        [(pnl1, "C1"), (pnl2, "C2"), (pnl3, "C3")],
        key=lambda t: 0 if t[0].empty else
            metrics_freq(gate_by_dispersion(t[0], regime, 0.6), HOLD_DAYS).get("active_sharpe_annu", -10),
    )
    if not best_pnl.empty:
        gated = gate_by_dispersion(best_pnl, regime, 0.6)
        gated["year"] = gated["ts"].dt.year
        log.info("\n=== Per-year breakdown for best config: %s ===", best_name)
        log.info("  %-6s %5s %12s %12s %10s",
                 "year", "n_reb", "gross/d", "net/d", "active_Sh")
        for y, g in gated.groupby("year"):
            qm = metrics_freq(g, HOLD_DAYS)
            log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+10.2f",
                     y, qm["n_rebal"], qm["gross_bps_per_day"],
                     qm["net_bps_per_day"], qm["active_sharpe_annu"])


if __name__ == "__main__":
    main()
