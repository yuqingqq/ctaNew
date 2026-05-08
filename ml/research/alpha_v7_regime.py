"""Regime-conditional analysis of A+B weekly strategy.

Hypothesis: A+B's edge is concentrated in periods of high cross-sectional
dispersion (mainly: trending markets, sector rotations, vol regime shifts).
If true, gating trades by regime indicators could substantially improve
the unconditional Sharpe.

Procedure:
  1. Re-run A+B walk-forward (already cached panel).
  2. For each rebalance ts, attach regime indicators measured AT ts:
     - dispersion: cross-sectional std of 22d returns across universe
     - avg_corr: avg pairwise correlation of last 60d (low corr = high dispersion)
     - VIX_level: vol regime
     - SOXX_minus_SPY_22d: AI/semi-trend regime
     - XLK_minus_SPY_22d: broad-tech-trend regime
  3. Bucket P&L by quintile of each regime indicator.
  4. Compute Sharpe per quintile.
  5. Identify regimes where strategy works vs fails.

If clear pattern → propose regime-gated strategy.
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
    BETA_WINDOW, LGB_PARAMS, SEEDS, TOP_K,
    load_anchors, add_returns_and_basket,
    add_features_A, add_features_B,
)
from ml.research.alpha_v7_weekly import (
    HOLD_DAYS, COST_PER_TRADE_BPS,
    add_residual_5d, fit_predict, construct_portfolio_weekly,
    metrics_weekly, bootstrap_ci_weekly, make_folds,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def compute_regime_indicators(panel: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ts regime indicators based on universe-wide and macro state."""
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)

    # Per-symbol 22d return
    panel["_ret_22d"] = panel.groupby("symbol", group_keys=False)["close"].apply(
        lambda s: s.pct_change(22))

    # Per-ts: cross-sectional std of 22d returns (dispersion)
    by_ts = panel.groupby("ts").agg(
        disp_22d=("_ret_22d", "std"),
        univ_size=("symbol", "count"),
    ).reset_index()

    # Pairwise correlation: approximated by avg(corr_1d_vs_bk) is messy.
    # Simpler: rolling 60d corr of (ret_i, bk_ret) avg across symbols. Already
    # computed in v6_clean as corr_1d_vs_bk; recreate cheaply.
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    def _corr(g):
        return g["ret"].rolling(60).corr(g["bk_ret"])
    panel["_corr_60d"] = panel.groupby("symbol", group_keys=False).apply(_corr).values
    by_ts2 = panel.groupby("ts")["_corr_60d"].mean().rename("avg_corr_60d").reset_index()

    # Anchor-based regime indicators
    anchors = anchors.copy()
    anchors["ts"] = pd.to_datetime(anchors["ts"], utc=True).dt.normalize().astype("datetime64[ns, UTC]")
    if "VIX_close" in anchors.columns:
        anchors["VIX_level"] = anchors["VIX_close"]
    if "SOXX_close" in anchors.columns and "SPY_close" in anchors.columns:
        anchors["SOXX_minus_SPY_22d"] = (
            anchors["SOXX_close"].pct_change(22) - anchors["SPY_close"].pct_change(22)
        )
    if "XLK_close" in anchors.columns and "SPY_close" in anchors.columns:
        anchors["XLK_minus_SPY_22d"] = (
            anchors["XLK_close"].pct_change(22) - anchors["SPY_close"].pct_change(22)
        )
    if "TLT_close" in anchors.columns:
        anchors["TLT_ret_22d"] = anchors["TLT_close"].pct_change(22)
    keep = [c for c in ["ts", "VIX_level", "SOXX_minus_SPY_22d",
                         "XLK_minus_SPY_22d", "TLT_ret_22d"] if c in anchors.columns]
    anchors = anchors[keep]

    out = by_ts.merge(by_ts2, on="ts", how="left").merge(anchors, on="ts", how="left")
    return out


# ---- main --------------------------------------------------------------

def main() -> None:
    log.info("loading universe + earnings + cross-asset anchors...")
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
    log.info("regime cols: %s", list(regime.columns))

    # Run A+B walk-forward
    label = "fwd_resid_5d"
    pnl_label = "fwd_resid_5d"
    folds = make_folds(panel, train_min_days=365 * 3, test_days=365)
    feats = feats_A + feats_B + ["sym_id"]
    log.info("running A+B walk-forward (%d folds, %d features)...", len(folds), len(feats))

    all_pnls = []
    for fold in folds:
        train_end, test_start, test_end = fold
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        test_pred = fit_predict(train, test, feats, label)
        if test_pred.empty:
            continue
        lp = construct_portfolio_weekly(test_pred, "pred", pnl_label,
                                         top_k=TOP_K, cost_bps=COST_PER_TRADE_BPS,
                                         hold_days=HOLD_DAYS)
        if not lp.empty:
            all_pnls.append(lp)

    pnl = pd.concat(all_pnls, ignore_index=True)
    log.info("A+B stitched: n_rebal=%d", len(pnl))

    # Attach regime indicators at rebalance ts
    pnl = pnl.merge(regime, on="ts", how="left")

    # Overall metrics
    m = metrics_weekly(pnl)
    lo, hi = bootstrap_ci_weekly(pnl)
    log.info("OVERALL: gross=%+.2f bps/5d  net=%+.2f bps/5d  net_Sh=%+.2f  [%+.2f, %+.2f]  hit=%.0f%%",
             m["gross_bps_per_5d"], m["net_bps_per_5d"],
             m["net_sharpe_annu"], lo, hi, 100 * m["hit_rate"])

    # Per-regime quintiles
    regime_indicators = [
        ("disp_22d", "Cross-sectional dispersion (std of 22d returns across universe)"),
        ("avg_corr_60d", "Avg correlation to basket (HIGH = uniform; LOW = dispersion)"),
        ("VIX_level", "VIX level (vol regime)"),
        ("SOXX_minus_SPY_22d", "Semis vs SPY 22d (AI/semi trend strength)"),
        ("XLK_minus_SPY_22d", "Tech vs SPY 22d (broad-tech trend)"),
        ("TLT_ret_22d", "Bond 22d return (rate regime)"),
    ]

    for col, desc in regime_indicators:
        if col not in pnl.columns or pnl[col].isna().all():
            continue
        log.info("\n=== Bucket by %s ===  (%s)", col, desc)
        sub = pnl.dropna(subset=[col]).copy()
        sub["q"] = pd.qcut(sub[col], 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
                            duplicates="drop")
        log.info("  %-10s  %5s  %12s  %12s  %10s  %8s",
                 "quintile", "n", "gross/5d", "net/5d", "net_Sh", "hit")
        for q, g in sub.groupby("q"):
            qm = metrics_weekly(g)
            log.info("  %-10s  %5d  %+10.2fbps  %+10.2fbps  %+8.2f  %7.0f%%",
                     q, qm["n_rebal"], qm["gross_bps_per_5d"],
                     qm["net_bps_per_5d"], qm["net_sharpe_annu"],
                     100 * qm["hit_rate"])

    # Joint regime gating: best 2x2 cell among (dispersion, VIX) and (dispersion, SOXX-SPY)
    log.info("\n=== Joint regime: dispersion × VIX_level (above/below median) ===")
    if "VIX_level" in pnl.columns and "disp_22d" in pnl.columns:
        sub = pnl.dropna(subset=["disp_22d", "VIX_level"]).copy()
        d_med = sub["disp_22d"].median()
        v_med = sub["VIX_level"].median()
        sub["disp_hi"] = (sub["disp_22d"] >= d_med).astype(int)
        sub["vix_hi"] = (sub["VIX_level"] >= v_med).astype(int)
        log.info("  %-30s  %5s  %12s  %12s  %10s",
                 "regime", "n", "gross/5d", "net/5d", "net_Sh")
        for (d, v), g in sub.groupby(["disp_hi", "vix_hi"]):
            qm = metrics_weekly(g)
            tag = f"disp={'HI' if d else 'LO'}, vix={'HI' if v else 'LO'}"
            log.info("  %-30s  %5d  %+10.2fbps  %+10.2fbps  %+8.2f",
                     tag, qm["n_rebal"], qm["gross_bps_per_5d"],
                     qm["net_bps_per_5d"], qm["net_sharpe_annu"])

    log.info("\n=== Joint regime: dispersion × SOXX_minus_SPY_22d (above/below median) ===")
    if "SOXX_minus_SPY_22d" in pnl.columns and "disp_22d" in pnl.columns:
        sub = pnl.dropna(subset=["disp_22d", "SOXX_minus_SPY_22d"]).copy()
        d_med = sub["disp_22d"].median()
        s_med = sub["SOXX_minus_SPY_22d"].median()
        sub["disp_hi"] = (sub["disp_22d"] >= d_med).astype(int)
        sub["sx_hi"] = (sub["SOXX_minus_SPY_22d"] >= s_med).astype(int)
        log.info("  %-30s  %5s  %12s  %12s  %10s",
                 "regime", "n", "gross/5d", "net/5d", "net_Sh")
        for (d, s), g in sub.groupby(["disp_hi", "sx_hi"]):
            qm = metrics_weekly(g)
            tag = f"disp={'HI' if d else 'LO'}, sx={'HI' if s else 'LO'}"
            log.info("  %-30s  %5d  %+10.2fbps  %+10.2fbps  %+8.2f",
                     tag, qm["n_rebal"], qm["gross_bps_per_5d"],
                     qm["net_bps_per_5d"], qm["net_sharpe_annu"])

    # Identify "best regime" — highest Sharpe quintile across primary indicator
    log.info("\n=== If we ONLY trade in top-2 dispersion quintiles ===")
    if "disp_22d" in pnl.columns:
        sub = pnl.dropna(subset=["disp_22d"]).copy()
        sub["q"] = pd.qcut(sub["disp_22d"], 5, labels=False, duplicates="drop")
        gated = sub[sub["q"] >= 3]  # top 2 quintiles
        m = metrics_weekly(gated)
        lo, hi = bootstrap_ci_weekly(gated)
        log.info("  GATED n=%d (%.0f%% of original) gross=%+.2f bps/5d  net=%+.2f  net_Sh=%+.2f  [%+.2f, %+.2f]",
                 m["n_rebal"], 100 * m["n_rebal"] / len(pnl),
                 m["gross_bps_per_5d"], m["net_bps_per_5d"],
                 m["net_sharpe_annu"], lo, hi)


if __name__ == "__main__":
    main()
