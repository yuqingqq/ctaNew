"""v7 multi-alpha weekly v2 — redesigned C and D features for weekly cadence.

Baseline (A+B from v7_weekly): net Sharpe +0.40 [-0.10, +0.97].
v1 result showed C and D HURT at weekly because their features were daily-tuned.

Redesigns:
  C2 (cross-asset, weekly-tuned, 12 features):
    - LEVEL features for regime (not 5d returns): VIX level, TNX yield level
    - 22d-window returns (not 5d): SPY/TLT/GLD/USD 22d returns
    - Sector-rotation: SOXX-SPY spread, XLK-SPY spread
    - Per-name regime betas: TLT_beta_60d, VIX_beta_60d, SOXX_beta_60d

  D2 (calendar, weekly-tuned, 7 features):
    - Drop dow (meaningless — always rebalance Mondays)
    - Keep dom, month_end, year_end
    - Add quarter_end, fomc_week, jan_effect, earnings_season

Compares:
  - A+B (baseline, expected +0.40)
  - A+B+C2 (does redesigned C add value?)
  - A+B+D2 (does redesigned D add value?)
  - A+B+C2+D2 (full redesign)
  - C2 only / D2 only (sanity check)
"""
from __future__ import annotations

import logging
import warnings
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_collectors.sp100_loader import load_universe
from ml.research.alpha_v7_multi import (
    BETA_WINDOW, LGB_PARAMS, SEEDS, TOP_K, PEAD_MAX_DAYS,
    load_anchors, add_returns_and_basket,
    add_features_A, add_features_B,
)
from ml.research.alpha_v7_weekly import (
    FWD_DAYS, HOLD_DAYS, COST_PER_TRADE_BPS,
    add_residual_5d, fit_predict, construct_portfolio_weekly,
    metrics_weekly, bootstrap_ci_weekly, make_folds,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---- C2: redesigned cross-asset features for weekly cadence -----------

def add_features_C2(panel: pd.DataFrame, anchors: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Cross-asset features tuned for 5-day forward prediction.

    Drops 5-day returns (noise at weekly horizon). Adds level-based regime
    features and longer-window (22d) returns. Adds per-name beta to key
    anchors as regime-conditioning features.
    """
    anchors = anchors.copy()
    anchors["ts"] = pd.to_datetime(anchors["ts"], utc=True).dt.normalize().astype("datetime64[ns, UTC]")
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True).astype("datetime64[ns, UTC]")
    panel = panel.merge(anchors, on="ts", how="left")
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)

    feats = []

    # --- LEVEL features (regime indicators, not returns) ---
    # VIX level + z-score
    if "VIX_close" in panel.columns:
        panel["C2_VIX_level"] = panel["VIX_close"]
        panel["C2_VIX_z_60d"] = (
            (panel["VIX_close"] - panel["VIX_close"].rolling(60).mean())
            / panel["VIX_close"].rolling(60).std().replace(0, np.nan)
        ).clip(-5, 5)
        feats.extend(["C2_VIX_level", "C2_VIX_z_60d"])

    # 10y yield level + 22d change (rate-regime indicators)
    if "TNX_close" in panel.columns:
        panel["C2_TNX_level"] = panel["TNX_close"]
        panel["C2_TNX_change_22d"] = panel["TNX_close"] - panel["TNX_close"].shift(22)
        feats.extend(["C2_TNX_level", "C2_TNX_change_22d"])

    # --- 22d returns of major asset class anchors (regime persistence) ---
    for a in ("SPY", "TLT", "UUP", "GLD"):
        col = f"{a}_close"
        if col not in panel.columns:
            continue
        panel[f"C2_{a}_ret_22d"] = panel[col].pct_change(22)
        feats.append(f"C2_{a}_ret_22d")

    # --- Sector rotation (semi vs market, tech vs market) ---
    if all(c in panel.columns for c in ["SOXX_close", "SPY_close"]):
        panel["C2_SOXX_minus_SPY_22d"] = (
            panel["SOXX_close"].pct_change(22) - panel["SPY_close"].pct_change(22)
        )
        feats.append("C2_SOXX_minus_SPY_22d")
    if all(c in panel.columns for c in ["XLK_close", "SPY_close"]):
        panel["C2_XLK_minus_SPY_22d"] = (
            panel["XLK_close"].pct_change(22) - panel["SPY_close"].pct_change(22)
        )
        feats.append("C2_XLK_minus_SPY_22d")

    # --- Per-name beta to key anchors (slow-moving regime conditioning) ---
    def _rolling_beta_to(g_, anchor_ret_col):
        ret = g_["ret"]
        bret = g_[anchor_ret_col]
        cov = (ret * bret).rolling(BETA_WINDOW).mean() - \
              ret.rolling(BETA_WINDOW).mean() * bret.rolling(BETA_WINDOW).mean()
        var = bret.rolling(BETA_WINDOW).var().replace(0, np.nan)
        return (cov / var).clip(-5, 5).shift(1)

    for anchor, fname in [("TLT", "C2_TLT_beta_60d"),
                          ("VIX", "C2_VIX_beta_60d"),
                          ("SOXX", "C2_SOXX_beta_60d")]:
        col_ret = f"{anchor}_ret"
        if col_ret in panel.columns:
            panel[fname] = (g.apply(lambda gg, c=col_ret: _rolling_beta_to(gg, c))
                            .reset_index(level=0, drop=True))
            feats.append(fname)

    return panel, feats


# ---- D2: redesigned calendar features for weekly cadence --------------

# FOMC scheduled meetings (regular cycle, ~8/year). Kept approximate by
# hardcoding the typical date in each month they meet. Actual meetings
# fall within ±3 days of these typical dates.
FOMC_MEETING_MONTHS = {1, 3, 5, 6, 7, 9, 11, 12}  # 8 meetings/year typically
# Earnings season months for US large-caps: peak weeks are mid-month
EARNINGS_SEASON_MONTHS = {1, 4, 7, 10}


def add_features_D2(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Calendar features tuned for weekly rebalance.

    Removes day_of_week (irrelevant when always rebalancing Monday).
    Adds quarter-end, FOMC-week, January-effect, earnings-season indicators.
    """
    et = panel["ts"].dt.tz_convert("America/New_York")
    panel["D2_dom"] = et.dt.day.astype(float)
    panel["D2_is_month_end"] = (et.dt.day >= 25).astype(float)
    panel["D2_is_quarter_end"] = (
        et.dt.month.isin([3, 6, 9, 12]) & (et.dt.day >= 20)
    ).astype(float)
    panel["D2_is_year_end"] = (
        (et.dt.month == 12) & (et.dt.day >= 15)
    ).astype(float)
    # FOMC week heuristic: approximate week of meeting (mid-month for FOMC months)
    # FOMC meetings typically fall in 3rd or 4th week of meeting month
    panel["D2_is_fomc_week"] = (
        et.dt.month.isin(list(FOMC_MEETING_MONTHS))
        & et.dt.day.between(15, 25)
    ).astype(float)
    panel["D2_is_jan"] = (et.dt.month == 1).astype(float)
    panel["D2_is_earnings_season"] = (
        et.dt.month.isin(list(EARNINGS_SEASON_MONTHS))
        & et.dt.day.between(15, 31)
    ).astype(float)
    return panel, [
        "D2_dom", "D2_is_month_end", "D2_is_quarter_end", "D2_is_year_end",
        "D2_is_fomc_week", "D2_is_jan", "D2_is_earnings_season",
    ]


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
    panel, feats_C2 = add_features_C2(panel, anchors)
    panel, feats_D2 = add_features_D2(panel)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes

    log.info("feature groups: A=%d  B=%d  C2=%d  D2=%d",
             len(feats_A), len(feats_B), len(feats_C2), len(feats_D2))

    label = "fwd_resid_5d"
    pnl_label = "fwd_resid_5d"
    folds = make_folds(panel, train_min_days=365 * 3, test_days=365)
    log.info("folds: %d", len(folds))

    ablation_configs = {
        "A+B (baseline)": feats_A + feats_B + ["sym_id"],
        "C2 only": feats_C2 + ["sym_id"],
        "D2 only": feats_D2 + ["sym_id"],
        "A+B+C2": feats_A + feats_B + feats_C2 + ["sym_id"],
        "A+B+D2": feats_A + feats_B + feats_D2 + ["sym_id"],
        "A+B+C2+D2": feats_A + feats_B + feats_C2 + feats_D2 + ["sym_id"],
    }

    results = {}
    for cfg_name, feats in ablation_configs.items():
        log.info("\n>>> CONFIG: %s  (n_features=%d)", cfg_name, len(feats))
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
                lp["fold"] = fold[1].year
                all_pnls.append(lp)
        if not all_pnls:
            continue
        st = pd.concat(all_pnls, ignore_index=True)
        m = metrics_weekly(st)
        lo, hi = bootstrap_ci_weekly(st)
        results[cfg_name] = (m, lo, hi, st)
        log.info("  STITCHED: n_rebal=%d gross=%+.2fbps/5d (%.2fbps/d) net=%+.2fbps/5d "
                 "net_Sh=%+.2f  [%+.2f, %+.2f]  hit=%.0f%%",
                 m["n_rebal"], m["gross_bps_per_5d"], m["gross_bps_per_day"],
                 m["net_bps_per_5d"], m["net_sharpe_annu"], lo, hi,
                 100 * m["hit_rate"])

    log.info("\n=== ABLATION SUMMARY (cost=%d bps/trade-side, top_k=%d, weekly) ===",
             COST_PER_TRADE_BPS, TOP_K)
    log.info("  %-22s %5s %10s %10s %10s %18s",
             "config", "n_reb", "gross/5d", "net/5d", "net_Sh", "95% CI")
    for cfg, (m, lo, hi, _) in results.items():
        log.info("  %-22s %5d %+8.2fbps %+8.2fbps %+10.2f  [%+.2f, %+.2f]",
                 cfg, m["n_rebal"], m["gross_bps_per_5d"],
                 m["net_bps_per_5d"], m["net_sharpe_annu"], lo, hi)

    # Per-year breakdown for the best config
    if results:
        best_name = max(results.keys(), key=lambda k: results[k][0]["net_sharpe_annu"])
        log.info("\n=== Per-year breakdown for best config: %s ===", best_name)
        st = results[best_name][3].copy()
        st["year"] = st["ts"].dt.year
        log.info("  %-6s %5s %12s %12s %10s %8s",
                 "year", "n_reb", "gross/5d", "net/5d", "net_Sh", "hit")
        for y, g in st.groupby("year"):
            m = metrics_weekly(g)
            log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+8.2f %7.0f%%",
                     y, m["n_rebal"], m["gross_bps_per_5d"],
                     m["net_bps_per_5d"], m["net_sharpe_annu"],
                     100 * m["hit_rate"])


if __name__ == "__main__":
    main()
