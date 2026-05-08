"""A+B + dispersion gating restricted to xyz-tradeable universe.

xyz US equity perps: 33 names total. After filtering for 5y+ history (drops
post-2020 IPOs), usable backtest universe is ~24 names.

Tests 3 configurations:
  Config 1: train AND trade on xyz subset (~24 names) — pure xyz strategy
  Config 2: train on full S&P 100 (~100), trade only xyz subset — broader
            training cross-section, narrower execution
  Config 3: S&P 100 baseline (already tested, Sharpe +0.92 with gating)

Hypothesis: Config 2 should outperform Config 1 because more training data
helps the model learn better, and execution is just a subset filter at
prediction time. Config 1 may suffer from small-universe noise but is the
"pure" xyz strategy.

Each config: A+B features, weekly rebalance, dispersion-gated (top 40%).
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
from ml.research.alpha_v7_regime import compute_regime_indicators

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# xyz US equity universe (33 names from earlier survey).
# yfinance ticker form for those needing translation.
XYZ_US_EQUITY = [
    "AAPL", "AMD", "AMZN", "BX", "COIN", "COST", "CRCL", "CRWV",
    "DKNG", "EBAY", "GME", "GOOGL", "HIMS", "HOOD", "INTC", "LITE",
    "LLY", "META", "MRVL", "MSFT", "MSTR", "MU", "NFLX", "NVDA",
    "ORCL", "PLTR", "RIVN", "RKLB", "SNDK", "TSLA", "USAR", "BIRD",
    "ZM",
]
# Names with insufficient history for 13y backtest (post-2020 IPO etc.) — auto-filtered

# Smaller K for smaller universe
TOP_K_XYZ = 5    # vs 10 for S&P 100


# ---- portfolio with subset filter --------------------------------------

def construct_portfolio_subset(test_pred: pd.DataFrame, signal: str,
                                pnl_label: str, allowed_symbols: set,
                                top_k: int = TOP_K_XYZ,
                                cost_bps: float = COST_PER_TRADE_BPS,
                                hold_days: int = HOLD_DAYS) -> pd.DataFrame:
    """Same as construct_portfolio_weekly but restricts to allowed symbols at
    portfolio construction time. Predictions can be made on full universe
    but only allowed symbols are eligible for the long/short legs."""
    sub = test_pred[test_pred["symbol"].isin(allowed_symbols)].dropna(
        subset=[signal, pnl_label]).copy()
    unique_ts = sorted(sub["ts"].unique())
    if not unique_ts:
        return pd.DataFrame()
    rebal_ts = unique_ts[::hold_days]
    rows = []
    prev_long: set = set()
    prev_short: set = set()
    for ts in rebal_ts:
        bar = sub[sub["ts"] == ts]
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values(signal)
        long_leg = set(bar.tail(top_k)["symbol"])
        short_leg = set(bar.head(top_k)["symbol"])
        long_changes = len(long_leg.symmetric_difference(prev_long))
        short_changes = len(short_leg.symmetric_difference(prev_short))
        turnover = (long_changes + short_changes) / (2 * top_k)
        cost = turnover * cost_bps / 1e4
        long_alpha = bar[bar["symbol"].isin(long_leg)][pnl_label].mean()
        short_alpha = bar[bar["symbol"].isin(short_leg)][pnl_label].mean()
        spread = long_alpha - short_alpha
        rows.append({
            "ts": ts, "spread_alpha": spread,
            "long_alpha": long_alpha, "short_alpha": short_alpha,
            "turnover": turnover, "cost": cost,
            "net_alpha": spread - cost, "n_universe": len(bar),
        })
        prev_long, prev_short = long_leg, short_leg
    return pd.DataFrame(rows)


def gate_by_dispersion(pnl: pd.DataFrame, regime: pd.DataFrame,
                        threshold_pctile: float = 0.6) -> pd.DataFrame:
    """Keep only rebalances where dispersion is above the in-sample
    `threshold_pctile`-th percentile (NOTE: in-sample percentile is a
    look-ahead approximation; live would use trailing percentile)."""
    sub = pnl.merge(regime[["ts", "disp_22d"]], on="ts", how="left")
    if sub["disp_22d"].isna().all():
        return sub
    thresh = sub["disp_22d"].dropna().quantile(threshold_pctile)
    return sub[sub["disp_22d"] >= thresh].copy()


# ---- main --------------------------------------------------------------

def main() -> None:
    log.info("loading S&P 100 universe + earnings + cross-asset anchors...")
    panel, earnings, surv = load_universe()
    if panel.empty:
        return
    anchors = load_anchors()

    # Keep only S&P 100 names that overlap with xyz, plus xyz-only
    # NB: xyz names not in S&P 100 (CRCL, MSTR, MRVL, GME, RIVN, etc.) we'd
    # need to fetch separately. For this probe we use the OVERLAP — xyz US
    # equity names that are also in S&P 100. This gives a clean apples-to-
    # apples comparison without needing additional data fetches.
    sp100_set = set(surv)
    xyz_in_sp100 = [s for s in XYZ_US_EQUITY if s in sp100_set]
    log.info("xyz US equities: %d total in xyz, %d also in S&P 100",
             len(XYZ_US_EQUITY), len(xyz_in_sp100))
    log.info("  in-overlap: %s", xyz_in_sp100)
    not_in_overlap = sorted(set(XYZ_US_EQUITY) - sp100_set)
    log.info("  not yet fetched (would need extra fetch): %s", not_in_overlap)

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

    # === Config 2: train full S&P 100, trade only xyz overlap ===
    log.info("\n>>> Config 2: train S&P100 (%d), trade xyz overlap (%d)",
             len(sp100_set), len(xyz_in_sp100))
    all_pnls_c2 = []
    for fold in folds:
        train_end, test_start, test_end = fold
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        test_pred = fit_predict(train, test, feats, label)
        if test_pred.empty:
            continue
        # K=5 for smaller universe
        lp = construct_portfolio_subset(
            test_pred, "pred", pnl_label,
            allowed_symbols=set(xyz_in_sp100),
            top_k=TOP_K_XYZ, cost_bps=COST_PER_TRADE_BPS,
            hold_days=HOLD_DAYS,
        )
        if not lp.empty:
            all_pnls_c2.append(lp)
    pnl_c2 = pd.concat(all_pnls_c2, ignore_index=True) if all_pnls_c2 else pd.DataFrame()

    # === Config 1: train xyz only, trade xyz only ===
    log.info("\n>>> Config 1: train xyz overlap (%d), trade xyz overlap (%d)",
             len(xyz_in_sp100), len(xyz_in_sp100))
    panel_xyz = panel[panel["symbol"].isin(xyz_in_sp100)].copy()
    log.info("  xyz panel: %d rows", len(panel_xyz))
    all_pnls_c1 = []
    for fold in folds:
        train_end, test_start, test_end = fold
        train = panel_xyz[panel_xyz["ts"] <= train_end].copy()
        test = panel_xyz[(panel_xyz["ts"] >= test_start) & (panel_xyz["ts"] <= test_end)].copy()
        test_pred = fit_predict(train, test, feats, label)
        if test_pred.empty:
            continue
        lp = construct_portfolio_subset(
            test_pred, "pred", pnl_label,
            allowed_symbols=set(xyz_in_sp100),
            top_k=TOP_K_XYZ, cost_bps=COST_PER_TRADE_BPS,
            hold_days=HOLD_DAYS,
        )
        if not lp.empty:
            all_pnls_c1.append(lp)
    pnl_c1 = pd.concat(all_pnls_c1, ignore_index=True) if all_pnls_c1 else pd.DataFrame()

    # === Config 3: S&P 100 full (baseline) ===
    log.info("\n>>> Config 3: train S&P100, trade S&P100 (baseline)")
    all_pnls_c3 = []
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
            all_pnls_c3.append(lp)
    pnl_c3 = pd.concat(all_pnls_c3, ignore_index=True) if all_pnls_c3 else pd.DataFrame()

    # === Report ===
    def report(name: str, pnl: pd.DataFrame) -> None:
        if pnl.empty:
            log.info("  %s: NO RESULTS", name); return
        m = metrics_weekly(pnl)
        lo, hi = bootstrap_ci_weekly(pnl)
        log.info("  %-50s  n=%4d gross=%+.2f bps/5d net=%+.2f bps/5d  net_Sh=%+.2f  [%+.2f, %+.2f]",
                 name, m["n_rebal"], m["gross_bps_per_5d"],
                 m["net_bps_per_5d"], m["net_sharpe_annu"], lo, hi)

    log.info("\n=== UNGATED RESULTS ===")
    report("C1: train xyz, trade xyz", pnl_c1)
    report("C2: train S&P100, trade xyz", pnl_c2)
    report("C3: train S&P100, trade S&P100 (baseline)", pnl_c3)

    log.info("\n=== DISPERSION-GATED RESULTS (top 40%) ===")
    report("C1 + gate: train xyz, trade xyz",
           gate_by_dispersion(pnl_c1, regime, threshold_pctile=0.6))
    report("C2 + gate: train S&P100, trade xyz",
           gate_by_dispersion(pnl_c2, regime, threshold_pctile=0.6))
    report("C3 + gate: train S&P100, trade S&P100",
           gate_by_dispersion(pnl_c3, regime, threshold_pctile=0.6))

    log.info("\n=== Per-year for best gated config ===")
    candidates = {
        "C1+gate": gate_by_dispersion(pnl_c1, regime, 0.6),
        "C2+gate": gate_by_dispersion(pnl_c2, regime, 0.6),
        "C3+gate": gate_by_dispersion(pnl_c3, regime, 0.6),
    }
    if any(not p.empty for p in candidates.values()):
        best = max(candidates.items(),
                   key=lambda kv: metrics_weekly(kv[1]).get("net_sharpe_annu", -10) if not kv[1].empty else -10)
        log.info("best=%s", best[0])
        st = best[1].copy()
        st["year"] = st["ts"].dt.year
        log.info("  %-6s %5s %12s %12s %10s %8s",
                 "year", "n_reb", "gross/5d", "net/5d", "net_Sh", "hit")
        for y, g in st.groupby("year"):
            qm = metrics_weekly(g)
            log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+8.2f %7.0f%%",
                     y, qm["n_rebal"], qm["gross_bps_per_5d"],
                     qm["net_bps_per_5d"], qm["net_sharpe_annu"],
                     100 * qm["hit_rate"])


if __name__ == "__main__":
    main()
