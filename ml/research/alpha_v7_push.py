"""Push the +2.67 ensemble further with 3 combined optimizations.

  Push 1: Larger ensemble (5 horizons × 5 seeds = 25 models)
          horizons = [3, 5, 7, 10, 15] (we tested h ∈ same range earlier)
  Push 2: Add sector momentum features
          - own_sector_mom_5d: 5-day return of own GICS sector basket
          - own_sector_mom_22d: 22-day return of own sector
          - sector_relative_mom_5d: name's 5d return minus sector 5d return
  Push 3: Add days_until_earnings feature (anticipation effect)

Compare to baseline +2.67 multi-horizon ensemble. If combined lift > +0.2,
ship the new spec; if not, +2.67 is at the ceiling and we stop here.

All other parameters frozen at production values:
  - Hysteresis M=2, K=5
  - Gate: dispersion ≥ 60th pctile of trailing 252d
  - Cost: 1.5 bps/side patient maker
  - Daily rebalance, 1d hold (P&L), trained on slow signals
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay

from data_collectors.sp100_loader import load_universe, fetch_earnings
from ml.research.alpha_v7_multi import (
    LGB_PARAMS, SEEDS, add_returns_and_basket, add_features_A, load_anchors,
)
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100
from ml.research.alpha_v7_freq_sweep import (
    add_residual_and_label, metrics_freq, annualized_unconditional,
)
from ml.research.alpha_v7_pead_fixed import add_features_B_fixed
from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_daily_optimized import (
    daily_portfolio_hysteresis, make_folds, metrics_for, boot_ci,
)
from ml.research.alpha_v7_daily_v2 import run_walk_multihorizon

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOP_K = 5
COST_BPS_SIDE = 1.5
GATE_PCTILE = 0.6
GATE_WINDOW = 252


# ---- GICS sector mapping for S&P 100 (manual, current as of 2026) ----
SECTOR_MAP = {
    # Information Technology
    "AAPL": "IT", "MSFT": "IT", "NVDA": "IT", "AMD": "IT", "INTC": "IT",
    "MU": "IT", "ORCL": "IT", "CSCO": "IT", "AVGO": "IT", "ADBE": "IT",
    "CRM": "IT", "QCOM": "IT", "TXN": "IT", "ACN": "IT", "IBM": "IT",
    "INTU": "IT", "AMAT": "IT", "NOW": "IT", "LRCX": "IT", "PLTR": "IT",
    # Health Care
    "JNJ": "HC", "UNH": "HC", "LLY": "HC", "ABBV": "HC", "MRK": "HC",
    "PFE": "HC", "AMGN": "HC", "BMY": "HC", "GILD": "HC", "DHR": "HC",
    "TMO": "HC", "ABT": "HC", "MDT": "HC", "ISRG": "HC", "CVS": "HC",
    # Financials
    "JPM": "FIN", "BAC": "FIN", "WFC": "FIN", "GS": "FIN", "MS": "FIN",
    "BLK": "FIN", "SCHW": "FIN", "AXP": "FIN", "C": "FIN", "USB": "FIN",
    "COF": "FIN", "BK": "FIN", "BRK-B": "FIN", "MA": "FIN", "V": "FIN",
    # Consumer Discretionary
    "AMZN": "CD", "TSLA": "CD", "HD": "CD", "MCD": "CD", "NKE": "CD",
    "LOW": "CD", "BKNG": "CD", "SBUX": "CD", "GM": "CD", "UBER": "CD",
    # Communication Services
    "GOOGL": "COM", "GOOG": "COM", "META": "COM", "NFLX": "COM",
    "DIS": "COM", "T": "COM", "VZ": "COM", "CMCSA": "COM", "TMUS": "COM",
    # Consumer Staples
    "WMT": "CS", "COST": "CS", "KO": "CS", "PEP": "CS", "PG": "CS",
    "MO": "CS", "MDLZ": "CS", "PM": "CS", "CL": "CS",
    # Industrials
    "BA": "IND", "RTX": "IND", "HON": "IND", "GE": "IND", "UNP": "IND",
    "UPS": "IND", "CAT": "IND", "DE": "IND", "LMT": "IND", "FDX": "IND",
    "GD": "IND", "EMR": "IND", "MMM": "IND",
    # Energy / Materials / Real Estate / Utilities
    "XOM": "EN", "CVX": "EN", "COP": "EN",
    "LIN": "MAT",
    "AMT": "RE", "SPG": "RE",
    "NEE": "UTL", "DUK": "UTL", "SO": "UTL",
}


def add_sector_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute per-day sector basket returns, then per-name features."""
    panel = panel.copy()
    panel["sector"] = panel["symbol"].map(SECTOR_MAP)
    panel["sector"] = panel["sector"].fillna("OTH")

    # Per-day sector basket returns (equal-weight within sector)
    sector_ret = (panel.groupby(["ts", "sector"])["ret"].mean()
                   .reset_index().rename(columns={"ret": "sector_ret"}))
    panel = panel.merge(sector_ret, on=["ts", "sector"], how="left")

    # Per-name sector features
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)
    # Cumulative own-sector log-equivalent (sum of bp-returns over rolling window)
    panel["F_sector_mom_5d"] = (g["sector_ret"]
                                 .apply(lambda s: s.rolling(5).sum().shift(1)))
    panel["F_sector_mom_22d"] = (g["sector_ret"]
                                  .apply(lambda s: s.rolling(22).sum().shift(1)))
    # Name's 5d return relative to own sector
    panel["F_sector_relative_5d"] = (
        g["ret"].apply(lambda s: s.rolling(5).sum().shift(1))
        - g["sector_ret"].apply(lambda s: s.rolling(5).sum().shift(1))
    )
    return panel, ["F_sector_mom_5d", "F_sector_mom_22d", "F_sector_relative_5d"]


# ---- days_until_earnings feature ---------------------------------------

def add_days_until_earnings(panel: pd.DataFrame, earnings: pd.DataFrame,
                              cap_days: int = 90) -> tuple[pd.DataFrame, list[str]]:
    """For each (sym, day), compute days until next earnings event.
    Capped at cap_days (NaN → no upcoming earnings within cap window)."""
    earn = earnings.copy()
    earn["ts_et"] = pd.to_datetime(earn["ts"], utc=True).dt.tz_convert("America/New_York")
    # Effective announcement date (if AMC, the price reaction is next BDay,
    # but for "days until anticipation" we use the announcement date itself).
    earn["effective_date"] = earn["ts_et"].dt.normalize()
    earn["effective_date"] = earn["effective_date"].dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
    earn = earn.sort_values(["symbol", "effective_date"]).reset_index(drop=True)

    panel = panel.copy()
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True).astype("datetime64[ns, UTC]")
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)

    out_chunks = []
    for sym, g in panel.groupby("symbol"):
        e = earn[earn["symbol"] == sym][["effective_date"]].copy()
        if e.empty:
            g = g.copy()
            g["F_days_until_earn"] = np.nan
            out_chunks.append(g)
            continue
        e = e.sort_values("effective_date").rename(columns={"effective_date": "next_earn"})
        merged = pd.merge_asof(
            g.sort_values("ts"), e,
            left_on="ts", right_on="next_earn",
            allow_exact_matches=True, direction="forward",
        )
        merged["F_days_until_earn"] = (merged["next_earn"] - merged["ts"]).dt.days
        # Cap at cap_days; NaN if > cap
        merged.loc[merged["F_days_until_earn"] > cap_days, "F_days_until_earn"] = np.nan
        out_chunks.append(merged)
    panel_out = pd.concat(out_chunks, ignore_index=True)
    return panel_out, ["F_days_until_earn"]


def main() -> None:
    log.info("loading panel...")
    panel, earnings, _ = load_universe()
    if panel.empty: return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    for h in (1, 3, 5, 7, 10, 15):
        panel = add_residual_and_label(panel, h)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)

    # Push 2: sector features
    log.info("Adding sector momentum features...")
    panel, feats_F_sector = add_sector_features(panel)
    n_sector_valid = panel["F_sector_mom_5d"].notna().sum()
    log.info("  sector features: %d/%d non-null (%.0f%%)",
             n_sector_valid, len(panel), 100 * n_sector_valid / len(panel))

    # Push 3: days_until_earnings
    log.info("Adding days_until_earnings feature...")
    panel, feats_F_due = add_days_until_earnings(panel, earnings)
    n_due_valid = panel["F_days_until_earn"].notna().sum()
    log.info("  days_until_earn: %d/%d non-null (%.0f%%)",
             n_due_valid, len(panel), 100 * n_due_valid / len(panel))

    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    feats_baseline = feats_A + feats_B + ["sym_id"]
    feats_pushed = feats_A + feats_B + feats_F_sector + feats_F_due + ["sym_id"]
    allowed = set(XYZ_IN_SP100)
    folds = make_folds(panel)

    # Push 1: 5-horizon × 5-seed ensemble (25 models)
    train_labels_baseline = ["fwd_resid_3d", "fwd_resid_5d", "fwd_resid_10d"]
    train_labels_pushed = ["fwd_resid_3d", "fwd_resid_5d",
                            "fwd_resid_7d", "fwd_resid_10d", "fwd_resid_15d"]

    log.info("\n=== Tier A combined push ===")
    log.info("  baseline: %d features, %d horizons (3,5,10), 5 seeds = 15 models",
             len(feats_baseline), len(train_labels_baseline))
    log.info("  pushed:   %d features (+sector +days_until), %d horizons (3,5,7,10,15), 5 seeds = 25 models",
             len(feats_pushed), len(train_labels_pushed))

    log.info("\n>>> BASELINE (current production +2.67): A+B, 3-horizon ensemble")
    pnl0_pre = run_walk_multihorizon(
        panel, feats_baseline, train_labels_baseline, folds,
        daily_portfolio_hysteresis,
        {"pnl_label": "fwd_resid_1d", "allowed": allowed,
         "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": COST_BPS_SIDE},
    )
    pnl0 = gate_rolling(pnl0_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m0 = metrics_for(pnl0, 1)
    lo0, hi0 = boot_ci(pnl0, 1)
    log.info("  n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f ann=%+.2f%%",
             m0["n_rebal"], m0["active_sharpe"], lo0, hi0,
             m0["uncond_sharpe"], m0["annual_return_pct"])

    log.info("\n>>> PUSH 1 only: 5-horizon ensemble, baseline features")
    pnl1_pre = run_walk_multihorizon(
        panel, feats_baseline, train_labels_pushed, folds,
        daily_portfolio_hysteresis,
        {"pnl_label": "fwd_resid_1d", "allowed": allowed,
         "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": COST_BPS_SIDE},
    )
    pnl1 = gate_rolling(pnl1_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m1 = metrics_for(pnl1, 1)
    lo1, hi1 = boot_ci(pnl1, 1)
    log.info("  n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f ann=%+.2f%%",
             m1["n_rebal"], m1["active_sharpe"], lo1, hi1,
             m1["uncond_sharpe"], m1["annual_return_pct"])

    log.info("\n>>> PUSH 2 only: baseline ensemble + sector features")
    feats_p2 = feats_A + feats_B + feats_F_sector + ["sym_id"]
    pnl2_pre = run_walk_multihorizon(
        panel, feats_p2, train_labels_baseline, folds,
        daily_portfolio_hysteresis,
        {"pnl_label": "fwd_resid_1d", "allowed": allowed,
         "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": COST_BPS_SIDE},
    )
    pnl2 = gate_rolling(pnl2_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m2 = metrics_for(pnl2, 1)
    lo2, hi2 = boot_ci(pnl2, 1)
    log.info("  n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f ann=%+.2f%%",
             m2["n_rebal"], m2["active_sharpe"], lo2, hi2,
             m2["uncond_sharpe"], m2["annual_return_pct"])

    log.info("\n>>> PUSH 3 only: baseline ensemble + days_until_earn")
    feats_p3 = feats_A + feats_B + feats_F_due + ["sym_id"]
    pnl3_pre = run_walk_multihorizon(
        panel, feats_p3, train_labels_baseline, folds,
        daily_portfolio_hysteresis,
        {"pnl_label": "fwd_resid_1d", "allowed": allowed,
         "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": COST_BPS_SIDE},
    )
    pnl3 = gate_rolling(pnl3_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m3 = metrics_for(pnl3, 1)
    lo3, hi3 = boot_ci(pnl3, 1)
    log.info("  n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f ann=%+.2f%%",
             m3["n_rebal"], m3["active_sharpe"], lo3, hi3,
             m3["uncond_sharpe"], m3["annual_return_pct"])

    log.info("\n>>> PUSH ALL: 5-horizon + sector + days_until")
    pnl_all_pre = run_walk_multihorizon(
        panel, feats_pushed, train_labels_pushed, folds,
        daily_portfolio_hysteresis,
        {"pnl_label": "fwd_resid_1d", "allowed": allowed,
         "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": COST_BPS_SIDE},
    )
    pnl_all = gate_rolling(pnl_all_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m_all = metrics_for(pnl_all, 1)
    lo_all, hi_all = boot_ci(pnl_all, 1)
    log.info("  n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f ann=%+.2f%%",
             m_all["n_rebal"], m_all["active_sharpe"], lo_all, hi_all,
             m_all["uncond_sharpe"], m_all["annual_return_pct"])

    # Summary
    log.info("\n=== SUMMARY ===")
    log.info("  %-40s  %5s  %10s  %18s  %10s",
             "config", "n_reb", "active_Sh", "95% CI", "uncond_Sh")
    for name, m, lo, hi in [
        ("BASELINE (3 horizons, A+B)", m0, lo0, hi0),
        ("+ 5 horizons (push 1)", m1, lo1, hi1),
        ("+ sector features (push 2)", m2, lo2, hi2),
        ("+ days_until_earn (push 3)", m3, lo3, hi3),
        ("+ ALL pushes combined", m_all, lo_all, hi_all),
    ]:
        if not m: continue
        log.info("  %-40s  %5d  %+8.2f  [%+5.2f,%+5.2f]  %+8.2f",
                 name, m["n_rebal"], m["active_sharpe"], lo, hi, m["uncond_sharpe"])

    # Per-year for combined
    if not pnl_all.empty:
        pnl_all_y = pnl_all.copy()
        pnl_all_y["year"] = pnl_all_y["ts"].dt.year
        log.info("\n=== Per-year for ALL pushes combined ===")
        log.info("  %-6s %5s %12s %10s %8s",
                 "year", "n_reb", "net/d", "active_Sh", "hit%")
        for y, g in pnl_all_y.groupby("year"):
            ym = metrics_for(g, 1)
            hit = (g["spread_alpha"] > 0).mean()
            log.info("  %-6d %5d %+10.2fbps %+10.2f %7.0f%%",
                     y, ym["n_rebal"], ym["net_bps_per_rebal"],
                     ym["active_sharpe"], 100 * hit)


if __name__ == "__main__":
    main()
