"""Fixed PEAD timing + 3d hold xyz strategy.

Bug fix: The current `add_features_B` uses the residual on the earnings
calendar date as `B_event_day_resid`. But yfinance earnings timestamps
include the time-of-day, and most US large-cap earnings are released
after-market-close (16:00-18:00 ET). For those, the regular-session return
on the announcement date does NOT see the news — the market reaction
appears on the NEXT trading day's open (and close).

Correct logic:
  - AMC (announcement >= 16:00 ET): effective event date = next business day
  - BMO (announcement < 9:30 ET):    effective event date = same day
  - DMT (during regular hours):      effective event date = same day

The "effective event date" is the first trading day whose close reflects
the announcement. `B_event_day_resid` should be the residual on that day,
and `B_days_since_earn` should be measured from that day.

This script implements `add_features_B_fixed` and reruns the 3-day hold
strategy on the 15-name xyz STRICT universe.
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay

from data_collectors.sp100_loader import load_universe
from ml.research.alpha_v7_multi import (
    BETA_WINDOW, LGB_PARAMS, SEEDS, PEAD_MAX_DAYS,
    load_anchors, add_returns_and_basket,
    add_features_A, add_features_B,  # original for comparison
)
from ml.research.alpha_v7_weekly import (
    fit_predict, metrics_weekly, bootstrap_ci_weekly, make_folds,
)
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz import construct_portfolio_subset, gate_by_dispersion
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100
from ml.research.alpha_v7_freq_sweep import add_residual_and_label, metrics_freq, annualized_unconditional

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HOLD_DAYS = 3       # optimum from frequency sweep
COST_BPS_SIDE = 2.5  # xyz growth-mode estimate
TOP_K = 5


# ---- fixed PEAD feature -----------------------------------------------

def _to_effective_event_date(ts_utc_with_tz: pd.Timestamp) -> pd.Timestamp:
    """Return midnight UTC of the effective trading day.

    Panel.ts is midnight UTC of each trading day (yfinance daily data
    normalized). The "effective event date" is the trading day whose close
    fully reflects the announcement:
      - AMC (announcement >= 16:00 ET): next business day
      - BMO (announcement < 9:30 ET):    same day
      - DMT (during regular hours):      same day

    We compute the ET date, then return midnight UTC of that date so it
    aligns with the panel's daily timestamp keys."""
    et = ts_utc_with_tz.tz_convert("America/New_York")
    if et.hour >= 16:
        et_date = (et.normalize() + BusinessDay(1)).date()
    else:
        et_date = et.normalize().date()
    return pd.Timestamp(et_date, tz="UTC")


def add_features_B_fixed(panel: pd.DataFrame,
                          earnings: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Re-implementation of `add_features_B` with corrected event-day timing."""
    earn = earnings.copy()
    earn["ts_orig"] = pd.to_datetime(earn["ts"], utc=True)
    # Compute effective event date per row
    earn["effective_ts"] = earn["ts_orig"].apply(_to_effective_event_date)
    earn["effective_ts"] = earn["effective_ts"].astype("datetime64[ns, UTC]")
    earn = earn.sort_values(["symbol", "effective_ts"]).reset_index(drop=True)

    panel = panel.copy()
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True).astype("datetime64[ns, UTC]")
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)

    out_chunks = []
    for sym, g in panel.groupby("symbol"):
        e = earn[earn["symbol"] == sym][["effective_ts", "surprise_pct"]].dropna(
            subset=["surprise_pct"])
        if e.empty:
            g = g.copy()
            g["B_days_since_earn"] = np.nan
            g["B_surprise_pct"] = np.nan
            g["B_event_day_resid"] = np.nan
            g["B_decay_signal"] = np.nan
            g["earnings_ts"] = pd.NaT
            out_chunks.append(g)
            continue
        e_sorted = e.sort_values("effective_ts").rename(
            columns={"effective_ts": "earnings_ts"})
        # merge_asof: at panel ts T, find latest earnings effective_ts <= T
        merged = pd.merge_asof(
            g.sort_values("ts"), e_sorted,
            left_on="ts", right_on="earnings_ts",
            allow_exact_matches=True,  # day-of event IS day 0
            direction="backward",
        )
        out_chunks.append(merged)
    panel_b = pd.concat(out_chunks, ignore_index=True)

    panel_b["B_days_since_earn"] = (
        panel_b["ts"] - panel_b["earnings_ts"]
    ).dt.days
    # Day 0 is effective event day (close fully reflects news).
    # Drift period: days_since in [1, MAX] for "post-announcement drift".
    valid = panel_b["B_days_since_earn"].between(0, PEAD_MAX_DAYS, inclusive="both")
    panel_b["B_surprise_pct"] = panel_b["surprise_pct"].where(valid, np.nan)

    # B_event_day_resid: residual on effective event day (day 0)
    earn_resid = panel.merge(
        earn[["symbol", "effective_ts"]].drop_duplicates(),
        left_on=["symbol", "ts"], right_on=["symbol", "effective_ts"],
        how="inner")[["symbol", "effective_ts", "resid"]]
    earn_resid = earn_resid.rename(columns={"resid": "B_event_day_resid_join"})
    earn_resid = earn_resid.drop_duplicates(subset=["symbol", "effective_ts"], keep="first")
    panel_b = panel_b.merge(
        earn_resid, left_on=["symbol", "earnings_ts"], right_on=["symbol", "effective_ts"],
        how="left")
    panel_b = panel_b.drop_duplicates(subset=["symbol", "ts"], keep="first")
    panel_b = panel_b.rename(columns={"B_event_day_resid_join": "B_event_day_resid"})
    panel_b["B_event_day_resid"] = panel_b["B_event_day_resid"].where(valid, np.nan)
    panel_b["B_decay_signal"] = (
        panel_b["B_event_day_resid"]
        * (1 - panel_b["B_days_since_earn"].clip(upper=PEAD_MAX_DAYS) / PEAD_MAX_DAYS)
    )

    # Map back to original panel
    panel_b = panel_b.sort_values(["symbol", "ts"]).reset_index(drop=True)
    panel_out = panel.copy()
    for c in ("B_days_since_earn", "B_surprise_pct", "B_event_day_resid", "B_decay_signal"):
        panel_out[c] = panel_b[c].values

    return panel_out, ["B_days_since_earn", "B_surprise_pct",
                       "B_event_day_resid", "B_decay_signal"]


# ---- main: compare original vs fixed PEAD on 3d hold ------------------

def run_strategy(panel: pd.DataFrame, earnings: pd.DataFrame, anchors: pd.DataFrame,
                  use_fixed_pead: bool, hold_days: int = HOLD_DAYS) -> tuple:
    panel = panel.copy()
    panel = add_residual_and_label(panel, hold_days)
    panel, feats_A = add_features_A(panel)
    if use_fixed_pead:
        panel, feats_B = add_features_B_fixed(panel, earnings)
    else:
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
        lp = construct_portfolio_subset(
            test_pred, "pred", label,
            allowed_symbols=set(XYZ_IN_SP100),
            top_k=TOP_K, cost_bps=COST_BPS_SIDE * 2,
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

    # AMC vs BMO breakdown
    earn = earnings.copy()
    earn["ts_orig"] = pd.to_datetime(earn["ts"], utc=True)
    earn["et_hour"] = earn["ts_orig"].dt.tz_convert("America/New_York").dt.hour
    n_amc = (earn["et_hour"] >= 16).sum()
    n_bmo = ((earn["et_hour"] < 9) | ((earn["et_hour"] == 9) &
             (earn["ts_orig"].dt.tz_convert("America/New_York").dt.minute < 30))).sum()
    n_dmt = len(earn) - n_amc - n_bmo
    log.info("\nEarnings timing breakdown (n=%d total):", len(earn))
    log.info("  AMC (after market close, ≥16 ET):   %d (%.1f%%)", n_amc, 100*n_amc/len(earn))
    log.info("  BMO (before market open, <9:30 ET): %d (%.1f%%)", n_bmo, 100*n_bmo/len(earn))
    log.info("  DMT (during regular hours):         %d (%.1f%%)", n_dmt, 100*n_dmt/len(earn))
    log.info("  → AMC means our prior code was using the wrong day for %d/%d events",
             n_amc, len(earn))

    log.info("\n=== Original (buggy) PEAD timing, 3d hold ===")
    pnl_orig, m_orig, ann_orig = run_strategy(panel, earnings, anchors, use_fixed_pead=False)
    log.info("  n_rebal=%d  net/d=%+.2fbps  active_Sh=%+.2f  uncond_Sh=%+.2f  ann_ret=%+.2f%%",
             m_orig.get("n_rebal", 0), m_orig.get("net_bps_per_day", 0),
             m_orig.get("active_sharpe_annu", 0),
             ann_orig.get("unconditional_sharpe", 0),
             ann_orig.get("annual_return_pct", 0))

    log.info("\n=== Fixed PEAD timing (AMC→next BDay), 3d hold ===")
    pnl_fix, m_fix, ann_fix = run_strategy(panel, earnings, anchors, use_fixed_pead=True)
    log.info("  n_rebal=%d  net/d=%+.2fbps  active_Sh=%+.2f  uncond_Sh=%+.2f  ann_ret=%+.2f%%",
             m_fix.get("n_rebal", 0), m_fix.get("net_bps_per_day", 0),
             m_fix.get("active_sharpe_annu", 0),
             ann_fix.get("unconditional_sharpe", 0),
             ann_fix.get("annual_return_pct", 0))

    # Sharpe lift from the fix
    delta_active = m_fix.get("active_sharpe_annu", 0) - m_orig.get("active_sharpe_annu", 0)
    delta_uncond = ann_fix.get("unconditional_sharpe", 0) - ann_orig.get("unconditional_sharpe", 0)
    log.info("\n  >>> Sharpe lift from fix: active %+.2f  uncond %+.2f", delta_active, delta_uncond)

    # Per-year for fixed
    if not pnl_fix.empty:
        pnl_y = pnl_fix.copy()
        pnl_y["year"] = pnl_y["ts"].dt.year
        log.info("\n=== Per-year (fixed PEAD, 3d) ===")
        log.info("  %-6s %5s %12s %12s %10s",
                 "year", "n_reb", "gross/d", "net/d", "active_Sh")
        for y, g in pnl_y.groupby("year"):
            qm = metrics_freq(g, HOLD_DAYS)
            log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+10.2f",
                     y, qm["n_rebal"], qm["gross_bps_per_day"],
                     qm["net_bps_per_day"], qm["active_sharpe_annu"])


if __name__ == "__main__":
    main()
