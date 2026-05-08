"""Refine the days_until_earn feature.

Naive days_until_earn HURT the strategy (+2.67 → +1.80). Likely reason:
pre-earnings positioning is direction-less without context. Refinements:

  V0: baseline +3.16 (sector features, no earnings-window feature)
  V1: signed_anticipation = (90 - days_until) × sign(last_surprise) clipped
       — captures: anticipation strength × prior PEAD direction
  V2: is_pre_event_5d binary (1 if next earnings within 5 days)
  V3: skip names within ±3d of earnings (universe filter at portfolio level)
  V4: skip names within ±5d of earnings
  V5: Combine: V1 + V3 (best feature + best filter)
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
from ml.research.alpha_v7_push import (
    add_sector_features, add_days_until_earnings,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOP_K = 5
COST_BPS_SIDE = 1.5
GATE_PCTILE = 0.6
GATE_WINDOW = 252


def add_earnings_refined(panel: pd.DataFrame, earnings: pd.DataFrame,
                          B_surprise_col: str = "B_surprise_pct",
                          cap_days: int = 90) -> tuple[pd.DataFrame, list[str]]:
    """Refined earnings features using days_until_earn + last surprise direction."""
    # First add raw days_until_earn (need it for signed feature)
    panel, _ = add_days_until_earnings(panel, earnings, cap_days=cap_days)
    # Need most recent surprise direction
    # B_surprise_pct from add_features_B_fixed has surprise of MOST RECENT past earnings
    # We need direction × anticipation
    panel["F_anticipation_strength"] = (
        cap_days - panel["F_days_until_earn"].clip(upper=cap_days)
    ).fillna(0)  # 0 if no upcoming earnings within cap
    # signed by sign of last surprise (positive if last beat → bigger long-side anticipation)
    panel["F_signed_anticipation"] = (
        panel["F_anticipation_strength"]
        * np.sign(panel[B_surprise_col].fillna(0))
    )
    # binary pre-event flag
    panel["F_is_pre_event_5d"] = (
        (panel["F_days_until_earn"] <= 5) & (panel["F_days_until_earn"] >= 0)
    ).astype(float)
    return panel, ["F_signed_anticipation", "F_anticipation_strength",
                   "F_is_pre_event_5d"]


def daily_portfolio_hysteresis_skip(test_pred: pd.DataFrame, signal: str,
                                       pnl_label: str, allowed: set,
                                       top_k: int, exit_buffer: int,
                                       cost_bps_side: float,
                                       skip_window_days: int = 0) -> pd.DataFrame:
    """Same as daily_portfolio_hysteresis but skips names within
    ±skip_window_days of any earnings (uses F_days_until_earn column)."""
    sub = test_pred[test_pred["symbol"].isin(allowed)].dropna(
        subset=[signal, pnl_label]).copy()
    # Filter pre-event names
    if skip_window_days > 0 and "F_days_until_earn" in sub.columns:
        sub = sub[
            (sub["F_days_until_earn"].isna())  # no upcoming earnings
            | (sub["F_days_until_earn"] > skip_window_days)  # > N days away
            | (sub["F_days_until_earn"] < 0)
        ].copy()
        log.info("    skipped pre-earnings: kept %d rows (skip window=%dd)",
                 len(sub), skip_window_days)
    rows = []
    cur_long, cur_short = set(), set()
    for ts, bar in sub.groupby("ts"):
        if len(bar) < 2 * top_k + exit_buffer:
            continue
        bar = bar.sort_values(signal).reset_index(drop=True)
        n = len(bar)
        bar["rank_top"] = n - 1 - bar.index
        bar["rank_bot"] = bar.index

        new_long = set(cur_long)
        for s in list(new_long):
            r = bar[bar["symbol"] == s]
            if r.empty or r["rank_top"].iloc[0] > top_k + exit_buffer - 1:
                new_long.discard(s)
        candidates = bar[bar["rank_top"] < top_k]["symbol"].tolist()
        for s in candidates:
            if len(new_long) >= top_k: break
            new_long.add(s)
        if len(new_long) > top_k:
            ranked = bar[bar["symbol"].isin(new_long)].sort_values("rank_top")
            new_long = set(ranked.head(top_k)["symbol"])

        new_short = set(cur_short)
        for s in list(new_short):
            r = bar[bar["symbol"] == s]
            if r.empty or r["rank_bot"].iloc[0] > top_k + exit_buffer - 1:
                new_short.discard(s)
        candidates_s = bar[bar["rank_bot"] < top_k]["symbol"].tolist()
        for s in candidates_s:
            if len(new_short) >= top_k: break
            new_short.add(s)
        if len(new_short) > top_k:
            ranked = bar[bar["symbol"].isin(new_short)].sort_values("rank_bot")
            new_short = set(ranked.head(top_k)["symbol"])

        long_chg = len(new_long.symmetric_difference(cur_long))
        short_chg = len(new_short.symmetric_difference(cur_short))
        turnover = (long_chg + short_chg) / (2 * top_k)
        cost = turnover * cost_bps_side * 2 / 1e4

        if not new_long or not new_short:
            cur_long, cur_short = new_long, new_short
            continue
        long_a = bar[bar["symbol"].isin(new_long)][pnl_label].mean()
        short_a = bar[bar["symbol"].isin(new_short)][pnl_label].mean()
        spread = long_a - short_a
        rows.append({"ts": ts, "spread_alpha": spread, "long_alpha": long_a,
                     "short_alpha": short_a, "turnover": turnover, "cost": cost,
                     "net_alpha": spread - cost, "n_universe": n})
        cur_long, cur_short = new_long, new_short
    return pd.DataFrame(rows)


def main() -> None:
    log.info("loading panel...")
    panel, earnings, _ = load_universe()
    if panel.empty: return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    for h in (1, 3, 5, 10):
        panel = add_residual_and_label(panel, h)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel, feats_F_sector = add_sector_features(panel)
    panel, feats_F_refined = add_earnings_refined(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    feats_v0 = feats_A + feats_B + feats_F_sector + ["sym_id"]  # baseline +3.16
    allowed = set(XYZ_IN_SP100)
    folds = make_folds(panel)
    train_labels = ["fwd_resid_3d", "fwd_resid_5d", "fwd_resid_10d"]

    log.info("\n=== Earnings refinement comparison ===")
    log.info("  %-50s  %5s  %10s  %18s  %10s",
             "config", "n_reb", "active_Sh", "95% CI", "uncond_Sh")

    # V0: baseline +3.16
    log.info("\n>>> V0 baseline (sector features, no earnings-window)")
    pnl0_pre = run_walk_multihorizon(
        panel, feats_v0, train_labels, folds, daily_portfolio_hysteresis,
        {"pnl_label": "fwd_resid_1d", "allowed": allowed,
         "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": COST_BPS_SIDE})
    pnl0 = gate_rolling(pnl0_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m0 = metrics_for(pnl0, 1)
    lo0, hi0 = boot_ci(pnl0, 1)
    log.info("  V0 baseline                                       n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f",
             m0["n_rebal"], m0["active_sharpe"], lo0, hi0, m0["uncond_sharpe"])

    # V1: + signed_anticipation
    feats_v1 = feats_v0 + ["F_signed_anticipation"]
    log.info("\n>>> V1 + F_signed_anticipation feature")
    pnl1_pre = run_walk_multihorizon(
        panel, feats_v1, train_labels, folds, daily_portfolio_hysteresis,
        {"pnl_label": "fwd_resid_1d", "allowed": allowed,
         "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": COST_BPS_SIDE})
    pnl1 = gate_rolling(pnl1_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m1 = metrics_for(pnl1, 1)
    lo1, hi1 = boot_ci(pnl1, 1)
    log.info("  V1                                                n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f",
             m1["n_rebal"], m1["active_sharpe"], lo1, hi1, m1["uncond_sharpe"])

    # V2: + is_pre_event_5d
    feats_v2 = feats_v0 + ["F_is_pre_event_5d"]
    log.info("\n>>> V2 + F_is_pre_event_5d binary feature")
    pnl2_pre = run_walk_multihorizon(
        panel, feats_v2, train_labels, folds, daily_portfolio_hysteresis,
        {"pnl_label": "fwd_resid_1d", "allowed": allowed,
         "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": COST_BPS_SIDE})
    pnl2 = gate_rolling(pnl2_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m2 = metrics_for(pnl2, 1)
    lo2, hi2 = boot_ci(pnl2, 1)
    log.info("  V2                                                n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f",
             m2["n_rebal"], m2["active_sharpe"], lo2, hi2, m2["uncond_sharpe"])

    # V3: skip names within ±3 days of earnings (filter at portfolio)
    log.info("\n>>> V3 skip names within ±3d of earnings (portfolio filter)")
    # Need to use modified portfolio function
    all_pnls = []
    for fold in folds:
        train_end, test_start, test_end = fold
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        # Multi-horizon predict
        ensemble_preds = []
        sub_test = None
        for label in train_labels:
            train_ = train.dropna(subset=feats_v0 + [label])
            if len(train_) < 1000:
                continue
            sub = test.dropna(subset=feats_v0).copy()
            if sub_test is None:
                sub_test = sub.copy()
            for seed in SEEDS:
                m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
                m.fit(train_[feats_v0], train_[label])
                ensemble_preds.append(m.predict(sub[feats_v0]))
        if not ensemble_preds or sub_test is None:
            continue
        sub_test["pred"] = np.mean(ensemble_preds, axis=0)
        lp = daily_portfolio_hysteresis_skip(
            sub_test, "pred", "fwd_resid_1d", allowed,
            TOP_K, 2, COST_BPS_SIDE, skip_window_days=3)
        if not lp.empty:
            all_pnls.append(lp)
    pnl3_pre = pd.concat(all_pnls, ignore_index=True) if all_pnls else pd.DataFrame()
    pnl3 = gate_rolling(pnl3_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m3 = metrics_for(pnl3, 1)
    lo3, hi3 = boot_ci(pnl3, 1)
    log.info("  V3                                                n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f",
             m3["n_rebal"], m3["active_sharpe"], lo3, hi3, m3["uncond_sharpe"])

    # V4: skip names within ±5 days of earnings
    log.info("\n>>> V4 skip names within ±5d of earnings (portfolio filter)")
    all_pnls = []
    for fold in folds:
        train_end, test_start, test_end = fold
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        ensemble_preds = []
        sub_test = None
        for label in train_labels:
            train_ = train.dropna(subset=feats_v0 + [label])
            if len(train_) < 1000:
                continue
            sub = test.dropna(subset=feats_v0).copy()
            if sub_test is None:
                sub_test = sub.copy()
            for seed in SEEDS:
                m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
                m.fit(train_[feats_v0], train_[label])
                ensemble_preds.append(m.predict(sub[feats_v0]))
        if not ensemble_preds or sub_test is None:
            continue
        sub_test["pred"] = np.mean(ensemble_preds, axis=0)
        lp = daily_portfolio_hysteresis_skip(
            sub_test, "pred", "fwd_resid_1d", allowed,
            TOP_K, 2, COST_BPS_SIDE, skip_window_days=5)
        if not lp.empty:
            all_pnls.append(lp)
    pnl4_pre = pd.concat(all_pnls, ignore_index=True) if all_pnls else pd.DataFrame()
    pnl4 = gate_rolling(pnl4_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m4 = metrics_for(pnl4, 1)
    lo4, hi4 = boot_ci(pnl4, 1)
    log.info("  V4                                                n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f",
             m4["n_rebal"], m4["active_sharpe"], lo4, hi4, m4["uncond_sharpe"])

    # Final summary
    log.info("\n=== SUMMARY ===")
    log.info("  %-50s  %5s  %10s  %18s  %10s  %12s",
             "config", "n_reb", "active_Sh", "95% CI", "uncond_Sh", "ann_ret%")
    for name, m, lo, hi in [
        ("V0 baseline (sector features only) +3.16", m0, lo0, hi0),
        ("V1 + signed_anticipation", m1, lo1, hi1),
        ("V2 + is_pre_event_5d binary", m2, lo2, hi2),
        ("V3 skip ±3d earnings", m3, lo3, hi3),
        ("V4 skip ±5d earnings", m4, lo4, hi4),
    ]:
        if not m: continue
        log.info("  %-50s  %5d  %+8.2f  [%+5.2f,%+5.2f]  %+8.2f  %+8.2f%%",
                 name, m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["uncond_sharpe"], m["annual_return_pct"])


if __name__ == "__main__":
    main()
