"""End-to-end daily rebalance optimization.

Daily-specific optimizations layered on the v7 baseline:

  Key insight: train on fwd_resid_5d (slow signal) but realize daily P&L.
  - Most of our alpha is PEAD drift (60-day decay) — predicting 1d forward
    is too short-horizon and noisy.
  - Predicting 5d forward gives a stable signal; holding 1 day captures
    1/5 of the predicted move per rebalance.

  Optimizations tested:
    D0: daily rebalance, predict + hold 1d (the C4 baseline +1.22)
    D1: predict 5d, rebalance daily, top-K (no inertia)
    D2: D1 + hysteresis (K=5 enter, K+M=2 exit threshold)
    D3: D1 + larger K=8 with rank-weighted positions
    D4: D2 + tighter dispersion gate (top 30% only)
    D5: D2 + lower cost (1.5 bps/side, patient execution)
    D6: full combined

  All variants are dispersion-gated (binary, top 40% trailing 252d).
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
from ml.research.alpha_v7_honest import (
    fit_predict, gate_rolling, bootstrap_active_sharpe_ci,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOP_K = 5
COST_BPS_SIDE = 2.5
GATE_PCTILE = 0.6
GATE_WINDOW = 252


def make_folds(panel, train_min_days=365 * 3, test_days=365, embargo_days=5):
    panel = panel.sort_values("ts")
    t0 = panel["ts"].min().normalize()
    t_max = panel["ts"].max()
    folds = []
    days = train_min_days
    while True:
        te = t0 + timedelta(days=days)
        ts = te + timedelta(days=embargo_days)
        te2 = ts + timedelta(days=test_days)
        if ts >= t_max: break
        if te2 > t_max: te2 = t_max
        folds.append((te, ts, te2))
        days += test_days
    return folds


# ---- daily portfolios with various tweaks -----------------------------

def daily_portfolio_basic(test_pred: pd.DataFrame, signal: str, pnl_label: str,
                            allowed: set, top_k: int, cost_bps_side: float) -> pd.DataFrame:
    """D0/D1: each day, rebalance to top-K based on signal. P&L = 1d forward."""
    sub = test_pred[test_pred["symbol"].isin(allowed)].dropna(
        subset=[signal, pnl_label]).copy()
    rows = []
    prev_long, prev_short = set(), set()
    for ts, bar in sub.groupby("ts"):
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values(signal)
        long_leg = set(bar.tail(top_k)["symbol"])
        short_leg = set(bar.head(top_k)["symbol"])
        long_chg = len(long_leg.symmetric_difference(prev_long))
        short_chg = len(short_leg.symmetric_difference(prev_short))
        turnover = (long_chg + short_chg) / (2 * top_k)
        cost = turnover * cost_bps_side * 2 / 1e4
        long_a = bar[bar["symbol"].isin(long_leg)][pnl_label].mean()
        short_a = bar[bar["symbol"].isin(short_leg)][pnl_label].mean()
        spread = long_a - short_a
        rows.append({"ts": ts, "spread_alpha": spread, "long_alpha": long_a,
                     "short_alpha": short_a, "turnover": turnover, "cost": cost,
                     "net_alpha": spread - cost, "n_universe": len(bar)})
        prev_long, prev_short = long_leg, short_leg
    return pd.DataFrame(rows)


def daily_portfolio_hysteresis(test_pred: pd.DataFrame, signal: str, pnl_label: str,
                                 allowed: set, top_k: int, exit_buffer: int,
                                 cost_bps_side: float) -> pd.DataFrame:
    """D2: enter at rank ≤ K, exit at rank > K + exit_buffer.
    Reduces churn from names bouncing around the top-K boundary."""
    sub = test_pred[test_pred["symbol"].isin(allowed)].dropna(
        subset=[signal, pnl_label]).copy()
    rows = []
    cur_long, cur_short = set(), set()
    n_universe_size = sub.groupby("ts")["symbol"].count().max()
    exit_long_rank = top_k + exit_buffer  # if rank > this (0-indexed from bottom), exit
    enter_long_rank = top_k
    for ts, bar in sub.groupby("ts"):
        if len(bar) < 2 * top_k + exit_buffer:
            continue
        bar = bar.sort_values(signal).reset_index(drop=True)
        n = len(bar)
        # Long ranks: highest signal at last position. rank from top: n-1, n-2, ...
        # bar index 0 = lowest signal, bar index n-1 = highest signal
        # "rank from top" = n - 1 - bar_index, so 0 = best, K-1 = K-th best
        bar["rank_top"] = n - 1 - bar.index
        bar["rank_bot"] = bar.index  # 0 = worst signal

        # Update long set
        new_long = set(cur_long)
        # Exit condition
        for s in list(new_long):
            r = bar[bar["symbol"] == s]
            if r.empty:
                new_long.discard(s); continue
            if r["rank_top"].iloc[0] > exit_long_rank - 1:
                new_long.discard(s)
        # Entry condition: top-K candidates not yet in long
        candidates = bar[bar["rank_top"] < enter_long_rank]["symbol"].tolist()
        for s in candidates:
            if len(new_long) >= top_k:
                break
            if s in new_long:
                continue
            new_long.add(s)
        # Trim if too many (shouldn't happen but safety)
        if len(new_long) > top_k:
            ranked = bar[bar["symbol"].isin(new_long)].sort_values("rank_top")
            new_long = set(ranked.head(top_k)["symbol"])

        # Same for short
        new_short = set(cur_short)
        for s in list(new_short):
            r = bar[bar["symbol"] == s]
            if r.empty:
                new_short.discard(s); continue
            if r["rank_bot"].iloc[0] > exit_long_rank - 1:
                new_short.discard(s)
        candidates_s = bar[bar["rank_bot"] < enter_long_rank]["symbol"].tolist()
        for s in candidates_s:
            if len(new_short) >= top_k:
                break
            if s in new_short:
                continue
            new_short.add(s)
        if len(new_short) > top_k:
            ranked = bar[bar["symbol"].isin(new_short)].sort_values("rank_bot")
            new_short = set(ranked.head(top_k)["symbol"])

        # Compute turnover and P&L
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
                     "net_alpha": spread - cost, "n_universe": n,
                     "n_long": len(new_long), "n_short": len(new_short)})
        cur_long, cur_short = new_long, new_short
    return pd.DataFrame(rows)


def daily_portfolio_rank_weighted(test_pred: pd.DataFrame, signal: str, pnl_label: str,
                                    allowed: set, top_k: int,
                                    cost_bps_side: float) -> pd.DataFrame:
    """D3: weight by rank within top-K. Top name gets highest weight,
    K-th gets lowest. Reduces turnover sensitivity to marginal rank changes."""
    sub = test_pred[test_pred["symbol"].isin(allowed)].dropna(
        subset=[signal, pnl_label]).copy()
    rows = []
    prev_long_w, prev_short_w = {}, {}
    for ts, bar in sub.groupby("ts"):
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values(signal)
        long_bar = bar.tail(top_k)
        short_bar = bar.head(top_k)
        # Rank-weighted: 1, 2, 3, ..., K → weight ∝ rank
        ranks_l = np.arange(1, top_k + 1, dtype=float)
        w_l = ranks_l / ranks_l.sum()
        ranks_s = np.arange(top_k, 0, -1, dtype=float)
        w_s = ranks_s / ranks_s.sum()

        long_a = (long_bar[pnl_label].values * w_l).sum()
        short_a = (short_bar[pnl_label].values * w_s).sum()
        spread = long_a - short_a

        cur_l = dict(zip(long_bar["symbol"].values, w_l))
        cur_s = dict(zip(short_bar["symbol"].values, w_s))
        all_l = set(prev_long_w) | set(cur_l)
        all_s = set(prev_short_w) | set(cur_s)
        long_turn = sum(abs(cur_l.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
        short_turn = sum(abs(cur_s.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        turnover = (long_turn + short_turn) / 2.0
        cost = turnover * cost_bps_side * 2 / 1e4
        rows.append({"ts": ts, "spread_alpha": spread, "long_alpha": long_a,
                     "short_alpha": short_a, "turnover": turnover, "cost": cost,
                     "net_alpha": spread - cost, "n_universe": len(bar)})
        prev_long_w, prev_short_w = cur_l, cur_s
    return pd.DataFrame(rows)


# ---- run helpers ------------------------------------------------------

def run_walk(panel, feats, train_label, folds, port_fn, port_kwargs):
    all_pnls = []
    for fold in folds:
        te, ts, te2 = fold
        train = panel[panel["ts"] <= te].copy()
        test = panel[(panel["ts"] >= ts) & (panel["ts"] <= te2)].copy()
        test_pred = fit_predict(train, test, feats, train_label)
        if test_pred.empty:
            continue
        lp = port_fn(test_pred, "pred", **port_kwargs)
        if not lp.empty:
            all_pnls.append(lp)
    return pd.concat(all_pnls, ignore_index=True) if all_pnls else pd.DataFrame()


def metrics_for(pnl, hold_for_annu=1):
    if pnl.empty:
        return {"n": 0}
    n = len(pnl)
    rebals_per_year = 252 / hold_for_annu
    g_sh = (pnl["spread_alpha"].mean() / pnl["spread_alpha"].std()
            * np.sqrt(rebals_per_year)) if pnl["spread_alpha"].std() > 0 else 0
    n_sh = (pnl["net_alpha"].mean() / pnl["net_alpha"].std()
            * np.sqrt(rebals_per_year)) if pnl["net_alpha"].std() > 0 else 0
    pnl_y = pnl.copy()
    pnl_y["year"] = pd.to_datetime(pnl_y["ts"]).dt.year
    n_years = pnl_y["year"].nunique()
    rebals_per_year_actual = n / max(n_years, 1)
    annual_mean = pnl["net_alpha"].mean() * rebals_per_year_actual
    annual_std = pnl["net_alpha"].std() * np.sqrt(rebals_per_year_actual)
    uncond = annual_mean / annual_std if annual_std > 0 else 0
    return {"n_rebal": n, "active_sharpe": n_sh, "uncond_sharpe": uncond,
            "annual_return_pct": annual_mean * 100,
            "avg_turnover_pct": pnl["turnover"].mean() * 100,
            "annual_cost_bps": pnl["cost"].mean() * 1e4 * rebals_per_year_actual,
            "net_bps_per_rebal": pnl["net_alpha"].mean() * 1e4}


def boot_ci(pnl, hold_for_annu=1, n_boot=2000):
    if pnl.empty or len(pnl) < 30:
        return np.nan, np.nan
    n = len(pnl)
    rng = np.random.default_rng(42)
    rpy = 252 / hold_for_annu
    sh = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = pnl["net_alpha"].iloc[idx]
        if s.std() > 0:
            sh.append(s.mean() / s.std() * np.sqrt(rpy))
    if not sh:
        return np.nan, np.nan
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))


# ---- main ---------------------------------------------------------------

def main() -> None:
    log.info("loading panel...")
    panel, earnings, _ = load_universe()
    if panel.empty:
        return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)

    # Build BOTH labels so we can train on 5d, P&L on 1d
    panel = add_residual_and_label(panel, 1)
    panel = add_residual_and_label(panel, 5)
    panel = add_residual_and_label(panel, 3)

    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    feats = feats_A + feats_B + ["sym_id"]
    allowed = set(XYZ_IN_SP100)
    folds = make_folds(panel)

    log.info("\n=== Daily rebalance optimization ===")
    log.info("  %-40s  %5s  %10s  %18s  %10s  %12s  %10s",
             "config", "n_reb", "active_Sh", "95% CI", "uncond_Sh",
             "annu_cost_bps", "turn%/d")

    results = {}

    # D0: daily rebalance, train+P&L on 1d (current C4 baseline +1.22)
    log.info("\n>>> D0 daily, train+P&L on 1d, no inertia")
    pnl0 = run_walk(panel, feats, "fwd_resid_1d", folds, daily_portfolio_basic,
                     {"pnl_label": "fwd_resid_1d", "allowed": allowed,
                      "top_k": TOP_K, "cost_bps_side": COST_BPS_SIDE})
    pnl0 = gate_rolling(pnl0, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    results["D0 daily 1d→1d"] = pnl0

    # D1: train on 5d (slow signal), realize 1d (KEY OPTIMIZATION)
    log.info(">>> D1 daily, train on 5d, P&L on 1d")
    pnl1_pre = run_walk(panel, feats, "fwd_resid_5d", folds, daily_portfolio_basic,
                         {"pnl_label": "fwd_resid_1d", "allowed": allowed,
                          "top_k": TOP_K, "cost_bps_side": COST_BPS_SIDE})
    pnl1 = gate_rolling(pnl1_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    results["D1 daily 5d→1d (slow signal)"] = pnl1

    # D2: D1 + hysteresis
    log.info(">>> D2 D1 + hysteresis (enter K=5, exit at rank>7)")
    pnl2_pre = run_walk(panel, feats, "fwd_resid_5d", folds, daily_portfolio_hysteresis,
                         {"pnl_label": "fwd_resid_1d", "allowed": allowed,
                          "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": COST_BPS_SIDE})
    pnl2 = gate_rolling(pnl2_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    results["D2 + hysteresis"] = pnl2

    # D3: D1 + K=7 with rank weights (max for 15-name universe)
    log.info(">>> D3 D1 + K=7 rank-weighted")
    pnl3_pre = run_walk(panel, feats, "fwd_resid_5d", folds, daily_portfolio_rank_weighted,
                         {"pnl_label": "fwd_resid_1d", "allowed": allowed,
                          "top_k": 7, "cost_bps_side": COST_BPS_SIDE})
    if not pnl3_pre.empty:
        pnl3 = gate_rolling(pnl3_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
        results["D3 + K=7 rank-weighted"] = pnl3
    else:
        log.warning("D3 returned empty (universe too small for K=7)")

    # D4: D2 + tighter gate (top 30%)
    log.info(">>> D4 D2 + tighter dispersion gate (top 30%)")
    pnl4 = gate_rolling(pnl2_pre, regime, pctile=0.7, window_days=GATE_WINDOW)
    results["D4 D2 + tight gate (top 30%)"] = pnl4

    # D5: D2 + lower cost (1.5 bps/side)
    log.info(">>> D5 D2 + lower cost (1.5 bps/side)")
    pnl5_pre = run_walk(panel, feats, "fwd_resid_5d", folds, daily_portfolio_hysteresis,
                         {"pnl_label": "fwd_resid_1d", "allowed": allowed,
                          "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": 1.5})
    pnl5 = gate_rolling(pnl5_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    results["D5 D2 + 1.5bps cost"] = pnl5

    # D6: full combined — 5d→1d, hysteresis K=5 buffer 2, tight gate, low cost
    log.info(">>> D6 full combined")
    pnl6_pre = run_walk(panel, feats, "fwd_resid_5d", folds, daily_portfolio_hysteresis,
                         {"pnl_label": "fwd_resid_1d", "allowed": allowed,
                          "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": 1.5})
    pnl6 = gate_rolling(pnl6_pre, regime, pctile=0.7, window_days=GATE_WINDOW)
    results["D6 full combined"] = pnl6

    # Reference: 3d hold (the +1.40 baseline)
    log.info(">>> REF 3d hold baseline (for reference)")
    pnl_ref = run_walk(panel, feats, "fwd_resid_3d", folds, daily_portfolio_basic,
                        {"pnl_label": "fwd_resid_3d", "allowed": allowed,
                         "top_k": TOP_K, "cost_bps_side": COST_BPS_SIDE})
    # for 3d, only rebalance every 3rd day
    pnl_ref = pnl_ref.iloc[::3].copy()
    pnl_ref = gate_rolling(pnl_ref, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    results["REF 3d baseline"] = pnl_ref

    # Summary
    log.info("\n=== SUMMARY ===")
    log.info("  %-40s  %5s  %10s  %18s  %10s  %12s  %10s",
             "config", "n_reb", "active_Sh", "95% CI",
             "uncond_Sh", "annu_cost_bps", "turn%/reb")
    for name, pnl in results.items():
        # Use 1 for daily rebalance configs, 3 for 3d ref
        hold = 3 if "REF 3d" in name else 1
        m = metrics_for(pnl, hold)
        if m.get("n_rebal", 0) == 0:
            continue
        lo, hi = boot_ci(pnl, hold)
        log.info("  %-40s  %5d  %+8.2f  [%+5.2f,%+5.2f]  %+8.2f  %12.0f  %8.0f%%",
                 name, m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["uncond_sharpe"], m["annual_cost_bps"], m["avg_turnover_pct"])


if __name__ == "__main__":
    main()
