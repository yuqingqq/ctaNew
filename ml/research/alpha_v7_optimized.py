"""Tier-A optimizations + daily rebalance test.

Builds on the +1.40 active Sharpe baseline (alpha_v7_honest.py C1) and
layers in:

  Opt 1 — Vol-target sizing (replace equal-weight legs with 1/idio_vol)
  Opt 2 — Rolling 5y training window (replace expanding window)
  Opt 3 — Continuous regime sizing (scale book by dispersion z, not binary)
  Opt 4 — Daily rebalance with smoothed predictions (predict every day,
          rank by 3-day rolling avg of predictions to limit churn)

Configurations tested:
  C0 — baseline (3d hold, equal-weight, expanding window, binary gate)
  C1 — + vol-target sizing
  C2 — + rolling 5y window
  C3 — C2 + continuous regime
  C4 — daily rebalance, smoothed (no other opts)
  C5 — C3 + daily smoothed rebalance (full stack)
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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HOLD_DAYS = 3       # baseline
COST_BPS_SIDE = 2.5
TOP_K = 5
GATE_PCTILE = 0.6
GATE_WINDOW_DAYS = 252


# ---- folds: rolling vs expanding ---------------------------------------

def make_folds_v2(panel: pd.DataFrame, train_window_days: int | None = None,
                   train_min_days: int = 365 * 3, test_days: int = 365,
                   embargo_days: int = 5) -> list[tuple]:
    """If train_window_days is set, use rolling window of that size.
    Otherwise expanding window starting from train_min_days."""
    panel = panel.sort_values("ts")
    t0 = panel["ts"].min().normalize()
    t_max = panel["ts"].max()
    folds = []
    days = train_min_days
    while True:
        train_end = t0 + timedelta(days=days)
        if train_window_days is None:
            train_start = t0
        else:
            train_start = max(t0, train_end - timedelta(days=train_window_days))
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_days)
        if test_start >= t_max:
            break
        if test_end > t_max:
            test_end = t_max
        folds.append((train_start, train_end, test_start, test_end))
        days += test_days
    return folds


# ---- prediction --------------------------------------------------------

def fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                features: list[str], label: str) -> pd.DataFrame:
    train_ = train.dropna(subset=features + [label])
    if len(train_) < 1000:
        return pd.DataFrame()
    sub = test.dropna(subset=features).copy()
    if sub.empty:
        return pd.DataFrame()
    preds = []
    for seed in SEEDS:
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(train_[features], train_[label])
        preds.append(m.predict(sub[features]))
    sub["pred"] = np.mean(preds, axis=0)
    return sub


# ---- portfolio constructions ------------------------------------------

def portfolio_baseline(test_pred: pd.DataFrame, signal: str, pnl_label: str,
                        allowed: set, top_k: int, cost_bps_side: float,
                        hold_days: int) -> pd.DataFrame:
    """C0: equal-weight top-K, hold for hold_days, rebalance every hold_days."""
    sub = test_pred[test_pred["symbol"].isin(allowed)].dropna(
        subset=[signal, pnl_label]).copy()
    unique_ts = sorted(sub["ts"].unique())
    rebal_ts = unique_ts[::hold_days]
    rows = []
    prev_long = set()
    prev_short = set()
    for ts in rebal_ts:
        bar = sub[sub["ts"] == ts]
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


def portfolio_voltarget(test_pred: pd.DataFrame, signal: str, pnl_label: str,
                          allowed: set, top_k: int, cost_bps_side: float,
                          hold_days: int, vol_col: str = "A_idio_vol_22d") -> pd.DataFrame:
    """C1: vol-target weights. weight_i ∝ 1 / vol_i, normalized so leg sums
    to 1.0. Reduces concentration in high-vol names."""
    sub = test_pred[test_pred["symbol"].isin(allowed)].dropna(
        subset=[signal, pnl_label, vol_col]).copy()
    unique_ts = sorted(sub["ts"].unique())
    rebal_ts = unique_ts[::hold_days]
    rows = []
    prev_long_w: dict = {}
    prev_short_w: dict = {}
    for ts in rebal_ts:
        bar = sub[sub["ts"] == ts]
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values(signal)
        long_bar = bar.tail(top_k)
        short_bar = bar.head(top_k)
        # 1/vol weights, normalized
        inv_l = 1.0 / long_bar[vol_col].clip(lower=1e-6)
        inv_s = 1.0 / short_bar[vol_col].clip(lower=1e-6)
        w_l = (inv_l / inv_l.sum()).values
        w_s = (inv_s / inv_s.sum()).values
        long_alpha = (long_bar[pnl_label].values * w_l).sum()
        short_alpha = (short_bar[pnl_label].values * w_s).sum()
        spread = long_alpha - short_alpha
        # Turnover: weighted weight changes
        cur_l = dict(zip(long_bar["symbol"].values, w_l))
        cur_s = dict(zip(short_bar["symbol"].values, w_s))
        all_l = set(prev_long_w) | set(cur_l)
        all_s = set(prev_short_w) | set(cur_s)
        long_turn = sum(abs(cur_l.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
        short_turn = sum(abs(cur_s.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        turnover = (long_turn + short_turn) / 2.0
        cost = turnover * cost_bps_side * 2 / 1e4
        rows.append({"ts": ts, "spread_alpha": spread, "long_alpha": long_alpha,
                     "short_alpha": short_alpha, "turnover": turnover, "cost": cost,
                     "net_alpha": spread - cost, "n_universe": len(bar)})
        prev_long_w, prev_short_w = cur_l, cur_s
    return pd.DataFrame(rows)


def portfolio_daily_smoothed(test_pred: pd.DataFrame, signal: str, pnl_label: str,
                              allowed: set, top_k: int, cost_bps_side: float,
                              smooth_days: int = 3,
                              vol_col: str | None = "A_idio_vol_22d") -> pd.DataFrame:
    """C4/C5: daily rebalance with N-day smoothed predictions to limit churn.

    At each day, rank by trailing N-day mean of predictions. Top-K long,
    bottom-K short. Use vol-target if vol_col given, else equal-weight.
    """
    sub = test_pred[test_pred["symbol"].isin(allowed)].dropna(
        subset=[signal, pnl_label]).copy()
    if vol_col and vol_col in sub.columns:
        sub = sub.dropna(subset=[vol_col])
    sub = sub.sort_values(["symbol", "ts"]).reset_index(drop=True)
    # Smoothed prediction: rolling N-day mean per symbol
    sub["smooth_pred"] = (sub.groupby("symbol")[signal]
                          .transform(lambda s: s.rolling(smooth_days, min_periods=1).mean()))
    rows = []
    prev_long_w: dict = {}
    prev_short_w: dict = {}
    for ts, bar in sub.groupby("ts"):
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values("smooth_pred")
        long_bar = bar.tail(top_k)
        short_bar = bar.head(top_k)
        if vol_col and vol_col in bar.columns:
            inv_l = 1.0 / long_bar[vol_col].clip(lower=1e-6)
            inv_s = 1.0 / short_bar[vol_col].clip(lower=1e-6)
            w_l = (inv_l / inv_l.sum()).values
            w_s = (inv_s / inv_s.sum()).values
        else:
            w_l = np.ones(top_k) / top_k
            w_s = np.ones(top_k) / top_k
        # The label is daily forward residual — fwd_resid_1d
        long_alpha = (long_bar[pnl_label].values * w_l).sum()
        short_alpha = (short_bar[pnl_label].values * w_s).sum()
        spread = long_alpha - short_alpha
        cur_l = dict(zip(long_bar["symbol"].values, w_l))
        cur_s = dict(zip(short_bar["symbol"].values, w_s))
        all_l = set(prev_long_w) | set(cur_l)
        all_s = set(prev_short_w) | set(cur_s)
        long_turn = sum(abs(cur_l.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
        short_turn = sum(abs(cur_s.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        turnover = (long_turn + short_turn) / 2.0
        cost = turnover * cost_bps_side * 2 / 1e4
        rows.append({"ts": ts, "spread_alpha": spread, "long_alpha": long_alpha,
                     "short_alpha": short_alpha, "turnover": turnover, "cost": cost,
                     "net_alpha": spread - cost, "n_universe": len(bar)})
        prev_long_w, prev_short_w = cur_l, cur_s
    return pd.DataFrame(rows)


# ---- gates: binary vs continuous --------------------------------------

def gate_rolling_binary(pnl: pd.DataFrame, regime: pd.DataFrame,
                         pctile: float = GATE_PCTILE,
                         window_days: int = GATE_WINDOW_DAYS) -> pd.DataFrame:
    regime = regime.sort_values("ts").reset_index(drop=True).copy()
    regime["thresh"] = (regime["disp_22d"]
                        .rolling(window=window_days, min_periods=60)
                        .quantile(pctile).shift(1))
    sub = pnl.merge(regime[["ts", "disp_22d", "thresh"]], on="ts", how="left")
    sub = sub.dropna(subset=["thresh"])
    return sub[sub["disp_22d"] >= sub["thresh"]].copy()


def gate_continuous(pnl: pd.DataFrame, regime: pd.DataFrame,
                     window_days: int = GATE_WINDOW_DAYS) -> pd.DataFrame:
    """Continuous: scale net_alpha and cost by regime-conditional position size.
    position_scale[t] = clip(quantile_rank(disp_22d[t-window:t-1]), 0, 1).
    Names below 50th pctile get scale 0, between 50-100 linearly scale to 1."""
    regime = regime.sort_values("ts").reset_index(drop=True).copy()
    # Trailing rank pctile of current disp vs trailing window
    def _trailing_pctile(s):
        out = pd.Series(np.nan, index=s.index)
        for i in range(60, len(s)):
            window = s.iloc[max(0, i - window_days):i]
            cur = s.iloc[i]
            if not np.isnan(cur) and len(window.dropna()) > 30:
                out.iloc[i] = (window < cur).sum() / len(window.dropna())
        return out
    regime["pct_rank"] = _trailing_pctile(regime["disp_22d"])
    # Position scale: 0 below 50th, ramp to 1 between 50-100th pctile
    regime["pos_scale"] = (2 * (regime["pct_rank"] - 0.5)).clip(0, 1)
    sub = pnl.merge(regime[["ts", "pos_scale"]], on="ts", how="left")
    sub = sub.dropna(subset=["pos_scale"])
    sub["spread_alpha"] = sub["spread_alpha"] * sub["pos_scale"]
    sub["cost"] = sub["cost"] * sub["pos_scale"]
    sub["net_alpha"] = sub["spread_alpha"] - sub["cost"]
    return sub.copy()


# ---- run helpers ------------------------------------------------------

def run_walk(panel: pd.DataFrame, feats: list[str], label: str,
              folds: list, port_fn, port_kwargs: dict) -> pd.DataFrame:
    all_pnls = []
    for fold in folds:
        train_start, train_end, test_start, test_end = fold
        train = panel[(panel["ts"] >= train_start) & (panel["ts"] <= train_end)].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        test_pred = fit_predict(train, test, feats, label)
        if test_pred.empty:
            continue
        lp = port_fn(test_pred, "pred", label, **port_kwargs)
        if not lp.empty:
            all_pnls.append(lp)
    return pd.concat(all_pnls, ignore_index=True) if all_pnls else pd.DataFrame()


def metrics_for(pnl: pd.DataFrame, hold_days_for_annu: int) -> dict:
    if pnl.empty:
        return {"n": 0}
    n = len(pnl)
    rebals_per_year = 252 / hold_days_for_annu
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
    return {
        "n_rebal": n,
        "net_bps_per_rebal": pnl["net_alpha"].mean() * 1e4,
        "active_sharpe": n_sh,
        "uncond_sharpe": uncond,
        "annual_return_pct": annual_mean * 100,
        "avg_turnover_pct": pnl["turnover"].mean() * 100,
        "annual_cost_bps": pnl["cost"].mean() * 1e4 * rebals_per_year_actual,
    }


def boot_ci(pnl: pd.DataFrame, hold_days_for_annu: int,
             n_boot: int = 2000) -> tuple[float, float]:
    if pnl.empty or len(pnl) < 30:
        return np.nan, np.nan
    n = len(pnl)
    rng = np.random.default_rng(42)
    rpy = 252 / hold_days_for_annu
    sh = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = pnl["net_alpha"].iloc[idx]
        if s.std() > 0:
            sh.append(s.mean() / s.std() * np.sqrt(rpy))
    if not sh:
        return np.nan, np.nan
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))


# ---- main -------------------------------------------------------------

def main() -> None:
    log.info("loading panel...")
    panel, earnings, _ = load_universe()
    if panel.empty:
        return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)

    # Build labels for both 1d and 3d
    panel = add_residual_and_label(panel, 3)
    panel = add_residual_and_label(panel, 1)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    feats = feats_A + feats_B + ["sym_id"]
    allowed = set(XYZ_IN_SP100)

    # Folds
    folds_exp = make_folds_v2(panel, train_window_days=None,
                                train_min_days=365 * 3, test_days=365)
    folds_roll = make_folds_v2(panel, train_window_days=365 * 5,
                                 train_min_days=365 * 3, test_days=365)
    log.info("folds: expanding=%d, rolling-5y=%d", len(folds_exp), len(folds_roll))

    log.info("\n=== Tier-A optimizations + daily rebalance ===")
    log.info("  %-30s  %5s  %12s  %10s  %10s  %10s  %10s",
             "config", "n_reb", "net/rebal", "active_Sh", "uncond_Sh",
             "ann_ret%", "turn%/rb")

    results = {}

    # C0: baseline (3d hold, equal-weight, expanding, binary gate)
    log.info("\n>>> C0 baseline (3d, equal-weight, expanding, binary gate)")
    pnl0 = run_walk(panel, feats, "fwd_resid_3d", folds_exp,
                     portfolio_baseline,
                     {"allowed": allowed, "top_k": TOP_K,
                      "cost_bps_side": COST_BPS_SIDE, "hold_days": HOLD_DAYS})
    pnl0 = gate_rolling_binary(pnl0, regime)
    results["C0 baseline"] = (pnl0, HOLD_DAYS)

    # C1: + vol-target sizing
    log.info(">>> C1 + vol-target sizing")
    pnl1 = run_walk(panel, feats, "fwd_resid_3d", folds_exp,
                     portfolio_voltarget,
                     {"allowed": allowed, "top_k": TOP_K,
                      "cost_bps_side": COST_BPS_SIDE, "hold_days": HOLD_DAYS})
    pnl1 = gate_rolling_binary(pnl1, regime)
    results["C1 vol-target"] = (pnl1, HOLD_DAYS)

    # C2: + rolling 5y window
    log.info(">>> C2 vol-target + rolling 5y window")
    pnl2 = run_walk(panel, feats, "fwd_resid_3d", folds_roll,
                     portfolio_voltarget,
                     {"allowed": allowed, "top_k": TOP_K,
                      "cost_bps_side": COST_BPS_SIDE, "hold_days": HOLD_DAYS})
    pnl2 = gate_rolling_binary(pnl2, regime)
    results["C2 +rolling 5y"] = (pnl2, HOLD_DAYS)

    # C3: C2 + continuous regime
    log.info(">>> C3 + continuous regime sizing")
    pnl3_ungated = run_walk(panel, feats, "fwd_resid_3d", folds_roll,
                             portfolio_voltarget,
                             {"allowed": allowed, "top_k": TOP_K,
                              "cost_bps_side": COST_BPS_SIDE, "hold_days": HOLD_DAYS})
    pnl3 = gate_continuous(pnl3_ungated, regime)
    results["C3 +continuous regime"] = (pnl3, HOLD_DAYS)

    # C4: daily rebalance (no other opts), label = fwd_resid_1d, smooth=3
    log.info(">>> C4 daily rebalance, smooth=3, vol-target")
    pnl4_pre = run_walk(panel, feats, "fwd_resid_1d", folds_exp,
                         portfolio_daily_smoothed,
                         {"allowed": allowed, "top_k": TOP_K,
                          "cost_bps_side": COST_BPS_SIDE, "smooth_days": 3,
                          "vol_col": "A_idio_vol_22d"})
    pnl4 = gate_rolling_binary(pnl4_pre, regime)
    results["C4 daily smooth=3"] = (pnl4, 1)

    # C5: full stack — daily smoothed + vol-target + rolling 5y + continuous regime
    log.info(">>> C5 full stack (daily smooth + voltarget + rolling 5y + continuous regime)")
    pnl5_pre = run_walk(panel, feats, "fwd_resid_1d", folds_roll,
                         portfolio_daily_smoothed,
                         {"allowed": allowed, "top_k": TOP_K,
                          "cost_bps_side": COST_BPS_SIDE, "smooth_days": 3,
                          "vol_col": "A_idio_vol_22d"})
    pnl5 = gate_continuous(pnl5_pre, regime)
    results["C5 full stack"] = (pnl5, 1)

    # Report
    log.info("\n=== SUMMARY ===")
    log.info("  %-30s  %5s  %10s  %18s  %10s  %10s  %10s",
             "config", "n_reb", "active_Sh", "95% CI",
             "uncond_Sh", "ann_ret%", "turn%/rb")
    for name, (pnl, hd) in results.items():
        m = metrics_for(pnl, hd)
        if m.get("n_rebal", 0) == 0:
            continue
        lo, hi = boot_ci(pnl, hd)
        log.info("  %-30s  %5d  %+8.2f  [%+5.2f,%+5.2f]  %+8.2f  %+8.2f%%  %8.0f%%",
                 name, m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["uncond_sharpe"], m["annual_return_pct"],
                 m["avg_turnover_pct"])


if __name__ == "__main__":
    main()
