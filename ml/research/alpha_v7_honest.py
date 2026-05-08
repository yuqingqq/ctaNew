"""Honest backtest of v7 xyz strategy — addresses 3 reliability issues:

  Fix 1: Rolling-window dispersion gate (no look-ahead).
         Use trailing 252-day dispersion percentile, .shift(1) for PIT.
         Replaces in-sample full-period quantile.

  Fix 2: Hard out-of-sample period validation.
         Train ONCE on 2013-2019 with frozen hyperparams + features.
         Test on 2020-2026 with no retraining.
         This is the strictest test — eliminates expanding-window
         "training set keeps growing" benefit.

  Fix 3: Fixed feature set with fixed hyperparams.
         No ablation, no cherry-picking. Use A + B (fixed PEAD) only.
         top_K=5, hold=3d, gate=60th rolling percentile, K-bar features.
         All decisions made BEFORE seeing results.

Compares 4 configurations:
  C0  Original (in-sample gate, walk-forward)            ← biased, +1.63
  C1  Rolling gate (PIT) + walk-forward                  ← Fix 1 only
  C2  Rolling gate + hard train-2013-2019 / test-2020+   ← Fix 1 + 2
  C3  Same as C2 but use A+B only (no further selection) ← Fix 1 + 2 + 3
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
from ml.research.alpha_v7_weekly import make_folds
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz import construct_portfolio_subset
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100
from ml.research.alpha_v7_freq_sweep import (
    add_residual_and_label, metrics_freq, annualized_unconditional,
)
from ml.research.alpha_v7_pead_fixed import add_features_B_fixed

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HOLD_DAYS = 3
COST_BPS_SIDE = 2.5
TOP_K = 5
GATE_PCTILE = 0.6
GATE_WINDOW_DAYS = 252  # trailing 1y for rolling gate


# ---- gates --------------------------------------------------------------

def gate_in_sample(pnl: pd.DataFrame, regime: pd.DataFrame,
                    pctile: float = GATE_PCTILE) -> pd.DataFrame:
    """Original (look-ahead) gate using full-sample percentile."""
    sub = pnl.merge(regime[["ts", "disp_22d"]], on="ts", how="left")
    if sub["disp_22d"].isna().all():
        return sub
    thresh = sub["disp_22d"].dropna().quantile(pctile)
    return sub[sub["disp_22d"] >= thresh].copy()


def gate_rolling(pnl: pd.DataFrame, regime: pd.DataFrame,
                  pctile: float = GATE_PCTILE,
                  window_days: int = GATE_WINDOW_DAYS) -> pd.DataFrame:
    """PIT-correct gate: trailing window quantile, shifted by 1 day."""
    regime = regime.sort_values("ts").reset_index(drop=True).copy()
    regime["thresh"] = (regime["disp_22d"]
                        .rolling(window=window_days, min_periods=60)
                        .quantile(pctile)
                        .shift(1))  # use only past data
    sub = pnl.merge(regime[["ts", "disp_22d", "thresh"]], on="ts", how="left")
    sub = sub.dropna(subset=["thresh"])
    return sub[sub["disp_22d"] >= sub["thresh"]].copy()


# ---- fit + predict ------------------------------------------------------

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


# ---- walk-forward (the C1 setup) ---------------------------------------

def walk_forward(panel: pd.DataFrame, features: list[str], label: str,
                 folds: list, top_k: int, cost_bps_side: float,
                 hold_days: int, allowed: set) -> pd.DataFrame:
    all_pnls = []
    for fold in folds:
        train_end, test_start, test_end = fold
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        test_pred = fit_predict(train, test, features, label)
        if test_pred.empty:
            continue
        lp = construct_portfolio_subset(
            test_pred, "pred", label, allowed_symbols=allowed,
            top_k=top_k, cost_bps=cost_bps_side * 2, hold_days=hold_days,
        )
        if not lp.empty:
            all_pnls.append(lp)
    return pd.concat(all_pnls, ignore_index=True) if all_pnls else pd.DataFrame()


# ---- hard split: train once 2013-2019, test 2020-2026 (C2 setup) -------

def hard_split_test(panel: pd.DataFrame, features: list[str], label: str,
                     train_end: pd.Timestamp,
                     test_start: pd.Timestamp, test_end: pd.Timestamp,
                     top_k: int, cost_bps_side: float,
                     hold_days: int, allowed: set) -> pd.DataFrame:
    train = panel[panel["ts"] <= train_end].copy()
    test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
    log.info("  hard split: train n=%d (≤%s)  test n=%d (%s..%s)",
             len(train), train_end.strftime("%Y-%m-%d"),
             len(test), test_start.strftime("%Y-%m-%d"),
             test_end.strftime("%Y-%m-%d"))
    test_pred = fit_predict(train, test, features, label)
    if test_pred.empty:
        return pd.DataFrame()
    lp = construct_portfolio_subset(
        test_pred, "pred", label, allowed_symbols=allowed,
        top_k=top_k, cost_bps=cost_bps_side * 2, hold_days=hold_days,
    )
    return lp


# ---- bootstrap CI -------------------------------------------------------

def bootstrap_active_sharpe_ci(pnl: pd.DataFrame, hold_days: int,
                                n_boot: int = 2000) -> tuple[float, float]:
    if pnl.empty or len(pnl) < 30:
        return np.nan, np.nan
    n = len(pnl)
    rng = np.random.default_rng(42)
    rebals_per_year_max = 252 / hold_days
    sharpes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = pnl["net_alpha"].iloc[idx]
        if sample.std() > 0:
            sharpes.append(sample.mean() / sample.std() * np.sqrt(rebals_per_year_max))
    if not sharpes:
        return np.nan, np.nan
    return float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))


# ---- report -------------------------------------------------------------

def report(name: str, pnl: pd.DataFrame, hold_days: int = HOLD_DAYS) -> dict:
    if pnl.empty:
        log.info("  %s: NO RESULTS", name); return {}
    m = metrics_freq(pnl, hold_days)
    ann = annualized_unconditional(pnl, hold_days)
    lo, hi = bootstrap_active_sharpe_ci(pnl, hold_days)
    log.info("  %-50s  n=%4d  net/d=%+6.2f  active_Sh=%+5.2f  [%+5.2f,%+5.2f]  uncond=%+5.2f  ann=%+5.2f%%",
             name, m["n_rebal"], m["net_bps_per_day"],
             m["active_sharpe_annu"], lo, hi,
             ann["unconditional_sharpe"], ann["annual_return_pct"])
    return {"m": m, "ann": ann, "ci": (lo, hi), "pnl": pnl}


# ---- main ---------------------------------------------------------------

def main() -> None:
    log.info("loading universe + earnings + anchors...")
    panel, earnings, _ = load_universe()
    if panel.empty:
        return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    panel = add_residual_and_label(panel, HOLD_DAYS)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    label = f"fwd_resid_{HOLD_DAYS}d"
    feats = feats_A + feats_B + ["sym_id"]
    log.info("Fixed feature set: A+B+sym_id (%d features)", len(feats))
    folds = make_folds(panel, train_min_days=365 * 3, test_days=365)

    # Run base walk-forward (no gate yet)
    log.info("\nRunning walk-forward (base predictions)...")
    pnl_base = walk_forward(panel, feats, label, folds,
                             top_k=TOP_K, cost_bps_side=COST_BPS_SIDE,
                             hold_days=HOLD_DAYS,
                             allowed=set(XYZ_IN_SP100))
    log.info("base walk-forward: %d rebalances", len(pnl_base))

    log.info("\n=== ABLATION ===")
    log.info("  %-50s  %4s  %6s  %10s  %16s  %8s  %8s",
             "config", "n", "net/d", "active_Sh", "95% CI", "uncond", "ann_ret%")

    # C0: in-sample gate (the look-ahead reference number, +1.63)
    pnl_c0 = gate_in_sample(pnl_base, regime, pctile=GATE_PCTILE)
    r0 = report("C0  Look-ahead in-sample gate (current claim)", pnl_c0)

    # C1: rolling gate (PIT)
    pnl_c1 = gate_rolling(pnl_base, regime, pctile=GATE_PCTILE,
                           window_days=GATE_WINDOW_DAYS)
    r1 = report("C1  PIT rolling gate (252d, ≥60th pctile)", pnl_c1)

    # C2: hard 2013-2019 train, 2020-2026 test, with rolling gate
    log.info("\nC2: hard split 2013-2019 train → 2020-2026 test (no retraining)...")
    pnl_c2_raw = hard_split_test(
        panel, feats, label,
        train_end=pd.Timestamp("2019-12-31", tz="UTC"),
        test_start=pd.Timestamp("2020-01-01", tz="UTC"),
        test_end=pd.Timestamp("2026-05-05", tz="UTC"),
        top_k=TOP_K, cost_bps_side=COST_BPS_SIDE,
        hold_days=HOLD_DAYS, allowed=set(XYZ_IN_SP100),
    )
    pnl_c2 = gate_rolling(pnl_c2_raw, regime, pctile=GATE_PCTILE,
                           window_days=GATE_WINDOW_DAYS)
    r2 = report("C2  Hard split + PIT rolling gate", pnl_c2)

    # C3: same as C2 but with explicit "fixed feature set, no further tuning"
    # (already C2's setup since we only used A+B+sym_id with fixed PEAD).
    # Add an alternative: drop sym_id (a parameter we may have implicitly tuned)
    log.info("\nC3: hard split + rolling gate + drop sym_id (test feature minimalism)...")
    feats_no_symid = feats_A + feats_B
    pnl_c3_raw = hard_split_test(
        panel, feats_no_symid, label,
        train_end=pd.Timestamp("2019-12-31", tz="UTC"),
        test_start=pd.Timestamp("2020-01-01", tz="UTC"),
        test_end=pd.Timestamp("2026-05-05", tz="UTC"),
        top_k=TOP_K, cost_bps_side=COST_BPS_SIDE,
        hold_days=HOLD_DAYS, allowed=set(XYZ_IN_SP100),
    )
    pnl_c3 = gate_rolling(pnl_c3_raw, regime, pctile=GATE_PCTILE,
                           window_days=GATE_WINDOW_DAYS)
    r3 = report("C3  Hard split + PIT gate + no sym_id", pnl_c3)

    # Per-year for the most honest config (C2)
    if r2 and not r2["pnl"].empty:
        pnl_y = r2["pnl"].copy()
        pnl_y["year"] = pnl_y["ts"].dt.year
        log.info("\n=== Per-year for C2 (hard split + PIT gate, OOS only 2020-2026) ===")
        log.info("  %-6s %5s %12s %12s %10s",
                 "year", "n_reb", "gross/d", "net/d", "active_Sh")
        for y, g in pnl_y.groupby("year"):
            qm = metrics_freq(g, HOLD_DAYS)
            log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+10.2f",
                     y, qm["n_rebal"], qm["gross_bps_per_day"],
                     qm["net_bps_per_day"], qm["active_sharpe_annu"])

    # Summary table
    log.info("\n=== HONEST RELIABILITY SUMMARY ===")
    log.info("  %-50s  %4s  %10s  %16s  %8s",
             "config", "n", "active_Sh", "95% CI", "uncond")
    for name, r in [("C0 in-sample gate (biased headline)", r0),
                     ("C1 PIT rolling gate", r1),
                     ("C2 hard split + PIT gate", r2),
                     ("C3 hard split + PIT gate + no sym_id", r3)]:
        if not r:
            continue
        m, ann, ci = r["m"], r["ann"], r["ci"]
        log.info("  %-50s  %4d  %+8.2f  [%+5.2f,%+5.2f]  %+8.2f",
                 name, m["n_rebal"], m["active_sharpe_annu"],
                 ci[0], ci[1], ann["unconditional_sharpe"])


if __name__ == "__main__":
    main()
