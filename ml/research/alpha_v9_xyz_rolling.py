"""Rolling-window v7-style LGBM, compare to expanding-window v7 baseline.

Mirrors v7 spec exactly (same 18 features, target = fwd_resid_5d, P&L on
fwd_resid_1d, hyperparams unchanged) but trains on a rolling window
instead of expanding.

Folds: 24mo training, 3mo test, 3mo step. Walk-forward across 2024-2026
to align with the intraday-strategy comparison window.

Tests:
  1. Standalone Sharpe of rolling on Tier A+B vs v7 cached (expanding)
  2. Correlation between rolling preds and v7 preds
  3. Naive portfolio-level blend: w * v7 + (1-w) * rolling
  4. Per-fold breakdown

Hypothesis being tested: a shorter training window adapts faster to
regime shifts (e.g., 2024-mid PEAD weakness, 2025+ strong regime). If
true, rolling outperforms in recent OOS or adds orthogonal value to v7.

Memory note: crypto v6_clean tested rolling 5y vs full-history and found
"-0.84 Sh, more training data is better." Equity might differ — this
is the test.

Usage:
    python -m ml.research.alpha_v9_xyz_rolling [--window-months 24]
"""
from __future__ import annotations

import argparse
import logging
import warnings
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_collectors.sp100_loader import load_universe
from ml.research.alpha_v7_freq_sweep import add_residual_and_label
from ml.research.alpha_v7_multi import (
    LGB_PARAMS, SEEDS, add_features_A, add_returns_and_basket, load_anchors,
)
from ml.research.alpha_v7_pead_fixed import add_features_B_fixed
from ml.research.alpha_v7_push import add_sector_features
from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_daily_optimized import (
    daily_portfolio_hysteresis, metrics_for, boot_ci,
)
from ml.research.alpha_v9_xyz_pm import load_or_compute_regime
from ml.research.alpha_v7_tier_a import TIER_AB

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
PRED_CACHE_V7 = CACHE / "v7_tier_a_walkfwd_preds.parquet"
HORIZONS = (3, 5, 10)
RIDUCED_SEEDS = SEEDS[:3]  # 3 seeds × 3 horizons = 9 models per fold (vs 15 in full v7)
GATE_PCTILE = 0.6
GATE_WINDOW = 252
COST_BPS_SIDE = 0.8


def make_rolling_folds(panel: pd.DataFrame, window_months: int = 24,
                         test_months: int = 3,
                         start_test: pd.Timestamp = pd.Timestamp("2024-06-01", tz="UTC"),
                         end_test: pd.Timestamp = pd.Timestamp("2026-05-01", tz="UTC")
                         ) -> list[dict]:
    """Rolling folds: each fold's train_window is the window_months prior to
    test_start. Folds advance by test_months each step."""
    folds = []
    cur = start_test
    while cur < end_test:
        train_start = cur - pd.DateOffset(months=window_months)
        train_end = cur
        test_start = cur
        test_end = cur + pd.DateOffset(months=test_months)
        if test_end > end_test:
            test_end = end_test
        folds.append({
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
        })
        cur = test_end
    return folds


def compute_rolling_preds(panel: pd.DataFrame, feats: list[str],
                            folds: list[dict]) -> pd.DataFrame:
    out_rows = []
    pnl_label_cols = [f"fwd_resid_{h}d" for h in (1,) + HORIZONS]
    for fi, fold in enumerate(folds, 1):
        log.info("rolling fold %d/%d  train=[%s, %s)  test=[%s, %s)",
                 fi, len(folds),
                 fold["train_start"].date(), fold["train_end"].date(),
                 fold["test_start"].date(), fold["test_end"].date())
        train = panel[(panel["ts"] >= fold["train_start"]) &
                       (panel["ts"] < fold["train_end"])]
        test = panel[(panel["ts"] >= fold["test_start"]) &
                      (panel["ts"] < fold["test_end"])]
        if len(train) < 1000 or len(test) < 100:
            log.warning("  skip: train=%d test=%d", len(train), len(test))
            continue
        ensemble_preds = []
        sub_test = None
        for h in HORIZONS:
            label = f"fwd_resid_{h}d"
            train_ = train.dropna(subset=feats + [label])
            if len(train_) < 1000:
                continue
            sub = test.dropna(subset=feats).copy()
            if sub_test is None:
                sub_test = sub.copy()
            X_train = train_[feats].to_numpy(dtype=np.float32)
            y_train = train_[label].to_numpy(dtype=np.float32)
            X_test = sub[feats].to_numpy(dtype=np.float32)
            seed_preds = []
            for seed in RIDUCED_SEEDS:
                m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
                m.fit(X_train, y_train)
                seed_preds.append(m.predict(X_test))
            ensemble_preds.append(np.mean(seed_preds, axis=0))
        if not ensemble_preds or sub_test is None:
            continue
        sub_test = sub_test.copy()
        sub_test["pred"] = np.mean(ensemble_preds, axis=0)
        keep = ["ts", "symbol", "pred"] + [c for c in pnl_label_cols
                                              if c in sub_test.columns]
        out_rows.append(sub_test[keep])
    if not out_rows: return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-months", type=int, default=24)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    cache_path = CACHE / f"v9_rolling_{args.window_months}mo_preds.parquet"

    log.info("loading panel ...")
    panel, earnings, _ = load_universe()
    if panel.empty: return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    for h in (1,) + HORIZONS:
        panel = add_residual_and_label(panel, h)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel, feats_F = add_sector_features(panel)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    feats = feats_A + feats_B + feats_F + ["sym_id"]
    log.info("  panel: %d rows, %d feats", len(panel), len(feats))

    if cache_path.exists() and not args.rebuild:
        log.info("loading cached rolling preds: %s", cache_path)
        rolling_preds = pd.read_parquet(cache_path)
    else:
        folds = make_rolling_folds(panel, window_months=args.window_months)
        log.info("rolling folds: %d  window=%dmo  test_months=3",
                 len(folds), args.window_months)
        rolling_preds = compute_rolling_preds(panel, feats, folds)
        if rolling_preds.empty:
            log.error("empty rolling preds"); return
        rolling_preds.to_parquet(cache_path)
        log.info("  cached %d rolling pred rows to %s",
                 len(rolling_preds), cache_path)

    log.info("loading v7 cached preds (expanding-window baseline) ...")
    v7_preds = pd.read_parquet(PRED_CACHE_V7)

    # Restrict both to common ts × symbol intersection in the OOS window
    rolling_preds["ts"] = pd.to_datetime(rolling_preds["ts"], utc=True)
    v7_preds["ts"] = pd.to_datetime(v7_preds["ts"], utc=True)
    common_ts = sorted(set(rolling_preds["ts"]) & set(v7_preds["ts"]))
    log.info("  rolling ts: %d  v7 ts: %d  common: %d",
             rolling_preds["ts"].nunique(), v7_preds["ts"].nunique(),
             len(common_ts))

    rolling_sub = rolling_preds[rolling_preds["ts"].isin(common_ts)].copy()
    v7_sub = v7_preds[v7_preds["ts"].isin(common_ts)].copy()

    # Pred correlation per (ts, symbol)
    merged = rolling_sub[["ts", "symbol", "pred"]].rename(
        columns={"pred": "pred_rolling"}).merge(
        v7_sub[["ts", "symbol", "pred"]].rename(
            columns={"pred": "pred_v7"}), on=["ts", "symbol"])
    merged_xs = merged.copy()
    # Cross-sectional residualize each
    for c in ["pred_rolling", "pred_v7"]:
        merged_xs[c + "_xs"] = (
            merged_xs[c] - merged_xs.groupby("ts")[c].transform("median"))
    rho = merged_xs[["pred_rolling_xs", "pred_v7_xs"]].corr().iloc[0, 1]
    log.info("\n  corr(rolling pred, v7 pred) cross-sectionally residualized: %+.3f", rho)

    # Run portfolio for each
    regime = load_or_compute_regime()

    log.info("\n=== Standalone portfolio Sharpe (Tier A+B, K=4 M=1, gate, 0.8 bps/side) ===")
    log.info("  %-22s %5s %10s %18s %12s",
             "config", "n", "active_Sh", "95% CI", "net bps/cyc")
    for label, df in [("v7 (expanding)", v7_sub), ("rolling-%dmo" % args.window_months, rolling_sub)]:
        pre = daily_portfolio_hysteresis(df, "pred", "fwd_resid_1d",
                                            set(TIER_AB), 4, 1, COST_BPS_SIDE)
        pnl = gate_rolling(pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
        if pnl.empty:
            log.info("  %-22s empty", label); continue
        m = metrics_for(pnl, 1)
        lo, hi = boot_ci(pnl, 1)
        log.info("  %-22s %5d %+8.2f [%+5.2f,%+5.2f] %+10.2f",
                 label, m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["net_bps_per_rebal"])

    # ---- portfolio-level blend ----
    log.info("\n=== Portfolio-level blend: w_v7 × v7 + (1-w_v7) × rolling ===")
    pre_v7 = daily_portfolio_hysteresis(v7_sub, "pred", "fwd_resid_1d",
                                          set(TIER_AB), 4, 1, COST_BPS_SIDE)
    pnl_v7 = gate_rolling(pre_v7, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    pre_r = daily_portfolio_hysteresis(rolling_sub, "pred", "fwd_resid_1d",
                                          set(TIER_AB), 4, 1, COST_BPS_SIDE)
    pnl_r = gate_rolling(pre_r, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)

    merged_pnl = pnl_v7[["ts", "net_alpha"]].rename(columns={"net_alpha": "v7_net"}).merge(
        pnl_r[["ts", "net_alpha"]].rename(columns={"net_alpha": "r_net"}),
        on="ts", how="inner")
    rho_pnl = merged_pnl[["v7_net", "r_net"]].corr().iloc[0, 1]
    log.info("  shared ts: %d  corr(v7_pnl, rolling_pnl) = %+.3f",
             len(merged_pnl), rho_pnl)

    log.info("\n  %-12s %5s %10s %14s %14s",
             "weight", "n", "Sharpe", "net bps/cyc", "ΔSh vs v7")
    for w_v7 in [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.30, 0.00]:
        merged_pnl["blend"] = w_v7 * merged_pnl["v7_net"] + (1 - w_v7) * merged_pnl["r_net"]
        n = len(merged_pnl)
        mu = merged_pnl["blend"].mean(); sd = merged_pnl["blend"].std()
        sh = mu / sd * np.sqrt(252) if sd > 0 else 0
        sh_v7 = (merged_pnl["v7_net"].mean() / merged_pnl["v7_net"].std()
                  * np.sqrt(252)) if merged_pnl["v7_net"].std() > 0 else 0
        log.info("  w_v7=%.2f   %5d %+8.2f %+12.2f      %+10.2f",
                 w_v7, n, sh, mu * 1e4, sh - sh_v7)

    # ---- per-fold breakdown ----
    log.info("\n=== Per-quarter Sharpe on the OOS window ===")
    log.info("  %-10s %5s %10s %10s",
             "quarter", "n", "v7 Sh", "rolling Sh")
    merged_pnl["quarter"] = merged_pnl["ts"].dt.to_period("Q").astype(str)
    for q, g in merged_pnl.groupby("quarter"):
        n = len(g)
        if n < 5: continue
        sh_v7 = (g["v7_net"].mean() / g["v7_net"].std()
                  * np.sqrt(252)) if g["v7_net"].std() > 0 else 0
        sh_r = (g["r_net"].mean() / g["r_net"].std()
                 * np.sqrt(252)) if g["r_net"].std() > 0 else 0
        log.info("  %-10s %5d %+8.2f   %+8.2f", q, n, sh_v7, sh_r)


if __name__ == "__main__":
    main()
