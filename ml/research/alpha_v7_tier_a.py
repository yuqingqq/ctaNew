"""Re-validate v7 spec on Tier A 8-name execution universe (K=3, M=1).

Optimization: predictions are computed ONCE per fold and cached to parquet.
Portfolio sweeps re-use the cached predictions (cheap), making K/M sensitivity
analysis ~50× faster than re-training per config.

Compares 4 configs:
  C1: 15 names, K=5, M=2  — backtest reference (memory: +3.16 active Sh)
  C2: 8 Tier A,   K=3, M=1  — production candidate
  C3: 8 Tier A,   K=2, M=1  — more conservative
  C4: 8 Tier A,   K=2, M=2  — most conservative

Usage:
    python -m ml.research.alpha_v7_tier_a              # run all 4 configs
    python -m ml.research.alpha_v7_tier_a --rebuild    # force re-train (ignore cache)
"""
from __future__ import annotations

import argparse
import logging
import warnings
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
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100
from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_daily_optimized import (
    daily_portfolio_hysteresis, make_folds, metrics_for, boot_ci,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TIER_A = ["AAPL", "GOOGL", "META", "MSFT", "MU", "NVDA", "PLTR", "TSLA"]
TIER_B = ["AMZN", "ORCL", "NFLX"]
TIER_AB = TIER_A + TIER_B
COST_BPS_SIDE = 1.5  # default patient-maker assumption
COST_BPS_SIDE_LOW = 0.8  # current xyz taker (<1 bps per user)
GATE_PCTILE = 0.6
GATE_WINDOW = 252
HORIZONS = (3, 5, 10)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
PRED_CACHE = CACHE / "v7_tier_a_walkfwd_preds.parquet"


def compute_walkforward_preds(panel, feats, folds) -> pd.DataFrame:
    """Train ensemble per fold, return DataFrame with one row per (sym, ts)
    in the OOS test windows, columns: ts, symbol, pred, all the fwd_resid_*d.

    Predictions are averaged across HORIZONS × SEEDS.
    """
    out_rows = []
    pnl_label_cols = [f"fwd_resid_{h}d" for h in (1,) + HORIZONS]
    for fi, (te, ts, te2) in enumerate(folds, 1):
        log.info("fold %d/%d  train≤%s  test=[%s, %s]",
                 fi, len(folds), te.date(), ts.date(), te2.date())
        train = panel[panel["ts"] <= te]
        test = panel[(panel["ts"] >= ts) & (panel["ts"] <= te2)]
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
            for seed in SEEDS:
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
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def evaluate_config(preds: pd.DataFrame, regime: pd.DataFrame, *,
                     allowed: set, K: int, M: int, name: str,
                     cost_bps_side: float = COST_BPS_SIDE) -> dict | None:
    log.info(">>> %s  (universe=%d, K=%d, M=%d, cost=%.1f bps/side)",
             name, len(allowed), K, M, cost_bps_side)
    pnl_pre = daily_portfolio_hysteresis(
        preds, "pred", "fwd_resid_1d",
        allowed, K, M, cost_bps_side,
    )
    if pnl_pre.empty:
        log.warning("  empty pnl"); return None
    pnl = gate_rolling(pnl_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    if pnl.empty:
        log.warning("  empty after gate"); return None
    m = metrics_for(pnl, 1)
    lo, hi = boot_ci(pnl, 1)
    log.info("  n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f ann=%+.2f%%",
             m["n_rebal"], m["active_sharpe"], lo, hi,
             m["uncond_sharpe"], m["annual_return_pct"])
    pnl["year"] = pnl["ts"].dt.year
    yr_lines = []
    for y, g in pnl.groupby("year"):
        if len(g) < 5: continue
        ym = metrics_for(g, 1)
        yr_lines.append((y, ym["n_rebal"], ym["net_bps_per_rebal"],
                         ym["active_sharpe"]))
    return {"name": name, "metrics": m, "ci": (lo, hi), "pnl": pnl,
            "per_year": yr_lines}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                         help="force re-train (ignore prediction cache)")
    args = parser.parse_args()

    log.info("loading panel...")
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
    regime = compute_regime_indicators(panel, anchors)

    feats = feats_A + feats_B + feats_F + ["sym_id"]
    folds = make_folds(panel)
    log.info("panel: %d rows, %d folds, %d feats", len(panel), len(folds), len(feats))

    if PRED_CACHE.exists() and not args.rebuild:
        log.info("loading cached predictions: %s", PRED_CACHE)
        preds = pd.read_parquet(PRED_CACHE)
        log.info("  preds: %d rows, %d symbols",
                 len(preds), preds["symbol"].nunique())
    else:
        log.info("computing walk-forward predictions (cache miss)...")
        preds = compute_walkforward_preds(panel, feats, folds)
        if preds.empty:
            log.error("empty preds"); return
        PRED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(PRED_CACHE)
        log.info("  cached %d pred rows to %s", len(preds), PRED_CACHE)

    out = []
    # Cost-sensitivity sweep on Tier A+B (the chosen production preset).
    # Real per-side cost = slippage + fee. Today's L2 measurement showed
    # avg slippage +2.87 bps/side at $10k/leg + 0.8 bps fee = ~3.67 bps/side.
    for cost in (0.8, 1.5, 2.5, 3.5, 5.0, 7.0):
        out.append(evaluate_config(
            preds, regime, allowed=set(TIER_AB), K=4, M=1,
            cost_bps_side=cost,
            name=f"Tier A+B (K=4, M=1) @{cost:.1f}bps/side"))
    # Reference: 15-name backtest at low and realistic cost
    out.append(evaluate_config(
        preds, regime, allowed=set(XYZ_IN_SP100), K=5, M=2,
        cost_bps_side=0.8, name="15 names (K=5, M=2) @0.8 (ref)"))
    out.append(evaluate_config(
        preds, regime, allowed=set(XYZ_IN_SP100), K=5, M=2,
        cost_bps_side=3.5, name="15 names (K=5, M=2) @3.5 (real)"))

    log.info("\n=== SUMMARY ===")
    log.info("  %-50s %5s %10s %18s %10s %10s",
             "config", "n", "active_Sh", "95% CI", "uncond", "ann_ret%")
    for r in out:
        if r is None: continue
        m = r["metrics"]; lo, hi = r["ci"]
        log.info("  %-50s %5d %+8.2f [%+5.2f,%+5.2f] %+8.2f %+8.2f",
                 r["name"], m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["uncond_sharpe"], m["annual_return_pct"])

    log.info("\n=== Per-year (best config) ===")
    best = max([r for r in out if r is not None],
               key=lambda r: r["metrics"]["active_sharpe"])
    log.info("  best: %s", best["name"])
    log.info("  %-6s %5s %12s %10s", "year", "n", "net/d", "active_Sh")
    for y, n, ndd, sh in best["per_year"]:
        log.info("  %-6d %5d %+10.2fbps %+10.2f", y, n, ndd, sh)


if __name__ == "__main__":
    main()
