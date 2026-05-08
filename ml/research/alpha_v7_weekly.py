"""v7 multi-alpha — weekly variant.

Same features as alpha_v7_multi.py (groups A/B/C/D), but:
  - Forward target = fwd_resid_5d (sum of next 5 days' residuals)
  - Rebalance every 5 trading days (weekly, hold 5d)
  - Cost charged per rebalance (so 1/5 the daily-rebalance burn)

Tests whether the multi-alpha edge (gross +4.26 bps/d in daily) survives
into a tradeable net Sharpe once we stop burning it on daily turnover.

If net Sharpe lower-CI > 0: alpha is real and tradeable, feature redesign
becomes optional optimization.
If still ~0: features need redesign for the weekly horizon (path 2).
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
    ANCHOR_TICKERS, BETA_WINDOW, LGB_PARAMS, SEEDS, TOP_K,
    PEAD_MAX_DAYS,
    load_anchors, add_returns_and_basket,
    add_features_A, add_features_B, add_features_C, add_features_D,
    metrics, bootstrap_ci,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FWD_DAYS = 5                        # 5-day forward residual
HOLD_DAYS = 5                       # rebalance every 5 trading days
COST_PER_TRADE_BPS = 5              # per-rebalance cost


def add_residual_5d(panel: pd.DataFrame) -> pd.DataFrame:
    """Same beta+resid as v7, but label is sum of next 5 days' residuals."""
    def _beta(g):
        cov = (g["ret"] * g["bk_ret"]).rolling(BETA_WINDOW).mean() - \
              g["ret"].rolling(BETA_WINDOW).mean() * g["bk_ret"].rolling(BETA_WINDOW).mean()
        var = g["bk_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
        return (cov / var).clip(-5, 5).shift(1)
    panel["beta"] = panel.groupby("symbol", group_keys=False).apply(_beta).values
    panel["resid"] = panel["ret"] - panel["beta"] * panel["bk_ret"]
    # Forward 5-day cumulative residual
    panel["fwd_resid_5d"] = (panel.groupby("symbol", group_keys=False)["resid"]
                             .apply(lambda s: s.rolling(FWD_DAYS).sum().shift(-FWD_DAYS))
                             .values)
    return panel


def fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                features: list[str], label: str = "fwd_resid_5d") -> pd.DataFrame:
    train_ = train.dropna(subset=features + [label])
    if len(train_) < 1000:
        return pd.DataFrame()
    preds = []
    sub = test.dropna(subset=features).copy()
    for seed in SEEDS:
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(train_[features], train_[label])
        preds.append(m.predict(sub[features]))
    sub["pred"] = np.mean(preds, axis=0)
    return sub


def construct_portfolio_weekly(test_pred: pd.DataFrame, signal: str,
                                pnl_label: str, top_k: int = TOP_K,
                                cost_bps: float = COST_PER_TRADE_BPS,
                                hold_days: int = HOLD_DAYS) -> pd.DataFrame:
    """Rebalance every hold_days trading days. Hold 5 days, realize fwd_resid_5d."""
    sub = test_pred.dropna(subset=[signal, pnl_label]).copy()
    unique_ts = sorted(sub["ts"].unique())
    if not unique_ts:
        return pd.DataFrame()
    rebal_ts = unique_ts[::hold_days]      # every 5 trading days
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


def metrics_weekly(pnl: pd.DataFrame) -> dict:
    """Annualize using 52 weeks, since rebalance is weekly."""
    if pnl.empty:
        return {"n": 0}
    n = len(pnl)
    g_sh = (pnl["spread_alpha"].mean() / pnl["spread_alpha"].std()
            * np.sqrt(52)) if pnl["spread_alpha"].std() > 0 else 0
    n_sh = (pnl["net_alpha"].mean() / pnl["net_alpha"].std()
            * np.sqrt(52)) if pnl["net_alpha"].std() > 0 else 0
    return {
        "n_rebal": n,
        "gross_bps_per_5d": pnl["spread_alpha"].mean() * 1e4,
        "net_bps_per_5d": pnl["net_alpha"].mean() * 1e4,
        "gross_bps_per_day": pnl["spread_alpha"].mean() * 1e4 / 5,
        "net_bps_per_day": pnl["net_alpha"].mean() * 1e4 / 5,
        "annual_cost_bps": pnl["cost"].mean() * 1e4 * 52,
        "turnover_pct_per_rebal": pnl["turnover"].mean() * 100,
        "univ_size": pnl["n_universe"].mean(),
        "gross_sharpe_annu": g_sh, "net_sharpe_annu": n_sh,
        "hit_rate": float((pnl["spread_alpha"] > 0).mean()),
    }


def bootstrap_ci_weekly(pnl: pd.DataFrame, block_weeks: int = 8,
                         n_boot: int = 2000) -> tuple[float, float]:
    """Block bootstrap on rebalance-level series, annualized via sqrt(52)."""
    if pnl.empty or len(pnl) < block_weeks * 2:
        return np.nan, np.nan
    arr = pnl["net_alpha"].values
    n_blocks = max(1, len(arr) // block_weeks)
    rng = np.random.default_rng(42)
    sh = []
    for _ in range(n_boot):
        starts = rng.integers(0, len(arr) - block_weeks + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + block_weeks] for s in starts])
        if sample.std() > 0:
            sh.append(sample.mean() / sample.std() * np.sqrt(52))
    if not sh:
        return np.nan, np.nan
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))


def make_folds(panel: pd.DataFrame, train_min_days: int = 365 * 3,
               test_days: int = 365, embargo_days: int = 10) -> list[tuple]:
    panel = panel.sort_values("ts")
    t0 = panel["ts"].min().normalize()
    t_max = panel["ts"].max()
    folds = []
    days = train_min_days
    while True:
        train_end = t0 + timedelta(days=days)
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_days)
        if test_start >= t_max:
            break
        if test_end > t_max:
            test_end = t_max
        folds.append((train_end, test_start, test_end))
        days += test_days
    return folds


def main() -> None:
    log.info("loading universe + earnings + cross-asset anchors...")
    panel, earnings, surv = load_universe()
    if panel.empty:
        log.error("no data")
        return
    anchors = load_anchors()

    panel = add_returns_and_basket(panel)
    panel = add_residual_5d(panel)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B(panel, earnings)
    panel, feats_C = add_features_C(panel, anchors)
    panel, feats_D = add_features_D(panel)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes

    log.info("residualization sanity: median beta=%.2f IQR=[%.2f,%.2f]",
             panel["beta"].median(),
             panel["beta"].quantile(0.25), panel["beta"].quantile(0.75))
    log.info("feature groups: A=%d  B=%d  C=%d  D=%d", len(feats_A), len(feats_B),
             len(feats_C), len(feats_D))

    feature_groups = {
        "A_price": feats_A,
        "B_pead": feats_B,
        "C_cross_asset": feats_C,
        "D_calendar": feats_D,
    }
    all_features = sum(feature_groups.values(), []) + ["sym_id"]
    label = "fwd_resid_5d"
    pnl_label = "fwd_resid_5d"

    folds = make_folds(panel, train_min_days=365 * 3, test_days=365)
    log.info("\nfolds: %d", len(folds))
    for i, f in enumerate(folds):
        log.info("  fold %d: train<=%s  test=[%s, %s]",
                 i + 1, f[0].strftime("%Y-%m-%d"),
                 f[1].strftime("%Y-%m-%d"), f[2].strftime("%Y-%m-%d"))

    ablation_configs = {
        "A only": feats_A + ["sym_id"],
        "B only": feats_B + ["sym_id"],
        "C only": feats_C + ["sym_id"],
        "D only": feats_D + ["sym_id"],
        "A+B": feats_A + feats_B + ["sym_id"],
        "A+B+C": feats_A + feats_B + feats_C + ["sym_id"],
        "ALL": all_features,
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
        log.info("  STITCHED: n_rebal=%d gross=%+.2f bps/5d (%.2f bps/d)  "
                 "net=%+.2f bps/5d (%.2f bps/d)  net_Sh=%+.2f  [%+.2f, %+.2f]  hit=%.0f%%",
                 m["n_rebal"], m["gross_bps_per_5d"], m["gross_bps_per_day"],
                 m["net_bps_per_5d"], m["net_bps_per_day"],
                 m["net_sharpe_annu"], lo, hi, 100 * m["hit_rate"])

    log.info("\n=== ABLATION SUMMARY (cost=%d bps/trade-side, top_k=%d, weekly rebal) ===",
             COST_PER_TRADE_BPS, TOP_K)
    log.info("  %-18s %5s %12s %12s %12s %18s",
             "config", "n_reb", "gross/5d", "net/5d", "net_Sh", "95% CI")
    for cfg, (m, lo, hi, _) in results.items():
        log.info("  %-18s %5d %+10.2f %+10.2f %+12.2f  [%+.2f, %+.2f]",
                 cfg, m["n_rebal"], m["gross_bps_per_5d"],
                 m["net_bps_per_5d"], m["net_sharpe_annu"], lo, hi)


if __name__ == "__main__":
    main()
