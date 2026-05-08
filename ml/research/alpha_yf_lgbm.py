"""Walk-forward LGBM probe on yfinance 5m cash-equity data.

Builds on alpha_yf_probe.py: same panel, same features, same residualization.
Adds: pooled LGBM training (5-seed ensemble), train/test split, top-K
cross-sectional portfolio with cost, OOS Sharpe + bootstrap CI.

Setup:
  - 60 days yfinance 5m, 18 names, RTH only, equal-weight in-universe basket
  - Train: first 40d. Test: last 20d. Embargo: h=48 + 12 bars between train/test
  - Label: fwd_resid_48 (4h forward sum of residual returns)
  - Features: all 27 v6_clean features + sym_id (ID feature for pooled fit)
  - Model: LGBMRegressor, 5 seeds, predictions averaged
  - Portfolio: rebalance every 48 bars (4h). At each rebalance, sort test
    bars at ts t by predicted fwd_resid_48; long top K=4, short bottom K=4.
    Realized P&L = mean(label[long]) - mean(label[short]) per rebalance.
  - Cost: 24 bps round-trip per rebalance (12 bps × 2 legs, full rotation).
  - Sharpe: from per-rebalance P&L series, annualized.
  - 95% CI via 7-day block bootstrap on daily-aggregated P&L.

Decision criterion:
  Net Sharpe lower-CI > 0 → real edge, proceed to system design.
  Net Sharpe lower-CI ≤ 0 → IS-IC was overfit, stop.
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml.research.alpha_yf_probe import (
    UNIVERSE, FEATURE_GROUPS, BARS_4H, BARS_1D,
    load_panel, add_returns, build_basket,
    add_base_features, add_cross_features,
    add_flow_features, add_xs_rank_features, add_label,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Concatenate all feature groups → ~33 unique features
ALL_FEATURES = []
for grp in FEATURE_GROUPS.values():
    for f in grp:
        if f not in ALL_FEATURES:
            ALL_FEATURES.append(f)

# Drop bk_ema_slope_4h: low coverage from earlier probe (n_ts=586 vs 4344)
ALL_FEATURES = [f for f in ALL_FEATURES if f != "bk_ema_slope_4h"]

H = BARS_4H              # 48 5m bars = 4h
HOLD = BARS_4H           # rebalance cadence = label horizon
TRAIN_DAYS = 40
TEST_DAYS = 20
EMBARGO_BARS = 60        # h + 12 bars buffer
TOP_K = 4
COST_BPS = 24            # 12 bps RT × 2 legs (long basket + short basket)
SEEDS = (42, 7, 123, 99, 314)
BOOT_N = 1000
BLOCK_DAYS = 5

LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    num_leaves=31, max_depth=6, learning_rate=0.03,
    n_estimators=200, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, min_child_samples=100, verbose=-1,
)


# ---- panel + features --------------------------------------------------

def build_panel() -> pd.DataFrame:
    log.info("loading 18-name yfinance 5m panel...")
    panel = load_panel()
    panel = add_returns(panel)
    bk = build_basket(panel)
    log.info("computing features...")
    panel = add_base_features(panel)
    panel = add_cross_features(panel, bk)
    panel = add_flow_features(panel)
    panel = add_xs_rank_features(panel)
    # Add sym_id for pooled fit
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    panel = add_label(panel, H)
    return panel


def split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    panel = panel.sort_values("ts").reset_index(drop=True)
    ts_max = panel["ts"].max()
    test_start = ts_max - timedelta(days=TEST_DAYS)
    embargo_delta = timedelta(minutes=EMBARGO_BARS * 5)
    train_end = test_start - embargo_delta

    train = panel[panel["ts"] <= train_end].copy()
    test = panel[panel["ts"] >= test_start].copy()
    log.info("split: train=%s..%s (n=%d)  test=%s..%s (n=%d)  embargo=%dmin",
             train["ts"].min().strftime("%m-%d %H:%M"),
             train["ts"].max().strftime("%m-%d %H:%M"), len(train),
             test["ts"].min().strftime("%m-%d %H:%M"),
             test["ts"].max().strftime("%m-%d %H:%M"), len(test),
             EMBARGO_BARS * 5)
    return train, test, test_start


# ---- model fit + predict -----------------------------------------------

def fit_ensemble(train: pd.DataFrame, features: list[str], label: str) -> list:
    sub = train.dropna(subset=features + [label])
    log.info("training ensemble (%d models, %d obs, %d features)...",
             len(SEEDS), len(sub), len(features))
    models = []
    for seed in SEEDS:
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(sub[features], sub[label])
        models.append(m)
    return models


def predict_oos(models: list, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    sub = test.dropna(subset=features).copy()
    preds = np.mean([m.predict(sub[features]) for m in models], axis=0)
    sub["pred"] = preds
    return sub


# ---- portfolio + PnL ---------------------------------------------------

def construct_portfolio(test_pred: pd.DataFrame, label: str,
                        top_k: int = TOP_K) -> pd.DataFrame:
    """Rebalance every HOLD bars from start of test. Each rebalance:
    rank by pred at ts, long top_k, short bottom_k. Realized P&L = label
    (forward residual) of each leg."""
    test_pred = test_pred.dropna(subset=["pred", label]).copy()
    unique_ts = sorted(test_pred["ts"].unique())
    if len(unique_ts) < HOLD * 2:
        return pd.DataFrame()

    # Pick non-overlapping rebalance ts every HOLD bars
    rebal_ts = unique_ts[::HOLD]
    log.info("rebalancing at %d non-overlapping ts (every %d bars = %dh)",
             len(rebal_ts), HOLD, HOLD * 5 // 60)

    rows = []
    for ts in rebal_ts:
        bar = test_pred[test_pred["ts"] == ts]
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values("pred")
        long_leg = bar.tail(top_k)
        short_leg = bar.head(top_k)
        rows.append({
            "ts": ts,
            "n_long": len(long_leg),
            "n_short": len(short_leg),
            "pred_spread": long_leg["pred"].mean() - short_leg["pred"].mean(),
            "spread_alpha": long_leg[label].mean() - short_leg[label].mean(),
            "long_alpha": long_leg[label].mean(),
            "short_alpha": short_leg[label].mean(),
        })
    return pd.DataFrame(rows)


def compute_metrics(pnl: pd.DataFrame, cost_bps: float = COST_BPS) -> dict:
    pnl = pnl.copy()
    pnl["cost"] = cost_bps / 1e4
    pnl["net_alpha"] = pnl["spread_alpha"] - pnl["cost"]
    n = len(pnl)
    if n < 5:
        return {"n_rebalances": n}

    # Per-rebalance Sharpe, annualized:
    # Each rebalance covers HOLD bars × 5min = HOLD/12 hours.
    # Bars per RTH year = 252 days × 78 bars/day = 19656.
    # Rebalances per year = 19656 / HOLD.
    rebals_per_year = 19656 / HOLD
    gross_sharpe = (pnl["spread_alpha"].mean() / pnl["spread_alpha"].std()
                    * np.sqrt(rebals_per_year)) if pnl["spread_alpha"].std() > 0 else 0
    net_sharpe = (pnl["net_alpha"].mean() / pnl["net_alpha"].std()
                  * np.sqrt(rebals_per_year)) if pnl["net_alpha"].std() > 0 else 0

    return {
        "n_rebalances": n,
        "gross_alpha_bps_mean": pnl["spread_alpha"].mean() * 1e4,
        "gross_alpha_bps_std": pnl["spread_alpha"].std() * 1e4,
        "cost_bps": cost_bps,
        "net_alpha_bps_mean": pnl["net_alpha"].mean() * 1e4,
        "gross_sharpe_annu": gross_sharpe,
        "net_sharpe_annu": net_sharpe,
        "hit_rate": float((pnl["spread_alpha"] > 0).mean()),
        "long_alpha_bps": pnl["long_alpha"].mean() * 1e4,
        "short_alpha_bps": pnl["short_alpha"].mean() * 1e4,
    }


def bootstrap_sharpe_ci(pnl: pd.DataFrame, cost_bps: float = COST_BPS,
                        n_boot: int = BOOT_N) -> tuple[float, float]:
    """Block bootstrap on 5-day blocks of daily-aggregated net P&L."""
    pnl = pnl.copy()
    pnl["net_alpha"] = pnl["spread_alpha"] - cost_bps / 1e4
    pnl["date"] = pnl["ts"].dt.date
    daily = pnl.groupby("date")["net_alpha"].sum()
    arr = daily.values
    if len(arr) < BLOCK_DAYS * 2:
        return np.nan, np.nan
    n_blocks = max(1, len(arr) // BLOCK_DAYS)
    rng = np.random.default_rng(42)
    sharpes = []
    for _ in range(n_boot):
        starts = rng.integers(0, len(arr) - BLOCK_DAYS + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + BLOCK_DAYS] for s in starts])
        if sample.std() > 0:
            sharpes.append(sample.mean() / sample.std() * np.sqrt(252))
    if not sharpes:
        return np.nan, np.nan
    return float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))


# ---- IC at OOS ---------------------------------------------------------

def oos_ic(test_pred: pd.DataFrame, label: str) -> tuple[float, int]:
    sub = test_pred.dropna(subset=["pred", label])
    def _ic(g):
        if len(g) < 4: return np.nan
        r, _ = spearmanr(g["pred"], g[label])
        return r
    ic_series = sub.groupby("ts").apply(_ic).dropna()
    return float(ic_series.mean()), len(ic_series)


# ---- main --------------------------------------------------------------

def main() -> None:
    panel = build_panel()
    label = f"fwd_resid_{H}"
    train, test, _ = split(panel)
    models = fit_ensemble(train, ALL_FEATURES + ["sym_id"], label)

    # Feature importance on first model (sanity)
    fi = pd.DataFrame({
        "feature": ALL_FEATURES + ["sym_id"],
        "gain": models[0].booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    log.info("top-10 feature importance (gain, model 0):")
    for _, r in fi.head(10).iterrows():
        log.info("  %-30s  %12.1f", r["feature"], r["gain"])

    test_pred = predict_oos(models, test, ALL_FEATURES + ["sym_id"])
    ic, n = oos_ic(test_pred, label)
    log.info("OOS predictive IC: %+.4f  (over %d ts)", ic, n)

    pnl = construct_portfolio(test_pred, label, top_k=TOP_K)
    if pnl.empty:
        log.warning("portfolio empty"); return

    metrics = compute_metrics(pnl, cost_bps=COST_BPS)
    log.info("--- portfolio metrics ---")
    for k, v in metrics.items():
        log.info("  %-25s  %s", k, f"{v:+.3f}" if isinstance(v, float) else v)

    sh_lo, sh_hi = bootstrap_sharpe_ci(pnl, cost_bps=COST_BPS)
    log.info("net Sharpe 95%% CI: [%+.2f, %+.2f]", sh_lo, sh_hi)

    # Cost sensitivity
    log.info("\ncost sensitivity (net Sharpe annu):")
    log.info("  %-12s %-15s %-20s", "cost (bps)", "net alpha (bps)", "net Sharpe")
    for c in (0, 6, 12, 18, 24, 36, 48):
        m = compute_metrics(pnl, cost_bps=c)
        lo, hi = bootstrap_sharpe_ci(pnl, cost_bps=c)
        log.info("  %-12s %+15.2f %+8.2f  [%+.2f, %+.2f]",
                 c, m.get("net_alpha_bps_mean", 0),
                 m.get("net_sharpe_annu", 0), lo, hi)


if __name__ == "__main__":
    main()
