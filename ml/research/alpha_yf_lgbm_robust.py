"""Robustness checks on the yf-cash 5m alpha-residual strategy.

Three checks:

  1. 3-fold expanding-window walk-forward (train [0,30d]→test [30,40], etc.)
     Stitches OOS predictions across folds → ~30 rebalances, more reliable
     Sharpe estimate than the single-fold probe.
  2. Simple-rule baseline: long top-K by return_1d, short bottom-K by
     return_1d. No model. Tests whether LGBM adds value over the strongest
     single-feature predictor.
  3. Earnings mask: drop (sym, ts) where ts is within ±1 trading day of
     that sym's earnings. Tests whether the alpha is concentrated in
     earnings windows (suggesting the model is "predicting" earnings
     reactions, which would not be a stable signal).

Output: 4 strategies (LGBM±mask × simple±mask) × stitched-OOS metrics.
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

from ml.research.alpha_yf_probe import (
    UNIVERSE, FEATURE_GROUPS, BARS_4H,
    load_panel, add_returns, build_basket,
    add_base_features, add_cross_features,
    add_flow_features, add_xs_rank_features, add_label,
)
from ml.research.alpha_yf_lgbm import (
    ALL_FEATURES, LGB_PARAMS, SEEDS, COST_BPS, TOP_K, BLOCK_DAYS,
    BOOT_N, fit_ensemble, predict_oos,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"

H = BARS_4H  # 48 5m bars = 4h
HOLD = BARS_4H
EARNINGS_MASK_BARS = 78  # ±1 trading day


# ---- earnings calendar -------------------------------------------------

def fetch_earnings(symbol: str) -> pd.DatetimeIndex:
    """Cached fetch of earnings dates as UTC DatetimeIndex."""
    cache = CACHE / f"earn_{symbol}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["ts"].pipe(pd.DatetimeIndex)
    df = yf.Ticker(symbol).get_earnings_dates(limit=12)
    if df is None or df.empty:
        idx = pd.DatetimeIndex([])
    else:
        idx = df.index.tz_convert("UTC")
    pd.DataFrame({"ts": idx}).to_parquet(cache)
    return idx


def build_earnings_mask(panel: pd.DataFrame, mask_bars: int = EARNINGS_MASK_BARS) -> pd.Series:
    """Return boolean Series aligned with panel rows: True = should be masked
    (within ±mask_bars of an earnings date for that sym)."""
    mask = pd.Series(False, index=panel.index)
    for sym in UNIVERSE:
        idx = fetch_earnings(sym)
        if len(idx) == 0:
            continue
        rows = panel.index[panel["symbol"] == sym]
        if len(rows) == 0:
            continue
        ts = panel.loc[rows, "ts"]
        for e in idx:
            # ±1 trading day window around e
            lo = e - timedelta(days=2)
            hi = e + timedelta(days=2)
            in_win = ts.between(lo, hi)
            mask.loc[rows[in_win]] = True
    return mask


# ---- portfolio + cost --------------------------------------------------

def construct_portfolio(test_pred: pd.DataFrame, label: str,
                        top_k: int = TOP_K) -> pd.DataFrame:
    test_pred = test_pred.dropna(subset=["pred", label]).copy()
    unique_ts = sorted(test_pred["ts"].unique())
    if not unique_ts:
        return pd.DataFrame()
    rebal_ts = unique_ts[::HOLD]
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
            "spread_alpha": long_leg[label].mean() - short_leg[label].mean(),
            "long_alpha": long_leg[label].mean(),
            "short_alpha": short_leg[label].mean(),
            "n_long": len(long_leg),
            "n_short": len(short_leg),
        })
    return pd.DataFrame(rows)


def metrics(pnl: pd.DataFrame, cost_bps: float = COST_BPS) -> dict:
    if pnl.empty:
        return {"n": 0}
    pnl = pnl.copy()
    pnl["net_alpha"] = pnl["spread_alpha"] - cost_bps / 1e4
    rebals_per_year = 19656 / HOLD
    g_sh = (pnl["spread_alpha"].mean() / pnl["spread_alpha"].std()
            * np.sqrt(rebals_per_year)) if pnl["spread_alpha"].std() > 0 else 0
    n_sh = (pnl["net_alpha"].mean() / pnl["net_alpha"].std()
            * np.sqrt(rebals_per_year)) if pnl["net_alpha"].std() > 0 else 0
    return {
        "n": len(pnl),
        "gross_bps_per_4h": pnl["spread_alpha"].mean() * 1e4,
        "net_bps_per_4h": pnl["net_alpha"].mean() * 1e4,
        "gross_sharpe": g_sh,
        "net_sharpe": n_sh,
        "hit_rate": float((pnl["spread_alpha"] > 0).mean()),
        "long_bps": pnl["long_alpha"].mean() * 1e4,
        "short_bps": pnl["short_alpha"].mean() * 1e4,
    }


def bootstrap_sharpe_ci(pnl: pd.DataFrame, cost_bps: float = COST_BPS,
                        n_boot: int = BOOT_N) -> tuple[float, float]:
    if pnl.empty:
        return np.nan, np.nan
    pnl = pnl.copy()
    pnl["net_alpha"] = pnl["spread_alpha"] - cost_bps / 1e4
    pnl["date"] = pnl["ts"].dt.date
    daily = pnl.groupby("date")["net_alpha"].sum()
    arr = daily.values
    if len(arr) < BLOCK_DAYS * 2:
        return np.nan, np.nan
    n_blocks = max(1, len(arr) // BLOCK_DAYS)
    rng = np.random.default_rng(42)
    sh = []
    for _ in range(n_boot):
        starts = rng.integers(0, len(arr) - BLOCK_DAYS + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + BLOCK_DAYS] for s in starts])
        if sample.std() > 0:
            sh.append(sample.mean() / sample.std() * np.sqrt(252))
    return (float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))) if sh else (np.nan, np.nan)


# ---- folds -------------------------------------------------------------

def make_folds(panel: pd.DataFrame, n_folds: int = 3,
               train_min_days: int = 30, test_days: int = 10,
               embargo_min: int = 60 * 5) -> list[tuple]:
    """Expanding-window folds. Returns list of (train_end, test_start, test_end)."""
    panel = panel.sort_values("ts")
    t0 = panel["ts"].min().normalize()
    t_max = panel["ts"].max()
    folds = []
    for i in range(n_folds):
        train_end = t0 + timedelta(days=train_min_days + i * test_days)
        test_start = train_end + timedelta(minutes=embargo_min)
        test_end = test_start + timedelta(days=test_days)
        if test_end > t_max + timedelta(days=1):
            test_end = t_max
        folds.append((train_end, test_start, test_end))
    return folds


def run_one_fold(panel: pd.DataFrame, fold: tuple, label: str,
                 features: list[str], strategy: str, mask: pd.Series | None
                 ) -> pd.DataFrame:
    """strategy in {'lgbm', 'simple_return_1d'}.  Returns per-rebalance pnl df."""
    train_end, test_start, test_end = fold
    panel_use = panel.copy()
    if mask is not None:
        panel_use = panel_use[~mask].copy()
    train = panel_use[panel_use["ts"] <= train_end]
    test = panel_use[(panel_use["ts"] >= test_start) & (panel_use["ts"] <= test_end)]
    if test.empty:
        return pd.DataFrame()

    if strategy == "lgbm":
        models = fit_ensemble(train, features, label)
        test_pred = predict_oos(models, test, features)
    elif strategy == "simple_return_1d":
        # baseline: just rank by return_1d (the strongest single-feature IC)
        test_pred = test.dropna(subset=["return_1d", label]).copy()
        test_pred["pred"] = test_pred["return_1d"]
    else:
        raise ValueError(strategy)

    pnl = construct_portfolio(test_pred, label, top_k=TOP_K)
    pnl["fold_train_end"] = train_end
    pnl["fold_test_start"] = test_start
    pnl["strategy"] = strategy
    pnl["mask_applied"] = mask is not None
    return pnl


# ---- main --------------------------------------------------------------

def main() -> None:
    log.info("loading + featurizing yfinance 5m panel...")
    panel = load_panel()
    panel = add_returns(panel)
    bk = build_basket(panel)
    panel = add_base_features(panel)
    panel = add_cross_features(panel, bk)
    panel = add_flow_features(panel)
    panel = add_xs_rank_features(panel)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    panel = add_label(panel, H)
    label = f"fwd_resid_{H}"
    feats = [f for f in ALL_FEATURES + ["sym_id"] if f in panel.columns]

    log.info("fetching earnings dates for %d names...", len(UNIVERSE))
    mask = build_earnings_mask(panel)
    log.info("earnings mask: %d / %d (sym, ts) pairs masked (%.1f%%)",
             mask.sum(), len(mask), 100 * mask.mean())

    folds = make_folds(panel, n_folds=3, train_min_days=30, test_days=10)
    for i, f in enumerate(folds):
        log.info("fold %d: train<=%s  test=[%s, %s]",
                 i + 1, f[0].strftime("%m-%d %H:%M"),
                 f[1].strftime("%m-%d %H:%M"),
                 f[2].strftime("%m-%d %H:%M"))

    runs = []
    for strategy in ("lgbm", "simple_return_1d"):
        for use_mask in (False, True):
            log.info("\n>>> strategy=%s mask=%s", strategy, use_mask)
            fold_pnls = []
            for fold in folds:
                pnl = run_one_fold(
                    panel, fold, label, feats, strategy,
                    mask if use_mask else None,
                )
                if not pnl.empty:
                    fold_pnls.append(pnl)
                    m = metrics(pnl)
                    log.info("  fold (test_start %s): n=%d gross=%+.1fbps/4h "
                             "net=%+.1fbps/4h net_sharpe=%+.2f hit=%.0f%%",
                             fold[1].strftime("%m-%d"),
                             m["n"], m["gross_bps_per_4h"],
                             m["net_bps_per_4h"], m["net_sharpe"],
                             100 * m["hit_rate"])
            stitched = pd.concat(fold_pnls, ignore_index=True) if fold_pnls else pd.DataFrame()
            m = metrics(stitched)
            lo, hi = bootstrap_sharpe_ci(stitched)
            runs.append({
                "strategy": strategy,
                "mask": use_mask,
                "n": m.get("n", 0),
                "gross_bps_per_4h": m.get("gross_bps_per_4h", 0),
                "net_bps_per_4h": m.get("net_bps_per_4h", 0),
                "net_sharpe": m.get("net_sharpe", 0),
                "net_sharpe_lo": lo,
                "net_sharpe_hi": hi,
                "hit_rate": m.get("hit_rate", 0),
                "long_bps": m.get("long_bps", 0),
                "short_bps": m.get("short_bps", 0),
            })
            log.info("  STITCHED: n=%d gross=%+.1f net=%+.1f net_sharpe=%+.2f "
                     "[%+.2f, %+.2f] hit=%.0f%%",
                     m.get("n", 0), m.get("gross_bps_per_4h", 0),
                     m.get("net_bps_per_4h", 0), m.get("net_sharpe", 0),
                     lo, hi, 100 * m.get("hit_rate", 0))

    log.info("\n=== summary table (cost=%d bps RT) ===", COST_BPS)
    log.info("  %-22s %5s %12s %12s %12s %18s",
             "strategy", "n", "gross/4h", "net/4h", "net_sharpe", "95% CI")
    for r in runs:
        tag = f"{r['strategy']}{'+earn_mask' if r['mask'] else ''}"
        log.info("  %-22s %5d %+10.1fbps %+10.1fbps %+12.2f  [%+.2f, %+.2f]",
                 tag, r["n"], r["gross_bps_per_4h"], r["net_bps_per_4h"],
                 r["net_sharpe"], r["net_sharpe_lo"], r["net_sharpe_hi"])


if __name__ == "__main__":
    main()
