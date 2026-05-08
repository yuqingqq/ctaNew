"""Native-cadence LGBM probe on Polygon.io 5m × 2y US equity data.

This is the strict version of the test. Same v6_clean feature stack the
crypto pipeline uses, on real 5m bars at native cadence, with ~6mo of
training data per fold (matches v6_clean's typical fold size on crypto).

Setup:
  Universe: 12 mature US tech names (NVDA, TSLA, AMD, AMZN, GOOGL, META,
            AAPL, MSFT, ORCL, INTC, MU, NFLX). All have full 2y history
            from Polygon free tier (~700-day window).
  Data: Polygon.io 5m, RTH only (filtered by US/Eastern hours).
  Anchor: equal-weight in-universe basket.
  Features: same v6_clean port as alpha_yf_probe.py (windows BARS_1H=12,
            BARS_1D=288, BARS_7D=2016 — native crypto-equivalent).
  Label: fwd_resid_48 (4h forward residual sum at 5m).
  Hold: 48 bars (4h). Rebalance every 48 bars.
  Walk-forward: expanding window, 5-6 folds, ~6mo train min / 3mo test.

Compare:
  Strategy A: LGBM ensemble (5 seeds), 37 features.
  Strategy B: Simple long-short top-K return_1d (no model).
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_collectors.polygon_loader import fetch_aggs
from ml.research.alpha_yf_probe import (
    FEATURE_GROUPS, BARS_4H,
    add_returns, build_basket,
    add_base_features, add_cross_features,
    add_flow_features, add_xs_rank_features, add_label,
)
from ml.research.alpha_yf_lgbm import (
    LGB_PARAMS, SEEDS, COST_BPS, TOP_K,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

UNIVERSE_POLY = [
    "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META",
    "AAPL", "MSFT", "ORCL", "INTC", "MU", "NFLX",
]

H = BARS_4H  # 48 5m bars = 4h
HOLD = BARS_4H

ALL_FEATURES = []
for grp in FEATURE_GROUPS.values():
    for f in grp:
        if f not in ALL_FEATURES:
            ALL_FEATURES.append(f)
ALL_FEATURES = [f for f in ALL_FEATURES if f != "bk_ema_slope_4h"]


# ---- data --------------------------------------------------------------

def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only US RTH bars (9:30-16:00 ET, weekdays). Polygon includes
    pre/post-market by default; we drop them to match yfinance's RTH-only
    panel and the v6_clean cadence."""
    if df.empty:
        return df
    et = df["ts"].dt.tz_convert("America/New_York")
    in_rth = (
        (et.dt.dayofweek < 5)
        & ((et.dt.hour > 9) | ((et.dt.hour == 9) & (et.dt.minute >= 30)))
        & (et.dt.hour < 16)
    )
    return df[in_rth].copy()


def load_panel() -> pd.DataFrame:
    frames = []
    for sym in UNIVERSE_POLY:
        df = fetch_aggs(sym, "5m")
        if df.empty:
            log.warning("  %s: empty", sym)
            continue
        df = filter_rth(df)
        df["symbol"] = sym
        # Match yfinance schema (n_trades, vwap optional)
        keep = ["ts", "symbol", "open", "high", "low", "close", "volume"]
        if "n_trades" in df.columns:
            keep.append("n_trades")
        df = df[keep]
        log.info("  %-6s n=%6d  %s -> %s", sym, len(df),
                 df["ts"].iloc[0].strftime("%Y-%m-%d"),
                 df["ts"].iloc[-1].strftime("%Y-%m-%d"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- folds + training --------------------------------------------------

def make_folds(panel: pd.DataFrame, train_min_days: int = 180,
               test_days: int = 90, embargo_bars: int = 60) -> list[tuple]:
    """Expanding-window folds. Each fold extends train by test_days; test
    is the next test_days. Embargo embargo_bars (5min × 60 = 5h) between."""
    panel = panel.sort_values("ts")
    t0 = panel["ts"].min().normalize()
    t_max = panel["ts"].max()
    folds = []
    days = train_min_days
    while True:
        train_end = t0 + timedelta(days=days)
        test_start = train_end + timedelta(minutes=embargo_bars * 5)
        test_end = test_start + timedelta(days=test_days)
        if test_start >= t_max:
            break
        if test_end > t_max:
            test_end = t_max
        folds.append((train_end, test_start, test_end))
        days += test_days
    return folds


def fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                features: list[str], label: str) -> pd.DataFrame:
    train_ = train.dropna(subset=features + [label])
    if len(train_) < 1000:
        return pd.DataFrame()
    log.info("    train n=%d, test n=%d, features=%d",
             len(train_), len(test), len(features))
    preds = []
    for seed in SEEDS:
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(train_[features], train_[label])
        sub = test.dropna(subset=features).copy()
        preds.append(m.predict(sub[features]))
    sub = test.dropna(subset=features).copy()
    sub["pred"] = np.mean(preds, axis=0)
    return sub


# ---- portfolio ---------------------------------------------------------

def portfolio(test_pred: pd.DataFrame, signal: str, label: str,
              top_k: int = TOP_K) -> pd.DataFrame:
    sub = test_pred.dropna(subset=[signal, label]).copy()
    unique_ts = sorted(sub["ts"].unique())
    if not unique_ts:
        return pd.DataFrame()
    rebal_ts = unique_ts[::HOLD]
    rows = []
    for ts in rebal_ts:
        bar = sub[sub["ts"] == ts]
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values(signal)
        long_leg = bar.tail(top_k)
        short_leg = bar.head(top_k)
        rows.append({
            "ts": ts,
            "spread_alpha": long_leg[label].mean() - short_leg[label].mean(),
            "long_alpha": long_leg[label].mean(),
            "short_alpha": short_leg[label].mean(),
        })
    return pd.DataFrame(rows)


def metrics(pnl: pd.DataFrame, cost_bps: float = COST_BPS) -> dict:
    if pnl.empty:
        return {"n": 0}
    pnl = pnl.copy()
    pnl["net"] = pnl["spread_alpha"] - cost_bps / 1e4
    rebals_per_year = (252 * 78) / HOLD
    g_sh = (pnl["spread_alpha"].mean() / pnl["spread_alpha"].std()
            * np.sqrt(rebals_per_year)) if pnl["spread_alpha"].std() > 0 else 0
    n_sh = (pnl["net"].mean() / pnl["net"].std()
            * np.sqrt(rebals_per_year)) if pnl["net"].std() > 0 else 0
    return {
        "n": len(pnl),
        "gross_bps": pnl["spread_alpha"].mean() * 1e4,
        "net_bps": pnl["net"].mean() * 1e4,
        "gross_sharpe": g_sh,
        "net_sharpe": n_sh,
        "hit_rate": float((pnl["spread_alpha"] > 0).mean()),
    }


def bootstrap_ci(pnl: pd.DataFrame, cost_bps: float = COST_BPS,
                 block_days: int = 30, n_boot: int = 2000) -> tuple[float, float]:
    if pnl.empty:
        return np.nan, np.nan
    pnl = pnl.copy()
    pnl["net"] = pnl["spread_alpha"] - cost_bps / 1e4
    pnl["date"] = pnl["ts"].dt.date
    daily = pnl.groupby("date")["net"].sum()
    arr = daily.values
    if len(arr) < block_days * 2:
        return np.nan, np.nan
    n_blocks = max(1, len(arr) // block_days)
    rng = np.random.default_rng(42)
    sh = []
    for _ in range(n_boot):
        starts = rng.integers(0, len(arr) - block_days + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + block_days] for s in starts])
        if sample.std() > 0:
            sh.append(sample.mean() / sample.std() * np.sqrt(252))
    if not sh:
        return np.nan, np.nan
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))


# ---- main --------------------------------------------------------------

def main() -> None:
    log.info("loading polygon 5m panel for %d names...", len(UNIVERSE_POLY))
    panel = load_panel()
    if panel.empty:
        log.error("no data — run 'python3 -m data_collectors.polygon_loader' first")
        return

    log.info("computing v6_clean features (native 5m cadence)...")
    panel = add_returns(panel)
    bk = build_basket(panel)
    panel = add_base_features(panel)
    panel = add_cross_features(panel, bk)
    panel = add_flow_features(panel)
    panel = add_xs_rank_features(panel)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    panel = add_label(panel, H)
    label = f"fwd_resid_{H}"

    log.info("residualization sanity: median beta=%.2f IQR=[%.2f,%.2f]",
             panel["beta_short_vs_bk"].median(),
             panel["beta_short_vs_bk"].quantile(0.25),
             panel["beta_short_vs_bk"].quantile(0.75))

    feats = [f for f in ALL_FEATURES + ["sym_id"] if f in panel.columns]
    folds = make_folds(panel, train_min_days=180, test_days=90)
    log.info("\nfolds:")
    for i, (te, ts_, te2) in enumerate(folds):
        log.info("  fold %d: train<=%s  test=[%s, %s]",
                 i + 1, te.strftime("%Y-%m-%d"),
                 ts_.strftime("%Y-%m-%d"), te2.strftime("%Y-%m-%d"))

    lgbm_pnls = []
    simple_pnls = []
    for i, (train_end, test_start, test_end) in enumerate(folds):
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        log.info("\n>>> Fold %d (test %s -> %s)", i + 1,
                 test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"))

        log.info("  [LGBM]")
        test_pred = fit_predict(train, test, feats, label)
        if not test_pred.empty:
            lp = portfolio(test_pred, "pred", label)
            if not lp.empty:
                m = metrics(lp)
                log.info("    n=%d gross=%+.1fbps net=%+.1fbps net_Sh=%+.2f hit=%.0f%%",
                         m["n"], m["gross_bps"], m["net_bps"],
                         m["net_sharpe"], 100 * m["hit_rate"])
                lp["fold"] = i + 1
                lgbm_pnls.append(lp)

        log.info("  [SIMPLE return_1d]")
        sp = portfolio(test, "return_1d", label)
        if not sp.empty:
            m = metrics(sp)
            log.info("    n=%d gross=%+.1fbps net=%+.1fbps net_Sh=%+.2f hit=%.0f%%",
                     m["n"], m["gross_bps"], m["net_bps"],
                     m["net_sharpe"], 100 * m["hit_rate"])
            sp["fold"] = i + 1
            simple_pnls.append(sp)

    log.info("\n=== STITCHED OOS METRICS (cost=%d bps) ===", COST_BPS)
    log.info("  %-22s %6s %12s %12s %12s %18s",
             "strategy", "n", "gross/4h", "net/4h", "net_Sh", "95% CI")
    for label_, all_pnls in [("LGBM (v6_clean × 5m)", lgbm_pnls),
                              ("simple return_1d", simple_pnls)]:
        if not all_pnls:
            continue
        st = pd.concat(all_pnls, ignore_index=True)
        m = metrics(st)
        lo, hi = bootstrap_ci(st)
        log.info("  %-22s %6d %+10.1fbps %+10.1fbps %+12.2f  [%+.2f, %+.2f]",
                 label_, m["n"], m["gross_bps"], m["net_bps"],
                 m["net_sharpe"], lo, hi)

    if lgbm_pnls:
        st = pd.concat(lgbm_pnls, ignore_index=True)
        log.info("\n=== LGBM cost sensitivity ===")
        log.info("  %-12s %-15s %-12s %-18s",
                 "cost (bps)", "net /4h", "net_Sharpe", "95% CI")
        for c in (0, 6, 12, 18, 24, 36):
            m = metrics(st, cost_bps=c)
            lo, hi = bootstrap_ci(st, cost_bps=c)
            log.info("  %-12d %+13.1f %+12.2f  [%+.2f, %+.2f]",
                     c, m["net_bps"], m["net_sharpe"], lo, hi)


if __name__ == "__main__":
    main()
