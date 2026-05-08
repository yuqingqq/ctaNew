"""Long-horizon daily test of cross-sectional residual momentum.

Question: does cross-sectional return-based momentum on tech-heavy mega-caps
work *at all*, at *any* horizon, across multiple market regimes?

The 1h × 3y test was inconclusive (CI straddled zero, only 2026 positive).
Daily data goes back further, lets us see the signal across multiple
distinct regimes (2014-2015 calm, 2016-2017 low-vol, 2018 vol spike, 2019
recovery, 2020 COVID, 2021 meme/retail, 2022 bear, 2023-2024 AI rally,
2025-2026 continuation).

Setup:
  Universe: 12 mature US tech names (NVDA, TSLA, AMD, AMZN, GOOGL, META,
            AAPL, MSFT, ORCL, INTC, MU, NFLX). META starts 2012, so we
            use 2013-01-01 onward for clean full-universe panel.
  Data: yfinance daily, 2013-01-01 to 2026-05-05 (~13 years).
  Anchor: equal-weight in-universe basket.
  Beta: rolling 60-day, .shift(1).
  Lookback windows tested: 1d, 5d, 20d, 60d  — spans the literature
            (1d Heston-Sadka short-term, 5-20d intermediate, 60d momentum).
  Forward target: 1-day forward residual return.
  Strategy: at every daily close, rank by trailing return at lookback W,
            long top-K=3, short bottom-K=3. Rebalance daily, hold 1d.
  Cost: 24 bps RT per rebalance (12 bps × 2 legs, full rotation).
        Also report at 6 / 12 / 18 bps for sensitivity.
"""
from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"

UNIVERSE = [
    "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META",
    "AAPL", "MSFT", "ORCL", "INTC", "MU", "NFLX",
]
START = "2013-01-01"
END = "2026-05-06"

LOOKBACKS = (1, 5, 20, 60)
HOLD_DAYS = 1
BETA_WINDOW = 60        # trailing 60-day rolling beta
TOP_K = 3
COST_BPS = 24           # 12 bps × 2 legs


# ---- data --------------------------------------------------------------

def fetch(symbol: str) -> pd.DataFrame:
    cache = CACHE / f"yf_{symbol}_1d_long.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = yf.Ticker(symbol).history(start=START, end=END, interval="1d",
                                   auto_adjust=True)
    if df.empty:
        return df
    df.index = df.index.tz_convert("UTC")
    df = df.reset_index().rename(columns={
        "Date": "ts", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.normalize()
    df["symbol"] = symbol
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df.to_parquet(cache)
    return df


def load_panel() -> pd.DataFrame:
    frames = []
    for s in UNIVERSE:
        df = fetch(s)
        if df.empty:
            continue
        log.info("  %-6s n=%5d  %s -> %s", s, len(df),
                 df["ts"].iloc[0].strftime("%Y-%m-%d"),
                 df["ts"].iloc[-1].strftime("%Y-%m-%d"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- features + label --------------------------------------------------

def add_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    panel["ret"] = (panel.groupby("symbol")["close"]
                    .transform(lambda s: np.log(s / s.shift(1))))
    return panel


def build_basket(panel: pd.DataFrame) -> pd.DataFrame:
    bk_ret = panel.groupby("ts")["ret"].mean().rename("bk_ret")
    return bk_ret.reset_index()


def residualize(panel: pd.DataFrame, bk: pd.DataFrame) -> pd.DataFrame:
    panel = panel.merge(bk, on="ts", how="left").sort_values(["symbol", "ts"]).reset_index(drop=True)

    def _beta(g):
        ret, bret = g["ret"], g["bk_ret"]
        cov = (ret * bret).rolling(BETA_WINDOW).mean() - \
              ret.rolling(BETA_WINDOW).mean() * bret.rolling(BETA_WINDOW).mean()
        var = bret.rolling(BETA_WINDOW).var().replace(0, np.nan)
        return (cov / var).clip(-5, 5).shift(1)

    panel["beta"] = (panel.groupby("symbol", group_keys=False)
                     .apply(_beta).values)
    panel["resid"] = panel["ret"] - panel["beta"] * panel["bk_ret"]
    return panel


def add_signals(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("symbol", group_keys=False)
    for w in LOOKBACKS:
        panel[f"ret_{w}d"] = g["close"].apply(
            lambda s: s.pct_change(w)).shift(1)
    panel["fwd_resid_1d"] = g["resid"].shift(-1)
    return panel


# ---- portfolio ---------------------------------------------------------

def construct_portfolio(panel: pd.DataFrame, signal: str,
                        top_k: int = TOP_K) -> pd.DataFrame:
    panel = panel.dropna(subset=[signal, "fwd_resid_1d", "beta"]).copy()
    rows = []
    for ts, bar in panel.groupby("ts"):
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values(signal)
        long_leg = bar.tail(top_k)
        short_leg = bar.head(top_k)
        rows.append({
            "ts": ts,
            "spread_alpha": long_leg["fwd_resid_1d"].mean()
                            - short_leg["fwd_resid_1d"].mean(),
            "long_alpha": long_leg["fwd_resid_1d"].mean(),
            "short_alpha": short_leg["fwd_resid_1d"].mean(),
        })
    return pd.DataFrame(rows)


def metrics(pnl: pd.DataFrame, cost_bps: float = COST_BPS) -> dict:
    if pnl.empty:
        return {"n": 0}
    pnl = pnl.copy()
    pnl["net"] = pnl["spread_alpha"] - cost_bps / 1e4
    g_sh = (pnl["spread_alpha"].mean() / pnl["spread_alpha"].std()
            * np.sqrt(252)) if pnl["spread_alpha"].std() > 0 else 0
    n_sh = (pnl["net"].mean() / pnl["net"].std()
            * np.sqrt(252)) if pnl["net"].std() > 0 else 0
    return {
        "n": len(pnl),
        "gross_bps": pnl["spread_alpha"].mean() * 1e4,
        "net_bps": pnl["net"].mean() * 1e4,
        "gross_sharpe": g_sh,
        "net_sharpe": n_sh,
        "hit_rate": float((pnl["spread_alpha"] > 0).mean()),
    }


def bootstrap_sharpe_ci(pnl: pd.DataFrame, cost_bps: float = COST_BPS,
                        block_days: int = 60, n_boot: int = 2000) -> tuple[float, float]:
    if pnl.empty or len(pnl) < block_days * 2:
        return np.nan, np.nan
    pnl = pnl.copy()
    pnl["net"] = pnl["spread_alpha"] - cost_bps / 1e4
    arr = pnl["net"].values
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
    log.info("loading 12-name yfinance daily panel %s -> %s", START, END)
    panel = load_panel()
    panel = add_returns(panel)
    bk = build_basket(panel)
    panel = residualize(panel, bk)
    panel = add_signals(panel)

    log.info("residualization sanity: median beta=%.2f IQR=[%.2f,%.2f]",
             panel["beta"].median(),
             panel["beta"].quantile(0.25), panel["beta"].quantile(0.75))

    # For each lookback, compute portfolio and metrics
    log.info("\n=== HEADLINE: gross / net Sharpe by lookback (cost=%d bps) ===", COST_BPS)
    log.info("  %-10s %6s %12s %12s %12s %12s %18s",
             "lookback", "n", "gross/d", "net/d", "gross_Sh", "net_Sh", "net_Sh 95%CI")
    by_lookback = {}
    for w in LOOKBACKS:
        pnl = construct_portfolio(panel, f"ret_{w}d")
        m = metrics(pnl)
        lo, hi = bootstrap_sharpe_ci(pnl)
        by_lookback[w] = (pnl, m, lo, hi)
        log.info("  %-10s %6d %+10.1fbps %+10.1fbps %+12.2f %+12.2f  [%+.2f, %+.2f]",
                 f"{w}d", m["n"], m["gross_bps"], m["net_bps"],
                 m["gross_sharpe"], m["net_sharpe"], lo, hi)

    # Per-year breakdown for the most-promising lookback
    best_w = max(by_lookback, key=lambda w: by_lookback[w][1]["gross_sharpe"])
    log.info("\n=== PER-YEAR for lookback=%dd (best gross Sharpe) ===", best_w)
    pnl = by_lookback[best_w][0].copy()
    pnl["year"] = pnl["ts"].dt.year
    log.info("  %-6s %5s %12s %12s %12s %8s",
             "year", "n", "gross/d", "net/d", "net_Sh", "hit")
    for y, g in pnl.groupby("year"):
        m = metrics(g)
        log.info("  %-6d %5d %+10.1fbps %+10.1fbps %+12.2f %7.0f%%",
                 y, m["n"], m["gross_bps"], m["net_bps"],
                 m["net_sharpe"], 100 * m["hit_rate"])

    # Cost sensitivity for best lookback
    log.info("\n=== COST SENSITIVITY for lookback=%dd ===", best_w)
    pnl_best = by_lookback[best_w][0]
    log.info("  %-12s %-15s %-12s %-18s",
             "cost (bps)", "net /d (bps)", "net Sharpe", "95% CI")
    for c in (0, 6, 12, 18, 24, 36):
        m = metrics(pnl_best, cost_bps=c)
        lo, hi = bootstrap_sharpe_ci(pnl_best, cost_bps=c)
        log.info("  %-12d %+13.1f %+12.2f  [%+.2f, %+.2f]",
                 c, m["net_bps"], m["net_sharpe"], lo, hi)


if __name__ == "__main__":
    main()
