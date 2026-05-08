"""Long-history validation: simple top-K cross-sectional momentum rule on
~3 years of yfinance 1h cash-equity data.

Question: does the +50 bps/4h net edge from the 60-day 5m probe survive
across multiple market regimes?

Setup:
  Universe: 12 mature US tech names with 3y+ history (NVDA, TSLA, AMD,
            AMZN, GOOGL, META, AAPL, MSFT, ORCL, INTC, MU, NFLX).
            Drops newer listings (CRCL, HOOD, COIN, MSTR, PLTR, SNDK).
  Data: yfinance 1h, period=730d (yields ~1060d in practice), RTH only,
        auto_adjust=True.
  Anchor: equal-weight in-universe basket.
  Signal: trailing 1-day return (7 1h-bars).
  Forward target: 4-bar residual return = 4h forward (matches 5m × h=48
                  cadence calendar-time).
  Strategy: at every 4-bar rebalance, sort by trailing return_1d, long
            top-K=3, short bottom-K=3 (out of 12). Hold 4 bars.
  Cost: 24 bps RT per rebalance (12 bps × 2 legs).
  Metrics: gross/net edge per 4h, Sharpe annualized, per-year breakdown,
           bootstrap CI.
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

INTERVAL = "1h"
PERIOD = "730d"
RETURN_1D_BARS = 7         # ~1 RTH day at 1h cadence
HOLD_BARS = 4              # 4h hold = same calendar as 5m × h=48
BETA_WINDOW = 168          # ~25 RTH days
TOP_K = 3
COST_BPS = 24
BARS_PER_RTH_YEAR = 252 * 7  # 1764


# ---- data --------------------------------------------------------------

def fetch_yf_1h(symbol: str) -> pd.DataFrame:
    cache = CACHE / f"yf_{symbol}_1h_long.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = yf.Ticker(symbol).history(period=PERIOD, interval=INTERVAL,
                                   auto_adjust=True, prepost=False)
    if df.empty:
        return df
    df.index = df.index.tz_convert("UTC")
    df = df.reset_index().rename(columns={
        "Datetime": "ts", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["symbol"] = symbol
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df.to_parquet(cache)
    return df


def load_panel() -> pd.DataFrame:
    frames = []
    for sym in UNIVERSE:
        df = fetch_yf_1h(sym)
        if df.empty:
            continue
        log.info("  %-6s n=%5d  %s -> %s",
                 sym, len(df),
                 df["ts"].iloc[0].strftime("%Y-%m-%d"),
                 df["ts"].iloc[-1].strftime("%Y-%m-%d"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- features + labels -------------------------------------------------

def add_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    panel["ret"] = (panel.groupby("symbol")["close"]
                    .transform(lambda s: np.log(s / s.shift(1))))
    return panel


def build_basket(panel: pd.DataFrame) -> pd.DataFrame:
    bk_ret = panel.groupby("ts")["ret"].mean().rename("bk_ret")
    bk_close = (1.0 + bk_ret.fillna(0.0)).cumprod().rename("bk_close")
    return pd.concat([bk_ret, bk_close], axis=1).reset_index()


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


def add_signal_and_label(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("symbol", group_keys=False)
    panel["return_1d"] = g["close"].apply(
        lambda s: s.pct_change(RETURN_1D_BARS)).shift(1)
    panel["fwd_resid_4h"] = g["resid"].apply(
        lambda s: s.rolling(HOLD_BARS).sum().shift(-HOLD_BARS)).values
    return panel


# ---- portfolio ---------------------------------------------------------

def construct_portfolio(panel: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    panel = panel.dropna(subset=["return_1d", "fwd_resid_4h"]).copy()
    unique_ts = sorted(panel["ts"].unique())
    rebal_ts = unique_ts[::HOLD_BARS]
    log.info("rebalances: %d (every %d bars), unique_ts=%d",
             len(rebal_ts), HOLD_BARS, len(unique_ts))

    rows = []
    for ts in rebal_ts:
        bar = panel[panel["ts"] == ts]
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values("return_1d")
        long_leg = bar.tail(top_k)
        short_leg = bar.head(top_k)
        rows.append({
            "ts": ts,
            "spread_alpha": long_leg["fwd_resid_4h"].mean()
                            - short_leg["fwd_resid_4h"].mean(),
            "long_alpha": long_leg["fwd_resid_4h"].mean(),
            "short_alpha": short_leg["fwd_resid_4h"].mean(),
        })
    return pd.DataFrame(rows)


def metrics(pnl: pd.DataFrame, cost_bps: float = COST_BPS) -> dict:
    if pnl.empty:
        return {"n": 0}
    pnl = pnl.copy()
    pnl["net"] = pnl["spread_alpha"] - cost_bps / 1e4
    rebals_per_year = BARS_PER_RTH_YEAR / HOLD_BARS
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
        "long_bps": pnl["long_alpha"].mean() * 1e4,
        "short_bps": pnl["short_alpha"].mean() * 1e4,
    }


def bootstrap_sharpe_ci(pnl: pd.DataFrame, cost_bps: float = COST_BPS,
                        block_days: int = 20, n_boot: int = 1000) -> tuple[float, float]:
    if len(pnl) < block_days * 2:
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
    log.info("loading 12-name yfinance 1h panel (~3y history)...")
    panel = load_panel()
    panel = add_returns(panel)
    bk = build_basket(panel)
    panel = residualize(panel, bk)
    panel = add_signal_and_label(panel)

    log.info("residualization sanity: median beta=%.2f IQR=[%.2f,%.2f]",
             panel["beta"].median(),
             panel["beta"].quantile(0.25), panel["beta"].quantile(0.75))

    pnl = construct_portfolio(panel)
    if pnl.empty:
        log.error("no rebalances")
        return

    pnl["year"] = pnl["ts"].dt.year

    # overall
    log.info("\n=== ALL DATA (cost=%d bps) ===", COST_BPS)
    m = metrics(pnl)
    lo, hi = bootstrap_sharpe_ci(pnl)
    log.info("  n_rebalances    = %d", m["n"])
    log.info("  gross / 4h      = %+.2f bps", m["gross_bps"])
    log.info("  net / 4h        = %+.2f bps", m["net_bps"])
    log.info("  gross sharpe    = %+.2f", m["gross_sharpe"])
    log.info("  net sharpe      = %+.2f  [%+.2f, %+.2f] 95%% CI", m["net_sharpe"], lo, hi)
    log.info("  hit rate        = %.0f%%", 100 * m["hit_rate"])
    log.info("  long  / short   = %+.1f / %+.1f bps", m["long_bps"], m["short_bps"])

    # per year
    log.info("\n=== PER-YEAR (cost=%d bps) ===", COST_BPS)
    log.info("  %-6s %5s %12s %12s %12s %8s",
             "year", "n", "gross/4h", "net/4h", "net_sharpe", "hit")
    for y, g in pnl.groupby("year"):
        m = metrics(g)
        log.info("  %-6d %5d %+10.1fbps %+10.1fbps %+12.2f %7.0f%%",
                 y, m["n"], m["gross_bps"], m["net_bps"],
                 m["net_sharpe"], 100 * m["hit_rate"])

    # cost sensitivity
    log.info("\n=== COST SENSITIVITY ===")
    log.info("  %-12s %-15s %-12s %-18s",
             "cost (bps)", "net /4h (bps)", "net Sharpe", "95% CI")
    for c in (0, 6, 12, 18, 24, 36, 48):
        m = metrics(pnl, cost_bps=c)
        lo, hi = bootstrap_sharpe_ci(pnl, cost_bps=c)
        log.info("  %-12d %+13.1f %+12.2f  [%+.2f, %+.2f]",
                 c, m["net_bps"], m["net_sharpe"], lo, hi)


if __name__ == "__main__":
    main()
