"""S&P 100 universe loader: yfinance daily OHLCV + earnings dates.

Caches each per-symbol result to data/ml/cache/. Can be run standalone:
    python3 -m data_collectors.sp100_loader

Filters:
    - names with < min_history_days are excluded from final universe.
    - names with no fetchable earnings calendar are excluded.

Result: a "panel-ready" set of tickers all with daily OHLCV and earnings
dates, suitable for PEAD-style cross-sectional probes.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "ml" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# S&P 100 (queried 2026-05-06 from Wikipedia)
SP100_RAW = (
    "AAPL,ABBV,ABT,ACN,ADBE,AMAT,AMD,AMGN,AMT,AMZN,AVGO,AXP,BA,BAC,BK,"
    "BKNG,BLK,BMY,BRK.B,C,CAT,CL,CMCSA,COF,COP,COST,CRM,CSCO,CVS,CVX,DE,"
    "DHR,DIS,DUK,EMR,FDX,GD,GE,GEV,GILD,GM,GOOG,GOOGL,GS,HD,HON,IBM,INTC,"
    "INTU,ISRG,JNJ,JPM,KO,LIN,LLY,LMT,LOW,LRCX,MA,MCD,MDLZ,MDT,META,MMM,"
    "MO,MRK,MS,MSFT,MU,NEE,NFLX,NKE,NOW,NVDA,ORCL,PEP,PFE,PG,PLTR,PM,"
    "QCOM,RTX,SBUX,SCHW,SO,SPG,T,TMO,TMUS,TSLA,TXN,UBER,UNH,UNP,UPS,USB,"
    "V,VZ,WFC,WMT,XOM"
)
SP100 = [t.replace(".", "-") for t in SP100_RAW.split(",")]


def fetch_daily(symbol: str, start: str = "2013-01-01",
                end: str | None = None) -> pd.DataFrame:
    cache = CACHE / f"yf_{symbol}_1d_sp100.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        df = yf.Ticker(symbol).history(start=start, end=end, interval="1d",
                                       auto_adjust=True)
    except Exception as e:
        log.warning("  %s daily fetch failed: %s", symbol, e)
        return pd.DataFrame()
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


def fetch_earnings(symbol: str, limit: int = 80) -> pd.DataFrame:
    """Return DataFrame with cols [ts, eps_est, eps_actual, surprise_pct]."""
    cache = CACHE / f"yf_{symbol}_earnings.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    try:
        df = yf.Ticker(symbol).get_earnings_dates(limit=limit)
    except Exception as e:
        log.warning("  %s earnings fetch failed: %s", symbol, e)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if "earnings date" in cl: rename[c] = "ts"
        elif "eps estimate" in cl: rename[c] = "eps_est"
        elif "reported eps" in cl: rename[c] = "eps_actual"
        elif "surprise" in cl: rename[c] = "surprise_pct"
    df = df.rename(columns=rename)
    keep = [c for c in ("ts", "eps_est", "eps_actual", "surprise_pct") if c in df.columns]
    df = df[keep]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["symbol"] = symbol
    df.to_parquet(cache)
    return df


def load_universe(min_history_days: int = 5 * 252,
                   start: str = "2013-01-01") -> pd.DataFrame:
    """Fetch all S&P 100 names, return a long-format daily panel for those
    with sufficient history and at least 4 earnings events on record.
    """
    rows = []
    earnings_rows = []
    surviving = []
    for i, sym in enumerate(SP100):
        d = fetch_daily(sym, start=start)
        if d.empty or len(d) < min_history_days:
            log.info("  %3d/%d %-6s skip (n=%d)", i + 1, len(SP100), sym, len(d))
            continue
        e = fetch_earnings(sym)
        if e.empty or len(e) < 4:
            log.info("  %3d/%d %-6s skip (earnings=%d)", i + 1, len(SP100), sym, len(e))
            continue
        rows.append(d)
        earnings_rows.append(e)
        surviving.append(sym)
        log.info("  %3d/%d %-6s daily n=%5d  earnings n=%2d  %s -> %s",
                 i + 1, len(SP100), sym, len(d), len(e),
                 d["ts"].iloc[0].strftime("%Y-%m-%d"),
                 d["ts"].iloc[-1].strftime("%Y-%m-%d"))
        # gentle pacing for Yahoo
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), []
    panel = pd.concat(rows, ignore_index=True)
    earnings = pd.concat(earnings_rows, ignore_index=True)
    return panel, earnings, surviving


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log.info("loading S&P 100 universe (~100 names)...")
    panel, earnings, surv = load_universe()
    log.info("kept %d names", len(surv))
    log.info("daily panel: %d rows", len(panel))
    log.info("earnings events: %d total (%.1f per name avg)",
             len(earnings), len(earnings) / max(len(surv), 1))
