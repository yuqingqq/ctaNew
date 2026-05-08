"""Polygon.io fetcher for US equity intraday bars.

Free tier: 5 req/min, 2 years of 5-minute history, 15-min delayed.
Endpoint: GET /v2/aggs/ticker/{ticker}/range/{mult}/{span}/{from}/{to}

Auth: API key as `apiKey` query param. Stored in /home/yuqing/ctaNew/.env
as POLYGON_API_KEY=<key>.

For 5m × 2y × 1 symbol = ~39k bars, fits in single request (50k cap).
Pagination handled via `next_url` if it appears.

Cache: data/ml/cache/poly_<sym>_<interval>.parquet
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "ml" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.polygon.io"
MIN_REQ_INTERVAL = 12.5   # 5 req/min = 1 req per 12s; pad a bit


class PolygonRateLimiter:
    def __init__(self, min_interval: float = MIN_REQ_INTERVAL):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        now = time.time()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.time()


def _api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("POLYGON_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "POLYGON_API_KEY missing. Append to /home/yuqing/ctaNew/.env: "
        "POLYGON_API_KEY=<your_key>"
    )


def fetch_aggs(symbol: str, interval: str = "5m",
               start: str = "2023-01-01", end: str | None = None,
               adjusted: bool = True,
               limiter: PolygonRateLimiter | None = None) -> pd.DataFrame:
    """Fetch aggregated bars. Returns long-format df with ts (UTC), OHLCV.

    interval: '1m', '5m', '15m', '30m', '1h', '1d' — parsed into mult+span.
    """
    cache = CACHE / f"poly_{symbol}_{interval}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        log.info("  [cache] %s %s n=%d  %s -> %s",
                 symbol, interval, len(df),
                 df["ts"].iloc[0].strftime("%Y-%m-%d"),
                 df["ts"].iloc[-1].strftime("%Y-%m-%d"))
        return df

    limiter = limiter or PolygonRateLimiter()
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    mult, span = _parse_interval(interval)
    url = (f"{API_BASE}/v2/aggs/ticker/{symbol.upper()}/range/{mult}/{span}/"
           f"{start}/{end}")
    params = {
        "adjusted": str(adjusted).lower(),
        "sort": "asc",
        "limit": 50000,
        "apiKey": _api_key(),
    }

    rows: list[dict] = []
    next_url = None
    while True:
        limiter.wait()
        if next_url is None:
            r = requests.get(url, params=params, timeout=60)
        else:
            # next_url already includes everything; just append apiKey
            sep = "&" if "?" in next_url else "?"
            r = requests.get(f"{next_url}{sep}apiKey={_api_key()}", timeout=60)
        if r.status_code == 429:
            log.warning("  rate-limited; sleeping 30s")
            time.sleep(30)
            continue
        r.raise_for_status()
        d = r.json()
        if d.get("status") == "ERROR":
            raise RuntimeError(f"Polygon error for {symbol}: {d.get('error') or d.get('message')}")
        results = d.get("results") or []
        rows.extend(results)
        next_url = d.get("next_url")
        if not next_url:
            break

    if not rows:
        log.warning("  %s: no bars returned", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "t": "ts_ms", "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "n": "n_trades", "vw": "vwap",
    })
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["symbol"] = symbol.upper()
    keep_cols = ["ts", "symbol", "open", "high", "low", "close",
                 "volume", "n_trades", "vwap"]
    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df.to_parquet(cache)
    log.info("  fetched %s %s n=%d  %s -> %s",
             symbol, interval, len(df),
             df["ts"].iloc[0].strftime("%Y-%m-%d"),
             df["ts"].iloc[-1].strftime("%Y-%m-%d"))
    return df


def _parse_interval(s: str) -> tuple[int, str]:
    s = s.lower()
    if s.endswith("m"):
        return int(s[:-1]), "minute"
    if s.endswith("h"):
        return int(s[:-1]), "hour"
    if s.endswith("d"):
        return int(s[:-1]), "day"
    raise ValueError(f"unrecognized interval: {s}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    UNIVERSE = ["NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META",
                "AAPL", "MSFT", "ORCL", "INTC", "MU", "NFLX"]
    # Polygon free-tier 5m boundary is ~730d. Use 700d for safety margin.
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=700)).strftime("%Y-%m-%d")
    log.info("fetching %d symbols, 5m, %s -> %s", len(UNIVERSE), start, end)
    limiter = PolygonRateLimiter()
    for sym in UNIVERSE:
        fetch_aggs(sym, "5m", start=start, end=end, limiter=limiter)
