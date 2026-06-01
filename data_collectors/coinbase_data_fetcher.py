"""Coinbase Spot data fetcher (Coinbase Exchange public API).

Public endpoint, no auth required.
Granularity: 60, 300, 900, 3600, 21600, 86400 seconds.
Max 300 candles per request.
Rate limit: 10 req/sec public.

Note: Coinbase quotes in USD (not USDT). USDT-USD basis is ~5-10 bps typically.
"""
from __future__ import annotations
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


class CoinbaseDataFetcher:
    def __init__(self, base_url: str = "https://api.exchange.coinbase.com"):
        self.base_url = base_url
        self.session = requests.Session()

    def binance_sym_to_coinbase(self, sym: str) -> str:
        """BTCUSDT -> BTC-USD."""
        if sym.endswith("USDT"):
            return f"{sym[:-4]}-USD"
        return sym

    def list_products(self) -> pd.DataFrame:
        r = self.session.get(f"{self.base_url}/products", timeout=15)
        r.raise_for_status()
        return pd.DataFrame(r.json())

    def fetch_candles_chunk(self, product: str, granularity: int,
                             start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch up to 300 candles. start/end inclusive in ISO8601."""
        url = f"{self.base_url}/products/{product}/candles"
        params = {
            "granularity": str(granularity),
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":   end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        r = self.session.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        rows = r.json()
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        df["open_time"] = pd.to_datetime(pd.to_numeric(df["time"]), unit="s", utc=True)
        df = df.sort_values("open_time").reset_index(drop=True)
        return df[["open_time", "open", "high", "low", "close", "volume"]]

    def fetch_backfill(self, product: str, start: datetime, end: Optional[datetime] = None,
                        granularity: int = 3600, sleep_ms: int = 120) -> pd.DataFrame:
        """Backfill by chunked requests of 300 candles each."""
        if end is None:
            end = datetime.now(timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        chunk_seconds = 300 * granularity   # 300 candles × granularity sec
        all_frames = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(seconds=chunk_seconds), end)
            df = self.fetch_candles_chunk(product, granularity, cursor, chunk_end)
            if len(df) > 0:
                all_frames.append(df)
            cursor = chunk_end + timedelta(seconds=granularity)   # advance past last candle
            time.sleep(sleep_ms / 1000.0)
        if not all_frames:
            return pd.DataFrame()
        out = pd.concat(all_frames, ignore_index=True).drop_duplicates("open_time")
        out = out[(out["open_time"] >= pd.Timestamp(start)) &
                  (out["open_time"] <= pd.Timestamp(end))]
        return out.sort_values("open_time").reset_index(drop=True)


if __name__ == "__main__":
    f = CoinbaseDataFetcher()
    df = f.fetch_backfill("BTC-USD",
        start=datetime(2026, 4, 1, tzinfo=timezone.utc),
        end=datetime(2026, 4, 5, tzinfo=timezone.utc),
        granularity=3600, sleep_ms=120)
    print(f"BTC-USD: {len(df)} rows, {df['open_time'].min()} → {df['open_time'].max()}")
