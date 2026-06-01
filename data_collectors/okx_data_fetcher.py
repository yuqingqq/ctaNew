"""OKX USDT-perpetual swap data fetcher.

Public market endpoints (no auth needed):
  - GET /api/v5/public/instruments?instType=SWAP  → list available swaps
  - GET /api/v5/market/candles                    → recent candles (300 per request)
  - GET /api/v5/market/history-candles            → historical candles (100 per req, deeper history)

Rate limit: 40 req/2s on market endpoints.
"""
from __future__ import annotations
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List

import pandas as pd
import requests


class OKXDataFetcher:
    def __init__(self, base_url: str = "https://www.okx.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def list_swap_instruments(self) -> pd.DataFrame:
        """Return all live USDT-perpetual SWAP instruments."""
        url = f"{self.base_url}/api/v5/public/instruments"
        r = self.session.get(url, params={"instType": "SWAP"}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX error: {data}")
        df = pd.DataFrame(data["data"])
        # filter to USDT-margined perpetuals
        df = df[df["settleCcy"] == "USDT"].copy()
        return df

    def binance_sym_to_okx(self, sym: str) -> str:
        """BTCUSDT -> BTC-USDT-SWAP."""
        if sym.endswith("USDT"):
            base = sym[:-4]
            return f"{base}-USDT-SWAP"
        return sym

    def fetch_candles(
        self,
        okx_symbol: str,
        bar: str = "1H",      # 1m, 3m, 5m, 15m, 30m, 1H, 4H, 1D
        before_ms: Optional[int] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch historical candles via history-candles (deeper history).
        Returns df with cols [open_time, open, high, low, close, volume, ...]
        Most recent first."""
        url = f"{self.base_url}/api/v5/market/history-candles"
        params = {"instId": okx_symbol, "bar": bar, "limit": str(limit)}
        if before_ms is not None:
            params["after"] = str(before_ms)
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "0":
            return pd.DataFrame()
        rows = data.get("data", [])
        if not rows:
            return pd.DataFrame()
        # columns per OKX docs: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        df = pd.DataFrame(rows, columns=[
            "ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"
        ])
        df["open_time"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "vol"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_values("open_time").reset_index(drop=True)
        return df[["open_time", "open", "high", "low", "close", "vol"]]

    def fetch_backfill(
        self,
        okx_symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        bar: str = "1H",
        limit_per_req: int = 100,
        sleep_ms: int = 60,
    ) -> pd.DataFrame:
        """Iteratively backfill candles from end → start using history-candles."""
        if end is None:
            end = datetime.now(timezone.utc)
        end_ms = int(end.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000)
        all_frames = []
        cursor_ms = end_ms
        seen_oldest = None
        while True:
            df = self.fetch_candles(okx_symbol, bar=bar, before_ms=cursor_ms, limit=limit_per_req)
            if len(df) == 0:
                break
            oldest_ms = int(df["open_time"].min().timestamp() * 1000)
            all_frames.append(df)
            # stop if we've reached past start
            if oldest_ms <= start_ms:
                break
            # advance cursor (must move back in time)
            if seen_oldest == oldest_ms:
                # API returned no new data
                break
            seen_oldest = oldest_ms
            cursor_ms = oldest_ms
            time.sleep(sleep_ms / 1000.0)
        if not all_frames:
            return pd.DataFrame()
        out = pd.concat(all_frames, ignore_index=True).drop_duplicates("open_time")
        out = out[(out["open_time"] >= pd.Timestamp(start)) &
                  (out["open_time"] <= pd.Timestamp(end))]
        out = out.sort_values("open_time").reset_index(drop=True)
        return out


if __name__ == "__main__":
    f = OKXDataFetcher()
    inst = f.list_swap_instruments()
    print(f"OKX USDT-perp swaps: {len(inst)}")
    print(inst[["instId", "settleCcy", "ctVal", "state"]].head(10).to_string())
