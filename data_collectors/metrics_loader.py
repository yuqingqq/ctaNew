"""Binance Vision FUTURES /metrics/ archive loader.

Provides per-symbol open interest + long/short ratios at 5-min cadence
from public Binance daily archives.

URL pattern:
    https://data.binance.vision/data/futures/um/daily/metrics/{SYM}/
                                                    {SYM}-metrics-{YYYY-MM-DD}.zip

Each daily zip contains a CSV with 288 rows (5min × 24h):
    create_time, symbol, sum_open_interest, sum_open_interest_value,
    count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
    count_long_short_ratio, sum_taker_long_short_vol_ratio

Outputs concatenated per-symbol parquet to data/ml/cache/metrics_<SYM>.parquet
indexed by open_time (UTC, tz-aware).
"""
from __future__ import annotations
import io
import logging
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "data/ml/cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
NUM_WORKERS = 8


def _date_range(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _fetch_one(symbol: str, day: date, *, timeout: int = 30) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{symbol}/{symbol}-metrics-{day:%Y-%m-%d}.zip"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                return None  # day not available
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = z.namelist()[0]
                with z.open(name) as f:
                    df = pd.read_csv(f)
            return df
        except (requests.exceptions.RequestException, zipfile.BadZipFile) as e:
            if attempt == 2:
                log.warning("[%s %s] fetch failed: %s", symbol, day, e)
                return None
            time.sleep(2 ** attempt)


def fetch_metrics(symbol: str, start: date, end: date,
                    *, force: bool = False) -> pd.DataFrame:
    """Pull metrics for one symbol; cache concatenated output to parquet."""
    cache = CACHE_DIR / f"metrics_{symbol}.parquet"
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        if df.index.min().date() <= start and df.index.max().date() >= end:
            return df.loc[
                (df.index.date >= start) & (df.index.date <= end)
            ].copy()

    days = list(_date_range(start, end))
    log.info("[%s] pulling metrics for %d days (%s → %s)",
              symbol, len(days), start, end)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(_fetch_one, symbol, d): d for d in days}
        frames = []
        n_done = 0
        for f in as_completed(futs):
            df = f.result()
            if df is not None:
                frames.append(df)
            n_done += 1
            if n_done % 50 == 0:
                log.info("  [%s] %d/%d days fetched", symbol, n_done, len(days))

    if not frames:
        log.warning("[%s] no data fetched", symbol)
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["create_time"] = pd.to_datetime(out["create_time"], utc=True)
    out = out.set_index("create_time").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    # downcast
    for c in out.select_dtypes("float64").columns:
        out[c] = out[c].astype("float32")
    out.to_parquet(cache, compression="zstd")
    log.info("[%s] saved %d rows → %s", symbol, len(out), cache)
    return out


def fetch_all(symbols: list[str], start: date, end: date) -> dict:
    """Fetch metrics for all symbols, return dict[sym] → DataFrame."""
    return {s: fetch_metrics(s, start, end) for s in symbols}


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: metrics_loader.py SYM1,SYM2,... START END")
        print("       metrics_loader.py BTCUSDT,ETHUSDT 2025-03-20 2026-04-28")
        sys.exit(1)
    syms = sys.argv[1].split(",")
    start_d = datetime.strptime(sys.argv[2], "%Y-%m-%d").date()
    end_d = datetime.strptime(sys.argv[3], "%Y-%m-%d").date()
    for s in syms:
        fetch_metrics(s, start_d, end_d)
