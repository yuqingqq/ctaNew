"""Binance Vision daily-archive loader.

Downloads daily zipped CSV dumps from data.binance.vision (free, no auth) for
USDT-margined perpetual futures and writes a single parquet per dataset+symbol+
date-range.

Datasets supported:
    - klines:    Kline/candlestick (any interval; default 5m)
    - aggTrades: Aggregated trades (one row per match-engine aggregation)

Files cache to <out_dir>/raw/<dataset>/<symbol>/<YYYY-MM-DD>.csv.zip and parse
to <out_dir>/parquet/<dataset>/<symbol>/<interval-or-empty>/.../{date}.parquet.
Missing days are downloaded; existing days are skipped.
"""

from __future__ import annotations

import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um/daily"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
AGGTRADE_COLUMNS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
]


@dataclass
class LoaderConfig:
    symbol: str = "BTCUSDT"
    out_dir: Path = Path("data/ml")
    max_workers: int = 4
    timeout_s: int = 60


def _daterange(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _archive_url(dataset: str, symbol: str, day: date, interval: str | None) -> str:
    if dataset == "klines":
        assert interval is not None, "klines requires interval"
        return f"{BASE_URL}/klines/{symbol}/{interval}/{symbol}-{interval}-{day:%Y-%m-%d}.zip"
    if dataset == "aggTrades":
        return f"{BASE_URL}/aggTrades/{symbol}/{symbol}-aggTrades-{day:%Y-%m-%d}.zip"
    raise ValueError(f"unknown dataset: {dataset}")


def _raw_path(cfg: LoaderConfig, dataset: str, day: date, interval: str | None) -> Path:
    sub = f"{dataset}/{cfg.symbol}" + (f"/{interval}" if interval else "")
    return cfg.out_dir / "raw" / sub / f"{day:%Y-%m-%d}.zip"


def _parquet_path(cfg: LoaderConfig, dataset: str, day: date, interval: str | None) -> Path:
    sub = f"{dataset}/{cfg.symbol}" + (f"/{interval}" if interval else "")
    return cfg.out_dir / "parquet" / sub / f"{day:%Y-%m-%d}.parquet"


def _download_one(url: str, dest: Path, timeout_s: int) -> bool:
    """Download `url` to `dest`. Returns True if downloaded, False if skipped."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout_s)
    if resp.status_code == 404:
        log.warning("404 (not yet published?): %s", url)
        return False
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return True


def _parse_zip_to_df(zip_bytes: bytes, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Each archive contains exactly one CSV with the same name as the zip.
        name = zf.namelist()[0]
        with zf.open(name) as f:
            raw = f.read()
    # Binance Vision archives added a header row at some point in their history;
    # detect by looking at the first byte of the decompressed CSV. Numeric → no
    # header (timestamp); alphabetic → header row to skip.
    has_header = raw[:1].isalpha()
    df = pd.read_csv(
        io.BytesIO(raw),
        header=0 if has_header else None,
        names=columns,
    )
    return df


def _kline_postprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = df[col].astype("float64")
    df["count"] = df["count"].astype("int64")
    return df.drop(columns=["ignore"])


def _aggtrade_postprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["transact_time"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df["price"] = df["price"].astype("float64")
    df["quantity"] = df["quantity"].astype("float64")
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    return df


def _process_day(
    cfg: LoaderConfig,
    dataset: str,
    day: date,
    interval: str | None,
) -> Path | None:
    """Download (if missing) and parse one day. Returns parquet path or None on 404."""
    raw = _raw_path(cfg, dataset, day, interval)
    pq = _parquet_path(cfg, dataset, day, interval)
    if pq.exists() and pq.stat().st_size > 0:
        return pq

    url = _archive_url(dataset, cfg.symbol, day, interval)
    _download_one(url, raw, cfg.timeout_s)
    if not raw.exists():
        return None

    columns = KLINE_COLUMNS if dataset == "klines" else AGGTRADE_COLUMNS
    df = _parse_zip_to_df(raw.read_bytes(), columns)
    df = _kline_postprocess(df) if dataset == "klines" else _aggtrade_postprocess(df)

    pq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq, compression="zstd", index=False)
    return pq


def fetch_klines(
    start: date,
    end: date,
    *,
    interval: str = "5m",
    cfg: LoaderConfig | None = None,
) -> pd.DataFrame:
    """Fetch klines for [start, end] inclusive; concatenate to one DataFrame."""
    cfg = cfg or LoaderConfig()
    days = list(_daterange(start, end))
    paths = _parallel_process(cfg, "klines", days, interval)
    if not paths:
        return pd.DataFrame(columns=[c for c in KLINE_COLUMNS if c != "ignore"])
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return df.sort_values("open_time").reset_index(drop=True)


def fetch_aggtrades(
    start: date,
    end: date,
    *,
    cfg: LoaderConfig | None = None,
) -> pd.DataFrame:
    """Fetch aggTrades for [start, end] inclusive. Returns concatenated DataFrame.

    AggTrades volume is large (10s of millions of rows over 30d for SOL).
    Prefer iterating over per-day parquets via list_aggtrade_paths() for big ranges.
    """
    cfg = cfg or LoaderConfig()
    days = list(_daterange(start, end))
    paths = _parallel_process(cfg, "aggTrades", days, None)
    if not paths:
        return pd.DataFrame(columns=AGGTRADE_COLUMNS)
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return df.sort_values("transact_time").reset_index(drop=True)


def list_aggtrade_paths(
    start: date,
    end: date,
    *,
    cfg: LoaderConfig | None = None,
) -> list[Path]:
    """Download (if missing) and return per-day parquet paths without concatenating."""
    cfg = cfg or LoaderConfig()
    days = list(_daterange(start, end))
    return _parallel_process(cfg, "aggTrades", days, None)


def _parallel_process(
    cfg: LoaderConfig,
    dataset: str,
    days: Iterable[date],
    interval: str | None,
) -> list[Path]:
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = {ex.submit(_process_day, cfg, dataset, d, interval): d for d in days}
        for fut in as_completed(futures):
            day = futures[fut]
            try:
                p = fut.result()
            except Exception as e:
                log.error("failed %s %s: %s", dataset, day, e)
                continue
            if p is not None:
                paths.append(p)
    return sorted(paths)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Binance Vision daily-archive loader")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--dataset", choices=["klines", "aggTrades"], required=True)
    ap.add_argument("--interval", default="5m", help="kline interval (klines only)")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out-dir", default="data/ml", type=Path)
    ap.add_argument("--workers", default=4, type=int)
    args = ap.parse_args()

    cfg = LoaderConfig(symbol=args.symbol, out_dir=args.out_dir, max_workers=args.workers)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    if args.dataset == "klines":
        df = fetch_klines(start, end, interval=args.interval, cfg=cfg)
        log.info("loaded %d kline rows: %s → %s", len(df), df["open_time"].min(), df["open_time"].max())
    else:
        paths = list_aggtrade_paths(start, end, cfg=cfg)
        log.info("loaded %d aggTrade per-day parquets", len(paths))
