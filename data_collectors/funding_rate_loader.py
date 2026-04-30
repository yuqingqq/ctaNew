"""Binance Vision funding-rate loader.

Pulls monthly funding-rate zip archives from data.binance.vision and caches
them as parquet per symbol. Funding is published every 8h on Binance USDM
perps, so each daily-month file has ~93 rows.

Usage:
    from data_collectors.funding_rate_loader import load_funding_rate
    df = load_funding_rate("BTCUSDT", start_month="2025-01", end_month="2026-03")
    # → DataFrame indexed by calc_time UTC, columns: funding_rate, interval_hours

Anti-leakage note: each row's `calc_time` is the moment Binance computes and
publishes the funding rate. To use it as a feature at bar t, the rate must
have `calc_time <= t`. Use the merge_asof pattern to enforce this.
"""
from __future__ import annotations

import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
CACHE_DIR = Path("data/ml/cache")


def _months_between(start_month: str, end_month: str) -> list[str]:
    """Inclusive list of YYYY-MM strings between start_month and end_month."""
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def _download_month(symbol: str, ym: str) -> pd.DataFrame | None:
    """Fetch one month of funding rates. Returns DataFrame or None on 404."""
    url = f"{BASE_URL}/{symbol}/{symbol}-fundingRate-{ym}.zip"
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        log.warning("[%s] %s not available (404)", symbol, ym)
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(f)
    df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    df = df.rename(columns={
        "last_funding_rate": "funding_rate",
        "funding_interval_hours": "interval_hours",
    })
    return df[["calc_time", "interval_hours", "funding_rate"]].sort_values("calc_time")


def load_funding_rate(symbol: str, start_month: str = "2024-12",
                       end_month: str = "2026-03",
                       *, force_rebuild: bool = False) -> pd.DataFrame:
    """Load funding rate for one symbol. Caches per-symbol parquet.

    Returns DataFrame with columns: calc_time (UTC tz-aware), interval_hours,
    funding_rate. One row per funding settlement (typically every 8h).
    """
    cache = CACHE_DIR / f"funding_{symbol}.parquet"
    if not force_rebuild and cache.exists():
        return pd.read_parquet(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)

    months = _months_between(start_month, end_month)
    parts = []
    for ym in months:
        try:
            df = _download_month(symbol, ym)
        except requests.HTTPError as e:
            log.error("[%s] %s download failed: %s", symbol, ym, e)
            continue
        if df is not None:
            parts.append(df)
    if not parts:
        log.warning("[%s] no funding data downloaded", symbol)
        return pd.DataFrame(columns=["calc_time", "interval_hours", "funding_rate"])
    out = pd.concat(parts, ignore_index=True).sort_values("calc_time").drop_duplicates(subset="calc_time")
    out.to_parquet(cache, compression="zstd")
    log.info("[%s] cached %d funding rows from %s to %s",
              symbol, len(out), out["calc_time"].iloc[0], out["calc_time"].iloc[-1])
    return out


def load_funding_for_universe(symbols: list[str], **kwargs) -> dict[str, pd.DataFrame]:
    """Parallel pull for a list of symbols."""
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(load_funding_rate, s, **kwargs): s for s in symbols}
        for f in as_completed(futures):
            sym = futures[f]
            try:
                out[sym] = f.result()
            except Exception as e:
                log.error("[%s] failed: %s", sym, e)
                out[sym] = pd.DataFrame()
    return out


def align_funding_to_bars(funding: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.Series:
    """Align funding rates to a 5min bar grid via forward-fill.

    For each bar at open_time t, use the most recent funding_rate whose
    calc_time <= t. This is point-in-time: at bar t, only past funding
    publications are visible. Intervals between publications get the most
    recent value (constant within the 8h window).
    """
    if funding.empty:
        return pd.Series(index=bar_index, dtype="float64", name="funding_rate")
    f = funding.set_index("calc_time").sort_index()
    if bar_index.tz is None:
        bar_index = bar_index.tz_localize("UTC")
    # Use merge_asof to find the last calc_time <= each bar
    bars_df = pd.DataFrame({"open_time": bar_index})
    f_df = f[["funding_rate"]].reset_index().rename(columns={"calc_time": "open_time"})
    merged = pd.merge_asof(bars_df.sort_values("open_time"),
                             f_df.sort_values("open_time"),
                             on="open_time", direction="backward")
    return pd.Series(merged["funding_rate"].to_numpy(), index=bar_index, name="funding_rate")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    syms = sys.argv[1:] if len(sys.argv) > 1 else ["BTCUSDT"]
    for s in syms:
        df = load_funding_rate(s)
        print(s, len(df), df["calc_time"].min(), "→", df["calc_time"].max())
