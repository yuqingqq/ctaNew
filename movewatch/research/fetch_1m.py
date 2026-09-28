"""Historical 1m OHLCV fetchers, one per venue, for case studies and replay.

Every fetcher returns a DataFrame indexed by UTC minute (open time) with columns
open, high, low, close, volume (base units), quote_volume (quote units; USD-ish
for USDT/USD pairs), and optionally taker_buy_quote. Missing minutes are left
missing (no forward fill) -- a gap is information, not a zero.

Endpoint notes (verified from this box 2026-09-28):
  - api.binance.com / fapi.binance.com return HTTP 451 here (geo-block).
    Spot REST works via data-api.binance.vision; USDM history comes from the
    Binance Vision daily archives (T+1, so the current UTC day is missing).
  - api.bybit.com REST returns 403 here; Bybit history comes from the
    public.bybit.com trade archive (T+1), aggregated to 1m.
  - OKX, Coinbase, Bitget, Gate, KuCoin, MEXC, Upbit, Hyperliquid REST are reachable.

Cache: movewatch/data/hist/<venue>/<symbol>/<YYYY-MM-DD>.parquet, complete
past UTC days only (the current day is always refetched).
"""
from __future__ import annotations

import gzip
import io
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "hist"
UA = {"User-Agent": "movewatch-research/0.1"}
COLS = ["open", "high", "low", "close", "volume", "quote_volume"]


def _get(url: str, params: dict | None = None, tries: int = 4, timeout: float = 20.0):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=UA)
            if r.status_code == 429:
                time.sleep(1.5 * (k + 1))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # network hiccup: back off and retry
            last = e
            time.sleep(1.0 * (k + 1))
    raise RuntimeError(f"GET failed after {tries} tries: {url} {params} ({last})")


def _frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    for c in df.columns:
        df[c] = df[c].astype(float)
    return df


def _days(start: datetime, end: datetime) -> list[datetime]:
    d = start.replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    while d < end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _cached_by_day(venue: str, symbol: str, start: datetime, end: datetime, fetch_day) -> pd.DataFrame:
    """Fetch per UTC day through fetch_day(day_start, day_end) with a parquet cache
    for complete past days."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    parts = []
    for d in _days(start, end):
        d_end = d + timedelta(days=1)
        path = CACHE / venue / symbol / f"{d:%Y-%m-%d}.parquet"
        if d < today and path.exists():
            parts.append(pd.read_parquet(path))
            continue
        df = fetch_day(d, min(d_end, datetime.now(timezone.utc)))
        if d < today and df is not None and len(df):
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
        if df is not None and len(df):
            parts.append(df)
    if not parts:
        return _frame([])
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[(out.index >= start) & (out.index < end)]


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


# --------------------------------------------------------------------------- Binance spot
def binance_spot(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    def day(d0, d1):
        rows, t = [], _ms(d0)
        while t < _ms(d1):
            r = _get("https://data-api.binance.vision/api/v3/klines",
                     dict(symbol=symbol, interval="1m", startTime=t, endTime=_ms(d1) - 1, limit=1000))
            data = r.json() if r is not None else []
            if not data:
                break
            for k in data:
                rows.append(dict(ts=k[0], open=k[1], high=k[2], low=k[3], close=k[4], volume=k[5],
                                 quote_volume=k[7], taker_buy_quote=k[10], trades=k[8]))
            t = data[-1][0] + 60_000
        return _frame(rows)
    return _cached_by_day("binance_spot", symbol, start, end, day)


# --------------------------------------------------------------------------- Binance USDM (Vision, T+1)
def binance_usdm(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    def day(d0, d1):
        url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/1m/{symbol}-1m-{d0:%Y-%m-%d}.zip"
        r = _get(url)
        if r is None:
            return _frame([])
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = z.read(z.namelist()[0]).decode()
        lines = [ln for ln in raw.splitlines() if ln and ln[0].isdigit()]  # skip optional header
        rows = []
        for ln in lines:
            k = ln.split(",")
            rows.append(dict(ts=int(k[0]), open=k[1], high=k[2], low=k[3], close=k[4], volume=k[5],
                             quote_volume=k[7], taker_buy_quote=k[10], trades=k[8]))
        return _frame(rows)
    return _cached_by_day("binance_usdm", symbol, start, end, day)


def binance_usdm_metrics(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """5-minute open interest + long/short ratios from Vision daily metrics (T+1)."""
    def day(d0, d1):
        url = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{d0:%Y-%m-%d}.zip"
        r = _get(url)
        if r is None:
            return _frame([])
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            df = pd.read_csv(z.open(z.namelist()[0]))
        df["ts"] = pd.to_datetime(df["create_time"], utc=True)
        df = df.drop(columns=["create_time", "symbol"]).set_index("ts").sort_index()
        return df.astype(float)
    return _cached_by_day("binance_usdm_metrics", symbol, start, end, day)


# --------------------------------------------------------------------------- Bybit linear (archive, T+1)
def bybit_linear(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    def day(d0, d1):
        url = f"https://public.bybit.com/trading/{symbol}/{symbol}{d0:%Y-%m-%d}.csv.gz"
        r = _get(url, timeout=60)
        if r is None:
            return _frame([])
        tr = pd.read_csv(io.BytesIO(gzip.decompress(r.content)))
        # columns: timestamp (s, float), symbol, side, size, price, tickDirection, trdMatchID,
        #          grossValue, homeNotional, foreignNotional
        tr["ts"] = pd.to_datetime((tr["timestamp"] * 1000).astype("int64"), unit="ms", utc=True)
        tr["q"] = tr["foreignNotional"].astype(float)
        tr["tb"] = np.where(tr["side"] == "Buy", tr["q"], 0.0)
        g = tr.set_index("ts").sort_index().groupby(pd.Grouper(freq="1min"))
        df = pd.DataFrame({
            "open": g["price"].first(), "high": g["price"].max(), "low": g["price"].min(),
            "close": g["price"].last(), "volume": g["homeNotional"].sum(),
            "quote_volume": g["q"].sum(), "taker_buy_quote": g["tb"].sum(), "trades": g["price"].count(),
        }).dropna(subset=["close"])
        return df
    return _cached_by_day("bybit_linear", symbol, start, end, day)


# --------------------------------------------------------------------------- OKX (spot or swap)
def okx(inst_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    def day(d0, d1):
        rows, after = [], _ms(d1)
        while True:
            r = _get("https://www.okx.com/api/v5/market/history-candles",
                     dict(instId=inst_id, bar="1m", after=after, limit=100))
            data = r.json().get("data", []) if r is not None else []
            if not data:
                break
            for k in data:  # [ts, o, h, l, c, vol(contracts/base), volCcy(base), volCcyQuote, confirm]
                ts = int(k[0])
                if ts < _ms(d0):
                    continue
                rows.append(dict(ts=ts, open=k[1], high=k[2], low=k[3], close=k[4],
                                 volume=k[6], quote_volume=k[7]))
            after = int(data[-1][0])
            if after <= _ms(d0):
                break
            time.sleep(0.12)
        return _frame(rows)
    return _cached_by_day("okx", inst_id, start, end, day)


# --------------------------------------------------------------------------- Coinbase
def coinbase(product: str, start: datetime, end: datetime) -> pd.DataFrame:
    def day(d0, d1):
        rows, t = [], d0
        while t < d1:
            t2 = min(t + timedelta(minutes=300), d1)
            r = _get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                     dict(granularity=60, start=t.isoformat(), end=t2.isoformat()))
            for k in (r.json() if r is not None else []):  # [time(s), low, high, open, close, volume]
                if _ms(d0) <= k[0] * 1000 < _ms(d1):
                    rows.append(dict(ts=k[0] * 1000, open=k[3], high=k[2], low=k[1], close=k[4],
                                     volume=k[5], quote_volume=float(k[5]) * float(k[4])))
            t = t2
            time.sleep(0.15)
        return _frame(rows)
    return _cached_by_day("coinbase", product, start, end, day)


# --------------------------------------------------------------------------- Bitget (spot / USDT-FUTURES)
def bitget(symbol: str, start: datetime, end: datetime, futures: bool = False) -> pd.DataFrame:
    venue = "bitget_fut" if futures else "bitget_spot"

    def day(d0, d1):
        rows, end_t = [], _ms(d1)
        while end_t > _ms(d0):
            if futures:
                r = _get("https://api.bitget.com/api/v2/mix/market/history-candles",
                         dict(symbol=symbol, productType="USDT-FUTURES", granularity="1m",
                              endTime=end_t, limit=200))
            else:
                r = _get("https://api.bitget.com/api/v2/spot/market/history-candles",
                         dict(symbol=symbol, granularity="1min", endTime=end_t, limit=200))
            data = r.json().get("data", []) if r is not None else []
            if not data:
                break
            for k in data:  # [ts, o, h, l, c, baseVol, quoteVol(, usdtVol)]
                ts = int(k[0])
                if _ms(d0) <= ts < _ms(d1):
                    rows.append(dict(ts=ts, open=k[1], high=k[2], low=k[3], close=k[4],
                                     volume=k[5], quote_volume=k[6]))
            first = min(int(k[0]) for k in data)
            if first >= end_t:
                break
            end_t = first
            time.sleep(0.08)
        return _frame(rows)
    return _cached_by_day(venue, symbol, start, end, day)


# --------------------------------------------------------------------------- Gate spot
def gate(pair: str, start: datetime, end: datetime) -> pd.DataFrame:
    def day(d0, d1):
        rows, t = [], int(d0.timestamp())
        while t < int(d1.timestamp()):
            t2 = min(t + 900 * 60, int(d1.timestamp()))
            r = _get("https://api.gateio.ws/api/v4/spot/candlesticks",
                     dict(currency_pair=pair, interval="1m", **{"from": t, "to": t2 - 1}))
            for k in (r.json() if r is not None else []):  # [t, quote_vol, close, high, low, open, base_vol, closed]
                rows.append(dict(ts=int(k[0]) * 1000, open=k[5], high=k[3], low=k[4], close=k[2],
                                 volume=k[6], quote_volume=k[1]))
            t = t2
            time.sleep(0.1)
        return _frame(rows)
    return _cached_by_day("gate", pair, start, end, day)


# --------------------------------------------------------------------------- KuCoin spot
def kucoin(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    def day(d0, d1):
        r = _get("https://api.kucoin.com/api/v1/market/candles",
                 dict(type="1min", symbol=symbol, startAt=int(d0.timestamp()), endAt=int(d1.timestamp())))
        data = r.json().get("data", []) if r is not None else []
        rows = [dict(ts=int(k[0]) * 1000, open=k[1], close=k[2], high=k[3], low=k[4],
                     volume=k[5], quote_volume=k[6]) for k in data]  # [t, o, c, h, l, vol, turnover]
        return _frame(rows)
    return _cached_by_day("kucoin", symbol, start, end, day)


# --------------------------------------------------------------------------- MEXC spot
def mexc(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    def day(d0, d1):
        rows, t = [], _ms(d0)
        while t < _ms(d1):
            r = _get("https://api.mexc.com/api/v3/klines",
                     dict(symbol=symbol, interval="1m", startTime=t, endTime=_ms(d1) - 1, limit=500))
            data = r.json() if r is not None else []
            if not data:
                break
            for k in data:
                rows.append(dict(ts=k[0], open=k[1], high=k[2], low=k[3], close=k[4],
                                 volume=k[5], quote_volume=k[7]))
            t = int(data[-1][0]) + 60_000
            time.sleep(0.1)
        return _frame(rows)
    return _cached_by_day("mexc", symbol, start, end, day)


# --------------------------------------------------------------------------- Upbit (KRW)
def upbit(market: str, start: datetime, end: datetime) -> pd.DataFrame:
    def day(d0, d1):
        rows, to = [], d1
        while to > d0:
            r = _get("https://api.upbit.com/v1/candles/minutes/1",
                     dict(market=market, to=to.strftime("%Y-%m-%dT%H:%M:%SZ"), count=200))
            data = r.json() if r is not None else []
            if not data:
                break
            for k in data:
                ts = int(pd.Timestamp(k["candle_date_time_utc"], tz="UTC").timestamp() * 1000)
                if _ms(d0) <= ts < _ms(d1):
                    rows.append(dict(ts=ts, open=k["opening_price"], high=k["high_price"],
                                     low=k["low_price"], close=k["trade_price"],
                                     volume=k["candle_acc_trade_volume"],
                                     quote_volume=k["candle_acc_trade_price"]))
            oldest = min(pd.Timestamp(k["candle_date_time_utc"], tz="UTC") for k in data)
            if oldest.to_pydatetime() >= to:
                break
            to = oldest.to_pydatetime()
            time.sleep(0.12)
        return _frame(rows)
    return _cached_by_day("upbit", market, start, end, day)


VENUE_FETCHERS = {
    "binance_spot": binance_spot,
    "binance_usdm": binance_usdm,
    "bybit_linear": bybit_linear,
    "okx": okx,
    "coinbase": coinbase,
    "bitget_spot": lambda s, a, b: bitget(s, a, b, futures=False),
    "bitget_fut": lambda s, a, b: bitget(s, a, b, futures=True),
    "gate": gate,
    "kucoin": kucoin,
    "mexc": mexc,
    "upbit": upbit,
}
