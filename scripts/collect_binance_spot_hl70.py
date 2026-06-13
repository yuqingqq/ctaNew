"""Collect Binance SPOT 1h klines for all 70 HL syms via Binance public API.

Saves: data/ml/cache/bn_spot_<SYM>_1h.parquet

Note: Binance SPOT may not list all HL-tradeable syms (recent listings often perp-only).
"""
from __future__ import annotations
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO = Path("/home/yuqing/ctaNew")
CACHE = REPO / "data/ml/cache"

START_MS = int(datetime(2025, 3, 23, tzinfo=timezone.utc).timestamp() * 1000)
END_MS = int(datetime(2026, 5, 7, tzinfo=timezone.utc).timestamp() * 1000)
INTERVAL = "1h"
LIMIT = 1000  # max per req


def fetch_spot_klines(symbol: str) -> pd.DataFrame:
    """Page through Binance spot klines."""
    url = "https://api.binance.com/api/v3/klines"
    all_rows = []
    cursor = START_MS
    while cursor < END_MS:
        params = dict(symbol=symbol, interval=INTERVAL, startTime=cursor,
                       endTime=END_MS, limit=LIMIT)
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                if r.status_code == 400:
                    return pd.DataFrame()  # sym not listed on spot
                print(f"    HTTP {r.status_code}: {r.text[:100]}")
                time.sleep(2)
                continue
            data = r.json()
            if not data: break
            all_rows.extend(data)
            cursor = data[-1][0] + 3600 * 1000  # advance by 1 hour
            time.sleep(0.05)  # rate limit
        except Exception as e:
            print(f"    ERR: {e}")
            time.sleep(2)
            continue
    if not all_rows: return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df[["open_time", "close", "volume"]]


def main():
    # 70 HL syms
    hm = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    syms = sorted(hm[hm.on_hl]["symbol"].tolist())
    print(f"Collecting Binance SPOT for {len(syms)} HL-70 syms")
    success, failed = 0, 0
    for i, sym in enumerate(syms, 1):
        out_path = CACHE / f"bn_spot_{sym}_1h.parquet"
        if out_path.exists():
            print(f"  [{i}/{len(syms)}] {sym}: cached, skip")
            success += 1
            continue
        print(f"  [{i}/{len(syms)}] {sym}...", flush=True)
        try:
            df = fetch_spot_klines(sym)
            if len(df) == 0:
                print(f"    no data (sym not on Binance SPOT)")
                failed += 1
                continue
            df.to_parquet(out_path, index=False)
            print(f"    saved {len(df):,} rows")
            success += 1
        except Exception as e:
            print(f"    ERR: {e}")
            failed += 1
    print(f"\nDone: {success}/{len(syms)} success, {failed} failed/missing")


if __name__ == "__main__":
    main()
