"""Collect Binance SPOT 1h klines via data.binance.vision (Vision archives).

Binance public API (api.binance.com) is geo-blocked from this server.
Vision is accessible. Downloads monthly archives, parses CSVs.

URL pattern: https://data.binance.vision/data/spot/monthly/klines/<SYM>/1h/<SYM>-1h-<YYYY-MM>.zip
"""
from __future__ import annotations
import sys, time, io, zipfile
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd
import requests

REPO = Path("/home/yuqing/ctaNew")
CACHE = REPO / "data/ml/cache"
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

START_DATE = date(2025, 3, 1)
END_DATE = date(2026, 5, 1)
INTERVAL = "1h"


def fetch_spot_monthly(sym: str, ym: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{sym}/{INTERVAL}/{sym}-{INTERVAL}-{ym}.zip"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return None  # sym not listed on spot or no data this month
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "n_trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ])
        # open_time is ms epoch
        if df["open_time"].max() > 1e14:
            # microsecond epoch
            df["open_time"] = pd.to_datetime(df["open_time"], unit="us", utc=True)
        else:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df[["open_time", "close", "volume"]]
    except requests.HTTPError as e:
        return None
    except Exception as e:
        print(f"    {ym}: ERR {e}")
        return None


def main():
    hm = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    syms = sorted(hm[hm.on_hl]["symbol"].tolist())
    print(f"Collecting Binance SPOT via Vision for {len(syms)} HL-70 syms")

    # Generate month tags Y-M
    months = []
    cur = START_DATE
    while cur < END_DATE:
        months.append(f"{cur.year}-{cur.month:02d}")
        if cur.month == 12: cur = date(cur.year + 1, 1, 1)
        else: cur = date(cur.year, cur.month + 1, 1)

    success, missing = 0, 0
    for i, sym in enumerate(syms, 1):
        out_path = CACHE / f"bn_spot_{sym}_1h.parquet"
        if out_path.exists():
            print(f"  [{i}/{len(syms)}] {sym}: cached, skip")
            success += 1
            continue
        all_dfs = []
        for ym in months:
            df = fetch_spot_monthly(sym, ym)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
            time.sleep(0.05)
        if not all_dfs:
            print(f"  [{i}/{len(syms)}] {sym}: no data on Binance SPOT")
            missing += 1
            continue
        combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
        combined.to_parquet(out_path, index=False)
        print(f"  [{i}/{len(syms)}] {sym}: saved {len(combined):,} rows")
        success += 1

    print(f"\nDone: {success}/{len(syms)} have data, {missing} missing from Binance SPOT")


if __name__ == "__main__":
    main()
