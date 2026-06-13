"""X60 — Extend historical klines/aggT/OKX/CB back to 2023-01-01.

Identifies syms with sufficient pre-2025-03 history and downloads:
  - Binance perp klines (5m) via Binance Vision daily archives
  - Binance perp aggTrades
  - OKX swap + spot 1h klines
  - Coinbase 1h klines
  - Binance spot 1h klines via Vision

Saves to existing cache locations (extends earlier date range).
Syms only extended IF they were listed before 2023-01.
"""
from __future__ import annotations
import sys, time, io, zipfile
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd
import requests

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
CACHE = REPO / "data/ml/cache"
KLINES = REPO / "data/ml/test/parquet/klines"

START_NEW = date(2023, 1, 1)
END_NEW = date(2025, 3, 22)  # don't overlap existing data


def fetch_bn_spot_monthly(sym, ym):
    """Binance Vision SPOT monthly archive."""
    url = f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1h/{sym}-1h-{ym}.zip"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404: return None
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "n_trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ])
        if df["open_time"].max() > 1e14:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="us", utc=True)
        else:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df[["open_time", "close", "volume"]]
    except Exception:
        return None


def extend_bn_spot(sym):
    """Extend BN spot history."""
    out_path = CACHE / f"bn_spot_{sym}_1h.parquet"
    if not out_path.exists():
        print(f"  {sym}: bn_spot doesn't exist; skip"); return
    existing = pd.read_parquet(out_path)
    existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True)
    existing_start = existing["open_time"].min()
    if existing_start <= pd.Timestamp("2023-01-15", tz="UTC"):
        print(f"  {sym}: already has pre-2023 history"); return

    # Fetch months from START_NEW to existing_start
    months = []
    cur = START_NEW
    while cur < existing_start.date():
        months.append(f"{cur.year}-{cur.month:02d}")
        if cur.month == 12: cur = date(cur.year+1, 1, 1)
        else: cur = date(cur.year, cur.month+1, 1)

    new_dfs = []
    for ym in months:
        df = fetch_bn_spot_monthly(sym, ym)
        if df is not None and len(df) > 0:
            new_dfs.append(df)
        time.sleep(0.05)
    if not new_dfs:
        print(f"  {sym}: no pre-2025 SPOT data on Binance Vision; skip")
        return
    new = pd.concat(new_dfs, ignore_index=True)
    combined = pd.concat([new, existing], ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    combined.to_parquet(out_path, index=False)
    print(f"  {sym}: extended SPOT — added {len(new):,} rows, total {len(combined):,}")


def main():
    print("=== X60 extend historical data to 2023-01 ===\n")
    # Syms with longest history likely: BTC, ETH, SOL, ETH derivatives, etc.
    # Start with the fully-covered 50; extend only syms that exist on each venue pre-2025
    candidates = [
        "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT",
        "AVAXUSDT","LINKUSDT","DOTUSDT","ATOMUSDT","LTCUSDT","BCHUSDT","NEARUSDT",
        "UNIUSDT","TIAUSDT","SUIUSDT","SEIUSDT","INJUSDT","ARBUSDT","APTUSDT","OPUSDT",
        "AAVEUSDT","AXSUSDT","FILUSDT","ETCUSDT","TRBUSDT","WLDUSDT","ICPUSDT","ONDOUSDT",
        "PENDLEUSDT","LDOUSDT","JTOUSDT","ENAUSDT","HBARUSDT","TONUSDT","STRKUSDT",
        "WIFUSDT","ORDIUSDT","BIOUSDT","PENGUUSDT","ZECUSDT","RUNEUSDT","JUPUSDT","GMXUSDT",
        "AERO","FARTCOIN","TRUMP",  # newer; may not extend
    ]
    print(f"Candidates: {len(candidates)} syms\n")

    # === Phase 1: extend Binance SPOT (already in cache via Vision) ===
    print("--- Phase 1: extend BN SPOT 1h ---")
    for sym in candidates:
        try:
            extend_bn_spot(sym)
        except Exception as e:
            print(f"  {sym}: ERR {e}")
    print(f"\n--- Phase 1 complete ---\n")

    # Note: extending PERP klines, aggT, OKX, CB is more complex (different storage formats).
    # For initial test, focus on BN SPOT extension to verify the approach.
    # Once verified, add Phase 2/3/4 for other data sources.
    print("Note: BN PERP klines, aggT, OKX, CB extension not implemented in this iteration.")
    print("If BN SPOT extension works well, will extend other sources.")


if __name__ == "__main__":
    main()
