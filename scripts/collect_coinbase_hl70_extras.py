"""Collect Coinbase 1h spot klines for missing HL-70 syms."""
from __future__ import annotations
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from data_collectors.coinbase_data_fetcher import CoinbaseDataFetcher

CACHE = REPO / "data/ml/cache"
START = datetime(2025, 3, 23, tzinfo=timezone.utc)
END = datetime(2026, 5, 7, tzinfo=timezone.utc)

# 24 missing syms (20 new HL + missing CB for some existing)
MISSING = [
    "AEROUSDT", "AIXBTUSDT", "ANIMEUSDT", "BERAUSDT", "FARTCOINUSDT",
    "GRIFFAINUSDT", "IPUSDT", "KAITOUSDT", "LAYERUSDT", "MELANIAUSDT",
    "MEUSDT", "MOVEUSDT", "NILUSDT", "PAXGUSDT", "SPXUSDT",
    "SUSDT", "TRUMPUSDT", "TSTUSDT", "USUALUSDT", "VINEUSDT",
]


def main():
    fetcher = CoinbaseDataFetcher()
    print(f"Collecting Coinbase for {len(MISSING)} HL-70 missing syms")
    for i, sym in enumerate(MISSING, 1):
        out_path = CACHE / f"cb_spot_{sym}_1h.parquet"
        if out_path.exists():
            print(f"  [{i}/{len(MISSING)}] {sym}: cached, skip")
            continue
        base = sym.replace("USDT", "")
        # Coinbase uses BASE-USD (not USDT)
        cb_sym = f"{base}-USD"
        print(f"  [{i}/{len(MISSING)}] {sym} ({cb_sym})...", flush=True)
        try:
            df = fetcher.fetch_backfill(cb_sym, start=START, end=END, granularity=3600)
            if df is None or len(df) == 0:
                print(f"    no data; skip")
                continue
            df.to_parquet(out_path, index=False)
            print(f"    saved {len(df):,} rows")
        except Exception as e:
            print(f"    ERR: {e}")


if __name__ == "__main__":
    main()
