"""Collect Coinbase Spot 1h klines for our 50-sym universe, 2025-04 → 2026-05.

Saves to data/ml/cache/cb_spot_<SYM>_1h.parquet.
Resumable: skips already-cached files.
"""
from __future__ import annotations
import sys, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from data_collectors.coinbase_data_fetcher import CoinbaseDataFetcher

CACHE = REPO / "data/ml/cache"; CACHE.mkdir(parents=True, exist_ok=True)
START = datetime(2025, 4, 1, tzinfo=timezone.utc)
END = datetime(2026, 5, 7, tzinfo=timezone.utc)
GRANULARITY = 3600   # 1h
RECOLLECT = False


def main():
    f = CoinbaseDataFetcher()
    products = f.list_products()
    cb_usd_syms = set(products[(products["quote_currency"] == "USD")
                                & (products["status"] == "online")]
                      ["base_currency"].tolist())

    p51 = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                          columns=["symbol"])
    syms = ["BTCUSDT"] + sorted(set(p51["symbol"].unique()) - {"BTCUSDT"})

    t0 = time.time()
    fetched = cached = skipped = 0
    for i, sym in enumerate(syms, 1):
        base = sym[:-4]
        cb_id = f"{base}-USD"
        if base not in cb_usd_syms:
            print(f"[{i}/{len(syms)}] {sym} not on Coinbase USD", flush=True)
            skipped += 1
            continue
        out_path = CACHE / f"cb_spot_{sym}_1h.parquet"
        if out_path.exists() and not RECOLLECT:
            ex = pd.read_parquet(out_path)
            print(f"[{i}/{len(syms)}] {sym} cached ({len(ex)} rows, "
                  f"{ex['open_time'].min().date()}→{ex['open_time'].max().date()})",
                  flush=True)
            cached += 1
            continue
        print(f"[{i}/{len(syms)}] fetching {sym}={cb_id}...", flush=True, end=" ")
        try:
            df = f.fetch_backfill(cb_id, start=START, end=END,
                                  granularity=GRANULARITY, sleep_ms=120)
            if len(df) == 0:
                print("EMPTY", flush=True)
                continue
            df.to_parquet(out_path, index=False)
            print(f"OK {len(df)} rows ({df['open_time'].min().date()}→"
                  f"{df['open_time'].max().date()}) [{time.time()-t0:.0f}s]",
                  flush=True)
            fetched += 1
        except Exception as e:
            print(f"ERR {e}", flush=True)

    print(f"\n=== Coinbase Spot collection summary ===")
    print(f"  fetched: {fetched}, cached: {cached}, skipped: {skipped}")
    print(f"  elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
