"""Collect OKX 1h klines for our 50-sym universe, backfill to 2025-04-01.

Saves to data/ml/cache/okx_klines_<SYM>_1h.parquet.
Resumable: skips symbols whose cache is already complete.
"""
from __future__ import annotations
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from data_collectors.okx_data_fetcher import OKXDataFetcher

CACHE = REPO / "data/ml/cache"; CACHE.mkdir(parents=True, exist_ok=True)
START = datetime(2025, 4, 1, tzinfo=timezone.utc)
END = datetime(2026, 5, 7, tzinfo=timezone.utc)
BAR = "1H"
RECOLLECT = False   # set True to overwrite caches


def main():
    f = OKXDataFetcher()
    inst = f.list_swap_instruments()
    okx_swaps = set(inst["instId"].tolist())

    # 50 HL-tradeable from 51-panel minus BTC (which is in 51-panel)
    p51 = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                          columns=["symbol"])
    syms51 = sorted(set(p51["symbol"].unique()) - {"BTCUSDT"})
    syms_with_okx = [s for s in syms51 if f.binance_sym_to_okx(s) in okx_swaps]
    print(f"target syms (overlap with OKX): {len(syms_with_okx)}")

    # also include BTCUSDT (for basis reference even if not in trading panel)
    syms_with_okx = ["BTCUSDT"] + syms_with_okx

    t0 = time.time()
    for i, sym in enumerate(syms_with_okx, 1):
        okx_sym = f.binance_sym_to_okx(sym)
        out_path = CACHE / f"okx_klines_{sym}_1h.parquet"
        if out_path.exists() and not RECOLLECT:
            existing = pd.read_parquet(out_path)
            print(f"[{i}/{len(syms_with_okx)}] {sym} cached ({len(existing)} rows, "
                  f"{existing['open_time'].min().date()} → "
                  f"{existing['open_time'].max().date()})", flush=True)
            continue
        print(f"[{i}/{len(syms_with_okx)}] fetching {sym} = {okx_sym}...", flush=True, end=" ")
        try:
            df = f.fetch_backfill(okx_sym, start=START, end=END, bar=BAR,
                                  limit_per_req=100, sleep_ms=70)
            if len(df) == 0:
                print("EMPTY")
                continue
            df.to_parquet(out_path, index=False)
            print(f"OK {len(df)} rows ({df['open_time'].min().date()} → "
                  f"{df['open_time'].max().date()}) [{time.time()-t0:.0f}s]",
                  flush=True)
        except Exception as e:
            print(f"ERR {e}")
            continue
    print(f"\nDONE [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
