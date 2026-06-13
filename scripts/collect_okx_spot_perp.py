"""Collect OKX 1h klines for both SPOT and SWAP (perp) over our 50-sym universe.

Saves:
  data/ml/cache/okx_spot_<SYM>_1h.parquet   ← OKX SPOT (BASE-USDT)
  data/ml/cache/okx_swap_<SYM>_1h.parquet   ← OKX SWAP (BASE-USDT-SWAP)

Coverage from earlier check:
  - 47 syms have both spot + swap
  - TAO: swap only
  - RUNE, VVV: neither (Binance-only listings)

Backfill window: 2025-04-01 → 2026-05-07 (matches the 51-panel sample).
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
RECOLLECT = False


def fetch_one(f, sym, kind, okx_id):
    """kind in {'spot','swap'}."""
    out_path = CACHE / f"okx_{kind}_{sym}_1h.parquet"
    if out_path.exists() and not RECOLLECT:
        existing = pd.read_parquet(out_path)
        return False, len(existing), existing
    try:
        df = f.fetch_backfill(okx_id, start=START, end=END, bar=BAR,
                              limit_per_req=100, sleep_ms=80)
        if len(df) == 0:
            return False, 0, None
        df.to_parquet(out_path, index=False)
        return True, len(df), df
    except Exception as e:
        return False, 0, None


def main():
    f = OKXDataFetcher()

    # Get instrument universes
    swap_inst = f.list_swap_instruments()
    okx_swap_ids = set(swap_inst["instId"].tolist())

    import requests
    r = requests.get("https://www.okx.com/api/v5/public/instruments?instType=SPOT", timeout=30)
    spot_df = pd.DataFrame(r.json()["data"])
    okx_spot_ids = set(spot_df[spot_df["quoteCcy"] == "USDT"]["instId"].tolist())

    # 50 HL-tradeable syms (51-panel minus BTC), include BTC at top
    p51 = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                          columns=["symbol"])
    syms = ["BTCUSDT"] + sorted(set(p51["symbol"].unique()) - {"BTCUSDT"})

    t0 = time.time()
    swap_fetched = swap_cached = swap_skipped = 0
    spot_fetched = spot_cached = spot_skipped = 0
    for i, sym in enumerate(syms, 1):
        base = sym[:-4]
        swap_id = f"{base}-USDT-SWAP"
        spot_id = f"{base}-USDT"

        # SWAP
        if swap_id in okx_swap_ids:
            out_path = CACHE / f"okx_swap_{sym}_1h.parquet"
            if out_path.exists() and not RECOLLECT:
                ex = pd.read_parquet(out_path)
                print(f"[{i}/{len(syms)}] {sym} swap cached ({len(ex)} rows, "
                      f"{ex['open_time'].min().date()}→{ex['open_time'].max().date()})",
                      flush=True)
                swap_cached += 1
            else:
                print(f"[{i}/{len(syms)}] fetching swap {sym}={swap_id}...",
                      flush=True, end=" ")
                ok, n, df = fetch_one(f, sym, "swap", swap_id)
                if ok:
                    print(f"OK {n} rows ({df['open_time'].min().date()}→"
                          f"{df['open_time'].max().date()}) [{time.time()-t0:.0f}s]",
                          flush=True)
                    swap_fetched += 1
                else:
                    print(f"FAIL", flush=True)
        else:
            swap_skipped += 1

        # SPOT
        if spot_id in okx_spot_ids:
            out_path = CACHE / f"okx_spot_{sym}_1h.parquet"
            if out_path.exists() and not RECOLLECT:
                ex = pd.read_parquet(out_path)
                print(f"[{i}/{len(syms)}] {sym} spot cached ({len(ex)} rows)",
                      flush=True)
                spot_cached += 1
            else:
                print(f"[{i}/{len(syms)}] fetching spot {sym}={spot_id}...",
                      flush=True, end=" ")
                ok, n, df = fetch_one(f, sym, "spot", spot_id)
                if ok:
                    print(f"OK {n} rows ({df['open_time'].min().date()}→"
                          f"{df['open_time'].max().date()}) [{time.time()-t0:.0f}s]",
                          flush=True)
                    spot_fetched += 1
                else:
                    print(f"FAIL", flush=True)
        else:
            spot_skipped += 1

    print(f"\n=== Collection summary ===")
    print(f"  swap:  fetched={swap_fetched}, cached={swap_cached}, skipped (not on OKX)={swap_skipped}")
    print(f"  spot:  fetched={spot_fetched}, cached={spot_cached}, skipped (not on OKX)={spot_skipped}")
    print(f"  elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
