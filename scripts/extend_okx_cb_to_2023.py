"""Extend OKX swap+spot and Coinbase 1h klines back to 2023-01 for V5 crossX.

Fetches 2023-01 → 2025-03-23 (the pre-period missing from existing caches) and
merges with existing recent data. Saves to same cache paths.
"""
from __future__ import annotations
import sys, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from data_collectors.okx_data_fetcher import OKXDataFetcher
from data_collectors.coinbase_data_fetcher import CoinbaseDataFetcher

CACHE = REPO/"data/ml/cache"
START = datetime(2023,1,1,tzinfo=timezone.utc)
END = datetime(2025,3,24,tzinfo=timezone.utc)

CANDIDATES = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT",
    "AVAXUSDT","LINKUSDT","DOTUSDT","ATOMUSDT","LTCUSDT","BCHUSDT","NEARUSDT",
    "UNIUSDT","TIAUSDT","SUIUSDT","SEIUSDT","INJUSDT","ARBUSDT","APTUSDT","OPUSDT",
    "AAVEUSDT","AXSUSDT","FILUSDT","ETCUSDT","TRBUSDT","WLDUSDT","ICPUSDT","ONDOUSDT",
    "PENDLEUSDT","LDOUSDT","JTOUSDT","ENAUSDT","HBARUSDT","TONUSDT","STRKUSDT",
    "WIFUSDT","ORDIUSDT","JUPUSDT","GMXUSDT","TAOUSDT","RUNEUSDT","SUSDT","ZECUSDT",
]


def extend_file(path, fetch_fn):
    """Fetch pre-period, merge with existing if present."""
    if path.exists():
        existing = pd.read_parquet(path)
        existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True)
        if existing["open_time"].min() <= pd.Timestamp("2023-02-01", tz="UTC"):
            return "already-extended", len(existing)
    else:
        existing = None
    try:
        new = fetch_fn()
    except Exception as e:
        return f"ERR {e}", 0
    if new is None or len(new) == 0:
        return "no-data", 0
    new["open_time"] = pd.to_datetime(new["open_time"], utc=True)
    if existing is not None:
        combined = pd.concat([new, existing], ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    else:
        combined = new.sort_values("open_time")
    combined.to_parquet(path, index=False)
    return "extended", len(combined)


def main():
    okx = OKXDataFetcher(); cb = CoinbaseDataFetcher()
    print(f"Extending OKX+CB to 2023-01 for {len(CANDIDATES)} syms\n", flush=True)
    for i, sym in enumerate(CANDIDATES, 1):
        base = sym.replace("USDT","")
        # OKX swap
        r1, n1 = extend_file(CACHE/f"okx_swap_{sym}_1h.parquet",
            lambda: okx.fetch_backfill(f"{base}-USDT-SWAP", start=START, end=END, bar="1H", limit_per_req=100, sleep_ms=60))
        # OKX spot
        r2, n2 = extend_file(CACHE/f"okx_spot_{sym}_1h.parquet",
            lambda: okx.fetch_backfill(f"{base}-USDT", start=START, end=END, bar="1H", limit_per_req=100, sleep_ms=60))
        # Coinbase spot
        r3, n3 = extend_file(CACHE/f"cb_spot_{sym}_1h.parquet",
            lambda: cb.fetch_backfill(f"{base}-USD", start=START, end=END, granularity=3600, sleep_ms=120))
        print(f"[{i}/{len(CANDIDATES)}] {sym}: okx_swap={r1}({n1}) okx_spot={r2}({n2}) cb={r3}({n3})", flush=True)
    print("\nDone OKX+CB extension.", flush=True)


if __name__ == "__main__":
    main()
