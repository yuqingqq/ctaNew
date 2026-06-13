"""Collect OKX 1h klines for 22 missing HL-70 syms (20 extras + RUNE + VVV)."""
from __future__ import annotations
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from data_collectors.okx_data_fetcher import OKXDataFetcher

CACHE = REPO / "data/ml/cache"; CACHE.mkdir(parents=True, exist_ok=True)
START = datetime(2025, 3, 23, tzinfo=timezone.utc)
END = datetime(2026, 5, 7, tzinfo=timezone.utc)
BAR = "1H"

MISSING = [
    "AEROUSDT", "AIXBTUSDT", "ANIMEUSDT", "BERAUSDT", "FARTCOINUSDT",
    "GRIFFAINUSDT", "IPUSDT", "KAITOUSDT", "LAYERUSDT", "MELANIAUSDT",
    "MEUSDT", "MOVEUSDT", "NILUSDT", "PAXGUSDT", "RUNEUSDT", "SPXUSDT",
    "SUSDT", "TRUMPUSDT", "TSTUSDT", "USUALUSDT", "VINEUSDT", "VVVUSDT",
]


def main():
    fetcher = OKXDataFetcher()
    print(f"Collecting OKX for {len(MISSING)} HL-70 missing syms")
    for i, sym in enumerate(MISSING, 1):
        base = sym.replace("USDT", "")
        for inst_type, fname in [("SPOT", "okx_spot"), ("SWAP", "okx_swap")]:
            okx_sym = f"{base}-USDT" if inst_type == "SPOT" else f"{base}-USDT-SWAP"
            out_path = CACHE / f"{fname}_{sym}_1h.parquet"
            if out_path.exists():
                print(f"  [{i}/{len(MISSING)}] {sym} {inst_type}: cached, skip")
                continue
            print(f"  [{i}/{len(MISSING)}] {sym} {inst_type} ({okx_sym})...", flush=True)
            try:
                df = fetcher.fetch_backfill(okx_sym, start=START, end=END, bar=BAR,
                                              limit_per_req=100, sleep_ms=80)
                if df is None or len(df) == 0:
                    print(f"    no data; skip")
                    continue
                df.to_parquet(out_path, index=False)
                print(f"    saved {len(df):,} rows → {out_path.name}")
            except Exception as e:
                print(f"    ERR: {e}")


if __name__ == "__main__":
    main()
