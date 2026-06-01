"""Fetch 2021-2022 5m klines (+ funding) for the 23 pre-2023 symbols, to extend the panel
back so the sign-flip predictor has multiple regime flips to learn from (task #109)."""
from __future__ import annotations
import json, time, sys
from datetime import date
from pathlib import Path
from data_collectors.binance_vision_loader import fetch_klines, LoaderConfig

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO/"data/ml/test"   # → data/ml/test/parquet/klines/{sym}/5m/{date}.parquet (matches pipeline)
SYMS = json.loads(Path("/tmp/pre23_syms.json").read_text()) + ["BTCUSDT"]
START, END = date(2021,1,1), date(2022,12,31)

def main():
    t0=time.time()
    print(f"Fetching 5m klines {START}..{END} for {len(SYMS)} syms → {OUT}", flush=True)
    for i,sym in enumerate(SYMS,1):
        try:
            df=fetch_klines(START, END, interval="5m", cfg=LoaderConfig(symbol=sym, out_dir=OUT, max_workers=6))
            n=len(df); rng=(df["open_time"].min(), df["open_time"].max()) if n else ("-","-")
            print(f"  [{i}/{len(SYMS)}] {sym:<12} {n:>8} rows  {rng[0]}..{rng[1]}  [{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(SYMS)}] {sym:<12} ERROR {e}", flush=True)
    print(f"DONE klines [{time.time()-t0:.0f}s]", flush=True)

if __name__=="__main__":
    main()
