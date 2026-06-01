"""Fetch 5m klines for the 106 HL-tradable perps that are ALSO on Binance USDM but NOT yet in the
HL70 panel — to expand the universe (iter-031: breadth = edge). 2021-01..2026-05; Binance Vision
handles per-symbol listing dates via 404s. Then funding + panel rebuild + retrain follow."""
from __future__ import annotations
import json, time
from datetime import date
from pathlib import Path
from data_collectors.binance_vision_loader import fetch_klines, LoaderConfig

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO/"data/ml/test"   # → data/ml/test/parquet/klines/{sym}/5m/{date}.parquet (pipeline-compatible)
ADDABLE = json.loads(Path("/tmp/addable.json").read_text())  # {HLname: BinanceSym}
SYMS = sorted(set(ADDABLE.values()))
START, END = date(2021,1,1), date(2026,5,11)

def main():
    t0=time.time()
    print(f"Fetching 5m klines {START}..{END} for {len(SYMS)} expansion symbols → {OUT}", flush=True)
    done=0
    for i,sym in enumerate(SYMS,1):
        try:
            df=fetch_klines(START, END, interval="5m", cfg=LoaderConfig(symbol=sym, out_dir=OUT, max_workers=8))
            n=len(df); rng=(df["open_time"].min(), df["open_time"].max()) if n else ("-","-")
            done+= (n>0)
            print(f"  [{i}/{len(SYMS)}] {sym:<16} {n:>8} rows {rng[0]}..{rng[1]} [{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(SYMS)}] {sym:<16} ERR {e}", flush=True)
    print(f"DONE expansion klines: {done}/{len(SYMS)} with data [{time.time()-t0:.0f}s]", flush=True)

if __name__=="__main__":
    main()
