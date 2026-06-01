"""Fetch Binance futures METRICS (open interest + long/short ratios, free Vision archive) for the
23-sym EXT multi-episode universe + BTC, 2021-2026 — to test whether market-wide POSITIONING/leverage
fragility LEADS the correlated alt selloffs (the leading-signal the coincident price/vol data lacked)."""
from __future__ import annotations
import json, time
from datetime import date
from pathlib import Path
from data_collectors.metrics_loader import fetch_metrics

REPO = Path("/home/yuqing/ctaNew")
SYMS = json.loads(Path("/tmp/pre23_syms.json").read_text()) + ["BTCUSDT"]  # 23 EXT alts + BTC
START, END = date(2021,1,1), date(2026,5,11)

def main():
    t0=time.time()
    print(f"Fetching metrics (OI + long/short) {START}..{END} for {len(SYMS)} syms", flush=True)
    for i,sym in enumerate(SYMS,1):
        try:
            df=fetch_metrics(sym, START, END)
            n=len(df) if df is not None else 0
            rng=(df['create_time'].min(), df['create_time'].max()) if n else ('-','-')
            print(f"  [{i}/{len(SYMS)}] {sym:<12} {n:>8} rows  {rng[0]}..{rng[1]}  [{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(SYMS)}] {sym:<12} ERR {e}", flush=True)
    print(f"DONE metrics [{time.time()-t0:.0f}s] → data/ml/cache/metrics_<SYM>.parquet", flush=True)

if __name__=="__main__":
    main()
