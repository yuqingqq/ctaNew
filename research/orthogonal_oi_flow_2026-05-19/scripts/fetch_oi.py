"""Stage-0 cheap fetch: Binance metrics (OI/long-short) for the 28 missing
symbols, full panel range. Caches data/ml/cache/metrics_<SYM>.parquet."""
import sys, time
from datetime import date
from pathlib import Path
sys.path.insert(0, "/home/yuqing/ctaNew")
from data_collectors.metrics_loader import fetch_metrics

MISS = Path("/home/yuqing/ctaNew/research/orthogonal_oi_flow_2026-05-19/results/_missing_oi.txt").read_text().split()
START, END = date(2025, 3, 23), date(2026, 5, 6)
t0 = time.time()
ok, empty = [], []
for i, s in enumerate(MISS, 1):
    try:
        df = fetch_metrics(s, START, END)
        n = len(df)
        (ok if n > 0 else empty).append(s)
        print(f"[{i}/{len(MISS)}] {s}: {n} rows ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        empty.append(s)
        print(f"[{i}/{len(MISS)}] {s}: ERROR {e}", flush=True)
print(f"\nOI fetch done [{time.time()-t0:.0f}s] ok={len(ok)} empty={len(empty)} "
      f"empty_syms={empty}", flush=True)
print("OI_FETCH_DONE", flush=True)
