"""Phase E3: Download 5-min klines + funding for 60 expansion candidates.

Uses existing binance_vision_loader (8 parallel workers per symbol).
Saves to data/ml/test/parquet/klines/{SYMBOL}/5m/ matching existing layout.
"""
from __future__ import annotations
import sys, time, logging
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_collectors.binance_vision_loader import LoaderConfig, fetch_klines
from data_collectors.funding_rate_loader import load_funding_rate

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(message)s")

START = date(2024, 12, 1)   # captures pre-backtest history for trailing windows
END = date(2026, 5, 10)
OUT_DIR = REPO / "data/ml/test"
FUNDING_DIR = REPO / "data/ml/cache"


def fetch_one_symbol(sym: str) -> dict:
    """Download klines + funding for one symbol. Returns metadata."""
    t0 = time.time()
    res = {"symbol": sym, "kline_rows": 0, "kline_days": 0, "funding_rows": 0,
            "status": "ok", "elapsed_s": 0.0}
    try:
        cfg = LoaderConfig(symbol=sym, out_dir=OUT_DIR, max_workers=6)
        df = fetch_klines(START, END, interval="5m", cfg=cfg)
        res["kline_rows"] = len(df)
        res["kline_days"] = df["open_time"].dt.normalize().nunique() if not df.empty else 0
    except Exception as e:
        res["status"] = f"kline_fail: {e}"

    try:
        fdf = load_funding_rate(sym, start_month="2024-12", end_month="2026-05")
        res["funding_rows"] = len(fdf) if fdf is not None else 0
    except Exception as e:
        res["status"] += f" | funding_fail: {e}"

    res["elapsed_s"] = time.time() - t0
    return res


def main():
    print(f"=== Phase E3: Download 5-min klines + funding for expansion candidates ===\n",
          flush=True)
    candidates = pd.read_csv(REPO / "outputs/vBTC_universe_expansion/final_candidates.csv")
    syms = candidates["symbol"].tolist()
    print(f"  Symbols to download: {len(syms)}", flush=True)
    print(f"  Date range: {START} → {END}", flush=True)
    print(f"  Out dir: {OUT_DIR}", flush=True)
    print(f"  Funding dir: {FUNDING_DIR}", flush=True)

    t_all = time.time()
    results = []
    for i, sym in enumerate(syms, 1):
        t0 = time.time()
        res = fetch_one_symbol(sym)
        results.append(res)
        elapsed = time.time() - t0
        total = time.time() - t_all
        print(f"  [{i:>2}/{len(syms)}] {sym:<16}  klines: {res['kline_rows']:>8,} rows / "
              f"{res['kline_days']:>3} days  funding: {res['funding_rows']:>5}  "
              f"({elapsed:.0f}s, total {total/60:.1f} min)", flush=True)

    print(f"\n  Total time: {(time.time()-t_all)/60:.1f} min", flush=True)
    pd.DataFrame(results).to_csv(REPO / "outputs/vBTC_universe_expansion/download_log.csv",
                                    index=False)


if __name__ == "__main__":
    main()
