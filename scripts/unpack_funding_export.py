"""Unpack the consolidated funding export into the per-symbol loader cache.

Run this on the (geo-blocked) backtest server after pulling the repo. It MERGES the
exported funding into data/ml/cache/funding_{sym}.parquet (concat + dedup on calc_time),
so any pre-existing history is preserved and June is added. After this, the normal
data_collectors.funding_rate_loader.load_funding_rate() cache-hit path serves it — no
Binance fetch needed.

Usage: python scripts/unpack_funding_export.py [--export PATH] [--cache data/ml/cache]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
COLS = ["calc_time", "interval_hours", "funding_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=str(REPO / "data/funding_export/funding_universe.parquet"))
    ap.add_argument("--cache", default=str(REPO / "data/ml/cache"))
    a = ap.parse_args()
    exp = pd.read_parquet(a.export)
    exp["calc_time"] = pd.to_datetime(exp["calc_time"], utc=True)
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    n_new = n_merged = 0
    for sym, g in exp.groupby("symbol"):
        g = g[COLS].sort_values("calc_time")
        p = cache / f"funding_{sym}.parquet"
        if p.exists():
            old = pd.read_parquet(p); old["calc_time"] = pd.to_datetime(old["calc_time"], utc=True)
            comb = (pd.concat([old[COLS], g], ignore_index=True)
                    .drop_duplicates("calc_time", keep="last").sort_values("calc_time"))
            n_merged += 1
        else:
            comb = g; n_new += 1
        comb.to_parquet(p, index=False)
    print(f"[unpack_funding_export] {exp.symbol.nunique()} syms | {n_merged} merged + {n_new} new "
          f"-> {cache} | export range {exp.calc_time.min()} -> {exp.calc_time.max()}")


if __name__ == "__main__":
    main()
