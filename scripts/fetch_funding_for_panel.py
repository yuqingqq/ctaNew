"""Fetch funding rate data for all 51 symbols in the vBTC panel.

Force-rebuilds cache to:
  1. Add 6 missing symbols (ASTER, AXS, GMX, ICP, ORDI, TAO)
  2. Extend through April 2026 for all (existing data ends 2026-03)
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_collectors.funding_rate_loader import load_funding_for_universe

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants.parquet"


def main():
    print(f"Loading panel to extract symbols...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    syms = sorted(panel["symbol"].unique().tolist())
    print(f"  {len(syms)} symbols: {syms}", flush=True)

    print(f"\nFetching funding rates 2025-01 to 2026-04 (force rebuild)...", flush=True)
    t0 = time.time()
    out = load_funding_for_universe(syms, start_month="2025-01", end_month="2026-04",
                                      force_rebuild=True)
    print(f"\nDone in {time.time()-t0:.0f}s", flush=True)

    print(f"\nResult summary:", flush=True)
    print(f"  {'symbol':<14} {'rows':>8}  {'date_min':<26}  {'date_max':<26}", flush=True)
    rows_total = 0
    no_data = []
    for s in syms:
        df = out.get(s, pd.DataFrame())
        if df.empty:
            no_data.append(s)
            print(f"  {s:<14} {'EMPTY':>8}  {'-':<26}  {'-':<26}", flush=True)
            continue
        rows_total += len(df)
        dt_min = str(df["calc_time"].min())
        dt_max = str(df["calc_time"].max())
        print(f"  {s:<14} {len(df):>8,}  {dt_min:<26}  {dt_max:<26}", flush=True)

    print(f"\n  Total funding rows: {rows_total:,}", flush=True)
    if no_data:
        print(f"  Symbols with no funding data: {no_data}", flush=True)


if __name__ == "__main__":
    main()
