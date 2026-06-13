"""Phase E4.5: build per-(symbol, date) trailing-30d median quote volume table.

For each symbol in expanded universe (51 existing + 60 new), compute:
  daily_quote_volume[sym, date] = sum of 5-min quote_volumes that day
  trailing_30d_median[sym, date] = rolling 30-day median of daily_quote_volume

This becomes a PIT-correct volume lookup: at any cycle t, the eligibility
check uses only data up to t-1 day.

Output: outputs/vBTC_features_expanded/volume_pit_table.parquet
       (columns: symbol, date, daily_qvol, trailing_30d_median_qvol)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_features_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_symbol_volumes(symbol: str) -> pd.DataFrame:
    """Compute daily and trailing-30d-median quote volume for one symbol."""
    sym_dir = KLINES_DIR / symbol / "5m"
    if not sym_dir.exists():
        return pd.DataFrame()
    files = sorted(sym_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    # Read only what we need
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["open_time", "quote_volume"])
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    klines = pd.concat(frames, ignore_index=True)
    klines["open_time"] = pd.to_datetime(klines["open_time"], utc=True, errors="coerce")
    klines = klines.dropna(subset=["open_time"])
    klines["date"] = klines["open_time"].dt.date
    # Sum quote_volume per day
    daily_qvol = klines.groupby("date")["quote_volume"].sum().sort_index()
    # Trailing 30-day median, min 15 days of data
    trailing_30d_med = daily_qvol.rolling(30, min_periods=15).median()
    out = pd.DataFrame({
        "symbol": symbol,
        "date": daily_qvol.index,
        "daily_qvol": daily_qvol.values,
        "trailing_30d_median_qvol": trailing_30d_med.values,
    })
    return out


def main():
    print(f"=== Phase E4.5: build PIT volume eligibility table ===\n", flush=True)

    # Symbol list: existing 51 panel + 60 new candidates
    existing = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                                  columns=["symbol"])
    existing_syms = sorted(existing["symbol"].unique())
    new_candidates = pd.read_csv(REPO / "outputs/vBTC_universe_expansion/final_candidates.csv")
    new_syms = sorted(new_candidates["symbol"].tolist())
    all_syms = sorted(set(existing_syms) | set(new_syms))
    print(f"  Total symbols: {len(all_syms)} (existing {len(existing_syms)} + new {len(new_syms)})",
          flush=True)

    print(f"\n  Computing daily volumes (parallel)...", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=8) as exe:
        futures = {exe.submit(compute_symbol_volumes, s): s for s in all_syms}
        for i, fut in enumerate(as_completed(futures), 1):
            s = futures[fut]
            try:
                df = fut.result()
                if not df.empty:
                    results.append(df)
            except Exception as e:
                print(f"    {s} failed: {e}", flush=True)
            if i % 25 == 0:
                print(f"    {i}/{len(all_syms)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    vol_table = pd.concat(results, ignore_index=True)
    vol_table["date"] = pd.to_datetime(vol_table["date"])
    print(f"  Volume table: {len(vol_table):,} rows, "
          f"{vol_table['symbol'].nunique()} symbols", flush=True)

    # Diagnostics
    print(f"\n  Trailing-30d-median quote volume distribution (across all rows):", flush=True)
    for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
        v = vol_table["trailing_30d_median_qvol"].quantile(q)
        print(f"    p{int(q*100):>2}: ${v:>15,.0f}", flush=True)

    # How many symbol-days pass $30M threshold?
    thresh = 30e6
    n_passing = (vol_table["trailing_30d_median_qvol"] >= thresh).sum()
    n_total = len(vol_table)
    print(f"\n  Symbol-days with trailing_30d_median >= ${thresh:,.0f}: "
          f"{n_passing:,} / {n_total:,} ({n_passing/n_total*100:.1f}%)", flush=True)

    # Per-symbol fraction of days passing threshold
    print(f"\n  Per-symbol % of days passing $30M threshold:", flush=True)
    pass_frac = vol_table.groupby("symbol").apply(
        lambda g: (g["trailing_30d_median_qvol"] >= thresh).mean() if len(g) > 0 else 0.0
    ).sort_values(ascending=False)
    print(f"    Distribution: p10={pass_frac.quantile(0.10):.0%}  p50={pass_frac.quantile(0.50):.0%}  "
          f"p90={pass_frac.quantile(0.90):.0%}", flush=True)

    print(f"\n  Top-10 symbols (most often eligible at $30M):", flush=True)
    for sym in pass_frac.head(10).index:
        print(f"    {sym:<14}  {pass_frac[sym]:.0%}", flush=True)
    print(f"\n  Bottom-10 symbols (least often eligible):", flush=True)
    for sym in pass_frac.tail(10).index:
        print(f"    {sym:<14}  {pass_frac[sym]:.0%}", flush=True)

    # Save
    out_path = OUT_DIR / "volume_pit_table.parquet"
    vol_table.to_parquet(out_path, compression="zstd", index=False)
    print(f"\n  saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
