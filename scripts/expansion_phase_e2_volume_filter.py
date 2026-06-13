"""Phase E2: Volume filter for T1 candidates.

For each T1 candidate (listed by 2025-03-27, still active), download monthly
1d klines covering the backtest window, compute daily quote volume time series,
apply minimum trailing-30d-median threshold.

Output: filtered list of survivors with volume metadata.

This is much faster than downloading 5-min data first because monthly 1d files
are tiny (~30 candles each, ~1 KB compressed).
"""
from __future__ import annotations
import sys, time, io, zipfile, urllib.request, urllib.error
from pathlib import Path
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT_DIR = REPO / "outputs/vBTC_universe_expansion"

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
MONTHS = [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 6)]

# Volume filter thresholds (PIT, applied per-cycle later)
MIN_30D_MEDIAN_USD = 30_000_000  # $30M daily median quote volume
KEEP_TOP_N = 60  # cap on final candidates we'll download 5-min for

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "count",
              "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def fetch_monthly_1d(symbol: str, year: int, month: int) -> pd.DataFrame:
    url = f"{BASE_URL}/{symbol}/1d/{symbol}-1d-{year}-{month:02d}.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return pd.DataFrame()
        raise
    except Exception:
        return pd.DataFrame()
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            fname = zf.namelist()[0]
            with zf.open(fname) as f:
                df = pd.read_csv(f, names=KLINE_COLS, header=None)
    except Exception:
        return pd.DataFrame()
    # First row might be a header row (newer files have headers); detect
    if df.iloc[0]["open_time"] == "open_time":
        df = df.iloc[1:].reset_index(drop=True)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["quote_volume"] = pd.to_numeric(df["quote_volume"], errors="coerce")
    df = df.dropna(subset=["open_time", "quote_volume"])
    if df["open_time"].iloc[0] > 1e15:  # microseconds
        df["open_time"] = pd.to_datetime(df["open_time"], unit="us", utc=True)
    else:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["open_time", "quote_volume"]]


def fetch_symbol_volumes(symbol: str) -> pd.DataFrame:
    frames = []
    for y, m in MONTHS:
        df = fetch_monthly_1d(symbol, y, m)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("open_time")


def main():
    print(f"=== Phase E2: Volume filter for T1 candidates ===\n", flush=True)
    candidates = pd.read_csv(OUT_DIR / "candidate_universe.csv")
    candidates["first_kline_date"] = pd.to_datetime(candidates["first_kline_date"]).dt.date
    candidates["last_kline_date"] = pd.to_datetime(candidates["last_kline_date"]).dt.date

    BACKTEST_START = date(2025, 3, 27)
    LATEST = date(2026, 4, 15)
    t1 = candidates[(candidates["first_kline_date"] <= BACKTEST_START) &
                       (candidates["last_kline_date"] >= LATEST)]
    syms = t1["symbol"].tolist()
    print(f"  T1 (full-window) candidates: {len(syms)}", flush=True)

    print(f"\nFetching monthly 1d klines for {len(syms)} symbols × {len(MONTHS)} months...",
          flush=True)
    t0 = time.time()
    sym_to_df = {}
    with ThreadPoolExecutor(max_workers=12) as exe:
        futures = {exe.submit(fetch_symbol_volumes, s): s for s in syms}
        done = 0
        for fut in as_completed(futures):
            s = futures[fut]
            df = fut.result()
            sym_to_df[s] = df
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(syms)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  Fetched in {time.time()-t0:.0f}s", flush=True)

    # Compute trailing-30d median quote volume per symbol per day
    print(f"\nComputing volume metrics...", flush=True)
    rows = []
    for sym, df in sym_to_df.items():
        if df.empty or len(df) < 30:
            continue
        df = df.sort_values("open_time").copy()
        df["med30d"] = df["quote_volume"].rolling(30, min_periods=15).median()
        # Use median across full available period AND the peak 30d-median as a proxy
        rows.append({
            "symbol": sym,
            "n_days": len(df),
            "mean_daily_qvol": float(df["quote_volume"].mean()),
            "median_daily_qvol": float(df["quote_volume"].median()),
            "max_30d_median": float(df["med30d"].max()),
            "min_30d_median": float(df["med30d"].min()),
            "last_30d_median": float(df["med30d"].iloc[-1]) if pd.notna(df["med30d"].iloc[-1]) else 0.0,
        })

    vol_df = pd.DataFrame(rows).sort_values("median_daily_qvol", ascending=False)
    print(f"  Got volume stats for {len(vol_df)} symbols", flush=True)

    # Apply filter: keep symbols where median daily qvol >= MIN_30D_MEDIAN_USD
    # at some point in the backtest window (max_30d_median >= threshold)
    print(f"\n  Distribution of max_30d_median quote volume (USD):", flush=True)
    print(f"    p10:  ${vol_df['max_30d_median'].quantile(0.10):>15,.0f}", flush=True)
    print(f"    p25:  ${vol_df['max_30d_median'].quantile(0.25):>15,.0f}", flush=True)
    print(f"    p50:  ${vol_df['max_30d_median'].quantile(0.50):>15,.0f}", flush=True)
    print(f"    p75:  ${vol_df['max_30d_median'].quantile(0.75):>15,.0f}", flush=True)
    print(f"    p90:  ${vol_df['max_30d_median'].quantile(0.90):>15,.0f}", flush=True)
    print(f"    max:  ${vol_df['max_30d_median'].max():>15,.0f}", flush=True)

    survivors = vol_df[vol_df["max_30d_median"] >= MIN_30D_MEDIAN_USD].copy()
    print(f"\n  Symbols with max_30d_median >= ${MIN_30D_MEDIAN_USD:,}: {len(survivors)}",
          flush=True)

    # Keep top N by max_30d_median (cap final set)
    final = survivors.head(KEEP_TOP_N).copy()
    print(f"\n=== Final candidate list (top {len(final)} by max_30d_median) ===", flush=True)
    print(f"  {'symbol':<14}  {'mean_qvol_M':>12}  {'med_qvol_M':>12}  "
          f"{'max_30d_M':>10}  {'last_30d_M':>10}", flush=True)
    for _, r in final.iterrows():
        print(f"  {r['symbol']:<14}  ${r['mean_daily_qvol']/1e6:>11.1f}M  "
              f"${r['median_daily_qvol']/1e6:>11.1f}M  "
              f"${r['max_30d_median']/1e6:>9.1f}M  "
              f"${r['last_30d_median']/1e6:>9.1f}M", flush=True)

    vol_df.to_csv(OUT_DIR / "t1_volume_stats.csv", index=False)
    final.to_csv(OUT_DIR / "final_candidates.csv", index=False)
    print(f"\n  saved → {OUT_DIR}/final_candidates.csv ({len(final)} symbols)", flush=True)


if __name__ == "__main__":
    main()
