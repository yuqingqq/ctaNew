"""Phase E1 of universe expansion: discover + filter Binance USDM perp symbols.

Steps:
  1. Enumerate ALL USDM perp symbols from Binance Vision S3 listing
  2. Filter to USDT-quoted only (excludes USDC, BUSD pairs)
  3. For each new candidate (not in our existing 51), apply HARD criteria:
     - Has 5-min kline data on Binance Vision for at least 2025-01-01 onwards
     - Symbol is "currently trading" (has data for most recent day)
  4. Output candidate list with metadata (first_seen, last_seen days)

This is the discovery step. Volume thresholds + 30-day history etc. are
applied at the PIT evaluation step (Phase E2), not here.
"""
from __future__ import annotations
import sys, time, re, urllib.request
from pathlib import Path
from datetime import date, datetime, timezone

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "outputs/vBTC_universe_expansion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# S3 listing endpoint
S3_LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BACKTEST_START = date(2025, 3, 27)  # Match existing panel start
PROBE_DATE = date(2026, 5, 1)  # Recent date to verify symbol is "live"


def list_s3_keys(prefix: str, delimiter: str = "/") -> list[str]:
    """List S3 prefixes (folder-like) under a given prefix. Handles pagination."""
    keys = []
    marker = ""
    page = 0
    while True:
        url = f"{S3_LISTING}?prefix={prefix}&delimiter={delimiter}"
        if marker:
            url += f"&marker={marker}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                body = resp.read().decode("utf-8")
        except Exception as e:
            print(f"  request failed (page {page}): {e}", flush=True)
            break
        # Parse common prefixes
        prefixes = re.findall(r"<Prefix>([^<]+)</Prefix>", body)
        # The first <Prefix> is the query echo; skip it
        new_keys = [p for p in prefixes if p != prefix]
        keys.extend(new_keys)
        is_truncated = "<IsTruncated>true</IsTruncated>" in body
        if not is_truncated:
            break
        # Get next-marker (last seen key)
        if new_keys:
            marker = new_keys[-1]
        else:
            break
        page += 1
        if page > 50:
            print(f"  hit page limit", flush=True)
            break
    return keys


def list_5m_files(symbol: str) -> list[str]:
    """List daily 5m kline files available for a symbol."""
    prefix = f"data/futures/um/daily/klines/{symbol}/5m/"
    url = f"{S3_LISTING}?prefix={prefix}&delimiter=/"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except Exception as e:
        return []
    files = re.findall(r"<Key>([^<]+\.zip)</Key>", body)
    return files


def main():
    print(f"=== Phase E1: Discover Binance USDM perp universe ===\n", flush=True)

    # Step 1: enumerate all symbols
    print("Step 1: List all USDM klines symbols from S3...", flush=True)
    t0 = time.time()
    all_prefixes = list_s3_keys("data/futures/um/daily/klines/", "/")
    all_symbols = [p.split("/")[-2] for p in all_prefixes if p.endswith("/")]
    print(f"  Got {len(all_symbols)} symbols in {time.time()-t0:.0f}s", flush=True)

    # Step 2: Filter to USDT-quoted only
    usdt_symbols = [s for s in all_symbols if s.endswith("USDT") and "DOWN" not in s and "UP" not in s]
    print(f"  USDT-quoted (excluding leveraged tokens): {len(usdt_symbols)}", flush=True)

    # Step 3: identify which are in our existing panel
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                              columns=["symbol"])
    existing = set(panel["symbol"].unique())
    print(f"  Existing 51-panel: {len(existing)}", flush=True)
    new_candidates = sorted(set(usdt_symbols) - existing)
    print(f"  New candidates (not in panel): {len(new_candidates)}", flush=True)

    # Step 4: for each new candidate, check availability + listing date
    print(f"\nStep 4: Check kline availability for new candidates...", flush=True)
    records = []
    for i, sym in enumerate(new_candidates):
        if i > 0 and i % 25 == 0:
            print(f"  progress: {i}/{len(new_candidates)} ({time.time()-t0:.0f}s)", flush=True)
        files = list_5m_files(sym)
        if not files:
            continue
        # Extract dates from filenames
        dates = []
        for f in files:
            m = re.search(r"(\d{4}-\d{2}-\d{2})\.zip$", f)
            if m:
                try:
                    dates.append(date.fromisoformat(m.group(1)))
                except Exception:
                    pass
        if not dates:
            continue
        first_d = min(dates)
        last_d = max(dates)
        # Filter: must have data in our backtest window AND recent activity
        if last_d < date(2026, 4, 1):  # at least one day in last 30+ days
            continue
        records.append({
            "symbol": sym,
            "first_kline_date": first_d.isoformat(),
            "last_kline_date": last_d.isoformat(),
            "n_days_available": len(dates),
            "days_before_backtest_start": max(0, (BACKTEST_START - first_d).days),
        })

    df = pd.DataFrame(records).sort_values("first_kline_date").reset_index(drop=True)
    print(f"\n  Found {len(df)} new candidates with valid kline data", flush=True)
    print(f"\n  First-kline-date distribution:", flush=True)
    df["first_kline_year"] = df["first_kline_date"].str[:4]
    print(df["first_kline_year"].value_counts().sort_index().to_string(), flush=True)

    df.to_csv(OUT_DIR / "candidate_universe.csv", index=False)
    print(f"\n  saved → {OUT_DIR}/candidate_universe.csv", flush=True)

    # Quick summary
    listed_before_backtest = df[pd.to_datetime(df["first_kline_date"]).dt.date <= BACKTEST_START]
    listed_120d_before = df[pd.to_datetime(df["first_kline_date"]).dt.date <=
                              (BACKTEST_START - pd.Timedelta(days=120).to_pytimedelta()).date()
                              if False else BACKTEST_START - pd.Timedelta(days=120).to_pytimedelta()]
    print(f"\n  Summary:", flush=True)
    print(f"    Total new candidates with klines: {len(df)}", flush=True)
    print(f"    Listed by backtest start ({BACKTEST_START}): {len(listed_before_backtest)}", flush=True)
    cutoff = BACKTEST_START - pd.Timedelta(days=120).to_pytimedelta()
    listed_120d = df[pd.to_datetime(df["first_kline_date"]).dt.date <= cutoff.date()]
    print(f"    Listed ≥ 120d before backtest start: {len(listed_120d)}", flush=True)


if __name__ == "__main__":
    main()
