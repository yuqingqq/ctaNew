"""Fetch funding-rate history from Binance FAPI for the live universe and write ONE
consolidated parquet, for transport to a geo-blocked backtest server.

The normal path (data_collectors/funding_rate_loader) pulls MONTHLY Vision archives,
which are not published intra-month — so June funding for the non-traded peers isn't
available there, and the live collector only funds the ~94 traded symbols. This pulls
the authoritative FAPI fundingRate endpoint for ALL universe symbols over a window that
covers June + a trailing buffer (for the 7d funding z-score), uniform-source.

Output schema matches the loader cache (calc_time, interval_hours, funding_rate) plus a
`symbol` column. Run on a box that can reach Binance; then commit the parquet and run
scripts/unpack_funding_export.py on the backtest server to merge it into data/ml/cache.

Usage: python scripts/fetch_funding_fapi.py [--start 2026-05-01] [--out PATH]
"""
from __future__ import annotations
import argparse, json, time, urllib.request
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"
THROTTLE = 0.45          # ≥0.4s between FAPI calls (ban-safe, per the live-box convention)


def fetch_symbol(sym: str, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    rows, cur = [], start_ms
    while cur < end_ms:
        url = f"{FAPI}?symbol={sym}&startTime={cur}&endTime={end_ms}&limit=1000"
        page = json.load(urllib.request.urlopen(url, timeout=20))
        if not page:
            break
        rows.extend(page)
        last = int(page[-1]["fundingTime"])
        if len(page) < 1000 or last <= cur:
            break
        cur = last + 1
        time.sleep(THROTTLE)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["calc_time"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True).round("h")
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df.drop_duplicates("calc_time").sort_values("calc_time")
    iv = (df["calc_time"].diff().dt.total_seconds() / 3600).round()   # interval = gap to prior settlement
    iv = iv.bfill().fillna(8).astype(int)                              # first row: next interval; default 8h
    df["interval_hours"] = iv
    df["symbol"] = sym
    return df[["symbol", "calc_time", "interval_hours", "funding_rate"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--universe", default=str(REPO / "live/collector_universe.txt"))
    ap.add_argument("--out", default=str(REPO / "data/funding_export/funding_universe.parquet"))
    a = ap.parse_args()
    syms = Path(a.universe).read_text().split()
    start_ms = int(pd.Timestamp(a.start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    parts, fails = [], []
    for i, s in enumerate(syms):
        try:
            df = fetch_symbol(s, start_ms, end_ms)
            if df is not None and len(df):
                parts.append(df)
        except Exception as e:
            fails.append((s, str(e)[:60]))
        time.sleep(THROTTLE)
        if i % 30 == 0:
            print(f"  {i}/{len(syms)} ...", flush=True)
    res = pd.concat(parts, ignore_index=True).sort_values(["symbol", "calc_time"])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out, index=False)
    print(f"[fetch_funding_fapi] {len(res)} rows, {res.symbol.nunique()} syms -> {out}")
    print(f"  range {res.calc_time.min()} -> {res.calc_time.max()} | fails {len(fails)}: {fails[:6]}")


if __name__ == "__main__":
    main()
