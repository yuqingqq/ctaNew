"""Bulk-download Binance USDM 1m klines (all USDT perps) from Binance Vision.

Calibration/validation corpus for the detector:
  monthly zips for --months (default 2026-08) + daily zips for --days-from..--days-to.
Output: data/hist/usdm_1m/<SYM>.parquet (float32 OHLC + quote volume + taker-buy
quote + trades), one file per symbol spanning the whole requested range.

A symbol with no file in the range is recorded as a status (not silently dropped)
in data/hist/usdm_1m/_manifest.json.

Run (light, network-bound):
  .worktrees/research_slot.sh 3 -- python3 -m research.download_usdm_1m
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hist" / "usdm_1m"
BASE = "https://data.binance.vision/data/futures/um"
SESSION = requests.Session()


def _fetch_zip(url: str) -> pd.DataFrame | None:
    for k in range(4):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = z.read(z.namelist()[0]).decode()
            lines = [ln for ln in raw.splitlines() if ln and ln[0].isdigit()]
            if not lines:
                return None
            a = np.array([ln.split(",")[:11] for ln in lines])
            df = pd.DataFrame({
                "ts": pd.to_datetime(a[:, 0].astype("int64"), unit="ms", utc=True),
                "open": a[:, 1].astype("float32"), "high": a[:, 2].astype("float32"),
                "low": a[:, 3].astype("float32"), "close": a[:, 4].astype("float32"),
                "quote_volume": a[:, 7].astype("float32"),
                "trades": a[:, 8].astype("int32"),
                "taker_buy_quote": a[:, 10].astype("float32"),
            })
            return df
        except (requests.RequestException, zipfile.BadZipFile) as e:
            time.sleep(1.5 * (k + 1))
            last = e
    raise RuntimeError(f"failed {url}: {last}")


def one_symbol(sym: str, months: list[str], days: list[date]) -> tuple[str, str, int]:
    path = OUT / f"{sym}.parquet"   # OUT may be re-pointed by --out-subdir
    if path.exists():
        return sym, "cached", -1
    parts = []
    for m in months:
        df = _fetch_zip(f"{BASE}/monthly/klines/{sym}/1m/{sym}-1m-{m}.zip")
        if df is not None:
            parts.append(df)
    for d in days:
        df = _fetch_zip(f"{BASE}/daily/klines/{sym}/1m/{sym}-1m-{d:%Y-%m-%d}.zip")
        if df is not None:
            parts.append(df)
    if not parts:
        return sym, "no_data_in_range", 0
    df = pd.concat(parts).drop_duplicates("ts").sort_values("ts").set_index("ts")
    df.to_parquet(path)
    return sym, "ok", len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="2026-08")
    ap.add_argument("--days-from", default="2026-09-01")
    ap.add_argument("--days-to", default="2026-09-27")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--symbols-file", default=str(ROOT / "data" / "usdm_symbol_dirs.txt"))
    ap.add_argument("--out-subdir", default="usdm_1m", help="data/hist/<subdir> (e.g. usdm_1m_holdout)")
    args = ap.parse_args()

    global OUT
    OUT = ROOT / "data" / "hist" / args.out_subdir
    OUT.mkdir(parents=True, exist_ok=True)
    syms = [s.strip() for s in open(args.symbols_file) if s.strip()]
    months = [m for m in args.months.split(",") if m]
    d0, d1 = date.fromisoformat(args.days_from), date.fromisoformat(args.days_to)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]

    # Skip symbols that were delisted before the range: probe the last day first.
    status: dict[str, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one_symbol, s, months, days): s for s in syms}
        for i, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            try:
                sym, st, n = f.result()
                status[sym] = {"status": st, "rows": n}
            except Exception as e:  # noqa: BLE001 -- recorded as a status, never dropped
                status[s] = {"status": "error", "error": str(e)[:200]}
            if i % 50 == 0:
                print(f"{i}/{len(syms)} done, {time.time()-t0:.0f}s", flush=True)
    manifest = {
        "months": months, "days_from": str(d0), "days_to": str(d1),
        "n_symbols_listed": len(syms),
        "counts": pd.Series([v["status"] for v in status.values()]).value_counts().to_dict(),
        "symbols": status,
    }
    (OUT / "_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["counts"]), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
