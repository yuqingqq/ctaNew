"""Refresh funding caches to the latest available Binance Vision MONTHLY archive (durable fix for the
2026-04→05 stall: the loader only pulled monthly archives and the daily pipeline never refreshed funding,
so the panel ffilled a stale default and funding_rate_z_7d collapsed to 0 for ~all symbols).

Run before each monthly retrain. Appends the last `--months` complete months to every existing funding
cache (idempotent: dedups on calc_time). The current month's archive only publishes after month-end, so
this keeps funding fresh up to the last completed month — which is exactly what a 1st-of-month retrain needs.

Usage: python3 live/ingest_funding.py [--months 2]
"""
import sys, glob, io, zipfile, argparse
from pathlib import Path
from datetime import date
from concurrent.futures import ThreadPoolExecutor
import requests, pandas as pd
CACHE = Path("/home/yuqing/ctaNew/data/ml/cache")
BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"


def recent_months(n):
    y, m = date.today().year, date.today().month
    out = []
    for _ in range(n + 1):                       # include current (may 404) + n prior
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0: y -= 1; m = 12
    return out


def dl(sym, ym):
    r = requests.get(f"{BASE}/{sym}/{sym}-fundingRate-{ym}.zip", timeout=30)
    if r.status_code == 404: return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    return df.rename(columns={"last_funding_rate": "funding_rate", "funding_interval_hours": "interval_hours"})[
        ["calc_time", "interval_hours", "funding_rate"]]


def patch(fp, months):
    sym = Path(fp).stem.replace("funding_", "")
    old = pd.read_parquet(fp); old["calc_time"] = pd.to_datetime(old["calc_time"], utc=True)
    new = []
    for ym in months:
        try:
            d = dl(sym, ym)
            if d is not None: new.append(d)
        except Exception:
            pass
    if not new: return (sym, 0, old["calc_time"].max())
    comb = pd.concat([old] + new, ignore_index=True).sort_values("calc_time").drop_duplicates("calc_time", keep="last")
    comb.to_parquet(fp, compression="zstd")
    return (sym, len(comb) - len(old), comb["calc_time"].max())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--months", type=int, default=2); a = ap.parse_args()
    months = recent_months(a.months)
    files = sorted(glob.glob(str(CACHE / "funding_*.parquet")))
    added = 0; latest = None
    with ThreadPoolExecutor(max_workers=12) as ex:
        for sym, n, mx in ex.map(lambda f: patch(f, months), files):
            added += n; latest = mx if latest is None else max(latest, mx)
    print(f"[ingest_funding] refreshed {len(files)} caches over {months}; +{added} rows total; latest calc_time {latest}")


if __name__ == "__main__":
    main()
