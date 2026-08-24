"""Signal-diversity loop — fetch free on-chain metrics (S2).

CORRECTION to an earlier claim in this loop: I asserted that free historical per-token address counts do not
exist for our universe. That was wrong. The CoinMetrics **Community API** (no key, no cost) serves daily
AdrActCnt / TxCnt with history back to 2023 for **27 of our 176 base assets** — 15% of the universe, but
weighted toward the majors we actually trade, which is the deployable universe anyway.

Fetches daily active addresses + transaction count per covered asset, 2023-01-01 -> today, caches to
live/state/cost_loop/onchain_daily.parquet (symbol_base, date, AdrActCnt, TxCnt).
Run: python3 -u -m live.sd_onchain_fetch
"""
from __future__ import annotations

import glob
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "live/state/cost_loop/onchain_daily.parquet"
METRICS = "AdrActCnt,TxCnt"
START = "2023-01-01"
API = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
MULT = re.compile(r"^(1000000|10000|1000|100)(.+)$")


def base_of(sym: str) -> str:
    s = sym.replace("USDT", "").replace("USDC", "")
    m = MULT.match(s)
    return (m.group(2) if m else s).lower()


def fetch_asset(a: str) -> pd.DataFrame | None:
    rows, token = [], None
    for _ in range(60):                                   # paginate
        u = (f"{API}?assets={a}&metrics={METRICS}&frequency=1d&start_time={START}&page_size=10000")
        if token:
            u += f"&next_page_token={token}"
        try:
            d = json.load(urllib.request.urlopen(u, timeout=45))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5); continue
            return None
        except Exception:
            time.sleep(2); continue
        rows += d.get("data", [])
        token = d.get("next_page_token")
        if not token:
            break
    if not rows:
        return None
    D = pd.DataFrame(rows)
    D["date"] = pd.to_datetime(D["time"], utc=True).dt.normalize()
    for c in ("AdrActCnt", "TxCnt"):
        D[c] = pd.to_numeric(D.get(c), errors="coerce")
    return D[["asset", "date", "AdrActCnt", "TxCnt"]].dropna(subset=["AdrActCnt"])


def main():
    syms = sorted(Path(f).stem.replace("metrics_", "")
                  for f in glob.glob(str(REPO / "data/ml/cache/metrics_*.parquet")))
    bases = sorted({base_of(s) for s in syms})
    print(f"universe {len(syms)} symbols -> {len(bases)} base assets; probing CoinMetrics community",
          flush=True)
    with ThreadPoolExecutor(5) as ex:
        parts = list(ex.map(fetch_asset, bases))
    parts = [p for p in parts if p is not None and len(p)]
    D = pd.concat(parts, ignore_index=True)
    D.to_parquet(OUT, index=False)
    cov = D.groupby("asset")["date"].agg(["min", "max", "count"])
    print(f"\nwrote {OUT}: {len(D):,} rows, {D.asset.nunique()} assets", flush=True)
    print(f"  date range {D.date.min().date()} -> {D.date.max().date()}", flush=True)
    print(f"  assets: {sorted(D.asset.unique())}", flush=True)
    print(f"  median days per asset: {cov['count'].median():.0f}", flush=True)
    print("ONCHAINFETCHDONE", flush=True)


if __name__ == "__main__":
    main()
