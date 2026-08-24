"""Fetch 365 days of daily market-cap history from CoinGecko's free tier (the maximum it allows) for the
perp universe, so the OI/MC leverage ratio can be tested on real market caps over the RECENT window.

Why only 365d: the free API rejects longer ranges (error 10012). That covers RECENT (2025-10 -> 2026-06) but
not OOS, so anything measured here is SINGLE-ERA and diagnostic only — not adoptable under the loop's rules.
Its job is narrower: decide whether a PAID full-history market-cap feed is worth buying.

Caches to live/state/cost_loop/cg_mcap_365d.parquet (symbol, date, mcap). Rate-limited, resumable.
Run: python3 -u -m live.mc_fetch_365d
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from live.mc_oi_probe import base_of, fetch_mcap
from live.cost_loop_harness import CACHE, REPO

OUT = CACHE / "cg_mcap_365d.parquet"
PART = CACHE / "_cg_mcap_partial.json"
SLEEP = 4.0


def universe() -> list[str]:
    import glob
    return sorted(Path(f).stem.replace("metrics_", "")
                  for f in glob.glob(str(REPO / "data/ml/cache/metrics_*.parquet")))


def main():
    mc = fetch_mcap()                      # ticker -> {mcap, id, ...}
    syms = universe()
    want = {}
    for s in syms:
        b, _ = base_of(s)
        cid = mc.get(b, {}).get("id")
        if cid:
            want[s] = cid
    print(f"universe {len(syms)} | coingecko id resolved for {len(want)}", flush=True)

    done = json.loads(PART.read_text()) if PART.exists() else {}
    todo = [s for s in want if s not in done]
    print(f"already cached {len(done)} | to fetch {len(todo)} (~{len(todo)*SLEEP/60:.0f} min)", flush=True)

    for i, s in enumerate(todo, 1):
        cid = want[s]
        try:
            r = requests.get(f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart",
                             params=dict(vs_currency="usd", days=365, interval="daily"), timeout=45)
            if r.status_code == 429:
                print(f"  [{i}/{len(todo)}] {s}: 429 rate-limited, backing off 60s", flush=True)
                time.sleep(60); continue
            if r.status_code != 200:
                print(f"  [{i}/{len(todo)}] {s}: HTTP {r.status_code}", flush=True)
                done[s] = []
            else:
                done[s] = r.json().get("market_caps", [])
                if i % 10 == 0:
                    print(f"  [{i}/{len(todo)}] {s}: {len(done[s])} points", flush=True)
                    PART.write_text(json.dumps(done))
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {s}: ERR {str(e)[:60]}", flush=True)
        time.sleep(SLEEP)
    PART.write_text(json.dumps(done))

    rows = []
    for s, pts in done.items():
        for ts, v in pts or []:
            rows.append((s, pd.Timestamp(ts, unit="ms", tz="UTC").normalize(), float(v)))
    D = pd.DataFrame(rows, columns=["symbol", "date", "mcap"]).dropna()
    D = D[D.mcap > 0].drop_duplicates(["symbol", "date"])
    D.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(D):,} rows, {D.symbol.nunique()} syms, "
          f"{D.date.min().date()} -> {D.date.max().date()}", flush=True)
    print("MCFETCHDONE", flush=True)


if __name__ == "__main__":
    main()
