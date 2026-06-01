"""Build the lightweight maturity-meta file the convexity bot reads as UNIVERSE_META_PREDS.

The bot's eligibility filter only needs each symbol's EARLIEST date — `td = (now - earliest).days`
gated at MIN_HISTORY_DAYS=180. The CORRECT, STABLE source for that is the symbol's actual listing
date (`onboardDate` from Binance FAPI exchangeInfo) — NOT the earliest kline on disk, which
`prune_raw_data.sh` trims (that bug made every sym look ~120d old → maturity filter rejected all →
n_universe=0). onboardDate is one weight-1 API call and never moves.

Emits a cheap (symbol, open_time) 4h grid spanning each symbol's listing → now (grid start floored to
now-400d to bound size; doesn't affect the 180d gate for mature names). Writes the bot's
UNIVERSE_META_PREDS path. Decoupled from the feature panel and from kline retention.

Usage: PYTHONPATH=. .venv/bin/python live/build_maturity_meta.py
"""
import sys
from pathlib import Path
import pandas as pd, requests
REPO = Path("/home/yuqing/ctaNew")
# DEDICATED maturity-meta file (NOT the shared x132 preds — overwriting that broke loop2_iter28's BASELINE).
# The bot reads it via CONVEXITY_UNIV_META; the supervisor exports that env to point here.
META = REPO/"live/state/convexity/maturity_meta.parquet"
FAPI = "https://fapi.binance.com"


def main():
    syms = (REPO/"live/state/universe174.txt").read_text().split()
    now = (pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tz is None
           else pd.Timestamp.utcnow()).floor("4h")
    floor = now - pd.Timedelta(days=400)                      # bound grid size; > 180d so mature syms pass
    j = requests.get(f"{FAPI}/fapi/v1/exchangeInfo", timeout=20).json()
    onboard = {s["symbol"]: pd.to_datetime(s["onboardDate"], unit="ms", utc=True)
               for s in j["symbols"] if "onboardDate" in s}
    rows, n_ok = [], 0
    for sym in syms:
        ob = onboard.get(sym)
        if ob is None:
            continue
        start = max(ob, floor).floor("4h")                   # earliest = listing date (or 400d floor)
        grid = pd.date_range(start, now, freq="4h", tz="UTC")
        rows.append(pd.DataFrame({"symbol": sym, "open_time": grid})); n_ok += 1
    meta = pd.concat(rows, ignore_index=True)
    META.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(META, index=False)
    span = meta.groupby("symbol")["open_time"].min()
    td = (now - span).dt.days
    print(f"maturity meta: {n_ok} syms (onboardDate) → {META.name} | "
          f"#≥180d = {(td >= 180).sum()}/{n_ok}, earliest {span.min().date()}…{span.max().date()}")


if __name__ == "__main__":
    main()
