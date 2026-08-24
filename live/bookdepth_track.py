"""Forward TRACKER for the order-book imbalance dataset (user: "keep it tracked"). Appends the latest available
bookDepth days for the convexity panel universe to the l2_* feature cache. Idempotent — the loader merges on write,
so re-running is safe and fills any gap up to `--days` back. bookDepth for day D publishes ~D+1, so it tracks with a
~1-day lag. Run daily (cron) to keep the 4h-imbalance features current.

  python3 -m live.bookdepth_track [--days 5] [--syms BTCUSDT,ETHUSDT]   # default: panel 175 + BTC, trailing 5 days

Cron (daily 06:00 UTC):
  0 6 * * *  cd /home/yuqing/ctaNew && /usr/bin/python3 -m live.bookdepth_track --days 3 >> live/state/l2_track.log 2>&1
"""
import argparse
import datetime as dt
from pathlib import Path
import pandas as pd
from live.bookdepth_loader import load_symbol, CACHE

def panel_syms():
    p = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet"
    s = sorted(pd.read_parquet(p, columns=["symbol"]).symbol.unique())
    return s + ["BTCUSDT"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5, help="trailing days to (re)fetch and append")
    ap.add_argument("--syms", default=None, help="comma list; default = panel universe + BTC")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    syms = a.syms.split(",") if a.syms else panel_syms()
    today = dt.date.today()
    days = pd.to_datetime([today - dt.timedelta(days=k) for k in range(1, a.days + 1)]).sort_values()
    stamp = dt.datetime.utcnow().isoformat(timespec="seconds")
    print(f"[{stamp}] tracking bookDepth: {len(syms)} syms, days {days[0].date()}..{days[-1].date()}", flush=True)
    n_new = 0
    for i, sym in enumerate(syms, 1):
        out = load_symbol(sym, days, a.workers)
        if out is None:
            continue
        p = CACHE / f"l2_{sym}.parquet"
        before = 0
        if p.exists():
            old = pd.read_parquet(p); before = len(old)
            out = pd.concat([old, out]); out = out[~out.index.duplicated(keep="last")].sort_index()
        out.to_parquet(p)
        added = len(out) - before
        if added > 0:
            n_new += added
        if i % 40 == 0:
            print(f"  {i}/{len(syms)} (new bars so far: {n_new})", flush=True)
    print(f"[{stamp}] TRACKDONE: appended {n_new} new 4h-bars across {len(syms)} symbols", flush=True)

if __name__ == "__main__":
    main()
