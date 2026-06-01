"""Live 5m klines via Binance USDM FAPI REST → Vision-format daily parquets.

The convexity pipeline (refresh_convexity_panel → X132 panel) reads klines from
    data/ml/test/parquet/klines/<SYM>/5m/<YYYY-MM-DD>.parquet
which the Binance Vision daily-archive loader produces — but Vision lags ~1 day.
This script writes the SAME parquet format for the recent (today + last few) days
straight from the real-time FAPI endpoint, into the SAME paths. Because
binance_vision_loader._process_day short-circuits when a day's parquet already
exists, the rest of the pipeline (refresh_convexity_panel, X132, incremental
xs_feats) consumes the live data with NO code changes — Vision still owns the
settled history, the live feed owns the unpublished edge.

Klines never revise once a bar closes, so a FAPI-derived complete day is byte-
identical to the Vision archive (validated; see live.fetch_live_klines --validate).

Usage:
  PYTHONPATH=. python3 live/fetch_live_klines.py [--days-back 3] [--workers 8]
  PYTHONPATH=. python3 live/fetch_live_klines.py --validate XRPUSDT 2026-05-02
"""
from __future__ import annotations
import argparse, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests

# --- global throttle: FAPI IP weight limit is 2400/min; klines @limit>1000 = weight 10.
# A burst of unthrottled workers triggers HTTP 429 → then 418 (timed IP ban). We pace every
# request through a shared min-interval gate and back off hard on 429/418 instead of hammering.
_THROTTLE = threading.Lock()
_LAST_REQ = [0.0]
_MIN_INTERVAL = [0.5]    # seconds between requests across all workers (tunable via --min-interval).
                         # klines limit=1500 → weight 10; 0.5s ⇒ ~120 req/min ⇒ 1200 weight/min (½ of the
                         # 2400/min IP cap). 0.15s re-triggered a 418 ban (2026-06-01) — keep this ≥0.4.
_BAN_UNTIL = [0.0]       # if a 429/418 sets a cooldown, all workers wait past it


def _throttled_get(session, url, params, timeout=15):
    """Rate-limited GET with 429/418 (Retry-After) backoff. Raises on persistent failure."""
    for attempt in range(5):
        with _THROTTLE:
            now = time.time()
            wait = max(_BAN_UNTIL[0] - now, _LAST_REQ[0] + _MIN_INTERVAL[0] - now)
            if wait > 0:
                time.sleep(wait)
            _LAST_REQ[0] = time.time()
        r = session.get(url, params=params, timeout=timeout)
        if r.status_code in (429, 418):
            ra = int(r.headers.get("Retry-After", "0") or 0)
            cooldown = max(ra, 2 ** attempt * 5)   # honor Retry-After, else exponential
            with _THROTTLE:
                _BAN_UNTIL[0] = max(_BAN_UNTIL[0], time.time() + cooldown)
            time.sleep(min(cooldown, 30))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"persistent 429/418 after retries: {url}")

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
KLINES_ROOT = REPO / "data/ml/test/parquet/klines"
FAPI = "https://fapi.binance.com"

# Vision kline parquet schema (post-_kline_postprocess, 'ignore' dropped, index=False).
OUT_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]


def _fapi_klines(symbol: str, start_ms: int, end_ms: int, interval: str = "5m",
                 session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch [start_ms, end_ms) of klines, paginating 1500/req. Vision schema/dtypes."""
    s = session or requests.Session()
    rows, cursor = [], start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": cursor, "endTime": end_ms, "limit": 1500}
        r = _throttled_get(s, f"{FAPI}/fapi/v1/klines", params)
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        cursor = chunk[-1][0] + 1
        if len(chunk) < 1500:
            break
    if not rows:
        return pd.DataFrame(columns=OUT_COLS)
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_volume", "taker_buy_quote_volume"):
        df[c] = df[c].astype("float64")
    df["count"] = df["count"].astype("int64")
    return df[OUT_COLS]


def fetch_symbol(symbol: str, days_back: int, session: requests.Session | None = None) -> int:
    """Write Vision-format daily parquets for the last `days_back` UTC days (today
    partial + recent complete) from FAPI. Only closed bars are written; the current
    in-progress bar is excluded. Returns number of day-files written."""
    now = datetime.now(timezone.utc)
    start_day = (now - timedelta(days=days_back)).date()
    start_ms = int(datetime(start_day.year, start_day.month, start_day.day,
                            tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    df = _fapi_klines(symbol, start_ms, end_ms, session=session)
    if df.empty:
        return 0
    # drop the still-open current bar (close_time in the future)
    df = df[df["close_time"] <= now]
    if df.empty:
        return 0
    written = 0
    for day, g in df.groupby(df["open_time"].dt.date):
        out = KLINES_ROOT / symbol / "5m" / f"{day:%Y-%m-%d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        g.reset_index(drop=True).to_parquet(out, compression="zstd", index=False)
        written += 1
    return written


def _panel_syms() -> list[str]:
    from live.refresh_convexity_panel import list_syms_from_panel
    return list_syms_from_panel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-back", type=int, default=3,
                    help="UTC days of live edge to (re)write from FAPI (default 3)")
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent symbol fetchers (low — requests are throttled globally)")
    ap.add_argument("--min-interval", type=float, default=0.5,
                    help="min seconds between FAPI requests across all workers (keep >=0.4 to stay under the IP weight cap)")
    ap.add_argument("--syms", nargs="*", default=None,
                    help="explicit symbols; default = panel universe")
    ap.add_argument("--validate", nargs=2, metavar=("SYM", "DAY"),
                    help="compare FAPI vs the on-disk Vision parquet for SYM on DAY (YYYY-MM-DD)")
    args = ap.parse_args()

    if args.validate:
        _validate(*args.validate)
        return

    _MIN_INTERVAL[0] = args.min_interval
    syms = args.syms or _panel_syms()
    t0 = time.time()
    print(f"live klines: {len(syms)} syms × last {args.days_back}d from FAPI", flush=True)
    ok = files = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        sess = requests.Session()
        futs = {ex.submit(fetch_symbol, s, args.days_back, sess): s for s in syms}
        for i, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                n = fut.result()
                ok += int(n > 0); files += n
            except Exception as e:
                print(f"  {s:<14} ERR {type(e).__name__}: {e}", flush=True)
            if i % 50 == 0 or i == len(syms):
                print(f"  [{i}/{len(syms)}] {files} day-files [{time.time()-t0:.0f}s]", flush=True)
    print(f"live klines DONE: {ok}/{len(syms)} syms, {files} day-files [{time.time()-t0:.0f}s]", flush=True)


def _validate(sym: str, day: str):
    """Parity check: FAPI-fetched day vs the existing Vision parquet for that day."""
    vis_path = KLINES_ROOT / sym / "5m" / f"{day}.parquet"
    if not vis_path.exists():
        print(f"no Vision parquet at {vis_path} — fetch a settled day first")
        return
    vis = pd.read_parquet(vis_path)
    d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    fapi = _fapi_klines(sym, int(d0.timestamp() * 1000),
                        int((d0 + timedelta(days=1)).timestamp() * 1000))
    fapi = fapi[fapi["open_time"].dt.date == d0.date()].reset_index(drop=True)
    vis = vis.reset_index(drop=True)
    print(f"{sym} {day}: vision {len(vis)} bars, fapi {len(fapi)} bars")
    if len(vis) != len(fapi):
        print("  ROW COUNT MISMATCH"); return
    num = ["open", "high", "low", "close", "volume", "quote_volume",
           "count", "taker_buy_volume", "taker_buy_quote_volume"]
    maxdiff = (vis[num].astype("float64") - fapi[num].astype("float64")).abs().max()
    tdiff = (vis["open_time"].values != fapi["open_time"].values).sum()
    print(f"  open_time mismatches: {tdiff}")
    print(f"  max abs numeric diff:\n{maxdiff.to_string()}")
    print("  PARITY OK" if maxdiff.max() == 0 and tdiff == 0 else "  DIFFERENCES PRESENT")


if __name__ == "__main__":
    main()
