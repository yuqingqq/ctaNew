"""Backfill 5m klines bars the WS collector dropped, from FAPI.

The websocket feed intermittently loses 5m bars (e.g. XLM lost 8 bars on 6/4 11:35-15:20). The feature
pipeline indexes by ROW (return_1d = close.pct_change(288), rolling(288), bars_since_high, obv cumsum), so a
missing bar makes those reach back the wrong distance — XLM 6/4 return_1d came out -0.086 vs the true -0.065,
off by exactly the gap; obv/atr also went wrong because the gap bars' real volume/range were absent. FAPI has
the real bars, so we fetch + merge them into the daily archive files → gapless → features align AND get the
right values. (incremental_xs_feats also reindexes to a gapless grid as a ffill safety-net for any gap this
misses, but ffill can't recover the missing bars' volume/range — backfilling the real bars is the actual fix.)

REST is ban-prone, so: throttled, and only symbols with an actual gap in the recent window get a fetch.

Usage:
  python3 live/backfill_klines_gaps.py                 # all panel-universe syms
  python3 live/backfill_klines_gaps.py --symbols XLMUSDT SEIUSDT
"""
from __future__ import annotations
import argparse, sys, time, json, urllib.request
from pathlib import Path
import pandas as pd
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
KLINES = REPO / "data/ml/test/parquet/klines"
HEALTH = REPO / "live/state/data_health.json"   # read by decide_v1's freshness gate + cycle_monitor
LOOKBACK_DAYS = 8                  # ≥ the longest V0 feature lookback (autocorr 7d, beta-change ~6d) so the
                                   # CURRENT decision bar's features are gapless; older gaps are settled history
THROTTLE = 0.45                    # ≥0.4s between FAPI calls (ban-safe)
FAPI = "https://fapi.binance.com/fapi/v1/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]


def _recent_files(sym):
    return sorted((KLINES / sym / "5m").glob("*.parquet"))[-(LOOKBACK_DAYS + 2):]


def _load(paths):
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.drop_duplicates("open_time").sort_values("open_time")


def _count_gaps(df, since):
    s = df[df["open_time"] >= since]
    if len(s) < 2:
        return 0
    grid = pd.date_range(s["open_time"].min(), s["open_time"].max(), freq="5min")
    return len(grid) - len(s)


def _fapi(sym, start_ms, end_ms):
    raw = []                                                        # paginate: FAPI caps at 1500 bars/call
    cur = start_ms
    while cur < end_ms:
        url = f"{FAPI}?symbol={sym}&interval=5m&startTime={cur}&endTime={end_ms}&limit=1500"
        page = json.load(urllib.request.urlopen(url, timeout=15))
        if not page:
            break
        raw.extend(page)
        last = int(page[-1][0])
        if len(page) < 1500 or last <= cur:
            break
        cur = last + 300_000                                        # +5m past the last bar
        time.sleep(THROTTLE)
    if not raw:
        return None
    df = pd.DataFrame(raw).iloc[:, :11].drop_duplicates(0)
    df.columns = COLS
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_volume", "taker_buy_quote_volume"]:
        df[c] = df[c].astype(float)
    df["count"] = df["count"].astype("int64")
    return df


def backfill_sym(sym):
    """Returns (status, residual_gaps): residual_gaps = bars STILL missing after the FAPI fetch
    (FAPI down/banned or genuinely missing). 0 = fully repaired."""
    paths = _recent_files(sym)
    if not paths:
        return "no-data", 0
    df = _load(paths)
    since = df["open_time"].max().floor("5min") - pd.Timedelta(days=LOOKBACK_DAYS)
    g0 = _count_gaps(df, since)
    if g0 == 0:
        return "gapless", 0
    start_ms = int(since.timestamp() * 1000)
    end_ms = int(df["open_time"].max().timestamp() * 1000)
    fa = _fapi(sym, start_ms, end_ms)
    time.sleep(THROTTLE)
    if fa is None or fa.empty:
        return "fapi-empty", g0
    have = set(df["open_time"])
    new = fa[~fa["open_time"].isin(have)].copy()
    if new.empty:
        return "no-new", g0
    new["__day"] = new["open_time"].dt.strftime("%Y-%m-%d")          # merge each bar into its day's file
    n = 0
    for day, g in new.groupby("__day"):
        fp = KLINES / sym / "5m" / f"{day}.parquet"
        g = g.drop(columns="__day")
        if fp.exists():
            ex = pd.read_parquet(fp); ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
            comb = (pd.concat([ex, g.reindex(columns=ex.columns)])
                    .drop_duplicates("open_time").sort_values("open_time"))
        else:
            comb = g.sort_values("open_time")
        comb.to_parquet(fp, index=False)
        n += len(g)
    resid = _count_gaps(_load(_recent_files(sym)), since)            # FAPI may not have every bar either
    return f"filled {n}/resid {resid}", resid


def _safe(sym):
    try:
        st, resid = backfill_sym(sym)
        return sym, st, resid
    except Exception as e:
        return sym, f"FAIL {str(e)[:70]}", -1                        # -1 = FAPI/exception (counts as error)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    a = ap.parse_args()
    syms = a.symbols or sorted(set(pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_expanded_v0.parquet", columns=["symbol"]).symbol.unique()))
    t0 = time.time(); filled = gapless = 0
    residual_gappy = []; errors = []
    for s in syms:                                                   # serial: throttle the FAPI calls
        _, r, resid = _safe(s)
        if r == "gapless":
            gapless += 1
        elif str(r).startswith("filled"):
            filled += 1; print(f"  {s}: {r}", flush=True)
        if resid == -1:
            errors.append(s)
        elif resid > 0:
            residual_gappy.append(s)
    # health report for the freshness gate (decide_v1) + monitor: what the feed couldn't repair this cycle
    try:
        HEALTH.parent.mkdir(parents=True, exist_ok=True)
        HEALTH.write_text(json.dumps({
            "ts": pd.Timestamp.utcnow().isoformat(), "n_syms": len(syms),
            "backfilled": filled, "gapless": gapless,
            "residual_gappy": sorted(residual_gappy), "fapi_errors": sorted(errors)}, indent=0))
    except Exception:
        pass
    print(f"[backfill_klines_gaps] {filled} backfilled, {gapless} gapless, "
          f"{len(residual_gappy)} still-gappy, {len(errors)} errors [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
