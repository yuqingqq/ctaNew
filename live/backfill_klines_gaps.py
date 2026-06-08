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
import argparse, os, sys, time, json, urllib.request
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


def _missing_bars(df, since, target):
    """Bars missing in [max(since, first_present) .. target] — counts BOTH internal gaps AND a
    stale tail (a symbol frozen before `target`). Returns the sorted list of missing timestamps."""
    lo = max(since, df["open_time"].min())
    grid = pd.date_range(lo, target, freq="5min")
    present = set(df.loc[df["open_time"] >= lo, "open_time"])
    return [t for t in grid if t not in present]


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


def backfill_sym(sym, target=None):
    """Bring `sym`'s recent klines current up to `target` (the universe's freshest closed bar).
    Fills internal gaps AND a stale tail — a non-traded symbol the WS collector stopped feeding
    (its klines freeze in the past) must still be carried forward, because it's a peer in the
    cross-sectional bars_since_high_xs_rank cohort (drop it and the whole rank mis-scales).

    Returns (status, residual): residual = bars STILL missing after the FAPI fetch (0 = repaired).
    If `target` is None, falls back to the symbol's own last bar (legacy gap-only behaviour)."""
    paths = _recent_files(sym)
    if not paths:
        return "no-data", 0
    df = _load(paths)
    tgt = (target or df["open_time"].max()).floor("5min")
    since = tgt - pd.Timedelta(days=LOOKBACK_DAYS)
    miss = _missing_bars(df, since, tgt)
    if not miss:
        return "gapless", 0
    stale = df["open_time"].max() < tgt - pd.Timedelta(minutes=5)    # frozen tail vs just internal holes
    start_ms = int(miss[0].timestamp() * 1000)                       # fetch from the first hole forward
    end_ms = int(tgt.timestamp() * 1000)
    fa = _fapi(sym, start_ms, end_ms)
    time.sleep(THROTTLE)
    if fa is None or fa.empty:
        return "fapi-empty", len(miss)
    have = set(df["open_time"])
    new = fa[~fa["open_time"].isin(have)].copy()
    if new.empty:
        return "no-new", len(miss)
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
        tmp = fp.with_name(f"{fp.name}.{os.getpid()}.tmp"); comb.to_parquet(tmp, index=False); tmp.replace(fp)  # unique-per-proc atomic
        n += len(g)
    resid = len(_missing_bars(_load(_recent_files(sym)), since, tgt))  # FAPI may not have every bar either
    tag = "refreshed" if stale else "filled"
    return f"{tag} {n}/resid {resid}", resid


def _safe(sym, target=None):
    try:
        st, resid = backfill_sym(sym, target)
        return sym, st, resid
    except Exception as e:
        return sym, f"FAIL {str(e)[:70]}", -1                        # -1 = FAPI/exception (counts as error)


def _latest_bar(sym):
    files = sorted((KLINES / sym / "5m").glob("*.parquet"))
    if not files:
        return None
    try:
        m = pd.to_datetime(pd.read_parquet(files[-1], columns=["open_time"])["open_time"], utc=True)
        return m.max()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    a = ap.parse_args()
    syms = a.symbols or sorted(set(pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_expanded_v0.parquet", columns=["symbol"]).symbol.unique()))
    # Universe-wide TARGET = the freshest closed bar any symbol has reached (the traded set, fed by the WS
    # collector). Stale non-traded peers (high-vol names kept only for the xs-rank cohort) are pulled UP to
    # it; max() ignores their frozen tails. Without this, a non-traded symbol frozen in the past reads
    # "gapless" against its own last bar and silently drops out of the cross-sectional cohort.
    lat = [x for x in (_latest_bar(s) for s in syms) if x is not None]
    now_bar = pd.Timestamp.utcnow().floor("5min") - pd.Timedelta(minutes=5)   # last legitimately-closed 5m bar
    target = min(max(lat), now_bar) if lat else None    # cap: one symbol's future/ahead bar can't inflate target
    t0 = time.time(); filled = gapless = refreshed = 0
    residual_gappy = []; errors = []; stale_refreshed = []
    for s in syms:                                                   # serial: throttle the FAPI calls
        _, r, resid = _safe(s, target)
        if r == "gapless":
            gapless += 1
        elif str(r).startswith("refreshed"):
            refreshed += 1; stale_refreshed.append(s); print(f"  {s}: {r}", flush=True)
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
            "target_bar": str(target), "backfilled": filled, "stale_refreshed": refreshed,
            "gapless": gapless, "residual_gappy": sorted(residual_gappy),
            "stale_refreshed_syms": sorted(stale_refreshed), "fapi_errors": sorted(errors)}, indent=0))
    except Exception:
        pass
    print(f"[backfill_klines_gaps] target={target} | {filled} gap-filled, {refreshed} stale-refreshed, "
          f"{gapless} gapless, {len(residual_gappy)} still-gappy, {len(errors)} errors [{time.time()-t0:.0f}s]",
          flush=True)


if __name__ == "__main__":
    main()
