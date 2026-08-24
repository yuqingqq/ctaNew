"""Memory-bounded, resumable Binance bookDepth backfill.

The normal loader intentionally keeps a symbol's fetched daily frames in memory
until it has aggregated them.  That is convenient for small tracking windows,
but unsuitable for a multi-month historical repair.  This driver invokes that
same PIT aggregator in short date chunks and atomically merges each chunk into
the existing ``l2_<SYMBOL>.parquet`` cache.

Default universe: the fixed bridge cohort that already has local L2 on both
sides of the 2024-07-01..2025-09-30 collection gap.  It gives a genuinely
balanced historical panel; newer listings cannot be made historical by a
backfill.

Example (the conservative default):
    python3 -m live.bookdepth_backfill

Interrupted runs are safe to restart.  Fully cached source days are skipped;
the only retained data are the aggregated 4h feature parquets and a compact
CSV manifest of archive days that returned no usable data.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd

from live.bookdepth_loader import CACHE, load_symbol

GAP_START = pd.Timestamp("2024-07-01", tz="UTC")
GAP_END = pd.Timestamp("2025-09-30", tz="UTC")
LEFT_ANCHOR = pd.Timestamp("2024-06-30 20:00", tz="UTC")
RIGHT_ANCHOR = pd.Timestamp("2025-10-01 00:00", tz="UTC")
PANEL = Path("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet")


def bridge_symbols() -> list[str]:
    """Symbols with cached L2 on both sides of the collection gap."""
    panel = sorted(pd.read_parquet(PANEL, columns=["symbol"])["symbol"].unique())
    out: list[str] = []
    for sym in panel + ["BTCUSDT"]:
        f = CACHE / f"l2_{sym}.parquet"
        if not f.exists():
            continue
        idx = pd.to_datetime(pd.read_parquet(f, columns=[]).index, utc=True)
        if (idx <= LEFT_ANCHOR).any() and (idx >= RIGHT_ANCHOR).any():
            out.append(sym)
    return sorted(set(out))


def complete_days(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """A normal calendar UTC day contributes six 4h observation bars."""
    if not len(index):
        return set()
    day_counts = pd.Series(1, index=index.floor("1D")).groupby(level=0).sum()
    return set(day_counts[day_counts >= 6].index)


def atomic_merge(path: Path, old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([old, new]) if len(old) else new.copy()
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    tmp = path.with_suffix(".tmp.parquet")
    merged.to_parquet(tmp)
    tmp.replace(path)
    return merged


def backfill_symbol(sym: str, days: pd.DatetimeIndex, chunk_days: int, workers: int) -> tuple[int, int]:
    path = CACHE / f"l2_{sym}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if len(old):
        old.index = pd.to_datetime(old.index, utc=True)
    cached = complete_days(old.index) if len(old) else set()
    added = missing = 0
    for lo in range(0, len(days), chunk_days):
        chunk = days[lo : lo + chunk_days]
        todo = [d for d in chunk if d not in cached]
        if not todo:
            continue
        out = load_symbol(sym, todo, workers=workers)
        if out is None:
            missing += len(todo)
            continue
        old = atomic_merge(path, old, out)
        got = complete_days(out.index)
        added += len(got)
        missing += len(todo) - len(got)
        cached |= got
        del out
        gc.collect()
    return added, missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Resumable, memory-bounded bookDepth backfill")
    ap.add_argument("--start", default=str(GAP_START.date()))
    ap.add_argument("--end", default=str(GAP_END.date()))
    ap.add_argument("--syms", default=None, help="comma-separated; default is the fixed bridge cohort")
    ap.add_argument("--chunk-days", type=int, default=14, help="days retained per parser call (default: 14)")
    ap.add_argument("--workers", type=int, default=4, help="simultaneous archive requests within one chunk")
    args = ap.parse_args()
    if args.chunk_days < 1 or args.workers < 1:
        ap.error("--chunk-days and --workers must be positive")
    days = pd.date_range(args.start, args.end, freq="D", tz="UTC")
    syms = args.syms.split(",") if args.syms else bridge_symbols()
    if not syms:
        raise RuntimeError("no bridge symbols found; do not start a blind full-universe backfill")
    CACHE.mkdir(parents=True, exist_ok=True)
    print(
        f"BACKFILL start: {len(syms)} symbols, {len(days)} days, {days[0].date()}..{days[-1].date()}, "
        f"chunks={args.chunk_days}d, workers={args.workers}",
        flush=True,
    )
    manifest = []
    for i, sym in enumerate(syms, 1):
        added, missing = backfill_symbol(sym, days, args.chunk_days, args.workers)
        manifest.append({"symbol": sym, "new_complete_days": added, "missing_or_partial_days": missing})
        print(f"[{i:03d}/{len(syms)}] {sym:14s} +{added:3d} complete days; {missing:3d} missing/partial", flush=True)
        gc.collect()
    mp = CACHE / f"l2_backfill_manifest_{days[0]:%Y%m%d}_{days[-1]:%Y%m%d}.csv"
    pd.DataFrame(manifest).to_csv(mp, index=False)
    print(f"BACKFILL DONE; manifest={mp}", flush=True)


if __name__ == "__main__":
    main()
