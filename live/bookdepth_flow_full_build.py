"""Resumable full-history builder for snapshot + aggTrade dynamics research.

Every partition is calculated from the raw 30-second Binance Vision bookDepth
snapshots and exact local aggTrades.  By default it persists the last observed
snapshot in each 5-minute bin, retaining point-in-time feature timestamps while
avoiding ten heavily overlapping rows per reaction window.

Output is research cache only:

    data/ml/cache/research/bookdepth_flow_all_5min_v2/<SYMBOL>/<YYYY-MM-DD>.parquet

The job is safe to resume: non-empty partitions are skipped, writes are atomic,
and a partial manifest is checkpointed by the parent process.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from live.bookdepth_flow_dynamics import AGG_ROOT, build_dynamics
from live.bookdepth_timing_corrected import fixed_universe


REPO = Path("/home/yuqing/ctaNew")
DEFAULT_OUT = REPO / "data/ml/cache/research/bookdepth_flow_all_5min_v2"


@dataclass
class Result:
    symbol: str
    day: str
    status: str
    rows: int = 0
    gap_rows: int = 0
    ask_candidates: int = 0
    bid_candidates: int = 0
    bytes: int = 0
    seconds: float = 0.0
    raw_gap_window_rows: int = 0
    extreme_window_rows: int = 0
    quality_valid_rows: int = 0
    source_day_complete: bool = False
    error: str = ""


def _local_days(symbol: str, start: str | None, end: str | None) -> list[pd.Timestamp]:
    files = sorted((AGG_ROOT / symbol).glob("*.parquet"))
    days = []
    lo = pd.Timestamp(start, tz="UTC") if start else None
    hi = pd.Timestamp(end, tz="UTC") if end else None
    for path in files:
        try:
            day = pd.Timestamp(path.stem, tz="UTC")
        except ValueError:
            continue
        if lo is not None and day < lo:
            continue
        if hi is not None and day > hi:
            continue
        days.append(day)
    return days


def _all_local_symbols() -> list[str]:
    return sorted(
        path.name for path in AGG_ROOT.iterdir()
        if path.is_dir() and any(path.glob("*.parquet"))
    )


def _partition_path(root: Path, symbol: str, day: pd.Timestamp) -> Path:
    return root / symbol / f"{day:%Y-%m-%d}.parquet"


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _build_one(
    symbol: str,
    day: pd.Timestamp,
    *,
    root: Path,
    window: str,
    resolution: str,
    retries: int,
    overwrite: bool,
) -> Result:
    tic = time.monotonic()
    path = _partition_path(root, symbol, day)
    if not overwrite and path.exists() and path.stat().st_size > 0:
        suffix = window
        prior = pd.read_parquet(path, columns=[
            "gap_interval", f"any_raw_gap_{suffix}", f"extreme_imbalance_{suffix}",
            f"quality_valid_{suffix}", f"ask_absorption_candidate_{suffix}",
            f"bid_absorption_candidate_{suffix}", "source_day_complete",
        ])
        return Result(
            symbol=symbol, day=day.strftime("%Y-%m-%d"), status="skipped", rows=len(prior),
            gap_rows=int(prior["gap_interval"].sum()),
            raw_gap_window_rows=int(prior[f"any_raw_gap_{suffix}"].sum()),
            extreme_window_rows=int(prior[f"extreme_imbalance_{suffix}"].sum()),
            quality_valid_rows=int(prior[f"quality_valid_{suffix}"].sum()),
            source_day_complete=bool(prior["source_day_complete"].all()),
            ask_candidates=int(prior[f"ask_absorption_candidate_{suffix}"].sum()),
            bid_candidates=int(prior[f"bid_absorption_candidate_{suffix}"].sum()),
            bytes=path.stat().st_size,
        )

    d = pd.DataFrame()
    for attempt in range(retries + 1):
        try:
            d = build_dynamics(symbol, pd.DatetimeIndex([day]), window=window)
        except Exception as exc:
            if attempt == retries:
                return Result(
                    symbol, day.strftime("%Y-%m-%d"), "error",
                    seconds=time.monotonic() - tic,
                    error=f"{type(exc).__name__}: {str(exc)[:240]}",
                )
        if not d.empty:
            break
        if attempt < retries:
            time.sleep(1.0 + attempt)
    if d.empty:
        return Result(
            symbol, day.strftime("%Y-%m-%d"), "empty",
            seconds=time.monotonic() - tic,
            error="no overlapping bookDepth + aggTrades after retries",
        )

    start = day.floor("1D")
    d = d[(d.index >= start) & (d.index < start + pd.Timedelta("1D"))]
    source_snapshot_count_day = len(d)
    if resolution != "30s":
        # Keep the last actual snapshot within each output bin. The stored
        # snapshot_time—not the bin boundary—is the feature availability time.
        d = d.groupby(d.index.floor(resolution), sort=True).tail(1)
    if d.empty:
        return Result(symbol, day.strftime("%Y-%m-%d"), "empty", seconds=time.monotonic() - tic)

    suffix = window
    source_day_bar_count = len(d)
    complete_threshold = 280 if resolution == "5min" else 2700
    source_day_complete = source_day_bar_count >= complete_threshold
    d["source_snapshot_count_day"] = source_snapshot_count_day
    d["source_day_bar_count"] = source_day_bar_count
    d["source_day_complete"] = source_day_complete
    d[f"quality_valid_{suffix}"] = d[f"window_data_valid_{suffix}"] & source_day_complete
    for side in ["ask", "bid"]:
        flag = f"{side}_absorption_candidate_{suffix}"
        d[flag] = d[flag] & d[f"quality_valid_{suffix}"]

    out = d.reset_index(names="snapshot_time")
    out.insert(1, "bar_time", out["snapshot_time"].dt.floor(resolution))
    _write_atomic(out, path)
    return Result(
        symbol=symbol, day=day.strftime("%Y-%m-%d"), status="complete", rows=len(out),
        gap_rows=int(out["gap_interval"].sum()),
        raw_gap_window_rows=int(out[f"any_raw_gap_{suffix}"].sum()),
        extreme_window_rows=int(out[f"extreme_imbalance_{suffix}"].sum()),
        quality_valid_rows=int(out[f"quality_valid_{suffix}"].sum()),
        source_day_complete=source_day_complete,
        ask_candidates=int(out[f"ask_absorption_candidate_{suffix}"].sum()),
        bid_candidates=int(out[f"bid_absorption_candidate_{suffix}"].sum()),
        bytes=path.stat().st_size, seconds=time.monotonic() - tic,
    )


def _checkpoint(results: list[Result], root: Path, final: bool = False) -> None:
    if not results:
        return
    frame = pd.DataFrame([asdict(x) for x in results])
    frame = frame.drop_duplicates(["symbol", "day"], keep="last").sort_values(["symbol", "day"])
    name = "_manifest.parquet" if final else "_manifest.partial.parquet"
    _write_atomic(frame, root / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None, help="comma-separated override")
    ap.add_argument("--universe", choices=["all-local", "fixed64"], default="all-local")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--window", default="5min")
    ap.add_argument("--resolution", default="5min", choices=["5min", "30s"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--executor", choices=["processes", "threads"], default="processes")
    ap.add_argument("--max-tasks-per-child", type=int, default=0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test first N symbol-days")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _all_local_symbols() if args.universe == "all-local" else fixed_universe()
    root = Path(args.out)
    all_tasks = [(s, d) for s in symbols for d in _local_days(s, args.start, args.end)]
    if args.limit is not None:
        all_tasks = all_tasks[:args.limit]
    if not all_tasks:
        raise SystemExit("no local aggTrade partitions in requested scope")
    task_keys = {(s, d.strftime("%Y-%m-%d")) for s, d in all_tasks}
    existing = [(s, d) for s, d in all_tasks if _partition_path(root, s, d).exists()]
    existing_keys = {(s, d.strftime("%Y-%m-%d")) for s, d in existing}
    tasks = all_tasks if args.overwrite else [x for x in all_tasks if (x[0], x[1].strftime("%Y-%m-%d")) not in existing_keys]

    prior_by_key: dict[tuple[str, str], Result] = {}
    manifests = [p for p in [root / "_manifest.parquet", root / "_manifest.partial.parquet"] if p.exists()]
    if manifests:
        prior = pd.read_parquet(max(manifests, key=lambda p: p.stat().st_mtime))
        fields = set(Result.__dataclass_fields__)
        for record in prior.to_dict(orient="records"):
            key = (record["symbol"], record["day"])
            if key in task_keys and key in existing_keys:
                prior_by_key[key] = Result(**{k: record[k] for k in fields})
    results: list[Result] = list(prior_by_key.values())
    for s, d in existing:
        key = (s, d.strftime("%Y-%m-%d"))
        if key not in prior_by_key:
            results.append(_build_one(
                s, d, root=root, window=args.window, resolution=args.resolution,
                retries=0, overwrite=False,
            ))
    print(
        f"pending {len(tasks)} / total {len(all_tasks)} | existing {len(existing)} | "
        f"symbols {len(set(s for s, _ in all_tasks))} | resolution {args.resolution} | "
        f"{args.executor} {args.workers} | out {root}",
        flush=True,
    )
    if not tasks:
        _checkpoint(results, root, final=True)
        print("nothing pending; manifest finalized", flush=True)
        print("FULLFLOWDONE", flush=True)
        return

    tic = time.monotonic()
    if args.executor == "processes":
        pool_kwargs = {"max_workers": args.workers}
        if args.max_tasks_per_child > 0:
            pool_kwargs["max_tasks_per_child"] = args.max_tasks_per_child
        pool = ProcessPoolExecutor(**pool_kwargs)
    else:
        pool = ThreadPoolExecutor(max_workers=args.workers)
    with pool:
        futures = {
            pool.submit(
                _build_one, s, d, root=root, window=args.window,
                resolution=args.resolution, retries=args.retries,
                overwrite=args.overwrite,
            ): (s, d)
            for s, d in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            s, d = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = Result(s, d.strftime("%Y-%m-%d"), "error", error=f"{type(exc).__name__}: {exc}")
            results.append(result)
            if i % 250 == 0 or i == len(tasks):
                elapsed = time.monotonic() - tic
                rate = i / elapsed if elapsed else 0.0
                eta = (len(tasks) - i) / rate if rate else float("nan")
                counts = pd.Series([x.status for x in results]).value_counts().to_dict()
                print(
                    f"  {i}/{len(tasks)} | {rate:.2f} days/s | ETA {eta/60:.1f}m | {counts}",
                    flush=True,
                )
                _checkpoint(results, root, final=False)

    _checkpoint(results, root, final=True)
    counts = pd.Series([x.status for x in results]).value_counts().to_dict()
    complete = [x for x in results if x.status in {"complete", "skipped"}]
    print(
        f"done {counts} | rows {sum(x.rows for x in complete):,} | "
        f"new bytes {sum(x.bytes for x in results if x.status == 'complete')/1e9:.2f} GB | "
        f"elapsed {(time.monotonic()-tic)/60:.1f}m",
        flush=True,
    )
    print("FULLFLOWDONE")


if __name__ == "__main__":
    main()
