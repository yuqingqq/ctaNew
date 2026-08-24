"""Resumable full-history builder for exact recovered reaction features.

The builder writes a separate v3 cache and never mutates v2.  Consecutive
symbol-days are processed in small chunks so the previous day's raw tail can
warm up midnight windows without downloading every archive twice.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from live.bookdepth_flow_dynamics import AGG_ROOT, _load_trades
from live.bookdepth_flow_recovery import (
    build_recovered_dynamics,
    finalize_recovered_day,
    prepare_book_snapshots,
)
from live.bookdepth_loader import _fetch_day


REPO = Path("/home/yuqing/ctaNew")
DEFAULT_OUT = REPO / "data/ml/cache/research/bookdepth_flow_all_5min_v3_recovered"


@dataclass
class Result:
    symbol: str
    day: str
    status: str
    rows: int = 0
    quality_valid_rows: int = 0
    old_gap_window_rows: int = 0
    recovered_internal_gap_rows: int = 0
    recovered_cross_day_rows: int = 0
    stale_start_rows: int = 0
    stale_end_rows: int = 0
    extreme_window_rows: int = 0
    source_day_complete: bool = False
    ask_candidates: int = 0
    bid_candidates: int = 0
    bytes: int = 0
    seconds: float = 0.0
    error: str = ""


def _all_local_symbols() -> list[str]:
    return sorted(
        p.name for p in AGG_ROOT.iterdir()
        if p.is_dir() and any(p.glob("*.parquet"))
    )


def _local_days(symbol: str, start: str | None, end: str | None) -> list[pd.Timestamp]:
    lo = pd.Timestamp(start, tz="UTC") if start else None
    hi = pd.Timestamp(end, tz="UTC") if end else None
    days: list[pd.Timestamp] = []
    for path in sorted((AGG_ROOT / symbol).glob("*.parquet")):
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


def _fetch_book(symbol: str, day: pd.Timestamp, retries: int) -> pd.DataFrame:
    for attempt in range(retries + 1):
        raw = _fetch_day(symbol, day)
        if raw is not None and not raw.empty:
            return prepare_book_snapshots([raw])
        if attempt < retries:
            time.sleep(1.0 + attempt)
    return pd.DataFrame()


def _trade_day(symbol: str, day: pd.Timestamp) -> pd.DataFrame:
    path = AGG_ROOT / symbol / f"{day:%Y-%m-%d}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return _load_trades(symbol, pd.DatetimeIndex([day]))


def _tail(frame: pd.DataFrame, cutoff: pd.Timestamp, time_column: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    if time_column is None:
        return frame[frame.index >= cutoff].copy()
    return frame[frame[time_column] >= cutoff].copy()


def _result_from_frame(
    symbol: str,
    day: pd.Timestamp,
    out: pd.DataFrame,
    path: Path,
    seconds: float,
    status: str,
) -> Result:
    suffix = "5min"
    return Result(
        symbol=symbol,
        day=day.strftime("%Y-%m-%d"),
        status=status,
        rows=len(out),
        quality_valid_rows=int(out[f"quality_valid_{suffix}"].sum()),
        old_gap_window_rows=int(out[f"any_raw_gap_{suffix}"].sum()),
        recovered_internal_gap_rows=int(out[f"recovered_internal_gap_{suffix}"].sum()),
        recovered_cross_day_rows=int(out[f"recovered_cross_day_{suffix}"].sum()),
        stale_start_rows=int((~out[f"start_endpoint_fresh_{suffix}"]).sum()),
        stale_end_rows=int((~out[f"end_endpoint_fresh_{suffix}"]).sum()),
        extreme_window_rows=int(out[f"extreme_imbalance_{suffix}"].sum()),
        source_day_complete=bool(out["source_day_complete"].all()),
        ask_candidates=int(out[f"ask_absorption_candidate_{suffix}"].sum()),
        bid_candidates=int(out[f"bid_absorption_candidate_{suffix}"].sum()),
        bytes=path.stat().st_size if path.exists() else 0,
        seconds=seconds,
    )


def _build_chunk(
    symbol: str,
    day_strings: list[str],
    *,
    root_string: str,
    retries: int,
    overwrite: bool,
    max_endpoint_staleness_seconds: float,
) -> list[Result]:
    root = Path(root_string)
    days = [pd.Timestamp(x, tz="UTC") for x in day_strings]
    results: list[Result] = []
    previous_day: pd.Timestamp | None = None
    book_tail = pd.DataFrame()
    trade_tail = pd.DataFrame()

    for day in days:
        tic = time.monotonic()
        try:
            if previous_day is None or day != previous_day + pd.Timedelta(days=1):
                context_day = day - pd.Timedelta(days=1)
                context_book = _fetch_book(symbol, context_day, retries)
                context_trades = _trade_day(symbol, context_day)
                cutoff = day - pd.Timedelta("12min")
                book_tail = _tail(context_book, cutoff)
                trade_tail = _tail(context_trades, cutoff, "transact_time")

            current_book = _fetch_book(symbol, day, retries)
            current_trades = _trade_day(symbol, day)
            path = _partition_path(root, symbol, day)
            if current_book.empty or current_trades.empty:
                results.append(Result(
                    symbol=symbol,
                    day=day.strftime("%Y-%m-%d"),
                    status="empty",
                    seconds=time.monotonic() - tic,
                    error="no overlapping bookDepth + aggTrades after retries",
                ))
                book_tail = pd.DataFrame()
                trade_tail = pd.DataFrame()
                previous_day = day
                continue

            book = pd.concat([book_tail, current_book]).sort_index()
            book = book[~book.index.duplicated(keep="last")]
            trades = pd.concat([trade_tail, current_trades], ignore_index=True)
            if not (
                trades["transact_time"].is_monotonic_increasing
                and trades["agg_trade_id"].is_monotonic_increasing
            ):
                trades = trades.sort_values(
                    ["transact_time", "agg_trade_id"], kind="mergesort"
                )
            dynamics = build_recovered_dynamics(
                symbol,
                book,
                trades,
                window="5min",
                max_endpoint_staleness_seconds=max_endpoint_staleness_seconds,
            )
            out = finalize_recovered_day(
                dynamics,
                day,
                window="5min",
                resolution="5min",
                max_endpoint_staleness_seconds=max_endpoint_staleness_seconds,
            )
            if out.empty:
                results.append(Result(
                    symbol=symbol,
                    day=day.strftime("%Y-%m-%d"),
                    status="empty",
                    seconds=time.monotonic() - tic,
                    error="no recovered rows after day filter",
                ))
            elif path.exists() and path.stat().st_size > 0 and not overwrite:
                prior = pd.read_parquet(path)
                results.append(_result_from_frame(
                    symbol, day, prior, path, time.monotonic() - tic, "skipped"
                ))
            else:
                _write_atomic(out, path)
                results.append(_result_from_frame(
                    symbol, day, out, path, time.monotonic() - tic, "complete"
                ))

            cutoff = day + pd.Timedelta(days=1) - pd.Timedelta("12min")
            book_tail = _tail(current_book, cutoff)
            trade_tail = _tail(current_trades, cutoff, "transact_time")
            previous_day = day
            del current_book, current_trades, book, trades, dynamics, out
            gc.collect()
            try:
                import pyarrow as pa
                pa.default_memory_pool().release_unused()
            except Exception:
                pass
        except Exception as exc:
            results.append(Result(
                symbol=symbol,
                day=day.strftime("%Y-%m-%d"),
                status="error",
                seconds=time.monotonic() - tic,
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
            ))
            book_tail = pd.DataFrame()
            trade_tail = pd.DataFrame()
            previous_day = day
    return results


def _chunks(days: list[pd.Timestamp], size: int) -> list[list[pd.Timestamp]]:
    return [days[i:i + size] for i in range(0, len(days), size)]


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
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk-days", type=int, default=7)
    ap.add_argument("--max-tasks-per-child", type=int, default=0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--max-endpoint-staleness-seconds", type=float, default=90.0)
    ap.add_argument("--limit", type=int, default=None, help="limit symbol-days for a smoke test")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    symbols = (
        [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
        if args.symbols else _all_local_symbols()
    )
    all_days = {s: _local_days(s, args.start, args.end) for s in symbols}
    if args.limit is not None:
        remaining = args.limit
        limited: dict[str, list[pd.Timestamp]] = {}
        for symbol in symbols:
            limited[symbol] = all_days[symbol][:remaining]
            remaining -= len(limited[symbol])
            if remaining <= 0:
                break
        all_days = limited

    root = Path(args.out)
    all_keys = {(s, d.strftime("%Y-%m-%d")) for s, ds in all_days.items() for d in ds}
    prior_by_key: dict[tuple[str, str], Result] = {}
    manifests = [p for p in [root / "_manifest.parquet", root / "_manifest.partial.parquet"] if p.exists()]
    if manifests:
        prior = pd.read_parquet(max(manifests, key=lambda p: p.stat().st_mtime))
        fields = set(Result.__dataclass_fields__)
        for record in prior.to_dict(orient="records"):
            key = (record["symbol"], record["day"])
            if key in all_keys:
                prior_by_key[key] = Result(**{k: record[k] for k in fields if k in record})

    # Atomic partitions are the source of truth during a resume. A process can
    # stop before its next manifest checkpoint, so recover those summaries.
    for symbol, days in all_days.items():
        for day in days:
            key = (symbol, day.strftime("%Y-%m-%d"))
            path = _partition_path(root, symbol, day)
            if key not in prior_by_key and path.exists() and path.stat().st_size > 0:
                prior = pd.read_parquet(path)
                prior_by_key[key] = _result_from_frame(
                    symbol, day, prior, path, 0.0, "skipped"
                )

    tasks: list[tuple[str, list[pd.Timestamp]]] = []
    for symbol, days in all_days.items():
        for chunk in _chunks(days, args.chunk_days):
            if args.overwrite or any(
                not _partition_path(root, symbol, day).exists() for day in chunk
            ):
                tasks.append((symbol, chunk))
    results = list(prior_by_key.values())
    total_days = len(all_keys)
    pending_days = sum(len(x[1]) for x in tasks)
    print(
        f"pending chunks {len(tasks):,} / days {pending_days:,} / total {total_days:,} | "
        f"symbols {len(all_days)} | workers {args.workers} | out {root}",
        flush=True,
    )
    if not tasks:
        _checkpoint(results, root, final=True)
        print("nothing pending; manifest finalized", flush=True)
        print("RECOVERYDONE", flush=True)
        return

    tic = time.monotonic()
    pool_kwargs: dict[str, int] = {"max_workers": args.workers}
    if args.max_tasks_per_child > 0:
        pool_kwargs["max_tasks_per_child"] = args.max_tasks_per_child
    with ProcessPoolExecutor(**pool_kwargs) as pool:
        futures = {
            pool.submit(
                _build_chunk,
                symbol,
                [d.strftime("%Y-%m-%d") for d in chunk],
                root_string=str(root),
                retries=args.retries,
                overwrite=args.overwrite,
                max_endpoint_staleness_seconds=args.max_endpoint_staleness_seconds,
            ): (symbol, chunk)
            for symbol, chunk in tasks
        }
        completed_days = 0
        for i, future in enumerate(as_completed(futures), 1):
            symbol, chunk = futures[future]
            try:
                chunk_results = future.result()
            except Exception as exc:
                chunk_results = [Result(
                    symbol=symbol,
                    day=day.strftime("%Y-%m-%d"),
                    status="error",
                    error=f"{type(exc).__name__}: {str(exc)[:300]}",
                ) for day in chunk]
            results.extend(chunk_results)
            completed_days += len(chunk)
            if i % 20 == 0 or i == len(tasks):
                elapsed = time.monotonic() - tic
                rate = completed_days / elapsed if elapsed else 0.0
                eta = (pending_days - completed_days) / rate if rate else float("nan")
                latest = pd.Series([x.status for x in results]).value_counts().to_dict()
                print(
                    f"  chunks {i:,}/{len(tasks):,} | days {completed_days:,}/{pending_days:,} | "
                    f"{rate:.2f} days/s | ETA {eta/60:.1f}m | {latest}",
                    flush=True,
                )
                _checkpoint(results, root, final=False)

    _checkpoint(results, root, final=True)
    final = pd.DataFrame([asdict(x) for x in results]).drop_duplicates(["symbol", "day"], keep="last")
    print(
        f"done {final.status.value_counts().to_dict()} | rows {final.rows.sum():,} | "
        f"quality {final.quality_valid_rows.sum():,} | recovered gaps "
        f"{final.recovered_internal_gap_rows.sum():,} | elapsed {(time.monotonic()-tic)/60:.1f}m",
        flush=True,
    )
    print("RECOVERYDONE", flush=True)


if __name__ == "__main__":
    main()
