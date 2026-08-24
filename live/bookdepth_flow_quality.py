"""Quality audit for the full snapshot + aggTrade dynamics research cache.

The audit is deliberately separate from the builder.  It does not repair or
rewrite any data.  It checks the complete persisted cache in symbol-sized
chunks, then rebuilds a deterministic cross-symbol sample directly from the
raw Binance Vision archive and local aggTrades.

Outputs (under the audited cache root):

    _quality_report.json
    _quality_symbol.parquet
    _quality_recompute.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from live.bookdepth_flow_dynamics import AGG_ROOT, _load_trades, build_dynamics
from live.bookdepth_flow_full_build import DEFAULT_OUT, _all_local_symbols, _local_days


WINDOW = "5min"
CORE_5MIN = [
    "return_5min",
    "buy_to_ask_5min",
    "sell_to_bid_5min",
    "ask_depth_residual_5min",
    "bid_depth_residual_5min",
]
INTERVAL_DERIVED = [
    "return",
    "bid_change",
    "ask_change",
    "ask_bid_ratio_change",
    "imb_change",
    "buy_to_ask",
    "sell_to_bid",
    "signed_pressure",
    "ask_depth_residual",
    "bid_depth_residual",
    "return_bps",
    "impact_bps_per_pressure",
]
SAMPLE_COLUMNS = [
    "return",
    "return_5min",
    "buy_to_ask",
    "sell_to_bid",
    "buy_to_ask_5min",
    "sell_to_bid_5min",
    "signed_pressure_5min",
    "ask_depth_residual_5min",
    "bid_depth_residual_5min",
    "impact_bps_per_pressure_5min",
]


def _keys_from_paths(paths: list[Path]) -> set[tuple[str, str]]:
    return {(p.parent.name, p.stem) for p in paths}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _close(a: pd.Series, b: pd.Series, *, rtol: float = 1e-9, atol: float = 1e-11) -> np.ndarray:
    return np.isclose(
        a.to_numpy(dtype=float), b.to_numpy(dtype=float),
        rtol=rtol, atol=atol, equal_nan=True,
    )


def _scan_full(root: Path, files_by_symbol: dict[str, list[Path]]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    violations: Counter[str] = Counter()
    nulls: Counter[str] = Counter()
    infs: Counter[str] = Counter()
    rows_per_file: Counter[int] = Counter()
    symbol_rows: list[dict] = []
    samples: list[pd.DataFrame] = []
    total_rows = 0
    total_files_seen = 0

    for symbol_i, (symbol, paths) in enumerate(sorted(files_by_symbol.items()), 1):
        dataset = ds.dataset([str(p) for p in paths], format="parquet")
        table = dataset.to_table(columns=["__filename", *dataset.schema.names], use_threads=True)
        d = table.to_pandas(split_blocks=True, self_destruct=True)
        del table
        total_rows += len(d)

        for col, count in d.isna().sum().items():
            nulls[col] += int(count)
        numeric_cols = d.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            infs[col] += int(np.isinf(d[col].to_numpy(dtype=float, na_value=np.nan)).sum())

        # Partition/key/timestamp structure.
        file_counts = d.groupby("__filename", observed=True).size()
        total_files_seen += len(file_counts)
        rows_per_file.update(file_counts.astype(int).tolist())
        violations["file_count_mismatch"] += abs(len(file_counts) - len(paths))
        violations["symbol_value_mismatch"] += int(d["symbol"].ne(symbol).sum())
        violations["duplicate_symbol_bar_time"] += int(d["bar_time"].duplicated().sum())
        violations["bar_not_5min_aligned"] += int((d["bar_time"].dt.floor("5min") != d["bar_time"]).sum())
        violations["snapshot_before_bar"] += int((d["snapshot_time"] < d["bar_time"]).sum())
        violations["snapshot_outside_bar"] += int(
            (d["snapshot_time"] >= d["bar_time"] + pd.Timedelta("5min")).sum()
        )
        path_day = pd.to_datetime(
            d["__filename"].str.extract(r"/(\d{4}-\d{2}-\d{2})\.parquet$", expand=False),
            utc=True,
        )
        violations["partition_day_mismatch"] += int(
            (d["bar_time"].dt.floor("1D") != path_day).sum()
        )
        violations["rows_above_288_per_day"] += int((file_counts > 288).sum())

        # Static domains and exact construction identities. Exact zero on one
        # side is preserved raw data and is quarantined by the extreme-book flag.
        for col in ["bid1", "ask1"]:
            violations[f"negative_{col}"] += int(d[col].lt(0).sum())
        violations["nonpositive_price"] += int(d["price"].le(0).sum())
        violations["imb1_out_of_bounds"] += int((d["imb1"].abs() > 1.0 + 1e-12).sum())
        for col in ["buy_quote", "sell_quote", "buy_count", "sell_count"]:
            violations[f"negative_{col}"] += int(d[col].lt(0).sum())
        violations["nonpositive_interval_seconds"] += int(d["interval_seconds"].le(0).sum())
        violations["gap_flag_disagrees_with_interval"] += int(
            (d["gap_interval"] != d["interval_seconds"].gt(90.0)).sum()
        )
        violations["imb1_identity"] += int((~_close(
            d["imb1"], (d["bid1"] - d["ask1"]) / (d["bid1"] + d["ask1"])
        )).sum())
        violations["ask_bid_ratio_identity"] += int((~_close(
            d["ask_bid_ratio"], d["ask1"] / d["bid1"].where(d["bid1"].gt(0))
        )).sum())

        # Interval and trailing-window algebra.
        identities = {
            "interval_signed_pressure_identity": (
                d["signed_pressure"], d["buy_to_ask"] - d["sell_to_bid"]
            ),
            "interval_ask_residual_identity": (
                d["ask_depth_residual"], d["ask_change"] + d["buy_to_ask"]
            ),
            "interval_bid_residual_identity": (
                d["bid_depth_residual"], d["bid_change"] + d["sell_to_bid"]
            ),
            "interval_return_bps_identity": (d["return_bps"], d["return"] * 1e4),
            "window_return_identity": (
                d["return_5min"], d["price"] / d["price_start_5min"] - 1.0
            ),
            "window_bid_change_identity": (
                d["bid_change_5min"], d["bid1"] / d["bid1_start_5min"] - 1.0
            ),
            "window_ask_change_identity": (
                d["ask_change_5min"], d["ask1"] / d["ask1_start_5min"] - 1.0
            ),
            "window_imb_change_identity": (
                d["imb_change_5min"], d["imb1"] - d["imb1_start_5min"]
            ),
            "window_ratio_change_identity": (
                d["ask_bid_ratio_change_5min"],
                d["ask_bid_ratio"] / d["ask_bid_ratio_start_5min"] - 1.0,
            ),
            "window_buy_normalization_identity": (
                d["buy_to_ask_5min"], d["buy_quote_5min"] / d["ask1_start_5min"]
            ),
            "window_sell_normalization_identity": (
                d["sell_to_bid_5min"], d["sell_quote_5min"] / d["bid1_start_5min"]
            ),
            "window_signed_pressure_identity": (
                d["signed_pressure_5min"], d["buy_to_ask_5min"] - d["sell_to_bid_5min"]
            ),
            "window_ask_residual_identity": (
                d["ask_depth_residual_5min"],
                d["ask_change_5min"] + d["buy_to_ask_5min"],
            ),
            "window_bid_residual_identity": (
                d["bid_depth_residual_5min"],
                d["bid_change_5min"] + d["sell_to_bid_5min"],
            ),
        }
        for name, (left, right) in identities.items():
            violations[name] += int((~_close(left, right)).sum())

        gap = d["gap_interval"]
        violations["gap_interval_derived_values_not_null"] += int(
            d.loc[gap, INTERVAL_DERIVED].notna().any(axis=1).sum()
        )
        violations["gap_interval_flow_not_null"] += int(
            d.loc[gap, ["buy_quote", "sell_quote", "buy_count", "sell_count"]]
            .notna().any(axis=1).sum()
        )

        core_quality = d[CORE_5MIN].notna().all(axis=1)
        expected_day_rows = d.groupby("__filename", sort=False)["bar_time"].transform("size")
        violations["source_day_bar_count_identity"] += int(
            d["source_day_bar_count"].ne(expected_day_rows).sum()
        )
        violations["source_snapshot_count_too_small"] += int(
            d["source_snapshot_count_day"].lt(d["source_day_bar_count"]).sum()
        )
        violations["source_day_complete_definition"] += int(
            (d["source_day_complete"] != d["source_day_bar_count"].ge(280)).sum()
        )
        violations["raw_gap_window_flag_definition"] += int(
            (d["any_raw_gap_5min"] != d["gap_count_5min"].gt(0)).sum()
        )
        violations["extreme_current_definition"] += int(
            (d["extreme_imbalance_1pct"] != d["imb1"].abs().gt(0.999)).sum()
        )
        extreme_window_expected = (
            d["imb1"].abs().gt(0.999) | d["imb1_start_5min"].abs().gt(0.999)
        )
        violations["extreme_window_definition"] += int(
            (d["extreme_imbalance_5min"] != extreme_window_expected).sum()
        )
        window_valid_expected = (
            core_quality & ~d["any_raw_gap_5min"] & ~d["extreme_imbalance_5min"]
        )
        violations["window_data_valid_definition"] += int(
            (d["window_data_valid_5min"] != window_valid_expected).sum()
        )
        quality_expected = window_valid_expected & d["source_day_complete"]
        violations["quality_valid_definition"] += int(
            (d["quality_valid_5min"] != quality_expected).sum()
        )

        ask_condition = (
            d["quality_valid_5min"]
            & (d["signed_pressure_5min"] >= 0.25)
            & (d["ask_depth_residual_5min"] > 0)
            & (d["return_5min"] <= 0)
        )
        bid_condition = (
            d["quality_valid_5min"]
            & (d["signed_pressure_5min"] <= -0.25)
            & (d["bid_depth_residual_5min"] > 0)
            & (d["return_5min"] >= 0)
        )
        violations["ask_candidate_definition"] += int(
            (d["ask_absorption_candidate_5min"] != ask_condition).sum()
        )
        violations["bid_candidate_definition"] += int(
            (d["bid_absorption_candidate_5min"] != bid_condition).sum()
        )

        # Daily warm-up is expected; invalid later rows are separately visible.
        row_in_file = d.groupby("__filename", sort=False).cumcount()
        first = row_in_file.eq(0)
        core_valid = d[CORE_5MIN].notna().all(axis=1)
        first_invalid = int((first & ~core_valid).sum())
        later_invalid = int((~first & ~core_valid).sum())

        optional02 = d[["bid02", "ask02", "imb02"]].notna().all(axis=1)
        symbol_rows.append({
            "symbol": symbol,
            "partitions": len(file_counts),
            "rows": len(d),
            "first_day": d["bar_time"].min(),
            "last_day": d["bar_time"].max(),
            "core_5min_valid": int(core_valid.sum()),
            "core_5min_valid_rate": float(core_valid.mean()),
            "window_data_valid": int(d["window_data_valid_5min"].sum()),
            "quality_valid": int(d["quality_valid_5min"].sum()),
            "quality_valid_rate": float(d["quality_valid_5min"].mean()),
            "source_complete_partitions": int(d.groupby("__filename")["source_day_complete"].first().sum()),
            "first_row_core_invalid": first_invalid,
            "later_row_core_invalid": later_invalid,
            "gap_rows_stored": int(gap.sum()),
            "ask_candidates": int(d["ask_absorption_candidate_5min"].sum()),
            "bid_candidates": int(d["bid_absorption_candidate_5min"].sum()),
            "band_02_valid_rate": float(optional02.mean()),
        })
        take = min(10_000, len(d))
        samples.append(d[SAMPLE_COLUMNS].sample(n=take, random_state=17_000 + symbol_i))
        print(
            f"  scan {symbol_i:02d}/{len(files_by_symbol)} {symbol}: "
            f"{len(d):,} rows, core-valid {core_valid.mean():.3%}, "
            f"stored gaps {int(gap.sum()):,}",
            flush=True,
        )
        del d

    sample = pd.concat(samples, ignore_index=True)
    q = sample.quantile([0.001, 0.01, 0.5, 0.99, 0.999]).T
    q.columns = ["p001", "p01", "p50", "p99", "p999"]
    scan = {
        "rows": total_rows,
        "files_seen": total_files_seen,
        "rows_per_file_distribution": dict(sorted(rows_per_file.items())),
        "null_counts": dict(nulls),
        "infinite_counts": dict(infs),
        "violations": dict(violations),
        "sample_rows_for_quantiles": len(sample),
    }
    return scan, pd.DataFrame(symbol_rows), q.reset_index(names="field")


def _compare_rebuild(stored: pd.DataFrame, rebuilt: pd.DataFrame) -> dict:
    stored = stored.set_index("snapshot_time").sort_index()
    rebuilt = rebuilt.sort_index()
    common = stored.index.intersection(rebuilt.index)
    result = {
        "stored_rows": len(stored),
        "rebuilt_rows": len(rebuilt),
        "common_rows": len(common),
        "timestamp_set_mismatch": len(stored.index.symmetric_difference(rebuilt.index)),
        "value_mismatches": 0,
        "max_float_abs_error": 0.0,
    }
    if len(common) == 0:
        result["value_mismatches"] = max(len(stored), len(rebuilt))
        return result
    left = stored.loc[common]
    right = rebuilt.loc[common]
    for col in stored.columns:
        if col not in right.columns:
            result["value_mismatches"] += len(common)
            continue
        if pd.api.types.is_numeric_dtype(left[col]) and not pd.api.types.is_bool_dtype(left[col]):
            a = left[col].to_numpy(dtype=float)
            b = right[col].to_numpy(dtype=float)
            close = np.isclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)
            result["value_mismatches"] += int((~close).sum())
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                result["max_float_abs_error"] = max(
                    result["max_float_abs_error"], float(np.max(np.abs(a[finite] - b[finite])))
                )
        else:
            equal = left[col].eq(right[col]) | (left[col].isna() & right[col].isna())
            result["value_mismatches"] += int((~equal).sum())
    return result


def _recompute_one(symbol: str, path: Path, retries: int = 2) -> dict:
    day = pd.Timestamp(path.stem, tz="UTC")
    base = {"symbol": symbol, "day": path.stem, "path": str(path)}
    error = ""
    raw = pd.DataFrame()
    for attempt in range(retries + 1):
        try:
            raw = build_dynamics(symbol, pd.DatetimeIndex([day]), window=WINDOW)
            if not raw.empty:
                break
            error = "empty raw rebuild"
        except Exception as exc:  # audit records failures without hiding them
            error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(1 + attempt)
    if raw.empty:
        return {**base, "status": "failed", "error": error}

    start = day.floor("1D")
    raw = raw[(raw.index >= start) & (raw.index < start + pd.Timedelta("1D"))]
    source_snapshot_count_day = len(raw)
    gap_in_window = raw["gap_interval"].rolling(WINDOW, min_periods=1).max().astype(bool)
    rebuilt = raw.groupby(raw.index.floor(WINDOW), sort=True).tail(1).copy()
    rebuilt.insert(0, "bar_time", rebuilt.index.floor(WINDOW))
    rebuilt["source_snapshot_count_day"] = source_snapshot_count_day
    rebuilt["source_day_bar_count"] = len(rebuilt)
    rebuilt["source_day_complete"] = len(rebuilt) >= 280
    rebuilt["quality_valid_5min"] = (
        rebuilt["window_data_valid_5min"] & rebuilt["source_day_complete"]
    )
    for side in ["ask", "bid"]:
        flag = f"{side}_absorption_candidate_5min"
        rebuilt[flag] = rebuilt[flag] & rebuilt["quality_valid_5min"]
    internal_gap = gap_in_window.loc[rebuilt.index] & ~rebuilt["gap_interval"]
    core_valid = rebuilt[CORE_5MIN].notna().all(axis=1)

    stored = pd.read_parquet(path)
    compared = _compare_rebuild(stored, rebuilt)

    # Independently re-sum the local raw trades assigned to non-gap intervals.
    trades = _load_trades(symbol, pd.DatetimeIndex([day]))
    flow_buy_error = np.nan
    flow_sell_error = np.nan
    eligible_trade_rows = 0
    if not trades.empty and len(raw) > 1:
        snap_ns = raw.index.as_unit("ns").asi8
        trade_ns = trades["transact_time"].array.asi8
        pos = np.searchsorted(snap_ns, trade_ns, side="left")
        valid = (pos > 0) & (pos < len(raw))
        valid_idx = np.flatnonzero(valid)
        keep = ~raw["gap_interval"].to_numpy()[pos[valid]]
        eligible = valid_idx[keep]
        eligible_trade_rows = len(eligible)
        expected_buy = float(trades["buy_quote"].to_numpy()[eligible].sum())
        expected_sell = float(trades["sell_quote"].to_numpy()[eligible].sum())
        observed_buy = float(raw.loc[~raw["gap_interval"], "buy_quote"].sum())
        observed_sell = float(raw.loc[~raw["gap_interval"], "sell_quote"].sum())
        flow_buy_error = observed_buy - expected_buy
        flow_sell_error = observed_sell - expected_sell

    return {
        **base,
        "status": "ok",
        "error": "",
        **compared,
        "raw_snapshot_rows": len(raw),
        "raw_gap_intervals": int(raw["gap_interval"].sum()),
        "stored_final_interval_gaps": int(rebuilt["gap_interval"].sum()),
        "windows_with_any_raw_gap": int(gap_in_window.loc[rebuilt.index].sum()),
        "internal_gap_not_in_stored_flag": int(internal_gap.sum()),
        "internal_gap_with_core_values": int((internal_gap & core_valid).sum()),
        "internal_gap_ask_candidates": int(
            (internal_gap & rebuilt["ask_absorption_candidate_5min"]).sum()
        ),
        "internal_gap_bid_candidates": int(
            (internal_gap & rebuilt["bid_absorption_candidate_5min"]).sum()
        ),
        "eligible_trade_rows": eligible_trade_rows,
        "buy_flow_sum_error": flow_buy_error,
        "sell_flow_sum_error": flow_sell_error,
    }


def _choose_recompute_sample(files_by_symbol: dict[str, list[Path]], limit: int | None) -> list[tuple[str, Path]]:
    chosen = []
    fractions = [0.10, 0.35, 0.65, 0.90]
    for i, (symbol, paths) in enumerate(sorted(files_by_symbol.items())):
        pos = min(len(paths) - 1, int((len(paths) - 1) * fractions[i % len(fractions)]))
        chosen.append((symbol, paths[pos]))
    if limit is not None:
        chosen = chosen[:limit]
    return chosen


def _recompute_sample(
    files_by_symbol: dict[str, list[Path]], *, workers: int, limit: int | None
) -> pd.DataFrame:
    tasks = _choose_recompute_sample(files_by_symbol, limit)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_recompute_one, symbol, path): (symbol, path) for symbol, path in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            symbol, path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "symbol": symbol,
                    "day": path.stem,
                    "path": str(path),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(result)
            print(
                f"  raw {i:02d}/{len(tasks)} {symbol} {path.stem}: "
                f"{result['status']} mismatches={result.get('value_mismatches', 'NA')} "
                f"internal-gaps={result.get('internal_gap_not_in_stored_flag', 'NA')}",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(["symbol", "day"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_OUT))
    ap.add_argument("--raw-workers", type=int, default=4)
    ap.add_argument("--raw-limit", type=int, default=None, help="debug: recompute first N symbols")
    ap.add_argument("--skip-raw", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    symbols = _all_local_symbols()
    paths = sorted(root.glob("*/*.parquet"))
    files_by_symbol = {s: sorted(root.joinpath(s).glob("*.parquet")) for s in symbols}
    files_by_symbol = {s: p for s, p in files_by_symbol.items() if p}
    actual_keys = _keys_from_paths(paths)
    expected_keys = {
        (symbol, day.strftime("%Y-%m-%d"))
        for symbol in symbols for day in _local_days(symbol, None, None)
    }
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)

    manifest_path = root / "_manifest.parquet"
    manifest = pd.read_parquet(manifest_path) if manifest_path.exists() else pd.DataFrame()
    manifest_empty = set()
    if not manifest.empty:
        manifest_empty = set(
            manifest.loc[manifest["status"].eq("empty"), ["symbol", "day"]]
            .itertuples(index=False, name=None)
        )

    print(
        f"coverage: expected {len(expected_keys):,}, actual {len(actual_keys):,}, "
        f"missing {len(missing):,}, extra {len(extra):,}",
        flush=True,
    )
    scan, symbol_report, quantiles = _scan_full(root, files_by_symbol)

    recompute = pd.DataFrame()
    if not args.skip_raw:
        recompute = _recompute_sample(
            files_by_symbol, workers=args.raw_workers, limit=args.raw_limit
        )

    structural_hard = {
        k: scan["violations"].get(k, 0)
        for k in [
            "file_count_mismatch", "symbol_value_mismatch", "duplicate_symbol_bar_time",
            "bar_not_5min_aligned", "snapshot_before_bar", "snapshot_outside_bar",
            "partition_day_mismatch", "rows_above_288_per_day",
        ]
    }
    algebra_hard = {
        k: v for k, v in scan["violations"].items()
        if k not in structural_hard and v != 0
    }
    unexpected_missing = sorted(set(missing) - manifest_empty)
    raw_ok = recompute["status"].eq("ok") if not recompute.empty else pd.Series(dtype=bool)
    raw_failures = int((~raw_ok).sum()) if not recompute.empty else None
    raw_mismatches = (
        int(recompute.loc[raw_ok, "value_mismatches"].fillna(0).sum())
        if not recompute.empty else None
    )
    flow_error_max = None
    if not recompute.empty and raw_ok.any():
        flow_error_max = float(
            recompute.loc[raw_ok, ["buy_flow_sum_error", "sell_flow_sum_error"]]
            .abs().max().max()
        )
    hard_pass = (
        not unexpected_missing
        and not extra
        and sum(structural_hard.values()) == 0
        and not algebra_hard
        and (raw_failures in {0, None})
        and (raw_mismatches in {0, None})
        and sum(scan["infinite_counts"].values()) == 0
        and (flow_error_max is None or flow_error_max < 1e-3)
    )

    total_core_invalid = int(
        symbol_report["rows"].sum() - symbol_report["core_5min_valid"].sum()
    )
    first_invalid = int(symbol_report["first_row_core_invalid"].sum())
    total_window_valid = int(symbol_report["window_data_valid"].sum())
    total_quality_valid = int(symbol_report["quality_valid"].sum())
    source_complete_partitions = int(symbol_report["source_complete_partitions"].sum())
    later_invalid = int(symbol_report["later_row_core_invalid"].sum())
    internal_gap_windows = (
        int(recompute.loc[raw_ok, "internal_gap_not_in_stored_flag"].fillna(0).sum())
        if not recompute.empty else None
    )
    report = {
        "audit_version": 2,
        "root": str(root),
        "verdict": "conditional_pass" if hard_pass else "fail",
        "interpretation": (
            "No structural, algebraic, or sampled raw-source corruption found; "
            "use only rows passing the documented validity filters."
            if hard_pass else
            "At least one hard structural, algebraic, or raw-source validation failed."
        ),
        "coverage": {
            "symbols_expected": len(symbols),
            "symbols_actual": len(files_by_symbol),
            "expected_symbol_days_from_local_aggtrades": len(expected_keys),
            "actual_partitions": len(actual_keys),
            "missing_partitions": len(missing),
            "missing_matching_manifest_source_empty": len(set(missing) & manifest_empty),
            "unexpected_missing_partitions": len(unexpected_missing),
            "extra_partitions": len(extra),
            "missing_by_date": dict(Counter(day for _, day in missing)),
            "unexpected_missing_examples": unexpected_missing[:20],
            "extra_examples": extra[:20],
        },
        "full_scan": scan,
        "validity": {
            "core_5min_fields": CORE_5MIN,
            "core_5min_invalid_rows": total_core_invalid,
            "expected_daily_first_row_invalid": first_invalid,
            "later_core_invalid_rows": later_invalid,
            "window_data_valid_rows": total_window_valid,
            "quality_valid_rows": total_quality_valid,
            "source_complete_partitions": source_complete_partitions,
            "note": "Daily partitions are independent; the first persisted 5-minute row has no prior-day warm-up.",
        },
        "structural_hard_violations": structural_hard,
        "other_nonzero_violations": algebra_hard,
        "sample_quantiles": quantiles.set_index("field").to_dict(orient="index"),
        "raw_recompute": {
            "requested": 0 if args.skip_raw else len(_choose_recompute_sample(files_by_symbol, args.raw_limit)),
            "successful": int(raw_ok.sum()) if not recompute.empty else 0,
            "failures": raw_failures,
            "stored_value_mismatches": raw_mismatches,
            "max_trade_flow_sum_abs_error": flow_error_max,
            "sampled_windows_with_internal_gap_not_in_stored_gap_flag": internal_gap_windows,
            "note": (
                "gap_interval describes only the retained final 30-second interval; "
                "any_raw_gap_5min now quarantines a gap anywhere in the full trailing window."
            ),
        },
        "required_filters": [
            "Use only rows where quality_valid_5min is true.",
            "Enforce point-in-time cross-sectional universe coverage at every decision time.",
            "Winsorize or rank-transform impact_bps_per_pressure_5min before modeling; small pressure denominators create long tails.",
            "Treat depth residuals as displayed-depth proxies, not exact queue replenishment, because cancellations and moving bands are unobserved.",
        ],
    }

    symbol_path = root / "_quality_symbol.parquet"
    recompute_path = root / "_quality_recompute.parquet"
    report_path = root / "_quality_report.json"
    symbol_report.to_parquet(symbol_path, index=False)
    if not recompute.empty:
        recompute.to_parquet(recompute_path, index=False)
    with report_path.open("w") as handle:
        json.dump(_jsonable(report), handle, indent=2, sort_keys=True)
    print(
        f"verdict {report['verdict']} | rows {scan['rows']:,} | "
        f"core-invalid {total_core_invalid:,} ({first_invalid:,} first + {later_invalid:,} later) | "
        f"raw mismatches {raw_mismatches}",
        flush=True,
    )
    print(f"wrote {report_path}", flush=True)
    print("FLOWQUALITYDONE", flush=True)


if __name__ == "__main__":
    main()
