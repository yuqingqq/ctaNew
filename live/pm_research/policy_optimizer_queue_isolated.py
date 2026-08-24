"""Iteration 006: isolate queue-realistic policy arms on separate clocks."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import adverse_move_fast as linear
import adverse_move_harmful as v5
import flow_intensity as fi
import policy_bounds_v1 as pb
import policy_optimizer_cancel_skew as old
import policy_optimizer_queue_realistic as qr
import warning_window as ww


OUT = fi.PM / "derived/policy_optimizer_queue_isolated_v1.json"
PROTOCOL = Path(__file__).with_name("QUEUE_ARM_ISOLATION_PROTOCOL.md")
PARENT = qr.OUT
MODEL = old.MODEL_ARTIFACT
CELLS = (qr.QR_SKEW, qr.QR_BASELINE)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def replay_cells_isolated(path: Path, up_id: str, down_id: str,
                          gaps: Sequence[tuple[float, float]],
                          specs: Sequence[dict[str, Any]],
                          signals: dict[str, Sequence[tuple[float, bool]]],
                          lag_s: float = 0.0) -> dict[str, Any] | None:
    """Run each spec through a separate event loop; combine only results."""
    answer: dict[str, Any] = {}
    for spec in specs:
        got = qr.replay_cells_queue_realistic(
            path, up_id, down_id, gaps, [spec], signals, lag_s=lag_s)
        if got is None:
            return None
        cell = spec["cell"]
        answer[cell] = got[cell]
    return answer


def _specs(coin: str) -> list[dict[str, Any]]:
    latency = old.CANDIDATE_CELLS[coin]["latency_ms"]
    return [qr._qr_spec(qr.QR_SKEW, latency, False),
            qr._qr_spec(qr.QR_BASELINE, latency, True)]


def _same(a: Any, b: Any) -> bool:
    return pb.conformant(a, b) and a.diagnostics == b.diagnostics


def run() -> dict[str, Any]:
    if not PARENT.exists() or not MODEL.exists():
        raise RuntimeError("parent queue/model artifact missing")
    parent = json.loads(PARENT.read_text())
    artifact = json.loads(MODEL.read_text())
    batches, sampled, slugs = linear.build_batches(1)
    selected = {item[0]: item for _, items in ww.select_by_day(1).items()
                for item in items if item[0] in set(slugs)}
    schedules: dict[str, Any] = {}
    for batch in batches:
        schedules[batch.slug], _ = old._signals_for_batch(batch, artifact)

    windows: dict[tuple[str, str, str], list[Any]] = collections.defaultdict(list)
    for index, batch in enumerate(batches, 1):
        slug, path, up, down, gaps = selected[batch.slug]
        got = replay_cells_isolated(
            path, up, down, gaps, _specs(batch.coin), schedules[slug], 0.0)
        if got is None:
            raise RuntimeError(f"isolated replay unavailable {slug}")
        for cell, window in got.items():
            windows[(cell, batch.coin, batch.day)].append(window)
        print(f"[qr-isolated] {index}/{len(batches)} {slug}", flush=True)

    first = batches[0]
    slug, path, up, down, gaps = selected[first.slug]
    specs = _specs(first.coin)
    forward = replay_cells_isolated(
        path, up, down, gaps, specs, schedules[slug], 0.0)
    reverse = replay_cells_isolated(
        path, up, down, gaps, list(reversed(specs)), schedules[slug], 0.0)
    single = replay_cells_isolated(
        path, up, down, gaps, [specs[1]], schedules[slug], 0.0)
    repeat = replay_cells_isolated(
        path, up, down, gaps, specs, schedules[slug], 0.0)
    false = {side: [(when, False) for when, _ in rows]
             for side, rows in schedules[slug].items()}
    false_got = replay_cells_isolated(path, up, down, gaps, specs, false, 0.0)
    assert all(x is not None for x in (forward, reverse, single, repeat, false_got))
    controls = {
        "cell_order_invariant": all(
            _same(forward[cell], reverse[cell]) for cell in CELLS),
        "other_cell_presence_invariant": _same(
            forward[qr.QR_BASELINE], single[qr.QR_BASELINE]),
        "deterministic": all(
            _same(forward[cell], repeat[cell]) for cell in CELLS),
        "all_false_baseline_equals_skew": pb.conformant(
            false_got[qr.QR_BASELINE], false_got[qr.QR_SKEW]),
        "source_code_protocol_receipts_present": bool(
            sampled and PROTOCOL.exists() and PARENT.exists() and MODEL.exists()),
    }

    days = sorted({batch.day for batch in batches})
    coins = linear.COINS
    cells = {cell: {coin: {day: qr._cell_metrics(
        windows.get((cell, coin, day), []), cell)
        for day in days} for coin in coins} for cell in CELLS}
    comparison: dict[str, Any] = {}
    old_comparison: dict[str, Any] = {}
    for coin in coins:
        comparison[coin] = {}
        old_comparison[coin] = {}
        for day in days:
            comparison[coin][day] = (
                cells[qr.QR_BASELINE][coin][day]["pnl_per_window_cents"]
                - cells[qr.QR_SKEW][coin][day]["pnl_per_window_cents"])
            old_comparison[coin][day] = {
                cell: (cells[cell][coin][day]["pnl_per_window_cents"]
                       - parent["cells"][cell][coin][day]["pnl_per_window_cents"])
                for cell in CELLS}

    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "QUEUE_REALISTIC_PER_ARM_CLOCK_ISOLATION_V1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "CORRECTNESS_REBUILD",
        "verdict": "CORRECTED_BASELINE" if all(controls.values())
                   else "CORRECTION_FAILED",
        "decision_eligible": False,
        "promotion_authorized": False,
        "cells": cells,
        "baseline_minus_skew_pnl_cents": comparison,
        "delta_pnl_cents_vs_contaminated_parent": old_comparison,
        "controls": controls,
        "population": {
            "days": days,
            "training_days": list(v5.TRAIN_DAYS),
            "development_days": list(v5.HOLDOUT_DAYS),
            "selected_slugs": slugs,
        },
        "semantics": {
            "arm_clock": "ONE_STATE_SIGNAL_CANCEL_HEAP_PER_CELL",
            "post_fill_placement": "NEXT_EVENT_ON_OWN_ARM_CLOCK_UNCHANGED",
            "same_price_zero_queue": "FORBIDDEN",
            "latency": "ASSUMED_CANCEL_EFFECTIVE_NOT_MEASURED",
            "incentives": "EXCLUDED",
            "seen_development_only": True,
        },
        "provenance": {
            "code_sha256": _file_sha(Path(__file__)),
            "protocol_sha256": _file_sha(PROTOCOL),
            "parent_artifact_sha256": _file_sha(PARENT),
            "model_artifact_sha256": _file_sha(MODEL),
            "queue_engine_sha256": _file_sha(Path(qr.__file__)),
            "polymarket": fi.provenance(sampled=sampled),
            "hf_source_identity": {
                "kind": "PATH_SIZE_MTIME_RECEIPT_NOT_CONTENT_DIGEST",
                "files": linear._hf_manifest(slugs),
            },
        },
    }
    result["artifact_id"] = _sha(result)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, allow_nan=False))
    print(f"[qr-isolated] controls {controls}", flush=True)
    print(f"[qr-isolated] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0
    def ok(value: bool, name: str) -> None:
        nonlocal checks
        if not value:
            raise AssertionError(name)
        checks += 1
    ok(CELLS == ("QR_SKEW_ONLY", "QR_CANCEL_HOLD_X_SKEW"),
       "baseline pair frozen")
    ok(_specs("btc")[1]["cancel_latency_ms"] == 25,
       "BTC latency unchanged")
    ok(_specs("eth")[1]["cancel_latency_ms"] == 100,
       "ETH latency unchanged")
    ok(all(spec["queue_realistic"] for spec in _specs("btc")),
       "same-price queue-realistic semantics retained")
    ok(PROTOCOL.exists(), "protocol frozen")
    print(f"[qr-isolated] selftest OK — {checks} checks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("command", nargs="?")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.command == "run":
        run()
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
