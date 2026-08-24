"""Iteration 001: queue-realistic horizon-aligned minimum harmful hold.

Research only. The candidate differs from QR_CANCEL_HOLD_X_SKEW only by
forbidding signal-clear repost until one frozen prediction horizon has elapsed
from cancel-effective time. Inventory-reducing release stays immediate.

Commands::

    python3 live/pm_research/policy_optimizer_min_horizon_hold.py --selftest
    python3 live/pm_research/policy_optimizer_min_horizon_hold.py run
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import heapq
import json
from pathlib import Path
from typing import Any, Sequence

import policy_bounds_v1 as pb
import policy_optimizer as opt
import policy_optimizer_cancel_skew as base
import policy_optimizer_queue_realistic as qr
import warning_window as ww


OUT = base.fi.PM / "derived/policy_optimizer_min_horizon_hold_v1.json"
PROTOCOL = Path(__file__).with_name("MIN_HORIZON_HOLD_PROTOCOL.md")
SOURCE_ARTIFACT = qr.OUT
CANDIDATE = "QR_CANCEL_MINH_HOLD_X_SKEW"
CELL_NAMES = (*qr.CELL_NAMES, CANDIDATE)


def _candidate_spec(latency_ms: int, horizon_ms: int) -> dict[str, Any]:
    spec = qr._qr_spec(CANDIDATE, latency_ms, True)
    spec["minimum_hold_ms"] = int(horizon_ms)
    return spec


def _specs(latency_ms: int, horizon_ms: int) -> list[dict[str, Any]]:
    return [
        *qr._specs(latency_ms),
        _candidate_spec(latency_ms, horizon_ms),
    ]


class MinHorizonArm(qr.QueueRealisticArm):
    """Queue-realistic arm with a lower bound on signal-clear hold time."""

    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        self.minimum_hold_s = float(spec.get("minimum_hold_ms", 0)) / 1000.0
        self.minimum_hold_deadline: dict[str, float | None] = {
            side: None for side in self.SIDES}
        self.deadline_seen_while_harmful = {
            side: False for side in self.SIDES}

    def release_hold(
            self, maker_side: str, when: float,
            quote: tuple[float, float, float, float, float] | None,
            reason: str) -> bool:
        deadline = self.minimum_hold_deadline[maker_side]
        released = super().release_hold(maker_side, when, quote, reason)
        if released:
            if deadline is not None and when + 1e-12 >= deadline:
                self.cancel_counts["minimum_hold_completed_releases"] += 1
                self.cancel_counts[
                    "minimum_hold_release_lateness_microseconds"] += max(
                        0, round((when - deadline) * 1_000_000))
            self.minimum_hold_deadline[maker_side] = None
            self.deadline_seen_while_harmful[maker_side] = False
        return released

    def observe_signal(
            self, maker_side: str, harmful: bool, when: float,
            arm_index: int,
            pending: list[tuple[float, int, int, str, int]],
            sequence: int,
            quote: tuple[float, float, float, float, float] | None = None
            ) -> int:
        deadline = self.minimum_hold_deadline[maker_side]
        if (self.hold_enabled and self.held[maker_side] and not harmful
                and deadline is not None and when + 1e-12 < deadline):
            self.signal_harmful[maker_side] = False
            self.armed[maker_side] = True
            self.cancel_counts["signal_evaluations"] += 1
            self.cancel_counts["minimum_hold_early_clear_suppressed"] += 1
            return sequence
        return super().observe_signal(
            maker_side, harmful, when, arm_index, pending, sequence, quote)

    def apply_cancel(
            self, maker_side: str, generation: int,
            quote: tuple[float, float, float, float, float] | None,
            when: float | None = None) -> None:
        was_held = self.held[maker_side]
        super().apply_cancel(maker_side, generation, quote, when)
        if (not was_held and self.held[maker_side]
                and self.minimum_hold_s > 0.0):
            if when is None:
                raise ValueError("minimum hold requires effective time")
            self.minimum_hold_deadline[maker_side] = (
                when + self.minimum_hold_s)
            self.deadline_seen_while_harmful[maker_side] = False
            self.cancel_counts["minimum_hold_started"] += 1

    def release_if_reducing(
            self, when: float,
            quote: tuple[float, float, float, float, float] | None) -> None:
        super().release_if_reducing(when, quote)
        for maker_side in self.SIDES:
            deadline = self.minimum_hold_deadline[maker_side]
            if not self.held[maker_side] or deadline is None:
                continue
            if when + 1e-12 < deadline:
                continue
            if self.signal_harmful[maker_side]:
                if not self.deadline_seen_while_harmful[maker_side]:
                    self.cancel_counts[
                        "minimum_hold_deadline_harmful"] += 1
                    self.deadline_seen_while_harmful[maker_side] = True
                continue
            self.release_hold(
                maker_side, when, quote, "signal_clear_after_minimum")


def replay_cells_min_horizon(
        path: Path, up_id: str, down_id: str,
        gaps: Sequence[tuple[float, float]],
        specs: Sequence[dict[str, Any]],
        signals: dict[str, Sequence[tuple[float, bool]]],
        lag_s: float = base.STATE_LAG_S) -> dict[str, base.el.WindowFills] | None:
    """Run the audited queue event loop with the minimum-hold arm class.

    The queue runner resolves its arm class from its module global. The swap is
    scoped and restored in ``finally``; this research runner is deliberately
    single-threaded. All eight parent arms are exact-parity controls.
    """
    original = qr.QueueRealisticArm
    qr.QueueRealisticArm = MinHorizonArm
    try:
        return qr.replay_cells_queue_realistic(
            path, up_id, down_id, gaps, specs, signals, lag_s=lag_s)
    finally:
        qr.QueueRealisticArm = original


def _diagnostic_sum(windows: Sequence[Any], cell: str, name: str) -> float:
    key = f"cancel_{name}:{cell}"
    return float(sum(window.diagnostics.get(key, 0) for window in windows))


def _cell_metrics(windows: Sequence[Any], cell: str) -> dict[str, Any]:
    result = qr._cell_metrics(windows, cell)
    releases = _diagnostic_sum(
        windows, cell, "minimum_hold_completed_releases")
    lateness_us = _diagnostic_sum(
        windows, cell, "minimum_hold_release_lateness_microseconds")
    result.update({
        "minimum_hold_started": int(_diagnostic_sum(
            windows, cell, "minimum_hold_started")),
        "minimum_hold_early_clear_suppressed": int(_diagnostic_sum(
            windows, cell, "minimum_hold_early_clear_suppressed")),
        "minimum_hold_deadline_harmful": int(_diagnostic_sum(
            windows, cell, "minimum_hold_deadline_harmful")),
        "minimum_hold_completed_releases": int(releases),
        "minimum_hold_release_mean_lateness_ms": (
            lateness_us / releases / 1000.0 if releases else None),
        "cancel_repost_traffic": int(
            result["cancel_submitted"] + result["cancel_reposts"]),
    })
    return result


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _development_gate(cells: dict[str, Any], coin: str) -> dict[str, Any]:
    days = list(base.v5.HOLDOUT_DAYS)
    candidate = cells[CANDIDATE][coin]
    incumbent = cells[qr.QR_BASELINE][coin]
    skew = cells[qr.QR_SKEW][coin]
    deltas = {
        day: float(candidate[day]["pnl_per_window_cents"]
                   - incumbent[day]["pnl_per_window_cents"])
        for day in days
    }
    candidate_pnl = _mean([
        candidate[day]["pnl_per_window_cents"] for day in days])
    skew_pnl = _mean([
        skew[day]["pnl_per_window_cents"] for day in days])
    candidate_inventory = _mean([
        candidate[day]["terminal_abs_net_mean_shares"] for day in days])
    incumbent_inventory = _mean([
        incumbent[day]["terminal_abs_net_mean_shares"] for day in days])
    candidate_effective = sum(
        candidate[day]["cancel_effective"] for day in days)
    incumbent_effective = sum(
        incumbent[day]["cancel_effective"] for day in days)
    candidate_traffic = sum(
        candidate[day]["cancel_repost_traffic"] for day in days)
    incumbent_traffic = sum(
        incumbent[day]["cancel_submitted"]
        + incumbent[day]["cancel_reposts"] for day in days)
    bars = {
        "positive_incumbent_delta_both_days": all(
            value > 0 for value in deltas.values()),
        "dev2_mean_beats_qr_skew": candidate_pnl > skew_pnl,
        "terminal_abs_inventory_not_higher": (
            candidate_inventory <= incumbent_inventory + 1e-12),
        "effective_cancels_not_higher": (
            candidate_effective <= incumbent_effective),
        "cancel_repost_traffic_not_higher": (
            candidate_traffic <= incumbent_traffic),
    }
    return {
        "verdict": "ADOPT_DIAGNOSTIC" if all(bars.values()) else "REJECT",
        "bars": bars,
        "per_day_delta_vs_incumbent_cents": deltas,
        "candidate_dev2_pnl_cents": candidate_pnl,
        "qr_skew_dev2_pnl_cents": skew_pnl,
        "candidate_dev2_terminal_abs_inventory": candidate_inventory,
        "incumbent_dev2_terminal_abs_inventory": incumbent_inventory,
        "candidate_dev2_effective_cancels": candidate_effective,
        "incumbent_dev2_effective_cancels": incumbent_effective,
        "candidate_dev2_cancel_repost_traffic": candidate_traffic,
        "incumbent_dev2_cancel_repost_traffic": incumbent_traffic,
    }


def _controls(
        item: Sequence[Any],
        signals: dict[str, list[tuple[float, bool]]],
        latency_ms: int, horizon_ms: int) -> dict[str, Any]:
    _, path, up_id, down_id, gaps = item
    reference = qr.replay_cells_queue_realistic(
        path, up_id, down_id, gaps, qr._specs(latency_ms), signals,
        lag_s=base.STATE_LAG_S)
    candidate = replay_cells_min_horizon(
        path, up_id, down_id, gaps, _specs(latency_ms, horizon_ms), signals,
        lag_s=base.STATE_LAG_S)
    if reference is None or candidate is None:
        raise RuntimeError("minimum-hold parent parity unavailable")
    parity = all(
        pb.fill_key(reference[cell]) == pb.fill_key(candidate[cell])
        and reference[cell].diagnostics == candidate[cell].diagnostics
        for cell in qr.CELL_NAMES)
    if not parity:
        raise RuntimeError("minimum-hold loop changed a parent arm")

    zero_spec = qr._qr_spec(qr.QR_BASELINE, latency_ms, True)
    zero_spec["minimum_hold_ms"] = 0
    zero = replay_cells_min_horizon(
        path, up_id, down_id, gaps, [zero_spec], signals,
        lag_s=base.STATE_LAG_S)
    baseline = qr.replay_cells_queue_realistic(
        path, up_id, down_id, gaps,
        [qr._qr_spec(qr.QR_BASELINE, latency_ms, True)], signals,
        lag_s=base.STATE_LAG_S)
    if zero is None or baseline is None or not (
            pb.fill_key(zero[qr.QR_BASELINE])
            == pb.fill_key(baseline[qr.QR_BASELINE])
            and zero[qr.QR_BASELINE].diagnostics
            == baseline[qr.QR_BASELINE].diagnostics):
        raise RuntimeError("zero-minimum parity failure")

    first = replay_cells_min_horizon(
        path, up_id, down_id, gaps, _specs(latency_ms, horizon_ms), signals,
        lag_s=base.STATE_LAG_S)
    second = replay_cells_min_horizon(
        path, up_id, down_id, gaps, _specs(latency_ms, horizon_ms), signals,
        lag_s=base.STATE_LAG_S)
    deterministic = bool(first is not None and second is not None and all(
        pb.fill_key(first[cell]) == pb.fill_key(second[cell])
        and first[cell].diagnostics == second[cell].diagnostics
        for cell in CELL_NAMES))
    if not deterministic:
        raise RuntimeError("minimum-hold deterministic replay failure")
    return {
        "all_eight_parent_arms_exact_parity": True,
        "zero_minimum_exact_incumbent_parity": True,
        "deterministic": True,
    }


def run() -> dict[str, Any]:
    if not base.MODEL_ARTIFACT.exists():
        raise RuntimeError("v5 harmful-flow model artifact is missing")
    if not SOURCE_ARTIFACT.exists():
        raise RuntimeError("queue-realistic source artifact is missing")
    model_artifact = json.loads(base.MODEL_ARTIFACT.read_text())
    source_artifact = json.loads(SOURCE_ARTIFACT.read_text())
    batches, sampled, slugs = base.v5.linear.build_batches(1)
    selected = {
        item[0]: item
        for _, items in ww.select_by_day(1).items()
        for item in items
        if item[0] in set(slugs)
    }
    schedules: dict[str, dict[str, list[tuple[float, bool]]]] = {}
    schedule_audit: dict[str, Any] = {}
    for batch in batches:
        schedule, audit = base._signals_for_batch(batch, model_artifact)
        schedules[batch.slug] = schedule
        schedule_audit[batch.slug] = audit

    first_batch = batches[0]
    first_cell = base.CANDIDATE_CELLS[first_batch.coin]
    controls = _controls(
        selected[first_batch.slug], schedules[first_batch.slug],
        first_cell["latency_ms"], first_cell["horizon_ms"])
    print(f"[min-horizon-hold] controls PASS {controls}", flush=True)

    windows: dict[tuple[str, str, str], list[Any]] = (
        collections.defaultdict(list))
    for index, batch in enumerate(batches, 1):
        slug, path, up_id, down_id, gaps = selected[batch.slug]
        cell = base.CANDIDATE_CELLS[batch.coin]
        got = replay_cells_min_horizon(
            path, up_id, down_id, gaps,
            _specs(cell["latency_ms"], cell["horizon_ms"]),
            schedules[slug], lag_s=base.STATE_LAG_S)
        if got is None:
            continue
        for arm, window in got.items():
            windows[(arm, batch.coin, batch.day)].append(window)
        print(f"[min-horizon-hold] {index}/{len(batches)} {slug}", flush=True)

    days = sorted({batch.day for batch in batches})
    cells: dict[str, Any] = {}
    for cell in CELL_NAMES:
        cells[cell] = {}
        for coin in opt.VERDICT_COINS:
            cells[cell][coin] = {}
            for day in days:
                cells[cell][coin][day] = _cell_metrics(
                    windows.get((cell, coin, day), []), cell)

    gates = {
        coin: _development_gate(cells, coin)
        for coin in opt.VERDICT_COINS
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "PM_OPT_LOOP_ITER_001_MIN_HORIZON_HOLD",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT_DIAGNOSTIC",
        "verdict": (
            "ADOPT_DIAGNOSTIC" if any(
                gate["verdict"] == "ADOPT_DIAGNOSTIC"
                for gate in gates.values()) else "REJECT"),
        "decision_eligible": False,
        "promotion_authorized": False,
        "iteration": 1,
        "incumbent": qr.QR_BASELINE,
        "required_comparator": qr.QR_SKEW,
        "candidate": CANDIDATE,
        "candidate_by_coin": {
            coin: {
                "horizon_ms": base.CANDIDATE_CELLS[coin]["horizon_ms"],
                "latency_ms": base.CANDIDATE_CELLS[coin]["latency_ms"],
                "minimum_hold_ms": base.CANDIDATE_CELLS[coin]["horizon_ms"],
            }
            for coin in opt.VERDICT_COINS
        },
        "population": {
            "n_windows": len(batches),
            "days": days,
            "training_days": list(base.v5.TRAIN_DAYS),
            "development_days": list(base.v5.HOLDOUT_DAYS),
            "independent_forward_days": [],
            "selected_slugs": slugs,
        },
        "semantics": {
            **source_artifact["semantics"],
            "minimum_signal_clear_hold": "ONE_PREDICTION_HORIZON",
            "minimum_hold_clock_starts": "CANCEL_EFFECTIVE",
            "release_evaluation": "FIRST_REPLAY_EVENT_AT_OR_AFTER_DEADLINE",
            "inventory_reducing_release": "IMMEDIATE",
        },
        "controls": controls,
        "gates": gates,
        "signal_audit": schedule_audit,
        "cells": cells,
        "reasons": [
            "v5_harmful_flow_model_failed_original_gate",
            "development_days_repeatedly_visible",
            "one_window_per_coin_day",
            "latency_assumed_not_measured",
            "no_independent_forward_days",
        ],
        "source_queue_artifact_id": source_artifact["artifact_id"],
        "provenance": {
            "polymarket": base.fi.provenance(sampled=sampled),
            "v5_model_artifact_sha256": base._file_sha(base.MODEL_ARTIFACT),
            "queue_artifact_sha256": base._file_sha(SOURCE_ARTIFACT),
            "code_sha256": base._file_sha(Path(__file__)),
            "protocol_sha256": base._file_sha(PROTOCOL),
            "queue_engine_sha256": base._file_sha(Path(qr.__file__)),
            "hold_engine_sha256": base._file_sha(Path(qr.hold.__file__)),
            "feature_builder_sha256": base._file_sha(
                Path(base.v5.fast.__file__)),
            "hf_source_identity": {
                "kind": "PATH_SIZE_MTIME_RECEIPT_NOT_CONTENT_DIGEST",
                "files": base.v5.linear._hf_manifest(slugs),
            },
        },
    }
    result["artifact_id"] = base._sha(result)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, allow_nan=False))
    print(f"[min-horizon-hold] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0

    def ok(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    ok(_candidate_spec(25, 50)["minimum_hold_ms"] == 50,
       "BTC minimum hold equals prediction horizon")
    ok(_candidate_spec(100, 250)["minimum_hold_ms"] == 250,
       "ETH minimum hold equals prediction horizon")
    ok(len(_specs(25, 50)) == 9,
       "iteration adds exactly one arm to the eight-arm receipt")

    quote = (0.48, 0.50, 10.0, 11.0, 0.01)
    arm = MinHorizonArm(_candidate_spec(25, 50))
    arm.sync_from_quote(quote)
    pending: list[tuple[float, int, int, str, int]] = []
    sequence = arm.observe_signal(
        "BUY_UP", True, 1.0, 0, pending, 0, quote)
    ok(sequence == 1 and len(pending) == 1,
       "harmful joined order schedules one cancel")
    effective, _, _, side, generation = heapq.heappop(pending)
    arm.apply_cancel(side, generation, quote, effective)
    deadline = effective + 0.05
    ok(arm.held[side] and arm.side(side).level is None,
       "effective cancel enters a non-fillable hold")
    ok(abs((arm.minimum_hold_deadline[side] or 0.0) - deadline) < 1e-12,
       "minimum clock starts at cancel-effective time")

    arm.observe_signal(
        side, False, effective + 0.01, 0, pending, sequence, quote)
    ok(arm.held[side]
       and arm.cancel_counts["minimum_hold_early_clear_suppressed"] == 1,
       "early signal clear cannot repost")
    arm.release_if_reducing(deadline - 1e-9, quote)
    ok(arm.held[side], "just-before-deadline decision remains held")
    arm.release_if_reducing(deadline, quote)
    ok(not arm.held[side]
       and arm.placement_kind[side] == qr.JOIN_EXISTING,
       "clear at deadline reposts with queue-realistic placement")
    ok(arm.cancel_counts["minimum_hold_completed_releases"] == 1,
       "deadline-completed release is diagnosed")

    persistent = MinHorizonArm(_candidate_spec(25, 50))
    persistent.sync_from_quote(quote)
    persistent_pending: list[tuple[float, int, int, str, int]] = []
    seq = persistent.observe_signal(
        "BUY_UP", True, 2.0, 0, persistent_pending, 0, quote)
    effective2, _, _, side2, generation2 = heapq.heappop(
        persistent_pending)
    persistent.apply_cancel(side2, generation2, quote, effective2)
    persistent.release_if_reducing(effective2 + 0.05, quote)
    ok(persistent.held[side2]
       and persistent.cancel_counts["minimum_hold_deadline_harmful"] == 1,
       "persistent harm remains held at deadline")
    persistent.observe_signal(
        side2, False, effective2 + 0.06, 0, persistent_pending, seq, quote)
    ok(not persistent.held[side2],
       "post-deadline clear releases immediately")

    reducing = MinHorizonArm(_candidate_spec(25, 50))
    reducing.sync_from_quote(quote)
    reducing_pending: list[tuple[float, int, int, str, int]] = []
    reducing.observe_signal(
        "BUY_UP", True, 3.0, 0, reducing_pending, 0, quote)
    effective3, _, _, side3, generation3 = heapq.heappop(reducing_pending)
    reducing.apply_cancel(side3, generation3, quote, effective3)
    reducing.led_q_up = -5.1
    reducing.apply_skew_intent()
    reducing.release_if_reducing(effective3 + 0.01, quote)
    ok(not reducing.held[side3],
       "inventory-reducing transition bypasses the minimum")
    ok(reducing.placement_kind[side3] == qr.PRICE_IMPROVE,
       "early risk release uses current queue-realistic reducing placement")

    zero = MinHorizonArm({
        **qr._qr_spec(qr.QR_BASELINE, 25, True), "minimum_hold_ms": 0})
    ok(zero.minimum_hold_s == 0.0,
       "zero-minimum control is representable")
    ok(base.CANDIDATE_CELLS == {
        "btc": {"horizon_ms": 50, "latency_ms": 25},
        "eth": {"horizon_ms": 250, "latency_ms": 100},
    }, "signal cells remain frozen")
    ok(base.STATE_LAG_S == 0.0, "source state lag remains frozen")
    ok(qr.QR_BASELINE == "QR_CANCEL_HOLD_X_SKEW",
       "queue-realistic incumbent remains pinned")
    print(f"[min-horizon-hold] selftest OK — {checks} checks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.command == "run":
        selftest()
        run()
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
