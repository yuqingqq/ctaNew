"""Iteration 002: exact internal timer for the minimum harmful hold.

Research only. The candidate schedules a replay-internal deadline at assumed
cancel-effective time plus the frozen prediction horizon. No live venue,
order, cancel, or execution port is present.
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
import policy_optimizer_min_horizon_hold as minh
import policy_optimizer_queue_realistic as qr
import warning_window as ww


OUT = base.fi.PM / "derived/policy_optimizer_exact_horizon_timer_v1.json"
PROTOCOL = Path(__file__).with_name("EXACT_HORIZON_TIMER_PROTOCOL.md")
SOURCE_ARTIFACT = minh.OUT
CANDIDATE = "QR_CANCEL_MINH_TIMER_X_SKEW"
CELL_NAMES = (*minh.CELL_NAMES, CANDIDATE)
TIMER_GENERATION = -1


def _candidate_spec(latency_ms: int, horizon_ms: int) -> dict[str, Any]:
    spec = minh._candidate_spec(latency_ms, horizon_ms)
    spec.update({
        "cell": CANDIDATE,
        "exact_minimum_hold_timer": True,
    })
    return spec


def _specs(latency_ms: int, horizon_ms: int) -> list[dict[str, Any]]:
    return [
        *minh._specs(latency_ms, horizon_ms),
        _candidate_spec(latency_ms, horizon_ms),
    ]


class ExactTimerArm(minh.MinHorizonArm):
    """Minimum-hold arm with an internal exact-deadline event."""

    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        self.exact_timer = bool(spec.get("exact_minimum_hold_timer", False))

    def observe_signal(
            self, maker_side: str, harmful: bool, when: float,
            arm_index: int,
            pending: list[tuple[float, int, int, str, int]],
            sequence: int,
            quote: tuple[float, float, float, float, float] | None = None
            ) -> int:
        submitted_before = self.cancel_counts["submitted"]
        sequence = super().observe_signal(
            maker_side, harmful, when, arm_index, pending, sequence, quote)
        if (self.exact_timer
                and self.cancel_counts["submitted"] > submitted_before):
            sequence += 1
            heapq.heappush(pending, (
                when + self.cancel_latency_s + self.minimum_hold_s,
                sequence, arm_index, maker_side, TIMER_GENERATION))
            self.cancel_counts["exact_timer_scheduled"] += 1
        return sequence

    def apply_cancel(
            self, maker_side: str, generation: int,
            quote: tuple[float, float, float, float, float] | None,
            when: float | None = None) -> None:
        if generation != TIMER_GENERATION:
            super().apply_cancel(maker_side, generation, quote, when)
            return
        if not self.exact_timer:
            raise RuntimeError("timer generation reached a non-timer arm")
        if when is None:
            raise ValueError("exact hold timer requires event time")
        self.cancel_counts["exact_timer_fired"] += 1
        deadline = self.minimum_hold_deadline[maker_side]
        if (not self.held[maker_side] or deadline is None
                or abs(deadline - when) > 1e-9):
            self.cancel_counts["exact_timer_stale_or_no_hold"] += 1
            return
        if self.signal_harmful[maker_side]:
            self.deadline_seen_while_harmful[maker_side] = True
            self.cancel_counts["exact_timer_harmful"] += 1
            return
        self.cancel_counts["exact_timer_clear"] += 1
        self.release_hold(
            maker_side, when, quote, "signal_clear_at_exact_timer")

    def release_if_reducing(
            self, when: float,
            quote: tuple[float, float, float, float, float] | None) -> None:
        if not self.exact_timer:
            super().release_if_reducing(when, quote)
            return
        # Bypass iteration 001's next-event signal-clear release. The inherited
        # HoldArm method still performs immediate inventory-reducing release.
        qr.hold.HoldArm.release_if_reducing(self, when, quote)


def replay_cells_exact_timer(
        path: Path, up_id: str, down_id: str,
        gaps: Sequence[tuple[float, float]],
        specs: Sequence[dict[str, Any]],
        signals: dict[str, Sequence[tuple[float, bool]]],
        lag_s: float = base.STATE_LAG_S) -> dict[str, base.el.WindowFills] | None:
    """Run the audited queue loop with exact timer entries in cancel heap."""
    original = qr.QueueRealisticArm
    qr.QueueRealisticArm = ExactTimerArm
    try:
        return qr.replay_cells_queue_realistic(
            path, up_id, down_id, gaps, specs, signals, lag_s=lag_s)
    finally:
        qr.QueueRealisticArm = original


def replay_all_cells(
        path: Path, up_id: str, down_id: str,
        gaps: Sequence[tuple[float, float]],
        latency_ms: int, horizon_ms: int,
        signals: dict[str, Sequence[tuple[float, bool]]],
        lag_s: float = base.STATE_LAG_S) -> dict[str, base.el.WindowFills] | None:
    """Replay parents and timer candidate on independent internal clocks.

    The exact timer is a counterfactual event belonging only to its candidate.
    If it shared a heap with the next-event parent, its timestamp would wake and
    alter that control arm. Public market events remain identical in both runs.
    """
    parents = minh.replay_cells_min_horizon(
        path, up_id, down_id, gaps,
        minh._specs(latency_ms, horizon_ms), signals, lag_s=lag_s)
    timer = replay_cells_exact_timer(
        path, up_id, down_id, gaps,
        [_candidate_spec(latency_ms, horizon_ms)], signals, lag_s=lag_s)
    if parents is None or timer is None:
        return None
    return {**parents, CANDIDATE: timer[CANDIDATE]}


def _diagnostic_sum(windows: Sequence[Any], cell: str, name: str) -> float:
    key = f"cancel_{name}:{cell}"
    return float(sum(window.diagnostics.get(key, 0) for window in windows))


def _cell_metrics(windows: Sequence[Any], cell: str) -> dict[str, Any]:
    result = minh._cell_metrics(windows, cell)
    result.update({
        "exact_timer_scheduled": int(_diagnostic_sum(
            windows, cell, "exact_timer_scheduled")),
        "exact_timer_fired": int(_diagnostic_sum(
            windows, cell, "exact_timer_fired")),
        "exact_timer_clear": int(_diagnostic_sum(
            windows, cell, "exact_timer_clear")),
        "exact_timer_harmful": int(_diagnostic_sum(
            windows, cell, "exact_timer_harmful")),
        "exact_timer_stale_or_no_hold": int(_diagnostic_sum(
            windows, cell, "exact_timer_stale_or_no_hold")),
    })
    return result


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _development_gate(cells: dict[str, Any], coin: str) -> dict[str, Any]:
    days = list(base.v5.HOLDOUT_DAYS)
    candidate = cells[CANDIDATE][coin]
    incumbent = cells[qr.QR_BASELINE][coin]
    skew = cells[qr.QR_SKEW][coin]
    iteration1 = cells[minh.CANDIDATE][coin]
    deltas = {
        day: float(candidate[day]["pnl_per_window_cents"]
                   - incumbent[day]["pnl_per_window_cents"])
        for day in days
    }
    timer_deltas = {
        day: float(candidate[day]["pnl_per_window_cents"]
                   - iteration1[day]["pnl_per_window_cents"])
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
        "per_day_exact_timer_value_vs_iteration1_cents": timer_deltas,
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
    reference = minh.replay_cells_min_horizon(
        path, up_id, down_id, gaps,
        minh._specs(latency_ms, horizon_ms), signals,
        lag_s=base.STATE_LAG_S)
    candidate = replay_all_cells(
        path, up_id, down_id, gaps, latency_ms, horizon_ms, signals,
        lag_s=base.STATE_LAG_S)
    if reference is None or candidate is None:
        raise RuntimeError("exact-timer parent parity unavailable")
    parity = all(
        pb.fill_key(reference[cell]) == pb.fill_key(candidate[cell])
        and reference[cell].diagnostics == candidate[cell].diagnostics
        for cell in minh.CELL_NAMES)
    if not parity:
        raise RuntimeError("exact-timer loop changed a parent arm")

    disabled_spec = minh._candidate_spec(latency_ms, horizon_ms)
    disabled = replay_cells_exact_timer(
        path, up_id, down_id, gaps, [disabled_spec], signals,
        lag_s=base.STATE_LAG_S)
    parent = minh.replay_cells_min_horizon(
        path, up_id, down_id, gaps, [disabled_spec], signals,
        lag_s=base.STATE_LAG_S)
    if disabled is None or parent is None or not (
            pb.fill_key(disabled[minh.CANDIDATE])
            == pb.fill_key(parent[minh.CANDIDATE])
            and disabled[minh.CANDIDATE].diagnostics
            == parent[minh.CANDIDATE].diagnostics):
        raise RuntimeError("disabled exact-timer parity failure")

    first = replay_all_cells(
        path, up_id, down_id, gaps, latency_ms, horizon_ms, signals,
        lag_s=base.STATE_LAG_S)
    second = replay_all_cells(
        path, up_id, down_id, gaps, latency_ms, horizon_ms, signals,
        lag_s=base.STATE_LAG_S)
    deterministic = bool(first is not None and second is not None and all(
        pb.fill_key(first[cell]) == pb.fill_key(second[cell])
        and first[cell].diagnostics == second[cell].diagnostics
        for cell in CELL_NAMES))
    if not deterministic:
        raise RuntimeError("exact-timer deterministic replay failure")
    return {
        "all_nine_parent_arms_exact_parity": True,
        "disabled_timer_exact_iteration1_parity": True,
        "deterministic": True,
    }


def run() -> dict[str, Any]:
    if not base.MODEL_ARTIFACT.exists():
        raise RuntimeError("v5 harmful-flow model artifact is missing")
    if not SOURCE_ARTIFACT.exists():
        raise RuntimeError("iteration-001 source artifact is missing")
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
    print(f"[exact-horizon-timer] controls PASS {controls}", flush=True)

    windows: dict[tuple[str, str, str], list[Any]] = (
        collections.defaultdict(list))
    for index, batch in enumerate(batches, 1):
        slug, path, up_id, down_id, gaps = selected[batch.slug]
        cell = base.CANDIDATE_CELLS[batch.coin]
        got = replay_all_cells(
            path, up_id, down_id, gaps,
            cell["latency_ms"], cell["horizon_ms"], schedules[slug],
            lag_s=base.STATE_LAG_S)
        if got is None:
            continue
        for arm, window in got.items():
            windows[(arm, batch.coin, batch.day)].append(window)
        print(f"[exact-horizon-timer] {index}/{len(batches)} {slug}", flush=True)

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
        "protocol": "PM_OPT_LOOP_ITER_002_EXACT_HORIZON_TIMER",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT_DIAGNOSTIC",
        "verdict": (
            "ADOPT_DIAGNOSTIC" if any(
                gate["verdict"] == "ADOPT_DIAGNOSTIC"
                for gate in gates.values()) else "REJECT"),
        "decision_eligible": False,
        "promotion_authorized": False,
        "iteration": 2,
        "incumbent": qr.QR_BASELINE,
        "required_comparator": qr.QR_SKEW,
        "candidate": CANDIDATE,
        "candidate_by_coin": {
            coin: {
                "horizon_ms": base.CANDIDATE_CELLS[coin]["horizon_ms"],
                "latency_ms": base.CANDIDATE_CELLS[coin]["latency_ms"],
                "minimum_hold_ms": base.CANDIDATE_CELLS[coin]["horizon_ms"],
                "exact_internal_timer": True,
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
            "release_evaluation": "EXACT_INTERNAL_DEADLINE_EVENT",
            "timer_deadline": "CANCEL_SUBMIT_PLUS_ASSUMED_L_PLUS_H",
            "timer_stale_rule": "NO_OP_UNLESS_MATCHING_HOLD_DEADLINE",
        },
        "controls": controls,
        "gates": gates,
        "signal_audit": schedule_audit,
        "cells": cells,
        "reasons": source_artifact["reasons"],
        "source_iteration1_artifact_id": source_artifact["artifact_id"],
        "provenance": {
            "polymarket": base.fi.provenance(sampled=sampled),
            "v5_model_artifact_sha256": base._file_sha(base.MODEL_ARTIFACT),
            "iteration1_artifact_sha256": base._file_sha(SOURCE_ARTIFACT),
            "code_sha256": base._file_sha(Path(__file__)),
            "protocol_sha256": base._file_sha(PROTOCOL),
            "iteration1_engine_sha256": base._file_sha(Path(minh.__file__)),
            "queue_engine_sha256": base._file_sha(Path(qr.__file__)),
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
    print(f"[exact-horizon-timer] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0

    def ok(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    ok(_candidate_spec(25, 50)["exact_minimum_hold_timer"],
       "exact timer is enabled only by candidate spec")
    ok(len(_specs(25, 50)) == 10,
       "iteration adds one arm to the nine-arm parent")
    quote = (0.48, 0.50, 10.0, 11.0, 0.01)
    arm = ExactTimerArm(_candidate_spec(25, 50))
    arm.sync_from_quote(quote)
    pending: list[tuple[float, int, int, str, int]] = []
    sequence = arm.observe_signal(
        "BUY_UP", True, 1.0, 0, pending, 0, quote)
    ok(sequence == 2 and len(pending) == 2,
       "one harmful edge schedules cancel-effective and timer events")
    effective, _, _, side, generation = heapq.heappop(pending)
    timer_when, _, _, timer_side, timer_generation = heapq.heappop(pending)
    ok(generation >= 0 and timer_generation == TIMER_GENERATION,
       "timer uses a disjoint generation sentinel")
    ok(abs(timer_when - effective - 0.05) < 1e-12,
       "timer is one prediction horizon after effective time")
    arm.apply_cancel(side, generation, quote, effective)
    arm.observe_signal(
        side, False, effective + 0.01, 0, pending, sequence, quote)
    ok(arm.held[side], "pre-timer clear remains held")
    arm.apply_cancel(timer_side, timer_generation, quote, timer_when)
    ok(not arm.held[side]
       and arm.cancel_counts["exact_timer_clear"] == 1,
       "clear signal releases at exact timer")
    ok(arm.cancel_counts[
        "minimum_hold_release_lateness_microseconds"] == 0,
       "exact timer has zero deadline lateness")

    harmful = ExactTimerArm(_candidate_spec(25, 50))
    harmful.sync_from_quote(quote)
    harmful_pending: list[tuple[float, int, int, str, int]] = []
    seq = harmful.observe_signal(
        "BUY_UP", True, 2.0, 0, harmful_pending, 0, quote)
    effective2, _, _, side2, generation2 = heapq.heappop(harmful_pending)
    timer2, _, _, _, timer_generation2 = heapq.heappop(harmful_pending)
    harmful.apply_cancel(side2, generation2, quote, effective2)
    harmful.apply_cancel(side2, timer_generation2, quote, timer2)
    ok(harmful.held[side2]
       and harmful.cancel_counts["exact_timer_harmful"] == 1,
       "harm at exact timer stays held")
    harmful.observe_signal(
        side2, False, timer2 + 0.01, 0, harmful_pending, seq, quote)
    ok(not harmful.held[side2], "later clear releases a timer-held side")

    stale = ExactTimerArm(_candidate_spec(25, 50))
    stale.apply_cancel("BUY_UP", TIMER_GENERATION, quote, 3.0)
    ok(stale.cancel_counts["exact_timer_stale_or_no_hold"] == 1,
       "timer without matching hold is a no-op")

    reducing = ExactTimerArm(_candidate_spec(25, 50))
    reducing.sync_from_quote(quote)
    reducing_pending: list[tuple[float, int, int, str, int]] = []
    reducing.observe_signal(
        "BUY_UP", True, 4.0, 0, reducing_pending, 0, quote)
    effective3, _, _, side3, generation3 = heapq.heappop(reducing_pending)
    timer3, _, _, _, timer_generation3 = heapq.heappop(reducing_pending)
    reducing.apply_cancel(side3, generation3, quote, effective3)
    reducing.led_q_up = -5.1
    reducing.apply_skew_intent()
    reducing.release_if_reducing(effective3 + 0.01, quote)
    ok(not reducing.held[side3],
       "inventory reduction releases before timer")
    reducing.apply_cancel(side3, timer_generation3, quote, timer3)
    ok(reducing.cancel_counts["exact_timer_stale_or_no_hold"] == 1,
       "timer after risk release is a no-op")

    parent = ExactTimerArm(minh._candidate_spec(25, 50))
    ok(not parent.exact_timer,
       "iteration-001 parent keeps next-event behavior")
    ok(base.CANDIDATE_CELLS == {
        "btc": {"horizon_ms": 50, "latency_ms": 25},
        "eth": {"horizon_ms": 250, "latency_ms": 100},
    }, "signal cells remain frozen")
    ok(qr.QR_BASELINE == "QR_CANCEL_HOLD_X_SKEW",
       "adoption incumbent remains original queue baseline")
    ok(base.STATE_LAG_S == 0.0, "source state lag remains frozen")
    print(f"[exact-horizon-timer] selftest OK — {checks} checks")
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
