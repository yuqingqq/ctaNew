"""Iteration 003: fixed harmful-probability hysteresis on queue baseline.

Research only. The existing v5 harmful probability is converted to a per-side
Schmitt state with frozen 0.55 entry and 0.45 exit thresholds. No live venue,
order, cancel, or execution port is present.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import policy_bounds_v1 as pb
import policy_optimizer as opt
import policy_optimizer_cancel_skew as base
import policy_optimizer_exact_horizon_timer as timer
import policy_optimizer_queue_realistic as qr
import warning_window as ww


OUT = base.fi.PM / "derived/policy_optimizer_harmful_q_hysteresis_v1.json"
PROTOCOL = Path(__file__).with_name("HARMFUL_Q_HYSTERESIS_PROTOCOL.md")
SOURCE_ARTIFACT = timer.OUT
CANDIDATE = "QR_CANCEL_QHYST_X_SKEW"
ENTRY_Q = 0.55
EXIT_Q = 0.45
CELL_NAMES = (*timer.CELL_NAMES, CANDIDATE)


def _candidate_spec(latency_ms: int) -> dict[str, Any]:
    spec = qr._qr_spec(CANDIDATE, latency_ms, True)
    spec.update({
        "harmful_state": "Q_HYSTERESIS",
        "harmful_entry_q": ENTRY_Q,
        "harmful_exit_q": EXIT_Q,
    })
    return spec


def _scores_for_batch(
        batch: Any, artifact: dict[str, Any]
        ) -> tuple[dict[str, list[tuple[float, float]]], dict[str, Any]]:
    candidate = base.CANDIDATE_CELLS[batch.coin]
    horizon = str(candidate["horizon_ms"])
    latency = str(candidate["latency_ms"])
    receipts = artifact["coins"][batch.coin]["model_artifact"][
        "models_by_horizon_ms"][horizon]["latencies"][latency]
    harmful_model = base._load_booster(receipts["value_weighted_harmful_fill"])
    q = np.asarray(harmful_model.predict(batch.x), dtype=float)
    start = int(batch.slug.rsplit("-", 1)[1])
    elapsed = (
        batch.as_of_ns - start * 1_000_000_000
    ).astype(np.float64) / 1e9
    scores: dict[str, list[tuple[float, float]]] = {
        "BUY_UP": [], "SELL_UP": []}
    for index, when in enumerate(elapsed):
        side = "BUY_UP" if batch.maker_side_sign[index] > 0 else "SELL_UP"
        scores[side].append((float(when), float(q[index])))
    return scores, {
        "n_rows": int(len(q)),
        "q_mean": float(q.mean()),
        "q_min": float(q.min()),
        "q_max": float(q.max()),
        "q_exactly_entry": int(np.count_nonzero(q == ENTRY_Q)),
        "q_exactly_exit": int(np.count_nonzero(q == EXIT_Q)),
        "q_exactly_half": int(np.count_nonzero(q == 0.5)),
        "deadband_fraction": float(
            np.mean((q >= EXIT_Q) & (q <= ENTRY_Q))),
    }


def _raw_schedule(
        scores: dict[str, Sequence[tuple[float, float]]]
        ) -> dict[str, list[tuple[float, bool]]]:
    return {
        side: [(when, bool(q > 0.5)) for when, q in rows]
        for side, rows in scores.items()
    }


def _hysteresis_schedule(
        scores: dict[str, Sequence[tuple[float, float]]],
        entry_q: float = ENTRY_Q, exit_q: float = EXIT_Q
        ) -> tuple[dict[str, list[tuple[float, bool]]], dict[str, Any]]:
    if not 0.0 <= exit_q <= entry_q <= 1.0:
        raise ValueError("hysteresis thresholds must satisfy 0<=exit<=entry<=1")
    result: dict[str, list[tuple[float, bool]]] = {}
    transitions: dict[str, dict[str, int]] = {}
    harmful_rows = 0
    total_rows = 0
    for side, rows in scores.items():
        state = False
        output: list[tuple[float, bool]] = []
        enters = 0
        exits = 0
        for when, q in rows:
            prior = state
            if not state and q > entry_q:
                state = True
            elif state and q < exit_q:
                state = False
            enters += int(state and not prior)
            exits += int(prior and not state)
            harmful_rows += int(state)
            total_rows += 1
            output.append((float(when), state))
        result[side] = output
        transitions[side] = {"entries": enters, "exits": exits}
    return result, {
        "entry_q": entry_q,
        "exit_q": exit_q,
        "harmful_fraction": harmful_rows / total_rows if total_rows else None,
        "transitions": transitions,
    }


def replay_all_cells(
        path: Path, up_id: str, down_id: str,
        gaps: Sequence[tuple[float, float]],
        latency_ms: int, horizon_ms: int,
        raw_signals: dict[str, Sequence[tuple[float, bool]]],
        hysteresis_signals: dict[str, Sequence[tuple[float, bool]]],
        lag_s: float = base.STATE_LAG_S) -> dict[str, base.el.WindowFills] | None:
    """Replay ten parents and hysteresis candidate independently."""
    parents = timer.replay_all_cells(
        path, up_id, down_id, gaps, latency_ms, horizon_ms,
        raw_signals, lag_s=lag_s)
    candidate = qr.replay_cells_queue_realistic(
        path, up_id, down_id, gaps, [_candidate_spec(latency_ms)],
        hysteresis_signals, lag_s=lag_s)
    if parents is None or candidate is None:
        return None
    return {**parents, CANDIDATE: candidate[CANDIDATE]}


def _cell_metrics(windows: Sequence[Any], cell: str) -> dict[str, Any]:
    result = timer._cell_metrics(windows, cell)
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
        candidate[day]["cancel_submitted"]
        + candidate[day]["cancel_reposts"] for day in days)
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
        raw_signals: dict[str, list[tuple[float, bool]]],
        hysteresis_signals: dict[str, list[tuple[float, bool]]],
        degenerate_signals: dict[str, list[tuple[float, bool]]],
        latency_ms: int, horizon_ms: int) -> dict[str, Any]:
    _, path, up_id, down_id, gaps = item
    reference = timer.replay_all_cells(
        path, up_id, down_id, gaps, latency_ms, horizon_ms,
        raw_signals, lag_s=base.STATE_LAG_S)
    candidate = replay_all_cells(
        path, up_id, down_id, gaps, latency_ms, horizon_ms,
        raw_signals, hysteresis_signals, lag_s=base.STATE_LAG_S)
    if reference is None or candidate is None:
        raise RuntimeError("hysteresis parent parity unavailable")
    parity = all(
        pb.fill_key(reference[cell]) == pb.fill_key(candidate[cell])
        and reference[cell].diagnostics == candidate[cell].diagnostics
        for cell in timer.CELL_NAMES)
    if not parity:
        raise RuntimeError("hysteresis replay changed a parent arm")

    incumbent_spec = qr._qr_spec(qr.QR_BASELINE, latency_ms, True)
    degenerate_spec = {**incumbent_spec}
    baseline = qr.replay_cells_queue_realistic(
        path, up_id, down_id, gaps, [incumbent_spec], raw_signals,
        lag_s=base.STATE_LAG_S)
    degenerate = qr.replay_cells_queue_realistic(
        path, up_id, down_id, gaps, [degenerate_spec], degenerate_signals,
        lag_s=base.STATE_LAG_S)
    if baseline is None or degenerate is None or not (
            pb.fill_key(baseline[qr.QR_BASELINE])
            == pb.fill_key(degenerate[qr.QR_BASELINE])
            and baseline[qr.QR_BASELINE].diagnostics
            == degenerate[qr.QR_BASELINE].diagnostics):
        raise RuntimeError("q=0.5 degenerate parity failure")

    first = qr.replay_cells_queue_realistic(
        path, up_id, down_id, gaps, [_candidate_spec(latency_ms)],
        hysteresis_signals, lag_s=base.STATE_LAG_S)
    second = qr.replay_cells_queue_realistic(
        path, up_id, down_id, gaps, [_candidate_spec(latency_ms)],
        hysteresis_signals, lag_s=base.STATE_LAG_S)
    deterministic = bool(first is not None and second is not None
                         and pb.fill_key(first[CANDIDATE])
                         == pb.fill_key(second[CANDIDATE])
                         and first[CANDIDATE].diagnostics
                         == second[CANDIDATE].diagnostics)
    if not deterministic:
        raise RuntimeError("hysteresis deterministic replay failure")
    return {
        "all_ten_parent_arms_exact_parity": True,
        "q_half_degenerate_exact_incumbent_parity": True,
        "hysteresis_candidate_deterministic": True,
    }


def run() -> dict[str, Any]:
    if not base.MODEL_ARTIFACT.exists():
        raise RuntimeError("v5 harmful-flow model artifact is missing")
    if not SOURCE_ARTIFACT.exists():
        raise RuntimeError("iteration-002 source artifact is missing")
    if not qr.OUT.exists():
        raise RuntimeError("queue-realistic source artifact is missing")
    model_artifact = json.loads(base.MODEL_ARTIFACT.read_text())
    source_artifact = json.loads(SOURCE_ARTIFACT.read_text())
    queue_artifact = json.loads(qr.OUT.read_text())
    batches, sampled, slugs = base.v5.linear.build_batches(1)
    selected = {
        item[0]: item
        for _, items in ww.select_by_day(1).items()
        for item in items
        if item[0] in set(slugs)
    }
    raw_schedules: dict[str, dict[str, list[tuple[float, bool]]]] = {}
    hyst_schedules: dict[str, dict[str, list[tuple[float, bool]]]] = {}
    degenerate_schedules: dict[str, dict[str, list[tuple[float, bool]]]] = {}
    score_audit: dict[str, Any] = {}
    for batch in batches:
        scores, audit = _scores_for_batch(batch, model_artifact)
        raw = _raw_schedule(scores)
        incumbent, incumbent_audit = base._signals_for_batch(
            batch, model_artifact)
        if raw != incumbent:
            raise RuntimeError(f"raw q reconstruction mismatch {batch.slug}")
        hyst, hyst_audit = _hysteresis_schedule(scores)
        degenerate, degenerate_audit = _hysteresis_schedule(
            scores, 0.5, 0.5)
        if audit["q_exactly_half"] == 0 and degenerate != raw:
            raise RuntimeError(f"degenerate q parity mismatch {batch.slug}")
        raw_schedules[batch.slug] = raw
        hyst_schedules[batch.slug] = hyst
        degenerate_schedules[batch.slug] = degenerate
        score_audit[batch.slug] = {
            **audit,
            "incumbent": incumbent_audit,
            "hysteresis": hyst_audit,
            "degenerate": degenerate_audit,
        }

    first_batch = batches[0]
    first_cell = base.CANDIDATE_CELLS[first_batch.coin]
    controls = _controls(
        selected[first_batch.slug], raw_schedules[first_batch.slug],
        hyst_schedules[first_batch.slug],
        degenerate_schedules[first_batch.slug],
        first_cell["latency_ms"], first_cell["horizon_ms"])
    print(f"[harmful-q-hysteresis] controls PASS {controls}", flush=True)

    windows: dict[tuple[str, str, str], list[Any]] = (
        collections.defaultdict(list))
    for index, batch in enumerate(batches, 1):
        slug, path, up_id, down_id, gaps = selected[batch.slug]
        cell = base.CANDIDATE_CELLS[batch.coin]
        got = replay_all_cells(
            path, up_id, down_id, gaps,
            cell["latency_ms"], cell["horizon_ms"],
            raw_schedules[slug], hyst_schedules[slug],
            lag_s=base.STATE_LAG_S)
        if got is None:
            continue
        for arm, window in got.items():
            windows[(arm, batch.coin, batch.day)].append(window)
        print(f"[harmful-q-hysteresis] {index}/{len(batches)} {slug}",
              flush=True)

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
        "protocol": "PM_OPT_LOOP_ITER_003_HARMFUL_Q_HYSTERESIS",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT_DIAGNOSTIC",
        "verdict": (
            "ADOPT_DIAGNOSTIC" if any(
                gate["verdict"] == "ADOPT_DIAGNOSTIC"
                for gate in gates.values()) else "REJECT"),
        "decision_eligible": False,
        "promotion_authorized": False,
        "iteration": 3,
        "incumbent": qr.QR_BASELINE,
        "required_comparator": qr.QR_SKEW,
        "candidate": CANDIDATE,
        "thresholds": {"entry_q": ENTRY_Q, "exit_q": EXIT_Q},
        "population": {
            "n_windows": len(batches),
            "days": days,
            "training_days": list(base.v5.TRAIN_DAYS),
            "development_days": list(base.v5.HOLDOUT_DAYS),
            "independent_forward_days": [],
            "selected_slugs": slugs,
        },
        "semantics": {
            **queue_artifact["semantics"],
            "harmful_state": "PER_SIDE_SCHMITT_Q_STATE",
            "harmful_entry": "Q_GT_0_55",
            "harmful_exit": "Q_LT_0_45",
            "deadband": "RETAIN_PREVIOUS_STATE",
            "minimum_hold": "NONE",
        },
        "controls": controls,
        "gates": gates,
        "score_audit": score_audit,
        "cells": cells,
        "reasons": source_artifact["reasons"],
        "source_iteration2_artifact_id": source_artifact["artifact_id"],
        "provenance": {
            "polymarket": base.fi.provenance(sampled=sampled),
            "v5_model_artifact_sha256": base._file_sha(base.MODEL_ARTIFACT),
            "iteration2_artifact_sha256": base._file_sha(SOURCE_ARTIFACT),
            "queue_artifact_sha256": base._file_sha(qr.OUT),
            "code_sha256": base._file_sha(Path(__file__)),
            "protocol_sha256": base._file_sha(PROTOCOL),
            "timer_engine_sha256": base._file_sha(Path(timer.__file__)),
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
    print(f"[harmful-q-hysteresis] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0

    def ok(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    ok(ENTRY_Q == 0.55 and EXIT_Q == 0.45,
       "single symmetric deadband is pinned")
    ok(_candidate_spec(25)["harmful_state"] == "Q_HYSTERESIS",
       "candidate spec names hysteresis")
    synthetic = {
        "BUY_UP": [
            (1.0, 0.50), (1.1, 0.56), (1.2, 0.50),
            (1.3, 0.45), (1.4, 0.44), (1.5, 0.55)],
        "SELL_UP": [(1.0, 0.54), (1.1, 0.56), (1.2, 0.44)],
    }
    schedule, audit = _hysteresis_schedule(synthetic)
    ok([state for _, state in schedule["BUY_UP"]]
       == [False, True, True, True, False, False],
       "entry, deadband retention, strict exit and equality are pinned")
    ok([state for _, state in schedule["SELL_UP"]]
       == [False, True, False],
       "maker-side states evolve independently")
    ok([when for when, _ in schedule["BUY_UP"]]
       == [when for when, _ in synthetic["BUY_UP"]],
       "score timestamps are preserved exactly")
    ok(audit["transitions"]["BUY_UP"] == {"entries": 1, "exits": 1},
       "transition audit is exact")
    raw = _raw_schedule(synthetic)
    ok([state for _, state in raw["BUY_UP"]]
       == [False, True, False, False, False, True],
       "raw comparator remains strict q>0.5")
    degenerate, _ = _hysteresis_schedule({
        "BUY_UP": [(1.0, 0.49), (1.1, 0.51), (1.2, 0.49)],
        "SELL_UP": []}, 0.5, 0.5)
    ok([state for _, state in degenerate["BUY_UP"]]
       == [False, True, False],
       "degenerate thresholds match raw away from exact equality")
    try:
        _hysteresis_schedule(synthetic, 0.4, 0.6)
    except ValueError:
        invalid_rejected = True
    else:
        invalid_rejected = False
    ok(invalid_rejected, "inverted thresholds are rejected")
    ok(base.CANDIDATE_CELLS == {
        "btc": {"horizon_ms": 50, "latency_ms": 25},
        "eth": {"horizon_ms": 250, "latency_ms": 100},
    }, "signal cells remain frozen")
    ok(qr.QR_BASELINE == "QR_CANCEL_HOLD_X_SKEW",
       "adoption incumbent remains queue-realistic cancel-hold-skew")
    ok(base.STATE_LAG_S == 0.0, "source state lag remains frozen")
    print(f"[harmful-q-hysteresis] selftest OK — {checks} checks")
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
