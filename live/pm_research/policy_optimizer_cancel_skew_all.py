"""Offline stress test: apply the JOIN-trained cancel signal to all skew orders.

This module preserves the existing JOIN-schema-only cancel×skew receipt and
adds one comparator that also permits cancellation of FRONT placements.  The
extension is deliberately non-promotable because the v5 harmful-flow model was
not trained on FRONT outcomes.  There is no live order or venue port.

Commands::

    python3 live/pm_research/policy_optimizer_cancel_skew_all.py --selftest
    python3 live/pm_research/policy_optimizer_cancel_skew_all.py run
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
import warning_window as ww


OUT = base.fi.PM / "derived/policy_optimizer_cancel_skew_all_v2.json"
PROTOCOL = Path(__file__).with_name("CANCEL_SKEW_ALL_PROTOCOL.md")
JOIN_ONLY_ARTIFACT = base.OUT
ALL_CELL = "CANCEL_X_SKEW_ALL"
CELL_NAMES = (*base.CELL_NAMES, ALL_CELL)


def _specs(latency_ms: int) -> list[dict[str, Any]]:
    """Original frozen grid plus the sole cancel-all treatment."""
    specs = base._specs(latency_ms)
    specs.append({
        "cell": ALL_CELL,
        "placement": "SKEW_LB",
        "skew": True,
        "skew_band_shares": base.ps.SKEW_BAND_SHARES,
        "front_on_repost": False,
        "cancel": True,
        "cancel_latency_ms": latency_ms,
        "cancel_join_only": False,
        "r_cut": 0,
        "size": 5.0,
    })
    return specs


def _controls(item: Sequence[Any],
              signals: dict[str, list[tuple[float, bool]]],
              latency_ms: int) -> dict[str, Any]:
    """Retain v1 controls and pin the changed FRONT lifecycle."""
    original = base._controls(item, signals, latency_ms)
    _, path, up_id, down_id, gaps = item
    specs = _specs(latency_ms)

    false_signals = {
        side: [(when, False) for when, _ in rows]
        for side, rows in signals.items()
    }
    false_result = base.replay_cells_with_cancel(
        path, up_id, down_id, gaps, specs, false_signals,
        lag_s=base.STATE_LAG_S)
    if false_result is None or not pb.conformant(
            false_result[ALL_CELL], false_result["SKEW_ONLY"]):
        raise RuntimeError("cancel-all all-false parity failure")

    first = base.replay_cells_with_cancel(
        path, up_id, down_id, gaps, specs, signals,
        lag_s=base.STATE_LAG_S)
    second = base.replay_cells_with_cancel(
        path, up_id, down_id, gaps, specs, signals,
        lag_s=base.STATE_LAG_S)
    deterministic = bool(first is not None and second is not None and all(
        pb.fill_key(first[cell]) == pb.fill_key(second[cell])
        and first[cell].diagnostics == second[cell].diagnostics
        for cell in CELL_NAMES
    ))
    if not deterministic:
        raise RuntimeError("cancel-all deterministic replay failure")
    return {
        "original_join_schema_controls": original,
        "cancel_all_false_signal_parity": True,
        "cancel_all_deterministic": True,
        "action_schema": "EXPERIMENTAL_JOIN_TRAINED_SIGNAL_ON_JOIN_AND_FRONT",
    }


def _comparison(cells: dict[str, Any], days: Sequence[str], coin: str,
                candidate: str, baseline: str) -> dict[str, Any]:
    per_day: dict[str, float | None] = {}
    for day in days:
        candidate_value = cells[candidate][coin][day][
            "pnl_per_window_cents"]
        baseline_value = cells[baseline][coin][day][
            "pnl_per_window_cents"]
        per_day[day] = (
            None if candidate_value is None or baseline_value is None
            else float(candidate_value - baseline_value)
        )
    return {
        "per_day_delta_pnl_cents": per_day,
        "positive_all_days": bool(
            per_day and all(value is not None and value > 0
                            for value in per_day.values())),
        "positive_development_days": bool(all(
            per_day.get(day) is not None and per_day[day] > 0
            for day in base.v5.HOLDOUT_DAYS)),
    }


def run() -> dict[str, Any]:
    if not base.MODEL_ARTIFACT.exists():
        raise RuntimeError("v5 harmful-flow model artifact is missing")
    if not JOIN_ONLY_ARTIFACT.exists():
        raise RuntimeError("frozen JOIN-only cancel×skew artifact is missing")
    artifact = json.loads(base.MODEL_ARTIFACT.read_text())
    join_only_artifact = json.loads(JOIN_ONLY_ARTIFACT.read_text())
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
        signals, audit = base._signals_for_batch(batch, artifact)
        schedules[batch.slug] = signals
        schedule_audit[batch.slug] = audit

    first_batch = batches[0]
    controls = _controls(
        selected[first_batch.slug], schedules[first_batch.slug],
        base.CANDIDATE_CELLS[first_batch.coin]["latency_ms"])
    print(f"[cancel-skew-all] controls PASS {controls}", flush=True)

    windows: dict[tuple[str, str, str], list[Any]] = (
        collections.defaultdict(list))
    for index, batch in enumerate(batches, 1):
        slug, path, up_id, down_id, gaps = selected[batch.slug]
        coin, day = batch.coin, batch.day
        got = base.replay_cells_with_cancel(
            path, up_id, down_id, gaps,
            _specs(base.CANDIDATE_CELLS[coin]["latency_ms"]),
            schedules[slug], lag_s=base.STATE_LAG_S)
        if got is None:
            continue
        for cell, window in got.items():
            windows[(cell, coin, day)].append(window)
        print(
            f"[cancel-skew-all] {index}/{len(batches)} {slug}",
            flush=True)

    days = sorted({batch.day for batch in batches})
    cells: dict[str, Any] = {}
    for cell in CELL_NAMES:
        cells[cell] = {}
        for coin in opt.VERDICT_COINS:
            cells[cell][coin] = {}
            for day in days:
                cells[cell][coin][day] = base._cell_metrics(
                    windows.get((cell, coin, day), []), cell)

    comparisons: dict[str, Any] = {}
    for coin in opt.VERDICT_COINS:
        comparisons[coin] = {
            "cancel_all_vs_join_schema_cancel_x_skew": _comparison(
                cells, days, coin, ALL_CELL, "CANCEL_X_SKEW"),
            "cancel_all_vs_skew": _comparison(
                cells, days, coin, ALL_CELL, "SKEW_ONLY"),
            "cancel_all_vs_cancel_only": _comparison(
                cells, days, coin, ALL_CELL, "CANCEL_ONLY"),
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "HARMFUL_FLOW_CANCEL_ALL_X_SKEW_OFFLINE_V2",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT_DIAGNOSTIC",
        "verdict": "NON_PROMOTABLE_OUT_OF_ACTION_SCHEMA",
        "decision_eligible": False,
        "candidate_frozen": False,
        "promotion_authorized": False,
        "reasons": [
            "v5_harmful_flow_model_was_trained_on_join_not_front",
            "harmful_flow_v5_failed_its_predeclared_model_gate",
            "same_visible_three_train_two_development_days",
            "one_window_per_coin_day",
            "latency_is_assumed_not_measured_cancel_effective",
            "cancel_rejoin_cost_and_live_ack_are_unavailable",
        ],
        "candidate_cells": base.CANDIDATE_CELLS,
        "population": {
            "n_windows": len(batches),
            "days": days,
            "training_days": list(base.v5.TRAIN_DAYS),
            "development_days": list(base.v5.HOLDOUT_DAYS),
            "selected_slugs": slugs,
            "rows_by_slug": {
                batch.slug: batch.n_rows for batch in batches},
        },
        "semantics": {
            "state_lag_ms": 0,
            "decision_cooldown_ms": base.v5.fast.COOLDOWN_MS,
            "harmful_threshold": 0.5,
            "signal_rearm": "FALSE_THEN_TRUE_EDGE",
            "action_schema": (
                "EXPERIMENTAL_CANCEL_ALL_RESTING_JOIN_AND_FRONT_"
                "WITH_JOIN_TRAINED_SIGNAL"),
            "partial_fills_before_effective": "RETAINED",
            "cancel_binding": "ORDER_GENERATION",
            "repost": "BACK_OF_CURRENT_DISPLAYED_TOUCH",
            "front_cancel_repost": "JOIN_NOT_FRONT",
            "incentives": "EXCLUDED_BY_USER_DIRECTION",
            "cancel_rejoin_cost": "UNAVAILABLE",
        },
        "controls": controls,
        "signal_audit": schedule_audit,
        "cells": cells,
        "comparisons": comparisons,
        "source_join_only_artifact_id": join_only_artifact["artifact_id"],
        "provenance": {
            "polymarket": base.fi.provenance(sampled=sampled),
            "v5_model_artifact_sha256": base._file_sha(base.MODEL_ARTIFACT),
            "join_only_artifact_sha256": base._file_sha(JOIN_ONLY_ARTIFACT),
            "code_sha256": base._file_sha(Path(__file__)),
            "protocol_sha256": base._file_sha(PROTOCOL),
            "cancel_engine_sha256": base._file_sha(Path(base.__file__)),
            "policy_engine_sha256": base._file_sha(Path(opt.__file__)),
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
    print(f"[cancel-skew-all] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0

    def ok(condition: bool, name: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(name)
        checks += 1

    specs = _specs(25)
    ok(len(specs) == 6 and tuple(spec["cell"] for spec in specs) == CELL_NAMES,
       "six-arm extension preserves the original grid order")
    all_spec = specs[-1]
    ok(all_spec["cancel"] and not all_spec["cancel_join_only"],
       "cancel-all arm enables FRONT cancellation")
    ok(base._specs(25) == specs[:5],
       "original five-arm specification is unchanged")

    arm = base.CancelArm(all_spec)
    arm.buy.front = True
    arm.reposition("BUY_UP", 0.49, 9.0)
    ok(arm.placement_front["BUY_UP"] and arm.buy.qahead == 0.0,
       "test order is actually fronted")
    pending: list[tuple[float, int, int, str, int]] = []
    sequence = arm.observe_signal("BUY_UP", True, 1.0, 0, pending, 0)
    ok(sequence == 1 and len(pending) == 1,
       "harmful edge submits cancellation for a FRONT order")
    effective, _, _, side, generation = heapq.heappop(pending)
    ok(abs(effective - 1.025) < 1e-12,
       "FRONT cancellation obeys assumed latency")
    arm.apply_cancel(side, generation, (0.49, 0.51, 9.0, 8.0, 0.01))
    ok(arm.cancel_counts["effective"] == 1,
       "matching FRONT generation is cancelled")
    ok(not arm.placement_front["BUY_UP"]
       and arm.buy.resting == 5.0 and arm.buy.qahead == 9.0,
       "cancelled FRONT order reposts as JOIN behind displayed depth")

    join_schema_arm = base.CancelArm(base._specs(25)[-1])
    join_schema_arm.buy.front = True
    join_schema_arm.reposition("BUY_UP", 0.49, 9.0)
    join_pending: list[tuple[float, int, int, str, int]] = []
    join_schema_arm.observe_signal(
        "BUY_UP", True, 1.0, 0, join_pending, 0)
    ok(not join_pending, "frozen JOIN-schema arm still skips FRONT")
    ok(base.CANDIDATE_CELLS["btc"] == {
        "horizon_ms": 50, "latency_ms": 25},
       "BTC cell remains pinned")
    ok(base.CANDIDATE_CELLS["eth"] == {
        "horizon_ms": 250, "latency_ms": 100},
       "ETH cell remains pinned")
    ok(base.STATE_LAG_S == 0.0,
       "source state profile remains identical to v5")
    print(f"[cancel-skew-all] selftest OK — {checks} checks")
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
