"""Iteration 004: q hysteresis with a half-order inventory skew band.

Research only. The candidate keeps iteration 003's frozen 0.55/0.45 harmful
state and changes only the skew band from 5.0 to 2.5 shares.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any, Sequence

import policy_bounds_v1 as pb
import policy_optimizer as opt
import policy_optimizer_cancel_skew as base
import policy_optimizer_harmful_q_hysteresis as hyst
import policy_optimizer_queue_realistic as qr
import warning_window as ww


OUT = base.fi.PM / "derived/policy_optimizer_hysteresis_half_band_v1.json"
PROTOCOL = Path(__file__).with_name("HYSTERESIS_HALF_BAND_PROTOCOL.md")
SOURCE_ARTIFACT = hyst.OUT
CANDIDATE = "QR_CANCEL_QHYST_SKEW2P5"
SKEW_BAND_SHARES = 2.5
CELL_NAMES = (*hyst.CELL_NAMES, CANDIDATE)


def _candidate_spec(latency_ms: int) -> dict[str, Any]:
    spec = hyst._candidate_spec(latency_ms)
    spec.update({
        "cell": CANDIDATE,
        "skew_band_shares": SKEW_BAND_SHARES,
    })
    return spec


def replay_all_cells(
        path: Path, up_id: str, down_id: str,
        gaps: Sequence[tuple[float, float]],
        latency_ms: int, horizon_ms: int,
        raw_signals: dict[str, Sequence[tuple[float, bool]]],
        hysteresis_signals: dict[str, Sequence[tuple[float, bool]]],
        lag_s: float = base.STATE_LAG_S) -> dict[str, base.el.WindowFills] | None:
    """Replay eleven parents and the half-band candidate independently."""
    parents = hyst.replay_all_cells(
        path, up_id, down_id, gaps, latency_ms, horizon_ms,
        raw_signals, hysteresis_signals, lag_s=lag_s)
    candidate = qr.replay_cells_queue_realistic(
        path, up_id, down_id, gaps, [_candidate_spec(latency_ms)],
        hysteresis_signals, lag_s=lag_s)
    if parents is None or candidate is None:
        return None
    return {**parents, CANDIDATE: candidate[CANDIDATE]}


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _development_gate(cells: dict[str, Any], coin: str) -> dict[str, Any]:
    days = list(base.v5.HOLDOUT_DAYS)
    candidate = cells[CANDIDATE][coin]
    incumbent = cells[qr.QR_BASELINE][coin]
    skew = cells[qr.QR_SKEW][coin]
    hysteresis = cells[hyst.CANDIDATE][coin]
    deltas = {
        day: float(candidate[day]["pnl_per_window_cents"]
                   - incumbent[day]["pnl_per_window_cents"])
        for day in days
    }
    band_deltas = {
        day: float(candidate[day]["pnl_per_window_cents"]
                   - hysteresis[day]["pnl_per_window_cents"])
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
        "per_day_half_band_value_vs_hysteresis_cents": band_deltas,
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
        latency_ms: int, horizon_ms: int) -> dict[str, Any]:
    _, path, up_id, down_id, gaps = item
    reference = hyst.replay_all_cells(
        path, up_id, down_id, gaps, latency_ms, horizon_ms,
        raw_signals, hysteresis_signals, lag_s=base.STATE_LAG_S)
    candidate = replay_all_cells(
        path, up_id, down_id, gaps, latency_ms, horizon_ms,
        raw_signals, hysteresis_signals, lag_s=base.STATE_LAG_S)
    if reference is None or candidate is None:
        raise RuntimeError("half-band parent parity unavailable")
    parity = all(
        pb.fill_key(reference[cell]) == pb.fill_key(candidate[cell])
        and reference[cell].diagnostics == candidate[cell].diagnostics
        for cell in hyst.CELL_NAMES)
    if not parity:
        raise RuntimeError("half-band replay changed a parent arm")

    parent_spec = hyst._candidate_spec(latency_ms)
    five_band_spec = {**parent_spec, "skew_band_shares": 5.0}
    parent = qr.replay_cells_queue_realistic(
        path, up_id, down_id, gaps, [parent_spec], hysteresis_signals,
        lag_s=base.STATE_LAG_S)
    five_band = qr.replay_cells_queue_realistic(
        path, up_id, down_id, gaps, [five_band_spec], hysteresis_signals,
        lag_s=base.STATE_LAG_S)
    if parent is None or five_band is None or not (
            pb.fill_key(parent[hyst.CANDIDATE])
            == pb.fill_key(five_band[hyst.CANDIDATE])
            and parent[hyst.CANDIDATE].diagnostics
            == five_band[hyst.CANDIDATE].diagnostics):
        raise RuntimeError("five-share-band parent parity failure")

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
        raise RuntimeError("half-band deterministic replay failure")
    return {
        "all_eleven_parent_arms_exact_parity": True,
        "five_share_band_exact_hysteresis_parent_parity": True,
        "half_band_candidate_deterministic": True,
    }


def run() -> dict[str, Any]:
    if not base.MODEL_ARTIFACT.exists():
        raise RuntimeError("v5 harmful-flow model artifact is missing")
    if not SOURCE_ARTIFACT.exists():
        raise RuntimeError("iteration-003 source artifact is missing")
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
    score_audit: dict[str, Any] = {}
    for batch in batches:
        scores, audit = hyst._scores_for_batch(batch, model_artifact)
        raw = hyst._raw_schedule(scores)
        hysteresis, hyst_audit = hyst._hysteresis_schedule(scores)
        raw_schedules[batch.slug] = raw
        hyst_schedules[batch.slug] = hysteresis
        score_audit[batch.slug] = {**audit, "hysteresis": hyst_audit}

    first_batch = batches[0]
    first_cell = base.CANDIDATE_CELLS[first_batch.coin]
    controls = _controls(
        selected[first_batch.slug], raw_schedules[first_batch.slug],
        hyst_schedules[first_batch.slug],
        first_cell["latency_ms"], first_cell["horizon_ms"])
    print(f"[hysteresis-half-band] controls PASS {controls}", flush=True)

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
        print(f"[hysteresis-half-band] {index}/{len(batches)} {slug}",
              flush=True)

    days = sorted({batch.day for batch in batches})
    cells: dict[str, Any] = {}
    for cell in CELL_NAMES:
        cells[cell] = {}
        for coin in opt.VERDICT_COINS:
            cells[cell][coin] = {}
            for day in days:
                cells[cell][coin][day] = hyst._cell_metrics(
                    windows.get((cell, coin, day), []), cell)
    gates = {
        coin: _development_gate(cells, coin)
        for coin in opt.VERDICT_COINS
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "PM_OPT_LOOP_ITER_004_HYSTERESIS_HALF_BAND",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT_DIAGNOSTIC",
        "verdict": (
            "ADOPT_DIAGNOSTIC" if any(
                gate["verdict"] == "ADOPT_DIAGNOSTIC"
                for gate in gates.values()) else "REJECT"),
        "decision_eligible": False,
        "promotion_authorized": False,
        "iteration": 4,
        "incumbent": qr.QR_BASELINE,
        "required_comparator": qr.QR_SKEW,
        "candidate": CANDIDATE,
        "thresholds": {
            "entry_q": hyst.ENTRY_Q,
            "exit_q": hyst.EXIT_Q,
            "skew_band_shares": SKEW_BAND_SHARES,
            "quote_size_shares": 5.0,
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
            **queue_artifact["semantics"],
            "harmful_state": "PER_SIDE_Q_0_55_0_45_HYSTERESIS",
            "skew_band": "HALF_FIVE_SHARE_QUOTE",
            "skew_engages": "ABS_NET_STRICTLY_GT_2_5_SHARES",
        },
        "controls": controls,
        "gates": gates,
        "score_audit": score_audit,
        "cells": cells,
        "reasons": source_artifact["reasons"],
        "source_iteration3_artifact_id": source_artifact["artifact_id"],
        "provenance": {
            "polymarket": base.fi.provenance(sampled=sampled),
            "v5_model_artifact_sha256": base._file_sha(base.MODEL_ARTIFACT),
            "iteration3_artifact_sha256": base._file_sha(SOURCE_ARTIFACT),
            "queue_artifact_sha256": base._file_sha(qr.OUT),
            "code_sha256": base._file_sha(Path(__file__)),
            "protocol_sha256": base._file_sha(PROTOCOL),
            "hysteresis_engine_sha256": base._file_sha(Path(hyst.__file__)),
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
    print(f"[hysteresis-half-band] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0

    def ok(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    ok(SKEW_BAND_SHARES == 2.5, "half-order band is pinned")
    ok(_candidate_spec(25)["skew_band_shares"] == 2.5,
       "candidate changes only the band field")
    quote_two_ticks = (0.48, 0.50, 10.0, 11.0, 0.01)
    quote_one_tick = (0.49, 0.50, 10.0, 11.0, 0.01)

    flat = qr.QueueRealisticArm(_candidate_spec(25))
    flat.led_q_up = 2.50
    flat.apply_skew_intent()
    flat.sync_from_quote(quote_two_ticks)
    ok(flat.placement_kind == {
        "BUY_UP": qr.JOIN_EXISTING, "SELL_UP": qr.JOIN_EXISTING},
       "exact band remains two-sided JOIN")

    long_up = qr.QueueRealisticArm(_candidate_spec(25))
    long_up.led_q_up = 2.51
    long_up.apply_skew_intent()
    long_up.sync_from_quote(quote_two_ticks)
    ok(long_up.placement_kind["SELL_UP"] == qr.PRICE_IMPROVE
       and long_up.placement_kind["BUY_UP"] == qr.JOIN_EXISTING,
       "above-band long inventory improves SELL only")
    ok(abs((long_up.sell.level or 0.0) - 0.49) < 1e-12,
       "reducing SELL improves exactly one tick without crossing")

    short_up = qr.QueueRealisticArm(_candidate_spec(25))
    short_up.led_q_up = -2.51
    short_up.apply_skew_intent()
    short_up.sync_from_quote(quote_two_ticks)
    ok(short_up.placement_kind["BUY_UP"] == qr.PRICE_IMPROVE
       and short_up.placement_kind["SELL_UP"] == qr.JOIN_EXISTING,
       "below-band short inventory improves BUY only")

    tight = qr.QueueRealisticArm(_candidate_spec(25))
    tight.led_q_up = 2.51
    tight.apply_skew_intent()
    tight.sync_from_quote(quote_one_tick)
    ok(tight.placement_kind == {
        "BUY_UP": qr.JOIN_EXISTING, "SELL_UP": qr.JOIN_EXISTING},
       "one-tick spread forces both sides to JOIN")

    pending: list[tuple[float, int, int, str, int]] = []
    sequence = long_up.observe_signal(
        "SELL_UP", True, 1.0, 0, pending, 0, quote_two_ticks)
    ok(sequence == 0 and not pending,
       "reducing-side harmful signal cannot cancel")
    sequence = long_up.observe_signal(
        "BUY_UP", True, 1.1, 0, pending, sequence, quote_two_ticks)
    ok(sequence == 1 and len(pending) == 1,
       "inventory-increasing joined side remains cancel eligible")

    five = _candidate_spec(25)
    five["skew_band_shares"] = 5.0
    ok(five["skew_band_shares"] == qr.base.ps.SKEW_BAND_SHARES,
       "five-share degenerate is the parent band")
    ok(hyst.ENTRY_Q == 0.55 and hyst.EXIT_Q == 0.45,
       "hysteresis thresholds remain frozen")
    ok(base.CANDIDATE_CELLS == {
        "btc": {"horizon_ms": 50, "latency_ms": 25},
        "eth": {"horizon_ms": 250, "latency_ms": 100},
    }, "signal cells remain frozen")
    ok(qr.QR_BASELINE == "QR_CANCEL_HOLD_X_SKEW",
       "adoption incumbent remains unchanged")
    print(f"[hysteresis-half-band] selftest OK — {checks} checks")
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
