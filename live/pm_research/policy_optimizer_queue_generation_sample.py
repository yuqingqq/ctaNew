"""Iteration 008: unchanged generation model on five windows/coin/day."""

from __future__ import annotations

import argparse
import base64
import collections
import datetime as dt
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

import adverse_feature_rows as v1
import adverse_move_fast as linear
import adverse_move_harmful as v5
import flow_intensity as fi
import policy_bounds_v1 as pb
import policy_optimizer_cancel_skew as old
import policy_optimizer_queue_action_harmful as qact
import policy_optimizer_queue_generation_model as qgen
import policy_optimizer_queue_isolated as isolated
import policy_optimizer_queue_realistic as qr
import warning_window as ww


PER_COIN_DAY = 5
CANDIDATE = "QR_CANCEL_QGEN5_X_SKEW"
OUT = fi.PM / "derived/policy_optimizer_queue_generation_sample_v1.json"
PROTOCOL = Path(__file__).with_name("QUEUE_GENERATION_SAMPLE_PROTOCOL.md")


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


def _mean(cells: dict[str, Any], cell: str, coin: str,
          metric: str) -> float:
    return float(np.mean([
        cells[cell][coin][day][metric] for day in v5.HOLDOUT_DAYS]))


def run() -> dict[str, Any]:
    old_artifact = json.loads(old.MODEL_ARTIFACT.read_text())
    base_batches, sampled, slugs = linear.build_batches(PER_COIN_DAY)
    expected = PER_COIN_DAY * 2 * (
        len(v5.TRAIN_DAYS) + len(v5.HOLDOUT_DAYS))
    if len(base_batches) != expected or any(batch.n_rows == 0 for batch in base_batches):
        raise RuntimeError(
            f"frozen 50-window sample incomplete: {len(base_batches)}/{expected}")
    selected = {item[0]: item for _, items in ww.select_by_day(PER_COIN_DAY).items()
                for item in items if item[0] in set(slugs)}
    action_batches: list[qact.ActionBatch] = []
    old_schedules: dict[str, Any] = {}
    for index, base in enumerate(base_batches, 1):
        _, path, up, down, gaps = selected[base.slug]
        tape = v1.build_pm_tape(path, up, down, gaps, feature_state_lag_s=0.0)
        batch = qact.trace_action_batch(base, tape)
        if len(batch.x) == 0:
            raise RuntimeError(f"no eligible trace rows {base.slug}")
        action_batches.append(batch)
        old_schedules[base.slug], _ = old._signals_for_batch(base, old_artifact)
        print(f"[qgen5] trace {index}/{expected} {base.slug}", flush=True)

    reports: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for coin in linear.COINS:
        report, got = qgen.fit_coin(
            coin, [b for b in action_batches if b.coin == coin], old_artifact)
        reports[coin] = report
        predictions.update(got)
        print(f"[qgen5] {coin} model gate "
              f"{report['model_gate']['pass']}", flush=True)

    windows: dict[tuple[str, str, str], list[Any]] = collections.defaultdict(list)
    for index, batch in enumerate(action_batches, 1):
        slug, path, up, down, gaps = selected[batch.slug]
        latency = old.CANDIDATE_CELLS[batch.coin]["latency_ms"]
        parent_specs = [qr._qr_spec(qr.QR_SKEW, latency, False),
                        qr._qr_spec(qr.QR_BASELINE, latency, True)]
        parent = isolated.replay_cells_isolated(
            path, up, down, gaps, parent_specs, old_schedules[slug], 0.0)
        candidate = isolated.replay_cells_isolated(
            path, up, down, gaps,
            [qr._qr_spec(CANDIDATE, latency, True)],
            qact._signals(batch, predictions[slug]), 0.0)
        if parent is None or candidate is None:
            raise RuntimeError(f"isolated replay unavailable {slug}")
        for cell, window in {**parent, **candidate}.items():
            windows[(cell, batch.coin, batch.day)].append(window)
        print(f"[qgen5] replay {index}/{expected} {slug}", flush=True)

    first = action_batches[0]
    slug, path, up, down, gaps = selected[first.slug]
    latency = old.CANDIDATE_CELLS[first.coin]["latency_ms"]
    cand = qr._qr_spec(CANDIDATE, latency, True)
    skew = qr._qr_spec(qr.QR_SKEW, latency, False)
    signal = qact._signals(first, predictions[slug])
    one = isolated.replay_cells_isolated(path, up, down, gaps, [cand], signal, 0.0)
    two = isolated.replay_cells_isolated(path, up, down, gaps, [cand], signal, 0.0)
    false = {side: [(when, False) for when, _ in rows]
             for side, rows in signal.items()}
    false_got = isolated.replay_cells_isolated(
        path, up, down, gaps, [cand, skew], false, 0.0)
    controls = {
        "frozen_sample_complete": len(action_batches) == expected,
        "five_windows_each_coin_day": all(sum(
            batch.coin == coin and batch.day == day for batch in action_batches)
            == PER_COIN_DAY for coin in linear.COINS
            for day in (*v5.TRAIN_DAYS, *v5.HOLDOUT_DAYS)),
        "first_generation_keys_unique": all(
            int(qgen.first_generation_mask(batch).sum()) == len({
                (int(batch.base.maker_side_sign[int(base_i)]), int(generation))
                for base_i, generation in zip(batch.base_index, batch.generation)})
            for batch in action_batches),
        "isolated_candidate_deterministic": bool(
            one is not None and two is not None
            and isolated._same(one[CANDIDATE], two[CANDIDATE])),
        "all_false_candidate_equals_skew": bool(
            false_got is not None and pb.conformant(
                false_got[CANDIDATE], false_got[qr.QR_SKEW])),
        "model_receipts_roundtrip": True,
    }
    for report in reports.values():
        if report.get("receipt"):
            receipt = report["receipt"]
            raw = zlib.decompress(base64.b64decode(
                receipt["model_text_zlib_b64"]))
            controls["model_receipts_roundtrip"] &= bool(
                hashlib.sha256(raw).hexdigest() == receipt["model_text_sha256"])

    days = sorted({batch.day for batch in action_batches})
    cells = {cell: {coin: {day: qr._cell_metrics(
        windows.get((cell, coin, day), []), cell)
        for day in days} for coin in linear.COINS}
        for cell in (qr.QR_BASELINE, qr.QR_SKEW, CANDIDATE)}
    adoption: dict[str, Any] = {}
    for coin in linear.COINS:
        per_day = {day: (
            cells[CANDIDATE][coin][day]["pnl_per_window_cents"]
            - cells[qr.QR_BASELINE][coin][day]["pnl_per_window_cents"])
            for day in days}
        checks = {
            "model_gate": reports[coin]["model_gate"]["pass"],
            "positive_incumbent_delta_each_dev_day": all(
                per_day[d] > 0 for d in v5.HOLDOUT_DAYS),
            "dev_mean_above_qr_skew": _mean(
                cells, CANDIDATE, coin, "pnl_per_window_cents") > _mean(
                    cells, qr.QR_SKEW, coin, "pnl_per_window_cents"),
            "inventory_not_increased": _mean(
                cells, CANDIDATE, coin,
                "terminal_abs_net_mean_shares") <= _mean(
                    cells, qr.QR_BASELINE, coin,
                    "terminal_abs_net_mean_shares") + 1e-12,
            "effective_cancels_not_increased": sum(
                cells[CANDIDATE][coin][d]["cancel_effective"]
                for d in v5.HOLDOUT_DAYS) <= sum(
                    cells[qr.QR_BASELINE][coin][d]["cancel_effective"]
                    for d in v5.HOLDOUT_DAYS),
            "cancel_repost_traffic_not_increased": sum(
                cells[CANDIDATE][coin][d]["cancel_effective"]
                + cells[CANDIDATE][coin][d]["cancel_reposts"]
                for d in v5.HOLDOUT_DAYS) <= sum(
                    cells[qr.QR_BASELINE][coin][d]["cancel_effective"]
                    + cells[qr.QR_BASELINE][coin][d]["cancel_reposts"]
                    for d in v5.HOLDOUT_DAYS),
            "controls": all(controls.values()),
        }
        adoption[coin] = {
            "verdict": "ADOPT_DIAGNOSTIC" if all(checks.values()) else "REJECT",
            "checks": checks,
            "per_day_delta_pnl_cents_vs_incumbent": per_day,
        }
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "QUEUE_GENERATION_MODEL_FIVE_WINDOWS_V1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT_DIAGNOSTIC",
        "verdict": {coin: adoption[coin]["verdict"] for coin in linear.COINS},
        "decision_eligible": False,
        "promotion_authorized": False,
        "candidate": CANDIDATE,
        "model": reports,
        "controls": controls,
        "cells": cells,
        "adoption": adoption,
        "population": {
            "per_coin_day": PER_COIN_DAY,
            "n_windows": len(action_batches),
            "training_days": list(v5.TRAIN_DAYS),
            "development_days": list(v5.HOLDOUT_DAYS),
            "selected_slugs": slugs,
        },
        "semantics": {
            "only_change_vs_iteration_007": "ONE_TO_FIVE_WINDOWS_PER_COIN_DAY",
            "fit_unit": "FIRST_ELIGIBLE_ROW_PER_SLUG_SIDE_GENERATION",
            "inference": "ALL_EXACT_EVENT_ELIGIBLE_ROWS",
            "replay": "ONE_ISOLATED_ARM_PER_EVENT_LOOP",
            "forward_days_observed": 0,
        },
        "provenance": {
            "code_sha256": _file_sha(Path(__file__)),
            "protocol_sha256": _file_sha(PROTOCOL),
            "old_model_artifact_sha256": _file_sha(old.MODEL_ARTIFACT),
            "generation_engine_sha256": _file_sha(Path(qgen.__file__)),
            "action_builder_sha256": _file_sha(Path(qact.__file__)),
            "isolation_engine_sha256": _file_sha(Path(isolated.__file__)),
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
    print(f"[qgen5] controls {controls}", flush=True)
    print(f"[qgen5] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0
    def ok(value: bool, name: str) -> None:
        nonlocal checks
        if not value:
            raise AssertionError(name)
        checks += 1
    ok(PER_COIN_DAY == 5, "sample multiple frozen")
    ok(CANDIDATE == "QR_CANCEL_QGEN5_X_SKEW", "candidate frozen")
    ok(PROTOCOL.exists(), "protocol exists")
    print(f"[qgen5] selftest OK — {checks} checks")
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
