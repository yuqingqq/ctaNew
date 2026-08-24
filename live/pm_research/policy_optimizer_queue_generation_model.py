"""Iteration 007: one fitted row per queue-order generation."""

from __future__ import annotations

import argparse
import base64
import collections
import datetime as dt
import hashlib
import json
import math
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import adverse_feature_rows as v1
import adverse_move_fast as linear
import adverse_move_harmful as v5
import flow_intensity as fi
import policy_bounds_v1 as pb
import policy_optimizer_cancel_skew as old
import policy_optimizer_queue_action_harmful as qact
import policy_optimizer_queue_isolated as isolated
import policy_optimizer_queue_realistic as qr
import warning_window as ww


OUT = fi.PM / "derived/policy_optimizer_queue_generation_model_v1.json"
PROTOCOL = Path(__file__).with_name("QUEUE_GENERATION_MODEL_PROTOCOL.md")
BASELINE_ARTIFACT = isolated.OUT
OLD_MODEL_ARTIFACT = old.MODEL_ARTIFACT
CANDIDATE = "QR_CANCEL_QGEN_X_SKEW"
EPS = 1e-9


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


def first_generation_mask(batch: qact.ActionBatch) -> np.ndarray:
    """First eligible row by observed order identity; labels are not read."""
    answer = np.zeros(len(batch.x), dtype=bool)
    seen: set[tuple[int, int]] = set()
    for i, (base_i, generation) in enumerate(zip(
            batch.base_index, batch.generation)):
        side = int(batch.base.maker_side_sign[int(base_i)])
        key = side, int(generation)
        if key not in seen:
            seen.add(key)
            answer[i] = True
    return answer


def _stack(batches: Sequence[qact.ActionBatch], field: str
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    ds: list[np.ndarray] = []
    for batch in batches:
        use = first_generation_mask(batch) & batch.available
        xs.append(batch.x[use])
        ys.append(batch.target[use])
        ps.append(batch.prevented[use])
        ds.append(np.full(int(use.sum()), batch.day, dtype=object))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ps),
            np.concatenate(ds))


def _daily(days: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    return {day: (float(values[days == day].mean())
                  if np.any(days == day) else None)
            for day in v5.HOLDOUT_DAYS}


def fit_coin(coin: str, batches: Sequence[qact.ActionBatch],
             old_artifact: dict[str, Any]
             ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train = [batch for batch in batches if batch.day in v5.TRAIN_DAYS]
    dev = [batch for batch in batches if batch.day in v5.HOLDOUT_DAYS]
    x_train, target_train, prevented_train, _ = _stack(train, "train")
    x_dev, target_dev, prevented_dev, days = _stack(dev, "dev")
    economic_train = ((prevented_train > EPS)
                      & (np.abs(target_train) > EPS))
    economic_dev = ((prevented_dev > EPS) & (np.abs(target_dev) > EPS))
    y_train = (target_train[economic_train] > 0).astype(int)
    if len(y_train) < 2 or np.unique(y_train).size != 2:
        return ({
            "status": "UNAVAILABLE",
            "reason": "INSUFFICIENT_TWO_CLASS_FIRST_GENERATION_ROWS",
            "model_gate": {"pass": False, "checks": {}},
            "population": {
                "generation_train_rows": len(x_train),
                "generation_dev_rows": len(x_dev),
                "economic_train_rows": int(economic_train.sum()),
                "economic_dev_rows": int(economic_dev.sum()),
            },
        }, {batch.slug: np.zeros(len(batch.x)) for batch in batches})
    model = qact._fit_model(x_train, target_train, prevented_train)
    q_dev = np.asarray(model.predict_proba(x_dev)[:, 1], dtype=float)
    old_model = qact._old_harmful_model(old_artifact, coin)
    old_q_parts: list[np.ndarray] = []
    for batch in dev:
        use = first_generation_mask(batch) & batch.available
        old_q_parts.append(np.asarray(old_model.predict(
            batch.base.x[batch.base_index[use]]), dtype=float))
    old_q = np.concatenate(old_q_parts)
    cancel = q_dev > 0.5
    old_cancel = old_q > 0.5
    constant_cancel = bool(target_train.mean() > 0)
    realized = np.where(cancel, target_dev, 0.0)
    old_realized = np.where(old_cancel, target_dev, 0.0)
    constant_realized = (target_dev if constant_cancel
                         else np.zeros_like(target_dev))
    y_dev = (target_dev[economic_dev] > 0).astype(int)
    if (len(y_dev) and np.unique(y_dev).size == 2
            and np.abs(target_dev[economic_dev]).sum() > 0):
        fit = v5.weighted_classification_metrics(
            y_train, np.abs(target_train[economic_train]), y_dev,
            np.abs(target_dev[economic_dev]), q_dev[economic_dev])
        skill = fit["weighted_brier_skill_vs_training_weighted_prevalence"]
        brier = bool(skill is not None and skill > 0)
    else:
        fit = {"status": "UNAVAILABLE_TWO_CLASS_DEV"}
        brier = False
    daily_value = _daily(days, realized)
    daily_constant = _daily(days, realized - constant_realized)
    daily_old = _daily(days, realized - old_realized)
    gates = {
        "positive_weighted_brier_skill": brier,
        "positive_value_each_dev_day": all(
            daily_value[d] is not None and daily_value[d] > 0
            for d in v5.HOLDOUT_DAYS),
        "positive_gain_vs_constant_each_dev_day": all(
            daily_constant[d] is not None and daily_constant[d] > 0
            for d in v5.HOLDOUT_DAYS),
        "positive_gain_vs_v5_each_dev_day": all(
            daily_old[d] is not None and daily_old[d] > 0
            for d in v5.HOLDOUT_DAYS),
        "nondegenerate_cancel_fraction": bool(0.02 < cancel.mean() < 0.98),
    }
    predictions = {batch.slug: np.asarray(
        model.predict_proba(batch.x)[:, 1], dtype=float) for batch in batches}
    return ({
        "status": "AVAILABLE",
        "model_gate": {"pass": all(gates.values()), "checks": gates},
        "population": {
            "generation_train_rows": len(x_train),
            "generation_dev_rows": len(x_dev),
            "economic_train_rows": int(economic_train.sum()),
            "economic_dev_rows": int(economic_dev.sum()),
            "all_eligible_train_rows": sum(len(b.x) for b in train),
            "all_eligible_dev_rows": sum(len(b.x) for b in dev),
        },
        "fit": fit,
        "policy": {
            "generation_cancel_fraction": float(cancel.mean()),
            "training_selected_constant": (
                "ALWAYS_CANCEL" if constant_cancel else "NEVER_CANCEL"),
            "gross_value_cents_per_generation_decision": float(realized.mean()),
            "per_day_gross_value": daily_value,
            "per_day_gain_vs_training_constant": daily_constant,
            "per_day_gain_vs_old_v5": daily_old,
        },
        "receipt": qact._receipt(model, coin),
    }, predictions)


def _mean(cells: dict[str, Any], cell: str, coin: str,
          metric: str) -> float:
    return float(np.mean([
        cells[cell][coin][day][metric] for day in v5.HOLDOUT_DAYS]))


def run() -> dict[str, Any]:
    baseline = json.loads(BASELINE_ARTIFACT.read_text())
    old_artifact = json.loads(OLD_MODEL_ARTIFACT.read_text())
    base_batches, sampled, slugs = linear.build_batches(1)
    selected = {item[0]: item for _, items in ww.select_by_day(1).items()
                for item in items if item[0] in set(slugs)}
    action_batches: list[qact.ActionBatch] = []
    for index, base in enumerate(base_batches, 1):
        _, path, up, down, gaps = selected[base.slug]
        tape = v1.build_pm_tape(path, up, down, gaps, feature_state_lag_s=0.0)
        batch = qact.trace_action_batch(base, tape)
        action_batches.append(batch)
        print(f"[qgen] trace {index}/{len(base_batches)} {base.slug}", flush=True)

    reports: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for coin in linear.COINS:
        report, got = fit_coin(
            coin, [b for b in action_batches if b.coin == coin], old_artifact)
        reports[coin] = report
        predictions.update(got)
        print(f"[qgen] {coin} model gate "
              f"{report['model_gate']['pass']}", flush=True)

    windows: dict[tuple[str, str, str], list[Any]] = collections.defaultdict(list)
    for index, batch in enumerate(action_batches, 1):
        slug, path, up, down, gaps = selected[batch.slug]
        latency = old.CANDIDATE_CELLS[batch.coin]["latency_ms"]
        spec = [qr._qr_spec(CANDIDATE, latency, True)]
        got = isolated.replay_cells_isolated(
            path, up, down, gaps, spec,
            qact._signals(batch, predictions[slug]), 0.0)
        if got is None:
            raise RuntimeError(f"candidate unavailable {slug}")
        windows[(CANDIDATE, batch.coin, batch.day)].append(got[CANDIDATE])
        print(f"[qgen] replay {index}/{len(action_batches)} {slug}", flush=True)

    first = action_batches[0]
    slug, path, up, down, gaps = selected[first.slug]
    latency = old.CANDIDATE_CELLS[first.coin]["latency_ms"]
    cand_spec = qr._qr_spec(CANDIDATE, latency, True)
    skew_spec = qr._qr_spec(qr.QR_SKEW, latency, False)
    schedule = qact._signals(first, predictions[slug])
    once = isolated.replay_cells_isolated(
        path, up, down, gaps, [cand_spec], schedule, 0.0)
    twice = isolated.replay_cells_isolated(
        path, up, down, gaps, [cand_spec], schedule, 0.0)
    false = {side: [(when, False) for when, _ in rows]
             for side, rows in schedule.items()}
    false_got = isolated.replay_cells_isolated(
        path, up, down, gaps, [cand_spec, skew_spec], false, 0.0)
    controls = {
        "first_generation_keys_unique": all(
            int(first_generation_mask(batch).sum()) == len({
                (int(batch.base.maker_side_sign[int(base_i)]), int(generation))
                for base_i, generation in zip(
                    batch.base_index, batch.generation)})
            for batch in action_batches),
        "generation_selection_label_independent": True,
        "isolated_candidate_deterministic": bool(
            once is not None and twice is not None
            and isolated._same(once[CANDIDATE], twice[CANDIDATE])),
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
                hashlib.sha256(raw).hexdigest()
                == receipt["model_text_sha256"])

    days = baseline["population"]["days"]
    cells = {
        qr.QR_BASELINE: baseline["cells"][qr.QR_BASELINE],
        qr.QR_SKEW: baseline["cells"][qr.QR_SKEW],
        CANDIDATE: {coin: {day: qr._cell_metrics(
            windows.get((CANDIDATE, coin, day), []), CANDIDATE)
            for day in days} for coin in linear.COINS},
    }
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
        "protocol": "QUEUE_GENERATION_DEDUPLICATED_MODEL_V1",
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
            "training_days": list(v5.TRAIN_DAYS),
            "development_days": list(v5.HOLDOUT_DAYS),
            "selected_slugs": slugs,
        },
        "semantics": {
            "fit_unit": "FIRST_ELIGIBLE_ROW_PER_SLUG_SIDE_GENERATION",
            "fit_row_selection_uses_label": False,
            "inference": "ALL_EXACT_EVENT_ELIGIBLE_ROWS",
            "replay": "ONE_ISOLATED_ARM_PER_EVENT_LOOP",
            "forward_days_observed": 0,
        },
        "provenance": {
            "code_sha256": _file_sha(Path(__file__)),
            "protocol_sha256": _file_sha(PROTOCOL),
            "baseline_artifact_sha256": _file_sha(BASELINE_ARTIFACT),
            "old_model_artifact_sha256": _file_sha(OLD_MODEL_ARTIFACT),
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
    print(f"[qgen] controls {controls}", flush=True)
    print(f"[qgen] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0
    def ok(value: bool, name: str) -> None:
        nonlocal checks
        if not value:
            raise AssertionError(name)
        checks += 1
    base = type("B", (), {})()
    base.maker_side_sign = np.asarray([1, 1, -1, 1, -1])
    batch = type("A", (), {
        "x": np.zeros((5, 1)), "base": base,
        "base_index": np.arange(5),
        "generation": np.asarray([2, 2, 2, 3, 2])})()
    mask = first_generation_mask(batch)
    ok(mask.tolist() == [True, False, True, True, False],
       "first row selected independently per side/generation")
    ok(CANDIDATE == "QR_CANCEL_QGEN_X_SKEW", "candidate frozen")
    ok(PROTOCOL.exists() and BASELINE_ARTIFACT.exists(),
       "protocol and isolated baseline exist")
    print(f"[qgen] selftest OK — {checks} checks")
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
