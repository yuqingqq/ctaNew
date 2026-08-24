"""Iteration 009 model-only fit with generation-compatible leaf support."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np

import adverse_feature_rows as v1
import adverse_move_fast as linear
import adverse_move_harmful as v5
import flow_intensity as fi
import policy_optimizer_cancel_skew as old
import policy_optimizer_queue_action_harmful as qact
import policy_optimizer_queue_generation_model as qgen
import policy_optimizer_queue_generation_sample as qgen5
import warning_window as ww


MIN_CHILD_SAMPLES = 20
OUT = fi.PM / "derived/adverse_move_queue_generation_leaf_v1.json"
PROTOCOL = Path(__file__).with_name("QUEUE_GENERATION_LEAF_PROTOCOL.md")
EPS = 1e-9
PARAMS = {**v5.TREE_PARAMS, "min_child_samples": MIN_CHILD_SAMPLES}


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


def _fit(x: np.ndarray, target: np.ndarray,
         prevented: np.ndarray) -> lgb.LGBMClassifier:
    economic = (prevented > EPS) & (np.abs(target) > EPS)
    y = (target[economic] > 0).astype(int)
    if len(y) < 2 or np.unique(y).size != 2:
        raise RuntimeError("insufficient two-class generation rows")
    model = lgb.LGBMClassifier(objective="binary", class_weight=None, **PARAMS)
    model.fit(x[economic], y, sample_weight=np.abs(target[economic]))
    return model


def _receipt(model: lgb.LGBMClassifier, coin: str) -> dict[str, Any]:
    raw = model.booster_.model_to_string().encode()
    return {
        "coin": coin,
        "feature_names": list(qact.FEATURE_NAMES),
        "feature_schema_hash": qact.FEATURE_SCHEMA_HASH,
        "params": {"objective": "binary", "class_weight": None, **PARAMS},
        "model_text_sha256": hashlib.sha256(raw).hexdigest(),
        "model_text_zlib_b64": base64.b64encode(
            zlib.compress(raw, level=9)).decode(),
        "feature_importance_gain": model.booster_.feature_importance(
            importance_type="gain").astype(float).tolist(),
    }


def _daily(days: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    return {day: (float(values[days == day].mean())
                  if np.any(days == day) else None)
            for day in v5.HOLDOUT_DAYS}


def fit_coin(coin: str, batches: Sequence[qact.ActionBatch],
             old_artifact: dict[str, Any]) -> dict[str, Any]:
    train = [batch for batch in batches if batch.day in v5.TRAIN_DAYS]
    dev = [batch for batch in batches if batch.day in v5.HOLDOUT_DAYS]
    x_train, target_train, prevented_train, _ = qgen._stack(train, "train")
    x_dev, target_dev, prevented_dev, days = qgen._stack(dev, "dev")
    economic_train = ((prevented_train > EPS) & (np.abs(target_train) > EPS))
    economic_dev = ((prevented_dev > EPS) & (np.abs(target_dev) > EPS))
    model = _fit(x_train, target_train, prevented_train)
    q = np.asarray(model.predict_proba(x_dev)[:, 1], dtype=float)
    old_model = qact._old_harmful_model(old_artifact, coin)
    old_q = np.concatenate([np.asarray(old_model.predict(
        batch.base.x[batch.base_index[
            qgen.first_generation_mask(batch) & batch.available]]), dtype=float)
        for batch in dev])
    cancel, old_cancel = q > 0.5, old_q > 0.5
    constant_cancel = bool(target_train.mean() > 0)
    realized = np.where(cancel, target_dev, 0.0)
    old_realized = np.where(old_cancel, target_dev, 0.0)
    constant_realized = (target_dev if constant_cancel
                         else np.zeros_like(target_dev))
    y_train = (target_train[economic_train] > 0).astype(int)
    y_dev = (target_dev[economic_dev] > 0).astype(int)
    fit = v5.weighted_classification_metrics(
        y_train, np.abs(target_train[economic_train]), y_dev,
        np.abs(target_dev[economic_dev]), q[economic_dev])
    skill = fit["weighted_brier_skill_vs_training_weighted_prevalence"]
    daily_value = _daily(days, realized)
    daily_constant = _daily(days, realized - constant_realized)
    daily_old = _daily(days, realized - old_realized)
    gates = {
        "positive_weighted_brier_skill": bool(skill is not None and skill > 0),
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
    return {
        "model_gate": {"pass": all(gates.values()), "checks": gates},
        "population": {
            "generation_train_rows": len(x_train),
            "generation_dev_rows": len(x_dev),
            "economic_train_rows": int(economic_train.sum()),
            "economic_dev_rows": int(economic_dev.sum()),
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
        "receipt": _receipt(model, coin),
    }


def run() -> dict[str, Any]:
    old_artifact = json.loads(old.MODEL_ARTIFACT.read_text())
    bases, sampled, slugs = linear.build_batches(qgen5.PER_COIN_DAY)
    expected = 50
    if len(bases) != expected or any(batch.n_rows == 0 for batch in bases):
        raise RuntimeError("frozen 50-window sample incomplete")
    selected = {item[0]: item for _, items in ww.select_by_day(
        qgen5.PER_COIN_DAY).items() for item in items if item[0] in set(slugs)}
    batches: list[qact.ActionBatch] = []
    for index, base in enumerate(bases, 1):
        _, path, up, down, gaps = selected[base.slug]
        tape = v1.build_pm_tape(path, up, down, gaps, feature_state_lag_s=0.0)
        batch = qact.trace_action_batch(base, tape)
        if len(batch.x) == 0:
            raise RuntimeError(f"no action rows {base.slug}")
        batches.append(batch)
        print(f"[qleaf] trace {index}/{expected} {base.slug}", flush=True)
    reports = {coin: fit_coin(
        coin, [batch for batch in batches if batch.coin == coin], old_artifact)
        for coin in linear.COINS}
    controls = {
        "frozen_sample_complete": len(batches) == expected,
        "leaf_floor_exact": PARAMS["min_child_samples"] == 20,
        "all_other_tree_params_unchanged": all(
            PARAMS[name] == value for name, value in v5.TREE_PARAMS.items()
            if name != "min_child_samples"),
        "first_generation_keys_unique": all(
            int(qgen.first_generation_mask(batch).sum()) == len({
                (int(batch.base.maker_side_sign[int(base_i)]), int(generation))
                for base_i, generation in zip(batch.base_index, batch.generation)})
            for batch in batches),
        "model_receipts_roundtrip": True,
    }
    for report in reports.values():
        receipt = report["receipt"]
        raw = zlib.decompress(base64.b64decode(receipt["model_text_zlib_b64"]))
        controls["model_receipts_roundtrip"] &= bool(
            hashlib.sha256(raw).hexdigest() == receipt["model_text_sha256"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "QUEUE_GENERATION_LEAF20_MODEL_ONLY_V1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT_MODEL_ONLY",
        "verdict": {coin: ("MODEL_GATE_PASS" if
                            reports[coin]["model_gate"]["pass"] else
                            "MODEL_GATE_FAIL") for coin in linear.COINS},
        "decision_eligible": False,
        "promotion_authorized": False,
        "policy_replay_performed": False,
        "model": reports,
        "controls": controls,
        "population": {
            "per_coin_day": qgen5.PER_COIN_DAY,
            "n_windows": len(batches),
            "training_days": list(v5.TRAIN_DAYS),
            "development_days": list(v5.HOLDOUT_DAYS),
            "selected_slugs": slugs,
        },
        "semantics": {
            "only_change_vs_iteration_008": "MIN_CHILD_SAMPLES_200_TO_20",
            "model_only": True,
            "forward_days_observed": 0,
        },
        "provenance": {
            "code_sha256": _file_sha(Path(__file__)),
            "protocol_sha256": _file_sha(PROTOCOL),
            "iteration_008_artifact_sha256": _file_sha(qgen5.OUT),
            "action_builder_sha256": _file_sha(Path(qact.__file__)),
            "generation_engine_sha256": _file_sha(Path(qgen.__file__)),
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
    print(f"[qleaf] verdict {result['verdict']}", flush=True)
    print(f"[qleaf] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0
    def ok(value: bool, name: str) -> None:
        nonlocal checks
        if not value:
            raise AssertionError(name)
        checks += 1
    ok(MIN_CHILD_SAMPLES == 20, "fixed default leaf floor")
    ok(PARAMS["min_child_samples"] != v5.TREE_PARAMS["min_child_samples"],
       "one intended parameter differs")
    ok(sum(PARAMS[k] != v for k, v in v5.TREE_PARAMS.items()) == 1,
       "exactly one tree parameter differs")
    ok(PROTOCOL.exists() and qgen5.OUT.exists(), "protocol and parent exist")
    print(f"[qleaf] selftest OK — {checks} checks")
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
