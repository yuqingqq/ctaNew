"""Nonlinear adverse-action value development comparison (v3).

Research only. This module reads recorded exact-event shadow-action rows and
compares pinned LightGBM models with the v2 linear family on an identical,
incentive-free target. It has no venue, order, cancel, or execution port.

Commands:

    python3 live/pm_research/adverse_move_nonlinear.py --selftest
    python3 live/pm_research/adverse_move_nonlinear.py run --per-coin-day 1
"""

from __future__ import annotations

import argparse
import base64
import collections
import datetime as dt
import hashlib
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
from sklearn.metrics import brier_score_loss

import adverse_feature_rows_fast as fast
import adverse_move_fast as linear
import flow_intensity as fi

OUT = fi.PM / "derived/adverse_move_nonlinear_development_v3.json"
PLAN = Path(__file__).with_name("plans") / "BE_ADVERSE_MOVE_PLAN.md"

TRAIN_DAYS = linear.TRAIN_DAYS
HOLDOUT_DAYS = linear.HOLDOUT_DAYS
COINS = linear.COINS
SEED = 20260824
REQUIRED_FORWARD_DAYS = 10

TREE_PARAMS = {
    "n_estimators": 128,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 200,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "random_state": SEED,
    "n_jobs": 4,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}

MODEL_CONFIG = {
    "nonlinear_classifier": {
        "model": "LGBMClassifier",
        "objective": "binary",
        "class_weight": None,
        **TREE_PARAMS,
    },
    "nonlinear_signed_value": {
        "model": "LGBMRegressor",
        "objective": "regression",
        **TREE_PARAMS,
    },
    "linear_comparator": linear.MODEL_CONFIG,
    "early_stopping": False,
    "threshold_selection": "NONE_CANCEL_IFF_PREDICTION_GT_ZERO",
    "hyperparameter_selection":
        "PINNED_IN_PLAN_SECTION_10_BEFORE_V3_DEVELOPMENT_RUN_NO_TUNING",
}


def _sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


@dataclass(slots=True)
class TreeModel:
    model: lgb.LGBMClassifier | lgb.LGBMRegressor
    target: str
    kind: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.booster_.predict(x), dtype=float)

    def receipt(self) -> dict[str, Any]:
        text = self.model.booster_.model_to_string().encode()
        compressed = zlib.compress(text, level=9)
        importance = self.model.booster_.feature_importance(
            importance_type="gain").astype(float)
        return {
            "target": self.target,
            "kind": type(self.model).__name__,
            "feature_names": list(fast.FEATURE_NAMES),
            "feature_schema_hash": fast.FEATURE_SCHEMA_HASH,
            "source_profile_hash": fast.SOURCE_PROFILE_HASH,
            "action_schema_hash": fast.ACTION_SCHEMA_HASH,
            "params": MODEL_CONFIG[
                "nonlinear_classifier" if self.kind == "classifier"
                else "nonlinear_signed_value"],
            "feature_importance_gain": importance.tolist(),
            "model_text_sha256": hashlib.sha256(text).hexdigest(),
            "model_text_zlib_b64": base64.b64encode(compressed).decode(),
        }


def fit_tree_classifier(x: np.ndarray, y: np.ndarray,
                        target: str) -> TreeModel | None:
    if len(y) < 2 or np.unique(y).size != 2:
        return None
    model = lgb.LGBMClassifier(
        objective="binary", class_weight=None, **TREE_PARAMS)
    model.fit(x, y)
    return TreeModel(model, target, "classifier")


def fit_tree_regressor(x: np.ndarray, y: np.ndarray,
                       target: str) -> TreeModel | None:
    if len(y) < 2:
        return None
    model = lgb.LGBMRegressor(objective="regression", **TREE_PARAMS)
    model.fit(x, y)
    return TreeModel(model, target, "regressor")


def _gross_cancel_target(cancel_delta: np.ndarray,
                         prevented_shares: np.ndarray) -> np.ndarray:
    """Remove the v2 maker rebate from the avoid-fill value target."""
    return cancel_delta + prevented_shares * fast.MAKER_REBATE_CENTS_PER_SHARE


def _classification_metrics(y_train: np.ndarray, y_eval: np.ndarray,
                            prediction: np.ndarray) -> dict[str, Any]:
    prevalence = float(y_train.mean())
    baseline = np.full(len(y_eval), prevalence)
    score = float(brier_score_loss(y_eval, prediction))
    base_score = float(brier_score_loss(y_eval, baseline))
    return {
        "n_train": len(y_train),
        "n_holdout": len(y_eval),
        "train_prevalence": prevalence,
        "holdout_prevalence": float(y_eval.mean()),
        "brier": score,
        "training_prevalence_baseline_brier": base_score,
        "brier_skill_vs_training_prevalence":
            1.0 - score / base_score if base_score > 0 else None,
        "ece_equal_width_10":
            linear.expected_calibration_error(y_eval, prediction),
        "prediction_mean": float(prediction.mean()),
    }


def _day_means(days: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    answer: dict[str, float | None] = {}
    for code, day in enumerate(HOLDOUT_DAYS):
        mask = days == code
        answer[day] = float(values[mask].mean()) if mask.any() else None
    return answer


def _policy_report(target_train: np.ndarray, target_eval: np.ndarray,
                   prediction: np.ndarray, days: np.ndarray,
                   prevented: np.ndarray, filled: np.ndarray) -> tuple[
                       dict[str, Any], np.ndarray]:
    report = linear.policy_metrics(
        target_train, target_eval, prediction, days, prevented, filled)
    cancel = prediction > 0.0
    realized = np.where(cancel, target_eval, 0.0)
    baseline_cancel = bool(target_train.mean() > 0.0)
    baseline = target_eval if baseline_cancel else np.zeros(len(target_eval))
    incremental = realized - baseline
    report["per_day_selection_gain_cents_per_decision"] = _day_means(
        days, incremental)
    report["per_day_realized_delta_cents_per_decision"] = _day_means(
        days, realized)
    return report, realized


def _comparison_report(days: np.ndarray, nonlinear_realized: np.ndarray,
                       linear_realized: np.ndarray) -> dict[str, Any]:
    delta = nonlinear_realized - linear_realized
    return {
        "nonlinear_minus_linear_cents_per_decision": float(delta.mean()),
        "day_clustered_95pct_ci": linear.day_cluster_mean_ci(days, delta),
        "per_day_cents_per_decision": _day_means(days, delta),
    }


def _signal_gate(nonlinear_fit: dict[str, Any],
                 nonlinear_policy: dict[str, Any],
                 comparison: dict[str, Any]) -> dict[str, Any]:
    day_gain = nonlinear_policy["per_day_selection_gain_cents_per_decision"]
    day_compare = comparison["per_day_cents_per_decision"]
    checks = {
        "positive_holdout_r2": bool(
            nonlinear_fit.get("r2") is not None and nonlinear_fit["r2"] > 0),
        "positive_aggregate_gain_vs_constant": bool(
            nonlinear_policy[
                "selection_gain_vs_training_selected_constant_cents_per_decision"]
            > 0),
        "positive_gain_vs_constant_each_day": bool(
            all(value is not None and value > 0 for value in day_gain.values())),
        "positive_aggregate_gain_vs_linear": bool(
            comparison["nonlinear_minus_linear_cents_per_decision"] > 0),
        "positive_gain_vs_linear_each_day": bool(
            all(value is not None and value > 0
                for value in day_compare.values())),
        "nondegenerate_cancel_fraction": bool(
            0.02 < nonlinear_policy["cancel_fraction"] < 0.98),
    }
    return {
        "label": "MODEL_SIGNAL_PRESENT" if all(checks.values())
        else "MODEL_SIGNAL_NOT_ESTABLISHED",
        "checks": checks,
        "development_only": True,
    }


def fit_coin(coin: str, batches: Sequence[fast.FastWindowBatch]) -> dict[str, Any]:
    train_batches = [b for b in batches if b.day in TRAIN_DAYS]
    holdout_batches = [b for b in batches if b.day in HOLDOUT_DAYS]
    x_train_all = linear._concat_x(train_batches)
    x_holdout_all = linear._concat_x(holdout_batches)
    holdout_days_all = linear._day_codes(holdout_batches)
    horizons: dict[str, Any] = {}
    model_artifacts: dict[str, Any] = {}
    signal_cells: list[str] = []

    for horizon in fast.PREDICTION_HORIZONS_MS:
        toxic_train_all = linear._concat_target(train_batches, "toxic", horizon)
        toxic_holdout_all = linear._concat_target(
            holdout_batches, "toxic", horizon)
        train_mask = toxic_train_all >= 0
        holdout_mask = toxic_holdout_all >= 0
        x_train = x_train_all[train_mask]
        x_holdout = x_holdout_all[holdout_mask]
        toxic_train = toxic_train_all[train_mask].astype(int)
        toxic_holdout = toxic_holdout_all[holdout_mask].astype(int)
        holdout_days = holdout_days_all[holdout_mask]
        filled_holdout = linear._concat_target(
            holdout_batches, "filled_shares", horizon)[holdout_mask]

        tree_classifier = fit_tree_classifier(
            x_train, toxic_train, f"joint_toxic_fill_H{horizon}ms")
        logistic = linear.fit_logistic(
            x_train, toxic_train, f"joint_toxic_fill_H{horizon}ms")
        classification: dict[str, Any]
        if (tree_classifier is None or logistic is None
                or len(toxic_holdout) == 0):
            classification = {
                "status": "UNAVAILABLE",
                "reason": "INSUFFICIENT_TWO_CLASS_TRAIN_OR_HOLDOUT_ROWS",
            }
        else:
            classification = {
                "status": "AVAILABLE",
                "nonlinear": _classification_metrics(
                    toxic_train, toxic_holdout,
                    tree_classifier.predict(x_holdout)),
                "linear": _classification_metrics(
                    toxic_train, toxic_holdout, logistic.predict(x_holdout)),
            }

        latency_results: dict[str, Any] = {}
        latency_models: dict[str, Any] = {}
        for latency in fast.LATENCY_MS:
            if latency >= horizon:
                continue
            key = (horizon, latency)
            cancel_train = linear._concat_target(
                train_batches, "cancel_delta", key)[train_mask]
            cancel_holdout = linear._concat_target(
                holdout_batches, "cancel_delta", key)[holdout_mask]
            prevented_train = linear._concat_target(
                train_batches, "prevented_shares", key)[train_mask]
            prevented_holdout = linear._concat_target(
                holdout_batches, "prevented_shares", key)[holdout_mask]
            gross_train = _gross_cancel_target(cancel_train, prevented_train)
            gross_holdout = _gross_cancel_target(cancel_holdout, prevented_holdout)
            finite_train = np.isfinite(gross_train) & np.isfinite(prevented_train)
            finite_holdout = (
                np.isfinite(gross_holdout) & np.isfinite(prevented_holdout))
            tree = fit_tree_regressor(
                x_train[finite_train], gross_train[finite_train],
                f"gross_signed_cancel_value_H{horizon}ms_L{latency}ms")
            ridge = linear.fit_ridge(
                x_train[finite_train], gross_train[finite_train],
                f"gross_signed_cancel_value_H{horizon}ms_L{latency}ms",
                scaler=logistic.scaler if logistic is not None else None)
            if tree is None or ridge is None or not finite_holdout.any():
                latency_results[str(latency)] = {
                    "status": "UNAVAILABLE",
                    "reason": "INSUFFICIENT_TRAIN_OR_HOLDOUT_ROWS",
                }
                continue
            x_eval = x_holdout[finite_holdout]
            target_eval = gross_holdout[finite_holdout].astype(float)
            target_train = gross_train[finite_train].astype(float)
            days = holdout_days[finite_holdout]
            prevented = prevented_holdout[finite_holdout].astype(float)
            filled = filled_holdout[finite_holdout].astype(float)
            nonlinear_prediction = tree.predict(x_eval)
            linear_prediction = ridge.predict(x_eval)
            nonlinear_policy, nonlinear_realized = _policy_report(
                target_train, target_eval, nonlinear_prediction, days,
                prevented, filled)
            linear_policy, linear_realized = _policy_report(
                target_train, target_eval, linear_prediction, days,
                prevented, filled)
            nonlinear_fit = linear.regression_metrics(
                target_eval, nonlinear_prediction)
            comparison = _comparison_report(
                days, nonlinear_realized, linear_realized)
            gate = _signal_gate(nonlinear_fit, nonlinear_policy, comparison)
            cell = f"H{horizon}ms_L{latency}ms"
            if gate["label"] == "MODEL_SIGNAL_PRESENT":
                signal_cells.append(cell)
            latency_results[str(latency)] = {
                "status": "AVAILABLE",
                "latency_profile_status": "ASSUMED_COUNTERFACTUAL",
                "target": "INCENTIVE_FREE_GROSS_CANCEL_VALUE_CENTS",
                "nonlinear": {
                    "direct_value_fit": nonlinear_fit,
                    "policy": nonlinear_policy,
                },
                "linear": {
                    "direct_value_fit": linear.regression_metrics(
                        target_eval, linear_prediction),
                    "policy": linear_policy,
                },
                "nonlinear_vs_linear": comparison,
                "development_signal_gate": gate,
            }
            latency_models[str(latency)] = {
                "nonlinear": tree.receipt(),
                "linear": ridge.receipt(),
            }

        horizons[str(horizon)] = {
            "classification": classification,
            "latencies_ms": latency_results,
        }
        model_artifacts[str(horizon)] = {
            "nonlinear_classification":
                tree_classifier.receipt() if tree_classifier else None,
            "linear_classification": logistic.receipt() if logistic else None,
            "latencies": latency_models,
        }

    candidate = {
        "coin": coin,
        "status": "DEVELOPMENT",
        "decision_eligible": False,
        "training_days": list(TRAIN_DAYS),
        "model_config": MODEL_CONFIG,
        "models_by_horizon_ms": model_artifacts,
    }
    return {
        "status": "DEVELOPMENT",
        "verdict": "INSUFFICIENT_EVIDENCE",
        "decision_eligible": False,
        "candidate_hash": _sha(candidate),
        "n_windows_train": len(train_batches),
        "n_windows_holdout": len(holdout_batches),
        "n_rows_train": sum(b.n_rows for b in train_batches),
        "n_rows_holdout": sum(b.n_rows for b in holdout_batches),
        "development_signal_cells": signal_cells,
        "horizons_ms": horizons,
        "model_artifact": candidate,
    }


def run(per_coin_day: int) -> dict[str, Any]:
    batches, sampled, slugs = linear.build_batches(per_coin_day)
    diagnostics: collections.Counter[str] = collections.Counter()
    for batch in batches:
        diagnostics.update(batch.diagnostics)
    result: dict[str, Any] = {
        "schema_version": 3,
        "protocol": "BE_ADVERSE_MOVE_NONLINEAR_DEVELOPMENT_V3",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT",
        "verdict": "INSUFFICIENT_EVIDENCE",
        "decision_eligible": False,
        "candidate_frozen": False,
        "reasons": [
            "v3_uses_already_seen_v2_development_days",
            "only_two_development_holdout_days_not_ten_forward_days",
            "cancellation_latency_rungs_are_assumed_not_measured_end_to_end",
            "static_shadow_actions_do_not_form_an_inventory_path",
            "cancel_rejoin_queue_value_is_outside_this_static_target",
        ],
        "split": {
            "training_days": list(TRAIN_DAYS),
            "development_holdout_days": list(HOLDOUT_DAYS),
            "required_future_days_after_candidate_freeze": REQUIRED_FORWARD_DAYS,
            "observed_strictly_forward_days": 0,
        },
        "timing": {
            "decision_mode": "EXACT_EVENT_WITH_FIXED_COOLDOWN",
            "decision_cooldown_ms": fast.COOLDOWN_MS,
            "feature_windows_ms": list(fast.FAST_WINDOWS_MS),
            "prediction_horizons_ms": list(fast.PREDICTION_HORIZONS_MS),
            "assumed_cancellation_latency_ms": list(fast.LATENCY_MS),
            "tau_operative": None,
        },
        "feature_contract": {
            "names": list(fast.FEATURE_NAMES),
            "feature_schema_hash": fast.FEATURE_SCHEMA_HASH,
            "source_profile": fast.SOURCE_PROFILE,
            "source_profile_hash": fast.SOURCE_PROFILE_HASH,
            "action_schema_hash": fast.ACTION_SCHEMA_HASH,
        },
        "model_config": MODEL_CONFIG,
        "economics": {
            "maker_rebate_cents_per_share": 0.0,
            "liquidity_rewards": "EXCLUDED_BY_USER_DIRECTION",
            "target": "GROSS_MARKOUT_VALUE_OF_PREVENTED_FILL",
            "spread_capture": "INCLUDED_IN_OBSERVED_FILL_MARKOUT",
            "rejoin_queue_value": "OUTSIDE_STATIC_SHADOW_TARGET",
        },
        "skew": {
            "model_fit_population": "STATIC_JOIN_SHADOW_ACTIONS_NO_INVENTORY",
            "separate_stateful_test": "POLICY_OPTIMIZER_STAGE_B_SKEW_LB",
            "dynamic_composition_licensed_only_if_model_signal_present": True,
        },
        "population": {
            "per_coin_day_requested": per_coin_day,
            "selected_slugs": slugs,
            "n_windows": len(batches),
            "n_feature_rows": sum(batch.n_rows for batch in batches),
            "diagnostics": dict(sorted(diagnostics.items())),
            "rows_by_slug": {batch.slug: batch.n_rows for batch in batches},
        },
        "coins": {
            coin: fit_coin(coin, [b for b in batches if b.coin == coin])
            for coin in COINS
        },
        "provenance": {
            "polymarket": fi.provenance(sampled=sampled),
            "polymarket_files": [
                {
                    "path": str(path.relative_to(fast.REPO)),
                    "sha256": _file_sha(path),
                    "size": path.stat().st_size,
                }
                for path in sampled
            ],
            "hf_source_identity": {
                "kind": "PATH_SIZE_MTIME_RECEIPT_NOT_CONTENT_DIGEST",
                "files": linear._hf_manifest(slugs),
            },
            "code_sha256": _file_sha(Path(__file__)),
            "feature_builder_sha256": _file_sha(Path(fast.__file__)),
            "plan_sha256": _file_sha(PLAN),
        },
        "feature_samples_without_labels": [
            batch.feature_sample(index)
            for batch in batches[:2]
            for index in range(min(1, batch.n_rows))
        ],
    }
    result["development_signal_cells"] = {
        coin: result["coins"][coin]["development_signal_cells"] for coin in COINS
    }
    result["artifact_id"] = _sha(result)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, allow_nan=False))
    print(f"[adverse-nonlinear] wrote {OUT}", flush=True)
    print(f"[adverse-nonlinear] signal cells "
          f"{result['development_signal_cells']}", flush=True)
    return result


def selftest() -> int:
    checks = 0

    def ok(condition: bool, name: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(name)
        checks += 1

    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(6_000, len(fast.FEATURE_NAMES))).astype(np.float32)
    score = x[:, 0] * x[:, 1] + 0.35 * x[:, 2] * x[:, 3]
    y = (score > 0).astype(int)
    train, test = np.arange(4_000), np.arange(4_000, 6_000)
    tree_c = fit_tree_classifier(x[train], y[train], "synthetic-xor")
    logit = linear.fit_logistic(x[train], y[train], "synthetic-xor")
    ok(tree_c is not None and logit is not None, "both classifiers fit")
    assert tree_c is not None and logit is not None
    tree_brier = brier_score_loss(y[test], tree_c.predict(x[test]))
    linear_brier = brier_score_loss(y[test], logit.predict(x[test]))
    ok(tree_brier < linear_brier, "tree learns nonlinear interaction")
    tree_r = fit_tree_regressor(x[train], score[train], "synthetic-value")
    ok(tree_r is not None, "tree regressor fits")
    assert tree_r is not None
    r2 = linear.regression_metrics(score[test], tree_r.predict(x[test]))["r2"]
    ok(r2 is not None and r2 > 0.55, "tree regressor learns nonlinear value")
    receipt = tree_r.receipt()
    raw = zlib.decompress(base64.b64decode(receipt["model_text_zlib_b64"]))
    ok(hashlib.sha256(raw).hexdigest() == receipt["model_text_sha256"],
       "compressed model receipt round-trips")
    ok(receipt["feature_schema_hash"] == fast.FEATURE_SCHEMA_HASH,
       "model binds feature schema")
    cancel = np.asarray([-1.0, 0.0, 2.0])
    prevented = np.asarray([2.0, 0.0, 1.0])
    gross = _gross_cancel_target(cancel, prevented)
    ok(np.allclose(gross, cancel + prevented * fast.MAKER_REBATE_CENTS_PER_SHARE),
       "gross target removes maker rebate exactly")
    fake_fit = {"r2": 0.1}
    fake_policy = {
        "selection_gain_vs_training_selected_constant_cents_per_decision": 1.0,
        "per_day_selection_gain_cents_per_decision": {
            HOLDOUT_DAYS[0]: 1.0, HOLDOUT_DAYS[1]: -0.1},
        "cancel_fraction": 0.5,
    }
    fake_cmp = {
        "nonlinear_minus_linear_cents_per_decision": 1.0,
        "per_day_cents_per_decision": {
            HOLDOUT_DAYS[0]: 1.0, HOLDOUT_DAYS[1]: 1.0},
    }
    ok(_signal_gate(fake_fit, fake_policy, fake_cmp)["label"]
       == "MODEL_SIGNAL_NOT_ESTABLISHED", "one bad day fails gate")
    fake_policy["per_day_selection_gain_cents_per_decision"][HOLDOUT_DAYS[1]] = 0.1
    ok(_signal_gate(fake_fit, fake_policy, fake_cmp)["label"]
       == "MODEL_SIGNAL_PRESENT", "all frozen gate checks can pass")
    ok(MODEL_CONFIG["early_stopping"] is False,
       "holdout cannot tune tree count")
    ok(MODEL_CONFIG["nonlinear_signed_value"]["objective"] == "regression",
       "value model estimates mean with squared error")
    ok(REQUIRED_FORWARD_DAYS == 10, "forward validation floor preserved")
    print(f"[adverse-nonlinear] selftest OK — {checks} checks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    sub = parser.add_subparsers(dest="command")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--per-coin-day", type=int, default=1)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.command == "run":
        if args.per_coin_day < 1:
            parser.error("--per-coin-day must be positive")
        run(args.per_coin_day)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
