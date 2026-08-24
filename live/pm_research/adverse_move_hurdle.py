"""Zero-inflated adverse-action value development comparison (v4).

Research only.  This module decomposes the incentive-free cancellation target
into two action-conditioned stages on the existing exact-event shadow rows:

1. probability that a fill remains preventable after cancellation latency;
2. expected gross cancellation value conditional on such a fill.

Their product is the unconditional expected cancellation value used by the
frozen ``CANCEL iff expected value > 0`` rule.  The script has no venue,
order, cancellation, or execution port.

Commands::

    python3 live/pm_research/adverse_move_hurdle.py --selftest
    python3 live/pm_research/adverse_move_hurdle.py run --per-coin-day 1
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import adverse_feature_rows_fast as fast
import adverse_move_fast as linear
import adverse_move_nonlinear as direct
import flow_intensity as fi

OUT = fi.PM / "derived/adverse_move_hurdle_development_v4.json"
PLAN = Path(__file__).with_name("plans") / "BE_ADVERSE_MOVE_PLAN.md"

TRAIN_DAYS = linear.TRAIN_DAYS
HOLDOUT_DAYS = linear.HOLDOUT_DAYS
COINS = linear.COINS
SEED = 20260824
REQUIRED_FORWARD_DAYS = 10
EVENT_EPSILON_SHARES = 1e-9

# Reuse the already pinned v3 tree capacity.  This is a target decomposition,
# not a hyperparameter search.
TREE_PARAMS = dict(direct.TREE_PARAMS)
MODEL_CONFIG = {
    "family": "ACTION_CONDITIONED_ZERO_INFLATED_HURDLE",
    "stage_1_preventable_fill": {
        "model": "LGBMClassifier",
        "objective": "binary",
        "label": "prevented_shares_after_latency_gt_0",
        "class_weight": None,
        **TREE_PARAMS,
    },
    "stage_2_conditional_value": {
        "model": "LGBMRegressor",
        "objective": "regression",
        "population": "prevented_shares_after_latency_gt_0",
        "label": "incentive_free_gross_cancel_value_cents",
        **TREE_PARAMS,
    },
    "composition": "P_PREVENTABLE_FILL_X_E_GROSS_VALUE_GIVEN_PREVENTABLE_FILL",
    "direct_tree_comparator": direct.MODEL_CONFIG["nonlinear_signed_value"],
    "linear_comparator": linear.MODEL_CONFIG["signed_cancel_value"],
    "early_stopping": False,
    "threshold_selection": "NONE_CANCEL_IFF_EXPECTED_VALUE_GT_ZERO",
    "hyperparameter_selection":
        "PINNED_FROM_V3_BEFORE_V4_DEVELOPMENT_RUN_NO_TUNING",
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


def _safe(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(slots=True)
class TreeStage:
    model: lgb.LGBMClassifier | lgb.LGBMRegressor
    target: str
    stage: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.booster_.predict(x), dtype=float)

    def receipt(self) -> dict[str, Any]:
        text = self.model.booster_.model_to_string().encode()
        compressed = zlib.compress(text, level=9)
        importance = self.model.booster_.feature_importance(
            importance_type="gain").astype(float)
        return {
            "stage": self.stage,
            "target": self.target,
            "kind": type(self.model).__name__,
            "feature_names": list(fast.FEATURE_NAMES),
            "feature_schema_hash": fast.FEATURE_SCHEMA_HASH,
            "source_profile_hash": fast.SOURCE_PROFILE_HASH,
            "action_schema_hash": fast.ACTION_SCHEMA_HASH,
            "params": MODEL_CONFIG[self.stage],
            "feature_importance_gain": importance.tolist(),
            "model_text_sha256": hashlib.sha256(text).hexdigest(),
            "model_text_zlib_b64": base64.b64encode(compressed).decode(),
        }


@dataclass(slots=True)
class HurdleModel:
    preventable_fill: TreeStage
    conditional_value: TreeStage
    horizon_ms: int
    latency_ms: int

    def predict_components(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probability = np.clip(self.preventable_fill.predict(x), 0.0, 1.0)
        conditional_value = self.conditional_value.predict(x)
        return probability, conditional_value

    def predict(self, x: np.ndarray) -> np.ndarray:
        probability, conditional_value = self.predict_components(x)
        return probability * conditional_value

    def receipt(self) -> dict[str, Any]:
        return {
            "horizon_ms": self.horizon_ms,
            "latency_ms": self.latency_ms,
            "composition": MODEL_CONFIG["composition"],
            "preventable_fill": self.preventable_fill.receipt(),
            "conditional_value": self.conditional_value.receipt(),
        }


def _fit_classifier(x: np.ndarray, y: np.ndarray, target: str
                    ) -> TreeStage | None:
    if len(y) < 2 or np.unique(y).size != 2:
        return None
    model = lgb.LGBMClassifier(
        objective="binary", class_weight=None, **TREE_PARAMS)
    model.fit(x, y)
    return TreeStage(model, target, "stage_1_preventable_fill")


def _fit_regressor(x: np.ndarray, y: np.ndarray, target: str
                   ) -> TreeStage | None:
    if len(y) < 2 or not np.isfinite(y).all():
        return None
    model = lgb.LGBMRegressor(objective="regression", **TREE_PARAMS)
    model.fit(x, y)
    return TreeStage(model, target, "stage_2_conditional_value")


def fit_hurdle(x: np.ndarray, gross_value: np.ndarray,
               prevented_shares: np.ndarray, horizon_ms: int,
               latency_ms: int) -> HurdleModel | None:
    """Fit both stages on one action/horizon/latency population."""
    finite = np.isfinite(gross_value) & np.isfinite(prevented_shares)
    if finite.sum() < 2:
        return None
    x_fit = x[finite]
    gross_fit = gross_value[finite]
    event = prevented_shares[finite] > EVENT_EPSILON_SHARES
    classifier = _fit_classifier(
        x_fit, event.astype(int),
        f"preventable_fill_H{horizon_ms}ms_L{latency_ms}ms")
    regressor = _fit_regressor(
        x_fit[event], gross_fit[event],
        f"gross_value_given_preventable_fill_H{horizon_ms}ms_L{latency_ms}ms")
    if classifier is None or regressor is None:
        return None
    return HurdleModel(classifier, regressor, horizon_ms, latency_ms)


def _gross_cancel_target(cancel_delta: np.ndarray,
                         prevented_shares: np.ndarray) -> np.ndarray:
    """Remove the v2 maker rebate from the avoid-fill value target."""
    return cancel_delta + prevented_shares * fast.MAKER_REBATE_CENTS_PER_SHARE


def _day_means(days: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    answer: dict[str, float | None] = {}
    for code, day in enumerate(HOLDOUT_DAYS):
        mask = days == code
        answer[day] = float(values[mask].mean()) if mask.any() else None
    return answer


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


def _ranking_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    profitable = target > 0.0
    if len(target) == 0:
        return {
            "n": 0,
            "profitable_prevalence": None,
            "average_precision": None,
            "roc_auc": None,
        }
    two_class = np.unique(profitable).size == 2
    return {
        "n": len(target),
        "profitable_prevalence": float(profitable.mean()),
        "average_precision": _safe(average_precision_score(profitable, score))
            if profitable.any() else None,
        "roc_auc": _safe(roc_auc_score(profitable, score)) if two_class else None,
        "diagnostic_only": True,
        "score_is_unconditional_expected_cancel_value": True,
    }


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
    report["rule"] = "CANCEL_IFF_HURDLE_EXPECTED_GROSS_VALUE_CENTS_GT_0"
    report["per_day_selection_gain_cents_per_decision"] = _day_means(
        days, incremental)
    report["per_day_realized_delta_cents_per_decision"] = _day_means(
        days, realized)
    return report, realized


def _comparison_report(days: np.ndarray, candidate_realized: np.ndarray,
                       comparator_realized: np.ndarray,
                       comparator_name: str) -> dict[str, Any]:
    delta = candidate_realized - comparator_realized
    return {
        "comparator": comparator_name,
        "candidate_minus_comparator_cents_per_decision": float(delta.mean()),
        "day_clustered_95pct_ci": linear.day_cluster_mean_ci(days, delta),
        "per_day_cents_per_decision": _day_means(days, delta),
    }


def _signal_gate(fill_fit: dict[str, Any], combined_fit: dict[str, Any],
                 hurdle_policy: dict[str, Any],
                 direct_comparison: dict[str, Any]) -> dict[str, Any]:
    day_gain = hurdle_policy["per_day_selection_gain_cents_per_decision"]
    day_compare = direct_comparison["per_day_cents_per_decision"]
    fill_skill = fill_fit.get("brier_skill_vs_training_prevalence")
    checks = {
        "positive_preventable_fill_brier_skill": bool(
            fill_skill is not None and fill_skill > 0),
        "positive_combined_holdout_r2": bool(
            combined_fit.get("r2") is not None and combined_fit["r2"] > 0),
        "positive_aggregate_gain_vs_constant": bool(
            hurdle_policy[
                "selection_gain_vs_training_selected_constant_cents_per_decision"]
            > 0),
        "positive_gain_vs_constant_each_day": bool(
            all(value is not None and value > 0 for value in day_gain.values())),
        "positive_aggregate_gain_vs_direct_tree": bool(
            direct_comparison[
                "candidate_minus_comparator_cents_per_decision"] > 0),
        "positive_gain_vs_direct_tree_each_day": bool(
            all(value is not None and value > 0
                for value in day_compare.values())),
        "nondegenerate_cancel_fraction": bool(
            0.02 < hurdle_policy["cancel_fraction"] < 0.98),
    }
    return {
        "label": "MODEL_SIGNAL_PRESENT" if all(checks.values())
        else "MODEL_SIGNAL_NOT_ESTABLISHED",
        "checks": checks,
        "development_only": True,
    }


def fit_coin(coin: str, batches: Sequence[fast.FastWindowBatch]) -> dict[str, Any]:
    train_batches = [batch for batch in batches if batch.day in TRAIN_DAYS]
    holdout_batches = [batch for batch in batches if batch.day in HOLDOUT_DAYS]
    x_train_all = linear._concat_x(train_batches)
    x_holdout_all = linear._concat_x(holdout_batches)
    holdout_days_all = linear._day_codes(holdout_batches)
    horizons: dict[str, Any] = {}
    model_artifacts: dict[str, Any] = {}
    signal_cells: list[str] = []

    for horizon in fast.PREDICTION_HORIZONS_MS:
        availability_train = linear._concat_target(train_batches, "toxic", horizon)
        availability_holdout = linear._concat_target(
            holdout_batches, "toxic", horizon)
        train_mask = availability_train >= 0
        holdout_mask = availability_holdout >= 0
        x_train = x_train_all[train_mask]
        x_holdout = x_holdout_all[holdout_mask]
        holdout_days = holdout_days_all[holdout_mask]
        filled_holdout = linear._concat_target(
            holdout_batches, "filled_shares", horizon)[holdout_mask]

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
            hurdle = fit_hurdle(
                x_train[finite_train], gross_train[finite_train],
                prevented_train[finite_train], horizon, latency)
            direct_tree = direct.fit_tree_regressor(
                x_train[finite_train], gross_train[finite_train],
                f"direct_gross_cancel_value_H{horizon}ms_L{latency}ms")
            ridge = linear.fit_ridge(
                x_train[finite_train], gross_train[finite_train],
                f"linear_gross_cancel_value_H{horizon}ms_L{latency}ms")
            if (hurdle is None or direct_tree is None or ridge is None
                    or not finite_holdout.any()):
                latency_results[str(latency)] = {
                    "status": "UNAVAILABLE",
                    "reason": "INSUFFICIENT_TWO_CLASS_OR_CONDITIONAL_VALUE_ROWS",
                }
                continue

            x_eval = x_holdout[finite_holdout]
            target_eval = gross_holdout[finite_holdout].astype(float)
            target_train = gross_train[finite_train].astype(float)
            prevented_eval = prevented_holdout[finite_holdout].astype(float)
            prevented_fit = prevented_train[finite_train].astype(float)
            days = holdout_days[finite_holdout]
            filled = filled_holdout[finite_holdout].astype(float)
            event_train = prevented_fit > EVENT_EPSILON_SHARES
            event_eval = prevented_eval > EVENT_EPSILON_SHARES

            fill_probability, conditional_prediction = (
                hurdle.predict_components(x_eval))
            hurdle_prediction = fill_probability * conditional_prediction
            hurdle_train_prediction = hurdle.predict(x_train[finite_train])
            direct_prediction = direct_tree.predict(x_eval)
            ridge_prediction = ridge.predict(x_eval)

            hurdle_policy, hurdle_realized = _policy_report(
                target_train, target_eval, hurdle_prediction, days,
                prevented_eval, filled)
            direct_policy, direct_realized = direct._policy_report(
                target_train, target_eval, direct_prediction, days,
                prevented_eval, filled)
            ridge_policy, ridge_realized = direct._policy_report(
                target_train, target_eval, ridge_prediction, days,
                prevented_eval, filled)
            direct_comparison = _comparison_report(
                days, hurdle_realized, direct_realized, "DIRECT_LIGHTGBM_V3")
            ridge_comparison = _comparison_report(
                days, hurdle_realized, ridge_realized, "RIDGE_V2_REFIT")
            fill_fit = _classification_metrics(
                event_train.astype(int), event_eval.astype(int), fill_probability)
            conditional_fit = linear.regression_metrics(
                target_eval[event_eval], conditional_prediction[event_eval])
            combined_fit = linear.regression_metrics(
                target_eval, hurdle_prediction)
            gate = _signal_gate(
                fill_fit, combined_fit, hurdle_policy, direct_comparison)
            cell = f"H{horizon}ms_L{latency}ms"
            if gate["label"] == "MODEL_SIGNAL_PRESENT":
                signal_cells.append(cell)

            non_event_train_max = (
                float(np.abs(target_train[~event_train]).max())
                if (~event_train).any() else 0.0)
            non_event_eval_max = (
                float(np.abs(target_eval[~event_eval]).max())
                if (~event_eval).any() else 0.0)
            latency_results[str(latency)] = {
                "status": "AVAILABLE",
                "latency_profile_status": "ASSUMED_COUNTERFACTUAL",
                "target": "INCENTIVE_FREE_GROSS_CANCEL_VALUE_CENTS",
                "decomposition_audit": {
                    "nonpreventable_target_max_abs_train": non_event_train_max,
                    "nonpreventable_target_max_abs_holdout": non_event_eval_max,
                    "identity_holds_exactly": bool(
                        non_event_train_max <= 1e-12
                        and non_event_eval_max <= 1e-12),
                    "conditional_train_rows": int(event_train.sum()),
                    "conditional_holdout_rows": int(event_eval.sum()),
                    "conditional_train_value_mean_cents":
                        float(target_train[event_train].mean()),
                    "conditional_holdout_value_mean_cents":
                        float(target_eval[event_eval].mean()),
                },
                "hurdle": {
                    "preventable_fill_fit": fill_fit,
                    "conditional_value_fit_on_realized_preventable_fills":
                        conditional_fit,
                    "combined_unconditional_value_fit": combined_fit,
                    "ranking": _ranking_metrics(target_eval, hurdle_prediction),
                    "policy": hurdle_policy,
                    "training_prediction_mean_cents":
                        float(hurdle_train_prediction.mean()),
                },
                "direct_tree": {
                    "fit": linear.regression_metrics(
                        target_eval, direct_prediction),
                    "policy": direct_policy,
                },
                "ridge": {
                    "fit": linear.regression_metrics(target_eval, ridge_prediction),
                    "policy": ridge_policy,
                },
                "hurdle_vs_direct_tree": direct_comparison,
                "hurdle_vs_ridge": ridge_comparison,
                "development_signal_gate": gate,
            }
            latency_models[str(latency)] = hurdle.receipt()

        horizons[str(horizon)] = {"latencies_ms": latency_results}
        model_artifacts[str(horizon)] = {"latencies": latency_models}

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
        "n_rows_train": sum(batch.n_rows for batch in train_batches),
        "n_rows_holdout": sum(batch.n_rows for batch in holdout_batches),
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
        "schema_version": 4,
        "protocol": "BE_ADVERSE_MOVE_HURDLE_DEVELOPMENT_V4",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT",
        "verdict": "INSUFFICIENT_EVIDENCE",
        "decision_eligible": False,
        "candidate_frozen": False,
        "reasons": [
            "v4_was_designed_after_v3_and_uses_already_seen_development_days",
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
            "markout_horizon_s": fast.MARKOUT_HORIZON_S,
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
        "population": {
            "per_coin_day_requested": per_coin_day,
            "selected_slugs": slugs,
            "n_windows": len(batches),
            "n_feature_rows": sum(batch.n_rows for batch in batches),
            "diagnostics": dict(sorted(diagnostics.items())),
            "rows_by_slug": {batch.slug: batch.n_rows for batch in batches},
        },
        "coins": {
            coin: fit_coin(coin, [batch for batch in batches if batch.coin == coin])
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
    print(f"[adverse-hurdle] wrote {OUT}", flush=True)
    print(f"[adverse-hurdle] signal cells "
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
    x = rng.normal(size=(12_000, len(fast.FEATURE_NAMES))).astype(np.float32)
    fill_probability = 1.0 / (1.0 + np.exp(-(
        1.4 * x[:, 0] * x[:, 1] - 0.4)))
    prevented = (rng.random(len(x)) < fill_probability).astype(float)
    conditional = 1.8 * x[:, 2] * x[:, 3] + 0.6
    gross = prevented * conditional
    train = np.arange(8_000)
    test = np.arange(8_000, len(x))
    model = fit_hurdle(
        x[train], gross[train], prevented[train], 100, 25)
    ok(model is not None, "synthetic hurdle fits")
    assert model is not None
    probability, conditional_prediction = model.predict_components(x[test])
    event_test = prevented[test] > 0
    skill = _classification_metrics(
        (prevented[train] > 0).astype(int), event_test.astype(int), probability)
    ok(skill["brier_skill_vs_training_prevalence"] > 0.05,
       "preventable-fill stage learns nonlinear event")
    conditional_r2 = linear.regression_metrics(
        gross[test][event_test], conditional_prediction[event_test])["r2"]
    ok(conditional_r2 is not None and conditional_r2 > 0.45,
       "conditional-value stage learns nonlinear value")
    expected = model.predict(x[test])
    ok(np.allclose(expected, probability * conditional_prediction),
       "hurdle composition is exact")
    receipt = model.receipt()
    for stage in ("preventable_fill", "conditional_value"):
        stage_receipt = receipt[stage]
        raw = zlib.decompress(base64.b64decode(
            stage_receipt["model_text_zlib_b64"]))
        ok(hashlib.sha256(raw).hexdigest()
           == stage_receipt["model_text_sha256"],
           f"{stage} model receipt round-trips")
        ok(stage_receipt["feature_schema_hash"] == fast.FEATURE_SCHEMA_HASH,
           f"{stage} binds feature schema")

    cancel = np.asarray([-1.0, 0.0, 2.0])
    shares = np.asarray([2.0, 0.0, 1.0])
    gross_without_rebate = _gross_cancel_target(cancel, shares)
    ok(np.allclose(
        gross_without_rebate,
        cancel + shares * fast.MAKER_REBATE_CENTS_PER_SHARE),
       "gross target removes maker rebate exactly")
    ranking = _ranking_metrics(
        np.asarray([-2.0, 0.0, 1.0, 3.0]),
        np.asarray([-1.0, -0.5, 0.2, 0.8]))
    ok(ranking["average_precision"] == 1.0 and ranking["roc_auc"] == 1.0,
       "economic ranking diagnostic has correct orientation")

    fill_fit = {"brier_skill_vs_training_prevalence": 0.1}
    combined_fit = {"r2": 0.1}
    policy = {
        "selection_gain_vs_training_selected_constant_cents_per_decision": 1.0,
        "per_day_selection_gain_cents_per_decision": {
            HOLDOUT_DAYS[0]: 1.0, HOLDOUT_DAYS[1]: -0.1},
        "cancel_fraction": 0.5,
    }
    comparison = {
        "candidate_minus_comparator_cents_per_decision": 1.0,
        "per_day_cents_per_decision": {
            HOLDOUT_DAYS[0]: 1.0, HOLDOUT_DAYS[1]: 1.0},
    }
    ok(_signal_gate(fill_fit, combined_fit, policy, comparison)["label"]
       == "MODEL_SIGNAL_NOT_ESTABLISHED", "one bad day fails hurdle gate")
    policy["per_day_selection_gain_cents_per_decision"][HOLDOUT_DAYS[1]] = 0.1
    ok(_signal_gate(fill_fit, combined_fit, policy, comparison)["label"]
       == "MODEL_SIGNAL_PRESENT", "all frozen hurdle checks can pass")
    ok(MODEL_CONFIG["early_stopping"] is False,
       "holdout cannot tune tree count")
    ok(REQUIRED_FORWARD_DAYS == 10, "forward validation floor preserved")
    print(f"[adverse-hurdle] selftest OK — {checks} checks")
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
