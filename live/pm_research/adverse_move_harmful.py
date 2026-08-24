"""Value-weighted harmful-flow cancellation experiment (v5).

Research only.  On action-conditioned rows with a latency-preventable fill,
this module classifies whether cancellation has positive incentive-free gross
value.  The primary classifier weights each training row by the absolute
realized value, making its fixed 0.5 threshold an economic action-sign rule.
There is no venue, order, cancellation, or execution port.

Commands::

    python3 live/pm_research/adverse_move_harmful.py --selftest
    python3 live/pm_research/adverse_move_harmful.py run --per-coin-day 1
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
from sklearn.metrics import average_precision_score, roc_auc_score

import adverse_feature_rows_fast as fast
import adverse_move_fast as linear
import adverse_move_hurdle as hurdle
import adverse_move_nonlinear as direct
import flow_intensity as fi

OUT = fi.PM / "derived/adverse_move_harmful_development_v5.json"
PLAN = Path(__file__).with_name("plans") / "BE_ADVERSE_MOVE_PLAN.md"

TRAIN_DAYS = linear.TRAIN_DAYS
HOLDOUT_DAYS = linear.HOLDOUT_DAYS
COINS = linear.COINS
SEED = 20260824
REQUIRED_FORWARD_DAYS = 10
VALUE_EPSILON_CENTS = 1e-9
TREE_PARAMS = dict(direct.TREE_PARAMS)

MODEL_CONFIG = {
    "family": "ACTION_CONDITIONED_VALUE_WEIGHTED_HARMFUL_FLOW",
    "preventable_fill": hurdle.MODEL_CONFIG["stage_1_preventable_fill"],
    "unweighted_harmful_fill": {
        "model": "LGBMClassifier",
        "objective": "binary",
        "population": "latency_preventable_nonzero_value_fills",
        "label": "incentive_free_gross_cancel_value_cents_gt_0",
        "sample_weight": "ONE_PER_ROW",
        "class_weight": None,
        **TREE_PARAMS,
    },
    "value_weighted_harmful_fill": {
        "model": "LGBMClassifier",
        "objective": "binary",
        "population": "latency_preventable_nonzero_value_fills",
        "label": "incentive_free_gross_cancel_value_cents_gt_0",
        "sample_weight": "ABS_GROSS_CANCEL_VALUE_CENTS",
        "class_weight": None,
        **TREE_PARAMS,
    },
    "action_score": "P_PREVENTABLE_FILL_X_2Q_VALUE_WEIGHTED_MINUS_1",
    "action_rule": "CANCEL_IFF_Q_VALUE_WEIGHTED_GT_0_5",
    "comparators": [
        "UNWEIGHTED_HARMFUL_FILL",
        "HURDLE_V4_SIGNED_VALUE",
        "DIRECT_LIGHTGBM_V3",
    ],
    "early_stopping": False,
    "weight_winsorization": False,
    "threshold_selection": "NONE_FIXED_AT_VALUE_WEIGHTED_PROBABILITY_0_5",
    "hyperparameter_selection":
        "PINNED_FROM_V3_V4_BEFORE_V5_DEVELOPMENT_RUN_NO_TUNING",
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
class ClassifierStage:
    model: lgb.LGBMClassifier
    target: str
    config_key: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self.model.booster_.predict(x)
        return np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)

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
            "params": MODEL_CONFIG[self.config_key],
            "feature_importance_gain": importance.tolist(),
            "model_text_sha256": hashlib.sha256(text).hexdigest(),
            "model_text_zlib_b64": base64.b64encode(compressed).decode(),
        }


def fit_classifier(x: np.ndarray, y: np.ndarray, target: str,
                   config_key: str,
                   sample_weight: np.ndarray | None = None
                   ) -> ClassifierStage | None:
    if len(y) < 2 or np.unique(y).size != 2:
        return None
    if sample_weight is not None:
        if (len(sample_weight) != len(y) or not np.isfinite(sample_weight).all()
                or (sample_weight < 0).any() or sample_weight.sum() <= 0):
            return None
        positive_weight = sample_weight > 0
        if np.unique(y[positive_weight]).size != 2:
            return None
    model = lgb.LGBMClassifier(
        objective="binary", class_weight=None, **TREE_PARAMS)
    model.fit(x, y, sample_weight=sample_weight)
    return ClassifierStage(model, target, config_key)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _weight_audit(weights: np.ndarray) -> dict[str, Any]:
    total = float(weights.sum())
    square_sum = float(np.square(weights).sum())
    return {
        "n": len(weights),
        "sum": total,
        "mean": float(weights.mean()) if len(weights) else None,
        "max": float(weights.max()) if len(weights) else None,
        "effective_sample_size":
            total * total / square_sum if square_sum > 0 else None,
        "max_share_of_total":
            float(weights.max() / total) if len(weights) and total > 0 else None,
    }


def _weighted_ece(y: np.ndarray, prediction: np.ndarray,
                  weights: np.ndarray, n_bins: int = 10) -> float | None:
    total = float(weights.sum())
    if len(y) == 0 or total <= 0:
        return None
    answer = 0.0
    for index in range(n_bins):
        lo, hi = index / n_bins, (index + 1) / n_bins
        mask = ((prediction >= lo)
                & ((prediction < hi) if index + 1 < n_bins
                   else prediction <= hi))
        bin_weight = float(weights[mask].sum())
        if bin_weight > 0:
            answer += bin_weight / total * abs(
                _weighted_mean(prediction[mask], weights[mask])
                - _weighted_mean(y[mask], weights[mask]))
    return answer


def weighted_classification_metrics(
        y_train: np.ndarray, train_weight: np.ndarray,
        y_eval: np.ndarray, eval_weight: np.ndarray,
        prediction: np.ndarray) -> dict[str, Any]:
    train_prevalence = _weighted_mean(y_train, train_weight)
    holdout_prevalence = _weighted_mean(y_eval, eval_weight)
    score = _weighted_mean(np.square(y_eval - prediction), eval_weight)
    baseline = np.full(len(y_eval), train_prevalence)
    base_score = _weighted_mean(np.square(y_eval - baseline), eval_weight)
    skill = 1.0 - score / base_score if base_score > 0 else None
    if skill is not None and abs(skill) < 1e-12:
        skill = 0.0
    two_class = np.unique(y_eval[eval_weight > 0]).size == 2
    return {
        "n_train": len(y_train),
        "n_holdout": len(y_eval),
        "train_value_weighted_prevalence": train_prevalence,
        "holdout_value_weighted_prevalence": holdout_prevalence,
        "weighted_brier": score,
        "training_weighted_prevalence_baseline_brier": base_score,
        "weighted_brier_skill_vs_training_weighted_prevalence": skill,
        "weighted_ece_equal_width_10":
            _weighted_ece(y_eval, prediction, eval_weight),
        "value_weighted_average_precision": _safe(average_precision_score(
            y_eval, prediction, sample_weight=eval_weight)),
        "value_weighted_roc_auc": _safe(roc_auc_score(
            y_eval, prediction, sample_weight=eval_weight))
            if two_class else None,
        "prediction_weighted_mean": _weighted_mean(prediction, eval_weight),
        "train_weight_audit": _weight_audit(train_weight),
        "holdout_weight_audit": _weight_audit(eval_weight),
    }


def _day_means(days: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    answer: dict[str, float | None] = {}
    for code, day in enumerate(HOLDOUT_DAYS):
        mask = days == code
        answer[day] = float(values[mask].mean()) if mask.any() else None
    return answer


def _policy_report(target_train: np.ndarray, target_eval: np.ndarray,
                   score: np.ndarray, days: np.ndarray,
                   prevented: np.ndarray, filled: np.ndarray,
                   rule: str) -> tuple[dict[str, Any], np.ndarray]:
    report = linear.policy_metrics(
        target_train, target_eval, score, days, prevented, filled)
    cancel = score > 0.0
    realized = np.where(cancel, target_eval, 0.0)
    baseline_cancel = bool(target_train.mean() > 0.0)
    baseline = target_eval if baseline_cancel else np.zeros(len(target_eval))
    incremental = realized - baseline
    report["rule"] = rule
    report["per_day_selection_gain_cents_per_decision"] = _day_means(
        days, incremental)
    report["per_day_realized_delta_cents_per_decision"] = _day_means(
        days, realized)
    return report, realized


def _conditional_economic_report(
        target: np.ndarray, score: np.ndarray,
        economic_mask: np.ndarray, days: np.ndarray) -> dict[str, Any]:
    selected_target = target[economic_mask]
    selected_score = score[economic_mask]
    selected_days = days[economic_mask]
    cancel = selected_score > 0.0
    realized = np.where(cancel, selected_target, 0.0)
    return {
        "n_preventable_nonzero_value_fills": len(selected_target),
        "cancel_fraction": float(cancel.mean()) if len(cancel) else None,
        "realized_delta_cents_per_preventable_nonzero_value_fill":
            float(realized.mean()) if len(realized) else None,
        "per_day_realized_delta_cents_per_preventable_nonzero_value_fill":
            _day_means(selected_days, realized),
    }


def _comparison_report(days: np.ndarray, candidate: np.ndarray,
                       comparator: np.ndarray,
                       name: str) -> dict[str, Any]:
    delta = candidate - comparator
    return {
        "comparator": name,
        "candidate_minus_comparator_cents_per_decision": float(delta.mean()),
        "day_clustered_95pct_ci": linear.day_cluster_mean_ci(days, delta),
        "per_day_cents_per_decision": _day_means(days, delta),
    }


def _all_positive(values: dict[str, float | None]) -> bool:
    return all(value is not None and value > 0 for value in values.values())


def _signal_gate(weighted_fit: dict[str, Any], policy: dict[str, Any],
                 vs_unweighted: dict[str, Any],
                 vs_hurdle: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "positive_value_weighted_brier_skill": bool(
            weighted_fit.get(
                "weighted_brier_skill_vs_training_weighted_prevalence", -1)
            > 0),
        "positive_realized_value_aggregate": bool(
            policy["realized_delta_cents_per_decision"] > 0),
        "positive_realized_value_each_day": _all_positive(
            policy["per_day_realized_delta_cents_per_decision"]),
        "positive_gain_vs_constant_aggregate": bool(
            policy[
                "selection_gain_vs_training_selected_constant_cents_per_decision"]
            > 0),
        "positive_gain_vs_constant_each_day": _all_positive(
            policy["per_day_selection_gain_cents_per_decision"]),
        "positive_gain_vs_unweighted_aggregate": bool(
            vs_unweighted[
                "candidate_minus_comparator_cents_per_decision"] > 0),
        "positive_gain_vs_unweighted_each_day": _all_positive(
            vs_unweighted["per_day_cents_per_decision"]),
        "positive_gain_vs_hurdle_aggregate": bool(
            vs_hurdle["candidate_minus_comparator_cents_per_decision"] > 0),
        "positive_gain_vs_hurdle_each_day": _all_positive(
            vs_hurdle["per_day_cents_per_decision"]),
        "nondegenerate_cancel_fraction": bool(
            0.02 < policy["cancel_fraction"] < 0.98),
    }
    return {
        "label": "HARMFUL_FLOW_SIGNAL_PRESENT" if all(checks.values())
        else "HARMFUL_FLOW_SIGNAL_NOT_ESTABLISHED",
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
            gross_train = hurdle._gross_cancel_target(
                cancel_train, prevented_train)
            gross_holdout = hurdle._gross_cancel_target(
                cancel_holdout, prevented_holdout)
            finite_train = np.isfinite(gross_train) & np.isfinite(prevented_train)
            finite_holdout = (
                np.isfinite(gross_holdout) & np.isfinite(prevented_holdout))
            x_fit = x_train[finite_train]
            target_train = gross_train[finite_train].astype(float)
            prevented_fit = prevented_train[finite_train].astype(float)

            v4_model = hurdle.fit_hurdle(
                x_fit, target_train, prevented_fit, horizon, latency)
            event_train = prevented_fit > hurdle.EVENT_EPSILON_SHARES
            economic_train = event_train & (
                np.abs(target_train) > VALUE_EPSILON_CENTS)
            helpful_train = (target_train[economic_train] > 0.0).astype(int)
            value_weight_train = np.abs(target_train[economic_train])
            unweighted = fit_classifier(
                x_fit[economic_train], helpful_train,
                f"helpful_cancel_unweighted_H{horizon}ms_L{latency}ms",
                "unweighted_harmful_fill")
            weighted = fit_classifier(
                x_fit[economic_train], helpful_train,
                f"helpful_cancel_value_weighted_H{horizon}ms_L{latency}ms",
                "value_weighted_harmful_fill", value_weight_train)
            direct_tree = direct.fit_tree_regressor(
                x_fit, target_train,
                f"direct_gross_cancel_value_H{horizon}ms_L{latency}ms")

            if (v4_model is None or unweighted is None or weighted is None
                    or direct_tree is None or not finite_holdout.any()):
                latency_results[str(latency)] = {
                    "status": "UNAVAILABLE",
                    "reason": "INSUFFICIENT_TWO_CLASS_VALUE_WEIGHTED_ROWS",
                }
                continue

            x_eval = x_holdout[finite_holdout]
            target_eval = gross_holdout[finite_holdout].astype(float)
            prevented_eval = prevented_holdout[finite_holdout].astype(float)
            days = holdout_days[finite_holdout]
            filled = filled_holdout[finite_holdout].astype(float)
            event_eval = prevented_eval > hurdle.EVENT_EPSILON_SHARES
            economic_eval = event_eval & (
                np.abs(target_eval) > VALUE_EPSILON_CENTS)
            helpful_eval = (target_eval[economic_eval] > 0.0).astype(int)
            value_weight_eval = np.abs(target_eval[economic_eval])
            if (not economic_eval.any()
                    or np.unique(helpful_eval).size != 2
                    or value_weight_eval.sum() <= 0):
                latency_results[str(latency)] = {
                    "status": "UNAVAILABLE",
                    "reason": "INSUFFICIENT_TWO_CLASS_HOLDOUT_ECONOMIC_ROWS",
                }
                continue

            fill_probability, v4_conditional = v4_model.predict_components(x_eval)
            weighted_probability = weighted.predict(x_eval)
            unweighted_probability = unweighted.predict(x_eval)
            weighted_score = fill_probability * (2.0 * weighted_probability - 1.0)
            unweighted_score = (
                fill_probability * (2.0 * unweighted_probability - 1.0))
            v4_score = fill_probability * v4_conditional
            direct_score = direct_tree.predict(x_eval)

            weighted_policy, weighted_realized = _policy_report(
                target_train, target_eval, weighted_score, days,
                prevented_eval, filled,
                "CANCEL_IFF_VALUE_WEIGHTED_HARMFUL_PROBABILITY_GT_0_5")
            unweighted_policy, unweighted_realized = _policy_report(
                target_train, target_eval, unweighted_score, days,
                prevented_eval, filled,
                "CANCEL_IFF_UNWEIGHTED_HARMFUL_PROBABILITY_GT_0_5")
            v4_policy, v4_realized = _policy_report(
                target_train, target_eval, v4_score, days,
                prevented_eval, filled,
                "CANCEL_IFF_HURDLE_V4_EXPECTED_VALUE_GT_0")
            direct_policy, direct_realized = _policy_report(
                target_train, target_eval, direct_score, days,
                prevented_eval, filled,
                "CANCEL_IFF_DIRECT_V3_EXPECTED_VALUE_GT_0")

            weighted_fit = weighted_classification_metrics(
                helpful_train, value_weight_train,
                helpful_eval, value_weight_eval,
                weighted_probability[economic_eval])
            unweighted_fit = hurdle._classification_metrics(
                helpful_train, helpful_eval,
                unweighted_probability[economic_eval])
            vs_unweighted = _comparison_report(
                days, weighted_realized, unweighted_realized,
                "UNWEIGHTED_HARMFUL_FILL")
            vs_hurdle = _comparison_report(
                days, weighted_realized, v4_realized,
                "HURDLE_V4_SIGNED_VALUE")
            vs_direct = _comparison_report(
                days, weighted_realized, direct_realized,
                "DIRECT_LIGHTGBM_V3")
            gate = _signal_gate(
                weighted_fit, weighted_policy, vs_unweighted, vs_hurdle)
            cell = f"H{horizon}ms_L{latency}ms"
            if gate["label"] == "HARMFUL_FLOW_SIGNAL_PRESENT":
                signal_cells.append(cell)

            latency_results[str(latency)] = {
                "status": "AVAILABLE",
                "latency_profile_status": "ASSUMED_COUNTERFACTUAL",
                "target": "INCENTIVE_FREE_GROSS_CANCEL_VALUE_SIGN",
                "population_audit": {
                    "preventable_train_rows": int(event_train.sum()),
                    "preventable_holdout_rows": int(event_eval.sum()),
                    "nonzero_value_train_rows": int(economic_train.sum()),
                    "nonzero_value_holdout_rows": int(economic_eval.sum()),
                    "neutral_preventable_train_rows":
                        int((event_train & ~economic_train).sum()),
                    "neutral_preventable_holdout_rows":
                        int((event_eval & ~economic_eval).sum()),
                },
                "preventable_fill_fit": hurdle._classification_metrics(
                    event_train.astype(int), event_eval.astype(int),
                    fill_probability),
                "value_weighted_harmful_fill": {
                    "fit": weighted_fit,
                    "policy": weighted_policy,
                    "conditional_economic_policy": _conditional_economic_report(
                        target_eval, weighted_score, economic_eval, days),
                },
                "unweighted_harmful_fill": {
                    "fit": unweighted_fit,
                    "policy": unweighted_policy,
                },
                "hurdle_v4": {"policy": v4_policy},
                "direct_tree_v3": {"policy": direct_policy},
                "value_weighted_vs_unweighted": vs_unweighted,
                "value_weighted_vs_hurdle_v4": vs_hurdle,
                "value_weighted_vs_direct_tree_v3": vs_direct,
                "development_signal_gate": gate,
            }
            latency_models[str(latency)] = {
                "preventable_fill": v4_model.preventable_fill.receipt(),
                "value_weighted_harmful_fill": weighted.receipt(),
                "unweighted_harmful_fill_comparator": unweighted.receipt(),
            }

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
        "schema_version": 5,
        "protocol": "BE_ADVERSE_MOVE_HARMFUL_FLOW_DEVELOPMENT_V5",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT",
        "verdict": "INSUFFICIENT_EVIDENCE",
        "decision_eligible": False,
        "candidate_frozen": False,
        "reasons": [
            "v5_uses_already_seen_v3_v4_development_days",
            "only_two_development_holdout_days_not_ten_forward_days",
            "cancellation_latency_rungs_are_assumed_not_measured_end_to_end",
            "cancel_rejoin_queue_value_is_outside_static_shadow_target",
            "action_score_sign_is_economic_but_magnitude_is_not_calibrated_cents",
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
            "target": "SIGN_OF_GROSS_MARKOUT_VALUE_OF_PREVENTED_FILL",
            "economic_sample_weight": "ABS_GROSS_CANCEL_VALUE_CENTS",
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
    print(f"[adverse-harmful] wrote {OUT}", flush=True)
    print(f"[adverse-harmful] signal cells "
          f"{result['development_signal_cells']}", flush=True)
    return result


def selftest() -> int:
    checks = 0

    def ok(condition: bool, name: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(name)
        checks += 1

    value = np.asarray([-3.0, -1.0, 2.0, 5.0])
    helpful = (value > 0).astype(int)
    weights = np.abs(value)
    q = _weighted_mean(helpful, weights)
    ok(q == 7.0 / 11.0 and q > 0.5 and value.mean() > 0,
       "value-weighted prevalence sign matches positive mean value")
    negative_value = np.asarray([-6.0, -1.0, 2.0, 3.0])
    negative_q = _weighted_mean(
        (negative_value > 0).astype(int), np.abs(negative_value))
    ok(negative_q < 0.5 and negative_value.mean() < 0,
       "value-weighted prevalence sign matches negative mean value")

    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(12_000, len(fast.FEATURE_NAMES))).astype(np.float32)
    nonlinear_score = x[:, 0] * x[:, 1] + 0.25 * x[:, 2] * x[:, 3]
    y = (nonlinear_score > 0).astype(int)
    sample_weight = 0.2 + np.abs(nonlinear_score)
    train = np.arange(8_000)
    test = np.arange(8_000, len(x))
    model = fit_classifier(
        x[train], y[train], "synthetic_harmful",
        "value_weighted_harmful_fill", sample_weight[train])
    ok(model is not None, "value-weighted classifier fits")
    assert model is not None
    prediction = model.predict(x[test])
    metrics = weighted_classification_metrics(
        y[train], sample_weight[train], y[test], sample_weight[test], prediction)
    ok(metrics["weighted_brier_skill_vs_training_weighted_prevalence"] > 0.2,
       "value-weighted classifier learns nonlinear harmful flow")
    ok(metrics["value_weighted_roc_auc"] > 0.8,
       "weighted harmful-flow ranking has correct orientation")
    audit = metrics["train_weight_audit"]
    ok(0 < audit["effective_sample_size"] <= len(train),
       "economic weight effective sample size is bounded")
    receipt = model.receipt()
    raw = zlib.decompress(base64.b64decode(receipt["model_text_zlib_b64"]))
    ok(hashlib.sha256(raw).hexdigest() == receipt["model_text_sha256"],
       "weighted model receipt round-trips")
    ok(receipt["feature_schema_hash"] == fast.FEATURE_SCHEMA_HASH,
       "weighted model binds feature schema")

    fill_probability = np.asarray([0.1, 0.5, 0.9])
    harmful_probability = np.asarray([0.2, 0.5, 0.8])
    score = fill_probability * (2.0 * harmful_probability - 1.0)
    ok(np.allclose(score, [-0.06, 0.0, 0.54]),
       "occurrence-aware score composition is exact")
    ok(np.array_equal(score > 0, harmful_probability > 0.5),
       "fill probability changes rank but not economic sign")

    fake_fit = {
        "weighted_brier_skill_vs_training_weighted_prevalence": 0.1}
    fake_policy = {
        "realized_delta_cents_per_decision": 1.0,
        "per_day_realized_delta_cents_per_decision": {
            HOLDOUT_DAYS[0]: 1.0, HOLDOUT_DAYS[1]: 1.0},
        "selection_gain_vs_training_selected_constant_cents_per_decision": 1.0,
        "per_day_selection_gain_cents_per_decision": {
            HOLDOUT_DAYS[0]: 1.0, HOLDOUT_DAYS[1]: 1.0},
        "cancel_fraction": 0.5,
    }
    fake_comparison = {
        "candidate_minus_comparator_cents_per_decision": 1.0,
        "per_day_cents_per_decision": {
            HOLDOUT_DAYS[0]: 1.0, HOLDOUT_DAYS[1]: 1.0},
    }
    ok(_signal_gate(
        fake_fit, fake_policy, fake_comparison, fake_comparison)["label"]
       == "HARMFUL_FLOW_SIGNAL_PRESENT", "all harmful-flow gates can pass")
    fake_policy["per_day_realized_delta_cents_per_decision"][
        HOLDOUT_DAYS[1]] = -0.1
    ok(_signal_gate(
        fake_fit, fake_policy, fake_comparison, fake_comparison)["label"]
       == "HARMFUL_FLOW_SIGNAL_NOT_ESTABLISHED",
       "negative realized value on one day fails gate")
    ok(MODEL_CONFIG["weight_winsorization"] is False,
       "economic weights are not outcome-tuned")
    ok(MODEL_CONFIG["early_stopping"] is False,
       "holdout cannot tune tree count")
    ok(REQUIRED_FORWARD_DAYS == 10, "forward validation floor preserved")
    print(f"[adverse-harmful] selftest OK — {checks} checks")
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
