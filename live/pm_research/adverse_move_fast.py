"""Fast-path v2 adverse-selection development fit.

Research only. Fits exact-event models from recorded BTC/ETH Polymarket and
Binance feeds and writes a non-decision-eligible receipt. No live adapter,
order sender, cancellation command, or execution server exists here.

Commands:

    python3 live/pm_research/adverse_move_fast.py --selftest
    python3 live/pm_research/adverse_move_fast.py run --per-coin-day 1
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import adverse_feature_rows_fast as fast
import flow_intensity as fi
import warning_window as ww

OUT = fi.PM / "derived/adverse_move_fast_development_v2.json"
PLAN = Path(__file__).with_name("plans") / "BE_ADVERSE_MOVE_PLAN.md"

TRAIN_DAYS = ("2026-08-20", "2026-08-21", "2026-08-22")
HOLDOUT_DAYS = ("2026-08-23", "2026-08-24")
COINS = ("btc", "eth")
LOGISTIC_C = 1.0
RIDGE_ALPHA = 10.0
N_BOOT = 2_000
SEED = 20260824
REQUIRED_FORWARD_DAYS = 10

MODEL_CONFIG = {
    "feature_scaling": "StandardScaler per coin/horizon on training rows only",
    "toxic_fill": {
        "model": "LogisticRegression",
        "C": LOGISTIC_C,
        "class_weight": None,
        "solver": "lbfgs",
        "max_iter": 2_000,
    },
    "signed_cancel_value": {"model": "Ridge", "alpha": RIDGE_ALPHA},
    "hyperparameter_selection": "PINNED_BEFORE_V2_DEVELOPMENT_RUN_NO_TUNING",
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
class ScaledModel:
    scaler: StandardScaler
    model: LogisticRegression | Ridge
    target: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(x)
        if isinstance(self.model, LogisticRegression):
            return self.model.predict_proba(z)[:, 1]
        return self.model.predict(z)

    def receipt(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "kind": type(self.model).__name__,
            "feature_names": list(fast.FEATURE_NAMES),
            "feature_schema_hash": fast.FEATURE_SCHEMA_HASH,
            "source_profile_hash": fast.SOURCE_PROFILE_HASH,
            "action_schema_hash": fast.ACTION_SCHEMA_HASH,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "coef": np.asarray(self.model.coef_).reshape(-1).tolist(),
            "intercept": np.asarray(self.model.intercept_).reshape(-1).tolist(),
            "params": {
                "C": LOGISTIC_C if isinstance(self.model, LogisticRegression) else None,
                "alpha": RIDGE_ALPHA if isinstance(self.model, Ridge) else None,
            },
        }


def fit_logistic(x: np.ndarray, y: np.ndarray, target: str
                 ) -> ScaledModel | None:
    if len(y) < 2 or np.unique(y).size != 2:
        return None
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    model = LogisticRegression(
        C=LOGISTIC_C,
        class_weight=None,
        solver="lbfgs",
        max_iter=2_000,
    )
    model.fit(z, y)
    return ScaledModel(scaler, model, target)


def fit_ridge(x: np.ndarray, y: np.ndarray, target: str,
              scaler: StandardScaler | None = None) -> ScaledModel | None:
    if len(y) < 2:
        return None
    use_scaler = scaler or StandardScaler().fit(x)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(use_scaler.transform(x), y)
    return ScaledModel(use_scaler, model, target)


def expected_calibration_error(y: np.ndarray, prediction: np.ndarray,
                               n_bins: int = 10) -> float | None:
    if len(y) == 0:
        return None
    answer = 0.0
    for index in range(n_bins):
        lo, hi = index / n_bins, (index + 1) / n_bins
        mask = ((prediction >= lo)
                & ((prediction < hi) if index + 1 < n_bins else prediction <= hi))
        if mask.any():
            answer += float(mask.mean()) * abs(
                float(prediction[mask].mean() - y[mask].mean()))
    return answer


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if len(y) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = None if denominator <= 0 else (
        1.0 - float(np.sum((y - prediction) ** 2)) / denominator)
    return {
        "n": len(y),
        "target_mean": _safe(y.mean()),
        "prediction_mean": _safe(prediction.mean()),
        "mae": _safe(mean_absolute_error(y, prediction)),
        "rmse": _safe(math.sqrt(mean_squared_error(y, prediction))),
        "r2": _safe(r2) if r2 is not None else None,
    }


def day_cluster_mean_ci(days: np.ndarray, values: np.ndarray,
                        n_boot: int = N_BOOT,
                        seed: int = SEED) -> list[float] | None:
    unique = np.unique(days)
    if len(unique) == 0:
        return None
    groups = {day: values[days == day] for day in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for index in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws[index] = np.concatenate([groups[day] for day in sampled]).mean()
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _concat_x(batches: Sequence[fast.FastWindowBatch]) -> np.ndarray:
    if not batches:
        return np.empty((0, len(fast.FEATURE_NAMES)), dtype=np.float32)
    return np.concatenate([batch.x for batch in batches])


def _concat_target(batches: Sequence[fast.FastWindowBatch],
                   values: str, key: Any) -> np.ndarray:
    if not batches:
        return np.asarray([])
    return np.concatenate([getattr(batch, values)[key] for batch in batches])


def _day_codes(batches: Sequence[fast.FastWindowBatch]) -> np.ndarray:
    if not batches:
        return np.asarray([], dtype=np.int8)
    mapping = {day: index for index, day in enumerate(HOLDOUT_DAYS)}
    return np.concatenate([
        np.full(batch.n_rows, mapping.get(batch.day, -1), dtype=np.int8)
        for batch in batches
    ])


def policy_metrics(target_train: np.ndarray, target_eval: np.ndarray,
                   prediction: np.ndarray, days: np.ndarray,
                   prevented: np.ndarray, filled: np.ndarray) -> dict[str, Any]:
    cancel = prediction > 0.0
    realized = np.where(cancel, target_eval, 0.0)
    baseline_cancel = bool(target_train.mean() > 0.0)
    baseline = target_eval if baseline_cancel else np.zeros(len(target_eval))
    incremental = realized - baseline
    total_prevented = float(np.where(cancel, prevented, 0.0).sum())
    total_filled = float(filled.sum())
    return {
        "rule": "CANCEL_IFF_PREDICTED_SIGNED_CANCEL_VALUE_CENTS_GT_0",
        "n_decisions": len(target_eval),
        "n_cancel": int(cancel.sum()),
        "cancel_fraction": float(cancel.mean()),
        "realized_delta_cents_per_decision": float(realized.mean()),
        "training_selected_constant_rule":
            "ALWAYS_CANCEL" if baseline_cancel else "NEVER_CANCEL",
        "training_target_mean_cents_per_decision": float(target_train.mean()),
        "constant_delta_cents_per_decision": float(baseline.mean()),
        "selection_gain_vs_training_selected_constant_cents_per_decision":
            float(incremental.mean()),
        "day_clustered_95pct_ci_selection_gain":
            day_cluster_mean_ci(days, incremental),
        "prevented_shares_under_policy": total_prevented,
        "filled_shares_if_kept": total_filled,
        "policy_prevented_share_fraction":
            total_prevented / total_filled if total_filled > 0 else None,
        "counterfactual_latency_survival_fraction":
            float(prevented.sum() / total_filled) if total_filled > 0 else None,
        "oracle_upper_bound_cents_per_decision":
            float(np.maximum(target_eval, 0.0).mean()),
        "oracle_is_descriptive_only": True,
    }


def fit_coin(coin: str, batches: Sequence[fast.FastWindowBatch]) -> dict[str, Any]:
    train_batches = [batch for batch in batches if batch.day in TRAIN_DAYS]
    holdout_batches = [batch for batch in batches if batch.day in HOLDOUT_DAYS]
    x_train_all = _concat_x(train_batches)
    x_holdout_all = _concat_x(holdout_batches)
    holdout_days_all = _day_codes(holdout_batches)
    horizons: dict[str, Any] = {}
    serialized: dict[str, Any] = {}

    for horizon in fast.PREDICTION_HORIZONS_MS:
        toxic_train_all = _concat_target(train_batches, "toxic", horizon)
        toxic_holdout_all = _concat_target(holdout_batches, "toxic", horizon)
        train_mask = toxic_train_all >= 0
        holdout_mask = toxic_holdout_all >= 0
        x_train = x_train_all[train_mask]
        x_holdout = x_holdout_all[holdout_mask]
        toxic_train = toxic_train_all[train_mask].astype(int)
        toxic_holdout = toxic_holdout_all[holdout_mask].astype(int)
        holdout_days = holdout_days_all[holdout_mask]
        filled_holdout = _concat_target(
            holdout_batches, "filled_shares", horizon)[holdout_mask]

        classifier = fit_logistic(
            x_train, toxic_train, f"joint_toxic_fill_H{horizon}ms")
        if classifier is None or len(toxic_holdout) == 0:
            classification: dict[str, Any] = {
                "status": "UNAVAILABLE",
                "reason": "INSUFFICIENT_TWO_CLASS_TRAIN_OR_HOLDOUT_ROWS",
            }
        else:
            probability = classifier.predict(x_holdout)
            train_prevalence = float(toxic_train.mean())
            baseline = np.full(len(toxic_holdout), train_prevalence)
            model_brier = float(brier_score_loss(toxic_holdout, probability))
            baseline_brier = float(brier_score_loss(toxic_holdout, baseline))
            classification = {
                "status": "AVAILABLE",
                "n_train": len(toxic_train),
                "n_holdout": len(toxic_holdout),
                "train_prevalence": train_prevalence,
                "holdout_prevalence": float(toxic_holdout.mean()),
                "holdout_fill_fraction": float((filled_holdout > 0).mean()),
                "brier": model_brier,
                "training_prevalence_baseline_brier": baseline_brier,
                "brier_skill_vs_training_prevalence":
                    1.0 - model_brier / baseline_brier
                    if baseline_brier > 0 else None,
                "ece_equal_width_10":
                    expected_calibration_error(toxic_holdout, probability),
                "prediction_mean": float(probability.mean()),
            }

        latency_results: dict[str, Any] = {}
        latency_models: dict[str, Any] = {}
        scaler = classifier.scaler if classifier is not None else None
        for latency in fast.LATENCY_MS:
            if latency >= horizon:
                continue
            key = (horizon, latency)
            cancel_train_all = _concat_target(train_batches, "cancel_delta", key)
            cancel_holdout_all = _concat_target(holdout_batches, "cancel_delta", key)
            prevented_holdout_all = _concat_target(
                holdout_batches, "prevented_shares", key)
            cancel_train = cancel_train_all[train_mask]
            cancel_holdout = cancel_holdout_all[holdout_mask]
            prevented_holdout = prevented_holdout_all[holdout_mask]
            finite_train = np.isfinite(cancel_train)
            finite_holdout = np.isfinite(cancel_holdout) & np.isfinite(prevented_holdout)
            model = fit_ridge(
                x_train[finite_train], cancel_train[finite_train],
                f"signed_cancel_value_H{horizon}ms_L{latency}ms",
                scaler=scaler,
            )
            if model is None or not finite_holdout.any():
                latency_results[str(latency)] = {
                    "status": "UNAVAILABLE",
                    "reason": "INSUFFICIENT_TRAIN_OR_HOLDOUT_ROWS",
                }
                continue
            prediction = model.predict(x_holdout[finite_holdout])
            target_eval = cancel_holdout[finite_holdout].astype(float)
            target_train = cancel_train[finite_train].astype(float)
            latency_results[str(latency)] = {
                "status": "AVAILABLE",
                "latency_profile_status": "ASSUMED_COUNTERFACTUAL",
                "direct_cancel_value_fit": regression_metrics(
                    target_eval, prediction),
                "policy": policy_metrics(
                    target_train,
                    target_eval,
                    prediction,
                    holdout_days[finite_holdout],
                    prevented_holdout[finite_holdout].astype(float),
                    filled_holdout[finite_holdout].astype(float),
                ),
            }
            latency_models[str(latency)] = model.receipt()

        model_artifact = {
            "classification": classifier.receipt() if classifier else None,
            "latencies": latency_models,
        }
        horizons[str(horizon)] = {
            "classification": classification,
            "latencies_ms": latency_results,
        }
        serialized[str(horizon)] = model_artifact

    candidate = {
        "coin": coin,
        "status": "DEVELOPMENT",
        "decision_eligible": False,
        "training_days": list(TRAIN_DAYS),
        "model_config": MODEL_CONFIG,
        "models_by_horizon_ms": serialized,
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
        "horizons_ms": horizons,
        "model_artifact": candidate,
    }


def _hf_manifest(slugs: Sequence[str]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for slug in slugs:
        coin = slug.split("-")[0]
        symbol = {"btc": "BTCUSDT", "eth": "ETHUSDT"}[coin]
        start = int(slug.rsplit("-", 1)[1])
        for key in fast._hour_keys(
                start * 1_000_000_000,
                (start + int(fi.WINDOW_S)) * 1_000_000_000):
            for stream in ("bookTicker", "trade", "depth20"):
                identity = coin, stream, key
                if identity in seen:
                    continue
                seen.add(identity)
                root = fast.HF_RAW / stream / symbol
                path = next((candidate for candidate in (
                    root / f"{key}.csv.gz", root / f"{key}.csv")
                    if candidate.exists()), None)
                if path is None:
                    result.append({
                        "coin": coin, "stream": stream, "hour": key,
                        "status": "MISSING",
                    })
                else:
                    stat = path.stat()
                    result.append({
                        "coin": coin,
                        "stream": stream,
                        "hour": key,
                        "path": str(path.relative_to(fast.REPO)),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "status": "PRESENT",
                    })
    return result


def build_batches(per_coin_day: int
                  ) -> tuple[list[fast.FastWindowBatch], list[Path], list[str]]:
    selected = ww.select_by_day(per_coin_day)
    protocol_days = set(TRAIN_DAYS) | set(HOLDOUT_DAYS)
    chosen = [
        item
        for day, items in selected.items() if day in protocol_days
        for item in items if item[0].split("-")[0] in COINS
    ]
    batches: list[fast.FastWindowBatch] = []
    sampled: list[Path] = []
    slugs: list[str] = []
    for index, (slug, path, up, down, gaps) in enumerate(chosen, 1):
        print(f"[adverse-fast] {index:02d}/{len(chosen):02d} {slug}", flush=True)
        batches.append(fast.load_and_materialize(path, up, down, gaps))
        sampled.append(path)
        slugs.append(slug)
    if not batches:
        raise RuntimeError("no fast-path protocol windows selected")
    return batches, sampled, slugs


def run(per_coin_day: int) -> dict[str, Any]:
    batches, sampled, slugs = build_batches(per_coin_day)
    diagnostics: collections.Counter[str] = collections.Counter()
    for batch in batches:
        diagnostics.update(batch.diagnostics)
    result: dict[str, Any] = {
        "schema_version": 2,
        "protocol": "BE_ADVERSE_MOVE_FAST_DEVELOPMENT_V2",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT",
        "verdict": "INSUFFICIENT_EVIDENCE",
        "decision_eligible": False,
        "candidate_frozen": False,
        "reasons": [
            "v2_was_designed_after_v1_and_all_evaluation_days_were_seen",
            "only_two_development_holdout_days_not_ten_strictly_forward_days",
            "cancellation_latency_rungs_are_assumed_not_measured_end_to_end",
            "liquidity_reward_opportunity_cost_is_unavailable",
            "cancel_rejoin_queue_value_is_unavailable",
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
            "pm_artificial_feature_lag_ms": 0,
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
            "maker_rebate_cents_per_share": fast.MAKER_REBATE_CENTS_PER_SHARE,
            "gross_markout_includes_spread_capture": True,
            "liquidity_rewards": "UNAVAILABLE",
            "rejoin_queue_value": "UNAVAILABLE",
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
                "files": _hf_manifest(slugs),
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
    result["artifact_id"] = _sha(result)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, allow_nan=False))
    print(f"[adverse-fast] wrote {OUT}", flush=True)
    for coin, coin_result in result["coins"].items():
        for horizon in fast.PREDICTION_HORIZONS_MS:
            classification = coin_result["horizons_ms"][str(horizon)]["classification"]
            print(
                f"[adverse-fast] {coin} H={horizon:4d}ms "
                f"brier_skill={classification.get('brier_skill_vs_training_prevalence')} "
                f"verdict=INSUFFICIENT_EVIDENCE",
                flush=True,
            )
    return result


def selftest() -> int:
    checks = 0

    def ok(condition: bool, name: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(name)
        checks += 1

    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(300, len(fast.FEATURE_NAMES)))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    classifier = fit_logistic(x, y, "synthetic")
    ok(classifier is not None, "two-class fast logistic fits")
    assert classifier is not None
    prediction = classifier.predict(x)
    ok(brier_score_loss(y, prediction) < 0.1,
       "fast logistic learns synthetic target")
    ridge = fit_ridge(x, x[:, 2], "synthetic-ridge", classifier.scaler)
    ok(ridge is not None, "fast ridge fits")
    assert ridge is not None
    metrics = regression_metrics(x[:, 2], ridge.predict(x))
    ok(metrics["r2"] is not None and metrics["r2"] > 0.98,
       "fast ridge learns synthetic target")
    ci = day_cluster_mean_ci(
        np.asarray([0, 0, 1]), np.asarray([1.0, 1.0, -1.0]))
    ok(ci is not None and ci[0] <= ci[1], "day-cluster CI")
    receipt = classifier.receipt()
    ok(receipt["feature_schema_hash"] == fast.FEATURE_SCHEMA_HASH,
       "model binds fast feature schema")
    ok(receipt["source_profile_hash"] == fast.SOURCE_PROFILE_HASH,
       "model binds exact-event source profile")
    ok(receipt["action_schema_hash"] == fast.ACTION_SCHEMA_HASH,
       "model binds multi-horizon action schema")
    ok(all(latency < horizon
           for horizon in fast.PREDICTION_HORIZONS_MS
           for latency in fast.LATENCY_MS if latency < horizon),
       "latency grid is horizon-gated")
    ok(MODEL_CONFIG["toxic_fill"]["class_weight"] is None,
       "natural class prevalence pinned")
    ok(REQUIRED_FORWARD_DAYS == 10, "forward-day floor preserved")
    print(f"[adverse-fast] selftest OK — {checks} checks")
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
