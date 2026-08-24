"""Development adverse-selection fit and forward-day diagnostic.

Research only. This module consumes immutable recorded tapes, fits the
interpretable candidate pinned in plans/BE_ADVERSE_MOVE_PLAN.md section 6,
and writes a non-decision-eligible receipt. It has no exchange, order, or
cancellation port.

Commands:

    python3 live/pm_research/adverse_move.py --selftest
    python3 live/pm_research/adverse_move.py run --per-coin-day 1
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

import adverse_feature_rows as afr
import flow_intensity as fi
import warning_window as ww

OUT = fi.PM / "derived/adverse_move_development_v1.json"
PLAN = Path(__file__).with_name("plans") / "BE_ADVERSE_MOVE_PLAN.md"

TRAIN_DAYS = ("2026-08-20", "2026-08-21", "2026-08-22")
HOLDOUT_DAYS = ("2026-08-23", "2026-08-24")
COINS = ("btc", "eth")
LOGISTIC_C = 1.0
RIDGE_ALPHA = 10.0
N_CALIBRATION_BINS = 10
N_BOOT = 2_000
SEED = 20260824
REQUIRED_FORWARD_DAYS = 10

MODEL_CONFIG = {
    "feature_scaling": "StandardScaler fitted on training rows only",
    "toxic_fill": {
        "model": "LogisticRegression",
        "C": LOGISTIC_C,
        "class_weight": None,
        "solver": "lbfgs",
        "max_iter": 2_000,
    },
    "conditional_markout": {"model": "Ridge", "alpha": RIDGE_ALPHA},
    "avoidable_adverse": {"model": "Ridge", "alpha": RIDGE_ALPHA},
    "signed_cancel_value": {"model": "Ridge", "alpha": RIDGE_ALPHA},
    "hyperparameter_selection": "PINNED_BEFORE_DEVELOPMENT_HOLDOUT_NO_TUNING",
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


def _safe_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass
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
            "feature_names": list(afr.FEATURE_NAMES),
            "feature_schema_hash": afr.FEATURE_SCHEMA_HASH,
            "source_profile_hash": afr.SOURCE_PROFILE_HASH,
            "action_schema_hash": afr.ACTION_SCHEMA_HASH,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "coef": np.asarray(self.model.coef_).reshape(-1).tolist(),
            "intercept": np.asarray(self.model.intercept_).reshape(-1).tolist(),
            "params": {
                "C": LOGISTIC_C if isinstance(self.model, LogisticRegression) else None,
                "alpha": RIDGE_ALPHA if isinstance(self.model, Ridge) else None,
            },
        }


Pair = tuple[afr.AdverseFeatureRow, afr.ActionLabel]


def _matrix(pairs: Sequence[Pair]) -> np.ndarray:
    return np.asarray(
        [[row.values[name] for name in afr.FEATURE_NAMES] for row, _ in pairs],
        dtype=float,
    )


def fit_logistic(pairs: Sequence[Pair], target: str = "joint_toxic_fill"
                 ) -> ScaledModel | None:
    if len(pairs) < 2:
        return None
    y = np.asarray([label.toxic_fill for _, label in pairs], dtype=int)
    if np.unique(y).size != 2:
        return None
    scaler = StandardScaler()
    x = scaler.fit_transform(_matrix(pairs))
    model = LogisticRegression(
        C=LOGISTIC_C,
        class_weight=None,
        solver="lbfgs",
        max_iter=2_000,
    )
    model.fit(x, y)
    return ScaledModel(scaler, model, target)


def fit_ridge(pairs: Sequence[Pair], y: Sequence[float], target: str
              ) -> ScaledModel | None:
    if len(pairs) < 2 or len(pairs) != len(y):
        return None
    scaler = StandardScaler()
    x = scaler.fit_transform(_matrix(pairs))
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x, np.asarray(y, dtype=float))
    return ScaledModel(scaler, model, target)


def expected_calibration_error(y: Sequence[int], p: Sequence[float],
                               n_bins: int = N_CALIBRATION_BINS) -> float | None:
    if len(y) == 0:
        return None
    ya, pa = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    answer = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (pa >= lo) & ((pa < hi) if i + 1 < n_bins else (pa <= hi))
        if mask.any():
            answer += mask.mean() * abs(float(pa[mask].mean() - ya[mask].mean()))
    return float(answer)


def regression_metrics(y: Sequence[float], pred: Sequence[float]) -> dict[str, Any]:
    if len(y) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    ya, pa = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    denom = float(np.sum((ya - ya.mean()) ** 2))
    r2 = None if denom <= 0 else 1.0 - float(np.sum((ya - pa) ** 2)) / denom
    return {
        "n": len(ya),
        "mae": _safe_float(mean_absolute_error(ya, pa)),
        "rmse": _safe_float(math.sqrt(mean_squared_error(ya, pa))),
        "r2": _safe_float(r2) if r2 is not None else None,
        "target_mean": _safe_float(ya.mean()),
        "prediction_mean": _safe_float(pa.mean()),
    }


def day_cluster_mean_ci(records: Sequence[tuple[str, float]],
                        n_boot: int = N_BOOT,
                        seed: int = SEED) -> list[float] | None:
    by_day: dict[str, list[float]] = collections.defaultdict(list)
    for day, value in records:
        by_day[day].append(float(value))
    days = sorted(by_day)
    if not days:
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(days, size=len(days), replace=True)
        values = [v for day in sampled for v in by_day[str(day)]]
        draws[i] = np.mean(values)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def _policy_metrics(pairs: Sequence[Pair], prediction: Sequence[float],
                    latency: int,
                    train_target: Sequence[float]) -> dict[str, Any]:
    cancel = np.asarray(prediction) > 0.0
    realized = np.asarray([
        label.cancel_delta_cents[latency] if take else 0.0
        for (_, label), take in zip(pairs, cancel)
    ])
    prevented = np.asarray([
        label.prevented_shares[latency] if take else 0.0
        for (_, label), take in zip(pairs, cancel)
    ])
    filled = np.asarray([label.filled_shares > 0 for _, label in pairs])
    total = float(realized.sum())
    n_cancel = int(cancel.sum())
    prevented_total = float(prevented.sum())
    records = [(row.day, value) for (row, _), value in zip(pairs, realized)]
    oracle = np.asarray([
        max(0.0, label.cancel_delta_cents[latency]) for _, label in pairs
    ])
    always = np.asarray([
        label.cancel_delta_cents[latency] for _, label in pairs
    ])
    # This comparator is chosen from the training target only, then carried
    # unchanged into the holdout. It asks whether x_t selects better than a
    # constant action; positive PnL versus KEEP alone cannot answer that.
    baseline_cancel = bool(np.mean(train_target) > 0.0)
    baseline = always if baseline_cancel else np.zeros(len(pairs))
    incremental = realized - baseline
    incremental_records = [
        (row.day, value) for (row, _), value in zip(pairs, incremental)
    ]
    return {
        "rule": "CANCEL_IFF_PREDICTED_SIGNED_CANCEL_VALUE_CENTS_GT_0",
        "n_decisions": len(pairs),
        "n_cancel": n_cancel,
        "cancel_fraction": n_cancel / len(pairs) if pairs else None,
        "n_filled_decisions": int(filled.sum()),
        "prevented_shares": prevented_total,
        "realized_delta_total_cents": total,
        "realized_delta_cents_per_decision": total / len(pairs) if pairs else None,
        "day_clustered_95pct_ci_delta_cents_per_decision":
            day_cluster_mean_ci(records),
        "realized_delta_cents_per_cancel":
            total / n_cancel if n_cancel else None,
        "realized_delta_cents_per_filled_decision":
            float(realized[filled].sum() / filled.sum()) if filled.any() else None,
        "realized_delta_cents_per_prevented_share":
            total / prevented_total if prevented_total > 0 else None,
        "oracle_upper_bound_total_cents": float(oracle.sum()),
        "oracle_is_descriptive_only": True,
        "constant_baselines": {
            "never_cancel_delta_cents_per_decision": 0.0,
            "always_cancel_delta_cents_per_decision":
                float(always.mean()) if len(always) else None,
            "training_selected_rule":
                "ALWAYS_CANCEL" if baseline_cancel else "NEVER_CANCEL",
            "training_target_mean_cents_per_decision":
                float(np.mean(train_target)),
            "selected_delta_cents_per_decision":
                float(baseline.mean()) if len(baseline) else None,
        },
        "selection_gain_vs_training_selected_constant_cents_per_decision":
            float(incremental.mean()) if len(incremental) else None,
        "day_clustered_95pct_ci_selection_gain_vs_training_selected_constant":
            day_cluster_mean_ci(incremental_records),
    }


def _counts(pairs: Sequence[Pair]) -> dict[str, Any]:
    return {
        "n": len(pairs),
        "n_fill": sum(label.filled_shares > 0 for _, label in pairs),
        "n_toxic_fill": sum(label.toxic_fill for _, label in pairs),
        "filled_shares": sum(label.filled_shares for _, label in pairs),
        "by_day": dict(sorted(collections.Counter(row.day for row, _ in pairs).items())),
    }


def fit_coin(coin: str, pairs: Sequence[Pair]) -> dict[str, Any]:
    train = [(r, l) for r, l in pairs if r.day in TRAIN_DAYS]
    holdout = [(r, l) for r, l in pairs if r.day in HOLDOUT_DAYS]
    toxic = fit_logistic(train)
    filled_train = [(r, l) for r, l in train
                    if l.filled_shares > 0 and l.markout_cents_per_share is not None]
    filled_holdout = [(r, l) for r, l in holdout
                      if l.filled_shares > 0 and l.markout_cents_per_share is not None]
    markout = fit_ridge(
        filled_train,
        [float(l.markout_cents_per_share) for _, l in filled_train],
        "fill_conditional_markout_cents_per_share",
    )

    if toxic is None:
        classification: dict[str, Any] = {
            "status": "UNAVAILABLE",
            "reason": "TRAIN_TARGET_HAS_FEWER_THAN_TWO_CLASSES_OR_ROWS",
        }
    elif not holdout:
        classification = {"status": "UNAVAILABLE", "reason": "NO_HOLDOUT_ROWS"}
    else:
        pred = toxic.predict(_matrix(holdout))
        y = [l.toxic_fill for _, l in holdout]
        train_y = [l.toxic_fill for _, l in train]
        train_prevalence = sum(train_y) / len(train_y)
        baseline_brier = brier_score_loss(y, np.full(len(y), train_prevalence))
        model_brier = brier_score_loss(y, pred)
        classification = {
            "status": "AVAILABLE",
            "n": len(y),
            "prevalence": sum(y) / len(y),
            "training_prevalence": train_prevalence,
            "brier": _safe_float(model_brier),
            "training_prevalence_baseline_brier": _safe_float(baseline_brier),
            "brier_skill_vs_training_prevalence":
                _safe_float(1.0 - model_brier / baseline_brier)
                if baseline_brier > 0 else None,
            "ece_equal_width_10": expected_calibration_error(y, pred),
            "prediction_mean": _safe_float(pred.mean()),
        }

    if markout is None:
        markout_eval: dict[str, Any] = {
            "status": "UNAVAILABLE",
            "reason": "FEWER_THAN_TWO_FILLED_TRAIN_ROWS",
        }
    elif not filled_holdout:
        markout_eval = {"status": "UNAVAILABLE", "reason": "NO_FILLED_HOLDOUT_ROWS"}
    else:
        y_mark = [float(l.markout_cents_per_share) for _, l in filled_holdout]
        markout_eval = {
            "status": "AVAILABLE",
            **regression_metrics(y_mark, markout.predict(_matrix(filled_holdout))),
        }

    latency_results: dict[str, Any] = {}
    serialized_latency: dict[str, Any] = {}
    for latency in afr.LATENCY_MS:
        damage_y = [l.avoidable_adverse_cents[latency] for _, l in train]
        cancel_y = [l.cancel_delta_cents[latency] for _, l in train]
        damage = fit_ridge(train, damage_y, f"avoidable_adverse_cents_L{latency}ms")
        cancel_value = fit_ridge(
            train, cancel_y, f"signed_cancel_value_cents_L{latency}ms")
        if damage is None or cancel_value is None or not holdout:
            latency_results[str(latency)] = {
                "status": "UNAVAILABLE",
                "reason": "INSUFFICIENT_TRAIN_OR_HOLDOUT_ROWS",
            }
            continue
        x_holdout = _matrix(holdout)
        damage_pred = damage.predict(x_holdout)
        cancel_pred = cancel_value.predict(x_holdout)
        latency_results[str(latency)] = {
            "status": "AVAILABLE",
            "latency_profile_status": "ASSUMED_COUNTERFACTUAL",
            "avoidable_adverse_fit": regression_metrics(
                [l.avoidable_adverse_cents[latency] for _, l in holdout],
                damage_pred,
            ),
            "signed_cancel_value_fit": regression_metrics(
                [l.cancel_delta_cents[latency] for _, l in holdout],
                cancel_pred,
            ),
            "policy": _policy_metrics(holdout, cancel_pred, latency, cancel_y),
        }
        serialized_latency[str(latency)] = {
            "avoidable_adverse": damage.receipt(),
            "signed_cancel_value": cancel_value.receipt(),
        }

    models: dict[str, Any] = {
        "toxic_fill": toxic.receipt() if toxic else None,
        "conditional_markout": markout.receipt() if markout else None,
        "latencies": serialized_latency,
    }
    candidate = {
        "coin": coin,
        "status": "DEVELOPMENT",
        "decision_eligible": False,
        "training_days": list(TRAIN_DAYS),
        "model_config": MODEL_CONFIG,
        "models": models,
    }
    return {
        "status": "DEVELOPMENT",
        "verdict": "INSUFFICIENT_EVIDENCE",
        "decision_eligible": False,
        "candidate_hash": _sha(candidate),
        "population": {
            "train": _counts(train),
            "development_holdout": _counts(holdout),
        },
        "classification": classification,
        "conditional_markout": markout_eval,
        "latencies_ms": latency_results,
        "model_artifact": candidate,
    }


def _hf_file_manifest(rows: Sequence[afr.AdverseFeatureRow]) -> list[dict[str, Any]]:
    hours = sorted({(r.coin, r.as_of_ns // 3_600_000_000_000)
                    for r in rows if r.coin in COINS})
    symbols = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
    result: list[dict[str, Any]] = []
    for coin, hour_bucket in hours:
        hour = dt.datetime.fromtimestamp(
            hour_bucket * 3_600, dt.timezone.utc).strftime("%Y%m%d_%H")
        for stream in ("bookTicker", "trade", "depth20"):
            base = afr.HF_RAW / stream / symbols[coin]
            choices = (base / f"{hour}.csv.gz", base / f"{hour}.csv")
            path = next((p for p in choices if p.exists()), None)
            if path is None:
                result.append({
                    "coin": coin, "stream": stream, "hour": hour,
                    "status": "MISSING",
                })
            else:
                st = path.stat()
                result.append({
                    "coin": coin,
                    "stream": stream,
                    "hour": hour,
                    "path": str(path.relative_to(afr.REPO)),
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "status": "PRESENT",
                })
    return result


def build_dataset(per_coin_day: int
                  ) -> tuple[list[afr.AdverseFeatureRow], list[afr.ActionLabel],
                             list[Path], list[str]]:
    selected = ww.select_by_day(per_coin_day)
    rows: list[afr.AdverseFeatureRow] = []
    labels: list[afr.ActionLabel] = []
    sampled: list[Path] = []
    slugs: list[str] = []
    protocol_days = set(TRAIN_DAYS) | set(HOLDOUT_DAYS)
    chosen = [
        item
        for day, items in selected.items() if day in protocol_days
        for item in items if item[0].split("-")[0] in COINS
    ]
    for i, (slug, path, up, down, gaps) in enumerate(chosen, 1):
        print(f"[adverse] PM {i:02d}/{len(chosen):02d} {slug}", flush=True)
        tape = afr.build_pm_tape(path, up, down, gaps)
        feature_rows, action_labels = afr.materialize_pm_rows(tape)
        rows.extend(feature_rows)
        labels.extend(action_labels)
        sampled.append(path)
        slugs.append(slug)
    if not rows:
        raise RuntimeError("no protocol rows selected")
    afr.enrich_hf(rows)
    return rows, labels, sampled, slugs


def run(per_coin_day: int) -> dict[str, Any]:
    rows, labels, sampled, slugs = build_dataset(per_coin_day)
    label_by_id = {x.row_id: x for x in labels}
    feature_unavailable = collections.Counter(
        r.unavailable_reason for r in rows if r.status != "AVAILABLE")
    label_unavailable = collections.Counter(
        l.unavailable_reason for l in labels if l.status != "AVAILABLE")
    admitted: dict[str, list[Pair]] = collections.defaultdict(list)
    for row in rows:
        label = label_by_id[row.row_id]
        if row.status == "AVAILABLE" and label.status == "AVAILABLE":
            admitted[row.coin].append((row, label))

    hf_manifest = _hf_file_manifest(rows)
    source_identity = {
        "kind": "PATH_SIZE_MTIME_RECEIPT_NOT_CONTENT_DIGEST",
        "files": hf_manifest,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "BE_ADVERSE_MOVE_DEVELOPMENT_V1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT",
        "verdict": "INSUFFICIENT_EVIDENCE",
        "decision_eligible": False,
        "candidate_frozen": False,
        "reasons": [
            "design_and_available_days_were_seen_before_this_development_fit",
            "only_two_development_holdout_days_not_ten_strictly_forward_days",
            "cancellation_latency_ladder_is_assumed_not_measured_end_to_end",
            "liquidity_reward_opportunity_cost_is_unavailable",
            "cancel_rejoin_queue_value_is_unavailable",
        ],
        "split": {
            "training_days": list(TRAIN_DAYS),
            "development_holdout_days": list(HOLDOUT_DAYS),
            "required_future_days_after_candidate_freeze": REQUIRED_FORWARD_DAYS,
            "observed_strictly_forward_days": 0,
        },
        "action": {
            "size_shares": afr.ACTION_SIZE,
            "placement": "JOIN_TOUCH_BACK_DISPLAYED",
            "prediction_horizon_s": afr.PREDICTION_HORIZON_S,
            "markout_horizon_s": afr.MARKOUT_HORIZON_S,
            "markout_clock": "UNLAGGED_LOCAL_RECEIPT",
        },
        "latency": {
            "rungs_ms": list(afr.LATENCY_MS),
            "status": "ASSUMED_COUNTERFACTUAL",
            "tau_operative": None,
        },
        "economics": {
            "maker_rebate_cents_per_share": afr.MAKER_REBATE_CENTS_PER_SHARE,
            "gross_markout_includes_spread_capture": True,
            "liquidity_rewards": "UNAVAILABLE",
            "rejoin_queue_value": "UNAVAILABLE",
        },
        "model_config": MODEL_CONFIG,
        "feature_contract": {
            "names": list(afr.FEATURE_NAMES),
            "feature_schema_hash": afr.FEATURE_SCHEMA_HASH,
            "source_profile_hash": afr.SOURCE_PROFILE_HASH,
            "source_profile": afr.SOURCE_PROFILE,
            "action_schema_hash": afr.ACTION_SCHEMA_HASH,
            "completed_bucket_ms": afr.GRID_MS,
            "pm_feature_state_lag_ms": int(afr.fd.STATE_LAG_S * 1000),
        },
        "population": {
            "per_coin_day_requested": per_coin_day,
            "selected_slugs": slugs,
            "n_feature_rows": len(rows),
            "n_label_rows": len(labels),
            "n_admitted_pairs": sum(len(v) for v in admitted.values()),
            "feature_unavailable_reasons":
                {str(k): v for k, v in sorted(feature_unavailable.items())},
            "label_unavailable_reasons":
                {str(k): v for k, v in sorted(label_unavailable.items())},
        },
        "coins": {coin: fit_coin(coin, admitted.get(coin, [])) for coin in COINS},
        "provenance": {
            "polymarket": fi.provenance(sampled=sampled),
            "polymarket_files": [
                {
                    "path": str(p.relative_to(afr.REPO)),
                    "sha256": _file_sha(p),
                    "size": p.stat().st_size,
                }
                for p in sampled
            ],
            "hf_source_identity": source_identity,
            "code_sha256": _file_sha(Path(__file__)),
            "feature_builder_sha256": _file_sha(Path(afr.__file__)),
            "plan_sha256": _file_sha(PLAN),
        },
        "feature_samples_without_labels": [
            r.contract_dict() for r in rows if r.status == "AVAILABLE"
        ][:2],
    }
    result["artifact_id"] = _sha(result)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, allow_nan=False))
    print(f"[adverse] wrote {OUT}", flush=True)
    for coin, payload in result["coins"].items():
        c = payload["classification"]
        print(
            f"[adverse] {coin} train={payload['population']['train']['n']:,} "
            f"holdout={payload['population']['development_holdout']['n']:,} "
            f"toxic_brier={c.get('brier')} verdict=INSUFFICIENT_EVIDENCE",
            flush=True,
        )
    return result


def selftest() -> int:
    checks = 0

    def ok(value: bool, name: str) -> None:
        nonlocal checks
        if not value:
            raise AssertionError(name)
        checks += 1

    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(100, len(afr.FEATURE_NAMES)))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    pairs: list[Pair] = []
    for i in range(len(x)):
        values = dict(zip(afr.FEATURE_NAMES, x[i]))
        row = afr.AdverseFeatureRow(
            str(i), str(i), "btc-updown-5m-1780000000", "btc",
            "2026-08-20", i * afr.BUCKET_NS, i / 10, "BUY_UP", values,
            afr.FEATURE_SCHEMA_HASH, i * afr.BUCKET_NS, 0.0, str(i),
            afr.SOURCE_PROFILE_HASH,
        )
        zeros = {latency: 0.0 for latency in afr.LATENCY_MS}
        label = afr.ActionLabel(
            str(i), "AVAILABLE", None, float(y[i]), int(y[i]), float(x[i, 0]),
            float(x[i, 0]), dict(zeros), dict(zeros), dict(zeros),
        )
        pairs.append((row, label))
    classifier = fit_logistic(pairs)
    ok(classifier is not None, "two-class logistic fits")
    assert classifier is not None
    pred = classifier.predict(_matrix(pairs))
    ok(float(brier_score_loss(y, pred)) < 0.15, "logistic learns synthetic target")
    ridge = fit_ridge(pairs, x[:, 2], "synthetic")
    ok(ridge is not None, "ridge fits")
    assert ridge is not None
    r2 = regression_metrics(x[:, 2], ridge.predict(_matrix(pairs)))["r2"]
    ok(r2 is not None and r2 > 0.95, "ridge learns synthetic target")
    ece = expected_calibration_error([0, 1], [0.1, 0.9])
    ok(ece is not None and abs(ece - 0.1) < 1e-12, "ECE units and binning")
    ci = day_cluster_mean_ci([("a", 1.0), ("a", 1.0), ("b", -1.0)])
    ok(ci is not None and ci[0] <= ci[1], "day-cluster interval")
    receipt = classifier.receipt()
    ok(receipt["feature_schema_hash"] == afr.FEATURE_SCHEMA_HASH,
       "model binds feature schema")
    ok(receipt["source_profile_hash"] == afr.SOURCE_PROFILE_HASH,
       "model binds source profile")
    ok(receipt["action_schema_hash"] == afr.ACTION_SCHEMA_HASH,
       "model binds action and label semantics")
    ok(MODEL_CONFIG["toxic_fill"]["class_weight"] is None,
       "natural prevalence pinned")
    ok(REQUIRED_FORWARD_DAYS == 10, "promotion floor remains ten days")
    print(f"[adverse] selftest OK — {checks} checks")
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
