"""Iteration 005: action-conditioned harmful-flow model and shadow replay.

Research only.  Builds labels from the actual order generations on a
queue-realistic no-cancel shadow path, fits the frozen value-weighted model,
and feeds its signals to the existing offline cancel-and-hold engine.
"""

from __future__ import annotations

import argparse
import base64
import bisect
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

import adverse_feature_rows as v1
import adverse_feature_rows_fast as fast
import adverse_move_fast as linear
import adverse_move_harmful as v5
import edge_layer1 as el
import flow_fill_development as fd
import flow_intensity as fi
import policy_bounds_v1 as pb
import policy_optimizer_cancel_skew as old_policy
import policy_optimizer_queue_realistic as qr
import warning_window as ww


OUT = fi.PM / "derived/policy_optimizer_queue_action_harmful_v1.json"
PROTOCOL = Path(__file__).with_name("QUEUE_ACTION_HARMFUL_PROTOCOL.md")
PARENT_ARTIFACT = qr.OUT
OLD_MODEL_ARTIFACT = old_policy.MODEL_ARTIFACT
CANDIDATE = "QR_CANCEL_QACT_X_SKEW"
ACTION_FEATURE_NAMES = (
    "actual_queue_ahead_log1p",
    "actual_resting_fraction_of_quote",
    "actual_order_age_ms",
    "actual_filled_fraction_of_quote",
    "maker_signed_inventory_quotes",
    "absolute_inventory_quotes",
)
FEATURE_NAMES = fast.FEATURE_NAMES + ACTION_FEATURE_NAMES
FEATURE_SCHEMA_HASH = fast._stable_hash({
    "base_schema": fast.FEATURE_SCHEMA_HASH,
    "action_features": ACTION_FEATURE_NAMES,
    "behavior": qr.QR_SKEW,
})
CELLS = old_policy.CANDIDATE_CELLS
TRAIN_DAYS = v5.TRAIN_DAYS
DEV_DAYS = v5.HOLDOUT_DAYS
EPS = 1e-9


def _sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


@dataclass(slots=True)
class TraceFill:
    t: float
    maker_side: str
    generation: int
    level: float
    size: float


@dataclass(slots=True)
class ActionBatch:
    base: fast.FastWindowBatch
    base_index: np.ndarray
    x: np.ndarray
    target: np.ndarray
    prevented: np.ndarray
    available: np.ndarray
    generation: np.ndarray
    diagnostics: dict[str, int]

    @property
    def coin(self) -> str:
        return self.base.coin

    @property
    def day(self) -> str:
        return self.base.day

    @property
    def slug(self) -> str:
        return self.base.slug


class TraceArm(qr.QueueRealisticArm):
    """No-cancel arm with point-in-time generation birth clocks."""

    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        self.clock = -60.0
        self.placed_at: dict[str, float | None] = {
            side: None for side in self.SIDES}

    def reposition(self, maker_side: str, level: float | None,
                   displayed: float) -> None:
        super().reposition(maker_side, level, displayed)
        self.placed_at[maker_side] = self.clock if level is not None else None

    def consume(self, maker_side: str, volume: float,
                displayed: float) -> float:
        before = self.side(maker_side).resting
        filled = super().consume(maker_side, volume, displayed)
        if filled > 0 and before <= filled + 1e-12:
            self.placed_at[maker_side] = self.clock
        return filled


def _quote(state: v1.PMState | None
           ) -> tuple[float, float, float, float, float] | None:
    if state is None:
        return None
    return state.bid, state.ask, state.bid_size, state.ask_size, state.tick


def trace_action_batch(base: fast.FastWindowBatch, tape: v1.PMTape
                       ) -> ActionBatch:
    """Attach actual QR_SKEW order state and same-generation future value."""
    latency_ms = CELLS[base.coin]["latency_ms"]
    horizon_ms = CELLS[base.coin]["horizon_ms"]
    arm = TraceArm(qr._qr_spec(qr.QR_SKEW, latency_ms, False))
    current: v1.PMState | None = None
    fills: list[TraceFill] = []
    snapshots: dict[int, tuple[int, list[float]]] = {}
    diag: collections.Counter[str] = collections.Counter()
    by_ns: dict[int, list[int]] = collections.defaultdict(list)
    for index, value in enumerate(base.as_of_ns):
        by_ns[int(value)].append(index)
    window_ns = tape.window_start * 1_000_000_000

    events: list[tuple[float, int, int]] = []
    events.extend((trade.t, 0, index)
                  for index, trade in enumerate(tape.trades))
    events.extend((when, 1, index)
                  for index, when in enumerate(tape.state_t))
    events.extend(((event_ns - window_ns) / 1e9, 2, event_ns)
                  for event_ns in by_ns)
    events.sort(key=lambda item: (item[0], item[1]))

    def sync(when: float) -> None:
        arm.clock = when
        quote = _quote(current)
        if quote is None:
            for side in arm.SIDES:
                if arm.side(side).level is not None:
                    arm.reposition(side, None, 0.0)
                    arm.placement_kind[side] = qr.ABSENT
            return
        arm.sync_from_quote(quote)

    for when, kind, ref in events:
        arm.clock = when
        if kind == 0:
            trade = tape.trades[ref]
            quote = _quote(current)
            if quote is None:
                continue
            if (trade.taker_side == "BUY" and arm.sell.level is not None
                    and trade.exec_p_up + 1e-12 >= arm.sell.level):
                generation, level = arm.generation["SELL_UP"], arm.sell.level
                got = arm.consume("SELL_UP", trade.size,
                                  arm.displayed_for_fill("SELL_UP", quote))
                if got > 0:
                    fills.append(TraceFill(when, "SELL_UP", generation,
                                           float(level), got))
                    arm.led_q_dn += got
                    arm.apply_skew_intent()
            elif (trade.taker_side == "SELL" and arm.buy.level is not None
                  and trade.exec_p_up <= arm.buy.level + 1e-12):
                generation, level = arm.generation["BUY_UP"], arm.buy.level
                got = arm.consume("BUY_UP", trade.size,
                                  arm.displayed_for_fill("BUY_UP", quote))
                if got > 0:
                    fills.append(TraceFill(when, "BUY_UP", generation,
                                           float(level), got))
                    arm.led_q_up += got
                    arm.apply_skew_intent()
            continue
        if kind == 1:
            current = tape.states[ref]
            sync(when)
            continue

        event_ns = ref
        sync(when)  # signal decisions occur after same-time PM events/resync
        for base_i in by_ns[event_ns]:
            maker_side = ("BUY_UP" if base.maker_side_sign[base_i] > 0
                          else "SELL_UP")
            side = arm.side(maker_side)
            if side.level is None or side.resting <= 1e-12:
                diag["ABSENT_OR_EMPTY"] += 1
                continue
            if arm.placement_kind[maker_side] != qr.JOIN_EXISTING:
                diag["NOT_JOIN_EXISTING"] += 1
                continue
            if arm.target_reducing(maker_side):
                diag["INVENTORY_REDUCING"] += 1
                continue
            born = arm.placed_at[maker_side]
            if born is None:
                raise AssertionError("live trace order lacks birth time")
            sign = float(el.maker_sign(maker_side))
            action = [
                math.log1p(max(0.0, side.qahead)),
                side.resting / float(side.size),
                max(0.0, (when - born) * 1000.0),
                arm.filled_current_order[maker_side] / float(side.size),
                sign * arm.net / float(side.size),
                abs(arm.net) / float(side.size),
            ]
            snapshots[base_i] = (arm.generation[maker_side], action)

    fills_by_generation: dict[tuple[str, int], list[TraceFill]] = (
        collections.defaultdict(list))
    for fill in fills:
        fills_by_generation[(fill.maker_side, fill.generation)].append(fill)

    indices: list[int] = []
    vectors: list[np.ndarray] = []
    targets: list[float] = []
    prevented: list[float] = []
    available: list[bool] = []
    generations: list[int] = []
    for base_i in sorted(snapshots):
        generation, action = snapshots[base_i]
        maker_side = ("BUY_UP" if base.maker_side_sign[base_i] > 0
                      else "SELL_UP")
        start = (int(base.as_of_ns[base_i]) - window_ns) / 1e9
        end = start + horizon_ms / 1000.0
        okay = not tape.touched(start, end)
        marked: list[tuple[float, float, float]] = []
        if okay:
            for fill in fills_by_generation.get((maker_side, generation), ()):
                if fill.t <= start + 1e-12 or fill.t > end + 1e-12:
                    continue
                if tape.touched(fill.t, fill.t + fast.MARKOUT_HORIZON_S):
                    okay = False
                    diag["LABEL_GAP_OR_TICK"] += 1
                    break
                later = tape.mark_state_at(fill.t + fast.MARKOUT_HORIZON_S)
                if later is None:
                    okay = False
                    diag["LABEL_NO_MARKOUT"] += 1
                    break
                markout = (el.maker_sign(maker_side)
                           * (later.mid - fill.level) * 100.0)
                marked.append((fill.t, fill.size, markout))
        effective = start + latency_ms / 1000.0
        eligible = [(t, size, value) for t, size, value in marked
                    if t >= effective - 1e-12]
        indices.append(base_i)
        vectors.append(np.concatenate((base.x[base_i],
                                       np.asarray(action, dtype=np.float32))))
        targets.append(-sum(size * value for _, size, value in eligible)
                       if okay else math.nan)
        prevented.append(sum(size for _, size, _ in eligible)
                         if okay else math.nan)
        available.append(okay)
        generations.append(generation)
    x = (np.asarray(vectors, dtype=np.float32)
         if vectors else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32))
    return ActionBatch(
        base, np.asarray(indices, dtype=np.int64), x,
        np.asarray(targets, dtype=float), np.asarray(prevented, dtype=float),
        np.asarray(available, dtype=bool), np.asarray(generations, dtype=np.int64),
        dict(diag))


def _fit_model(x: np.ndarray, target: np.ndarray,
               prevented: np.ndarray) -> lgb.LGBMClassifier:
    economic = ((prevented > EPS) & (np.abs(target) > EPS)
                & np.isfinite(target))
    y = (target[economic] > 0).astype(int)
    weight = np.abs(target[economic])
    if len(y) < 2 or np.unique(y).size != 2:
        raise RuntimeError("insufficient two-class economic action rows")
    model = lgb.LGBMClassifier(
        objective="binary", class_weight=None, **v5.TREE_PARAMS)
    model.fit(x[economic], y, sample_weight=weight)
    return model


def _receipt(model: lgb.LGBMClassifier, coin: str) -> dict[str, Any]:
    raw = model.booster_.model_to_string().encode()
    return {
        "coin": coin,
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "base_feature_schema_hash": fast.FEATURE_SCHEMA_HASH,
        "params": {"objective": "binary", "class_weight": None,
                   **v5.TREE_PARAMS},
        "model_text_sha256": hashlib.sha256(raw).hexdigest(),
        "model_text_zlib_b64": base64.b64encode(
            zlib.compress(raw, level=9)).decode(),
        "feature_importance_gain": model.booster_.feature_importance(
            importance_type="gain").astype(float).tolist(),
    }


def _old_harmful_model(artifact: dict[str, Any], coin: str) -> lgb.Booster:
    cell = CELLS[coin]
    receipt = artifact["coins"][coin]["model_artifact"][
        "models_by_horizon_ms"][str(cell["horizon_ms"])]["latencies"][
            str(cell["latency_ms"])]["value_weighted_harmful_fill"]
    return old_policy._load_booster(receipt)


def _daily(days: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    return {day: (float(values[days == day].mean())
                  if np.any(days == day) else None)
            for day in DEV_DAYS}


def _fit_and_score(coin: str, batches: Sequence[ActionBatch],
                   old_artifact: dict[str, Any]
                   ) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    train = [b for b in batches if b.day in TRAIN_DAYS]
    dev = [b for b in batches if b.day in DEV_DAYS]
    x_train = np.concatenate([b.x[b.available] for b in train])
    y_train = np.concatenate([b.target[b.available] for b in train])
    p_train = np.concatenate([b.prevented[b.available] for b in train])
    model = _fit_model(x_train, y_train, p_train)
    economic_train = (p_train > EPS) & (np.abs(y_train) > EPS)

    x_dev = np.concatenate([b.x[b.available] for b in dev])
    y_dev = np.concatenate([b.target[b.available] for b in dev])
    p_dev = np.concatenate([b.prevented[b.available] for b in dev])
    days = np.concatenate([
        np.full(int(b.available.sum()), b.day, dtype=object) for b in dev])
    q = np.asarray(model.predict_proba(x_dev)[:, 1], dtype=float)
    old_model = _old_harmful_model(old_artifact, coin)
    old_q = np.concatenate([
        np.asarray(old_model.predict(b.base.x[b.base_index[b.available]]),
                   dtype=float) for b in dev])
    cancel = q > 0.5
    old_cancel = old_q > 0.5
    constant_cancel = bool(float(y_train.mean()) > 0.0)
    realized = np.where(cancel, y_dev, 0.0)
    old_realized = np.where(old_cancel, y_dev, 0.0)
    constant_realized = y_dev if constant_cancel else np.zeros_like(y_dev)

    economic_dev = (p_dev > EPS) & (np.abs(y_dev) > EPS)
    y_class_train = (y_train[economic_train] > 0).astype(int)
    y_class_dev = (y_dev[economic_dev] > 0).astype(int)
    weights_train = np.abs(y_train[economic_train])
    weights_dev = np.abs(y_dev[economic_dev])
    if (len(y_class_dev) and np.unique(y_class_dev).size == 2
            and weights_dev.sum() > 0):
        fit = v5.weighted_classification_metrics(
            y_class_train, weights_train, y_class_dev, weights_dev,
            q[economic_dev])
        brier_positive = bool(
            fit["weighted_brier_skill_vs_training_weighted_prevalence"]
            is not None and
            fit["weighted_brier_skill_vs_training_weighted_prevalence"] > 0)
    else:
        fit = {"status": "UNAVAILABLE_TWO_CLASS_DEV"}
        brier_positive = False
    daily_value = _daily(days, realized)
    daily_constant_gain = _daily(days, realized - constant_realized)
    daily_old_gain = _daily(days, realized - old_realized)
    gates = {
        "positive_weighted_brier_skill": brier_positive,
        "positive_value_each_dev_day": all(
            daily_value[d] is not None and daily_value[d] > 0 for d in DEV_DAYS),
        "positive_gain_vs_constant_each_dev_day": all(
            daily_constant_gain[d] is not None and daily_constant_gain[d] > 0
            for d in DEV_DAYS),
        "positive_gain_vs_v5_each_dev_day": all(
            daily_old_gain[d] is not None and daily_old_gain[d] > 0
            for d in DEV_DAYS),
        "nondegenerate_cancel_fraction": bool(0.02 < cancel.mean() < 0.98),
    }
    report = {
        "model_gate": {"pass": all(gates.values()), "checks": gates},
        "fit": fit,
        "population": {
            "eligible_train_rows": len(x_train),
            "eligible_dev_rows": len(x_dev),
            "economic_train_rows": int(economic_train.sum()),
            "economic_dev_rows": int(economic_dev.sum()),
        },
        "policy": {
            "cancel_fraction": float(cancel.mean()),
            "old_v5_cancel_fraction_same_rows": float(old_cancel.mean()),
            "training_selected_constant": (
                "ALWAYS_CANCEL" if constant_cancel else "NEVER_CANCEL"),
            "gross_value_cents_per_eligible_decision": float(realized.mean()),
            "per_day_gross_value_cents_per_eligible_decision": daily_value,
            "per_day_gain_vs_training_constant": daily_constant_gain,
            "per_day_gain_vs_old_v5": daily_old_gain,
        },
        "receipt": _receipt(model, coin),
    }
    predictions: dict[str, np.ndarray] = {}
    for batch in batches:
        predictions[batch.slug] = np.asarray(
            model.predict_proba(batch.x)[:, 1], dtype=float)
    return report, predictions, {"model": model}


def _signals(batch: ActionBatch, q: np.ndarray
             ) -> dict[str, list[tuple[float, bool]]]:
    by_base = {int(index): bool(value > 0.5)
               for index, value in zip(batch.base_index, q)}
    start = int(batch.slug.rsplit("-", 1)[1])
    elapsed = batch.base.as_of_ns.astype(np.float64) / 1e9 - start
    result: dict[str, list[tuple[float, bool]]] = {
        "BUY_UP": [], "SELL_UP": []}
    for index, when in enumerate(elapsed):
        side = "BUY_UP" if batch.base.maker_side_sign[index] > 0 else "SELL_UP"
        result[side].append((float(when), by_base.get(index, False)))
    return result


def _mean_metric(cells: dict[str, Any], cell: str, coin: str,
                 days: Sequence[str], metric: str) -> float:
    values = [cells[cell][coin][day][metric] for day in days]
    return float(np.mean(values))


def run() -> dict[str, Any]:
    if not PARENT_ARTIFACT.exists() or not OLD_MODEL_ARTIFACT.exists():
        raise RuntimeError("frozen parent/model receipt missing")
    parent = json.loads(PARENT_ARTIFACT.read_text())
    old_artifact = json.loads(OLD_MODEL_ARTIFACT.read_text())
    base_batches, sampled, slugs = linear.build_batches(1)
    selected = {item[0]: item for _, items in ww.select_by_day(1).items()
                for item in items if item[0] in set(slugs)}
    action_batches: list[ActionBatch] = []
    for index, base in enumerate(base_batches, 1):
        _, path, up, down, gaps = selected[base.slug]
        tape = v1.build_pm_tape(path, up, down, gaps, feature_state_lag_s=0.0)
        action = trace_action_batch(base, tape)
        action_batches.append(action)
        print(f"[qact] trace {index}/{len(base_batches)} {base.slug} "
              f"eligible={len(action.x):,}", flush=True)

    model_reports: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for coin in linear.COINS:
        report, scored, _ = _fit_and_score(
            coin, [b for b in action_batches if b.coin == coin], old_artifact)
        model_reports[coin] = report
        predictions.update(scored)
        print(f"[qact] {coin} model gate "
              f"{report['model_gate']['pass']}", flush=True)

    windows: dict[tuple[str, str, str], list[Any]] = collections.defaultdict(list)
    controls = {
        "trace_deterministic": True,
        "qr_skew_disabled_path_parity": False,
        "all_false_candidate_parity": True,
        "model_receipt_roundtrip": True,
        "feature_asof_future_exclusion": bool(
            fast.SOURCE_PROFILE["future_events_excluded"]
            and all(np.isfinite(batch.x).all() for batch in action_batches)),
    }
    for index, batch in enumerate(action_batches, 1):
        slug, path, up, down, gaps = selected[batch.slug]
        latency = CELLS[batch.coin]["latency_ms"]
        specs = [qr._qr_spec(qr.QR_SKEW, latency, False),
                 qr._qr_spec(CANDIDATE, latency, True)]
        got = qr.replay_cells_queue_realistic(
            path, up, down, gaps, specs,
            _signals(batch, predictions[slug]), lag_s=0.0)
        if got is None:
            raise RuntimeError(f"candidate replay unavailable {slug}")
        for cell, window in got.items():
            windows[(cell, batch.coin, batch.day)].append(window)
        print(f"[qact] replay {index}/{len(action_batches)} {slug}", flush=True)

    # Real-data lifecycle controls on the first window.
    first = action_batches[0]
    slug, path, up, down, gaps = selected[first.slug]
    latency = CELLS[first.coin]["latency_ms"]
    spec = [qr._qr_spec(qr.QR_SKEW, latency, False),
            qr._qr_spec(CANDIDATE, latency, True)]
    false = {side: [(t, False) for t, _ in rows]
             for side, rows in _signals(first, predictions[slug]).items()}
    false_got = qr.replay_cells_queue_realistic(
        path, up, down, gaps, spec, false, lag_s=0.0)
    controls["all_false_candidate_parity"] = bool(
        false_got is not None and pb.conformant(
            false_got[CANDIDATE], false_got[qr.QR_SKEW]))
    again_tape = v1.build_pm_tape(path, up, down, gaps, feature_state_lag_s=0.0)
    again = trace_action_batch(first.base, again_tape)
    controls["trace_deterministic"] = bool(
        np.array_equal(first.base_index, again.base_index)
        and np.array_equal(first.generation, again.generation)
        and np.allclose(first.x, again.x)
        and np.allclose(first.target, again.target, equal_nan=True))
    for coin, report in model_reports.items():
        receipt = report["receipt"]
        raw = zlib.decompress(base64.b64decode(receipt["model_text_zlib_b64"]))
        controls["model_receipt_roundtrip"] &= bool(
            hashlib.sha256(raw).hexdigest() == receipt["model_text_sha256"])

    days = sorted({b.day for b in action_batches})
    shadow_skew_cells = {coin: {day: qr._cell_metrics(
        windows.get((qr.QR_SKEW, coin, day), []), qr.QR_SKEW)
        for day in days} for coin in linear.COINS}
    controls["qr_skew_disabled_path_parity"] = all(
        shadow_skew_cells[coin][day] == parent["cells"][qr.QR_SKEW][coin][day]
        for coin in linear.COINS for day in days)
    cells = {
        qr.QR_BASELINE: parent["cells"][qr.QR_BASELINE],
        qr.QR_SKEW: parent["cells"][qr.QR_SKEW],
        CANDIDATE: {coin: {day: qr._cell_metrics(
            windows.get((CANDIDATE, coin, day), []), CANDIDATE)
            for day in days} for coin in linear.COINS},
    }
    adoption: dict[str, Any] = {}
    for coin in linear.COINS:
        per_day = {day: (cells[CANDIDATE][coin][day]["pnl_per_window_cents"]
                         - cells[qr.QR_BASELINE][coin][day]["pnl_per_window_cents"])
                   for day in days}
        checks = {
            "model_gate": model_reports[coin]["model_gate"]["pass"],
            "positive_incumbent_delta_each_dev_day": all(
                per_day[day] > 0 for day in DEV_DAYS),
            "dev_mean_above_qr_skew": _mean_metric(
                cells, CANDIDATE, coin, DEV_DAYS, "pnl_per_window_cents")
                > _mean_metric(cells, qr.QR_SKEW, coin, DEV_DAYS,
                               "pnl_per_window_cents"),
            "inventory_not_increased": _mean_metric(
                cells, CANDIDATE, coin, DEV_DAYS,
                "terminal_abs_net_mean_shares") <= _mean_metric(
                    cells, qr.QR_BASELINE, coin, DEV_DAYS,
                    "terminal_abs_net_mean_shares") + 1e-12,
            "effective_cancels_not_increased": sum(
                cells[CANDIDATE][coin][d]["cancel_effective"] for d in DEV_DAYS)
                <= sum(cells[qr.QR_BASELINE][coin][d]["cancel_effective"]
                       for d in DEV_DAYS),
            "cancel_repost_traffic_not_increased": sum(
                cells[CANDIDATE][coin][d]["cancel_effective"]
                + cells[CANDIDATE][coin][d]["cancel_reposts"] for d in DEV_DAYS)
                <= sum(cells[qr.QR_BASELINE][coin][d]["cancel_effective"]
                       + cells[qr.QR_BASELINE][coin][d]["cancel_reposts"]
                       for d in DEV_DAYS),
            "controls": all(controls.values()),
        }
        adoption[coin] = {
            "verdict": "ADOPT_DIAGNOSTIC" if all(checks.values()) else "REJECT",
            "checks": checks,
            "per_day_delta_pnl_cents_vs_incumbent": per_day,
        }
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "QUEUE_ACTION_CONDITIONED_HARMFUL_ITERATION_005",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT_DIAGNOSTIC",
        "verdict": {coin: adoption[coin]["verdict"] for coin in linear.COINS},
        "decision_eligible": False,
        "promotion_authorized": False,
        "candidate": CANDIDATE,
        "baseline": qr.QR_BASELINE,
        "required_comparator": qr.QR_SKEW,
        "frozen_cells": CELLS,
        "model": model_reports,
        "controls": controls,
        "cells": cells,
        "adoption": adoption,
        "population": {
            "training_days": list(TRAIN_DAYS),
            "development_days": list(DEV_DAYS),
            "selected_slugs": slugs,
            "rows": {b.slug: {"base": b.base.n_rows,
                               "eligible": len(b.x),
                               "available": int(b.available.sum()),
                               "diagnostics": b.diagnostics}
                     for b in action_batches},
        },
        "semantics": {
            "behavior_path": qr.QR_SKEW,
            "action_population": "LIVE_JOIN_EXISTING_INVENTORY_INCREASING",
            "label": "SAME_GENERATION_GROSS_PREVENTED_5S_MARKOUT",
            "incentives": "EXCLUDED",
            "threshold": 0.5,
            "decision_cooldown_ms": fast.COOLDOWN_MS,
            "latency": "ASSUMED_CANCEL_EFFECTIVE_NOT_MEASURED",
            "forward_days_observed": 0,
        },
        "provenance": {
            "code_sha256": _file_sha(Path(__file__)),
            "protocol_sha256": _file_sha(PROTOCOL),
            "parent_artifact_sha256": _file_sha(PARENT_ARTIFACT),
            "old_model_artifact_sha256": _file_sha(OLD_MODEL_ARTIFACT),
            "base_feature_builder_sha256": _file_sha(Path(fast.__file__)),
            "queue_engine_sha256": _file_sha(Path(qr.__file__)),
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
    print(f"[qact] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0
    def ok(value: bool, name: str) -> None:
        nonlocal checks
        if not value:
            raise AssertionError(name)
        checks += 1
    ok(len(FEATURE_NAMES) == len(fast.FEATURE_NAMES) + 6,
       "six action-state features appended")
    ok(len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "feature names unique")
    ok(CELLS == {"btc": {"horizon_ms": 50, "latency_ms": 25},
                 "eth": {"horizon_ms": 250, "latency_ms": 100}},
       "frozen H/L cells unchanged")
    arm = TraceArm(qr._qr_spec(qr.QR_SKEW, 25, False))
    arm.clock = 1.0
    arm.sync_from_quote((.49, .51, 10.0, 12.0, .01))
    ok(arm.buy.qahead == 10.0 and arm.placement_kind["BUY_UP"] == qr.JOIN_EXISTING,
       "occupied touch joins behind displayed queue")
    generation = arm.generation["BUY_UP"]
    arm.clock = 1.01
    got = arm.consume("BUY_UP", 12.0, 10.0)
    ok(got == 2.0 and arm.generation["BUY_UP"] == generation,
       "partial fill retains generation")
    arm.clock = 1.02
    arm.consume("BUY_UP", 3.0, 10.0)
    ok(arm.generation["BUY_UP"] == generation + 1,
       "full fill repost starts new generation")
    ok(PROTOCOL.exists() and PARENT_ARTIFACT.exists(),
       "frozen protocol and parent receipt exist")
    print(f"[qact] selftest OK — {checks} checks")
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
