"""Offline stateful composition of harmful-flow cancellation and SKEW_LB.

Research only.  This consumes the already fitted v5 model artifact and replays
five shadow arms on recorded PM/HF data.  Cancellation is restricted to actual
JOIN placements because the v5 action schema does not cover FRONT orders.
There is no live venue, order, cancel, or execution port.

Commands::

    python3 live/pm_research/policy_optimizer_cancel_skew.py --selftest
    python3 live/pm_research/policy_optimizer_cancel_skew.py run
"""

from __future__ import annotations

import argparse
import base64
import collections
import datetime as dt
import hashlib
import heapq
import json
import math
import zlib
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np

import adverse_move_harmful as v5
import edge_layer1 as el
import flow_fill_development as fd
import flow_intensity as fi
import placement_skew as ps
import policy_bounds_v1 as pb
import policy_optimizer as opt
import warning_window as ww

OUT = fi.PM / "derived/policy_optimizer_cancel_skew_harmful_v1.json"
MODEL_ARTIFACT = fi.PM / "derived/adverse_move_harmful_development_v5.json"
PROTOCOL = Path(__file__).with_name("CANCEL_SKEW_HARMFUL_PROTOCOL.md")

H = 5.0
SIGNAL_EPSILON_S = 1e-9
STATE_LAG_S = 0.0
CANDIDATE_CELLS = {
    "btc": {"horizon_ms": 50, "latency_ms": 25},
    "eth": {"horizon_ms": 250, "latency_ms": 100},
}
CELL_NAMES = (
    "JOIN_ONLY",
    "FRONT_ONLY",
    "SKEW_ONLY",
    "CANCEL_ONLY",
    "CANCEL_X_SKEW",
)


def _sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _specs(latency_ms: int) -> list[dict[str, Any]]:
    common = {"r_cut": 0, "size": 5.0}
    skew = {
        "placement": "SKEW_LB",
        "skew": True,
        "skew_band_shares": ps.SKEW_BAND_SHARES,
        "front_on_repost": False,
    }
    return [
        {"cell": "JOIN_ONLY", "placement": "JOIN", **common},
        {"cell": "FRONT_ONLY", "placement": "FRONT", **common},
        {"cell": "SKEW_ONLY", **skew, **common},
        {"cell": "CANCEL_ONLY", "placement": "JOIN", "cancel": True,
         "cancel_latency_ms": latency_ms, "cancel_join_only": True, **common},
        {"cell": "CANCEL_X_SKEW", **skew, "cancel": True,
         "cancel_latency_ms": latency_ms, "cancel_join_only": True, **common},
    ]


class CancelArm(opt.SimArm):
    """SimArm with generation-bound cancel/rejoin lifecycle state."""

    SIDES = ("BUY_UP", "SELL_UP")

    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        self.cancel_enabled = bool(spec.get("cancel", False))
        self.cancel_latency_s = float(spec.get("cancel_latency_ms", 0)) / 1000.0
        self.cancel_join_only = bool(spec.get("cancel_join_only", True))
        self.generation = {side: 0 for side in self.SIDES}
        self.placement_front = {side: False for side in self.SIDES}
        self.filled_current_order = {side: 0.0 for side in self.SIDES}
        self.armed = {side: True for side in self.SIDES}
        self.pending_generation: dict[str, int | None] = {
            side: None for side in self.SIDES}
        self.cancel_counts: collections.Counter[str] = collections.Counter()

    def side(self, maker_side: str) -> Any:
        return self.buy if maker_side == "BUY_UP" else self.sell

    def reposition(self, maker_side: str, level: float | None,
                   displayed: float) -> None:
        side = self.side(maker_side)
        side.reposition(level, displayed)
        self.generation[maker_side] += 1
        self.placement_front[maker_side] = bool(
            level is not None and side.front)
        self.filled_current_order[maker_side] = 0.0

    def consume(self, maker_side: str, volume: float,
                displayed: float) -> float:
        side = self.side(maker_side)
        before = side.resting
        filled = side.consume(volume, displayed)
        if filled <= 0:
            return 0.0
        self.filled_current_order[maker_side] += filled
        if before <= filled + 1e-12:
            self.generation[maker_side] += 1
            grant = bool(
                side.front and getattr(side, "front_on_repost", True))
            self.placement_front[maker_side] = grant
            self.filled_current_order[maker_side] = 0.0
        return filled

    def observe_signal(
            self, maker_side: str, harmful: bool, when: float,
            arm_index: int,
            pending: list[tuple[float, int, int, str, int]],
            sequence: int) -> int:
        if not self.cancel_enabled:
            return sequence
        self.cancel_counts["signal_evaluations"] += 1
        if not harmful:
            self.armed[maker_side] = True
            return sequence
        self.cancel_counts["harmful_signal_evaluations"] += 1
        side = self.side(maker_side)
        supported = (
            side.level is not None
            and (not self.cancel_join_only
                 or not self.placement_front[maker_side]))
        if not supported:
            self.cancel_counts["front_or_absent_signal_skips"] += 1
            return sequence
        if (not self.armed[maker_side]
                or self.pending_generation[maker_side] is not None):
            self.cancel_counts["persistent_signal_skips"] += 1
            return sequence
        generation = self.generation[maker_side]
        effective = when + self.cancel_latency_s
        sequence += 1
        heapq.heappush(
            pending, (effective, sequence, arm_index, maker_side, generation))
        self.pending_generation[maker_side] = generation
        self.armed[maker_side] = False
        self.cancel_counts["submitted"] += 1
        self.actions += 1
        return sequence

    def apply_cancel(self, maker_side: str, generation: int,
                     quote: tuple[float, float, float, float, float] | None
                     ) -> None:
        if self.pending_generation[maker_side] == generation:
            self.pending_generation[maker_side] = None
        side = self.side(maker_side)
        if (generation != self.generation[maker_side]
                or side.level is None or quote is None
                or (self.cancel_join_only
                    and self.placement_front[maker_side])):
            self.cancel_counts["stale_or_unsupported_effective"] += 1
            return
        bid, ask, bid_size, ask_size, _ = quote
        level = bid if maker_side == "BUY_UP" else ask
        displayed = bid_size if maker_side == "BUY_UP" else ask_size
        if self.filled_current_order[maker_side] > 1e-12:
            self.cancel_counts["partial_fill_then_cancel"] += 1
        # A cancellation repost always joins behind displayed depth, even if
        # the skew intent would front on the next genuine level formation.
        side.level = level
        side.resting = side.size
        side.qahead = max(0.0, displayed)
        self.generation[maker_side] += 1
        self.placement_front[maker_side] = False
        self.filled_current_order[maker_side] = 0.0
        self.cancel_counts["effective"] += 1
        self.cancel_counts["reposts"] += 1
        self.actions += 1


def replay_cells_with_cancel(
        path: Path, up_id: str, down_id: str,
        gaps: Sequence[tuple[float, float]],
        specs: Sequence[dict[str, Any]],
        signals: dict[str, Sequence[tuple[float, bool]]],
        lag_s: float = STATE_LAG_S) -> dict[str, el.WindowFills] | None:
    """Replay stateful arms with generation-bound cancellation schedules."""
    slug = path.name.split(".jsonl")[0]
    try:
        window_start = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None

    state = fd.BookState()
    arms = [CancelArm(spec) for spec in specs]
    diagnostics: collections.Counter[str] = collections.Counter()
    seen_transactions: set[str] = set()
    mid_t: list[float] = []
    mid_v: list[float] = []
    bad_intervals = [
        (start, end) for start, end in gaps
        if end >= 0.0 and start <= fi.WINDOW_S
    ]
    state_pending: list[tuple[float, int, str, dict[str, Any]]] = []
    cancel_pending: list[tuple[float, int, int, str, int]] = []
    state_sequence = 0
    cancel_sequence = 0
    gap_starts = sorted(
        start for start, _ in gaps if 0.0 <= start <= fi.WINDOW_S)
    gap_index = 0
    signal_sequence = 0
    signal_pending: list[tuple[float, int, str, bool]] = []
    for maker_side in CancelArm.SIDES:
        for when, harmful in signals.get(maker_side, ()):
            signal_sequence += 1
            heapq.heappush(signal_pending, (
                when + SIGNAL_EPSILON_S, signal_sequence,
                maker_side, bool(harmful)))

    def record_mid(when: float) -> None:
        quote = state.quote()
        if quote is None:
            return
        mid = (quote[0] + quote[1]) / 2.0
        if mid_v and abs(mid_v[-1] - mid) < 1e-12:
            return
        if mid_t and when <= mid_t[-1]:
            mid_v[-1] = mid
            return
        mid_t.append(when)
        mid_v.append(mid)

    def resync(when: float) -> None:
        quote = state.quote()
        for arm in arms:
            if arm.dead(when) or quote is None:
                if arm.buy.level is not None:
                    arm.reposition("BUY_UP", None, 0.0)
                if arm.sell.level is not None:
                    arm.reposition("SELL_UP", None, 0.0)
                continue
            bid, ask, bid_size, ask_size, _ = quote
            arm.apply_skew_intent()
            if arm.buy.level is None or abs(arm.buy.level - bid) > 1e-12:
                arm.reposition("BUY_UP", bid, bid_size)
                arm.actions += 1
            if arm.sell.level is None or abs(arm.sell.level - ask) > 1e-12:
                arm.reposition("SELL_UP", ask, ask_size)
                arm.actions += 1
        record_mid(when)

    def advance(to: float) -> None:
        nonlocal gap_index, cancel_sequence
        while True:
            candidates: list[float] = []
            if state_pending:
                candidates.append(state_pending[0][0])
            if signal_pending:
                candidates.append(signal_pending[0][0])
            if cancel_pending:
                candidates.append(cancel_pending[0][0])
            if gap_index < len(gap_starts):
                candidates.append(gap_starts[gap_index])
            if not candidates or min(candidates) > to + 1e-12:
                break
            when = min(candidates)
            if (gap_index < len(gap_starts)
                    and abs(gap_starts[gap_index] - when) < 1e-12):
                state.clear()
                state_pending.clear()
                for arm in arms:
                    arm.reposition("BUY_UP", None, 0.0)
                    arm.reposition("SELL_UP", None, 0.0)
                diagnostics["gap_state_resets"] += 1
                gap_index += 1
                while (gap_index < len(gap_starts)
                       and abs(gap_starts[gap_index] - when) < 1e-12):
                    gap_index += 1
            while state_pending and state_pending[0][0] <= when + 1e-12:
                _, _, kind, data = heapq.heappop(state_pending)
                state.apply(kind, data)
            resync(when)
            while signal_pending and signal_pending[0][0] <= when + 1e-12:
                _, _, maker_side, harmful = heapq.heappop(signal_pending)
                for arm_index, arm in enumerate(arms):
                    cancel_sequence = arm.observe_signal(
                        maker_side, harmful, when, arm_index,
                        cancel_pending, cancel_sequence)
            while cancel_pending and cancel_pending[0][0] <= when + 1e-12:
                _, _, arm_index, maker_side, generation = heapq.heappop(
                    cancel_pending)
                arms[arm_index].apply_cancel(
                    maker_side, generation, state.quote())
            record_mid(when)

    def schedule_state(received: float, kind: str,
                       data: dict[str, Any]) -> None:
        nonlocal state_sequence
        state_sequence += 1
        heapq.heappush(
            state_pending,
            (received + lag_s, state_sequence, kind, data))

    for line in fi._gz_lines(path):
        if not any(marker in line for marker in (
                fi.TRADE_MARK, fi.QUOTE_MARK, fd.BOOK_MARK, fd.TICK_MARK)):
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            received = int(parts[0]) / 1e9 - window_start
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            diagnostics["malformed"] += 1
            continue
        if received < -60.0 or received > fi.WINDOW_S:
            continue
        advance(received)

        for message in payload if isinstance(payload, list) else [payload]:
            if not isinstance(message, dict):
                continue
            event_type = message.get("event_type")
            asset_id = str(message.get("asset_id"))
            if (event_type == "book"
                    or ("bids" in message and "asks" in message)) \
                    and asset_id == up_id:
                data = fd._parse_book(message)
                if data:
                    schedule_state(received, "book", data)
                continue
            if event_type == "price_change":
                for change in message.get("price_changes", []):
                    if str(change.get("asset_id")) != up_id:
                        continue
                    try:
                        data = {
                            "side": str(change["side"]).upper(),
                            "price": float(change["price"]),
                            "size": float(change["size"]),
                            "best_bid": float(change["best_bid"]),
                            "best_ask": float(change["best_ask"]),
                        }
                    except (KeyError, TypeError, ValueError):
                        continue
                    if 0.0 <= data["best_bid"] < data["best_ask"] <= 1.0:
                        schedule_state(received, "price", data)
                continue
            if event_type == "tick_size_change" and asset_id == up_id:
                bad_intervals.append((
                    max(0.0, received - 1e-9),
                    received + max(el.HORIZONS)))
                diagnostics["tick_changes"] += 1
                try:
                    schedule_state(
                        received, "tick",
                        {"tick": float(message["new_tick_size"])})
                except (KeyError, TypeError, ValueError):
                    pass
                continue
            if (event_type != "last_trade_price"
                    or asset_id not in (up_id, down_id)):
                continue
            transaction = str(message.get("transaction_hash") or "")
            if transaction and transaction in seen_transactions:
                diagnostics["duplicate_transaction"] += 1
                continue
            if transaction:
                seen_transactions.add(transaction)
            try:
                native_price = float(message["price"])
                size = float(message["size"])
                native_side = str(message["side"]).upper()
            except (KeyError, TypeError, ValueError):
                continue
            is_down = asset_id == down_id
            execution_price = fi.fold_price(native_price, is_down)
            taker_side = fi.fold_side(native_side, is_down)
            quote = state.quote()
            if quote is None:
                diagnostics["trades_no_state"] += 1
                continue
            bid, ask, bid_size, ask_size, _ = quote
            mid_now = (bid + ask) / 2.0
            record_mid(received)
            micro = abs(size - fi.MICRO_SIZE) < 1e-9
            for arm in arms:
                if (taker_side == "BUY" and arm.sell.level is not None
                        and execution_price + 1e-12 >= arm.sell.level):
                    level = arm.sell.level
                    filled = arm.consume("SELL_UP", size, ask_size)
                    if filled > 0:
                        arm.fills.append(el.Fill(
                            received, "SELL_UP", level, filled, mid_now, micro))
                        arm.led_q_dn += filled
                        arm.apply_skew_intent()
                elif (taker_side == "SELL" and arm.buy.level is not None
                      and execution_price <= arm.buy.level + 1e-12):
                    level = arm.buy.level
                    filled = arm.consume("BUY_UP", size, bid_size)
                    if filled > 0:
                        arm.fills.append(el.Fill(
                            received, "BUY_UP", level, filled, mid_now, micro))
                        arm.led_q_up += filled
                        arm.apply_skew_intent()
        # At zero lag, apply the current PM state before a same-time feature
        # signal.  The one-nanosecond signal offset preserves event causality.
        advance(received + SIGNAL_EPSILON_S)

    advance(fi.WINDOW_S)
    if not mid_t:
        return None
    output: dict[str, el.WindowFills] = {}
    coin = slug.split("-")[0]
    for arm in arms:
        bought = sum(
            fill.size for fill in arm.fills if fill.maker_side == "BUY_UP")
        sold = sum(
            fill.size for fill in arm.fills if fill.maker_side == "SELL_UP")
        if (abs(bought - arm.led_q_up) > 1e-9
                or abs(sold - arm.led_q_dn) > 1e-9):
            raise SystemExit(
                f"[cancel-skew] reconciliation break {slug} "
                f"{arm.spec['cell']}")
        item_diagnostics = dict(diagnostics)
        cell = arm.spec["cell"]
        item_diagnostics[f"actions:{cell}"] = arm.actions
        item_diagnostics[f"skew_intent_flips:{cell}"] = arm.skew_intent_flips
        for name, value in arm.cancel_counts.items():
            item_diagnostics[f"cancel_{name}:{cell}"] = value
        output[cell] = el.WindowFills(
            slug, coin, arm.fills, mid_t, mid_v,
            list(bad_intervals), item_diagnostics)
    return output


def _load_booster(receipt: dict[str, Any]) -> lgb.Booster:
    raw = zlib.decompress(base64.b64decode(receipt["model_text_zlib_b64"]))
    if hashlib.sha256(raw).hexdigest() != receipt["model_text_sha256"]:
        raise RuntimeError("model receipt hash mismatch")
    return lgb.Booster(model_str=raw.decode())


def _signals_for_batch(batch: Any, artifact: dict[str, Any]
                       ) -> tuple[dict[str, list[tuple[float, bool]]],
                                  dict[str, Any]]:
    candidate = CANDIDATE_CELLS[batch.coin]
    horizon = str(candidate["horizon_ms"])
    latency = str(candidate["latency_ms"])
    receipts = artifact["coins"][batch.coin]["model_artifact"][
        "models_by_horizon_ms"][horizon]["latencies"][latency]
    fill_model = _load_booster(receipts["preventable_fill"])
    harmful_model = _load_booster(receipts["value_weighted_harmful_fill"])
    fill_probability = np.asarray(fill_model.predict(batch.x), dtype=float)
    harmful_probability = np.asarray(harmful_model.predict(batch.x), dtype=float)
    start = int(batch.slug.rsplit("-", 1)[1])
    elapsed = batch.as_of_ns.astype(np.float64) / 1e9 - start
    result: dict[str, list[tuple[float, bool]]] = {
        "BUY_UP": [], "SELL_UP": []}
    for index, when in enumerate(elapsed):
        side = "BUY_UP" if batch.maker_side_sign[index] > 0 else "SELL_UP"
        result[side].append((
            float(when), bool(harmful_probability[index] > 0.5)))
    transitions = {}
    for side, rows in result.items():
        values = [harmful for _, harmful in rows]
        transitions[side] = sum(
            int(current and not previous)
            for previous, current in zip([False] + values[:-1], values))
    return result, {
        "n_rows": len(batch.x),
        "harmful_fraction": float((harmful_probability > 0.5).mean()),
        "harmful_probability_mean": float(harmful_probability.mean()),
        "preventable_fill_probability_mean": float(fill_probability.mean()),
        "false_to_true_transitions": transitions,
    }


def _fill_net(window: el.WindowFills) -> float:
    bought = sum(
        fill.size for fill in window.fills if fill.maker_side == "BUY_UP")
    sold = sum(
        fill.size for fill in window.fills if fill.maker_side == "SELL_UP")
    return bought - sold


def _cash_at_risk(window: el.WindowFills) -> float:
    net = _fill_net(window)
    mid = window.mid_v[-1]
    return net * mid if net > 0 else -net * (1.0 - mid)


def _diagnostic_total(windows: Sequence[el.WindowFills],
                      cell: str, name: str) -> int:
    key = f"cancel_{name}:{cell}"
    return sum(int(window.diagnostics.get(key, 0)) for window in windows)


def _cell_metrics(windows: Sequence[el.WindowFills],
                  cell: str) -> dict[str, Any]:
    rows_by_window = [pb.rows_h(window, H)[0] for window in windows]
    rows = [row for group in rows_by_window for row in group]
    n_windows = len(windows)
    shares = sum(row.size for row in rows)
    spread = sum(row.spread * row.size for row in rows)
    drift = sum(row.drift * row.size for row in rows)
    nets = np.asarray([abs(_fill_net(window)) for window in windows], dtype=float)
    cash = np.asarray([_cash_at_risk(window) for window in windows], dtype=float)
    return {
        "n_windows": n_windows,
        "n_fills": sum(len(window.fills) for window in windows),
        "admitted_markout_rows": len(rows),
        "shares_per_window": shares / n_windows if n_windows else None,
        "pnl_per_window_cents": opt.total_pnl_per_window(rows_by_window),
        "spread_capture_per_window_cents":
            spread * 100.0 / n_windows if n_windows else None,
        "post_fill_drift_per_window_cents":
            drift * 100.0 / n_windows if n_windows else None,
        "swm_cents": pb.swm(rows),
        "terminal_abs_net_mean_shares":
            float(nets.mean()) if len(nets) else None,
        "terminal_cash_at_risk_mean_usd":
            float(cash.mean()) if len(cash) else None,
        "cancel_submitted": _diagnostic_total(windows, cell, "submitted"),
        "cancel_effective": _diagnostic_total(windows, cell, "effective"),
        "cancel_reposts": _diagnostic_total(windows, cell, "reposts"),
        "cancel_partial_fill_then_cancel": _diagnostic_total(
            windows, cell, "partial_fill_then_cancel"),
        "cancel_stale_or_unsupported_effective": _diagnostic_total(
            windows, cell, "stale_or_unsupported_effective"),
        "cancel_front_or_absent_signal_skips": _diagnostic_total(
            windows, cell, "front_or_absent_signal_skips"),
    }


def _controls(item: Sequence[Any], signals: dict[str, list[tuple[float, bool]]],
              latency_ms: int) -> dict[str, Any]:
    _, path, up_id, down_id, gaps = item
    baseline_specs = _specs(latency_ms)[:3]
    reference = opt.replay_cells(
        path, up_id, down_id, gaps, baseline_specs, lag_s=STATE_LAG_S)
    candidate = replay_cells_with_cancel(
        path, up_id, down_id, gaps, baseline_specs, {}, lag_s=STATE_LAG_S)
    if reference is None or candidate is None:
        raise RuntimeError("baseline parity control unavailable")
    parity = all(pb.conformant(candidate[cell], reference[cell])
                 for cell in ("JOIN_ONLY", "FRONT_ONLY", "SKEW_ONLY"))
    if not parity:
        raise RuntimeError("cancel engine disabled-path parity failure")

    false_signals = {
        side: [(when, False) for when, _ in rows]
        for side, rows in signals.items()
    }
    false_result = replay_cells_with_cancel(
        path, up_id, down_id, gaps, _specs(latency_ms),
        false_signals, lag_s=STATE_LAG_S)
    if false_result is None:
        raise RuntimeError("all-false control unavailable")
    all_false = (
        pb.conformant(false_result["CANCEL_ONLY"], false_result["JOIN_ONLY"])
        and pb.conformant(
            false_result["CANCEL_X_SKEW"], false_result["SKEW_ONLY"]))
    if not all_false:
        raise RuntimeError("all-false cancellation control failure")

    first = replay_cells_with_cancel(
        path, up_id, down_id, gaps, _specs(latency_ms),
        signals, lag_s=STATE_LAG_S)
    second = replay_cells_with_cancel(
        path, up_id, down_id, gaps, _specs(latency_ms),
        signals, lag_s=STATE_LAG_S)
    deterministic = bool(first is not None and second is not None and all(
        pb.fill_key(first[cell]) == pb.fill_key(second[cell])
        and first[cell].diagnostics == second[cell].diagnostics
        for cell in CELL_NAMES))
    if not deterministic:
        raise RuntimeError("cancel×skew determinism control failure")
    return {
        "disabled_path_parity": True,
        "all_false_signal_parity": True,
        "deterministic": True,
        "action_schema": "JOIN_SCHEMA_ONLY",
    }


def run() -> dict[str, Any]:
    if not MODEL_ARTIFACT.exists():
        raise RuntimeError("v5 harmful-flow model artifact is missing")
    artifact = json.loads(MODEL_ARTIFACT.read_text())
    batches, sampled, slugs = v5.linear.build_batches(1)
    selected = {
        item[0]: item
        for _, items in ww.select_by_day(1).items()
        for item in items
        if item[0] in set(slugs)
    }
    schedules: dict[str, dict[str, list[tuple[float, bool]]]] = {}
    schedule_audit: dict[str, Any] = {}
    for batch in batches:
        signals, audit = _signals_for_batch(batch, artifact)
        schedules[batch.slug] = signals
        schedule_audit[batch.slug] = audit

    first_batch = batches[0]
    first_item = selected[first_batch.slug]
    controls = _controls(
        first_item, schedules[first_batch.slug],
        CANDIDATE_CELLS[first_batch.coin]["latency_ms"])
    print(f"[cancel-skew] controls PASS {controls}", flush=True)

    windows: dict[tuple[str, str, str], list[el.WindowFills]] = (
        collections.defaultdict(list))
    for index, batch in enumerate(batches, 1):
        slug, path, up_id, down_id, gaps = selected[batch.slug]
        coin, day = batch.coin, batch.day
        got = replay_cells_with_cancel(
            path, up_id, down_id, gaps,
            _specs(CANDIDATE_CELLS[coin]["latency_ms"]),
            schedules[slug], lag_s=STATE_LAG_S)
        if got is None:
            continue
        for cell, window in got.items():
            windows[(cell, coin, day)].append(window)
        print(f"[cancel-skew] {index}/{len(batches)} {slug}", flush=True)

    days = sorted({batch.day for batch in batches})
    cells: dict[str, Any] = {}
    for cell in CELL_NAMES:
        cells[cell] = {}
        for coin in opt.VERDICT_COINS:
            cells[cell][coin] = {}
            for day in days:
                cells[cell][coin][day] = _cell_metrics(
                    windows.get((cell, coin, day), []), cell)

    comparisons: dict[str, Any] = {}
    for coin in opt.VERDICT_COINS:
        comparisons[coin] = {}
        for name, candidate_cell, baseline_cell in (
                ("cancel_only_vs_join", "CANCEL_ONLY", "JOIN_ONLY"),
                ("cancel_x_skew_vs_skew", "CANCEL_X_SKEW", "SKEW_ONLY"),
                ("cancel_x_skew_vs_cancel_only",
                 "CANCEL_X_SKEW", "CANCEL_ONLY")):
            per_day: dict[str, float | None] = {}
            for day in days:
                candidate_value = cells[candidate_cell][coin][day][
                    "pnl_per_window_cents"]
                baseline_value = cells[baseline_cell][coin][day][
                    "pnl_per_window_cents"]
                per_day[day] = (
                    None if candidate_value is None or baseline_value is None
                    else float(candidate_value - baseline_value))
            comparisons[coin][name] = {
                "per_day_delta_pnl_cents": per_day,
                "positive_all_days": bool(
                    per_day and all(value is not None and value > 0
                                    for value in per_day.values())),
                "positive_development_days": bool(all(
                    per_day.get(day) is not None and per_day[day] > 0
                    for day in v5.HOLDOUT_DAYS)),
            }

    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "HARMFUL_FLOW_CANCEL_X_SKEW_OFFLINE_V1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT_DIAGNOSTIC",
        "verdict": "NON_PROMOTABLE_MODEL_GATE_FAILED",
        "decision_eligible": False,
        "candidate_frozen": False,
        "promotion_authorized": False,
        "reasons": [
            "harmful_flow_v5_failed_its_predeclared_model_gate",
            "same_visible_three_train_two_development_days",
            "one_window_per_coin_day",
            "latency_is_assumed_not_measured_cancel_effective",
            "cancel_rejoin_cost_and_live_ack_are_unavailable",
            "v5_action_schema_only_supports_joined_orders",
        ],
        "candidate_cells": CANDIDATE_CELLS,
        "population": {
            "n_windows": len(batches),
            "days": days,
            "training_days": list(v5.TRAIN_DAYS),
            "development_days": list(v5.HOLDOUT_DAYS),
            "selected_slugs": slugs,
            "rows_by_slug": {batch.slug: batch.n_rows for batch in batches},
        },
        "semantics": {
            "state_lag_ms": 0,
            "decision_cooldown_ms": v5.fast.COOLDOWN_MS,
            "harmful_threshold": 0.5,
            "signal_rearm": "FALSE_THEN_TRUE_EDGE",
            "action_schema": "CANCEL_JOINED_ORDER_ONLY_FRONT_UNTOUCHED",
            "partial_fills_before_effective": "RETAINED",
            "cancel_binding": "ORDER_GENERATION",
            "repost": "BACK_OF_CURRENT_DISPLAYED_TOUCH",
            "incentives": "EXCLUDED_BY_USER_DIRECTION",
            "cancel_rejoin_cost": "UNAVAILABLE",
        },
        "controls": controls,
        "signal_audit": schedule_audit,
        "cells": cells,
        "comparisons": comparisons,
        "provenance": {
            "polymarket": fi.provenance(sampled=sampled),
            "v5_model_artifact_sha256": _file_sha(MODEL_ARTIFACT),
            "code_sha256": _file_sha(Path(__file__)),
            "protocol_sha256": _file_sha(PROTOCOL),
            "policy_engine_sha256": _file_sha(Path(opt.__file__)),
            "feature_builder_sha256": _file_sha(Path(v5.fast.__file__)),
            "hf_source_identity": {
                "kind": "PATH_SIZE_MTIME_RECEIPT_NOT_CONTENT_DIGEST",
                "files": v5.linear._hf_manifest(slugs),
            },
        },
    }
    result["artifact_id"] = _sha(result)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, allow_nan=False))
    print(f"[cancel-skew] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0

    def ok(condition: bool, name: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(name)
        checks += 1

    spec = {
        "cell": "TEST", "placement": "JOIN", "r_cut": 0, "size": 5.0,
        "cancel": True, "cancel_latency_ms": 25, "cancel_join_only": True,
    }
    arm = CancelArm(spec)
    arm.reposition("BUY_UP", 0.49, 10.0)
    ok(not arm.placement_front["BUY_UP"], "joined placement is cancellable")
    pending: list[tuple[float, int, int, str, int]] = []
    sequence = arm.observe_signal(
        "BUY_UP", True, 1.0, 0, pending, 0)
    ok(len(pending) == 1 and abs(pending[0][0] - 1.025) < 1e-12,
       "cancel becomes effective after assumed latency")
    sequence2 = arm.observe_signal(
        "BUY_UP", True, 1.005, 0, pending, sequence)
    ok(sequence2 == sequence and arm.cancel_counts["submitted"] == 1,
       "persistent true signal does not repeatedly cancel")
    filled = arm.consume("BUY_UP", 12.0, 10.0)
    arm.led_q_up += filled
    ok(filled == 2.0 and arm.filled_current_order["BUY_UP"] == 2.0,
       "partial fill before effective time is retained")
    _, _, _, side, generation = heapq.heappop(pending)
    arm.apply_cancel(side, generation, (0.49, 0.51, 10.0, 11.0, 0.01))
    ok(arm.cancel_counts["effective"] == 1
       and arm.cancel_counts["partial_fill_then_cancel"] == 1,
       "matching generation records partial-fill cancellation")
    ok(arm.buy.resting == 5.0 and arm.buy.qahead == 10.0,
       "effective cancellation reposts full size behind displayed depth")
    arm.observe_signal("BUY_UP", False, 1.1, 0, pending, sequence)
    sequence = arm.observe_signal("BUY_UP", True, 1.2, 0, pending, sequence)
    _, _, _, side, old_generation = heapq.heappop(pending)
    arm.reposition("BUY_UP", 0.48, 7.0)
    arm.apply_cancel(side, old_generation, (0.48, 0.51, 7.0, 8.0, 0.01))
    ok(arm.cancel_counts["stale_or_unsupported_effective"] == 1,
       "natural replacement makes pending cancel stale")
    ok(arm.buy.level == 0.48 and arm.buy.qahead == 7.0,
       "stale cancel cannot remove replacement order")

    skew_spec = {
        "cell": "SKEW", "placement": "SKEW_LB", "r_cut": 0, "size": 5.0,
        "skew": True, "skew_band_shares": 5.0,
        "front_on_repost": False, "cancel": True,
        "cancel_latency_ms": 25, "cancel_join_only": True,
    }
    skew_arm = CancelArm(skew_spec)
    skew_arm.buy.front = True
    skew_arm.reposition("BUY_UP", 0.49, 9.0)
    skip_pending: list[tuple[float, int, int, str, int]] = []
    skew_arm.observe_signal("BUY_UP", True, 1.0, 0, skip_pending, 0)
    ok(not skip_pending
       and skew_arm.cancel_counts["front_or_absent_signal_skips"] == 1,
       "front placement is never cancelled by JOIN-schema model")
    ok(len(_specs(25)) == 5 and _specs(25)[-1]["cancel"],
       "five-arm composition grid is pinned")
    ok(CANDIDATE_CELLS["btc"] == {"horizon_ms": 50, "latency_ms": 25},
       "fast BTC diagnostic cell is pinned")
    ok(CANDIDATE_CELLS["eth"] == {"horizon_ms": 250, "latency_ms": 100},
       "strongest ETH diagnostic cell is pinned")
    ok(STATE_LAG_S == 0.0, "composition uses v5 unlagged PM source profile")
    print(f"[cancel-skew] selftest OK — {checks} checks")
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
