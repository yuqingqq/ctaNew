"""Offline stateful cancel-and-hold composition with pessimistic SKEW_LB.

The v5 harmful-flow signal may cancel an eligible joined quote after assumed
latency.  Unlike the frozen immediate-repost comparator, a matching cancel
holds the quote out until the signal clears or the side becomes inventory
reducing.  Research only: there is no live order, cancel, or venue port.

Commands::

    python3 live/pm_research/policy_optimizer_cancel_hold_skew.py --selftest
    python3 live/pm_research/policy_optimizer_cancel_hold_skew.py run
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import heapq
import json
from pathlib import Path
from typing import Any, Sequence

import policy_bounds_v1 as pb
import policy_optimizer as opt
import policy_optimizer_cancel_skew as base
import warning_window as ww


OUT = base.fi.PM / "derived/policy_optimizer_cancel_hold_skew_v1.json"
PROTOCOL = Path(__file__).with_name("CANCEL_HOLD_SKEW_PROTOCOL.md")
IMMEDIATE_ARTIFACT = base.OUT
HOLD_CELL = "CANCEL_HOLD_X_SKEW"
CELL_NAMES = (*base.CELL_NAMES, HOLD_CELL)


def _specs(latency_ms: int) -> list[dict[str, Any]]:
    specs = base._specs(latency_ms)
    specs.append({
        "cell": HOLD_CELL,
        "placement": "SKEW_LB",
        "skew": True,
        "skew_band_shares": base.ps.SKEW_BAND_SHARES,
        "front_on_repost": False,
        "cancel": True,
        "cancel_latency_ms": latency_ms,
        "cancel_join_only": True,
        "cancel_hold": True,
        "protect_reducing_side": True,
        "r_cut": 0,
        "size": 5.0,
    })
    return specs


class HoldArm(base.CancelArm):
    """CancelArm with an explicit quote-ineligible state after cancellation."""

    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        self.hold_enabled = bool(spec.get("cancel_hold", False))
        self.protect_reducing = bool(spec.get("protect_reducing_side", False))
        self.held = {side: False for side in self.SIDES}
        self.hold_started_at: dict[str, float | None] = {
            side: None for side in self.SIDES}
        self.signal_harmful = {side: False for side in self.SIDES}
        self.release_repost_pending = {side: False for side in self.SIDES}

    def target_reducing(self, maker_side: str) -> bool:
        if not self.skew:
            return False
        buy_front, sell_front = base.ps._target_front(
            self.net, self.skew_band)
        return buy_front if maker_side == "BUY_UP" else sell_front

    def ensure_position(self, maker_side: str, level: float,
                        displayed: float) -> bool:
        """Post/reposition unless the side is deliberately held out."""
        if self.held[maker_side]:
            self.cancel_counts["held_reposition_suppressed"] += 1
            return False
        self.reposition(maker_side, level, displayed)
        self.actions += 1
        if self.release_repost_pending[maker_side]:
            self.release_repost_pending[maker_side] = False
            self.cancel_counts["reposts"] += 1
            self.cancel_counts["hold_release_reposts"] += 1
        return True

    def _close_hold_clock(self, maker_side: str, when: float) -> None:
        started = self.hold_started_at[maker_side]
        if started is not None:
            self.cancel_counts["held_side_milliseconds"] += max(
                0, round((when - started) * 1000))
        self.hold_started_at[maker_side] = None

    def release_hold(self, maker_side: str, when: float,
                     quote: tuple[float, float, float, float, float] | None,
                     reason: str) -> bool:
        if not self.held[maker_side]:
            return False
        self._close_hold_clock(maker_side, when)
        self.held[maker_side] = False
        self.cancel_counts["hold_releases"] += 1
        self.cancel_counts[f"hold_release_{reason}"] += 1
        self.apply_skew_intent()
        side = self.side(maker_side)
        if quote is None:
            side.reposition(None, 0.0)
            self.generation[maker_side] += 1
            self.placement_front[maker_side] = False
            self.filled_current_order[maker_side] = 0.0
            self.release_repost_pending[maker_side] = True
            self.cancel_counts["hold_release_waiting_for_book"] += 1
            return True
        bid, ask, bid_size, ask_size, _ = quote
        level = bid if maker_side == "BUY_UP" else ask
        displayed = bid_size if maker_side == "BUY_UP" else ask_size
        self.reposition(maker_side, level, displayed)
        self.actions += 1
        self.cancel_counts["reposts"] += 1
        self.cancel_counts["hold_release_reposts"] += 1
        return True

    def release_if_reducing(
            self, when: float,
            quote: tuple[float, float, float, float, float] | None) -> None:
        if not self.hold_enabled or not self.protect_reducing:
            return
        for maker_side in self.SIDES:
            if self.held[maker_side] and self.target_reducing(maker_side):
                self.release_hold(
                    maker_side, when, quote, "inventory_reducing")

    def observe_signal(
            self, maker_side: str, harmful: bool, when: float,
            arm_index: int,
            pending: list[tuple[float, int, int, str, int]],
            sequence: int,
            quote: tuple[float, float, float, float, float] | None = None
            ) -> int:
        if not self.hold_enabled:
            return super().observe_signal(
                maker_side, harmful, when, arm_index, pending, sequence)

        self.signal_harmful[maker_side] = bool(harmful)
        self.cancel_counts["signal_evaluations"] += 1
        if not harmful:
            self.armed[maker_side] = True
            self.release_hold(maker_side, when, quote, "signal_clear")
            return sequence

        self.cancel_counts["harmful_signal_evaluations"] += 1
        if not self.armed[maker_side]:
            self.cancel_counts["persistent_signal_skips"] += 1
            return sequence
        # The true edge is consumed even when the current action is ineligible.
        self.armed[maker_side] = False
        if self.held[maker_side]:
            self.cancel_counts["held_signal_skips"] += 1
            return sequence
        side = self.side(maker_side)
        if self.protect_reducing and self.target_reducing(maker_side):
            self.cancel_counts["reducing_side_signal_skips"] += 1
            return sequence
        if (side.level is None
                or (self.cancel_join_only
                    and self.placement_front[maker_side])):
            self.cancel_counts["front_or_absent_signal_skips"] += 1
            return sequence
        if self.pending_generation[maker_side] is not None:
            self.cancel_counts["pending_signal_skips"] += 1
            return sequence

        generation = self.generation[maker_side]
        sequence += 1
        heapq.heappush(pending, (
            when + self.cancel_latency_s, sequence, arm_index,
            maker_side, generation))
        self.pending_generation[maker_side] = generation
        self.cancel_counts["submitted"] += 1
        self.actions += 1
        return sequence

    def apply_cancel(
            self, maker_side: str, generation: int,
            quote: tuple[float, float, float, float, float] | None,
            when: float | None = None) -> None:
        if not self.hold_enabled:
            super().apply_cancel(maker_side, generation, quote)
            return
        if when is None:
            raise ValueError("hold cancellation requires effective time")
        if self.pending_generation[maker_side] == generation:
            self.pending_generation[maker_side] = None
        side = self.side(maker_side)
        if (generation != self.generation[maker_side]
                or side.level is None or quote is None
                or (self.cancel_join_only
                    and self.placement_front[maker_side])):
            self.cancel_counts["stale_or_unsupported_effective"] += 1
            return

        cleared = not self.signal_harmful[maker_side]
        reducing = self.protect_reducing and self.target_reducing(maker_side)
        if cleared or reducing:
            super().apply_cancel(maker_side, generation, quote)
            key = "cleared" if cleared else "inventory_reducing"
            self.cancel_counts[f"effective_immediate_repost_{key}"] += 1
            return

        if self.filled_current_order[maker_side] > 1e-12:
            self.cancel_counts["partial_fill_then_cancel"] += 1
        side.reposition(None, 0.0)
        self.generation[maker_side] += 1
        self.placement_front[maker_side] = False
        self.filled_current_order[maker_side] = 0.0
        self.held[maker_side] = True
        self.hold_started_at[maker_side] = when
        self.cancel_counts["effective"] += 1
        self.cancel_counts["hold_entries"] += 1

    def finalize_holds(self, when: float) -> None:
        for maker_side in self.SIDES:
            if self.held[maker_side]:
                self._close_hold_clock(maker_side, when)


def replay_cells_with_hold(
        path: Path, up_id: str, down_id: str,
        gaps: Sequence[tuple[float, float]],
        specs: Sequence[dict[str, Any]],
        signals: dict[str, Sequence[tuple[float, bool]]],
        lag_s: float = base.STATE_LAG_S) -> dict[str, base.el.WindowFills] | None:
    """Reference event loop with an explicit quote-eligibility hold state."""
    slug = path.name.split(".jsonl")[0]
    try:
        window_start = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None

    state = base.fd.BookState()
    arms = [HoldArm(spec) for spec in specs]
    diagnostics: collections.Counter[str] = collections.Counter()
    seen_transactions: set[str] = set()
    mid_t: list[float] = []
    mid_v: list[float] = []
    bad_intervals = [
        (start, end) for start, end in gaps
        if end >= 0.0 and start <= base.fi.WINDOW_S
    ]
    state_pending: list[tuple[float, int, str, dict[str, Any]]] = []
    cancel_pending: list[tuple[float, int, int, str, int]] = []
    state_sequence = 0
    cancel_sequence = 0
    gap_starts = sorted(
        start for start, _ in gaps if 0.0 <= start <= base.fi.WINDOW_S)
    gap_index = 0
    signal_sequence = 0
    signal_pending: list[tuple[float, int, str, bool]] = []
    for maker_side in HoldArm.SIDES:
        for when, harmful in signals.get(maker_side, ()):
            signal_sequence += 1
            heapq.heappush(signal_pending, (
                when + base.SIGNAL_EPSILON_S, signal_sequence,
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
            arm.release_if_reducing(when, quote)
            if arm.buy.level is None or abs(arm.buy.level - bid) > 1e-12:
                arm.ensure_position("BUY_UP", bid, bid_size)
            if arm.sell.level is None or abs(arm.sell.level - ask) > 1e-12:
                arm.ensure_position("SELL_UP", ask, ask_size)
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
                quote = state.quote()
                for arm_index, arm in enumerate(arms):
                    cancel_sequence = arm.observe_signal(
                        maker_side, harmful, when, arm_index,
                        cancel_pending, cancel_sequence, quote)
            while cancel_pending and cancel_pending[0][0] <= when + 1e-12:
                _, _, arm_index, maker_side, generation = heapq.heappop(
                    cancel_pending)
                arms[arm_index].apply_cancel(
                    maker_side, generation, state.quote(), when)
            record_mid(when)

    def schedule_state(received: float, kind: str,
                       data: dict[str, Any]) -> None:
        nonlocal state_sequence
        state_sequence += 1
        heapq.heappush(state_pending, (
            received + lag_s, state_sequence, kind, data))

    for line in base.fi._gz_lines(path):
        if not any(marker in line for marker in (
                base.fi.TRADE_MARK, base.fi.QUOTE_MARK,
                base.fd.BOOK_MARK, base.fd.TICK_MARK)):
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
        if received < -60.0 or received > base.fi.WINDOW_S:
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
                data = base.fd._parse_book(message)
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
                    received + max(base.el.HORIZONS)))
                diagnostics["tick_changes"] += 1
                try:
                    schedule_state(received, "tick", {
                        "tick": float(message["new_tick_size"])})
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
            execution_price = base.fi.fold_price(native_price, is_down)
            taker_side = base.fi.fold_side(native_side, is_down)
            quote = state.quote()
            if quote is None:
                diagnostics["trades_no_state"] += 1
                continue
            bid, ask, bid_size, ask_size, _ = quote
            mid_now = (bid + ask) / 2.0
            record_mid(received)
            micro = abs(size - base.fi.MICRO_SIZE) < 1e-9
            for arm in arms:
                if (taker_side == "BUY" and arm.sell.level is not None
                        and execution_price + 1e-12 >= arm.sell.level):
                    level = arm.sell.level
                    filled = arm.consume("SELL_UP", size, ask_size)
                    if filled > 0:
                        arm.fills.append(base.el.Fill(
                            received, "SELL_UP", level, filled,
                            mid_now, micro))
                        arm.led_q_dn += filled
                        arm.apply_skew_intent()
                        arm.release_if_reducing(received, quote)
                elif (taker_side == "SELL" and arm.buy.level is not None
                      and execution_price <= arm.buy.level + 1e-12):
                    level = arm.buy.level
                    filled = arm.consume("BUY_UP", size, bid_size)
                    if filled > 0:
                        arm.fills.append(base.el.Fill(
                            received, "BUY_UP", level, filled,
                            mid_now, micro))
                        arm.led_q_up += filled
                        arm.apply_skew_intent()
                        arm.release_if_reducing(received, quote)
        advance(received + base.SIGNAL_EPSILON_S)

    advance(base.fi.WINDOW_S)
    if not mid_t:
        return None
    output: dict[str, base.el.WindowFills] = {}
    coin = slug.split("-")[0]
    for arm in arms:
        arm.finalize_holds(base.fi.WINDOW_S)
        bought = sum(
            fill.size for fill in arm.fills
            if fill.maker_side == "BUY_UP")
        sold = sum(
            fill.size for fill in arm.fills
            if fill.maker_side == "SELL_UP")
        if (abs(bought - arm.led_q_up) > 1e-9
                or abs(sold - arm.led_q_dn) > 1e-9):
            raise SystemExit(
                f"[cancel-hold-skew] reconciliation break {slug} "
                f"{arm.spec['cell']}")
        item_diagnostics = dict(diagnostics)
        cell = arm.spec["cell"]
        item_diagnostics[f"actions:{cell}"] = arm.actions
        item_diagnostics[f"skew_intent_flips:{cell}"] = (
            arm.skew_intent_flips)
        for name, value in arm.cancel_counts.items():
            item_diagnostics[f"cancel_{name}:{cell}"] = value
        output[cell] = base.el.WindowFills(
            slug, coin, arm.fills, mid_t, mid_v,
            list(bad_intervals), item_diagnostics)
    return output


def _diagnostic_sum(windows: Sequence[Any], cell: str, name: str) -> float:
    key = f"cancel_{name}:{cell}"
    return float(sum(window.diagnostics.get(key, 0) for window in windows))


def _cell_metrics(windows: Sequence[Any], cell: str) -> dict[str, Any]:
    result = base._cell_metrics(windows, cell)
    held_ms = _diagnostic_sum(windows, cell, "held_side_milliseconds")
    side_seconds = 2.0 * base.fi.WINDOW_S * len(windows)
    result.update({
        "cancel_hold_entries": int(_diagnostic_sum(
            windows, cell, "hold_entries")),
        "cancel_hold_releases": int(_diagnostic_sum(
            windows, cell, "hold_releases")),
        "cancel_hold_release_signal_clear": int(_diagnostic_sum(
            windows, cell, "hold_release_signal_clear")),
        "cancel_hold_release_inventory_reducing": int(_diagnostic_sum(
            windows, cell, "hold_release_inventory_reducing")),
        "cancel_reducing_side_signal_skips": int(_diagnostic_sum(
            windows, cell, "reducing_side_signal_skips")),
        "held_side_seconds": held_ms / 1000.0,
        "held_side_fraction": (
            held_ms / 1000.0 / side_seconds if side_seconds else None),
    })
    return result


def _comparison(cells: dict[str, Any], days: Sequence[str], coin: str,
                candidate: str, baseline_cell: str) -> dict[str, Any]:
    per_day: dict[str, float | None] = {}
    for day in days:
        candidate_value = cells[candidate][coin][day][
            "pnl_per_window_cents"]
        baseline_value = cells[baseline_cell][coin][day][
            "pnl_per_window_cents"]
        per_day[day] = (
            None if candidate_value is None or baseline_value is None
            else float(candidate_value - baseline_value))
    return {
        "per_day_delta_pnl_cents": per_day,
        "positive_all_days": bool(
            per_day and all(value is not None and value > 0
                            for value in per_day.values())),
        "positive_development_days": bool(all(
            per_day.get(day) is not None and per_day[day] > 0
            for day in base.v5.HOLDOUT_DAYS)),
    }


def _controls(item: Sequence[Any],
              signals: dict[str, list[tuple[float, bool]]],
              latency_ms: int) -> dict[str, Any]:
    _, path, up_id, down_id, gaps = item
    original_specs = base._specs(latency_ms)
    reference = base.replay_cells_with_cancel(
        path, up_id, down_id, gaps, original_specs, signals,
        lag_s=base.STATE_LAG_S)
    candidate = replay_cells_with_hold(
        path, up_id, down_id, gaps, original_specs, signals,
        lag_s=base.STATE_LAG_S)
    if reference is None or candidate is None:
        raise RuntimeError("existing-arm parity control unavailable")
    original_parity = all(
        pb.fill_key(reference[cell]) == pb.fill_key(candidate[cell])
        and reference[cell].diagnostics == candidate[cell].diagnostics
        for cell in base.CELL_NAMES)
    if not original_parity:
        raise RuntimeError("hold loop changed an existing arm")

    false_signals = {
        side: [(when, False) for when, _ in rows]
        for side, rows in signals.items()
    }
    false_result = replay_cells_with_hold(
        path, up_id, down_id, gaps, _specs(latency_ms), false_signals,
        lag_s=base.STATE_LAG_S)
    if false_result is None or not pb.conformant(
            false_result[HOLD_CELL], false_result["SKEW_ONLY"]):
        raise RuntimeError("hold all-false parity failure")

    first = replay_cells_with_hold(
        path, up_id, down_id, gaps, _specs(latency_ms), signals,
        lag_s=base.STATE_LAG_S)
    second = replay_cells_with_hold(
        path, up_id, down_id, gaps, _specs(latency_ms), signals,
        lag_s=base.STATE_LAG_S)
    deterministic = bool(first is not None and second is not None and all(
        pb.fill_key(first[cell]) == pb.fill_key(second[cell])
        and first[cell].diagnostics == second[cell].diagnostics
        for cell in CELL_NAMES))
    if not deterministic:
        raise RuntimeError("hold deterministic replay failure")
    return {
        "all_five_existing_arms_exact_parity": True,
        "hold_all_false_signal_parity": True,
        "hold_deterministic": True,
        "action_schema": "JOIN_NON_REDUCING_CANCEL_AND_HOLD",
    }


def run() -> dict[str, Any]:
    if not base.MODEL_ARTIFACT.exists():
        raise RuntimeError("v5 harmful-flow model artifact is missing")
    if not IMMEDIATE_ARTIFACT.exists():
        raise RuntimeError("immediate-repost cancel×skew artifact is missing")
    artifact = json.loads(base.MODEL_ARTIFACT.read_text())
    immediate_artifact = json.loads(IMMEDIATE_ARTIFACT.read_text())
    batches, sampled, slugs = base.v5.linear.build_batches(1)
    selected = {
        item[0]: item
        for _, items in ww.select_by_day(1).items()
        for item in items
        if item[0] in set(slugs)
    }
    schedules: dict[str, dict[str, list[tuple[float, bool]]]] = {}
    schedule_audit: dict[str, Any] = {}
    for batch in batches:
        schedule, audit = base._signals_for_batch(batch, artifact)
        schedules[batch.slug] = schedule
        schedule_audit[batch.slug] = audit

    first_batch = batches[0]
    controls = _controls(
        selected[first_batch.slug], schedules[first_batch.slug],
        base.CANDIDATE_CELLS[first_batch.coin]["latency_ms"])
    print(f"[cancel-hold-skew] controls PASS {controls}", flush=True)

    windows: dict[tuple[str, str, str], list[Any]] = (
        collections.defaultdict(list))
    for index, batch in enumerate(batches, 1):
        slug, path, up_id, down_id, gaps = selected[batch.slug]
        coin, day = batch.coin, batch.day
        got = replay_cells_with_hold(
            path, up_id, down_id, gaps,
            _specs(base.CANDIDATE_CELLS[coin]["latency_ms"]),
            schedules[slug], lag_s=base.STATE_LAG_S)
        if got is None:
            continue
        for cell, window in got.items():
            windows[(cell, coin, day)].append(window)
        print(
            f"[cancel-hold-skew] {index}/{len(batches)} {slug}", flush=True)

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
        comparisons[coin] = {
            "cancel_hold_vs_immediate_repost": _comparison(
                cells, days, coin, HOLD_CELL, "CANCEL_X_SKEW"),
            "cancel_hold_vs_skew": _comparison(
                cells, days, coin, HOLD_CELL, "SKEW_ONLY"),
            "cancel_hold_vs_cancel_only": _comparison(
                cells, days, coin, HOLD_CELL, "CANCEL_ONLY"),
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "HARMFUL_FLOW_CANCEL_HOLD_X_SKEW_OFFLINE_V1",
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
            "queue_rejoin_cost_and_live_ack_are_unavailable",
        ],
        "candidate_cells": base.CANDIDATE_CELLS,
        "population": {
            "n_windows": len(batches),
            "days": days,
            "training_days": list(base.v5.TRAIN_DAYS),
            "development_days": list(base.v5.HOLDOUT_DAYS),
            "selected_slugs": slugs,
            "rows_by_slug": {
                batch.slug: batch.n_rows for batch in batches},
        },
        "semantics": {
            "state_lag_ms": 0,
            "decision_cooldown_ms": base.v5.fast.COOLDOWN_MS,
            "harmful_threshold": 0.5,
            "signal_rearm": "FALSE_THEN_TRUE_EDGE",
            "cancel_eligible": "ACTUAL_JOIN_AND_NOT_INVENTORY_REDUCING",
            "effective_while_harmful": "HELD_OUT",
            "release": "SIGNAL_CLEAR_OR_INVENTORY_REDUCING",
            "hold_timeout": "NONE",
            "release_repost": "CURRENT_SKEW_INTENT_AT_CURRENT_TOUCH",
            "partial_fills_before_effective": "RETAINED",
            "cancel_binding": "ORDER_GENERATION",
            "incentives": "EXCLUDED_BY_USER_DIRECTION",
            "cancel_rejoin_cost": "UNAVAILABLE",
        },
        "controls": controls,
        "signal_audit": schedule_audit,
        "cells": cells,
        "comparisons": comparisons,
        "source_immediate_artifact_id": immediate_artifact["artifact_id"],
        "provenance": {
            "polymarket": base.fi.provenance(sampled=sampled),
            "v5_model_artifact_sha256": base._file_sha(base.MODEL_ARTIFACT),
            "immediate_artifact_sha256": base._file_sha(IMMEDIATE_ARTIFACT),
            "code_sha256": base._file_sha(Path(__file__)),
            "protocol_sha256": base._file_sha(PROTOCOL),
            "immediate_cancel_engine_sha256": base._file_sha(
                Path(base.__file__)),
            "policy_engine_sha256": base._file_sha(Path(opt.__file__)),
            "feature_builder_sha256": base._file_sha(
                Path(base.v5.fast.__file__)),
            "hf_source_identity": {
                "kind": "PATH_SIZE_MTIME_RECEIPT_NOT_CONTENT_DIGEST",
                "files": base.v5.linear._hf_manifest(slugs),
            },
        },
    }
    result["artifact_id"] = base._sha(result)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, allow_nan=False))
    print(f"[cancel-hold-skew] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0

    def ok(condition: bool, name: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(name)
        checks += 1

    specs = _specs(25)
    ok(len(specs) == 6 and tuple(spec["cell"] for spec in specs) == CELL_NAMES,
       "six-arm hold extension is pinned")
    ok(base._specs(25) == specs[:5],
       "original five-arm grid is unchanged")
    spec = specs[-1]
    ok(spec["cancel_hold"] and spec["protect_reducing_side"],
       "hold and reducing-side protection are explicit")

    arm = HoldArm(spec)
    arm.reposition("BUY_UP", 0.49, 10.0)
    pending: list[tuple[float, int, int, str, int]] = []
    sequence = arm.observe_signal(
        "BUY_UP", True, 1.0, 0, pending, 0,
        (0.49, 0.51, 10.0, 11.0, 0.01))
    ok(sequence == 1 and len(pending) == 1,
       "harmful JOIN edge submits one cancellation")
    effective, _, _, side, generation = heapq.heappop(pending)
    arm.apply_cancel(
        side, generation, (0.49, 0.51, 10.0, 11.0, 0.01), effective)
    ok(arm.held["BUY_UP"] and arm.buy.level is None,
       "matching harmful cancellation enters HELD_OUT")
    ok(not arm.ensure_position("BUY_UP", 0.48, 7.0)
       and arm.buy.level is None,
       "book change cannot repost a held quote")
    ok(arm.consume("BUY_UP", 100.0, 0.0) == 0.0,
       "held quote cannot fill")
    arm.observe_signal(
        "BUY_UP", False, 1.1, 0, pending, sequence,
        (0.48, 0.52, 7.0, 8.0, 0.01))
    ok(not arm.held["BUY_UP"] and arm.buy.level == 0.48
       and arm.buy.qahead == 7.0,
       "signal clear releases and rejoins behind displayed depth")

    reducing = HoldArm(spec)
    reducing.reposition("SELL_UP", 0.51, 8.0)
    reducing.led_q_up = 5.1
    reducing.apply_skew_intent()
    reducing_pending: list[tuple[float, int, int, str, int]] = []
    reducing.observe_signal(
        "SELL_UP", True, 1.0, 0, reducing_pending, 0,
        (0.49, 0.51, 7.0, 8.0, 0.01))
    ok(not reducing_pending
       and reducing.cancel_counts["reducing_side_signal_skips"] == 1,
       "inventory-reducing side cannot submit hold cancellation")

    front = HoldArm(spec)
    front.buy.front = True
    front.reposition("BUY_UP", 0.49, 9.0)
    front_pending: list[tuple[float, int, int, str, int]] = []
    front.observe_signal(
        "BUY_UP", True, 1.0, 0, front_pending, 0,
        (0.49, 0.51, 9.0, 8.0, 0.01))
    ok(not front_pending,
       "actual FRONT placement cannot submit hold cancellation")

    partial = HoldArm(spec)
    partial.reposition("BUY_UP", 0.49, 2.0)
    partial_pending: list[tuple[float, int, int, str, int]] = []
    partial.observe_signal(
        "BUY_UP", True, 1.0, 0, partial_pending, 0,
        (0.49, 0.51, 2.0, 8.0, 0.01))
    filled = partial.consume("BUY_UP", 3.0, 2.0)
    partial.led_q_up += filled
    effective, _, _, side, generation = heapq.heappop(partial_pending)
    partial.apply_cancel(
        side, generation, (0.49, 0.51, 2.0, 8.0, 0.01), effective)
    ok(filled == 1.0
       and partial.cancel_counts["partial_fill_then_cancel"] == 1,
       "partial fill before effective cancellation is retained")

    stale = HoldArm(spec)
    stale.reposition("BUY_UP", 0.49, 2.0)
    stale_pending: list[tuple[float, int, int, str, int]] = []
    stale.observe_signal(
        "BUY_UP", True, 1.0, 0, stale_pending, 0,
        (0.49, 0.51, 2.0, 8.0, 0.01))
    effective, _, _, side, generation = heapq.heappop(stale_pending)
    stale.reposition("BUY_UP", 0.48, 6.0)
    stale.apply_cancel(
        side, generation, (0.48, 0.52, 6.0, 8.0, 0.01), effective)
    ok(not stale.held["BUY_UP"]
       and stale.cancel_counts["stale_or_unsupported_effective"] == 1,
       "stale generation cannot hold out replacement order")

    cleared = HoldArm(spec)
    cleared.reposition("BUY_UP", 0.49, 2.0)
    cleared_pending: list[tuple[float, int, int, str, int]] = []
    seq = cleared.observe_signal(
        "BUY_UP", True, 1.0, 0, cleared_pending, 0,
        (0.49, 0.51, 2.0, 8.0, 0.01))
    cleared.observe_signal(
        "BUY_UP", False, 1.01, 0, cleared_pending, seq,
        (0.49, 0.51, 2.0, 8.0, 0.01))
    effective, _, _, side, generation = heapq.heappop(cleared_pending)
    cleared.apply_cancel(
        side, generation, (0.49, 0.51, 2.0, 8.0, 0.01), effective)
    ok(not cleared.held["BUY_UP"] and cleared.buy.level == 0.49
       and cleared.cancel_counts["effective_immediate_repost_cleared"] == 1,
       "clear before effective cancel reposts rather than holding")

    inventory_release = HoldArm(spec)
    inventory_release.reposition("SELL_UP", 0.51, 8.0)
    inventory_release.signal_harmful["SELL_UP"] = True
    inventory_release.held["SELL_UP"] = True
    inventory_release.hold_started_at["SELL_UP"] = 1.0
    inventory_release.sell.reposition(None, 0.0)
    inventory_release.led_q_up = 5.1
    inventory_release.apply_skew_intent()
    inventory_release.release_if_reducing(
        1.2, (0.49, 0.51, 7.0, 8.0, 0.01))
    ok(not inventory_release.held["SELL_UP"]
       and inventory_release.placement_front["SELL_UP"],
       "inventory-reducing transition releases held side at FRONT")

    ok(base.CANDIDATE_CELLS["btc"] == {
        "horizon_ms": 50, "latency_ms": 25},
       "BTC diagnostic cell remains pinned")
    ok(base.CANDIDATE_CELLS["eth"] == {
        "horizon_ms": 250, "latency_ms": 100},
       "ETH diagnostic cell remains pinned")
    ok(base.STATE_LAG_S == 0.0,
       "source state profile remains identical to v5")
    print(f"[cancel-hold-skew] selftest OK — {checks} checks")
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
