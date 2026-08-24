"""Queue-realistic skew and cancel-and-hold offline replay.

Same-price zero-queue FRONT remains only as a frozen upper-bound comparator.
Corrected arms join existing touch depth and may improve the inventory-reducing
side by one tick only when the spread is at least two ticks.  Research only;
there is no live order, cancel, or venue port.

Commands::

    python3 live/pm_research/policy_optimizer_queue_realistic.py --selftest
    python3 live/pm_research/policy_optimizer_queue_realistic.py run
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
import policy_optimizer_cancel_hold_skew as hold
import policy_optimizer_cancel_skew as base
import warning_window as ww


OUT = base.fi.PM / "derived/policy_optimizer_queue_realistic_v1.json"
PROTOCOL = Path(__file__).with_name(
    "QUEUE_REALISTIC_CANCEL_SKEW_PROTOCOL.md")
HOLD_ARTIFACT = hold.OUT
QR_SKEW = "QR_SKEW_ONLY"
QR_BASELINE = "QR_CANCEL_HOLD_X_SKEW"
CELL_NAMES = (*hold.CELL_NAMES, QR_SKEW, QR_BASELINE)

JOIN_EXISTING = "JOIN_EXISTING"
PRICE_IMPROVE = "PRICE_IMPROVE_1T"
ABSENT = "ABSENT"
HELD_OUT = "HELD_OUT"


def _qr_spec(cell: str, latency_ms: int, cancel: bool) -> dict[str, Any]:
    return {
        "cell": cell,
        "placement": "QUEUE_REALISTIC_SKEW",
        "skew": True,
        "skew_band_shares": base.ps.SKEW_BAND_SHARES,
        "front_on_repost": False,
        "queue_realistic": True,
        "cancel": cancel,
        "cancel_latency_ms": latency_ms,
        "cancel_join_only": True,
        "cancel_hold": cancel,
        "protect_reducing_side": cancel,
        "r_cut": 0,
        "size": 5.0,
    }


def _specs(latency_ms: int) -> list[dict[str, Any]]:
    return [
        *hold._specs(latency_ms),
        _qr_spec(QR_SKEW, latency_ms, False),
        _qr_spec(QR_BASELINE, latency_ms, True),
    ]


class QueueRealisticArm(hold.HoldArm):
    """HoldArm whose corrected cells use executable price/queue placement."""

    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        self.queue_realistic = bool(spec.get("queue_realistic", False))
        self.placement_kind = {side: ABSENT for side in self.SIDES}

    @staticmethod
    def _valid_tick(tick: float) -> bool:
        return 0.0 < tick < 1.0

    def desired_order(
            self, maker_side: str,
            quote: tuple[float, float, float, float, float]
            ) -> tuple[float, float, bool, str, bool]:
        """Return level, queue ahead, front flag, kind, improve eligibility."""
        bid, ask, bid_size, ask_size, tick = quote
        reducing = self.target_reducing(maker_side)
        can_improve = bool(
            reducing and self._valid_tick(tick)
            and ask - bid >= 2.0 * tick - 1e-12)
        if can_improve:
            level = bid + tick if maker_side == "BUY_UP" else ask - tick
            level = round(level, 10)
            if not (bid < level < ask):
                raise RuntimeError("price-improvement order would cross or join")
            return level, 0.0, True, PRICE_IMPROVE, True
        if maker_side == "BUY_UP":
            return bid, max(0.0, bid_size), False, JOIN_EXISTING, False
        return ask, max(0.0, ask_size), False, JOIN_EXISTING, False

    def _place_from_quote(
            self, maker_side: str,
            quote: tuple[float, float, float, float, float],
            *, force: bool = False, release_repost: bool = False) -> bool:
        if self.held[maker_side]:
            self.cancel_counts["held_reposition_suppressed"] += 1
            return False
        level, displayed, front, kind, eligible = self.desired_order(
            maker_side, quote)
        self.cancel_counts["qr_sync_evaluations"] += 1
        if eligible:
            self.cancel_counts["qr_price_improve_eligible_syncs"] += 1
        side = self.side(maker_side)
        changed = bool(
            force or side.level is None
            or abs(side.level - level) > 1e-12
            or self.placement_kind[maker_side] != kind)
        if not changed:
            return False
        side.front = front
        self.reposition(maker_side, level, displayed)
        self.placement_kind[maker_side] = kind
        self.actions += 1
        self.cancel_counts[
            "qr_price_improve_placements" if kind == PRICE_IMPROVE
            else "qr_join_placements"] += 1
        if release_repost or self.release_repost_pending[maker_side]:
            self.release_repost_pending[maker_side] = False
            self.cancel_counts["reposts"] += 1
            self.cancel_counts["hold_release_reposts"] += 1
        return True

    def sync_from_quote(
            self, quote: tuple[float, float, float, float, float]) -> None:
        if not self.queue_realistic:
            raise RuntimeError("sync_from_quote called on legacy arm")
        self.apply_skew_intent()
        for maker_side in self.SIDES:
            self._place_from_quote(maker_side, quote)

    def release_hold(
            self, maker_side: str, when: float,
            quote: tuple[float, float, float, float, float] | None,
            reason: str) -> bool:
        if not self.queue_realistic:
            return super().release_hold(maker_side, when, quote, reason)
        if not self.held[maker_side]:
            return False
        self._close_hold_clock(maker_side, when)
        self.held[maker_side] = False
        self.cancel_counts["hold_releases"] += 1
        self.cancel_counts[f"hold_release_{reason}"] += 1
        self.apply_skew_intent()
        if quote is None:
            side = self.side(maker_side)
            side.reposition(None, 0.0)
            self.generation[maker_side] += 1
            self.placement_front[maker_side] = False
            self.filled_current_order[maker_side] = 0.0
            self.placement_kind[maker_side] = ABSENT
            self.release_repost_pending[maker_side] = True
            self.cancel_counts["hold_release_waiting_for_book"] += 1
            return True
        self._place_from_quote(
            maker_side, quote, force=True, release_repost=True)
        return True

    def apply_cancel(
            self, maker_side: str, generation: int,
            quote: tuple[float, float, float, float, float] | None,
            when: float | None = None) -> None:
        if not self.queue_realistic:
            super().apply_cancel(maker_side, generation, quote, when)
            return
        if not self.hold_enabled:
            super().apply_cancel(maker_side, generation, quote, when)
            return
        if when is None:
            raise ValueError("queue-realistic hold requires effective time")
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
        if self.filled_current_order[maker_side] > 1e-12:
            self.cancel_counts["partial_fill_then_cancel"] += 1
        if cleared or reducing:
            self._place_from_quote(maker_side, quote, force=True)
            self.cancel_counts["effective"] += 1
            self.cancel_counts["reposts"] += 1
            self.cancel_counts[
                "effective_immediate_repost_cleared" if cleared
                else "effective_immediate_repost_inventory_reducing"] += 1
            return

        side.reposition(None, 0.0)
        self.generation[maker_side] += 1
        self.placement_front[maker_side] = False
        self.filled_current_order[maker_side] = 0.0
        self.placement_kind[maker_side] = HELD_OUT
        self.held[maker_side] = True
        self.hold_started_at[maker_side] = when
        self.cancel_counts["effective"] += 1
        self.cancel_counts["hold_entries"] += 1

    def displayed_for_fill(
            self, maker_side: str,
            quote: tuple[float, float, float, float, float]) -> float:
        if (self.queue_realistic
                and self.placement_kind[maker_side] == PRICE_IMPROVE):
            return 0.0
        return quote[2] if maker_side == "BUY_UP" else quote[3]

    def consume(self, maker_side: str, volume: float,
                displayed: float) -> float:
        kind = self.placement_kind[maker_side]
        side = self.side(maker_side)
        before = side.resting
        filled = super().consume(maker_side, volume, displayed)
        if self.queue_realistic and filled > 0:
            prefix = (
                "qr_price_improve" if kind == PRICE_IMPROVE else "qr_join")
            self.cancel_counts[f"{prefix}_fill_events"] += 1
            self.cancel_counts[f"{prefix}_fill_microshares"] += round(
                filled * 1_000_000)
            if before <= filled + 1e-12 and kind == PRICE_IMPROVE:
                # The hypothetical inside-spread level has no displayed queue.
                self.placement_front[maker_side] = True
                self.cancel_counts["qr_price_improve_reposts"] += 1
        return filled


def replay_cells_queue_realistic(
        path: Path, up_id: str, down_id: str,
        gaps: Sequence[tuple[float, float]],
        specs: Sequence[dict[str, Any]],
        signals: dict[str, Sequence[tuple[float, bool]]],
        lag_s: float = base.STATE_LAG_S) -> dict[str, base.el.WindowFills] | None:
    """Hold event loop plus explicit JOIN/one-tick-improve placement."""
    slug = path.name.split(".jsonl")[0]
    try:
        window_start = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None

    state = base.fd.BookState()
    arms = [QueueRealisticArm(spec) for spec in specs]
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
    for maker_side in QueueRealisticArm.SIDES:
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
            if arm.queue_realistic:
                arm.apply_skew_intent()
                arm.release_if_reducing(when, quote)
                arm.sync_from_quote(quote)
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
            recv_ns = int(parts[0])
            received = (
                recv_ns - window_start * 1_000_000_000
            ) / 1e9
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
            bid, ask, _, _, _ = quote
            mid_now = (bid + ask) / 2.0
            record_mid(received)
            micro = abs(size - base.fi.MICRO_SIZE) < 1e-9
            for arm in arms:
                if (taker_side == "BUY" and arm.sell.level is not None
                        and execution_price + 1e-12 >= arm.sell.level):
                    level = arm.sell.level
                    displayed = arm.displayed_for_fill("SELL_UP", quote)
                    filled = arm.consume("SELL_UP", size, displayed)
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
                    displayed = arm.displayed_for_fill("BUY_UP", quote)
                    filled = arm.consume("BUY_UP", size, displayed)
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
                f"[queue-realistic] reconciliation break {slug} "
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
    result = hold._cell_metrics(windows, cell)
    syncs = _diagnostic_sum(windows, cell, "qr_sync_evaluations")
    eligible = _diagnostic_sum(
        windows, cell, "qr_price_improve_eligible_syncs")
    result.update({
        "qr_join_placements": int(_diagnostic_sum(
            windows, cell, "qr_join_placements")),
        "qr_price_improve_placements": int(_diagnostic_sum(
            windows, cell, "qr_price_improve_placements")),
        "qr_price_improve_eligible_sync_fraction": (
            eligible / syncs if syncs else None),
        "qr_join_fill_events": int(_diagnostic_sum(
            windows, cell, "qr_join_fill_events")),
        "qr_price_improve_fill_events": int(_diagnostic_sum(
            windows, cell, "qr_price_improve_fill_events")),
        "qr_join_fill_shares": _diagnostic_sum(
            windows, cell, "qr_join_fill_microshares") / 1_000_000.0,
        "qr_price_improve_fill_shares": _diagnostic_sum(
            windows, cell, "qr_price_improve_fill_microshares") / 1_000_000.0,
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
    legacy_specs = hold._specs(latency_ms)
    reference = hold.replay_cells_with_hold(
        path, up_id, down_id, gaps, legacy_specs, signals,
        lag_s=base.STATE_LAG_S)
    candidate = replay_cells_queue_realistic(
        path, up_id, down_id, gaps, legacy_specs, signals,
        lag_s=base.STATE_LAG_S)
    if reference is None or candidate is None:
        raise RuntimeError("legacy parity control unavailable")
    legacy_parity = all(
        pb.fill_key(reference[cell]) == pb.fill_key(candidate[cell])
        and reference[cell].diagnostics == candidate[cell].diagnostics
        for cell in hold.CELL_NAMES)
    if not legacy_parity:
        raise RuntimeError("queue loop changed a legacy arm")

    false_signals = {
        side: [(when, False) for when, _ in rows]
        for side, rows in signals.items()
    }
    false_result = replay_cells_queue_realistic(
        path, up_id, down_id, gaps, _specs(latency_ms), false_signals,
        lag_s=base.STATE_LAG_S)
    if false_result is None or not pb.conformant(
            false_result[QR_BASELINE], false_result[QR_SKEW]):
        raise RuntimeError("queue-realistic all-false parity failure")

    first = replay_cells_queue_realistic(
        path, up_id, down_id, gaps, _specs(latency_ms), signals,
        lag_s=base.STATE_LAG_S)
    second = replay_cells_queue_realistic(
        path, up_id, down_id, gaps, _specs(latency_ms), signals,
        lag_s=base.STATE_LAG_S)
    deterministic = bool(first is not None and second is not None and all(
        pb.fill_key(first[cell]) == pb.fill_key(second[cell])
        and first[cell].diagnostics == second[cell].diagnostics
        for cell in CELL_NAMES))
    if not deterministic:
        raise RuntimeError("queue-realistic deterministic replay failure")
    return {
        "all_six_legacy_arms_exact_parity": True,
        "queue_realistic_all_false_parity": True,
        "queue_realistic_deterministic": True,
        "baseline": QR_BASELINE,
    }


def run() -> dict[str, Any]:
    if not base.MODEL_ARTIFACT.exists():
        raise RuntimeError("v5 harmful-flow model artifact is missing")
    if not HOLD_ARTIFACT.exists():
        raise RuntimeError("cancel-and-hold artifact is missing")
    artifact = json.loads(base.MODEL_ARTIFACT.read_text())
    hold_artifact = json.loads(HOLD_ARTIFACT.read_text())
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
    print(f"[queue-realistic] controls PASS {controls}", flush=True)

    windows: dict[tuple[str, str, str], list[Any]] = (
        collections.defaultdict(list))
    for index, batch in enumerate(batches, 1):
        slug, path, up_id, down_id, gaps = selected[batch.slug]
        coin, day = batch.coin, batch.day
        got = replay_cells_queue_realistic(
            path, up_id, down_id, gaps,
            _specs(base.CANDIDATE_CELLS[coin]["latency_ms"]),
            schedules[slug], lag_s=base.STATE_LAG_S)
        if got is None:
            continue
        for cell, window in got.items():
            windows[(cell, coin, day)].append(window)
        print(f"[queue-realistic] {index}/{len(batches)} {slug}", flush=True)

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
            "corrected_baseline_vs_old_hold": _comparison(
                cells, days, coin, QR_BASELINE, hold.HOLD_CELL),
            "corrected_baseline_vs_corrected_skew": _comparison(
                cells, days, coin, QR_BASELINE, QR_SKEW),
            "corrected_baseline_vs_old_immediate": _comparison(
                cells, days, coin, QR_BASELINE, "CANCEL_X_SKEW"),
            "corrected_skew_vs_old_skew": _comparison(
                cells, days, coin, QR_SKEW, "SKEW_ONLY"),
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "QUEUE_REALISTIC_CANCEL_HOLD_X_SKEW_OFFLINE_V1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT_DIAGNOSTIC",
        "verdict": "NON_PROMOTABLE_QUEUE_CORRECTION",
        "decision_eligible": False,
        "candidate_frozen": False,
        "promotion_authorized": False,
        "reasons": [
            "harmful_flow_v5_failed_its_predeclared_model_gate",
            "same_visible_three_train_two_development_days",
            "one_window_per_coin_day",
            "latency_is_assumed_not_measured_cancel_effective",
            "inside_spread_competitor_response_is_unavailable",
            "queue_rejoin_cost_and_live_ack_are_unavailable",
        ],
        "baseline": QR_BASELINE,
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
            "existing_level": "JOIN_BEHIND_DISPLAYED_DEPTH",
            "price_improvement": "REDUCING_SIDE_ONE_TICK_ONLY",
            "price_improvement_min_spread_ticks": 2,
            "same_price_zero_queue_ahead": "FORBIDDEN",
            "new_level_race_win": "NOT_ASSUMED",
            "harmful_threshold": 0.5,
            "cancel_eligible": "ACTUAL_JOIN_AND_NOT_INVENTORY_REDUCING",
            "effective_while_harmful": "HELD_OUT",
            "release": "SIGNAL_CLEAR_OR_INVENTORY_REDUCING",
            "hold_timeout": "NONE",
            "incentives": "EXCLUDED_BY_USER_DIRECTION",
        },
        "controls": controls,
        "signal_audit": schedule_audit,
        "cells": cells,
        "comparisons": comparisons,
        "source_hold_artifact_id": hold_artifact["artifact_id"],
        "provenance": {
            "polymarket": base.fi.provenance(sampled=sampled),
            "v5_model_artifact_sha256": base._file_sha(base.MODEL_ARTIFACT),
            "hold_artifact_sha256": base._file_sha(HOLD_ARTIFACT),
            "code_sha256": base._file_sha(Path(__file__)),
            "protocol_sha256": base._file_sha(PROTOCOL),
            "hold_engine_sha256": base._file_sha(Path(hold.__file__)),
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
    print(f"[queue-realistic] receipt -> {OUT}", flush=True)
    return result


def selftest() -> int:
    checks = 0

    def ok(condition: bool, name: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(name)
        checks += 1

    specs = _specs(25)
    ok(len(specs) == 8 and tuple(spec["cell"] for spec in specs) == CELL_NAMES,
       "eight-arm queue correction grid is pinned")
    ok(hold._specs(25) == specs[:6], "six legacy arms remain unchanged")

    one_tick = QueueRealisticArm(_qr_spec(QR_SKEW, 25, False))
    one_tick.led_q_up = 5.1
    one_tick.apply_skew_intent()
    one_tick_quote = (0.49, 0.50, 10.0, 11.0, 0.01)
    one_tick.sync_from_quote(one_tick_quote)
    ok(one_tick.placement_kind["SELL_UP"] == JOIN_EXISTING
       and one_tick.sell.level == 0.50 and one_tick.sell.qahead == 11.0,
       "one-tick reducing side joins behind displayed depth")
    ok(one_tick.placement_kind["BUY_UP"] == JOIN_EXISTING
       and one_tick.buy.level == 0.49 and one_tick.buy.qahead == 10.0,
       "one-tick increasing side also joins")
    ok(not one_tick.placement_front["BUY_UP"]
       and not one_tick.placement_front["SELL_UP"],
       "no same-price order receives zero queue ahead")

    two_tick = QueueRealisticArm(_qr_spec(QR_SKEW, 25, False))
    two_tick.led_q_up = 5.1
    two_tick.apply_skew_intent()
    two_tick_quote = (0.48, 0.50, 10.0, 11.0, 0.01)
    two_tick.sync_from_quote(two_tick_quote)
    ok(two_tick.placement_kind["SELL_UP"] == PRICE_IMPROVE
       and two_tick.sell.level == 0.49 and two_tick.sell.qahead == 0.0,
       "two-tick reducing side improves exactly one tick")
    ok(0.48 < two_tick.sell.level < 0.50,
       "price-improvement quote is strictly non-crossing")
    ok(two_tick.placement_kind["BUY_UP"] == JOIN_EXISTING
       and two_tick.buy.qahead == 10.0,
       "two-tick increasing side still joins existing depth")

    flat = QueueRealisticArm(_qr_spec(QR_SKEW, 25, False))
    flat.sync_from_quote(two_tick_quote)
    ok(flat.placement_kind == {
        "BUY_UP": JOIN_EXISTING, "SELL_UP": JOIN_EXISTING},
       "near-flat policy never price-improves")

    cancel = QueueRealisticArm(_qr_spec(QR_BASELINE, 25, True))
    cancel.sync_from_quote(one_tick_quote)
    pending: list[tuple[float, int, int, str, int]] = []
    seq = cancel.observe_signal(
        "BUY_UP", True, 1.0, 0, pending, 0, one_tick_quote)
    ok(seq == 1 and len(pending) == 1,
       "joined non-reducing quote is cancel eligible")
    effective, _, _, side, generation = heapq.heappop(pending)
    cancel.apply_cancel(side, generation, one_tick_quote, effective)
    ok(cancel.held["BUY_UP"] and cancel.buy.level is None,
       "effective harmful cancel enters held state")
    cancel.observe_signal(
        "BUY_UP", False, 1.1, 0, pending, seq, one_tick_quote)
    ok(not cancel.held["BUY_UP"]
       and cancel.placement_kind["BUY_UP"] == JOIN_EXISTING
       and cancel.buy.qahead == 10.0,
       "signal clear reposts with queue-realistic JOIN")

    protected = QueueRealisticArm(_qr_spec(QR_BASELINE, 25, True))
    protected.led_q_up = 5.1
    protected.apply_skew_intent()
    protected.sync_from_quote(two_tick_quote)
    protected_pending: list[tuple[float, int, int, str, int]] = []
    protected.observe_signal(
        "SELL_UP", True, 1.0, 0, protected_pending, 0, two_tick_quote)
    ok(not protected_pending,
       "price-improved reducing quote remains cancellation protected")

    ok(base.CANDIDATE_CELLS["btc"] == {
        "horizon_ms": 50, "latency_ms": 25},
       "BTC diagnostic cell remains pinned")
    ok(base.CANDIDATE_CELLS["eth"] == {
        "horizon_ms": 250, "latency_ms": 100},
       "ETH diagnostic cell remains pinned")
    ok(base.STATE_LAG_S == 0.0,
       "source state profile remains identical to v5")
    print(f"[queue-realistic] selftest OK — {checks} checks")
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
