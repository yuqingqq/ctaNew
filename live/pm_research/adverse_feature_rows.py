"""Knowledge-time adverse-move feature rows and action-bound labels.

Research only: this module reads recorded Polymarket and Binance tapes.  It has
no venue, order, or cancellation port.  The implementation follows
``plans/BE_ADVERSE_MOVE_PLAN.md`` section 6.

Feature objects and future labels are separate dataclasses on purpose.  A
feature row can therefore be serialized or passed to a fit without carrying the
event that reveals its outcome.
"""

from __future__ import annotations

import bisect
import collections
import datetime as dt
import gzip
import hashlib
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TextIO

import edge_layer1 as el
import flow_fill_development as fd
import flow_intensity as fi

REPO = Path(__file__).resolve().parents[2]
HF_RAW = REPO / "data/mm_hf/raw"

GRID_MS = 100
BUCKET_NS = GRID_MS * 1_000_000
LOOKBACK_MS = 5_000
PREDICTION_HORIZON_S = 1.0
MARKOUT_HORIZON_S = 5.0
ACTION_SIZE = 5.0
LATENCY_MS = (0, 150, 250, 350, 500)
MAKER_REBATE_CENTS_PER_SHARE = 0.168
MAX_BOOK_STALENESS_MS = 500.0
MAX_DEPTH_STALENESS_MS = 500.0
MAX_BOOK_GAP_MS = 2_000.0

SOURCE_PROFILE = {
    "feed_class": "DIRECT_EVENT_WS",
    "clock": "LOCAL_RECV_NS",
    "polymarket": "CLOB_WS_LOCAL_RECEIPT",
    "pm_feature_state_lag_ms": int(fd.STATE_LAG_S * 1000),
    "label_markout_clock": "UNLAGGED_LOCAL_RECEIPT",
    "bin_ms": GRID_MS,
    "bookTicker": "recv_ns,E,T,u,bid,bid_qty,ask,ask_qty",
    "trade": "recv_ns,E,T,trade_id,price,qty,is_buyer_maker",
    "depth20": "recv_ns,E,T,u,bids,asks",
    "completed_buckets_only": True,
}

FEATURE_NAMES = (
    "maker_side_sign",
    "pm_mid",
    "pm_logit",
    "pm_spread_cents",
    "pm_spread_ticks",
    "pm_queue_log1p",
    "pm_imbalance",
    "pm_maker_signed_imbalance",
    "pm_moneyness",
    "time_remaining_frac",
    "hf_maker_signed_ret_100ms_bps",
    "hf_maker_signed_ret_200ms_bps",
    "hf_maker_signed_ret_500ms_bps",
    "hf_maker_signed_ret_1000ms_bps",
    "hf_maker_signed_ret_2000ms_bps",
    "hf_maker_signed_ret_5000ms_bps",
    "hf_spread_bps",
    "hf_book_imbalance",
    "hf_maker_signed_book_imbalance",
    "hf_microprice_offset_bps",
    "hf_maker_signed_microprice_offset_bps",
    "hf_book_updates_100ms",
    "hf_book_updates_200ms",
    "hf_book_updates_1000ms",
    "hf_maker_signed_trade_qty_100ms",
    "hf_maker_signed_trade_qty_200ms",
    "hf_maker_signed_trade_qty_500ms",
    "hf_maker_signed_trade_qty_1000ms",
    "hf_maker_signed_trade_qty_5000ms",
    "hf_trade_abs_qty_1000ms",
    "hf_trade_abs_qty_5000ms",
    "hf_trade_count_1000ms",
    "hf_trade_count_5000ms",
    "hf_maker_signed_trade_imbalance_1000ms",
    "hf_maker_signed_trade_imbalance_5000ms",
    "hf_depth5_imbalance",
    "hf_maker_signed_depth5_imbalance",
    "hf_depth20_imbalance",
    "hf_maker_signed_depth20_imbalance",
    "hf_depth5_depletion_1000ms",
)


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


FEATURE_SCHEMA_HASH = _stable_hash({
    "names": FEATURE_NAMES,
    "grid_ms": GRID_MS,
    "lookback_ms": LOOKBACK_MS,
    "completed_buckets_only": True,
})
SOURCE_PROFILE_HASH = _stable_hash(SOURCE_PROFILE)
ACTION_SCHEMA_HASH = _stable_hash({
    "placement": "JOIN_TOUCH_BACK_DISPLAYED",
    "size_shares": ACTION_SIZE,
    "queue_rule": "BACK_DISPLAYED",
    "prediction_horizon_s": PREDICTION_HORIZON_S,
    "markout_horizon_s": MARKOUT_HORIZON_S,
    "markout_clock": "UNLAGGED_LOCAL_RECEIPT",
    "latency_rungs_ms": LATENCY_MS,
    "maker_rebate_cents_per_share": MAKER_REBATE_CENTS_PER_SHARE,
})


@dataclass(frozen=True, slots=True)
class PMState:
    t: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    tick: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True, slots=True)
class PMTrade:
    t: float
    taker_side: str
    exec_p_up: float
    size: float
    event_ms: int | None = None
    recv_ns: int | None = None
    transaction_hash: str | None = None


@dataclass(slots=True)
class PMTape:
    slug: str
    coin: str
    window_start: int
    state_t: list[float]
    states: list[PMState | None]
    trades: list[PMTrade]
    gaps: list[tuple[float, float]]
    tick_changes: list[float]
    mark_state_t: list[float] = field(default_factory=list)
    mark_states: list[PMState | None] = field(default_factory=list)
    event_clock_floor_ms: float | None = None
    event_clock_observations: int = 0
    replay_state_t: list[float] = field(default_factory=list)
    replay_states: list[PMState | None] = field(default_factory=list)
    replay_state_ns: list[int] = field(default_factory=list)
    trade_t: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.trade_t = [x.t for x in self.trades]
        if not self.mark_state_t:
            self.mark_state_t = self.state_t
            self.mark_states = self.states
        if not self.replay_state_t:
            self.replay_state_t = self.state_t
            self.replay_states = self.states
        if not self.replay_state_ns:
            window_ns = self.window_start * 1_000_000_000
            self.replay_state_ns = [
                window_ns + round(value * 1_000_000_000)
                for value in self.replay_state_t
            ]
        if not (len(self.replay_state_t) == len(self.replay_states)
                == len(self.replay_state_ns)):
            raise ValueError("replay state clocks have inconsistent lengths")

    def state_at(self, t: float) -> PMState | None:
        i = bisect.bisect_right(self.state_t, t) - 1
        return None if i < 0 else self.states[i]

    def mark_state_at(self, t: float) -> PMState | None:
        """Unlagged receipt-time state, used only after an action for labels."""
        i = bisect.bisect_right(self.mark_state_t, t) - 1
        return None if i < 0 else self.mark_states[i]

    def touched(self, lo: float, hi: float) -> bool:
        if any(not (b < lo or a > hi) for a, b in self.gaps):
            return True
        i = bisect.bisect_left(self.tick_changes, lo)
        return i < len(self.tick_changes) and self.tick_changes[i] <= hi


@dataclass(slots=True)
class AdverseFeatureRow:
    row_id: str
    action_ref: str
    slug: str
    coin: str
    day: str
    as_of_ns: int
    elapsed_s: float
    maker_side: str
    values: dict[str, float]
    feature_schema_hash: str
    input_t_known_max: int
    input_staleness_ms: float
    state_hash: str
    source_profile_hash: str
    status: str = "AVAILABLE"
    unavailable_reason: str | None = None

    def contract_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "action_ref": self.action_ref,
            "as_of": self.as_of_ns,
            "maker_side": self.maker_side,
            "values": dict(self.values),
            "feature_schema_hash": self.feature_schema_hash,
            "input_t_known_max": self.input_t_known_max,
            "input_staleness": self.input_staleness_ms / 1000.0,
            "state_hash": self.state_hash,
            "source_profile_hash": self.source_profile_hash,
            "status": self.status,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class ActionLabel:
    row_id: str
    status: str
    unavailable_reason: str | None
    filled_shares: float
    toxic_fill: int
    markout_cents_per_share: float | None
    gross_keep_pnl_cents: float
    avoidable_adverse_cents: dict[int, float]
    cancel_delta_cents: dict[int, float]
    prevented_shares: dict[int, float]


def _same_state(a: PMState | None, b: PMState | None) -> bool:
    if a is None or b is None:
        return a is b
    return (a.bid, a.ask, a.bid_size, a.ask_size, a.tick) == \
           (b.bid, b.ask, b.bid_size, b.ask_size, b.tick)


def build_pm_tape(path: Path, up_id: str, down_id: str,
                  gaps: Sequence[tuple[float, float]],
                  feature_state_lag_s: float = fd.STATE_LAG_S) -> PMTape:
    """Build a point-in-time PM state tape and complement-folded trade tape."""
    if feature_state_lag_s < 0:
        raise ValueError("feature_state_lag_s must be non-negative")
    slug = path.name.split(".jsonl")[0]
    ws = int(slug.rsplit("-", 1)[1])
    window_ns = ws * 1_000_000_000
    state = fd.BookState()
    mark_state = fd.BookState()
    state_t: list[float] = [-60.0]
    states: list[PMState | None] = [None]
    mark_state_t: list[float] = [-60.0]
    mark_states: list[PMState | None] = [None]
    replay_state_t: list[float] = [-60.0]
    replay_states: list[PMState | None] = [None]
    replay_state_ns: list[int] = [window_ns - 60_000_000_000]
    trades: list[PMTrade] = []
    ticks: list[float] = []
    pending: list[tuple[int, int, str, dict[str, Any]]] = []
    gap_starts_ns = sorted(
        window_ns + round(a * 1_000_000_000)
        for a, _ in gaps if 0.0 <= a <= fi.WINDOW_S)
    gap_i = 0
    seq = 0
    seen_tx: set[str] = set()
    event_clock_floor_ms: float | None = None
    event_clock_observations = 0

    def snapshot(t: float) -> None:
        q = state.quote()
        row = None if q is None else PMState(t, q[0], q[1], q[2], q[3], q[4])
        if not _same_state(states[-1], row):
            state_t.append(t)
            states.append(row)

    def replay_snapshot(t: float, t_ns: int) -> None:
        """Retain every queue-engine resync, even if the quote is unchanged.

        An unchanged visible quote is feature-equivalent, but it is not always
        order-state-equivalent: a fill may have changed inventory skew since
        the preceding book mutation.  The authoritative queue loop resyncs on
        every applied mutation, so the trace tape must preserve that clock.
        """
        q = state.quote()
        row = None if q is None else PMState(t, q[0], q[1], q[2], q[3], q[4])
        replay_state_t.append(t)
        replay_states.append(row)
        replay_state_ns.append(t_ns)

    def mark_snapshot(t: float) -> None:
        q = mark_state.quote()
        row = None if q is None else PMState(t, q[0], q[1], q[2], q[3], q[4])
        if not _same_state(mark_states[-1], row):
            mark_state_t.append(t)
            mark_states.append(row)

    def advance(to_ns: int) -> None:
        nonlocal gap_i
        while True:
            candidates: list[int] = []
            if pending:
                candidates.append(pending[0][0])
            if gap_i < len(gap_starts_ns):
                candidates.append(gap_starts_ns[gap_i])
            if not candidates or min(candidates) > to_ns:
                return
            when_ns = min(candidates)
            when = (when_ns - window_ns) / 1e9
            if (gap_i < len(gap_starts_ns)
                    and gap_starts_ns[gap_i] == when_ns):
                state.clear()
                mark_state.clear()
                pending.clear()
                heapq.heapify(pending)
                snapshot(when)
                replay_snapshot(when, when_ns)
                mark_snapshot(when)
                while (gap_i < len(gap_starts_ns)
                       and gap_starts_ns[gap_i] == when_ns):
                    gap_i += 1
            mutated = False
            while pending and pending[0][0] <= when_ns:
                _, _, kind, data = heapq.heappop(pending)
                state.apply(kind, data)
                mutated = True
            if mutated:
                snapshot(when)
                replay_snapshot(when, when_ns)

    def schedule(received_ns: int, kind: str, data: dict[str, Any]) -> None:
        nonlocal seq
        seq += 1
        received = (received_ns - window_ns) / 1e9
        mark_state.apply(kind, data)
        mark_snapshot(received)
        if kind == "tick":
            ticks.append(received)
        effective_ns = received_ns + round(feature_state_lag_s * 1_000_000_000)
        heapq.heappush(pending, (effective_ns, seq, kind, data))

    for line in fi._gz_lines(path):
        if not any(x in line for x in (fi.TRADE_MARK, fi.QUOTE_MARK,
                                       fd.BOOK_MARK, fd.TICK_MARK)):
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            recv_ns = int(parts[0])
            recv = (recv_ns - window_ns) / 1e9
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            continue
        if recv < -60.0 or recv > fi.WINDOW_S:
            continue
        advance(recv_ns)
        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict):
                continue
            try:
                event_ms = int(msg["timestamp"])
            except (KeyError, TypeError, ValueError):
                event_ms = None
            if event_ms is not None:
                delta_ms = (
                    recv_ns - event_ms * 1_000_000
                ) / 1_000_000
                event_clock_observations += 1
                if (event_clock_floor_ms is None
                        or delta_ms < event_clock_floor_ms):
                    event_clock_floor_ms = delta_ms
            et = msg.get("event_type")
            aid = str(msg.get("asset_id"))
            if (et == "book" or ("bids" in msg and "asks" in msg)) and aid == up_id:
                data = fd._parse_book(msg)
                if data:
                    schedule(recv_ns, "book", data)
                continue
            if et == "price_change":
                for pc in msg.get("price_changes", []):
                    if str(pc.get("asset_id")) != up_id:
                        continue
                    try:
                        data = {"side": str(pc["side"]).upper(),
                                "price": float(pc["price"]), "size": float(pc["size"]),
                                "best_bid": float(pc["best_bid"]),
                                "best_ask": float(pc["best_ask"])}
                    except (KeyError, TypeError, ValueError):
                        continue
                    if 0.0 <= data["best_bid"] < data["best_ask"] <= 1.0:
                        schedule(recv_ns, "price", data)
                continue
            if et == "tick_size_change" and aid == up_id:
                try:
                    schedule(recv_ns, "tick", {"tick": float(msg["new_tick_size"])})
                except (KeyError, TypeError, ValueError):
                    pass
                continue
            if et != "last_trade_price" or aid not in (up_id, down_id):
                continue
            tx = str(msg.get("transaction_hash") or "")
            if tx and tx in seen_tx:
                continue
            if tx:
                seen_tx.add(tx)
            try:
                native_px = float(msg["price"])
                size = float(msg["size"])
                native_side = str(msg["side"]).upper()
            except (KeyError, TypeError, ValueError):
                continue
            # The queue replay deliberately warms up from -60s.  Keep those
            # trades on the same tape as the warm-up book state; feature and
            # label readers already slice strictly after their action start.
            if -60.0 <= recv <= fi.WINDOW_S:
                down = aid == down_id
                trades.append(PMTrade(recv, fi.fold_side(native_side, down),
                                      fi.fold_price(native_px, down), size,
                                      event_ms=event_ms, recv_ns=recv_ns,
                                      transaction_hash=tx or None))
    advance(window_ns + fi.WINDOW_S * 1_000_000_000)
    trades.sort(key=lambda x: x.t)
    return PMTape(
        slug, slug.split("-")[0], ws, state_t, states, trades,
        sorted(gaps), sorted(ticks), mark_state_t=mark_state_t,
        mark_states=mark_states,
        event_clock_floor_ms=event_clock_floor_ms,
        event_clock_observations=event_clock_observations,
        replay_state_t=replay_state_t,
        replay_states=replay_states,
        replay_state_ns=replay_state_ns,
    )


def _unavailable_label(row_id: str, reason: str) -> ActionLabel:
    zeros = {x: 0.0 for x in LATENCY_MS}
    return ActionLabel(row_id, "UNAVAILABLE", reason, 0.0, 0, None, 0.0,
                       dict(zeros), dict(zeros), dict(zeros))


def label_action(tape: PMTape, row_id: str, start: float, maker_side: str,
                 level: float, queue_ahead: float) -> ActionLabel:
    end = start + PREDICTION_HORIZON_S
    if tape.touched(start, end):
        return _unavailable_label(row_id, "PM_GAP_OR_TICK_IN_ACTION_HORIZON")
    lo = bisect.bisect_right(tape.trade_t, start)
    hi = bisect.bisect_right(tape.trade_t, end)
    cumulative = 0.0
    filled_before = 0.0
    tranches: list[tuple[float, float]] = []
    for trade in tape.trades[lo:hi]:
        if not fd.reaches_action(trade.taker_side, trade.exec_p_up, maker_side, level):
            continue
        cumulative += trade.size
        filled_now = min(ACTION_SIZE, max(0.0, cumulative - queue_ahead))
        delta = filled_now - filled_before
        if delta > 1e-12:
            tranches.append((trade.t, delta))
            filled_before = filled_now
        if filled_before >= ACTION_SIZE - 1e-12:
            break
    if not tranches:
        zeros = {x: 0.0 for x in LATENCY_MS}
        return ActionLabel(row_id, "AVAILABLE", None, 0.0, 0, None, 0.0,
                           dict(zeros), dict(zeros), dict(zeros))
    if any(tape.touched(ft, ft + MARKOUT_HORIZON_S) for ft, _ in tranches):
        return _unavailable_label(row_id, "PM_GAP_OR_TICK_IN_MARKOUT_HORIZON")

    marked: list[tuple[float, float, float]] = []
    for fill_t, size in tranches:
        later = tape.mark_state_at(fill_t + MARKOUT_HORIZON_S)
        if later is None:
            return _unavailable_label(row_id, "NO_PM_MARKOUT_STATE")
        markout_cents = el.maker_sign(maker_side) * (later.mid - level) * 100.0
        marked.append((fill_t, size, markout_cents))
    shares = sum(size for _, size, _ in marked)
    pnl = sum(size * markout for _, size, markout in marked)
    swm = pnl / shares
    avoidable: dict[int, float] = {}
    cancel_delta: dict[int, float] = {}
    prevented: dict[int, float] = {}
    for latency in LATENCY_MS:
        effective = start + latency / 1000.0
        eligible = [(ft, size, mk) for ft, size, mk in marked if ft >= effective - 1e-12]
        prevented[latency] = sum(size for _, size, _ in eligible)
        avoidable[latency] = sum(size * max(-mk, 0.0) for _, size, mk in eligible)
        cancel_delta[latency] = -sum(
            size * (mk + MAKER_REBATE_CENTS_PER_SHARE) for _, size, mk in eligible)
    return ActionLabel(row_id, "AVAILABLE", None, shares, int(swm < 0.0), swm,
                       pnl, avoidable, cancel_delta, prevented)


def materialize_pm_rows(tape: PMTape) -> tuple[list[AdverseFeatureRow], list[ActionLabel]]:
    rows: list[AdverseFeatureRow] = []
    labels: list[ActionLabel] = []
    start_ms = LOOKBACK_MS
    stop_ms = int((fi.WINDOW_S - PREDICTION_HORIZON_S - MARKOUT_HORIZON_S) * 1000)
    day = fi.slug_day(tape.slug)
    for elapsed_ms in range(start_ms, stop_ms + 1, GRID_MS):
        elapsed = elapsed_ms / 1000.0
        state = tape.state_at(elapsed)
        if state is None:
            continue
        mid = state.mid
        denom = state.bid * state.bid_size + state.ask * state.ask_size
        imbalance = ((state.bid * state.bid_size - state.ask * state.ask_size) / denom
                     if denom > 0 else 0.0)
        for maker_side, level, queue in (
                ("BUY_UP", state.bid, state.bid_size),
                ("SELL_UP", state.ask, state.ask_size)):
            sign = el.maker_sign(maker_side)
            action = {"slug": tape.slug, "elapsed_ms": elapsed_ms,
                      "maker_side": maker_side, "level": level,
                      "size": ACTION_SIZE, "queue_rule": "BACK_DISPLAYED",
                      "queue_ahead": queue}
            action_ref = _stable_hash(action)
            row_id = _stable_hash({"action_ref": action_ref, "as_of_ms": elapsed_ms})
            as_of_ns = tape.window_start * 1_000_000_000 + elapsed_ms * 1_000_000
            known_ns = tape.window_start * 1_000_000_000 + int(state.t * 1e9)
            values = {
                "maker_side_sign": sign,
                "pm_mid": mid,
                "pm_logit": math.log(min(max(mid, 1e-6), 1 - 1e-6) /
                                     (1 - min(max(mid, 1e-6), 1 - 1e-6))),
                "pm_spread_cents": (state.ask - state.bid) * 100.0,
                "pm_spread_ticks": (state.ask - state.bid) / state.tick,
                "pm_queue_log1p": math.log1p(max(0.0, queue)),
                "pm_imbalance": imbalance,
                "pm_maker_signed_imbalance": sign * imbalance,
                "pm_moneyness": abs(mid - 0.5),
                "time_remaining_frac": (fi.WINDOW_S - elapsed) / fi.WINDOW_S,
            }
            state_hash = _stable_hash({
                "state": {
                    "t": state.t,
                    "bid": state.bid,
                    "ask": state.ask,
                    "bid_size": state.bid_size,
                    "ask_size": state.ask_size,
                    "tick": state.tick,
                },
                "action": action,
            })
            row = AdverseFeatureRow(
                row_id, action_ref, tape.slug, tape.coin, day, as_of_ns, elapsed,
                maker_side, values, FEATURE_SCHEMA_HASH, known_ns,
                max(0.0, (as_of_ns - known_ns) / 1e6), state_hash,
                SOURCE_PROFILE_HASH)
            rows.append(row)
            labels.append(label_action(tape, row_id, elapsed, maker_side, level, queue))
    return rows, labels


@dataclass(slots=True)
class HFHour:
    book: list[tuple[int, int, float, float, float, float, int]]
    trades: list[tuple[int, int, float, float, int]]
    depth: list[tuple[int, int, float, float, float, float]]
    book_keys: list[int] = field(init=False)
    trade_keys: list[int] = field(init=False)
    depth_keys: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.book_keys = [x[0] for x in self.book]
        self.trade_keys = [x[0] for x in self.trades]
        self.depth_keys = [x[0] for x in self.depth]


def _stream_file(stream: str, symbol: str, key: str) -> TextIO | None:
    root = HF_RAW / stream / symbol
    gz = root / f"{key}.csv.gz"
    raw = root / f"{key}.csv"
    if gz.exists():
        return gzip.open(gz, "rt")
    if raw.exists():
        return raw.open()
    return None


def _depth_sums(encoded: str) -> tuple[float, float]:
    vals: list[float] = []
    for item in encoded.split("|"):
        try:
            _, qty = item.split("@", 1)
            vals.append(float(qty))
        except (ValueError, TypeError):
            continue
    return sum(vals[:5]), sum(vals[:20])


def load_hf_hour(symbol: str, hour_start_ns: int) -> HFHour:
    hour = dt.datetime.fromtimestamp(hour_start_ns / 1e9, dt.timezone.utc)
    key = hour.strftime("%Y%m%d_%H")
    end_ns = hour_start_ns + 3_600_000_000_000
    book_map: dict[int, list[float]] = {}
    trade_map: dict[int, list[float]] = {}
    depth_map: dict[int, tuple[int, float, float, float, float]] = {}

    fh = _stream_file("bookTicker", symbol, key)
    if fh is not None:
        with fh:
            for line in fh:
                p = line.rstrip().split(",")
                if len(p) < 8:
                    continue
                try:
                    recv = int(p[0])
                    if not (hour_start_ns <= recv < end_ns):
                        continue
                    bucket = recv // BUCKET_NS
                    old = book_map.get(bucket)
                    count = 1 if old is None else int(old[5]) + 1
                    book_map[bucket] = [recv, float(p[4]), float(p[5]),
                                        float(p[6]), float(p[7]), count]
                except ValueError:
                    continue

    fh = _stream_file("trade", symbol, key)
    if fh is not None:
        with fh:
            for line in fh:
                p = line.rstrip().split(",")
                if len(p) < 7:
                    continue
                try:
                    recv = int(p[0])
                    if not (hour_start_ns <= recv < end_ns):
                        continue
                    bucket = recv // BUCKET_NS
                    qty = float(p[5])
                    signed = -qty if int(p[6]) else qty
                    old = trade_map.setdefault(bucket, [recv, 0.0, 0.0, 0.0])
                    old[0] = max(old[0], recv)
                    old[1] += signed
                    old[2] += qty
                    old[3] += 1
                except ValueError:
                    continue

    fh = _stream_file("depth20", symbol, key)
    if fh is not None:
        with fh:
            for line in fh:
                p = line.rstrip().split(",", 5)
                if len(p) < 6:
                    continue
                try:
                    recv = int(p[0])
                    if not (hour_start_ns <= recv < end_ns):
                        continue
                    b5, b20 = _depth_sums(p[4])
                    a5, a20 = _depth_sums(p[5])
                    depth_map[recv // BUCKET_NS] = (recv, b5, a5, b20, a20)
                except ValueError:
                    continue

    book = [(k, int(v[0]), v[1], v[2], v[3], v[4], int(v[5]))
            for k, v in sorted(book_map.items())]
    trades = [(k, int(v[0]), v[1], v[2], int(v[3]))
              for k, v in sorted(trade_map.items())]
    depth = [(k, v[0], v[1], v[2], v[3], v[4])
             for k, v in sorted(depth_map.items())]
    return HFHour(book, trades, depth)


def _last_before(keys: Sequence[int], cutoff_bucket: int) -> int:
    return bisect.bisect_left(keys, cutoff_bucket) - 1


def _imbalance(bid: float, ask: float) -> float:
    return (bid - ask) / (bid + ask) if bid + ask > 0 else 0.0


def _trade_stats(hour: HFHour, start_bucket: int,
                 end_bucket: int) -> tuple[float, float, int, int | None]:
    lo = bisect.bisect_left(hour.trade_keys, start_bucket)
    hi = bisect.bisect_left(hour.trade_keys, end_bucket)
    xs = hour.trades[lo:hi]
    return (sum(x[2] for x in xs), sum(x[3] for x in xs),
            sum(x[4] for x in xs), max((x[1] for x in xs), default=None))


def hf_features_at(hour: HFHour, as_of_ns: int,
                   maker_sign: float) -> tuple[dict[str, float] | None, int, float, str | None]:
    cutoff = as_of_ns // BUCKET_NS
    bi = _last_before(hour.book_keys, cutoff)
    di = _last_before(hour.depth_keys, cutoff)
    if bi < 0 or di < 0:
        return None, 0, math.inf, "HF_NO_CURRENT_BOOK_OR_DEPTH"
    b = hour.book[bi]
    d = hour.depth[di]
    book_stale = (as_of_ns - b[1]) / 1e6
    depth_stale = (as_of_ns - d[1]) / 1e6
    if book_stale > MAX_BOOK_STALENESS_MS:
        return None, b[1], book_stale, "HF_BOOK_STALE"
    if depth_stale > MAX_DEPTH_STALENESS_MS:
        return None, max(b[1], d[1]), depth_stale, "HF_DEPTH_STALE"

    mid = (b[2] + b[4]) / 2.0
    rets: dict[int, float] = {}
    for w in (100, 200, 500, 1000, 2000, 5000):
        old_cut = (as_of_ns - w * 1_000_000) // BUCKET_NS
        oi = _last_before(hour.book_keys, old_cut)
        if oi < 0:
            return None, max(b[1], d[1]), max(book_stale, depth_stale), "HF_INCOMPLETE_5S_HISTORY"
        old = hour.book[oi]
        old_mid = (old[2] + old[4]) / 2.0
        rets[w] = 10_000.0 * math.log(mid / old_mid)

    history_lo = bisect.bisect_left(hour.book_keys, cutoff - LOOKBACK_MS // GRID_MS)
    history = hour.book[history_lo:bi + 1]
    pts = [as_of_ns - LOOKBACK_MS * 1_000_000] + [x[1] for x in history] + [as_of_ns]
    if max((y - x) / 1e6 for x, y in zip(pts, pts[1:])) > MAX_BOOK_GAP_MS:
        return None, max(b[1], d[1]), max(book_stale, depth_stale), "HF_BOOK_GAP_IN_LOOKBACK"

    def updates(w: int) -> float:
        lo = bisect.bisect_left(hour.book_keys, cutoff - math.ceil(w / GRID_MS))
        return float(sum(x[6] for x in hour.book[lo:bi + 1]))

    trades: dict[int, tuple[float, float, int, int | None]] = {}
    for w in (100, 200, 500, 1000, 5000):
        trades[w] = _trade_stats(hour, (as_of_ns - w * 1_000_000) // BUCKET_NS, cutoff)

    old_di = _last_before(hour.depth_keys,
                          (as_of_ns - 1_000_000_000) // BUCKET_NS)
    if old_di < 0:
        return None, max(b[1], d[1]), max(book_stale, depth_stale), "HF_NO_DEPTH_HISTORY"
    old_d = hour.depth[old_di]
    book_imb = _imbalance(b[3], b[5])
    micro = ((b[4] * b[3] + b[2] * b[5]) / (b[3] + b[5])
             if b[3] + b[5] > 0 else mid)
    depth5 = _imbalance(d[2], d[3])
    depth20 = _imbalance(d[4], d[5])
    depletion = math.log((d[2] + d[3] + 1e-9) / (old_d[2] + old_d[3] + 1e-9))

    values: dict[str, float] = {
        **{f"hf_maker_signed_ret_{w}ms_bps": maker_sign * rets[w]
           for w in (100, 200, 500, 1000, 2000, 5000)},
        "hf_spread_bps": 10_000.0 * (b[4] - b[2]) / mid,
        "hf_book_imbalance": book_imb,
        "hf_maker_signed_book_imbalance": maker_sign * book_imb,
        "hf_microprice_offset_bps": 10_000.0 * (micro - mid) / mid,
        "hf_maker_signed_microprice_offset_bps": maker_sign * 10_000.0 * (micro - mid) / mid,
        "hf_book_updates_100ms": updates(100),
        "hf_book_updates_200ms": updates(200),
        "hf_book_updates_1000ms": updates(1000),
        **{f"hf_maker_signed_trade_qty_{w}ms": maker_sign * trades[w][0]
           for w in (100, 200, 500, 1000, 5000)},
        "hf_trade_abs_qty_1000ms": trades[1000][1],
        "hf_trade_abs_qty_5000ms": trades[5000][1],
        "hf_trade_count_1000ms": float(trades[1000][2]),
        "hf_trade_count_5000ms": float(trades[5000][2]),
        "hf_maker_signed_trade_imbalance_1000ms": maker_sign * (
            trades[1000][0] / trades[1000][1] if trades[1000][1] > 0 else 0.0),
        "hf_maker_signed_trade_imbalance_5000ms": maker_sign * (
            trades[5000][0] / trades[5000][1] if trades[5000][1] > 0 else 0.0),
        "hf_depth5_imbalance": depth5,
        "hf_maker_signed_depth5_imbalance": maker_sign * depth5,
        "hf_depth20_imbalance": depth20,
        "hf_maker_signed_depth20_imbalance": maker_sign * depth20,
        "hf_depth5_depletion_1000ms": depletion,
    }
    last_trade = trades[5000][3] or 0
    known_max = max(b[1], d[1], last_trade)
    return values, known_max, max(book_stale, depth_stale), None


def enrich_hf(rows: Sequence[AdverseFeatureRow]) -> dict[str, int]:
    groups: dict[tuple[str, int], list[AdverseFeatureRow]] = collections.defaultdict(list)
    symbols = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
    hour_ns = 3_600_000_000_000
    for row in rows:
        groups[(row.coin, row.as_of_ns // hour_ns * hour_ns)].append(row)
    reasons: collections.Counter[str] = collections.Counter()
    for (coin, hour), group in sorted(groups.items()):
        symbol = symbols.get(coin)
        if symbol is None:
            for row in group:
                row.status, row.unavailable_reason = "UNAVAILABLE", "UNSUPPORTED_COIN"
            reasons["UNSUPPORTED_COIN"] += len(group)
            continue
        print(f"[adverse-feature] {coin} {dt.datetime.fromtimestamp(hour / 1e9, dt.timezone.utc):%Y-%m-%dT%H}:00 rows={len(group)}", flush=True)
        data = load_hf_hour(symbol, hour)
        for row in group:
            hf, known, stale, reason = hf_features_at(
                data, row.as_of_ns, row.values["maker_side_sign"])
            if hf is None:
                row.status, row.unavailable_reason = "UNAVAILABLE", reason
                reasons[str(reason)] += 1
                continue
            row.values.update(hf)
            row.input_t_known_max = max(row.input_t_known_max, known)
            row.input_staleness_ms = max(row.input_staleness_ms, stale)
            if row.input_t_known_max > row.as_of_ns:
                raise AssertionError("future HF event entered AdverseFeatureRow")
            if tuple(row.values) != FEATURE_NAMES:
                raise AssertionError("feature order/schema mismatch")
            if not all(math.isfinite(v) for v in row.values.values()):
                row.status, row.unavailable_reason = "UNAVAILABLE", "NONFINITE_FEATURE"
                reasons["NONFINITE_FEATURE"] += 1
    return dict(reasons)


def paired_rows(features: Sequence[AdverseFeatureRow], labels: Sequence[ActionLabel]
                ) -> Iterator[tuple[AdverseFeatureRow, ActionLabel]]:
    by_id = {x.row_id: x for x in labels}
    for row in features:
        label = by_id.get(row.row_id)
        if label is None:
            raise AssertionError(f"missing label for {row.row_id}")
        yield row, label


def selftest() -> int:
    checks = 0

    def ok(cond: bool, name: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(name)
        checks += 1

    ok(len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "feature names unique")
    ok(FEATURE_SCHEMA_HASH == _stable_hash({"names": FEATURE_NAMES, "grid_ms": 100,
                                             "lookback_ms": 5000,
                                             "completed_buckets_only": True}),
       "feature schema hash deterministic")
    states = [PMState(0.0, 0.49, 0.51, 10.0, 10.0, 0.01),
              PMState(5.6, 0.46, 0.48, 10.0, 10.0, 0.01)]
    tape = PMTape("btc-updown-5m-1780000000", "btc", 1780000000,
                  [0.0, 5.6], states,
                  [PMTrade(0.2, "SELL", 0.49, 6.0),
                   PMTrade(0.6, "SELL", 0.49, 8.0)], [], [])
    label = label_action(tape, "r", 0.0, "BUY_UP", 0.49, 10.0)
    ok(label.status == "AVAILABLE" and abs(label.filled_shares - 4.0) < 1e-12,
       "displayed-back queue fill")
    ok(label.toxic_fill == 1 and abs(label.markout_cents_per_share + 2.0) < 1e-12,
       "maker-signed toxic markout")
    ok(label.prevented_shares[500] == 4.0,
       "latency is inside counterfactual")
    ok(label.cancel_delta_cents[500] > 0.0, "cancel saves adverse fill net of rebate")

    good = PMTape("btc-updown-5m-1780000000", "btc", 1780000000,
                  [0.0, 5.6], [states[0], PMState(5.6, 0.51, 0.53, 10, 10, .01)],
                  tape.trades, [], [])
    good_label = label_action(good, "g", 0.0, "BUY_UP", 0.49, 10.0)
    ok(good_label.toxic_fill == 0 and good_label.cancel_delta_cents[500] < 0,
       "cancel forfeits favourable spread and drift")
    nofill = label_action(tape, "n", 0.0, "SELL_UP", 0.51, 10.0)
    dual_clock = PMTape(
        "btc-updown-5m-1780000000", "btc", 1780000000,
        good.state_t, good.states, tape.trades, [], [],
        mark_state_t=tape.state_t, mark_states=tape.states,
    )
    dual_label = label_action(dual_clock, "d", 0.0, "BUY_UP", 0.49, 10.0)
    ok(dual_label.toxic_fill == 1,
       "future label uses unlagged mark tape, not lagged feature tape")

    ok(nofill.toxic_fill == 0 and nofill.filled_shares == 0,
       "no-fill is a joint-target negative")

    # The row at q=1.0 s must use bucket 9 and never bucket 10, which may contain
    # events received after q. This is the load-bearing no-lookahead boundary.
    q = 1_000_000_000
    book = [(9, 950_000_000, 100, 2, 101, 2, 1),
            (10, 1_050_000_000, 200, 2, 201, 2, 1)]
    ok(_last_before([x[0] for x in book], q // BUCKET_NS) == 0,
       "incomplete current bucket excluded")
    ok(el.maker_sign("BUY_UP") == 1 and el.maker_sign("SELL_UP") == -1,
       "side sign pinned")
    print(f"[adverse-feature] selftest OK — {checks} checks")
    return 0


if __name__ == "__main__":
    selftest()
