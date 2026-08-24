"""Exact-event fast-path adverse-selection research rows.

This is the v2 DEVELOPMENT feature/label builder described in
``plans/BE_ADVERSE_MOVE_PLAN.md`` section 8. It reads recorded direct-event
feeds only. It has no venue, order, or cancellation port.

Unlike v1, decisions are triggered by receipt-time events and are not binned.
Feature matrices and future-label matrices are stored separately in each batch
so a fit cannot receive an outcome field through its feature object.
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

import numpy as np

import adverse_feature_rows as v1
import edge_layer1 as el
import flow_fill_development as fd
import flow_intensity as fi

REPO = Path(__file__).resolve().parents[2]
HF_RAW = REPO / "data/mm_hf/raw"

COOLDOWN_MS = 10
COOLDOWN_NS = COOLDOWN_MS * 1_000_000
FAST_WINDOWS_MS = (10, 25, 50, 100, 250, 500)
PREDICTION_HORIZONS_MS = (50, 100, 250, 500, 1000)
LATENCY_MS = (10, 25, 50, 75, 100, 150, 250)
MARKOUT_HORIZON_S = 5.0
ACTION_SIZE = 5.0
MAKER_REBATE_CENTS_PER_SHARE = v1.MAKER_REBATE_CENTS_PER_SHARE
START_HISTORY_MS = max(FAST_WINDOWS_MS)
MAX_BOOK_STALENESS_MS = 2_000.0
MAX_DEPTH_STALENESS_MS = 500.0
MAX_PM_STALENESS_MS = 2_000.0

TRIGGER_HF_BOOK = 1
TRIGGER_HF_TRADE = 2
TRIGGER_PM_BOOK = 4
TRIGGER_PM_TRADE = 8

PM_FEATURE_NAMES = (
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
    "pm_state_staleness_ms",
    "trigger_hf_book",
    "trigger_hf_trade",
    "trigger_pm_book",
    "trigger_pm_trade",
)

FAST_SIGNAL_NAMES = tuple(
    name
    for window in FAST_WINDOWS_MS
    for name in (
        f"hf_maker_signed_ret_{window}ms_bps",
        f"hf_book_updates_{window}ms",
        f"hf_maker_signed_trade_qty_{window}ms",
        f"hf_trade_abs_qty_{window}ms",
        f"hf_trade_count_{window}ms",
        f"hf_maker_signed_trade_imbalance_{window}ms",
    )
)

STATE_SIGNAL_NAMES = (
    "hf_spread_bps",
    "hf_book_imbalance",
    "hf_maker_signed_book_imbalance",
    "hf_microprice_offset_bps",
    "hf_maker_signed_microprice_offset_bps",
    "hf_depth5_imbalance",
    "hf_maker_signed_depth5_imbalance",
    "hf_depth20_imbalance",
    "hf_maker_signed_depth20_imbalance",
    "hf_maker_side_depth5_log_change_500ms",
    "hf_book_staleness_ms",
    "hf_depth_staleness_ms",
)

FEATURE_NAMES = PM_FEATURE_NAMES + FAST_SIGNAL_NAMES + STATE_SIGNAL_NAMES

SOURCE_PROFILE = {
    "feed_class": "DIRECT_EVENT_WS_EXACT_RECEIPT",
    "clock": "LOCAL_RECV_NS",
    "decision_trigger": "HF_BOOK_OR_TRADE_OR_PM_BOOK_OR_TRADE",
    "decision_cooldown_ms": COOLDOWN_MS,
    "polymarket_feature_state_lag_ms": 0,
    "binance_bookTicker": "EVERY_RECORDED_EVENT",
    "binance_trade": "EVERY_RECORDED_EVENT",
    "binance_depth20": "RECORDED_100MS_CONTEXT_NOT_FAST_TRIGGER",
    "lookback_windows_ms": FAST_WINDOWS_MS,
    "future_events_excluded": True,
}


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


FEATURE_SCHEMA_HASH = _stable_hash({
    "names": FEATURE_NAMES,
    "windows_ms": FAST_WINDOWS_MS,
    "cooldown_ms": COOLDOWN_MS,
    "exact_receipt_events": True,
})
SOURCE_PROFILE_HASH = _stable_hash(SOURCE_PROFILE)
ACTION_SCHEMA_HASH = _stable_hash({
    "placement": "JOIN_TOUCH_BACK_DISPLAYED",
    "size_shares": ACTION_SIZE,
    "prediction_horizons_ms": PREDICTION_HORIZONS_MS,
    "markout_horizon_s": MARKOUT_HORIZON_S,
    "markout_clock": "UNLAGGED_LOCAL_RECEIPT",
    "latency_rungs_ms": LATENCY_MS,
    "maker_rebate_cents_per_share": MAKER_REBATE_CENTS_PER_SHARE,
})


@dataclass(slots=True)
class HFExactWindow:
    book_t: list[int]
    bid: np.ndarray
    bid_qty: np.ndarray
    ask: np.ndarray
    ask_qty: np.ndarray
    trade_t: list[int]
    trade_signed_qty: np.ndarray
    trade_abs_qty: np.ndarray
    depth_t: list[int]
    depth_bid5: np.ndarray
    depth_ask5: np.ndarray
    depth_bid20: np.ndarray
    depth_ask20: np.ndarray
    trade_signed_prefix: np.ndarray = field(init=False)
    trade_abs_prefix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.trade_signed_prefix = np.concatenate(
            ([0.0], np.cumsum(self.trade_signed_qty, dtype=float)))
        self.trade_abs_prefix = np.concatenate(
            ([0.0], np.cumsum(self.trade_abs_qty, dtype=float)))


@dataclass(frozen=True, slots=True)
class FastActionLabel:
    status: str
    unavailable_reason: str | None
    filled_shares: float
    toxic_fill: int
    markout_cents_per_share: float | None
    cancel_delta_cents: dict[int, float]
    prevented_shares: dict[int, float]


@dataclass(slots=True)
class FastWindowBatch:
    slug: str
    coin: str
    day: str
    x: np.ndarray
    as_of_ns: np.ndarray
    maker_side_sign: np.ndarray
    level: np.ndarray
    queue_ahead: np.ndarray
    trigger_mask: np.ndarray
    toxic: dict[int, np.ndarray]
    cancel_delta: dict[tuple[int, int], np.ndarray]
    prevented_shares: dict[tuple[int, int], np.ndarray]
    filled_shares: dict[int, np.ndarray]
    diagnostics: dict[str, int]

    @property
    def n_rows(self) -> int:
        return int(self.x.shape[0])

    def feature_sample(self, index: int) -> dict[str, Any]:
        side = "BUY_UP" if self.maker_side_sign[index] > 0 else "SELL_UP"
        action = {
            "slug": self.slug,
            "as_of_ns": int(self.as_of_ns[index]),
            "maker_side": side,
            "level": float(self.level[index]),
            "queue_ahead": float(self.queue_ahead[index]),
            "size_shares": ACTION_SIZE,
        }
        values = dict(zip(FEATURE_NAMES, map(float, self.x[index])))
        return {
            "row_id": _stable_hash(action),
            "slug": self.slug,
            "coin": self.coin,
            "day": self.day,
            "as_of": int(self.as_of_ns[index]),
            "maker_side": side,
            "action_ref": _stable_hash({**action, "queue_rule": "BACK_DISPLAYED"}),
            "values": values,
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "source_profile_hash": SOURCE_PROFILE_HASH,
            "action_schema_hash": ACTION_SCHEMA_HASH,
            "input_t_known_max": int(self.as_of_ns[index]),
            "input_staleness": max(
                values["pm_state_staleness_ms"],
                values["hf_book_staleness_ms"],
                values["hf_depth_staleness_ms"],
            ) / 1000.0,
            "state_hash": _stable_hash({"action": action, "values": values}),
        }


def _stream_file(stream: str, symbol: str, key: str) -> TextIO | None:
    root = HF_RAW / stream / symbol
    gz = root / f"{key}.csv.gz"
    raw = root / f"{key}.csv"
    if gz.exists():
        return gzip.open(gz, "rt")
    if raw.exists():
        return raw.open()
    return None


def _hour_keys(start_ns: int, end_ns: int) -> Iterator[str]:
    hour_ns = 3_600_000_000_000
    current = start_ns // hour_ns * hour_ns
    last = end_ns // hour_ns * hour_ns
    while current <= last:
        yield dt.datetime.fromtimestamp(
            current / 1e9, dt.timezone.utc).strftime("%Y%m%d_%H")
        current += hour_ns


def _sorted_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    if any(rows[i][0] > rows[i + 1][0] for i in range(len(rows) - 1)):
        rows.sort(key=lambda row: row[0])
    return rows


def load_hf_window(symbol: str, window_start_ns: int,
                   window_end_ns: int) -> HFExactWindow:
    """Load exact receipt events, retaining only the required window history."""
    lower_ns = window_start_ns - START_HISTORY_MS * 1_000_000
    book: list[tuple[int, float, float, float, float]] = []
    trades: list[tuple[int, float, float]] = []
    depth: list[tuple[int, float, float, float, float]] = []

    for key in _hour_keys(window_start_ns, window_end_ns):
        fh = _stream_file("bookTicker", symbol, key)
        if fh is not None:
            previous: tuple[int, float, float, float, float] | None = None
            with fh:
                for line in fh:
                    p = line.rstrip().split(",")
                    if len(p) < 8:
                        continue
                    try:
                        recv = int(p[0])
                        row = (recv, float(p[4]), float(p[5]),
                               float(p[6]), float(p[7]))
                    except ValueError:
                        continue
                    if recv < lower_ns:
                        previous = row
                        continue
                    if recv > window_end_ns:
                        break
                    book.append(row)
            if previous is not None and (not book or previous[0] < book[0][0]):
                book.insert(0, previous)

        fh = _stream_file("trade", symbol, key)
        if fh is not None:
            with fh:
                for line in fh:
                    p = line.rstrip().split(",")
                    if len(p) < 7:
                        continue
                    try:
                        recv = int(p[0])
                        qty = float(p[5])
                        signed = -qty if int(p[6]) else qty
                    except ValueError:
                        continue
                    if recv < lower_ns:
                        continue
                    if recv > window_end_ns:
                        break
                    trades.append((recv, signed, qty))

        fh = _stream_file("depth20", symbol, key)
        if fh is not None:
            previous_depth: tuple[int, float, float, float, float] | None = None
            with fh:
                for line in fh:
                    p = line.rstrip().split(",", 5)
                    if len(p) < 6:
                        continue
                    try:
                        recv = int(p[0])
                        b5, b20 = v1._depth_sums(p[4])
                        a5, a20 = v1._depth_sums(p[5])
                        row = (recv, b5, a5, b20, a20)
                    except ValueError:
                        continue
                    if recv < lower_ns:
                        previous_depth = row
                        continue
                    if recv > window_end_ns:
                        break
                    depth.append(row)
            if (previous_depth is not None
                    and (not depth or previous_depth[0] < depth[0][0])):
                depth.insert(0, previous_depth)

    _sorted_rows(book)
    _sorted_rows(trades)
    _sorted_rows(depth)
    return HFExactWindow(
        [int(x[0]) for x in book],
        np.asarray([x[1] for x in book], dtype=float),
        np.asarray([x[2] for x in book], dtype=float),
        np.asarray([x[3] for x in book], dtype=float),
        np.asarray([x[4] for x in book], dtype=float),
        [int(x[0]) for x in trades],
        np.asarray([x[1] for x in trades], dtype=float),
        np.asarray([x[2] for x in trades], dtype=float),
        [int(x[0]) for x in depth],
        np.asarray([x[1] for x in depth], dtype=float),
        np.asarray([x[2] for x in depth], dtype=float),
        np.asarray([x[3] for x in depth], dtype=float),
        np.asarray([x[4] for x in depth], dtype=float),
    )


def _last_at(times: Sequence[int], as_of_ns: int) -> int:
    return bisect.bisect_right(times, as_of_ns) - 1


def _imbalance(bid: float, ask: float) -> float:
    return (bid - ask) / (bid + ask) if bid + ask > 0 else 0.0


def hf_snapshot_at(data: HFExactWindow, as_of_ns: int
                   ) -> tuple[dict[str, Any] | None, str | None]:
    bi = _last_at(data.book_t, as_of_ns)
    di = _last_at(data.depth_t, as_of_ns)
    if bi < 0 or di < 0:
        return None, "HF_NO_CURRENT_BOOK_OR_DEPTH"
    book_stale = (as_of_ns - data.book_t[bi]) / 1e6
    depth_stale = (as_of_ns - data.depth_t[di]) / 1e6
    if book_stale > MAX_BOOK_STALENESS_MS:
        return None, "HF_BOOK_STALE"
    if depth_stale > MAX_DEPTH_STALENESS_MS:
        return None, "HF_DEPTH_STALE"

    mid = (data.bid[bi] + data.ask[bi]) / 2.0
    windows: dict[int, dict[str, float]] = {}
    for window in FAST_WINDOWS_MS:
        cutoff = as_of_ns - window * 1_000_000
        old_bi = _last_at(data.book_t, cutoff)
        if old_bi < 0:
            return None, "HF_INCOMPLETE_500MS_HISTORY"
        old_mid = (data.bid[old_bi] + data.ask[old_bi]) / 2.0
        book_lo = bisect.bisect_right(data.book_t, cutoff)
        trade_lo = bisect.bisect_right(data.trade_t, cutoff)
        trade_hi = bisect.bisect_right(data.trade_t, as_of_ns)
        signed = float(data.trade_signed_prefix[trade_hi]
                       - data.trade_signed_prefix[trade_lo])
        absolute = float(data.trade_abs_prefix[trade_hi]
                         - data.trade_abs_prefix[trade_lo])
        windows[window] = {
            "ret_bps": 10_000.0 * math.log(mid / old_mid),
            "book_updates": float(bi + 1 - book_lo),
            "trade_signed_qty": signed,
            "trade_abs_qty": absolute,
            "trade_count": float(trade_hi - trade_lo),
            "trade_imbalance": signed / absolute if absolute > 0 else 0.0,
        }

    old_di = _last_at(data.depth_t, as_of_ns - 500_000_000)
    if old_di < 0:
        return None, "HF_INCOMPLETE_DEPTH_HISTORY"
    book_imb = _imbalance(data.bid_qty[bi], data.ask_qty[bi])
    micro = (
        (data.ask[bi] * data.bid_qty[bi] + data.bid[bi] * data.ask_qty[bi])
        / (data.bid_qty[bi] + data.ask_qty[bi])
        if data.bid_qty[bi] + data.ask_qty[bi] > 0 else mid
    )
    return {
        "windows": windows,
        "mid": mid,
        "spread_bps": 10_000.0 * (data.ask[bi] - data.bid[bi]) / mid,
        "book_imbalance": book_imb,
        "microprice_offset_bps": 10_000.0 * (micro - mid) / mid,
        "depth5_imbalance": _imbalance(data.depth_bid5[di], data.depth_ask5[di]),
        "depth20_imbalance": _imbalance(data.depth_bid20[di], data.depth_ask20[di]),
        "depth_bid5_log_change": math.log(
            (data.depth_bid5[di] + 1e-9) / (data.depth_bid5[old_di] + 1e-9)),
        "depth_ask5_log_change": math.log(
            (data.depth_ask5[di] + 1e-9) / (data.depth_ask5[old_di] + 1e-9)),
        "book_staleness_ms": book_stale,
        "depth_staleness_ms": depth_stale,
        "known_max_ns": max(
            data.book_t[bi], data.depth_t[di],
            data.trade_t[bisect.bisect_right(data.trade_t, as_of_ns) - 1]
            if bisect.bisect_right(data.trade_t, as_of_ns) else 0,
        ),
    }, None


def decision_events(data: HFExactWindow, tape: v1.PMTape,
                    start_ns: int, stop_ns: int
                    ) -> list[tuple[int, int]]:
    """First exact event after each fixed cooldown, with source bits combined."""
    streams: list[Iterable[tuple[int, int]]] = [
        ((t, TRIGGER_HF_BOOK) for t in data.book_t if start_ns <= t <= stop_ns),
        ((t, TRIGGER_HF_TRADE) for t in data.trade_t if start_ns <= t <= stop_ns),
        ((tape.window_start * 1_000_000_000 + int(t * 1e9), TRIGGER_PM_BOOK)
         for t, state in zip(tape.state_t, tape.states)
         if state is not None and start_ns <= tape.window_start * 1_000_000_000
         + int(t * 1e9) <= stop_ns),
        ((tape.window_start * 1_000_000_000 + int(x.t * 1e9), TRIGGER_PM_TRADE)
         for x in tape.trades
         if start_ns <= tape.window_start * 1_000_000_000
         + int(x.t * 1e9) <= stop_ns),
    ]
    merged = heapq.merge(*streams, key=lambda item: item[0])
    combined: list[tuple[int, int]] = []
    current_t: int | None = None
    current_mask = 0
    for event_t, source in merged:
        if current_t is None or event_t != current_t:
            if current_t is not None:
                combined.append((current_t, current_mask))
            current_t, current_mask = event_t, source
        else:
            current_mask |= source
    if current_t is not None:
        combined.append((current_t, current_mask))

    selected: list[tuple[int, int]] = []
    last = -10**30
    for event_t, source in combined:
        if event_t - last >= COOLDOWN_NS:
            selected.append((event_t, source))
            last = event_t
    return selected


def label_action(tape: v1.PMTape, start: float, maker_side: str,
                 level: float, queue_ahead: float,
                 horizon_ms: int) -> FastActionLabel:
    end = start + horizon_ms / 1000.0
    if tape.touched(start, end):
        return FastActionLabel(
            "UNAVAILABLE", "PM_GAP_OR_TICK_IN_ACTION_HORIZON",
            0.0, 0, None, {}, {})
    lo = bisect.bisect_right(tape.trade_t, start)
    hi = bisect.bisect_right(tape.trade_t, end)
    cumulative = 0.0
    filled_before = 0.0
    tranches: list[tuple[float, float]] = []
    for trade in tape.trades[lo:hi]:
        if not fd.reaches_action(
                trade.taker_side, trade.exec_p_up, maker_side, level):
            continue
        cumulative += trade.size
        filled_now = min(ACTION_SIZE, max(0.0, cumulative - queue_ahead))
        delta = filled_now - filled_before
        if delta > 1e-12:
            tranches.append((trade.t, delta))
            filled_before = filled_now
        if filled_before >= ACTION_SIZE - 1e-12:
            break
    latencies = [latency for latency in LATENCY_MS if latency < horizon_ms]
    if not tranches:
        zeros = {latency: 0.0 for latency in latencies}
        return FastActionLabel(
            "AVAILABLE", None, 0.0, 0, None, dict(zeros), dict(zeros))
    if any(tape.touched(fill_t, fill_t + MARKOUT_HORIZON_S)
           for fill_t, _ in tranches):
        return FastActionLabel(
            "UNAVAILABLE", "PM_GAP_OR_TICK_IN_MARKOUT_HORIZON",
            0.0, 0, None, {}, {})

    marked: list[tuple[float, float, float]] = []
    for fill_t, size in tranches:
        later = tape.mark_state_at(fill_t + MARKOUT_HORIZON_S)
        if later is None:
            return FastActionLabel(
                "UNAVAILABLE", "NO_PM_MARKOUT_STATE", 0.0, 0, None, {}, {})
        markout = el.maker_sign(maker_side) * (later.mid - level) * 100.0
        marked.append((fill_t, size, markout))
    shares = sum(size for _, size, _ in marked)
    pnl = sum(size * markout for _, size, markout in marked)
    cancel_delta: dict[int, float] = {}
    prevented: dict[int, float] = {}
    for latency in latencies:
        effective = start + latency / 1000.0
        eligible = [(fill_t, size, markout)
                    for fill_t, size, markout in marked
                    if fill_t >= effective - 1e-12]
        prevented[latency] = sum(size for _, size, _ in eligible)
        cancel_delta[latency] = -sum(
            size * (markout + MAKER_REBATE_CENTS_PER_SHARE)
            for _, size, markout in eligible)
    mean_markout = pnl / shares
    return FastActionLabel(
        "AVAILABLE", None, shares, int(mean_markout < 0.0), mean_markout,
        cancel_delta, prevented)


def _feature_vector(state: v1.PMState, elapsed: float,
                    maker_side: str, trigger: int,
                    hf: dict[str, Any]) -> list[float]:
    sign = float(el.maker_sign(maker_side))
    mid = state.mid
    queue = state.bid_size if maker_side == "BUY_UP" else state.ask_size
    denom = state.bid * state.bid_size + state.ask * state.ask_size
    imbalance = ((state.bid * state.bid_size - state.ask * state.ask_size) / denom
                 if denom > 0 else 0.0)
    values = [
        sign,
        mid,
        math.log(min(max(mid, 1e-6), 1 - 1e-6)
                 / (1 - min(max(mid, 1e-6), 1 - 1e-6))),
        (state.ask - state.bid) * 100.0,
        (state.ask - state.bid) / state.tick,
        math.log1p(max(0.0, queue)),
        imbalance,
        sign * imbalance,
        abs(mid - 0.5),
        (fi.WINDOW_S - elapsed) / fi.WINDOW_S,
        max(0.0, (elapsed - state.t) * 1000.0),
        float(bool(trigger & TRIGGER_HF_BOOK)),
        float(bool(trigger & TRIGGER_HF_TRADE)),
        float(bool(trigger & TRIGGER_PM_BOOK)),
        float(bool(trigger & TRIGGER_PM_TRADE)),
    ]
    for window in FAST_WINDOWS_MS:
        item = hf["windows"][window]
        values.extend((
            sign * item["ret_bps"],
            item["book_updates"],
            sign * item["trade_signed_qty"],
            item["trade_abs_qty"],
            item["trade_count"],
            sign * item["trade_imbalance"],
        ))
    values.extend((
        hf["spread_bps"],
        hf["book_imbalance"],
        sign * hf["book_imbalance"],
        hf["microprice_offset_bps"],
        sign * hf["microprice_offset_bps"],
        hf["depth5_imbalance"],
        sign * hf["depth5_imbalance"],
        hf["depth20_imbalance"],
        sign * hf["depth20_imbalance"],
        hf["depth_bid5_log_change"] if maker_side == "BUY_UP"
        else hf["depth_ask5_log_change"],
        hf["book_staleness_ms"],
        hf["depth_staleness_ms"],
    ))
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("fast feature order/schema mismatch")
    return values


def materialize_window(tape: v1.PMTape, data: HFExactWindow
                       ) -> FastWindowBatch:
    window_ns = tape.window_start * 1_000_000_000
    start_ns = window_ns + START_HISTORY_MS * 1_000_000
    stop_ns = window_ns + int(
        (fi.WINDOW_S - max(PREDICTION_HORIZONS_MS) / 1000.0
         - MARKOUT_HORIZON_S) * 1e9)
    decisions = decision_events(data, tape, start_ns, stop_ns)
    capacity = len(decisions) * 2
    x = np.empty((capacity, len(FEATURE_NAMES)), dtype=np.float32)
    as_of = np.empty(capacity, dtype=np.int64)
    side_sign = np.empty(capacity, dtype=np.int8)
    level = np.empty(capacity, dtype=np.float32)
    queue = np.empty(capacity, dtype=np.float32)
    triggers = np.empty(capacity, dtype=np.uint8)
    toxic = {h: np.full(capacity, -1, dtype=np.int8)
             for h in PREDICTION_HORIZONS_MS}
    filled = {h: np.full(capacity, np.nan, dtype=np.float32)
              for h in PREDICTION_HORIZONS_MS}
    cancel = {
        (h, latency): np.full(capacity, np.nan, dtype=np.float32)
        for h in PREDICTION_HORIZONS_MS
        for latency in LATENCY_MS if latency < h
    }
    prevented = {
        key: np.full(capacity, np.nan, dtype=np.float32)
        for key in cancel
    }
    diagnostics: collections.Counter[str] = collections.Counter()
    n = 0
    for event_ns, trigger in decisions:
        elapsed = (event_ns - window_ns) / 1e9
        state = tape.state_at(elapsed)
        if state is None:
            diagnostics["PM_NO_CURRENT_STATE"] += 2
            continue
        pm_stale = elapsed * 1000.0 - state.t * 1000.0
        if pm_stale > MAX_PM_STALENESS_MS:
            diagnostics["PM_STATE_STALE"] += 2
            continue
        hf, reason = hf_snapshot_at(data, event_ns)
        if hf is None:
            diagnostics[str(reason)] += 2
            continue
        for maker_side, action_level, queue_ahead in (
                ("BUY_UP", state.bid, state.bid_size),
                ("SELL_UP", state.ask, state.ask_size)):
            vector = _feature_vector(
                state, elapsed, maker_side, trigger, hf)
            if not all(math.isfinite(value) for value in vector):
                diagnostics["NONFINITE_FEATURE"] += 1
                continue
            x[n] = vector
            as_of[n] = event_ns
            side_sign[n] = el.maker_sign(maker_side)
            level[n] = action_level
            queue[n] = queue_ahead
            triggers[n] = trigger
            for horizon in PREDICTION_HORIZONS_MS:
                label = label_action(
                    tape, elapsed, maker_side, action_level, queue_ahead, horizon)
                if label.status != "AVAILABLE":
                    diagnostics[f"LABEL_{horizon}MS_{label.unavailable_reason}"] += 1
                    continue
                toxic[horizon][n] = label.toxic_fill
                filled[horizon][n] = label.filled_shares
                for latency, value in label.cancel_delta_cents.items():
                    cancel[(horizon, latency)][n] = value
                    prevented[(horizon, latency)][n] = label.prevented_shares[latency]
            n += 1
    return FastWindowBatch(
        tape.slug,
        tape.coin,
        fi.slug_day(tape.slug),
        x[:n].copy(),
        as_of[:n].copy(),
        side_sign[:n].copy(),
        level[:n].copy(),
        queue[:n].copy(),
        triggers[:n].copy(),
        {h: values[:n].copy() for h, values in toxic.items()},
        {key: values[:n].copy() for key, values in cancel.items()},
        {key: values[:n].copy() for key, values in prevented.items()},
        {h: values[:n].copy() for h, values in filled.items()},
        dict(diagnostics),
    )


def load_and_materialize(path: Path, up_id: str, down_id: str,
                         gaps: Sequence[tuple[float, float]]) -> FastWindowBatch:
    tape = v1.build_pm_tape(
        path, up_id, down_id, gaps, feature_state_lag_s=0.0)
    window_ns = tape.window_start * 1_000_000_000
    symbol = {"btc": "BTCUSDT", "eth": "ETHUSDT"}.get(tape.coin)
    if symbol is None:
        raise ValueError(f"unsupported fast-path coin {tape.coin}")
    print(f"[adverse-fast-feature] load {tape.slug}", flush=True)
    data = load_hf_window(
        symbol, window_ns, window_ns + int(fi.WINDOW_S * 1e9))
    print(
        f"[adverse-fast-feature] exact events book={len(data.book_t):,} "
        f"trade={len(data.trade_t):,} depth={len(data.depth_t):,}",
        flush=True,
    )
    batch = materialize_window(tape, data)
    print(
        f"[adverse-fast-feature] {tape.slug} rows={batch.n_rows:,} "
        f"refused={sum(batch.diagnostics.values()):,}",
        flush=True,
    )
    return batch


def selftest() -> int:
    checks = 0

    def ok(condition: bool, name: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(name)
        checks += 1

    ok(len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "feature names unique")
    ok(_last_at([10, 20], 20) == 1 and _last_at([10, 20], 19) == 0,
       "exact as-of includes current receipt but excludes future")

    synthetic = HFExactWindow(
        [0, 10_000_000, 20_000_000, 500_000_000],
        np.asarray([100.0, 100.1, 100.2, 101.0]),
        np.ones(4), np.asarray([100.2, 100.3, 100.4, 101.2]), np.ones(4),
        [12_000_000, 20_000_000], np.asarray([2.0, -1.0]),
        np.asarray([2.0, 1.0]),
        [0, 500_000_000], np.asarray([10.0, 8.0]), np.asarray([10.0, 12.0]),
        np.asarray([20.0, 18.0]), np.asarray([20.0, 22.0]),
    )
    snap, reason = hf_snapshot_at(synthetic, 500_000_000)
    ok(reason is None and snap is not None, "exact synthetic snapshot available")
    assert snap is not None
    ok(snap["windows"][500]["trade_count"] == 2.0,
       "rolling window uses exact receipt boundary")

    empty_tape = v1.PMTape("btc-updown-5m-0", "btc", 0, [0.0],
                           [v1.PMState(0.0, .49, .51, 1, 1, .01)], [], [], [])
    selected = decision_events(
        HFExactWindow(
            [0, 5_000_000, 10_000_000, 11_000_000, 21_000_000],
            np.ones(5), np.ones(5), np.ones(5) * 2, np.ones(5),
            [], np.asarray([]), np.asarray([]),
            [0], np.ones(1), np.ones(1), np.ones(1), np.ones(1)),
        empty_tape, 0, 30_000_000)
    ok([t for t, _ in selected] == [0, 10_000_000, 21_000_000],
       "10ms event cooldown is deterministic")

    states = [v1.PMState(0.0, .49, .51, 0, 10, .01),
              v1.PMState(5.01, .46, .48, 10, 10, .01)]
    tape = v1.PMTape(
        "btc-updown-5m-1780000000", "btc", 1780000000,
        [0.0, 5.01], states,
        [v1.PMTrade(.020, "SELL", .49, 5.0)], [], [])
    label = label_action(tape, 0.0, "BUY_UP", .49, 0.0, 50)
    ok(label.status == "AVAILABLE" and label.toxic_fill == 1,
       "50ms toxic-fill label")
    ok(label.prevented_shares[10] == 5.0 and label.prevented_shares[25] == 0.0,
       "10ms cancellation can prevent a 20ms fill but 25ms cannot")
    ok(label.cancel_delta_cents[10] > 0.0,
       "fast cancellation target includes adverse markout and rebate")
    ok(all(latency < 50 for latency in label.cancel_delta_cents),
       "only latency rungs below the fill horizon are labelled")
    ok(SOURCE_PROFILE["polymarket_feature_state_lag_ms"] == 0,
       "fast candidate has no artificial PM lag")
    ok(len(FEATURE_NAMES) == 63, "fast feature count pinned")
    print(f"[adverse-fast-feature] selftest OK — {checks} checks")
    return 0


if __name__ == "__main__":
    selftest()
