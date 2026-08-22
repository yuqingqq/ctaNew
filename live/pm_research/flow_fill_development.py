"""Development-only flow baseline, mark, Hawkes, and queue-fill diagnostics.

This is the executable DEVELOPMENT lane of BE_FLOWANDFILLS_MODEL_PLAN Revision 4.
It is research code, consumes only the recorded tape, and has no venue/order API.

The probe deliberately separates two questions:

* Can the estimator and action mapping be built and tested on hours of data? YES.
* Has any fit generalized across independent days? NO; that is a later gate.

Commands:

    python3 live/pm_research/flow_fill_development.py --selftest
    python3 live/pm_research/flow_fill_development.py run --per-coin 24
"""

from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import hashlib
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

import flow_intensity as fi

PM = fi.PM
OUT = PM / "derived/flow_fill_development_v1.json"
PROTOCOL = Path(__file__).with_name("FLOW_MODEL_PROTOCOL_V4.yaml")

R_EDGES = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)  # elapsed; inverse of r
P_EDGES = fi.FP_EDGES
PARENT_SHRINK_S = 60.0
ACTION_TIMES = (30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0)
ACTION_HORIZONS = (5.0, 15.0, 30.0)
ACTION_SIZE = 5.0
STATE_LAG_S = fi.QUOTE_STATE_LAG_S
MICRO_SIZE = fi.MICRO_SIZE

BOOK_MARK = b'"event_type":"book"'
TICK_MARK = b'"event_type":"tick_size_change"'

Cell = tuple[int, int]


def r_bin(elapsed: float) -> int:
    if not (0.0 <= elapsed <= fi.WINDOW_S):
        raise ValueError(f"elapsed {elapsed} outside window")
    return min(int(elapsed // 60.0), len(R_EDGES) - 2)


def cell_at(elapsed: float, mid: float) -> Cell:
    return r_bin(elapsed), fi.p_bin(mid)


def action_fill(cumulative: float, size: float, queue_ahead: float) -> tuple[float, float]:
    """Return optimistic-front and conservative-back filled shares."""
    if min(cumulative, size, queue_ahead) < 0:
        raise ValueError("fill inputs must be non-negative")
    front = min(size, cumulative)
    back = min(size, max(0.0, cumulative - queue_ahead))
    return front, back


def reaches_action(taker_side: str, exec_p_up: float,
                   maker_side: str, level_up: float) -> bool:
    """Whether a complement-folded aggressive trade reaches a maker action."""
    if maker_side == "SELL_UP":
        return taker_side == "BUY" and exec_p_up + 1e-12 >= level_up
    if maker_side == "BUY_UP":
        return taker_side == "SELL" and exec_p_up <= level_up + 1e-12
    raise ValueError(f"unknown maker side {maker_side}")


@dataclass
class BookState:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    best_bid: float | None = None
    best_ask: float | None = None
    tick: float | None = None
    ready: bool = False

    @staticmethod
    def key(price: float) -> float:
        return round(float(price), 9)

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.best_bid = self.best_ask = self.tick = None
        self.ready = False

    def apply(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "book":
            self.bids = {self.key(p): float(s) for p, s in data["bids"] if float(s) > 0}
            self.asks = {self.key(p): float(s) for p, s in data["asks"] if float(s) > 0}
            self.best_bid = max(self.bids, default=None)
            self.best_ask = min(self.asks, default=None)
            self.tick = data.get("tick") or self.tick
            self.ready = self.best_bid is not None and self.best_ask is not None
            return
        if kind == "price":
            if not self.ready:
                return  # require a full post-gap snapshot before queue inference
            side = data["side"]
            book = self.bids if side == "BUY" else self.asks
            px, sz = self.key(data["price"]), float(data["size"])
            if sz > 0:
                book[px] = sz
            else:
                book.pop(px, None)
            self.best_bid = self.key(data["best_bid"])
            self.best_ask = self.key(data["best_ask"])
            self.ready = self.best_bid < self.best_ask
            return
        if kind == "tick":
            self.tick = float(data["tick"])
            return
        raise ValueError(f"unknown mutation {kind}")

    def quote(self) -> tuple[float, float, float, float, float] | None:
        if not self.ready or self.best_bid is None or self.best_ask is None:
            return None
        bid_size = self.bids.get(self.key(self.best_bid))
        ask_size = self.asks.get(self.key(self.best_ask))
        if bid_size is None or ask_size is None or bid_size < 0 or ask_size < 0:
            return None
        tick = self.tick or 0.01
        return self.best_bid, self.best_ask, bid_size, ask_size, tick


@dataclass
class DevTrade:
    elapsed: float
    cell: Cell
    event_type: str
    side: str
    exec_p_up: float
    size: float
    notional: float
    reach_class: str
    distance_ticks: float | None
    tick_tail: float
    book_x: tuple[float, float, float]


@dataclass
class ExposurePiece:
    start: float
    end: float
    cell: Cell
    tick_tail: float
    book_x: tuple[float, float, float]


@dataclass
class ShadowAction:
    start: float
    horizon: float
    maker_side: str
    level: float
    size: float
    queue_ahead: float
    cumulative_reaching: float = 0.0
    first_front: float | None = None
    first_back: float | None = None
    unavailable_reason: str | None = None
    touch_departed: bool = False

    @property
    def end(self) -> float:
        return min(fi.WINDOW_S, self.start + self.horizon)

    def observe(self, elapsed: float, taker_side: str, exec_p_up: float,
                size: float) -> None:
        if self.unavailable_reason is not None:
            return
        if not (self.start < elapsed <= self.end):
            return
        if not reaches_action(taker_side, exec_p_up, self.maker_side, self.level):
            return
        before = self.cumulative_reaching
        self.cumulative_reaching += size
        if self.first_front is None and self.cumulative_reaching > 0:
            self.first_front = elapsed
        if self.first_back is None and before <= self.queue_ahead < self.cumulative_reaching:
            self.first_back = elapsed

    def invalidate(self, elapsed: float, reason: str) -> None:
        if self.start < elapsed <= self.end and self.unavailable_reason is None:
            self.unavailable_reason = reason

    def note_touch(self, elapsed: float, bid: float, ask: float) -> None:
        if not (self.start < elapsed <= self.end):
            return
        touch = bid if self.maker_side == "BUY_UP" else ask
        if abs(touch - self.level) > 1e-12:
            self.touch_departed = True

    def result(self) -> dict[str, Any]:
        if self.unavailable_reason is not None:
            return {
                "start": self.start,
                "horizon": self.horizon,
                "maker_side": self.maker_side,
                "level_up": self.level,
                "size_shares": self.size,
                "queue_ahead": self.queue_ahead,
                "status": "UNAVAILABLE",
                "unavailable_reason": self.unavailable_reason,
                "touch_departed": self.touch_departed,
                "cumulative_reaching": None,
                "front_fill": None,
                "back_fill": None,
                "front_first_fill_s": None,
                "back_first_fill_s": None,
            }
        front, back = action_fill(self.cumulative_reaching, self.size, self.queue_ahead)
        return {
            "start": self.start,
            "horizon": self.horizon,
            "maker_side": self.maker_side,
            "level_up": self.level,
            "size_shares": self.size,
            "queue_ahead": self.queue_ahead,
            "status": "AVAILABLE",
            "unavailable_reason": None,
            "touch_departed": self.touch_departed,
            "cumulative_reaching": self.cumulative_reaching,
            "front_fill": front,
            "back_fill": back,
            "front_first_fill_s": None if self.first_front is None else self.first_front - self.start,
            "back_first_fill_s": None if self.first_back is None else self.first_back - self.start,
        }


@dataclass
class DevWindow:
    slug: str
    coin: str
    exposure: dict[Cell, float]
    counts: dict[Cell, int]
    pieces: list[ExposurePiece]
    trades: list[DevTrade]
    actions: list[dict[str, Any]]
    diagnostics: dict[str, int]


def _parse_book(msg: dict[str, Any]) -> dict[str, Any] | None:
    try:
        bids = [(float(x["price"]), float(x["size"])) for x in msg.get("bids", [])]
        asks = [(float(x["price"]), float(x["size"])) for x in msg.get("asks", [])]
        tick = float(msg["tick_size"]) if msg.get("tick_size") else None
    except (KeyError, TypeError, ValueError):
        return None
    if not bids or not asks:
        return None
    return {"bids": bids, "asks": asks, "tick": tick}


def build_window(path: Path, up_id: str, down_id: str,
                 gaps: Sequence[tuple[float, float]]) -> DevWindow:
    slug = path.name.split(".jsonl")[0]
    try:
        ws = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"bad slug {slug}") from exc

    state = BookState()
    exposure: collections.defaultdict[Cell, float] = collections.defaultdict(float)
    counts: collections.defaultdict[Cell, int] = collections.defaultdict(int)
    pieces: list[ExposurePiece] = []
    trades: list[DevTrade] = []
    actions: list[ShadowAction] = []
    diag: collections.Counter[str] = collections.Counter()

    pending: list[tuple[float, int, str, dict[str, Any]]] = []
    seq = 0
    clock = -60.0
    gap_starts = sorted(g0 for g0, _ in gaps if 0.0 <= g0 <= fi.WINDOW_S)
    gap_i = 0
    action_i = 0
    boundary_times = list(R_EDGES)
    boundary_i = 0
    seen_tx: set[str] = set()

    def current_state(at: float) -> tuple[Cell, float, tuple[float, float, float]] | None:
        q = state.quote()
        if q is None or not (0.0 <= at < fi.WINDOW_S):
            return None
        bid, ask, bid_size, ask_size, tick = q
        mid = (bid + ask) / 2.0
        bid_notional, ask_notional = bid * bid_size, ask * ask_size
        total_touch = bid_notional + ask_notional
        imbalance = ((bid_notional - ask_notional) / total_touch
                     if total_touch > 0 else 0.0)
        raw_book = (
            math.log1p(max(0.0, total_touch)),
            min(10.0, max(0.0, (ask - bid) / tick)),
            min(1.0, max(-1.0, imbalance)),
        )
        in_tail = mid < 0.15 or mid >= 0.85
        tick_tail = float(abs(tick - 0.001) < 1e-9 and in_tail)
        return cell_at(at, mid), tick_tail, raw_book

    def current_cell(at: float) -> Cell | None:
        current = current_state(at)
        return None if current is None else current[0]

    def accrue(to: float) -> None:
        nonlocal clock
        lo, hi = max(clock, 0.0), min(to, fi.WINDOW_S)
        if hi > lo:
            current = current_state((lo + hi) / 2.0)
            if current is not None:
                cell, tick_tail, book_x = current
                exposure[cell] += hi - lo
                if (pieces and pieces[-1].cell == cell
                        and pieces[-1].tick_tail == tick_tail
                        and pieces[-1].book_x == book_x
                        and abs(pieces[-1].end - lo) < 1e-9):
                    pieces[-1].end = hi
                else:
                    pieces.append(ExposurePiece(lo, hi, cell, tick_tail, book_x))
        clock = max(clock, to)

    def make_actions(at: float) -> None:
        q = state.quote()
        planned = len(ACTION_HORIZONS) * 2
        if q is None:
            diag["actions_no_state"] += planned
            return
        bid, ask, bid_size, ask_size, _ = q
        for horizon in ACTION_HORIZONS:
            actions.append(ShadowAction(at, horizon, "BUY_UP", bid, ACTION_SIZE, bid_size))
            actions.append(ShadowAction(at, horizon, "SELL_UP", ask, ACTION_SIZE, ask_size))

    def advance(to: float) -> None:
        nonlocal clock, gap_i, action_i, boundary_i, pending
        while True:
            candidates: list[float] = []
            if pending:
                candidates.append(pending[0][0])
            if gap_i < len(gap_starts):
                candidates.append(gap_starts[gap_i])
            if action_i < len(ACTION_TIMES):
                candidates.append(ACTION_TIMES[action_i])
            if boundary_i < len(boundary_times):
                candidates.append(boundary_times[boundary_i])
            if not candidates or min(candidates) > to + 1e-12:
                break
            when = min(candidates)
            accrue(when)

            if gap_i < len(gap_starts) and abs(gap_starts[gap_i] - when) < 1e-12:
                for action in actions:
                    action.invalidate(when, "COLLECTOR_GAP")
                state.clear()
                pending.clear()  # pre-gap quotes cannot mature after reconnect
                heapq.heapify(pending)
                diag["gap_state_resets"] += 1
                gap_i += 1
                while gap_i < len(gap_starts) and abs(gap_starts[gap_i] - when) < 1e-12:
                    gap_i += 1

            while pending and pending[0][0] <= when + 1e-12:
                _, _, kind, data = heapq.heappop(pending)
                if kind == "tick":
                    for action in actions:
                        action.invalidate(when, "TICK_SIZE_CHANGE")
                state.apply(kind, data)
                q = state.quote()
                if kind in ("book", "price") and q is not None:
                    for action in actions:
                        action.note_touch(when, q[0], q[1])

            while boundary_i < len(boundary_times) and abs(boundary_times[boundary_i] - when) < 1e-12:
                boundary_i += 1

            while action_i < len(ACTION_TIMES) and abs(ACTION_TIMES[action_i] - when) < 1e-12:
                make_actions(when)
                action_i += 1
        accrue(to)

    def schedule(received: float, kind: str, data: dict[str, Any]) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(pending, (received + STATE_LAG_S, seq, kind, data))

    for line in fi._gz_lines(path):
        if not any(mark in line for mark in
                   (fi.TRADE_MARK, fi.QUOTE_MARK, BOOK_MARK, TICK_MARK)):
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            recv = int(parts[0]) / 1e9 - ws
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            diag["malformed"] += 1
            continue
        if recv < -60.0 or recv > fi.WINDOW_S:
            continue
        advance(recv)

        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict):
                continue
            et = msg.get("event_type")
            aid = str(msg.get("asset_id"))
            if (et == "book" or ("bids" in msg and "asks" in msg)) and aid == up_id:
                data = _parse_book(msg)
                if data:
                    schedule(recv, "book", data)
                continue
            if et == "price_change":
                for pc in msg.get("price_changes", []):
                    if str(pc.get("asset_id")) != up_id:
                        continue
                    try:
                        data = {
                            "side": str(pc["side"]).upper(),
                            "price": float(pc["price"]),
                            "size": float(pc["size"]),
                            "best_bid": float(pc["best_bid"]),
                            "best_ask": float(pc["best_ask"]),
                        }
                    except (KeyError, TypeError, ValueError):
                        diag["bad_price_change"] += 1
                        continue
                    if 0.0 <= data["best_bid"] < data["best_ask"] <= 1.0:
                        schedule(recv, "price", data)
                continue
            if et == "tick_size_change" and aid == up_id:
                try:
                    schedule(recv, "tick", {"tick": float(msg["new_tick_size"])})
                except (KeyError, TypeError, ValueError):
                    diag["bad_tick"] += 1
                continue
            if et != "last_trade_price" or aid not in (up_id, down_id):
                continue

            tx = str(msg.get("transaction_hash") or "")
            if tx and tx in seen_tx:
                diag["duplicate_transaction"] += 1
                continue
            if tx:
                seen_tx.add(tx)
            try:
                native_px = float(msg["price"])
                size = float(msg["size"])
                native_side = str(msg["side"]).upper()
            except (KeyError, TypeError, ValueError):
                diag["bad_trade"] += 1
                continue
            is_down = aid == down_id
            exec_p_up = fi.fold_price(native_px, is_down)
            taker_side = fi.fold_side(native_side, is_down)
            q = state.quote()
            cell = current_cell(recv)
            if q is None or cell is None:
                diag["trades_no_state"] += 1
                continue
            bid, ask, _, _, tick = q
            current = current_state(recv)
            if current is None:
                diag["trades_no_state"] += 1
                continue
            _, tick_tail, book_x = current
            counts[cell] += 1
            if taker_side == "BUY" and exec_p_up + 1e-12 >= ask:
                reach_class = "AT_OR_THROUGH_ASK"
                dist = max(0.0, (exec_p_up - ask) / tick)
            elif taker_side == "SELL" and exec_p_up <= bid + 1e-12:
                reach_class = "AT_OR_THROUGH_BID"
                dist = max(0.0, (bid - exec_p_up) / tick)
            else:
                reach_class = "NOT_MARKETABLE_AT_LAGGED_STATE"
                dist = None
            trades.append(DevTrade(
                recv, cell, "MICRO_002" if abs(size - MICRO_SIZE) < fi.MICRO_TOL else "MARKET",
                taker_side, exec_p_up, size, size * native_px, reach_class, dist,
                tick_tail, book_x,
            ))
            for action in actions:
                action.observe(recv, taker_side, exec_p_up, size)

    advance(fi.WINDOW_S)
    return DevWindow(
        slug=slug,
        coin=slug.split("-")[0],
        exposure=dict(exposure),
        counts=dict(counts),
        pieces=pieces,
        trades=trades,
        actions=[a.result() for a in actions],
        diagnostics=dict(diag),
    )


def fit_rates(windows: Sequence[DevWindow]) -> tuple[dict[int, float], dict[Cell, float]]:
    """Frozen B0/B1 development estimator."""
    expo_r: collections.defaultdict[int, float] = collections.defaultdict(float)
    count_r: collections.defaultdict[int, int] = collections.defaultdict(int)
    expo_c: collections.defaultdict[Cell, float] = collections.defaultdict(float)
    count_c: collections.defaultdict[Cell, int] = collections.defaultdict(int)
    for w in windows:
        for cell, e in w.exposure.items():
            expo_c[cell] += e
            expo_r[cell[0]] += e
        for cell, n in w.counts.items():
            count_c[cell] += n
            count_r[cell[0]] += n
    b0 = {r: (count_r[r] + 0.5) / (expo_r[r] + 1.0)
          for r in range(len(R_EDGES) - 1)}
    b1 = {(r, p): (count_c[(r, p)] + PARENT_SHRINK_S * b0[r]) /
                   (expo_c[(r, p)] + PARENT_SHRINK_S)
          for r in range(len(R_EDGES) - 1)
          for p in range(len(P_EDGES) - 1)}
    return b0, b1


@dataclass
class BaselineFit:
    b0: dict[int, float]
    b1: dict[Cell, float]
    b2_tick_tail_beta: float
    book_mean: tuple[float, float, float]
    book_scale: tuple[float, float, float]
    b3_gamma: tuple[float, float, float]
    b3_optimizer_success: bool
    b3_gradient_norm: float
    b2_zero_event_fence: bool

    def standardized(self, x: tuple[float, float, float]) -> np.ndarray:
        return (np.asarray(x, dtype=float) - np.asarray(self.book_mean)) / np.asarray(self.book_scale)

    def rate(self, cell: Cell, tick_tail: float,
             book_x: tuple[float, float, float], layer: str) -> float:
        if layer == "B0":
            return self.b0[cell[0]]
        rate = self.b1[cell]
        if layer == "B1":
            return rate
        rate *= math.exp(self.b2_tick_tail_beta * tick_tail)
        if layer == "B2":
            return rate
        if layer != "B3":
            raise ValueError(f"unknown baseline layer {layer}")
        log_book = float(np.dot(np.asarray(self.b3_gamma), self.standardized(book_x)))
        return rate * math.exp(min(30.0, max(-30.0, log_book)))


def fit_baseline(windows: Sequence[DevWindow]) -> BaselineFit:
    """Fit the frozen nested B0--B3 candidate on development windows."""
    b0, b1 = fit_rates(windows)
    b2_count = sum(t.tick_tail for w in windows for t in w.trades)
    b2_expected = sum(
        (piece.end - piece.start) * b1[piece.cell] * piece.tick_tail
        for w in windows for piece in w.pieces
    )
    b2_zero = b2_count <= 0 or b2_expected <= 0
    # The half-event is a named numerical fence for a zero-count DEVELOPMENT
    # cell. Such a fit is never promotable without forward support.
    b2_beta = math.log((b2_count if b2_count > 0 else 0.5) /
                       (b2_expected if b2_expected > 0 else 0.5))

    pieces = [piece for w in windows for piece in w.pieces]
    total_exposure = sum(piece.end - piece.start for piece in pieces)
    if total_exposure <= 0:
        raise ValueError("baseline has no knowledge-admissible exposure")
    raw = np.asarray([piece.book_x for piece in pieces], dtype=float)
    weights = np.asarray([piece.end - piece.start for piece in pieces], dtype=float)
    mean = np.average(raw, axis=0, weights=weights)
    variance = np.average((raw - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-12))

    event_x = np.asarray([
        (np.asarray(t.book_x) - mean) / scale
        for w in windows for t in w.trades
    ], dtype=float)
    piece_x = (raw - mean) / scale
    offsets = np.asarray([
        b1[piece.cell] * math.exp(b2_beta * piece.tick_tail)
        for piece in pieces
    ], dtype=float)

    def objective(gamma: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(piece_x @ gamma, -30.0, 30.0)
        expected = weights * offsets * np.exp(eta)
        value = float(expected.sum())
        gradient = expected @ piece_x
        if len(event_x):
            value -= float((event_x @ gamma).sum())
            gradient -= event_x.sum(axis=0)
        return value, np.asarray(gradient)

    solved = minimize(
        lambda gamma: objective(gamma)[0],
        np.zeros(3),
        jac=lambda gamma: objective(gamma)[1],
        method="BFGS",
        options={"gtol": 1e-5, "maxiter": 500},
    )
    gamma = solved.x if np.all(np.isfinite(solved.x)) else np.zeros(3)
    return BaselineFit(
        b0=b0,
        b1=b1,
        b2_tick_tail_beta=float(b2_beta),
        book_mean=tuple(float(x) for x in mean),
        book_scale=tuple(float(x) for x in scale),
        b3_gamma=tuple(float(x) for x in gamma),
        b3_optimizer_success=bool(solved.success),
        b3_gradient_norm=float(np.linalg.norm(objective(gamma)[1])),
        b2_zero_event_fence=b2_zero,
    )


def poisson_nll(window: DevWindow, fit: BaselineFit, layer: str) -> float:
    """Point-process NLL without constants, evaluated on exact state pieces."""
    integral = sum(
        fit.rate(piece.cell, piece.tick_tail, piece.book_x, layer)
        * (piece.end - piece.start)
        for piece in window.pieces
    )
    event_log_rate = sum(
        math.log(max(fit.rate(t.cell, t.tick_tail, t.book_x, layer), 1e-300))
        for t in window.trades
    )
    return integral - event_log_rate


def operational_window(window: DevWindow, fit: BaselineFit) -> tuple[list[float], float]:
    """Map admitted arrivals into full-B3 operational time for one risk path."""
    events = sorted(window.trades, key=lambda x: x.elapsed)
    i, u = 0, 0.0
    out: list[float] = []
    for piece in window.pieces:
        rate = fit.rate(piece.cell, piece.tick_tail, piece.book_x, "B3")
        while i < len(events) and events[i].elapsed < piece.start - 1e-12:
            i += 1
        j = i
        while j < len(events) and events[j].elapsed < piece.end - 1e-12:
            out.append(u + rate * max(0.0, events[j].elapsed - piece.start))
            j += 1
        i = j
        u += rate * (piece.end - piece.start)
    return out, u


def hawkes_loglik(paths: Sequence[tuple[Sequence[float], float]],
                  branching: float, beta: float) -> float:
    if not (0.0 <= branching < 1.0) or beta <= 0:
        return -math.inf
    ll = 0.0
    for events, end in paths:
        history, prev = 0.0, 0.0
        for event in events:
            history *= math.exp(-beta * max(0.0, event - prev))
            intensity = (1.0 - branching) + branching * beta * history
            if intensity <= 0:
                return -math.inf
            ll += math.log(intensity)
            history += 1.0
            prev = event
        integral = (1.0 - branching) * end
        integral += sum(branching * (1.0 - math.exp(-beta * max(0.0, end - event)))
                        for event in events)
        ll -= integral
    return ll


# The venue timestamps in MILLISECONDS (13-digit payload `timestamp`), so no
# structure below 1 ms exists in the data at all. Our own `recv_ns` is stamped at
# PARSE time, which manufactures a 0-50 us pile-up (26 us median, 16.2x Poisson
# on btc) carrying processing cadence rather than arrival time. A kernel whose
# half-life sits near venue resolution is fitting mostly unresolvable structure,
# so the grid is floored at TEN venue ticks of wall clock.
VENUE_RESOLUTION_S = 0.001
HALF_LIFE_FLOOR_TICKS = 10.0


def hawkes_floor_operational(paths: Sequence[tuple[Sequence[float], float]],
                             wall_seconds: float) -> float:
    """Operational-time floor equal to HALF_LIFE_FLOOR_TICKS of wall clock.

    Operational time accrues at the baseline rate (`du = lam*dt`), so a wall
    duration maps to `lam * dt` operational units. The floor is a single
    instrument limit in SECONDS but is COIN-DEPENDENT in operational units.
    """
    if wall_seconds <= 0:
        return 0.0
    lam = sum(end for _, end in paths) / wall_seconds
    return lam * HALF_LIFE_FLOOR_TICKS * VENUE_RESOLUTION_S


def fit_hawkes(paths: Sequence[tuple[Sequence[float], float]],
               wall_seconds: float | None = None) -> dict[str, Any]:
    """Grid-seed, then REFINE CONTINUOUSLY off the grid.

    A pure grid gives quantised estimates whose bootstrap draws land only on grid
    points -- btc's degenerate [0.180, 0.180] was that artefact, and such an
    interval shows fit stability, not sampling uncertainty. The grid now only
    seeds a continuous optimiser.
    """
    grid = (0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    floor = hawkes_floor_operational(paths, wall_seconds) if wall_seconds else 0.0
    half_lives = tuple(h for h in grid if h >= floor) or (grid[-1],)
    excluded = [h for h in grid if h < floor]

    branching_grid = tuple(i / 20 for i in range(19))
    baseline_ll = hawkes_loglik(paths, 0.0, 1.0)
    best = (baseline_ll, 0.0, half_lives[0])
    for half_life in half_lives:
        beta = math.log(2.0) / half_life
        for branching in branching_grid:
            ll = hawkes_loglik(paths, branching, beta)
            if ll > best[0]:
                best = (ll, branching, half_life)

    lo_h, hi_h = half_lives[0], half_lives[-1]

    def neg_ll(theta) -> float:
        b = 1.0 / (1.0 + math.exp(-float(theta[0])))
        h = math.exp(float(theta[1]))
        if not (lo_h <= h <= hi_h):
            return 1e12
        v = hawkes_loglik(paths, b, math.log(2.0) / h)
        return 1e12 if not math.isfinite(v) else -v

    refined = None
    if best[1] > 0.0:
        b0 = min(max(best[1], 1e-3), 0.95)
        theta0 = np.array([math.log(b0 / (1 - b0)), math.log(best[2])])
        try:
            r = minimize(neg_ll, theta0, method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400})
            # Accept on LIKELIHOOD, never on scipy's `success` flag. Nelder-Mead
            # routinely hits maxiter with the improvement already found -- measured
            # on the bivariate fit, nit=80 success=False and nit=94 success=True
            # returned the IDENTICAL +4.1258 gain. Gating on the flag rejected
            # every refinement there. The likelihood comparison is the real guard;
            # the flag is a convergence report, not a correctness test.
            if -float(r.fun) >= best[0] - 1e-9:
                rb = 1.0 / (1.0 + math.exp(-float(r.x[0])))
                refined = (-float(r.fun), rb, math.exp(float(r.x[1])))
        except Exception:  # noqa: BLE001
            refined = None

    b_hat = refined[1] if refined else best[1]
    h_hat = refined[2] if refined else best[2]
    ll_hat = refined[0] if refined else best[0]

    gaps = [b - a for events, _ in paths for a, b in zip(events, events[1:])]
    at_floor = abs(h_hat - half_lives[0]) < 1e-9
    return {
        "status": "DEVELOPMENT",
        "boundary_policy": "RESET_EACH_WINDOW_NO_WARMUP",
        "branching": b_hat,
        "half_life_operational": h_hat,
        "branching_grid_seed": best[1],
        "half_life_grid_seed": best[2],
        "continuous_refinement": refined is not None,
        "loglik": ll_hat,
        "poisson_loglik": baseline_ll,
        "delta_loglik": ll_hat - baseline_ll,
        "n_events": sum(len(events) for events, _ in paths),
        "n_paths": len(paths),
        "mean_operational_gap": sum(gaps) / len(gaps) if gaps else None,
        "short_gap_share_lt_0_25": (sum(g < 0.25 for g in gaps) / len(gaps)) if gaps else None,
        "branching_boundary_hit": b_hat <= 1e-6 or b_hat >= 0.899,
        "half_life_boundary_hit": at_floor or abs(h_hat - half_lives[-1]) < 1e-9,
        "half_life_grid": list(half_lives),
        "instrument_floor_operational": floor,
        "grid_points_excluded_below_instrument_floor": excluded,
        "venue_resolution_s": VENUE_RESOLUTION_S,
        "half_life_floor_ticks": HALF_LIFE_FLOOR_TICKS,
        "censored_at_instrument_floor": at_floor,
        "parameter_boundary_hit": (b_hat <= 1e-6 or b_hat >= 0.899 or at_floor),
    }


def assert_protocol_conformance(path: Path = PROTOCOL) -> int:
    """Check code constants against the FROZEN protocol. Raise on divergence.

    A `reference_snapshot` SHA-256 proves a file has not changed since it was
    hashed. It proves NOTHING about code conforming to a declaration, and this
    programme has now been bitten four times by a name that was not a definition.
    The fourth was this file: commit 332b09b extended the Hawkes half-life grid
    in code, refreshed the snapshot, left the DECLARED grid untouched in both
    frozen protocols, and the snapshot verified clean while the committed code
    conformed to no protocol at all. A change-detector is not a conformance
    checker; this is the conformance checker.
    """
    import yaml
    spec = yaml.safe_load(path.read_text())
    checks = 0
    problems: list[str] = []

    def same(label: str, code_value, declared, tol: float = 1e-12) -> None:
        nonlocal checks
        checks += 1
        a, b = list(code_value), list(declared)
        if len(a) != len(b) or any(abs(float(x) - float(y)) > tol for x, y in zip(a, b)):
            problems.append(f"{label}: code {a} != protocol {b}")

    same("hawkes.candidate_half_lives_operational",
         (0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
         spec["hawkes"]["candidate_half_lives_operational"])
    same("hawkes.branching_grid", [i / 20 for i in range(19)],
         spec["hawkes"]["branching_grid"])
    same("ground_process.remaining_time_edges_s", R_EDGES,
         spec["ground_process"]["remaining_time_edges_s"])
    same("ground_process.price_edges", P_EDGES, spec["ground_process"]["price_edges"])
    same("fill_actions.action_times_elapsed_s", ACTION_TIMES,
         spec["fill_actions"]["action_times_elapsed_s"])
    same("fill_actions.horizons_s", ACTION_HORIZONS, spec["fill_actions"]["horizons_s"])
    same("fill_actions.size_shares", [ACTION_SIZE], [spec["fill_actions"]["size_shares"]])
    same("marks.micro_size_shares", [MICRO_SIZE], [spec["marks"]["micro_size_shares"]])
    same("state.knowledge_lag_ms", [STATE_LAG_S * 1000.0],
         [spec["state"]["knowledge_lag_ms"]])
    same("state.minimum_price_bin_dwell_s", [fi.MIN_BIN_DWELL_S],
         [spec["state"]["minimum_price_bin_dwell_s"]])

    if problems:
        raise AssertionError("CODE DOES NOT CONFORM TO THE FROZEN PROTOCOL:\n  "
                             + "\n  ".join(problems))
    return checks


def a1_independence(windows: Sequence[DevWindow], tau: float = 0.25,
                    n_perm: int = 2000, seed: int = 20260821) -> dict[str, Any]:
    """A1 -- is the MICRO_002 class independent of market flow?

    Excluding the micro class is innocent ONLY under independence: a superposition
    of independent point processes decomposes, so deleting one component leaves
    the other's intensity AND branching structure intact. Without independence,
    ex-micro is a CONDITIONED SUB-PROCESS rather than a component, and every
    downstream quantity inherits the error silently -- the arithmetic still runs.
    Exposure is the per-coin micro share, 2% on btc against ~90% on hype.

    Statistic: cross-pairs within `tau` seconds. Null: a circular time-shift of
    the micro process inside each window, which preserves its own inter-arrival
    structure while destroying alignment with market arrivals.

    `tau` MUST be small relative to the market inter-arrival time. If 2*tau spans
    a typical market gap, every micro event has a neighbour whatever the
    alignment, the statistic saturates, and the shift cannot discriminate -- the
    test then returns NOT_REJECTED for a perfectly shadowing process. The
    selftest pins both directions precisely so this cannot regress silently.
    At btc's ~7 events/s, tau=0.25 spans ~3.5 market events; at hype's ~0.4/s it
    spans ~0.2, which is thin but permutation-valid.

    Failure to reject is NOT independence, so PASS requires the effect-size
    interval INSIDE a tolerance. Underpowered defaults to FAILING A1.
    """
    min_events = 50
    tol = 0.10
    rng = __import__("random").Random(seed)

    per_window: list[tuple[list[float], list[float], float]] = []
    for w in windows:
        micro = sorted(t.elapsed for t in w.trades if t.event_type == "MICRO_002")
        market = sorted(t.elapsed for t in w.trades if t.event_type != "MICRO_002")
        span = max((p.end for p in w.pieces), default=0.0)
        if micro and market and span > 0:
            per_window.append((micro, market, span))

    def cross_pairs(micro: Sequence[float], market: Sequence[float]) -> int:
        # market is sorted; count micro events with a market event within tau
        import bisect
        n = 0
        for m in micro:
            lo = bisect.bisect_left(market, m - tau)
            hi = bisect.bisect_right(market, m + tau)
            n += hi - lo
        return n

    def directed_pairs(micro: Sequence[float], market: Sequence[float]) -> tuple[int, int]:
        """(micro_leads, micro_follows) -- DIRECTION DECIDES THE CONSEQUENCE.

        A symmetric statistic detects association but cannot say which way it
        runs, and the two readings have opposite implications for the ex-micro
        process. If micro FOLLOWS market it is reactive, and deleting it leaves
        the market component's intensity and branching structure intact. If micro
        LEADS market it may be triggering, and deletion is not innocent.
        """
        import bisect
        leads = follows = 0
        for m in micro:
            leads += (bisect.bisect_right(market, m + tau)
                      - bisect.bisect_right(market, m))     # market strictly after
            follows += (bisect.bisect_left(market, m)
                        - bisect.bisect_left(market, m - tau))  # market strictly before
        return leads, follows

    n_micro = sum(len(m) for m, _, _ in per_window)
    n_market = sum(len(k) for _, k, _ in per_window)
    observed = sum(cross_pairs(m, k) for m, k, _ in per_window)
    obs_lead = sum(directed_pairs(m, k)[0] for m, k, _ in per_window)
    obs_follow = sum(directed_pairs(m, k)[1] for m, k, _ in per_window)

    null_counts: list[int] = []
    null_lead: list[int] = []
    null_follow: list[int] = []
    for _ in range(n_perm):
        total = lead = follow = 0
        for micro, market, span in per_window:
            shift = rng.uniform(0.0, span)
            shifted = sorted(((t + shift) % span) for t in micro)
            total += cross_pairs(shifted, market)
            dl, df = directed_pairs(shifted, market)
            lead += dl
            follow += df
        null_counts.append(total)
        null_lead.append(lead)
        null_follow.append(follow)
    mean_lead = sum(null_lead) / len(null_lead) if null_lead else 0.0
    mean_follow = sum(null_follow) / len(null_follow) if null_follow else 0.0
    lead_ratio = obs_lead / mean_lead if mean_lead > 0 else None
    follow_ratio = obs_follow / mean_follow if mean_follow > 0 else None
    p_lead = (sum(c >= obs_lead for c in null_lead) / max(len(null_lead), 1))
    p_follow = (sum(c >= obs_follow for c in null_follow) / max(len(null_follow), 1))
    mean_null = sum(null_counts) / len(null_counts) if null_counts else 0.0

    ge = sum(c >= observed for c in null_counts)
    le = sum(c <= observed for c in null_counts)
    p_two = min(1.0, 2.0 * min(ge, le) / max(len(null_counts), 1))
    ratio = observed / mean_null if mean_null > 0 else None

    # window-clustered bootstrap on the observed statistic
    lo = hi = None
    if per_window and mean_null > 0:
        boots = []
        for _ in range(1000):
            pick = [per_window[rng.randrange(len(per_window))] for _ in per_window]
            boots.append(sum(cross_pairs(m, k) for m, k, _ in pick) / mean_null)
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]

    if n_micro < min_events or n_market < min_events:
        verdict, holds = "INSUFFICIENT_POWER", False
    elif p_two < 0.05:
        verdict, holds = "DEPENDENT", False
    elif lo is not None and lo >= 1.0 - tol and hi <= 1.0 + tol:
        verdict, holds = "SUPPORTED", True
    else:
        verdict, holds = "NOT_REJECTED_NOT_EQUIVALENT", False

    # Direction, and what it licenses.
    if verdict == "DEPENDENT":
        lead_hot = p_lead < 0.05
        follow_hot = p_follow < 0.05
        if follow_hot and not lead_hot:
            direction = "MICRO_FOLLOWS_MARKET_REACTIVE"
            implication = "DELETION_LEAVES_MARKET_COMPONENT_INTACT"
        elif lead_hot and not follow_hot:
            direction = "MICRO_LEADS_MARKET"
            implication = "DELETION_NOT_INNOCENT_EX_MICRO_IS_CONDITIONED"
        elif lead_hot and follow_hot:
            direction = "BIDIRECTIONAL_OR_COMMON_DRIVER"
            implication = "DELETION_NOT_INNOCENT_EX_MICRO_IS_CONDITIONED"
        else:
            direction = "ASSOCIATION_WITHOUT_RESOLVED_DIRECTION"
            implication = "UNRESOLVED_TREAT_AS_CONDITIONED"
    else:
        direction = "NOT_APPLICABLE"
        implication = "NOT_APPLICABLE"

    return {
        "verdict": verdict,
        "a1_holds": holds,
        "direction": direction,
        "direction_implication": implication,
        "lead_ratio": lead_ratio, "follow_ratio": follow_ratio,
        "p_lead": p_lead, "p_follow": p_follow,
        "observed_lead_pairs": obs_lead, "observed_follow_pairs": obs_follow,
        "consequence_if_false": "EX_MICRO_IS_A_CONDITIONED_SUBPROCESS_NOT_A_COMPONENT",
        "tau_s": tau, "tolerance": tol, "min_events": min_events,
        "n_micro": n_micro, "n_market": n_market, "n_windows": len(per_window),
        "observed_cross_pairs": observed,
        "mean_null_cross_pairs": mean_null,
        "ratio": ratio, "ratio_ci95": [lo, hi],
        "p_two_sided": p_two, "n_permutations": n_perm,
    }


def _shift_book_covariates(window: DevWindow, k: int) -> DevWindow:
    """Circularly shift the piece covariate sequence by k pieces."""
    pieces = window.pieces
    n = len(pieces)
    if n < 2:
        return window
    shifted_x = [pieces[(i + k) % n].book_x for i in range(n)]
    new_pieces = [ExposurePiece(p.start, p.end, p.cell, p.tick_tail, shifted_x[i])
                  for i, p in enumerate(pieces)]
    starts = [p.start for p in pieces]
    import bisect
    new_trades = []
    for t in window.trades:
        idx = min(max(bisect.bisect_right(starts, t.elapsed) - 1, 0), n - 1)
        new_trades.append(DevTrade(
            t.elapsed, t.cell, t.event_type, t.side, t.exec_p_up, t.size,
            t.notional, t.reach_class, t.distance_ticks, t.tick_tail,
            shifted_x[idx]))
    out = copy.copy(window)
    out.pieces = new_pieces
    out.trades = new_trades
    return out


def b3_placebo(windows: Sequence[DevWindow], n_events: int) -> dict[str, Any]:
    """Negative control for B3, because a frozen lag is asserted, not demonstrated.

    `price_change` carries POST-change quotes and can be emitted BEFORE the
    `last_trade_price` for the same match, so the 250 ms pre-arrival read is a
    hope until something shows it worked. Circularly shifting the covariate
    sequence preserves its marginal distribution and autocorrelation while
    destroying alignment with arrivals. A B3 gain that SURVIVES the shift is not
    covariate information -- it is flexibility.
    """
    if len(windows) < 3:
        return {"status": "INSUFFICIENT_WINDOWS"}
    shifted = [_shift_book_covariates(w, max(1, len(w.pieces) // 3)) for w in windows]
    real = placebo = 0.0
    for i in range(len(windows)):
        train_r = [w for j, w in enumerate(windows) if j != i]
        train_p = [w for j, w in enumerate(shifted) if j != i]
        fit_r, fit_p = fit_baseline(train_r), fit_baseline(train_p)
        real += poisson_nll(windows[i], fit_r, "B3") - poisson_nll(windows[i], fit_r, "B2")
        placebo += poisson_nll(shifted[i], fit_p, "B3") - poisson_nll(shifted[i], fit_p, "B2")
    per = (lambda v: v / n_events if n_events else None)
    # A placebo gain at or beyond the real one means B3 carries no covariate signal.
    survives = (placebo < 0) and (real >= 0 or placebo <= real * 0.5)
    return {
        "status": "RUN",
        "real_b3_minus_b2_per_event": per(real),
        "placebo_b3_minus_b2_per_event": per(placebo),
        "placebo_share_of_real": (placebo / real) if real not in (0.0, None) else None,
        "verdict": "PLACEBO_ALSO_IMPROVES_B3_GAIN_NOT_COVARIATE_INFORMATION" if survives
                   else "PLACEBO_DOES_NOT_REPRODUCE_THE_GAIN",
    }


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    return data[min(int(q * len(data)), len(data) - 1)]


def summarize_coin(windows: Sequence[DevWindow]) -> dict[str, Any]:
    if len(windows) < 3:
        raise ValueError("development protocol requires at least three windows per coin")
    nll = {layer: 0.0 for layer in ("B0", "B1", "B2", "B3")}
    paths = []
    fold_diagnostics = []
    for i, held in enumerate(windows):
        train = [w for j, w in enumerate(windows) if j != i]
        fit = fit_baseline(train)
        for layer in nll:
            nll[layer] += poisson_nll(held, fit, layer)
        paths.append(operational_window(held, fit))
        fold_diagnostics.append({
            "held_slug": held.slug,
            "b2_tick_tail_beta": fit.b2_tick_tail_beta,
            "b2_zero_event_fence": fit.b2_zero_event_fence,
            "b3_gamma": fit.b3_gamma,
            "b3_optimizer_success": fit.b3_optimizer_success,
            "b3_gradient_norm": fit.b3_gradient_norm,
        })

    trades = [trade for w in windows for trade in w.trades]
    actions = [action for w in windows for action in w.actions]
    notional = [t.notional for t in trades]
    distances = [t.distance_ticks for t in trades if t.distance_ticks is not None]

    fill: dict[str, Any] = {}
    for horizon in ACTION_HORIZONS:
        all_subset = [a for a in actions if abs(a["horizon"] - horizon) < 1e-12]
        subset = [a for a in all_subset if a["status"] == "AVAILABLE"]
        if not subset:
            if all_subset:
                fill[str(int(horizon))] = {
                    "n_actions": len(all_subset),
                    "n_available": 0,
                    "n_unavailable": len(all_subset),
                    "n_touch_departed": sum(a["touch_departed"] for a in all_subset),
                    "unavailable_reasons": dict(collections.Counter(
                        a["unavailable_reason"] for a in all_subset
                    )),
                }
            continue
        front = [a["front_fill"] for a in subset]
        back = [a["back_fill"] for a in subset]
        fill[str(int(horizon))] = {
            "n_actions": len(all_subset),
            "n_available": len(subset),
            "n_unavailable": len(all_subset) - len(subset),
            "n_touch_departed": sum(a["touch_departed"] for a in all_subset),
            "unavailable_reasons": dict(collections.Counter(
                a["unavailable_reason"] for a in all_subset
                if a["status"] != "AVAILABLE"
            )),
            "front_any_fill": sum(x > 0 for x in front) / len(front),
            "back_any_fill": sum(x > 0 for x in back) / len(back),
            "front_complete": sum(x >= ACTION_SIZE - 1e-12 for x in front) / len(front),
            "back_complete": sum(x >= ACTION_SIZE - 1e-12 for x in back) / len(back),
            "front_mean_filled_shares": sum(front) / len(front),
            "back_mean_filled_shares": sum(back) / len(back),
            "mean_queue_bracket_width_shares": sum(f - b for f, b in zip(front, back)) / len(front),
        }

    diagnostics = collections.Counter()
    for w in windows:
        diagnostics.update(w.diagnostics)
    n_events = len(trades)
    return {
        "status": "DEVELOPMENT",
        "n_windows": len(windows),
        "n_events_state_admitted": n_events,
        "exposure_s": sum(sum(w.exposure.values()) for w in windows),
        "baseline": {
            "split": "LEAVE_ONE_WINDOW_OUT",
            "nll": {layer.lower(): value for layer, value in nll.items()},
            "delta_nll": {
                "b1_minus_b0": nll["B1"] - nll["B0"],
                "b2_minus_b1": nll["B2"] - nll["B1"],
                "b3_minus_b2": nll["B3"] - nll["B2"],
            },
            "delta_nll_per_event": {
                "b1_minus_b0": (nll["B1"] - nll["B0"]) / n_events if n_events else None,
                "b2_minus_b1": (nll["B2"] - nll["B1"]) / n_events if n_events else None,
                "b3_minus_b2": (nll["B3"] - nll["B2"]) / n_events if n_events else None,
            },
            # ADJACENT DELTAS HIDE A NON-MONOTONE STACK. "B3 improves on B2" reads
            # as "B3 is best", but if B2 costs more than B3 returns the full stack
            # is worse than B1 alone. Report cumulative-vs-B0 and name the winner.
            "cumulative_delta_nll_per_event_vs_b0": {
                layer.lower(): (nll[layer] - nll["B0"]) / n_events if n_events else None
                for layer in ("B1", "B2", "B3")
            },
            "best_layer": min(("B0", "B1", "B2", "B3"), key=lambda L: nll[L]),
            "full_stack_is_best": min(("B0", "B1", "B2", "B3"), key=lambda L: nll[L]) == "B3",
            "fold_diagnostics": fold_diagnostics,
        },
        "marks": {
            "micro_event_share": sum(t.event_type == "MICRO_002" for t in trades) / n_events if n_events else None,
            "unified_buy_share": sum(t.side == "BUY" for t in trades) / n_events if n_events else None,
            "marketable_at_lagged_state_share": sum(t.reach_class != "NOT_MARKETABLE_AT_LAGGED_STATE" for t in trades) / n_events if n_events else None,
            "native_notional_mean": sum(notional) / len(notional) if notional else None,
            "native_notional_p50": _quantile(notional, 0.5),
            "execution_distance_ticks_p50": _quantile(distances, 0.5),
        },
        "a1_micro_market_independence": a1_independence(windows),
        "b3_negative_control": b3_placebo(windows, n_events),
        "hawkes": fit_hawkes(paths, wall_seconds=len(windows) * fi.WINDOW_S),
        "join_touch_fill_bounds": fill,
        "n_actions_available": sum(a["status"] == "AVAILABLE" for a in actions),
        "n_actions_unavailable": sum(a["status"] != "AVAILABLE" for a in actions),
        "diagnostics": dict(diagnostics),
    }


def select_windows(per_coin: int) -> list[tuple[str, Path, str, str, list[tuple[float, float]]]]:
    paths = fi._archive_paths()
    tokens = fi.token_map()
    gaps = fi.gaps_by_slug(fi.ERA)
    picked: collections.Counter[str] = collections.Counter()
    out = []
    for slug in sorted(fi.covered_slugs(fi.ERA)):
        coin = slug.split("-")[0]
        if picked[coin] >= per_coin or slug not in paths or slug not in tokens:
            continue
        up, down = tokens[slug]
        out.append((slug, paths[slug], up, down, gaps.get(slug, [])))
        picked[coin] += 1
    return out


def run(per_coin: int) -> dict[str, Any]:
    assert_protocol_conformance()   # fail-closed before any measurement
    selected = select_windows(per_coin)
    by_coin: collections.defaultdict[str, list[DevWindow]] = collections.defaultdict(list)
    for i, (slug, path, up, down, gaps) in enumerate(selected, 1):
        print(f"[development] {i:02d}/{len(selected):02d} {slug}", flush=True)
        by_coin[slug.split("-")[0]].append(build_window(path, up, down, gaps))
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "FLOW_AND_FILLS_V4",
        "status": "DEVELOPMENT",
        "decision_eligible": False,
        "claim": "ENGINEERING_AND_WITHIN_DESIGN_DIAGNOSTICS_ONLY",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "source_window_start": dt.datetime.fromtimestamp(
            min(int(slug.rsplit("-", 1)[1]) for slug, *_ in selected), dt.timezone.utc
        ).isoformat(),
        "source_window_end": dt.datetime.fromtimestamp(
            max(int(slug.rsplit("-", 1)[1]) for slug, *_ in selected) + int(fi.WINDOW_S),
            dt.timezone.utc,
        ).isoformat(),
        "selected_slugs_by_coin": {
            coin: [slug for slug, *_ in selected if slug.startswith(coin + "-")]
            for coin in sorted(by_coin)
        },
        "state_lag_s": STATE_LAG_S,
        "r_edges_elapsed_s": R_EDGES,
        "p_edges": P_EDGES,
        "action_times_elapsed_s": ACTION_TIMES,
        "action_horizons_s": ACTION_HORIZONS,
        "action_size_shares": ACTION_SIZE,
        "coins": {coin: summarize_coin(windows) for coin, windows in sorted(by_coin.items())},
    }
    result["artifact_id"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    for coin, data in result["coins"].items():
        b = data["baseline"]
        h = data["hawkes"]
        f15 = data["join_touch_fill_bounds"].get("15", {})
        d = b["delta_nll_per_event"]
        print(
            f"{coin:6s} windows={data['n_windows']:2d} events={data['n_events_state_admitted']:6,d} "
            f"dNLL/event={d['b1_minus_b0']:+.4f}/{d['b2_minus_b1']:+.4f}/{d['b3_minus_b2']:+.4f} "
            f"hawkes_n={h['branching']:.2f} dLL={h['delta_loglik']:+.1f} "
            f"fill15 front/back={f15.get('front_any_fill', float('nan')):.3f}/"
            f"{f15.get('back_any_fill', float('nan')):.3f}"
        )
    return result


def selftest() -> int:
    checks = 0

    def ok(condition: bool, label: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1

    checks += assert_protocol_conformance()
    ok(True, "code conforms to the frozen protocol")

    # --- instrument floor: a coin-dependent operational floor from one wall limit
    fast = [([0.0, 1.0, 2.0], 3000.0)]          # 3000 expected arrivals in 300 s -> 10/s
    slow = [([0.0, 1.0, 2.0], 30.0)]            # 0.1/s
    f_fast = hawkes_floor_operational(fast, 300.0)
    f_slow = hawkes_floor_operational(slow, 300.0)
    ok(f_fast > f_slow, "busier coin must get a HIGHER operational floor")
    ok(abs(f_fast - 10.0 * HALF_LIFE_FLOOR_TICKS * VENUE_RESOLUTION_S) < 1e-9,
       f"floor must equal lam*ticks*resolution, got {f_fast}")
    ok(hawkes_floor_operational(fast, 0.0) == 0.0, "zero exposure must not divide")

    # CONTROL: the floor must actually REMOVE grid points on a fast coin, else it
    # is decorative -- this is the whole point of the fix.
    fitted_fast = fit_hawkes(fast, wall_seconds=300.0)
    ok(fitted_fast["grid_points_excluded_below_instrument_floor"],
       "instrument floor must exclude sub-resolution grid points on a fast coin")
    ok(min(fitted_fast["half_life_grid"]) >= fitted_fast["instrument_floor_operational"],
       "no retained grid point may sit below the instrument floor")
    ok(not fit_hawkes(fast)["grid_points_excluded_below_instrument_floor"],
       "with no wall_seconds the floor is inactive and excludes nothing")

    # --- continuous refinement must leave the grid, and must not lose likelihood
    rng_h = __import__("random").Random(11)
    ev, t = [], 0.0
    while t < 200.0:                       # clustered enough that branching > 0
        t += rng_h.expovariate(0.5)
        if t < 200.0:
            ev.append(t)
            if rng_h.random() < 0.35:
                t2 = t + rng_h.expovariate(4.0)
                if t2 < 200.0:
                    ev.append(t2)
    ev.sort()
    fit_c = fit_hawkes([(ev, 200.0)])
    ok(fit_c["loglik"] >= fit_c["poisson_loglik"], "fit must not be worse than Poisson")
    if fit_c["continuous_refinement"]:
        on_grid = any(abs(fit_c["branching"] - g / 20) < 1e-9 for g in range(19))
        ok(not on_grid or fit_c["branching"] == 0.0,
           "refined branching must leave the quantised grid")
        ok(fit_c["loglik"] >= fit_c["poisson_loglik"], "refined loglik >= baseline")
    else:
        checks += 2

    # CONTROL: the conformance checker must FAIL on a divergent protocol.
    # Without this it is exactly the unfalsifiable mechanism it was added to
    # replace -- a checker that reports success without checking.
    import tempfile, yaml as _yaml
    _spec = _yaml.safe_load(PROTOCOL.read_text())
    _spec["hawkes"]["candidate_half_lives_operational"] = [0.25, 0.5, 1, 2, 5, 10]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        _yaml.safe_dump(_spec, fh)
        _bad = Path(fh.name)
    try:
        assert_protocol_conformance(_bad)
    except AssertionError:
        checks += 1
    else:
        raise AssertionError("conformance checker must reject a divergent protocol")
    finally:
        _bad.unlink(missing_ok=True)

    ok(r_bin(0.0) == 0 and r_bin(299.9) == 4, "five frozen elapsed bands")
    ok(cell_at(30.0, 0.10) == (0, fi.p_bin(0.10)), "joint r/p cell")
    try:
        r_bin(301.0)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("r_bin must reject outside window")

    ok(action_fill(3.0, 5.0, 10.0) == (3.0, 0.0), "front partial/back none")
    ok(action_fill(12.0, 5.0, 10.0) == (5.0, 2.0), "front full/back partial")
    ok(action_fill(20.0, 5.0, 10.0) == (5.0, 5.0), "both full")
    try:
        action_fill(-1.0, 5.0, 0.0)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("negative fill inputs must fail")

    ok(reaches_action("BUY", 0.61, "SELL_UP", 0.60), "buy reaches maker ask")
    ok(not reaches_action("SELL", 0.61, "SELL_UP", 0.60), "wrong side cannot reach ask")
    ok(reaches_action("SELL", 0.39, "BUY_UP", 0.40), "sell reaches maker bid")
    ok(fi.fold_side("BUY", True) == "SELL" and abs(fi.fold_price(0.60, True) - 0.40) < 1e-12,
       "Down trade folds into unified side and price")

    state = BookState()
    state.apply("book", {"bids": [(0.49, 10.0)], "asks": [(0.51, 20.0)], "tick": 0.01})
    ok(state.quote() == (0.49, 0.51, 10.0, 20.0, 0.01), "full book initializes queue")
    state.apply("price", {"side": "BUY", "price": 0.49, "size": 7.0,
                          "best_bid": 0.49, "best_ask": 0.51})
    ok(state.quote()[2] == 7.0, "price_change size is level total")
    state.clear()
    ok(state.quote() is None, "gap clear removes queue state")
    state.apply("price", {"side": "BUY", "price": 0.49, "size": 7.0,
                          "best_bid": 0.49, "best_ask": 0.51})
    ok(state.quote() is None, "delta cannot recreate full post-gap queue")

    synthetic = DevWindow("x", "x", {(0, 0): 100.0}, {(0, 0): 10},
                          [(0.0, 100.0, (0, 0))], [], [], {})
    b0, b1 = fit_rates([synthetic])
    ok(b0[0] > 0 and b1[(0, 0)] > 0, "development rates remain positive")
    ok(abs(hawkes_loglik([([], 10.0)], 0.0, 1.0) + 10.0) < 1e-12,
       "unit Poisson empty-path likelihood")
    independent = [([1.0, 2.0, 3.0], 4.0)]
    ok(math.isfinite(hawkes_loglik(independent, 0.2, 1.0)), "Hawkes likelihood finite")
    h = fit_hawkes(independent)
    ok(h["status"] == "DEVELOPMENT" and h["n_events"] == 3,
       "Hawkes fit is explicitly development-only")

    action = ShadowAction(0.0, 10.0, "SELL_UP", 0.60, 5.0, 10.0)
    action.observe(1.0, "BUY", 0.60, 12.0)
    ar = action.result()
    ok(ar["front_fill"] == 5.0 and ar["back_fill"] == 2.0,
       "shadow action accumulates reaching volume into queue bounds")
    ok(ar["front_first_fill_s"] == 1.0 and ar["back_first_fill_s"] == 1.0,
       "first fill times carried at both bounds")
    invalid = ShadowAction(0.0, 10.0, "BUY_UP", 0.40, 5.0, 3.0)
    invalid.invalidate(2.0, "COLLECTOR_GAP")
    ok(invalid.result()["status"] == "UNAVAILABLE"
       and invalid.result()["unavailable_reason"] == "COLLECTOR_GAP",
       "gap-crossing action remains as an explicit unavailable row")
    moving = ShadowAction(0.0, 10.0, "SELL_UP", 0.60, 5.0, 3.0)
    moving.note_touch(2.0, 0.50, 0.61)
    ok(moving.result()["touch_departed"],
       "touch departure is retained without erasing the resting level")

    # --- A1 independence: must DETECT dependence and not cry wolf on independence.
    # Without both directions the verdict would be unfalsifiable.
    def _win(micro, market, span=300.0):
        w = DevWindow.__new__(DevWindow)
        w.slug = "t"
        w.pieces = [ExposurePiece(0.0, span, (0, 0), 0.0, (0.0, 0.0, 0.0))]
        w.trades = ([DevTrade(t, (0, 0), "MICRO_002", "SELL", 0.5, 0.02, 0.01,
                              "X", None, 0.0, (0.0, 0.0, 0.0)) for t in micro]
                    + [DevTrade(t, (0, 0), "MARKET", "BUY", 0.5, 5.0, 2.5,
                                "X", None, 0.0, (0.0, 0.0, 0.0)) for t in market])
        w.actions, w.exposure, w.diagnostics = [], {}, {}
        return w

    market_times = [i * 1.7 for i in range(120)]
    locked = _win([t + 0.05 for t in market_times], market_times)      # micro shadows market
    dep = a1_independence([locked, locked, locked], tau=0.1, n_perm=200, seed=1)
    ok(dep["verdict"] == "DEPENDENT" and not dep["a1_holds"],
       f"A1 must detect a shadowing micro process, got {dep['verdict']}")

    # A genuinely INDEPENDENT micro process must be random. Two deterministic
    # lattices are not independent -- their cross-correlation has moire structure
    # and the test rightly flags it.
    _rng = __import__("random").Random(4242)
    spread = _win(sorted(_rng.uniform(0.0, 300.0) for _ in range(120)), market_times)
    ind = a1_independence([spread, spread, spread], tau=0.1, n_perm=200, seed=1)
    ok(ind["verdict"] != "DEPENDENT",
       f"A1 must not flag an unaligned process, got {ind['verdict']}")

    ok(dep["direction"] == "MICRO_FOLLOWS_MARKET_REACTIVE",
       f"micro built 0.05s AFTER each market must resolve as FOLLOWS, got {dep['direction']}")
    ok(dep["follow_ratio"] > dep["lead_ratio"],
       "the follow channel must dominate for a lagging fixture")

    lead_fix = _win([t - 0.05 for t in market_times if t > 0.05], market_times)
    led = a1_independence([lead_fix] * 3, tau=0.1, n_perm=200, seed=1)
    ok(led["direction"] == "MICRO_LEADS_MARKET",
       f"micro built 0.05s BEFORE each market must resolve as LEADS, got {led['direction']}")
    ok(led["direction_implication"].startswith("DELETION_NOT_INNOCENT"),
       "a leading micro process must block innocent deletion")

    sat = a1_independence([locked, locked, locked], tau=1.0, n_perm=200, seed=1)
    ok(sat["verdict"] != "DEPENDENT",
       "control: at a saturating tau even a shadowing process is undetectable -- "
       "this is why tau must stay below the market inter-arrival time")

    thin = _win([1.0, 2.0], [3.0, 4.0])
    ok(a1_independence([thin, thin, thin], tau=0.1, n_perm=50)["verdict"] == "INSUFFICIENT_POWER",
       "A1 underpowered must FAIL, never default to independence")
    ok(not a1_independence([thin, thin, thin], tau=0.1, n_perm=50)["a1_holds"],
       "A1 underpowered must not report a1_holds")

    # --- placebo: the shift must actually move covariates, else the control is vacuous
    w = DevWindow.__new__(DevWindow)
    w.slug = "t"
    w.pieces = [ExposurePiece(float(i), float(i + 1), (0, 0), 0.0, (float(i), 0.0, 0.0))
                for i in range(6)]
    w.trades = [DevTrade(2.5, (0, 0), "MARKET", "BUY", 0.5, 1.0, 0.5, "X", None,
                         0.0, (2.0, 0.0, 0.0))]
    w.actions, w.exposure, w.diagnostics = [], {}, {}
    sh = _shift_book_covariates(w, 2)
    ok([pc.book_x for pc in sh.pieces] != [pc.book_x for pc in w.pieces],
       "covariate shift must change the piece sequence")
    ok(sh.trades[0].book_x != w.trades[0].book_x,
       "covariate shift must re-key the trade covariate")
    ok([(pc.start, pc.end) for pc in sh.pieces] == [(pc.start, pc.end) for pc in w.pieces],
       "covariate shift must NOT move piece boundaries")

    print(f"flow_fill_development selftest: {checks} checks OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", nargs="?", choices=["run"])
    parser.add_argument("--per-coin", type=int, default=24)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.cmd == "run":
        run(args.per_coin)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
