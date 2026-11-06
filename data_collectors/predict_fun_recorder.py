"""Read-only Predict.fun research recorder. See docs/PREDICT_FUN_RESEARCH.md.

Only public market-data GETs and WS subscriptions; no wallet, JWT or signing.
Raw receipt order is preserved. REST matches are NOT a real-time trade tape.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import os
import queue
import re
import signal
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

SCHEMA = "predict_research_raw_v2"
REST = "https://api.predict.fun"
TESTNET_REST = "https://api-testnet.predict.fun"
WS = "wss://ws.predict.fun/ws"
BINANCE_SPOT_WS = "wss://data-stream.binance.vision:443/stream"
BINANCE_FUTURES_WS = "wss://fstream.binance.com/stream"
TOPICS = ("predictOrderbook", "predictTradingStatus", "predictMarketStatus", "predictMarketChanged")
FETCH_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, TypeError)


def clock() -> dict:
    return {"recv_ns": time.time_ns(), "recv_mono_ns": time.monotonic_ns()}


def utc_timestamp(value):
    """Venue times must carry an offset; never fall back to the host timezone."""
    if not isinstance(value, str):
        raise ValueError("timestamp missing/invalid")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("timestamp timezone missing")
    return parsed.timestamp()


def parse_predict_frame(raw):
    frame = json.loads(raw)
    if not isinstance(frame, dict) or frame.get("type") not in ("M", "R"):
        raise ValueError("invalid WS envelope/type")
    if frame["type"] == "M":
        if not isinstance(frame.get("topic"), str) or not frame["topic"] or "data" not in frame:
            raise ValueError("invalid WS message topic/data")
    elif type(frame.get("requestId")) is not int or frame["requestId"] <= 0:
        raise ValueError("invalid WS response request identity")
    return frame


def parse_rest_envelope(raw):
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise ValueError("REST envelope did not report success")
    return payload


def match_page(payload, market_id):
    """Validate a whole page before it can change traversal state."""
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("match data is not a list")
    times = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("market"), dict) or row["market"].get("id") != market_id:
            raise ValueError("match market identity mismatch")
        times.append(utc_timestamp(row.get("executedAt")))
    cursor = payload.get("cursor")
    if cursor is not None and (not isinstance(cursor, str) or not cursor):
        raise ValueError("invalid match cursor")
    if times != sorted(times, reverse=True):
        raise ValueError("match page is not newest first")
    return times, cursor


def book_diagnostics(payload, expected_market_id):
    """Shared live/offline depth validation; empty/one-sided books are valid."""
    if not isinstance(payload, dict) or type(payload.get("marketId")) is not int or payload["marketId"] != expected_market_id:
        raise ValueError("book market identity mismatch")
    if type(payload.get("updateTimestampMs")) is not int or payload["updateTimestampMs"] < 0:
        raise ValueError("book source timestamp missing/invalid")
    sides = []
    for side in ("bids", "asks"):
        levels = payload.get(side)
        if not isinstance(levels, list):
            raise ValueError("book levels not a list")
        prices = []
        for level in levels:
            if not isinstance(level, list) or len(level) != 2 or any(isinstance(v, bool) for v in level):
                raise ValueError("malformed price level")
            try:
                price, size = map(float, level)
            except (TypeError, ValueError, OverflowError):
                raise ValueError("invalid book number") from None
            if not (math.isfinite(price) and 0 <= price <= 1 and math.isfinite(size) and size >= 0):
                raise ValueError("invalid book price/size")
            prices.append(price)
        if prices != sorted(prices, reverse=side == "bids") or len(set(prices)) != len(prices):
            raise ValueError("unsorted/duplicate book levels")
        sides.append(prices)
    bids, asks = sides
    return {"two_sided": bool(bids and asks), "empty": not bids and not asks,
            "locked_or_crossed": bool(bids and asks and bids[0] >= asks[0])}


def slugs_at(now: float, coins: list[str], intervals: list[int]) -> list[str]:
    """Previous/current/next windows; returned metadata must confirm identity."""
    return [f"{coin}-updown-{minutes}m-{int(now) // (minutes * 60) * minutes * 60 + offset * minutes * 60}"
            for coin in coins for minutes in intervals for offset in (-1, 0, 1)]


def retained_slugs_at(now: float, coins: list[str], intervals: list[int], retention_s: int) -> set[str]:
    """Recompute the retention window, including after restarts/discovery outages."""
    slugs = set()
    for coin in coins:
        for minutes in intervals:
            width = minutes * 60
            slot = int(now) // width * width
            earliest = int((now - retention_s) // width) * width - width
            for start in range(earliest, slot + width + 1, width):
                if now <= start + width + retention_s:
                    slugs.add(f"{coin}-updown-{minutes}m-{start}")
    return slugs


def category_markets(data: dict, slug: str) -> tuple[int, list[int]]:
    """Refuse ambiguous discovery instead of guessing token or market identity."""
    match = re.fullmatch(r"([a-z0-9]+)-updown-(5|15)m-(\d+)", slug)
    if not isinstance(data, dict) or not match or data.get("slug") != slug or data.get("marketVariant") != "CRYPTO_UP_DOWN":
        raise ValueError("category identity/variant mismatch")
    start = int(match[3])
    end = start + int(match[2]) * 60
    for key, expected in (("startsAt", start), ("endsAt", end)):
        if abs(utc_timestamp(data.get(key)) - expected) > 1:
            raise ValueError("category window mismatch")
    markets = data.get("markets")
    if not isinstance(markets, list) or not markets:
        raise ValueError("category markets missing")
    ids = []
    for market in markets:
        if not isinstance(market, dict) or type(market.get("id")) is not int or market["id"] <= 0:
            raise ValueError("invalid market id")
        outcomes = market.get("outcomes", [])
        if not isinstance(outcomes, list) or not all(isinstance(o, dict) and isinstance(o.get("name"), str)
                                                   and type(o.get("indexSet")) is int for o in outcomes):
            raise ValueError("invalid outcome schema")
        identity = {(o.get("name", "").lower(), o.get("indexSet")) for o in outcomes}
        if len(outcomes) != 2 or identity != {("up", 1), ("down", 2)}:
            raise ValueError("UP/YES outcome mapping unconfirmed")
        if not all(isinstance(o.get("onChainId"), str) and o["onChainId"].isdigit() for o in outcomes):
            raise ValueError("outcome token identity missing")
        if len({o["onChainId"] for o in outcomes}) != 2:
            raise ValueError("duplicate outcome tokens")
        if market.get("categorySlug", slug) != slug:
            raise ValueError("market category mismatch")
        if market["id"] in ids:
            raise ValueError("duplicate market id")
        ids.append(market["id"])
    return end, ids


class RawWriter:
    """Disk/compression runs off the reader thread; overflow aborts visibly.

    Each run has exclusive files. Flushes are not fsync durability guarantees.
    Receipt clocks describe userspace delivery, not exchange or kernel arrival.
    """
    def __init__(self, root: Path, metadata: dict, capacity: int = 20000):
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:10]
        self.path = root / "runs" / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        source = Path(__file__).read_bytes()
        (self.path / "collector_source.py").write_bytes(source)
        self.metadata = {**metadata, "schema": SCHEMA, "run_id": self.run_id, "started": clock(),
                         "source_sha256": hashlib.sha256(source).hexdigest(),
                         "stamp_point": "AFTER_AIOHTTP_RECEIVE_BEFORE_JSON_PARSE",
                         "research_status": "EXPLORATORY_UNCALIBRATED",
                         "durability": "gzip flush at most 1s; no fsync; abrupt loss may truncate tail"}
        (self.path / "manifest.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        self.q = queue.Queue(maxsize=capacity)
        self.seq = 0
        self.persisted = 0
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def emit(self, source: str, kind: str, *, stamp: dict | None = None, **fields):
        if self.error:
            raise RuntimeError("raw writer failed") from self.error
        row = {"run_id": self.run_id, "seq": self.seq, "source": source, "kind": kind,
               **(stamp or clock()), **fields}
        try:
            self.q.put_nowait(row)
        except queue.Full as exc:
            raise RuntimeError("raw writer overflow; run incomplete") from exc
        self.seq += 1

    def _write(self):
        handles = {}
        created = set()
        last_flush = time.monotonic()
        try:
            while True:
                try:
                    row = self.q.get(timeout=0.5)
                except queue.Empty:
                    row = False
                if row is None:
                    break
                if row is not False:
                    hour = datetime.fromtimestamp(row["recv_ns"] / 1e9, timezone.utc).strftime("%Y%m%d_%H")
                    source = row["source"]
                    if source not in handles or handles[source][0] != hour:
                        if source in handles:
                            handles[source][1].close()
                        path = self.path / row["source"] / f"{hour}.jsonl.gz"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        # A backward clock step may append a second gzip member only
                        # to a path created by this run. First creation is exclusive.
                        handles[source] = hour, gzip.open(path, "at" if path in created else "xt", encoding="utf-8")
                        created.add(path)
                    handles[source][1].write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    self.persisted += 1
                if time.monotonic() - last_flush >= 1:
                    for _, handle in handles.values():
                        handle.flush()
                    last_flush = time.monotonic()
        except BaseException as exc:
            self.error = exc
        finally:
            for _, handle in handles.values():
                try:
                    handle.close()
                except BaseException as exc:
                    self.error = exc

    async def close(self, status: str):
        while self.thread.is_alive():
            try:
                self.q.put_nowait(None)
                break
            except queue.Full:
                await asyncio.sleep(0.05)
        await asyncio.to_thread(self.thread.join)
        summary = {"run_id": self.run_id, "ended": clock(), "status": status,
                   "enqueued": self.seq, "persisted": self.persisted,
                   "writer_error": type(self.error).__name__ if self.error else None}
        (self.path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if self.error or self.seq != self.persisted:
            raise RuntimeError("run did not persist all queued records") from self.error


class FatalAccess(RuntimeError):
    pass


class MatchCursorRefused(ValueError):
    pass


class Recorder:
    def __init__(self, args, writer: RawWriter):
        self.args, self.writer = args, writer
        self.base = TESTNET_REST if args.testnet else REST
        self.key = os.environ.get(args.api_key_env, "")
        self.markets = {}  # id -> (slug, end epoch seconds)
        self.categories = {}  # slug -> end; retain for late resolution reads
        self.match_cutoff = {}
        self.match_backlogs = {}  # in-process cursor continuation; raw pages remain authoritative
        self.initial_match_cutoff = time.time() - 60
        self.next_match_poll = {}
        self.resnapshot_due = set()
        self.last_request = 0.0
        self.rest_lock = asyncio.Lock()
        self.stop = asyncio.Event()
        self.session = None
        self.predict_connected = False
        self.last_predict_frame = None
        self.book_receipts = {}

    def event(self, kind, **fields):
        self.writer.emit("events", kind, **fields)

    async def pause(self, seconds):
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def get(self, path: str, params=None):
        # This allowlist is also a guard against accidental credential-bearing writes.
        if not re.fullmatch(r"/v1/(categories/[a-z0-9-]+|markets/\d+(/orderbook)?|orders/matches)", path):
            raise ValueError("not an allowed market-data GET")
        async with self.rest_lock:
            await asyncio.sleep(max(0, self.last_request + 60 / self.args.rest_rpm - time.monotonic()))
            self.last_request = time.monotonic()
            sent = clock()
            headers = {"x-api-key": self.key} if self.key else {}
            async with self.session.get(self.base + path, params=params, headers=headers,
                                        allow_redirects=False) as response:
                raw = await response.text()
                stamp = clock()
                self.writer.emit("rest", "response", stamp=stamp, path=path, params=params,
                                 sent=sent, status=response.status, raw=raw)
                if response.status in (401, 403):
                    raise FatalAccess(f"market-data access refused: HTTP {response.status}")
                if response.status == 429:
                    # No key or URL is printed in exceptions. Preserve the response on disk.
                    delay = response.headers.get("Retry-After", "10")
                    try:
                        delay = max(1.0, min(300.0, float(delay)))
                    except ValueError:
                        delay = 10.0
                    self.event("REST_RATE_LIMIT", retry_after_s=delay)
                    await self.pause(delay)
                    return None
                if response.status != 200:
                    self.event("REST_UNAVAILABLE", path=path, status=response.status)
                    if path == "/v1/orders/matches" and params and params.get("after") and response.status in (400, 404):
                        raise MatchCursorRefused("match cursor request refused")
                    return None
                return parse_rest_envelope(raw)

    async def discover_once(self):
        now = time.time()
        slugs = retained_slugs_at(now, self.args.coins, self.args.intervals, self.args.retention_s)
        slugs.update(slug for slug, end in self.categories.items() if now <= end + self.args.retention_s)
        for slug in sorted(slugs, key=lambda value: int(value.rsplit("-", 1)[1]), reverse=True):
            if self.stop.is_set():
                break
            try:
                payload = await self.get("/v1/categories/" + slug)
            except FETCH_ERRORS as exc:
                self.event("CATEGORY_FETCH_FAILED", slug=slug, error=type(exc).__name__)
                continue
            if payload is None:
                continue
            try:
                end, ids = category_markets(payload["data"], slug)
            except (ValueError, TypeError, KeyError) as exc:
                self.event("CATEGORY_REFUSED", slug=slug, reason=str(exc))
                continue
            self.categories[slug] = end
            for market_id in ids:
                if market_id in self.markets and self.markets[market_id] != (slug, end):
                    self.event("CATEGORY_REFUSED", slug=slug, reason="market id already mapped to another window")
                    continue
                if market_id not in self.markets:
                    self.event("MARKET_DISCOVERED", market_id=market_id, slug=slug, end=end)
                    self.markets[market_id] = (slug, end)
                    self.resnapshot_due.add(market_id)
                    await self.refresh_snapshot(market_id)
        now = time.time()
        for market_id, (slug, end) in list(self.markets.items()):
            if now > end + self.args.retention_s:
                self.event("MARKET_RETIRED", market_id=market_id, slug=slug,
                           resolution_may_still_be_pending=True)
                self.markets.pop(market_id)
                self.match_cutoff.pop(market_id, None)
                backlog = self.match_backlogs.pop(market_id, None)
                if backlog:
                    self.event("MATCH_POLL_INCOMPLETE", market_id=market_id,
                               reason="RETIRED_WITH_BACKLOG", pages_completed=backlog["pages"])
                self.next_match_poll.pop(market_id, None)
                self.resnapshot_due.discard(market_id)
                self.book_receipts.pop(market_id, None)
        self.categories = {slug: end for slug, end in self.categories.items()
                           if now <= end + self.args.retention_s}

    async def refresh_snapshot(self, market_id):
        """A failed/malformed snapshot stays due until a valid book is read."""
        try:
            snapshot = await self.get(f"/v1/markets/{market_id}/orderbook")
            if snapshot is not None:
                book_diagnostics(snapshot.get("data"), market_id)
        except FETCH_ERRORS as exc:
            self.event("SNAPSHOT_FETCH_FAILED", market_id=market_id, error=type(exc).__name__)
            return
        if snapshot is not None:
            self.resnapshot_due.discard(market_id)
            self.event("RESNAPSHOT_FETCHED", market_id=market_id, reconstruct_by_receipt_only=True)

    async def matches_once(self, market_id):
        # A page budget yields to other markets; it must not force every future
        # poll to reread the same newest pages forever. The cutoff moves only
        # when the entire traversal finishes. New head rows enter the next sweep.
        state = self.match_backlogs.get(market_id)
        if state is None:
            cutoff = self.match_cutoff.get(market_id, self.initial_match_cutoff)
            state = {"cutoff": cutoff, "cursor": None, "visited": set(), "newest": cutoff, "pages": 0}
        cutoff = state["cutoff"]
        started = time.monotonic()
        for _ in range(self.args.match_pages):
            params = {"marketId": str(market_id), "first": "100"}
            if state["cursor"]:
                params["after"] = state["cursor"]
            try:
                payload = await self.get("/v1/orders/matches", params)
            except MatchCursorRefused:
                self.match_backlogs.pop(market_id, None)
                self.event("MATCH_POLL_INCOMPLETE", market_id=market_id, reason="CURSOR_REFUSED_RESTART_FROM_HEAD")
                return
            if payload is None:
                self.event("MATCH_POLL_INCOMPLETE", market_id=market_id, reason="HTTP_UNAVAILABLE")
                return
            try:
                times, cursor = match_page(payload, market_id)
            except (ValueError, TypeError, KeyError):
                self.match_backlogs.pop(market_id, None)
                raise
            if times and max(times) > state.get("oldest", float("inf")):
                self.match_backlogs.pop(market_id, None)
                self.event("MATCH_POLL_INCOMPLETE", market_id=market_id, reason="PAGINATION_ORDER_CHANGED")
                return
            state["newest"] = max([state["newest"], *times])
            state["pages"] += 1
            if not cursor or (times and min(times) < cutoff):
                # This records traversal, not completeness: late indexed trades and
                # settlement-vs-match clock semantics remain empirical questions.
                self.match_cutoff[market_id] = max(cutoff, state["newest"] - 60)
                self.match_backlogs.pop(market_id, None)
                self.event("MATCH_RANGE_TRAVERSED", market_id=market_id, pages=state["pages"],
                           cutoff=cutoff, newest=state["newest"], completeness_certified=False)
                return
            if cursor in state["visited"]:
                self.match_backlogs.pop(market_id, None)
                self.event("MATCH_POLL_INCOMPLETE", market_id=market_id, reason="CURSOR_LOOP")
                return
            state["visited"].add(cursor)
            state["cursor"] = cursor
            if times:
                state["oldest"] = min(times)
            self.match_backlogs[market_id] = state
            if time.monotonic() - started >= self.args.match_batch_s:
                self.event("MATCH_POLL_INCOMPLETE", market_id=market_id, reason="BATCH_TIME_LIMIT",
                           resume_queued=True, pages_completed=state["pages"])
                return
        self.event("MATCH_POLL_INCOMPLETE", market_id=market_id, reason="PAGE_LIMIT",
                   resume_queued=True, pages_completed=state["pages"])

    async def rest_loop(self):
        next_discovery = 0.0
        while not self.stop.is_set():
            try:
                if time.monotonic() >= next_discovery:
                    await self.discover_once()
                    next_discovery = time.monotonic() + self.args.discovery_s
                # Prioritize live/upcoming windows. Ended markets retain late-index
                # overlap, but must not monopolize the shared REST budget.
                for market_id in sorted(self.markets, key=lambda mid: self.markets[mid][1], reverse=True):
                    if self.stop.is_set() or time.monotonic() >= next_discovery:
                        break
                    if market_id in self.resnapshot_due:
                        await self.refresh_snapshot(market_id)
                    if time.monotonic() < self.next_match_poll.get(market_id, 0):
                        continue
                    try:
                        await self.matches_once(market_id)
                    except FETCH_ERRORS as exc:
                        self.event("MATCH_POLL_INCOMPLETE", market_id=market_id,
                                   reason="FETCH_OR_SCHEMA_ERROR", error=type(exc).__name__)
                    ended = time.time() > self.markets[market_id][1] + 60
                    interval = self.args.late_match_poll_s if ended else self.args.match_poll_s
                    self.next_match_poll[market_id] = time.monotonic() + interval
            except FatalAccess:
                raise
            except FETCH_ERRORS as exc:
                self.event("REST_CYCLE_FAILED", error=type(exc).__name__)
            await self.pause(min(self.args.match_poll_s, 1))

    async def predict_loop(self):
        attempt = 0
        while not self.stop.is_set():
            connection_id = uuid.uuid4().hex
            try:
                async with self.session.ws_connect(WS, headers={"x-api-key": self.key},
                                                   autoping=True, max_msg_size=16 * 1024 * 1024) as ws:
                    self.event("WS_CONNECTED", connection_id=connection_id, feed="predict")
                    self.predict_connected = True
                    self.resnapshot_due.update(self.markets)
                    subscribed, pending, request_id = set(), {}, 0
                    last_rx = time.monotonic()
                    while not self.stop.is_set():
                        desired = {f"{topic}/{mid}" for mid in self.markets for topic in TOPICS}
                        for method, topics in (("subscribe", desired - subscribed), ("unsubscribe", subscribed - desired)):
                            for topic in sorted(topics):
                                request_id += 1
                                await ws.send_json({"method": method, "requestId": request_id, "params": [topic]})
                                pending[request_id] = (time.monotonic(), topic)
                                self.event("WS_REQUEST", connection_id=connection_id, method=method,
                                           topic=topic, request_id=request_id)
                                if method == "subscribe":
                                    subscribed.add(topic)
                                else:
                                    subscribed.remove(topic)
                        if any(time.monotonic() - start > 15 for start, _ in pending.values()):
                            raise ValueError("subscription acknowledgement timeout")
                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=1)
                        except asyncio.TimeoutError:
                            if time.monotonic() - last_rx > 40:
                                raise ValueError("websocket silent")
                            continue
                        stamp = clock()  # Before JSON parsing and disk work.
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            raise ValueError("websocket closed or non-text frame")
                        last_rx = time.monotonic()
                        self.writer.emit("predict_ws", "frame", stamp=stamp,
                                         connection_id=connection_id, raw=msg.data)
                        self.last_predict_frame = stamp
                        data = parse_predict_frame(msg.data)
                        self.observe_book(data, stamp)
                        if data.get("type") == "M" and data.get("topic") == "heartbeat":
                            await ws.send_json({"method": "heartbeat", "data": data["data"]})
                        elif data.get("type") == "R":
                            if data.get("success") is not True or data.get("requestId") not in pending:
                                raise ValueError("subscription response refused/unmatched")
                            pending.pop(data["requestId"])
                            attempt = 0
            except aiohttp.WSServerHandshakeError as exc:
                if exc.status in (401, 403):
                    raise FatalAccess(f"WS access refused: HTTP {exc.status}") from None
                self.event("WS_FAILURE", feed="predict", connection_id=connection_id, error=type(exc).__name__)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, TypeError) as exc:
                self.event("WS_FAILURE", feed="predict", connection_id=connection_id, error=type(exc).__name__)
            finally:
                self.predict_connected = False
                self.event("WS_BOUNDARY", feed="predict", connection_id=connection_id,
                           status="STOP" if self.stop.is_set() else "DISCONNECTED_UNRECOVERABLE_BOOK_INTERVAL")
            if not self.stop.is_set():
                attempt += 1
                await self.pause(min(30, 2 ** min(attempt, 5)))

    async def binance_loop(self, feed):
        streams = "/".join(f"{coin}usdt@{stream}" for coin in self.args.coins
                           for stream in ("bookTicker", "trade"))
        base = BINANCE_SPOT_WS if feed == "binance_spot" else BINANCE_FUTURES_WS
        attempt = 0
        while not self.stop.is_set():
            connection_id = uuid.uuid4().hex
            try:
                async with self.session.ws_connect(base + "?streams=" + streams, autoping=True) as ws:
                    self.event("WS_CONNECTED", feed=feed, connection_id=connection_id)
                    while not self.stop.is_set():
                        msg = await asyncio.wait_for(ws.receive(), timeout=45)
                        stamp = clock()
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            raise ValueError("websocket closed or non-text frame")
                        self.writer.emit(feed, "frame", stamp=stamp, connection_id=connection_id, raw=msg.data)
                        attempt = 0
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                self.event("WS_FAILURE", feed=feed, connection_id=connection_id,
                           error=type(exc).__name__, status=getattr(exc, "status", None))
            finally:
                self.event("WS_BOUNDARY", feed=feed, connection_id=connection_id,
                           status="STOP" if self.stop.is_set() else "DISCONNECTED_UNRECOVERABLE_INTERVAL")
            if not self.stop.is_set():
                attempt += 1
                await self.pause(min(30, 2 ** min(attempt, 5)))

    def observe_book(self, frame, stamp):
        """Track receipt freshness separately from heartbeat/control traffic.

        This is telemetry, not a reconstruction or admission check. Empty or
        one-sided books are retained because they may be legitimate venue states.
        """
        if not isinstance(frame, dict):
            raise ValueError("invalid WS envelope")
        if frame.get("type") != "M":
            return
        topic = frame.get("topic", "")
        if not isinstance(topic, str):
            raise ValueError("invalid WS topic")
        if not topic.startswith("predictOrderbook/"):
            return
        payload = frame.get("data")
        match = re.fullmatch(r"predictOrderbook/([1-9]\d*)", topic)
        if not match:
            raise ValueError("invalid book topic identity")
        market_id = int(match[1])
        try:
            book_diagnostics(payload, market_id)
        except ValueError as exc:
            self.event("BOOK_OBSERVATION_REFUSED", topic=topic, reason=str(exc))
            return
        if market_id in self.markets:
            self.book_receipts[market_id] = {**stamp, "updateTimestampMs": payload.get("updateTimestampMs")}

    def health_snapshot(self):
        now_wall, now_mono = time.time_ns(), time.monotonic_ns()
        books = []
        for mid, (slug, end) in sorted(self.markets.items()):
            receipt = self.book_receipts.get(mid)
            books.append({"market_id": mid, "slug": slug, "end_s": end,
                          "book_observed_on_this_process": receipt is not None,
                          "book_recv_age_s": (now_mono - receipt["recv_mono_ns"]) / 1e9 if receipt else None,
                          "book_source_ms": receipt.get("updateTimestampMs") if receipt else None,
                          "snapshot_retry_due": mid in self.resnapshot_due,
                          "match_backlog_pages": self.match_backlogs.get(mid, {}).get("pages", 0)})
        return {"run_id": self.writer.run_id, "sampled_at_ns": now_wall,
                "pid": os.getpid(), "no_binance": self.args.no_binance,
                "n_markets": len(self.markets), "queue_depth": self.writer.q.qsize(),
                "enqueued": self.writer.seq, "persisted": self.writer.persisted,
                "predict_connected": self.predict_connected,
                "last_predict_frame_age_s": (now_mono - self.last_predict_frame["recv_mono_ns"]) / 1e9 if self.last_predict_frame else None,
                "books": books, "content_liveness": "NOT_CERTIFIED"}

    async def health_loop(self):
        while not self.stop.is_set():
            health = self.health_snapshot()
            self.event("PROCESS_HEARTBEAT", **{k: v for k, v in health.items() if k != "run_id"})
            def save():
                pending = self.writer.path / "health.json.pending"
                pending.write_text(json.dumps(health, indent=2, allow_nan=False) + "\n")
                pending.replace(self.writer.path / "health.json")
            await asyncio.to_thread(save)
            print(f"[predict] markets={health['n_markets']} ws_connected={health['predict_connected']} "
                  f"queue={health['queue_depth']} persisted={health['persisted']}", flush=True)
            await self.pause(30)

    async def run(self):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            self.session = session
            if self.args.probe:
                await self.discover_once()
                if not self.markets:
                    raise RuntimeError("probe found no mapped markets; inspect recorded discovery refusals")
                for mid in list(self.markets):
                    await self.matches_once(mid)
                return
            coroutines = [self.rest_loop(), self.health_loop()]
            if not self.args.rest_only:
                coroutines.append(self.predict_loop())
            if not self.args.no_binance:
                coroutines += [self.binance_loop("binance_spot"), self.binance_loop("binance_futures")]
            tasks = [asyncio.create_task(coro) for coro in coroutines]
            stop_task = asyncio.create_task(self.stop.wait())
            timer = asyncio.create_task(asyncio.sleep(self.args.duration_s)) if self.args.duration_s else None
            try:
                await asyncio.wait([*tasks, stop_task, *([timer] if timer else [])], return_when=asyncio.FIRST_COMPLETED)
                for task in tasks:
                    if task.done():
                        task.result()  # Propagate failed readers; never leave a partial run alive.
            finally:
                self.stop.set()
                for task in [*tasks, stop_task, *([timer] if timer else [])]:
                    task.cancel()
                await asyncio.gather(*tasks, stop_task, *([timer] if timer else []), return_exceptions=True)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/predict_fun"))
    parser.add_argument("--api-key-env", default="PREDICT_FUN_API_KEY")
    parser.add_argument("--coins", default="btc")
    parser.add_argument("--intervals", default="5", help="comma-separated minutes: 5,15")
    parser.add_argument("--duration-s", type=float, default=0, help="0 runs until SIGTERM")
    parser.add_argument("--probe", action="store_true", help="one bounded REST discovery/match sweep")
    parser.add_argument("--testnet", action="store_true", help="REST schema exercise only")
    parser.add_argument("--rest-only", action="store_true")
    parser.add_argument("--no-binance", action="store_true")
    parser.add_argument("--rest-rpm", type=int, default=120, help="shared GET budget, below default 240/min")
    parser.add_argument("--discovery-s", type=float, default=30)
    parser.add_argument("--match-poll-s", type=float, default=5)
    parser.add_argument("--late-match-poll-s", type=float, default=60)
    parser.add_argument("--match-pages", type=int, default=40)
    parser.add_argument("--match-batch-s", type=float, default=5,
                        help="yield a match traversal between pages after this duration")
    parser.add_argument("--retention-s", type=int, default=1800, help="retain ended markets for late reads")
    args = parser.parse_args(argv)
    args.coins = sorted(set(args.coins.lower().split(",")))
    try:
        args.intervals = sorted(set(int(v) for v in args.intervals.split(",")))
    except ValueError:
        parser.error("intervals must be 5 and/or 15")
    if not all(re.fullmatch(r"[a-z0-9]{2,10}", coin) for coin in args.coins):
        parser.error("invalid coin symbol")
    if not args.intervals or not set(args.intervals) <= {5, 15}:
        parser.error("intervals must be 5 and/or 15")
    if not 1 <= args.rest_rpm <= 180 or args.match_pages < 1 or args.retention_s < 60:
        parser.error("invalid rate, pagination or retention setting")
    if not all(0 < v < float("inf") for v in (args.discovery_s, args.match_poll_s, args.late_match_poll_s, args.match_batch_s)) or not 0 <= args.duration_s < float("inf"):
        parser.error("invalid time setting")
    if args.testnet and not args.rest_only:
        parser.error("testnet WS is unverified; use --rest-only")
    if args.testnet:
        args.no_binance = True
    if not args.testnet and not os.environ.get(args.api_key_env):
        parser.error(f"mainnet needs API key in environment variable {args.api_key_env}")
    return args


async def main(argv=None):
    args = arguments(argv)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata["network"] = "TESTNET_SCHEMA_ONLY" if args.testnet else "MAINNET"
    writer = RawWriter(args.root, metadata)
    recorder = Recorder(args, writer)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, recorder.stop.set)
    status = "STOPPED_WITH_BOUNDARIES_NO_DATA_QUALITY_CERTIFICATION"
    try:
        print(f"Recording read-only data to {writer.path}", flush=True)
        await recorder.run()
    except BaseException as exc:
        status = "FAILED_" + type(exc).__name__
        raise
    finally:
        await writer.close(status)
    if args.probe:
        print(f"Probe found {len(recorder.markets)} mapped markets; inspect raw statuses before use.")


if __name__ == "__main__":
    asyncio.run(main())
