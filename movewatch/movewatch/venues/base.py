"""Shared websocket plumbing for venue feeds.

A Feed discovers instruments, splits them into shards (one WS connection each),
and runs every shard with: reconnect + exponential backoff, a silence watchdog
(no message for watchdog_s -> force reconnect), optional app-level pings, and
outage marking (store.set_down) so a reconnect never reads as a price jump.

Candle-style streams (cumulative volume within the minute) go through
CandleDelta, which turns them into (price, incremental quote volume) updates.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field

import aiohttp
import websockets

from ..instruments import Instrument
from ..store import BarStore

log = logging.getLogger("movewatch.venue")

HTTP_TIMEOUT = aiohttp.ClientTimeout(total=20)


@dataclass
class Shard:
    name: str
    url: str
    insts: list[Instrument]
    idx: list[int] = field(default_factory=list)
    down: bool = True
    msgs: int = 0
    reconnects: int = 0
    last_msg: float = 0.0
    connected_at: float = 0.0


class CandleDelta:
    """Per-instrument (minute, cumulative quote volume) -> incremental volume."""

    def __init__(self):
        self.state: dict[int, tuple[int, float]] = {}

    def update(self, store: BarStore, idx: int, minute_ms: int, close: float, cum_quote: float) -> None:
        prev = self.state.get(idx)
        if prev is None or minute_ms > prev[0]:
            dq = cum_quote if prev is not None else 0.0   # first sight: unknown prior part of minute
            self.state[idx] = (minute_ms, cum_quote)
            store.on_trade(idx, close, dq)
        elif minute_ms == prev[0]:
            dq = cum_quote - prev[1]
            if dq < 0:
                dq = 0.0
            self.state[idx] = (minute_ms, max(cum_quote, prev[1]))
            store.on_trade(idx, close, dq)
        # older minute (late final update): ignore -- its close is not the latest price


async def http_json(session: aiohttp.ClientSession, method: str, url: str, **kw):
    for k in range(4):
        try:
            async with session.request(method, url, timeout=HTTP_TIMEOUT, **kw) as r:
                if r.status == 429:
                    await asyncio.sleep(2 * (k + 1))
                    continue
                r.raise_for_status()
                return await r.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last = e
            await asyncio.sleep(1.5 * (k + 1))
    raise RuntimeError(f"{method} {url} failed: {last}")


class Feed:
    venue = "base"
    shard_size = 200
    ping_interval: float | None = None      # app-level ping cadence (s)
    watchdog_s = 60.0
    max_conn_age_s = 23 * 3600              # recycle before venue-side 24h caps

    def __init__(self, store: BarStore, cfg: dict | None = None):
        self.store = store
        self.cfg = cfg or {}
        self.shards: list[Shard] = []
        self.insts: list[Instrument] = []
        self.by_symbol: dict[str, int] = {}
        self.delta = CandleDelta()
        self._stop = asyncio.Event()

    # -------------------------------------------------------------- per-venue hooks
    async def discover(self, session: aiohttp.ClientSession) -> list[Instrument]:
        raise NotImplementedError

    def url_for(self, shard: Shard) -> str:
        return shard.url

    async def on_open(self, ws, shard: Shard) -> None:
        """Send subscriptions (URL-subscribed venues do nothing)."""

    def ping_payload(self) -> str | None:
        return None

    def handle(self, raw, shard: Shard) -> None:
        raise NotImplementedError

    # -------------------------------------------------------------- common
    def register(self, inst: Instrument) -> int:
        i = self.store.register(inst)
        self.by_symbol[inst.symbol] = i
        return i

    def make_shards(self, url: str) -> None:
        self.shards = []
        for k in range(0, len(self.insts), self.shard_size):
            chunk = self.insts[k: k + self.shard_size]
            sh = Shard(name=f"{self.venue}#{k // self.shard_size}", url=url, insts=chunk)
            sh.idx = [self.by_symbol[i.symbol] for i in chunk]
            self.shards.append(sh)

    async def start(self, session: aiohttp.ClientSession) -> list[asyncio.Task]:
        insts = await self.discover(session)
        self.insts = insts
        for ins in insts:
            self.register(ins)
        self.build_shards()
        log.info("%s: %d instruments in %d shards", self.venue, len(self.insts), len(self.shards))
        return [asyncio.create_task(self._run_shard(sh), name=sh.name) for sh in self.shards]

    def build_shards(self) -> None:
        raise NotImplementedError

    async def _run_shard(self, sh: Shard) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url_for(sh), open_timeout=15, close_timeout=2,
                                              ping_interval=20, ping_timeout=20,
                                              max_size=2 ** 24, max_queue=1024) as ws:
                    sh.connected_at = time.time()
                    await self.on_open(ws, sh)
                    pinger = None
                    if self.ping_interval and self.ping_payload():
                        pinger = asyncio.create_task(self._pinger(ws))
                    try:
                        while True:
                            timeout = self.watchdog_s
                            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                            sh.last_msg = time.time()
                            sh.msgs += 1
                            try:
                                self.handle(raw, sh)
                            except Exception:  # noqa: BLE001 -- one bad message never kills a shard
                                log.exception("%s: handler error on %r", sh.name, str(raw)[:200])
                            if sh.down:
                                sh.down = False
                                self.store.set_down(sh.idx, False)
                                backoff = 1.0
                            if time.time() - sh.connected_at > self.max_conn_age_s:
                                log.info("%s: recycling connection (age cap)", sh.name)
                                break
                    finally:
                        if pinger:
                            pinger.cancel()
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                log.warning("%s: silent for %.0fs, reconnecting", sh.name, self.watchdog_s)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: %s: %s", sh.name, type(e).__name__, str(e)[:200])
            sh.down = True
            sh.reconnects += 1
            self.store.set_down(sh.idx, True)
            await asyncio.sleep(backoff + random.random())
            backoff = min(backoff * 2, 60.0)

    async def _pinger(self, ws) -> None:
        while True:
            await asyncio.sleep(self.ping_interval)
            try:
                await ws.send(self.ping_payload())
            except Exception:  # noqa: BLE001
                return

    def stats(self) -> dict:
        now = time.time()
        return {
            "shards": len(self.shards),
            "up": sum(1 for s in self.shards if not s.down),
            "msgs": sum(s.msgs for s in self.shards),
            "reconnects": sum(s.reconnects for s in self.shards),
            "max_silence_s": round(max((now - s.last_msg for s in self.shards if s.last_msg), default=-1), 1),
            "instruments": len(self.insts),
        }


def loads(raw):
    return json.loads(raw)
