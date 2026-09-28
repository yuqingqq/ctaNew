"""Binance spot + USDM perps (kline_1m streams).

Endpoints verified from this box 2026-09-28 (api/fapi REST return HTTP 451 here):
  spot REST   https://data-api.binance.vision/api/v3/...
  spot WS     wss://data-stream.binance.vision/stream?streams=<s>@kline_1m/...
  USDM WS     wss://fstream.binance.com/market/stream?streams=<s>@kline_1m/...
              Market-category streams (kline, !miniTicker@arr, !markPrice@arr@1s) push
              ONLY on the routed /market/ path; bare /stream?streams= ACKs and stays
              silent for them (same pitfall as data_collectors/binance_ws_collector.py).
USDM universe: fapi exchangeInfo is geo-blocked, so symbols come from the
!miniTicker@arr stream (every symbol that traded in a ~5 s window, with 24h quote
volume for the liquidity filter).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets

from ..instruments import make, split_concat_symbol
from .base import Feed, Shard, http_json

log = logging.getLogger("movewatch.binance")

SPOT_REST = "https://data-api.binance.vision/api/v3"
SPOT_WS = "wss://data-stream.binance.vision/stream?streams="
USDM_WS = "wss://fstream.binance.com/market/stream?streams="


class _BinanceKline(Feed):
    shard_size = 200
    ws_base = ""

    def build_shards(self) -> None:
        self.make_shards(self.ws_base)

    def url_for(self, shard: Shard) -> str:
        return self.ws_base + "/".join(f"{i.symbol.lower()}@kline_1m" for i in shard.insts)

    def handle(self, raw, shard: Shard) -> None:
        m = json.loads(raw)
        d = m.get("data")
        if not d or d.get("e") != "kline":
            return
        k = d["k"]
        idx = self.by_symbol.get(k["s"])
        if idx is not None:
            self.delta.update(self.store, idx, int(k["t"]), float(k["c"]), float(k["q"]))


class BinanceSpot(_BinanceKline):
    venue = "binance_spot"
    ws_base = SPOT_WS

    async def discover(self, session):
        min_qv = float(self.cfg.get("min_quote_volume_24h", 100_000))
        quotes = set(self.cfg.get("quotes", ["USDT", "USDC"]))
        info = await http_json(session, "GET", f"{SPOT_REST}/exchangeInfo", params={"permissions": "SPOT"})
        tick = await http_json(session, "GET", f"{SPOT_REST}/ticker/24hr", params={"type": "MINI"})
        qv = {t["symbol"]: float(t.get("quoteVolume") or 0) for t in tick}
        out = []
        for s in info["symbols"]:
            if s["status"] != "TRADING" or s["quoteAsset"] not in quotes:
                continue
            if qv.get(s["symbol"], 0.0) < min_qv:
                continue
            ins = make(self.venue, s["symbol"], s["baseAsset"], s["quoteAsset"], "spot")
            if ins:
                out.append(ins)
        return out


class BinanceUsdm(_BinanceKline):
    venue = "binance_usdm"
    ws_base = USDM_WS

    def __init__(self, store, cfg=None):
        super().__init__(store, cfg)
        self.funding: dict[str, float] = {}   # symbol -> current funding rate (fraction per interval)

    async def start(self, session):
        tasks = await super().start(session)
        tasks.append(asyncio.create_task(self._mark_price(), name="binance_usdm#markPrice"))
        return tasks

    async def _mark_price(self) -> None:
        """!markPrice@arr@1s: mark price + funding for every USDM symbol, one connection.
        Context only (alert text), never a trigger input."""
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(USDM_WS + "!markPrice@arr@1s", open_timeout=15,
                                              max_size=2 ** 24) as ws:
                    backoff = 1.0
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=60)
                        for x in json.loads(raw).get("data", []):
                            try:
                                self.funding[x["s"]] = float(x["r"])
                            except (KeyError, ValueError):
                                continue
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("binance_usdm markPrice: %s", str(e)[:150])
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    async def discover(self, session):
        min_qv = float(self.cfg.get("min_quote_volume_24h", 100_000))
        seen: dict[str, float] = {}
        t_end = time.time() + 6
        async with websockets.connect(USDM_WS + "!miniTicker@arr", open_timeout=15, max_size=2 ** 24) as ws:
            while time.time() < t_end:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, t_end - time.time()))
                except asyncio.TimeoutError:
                    break
                for t in json.loads(raw).get("data", []):
                    seen[t["s"]] = float(t.get("q") or 0)
        out = []
        for sym, q in sorted(seen.items()):
            sp = split_concat_symbol(sym, quotes=("USDT", "USDC"))
            if not sp or q < min_qv:
                continue
            ins = make(self.venue, sym, sp[0], sp[1], "perp")
            if ins:
                out.append(ins)
        if not out:
            raise RuntimeError("binance_usdm discovery saw no symbols on !miniTicker@arr")
        return out
