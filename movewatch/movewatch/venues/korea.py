"""Upbit + Bithumb KRW markets (trade stream, Upbit-compatible v1 protocol).

KRW-USDT prints set the live KRW/USD rate (store.fx) used for USD volumes and the
cross-venue price-consistency check. Korean venues are where many alt moves start
(retail flow, listings, deposit/withdrawal suspensions), and a KRW-only move with no
global confirmation is itself informative (premium / wallet-suspension signature).
"""
from __future__ import annotations

import json
import logging
import uuid

from ..instruments import make
from .base import Feed, Shard, http_json

log = logging.getLogger("movewatch.korea")


class _UpbitLike(Feed):
    shard_size = 400
    rest = ""
    ws = ""

    def build_shards(self) -> None:
        self.make_shards(self.ws)

    async def discover(self, session):
        mk = await http_json(session, "GET", self.rest, params={"isDetails": "false"})
        out = []
        for m in mk:
            q, b = m["market"].split("-", 1)
            if q != "KRW":
                continue
            i = make(self.venue, m["market"], b, "KRW", "spot")
            if i:
                out.append(i)
        return out

    async def on_open(self, ws, shard: Shard) -> None:
        codes = [i.symbol for i in shard.insts] + ["KRW-USDT"]
        await ws.send(json.dumps([{"ticket": f"movewatch-{uuid.uuid4().hex[:8]}"},
                                  {"type": "trade", "codes": codes}, {"format": "SIMPLE"}]))

    def handle(self, raw, shard: Shard) -> None:
        m = json.loads(raw)
        if m.get("ty") != "trade":
            return
        cd, px, qty = m["cd"], float(m["tp"]), float(m["tv"])
        if cd == "KRW-USDT":
            if self.venue == "upbit" or not self.store.fx.krw_live:
                self.store.fx.set_krw(px)
            return
        if m.get("st") == "SNAPSHOT":  # Bithumb replays the last trade on subscribe
            idx = self.by_symbol.get(cd)
            if idx is not None:
                self.store.on_trade(idx, px, 0.0)
            return
        idx = self.by_symbol.get(cd)
        if idx is not None:
            self.store.on_trade(idx, px, px * qty, int(m.get("ttms") or 0) or None)


class Upbit(_UpbitLike):
    venue = "upbit"
    rest = "https://api.upbit.com/v1/market/all"
    ws = "wss://api.upbit.com/websocket/v1"


class Bithumb(_UpbitLike):
    venue = "bithumb"
    rest = "https://api.bithumb.com/v1/market/all"
    ws = "wss://ws-api.bithumb.com/websocket/v1"
