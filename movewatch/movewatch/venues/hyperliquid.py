"""Hyperliquid perps (candle 1m). Candle volume `v` is base units; quote volume is
accumulated as delta(v) * close. kPEPE-style names quote 1000 units."""
from __future__ import annotations

import asyncio
import json
import logging

from ..instruments import make
from .base import Feed, Shard, http_json

log = logging.getLogger("movewatch.hyperliquid")

INFO = "https://api.hyperliquid.xyz/info"
WS = "wss://api.hyperliquid.xyz/ws"


class Hyperliquid(Feed):
    venue = "hyperliquid"
    shard_size = 100
    ping_interval = 50.0

    def __init__(self, store, cfg=None):
        super().__init__(store, cfg)
        self._vol: dict[int, tuple[int, float]] = {}

    def ping_payload(self):
        return '{"method":"ping"}'

    def build_shards(self) -> None:
        self.make_shards(WS)

    async def discover(self, session):
        min_qv = float(self.cfg.get("min_quote_volume_24h", 100_000))
        meta, ctxs = await http_json(session, "POST", INFO, json={"type": "metaAndAssetCtxs"})
        out = []
        for u, c in zip(meta["universe"], ctxs):
            if u.get("isDelisted") or float(c.get("dayNtlVlm") or 0) < min_qv:
                continue
            i = make(self.venue, u["name"], u["name"], "USD", "perp")
            if i:
                out.append(i)
        return out

    async def on_open(self, ws, shard: Shard) -> None:
        for i in shard.insts:
            await ws.send(json.dumps({"method": "subscribe",
                                      "subscription": {"type": "candle", "coin": i.symbol, "interval": "1m"}}))
            await asyncio.sleep(0.01)

    def handle(self, raw, shard: Shard) -> None:
        m = json.loads(raw)
        if m.get("channel") != "candle":
            return
        d = m["data"]
        idx = self.by_symbol.get(d["s"])
        if idx is None:
            return
        t, c, v = int(d["t"]), float(d["c"]), float(d["v"])
        prev = self._vol.get(idx)
        if prev is None:
            dv = 0.0
        elif t > prev[0]:
            dv = v
        elif t == prev[0]:
            dv = max(0.0, v - prev[1])
        else:
            return
        self._vol[idx] = (t, v if prev is None or t >= prev[0] else prev[1])
        self.store.on_trade(idx, c, dv * c)
