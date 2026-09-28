"""Coinbase Exchange spot (matches channel = every trade)."""
from __future__ import annotations

import json
import logging

from ..instruments import make
from .base import Feed, Shard, http_json

log = logging.getLogger("movewatch.coinbase")

REST = "https://api.exchange.coinbase.com"
WS = "wss://ws-feed.exchange.coinbase.com"


class Coinbase(Feed):
    venue = "coinbase"
    shard_size = 150

    def build_shards(self) -> None:
        self.make_shards(WS)

    async def discover(self, session):
        quotes = set(self.cfg.get("quotes", ["USD", "USDT", "USDC"]))
        prods = await http_json(session, "GET", f"{REST}/products")
        out = []
        for p in prods:
            if p.get("status") != "online" or p.get("trading_disabled") or p.get("quote_currency") not in quotes:
                continue
            i = make(self.venue, p["id"], p["base_currency"], p["quote_currency"], "spot")
            if i:
                out.append(i)
        return out

    async def on_open(self, ws, shard: Shard) -> None:
        await ws.send(json.dumps({"type": "subscribe", "product_ids": [i.symbol for i in shard.insts],
                                  "channels": ["matches"]}))

    def handle(self, raw, shard: Shard) -> None:
        m = json.loads(raw)
        t = m.get("type")
        if t == "match":
            idx = self.by_symbol.get(m["product_id"])
            if idx is not None:
                px = float(m["price"])
                self.store.on_trade(idx, px, px * float(m["size"]))
        elif t == "error":
            log.warning("%s: %s", shard.name, str(m)[:200])
