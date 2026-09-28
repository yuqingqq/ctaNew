"""Bybit linear perps + spot (kline.1 topics).

REST (api.bybit.com, api.bytick.com) returns 403 from this box, so the
instrument list cannot be fetched. Discovery is by subscription instead:
candidates are subscribed ONE topic per message (a batch containing a single
unknown symbol fails as a whole: "handler not found"), and an instrument is
registered on its first kline message. Candidates:
  linear: public.bybit.com/trading/ archive listing (includes delisted names)
  spot:   the Binance-spot USDT list passed in by the engine, plus linear bases
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from ..instruments import make, split_concat_symbol
from .base import Feed, Shard

log = logging.getLogger("movewatch.bybit")

WS = {"bybit_linear": "wss://stream.bybit.com/v5/public/linear",
      "bybit_spot": "wss://stream.bybit.com/v5/public/spot"}


class _Bybit(Feed):
    shard_size = 190
    ping_interval = 20.0
    kind = "perp"

    def __init__(self, store, cfg=None, candidates: list[str] | None = None):
        super().__init__(store, cfg)
        self.candidates = candidates or []
        self.rejected: set[str] = set()

    def ping_payload(self):
        return '{"op":"ping"}'

    def build_shards(self) -> None:
        self.shards = []
        c = sorted(set(self.candidates))
        for k in range(0, len(c), self.shard_size):
            sh = Shard(name=f"{self.venue}#{k // self.shard_size}", url=WS[self.venue], insts=[])
            sh.symbols = c[k: k + self.shard_size]  # type: ignore[attr-defined]
            self.shards.append(sh)

    async def start(self, session):
        self.insts = []
        self.build_shards()
        log.info("%s: %d candidate symbols in %d shards", self.venue, len(self.candidates), len(self.shards))
        return [asyncio.create_task(self._run_shard(sh), name=sh.name) for sh in self.shards]

    async def on_open(self, ws, shard: Shard) -> None:
        for s in shard.symbols:  # type: ignore[attr-defined]
            if s in self.rejected:
                continue
            await ws.send(json.dumps({"op": "subscribe", "req_id": s, "args": [f"kline.1.{s}"]}))
            await asyncio.sleep(0.005)

    def handle(self, raw, shard: Shard) -> None:
        m = json.loads(raw)
        if "topic" not in m:
            if m.get("op") == "subscribe" and m.get("success") is False and m.get("req_id"):
                self.rejected.add(m["req_id"])
            return
        sym = m["topic"].rsplit(".", 1)[-1]
        idx = self.by_symbol.get(sym)
        if idx is None:
            sp = split_concat_symbol(sym, quotes=("USDT", "USDC"))
            ins = make(self.venue, sym, sp[0], sp[1], self.kind) if sp else None
            if ins is None:
                return
            idx = self.register(ins)
            self.insts.append(ins)
            shard.idx.append(idx)
        for k in m.get("data", []):
            self.delta.update(self.store, idx, int(k["start"]), float(k["close"]), float(k["turnover"]))


class BybitLinear(_Bybit):
    venue = "bybit_linear"
    kind = "perp"


class BybitSpot(_Bybit):
    venue = "bybit_spot"
    kind = "spot"


async def linear_archive_candidates(session) -> list[str]:
    async with session.get("https://public.bybit.com/trading/") as r:
        txt = await r.text()
    return re.findall(r'href="([A-Z0-9]+USDT)/"', txt)
