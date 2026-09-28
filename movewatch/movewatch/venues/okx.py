"""OKX spot + USDT-margined swaps (candle1m on the business endpoint).

instCategory filter is mandatory: OKX lists equity perps/spot under crypto-looking
tickers (QNT-USDT-SWAP is an equity perp, ~$48 when Quant was ~$220, 2026-09-28).
Only instCategory == "1" (crypto) is kept.
Candle row: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]; volCcyQuote is the
quote-currency volume for both SPOT and SWAP.
"""
from __future__ import annotations

import json
import logging

from ..instruments import make
from .base import Feed, Shard, http_json

log = logging.getLogger("movewatch.okx")

REST = "https://www.okx.com/api/v5"
WS = "wss://ws.okx.com:8443/ws/v5/business"


class _Okx(Feed):
    shard_size = 100
    ping_interval = 20.0
    inst_type = "SPOT"

    def ping_payload(self):
        return "ping"

    def build_shards(self) -> None:
        self.make_shards(WS)

    async def discover(self, session):
        min_qv = float(self.cfg.get("min_quote_volume_24h", 100_000))
        ins = (await http_json(session, "GET", f"{REST}/public/instruments", params={"instType": self.inst_type}))["data"]
        tk = (await http_json(session, "GET", f"{REST}/market/tickers", params={"instType": self.inst_type}))["data"]
        qv = {}
        for t in tk:
            last = float(t.get("last") or 0)
            vc = float(t.get("volCcy24h") or 0)
            qv[t["instId"]] = vc if self.inst_type == "SPOT" else vc * last  # SWAP volCcy24h is in base
        out = []
        for x in ins:
            if x.get("state") != "live" or x.get("instCategory") != "1":
                continue
            if qv.get(x["instId"], 0.0) < min_qv:
                continue
            if self.inst_type == "SPOT":
                if x["quoteCcy"] not in ("USDT", "USDC", "USD"):
                    continue
                i = make(self.venue, x["instId"], x["baseCcy"], x["quoteCcy"], "spot")
            else:
                if x.get("settleCcy") != "USDT" or x.get("ctType") != "linear":
                    continue
                base = x["instFamily"].split("-")[0]
                i = make(self.venue, x["instId"], base, "USDT", "perp")
            if i:
                out.append(i)
        return out

    async def on_open(self, ws, shard: Shard) -> None:
        args = [{"channel": "candle1m", "instId": i.symbol} for i in shard.insts]
        await ws.send(json.dumps({"op": "subscribe", "args": args}))

    def handle(self, raw, shard: Shard) -> None:
        if raw == "pong":
            return
        m = json.loads(raw)
        data = m.get("data")
        if not data:
            if m.get("event") == "error":
                log.warning("%s: %s", shard.name, str(m)[:200])
            return
        idx = self.by_symbol.get(m["arg"]["instId"])
        if idx is None:
            return
        for k in data:
            self.delta.update(self.store, idx, int(k[0]), float(k[4]), float(k[7]))


class OkxSpot(_Okx):
    venue = "okx_spot"
    inst_type = "SPOT"


class OkxSwap(_Okx):
    venue = "okx_swap"
    inst_type = "SWAP"
