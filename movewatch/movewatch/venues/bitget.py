"""Bitget spot + USDT-FUTURES (candle1m).

The subscribe snapshot carries ~500 recent 1m candles per instrument; they are
used to backfill the minute ring (warm-up after a restart). Row layout for both
products: [ts, o, h, l, c, baseVol, quoteVol, usdtVol].
"""
from __future__ import annotations

import asyncio
import json
import logging

from ..instruments import make
from .base import Feed, Shard, http_json

log = logging.getLogger("movewatch.bitget")

REST = "https://api.bitget.com/api/v2"
WS = "wss://ws.bitget.com/v2/ws/public"


class _Bitget(Feed):
    shard_size = 100
    ping_interval = 25.0
    inst_type = "SPOT"

    def ping_payload(self):
        return "ping"

    def build_shards(self) -> None:
        self.make_shards(WS)

    async def discover(self, session):
        min_qv = float(self.cfg.get("min_quote_volume_24h", 250_000))
        out = []
        if self.inst_type == "SPOT":
            syms = (await http_json(session, "GET", f"{REST}/spot/public/symbols"))["data"]
            tk = (await http_json(session, "GET", f"{REST}/spot/market/tickers"))["data"]
            qv = {t["symbol"]: float(t.get("usdtVolume") or t.get("quoteVolume") or 0) for t in tk}
            for s in syms:
                if s.get("status") != "online" or s["quoteCoin"] not in ("USDT", "USDC"):
                    continue
                if qv.get(s["symbol"], 0.0) < min_qv:
                    continue
                i = make(self.venue, s["symbol"], s["baseCoin"], s["quoteCoin"], "spot")
                if i:
                    out.append(i)
        else:
            cs = (await http_json(session, "GET", f"{REST}/mix/market/contracts",
                                  params={"productType": "USDT-FUTURES"}))["data"]
            tk = (await http_json(session, "GET", f"{REST}/mix/market/tickers",
                                  params={"productType": "USDT-FUTURES"}))["data"]
            qv = {t["symbol"]: float(t.get("usdtVolume") or t.get("quoteVolume") or 0) for t in tk}
            for c in cs:
                if c.get("symbolStatus") != "normal":
                    continue
                if qv.get(c["symbol"], 0.0) < min_qv:
                    continue
                i = make(self.venue, c["symbol"], c["baseCoin"], c["quoteCoin"], "perp")
                if i:
                    out.append(i)
        return out

    async def on_open(self, ws, shard: Shard) -> None:
        args = [{"instType": self.inst_type, "channel": "candle1m", "instId": i.symbol} for i in shard.insts]
        for k in range(0, len(args), 50):
            await ws.send(json.dumps({"op": "subscribe", "args": args[k: k + 50]}))
            await asyncio.sleep(0.2)

    def handle(self, raw, shard: Shard) -> None:
        if raw == "pong":
            return
        m = json.loads(raw)
        act = m.get("action")
        if act not in ("snapshot", "update"):
            if m.get("event") == "error":
                log.warning("%s: %s", shard.name, str(m)[:200])
            return
        idx = self.by_symbol.get(m["arg"]["instId"])
        if idx is None:
            return
        rows = sorted(m["data"], key=lambda k: int(k[0]))
        if act == "snapshot" and len(rows) > 1:
            for k in rows[:-1]:
                self.store.backfill_minute(idx, int(k[0]), float(k[4]), float(k[6]))
            rows = rows[-1:]
        for k in rows:
            self.delta.update(self.store, idx, int(k[0]), float(k[4]), float(k[6]))


class BitgetSpot(_Bitget):
    venue = "bitget_spot"
    inst_type = "SPOT"


class BitgetFut(_Bitget):
    venue = "bitget_fut"
    inst_type = "USDT-FUTURES"
