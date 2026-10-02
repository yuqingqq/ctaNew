"""Binance spot + USDM perps (kline_1m streams).

Endpoints verified from this box 2026-09-28 (api/fapi REST return HTTP 451 here):
  spot REST   https://data-api.binance.vision/api/v3/...
  spot WS     wss://data-stream.binance.vision/stream?streams=<s>@kline_1m/...
  USDM WS     wss://fstream.binance.com/market/stream?streams=<s>@kline_1m/...
              Market-category streams (kline, !miniTicker@arr, !markPrice@arr@1s) push
              ONLY on the routed /market/ path; bare /stream?streams= ACKs and stays
              silent for them (same pitfall as data_collectors/binance_ws_collector.py).
USDM universe: fapi exchangeInfo is geo-blocked. !markPrice@arr@1s carries EVERY perp
each second, so it is the universe; !miniTicker@arr carries only symbols that ticked in
that second, so it can only DROP a symbol it shows below the volume floor. (Universe from a
6 s miniTicker window alone: 347-574 instruments per restart, 2026-09-28..10-01.)
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


async def _collect(url: str, secs: float) -> dict[str, dict]:
    """{symbol: last record} from an array stream (one dict per symbol per frame) for `secs`."""
    seen: dict[str, dict] = {}
    t_end = time.time() + secs
    async with websockets.connect(url, open_timeout=15, max_size=2 ** 24) as ws:
        while time.time() < t_end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, t_end - time.time()))
            except asyncio.TimeoutError:
                break
            for x in json.loads(raw).get("data", []):
                seen[x["s"]] = x
    return seen


def _settled(x: dict | None) -> bool:
    """markPrice keeps publishing delisted contracts: mark == index exactly and funding exactly 0
    (AGIX, AMB, ALPHA, BADGER on 2026-10-02), or no next funding time. Live perps carry a
    mark/index gap and nonzero funding. Applied only to symbols that did not trade in the window."""
    if not x:
        return True
    try:
        return int(x.get("T") or 0) == 0 or (float(x["r"]) == 0.0 and float(x["p"]) == float(x["i"]))
    except (KeyError, ValueError):
        return False


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
        mark, vol = await asyncio.gather(_collect(USDM_WS + "!markPrice@arr@1s", 6.0),
                                         _collect(USDM_WS + "!miniTicker@arr", 10.0))
        if not mark:
            raise RuntimeError("binance_usdm discovery saw no symbols on !markPrice@arr@1s")
        out, low, unseen, dead = [], 0, 0, 0
        for sym in sorted(set(mark) | set(vol)):
            sp = split_concat_symbol(sym, quotes=("USDT", "USDC"))
            if not sp:
                continue
            q = vol.get(sym)
            if q is not None and float(q.get("q") or 0) < min_qv:
                low += 1
                continue
            if q is None and _settled(mark.get(sym)):
                dead += 1
                continue
            unseen += q is None
            ins = make(self.venue, sym, sp[0], sp[1], "perp")
            if ins:
                out.append(ins)
        log.info("binance_usdm universe: %d on markPrice, %d kept (%d not in the 10s ticker window), %d below $%.0f, "
                 "%d settled", len(mark), len(out), unseen, low, min_qv, dead)
        return out
