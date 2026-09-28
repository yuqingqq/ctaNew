"""Listing + exchange-flag watcher (REST polling).

Many sudden moves start with a venue event rather than a price pattern: a new
Upbit/Coinbase/Binance market, or an Upbit caution flag (the exchange's own
"abnormal move" detector). The first poll of every source only records a
baseline, so a restart never floods old markets as "new".

Sources (all reachable from this box, 2026-09-28):
  upbit        /v1/market/all?isDetails=true   new markets + market_event flags
  bithumb      /v1/market/all?isDetails=true   new markets + market_warning
  coinbase     /products                       new online products
  binance_spot data-api /api/v3/ticker/price   new symbols
  okx          /public/instruments (crypto only) new SPOT/SWAP instruments
"""
from __future__ import annotations

import asyncio
import logging
import time

from .notify import esc
from .venues.base import http_json

log = logging.getLogger("movewatch.watchers")

UPBIT_FLAGS = {"PRICE_FLUCTUATIONS": "price fluctuation", "TRADING_VOLUME_SOARING": "volume soaring",
               "DEPOSIT_AMOUNT_SOARING": "deposit soaring", "GLOBAL_PRICE_DIFFERENCES": "global price gap",
               "CONCENTRATION_OF_SMALL_ACCOUNTS": "small-account concentration",
               "WARNING": "investment WARNING (유의)"}


class ListingWatcher:
    def __init__(self, cfg: dict, engine):
        self.cfg = cfg or {}
        self.engine = engine
        self.interval = float(self.cfg.get("interval_s", 60))
        self.flag_cooldown = float(self.cfg.get("flag_cooldown_s", 6 * 3600))
        # Upbit flags are noisy on thin USDT/BTC books (GLOBAL_PRICE_DIFFERENCES was active on
        # 164 markets at once, 2026-09-28) and TRADING_VOLUME_SOARING duplicates our own volume
        # test, so by default: KRW markets only, and not the volume flag.
        self.korea_quotes = set(self.cfg.get("korea_quotes", ["KRW"]))
        self.upbit_flags = set(self.cfg.get("upbit_flags", ["PRICE_FLUCTUATIONS", "DEPOSIT_AMOUNT_SOARING",
                                                            "GLOBAL_PRICE_DIFFERENCES", "WARNING"]))
        self.known: dict[str, set] = {}
        self.flags: dict[str, set] = {}
        self._flag_last: dict[tuple[str, str], float] = {}
        self.polls = 0

    async def run(self, session) -> None:
        while True:
            for name, fn in (("upbit", self._upbit), ("bithumb", self._bithumb), ("coinbase", self._coinbase),
                             ("binance_spot", self._binance), ("okx", self._okx)):
                if name == "okx" and self.polls % 3:
                    continue
                try:
                    await fn(session)
                except Exception as e:  # noqa: BLE001 -- a watcher source failing is logged, never fatal
                    log.warning("watcher %s: %s", name, str(e)[:200])
            self.polls += 1
            await asyncio.sleep(self.interval)

    # ------------------------------------------------------------------ helpers
    def _diff(self, source: str, ids: set) -> set:
        prev = self.known.get(source)
        self.known[source] = ids
        if prev is None:
            log.info("watcher %s: baseline %d ids", source, len(ids))
            return set()
        return ids - prev

    @staticmethod
    def _key(venue: str, market: str) -> str | None:
        v = {"Upbit": "upbit", "Bithumb": "bithumb", "Coinbase": "coinbase", "Binance spot": "binance_spot"}.get(venue)
        if venue == "OKX":
            v = "okx_swap" if market.endswith("-SWAP") else "okx_spot"
        return f"{v}:{market}" if v else None

    def _listing(self, venue: str, market: str, base: str) -> None:
        ctx = self.engine.describe(base, self._key(venue, market))
        text = f"🆕 <b>{esc(base)}</b> new market on {esc(venue)}: {esc(market)}\n{ctx}"
        self.engine.notifier.emit(text, {"kind": "LISTING", "venue": venue, "market": market, "asset": base})

    def _flag(self, venue: str, market: str, base: str, flag: str) -> None:
        now = time.time()
        k = (market, flag)
        if now - self._flag_last.get(k, 0) < self.flag_cooldown:
            return
        self._flag_last[k] = now
        ctx = self.engine.describe(base, self._key(venue, market))
        text = f"⚠️ <b>{esc(base)}</b> {esc(venue)} flag: {esc(UPBIT_FLAGS.get(flag, flag))} ({esc(market)})\n{ctx}"
        self.engine.notifier.emit(text, {"kind": "FLAG", "venue": venue, "market": market, "flag": flag,
                                         "asset": base})

    # ------------------------------------------------------------------ sources
    async def _upbit(self, session) -> None:
        mk = await http_json(session, "GET", "https://api.upbit.com/v1/market/all", params={"isDetails": "true"})
        ids = {m["market"] for m in mk if m["market"].split("-", 1)[0] in self.korea_quotes}
        for market in sorted(self._diff("upbit", ids)):
            self._listing("Upbit", market, market.split("-", 1)[1])
        first = "upbit_flags" not in self.known
        for m in mk:
            if m["market"].split("-", 1)[0] not in self.korea_quotes:
                continue
            ev = m.get("market_event") or {}
            active = {k for k, v in (ev.get("caution") or {}).items() if v and k in self.upbit_flags}
            if ev.get("warning") and "WARNING" in self.upbit_flags:
                active.add("WARNING")
            prev = self.flags.get(m["market"], set())
            self.flags[m["market"]] = active
            if first:
                continue
            for f in sorted(active - prev):
                self._flag("Upbit", m["market"], m["market"].split("-", 1)[1], f)
        self.known["upbit_flags"] = {"ok"}

    async def _bithumb(self, session) -> None:
        mk = await http_json(session, "GET", "https://api.bithumb.com/v1/market/all", params={"isDetails": "true"})
        mk = [m for m in mk if m["market"].split("-", 1)[0] in self.korea_quotes]
        ids = {m["market"] for m in mk}
        for market in sorted(self._diff("bithumb", ids)):
            self._listing("Bithumb", market, market.split("-", 1)[1])
        first = "bithumb_flags" not in self.known
        for m in mk:
            w = m.get("market_warning")
            active = {f"WARNING:{w}"} if w and w not in ("NONE",) else set()
            key = "bithumb:" + m["market"]
            prev = self.flags.get(key, set())
            self.flags[key] = active
            if not first:
                for f in sorted(active - prev):
                    self._flag("Bithumb", m["market"], m["market"].split("-", 1)[1], f)
        self.known["bithumb_flags"] = {"ok"}

    async def _coinbase(self, session) -> None:
        ps = await http_json(session, "GET", "https://api.exchange.coinbase.com/products")
        on = {p["id"]: p["base_currency"] for p in ps if p.get("status") == "online"}
        for pid in sorted(self._diff("coinbase", set(on))):
            self._listing("Coinbase", pid, on[pid])

    async def _binance(self, session) -> None:
        px = await http_json(session, "GET", "https://data-api.binance.vision/api/v3/ticker/price")
        syms = {p["symbol"] for p in px if p["symbol"].endswith(("USDT", "USDC"))}
        for s in sorted(self._diff("binance_spot", syms)):
            base = s[:-4]
            self._listing("Binance spot", s, base)

    async def _okx(self, session) -> None:
        ids = {}
        for t in ("SPOT", "SWAP"):
            d = (await http_json(session, "GET", "https://www.okx.com/api/v5/public/instruments",
                                 params={"instType": t}))["data"]
            for x in d:
                if x.get("instCategory") == "1" and x.get("state") in ("live", "preopen"):
                    ids[x["instId"]] = x["instId"].split("-")[0]
        for i in sorted(self._diff("okx", set(ids))):
            self._listing("OKX", i, ids[i])
