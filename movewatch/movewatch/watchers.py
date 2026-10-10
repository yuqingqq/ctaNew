"""Listing + exchange-flag watcher (REST polling).

Many sudden moves start with a venue event rather than a price pattern: a new
Upbit/Coinbase/Binance market, or an Upbit caution flag (the exchange's own
"abnormal move" detector). The first poll of every source only records a
baseline, so a restart never floods old markets as "new".

Also the extreme-crash trigger (docs/EXTREME_CRASH.md, validated 2026-10-09): CrashWatcher reads
Binance USDM !miniTicker@arr and alerts when >= half the liquid perps fell >= 10% in 15 minutes.
Also FadeCostWatcher (docs/FORWARD_FADE_F1.md, descriptive): at each Binance-priced pump's +4 h and
+24 h follow-up record it snapshots the public order book and funding, so forward test F1 can
compare the assumed cost with the book. Silent: its own records, no messages.
This file is outside the forward-test fingerprint (followup.FINGERPRINT_SKIP_FILES).

Sources (all reachable from this box, 2026-09-28):
  upbit        /v1/market/all?isDetails=true   new markets + market_event flags
  bithumb      /v1/market/all?isDetails=true   new markets + market_warning
  coinbase     /products                       new online products
  binance_spot data-api /api/v3/ticker/price   new symbols
  okx          /public/instruments (crypto only) new SPOT/SWAP instruments
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

from .notify import esc
from .tradfi import STATIC_BINANCE, is_tradfi
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
        crash = self.cfg.get("crash", {})
        if crash.get("enabled", False):                     # own task; a failure there never stops listings
            self._crash_task = asyncio.create_task(CrashWatcher(crash, self.engine).run(), name="crash_watcher")
        fade = self.cfg.get("fade_cost", {})
        if fade.get("enabled", False):
            self._fade_task = asyncio.create_task(FadeCostWatcher(fade, self.engine).run(session), name="fade_cost")
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


# ---------------------------------------------------------------------- extreme-crash trigger
USDM_MINI = "wss://fstream.binance.com/market/stream?streams=!miniTicker@arr"
STABLE_BASES = {"USDC", "FDUSD", "TUSD", "USDP", "BUSD", "DAI", "USDE", "EUR", "USD1", "XUSD", "BFUSD", "PAXG",
                "XAUT", "RLUSD"}                                  # research.replay.STABLES
INDEX_BASES = {"BTCDOM", "DEFI", "BLUEBIRD", "FOOTBALL"}          # research.flush_signal.INDEX


class CrashDetector:
    """docs/EXTREME_CRASH.md trigger on minute closes. Universe: Binance USDT perps (no stables, index or
    stock perps, no dated contracts) with 24 h quote volume >= liq_min and a trade in the last `window`
    minutes; >= n_min coins. Fires when >= `share` of them fell >= `depth` over `window` minutes, at most
    once per `refractory_s`. The basket is the hardest-hit fifth."""

    def __init__(self, depth: float = 0.10, share: float = 0.5, liq_min: float = 5e6, n_min: int = 50,
                 window: int = 15, refractory_s: float = 86400.0, excluded=lambda base, px: False):
        self.depth, self.share, self.liq_min, self.n_min, self.window = depth, share, liq_min, n_min, window
        self.refractory_s, self.excluded = refractory_s, excluded
        self.px: dict[str, float] = {}
        self.qv: dict[str, float] = {}
        self.last_tick: dict[str, float] = {}
        self.closes: dict[int, dict[str, float]] = {}
        self.last_fire = -math.inf

    def tick(self, sym: str, price: float, qv: float, ts: float) -> None:
        self.px[sym], self.qv[sym], self.last_tick[sym] = price, qv, ts

    def eligible(self, sym: str, px: float) -> bool:
        if not sym.endswith("USDT") or "_" in sym:
            return False
        base = sym[:-4]
        return base not in STABLE_BASES and base not in INDEX_BASES and base not in STATIC_BINANCE \
            and not self.excluded(base, px)

    def close_minute(self, minute: int) -> dict:
        """Record the closes of epoch-minute `minute` (just ended) and evaluate the trigger."""
        self.closes[minute] = dict(self.px)
        for m in [m for m in self.closes if m < minute - self.window]:
            del self.closes[m]
        prev = self.closes.get(minute - self.window)
        if prev is None:
            return {"status": "NO_HISTORY", "minute": minute}
        start = (minute - self.window + 1) * 60                 # first second of the last `window` minutes
        rows = []
        for s, p in self.closes[minute].items():
            p0 = prev.get(s)
            if (self.qv.get(s, 0.0) < self.liq_min or not p0 or p0 <= 0 or p <= 0 or not self.eligible(s, p)
                    or self.last_tick.get(s, -math.inf) < start):
                continue
            rows.append((s, math.log(p / p0)))
        n = len(rows)
        if n < self.n_min:
            return {"status": "THIN_UNIVERSE", "minute": minute, "n": n}
        rows.sort(key=lambda x: x[1])
        r = [x for _, x in rows]
        med = (r[(n - 1) // 2] + r[n // 2]) / 2
        info = {"status": "OK", "minute": minute, "n": n, "median": med, "fire": False,
                **{f"share{int(round(d * 100))}": sum(x <= math.log(1 - d) for x in r) / n for d in (0.03, 0.05, self.depth)}}
        share = info[f"share{int(round(self.depth * 100))}"]
        if share >= self.share and minute * 60 - self.last_fire >= self.refractory_s:
            self.last_fire = minute * 60
            info.update(fire=True, share=share, basket=rows[:math.ceil(n / 5)])
        return info


class CrashWatcher:
    """Streams Binance USDM !miniTicker@arr, closes each minute on the first message of the next one,
    and alerts on the extreme-crash trigger, with a follow-up at the 4 h exit. Alert only: it never
    trades. Each minute of delay cost ~1-2 points in the execution stress test."""

    def __init__(self, cfg: dict, engine, url: str = USDM_MINI):
        self.cfg, self.engine, self.url = cfg or {}, engine, url
        self.hold_min = int(self.cfg.get("hold_min", 240))
        self.det = CrashDetector(depth=float(self.cfg.get("depth", 0.10)), share=float(self.cfg.get("share", 0.5)),
                                 liq_min=float(self.cfg.get("liq_min", 5e6)), n_min=int(self.cfg.get("n_min", 50)),
                                 window=int(self.cfg.get("window_min", 15)), excluded=self._excluded)
        self.minute: int | None = None
        self.pending: list[dict] = []
        self._health = 0.0

    def _excluded(self, base: str, px: float) -> bool:
        """Stock/commodity perps: the engine's cluster verdict, or the engine's rule on this symbol's price."""
        return base in getattr(self.engine, "tradfi_assets", ()) or \
            is_tradfi(base, px, getattr(self.engine, "tradfi_ref", {}) or {}, has_binance_usdm=True)

    async def run(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.url, open_timeout=15, max_size=2 ** 24) as ws:
                    log.info("crash watcher: connected")
                    backoff = 1.0
                    while True:
                        self.on_message(json.loads(await asyncio.wait_for(ws.recv(), timeout=60)))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 -- reconnect; minutes missed during the gap read NO_HISTORY
                log.warning("crash watcher: %s", str(e)[:150])
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    def on_message(self, msg: dict) -> None:
        data = msg.get("data") or []
        if not data:
            return
        m = int(max(d.get("E", 0) for d in data) / 1000 // 60)
        if self.minute is None:
            self.minute = m
        elif m > self.minute:
            self.on_close(self.minute)
            self.minute = m
        for d in data:
            try:
                self.det.tick(d["s"], float(d["c"]), float(d["q"]), d["E"] / 1000)
            except (KeyError, TypeError, ValueError):
                continue

    def on_close(self, minute: int) -> None:
        info = self.det.close_minute(minute)
        for p in self.pending:
            if p.get("entry") is None and minute >= p["minute"] + 1:   # E1 entry = close of t + 1
                p["entry"] = {s: self.det.closes[p["minute"] + 1].get(s) for s, _ in p["basket"]} \
                    if p["minute"] + 1 in self.det.closes else {s: self.det.px.get(s) for s, _ in p["basket"]}
        for p in [p for p in self.pending if minute >= p["minute"] + 1 + self.hold_min]:
            self.pending.remove(p)
            self._follow_up(p)
        now = time.time()
        if info.get("status") == "OK" and info["share3"] >= 0.5:
            k = f"share{int(round(self.det.depth * 100))}"
            log.info("crash watch %s: median %+.1f%%, %s %.2f, n %d", _hhmm(minute), 100 * info["median"], k,
                     info[k], info["n"])
        if now - self._health >= 3600:
            self._health = now
            log.info("crash watcher health: %s n=%s median15=%s", info.get("status"), info.get("n"),
                     f"{100 * info['median']:+.2f}%" if "median" in info else "-")
        if info.get("fire"):
            self._alert(info)

    def _alert(self, info: dict) -> None:
        exit_m = info["minute"] + 1 + self.hold_min
        basket = info["basket"]
        coins = ", ".join(f"{esc(s[:-4])} {100 * (math.exp(x) - 1):+.0f}%" for s, x in basket)
        text = (f"🚨 <b>EXTREME CRASH</b>: {round(info['share'] * info['n'])} of {info['n']} Binance perps down "
                f"≥{100 * self.det.depth:.0f}% "
                f"in 15 min (median {100 * (math.exp(info['median']) - 1):+.1f}%)\n"
                f"Rule (docs/EXTREME_CRASH.md, validated): BUY the {len(basket)} hardest-hit NOW, equal weight, "
                f"NO leverage, NO stop. SELL all at {_hhmm(exit_m)} UTC (4 h).\n"
                f"Basket: {coins}\n"
                f"History: +8.6%/trade, 12/14 on untouched 2020-21. Each minute of delay costs ~1-2 pts; "
                f"baskets have fallen up to 52% before recovering.")
        self.engine.notifier.emit(text, {"kind": "CRASH", "minute": info["minute"], "n": info["n"],
                                         "share": info["share"], "median": info["median"],
                                         "basket": [[s, x] for s, x in basket], "exit_minute": exit_m})
        self.pending.append({"minute": info["minute"], "basket": basket, "entry": None})

    def _follow_up(self, p: dict) -> None:
        entry = p.get("entry") or {}
        rs = [self.det.px[s] / e - 1 for s, e in entry.items() if e and s in self.det.px]
        g = sum(rs) / len(rs) if rs else float("nan")
        text = (f"🚨 EXTREME CRASH follow-up ({_hhmm(p['minute'])} trigger): the basket is {100 * g:+.1f}% gross "
                f"since the 1-minute entry ({len(rs)} coins). Rule: sell now.")
        self.engine.notifier.emit(text, {"kind": "CRASH_FOLLOWUP", "minute": p["minute"], "gross": g, "coins": len(rs)})


FAPI = "https://fapi.binance.com/fapi/v1"


def book_cost(bids: list, asks: list, notionals: list[float]) -> dict:
    """Spread and walk-the-book cost vs mid, in bps, for a market SELL (into bids) and BUY (into
    asks) of each USD notional. None where the snapshot's book is too thin to fill it. Also the
    USD depth within 0.5 / 1 / 2% of mid on each side."""
    b = [(float(p), float(q)) for p, q in bids if float(q) > 0]
    a = [(float(p), float(q)) for p, q in asks if float(q) > 0]
    if not b or not a or b[0][0] >= a[0][0]:
        raise ValueError("empty or crossed book")
    mid = (b[0][0] + a[0][0]) / 2

    def walk(levels, usd):
        got_usd = got_qty = 0.0
        for p, q in levels:
            take = min(q, (usd - got_usd) / p)
            got_usd += take * p
            got_qty += take
            if got_usd >= usd * (1 - 1e-9):
                return got_usd / got_qty
        return None

    out = {"mid": mid, "spread_bps": 1e4 * (a[0][0] - b[0][0]) / mid, "sell_bps": {}, "buy_bps": {}}
    for n in notionals:
        vs, vb = walk(b, n), walk(a, n)
        out["sell_bps"][str(int(n))] = None if vs is None else 1e4 * (mid - vs) / mid
        out["buy_bps"][str(int(n))] = None if vb is None else 1e4 * (vb - mid) / mid
    for pct in (0.5, 1.0, 2.0):
        out[f"bid_usd_{pct}"] = sum(p * q for p, q in b if p >= mid * (1 - pct / 100))
        out[f"ask_usd_{pct}"] = sum(p * q for p, q in a if p <= mid * (1 + pct / 100))
    return out


class FadeCostWatcher:
    """Tails the engine's follow-up log. For every pump whose FU_4H (entry) or FU_24H (exit) record
    was priced on a Binance USDM perp, it fetches that perp's order book and premium index right
    away and writes one FADE_COST record to data/alerts/fadecost-YYYYMMDD.jsonl. It starts at the
    end of the current files, so a restart never snapshots old records."""

    def __init__(self, cfg: dict, engine):
        self.cfg, self.engine = cfg or {}, engine
        self.poll = float(self.cfg.get("poll_s", 10))
        self.limit = int(self.cfg.get("depth_limit", 500))
        self.notionals = [float(x) for x in self.cfg.get("notionals_usd", [1000, 10000, 50000, 100000])]
        self.heavy_ratio = float(self.cfg.get("heavy_ratio", 2.2525))
        self.dir = Path(getattr(engine, "fu_dir", "data/alerts"))
        self.pos: dict[str, int] = {}
        self.started = False

    def new_records(self, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        out = []
        for day in sorted({time.strftime("%Y%m%d", time.gmtime(now - 86400)), time.strftime("%Y%m%d", time.gmtime(now))}):
            p = self.dir / f"followups-{day}.jsonl"
            if not p.exists():
                continue
            size = p.stat().st_size
            if str(p) not in self.pos:
                self.pos[str(p)] = size if not self.started else 0
            with open(p, "rb") as f:
                f.seek(self.pos[str(p)])
                chunk = f.read(max(0, size - self.pos[str(p)]))
            end = chunk.rfind(b"\n") + 1                   # complete lines only
            self.pos[str(p)] += end
            for line in chunk[:end].splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        self.started = True
        return out

    def wanted(self, rec: dict) -> dict | None:
        leg = {"FU_4H": "entry", "FU_24H": "exit"}.get(rec.get("kind"))
        key = rec.get("px_key") or ""
        if leg is None or (rec.get("dir") or 0) <= 0 or not key.startswith("binance_usdm:"):
            return None
        out = {"kind": "FADE_COST", "leg": leg, "id": rec.get("id"), "asset": rec.get("asset"), "symbol": key.split(":", 1)[1],
               "t_rec": rec.get("t"), "late_s_rec": rec.get("late_s"), "px_rec": rec.get("px")}
        if leg == "entry":
            vr = rec.get("vol_ratio_binance_usdm")
            ok = isinstance(vr, (int, float)) and math.isfinite(vr)
            out.update(vol_ratio_bn=vr if ok else None, heavy=bool(ok and vr > self.heavy_ratio))
        return out

    async def snapshot(self, session, out: dict) -> dict:
        try:
            book = await http_json(session, "GET", f"{FAPI}/depth", params={"symbol": out["symbol"], "limit": self.limit})
            out.update(t=time.time(), **book_cost(book.get("bids", []), book.get("asks", []), self.notionals))
            pi = await http_json(session, "GET", f"{FAPI}/premiumIndex", params={"symbol": out["symbol"]})
            out.update(funding_rate=float(pi["lastFundingRate"]), next_funding_ms=int(pi["nextFundingTime"]),
                       mark=float(pi["markPrice"]), index=float(pi["indexPrice"]))
        except Exception as e:  # noqa: BLE001 -- a failed snapshot is a record with an error, never fatal
            out.update(t=out.get("t", time.time()), error=str(e)[:200])
        return out

    def write(self, rec: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"fadecost-{time.strftime('%Y%m%d', time.gmtime())}.jsonl", "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    async def run(self, session) -> None:
        log.info("fade cost watcher: started (%s)", self.dir)
        while True:
            try:
                for rec in self.new_records():
                    want = self.wanted(rec)
                    if want is None:
                        continue
                    r = await self.snapshot(session, want)
                    self.write(r)
                    log.info("fade cost %s %s %s: spread %s bps, 10k sell %s bps%s", r["leg"], r["asset"], r["symbol"],
                             f"{r['spread_bps']:.1f}" if "spread_bps" in r else "-",
                             f"{r['sell_bps']['10000']:.1f}" if r.get("sell_bps", {}).get("10000") is not None else "-",
                             f" ERROR {r['error']}" if "error" in r else "")
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 -- the watcher never stops the service
                log.warning("fade cost watcher: %s", str(e)[:200])
            await asyncio.sleep(self.poll)


def _hhmm(minute: int) -> str:
    return datetime.fromtimestamp(minute * 60, tz=timezone.utc).strftime("%H:%M")
