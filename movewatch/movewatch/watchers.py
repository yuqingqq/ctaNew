"""Listing + exchange-flag watcher (REST polling).

Many sudden moves start with a venue event rather than a price pattern: a new
Upbit/Coinbase/Binance market, or an Upbit caution flag (the exchange's own
"abnormal move" detector). The first poll of every source only records a
baseline, so a restart never floods old markets as "new".

Also the extreme-crash trigger (docs/EXTREME_CRASH.md, validated 2026-10-09): CrashWatcher reads
Binance USDM !miniTicker@arr and alerts when >= half the liquid perps fell >= 10% in 15 minutes.
Also FadeCostWatcher (docs/FORWARD_FADE_F1.md, descriptive): at each Binance-priced pump's +4 h and
+24 h follow-up record it snapshots the public order book and funding, so forward test F1 can
compare the assumed cost with the book. It also sends F1's SHORT / COVER messages and keeps its
paper book (results shown live by user decision, docs/FORWARD_UNBLIND_2026-10-10.md).
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
import csv
import io
import json
import logging
import math
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import websockets

from .notify import esc
from .tradfi import STATIC_BINANCE, is_tradfi
from .venues.base import HTTP_TIMEOUT, http_json

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


WS_PUBLIC = "wss://fstream.binance.com/public/stream?streams="    # partial depth (fapi REST answers 451 here)
WS_MARKET = "wss://fstream.binance.com/market/stream?streams="    # markPrice: mark, index, funding rate


async def _first(url: str, timeout: float = 10.0) -> dict:
    """The payload of the first message on a stream."""
    async with websockets.connect(url, open_timeout=timeout) as ws:
        m = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
    return m.get("data", m)


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

    out = {"mid": mid, "spread_bps": 1e4 * (a[0][0] - b[0][0]) / mid, "sell_bps": {}, "buy_bps": {},
           "levels": min(len(b), len(a)), "bid_usd_book": sum(p * q for p, q in b), "ask_usd_book": sum(p * q for p, q in a)}
    for n in notionals:
        vs, vb = walk(b, n), walk(a, n)
        out["sell_bps"][str(int(n))] = None if vs is None else 1e4 * (mid - vs) / mid
        out["buy_bps"][str(int(n))] = None if vb is None else 1e4 * (vb - mid) / mid
    for pct in (0.5, 1.0, 2.0):
        out[f"bid_usd_{pct}"] = sum(p * q for p, q in b if p >= mid * (1 - pct / 100))
        out[f"ask_usd_{pct}"] = sum(p * q for p, q in a if p <= mid * (1 + pct / 100))
    return out


PKG_ROOT = Path(__file__).resolve().parents[1]
F1_MEME = PKG_ROOT / "docs" / "coingecko_meme_rank1500_2026-10-09.json"      # frozen with F1
F1_CUTS = PKG_ROOT / "docs" / "SIGNAL_SEARCH_FREEZE.json"                    # A2 heavy cut (log10 volume ratio)
F1_IDENTITY = "2375f81d463303ac1136241af07ebe416fd27c4381cd89e23a929e2445a402a0"   # forward test v2's recorder
F1_START = 1790952059.0789132                                                # its first admitted start (2026-10-02 14:40:59Z)
F1_LATE = {"FU_4H": 600, "FU_24H": 3600}
F1_GAP_MIN, F1_REPEAT_S, F1_DUP_S, F1_DUP_LOG_PX = 5, 72 * 3600, 300, 0.01
F1_COST = 0.004                                                              # flat round trip (docs/FORWARD_FADE_F1.md)
F1_POLL_S, F1_POLL_GAP_S = 600, 3000
VISION_FUND = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"


def _base(asset: str) -> str:
    """research/premium_angles.py::base (live asset names carry no quote suffix)."""
    s = asset.removesuffix("USDT")
    for pre in ("1000000", "1000", "1M"):
        if s.startswith(pre) and len(s) > len(pre) + 1:
            return s[len(pre):]
    return s


def _px(x) -> str:
    return f"{x:.5g}" if isinstance(x, (int, float)) and x == x and x else "?"


def _pct(x, d=1) -> str:
    return "?" if x is None else f"{100 * x:+.{d}f}%"


def _months(t0: float, t1: float) -> list[str]:
    a, b = datetime.fromtimestamp(t0, tz=timezone.utc), datetime.fromtimestamp(t1, tz=timezone.utc)
    out, y, m = [], a.year, a.month
    while (y, m) <= (b.year, b.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def funding_from_polls(polls: list, t4: float, t24: float) -> tuple[float | None, int, bool]:
    """The short's funding over (t4, t24] from markPrice polls (t_obs s, rate, next funding ms): for
    each settlement time seen in the hold, the last rate observed before it. Complete when the polls
    cover the hold with no gap above 50 min (no settlement can be missed at intervals >= 1 h)."""
    last: dict[float, float] = {}
    for t, r, T in sorted(polls):
        Ts = T / 1000.0
        if t4 < Ts <= t24 and t < Ts:
            last[Ts] = r
    ts = sorted(t for t, _, _ in polls if t4 - 600 <= t <= t24 + 600)
    ok = bool(ts) and ts[0] <= t4 + 900 and ts[-1] >= t24 - 900 and all(b - a <= F1_POLL_GAP_S for a, b in zip(ts, ts[1:]))
    return (float(sum(last.values())) if ok else None), len(last), ok


class PaperBook:
    """F1's paper ledger (data/alerts/fadebook.jsonl): an `open` and a `close` event per trade, and a
    `fund_exact` event once Binance Vision publishes the hold's month(s). Net per trade = gross −
    0.40% + funding (exact, else the live estimate, else pending). Shown live by user decision
    2026-10-10 (docs/FORWARD_UNBLIND_2026-10-10.md); the F1 read and pass rule are unchanged."""

    def __init__(self):
        self.tr: dict[str, dict] = {}
        self.last_summary: str | None = None

    def apply(self, ev: dict) -> None:
        k, i = ev.get("ev"), ev.get("id")
        if k == "open" and i not in self.tr:
            self.tr[i] = dict(ev)
        elif k == "close" and i in self.tr and "t24" not in self.tr[i]:
            self.tr[i].update({x: ev.get(x) for x in ("t24", "px24", "void", "fund_est", "fund_n", "fund_ok")})
        elif k == "fund_exact" and i in self.tr:
            self.tr[i].update(fund_exact=ev.get("funding"), fund_exact_done=True)
        elif k == "summary":
            self.last_summary = ev.get("day")

    @staticmethod
    def gross(t: dict) -> float:
        return -(t["px24"] / t["px4"] - 1.0)

    @staticmethod
    def funding(t: dict) -> tuple[float | None, str]:
        if t.get("fund_exact") is not None:
            return t["fund_exact"], "exact"
        if t.get("fund_ok") and t.get("fund_est") is not None:
            return t["fund_est"], "est."
        return None, "pending"

    def net(self, t: dict) -> float | None:
        f = self.funding(t)[0]
        return None if f is None else self.gross(t) - F1_COST + f

    def closed(self) -> list[dict]:
        return sorted((t for t in self.tr.values() if t.get("t24") is not None and not t.get("void")), key=lambda t: t["t4"])

    def open_trades(self) -> list[dict]:
        return sorted((t for t in self.tr.values() if t.get("t24") is None), key=lambda t: t["t4"])

    def summary(self) -> dict:
        c = self.closed()
        g = [self.gross(t) for t in c]
        nets = [self.net(t) for t in c]
        nk = [x for x in nets if x is not None]
        best = [n if n is not None else self.gross(t) - F1_COST for t, n in zip(c, nets)]
        return {"closed": len(c), "wins": sum(1 for x in best if x > 0), "gross_avg": sum(g) / len(g) if g else None,
                "net_avg": sum(nk) / len(nk) if nk else None, "fund_known": len(nk), "pending": len(c) - len(nk),
                "worst": min(best) if best else None, "best": max(best) if best else None,
                "total_capital": 0.01 * sum(best), "void": sum(1 for t in self.tr.values() if t.get("void")),
                "open": len(self.open_trades())}

    def line(self) -> str:
        s = self.summary()
        if not s["closed"]:
            return f"Paper book: no closed trade yet · {s['open']} open"
        out = (f"Paper book: {s['closed']} closed · {s['wins']} won · per trade gross {_pct(s['gross_avg'])}, "
               f"net {_pct(s['net_avg'])}" + (f" (on {s['fund_known']})" if s["pending"] else "")
               + f" · worst {_pct(s['worst'], 0)} · total {_pct(s['total_capital'], 2)} of capital at 1%/trade")
        if s["pending"]:
            out += f" · {s['pending']} with funding pending (counted at 0; research average −0.78%)"
        return out


class FadeCostWatcher:
    """Forward test F1's live companion (docs/FORWARD_FADE_F1_v2.md). Tails the engine's follow-up
    log.
    - Cost: for every pump whose FU_4H (entry) or FU_24H (exit) record was priced on a Binance USDM
      perp, it takes that perp's first 20-level partial-depth and markPrice websocket messages and
      writes one FADE_COST record to data/alerts/fadecost-YYYYMMDD.jsonl.
    - Signals (config `signals`): at an F1 entry (the evaluator's population, research/forward_fade.py)
      a SHORT message with the cover time; at its FU_24H record a COVER message with the paper result.
    - Paper book (PaperBook, data/alerts/fadebook.jsonl): every F1 trade since F1 began, funding polled
      every 10 min while open, exact funding from Vision once published, a daily summary.
    Fail closed: without the meme list or the cut, no signal and no book (cost records go on)."""

    def __init__(self, cfg: dict, engine):
        self.cfg, self.engine = cfg or {}, engine
        self.poll = float(self.cfg.get("poll_s", 10))
        self.notionals = [float(x) for x in self.cfg.get("notionals_usd", [1000, 10000, 50000, 100000])]
        self.dir = Path(getattr(engine, "fu_dir", "data/alerts"))
        self.fund_dir = Path(self.cfg.get("funding_cache", PKG_ROOT / "data" / "forward_fade" / "funding"))
        self.summary_min = int(self.cfg.get("summary_utc_min", 5))           # minutes after 00:00 UTC
        self.identity = str(self.cfg.get("f1_identity", F1_IDENTITY))
        self.f1_start = float(self.cfg.get("f1_start", F1_START))
        self.pos: dict[str, int] = {}
        self.started = False
        self.signals = bool(self.cfg.get("signals", True))
        try:
            self.meme = set(json.loads(Path(self.cfg.get("meme_file", F1_MEME)).read_text())["symbols"])
            cells = json.loads(Path(self.cfg.get("cuts_file", F1_CUTS)).read_text())["cells"]
            self.heavy_log = float(next(c for c in cells if c["id"] == "A2")["cuts"][1])
            self.ready = True
        except Exception as e:  # noqa: BLE001 -- no reference data, no signals, no book
            log.warning("fade watcher: signals and book OFF, reference data unreadable: %s", str(e)[:150])
            self.meme, self.heavy_log, self.signals, self.ready = None, float("inf"), False, False
        self.runs: set[str] = set()             # admitted runs (frozen identity, not dry)
        self.ended_at: float | None = None      # first production start with other code: F1 ends
        self.opens: dict[str, dict] = {}        # admitted in-test FU_OPEN by id
        self.kept_t: dict[str, float] = {}      # asset -> last alert kept by the 72 h rule
        self.excluded: set[str] = set()         # DUPLICATE or REPEAT_72H ids
        self.recent: list[dict] = []            # kept opens of the last 5 min (duplicate check)
        self.book = PaperBook()
        self.polls: dict[str, list] = {}        # id -> [(t, rate, next_T_ms)]
        self.marks: dict[str, tuple] = {}       # id -> (t, mark)
        self._published: dict[str, bool] = {}
        self._ended_msg = False

    # ------------------------------------------------------------------ files
    def _read(self, p: Path, start: int) -> tuple[list[dict], int]:
        size = p.stat().st_size
        with open(p, "rb") as f:
            f.seek(start)
            chunk = f.read(max(0, size - start))
        end = chunk.rfind(b"\n") + 1                        # complete lines only
        out = []
        for line in chunk[:end].splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out, start + end

    def new_records(self, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        out = []
        for day in sorted({time.strftime("%Y%m%d", time.gmtime(now - 86400)), time.strftime("%Y%m%d", time.gmtime(now))}):
            p = self.dir / f"followups-{day}.jsonl"
            if not p.exists():
                continue
            if str(p) not in self.pos:
                self.pos[str(p)] = p.stat().st_size if not self.started else 0
            recs, self.pos[str(p)] = self._read(p, self.pos[str(p)])
            out += recs
        self.started = True
        return out

    def write(self, rec: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"fadecost-{time.strftime('%Y%m%d', time.gmtime())}.jsonl", "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    def book_event(self, ev: dict) -> None:
        self.book.apply(ev)
        with open(self.dir / "fadebook.jsonl", "a") as f:
            f.write(json.dumps(ev, default=float) + "\n")

    def prime(self, now: float | None = None) -> None:
        """Load the ledger, then replay the follow-up records it may have missed: every day since F1
        began when there is no ledger yet (the one-time history backfill), else the last 6 days.
        Missed entries and exits are booked silently; the tail continues after the replayed lines."""
        now = time.time() if now is None else now
        book = self.dir / "fadebook.jsonl"
        if book.exists():
            for ev in self._read(book, 0)[0]:
                self.book.apply(ev)
        first = self.f1_start if not book.exists() else max(self.f1_start, now - 6 * 86400)
        day = datetime.fromtimestamp(first, tz=timezone.utc).date()
        while day <= datetime.fromtimestamp(now, tz=timezone.utc).date():
            p = self.dir / f"followups-{day:%Y%m%d}.jsonl"
            if p.exists():
                recs, self.pos[str(p)] = self._read(p, 0)
                for r in recs:
                    self.step(r, live=False)
            day = date.fromordinal(day.toordinal() + 1)
        for k in (2, 1, 0):
            p = self.dir / f"fadecost-{time.strftime('%Y%m%d', time.gmtime(now - k * 86400))}.jsonl"
            if p.exists():
                for r in self._read(p, 0)[0]:
                    self.add_poll(r)
        self.started = True

    # ------------------------------------------------------------------ F1 rules
    def admitted(self, rec: dict) -> bool:
        return rec.get("run") in self.runs and (self.ended_at is None or float(rec.get("t", 0)) < self.ended_at)

    def on_start(self, r: dict) -> None:
        if r.get("dry_run"):
            return
        if r.get("code_sha") == self.identity:
            if self.ended_at is None:
                self.runs.add(r.get("run"))
        elif float(r.get("t", 0)) > self.f1_start and self.ended_at is None:
            self.ended_at = float(r["t"])

    def on_open(self, o: dict) -> None:
        """F1 rows are actions: admitted in-test alerts, the v2 DUPLICATE rule, then the first alert
        per asset in any 72 h (any direction)."""
        t = float(o.get("t", 0.0))
        if not self.admitted(o) or t < self.f1_start:
            return
        self.recent = [p for p in self.recent if t - p["t"] <= F1_DUP_S]
        for p in self.recent:
            if (p["asset"] != o.get("asset") and p.get("dir") == o.get("dir") and (p.get("alert_px") or 0) > 0
                    and (o.get("alert_px") or 0) > 0 and abs(math.log(o["alert_px"] / p["alert_px"])) <= F1_DUP_LOG_PX):
                self.excluded.add(o["id"])
                return
        self.recent.append(o)
        self.opens[o["id"]] = o
        lt = self.kept_t.get(o.get("asset"))
        if lt is not None and t - lt < F1_REPEAT_S:
            self.excluded.add(o["id"])
            return
        self.kept_t[o.get("asset")] = t

    def _rec_ok(self, rec: dict, ref: str) -> bool:
        return (self.admitted(rec) and (rec.get("late_s") or 0) <= F1_LATE[rec["kind"]]
                and (rec.get("px") or 0) > 0 and rec.get("px_key") == ref)

    def entry_ok(self, rec: dict) -> bool:
        """F1's entry rules on an FU_4H record (research/forward_fade.rows up to the 4 h checks)."""
        o = self.opens.get(rec.get("id"))
        if not self.ready or o is None or rec.get("kind") != "FU_4H" or (o.get("dir") or 0) <= 0:
            return False
        ref = o.get("ref_key") or ""
        vr = rec.get("vol_ratio_binance_usdm")
        gaps = max(rec.get("gap_min_before", math.inf), rec.get("gap_min_after", math.inf))
        return (rec["id"] not in self.excluded and ref.startswith("binance_usdm:") and _base(o.get("asset", "")) not in self.meme
                and self._rec_ok(rec, ref) and isinstance(vr, (int, float)) and math.isfinite(vr) and vr > 0
                and gaps <= F1_GAP_MIN and math.log10(vr) > self.heavy_log)

    def step(self, rec: dict, live: bool) -> str | None:
        """Book one follow-up record. Returns 'short' / 'cover' when a live message is due."""
        k = rec.get("kind")
        if k == "FU_START":
            self.on_start(rec)
        elif k == "FU_OPEN":
            self.on_open(rec)
        elif k == "FU_4H" and self.entry_ok(rec) and rec["id"] not in self.book.tr:
            o = self.opens[rec["id"]]
            self.book_event({"ev": "open", "id": rec["id"], "asset": o["asset"], "symbol": o["ref_key"].split(":", 1)[1],
                             "alert_t": o["t"], "t4": rec["t"], "px4": rec["px"], "vol_ratio_bn": rec.get("vol_ratio_binance_usdm"),
                             "src": "live" if live and self.signals else "history"})
            return "short" if live and self.signals else None
        elif k == "FU_24H" and rec.get("id") in self.book.tr and self.book.tr[rec["id"]].get("t24") is None:
            tr = self.book.tr[rec["id"]]
            ok = self._rec_ok(rec, self.opens.get(rec["id"], {}).get("ref_key") or f"binance_usdm:{tr['symbol']}")
            est, n, fok = funding_from_polls(self.polls.get(rec["id"], []), tr["t4"], rec["t"])
            self.book_event({"ev": "close", "id": rec["id"], "t24": rec["t"], "px24": rec.get("px"),
                             "void": None if ok else "24H record late, missing price or other instrument",
                             "fund_est": est, "fund_n": n, "fund_ok": fok})
            self.polls.pop(rec["id"], None)
            return "cover" if live and tr.get("src") == "live" else None
        return None

    def add_poll(self, r: dict) -> None:
        i = r.get("id")
        if i in self.book.tr and self.book.tr[i].get("t24") is None and "funding_rate" in r and r.get("next_funding_ms"):
            self.polls.setdefault(i, []).append((float(r["t"]), float(r["funding_rate"]), int(r["next_funding_ms"])))
            if r.get("mark"):
                self.marks[i] = (float(r["t"]), float(r["mark"]))

    # ------------------------------------------------------------------ network
    def wanted(self, rec: dict) -> dict | None:
        leg = {"FU_4H": "entry", "FU_24H": "exit"}.get(rec.get("kind"))
        key = rec.get("px_key") or ""
        if leg is None or (rec.get("dir") or 0) <= 0 or not key.startswith("binance_usdm:"):
            return None
        out = {"kind": "FADE_COST", "leg": leg, "id": rec.get("id"), "asset": rec.get("asset"), "symbol": key.split(":", 1)[1],
               "t_rec": rec.get("t"), "late_s_rec": rec.get("late_s"), "px_rec": rec.get("px")}
        if leg == "entry":
            vr = rec.get("vol_ratio_binance_usdm")
            ok = isinstance(vr, (int, float)) and math.isfinite(vr) and vr > 0
            out.update(vol_ratio_bn=vr if ok else None, heavy=bool(ok and math.log10(vr) > self.heavy_log))
        return out

    async def snapshot(self, session, out: dict) -> dict:
        try:
            sym = out["symbol"].lower()
            book = await _first(f"{WS_PUBLIC}{sym}@depth20@100ms")
            out.update(t=time.time(), **book_cost(book.get("b", []), book.get("a", []), self.notionals))
            out.update(await self._mark(sym))
        except Exception as e:  # noqa: BLE001 -- a failed snapshot is a record with an error, never fatal
            out.update(t=out.get("t", time.time()), error=str(e)[:200])
        return out

    async def _mark(self, sym: str) -> dict:
        mp = await _first(f"{WS_MARKET}{sym.lower()}@markPrice@1s")
        return {"funding_rate": float(mp["r"]), "next_funding_ms": int(mp["T"]), "mark": float(mp["p"]), "index": float(mp["i"])}

    async def poll_funding(self, now: float) -> None:
        for tr in self.book.open_trades():
            last = max((t for t, _, _ in self.polls.get(tr["id"], [])), default=0.0)
            if now - last < F1_POLL_S:
                continue
            r = {"kind": "FADE_COST", "leg": "fund", "id": tr["id"], "asset": tr["asset"], "symbol": tr["symbol"], "t": now}
            try:
                r.update(await self._mark(tr["symbol"]))
            except Exception as e:  # noqa: BLE001
                r["error"] = str(e)[:200]
            self.write(r)
            self.add_poll(r)

    async def _is_published(self, session, month: str) -> bool:
        if not self._published.get(month):
            async with session.head(VISION_FUND.format(s="BTCUSDT", m=month), allow_redirects=True, timeout=HTTP_TIMEOUT) as r:
                self._published[month] = r.status == 200
        return self._published[month]

    async def _month_rates(self, session, symbol: str, month: str) -> list | None:
        p = self.fund_dir / f"{symbol}_{month}.csv"
        if not p.exists():
            async with session.get(VISION_FUND.format(s=symbol, m=month), timeout=HTTP_TIMEOUT) as r:
                if r.status == 404:
                    return None
                r.raise_for_status()
                data = await r.read()
            z = zipfile.ZipFile(io.BytesIO(data))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.open(z.namelist()[0]).read())
        rows = list(csv.DictReader(p.read_text().splitlines()))
        return [(int(x["calc_time"]), float(x["last_funding_rate"])) for x in rows]

    async def exact_funding(self, session) -> int:
        """Replace estimates with Vision's settled rates (research/btc_hedge.Funding.total) once every
        month of a closed trade's hold is published. A missing coin archive is booked as None."""
        done = 0
        for tr in self.book.closed():
            if tr.get("fund_exact_done"):
                continue
            ms = _months(tr["t4"], tr["t24"])
            if not all([await self._is_published(session, m) for m in ms]):
                continue
            total = 0.0
            for m in ms:
                rows = await self._month_rates(session, tr["symbol"], m)
                if rows is None:
                    total = None
                    break
                total += sum(r for ct, r in rows if tr["t4"] * 1000 < ct <= tr["t24"] * 1000)
            self.book_event({"ev": "fund_exact", "id": tr["id"], "funding": total})
            done += 1
        return done

    # ------------------------------------------------------------------ messages
    def entry_text(self, rec: dict, snap: dict) -> str:
        o = self.opens[rec["id"]]
        cover = _hhmm(int((float(o["t"]) + 86400) // 60))
        lines = [f"📉 <b>FADE: SHORT {esc(o['asset'])}</b> · paper test F1, NOT validated",
                 f"Volume kept up: hours 1–4 ran {rec['vol_ratio_binance_usdm']:.1f}× the pre-alert pace (heavy).",
                 f"Rule: SHORT {snap['symbol']} perp now (~${_px(rec.get('px'))}), 1% of capital, no stop. "
                 f"COVER at {cover} UTC (alert + 24 h)."]
        cost = []
        if "spread_bps" in snap:
            s10 = (snap.get("sell_bps") or {}).get("10000")
            cost.append(f"spread {snap['spread_bps']:.0f} bps" + (f", $10k sell {s10:.0f} bps" if s10 is not None else ", $10k beyond 20 levels"))
        if "funding_rate" in snap:
            fr = snap["funding_rate"]
            cost.append(f"funding {100 * fr:+.3f}%/period ({'short pays' if fr < 0 else 'short receives'})")
        if cost:
            lines.append("Now: " + " · ".join(cost))
        lines.append("History after costs + funding: +0.5%/trade 2022–25, +1.5% in 2026; ~62% win; "
                     "~2% of trades lose > 50% (runners). Funding is the biggest cost.")
        lines.append(self.book.line())
        return "\n".join(lines)

    def exit_text(self, tr: dict) -> str:
        head = f"📉 <b>FADE: COVER {esc(tr.get('asset', '?'))}</b> now · paper test F1 (short opened {_hhmm(int(float(tr['t4']) // 60))} UTC)"
        if tr.get("void"):
            return head + f"\nPaper: trade void in the test ({tr['void']})\n" + self.book.line()
        f, how = self.book.funding(tr)
        fund = f"funding {_pct(f, 2)} ({how}, {tr.get('fund_n', 0)} settlements)" if f is not None else "funding pending"
        net = self.book.net(tr)
        return (head + f"\nPaper: short ${_px(tr['px4'])} → cover ${_px(tr['px24'])} = {_pct(self.book.gross(tr), 2)} gross · "
                f"cost −0.40% · {fund} → net {_pct(net, 2) if net is not None else '?'}\n" + self.book.line())

    def summary_text(self, now: float) -> str:
        lines = ["📊 <b>FADE paper book</b> (F1 since 10-02, 1% of capital per trade)", self.book.line()]
        for tr in self.book.open_trades():
            m = self.marks.get(tr["id"])
            un = f", now {_pct(-(m[1] / tr['px4'] - 1.0))} gross" if m else ""
            lines.append(f"open: {esc(tr['asset'])} short ${_px(tr['px4'])} since {_hhmm(int(tr['t4'] // 60))}, "
                         f"cover {_hhmm(int((tr['alert_t'] + 86400) // 60))} UTC{un}")
        if self.ended_at:
            lines.append("F1 has ENDED: the service code changed. Signals stopped.")
        lines.append("F1 is judged once at ≥ 400 trades (~Jun 2027); nothing changes before then.")
        return "\n".join(lines)

    async def handle(self, session, rec: dict) -> None:
        want = self.wanted(rec)
        snap = await self.snapshot(session, want) if want is not None else None
        if snap is not None:
            self.add_poll(snap)                  # exit: its markPrice is the last funding poll of the hold
        act = self.step(rec, live=True) if self.ready else None
        if snap is not None:
            self.add_poll(snap)                  # entry: the first poll (add_poll books only open trades)
            if act:
                snap["signal"] = True
            self.write(snap)
            log.info("fade cost %s %s %s: spread %s bps, 10k sell %s bps%s%s", snap["leg"], snap["asset"], snap["symbol"],
                     f"{snap['spread_bps']:.1f}" if "spread_bps" in snap else "-",
                     f"{snap['sell_bps']['10000']:.1f}" if (snap.get("sell_bps") or {}).get("10000") is not None else "-",
                     " SIGNAL" if act else "", f" ERROR {snap['error']}" if "error" in snap else "")
        if act == "short":
            self.engine.notifier.emit(self.entry_text(rec, snap or {"symbol": self.book.tr[rec["id"]]["symbol"]}),
                                      {"kind": "FADE_SHORT", "id": rec["id"], "asset": rec.get("asset")})
        elif act == "cover":
            tr = self.book.tr[rec["id"]]
            self.engine.notifier.emit(self.exit_text(tr), {"kind": "FADE_COVER", "id": rec["id"], "asset": tr.get("asset"),
                                                           "net": self.book.net(tr), "gross": self.book.gross(tr) if tr.get("px24") else None})
        if self.ended_at and not self._ended_msg:
            self._ended_msg = True
            self.engine.notifier.emit("📉 FADE: forward test F1 has ENDED (the service code changed). No more signals.",
                                      {"kind": "FADE_ENDED"})

    async def run(self, session) -> None:
        try:
            if self.ready:
                self.prime()
        except Exception as e:  # noqa: BLE001
            log.warning("fade watcher: prime failed, signals and book OFF: %s", str(e)[:200])
            self.ready = self.signals = False
        s = self.book.summary()
        log.info("fade watcher: started (%s), signals %s, book %d closed / %d open, %d runs admitted",
                 self.dir, "ON" if self.signals else "OFF", s["closed"], s["open"], len(self.runs))
        last_exact = 0.0
        while True:
            now = time.time()
            try:
                for rec in self.new_records(now):
                    await self.handle(session, rec)
                if self.ready:
                    await self.poll_funding(now)
                    if now - last_exact >= 6 * 3600:
                        last_exact = now
                        n = await self.exact_funding(session)
                        if n:
                            log.info("fade watcher: exact funding booked for %d trades", n)
                    day = time.strftime("%Y-%m-%d", time.gmtime(now))
                    if self.book.last_summary != day and (now % 86400) >= self.summary_min * 60:
                        self.book_event({"ev": "summary", "id": None, "day": day})
                        self.engine.notifier.emit(self.summary_text(now), {"kind": "FADE_SUMMARY", **self.book.summary()})
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 -- the watcher never stops the service
                log.warning("fade watcher: %s", str(e)[:200])
            await asyncio.sleep(self.poll)


def _hhmm(minute: int) -> str:
    return datetime.fromtimestamp(minute * 60, tz=timezone.utc).strftime("%H:%M")
