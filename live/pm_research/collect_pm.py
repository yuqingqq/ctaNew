"""
Polymarket crypto 5-min "Up or Down" market collector (P-2026-003, E0).

Markets: {btc,eth,sol,xrp}-updown-5m-<unix_window_start>, 300 s-aligned windows,
pre-created ~5 min ahead (verified 2026-08-19). Per market we record the full
CLOB market channel (book snapshots, price_change deltas, last_trade_price —
which carries fee_rate_bps) plus Gamma metadata and the post-close resolution.

Architecture (all asyncio, one process):
  discovery  every 30 s: for each coin × window ∈ {current, next}, resolve the
             slug via Gamma; new market → append metadata to markets.jsonl and
             spawn a market task
  market     one WS connection per market (2 asset ids) to
             wss://ws-subscriptions-clob.polymarket.com/ws/market; every raw
             message stored as "<recv_ns>\t<raw json>" in
             raw/<YYYYMMDD>/<slug>.jsonl; closed at window_end + 90 s grace,
             then gzipped
  resolution every 60 s: markets past end+120 s are polled on Gamma until
             closed/outcomePrices appear → append to resolutions.jsonl

Signal-side data (Binance bookTicker/depth/trades for the same coins) is already
collected by live/mm_research/collect_hf.py — this collector is the market side.

NOTE Gamma rejects the default python UA (Cloudflare 403) — a curl UA works.

Run:  nohup python3 live/pm_research/collect_pm.py > data/pm_5min/collector.log 2>&1 &
"""
from __future__ import annotations

import asyncio
import gzip
import json
import shutil
import signal
import time
import urllib.request
from pathlib import Path

import websockets

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/pm_5min"
RAW = ROOT / "raw"

GAMMA = "https://gamma-api.polymarket.com/markets?slug="
CLOB = "https://clob.polymarket.com/markets/"      # by conditionId — persists post-resolution
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
UA = {"User-Agent": "curl/8.5.0"}
COINS = ("btc", "eth", "sol", "xrp", "doge", "bnb", "hype")  # doge/bnb/hype exist (iter-1)
WINDOW_S = 300
GRACE_S = 90                 # keep recording this long past window end
DISCOVER_S = 30
RESOLVE_S = 60
RESOLVE_AFTER_S = 120        # start polling resolution this long past end
RESOLVE_GIVEUP_S = 7200


def gget(slug: str):
    req = urllib.request.Request(GAMMA + slug, headers=UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        out = json.load(r)
    return out[0] if out else None


def rewards_registry() -> dict:
    """AUTHORITATIVE rewards params per condition_id (sketch-review M MUST-FIX 2:
    Gamma's rewardsMaxSpread/MinSize are stale — the band was re-cut 2026-08-20
    and Gamma still served the old number). Also yields the per-market
    rate_per_day, which Gamma does not expose at all. Paginated, 500/page."""
    out, cursor = {}, ""
    for _ in range(60):                      # hard page cap (registry ≈33 pages 2026-08-20)
        url = "https://clob.polymarket.com/rewards/markets/current?limit=500"
        if cursor:
            url += f"&next_cursor={cursor}"
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.load(r)
        rows = d.get("data", d if isinstance(d, list) else [])
        for row in rows:
            cid = row.get("condition_id")
            if cid:
                out[cid] = {k: row.get(k) for k in
                            ("rewards_max_spread", "rewards_min_size",
                             "native_daily_rate", "total_daily_rate", "rewards_config")}
        cursor = d.get("next_cursor") or ""
        if not cursor or cursor == "LTE=" or not rows:
            break
    return out


def clob_market(condition_id: str):
    """RESOLUTION SOURCE (verified 2026-08-19): Gamma slug queries return EMPTY
    shortly after a 5-min market resolves, so Gamma can never confirm outcomes.
    The CLOB market endpoint persists post-resolution and carries closed=True +
    per-token winner flags."""
    req = urllib.request.Request(CLOB + condition_id, headers=UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def jl_append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")


def is_final(m: dict) -> bool:
    """A market is RESOLVED only if closed, or outcomePrices degenerate to {0,1}.
    Gamma populates outcomePrices continuously for OPEN markets (live prices) —
    treating their mere presence as resolution records garbage (bug found
    2026-08-19: rows with closed=false and prices like 0.165/0.835)."""
    if m.get("closed") is True or m.get("gave_up"):
        return True
    op = m.get("outcomePrices")
    if op:
        try:
            vals = json.loads(op) if isinstance(op, str) else op
            return set(float(x) for x in vals) <= {0.0, 1.0}
        except Exception:
            return False
    return False


class PMCollector:
    def __init__(self):
        self.stop = False
        self.known: set[str] = set()          # slugs with a running/finished market task
        self.pending_res: dict[str, tuple[int, str]] = {}  # slug → (window_end, conditionId)
        self.resume: list[tuple[str, int, list]] = []       # in-flight windows to re-spawn
        self.rewards: dict = {}            # condition_id → authoritative rewards params
        self.rewards_ts = 0.0
        self.counts = {"msgs": 0, "markets": 0, "resolved": 0}
        self._load_state()

    def _load_state(self) -> None:
        """Resume: don't re-collect known markets; re-poll unresolved ones."""
        resolved = set()
        if (ROOT / "resolutions.jsonl").exists():
            for ln in open(ROOT / "resolutions.jsonl"):
                try:
                    r = json.loads(ln)
                    if is_final(r):          # non-final rows (bug above) re-poll
                        resolved.add(r["slug"])
                except Exception:
                    pass
        if (ROOT / "markets.jsonl").exists():
            for ln in open(ROOT / "markets.jsonl"):
                try:
                    m = json.loads(ln)
                    self.known.add(m["slug"])
                    if m["slug"] not in resolved:
                        self.pending_res[m["slug"]] = (m["window_end"], m["conditionId"])
                    # MUST-FIX iter-1: re-spawn recording for windows still open at
                    # startup — plain `known` marking silently dropped them (verified
                    # data loss at the 15:12 restart: one window empty, one truncated).
                    if m["window_end"] + GRACE_S > time.time():
                        self.resume.append((m["slug"], m["window_start"], m["clobTokenIds"]))
                except Exception:
                    pass
        if self.known:
            print(f"[pm] resumed: {len(self.known)} known, "
                  f"{len(self.pending_res)} pending resolution", flush=True)

    # ---- discovery ------------------------------------------------------------
    async def _discovery(self) -> None:
        loop = asyncio.get_event_loop()
        while not self.stop:
            now = int(time.time())
            if now - self.rewards_ts > 600:        # refresh registry every 10 min
                try:
                    self.rewards = await loop.run_in_executor(None, rewards_registry)
                    self.rewards_ts = now
                    jl_append(ROOT / "rewards_registry.jsonl",
                              {"recv_ns": time.time_ns(), "n": len(self.rewards)})
                except Exception as ex:
                    print(f"[pm] rewards registry: {type(ex).__name__} {str(ex)[:60]}",
                          flush=True)
            cur = now - now % WINDOW_S
            for coin in COINS:
                for ts in (cur, cur + WINDOW_S):
                    slug = f"{coin}-updown-5m-{ts}"
                    if slug in self.known:
                        continue
                    try:
                        m = await loop.run_in_executor(None, gget, slug)
                    except Exception as ex:
                        print(f"[pm] gamma {slug}: {type(ex).__name__} {str(ex)[:60]}",
                              flush=True)
                        continue
                    if not m:
                        continue
                    self.known.add(slug)
                    toks = json.loads(m["clobTokenIds"])
                    # per-market CLOB fee/rewards params (iter-1: Gamma and docs
                    # disagree on fees — capture the venue's own numbers)
                    clob_extra = {}
                    try:
                        c = await loop.run_in_executor(None, clob_market, m.get("conditionId"))
                        clob_extra = {k: c.get(k) for k in
                                      ("maker_base_fee", "taker_base_fee", "min_incentive_size",
                                       "max_incentive_spread", "rewards", "neg_risk")}
                    except Exception:
                        pass
                    meta = {"recv_ns": time.time_ns(), "slug": slug, "coin": coin,
                            "window_start": ts, "window_end": ts + WINDOW_S,
                            "conditionId": m.get("conditionId"),
                            "clobTokenIds": toks, "outcomes": m.get("outcomes"),
                            "question": m.get("question"),
                            # FULL description — it names the exact settlement stream
                            # (e.g. btc-usd-twap-60s-streams); truncation loses it
                            "description": m.get("description"),
                            "resolutionSource": m.get("resolutionSource"),
                            # rewards/microstructure params (per-market, versioned)
                            "rewardsMinSize": m.get("rewardsMinSize"),
                            "rewardsMaxSpread": m.get("rewardsMaxSpread"),
                            "orderPriceMinTickSize": m.get("orderPriceMinTickSize"),
                            "orderMinSize": m.get("orderMinSize"),
                            "clob": clob_extra,
                            # authoritative (Gamma's copies above are kept only to
                            # measure the discrepancy — never used for decisions)
                            # NB (2026-08-20): 5-min crypto markets are ABSENT from the
                            # CLOB rewards registry (exhausted 33 pages / 16,172 rows;
                            # per-market lookup also empty) — so this is normally None and
                            # the None ITSELF is the finding. Whether these markets are
                            # reward-eligible at all, and under which params, is an open
                            # question gating G3b. Keep recording: if they appear later,
                            # we capture the transition.
                            "rewards_authoritative": self.rewards.get(m.get("conditionId")),
                            "rewards_registry_n": len(self.rewards),
                            "endDate": m.get("endDate")}
                    jl_append(ROOT / "markets.jsonl", meta)
                    self.pending_res[slug] = (ts + WINDOW_S, meta["conditionId"])
                    self.counts["markets"] += 1
                    asyncio.create_task(self._market(slug, ts, toks))
            await self._sleep(DISCOVER_S)

    # ---- per-market WS recorder -----------------------------------------------
    async def _market(self, slug: str, ts: int, toks: list[str]) -> None:
        day = time.strftime("%Y%m%d", time.gmtime(ts))
        path = RAW / day / f"{slug}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        stop_at = ts + WINDOW_S + GRACE_S
        f = open(path, "a", buffering=1 << 20)     # 1 MB userspace buffer
        buf: list[str] = []
        try:
            while not self.stop and time.time() < stop_at:
                try:
                    # max_queue: the default (32) makes the server drop us with
                    # 1013 "slow consumer" on hot markets — observed repeatedly on
                    # BTC, i.e. load-correlated loss on ~85% of notional. Deep queue
                    # + batched writes keep the recv loop free of blocking I/O.
                    async with websockets.connect(WS_URL, ping_interval=10,
                                                  ping_timeout=10, open_timeout=15,
                                                  max_queue=2 ** 16) as ws:
                        await ws.send(json.dumps({"assets_ids": toks, "type": "market"}))
                        while not self.stop and time.time() < stop_at:
                            timeout = max(1.0, stop_at - time.time())
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=min(timeout, 30))
                            except asyncio.TimeoutError:
                                if buf:
                                    f.write("".join(buf)); buf.clear()
                                if time.time() >= stop_at:
                                    break
                                continue      # quiet stretch mid-window is normal
                            buf.append(f"{time.time_ns()}\t{raw}\n")
                            self.counts["msgs"] += 1
                            if len(buf) >= 200:
                                f.write("".join(buf)); buf.clear()
                except Exception as ex:
                    if buf:
                        f.write("".join(buf)); buf.clear()
                    if "1013" in str(ex) or "slow" in str(ex).lower():
                        self.counts["slow_drops"] = self.counts.get("slow_drops", 0) + 1
                    if time.time() < stop_at and not self.stop:
                        print(f"[pm] {slug} ws: {type(ex).__name__} {str(ex)[:60]} — retry",
                              flush=True)
                        await asyncio.sleep(1)
        finally:
            if buf:
                f.write("".join(buf))
            f.close()
            try:
                # numbered shards: a resumed window must never clobber the gz a
                # previous (truncated) run wrote — consumers concat all shards
                gz = Path(str(path) + ".gz")
                i = 1
                while gz.exists():
                    gz = Path(str(path) + f".{i}.gz")
                    i += 1
                with open(path, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
                    shutil.copyfileobj(src, dst)
                path.unlink()
            except Exception as ex:
                print(f"[pm] gzip {slug}: {ex}", flush=True)

    # ---- resolution poller ----------------------------------------------------
    async def _resolver(self) -> None:
        loop = asyncio.get_event_loop()
        while not self.stop:
            await self._sleep(RESOLVE_S)
            now = time.time()
            for slug, (end, cond) in list(self.pending_res.items()):
                if now < end + RESOLVE_AFTER_S:
                    continue
                try:
                    c = await loop.run_in_executor(None, clob_market, cond)
                except Exception:
                    continue
                if c and c.get("closed") is True:
                    winners = {t.get("outcome"): t.get("winner")
                               for t in c.get("tokens", [])}
                    jl_append(ROOT / "resolutions.jsonl", {
                        "recv_ns": time.time_ns(), "slug": slug,
                        "conditionId": cond, "closed": True,
                        "winners": winners, "source": "clob"})
                    del self.pending_res[slug]
                    self.counts["resolved"] += 1
                elif now > end + RESOLVE_GIVEUP_S:
                    jl_append(ROOT / "resolutions.jsonl",
                              {"recv_ns": time.time_ns(), "slug": slug,
                               "conditionId": cond, "closed": None, "gave_up": True})
                    del self.pending_res[slug]

    # ---- plumbing -------------------------------------------------------------
    async def _sleep(self, secs: float) -> None:
        t0 = time.time()
        while not self.stop and time.time() - t0 < secs:
            await asyncio.sleep(1)

    async def _heartbeat(self) -> None:
        while not self.stop:
            await self._sleep(60)
            if self.stop:
                break
            print(f"[pm] {time.strftime('%H:%M:%S', time.gmtime())}Z "
                  f"markets={self.counts['markets']} msgs={self.counts['msgs']} "
                  f"resolved={self.counts['resolved']} pending={len(self.pending_res)}",
                  flush=True)

    async def run(self) -> None:
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"[pm] coins={COINS} window={WINDOW_S}s grace={GRACE_S}s", flush=True)
        for slug, ts, toks in self.resume:   # MUST-FIX iter-1: continue in-flight windows
            print(f"[pm] resuming in-flight {slug}", flush=True)
            asyncio.create_task(self._market(slug, ts, toks))
        await asyncio.gather(self._discovery(), self._resolver(), self._heartbeat())
        print("[pm] stopped", flush=True)


def main() -> None:
    c = PMCollector()

    def _sig(*_):
        print("[pm] shutdown signal", flush=True)
        c.stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    asyncio.run(c.run())


if __name__ == "__main__":
    main()
