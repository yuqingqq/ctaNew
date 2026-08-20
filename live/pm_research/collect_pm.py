"""
Polymarket crypto 5-min "Up or Down" market collector (P-2026-003, E0).

Markets: {btc,eth,sol,xrp,doge,bnb,hype}-updown-5m-<unix_window_start>,
300 s-aligned windows, pre-created ~5 min ahead (verified 2026-08-19). Per
market we record the full
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
  resolution every 60 s: markets past end+120 s are polled on the persistent
             CLOB endpoint until closed → append to resolutions.jsonl

Signal-side data (Binance bookTicker/depth/trades for the same coins) is already
collected by live/mm_research/collect_hf.py — this collector is the market side.

NOTE Gamma rejects the default python UA (Cloudflare 403) — a curl UA works.

Run:  nohup python3 live/pm_research/collect_pm.py > data/pm_5min/collector.log 2>&1 &
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import shutil
import signal
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
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
WRITE_BATCH = 2_000
WRITE_QUEUE_MAX = 32
GAP_LEDGER = ROOT / "collector_gaps.jsonl"
COLLECTOR_VERSION = "clob_v2"


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


def gzip_atomic(path: Path, dest: Path) -> None:
    """Publish gzip only after a complete archive exists; preserve raw on failure."""
    tmp = Path(f"{dest}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with open(path, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        tmp.replace(dest)
        path.unlink()
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def ws_cause(ex: BaseException) -> str:
    text = str(ex).lower()
    if "1013" in text or "slow" in text:
        return "SLOW_CONSUMER_1013"
    if "ping timeout" in text:
        return "PING_TIMEOUT"
    if "no close" in text:
        return "NO_CLOSE_FRAME"
    return type(ex).__name__.upper()


def is_final(m: dict) -> bool:
    """A market is RESOLVED only if closed, or outcomePrices degenerate to {0,1}.
    Gamma populates outcomePrices continuously for OPEN markets (live prices) —
    treating their mere presence as resolution records garbage (bug found
    2026-08-19: rows with closed=false and prices like 0.165/0.835)."""
    if m.get("closed") is True:
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
        self.gave_up: set[str] = set()       # logged once, but still polled until final
        self.rewards: dict = {}            # condition_id → authoritative rewards params
        self.rewards_ts = 0.0
        self.counts = defaultdict(int)
        self.msg_by_coin = Counter()
        self.retry_by_coin = Counter()
        self.slow_by_coin = Counter()
        self.market_tasks: set[asyncio.Task] = set()
        self._load_state()

    def _load_state(self) -> None:
        """Resume: don't re-collect known markets; re-poll unresolved ones."""
        resolved = set()
        if (ROOT / "resolutions.jsonl").exists():
            for ln in open(ROOT / "resolutions.jsonl"):
                try:
                    r = json.loads(ln)
                    if r.get("gave_up"):
                        self.gave_up.add(r["slug"])
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

    def _audit(self, obj: dict) -> None:
        try:
            jl_append(GAP_LEDGER, {"recv_ns": time.time_ns(),
                                   "collector_version": COLLECTOR_VERSION, **obj})
        except Exception as ex:
            self.counts["audit_errors"] += 1
            print(f"[pm] audit append: {type(ex).__name__} {str(ex)[:60]}", flush=True)

    async def _rewards_refresh(self) -> None:
        loop = asyncio.get_event_loop()
        while not self.stop:
            now = int(time.time())
            try:
                self.rewards = await loop.run_in_executor(None, rewards_registry)
                self.rewards_ts = now
                jl_append(ROOT / "rewards_registry.jsonl",
                          {"recv_ns": time.time_ns(), "n": len(self.rewards)})
            except Exception as ex:
                print(f"[pm] rewards registry: {type(ex).__name__} {str(ex)[:60]}",
                      flush=True)
            await self._sleep(600)

    def _spawn_market(self, slug: str, ts: int, toks: list[str]) -> None:
        """Keep strong references so shutdown can drain every raw-data writer."""
        task = asyncio.create_task(self._market(slug, ts, toks))
        self.market_tasks.add(task)

        def done(finished: asyncio.Task) -> None:
            self.market_tasks.discard(finished)
            if finished.cancelled():
                return
            try:
                finished.result()
            except Exception as ex:
                self.counts["market_task_errors"] += 1
                print(f"[pm] market task {slug}: {type(ex).__name__} {str(ex)[:80]}",
                      flush=True)

        task.add_done_callback(done)

    # ---- discovery ------------------------------------------------------------
    async def _discovery(self) -> None:
        loop = asyncio.get_event_loop()
        while not self.stop:
            now = int(time.time())
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
                    self._spawn_market(slug, ts, toks)
            await self._sleep(DISCOVER_S)

    # ---- per-market WS recorder -----------------------------------------------
    async def _market(self, slug: str, ts: int, toks: list[str]) -> None:
        day = time.strftime("%Y%m%d", time.gmtime(ts))
        path = RAW / day / f"{slug}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        stop_at = ts + WINDOW_S + GRACE_S
        f = open(path, "a", buffering=1 << 20)     # 1 MB userspace buffer
        buf: list[str] = []
        write_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=WRITE_QUEUE_MAX)
        coin = slug.split("-", 1)[0]
        last_recv_ns: int | None = None
        open_gap: dict | None = None

        async def writer() -> None:
            while True:
                chunk = await write_q.get()
                try:
                    if chunk is None:
                        return
                    await asyncio.to_thread(f.write, chunk)
                finally:
                    write_q.task_done()

        writer_task = asyncio.create_task(writer())

        async def flush_buf() -> None:
            if not buf:
                return
            if writer_task.done():
                writer_task.result()
            n_messages = len(buf)
            chunk = "".join(buf)
            buf.clear()
            try:
                write_q.put_nowait(chunk)
            except asyncio.QueueFull:
                self.counts["writer_backpressure"] += 1
                await write_q.put(chunk)
            self.counts["writer_queue_highwater"] = max(
                self.counts["writer_queue_highwater"], write_q.qsize())
            self.msg_by_coin[coin] += n_messages

        async def receive(ws) -> None:
            nonlocal last_recv_ns, open_gap
            while not self.stop:
                raw = await ws.recv()
                recv_ns = time.time_ns()
                if open_gap is not None:
                    self._audit({
                        "event": "gap_closed", "slug": slug, "coin": coin,
                        "tokens": toks, "window_start": ts,
                        "cause": open_gap["cause"],
                        "gap_start_ns": open_gap["gap_start_ns"],
                        "gap_end_ns": recv_ns,
                        "duration_ms": (recv_ns - open_gap["gap_start_ns"]) / 1e6,
                    })
                    open_gap = None
                last_recv_ns = recv_ns
                buf.append(f"{recv_ns}\t{raw}\n")
                self.counts["msgs"] += 1
                if len(buf) >= WRITE_BATCH:
                    await flush_buf()

        writer_ok = True
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
                        try:
                            # Exactly one deadline timer per connection. The receive
                            # loop does no JSON parsing and never waits for a disk write.
                            await asyncio.wait_for(receive(ws),
                                                   timeout=max(0.0, stop_at - time.time()))
                        except asyncio.TimeoutError:
                            break
                except Exception as ex:
                    cause = ws_cause(ex)
                    err_ns = time.time_ns()
                    self.counts["retries"] += 1
                    self.retry_by_coin[coin] += 1
                    if cause == "SLOW_CONSUMER_1013":
                        self.counts["slow_drops"] += 1
                        self.slow_by_coin[coin] += 1
                    if open_gap is None:
                        open_gap = {"cause": cause,
                                    "gap_start_ns": last_recv_ns or err_ns}
                    self._audit({
                        "event": "disconnect", "slug": slug, "coin": coin,
                        "tokens": toks, "window_start": ts, "cause": cause,
                        "last_message_recv_ns": last_recv_ns,
                        "error": str(ex)[:240],
                    })
                    if time.time() < stop_at and not self.stop:
                        print(f"[pm] {slug} ws: {type(ex).__name__} {str(ex)[:60]} — retry",
                              flush=True)
                        await asyncio.sleep(1)
        finally:
            try:
                await flush_buf()
                if not writer_task.done():
                    await write_q.put(None)
                await writer_task
            except Exception as ex:
                writer_ok = False
                self.counts["writer_errors"] += 1
                print(f"[pm] writer {slug}: {type(ex).__name__} {str(ex)[:80]}", flush=True)
                if not writer_task.done():
                    writer_task.cancel()
                    await asyncio.gather(writer_task, return_exceptions=True)
            f.close()
            try:
                if not writer_ok:
                    raise RuntimeError("raw writer failed; preserving uncompressed shard")
                # numbered shards: a resumed window must never clobber the gz a
                # previous (truncated) run wrote — consumers concat all shards
                gz = Path(str(path) + ".gz")
                i = 1
                while gz.exists():
                    gz = Path(str(path) + f".{i}.gz")
                    i += 1
                gzip_atomic(path, gz)
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
                    self.gave_up.discard(slug)
                    self.counts["resolved"] += 1
                elif now > end + RESOLVE_GIVEUP_S and slug not in self.gave_up:
                    jl_append(ROOT / "resolutions.jsonl",
                              {"recv_ns": time.time_ns(), "slug": slug,
                               "conditionId": cond, "closed": None, "gave_up": True})
                    self.gave_up.add(slug)
                    self.counts["resolution_giveups"] += 1

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
                  f"resolved={self.counts['resolved']} pending={len(self.pending_res)} "
                  f"retries={self.counts['retries']} slow={self.counts['slow_drops']} "
                  f"writer_wait={self.counts['writer_backpressure']} "
                  f"q_hi={self.counts['writer_queue_highwater']} "
                  f"msg_by_coin={dict(self.msg_by_coin)} "
                  f"retry_by_coin={dict(self.retry_by_coin)} "
                  f"slow_by_coin={dict(self.slow_by_coin)}",
                  flush=True)

    async def run(self) -> None:
        RAW.mkdir(parents=True, exist_ok=True)
        self._audit({"event": "collector_start", "pid": os.getpid()})
        print(f"[pm] version={COLLECTOR_VERSION} coins={COINS} "
              f"window={WINDOW_S}s grace={GRACE_S}s", flush=True)
        for slug, ts, toks in self.resume:   # MUST-FIX iter-1: continue in-flight windows
            print(f"[pm] resuming in-flight {slug}", flush=True)
            self._spawn_market(slug, ts, toks)
        try:
            await asyncio.gather(self._discovery(), self._resolver(), self._heartbeat(),
                                 self._rewards_refresh())
        finally:
            # A signal may arrive while every market socket is blocked in recv().
            # Cancel explicitly, then wait for each task's finally block to flush,
            # close, and atomically archive its raw shard.
            active = list(self.market_tasks)
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            self._audit({"event": "collector_stop", "pid": os.getpid(),
                         "active_markets_drained": len(active)})
            print("[pm] stopped", flush=True)


def selftest() -> int:
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d) / "raw.jsonl"
        dest = Path(d) / "raw.jsonl.gz"
        raw.write_bytes(b"one\ntwo\n")
        gzip_atomic(raw, dest)
        atomic_ok = (not raw.exists() and gzip.open(dest, "rb").read() == b"one\ntwo\n"
                     and not list(Path(d).glob("*.tmp-*")))
    checks = [
        ("closed resolution is final", is_final({"closed": True})),
        ("give-up remains retryable", not is_final({"gave_up": True})),
        ("degenerate outcome is final", is_final({"outcomePrices": '["1", "0"]'})),
        ("open live prices are not final", not is_final({"outcomePrices": '[".4", ".6"]'})),
        ("slow close classified", ws_cause(Exception("received 1013 slow consumer"))
         == "SLOW_CONSUMER_1013"),
        ("ping timeout classified", ws_cause(Exception("keepalive ping timeout"))
         == "PING_TIMEOUT"),
        ("gzip publish is atomic", atomic_ok),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(ok for _, ok in checks) else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    c = PMCollector()

    def _sig(*_):
        print("[pm] shutdown signal", flush=True)
        c.stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    asyncio.run(c.run())


if __name__ == "__main__":
    main()
