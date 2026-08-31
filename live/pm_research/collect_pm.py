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
import random
import shutil
import signal
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
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
# The service starts without arguments, so the reviewed v4 behavior remains
# the restart-safe default until a separately stamped boundary opts into v5.
# This prevents an unrelated process restart from creating an unrecorded era.
COLLECTOR_VERSION = "clob_v4"
HEARTBEAT_CONTROL_V4 = "control-v4"
HEARTBEAT_APP_V5 = "app-v5"
HEARTBEAT_MODES = (HEARTBEAT_CONTROL_V4, HEARTBEAT_APP_V5)
# The venue documents a TEN-SECOND PING/PONG cadence; that constrains how
# often we SEND, not how long we wait for the answer, which is the client's
# choice. Worst-case dead-socket blindness is interval + timeout, and the
# v4 keepalive O1a tuned it to ~6 s (77.1% of btc lost seconds was detection
# lag). A 10 s deadline would have restored ~20 s by a different mechanism —
# the exact metric O1a fixed, and it lands INSIDE the recorded gap durations
# the day-quality gate accrues on. Observed PONG round-trip on the live BTC
# channel is ~90 ms, so 3 s is ~33x the observed latency and still bounds
# blindness, which with the 3 s interval below is 6.0 s.
# AUTHORITY (V5-C5-4): the venue documentation says "Client heartbeat — send
# every 10 seconds." It does NOT call ten seconds a minimum and does not
# authorize a faster cadence. 3 s is an EMPIRICALLY TESTED DEVIATION: live
# shadow probes on the real BTC channel returned 24/24 and 8/8 PONGs with
# zero disconnects, but both attached to an EXPIRED slug, so they establish
# transport tolerance only — not concurrent-flow behaviour, not long-run
# server policy. Residual accepted by USER ruling 2026-08-31.
# At 3 s + 3 s the worst-case blindness
# matches the v4 keepalive O1a tuned to (~6 s), which removes the detection
# regression entirely rather than merely reducing it. Cost: one 4-byte text
# frame per socket per 3 s (~7/s across ~21 sockets), against ~600 market
# messages/s on the same connections.
APP_HEARTBEAT_INTERVAL_S = 3.0
APP_HEARTBEAT_TIMEOUT_S = 3.0

# v3: disk work and HTTP work get SEPARATE executors. They shared the default
# 20-worker pool, where four run_in_executor HTTP calls (urlopen, 15-25 s
# timeouts) could starve the writers' to_thread; a full write_q then makes
# flush_buf block INSIDE the receive loop, which is exactly how a 1013 is
# produced. Disk is serialised on purpose: one shard compressed at a time
# bounds peak memory and keeps gzip off both the loop and the HTTP path.
DISK_WORKERS = 2
HTTP_WORKERS = 8
LAG_PROBE_S = 0.1            # event-loop lag probe interval
LAG_WARN_MS = 250.0          # log a stall at or above this


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


class SubscribeUnconfirmed(Exception):
    """O1c (R-232): no first message within SUBSCRIBE_CONFIRM_S of subscribe.

    Before this, a silent no-subscribe was indistinguishable from a quiet
    market — an invisible-hole class. The CLOB market channel sends a book
    snapshot on subscribe, so a silent socket is a broken subscription, and
    raising here re-enters the connect loop (which re-subscribes) under a
    cause of its own."""


SUBSCRIBE_CONFIRM_S = 10.0


class AppHeartbeatTimeout(Exception):
    """The documented text PING didn't receive an exact text PONG in time."""


def connect_keepalive_kwargs(heartbeat_mode: str) -> dict:
    """Return explicit library keepalive settings for a collector era.

    Polymarket's market-channel contract is an application message: send the
    exact text ``PING`` every ten seconds and consume exact text ``PONG``.
    ``websockets`` automatic keepalive is a different protocol mechanism (RFC
    control frames).  v5 disables it rather than running two independent
    liveness authorities on the same socket.
    """
    if heartbeat_mode == HEARTBEAT_CONTROL_V4:
        return {"ping_interval": 3, "ping_timeout": 3}
    if heartbeat_mode == HEARTBEAT_APP_V5:
        return {"ping_interval": None, "ping_timeout": None}
    raise ValueError(f"unknown heartbeat mode: {heartbeat_mode!r}")


async def application_heartbeat(ws, subscription_ready: asyncio.Event,
                                pong_received: asyncio.Event,
                                counts=None) -> None:
    """Send one documented heartbeat at a time after subscription confirms."""
    await subscription_ready.wait()
    await asyncio.sleep(APP_HEARTBEAT_INTERVAL_S)
    while True:
        sent_at = time.monotonic()
        pong_received.clear()
        await ws.send("PING")
        if counts is not None:
            counts["app_heartbeat_pings"] += 1
        try:
            await asyncio.wait_for(pong_received.wait(),
                                   timeout=APP_HEARTBEAT_TIMEOUT_S)
        except asyncio.TimeoutError as ex:
            raise AppHeartbeatTimeout(
                f"no exact text PONG within {APP_HEARTBEAT_TIMEOUT_S:.1f}s"
            ) from ex
        # Schedule from send time rather than response time. A normal 90 ms
        # response must not slowly turn the documented 10 s cadence into 10.09,
        # 10.18, ... across a long-lived connection.
        elapsed = time.monotonic() - sent_at
        await asyncio.sleep(max(0.0, APP_HEARTBEAT_INTERVAL_S - elapsed))


async def run_with_application_heartbeat(ws, receive_coro,
                                         subscription_ready: asyncio.Event,
                                         pong_received: asyncio.Event,
                                         counts=None) -> None:
    """Run the sole receiver and heartbeat sender; propagate either failure."""
    receive_task = asyncio.create_task(receive_coro)
    heartbeat_task = asyncio.create_task(
        application_heartbeat(ws, subscription_ready, pong_received, counts)
    )
    tasks = (receive_task, heartbeat_task)
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        # Awaiting the completed task is what turns a heartbeat refusal into a
        # classified reconnect. Merely noticing `done` would hide the error.
        for task in done:
            await task
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def ws_cause(ex: BaseException) -> str:
    if isinstance(ex, SubscribeUnconfirmed):
        return "SUBSCRIBE_UNCONFIRMED"
    if isinstance(ex, AppHeartbeatTimeout):
        return "APP_HEARTBEAT_TIMEOUT"
    text = str(ex).lower()
    if "1013" in text or "slow" in text:
        return "SLOW_CONSUMER_1013"
    if "ping timeout" in text:
        return "PING_TIMEOUT"
    if "no close" in text:
        return "NO_CLOSE_FRAME"
    return type(ex).__name__.upper()


def reconnect_delay(attempt: int, cause: str, u: float) -> float:
    """O1b (R-232): cause-aware backoff with full jitter, replacing flat 1 s.

    Persistent faults were hammered at 1 Hz — consistent with the measured
    49%-within-60s burst clustering. Exponential per consecutive FAILED
    attempt (a connection that delivered messages resets the ladder), cap
    30 s. SLOW_CONSUMER_1013 starts at 2 s so the venue's backoff signal is
    never retried FASTER than network causes. `u` in [0,1) scales the delay
    into (0.5..1.0]x so synchronized reconnect stampedes decorrelate."""
    base = 2.0 if cause == "SLOW_CONSUMER_1013" else 1.0
    d = min(30.0, base * (2 ** min(max(attempt - 1, 0), 5)))
    return d * (0.5 + 0.5 * u)


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
    def __init__(self, heartbeat_mode: str = HEARTBEAT_CONTROL_V4):
        if heartbeat_mode not in HEARTBEAT_MODES:
            raise ValueError(f"unknown heartbeat mode: {heartbeat_mode!r}")
        self.heartbeat_mode = heartbeat_mode
        self.collector_version = (
            "clob_v5" if heartbeat_mode == HEARTBEAT_APP_V5 else COLLECTOR_VERSION
        )
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
        self.active_market_coin: dict[str, str] = {}
        self.last_recv_by_slug: dict[str, int] = {}
        self.markets_completed = 0          # exited cooperatively, before shutdown
        # Separate pools so a stalled HTTP call can never delay a disk write.
        self.disk_pool = ThreadPoolExecutor(max_workers=DISK_WORKERS,
                                            thread_name_prefix="pm-disk")
        self.http_pool = ThreadPoolExecutor(max_workers=HTTP_WORKERS,
                                            thread_name_prefix="pm-http")
        self.lag_ms_max = 0.0               # worst loop stall since last heartbeat
        self.lag_ms_max_ever = 0.0
        self.lag_stalls = 0
        self._load_state()

    async def _lag_probe(self) -> None:
        """Measure event-loop stalls directly.

        A 1013 has two candidate causes with OPPOSITE fixes: the loop stalled
        (offload work) or the socket rate genuinely exceeded capacity (shard
        connections). Without this number the next disconnect is a guess. The
        overshoot of a fixed-interval sleep IS the time the loop spent unable to
        drain any socket.
        """
        while not self.stop:
            t0 = time.perf_counter()
            await asyncio.sleep(LAG_PROBE_S)
            lag = (time.perf_counter() - t0 - LAG_PROBE_S) * 1e3
            if lag > self.lag_ms_max:
                self.lag_ms_max = lag
            if lag > self.lag_ms_max_ever:
                self.lag_ms_max_ever = lag
            if lag >= LAG_WARN_MS:
                self.lag_stalls += 1
                self._audit({"event": "loop_stall", "lag_ms": round(lag, 1)})

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
                                   "collector_version": self.collector_version,
                                   **obj})
        except Exception as ex:
            self.counts["audit_errors"] += 1
            print(f"[pm] audit append: {type(ex).__name__} {str(ex)[:60]}", flush=True)

    async def _rewards_refresh(self) -> None:
        loop = asyncio.get_event_loop()
        while not self.stop:
            now = int(time.time())
            try:
                self.rewards = await loop.run_in_executor(self.http_pool, rewards_registry)
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
            self.markets_completed += 1
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
                        m = await loop.run_in_executor(self.http_pool, gget, slug)
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
                        c = await loop.run_in_executor(self.http_pool, clob_market, m.get("conditionId"))
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
        self.active_market_coin[slug] = coin
        last_recv_ns: int | None = None
        open_gap: dict | None = None
        qmax = 0                 # deepest inbound frame backlog seen
        ever_paused = False      # did the library ever stop reading the socket?

        async def writer() -> None:
            while True:
                chunk = await write_q.get()
                try:
                    if chunk is None:
                        return
                    await asyncio.get_running_loop().run_in_executor(
                        self.disk_pool, f.write, chunk)
                finally:
                    write_q.task_done()

        writer_task = asyncio.create_task(writer())

        pending: str | None = None      # chunk taken from buf, not yet queued

        async def flush_buf() -> None:
            nonlocal pending
            if pending is None:
                if not buf:
                    return
                if writer_task.done():
                    writer_task.result()
                pending = "".join(buf)
                buf.clear()
            try:
                write_q.put_nowait(pending)
            except asyncio.QueueFull:
                # `pending` survives a cancellation landing on this await, so
                # the finally can re-flush it. Clearing buf before the chunk is
                # safely queued was a (narrow) data-loss window.
                self.counts["writer_backpressure"] += 1
                await write_q.put(pending)
            pending = None
            self.counts["writer_queue_highwater"] = max(
                self.counts["writer_queue_highwater"], write_q.qsize())

        async def receive(ws, subscription_ready: asyncio.Event | None = None,
                          pong_received: asyncio.Event | None = None) -> None:
            nonlocal last_recv_ns, open_gap, qmax, ever_paused
            confirmed = False
            while not self.stop:
                if confirmed:
                    raw = await ws.recv()
                else:
                    # O1c (R-232): the FIRST message doubles as the subscribe
                    # confirmation. The market channel sends a book snapshot on
                    # subscribe, so silence here is a dead subscription, not a
                    # quiet market. One bool branch; the steady-state recv path
                    # above is unchanged.
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=SUBSCRIBE_CONFIRM_S)
                    except asyncio.TimeoutError:
                        raise SubscribeUnconfirmed(
                            f"no first message within {SUBSCRIBE_CONFIRM_S}s "
                            f"of subscribe")
                # v5 heartbeat frames are application TEXT messages, not RFC
                # control frames. They share the one receive stream, so consume
                # exact PONG here and never write it into the JSON market tape.
                if pong_received is not None and raw == "PONG":
                    pong_received.set()
                    self.counts["app_heartbeat_pongs"] += 1
                    continue
                if not confirmed:
                    confirmed = True
                    if subscription_ready is not None:
                        subscription_ready.set()
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
                self.last_recv_by_slug[slug] = recv_ns
                # THE DISCRIMINATOR (clob_v3_1). websockets buffers inbound
                # frames in an Assembler and PAUSES READING FROM THE TRANSPORT
                # when the backlog exceeds its high-water mark -- which is
                # exactly what makes the venue see a slow consumer and send
                # 1013. So `paused` answers the question directly:
                #   paused/deep backlog at a 1013  => WE are genuinely behind
                #                                     (reduce per-message work,
                #                                      or shard sockets)
                #   backlog ~0 and never paused    => the venue closed us for
                #                                     its own reasons; not
                #                                     fixable client-side
                # v3 ruled out loop stalls, write backpressure, memory and raw
                # rate, and still could not tell these two apart.
                try:
                    asm = ws.recv_messages
                    depth = len(asm.frames)
                    if depth > qmax:
                        qmax = depth
                    if asm.paused:
                        ever_paused = True
                    self.counts["ws_queue_highwater"] = max(
                        self.counts["ws_queue_highwater"], depth)
                except Exception:
                    pass
                buf.append(f"{recv_ns}\t{raw}\n")
                self.counts["msgs"] += 1
                self.msg_by_coin[coin] += 1
                if len(buf) >= WRITE_BATCH:
                    await flush_buf()

        writer_ok = True
        # O1d (R-232): coverage for this window-task begins HERE. A socket that
        # never delivers a message has been blind since this instant, not since
        # the moment its error surfaced — stamping gap_start at the error
        # recorded such gaps SHORTER than they were, and this ledger is what
        # the tape pins and the gate counts.
        scope_start_ns = time.time_ns()
        consec_fail = 0
        try:
            while not self.stop and time.time() < stop_at:
                attempt_msgs = self.msg_by_coin.get(coin, 0)
                try:
                    # max_queue: the default (32) makes the server drop us with
                    # 1013 "slow consumer" on hot markets — observed repeatedly on
                    # BTC, i.e. load-correlated loss on ~85% of notional. Deep queue
                    # + batched writes keep the recv loop free of blocking I/O.
                    # v4 O1a uses RFC control ping 3/3. The inert v5 candidate
                    # instead disables that mechanism and runs the documented
                    # text PING/PONG lifecycle in the single receive stream.
                    keepalive = connect_keepalive_kwargs(self.heartbeat_mode)
                    async with websockets.connect(WS_URL, open_timeout=15,
                                                  max_queue=2 ** 16,
                                                  **keepalive) as ws:
                        await ws.send(json.dumps({"assets_ids": toks, "type": "market"}))
                        try:
                            # Exactly one deadline timer per connection. The receive
                            # loop does no JSON parsing and never waits for a disk write.
                            if self.heartbeat_mode == HEARTBEAT_APP_V5:
                                subscription_ready = asyncio.Event()
                                pong_received = asyncio.Event()
                                connection = run_with_application_heartbeat(
                                    ws,
                                    receive(ws, subscription_ready, pong_received),
                                    subscription_ready,
                                    pong_received,
                                    self.counts,
                                )
                            else:
                                connection = receive(ws)
                            await asyncio.wait_for(
                                connection,
                                timeout=max(0.0, stop_at - time.time()),
                            )
                        except asyncio.TimeoutError:
                            break
                except Exception as ex:
                    cause = ws_cause(ex)
                    self.counts["retries"] += 1
                    self.retry_by_coin[coin] += 1
                    if self.msg_by_coin.get(coin, 0) > attempt_msgs:
                        consec_fail = 1        # that connection worked, then died
                    else:
                        consec_fail += 1       # never delivered a message
                    if cause == "SLOW_CONSUMER_1013":
                        self.counts["slow_drops"] += 1
                        self.slow_by_coin[coin] += 1
                    if open_gap is None:
                        # O1d: fall back to last COVERAGE (scope start), never
                        # to the error instant — see scope_start_ns above.
                        open_gap = {"cause": cause,
                                    "gap_start_ns": last_recv_ns or scope_start_ns}
                    self._audit({
                        "event": "disconnect", "slug": slug, "coin": coin,
                        "tokens": toks, "window_start": ts, "cause": cause,
                        "last_message_recv_ns": last_recv_ns,
                        # attribution: was the loop stalled, or did the socket
                        # rate simply exceed capacity? These have opposite fixes.
                        "lag_ms_max_interval": round(self.lag_ms_max, 1),
                        "lag_ms_max_ever": round(self.lag_ms_max_ever, 1),
                        "coin_msg_rate_hint": self.msg_by_coin.get(coin, 0),
                        # clob_v3_1: were WE the slow consumer, or the venue?
                        "ws_queue_depth_max": qmax,
                        "ws_ever_paused": ever_paused,
                        "error": str(ex)[:240],
                    })
                    if time.time() < stop_at and not self.stop:
                        # O1b: cause-aware exponential backoff with jitter.
                        delay = reconnect_delay(consec_fail, cause, random.random())
                        print(f"[pm] {slug} ws: {type(ex).__name__} {str(ex)[:60]}"
                              f" — retry in {delay:.1f}s (fail #{consec_fail})",
                              flush=True)
                        await asyncio.sleep(delay)
        finally:
            if open_gap is not None:
                # An outage running to window end otherwise leaves a `disconnect`
                # with no matching `gap_closed`, and a consumer cannot tell that
                # from a lost close record. The Route-A selection audit needs
                # every gap to have an explicit end.
                end_ns = time.time_ns()
                self._audit({
                    "event": "gap_open_at_exit", "slug": slug, "coin": coin,
                    "tokens": toks, "window_start": ts,
                    "cause": open_gap["cause"],
                    "gap_start_ns": open_gap["gap_start_ns"],
                    "gap_end_ns": end_ns,
                    "duration_ms": (end_ns - open_gap["gap_start_ns"]) / 1e6,
                    "note": "outage ran to window end; never reconnected",
                })
                open_gap = None
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
                # NEVER on the event loop: measured 1.8-1.9 s to compress a
                # ~180 MB BTC shard at level 6, during which NO socket is
                # drained and the server's send buffer to us fills -> 1013.
                # Runs on the disk pool, not the shared default pool, so a
                # stalled HTTP call cannot delay it and it cannot delay a write.
                t0 = time.perf_counter()
                await asyncio.get_running_loop().run_in_executor(
                    self.disk_pool, gzip_atomic, path, gz)
                gz_ms = (time.perf_counter() - t0) * 1e3
                self.counts["gzip_ms_max"] = max(self.counts["gzip_ms_max"],
                                                 int(gz_ms))
            except Exception as ex:
                print(f"[pm] gzip {slug}: {ex}", flush=True)
            self.active_market_coin.pop(slug, None)
            self.last_recv_by_slug.pop(slug, None)

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
                    c = await loop.run_in_executor(self.http_pool, clob_market, cond)
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
        prior = Counter()
        prior_at = time.monotonic()
        while not self.stop:
            await self._sleep(60)
            if self.stop:
                break
            now_mono = time.monotonic()
            elapsed = max(now_mono - prior_at, 1e-9)
            rates = {coin: round((count - prior.get(coin, 0)) / elapsed, 1)
                     for coin, count in sorted(self.msg_by_coin.items())}
            prior = Counter(self.msg_by_coin)
            prior_at = now_mono
            active = Counter(self.active_market_coin.values())
            unseen = Counter()
            oldest_age: dict[str, float] = {}
            now_ns = time.time_ns()
            for slug, coin in self.active_market_coin.items():
                recv_ns = self.last_recv_by_slug.get(slug)
                if recv_ns is None:
                    unseen[coin] += 1
                else:
                    age = round((now_ns - recv_ns) / 1e9, 2)
                    oldest_age[coin] = max(oldest_age.get(coin, 0.0), age)
            print(f"[pm] {time.strftime('%H:%M:%S', time.gmtime())}Z "
                  f"markets={self.counts['markets']} msgs={self.counts['msgs']} "
                  f"resolved={self.counts['resolved']} pending={len(self.pending_res)} "
                  f"retries={self.counts['retries']} slow={self.counts['slow_drops']} "
                  f"app_ping={self.counts['app_heartbeat_pings']} "
                  f"app_pong={self.counts['app_heartbeat_pongs']} "
                  f"writer_wait={self.counts['writer_backpressure']} "
                  f"q_hi={self.counts['writer_queue_highwater']} "
                  f"lag_ms_max={self.lag_ms_max:.0f} "
                  f"lag_stalls={self.lag_stalls} "
                  f"gzip_ms_max={self.counts['gzip_ms_max']} "
                  f"ws_q_hi={self.counts['ws_queue_highwater']} "
                  f"active={dict(active)} unseen={dict(unseen)} "
                  f"oldest_age_s={oldest_age} rate_msg_s={rates} "
                  f"msg_by_coin={dict(self.msg_by_coin)} "
                  f"retry_by_coin={dict(self.retry_by_coin)} "
                  f"slow_by_coin={dict(self.slow_by_coin)}",
                  flush=True)
            self.lag_ms_max = 0.0        # per-interval high-water

    async def run(self) -> None:
        RAW.mkdir(parents=True, exist_ok=True)
        self._audit({"event": "collector_start", "pid": os.getpid()})
        print(f"[pm] version={self.collector_version} "
              f"heartbeat_mode={self.heartbeat_mode} coins={COINS} "
              f"window={WINDOW_S}s grace={GRACE_S}s", flush=True)
        for slug, ts, toks in self.resume:   # MUST-FIX iter-1: continue in-flight windows
            print(f"[pm] resuming in-flight {slug}", flush=True)
            self._spawn_market(slug, ts, toks)
        try:
            await asyncio.gather(self._discovery(), self._resolver(), self._heartbeat(),
                                 self._rewards_refresh(), self._lag_probe())
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
                         # v2 called this "active_markets_drained", which read as
                         # "markets drained" (14 at the v2 stop) when it counts
                         # only those needing a forced cancel (2). A later
                         # auditor would reasonably infer 12 lost shards.
                         "markets_force_cancelled": len(active),
                         "markets_completed_cooperatively": self.markets_completed,
                         "lag_ms_max_ever": round(self.lag_ms_max_ever, 1),
                         "loop_stalls": self.lag_stalls})
            self.disk_pool.shutdown(wait=True)
            self.http_pool.shutdown(wait=False)
            print("[pm] stopped", flush=True)


def selftest() -> int:
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d) / "raw.jsonl"
        dest = Path(d) / "raw.jsonl.gz"
        raw.write_bytes(b"one\ntwo\n")
        gzip_atomic(raw, dest)
        atomic_ok = (not raw.exists() and gzip.open(dest, "rb").read() == b"one\ntwo\n"
                     and not list(Path(d).glob("*.tmp-*")))
    try:
        connect_keepalive_kwargs("unknown")
        unknown_mode_refused = False
    except ValueError:
        unknown_mode_refused = True
    checks = [
        ("closed resolution is final", is_final({"closed": True})),
        ("give-up remains retryable", not is_final({"gave_up": True})),
        ("degenerate outcome is final", is_final({"outcomePrices": '["1", "0"]'})),
        ("open live prices are not final", not is_final({"outcomePrices": '[".4", ".6"]'})),
        ("slow close classified", ws_cause(Exception("received 1013 slow consumer"))
         == "SLOW_CONSUMER_1013"),
        ("ping timeout classified", ws_cause(Exception("keepalive ping timeout"))
         == "PING_TIMEOUT"),
        ("application heartbeat timeout classified by identity",
         ws_cause(AppHeartbeatTimeout("no exact text PONG"))
         == "APP_HEARTBEAT_TIMEOUT"),
        ("v5 disables RFC control-frame liveness",
         connect_keepalive_kwargs(HEARTBEAT_APP_V5)
         == {"ping_interval": None, "ping_timeout": None}),
        ("known-bad heartbeat mode is refused", unknown_mode_refused),
        ("gzip publish is atomic", atomic_ok),
    ]
    checks.extend(_async_checks())
    return _report(checks)


def _report(checks) -> int:
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(ok for _, ok in checks) else 1


def _async_checks() -> list[tuple[str, bool]]:
    """The v2 defect was a 1.9 s event-loop stall while gzipping a BTC shard.
    Asserting the call is `await`ed is not enough -- measure the loop."""
    async def _run() -> list[tuple[str, bool]]:
        out = []
        pool = ThreadPoolExecutor(max_workers=DISK_WORKERS)
        worst = 0.0

        async def probe() -> None:
            nonlocal worst
            while True:
                t0 = time.perf_counter()
                await asyncio.sleep(0.005)
                worst = max(worst, (time.perf_counter() - t0 - 0.005) * 1e3)

        async def measure(fn) -> float:
            """Worst loop lag observed while `fn` runs. Always lets the probe
            resume BEFORE cancelling, or a blocking call is never observed."""
            nonlocal worst
            worst = 0.0
            task = asyncio.create_task(probe())
            await asyncio.sleep(0.02)          # let the probe reach its first await
            await fn()
            await asyncio.sleep(0.02)          # let it resume and record the overshoot
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return worst

        with tempfile.TemporaryDirectory() as d:
            # Sized so that compressing it ON the loop clearly exceeds
            # LAG_WARN_MS; a shard too small to stall proves nothing.
            line = ('{"event_type":"book","asset_id":"%s","bids":[{"price":"0.42",'
                    '"size":"1500"}],"asks":[{"price":"0.44","size":"900"}]}\n' % ("7" * 60))
            payload = line * 400_000

            big = Path(d) / "big.jsonl"
            big.write_text(payload)
            gz = Path(str(big) + ".gz")
            t0 = time.perf_counter()
            off = await measure(lambda: asyncio.get_running_loop().run_in_executor(
                pool, gzip_atomic, big, gz))
            gz_ms = (time.perf_counter() - t0) * 1e3
            out.append((f"gzip ran off-loop: {gz_ms:.0f} ms of work, "
                        f"worst loop lag {off:.1f} ms", off < LAG_WARN_MS))
            out.append(("gzip still published atomically from the pool",
                        gz.exists() and not big.exists()))

            # Control: the SAME work inline must trip the probe, else the test
            # above proves nothing about the fix.
            big2 = Path(d) / "big2.jsonl"
            big2.write_text(payload)

            async def inline() -> None:
                gzip_atomic(big2, Path(str(big2) + ".gz"))

            on = await measure(inline)
            # Oracle is the SEPARATION, not LAG_WARN_MS: that constant is a
            # production alerting threshold and tuning it must not break this
            # test. On-loop must stall by orders of magnitude more, and by an
            # amount that is unambiguously a stall.
            out.append((f"control: same gzip ON the loop stalls it "
                        f"({on:.0f} ms vs {off:.1f} ms off-loop, "
                        f"{on / max(off, 0.1):.0f}x)",
                        on >= 100.0 and on > 20 * max(off, 0.1)))
        pool.shutdown(wait=True)

        c = PMCollector.__new__(PMCollector)
        c.disk_pool = ThreadPoolExecutor(max_workers=DISK_WORKERS)
        c.http_pool = ThreadPoolExecutor(max_workers=HTTP_WORKERS)
        out.append(("disk and HTTP pools are distinct objects",
                    c.disk_pool is not c.http_pool))
        import threading
        release = threading.Event()
        for _ in range(HTTP_WORKERS):
            c.http_pool.submit(release.wait)
        t0 = time.perf_counter()
        await asyncio.get_running_loop().run_in_executor(c.disk_pool, lambda: None)
        starve_ms = (time.perf_counter() - t0) * 1e3
        release.set()
        out.append((f"disk work unaffected by a fully saturated HTTP pool "
                    f"({starve_ms:.1f} ms)", starve_ms < 100))
        c.disk_pool.shutdown(wait=True)
        c.http_pool.shutdown(wait=True)

        # clob_v3_1: prove the slow-consumer discriminator actually flips.
        # A flag that never becomes True would be indistinguishable from
        # "we were never behind" -- the failure mode this whole session keeps
        # finding.
        try:
            import websockets.asyncio.messages as wsm
            from websockets.frames import Frame, Opcode
            asm = wsm.Assembler(high=2, low=1)
            out.append(("discriminator starts clean",
                        len(asm.frames) == 0 and not asm.paused))
            for _ in range(5):
                asm.put(Frame(Opcode.TEXT, b"x", fin=True))
            out.append((f"discriminator FLIPS when the backlog exceeds high "
                        f"(depth={len(asm.frames)}, paused={asm.paused})",
                        len(asm.frames) > asm.high and asm.paused is True))
        except Exception as ex:
            out.append((f"discriminator testable against installed websockets "
                        f"({type(ex).__name__}: {str(ex)[:60]})", False))
        return out

    return asyncio.run(_run())



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--heartbeat-mode", choices=HEARTBEAT_MODES,
                    default=HEARTBEAT_CONTROL_V4,
                    help=("restart-safe default is reviewed clob_v4; app-v5 "
                          "requires a separately stamped deployment boundary"))
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    c = PMCollector(heartbeat_mode=args.heartbeat_mode)

    def _sig(*_):
        print("[pm] shutdown signal", flush=True)
        c.stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    asyncio.run(c.run())


if __name__ == "__main__":
    main()
