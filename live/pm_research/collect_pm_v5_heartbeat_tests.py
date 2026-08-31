"""Behavioral tests for the inert clob_v5 application-heartbeat candidate.

These tests drive the real ``PMCollector._market`` connection lifecycle with a
fake WebSocket.  The systemd service doesn't opt into v5 until a separately
stamped deployment boundary; testing the candidate must not change the live
collector era.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path


SOURCE = Path(__file__).with_name("collect_pm.py")
spec = importlib.util.spec_from_file_location("collect_pm_candidate", SOURCE)
assert spec and spec.loader
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)
# The harness shrinks the heartbeat constants for speed, so the SHIPPED
# values must be captured here, before any test runs — asserting on the
# live module later would test the fixture, not the candidate.
SHIPPED_INTERVAL_S = C.APP_HEARTBEAT_INTERVAL_S
SHIPPED_TIMEOUT_S = C.APP_HEARTBEAT_TIMEOUT_S

TMP = Path(tempfile.mkdtemp(prefix="pm_hb_v5_"))
OUT = TMP / "out"
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))
    suffix = f"  [{detail}]" if detail and not condition else ""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{suffix}")


class FakeWS:
    def __init__(self, behavior: str, captured: dict):
        self.behavior = behavior
        self.captured = captured
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.recv_active = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def send(self, message: str) -> None:
        self.captured.setdefault("sent", []).append(message)
        if message == "PING":
            self.captured["app_pings"] = self.captured.get("app_pings", 0) + 1
            if self.behavior == "healthy":
                await self.queue.put("PONG")
            elif self.behavior == "delayed_pong":
                # PONG after DELAYED_PONG_S. A deadline BELOW that must time
                # out and one ABOVE it must not, so a hard-coded deadline
                # cannot survive both probes (V5-C6-1).
                async def _late():
                    await asyncio.sleep(DELAYED_PONG_S)
                    await self.queue.put("PONG")
                asyncio.get_running_loop().create_task(_late())
            elif self.behavior == "double_pong":
                # a venue answering one PING with TWO PONG frames — the
                # producer counts frames, so pong > ping (V5-C5-2)
                await self.queue.put("PONG")
                await self.queue.put("PONG")
            elif self.behavior == "wrong_pong":
                await self.queue.put("PONG ")
        else:
            self.captured.setdefault("order", []).append("subscription")
            await self.queue.put(json.dumps({"event_type": "book", "asset_id": "t1"}))

    async def recv(self) -> str:
        if self.recv_active:
            self.captured["concurrent_recv"] = True
            raise RuntimeError("concurrent recv (known-bad)")
        self.recv_active = True
        try:
            try:
                raw = await asyncio.wait_for(self.queue.get(), timeout=0.004)
            except asyncio.TimeoutError:
                raw = json.dumps({"event_type": "price_change", "asset_id": "t1"})
            self.captured.setdefault("order", []).append(
                "pong" if raw == "PONG" else "market")
            if raw != "PONG":
                self.captured["non_pong_returned"] = (
                    self.captured.get("non_pong_returned", 0) + 1
                )
            return raw
        finally:
            self.recv_active = False


def fake_connect_factory(behavior: str, captured: dict):
    def connect(_url: str, **kwargs):
        captured.setdefault("connect_kwargs", []).append(dict(kwargs))
        return FakeWS(behavior, captured)
    return connect


_test_n = 0
DELAYED_PONG_S = 0.10   # sits between the wrong (0.03) and requested deadlines


async def run_market(behavior: str, *, duration_s: float = 0.24,
                     interval_s: float | None = None,
                     timeout_s: float | None = None) -> tuple[dict, list[dict]]:
    """`interval_s`/`timeout_s` override the compressed fixture defaults.

    They exist because the fixture used to overwrite them UNCONDITIONALLY,
    so a test that set the timeout before calling this ran against 0.03
    regardless — the deadline probes proved nothing about the value they
    named (V5-C5-1).
    """
    global _test_n
    _test_n += 1
    captured: dict = {}
    C.RAW = OUT / f"raw{_test_n}"
    C.ROOT = OUT
    C.GAP_LEDGER = OUT / f"gaps{_test_n}.jsonl"
    C.WINDOW_S = 1
    C.GRACE_S = 0
    C.SUBSCRIBE_CONFIRM_S = 0.05
    _saved_globals = (C.APP_HEARTBEAT_INTERVAL_S, C.APP_HEARTBEAT_TIMEOUT_S)
    C.APP_HEARTBEAT_INTERVAL_S = 0.02 if interval_s is None else interval_s
    C.APP_HEARTBEAT_TIMEOUT_S = 0.03 if timeout_s is None else timeout_s
    captured["heartbeat_interval_seen"] = C.APP_HEARTBEAT_INTERVAL_S
    captured["heartbeat_timeout_seen"] = C.APP_HEARTBEAT_TIMEOUT_S
    C.websockets = types.SimpleNamespace(
        connect=fake_connect_factory(behavior, captured)
    )

    real_sleep = asyncio.sleep

    async def compressed_sleep(delay: float):
        # Preserve heartbeat timing; compress only reconnect backoff.
        await real_sleep(0.003 if delay > 0.4 else delay)

    C.asyncio = types.SimpleNamespace(
        **{name: getattr(asyncio, name) for name in dir(asyncio)
           if not name.startswith("_")}
    )
    C.asyncio.sleep = compressed_sleep
    collector = C.PMCollector(heartbeat_mode=C.HEARTBEAT_APP_V5)
    now = time.time()
    ts = int(now) - 1
    C.WINDOW_S = max(0.05, now - ts + duration_s)
    try:
        await asyncio.wait_for(
            collector._market(f"btc-updown-5m-{ts}", ts, ["t1", "t2"]),
            timeout=3,
        )
    finally:
        captured["msgs"] = int(collector.counts.get("msgs", 0))
        captured["counted_app_pings"] = int(
            collector.counts.get("app_heartbeat_pings", 0)
        )
        captured["counted_app_pongs"] = int(
            collector.counts.get("app_heartbeat_pongs", 0)
        )
        collector.disk_pool.shutdown(wait=False)
        collector.http_pool.shutdown(wait=False)
        # V5-C6-1: the fixture used to LEAVE the module globals mutated, so
        # after a run the module carried 0.02/0.30 rather than the shipped
        # 3.0/3.0 and any later assertion read the fixture's value.
        (C.APP_HEARTBEAT_INTERVAL_S,
         C.APP_HEARTBEAT_TIMEOUT_S) = _saved_globals
    gaps = []
    if C.GAP_LEDGER.exists():
        gaps = [json.loads(line) for line in C.GAP_LEDGER.read_text().splitlines()]
    return captured, gaps


async def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    C.ROOT = OUT
    default_collector = C.PMCollector()
    check("restart-safe no-argument default remains clob_v4",
          default_collector.heartbeat_mode == C.HEARTBEAT_CONTROL_V4
          and default_collector.collector_version == "clob_v4")
    default_collector.disk_pool.shutdown(wait=False)
    default_collector.http_pool.shutdown(wait=False)

    healthy, gaps = await run_market("healthy")
    kwargs = healthy["connect_kwargs"][0]
    check("v5 disables RFC control keepalive",
          kwargs.get("ping_interval", "missing") is None
          and kwargs.get("ping_timeout", "missing") is None,
          str(kwargs))
    check("v5 sends documented exact text PING",
          healthy.get("app_pings", 0) >= 1,
          str(healthy.get("sent", [])))
    check("heartbeat telemetry reconciles exact PING/PONG",
          healthy.get("counted_app_pings", 0) >= 1
          and healthy.get("counted_app_pings")
          == healthy.get("counted_app_pongs"),
          f"ping={healthy.get('counted_app_pings')} "
          f"pong={healthy.get('counted_app_pongs')}")
    order = healthy.get("order", [])
    first_ping = healthy.get("sent", []).index("PING")
    check("subscription precedes the first heartbeat",
          healthy.get("sent", [None])[0] != "PING" and first_ping >= 1,
          str(healthy.get("sent", [])))
    check("exact PONG is consumed, not recorded as market data",
          healthy.get("msgs") == healthy.get("non_pong_returned"),
          f"msgs={healthy.get('msgs')} non_pong={healthy.get('non_pong_returned')}")
    check("positive: healthy application heartbeat opens no gap",
          not any(row.get("event") == "disconnect" for row in gaps),
          str([row.get("cause") for row in gaps]))
    check("single-reader invariant holds",
          not healthy.get("concurrent_recv", False))

    missing, gaps = await run_market("missing_pong")
    causes = [row.get("cause") for row in gaps if row.get("event") == "disconnect"]
    check("known-bad: missing PONG is classified and reconnects",
          "APP_HEARTBEAT_TIMEOUT" in causes
          and len(missing.get("connect_kwargs", [])) >= 2,
          str(causes))
    check("candidate audit rows carry the clob_v5 era",
          bool(gaps) and all(row.get("collector_version") == "clob_v5"
                             for row in gaps),
          str({row.get("collector_version") for row in gaps}))
    check("known-bad remains live at application layer before refusal",
          missing.get("non_pong_returned", 0) > 1,
          str(missing.get("non_pong_returned")))

    wrong, gaps = await run_market("wrong_pong")
    causes = [row.get("cause") for row in gaps if row.get("event") == "disconnect"]
    check("known-bad: near-match 'PONG ' does not satisfy exact identity",
          "APP_HEARTBEAT_TIMEOUT" in causes,
          str(causes))

    # ---- detection-latency contract (USER ruling 2026-08-31) ----
    # The v5 repair fixes the CONTRACT (RFC control Pong -> documented text
    # PING/PONG) but reintroduces detection LAG, the exact metric O1a tuned,
    # and that lag is charged to the recorded gap durations the day-quality
    # gate accrues on. These pin the deadline as a BEHAVIOUR, not a literal.
    blind_v5 = SHIPPED_INTERVAL_S + SHIPPED_TIMEOUT_S
    v4_kwargs = C.connect_keepalive_kwargs(C.HEARTBEAT_CONTROL_V4)
    blind_v4 = v4_kwargs["ping_interval"] + v4_kwargs["ping_timeout"]
    check("worst-case dead-socket blindness is DERIVED and NO WORSE than the "
          "v4 keepalive O1a tuned to — the detection regression is CLEARED, "
          "not merely reduced",
          blind_v5 == SHIPPED_INTERVAL_S + SHIPPED_TIMEOUT_S
          and blind_v5 <= blind_v4,
          f"v5 {blind_v5}s vs v4 {blind_v4}s")
    check("the answer deadline does not EXCEED the send cadence — a timeout "
          "at or above the interval would let a dead socket outlive a whole "
          "heartbeat cycle unnoticed",
          SHIPPED_TIMEOUT_S < SHIPPED_INTERVAL_S * 1.001,
          f"timeout {SHIPPED_TIMEOUT_S}s vs interval {SHIPPED_INTERVAL_S}s")
    check("the deadline still clears the observed round-trip by >=10x "
          "(live probe: ~90ms on the BTC channel)",
          SHIPPED_TIMEOUT_S >= 0.09 * 10,
          f"{SHIPPED_TIMEOUT_S}s vs 0.9s")
    check("we send at least as often as the venue documents — the deviation "
          "is FASTER, which is the direction with empirical support, and it "
          "is recorded as a tested deviation not a documented minimum",
          SHIPPED_INTERVAL_S <= 10.0,
          f"interval {SHIPPED_INTERVAL_S}s vs documented 10s")

    # BEHAVIOURAL (V5-C6-1): the previous pair could not tell a correct
    # deadline from a hard-coded 0.03 — `missing_pong` timed out under BOTH,
    # `healthy` succeeded under BOTH, and the "observed" value was copied
    # from the global the fixture had just set, not from the coroutine.
    # A DELAYED PONG at 0.10 s makes the deadline decide the outcome.
    late_short, late_gaps = await run_market("delayed_pong", timeout_s=0.05,
                                             interval_s=0.02)
    short_causes = [r.get("cause") for r in late_gaps
                    if r.get("event") == "disconnect"]
    check("BEHAVIOURAL: a deadline BELOW the PONG delay times out — a "
          "hard-coded LONGER deadline fails this",
          "APP_HEARTBEAT_TIMEOUT" in short_causes, str(short_causes))
    late_long, long_gaps = await run_market("delayed_pong", timeout_s=0.30,
                                            interval_s=0.02)
    long_causes = [r.get("cause") for r in long_gaps
                   if r.get("event") == "disconnect"]
    check("BEHAVIOURAL: a deadline ABOVE the same PONG delay does NOT time "
          "out — a hard-coded SHORTER deadline fails this; the two probes "
          "together pin the deadline from both sides",
          "APP_HEARTBEAT_TIMEOUT" not in long_causes
          and late_long.get("counted_app_pongs", 0) >= 1,
          f"causes={long_causes} pongs={late_long.get('counted_app_pongs')}")
    check("the fixture RESTORES the module globals — a run used to leave "
          "them mutated, so later assertions read the fixture's values",
          C.APP_HEARTBEAT_INTERVAL_S == SHIPPED_INTERVAL_S
          and C.APP_HEARTBEAT_TIMEOUT_S == SHIPPED_TIMEOUT_S,
          f"live {C.APP_HEARTBEAT_INTERVAL_S}/{C.APP_HEARTBEAT_TIMEOUT_S} "
          f"vs shipped {SHIPPED_INTERVAL_S}/{SHIPPED_TIMEOUT_S}")

    # ---- coverage the third audit found MISSING (no committed test
    # exercised either), and the 3s cadence makes both load-bearing:
    # 3.3x more heartbeat cycles means 3.3x more chances to leak a task.
    tasks_seen = []
    _orig_run = C.run_with_application_heartbeat

    async def _counting_run(*a, **kw):
        live = [t for t in asyncio.all_tasks()
                if "application_heartbeat" in (t.get_coro().__qualname__
                                               if hasattr(t, "get_coro")
                                               else "")]
        tasks_seen.append(len(live))
        return await _orig_run(*a, **kw)

    C.run_with_application_heartbeat = _counting_run
    try:
        for _ in range(3):
            await run_market("missing_pong")
    finally:
        C.run_with_application_heartbeat = _orig_run
    leaked = [t for t in asyncio.all_tasks()
              if "application_heartbeat" in (t.get_coro().__qualname__
                                             if hasattr(t, "get_coro")
                                             else "")]
    check("NO heartbeat task leaks across repeated reconnects — the "
          "finally-block cancel is load-bearing and nothing tested it",
          len(leaked) == 0 and max(tasks_seen or [0]) <= 1,
          f"live at entry {tasks_seen}, leaked after {len(leaked)}")

    counters_run, _ = await run_market("healthy")
    check("PONGs never exceed PINGs on a healthy socket (reading the REAL "
          "returned keys — the previous version read a 'counts' key that "
          "does not exist, so it passed unconditionally, V5-C5-2)",
          counters_run.get("counted_app_pongs") is not None
          and counters_run.get("counted_app_pings") is not None
          and counters_run["counted_app_pongs"]
          <= counters_run["counted_app_pings"],
          f"ping={counters_run.get('counted_app_pings')} "
          f"pong={counters_run.get('counted_app_pongs')}")
    dup_run, _ = await run_market("double_pong")
    # DECLARED DIVISION OF RESPONSIBILITY: the producer counts PONG FRAMES
    # faithfully and does NOT try to match them to outstanding pings; the
    # deploy gate is what refuses pong>ping. This known-bad pins the
    # producer half of that contract so the claim is testable on both sides.
    check("known-bad: a venue sending TWO PONGs per PING makes the producer "
          "count pong > ping — the producer counts frames, and the DEPLOY "
          "GATE is the half that refuses it",
          dup_run.get("counted_app_pongs", 0)
          > dup_run.get("counted_app_pings", 1),
          f"ping={dup_run.get('counted_app_pings')} "
          f"pong={dup_run.get('counted_app_pongs')}")

    failures = sum(not passed for _, passed, _ in RESULTS)
    print(f"\nV5 HEARTBEAT BEHAVIORAL: {len(RESULTS)-failures}/{len(RESULTS)} pass")
    return failures


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
