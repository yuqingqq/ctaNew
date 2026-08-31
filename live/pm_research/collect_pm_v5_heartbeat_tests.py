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


async def run_market(behavior: str, *, duration_s: float = 0.24) -> tuple[dict, list[dict]]:
    global _test_n
    _test_n += 1
    captured: dict = {}
    C.RAW = OUT / f"raw{_test_n}"
    C.ROOT = OUT
    C.GAP_LEDGER = OUT / f"gaps{_test_n}.jsonl"
    C.WINDOW_S = 1
    C.GRACE_S = 0
    C.SUBSCRIBE_CONFIRM_S = 0.05
    C.APP_HEARTBEAT_INTERVAL_S = 0.02
    C.APP_HEARTBEAT_TIMEOUT_S = 0.03
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

    failures = sum(not passed for _, passed, _ in RESULTS)
    print(f"\nV5 HEARTBEAT BEHAVIORAL: {len(RESULTS)-failures}/{len(RESULTS)} pass")
    return failures


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
