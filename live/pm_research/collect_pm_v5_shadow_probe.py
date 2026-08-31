"""Run the clob_v5 heartbeat path against one live market in scratch space.

This never writes the production tape or restarts the service.  It is a narrow
transport seam check: the real candidate connection must exchange documented
application PING/PONG messages while market rows continue to arrive.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import collect_pm as C


DEFAULT_LEDGER = Path("data/pm_5min/collector_gaps.jsonl")


def latest_market_identity(path: Path, coin: str) -> tuple[str, list[str], int]:
    found = None
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if (row.get("coin") == coin and type(row.get("recv_ns")) is int
                    and isinstance(row.get("tokens"), list)
                    and row.get("slug")):
                if found is None or row["recv_ns"] > found[2]:
                    found = (row["slug"], row["tokens"], row["recv_ns"])
    if found is None:
        raise RuntimeError(f"no stamped {coin} market identity in {path}")
    return found


def selftest() -> int:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ledger.jsonl"
        path.write_text(
            json.dumps({"recv_ns": 2, "coin": "btc", "slug": "new",
                        "tokens": ["a", "b"]}) + "\n"
            + json.dumps({"recv_ns": 1, "coin": "btc", "slug": "old",
                          "tokens": ["c", "d"]}) + "\n"
        )
        positive = latest_market_identity(path, "btc") == ("new", ["a", "b"], 2)
        try:
            latest_market_identity(path, "eth")
            known_bad = False
        except RuntimeError:
            known_bad = True
    checks = [
        ("positive: newest exact-coin identity selected", positive),
        ("known-bad: missing coin identity refused", known_bad),
    ]
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    return 0 if all(passed for _, passed in checks) else 1


async def probe(ledger: Path, coin: str, duration_s: float) -> dict:
    slug, tokens, identity_recv_ns = latest_market_identity(ledger, coin)
    with tempfile.TemporaryDirectory(prefix="pm_hb_shadow_") as directory:
        root = Path(directory)
        C.ROOT = root
        C.RAW = root / "raw"
        C.GAP_LEDGER = root / "gaps.jsonl"
        C.GRACE_S = 0
        now = time.time()
        ts = int(now)
        C.WINDOW_S = (now - ts) + duration_s
        collector = C.PMCollector(heartbeat_mode=C.HEARTBEAT_APP_V5)
        started_ns = time.time_ns()
        try:
            await collector._market(slug, ts, tokens)
        finally:
            ended_ns = time.time_ns()
            collector.disk_pool.shutdown(wait=True)
            collector.http_pool.shutdown(wait=False)
        gaps = []
        if C.GAP_LEDGER.exists():
            gaps = [json.loads(line) for line in C.GAP_LEDGER.read_text().splitlines()]
        return {
            "instrument": "collect_pm_v5_shadow_probe_v1",
            "collector_version": collector.collector_version,
            "heartbeat_mode": collector.heartbeat_mode,
            "source_identity": {
                "coin": coin,
                "slug": slug,
                "ledger_recv_ns": identity_recv_ns,
            },
            "population": {
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "duration_s": round((ended_ns - started_ns) / 1e9, 3),
                "market_messages_n": int(collector.counts.get("msgs", 0)),
                "app_pings_n": int(collector.counts.get("app_heartbeat_pings", 0)),
                "app_pongs_n": int(collector.counts.get("app_heartbeat_pongs", 0)),
                "disconnects_n": sum(row.get("event") == "disconnect" for row in gaps),
                "disconnect_causes": [
                    row.get("cause") for row in gaps
                    if row.get("event") == "disconnect"
                ],
            },
            "scope": "scratch-only transport probe; not a day-bar or deployment verdict",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--coin", default="btc")
    parser.add_argument("--duration-s", type=float, default=35.0)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if args.duration_s < 21.0:
        raise SystemExit("duration must be >=21s to exercise at least two 10s heartbeats")
    print(json.dumps(asyncio.run(probe(args.ledger, args.coin, args.duration_s)),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
