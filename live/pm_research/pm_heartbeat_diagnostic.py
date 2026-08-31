"""Reproduce the clob_v4 heartbeat-failure signature from the gap ledger.

This is a measurement instrument, not an admission or deployment gate.  It
reports the exact disconnect population, the age of the last market message at
each keepalive timeout, and the collector overload controls that were already
stamped on each row.  Interpretation belongs in the accompanying review.

The important clock is ``recv_ns``.  Never select this population by file or
line position: the ledger spans collector eras and grows while it is read.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_LEDGER = Path("data/pm_5min/collector_gaps.jsonl")


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _utc(recv_ns: int | None) -> str | None:
    if recv_ns is None:
        return None
    return datetime.fromtimestamp(recv_ns / 1e9, timezone.utc).isoformat()


def analyze(rows: list[dict], *, collector_version: str, coin: str,
            since_recv_ns: int) -> dict:
    population = [
        row for row in rows
        if row.get("event") == "disconnect"
        and row.get("collector_version") == collector_version
        and row.get("coin") == coin
        and type(row.get("recv_ns")) is int
        and row["recv_ns"] >= since_recv_ns
    ]
    causes = Counter(row.get("cause", "MISSING") for row in population)
    ping_rows = [row for row in population if row.get("cause") == "PING_TIMEOUT"]
    message_ages_ms = [
        (row["recv_ns"] - row["last_message_recv_ns"]) / 1e6
        for row in ping_rows
        if type(row.get("last_message_recv_ns")) is int
        and row["recv_ns"] >= row["last_message_recv_ns"]
    ]
    loop_lag_ms = [
        float(row["lag_ms_max_interval"])
        for row in ping_rows
        if type(row.get("lag_ms_max_interval")) in (int, float)
    ]
    queue_depth = [
        row["ws_queue_depth_max"]
        for row in ping_rows
        if type(row.get("ws_queue_depth_max")) is int
    ]
    as_of = max((row["recv_ns"] for row in population), default=None)
    n = len(population)
    n_ping = len(ping_rows)
    local_1011 = sum(
        str(row.get("error", "")).startswith(
            "sent 1011 (internal error) keepalive ping timeout"
        )
        for row in ping_rows
    )
    return {
        "instrument": "pm_heartbeat_diagnostic_v1",
        "population": {
            "collector_version": collector_version,
            "coin": coin,
            "since_recv_ns_inclusive": since_recv_ns,
            "disconnects_n": n,
            "ping_timeouts_n": n_ping,
            "causes": dict(sorted(causes.items())),
            "as_of_recv_ns": as_of,
            "as_of_utc": _utc(as_of),
        },
        "client_timeout_signature": {
            "ping_timeout_share": round(n_ping / n, 6) if n else None,
            "local_1011_keepalive_close_n": local_1011,
            "local_1011_keepalive_close_share": (
                round(local_1011 / n_ping, 6) if n_ping else None
            ),
            "last_market_message_age_ms": {
                "evaluable_n": len(message_ages_ms),
                "median": _rounded(_quantile(message_ages_ms, 0.5)),
                "p90": _rounded(_quantile(message_ages_ms, 0.9)),
                "under_3000ms_n": sum(value < 3000 for value in message_ages_ms),
                "under_3000ms_share": (
                    round(sum(value < 3000 for value in message_ages_ms)
                          / len(message_ages_ms), 6)
                    if message_ages_ms else None
                ),
            },
            "loop_lag_ms": {
                "evaluable_n": len(loop_lag_ms),
                "median": _rounded(_quantile(loop_lag_ms, 0.5)),
                "p90": _rounded(_quantile(loop_lag_ms, 0.9)),
                "max": _rounded(max(loop_lag_ms) if loop_lag_ms else None),
            },
            "websocket_queue": {
                "evaluable_n": len(queue_depth),
                "median_depth": _rounded(_quantile(queue_depth, 0.5)),
                "p90_depth": _rounded(_quantile(queue_depth, 0.9)),
                "max_depth": max(queue_depth) if queue_depth else None,
                "ever_paused_n": sum(
                    row.get("ws_ever_paused") is True for row in ping_rows
                ),
            },
        },
        "interpretation_bound": (
            "Measurements only. Match these rows to the heartbeat implementation "
            "and the venue's documented protocol before assigning cause."
        ),
    }


def _selftest() -> int:
    base = 2_000_000_000_000_000_000
    positive = []
    for i in range(9):
        positive.append({
            "recv_ns": base + i * 10_000_000_000,
            "collector_version": "candidate",
            "event": "disconnect",
            "coin": "btc",
            "cause": "PING_TIMEOUT",
            "last_message_recv_ns": base + i * 10_000_000_000 - 2_000_000_000,
            "lag_ms_max_interval": 2.0,
            "ws_queue_depth_max": 4,
            "ws_ever_paused": False,
            "error": ("sent 1011 (internal error) keepalive ping timeout; "
                      "no close frame received"),
        })
    positive.append({
        "recv_ns": base + 100_000_000_000,
        "collector_version": "candidate",
        "event": "disconnect",
        "coin": "btc",
        "cause": "NO_CLOSE_FRAME",
        "last_message_recv_ns": base + 99_000_000_000,
        "lag_ms_max_interval": 2.0,
        "ws_queue_depth_max": 4,
        "ws_ever_paused": False,
    })
    measured = analyze(positive, collector_version="candidate", coin="btc",
                       since_recv_ns=base)
    sig = measured["client_timeout_signature"]
    checks = [
        ("positive: action population is exact",
         measured["population"]["disconnects_n"] == 10),
        ("positive: ping cause and recent-data evidence are counted",
         measured["population"]["ping_timeouts_n"] == 9
         and sig["last_market_message_age_ms"]["under_3000ms_n"] == 9
         and sig["local_1011_keepalive_close_n"] == 9),
        ("positive: overload controls stay attached to the same rows",
         sig["loop_lag_ms"]["max"] == 2.0
         and sig["websocket_queue"]["ever_paused_n"] == 0),
    ]

    known_bad = [dict(row, collector_version="other") for row in positive]
    refused = analyze(known_bad, collector_version="candidate", coin="btc",
                      since_recv_ns=base)
    checks.append(("known-bad: a foreign collector era is refused",
                   refused["population"]["disconnects_n"] == 0
                   and refused["client_timeout_signature"]
                   ["ping_timeout_share"] is None))
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    return 0 if all(passed for _, passed in checks) else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--collector-version", default="clob_v4")
    parser.add_argument("--coin", default="btc")
    parser.add_argument("--since-recv-ns", type=int, default=0)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())

    rows = []
    unparseable = 0
    with args.ledger.open() as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                unparseable += 1
    result = analyze(rows, collector_version=args.collector_version,
                     coin=args.coin, since_recv_ns=args.since_recv_ns)
    result["ledger"] = {
        "path": str(args.ledger),
        "lines_read": len(rows) + unparseable,
        "unparseable_lines": unparseable,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
