#!/usr/bin/env python3
"""Independent shadow subscription — the discriminator this programme lacks.

Codex `518c593`: three unexplained events are on record and NO instrument can
say which side of the wire produced any of them.

  2026-08-25 →         btc-only, disconnect-heavy, cause unknown
  2026-08-26 04:35Z    3h20m, all coins, healthy sockets, no gap rows
  2026-08-31 06:32Z    4h09m, all coins, receive counter at 0.51% of normal

For the last one the collector's own counter advanced 47,307 messages against
9,311,060 in a comparable stretch, and the tape reconciles to it at 99.77%.
So the collector wrote what it saw. **What nobody can say is whether the
venue sent nothing, or sent to everyone except us.** Row density cannot
distinguish those, and no amount of re-reading this tape ever will.

A SECOND, INDEPENDENT SUBSCRIPTION CAN, on the next occurrence:

  production and shadow collapse together  -> upstream venue/market regime
  production alone collapses               -> collector/process/path defect
  counters rise but the tape does not      -> writer/storage defect

That is why this must be RUNNING BEFORE the next event, not written after it.
Every retrospective theory this programme has produced about these events has
been wrong; a prospective discriminator is cheap and decides.

SAFETY, because the last "harmless" side process cost 302,941 lines of tape:
this NEVER writes the production tape or any ledger. Its only output is its
own JSONL under `data/pm_5min/derived/shadow/`, it subscribes read-only, and
it holds a small fixed number of sockets. It is INERT BY CONSTRUCTION with
respect to production — there is no code path here that opens a production
file for writing.

    python3 live/pm_research/pm_shadow_observer.py --selftest
    python3 live/pm_research/pm_shadow_observer.py --duration-s 0
    python3 live/pm_research/pm_shadow_observer.py --verify-output
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data/pm_5min/derived/shadow"
UNIT_FILE = Path(__file__).resolve().parent / "ops/pm-shadow-observer.service"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
SAMPLE_S = 60.0
WINDOW_S = 300
ROTATE_GRACE_S = 5.0
RECONNECT_S = 2.0
VERIFY_FRESH_S = 150.0
# Deliberately generous: this observer must never be the thing that fails.
PING_INTERVAL_S = 10
PING_TIMEOUT_S = 10


class Refused(Exception):
    """A configuration this observer must not run under."""


def guard_output(path: Path) -> None:
    """Refuse to write anywhere production data lives.

    The 2026-08-31 contamination happened because a side process reused a
    production write path. A comment saying 'this is read-only' is not a
    guard; this is.
    """
    p = path.resolve()
    forbidden = [(REPO / "data/pm_5min/raw").resolve(),
                 (REPO / "data/pm_5min/collector_gaps.jsonl").resolve(),
                 (REPO / "data/pm_5min/collector_runs.jsonl").resolve(),
                 (REPO / "data/pm_5min/markets.jsonl").resolve()]
    for f in forbidden:
        if p == f or f in p.parents:
            raise Refused(f"shadow output {p} is inside the production path "
                          f"{f} — REFUSING. A side process reusing a "
                          f"production write path cost 302,941 lines of tape "
                          f"on 2026-08-31")
    if OUT_DIR.resolve() not in p.parents and p.parent != OUT_DIR.resolve():
        raise Refused(f"shadow output {p} is outside {OUT_DIR} — this "
                      f"observer writes to exactly one directory")


def summarise(counts: dict, ages: dict, now: float,
              connection: dict | None = None) -> dict:
    """One sample row. Reports STATUSES, never a silent zero (rule 4)."""
    return {
        "event": "shadow_sample",
        "recv_ns": time.time_ns(),
        "msgs_by_coin": dict(counts),
        # None means "never received", which is NOT the same as "old"; a
        # coin that never delivered must not read as a fresh one.
        "last_msg_age_s": {c: (None if t is None else round(now - t, 2))
                           for c, t in ages.items()},
        "n_coins_silent": sum(1 for t in ages.values() if t is None),
        "connection": dict(connection or {}),
    }


def _token_ids(market: dict) -> list[str]:
    raw = market.get("clobTokenIds")
    try:
        values = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    if not isinstance(values, list):
        return []
    return [v for v in values if isinstance(v, str) and v]


def discover_current_slugs(gget, coins: tuple[str, ...], now_s: float,
                           window_s: int = WINDOW_S) -> dict:
    """Resolve current+next tokens without touching production state.

    The first observer called a nonexistent ``collect_pm`` helper and therefore
    always exited with an empty subscription. Discovery lives here and takes
    the collector's read-only Gamma function as an injected dependency so the
    real path is testable without network access.
    """
    current = int(now_s) - int(now_s) % window_s
    out: dict[str, list[str]] = {}
    for coin in coins:
        tokens = []
        for start in (current, current + window_s):
            market = gget(f"{coin}-updown-5m-{start}")
            if market:
                tokens.extend(_token_ids(market))
        if tokens:
            out[coin] = list(dict.fromkeys(tokens))
    return out


def require_complete_snapshot(slugs: dict,
                              coins: tuple[str, ...]) -> None:
    missing = [coin for coin in coins if not slugs.get(coin)]
    if missing:
        raise Refused(f"shadow discovery has no tokens for {missing} — a "
                      f"partial observer cannot attribute an all-coin event")


def verify_output_rows(rows: list, coins: tuple[str, ...], now_ns: int,
                       fresh_s: float = VERIFY_FRESH_S) -> dict:
    """Prove the independent path is connected and receiving every coin."""
    samples = [r for r in rows if r.get("event") == "shadow_sample"]
    if not samples:
        raise Refused("no shadow_sample row exists")
    row = samples[-1]
    sample_index = max(i for i, value in enumerate(rows) if value is row)
    terminal_after = [r.get("event") for r in rows[sample_index + 1:]
                      if r.get("event") in ("shadow_error", "shadow_end")]
    if terminal_after:
        raise Refused(f"shadow event(s) after the latest sample: "
                      f"{terminal_after}; no later healthy sample exists")
    recv_ns = row.get("recv_ns")
    if type(recv_ns) is not int or recv_ns <= 0:
        raise Refused("latest shadow sample has no valid recv_ns")
    age_s = (now_ns - recv_ns) / 1e9
    if age_s < -5 or age_s > fresh_s:
        raise Refused(f"latest shadow sample is {age_s:.1f}s old; maximum is "
                      f"{fresh_s:.0f}s")
    connection = row.get("connection") or {}
    if connection.get("connected") is not True:
        raise Refused("latest shadow sample is not from a connected socket")
    counts = row.get("msgs_by_coin") or {}
    ages = row.get("last_msg_age_s") or {}
    missing = [c for c in coins if c not in counts or c not in ages]
    silent = [c for c in coins if counts.get(c, 0) <= 0 or ages.get(c) is None]
    stale = [c for c in coins if isinstance(ages.get(c), (int, float))
             and ages[c] > fresh_s]
    if missing:
        raise Refused(f"latest shadow sample omits coins {missing}")
    if silent:
        raise Refused(f"shadow has never received coins {silent}")
    if stale:
        raise Refused(f"shadow coins are stale beyond {fresh_s:.0f}s: {stale}")
    return row


async def observe(discover, coins: tuple[str, ...], duration_s: float,
                  out_path: Path) -> int:
    import websockets                                    # noqa: PLC0415
    guard_output(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if duration_s < 0:
        raise Refused("duration must be >= 0 (zero means run continuously)")
    counts = {c: 0 for c in coins}
    ages: dict[str, float | None] = {c: None for c in coins}
    stop_at = None if duration_s == 0 else time.time() + duration_s
    connection = {"connected": False, "generation": 0, "n_tokens": 0}

    def running() -> bool:
        return stop_at is None or time.time() < stop_at

    async def sampler(fh):
        while running():
            wait = SAMPLE_S if stop_at is None else min(
                SAMPLE_S, max(0.0, stop_at - time.time()))
            if wait <= 0:
                break
            await asyncio.sleep(wait)
            if not running() and stop_at is not None:
                break
            fh.write(json.dumps(summarise(counts, ages, time.time(),
                                          connection)) + "\n")
            fh.flush()

    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "shadow_start",
                             "recv_ns": time.time_ns(),
                             "coins": sorted(coins),
                             "duration_s": duration_s}) + "\n")
        fh.flush()
        task = asyncio.create_task(sampler(fh))
        try:
            while running():
                try:
                    slugs = await asyncio.to_thread(discover)
                    require_complete_snapshot(slugs, coins)
                    tokens = [t for toks in slugs.values() for t in toks]
                    by_token = {t: c for c, toks in slugs.items() for t in toks}
                    if not tokens:
                        raise Refused("discovery returned no tokens")
                    connection["generation"] += 1
                    connection["n_tokens"] = len(tokens)
                    # Current+next are subscribed together. Rotate once just
                    # after each 5-minute boundary so the newly-created next
                    # market enters without churning a healthy socket every 30s.
                    now = time.time()
                    rotate_at = ((int(now) // WINDOW_S) + 1) * WINDOW_S \
                        + ROTATE_GRACE_S
                    fh.write(json.dumps({"event": "shadow_subscription",
                                         "recv_ns": time.time_ns(),
                                         "generation": connection["generation"],
                                         "n_tokens": len(tokens),
                                         "coins": sorted(slugs)}) + "\n")
                    fh.flush()
                    async with websockets.connect(
                            WS_URL, open_timeout=15,
                            ping_interval=PING_INTERVAL_S,
                            ping_timeout=PING_TIMEOUT_S) as ws:
                        await ws.send(json.dumps({"assets_ids": tokens,
                                                  "type": "market"}))
                        connection["connected"] = True
                        while running() and time.time() < rotate_at:
                            deadline = rotate_at
                            if stop_at is not None:
                                deadline = min(deadline, stop_at)
                            try:
                                raw = await asyncio.wait_for(
                                    ws.recv(), timeout=max(
                                        0.1, deadline - time.time()))
                            except asyncio.TimeoutError:
                                break
                            now = time.time()
                            try:
                                msgs = json.loads(raw)
                            except (TypeError, ValueError):
                                continue
                            for message in (msgs if isinstance(msgs, list)
                                            else [msgs]):
                                if not isinstance(message, dict):
                                    continue
                                coin = by_token.get(message.get("asset_id"))
                                if coin:
                                    counts[coin] += 1
                                    ages[coin] = now
                except asyncio.CancelledError:
                    raise
                except Exception as ex:                    # noqa: BLE001
                    fh.write(json.dumps({"event": "shadow_error",
                                         "recv_ns": time.time_ns(),
                                         "error": (f"{type(ex).__name__}: "
                                                   f"{ex}")[:200]}) + "\n")
                    fh.flush()
                    if running():
                        await asyncio.sleep(RECONNECT_S)
                finally:
                    connection["connected"] = False
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            fh.write(json.dumps({**summarise(counts, ages, time.time(),
                                             connection),
                                 "event": "shadow_end"}) + "\n")
            fh.flush()
    if duration_s >= SAMPLE_S:
        silent = [coin for coin in coins if counts.get(coin, 0) <= 0]
        if silent:
            raise Refused(f"finite shadow run ended without messages for "
                          f"{silent}")
        if connection.get("generation", 0) <= 0:
            raise Refused("finite shadow run never opened a subscription")
    return 0


def selftest() -> int:
    checks = []

    def ok(cond, label):
        checks.append(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            print(f"SELFTEST FAILED at check {len(checks)}")
            raise SystemExit(1)

    for bad in (REPO / "data/pm_5min/raw/20260831/x.jsonl.gz",
                REPO / "data/pm_5min/collector_gaps.jsonl",
                REPO / "data/pm_5min/collector_runs.jsonl"):
        try:
            guard_output(bad)
            ok(False, f"writing to {bad} must REFUSE")
        except Refused:
            ok(True, f"KNOWN-BAD: output inside the production path "
                     f"({bad.name}) REFUSES — inert by construction, not by "
                     f"comment")
    guard_output(OUT_DIR / "shadow_20260901.jsonl")
    ok(True, "POSITIVE: the designated shadow path is accepted, so the guard "
             "refuses the right things rather than everything")

    s = summarise({"btc": 5, "eth": 0}, {"btc": 100.0, "eth": None}, 110.0)
    ok(s["last_msg_age_s"]["btc"] == 10.0
       and s["last_msg_age_s"]["eth"] is None
       and s["n_coins_silent"] == 1,
       "a coin that NEVER delivered reports age None and counts as SILENT — "
       "not as age 0, which would read as the freshest coin in the sample")
    ok(summarise({}, {}, 0.0)["n_coins_silent"] == 0
       and summarise({}, {}, 0.0)["msgs_by_coin"] == {},
       "an empty sample is empty rather than fabricated")

    calls = []
    def fake_gget(slug):
        calls.append(slug)
        return {"clobTokenIds": json.dumps([slug + "-up", slug + "-down"])}
    found = discover_current_slugs(fake_gget, ("btc", "eth"), 601,
                                   window_s=300)
    ok(set(found) == {"btc", "eth"} and all(len(v) == 4
                                              for v in found.values())
       and calls == ["btc-updown-5m-600", "btc-updown-5m-900",
                     "eth-updown-5m-600", "eth-updown-5m-900"],
       "POSITIVE: the REAL startup discovery path resolves current+next "
       "tokens for every requested coin")
    try:
        require_complete_snapshot({"btc": ["x"]}, ("btc", "eth"))
        ok(False, "partial discovery must refuse")
    except Refused:
        ok(True, "KNOWN-BAD: partial discovery refuses — a one-coin shadow "
                 "cannot attribute an all-coin event")

    good_row = {"event": "shadow_sample", "recv_ns": 1_000_000_000_000,
                "msgs_by_coin": {"btc": 10, "eth": 4},
                "last_msg_age_s": {"btc": 1.0, "eth": 2.0},
                "connection": {"connected": True, "generation": 1}}
    ok(verify_output_rows([good_row], ("btc", "eth"),
                          1_001_000_000_000) is good_row,
       "POSITIVE: fresh connected output with every coin passes")
    try:
        verify_output_rows([{**good_row,
                             "connection": {"connected": False}}],
                           ("btc", "eth"), 1_001_000_000_000)
        ok(False, "disconnected output must refuse")
    except Refused:
        ok(True, "KNOWN-BAD: a fresh-looking sample from a disconnected "
                 "shadow refuses")
    try:
        verify_output_rows([{**good_row,
                             "msgs_by_coin": {"btc": 10},
                             "last_msg_age_s": {"btc": 1.0}}],
                           ("btc", "eth"), 1_001_000_000_000)
        ok(False, "missing coin output must refuse")
    except Refused:
        ok(True, "KNOWN-BAD: output omitting one coin refuses")
    try:
        verify_output_rows([good_row, {"event": "shadow_end"}],
                           ("btc", "eth"), 1_001_000_000_000)
        ok(False, "a stopped observer must refuse")
    except Refused:
        ok(True, "KNOWN-BAD: a healthy sample followed by shadow_end refuses "
                 "instead of certifying a stopped observer")
    unit = UNIT_FILE.read_text()
    ok("pm_shadow_observer.py --duration-s 0" in unit
       and "Restart=always" in unit and "Slice=collectors.slice" in unit,
       "the tracked service runs continuously, restarts on failure, and stays "
       "inside the collector resource slice")
    print(f"pm_shadow_observer selftests: {len(checks)} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration-s", type=float, default=0.0,
                    help="0 runs continuously; positive values stop after N s")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-output", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import collect_pm as C                                # noqa: PLC0415
    coins = tuple(C.COINS)
    if a.verify_output:
        active = subprocess.run(
            ["systemctl", "--user", "is-active", "pm-shadow-observer.service"],
            capture_output=True, text=True, check=False)
        if active.returncode != 0 or active.stdout.strip() != "active":
            raise Refused("pm-shadow-observer.service is not active")
        paths = sorted(OUT_DIR.glob("shadow_*.jsonl"),
                       key=lambda p: p.stat().st_mtime)
        if not paths:
            raise Refused(f"no shadow output exists under {OUT_DIR}")
        rows = []
        with paths[-1].open() as fh:
            for lineno, line in enumerate(fh, 1):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    raise Refused(f"{paths[-1]} line {lineno} is not JSON")
        row = verify_output_rows(rows, coins, time.time_ns())
        print(f"OK shadow: {paths[-1].name}, generation="
              f"{row['connection'].get('generation')}, every coin receiving")
        return 0
    discover = lambda: discover_current_slugs(C.gget, coins, time.time(),
                                               C.WINDOW_S)
    out = OUT_DIR / ("shadow_" +
                     time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) +
                     f"_{os.getpid()}.jsonl")
    return asyncio.run(observe(discover, coins, a.duration_s, out))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as ex:
        print(f"REFUSED: {ex}", file=sys.stderr)
        sys.exit(2)
