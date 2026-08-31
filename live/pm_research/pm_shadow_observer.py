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
    python3 live/pm_research/pm_shadow_observer.py --duration-s 3600
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data/pm_5min/derived/shadow"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
SAMPLE_S = 60.0
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


def summarise(counts: dict, ages: dict, now: float) -> dict:
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
    }


async def observe(slugs: dict, duration_s: float, out_path: Path) -> int:
    import websockets                                    # noqa: PLC0415
    guard_output(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {c: 0 for c in slugs}
    ages: dict[str, float | None] = {c: None for c in slugs}
    stop_at = time.time() + duration_s
    tokens = [t for toks in slugs.values() for t in toks]
    if not tokens:
        raise Refused("no tokens to subscribe — an observer with nothing to "
                      "observe would report a silent zero forever")
    by_token = {t: c for c, toks in slugs.items() for t in toks}

    async def sampler(fh):
        while time.time() < stop_at:
            await asyncio.sleep(SAMPLE_S)
            fh.write(json.dumps(summarise(counts, ages, time.time())) + "\n")
            fh.flush()

    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "shadow_start",
                             "recv_ns": time.time_ns(),
                             "coins": sorted(slugs),
                             "duration_s": duration_s}) + "\n")
        fh.flush()
        task = asyncio.create_task(sampler(fh))
        try:
            async with websockets.connect(
                    WS_URL, open_timeout=15,
                    ping_interval=PING_INTERVAL_S,
                    ping_timeout=PING_TIMEOUT_S) as ws:
                await ws.send(json.dumps({"assets_ids": tokens,
                                          "type": "market"}))
                while time.time() < stop_at:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=max(0.1, stop_at - time.time()))
                    except asyncio.TimeoutError:
                        break
                    now = time.time()
                    try:
                        msgs = json.loads(raw)
                    except ValueError:
                        continue
                    for m in (msgs if isinstance(msgs, list) else [msgs]):
                        c = by_token.get(m.get("asset_id"))
                        if c:
                            counts[c] += 1
                            ages[c] = now
        except Exception as ex:                            # noqa: BLE001
            fh.write(json.dumps({"event": "shadow_error",
                                 "recv_ns": time.time_ns(),
                                 "error": f"{type(ex).__name__}: {ex}"[:200]})
                     + "\n")
        finally:
            task.cancel()
            fh.write(json.dumps({**summarise(counts, ages, time.time()),
                                 "event": "shadow_end"}) + "\n")
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
    print(f"pm_shadow_observer selftests: {len(checks)} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration-s", type=float, default=3600.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import collect_pm as C                                # noqa: PLC0415
    slugs = C.discover_current_slugs() if hasattr(
        C, "discover_current_slugs") else {}
    if not slugs:
        print("no live slugs resolved — refusing to run an observer that "
              "would record a silent zero (rule 4)")
        return 1
    out = OUT_DIR / f"shadow_{time.strftime('%Y%m%d', time.gmtime())}.jsonl"
    return asyncio.run(observe(slugs, a.duration_s, out))


if __name__ == "__main__":
    sys.exit(main())
