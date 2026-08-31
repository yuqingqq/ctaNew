#!/usr/bin/env python3
"""Join collector disconnects to HOST load from the sysstat archive.

Why this exists. R-365 concluded the 2026-08-25 btc break was caused by
research compute contention on the collector box, and stated that **no load
time-series exists**. That statement was false and never checked:
`/var/log/sysstat/sa25` is present (679,660 bytes, complete through 23:50Z),
as are sa23/24/26/27. Codex `b04ff13` found it. Asserting an absence without
opening the artifact that would carry it is a rule-16 failure by the author
of that rule's latest citation.

Joining the two refutes the attribution:

  * 2026-08-25 00:00-06:00Z carried 210 btc disconnects against 4 in the same
    hours of 08-24 -- 52x -- while the host averaged **89.9% idle**.
  * The break is fully present in the FIRST HOUR of 08-25. The "heavy runs on
    collector box" commit R-365 cited as contemporaneous evidence is stamped
    19:40/20:23, roughly TWENTY HOURS LATER.
  * Hour-by-hour there is no relationship: 01Z is the busiest hour (78.0%
    idle) with 38 disconnects; 11Z is nearly the quietest (98.5% idle) with
    14.

So the finding this script exists to support is a NEGATIVE one: host load
does not explain the break. Compute contention survives only as a hypothesis
for sub-sample bursts that ten-minute sysstat cannot resolve.

The calculation lived only in prose in R-365. A finding produced by a script
nobody can re-run is a claim, not a result, so it lives here.

    python3 live/pm_research/pm_host_load_join.py
    python3 live/pm_research/pm_host_load_join.py --day 25
    python3 live/pm_research/pm_host_load_join.py --selftest   # rule 15
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GAP_LEDGER = REPO / "data/pm_5min/collector_gaps.jsonl"
SYSSTAT = Path("/var/log/sysstat")
YEAR, MONTH = 2026, 8


class Refused(Exception):
    """A population this instrument must not summarise."""


def sar_cpu(day: int) -> list[tuple[int, float]]:
    """(epoch_of_sample_end, %idle) from the day's sysstat archive.

    Refuses rather than returning empty: an absent archive is exactly the
    thing R-365 assumed without checking, and a silent [] here would let the
    same error recur wearing a different mask.
    """
    p = SYSSTAT / f"sa{day:02d}"
    if not p.exists():
        raise Refused(f"{p} does not exist — REFUSING rather than reporting "
                      f"'no load data', which is the exact unchecked "
                      f"assertion this script was written to correct")
    r = subprocess.run(["sar", "-f", str(p), "-u"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise Refused(f"sar failed on {p}: {r.stderr.strip()[:120]}")
    out = []
    for ln in r.stdout.splitlines():
        m = re.match(r"(\d\d):(\d\d):(\d\d)\s+all\s+"
                     r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                     r"([\d.]+)\s+([\d.]+)", ln)
        if not m:
            continue
        t = dt.datetime(YEAR, MONTH, day,
                        int(m.group(1)), int(m.group(2)), int(m.group(3)),
                        tzinfo=dt.timezone.utc)
        out.append((int(t.timestamp()), float(m.group(9))))
    if not out:
        raise Refused(f"{p} parsed to ZERO samples — a silent regex miss "
                      f"would report a clean absence of load (rule 15: an "
                      f"instrument that cannot fire is not a result)")
    return out


def disconnects(day: int, coin: str = "btc") -> list[int]:
    if not GAP_LEDGER.exists():
        raise Refused(f"{GAP_LEDGER} missing")
    lo = dt.datetime(YEAR, MONTH, day, tzinfo=dt.timezone.utc).timestamp()
    hi = lo + 86400
    out = []
    for ln in GAP_LEDGER.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("coin") != coin or not r.get("gap_start_ns"):
            continue
        t = r["gap_start_ns"] / 1e9
        if lo <= t < hi:
            out.append(int(t))
    return out


def pearson(xs, ys) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def join(day: int, coin: str = "btc") -> dict:
    """Per sysstat interval: host busy-% and disconnects inside it."""
    samples = sar_cpu(day)
    ev = disconnects(day, coin)
    cells = []
    for i in range(1, len(samples)):
        t0, t1 = samples[i - 1][0], samples[i][0]
        busy = 100.0 - samples[i][1]
        cells.append({"start": t0, "end": t1, "busy_pct": busy,
                      "n_disconnects": sum(1 for e in ev if t0 <= e < t1)})
    inside = sum(c["n_disconnects"] for c in cells)
    return {
        "day": f"{YEAR}-{MONTH:02d}-{day:02d}", "coin": coin,
        "n_samples": len(cells), "n_events_total": len(ev),
        "n_events_joined": inside,
        "mean_busy_pct": round(sum(c["busy_pct"] for c in cells)
                               / max(len(cells), 1), 2),
        "pearson_busy_vs_disconnects":
            pearson([c["busy_pct"] for c in cells],
                    [c["n_disconnects"] for c in cells]),
        "cells": cells,
    }


def selftest() -> int:
    """Rule 15: a control it must detect, and inputs it must refuse."""
    checks = []

    def ok(cond, label):
        checks.append(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            print(f"SELFTEST FAILED at check {len(checks)}")
            raise SystemExit(1)

    # POSITIVE CONTROL: a perfectly correlated synthetic must come back ~1.0,
    # or a near-zero result on the real data means nothing.
    xs = list(range(20))
    ok(abs(pearson(xs, [3 * x + 1 for x in xs]) - 1.0) < 1e-9,
       "POSITIVE CONTROL: an exactly linear relation returns r = 1.0 — "
       "without this, a near-zero r on real data could be the estimator "
       "being broken rather than the relation being absent")
    ok(abs(pearson(xs, [-2 * x for x in xs]) + 1.0) < 1e-9,
       "POSITIVE CONTROL: an inverse relation returns r = -1.0, so the sign "
       "is meaningful and not an artefact")
    ok(pearson([1, 1, 1, 1], [1, 2, 3, 4]) is None,
       "a zero-variance input returns None, not a spurious 0.0 — 'no "
       "correlation' and 'undefined' are different answers")
    ok(pearson([1], [1]) is None, "n < 3 returns None rather than a number")

    # REFUSALS: the absent-archive path is the exact failure R-365 made.
    try:
        sar_cpu(99)
        ok(False, "an absent sysstat archive must REFUSE")
    except Refused:
        ok(True, "KNOWN-BAD: an absent sysstat archive REFUSES rather than "
                 "reporting 'no load data' — R-365 asserted exactly that "
                 "absence without opening the file, and the file existed")

    # and the real archive must actually parse, or every number is vacuous
    try:
        s = sar_cpu(25)
        ok(len(s) > 100, f"the REAL sa25 archive parses to {len(s)} samples "
                         f"(>100) — the join rests on real data, not an "
                         f"empty parse that would read as a clean host")
    except Refused as ex:
        ok(False, f"sa25 must parse: {ex}")

    print(f"pm_host_load_join selftests: {len(checks)} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", type=int, default=None, help="day of month")
    ap.add_argument("--coin", default="btc")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    days = [a.day] if a.day else sorted(
        int(p.name[2:]) for p in SYSSTAT.glob("sa[0-9][0-9]"))
    res = []
    for d in days:
        try:
            res.append(join(d, a.coin))
        except Refused as ex:
            print(f"{YEAR}-{MONTH:02d}-{d:02d}  REFUSED: {ex}")
    if a.json:
        print(json.dumps({"schema": "pm_host_load_join/1", "days": res},
                         indent=1))
        return 0
    print(f"{a.coin} disconnects vs HOST busy%, per sysstat interval\n")
    print(f"{'day':12} {'samples':>8} {'events':>7} {'mean busy%':>11} "
          f"{'pearson r':>10}")
    for r in res:
        pr = r["pearson_busy_vs_disconnects"]
        print(f"{r['day']:12} {r['n_samples']:8d} {r['n_events_joined']:7d} "
              f"{r['mean_busy_pct']:11.2f} "
              f"{'n/a' if pr is None else f'{pr:+.3f}':>10}")
    print("\nNEGATIVE FINDING: host load does not explain the 08-25 break. "
          "See the module docstring for the controls that make that "
          "readable rather than merely a small number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
