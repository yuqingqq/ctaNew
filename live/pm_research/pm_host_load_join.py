#!/usr/bin/env python3
"""Join collector disconnects to HOST load from the sysstat archive.

Why this exists. R-365 concluded the 2026-08-25 btc break was caused by
research compute contention on the collector box, and stated that **no load
time-series exists**. That statement was false and never checked:
`/var/log/sysstat/sa25` is present (679,660 bytes, complete through 23:50Z),
as are sa23/24/26/27. Codex `b04ff13` found it. Asserting an absence without
opening the artifact that would carry it is a rule-16 failure by the author
of that rule's latest citation.

POPULATION (corrected under Codex HJ-R1; the first version mislabelled it).
143 sysstat cells per full day, the first ANCHORED AT DAY START rather than
discarded; the decision population is `event=disconnect` rows at their own
`recv_ns`, not `gap_closed` rows binned at `gap_start_ns` (a nearby proxy,
which rule 3 forbids); and events past the last sysstat endpoint are reported
as UNEVALUABLE with an as-of, never dropped. On 08-25: 143 cells, 665 events,
663 joined, 2 unevaluable, as-of 23:50:15Z, r = +0.033. The pre-repair run
said "143 samples" while correlating over 142 and silently losing 11 events.

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
import hashlib
import io
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GAP_LEDGER = REPO / "data/pm_5min/collector_gaps.jsonl"
SRC = Path(__file__).resolve().read_text()
SYSSTAT = Path("/var/log/sysstat")
YEAR, MONTH = 2026, 8


class Refused(Exception):
    """A population this instrument must not summarise."""


#: The parser's control, content-addressed. Produced by `sadc 1 8` on this
#: host: a REAL binary sysstat archive that `sar -f` reads, small enough to
#: commit. It exists because the old control was BOTH a parser test and a
#: liveness test, and sysstat recycles archive names by day-of-month
#: (HISTORY=7), so the parser half was a function of the calendar and went red
#: on 2026-09-03 when `sa25` was deleted. This half can ALWAYS fire (rule 15).
#:
#: WHY BINARY AND NOT THE TEXT REPORT: `sar -f` REFUSES the text form
#: ("Invalid system activity file"); the text also repeats every timestamp
#: twice and carries a different column set, so reusing this module's regex on
#: it would bind %irq where it means %idle. Measured in Q-DA-215.
#:
#: WHY NOT A SLICE OF A REAL DAY: `sar -f <src> -o <dst>` refuses --
#: "-f and -o options are mutually exclusive" -- so a real archive cannot be
#: trimmed. `sadc` writing a fresh short archive is the only route the tools
#: offer.
PARSER_FIXTURE = (Path(__file__).resolve().parent / "fixtures"
                  / "sysstat_parser_control.sa")
PARSER_FIXTURE_SHA256 = ("d663ce6b5d333ee6decca6c9b45e4911"
                         "b3badf672edc261807a3791c707359a4")
PARSER_FIXTURE_BYTES = 31428
PARSER_FIXTURE_ROWS = 7
PARSER_FIXTURE_NOTE = ("sysstat 12.6.1 binary; the format is versioned, so a "
                       "sysstat major upgrade may refuse it -- which this "
                       "control must then report LOUDLY rather than skip")

#: The collector's cadence, DERIVED from the unit rather than written down.
#: House pattern (`de_*`): the check count is ASSERTED at run time, so a
#: check that stops running is a failure rather than a smaller number nobody
#: reads. It moved 6 -> 39 when the calendar-bound control was split.
EXPECTED_CHECKS = 39

SYSSTAT_TIMER = "sysstat-collect.timer"
SECONDS_PER_DAY = 86400


def _interval_from_unit_text(text: str, source: str) -> int:
    """Seconds between samples, PARSED from a systemd unit's OnCalendar.

    Pure so a control can drive it: the selftest feeds it a unit body whose
    cadence it already knows, and a body carrying no cadence at all.
    """
    m = re.search(r"^OnCalendar=\*:0*/?(\d+)\s*$", text, re.M)
    if not m:
        raise Refused(f"{source} carries no `OnCalendar=*:00/N` line this "
                      f"parser understands; REFUSING rather than assuming a "
                      f"cadence -- a guessed interval is a literal wearing a "
                      f"function's clothes")
    return int(m.group(1)) * 60


def collector_interval_s() -> int:
    """Seconds between sysstat samples, read from the timer on this host.

    NO CADENCE LITERAL ENTERS THIS MODULE. `OnCalendar=*:00/10` means every
    ten minutes; if the host's cadence changes, every expectation below moves
    with it instead of going quietly wrong.
    """
    r = subprocess.run(["systemctl", "cat", SYSSTAT_TIMER],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise Refused(f"cannot read {SYSSTAT_TIMER}: "
                      f"{r.stderr.strip()[:120]} -- the sample interval is "
                      f"DERIVED from the unit, and guessing it would put a "
                      f"literal back where the calendar bug lived")
    return _interval_from_unit_text(r.stdout, SYSSTAT_TIMER)


def expected_rows(interval_s: int) -> int:
    """`sar -u` prints one row per COMPLETED interval, so a full UTC day is
    N-1 of them: the sample at 00:00 belongs to the previous day's archive."""
    return SECONDS_PER_DAY // interval_s - 1


def expected_last_second_of_day(interval_s: int) -> int:
    return SECONDS_PER_DAY - interval_s


def cadence_literals_in_code(src: str, interval_s: int) -> list[str]:
    """Every place the production region WRITES DOWN a derived expectation.

    The old host-load check hardcoded its own answer (`len(s) > 100` against
    a pinned `sa25`), so when the host changed the check went red for a
    reason that had nothing to do with what it measured. The fix is only real
    if the numbers stay derived, and that is a property of the SOURCE -- so
    it is checked, not promised.

    Prose is exempt and executable code is not: comments and docstrings
    RECORD what was measured on a past day (this module's header reports 143
    cells on 08-25, and that number is a finding, not an expectation). A
    number in an expression is the thing that rots. The exemption is
    deliberate and its boundary is controlled in the selftest, both ways.
    """
    want_n = expected_rows(interval_s)
    want_sod = expected_last_second_of_day(interval_s)
    bad_nums = {str(n) for n in range(want_n - 3, want_n + 2)}
    bad_nums |= {str(want_sod), str(want_sod // 60)}
    bad_strs = {f"{want_sod // 3600:02d}:{want_sod % 3600 // 60:02d}",
                f"{want_sod // 3600:02d}:00"}
    hits, prev_significant = [], tokenize.NEWLINE
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NUMBER and tok.string in bad_nums:
            hits.append(f"L{tok.start[0]}: numeric literal {tok.string}")
        elif tok.type == tokenize.STRING:
            # a STRING alone on a logical line is a docstring: prose.
            if prev_significant not in (tokenize.NEWLINE, tokenize.NL,
                                        tokenize.INDENT, tokenize.DEDENT):
                for b in bad_strs:
                    if b in tok.string:
                        hits.append(f"L{tok.start[0]}: time literal {b!r} "
                                    f"in an operand string")
        if tok.type not in (tokenize.COMMENT,):
            prev_significant = tok.type
    return hits


def _sar_rows(path: Path) -> list[tuple[int, float]]:
    """(second_of_day, %idle) from ANY binary archive `sar -f` can read.

    THE ONE PARSER. Split out from `sar_cpu` so the PARSER control can point
    at a committed fixture while the LIVENESS check points at whatever the
    host currently holds -- the two were one check, and because sysstat
    recycles archive names by day-of-month the parser half was a function of
    the calendar. It returns seconds-of-day, not epochs, so nothing here has
    to invent a date for a file whose date it does not know.

    The columns are `sar -u`'s six: %user %nice %system %iowait %steal %idle.
    Group 9 is %idle. Reading the TEXT report here instead would bind group 9
    to a different column -- see PARSER_FIXTURE_NOTE.
    """
    if not Path(path).exists():
        raise Refused(f"{path} does not exist — REFUSING rather than "
                      f"reporting 'no load data', which is the exact "
                      f"unchecked assertion this script was written to "
                      f"correct")
    r = subprocess.run(["sar", "-f", str(path), "-u"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise Refused(f"sar failed on {path}: "
                      f"{(r.stderr or r.stdout).strip()[:120]}")
    out = []
    for ln in r.stdout.splitlines():
        m = re.match(r"(\d\d):(\d\d):(\d\d)\s+all\s+"
                     r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                     r"([\d.]+)\s+([\d.]+)", ln)
        if not m:
            continue
        sod = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        out.append((sod, float(m.group(9))))
    if not out:
        raise Refused(f"{path} parsed to ZERO samples — a silent regex miss "
                      f"would report a clean absence of load (rule 15: an "
                      f"instrument that cannot fire is not a result)")
    return out


def archives_present() -> list[tuple[int, Path]]:
    """(day_of_month, path) for every `saDD` sysstat holds, newest first."""
    if not SYSSTAT.is_dir():
        raise Refused(f"{SYSSTAT} is not a directory — refusing rather than "
                      f"reporting an empty host")
    out = []
    for q in SYSSTAT.iterdir():
        m = re.fullmatch(r"sa(\d\d)", q.name)
        if m:
            out.append((int(m.group(1)), q))
    return sorted(out, key=lambda dq: dq[1].stat().st_mtime, reverse=True)


def completeness_of_rows(rows: list[tuple[int, float]],
                         interval_s: int) -> dict:
    """Does this row set cover a whole UTC day? A STATUS, never a drop.

    Pure, so the selftest can drive it with a day it built: a synthetic full
    day must read COMPLETE, one row short must read TRUNCATED on the COUNT
    leg, and a full count that stops at breakfast must read TRUNCATED on the
    REACH leg. Both expectations are DERIVED from `interval_s`.
    """
    want_n = expected_rows(interval_s)
    want_sod = expected_last_second_of_day(interval_s)
    last = rows[-1][0]
    # sadc fires ON the timer but the sample lands a few seconds later
    # (measured 0-20 s of jitter on this host), so the reach test asks
    # whether the last sample falls inside the FINAL EXPECTED INTERVAL --
    # itself derived from the cadence -- not whether it equals a second.
    count_ok = len(rows) == want_n
    reach_ok = want_sod <= last < want_sod + interval_s
    complete = count_ok and reach_ok
    d = {"n": len(rows), "expected_n": want_n, "complete": complete,
         "count_ok": count_ok, "reach_ok": reach_ok,
         "last_sample_hhmmss":
             f"{last // 3600:02d}:{last % 3600 // 60:02d}:{last % 60:02d}",
         "final_interval_opens_hhmm":
             f"{want_sod // 3600:02d}:{want_sod % 3600 // 60:02d}",
         "status": "COMPLETE" if complete else "TRUNCATED"}
    if not complete:
        d["why"] = (f"{len(rows)} of {want_n} samples "
                    f"(count_ok={count_ok}), last at "
                    f"{d['last_sample_hhmmss']}Z against a final interval "
                    f"opening {d['final_interval_opens_hhmm']}Z "
                    f"(reach_ok={reach_ok}) — TRUNCATED, which is the normal "
                    f"state of the archive for the day still in progress")
    return d


def archive_completeness(day: int, path: Path, interval_s: int) -> dict:
    """`completeness_of_rows` for one `saDD`, carrying the day it describes.

    The day is IN the return value, so a caller cannot quote a completeness
    verdict without saying which archive earned it -- and an unreadable
    archive comes back as a named status rather than as an absence.
    """
    try:
        rows = _sar_rows(path)
    except Refused as e:
        return {"day": day, "path": str(path), "status": "UNREADABLE",
                "complete": False, "count_ok": False, "reach_ok": False,
                "why": f"sa{day:02d}: {str(e)[:160]}"}
    d = completeness_of_rows(rows, interval_s)
    d["day"], d["path"] = day, str(path)
    if "why" in d:
        d["why"] = f"sa{day:02d} holds " + d["why"]
    return d


def newest_complete_archive() -> dict:
    """The day-relative LIVENESS subject, with every candidate's count.

    A day-of-month pin ages out -- sysstat's HISTORY deletes old `saDD`, and
    that is exactly how `sa25` disappeared under a check that had asserted it
    for weeks. This names no day in advance, reports the one it chose, and
    when nothing on the host is complete says so as a NAMED status instead of
    failing: an incomplete host is a fact about the host, and the parser's
    own control is elsewhere and always fires.
    """
    interval = collector_interval_s()
    cands = [archive_completeness(day, q, interval)
             for day, q in archives_present()]
    chosen = next((c for c in cands if c["complete"]), None)
    return {
        "status": "COMPLETE_ARCHIVE_FOUND" if chosen else "NO_COMPLETE_ARCHIVE",
        "chosen_day": chosen["day"] if chosen else None,
        "chosen_path": chosen["path"] if chosen else None,
        "n": chosen["n"] if chosen else None,
        "interval_s": interval, "expected_n": expected_rows(interval),
        "n_candidates": len(cands), "candidates": cands,
        "as_of_utc": dt.datetime.now(dt.timezone.utc)
                       .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def archive_month_check(day: int, mtime_epoch: float) -> dict:
    """Does `saDD` hold the STUDY month's day, or another month's?

    sysstat names archives by day-of-month and recycles them, so `sa03` holds
    whichever month last wrote it. `join()` dates every sample with
    YEAR/MONTH and joins it to that day's gap window, which means a recycled
    archive silently labels ANOTHER MONTH'S CPU as the study day -- a
    timestamp taken from a nearby proxy instead of the event that carries it
    (repo rule 3). This is the same calendar-dependence that made the old
    `sa25` control unreadable, seen from the other side: there it caused a
    red, here it would have caused a WRONG NUMBER.

    Returns a status, never a bare bool: the caller reports it.
    """
    m = dt.datetime.fromtimestamp(mtime_epoch, dt.timezone.utc)
    # sadc writes the day's last samples just before midnight and the
    # rotation lands early the next morning, so a study-month archive is
    # last written on its own day or the one after it.
    ok_dates = set()
    for delta in (0, 1):
        try:
            ok_dates.add((dt.datetime(YEAR, MONTH, day, tzinfo=dt.timezone.utc)
                          + dt.timedelta(days=delta)).date())
        except ValueError:
            pass
    hit = m.date() in ok_dates
    return {"day": day, "belongs": hit,
            "last_written_utc": m.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "study_month": f"{YEAR}-{MONTH:02d}",
            "why": None if hit else
            (f"sa{day:02d} was last written {m.date()}, so it holds "
             f"{m.strftime('%B')} data, not {YEAR}-{MONTH:02d}-{day:02d}. "
             f"Joining it to the study day's gap window would label another "
             f"month's CPU as the study day — REFUSING instead")}


def sar_cpu(day: int) -> list[tuple[int, float]]:
    """(epoch_of_sample_end, %idle) for a day of the STUDY month YEAR/MONTH.

    Refuses rather than returning empty: an absent archive is exactly the
    thing R-365 assumed without checking, and a silent [] here would let the
    same error recur wearing a different mask.
    """
    src = SYSSTAT / f"sa{day:02d}"
    rows = _sar_rows(src)                        # refuses first, dates after
    mc = archive_month_check(day, src.stat().st_mtime)
    if not mc["belongs"]:
        raise Refused(mc["why"])
    try:
        day_start = int(dt.datetime(YEAR, MONTH, day,
                                    tzinfo=dt.timezone.utc).timestamp())
    except ValueError as e:
        raise Refused(f"day {day} is not a day of {YEAR}-{MONTH:02d}: {e}")
    return [(day_start + sod, idle) for sod, idle in rows]


def disconnects(day: int, coin: str = "btc") -> list[int]:
    """The DECISION rows at their OWN timestamp (Codex HJ-R1).

    The first version selected any row carrying `gap_start_ns` — which on
    this ledger is the `gap_closed` row — and binned it at `gap_start_ns`,
    the LAST MARKET MESSAGE time. That is a nearby proxy, not the event's own
    clock, and repo rule 3 says timestamps come from the event that carries
    them. On 08-25 the two populations happen to be equal (665 each) and no
    event crosses a ten-minute bin, so the reported sign does not move; on
    another day it would. Fixed at the definition rather than argued away.
    """
    if not GAP_LEDGER.exists():
        raise Refused(f"{GAP_LEDGER} missing")
    lo = int(dt.datetime(YEAR, MONTH, day,
                         tzinfo=dt.timezone.utc).timestamp())
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
        if r.get("coin") != coin or r.get("event") != "disconnect":
            continue
        ns = r.get("recv_ns")
        if type(ns) is not int or ns <= 0:
            raise Refused(f"a {coin} disconnect row carries recv_ns={ns!r}, "
                          f"not a positive int — REFUSING rather than "
                          f"skipping it, because a silently dropped decision "
                          f"row lowers every count in this table")
        if lo <= ns // 10**9 < hi:
            out.append(ns // 10**9)
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
    day_start = int(dt.datetime(YEAR, MONTH, day,
                                tzinfo=dt.timezone.utc).timestamp())
    # Codex HJ-R1: starting at the SECOND row dropped the 00:00-00:10
    # interval — the single most-cited control in R-366 ("9 disconnects while
    # the host was 98.36% idle") was therefore absent from the correlation it
    # was quoted beside. sar's first row is the ENDPOINT of the first
    # interval, so anchor it at day start rather than discarding it.
    cells = []
    prev = day_start
    for end, idle in samples:
        cells.append({"start": prev, "end": end, "busy_pct": 100.0 - idle,
                      "n_disconnects": sum(1 for e in ev if prev <= e < end)})
        prev = end
    inside = sum(c["n_disconnects"] for c in cells)
    # rule 4: exclusions are STATUSES, never silent drops. 11 events used to
    # vanish here with nothing said.
    last_end = samples[-1][0]
    after = [e for e in ev if e >= last_end]
    before = [e for e in ev if e < day_start]
    return {
        "day": f"{YEAR}-{MONTH:02d}-{day:02d}", "coin": coin,
        "n_samples": len(cells), "n_events_total": len(ev),
        "n_events_joined": inside,
        "n_events_after_last_sample": len(after),
        "n_events_before_day_start": len(before),
        "coverage_as_of_utc": dt.datetime.fromtimestamp(
            last_end, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unevaluable_note": ("events at/after coverage_as_of_utc have no "
                             "sysstat interval and are UNEVALUABLE, reported "
                             "rather than dropped"),
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

    # THE JOIN CONTROL Codex HJ-R1 required, and the omission it names is the
    # sharpest thing in that finding: this instrument was written to correct a
    # rule-16 failure and shipped WITHOUT A CONTROL FOR ITS OWN CENTRAL
    # OPERATION. The six original checks exercised Pearson arithmetic and the
    # archive-absence refusal — everything EXCEPT the binning the tool exists
    # to do. A falsifier that tests the scaffolding is not a falsifier for
    # the mechanism.
    _samples = [(1000, 90.0), (2000, 80.0), (3000, 70.0)]

    def _bin(evts, day_start=0):
        cells, prev = [], day_start
        for end, idle in _samples:
            cells.append({"start": prev, "end": end,
                          "busy_pct": 100.0 - idle,
                          "n_disconnects": sum(1 for e in evts
                                               if prev <= e < end)})
            prev = end
        return cells
    c = _bin([500, 1500, 1500, 2500])
    ok([x["n_disconnects"] for x in c] == [1, 2, 1],
       "JOIN CONTROL: events at 500/1500/1500/2500 land in intervals "
       "[0,1000)/[1000,2000)/[2000,3000) as 1/2/1 — the binning is exercised, "
       "not assumed")
    ok(_bin([500])[0]["n_disconnects"] == 1,
       "JOIN CONTROL: an event BEFORE the first sar endpoint is counted in "
       "the anchored first interval. The pre-HJ-R1 code started at the "
       "SECOND row and dropped it — and that interval held the single most "
       "cited control in R-366 (9 disconnects at 98.36% idle)")
    _shift = _bin([999])
    ok(_shift[0]["n_disconnects"] == 1 and _shift[1]["n_disconnects"] == 0,
       "JOIN CONTROL (off-by-one): an event one second inside a boundary "
       "stays in the LEFT interval; shifting it would move the count and "
       "this check would fail")
    ok(_bin([3000])[-1]["n_disconnects"] == 0,
       "JOIN CONTROL: an event AT the right edge is excluded from the last "
       "half-open interval rather than double-counted")

    # ---------------------------------------------------------------- #
    # Q-DA-216 / DA-HOST-1. The old last check was ONE check doing TWO
    # jobs: `sar_cpu(25)` proved the parser works AND that the host still
    # held 08-25. sysstat recycles `saDD` by day-of-month, so the parser
    # half was a function of the calendar and went red on 2026-09-03 when
    # sa25 aged out -- a red that said nothing about the parser. Split:
    # a PARSER control that can always fire, and a LIVENESS check that is
    # day-relative and reports the host as a status.
    # ---------------------------------------------------------------- #

    # -- cadence: DERIVED, and the derivation itself is controlled --------
    ok(_interval_from_unit_text("[Timer]\nOnCalendar=*:00/10\n", "ctl")
       == 600,
       "POSITIVE CONTROL (cadence): `OnCalendar=*:00/10` parses to 600 s")
    ok(_interval_from_unit_text("[Timer]\nOnCalendar=*:00/05\n", "ctl")
       == 300,
       "POSITIVE CONTROL (cadence): `OnCalendar=*:00/05` parses to 300 s — "
       "the same code returns a DIFFERENT answer for a different unit, so "
       "the interval is read rather than pinned")
    for _bad, _why in (("[Timer]\nOnBootSec=5min\n", "no OnCalendar at all"),
                       ("[Timer]\nOnCalendar=daily\n", "a form it cannot "
                        "convert to seconds")):
        try:
            _interval_from_unit_text(_bad, "ctl")
            ok(False, f"KNOWN-BAD (cadence): a unit with {_why} must REFUSE")
        except Refused:
            ok(True, f"KNOWN-BAD (cadence): a unit with {_why} REFUSES "
                     f"rather than falling back to a guessed interval")
    _iv = 600
    ok(expected_rows(_iv) == 143 and expected_last_second_of_day(_iv) == 85800,
       "DERIVATION: at a 600 s cadence a complete UTC day is 86400/600-1 = "
       "143 rows ending in the interval that opens 23:50Z. The numbers are "
       "stated HERE and computed THERE — the module holds no such literal")
    _prod = SRC[:SRC.index("def selftest(")]
    _lits = cadence_literals_in_code(_prod, _iv)
    ok(not _lits,
       f"NO-LITERAL (self-read): the production region writes down no "
       f"derived expectation — no 140-144, no 85800/1430, no '23:50'/'23:00' "
       f"in an operand. Every one is computed from the timer. Found: {_lits}")
    ok(cadence_literals_in_code("want = 143\nif n >= 140: pass\n", _iv),
       "POSITIVE CONTROL (no-literal): the scanner FIRES on `want = 143` and "
       "`n >= 140` — without this the clean result above could be a regex "
       "that never matches anything (rule 15)")
    ok(cadence_literals_in_code('t = "23:50"\n', _iv),
       "POSITIVE CONTROL (no-literal): it fires on a wall-clock literal used "
       "as an operand")
    ok(not cadence_literals_in_code(
           '"""A docstring reporting 143 cells through 23:50Z."""\n', _iv),
       "KNOWN-BAD (no-literal): a docstring RECORDING 143 cells through "
       "23:50Z does NOT fire — prose is exempt on purpose, and the exemption "
       "is bounded by the two controls above rather than assumed")

    # -- the PARSER control: content-addressed, calendar-independent -------
    ok(PARSER_FIXTURE.exists(), f"PARSER CONTROL: the committed fixture "
                                f"{PARSER_FIXTURE.name} is present")
    _b = PARSER_FIXTURE.read_bytes()
    ok(len(_b) == PARSER_FIXTURE_BYTES
       and hashlib.sha256(_b).hexdigest() == PARSER_FIXTURE_SHA256,
       f"PARSER CONTROL: the fixture is content-addressed — "
       f"{len(_b)} B / sha256 {hashlib.sha256(_b).hexdigest()[:16]}… matches "
       f"the constants, so this control cannot be quietly re-recorded")
    _fx = _sar_rows(PARSER_FIXTURE)
    ok(len(_fx) == PARSER_FIXTURE_ROWS,
       f"POSITIVE CONTROL (parser): the fixture parses to "
       f"{len(_fx)} == PARSER_FIXTURE_ROWS rows. This check does not depend "
       f"on the calendar, on sysstat's HISTORY, or on the host being up — "
       f"it can always fire (rule 15)")
    ok([r[0] for r in _fx] == sorted(r[0] for r in _fx)
       and len(set(r[0] for r in _fx)) == len(_fx),
       "POSITIVE CONTROL (parser): sample stamps are strictly increasing "
       "seconds-of-day, so the ordering the join relies on is exercised")
    # COLUMN BINDING, read INDEPENDENTLY of the production regex: group 9 is
    # %idle. Bound to %steal instead, every busy_pct in this repo would be
    # ~100 and the R-366 table would have read the opposite way round.
    _txt = subprocess.run(["sar", "-f", str(PARSER_FIXTURE), "-u"],
                          capture_output=True, text=True).stdout
    _indep = [ln.split() for ln in _txt.splitlines()
              if re.match(r"\d\d:\d\d:\d\d\s+all\b", ln)]
    ok(len(_indep) == len(_fx)
       and all(abs(float(f[-1]) - v) < 1e-9
               for f, (_, v) in zip(_indep, _fx)),
       "COLUMN BINDING: a SECOND, independent read of the same report (split "
       "on whitespace, last field) agrees with the production regex's group 9 "
       "row for row — %idle, not %steal or %iowait")
    ok(all(float(f[-1]) == max(float(x) for x in f[2:]) for f in _indep),
       "COLUMN BINDING: on this idle-host fixture %idle is the LARGEST of the "
       "six columns in every row, so a shifted binding would be visible")
    for _badf, _why in ((Path(__file__).resolve(), "a text file that is not "
                                                   "a sysstat archive"),
                        (SYSSTAT / "sa99", "an absent path")):
        try:
            _sar_rows(_badf)
            ok(False, f"KNOWN-BAD (parser): {_why} must REFUSE")
        except Refused:
            ok(True, f"KNOWN-BAD (parser): {_why} REFUSES — the control "
                     f"proves the parser can go red, not merely that it "
                     f"stayed green")

    # -- the LIVENESS check: day-relative, a STATUS about the host --------
    _full = [(_iv * i, 90.0) for i in range(1, expected_rows(_iv) + 1)]
    ok(completeness_of_rows(_full, _iv)["status"] == "COMPLETE",
       "POSITIVE CONTROL (liveness): a synthetic full day reads COMPLETE")
    ok(completeness_of_rows(_full[:-1], _iv)["count_ok"] is False,
       "KNOWN-BAD (liveness): one row short fails the COUNT leg — a day with "
       "a hole in the middle is not complete even if it reaches midnight")
    _stops = [(_iv * i, 90.0) for i in range(1, 6)]
    _sd = completeness_of_rows(_stops, _iv)
    ok(_sd["reach_ok"] is False and "00:50" in _sd["why"],
       "KNOWN-BAD (liveness): an archive that stops at 00:50Z fails the REACH "
       "leg and its `why` NAMES the time it reached")
    _jit = _full[:-1] + [(85800 + _iv - 1, 90.0)]
    ok(completeness_of_rows(_jit, _iv)["reach_ok"] is True,
       "BOUNDARY: a sample one second before the next interval opens still "
       "counts — sadc jitters a few seconds past the timer (measured 0-20 s "
       "on this host) and an equality test would call every real day short")
    _over = _full[:-1] + [(85800 + _iv, 90.0)]
    ok(completeness_of_rows(_over, _iv)["reach_ok"] is False,
       "BOUNDARY: a sample a full interval late does NOT count — the "
       "tolerance is one interval wide, derived, not open-ended")
    _today = int(dt.datetime.now(dt.timezone.utc).strftime("%d"))
    _tc = archive_completeness(_today, SYSSTAT / f"sa{_today:02d}",
                               collector_interval_s())
    ok(_tc["complete"] is False and f"sa{_today:02d}" in _tc["why"],
       f"KNOWN-BAD (liveness): TODAY's archive sa{_today:02d} cannot be "
       f"complete — the day has not ended — and it is refused BY NAME with "
       f"its count, not silently skipped: {_tc.get('status')}")
    _sm = dt.datetime(YEAR, MONTH, 25, 12, tzinfo=dt.timezone.utc).timestamp()
    ok(archive_month_check(25, _sm)["belongs"] is True,
       "POSITIVE CONTROL (recycling): an archive last written on its own day "
       "of the study month BELONGS — the guard admits the data it exists to "
       "protect")
    ok(archive_month_check(25, _sm + 86400)["belongs"] is True,
       "POSITIVE CONTROL (recycling): last written the FOLLOWING morning "
       "still belongs — sysstat rotates just after midnight, and a guard "
       "that refused that would refuse every real archive")
    _rc = archive_month_check(3, dt.datetime(YEAR, MONTH + 1, 3, 1,
                                             tzinfo=dt.timezone.utc)
                              .timestamp())
    ok(_rc["belongs"] is False and "September" in _rc["why"],
       "KNOWN-BAD (recycling): `sa03` last written in SEPTEMBER is refused "
       "for the August study day it would otherwise be dated as. Without "
       "this, `--json` with no --day joins today's CPU to an August gap "
       "window and prints it under an August label (repo rule 3)")
    try:
        sar_cpu(_today)
        ok(False, "KNOWN-BAD (recycling): today's recycled archive must "
                  "REFUSE through the real path, not only the pure one")
    except Refused as _ex:
        ok("holds" in str(_ex) and f"sa{_today:02d}" in str(_ex),
           f"KNOWN-BAD (recycling), REAL PATH: sar_cpu({_today}) refuses BY "
           f"NAME because sa{_today:02d} carries this month's data while the "
           f"study month is {YEAR}-{MONTH:02d} — the refusal reaches the "
           f"caller, which reports it as a status rather than a table row")

    _lv = newest_complete_archive()
    ok(_lv["status"] in ("COMPLETE_ARCHIVE_FOUND", "NO_COMPLETE_ARCHIVE"),
       f"LIVENESS: the host reports a NAMED status — {_lv['status']} over "
       f"{_lv['n_candidates']} archives, as of {_lv['as_of_utc']}. "
       f"'No complete archive' is a fact about the HOST and is allowed to be "
       f"the answer; the parser's control is above and fires regardless")
    _cands = _lv["candidates"]
    _first = next((i for i, c in enumerate(_cands) if c["complete"]), None)
    ok((_lv["chosen_day"] is None) == (_first is None)
       and (_first is None
            or _cands[_first]["day"] == _lv["chosen_day"]),
       f"SELECTOR COHERENCE: the chosen day is the FIRST complete candidate "
       f"in newest-first order (chose {_lv['chosen_day']}), and is None "
       f"exactly when none is complete — the selector's report cannot "
       f"disagree with its own table")

    ok(len(checks) + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {len(checks) + 1} == "
       f"{EXPECTED_CHECKS}")
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
    print(f"{'day':12} {'cells':>6} {'joined':>7} {'unevaluable':>12} "
          f"{'mean busy%':>11} {'pearson r':>10}")
    for r in res:
        pr = r["pearson_busy_vs_disconnects"]
        print(f"{r['day']:12} {r['n_samples']:6d} {r['n_events_joined']:7d} "
              f"{r['n_events_after_last_sample']:12d} "
              f"{r['mean_busy_pct']:11.2f} "
              f"{'n/a' if pr is None else f'{pr:+.3f}':>10}")
    # rule 10: compute the predicate, never print the conclusion. This
    # trailer used to print unconditionally -- including, as of 2026-09-03,
    # under an EMPTY table, because every archive it could reach had been
    # recycled into September. A verdict beside no rows is the failure the
    # rule names.
    r25 = next((r for r in res if r["day"].endswith("-25")), None)
    if r25 is None:
        print(f"\nNO ROW FOR 08-25 in this run ({len(res)} day(s) joined). "
              f"R-366's negative finding stands on its receipt, not on this "
              f"table: sysstat recycles saDD by day-of-month and 08-25 has "
              f"aged out of /var/log/sysstat. Nothing here re-establishes it.")
    else:
        pr = r25["pearson_busy_vs_disconnects"]
        print(f"\n08-25 REJOINED: r = {'n/a' if pr is None else f'{pr:+.3f}'} "
              f"over {r25['n_samples']} cells, mean busy "
              f"{r25['mean_busy_pct']}% — host load does not explain the "
              f"break. See the module docstring for the controls that make "
              f"that readable rather than merely a small number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
