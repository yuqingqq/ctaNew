#!/usr/bin/env python3
"""Collector-wide DARK INTERVALS: contiguous low-content time no gap row knows.

WHAT IS MISSING THAT THIS SUPPLIES. The two 2026-09-01 outages -- 00:00-01:05Z
(65 min) and 22:45-23:35Z (50 min), every coin at a fraction of a percent of
normal content, no gap rows -- are an ESTABLISHED FACT (RESULTS.md SS3; two
independent instruments, collector-log msgs/s and raw gzip-trailer bytes,
agreeing to one minute). What did not exist was a CHECKER: something that
computes the predicate, ships a falsifier in both directions, and can be swept
over the tape so the next one is found by a run rather than by a person
noticing. This is that checker. It re-derives nothing about WHY those intervals
happened and asserts nothing about it.

WHY THE EXISTING INSTRUMENTS DO NOT COVER IT, computed rather than asserted
(see `--selftest`, REAL-3 and REAL-5):

  * the DURATION bars (P1/P2/P3) charge for time the GAP LEDGER knows about.
    Inside 09-01 00:00-01:05Z the ledger knows **18.7 of 27,300 coin-seconds
    (0.07%)**; inside 22:45-23:35Z, 785.5 of 21,000 (3.74%). The bars pass
    straight through both because almost nothing was reported lost.
  * `pm_tape_density` counts THIN WINDOWS per (day, coin) against THAT DAY's
    median -- window-level, per-coin, and relative to a denominator the outage
    itself drags down.
  * the frozen content-liveness rules judge PER COIN. v1 is relative to the
    same day (RR6-1's blind spot); v2 (R-424) fixes the denominator but is
    still one coin at a time, and its run bar is 12 windows -- the 50-minute
    interval is 10.

  This instrument asks the question none of them asks: **were ALL judged coins
  dark AT THE SAME TIME, for how long, and how much of it did the ledger see.**
  Seven simultaneous coins is an infrastructure- or venue-level event; one coin
  is a coin's problem. That is why breadth is the definition and not a knob
  tuned to a result.

THIS ROUND IT REPORTS AND GOVERNS NOTHING (rule 14, and the same disposition
`pm_tape_density` carries under R-362). It stamps nothing into a verdict, moves
no frozen bar and no frozen constant, and no module imports it -- a property
asserted by a check in this file's own suite, with a planted import driving
that check red. Whether it ever governs is a USER freeze, routed by the
coordinator with the sweep in hand.

THE ONE CONSTANT THAT IS MINE WAS CHOSEN AFTER SEEING (rule 11), AND IS
THEREFORE REPORTED WITH ITS CURVE, NEVER USED AS A BAR. `LOW_FRAC` = 0.10
reproduces both established 09-01 intervals boundary-for-boundary. It was
picked knowing that. The whole sensitivity curve is emitted beside every
result, and it is not decoration: at 0.05 -- `pm_tape_density.THIN_FRAC` and
v2's `V2_DARK_FRAC`, both frozen -- the 00:00-01:05Z interval DISAPPEARS
ENTIRELY, because `hype` sits at 5.5-6.0% of its reference while the other six
are at 0.1-3%. A single coin one and a half points above a frozen threshold
hides a 65-minute all-coin blackout. That is the reading this file exists to
make impossible to miss, and it is a fact about the threshold, not a proposal
to move one.

THE REFERENCE IS THE FROZEN ONE, IMPORTED AND NOT RESTATED. Darkness is
measured against `da_content_liveness_v2_check.trailing_reference` -- the
median of the coin's prior complete days, point-in-time by construction, so
the day under test cannot move its own denominator. `V2_TRAILING_DAYS` (7) and
`V2_MIN_REFERENCE_DAYS` (3) are v2's, unchanged and un-re-chosen; a coin
without a reference is NO_REFERENCE, which is a status and never a pass.

EXCLUSIONS ARE STATUSES AND AN EMPTY POPULATION REFUSES (rule 4, standing rule
11). A day with no raw directory, no window files, or no coin that can be
judged does not report "0 dark intervals" -- it raises `ScanRefused` by name.
Every day row carries the count of coins judged, the count without a reference
and the count with too few windows, so a zero is never bare.

    python3 live/pm_research/da_dark_interval_scan.py                # sweep
    python3 live/pm_research/da_dark_interval_scan.py --day 20260901
    python3 live/pm_research/da_dark_interval_scan.py --json
    python3 live/pm_research/da_dark_interval_scan.py --selftest     # rule 15
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pm_tape_density as TD                                    # noqa: E402
import da_content_liveness_v2_check as V2                       # noqa: E402

CODE_ROOT = Path(__file__).resolve().parents[2]

WINDOW_S = TD.WINDOW_S

#: MINE, AND CHOSEN AFTER SEEING. See the module docstring: 0.10 reproduces
#: both established 09-01 intervals exactly, and it was selected knowing that.
#: It is therefore a REPORTING setting with a published curve, never a bar.
#: A GOVERNING floor would have to be pre-registered against days nobody has
#: looked at -- a USER act (rule 14), not this file's.
LOW_FRAC = 0.10

#: An INTERVAL, not a window. Two consecutive windows is the smallest thing
#: the word can mean; it is the least selective choice available, so it hides
#: nothing. Single-window dips are still counted and reported separately.
MIN_RUN_WINDOWS = 2

#: THE DEFINITION, NOT A KNOB. 1.0 = every judged coin dark in the same
#: window. This is what separates an infrastructure/venue event from one
#: coin's feed, and it is the reason this instrument is not a re-parameterised
#: `pm_tape_density`. Exposed as an argument so the suite can drive it in both
#: directions, and swept in the curve so the reader sees what it costs.
BREADTH = 1.0

#: Published with every result. The event COUNT is stable over [0.10, 0.25];
#: below it the set collapses and the 09-01 00:00-01:05Z interval vanishes.
SENSITIVITY_FRACS = (0.02, 0.05, 0.10, 0.15, 0.25)
SENSITIVITY_BREADTHS = (1.0, 6.0 / 7.0)

#: The same disposition `pm_tape_density` carries (R-362) and for the same
#: reason. Asserted in the suite by `no_module_imports_this()`.
DISPOSITION = "REPORTED_NOT_GOVERNING"

#: This suite's own total, asserted over ran + skipped. There are NO skips:
#: the real-tape checks FAIL BY NAME when the tape is absent rather than
#: quietly shrinking the count (DA20-R3's class), and a named SKIP standing in
#: for a positive control is ruled out.
EXPECTED_CHECKS = 38


class ScanRefused(Exception):
    """A population this instrument must not summarise."""


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _hhmm(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%H:%M:%SZ")


def _contiguous(windows: Iterable[int]) -> list[list[int]]:
    """Maximal runs of window starts spaced exactly one window apart."""
    out: list[list[int]] = []
    cur: list[int] = []
    for w in sorted(windows):
        if cur and w - cur[-1] == WINDOW_S:
            cur.append(w)
        else:
            if cur:
                out.append(cur)
            cur = [w]
    if cur:
        out.append(cur)
    return out


def ledger_seconds_in_span(gaps: dict, coins: Iterable[str],
                           t0: float, t1: float) -> dict[str, Any]:
    """How much of this span the GAP LEDGER already knows about.

    This is the quantity that decides whether an interval is invisible to the
    duration bars, and it is the honest form of "no gap rows": not a boolean
    over rows, but the SECONDS the bars were charged against the seconds that
    actually went dark. A single 18.7 s row inside 27,300 coin-seconds is not
    "the ledger saw it".
    """
    per: dict[str, dict[str, Any]] = {}
    total = 0.0
    rows = 0
    for c in sorted(coins):
        s = 0.0
        n = 0
        for gs, ge in gaps.get(c, ()):
            overlap = min(ge, t1) - max(gs, t0)
            if overlap > 0:
                s += overlap
                n += 1
        per[c] = {"ledger_seconds": round(s, 1), "n_rows": n}
        total += s
        rows += n
    coin_seconds = (t1 - t0) * len(list(per))
    return {
        "coin_seconds_in_span": int(coin_seconds),
        "ledger_known_coin_seconds": round(total, 1),
        "ledger_known_share": (round(total / coin_seconds, 6)
                               if coin_seconds else None),
        "n_gap_rows_intersecting": rows,
        # The STRICT form, kept beside the share so neither stands alone.
        "unaccounted": rows == 0,
        "per_coin": per,
    }


def day_references(day: str, all_days: list[str], medians: dict,
                   coins: Iterable[str]) -> dict[str, float | None]:
    """The FROZEN v2 point-in-time reference, per coin. Not restated here."""
    return {c: V2.trailing_reference(day, c, all_days, medians)
            for c in sorted(coins)}


def scan_day(day: str, all_days: list[str], medians: dict, gaps: dict,
             frac: float = LOW_FRAC, breadth: float = BREADTH,
             min_run: int = MIN_RUN_WINDOWS,
             raw_root: Path | None = None) -> dict[str, Any]:
    """Dark intervals for one UTC day.

    REFUSES rather than reporting a clean day when the population cannot
    support the question: no raw directory, no window files, or no coin with
    both a reference and enough windows. "0 intervals" must mean "the tape was
    read and held none", never "there was nothing to read".
    """
    agg = TD.scan_day(day, TD.RAW if raw_root is None else raw_root)
    per: dict[str, dict[int, int]] = collections.defaultdict(dict)
    for (c, w), b in agg.items():
        per[c][w] = b
    coins = sorted(per)
    refs = day_references(day, all_days, medians, coins)

    coin_status: dict[str, Any] = {}
    judged: list[str] = []
    for c in coins:
        if len(per[c]) < TD.MIN_WINDOWS_FOR_MEDIAN:
            coin_status[c] = {
                "status": "TOO_FEW_WINDOWS", "n_windows": len(per[c]),
                "why": f"{len(per[c])} windows < "
                       f"{TD.MIN_WINDOWS_FOR_MEDIAN}"}
        elif refs[c] is None:
            coin_status[c] = {
                "status": "NO_REFERENCE", "n_windows": len(per[c]),
                "why": f"fewer than {V2.V2_MIN_REFERENCE_DAYS} prior complete "
                       f"days in the trailing {V2.V2_TRAILING_DAYS}; "
                       f"NO_REFERENCE is not a pass"}
        else:
            coin_status[c] = {"status": "JUDGED", "n_windows": len(per[c]),
                              "reference_bytes": int(refs[c])}
            judged.append(c)

    if not judged:
        raise ScanRefused(
            f"REFUSED for {day}: NO coin can be judged "
            f"({ {c: v['status'] for c, v in coin_status.items()} }). A day "
            f"with no judgeable coin has no dark-interval count, and "
            f"reporting 0 for it would be the empty-set trap on the "
            f"instrument that exists to see a blackout.")

    need = int(round(breadth * len(judged)))
    if need < 1:
        raise ScanRefused(
            f"REFUSED for {day}: breadth {breadth} over {len(judged)} judged "
            f"coins requires {need} dark coins -- a criterion 0 coins satisfy "
            f"would flag every window.")

    # THE GRID IS THE OBSERVED SPAN, NOT THE OBSERVED WINDOWS. A window
    # missing for EVERY coin is a hole in the middle of the day, and taking
    # the union of what exists would make it disappear rather than read as
    # dark -- the worst case reading cleanest. The grid runs from the first
    # observed window to the last, so a day still in progress is not charged
    # for the hours that have not happened yet.
    observed = sorted({w for c in judged for w in per[c]})
    all_windows = list(range(observed[0], observed[-1] + WINDOW_S, WINDOW_S))
    n_absent_all = sum(1 for w in all_windows
                       if not any(w in per[c] for c in judged))
    dark_at: dict[int, list[str]] = {}
    for w in all_windows:
        # ABSENT IS DARK, DELIBERATELY: a window with no file carries no
        # content, and treating a missing file as 'not dark' would let the
        # worst case read cleanest.
        d = [c for c in judged if per[c].get(w, 0) < refs[c] * frac]
        if len(d) >= need:
            dark_at[w] = d

    instances = []
    singles = 0
    for run in _contiguous(dark_at):
        if len(run) < min_run:
            singles += 1
            continue
        t0, t1 = run[0], run[-1] + WINDOW_S
        cells = [(c, w) for w in run for c in dark_at[w]]
        covered = [(c, w) for c, w in cells
                   if TD.gap_overlaps(gaps, c, w, w + WINDOW_S)]
        fracs = [per[c].get(w, 0) / refs[c] for w in run for c in judged]
        led = ledger_seconds_in_span(gaps, judged, t0, t1)
        instances.append({
            "day": day,
            "start_utc": _iso(t0), "end_utc": _iso(t1),
            "start_hhmm": _hhmm(t0), "end_hhmm": _hhmm(t1),
            "n_windows": len(run), "span_s": int(t1 - t0),
            "n_coins_dark_min": min(len(dark_at[w]) for w in run),
            "n_coins_judged": len(judged),
            "coins_dark_throughout": sorted(
                set.intersection(*[set(dark_at[w]) for w in run])),
            "n_cells": len(cells),
            "n_cells_gap_covered": len(covered),
            "worst_frac_of_reference": round(max(fracs), 6),
            "median_frac_of_reference": round(statistics.median(fracs), 6),
            **led,
        })

    return {
        "day": day,
        "setting": {"low_frac": frac, "breadth": breadth,
                    "min_run_windows": min_run,
                    "coins_required_dark": need},
        "n_coins_total": len(coins),
        "n_coins_judged": len(judged),
        "n_coins_no_reference": sum(
            1 for v in coin_status.values() if v["status"] == "NO_REFERENCE"),
        "n_coins_too_few_windows": sum(
            1 for v in coin_status.values()
            if v["status"] == "TOO_FEW_WINDOWS"),
        "coins": coin_status,
        "n_windows_scanned": len(all_windows),
        "n_windows_observed": len(observed),
        "n_windows_absent_for_all_coins": n_absent_all,
        "n_intervals": len(instances),
        "n_single_window_dips_excluded": singles,
        "intervals": instances,
    }


def sweep(days: list[str] | None = None, frac: float = LOW_FRAC,
          breadth: float = BREADTH, min_run: int = MIN_RUN_WINDOWS,
          raw_root: Path | None = None, gaps: dict | None = None,
          medians: dict | None = None) -> dict[str, Any]:
    """Every day on the tape, each one judged or REFUSED by name."""
    root = TD.RAW if raw_root is None else raw_root
    all_days = TD.all_days(root)
    if not all_days:
        raise ScanRefused(
            f"REFUSED: no day directories under {root}. An empty tape is not "
            f"a clean tape, and a sweep of nothing must not report 0 dark "
            f"intervals.")
    want = list(all_days) if days is None else list(days)
    if not want:
        raise ScanRefused("REFUSED: an empty day list is the empty-set trap, "
                          "not a clean sweep.")
    gaps = TD.load_gaps() if gaps is None else gaps
    medians = V2.day_medians(all_days, root) if medians is None else medians

    rows = []
    for d in want:
        try:
            rows.append(scan_day(d, all_days, medians, gaps, frac, breadth,
                                 min_run, root))
        except (ScanRefused, TD.Refused) as e:
            rows.append({"day": d, "status": "REFUSED", "why": str(e)})
    judged_rows = [r for r in rows if r.get("status") != "REFUSED"]
    if not judged_rows:
        raise ScanRefused(
            f"REFUSED: every one of {len(want)} day(s) refused; a sweep in "
            f"which nothing could be judged has no interval count.")
    intervals = [iv for r in judged_rows for iv in r["intervals"]]
    return {
        "instrument": "da_dark_interval_scan",
        "disposition": DISPOSITION,
        "as_of_utc": _iso(dt.datetime.now(dt.timezone.utc).timestamp()),
        "data_root": str(TD.DATA_ROOT), "data_root_branch": TD.DATA_ROOT_BRANCH,
        "raw_root": str(root),
        "setting": {"low_frac": frac, "breadth": breadth,
                    "min_run_windows": min_run},
        "reference": {
            "module": "da_content_liveness_v2_check",
            "function": "trailing_reference",
            "trailing_days": V2.V2_TRAILING_DAYS,
            "min_reference_days": V2.V2_MIN_REFERENCE_DAYS,
            "frozen_by_user": V2.FROZEN_BY_USER,
            "note": "point-in-time median of the coin's prior complete days; "
                    "the day under test cannot move its own denominator",
        },
        "n_days_requested": len(want),
        "n_days_judged": len(judged_rows),
        "n_days_refused": len(rows) - len(judged_rows),
        "days_refused": [r["day"] for r in rows if r.get("status") == "REFUSED"],
        "n_intervals": len(intervals),
        "n_days_with_an_interval": sum(1 for r in judged_rows
                                       if r["n_intervals"]),
        # DA27/DA24-class: THE EXCLUDED POPULATION, IN THE HEADLINE.
        # `min_run` is a REPORTING threshold, not a detection one: a
        # single-window dropout is DETECTED (it reads 0.00% of reference when
        # every coin's file is absent) and then dropped from `intervals`
        # because one window is not an "interval". That is defensible as a
        # definition and indefensible as a silence -- 2026-09-03T15:20:00Z,
        # the third instance of the class this module exists for, was excluded
        # exactly this way and nobody saw it. The count was already computed
        # per day; it is now carried in the sweep, so the number a reader
        # meets includes what the definition threw away.
        "n_single_window_dips_excluded": sum(
            r["n_single_window_dips_excluded"] for r in judged_rows),
        "single_window_dip_days": sorted(
            r["day"] for r in judged_rows
            if r["n_single_window_dips_excluded"]),
        "intervals": intervals,
        "spanning_events": merge_adjacent(intervals),
        "n_spanning_events": len(merge_adjacent(intervals)),
        "days": rows,
        "threshold_note": (
            "low_frac was chosen AFTER seeing the 2026-09-01 intervals "
            "(rule 11). This instrument therefore REPORTS and does not "
            "govern; read `sensitivity` beside every count."),
    }


def sensitivity(days: list[str] | None = None, raw_root: Path | None = None,
                gaps: dict | None = None, medians: dict | None = None,
                fracs: tuple = SENSITIVITY_FRACS,
                breadths: tuple = SENSITIVITY_BREADTHS) -> list[dict]:
    """The whole grid, because one reading privileges one choice."""
    root = TD.RAW if raw_root is None else raw_root
    all_days = TD.all_days(root)
    gaps = TD.load_gaps() if gaps is None else gaps
    medians = V2.day_medians(all_days, root) if medians is None else medians
    out = []
    for b in breadths:
        for f in fracs:
            s = sweep(days, f, b, MIN_RUN_WINDOWS, root, gaps, medians)
            out.append({
                "low_frac": f, "breadth": round(b, 4),
                "n_intervals": s["n_intervals"],
                "n_days_with_an_interval": s["n_days_with_an_interval"],
                "total_dark_window_slots": sum(i["n_windows"]
                                               for i in s["intervals"]),
                "spans": [f"{i['day']} {i['start_hhmm']}-{i['end_hhmm']}"
                          for i in s["intervals"]],
            })
    return out


def merge_adjacent(intervals: list[dict]) -> list[dict]:
    """Instances that touch end-to-start are ONE event, and midnight is not.

    The scan is per UTC day because the tape is stored per UTC day. That is a
    property of the storage, not of the feed, and taking it for a property of
    the event is how a 105-minute blackout gets recorded as a 65-minute one:
    2026-08-31 23:20Z -> 2026-09-01 01:05Z is CONTIGUOUS, and only the day
    boundary between them makes it two rows. Reported BESIDE the per-day
    instances, never instead of them -- the day rows are what a day verdict
    would ever read, and this is what a reader should quote.
    """
    out: list[dict] = []
    for iv in sorted(intervals, key=lambda x: x["start_utc"]):
        if out and out[-1]["end_utc"] == iv["start_utc"]:
            g = out[-1]
            g["end_utc"] = iv["end_utc"]
            g["days"].append(iv["day"])
            g["n_windows"] += iv["n_windows"]
            g["span_s"] += iv["span_s"]
            g["members"].append(f"{iv['day']} {iv['start_hhmm']}-"
                                f"{iv['end_hhmm']}")
            g["ledger_known_coin_seconds"] = round(
                g["ledger_known_coin_seconds"]
                + iv["ledger_known_coin_seconds"], 1)
            g["coin_seconds_in_span"] += iv["coin_seconds_in_span"]
            g["n_gap_rows_intersecting"] += iv["n_gap_rows_intersecting"]
            g["crosses_day_boundary"] = len(set(g["days"])) > 1
        else:
            out.append({
                "start_utc": iv["start_utc"], "end_utc": iv["end_utc"],
                "days": [iv["day"]], "n_windows": iv["n_windows"],
                "span_s": iv["span_s"],
                "members": [f"{iv['day']} {iv['start_hhmm']}-"
                            f"{iv['end_hhmm']}"],
                "ledger_known_coin_seconds": iv["ledger_known_coin_seconds"],
                "coin_seconds_in_span": iv["coin_seconds_in_span"],
                "n_gap_rows_intersecting": iv["n_gap_rows_intersecting"],
                "crosses_day_boundary": False,
            })
    for g in out:
        g["ledger_known_share"] = (
            round(g["ledger_known_coin_seconds"] / g["coin_seconds_in_span"], 6)
            if g["coin_seconds_in_span"] else None)
    return out


def no_module_imports_this(root: Path | None = None) -> dict[str, Any]:
    """Is this instrument wired into anything? This round it must not be.

    Rule 17 in the other direction: the usual failure is a green suite with no
    call site, and the usual fix is to wire it. Here the ROUND's declaration is
    that nothing consumes this yet -- so the declaration is a computed
    predicate over the source tree, not a sentence. The scanner is driven
    against a PLANTED import in the suite, so it has shown it can fire.
    """
    root = (CODE_ROOT / "live" / "pm_research") if root is None else root
    me = Path(__file__).stem
    hits = []
    for p in sorted(root.glob("*.py")):
        if p.resolve() == Path(__file__).resolve():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for n, line in enumerate(txt.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if (f"import {me}" in s) or (f"from {me} " in s) \
                    or (f'"{me}"' in s and "import" in s):
                hits.append({"file": p.name, "line": n, "text": s[:120]})
    return {"scanned_dir": str(root), "n_files_scanned":
            len(list(root.glob("*.py"))), "importers": hits,
            "unwired": not hits}


# --------------------------------------------------------------------- CLI
def _emit(s: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(s, indent=2, sort_keys=True))
        return
    print(f"da_dark_interval_scan  [{DISPOSITION}]  as_of {s['as_of_utc']}")
    print(f"  root {s['raw_root']}  branch {s['data_root_branch']}")
    print(f"  setting {s['setting']}")
    print(f"  days judged {s['n_days_judged']}/{s['n_days_requested']}"
          f"  refused {s['n_days_refused']} {s['days_refused']}")
    print(f"  intervals {s['n_intervals']} over "
          f"{s['n_days_with_an_interval']} day(s)")
    print(f"  single-window dips EXCLUDED by min_run="
          f"{s['setting']['min_run_windows']}: "
          f"{s['n_single_window_dips_excluded']} on "
          f"{s['single_window_dip_days']}  <-- detected, not reported as "
          f"intervals; re-run with --min-run 1 to see them")
    for i in s["intervals"]:
        print(f"    {i['day']}  {i['start_hhmm']}-{i['end_hhmm']}  "
              f"{i['n_windows']:>3}w {i['span_s']:>6}s  "
              f"coins {i['n_coins_dark_min']}/{i['n_coins_judged']}  "
              f"worst {100*i['worst_frac_of_reference']:6.2f}% of ref  "
              f"ledger knew {i['ledger_known_coin_seconds']:>8.1f}s of "
              f"{i['coin_seconds_in_span']:>6d} "
              f"({100*i['ledger_known_share']:5.2f}%)  rows "
              f"{i['n_gap_rows_intersecting']}")
    cross = [g for g in s["spanning_events"] if g["crosses_day_boundary"]]
    print(f"  spanning events {s['n_spanning_events']} "
          f"({len(cross)} crossing a UTC day boundary)")
    for g in cross:
        print(f"    {g['start_utc']} -> {g['end_utc']}  "
              f"{g['n_windows']}w {g['span_s']}s  members {g['members']}")
    print(f"  {s['threshold_note']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", action="append", default=None)
    ap.add_argument("--low-frac", type=float, default=LOW_FRAC)
    ap.add_argument("--breadth", type=float, default=BREADTH)
    ap.add_argument("--min-run", type=int, default=MIN_RUN_WINDOWS)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    try:
        s = sweep(a.day, a.low_frac, a.breadth, a.min_run)
    except ScanRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    if a.sensitivity:
        s["sensitivity"] = sensitivity(a.day)
    _emit(s, a.json)
    if a.sensitivity and not a.json:
        print("  sensitivity (low_frac x breadth):")
        for r in s["sensitivity"]:
            print(f"    frac {r['low_frac']:<5} breadth {r['breadth']:<6} -> "
                  f"{r['n_intervals']:>2} interval(s) over "
                  f"{r['n_days_with_an_interval']} day(s), "
                  f"{r['total_dark_window_slots']} window slots")
    print(f"  wiring: {no_module_imports_this()}")
    return 0


# --------------------------------------------------------------- falsifier
def selftest() -> int:
    import gzip
    import shutil
    import tempfile

    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            print(f"FAIL: {label}")
            raise SystemExit(1)
        print(f"PASS: {label}")

    # ------------------------------------------------------------------
    # FIXTURE TIER -- runnable with NO tape at all. Every control fires in
    # both directions on data this file builds.
    # ------------------------------------------------------------------
    NORMAL = 20_000
    DAYS = ["20300101", "20300102", "20300103", "20300104", "20300105"]
    COINS = ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg"]
    NW = 48                                    # windows per fixture day

    def _day_epoch(tok: str) -> int:
        return int(dt.datetime(int(tok[:4]), int(tok[4:6]), int(tok[6:]),
                               tzinfo=dt.timezone.utc).timestamp())

    def build(root: Path, dark: dict[str, dict[int, int]] | None = None,
              days: list[str] = DAYS, coins: list[str] = COINS,
              nw: int = NW) -> None:
        """dark[day][window_index] -> {coin: bytes} override."""
        dark = dark or {}
        for d in days:
            dd = root / d
            dd.mkdir(parents=True, exist_ok=True)
            base = _day_epoch(d)
            for k in range(nw):
                w = base + k * WINDOW_S
                for c in coins:
                    n = dark.get(d, {}).get(k, {}).get(c, NORMAL)
                    (dd / f"{c}-updown-5m-{w}.jsonl.gz").write_bytes(
                        gzip.compress(b"x" * n))

    def med_for(root: Path, days: list[str] = DAYS) -> dict:
        return V2.day_medians(days, root)

    # --- FIX-1 .. FIX-3: the interval FIRES, and a healthy one of the SAME
    # length is ADMITTED. Both directions on one tape shape.
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        DARK = {DAYS[-1]: {k: {c: 200 for c in COINS} for k in range(10, 20)}}
        build(root, DARK)
        r = scan_day(DAYS[-1], DAYS, med_for(root), {}, raw_root=root)
        ok(r["n_intervals"] == 1,
           f"FIX-1 POSITIVE CONTROL: a 10-window all-coin interval at 1% of "
           f"reference is FLAGGED (got {r['n_intervals']} interval(s))")
        iv = r["intervals"][0]
        base = _day_epoch(DAYS[-1])
        ok(iv["n_windows"] == 10
           and iv["start_utc"] == _iso(base + 10 * WINDOW_S)
           and iv["end_utc"] == _iso(base + 20 * WINDOW_S),
           f"FIX-1b the flagged BOUNDARIES are the planted ones "
           f"({iv['start_utc']}..{iv['end_utc']}, {iv['n_windows']}w)")
        ok(iv["n_coins_dark_min"] == 7 and iv["n_coins_judged"] == 7
           and iv["coins_dark_throughout"] == sorted(COINS),
           "FIX-1c all seven judged coins are dark throughout")

        # THE OTHER DIRECTION, same length, same place: no interval at all.
        r2 = scan_day(DAYS[-2], DAYS, med_for(root), {}, raw_root=root)
        ok(r2["n_intervals"] == 0 and r2["n_coins_judged"] == 7,
           f"FIX-2 NEGATIVE CONTROL: a HEALTHY day of the same shape and the "
           f"same 10-window stretch is ADMITTED -- 0 intervals over 7 judged "
           f"coins (got {r2['n_intervals']})")
        ok(r2["n_windows_scanned"] == NW and r2["n_windows_observed"] == NW
           and r2["n_windows_absent_for_all_coins"] == 0,
           f"FIX-2b and it is a real scan, not an empty one: "
           f"{r2['n_windows_scanned']} windows read, none absent")

    # --- FIX-3: BREADTH is a conjunct that can fail AND pass.
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        D = {DAYS[-1]: {k: {c: (200 if c != "ggg" else NORMAL) for c in COINS}
                        for k in range(10, 20)}}
        build(root, D)
        m = med_for(root)
        r = scan_day(DAYS[-1], DAYS, m, {}, raw_root=root)
        ok(r["n_intervals"] == 0,
           f"FIX-3 six of seven coins dark is NOT a collector-wide interval "
           f"at breadth 1.0 (got {r['n_intervals']})")
        r = scan_day(DAYS[-1], DAYS, m, {}, breadth=6.0 / 7.0, raw_root=root)
        ok(r["n_intervals"] == 1 and r["intervals"][0]["n_windows"] == 10,
           "FIX-3b and the SAME tape flags it at breadth 6/7 -- the conjunct "
           "is driven in both directions, not asserted")

    # --- FIX-4: gap coverage separates ACCOUNTED from INVISIBLE loss.
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        DARK = {DAYS[-1]: {k: {c: 200 for c in COINS} for k in range(10, 20)}}
        build(root, DARK)
        m = med_for(root)
        base = _day_epoch(DAYS[-1])
        t0, t1 = base + 10 * WINDOW_S, base + 20 * WINDOW_S
        full = {c: [(t0, t1)] for c in COINS}
        r_inv = scan_day(DAYS[-1], DAYS, m, {}, raw_root=root)
        r_acc = scan_day(DAYS[-1], DAYS, m, full, raw_root=root)
        a, b = r_inv["intervals"][0], r_acc["intervals"][0]
        ok(a["unaccounted"] is True and a["ledger_known_share"] == 0.0
           and a["n_gap_rows_intersecting"] == 0,
           f"FIX-4 with an EMPTY ledger the interval reads wholly invisible "
           f"(share {a['ledger_known_share']}, rows "
           f"{a['n_gap_rows_intersecting']})")
        ok(b["unaccounted"] is False and b["ledger_known_share"] == 1.0
           and b["n_cells_gap_covered"] == b["n_cells"],
           f"FIX-4b with the SAME interval fully covered it reads wholly "
           f"ACCOUNTED (share {b['ledger_known_share']}, "
           f"{b['n_cells_gap_covered']}/{b['n_cells']} cells) -- the "
           f"attribute takes BOTH values, so it says something")

    # --- FIX-5b: THE EXCLUDED POPULATION IS CARRIED IN THE SWEEP.
    # A single-window dropout is DETECTED and then dropped from `intervals`
    # by min_run. The count must reach the headline, or the definition
    # becomes a silence -- which is what hid 2026-09-03T15:20:00Z.
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build(root, {DAYS[-1]: {12: {c: 200 for c in COINS}}})
        _sw1 = sweep(DAYS, raw_root=root, gaps={})
        ok(_sw1["n_intervals"] == 0
           and _sw1["n_single_window_dips_excluded"] == 1
           and _sw1["single_window_dip_days"] == [DAYS[-1]],
           f"FIX-5b THE SWEEP CARRIES WHAT min_run THREW AWAY: 0 intervals "
           f"and {_sw1['n_single_window_dips_excluded']} single-window dip on "
           f"{_sw1['single_window_dip_days']}. Detected, excluded by "
           f"definition, and NOT silent -- the shape that hid the third "
           f"instance of the class this module exists for")

    # --- FIX-5: MIN_RUN is a conjunct that can fail AND pass.
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        D = {DAYS[-1]: {12: {c: 200 for c in COINS}}}
        build(root, D)
        m = med_for(root)
        r = scan_day(DAYS[-1], DAYS, m, {}, raw_root=root)
        ok(r["n_intervals"] == 0 and r["n_single_window_dips_excluded"] == 1,
           "FIX-5 a ONE-window all-coin dip is not an interval at "
           "min_run=2, and it is COUNTED rather than dropped")
        r = scan_day(DAYS[-1], DAYS, m, {}, min_run=1, raw_root=root)
        ok(r["n_intervals"] == 1,
           "FIX-5b the same dip IS an interval at min_run=1")

    # --- FIX-6: NO_REFERENCE is a status and the day REFUSES.
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        short = DAYS[:2]                    # 1 prior day < V2_MIN_REFERENCE_DAYS
        build(root, {}, days=short)
        try:
            scan_day(short[-1], short, med_for(root, short), {}, raw_root=root)
            ok(False, "FIX-6 a day with too few priors must REFUSE")
        except ScanRefused as e:
            ok("NO coin can be judged" in str(e)
               and "empty-set trap" in str(e),
               f"FIX-6 too few prior days -> REFUSED BY NAME, not 0 intervals "
               f"({str(e)[:70]}...)")
        r = scan_day(short[-1], short, med_for(root, short), {},
                     raw_root=root) if False else None
        # and the per-coin status is NO_REFERENCE, not a silent drop
        agg_ok = True
        try:
            scan_day(short[-1], short, med_for(root, short), {}, raw_root=root)
        except ScanRefused as e:
            agg_ok = "NO_REFERENCE" in str(e)
        ok(agg_ok, "FIX-6b the refusal NAMES the per-coin status "
                   "(NO_REFERENCE), so the reason is in the message")

    # --- FIX-7: an empty tape / empty day list REFUSES.
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        try:
            sweep(None, raw_root=root)
            ok(False, "FIX-7 an empty raw root must REFUSE")
        except ScanRefused as e:
            ok("no day directories" in str(e),
               f"FIX-7 EMPTY TAPE -> REFUSED BY NAME ({str(e)[:60]}...)")
        build(root)
        try:
            sweep([], raw_root=root)
            ok(False, "FIX-7b an empty day list must REFUSE")
        except ScanRefused as e:
            ok("empty day list" in str(e),
               f"FIX-7b EMPTY DAY LIST -> REFUSED BY NAME ({str(e)[:50]}...)")
        try:
            scan_day("20300199", DAYS, med_for(root), {}, raw_root=root)
            ok(False, "FIX-7c an absent day must REFUSE")
        except TD.Refused as e:
            ok("absent day is not a clean day" in str(e),
               "FIX-7c an ABSENT day refuses through the lowest-level "
               "reader's own message, not this module's")
        # a directory that exists but holds no window files
        (root / "20300106").mkdir()
        try:
            scan_day("20300106", DAYS + ["20300106"], med_for(root), {},
                     raw_root=root)
            ok(False, "FIX-7d an empty day directory must REFUSE")
        except TD.Refused as e:
            ok("NO window files" in str(e),
               "FIX-7d a day directory with NO window files refuses")

    # --- FIX-8: a sweep whose every day refuses must not report 0 intervals.
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build(root, {}, days=DAYS[:2])
        try:
            sweep(DAYS[:2], raw_root=root)
            ok(False, "FIX-8 an all-refused sweep must REFUSE")
        except ScanRefused as e:
            ok("nothing could be judged" in str(e),
               f"FIX-8 a sweep in which every day refused REFUSES rather than "
               f"reporting a clean tape ({str(e)[:50]}...)")

    # --- FIX-9: absence is darkness, not health.
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build(root)
        base = _day_epoch(DAYS[-1])
        for k in range(10, 20):
            for c in COINS:
                (root / DAYS[-1] /
                 f"{c}-updown-5m-{base + k * WINDOW_S}.jsonl.gz").unlink()
        r = scan_day(DAYS[-1], DAYS, med_for(root), {}, raw_root=root)
        ok(r["n_intervals"] == 1 and r["intervals"][0]["n_windows"] == 10
           and r["intervals"][0]["worst_frac_of_reference"] == 0.0
           and r["n_windows_absent_for_all_coins"] == 10
           and r["n_windows_observed"] == NW - 10,
           f"FIX-9 ten windows ABSENT FOR EVERY COIN read as DARK, not as "
           f"absent -- the worst case must not read cleanest. Scanned "
           f"{r['n_windows_scanned']}, observed {r['n_windows_observed']}, "
           f"absent-for-all {r['n_windows_absent_for_all_coins']}")

    # --- FIX-10: the wiring scanner can FIRE (rule 15 on the checker itself).
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        (d / "innocent.py").write_text("import json\n# import "
                                       "da_dark_interval_scan is a comment\n")
        w = no_module_imports_this(d)
        ok(w["unwired"] is True and w["n_files_scanned"] == 1,
           "FIX-10 the wiring scanner ADMITS a tree with no importer, and a "
           "commented-out import is not an import")
        (d / "guilty.py").write_text("import da_dark_interval_scan as X\n")
        w = no_module_imports_this(d)
        ok(w["unwired"] is False and w["importers"][0]["file"] == "guilty.py",
           "FIX-10b and it FIRES on a PLANTED import -- the scanner has shown "
           "it can find one, so its zero on the real tree means something")

    # ------------------------------------------------------------------
    # REAL-TAPE TIER. NO NAMED SKIP EXISTS HERE. If the tape is absent these
    # FAIL BY NAME (DA20-R3's silent-shrink class is what a skip would be).
    # ------------------------------------------------------------------
    REAL = "20260901"
    ok(TD.RAW.is_dir() and (TD.RAW / REAL).is_dir(),
       f"REAL-0 PRECONDITION: the tape is readable at {TD.RAW} and carries "
       f"{REAL}. This suite ships NO skip for the real-tape controls -- a "
       f"named SKIP standing in for a positive control was ruled out, and a "
       f"silent shrink is worse. Set PM_DATA_ROOT to a tree holding "
       f"data/pm_5min/raw.")
    _all = TD.all_days()
    _g = TD.load_gaps()
    _m = V2.day_medians(_all)
    r01 = scan_day(REAL, _all, _m, _g)
    spans = [(i["start_hhmm"], i["end_hhmm"], i["n_windows"])
             for i in r01["intervals"]]
    ok(spans == [("00:00:00Z", "01:05:00Z", 13),
                 ("22:45:00Z", "23:35:00Z", 10)],
       f"REAL-1 POSITIVE CONTROL ON THE ESTABLISHED FACT: the two 2026-09-01 "
       f"intervals are found, boundary-for-boundary -- 00:00-01:05Z (13w) and "
       f"22:45-23:35Z (10w). Got {spans}")
    ok(all(i["n_coins_dark_min"] == 7 and i["n_coins_judged"] == 7
           for i in r01["intervals"]),
       "REAL-1b both are ALL SEVEN coins simultaneously, as recorded")

    # THE OTHER DIRECTION on the same real day: an equally long HEALTHY
    # stretch immediately after the first outage is not flagged.
    _lo = int(dt.datetime(2026, 9, 1, 1, 5, tzinfo=dt.timezone.utc).timestamp())
    _healthy = {_lo + k * WINDOW_S for k in range(13)}
    _flagged = set()
    for i in r01["intervals"]:
        t0 = dt.datetime.strptime(i["start_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc).timestamp()
        _flagged |= {int(t0) + k * WINDOW_S for k in range(i["n_windows"])}
    ok(not (_healthy & _flagged),
       "REAL-2 NEGATIVE CONTROL: the 13 windows from 01:05Z -- the SAME "
       "LENGTH as the first outage, on the same day, same coins, same "
       "instrument -- are ADMITTED. The detector distinguishes the interval "
       "from its neighbour, not merely from a different day")

    inv = r01["intervals"][0]
    ok(inv["n_gap_rows_intersecting"] == 1
       and inv["ledger_known_coin_seconds"] == 18.7
       and inv["coin_seconds_in_span"] == 27300
       and inv["ledger_known_share"] < 0.001,
       f"REAL-3 THE INVISIBILITY, COMPUTED: inside 09-01 00:00-01:05Z the gap "
       f"ledger knows {inv['ledger_known_coin_seconds']}s of "
       f"{inv['coin_seconds_in_span']} coin-seconds "
       f"({100*inv['ledger_known_share']:.3f}%) across "
       f"{inv['n_gap_rows_intersecting']} row(s) -- which is why every "
       f"duration bar passed through it")
    inv2 = r01["intervals"][1]
    ok(inv2["ledger_known_share"] < 0.05,
       f"REAL-3b and inside 22:45-23:35Z it knows "
       f"{inv2['ledger_known_coin_seconds']}s of "
       f"{inv2['coin_seconds_in_span']} "
       f"({100*inv2['ledger_known_share']:.2f}%)")

    ok(inv["worst_frac_of_reference"] < 0.10
       and inv["median_frac_of_reference"] < 0.03,
       f"REAL-4 the content level is the recorded one: worst coin "
       f"{100*inv['worst_frac_of_reference']:.2f}% of its point-in-time "
       f"reference, median {100*inv['median_frac_of_reference']:.2f}%")

    # WHAT DEPENDS ON THE CHOICE, driven rather than described.
    r05 = scan_day(REAL, _all, _m, _g, frac=0.05)
    s05 = [(i["start_hhmm"], i["n_windows"]) for i in r05["intervals"]]
    ok(("00:00:00Z", 13) not in s05,
       f"REAL-5 AT THE FROZEN 0.05 THE 65-MINUTE INTERVAL DISAPPEARS "
       f"(got {s05}) -- one coin at 5.5-6.0% of reference while six sit at "
       f"0.1-3% is enough to hide an all-coin blackout from "
       f"pm_tape_density.THIN_FRAC and from v2's V2_DARK_FRAC. A fact about "
       f"the threshold, not a proposal to move one")
    ok(TD.THIN_FRAC == 0.05 and V2.V2_DARK_FRAC == 0.05,
       "REAL-5b and those two frozen constants are read from their own "
       "modules, not restated here")

    _sw = sweep()
    ok(_sw["n_days_refused"] == 3
       and _sw["days_refused"] == ["20260819", "20260820", "20260821"],
       f"REAL-6 the first three days of the tape REFUSE for want of a "
       f"reference and are NAMED ({_sw['days_refused']}) -- they are not "
       f"counted as clean days")
    ok(_sw["n_days_judged"] + _sw["n_days_refused"] == _sw["n_days_requested"],
       f"REAL-6b judged + refused = requested "
       f"({_sw['n_days_judged']} + {_sw['n_days_refused']} = "
       f"{_sw['n_days_requested']}), so no day is silently missing")

    _w = no_module_imports_this()
    ok(_w["unwired"] is True,
       f"REAL-7 THIS ROUND IT GOVERNS NOTHING: no module under "
       f"{_w['scanned_dir']} imports it ({_w['n_files_scanned']} files "
       f"scanned, importers {_w['importers']}). Whether it ever governs is a "
       f"USER freeze, and this check is what stops that becoming true by "
       f"accident")
    ok(DISPOSITION == "REPORTED_NOT_GOVERNING",
       "REAL-7b and the disposition is declared in the module, matching "
       "pm_tape_density's under R-362")

    _cross = [g for g in _sw["spanning_events"] if g["crosses_day_boundary"]]
    ok(len(_cross) == 1
       and _cross[0]["start_utc"] == "2026-08-31T23:20:00Z"
       and _cross[0]["end_utc"] == "2026-09-01T01:05:00Z"
       and _cross[0]["n_windows"] == 21 and _cross[0]["span_s"] == 6300,
       f"REAL-8 THE 09-01 OUTAGE IS THE SECOND HALF OF A 105-MINUTE EVENT: "
       f"2026-08-31T23:20Z -> 2026-09-01T01:05Z, 21 windows, 6300 s, "
       f"unbroken across the UTC boundary. The record's '00:00-01:05Z (65 "
       f"min)' is the post-midnight remainder. Got {_cross}")

    # --- FIX-11: adjacency merging, driven in BOTH directions on inputs
    # that supply the ADJACENCY and never the grouping.
    def _stub(day, t0, t1, nw):
        return {"day": day, "start_utc": t0, "end_utc": t1,
                "start_hhmm": t0[11:], "end_hhmm": t1[11:], "n_windows": nw,
                "span_s": nw * WINDOW_S, "ledger_known_coin_seconds": 0.0,
                "coin_seconds_in_span": nw * WINDOW_S * 7,
                "n_gap_rows_intersecting": 0}
    _adj = merge_adjacent([
        _stub("20260831", "2026-08-31T23:20:00Z", "2026-09-01T00:00:00Z", 8),
        _stub("20260901", "2026-09-01T00:00:00Z", "2026-09-01T01:05:00Z", 13)])
    ok(len(_adj) == 1 and _adj[0]["n_windows"] == 21
       and _adj[0]["crosses_day_boundary"] is True
       and _adj[0]["days"] == ["20260831", "20260901"],
       f"FIX-11 two instances that touch END-TO-START across midnight merge "
       f"into ONE event ({_adj[0]['n_windows']}w, days {_adj[0]['days']})")
    _sep = merge_adjacent([
        _stub("20260831", "2026-08-31T23:20:00Z", "2026-08-31T23:55:00Z", 7),
        _stub("20260901", "2026-09-01T00:00:00Z", "2026-09-01T01:05:00Z", 13)])
    ok(len(_sep) == 2 and all(g["crosses_day_boundary"] is False
                              for g in _sep),
       f"FIX-11b and a FIVE-MINUTE separation does NOT merge (got "
       f"{len(_sep)} events) -- the predicate is contiguity, not same-ish "
       f"time on adjacent days")

    print(f"\nda_dark_interval_scan selftest: {checks} checks PASSED")
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: EXPECTED_CHECKS={EXPECTED_CHECKS} but {checks} ran. A "
              f"check that vanished must fail the suite, never shrink it.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
