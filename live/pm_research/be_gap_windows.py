#!/usr/bin/env python3
"""BE 141 -- THE DECLARED SPINE: one gap-window artifact per population day.

`de_revaluation_emit` keys its per-window table on the day's gap-bearing
window list under the FIXED era. 09-07's was produced by hand at BE 137; every
later day needs the SAME SHAPE so the emit reads it without a code change and
REV can cross-check it against the ledger.

WHAT "GAP-BEARING" MEANS HERE, and it is the narrow reading on purpose: a
supplied window for which `flow_intensity.gaps_by_slug(era)` returns a
non-empty interval list. That is the source the FRAGMENT consumes, so it is
what the replay actually sees. A gap the ledger records and the replay does
not carry is NOT in this list -- it is reported separately under
`GAP_RECORDED_NOT_SEEN_BY_REPLAY`, because a gap the replay never sees cannot
contribute to delta-D and must not sit in a table that has to SUM to it.

THE POSITIVE CONTROL IS 09-07 ITSELF. The selftest regenerates 09-07 and
requires every generic field to equal the landed BE 137 artifact -- 27 ids,
the missing interior window, the boundary flag. An instrument that produced a
DIFFERENT spine for the one day already cross-checked (REV: 27/27, 35/35)
would be wrong, and this is the cheapest way to find that out.

Exit codes (75 is the wrapper's and is not among them, rule 20):
  0  written   1  a control FAILED   2  usage   3  input refused
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

NOT_SEEN = "GAP_RECORDED_NOT_SEEN_BY_REPLAY"
DERIVED = Path("data/pm_5min/derived")
EXIT_CODES = {0: "WRITTEN", 1: "CONTROL_FAILED", 2: "USAGE", 3: "INPUT_REFUSED"}
assert 75 not in EXIT_CODES, "75 is the wrapper's conflict code (rule 20)"

# The fields the emit keys on. Named here so a shape change is a diff, not a
# surprise at read time.
GENERIC_FIELDS = (
    "day", "era", "n_windows", "gap_bearing_window_starts",
    "gap_bearing_window_starts_utc", "n_gap_bearing",
    "missing_interior_window", "missing_interior_window_utc",
    "first_window", "last_window", "boundary_window_2355_present",
)


class SpineRefused(RuntimeError):
    """The spine cannot be produced, so none is written."""


def _utc(t) -> str:
    return datetime.datetime.fromtimestamp(int(t), datetime.UTC).strftime(
        "%H:%M:%SZ")


def _utcfull(t) -> str:
    return datetime.datetime.fromtimestamp(int(t), datetime.UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def spine(day: str, coin: str = "btc") -> dict:
    import flow_intensity as fi
    import be_era_for_day as EFD
    import be_gate1_fragment as FR
    import be_gap_census as CEN

    pop = FR.population(day, coin)
    want = sorted(pop["slugs"])
    if not want:
        raise SpineRefused(f"REFUSED: {day} supplies no slugs")
    era = EFD.resolve(fi, pop["day"], want)["era"]
    gaps = fi.gaps_by_slug(era)

    rows = []
    for s in want:
        iv = gaps.get(s) or []
        if iv:
            rows.append((int(s.rsplit("-", 1)[1]), s, iv))
    rows.sort()

    starts = sorted(int(s.rsplit("-", 1)[1]) for s in want)
    first, last = starts[0], starts[-1]
    expected = list(range(first, last + 300, 300))
    missing = sorted(set(expected) - set(starts))

    # the census side: ledger gaps the replay does NOT carry
    cen = CEN.census(day, coin)
    not_seen = cen[NOT_SEEN]

    return {
        "protocol": "BE_GAP_WINDOWS_V1",
        "day": day, "era": era,
        "n_windows": pop["n_windows"],
        "gap_bearing_window_starts": [r[0] for r in rows],
        "gap_bearing_window_starts_utc": [_utc(r[0]) for r in rows],
        "n_gap_bearing": len(rows),
        "missing_interior_window": missing,
        "missing_interior_window_utc": [_utc(m) for m in missing],
        "first_window": first, "last_window": last,
        "boundary_window_2355_present": last in starts,
        "gap_intervals_by_window": {
            str(r[0]): [[round(float(a), 3), round(float(b), 3)]
                        for a, b in r[2]] for r in rows},
        "WHAT_GAP_BEARING_MEANS": (
            "a SUPPLIED window for which flow_intensity.gaps_by_slug(era) "
            "returns a non-empty interval list -- the source the fragment "
            "consumes, so it is what the replay sees"),
        NOT_SEEN: {
            "n": not_seen["n"],
            "rows": not_seen["rows"],
            "windows_that_gain_a_gap_under_wall_clock":
                not_seen["windows_that_gain_a_gap_under_wall_clock"],
            "why_they_are_not_in_the_spine": (
                "the collector recorded these gaps and the replay does not "
                "carry them: each row is stamped to a window that does not "
                "contain the gap's own instant, so its clamped interval is "
                "empty and it is dropped -- and never re-attributed. A gap "
                "the replay never sees cannot contribute to delta-D, so it "
                "must NOT sit in a table that has to SUM to delta-D. Counted "
                "here so it is an accounted exclusion (rule 4)."),
        },
        "as_of_utc": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "first_window_utc": _utcfull(first), "last_window_utc": _utcfull(last),
    }


def out_path(day: str) -> Path:
    return DERIVED / f"be137_gap_windows_{day}.json"


def falsify() -> int:
    """Rule 15/16 -- and the positive control is a REAL day, not a fixture."""
    checks = []

    def note(n, ok):
        checks.append((n, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")

    landed_path = out_path("20260907")
    if not landed_path.is_file():
        note("the landed BE 137 artifact is present to check against", False)
        print(json.dumps({"falsifier": "be_gap_windows", "n": 1,
                          "n_failed": 1, "failed": ["no 09-07 artifact"]}))
        return 1
    landed = json.loads(landed_path.read_text())
    made = spine("20260907")

    note("regenerating 09-07 reproduces the landed spine EXACTLY, field for "
         "field", all(made[f] == landed[f] for f in GENERIC_FIELDS))
    for f in GENERIC_FIELDS:
        if made[f] != landed[f]:
            print(f"        DIFFERS at {f}: made={str(made[f])[:80]} "
                  f"landed={str(landed[f])[:80]}")
    note("and it is the 27 REV cross-checked, not a recount",
         made["n_gap_bearing"] == 27 == len(made["gap_bearing_window_starts"]))
    note("the spine EXCLUDES the 15:55 window the replay never sees",
         1788796500 not in made["gap_bearing_window_starts"])
    note("and that window is REPORTED under the not-seen status instead",
         1788796500 in made[NOT_SEEN][
             "windows_that_gain_a_gap_under_wall_clock"])
    note("the missing interior window is carried, not silently dropped",
         made["missing_interior_window"] == [1788807300])
    note("every gap-bearing window carries its intervals",
         set(made["gap_intervals_by_window"])
         == {str(w) for w in made["gap_bearing_window_starts"]}
         and all(v for v in made["gap_intervals_by_window"].values()))
    note("every spine id is a SUPPLIED window (first <= id <= last, on grid)",
         all(made["first_window"] <= w <= made["last_window"]
             and (w - made["first_window"]) % 300 == 0
             for w in made["gap_bearing_window_starts"]))
    note("a day that supplies nothing REFUSES rather than writing an empty "
         "spine",
         _refuses_on_empty())

    bad = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_gap_windows", "n": len(checks),
                      "n_failed": len(bad), "failed": bad}))
    return 1 if bad else 0


def _refuses_on_empty() -> bool:
    import be_gate1_fragment as FR
    real = FR.population
    try:
        FR.population = lambda day, coin="btc": {
            "slugs": [], "n_windows": 0, "day": day}
        try:
            spine("20260907")
            return False
        except SpineRefused:
            return True
        except Exception:
            return False
    finally:
        FR.population = real


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return falsify()
    days = [a for a in argv if a.isdigit()]
    if not days:
        print("usage: be_gap_windows.py <YYYYMMDD> [...] | --selftest")
        return 2
    for d in days:
        try:
            s = spine(d)
        except SpineRefused as e:
            print(json.dumps({"refused": str(e)}, indent=1))
            return 3
        except Exception as e:
            print(json.dumps({"refused": f"{type(e).__name__}: {e}"}, indent=1))
            return 3
        p = out_path(d)
        if p.exists():
            print(json.dumps({"refused": f"REFUSED: {p} already exists; a "
                              f"spine is never overwritten (rule 13)"}))
            return 3
        p.write_text(json.dumps(s, indent=1) + "\n")
        print(f"{d} era={s['era']} supplied={s['n_windows']} "
              f"gap_bearing={s['n_gap_bearing']} "
              f"not_seen={s[NOT_SEEN]['n']} "
              f"missing_interior={s['missing_interior_window']} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
