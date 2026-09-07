"""THE ONE EMITTER of the `be_race_read_feed_pins` chain (REV 89 §8 row 7).

WHY IT EXISTS AT ALL. `grep -rln "BE_RACE_READ_FEED_PINS\\|feed_pins"` over
`live/` and `scripts/` at BE 92 found three modules and every one of them
READS: `be_race_reader.pins()`, `da_race_read_verify`, and
`be_race_read_declaration_v3`. **v1 and v2 were written by scratch scripts.**
The pins are the second read's whole-set precondition -- the artifact the
read refuses on -- and CLAUDE.md rule 12 already names a scratch-dir builder
as the thing that voided a freeze. So this module is the producer that should
have existed before v1.

WHAT REV 89 ROUTED. v2 carries a top-level `all_five_present` whose NAME
states a count the file no longer matches: v1 pinned five days, v2 pins six,
and a reader who takes the name at face value is counting the wrong
population. v2 is LANDED and IMMUTABLE (rule 20) -- it is not edited. The
correction is in band, in the next version:

  * `all_pinned_days_present` -- the field the name should always have been,
    DERIVED over this version's own `per_day`, whatever it holds;
  * `all_five_present` -- KEPT, and kept meaning exactly what v1 meant: all
    five of the FIRST read's days present. It is derived over those five days
    by name, so the field and its name agree again. It is kept rather than
    dropped because `da_race_read_verify` reads it, and a reader doing
    `bool(d.get("all_five_present"))` against a version that dropped it gets
    a silently defaulted `False` -- an absent field reading as a measured
    one, which is worse than a badly named present one;
  * `all_five_present_is_v1s_name` -- says the above in the artifact, so the
    next reader does not have to find this docstring.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import declaration_chain as DC                                 # noqa: E402

FAMILY = "be_race_read_feed_pins"
DECL = HERE / "declarations"
PROTOCOL = "BE_RACE_READ_FEED_PINS_V2"

#: The FIRST read's population, by name. `all_five_present` is a statement
#: about THESE days and no others -- which is what makes the inherited name
#: true again rather than merely tolerated.
FIRST_READ_DAYS = ("20260901", "20260902", "20260903", "20260904", "20260905")


class PinsRefused(RuntimeError):
    """A named refusal."""


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def derived_presence(per_day: dict) -> dict:
    """The three presence fields, every one COMPUTED from `per_day`.

    Never typed, and never inherited from the version being superseded: a
    presence field copied forward is how v2's said `five` about six days.
    """
    missing_first = [d for d in FIRST_READ_DAYS
                     if not per_day.get(d, {}).get("exists")]
    return {
        "all_pinned_days_present": all(
            bool(v.get("exists")) for v in per_day.values()),
        "all_five_present": not missing_first,
        "all_five_present_is_v1s_name": (
            "INHERITED NAME, KEPT TRUE. It is v1's field and it counts the "
            f"FIRST read's five days {list(FIRST_READ_DAYS)} -- not this "
            f"file's {len(per_day)}. v2 carried the same name over six days, "
            "which REV 89 §8 row 7 routed as a count the file no longer "
            "matched; v2 is landed and immutable (rule 20), so this is the "
            "in-band correction. `all_pinned_days_present` is the field to "
            "read for THIS version's population. Kept rather than dropped "
            "because da_race_read_verify reads it and an absent field would "
            "read as a measured False."),
    }


def build_next(day: str, feed_path, scores_path=None, produced_by=None,
               declarations=None) -> dict:
    """The payload for the NEXT pins version: the head's days plus `day`.

    The head is resolved through the shared chain (never a filename), every
    inherited day is carried BYTE-IDENTICAL, and the new day's entry is the
    feed's own {path, sha256, bytes} -- the file is hashed here and nothing
    inside it is read, counted or opened.
    """
    d = Path(declarations) if declarations else DECL
    head = DC.resolve_head(d, FAMILY)
    prev = json.loads(Path(head["path"]).read_text())
    per_day = dict(prev.get("per_day") or {})
    if day in per_day and per_day[day].get("exists"):
        raise PinsRefused(
            f"DAY_ALREADY_PINNED: {day} is already present in {head['name']} "
            f"with a digest. A landed pin is immutable (rule 20); a day is "
            f"pinned once, at its close.")
    feed = Path(feed_path)
    if not feed.exists():
        raise PinsRefused(
            f"FEED_ABSENT: no sealed feed at {feed}. The pin is the FEED's "
            f"bytes; there is nothing to pin and the day is not marked "
            f"present on a promise.")
    entry = {"exists": True, "path": str(feed), "sha256": _sha(feed),
             "bytes": feed.stat().st_size,
             "mtime_utc": dt.datetime.fromtimestamp(
                 feed.stat().st_mtime, dt.timezone.utc).strftime(
                     "%Y-%m-%dT%H:%M:%SZ")}
    if scores_path:
        entry["v2_scores_path"] = str(scores_path)
        entry["v2_scores_sha256"] = _sha(Path(scores_path))
    per_day[day] = entry
    payload = {
        "protocol": PROTOCOL,
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "per_day": per_day,
        "method": prev.get("method"),
        "pinned_before": prev.get("pinned_before"),
        "pinned_by": produced_by or prev.get("pinned_by"),
        "produced_by": prev.get("produced_by"),
        "scores_untouched": prev.get("scores_untouched"),
        "the_first_reads_days": prev.get("the_first_reads_days"),
        "the_second_reads_days": prev.get("the_second_reads_days"),
        "the_read_voids_on_mismatch": True,
        "supersedes": {"path": str(head["path"]), "sha256": head["sha256"],
                       "rule": "13 / R-608 -- vN+1 by the {path, sha256} "
                               "PAIR; the superseded version is NOT edited"},
        "emitted_by": "live/pm_research/be_race_feed_pins.py -- the chain's "
                      "ONE emitter. v1 and v2 were written by scratch "
                      "scripts (REV 89 §3.2's class, one artifact along).",
    }
    payload.update(derived_presence(per_day))
    return payload


def write_next(day, feed_path, scores_path=None, produced_by=None,
               declarations=None) -> dict:
    d = Path(declarations) if declarations else DECL
    head = DC.resolve_head(d, FAMILY)
    payload = build_next(day, feed_path, scores_path, produced_by, d)
    return DC.write_next_version(d, FAMILY, payload, head["pair"])


def selftest() -> int:
    import tempfile
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    # ---- THE PRESENCE FIELDS ARE DERIVED, AND BOTH DIRECTIONS DRIVEN ----
    five_ok = {d: {"exists": True} for d in FIRST_READ_DAYS}
    p1 = derived_presence(dict(five_ok, **{"20260906": {"exists": True}}))
    p2 = derived_presence(dict(five_ok, **{"20260906": {"exists": False}}))
    p3 = derived_presence(dict(five_ok, **{"20260901": {"exists": False},
                                           "20260906": {"exists": True}}))
    ok(p1["all_five_present"] is True and p1["all_pinned_days_present"] is True
       and p2["all_five_present"] is True
       and p2["all_pinned_days_present"] is False
       and p3["all_five_present"] is False
       and p3["all_pinned_days_present"] is False,
       f"THE TWO FIELDS ARE DIFFERENT QUESTIONS AND THE FIXTURE SEPARATES "
       f"THEM: with all five first-read days present and a SIXTH absent, "
       f"all_five_present={p2['all_five_present']} while "
       f"all_pinned_days_present={p2['all_pinned_days_present']} -- the case "
       f"v2's single field could not express, and the reason its name stated "
       f"a count the file no longer matched")
    ok(derived_presence(json.loads(
        (DECL / f"{FAMILY}_v2.json").read_text())["per_day"]
    )["all_five_present"] is False,
       "AND ON v2's OWN per_day THE DERIVED VALUE REPRODUCES v2's LANDED "
       "`false` -- so the correction is to the NAME's scope and the emitter's "
       "derivation, not to the answer v2 gave. v2 is not edited (rule 20)")
    ok("INHERITED NAME" in p1["all_five_present_is_v1s_name"]
       and "all_pinned_days_present" in p1["all_five_present_is_v1s_name"],
       "and the artifact SAYS SO ITSELF: the note names v1 as the source of "
       "the name and points the reader at the field for this version's "
       "population -- a reader with only the file has the answer, without "
       "this docstring")

    # ---- THE EMITTER: PRESENCE IS NEVER INHERITED -----------------------
    d = Path(tempfile.mkdtemp(prefix="pins_emit_"))
    feed = d / "feed_20990102.jsonl"
    feed.write_text('{"a": 1}\n')
    v1 = {"protocol": PROTOCOL, "supersedes": None,
          "per_day": {"20990101": {"exists": True, "path": "x",
                                   "sha256": "0" * 64}},
          # A STALE PRESENCE FIELD IN THE PARENT, deliberately WRONG:
          "all_five_present": True, "all_pinned_days_present": True}
    (d / f"{FAMILY}_v1.json").write_text(json.dumps(v1, indent=1,
                                                    sort_keys=True) + "\n")
    nxt = build_next("20990102", feed, declarations=d)
    ok(nxt["all_five_present"] is False
       and nxt["all_pinned_days_present"] is True
       and nxt["per_day"]["20990102"]["sha256"] == _sha(feed)
       and nxt["per_day"]["20990102"]["bytes"] == 9,
       f"KNOWN-BAD: the parent carries `all_five_present: true` and the "
       f"emitter DOES NOT INHERIT IT -- it recomputes "
       f"{nxt['all_five_present']} from the days actually pinned (none of "
       f"the first read's five are), while all_pinned_days_present is "
       f"{nxt['all_pinned_days_present']} over this file's own two. Copying "
       f"the parent's value forward is exactly how the field went stale")
    try:
        build_next("20990101", feed, declarations=d); r1 = "NOT REFUSED"
    except PinsRefused as e:
        r1 = str(e).split(":")[0]
    ok(r1 == "DAY_ALREADY_PINNED",
       f"KNOWN-BAD: re-pinning a day that already carries a digest is "
       f"REFUSED BY NAME ({r1}) -- a landed pin is immutable and a day is "
       f"pinned once, at its close")
    try:
        build_next("20990103", d / "nope.jsonl", declarations=d); r2 = "NOT REFUSED"
    except PinsRefused as e:
        r2 = str(e).split(":")[0]
    ok(r2 == "FEED_ABSENT",
       f"KNOWN-BAD: an absent feed is REFUSED BY NAME ({r2}) rather than "
       f"marking the day present on a promise -- the pin IS the feed's bytes")
    ok(nxt["supersedes"]["sha256"] == DC.resolve_head(d, FAMILY)["sha256"]
       and Path(nxt["supersedes"]["path"]).name == f"{FAMILY}_v1.json",
       "POSITIVE CONTROL: the head is resolved through the shared chain and "
       "superseded BY THE PAIR, never by a filename -- and the write itself "
       "goes through declaration_chain's compare-and-swap")
    # REV 84 §3.2 -- ONE IMPLEMENTATION, N DETECTORS. This module imports
    # `declaration_chain`, so it RUNS that module's own falsifier as a
    # subprocess cell: a regression there fails every importer at once, and
    # no importer re-implements the logic. (REV 89 §8 row 8 routed four
    # importers that ship no such cell; a new one is not a fifth.)
    import be_rule22 as _R22
    _sf = _R22.shared_falsifier()
    ok(_sf["ok"],
       f"REV 84 §3.2: this battery RUNS `declaration_chain.py --falsify` as "
       f"a subprocess -> rc {_sf['rc']}, {_sf['summary']!r}")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--day")
    ap.add_argument("--feed")
    ap.add_argument("--scores")
    ap.add_argument("--produced-by")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.day and a.feed):
        ap.print_help()
        return 2
    if a.dry_run:
        print(json.dumps(build_next(a.day, a.feed, a.scores, a.produced_by),
                         indent=1, sort_keys=True))
        return 0
    print(json.dumps(write_next(a.day, a.feed, a.scores, a.produced_by),
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
