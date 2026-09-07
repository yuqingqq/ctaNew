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


#: The pins' OWN field, quoted rather than paraphrased: a content
#: disagreement fires THIS, by name, because the pins already declare what a
#: mismatch means and an instrument should not invent a second word for it.
VOIDS_FIELD = "the_read_voids_on_mismatch"

#: Every per-day verdict `verify` can return. Three are answers; two are
#: refusals. `NO_SOURCE` is a refusal and NOT a pass -- a day nothing can
#: re-derive is exactly the case a silent skip would hide (R-649).
VERDICTS = ("VERIFIED_FROM_RECEIPT", "VERIFIED_FROM_FILE",
            "CORROBORATED_ABSENT", "MISMATCH", "NO_SOURCE")


class PinsVerificationFailed(RuntimeError):
    """A named refusal: a landed pin does not survive re-derivation."""


def producing_receipt(pin: dict, day: str) -> Path:
    """The receipt of the run that WROTE this feed.

    Found from the pin's OWN path -- the feed and its receipt are written
    side by side by `be_forward_day` -- and not by searching for a receipt
    that agrees. LIMITATION STATED: the pin therefore chooses which receipt
    answers. What that cannot fake is the comparison: the receipt names the
    feed's path itself, so a pin pointing at the wrong run surfaces as a
    missing receipt or a PATH mismatch rather than passing quietly. (For
    20260906 two run directories exist -- `_be87`'s refused attempt and
    `_be88`'s -- and the pin's path selects be88, the run that produced the
    bytes.)
    """
    return Path(pin["path"]).parent / f"be_forward_day_receipt_{day}.json"


def rederive_day(pin: dict, day: str) -> dict:
    """Re-derive one day's {path, sha256, bytes} WITHOUT reading the feed.

    THE ORDER OF SOURCES, and both are repo-produced:

      1. THE PRODUCING RECEIPT. `be_forward_day.py` records the feed's path,
         sha256 and byte count in its own `feed` block at emit. That is the
         producer attesting its own output and is preferred wherever it
         exists.
      2. THE FILE'S BYTES. sha256 and `st_size`, nothing else -- the file is
         never parsed, no line is read or counted, and this module never
         touches the reader or `--open`.

    Which one answered is REPORTED per day, because "verified" from a
    producer's attestation and "verified" by re-hashing are not the same
    claim and a reader must be able to tell them apart.
    """
    out = {"day": day, "source": None, "path": None, "sha256": None,
           "bytes": None, "receipt": None, "receipt_exists": False,
           "file_exists": False}
    rcp = producing_receipt(pin, day)
    out["receipt"] = str(rcp)
    out["receipt_exists"] = rcp.exists()
    feed = Path(pin["path"])
    out["file_exists"] = feed.exists()
    if rcp.exists():
        try:
            doc = json.loads(rcp.read_text())
        except (OSError, json.JSONDecodeError) as e:
            doc = None
            out["receipt_unreadable"] = f"{type(e).__name__}: {e}"
        blk = (doc or {}).get("feed")
        if isinstance(blk, dict) and isinstance(blk.get("sha256"), str):
            out.update({"source": "PRODUCING_RECEIPT",
                        "path": blk.get("path"),
                        "sha256": blk.get("sha256"),
                        "bytes": blk.get("bytes")})
            return out
    if feed.exists():
        st = feed.stat()
        out.update({"source": "THE_FILE_BYTES", "path": str(feed),
                    "sha256": _sha(feed), "bytes": st.st_size})
        return out
    out["source"] = "NEITHER"
    return out


def verify(version=None, declarations=None) -> dict:
    """Re-derive a LANDED pins version's per_day content and compare it.

    A byte-identical re-derive turns a scratch-built pin into a repo-verified
    one WITHOUT touching it: this writes nothing, supersedes nothing and is
    not a version (rule 13 / R-711 untouched). What it can and cannot settle
    is stated in the result rather than left to the reader: it establishes
    CONTENT -- the file at this path had this digest and this size -- and it
    does NOT establish the pins' TIMING claim (`pinned_before`), which no
    digest can carry (REV 90 §B4).
    """
    d = Path(declarations) if declarations else DECL
    if version is None:
        head = DC.resolve_head(d, FAMILY)
        path, name = Path(head["path"]), head["name"]
    else:
        name = (version if str(version).endswith(".json")
                else f"{FAMILY}_v{version}.json")
        path = d / name
        if not path.exists():
            raise PinsVerificationFailed(
                f"VERSION_ABSENT: no {name} under {d}. A check that depends "
                f"on a declaration FAILS when it is gone (R-649).")
    doc = json.loads(path.read_text())
    per_day = doc.get("per_day") or {}
    days = []
    for day in sorted(per_day):
        pin = per_day[day]
        got = rederive_day(pin, day)
        row = {"day": day, "pin_says_exists": bool(pin.get("exists")),
               "source": got["source"], "receipt": got["receipt"],
               "receipt_exists": got["receipt_exists"],
               "file_exists": got["file_exists"], "fields": {}}
        if not pin.get("exists"):
            # THE ABSENT PINS ARE CHECKED, NOT SKIPPED. `exists: false` is a
            # claim -- that no feed was produced -- and it is falsified by
            # either source producing one.
            if got["source"] == "NEITHER":
                row["verdict"] = "CORROBORATED_ABSENT"
                row["detail"] = ("the producing receipt carries no `feed` "
                                 "block and no file is at the pinned path: "
                                 "both sources agree there is nothing")
            else:
                row["verdict"] = "MISMATCH"
                row["detail"] = (f"the pin says `exists: false` but "
                                 f"{got['source']} produced a feed "
                                 f"({str(got['sha256'])[:16]}…, "
                                 f"{got['bytes']} bytes)")
            days.append(row)
            continue
        if got["source"] == "NEITHER":
            row["verdict"] = "NO_SOURCE"
            row["detail"] = (f"nothing can re-derive this day: no `feed` "
                             f"block in {Path(got['receipt']).name} "
                             f"(exists: {got['receipt_exists']}) and no file "
                             f"at {pin['path']}. NOT a pass")
            days.append(row)
            continue
        bad = []
        for f in ("path", "sha256", "bytes"):
            want, have = pin.get(f), got.get(f)
            row["fields"][f] = {"landed": want, "rederived": have,
                                "equal": want == have}
            if want != have:
                bad.append(f)
        row["verdict"] = "MISMATCH" if bad else (
            "VERIFIED_FROM_RECEIPT" if got["source"] == "PRODUCING_RECEIPT"
            else "VERIFIED_FROM_FILE")
        if bad:
            row["detail"] = (f"{', '.join(bad)} disagree(s) with "
                             f"{got['source']}")
        days.append(row)
    bad_days = [r for r in days if r["verdict"] in ("MISMATCH", "NO_SOURCE")]
    res = {
        "verified": name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_days": len(days),
        "by_verdict": {v: sum(1 for r in days if r["verdict"] == v)
                       for v in VERDICTS
                       if any(r["verdict"] == v for r in days)},
        "days": days,
        "wrote_nothing": True,
        "what_this_establishes": "CONTENT: each day's feed path, sha256 and "
                                 "byte count, re-derived from the producing "
                                 "receipt or the file's bytes and compared "
                                 "field by field to the landed pin.",
        "what_this_does_NOT_establish": "TIMING. The pins claim `pinned_"
                                        "before: any read of the feed`; no "
                                        "digest can carry that, and this "
                                        "check does not pretend to "
                                        "(REV 90 §B4).",
        "never_read": "no feed was parsed; no line, row or field was read or "
                      "counted; the reader was not invoked and nothing was "
                      "--open'ed.",
    }
    if bad_days:
        res["ok"] = False
        res[VOIDS_FIELD] = doc.get(VOIDS_FIELD)
        raise PinsVerificationFailed(
            f"PINS_CONTENT_DISAGREES: {name} -- "
            f"{[(r['day'], r['verdict']) for r in bad_days]}. The pins' own "
            f"`{VOIDS_FIELD}` is {doc.get(VOIDS_FIELD)!r}, and this fires it "
            f"by name. {bad_days[0]['detail']}", res)
    res["ok"] = True
    return res


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
    # ---- `--verify`: FIVE PER-DAY VERDICTS, DRIVEN ON SCRATCH ----------
    # No feed is parsed anywhere below: the fixtures write tiny files and the
    # module hashes their BYTES. The real reader is never invoked and nothing
    # is --open'ed.
    vd = Path(tempfile.mkdtemp(prefix="pins_verify_"))
    run = vd / "run"
    run.mkdir()

    def _fixture(n, per_day, voids=True):
        (vd / f"{FAMILY}_v{n}.json").write_text(json.dumps(
            {"protocol": PROTOCOL, "supersedes": None, "per_day": per_day,
             "the_read_voids_on_mismatch": voids}, indent=1,
            sort_keys=True) + "\n")

    feed_a = run / "be_forward_day_SEALED_feed_20990101.jsonl"
    feed_a.write_text('{"row": 1}\n')
    sha_a, bytes_a = _sha(feed_a), feed_a.stat().st_size
    (run / "be_forward_day_receipt_20990101.json").write_text(json.dumps(
        {"feed": {"path": str(feed_a), "sha256": sha_a, "bytes": bytes_a}}))
    good = {"20990101": {"exists": True, "path": str(feed_a),
                         "sha256": sha_a, "bytes": bytes_a}}
    _fixture(1, good)
    r = verify(1, vd)
    ok(r["ok"] is True and r["days"][0]["verdict"] == "VERIFIED_FROM_RECEIPT"
       and all(r["days"][0]["fields"][f]["equal"] for f in
               ("path", "sha256", "bytes")) and r["wrote_nothing"] is True,
       f"POSITIVE CONTROL: a pin whose three fields agree with the PRODUCING "
       f"RECEIPT verifies ({r['days'][0]['verdict']}) and the check writes "
       f"nothing -- a verification, not a version (rule 13 untouched). A "
       f"control shown only to refuse has not been shown to admit")

    # (a) ONE MOVED DIGEST -- refuses NAMING THE DAY.
    moved = {"20990101": dict(good["20990101"], sha256="e" * 64)}
    _fixture(2, moved)
    try:
        verify(2, vd); ra = "NOT REFUSED"; res_a = {}
    except PinsVerificationFailed as e:
        ra, res_a = str(e.args[0]), (e.args[1] if len(e.args) > 1 else {})
    ok(ra.startswith("PINS_CONTENT_DISAGREES:") and "20990101" in ra
       and res_a.get("days", [{}])[0].get("verdict") == "MISMATCH"
       and res_a["days"][0]["fields"]["sha256"]["equal"] is False
       and res_a["days"][0]["fields"]["bytes"]["equal"] is True
       and res_a.get(VOIDS_FIELD) is True,
       f"KNOWN-BAD A -- ONE MOVED DIGEST: refused, NAMING THE DAY, and the "
       f"pins' OWN `{VOIDS_FIELD}` ({res_a.get(VOIDS_FIELD)!r}) is fired BY "
       f"NAME rather than an invented second word: {ra[:150]!r}. The report "
       f"is field by field -- sha256 unequal, bytes and path EQUAL -- so a "
       f"reader is told WHICH field moved, not merely that something did")

    # (b) NO RECEIPT AND NO FILE -- refuses, and does NOT read as absent.
    _fixture(3, {"20990102": {"exists": True,
                                 "path": str(run / "gone_20990102.jsonl"),
                                 "sha256": "a" * 64, "bytes": 7}})
    try:
        verify(3, vd); rb = "NOT REFUSED"; res_b = {}
    except PinsVerificationFailed as e:
        rb, res_b = str(e.args[0]), (e.args[1] if len(e.args) > 1 else {})
    ok("NO_SOURCE" in rb and res_b["days"][0]["verdict"] == "NO_SOURCE"
       and res_b["days"][0]["receipt_exists"] is False
       and res_b["days"][0]["file_exists"] is False
       and "NOT a pass" in res_b["days"][0]["detail"],
       f"KNOWN-BAD B -- NOTHING TO RE-DERIVE FROM: no producing receipt and "
       f"no file, and the day is `NO_SOURCE` and REFUSED, never silently "
       f"passed (R-649: a check that cannot run is not a check that ran). "
       f"The distinction that matters: this is a pin claiming a feed EXISTS "
       f"with nothing to confirm it -- not the same as a pin claiming none")

    # (c) THE FILE FALLBACK, exercised -- a receipt with no `feed` block.
    feed_c = run / "be_forward_day_SEALED_feed_20990103.jsonl"
    feed_c.write_text('{"row": 3}\n{"row": 4}\n')
    (run / "be_forward_day_receipt_20990103.json").write_text(
        json.dumps({"day": "20990103", "outcome": "no feed block here"}))
    _fixture(4, {"20990103": {"exists": True, "path": str(feed_c),
                                 "sha256": _sha(feed_c),
                                 "bytes": feed_c.stat().st_size}})
    rc = verify(4, vd)
    ok(rc["ok"] is True
       and rc["days"][0]["verdict"] == "VERIFIED_FROM_FILE"
       and rc["days"][0]["receipt_exists"] is True
       and rc["days"][0]["source"] == "THE_FILE_BYTES",
       f"THE SECOND SOURCE IS REACHABLE AND REPORTED: a receipt that exists "
       f"but carries no `feed` block falls through to the FILE'S BYTES and "
       f"the verdict SAYS which answered ({rc['days'][0]['verdict']}). "
       f"`verified from the producer's attestation` and `verified by "
       f"re-hashing` are different claims and the row distinguishes them")

    # (d) AN `exists: false` PIN IS CHECKED, NOT SKIPPED -- both directions.
    _fixture(5, {"20990104": {"exists": False,
                                 "path": str(run / "never_20990104.jsonl")}})
    rd = verify(5, vd)
    _fixture(6, {"20990101": {"exists": False, "path": str(feed_a)}})
    try:
        verify(6, vd); re_ = "NOT REFUSED"; res_e = {}
    except PinsVerificationFailed as e:
        re_, res_e = str(e.args[0]), (e.args[1] if len(e.args) > 1 else {})
    ok(rd["ok"] is True
       and rd["days"][0]["verdict"] == "CORROBORATED_ABSENT"
       and res_e.get("days", [{}])[0].get("verdict") == "MISMATCH"
       and "exists: false" in res_e["days"][0]["detail"],
       f"AN ABSENT PIN IS A CLAIM AND IS FALSIFIABLE: with neither source "
       f"producing a feed it is {rd['days'][0]['verdict']}; with a receipt "
       f"that DOES produce one the same pin is MISMATCH and refused. "
       f"Skipping `exists: false` days would have made two of the five "
       f"landed pins unexaminable")

    # ---- THE REAL v1 AND v2, DRIVEN HERE AND NOT ONLY IN A REPORT -------
    # REV 90 §B4 asked whether a scratch-built precondition can be trusted.
    # Its content can be RE-DERIVED, and that is what this cell asserts --
    # standing, so it fires the day either version stops re-deriving rather
    # than resting on one round's run. It does NOT assert the timing claim,
    # which no digest carries.
    real = {}
    for n in (1, 2):
        try:
            real[n] = verify(n)
        except PinsVerificationFailed as e:
            real[n] = {"ok": False, "by_verdict": str(e.args[0])[:160]}
    ok(real[1].get("ok") is True and real[2].get("ok") is True
       and real[1]["by_verdict"] == {"CORROBORATED_ABSENT": 2,
                                     "VERIFIED_FROM_RECEIPT": 3}
       and real[2]["by_verdict"] == {"CORROBORATED_ABSENT": 2,
                                     "VERIFIED_FROM_RECEIPT": 4},
       f"THE TWO LANDED VERSIONS RE-DERIVE BYTE FOR BYTE: v1 "
       f"{real[1].get('by_verdict')}, v2 {real[2].get('by_verdict')} -- every "
       f"present day answered by its PRODUCING RECEIPT (no file needed "
       f"hashing) and every absent day corroborated by both sources. That "
       f"converts v1 and v2 from scratch-built to repo-verified in CONTENT "
       f"without either file being touched")

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
    ap.add_argument("--verify", nargs="?", const="HEAD",
                    help="re-derive a LANDED version's per_day content and "
                         "compare it field by field; writes nothing")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.verify:
        try:
            r = verify(None if a.verify == "HEAD" else a.verify)
        except PinsVerificationFailed as e:
            print(json.dumps(e.args[1] if len(e.args) > 1 else {"error": str(e)},
                             indent=1, sort_keys=True))
            print(f"\nREFUSED: {e.args[0]}")
            return 1
        print(json.dumps(r, indent=1, sort_keys=True))
        return 0
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
