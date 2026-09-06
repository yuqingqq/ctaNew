"""THE MOVED-SET, COMPUTED — the mechanism A-2 asked for, not the sentence.

WHY THIS EXISTS. Three supersession records in a row stated a moved-set by
hand and the third was wrong on the very axis it was correcting: the
`025943Z` record said "8 substantive + 2 label + 1 status of 4,140 shared,
no residual moved", and the coordinator's own leaf diff found 19
non-provenance leaves moved, 14 excluding provenance-ish noise, 11 numeric,
and BOTH `identity_residual` leaves among them. A prose count beside a
table has now contradicted the table three times in this programme; the
durable fix is that the emitter COMPUTES the moved set and REFUSES if the
prose claims a count it did not produce.

CLASSES, declared here and not decided per artifact:
  numeric-substantive  a numeric leaf whose change is not epsilon
  numeric-epsilon      a numeric leaf on an `*_residual*` path with
                       |delta| < 1e-9 -- a residual tracks the magnitude
                       of what it checks, so it MOVES when that grows and
                       it is not a substantive change; it is also NOT an
                       "unchanged number", which is the error A-2 named
  string               a non-numeric leaf
  provenance           a leaf whose path ends in a declared provenance or
                       resource field (run id, as_of, digests, timings)

    python3 live/pm_research/de_supersession_diff.py --selftest
    python3 live/pm_research/de_supersession_diff.py --old A.json --new B.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROTOCOL = "P003_DE_SUPERSESSION_LEAF_DIFF_V1"
EXPECTED_CHECKS = 18
EPSILON = 1e-9

#: A-1b (reviewer 89e81d5). THE OLD RULE WAS `name OR parent`, so ANY leaf
#: called `as_of` or `producing_code` was provenance WHEREVER IT SAT --
#: `cancellation_economics.as_of` classified provenance with its parent
#: ignored. A provenance NAME under a result-bearing parent is a result.
#:
#: The rule is now (container OR declared pair), never name alone:
#:   CONTAINERS -- every leaf beneath one of these is provenance
#:   PAIRS      -- an explicit (parent, leaf) allowlist for stamps that sit
#:                 outside a container, built from the leaves that actually
#:                 occur in this programme's artifacts rather than guessed
PROVENANCE_CONTAINERS = ("code_identity", "provenance", "source_identity",
                         "resource_observation", "supersedes")
PROVENANCE_PAIRS = frozenset({
    ("", "run_id"), ("", "as_of"), ("", "total_wall_s"),
    ("", "peak_rss_gb"), ("", "max_rss_gb"), ("", "emitted"),
    ("", "generated_at"), ("", "elapsed_s"),
    ("population", "as_of"), ("population", "feed_wall_s"),
    ("population", "tape_index_s"), ("population", "assembly_s"),
    ("population", "peak_gb"), ("population", "wall_s"),
})

#: A residual is a numeric leaf whose job is to be ~0; it moves with the
#: magnitude of the quantity it checks.
RESIDUAL_MARKERS = ("residual", "_resid")

#: A sentence that embeds the run id or an emission timestamp differs on
#: every emission and says nothing about the result. Recognised by RULE --
#: the two strings become equal once ISO-8601 stamps are masked -- and not
#: by listing the fields that happen to do it today.
_ISO = re.compile(r"\d{4}-?\d{2}-?\d{2}[T ]\d{2}:?\d{2}:?\d{2}Z?")


class DiffRefused(RuntimeError):
    """A claimed count disagrees with the computed one."""


def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path.lstrip("."), obj


def _is_provenance(path: str) -> bool:
    """Provenance requires a CONTAINER or a declared (parent, leaf) PAIR.

    A provenance-sounding NAME is not enough: `cancellation_economics.as_of`
    is a result-bearing field that happens to be called `as_of`, and the
    old `name OR parent` rule swallowed it (A-1b)."""
    parts = [x.split("[")[0] for x in path.split(".")]
    last = parts[-1]
    parent = parts[-2] if len(parts) > 1 else ""
    if any(par in parts[:-1] for par in PROVENANCE_CONTAINERS):
        return True
    return (parent, last) in PROVENANCE_PAIRS


def _is_residual(path: str) -> bool:
    low = path.lower()
    return any(m in low for m in RESIDUAL_MARKERS)


def classify(path: str, old, new) -> str:
    if _is_provenance(path):
        return "provenance"
    num = (isinstance(old, (int, float)) and not isinstance(old, bool)
           and isinstance(new, (int, float)) and not isinstance(new, bool))
    if not num:
        if (isinstance(old, str) and isinstance(new, str)
                and _ISO.sub("<TS>", old) == _ISO.sub("<TS>", new)):
            return "string-embedded-provenance"
        return "string"
    if _is_residual(path) and abs(float(new) - float(old)) < EPSILON:
        return "numeric-epsilon"
    return "numeric-substantive"


def diff(old_doc, new_doc) -> dict:
    o, n = dict(leaves(old_doc)), dict(leaves(new_doc))
    shared = set(o) & set(n)
    moved = {}
    for p in sorted(shared):
        a, b = o[p], n[p]
        if isinstance(a, float) and isinstance(b, float) and a == b:
            continue
        if a == b:
            continue
        moved[p] = {"old": a, "new": b, "class": classify(p, a, b)}
    counts = {}
    for v in moved.values():
        counts[v["class"]] = counts.get(v["class"], 0) + 1
    for cls in ("numeric-substantive", "numeric-epsilon", "string",
                "string-embedded-provenance", "provenance"):
        counts.setdefault(cls, 0)
    return {
        "protocol": PROTOCOL,
        "n_old_leaves": len(o), "n_new_leaves": len(n),
        "n_shared": len(shared),
        "added": sorted(set(n) - set(o)), "removed": sorted(set(o) - set(n)),
        "n_added": len(set(n) - set(o)), "n_removed": len(set(o) - set(n)),
        "moved": moved, "n_moved_total": len(moved),
        "counts_by_class": counts,
        "epsilon": EPSILON,
        "sentence": (
            f"{counts['numeric-substantive']} substantive numeric, "
            f"{counts['numeric-epsilon']} epsilon residual, "
            f"{counts['string']} string, "
            f"{counts['string-embedded-provenance']} string-embedded-"
            f"provenance, {counts['provenance']} provenance "
            f"-- of {len(shared)} shared leaves"),
        "how_this_was_produced": "COMPUTED by a leaf diff inside the "
                                 "emitter against the named predecessor, "
                                 "never counted by hand (A-2)",
    }


def assert_claim(d: dict, claim: dict) -> dict:
    """REFUSE if the prose claims a count the diff did not produce."""
    bad = {k: {"claimed": v, "computed": d["counts_by_class"].get(k)}
           for k, v in claim.items()
           if d["counts_by_class"].get(k) != v}
    if bad:
        raise DiffRefused(
            f"REFUSED: the supersession claims counts the computed leaf "
            f"diff does not produce: {bad}. A prose count beside a table "
            f"has contradicted the table three times in this programme; "
            f"the claim is not published unless the diff makes it.")
    return {"claim_checked_against_the_computed_diff": True,
            "claimed": claim}


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_supersession_diff] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    a = {"x": 1.0, "s": "old", "run_id": "A", "r": {"identity_residual": 1e-16},
         "deep": {"code_identity": {"f.py": "aaaa"}}, "same": 3}
    b = {"x": 2.0, "s": "new", "run_id": "B", "r": {"identity_residual": 5e-16},
         "deep": {"code_identity": {"f.py": "bbbb"}}, "same": 3}
    d = diff(a, b)
    ok(d["n_moved_total"] == 5 and d["counts_by_class"] == {
        "numeric-substantive": 1, "numeric-epsilon": 1, "string": 1,
        "string-embedded-provenance": 0, "provenance": 2},
       f"POSITIVE CONTROL, AND IT ADMITS: one substantive number, one "
       f"epsilon residual, one string and two provenance leaves are each "
       f"classified as themselves -- {d['sentence']}")
    ok("same" not in d["moved"],
       "an unchanged leaf is not in the moved set")
    ok(d["moved"]["r.identity_residual"]["class"] == "numeric-epsilon",
       "A-2's OWN CASE: a residual moving by 4e-16 is numeric-EPSILON -- "
       "it moves with the magnitude of what it checks, and it is neither "
       "substantive NOR 'an unchanged number', which is the error the "
       "025943Z record made")
    ok(d["moved"]["run_id"]["class"] == "provenance"
       and d["moved"]["deep.code_identity.f.py"]["class"] == "provenance",
       "provenance is recognised by a declared top-level PAIR and by "
       "CONTAINER, so a digest nested under code_identity is not counted "
       "as a result")
    # ---- A-1b: a provenance NAME under a result-bearing parent ---------
    hole = diff({"cancellation_economics": {"as_of": "2026-09-05T10:00Z"},
                 "arms": {"X": {"producing_code": "aaaa"}},
                 "provenance": {"as_of": "2026-09-05T10:00Z"}},
                {"cancellation_economics": {"as_of": "2026-09-06T10:00Z"},
                 "arms": {"X": {"producing_code": "bbbb"}},
                 "provenance": {"as_of": "2026-09-06T10:00Z"}})
    ok(hole["moved"]["cancellation_economics.as_of"]["class"] == "string"
       and hole["moved"]["arms.X.producing_code"]["class"] == "string",
       "A-1b KNOWN-BAD, THE HOLE ITSELF: a leaf CALLED `as_of` or "
       "`producing_code` sitting under a RESULT-BEARING parent is "
       "SUBSTANTIVE, not provenance. The old rule was `name OR parent` and "
       "classified both as provenance with the parent ignored")
    ok(hole["moved"]["provenance.as_of"]["class"] == "provenance",
       "AND THE OTHER DIRECTION, so the fix is not just a refusal: a "
       "GENUINE `provenance.as_of` is still provenance -- the container "
       "carries it, and tightening the rule did not break the case it "
       "exists for")
    ok(_is_provenance("population.as_of")
       and _is_provenance("provenance.code_identity.f.py")
       and _is_provenance("run_id")
       and not _is_provenance("cascade_baseline_candidates.population_block"
                              ".source_cache"),
       "and the pairs are built from leaves that ACTUALLY OCCUR in this "
       "programme's artifacts -- population.as_of, the provenance "
       "container, top-level run_id -- while "
       "`population_block.source_cache` is NOT provenance under either "
       "rule. THE REVIEWER'S MECHANISM IS RIGHT AND ITS EXAMPLE IS NOT: "
       "last round's real transcription defect was never at risk of being "
       "hidden, and saying so is the difference between adopting a finding "
       "and inheriting it")
    big = {"r": {"identity_residual": 1.0}}
    big2 = {"r": {"identity_residual": 3.0}}
    ok(diff(big, big2)["counts_by_class"]["numeric-substantive"] == 1,
       "KNOWN-BAD, THE OTHER SIDE: a residual that moves by 2.0 is "
       "SUBSTANTIVE, not epsilon -- the epsilon class is a magnitude test, "
       "not a name test, so a genuinely broken identity cannot hide in it")
    ok(diff({"a": 1}, {"a": 1, "b": 2})["n_added"] == 1
       and diff({"a": 1, "b": 2}, {"a": 1})["n_removed"] == 1,
       "added and removed leaves are counted separately from moved ones")

    ok(assert_claim(d, {"numeric-substantive": 1, "string": 1})[
           "claim_checked_against_the_computed_diff"] is True,
       "POSITIVE CONTROL ON THE CLAIM CHECK, AND IT ADMITS: a claim the "
       "diff produces is accepted")
    for bad_claim, why in (
            ({"numeric-substantive": 8}, "the 025943Z record's own claim"),
            ({"numeric-epsilon": 0}, "'no residual moved' when one did"),
            ({"string": 3}, "an inflated string count")):
        try:
            assert_claim(d, bad_claim)
            ok(False, f"KNOWN-BAD: {why} was accepted")
        except DiffRefused as e:
            ok("does not produce" in str(e),
               f"KNOWN-BAD REFUSED -- {why}: a claimed count the diff does "
               f"not produce is not published")
    _ts = diff({"m": "n=12 windows of 2026-09-06T03:20:53Z. It excludes X"},
               {"m": "n=12 windows of 2026-09-06T03:45:12Z. It excludes X"})
    ok(_ts["moved"]["m"]["class"] == "string-embedded-provenance",
       "a sentence differing ONLY by an embedded emission timestamp is "
       "string-embedded-provenance -- recognised by RULE (the strings "
       "match once ISO stamps are masked), not by listing the fields that "
       "happen to do it today")
    _ts2 = diff({"m": "at 2026-09-06T03:20:53Z the value was 4"},
                {"m": "at 2026-09-06T03:45:12Z the value was 5"})
    ok(_ts2["moved"]["m"]["class"] == "string",
       "KNOWN-BAD, THE OTHER SIDE: a sentence that ALSO changes beyond its "
       "timestamp stays a plain string -- the rule cannot be used to hide "
       "a changed claim inside a re-stamped sentence")

    ok(_is_residual("arms.X.identity_residual_cents")
       and not _is_residual("arms.X.cascade_factor"),
       "the residual marker matches residual paths and not others")
    ok(_is_provenance("resource_observation.max_rss_kib")
       and not _is_provenance("arms.X.cascade_factor")
       and not _is_provenance("arms.X.max_rss_kib"),
       "and the provenance marker does not swallow a result field -- nor "
       "does a resource NAME parked under an arm")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_supersession_diff] PASS -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--old", type=Path)
    ap.add_argument("--new", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.old and a.new):
        ap.error("--old and --new are required")
    d = diff(json.loads(a.old.read_text()), json.loads(a.new.read_text()))
    d["old"] = a.old.name
    d["new"] = a.new.name
    txt = json.dumps(d, indent=2, sort_keys=True)
    if a.output:
        a.output.write_text(txt + "\n")
    print(json.dumps({"sentence": d["sentence"], "n_added": d["n_added"],
                      "n_removed": d["n_removed"],
                      "counts": d["counts_by_class"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
