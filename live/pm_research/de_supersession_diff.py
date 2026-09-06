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
import math
from pathlib import Path


PROTOCOL = "P003_DE_SUPERSESSION_LEAF_DIFF_V1"
EXPECTED_CHECKS = 13
EPSILON = 1e-9

#: Leaf names whose movement is provenance or resource, never a result.
PROVENANCE_LEAVES = frozenset({
    "run_id", "as_of", "carrying_commit", "carrying_commit_short",
    "producing_code", "producing_code_path", "producing_code_sha256",
    "wall_s", "feed_wall_s", "tape_index_s", "assembly_s", "total_wall_s",
    "peak_gb", "peak_rss_gb", "max_rss_gb", "max_rss_kib", "emitted",
    "generated_at", "elapsed_s", "wall_seconds", "user_cpu_seconds",
    "system_cpu_seconds", "n_bytes",
})
PROVENANCE_PARENTS = ("code_identity", "source_identity",
                      "resource_observation")

#: A residual is a numeric leaf whose job is to be ~0; it moves with the
#: magnitude of the quantity it checks.
RESIDUAL_MARKERS = ("residual", "_resid")


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
    last = path.rsplit(".", 1)[-1].split("[")[0]
    if last in PROVENANCE_LEAVES:
        return True
    return any(f".{par}." in f".{path}." for par in PROVENANCE_PARENTS)


def _is_residual(path: str) -> bool:
    low = path.lower()
    return any(m in low for m in RESIDUAL_MARKERS)


def classify(path: str, old, new) -> str:
    if _is_provenance(path):
        return "provenance"
    num = (isinstance(old, (int, float)) and not isinstance(old, bool)
           and isinstance(new, (int, float)) and not isinstance(new, bool))
    if not num:
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
                "provenance"):
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
            f"{counts['string']} string, {counts['provenance']} provenance "
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
        "provenance": 2},
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
       "provenance is recognised by LEAF NAME and by PARENT, so a digest "
       "nested under code_identity is not counted as a result")
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
    ok(_is_residual("arms.X.identity_residual_cents")
       and not _is_residual("arms.X.cascade_factor"),
       "the residual marker matches residual paths and not others")
    ok(_is_provenance("resource_observation.max_rss_kib")
       and not _is_provenance("arms.X.cascade_factor"),
       "and the provenance marker does not swallow a result field")

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
