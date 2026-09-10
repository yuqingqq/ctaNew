"""Does a rebuilt book carry the SAME numbers as its predecessor?

WHY THIS EXISTS (BE 153/154). `e6a214f` claims that accelerated fragment
chunking "changes memory and nothing else". That claim is TRUE BY SELFTEST
-- `de_head_scoring` passes 44/44 with exact-equality falsifiers -- and
UNTESTED ON A REAL BOOK. A fixture proves the arithmetic; only a day proves
the pipeline. The EV23 timing build was stopped before it could serve as
that test, so the test is still owed, and the next build of ANY day at a
tip above 941e688 IS the test whether or not anyone means it to be.

The comparison is free at the time and expensive to reconstruct later:
once the predecessor receipt is superseded or the tape moves on, there is
nothing left to compare against.

WHAT IT REFUSES TO DO. It never answers for a field it was not given.
An absent path is a REFUSAL by name, not a pass -- that is rule 28's
class, a check switched off by the absence of its own input. A non-finite
value is a refusal too, because `nan != nan` would otherwise report a
mismatch that is really a missing measurement (BE 144).

FLOATS ARE COMPARED EXACTLY. The claim under test is "nothing else
changes", so a one-ULP move IS a change. This receipt already carries the
same coverage at two precisions --
`assembly_evidence.UNCOVERED_GENERATIONS.coverage` ends ...18 while
`asm.coverage_by_head.*.coverage` ends ...17 -- so a tolerance would hide
exactly the drift it was meant to catch, and a path chosen by vocabulary
rather than identity would watch the wrong number (rule 16).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

#: The fields a rebuild must reproduce. Each is a full path, matched by
#: IDENTITY: `Type.field`, not a grep for the last component. Both
#: coverage paths are here on purpose -- they differ in the last digit
#: and a check that watched only one would be watching half the claim.
INVARIANTS = (
    "selection.era",
    "selection.n_gap_bearing_windows",
    "placement_latency_split.n_generations",
    "placement_latency_split.n_tranches_at_this_L",
    "placement_latency_split.n_tranches_before_this_L",
    "reference.statuses.TRANCHE_KEPT",
    "reference.statuses.TRANCHE_BEFORE_PLACEMENT_LATENCY",
    "assembly_evidence.UNCOVERED_GENERATIONS.coverage",
    "asm.coverage_by_head.incumbent_linear_d.coverage",
    "asm.coverage_by_head.q1_arrival_composed_lgbm.coverage",
    "reference.windows",
    "reference.generations",
)

ABSENT = "FIELD_ABSENT_NO_COMPARISON_POSSIBLE"
NOT_FINITE = "VALUE_NOT_FINITE_NO_COMPARISON_POSSIBLE"

#: A sentinel distinct from every JSON value, so "the path is missing"
#: cannot be confused with "the path holds null".
_MISSING = object()


class IdentityRefused(RuntimeError):
    """The comparison cannot be made, so no verdict is reported."""


def _dig(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _finite(x) -> bool:
    if isinstance(x, bool):
        return True
    if isinstance(x, (int, float)):
        return math.isfinite(float(x))
    return True


def _same(a, b) -> bool:
    """EXACT equality. Floats by bit pattern, so one ULP is a mismatch."""
    if isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return False
        return math.copysign(1.0, float(a)) == math.copysign(1.0, float(b)) \
            and float(a) == float(b) and repr(float(a)) == repr(float(b))
    return type(a) is type(b) and a == b


def compare(old: dict, new: dict, *, fields=INVARIANTS) -> dict:
    """Compare two receipts on the fields that a rebuild must not move.

    Returns per-field evidence and a computed `all_match`. Raises
    IdentityRefused if any field is absent from either receipt or holds a
    non-finite number -- an unanswerable comparison is never a pass.
    """
    rows = []
    for path in fields:
        a, b = _dig(old, path), _dig(new, path)
        where = []
        if a is _MISSING:
            where.append("old")
        if b is _MISSING:
            where.append("new")
        if where:
            raise IdentityRefused(
                f"REFUSED -- {ABSENT}: {path} missing from "
                f"{' and '.join(where)} receipt")
        if not (_finite(a) and _finite(b)):
            raise IdentityRefused(
                f"REFUSED -- {NOT_FINITE}: {path} is not finite "
                f"(old={a!r} new={b!r})")
        rows.append({"path": path, "old": a, "new": b, "matches": _same(a, b)})
    return {
        "fields_compared": len(rows),
        "rows": rows,
        "n_moved": sum(1 for r in rows if not r["matches"]),
        # COMPUTED, never typed (rule 10). A hardcoded verdict beside a
        # table has contradicted the table three times.
        "all_match": all(r["matches"] for r in rows),
    }


def falsify() -> int:
    """Every checker ships a falsifier (rule 15): a positive control it
    MUST flag, and a known-bad input it MUST refuse."""
    base = {
        "selection": {"era": "clob_v4_1", "n_gap_bearing_windows": 14},
        "placement_latency_split": {"n_generations": 300147,
                                    "n_tranches_at_this_L": 25818,
                                    "n_tranches_before_this_L": 23747},
        "reference": {"statuses": {"TRANCHE_KEPT": 25818,
                                   "TRANCHE_BEFORE_PLACEMENT_LATENCY": 23747},
                      "windows": 288, "generations": 300147},
        "assembly_evidence": {"UNCOVERED_GENERATIONS":
                              {"coverage": 0.9209220815133918}},
        "asm": {"coverage_by_head": {
            "incumbent_linear_d": {"coverage": 0.9209220815133917},
            "q1_arrival_composed_lgbm": {"coverage": 0.9209220815133917}}},
    }
    checks = []

    def note(name, ok):
        checks.append((name, bool(ok)))

    # 0. NEGATIVE CONTROL: identical receipts must match, or every
    #    positive below is meaningless.
    note("identical_receipts_match", compare(base, json.loads(
        json.dumps(base)))["all_match"] is True)

    # 1. POSITIVE CONTROL: an integer that moved must be flagged.
    moved = json.loads(json.dumps(base))
    moved["selection"]["n_gap_bearing_windows"] = 13
    r = compare(base, moved)
    note("moved_integer_is_flagged", r["all_match"] is False and r["n_moved"] == 1)

    # 2. POSITIVE CONTROL, ONE ULP. The whole point of exact comparison:
    #    the smallest representable move must still be a mismatch.
    ulp = json.loads(json.dumps(base))
    ulp["assembly_evidence"]["UNCOVERED_GENERATIONS"]["coverage"] = \
        math.nextafter(0.9209220815133918, 0.0)
    note("one_ulp_is_flagged", compare(base, ulp)["all_match"] is False)

    # 3. POSITIVE CONTROL: the two coverage paths differ in the last
    #    digit, so a checker that resolved them to one path would report
    #    a mismatch here. It must not.
    note("two_coverage_paths_are_distinct",
         base["assembly_evidence"]["UNCOVERED_GENERATIONS"]["coverage"]
         != base["asm"]["coverage_by_head"]["incumbent_linear_d"]["coverage"])

    # 4. KNOWN-BAD INPUT: a missing field must REFUSE, never pass.
    gone = json.loads(json.dumps(base))
    del gone["placement_latency_split"]["n_generations"]
    try:
        compare(base, gone)
        note("absent_field_refuses", False)
    except IdentityRefused as e:
        note("absent_field_refuses", ABSENT in str(e))

    # 5. KNOWN-BAD INPUT: NaN must REFUSE. Without this, `nan == nan`
    #    is False and a missing measurement reports as a real mismatch.
    nan = json.loads(json.dumps(base))
    nan["asm"]["coverage_by_head"]["incumbent_linear_d"]["coverage"] = \
        float("nan")
    try:
        compare(base, nan)
        note("nan_refuses", False)
    except IdentityRefused as e:
        note("nan_refuses", NOT_FINITE in str(e))

    # 6. A null is NOT the same as a missing path.
    nulled = json.loads(json.dumps(base))
    nulled["selection"]["era"] = None
    r = compare(base, nulled)
    note("null_is_a_mismatch_not_a_refusal", r["all_match"] is False)

    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    bad = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_book_identity_compare",
                      "n_checks": len(checks), "n_failed": len(bad),
                      "failed": bad}))
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return falsify()
    if len(argv) < 2:
        print("usage: be_book_identity_compare.py <old_receipt.json> "
              "<new_receipt.json> | --selftest")
        return 2
    old = json.loads(Path(argv[0]).read_text())
    new = json.loads(Path(argv[1]).read_text())
    try:
        out = compare(old, new)
    except IdentityRefused as e:
        print(json.dumps({"refused": str(e)}, indent=1))
        return 3
    for r in out["rows"]:
        flag = "    " if r["matches"] else "MOVED"
        print(f"  {flag}  {r['path']:<62} old={r['old']!r} new={r['new']!r}")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
    return 0 if out["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
