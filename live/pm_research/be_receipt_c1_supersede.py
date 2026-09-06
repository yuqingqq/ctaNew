"""C-1: THE `user_admission` GATE ROW READ AS ITS OWN OPPOSITE. SUPERSEDED.

`{"gate": "user_admission", "result": "PASS"}` reads as *a user admission
passed*. On a day carrying no admission the gate passes BECAUSE NONE WAS
NEEDED -- the opposite. 09-03 genuinely carries one ("RE-VERDICT UNDER A USER
RULING, ADMITTED BY THE USER"); 09-04 and 09-05 carry
`no_admission_covers_this_day: true`, and their rows said exactly what
09-03's said.

THE SCORER IS FIXED (`be_forward_day.py`, the `admission` and `means` keys on
that row) so no future day is emitted this way. THIS MODULE SUPERSEDES THE
DAYS ALREADY SEALED.

WHY NOT RE-RUN THE SCORER. Re-running would re-seal an accrued day, take ~30
minutes each, and change every timestamp and digest in the receipt -- which
is the opposite of "every other byte identical". The label is derived HERE
from the receipt's OWN `user_admission` block by the SAME expression the
fixed scorer now uses, and the claim that nothing else moved is a COMPUTED
PREDICATE, not a promise: the v2 is compared against v1 key-by-key and the
ONLY permitted difference is the two added keys on that one row.

v1 IS NOT EDITED. It stays at its bytes as provenance (rule 13) and the v2
names it by sha256, in the shape `be_forward_day_receipt_20260901.v2.json`
already established for a superseding forward receipt.

NO SEALED FILE IS OPENED AND NO SCORE IS READ.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import be_data_root as _BDR

ROOT = HERE.parents[1]
#: READ root resolves through the shared helper (R-559(C));
#: WRITES stay in this seat's worktree -- the ledger's
#: derived/ is the MAIN TREE's checkout.
_DATA_ROOT = Path(_BDR.resolve(ROOT)["data_root"])
DERIVED = _BDR.derived()  # BE48 B.4: one root, from the resolver. This was `ROOT / 'data/...'` -- a data root built on a CODE root, which the first version of `audit_derived_roots` could not see because the value holds no `parents` and no literal.

GATE = "user_admission"
MEANS_NONE = ("NO admission covers this day. The gate passes because the "
              "ORDINARY path applies and none was required -- it does NOT "
              "mean an admission was granted")
MEANS_PRESENT = "an admission covers this day and the gate accepted it"


class SupersedeRefused(RuntimeError):
    """A named refusal."""


def label_for(receipt: dict) -> tuple:
    """The SAME expression the fixed scorer uses, applied to v1's own block.

    Deriving it from the receipt rather than restating it is what makes this
    a supersession and not a second opinion."""
    adm = receipt.get("user_admission")
    if not isinstance(adm, dict):
        raise SupersedeRefused(
            f"REFUSED: receipt carries no `user_admission` block "
            f"({type(adm).__name__}). The label cannot be derived and this "
            f"module will not invent one.")
    if adm.get("no_admission_covers_this_day"):
        return "NONE", MEANS_NONE
    if "ADMISSION" not in adm:
        return "PRESENT_UNNAMED", MEANS_PRESENT
    return adm["ADMISSION"], MEANS_PRESENT


def relabel(receipt: dict) -> dict:
    """v1 with the two keys added to the one row. Nothing else touched."""
    out = copy.deepcopy(receipt)
    rows = [g for g in out.get("gates", []) if g.get("gate") == GATE]
    if len(rows) != 1:
        raise SupersedeRefused(
            f"REFUSED: {len(rows)} `{GATE}` rows in this receipt, expected "
            f"exactly 1. A relabel that cannot find its row uniquely would "
            f"be relabelling something else.")
    a, m = label_for(out)
    rows[0]["admission"] = a
    rows[0]["means"] = m
    return out


def identity_predicate(v1: dict, relabelled: dict) -> dict:
    """EVERY OTHER BYTE IDENTICAL -- COMPUTED, never asserted (rule 10).

    The relabelled copy has the two keys stripped again and is compared to v1
    under one canonical serialisation. Equal bytes is the whole claim."""
    stripped = copy.deepcopy(relabelled)
    added = []
    for g in stripped.get("gates", []):
        if g.get("gate") == GATE:
            for k in ("admission", "means"):
                if k in g:
                    added.append(f"gates[{GATE}].{k}")
                    del g[k]
    a = json.dumps(v1, sort_keys=True, separators=(",", ":"))
    b = json.dumps(stripped, sort_keys=True, separators=(",", ":"))
    return {
        "keys_added": sorted(added),
        "n_keys_added": len(added),
        "every_other_byte_identical": a == b,
        "how_checked": "the v2 with the added keys REMOVED is serialised "
                       "canonically and compared to v1 byte-for-byte",
        "v1_canonical_bytes": len(a),
        "v2_stripped_canonical_bytes": len(b),
        "v1_canonical_sha256": hashlib.sha256(a.encode()).hexdigest(),
        "v2_stripped_canonical_sha256": hashlib.sha256(b.encode()).hexdigest(),
    }


def build(day: str, *, derived: Path | None = None) -> dict:
    derived = Path(derived) if derived is not None else DERIVED
    p = derived / f"be_forward_day_receipt_{day}.json"
    if not p.exists():
        raise SupersedeRefused(f"REFUSED: no v1 receipt at {p}.")
    v1 = json.loads(p.read_text())
    if str(v1.get("day")) != str(day):
        raise SupersedeRefused(
            f"REFUSED: receipt says day {v1.get('day')!r}, asked {day!r}.")
    if not v1.get("sealed"):
        raise SupersedeRefused(
            f"REFUSED: {day} is not sealed. This module supersedes SEALED "
            f"forward receipts and opens nothing.")
    rl = relabel(v1)
    pred = identity_predicate(v1, rl)
    if not pred["every_other_byte_identical"]:
        raise SupersedeRefused(
            "REFUSED: the relabel changed more than the two keys. A "
            "supersession that cannot prove it moved one label is not one.")
    row = [g for g in rl["gates"] if g["gate"] == GATE][0]
    head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    return {
        "protocol": "BE_FORWARD_DAY_RECEIPT_C1_SUPERSEDE_V2",
        "data_root": _BDR.receipt_block(),
        "day": day,
        "supersedes_version": 1,
        "supersedes_receipt": {
            "path": str(p.relative_to(ROOT)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
            "status": "UNTOUCHED, kept as provenance (rule 13). A frozen "
                      "artifact is never edited; the correction supersedes "
                      "IN-BAND as this vN+1 receipt.",
        },
        "what_this_receipt_changes": (
            "ONE LABEL, on one gate row, and nothing else. "
            f"`gates[{GATE}]` now carries `admission` and `means` so the row "
            "says WHICH case it is. The scores, the run that produced them, "
            "every gate result, every count and every hash are unchanged."),
        "why": ("C-1 (reviewer 7a0c62e). `{\"gate\": \"user_admission\", "
                "\"result\": \"PASS\"}` reads as *a user admission passed*. "
                "On a day carrying none the gate passes BECAUSE NONE WAS "
                "NEEDED -- the opposite of what the row says. 09-03 has one; "
                "09-04 and 09-05 do not, and their rows were identical to "
                "09-03's."),
        "corrected_gate_row": row,
        "label_derived_from": "the receipt's OWN `user_admission` block, by "
                              "the same expression the fixed scorer uses -- "
                              "a supersession, not a second opinion",
        "every_other_byte_identical": pred,
        "scorer_fixed_so_no_future_day_needs_this": {
            "module": "be_forward_day.py",
            "keys_added_to_the_row": ["admission", "means"],
            "not_in_DECISION_VOCAB": True,
            "excused_paths_unchanged": ["gates[].gate"],
            "positive_control": "the driver's own suite now asserts a "
                                "no-admission day carries admission=NONE and "
                                "says in words that passing is not a grant",
        },
        "seal_state": "SEALED -- NOT OPENED BY THIS ACT",
        "no_sealed_file_read": True,
        "carrying_commit": head or None,
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 8


def selftest() -> int:
    import tempfile
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    base = {"day": "20260904", "sealed": True,
            "gates": [{"gate": GATE, "result": "PASS"},
                      {"gate": "other", "result": "PASS"}],
            "user_admission": {"no_admission_covers_this_day": True,
                               "note": "the ordinary gate applies, unchanged"},
            "population": {"n": 7}}
    a, m = label_for(base)
    ok(a == "NONE" and "does NOT mean an admission was granted" in m,
       "POSITIVE CONTROL, THE DEFECT'S OWN CASE: a receipt with NO admission "
       "labels the row NONE and says in words that passing is not a grant")
    withadm = dict(base, user_admission={"ADMISSION": "RE-VERDICT UNDER A "
                                         "USER RULING", "admitted_by": "USER"})
    a2, m2 = label_for(withadm)
    ok(a2.startswith("RE-VERDICT") and m2 == MEANS_PRESENT,
       f"POSITIVE CONTROL, THE OTHER CASE: a receipt WITH an admission "
       f"carries its own text ({a2[:28]}…) -- the two days are now "
       f"distinguishable at the row, which is the entire point")
    ok(a != a2,
       "and the two cases produce DIFFERENT labels -- before this they "
       "produced the same row")

    pred = identity_predicate(base, relabel(base))
    ok(pred["every_other_byte_identical"] and pred["n_keys_added"] == 2,
       f"IDENTITY IS COMPUTED: stripping the {pred['n_keys_added']} added "
       f"keys returns bytes identical to v1 "
       f"({pred['v1_canonical_sha256'][:12]}…)")
    tampered = relabel(base)
    tampered["population"]["n"] = 8
    ok(not identity_predicate(base, tampered)["every_other_byte_identical"],
       "KNOWN-BAD: a v2 that changed ANYTHING ELSE fails the identity "
       "predicate -- so a True is something that could have been False")

    for bad, needle, why in (
            (lambda: label_for({"day": "x"}),
             "carries no `user_admission`",
             "a receipt with no admission block REFUSES rather than "
             "inventing a label"),
            (lambda: relabel({"gates": [{"gate": GATE}, {"gate": GATE}],
                              "user_admission": {}}),
             "expected exactly 1",
             "two rows with the same gate name REFUSE -- a relabel that "
             "cannot find its row uniquely relabels something else")):
        try:
            bad(); ok(False, "must refuse: " + why)
        except SupersedeRefused as e:
            ok(needle in str(e), "KNOWN-BAD: " + why)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "be_forward_day_receipt_20260904.json").write_text(
            json.dumps(dict(base, sealed=False)))
        try:
            build("20260904", derived=d); ok(False, "unsealed must refuse")
        except SupersedeRefused as e:
            ok("is not sealed" in str(e),
               "KNOWN-BAD: an UNSEALED receipt REFUSES -- this module "
               "supersedes sealed days and opens nothing")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--day" in argv:
        day = argv[argv.index("--day") + 1]
        out = build(day)
        dst = DERIVED / f"be_forward_day_receipt_{day}.v2.json"
        dst.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(json.dumps({
            "written": str(dst),
            "admission": out["corrected_gate_row"]["admission"][:40],
            "every_other_byte_identical":
                out["every_other_byte_identical"]["every_other_byte_identical"],
        }))
        return 0
    print("usage: be_receipt_c1_supersede.py --selftest | --day <YYYYMMDD>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
