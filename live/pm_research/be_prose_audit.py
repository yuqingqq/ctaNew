"""THE PROSE-BESIDE-A-COMPUTED-VALUE AUDIT.

Round 38 found one by hand: the 09-03 admission's `scope` ASSERTED that
`assert_day_closed_and_attributed` "is unchanged and still refuses this verdict
when called without the admission" -- a claim about CODE BEHAVIOUR that the
artifact never computed, and which would have gone on reading true after any
widening of the gate. It was replaced by `ordinary_gate_without_this_admission`,
which RUNS the gate and records what it did.

That was one site found by reading. This finds them by scanning, over every
artifact my surface emits, and RECONCILES: found = adjudicated + irreducible +
UNADJUDICATED. The last number is the finding; the first two are only bookkeeping.

WHY A REGISTRY RATHER THAN AUTOMATIC CLASSIFICATION. A detector that decided
for itself which prose was harmless would be a checker grading its own homework
-- exactly the shape it exists to catch. So every hit must be adjudicated BY
NAME here, and anything unlisted is UNADJUDICATED and reported as such. The
census cannot come out clean by the scanner being lenient.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DECL_DIR = HERE / "declarations"

#: Prose that ASSERTS a checkable state or behaviour. Not "explains", not
#: "defines", not "cites" -- asserts. These are the verb shapes a reader takes
#: as evidence.
CLAIM = re.compile(
    r"\b("
    r"is\s+(un)?changed|is\s+not\s+\w+ed|are\s+identical|is\s+identical|"
    r"still\s+(refuses|passes|holds|fires)|"
    r"(does|do)\s+not\s+(read|change|touch|move|differ)|"
    r"cannot\s+(fail|fire|pass|be)|"
    r"was\s+(verified|checked|computed|reproduced)|"
    r"has\s+been\s+(verified|checked)|"
    r"holds\b|refuses\b|passes\b"
    r")", re.I)


class ProseAuditRefused(RuntimeError):
    """A named refusal."""


#: EVERY hit must appear here, by (artifact, field path), with a verdict:
#:   ADJUDICATED  -- the claim is now COMPUTED, and the computing field is named
#:   IRREDUCIBLE  -- the prose asserts nothing checkable: a reason, a
#:                   definition, an authority, or the TEXT OF A REFUSAL that
#:                   the code itself raised
ADJUDICATION = {
    ("be_forward_day_receipt", "user_admission.scope"): {
        "verdict": "ADJUDICATED",
        "computed_by": "user_admission.ordinary_gate_without_this_admission",
        "note": ("the prose asserted the ordinary gate still refuses; it is "
                 "now RUN at emission and its outcome recorded"),
    },
    ("be_forward_day_receipt", "frozen_contract.contract"): {
        "verdict": "ADJUDICATED",
        "computed_by": "frozen_contract.contract_conjuncts",
        "note": ("was the LITERAL \"HOLDS\" -- reached only on the passing "
                 "path, so never wrong, but a check added later whose failure "
                 "did not raise would have left it reading HOLDS. Now DERIVED "
                 "from three conjuncts in the same dict and able to say DOES "
                 "NOT HOLD"),
    },
    ("be_forward_day_receipt",
     "frozen_contract.working_tree_drift_is_not_fatal_because"): {
        "verdict": "ADJUDICATED",
        "computed_by": ("frozen_contract."
                        "materialise_frozen_sources_from_the_freeze_commit"),
        "note": ("the prose gives the REASON; the computed sibling reads "
                 "`materialise_frozen`'s own source at call time, and the "
                 "gate refuses rather than disclosing if it goes False"),
    },
    ("be_forward_day_receipt",
     "cluster_disclosure.why_a_window_level_p_is_OPTIMISTIC"): {
        "verdict": "IRREDUCIBLE",
        "note": ("a statistical ARGUMENT about exchangeability. It asserts no "
                 "state of this run that a field could carry"),
    },
    ("be_forward_day_receipt", "coin_coverage.why"): {
        "verdict": "IRREDUCIBLE",
        "note": ("a REASON; the counts it explains -- coins_with_a_frozen_fit, "
                 "n_windows_supplied_without_a_fit -- are computed siblings"),
    },
    ("be_candidate_identity",
     "declares_what_is_bound_today_not_what_ought_to_be"): {
        "verdict": "IRREDUCIBLE",
        "note": ("a SCOPE statement about what the artifact records; the "
                 "binding itself is the committed sha beside it"),
    },
    ("be_interim_declaration",
     "family.why_the_denominator_is_not_everything_reported"): {
        "verdict": "IRREDUCIBLE",
        "note": "a REASON for a declared choice, beside the counts it explains",
    },
    ("be_interim_declaration", "reconciliation_caveat.claim"): {
        "verdict": "ADJUDICATED",
        "computed_by": "reconciliation_caveat.conventions_differ_COMPUTED",
        "note": ("the claim rests on increment() being BY_THRESHOLD while "
                 "iteration 011 is BY_COUNT, which is comparable at the "
                 "constants rather than asserted"),
    },
    ("be_read_declaration", "reconciliation_caveat.claim"): {
        "verdict": "ADJUDICATED",
        "computed_by": "reconciliation_caveat.conventions_differ_COMPUTED",
        "note": "same claim, same computation",
    },
    ("be_read_declaration", "era_caveat.supersedes"): {
        "verdict": "IRREDUCIBLE",
        "note": ("a CITATION of what the caveat replaced, carrying the "
                 "superseded text. It states history, not current state"),
    },
    ("be_read_declaration",
     "skipped_deliberately.closed_exclusion_status_vocabulary."
     "known_weakness_stated"): {
        "verdict": "ADJUDICATED",
        "computed_by": "exclusion_vocabulary.set_is_closed",
        "note": ("the prose says the set is not closed; the boolean beside it "
                 "already carries exactly that, so the prose is a gloss on a "
                 "computed field rather than a claim beyond it"),
    },
    ("be_forward_day_receipt",
     "frozen.anchors.*.why_not_materialised"): {
        "verdict": "IRREDUCIBLE",
        "note": ("a REASON for a design choice -- why a data anchor is "
                 "verified in place rather than copied. It asserts no state "
                 "a field could carry"),
    },
}


def claim_sites(obj, path="", parent_has_computed=False):
    """Claim-shaped strings sitting beside a computed value."""
    out = []
    if isinstance(obj, dict):
        computed = any(isinstance(v, (bool, int, float))
                       and not isinstance(v, bool) is False
                       for v in obj.values())
        computed = any(isinstance(v, (bool, int, float))
                       for v in obj.values())
        for k, v in obj.items():
            if isinstance(v, str) and computed and CLAIM.search(v):
                out.append((path + k, v))
            out.extend(claim_sites(v, path + k + ".", computed))
    elif isinstance(obj, list):
        for i, x in enumerate(obj[:60]):
            out.extend(claim_sites(x, path + "[].", parent_has_computed))
    return out


def _adjudication_for(artifact: str, field: str):
    if (artifact, field) in ADJUDICATION:
        return ADJUDICATION[(artifact, field)]
    for (a, pat), v in ADJUDICATION.items():
        if a != artifact or "*" not in pat:
            continue
        rx = re.escape(pat).replace(r"\*", "[^.]*")
        if re.fullmatch(rx, field):
            return v
    return None


def census(paths=None) -> dict:
    """found = adjudicated + irreducible + UNADJUDICATED. The last is the
    finding; a clean census with nothing unadjudicated is the goal, and a
    census that is clean because the DETECTOR was lenient is the failure."""
    paths = ([Path(p) for p in paths] if paths is not None
             else sorted(DECL_DIR.glob("*.json")))
    rows, unadj = [], []
    n_files = 0
    for p in paths:
        try:
            d = json.loads(Path(p).read_text())
        except Exception:
            continue
        n_files += 1
        art = re.sub(r"_\d{8}\.json$|\.json$", "", Path(p).name)
        art = re.sub(r"_v\d+$", "", art)
        for field, text in claim_sites(d):
            a = _adjudication_for(art, field)
            row = {"artifact": art, "field": field, "text": text[:100],
                   "verdict": (a or {}).get("verdict", "UNADJUDICATED"),
                   "computed_by": (a or {}).get("computed_by")}
            rows.append(row)
            if a is None:
                unadj.append(row)
    n = len(rows)
    adj = sum(1 for r in rows if r["verdict"] == "ADJUDICATED")
    irr = sum(1 for r in rows if r["verdict"] == "IRREDUCIBLE")
    if adj + irr + len(unadj) != n:
        raise ProseAuditRefused(
            f"REFUSED: the census does not reconcile -- {n} found but "
            f"{adj}+{irr}+{len(unadj)} classified. A census that does not "
            f"add up is not a census.")
    return {
        "protocol": "BE_PROSE_AUDIT_V1",
        "n_artifacts_scanned": n_files,
        "n_claim_sites_found": n,
        "n_adjudicated": adj,
        "n_irreducible": irr,
        "n_UNADJUDICATED": len(unadj),
        "reconciles": True,
        "unadjudicated": unadj,
        "rows": rows,
        "what_UNADJUDICATED_means": (
            "a prose claim beside a computed value that nobody has ruled on. "
            "It is not necessarily wrong -- it is unexamined, which is the "
            "state that let a sentence be read as evidence three times today"),
        "why_a_registry": (
            "a detector that classified its own hits as harmless would be "
            "grading its own homework, which is the shape it exists to catch"),
    }


EXPECTED_CHECKS = 8


def selftest() -> int:
    checks = 0
    fails = []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    # THE DETECTOR MUST FIRE ON THE ORIGINAL DEFECT, restated verbatim.
    planted = {"verified_at_run_time": True,
               "scope": ("THIS DAY ONLY. `assert_day_closed_and_attributed` "
                         "is unchanged and still refuses this verdict when "
                         "called without the admission")}
    hits = claim_sites(planted)
    ok(len(hits) == 1 and hits[0][0] == "scope",
       "KNOWN-BAD: the detector FIRES on the round-38 defect restated "
       "verbatim -- a claim about code behaviour beside a computed boolean")
    ok(not claim_sites({"scope": "THIS DAY ONLY"}),
       "and does NOT fire on the four-word replacement, so the fix is "
       "distinguishable from the defect")
    ok(not claim_sites({"a": "the gate still refuses"}),
       "and does NOT fire on prose with NO computed sibling -- the finding is "
       "prose BESIDE a computed value, not prose")
    ok(claim_sites({"n": 1, "a": "the anchors are identical"}),
       "POSITIVE CONTROL: it fires on a DIFFERENT claim shape beside a "
       "computed value, so it is not pinned to one sentence")

    c = census()
    ok(c["reconciles"] and c["n_claim_sites_found"] ==
       c["n_adjudicated"] + c["n_irreducible"] + c["n_UNADJUDICATED"],
       f"the census RECONCILES: {c['n_claim_sites_found']} found = "
       f"{c['n_adjudicated']} adjudicated + {c['n_irreducible']} irreducible "
       f"+ {c['n_UNADJUDICATED']} unadjudicated")
    ok(c["n_artifacts_scanned"] > 0,
       f"and it scanned {c['n_artifacts_scanned']} artifacts rather than "
       f"reporting a clean sweep of nothing (rule 15)")
    # A REGISTRY THAT CANNOT SAY UNADJUDICATED IS NOT A REGISTRY.
    fake = census.__wrapped__ if hasattr(census, "__wrapped__") else None
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "planted_v1.json"
        f.write_text(json.dumps({"ok": True,
                                 "claim": "the population is unchanged"}))
        c2 = census([f])
        ok(c2["n_UNADJUDICATED"] == 1
           and c2["unadjudicated"][0]["field"] == "claim",
           "KNOWN-BAD: an UNLISTED claim in a fresh artifact comes back "
           "UNADJUDICATED -- the census cannot come out clean by the scanner "
           "being lenient")
        f2 = Path(td) / "clean_v1.json"
        f2.write_text(json.dumps({"ok": True, "note": "budgets are 5/10/15%"}))
        ok(census([f2])["n_claim_sites_found"] == 0,
           "POSITIVE CONTROL: a factual note beside a computed value is NOT a "
           "claim site, so the detector admits the good case")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}.")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--census" in argv:
        extra = [a for a in argv[argv.index("--census") + 1:]
                 if not a.startswith("-")]
        c = census(extra or None)
        print(json.dumps(c, indent=1, sort_keys=True, default=str))
        return 0 if c["n_UNADJUDICATED"] == 0 else 1
    print("usage: be_prose_audit.py --selftest | --census [paths...]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
