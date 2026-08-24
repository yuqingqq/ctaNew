"""Standing check: a DECISION reserved to an owner, encoded as a worker-produced FACT.

R-96(3). Three instances surfaced separately this session before anyone named the
class, the last being `GateEvidence.decision_eligible: bool` produced by
BE-Uncertainty while R-ADMISS reserves the SELECTION DECISION to the coordinator.
The coordinator's summary is the definition: **a boolean is the easiest place in
a schema to hide a decision.**

WHAT SEPARATES A DECISION FROM A FACT. A fact is derivable from data by a stated
rule; anyone with the data reaches the same value. A decision is not — it
allocates authority, and its value depends on WHO is entitled to set it. So the
question a flagged field must answer is not "is this true?" but "could a
measurement have produced it?"

FALSE-POSITIVE ANALYSIS (R-79 — every instrument ships with one). Decision
vocabulary appears constantly in field names that are not decisions. Run over
contracts v23 the naive keyword sweep flags 22 fields, of which the majority are
these classes, and each is excluded by an asserted rule rather than by eye:

  * TOOLING          `validator_version`, `validator_code_sha256`,
                     `n_validated_tx` -- "valid" inside "validator/validated".
                     Facts about the instrument, not about entitlement.
  * TEMPORAL/SCOPE   `valid_for`, `valid_from`, `valid_to` -- extents.
  * COMPUTED PROPERTY `psd_validated` -- positive semi-definiteness is decided by
                     arithmetic, and arithmetic is not an owner.
  * NAMED VALUE      `selected_alpha`, `selected_bias_coeff` -- "selected" is a
                     prefix on a parameter's VALUE; the decision lives elsewhere.
  * PREDICATE TEXT   `admissible_when`, `validation_gate` -- a string DESCRIBING
                     a rule is not the rule's output.

What survives is a bool/enum that names an entitlement and is emitted by a
module. That is a CANDIDATE, never a verdict: this check reports what needs
adjudicating and by whom. It does not adjudicate.

    python3 check_decision_as_fact.py --selftest
    python3 check_decision_as_fact.py contracts/contracts.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

DECISION_WORDS = ("eligible", "admissible", "approved", "allowed", "selected",
                  "accepted", "authorized", "authorised", "promoted", "enabled",
                  "licensed", "waived", "valid")

# Each exclusion is a CLASS with a reason, asserted in the selftest.
EXCLUSIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"validator|validated|validation"), "TOOLING — a fact about the instrument"),
    (re.compile(r"^valid_(for|from|to)$"),          "TEMPORAL/SCOPE — an extent, not an entitlement"),
    (re.compile(r"^selected_\w+"),                  "NAMED VALUE — a prefix on a parameter's value"),
    (re.compile(r"_when$|_gate$|_reason$"),         "PREDICATE TEXT — describes a rule, is not its output"),
]

# A decision hides most easily in a bool; an enum is the next easiest.
DECIDABLE_DECL = re.compile(r"^\s*(bool|enum:)", re.I)

CONTROL = ("GateEvidence", "decision_eligible")

# --- AXIS 2: IS THERE A LICENSING RULE AT ALL? -------------------------------
# The coordinator's example is the sharpest form of the class: `admissible` on
# calibration rows "with no A-CALIB-1 behind it" -- a boolean asserting an
# entitlement whose licensing rule DOES NOT EXIST. Axis 1 asks who emits the
# decision; axis 2 asks what entitles anyone to.
#
# MATCH ON IDENTITY, NEVER ON THE WORD. BE's first attempt searched rule bodies
# for the bare field name and reported `GateEvidence.admissible` as licensed by
# R-ONEROW and R-FLOW -- both of which merely contain the phrase
# "knowledge-admissible" in prose about unrelated subjects. That false positive
# would have withdrawn a correct finding, and it is R-79 exactly: a rule that
# DISCUSSES admissibility is not a rule that LICENSES this field. So a licence
# must name the field with the type, or name the field as a whole word inside a
# rule that also names the type.


def licensing_rules(doc: dict, tname: str, fname: str) -> list[str]:
    """Rules that LICENSE this field, by identity. Empty means unlicensed."""
    out: list[str] = []
    qualified = re.compile(rf"\b{re.escape(tname)}\.{re.escape(fname)}\b")
    bare = re.compile(rf"\b{re.escape(fname)}\b")
    tname_re = re.compile(rf"\b{re.escape(tname)}\b")
    for rname, body in (doc.get("rules") or {}).items():
        blob = yaml.safe_dump(body)
        if qualified.search(blob) or (bare.search(blob) and tname_re.search(blob)):
            out.append(rname)
    return out


def excluded(field: str) -> str | None:
    for pat, why in EXCLUSIONS:
        if pat.search(field):
            return why
    return None


def scan(doc: dict) -> tuple[list[dict], list[dict]]:
    producer: dict[str, list[str]] = {}
    for m, b in (doc.get("modules") or {}).items():
        for p in (b.get("produces") or []):
            producer.setdefault(str(p).split("[")[0], []).append(m)

    cands: list[dict] = []
    dropped: list[dict] = []
    for tn, tb in sorted((doc.get("types") or {}).items()):
        for fn, ft in (tb.get("fields") or {}).items():
            if not any(w in fn.lower() for w in DECISION_WORDS):
                continue
            row = {"type": tn, "field": fn, "decl": str(ft),
                   "producers": producer.get(tn, []),
                   "licences": licensing_rules(doc, tn, fn)}
            why = excluded(fn)
            if why:
                dropped.append({**row, "excluded_as": why}); continue
            if not DECIDABLE_DECL.match(str(ft)):
                dropped.append({**row, "excluded_as":
                                "NOT BOOL/ENUM — carries a value, not a verdict"}); continue
            cands.append(row)
    return cands, dropped


def run(path: Path) -> int:
    if not path.is_file():
        print(f"REFUSED: no such contract file: {path}", file=sys.stderr)
        return 2
    doc = yaml.safe_load(path.read_text())
    cands, dropped = scan(doc)

    if not any(c["type"] == CONTROL[0] and c["field"] == CONTROL[1] for c in cands):
        print(f"REFUSED: POSITIVE CONTROL FAILED — {CONTROL[0]}.{CONTROL[1]} is the "
              f"known instance of this class and the check did not flag it. Every "
              f"clean result below would be meaningless.", file=sys.stderr)
        return 2
    print(f"positive control PASSED ({CONTROL[0]}.{CONTROL[1]} flagged)")
    print(f"SCOPE — {path.name}, version {doc.get('version')}, "
          f"{len(doc.get('types') or {})} types, {len(doc.get('modules') or {})} modules\n")

    unlicensed = [c for c in cands if not c["licences"]]
    print(f"CANDIDATES — {len(cands)} (a bool/enum naming an entitlement, emitted by a module):")
    for c in cands:
        who = ", ".join(c["producers"]) or "**NO PRODUCER**"
        lic = ", ".join(c["licences"]) or "**NO LICENSING RULE**"
        print(f"  {c['type']}.{c['field']:<24} {c['decl']:<20} emitted by {who:<20} licensed by {lic}")
    print(f"\n  UNLICENSED: {len(unlicensed)} of {len(cands)} — a decision whose "
          f"entitlement no rule confers is the sharpest form of this class.")
    print(f"\nEXCLUDED — {len(dropped)}, each by a named class (R-79):")
    for d in dropped:
        print(f"  {d['type']}.{d['field']:<28} {d['excluded_as']}")
    return 1 if cands else 0


def selftest() -> int:
    checks = 0

    def ok(c: bool, label: str) -> None:
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    # every exclusion class asserted against a real field name
    ok(excluded("validator_version"), "TOOLING excluded")
    ok(excluded("n_validated_tx"), "TOOLING excluded even mid-name")
    ok(excluded("valid_from"), "TEMPORAL excluded")
    ok(excluded("selected_alpha"), "NAMED VALUE excluded")
    ok(excluded("admissible_when"), "PREDICATE TEXT excluded")
    ok(excluded("inadmissible_reason"), "a _reason field is text, not a verdict")
    ok(not excluded("decision_eligible"), "the KNOWN INSTANCE is NOT excluded")
    ok(not excluded("admissible"), "a bare entitlement is NOT excluded")

    doc = {"version": 1,
           "types": {"GateEvidence": {"fields": {"decision_eligible": "bool",
                                                 "admissible": "bool",
                                                 "inadmissible_reason": "str?",
                                                 "valid_from": "Timestamp"}},
                     "Other": {"fields": {"selected": "enum:A|B"}}},
           "modules": {"BE-Uncertainty": {"produces": ["GateEvidence"]}}}
    c, d = scan(doc)
    names = {(x["type"], x["field"]) for x in c}
    ok(("GateEvidence", "decision_eligible") in names, "the control is flagged")
    ok(("GateEvidence", "admissible") in names, "its twin on the same type is too")
    ok(("GateEvidence", "valid_from") not in names, "temporal is dropped")
    ok(("Other", "selected") in names, "an enum with NO producer is still a candidate")
    ok(next(x for x in c if x["field"] == "selected")["producers"] == [],
       "and its missing producer is reported, not defaulted")

    import tempfile
    p = Path(tempfile.mkdtemp()) / "c.yaml"
    p.write_text(yaml.safe_dump(doc))
    ok(run(p) == 1, "candidates exit non-zero")

    # --- AXIS 2's FALSIFIER: the case that would have caught `admissible` -----
    lic_doc = {"version": 1,
               "types": {"CalibrationPanelRow": {"fields": {"route_a_admissible": "bool"}},
                         "GateEvidence": {"fields": {"decision_eligible": "bool",
                                                     "admissible": "bool"}}},
               "modules": {"EV-Calibration": {"produces": ["CalibrationPanelRow"]},
                           "BE-Uncertainty": {"produces": ["GateEvidence"]}},
               "rules": {"R-ONEROW": {"body": "one row per window, including an "
                                              "explicit unavailable row when no quote "
                                              "is knowledge-admissible"},
                         "R-REAL": {"body": "GateEvidence.admissible is set only by "
                                            "the registry owner"}}}
    ok(licensing_rules(lic_doc, "CalibrationPanelRow", "route_a_admissible") == [],
       "A BOOLEAN WITH NO LICENSING RULE IS DETECTED — the case that catches "
       "`admissible` on calib rows with no A-CALIB-1 behind it")
    ok(licensing_rules(lic_doc, "GateEvidence", "admissible") == ["R-REAL"],
       "and a field a rule ACTUALLY names is reported as licensed")
    ok("R-ONEROW" not in licensing_rules(lic_doc, "GateEvidence", "admissible"),
       "while a rule merely CONTAINING the word 'admissible' in prose does NOT "
       "license it — R-79, and this exact false positive nearly withdrew a "
       "correct finding")
    ok(licensing_rules(lic_doc, "GateEvidence", "decision_eligible") == [],
       "the ruled instance is unlicensed, which is why it was ruled")
    p.write_text(yaml.safe_dump({"version": 1, "types": {}, "modules": {}}))
    ok(run(p) == 2, "POSITIVE CONTROL FAILING refuses rather than reporting clean")
    ok(run(Path("/nonexistent.yaml")) == 2, "a missing file REFUSES")

    print(f"check_decision_as_fact selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.path:
        ap.print_help(); return 2
    return run(a.path)


if __name__ == "__main__":
    raise SystemExit(main())
