"""Checker behind DE_REGISTRY_AMENDMENT_PROPOSAL.md -- predicates, not prose.

SURFACE AUTHORISATION (R-126, in-file): R-379 TASK 2 (registry closure),
DE seat. RESEARCH-ONLY, OFFLINE. This module READS `contracts/contracts.yaml`
and NEVER writes it: the registry wins on types and an amendment lands by
coordinator/USER act (SEAT_PROTOCOL rule 4 / the proposal's own header).

WHY IT READS THE PROPOSAL'S OWN YAML.  The amendment records are parsed OUT OF
`plans/DE_REGISTRY_AMENDMENT_PROPOSAL.md`, keyed by their `# DE-AMENDMENT-x`
marker, rather than restated here.  A checker holding its own copy of the thing
it checks proves the copy, not the document -- and the document is what a
reader reviews and what a coordinator would apply.  Same reason the
multiplicity derivation is recomputed from recorded inputs instead of
transcribed (LANE4 B4.1): a transcribed record cannot notice a doc edit.

BOTH DIRECTIONS (rule 15/16).  Every claim ships a positive control that must
ADMIT and a known-bad that must FIRE:
  * the real contracts.yaml passes `contract_check.invariants()` AND the
    amended document passes -- so "amended is clean" is not a vacuous pass from
    an invariant set that never fails;
  * an amendment naming an undeclared type MUST raise unresolved-reference;
  * a module record consuming a type with no producer and no config_supplied
    entry MUST be caught;
  * a doctored proposal with a block deleted MUST refuse, never pass on the
    blocks that remain (absence is the failure mode that reads as success).

    python3 live/pm_research/de_registry_amendment_check.py --selftest
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import re
import sys
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "live/pm_research/contracts/contracts.yaml"
PROPOSAL = ROOT / "live/pm_research/plans/DE_REGISTRY_AMENDMENT_PROPOSAL.md"
CONTRACT_CHECK_DIR = ROOT / "live/pm_research/contracts"

# Every amendment block the proposal MUST carry.  A block that is absent makes
# the check FAIL and is NAMED -- it never silently shrinks the amendment.
REQUIRED_BLOCKS = ("A", "B", "C", "D", "E")
BLOCK_TARGET = {"A": "types", "B": "modules", "C": "config_supplied",
                "D": "modules", "E": "migration"}


class ProposalMalformed(RuntimeError):
    """The proposal file does not carry the blocks this check reads."""


# ---------------------------------------------------------------------------

def _load_contract_check():
    """Import DA/coordinator-owned `contract_check` WITHOUT copying it."""
    if str(CONTRACT_CHECK_DIR) not in sys.path:
        sys.path.insert(0, str(CONTRACT_CHECK_DIR))
    import contract_check                                    # noqa: E402
    return contract_check


def parse_amendments(text: str) -> dict[str, Any]:
    """Pull the fenced yaml blocks marked `# DE-AMENDMENT-x` out of the doc.

    REFUSES on a missing required block rather than returning what it found:
    a partial amendment that validates is the shape that reads as success."""
    out: dict[str, Any] = {}
    for m in re.finditer(r"```yaml\n(.*?)```", text, re.S):
        body = m.group(1)
        tag = re.match(r"\s*#\s*DE-AMENDMENT-([A-Z])\s*\n", body)
        if not tag:
            continue
        key = tag.group(1)
        if key in out:
            raise ProposalMalformed(
                f"duplicate amendment block {key!r}: two blocks under one "
                f"marker means the applied one is whichever parsed last")
        out[key] = yaml.safe_load(body[tag.end():])
    missing = [b for b in REQUIRED_BLOCKS if b not in out]
    if missing:
        raise ProposalMalformed(
            f"proposal is MISSING amendment block(s) {missing}. A check that "
            f"passes on the blocks that remain would report a shrunken "
            f"amendment as a clean one.")
    return out


def apply_amendments(doc: dict[str, Any], am: dict[str, Any]) -> dict[str, Any]:
    """Return an AMENDED COPY.  The input document is never mutated, and this
    function never writes a file -- `contracts.yaml` is not DE's to edit."""
    d = copy.deepcopy(doc)
    for name, block in (("A", am["A"]), ):
        for tname, tbody in block.items():
            if tname in d["types"]:
                raise ProposalMalformed(
                    f"amendment {name} would REDEFINE existing type {tname!r}; "
                    f"an additive block that overwrites is not additive")
            d["types"][tname] = tbody
    for name in ("B", "D"):
        for mname, mbody in am[name].items():
            if mname in d["modules"]:
                raise ProposalMalformed(
                    f"amendment {name} would REDEFINE existing module "
                    f"{mname!r}")
            d["modules"][mname] = mbody
    for t in am["C"]["config_supplied"]:
        if t not in d["config_supplied"]:
            d["config_supplied"].append(t)
    return d


def producer_gaps(doc: dict[str, Any]) -> list[str]:
    """Types named by a FIELD of a config-supplied type that no module
    produces and nothing supplies.

    `contract_check.invariants()` cannot see these: its producer check runs
    over `modules[*].consumes`, and a field of a config-supplied type is never
    asked who fills it.  That is how `DecisionProblem.actions: ActionSpace`
    reached v24 with no producer anywhere.

    SCOPE, STATED SO THE OUTPUT IS NOT OVERREAD: this set is LARGE and mostly
    benign -- a nested value type (`Position` inside `PortfolioState`) arrives
    WITH the config-supplied record that contains it and needs no producer of
    its own.  Membership here is therefore not by itself a defect.  What
    singles `ActionSpace` out is stated separately and checked separately: it
    is the one member the programme has BUILT A MODULE FOR
    (`de_actionspace.py`), so its emptiness is an unwired seam rather than a
    nested value."""
    types = doc.get("types") or {}
    mods = doc.get("modules") or {}
    cfg = set(doc.get("config_supplied") or [])
    produced: set[str] = set()
    for m in mods.values():
        p = m.get("produces")
        for item in (p if isinstance(p, list) else [p] if p else []):
            produced.add(str(item).split("[")[0])
    gaps: list[str] = []
    for tname in sorted(cfg & set(types)):
        for fname, ftype in (types[tname].get("fields") or {}).items():
            for ref in re.findall(r"[A-Z][A-Za-z0-9_]*", str(ftype)):
                if ref.isupper() or ref not in types:
                    continue
                if ref in produced or ref in cfg:
                    continue
                gaps.append(f"{tname}.{fname} -> {ref}")
    return sorted(set(gaps))


def claims(doc: dict[str, Any]) -> dict[str, Any]:
    """The proposal's factual claims about v24, each COMPUTED (rule 10)."""
    mods = doc.get("modules") or {}
    types = doc.get("types") or {}
    produced: set[str] = set()
    for m in mods.values():
        p = m.get("produces")
        for item in (p if isinstance(p, list) else [p] if p else []):
            produced.add(str(item).split("[")[0])
    return {
        "version": doc.get("version"),
        "ev_replay_module_absent": "EV-Replay" not in mods,
        "replay_types_absent": not [t for t in types
                                    if "Replay" in t or t == "RunRecord"],
        "de_actionspace_module_absent": "DE-ActionSpace" not in mods,
        "actionspace_type_present": "ActionSpace" in types,
        "actionspace_code_present": (ROOT / "live/pm_research/"
                                     "de_actionspace.py").exists(),
        "op_latencybudget_module_absent": "OP-LatencyBudget" not in mods,
        "op_monitor_module_present": "OP-Monitor" in mods,
        "actionspace_has_no_producer": "ActionSpace" not in produced,
        "venuecapabilities_has_no_producer":
            "VenueCapabilities" not in produced,
        "actionset_is_config_supplied":
            "ActionSet" in (doc.get("config_supplied") or []),
        "producer_gaps": producer_gaps(doc),
    }


def report() -> dict[str, Any]:
    cc = _load_contract_check()
    doc = yaml.safe_load(CONTRACTS.read_text())
    am = parse_amendments(PROPOSAL.read_text())
    amended = apply_amendments(doc, am)
    return {
        "as_of_source": "contracts.yaml v%s" % doc.get("version"),
        "n_amendment_blocks": len(am),
        "blocks": {k: BLOCK_TARGET[k] for k in sorted(am)},
        "claims_v24": claims(doc),
        "invariants_v24": cc.invariants(doc),
        "invariants_amended": cc.invariants(amended),
        "producer_gaps_v24": producer_gaps(doc),
        "producer_gaps_amended": producer_gaps(amended),
        "modules_added": sorted(set(amended["modules"]) - set(doc["modules"])),
        "types_added": sorted(set(amended["types"]) - set(doc["types"])),
    }


# ---------------------------------------------------------------------------

EXPECTED_CHECKS = 26


def selftest() -> int:
    n = [0]

    def ok(cond: Any, label: str) -> None:
        if not cond:
            raise SystemExit(f"[de_registry_amendment_check] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(exc, fn, label: str) -> None:
        try:
            fn()
        except exc:
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_registry_amendment_check] FAIL (did not "
                         f"refuse): {label}")

    cc = _load_contract_check()
    doc = yaml.safe_load(CONTRACTS.read_text())
    am = parse_amendments(PROPOSAL.read_text())

    # ---- the proposal's factual claims about v24 -----------------------
    c = claims(doc)
    ok(c["version"] == 24, f"contracts.yaml is v24 (read: {c['version']})")
    ok(c["ev_replay_module_absent"] and c["replay_types_absent"],
       "R-379 REPRODUCED: EV-Replay has neither a module entry nor any "
       "replay-vocabulary type in v24")
    ok(c["de_actionspace_module_absent"] and c["actionspace_type_present"]
       and c["actionspace_code_present"],
       "DE-ActionSpace: TYPE present, CODE present, MODULE absent")
    ok(c["op_latencybudget_module_absent"] and c["op_monitor_module_present"],
       "OP-LatencyBudget absent while OP-Monitor is present -- the contrast "
       "that makes 'deferred' the claim to defend, not 'forgotten'")
    ok(c["actionspace_has_no_producer"]
       and c["venuecapabilities_has_no_producer"],
       "NEW FINDING, measured over the whole modules block: NO module "
       "produces ActionSpace or VenueCapabilities in v24")
    ok(c["actionset_is_config_supplied"],
       "ActionSet is config_supplied in v24 -- which is why amendment D "
       "makes E non-additive rather than free")
    ok("DecisionProblem.actions -> ActionSpace" in c["producer_gaps"],
       "DecisionProblem.actions -> ActionSpace is NAMED by the detector, not "
       "asserted in prose")
    ok(len(c["producer_gaps"]) > 5,
       f"AND THE CLAIM IS SCOPED, not inflated: the same set holds "
       f"{len(c['producer_gaps'])} entries, most of them benign nested value "
       f"types -- membership alone is not the finding")
    ok(c["actionspace_code_present"] and c["actionspace_has_no_producer"]
       and c["de_actionspace_module_absent"],
       "WHAT SINGLES ActionSpace OUT, as a conjunction rather than a vibe: "
       "no producer AND built code AND no module record")

    # ---- the detector can also come back EMPTY (rule 16: it must admit) --
    clean = {"types": {"Parent": {"fields": {"x": "Child"}},
                       "Child": {"fields": {}}},
             "modules": {"M": {"produces": ["Child"]}},
             "config_supplied": ["Parent"]}
    ok(producer_gaps(clean) == [],
       "POSITIVE CONTROL: the gap detector ADMITS a document whose "
       "config-supplied field HAS a producer (it does not always fire)")
    dirty = {"types": {"Parent": {"fields": {"x": "Child"}},
                       "Child": {"fields": {}}},
             "modules": {"M": {"produces": ["Other"]}},
             "config_supplied": ["Parent"]}
    ok(producer_gaps(dirty) == ["Parent.x -> Child"],
       "KNOWN-BAD: the same detector FIRES when the producer is removed")

    # ---- the proposal parses, completely -------------------------------
    ok(sorted(am) == list(REQUIRED_BLOCKS),
       f"all {len(REQUIRED_BLOCKS)} amendment blocks parse out of the "
       f"proposal: {sorted(am)}")
    ok(set(am["A"]) == {"ReplayFill", "UnavailableInterval",
                        "ReplayWindowSpec", "RunRecord", "ReplayReceipt",
                        "GenerationTrancheTable", "GenerationTranche"},
       f"amendment A carries exactly the seven drafted types: {sorted(am['A'])}")
    ok(list(am["B"]) == ["EV-Replay"] and list(am["D"]) == ["DE-ActionSpace"],
       "amendments B and D each carry exactly their one module record")
    ok(am["D"]["DE-ActionSpace"]["produces"] == ["ActionSet"],
       "amendment D records DE-ActionSpace as producing ActionSet -- NOT the "
       "type ActionSpace, which is a verb menu the code never emits")
    ok(am["E"][0]["operation"] == "remove"
       and am["E"][0]["key"] == "config_supplied:ActionSet"
       and am["E"][0]["from_version"] == 24,
       "amendment E is a v24->v25 REMOVE record, migration-shaped")

    # a doctored proposal missing a block must REFUSE, not pass on the rest
    doctored = PROPOSAL.read_text().replace("# DE-AMENDMENT-D", "# NOT-A-TAG")
    refuses(ProposalMalformed, lambda: parse_amendments(doctored),
            "KNOWN-BAD: a proposal with a deleted block REFUSES -- it does "
            "not validate the blocks that remain")
    dup = PROPOSAL.read_text().replace(
        "# DE-AMENDMENT-C", "# DE-AMENDMENT-B", 1)
    refuses(ProposalMalformed, lambda: parse_amendments(dup),
            "KNOWN-BAD: two blocks under one marker REFUSE (the applied one "
            "would be whichever parsed last)")

    # ---- invariants, both directions -----------------------------------
    inv0 = cc.invariants(doc)
    ok(inv0 == [], f"POSITIVE CONTROL: unamended v24 passes contract_check "
                   f"invariants ({len(inv0)} errors)")
    amended = apply_amendments(doc, am)
    inv1 = cc.invariants(amended)
    ok(inv1 == [], f"the AMENDED document also passes -- checked against the "
                   f"same invariant set that just admitted v24 ({inv1})")
    ok(sorted(set(amended["modules"]) - set(doc["modules"]))
       == ["DE-ActionSpace", "EV-Replay"],
       "exactly two module records are added")

    # known-bad 1: an amendment naming an undeclared type must be caught
    bad_a = copy.deepcopy(am)
    bad_a["A"]["RunRecord"]["fields"]["fills"] = "list[NoSuchType]"
    bad_doc = apply_amendments(doc, bad_a)
    ok(any("NoSuchType" in e for e in cc.invariants(bad_doc)),
       "KNOWN-BAD: an amendment field naming an undeclared type raises "
       "unresolved-reference -- so the clean run above is not vacuous")

    # known-bad 2: a module consuming a type with no producer must be caught
    bad_b = copy.deepcopy(am)
    bad_b["C"] = {"config_supplied": []}      # withdraw the supply statement
    bad_doc2 = apply_amendments(doc, bad_b)
    errs = cc.invariants(bad_doc2)
    ok(any("ReplayWindowSpec" in e and "no declared producer" in e
           for e in errs),
       "KNOWN-BAD: withdrawing amendment C makes EV-Replay consume "
       "ReplayWindowSpec with no declared producer, and the invariant FIRES")

    # known-bad 3: redefining an existing type is not additive
    bad_c = copy.deepcopy(am)
    bad_c["A"]["ActionSpace"] = {"fields": {}}
    refuses(ProposalMalformed, lambda: apply_amendments(doc, bad_c),
            "KNOWN-BAD: an 'additive' block that redefines an existing type "
            "is REFUSED")

    # ---- the amendment does NOT quietly close the gap it only reports ---
    ok("DecisionProblem.actions -> ActionSpace" in producer_gaps(amended),
       "the ownerless ActionSpace field is STILL open after the amendment -- "
       "the proposal reports it and does not silently fix a design decision "
       "that is not DE's to make")

    ok(n[0] + 1 == EXPECTED_CHECKS,          # +1: this assertion is a check
       f"check count asserted at run time, not remembered in prose: "
       f"{n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_registry_amendment_check] selftest OK -- {n[0]} checks")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.report:
        print(json.dumps(report(), indent=2, sort_keys=True))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
