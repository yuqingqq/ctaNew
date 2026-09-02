"""Checker behind DE_REGISTRY_AMENDMENT_PROPOSAL.md -- predicates, not prose.

SURFACE AUTHORISATION (R-126, in-file): R-379 TASK 2 (registry closure),
DE seat. RESEARCH-ONLY, OFFLINE. This module READS `contracts/contracts.yaml`
and NEVER writes it: the registry wins on types and an amendment lands by
coordinator/USER act (SEAT_PROTOCOL rule 4 / the proposal's own header).

TWO DOCUMENTS, AND EVERY CLAIM SAYS WHICH ONE.  The amendments have LANDED
(A-D at v25, E at v26 on BE's Q-BE-222 confirmation), so the working tree no
longer holds the document they apply to.  The BASELINE is read from a pinned
commit and REFUSES if that commit does not carry v24; the LIVE registry is
read from the tree.  A checker that kept treating the tree as its baseline
would go on asserting v24's absences against v26 -- right sentences, wrong
document, which is rule 16's know-what-KIND-of-document-you-are-reading trap
turned on the instrument itself.

THE APPLIER KNOWS THE REMOVAL SHAPE (RR4-4).  It once knew only the additive
blocks, so the reference application of A-E on v24 still contained
`ActionSet` and the equality control was structurally unable to certify the
one NON-additive amendment.  It now applies migration records too, and the
full chain v24 + A-E reproduces the LIVE v26 exactly on the sections the
amendment touches.  An operation or namespace it does not implement REFUSES
rather than being skipped: a skipped record reproduces LESS than the real
application, and the equality would then pass on a subset.

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
import subprocess
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

# THE BASELINE IS PINNED TO A COMMIT, NOT READ FROM THE TREE.
# The proposal was drafted against v24 and the amendments have since LANDED
# (A-D at v25, E at v26), so the working tree no longer holds the document
# the amendment applies to.  A checker that kept reading the tree would go on
# asserting v24's absences against v26 and be stale-true: right sentences,
# wrong document -- rule 16's know-what-KIND-of-document-you-are-reading trap,
# in the instrument's own chair.  So there are TWO documents here and each
# claim says which one it is about.
BASELINE_REF = "b6231c8"          # the commit whose contracts.yaml is v24
BASELINE_VERSION = 24
# Sections the amendment touches; equality is asserted over exactly these.
AMENDED_SECTIONS = ("types", "modules", "config_supplied")

# The migration-record shapes the reference applier understands.  An
# operation outside this set REFUSES rather than being skipped: a skipped
# record is a reference application that silently does less than the real
# one, which is the defect RR4-4 names.
MIGRATION_OPERATIONS = ("remove",)
MIGRATION_NAMESPACES = ("config_supplied",)


class MigrationRefused(RuntimeError):
    """A migration record the reference applier will not guess at."""


class BaselineRefused(RuntimeError):
    """The pinned baseline is not the document the proposal was drafted
    against.  Refusing beats silently certifying against the wrong version."""


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


def load_baseline(ref: str = BASELINE_REF) -> dict[str, Any]:
    """The v24 document, read from its COMMIT.

    REFUSES if the ref does not carry the expected version: a pin that
    silently drifted would make every equality below certify the wrong
    comparison, and a wrong comparison that passes is worse than none."""
    r = subprocess.run(["git", "show", f"{ref}:live/pm_research/contracts/"
                        f"contracts.yaml"], capture_output=True, text=True,
                       cwd=str(ROOT), timeout=60)
    if r.returncode != 0:
        raise BaselineRefused(
            f"cannot read contracts.yaml at {ref!r}: {r.stderr.strip()}")
    doc = yaml.safe_load(r.stdout)
    if doc.get("version") != BASELINE_VERSION:
        raise BaselineRefused(
            f"{ref!r} carries version {doc.get('version')!r}, not the "
            f"baseline {BASELINE_VERSION} the proposal was drafted against")
    return doc


def apply_migration_record(d: dict[str, Any], rec: dict[str, Any]) -> None:
    """Apply ONE migration record in place.  The non-additive shape.

    RR4-4: the applier knew only the additive blocks, so the full chain could
    not reproduce v26 and the equality control was structurally unable to
    certify the one amendment that removes something.  Version fields are
    read but NOT depended on -- E was drafted 24->25 and applied 25->26
    because A-D consumed that step, and the OPERATION is what reproduces the
    document; the version adaptation is asserted separately."""
    for k in ("operation", "key"):
        if k not in rec:
            raise MigrationRefused(f"migration record is MISSING {k!r}")
    op = rec["operation"]
    if op not in MIGRATION_OPERATIONS:
        raise MigrationRefused(
            f"operation {op!r} is not one this reference applier implements "
            f"({MIGRATION_OPERATIONS}); refusing rather than skipping it, "
            f"because a skipped record reproduces LESS than the real "
            f"application and the equality would then pass on a subset")
    key = str(rec["key"])
    if ":" not in key:
        raise MigrationRefused(
            f"migration key {key!r} carries no namespace; expected "
            f"'<namespace>:<member>'")
    ns, member = key.split(":", 1)
    if ns not in MIGRATION_NAMESPACES:
        raise MigrationRefused(
            f"namespace {ns!r} is not one this applier implements "
            f"({MIGRATION_NAMESPACES})")
    if op == "remove":
        if member not in d[ns]:
            raise MigrationRefused(
                f"REFUSING: {key!r} removes {member!r} from {ns}, which is "
                f"NOT THERE. A removal that finds nothing has either already "
                f"been applied or names the wrong member, and both are "
                f"reasons to stop rather than to continue as if it worked.")
        d[ns].remove(member)


def apply_amendments(doc: dict[str, Any], am: dict[str, Any]) -> dict[str, Any]:
    """Return an AMENDED COPY, additive blocks AND migration records.  The
    input document is never mutated, and this function never writes a file --
    `contracts.yaml` is not DE's to edit."""
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
    for rec in am["E"]:                       # the non-additive shape
        apply_migration_record(d, rec)
    return d


def sections(doc: dict[str, Any]) -> dict[str, Any]:
    """The sections the amendment touches, normalised for comparison."""
    return {
        "types": {k: doc["types"][k] for k in sorted(doc["types"])},
        "modules": {k: doc["modules"][k] for k in sorted(doc["modules"])},
        "config_supplied": sorted(doc["config_supplied"]),
    }


def section_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Where two documents differ on the amended sections.  Empty = equal."""
    sa, sb = sections(a), sections(b)
    out: dict[str, Any] = {}
    for k in AMENDED_SECTIONS:
        if k == "config_supplied":
            only_a = sorted(set(sa[k]) - set(sb[k]))
            only_b = sorted(set(sb[k]) - set(sa[k]))
        else:
            only_a = sorted(set(sa[k]) - set(sb[k]))
            only_b = sorted(set(sb[k]) - set(sa[k]))
            changed = sorted(x for x in set(sa[k]) & set(sb[k])
                             if sa[k][x] != sb[k][x])
            if changed:
                out[f"{k}_changed"] = changed
        if only_a:
            out[f"{k}_only_in_first"] = only_a
        if only_b:
            out[f"{k}_only_in_second"] = only_b
    return out


def config_supplied_is_invisible_to_contract_check(cc, doc) -> bool:
    """The review's own note, as a predicate rather than a remark.

    `contract_check.flatten` covers types, modules, ports and rules -- NOT
    `config_supplied` membership -- so a removal there produces no REMOVED
    row and "invariants CLEAN" is not evidence that E landed.  That is
    precisely why this instrument's equality control has to cover it."""
    flat = cc.flatten(doc)
    return not any("config_supplied" in k for k in flat)


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
    """The proposal's factual claims, each COMPUTED (rule 10), about
    WHICHEVER document is passed in.

    The function was always version-agnostic; its CALLERS were not -- they
    fed it the working tree and read the answers as v24's.  Every claim is now
    evaluated twice, once against the pinned baseline and once against the
    live registry, and the two are asserted separately."""
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


def applied_migration_records(key: str) -> list[dict[str, Any]]:
    """The LIVE migration records naming `key`, read from the applied file."""
    doc = yaml.safe_load((CONTRACT_CHECK_DIR / "migrations.yaml").read_text())
    recs = doc if isinstance(doc, list) else doc.get("migrations", [])
    return [r for r in recs if str(r.get("key")) == key]


def produced_at() -> dict[str, Any]:
    """The report's own commit binding, in the form rounds 2-3 established: a
    commit cannot contain its own id, so this records the commit the
    INSTRUMENT ran at plus the dirty paths, and a reader binds by re-reading
    the baseline at `BASELINE_REF` and the live file at the carrying commit."""
    def _git(*a, strip=True):
        r = subprocess.run(("git",) + a, capture_output=True, text=True,
                           cwd=str(ROOT), timeout=30)
        if r.returncode != 0:
            return None
        return r.stdout.strip() if strip else r.stdout.rstrip("\n")
    head = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain", strip=False)
    if head is None or porcelain is None:
        return {"produced_at_commit": None, "git_readable": False,
                "note": "git unreadable; the binding is UNKNOWN, which is not "
                        "the same as clean"}
    paths = sorted(l[3:].strip() for l in porcelain.split("\n")
                   if len(l) > 3)
    return {"produced_at_commit": head, "git_readable": True,
            "baseline_ref": BASELINE_REF,
            "working_tree_dirty": bool(paths), "dirty_paths": paths[:40],
            "note": "produced_at_commit is the commit the INSTRUMENT ran at; "
                    "the carrying commit does not exist at emission"}


def report() -> dict[str, Any]:
    cc = _load_contract_check()
    live = yaml.safe_load(CONTRACTS.read_text())
    base = load_baseline()
    am = parse_amendments(PROPOSAL.read_text())
    amended = apply_amendments(base, am)
    return {
        "baseline": {"ref": BASELINE_REF,
                     "version": base.get("version")},
        "live": {"path": str(CONTRACTS.relative_to(ROOT)),
                 "version": live.get("version")},
        "n_amendment_blocks": len(am),
        "blocks": {k: BLOCK_TARGET[k] for k in sorted(am)},
        "claims_baseline": claims(base),
        "claims_live": claims(live),
        "reference_application_equals_live": section_diff(amended, live) == {},
        "section_diff_reference_vs_live": section_diff(amended, live),
        "invariants_baseline": cc.invariants(base),
        "invariants_reference": cc.invariants(amended),
        "invariants_live": cc.invariants(live),
        "producer_gaps_baseline": producer_gaps(base),
        "producer_gaps_live": producer_gaps(live),
        "modules_added": sorted(set(amended["modules"]) - set(base["modules"])),
        "types_added": sorted(set(amended["types"]) - set(base["types"])),
        "config_supplied_removed": sorted(set(base["config_supplied"])
                                          - set(amended["config_supplied"])),
        "applied_migration_records_for_E":
            applied_migration_records("config_supplied:ActionSet"),
        "config_supplied_invisible_to_contract_check":
            config_supplied_is_invisible_to_contract_check(cc, live),
        "produced_at": produced_at(),
    }


# ---------------------------------------------------------------------------

EXPECTED_CHECKS = 44


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
    live = yaml.safe_load(CONTRACTS.read_text())
    base = load_baseline()
    doc = base                    # the document the amendment applies TO
    am = parse_amendments(PROPOSAL.read_text())

    # ---- TWO documents, and every claim says which one -----------------
    c = claims(base)
    cl = claims(live)
    ok(c["version"] == BASELINE_VERSION and cl["version"] >= 26,
       f"BASELINE v{c['version']} (pinned at {BASELINE_REF}) and LIVE "
       f"v{cl['version']} are read as two different documents -- the "
       f"amendments have LANDED, so a checker still reading the tree as its "
       f"baseline would assert v24's absences against v26")
    refuses(BaselineRefused, lambda: load_baseline("HEAD"),
            "KNOWN-BAD: a baseline ref that does not carry v24 is REFUSED "
            "rather than silently certifying the wrong comparison")

    # what the proposal FOUND, asserted against the document it found it in
    ok(c["ev_replay_module_absent"] and c["replay_types_absent"],
       "AT THE BASELINE, R-379 reproduced: EV-Replay had neither a module "
       "entry nor any replay-vocabulary type")
    ok(c["de_actionspace_module_absent"] and c["actionspace_type_present"]
       and c["actionspace_code_present"],
       "AT THE BASELINE, DE-ActionSpace: TYPE present, CODE present, MODULE "
       "absent")
    ok(c["actionspace_has_no_producer"]
       and c["venuecapabilities_has_no_producer"]
       and c["actionset_is_config_supplied"],
       "AT THE BASELINE: no producer for ActionSpace or VenueCapabilities, "
       "and ActionSet config_supplied -- which is what made E non-additive")

    # what is TRUE NOW, asserted against the applied registry
    ok(not cl["ev_replay_module_absent"] and not cl["replay_types_absent"]
       and not cl["de_actionspace_module_absent"],
       "AT THE LIVE REGISTRY the amendments are LAW: EV-Replay and "
       "DE-ActionSpace are registered modules and the replay types exist")
    ok(not cl["actionset_is_config_supplied"],
       "AND E LANDED: ActionSet is no longer config_supplied, so the type "
       "has one authority instead of two")
    ok(cl["op_latencybudget_module_absent"] and cl["op_monitor_module_present"],
       "OP-LatencyBudget is STILL absent beside a present OP-Monitor -- "
       "deferred-with-trigger, unchanged by this amendment")
    ok(cl["actionspace_has_no_producer"]
       and "DecisionProblem.actions -> ActionSpace" in cl["producer_gaps"],
       "and the ownerless ActionSpace field is STILL OPEN at v26: the "
       "amendment reported it and did not quietly close a design decision "
       "that was never DE's to make")
    ok("DecisionProblem.actions -> ActionSpace" in c["producer_gaps"],
       "AT THE BASELINE, DecisionProblem.actions -> ActionSpace is NAMED by "
       "the detector, not asserted in prose")
    ok(len(c["producer_gaps"]) > 5,
       f"AND THE CLAIM IS SCOPED, not inflated: at the baseline the same set "
       f"holds {len(c['producer_gaps'])} entries, most of them benign nested "
       f"value types -- membership alone is not the finding")
    ok(c["actionspace_code_present"] and c["actionspace_has_no_producer"]
       and c["de_actionspace_module_absent"],
       "WHAT SINGLED ActionSpace OUT AT THE BASELINE, as a conjunction "
       "rather than a vibe: no producer AND built code AND no module record. "
       "One conjunct has since changed -- DE-ActionSpace IS registered now -- "
       "which is why this claim names the document it is about")

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

    # ---- RR4-4: the REMOVAL shape, and the full-chain equality ---------
    # The applier knew only the additive blocks, so the reference application
    # of A-E on v24 still contained ActionSet and the equality control was
    # structurally unable to certify the one non-additive amendment.
    ok(am["E"][0]["operation"] == "remove"
       and am["E"][0]["key"] == "config_supplied:ActionSet",
       "amendment E parses to the removal shape the applier now implements")
    additive_only = copy.deepcopy(am)
    additive_only["E"] = []
    ok("ActionSet" in apply_amendments(doc, additive_only)["config_supplied"],
       "REPRODUCING RR4-4: with E withheld the reference application still "
       "carries ActionSet -- which is exactly what the old applier did on "
       "every run, silently")
    ok("ActionSet" not in apply_amendments(doc, am)["config_supplied"],
       "and with E applied it does not: the removal is what removes it, not "
       "a side effect of the additive blocks")

    # THE FULL CHAIN REPRODUCES THE LIVE REGISTRY, on the sections the
    # amendment touches -- the same equality the coordinator ran at R-397,
    # now covering every amendment class instead of only the additive ones.
    diff = section_diff(amended, live)
    ok(diff == {},
       f"FULL CHAIN: baseline v{c['version']} + A-E reproduces the LIVE "
       f"v{cl['version']} EXACTLY on {list(AMENDED_SECTIONS)} -- "
       f"section_diff {diff}")
    ok(sorted(set(base["config_supplied"]) - set(amended["config_supplied"]))
       == ["ActionSet"]
       and sorted(set(amended["config_supplied"])
                  - set(base["config_supplied"])) == ["ReplayWindowSpec"],
       "and the config_supplied delta is exactly C's addition and E's "
       "removal, in both directions")
    # STATED EXACTLY, not as "non-empty".  A disjunctive predicate here would
    # be satisfied by almost any difference, and a control loose enough to be
    # satisfied by a coincidence proves nothing when it passes.
    d_no_e = section_diff(apply_amendments(doc, additive_only), live)
    ok(d_no_e.get("config_supplied_only_in_first") == ["ActionSet"]
       and set(d_no_e) == {"config_supplied_only_in_first"},
       f"KNOWN-BAD ON THE EQUALITY ITSELF: withholding E makes the same "
       f"comparison differ, and differ in EXACTLY one named way -- ActionSet "
       f"present in the reference and absent from live ({d_no_e}). The "
       f"control can fail, and this is the shape it fails in.")

    # red-first on the removal shape itself, both directions
    refuses(MigrationRefused, lambda: apply_migration_record(
        copy.deepcopy(live), {"operation": "remove",
                              "key": "config_supplied:ActionSet"}),
        "KNOWN-BAD: removing a key that is NOT in config_supplied REFUSES -- "
        "a removal that finds nothing has either already been applied or "
        "names the wrong member, and both are reasons to stop")
    refuses(MigrationRefused, lambda: apply_migration_record(
        copy.deepcopy(doc), {"operation": "change",
                             "key": "config_supplied:ActionSet"}),
        "KNOWN-BAD: an operation the applier does not implement REFUSES "
        "rather than being SKIPPED -- a skipped record reproduces less than "
        "the real application and the equality would pass on a subset")
    refuses(MigrationRefused, lambda: apply_migration_record(
        copy.deepcopy(doc), {"operation": "remove", "key": "types:RunRecord"}),
        "KNOWN-BAD: a namespace the applier does not implement REFUSES")
    refuses(MigrationRefused, lambda: apply_migration_record(
        copy.deepcopy(doc), {"operation": "remove"}),
        "KNOWN-BAD: a migration record missing its key REFUSES")
    ok(apply_migration_record(copy.deepcopy(doc), am["E"][0]) is None,
       "POSITIVE CONTROL: the REAL amendment E applies cleanly to the "
       "baseline")

    # the version adaptation, asserted rather than tripped over
    applied_e = applied_migration_records("config_supplied:ActionSet")
    ok(len(applied_e) == 1
       and applied_e[0]["operation"] == am["E"][0]["operation"]
       and applied_e[0]["old"] == am["E"][0]["old"]
       and applied_e[0]["new"] == am["E"][0]["new"],
       "the LIVE migration record matches the drafted one on operation, key, "
       "old and new")
    ok((am["E"][0]["from_version"], am["E"][0]["to_version"]) == (24, 25)
       and (applied_e[0]["from_version"], applied_e[0]["to_version"]) == (25, 26),
       "AND THE VERSION ADAPTATION IS STATED, not tripped over: E was "
       "drafted 24->25 and applied 25->26 because A-D consumed that step. "
       "The applier reads the version fields and does NOT depend on them -- "
       "the OPERATION is what reproduces the document")

    # why this instrument has to cover it at all
    ok(config_supplied_is_invisible_to_contract_check(cc, live),
       "AND THE REASON THIS CONTROL IS NEEDED, as a predicate: "
       "contract_check.flatten covers types/modules/ports/rules and NOT "
       "config_supplied membership, so E's removal produces no REMOVED row "
       "and 'invariants CLEAN' is NOT evidence that E landed")
    ok(cc.invariants(live) == [],
       "POSITIVE CONTROL: the live registry's invariants ARE clean -- true, "
       "and now correctly reported as insufficient rather than as proof")

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
