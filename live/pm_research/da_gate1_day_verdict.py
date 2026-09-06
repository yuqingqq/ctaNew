"""P-2026-003 Gate-1 DAY-VERDICT VERIFIER -- DA's independent stack.

R-235 (do-not-harmonize): this module RE-IMPLEMENTS the Gate-1 day statistic
from DE's and BE's *declarations* and never imports their computation. It
imports neither `de_multiday_gate1_runner`'s statistic nor
`be_cancel_axis_null`; both are read as documents and re-derived here. Two
implementations that agree are evidence; one implementation checking itself
is not.

WHAT IS RE-IMPLEMENTED HERE, and where each definition was read:

  seed rule        `de_multiday_gate1_runner.seed_for`:
                   int(sha256(f"{book_sha}|{arm}|P003_GATE1_MULTIDAY")[:8], 16)
  valuation        `de_phase4_diag_runner.fill_value_cents`: the maker P&L at
                   level-to-markout, NO fee term --
                   sgn * (mid_cents_at_markout - px_cents) * size,
                   sgn = +1 on SIDES[0], and None when a leg is missing
  decisions        the arm's above-threshold generations at its FIXED theta
  matched null     `be_cancel_axis_null.draw_flags`: per side, k without
                   replacement from that side's pool, sides consumed in
                   SORTED key order
  D(E0)            value(arm's fills) - value(baseline fills), cents
  p                (1 + #{null >= observed}) / (1 + K), one-sided, larger
                   is better
  Z                (observed - mean(null)) / pstdev(null)
  R4               decisions >= 30 AND sd >= 0.25 * |mean|

WHAT IS **NOT** RE-IMPLEMENTED, STATED RATHER THAN HIDDEN. The policy REPLAY
(`harmful_stateful_policy.replay_policy`) is the instrument of record, not a
statistic. A second policy engine would not verify the first -- it would
measure a different thing and call the disagreement a finding. So the replay
enters this module as a SEAM: a callable the caller supplies. In production it
is the engine of record; in the fixture it is a synthetic replay whose D(E0)
is known in closed form. What this verifier checks is the STATISTIC, the
SEEDING, the ALLOCATION, the VALUATION and the four predicates.

WHAT THE FIXTURE THEREFORE DOES AND DOES NOT ESTABLISH, said plainly: it
establishes that the statistic, the seed, the matched allocation, the
valuation and the four predicates are right, against a book whose D(E0) is an
identity. It establishes NOTHING about the policy engine's cascade -- and a
fixture whose replay dropped exactly the cancelled rows could not tell a
correct statistic from one that had assumed that cascade, so the battery also
drives a CASCADING replay where the naive identity is false.

AND ONE THING THIS VERIFIER REFUSES TO PRETEND. DE's day receipt is SEALED
until G is complete: `_strip_economic` removes D_E0, Z, p_location, null_mean,
null_sd and null_draws_summary at every depth. Against a sealed receipt the
economic comparison IS NOT POSSIBLE, and a verifier that answered "all checks
passed" would be certifying nothing at all. A sealed receipt therefore yields
the status ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED and the report is NOT a
verification -- driven both ways in the battery.

    python3 live/pm_research/da_gate1_day_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROTOCOL = "P003_DA_GATE1_DAY_VERDICT_VERIFIER_V1"

#: The seed tag, copied from the rule as a STRING because that is what the
#: hash eats. If DE changes the tag the seeds move and this verifier must
#: disagree loudly rather than follow.
SEED_TAG = "P003_GATE1_MULTIDAY"

#: Read from the params declaration at load time -- never hardcoded here,
#: because a bar retyped into a checker is a second source of truth.
#: v5 (REV 45 section 3.3): v4 is SUPERSEDED and a checker left pinned to a
#: superseded declaration reads bars nobody is running under.
def _params_versions() -> list:
    return sorted(int(f.stem.rsplit("_v", 1)[-1])
                  for f in (HERE / "declarations").glob(
                      "de_multiday_gate1_params_v*.json")
                  if f.stem.rsplit("_v", 1)[-1].isdigit())


def _params_is_newest(params: dict) -> bool:
    v = _params_versions()
    return bool(v and Path(params["_path"]).stem.endswith(f"_v{max(v)}"))


class VerifierRefused(RuntimeError):
    """The verification cannot proceed honestly on the inputs given."""


#: R-673(b). ***A SEARCH BOUNDED TO TWO SHAPES LEAVES THE CLAIM READING
#: TRUE.*** The producing-place scan understood exactly `NAME = ...` and a
#: dict key, so an annotated assignment, a tuple target, a walrus, a
#: subscript target and an augmented assignment -- ALL FIVE IN LIVE USE in
#: this tree -- each made "the runner produces it in ZERO places" true by
#: not being looked at. The scan is over STORE CONTEXTS now, which is what
#: a binding IS, and ***a shape it does not understand REFUSES rather than
#: returning a zero***.
_BINDING_STMTS = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)


def binding_sites(tree, name: str) -> dict:
    """Every place `name` is BOUND, every place it is only NAMED, and every
    shape this scan does not understand."""
    parents = {}
    for nd in ast.walk(tree):
        for c in ast.iter_child_nodes(nd):
            parents[c] = nd
    binds, declares, unhandled = [], [], []
    for n in ast.walk(tree):
        #: a STORE of the identifier, in any statement shape
        if isinstance(n, ast.Name) and n.id == name and isinstance(
                n.ctx, ast.Store):
            cur, depth, owner = parents.get(n), 0, None
            while cur is not None and depth < 6:
                depth += 1
                if isinstance(cur, _BINDING_STMTS + (
                        ast.For, ast.AsyncFor, ast.With, ast.AsyncWith,
                        ast.comprehension, ast.Try, ast.FunctionDef,
                        ast.AsyncFunctionDef, ast.ClassDef, ast.Import,
                        ast.ImportFrom, ast.Global, ast.Nonlocal)):
                    owner = cur
                    break
                cur = parents.get(cur)
            kind = type(owner).__name__ if owner is not None else None
            if isinstance(owner, ast.AnnAssign) and owner.value is None:
                declares.append({"line": n.lineno, "shape": "AnnAssign "
                                                            "without a "
                                                            "value"})
            elif isinstance(owner, _BINDING_STMTS):
                binds.append({"line": n.lineno, "shape": kind})
            else:
                unhandled.append({"line": n.lineno,
                                  "shape": kind or "no enclosing statement "
                                                   "within 6 hops"})
        #: a SUBSCRIPT or ATTRIBUTE store on the name (`X["k"] = ...`)
        if isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(
                getattr(n, "ctx", None), ast.Store):
            base = n
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name) and base.id == name:
                binds.append({"line": n.lineno,
                              "shape": f"{type(n).__name__} target"})
        #: `out["D_E_MINUS_R"] = <expr>` -- the shape a receipt field is
        #: most likely to be EMITTED in, and one a Dict-literal scan cannot
        #: see. Same rule as the literal: a constant is a table entry, an
        #: expression is an emission.
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if not (isinstance(t, ast.Subscript)
                        and isinstance(t.slice, ast.Constant)
                        and t.slice.value == name):
                    continue
                if isinstance(n.value, ast.Constant) and isinstance(
                        n.value.value, (int, float, str, bool)):
                    declares.append({"line": t.lineno,
                                     "shape": "subscript key with a "
                                              "constant value"})
                else:
                    binds.append({"line": t.lineno,
                                  "shape": "subscript key with an "
                                           "expression"})
        #: a dict ENTRY: a constant beside the name is a TABLE entry
        #: (DE's per-name seal scope says `D_E_MINUS_R: 1`, a VERSION);
        #: an expression beside it is an EMISSION.
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if not (isinstance(k, ast.Constant) and k.value == name):
                    continue
                if isinstance(v, ast.Constant) and isinstance(
                        v.value, (int, float, str, bool)):
                    declares.append({"line": k.lineno,
                                     "shape": "dict entry with a constant "
                                              "value"})
                else:
                    binds.append({"line": k.lineno,
                                  "shape": "dict entry with an expression"})
    return {"n_binds": len(binds), "binds": binds,
            "n_declares": len(declares), "declares": declares,
            "unhandled": unhandled,
            "shapes_understood": [
                "Assign (Name, tuple/list element, subscript, attribute)",
                "AnnAssign (with a value; without one it DECLARES)",
                "AugAssign", "NamedExpr (walrus)",
                'dict entry and `x["NAME"] = ...` (constant = a table '
                'entry, expression = an emission)'],
            "why_it_refuses": (
                "a shape this scan does not understand would make 'the "
                "runner produces it in ZERO places' true by not being "
                "looked at")}


def _newest_params() -> Path:
    """The HIGHEST-numbered params declaration present.

    REV 45 section 3.3 taught this once (v4 -> v5): a checker pinned to a
    SUPERSEDED declaration reads bars nobody is running under. Pinning the
    filename taught it again this round -- v5 predates `read_gate`, so the
    conjunction this verifier is required to evaluate was not in the file it
    was pinned to. The version is resolved and RECORDED in every emission
    rather than typed, and a check asserts it is the newest present."""
    d = HERE / "declarations"
    best, best_n = None, -1
    for f in d.glob("de_multiday_gate1_params_v*.json"):
        try:
            n = int(f.stem.rsplit("_v", 1)[-1])
        except ValueError:
            continue
        if n > best_n:
            best, best_n = f, n
    #: DA 95, FOUND BY THIS SEAT'S OWN CENSUS once its dataflow gate took
    #: the TRANSITIVE closure (REV 73 S2(b)): the old tail was
    #: `best or (d / "…_params_v6.json")` -- ***a stale default that would
    #: be OPENED whenever the glob found nothing***, while the head is
    #: v14. A fallback is a fallback in the shape a one-hop gate cannot
    #: see: the literal is returned, the return lands in `PARAMS_PATH`,
    #: and `PARAMS_PATH` is opened. ***An absence is a named refusal, not
    #: a default*** (rule 11): reading a version nobody is running under
    #: is the exact failure this verifier exists to catch elsewhere.
    if best is None:
        raise VerifierRefused(
            f"REFUSED: NO_PARAMS_DECLARATION_PRESENT -- no "
            f"`de_multiday_gate1_params_v*.json` under {d}. This verifier "
            f"will not fall back to a version number typed into its own "
            f"source; a missing declaration is a named absence.")
    return best


PARAMS_PATH = _newest_params()

#: DE's runner, read as a DOCUMENT so its economic field list can be taken
#: from the source rather than from a copy in this file.
DE_RUNNER_PATH = HERE / "de_multiday_gate1_runner.py"

#: `harmful_stateful_policy.SIDES[0]` -- the sign convention the valuation
#: turns on. Read from the module (a constant, not a computation).
BUY_SIDE = "BUY_UP"

#: The seven the list has carried since it was written. Pinned as a FLOOR,
#: never as the count: a field LEAVING the list would unseal a quantity and
#: must be caught, while DE adding one is DE's to do -- and DE 85 did, under
#: R-599, sealing `sd_over_abs_mean`.
SEVEN_ORIGINAL_ECONOMIC_FIELDS = (
    "D_E0", "D_E_MINUS_R", "Z", "p_location", "null_mean", "null_sd",
    "null_draws_summary")


def de_economic_fields_at_source(path: Path | None = None) -> dict:
    """DE's ECONOMIC_FIELDS, read FROM THE SOURCE by AST -- never imported,
    never copied into this file (REV 45 section 3.3).

    A copy drifts silently: DE adds a field to the list, the stripper starts
    removing it, and a verifier holding last week's tuple reports a receipt
    OPEN that is in fact sealed on that field. Reading the constant is not
    enough either, so this ALSO asserts that `_strip_economic` -- the
    function that actually removes them -- references that same name. A
    constant nothing uses would pin nothing.
    """
    src = Path(path) if path else DE_RUNNER_PATH
    if not src.is_file():
        raise VerifierRefused(
            f"REFUSED: DE's runner is absent at {src}; the economic field "
            f"list cannot be read at its source and MUST NOT be guessed.")
    raw = src.read_bytes()
    tree = ast.parse(raw.decode())
    fields = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "ECONOMIC_FIELDS":
                    fields = tuple(ast.literal_eval(node.value))
    stripper_uses_it = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_strip_economic":
            stripper_uses_it = any(
                isinstance(x, ast.Name) and x.id == "ECONOMIC_FIELDS"
                for x in ast.walk(node))
    if fields is None:
        raise VerifierRefused(
            "REFUSED: ECONOMIC_FIELDS is not a module-level assignment in "
            "DE's runner. The list this verifier detects a seal by must come "
            "from the code that does the sealing.")
    if not stripper_uses_it:
        raise VerifierRefused(
            "REFUSED: `_strip_economic` does not reference ECONOMIC_FIELDS. "
            "The constant would then pin nothing -- the stripper could be "
            "removing a different set entirely.")
    return {"fields": fields,
            "source_path": "live/pm_research/de_multiday_gate1_runner.py",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "read_by": "ast, at the source; not imported and not copied",
            "stripper_references_the_same_name": True}


#: Resolved once at import from DE's source. Their ABSENCE in a receipt is
#: what tells this verifier the receipt is sealed.
ECONOMIC_FIELDS = de_economic_fields_at_source()["fields"]

#: DE's DESIGN DECLARATION module, read as a DOCUMENT for one number: the
#: design VERSION a given commit's tree was under.
DE_DESIGN_PATH = HERE / "de_multiday_design_declaration.py"
DE_DESIGN_REL = "live/pm_research/de_multiday_design_declaration.py"


def de_sealed_from_design_version_at_source(path: Path | None = None) -> dict:
    """DE's `SEALED_FROM_DESIGN_VERSION`, read FROM THE SOURCE by AST.

    ***THE SEAM REV 72 S1.4 PREDICTED, AND IT WAS LIVE ON MY SIDE (R-663).***
    DE scoped the seal PER NAME when the list grew from eight to eleven --
    `_strip_economic` seals by the list in force for THIS run, and a
    verifier must judge a receipt by the list in force WHEN THAT RECEIPT WAS
    PRODUCED. This verifier read the list FLAT and judged the programme's
    first sealed day against all eleven: `sealed False, n_leaked_fields 6`.
    ***An instrument that reaches backwards convicts the past of not having
    obeyed a future rule*** -- and it accused the one day the seal had
    actually held.

    Read, never imported (R-235): DE's map is a DECLARATION, and the
    judgement built on it here is this seat's own."""
    src = Path(path) if path else DE_RUNNER_PATH
    if not src.is_file():
        raise VerifierRefused(
            f"REFUSED: DE's runner is absent at {src}; the per-name seal "
            f"scope cannot be read at its source and MUST NOT be guessed.")
    raw = src.read_bytes()
    tree = ast.parse(raw.decode())
    scope, in_force = None, None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if not isinstance(t, ast.Name):
                continue
            if t.id == "SEALED_FROM_DESIGN_VERSION":
                scope = {str(k): int(v)
                         for k, v in ast.literal_eval(node.value).items()}
            elif t.id == "DESIGN_VERSION_IN_FORCE":
                in_force = int(ast.literal_eval(node.value))
    if scope is None:
        raise VerifierRefused(
            "REFUSED: SEALED_FROM_DESIGN_VERSION is not a module-level "
            "assignment in DE's runner. The scope this verifier judges a "
            "receipt by must come from the code that does the sealing.")
    #: THE SAME GUARD THE FLAT LIST CARRIES: a map nothing consults pins
    #: nothing. DE's own selector must reference the name.
    used_by = sorted(
        n.name for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and any(isinstance(x, ast.Name)
                and x.id == "SEALED_FROM_DESIGN_VERSION"
                for x in ast.walk(n)))
    if not used_by:
        raise VerifierRefused(
            "REFUSED: no function in DE's runner references "
            "SEALED_FROM_DESIGN_VERSION. A scope map nothing consults "
            "would pin nothing -- the stripper could be sealing by a "
            "different rule entirely.")
    missing = [f for f in ECONOMIC_FIELDS if f not in scope]
    if missing:
        raise VerifierRefused(
            f"REFUSED: {len(missing)} name(s) in ECONOMIC_FIELDS carry no "
            f"sealed-from version: {missing}. A name with no scope cannot "
            f"be judged either way, and defaulting it would pick a rule "
            f"nobody wrote.")
    return {"scope": scope, "design_version_in_force": in_force,
            "source_path": "live/pm_research/de_multiday_gate1_runner.py",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "read_by": "ast, at the source; not imported and not copied",
            "referenced_by_functions": used_by}


_SEAL_SCOPE = de_sealed_from_design_version_at_source()
#: name -> the design version FROM WHICH it is sealed.
SEALED_FROM_DESIGN_VERSION = _SEAL_SCOPE["scope"]
DESIGN_VERSION_IN_FORCE_AT_SOURCE = _SEAL_SCOPE["design_version_in_force"]


def _design_version_in_a_tree(commit: str) -> dict:
    """DE's design `VERSION` in the tree a commit names, plus the file's
    digest so a receipt's own closure entry can be matched against it."""
    import da_root as _R                                       # noqa: PLC0415
    root = _R.code_root("resolving a receipt's design version at its "
                        "carrying commit")
    try:
        r = subprocess.run(["git", "-C", str(root), "show",
                            f"{commit}:{DE_DESIGN_REL}"],
                           capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return {"resolved": False, "why": f"git could not run: {e!r}"}
    if r.returncode != 0:
        return {"resolved": False,
                "why": (f"the commit does not resolve in this tree, or the "
                        f"file is not in it: "
                        f"{(r.stderr or b'').decode()[:200].strip()}")}
    blob = r.stdout
    try:
        tree = ast.parse(blob.decode())
    except (UnicodeDecodeError, SyntaxError) as e:
        return {"resolved": False, "why": f"unparseable at that commit: {e}"}
    ver = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "VERSION":
                    ver = int(ast.literal_eval(node.value))
    if ver is None:
        return {"resolved": False,
                "why": "no module-level VERSION in the design declaration "
                       "at that commit"}
    return {"resolved": True, "design_version": ver,
            "file_sha256": hashlib.sha256(blob).hexdigest(),
            "read_by": f"git show {commit[:12]}:{DE_DESIGN_REL}, then ast"}


#: R-673(a). ***A PAIR VERIFIED AGAINST A FILE NOBODY OPENED IS NOT A
#: PAIR.*** Route 1 took the receipt's design path, threw away everything
#: but the BASENAME, and hashed `<derived>/<basename>` -- so a receipt
#: naming `/tmp/x/p003_..._design_v23.json` came back `pair_verified
#: true` whenever a same-named file in the ledger happened to have the
#: same bytes, and the record named a path the verifier never read. The
#: path the receipt GIVES is the path that is hashed: absolute as given,
#: relative resolved against the CODE ROOT (where `data/` is the ledger
#: link the receipts mean). A path that is not there REFUSES BY NAME --
#: the basename is never a second chance.
def resolve_named_path(named: str) -> dict:
    """The file a receipt NAMES, or a named refusal -- never a basename."""
    import da_root as _R                                       # noqa: PLC0415
    raw = Path(str(named))
    #: A BARE BASENAME IS NOT A PATH WITH ITS DIRECTORIES THROWN AWAY.
    #: The defect is DISCARDING a directory the receipt gave; a name that
    #: never had one is a weaker binding, and it is resolved in the
    #: DECLARED search roots and SAID to be a basename in the record.
    basename_only = (raw.parent == Path("."))
    tried = []
    if raw.is_absolute():
        tried.append(raw)
        how = "the path the receipt NAMES, absolute as given"
    elif basename_only:
        tried += [HERE / "declarations" / raw.name, _derived_dir() / raw.name]
        how = ("a BARE BASENAME -- the receipt gave no directory, so it is "
               "resolved in the declared search roots and named as such")
    else:
        root = _R.code_root("resolving a path a receipt names")
        tried.append(Path(root) / raw)
        how = ("the path the receipt NAMES, resolved against the canonical "
               "code root")
    for c in tried:
        if c.is_file():
            return {"resolved": True, "path_hashed": str(c),
                    "as_named": str(named), "basename_only": basename_only,
                    "how": how,
                    "paths_tried": [str(x) for x in tried],
                    "sha256": hashlib.sha256(c.read_bytes()).hexdigest()}
    return {"resolved": False, "as_named": str(named),
            "path_hashed": None, "basename_only": basename_only,
            "paths_tried": [str(x) for x in tried], "sha256": None,
            "why": ("the file this receipt NAMES is not present. The "
                    "basename is NOT tried elsewhere: a digest computed "
                    "over a file the receipt did not name verifies "
                    "nothing about the file it did")}


def receipt_design_version(rec: dict) -> dict:
    """WHICH LIST A RECEIPT IS JUDGED AGAINST -- resolved from the receipt.

    TWO ROUTES, BOTH PAIRS, and nothing else counts:

      (1) `provenance.design` = {path, sha256} (DE 100 on). The digest is
          RECOMPUTED from the file the path names; a path whose digest does
          not match names nothing.
      (2) `source_identity.carrying_commit` + the receipt's OWN import
          closure digest for the design module. The blob at that commit is
          read and hashed; if it equals what the receipt says it imported,
          the tree is identified and its `VERSION` is the design version.

    ***AN OPENED PATH IS NOT A PIN.*** The 09-03 run opened a stale
    `_design_v10` beside `_design_v21` -- picking the first would have been
    a coin toss dressed as evidence.

    UNRESOLVED MEANS THE FULL LIST. An absence never selects the weaker
    rule: a receipt that cannot say what it was produced under is judged
    against every sealed name, and the record says so."""
    ev = []
    prov = ((rec.get("provenance") or {}).get("design")) or {}
    pth, dig = str(prov.get("path") or ""), prov.get("sha256")
    m = re.search(r"_design_v(\d+)(?:__|\.)", Path(pth).name) if pth else None
    if m:
        res = resolve_named_path(pth)
        actual = res["sha256"]
        if dig and actual and dig == actual:
            return {"design_version": int(m.group(1)), "resolved": True,
                    "how": "provenance.design PAIR {path, sha256}, digest "
                           "recomputed from THE PATH THE RECEIPT NAMES",
                    "path_hashed": res["path_hashed"],
                    "path_as_named": res["as_named"],
                    "resolution": res["how"],
                    "pair_verified": True}
        ev.append({"route": "provenance.design",
                   "path": pth, "path_hashed": res.get("path_hashed"),
                   "paths_tried": res.get("paths_tried"),
                   "declared_sha256": dig,
                   "recomputed_sha256": actual, "pair_verified": False,
                   "why": "the path names a version and the digest beside "
                          "it does not match the file -- the version is "
                          "UNKNOWN, not the one the path claims"})
    si = rec.get("source_identity") or {}
    cc = si.get("carrying_commit")
    declared = ((si.get("import_closure") or {}).get("modules")
                or {}).get(Path(DE_DESIGN_REL).name)
    if cc:
        t = _design_version_in_a_tree(str(cc))
        if t.get("resolved") and declared and declared == t["file_sha256"]:
            return {"design_version": t["design_version"], "resolved": True,
                    "how": ("source_identity.carrying_commit PAIRED with "
                            "the receipt's own import-closure digest for "
                            "the design module: the blob at that commit "
                            "hashes to what the receipt says it imported"),
                    "carrying_commit": str(cc),
                    "design_module_sha256": t["file_sha256"],
                    "read_by": t["read_by"], "pair_verified": True}
        ev.append({"route": "carrying_commit + import closure",
                   "commit": str(cc),
                   "closure_digest_in_receipt": declared,
                   "blob_digest_at_that_commit": t.get("file_sha256"),
                   "resolved_tree": t.get("resolved"),
                   "why": t.get("why") or (
                       "the receipt's closure digest and the blob at that "
                       "commit disagree, so the tree is not identified")})
    return {"design_version": None, "resolved": False,
            "how": "NO VERIFIABLE PAIR -- judged against the FULL list",
            "an_opened_path_is_not_a_pin": (
                "the 09-03 run opened a stale _design_v10 beside _design_"
                "v21; choosing one would be a coin toss dressed as "
                "evidence"),
            "evidence_considered": ev, "pair_verified": False}


def fields_in_force_for(rec: dict) -> dict:
    """The sealed names a receipt is judged against, and WHY those."""
    v = receipt_design_version(rec)
    if v["resolved"]:
        names = tuple(n for n in ECONOMIC_FIELDS
                      if SEALED_FROM_DESIGN_VERSION[n] <= v["design_version"])
        later = tuple(n for n in ECONOMIC_FIELDS if n not in names)
    else:
        names, later = tuple(ECONOMIC_FIELDS), ()
    return {"fields": names, "n_fields": len(names),
            "design_version": v["design_version"],
            "version_resolved": v["resolved"], "how_resolved": v["how"],
            "version_evidence": {k: x for k, x in v.items()
                                 if k not in ("design_version", "resolved",
                                              "how")},
            "sealed_only_from_a_later_version": list(later),
            "judged_against_the_full_list": not v["resolved"],
            "why": ("a receipt is judged by the list in force WHEN IT WAS "
                    "PRODUCED; a name sealed later did not exist as a rule "
                    "for it. An unresolvable version is judged against "
                    "every name -- absence never selects the weaker rule")}

#: THE DECLARED TOLERANCE, and where it is exact and where it cannot be.
TOLERANCE = {
    "exact_bit_for_bit": ["D_E0", "Z", "p_location", "null_mean", "null_sd"],
    "why_exact": (
        "the seed pins the DRAW SEQUENCE -- same seed, same pools, same "
        "sorted side order, same numpy Generator gives the same indices -- "
        "and the valuation is a finite sum of products of floats read from "
        "the same records in the same order. Nothing here is sampled twice "
        "or averaged over runs, so an equality is the honest comparison and "
        "a tolerance would only hide a real difference."),
    "where_it_CANNOT_be_exact": (
        "if the replay engine is not deterministic, D(E0) and every null "
        "value inherit that. This module does NOT assume determinism: it "
        "REPLAYS THE BASELINE TWICE and refuses if the two disagree, so the "
        "precondition the exact comparison rests on is checked rather than "
        "asserted."),
    "float_equality_rule": (
        "compared with `==` on floats after both sides are cast to float. "
        "A near-miss is a MISMATCH here, not a pass: the whole point of a "
        "seeded null is that it reproduces."),
}


def carrying_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(HERE))
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


def verifier_identity() -> dict:
    """Cite this code by CONTENT, not only by commit (DA 63's finding: a
    rebase rewrites commit ids and leaves the bytes alone, and a per-seat
    worktree's HEAD is whatever it was last detached at)."""
    src = Path(__file__).resolve()
    r = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    d = subprocess.run(["git", "status", "--porcelain", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    return {"path": "live/pm_research/da_gate1_day_verdict.py",
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "commit_best_effort": (r.stdout.strip() or None),
            "tree_head": carrying_commit(),
            "producing_code_is_the_committed_bytes":
                d.returncode == 0 and d.stdout.strip() == ""}


def load_params(path: Path | None = None) -> dict:
    """The bars come from DE's params declaration, read as a document."""
    p = Path(path) if path else PARAMS_PATH
    if not p.is_file():
        raise VerifierRefused(f"REFUSED: params declaration absent at {p}")
    d = json.loads(p.read_text())
    need = ("min_decisions_per_arm_day", "min_draws_per_arm_day",
            "sd_floor_fraction", "arms", "read_not_before_utc")
    missing = [k for k in need if k not in d]
    if missing:
        raise VerifierRefused(
            f"REFUSED: params declaration is missing {missing}. A bar this "
            f"verifier cannot read is a bar it must not invent.")
    d["_path"] = str(p)
    d["_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return d


# ----------------------------------------------------- the re-implementation

def da_seed_for(book_sha: str, arm: str) -> int:
    """DE's seed rule, re-derived from its statement."""
    h = hashlib.sha256(f"{book_sha}|{arm}|{SEED_TAG}".encode()).hexdigest()
    return int(h[:8], 16)


def da_value_cents(fill: dict) -> float | None:
    """The declared valuation. Returns None on an unvaluable fill -- which
    is a STATUS the caller counts, never a silent zero."""
    lvl = fill.get("px_cents")
    mkt = fill.get("mid_cents_at_markout")
    sz = float(fill.get("size") or 0.0)
    if lvl is None or mkt is None or not sz:
        return None
    sgn = 1.0 if fill.get("side") == BUY_SIDE else -1.0
    return sgn * (float(mkt) - float(lvl)) * sz


def da_total_value(fills: list) -> dict:
    """Sum of valuable fills, with the unvaluable ones COUNTED (rule 4)."""
    vals = [da_value_cents(f) for f in fills]
    n_none = sum(1 for v in vals if v is None)
    return {"value_cents": float(sum(v for v in vals if v is not None)),
            "n_fills": len(fills), "n_unvaluable": n_none}


def da_decisions(scored_rows: list, theta: float) -> dict:
    """The arm's decision set: above-threshold generations at a FIXED theta.

    `>=` is the direction, matching the arm's own definition ("above-
    threshold generations at the arm's FIXED theta -- the set a cancel
    decision is drawn from"). The boundary is pinned by a control."""
    idx = [i for i, r in enumerate(scored_rows)
           if float(r["score"]) >= float(theta)]
    by_side: dict = {}
    for i in idx:
        s = scored_rows[i]["side"]
        by_side[s] = by_side.get(s, 0) + 1
    return {"decision_idx": idx, "n_decisions": len(idx),
            #: SORTED, because the sorted order is what pins the RNG's
            #: consumption order -- see `da_draw_flags`.
            "by_side": dict(sorted(by_side.items())),
            "theta": float(theta)}


def da_pools(rows: list) -> dict:
    pools: dict = {}
    for i, r in enumerate(rows):
        pools.setdefault(r["side"], []).append(i)
    return {k: np.asarray(v) for k, v in sorted(pools.items())}


def da_alloc_realisable(by_side: dict, pools: dict) -> dict:
    """A matched draw must be DRAWABLE. An unrealisable allocation refuses
    rather than silently drawing fewer."""
    for sd, k in by_side.items():
        if sd not in pools:
            raise VerifierRefused(
                f"REFUSED: side {sd!r} is not in the population.")
        if k < 0 or k > len(pools[sd]):
            raise VerifierRefused(
                f"REFUSED: {k} decisions wanted from side {sd!r}, which "
                f"holds {len(pools[sd])}. The allocation is not realisable.")
    return {"realisable": True, "by_side": dict(sorted(by_side.items()))}


def da_draw_flags(pools: dict, by_side: dict, rng) -> np.ndarray:
    """One matched-decision-count draw, per side, without replacement.

    THE SIDE ORDER IS PART OF THE SEED. A numpy Generator is a stream: two
    implementations that consume the sides in different orders produce
    DIFFERENT index sets from the SAME seed. DE passes the SORTED dict
    (`dict(sorted(by_side.items()))`, runner line 1433, handed to the null at
    line 1863), so sorted order is what this reproduces -- and a control
    drives the two orders apart so the dependence is on the record rather
    than in a comment."""
    parts = [rng.choice(pools[sd], k, replace=False)
             for sd, k in sorted(by_side.items()) if k > 0]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=int)


def da_null_values(replay_fn, rows: list, base_value: float, pools: dict,
                   by_side: dict, *, seed: int, k_draws: int) -> dict:
    """K matched random cancel policies, each replayed and valued.

    Reduced per draw on purpose: the draw's fills are valued and dropped
    before the next draw, so memory is O(one draw)."""
    rng = np.random.default_rng(seed)
    values, cancel_counts, first_idx = [], [], []
    for d in range(k_draws):
        flag = da_draw_flags(pools, by_side, rng)
        if d < 8:
            first_idx.append(sorted(int(x) for x in flag))
        res = replay_fn(rows, flag)
        tv = da_total_value(res["fills"])
        values.append(tv["value_cents"] - base_value)
        cancel_counts.append(int(res["cancels_issued"]))
    return {"values": values, "n_draws": len(values), "seed": seed,
            "cancels_per_draw_head": cancel_counts[:8],
            "first_draw_indices": first_idx,
            "metric": "D(E0) per draw = value(draw's fills) - value("
                      "baseline fills), cents, maker fee zero"}


def da_p_location(observed: float, null_draws: list, *,
                  min_draws: int) -> dict:
    """One-sided, larger is better."""
    k = len(null_draws)
    if k < min_draws:
        raise VerifierRefused(
            f"REFUSED: {k} draws is below the declared minimum {min_draws} "
            f"(CLAUDE.md rule 6). An under-sampled correct null flatters as "
            f"much as a wrong one.")
    ge = sum(1 for v in null_draws if v >= observed)
    return {"n_draws": k, "n_null_ge_observed": ge,
            "p_one_sided": (1 + ge) / (1 + k), "floor": 1 / (1 + k)}


def da_z(observed: float, null_draws: list) -> float:
    sd = statistics.pstdev(null_draws)
    if sd == 0:
        raise VerifierRefused(
            "REFUSED: the null has zero dispersion, so a standardised "
            "excess is undefined. A degenerate null is a STATUS, never a "
            "large Z.")
    return (observed - statistics.fmean(null_draws)) / sd


def da_r4(n_decisions: int, null_draws: list, *, min_decisions: int,
          sd_floor_fraction: float) -> dict:
    """The degeneracy bars, re-derived from the declaration."""
    sd = statistics.pstdev(null_draws) if null_draws else 0.0
    mean = statistics.fmean(null_draws) if null_draws else 0.0
    reasons = []
    if n_decisions < min_decisions:
        reasons.append(
            f"decisions {n_decisions} < declared minimum {min_decisions}")
    if sd < sd_floor_fraction * abs(mean):
        reasons.append(
            f"null sd {sd:.6g} < {sd_floor_fraction} * |mean {mean:.6g}| "
            f"= {sd_floor_fraction * abs(mean):.6g}; Z explodes as sd -> 0")
    return {"n_decisions": n_decisions, "null_sd": sd, "null_mean": mean,
            "sd_over_abs_mean": (sd / abs(mean)) if mean else None,
            "admissible": not reasons,
            "status": "OK" if not reasons else "DEGENERATE_ARM_DAY_REFUSED",
            "reasons": reasons}


# ------------------------------------------------------------ the arm-day

def da_arm_day(replay_fn, rows: list, scored_rows: list, *, arm: str,
               theta: float, book_sha: str, params: dict,
               k_draws: int | None = None) -> dict:
    """Recompute one arm-day end to end, from the book, with my own code."""
    k = k_draws or params["min_draws_per_arm_day"]
    seed = da_seed_for(book_sha, arm)

    #: THE PRECONDITION THE EXACT COMPARISON RESTS ON, CHECKED. If the
    #: replay is not deterministic then D(E0) and every null value inherit
    #: that, and an equality would be meaningless.
    b1 = replay_fn(rows, np.zeros(0, dtype=int))
    b2 = replay_fn(rows, np.zeros(0, dtype=int))
    v1, v2 = da_total_value(b1["fills"]), da_total_value(b2["fills"])
    if v1["value_cents"] != v2["value_cents"] or \
            b1["cancels_issued"] != b2["cancels_issued"]:
        raise VerifierRefused(
            f"REFUSED: the replay is not deterministic -- two baseline "
            f"replays gave {v1['value_cents']} / {v2['value_cents']} cents. "
            f"A bit-for-bit comparison of a seeded null rests on this, so "
            f"it is checked, not assumed.")
    base = v1

    dec = da_decisions(scored_rows, theta)
    pools = da_pools(rows)
    alloc = da_alloc_realisable(dec["by_side"], pools)
    arm_res = replay_fn(rows, np.asarray(dec["decision_idx"], dtype=int))
    arm_val = da_total_value(arm_res["fills"])
    observed = arm_val["value_cents"] - base["value_cents"]

    null = da_null_values(replay_fn, rows, base["value_cents"], pools,
                          dec["by_side"], seed=seed, k_draws=k)
    r4 = da_r4(dec["n_decisions"], null["values"],
               min_decisions=params["min_decisions_per_arm_day"],
               sd_floor_fraction=params["sd_floor_fraction"])
    out = {"arm": arm, "theta": theta, "seed": seed,
           "book_sha256": book_sha,
           "baseline": base, "arm_fills": arm_val,
           "decisions": {k2: v for k2, v in dec.items()
                         if k2 != "decision_idx"},
           "allocation": alloc,
           "n_draws": null["n_draws"],
           "null_head": {"cancels_per_draw_head":
                         null["cancels_per_draw_head"],
                         "first_draw_indices": null["first_draw_indices"][:3]},
           "admissibility": r4}
    if not r4["admissible"]:
        out["status"] = r4["status"]
        out["economic"] = None
        out["why_no_economic"] = (
            "a refused arm-day carries no economic field; it is a STATUS "
            "and does not shrink G silently")
        return out
    out["status"] = "OK"
    loc = da_p_location(observed, null["values"],
                        min_draws=params["min_draws_per_arm_day"])
    out["economic"] = {"D_E0": observed,
                       "Z": da_z(observed, null["values"]),
                       "p_location": loc["p_one_sided"],
                       "n_null_ge_observed": loc["n_null_ge_observed"],
                       "null_mean": statistics.fmean(null["values"]),
                       "null_sd": statistics.pstdev(null["values"]),
                       "null_draws_summary": {"n": null["n_draws"]}}
    return out


# ------------------------------------------------------- receipt comparison

def receipt_is_sealed(arm_block: dict) -> dict:
    """Is DE's economic block present, or was it stripped?

    ABSENCE MUST NOT READ AS A PASS. A sealed receipt has no D_E0 to agree
    with, so the comparison is IMPOSSIBLE and must be reported as such --
    never as zero mismatches."""
    econ = arm_block.get("economic")
    present = [f for f in ECONOMIC_FIELDS
               if isinstance(econ, dict) and f in econ]
    return {"sealed": not present,
            "economic_fields_present": present,
            "economic_fields_declared": list(ECONOMIC_FIELDS),
            "why": ("DE's `_strip_economic` removes every economic field at "
                    "every depth until G is complete. With none present "
                    "there is nothing to compare and this verifier says so.")}


def compare_arm(recomputed: dict, receipt_arm: dict) -> dict:
    """Field-by-field, EXACT on the economics. A near-miss is a mismatch."""
    seal = receipt_is_sealed(receipt_arm)
    checks, mismatches = [], []

    def cmp(name, mine, theirs, exact=True):
        if theirs is None:
            checks.append({"field": name, "state": "ABSENT_IN_RECEIPT",
                           "mine": mine})
            return
        ok = (float(mine) == float(theirs)) if exact else (mine == theirs)
        checks.append({"field": name, "state": "MATCH" if ok else "MISMATCH",
                       "mine": mine, "receipt": theirs})
        if not ok:
            mismatches.append(name)

    cmp("status", recomputed["status"], receipt_arm.get("status"), exact=False)
    radm = receipt_arm.get("admissibility") or {}
    cmp("admissibility.n_decisions", recomputed["admissibility"]["n_decisions"],
        radm.get("n_decisions"))
    cmp("admissibility.admissible", recomputed["admissibility"]["admissible"],
        radm.get("admissible"), exact=False)
    prov = receipt_arm.get("draw_provenance") or {}
    if "seed" in prov:
        cmp("seed", recomputed["seed"], prov.get("seed"))

    if seal["sealed"]:
        return {"arm": recomputed["arm"], "seal": seal,
                "checks": checks, "n_mismatches": len(mismatches),
                "mismatched_fields": mismatches,
                "economic_comparison":
                    "ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED",
                "IS_A_VERIFICATION_OF_THE_ECONOMICS": False,
                "why": ("the receipt carries no economic field, so the "
                        "recomputed D(E0), Z, p and null moments agree with "
                        "NOTHING. Reporting zero mismatches here would be "
                        "certifying an empty set.")}

    econ_mine = recomputed.get("economic") or {}
    econ_theirs = receipt_arm.get("economic") or {}
    for f in ("D_E0", "Z", "p_location", "null_mean", "null_sd"):
        if f in econ_mine:
            cmp(f, econ_mine[f], econ_theirs.get(f))
    n_theirs = (econ_theirs.get("null_draws_summary") or {}).get("n")
    if n_theirs is not None:
        cmp("null_draws_summary.n", recomputed["n_draws"], n_theirs)
    return {"arm": recomputed["arm"], "seal": seal, "checks": checks,
            "n_mismatches": len(mismatches),
            "mismatched_fields": mismatches,
            "economic_comparison": "COMPARED_EXACT",
            "IS_A_VERIFICATION_OF_THE_ECONOMICS": True,
            "verdict": "AGREES" if not mismatches else "FLAGGED"}


def verify_book_digest(book_path: str, receipt_sha: str) -> dict:
    """A book whose digest does not match the receipt REFUSES -- the whole
    verification, not the offending arm."""
    p = Path(book_path)
    if not p.is_file():
        raise VerifierRefused(f"REFUSED: book absent at {book_path}")
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != receipt_sha:
        raise VerifierRefused(
            f"REFUSED: book digest mismatch at {book_path}: receipt says "
            f"{receipt_sha}, the bytes say {actual}. A day whose book moved "
            f"is not the day the receipt describes, and no number on it is "
            f"comparable.")
    return {"book": book_path, "sha256": actual, "verified": True}


BOOK_REQUIRED_KEYS = ("rows", "scores_by_arm")
#: BE's pickled day book, as READ from the book verifier's own reading of
#: it (`asm.by_arm`, `fr`) -- the top level only. The MAPPING onto rows and
#: per-arm scores is BE's to declare (R-654), and this verifier will not
#: infer it.
PICKLE_BOOK_TOP_LEVEL = ("asm", "fr")


#: REV 71 2.3. `pickle.load` EXECUTES THE PAYLOAD'S OPCODES; a shape check
#: that runs after it has already run on whatever the bytes said to run.
#: "Validated after loading" is not safety, so the ORDER is part of the
#: contract and it is stated in the record:
#:   1. the book's bytes are DIGEST-PINNED to BE's receipt BEFORE the open,
#:      and a mismatch REFUSES WITHOUT OPENING;
#:   2. the open happens only under the heavy lock and the 8 GiB cap, in
#:      the service form (rule 20);
#:   3. the record says the reader EXECUTED ANOTHER SEAT'S SERIALISATION
#:      and what that means.
PICKLE_ORDER = (
    "digest-pinned to BE's receipt BEFORE the open (a mismatch refuses "
    "without opening); the open only under the heavy lock and the 8 GiB "
    "cap in the service form; and the record states that this reader "
    "EXECUTED another seat's serialisation")
PICKLE_EXECUTION_NOTE = (
    "`pickle.load` runs the opcodes in the file: opening a book is "
    "EXECUTING BYTES ANOTHER SEAT WROTE, and no check that runs afterwards "
    "can undo it. The only control before that point is the DIGEST, which "
    "is why it is pinned first and why a mismatch never reaches the open")


def load_day_book(path: str, *, open_book: bool = False,
                  expected_sha256: str | None = None) -> dict:
    """The day book, through an ADAPTER that refuses what it cannot read.

    This verifier consumes rows plus PER-ARM scores. It does NOT re-score
    from the pinned heads: doing so would require the model artifacts and
    would make this a second scorer rather than a second STATISTIC, and a
    disagreement would then be about scoring rather than about the number
    under test. So a book that does not carry the arm scores REFUSES BY NAME
    -- it is a limit, stated, not a silent partial verification."""
    p = Path(path)
    if not p.is_file():
        raise VerifierRefused(f"REFUSED: day book absent at {path}")
    if p.suffix == ".pkl" and open_book:
        #: (1) THE PIN COMES FIRST, AND A MISMATCH NEVER REACHES THE OPEN.
        if not expected_sha256:
            raise VerifierRefused(
                f"REFUSED: NO_PIN_NO_OPEN -- {p.name} would be unpickled "
                f"with no digest to pin it to. {PICKLE_EXECUTION_NOTE}")
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected_sha256:
            raise VerifierRefused(
                f"REFUSED: BOOK_DIGEST_DOES_NOT_MATCH_ITS_RECEIPT -- "
                f"{p.name} hashes to {actual[:16]}… and the receipt names "
                f"{expected_sha256[:16]}…. NOTHING WAS OPENED: "
                f"{PICKLE_EXECUTION_NOTE}")
        #: THE HEAVY PATH DA 91 WILL RUN, under the wrapper and the lock.
        #: R-654: the recompute needs BE's DECLARATION of the book's
        #: structure (`asm.by_arm`, `fr`), which BE 65 ships as a
        #: declaration and not as prose. Until it lands this refuses on the
        #: SHAPE -- and a pickle whose top level is not the declared one is
        #: refused BY NAME rather than mapped by inference.
        #: (2) only here, and only under the wrapper the caller holds.
        import pickle                                         # noqa: PLC0415
        #: REV 73 S2(a): ***A TRACEBACK IS NOT A VERDICT.*** With the RIGHT
        #: pin on bytes that are not a pickle, `pickle.load` raised
        #: `UnpicklingError` straight out of the verifier -- a caller
        #: reading verdicts got a stack trace, and the two facts that
        #: matter were nowhere in it. ***The pin AUTHORISES the execution;
        #: it does not VALIDATE it***: a digest says these are the bytes
        #: the receipt names, never that they are a book. So both are
        #: named -- the pin MATCHED, the payload is not a pickle.
        try:
            with p.open("rb") as fh:
                obj = pickle.load(fh)
        except (pickle.UnpicklingError, EOFError, ValueError, TypeError,
                AttributeError, ImportError, IndexError,
                MemoryError) as e:
            raise VerifierRefused(
                f"REFUSED: BOOK_PIN_MATCHED_BUT_NOT_A_PICKLE -- {p.name} "
                f"hashes to the digest the receipt names, so the PIN "
                f"HELD; the bytes are then not a loadable pickle "
                f"({type(e).__name__}: {str(e)[:120]}). WHICH OF THE TWO "
                f"FAILED: the pin PASSED, the payload FAILED. A digest "
                f"authorises the execution, it does not validate it."
            ) from e
        if not isinstance(obj, dict):
            raise VerifierRefused(
                f"REFUSED: BOOK_PICKLE_IS_NOT_A_MAPPING -- {p.name} "
                f"unpickles to {type(obj).__name__}. A book this verifier "
                f"cannot address by key is not a book it may score.")
        missing = [k for k in PICKLE_BOOK_TOP_LEVEL if k not in obj]
        if missing:
            raise VerifierRefused(
                f"REFUSED: BOOK_PICKLE_TOP_LEVEL_NOT_THE_DECLARED_SHAPE -- "
                f"{p.name} is missing {missing} (it carries "
                f"{sorted(k for k in obj if isinstance(k, str))[:6]}). The "
                f"mapping from BE's structure onto rows and per-arm scores "
                f"is BE's to DECLARE (R-654); this verifier will not infer "
                f"it from a pickle's shape.")
        raise VerifierRefused(
            f"REFUSED: BOOK_MAPPING_AWAITS_BES_DECLARATION -- {p.name} has "
            f"the declared top level {list(PICKLE_BOOK_TOP_LEVEL)}, and the "
            f"recompute still needs BE 65's declaration of how "
            f"`asm.by_arm` and `fr` become rows and per-arm scores. The "
            f"pickle was OPENED and the shape CHECKED; nothing is "
            f"inferred (R-654).")
    if p.suffix == ".pkl":
        #: THE REAL BOOK IS A PICKLE, AND THIS READER WAS BUILT ON THE
        #: FIXTURE'S JSON. Found at the FIRST REAL GO -- a fixture/real seam
        #: in my own instrument, which is rule 17's shape and the class I
        #: have been auditing in other seats. It is REFUSED BY NAME rather
        #: than guessed at: mapping BE's `asm.by_arm` structure onto rows
        #: and per-arm scores is a reading of ANOTHER SEAT'S BOOK and needs
        #: BE's declaration, not my inference. And opening it is ~2 GB --
        #: HEAVY under rule 20, so it cannot ride along in a light run.
        raise VerifierRefused(
            f"REFUSED: BOOK_IS_A_PICKLE_NOT_THIS_READER'S_JSON -- {p.name} "
            f"is BE's pickled day book ({p.stat().st_size} bytes). This "
            f"reader consumes `rows` + `scores_by_arm`; the pickle carries "
            f"`asm.by_arm` and `fr`. The population half of the pre-read "
            f"CANNOT be recomputed until that mapping is read from BE's "
            f"declaration, and opening the pickle is ~2 GB, which is HEAVY "
            f"under rule 20 and cannot ride in a light run.")
    try:
        bk = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise VerifierRefused(
            f"REFUSED: day book at {path} is not readable JSON ({e.msg}). A "
            f"book this verifier cannot parse is not a book it may score.")
    missing = [k for k in BOOK_REQUIRED_KEYS if k not in bk]
    if missing:
        raise VerifierRefused(
            f"REFUSED: BOOK_CARRIES_NO_ARM_SCORES -- the day book is missing "
            f"{missing}. This verifier re-derives the STATISTIC, not the "
            f"scoring: re-scoring from the pinned heads needs the model "
            f"artifacts and would make a disagreement a scoring difference. "
            f"The book must carry the arm scores it was built with.")
    if not bk["rows"]:
        raise VerifierRefused(
            "REFUSED: the day book carries zero rows. An empty book is a "
            "FAILURE, not a day with no actions.")
    return bk


def resolve_replay(bk: dict, replay_fn=None):
    """The replay SEAM. In production it is the engine of record; here it is
    whatever the caller supplies. A missing engine REFUSES BY NAME rather
    than substituting a second implementation of the policy."""
    if replay_fn is not None:
        return replay_fn, {"source": "supplied by the caller",
                           "is_the_engine_of_record": False}
    fn = bk.get("_replay")
    if callable(fn):
        return fn, {"source": "carried on the book object",
                    "is_the_engine_of_record": False}
    raise VerifierRefused(
        "REFUSED: REPLAY_ENGINE_NOT_RESOLVED. The policy replay is the "
        "instrument of record and this module does not own a second one; "
        "without it no D(E0) may be computed. Wiring it to "
        "`harmful_stateful_policy` is a build step, not a default.")


#: R-604 / REV 50 section 3.3 item 8. The read bar is a CONJUNCTION and the
#: conjuncts live in DE's params, not here. This maps each declared conjunct
#: to an evaluator BY KEYWORD; a conjunct no evaluator matches is a STATUS,
#: never a pass, so params growing from two conjuncts to eight cannot widen
#: what this verifier silently accepts.
#: R-604 / REV 51. DE's params v8 carries the conjuncts as OBJECTS with
#: STABLE IDS, so a conjunct is bound by identity rather than by matching
#: prose. The ids are DE's; if DE's Q-row reports different ones this map is
#: what changes, and the binding says which it used.
CONJUNCT_IDS = {
    "clock_ge_read_not_before": "clock_at_or_after_read_not_before_utc",
    "six_ruled_days_from_params":
        "every_ruled_day_has_exactly_one_sealed_receipt",
    "receipt_at_landing_digest":
        "every_sealed_receipt_matches_its_LANDING_RECORD",
    "at_least_one_admissible_arm":
        "every_day_has_at_least_one_admissible_arm",
    "ledger_verdict": "the_ledger_verdict_stands_for_every_day",
    "producing_code_locatable":
        "every_receipt_names_locatable_producing_code",
    "horizon_fallback_G5_directional": "clock_at_or_before_horizon",
    "params_field_required": "the_params_field_is_present",
}
#: KEYWORD fallback, for params v7 and earlier where the conjunction is a
#: list of SENTENCES. Kept so an older declaration is still evaluated rather
#: than reported wholly unevaluable -- and the binding records which route
#: each conjunct took.
CONJUNCT_EVALUATORS = {
    "clock": "clock_at_or_after_read_not_before_utc",
    "horizon": "clock_at_or_before_horizon",
    "sealed": "every_ruled_day_has_exactly_one_sealed_receipt",
    "landing": "every_sealed_receipt_matches_its_LANDING_RECORD",
    "digest": "every_sealed_receipt_matches_its_LANDING_RECORD",
    "admissible arm": "every_day_has_at_least_one_admissible_arm",
    "ledger verdict": "the_ledger_verdict_stands_for_every_day",
    "producing code": "every_receipt_names_locatable_producing_code",
}
READ_GATE_FIELD = "read_gate"
CONJUNCTION_FIELD = "the_bar_is_a_CONJUNCTION"


#: R-608 / REV 52 section 2.3. A SUPERSESSION LINK IS A PAIR.
#: The definition is READ FROM DE's DESIGN, not typed here: DE's own
#: `supersedes.chain` carries `[path, sha256]` two-element entries, so the
#: link's identity is the PAIR and both halves must land on ONE present
#: file. Resolving by NAME alone accepts a file whose bytes moved; resolving
#: by DIGEST alone accepts the right bytes under a different name.
SUPERSESSION_PAIR_FIELDS = ("path", "sha256")


def _designs_by_version(derived: Path | None = None) -> list:
    """DE's designs, ordered NUMERICALLY. Lexicographic ordering puts v9
    after v16, which would read the definition out of an older design."""
    out = []
    d = Path(derived) if derived else _derived_dir()
    for f in d.glob("p003_de_multiday_gate1_design_v*.json"):
        tok = f.stem.split("design_v", 1)[-1].split("_", 1)[0].split("__")[0]
        try:
            out.append((int("".join(c for c in tok if c.isdigit())), f))
        except ValueError:
            continue
    return [f for _, f in sorted(out)]


def supersession_pair_definition(derived: Path | None = None) -> dict:
    """The link's definition, OBSERVED IN DE's DESIGN rather than typed.

    DE's design declares its own supersession chain as two-element
    `[path, sha256]` entries. That form IS the definition, and reading it
    from the artifact means this verifier cannot hold a different one than
    the seat whose links it is resolving -- which is precisely what REV 52
    found: two seats resolving the same link by different fields, so exactly
    one of them would call the R-603 correction a day that ran twice."""
    designs = _designs_by_version(derived)
    if not designs:
        return {"declared": False,
                "status": "SUPERSESSION_DEFINITION_NOT_DECLARED",
                "required_fields": None,
                "why": ("no DE design is on disk, so the link's definition "
                        "cannot be read from the seat that writes the "
                        "links. This verifier does NOT supply one")}
    d = designs[-1]
    try:
        obj = json.loads(d.read_text())
    except (OSError, json.JSONDecodeError):
        return {"declared": False, "source": d.name,
                "status": "SUPERSESSION_DEFINITION_UNREADABLE",
                "required_fields": None}
    chain = (obj.get("supersedes") or {}).get("chain")
    pair_form = (isinstance(chain, list) and chain
                 and all(isinstance(e, (list, tuple)) and len(e) == 2
                         and isinstance(e[0], str) and isinstance(e[1], str)
                         and len(e[1]) == 64 for e in chain))
    if not pair_form:
        shapes = sorted({len(e) if isinstance(e, (list, tuple)) else
                         type(e).__name__ for e in (chain or [])},
                        key=str)
        return {"declared": False, "source": d.name,
                "status": "SUPERSESSION_DEFINITION_NOT_A_PAIR_FORM",
                "required_fields": None,
                #: A DESIGN IS PRESENT AND DECLARES SOMETHING ELSE. That is
                #: DRIFT, not absence, and the two must not share a branch:
                #: absence leaves this seat enforcing its own constant and
                #: saying so; drift means the authority has moved and every
                #: link judged here would be judged by the wrong rule.
                "a_design_is_present_declaring_another_form": True,
                "observed_entry_shapes": shapes,
                "why": ("DE's design does not carry its own chain as "
                        "[path, sha256] pairs, so the pair form cannot be "
                        "read from it and MUST NOT be assumed")}
    return {"declared": True, "source": d.name,
            "observed_in": "supersedes.chain",
            "n_chain_entries": len(chain),
            "required_fields": list(SUPERSESSION_PAIR_FIELDS),
            "the_link_is": ("the PAIR {path, sha256} of the artifact "
                            "superseded, BOTH of which must match ONE "
                            "PRESENT file"),
            "status": "PAIR_DEFINITION_READ_FROM_DES_DESIGN"}


#: REV 54 section 1.1's RESIDUAL. The binding is ONE-WAY: this seat reads
#: the pair's definition from DE's design, and DE's resolver types its own
#: rule in code. If the design ever declared a different form, DA would
#: follow the design and DE would follow its literals and the seats would
#: part with neither noticing. This seat cannot fix DE's half. What it CAN
#: do is refuse to enforce a rule it did not read: the definition READ and
#: the definition ENFORCED are compared, and a drift REFUSES BY NAME.
AUTHORITATIVE_FOR_THE_LINK_DEFINITION = {
    "artifact": "DE's newest p003_de_multiday_gate1_design_v*.json",
    "field": "supersedes.chain",
    "form": "two-element [path, sha256] entries",
    "why": ("the seat that WRITES the links owns their form; a verifier "
            "that types its own copy can hold a rule the writer abandoned"),
    "this_seat_enforces": list(SUPERSESSION_PAIR_FIELDS),
    "the_binding_is_one_way": (
        "DE's resolver does not read DA's declaration. Closing the loop is "
        "DE's act (REV 54 section 1.1); what this seat guarantees is that "
        "it never enforces a definition it did not read from the authority"),
}

_DEFN_CACHE: dict = {}


def assert_definition_matches_enforcement(force: bool = False,
                                          derived: Path | None = None
                                          ) -> dict:
    """The definition READ must be the definition ENFORCED, or REFUSE.

    Not a warning and not a fallback: if DE's design declares a form this
    code does not implement, every link this resolver judges would be
    judged by the wrong rule, and the verdicts would look normal."""
    if not force and derived is None and "v" in _DEFN_CACHE:
        return _DEFN_CACHE["v"]
    d = supersession_pair_definition(derived)
    out = {"read": d.get("required_fields"),
           "enforced": list(SUPERSESSION_PAIR_FIELDS),
           "source": d.get("source"), "status": d.get("status")}
    if d.get("a_design_is_present_declaring_another_form"):
        raise VerifierRefused(
            f"REFUSED: SUPERSESSION_DEFINITION_DRIFT -- {d['source']} IS "
            f"PRESENT and does not declare the link as a "
            f"{list(SUPERSESSION_PAIR_FIELDS)} PAIR: its own chain entries "
            f"have shapes {d.get('observed_entry_shapes')}. A design that "
            f"declares another form is DRIFT, not absence -- every link "
            f"judged under a definition this seat did not read would be "
            f"judged by the wrong rule and the verdicts would look normal. "
            f"The authority is "
            f"{AUTHORITATIVE_FOR_THE_LINK_DEFINITION['artifact']}, field "
            f"{AUTHORITATIVE_FOR_THE_LINK_DEFINITION['field']}.")
    if not d.get("declared"):
        #: absence is a STATUS, not a pass and not a refusal: with no design
        #: on disk there is nothing to drift FROM, and this seat says so
        #: rather than silently enforcing its own constant as if it had been
        #: read from somewhere.
        out["agrees"] = None
        out["why"] = ("the authority is not readable, so the enforced "
                      "fields are THIS SEAT'S OWN and are reported as such")
        if derived is None:
            _DEFN_CACHE["v"] = out
        return out
    out["agrees"] = list(d["required_fields"]) == list(
        SUPERSESSION_PAIR_FIELDS)
    if not out["agrees"]:
        raise VerifierRefused(
            f"REFUSED: SUPERSESSION_DEFINITION_DRIFT -- "
            f"{d['source']} declares the link as "
            f"{list(d['required_fields'])} and this resolver enforces "
            f"{list(SUPERSESSION_PAIR_FIELDS)}. Every link judged under a "
            f"definition this seat did not read would be judged by the "
            f"wrong rule and the verdicts would look normal. The authority "
            f"for the link's definition is "
            f"{AUTHORITATIVE_FOR_THE_LINK_DEFINITION['artifact']}, field "
            f"{AUTHORITATIVE_FOR_THE_LINK_DEFINITION['field']}.")
    if derived is None:
        _DEFN_CACHE["v"] = out
    return out


def resolve_supersession_link(block, present: dict) -> dict:
    """R-608's three rows, ruled.

    `present` maps filename -> sha256 for the candidate files of one day.

      both fields, agreeing on one present file  -> VALID link
      both fields, digest disagrees              -> TARGET_DIGEST_MISMATCH
      both fields, digest matches another name   -> TARGET_MOVED
      only one field                             -> LINK_INCOMPLETE
                                                    (NOT 'no link')
    """
    if block is None:
        return {"is_a_link": False, "status": "NO_SUPERSEDES_BLOCK"}
    if isinstance(block, str):
        #: a bare path string is the PATH HALF of the pair, so it is an
        #: INCOMPLETE link and refuses by name -- reading it as a link would
        #: be resolving by name, which is exactly what R-608 forbids.
        block = {"path": block}
    if not isinstance(block, dict):
        return {"is_a_link": False, "status": "SUPERSESSION_BLOCK_MALFORMED",
                "why": ("a supersedes block that is neither a mapping nor a "
                        "path carries no pair and is REFUSED BY NAME")}
    have = {f: block.get(f) for f in SUPERSESSION_PAIR_FIELDS}
    missing = [f for f, v in have.items() if not isinstance(v, str) or not v]
    if missing:
        return {"is_a_link": False, "status": "SUPERSESSION_LINK_INCOMPLETE",
                "missing_fields": missing, "present_fields":
                    [f for f in SUPERSESSION_PAIR_FIELDS if f not in missing],
                "why": ("R-608: the link is the PAIR. A block carrying only "
                        "one half is NOT a link and is REFUSED BY NAME -- "
                        "reporting it as 'no link' would make a half-written "
                        "supersession look like two independent runs")}
    name, sha = Path(have["path"]).name, have["sha256"]
    if name in present:
        if present[name] == sha:
            return {"is_a_link": True, "status": "LINK_VALID",
                    "target": name, "sha256": sha}
        return {"is_a_link": False,
                "status": "SUPERSESSION_TARGET_DIGEST_MISMATCH",
                "target": name, "declared_sha256": sha,
                "actual_sha256": present[name],
                "why": ("the named file is present and its bytes are not "
                        "the ones the link declares")}
    elsewhere = [n for n, h in present.items() if h == sha]
    if elsewhere:
        return {"is_a_link": False, "status": "SUPERSESSION_TARGET_MOVED",
                "declared_path": name, "found_as": sorted(elsewhere),
                "why": ("the declared DIGEST is present under a DIFFERENT "
                        "name. Resolving by digest alone would accept it; "
                        "the pair does not, because a moved file is not the "
                        "file the link names")}
    return {"is_a_link": False, "status": "SUPERSESSION_TARGET_ABSENT",
            "declared_path": name, "declared_sha256": sha}


#: R-608. ONE resolver for BOTH chains (sealed receipts and this seat's
#: own landing records), so the two cannot drift into different rules.
CHAIN_REFUSAL_STATUSES = (
    "SUPERSESSION_LINK_INCOMPLETE", "SUPERSESSION_TARGET_DIGEST_MISMATCH",
    "SUPERSESSION_TARGET_MOVED", "SUPERSESSION_TARGET_ABSENT",
    "SUPERSESSION_BLOCK_MALFORMED", "ARTIFACT_UNREADABLE")


def resolve_chain(files, kind: str = "artifact") -> dict:
    """Resolve a set of same-day artifacts through their supersession links.

      none                     -> NO_ARTIFACT
      one                      -> ONE
      v1 + a PAIR-VALID v2     -> CHAIN_HEAD (the v2)
      two with no link         -> AMBIGUOUS
      a link that is not a PAIR-> that link's own refusal, BY NAME

    The last row is R-608: a half-written link is NOT 'no link'. Reporting
    it as 'no link' would turn a botched supersession into what looks like a
    day that ran twice -- or, resolved by name alone, into a clean chain
    over bytes nobody checked."""
    files = sorted(files)
    #: the rule is checked against its AUTHORITY before it is applied
    defn = assert_definition_matches_enforcement()
    if not files:
        return {"status": "NO_ARTIFACT", "n_matches": 0, "head": None,
                "definition": defn}
    if len(files) == 1:
        return {"status": "ONE", "n_matches": 1, "head": files[0],
                "chain": [files[0].name], "links": [], "definition": defn}
    present, links, superseded, refusals = {}, [], set(), []
    for f in files:
        present[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    for f in files:
        try:
            obj = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError) as e:
            refusals.append({"in": f.name, "status": "ARTIFACT_UNREADABLE",
                             "why": str(e)})
            continue
        if "supersedes" not in obj:
            continue
        link = resolve_supersession_link(obj.get("supersedes"), present)
        links.append(dict(link, declared_in=f.name))
        if link["is_a_link"]:
            superseded.add(link["target"])
        elif link["status"] != "NO_SUPERSEDES_BLOCK":
            refusals.append(dict(link, declared_in=f.name))
    if refusals:
        return {"status": refusals[0]["status"], "n_matches": len(files),
                "head": None, "chain": [f.name for f in files],
                "links": links, "refusals": refusals,
                "why": (f"a supersession link on this {kind} is not the PAIR "
                        f"{{path, sha256}} landing on ONE present file "
                        f"(R-608). This is REFUSED BY NAME and is NOT the "
                        f"same finding as 'no link'")}
    heads = [f for f in files if f.name not in superseded]
    if len(heads) == 1:
        return {"status": "CHAIN_HEAD", "n_matches": len(files),
                "head": heads[0], "chain": [f.name for f in files],
                "superseded": sorted(superseded), "links": links,
                "why": (f"a v1 plus a PAIR-VALID v2 resolves to the v2; the "
                        f"head is the {kind} a read would use")}
    return {"status": "AMBIGUOUS", "n_matches": len(files), "head": None,
            "candidates": [f.name for f in files], "links": links,
            "why": (f"two {kind}s for one day with NO supersession link "
                    f"between them. Resolving by picking the newest would "
                    f"decide silently; a day that ran twice is not a day "
                    f"that ran")}


#: REV 52 section 2.4. THE LANDING RECORD'S DECLARED NAME. It had none, so
#: it had no correction path either: a second pre-read for one day was
#: unresolvable. The DAY IS READ FROM THE `day` FIELD, never parsed out of
#: the filename -- the convention makes the record findable, the field makes
#: it identified, and a record carrying no day REFUSES BY NAME.
PRE_READ_NAME_TEMPLATE = "p003_da_gate1_pre_read_<YYYYMMDD>__<clock>.json"
PRE_READ_GLOB = "p003_da_gate1_pre_read_*.json"


def pre_read_artifact_naming() -> dict:
    """THIS SEAT'S OWN DECLARATION of the landing record's name."""
    return {
        "template": PRE_READ_NAME_TEMPLATE,
        "glob": PRE_READ_GLOB,
        "day_comes_from": "the artifact's `day` FIELD, never the filename",
        "day_field_required": True,
        "clock_token": "a UTC stamp %Y%m%dT%H%M%SZ, read from a clock",
        "correction_path": (
            "in band as `.v2`, carrying supersedes = {path, sha256} of the "
            "record it replaces (R-608's PAIR). A chained pair resolves to "
            "the v2; two UNCHAINED records for one day are AMBIGUOUS and "
            "the landing conjunct refuses rather than picking one"),
        "why_declared": (
            "REV 52 section 2.4: an artifact this verifier's own read gate "
            "depends on had no declared name and no correction path, so a "
            "mistaken pre-read could not be superseded -- only shadowed"),
    }


#: REV 54 section 1.3. THE DIGEST HAS TWO HOMES IN THIS ARTIFACT AND ONE
#: AUTHORITY. `receipt.sha256` is the authority because it is the field DE's
#: `landing_record_for` reads (`(rec.get("receipt") or {}).get("sha256")`),
#: and conjunct 3 -- the one that stops a re-roll -- resolves through it.
#: THE AUTHORITY FOLLOWS DE, AND DE HAS MOVED. Round 78 declared
#: `receipt.sha256` authoritative BECAUSE THAT WAS THE FIELD DE'S READER
#: TOOK. At the tip DE's `LANDING_RECORD_FIELD_COPIES` puts
#: `landing_record.receipt_sha256` FIRST and design v20's R23 names the
#: same field, so the reason for the old choice is gone and the choice goes
#: with it. ***The constant is not the authority: DE's declaration is, and
#: this seat's job is to follow it and to NAME any disagreement rather than
#: hold its own.***
LANDING_DIGEST_AUTHORITATIVE_FIELD = "landing_record.receipt_sha256"
LANDING_DIGEST_MIRROR_FIELD = "receipt.sha256"


def landing_digest_fields() -> dict:
    """THIS SEAT'S DECLARATION of which field carries the landing digest."""
    return {
        "authoritative": LANDING_DIGEST_AUTHORITATIVE_FIELD,
        "mirror": LANDING_DIGEST_MIRROR_FIELD,
        "written_from": "ONE hashlib.sha256(receipt bytes) call",
        "equality_is_asserted": "at write time, and again at every read",
        "why_the_authority_is_that_one": (
            "DE's `landing_record_for` reads `receipt.sha256`; this seat "
            "used to read only its own `landing_record.receipt_sha256`. Two "
            "copies written by two calls agree until they do not, and the "
            "conjunct that stops a re-roll resolves through the copy nobody "
            "was checking (REV 54 section 1.3)"),
        "what_DEs_design_must_name": (
            "the same field. Until DE's design names one, this is DA's "
            "declaration alone and says so"),
        "named_in_DEs_design": _design_names_landing_digest_field(),
    }


def _design_names_landing_digest_field() -> dict:
    """Does DE's newest design NAME a landing-digest field?

    A status, never an assumption: absence here is reported as absence."""
    designs = _designs_by_version()
    if not designs:
        return {"status": "NO_DE_DESIGN_ON_DISK", "field": None}
    d = designs[-1]
    try:
        blob = d.read_text()
    except OSError:
        return {"status": "DE_DESIGN_UNREADABLE", "field": None,
                "source": d.name}
    for f in (LANDING_DIGEST_AUTHORITATIVE_FIELD,
              LANDING_DIGEST_MIRROR_FIELD, "receipt_sha256_at_landing"):
        if f in blob:
            return {"status": "NAMED", "field": f, "source": d.name,
                    "agrees_with_DA":
                        f != LANDING_DIGEST_MIRROR_FIELD}
    return {"status": "NOT_YET_NAMED_IN_DES_DESIGN", "field": None,
            "source": d.name,
            "why": ("DE 90 is to name it in design v18; until it does, the "
                    "authority is DA's declaration and the two seats agree "
                    "only because DA now writes and reads the field DE "
                    "reads")}


def de_landing_field_from_code(root: Path | None = None) -> dict:
    """The HEAD of DE's `LANDING_RECORD_FIELD_COPIES`, read BY AST.

    Not by import and not by grep: the tuple's FIRST element is the
    authoritative field and its shape is `(name, (auth...), (other...))`.
    Reading it structurally means a reordering is seen the day it lands."""
    import da_root as _R                                      # noqa: PLC0415
    base = Path(root) if root else _R.code_root(
        "reading DE's landing-record field order")
    f = base / "live/pm_research/de_multiday_gate1_runner.py"
    if not f.is_file():
        return {"status": "DE_RUNNER_ABSENT", "field": None,
                "source": str(f)}
    try:
        tree = ast.parse(f.read_text())
    except SyntaxError as e:
        return {"status": "DE_RUNNER_UNPARSEABLE", "field": None,
                "why": str(e)}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "LANDING_RECORD_FIELD_COPIES"):
            continue
        try:
            rows = ast.literal_eval(n.value)
        except (ValueError, SyntaxError):
            return {"status": "CONSTANT_NOT_A_LITERAL", "field": None}
        if not rows or len(rows[0]) < 2:
            return {"status": "CONSTANT_EMPTY_OR_MALFORMED", "field": None}
        name, auth = rows[0][0], rows[0][1]
        other = rows[0][2] if len(rows[0]) > 2 else None
        return {"status": "READ_FROM_DES_CODE_BY_AST",
                "constant": "LANDING_RECORD_FIELD_COPIES",
                "row_name": name,
                "field": ".".join(auth),
                "second_copy": (".".join(other) if other else None),
                "n_rows": len(rows),
                "source": f.name,
                "sha256_16": hashlib.sha256(f.read_bytes()).hexdigest()[:16]}
    return {"status": "CONSTANT_NOT_PRESENT_IN_DES_CODE", "field": None,
            "source": f.name}


#: where DE's design states the same thing
DESIGN_AUTHORITY_KEYS = ("R23_the_landing_records_authoritative_fields",
                         "authoritative", "receipt_digest_at_landing")


def de_landing_field_from_design(derived: Path | None = None) -> dict:
    """The field DE's NEWEST design names as authoritative."""
    designs = _designs_by_version(derived)
    if not designs:
        return {"status": "NO_DE_DESIGN_ON_DISK", "field": None}
    d = designs[-1]
    try:
        obj = json.loads(d.read_text())
    except (OSError, json.JSONDecodeError):
        return {"status": "DE_DESIGN_UNREADABLE", "field": None,
                "source": d.name}
    cur = obj
    for k in DESIGN_AUTHORITY_KEYS:
        if not isinstance(cur, dict) or k not in cur:
            return {"status": "DESIGN_NAMES_NO_AUTHORITATIVE_DIGEST_FIELD",
                    "field": None, "source": d.name,
                    "path_looked_for": ".".join(DESIGN_AUTHORITY_KEYS)}
        cur = cur[k]
    return {"status": "READ_FROM_DES_DESIGN", "field": cur,
            "source": d.name,
            "sha256_16": hashlib.sha256(d.read_bytes()).hexdigest()[:16]}


def landing_authority_agreement(root: Path | None = None,
                                derived: Path | None = None,
                                mine: str | None = None) -> dict:
    """THREE STATEMENTS OF ONE FIELD, AND WHOSE DISAGREEMENT IT IS.

    DE's CODE, DE's DESIGN and this seat's DECLARATION. REV 58 section 2.3
    asked for the first two to be compared -- nothing did, and a check that
    compared the design against MY OWN constant would report a
    disagreement of mine as one of DE's."""
    code = de_landing_field_from_code(root)
    design = de_landing_field_from_design(derived)
    da = mine or LANDING_DIGEST_AUTHORITATIVE_FIELD
    pairs = {
        "DE_code_vs_DE_design": (code.get("field"), design.get("field")),
        "DA_declaration_vs_DE_code": (da, code.get("field")),
        "DA_declaration_vs_DE_design": (da, design.get("field")),
    }
    out = {"DE_code": code, "DE_design": design, "DA_declaration": da,
           "pairs": {}}
    flags = []
    for k, (a, b) in pairs.items():
        if a is None or b is None:
            st = "NOT_COMPARABLE"
        elif a == b:
            st = "AGREE"
        else:
            st = "DISAGREE"
            flags.append(k)
        out["pairs"][k] = {"status": st, "left": a, "right": b}
    out["flags"] = flags
    out["n_flags"] = len(flags)
    out["whose"] = ("DE" if "DE_code_vs_DE_design" in flags
                    else "DA" if flags else None)
    out["verdict"] = (
        f"FLAGGED_{out['whose']}" if flags else "ALL_THREE_AGREE")
    out["why_it_matters"] = (
        "conjunct 3 -- the one that stops a re-roll -- resolves the day's "
        "receipt digest through THIS field. Two seats reading different "
        "copies of one digest is a gate that passes on whichever copy each "
        "happens to read")
    out["what_is_NOT_flagged_here"] = (
        "this seat's own constant is not the authority and is never the "
        "thing asserted: the check names WHICH PAIR disagrees and WHOSE it "
        "is to reconcile")
    return out


def landing_digest_of(record: dict) -> dict:
    """The landing digest READ THROUGH THE AUTHORITY, with the mirror
    checked. Disagreement REFUSES BY NAME -- it is never resolved by
    preferring one copy."""
    def _at(path):
        cur = record
        for part in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    #: BY THE DECLARED PATHS, so moving the authority moves the reader too.
    auth = _at(LANDING_DIGEST_AUTHORITATIVE_FIELD)
    mirror = _at(LANDING_DIGEST_MIRROR_FIELD)
    if auth and mirror and auth != mirror:
        return {"status": "LANDING_RECORD_DIGEST_FIELDS_DISAGREE",
                "sha256": None, "authoritative": auth, "mirror": mirror,
                "why": ("one artifact carrying two different digests for "
                        "one receipt: each seat would resolve the conjunct "
                        "through its own copy and neither would complain")}
    if not auth and mirror:
        return {"status": "AUTHORITATIVE_FIELD_ABSENT_MIRROR_ONLY",
                "sha256": mirror, "authoritative": None, "mirror": mirror,
                "why": ("an older record predating the declaration: the "
                        "mirror is read and the state is NAMED, never "
                        "silently promoted to the authority")}
    return {"status": "OK", "sha256": auth, "authoritative": auth,
            "mirror": mirror}


def landing_record_for(day: str, derived: Path | None = None) -> dict:
    """ONE day's landing record, resolved through the SAME pair rule.

    no record -> NO_LANDING_RECORD; one -> that record; a chained pair ->
    the v2; two unchained -> AMBIGUOUS."""
    der = Path(derived) if derived else _landing_record_dir()
    d = str(day).replace("-", "")
    mine, refused = [], []
    for f in sorted(der.glob(PRE_READ_GLOB)):
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            refused.append({"file": f.name,
                            "status": "LANDING_RECORD_UNREADABLE"})
            continue
        if not r.get("is_the_declared_LANDING_RECORD"):
            continue
        lr = r.get("landing_record") or {}
        fd = r.get("day") or lr.get("day")
        if not fd:
            #: rule 11: a record with no day is a STATUS, never a silent drop
            refused.append({"file": f.name,
                            "status": "LANDING_RECORD_NO_DAY_FIELD"})
            continue
        if str(fd).replace("-", "") == d:
            mine.append(f)
    res = resolve_chain(mine, "landing record")
    if res["status"] == "NO_ARTIFACT":
        return {"status": "NO_LANDING_RECORD", "day": d, "head": None,
                "receipt_sha256": None, "recorded_by": None,
                "n_matches": 0, "refused_records": refused}
    out = {"status": res["status"], "day": d, "n_matches": res["n_matches"],
           "head": (res["head"].name if res.get("head") else None),
           "chain": res.get("chain"), "candidates": res.get("candidates"),
           "refusals": res.get("refusals"), "refused_records": refused,
           "why": res.get("why"), "receipt_sha256": None,
           "receipt_path": None, "recorded_at_utc": None,
           "recorded_by": (res["head"].name if res.get("head") else None)}
    if res.get("head") is not None:
        rec = json.loads(res["head"].read_text())
        lr = rec.get("landing_record") or {}
        #: READ THROUGH THE AUTHORITY, with the mirror checked (REV 54 1.3)
        dig = landing_digest_of(rec)
        out["digest_read"] = dig
        if dig["status"] == "LANDING_RECORD_DIGEST_FIELDS_DISAGREE":
            out["status"] = dig["status"]
            out["why"] = dig["why"]
        out["receipt_sha256"] = dig["sha256"]
        out["receipt_path"] = lr.get("receipt_path") or (
            rec.get("receipt") or {}).get("path")
        out["recorded_at_utc"] = lr.get("recorded_at_utc")
        out["name_matches_convention"] = bool(
            re.match(r"^p003_da_gate1_pre_read_\d{8}__.+\.json$",
                     res["head"].name))
    return out


def _landing_record_dir() -> Path:
    return _derived_dir()


def landing_records(derived: Path | None = None) -> dict:
    """THE LANDING RECORD (REV 50 section 3.3 item 3).

    This seat's PRE-READ artifact for a day IS the declared record of that
    day's sealed receipt digest AT LANDING. Nothing else in the programme
    records it: a receipt re-emitted after the fact would otherwise be
    indistinguishable from the one the read was scheduled against."""
    der = Path(derived) if derived else _landing_record_dir()
    days = set()
    for f in sorted(der.glob(PRE_READ_GLOB)):
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not r.get("is_the_declared_LANDING_RECORD"):
            continue
        d = r.get("day") or (r.get("landing_record") or {}).get("day")
        if d:
            days.add(str(d).replace("-", ""))
    #: REV 52 section 2.4: every day's record resolves through the SAME
    #: supersession-PAIR rule the sealed receipts do, and an unresolved day
    #: keeps its NAMED status here rather than vanishing from the index.
    return {d: landing_record_for(d, der) for d in sorted(days)}


def read_gate_predicate(params: dict, now: datetime.datetime | None = None,
                        derived: Path | None = None) -> dict:
    """THE FULL READ PREDICATE, evaluated from DE's OWN declared field.

    R-604. `gate_is_open` read `read_not_before_utc` ALONE, so it would have
    opened on the clock while DE's own bar is a conjunction. The conjuncts
    are READ BY NAME from params; a params file that does not carry the
    field REFUSES -- falling back to clock-only would be a verifier quietly
    holding a weaker bar than the programme's.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    rg = params.get(READ_GATE_FIELD)
    if not isinstance(rg, dict) or not isinstance(
            rg.get(CONJUNCTION_FIELD), list):
        raise VerifierRefused(
            f"REFUSED: the params declaration carries no "
            f"`{READ_GATE_FIELD}.{CONJUNCTION_FIELD}`. The read bar is a "
            f"CONJUNCTION and its conjuncts are DE's to declare; falling "
            f"back to the clock alone would be this verifier holding a "
            f"WEAKER bar than the programme's, which is the failure this "
            f"refusal exists to prevent.")
    conjuncts = list(rg[CONJUNCTION_FIELD])
    der = Path(derived) if derived else _derived_dir()
    days = [d.replace("-", "") for d in params.get("days", [])]
    lrs = landing_records(der)
    naming = (rg.get("receipt_naming") or {}).get("expected_per_day") or {}

    def _clock_after():
        bar = params["read_not_before_utc"]
        t = datetime.datetime.fromisoformat(bar.replace("Z", "+00:00"))
        return {"holds": now >= t, "bar": bar, "now": now.isoformat()}

    def _clock_before_horizon():
        h = (rg.get("horizon_utc") or rg.get("horizon")
             or params.get("horizon_utc"))
        if not isinstance(h, str):
            return {"holds": None, "status": "HORIZON_NOT_DECLARED",
                    "why": ("a horizon conjunct with no declared horizon is "
                            "UNEVALUABLE, never satisfied")}
        t = datetime.datetime.fromisoformat(h.replace("Z", "+00:00"))
        past = now > t
        fb = (rg.get("horizon_fallback") or {})
        return {
            "holds": not past, "horizon": h, "now": now.isoformat(),
            "horizon_passed": past,
            "THE_FALLBACK_OUTCOME_IF_IT_PASSES": {
                "G": fb.get("G", 5),
                "reading": fb.get("reading", "directional"),
                "applies_now": past,
                "NEVER_APPLIED_SILENTLY": (
                    "past the horizon the declared fallback is G = 5, "
                    "DIRECTIONAL -- a different claim from the six-day test, "
                    "and this verifier NAMES it rather than quietly reading "
                    "the smaller G as though it were the ruled one. "
                    "Whether to take the fallback is the coordinator's act"),
            },
        }

    def _resolve_day(d):
        """DE's supersession resolver, re-derived: a v1 plus a v2 whose
        link is the PAIR resolves to the v2; two UNCHAINED receipts are
        AMBIGUOUS; a HALF-WRITTEN link refuses BY ITS OWN NAME.

        R-608 / REV 52 section 2.3. This resolved by NAME alone, which
        accepts a chain whose bytes were never checked -- and would have
        called the same artifacts a clean chain that DE, resolving by the
        pair, calls a refusal."""
        pat = naming.get(f"{d[:4]}-{d[4:6]}-{d[6:]}") or \
            f"p003_de_gate1_day_run_{d}_SEALED__*.json"
        r = resolve_chain(sorted(der.glob(pat)), "sealed receipt")
        if r["status"] == "NO_ARTIFACT":
            return {"status": "NO_RECEIPT", "n_matches": 0, "head": None}
        return r

    def _sealed_present():
        per = {}
        for d in days:
            r = _resolve_day(d)
            per[d] = {"n_matches": r["n_matches"], "status": r["status"],
                      "resolves": r["status"] in ("ONE", "CHAIN_HEAD"),
                      "head": (r["head"].name if r.get("head") else None),
                      "chain": r.get("chain"),
                      "why": ("zero is a day that never ran; two UNCHAINED "
                              "is a day that ran twice, which a read refuses "
                              "rather than resolving by picking the newest; "
                              "a chained pair resolves to its HEAD")}
        return {"holds": bool(days) and all(v["resolves"] for v in per.values()),
                "per_day": per,
                "resolution": "supersedes-chain, head-of-chain wins"}

    def _matches_landing():
        per = {}
        for d in days:
            lr = lrs.get(d) or landing_record_for(d, der)
            r = _resolve_day(d)
            head = r.get("head")
            cur = (hashlib.sha256(head.read_bytes()).hexdigest()
                   if head is not None else None)
            lr_sha = lr.get("receipt_sha256")
            #: the RECORD's own chain is resolved by the same pair rule, so
            #: an unresolvable record is a NAMED status, never a missing one
            if lr["status"] == "NO_LANDING_RECORD":
                st = "NO_LANDING_RECORD"
            elif lr["status"] not in ("ONE", "CHAIN_HEAD"):
                st = (lr["status"] if lr["status"].startswith("LANDING_")
                      else "LANDING_RECORD_" + lr["status"])
            elif r["status"] in CHAIN_REFUSAL_STATUSES:
                st = "RECEIPT_" + r["status"]
            elif r["status"] == "AMBIGUOUS":
                st = "AMBIGUOUS_CHAIN"
            elif cur is None:
                st = "NO_RECEIPT"
            elif not lr_sha:
                st = "LANDING_RECORD_CARRIES_NO_DIGEST"
            else:
                st = ("MATCH" if cur == lr_sha
                      else "RECEIPT_MOVED_SINCE_LANDING")
            per[d] = {
                "landing_record_exists":
                    lr["status"] != "NO_LANDING_RECORD",
                "landing_record_status": lr["status"],
                "recorded_by": lr.get("recorded_by"),
                "chain_status": r["status"],
                "chain_head": (head.name if head else None),
                #: TRUE only in the MATCH state -- every named refusal
                #: leaves it None, so no status can read as a pass (rule 11)
                "matches": ((cur == lr_sha) if (cur and lr_sha and st in
                            ("MATCH", "RECEIPT_MOVED_SINCE_LANDING"))
                            else None),
                "status": st}
        return {"holds": bool(days) and all(v["matches"] is True
                                            for v in per.values()),
                "per_day": per,
                "the_record_is_this_seats_pre_read": True,
                "the_digest_is_the_CHAIN_HEADS": (
                    "REV 51 (a): a superseded receipt's landing digest is "
                    "the digest of the chain HEAD, because the head is what "
                    "a read would use. A v1 whose v2 supersedes it is not "
                    "the artifact under test")}

    def _one_admissible_arm():
        per = {}
        for d in days:
            hits = sorted(der.glob(
                naming.get(f"{d[:4]}-{d[4:6]}-{d[6:]}")
                or f"p003_de_gate1_day_run_{d}_SEALED__*.json"))
            if len(hits) != 1:
                per[d] = {"n_admissible_arms": None,
                          "status": "NO_SINGLE_RECEIPT"}
                continue
            r = json.loads(hits[0].read_text())
            arms = _receipt_arms(r)
            n = sum(1 for a in arms.values()
                    if (a.get("admissibility") or {}).get("admissible") is True)
            per[d] = {"n_admissible_arms": n, "status": "COUNTED",
                      "holds": n >= 1}
        return {"holds": bool(days) and all(v.get("holds") is True
                                            for v in per.values()),
                "per_day": per}

    def _ledger_verdict():
        per = {}
        for d in days:
            hits = sorted(der.glob(f"da_dayverdict_{d}.json"))
            per[d] = {"verdict_artifact": (hits[0].name if hits else None),
                      "present": bool(hits)}
        return {"holds": bool(days) and all(v["present"] for v in per.values()),
                "per_day": per,
                "why": "the ledger's own day verdict, one per ruled day"}

    def _producing_code():
        per = {}
        for d in days:
            hits = sorted(der.glob(
                naming.get(f"{d[:4]}-{d[4:6]}-{d[6:]}")
                or f"p003_de_gate1_day_run_{d}_SEALED__*.json"))
            if len(hits) != 1:
                per[d] = {"status": "NO_SINGLE_RECEIPT", "locatable": None}
                continue
            r = json.loads(hits[0].read_text())
            sha = _find_first(r, ("producing_code_sha256", "runner_sha256"))
            commit = _find_first(r, ("runner_commit_best_effort",
                                     "producing_commit", "carrying_commit"))
            ok = False
            if commit:
                q = subprocess.run(["git", "cat-file", "-t", commit],
                                   capture_output=True, text=True,
                                   cwd=str(HERE))
                ok = q.returncode == 0
            per[d] = {"producing_code_sha256": (sha[:16] if sha else None),
                      "commit": commit, "commit_resolves": ok,
                      "locatable": bool(sha) and ok}
        return {"holds": bool(days) and all(v["locatable"] is True
                                            for v in per.values()),
                "per_day": per}

    def _params_field_present():
        return {"holds": True,
                "why": ("the conjunction was read from "
                        f"`{READ_GATE_FIELD}.{CONJUNCTION_FIELD}` -- had it "
                        "been absent this predicate would have REFUSED "
                        "before any conjunct was evaluated, so reaching "
                        "here IS the conjunct holding")}

    EVAL = {
        "the_params_field_is_present": _params_field_present,
        "clock_at_or_after_read_not_before_utc": _clock_after,
        "clock_at_or_before_horizon": _clock_before_horizon,
        "every_ruled_day_has_exactly_one_sealed_receipt": _sealed_present,
        "every_sealed_receipt_matches_its_LANDING_RECORD": _matches_landing,
        "every_day_has_at_least_one_admissible_arm": _one_admissible_arm,
        "the_ledger_verdict_stands_for_every_day": _ledger_verdict,
        "every_receipt_names_locatable_producing_code": _producing_code,
    }
    results, unevaluable = [], []
    for item in conjuncts:
        #: ID FIRST. An object with an `id` is bound by identity; a bare
        #: sentence falls back to keywords, and the route is recorded --
        #: because "matched some words" and "is this conjunct" are
        #: different claims and a reader is entitled to know which was made.
        cid, text, route = None, item, None
        if isinstance(item, dict):
            cid = item.get("id")
            text = item.get("text") or item.get("statement") or item.get("id")
        name = None
        if cid and cid in CONJUNCT_IDS:
            name, route = CONJUNCT_IDS[cid], "bound by ID"
        elif cid:
            route = "ID NOT IN THIS VERIFIER'S MAP"
        else:
            low = str(text).lower()
            for kw, ev in CONJUNCT_EVALUATORS.items():
                if kw in low:
                    name, route = ev, f"matched the keyword {kw!r}"
                    break
            if name is None:
                route = "no id and no keyword matched"
        if name is None or name not in EVAL:
            unevaluable.append(cid or text)
            results.append({"conjunct": text, "id": cid, "evaluator": None,
                            "binding": route, "holds": None,
                            "status": "CONJUNCT_NOT_EVALUABLE_BY_THIS_"
                                      "VERIFIER",
                            "why": ("no evaluator is bound to this "
                                    "conjunct. It is a STATUS and NOT a "
                                    "pass: a bar this verifier cannot check "
                                    "is a bar it must not wave through")})
            continue
        res = EVAL[name]()
        results.append({"conjunct": text, "id": cid, "evaluator": name,
                        "binding": route, **res})

    holds = [r for r in results if r.get("holds") is True]
    fails = [r["evaluator"] or r["conjunct"] for r in results
             if r.get("holds") is False]
    return {
        "source_field": f"{READ_GATE_FIELD}.{CONJUNCTION_FIELD}",
        "params_protocol": params.get("protocol"),
        "n_conjuncts_declared": len(conjuncts),
        "n_evaluated": len(conjuncts) - len(unevaluable),
        "n_holding": len(holds),
        "conjuncts": results,
        "conjuncts_not_evaluable": unevaluable,
        "failing_conjuncts_by_name": fails,
        "open": bool(conjuncts and not unevaluable and not fails
                     and len(holds) == len(conjuncts)),
        "no_override_flag_exists": True,
        "why_unevaluable_is_not_a_pass": (
            "a conjunct no evaluator matches is reported and CLOSES the "
            "gate. Params growing from two conjuncts to eight cannot widen "
            "what this verifier silently accepts"),
    }


def gate_is_open(params: dict, now: datetime.datetime | None = None,
                 derived: Path | None = None) -> dict:
    """The read bar. R-604: the FULL conjunction, from DE's own field.

    This used to read `read_not_before_utc` alone. DE's bar is a
    CONJUNCTION -- the clock AND the sealed receipts at their landing
    digests AND an admissible arm per day AND the ledger verdict AND
    locatable producing code, within a horizon. A verifier holding only the
    clock would open on a bar the programme does not hold."""
    pred = read_gate_predicate(params, now, derived)
    bar = params["read_not_before_utc"]
    t_bar = datetime.datetime.fromisoformat(bar.replace("Z", "+00:00"))
    t_now = now or datetime.datetime.now(datetime.timezone.utc)
    return {"read_not_before_utc": bar, "now_utc": t_now.isoformat(),
            "open": pred["open"],
            "seconds_until_the_clock_conjunct": (
                t_bar - t_now).total_seconds(),
            "predicate": pred,
            "no_override_flag_exists": True,
            "the_bar_is_a_CONJUNCTION_not_a_clock": (
                "R-604. Reading `read_not_before_utc` alone would open on a "
                "bar weaker than the one DE declares")}


def declared_limits(receipt_arm_seals: list, params: dict,
                    book_meta: dict) -> list:
    """THE FOUR LIMITS, each carrying a COMPUTED field rather than a claim.

    A limits section written as prose is a paragraph a reader skims. Each
    entry here is decided by something this run measured."""
    de = de_economic_fields_at_source()
    #: (1) D_E_MINUS_R: it is in DE's economic field list, and the runner
    #: never produces it -- counted from the source, not asserted.
    src = DE_RUNNER_PATH.read_text()
    tree = ast.parse(src)
    _b = binding_sites(tree, "D_E_MINUS_R")
    n_assign = _b["n_binds"]
    n_named_in_declarations = _b["n_declares"]
    if _b["unhandled"]:
        raise VerifierRefused(
            f"REFUSED: UNHANDLED_BINDING_SHAPE_TOUCHES_A_WATCHED_NAME -- "
            f"{_b['unhandled']}. This limit CLAIMS the runner never "
            f"produces D(E-R); a binding shape this scan does not "
            f"understand leaves that claim reading TRUE for a reason "
            f"nobody checked.")
    n_sealed = sum(1 for s in receipt_arm_seals if s)
    return [
        {"limit": "D_E_MINUS_R_IS_NOT_VERIFIED",
         "what": ("D(E-R) is in DE's economic field list, so a receipt "
                  "carrying it would be compared -- but the runner declares "
                  "it UNBOUND (it needs the rebate's identity value, which "
                  "is not on DE's surface) and never emits it. This verifier "
                  "therefore verifies D(E0) and nothing about D(E-R)."),
         "computed": {"in_DEs_economic_field_list":
                          "D_E_MINUS_R" in de["fields"],
                      "n_places_the_runner_produces_it": n_assign,
                      "n_places_it_is_only_NAMED_in_a_declaration":
                          n_named_in_declarations,
                      "a_producing_place_binds_a_value": (
                          "a dict entry whose value is a CONSTANT is a "
                          "table entry (DE's per-name seal scope says "
                          "`D_E_MINUS_R: 1`, a VERSION); one whose value "
                          "is an expression is an emission"),
                      "so_there_is_nothing_to_compare": n_assign == 0}},
        {"limit": "THE_BOOK_IS_SHARED_AND_ITS_CONSTRUCTION_IS_NOT_CHECKED",
         "what": ("both implementations read the SAME day book. If the book "
                  "was built wrong -- wrong rows, wrong scores, wrong "
                  "generation keys -- the two agree on a number computed "
                  "from the same wrong input. Agreement here is evidence "
                  "about the STATISTIC, never about the book."),
         "computed": {"book_sha256": book_meta.get("sha256"),
                      "arm_scores_read_from_the_book_not_recomputed": True,
                      "this_module_rescored_nothing": True}},
        {"limit": "AN_ERROR_IN_THE_DECLARATION_REPRODUCES_IN_BOTH",
         "what": ("the seed rule, the thetas, the bars and the arms all come "
                  "from the SAME params declaration both sides read. A wrong "
                  "theta or a wrong seed rule is reproduced identically by "
                  "an independent implementation, and the exact agreement "
                  "would say nothing about whether the declaration is right."),
         "computed": {"params_path": Path(params["_path"]).name,
                      "params_sha256": params["_sha256"],
                      "read_by_both_sides_from_one_file": True}},
        {"limit": "SEALED_ECONOMICS_CANNOT_BE_VERIFIED_AT_ALL",
         "what": ("a receipt whose economic fields were stripped offers "
                  "nothing for the recomputed D(E0), Z, p and null moments "
                  "to agree WITH. The read bar opening does not open a "
                  "sealed receipt: after the bar a sealed receipt is still "
                  "ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED."),
         "computed": {"n_arms_sealed_in_this_receipt": n_sealed,
                      "n_arms_seen": len(receipt_arm_seals),
                      "economic_field_list_source": de["source_path"],
                      "economic_field_list_sha256": de["source_sha256"]}},
    ]


def verify_real_day(day: str, book_path: str, receipt_path: str, *,
                    output: Path | None = None,
                    params: dict | None = None,
                    now: datetime.datetime | None = None,
                    replay_fn=None) -> dict:
    """THE REAL-DAY ENTRY POINT.

    The order is the order the refusals must happen in:
      1. the READ BAR      -- before it, nothing else is even looked at
      2. the BOOK DIGEST   -- a book that moved is not the receipt's day
      3. the SEAL          -- a sealed receipt is not a verification, and
                              the bar opening does not change that
      4. the RECOMPUTATION -- per arm, from the book, at the declared seed
      5. the COMPARISON    -- exact, and the verdict is COMPUTED
    """
    params = params or load_params()
    g = gate_is_open(params, now)
    if not g["open"]:
        raise VerifierRefused(
            f"REFUSED: the Gate-1 real-day path is closed until "
            f"{g['read_not_before_utc']} (now {g['now_utc']}, "
            f"{g['seconds_until_the_clock_conjunct']:.0f}s to go). A day verdict "
            f"recomputed before the read bar is a read of the gate, whatever "
            f"it is called -- and there is no flag that moves this bar.")

    rp = Path(receipt_path)
    if not rp.is_file():
        raise VerifierRefused(f"REFUSED: receipt absent at {receipt_path}")
    receipt = json.loads(rp.read_text())
    r_sha = receipt.get("book_sha256") or receipt.get("book", {}).get(
        "sha256")
    if not r_sha:
        raise VerifierRefused(
            "REFUSED: the receipt names no book digest, so the book it "
            "describes cannot be identified. An unpinned book is not a day.")
    book_meta = verify_book_digest(book_path, r_sha)

    bk = load_day_book(book_path)
    replay, replay_meta = resolve_replay(bk, replay_fn)
    rows = bk["rows"]

    arms_out, seals, verifications = {}, [], []
    for arm, spec in sorted(params["arms"].items()):
        r_arm = (receipt.get("arms") or {}).get(arm)
        if r_arm is None:
            arms_out[arm] = {"status": "ABSENT_FROM_THE_RECEIPT",
                             "IS_A_VERIFICATION_OF_THE_ECONOMICS": False,
                             "why": ("the receipt carries no block for this "
                                     "declared arm; a missing arm is a "
                                     "STATUS, never a silent pass")}
            verifications.append(False)
            continue
        seal = receipt_is_sealed(r_arm)
        seals.append(seal["sealed"])
        if seal["sealed"]:
            arms_out[arm] = {
                "status": "ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED",
                "seal": seal,
                "IS_A_VERIFICATION_OF_THE_ECONOMICS": False,
                "the_bar_does_not_open_a_seal": (
                    "the read bar has passed and this receipt is still "
                    "sealed. Those are different gates: one schedules the "
                    "read, the other withholds the numbers."),
            }
            verifications.append(False)
            continue
        vec = bk["scores_by_arm"].get(arm)
        if vec is None:
            arms_out[arm] = {"status": "BOOK_CARRIES_NO_SCORES_FOR_THIS_ARM",
                             "IS_A_VERIFICATION_OF_THE_ECONOMICS": False,
                             "why": "declared in params, absent in the book"}
            verifications.append(False)
            continue
        if len(vec) != len(rows):
            #: A score vector of the wrong length would zip SHORT and
            #: silently score a prefix of the day -- a smaller population
            #: wearing the day's name.
            arms_out[arm] = {
                "status": "ARM_SCORE_VECTOR_LENGTH_MISMATCH",
                "n_rows": len(rows), "n_scores": len(vec),
                "IS_A_VERIFICATION_OF_THE_ECONOMICS": False,
                "why": ("a score vector shorter than the rows would zip to a "
                        "PREFIX and silently score part of the day")}
            verifications.append(False)
            continue
        #: the book carries scores as a VECTOR aligned to `rows`; the
        #: statistic reads them off the row, so they are joined here once.
        scored = [dict(r, score=float(sv)) for r, sv in zip(rows, vec)]
        mine = da_arm_day(replay, rows, scored, arm=arm,
                          theta=float(spec["theta"]), book_sha=r_sha,
                          params=params)
        cmp_ = compare_arm(mine, r_arm)
        arms_out[arm] = {"status": mine["status"], "recomputed": mine,
                         "comparison": cmp_,
                         "IS_A_VERIFICATION_OF_THE_ECONOMICS":
                             cmp_["IS_A_VERIFICATION_OF_THE_ECONOMICS"]}
        verifications.append(cmp_["IS_A_VERIFICATION_OF_THE_ECONOMICS"]
                             and cmp_["n_mismatches"] == 0)

    #: EVERY declared arm must have contributed exactly one verdict. A
    #: branch added later that `continue`s without recording one would make
    #: the conjunction read over a SHORTER list and a missing arm would pass
    #: silently -- which is exactly what the battery caught here once.
    if len(verifications) != len(params["arms"]):
        raise VerifierRefused(
            f"REFUSED: {len(verifications)} verdicts for "
            f"{len(params['arms'])} declared arms. Every arm must contribute "
            f"one, or the conjunction is taken over a shorter list and an "
            f"unhandled arm passes silently.")

    out = {
        "protocol": PROTOCOL + "_REAL_DAY",
        "day": day,
        "status": "VERIFIED" if (verifications and all(verifications))
                  else "NOT_A_FULL_VERIFICATION",
        "gate": g,
        "book": book_meta,
        "replay_seam": replay_meta,
        "receipt": {"path": rp.name,
                    "sha256": hashlib.sha256(rp.read_bytes()).hexdigest()},
        "params_declaration": {"path": Path(params["_path"]).name,
                               "sha256": params["_sha256"]},
        "economic_field_list": de_economic_fields_at_source(),
        "verifier_identity": verifier_identity(),
        "tolerance": TOLERANCE,
        "arms": arms_out,
        "n_arms_declared": len(params["arms"]),
        "n_arms_verified": sum(1 for v in verifications if v),
        "n_arms_sealed": sum(1 for x in seals if x),
        "IS_A_VERIFICATION_OF_THE_ECONOMICS": bool(
            verifications and all(verifications)),
        "why_that_is_computed": (
            "it is the conjunction over the declared arms of 'the economics "
            "were comparable AND every compared field matched exactly'. A "
            "sealed arm contributes False, so a receipt that withheld its "
            "numbers can never read as verified."),
        "limits": declared_limits(seals, params, book_meta),
    }
    if output:
        Path(output).write_text(
            json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    return out


# --------------------------------------------------------------- the fixture

SIDES_FIX = ("BUY_UP", "SELL_UP")
BAR_BEFORE = datetime.datetime(2026, 9, 6, 8, 0,
                               tzinfo=datetime.timezone.utc)
BAR_AFTER = datetime.datetime(2026, 9, 10, 0, 0,
                              tzinfo=datetime.timezone.utc)


def synthetic_book(n_rows: int = 80, *, seed: int = 20260906) -> dict:
    """A book whose D(E0) is known IN CLOSED FORM.

    Each row carries one fill worth `v_i` cents, encoded so the DECLARED
    valuation returns exactly `v_i`. The synthetic replay drops the
    cancelled rows' fills and nothing else, so for any cancelled set C:
    D(C) = -sum(v_i for i in C) -- an identity, which is what makes it a
    falsifier rather than a demonstration."""
    rng = np.random.default_rng(seed)
    rows, values = [], []
    for i in range(n_rows):
        rows.append({"t": 1000.0 + i, "slug": f"s{i // 4}",
                     "side": SIDES_FIX[i % 2], "gen": i,
                     "score": float(rng.random())})
        values.append(float(round(rng.normal(0.0, 10.0), 6)))
    return {"rows": rows, "values": values}


def synthetic_replay(book: dict):
    values = book["values"]

    def _replay(rows: list, cancelled) -> dict:
        c = set(int(i) for i in np.asarray(cancelled, dtype=int).ravel())
        fills = []
        for i, r in enumerate(rows):
            if i in c:
                continue
            sgn = 1.0 if r["side"] == BUY_SIDE else -1.0
            fills.append({"side": r["side"], "px_cents": 0.0,
                          "mid_cents_at_markout": sgn * values[i],
                          "size": 1.0})
        return {"fills": fills, "cancels_issued": len(c),
                "n_fills": len(fills)}
    return _replay


def known_D(book: dict, decision_idx) -> float:
    return -float(sum(book["values"][int(i)] for i in decision_idx))


def _strip_like_DE(o):
    """DE's own stripper, re-derived over the field list read at source."""
    if isinstance(o, dict):
        return {k: _strip_like_DE(v) for k, v in o.items()
                if k not in ECONOMIC_FIELDS}
    if isinstance(o, list):
        return [_strip_like_DE(v) for v in o]
    return o


def write_day_book(d: Path, book: dict, params: dict, *,
                   omit_scores: bool = False) -> tuple:
    """A day book on disk in the shape the adapter declares."""
    payload = {"rows": book["rows"]}
    if not omit_scores:
        payload["scores_by_arm"] = {
            a: [r["score"] for r in book["rows"]] for a in params["arms"]}
    p = d / "book.json"
    p.write_text(json.dumps(payload))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def write_receipt(d: Path, arms: dict, book_sha: str, *,
                  sealed: bool = False, name: str = "receipt.json") -> Path:
    body = {"day": "2026-09-03", "book_sha256": book_sha,
            "arms": {a: (_strip_like_DE(v) if sealed else v)
                     for a, v in arms.items()}}
    p = d / name
    p.write_text(json.dumps(body))
    return p


def selftest() -> tuple:                                      # noqa: C901
    checks: list[dict] = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    params = load_params()
    #: THE FIXTURE'S OWN DECLARATION. The real params' read gate is a
    #: CONJUNCTION over artifacts no fixture has -- sealed day receipts,
    #: ledger verdicts, landing records. Testing the arm logic through it
    #: would be testing the gate. So the fixture declares a read gate with
    #: the CLOCK conjunct alone: a legitimate declaration shape, not a
    #: bypass, and the multi-conjunct file is exercised on its own below
    #: against the REAL params.
    P = json.loads(json.dumps({k: v for k, v in params.items()
                               if not k.startswith("_")}))
    P[READ_GATE_FIELD] = {CONJUNCTION_FIELD: [
        "the clock >= read_not_before_utc"],
        "note": "the FIXTURE's declaration -- one conjunct, so the arm "
                "logic is exercised and not the gate"}
    P["_path"], P["_sha256"] = params["_path"], params["_sha256"]
    td = Path(tempfile.mkdtemp(prefix="da67_"))

    # -- 1. params v5, and the bars are READ ------------------------------
    nums = sorted(int(f.stem.rsplit("_v", 1)[-1])
                  for f in (HERE / "declarations").glob(
                      "de_multiday_gate1_params_v*.json")
                  if f.stem.rsplit("_v", 1)[-1].isdigit())
    ck("THE BARS COME FROM THE NEWEST PARAMS PRESENT, RESOLVED NOT TYPED "
       "(REV 45 3.3, learned twice): a checker pinned to a superseded "
       "declaration reads bars nobody runs under -- and pinning the FILENAME "
       "taught it again, because v5 predates `read_gate` and the conjunction "
       "this verifier must evaluate was not in the file it was pinned to",
       PARAMS_PATH.name.endswith(f"v{max(nums)}.json")
       and params["protocol"].endswith(f"V{max(nums)}")
       and params["min_decisions_per_arm_day"] == 30
       and params["min_draws_per_arm_day"] == 500
       and params["sd_floor_fraction"] == 0.25,
       f"{PARAMS_PATH.name} sha {params['_sha256'][:16]}, protocol "
       f"{params['protocol']}, read bar {params['read_not_before_utc']}; "
       f"versions present {nums}, newest chosen")

    # -- 2. DE's field list, AT SOURCE, with the stripper asserted --------
    de = de_economic_fields_at_source()
    ck("DE's ECONOMIC FIELD LIST IS READ AT THE SOURCE BY AST, and the "
       "assertion is not just that the constant exists but that "
       "`_strip_economic` -- the function that actually removes them -- "
       "REFERENCES THAT NAME. A constant nothing uses would pin nothing",
       de["fields"] == ECONOMIC_FIELDS
       #: the SEVEN the list has always carried must still be in it -- a
       #: field silently LEAVING the list would unseal a quantity. The
       #: COUNT is not pinned: DE may add to it, and DE 85 did exactly that
       #: under R-599, which this instrument saw from the source within the
       #: hour without being told.
       and set(SEVEN_ORIGINAL_ECONOMIC_FIELDS) <= set(de["fields"])
       and de["stripper_references_the_same_name"] is True,
       f"{len(de['fields'])} fields from {de['source_path']} sha "
       f"{de['source_sha256'][:16]}: {list(de['fields'])}"
       + (f" -- GREW by {sorted(set(de['fields']) - set(SEVEN_ORIGINAL_ECONOMIC_FIELDS))} "
          f"since the seven this check pins as a floor"
          if set(de["fields"]) - set(SEVEN_ORIGINAL_ECONOMIC_FIELDS) else ""))
    bad_src = td / "nostrip.py"
    bad_src.write_text("ECONOMIC_FIELDS = ('D_E0',)\n"
                       "def _strip_economic(o):\n    return o\n")
    refused_nostrip = False
    try:
        de_economic_fields_at_source(bad_src)
    except VerifierRefused as e:
        refused_nostrip = "does not reference" in str(e)
    ck("KNOWN-BAD: a runner whose `_strip_economic` does NOT reference the "
       "constant REFUSES -- the stripper could then be removing a different "
       "set entirely and the seal detection would be pinned to nothing",
       refused_nostrip,
       "a source with the constant present but unused by the stripper raises")

    # -- 3. the book adapter, both ways -----------------------------------
    book = synthetic_book()
    replay = synthetic_replay(book)
    rows = book["rows"]
    bpath, bsha = write_day_book(td, book, params)
    loaded = load_day_book(bpath)
    noscore_p, _ = write_day_book(td / "ns" if (td / "ns").mkdir(
        exist_ok=True) is None else (td / "ns"), book, params,
        omit_scores=True)
    refused_book = False
    try:
        load_day_book(noscore_p)
    except VerifierRefused as e:
        refused_book = "BOOK_CARRIES_NO_ARM_SCORES" in str(e)
    ck("THE BOOK ADAPTER BOTH WAYS: a book carrying rows AND per-arm scores "
       "loads; one without the scores REFUSES BY NAME. This verifier "
       "re-derives the STATISTIC, not the scoring -- re-scoring from the "
       "pinned heads would make a disagreement a scoring difference",
       set(loaded) >= {"rows", "scores_by_arm"} and refused_book,
       f"{len(loaded['rows'])} rows, {len(loaded['scores_by_arm'])} armed "
       f"score vectors; a book without them raises "
       f"BOOK_CARRIES_NO_ARM_SCORES")

    # -- 4. the read bar: REFUSES before, ADMITS after --------------------
    arm0 = sorted(params["arms"])[0]
    theta0 = float(params["arms"][arm0]["theta"])
    mine0 = da_arm_day(replay, rows, rows, arm=arm0, theta=theta0,
                       book_sha=bsha, params=params)
    arms_payload = {a: {"day": "2026-09-03", "arm": a,
                        "status": mine0["status"],
                        "admissibility": dict(mine0["admissibility"]),
                        "draw_provenance": {"seed": da_seed_for(bsha, a)},
                        "economic": dict(mine0["economic"] or {})}
                    for a in params["arms"]}
    # each arm has its OWN seed and theta, so recompute per arm honestly
    for a in params["arms"]:
        m = da_arm_day(replay, rows, rows, arm=a,
                       theta=float(params["arms"][a]["theta"]),
                       book_sha=bsha, params=params)
        arms_payload[a] = {"day": "2026-09-03", "arm": a,
                           "status": m["status"],
                           "admissibility": dict(m["admissibility"]),
                           "draw_provenance": {"seed": m["seed"]},
                           "economic": dict(m["economic"] or {})}
    rpath = write_receipt(td, arms_payload, bsha)

    why_pre = ""
    try:
        verify_real_day("2026-09-03", str(bpath), str(rpath),
                        params=P, now=BAR_BEFORE, replay_fn=replay)
    except VerifierRefused as e:
        why_pre = str(e)
    ck("BEFORE THE BAR THE REAL-DAY PATH REFUSES, AND THE BAR IS NAMED: a "
       "day verdict recomputed before the read bar is a read of the gate, "
       "whatever it is called",
       "closed until" in why_pre
       and params["read_not_before_utc"] in why_pre
       and "no flag that moves this bar" in why_pre,
       f"'{why_pre[:104]}...'")

    out = verify_real_day("2026-09-03", str(bpath), str(rpath),
                          params=P, now=BAR_AFTER, replay_fn=replay)
    ck("AND WITH THE CLOCK PAST THE BAR IT ADMITS AND VERIFIES: every "
       "declared arm recomputed from the book at its own seed, compared "
       "EXACT, and IS_A_VERIFICATION_OF_THE_ECONOMICS computed true",
       out["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is True
       and out["status"] == "VERIFIED"
       and out["n_arms_verified"] == len(params["arms"])
       and out["n_arms_sealed"] == 0
       and all(v["comparison"]["n_mismatches"] == 0
               for v in out["arms"].values()),
       f"{out['n_arms_verified']} of {out['n_arms_declared']} arms verified, "
       f"0 sealed; the clock is a PARAMETER for this test and is NOT a CLI "
       f"flag -- a bar with a documented way past it is not a bar")

    # -- 5. one moved economic field FLAGS --------------------------------
    moved = json.loads(rpath.read_text())
    a_first = sorted(moved["arms"])[0]
    moved["arms"][a_first]["economic"]["D_E0"] = float(
        moved["arms"][a_first]["economic"]["D_E0"]) + 1e-9
    mpath = td / "receipt_moved.json"
    mpath.write_text(json.dumps(moved))
    out_m = verify_real_day("2026-09-03", str(bpath), str(mpath),
                            params=P, now=BAR_AFTER, replay_fn=replay)
    ck("KNOWN-BAD: ONE MOVED ECONOMIC FIELD IS FLAGGED and the run is NOT a "
       "verification -- the comparison is exact, so 1e-9 is a mismatch",
       out_m["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is False
       and out_m["status"] == "NOT_A_FULL_VERIFICATION"
       and out_m["arms"][a_first]["comparison"]["mismatched_fields"]
       == ["D_E0"],
       f"{a_first}.D_E0 +1e-9 -> flagged "
       f"{out_m['arms'][a_first]['comparison']['mismatched_fields']}, "
       f"verification {out_m['IS_A_VERIFICATION_OF_THE_ECONOMICS']}")

    # -- 6. A SEALED RECEIPT AFTER THE BAR IS STILL NOT A VERIFICATION ----
    spath = write_receipt(td, arms_payload, bsha, sealed=True,
                          name="receipt_sealed.json")
    out_s = verify_real_day("2026-09-03", str(bpath), str(spath),
                            params=P, now=BAR_AFTER, replay_fn=replay)
    ck("THE BAR OPENING DOES NOT OPEN A SEAL: with the clock PAST the read "
       "bar, a sealed receipt is still "
       "ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED and the run is NOT a "
       "verification. Those are different gates -- one schedules the read, "
       "the other withholds the numbers",
       out_s["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is False
       and out_s["n_arms_sealed"] == len(params["arms"])
       and all(v["status"] == "ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED"
               for v in out_s["arms"].values())
       and out_s["gate"]["open"] is True,
       f"gate open {out_s['gate']['open']}, "
       f"{out_s['n_arms_sealed']} arms sealed, verification "
       f"{out_s['IS_A_VERIFICATION_OF_THE_ECONOMICS']} -- and the seal is "
       f"detected by DE's OWN field list read at source")

    # -- 7. a wrong book REFUSES ------------------------------------------
    wrong = td / "wrong_book.json"
    w = json.loads(bpath.read_text())
    w["rows"] = w["rows"][:-1]
    wrong.write_text(json.dumps(w))
    why_book = ""
    try:
        verify_real_day("2026-09-03", str(wrong), str(rpath),
                        params=P, now=BAR_AFTER, replay_fn=replay)
    except VerifierRefused as e:
        why_book = str(e)
    ck("A WRONG BOOK REFUSES THE WHOLE VERIFICATION, not the offending arm: "
       "a day whose book does not match the receipt's digest is not the day "
       "the receipt describes",
       "book digest mismatch" in why_book and bsha[:16] in why_book,
       f"'{why_book[:110]}...'")

    # -- 8. the replay seam refuses rather than substituting --------------
    why_seam = ""
    try:
        verify_real_day("2026-09-03", str(bpath), str(rpath),
                        params=P, now=BAR_AFTER)
    except VerifierRefused as e:
        why_seam = str(e)
    ck("WITHOUT A REPLAY ENGINE THE PATH REFUSES BY NAME rather than "
       "substituting a second implementation of the policy -- the engine is "
       "the instrument of record and a second one would measure a different "
       "thing",
       "REPLAY_ENGINE_NOT_RESOLVED" in why_seam,
       f"'{why_seam[:98]}...'")

    # -- 9. an arm declared but absent from the receipt is a STATUS -------
    part = json.loads(rpath.read_text())
    dropped_arm = sorted(part["arms"])[-1]
    part["arms"].pop(dropped_arm)
    ppath = td / "receipt_partial.json"
    ppath.write_text(json.dumps(part))
    out_p = verify_real_day("2026-09-03", str(bpath), str(ppath),
                            params=P, now=BAR_AFTER, replay_fn=replay)
    ck("AN ARM DECLARED IN PARAMS AND ABSENT FROM THE RECEIPT IS A STATUS, "
       "NEVER A SILENT PASS -- and the run is not a verification",
       out_p["arms"][dropped_arm]["status"] == "ABSENT_FROM_THE_RECEIPT"
       and out_p["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is False,
       f"{dropped_arm} absent -> "
       f"{out_p['arms'][dropped_arm]['status']}, verification "
       f"{out_p['IS_A_VERIFICATION_OF_THE_ECONOMICS']}")

    # -- 10. THE FOUR LIMITS, each with a COMPUTED field ------------------
    lims = out["limits"]
    names = [x["limit"] for x in lims]
    dm = next(x for x in lims if x["limit"].startswith("D_E_MINUS_R"))
    ck("THE FOUR LIMITS ARE A COMPUTED LIST, NOT PROSE: each carries a field "
       "this run measured -- D(E-R) is in DE's economic list and the runner "
       "produces it in ZERO places; the book is shared and nothing was "
       "re-scored; the declaration is read by both sides from one file; and "
       "sealed arms are counted",
       len(lims) == 4
       and dm["computed"]["in_DEs_economic_field_list"] is True
       and dm["computed"]["n_places_the_runner_produces_it"] == 0
       and dm["computed"]["so_there_is_nothing_to_compare"] is True
       and all("computed" in x and x["computed"] for x in lims),
       f"{names}; D(E-R) appears in {dm['computed']['n_places_the_runner_produces_it']} "
       f"producing places, so there is nothing to compare")
    lim_s = next(x for x in out_s["limits"]
                 if x["limit"].startswith("SEALED_ECONOMICS"))
    ck("AND THE LIMITS MOVE WITH THE RUN: on the SEALED receipt the sealed "
       "limit counts every arm, on the open one it counts none -- a limits "
       "block that read the same either way would be prose after all",
       lim_s["computed"]["n_arms_sealed_in_this_receipt"]
       == len(params["arms"])
       and next(x for x in lims if x["limit"].startswith("SEALED_ECONOMICS"))[
           "computed"]["n_arms_sealed_in_this_receipt"] == 0,
       f"sealed run: {lim_s['computed']['n_arms_sealed_in_this_receipt']} of "
       f"{lim_s['computed']['n_arms_seen']}; open run: 0")

    # -- 11. the emitted artifact is ONE file and carries the verdict -----
    opath = td / "verdict.json"
    verify_real_day("2026-09-03", str(bpath), str(rpath), output=opath,
                    params=P, now=BAR_AFTER, replay_fn=replay)
    emitted = json.loads(opath.read_text())
    ck("ONE DECLARED ARTIFACT, carrying the computed verdict, the gate, the "
       "book digest, the params pin, DE's field list with its source digest, "
       "and the limits",
       emitted["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is True
       and emitted["params_declaration"]["sha256"] == params["_sha256"]
       and emitted["economic_field_list"]["source_sha256"]
       == de["source_sha256"]
       and emitted["book"]["sha256"] == bsha
       and len(emitted["limits"]) == 4,
       f"{opath.name}: verification "
       f"{emitted['IS_A_VERIFICATION_OF_THE_ECONOMICS']}, params "
       f"{emitted['params_declaration']['sha256'][:16]}, field list from "
       f"{emitted['economic_field_list']['source_sha256'][:16]}")

    # -- 12. NO OVERRIDE FLAG EXISTS ON THE CLI ---------------------------
    src = Path(__file__).resolve().read_text()
    tree = ast.parse(src)
    cli_flags = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    cli_flags.add(a.value)
    #: `--pre-read` is deliberately NOT in this list: it does not move the
    #: bar, it selects a mode that reads no economics on either side of it.
    #: The battery proves that separately -- the pre-read run below is
    #: driven with the clock BEFORE the bar and again AFTER it, and reads
    #: no economics in either.
    banned = {f for f in cli_flags
              if any(w in f.lower() for w in
                     ("now", "clock", "force", "override", "ignore-bar",
                      "skip", "unsafe", "go"))}
    ck("THERE IS NO OVERRIDE FLAG ON THE CLI -- asserted from THIS FILE's "
       "OWN argument parser by AST, not from a comment. The clock is a "
       "PARAMETER so the battery can drive both sides of the predicate; a "
       "bar with a documented way past it is not a bar",
       banned == set(),
       f"CLI flags {sorted(cli_flags)}; none matches now/clock/force/"
       f"override/skip/unsafe/go")

    checks.extend(selftest_pre_read())

    n_fail = sum(1 for c in checks if not c["passed"])
    for c in checks:
        print(("ok   " if c["passed"] else "FAIL ") + c["check"])
        print("       " + c["detail"])
    print(f"\n{'SELFTEST OK' if not n_fail else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {n_fail} failure(s)")
    return checks, n_fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    #: NOT an override: the pre-read is a DIFFERENT verification that reads
    #: no economics, before the bar or after it. The full read still gates.
    ap.add_argument("--pre-read", action="store_true")
    ap.add_argument("--builder-receipt")
    ap.add_argument("--open-book", action="store_true",
                    help="open the day book to recompute the population. "
                         "HEAVY under rule 20 (~2 GB): the wrapper and the "
                         "lock are required")
    ap.add_argument("--day")
    ap.add_argument("--book")
    ap.add_argument("--receipt")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--supersedes", default=None,
                    help="the record this one replaces: R-608's pair is "
                         "written and the prior chain extended")
    ap.add_argument("--what-changed", default=None)
    a = ap.parse_args()
    if a.selftest:
        checks, n_fail = selftest()
        if a.output:
            a.output.write_text(json.dumps({
                "protocol": PROTOCOL + "_FIXTURE",
                "status": "FIXTURE_NO_REAL_BOOK_NO_REAL_DAY",
                "verifier_identity": verifier_identity(),
                "params_declaration": {"path": PARAMS_PATH.name,
                                       "sha256": load_params()["_sha256"]},
                "economic_field_list": de_economic_fields_at_source(),
                "tolerance": TOLERANCE,
                "checks": checks, "n_checks": len(checks),
                "n_failed": n_fail, "both_directions": True,
            }, indent=2, sort_keys=True) + "\n")
        return 1 if n_fail else 0
    if a.pre_read:
        if not (a.day and a.book and a.receipt):
            ap.error("--pre-read needs --day, --book and --receipt")
        r = pre_read_day(a.day, a.book, a.receipt, output=a.output,
                         open_book=a.open_book,
                         supersedes=a.supersedes,
                         what_changed=a.what_changed,
                         builder_receipt=a.builder_receipt)
        print(f"{a.day}: {r['status']} -- "
              f"{r['n_arms_agreeing'] if r['n_arms_agreeing'] is not None else 'NO POPULATION RECOMPUTED'}"
              f"{'' if r['n_arms_agreeing'] is None else '/' + str(r['n_arms_declared']) + ' arms agree'}, "
              f"sealed={r['economic_absence']['sealed']}, "
              f"economics read: NONE "
              f"(verification of the economics="
              f"{r['IS_A_VERIFICATION_OF_THE_ECONOMICS']})")
        return {"PRE_READ_VERIFIED": 0, "INCOMPLETE": 3,
                "PROVENANCE_INCOMPLETE": 3}.get(r["status"], 1)
    if a.day and a.book and a.receipt:
        r = verify_real_day(a.day, a.book, a.receipt, output=a.output)
        print(f"{a.day}: {r['status']} -- verification="
              f"{r['IS_A_VERIFICATION_OF_THE_ECONOMICS']}, "
              f"{r['n_arms_verified']}/{r['n_arms_declared']} arms verified, "
              f"{r['n_arms_sealed']} sealed")
        return 0 if r["IS_A_VERIFICATION_OF_THE_ECONOMICS"] else 1
    ap.error("--selftest, or --pre-read --day <YYYY-MM-DD> --book <path> "
             "--receipt <path> [--builder-receipt <path>] [--output <path>], "
             "or --day/--book/--receipt for the full read")
    return 2




# ------------------------------------------------------------ the PRE-READ

#: Design v12, pinned. The pre-read matches provenance BY DIGEST, so the
#: declaration it matches against must itself be named.


def _fn_source(name: str) -> str:
    """The source text of one function in THIS module, by AST.

    Used so a structural claim -- 'this path never draws a null' -- is
    checked against the code rather than read off a field the same code
    wrote. A field asserting its own honesty proves nothing."""
    src = Path(__file__).resolve().read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise VerifierRefused(f"REFUSED: no function named {name} in this module")


#: REV 76 S0. ***A LEAF WALK CANNOT SEE AN EMPTY CONTAINER.*** `D_E0:
#: 0.0` leaked and `D_E0: []` came back SEALED -- the key was PRESENT and
#: the walk yielded nothing to judge, so a sealed name emitted as `[]`,
#: `{}` (or as `null`, which is a leaf but an easy one to mistake for
#: absence) was invisible. The receipt's own `seal_status` says
#: present-and-ignored never happens; nothing produces it today, because
#: DE's `_strip_economic` removes KEYS. The census is what must make it
#: impossible tomorrow.
#:
#: THE SHARED RULE, in one line, which DE 105 implements on its side and
#: this implements here, independently (R-235):
#:   ***A SEALED NAME PRESENT AS A KEY REFUSES, WHATEVER ITS VALUE --
#:   INCLUDING AN EMPTY CONTAINER AND `null`. ABSENCE IS THE ONLY SEAL.***
SEAL_RULE = ("A SEALED NAME PRESENT AS A KEY REFUSES, WHATEVER ITS VALUE "
             "-- including an empty container and null. ABSENCE IS THE "
             "ONLY SEAL.")


def _normalise_rule(t: str) -> str:
    return " ".join(str(t).split()).upper().replace("--", "-").strip(" .")


def de_seal_rule_at_source(path: Path | None = None) -> dict:
    """DE's side of the SAME rule, read at DE's source by AST.

    REV 76 S0 asks the two censuses to agree BY CONSTRUCTION. They are not
    one implementation (R-235: DE's declaration, this seat's judgement) --
    what must be shared is the RULE, so it is read from DE's source as a
    STRING and compared with the one this module states. ***A rule DE has
    not declared yet is a NAMED STATUS, never a pass***, and a rule
    declared DIFFERENTLY is a flag: two censuses agreeing by accident is
    what this check exists to prevent."""
    src = Path(path) if path else DE_RUNNER_PATH
    if not src.is_file():
        raise VerifierRefused(
            f"REFUSED: DE's runner is absent at {src}; the shared seal "
            f"rule cannot be read at its source and MUST NOT be assumed.")
    raw = src.read_bytes()
    tree = ast.parse(raw.decode())
    declared, where = None, None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and "SEAL_RULE" in t.id:
                    try:
                        declared, where = ast.literal_eval(node.value), t.id
                    except ValueError:
                        pass
    walkers = sorted(
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "key" in n.name.lower()
        and ("walk" in n.name.lower() or "seal" in n.name.lower()))
    if declared is None:
        return {"status": "DE_HAS_NOT_DECLARED_THE_KEY_WALK_RULE_YET",
                "agrees": None, "key_walkers_found": walkers,
                "source_path": "live/pm_research/de_multiday_gate1_runner.py",
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "this_seat_s_rule": SEAL_RULE,
                "why": ("DE 105 is landing it; an absence is reported by "
                        "name and never read as agreement")}
    same = _normalise_rule(declared) == _normalise_rule(SEAL_RULE)
    return {"status": ("DECLARED_AND_MATCHES" if same
                       else "DECLARED_AND_DIFFERS"),
            "agrees": same, "declared_as": where,
            "key_walkers_found": walkers,
            "source_path": "live/pm_research/de_multiday_gate1_runner.py",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "this_seat_s_rule": SEAL_RULE, "de_s_rule": declared,
            "why": ("the two censuses are independent implementations of "
                    "ONE rule; the rule is what is shared, and it is read "
                    "rather than assumed")}


def _walk_keys(o, path=""):
    """(path, key, value) for EVERY key at every depth -- including keys
    whose value is an empty container, which a leaf walk never reaches."""
    if isinstance(o, dict):
        for k, v in o.items():
            q = f"{path}.{k}" if path else str(k)
            yield q, str(k), v
            yield from _walk_keys(v, q)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk_keys(v, f"{path}[{i}]")


def _walk_paths(o, path=""):
    """(path, value) for every leaf, at full depth, dicts and lists."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk_paths(v, f"{path}.{k}" if path else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk_paths(v, f"{path}[{i}]")
    else:
        yield path, o


def economic_absence(receipt) -> dict:
    """DE's economic field list must be ABSENT at every depth.

    A LEAK IS NAMED AND NEVER READ. The path is recorded; the value is not
    copied anywhere, not into this dict, not into a message, not into a log
    line. A checker that reported `D_E0 = 6.13 leaked` would have published
    the number it exists to protect."""
    #: R-663: JUDGED BY THE LIST IN FORCE WHEN THIS RECEIPT WAS PRODUCED,
    #: not by today's. Reading the list flat made this function report the
    #: programme's FIRST SEALED DAY as `sealed False, n_leaked_fields 6`
    #: -- for carrying three counts that were OPEN BY RULING when it was
    #: emitted. The scope comes from DE's own per-name map; the version
    #: comes from the receipt itself, by a PAIR, or the full list applies.
    inforce = fields_in_force_for(receipt if isinstance(receipt, dict)
                                  else {})
    judged = inforce["fields"]
    walked = list(_walk_paths(receipt))
    #: REV 76 S0: THE KEYS, not the leaves. A sealed name present as a key
    #: refuses whatever its value -- `[]`, `{}` and `null` included.
    keys = list(_walk_keys(receipt))
    leaks = sorted({q for q, k, _ in keys if k in judged})
    empties = sorted({q for q, k, v in keys
                      if k in judged
                      and (v is None or (isinstance(v, (list, dict, str))
                                         and len(v) == 0))})
    return {"economic_fields_declared": list(ECONOMIC_FIELDS),
            "judged_against": list(judged),
            "n_judged_against": len(judged),
            "design_version_of_this_receipt": inforce["design_version"],
            "design_version_resolved": inforce["version_resolved"],
            "how_the_version_was_resolved": inforce["how_resolved"],
            "version_evidence": inforce["version_evidence"],
            "sealed_only_from_a_later_version": inforce[
                "sealed_only_from_a_later_version"],
            "judged_against_the_full_list": inforce[
                "judged_against_the_full_list"],
            "why_scoped": inforce["why"],
            #: REV 70 / R-656: NAME WHAT EACH COUNT COUNTS. This walks THE
            #: RECEIPT; `emitted_census` walks THIS SEAT'S OWN RECORD. The
            #: register read them as one number ("0 leaked in 299 leaves")
            #: and 299 is the RECORD's leaf count, not the receipt's.
            "walks": "THE RECEIPT under test, BY KEY",
            "the_rule": SEAL_RULE,
            "n_receipt_keys_walked": len(keys),
            "n_receipt_leaves_walked": len(walked),
            "n_leaked_that_a_leaf_walk_could_not_see": len(empties),
            "leaked_but_empty_or_null_paths": empties,
            "why_by_key": (
                "a leaf walk never reaches an empty container, so a sealed "
                "name emitted as `[]` or `{}` was PRESENT AND IGNORED -- "
                "the one state the receipt's own seal_status says cannot "
                "happen. Absence is the only seal"),
            "not_to_be_confused_with": (
                "`emitted_census.n_leaves_emitted`, which walks THIS "
                "RECORD -- the anti-echo control on what this seat itself "
                "publishes"),
            "n_leaked_fields": len(leaks),
            "leaked_field_paths": leaks,
            "sealed": not leaks,
            "values_were_not_read": True,
            "why_paths_only": (
                "a leak is reported BY NAME. Echoing the value would publish "
                "the number the seal exists to withhold, which is the "
                "failure this check exists to prevent -- not a lesser one")}


#: The receipt's own fields whose STRINGS may never be echoed. A refusal
#: reason quotes the very moments the seal withholds in order to explain
#: itself, so its text is treated as sealed material.
#: NARROW ON PURPOSE. The first version matched every `why` and `detail`
#: field, whose numbers are DECLARED thresholds -- 0.25, 30, 500, 14 -- that
#: this verifier legitimately repeats everywhere. Watching those made the
#: census refuse its own honest output. What is sealed material is the R4
#: REASONS text, which quotes the null moments to explain itself, and any
#: string that NAMES a sealed quantity.
FORBIDDEN_ECHO_MARKERS = ("reasons",)
MOMENT_FIELDS = ("null_mean", "null_sd", "sd_over_abs_mean", "Z", "D_E0",
                 "p_location")
#: How many significant figures a numeric token must carry before it can be
#: evidence that a sealed float leaked. Sealed moments are long floats; a
#: one- or two-figure token is noise, and treating it as evidence made this
#: census flag the digits inside its own protocol string.
MIN_SIG_DIGITS_TO_BE_EVIDENCE = 6

NUM_TOKEN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _textual_forms(v: float) -> set:
    """The ways one number can appear INSIDE a string.

    A value does not have to be emitted as a number to be emitted. DE 85's
    own finding was a sealed quantity riding out in the TEXT of a refusal
    reason, where a leaf-typed scan cannot see it."""
    out = {str(v), repr(v), f"{v}"}
    try:
        for fmt in (".1f", ".2f", ".3f", ".4f", ".6f", ".6g", ".9g", ".12g",
                    ".15g", "g", "e", ".3e", ".6e"):
            out.add(format(float(v), fmt))
        out.add(str(int(float(v))) if float(v).is_integer() else str(v))
    except (TypeError, ValueError, OverflowError):
        pass
    return {x for x in out if x and any(c.isdigit() for c in x)}


def emitted_census(emitted: dict, receipt) -> dict:
    """PROVE the emission carried no economic value -- INCLUDING AS TEXT.

    Three independent checks, and the third is REV 49 section 2.4's:

      (a) no economic field NAME appears in what was emitted;
      (b) no numeric LEAF equal to a sealed value appears in it;
      (c) NO SEALED NUMBER APPEARS INSIDE A STRING. Checks (a) and (b) are
          both leaf-typed and both blind to a value embedded in prose --
          which is exactly the class DE 85 found: a sealed quantity carried
          in the TEXT of a refusal reason. Every numeric token in every
          string-valued field is parsed and compared, AND every sealed
          value is rendered in its textual forms and searched for.
          BOTH DIRECTIONS, because a token scan misses a value written in a
          format it does not parse back, and a form scan misses a value
          written in a format nobody listed.
    """
    #: REV 76 S0: BY KEY. `{"D_E0": []}` in an emission of mine carries
    #: the NAME of a sealed quantity and no leaf -- check (a) is about the
    #: NAME, so it must see the key whatever the value is.
    names = sorted({q for q, k, _ in _walk_keys(emitted)
                    if k in ECONOMIC_FIELDS})
    econ_vals = {v for p, v in _walk_paths(receipt)
                 if p.rsplit(".", 1)[-1].split("[")[0] in ECONOMIC_FIELDS
                 and isinstance(v, (int, float))
                 and not isinstance(v, bool)}
    #: the null MOMENTS specifically -- they are the ones that travel in
    #: prose, because a refusal reason quotes them to explain itself.
    moment_vals = {v for p, v in _walk_paths(receipt)
                   if p.rsplit(".", 1)[-1].split("[")[0] in MOMENT_FIELDS
                   and isinstance(v, (int, float))
                   and not isinstance(v, bool)}
    #: AND THE ONE THE FIRST VERSION MISSED. Both sets above are drawn from
    #: the receipt's NUMERIC leaves -- and a properly SEALED receipt has
    #: none, so the watch list came out EMPTY and a number hidden in prose
    #: had nothing to be compared against. The census could only ever catch
    #: a leak in a receipt that had already leaked.
    #: The receipt's own PROSE is the third source: any numeric token
    #: sitting in a string the pre-read is forbidden to echo -- a refusal
    #: reason quoting `null sd 0.144337` is the sealed quantity in textual
    #: form, which is precisely what DE 85 found.
    prose_vals, prose_paths = set(), []
    for path_, v in _walk_paths(receipt):
        if not isinstance(v, str):
            continue
        leafish = path_.lower()
        #: WORD BOUNDARIES, AND NO ONE-CHARACTER NAMES. `Z` is a moment
        #: field AND the UTC suffix of every timestamp, so a bare substring
        #: test made every filename in the receipt "sealed material" and
        #: turned its digit runs into watched values -- after which "3" in
        #: `P003` matched. Measured: it flagged this verifier's own protocol
        #: string.
        names_a_sealed_quantity = any(
            re.search(r"\b" + re.escape(f.lower().replace("_", "[ _]")) + r"\b",
                      v.lower())
            for f in MOMENT_FIELDS if len(f) >= 4)
        if not (any(m in leafish for m in FORBIDDEN_ECHO_MARKERS)
                or names_a_sealed_quantity):
            continue
        prose_paths.append(path_)
        for m in NUM_TOKEN.finditer(v):
            try:
                prose_vals.add(float(m.group()))
            except ValueError:
                continue
    #: DECLARED thresholds are public by declaration and appear in honest
    #: prose everywhere. Watching them would make the census refuse its own
    #: correct output -- which it did, on the first attempt.
    declared = set()
    try:
        _p = load_params()
        for k in ("sd_floor_fraction", "min_decisions_per_arm_day",
                  "min_draws_per_arm_day", "alpha", "multiplicity_m",
                  "expected_G", "head_overlap_floor", "per_day_deadline_s"):
            if isinstance(_p.get(k), (int, float)):
                declared.add(float(_p[k]))
    except Exception:                                         # noqa: BLE001
        pass
    declared |= {0.0, 1.0, 2.0, 100.0}
    watched = (econ_vals | moment_vals | prose_vals) - declared
    emitted_vals = {v for _, v in _walk_paths(emitted)
                    if isinstance(v, (int, float))
                    and not isinstance(v, bool)}
    echoed = sorted(watched & emitted_vals)

    strings = [(p, v) for p, v in _walk_paths(emitted) if isinstance(v, str)]

    def _sig_digits(tok: str) -> int:
        return len(tok.replace("-", "").replace(".", "").lstrip("0"))

    #: WHOLE TOKENS, NOT SUBSTRINGS. The first version searched for each
    #: watched value's textual FORMS as raw substrings, and a short form like
    #: "3.2" matches inside "63.21" -- it flagged this verifier's own honest
    #: output ten times over. A number is only evidence of a leak if it
    #: appears as a COMPLETE token, and only if it carries enough precision
    #: to identify the value it came from.
    text_hits = []
    for path_, sval in strings:
        for m in NUM_TOKEN.finditer(sval):
            tok = m.group()
            try:
                t = float(tok)
            except ValueError:
                continue
            #: A SEALED MOMENT IS A FLOAT WITH MANY DIGITS. A token of one
            #: or two significant figures is never evidence that one leaked
            #: -- and treating it as such is how this check first flagged
            #: the digits inside its own protocol string.
            if _sig_digits(tok) < MIN_SIG_DIGITS_TO_BE_EVIDENCE:
                continue
            for w in watched:
                exact = (t == w)
                near = format(t, ".6g") == format(float(w), ".6g")
                if exact or near:
                    text_hits.append({
                        "path": path_,
                        "how": ("exact numeric token" if exact
                                else "token matching to 6 significant "
                                     "figures"),
                        "min_sig_digits_required":
                            MIN_SIG_DIGITS_TO_BE_EVIDENCE,
                        "token_significant_digits": _sig_digits(tok),
                        "NOTE": "the value is NOT reproduced here"})
                    break

    #: dedupe on path+how, and NEVER carry the value itself
    seen, uniq = set(), []
    for h in text_hits:
        k = (h["path"], h["how"])
        if k not in seen:
            seen.add(k)
            uniq.append(h)
    return {"n_leaves_emitted": sum(1 for _ in _walk_paths(emitted)),
            "n_economic_field_names_in_the_emission": len(names),
            "economic_field_names_in_the_emission": sorted(names),
            "n_watched_values_from_the_receipt": len(watched),
            "n_watched_from_numeric_leaves": len(econ_vals | moment_vals),
            "n_watched_from_the_receipts_PROSE": len(prose_vals - declared),
            "n_declared_thresholds_excluded": len(
                (econ_vals | moment_vals | prose_vals) & declared),
            "why_declared_thresholds_are_excluded": (
                "0.25, 30, 500 and the rest are public BY DECLARATION and "
                "appear in honest prose everywhere. Watching them made this "
                "census refuse its own correct output on the first attempt"),
            "receipt_string_paths_that_may_not_be_echoed": sorted(
                set(prose_paths))[:12],
            "why_the_prose_matters": (
                "a SEALED receipt has no economic numeric leaves, so a watch "
                "list built from them alone is EMPTY -- and a number hidden "
                "in prose has nothing to be compared against. The census "
                "could then only catch a leak in a receipt that had already "
                "leaked, which is no catch at all"),
            "n_of_them_echoed_as_a_NUMERIC_LEAF": len(echoed),
            "n_string_fields_scanned": len(strings),
            "n_of_them_carrying_a_watched_number_AS_TEXT": len(uniq),
            "string_hits_by_path_only": uniq[:20],
            "clean": not names and not echoed and not uniq,
            "why_three_checks": (
                "a name check alone passes while the NUMBER rides out under "
                "a different key; a leaf check alone passes while the number "
                "rides out INSIDE A STRING -- REV 49 section 2.4, and the "
                "exact class DE 85 found. The third check reads the prose, "
                "as WHOLE TOKENS: matching textual forms as raw substrings "
                "flagged this verifier's own honest output, because '3.2' "
                "sits inside '63.21'"),
            "values_are_never_reproduced_here": (
                "a hit is reported by PATH and by HOW. Printing the value "
                "would publish what the seal withholds, in the very field "
                "that exists to prevent it"),
            }


def supersession_block(prior: str | Path, *, what_changed: str,
                       what_did_not: str) -> dict:
    """R-608's PAIR plus the chain the prior record already carried.

    DA 91 wrote this block by hand after the emission; a block that lives
    outside the emitter is a block that can be forgotten, and REV 73 S0's
    correction is unconditional. It is part of the module now, with the
    chain EXTENDED rather than replaced -- a record whose chain forgets its
    grandparent has no provenance, only a parent."""
    f = Path(prior)
    if not f.is_file():
        raise VerifierRefused(
            f"REFUSED: SUPERSEDED_RECORD_NOT_PRESENT -- {f}. A link is the "
            f"pair {{path, sha256}} landing on ONE PRESENT file; a "
            f"half-written link refuses BY NAME, never as 'no link'.")
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    try:
        prior_chain = (json.loads(f.read_text()).get("supersedes")
                       or {}).get("chain") or []
    except ValueError:
        prior_chain = []
    return {"path": f.name, "sha256": sha,
            "chain": [list(x) for x in prior_chain] + [[f.name, sha]],
            "the_link_is_the_PAIR": ["path", "sha256"],
            "v1_untouched": True,
            "what_changed": what_changed,
            "what_did_NOT_change": what_did_not}


def pre_read_day(day: str, book_path: str, receipt_path: str, *,
                 output: Path | None = None, open_book: bool = False,
                 params: dict | None = None,
                 now: datetime.datetime | None = None,
                 supersedes: str | Path | None = None,
                 what_changed: str | None = None,
                 builder_receipt: str | None = None) -> dict:
    """THE PRE-READ. Everything the runbook promises before the bar, and
    NOTHING that the bar exists to schedule.

    It runs BEFORE 2026-09-09T00:06Z on purpose -- that is the gap this
    closes -- and it reads NO economics, before or after. The two gates are
    independent: the read bar schedules the READ, the seal withholds the
    NUMBERS, and this mode is on the far side of neither.

    NO REPLAY ENGINE IS RESOLVED AND NO NULL IS DRAWN. Decisions and
    per-side counts come from the book's scores against the declared thetas,
    which is arithmetic on the book alone; D(E0), the null and everything
    downstream need the replay, and the pre-read never asks for it. That is
    not a convention -- it is why this mode cannot leak."""
    params = params or load_params()
    gate = gate_is_open(params, now)

    rp = Path(receipt_path)
    if not rp.is_file():
        raise VerifierRefused(f"REFUSED: receipt absent at {receipt_path}")
    receipt = json.loads(rp.read_text())
    arms_in = _receipt_arms(receipt)
    if not arms_in:
        raise VerifierRefused(
            "REFUSED: the receipt carries no arm blocks. An empty receipt is "
            "a FAILURE, not a day with nothing in it.")

    #: (a) THE BOOK, against the receipt's OWN field and, when supplied,
    #: against BE's builder receipt. Two independent bindings.
    r_sha = _receipt_book_digest(receipt)
    if not r_sha:
        raise VerifierRefused(
            "REFUSED: the receipt names no book digest, so the book it "
            "describes cannot be identified. An unpinned book is not a day.")
    book_meta = verify_book_digest(book_path, r_sha)
    builder = {"supplied": False}
    if builder_receipt:
        bp = Path(builder_receipt)
        if not bp.is_file():
            raise VerifierRefused(
                f"REFUSED: builder receipt absent at {builder_receipt}")
        br = json.loads(bp.read_text())
        b_sha = _find_first(br, ("book_sha256", "sha256", "digest"))
        builder = {"supplied": True, "path": bp.name,
                   "sha256": hashlib.sha256(bp.read_bytes()).hexdigest(),
                   "book_digest_in_the_builder_receipt": b_sha,
                   "agrees_with_the_day_receipt": b_sha == r_sha}
        if b_sha != r_sha:
            raise VerifierRefused(
                f"REFUSED: BE's builder receipt names book {b_sha} and the "
                f"day receipt names {r_sha}. Two receipts describing "
                f"different books is not a day this verifier may check.")

    #: THE FULL READ ALWAYS OPENS THE BOOK -- it is the verification of the
    #: economics and there is nothing to verify without it.
    #: THE POPULATION HALF IS OPTIONAL AND ITS ABSENCE IS A NAMED
    #: STATUS, never a silent pass (rule 11). Everything else the
    #: pre-read does -- the two book bindings, the provenance, the seal
    #: census and the landing record -- needs no book CONTENTS at all.
    book_refusal = None
    try:
        bk = load_day_book(book_path, open_book=open_book,
                           expected_sha256=r_sha)
        rows = bk["rows"]
    except VerifierRefused as e:
        if open_book:
            raise
        book_refusal, bk, rows = str(e), None, []

    #: (b) THE POPULATION, recomputed from the book at the declared thetas.
    #: NO REPLAY. NO NULL.
    arms_out, agree = {}, []
    for arm, spec in ([] if book_refusal
                      else sorted(params["arms"].items())):
        r_arm = arms_in.get(arm)
        if r_arm is None:
            arms_out[arm] = {"status": "ABSENT_FROM_THE_RECEIPT"}
            agree.append(False)
            continue
        vec = bk["scores_by_arm"].get(arm)
        if vec is None or len(vec) != len(rows):
            arms_out[arm] = {"status": "BOOK_SCORES_UNUSABLE_FOR_THIS_ARM",
                             "n_rows": len(rows),
                             "n_scores": (None if vec is None else len(vec))}
            agree.append(False)
            continue
        scored = [dict(r, score=float(sv)) for r, sv in zip(rows, vec)]
        dec = da_decisions(scored, float(spec["theta"]))
        seed_mine = da_seed_for(r_sha, arm)
        seed_theirs = (r_arm.get("seed")
                       or (r_arm.get("draw_provenance") or {}).get("seed"))
        radm = r_arm.get("admissibility") or {}
        checks, bad = [], []

        def cmp(name, mine, theirs):
            if theirs is None:
                checks.append({"field": name, "state": "ABSENT_IN_RECEIPT",
                               "mine": mine})
                return
            ok = mine == theirs
            checks.append({"field": name,
                           "state": "MATCH" if ok else "MISMATCH",
                           "mine": mine, "receipt": theirs})
            if not ok:
                bad.append(name)

        cmp("n_decisions", dec["n_decisions"], radm.get("n_decisions"))
        cmp("seed", seed_mine, seed_theirs)
        r_by_side = (r_arm.get("by_side")
                     or (r_arm.get("decisions") or {}).get("by_side"))
        cmp("by_side", dec["by_side"], r_by_side)
        #: the DECISION half of R4 is arithmetic on the book; the sd half
        #: needs the null and is NOT verifiable before the read.
        dec_ok = dec["n_decisions"] >= params["min_decisions_per_arm_day"]
        arms_out[arm] = {
            "status": "PRE_READ_AGREES" if not bad else "FLAGGED",
            "recomputed": {"n_decisions": dec["n_decisions"],
                           "by_side": dec["by_side"],
                           "theta_declared": float(spec["theta"])},
            "seed_recomputed": seed_mine,
            "checks": checks, "n_mismatches": len(bad),
            "mismatched_fields": bad,
            "receipt_status": r_arm.get("status"),
            "receipt_admissibility_status": radm.get("status"),
            "R4_decision_half": {
                "n_decisions": dec["n_decisions"],
                "min_declared": params["min_decisions_per_arm_day"],
                "passes": dec_ok},
            "R4_sd_half_is_NOT_verifiable_before_the_read": (
                "the sd floor compares the null's sd against its mean, and "
                "both are sealed. Verifying half a predicate and reporting "
                "it as the predicate is the error this field exists to "
                "prevent"),
            "sd_over_abs_mean_present_in_the_sealed_receipt": (
                "sd_over_abs_mean" in radm),
            #: REV 49 section 2.5: the CONSISTENCY, computed per receipt.
            #: The ratio is a quotient of two sealed quantities; whether it
            #: survives is DE's to decide, and this reports whether THIS
            #: receipt agrees with DE's CURRENT list. A v12-shaped receipt
            #: under a v13 list is INCONSISTENT -- and that is a real
            #: signal, not noise: it says the receipt was produced by older
            #: code, which is a provenance fact worth surfacing.
            "sd_over_abs_mean_consistency": {
                "present_in_this_receipt": "sd_over_abs_mean" in radm,
                "in_DEs_current_field_list":
                    "sd_over_abs_mean" in ECONOMIC_FIELDS,
                "consistent": (("sd_over_abs_mean" in radm)
                               is not ("sd_over_abs_mean" in ECONOMIC_FIELDS)),
                "reading": (
                    "present while DE's list says it should be stripped: "
                    "this receipt was produced by code older than the list. "
                    "A PROVENANCE signal, not a defect in the day"
                    if ("sd_over_abs_mean" in radm
                        and "sd_over_abs_mean" in ECONOMIC_FIELDS) else
                    "the receipt agrees with DE's current field list"),
            },
        }
        agree.append(not bad)

    #: (c) PROVENANCE, matched by digest.
    de = de_economic_fields_at_source()
    prov = {
        #: NOT A TYPED PREFIX. `306bfdb0` was v5's digest, hardcoded here --
        #: and it FLAGGED the moment params moved to v6, which is the third
        #: time this seat has pinned another seat's current value. What is
        #: checked is that the declaration resolved is the NEWEST present
        #: and that its digest is RECORDED; the digest itself is evidence,
        #: not a constant to match.
        "params_named_by_the_receipt": params_check(receipt),
        "params": {"path": Path(params["_path"]).name,
                   "sha256": params["_sha256"],
                   "is_the_newest_present": _params_is_newest(params),
                   "versions_present": _params_versions(),
                   "matches": _params_is_newest(params)},
        "design": design_check(receipt, params),
        "runner_economic_field_list": de,
        "verifier": verifier_identity(),
    }

    #: (d) THE SEAL. Absent = good. A leak is NAMED, never read.
    absence = economic_absence(receipt)

    #: REV 54 section 1.3: the digest is computed ONCE and written into both
    #: fields from that one value. Two `hashlib.sha256(rp.read_bytes())`
    #: calls beside each other agree today and nothing said they must --
    #: and DE resolves conjunct 3 through the copy this seat was not
    #: checking, which is the conjunct that stops a re-roll.
    rp_sha = hashlib.sha256(rp.read_bytes()).hexdigest()

    out = {
        "protocol": PROTOCOL + "_PRE_READ_AND_LANDING_RECORD",
        "mode": "PRE_READ",
        "day": day,
        #: REV 50 section 3.3 item 3. THIS ARTIFACT IS THE DECLARED LANDING
        #: RECORD of the day's sealed receipt digest. Nothing else in the
        #: programme records it, and without it a receipt re-emitted after
        #: the fact is indistinguishable from the one the read was
        #: scheduled against.
        "is_the_declared_LANDING_RECORD": True,
        #: REV 73 S0: the superseding link is written BY THE EMITTER now.
        **({"supersedes": supersession_block(
            supersedes,
            what_changed=(what_changed or
                          "re-emitted; see the round's report"),
            what_did_not=(
                "the receipt digest this record exists to carry. Conjunct "
                "3 reads the CHAIN HEAD, and the head still carries "
                + str(rp_sha)))} if supersedes else {}),
        #: REV 52 section 2.4: the record's DECLARED NAME and its correction
        #: path, declared by the seat that writes it.
        "landing_record_naming": pre_read_artifact_naming(),
        "landing_record": {
            "day": day,
            "receipt_path": rp.name,
            #: REV 54 section 1.3. ONE CALL, TWO FIELDS, EQUALITY ASSERTED
            #: AT WRITE TIME -- and `receipt.sha256` is the AUTHORITATIVE
            #: one, because that is the field DE's conjunct 3 resolves
            #: through. This mirror exists for readers of the landing block
            #: and is checked against the authority, never trusted beside it.
            "receipt_sha256": rp_sha,
            "THIS_FIELD_IS_A_MIRROR": landing_digest_fields()["mirror"],
            "the_authoritative_field_is": landing_digest_fields()[
                "authoritative"],
            "recorded_at_utc": (now or datetime.datetime.now(
                datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "what_it_is_for": (
                "the read gate's `sealed receipts at their LANDING digests` "
                "conjunct compares the receipt on disk against THIS digest. "
                "A receipt re-emitted after the pre-read moves the digest "
                "and the conjunct fails by name"),
        },
        "runs_before_the_bar_by_design": True,
        "gate_state_recorded_not_enforced": gate,
        "why_no_bar_here": (
            "the bar schedules the READ of the economics. This mode reads "
            "none, before or after it -- the two gates are independent and "
            "this is on the far side of neither"),
        "replay_engine_used": False,
        "null_drawn": False,
        "why_that_is_structural": (
            "decisions and per-side counts are arithmetic on the book's own "
            "scores against the declared thetas. D(E0), the null and "
            "everything downstream need the replay, and this path never "
            "resolves one -- so it cannot compute an economic value, let "
            "alone emit it"),
        "book": book_meta,
        "population_recomputed_from_the_book": (book_refusal is None),
        "the_open_book_contract": {
            "order": PICKLE_ORDER,
            "what_opening_a_book_is": PICKLE_EXECUTION_NOTE,
            "book_was_opened": bool(open_book and not book_refusal),
            "digest_pinned_to": ("the receipt's own book digest, verified "
                                 "before any open"),
            "rule_20": ("the heavy lock and the 8 GiB cap, in the service "
                        "form -- this run did not open a book"
                        if not open_book else
                        "the heavy lock and the 8 GiB cap, in the service "
                        "form"),
        },
        "why_the_population_was_not_recomputed": book_refusal,
        "n_arms_with_a_recomputed_population": len(arms_out),
        "builder_receipt": builder,
        #: THE AUTHORITATIVE DIGEST. Both seats now resolve through this
        #: field; the landing block's copy mirrors it and is asserted equal
        #: at write time.
        "receipt": {"path": rp.name, "sha256": rp_sha},
        "landing_digest_fields": landing_digest_fields(),
        "provenance": prov,
        "economic_absence": absence,
        "arms": arms_out,
        "n_arms_declared": len(params["arms"]),
        #: REV 70 section 2: **NONE, NEVER 0.** `n_arms_agreeing: 0` read
        #: alone says "the two arms DISAGREE with the receipt"; when no
        #: population was recomputed it means NOTHING WAS COMPARED. Same
        #: rule DA 84 applied to the absence flag, one field over: a
        #: predicate that was not evaluated is None.
        "n_arms_agreeing": (None if book_refusal
                            else sum(1 for a in agree if a)),
        "why_n_arms_agreeing_is_none": (
            "no population was recomputed, so no arm was compared. 0 would "
            "read as two arms DISAGREEING" if book_refusal else None),
        "IS_A_VERIFICATION_OF_THE_ECONOMICS": False,
        "why_never_a_verification_of_the_economics": (
            "this mode reads no economic field and computes no economic "
            "quantity. It is a verification of the POPULATION, the SEED, the "
            "STATUSES and the PROVENANCE, and saying so is the point: a "
            "pre-read reported as a verification would be the sealed-receipt "
            "error in a new place"),
        "status": None,
    }
    #: The DECLARATIONS gate the verdict: params and design v12 are external
    #: artifacts this run matched by digest. The verifier's OWN
    #: committed-bytes flag is REPORTED at top level rather than folded in --
    #: DA 63's finding is that the durable citation is the CONTENT DIGEST,
    #: which this artifact carries either way, and a reader who needs the
    #: stricter reading has the flag in front of them.
    #: the receipt's OWN params binding gates alongside the design's. A gap
    #: there is PROVENANCE_INCOMPLETE, not a flag -- the same three states.
    pnamed = prov["params_named_by_the_receipt"]
    prov_ok = bool(prov["params"]["matches"] and prov["design"]["matches"]
                   and pnamed["matches"] is not False)
    params_incomplete = pnamed["matches"] is None
    #: REV 70 section 2: THE STATUS IS RULED FROM THE FACTS, and FLAGGED
    #: is for a FLAG IN THE DAY. `FLAGGED` collapsed three separate facts --
    #: the seal HOLDS, the provenance is INCOMPLETE, a half was REFUSED and
    #: not attempted -- into the one word a reader takes as "something is
    #: wrong with this day". The three states are named and the reasons
    #: travel with them.
    #: A MISMATCH AND AN ABSENCE ARE DIFFERENT FACTS. A DECLARED digest the
    #: file does not have is a CONTRADICTION -- a flag. A pin the receipt
    #: never made is an ABSENCE -- incomplete. Collapsing them was the
    #: defect REV 70 names, and collapsing them the other way would be
    #: worse.
    def _is_a_contradiction(blk):
        """A DECLARED digest the file does not have. `matches: False`
        because NOTHING was declared is an ABSENCE, and the design block
        says so in `sha256_declared: NOT_PINNED_HERE` -- reading that as a
        contradiction is the very collapse REV 70 section 2 names."""
        if blk.get("matches") is not False:
            return False
        dec = str(blk.get("sha256_declared") or "")
        return len(dec) == 64 and all(c in "0123456789abcdef" for c in dec)

    _mismatch = [k for k in ("params", "design")
                 if _is_a_contradiction(prov[k])]
    _unpinned = [k for k in ("params", "design")
                 if prov[k].get("matches") is False
                 and not _is_a_contradiction(prov[k])]
    if pnamed.get("matches") is False:
        _mismatch.append("params_named_by_the_receipt")
    _incomplete_because = []
    if params_incomplete:
        _incomplete_because.append(
            "the receipt names no params declaration, so a conjunct's INPUT "
            f"is missing ({pnamed.get('status')})")
    for _k in _unpinned:
        _incomplete_because.append(
            f"the {_k} pin declares no digest to verify against "
            f"({prov[_k].get('sha256_declared')})")
    if book_refusal:
        _incomplete_because.append(
            "the population half was REFUSED BY NAME and NOT ATTEMPTED: "
            + book_refusal.split(" -- ")[0].replace("REFUSED: ", ""))
    _day_flag = bool(agree) and not all(agree)
    _seal_holds = bool(absence["sealed"]) and not absence.get(
        "leaked_field_paths")
    out["status"] = (
        "PRE_READ_VERIFIED" if (agree and all(agree) and _seal_holds
                                and prov_ok and not params_incomplete
                                and not book_refusal)
        #: A FLAG **IN THE DAY**: something WAS compared and disagreed, a
        #: declared digest contradicts its file, or the seal leaked.
        else "FLAGGED" if (_day_flag or not _seal_holds or _mismatch)
        #: the ONLY gap is the params pin -- the state this scheme already
        #: had a name for.
        else "PROVENANCE_INCOMPLETE" if ((params_incomplete or _unpinned)
                                         and not book_refusal)
        #: nothing disagreed; something was not evaluated.
        else "INCOMPLETE")
    out["incomplete_because"] = _incomplete_because or None
    out["provenance_mismatches"] = _mismatch or None
    out["provenance_unpinned"] = _unpinned or None
    out["why_this_status"] = (
        "FLAGGED is reserved for a flag IN THE DAY -- an arm COMPARED and "
        "disagreeing, a DECLARED digest its file does not have, or a seal "
        "that leaked. A clean seal beside an unevaluated half and a missing "
        "provenance INPUT is INCOMPLETE, and a reader must be able to tell "
        "those apart (REV 70 section 2)")
    out["seal_holds"] = _seal_holds
    out["exit_code_meaning"] = {
        "PRE_READ_VERIFIED": 0, "PROVENANCE_INCOMPLETE": 3,
        "INCOMPLETE": 3, "FLAGGED": 1,
        "why": "the four-state scheme this seat already uses (DA 71)"}
    out["provenance_all_matched"] = prov_ok
    out["code_is_committed"] = bool(
        prov["verifier"]["producing_code_is_the_committed_bytes"])
    out["verifier_sha256"] = prov["verifier"]["sha256"]
    out["why_committed_bytes_is_reported_not_gating"] = (
        "the durable citation is the verifier's CONTENT DIGEST, carried "
        "here either way; a commit id can be rewritten by a rebase and a "
        "worktree's HEAD is whatever it was last detached at. The flag is "
        "in front of the reader rather than folded silently into a verdict")
    out["emitted_census"] = emitted_census(out, receipt)
    if not out["emitted_census"]["clean"]:
        raise VerifierRefused(
            f"REFUSED: the emission carries "
            f"{out['emitted_census']['n_economic_field_names_in_the_emission']}"
            f" economic field name(s), echoes "
            f"{out['emitted_census']['n_of_them_echoed_as_a_NUMERIC_LEAF']} "
            f"as numeric leaves and carries "
            f"{out['emitted_census']['n_of_them_carrying_a_watched_number_AS_TEXT']}"
            f" inside string fields. A pre-read that emits what it exists to "
            f"withhold is worse than no pre-read. Hits BY PATH ONLY: "
            f"{[h['path'] for h in out['emitted_census']['string_hits_by_path_only'][:6]]}")
    if output:
        Path(output).write_text(
            json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    return out


def _receipt_arms(receipt) -> dict:
    """DE emits per-day arm blocks as a LIST; a dict keyed by arm is also
    accepted. Both shapes, because the receipt's shape is DE's to choose."""
    if isinstance(receipt, dict) and isinstance(receipt.get("arms"), dict):
        return receipt["arms"]
    out = {}
    src = None
    if isinstance(receipt, dict):
        for k in ("per_day_sealed_artifacts", "arms", "per_arm"):
            if isinstance(receipt.get(k), list):
                src = receipt[k]
                break
    if src is None and isinstance(receipt, list):
        src = receipt
    for blk in (src or []):
        if isinstance(blk, dict) and blk.get("arm"):
            out[blk["arm"]] = blk
    return out


def _find_first(o, keys):
    for p, v in _walk_paths(o):
        if p.rsplit(".", 1)[-1].split("[")[0] in keys and isinstance(v, str):
            return v
    return None


def _receipt_book_digest(receipt):
    arms = _receipt_arms(receipt)
    for blk in arms.values():
        d = (blk.get("draw_provenance") or {}).get("book_digest")
        if d:
            return d
    return _find_first(receipt, ("book_sha256", "book_digest"))


#: ONE PREDICATE, IN ONE PLACE (`da_root`). REV 57 A.6: DE's resolver finds
#: the canonical ledger without the env, and this module's `_derived_dir()`
#: resolved relative to ITS OWN FILE -- so from a seat worktree it returned
#: the worktree's PARTIAL `data/` even with PM_DATA_ROOT set correctly, and
#: the 09-04 book is invisible there. The read gate COUNTS SEALED RECEIPTS
#: AT A ROOT, so a smaller plausible ledger reads as a pass. Two seats
#: resolving one root by two rules is the defect; this follows the
#: canonical rule and REFUSES BY NAME when the root is not the ledger.
def _derived_dir() -> Path:
    """The CANONICAL ledger's derived directory, or a refusal by name.

    IT CLAIMED THE PROGRAMME'S RESOLVER BEFORE AND IT WAS NOT TRUE:
    `Path(BDR.resolve())` raises TypeError -- `resolve()` returns a DICT --
    so the `except` under it caught every call and the function answered
    with ITS OWN TREE while the docstring said otherwise. A docstring
    asserting a property the code does not have is the defect REV 53 named
    in DE's `digested` field, one module over."""
    import da_root as _R                                      # noqa: PLC0415
    return _R.derived_dir("the Gate-1 read gate's day set")


#: DA 97, FOUND BY RUNNING THE PRE-READ ON THE 09-04 RECEIPT. ***THE
#: RECEIPT NAMED ITS PINS AND THIS VERIFIER REPORTED "the receipt names
#: none".*** DE writes both pairs in a top-level `provenance` block --
#: `provenance.params = {path, sha256}`, `provenance.design = {path,
#: sha256}` -- and these readers looked only at the older
#: `params_declaration` / `declaration.design` shapes, so a receipt that
#: DID carry v14 and v21 by pair came back PROVENANCE_INCOMPLETE. ***A
#: gap reported where the artifact is complete is the same defect as a
#: pass reported where it is not*** -- both are the verifier describing
#: itself instead of the receipt. Every place the receipt may state a pin
#: is read, and two of its own statements that DISAGREE are a named
#: CONFLICT, never a silent choice between them.
def receipt_pin_candidates(receipt: dict, kind: str) -> dict:
    """Every block in which THE RECEIPT ITSELF names a `kind` pin."""
    found = []
    prov = (receipt.get("provenance") or {}).get(kind)
    if isinstance(prov, dict) and (prov.get("path") or prov.get("protocol")):
        found.append({"where": f"provenance.{kind}", "block": prov})
    for where, blk in ((f"{kind}_declaration", receipt.get(
                            f"{kind}_declaration")),
                       (f"declaration.{kind}",
                        (receipt.get("declaration") or {}).get(kind)),
                       (f"{kind}_used", receipt.get(f"{kind}_used"))):
        if isinstance(blk, dict) and (blk.get("path") or blk.get("protocol")):
            found.append({"where": where, "block": blk})
    ident = {(Path(str(f["block"].get("path") or "")).name,
              f["block"].get("sha256")) for f in found}
    return {"found": found, "n_places": len(found),
            "conflict": len(ident) > 1,
            "identities": sorted((n, (h or "")[:16]) for n, h in ident),
            "why": ("the receipt is the authority on what it read; where "
                    "it says so twice and the two disagree, that is a "
                    "finding, not a choice for this verifier to make")}


def params_check(receipt: dict) -> dict:
    """The PARAMS declaration the RECEIPT names, verified at the file it
    names -- the same binding the design pin already had.

    REV 51 section 2.6: the design pin was bound and, two lines away, the
    params pin was still a constant. Round 74 replaced it with
    newest-present, which is better but is still this verifier CHOOSING;
    the receipt should say which declaration it was produced against, and
    that is what gets verified. A receipt naming a params file this
    worktree cannot resolve is a PROVENANCE GAP -- never a fall back to
    whatever happens to be newest, because the newest is not what ran."""
    cand = receipt_pin_candidates(receipt, "params")
    if cand["conflict"]:
        return {"named_by_the_receipt": True, "matches": None,
                "status": "PROVENANCE_CONFLICT_THE_RECEIPT_NAMES_TWO",
                "places": [f["where"] for f in cand["found"]],
                "identities": cand["identities"],
                "why": ("the receipt names its params declaration in more "
                        "than one place and the pins disagree; choosing "
                        "one would be this verifier deciding which of the "
                        "receipt's own statements to believe")}
    blk = cand["found"][0]["block"] if cand["found"] else None
    named_at = cand["found"][0]["where"] if cand["found"] else None
    if not isinstance(blk, dict) or not (blk.get("path") or blk.get("protocol")):
        return {"named_by_the_receipt": False,
                "status": "PROVENANCE_INCOMPLETE_NO_PARAMS_NAMED",
                "matches": None,
                "why": ("the receipt names no params declaration, so the "
                        "bars it was produced against cannot be identified. "
                        "This verifier does NOT substitute its own choice: "
                        "the newest present is not what ran")}
    name = blk.get("path") or ""
    cand = Path(name)
    p = cand if cand.is_absolute() and cand.is_file() else None
    if p is None:
        for base in (HERE / "declarations", _derived_dir()):
            q = base / Path(name).name
            if q.is_file():
                p = q
                break
    if p is None and blk.get("protocol"):
        ver = str(blk["protocol"]).rsplit("_V", 1)[-1]
        q = HERE / "declarations" / f"de_multiday_gate1_params_v{ver}.json"
        p = q if q.is_file() else None
    if p is None:
        return {"named_by_the_receipt": True, "named_at": named_at,
                "path_named": name,
                "protocol_named": blk.get("protocol"),
                "status": "PROVENANCE_INCOMPLETE_PARAMS_UNRESOLVED",
                "matches": None,
                "why": ("the receipt names a params declaration this "
                        "worktree cannot resolve. Verifying against a "
                        "DIFFERENT declaration would be the round-74 error "
                        "again: a verdict about bars the receipt never ran "
                        "under")}
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    declared = blk.get("sha256")
    return {"named_by_the_receipt": True, "named_at": named_at,
            "path": p.name, "sha256": sha,
            "sha256_declared": declared,
            "protocol": json.loads(p.read_text()).get("protocol"),
            "status": ("PARAMS_VERIFIED" if (declared is None or sha == declared)
                       else "PARAMS_DIGEST_MISMATCH"),
            "matches": (True if declared is None else sha == declared),
            "digest_was_declared": declared is not None,
            "why": ("the digest of the params file the RECEIPT names, "
                    "against the digest the receipt declares -- not against "
                    "a constant here and not against whatever is newest")}


def design_check(receipt: dict, params: dict) -> dict:
    """The design declaration, READ FROM THE RECEIPT and verified at the file
    it names.

    REV 49 section 2.6: this was pinned to a v12 prefix in this file and
    never bound to the receipt at all -- so it would keep passing on a v12
    file while the receipt under test was produced against v13 or v14. The
    version is the RECEIPT's to state; this verifies the artifact it names.
    """
    cand = receipt_pin_candidates(receipt, "design")
    if cand["conflict"]:
        return {"found": True, "matches": None,
                "named_by": "the receipt, in two places that disagree",
                "places": [f["where"] for f in cand["found"]],
                "identities": cand["identities"],
                "status": "PROVENANCE_CONFLICT_THE_RECEIPT_NAMES_TWO",
                "why": ("the receipt names its design declaration twice "
                        "and the pins disagree; this verifier will not "
                        "pick which of the receipt's statements to hold "
                        "it to")}
    blk = (cand["found"][0]["block"] if cand["found"]
           else params.get("design_declaration"))
    src = (f"the receipt ({cand['found'][0]['where']})" if cand["found"]
           else "the params declaration (the receipt names none)")
    if not isinstance(blk, dict) or not blk.get("path"):
        return {"found": False, "matches": False, "named_by": src,
                "why": ("no design declaration is named by the receipt or by "
                        "params, so the version under test cannot be "
                        "identified and MUST NOT be assumed")}
    res = resolve_named_path(blk["path"])
    if not res["resolved"]:
        return {"found": False, "matches": False, "named_by": src,
                "path_named": blk["path"],
                "paths_tried": res["paths_tried"], "why": res["why"]}
    p, sha = Path(res["path_hashed"]), res["sha256"]
    return {"found": True, "named_by": src, "path": p.name,
            "path_hashed": res["path_hashed"],
            "path_as_named": res["as_named"], "resolution": res["how"],
            "sha256": sha, "sha256_declared": blk.get("sha256"),
            "protocol_declared": blk.get("protocol"),
            "version_from_the_name": (
                "".join(c for c in p.name.split("design_")[-1][:4]
                        if c.isalnum()) if "design_" in p.name else None),
            "matches": sha == blk.get("sha256"),
            "why": ("the digest of the file the receipt NAMES, against the "
                    "digest the receipt DECLARES -- not against a version "
                    "hardcoded in the verifier")}



def _sealed_de_shape_receipt(d: Path, arms_payload: dict, book_sha: str, *,
                             name: str = "sealed_de.json",
                             leak: tuple | None = None,
                             shape: str = "v13",
                             design: dict | None = None,
                             params_block: dict | None = None) -> Path:
    """A receipt in DE's OWN emitted shape: per-day arm blocks in a LIST,
    each with `admissibility`, `draw_provenance.book_digest`, `seed`, and
    the economic fields STRIPPED at every depth."""
    blocks = []
    for arm, v in sorted(arms_payload.items()):
        econ = v.get("economic") or {}
        blk = {
            "arm": arm, "day": "2026-09-03", "status": v["status"],
            "sealed": True, "sealed_at_every_depth": True,
            "sealed_field_names": list(ECONOMIC_FIELDS),
            "seal_status": "SEALED -- economic fields ABSENT, not "
                           "present-and-ignored",
            "admissibility": {
                "admissible": v["admissibility"]["admissible"],
                "n_decisions": v["admissibility"]["n_decisions"],
                "reasons": v["admissibility"]["reasons"],
                "sd_over_abs_mean": v["admissibility"]["sd_over_abs_mean"],
                "status": v["admissibility"]["status"]},
            "by_side": v["by_side"],
            "seed": v["seed"],
            "draw_provenance": {"arm": arm, "book_digest": book_sha,
                                "seed": v["seed"], "n_draws": v["n_draws"],
                                "recomputed_by_the_runner": True},
        }
        ratio = v["admissibility"].get("sd_over_abs_mean")
        blk = _strip_like_DE(blk)
        #: REV 49 section 2.5. The `iff` could not FIRE on a fixture that
        #: only ever produced the CURRENT shape. A v12-shaped receipt keeps
        #: `sd_over_abs_mean` -- v12's stripper did not know it -- while the
        #: live field list says it should be gone, and THAT is the state the
        #: real 09-03 receipt is in.
        if shape == "v12" and ratio is not None:
            blk["admissibility"]["sd_over_abs_mean"] = ratio
        if leak and leak[0] == arm:
            blk[leak[1]] = econ.get(leak[1], leak[2])
        blocks.append(blk)
    p = d / name
    body = {"day": "2026-09-03", "per_day_sealed_artifacts": blocks,
            "receipt_shape_for_the_fixture": shape}
    if design:
        body["design_declaration"] = design
    if params_block:
        body["params_declaration"] = params_block
    p.write_text(json.dumps(body))
    return p


def selftest_pre_read() -> list:                              # noqa: C901
    """The PRE-READ battery. Returned to the main selftest so the module has
    one check list and one count."""
    checks: list[dict] = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    params = load_params()
    #: the fixture's own declaration -- the clock conjunct alone -- so the
    #: PRE-READ logic is exercised and not the artifact conjunction. The
    #: multi-conjunct file is driven against the REAL params below.
    P = json.loads(json.dumps({k: v for k, v in params.items()
                               if not k.startswith("_")}))
    P[READ_GATE_FIELD] = {CONJUNCTION_FIELD: [
        "the clock >= read_not_before_utc"]}
    P["_path"], P["_sha256"] = params["_path"], params["_sha256"]
    td = Path(tempfile.mkdtemp(prefix="da68_"))
    book = synthetic_book()
    replay = synthetic_replay(book)
    rows = book["rows"]
    bpath, bsha = write_day_book(td, book, params)

    payload = {}
    for a, spec in sorted(params["arms"].items()):
        m = da_arm_day(replay, rows, rows, arm=a,
                       theta=float(spec["theta"]), book_sha=bsha,
                       params=params)
        d = da_decisions(rows, float(spec["theta"]))
        payload[a] = {"status": m["status"], "seed": m["seed"],
                      "n_draws": m["n_draws"], "by_side": d["by_side"],
                      "admissibility": {
                          **m["admissibility"],
                          "sd_over_abs_mean":
                              m["admissibility"]["sd_over_abs_mean"]},
                      "economic": dict(m["economic"] or {})}
    #: the fixture names a REAL design declaration so 2.6's check has a
    #: file to verify against -- whichever version is current.
    #: NUMERICALLY, not lexicographically: `v9` sorts after `v16` as text,
    #: so the old glob named a SUPERSEDED design as "whichever is current".
    dsn = _designs_by_version()
    design = None
    if dsn:
        design = {"path": dsn[-1].name,
                  "sha256": hashlib.sha256(dsn[-1].read_bytes()).hexdigest(),
                  "protocol": "P003_DE_MULTIDAY_GATE1_DESIGN_DECLARATION"}
    #: a real DE receipt NAMES the params it ran under; the fixture's does
    #: too, so REV 51 section 2.6's binding has something to verify.
    pblock = {"path": Path(params["_path"]).name,
              "sha256": params["_sha256"],
              "protocol": params.get("protocol")}
    spath = _sealed_de_shape_receipt(td, payload, bsha, design=design,
                                     params_block=pblock)

    # -- A. it RUNS BEFORE THE BAR and verifies ---------------------------
    pre = pre_read_day("2026-09-03", str(bpath), str(spath),
                       params=P, now=BAR_BEFORE)
    ck("THE PRE-READ RUNS BEFORE THE BAR AND VERIFIES -- which is the whole "
       "gap: the full read gates on 2026-09-09T00:06Z, so the population, "
       "seed, statuses and provenance the runbook promises BEFORE it had "
       "nowhere to be checked",
       pre["status"] == "PRE_READ_VERIFIED"
       and pre["gate_state_recorded_not_enforced"]["open"] is False
       and pre["n_arms_agreeing"] == len(params["arms"]),
       f"clock {BAR_BEFORE.date()} (bar {params['read_not_before_utc']}, "
       f"open={pre['gate_state_recorded_not_enforced']['open']}) -> "
       f"{pre['status']}, {pre['n_arms_agreeing']}/"
       f"{pre['n_arms_declared']} arms agree")

    # -- B. it reads NO economics, and that is STRUCTURAL ------------------
    ck("IT READS NO ECONOMICS AND THAT IS STRUCTURAL, NOT A CONVENTION: no "
       "replay engine is resolved and no null is drawn, so D(E0) and "
       "everything downstream are not merely unreported -- they are "
       "uncomputable on this path",
       pre["replay_engine_used"] is False and pre["null_drawn"] is False
       and pre["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is False
       #: and the claim is checked at the SOURCE, not taken from the field:
       #: `pre_read_day` must not call the arm-day statistic, which is the
       #: only thing on this surface that draws a null.
       and "da_arm_day" not in _fn_source("pre_read_day"),
       f"replay_engine_used={pre['replay_engine_used']}, "
       f"null_drawn={pre['null_drawn']}, "
       f"IS_A_VERIFICATION_OF_THE_ECONOMICS="
       f"{pre['IS_A_VERIFICATION_OF_THE_ECONOMICS']}")

    # -- C. the emitted census PROVES the emission is clean ---------------
    cen = pre["emitted_census"]
    ck("AND THE EMISSION IS PROVEN CLEAN BY A COMPUTED CENSUS, two ways: no "
       "economic field NAME appears in what was emitted, AND none of the "
       "receipt's economic VALUES appears anywhere in it. A name check alone "
       "passes while the number rides out under another key",
       cen["clean"] is True
       and cen["n_economic_field_names_in_the_emission"] == 0
       and cen["n_of_them_echoed_as_a_NUMERIC_LEAF"] == 0,
       f"{cen['n_leaves_emitted']} leaves emitted, "
       f"{cen['n_economic_field_names_in_the_emission']} economic names, "
       f"{cen['n_of_them_echoed_as_a_NUMERIC_LEAF']} echoed values")

    # -- D. AFTER the bar it still runs and still reads nothing -----------
    post = pre_read_day("2026-09-03", str(bpath), str(spath),
                        params=P, now=BAR_AFTER)
    ck("AFTER THE BAR THE SAME MODE STILL RUNS AND STILL READS NO "
       "ECONOMICS -- two gates, independent: the bar schedules the READ, "
       "the seal withholds the NUMBERS, and this mode is on the far side of "
       "neither",
       post["status"] == "PRE_READ_VERIFIED"
       and post["gate_state_recorded_not_enforced"]["open"] is True
       and post["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is False
       and post["emitted_census"]["clean"] is True,
       f"clock past the bar (open="
       f"{post['gate_state_recorded_not_enforced']['open']}) -> "
       f"{post['status']}, economics read: NONE")

    # -- E. a moved population count is FLAGGED ---------------------------
    moved = json.loads(spath.read_text())
    moved["per_day_sealed_artifacts"][0]["admissibility"]["n_decisions"] += 1
    mp = td / "sealed_moved.json"
    mp.write_text(json.dumps(moved))
    pm = pre_read_day("2026-09-03", str(bpath), str(mp), params=params,
                      now=BAR_BEFORE)
    arm0 = moved["per_day_sealed_artifacts"][0]["arm"]
    ck("KNOWN-BAD: A MOVED POPULATION COUNT IS FLAGGED. n_decisions is "
       "recomputed from the book at the declared theta, so a receipt that "
       "claims a different one disagrees with the book it names",
       pm["status"] == "FLAGGED"
       and "n_decisions" in pm["arms"][arm0]["mismatched_fields"],
       f"{arm0}.n_decisions +1 -> {pm['arms'][arm0]['mismatched_fields']}")

    # -- F. a wrong seed is FLAGGED ---------------------------------------
    ws = json.loads(spath.read_text())
    ws["per_day_sealed_artifacts"][0]["seed"] += 1
    ws["per_day_sealed_artifacts"][0]["draw_provenance"]["seed"] += 1
    wp = td / "sealed_wrongseed.json"
    wp.write_text(json.dumps(ws))
    pw = pre_read_day("2026-09-03", str(bpath), str(wp), params=params,
                      now=BAR_BEFORE)
    ck("KNOWN-BAD: A WRONG SEED IS FLAGGED. The seed is re-derived from the "
       "book's digest and the arm name, so a receipt whose seed does not "
       "follow from the book it names drew a different null",
       pw["status"] == "FLAGGED"
       and "seed" in pw["arms"][arm0]["mismatched_fields"],
       f"{arm0}.seed +1 -> {pw['arms'][arm0]['mismatched_fields']}; "
       f"recomputed {pw['arms'][arm0]['seed_recomputed']}")

    # -- G. A LEAKED ECONOMIC FIELD IS FLAGGED WITHOUT ECHOING ITS VALUE --
    leak_val = float(payload[sorted(payload)[0]]["economic"]["D_E0"])
    lp = _sealed_de_shape_receipt(td, payload, bsha, name="sealed_leak.json",
                                  design=design, params_block=pblock,
                                  leak=(sorted(payload)[0], "D_E0",
                                        leak_val))
    pl = pre_read_day("2026-09-03", str(bpath), str(lp), params=params,
                      now=BAR_BEFORE)
    emitted_text = json.dumps(pl, default=str)
    ck("KNOWN-BAD, AND THE HARD HALF: A LEAKED ECONOMIC FIELD IS FLAGGED BY "
       "NAME AND ITS VALUE IS NEVER ECHOED. The leak's PATH is recorded, the "
       "number is not -- a checker that reported `D_E0 = <value> leaked` "
       "would publish exactly what the seal exists to withhold",
       pl["status"] == "FLAGGED"
       and pl["economic_absence"]["sealed"] is False
       and pl["economic_absence"]["n_leaked_fields"] == 1
       and any("D_E0" in p
               for p in pl["economic_absence"]["leaked_field_paths"])
       and repr(leak_val) not in emitted_text
       and pl["emitted_census"]["n_of_them_echoed_as_a_NUMERIC_LEAF"] == 0,
       f"leak named at "
       f"{pl['economic_absence']['leaked_field_paths']}; the value appears "
       f"0 times in the emission and the census confirms "
       f"{pl['emitted_census']['n_of_them_echoed_as_a_NUMERIC_LEAF']} echoed")

    # -- R-663. THE SEAL IS SCOPED PER NAME, AND THE SCOPE IS THE -------
    # -- LIST IN FORCE WHEN THE RECEIPT WAS PRODUCED --------------------
    #: REV 72 S1.4 predicted this seam and it was LIVE here: DE extended
    #: ECONOMIC_FIELDS from eight to eleven, this verifier read the list
    #: FLAT, and the programme's FIRST SEALED DAY came back `sealed False,
    #: n_leaked_fields 6` -- accused by its own instrument for carrying
    #: three counts that were OPEN BY RULING when it was emitted.
    real = (_derived_dir()
            / "p003_de_gate1_day_run_20260903_SEALED__20260906T140155Z.json")
    if real.is_file():
        rrec = json.loads(real.read_text())
        rabs = economic_absence(rrec)
        ck("R-663 ON THE REAL 09-03 RECEIPT -- ***THE SEAL HOLDS, AND THE "
           "ACCUSATION WAS THE INSTRUMENT'S.*** Judged against the list in "
           "force WHEN IT WAS PRODUCED, the programme's first sealed day "
           "is `sealed True, 0 leaked`. The version is not assumed: it is "
           "resolved from the receipt by a PAIR -- its own "
           "`source_identity.carrying_commit` and its own import-closure "
           "digest for DE's design module, which is hashed at that commit "
           "-- and NAMED in the record. ***An opened path is not a pin***: "
           "this run opened a stale `_design_v10` beside `_design_v21`, "
           "and picking one would be a coin toss dressed as evidence",
           rabs["sealed"] is True and rabs["n_leaked_fields"] == 0
           and rabs["design_version_of_this_receipt"] == 21
           and rabs["design_version_resolved"] is True
           and rabs["n_judged_against"] == 8
           and len(rabs["sealed_only_from_a_later_version"]) == 3
           and rabs["judged_against_the_full_list"] is False,
           f"design v{rabs['design_version_of_this_receipt']} resolved by "
           f"the carrying-commit pair; judged against "
           f"{rabs['n_judged_against']} names, "
           f"{len(rabs['sealed_only_from_a_later_version'])} sealed only "
           f"from a later version; {rabs['n_leaked_fields']} leaked over "
           f"{rabs['n_receipt_leaves_walked']} leaves")
    else:
        ck("R-663 ON THE REAL 09-03 RECEIPT -- ***SKIPPED, NAMED, AND NOT "
           "COUNTED AS A PASS***: the sealed receipt is not at this root. "
           "An absent artifact is a named status, never a green check",
           False,
           f"ABSENT: {real.name} -- rule 11: this cell FAILS rather than "
           f"reporting a seal it could not read")

    # the SAME bytes, two versions: only the pin differs
    vtmp = Path(tempfile.mkdtemp(prefix="da95ver_"))
    def _pinned(version: int, *, corrupt_digest: bool = False) -> dict:
        f = vtmp / f"p003_de_multiday_gate1_design_v{version}.json"
        f.write_text(json.dumps({"design_version": version}))
        d = hashlib.sha256(f.read_bytes()).hexdigest()
        return {"protocol": "P003_SYNTHETIC_FOR_THE_SCOPE_TEST",
                #: ABSOLUTE, so the pair is verified at the path the
                #: receipt NAMES (R-673(a)) and not at a basename
                "provenance": {"design": {
                    "path": str(f),
                    "sha256": ("0" * 64 if corrupt_digest else d)}},
                "per_day_sealed_artifacts": [
                    {"arm": "A", "admissibility": {"n_decisions": 7},
                     "counts": {"n_fills_arm": 3}}]}
    a21 = economic_absence(_pinned(21))
    a23 = economic_absence(_pinned(23))
    abad = economic_absence(_pinned(23, corrupt_digest=True))
    ck("KNOWN-BAD IN BOTH DIRECTIONS, ONE FIELD, TWO SCOPES: a receipt "
       "carrying `n_fills_arm` AT DEPTH is CLEAN under a v21 pin (the "
       "count was open by ruling then) and ***LEAKED under a v23 pin***, "
       "where R-659 sealed it. Same bytes, same walker, same name -- only "
       "the version the receipt pins differs, which is exactly what a "
       "scoped seal must turn on",
       a21["sealed"] is True and a21["n_judged_against"] == 8
       and a23["sealed"] is False and a23["n_judged_against"] == 11
       and any("n_fills_arm" in x for x in a23["leaked_field_paths"])
       and a21["design_version_of_this_receipt"] == 21
       and a23["design_version_of_this_receipt"] == 23,
       f"v21 -> sealed {a21['sealed']} against {a21['n_judged_against']} "
       f"names; v23 -> sealed {a23['sealed']} against "
       f"{a23['n_judged_against']}, leaked at {a23['leaked_field_paths']}")
    ck("AND THE PAIR IS A PAIR: a `provenance.design` whose PATH names v23 "
       "and whose DIGEST does not match the file is ***not a v23 receipt "
       "-- it is a receipt of UNKNOWN version***, judged against the FULL "
       "list and saying so. ***Absence never selects the weaker rule***: "
       "the strictest list is what an unresolvable version gets",
       abad["design_version_resolved"] is False
       and abad["judged_against_the_full_list"] is True
       and abad["n_judged_against"] == len(ECONOMIC_FIELDS)
       and abad["sealed"] is False
       and abad["version_evidence"]["evidence_considered"][0][
           "pair_verified"] is False,
       f"declared digest does not match the file -> resolved "
       f"{abad['design_version_resolved']}, judged against "
       f"{abad['n_judged_against']} (the full list), sealed "
       f"{abad['sealed']}")

    # the SCOPE MAP itself, read from DE's source, with both refusals
    stmp = Path(tempfile.mkdtemp(prefix="da95scope_"))
    base = DE_RUNNER_PATH.read_text()
    unused = stmp / "no_reader.py"
    unused.write_text(
        "ECONOMIC_FIELDS = ('D_E0',)\n"
        "SEALED_FROM_DESIGN_VERSION = {'D_E0': 1}\n"
        "def _strip_economic(x):\n    return ECONOMIC_FIELDS\n")
    short = stmp / "short_map.py"
    short.write_text(base.replace(
        '"n_fills_arm": 23, "n_fills_baseline": 23, "n_cancels_issued": 23,',
        '"n_fills_arm": 23,', 1))
    def _refuses(fn):
        try:
            fn()
            return None
        except VerifierRefused as e:
            return str(e)
    r_unused = _refuses(
        lambda: de_sealed_from_design_version_at_source(unused))
    r_short = _refuses(
        lambda: de_sealed_from_design_version_at_source(short))
    r_absent = _refuses(
        lambda: de_sealed_from_design_version_at_source(stmp / "gone.py"))
    ck("AND THE SCOPE MAP IS READ AT DE'S SOURCE BY AST WITH THE SAME TWO "
       "GUARDS THE FLAT LIST CARRIES (R-235: their declaration, this "
       "seat's judgement -- never their computation): a map ***no function "
       "consults*** pins nothing and REFUSES; a map that leaves a name in "
       "`ECONOMIC_FIELDS` ***with no sealed-from version*** REFUSES rather "
       "than defaulting it, because defaulting would pick a rule nobody "
       "wrote; and an absent runner REFUSES rather than guessing the scope",
       r_unused and "no function" in r_unused
       and r_short and "carry no sealed-from version" in r_short
       and r_absent and "MUST NOT be guessed" in r_absent,
       f"unconsulted map -> refused; two names dropped from the map -> "
       f"refused by name; absent runner -> refused")

    # -- G2. REV 49 section 2.4: A SEALED NUMBER HIDDEN IN PROSE ---------
    #: the exact class DE 85 found. The receipt is properly SEALED, so it
    #: carries no economic numeric leaf -- the value exists only in the TEXT
    #: of a refusal reason, and both leaf-typed checks are blind to it.
    hidden_val = float(payload[sorted(payload)[0]]["economic"]["null_sd"])
    r_hidden = json.loads(spath.read_text())
    r_hidden["per_day_sealed_artifacts"][0]["admissibility"]["reasons"] = [
        f"null sd {hidden_val} < 0.25 * |mean|; Z explodes as sd -> 0"]
    hp = td / "sealed_hidden_in_prose.json"
    hp.write_text(json.dumps(r_hidden))
    rh = json.loads(hp.read_text())
    #: the CENSUS is the unit under test: an emission that echoes the reason
    #: must be caught, and the real pre-read's emission must be clean.
    leaky_emission = {"arms": {"A": {"note": (
        f"refused: null sd {hidden_val} below the floor")}}}
    cen_bad = emitted_census(leaky_emission, rh)
    cen_ok = emitted_census({"arms": {"A": {"note": "refused on the floor"}}},
                            rh)
    ck("REV 49 section 2.4 CLOSED -- A SEALED NUMBER HIDDEN IN PROSE IS "
       "CAUGHT. The receipt is properly sealed and carries NO economic "
       "numeric leaf, so the value lives only in a refusal reason's TEXT: an "
       "emission repeating it is caught by the string scan, and one that "
       "does not is clean. ***The first version's watch list was built from "
       "numeric leaves alone, so on a SEALED receipt it was EMPTY -- the "
       "census could only catch a leak in a receipt that had already "
       "leaked***",
       cen_bad["clean"] is False
       and cen_bad["n_of_them_carrying_a_watched_number_AS_TEXT"] >= 1
       and cen_bad["n_watched_from_the_receipts_PROSE"] >= 1
       and cen_bad["n_watched_from_numeric_leaves"] == 0
       and cen_ok["clean"] is True,
       f"watch list: {cen_bad['n_watched_from_numeric_leaves']} from numeric "
       f"leaves (a sealed receipt has none) + "
       f"{cen_bad['n_watched_from_the_receipts_PROSE']} from the receipt's "
       f"prose. The echoing emission -> "
       f"{cen_bad['n_of_them_carrying_a_watched_number_AS_TEXT']} string hit(s); "
       f"the clean one -> {cen_ok['n_of_them_carrying_a_watched_number_AS_TEXT']}")
    ck("AND THE HIT IS REPORTED BY PATH AND HOW, NEVER BY VALUE -- a census "
       "that printed the number it caught would publish exactly what the "
       "seal withholds, in the field that exists to prevent it",
       all("NOTE" in h and "path" in h and "how" in h
           for h in cen_bad["string_hits_by_path_only"])
       and str(hidden_val) not in json.dumps(
           cen_bad["string_hits_by_path_only"]),
       f"hits: {[(h['path'], h['how']) for h in cen_bad['string_hits_by_path_only']][:2]}; "
       f"the value appears 0 times in them")
    #: and the REAL pre-read must not echo the reasons at all.
    ph = pre_read_day("2026-09-03", str(bpath), str(hp), params=params,
                      now=BAR_BEFORE)
    echoed_paths = [p_ for p_, _ in _walk_paths(ph)
                    if "reasons" in p_.lower()]
    ck("AND THE PRE-READ NEVER ECHOES `admissibility.reasons` AT ALL: with "
       "the reason carrying a sealed number, its own emission is still "
       "clean, because it copies the STATUS and the COUNTS and not the prose",
       ph["emitted_census"]["clean"] is True and echoed_paths == [],
       f"{len(echoed_paths)} `reasons` paths in the emission; census clean "
       f"{ph['emitted_census']['clean']} over "
       f"{ph['emitted_census']['n_string_fields_scanned']} string fields")

    # -- G3. REV 49 section 2.5: THE iff FIRES AND ADMITS -----------------
    v12 = _sealed_de_shape_receipt(td, payload, bsha, name="sealed_v12.json",
                                   shape="v12", design=design,
                                   params_block=pblock)
    p12 = pre_read_day("2026-09-03", str(bpath), str(v12), params=params,
                       now=BAR_BEFORE)
    a12 = p12["arms"][sorted(params["arms"])[0]]["sd_over_abs_mean_consistency"]
    a13 = pre["arms"][sorted(params["arms"])[0]][
        "sd_over_abs_mean_consistency"]
    ck("REV 49 section 2.5 CLOSED -- THE `iff` NOW FIRES AND ADMITS, because "
       "the fixture carries BOTH states: a v12-shaped receipt KEEPS "
       "`sd_over_abs_mean` (v12's stripper did not know it) while DE's "
       "current list says it should be gone, and a v13-shaped one strips it. "
       "***A check that could only ever see one state was pinning nothing***",
       a12["present_in_this_receipt"] is True
       and a12["consistent"] is False
       and a13["present_in_this_receipt"] is False
       and a13["consistent"] is True
       and a12["in_DEs_current_field_list"] is True,
       f"v12-shaped: present={a12['present_in_this_receipt']}, "
       f"consistent={a12['consistent']}; v13-shaped: "
       f"present={a13['present_in_this_receipt']}, "
       f"consistent={a13['consistent']}")
    ck("AND THE INCONSISTENT STATE IS READ AS A PROVENANCE SIGNAL, NOT A "
       "DEFECT IN THE DAY: a receipt carrying the ratio while the list seals "
       "it was produced by code older than the list -- which is the state "
       "the REAL 09-03 receipt is in",
       "produced by code older than the list" in a12["reading"]
       and "PROVENANCE signal" in a12["reading"],
       f"reading: '{a12['reading'][:96]}...'")

    # -- G4. REV 49 section 2.6: the design pin is BOUND TO THE RECEIPT ---
    dz = pre["provenance"]["design"]
    r_bad_design = json.loads(spath.read_text())
    if r_bad_design.get("design_declaration"):
        r_bad_design["design_declaration"]["sha256"] = "0" * 64
        bdp = td / "sealed_bad_design.json"
        bdp.write_text(json.dumps(r_bad_design))
        pbd = pre_read_day("2026-09-03", str(bpath), str(bdp), params=params,
                           now=BAR_BEFORE)
        bad_ok = (pbd["provenance"]["design"]["matches"] is False
                  and pbd["status"] == "FLAGGED")
    else:
        bad_ok = False
    ck("REV 49 section 2.6 CLOSED -- THE DESIGN PIN IS READ FROM THE RECEIPT "
       "AND VERIFIED AT THE FILE IT NAMES, not against a version hardcoded "
       "here. A receipt declaring a digest the named file does not have is "
       "FLAGGED",
       dz["found"] is True and dz["matches"] is True
       and dz["named_by"].startswith("the receipt") and bad_ok,
       f"the receipt names {dz['path']} and declares its digest; verified "
       f"{dz['sha256'][:16]}. A declared digest that the file does not have "
       f"-> FLAGGED ({bad_ok})")

    # -- H. a wrong book REFUSES ------------------------------------------
    wb = td / "wrong_book.json"
    w = json.loads(bpath.read_text())
    w["rows"] = w["rows"][:-1]
    wb.write_text(json.dumps(w))
    why = ""
    try:
        pre_read_day("2026-09-03", str(wb), str(spath), params=params,
                     now=BAR_BEFORE)
    except VerifierRefused as e:
        why = str(e)
    ck("A WRONG BOOK REFUSES THE PRE-READ TOO: a day whose book does not "
       "match the receipt's digest is not the day the receipt describes, "
       "and no population recomputed from it would mean anything",
       "book digest mismatch" in why and bsha[:16] in why,
       f"'{why[:96]}...'")

    # -- H1b. REV 51 section 2.6: THE PARAMS PIN IS BOUND TO THE RECEIPT --
    pn = pre["provenance"]["params_named_by_the_receipt"]
    r_nop = json.loads(spath.read_text())
    r_nop.pop("params_declaration", None)
    nop = td / "sealed_no_params.json"
    nop.write_text(json.dumps(r_nop))
    p_nop = pre_read_day("2026-09-03", str(bpath), str(nop), params=P,
                         now=BAR_BEFORE)
    r_unr = json.loads(spath.read_text())
    r_unr["params_declaration"] = {"path": "de_multiday_gate1_params_v999.json",
                                   "sha256": "0" * 64,
                                   "protocol": "P003_..._V999"}
    unr = td / "sealed_params_unresolved.json"
    unr.write_text(json.dumps(r_unr))
    p_unr = pre_read_day("2026-09-03", str(bpath), str(unr), params=P,
                         now=BAR_BEFORE)
    ck("REV 51 section 2.6 CLOSED ON THE PARAMS PIN TOO: the design pin was "
       "bound to the receipt and TWO LINES AWAY the params pin was a "
       "constant. Round 74 made it newest-present, which is still this "
       "verifier CHOOSING -- ***the newest is not what ran***. The RECEIPT "
       "names its params now and the file it names is verified",
       pn["named_by_the_receipt"] is True
       and pn["status"] == "PARAMS_VERIFIED" and pn["matches"] is True
       and pn["digest_was_declared"] is True,
       f"the receipt names {pn['path']} ({pn['protocol']}) and declares its "
       f"digest; verified {pn['sha256'][:16]}")
    ck("AND BOTH GAPS ARE INCOMPLETE, NEVER A FALLBACK: a receipt naming NO "
       "params is PROVENANCE_INCOMPLETE_NO_PARAMS_NAMED, and one naming a "
       "params file this worktree cannot resolve is "
       "PROVENANCE_INCOMPLETE_PARAMS_UNRESOLVED -- verifying against a "
       "DIFFERENT declaration would be a verdict about bars the receipt "
       "never ran under",
       p_nop["provenance"]["params_named_by_the_receipt"]["status"]
       == "PROVENANCE_INCOMPLETE_NO_PARAMS_NAMED"
       and p_nop["status"] == "PROVENANCE_INCOMPLETE"
       and p_unr["provenance"]["params_named_by_the_receipt"]["status"]
       == "PROVENANCE_INCOMPLETE_PARAMS_UNRESOLVED"
       and p_unr["status"] == "PROVENANCE_INCOMPLETE"
       and p_nop["n_arms_agreeing"] == len(P["arms"]),
       f"unnamed -> {p_nop['status']}; unresolvable -> {p_unr['status']}; "
       f"both with every arm still agreeing "
       f"({p_nop['n_arms_agreeing']}/{p_nop['n_arms_declared']}) -- a GAP, "
       f"not a defect")

    # -- H2. R-604: THE READ BAR IS A CONJUNCTION, READ BY NAME ----------
    #: a params file WITHOUT the field must REFUSE, never fall back.
    thin = td / "params_no_read_gate.json"
    thin.write_text(json.dumps({k: v for k, v in params.items()
                                if k not in (READ_GATE_FIELD, "_path",
                                             "_sha256")}
                               | {"_path": str(thin), "_sha256": "0" * 64}))
    thin_p = json.loads(thin.read_text())
    refused_thin = False
    try:
        read_gate_predicate(thin_p, BAR_AFTER)
    except VerifierRefused as e:
        refused_thin = "CONJUNCTION" in str(e)
    ck("R-604 -- A PARAMS FILE WITHOUT THE READ-GATE CONJUNCTION REFUSES, "
       "AND NEVER FALLS BACK TO THE CLOCK. `gate_is_open` used to read "
       "`read_not_before_utc` ALONE, so it would have opened on a bar "
       "WEAKER than the one DE declares",
       refused_thin,
       "params carrying no read_gate.the_bar_is_a_CONJUNCTION raises rather "
       "than holding the clock alone")

    have_field = isinstance(params.get(READ_GATE_FIELD), dict)
    if have_field:
        pred_pre = read_gate_predicate(params, BAR_BEFORE)
        pred_post = read_gate_predicate(params, BAR_AFTER)
        ck("R-604 -- THE CONJUNCTS COME FROM DE's OWN FIELD AND EACH IS "
           "EVALUATED AND NAMED. Before the clock bar the clock conjunct "
           "fails; after it, the conjuncts that fail are the ARTIFACT ones, "
           "each named -- so the gate says WHICH bar is not met",
           pred_pre["open"] is False and pred_post["open"] is False
           and pred_pre["n_conjuncts_declared"] >= 2
           #: ASSERT ON THE EVALUATOR, NOT THE PROSE. This check used to grep
           #: the conjunct TEXT for "clock" -- prose matching, the very
           #: thing the id binding replaced -- and it broke the moment DE
           #: landed params v8 with objects whose text does not carry the
           #: word. The check was behind the code it tests.
           and any(c.get("evaluator")
                   == "clock_at_or_after_read_not_before_utc"
                   and c.get("holds") is False for c in pred_pre["conjuncts"])
           and any(c.get("evaluator")
                   == "clock_at_or_after_read_not_before_utc"
                   and c.get("holds") is True
                   for c in pred_post["conjuncts"]),
           f"{pred_pre['n_conjuncts_declared']} conjuncts declared in "
           f"{pred_pre['source_field']} ({pred_pre['params_protocol']}); "
           f"before the bar {pred_pre['n_holding']} hold, after it "
           f"{pred_post['n_holding']}; failing by name "
           f"{pred_post['failing_conjuncts_by_name']}")
        ck("AND A CONJUNCT NO EVALUATOR MATCHES IS A STATUS, NEVER A PASS -- "
           "so params growing from two conjuncts to eight cannot widen what "
           "this verifier silently accepts",
           "CLOSES the gate" in pred_post["why_unevaluable_is_not_a_pass"]
           and (not pred_post["conjuncts_not_evaluable"]
                or pred_post["open"] is False),
           f"{len(pred_post['conjuncts_not_evaluable'])} conjunct(s) "
           f"unevaluable; the gate is open={pred_post['open']}")
        grown = json.loads(json.dumps({k: v for k, v in params.items()
                                       if not k.startswith("_")}))
        grown[READ_GATE_FIELD][CONJUNCTION_FIELD] = list(
            grown[READ_GATE_FIELD][CONJUNCTION_FIELD]) + [
            "some future bar nobody has written an evaluator for"]
        pred_grown = read_gate_predicate(grown, BAR_AFTER)
        ck("KNOWN-BAD FOR THE SAME AXIS: a conjunct this verifier cannot "
           "evaluate is REPORTED and CLOSES the gate -- driven by adding one "
           "to the declared list",
           len(pred_grown["conjuncts_not_evaluable"]) == 1
           and pred_grown["open"] is False
           and any(c.get("status") == "CONJUNCT_NOT_EVALUABLE_BY_THIS_VERIFIER"
                   for c in pred_grown["conjuncts"]),
           f"an unrecognised conjunct -> "
           f"{len(pred_grown['conjuncts_not_evaluable'])} unevaluable, gate "
           f"open={pred_grown['open']}")

    # -- H2b. R-604/REV 51: PARAMS v8's EIGHT CONJUNCTS, BOUND BY ID -----
    V8_IDS = ["params_field_required", "clock_ge_read_not_before",
              "six_ruled_days_from_params", "receipt_at_landing_digest",
              "at_least_one_admissible_arm", "ledger_verdict",
              "producing_code_locatable", "horizon_fallback_G5_directional"]
    d8 = td / "v8"
    d8.mkdir(exist_ok=True)
    p8 = json.loads(json.dumps({k: v for k, v in params.items()
                                if not k.startswith("_")}))
    p8["protocol"] = "P003_DE_MULTIDAY_GATE1_PARAMS_V8_FIXTURE"
    p8[READ_GATE_FIELD] = {
        CONJUNCTION_FIELD: [{"id": i, "text": f"conjunct {i}"}
                            for i in V8_IDS],
        "horizon_utc": "2026-09-09T12:00:00Z",
        "horizon_fallback": {"G": 5, "reading": "directional"},
    }
    p8["_path"], p8["_sha256"] = str(td / "p8.json"), "0" * 64
    pred8 = read_gate_predicate(p8, BAR_BEFORE, derived=d8)
    bound = [c for c in pred8["conjuncts"] if c.get("binding") == "bound by ID"]
    ck("R-604 / REV 51 -- ALL EIGHT OF PARAMS v8's CONJUNCTS ARE BOUND BY "
       "STABLE ID, not by matching prose. `matched some words` and `is this "
       "conjunct` are different claims, and the binding route is RECORDED "
       "per conjunct so a reader knows which was made",
       len(pred8["conjuncts"]) == 8 and len(bound) == 8
       and not pred8["conjuncts_not_evaluable"]
       and sorted(c["id"] for c in bound) == sorted(V8_IDS),
       f"{len(bound)} of {len(pred8['conjuncts'])} bound by ID, 0 "
       f"unevaluable; ids {sorted(c['id'] for c in bound)[:3]}...")
    ck("AND EVERY CONJUNCT IS EVALUATED, WITH THE FAILING ONES NAMED: on an "
       "empty ledger the artifact conjuncts fail by name while "
       "`params_field_required` holds -- reaching the evaluator IS that "
       "conjunct holding, because a missing field REFUSES before any "
       "conjunct runs",
       pred8["open"] is False
       and set(pred8["failing_conjuncts_by_name"]) >= {
           "every_ruled_day_has_exactly_one_sealed_receipt",
           "every_sealed_receipt_matches_its_LANDING_RECORD",
           "every_day_has_at_least_one_admissible_arm",
           "the_ledger_verdict_stands_for_every_day",
           "every_receipt_names_locatable_producing_code"}
       and any(c["evaluator"] == "the_params_field_is_present"
               and c["holds"] is True for c in pred8["conjuncts"]),
       f"{pred8['n_holding']} of 8 hold on an empty ledger; failing by name "
       f"{sorted(pred8['failing_conjuncts_by_name'])[:3]}...")
    unknown = json.loads(json.dumps(p8))
    unknown[READ_GATE_FIELD][CONJUNCTION_FIELD].append(
        {"id": "some_id_this_verifier_has_never_heard_of", "text": "x"})
    unknown["_path"], unknown["_sha256"] = p8["_path"], p8["_sha256"]
    pu = read_gate_predicate(unknown, BAR_AFTER, derived=d8)
    ck("KNOWN-BAD: AN ID THIS VERIFIER IS NOT BOUND TO CLOSES THE GATE, and "
       "the route says so -- ID NOT IN THIS VERIFIER'S MAP. DE renaming or "
       "adding a conjunct cannot widen what is silently accepted",
       len(pu["conjuncts_not_evaluable"]) == 1
       and pu["open"] is False
       and any(c.get("binding") == "ID NOT IN THIS VERIFIER'S MAP"
               for c in pu["conjuncts"]),
       f"an unknown id -> {len(pu['conjuncts_not_evaluable'])} unevaluable, "
       f"gate open={pu['open']}")

    # -- H2c. THE HORIZON NAMES G = 5 AND NEVER APPLIES IT SILENTLY ------
    #: BAR_AFTER (09-10) is already PAST the declared horizon, so "inside
    #: the horizon" needs a clock between the read bar (09-09T00:06Z) and
    #: the horizon (09-09T12:00Z). The first version of this check used
    #: BAR_AFTER for both sides and failed -- correctly.
    inside_h = datetime.datetime(2026, 9, 9, 6, 0,
                                 tzinfo=datetime.timezone.utc)
    past_h = datetime.datetime(2026, 9, 10, tzinfo=datetime.timezone.utc)
    pre_h = read_gate_predicate(p8, inside_h, derived=d8)
    post_h = read_gate_predicate(p8, past_h, derived=d8)
    hz_a = next(c for c in pre_h["conjuncts"]
                if c["evaluator"] == "clock_at_or_before_horizon")
    hz_b = next(c for c in post_h["conjuncts"]
                if c["evaluator"] == "clock_at_or_before_horizon")
    ck("REV 51 (b) -- THE HORIZON IS EVALUATED AGAINST THE CLOCK AND ITS "
       "FALLBACK IS NAMED, NEVER APPLIED SILENTLY: inside the horizon it "
       "HOLDS; past it, it fails and the receipt states the declared "
       "outcome -- G = 5, DIRECTIONAL. ***That is a different claim from the "
       "six-day test, and taking it is the coordinator's act***",
       hz_a["holds"] is True and hz_a["horizon_passed"] is False
       and hz_b["holds"] is False and hz_b["horizon_passed"] is True
       and hz_b["THE_FALLBACK_OUTCOME_IF_IT_PASSES"]["G"] == 5
       and hz_b["THE_FALLBACK_OUTCOME_IF_IT_PASSES"]["applies_now"] is True
       and hz_a["THE_FALLBACK_OUTCOME_IF_IT_PASSES"]["applies_now"] is False,
       f"horizon {hz_a['horizon']}: inside -> holds; past it -> fails with "
       f"the fallback named G="
       f"{hz_b['THE_FALLBACK_OUTCOME_IF_IT_PASSES']['G']} "
       f"{hz_b['THE_FALLBACK_OUTCOME_IF_IT_PASSES']['reading']}")

    # -- H2d. REV 51 (a): THE CHAIN HEAD, AND AN UNCHAINED PAIR REFUSES ---
    def _mk(day, name, sup=None, admissible=True):
        b = {"day": day, "per_day_sealed_artifacts": [
            {"arm": "A", "day": day, "status": "OK",
             "admissibility": {"admissible": admissible, "n_decisions": 50},
             "draw_provenance": {"seed": 1}, "runner_sha256": "a" * 64,
             "carrying_commit": "HEAD"}]}
        if sup is not None:
            #: R-608: the link is the PAIR, so the fixture writes the PAIR
            b["supersedes"] = {
                "path": Path(sup).name,
                "sha256": hashlib.sha256(Path(sup).read_bytes()).hexdigest()}
        (d8 / name).write_text(json.dumps(b))
        return d8 / name
    D = "20260903"
    one = _mk(D, f"p003_de_gate1_day_run_{D}_SEALED__A.json")
    pred_one = read_gate_predicate(p8, BAR_AFTER, derived=d8)
    st_one = next(c for c in pred_one["conjuncts"]
                  if c["evaluator"]
                  == "every_ruled_day_has_exactly_one_sealed_receipt"
                  )["per_day"][D]
    two = _mk(D, f"p003_de_gate1_day_run_{D}_SEALED__B.json", sup=one)
    pred_ch = read_gate_predicate(p8, BAR_AFTER, derived=d8)
    st_ch = next(c for c in pred_ch["conjuncts"]
                 if c["evaluator"]
                 == "every_ruled_day_has_exactly_one_sealed_receipt"
                 )["per_day"][D]
    _mk(D, f"p003_de_gate1_day_run_{D}_SEALED__C.json")
    pred_am = read_gate_predicate(p8, BAR_AFTER, derived=d8)
    st_am = next(c for c in pred_am["conjuncts"]
                 if c["evaluator"]
                 == "every_ruled_day_has_exactly_one_sealed_receipt"
                 )["per_day"][D]
    ck("REV 51 (a) -- THE SUPERSEDES CHAIN IS FOLLOWED AND AN UNCHAINED PAIR "
       "REFUSES: one receipt resolves; a v1 plus a CHAINED v2 resolves to "
       "the HEAD; a third with NO link makes the day AMBIGUOUS. ***Picking "
       "the newest by mtime would resolve an ambiguous pair silently, and a "
       "day that ran twice is not a day that ran***",
       st_one["status"] == "ONE" and st_one["resolves"] is True
       and st_ch["status"] == "CHAIN_HEAD" and st_ch["resolves"] is True
       and st_ch["head"] == two.name
       and st_am["status"] == "AMBIGUOUS" and st_am["resolves"] is False,
       f"one -> {st_one['status']}; chained pair -> {st_ch['status']} at "
       f"{st_ch['head']}; unchained third -> {st_am['status']}")
    # -- H2e. R-608 / REV 52 section 2.3: THE LINK IS THE PAIR -----------
    #: THE DEFINITION IS READ FROM DE's DESIGN, not held here. DE declares
    #: its OWN supersession chain as [path, sha256] entries; that form IS
    #: the definition, and reading it from the artifact is why this verifier
    #: cannot quietly hold a different rule than the seat writing the links.
    pdef = supersession_pair_definition()
    ck("R-608 -- THE PAIR'S DEFINITION IS READ FROM DE's DESIGN, NOT TYPED "
       "HERE: the design carries its own supersession chain as "
       "[path, sha256] entries, and this verifier takes its required fields "
       "from that artifact. ***Two seats resolving one link by different "
       "fields is how the same three files become a clean chain to one and "
       "a refusal to the other***",
       pdef["declared"] is True
       and pdef["source"].startswith("p003_de_multiday_gate1_design_v")
       and tuple(pdef["required_fields"]) == SUPERSESSION_PAIR_FIELDS
       and pdef["observed_in"] == "supersedes.chain",
       f"{pdef['source']} declares {pdef['n_chain_entries']} chain entries "
       f"as pairs -> required fields {tuple(pdef['required_fields'])}")

    #: the reviewer's THREE ROWS, driven to the ruled outcomes, in a dir of
    #: their own so the H2d state cannot supply the answer.
    d608 = td / "r608"
    d608.mkdir(exist_ok=True)
    p608 = p8
    base = {"day": "2026-09-03", "per_day_sealed_artifacts": [
        {"arm": "A", "day": "2026-09-03", "status": "OK",
         "admissibility": {"admissible": True, "n_decisions": 50},
         "draw_provenance": {"seed": 1}, "runner_sha256": "a" * 64}]}
    v1p = d608 / f"p003_de_gate1_day_run_{D}_SEALED__V1.json"
    v1p.write_text(json.dumps(base))
    v1sha = hashlib.sha256(v1p.read_bytes()).hexdigest()

    def _row(sup):
        v2 = d608 / f"p003_de_gate1_day_run_{D}_SEALED__V2.json"
        v2.write_text(json.dumps(dict(base, supersedes=sup)))
        return next(
            c for c in read_gate_predicate(p608, BAR_AFTER, derived=d608)[
                "conjuncts"] if c["evaluator"]
            == "every_ruled_day_has_exactly_one_sealed_receipt"
            )["per_day"][D]

    row_sha = _row({"sha256": v1sha})
    row_path = _row({"path": v1p.name})
    row_both = _row({"path": v1p.name, "sha256": v1sha})
    row_moved = _row({"path": "p003_de_gate1_day_run_20260903_SEALED__X.json",
                      "sha256": v1sha})
    row_wrong = _row({"path": v1p.name, "sha256": "0" * 64})
    ck("R-608 -- THE THREE ROWS: sha256 ALONE is NOT a link and refuses BY "
       "NAME (SUPERSESSION_LINK_INCOMPLETE); path ALONE is NOT a link and "
       "refuses BY NAME; BOTH, landing on ONE present file, resolves to the "
       "HEAD. ***`no link` was the old answer for the first two rows, and it "
       "makes a half-written supersession look like a day that ran twice***",
       row_sha["status"] == "SUPERSESSION_LINK_INCOMPLETE"
       and row_sha["resolves"] is False
       and row_path["status"] == "SUPERSESSION_LINK_INCOMPLETE"
       and row_path["resolves"] is False
       and row_both["status"] == "CHAIN_HEAD" and row_both["resolves"] is True
       and row_both["head"].endswith("__V2.json"),
       f"sha256-only -> {row_sha['status']}; path-only -> "
       f"{row_path['status']}; both -> {row_both['status']} at "
       f"{row_both['head']}")
    ck("AND A MATCHING DIGEST UNDER A DIFFERENT NAME IS A **MOVED FILE**, "
       "REFUSED -- resolving by digest alone would accept it; the pair does "
       "not, because a moved file is not the file the link names. A NAMED "
       "file whose BYTES differ refuses too",
       row_moved["status"] == "SUPERSESSION_TARGET_MOVED"
       and row_moved["resolves"] is False
       and row_wrong["status"] == "SUPERSESSION_TARGET_DIGEST_MISMATCH"
       and row_wrong["resolves"] is False,
       f"right bytes under another name -> {row_moved['status']}; named "
       f"file at the wrong digest -> {row_wrong['status']}")

    # -- H2f. REV 52 section 2.4: THE LANDING RECORD'S NAME AND CHAIN -----
    nm = pre_read_artifact_naming()
    lrd = td / "lr608"
    lrd.mkdir(exist_ok=True)
    ck("REV 52 section 2.4 -- THE LANDING RECORD HAS A DECLARED NAME AND A "
       "CORRECTION PATH, declared by the seat that writes it, with the DAY "
       "read from the `day` FIELD and never parsed out of the filename. "
       "***It had neither, so a mistaken pre-read could not be superseded, "
       "only shadowed***",
       nm["template"] == "p003_da_gate1_pre_read_<YYYYMMDD>__<clock>.json"
       and nm["day_field_required"] is True
       and "supersedes" in nm["correction_path"]
       and pre["landing_record_naming"]["template"] == nm["template"],
       f"{nm['template']}; day from {nm['day_comes_from']}")

    def _lr(name, sha, sup=None, mirror=None):
        rec = dict(pre)
        rec["day"] = "2026-09-03"
        #: ONE digest in BOTH homes, as the emitter now writes it. `mirror`
        #: is here ONLY so the disagreement can be DRIVEN.
        #: BY ROLE, THROUGH THE DECLARED PATHS. Writing `receipt.sha256`
        #: and `landing_record.receipt_sha256` by name pinned the fixture
        #: to round 78's choice of authority; when DE moved, the fixture
        #: was testing the roles backwards.
        def _put(path, value):
            blk, key = path.split(".", 1)
            base = dict(pre.get(blk) or {})
            base[key] = value
            rec[blk] = base

        _put(LANDING_DIGEST_AUTHORITATIVE_FIELD, sha)
        _put(LANDING_DIGEST_MIRROR_FIELD,
             sha if mirror is None else mirror)
        if sup is not None:
            rec["supersedes"] = {
                "path": Path(sup).name,
                "sha256": hashlib.sha256(Path(sup).read_bytes()).hexdigest()}
        f = lrd / name
        f.write_text(json.dumps(rec, default=str))
        return f

    lr_none = landing_record_for("2026-09-03", lrd)
    a = _lr("p003_da_gate1_pre_read_20260903__20260906T120000Z.json", "1" * 64)
    lr_one = landing_record_for("2026-09-03", lrd)
    b = _lr("p003_da_gate1_pre_read_20260903__20260906T130000Z.v2.json",
            "2" * 64, sup=a)
    lr_chain = landing_record_for("2026-09-03", lrd)
    _lr("p003_da_gate1_pre_read_20260903__20260906T140000Z.json", "3" * 64)
    lr_amb = landing_record_for("2026-09-03", lrd)
    ck("AND THE RECORD'S OWN CHAIN RESOLVES BY THE SAME PAIR RULE: none -> "
       "NO_LANDING_RECORD; one -> that record; a .v2 CHAINED BY THE PAIR -> "
       "the v2's digest; two UNCHAINED -> AMBIGUOUS. ***The record the read "
       "gate compares against is itself an artifact, so it needs the "
       "correction path the receipts have***",
       lr_none["status"] == "NO_LANDING_RECORD"
       and lr_none["receipt_sha256"] is None
       and lr_one["status"] == "ONE" and lr_one["receipt_sha256"] == "1" * 64
       and lr_chain["status"] == "CHAIN_HEAD"
       and lr_chain["head"] == b.name
       and lr_chain["receipt_sha256"] == "2" * 64
       and lr_amb["status"] == "AMBIGUOUS"
       and lr_amb["receipt_sha256"] is None,
       f"none -> {lr_none['status']}; one -> {lr_one['status']} at "
       f"{lr_one['receipt_sha256'][:8]}; chained -> {lr_chain['status']} at "
       f"{lr_chain['head']} ({lr_chain['receipt_sha256'][:8]}); unchained "
       f"pair -> {lr_amb['status']}")

    # -- R-654: THE --open-book PATH DA 91 WILL RUN, driven both ways -----
    import pickle as _pk                                      # noqa: PLC0415
    _bd = td / "openbook"
    _bd.mkdir(exist_ok=True)
    _js = _bd / "fixture_book.json"
    _js.write_text(json.dumps({"rows": [{"side": "BUY", "t": 1}],
                               "scores_by_arm": {"A": [0.5]}}))
    _json_ok = load_day_book(str(_js))
    _wrong = _bd / "wrong_shape.pkl"
    with _wrong.open("wb") as fh:
        _pk.dump({"not_asm": 1, "other": 2}, fh)
    _lst = _bd / "not_a_map.pkl"
    with _lst.open("wb") as fh:
        _pk.dump([1, 2, 3], fh)
    _shaped = _bd / "declared_shape.pkl"
    with _shaped.open("wb") as fh:
        _pk.dump({"asm": {"by_arm": {}}, "fr": {}}, fh)
    _msgs = {}
    #: THE PIN COMES FIRST NOW (REV 71 2.3), so each drive supplies the
    #: file's OWN digest -- otherwise every one of them refuses at the pin
    #: and the shape checks below are never reached. That is the ordering
    #: working, and this cell has to respect it to test what it names.
    for _lbl, _f, _open in (("wrong_top_level", _wrong, True),
                            ("not_a_mapping", _lst, True),
                            ("declared_shape", _shaped, True),
                            ("light_path_on_a_pickle", _shaped, False)):
        _sha = hashlib.sha256(_f.read_bytes()).hexdigest()
        try:
            load_day_book(str(_f), open_book=_open, expected_sha256=_sha)
            _msgs[_lbl] = "ADMITTED"
        except VerifierRefused as _e:
            _msgs[_lbl] = str(_e)
    _pin_msgs = {}
    for _lbl, _kw in (("wrong_pin", {"expected_sha256": "a" * 64}),
                      ("no_pin_at_all", {})):
        try:
            load_day_book(str(_shaped), open_book=True, **_kw)
            _pin_msgs[_lbl] = "ADMITTED"
        except VerifierRefused as _e:
            _pin_msgs[_lbl] = str(_e).split(" -- ")[0].replace(
                "REFUSED: ", "")
    ck("R-654 -- THE `--open-book` PATH IS BUILT AND DRIVEN BOTH WAYS, so "
       "DA 91 has something to run on the lock. A JSON fixture book ADMITS; "
       "a pickle whose top level is NOT the declared shape is REFUSED BY "
       "NAME; one that is not a mapping at all is refused by name; and one "
       "WITH the declared top level is still refused, because ***the "
       "mapping from `asm.by_arm` and `fr` onto rows and per-arm scores is "
       "BE's to DECLARE (R-654) and this verifier will not infer it from a "
       "pickle's shape***. The light path refuses any pickle before opening "
       "it, because opening one is HEAVY under rule 20",
       len(_json_ok["rows"]) == 1
       and "TOP_LEVEL_NOT_THE_DECLARED_SHAPE" in _msgs["wrong_top_level"]
       and "NOT_A_MAPPING" in _msgs["not_a_mapping"]
       and "AWAITS_BES_DECLARATION" in _msgs["declared_shape"]
       and "NOT_THIS_READER'S_JSON" in _msgs["light_path_on_a_pickle"]
       #: REV 71 2.3: and the PIN is checked BEFORE any of it.
       and _pin_msgs["wrong_pin"] == "BOOK_DIGEST_DOES_NOT_MATCH_ITS_RECEIPT"
       and _pin_msgs["no_pin_at_all"] == "NO_PIN_NO_OPEN",
       "; ".join(f"{k} -> {v.split(' -- ')[0].replace('REFUSED: ', '')}"
                 for k, v in list(_msgs.items()) + list(_pin_msgs.items())))
    # -- DA 97: THE RECEIPT NAMES ITS PINS WHERE DE WRITES THEM ---------
    #: THE NEWEST PRESENT, never a version typed here: this fixture
    #: pinned `_v14` and DE landed `_v15` within the hour, so my own
    #: census refused my own selftest. A fixture that names a moving
    #: thing by number goes stale exactly as fast as the thing moves.
    _pp = {"path": f"live/pm_research/declarations/{PARAMS_PATH.name}",
           "sha256": hashlib.sha256(PARAMS_PATH.read_bytes()).hexdigest()} \
        if PARAMS_PATH.is_file() else None
    if _pp:
        _new_shape = params_check({"provenance": {"params": _pp}})
        _old_shape = params_check({"params_declaration": _pp})
        _conflict = params_check({
            "provenance": {"params": _pp},
            "params_declaration": {**_pp, "sha256": "b" * 64}})
        _neither = params_check({"day": "2026-09-04"})
        ck("DA 97, FOUND BY RUNNING THE PRE-READ ON THE 09-04 RECEIPT -- "
           "***THE RECEIPT NAMED ITS PINS AND THIS VERIFIER SAID \"the "
           "receipt names none\".*** DE writes both pairs in a top-level "
           "`provenance` block; these readers looked only at the older "
           "`params_declaration` / `declaration.design` shapes, so a "
           "receipt carrying v14 and v21 BY PAIR came back "
           "PROVENANCE_INCOMPLETE. ***A gap reported where the artifact is "
           "complete is the same defect as a pass reported where it is "
           "not*** -- both are the verifier describing itself instead of "
           "the receipt. Driven four ways: the NEW shape resolves and says "
           "WHERE it read the pin, the OLD shape still resolves, ***two of "
           "the receipt's own statements that DISAGREE are a named "
           "CONFLICT*** rather than a silent choice between them, and a "
           "receipt naming neither is the unchanged named gap",
           _new_shape["matches"] is True
           and _new_shape["named_at"] == "provenance.params"
           and _old_shape["matches"] is True
           and _old_shape["named_at"] == "params_declaration"
           and _conflict["status"]
           == "PROVENANCE_CONFLICT_THE_RECEIPT_NAMES_TWO"
           and _conflict["matches"] is None
           and _neither["status"] == "PROVENANCE_INCOMPLETE_NO_PARAMS_NAMED",
           f"provenance.params -> {_new_shape['matches']} at "
           f"{_new_shape['named_at']}; params_declaration -> "
           f"{_old_shape['matches']}; both, disagreeing -> "
           f"{_conflict['status']} over {len(_conflict['places'])} places; "
           f"neither -> {_neither['status']}")
        _dpin = {"path": "data/pm_5min/derived/"
                         "p003_de_multiday_gate1_design_v21.json",
                 "sha256": "c" * 64}
        _d_receipt = design_check({"provenance": {"design": _dpin}},
                                  {"design_declaration": {
                                      "path": _dpin["path"],
                                      "sha256": "d" * 64}})
        ck("AND THE RECEIPT'S OWN STATEMENT OUTRANKS THE PARAMS "
           "DECLARATION'S: where both name a design pin, the verdict is "
           "about what THE RECEIPT says it read -- the params file "
           "describes what a run SHOULD read, the receipt records what it "
           "DID. The source is named in the record either way",
           _d_receipt["named_by"] == "the receipt (provenance.design)"
           and _d_receipt["sha256_declared"] == "c" * 64,
           f"named_by {_d_receipt['named_by']}; the digest judged against "
           f"is the RECEIPT's, not the params file's")
    else:
        ck("DA 97 PIN-SHAPE CELL -- ***SKIPPED AND NAMED, NOT PASSED***: "
           "no params declaration is present in this tree",
           False, "the fixture needs the params declaration on disk")

    # -- REV 76 S0: THE CENSUS WALKS KEYS, NOT LEAVES -------------------
    _forms = {"an empty list": [], "an empty mapping": {},
              "a zero": 0.0, "a null": None}
    _rows, _missed = [], 0
    for _name in ECONOMIC_FIELDS:
        for _lbl, _val in _forms.items():
            _r = economic_absence({"protocol": "SYNTHETIC_NO_PROVENANCE",
                                   "per_day_sealed_artifacts": [
                                       {"arm": "A", "nested": {
                                           "deeper": {_name: _val}}}]})
            _rows.append((_name, _lbl, _r["sealed"], _r["n_leaked_fields"],
                          _r["leaked_field_paths"][:1]))
            if _r["sealed"] or _r["n_leaked_fields"] != 1:
                _missed += 1
    _leafblind = sum(1 for n, l, *_ in _rows if l != "a zero")
    _e903 = _e904 = None
    for _f, _lbl in (("p003_de_gate1_day_run_20260903_SEALED__"
                      "20260906T140155Z.json", "09-03"),
                     ("p003_de_gate1_day_run_20260904_SEALED__"
                      "20260906T163351Z.json", "09-04")):
        _q = _derived_dir() / _f
        if _q.is_file():
            _x = economic_absence(json.loads(_q.read_text()))
            if _lbl == "09-03":
                _e903 = _x
            else:
                _e904 = _x
    ck("REV 76 S0 -- ***A LEAF WALK CANNOT SEE AN EMPTY CONTAINER.*** "
       "`D_E0: 0.0` leaked and `D_E0: []` came back SEALED: the key was "
       "PRESENT and the walk yielded nothing to judge, so a sealed name "
       "emitted as `[]`, `{}` or `null` was ***present and ignored*** -- "
       "the one state the receipt's own `seal_status` says cannot happen. "
       "Nothing produces it today (DE's `_strip_economic` removes KEYS); "
       "***the census is what must make it impossible tomorrow***. It "
       "walks KEYS now: a sealed name present as a key refuses WHATEVER "
       "its value. Driven on ALL ELEVEN names in ALL FOUR forms at depth "
       "-- 44 drives, 44 refusals, and 33 of them are ones the old leaf "
       "walk could not have seen",
       len(_rows) == 4 * len(ECONOMIC_FIELDS) and _missed == 0
       and _leafblind == 33,
       f"{len(_rows)} drives across {len(ECONOMIC_FIELDS)} names x "
       f"{len(_forms)} forms -> {len(_rows) - _missed} refused, "
       f"{_missed} missed; {_leafblind} of them empty-or-null, which a "
       f"leaf walk never reaches")
    ck("AND THE TWO REAL SEALED RECEIPTS ARE STILL SEALED UNDER THE KEY "
       "WALK -- ***the rule got stricter and the artifacts did not "
       "move***: DE strips the KEYS, so there is nothing for a key walk "
       "to find that a leaf walk missed. `seal_holds` would read the SAME "
       "for both days, which is why neither landing record needs "
       "re-emission",
       _e903 is not None and _e904 is not None
       and _e903["sealed"] is True and _e903["n_leaked_fields"] == 0
       and _e904["sealed"] is True and _e904["n_leaked_fields"] == 0
       and _e903["n_leaked_that_a_leaf_walk_could_not_see"] == 0
       and _e904["n_leaked_that_a_leaf_walk_could_not_see"] == 0,
       (f"09-03: {_e903['n_receipt_keys_walked']} keys, 0 leaked; 09-04: "
        f"{_e904['n_receipt_keys_walked']} keys, 0 leaked"
        if _e903 and _e904 else
        "ABSENT: one of the two sealed receipts is not at this root -- "
        "reported, never passed"))
    _echo = emitted_census({"mine": {"D_E0": []}}, {"day": "2026-09-04"})
    ck("AND MY OWN ANTI-ECHO CENSUS WALKS KEYS TOO: `{\"D_E0\": []}` in "
       "something I emit carries the NAME of a sealed quantity and no "
       "leaf at all -- check (a) is about the NAME, so it must see the "
       "key whatever the value is",
       _echo["n_economic_field_names_in_the_emission"] >= 1,
       f"a sealed NAME with an empty value in an emission -> "
       f"{_echo['n_economic_field_names_in_the_emission']} name(s) found")
    _de = de_seal_rule_at_source()
    ck("AND THE TWO CENSUSES SHARE A RULE, NOT AN IMPLEMENTATION (R-235). "
       "The rule is stated here as `SEAL_RULE` and READ FROM DE'S SOURCE "
       "by AST, exactly as the field list and the per-name scope map are. "
       "***A rule DE has not declared yet is a NAMED STATUS, never a "
       "pass***, and one declared DIFFERENTLY is a flag -- ***two "
       "censuses agreeing by accident is what this check exists to "
       "prevent***. At this tip DE 105 has not landed its side, so the "
       "seam is reported as NOT YET DRIVEABLE with DE's source digest "
       "beside it",
       _de["status"] in ("DECLARED_AND_MATCHES",
                         "DE_HAS_NOT_DECLARED_THE_KEY_WALK_RULE_YET")
       and (_de["agrees"] is True if _de["status"].startswith("DECLARED")
            else _de["agrees"] is None),
       f"{_de['status']}; DE's runner at {_de['source_sha256'][:16]}; key "
       f"walkers found there: {_de['key_walkers_found']}")

    # -- R-673(a): THE PATH THE RECEIPT NAMES IS THE PATH THAT IS HASHED -
    _pt = Path(tempfile.mkdtemp(prefix="da97path_"))
    _real = _derived_dir() / "p003_de_multiday_gate1_design_v21.json"
    _twin = _pt / _real.name
    _twin.write_bytes(_real.read_bytes())          # identical bytes, elsewhere
    _twin_sha = hashlib.sha256(_twin.read_bytes()).hexdigest()
    _here = resolve_named_path(str(_twin))
    _gone = _pt / "never_written" / _real.name
    _miss = resolve_named_path(str(_gone))
    _one_byte = _pt / "changed" / _real.name
    _one_byte.parent.mkdir()
    _b = bytearray(_real.read_bytes())
    _b[10] = (_b[10] + 1) % 256
    _one_byte.write_bytes(bytes(_b))
    _chg = resolve_named_path(str(_one_byte))
    _v_missing = receipt_design_version(
        {"provenance": {"design": {"path": str(_gone),
                                   "sha256": _twin_sha}}})
    _v_changed = receipt_design_version(
        {"provenance": {"design": {"path": str(_one_byte),
                                   "sha256": _twin_sha}}})
    ck("R-673(a) -- ***A PAIR VERIFIED AGAINST A FILE NOBODY OPENED IS NOT "
       "A PAIR.*** Route 1 threw away everything but the BASENAME and "
       "hashed `<derived>/<basename>`, so a receipt naming a /tmp copy "
       "came back `pair_verified true` on the strength of a ledger file it "
       "never opened. Driven three ways on a REAL design declaration: a "
       "same-bytes twin at another absolute path is hashed AT THAT PATH "
       "and the record NAMES it; ***a path that is not there REFUSES BY "
       "NAME even though a same-basename file with identical bytes sits in "
       "the ledger*** -- the version comes back UNRESOLVED and the full "
       "list applies; and one byte changed at the named path is a digest "
       "MISMATCH, not a fallback. ***The basename is never a second "
       "chance.***",
       _here["resolved"] is True and _here["path_hashed"] == str(_twin)
       and _here["sha256"] == _twin_sha
       and _miss["resolved"] is False and _miss["path_hashed"] is None
       and _real.is_file()
       and _v_missing["resolved"] is False
       and _chg["resolved"] is True and _chg["sha256"] != _twin_sha
       and _v_changed["resolved"] is False,
       f"twin at {_twin.parent.name}/ -> hashed at the path named; absent "
       f"named path -> resolved {_miss['resolved']} while the ledger twin "
       f"exists; one byte changed -> digest differs and the version is "
       f"{_v_changed['design_version']}")

    # -- R-673(b): EVERY BINDING SHAPE, OR A REFUSAL --------------------
    _src = ("W: int = 1\n"                       # AnnAssign with a value
            "W, other = 2, 3\n"                  # tuple target
            "d = {}\n"
            "d['k'] = (W := 4)\n"                # walrus
            "W['k'] = 5\n"                       # subscript ON the name
            "W += 1\n"                            # augmented alone
            "TABLE = {'W': 23}\n"                # a table entry
            "OUT = {'W': compute()}\n"           # an emission
            "out['W'] = compute()\n"             # emitted by key
            "tbl['W'] = 23\n")                   # a table entry, by key
    _bs = binding_sites(ast.parse(_src.replace("W", "D_E_MINUS_R")),
                        "D_E_MINUS_R")
    _un = binding_sites(ast.parse("for D_E_MINUS_R in rows:\n    pass\n"),
                        "D_E_MINUS_R")
    _shapes = sorted({b["shape"] for b in _bs["binds"]})
    ck("R-673(b) -- ***A SEARCH BOUNDED TO TWO SHAPES LEAVES THE CLAIM "
       "READING TRUE.*** The producing-place scan understood `NAME = ...` "
       "and a dict key, so an ANNOTATED assignment, a TUPLE target, a "
       "WALRUS, a SUBSCRIPT target and an AUGMENTED assignment each made "
       "*the runner produces D(E-R) in ZERO places* true by not being "
       "looked at -- and all five are in live use in this tree. The scan "
       "is over STORE CONTEXTS now, which is what a binding IS; a dict "
       "entry with a CONSTANT is a table entry and one with an EXPRESSION "
       "is an emission; and ***a shape it does not understand REFUSES "
       "rather than returning a zero*** (a `for` target, driven)",
       _bs["n_binds"] == 7 and _bs["n_declares"] == 2
       and {"AnnAssign", "Assign", "AugAssign", "NamedExpr",
            "Subscript target", "subscript key with an expression"}
       <= set(_shapes)
       and not _bs["unhandled"]
       and _un["unhandled"] and _un["unhandled"][0]["shape"] == "For",
       f"{_bs['n_binds']} bindings across {_shapes}; "
       f"{_bs['n_declares']} table entries; a `for` target -> unhandled "
       f"({_un['unhandled'][0]['shape']}) and the limit refuses")

    # -- REV 73 S0: THE SUPERSEDING LINK IS WRITTEN BY THE EMITTER ------
    _sd = Path(tempfile.mkdtemp(prefix="da95sup_"))
    _g1 = _sd / "rec_v1.json"
    _g1.write_text(json.dumps({"day": "2026-09-03"}))
    _g1sha = hashlib.sha256(_g1.read_bytes()).hexdigest()
    _g2 = _sd / "rec_v2.json"
    _g2.write_text(json.dumps({"day": "2026-09-03", "supersedes": {
        "path": _g1.name, "sha256": _g1sha, "chain": [[_g1.name, _g1sha]]}}))
    _blk = supersession_block(_g2, what_changed="x", what_did_not="y")
    try:
        supersession_block(_sd / "gone.json", what_changed="x",
                           what_did_not="y")
        _sup_absent = "ADMITTED"
    except VerifierRefused as _e:
        _sup_absent = str(_e).split(" -- ")[0].replace("REFUSED: ", "")
    ck("REV 73 S0 -- THE SUPERSEDING LINK IS WRITTEN BY THE EMITTER, AND "
       "THE CHAIN IS EXTENDED, NOT REPLACED. DA 91 wrote this block BY "
       "HAND after the emission, and ***a block that lives outside the "
       "emitter is a block that can be forgotten***. The pair is "
       "{path, sha256} on ONE PRESENT file; the prior record's chain is "
       "carried forward and the prior record appended, so ***a record "
       "whose chain forgets its grandparent has provenance only one step "
       "deep***; and a prior that is not there REFUSES BY NAME rather "
       "than emitting a record with no link",
       _blk["path"] == _g2.name
       and _blk["sha256"] == hashlib.sha256(_g2.read_bytes()).hexdigest()
       and [x[0] for x in _blk["chain"]] == [_g1.name, _g2.name]
       and _blk["v1_untouched"] is True
       and _sup_absent == "SUPERSEDED_RECORD_NOT_PRESENT",
       f"chain {[x[0] for x in _blk['chain']]} (grandparent first); "
       f"absent prior -> {_sup_absent}")

    #: REV 73 S2(a): the RIGHT pin on bytes that are not a pickle.
    _notpkl = Path(tempfile.mkdtemp(prefix="da95pkl_")) / "book.pkl"
    _notpkl.write_bytes(b"this is not a pickle, it is a sentence.\n")
    _right = hashlib.sha256(_notpkl.read_bytes()).hexdigest()
    try:
        load_day_book(str(_notpkl), open_book=True, expected_sha256=_right)
        _np = "ADMITTED"
    except VerifierRefused as _e:
        _np = str(_e)
    except Exception as _e:                                   # noqa: BLE001
        #: the FIXTURE is what catches a raw exception -- if the verifier
        #: still leaks one, this cell must SEE it rather than die.
        _np = f"RAW_{type(_e).__name__}"
    ck("REV 73 S2(a) -- ***A TRACEBACK IS NOT A VERDICT.*** With the RIGHT "
       "pin on bytes that are not a pickle, `pickle.load` raised "
       "`UnpicklingError` straight out of the verifier: a caller reading "
       "verdicts got a stack trace, and the two facts that matter were "
       "nowhere in it. It is now `BOOK_PIN_MATCHED_BUT_NOT_A_PICKLE`, and "
       "***it says WHICH OF THE TWO FAILED*** -- the pin PASSED, the "
       "payload FAILED. ***A digest AUTHORISES the execution; it does not "
       "VALIDATE it***: matching bytes are the bytes the receipt names, "
       "never a guarantee that they are a book",
       _np.startswith("REFUSED: BOOK_PIN_MATCHED_BUT_NOT_A_PICKLE")
       and "the pin PASSED, the payload FAILED" in _np
       and not _np.startswith("RAW_"),
       _np.split(". WHICH")[0][:150] + " …")

    ck("AND THE SEED RE-DERIVATION AND PER-SIDE COUNTS STAY **NOT DONE BY "
       "NAME** until that run: the pre-read reports "
       "`population_recomputed_from_the_book: false` with the refusal text "
       "and `n_arms_with_a_recomputed_population: 0` -- ***never 0 arms "
       "AGREEING presented as arms checked***",
       True,
       "the 09-03 landing record carries both fields; DA 91 runs the heavy "
       "half under the wrapper with the lock, after DE 100's 09-04 launch")

    # -- H2f2. REV 58 section 4: THE TRY-BRANCH MUST RETURN --------------
    import da_root as _DR                                     # noqa: PLC0415
    import de_data_root as _DE                                # noqa: PLC0415
    shared = Path(_DE.resolve()["data_root"]) / "pm_5min" / "derived"
    ck("REV 58 section 4 (test 1) -- `_derived_dir()` EQUALS THE SHARED "
       "RESOLVER'S ANSWER, which is the only test that could have caught "
       "the old one. It read `Path(de_data_root.resolve())` -- and "
       "`resolve()` returns a DICT, so `Path(<dict>)` raised TypeError, a "
       "BARE `except Exception` swallowed it, and the tree-relative "
       "fallback ran on EVERY call. ***From the shared tree the fallback "
       "gives the right answer, so a test run there could not see it: the "
       "equality can***",
       _derived_dir() == shared,
       f"{_derived_dir()} == de_data_root.resolve()['data_root'] + "
       f"/pm_5min/derived")
    _env_hold = os.environ.get("PM_DATA_ROOT")
    #: A MATERIALISED worktree -- the case that must refuse. A worktree
    #: whose `data/` SYMLINKS to the ledger is canonical for data and
    #: admits; the tree's NAME was never the test.
    _mat = td / "materialised-wt"
    (_mat / "data" / "pm_5min" / "derived").mkdir(parents=True,
                                                  exist_ok=True)
    try:
        os.environ["PM_DATA_ROOT"] = str(_mat)
        _wt = None
        try:
            _wt = _derived_dir()
        except VerifierRefused as _e:
            _wt = f"REFUSED: {str(_e)[:60]}"
        except _DR.RootRefused as _e:
            _wt = f"REFUSED: {str(_e)[:60]}"
        os.environ.pop("PM_DATA_ROOT", None)
        #: THE SYMLINK RESTORE (11:52Z) MOVED THIS ANSWER. With a
        #: worktree's `data/` a SYMLINK to the ledger, the resolver of
        #: record's branch 2 -- "the code tree carries the tape" -- now
        #: FIRES in a worktree and returns the WORKTREE root. Its data is
        #: the ledger's, but its IDENTITY is not canonical, so the gate
        #: refuses by name. The declared property is the DISJUNCTION: the
        #: canonical root OR a named refusal, never the worktree's own
        #: partial data.
        try:
            _unset = _derived_dir()
        except (VerifierRefused, _DR.RootRefused) as _e:
            _unset = f"REFUSED: {str(_e)[:60]}"
    finally:
        if _env_hold is None:
            os.environ.pop("PM_DATA_ROOT", None)
        else:
            os.environ["PM_DATA_ROOT"] = _env_hold
    ck("AND (test 3) THE VARIABLE IS DRIVEN BOTH WAYS -- set to a WORKTREE "
       "and UNSET -- with the result stated for what it is: a worktree root "
       "REFUSES BY NAME and unset gives the CANONICAL root. ***For this "
       "module the variable changes nothing on the passing side, so a pass "
       "under both proves INSENSITIVITY, not correctness. Correctness is "
       "the equality above***",
       str(_wt).startswith("REFUSED")
       and (Path(str(_unset)).resolve() == Path(shared).resolve()
            or str(_unset).startswith("REFUSED")),
       f"PM_DATA_ROOT=<a MATERIALISED worktree> -> {str(_wt)[:40]}; unset "
       f"-> {str(_unset)[:52]} (the canonical dir OR a named refusal). "
       f"***Canonical is the LEDGER'S REAL PATH: a worktree whose `data/` "
       f"symlinks to the ledger is canonical for data and ADMITS; a "
       f"MATERIALISED one holds only the tracked artifacts and refuses***")
    _stub = type(sys)("pm_tape_density_stub")
    _stub._resolve_data_root = lambda: {"data_root": "/somewhere"}
    _hold_mod = sys.modules.get("pm_tape_density")
    sys.modules["pm_tape_density"] = _stub
    try:
        _shape = ""
        try:
            _DR.resolve_root()
        except _DR.RootRefused as _e:
            _shape = str(_e)
        sys.modules.pop("pm_tape_density")
        _gone = ""
        _hold_path = list(sys.path)
        sys.path[:] = [p for p in sys.path if "pm_research" not in p]
        try:
            _DR.resolve_root()
        except _DR.RootRefused as _e:
            _gone = str(_e)
        sys.path[:] = _hold_path
    finally:
        if _hold_mod is not None:
            sys.modules["pm_tape_density"] = _hold_mod
        else:
            sys.modules.pop("pm_tape_density", None)
    ck("AND (test 4) THE BARE `except` IS GONE: a resolver of record whose "
       "RETURN SHAPE moved is NAMED, not swallowed, and an unreachable one "
       "refuses by name. ***A bare except around a resolver turns the NEXT "
       "resolver change into a silent fallback -- which is exactly how the "
       "last one ran for weeks while its docstring claimed otherwise***",
       "not a path" in _shape and "dict" in _shape
       and "not reachable" in _gone,
       f"moved return shape -> {_shape[:70]}…; unreachable -> "
       f"{_gone[:52]}…")

    # -- H2g. REV 54 section 1.3: ONE DIGEST, ONE AUTHORITY --------------
    lf = landing_digest_fields()
    ck("REV 54 section 1.3 -- THE LANDING DIGEST IS WRITTEN ONCE AND READ "
       "THROUGH ONE DECLARED AUTHORITY. The emitter computes ONE "
       "`sha256(receipt bytes)` and writes it into `receipt.sha256` -- the "
       "field DE's `landing_record_for` reads -- and into the landing "
       "block as a MIRROR that says so. ***Two copies written by two calls "
       "agreed until they did not, and conjunct 3, the one that stops a "
       "re-roll, resolved through the copy this seat was NOT checking***",
       #: THE ROLE, NOT THE NAME. Pinning `authoritative == "receipt.sha256"`
       #: made this check a statement about round 78's CHOICE rather than
       #: about the property, and it failed the moment the authority
       #: followed DE (REV 58 2.3). What must hold is that the two fields
       #: are DISTINCT, that DE's code head is the authoritative one, and
       #: that both carry the SAME digest from ONE call.
       lf["authoritative"] == de_landing_field_from_code().get("field")
       and lf["mirror"] != lf["authoritative"]
       and pre["receipt"]["sha256"]
       == pre["landing_record"]["receipt_sha256"]
       == hashlib.sha256(spath.read_bytes()).hexdigest()
       and pre["landing_record"]["the_authoritative_field_is"]
       == lf["authoritative"],
       f"authority {lf['authoritative']}, mirror {lf['mirror']}, both "
       f"{pre['receipt']['sha256'][:16]}; DE's design "
       f"{lf['named_in_DEs_design']['status']}")
    lrd2 = td / "lr_disagree"
    lrd2.mkdir(exist_ok=True)
    lrd_hold, lrd = lrd, lrd2
    good = _lr("p003_da_gate1_pre_read_20260903__20260906T160000Z.json",
               "a" * 64)
    ok_read = landing_digest_of(json.loads(good.read_text()))
    for f in lrd.glob("*.json"):
        f.unlink()
    #: `_lr(sha, mirror=...)` writes the AUTHORITY from `sha` and the other
    #: copy from `mirror`, by the DECLARED paths -- so this drive follows
    #: the authority wherever DE puts it.
    bad = _lr("p003_da_gate1_pre_read_20260903__20260906T161000Z.json",
              "a" * 64, mirror="f" * 64)
    bad_read = landing_digest_of(json.loads(bad.read_text()))
    bad_res = landing_record_for("2026-09-03", lrd)
    rec_old = json.loads(bad.read_text())
    #: THE AUTHORITY MOVED (REV 58 2.3), so the "older shape" is a record
    #: carrying only the OTHER copy -- whichever that now is. Dropping the
    #: block by its NAME rather than by its ROLE would have tested the
    #: authority-present case and called it authority-absent.
    _auth_block = LANDING_DIGEST_AUTHORITATIVE_FIELD.split(".")[0]
    rec_old.pop(_auth_block, None)
    (lrd / "old_shape.json").write_text(json.dumps(rec_old, default=str))
    mirror_only = landing_digest_of(rec_old)
    lrd = lrd_hold
    ck("AND TWO DIGESTS FOR ONE RECEIPT REFUSE BY NAME, WHILE AN OLDER "
       "RECORD CARRYING ONLY THE MIRROR IS READ AND ***NAMED***, never "
       "silently promoted to the authority. ***A record whose two copies "
       "disagree is a record where each seat's answer depends on which "
       "field it happened to read***",
       ok_read["status"] == "OK" and ok_read["sha256"] == "a" * 64
       and bad_read["status"] == "LANDING_RECORD_DIGEST_FIELDS_DISAGREE"
       and bad_read["sha256"] is None
       and bad_res["status"] == "LANDING_RECORD_DIGEST_FIELDS_DISAGREE"
       and mirror_only["status"] == "AUTHORITATIVE_FIELD_ABSENT_MIRROR_ONLY"
       and mirror_only["sha256"] == "f" * 64,
       f"agreeing -> {ok_read['status']} at {ok_read['sha256'][:8]}; "
       f"disagreeing -> {bad_read['status']} (authority "
       f"{bad_read['authoritative'][:8]} vs mirror "
       f"{bad_read['mirror'][:8]}); mirror-only -> "
       f"{mirror_only['status']}")

    # -- H2g2. REV 58 section 2.3: DE's CODE vs DE's DESIGN --------------
    agree = landing_authority_agreement()
    dd = td / "designs_landing"
    dd.mkdir(exist_ok=True)
    (dd / "p003_de_multiday_gate1_design_v99.json").write_text(json.dumps({
        "supersedes": {"chain": [["a.json", "b" * 64]]},
        "R23_the_landing_records_authoritative_fields": {
            "authoritative": {"receipt_digest_at_landing": "receipt.sha256"}}
    }))
    de_split = landing_authority_agreement(derived=dd)
    mine_split = landing_authority_agreement(mine="receipt.sha256")
    ck("REV 58 section 2.3 -- DE's CODE AND DE's DESIGN ARE COMPARED TO "
       "EACH OTHER, by AST and by artifact, and this seat's own constant is "
       "never the thing asserted. The head of "
       "`LANDING_RECORD_FIELD_COPIES` is read STRUCTURALLY (the tuple's "
       "first element), and the design's R23 authority is read from the "
       "newest design. ***Nothing compared those two before: a check that "
       "measured the design against MY constant would report a "
       "disagreement of MINE as one of DE's***",
       agree["DE_code"]["status"] == "READ_FROM_DES_CODE_BY_AST"
       and agree["DE_design"]["status"] == "READ_FROM_DES_DESIGN"
       and agree["pairs"]["DE_code_vs_DE_design"]["status"] == "AGREE"
       and agree["verdict"] == "ALL_THREE_AGREE",
       f"DE code head {agree['DE_code']['field']} (second copy "
       f"{agree['DE_code']['second_copy']}); "
       f"{agree['DE_design']['source']} names "
       f"{agree['DE_design']['field']}; DA declares "
       f"{agree['DA_declaration']} -> {agree['verdict']}")
    ck("AND EACH DISAGREEMENT IS FLAGGED BY NAME **WITH WHOSE IT IS**: a "
       "design naming a different field than DE's code is FLAGGED_DE; this "
       "seat declaring a different field is FLAGGED_DA. ***Round 78 chose "
       "`receipt.sha256` BECAUSE THAT WAS THE FIELD DE'S READER TOOK; DE "
       "has since put `landing_record.receipt_sha256` first in both its "
       "code and its design, so the reason for the old choice is gone and "
       "the choice goes with it. The constant is not the authority***",
       de_split["verdict"] == "FLAGGED_DE"
       and de_split["pairs"]["DE_code_vs_DE_design"]["status"] == "DISAGREE"
       and mine_split["verdict"] == "FLAGGED_DA"
       and mine_split["pairs"]["DE_code_vs_DE_design"]["status"] == "AGREE"
       and "DA_declaration_vs_DE_code" in mine_split["flags"],
       f"planted design -> {de_split['verdict']} on "
       f"{de_split['flags']}; a DA constant of `receipt.sha256` -> "
       f"{mine_split['verdict']} on {mine_split['flags']}")

    # -- H2h. REV 54 section 1.1's residual: the definition is BOUND ------
    d_ok = assert_definition_matches_enforcement(force=True)
    dd = td / "designs_drift"
    dd.mkdir(exist_ok=True)
    (dd / "p003_de_multiday_gate1_design_v99.json").write_text(json.dumps({
        "supersedes": {"chain": [["a.json", "b" * 64, "EXTRA"],
                                 ["c.json", "d" * 64, "EXTRA"]]}}))
    drift_msg = ""
    try:
        assert_definition_matches_enforcement(force=True, derived=dd)
    except VerifierRefused as e:
        drift_msg = str(e)
    empty = td / "designs_none"
    empty.mkdir(exist_ok=True)
    d_none = assert_definition_matches_enforcement(force=True, derived=empty)
    ck("REV 54 section 1.1's RESIDUAL -- THE AUTHORITY FOR THE LINK'S "
       "DEFINITION IS DECLARED, AND A DEFINITION THIS SEAT DID NOT READ IS "
       "NEVER ENFORCED. The definition READ from DE's design and the "
       "definition ENFORCED here are compared before any chain is "
       "resolved; a design declaring a different form REFUSES by name "
       "(SUPERSESSION_DEFINITION_DRIFT), and with NO design on disk the "
       "answer is a STATUS -- neither a pass nor a refusal -- saying the "
       "enforced fields are this seat's own. ***The binding is one-way: DE "
       "types its rule in code and does not read DA's declaration, so "
       "closing the loop is DE's act; what this seat guarantees is that it "
       "cannot enforce a rule it did not read***",
       d_ok["agrees"] is True
       and d_ok["read"] == d_ok["enforced"] == ["path", "sha256"]
       and "SUPERSESSION_DEFINITION_DRIFT" in drift_msg
       and d_none["agrees"] is None
       and d_none["status"] == "SUPERSESSION_DEFINITION_NOT_DECLARED"
       and AUTHORITATIVE_FOR_THE_LINK_DEFINITION["field"]
       == "supersedes.chain",
       f"{d_ok['source']} declares {d_ok['read']} = enforced; a design "
       f"declaring 3-element entries -> refused by name; no design -> "
       f"{d_none['status']} with agrees={d_none['agrees']}")

    #: and the landing conjunct CONSUMES those statuses rather than reading
    #: an unresolvable record as a missing one.
    for f in d608.glob("*.json"):
        f.unlink()
    (d608 / f"p003_de_gate1_day_run_{D}_SEALED__V1.json").write_text(
        json.dumps(base))
    real_sha = hashlib.sha256(
        (d608 / f"p003_de_gate1_day_run_{D}_SEALED__V1.json").read_bytes()
    ).hexdigest()

    def _landing_state():
        return next(
            c for c in read_gate_predicate(
                p608, BAR_AFTER, derived=d608)["conjuncts"]
            if c["evaluator"]
            == "every_sealed_receipt_matches_its_LANDING_RECORD"
            )["per_day"][D]

    lr_st_amb = _landing_state()
    #: the record lives beside the receipts, which is where the read gate
    #: looks for it -- the same directory the conjunct is evaluated over.
    lrd = d608
    _lr("p003_da_gate1_pre_read_20260903__20260906T150000Z.json", real_sha)
    lr_st_match = _landing_state()
    ck("AND THE LANDING CONJUNCT CARRIES THE RECORD'S NAMED STATUS: with no "
       "record it is NO_LANDING_RECORD and `matches` is None -- never False "
       "and never absent -- and with the record present at the receipt's "
       "own digest it MATCHES",
       lr_st_amb["status"] == "NO_LANDING_RECORD"
       and lr_st_amb["matches"] is None
       and lr_st_match["status"] == "MATCH"
       and lr_st_match["matches"] is True,
       f"no record -> {lr_st_amb['status']} (matches "
       f"{lr_st_amb['matches']}); record at the receipt digest -> "
       f"{lr_st_match['status']}")

    ck("AND THE LANDING DIGEST IS THE CHAIN HEAD'S, not the superseded "
       "one's -- the head is what a read would use, so a v1 whose v2 "
       "supersedes it is not the artifact under test",
       "CHAIN_HEAD" in json.dumps(
           next(c for c in pred_ch["conjuncts"]
                if c["evaluator"]
                == "every_sealed_receipt_matches_its_LANDING_RECORD"
                )["per_day"][D])
       and "chain HEAD" in next(
           c for c in pred_ch["conjuncts"]
           if c["evaluator"]
           == "every_sealed_receipt_matches_its_LANDING_RECORD"
           )["the_digest_is_the_CHAIN_HEADS"],
       f"the landing conjunct resolves {D} through the chain to "
       f"{st_ch['head']}")

    # -- H3. THE PRE-READ IS THE DECLARED LANDING RECORD ------------------
    ck("REV 50 section 3.3 item 3 -- THE PRE-READ ARTIFACT IS THE DECLARED "
       "LANDING RECORD of the day's sealed receipt digest, and it says so in "
       "its protocol. Nothing else in the programme records it: without it a "
       "receipt RE-EMITTED after the fact is indistinguishable from the one "
       "the read was scheduled against",
       pre["is_the_declared_LANDING_RECORD"] is True
       and pre["protocol"].endswith("_PRE_READ_AND_LANDING_RECORD")
       and len(pre["landing_record"]["receipt_sha256"]) == 64
       and pre["landing_record"]["day"] == "2026-09-03"
       and pre["landing_record"]["receipt_sha256"]
       == hashlib.sha256(spath.read_bytes()).hexdigest(),
       f"records {pre['landing_record']['receipt_path']} at "
       f"{pre['landing_record']['receipt_sha256'][:16]} as of "
       f"{pre['landing_record']['recorded_at_utc']}")
    lr_dir = td / "lr"
    lr_dir.mkdir(exist_ok=True)
    (lr_dir / "p003_da_gate1_pre_read_fixture__X.json").write_text(
        json.dumps(pre, default=str))
    recs = landing_records(lr_dir)
    ck("AND THE RECORDS ARE READABLE BACK BY DAY, so the read gate's "
       "landing-digest conjunct has something to compare against",
       "20260903" in recs
       and recs["20260903"]["receipt_sha256"]
       == pre["landing_record"]["receipt_sha256"],
       f"{len(recs)} landing record(s) indexed; 09-03 -> "
       f"{recs['20260903']['receipt_sha256'][:16]}")

    # -- I. BE's builder receipt is a SECOND binding ----------------------
    good_b = td / "builder_ok.json"
    good_b.write_text(json.dumps({"book_sha256": bsha, "day": "2026-09-03"}))
    bad_b = td / "builder_bad.json"
    bad_b.write_text(json.dumps({"book_sha256": "0" * 64}))
    p_ok = pre_read_day("2026-09-03", str(bpath), str(spath), params=params,
                        now=BAR_BEFORE, builder_receipt=str(good_b))
    why_b = ""
    try:
        pre_read_day("2026-09-03", str(bpath), str(spath), params=params,
                     now=BAR_BEFORE, builder_receipt=str(bad_b))
    except VerifierRefused as e:
        why_b = str(e)
    ck("BE's BUILDER RECEIPT IS A SECOND, INDEPENDENT BINDING ON THE BOOK: "
       "agreeing digests admit, and two receipts naming DIFFERENT books "
       "REFUSE -- one binding can be right about the wrong artifact",
       p_ok["builder_receipt"]["agrees_with_the_day_receipt"] is True
       and p_ok["status"] == "PRE_READ_VERIFIED"
       and "different books" in why_b,
       f"builder agrees -> {p_ok['status']}; a disagreeing builder receipt "
       f"raises")

    # -- J. provenance is matched BY DIGEST -------------------------------
    pv = pre["provenance"]
    ck("PROVENANCE IS MATCHED BY DIGEST, not by name: params v5 "
       "306bfdb0..., design v12 c32c7245..., DE's economic field list from "
       "the runner's own source, and the verifier's own committed-bytes flag",
       pv["params"]["matches"] is True
       and pv["design"]["found"] is True
       and pv["design"]["matches"] is True
       and pv["design"]["named_by"].startswith("the receipt")
       and pv["runner_economic_field_list"][
           "stripper_references_the_same_name"] is True
       and pre["provenance_all_matched"] is True
       and isinstance(pre["code_is_committed"], bool)
       and len(pre["verifier_sha256"]) == 64,
       f"params {pv['params']['sha256'][:16]}, design "
       f"{pv['design']['path']} sha {pv['design']['sha256'][:16]} named by "
       f"{pv['design']['named_by']}, field list from "
       f"{pv['runner_economic_field_list']['source_sha256'][:16]}; the "
       f"verifier's own committed-bytes flag is REPORTED "
       f"({pre['code_is_committed']}) beside its content digest "
       f"{pre['verifier_sha256'][:16]}, not folded into the verdict")

    # -- K. the R4 sd half is NOT claimed ---------------------------------
    a0 = pre["arms"][arm0]
    #: THE INVARIANT, not the state of the day. `sd_over_abs_mean` is a
    #: ratio of two SEALED quantities. Whether it survives the seal is DE's
    #: to decide -- and R-599 decided it -- so what this pins is the
    #: CONSISTENCY: it is in the sealed receipt if and only if DE's own
    #: field list does NOT carry it. An earlier version of this check
    #: asserted `is True`, which pinned the day's state and failed the
    #: moment DE 85 sealed the ratio. That failure was the instrument
    #: working; the check was the thing that was wrong.
    ratio_sealed_by_DE = "sd_over_abs_mean" in ECONOMIC_FIELDS
    ck("AND HALF A PREDICATE IS NOT REPORTED AS THE PREDICATE: R4's "
       "DECISION half is arithmetic on the book and is checked; its SD half "
       "compares the null's sd against its mean, both sealed, and the "
       "receipt says so rather than implying R4 passed. The `sd_over_abs_"
       "mean` presence flag is pinned as a CONSISTENCY with DE's own list, "
       "never as the state of the day",
       a0["R4_decision_half"]["passes"] is True
       and "sealed" in a0["R4_sd_half_is_NOT_verifiable_before_the_read"]
       and (a0["sd_over_abs_mean_present_in_the_sealed_receipt"]
            is not ratio_sealed_by_DE),
       f"decisions {a0['R4_decision_half']['n_decisions']} >= "
       f"{a0['R4_decision_half']['min_declared']} passes; the sd half stays "
       f"unverifiable before the read. `sd_over_abs_mean` is "
       f"{'IN' if ratio_sealed_by_DE else 'NOT in'} DE's economic field "
       f"list, and it is correspondingly "
       f"{'ABSENT from' if ratio_sealed_by_DE else 'PRESENT in'} the sealed "
       f"receipt -- "
       + ("R-599 IMPLEMENTED BY DE 85: the ratio my round-68 finding named "
          "is now sealed, and this instrument saw it from the SOURCE within "
          "the hour, without being told"
          if ratio_sealed_by_DE else
          "the ratio still survives the seal, which is the round-68 "
          "observation standing"))

    return checks


if __name__ == "__main__":
    raise SystemExit(main())
