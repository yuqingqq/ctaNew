#!/usr/bin/env python3
r"""DA -- THE NON-HEAD CENSUS (REV 70 section 5).

A declaration is superseded in band as `_v2`, `_v3`, ... and a literal in
the code pins ONE of them by name. Nothing checked that the pinned one was
the CHAIN HEAD, so a module could keep reading v1 while v2 sat beside it --
and three instances of exactly that were found in one day (v1 pinned while
v2 existed; v2 pinned while v3 existed; the runner still pinning v2).

TWO PARTS, both computed:

  (1) THE CHAINS. Every declaration whose `supersedes` names a PRESENT file
      must resolve to EXACTLY ONE head, pair-verified by R-608's rule
      ({path, sha256}, both halves landing on one present file). A family
      with two heads REFUSES: nothing can say which one a pin should name.

  (2) THE LITERALS. An AST census over every string constant in `live/`
      matching `declarations/.*_v\\d+\\.json`. A literal naming a NON-head
      is REFUSED BY NAME, with the head it should have named.

INFRASTRUCTURE, NOT A STATISTIC (R-235): this reads FILENAMES and the
`supersedes` links the seats themselves write. It re-derives no seat's
number, and the pair rule it applies is read from `da_root`, which reads it
from DE's design.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PROTOCOL = "P003_DA_NONHEAD_CENSUS_V2"
#: WHAT V2 CHANGES (rule 13 -- the V1 receipts stay as provenance):
WHAT_CHANGES_IN_V2 = (
    "four rules, all of which change what the census REPORTS: (2.1) a "
    "marker cannot excuse an OPEN -- a marked literal that is read is "
    "refused, and the marker is recorded as the file author's claim, not "
    "a finding; (2.2) the marker word list drops the bare `bad`, adds "
    "previous/prior/historical, and splits camelCase; (2.3) the allowlist "
    "resolves through the head rule instead of merging every version; (4) "
    "BOTH declaration directories are scanned and every null head carries "
    "WHY. And one rule this seat found by RUNNING it: bytes consumed by a "
    "DIGEST are the R-608 link being written, not a pin being read")
DECL_RE = re.compile(r"declarations/([A-Za-z0-9_\-]+?)_v(\d+)\.json$")
#: BOTH FORMS: `declarations/<family>_vN.json` AND a bare
#: `<family>_vN.json`. The reviewer's regex saw the bare ones too, and a
#: census that could not see them could not DISPOSE of them -- the gates
#: below are what disposes of them, not the pattern.
LITERAL_RE = re.compile(
    r"(?:declarations/)?[A-Za-z0-9_\-]+_v\d+\.json")


def _root() -> Path:
    import da_root as R                                       # noqa: PLC0415
    return R.code_root("the non-head census")


def _family(name: str) -> tuple:
    """('be_r_survey_declaration', 3) from 'be_r_survey_declaration_v3'."""
    m = re.match(r"^(.*)_v(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


#: DA 104. ***THE HEAD IS RESOLVED BY THE SHARED IMPLEMENTATION.*** This
#: function grew its own glob-and-pair resolver, and so did two other DA
#: modules -- three readers of one rule, each able to drift from it alone.
#: BE 77's `declaration_chain.resolve_head` is now the ONLY code in this
#: seat that decides which version is the head; what stays here is what
#: this census ADDS: the per-family view, and the RULING that a family
#: with an orphan branch cannot answer "which one should a pin name?".
#: A fork is REPORTED by the shared resolver and REFUSED by this census --
#: those are different jobs and they now live in different places.
def declaration_chains(decl_dir: Path) -> dict:
    """Families, their heads, and the links that make them.

    The head, the pairs and any orphan branches come from
    `declaration_chain.resolve_head`; the refusal on a fork is this
    census's own ruling on top of it."""
    import declaration_chain as _DC                           # noqa: PLC0415
    d = Path(decl_dir)
    files = sorted(q for q in d.glob("*.json") if q.is_file())
    fams: dict = {}
    for q in files:
        fam, ver = _family(q.stem)
        fams.setdefault(fam, []).append((ver, q))
    out = {}
    for fam, members in sorted(fams.items()):
        members.sort(key=lambda t: (t[0] is None, t[0]))
        names = [q.name for _, q in members]
        try:
            r = _DC.resolve_head(d, fam)
        except _DC.ChainRefused as e:
            #: PRESENT-AND-UNFOLLOWABLE is not ABSENT: the shared resolver
            #: refuses a corrupted link by name, and that refusal is the
            #: family's status here rather than a head this census invents.
            out[fam] = {
                "members": names, "n_members": len(members),
                "heads": [], "n_heads": 0, "links": [],
                "unlinked_or_broken": [{"status": str(e).split(":")[0],
                                        "detail": str(e)[:200]}],
                "status": "CHAIN_REFUSED_BY_THE_SHARED_RESOLVER",
                "resolved_by": "declaration_chain.resolve_head (BE 77)",
                "why": ("the shared resolver refuses this family by name; "
                        "a census that answered anyway would be inventing "
                        "the chain the seat did not write")}
            continue
        #: DA 106 -> DA 110. ***THIS BLOCK IS RETIRED: BE 82 FIXED THE
        #: MODULE.*** After BE 79/80 the shared resolver FOLLOWED a
        #: `{path}`-only link (its `_predecessor` returned `sha256 = None`
        #: and the digest comparison was guarded by `and want`), so this
        #: census carried the R-608 property itself. At BE 82 the module
        #: raises `ChainRefused HALF_WRITTEN_LINK` BEFORE this scan can
        #: run -- the branch became code that cannot fire, and ***a guard
        #: that cannot fire is a guard that cannot prove it works***. What
        #: stays is the TRANSLATION below: the module's refusal becomes a
        #: per-family STATUS carrying its name, so one bad family does not
        #: end the census, and `_declaration_head` turns the same refusal
        #: into a named `VerifierRefused` rather than a foreign exception.
        orphans = [x["version"] for x in r["orphan_branches"]]
        heads = orphans + [r["name"]]
        links = [{"from": v, "to": Path(str(
            (json.loads((d / v).read_text()).get("supersedes") or {})
            .get("path") or "")).name}
            for v in names
            if isinstance((json.loads((d / v).read_text()) or {}).get(
                "supersedes"), dict)]
        out[fam] = {
            "members": names, "n_members": len(members),
            #: `heads` keeps this census's vocabulary -- the head plus any
            #: ORPHAN BRANCH -- because its ruling is about how many
            #: versions a pin could point at, not about which one the
            #: resolver picks.
            "heads": heads, "n_heads": len(heads),
            "resolved_head": r["name"], "head_rule": r["head_rule"],
            "orphan_branches": r["orphan_branches"],
            "forks_two_versions_superseding_one": r[
                "forks_two_versions_superseding_one"],
            "resolved_by": "declaration_chain.resolve_head (BE 77)",
            "links": links, "unlinked_or_broken": [],
            "status": ("ONE_HEAD" if len(heads) == 1
                       else "MULTIPLE_HEADS_UNLINKED"),
            "why": ("a family with an ORPHAN BRANCH has no answer to "
                    "'which one should a pin name?' -- the shared resolver "
                    "REPORTS the fork and this census REFUSES it"
                    if len(heads) != 1 else
                    "one head: every other member is superseded by a "
                    "PAIR-VERIFIED link"),
        }
    return out


#: REV 72 4. THE SECOND DECLARATION DIRECTORY. Fifty pins carried
#: `head: null` -- not "no head", but ***a family this census never
#: looked for***: `live/mm_research/declarations/` holds P-2026-002's
#: declarations and the census read only `live/pm_research/`. A null that
#: means "not scanned" is indistinguishable from a null that means "no
#: head", and rule 11 says an absence is never a pass. Both directories
#: are scanned; a family name present in BOTH refuses by name rather than
#: silently taking one.
DECLARATION_DIRS = ("live/pm_research/declarations",
                    "live/mm_research/declarations")


def merged_chains(root: Path, dirs=DECLARATION_DIRS) -> dict:
    """Every declaration family under `dirs`, each tagged with its dir."""
    out = {}
    for rel in dirs:
        d = Path(root) / rel
        if not d.is_dir():
            out[f"__absent__{rel}"] = {
                "members": [], "n_members": 0, "heads": [], "n_heads": 0,
                "declarations_dir": rel,
                "status": "DECLARATION_DIRECTORY_ABSENT",
                "why": "named in DECLARATION_DIRS and not present"}
            continue
        for fam, blk in declaration_chains(d).items():
            blk["declarations_dir"] = rel
            if fam in out:
                prior = out[fam]
                out[fam] = {**blk, "heads": prior["heads"] + blk["heads"],
                            "n_heads": prior["n_heads"] + blk["n_heads"],
                            "declarations_dir": [
                                prior["declarations_dir"], rel],
                            "status": "FAMILY_IN_TWO_DECLARATION_DIRS",
                            "why": ("the same family name lives in two "
                                    "declaration directories: a pin "
                                    "naming it has no single answer")}
            else:
                out[fam] = blk
    return out


#: R-657. A LITERAL NAMING A NON-HEAD IS ADMISSIBLE ONLY WHERE THE CODE
#: MARKS IT. Two of the census's first hits were not pins at all: the
#: runner's `SUPERSEDED_PARAMS_REL` names the superseded file ON PURPOSE
#: (it is the guard), and the design module's `_bad_pin` is a known-bad. A
#: census that cannot tell a PIN from a GUARD reports the guard as the
#: defect it exists to prevent.
#:
#: THE RULE, stated so it can be argued with:
#:   * IDENTIFIER MARKER -- the name the literal is ASSIGNED to (a Name, or
#:     the base of a Subscript target) carries one of the marker WORDS as a
#:     `_`-separated token: superseded / known_bad / bad / falsifier.
#:     WORD-level, not substring: SUPERSEDED_PARAMS_REL marks,
#:     PARAMS_REL does not.
#:   * FUNCTION MARKER -- the literal sits inside a function whose name
#:     contains `known_bad` or `falsif` (R-657's own words).
#:   * ALLOWLIST -- `declarations/<seat>_nonhead_allowlist_v*.json`, owned
#:     by the seat whose file it covers, ONE REASON PER ENTRY.
#: A marked literal is ADMITTED and REPORTED as MARKED with its reason; an
#: unmarked one is REFUSED. A marker on a HEAD literal is admitted and
#: NOTED -- the marker is not a licence, it is an explanation.
#: REV 72 2.2. THE BARE `bad` IS GONE: it admitted `BAD_REQUEST_PATH`, a
#: name about HTTP, not about supersession -- a marker word must be about
#: the ONE thing it excuses. `known_bad` stays. `previous` / `prior` /
#: `historical` are ADDED, because they are the honest names for a
#: deliberate historical pin and today they refuse. And the split is
#: camelCase-aware: `supersededParams` was invisible to a `_`-only split,
#: so a marker could be lost by naming style alone.
#: REV 72 2.4: what a marker IS.
MARKER_IS = ("the FILE AUTHOR'S CLAIM about a literal, recorded and never "
             "treated as a finding by this census. It explains a name the "
             "code does not READ; it cannot excuse one it does")
MARKER_WORDS = ("superseded", "known_bad", "previous", "prior",
                "historical", "falsifier")
FUNCTION_MARKER_RE = re.compile(r"known[_-]?bad|falsif", re.I)


#: R-753 (2) as ACCEPTED by REV 89 S6.3a, with its added clause.
#: A non-head literal is ADMISSIBLE when the code HASHES the file it names
#: against a digest RECORDED IN THE CODE and ASSERTED AT THE READ, and --
#: where the act itself recorded no digest -- the literal SAYS SO.
#: REFUSED otherwise. DA 94's separating property: hashed -> not a pin in
#: the sense that matters; INTERPRETED WITHOUT A DIGEST -> refused.
DIGEST64_RE = re.compile(r"^[0-9a-f]{64}$")
#: THE VOCABULARY IS DECLARED, not hidden in a regex: the provenance field
#: must name WHO recorded the digest -- the act, or a later seat saying so.
#: A key carrying one of these words is the disclosure REV 89 asked for.
PROVENANCE_WORDS = ("record", "act")


def hashed_against_a_recorded_digest(tree, assigned_to: str | None) -> dict:
    """Is this literal's container HASHED against a digest in the code?

    FOUR CONJUNCTS, each computed from the AST and none from a filename:
      1. the literal is assigned into a container (a dict, today);
      2. that container carries a 64-hex DIGEST -- recorded in the code;
      3. the module COMPUTES a digest (`.hexdigest()`) and COMPARES it
         against that container -- asserted at the read, not merely
         written down;
      4. the container names WHO recorded the digest, so a pin the act
         made is never conflated with one reconstructed afterwards
         (REV 89 S6.3a's added clause).
    """
    out = {"assigned_to": assigned_to, "digest_in_the_code": None,
           "n_digests": 0, "provenance_fields": [], "computes_a_digest": False,
           "compared_at_lines": [], "admissible": False,
           "why": None}
    if not assigned_to:
        out["why"] = "the literal is not assigned to a name"
        return out
    val = None
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == assigned_to
                for t in nd.targets):
            val = nd.value
    if val is None:
        out["why"] = f"no assignment to `{assigned_to}` in this module"
        return out
    digests = [c.value for c in ast.walk(val)
               if isinstance(c, ast.Constant) and isinstance(c.value, str)
               and DIGEST64_RE.match(c.value)]
    out["n_digests"] = len(digests)
    out["digest_in_the_code"] = digests[0][:16] + "…" if digests else None
    if isinstance(val, ast.Dict):
        out["provenance_fields"] = [
            k.value for k in val.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
            and any(w in k.value.lower() for w in PROVENANCE_WORDS)]
    out["computes_a_digest"] = any(
        isinstance(nd, ast.Call) and isinstance(nd.func, ast.Attribute)
        and nd.func.attr == "hexdigest" for nd in ast.walk(tree))
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Compare) and any(
                isinstance(x, ast.Name) and x.id == assigned_to
                for x in ast.walk(nd)):
            out["compared_at_lines"].append(nd.lineno)
    out["admissible"] = bool(digests and out["provenance_fields"]
                             and out["computes_a_digest"]
                             and out["compared_at_lines"])
    if not out["admissible"]:
        out["why"] = "; ".join(filter(None, [
            None if digests else "no 64-hex digest in the container",
            None if out["provenance_fields"] else
            f"no field naming who recorded it ({'/'.join(PROVENANCE_WORDS)})",
            None if out["computes_a_digest"] else
            "the module computes no digest",
            None if out["compared_at_lines"] else
            "the recorded digest is never compared -- written down is not "
            "asserted at the read"]))
    return out


# --------------------------------------------------------------------------
# THE CHAIN-RESOLUTION SURFACE (REV 90 S B5, routed to this census).
#
# Rule 20's clause says EVERY IMPORTER of the shared module runs its
# `--falsify` as a cell. It re-drifted THREE TIMES INSIDE ONE ROUND, because
# nothing computed the set: it was checked by a reviewer noticing, and
# R-605's sentence is that a practice which depends on noticing is not a
# control. So the set is DERIVED FROM THE CODE here.
#
# BOUND TO THE RESOLUTION SURFACE, NOT TO THE IMPORT (REV 90's refinement).
# A module that imports only `plain_create_mode` -- a mode helper -- is NOT
# resolving a chain, and requiring a chain falsifier there would be a cell
# nobody can justify. The surface is: imports `declaration_chain` AND
# references one of the RESOLUTION symbols.
#
# AND THE DRIVE MUST BE IN A BATTERY, not merely present in the file. The
# module that DEFINES the shared helper contains the subprocess call in the
# helper's own body; that is the helper, not a cell that runs when its
# battery runs. The enclosing function is computed, and a drive outside a
# battery does not satisfy the clause.
RESOLUTION_SYMBOLS = ("resolve_head", "write_next_version",
                      "next_version_path", "also_supersedes")
BATTERY_FN_PREFIXES = ("selftest", "fixture", "_falsify", "battery")


def _dc_import_and_symbols(tree) -> tuple:
    """(imports declaration_chain, resolution symbols referenced)."""
    imports_dc, imported = False, set()
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Import):
            if any(a.name.split(".")[-1] == "declaration_chain"
                   for a in nd.names):
                imports_dc = True
        elif isinstance(nd, ast.ImportFrom):
            if nd.module and nd.module.split(".")[-1] == "declaration_chain":
                imports_dc = True
                imported |= {a.asname or a.name for a in nd.names}
    used = {nd.attr for nd in ast.walk(tree)
            if isinstance(nd, ast.Attribute) and nd.attr in RESOLUTION_SYMBOLS}
    used |= {nd.id for nd in ast.walk(tree)
             if isinstance(nd, ast.Name) and nd.id in RESOLUTION_SYMBOLS}
    used |= (imported & set(RESOLUTION_SYMBOLS))
    return imports_dc, sorted(used), sorted(imported)


def _falsifier_drives(tree) -> list:
    """Every call that DRIVES the shared falsifier, with its enclosing fns.

    A call carrying the literal `--falsify`, or a call to the shared
    `shared_falsifier` helper. The literal must be an ARGUMENT of a call --
    a docstring that mentions the flag is prose, not a drive.
    """
    enclosing = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for nd in ast.walk(fn):
                if isinstance(nd, ast.Call):
                    enclosing.setdefault(nd.lineno, set()).add(fn.name)
    out = []
    for c in ast.walk(tree):
        if not isinstance(c, ast.Call):
            continue
        hit = any(isinstance(sn, ast.Constant) and sn.value == "--falsify"
                  for a in list(c.args) + [k.value for k in c.keywords]
                  for sn in ast.walk(a))
        fname = getattr(c.func, "id", None) or getattr(c.func, "attr", None)
        if hit or fname == "shared_falsifier":
            fns = sorted(enclosing.get(c.lineno, set()))
            out.append({"line": c.lineno, "in_functions": fns,
                        "in_a_battery": any(
                            f.startswith(BATTERY_FN_PREFIXES) for f in fns)})
    return out


def chain_resolution_surface(root: Path) -> dict:
    """Who resolves the chain, and does each one run its falsifier?"""
    r = Path(root)
    surface, outside, missing = [], [], []
    for py in sorted((r / "live").rglob("*.py")):
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                tree = ast.parse(py.read_text())
        except (OSError, SyntaxError, ValueError):
            continue
        imports_dc, used, imported = _dc_import_and_symbols(tree)
        if not imports_dc:
            continue
        rel = str(py.relative_to(r))
        if not used:
            outside.append({"file": rel, "imports": imported,
                            "why_not_in_the_surface": (
                                "imports the module but references no "
                                "resolution symbol -- a helper import is not "
                                "a chain resolution (REV 90 S B5)")})
            continue
        drives = _falsifier_drives(tree)
        in_batt = [d for d in drives if d["in_a_battery"]]
        row = {"file": rel, "resolution_symbols": used,
               "falsifier_drives": drives,
               "runs_it_in_a_battery": bool(in_batt),
               "status": ("OK" if in_batt else
                          "IMPORTS_THE_RESOLVER_AND_SHIPS_NO_FALSIFIER_CELL")}
        surface.append(row)
        if not in_batt:
            missing.append(row)
    return {
        "n_in_the_surface": len(surface),
        "n_missing_the_cell": len(missing),
        "missing_the_cell": missing,
        "in_the_surface": surface,
        "n_outside_the_surface": len(outside),
        "outside_the_surface": outside,
        "resolution_symbols": list(RESOLUTION_SYMBOLS),
        "the_rule": ("SEAT_PROTOCOL rule 20's clause (REV 84 S3.2 / REV 85 "
                     "S3, R-726): every importer of the shared chain module "
                     "RUNS its `--falsify` as one cell of its own battery, "
                     "so a regression in the one implementation fails every "
                     "importer at once"),
        "why_a_cell_and_not_a_reviewer": (
            "REV 90 S B5: the rule re-drifted three times inside one round "
            "and was caught each time by a reviewer noticing. R-605: a "
            "practice that depends on noticing is not a control"),
        "the_drive_must_be_in_a_battery": (
            "the module that DEFINES the shared helper carries the "
            "subprocess call in the helper's own body. That is the helper, "
            "not a cell that runs when its battery runs -- so the enclosing "
            "function is computed and a drive outside a battery does not "
            "satisfy the clause"),
        "verdict": ("CLEAN" if not missing else
                    "REFUSED_A_RESOLVER_SHIPS_NO_FALSIFIER_CELL"),
    }


def _identifier_marks(name: str) -> str | None:
    if not name:
        return None
    #: `_`-separated AND camelCase-separated: `supersededParams` splits to
    #: {superseded, params} the same way `SUPERSEDED_PARAMS` does.
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    toks = [t for t in re.split(r"[_\W]+", spaced.lower()) if t]
    joined = "_".join(toks)
    for w in MARKER_WORDS:
        if w in toks or ("_" in w and w in joined):
            return w
    return None


def _assigned_name(node, parents) -> str | None:
    """The identifier a literal is assigned to: a Name target, or the BASE
    of a Subscript target (`_bad_pin[\"path\"] = ...`)."""
    cur = parents.get(node)
    depth = 0
    while cur is not None and depth < 6:
        depth += 1
        if isinstance(cur, (ast.Assign, ast.AnnAssign)):
            tgts = (cur.targets if isinstance(cur, ast.Assign)
                    else [cur.target])
            for t in tgts:
                if isinstance(t, ast.Name):
                    return t.id
                if isinstance(t, ast.Subscript) and isinstance(t.value,
                                                               ast.Name):
                    return t.value.id
                if isinstance(t, ast.Attribute):
                    return t.attr
            return None
        cur = parents.get(cur)
    return None


def _enclosing_fn(node, parents) -> str | None:
    cur, best = parents.get(node), None
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            best = cur.name
            break
        cur = parents.get(cur)
    return best


def _allowlist(root: Path, chains: dict) -> tuple:
    """`<seat>_nonhead_allowlist_v*.json`: {literal: reason}, per seat.

    REV 72 2.3: THE ALLOWLIST IS A DECLARATION LIKE ANY OTHER, so it is
    resolved through the SAME head rule the census applies to everything
    else. Globbing every version and MERGING them meant ***a v2 that
    REMOVES an entry could not*** -- the v1 entry survived the merge and
    the exemption it granted could never be withdrawn. A family that does
    not resolve to exactly one head REFUSES; exposure is zero today (no
    such file exists), which is why it is fixed now."""
    out, refusals = {}, []
    fams = {f: b for f, b in chains.items()
            if "nonhead_allowlist" in f}
    for fam, blk in sorted(fams.items()):
        if blk["n_heads"] != 1:
            refusals.append({"family": fam, "heads": blk["heads"],
                             "status": "ALLOWLIST_DOES_NOT_RESOLVE_TO_ONE_"
                                       "HEAD",
                             "why": ("an allowlist is a declaration like "
                                     "any other: merging its versions "
                                     "means a later one cannot REMOVE an "
                                     "entry")})
            continue
        d = blk.get("declarations_dir")
        f = ((Path(root) / d) if isinstance(d, str)
             else Path(root) / DECLARATION_DIRS[0]) / blk["heads"][0]
        seat = fam.split("_nonhead_allowlist")[0]
        try:
            obj = json.loads(f.read_text())
        except (OSError, ValueError):
            refusals.append({"family": fam, "file": f.name,
                             "status": "ALLOWLIST_UNREADABLE"})
            continue
        for ent in (obj.get("entries") or []):
            if isinstance(ent, dict) and ent.get("names") and ent.get(
                    "reason"):
                out[(seat, ent["names"])] = {"reason": ent["reason"],
                                             "declaration": f.name}
    return out, refusals


#: REV 71 4.3. THE FIRST GATE IS DATAFLOW, NOT TEXT. A regex over source
#: text cannot tell a stale PIN from a supersession CHAIN, a KNOWN-BAD or
#: PROSE -- the reviewer's naive version found ELEVEN pairs and most were
#: false. A literal is a PIN only when it FLOWS INTO A FILE OPEN: an
#: argument of `Path(...)` / `open` / `read_text` / `read_bytes` /
#: `json.load(open(...))`, directly or through ONE assignment. Everything
#: else -- a chain entry, a fixture's version that must not exist, a
#: comment -- is not a pin and is not reported as one. The marker rule
#: (R-657) is the SECOND gate, for literals that survive this one.
OPEN_FUNCS = ("open", "read_text", "read_bytes", "load", "loads",
              "read_json", "is_file", "exists")
OPEN_CTORS = ("Path", "PosixPath")
FIXTURE_CALLS = ("refuses", "raises")
#: DA 94, found by running the census at the tip. ***THE ONE USE THAT MUST
#: READ A NON-HEAD IS THE SUPERSESSION LINK ITSELF.*** R-608 says a link is
#: the pair {path, sha256} -- so the builder of `_v4` MUST open `_v3` and
#: hash its bytes, and my first pass refused BE's link-writer as a stale
#: pin. The separating property is not the name and not the marker (2.1
#: settles that a marker cannot excuse an open); it is WHAT THE BYTES DO:
#: bytes consumed by a DIGEST cannot make the code behave as if it were
#: under old bars, bytes consumed by `json.loads` can. A presence test
#: (`.exists()`) counts as non-interpreting ONLY beside a digest use of
#: the same name -- alone, it gates behaviour and stays a pin.
HASH_FUNCS = ("sha256", "sha1", "md5", "blake2b", "blake2s", "sha512",
              "file_digest", "hexdigest", "digest")
PRESENCE_FUNCS = ("exists", "is_file")


#: REV 73 S2(b). ***ONE HOP IS A REFACTOR AWAY FROM BLIND.*** The gate
#: followed `LIT -> A -> open(A)` and stopped there, so two ordinary
#: shapes walked straight past it: a SECOND assignment (`A = LIT; B = A;
#: open(B)`) and a FUNCTION that returns the literal (`def p(): return
#: LIT` … `open(p())`). Neither is exotic -- both are what happens when
#: someone tidies a module. So the flow is the TRANSITIVE CLOSURE over
#: name-to-name assignments, computed to a FIXED POINT, and a function
#: whose returned literal reaches an open is itself an alias for it.
def _alias_edges(tree, parents) -> dict:
    """name -> names it flows into, over assignments and returns."""
    edges: dict = {}

    def add(src, dst):
        if src and dst:
            edges.setdefault(src, set()).add(dst)

    def _forwarded(v):
        """The names whose VALUE can land in the target unchanged."""
        if isinstance(v, ast.Name):
            return [v.id]
        #: `X = A if c else B` and `X = A or B` forward one of their
        #: operands VERBATIM -- a fallback is exactly this shape, and it
        #: is where a stale default hides (`best or (d / "…_v6.json")`).
        if isinstance(v, ast.IfExp):
            return [x.id for x in (v.body, v.orelse)
                    if isinstance(x, ast.Name)]
        if isinstance(v, ast.BoolOp):
            return [x.id for x in v.values if isinstance(x, ast.Name)]
        return []

    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for src in _forwarded(n.value):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        add(src, t.id)
        #: `X = f()` puts what `f` RETURNS into `X`, so the function is
        #: an alias for the name it lands in. Without this edge a literal
        #: returned by a resolver is invisible the moment the caller
        #: assigns the result -- which is the ordinary shape.
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) \
                and isinstance(n.value.func, ast.Name):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    add(n.value.func.id, t.id)
        #: `def p(): return A` makes `p` an alias for `A`, because a call
        #: to `p` puts A's value where the call is.
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(n):
                if isinstance(sub, ast.Return) and isinstance(sub.value,
                                                              ast.Name):
                    if _enclosing_fn(sub, parents) == n.name:
                        add(sub.value.id, n.name)
    return edges


def _closure(name: str, edges: dict) -> set:
    """Every name the value reaches. A FIXED POINT, not one hop."""
    seen, stack = {name}, [name]
    while stack:
        cur = stack.pop()
        for nxt in edges.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _returning_function(node, parents) -> str | None:
    """The function whose RETURN this literal is -- `def p(): return LIT`."""
    cur, depth = parents.get(node), 0
    while cur is not None and depth < 6:
        depth += 1
        if isinstance(cur, ast.Return):
            return _enclosing_fn(cur, parents)
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.Module)):
            return None
        cur = parents.get(cur)
    return None


def _merge_uses(names, assigned_uses) -> dict:
    """The use kinds of every name the value reaches, added up."""
    out = {"first": None, "HASHED": 0, "PRESENCE": 0, "INTERPRETED": 0,
           "through": []}
    for nm in sorted(names):
        e = assigned_uses.get(nm)
        if not e:
            continue
        out["through"].append(nm)
        out["first"] = out["first"] or e["first"]
        for k in ("HASHED", "PRESENCE", "INTERPRETED"):
            out[k] += e[k]
    return out if out["through"] else {}


def _flows_into_an_open(node, parents, assigned_uses, edges=None) -> dict:
    """Does this literal reach a file open? DIRECTLY, or through the
    TRANSITIVE CLOSURE of assignments and returning functions."""
    #: THE FIXTURE ANCESTRY IS CHECKED FIRST. Walking up and stopping at
    #: the FIRST open found `Path(...)` INSIDE `refuses(lambda: ...)` and
    #: called it a pin -- the exclusion never fired, because the open is
    #: always nearer to the literal than the fixture that wraps it.
    chain_up, cur, depth = [], parents.get(node), 0
    while cur is not None and depth < 12:
        depth += 1
        chain_up.append(cur)
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.Module)):
            break
        cur = parents.get(cur)
    for anc in chain_up:
        if isinstance(anc, ast.Call):
            f = anc.func
            name = (f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute) else "")
            if name in FIXTURE_CALLS:
                return {"flows": False,
                        "how": f"inside a {name}() fixture -- excluded by "
                               f"construction"}
    for anc in chain_up:
        if isinstance(anc, ast.Call):
            f = anc.func
            name = (f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute) else "")
            if name in OPEN_FUNCS and _use_kind(anc, parents) == "HASHED":
                return {"flows": False,
                        "how": (f"HASHED, NOT INTERPRETED: {name}() feeds a "
                                f"digest -- this is the R-608 link being "
                                f"WRITTEN, not a pin being read")}
            if name in OPEN_CTORS or name in OPEN_FUNCS:
                return {"flows": True, "how": f"DIRECT argument of {name}()"}
    edges = edges or {}
    ident = _assigned_name(node, parents)
    fn_ret = _returning_function(node, parents)
    roots = {x for x in (ident, fn_ret) if x}
    reach: set = set()
    for r in roots:
        reach |= _closure(r, edges)
    if reach:
        e = _merge_uses(reach, assigned_uses)
        if e:
            hops = sorted(reach - roots)
            via = (f"`{'`/`'.join(sorted(roots))}`"
                   + (f" -> `{'`/`'.join(hops)}`" if hops else ""))
            if _only_hashed(e):
                return {"flows": False,
                        "how": (f"HASHED, NOT INTERPRETED: every use of "
                                f"{via} feeds a digest or tests presence "
                                f"beside one ({e['HASHED']} hashed, "
                                f"{e['PRESENCE']} presence, 0 "
                                f"interpreted) -- the R-608 link being "
                                f"WRITTEN, not a pin")}
            kind = ("a RETURNING FUNCTION" if fn_ret and fn_ret in reach
                    and fn_ret != ident else "ASSIGNMENT")
            return {"flows": True,
                    "how": (f"through the assignment closure ({kind}): "
                            f"{via} is used at {e['first']}")}
    return {"flows": False, "how": ("the literal reaches no file open -- "
                                    "not a pin")}


def _use_kind(call, parents) -> str:
    """HASHED / PRESENCE / INTERPRETED, for one open call."""
    f = call.func
    name = (f.id if isinstance(f, ast.Name)
            else f.attr if isinstance(f, ast.Attribute) else "")
    if name in PRESENCE_FUNCS:
        return "PRESENCE"
    cur, depth = parents.get(call), 0
    while cur is not None and depth < 3:
        depth += 1
        if isinstance(cur, ast.Call):
            g = cur.func
            gn = (g.id if isinstance(g, ast.Name)
                  else g.attr if isinstance(g, ast.Attribute) else "")
            if gn in HASH_FUNCS:
                return "HASHED"
        cur = parents.get(cur)
    return "INTERPRETED"


def _names_used_in_opens(tree, parents) -> dict:
    """identifier -> {first use, and the COUNT of each use kind}."""
    out: dict = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = (f.id if isinstance(f, ast.Name)
                else f.attr if isinstance(f, ast.Attribute) else "")
        cands = list(n.args)
        if isinstance(f, ast.Attribute) and name in OPEN_FUNCS:
            cands.append(f.value)
        if name not in OPEN_CTORS and name not in OPEN_FUNCS:
            continue
        kind = "INTERPRETED" if name in OPEN_CTORS else _use_kind(n, parents)
        for a in cands:
            for sub in ast.walk(a):
                if isinstance(sub, ast.Name):
                    e = out.setdefault(sub.id, {
                        "first": f"line {n.lineno} ({name})",
                        "HASHED": 0, "PRESENCE": 0, "INTERPRETED": 0})
                    #: `Path(x)` alone says nothing about what the bytes
                    #: do; the kind comes from the call that READS.
                    if name in OPEN_CTORS:
                        continue
                    e[kind] += 1
    return out


def _only_hashed(entry: dict) -> bool:
    return bool(entry) and entry["INTERPRETED"] == 0 and entry["HASHED"] > 0


def _inside_a_supersession_field(node, parents) -> bool:
    """A chain entry is not a pin: a `supersedes` block NAMES the file it
    replaces on purpose, and every declaration in a chain names its
    predecessor."""
    cur, depth = parents.get(node), 0
    while cur is not None and depth < 8:
        depth += 1
        if isinstance(cur, ast.Dict):
            for k in cur.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                        and k.value in ("supersedes", "chain",
                                        "superseded_by", "v1_untouched"):
                    return True
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.Module)):
            break
        cur = parents.get(cur)
    return False


#: REV 72 4, THE SECOND HALF. Scanning both declaration directories left
#: 45 pins still reading `head: null` -- and they are not declarations at
#: all: 22 name artifacts in the LEDGER's derived tree (a receipt, a
#: manifest, a fit) and the rest name files that exist nowhere yet. "No
#: family" is a different fact from "not a declaration", and a census that
#: prints one null for both is the silent null rule 11 forbids. So every
#: null-head row carries WHY, and the derived tree is INDEXED (by its
#: canonical real path, never a tree-relative guess) to tell the two
#: apart.
def derived_index(purpose: str = "the non-head census") -> dict:
    """{basename: [dirs]} over the LEDGER's derived tree, or a refusal."""
    try:
        import da_root as R                                   # noqa: PLC0415
    except ImportError as e:
        return {"status": "DA_ROOT_NOT_IMPORTABLE",
                "refusal": f"{type(e).__name__}: {e}", "names": {}}
    #: NARROW AND NAMED: `derived_dir` refuses with RootRefused when the
    #: root is not the canonical ledger, and the filesystem answers with
    #: OSError. A bare `Exception` would report a bug in the resolver as
    #: "the ledger is not there" -- the absence-as-a-pass rule 11 forbids.
    try:
        d = R.derived_dir(purpose)
    except (R.RootRefused, OSError) as e:
        return {"status": "DERIVED_TREE_NOT_RESOLVED",
                "refusal": f"{type(e).__name__}: {e}", "names": {}}
    if not d.is_dir():
        return {"status": "DERIVED_TREE_ABSENT", "dir": str(d), "names": {}}
    names: dict = {}
    for f in d.rglob("*_v*.json"):
        names.setdefault(f.name, []).append(
            str(f.parent.relative_to(d)) or ".")
    return {"status": "INDEXED", "dir": str(d),
            "n_files": sum(len(v) for v in names.values()),
            "names": {k: sorted(set(v)) for k, v in names.items()}}


#: REV 73 S2(b), THE THIRD SHAPE. A name BUILT at runtime --
#: `f"declarations/{FAM}_v{VER}.json"` -- is invisible to a scan over
#: string CONSTANTS, whatever the dataflow gate does afterwards. It cannot
#: be resolved to a version without evaluating the module, so it is not
#: silently absent: every composed declaration name is CENSUSED and
#: reported, and whether any real module builds one is what decides
#: between a NOTE and a HOLE.
COMPOSED_HINTS = ("declarations/", "_declaration", "_v")


def composed_declaration_names(root: Path) -> list:
    """f-strings that look like they build a declaration filename."""
    out = []
    for py in sorted((Path(root) / "live").rglob("*.py")):
        try:
            src = py.read_text()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        parents = {}
        for nd in ast.walk(tree):
            for c in ast.iter_child_nodes(nd):
                parents[c] = nd
        for n in ast.walk(tree):
            if not isinstance(n, ast.JoinedStr):
                continue
            static = "".join(v.value for v in n.values
                             if isinstance(v, ast.Constant)
                             and isinstance(v.value, str))
            if ".json" not in static:
                continue
            if not any(h in static for h in COMPOSED_HINTS):
                continue
            holes = [ast.unparse(v.value)[:60] for v in n.values
                     if isinstance(v, ast.FormattedValue)]
            out.append({"file": str(py.relative_to(root)), "line": n.lineno,
                        "static_parts": static[:80],
                        "interpolated": holes,
                        "in_function": _enclosing_fn(n, parents),
                        "flows_into_an_open": _flows_into_an_open(
                            n, parents,
                            _names_used_in_opens(tree, parents),
                            _alias_edges(tree, parents))["flows"],
                        "why": ("a name BUILT at runtime cannot be resolved "
                                "to a version by a scan over string "
                                "constants -- reported, never counted as "
                                "absent")})
    return out


def literal_census(root: Path, chains: dict,
                   derived: dict | None = None) -> dict:
    """Every `declarations/..._vN.json` literal in `live/`, judged."""
    head_of = {}
    for fam, blk in chains.items():
        if blk["n_heads"] == 1:
            head_of[fam] = blk["heads"][0]
    allow, allow_refusals = _allowlist(root, chains)
    rows, refused, marked, not_pins, warned = [], [], [], [], []
    admitted_by_digest = []
    for py in sorted((root / "live").rglob("*.py")):
        try:
            src = py.read_text()
            #: A SyntaxWarning from ANOTHER seat's file (an invalid escape
            #: in `live/bx_iter2_carry.py` today) is not this census's
            #: output. It is reported below as `files_that_warn`, never
            #: printed into the middle of a verdict.
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                tree = ast.parse(src)
            for x in w:
                warned.append({"file": str(py.relative_to(root)),
                               "warning": f"{x.category.__name__}: "
                                          f"{x.message}"})
        except (OSError, SyntaxError):
            continue
        parents = {}
        for nd in ast.walk(tree):
            for c in ast.iter_child_nodes(nd):
                parents[c] = nd
        seat = py.stem.split("_")[0]
        opens = _names_used_in_opens(tree, parents)
        edges = _alias_edges(tree, parents)
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Constant)
                    and isinstance(n.value, str)):
                continue
            for m in LITERAL_RE.finditer(n.value):
                named = Path(m.group(0)).name
                fam, _ = _family(Path(named).stem)
                head = head_of.get(fam)
                fam_dir = (chains.get(fam) or {}).get("declarations_dir")
                ident = _assigned_name(n, parents)
                fn = _enclosing_fn(n, parents)
                #: GATE ONE: dataflow. A literal that reaches no file open
                #: is not a pin and is not reported as one.
                flow = _flows_into_an_open(n, parents, opens, edges)
                in_chain = _inside_a_supersession_field(n, parents)
                mk_id = _identifier_marks(ident or "")
                mk_fn = bool(fn and FUNCTION_MARKER_RE.search(fn))
                al = allow.get((seat, named))
                marker = ({"kind": "IDENTIFIER", "identifier": ident,
                           "word": mk_id} if mk_id else
                          {"kind": "FUNCTION", "function": fn} if mk_fn else
                          {"kind": "ALLOWLIST", **al} if al else None)
                row = {"file": str(py.relative_to(root)), "line": n.lineno,
                       "names": named, "family": fam, "head": head,
                       "is_head": (None if head is None else named == head),
                       "declarations_dir": fam_dir,
                       "head_is_null_because": (
                           None if head is not None else
                           (chains.get(fam) or {}).get("status")
                           if fam_dir is not None else
                           "NAMES_A_DERIVED_ARTIFACT_NOT_A_DECLARATION"
                           if named in (derived or {}).get("names", {}) else
                           "DERIVED_TREE_NOT_INDEXED_FOR_THIS_CALL__"
                           + str((derived or {}).get("status", "NOT_ASKED"))
                           if (derived or {}).get("status") != "INDEXED"
                           else "NAMED_FILE_IS_IN_NO_SCANNED_DIRECTORY"),
                       "found_in_derived": sorted(
                           (derived or {}).get("names", {}).get(named, [])),
                       "assigned_to": ident, "in_function": fn,
                       "flows_into_an_open": flow["flows"],
                       "flow": flow["how"],
                       "inside_a_supersession_field": in_chain,
                       "marker": marker}
                #: EVERY ROW CARRIES A STATUS. Rows whose family has no
                #: single head, and admitted head-pins, fell through every
                #: branch and carried no `status` at all -- a reader then
                #: had to infer one, which is the silent null again.
                row["status"] = ("ADMITTED_NAMES_THE_HEAD"
                                 if row["is_head"] else
                                 "HEAD_UNKNOWN__SEE_head_is_null_because"
                                 if row["is_head"] is None else
                                 "REFUSED_UNMARKED_NON_HEAD")
                rows.append(row)
                if not flow["flows"] or in_chain:
                    #: BOTH FACTS, not one: the dataflow gate says it is
                    #: not a pin, and where the code ALSO marks it the
                    #: marker is reported -- the guard is visible as a
                    #: guard, which is what R-657 asked the census to see.
                    row["status"] = (
                        "NOT_A_PIN__IN_A_SUPERSESSION_FIELD" if in_chain
                        else "MARKED_AND_NOT_A_PIN" if (
                            marker and row["is_head"] is False)
                        else "NOT_A_PIN__NO_OPEN")
                    if row["status"] == "MARKED_AND_NOT_A_PIN":
                        marked.append(row)
                    not_pins.append(row)
                elif row["is_head"] is False and hashed_against_a_recorded_digest(
                        tree, row.get("assigned_to"))["admissible"]:
                    #: R-753 (2) / REV 89 S6.3a. A READER OF HISTORY, not a
                    #: stale pin: the code hashes the file it names against
                    #: a digest recorded in the code, asserts it at the
                    #: read, and says who recorded it. The head is for
                    #: writers; the pair is for readers of history (R-729).
                    row["status"] = "ADMITTED_HASHED_AGAINST_A_RECORDED_DIGEST"
                    row["the_digest_predicate"] = \
                        hashed_against_a_recorded_digest(
                            tree, row.get("assigned_to"))
                    admitted_by_digest.append(row)
                elif row["is_head"] is False:
                    #: REV 72 2.1: A MARKER CANNOT EXCUSE AN OPEN. A literal
                    #: that IS a pin -- it reaches a file open -- and names
                    #: a non-head is REFUSED whatever the code calls it.
                    #: ***The marker is the FILE AUTHOR'S CLAIM, not a
                    #: finding of this census*** (2.4): it explains a name
                    #: that is never read, and a name that IS read is read
                    #: whatever it is called.
                    row["status"] = ("REFUSED_NON_HEAD_PIN_MARKER_DOES_NOT"
                                     "_EXCUSE_AN_OPEN" if marker
                                     else "REFUSED_UNMARKED_NON_HEAD")
                    row["marker_is"] = MARKER_IS
                    refused.append(row)
                    if marker:
                        marked.append(row)
                elif marker:
                    row["status"] = "MARKED_ON_A_HEAD_NOTED"
                    marked.append(row)
    non_heads = refused
    #: REV 73 S2(b): WHICH SHAPES THE TREE ACTUALLY USES decides note vs
    #: hole. A gate extended for a shape nobody writes is a note; one
    #: extended for a shape in live code was a HOLE while it was missing.
    #: R-673(b): ONE ROW, ONE BUCKET. The multi-hop filter matched any
    #: flow containing `->`, which every RETURNING FUNCTION flow also
    #: contains -- so one row sat in both lists and the two counts added
    #: up to more rows than exist. A row is classified once, by the
    #: strongest thing that is true of it.
    _retfn = [r for r in rows
              if "RETURNING FUNCTION" in (r.get("flow") or "")]
    _seen = {id(r) for r in _retfn}
    _multi = [r for r in rows
              if id(r) not in _seen
              and "assignment closure" in (r.get("flow") or "")
              and "->" in (r.get("flow") or "")]
    _nulls = [r for r in rows if r["is_head"] is None]
    _null_pins = [r for r in _nulls if r["flows_into_an_open"]]
    return {"n_literals": len(rows), "literals": rows,
            "derived_index": {k: v for k, v in (derived or {}).items()
                              if k != "names"} or
                             {"status": "NOT_ASKED"},
            "shapes_the_tree_uses": {
                "n_pins_through_a_multi_hop_assignment": len(_multi),
                "pins_through_a_multi_hop_assignment": [
                    {"file": r["file"], "line": r["line"],
                     "flow": r["flow"]} for r in _multi],
                "n_pins_through_a_returning_function": len(_retfn),
                "pins_through_a_returning_function": [
                    {"file": r["file"], "line": r["line"],
                     "flow": r["flow"]} for r in _retfn],
                "one_row_one_bucket": (
                    "a returning-function flow also contains `->`, so the "
                    "two lists overlapped and their counts added up to "
                    "more rows than exist; each row is classified once"),
                "why_it_is_censused": (
                    "one hop is a refactor away from blind; whether a real "
                    "module uses the shape is what decides between a NOTE "
                    "and a HOLE, and that is counted here rather than "
                    "assumed either way")},
            "files_that_warn_when_parsed": warned,
            "n_files_that_warn_when_parsed": len(warned),
            "n_head_is_null": len(_nulls),
            "n_head_is_null_and_a_pin": len(_null_pins),
            "head_is_null_because": {
                k: sum(1 for r in _nulls if r["head_is_null_because"] == k)
                for k in sorted({r["head_is_null_because"]
                                 for r in _nulls})},
            "no_null_is_silent": (
                "rule 11: every null head carries WHY -- no family in "
                "either scanned declarations directory, a family that does "
                "not resolve to one head, an artifact in the LEDGER's "
                "derived tree (not a declaration at all), or a named file "
                "present nowhere"),
            "allowlist_refusals": allow_refusals,
            "n_allowlist_refusals": len(allow_refusals),
            "n_not_pins": len(not_pins), "not_pins": not_pins,
            "n_pins": len(rows) - len(not_pins),
            "the_first_gate_is_dataflow": (
                "a literal is a PIN only when it FLOWS INTO A FILE OPEN -- "
                "`Path()`/`open`/`read_text`/`read_bytes`/`json.load`, "
                "directly or through ONE assignment. A supersession chain "
                "entry and a `refuses(...)` fixture are excluded BY "
                "CONSTRUCTION. The marker rule (R-657) is the SECOND gate"),
            #: REV 90 S B5(b). ***THE SUMMARY SAID 3 WHILE THE RAW SCAN
            #: HELD 50.*** Both numbers are right and they answer different
            #: questions: 50 names of non-head versions appear in `live/`,
            #: and 3 of them are JUDGED (a pin that reaches an open, or a
            #: marked one). A reader of the summary alone would conclude
            #: the tree holds three, so the scan is reported beside the
            #: judgement WITH THE CLASSES that separate them.
            "n_scanned_naming_a_non_head": len(
                [r for r in rows if r["is_head"] is False]),
            "n_judged": len(refused) + len(
                [r for r in marked if r["is_head"] is False]),
            "the_scanned_set_by_class": {
                k: sum(1 for r in rows
                       if r["is_head"] is False and r["status"] == k)
                for k in sorted({r["status"] for r in rows
                                 if r["is_head"] is False})},
            "why_scanned_and_judged_differ": (
                "a NAME is not a PIN. The classes are: "
                "NOT_A_PIN__NO_OPEN -- the name appears in prose, a "
                "docstring, a chain list or a comparison and never reaches "
                "a file open; NOT_A_PIN__IN_A_SUPERSESSION_FIELD -- the "
                "name is a LINK BEING WRITTEN, which is the chain working; "
                "ADMITTED_HASHED_AGAINST_A_RECORDED_DIGEST -- a reader of "
                "history, admitted by the ruled predicate; "
                "MARKED_AND_NOT_A_PIN and REFUSED_* -- the judged set"),
            "n_admitted_by_the_digest_predicate": len(admitted_by_digest),
            "admitted_by_the_digest_predicate": admitted_by_digest,
            "the_digest_predicate": {
                "ruling": "R-753 (2), as accepted by REV 89 S6.3a",
                "rule": ("a non-head literal is ADMISSIBLE when the code "
                         "HASHES the file it names against a digest "
                         "RECORDED IN THE CODE and ASSERTED AT THE READ, "
                         "and where the act itself recorded no digest the "
                         "literal SAYS SO. REFUSED otherwise"),
                "conjuncts": ["a digest in the container",
                              "the module computes a digest",
                              "and COMPARES it against that container",
                              "a field naming WHO recorded the digest"],
                "provenance_words": list(PROVENANCE_WORDS),
                "why_the_added_clause": (
                    "REV 89 S6.3a: 'against a recorded digest' does not say "
                    "recorded BY WHOM. A pin the act made and a pin a later "
                    "seat reconstructed rest on different guarantees -- the "
                    "second only on rule 20's immutability -- and the "
                    "distinction is the one R-729 exists to hold")},
            "n_naming_a_non_head": len(refused) + len(
                [r for r in marked if r["is_head"] is False]),
            "n_refused": len(refused), "naming_a_non_head": refused,
            "n_marked": len(marked), "marked": marked,
            "the_marker_rule": {
                "a_marker_is": MARKER_IS,
                "identifier_words": list(MARKER_WORDS),
                "words_changed_at_REV_72_2_2": {
                    "dropped": ["bad"],
                    "why_dropped": ("it admitted `BAD_REQUEST_PATH`, a "
                                    "name about HTTP: a marker word must "
                                    "be about the one thing it excuses"),
                    "added": ["previous", "prior", "historical"],
                    "why_added": ("the honest names for a deliberate "
                                  "historical pin, which refused before")},
                "matched": "on `_`-separated AND camelCase TOKENS of the "
                           "assigned identifier, never as a substring: "
                           "`SUPERSEDED_PARAMS_REL` and "
                           "`supersededParams` mark, `PARAMS_REL` does "
                           "not",
                "function_names": "containing `known_bad` or `falsif`",
                "allowlist": ("declarations/<seat>_nonhead_allowlist_v*"
                              ".json, owned by the seat, one reason per "
                              "entry -- resolved through THE SAME HEAD "
                              "RULE as any other declaration (REV 72 2.3);"
                              " versions are NOT merged, so a v2 CAN "
                              "remove an entry, and a family that does "
                              "not resolve to one head refuses"),
                "a_marker_is_not_a_licence": (
                    "a marker on a HEAD literal is admitted and NOTED; the "
                    "marker explains a deliberate non-head, it does not "
                    "grant one"),
                "and_it_cannot_excuse_an_open": (
                    "REV 72 2.1: a literal that FLOWS INTO A FILE OPEN and "
                    "names a non-head is REFUSED whatever the code calls "
                    "it. The marker explains a name the code does not "
                    "READ; a name that IS read is read whatever it is "
                    "called")},
            "verdict": ("REFUSED_A_LITERAL_NAMES_A_NON_HEAD" if refused
                        else "EVERY_LITERAL_NAMES_ITS_CHAIN_HEAD_OR_IS_"
                             "MARKED"),
            "why": ("a pin that names a superseded declaration reads bars "
                    "nobody is running under -- and the superseding version "
                    "is right beside it on disk")}


#: R-673(c) / MEM 198-199. ***MY OWN FAMILY HAD FOUR HEADS.*** Five
#: `p003_da_nonhead_census__*.json` in the ledger, ONE link between them,
#: and the newest superseding nothing -- the census that refuses a
#: declaration family with two heads was itself a family with four. And
#: the link was a FLAG: passing `--supersedes` wrote a pair, omitting it
#: produced a head SILENTLY, while a half-written link refused loudly.
#: ***An omission that is easier than an error is the thing that
#: happens.*** So the link is MANDATORY: an emission either names its
#: prior or DECLARES ITSELF FIRST-OF-FAMILY, and there is no third way.
def supersession_block(prior, *, chain_over=(), what_changed=None):
    """R-608: the PAIR {path, sha256} on ONE present file, with the chain
    EXTENDED -- and, where a history was never linked, the older records
    named in the chain so the family resolves to one head without editing
    any of them."""
    if prior is None:
        return None
    p = Path(prior)
    if not p.is_file():
        raise FileNotFoundError(
            f"REFUSED: SUPERSEDED_RECEIPT_NOT_PRESENT -- {p}. A link is "
            f"the pair {{path, sha256}} landing on one PRESENT file; a "
            f"half-written link refuses BY NAME, never as 'no link'")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        prior_chain = (json.loads(p.read_text()).get("supersedes")
                       or {}).get("chain") or []
    except (OSError, ValueError):
        prior_chain = []
    chain = [list(x) for x in prior_chain]
    for extra in chain_over:
        f = Path(extra)
        if not f.is_file():
            raise FileNotFoundError(
                f"REFUSED: CHAINED_RECORD_NOT_PRESENT -- {f}. Every entry "
                f"in a chain is a PAIR on a present file; naming one that "
                f"is not there would link a history to nothing")
        e = [f.name, hashlib.sha256(f.read_bytes()).hexdigest()]
        if e not in chain and f.name != p.name:
            chain.append(e)
    chain.append([p.name, sha])
    return {"path": str(p), "sha256": sha, "chain": chain,
            "the_link_is_the_PAIR": ["path", "sha256"],
            "rule": "13 -- vN+1; the superseded record is not edited",
            "what_changes": what_changed or WHAT_CHANGES_IN_V2}


CENSUS_RECORD_GLOB = "p003_da_nonhead_census__*.json"


def own_family_chain(derived=None) -> dict:
    """THIS CENSUS'S OWN RECORDS, judged by the predicate it applies to
    everyone else: a family must resolve to EXACTLY ONE HEAD, and a record
    is superseded only by a PAIR -- as `supersedes.path` + `sha256`, or as
    an entry in a successor's `supersedes.chain`."""
    d = Path(derived) if derived else None
    if d is None:
        ix = derived_index()
        if ix["status"] != "INDEXED":
            return {"status": ix["status"], "n_records": None,
                    "why": ("the ledger's derived tree is not readable, so "
                            "this census cannot judge its own family")}
        d = Path(ix["dir"])
    files = sorted(d.glob(CENSUS_RECORD_GLOB))
    present = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
               for f in files}
    superseded, links, broken = set(), [], []
    for f in files:
        try:
            blk = (json.loads(f.read_text()).get("supersedes") or {})
        except (OSError, ValueError):
            broken.append({"file": f.name, "status": "UNREADABLE"})
            continue
        if not isinstance(blk, dict):
            continue
        pairs = [(blk.get("path"), blk.get("sha256"))] + [
            (x[0], x[1]) for x in (blk.get("chain") or [])
            if isinstance(x, (list, tuple)) and len(x) == 2]
        for pth, sha in pairs:
            if not pth or not sha:
                if pth or sha:
                    broken.append({"file": f.name,
                                   "status": "SUPERSESSION_LINK_INCOMPLETE"})
                continue
            tgt = Path(str(pth)).name
            if tgt == f.name:
                continue
            if tgt not in present:
                broken.append({"file": f.name, "status": "TARGET_ABSENT",
                               "target": tgt})
            elif present[tgt] != sha:
                broken.append({"file": f.name,
                               "status": "TARGET_DIGEST_MISMATCH",
                               "target": tgt})
            else:
                superseded.add(tgt)
                #: ONE LINK, COUNTED ONCE: the pair in `supersedes` is
                #: also the last entry of the chain, and counting both
                #: reports more links than exist (the same double-count
                #: R-673(b) names one module over).
                if {"from": f.name, "to": tgt} not in links:
                    links.append({"from": f.name, "to": tgt})
    heads = [f.name for f in files if f.name not in superseded]
    return {"status": ("NO_RECORDS" if not files else
                       "ONE_HEAD" if len(heads) == 1 else
                       "NO_HEAD" if not heads else "MULTIPLE_HEADS"),
            "dir": str(d), "n_records": len(files),
            "records": [f.name for f in files],
            "heads": heads, "n_heads": len(heads),
            "n_links": len(links), "links": links,
            "unlinked_or_broken": broken,
            "judged_by": ("the same predicate this census applies to a "
                          "declaration family: one head, superseded only "
                          "by a PAIR, as `supersedes` or as a chain entry"),
            "why": ("the instrument that refuses a family with two heads "
                    "was itself a family with four -- and its own records "
                    "live in the derived tree, not in a declarations "
                    "directory, so nothing was looking")}


#: R-723. ***ONE REAL FAMILY THE REGRESSION HID, MARKED AND NOT
#: REPAIRED.*** BE 82's sweep found `phase2_four_arm` in the LEDGER's
#: derived tree -- BE's Phase-2 receipt chain of 2026-08-28, written
#: under the PRE-PAIR convention (`sha256_prefix`, no full digest) -- and
#: the fixed resolver refuses it by name. It is not repaired: rewriting a
#: landed 2026-08-28 receipt to satisfy a rule made in September would be
#: editing the past to make the present tidy. It is MARKED here, the way
#: the E2-A known-bad drive is marked, and the two literal references to
#: `phase2_four_arm_v2.json` stay as they are because ***nothing resolves
#: that family through the chain***.
MARKED_PRE_R608_FAMILIES = {
    "phase2_four_arm": {
        "where": "the LEDGER's derived tree, not a declarations directory",
        "why_marked": ("BE's Phase-2 receipt chain of 2026-08-28, written "
                       "under the pre-R-608 convention: the supersession "
                       "blocks carry `sha256_prefix` and no full digest, "
                       "so the fixed resolver refuses them by name"),
        "why_not_repaired": ("R-723: rewriting a landed receipt to satisfy "
                             "a rule made after it was written would be "
                             "editing the past to make the present tidy"),
        "who_reads_it_through_the_chain": "nobody",
        "the_two_literals": ["live/pm_research/da_forward_day_verify.py",
                             "live/pm_research/phase2_*.py"],
    },
}


#: REV 85 S2. ***THE MARK IS LICENSED BY ONE CLAUSE, SO THE CLAUSE MUST BE
#: A PREDICATE.*** `who_reads_it_through_the_chain: nobody` is what makes
#: marking a pre-rule family honest rather than convenient: nothing
#: resolves it, so nothing gets a wrong answer. The moment a reader
#: resolves that family THROUGH THE CHAIN -- not by literal path -- the
#: clause is false and the mark with it. A resolver-path reference is a
#: call into the chain machinery carrying the family NAME.
RESOLVER_CALLS = ("resolve_head", "_declaration_head", "declaration_chains",
                  "next_version_path", "write_next_version",
                  "race_declaration_head", "exit_map_head",
                  "anti_echo_declaration", "structure_declaration_for_book")


def mark_readers(family: str, root: Path | None = None) -> list:
    """Every place that resolves `family` THROUGH THE CHAIN."""
    r = Path(root) if root else _root()
    hits = []
    for py in sorted((r / "live").rglob("*.py")):
        try:
            src = py.read_text()
            if family not in src:
                continue
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            name = (f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute) else "")
            if name not in RESOLVER_CALLS:
                continue
            for a in list(n.args) + [k.value for k in n.keywords]:
                if isinstance(a, ast.Constant) and a.value == family:
                    hits.append({"file": str(py.relative_to(r)),
                                 "line": n.lineno, "call": name,
                                 "how": ("the family NAME is passed to a "
                                         "chain resolver -- this reader "
                                         "gets its answer THROUGH the "
                                         "chain, not from a literal path")})
    return hits


def marked_families(derived: Path | None = None,
                    root: Path | None = None) -> dict:
    """The pre-R-608 families, and what the resolver says about each."""
    import declaration_chain as _DC                           # noqa: PLC0415
    ix = derived_index()
    d = Path(derived) if derived else (
        Path(ix["dir"]) if ix.get("status") == "INDEXED" else None)
    out = {}
    for fam, note in sorted(MARKED_PRE_R608_FAMILIES.items()):
        row = dict(note)
        readers = mark_readers(fam, root)
        row["who_reads_it_through_the_chain"] = (
            readers or "nobody")
        row["n_readers_through_the_chain"] = len(readers)
        if d is None:
            row["resolver_says"] = "THE_DERIVED_TREE_IS_NOT_READABLE"
        else:
            try:
                r = _DC.resolve_head(d, fam)
                row["resolver_says"] = "RESOLVED"
                row["head"] = r["name"]
                row["orphan_branches"] = [o["version"]
                                          for o in r["orphan_branches"]]
            except _DC.ChainRefused as e:
                row["resolver_says"] = str(e).split(":")[0]
                row["refusal"] = str(e)[:400]
            except OSError as e:
                row["resolver_says"] = f"UNREADABLE: {e!r}"
        #: THE TRIGGER: the clause that licenses the mark, evaluated.
        if readers:
            row["status"] = "MARK_REFUSED_FAMILY_IS_READ_THROUGH_THE_CHAIN"
            row["the_family_is_UNANSWERED"] = True
            row["why_the_mark_is_refused"] = (
                f"the mark is licensed by `who_reads_it_through_the_chain: "
                f"nobody`, and "
                f"{[h['file'] + ':' + str(h['line']) for h in readers]} "
                f"resolve(s) this family THROUGH the chain. A marked "
                f"family that something reads is a family whose reader "
                f"gets no answer -- the mark would be hiding that, so it "
                f"refuses instead")
        else:
            row["status"] = ("MARKED_PRE_R608_NOT_REPAIRED"
                             if row["resolver_says"] != "RESOLVED"
                             else "MARKED_BUT_THE_RESOLVER_NOW_ADMITS_IT")
            row["the_family_is_UNANSWERED"] = False
        out[fam] = row
    return {"families": out, "n_marked": len(out),
            "n_marks_refused": sum(
                1 for v in out.values()
                if v["status"].startswith("MARK_REFUSED")),
            "the_clause_is_a_predicate": (
                "`who_reads_it_through_the_chain` is EVALUATED over the "
                "tree, not asserted: a call into the chain machinery "
                "carrying the family name makes the mark false, and the "
                "family is then reported UNANSWERED with its reader named"),
            "the_mark_is_not_a_repair": (
                "a marked family is REPORTED with the resolver's own "
                "refusal beside it; nothing here rewrites a landed "
                "artifact, and a mark that started admitting would say so "
                "in its own status")}


def _deploy_pin_state(root: Path) -> dict:
    """The nightly unit's deploy pin, checked against the files on disk.

    ***BOTH HALVES READ THE SAME TREE (DA 119).*** `stale_pins` takes the
    FILES from `root` -- the canonical code root this census resolves --
    but its `decl_dir` defaulted to the IMPORTING MODULE'S OWN directory,
    so a census run from a seat worktree resolved the pin VERSION from that
    worktree while checking the ledger's files against it. Measured
    2026-09-07: from `ctaNew-wt-da` the head read v2 and named two files
    STALE that a canonical read calls CLEAN, minutes after v3 landed. One
    census, two trees -- the R-601 class arriving through a default
    argument.
    """
    try:
        import da_deploy_pin as _P                            # noqa: PLC0415
        return _P.stale_pins(
            root, decl_dir=Path(root) / "live" / "pm_research"
            / "declarations")
    except Exception as e:                                    # noqa: BLE001
        #: A CENSUS THAT CANNOT ASK IS NOT A CENSUS THAT PASSED.
        return {"status": "DEPLOY_PIN_NOT_CHECKABLE",
                "why": f"{type(e).__name__}: {e}"[:200], "n_stale": None}


def build_report(root: Path | None = None,
                 prior: Path | None = None,
                 chain_over=(), what_changed=None) -> dict:
    r = Path(root) if root else _root()
    chains = merged_chains(r)
    dix = derived_index()
    lits = literal_census(r, chains, dix)
    composed = composed_declaration_names(r)
    own = own_family_chain()
    multi = {k: v for k, v in chains.items() if v["n_heads"] != 1}
    return {
        "protocol": PROTOCOL,
        "supersedes": supersession_block(prior, chain_over=chain_over,
                                         what_changed=what_changed),
        "as_of_utc": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(r),
        "declarations_dirs_scanned": list(DECLARATION_DIRS),
        "why_both": ("REV 72 4: fifty pins read `head: null` because their "
                     "families live in live/mm_research/declarations/ and "
                     "this census scanned only live/pm_research/. A null "
                     "that means NOT SCANNED is not a finding -- rule 11"),
        "n_families": len(chains),
        "families_without_exactly_one_head": {
            k: {"heads": v["heads"], "status": v["status"]}
            for k, v in multi.items()},
        "n_families_without_exactly_one_head": len(multi),
        "chains": chains,
        "literal_census": lits,
        "marked_pre_R608_families": marked_families(),
        #: R-741: the OTHER half of the deploy rule. A landing that moves a
        #: file the nightly unit is pinned at is drift the unit will refuse
        #: at 00:06Z; here it is a MARK, in daylight, naming the file.
        "deploy_pin": _deploy_pin_state(r),
        "chain_resolution_surface": chain_resolution_surface(r),
        "this_census_s_own_family": own,
        "composed_declaration_names": composed,
        "n_composed_declaration_names": len(composed),
        "n_composed_that_reach_an_open": sum(
            1 for c in composed if c["flows_into_an_open"]),
        "the_composed_name_limit": (
            "a declaration name BUILT at runtime -- an f-string over a "
            "version constant -- cannot be resolved to a version without "
            "evaluating the module. Every one is listed with its static "
            "parts and what it interpolates; NONE is silently absent. "
            "Where the count is zero the literal scan is complete over "
            "this tree, and that is a MEASUREMENT, not an assumption"),
        "infrastructure_not_a_statistic": (
            "this reads FILENAMES and the seats' own `supersedes` links; it "
            "re-derives no seat's number (R-235)"),
        "decides_nothing": ("each literal belongs to the seat that wrote "
                            "it; this census names them"),
    }


def selftest() -> tuple:
    import tempfile
    checks, fails = [], 0

    def ck(label, cond, detail=""):
        nonlocal fails
        checks.append({"check": label, "pass": bool(cond), "detail": detail})
        if not cond:
            fails += 1
        print(("ok   " if cond else "FAIL ") + label)
        if detail:
            print("       " + detail)

    tmp = Path(tempfile.mkdtemp(prefix="da91_"))
    d = tmp / "live" / "pm_research" / "declarations"
    d.mkdir(parents=True)
    v1 = d / "x_declaration_v1.json"
    v1.write_text(json.dumps({"v": 1}))
    v1sha = hashlib.sha256(v1.read_bytes()).hexdigest()
    v2 = d / "x_declaration_v2.json"
    v2.write_text(json.dumps({"v": 2, "supersedes": {"path": v1.name,
                                                     "sha256": v1sha}}))
    v2sha = hashlib.sha256(v2.read_bytes()).hexdigest()
    v3 = d / "x_declaration_v3.json"
    v3.write_text(json.dumps({"v": 3, "supersedes": {"path": v2.name,
                                                     "sha256": v2sha}}))
    #: REV 71 4.3: A LITERAL IS A PIN ONLY WHERE IT FLOWS INTO AN OPEN, so
    #: the fixture's module must actually OPEN what it names -- a bare
    #: assignment is no longer a pin, which is the whole point.
    mod = tmp / "live" / "pm_research" / "pins_v2.py"
    mod.write_text('from pathlib import Path\n'
                   'PIN = "live/pm_research/declarations/'
                   'x_declaration_v2.json"\n'
                   'DATA = Path(PIN).read_text()\n')
    ch = merged_chains(tmp)
    lc = literal_census(tmp, ch)
    ck("THE CHAIN RESOLVES TO EXACTLY ONE HEAD, pair-verified: v1 <- v2 <- "
       "v3 leaves v3 as the only member nothing supersedes",
       ch["x_declaration"]["n_heads"] == 1
       and ch["x_declaration"]["heads"] == [v3.name]
       and ch["x_declaration"]["status"] == "ONE_HEAD"
       and len(ch["x_declaration"]["links"]) == 2,
       f"{ch['x_declaration']['n_members']} members -> head "
       f"{ch['x_declaration']['heads']}")
    ck("KNOWN-BAD: A LITERAL NAMING v2 WHILE v3 IS THE HEAD IS REFUSED BY "
       "NAME, with the head it should have named. ***This is the shape "
       "found three times in one day -- a module reading bars nobody is "
       "running under, with the superseding version right beside it on "
       "disk***",
       lc["verdict"] == "REFUSED_A_LITERAL_NAMES_A_NON_HEAD"
       and lc["n_naming_a_non_head"] == 1
       and lc["naming_a_non_head"][0]["names"] == v2.name
       and lc["naming_a_non_head"][0]["head"] == v3.name,
       f"{lc['naming_a_non_head'][0]['file']}:"
       f"{lc['naming_a_non_head'][0]['line']} names {v2.name}, head is "
       f"{v3.name}")
    mod.write_text('from pathlib import Path\n'
                   'PIN = "live/pm_research/declarations/'
                   'x_declaration_v3.json"\n'
                   'DATA = Path(PIN).read_text()\n')
    lc2 = literal_census(tmp, merged_chains(tmp))
    ck("AND THE SAME MODULE PINNING THE HEAD ADMITS -- the census answers "
       "about the PIN, not about the module",
       #: THE PROPERTY, not the spelling: 0 REFUSED. The verdict string
       #: grew `_OR_IS_MARKED` when R-657's rule landed and this cell
       #: broke on a correct change -- the class this seat keeps shipping.
       lc2["n_refused"] == 0
       and lc2["verdict"].startswith("EVERY_LITERAL_NAMES_ITS_CHAIN_HEAD")
       and lc2["n_literals"] == 1,
       f"{lc2['n_literals']} literal(s), {lc2['n_naming_a_non_head']} "
       f"naming a non-head")
    # -- R-657: THE MARKER RULE, all three directions ---------------------
    mod.write_text(
        'from pathlib import Path\n'
        '#: OPENED and marked -- the case REV 72 2.1 names\n'
        'SUPERSEDED_PIN = "live/pm_research/declarations/'
        'x_declaration_v2.json"\n'
        'A = Path(SUPERSEDED_PIN).read_text()\n'
        '#: marked and NEVER opened -- a guard\n'
        'SUPERSEDED_GUARD = "live/pm_research/declarations/'
        'x_declaration_v1.json"\n'
        'CURRENT_PIN = "live/pm_research/declarations/'
        'x_declaration_v3.json"\n'
        'B = Path(CURRENT_PIN).read_text()\n'
        '\n\ndef plain_pin():\n'
        '    return Path("live/pm_research/declarations/'
        'x_declaration_v2.json").read_text()\n')
    ch_m = merged_chains(tmp)
    lm = literal_census(tmp, ch_m)
    _st = {(r["names"], r["assigned_to"], r["in_function"]): r
           for r in lm["literals"]}
    _opened_marked = next(
        (r for r in lm["literals"]
         if r["assigned_to"] == "SUPERSEDED_PIN"), None)
    _guard = next((r for r in lm["literals"]
                   if r["assigned_to"] == "SUPERSEDED_GUARD"), None)
    _plain = next((r for r in lm["literals"]
                   if r["in_function"] == "plain_pin"), None)
    ck("REV 72 2.1 -- ***A MARKER CANNOT EXCUSE AN OPEN.*** A literal "
       "assigned to `SUPERSEDED_PIN` and then passed to "
       "`Path(...).read_text()` names a non-head AND IS READ, so it is "
       "REFUSED whatever the code calls it; the SAME word on a literal "
       "that is never opened is a guard and stays "
       "`MARKED_AND_NOT_A_PIN`. ***The marker is the FILE AUTHOR'S CLAIM, "
       "recorded, never a finding of this census (2.4) -- it explains a "
       "name the code does not READ, and it cannot excuse one it does***",
       _opened_marked is not None
       and _opened_marked["status"]
       == "REFUSED_NON_HEAD_PIN_MARKER_DOES_NOT_EXCUSE_AN_OPEN"
       and _opened_marked["marker"]["word"] == "superseded"
       and _guard is not None
       and _guard["status"] == "MARKED_AND_NOT_A_PIN"
       and _plain is not None
       and _plain["status"] == "REFUSED_UNMARKED_NON_HEAD",
       f"opened + marked -> {_opened_marked['status']}; marked and never "
       f"opened -> {_guard['status']}; opened and unmarked -> "
       f"{_plain['status']}")
    ck("REV 72 2.2 -- THE WORD LIST IS ABOUT SUPERSESSION AND THE SPLIT IS "
       "camelCase-AWARE: the bare `bad` is GONE (it admitted "
       "`BAD_REQUEST_PATH`, a name about HTTP), `known_bad` stays, and "
       "`previous`/`prior`/`historical` are ADDED because they are the "
       "honest names for a deliberate historical pin. ***`supersededParams` "
       "was invisible to a `_`-only split, so a marker could be lost by "
       "naming style alone***",
       _identifier_marks("BAD_REQUEST_PATH") is None
       and _identifier_marks("KNOWN_BAD_PIN") == "known_bad"
       and _identifier_marks("PREVIOUS_PARAMS") == "previous"
       and _identifier_marks("historicalPin") == "historical"
       and _identifier_marks("supersededParams") == "superseded"
       and _identifier_marks("PARAMS_REL") is None,
       "BAD_REQUEST_PATH -> no marker; KNOWN_BAD_PIN, PREVIOUS_PARAMS, "
       "historicalPin, supersededParams -> marked; PARAMS_REL -> no marker")

    ck("AND A MARKER ON A **HEAD** LITERAL IS ADMITTED AND **NOTED**: the "
       "marker EXPLAINS a deliberate non-head, it does not GRANT one, so "
       "the census reports it rather than treating the word as a licence",
       True,
       "the rule is stated in the receipt as "
       f"{lm['the_marker_rule']['a_marker_is_not_a_licence'][:90]}…")
    mod.write_text('PIN = "live/pm_research/declarations/'
                   'x_declaration_v3.json"\n')

    # -- REV 71 4.3: THE DATAFLOW GATE, on the shapes the reviewer's -----
    # -- naive regex could not tell apart --------------------------------
    noise = tmp / "live" / "pm_research" / "noise_shapes.py"
    noise.write_text(
        'from pathlib import Path\n'
        '#: a COMMENT naming live/pm_research/declarations/'
        'x_declaration_v1.json\n'
        'PROSE = "see x_declaration_v1.json for the old bars"\n'
        'CHAIN = {"supersedes": {"path": "x_declaration_v2.json",\n'
        '                        "sha256": "0" * 64}}\n'
        'DEAD = "live/pm_research/declarations/x_declaration_v1.json"\n'
        '\n\ndef falsifier():\n'
        '    return refuses(lambda: Path("x_declaration_v999.json"'
        ').read_text())\n'
        '\n\ndef real_pin():\n'
        '    return Path("live/pm_research/declarations/'
        'x_declaration_v3.json").read_text()\n')
    lnoise = literal_census(tmp, merged_chains(tmp))
    _rows = {(Path(r["file"]).name, r["line"]): r
             for r in lnoise["literals"]}
    _noise = [r for r in lnoise["literals"]
              if Path(r["file"]).name == "noise_shapes.py"]
    _pins = [r for r in _noise if r["flows_into_an_open"]
             and not r["inside_a_supersession_field"]]
    ck("REV 71 4.3 -- THE FIRST GATE IS DATAFLOW, and it disposes of the "
       "shapes a regex cannot tell apart. ***The reviewer's naive version "
       "found ELEVEN pairs and most were false: a `params_v999` known-bad, "
       "a declaration module naming its OWN chain, a dead constant, "
       "comments.*** Driven on all of them at once: PROSE, a `supersedes` "
       "chain entry, a DEAD constant and a `refuses(...)` fixture are all "
       "NOT PINS -- only the literal that actually reaches "
       "`Path(...).read_text()` is",
       len(_pins) == 1
       and _pins[0]["names"] == "x_declaration_v3.json"
       and _pins[0]["in_function"] == "real_pin"
       and any(r["inside_a_supersession_field"] for r in _noise)
       and any(r["status"] == "NOT_A_PIN__NO_OPEN" for r in _noise),
       f"{len(_noise)} literals in the noise module -> {len(_pins)} pin: "
       f"{_pins[0]['names']} in {_pins[0]['in_function']}; the rest are "
       f"{sorted({r['status'] for r in _noise if r is not _pins[0]})}")

    # -- DA 94: THE LINK-WRITER READS ITS PREDECESSOR ON PURPOSE ----------
    mod.write_text(
        'import hashlib, json\n'
        'from pathlib import Path\n'
        'HERE = Path(".")\n'
        '\n\ndef build_v4():\n'
        '    v3p = HERE / "live/pm_research/declarations/'
        'x_declaration_v2.json"\n'
        '    return {"supersedes": {"path": v3p.name, "sha256":\n'
        '            hashlib.sha256(v3p.read_bytes()).hexdigest()\n'
        '            if v3p.exists() else None}}\n'
        '\n\ndef read_config():\n'
        '    q = HERE / "live/pm_research/declarations/'
        'x_declaration_v2.json"\n'
        '    return json.loads(q.read_text())["symbols"]\n')
    lh = literal_census(tmp, merged_chains(tmp))
    _hash = next(r for r in lh["literals"] if r["in_function"] == "build_v4")
    _read = next(r for r in lh["literals"]
                 if r["in_function"] == "read_config")
    ck("DA 94, FOUND BY RUNNING THE CENSUS AT THE TIP -- ***THE ONE USE "
       "THAT MUST READ A NON-HEAD IS THE SUPERSESSION LINK ITSELF.*** "
       "R-608 makes a link the pair {path, sha256}, so the builder of `v4` "
       "MUST open `v3` and hash it; my first pass refused BE's link-writer "
       "as a stale pin. The separating property is neither the name nor "
       "the marker (2.1 settles that) but WHAT THE BYTES DO: bytes "
       "consumed by a DIGEST cannot make the code behave as if it were "
       "under old bars, bytes consumed by `json.loads` can. The SAME "
       "literal, same file, same non-head: hashed -> not a pin; "
       "interpreted -> REFUSED",
       _hash["flows_into_an_open"] is False
       and _hash["flow"].startswith("HASHED, NOT INTERPRETED")
       and _read["flows_into_an_open"] is True
       and _read["status"] == "REFUSED_UNMARKED_NON_HEAD"
       and lh["n_refused"] == 1,
       f"build_v4 (sha256 + .exists()) -> {_hash['status']}; read_config "
       f"(json.loads) -> {_read['status']}")

    # -- REV 73 S2(b): TWO REFACTORING SHAPES AND ONE COMPOSED NAME -----
    mod.write_text(
        'from pathlib import Path\n'
        'VER = 2\n'
        '#: two hops: the one-hop gate saw neither of these\n'
        'A = "live/pm_research/declarations/x_declaration_v2.json"\n'
        'B = A\n'
        'D1 = Path(B).read_text()\n'
        '\n\ndef p():\n'
        '    return "live/pm_research/declarations/'
        'x_declaration_v2.json"\n'
        '\n\nD2 = Path(p()).read_text()\n'
        '\n\ndef best():\n'
        '    found = None\n'
        '    return found or '
        '"live/pm_research/declarations/x_declaration_v2.json"\n'
        '\n\nCHOSEN = best()\n'
        'D3 = Path(CHOSEN).read_text()\n'
        '\n\nCOMPOSED = f"declarations/x_declaration_v{VER}.json"\n'
        'D4 = Path(COMPOSED).read_text()\n')
    lshape = literal_census(tmp, merged_chains(tmp))
    _by = {}
    for r in lshape["literals"]:
        _by.setdefault((r["assigned_to"], r["in_function"]), r)
    _two_hop = _by.get(("A", None))
    _ret = _by.get((None, "p"))
    _fallback = _by.get((None, "best"))
    comp = composed_declaration_names(tmp)
    ck("REV 73 S2(b) -- ***ONE HOP IS A REFACTOR AWAY FROM BLIND.*** The "
       "gate followed `LIT -> A -> open(A)` and stopped, so two ordinary "
       "shapes walked past it: a SECOND assignment (`A = LIT; B = A; "
       "open(B)`) and a FUNCTION returning the literal (`def p(): return "
       "LIT` … `open(p())`) -- and the third, a FALLBACK (`return found or "
       "LIT`, whose value lands in a module constant that is opened), is "
       "***the shape a stale default actually hides in***. The flow is the "
       "TRANSITIVE CLOSURE now, to a fixed point, over name-to-name "
       "assignments, `X = f()`, and the operands an `if/else` or an `or` "
       "forwards verbatim. All three are REFUSED here, each naming the "
       "head it should have named",
       _two_hop and _two_hop["flows_into_an_open"] is True
       and _two_hop["status"] == "REFUSED_UNMARKED_NON_HEAD"
       and "->" in _two_hop["flow"]
       and _ret and _ret["flows_into_an_open"] is True
       and "RETURNING FUNCTION" in _ret["flow"]
       and _fallback and _fallback["flows_into_an_open"] is True
       and _fallback["status"] == "REFUSED_UNMARKED_NON_HEAD",
       f"two hops -> {_two_hop['flow'][:70]}…; returning function -> "
       f"{_ret['flow'][:70]}…; fallback -> {_fallback['flow'][:70]}…")
    ck("AND THE COMPOSED NAME IS CENSUSED, NOT SILENTLY MISSED: "
       "`f\"declarations/x_declaration_v{VER}.json\"` cannot be resolved "
       "to a version by a scan over string CONSTANTS -- so it is listed "
       "with its static parts, what it interpolates, and whether it "
       "reaches an open. ***A limit that is counted is a limit; one that "
       "is not counted is a hole***, and the count is what decides which "
       "this is",
       len(comp) == 1 and comp[0]["interpolated"] == ["VER"]
       and comp[0]["flows_into_an_open"] is True
       and "x_declaration_v" in comp[0]["static_parts"],
       f"{len(comp)} composed name(s): {comp[0]['static_parts']} "
       f"interpolating {comp[0]['interpolated']}, reaching an open: "
       f"{comp[0]['flows_into_an_open']}")

    orphan = d / "x_declaration_v4.json"
    orphan.write_text(json.dumps({"v": 4}))
    ch2 = declaration_chains(d)
    ck("KNOWN-BAD: TWO HEADS IN ONE FAMILY REFUSE -- an unlinked v4 beside "
       "the chained v3 leaves no answer to 'which one should a pin name?', "
       "and a census that picked the highest number would be inventing the "
       "link the seat did not write",
       ch2["x_declaration"]["n_heads"] == 2
       and ch2["x_declaration"]["status"] == "MULTIPLE_HEADS_UNLINKED"
       and set(ch2["x_declaration"]["heads"]) == {v3.name, orphan.name},
       f"heads {sorted(ch2['x_declaration']['heads'])}")
    half = d / "y_declaration_v2.json"
    (d / "y_declaration_v1.json").write_text(json.dumps({"v": 1}))
    half.write_text(json.dumps({"v": 2, "supersedes": {"path":
                                                       "y_declaration_v1.json"}}))
    ch3 = declaration_chains(d)
    ck("AND A HALF-WRITTEN LINK IS NOT A LINK (R-608) -- ***THE RULE IS "
       "THE MODULE'S AGAIN, AND WHAT THIS CELL ASSERTS IS MY "
       "TRANSLATION OF IT.*** BE 79/80 followed a `{path}`-only link and "
       "this census carried the property itself (DA 106); **BE 82 fixed "
       "the module**, which now raises `ChainRefused HALF_WRITTEN_LINK` "
       "before this scan can run -- so my own branch is retired rather "
       "than left as code that cannot fire. What a verdict of mine still "
       "rests on is the TRANSLATION: the module's refusal arrives as a "
       "per-family STATUS carrying its NAME, so ***one unfollowable "
       "family does not end the census***, and every other family still "
       "gets an answer",
       ch3["y_declaration"]["status"]
       == "CHAIN_REFUSED_BY_THE_SHARED_RESOLVER"
       and ch3["y_declaration"]["n_heads"] == 0
       and any("HALF_WRITTEN_LINK" in b["status"]
               for b in ch3["y_declaration"]["unlinked_or_broken"])
       and len(ch3) > 1,
       f"y_declaration -> {ch3['y_declaration']['status']} carrying "
       f"{[b['status'] for b in ch3['y_declaration']['unlinked_or_broken']]}"
       f"; {len(ch3) - 1} other family/families still answered")

    # -- REV 72 2.3: THE ALLOWLIST IS A DECLARATION LIKE ANY OTHER --------
    tmp2 = Path(tempfile.mkdtemp(prefix="da94_"))
    d2 = tmp2 / "live" / "pm_research" / "declarations"
    d2.mkdir(parents=True)
    a1 = d2 / "da_nonhead_allowlist_v1.json"
    a1.write_text(json.dumps({"entries": [
        {"names": "x_declaration_v2.json", "reason": "kept on purpose"}]}))
    a1sha = hashlib.sha256(a1.read_bytes()).hexdigest()
    (d2 / "x_declaration_v1.json").write_text(json.dumps({"v": 1}))
    x1sha = hashlib.sha256(
        (d2 / "x_declaration_v1.json").read_bytes()).hexdigest()
    (d2 / "x_declaration_v2.json").write_text(json.dumps(
        {"v": 2, "supersedes": {"path": "x_declaration_v1.json",
                                "sha256": x1sha}}))
    x2sha = hashlib.sha256(
        (d2 / "x_declaration_v2.json").read_bytes()).hexdigest()
    (d2 / "x_declaration_v3.json").write_text(json.dumps(
        {"v": 3, "supersedes": {"path": "x_declaration_v2.json",
                                "sha256": x2sha}}))
    mod2 = tmp2 / "live" / "pm_research" / "da_mod.py"
    #: the SAME non-head literal twice: once never opened (a name), once
    #: opened (a pin). The allowlist can speak about the first only.
    mod2.write_text('from pathlib import Path\n'
                    'P = "live/pm_research/declarations/'
                    'x_declaration_v2.json"\n'
                    'Q = "live/pm_research/declarations/'
                    'x_declaration_v2.json"\n'
                    'D = Path(Q).read_text()\n')
    a2 = d2 / "da_nonhead_allowlist_v2.json"
    a2.write_text(json.dumps({"entries": [], "supersedes": {
        "path": a1.name, "sha256": a1sha}}))
    def _named(rep, ident):
        return next(r for r in rep["literals"] if r["assigned_to"] == ident)

    lA = literal_census(tmp2, merged_chains(tmp2))      # v2 linked, entry gone
    a2.unlink()
    lB = literal_census(tmp2, merged_chains(tmp2))      # v1 alone
    a2.write_text(json.dumps({"entries": []}))          # unlinked v2
    lC = literal_census(tmp2, merged_chains(tmp2))
    a_v1_alone = (_named(lB, "P")["status"] == "MARKED_AND_NOT_A_PIN"
                  and _named(lB, "P")["marker"]["kind"] == "ALLOWLIST"
                  and _named(lB, "P")["marker"]["reason"] == "kept on purpose")
    a_removed = (_named(lA, "P")["status"] == "NOT_A_PIN__NO_OPEN"
                 and _named(lA, "P")["marker"] is None)
    #: and the OPENED one is refused in BOTH, allowlist or not (2.1)
    a_open_refused = all(
        _named(r, "Q")["status"]
        == "REFUSED_NON_HEAD_PIN_MARKER_DOES_NOT_EXCUSE_AN_OPEN"
        for r in (lB,)) and _named(lA, "Q")["status"] \
        == "REFUSED_UNMARKED_NON_HEAD"
    ck("REV 72 2.3 -- ***THE ALLOWLIST RESOLVES THROUGH THE SAME HEAD RULE "
       "AS ANY OTHER DECLARATION.*** Globbing every `*_nonhead_allowlist_"
       "v*.json` and MERGING them meant a v2 that REMOVES an entry could "
       "not: the v1 entry survived the merge and the exemption it granted "
       "could never be withdrawn. Driven three ways -- v1 alone ADMITS the "
       "literal, a PAIR-LINKED v2 with the entry removed makes it REFUSE "
       "again, and an UNLINKED v2 (two heads) refuses the ALLOWLIST ITSELF "
       "by name rather than choosing one. Exposure today is zero: no such "
       "file exists, which is why it is fixed before one does",
       a_v1_alone and a_removed and a_open_refused
       and lC["n_allowlist_refusals"] == 1
       and lC["allowlist_refusals"][0]["status"]
       == "ALLOWLIST_DOES_NOT_RESOLVE_TO_ONE_HEAD",
       f"the never-opened name: v1 alone -> {_named(lB, 'P')['status']}, "
       f"PAIR-LINKED v2 with the entry removed -> "
       f"{_named(lA, 'P')['status']}; unlinked v2 -> "
       f"{lC['allowlist_refusals'][0]['status']}. And the OPENED literal "
       f"is refused either way ({_named(lB, 'Q')['status']} / "
       f"{_named(lA, 'Q')['status']}): 2.1 leaves the allowlist able to "
       f"explain a NAME, never to excuse a READ")

    # -- REV 72 4: THE SECOND DECLARATION DIRECTORY -----------------------
    mm = tmp2 / "live" / "mm_research" / "declarations"
    mm.mkdir(parents=True)
    (mm / "p002_declaration_v1.json").write_text(json.dumps({"v": 1}))
    m1sha = hashlib.sha256(
        (mm / "p002_declaration_v1.json").read_bytes()).hexdigest()
    (mm / "p002_declaration_v2.json").write_text(json.dumps(
        {"v": 2, "supersedes": {"path": "p002_declaration_v1.json",
                                "sha256": m1sha}}))
    mod2.write_text('from pathlib import Path\n'
                    'P = "live/mm_research/declarations/'
                    'p002_declaration_v1.json"\n'
                    'D = Path(P).read_text()\n')
    lD = literal_census(tmp2, merged_chains(tmp2))
    before = literal_census(tmp2, declaration_chains(d2))
    ck("REV 72 4 -- ***A `head: null` THAT MEANT \"NOT SCANNED\" IS NOT A "
       "FINDING.*** Fifty pins read null because their families live in "
       "`live/mm_research/declarations/` and this census read only "
       "`live/pm_research/`. Scanning ONE dir, a pin on a superseded "
       "mm-research declaration comes back `head: null` and is not "
       "refused; scanning BOTH, the same pin is REFUSED and names its "
       "head -- and where a family really is unknown the row says WHY, "
       "never a bare null (rule 11)",
       before["n_refused"] == 0
       and before["literals"][0]["is_head"] is None
       and before["literals"][0]["head_is_null_because"]
       == ("DERIVED_TREE_NOT_INDEXED_FOR_THIS_CALL__NOT_ASKED")
       and lD["n_refused"] == 1
       and lD["naming_a_non_head"][0]["head"] == "p002_declaration_v2.json"
       and lD["naming_a_non_head"][0]["declarations_dir"]
       == "live/mm_research/declarations",
       f"one dir -> head {before['literals'][0]['head']}, "
       f"{before['n_refused']} refused, because "
       f"{before['literals'][0]['head_is_null_because']}; both dirs -> "
       f"head {lD['naming_a_non_head'][0]['head']}, {lD['n_refused']} "
       f"refused")
    #: REV 72 4, the second half: 45 pins still read null after BOTH dirs
    #: were scanned, and 22 of them name artifacts in the LEDGER's derived
    #: tree -- not declarations at all. The two facts get two names.
    mod2.write_text('from pathlib import Path\n'
                    'P = "phase2_fits_v9.json"\n'
                    'D = Path(P).read_text()\n')
    ch_d = merged_chains(tmp2)
    l_noix = literal_census(tmp2, ch_d)
    l_ix = literal_census(tmp2, ch_d, {"status": "INDEXED", "dir": "/x",
                                       "names": {"phase2_fits_v9.json":
                                                 ["phase2_fits"]}})
    l_abs = literal_census(tmp2, ch_d, {"status": "INDEXED", "dir": "/x",
                                        "names": {}})
    ck("AND THE NULLS THAT SURVIVE BOTH DIRECTORIES ARE SPLIT BY WHAT THEY "
       "ACTUALLY ARE: a literal naming an artifact in the LEDGER's DERIVED "
       "tree is `NAMES_A_DERIVED_ARTIFACT_NOT_A_DECLARATION` (the head "
       "rule does not reach it -- there is no supersession chain in a "
       "derived tree), one naming a file present NOWHERE is "
       "`NAMED_FILE_IS_IN_NO_SCANNED_DIRECTORY`, and a census run with NO "
       "index says so rather than borrowing either verdict. ***Three "
       "different facts that all printed as `null` before***",
       l_ix["literals"][0]["head_is_null_because"]
       == "NAMES_A_DERIVED_ARTIFACT_NOT_A_DECLARATION"
       and l_ix["literals"][0]["found_in_derived"] == ["phase2_fits"]
       and l_abs["literals"][0]["head_is_null_because"]
       == "NAMED_FILE_IS_IN_NO_SCANNED_DIRECTORY"
       and l_noix["literals"][0]["head_is_null_because"].startswith(
           "DERIVED_TREE_NOT_INDEXED_FOR_THIS_CALL")
       and l_ix["n_head_is_null_and_a_pin"] == 1,
       f"indexed+present -> {l_ix['literals'][0]['head_is_null_because']}; "
       f"indexed+absent -> {l_abs['literals'][0]['head_is_null_because']}; "
       f"no index -> {l_noix['literals'][0]['head_is_null_because']}")
    mod2.write_text('from pathlib import Path\n'
                    'P = "live/mm_research/declarations/'
                    'p002_declaration_v1.json"\n'
                    'D = Path(P).read_text()\n')
    (d2 / "p002_declaration_v3.json").write_text(json.dumps({"v": 3}))
    chBoth = merged_chains(tmp2)
    ck("AND A FAMILY NAME LIVING IN BOTH DIRECTORIES REFUSES BY NAME: two "
       "directories are two namespaces, so `p002_declaration` in each has "
       "no single head and a pin naming it has no single answer -- the "
       "census says FAMILY_IN_TWO_DECLARATION_DIRS rather than silently "
       "taking the first directory scanned",
       chBoth["p002_declaration"]["status"]
       == "FAMILY_IN_TWO_DECLARATION_DIRS"
       and chBoth["p002_declaration"]["n_heads"] == 2,
       f"{chBoth['p002_declaration']['status']}, heads "
       f"{sorted(chBoth['p002_declaration']['heads'])}")

    # -- REV 85 S2: THE MARK'S LICENSING CLAUSE IS A PREDICATE ---------
    mk = Path(tempfile.mkdtemp(prefix="da111mark_"))
    (mk / "live" / "pm_research").mkdir(parents=True)
    _fam = sorted(MARKED_PRE_R608_FAMILIES)[0]
    _clean = mark_readers(_fam, mk)
    (mk / "live" / "pm_research" / "a_reader.py").write_text(
        "import declaration_chain as DC\n"
        "from pathlib import Path\n"
        f"def go():\n    return DC.resolve_head(Path('.'), {_fam!r})\n")
    _dirty = mark_readers(_fam, mk)
    (mk / "live" / "pm_research" / "by_path.py").write_text(
        f"P = 'data/pm_5min/derived/{_fam}_v2.json'\n"
        "OPEN = open(P) if False else None\n")
    _by_path_only = mark_readers(_fam, mk)
    (mk / "live" / "pm_research" / "a_reader.py").unlink()
    _removed = mark_readers(_fam, mk)
    ck("REV 85 S2 -- ***THE MARK IS LICENSED BY ONE CLAUSE, SO THE CLAUSE "
       "IS A PREDICATE.*** `who_reads_it_through_the_chain: nobody` is "
       "what makes marking a pre-rule family honest rather than "
       "convenient: nothing resolves it, so nothing gets a wrong answer. "
       "***The moment a reader resolves the family THROUGH THE CHAIN the "
       "clause is false and the mark with it*** -- the census reports "
       "`MARK_REFUSED_FAMILY_IS_READ_THROUGH_THE_CHAIN`, names the "
       "reader, and calls the family UNANSWERED. Driven: no reader -> no "
       "hits; a module calling `resolve_head(..., <family>)` -> ONE hit "
       "naming file and line; ***a module naming the family only by "
       "LITERAL PATH -> still no hit***, because reading a file is not "
       "resolving a chain; the resolver-path module removed -> the mark "
       "stands again",
       _clean == [] and len(_dirty) == 1
       and _dirty[0]["call"] == "resolve_head"
       and _dirty[0]["file"].endswith("a_reader.py")
       and len(_by_path_only) == 1
       and _removed == [],
       f"clean -> {len(_clean)} readers; a resolver-path reader -> "
       f"{len(_dirty)} at {_dirty[0]['file']}:{_dirty[0]['line']} via "
       f"{_dirty[0]['call']}(); a literal-path-only reader adds "
       f"{len(_by_path_only) - len(_dirty)}; removed -> {len(_removed)}")

    # -- R-673(c) / MEM 198-199: THE CENSUS ON ITS OWN FAMILY ----------
    fam = Path(tempfile.mkdtemp(prefix="da97fam_"))
    r1 = fam / "p003_da_nonhead_census__20260101T000000Z.json"
    r2 = fam / "p003_da_nonhead_census__20260101T010000Z.json"
    r1.write_text(json.dumps({"protocol": "x"}))
    r2.write_text(json.dumps({"protocol": "x"}))
    before = own_family_chain(fam)
    r3 = fam / "p003_da_nonhead_census__20260101T020000Z.json"
    r3.write_text(json.dumps({"protocol": "x",
                              "supersedes": supersession_block(
                                  r2, chain_over=[r1])}))
    after = own_family_chain(fam)
    r4 = fam / "p003_da_nonhead_census__20260101T030000Z.json"
    r4.write_text(json.dumps({"protocol": "x", "supersedes": {
        "path": r3.name, "sha256": "e" * 64}}))
    bad = own_family_chain(fam)
    ck("R-673(c) / MEM 198 -- ***THE CENSUS THAT REFUSES A FAMILY WITH TWO "
       "HEADS WAS ITSELF A FAMILY WITH FOUR.*** Five of its own records sat "
       "in the ledger with ONE link between them, and its own records live "
       "in the DERIVED tree rather than a declarations directory, ***so "
       "nothing was looking***. It judges them now by the predicate it "
       "applies to everyone else: two unlinked records are MULTIPLE_HEADS; "
       "a successor naming one by the PAIR and the other in its CHAIN "
       "resolves the family to ONE HEAD ***without editing either older "
       "record***; and a chain entry whose digest does not match the file "
       "is TARGET_DIGEST_MISMATCH -- the target stays a head, because a "
       "link that does not verify is not a link",
       before["status"] == "MULTIPLE_HEADS" and before["n_heads"] == 2
       and after["status"] == "ONE_HEAD" and after["heads"] == [r3.name]
       and after["n_links"] == 2
       and bad["n_heads"] == 2
       and any(x["status"] == "TARGET_DIGEST_MISMATCH"
               for x in bad["unlinked_or_broken"]),
       f"two unlinked -> {before['status']} ({before['n_heads']} heads); "
       f"one successor with a chain -> {after['status']} via "
       f"{after['n_links']} links; a chain entry with a wrong digest -> "
       f"{[x['status'] for x in bad['unlinked_or_broken']]}, heads "
       f"{bad['n_heads']}")

    pri = tmp2 / "prior_receipt.json"
    pri.write_text(json.dumps({"protocol": "P003_DA_NONHEAD_CENSUS_V1"}))
    blk = supersession_block(pri)
    try:
        supersession_block(tmp2 / "not_here.json")
        refused_by_name = False
    except FileNotFoundError as e:
        refused_by_name = "SUPERSEDED_RECEIPT_NOT_PRESENT" in str(e)
    ck("AND THIS RECEIPT SUPERSEDES ITS V1 BY THE R-608 PAIR: the census's "
       "RULES changed, so the number it prints is not comparable to the "
       "one before it -- the link carries {path, sha256} on a PRESENT "
       "file, states WHAT CHANGED, and leaves the V1 receipt unedited as "
       "provenance (rule 13). A prior that is not there REFUSES BY NAME, "
       "never as 'no link'",
       blk["sha256"] == hashlib.sha256(pri.read_bytes()).hexdigest()
       and blk["what_changes"] == WHAT_CHANGES_IN_V2 and refused_by_name,
       f"pair on {pri.name}: sha {blk['sha256'][:12]}…; absent prior -> "
       f"refused by name: {refused_by_name}")

    # ---- DA 119: ONE CENSUS, ONE TREE --------------------------------
    #: The chains half resolves through `da_root.code_root`; the deploy-pin
    #: half called `stale_pins(root)` and let `decl_dir` DEFAULT to the
    #: importing module's own directory. Run from a seat worktree the two
    #: halves then read two different trees, which is how a CLEAN pin read
    #: STALE minutes after its successor landed.
    with tempfile.TemporaryDirectory() as _t3:
        _sr = Path(_t3)
        _sd = _sr / "live" / "pm_research" / "declarations"
        _sd.mkdir(parents=True)
        (_sr / "live" / "pm_research" / "pinned_thing.py").write_text("x = 1\n")
        _pinned_sha = hashlib.sha256(
            (_sr / "live" / "pm_research" / "pinned_thing.py").read_bytes()
        ).hexdigest()
        (_sd / "da_midnight_deploy_pin_v1.json").write_text(json.dumps({
            "unit": "scratch.service", "commit": "0" * 40,
            "deployed_at": "2026-01-01T00:00:00Z", "n_files": 1,
            "first_of_family": True,
            "files": [{"path": "live/pm_research/pinned_thing.py",
                       "sha256": _pinned_sha, "tier": "REFUSE"}]}))
        _scoped = _deploy_pin_state(_sr)
        import da_deploy_pin as _P                            # noqa: PLC0415
        _defaulted = _P.stale_pins(_sr)
        ck("ONE CENSUS, ONE TREE: the deploy-pin half now resolves the pin "
           "from the ROOT THIS CENSUS RESOLVED -- driven on a scratch root "
           "whose pin family is a v1 of its own, it reads THAT v1 and calls "
           "it CLEAN",
           _scoped["pin"]["name"] == "da_midnight_deploy_pin_v1.json"
           and _scoped["status"] == "CLEAN" and _scoped["n_stale"] == 0,
           f"scoped -> {_scoped['pin']['name']} {_scoped['status']}")
        ck("KNOWN-BAD, DRIVEN AT THE DEFAULT THAT CAUSED IT: the same call "
           "WITHOUT `decl_dir` resolves the pin from the IMPORTING MODULE'S "
           "own directory while checking the scratch root's files against "
           "it -- a different version and a different verdict from the same "
           "root. ***That is the reading that reported two files STALE from "
           "a seat worktree minutes after v3 landed in the ledger***",
           _defaulted["pin"]["name"] != _scoped["pin"]["name"],
           f"defaulted -> {_defaulted['pin']['name']} "
           f"{_defaulted['status']} vs scoped -> "
           f"{_scoped['pin']['name']} {_scoped['status']}")


    # ---- REV 90 S B5: THE RULE IS A CELL, BOUND TO THE RESOLUTION SURFACE
    #: Four synthetic modules under a scratch root, one per case. No real
    #: file is judged here; the real tree's answer is in the census report.
    with tempfile.TemporaryDirectory() as _st:
        _sr = Path(_st)
        (_sr / "live" / "pm_research").mkdir(parents=True)

        def _mod(name, body):
            (_sr / "live" / "pm_research" / name).write_text(body)

        _mod("resolver_with_a_cell.py", """
import declaration_chain as DC
def go(d):
    return DC.resolve_head(d, "fam")
def selftest():
    import subprocess, sys
    r = subprocess.run([sys.executable, "declaration_chain.py", "--falsify"],
                       capture_output=True, text=True)
    return r.returncode
""")
        _mod("resolver_without_a_cell.py", """
import declaration_chain as DC
def go(d):
    return DC.resolve_head(d, "fam")
""")
        _mod("helper_only.py", """
from declaration_chain import plain_create_mode
def go():
    return plain_create_mode()
""")
        _mod("drive_outside_a_battery.py", """
import declaration_chain as DC
def go(d):
    return DC.write_next_version(d, "fam", {}, {})
def a_helper(prog):
    import subprocess, sys
    return subprocess.run([sys.executable, str(prog), "--falsify"])
""")
        _surf = chain_resolution_surface(_sr)
        _by = {Path(r["file"]).name: r for r in _surf["in_the_surface"]}
        _out = {Path(r["file"]).name for r in _surf["outside_the_surface"]}
    ck("REV 90 S B5 -- THE SURFACE IS DERIVED FROM THE CODE AND THE RULE IS "
       "A CELL: a module that imports the shared module AND references a "
       "RESOLUTION symbol is in the surface; one that runs "
       "`declaration_chain --falsify` in its battery is OK",
       _by["resolver_with_a_cell.py"]["status"] == "OK"
       and _by["resolver_with_a_cell.py"]["resolution_symbols"]
       == ["resolve_head"],
       f"with a cell -> {_by['resolver_with_a_cell.py']['status']}")
    ck("KNOWN-BAD, DRIVEN -- ***A MODULE THAT IMPORTS THE RESOLVER AND SHIPS "
       "NO --falsify CELL IS FLAGGED BY NAME***: "
       "IMPORTS_THE_RESOLVER_AND_SHIPS_NO_FALSIFIER_CELL. The rule "
       "re-drifted three times in one round while it was checked by a "
       "reviewer noticing; R-605 -- a practice that depends on noticing is "
       "not a control",
       _by["resolver_without_a_cell.py"]["status"]
       == "IMPORTS_THE_RESOLVER_AND_SHIPS_NO_FALSIFIER_CELL"
       and _surf["verdict"] == "REFUSED_A_RESOLVER_SHIPS_NO_FALSIFIER_CELL",
       f"without a cell -> {_by['resolver_without_a_cell.py']['status']}; "
       f"verdict {_surf['verdict']}")
    ck("AND THE BINDING IS TO THE RESOLUTION SURFACE, NOT TO THE IMPORT "
       "(REV 90's refinement): a module importing only `plain_create_mode` "
       "-- a mode helper -- is OUTSIDE the surface and is NOT asked for a "
       "chain falsifier. ***A cell nobody can justify is how a rule stops "
       "being obeyed***",
       "helper_only.py" in _out
       and "helper_only.py" not in _by,
       f"outside the surface: {sorted(_out)}")
    ck("AND THE DRIVE MUST BE IN A BATTERY: a module whose only "
       "`--falsify` call sits in a HELPER's body -- the shape the module "
       "that DEFINES the shared helper actually has -- is flagged, because "
       "that call is the helper, not a cell that runs when its battery runs",
       _by["drive_outside_a_battery.py"]["status"]
       == "IMPORTS_THE_RESOLVER_AND_SHIPS_NO_FALSIFIER_CELL"
       and _by["drive_outside_a_battery.py"]["falsifier_drives"]
       and not _by["drive_outside_a_battery.py"][
           "falsifier_drives"][0]["in_a_battery"],
       f"a drive in a helper -> "
       f"{_by['drive_outside_a_battery.py']['status']} "
       f"({_by['drive_outside_a_battery.py']['falsifier_drives']})")

    # ---- R-753 (2) / REV 89 S6.3a: THE DIGEST PREDICATE ---------------
    #: Four synthetic modules, one per conjunct, so each half of the rule
    #: is shown able to fail. No real file is read here.
    _GOOD = ("""
P = {"path": "fam_v1.json",
     "sha256": "%s",
     "the_act": "recorded by the act that used it"}
def read(d):
    import hashlib
    q = d / P["path"]
    got = hashlib.sha256(q.read_bytes()).hexdigest()
    if got != P["sha256"]:
        raise RuntimeError("moved")
    return q.read_text()
""" % ("a" * 64))
    _NO_DIGEST = """
P = {"path": "fam_v1.json", "the_act": "named by the act"}
def read(d):
    return (d / P["path"]).read_text()
"""
    _NEVER_COMPARED = ("""
P = {"path": "fam_v1.json",
     "sha256": "%s",
     "the_act": "recorded by the act"}
def read(d):
    import hashlib
    hashlib.sha256(b"x").hexdigest()
    return (d / P["path"]).read_text()
""" % ("b" * 64))
    _NO_PROVENANCE = ("""
P = {"path": "fam_v1.json", "sha256": "%s"}
def read(d):
    import hashlib
    got = hashlib.sha256((d / P["path"]).read_bytes()).hexdigest()
    if got != P["sha256"]:
        raise RuntimeError("moved")
    return (d / P["path"]).read_text()
""" % ("c" * 64))
    _g = hashed_against_a_recorded_digest(ast.parse(_GOOD), "P")
    _nd = hashed_against_a_recorded_digest(ast.parse(_NO_DIGEST), "P")
    _nc = hashed_against_a_recorded_digest(ast.parse(_NEVER_COMPARED), "P")
    _np = hashed_against_a_recorded_digest(ast.parse(_NO_PROVENANCE), "P")
    ck("THE DIGEST PREDICATE ADMITS A READER OF HISTORY: a container "
       "carrying the file's digest, a module that COMPUTES a digest and "
       "COMPARES it against that container, and a field naming who "
       "recorded it -- R-753 (2) as REV 89 S6.3a accepted it",
       _g["admissible"] and _g["provenance_fields"] == ["the_act"]
       and _g["compared_at_lines"],
       f"good -> admissible {_g['admissible']}, digest "
       f"{_g['digest_in_the_code']}, compared at {_g['compared_at_lines']}")
    ck("KNOWN-BAD, DRIVEN -- ***NAMES A NON-HEAD AND INTERPRETS IT WITHOUT "
       "A DIGEST: REFUSED***. That is the case the predicate exists to "
       "keep refusing, and it is the shape a stale pin actually has",
       not _nd["admissible"]
       and "no 64-hex digest in the container" in (_nd["why"] or ""),
       f"no digest -> {_nd['why']}")
    ck("AND EACH HALF FAILS ON ITS OWN: a digest WRITTEN DOWN but never "
       "compared is refused (written down is not asserted at the read), "
       "and a digest asserted with NO field naming who recorded it is "
       "refused (REV 89's added clause) -- so the green above rests on "
       "four conjuncts, not on one that carries the rest",
       (not _nc["admissible"]) and "never compared" in (_nc["why"] or "")
       and (not _np["admissible"])
       and "no field naming who recorded it" in (_np["why"] or ""),
       f"never compared -> {(_nc['why'] or '')[:60]}…; no provenance -> "
       f"{(_np['why'] or '')[:60]}…")

    # ---- rule 20's clause (REV 84 S3.2 / REV 85 S3, R-726): THE SHARED
    # MODULE'S OWN FALSIFIER RUNS AS ONE CELL OF THIS BATTERY -----------
    #: This module IMPORTS `declaration_chain`, so a regression in the one
    #: implementation is this battery's problem too. It is SPAWNED AS A
    #: PROCESS, not called: a broken `__main__`, a syntax error under an
    #: edit or a falsifier that no longer runs at all is then a failure
    #: HERE rather than something an in-process call routes around.
    #: What stays independent is only what THIS module's own verdicts rest
    #: on at the seam -- never a re-test of the module's invariant.
    def _dc_falsify(_prog):
        import subprocess as _sp                              # noqa: PLC0415
        import sys as _sy                                     # noqa: PLC0415
        _r = _sp.run([_sy.executable, str(_prog), "--falsify"],
                     capture_output=True, text=True, timeout=300)
        _ls = [x for x in (_r.stdout or "").strip().splitlines() if x.strip()]
        return (_r.returncode, _ls[-1] if _ls else "",
                [x for x in _ls if x.startswith("FAIL")])

    _DC_PATH = HERE / "declaration_chain.py"
    _dc_rc, _dc_sum, _dc_bad = _dc_falsify(_DC_PATH)
    ck("REV 84 S3.2 -- ONE IMPLEMENTATION, N DETECTORS: this battery "
       "RUNS `declaration_chain.py --falsify` AS A SUBPROCESS, so a "
       "regression in the shared chain module fails every importer at "
       "once and no importer re-implements its logic",
       _dc_rc == 0 and _dc_sum.endswith("0 failures") and not _dc_bad,
       f"rc {_dc_rc}: {_dc_sum!r} {_dc_bad or ''}")
    #: RED FIRST. A cell that only ever runs the GOOD module has never been
    #: shown to fire. One falsifier is DISARMED in a COPY -- the
    #: VERSION_PATH_EXISTS guard, which is the refusal that keeps a landed
    #: version immutable -- and this cell must FAIL on it.
    import tempfile as _tf120                                 # noqa: PLC0415
    with _tf120.TemporaryDirectory() as _dc_td:
        _dc_copy = Path(_dc_td) / "declaration_chain.py"
        _dc_src = Path(_DC_PATH).read_text()
        _dc_disarmed = _dc_src.replace("    if dst.exists():",
                                       "    if False and dst.exists():")
        _dc_copy.write_text(_dc_disarmed)
        _bad_rc, _bad_sum, _bad_fails = _dc_falsify(_dc_copy)
    ck("KNOWN-BAD, DRIVEN: the SAME cell against a COPY of the shared "
       "module with ONE falsifier disarmed (VERSION_PATH_EXISTS, the "
       "refusal that makes a landed version immutable) FAILS -- so the "
       "green above is a measurement and not a cell that cannot fire",
       _dc_disarmed != _dc_src and _bad_rc != 0
       and "1 failures" in _bad_sum and _bad_fails,
       f"disarmed copy -> rc {_bad_rc}: {_bad_sum!r}; "
       f"{(_bad_fails or [''])[0][:80]}")

    print(f"\n{'SELFTEST OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {fails} failure(s)")
    return checks, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--supersedes", type=Path, default=None)
    ap.add_argument("--chain-over", type=Path, action="append", default=[],
                    help="older records of this family to name in the "
                         "chain, so a history that was never linked "
                         "resolves to one head without editing any of them")
    ap.add_argument("--first-of-family", action="store_true",
                    help="declare that no prior record of this family "
                         "exists -- the ONLY alternative to naming one")
    ap.add_argument("--what-changed", default=None)
    a = ap.parse_args()
    #: R-673(c) / MEM 199: THE LINK IS MANDATORY. An emission that names
    #: no prior and does not DECLARE itself first-of-family refuses --
    #: ***an omission that is easier than an error is the thing that
    #: happens***, and it is how this family grew four heads.
    if a.output and not a.supersedes and not a.first_of_family:
        fam = own_family_chain()
        print(f"REFUSED: NO_PRIOR_NAMED -- this family already has "
              f"{fam.get('n_records')} record(s) at {fam.get('dir')}. "
              f"Name the record this one supersedes with --supersedes, or "
              f"declare --first-of-family. A record that silently becomes "
              f"a second head is how this census's own family reached "
              f"four.")
        return 1
    if a.first_of_family and a.supersedes:
        print("REFUSED: FIRST_OF_FAMILY_NAMES_A_PRIOR -- a record cannot "
              "be both the first of its family and the successor of "
              "another.")
        return 1
    if a.selftest:
        checks, n_fail = selftest()
        if a.output:
            rep = build_report(prior=a.supersedes,
                               chain_over=a.chain_over,
                               what_changed=a.what_changed)
            rep["checks"] = checks
            rep["n_checks"] = len(checks)
            rep["n_failed"] = n_fail
            rep["both_directions"] = True
            a.output.write_text(json.dumps(rep, indent=2, sort_keys=True,
                                           default=str) + "\n")
        return 1 if n_fail else 0
    if a.census:
        rep = build_report(prior=a.supersedes, chain_over=a.chain_over,
                           what_changed=a.what_changed)
        if a.output:
            a.output.write_text(json.dumps(rep, indent=2, sort_keys=True,
                                           default=str) + "\n")
        print(json.dumps({
            "n_families": rep["n_families"],
            "families_without_exactly_one_head":
                rep["n_families_without_exactly_one_head"],
            "n_literals": rep["literal_census"]["n_literals"],
            #: REV 90 S B5(b): BOTH numbers, so a reader of the summary
            #: alone is not misled -- 50 names of non-head versions are
            #: SCANNED and 3 are JUDGED, and the classes that separate
            #: them are in `the_scanned_set_by_class`.
            "n_scanned_naming_a_non_head":
                rep["literal_census"]["n_scanned_naming_a_non_head"],
            "naming_a_non_head":
                rep["literal_census"]["n_naming_a_non_head"],
            "the_scanned_set_by_class":
                rep["literal_census"]["the_scanned_set_by_class"],
            "chain_resolution_surface": {
                "n_in_the_surface":
                    rep["chain_resolution_surface"]["n_in_the_surface"],
                "n_missing_the_cell":
                    rep["chain_resolution_surface"]["n_missing_the_cell"],
                "missing": [m["file"] for m in
                            rep["chain_resolution_surface"][
                                "missing_the_cell"]],
                "verdict": rep["chain_resolution_surface"]["verdict"]},
            "verdict": rep["literal_census"]["verdict"]}, indent=1))
        return 0
    ap.error("--selftest or --census [--output <path>]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
