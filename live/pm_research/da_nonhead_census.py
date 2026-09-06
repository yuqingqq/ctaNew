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


def declaration_chains(decl_dir: Path) -> dict:
    """Families, their heads, and the links that make them.

    A head is a member NO OTHER MEMBER supersedes. The link is the R-608
    PAIR: both halves must land on one present file, or it is not a link
    and the family is reported as unlinked rather than chained."""
    files = sorted(p for p in decl_dir.glob("*.json") if p.is_file())
    present = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in files}
    fams: dict = {}
    for p in files:
        fam, ver = _family(p.stem)
        fams.setdefault(fam, []).append((ver, p))
    out = {}
    for fam, members in sorted(fams.items()):
        members.sort(key=lambda t: (t[0] is None, t[0]))
        superseded, links, bad = set(), [], []
        for _, p in members:
            try:
                obj = json.loads(p.read_text())
            except (OSError, ValueError) as e:
                bad.append({"file": p.name, "status": "UNREADABLE",
                            "why": str(e)[:120]})
                continue
            blk = obj.get("supersedes")
            if isinstance(blk, str):
                blk = {"path": blk}
            if not isinstance(blk, dict):
                continue
            path, sha = blk.get("path"), blk.get("sha256")
            if not path or not sha:
                bad.append({"file": p.name,
                            "status": "SUPERSESSION_LINK_INCOMPLETE",
                            "has": [k for k in ("path", "sha256")
                                    if blk.get(k)]})
                continue
            target = Path(path).name
            if target not in present:
                bad.append({"file": p.name, "status": "TARGET_ABSENT",
                            "target": target})
                continue
            if present[target] != sha:
                bad.append({"file": p.name,
                            "status": "TARGET_DIGEST_MISMATCH",
                            "target": target})
                continue
            superseded.add(target)
            links.append({"from": p.name, "to": target})
        heads = [p.name for _, p in members if p.name not in superseded]
        out[fam] = {
            "members": [p.name for _, p in members],
            "n_members": len(members),
            "heads": heads, "n_heads": len(heads),
            "links": links, "unlinked_or_broken": bad,
            "status": ("ONE_HEAD" if len(heads) == 1
                       else "NO_HEAD" if not heads
                       else "MULTIPLE_HEADS_UNLINKED"),
            "why": ("a family with more than one head has no answer to "
                    "'which one should a pin name?' -- the versions exist "
                    "and nothing links them"
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
    ck("AND A HALF-WRITTEN LINK IS NOT A LINK (R-608): a `supersedes` "
       "carrying only a path leaves BOTH versions as heads and is reported "
       "as SUPERSESSION_LINK_INCOMPLETE, never silently followed",
       ch3["y_declaration"]["n_heads"] == 2
       and any(b["status"] == "SUPERSESSION_LINK_INCOMPLETE"
               for b in ch3["y_declaration"]["unlinked_or_broken"]),
       f"y_declaration heads {sorted(ch3['y_declaration']['heads'])}, "
       f"broken {[b['status'] for b in ch3['y_declaration']['unlinked_or_broken']]}")

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
            "naming_a_non_head":
                rep["literal_census"]["n_naming_a_non_head"],
            "verdict": rep["literal_census"]["verdict"]}, indent=1))
        return 0
    ap.error("--selftest or --census [--output <path>]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
