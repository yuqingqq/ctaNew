"""WHAT MOVED, AND WHETHER ANY OF IT IS ON THE PATH THAT SCORED THE BOOK.

DA 168's design, adopted by the coordinator: **DO NOT NARROW THE CONDITION.
WIDEN THE PAYLOAD.**

`BOOK_BUILT_BY_DIFFERENT_SCORING_CODE` compares WHOLE MODULES, and that is
right: "these bytes produced this book" has no honest weaker form, because
static reachability is a LOWER BOUND (BE's own words in
`be_producing_closure`) and a predicate built on a lower bound under-covers
SILENTLY. So the condition stays whole-module and the run still refuses.

What was missing is EVIDENCE. On 2026-09-09 the EV21 09-03 book refused
because `de_multiday_gate1_runner.py` moved -- and establishing that
nothing which moved could change that book's scores cost a seat five
minutes of investigation the refusal should have carried: the scoring path
through that 16,205-line module is ONE function, the four commits that
moved it touched other functions, and the intersection is empty. **A
refusal that costs an investigation every time is a refusal that gets
routed around.**

THE OPERATION, stated before any answer:

    A definition is ON THE SCORING PATH if it is reachable, in the static
    call/attribute graph, from the builder's own scoring entry points
    (`be_producing_closure.SCORING_ENTRY_POINTS`) -- the same operation
    that derives the module set, one level finer.

    A definition CHANGED if its source segment differs between THE BYTES
    THE BOOK RECORDS and the bytes on disk. The book's bytes are resolved
    from git by DIGEST, never assumed: an unresolvable digest is a STATUS
    and the payload then reports the intersection as UNKNOWN, never empty.

    THE INTERSECTION is (reachable defs INTERSECT changed defs) UNION
    (module-level names READ BY a reachable def INTERSECT changed
    module-level names). A constant is on the path if a reachable function
    reads it -- `harmful_stateful_policy` is in the module set for exactly
    that reason, and the same logic one level down puts `PARAMS_REL` on
    this module's path.

WHAT THIS PAYLOAD IS NOT (and every exit says so):

  * **It is not a licence.** The refusal fires either way. An empty
    intersection is EVIDENCE FOR A HUMAN DECISION -- rebuild, or supersede
    with the reason recorded -- and rule 14 keeps the decision in the
    policy layer. Nothing here returns a boolean any caller may treat as
    permission, and the runner's call site is a `try` around evidence, not
    a branch around a refusal.
  * **It is not complete.** Reachability is a LOWER BOUND: it does not
    follow `getattr`, values passed as callbacks, or monkey-patching, and
    `be_producing_closure.KNOWN_DYNAMIC_SITES` names two real ones. It
    does not model module-level executable code, decorators applied from
    elsewhere, or a changed import binding a different implementation
    behind an unchanged call. Those live in `LIMITS` on every payload.
  * **An absent answer is not a clean answer.** Every failure inside this
    module becomes a STATUS on the payload with `intersection_known:
    False`; a reader that cannot tell "empty" from "not computed" is the
    `BINANCE_GAP_EXCLUDED: 0` shape.

Run `python3 de_scoring_path_delta.py --falsify` for the cells.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import be_producing_closure as PC          # noqa: E402

#: The one sentence that must survive being quoted out of context.
NOT_A_LICENCE = (
    "EVIDENCE FOR A HUMAN DECISION, NEVER A LICENCE TO PROCEED. The "
    "refusal stands whatever this payload says: the predicate is "
    "whole-module because static reachability is a lower bound, and a run "
    "that continued on the strength of its own payload would be the "
    "failure this predicate exists to prevent.")

LIMITS = (
    "REACHABILITY IS A LOWER BOUND -- it does not follow `getattr`, a "
    "callback passed as a value, or monkey-patching; "
    "`be_producing_closure.KNOWN_DYNAMIC_SITES` names two real sites in "
    "this tree.",
    "MODULE-LEVEL EXECUTABLE CODE is not modelled. A changed top-level "
    "statement that is neither a def nor a simple assignment is reported "
    "as `residual_top_level_differs` and is NOT resolved onto the path.",
    "A DECORATOR or an import binding a different implementation behind "
    "an unchanged call site changes behaviour without changing the "
    "caller's bytes; only the decorator's or the target's own module "
    "would show it.",
    "THE KEY SPACE IS THE BARE DEFINITION NAME -- `be_producing_closure`'s "
    "own space, so the two halves are comparable. Two definitions sharing "
    "a name are ONE unit here, and a change to either marks both.",
    "A METHOD CALLED THROUGH AN INSTANCE IS NOT FOLLOWED -- the walk "
    "resolves `alias.attr` only when `alias` is an imported MODULE, so "
    "`h = C(); h.method()` reaches nothing. A method is on the path only "
    "when it is reached as `module.name`, which is how this tree's "
    "scoring path is written. MEASURED, not assumed: there is a cell for "
    "it in `--falsify`.",
    "AN EMPTY INTERSECTION IS ABOUT THE MODULES THAT MOVED. It says "
    "nothing about a module the receipt never named (see "
    "MEMBERSHIP_LIMIT on the refusal itself).")


class DeltaRefused(RuntimeError):
    """A named refusal from this module. Bad or partial input only."""


#: DE 179, MEASURED THE HARD WAY. This was a MODULE-LEVEL cache and it
#: retained every AST it parsed for the life of the process -- which
#: pushed the runner battery's `ru_maxrss` past rule 20's 1.0 GiB bar and
#: made a FIXTURE day refuse as HEAVY WITHOUT THE LOCK. `ru_maxrss` is a
#: high-water mark for the PROCESS (DE_PROCEDURE section 7), so a cache
#: that never frees is not a cache, it is a leak with a good reason. It is
#: now created PER CALL and dies with the payload.


# --------------------------------------------------------------------------
# 1. THE BOOK'S BYTES, RESOLVED BY DIGEST -- NEVER ASSUMED
# --------------------------------------------------------------------------
def book_bytes_for(rel_path: str, declared: str, *,
                   repo_root: Path | None = None) -> dict:
    """The bytes whose sha256 is `declared`, found in git history.

    The receipt records a DIGEST, not the source. Every honest answer about
    what changed needs the other side of the comparison, so it is resolved
    the only way it can be: search the blobs git holds for this path and
    match on the digest the receipt itself declares. A prefix is accepted
    because the refusal's own `differ` rows carry prefixes, and a match is
    re-hashed in full before it is returned.

    NOT FOUND IS A STATUS, NOT AN ANSWER. If no blob matches, the caller
    reports the intersection as UNKNOWN -- an unresolvable digest must
    never read as "nothing changed on the path"."""
    if not rel_path or not isinstance(rel_path, str):
        raise DeltaRefused(
            "REFUSED SCORING_DELTA_NO_PATH: a module path is the subject of "
            "the search; without one there is nothing to resolve.")
    declared = str(declared or "")
    if len(declared) < 12 or any(c not in "0123456789abcdef"
                                 for c in declared.lower()):
        raise DeltaRefused(
            f"REFUSED SCORING_DELTA_DIGEST_MALFORMED: {declared!r} is not a "
            f"hex digest of at least 12 characters. A short or malformed "
            f"digest would match many blobs and the first match would be "
            f"reported as THE book's bytes.")
    root = Path(repo_root) if repo_root else _repo_root_of(HERE)
    t0 = time.time()  # noqa: F841 -- kept: every exit reports elapsed_s
    try:
        log = subprocess.run(
            ["git", "-C", str(root), "log", "--format=%H", "--raw",
             "--abbrev=40", "--all", "--", rel_path],
            capture_output=True, text=True, check=True).stdout
    except Exception as e:                                   # noqa: BLE001
        return {"resolved": False,
                "status": "GIT_LOG_FAILED",
                "why": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t0, 3)}
    blobs: list = []
    for line in log.splitlines():
        if line.startswith(":"):
            parts = line.split()
            if len(parts) >= 4:
                blobs += [parts[2], parts[3]]
    blobs = [b for b in dict.fromkeys(blobs)
             if len(b) == 40 and set(b) != {"0"}]
    if not blobs:
        return {"resolved": False,
                "status": "NO_BLOBS_FOR_THIS_PATH",
                "why": (f"git holds no blob for {rel_path}; an untracked or "
                        f"renamed file cannot be resolved by digest here"),
                "elapsed_s": round(time.time() - t0, 3)}
    try:
        cat = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"],
                             input=("\n".join(blobs) + "\n").encode(),
                             capture_output=True, check=True).stdout
    except Exception as e:                                   # noqa: BLE001
        return {"resolved": False, "status": "GIT_CAT_FILE_FAILED",
                "why": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t0, 3)}
    i, n_hashed, hit = 0, 0, None
    while i < len(cat):
        j = cat.index(b"\n", i)
        hdr = cat[i:j].split()
        if len(hdr) < 3:
            break
        size = int(hdr[2])
        body = cat[j + 1:j + 1 + size]
        full = hashlib.sha256(body).hexdigest()
        n_hashed += 1
        if full.startswith(declared.lower()[:64]):
            hit = (hdr[0].decode(), full, body)
            break
        i = j + 1 + size + 1
    if hit is None:
        return {"resolved": False,
                "status": "DIGEST_NOT_IN_GIT_HISTORY",
                "why": (f"none of the {n_hashed} blobs git holds for "
                        f"{rel_path} hashes to {declared[:16]}. The book "
                        f"names bytes this repository cannot produce, so "
                        f"WHAT CHANGED CANNOT BE COMPUTED -- which is a "
                        f"weaker position than a refusal, not a stronger "
                        f"one"),
                "n_blobs_searched": n_hashed,
                "elapsed_s": round(time.time() - t0, 3)}
    blob, full, body = hit
    commits = _commits_holding(root, rel_path, blob)
    return {"resolved": True,
            "status": "RESOLVED_FROM_GIT_BY_DIGEST",
            "git_blob": blob,
            "sha256": full,
            "n_bytes": len(body),
            "n_blobs_searched": n_hashed,
            "n_blobs_known_for_the_path": len(blobs),
            "commits_carrying_these_bytes": commits,
            "how": ("every blob git holds for this path, hashed with "
                    "sha256 and matched against the digest the RECEIPT "
                    "declares -- the receipt is the authority for which "
                    "bytes ran, and this only finds them"),
            "bytes": body,
            "elapsed_s": round(time.time() - t0, 3)}


def _repo_root_of(start: Path) -> Path:
    """The git root above `start`, or `start` if there is none."""
    try:
        out = subprocess.run(["git", "-C", str(start), "rev-parse",
                              "--show-toplevel"], capture_output=True,
                             text=True, check=True).stdout.strip()
        return Path(out) if out else start
    except Exception:                                        # noqa: BLE001
        return start


def _commits_holding(root: Path, rel_path: str, blob: str) -> list:
    """Commits whose tree carries `blob` at `rel_path` -- provenance."""
    out = []
    try:
        log = subprocess.run(
            ["git", "-C", str(root), "log", "--format=%H %cI %s", "--raw",
             "--abbrev=40", "--all", "--", rel_path],
            capture_output=True, text=True, check=True).stdout
    except Exception:                                        # noqa: BLE001
        return out
    cur = None
    for line in log.splitlines():
        if line and not line.startswith(":") and not line.startswith(" "):
            cur = line
        elif line.startswith(":") and cur:
            parts = line.split()
            if len(parts) >= 4 and parts[3] == blob:
                h, rest = cur.split(" ", 1)
                out.append({"commit": h[:12], "when": rest.split(" ", 1)[0],
                            "subject": rest.split(" ", 1)[1][:90]})
    return out[:8]


# --------------------------------------------------------------------------
# 2. THE UNITS OF ONE MODULE: DEFINITIONS AND MODULE-LEVEL NAMES
# --------------------------------------------------------------------------
def _seg(lines: list, node) -> str:
    """A node's source segment, decorators included."""
    start = node.lineno
    for d in getattr(node, "decorator_list", []) or []:
        start = min(start, d.lineno)
    return "".join(lines[start - 1:node.end_lineno])


def units(src: str, cache: dict | None = None, *, tree=None) -> dict:
    """`{defs, module_level, other_top_level, class_methods}` for a module.

    `defs` is keyed by BARE NAME to match `be_producing_closure._functions`
    -- the two must share a key space or the intersection is meaningless.
    A name defined more than once is ONE unit whose digest covers every
    definition of it, which is the safe direction: a change to either marks
    the name.

    `other_top_level` digests the top-level statements that are NEITHER a
    definition NOR a simple assignment -- imports, `if`/`try` at module
    level, the docstring. It is taken from AST NODES rather than from the
    lines they do not cover, because a residual computed by subtraction
    counts every comment and blank line and would raise its warning on
    every diff, which is a warning nobody reads.

    `class_methods` exists so a CLASS can be placed on the path: BE's
    walker keys methods by bare name and never names a class, so a change
    to a class body that is not a method body would otherwise be invisible
    to the intersection. A class whose method is reachable is reachable."""
    tree = ast.parse(src) if tree is None else tree
    lines = src.splitlines(keepends=True)
    defs: dict = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.setdefault(n.name, []).append(_seg(lines, n))
    class_methods: dict = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            defs.setdefault(n.name, []).append(_seg(lines, n))
            class_methods[n.name] = sorted(
                c.name for c in ast.walk(n)
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)))
    mod_level: dict = {}
    other: list = []
    for n in ast.iter_child_nodes(tree):
        tgts = []
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    tgts.append(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            tgts.append(n.target.id)
        if tgts:
            seg = _seg(lines, n)
            for t in tgts:
                mod_level.setdefault(t, []).append(seg)
        elif not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
            other.append(_seg(lines, n))
    return {
        "defs": {k: _sha("".join(sorted(v))) for k, v in defs.items()},
        "module_level": {k: _sha("".join(sorted(v)))
                         for k, v in mod_level.items()},
        "class_methods": class_methods,
        "other_top_level_sha256": _sha("".join(other)),
        "n_other_top_level": len(other),
        "n_defs": len(defs), "n_module_level": len(mod_level)}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def changed_units(book_src: str | None, disk_src: str | None,
                  cache: dict | None = None, *,
                  book_units: dict | None = None,
                  disk_units: dict | None = None) -> dict:
    """What differs between two versions of one module, by unit."""
    b = book_units if book_units is not None else units(book_src, cache)
    d = disk_units if disk_units is not None else units(disk_src, cache)
    bd, dd = b["defs"], d["defs"]
    bm, dm = b["module_level"], d["module_level"]
    return {
        "defs_added": sorted(set(dd) - set(bd)),
        "defs_removed": sorted(set(bd) - set(dd)),
        "defs_modified": sorted(k for k in set(bd) & set(dd)
                                if bd[k] != dd[k]),
        "module_level_added": sorted(set(dm) - set(bm)),
        "module_level_removed": sorted(set(bm) - set(dm)),
        "module_level_modified": sorted(k for k in set(bm) & set(dm)
                                        if bm[k] != dm[k]),
        "other_top_level_differs":
            b["other_top_level_sha256"] != d["other_top_level_sha256"],
        "n_defs_book": b["n_defs"], "n_defs_disk": d["n_defs"],
        "n_module_level_book": b["n_module_level"],
        "n_module_level_disk": d["n_module_level"],
        "_book": b, "_disk": d}


# --------------------------------------------------------------------------
# 3. REACHABILITY, ONE LEVEL FINER THAN BE'S -- AND CROSS-CHECKED AGAINST IT
# --------------------------------------------------------------------------
def reachable_functions(root: Path, closure: dict, seeds, *,
                        overrides: dict | None = None,
                        cache: dict | None = None,
                        units_for: set | None = None) -> dict:
    """`(module, function)` pairs reachable from `seeds`.

    THIS IS `be_producing_closure.reachable_modules` WITH ITS `seen_fn`
    KEPT. That walker already computes function-level reachability and
    returns only the module projection -- rule 28's shape, in the benign
    direction: the producer computes the evidence and the consumer throws
    it away. So the loop is repeated here to keep it, using BE's OWN
    primitives (`_module_aliases`, `_functions`, `_refs`), and
    `assert_agrees_with_BE` drives the projection back against BE's
    result. A second implementation that cannot disagree is the only kind
    worth having.

    `overrides` supplies a module's source directly, so the walk can run
    over THE BOOK'S bytes without materialising a tree."""
    root = Path(root)
    overrides = dict(overrides or {})
    cache = {} if cache is None else cache
    files = set(closure)
    units_for = set(units_for or ())
    out_units: dict = {}
    meta: dict = {}

    def load(mod: str):
        """COMPACT METADATA FOR ONE MODULE, AND THE TREE IS FREED HERE.

        The obvious implementation keeps every parsed module alive for the
        length of the walk. Measured on this tree that is ~0.17 GiB, and
        this payload is computed INSIDE the runner's battery, which must
        stay under rule 20's 1.0 GiB bar WITHOUT the heavy lock -- a
        fixture day refused as HEAVY at 1.01 GiB the first time these
        cells ran. So each module is parsed, reduced to the edges the walk
        actually follows, and dropped. `assert_agrees_with_BE` is what
        makes this restructuring safe: it drives the projection back
        against `be_producing_closure.reachable_modules`, which still does
        it the tree-holding way."""
        if mod in meta:
            return meta[mod]
        fn = f"{mod}.py"
        if fn not in files:
            meta[mod] = None
            return None
        if fn in overrides:
            src = overrides[fn]
        else:
            p = root / fn
            if not p.is_file():
                meta[mod] = None
                return None
            src = p.read_text()
        key = _sha(src)
        got = cache.get(key)
        if got is None:
            t = ast.parse(src)
            mod_names = {tg.id for n in ast.iter_child_nodes(t)
                         if isinstance(n, ast.Assign)
                         for tg in n.targets if isinstance(tg, ast.Name)}
            edges, reads = {}, {}
            for name, nodes in PC._functions(t).items():
                e, r = [], set()
                for node in nodes:
                    e.extend(PC._refs(node))
                    for nn in ast.walk(node):
                        if isinstance(nn, ast.Name) and nn.id in mod_names \
                                and isinstance(getattr(nn, "ctx", None),
                                               ast.Load):
                            r.add(nn.id)
                edges[name] = e
                reads[name] = sorted(r)
            got = {"aliases": PC._module_aliases(t), "edges": edges,
                   "reads": reads, "names": set(edges),
                   # THE UNIT DIGESTS COME OFF THIS SAME PARSE when the
                   # caller wants them. CPython does not hand freed arenas
                   # back to the OS, so every extra parse of an 875 KB
                   # module RATCHETS `ru_maxrss` whether or not the tree is
                   # freed -- and this payload runs inside a battery with
                   # ~186 MB of headroom under rule 20's bar. Four parses
                   # of the runner cost 154 MB; two cost what is measured
                   # in the falsifier.
                   "units": (units(src, tree=t) if fn in units_for
                             else None)}
            del t                       # the tree dies with this frame
            cache[key] = got
        if got.get("units") is not None:
            out_units[fn] = got["units"]
        meta[mod] = got
        return got

    seen_mod: dict = {}
    seen_fn: set = set()
    reads: dict = {}
    stack = []
    for m, f in seeds:
        if load(m) is None:
            raise DeltaRefused(
                f"REFUSED SCORING_DELTA_SEED_NOT_PRESENT: {m}.py is a "
                f"declared scoring entry point and is not in the set being "
                f"walked. A derivation seeded outside its own evidence "
                f"cannot be checked.")
        stack.append((m, f))
        seen_mod.setdefault(f"{m}.py", []).append(f"SEED:{m}.{f}")
    while stack:
        mod, fn = stack.pop()
        if (mod, fn) in seen_fn:
            continue
        seen_fn.add((mod, fn))
        md = load(mod)
        if md is None:
            continue
        mod_alias, name_alias = md["aliases"]
        for _n in md["reads"].get(fn, []):
            reads.setdefault(f"{mod}.py", {}).setdefault(
                _n, []).append(fn)
        for alias, attr, kind in md["edges"].get(fn, []):
            if alias is None:
                tgt = name_alias.get(attr)
                if tgt is not None:
                    tmod, tfn = tgt
                elif attr in md["names"]:
                    tmod, tfn = mod, attr
                else:
                    continue
            else:
                tmod = mod_alias.get(alias)
                if tmod is None:
                    continue
                tfn = attr
            if load(tmod) is None:
                continue
            seen_mod.setdefault(f"{tmod}.py", []).append(
                f"{mod}.{fn} -{kind}-> {tmod}.{tfn}")
            if kind == "CALL" and (tmod, tfn) not in seen_fn:
                stack.append((tmod, tfn))
    # A WANTED MODULE THE WALK NEVER REACHED still needs its units: an
    # unreachable module is exactly the case where the intersection must
    # be reported as empty FOR A STATED REASON, and that needs the delta.
    for _fn in sorted(units_for - set(out_units)):
        _p = root / _fn
        _src = overrides.get(_fn)
        if _src is None and _p.is_file():
            _src = _p.read_text()
        if _src is not None:
            out_units[_fn] = units(_src)
    return {"functions": sorted(f"{m}.{f}" for m, f in seen_fn),
            "units": out_units,
            "by_module": _by_module(seen_fn),
            "module_level_reads": {k: {n: sorted(set(v))
                                       for n, v in sorted(d.items())}
                                   for k, d in sorted(reads.items())},
            "modules": sorted(seen_mod),
            "trails": seen_mod}


def _by_module(seen_fn: set) -> dict:
    out: dict = {}
    for m, f in seen_fn:
        out.setdefault(f"{m}.py", []).append(f)
    return {k: sorted(v) for k, v in sorted(out.items())}


def assert_agrees_with_BE(root: Path, closure: dict, seeds, *,
                          mine: list | None = None) -> dict:
    """The projection of this walk equals BE's own module walk, or REFUSE.

    The duplication above is the risk this pays for: a second walker that
    drifted from BE's would report a path that BE's module set does not
    agree with, and nothing would say so."""
    mine = (list(mine) if mine is not None
            else reachable_functions(root, closure, seeds)["modules"])
    theirs = sorted(PC.reachable_modules(Path(root), closure, seeds))
    if mine != theirs:
        raise DeltaRefused(
            f"REFUSED SCORING_DELTA_WALK_DISAGREES_WITH_BE: the "
            f"function-level walk projects to {mine} and "
            f"`be_producing_closure.reachable_modules` gives {theirs}. The "
            f"finer walk must be the same operation as the coarser one.")
    return {"agrees": True, "n_modules": len(mine), "modules": mine}


# --------------------------------------------------------------------------
# 4. THE PAYLOAD
# --------------------------------------------------------------------------
def delta(differ, *, root: Path | None = None,
          repo_root: Path | None = None,
          seeds=None,
          path_prefix: str = "live/pm_research",
          _disk_override: dict | None = None) -> dict:
    """The evidence a `BOOK_BUILT_BY_DIFFERENT_SCORING_CODE` refusal owes.

    `differ` is the refusal's own list of `{module, declared, actual}`
    rows -- the payload is ABOUT the modules that moved, and computing it
    for anything else would answer a question nobody asked."""
    t0 = time.time()
    if not differ:
        raise DeltaRefused(
            "REFUSED SCORING_DELTA_NOTHING_DIFFERS: this payload exists to "
            "explain a refusal; with no differing module there is no "
            "refusal to explain, and an empty answer here would read as "
            "'nothing on the path changed'.")
    root = Path(root) if root else HERE
    seeds = tuple(seeds) if seeds else PC.SCORING_ENTRY_POINTS
    on_disk = {q.name: "ON_DISK" for q in Path(root).glob("*.py")}
    if not on_disk:
        raise DeltaRefused(
            f"REFUSED SCORING_DELTA_NO_MODULES_ON_DISK: {root} holds no .py "
            f"file, so a reachability walk there would reach nothing and "
            f"report every change as off the path.")
    # THE WALK IS OVER THE TREE, NOT OVER THE RECEIPT'S CLOSURE. BE 122
    # measured the receipt-scoped walk TRUNCATING (8 against 12): the
    # closure is a whitelist, so a module the recording does not name
    # halts the walk there. A truncated walk would SHRINK the reachable
    # set and bias this payload toward "nothing that matters moved" --
    # the one direction it must never be wrong in.
    # EACH WALK OWNS ITS CACHE, ON PURPOSE. Sharing one cache across the
    # disk walk and the book walk saved ~0.6 s and RAISED THE PEAK by
    # holding both sets of trees live at once -- and this payload runs
    # inside a battery that must stay under rule 20's 1.0 GiB bar without
    # the heavy lock. Measured: 0.191 GiB shared against 0.10 GiB
    # separate. Time is the cheaper resource here.
    _want = {(r or {}).get("module") for r in differ
             if (r or {}).get("module")}
    disk_reach = reachable_functions(root, on_disk, seeds, units_for=_want)
    agree = assert_agrees_with_BE(root, on_disk, seeds,
                                  mine=disk_reach["modules"])
    out_modules: dict = {}
    known = True
    for row in differ:
        name = (row or {}).get("module")
        declared = (row or {}).get("declared")
        actual = (row or {}).get("actual")
        rel = f"{path_prefix}/{name}" if path_prefix else str(name)
        entry: dict = {"book_digest": str(declared)[:16],
                       "disk_digest": str(actual)[:16]}
        reach_disk = disk_reach["by_module"].get(name, [])
        reads_disk = disk_reach["module_level_reads"].get(name, {})
        entry["reachable_on_disk"] = {"n": len(reach_disk),
                                      "defs": reach_disk,
                                      "module_level_names_read":
                                          sorted(reads_disk)}
        entry["how_the_module_is_reached"] = \
            disk_reach["trails"].get(name, [])[:6]
        if str(actual) == "ABSENT":
            entry["status"] = "MODULE_ABSENT_FROM_DISK"
            entry["intersection_known"] = False
            entry["why"] = ("the module the book names is not on disk, so "
                            "there are no current bytes to compare against")
            known = False
            out_modules[name] = entry
            continue
        # THE REPO IS THE ONE CONTAINING THE ROOT UNDER EXAMINATION, not
        # the one this module happens to live in. A caller checking a tree
        # elsewhere (a fixture, a second worktree) would otherwise have its
        # book bytes searched for in the wrong history and get
        # NO_BLOBS_FOR_THIS_PATH -- an absence that means nothing.
        bb = book_bytes_for(rel, str(declared),
                            repo_root=(repo_root or _repo_root_of(root)))
        entry["book_bytes"] = {k: v for k, v in bb.items() if k != "bytes"}
        if not bb.get("resolved"):
            entry["status"] = "BOOK_BYTES_NOT_RESOLVABLE"
            entry["intersection_known"] = False
            entry["why"] = (
                "the bytes the book names cannot be recovered, so WHAT "
                "CHANGED is not computable here. This is NOT an empty "
                "intersection: nothing was compared.")
            known = False
            out_modules[name] = entry
            continue
        book_src = bb["bytes"].decode()
        if _disk_override and name in _disk_override:
            disk_src = _disk_override[name]
        else:
            disk_src = (root / name).read_text()
        # Reachability in BOTH versions, unioned. A definition reachable in
        # either the code that scored the book or the code on disk is
        # decision-relevant; taking one side alone would under-report in
        # exactly the direction that flatters.
        book_reach = reachable_functions(
            root, on_disk, seeds, overrides={name: book_src},
            units_for={name})
        ch = changed_units(None, None,
                           book_units=book_reach["units"].get(name),
                           disk_units=disk_reach["units"].get(name))
        reach_book = book_reach["by_module"].get(name, [])
        reads_book = book_reach["module_level_reads"].get(name, {})
        reach = set(reach_disk) | set(reach_book)
        # A CLASS WHOSE METHOD IS ON THE PATH IS ON THE PATH. BE's walker
        # keys methods by bare name and never names the class, so a change
        # to a class BODY that is not a method body -- a class attribute,
        # a base class -- would otherwise be a change no intersection
        # could see. Under-reporting reachability is the one direction
        # this payload must never be wrong in.
        _classes = set()
        for _cm in (ch["_book"]["class_methods"], ch["_disk"]["class_methods"]):
            for _cls, _meths in _cm.items():
                if reach & set(_meths):
                    _classes.add(_cls)
        reach = sorted(reach | _classes)
        reads = sorted(set(reads_disk) | set(reads_book))
        changed_defs = set(ch["defs_added"]) | set(ch["defs_removed"]) \
            | set(ch["defs_modified"])
        changed_ml = set(ch["module_level_added"]) \
            | set(ch["module_level_removed"]) \
            | set(ch["module_level_modified"])
        hit_defs = sorted(changed_defs & set(reach))
        hit_ml = sorted(changed_ml & set(reads))
        entry["reachable_in_the_books_bytes"] = {"n": len(reach_book),
                                                 "defs": reach_book}
        entry["on_the_path"] = {
            "defs": reach, "n_defs": len(reach),
            "module_level_names_read_by_them": reads,
            "read_by": {n: sorted(set(reads_disk.get(n, []))
                                  | set(reads_book.get(n, [])))
                        for n in reads},
            "union_of_book_and_disk": True}
        entry["what_moved"] = {
            k: v for k, v in ch.items() if not k.startswith("_")}
        entry["what_moved"]["n_defs_changed"] = len(changed_defs)
        entry["what_moved"]["n_module_level_changed"] = len(changed_ml)
        entry["INTERSECTION"] = {
            "defs": hit_defs,
            "module_level_names": hit_ml,
            "n": len(hit_defs) + len(hit_ml),
            "is_empty": not (hit_defs or hit_ml),
            "operation": ("(reachable defs INTERSECT changed defs) UNION "
                          "(module-level names read by a reachable def "
                          "INTERSECT changed module-level names)")}
        entry["byte_identity_on_the_path"] = [
            {"unit": u, "kind": "def",
             "book_sha256": ch["_book"]["defs"].get(u, "ABSENT")[:16],
             "disk_sha256": ch["_disk"]["defs"].get(u, "ABSENT")[:16],
             "identical": (ch["_book"]["defs"].get(u)
                           == ch["_disk"]["defs"].get(u))}
            for u in reach] + [
            {"unit": u, "kind": "module_level_name",
             "book_sha256": ch["_book"]["module_level"].get(u, "ABSENT")[:16],
             "disk_sha256": ch["_disk"]["module_level"].get(u, "ABSENT")[:16],
             "identical": (ch["_book"]["module_level"].get(u)
                           == ch["_disk"]["module_level"].get(u))}
            for u in reads]
        entry["intersection_known"] = True
        entry["status"] = ("NOTHING_THAT_MOVED_IS_ON_THE_SCORING_PATH"
                           if entry["INTERSECTION"]["is_empty"]
                           else "A_CHANGE_IS_ON_THE_SCORING_PATH")
        if not reach:
            entry["note_on_reachability"] = (
                "NO definition of this module is reachable from the "
                "scoring entry points at all -- an empty intersection here "
                "is a statement about the MODULE, not about the changes")
        if ch["other_top_level_differs"]:
            entry["other_top_level_warning"] = (
                "TOP-LEVEL CODE THAT IS NEITHER A DEF NOR A SIMPLE "
                "ASSIGNMENT ALSO DIFFERS -- an import, a module-level "
                "statement, the docstring. It is not resolved onto the "
                "path, so the intersection above is not the whole delta")
        out_modules[name] = entry
    empties = [m for m, e in out_modules.items()
               if e.get("intersection_known") and e["INTERSECTION"]["is_empty"]]
    hits = [m for m, e in out_modules.items()
            if e.get("intersection_known")
            and not e["INTERSECTION"]["is_empty"]]
    unknown = [m for m, e in out_modules.items()
               if not e.get("intersection_known")]
    reading = {
        "all_intersections_known": known and not unknown,
        "modules_with_a_change_on_the_path": hits,
        "modules_where_nothing_that_moved_is_on_the_path": empties,
        "modules_where_it_could_not_be_computed": unknown}
    if unknown:
        reading["what_this_supports"] = (
            "NOTHING YET -- at least one module's delta could not be "
            "computed, and an uncomputed delta is not an empty one. "
            "Resolve it or treat the book as unexplained.")
    elif hits:
        reading["what_this_supports"] = (
            "A REBUILD IS SUBSTANTIVE: code that can change this book's "
            "scores has moved, so an existing book read with the current "
            "code would be read by different scoring logic than produced "
            "it.")
    else:
        reading["what_this_supports"] = (
            "A REBUILD WOULD BUY PROVENANCE, NOT CORRECTNESS: nothing "
            "that moved is reachable from the scoring entry points, so no "
            "moved byte can change this book's scores -- SUBJECT TO THE "
            "LIMITS BELOW, which is why this is evidence for a decision "
            "and not the decision.")
    return {"WHAT_THIS_IS": NOT_A_LICENCE,
            "the_run_still_refuses": True,
            "entry_points": [f"{m}.{f}" for m, f in seeds],
            "entry_points_source": ("be_producing_closure."
                                    "SCORING_ENTRY_POINTS -- the builder's "
                                    "own producer calls, not a list typed "
                                    "here"),
            "walk_scope": ("THE TREE ON DISK, not the receipt's closure -- "
                           "a whitelisted walk truncates (BE 122 measured "
                           "8 against 12) and would shrink the path"),
            "walk_agrees_with_be_producing_closure": agree,
            "digest_convention": (
                "sha256 of a unit's SOURCE SEGMENT -- for a def, its lines "
                "INCLUDING decorators; for a name defined more than once, "
                "the sorted concatenation of every segment. THESE DIGESTS "
                "ARE NOT COMPARABLE WITH ANOTHER IMPLEMENTATION'S unless "
                "it uses the same convention; what travels between "
                "implementations is the IDENTITY VERDICT, not the number"),
            "modules": out_modules,
            "READING": reading,
            "LIMITS": list(LIMITS),
            "elapsed_s": round(time.time() - t0, 3)}


# --------------------------------------------------------------------------
# 5. THE WAIVER, AS A PREDICATE
# --------------------------------------------------------------------------
#: The token a CALLER must type to ask for the waiver. Asking is a human
#: act and is recorded; GRANTING is the predicate's, and no amount of
#: asking makes an unavailable waiver available.
WAIVER_TOKEN = "ACCEPT_SCORING_PATH_UNCHANGED_WAIVER"

#: REV 146's caution, which travels into the receipt with every waiver.
WAIVER_CAUTION = (
    "THE BOOK-CODE PREDICATE REFUSES ON THIS BOOK, BY NAME "
    "(BOOK_BUILT_BY_DIFFERENT_SCORING_CODE), AND THIS WAIVER IS A DECISION "
    "TO OVERRIDE A CHECK THAT IS FIRING CORRECTLY. It is not a check that "
    "is wrong; it is one that is right, overridden on evidence -- the "
    "reachable closure from the scoring entry points, an EMPTY "
    "intersection with what changed, and the byte-identity of every unit "
    "on that path. A reader who wants to disagree with the decision has "
    "everything needed to, in this block.")

WAIVER_NOT_AVAILABLE = "BOOK_SCORING_WAIVER_NOT_AVAILABLE"


def waiver_available(payload: dict) -> dict:
    """MAY this refusal be waived? EVERY condition is computed here.

    The USER's ruling was that the waiver be a PREDICATE and not a
    judgement -- the same shape as the `MATCHES_WITH_UNNAMED_MEMBERS`
    ruling. So a human types the token to ASK; this decides. A waiver a
    human can assert without the predicate holding is not a waiver, it is
    a bypass, and the difference is that this function can say NO.

    CONDITION (d) IS STRICTER THAN THE RULING ASKED FOR, deliberately.
    The ruling named byte-identity on the path and an empty intersection.
    A changed top-level statement -- an import rebinding a name behind an
    unchanged call site -- would satisfy both and still change what the
    path DOES, and it is already named in this module's LIMITS. It refuses
    here instead of being listed as a caveat under a green."""
    conds = []

    def cond(name, holds, evidence):
        conds.append({"condition": name, "holds": bool(holds),
                      "evidence": evidence})
        return bool(holds)

    mods = (payload or {}).get("modules") or {}
    ok_a = cond(
        "every differing module's delta was COMPUTED",
        bool(mods) and all(m.get("intersection_known") for m in mods.values()),
        {k: m.get("status") for k, m in mods.items()})
    ok_b = cond(
        "every intersection is EMPTY",
        bool(mods) and all((m.get("INTERSECTION") or {}).get("is_empty")
                           for m in mods.values() if m.get("intersection_known")),
        {k: (m.get("INTERSECTION") or {}).get("defs", "NOT_COMPUTED")
         for k, m in mods.items()})
    _units = [(k, u) for k, m in mods.items()
              for u in (m.get("byte_identity_on_the_path") or [])]
    ok_c = cond(
        "every unit ON THE PATH is BYTE-IDENTICAL between book and disk",
        bool(mods) and all(u["identical"] for _, u in _units),
        [{"module": k, "unit": u["unit"], "kind": u["kind"],
          "book": u["book_sha256"], "disk": u["disk_sha256"],
          "identical": u["identical"]} for k, u in _units])
    ok_d = cond(
        "no module-level statement outside defs and assignments differs",
        all(not (m.get("what_moved") or {}).get("other_top_level_differs")
            for m in mods.values() if m.get("intersection_known")),
        {k: (m.get("what_moved") or {}).get("other_top_level_differs")
         for k, m in mods.items() if m.get("intersection_known")})
    ok_e = cond(
        "the function-level walk AGREES with be_producing_closure",
        ((payload or {}).get("walk_agrees_with_be_producing_closure") or {})
        .get("agrees") is True,
        (payload or {}).get("walk_agrees_with_be_producing_closure"))
    available = all([ok_a, ok_b, ok_c, ok_d, ok_e])
    return {
        "available": available,
        "token_a_caller_must_supply": WAIVER_TOKEN,
        "conditions": conds,
        "why_not": [c["condition"] for c in conds if not c["holds"]],
        "CAUTION": WAIVER_CAUTION,
        "what_it_does_NOT_establish": (
            "that the book and the code agree -- they do not, which is why "
            "the predicate refuses. It establishes that nothing which "
            "differs can reach the scores, SUBJECT TO THIS MODULE'S "
            "LIMITS, and those limits are the reason this is a decision "
            "with evidence rather than a check that passed."),
        "LIMITS": list(LIMITS)}


# --------------------------------------------------------------------------
# 5. THE FALSIFIER -- BOTH SIDES, EACH ANCHORED
# --------------------------------------------------------------------------
# REV has caught a one-sided check tonight and swept six others for the
# shape, so the condition on this instrument is explicit: **a change ON the
# scoring path must appear in the intersection, and a change OFF it must
# NOT -- driven separately.** The off-path cells are the ones that can pass
# for the wrong reason (an instrument that sees nothing reports an empty
# intersection), so every off-path cell ALSO asserts that the change WAS
# detected in `what_moved`. An unanchored empty is not evidence.
_FX_V1 = {
    "fx_seed.py": '''"""synthetic: the producer's own entry point."""
import fx_mid

CONST = 1
OFFCONST = 2


class Holder:
    ATTR = 10

    def method(self):
        return self.ATTR


def produce(x):
    h = Holder()
    return step(x) + h.method()


def step(x):
    return fx_mid.helper(x) + CONST


def offpath(x):
    return OFFCONST + x
''',
    "fx_mid.py": '''"""synthetic: reached only through fx_seed.step."""


class Mid:
    ATTR = 10

    def helper(self, x):
        return x + Mid.ATTR


def never_called(x):
    return x - 1
''',
    "fx_off.py": '''"""synthetic: reached from nothing."""


def lonely(x):
    return x
''',
}
_FX_SEEDS = (("fx_seed", "produce"),)

#: A SECOND fixture tree, named for BE's REAL `SCORING_ENTRY_POINTS`, so a
#: caller can drive `assert_book_scoring_code` end to end over a tree that
#: costs nothing to walk. The runner's battery uses this: its cells were
#: measured at ~140 MB against ~186 MB of headroom under rule 20's 1.0 GiB
#: bar, and a battery one cell away from refusing every fixture day as
#: HEAVY is not a battery anyone can add to.
_FX_REAL = {
    "de_phase4_diag_runner.py": '''"""fixture: the builder's producer calls."""
import de_multiday_gate1_runner as R


def assemble_streaming(x):
    return R.ruled_day_set(x)


def build_tape_index(x):
    return x
''',
    "de_multiday_gate1_runner.py": '''"""fixture: the module that moved."""

PARAMS_REL = "params_v29.json"
OFF_CONST = 3


def ruled_day_set(x):
    return PARAMS_REL


def run_day(x):
    return OFF_CONST
''',
}


def fixture_tree(tmp: Path) -> Path:
    """The `_FX_REAL` tree, committed, for driving the predicate cheaply."""
    pkg = tmp / "live" / "pm_research"
    pkg.mkdir(parents=True, exist_ok=True)
    for name, src in _FX_REAL.items():
        (pkg / name).write_text(src)
    env = {"GIT_AUTHOR_NAME": "fx", "GIT_AUTHOR_EMAIL": "fx@x",
           "GIT_COMMITTER_NAME": "fx", "GIT_COMMITTER_EMAIL": "fx@x",
           "PATH": "/usr/bin:/bin", "HOME": str(tmp)}
    subprocess.run(["git", "init", "-q", str(tmp)], check=True, env=env)
    subprocess.run(["git", "-C", str(tmp), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(tmp), "commit", "-qm", "v1"],
                   check=True, env=env)
    return pkg


def _fx_repo(tmp: Path) -> Path:
    """A real git repo holding the fixture tree at v1, committed."""
    pkg = tmp / "live" / "pm_research"
    pkg.mkdir(parents=True, exist_ok=True)
    for name, src in _FX_V1.items():
        (pkg / name).write_text(src)
    env = {"GIT_AUTHOR_NAME": "fx", "GIT_AUTHOR_EMAIL": "fx@x",
           "GIT_COMMITTER_NAME": "fx", "GIT_COMMITTER_EMAIL": "fx@x",
           "PATH": "/usr/bin:/bin", "HOME": str(tmp)}
    subprocess.run(["git", "init", "-q", str(tmp)], check=True, env=env)
    subprocess.run(["git", "-C", str(tmp), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(tmp), "commit", "-qm", "v1"],
                   check=True, env=env)
    return pkg


def _fx_delta(pkg: Path, tmp: Path, module: str, mutate) -> dict:
    """Mutate one fixture module on disk and run the payload against v1."""
    p = pkg / module
    v1 = p.read_text()
    declared = hashlib.sha256(v1.encode()).hexdigest()
    p.write_text(mutate(v1))
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        return delta([{"module": module, "declared": declared,
                       "actual": actual}],
                     root=pkg, repo_root=tmp, seeds=_FX_SEEDS)
    finally:
        p.write_text(v1)


def falsify() -> int:                                        # noqa: C901
    """Drive every side. Returns 0 when all cells hold."""
    import tempfile
    fails = []

    def ok(cond, label):
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
        if not cond:
            fails.append(label)

    def refuses(fn, needle, label):
        try:
            fn()
        except DeltaRefused as e:
            ok(needle in str(e), f"{label} -- refuses {needle}")
        except Exception as e:                               # noqa: BLE001
            ok(False, f"{label} -- raised {type(e).__name__}: {e}")
        else:
            ok(False, f"{label} -- DID NOT REFUSE")

    print("[de_scoring_path_delta] falsifier")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pkg = _fx_repo(tmp)

        # ---- the fixture's own geometry, asserted before it is used ----
        reach = reachable_functions(
            pkg, {q.name: "ON_DISK" for q in pkg.glob("*.py")}, _FX_SEEDS)
        on_seed = reach["by_module"].get("fx_seed.py", [])
        ok(set(on_seed) == {"produce", "step"},
           f"FIXTURE: fx_seed's path is produce/step, got {on_seed}")
        ok("method" not in on_seed,
           "FIXTURE: `Holder.method` is called through an INSTANCE and is "
           "NOT followed -- BE's walker resolves `alias.attr` only for an "
           "imported module alias, and this cell exists so that limit is "
           "measured rather than assumed")
        ok(reach["by_module"].get("fx_mid.py") == ["helper"],
           "FIXTURE: fx_mid is reached at `helper` only")
        ok("fx_off.py" not in reach["by_module"],
           "FIXTURE: fx_off is on no path")
        ok(sorted(reach["module_level_reads"]
                  .get("fx_seed.py", {})) == ["CONST"],
           "FIXTURE: CONST is read on the path and OFFCONST is not")

        # ---- SIDE 1: a change ON the path APPEARS ----------------------
        d = _fx_delta(pkg, tmp, "fx_seed.py",
                      lambda s: s.replace("return fx_mid.helper(x) + CONST",
                                          "return fx_mid.helper(x) + CONST + 0"))
        m = d["modules"]["fx_seed.py"]
        ok("step" in m["what_moved"]["defs_modified"], "ON-def: anchored -- the change was detected")
        ok(m["INTERSECTION"]["defs"] == ["step"], "ON-def: `step` is IN the intersection")
        ok(m["status"] == "A_CHANGE_IS_ON_THE_SCORING_PATH", "ON-def: status names it")
        ok(d["READING"]["what_this_supports"].startswith("A REBUILD IS SUBSTANTIVE"),
           "ON-def: the reading supports a rebuild")

        d = _fx_delta(pkg, tmp, "fx_seed.py",
                      lambda s: s.replace("CONST = 1", "CONST = 99"))
        m = d["modules"]["fx_seed.py"]
        ok("CONST" in m["what_moved"]["module_level_modified"], "ON-const: anchored")
        ok(m["INTERSECTION"]["module_level_names"] == ["CONST"],
           "ON-const: `CONST` is IN the intersection -- a constant a reachable def reads is ON the path")

        d = _fx_delta(pkg, tmp, "fx_mid.py",
                      lambda s: s.replace("return x + Mid.ATTR",
                                          "return x + Mid.ATTR + 0"))
        m = d["modules"]["fx_mid.py"]
        ok("helper" in m["what_moved"]["defs_modified"], "ON-transitive: anchored")
        ok(m["INTERSECTION"]["defs"] == ["Mid", "helper"],
           "ON-transitive: a module reached only THROUGH another is on the "
           "path -- and the enclosing class travels with the method, which "
           "is the safe direction and is asserted rather than tolerated")

        d = _fx_delta(pkg, tmp, "fx_mid.py",
                      lambda s: s.replace("ATTR = 10", "ATTR = 11"))
        m = d["modules"]["fx_mid.py"]
        ok("Mid" in m["what_moved"]["defs_modified"], "ON-class: anchored")
        ok("Mid" in m["INTERSECTION"]["defs"],
           "ON-class: a class whose METHOD is reachable is on the path, so a "
           "class-BODY change (a class attribute, not a method) lands")

        # ---- SIDE 2: a change OFF the path does NOT appear -------------
        d = _fx_delta(pkg, tmp, "fx_seed.py",
                      lambda s: s.replace("return OFFCONST + x",
                                          "return OFFCONST + x + 1"))
        m = d["modules"]["fx_seed.py"]
        ok(m["what_moved"]["defs_modified"] == ["offpath"],
           "OFF-def: ANCHORED -- the instrument SAW the change")
        ok(m["INTERSECTION"]["is_empty"],
           "OFF-def: and correctly left it OUT of the intersection")
        ok(m["status"] == "NOTHING_THAT_MOVED_IS_ON_THE_SCORING_PATH",
           "OFF-def: status names it")

        d = _fx_delta(pkg, tmp, "fx_seed.py",
                      lambda s: s.replace("OFFCONST = 2", "OFFCONST = 3"))
        m = d["modules"]["fx_seed.py"]
        ok(m["what_moved"]["module_level_modified"] == ["OFFCONST"],
           "OFF-const: ANCHORED -- the change was detected")
        ok(m["INTERSECTION"]["is_empty"],
           "OFF-const: a constant no reachable def reads is OFF the path")

        d = _fx_delta(pkg, tmp, "fx_seed.py",
                      lambda s: s.replace("    ATTR = 10", "    ATTR = 12"))
        m = d["modules"]["fx_seed.py"]
        ok("Holder" in m["what_moved"]["defs_modified"], "OFF-class: ANCHORED")
        ok(m["INTERSECTION"]["is_empty"],
           "OFF-class: a class NO reachable name touches stays off the path")

        d = _fx_delta(pkg, tmp, "fx_off.py",
                      lambda s: s.replace("return x", "return x + 1"))
        m = d["modules"]["fx_off.py"]
        ok(m["what_moved"]["defs_modified"] == ["lonely"], "OFF-module: ANCHORED")
        ok(m["INTERSECTION"]["is_empty"] and "note_on_reachability" in m,
           "OFF-module: an unreachable MODULE says so, rather than reporting a bare empty")

        # ---- SIDE 3: NOT COMPUTED MUST NOT READ AS EMPTY ---------------
        d = delta([{"module": "fx_seed.py",
                    "declared": "deadbeefdeadbeefdeadbeef",
                    "actual": "0" * 64}],
                  root=pkg, repo_root=tmp, seeds=_FX_SEEDS)
        m = d["modules"]["fx_seed.py"]
        ok(m["status"] == "BOOK_BYTES_NOT_RESOLVABLE"
           and m["intersection_known"] is False and "INTERSECTION" not in m,
           "UNRESOLVED: no intersection is reported at all -- not an empty one")
        ok(d["READING"]["what_this_supports"].startswith("NOTHING YET"),
           "UNRESOLVED: the reading supports NOTHING, and says so")
        ok(d["READING"]["all_intersections_known"] is False,
           "UNRESOLVED: `all_intersections_known` is False")

        d = delta([{"module": "fx_gone.py", "declared": "a" * 64,
                    "actual": "ABSENT"}],
                  root=pkg, repo_root=tmp, seeds=_FX_SEEDS)
        m = d["modules"]["fx_gone.py"]
        ok(m["status"] == "MODULE_ABSENT_FROM_DISK"
           and m["intersection_known"] is False,
           "ABSENT: a module that is gone is UNKNOWN, never empty")

        # ---- SIDE 4: PARTIAL AND BAD INPUT REFUSE ----------------------
        refuses(lambda: delta([], root=pkg, repo_root=tmp, seeds=_FX_SEEDS),
                "SCORING_DELTA_NOTHING_DIFFERS", "no differing module")
        refuses(lambda: book_bytes_for("x.py", "short", repo_root=tmp),
                "SCORING_DELTA_DIGEST_MALFORMED", "a 5-character digest")
        refuses(lambda: book_bytes_for("x.py", "z" * 40, repo_root=tmp),
                "SCORING_DELTA_DIGEST_MALFORMED", "a non-hex digest")
        refuses(lambda: book_bytes_for("", "a" * 40, repo_root=tmp),
                "SCORING_DELTA_NO_PATH", "no module path")
        empty_root = tmp / "empty"
        empty_root.mkdir(exist_ok=True)
        refuses(lambda: delta([{"module": "fx_seed.py", "declared": "a" * 40,
                                "actual": "b" * 40}],
                              root=empty_root, repo_root=tmp,
                              seeds=_FX_SEEDS),
                "SCORING_DELTA_NO_MODULES_ON_DISK", "a root with no .py")
        refuses(lambda: reachable_functions(
                    pkg, {q.name: "ON_DISK" for q in pkg.glob("*.py")},
                    (("no_such_module", "produce"),)),
                "SCORING_DELTA_SEED_NOT_PRESENT", "a seed outside the set")

        # ---- SIDE 5: THE TWO WALKS MUST AGREE, AND SAY SO -------------
        clo = {q.name: "ON_DISK" for q in pkg.glob("*.py")}
        ok(assert_agrees_with_BE(pkg, clo, _FX_SEEDS)["agrees"],
           "AGREEMENT: the finer walk projects to BE's module set")
        refuses(lambda: assert_agrees_with_BE(pkg, clo, _FX_SEEDS,
                                              mine=["fx_off.py"]),
                "SCORING_DELTA_WALK_DISAGREES_WITH_BE",
                "a forged projection")

    # ---- SIDE 6: THE REAL THING (rule 33: pass on the real artifact) ---
    real = HERE / "de_multiday_gate1_runner.py"
    if real.is_file():
        try:
            # BOTH SIDES ARE THE BOOK'S COMMITTED BYTES. A self-comparison
            # against the WORKING TREE cannot resolve: the book side comes
            # from git by digest, and a seat's working copy mid-round is
            # not committed. That is the payload behaving correctly -- and
            # it is why a book built from a dirty tree is unresolvable and
            # therefore never waivable.
            _bb = book_bytes_for(
                "live/pm_research/de_multiday_gate1_runner.py",
                "b4532d00c9ac6c19")
            _bs = _bb["bytes"].decode()
            live = hashlib.sha256(_bb["bytes"]).hexdigest()
            d = delta([{"module": "de_multiday_gate1_runner.py",
                        "declared": live, "actual": live}], root=HERE,
                      _disk_override={"de_multiday_gate1_runner.py": _bs})
            m = d["modules"]["de_multiday_gate1_runner.py"]
            ok(m["intersection_known"] and m["INTERSECTION"]["is_empty"],
               "REAL: a module compared with ITSELF has an empty intersection")
            ok(m["on_the_path"]["defs"] == ["ruled_day_set"],
               f"REAL: the runner's scoring path is `ruled_day_set` alone, "
               f"got {m['on_the_path']['defs']}")
            ok(m["on_the_path"]["module_level_names_read_by_them"]
               == ["PARAMS_REL"],
               "REAL: `PARAMS_REL` is the module-level name that path reads")
        except Exception as e:                               # noqa: BLE001
            ok(False, f"REAL: raised {type(e).__name__}: {e}")

    # ---- SIDE 7: THE PAYLOAD NEVER SOFTENS THE REFUSAL ----------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pkg = _fx_repo(tmp)
        d = _fx_delta(pkg, tmp, "fx_seed.py",
                      lambda s: s.replace("return OFFCONST + x",
                                          "return OFFCONST + x + 1"))
        ok(d["the_run_still_refuses"] is True,
           "NOT A LICENCE: the payload declares the run still refuses")
        ok("NEVER A LICENCE" in d["WHAT_THIS_IS"],
           "NOT A LICENCE: it says so where a reader meets it")
        ok(len(d["LIMITS"]) >= 5 and any("LOWER BOUND" in x
                                         for x in d["LIMITS"]),
           "NOT A LICENCE: the lower-bound limit travels on every payload")

    # ---- SIDE 8: THE WAIVER SAYS NO, AND SAYS WHY ---------------------
    # The cells that matter here are the NEGATIVE ones: a waiver that
    # cannot refuse is a bypass with a longer name.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pkg = _fx_repo(tmp)

        d = _fx_delta(pkg, tmp, "fx_seed.py",
                      lambda s: s.replace("return fx_mid.helper(x) + CONST",
                                          "return fx_mid.helper(x) + CONST + 0"))
        w = waiver_available(d)
        ok(w["available"] is False
           and "every intersection is EMPTY" in w["why_not"],
           "WAIVER: an ON-PATH change is NOT waivable, and the condition "
           "that failed is named")

        d = _fx_delta(pkg, tmp, "fx_seed.py",
                      lambda s: s.replace('"""synthetic: the producer\'s own entry point."""',
                                          '"""synthetic: the producer\'s own entry point."""\nimport json'))
        m = d["modules"]["fx_seed.py"]
        w = waiver_available(d)
        ok(m["INTERSECTION"]["is_empty"]
           and m["what_moved"]["other_top_level_differs"] is True,
           "WAIVER: ANCHORED -- a bare `import json` leaves the "
           "intersection empty and moves the top level")
        ok(w["available"] is False
           and "no module-level statement outside defs and assignments "
               "differs" in w["why_not"],
           "WAIVER: and it REFUSES anyway -- an import can rebind a name "
           "behind an unchanged call, so condition (d) bites where the "
           "ruling's two conditions would both have passed")

        d = delta([{"module": "fx_seed.py",
                    "declared": "deadbeefdeadbeefdeadbeef",
                    "actual": "0" * 64}],
                  root=pkg, repo_root=tmp, seeds=_FX_SEEDS)
        w = waiver_available(d)
        ok(w["available"] is False
           and "every differing module's delta was COMPUTED" in w["why_not"],
           "WAIVER: an UNCOMPUTED delta is not a waivable one")

        d = _fx_delta(pkg, tmp, "fx_seed.py",
                      lambda s: s.replace("return OFFCONST + x",
                                          "return OFFCONST + x + 1"))
        w = waiver_available(d)
        ok(w["available"] is True and not w["why_not"],
           "WAIVER: an OFF-PATH change IS waivable -- the positive side")
        ok("OVERRIDE A CHECK THAT IS FIRING CORRECTLY" in w["CAUTION"],
           "WAIVER: REV's caution travels on the grant, not only on the "
           "refusal")

    ok(waiver_available({})["available"] is False,
       "WAIVER: an EMPTY payload is not a waiver -- absence of evidence "
       "never grants")

    # ---- SIDE 9: THE REAL DECISION THIS WAS BUILT FOR -----------------
    # The 09-03 EV21 book against the runner as it stands. IF THIS CELL
    # EVER GOES RED, THE SCORING PATH HAS MOVED and the waiver must not be
    # used on that book -- the correct response is a rebuild, never a
    # loosened cell.
    try:
        d = delta([{"module": "de_multiday_gate1_runner.py",
                    "declared": "b4532d00c9ac6c19",
                    "actual": hashlib.sha256(
                        (HERE / "de_multiday_gate1_runner.py").read_bytes()
                    ).hexdigest()}], root=HERE)
        m = d["modules"]["de_multiday_gate1_runner.py"]
        w = waiver_available(d)
        ok(m["on_the_path"]["defs"] == ["ruled_day_set"]
           and m["INTERSECTION"]["is_empty"],
           "EV21: the 09-03 book's path is `ruled_day_set` and nothing "
           "that moved touches it")
        ok(all(u["identical"] for u in m["byte_identity_on_the_path"]),
           "EV21: every unit on that path is byte-identical to disk")
        ok(w["available"] is True,
           "EV21: the waiver IS available on the real book -- if this goes "
           "red the scoring path moved; rebuild, do not loosen")
    except Exception as e:                                   # noqa: BLE001
        ok(False, f"EV21: raised {type(e).__name__}: {e}")

    print(f"[de_scoring_path_delta] "
          f"{'PASS' if not fails else 'FAIL'} -- {len(fails)} failing")
    for f in fails:
        print(f"    FAILED: {f}")
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print(__doc__)
    print("usage: de_scoring_path_delta.py --falsify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
