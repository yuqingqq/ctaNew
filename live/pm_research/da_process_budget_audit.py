#!/usr/bin/env python3
"""DA — THE PROCESS-BUDGET AND RULE-22 SWEEP (REV 53 section 2).

REV 53 killed a 1 h 24 min run and named what it had NOT established: whether
BE's or DA's producers carry the same shape. This is that sweep, over the
three heavy BE producers and this seat's own instruments.

THE SHAPE, stated as a predicate rather than a story. A budget refusal is
sound only when the MEASURED quantity and the BUDGETED scope are the same
scope. `getrusage(RUSAGE_SELF).ru_maxrss` is a PROCESS-WIDE HIGH-WATER: it
covers every allocation the process has ever made and it NEVER FALLS. A
budget that is narrower than the process -- one stage's, one fixture's --
compared against that number is not measuring the thing it names, and once
earlier work in the same process has raised the high-water the narrow check
can never pass again. That is what happened: a 700 MB fixture budget against
a 2,426 MB process peak set by the real day that ran first.

Three questions, per module, computed:

  (a) does a budget comparison read a PROCESS-WIDE high-water (ru_maxrss,
      cgroup memory.peak) rather than a per-stage or per-fixture DELTA?
  (b) is the budget's own scope NARROWER than the measurement's?
  (c) is a battery (selftest / fixture / day-path checks) invoked from the
      REAL path, in the same process, so that (a) and (b) can meet?

And rule 22 / R-605: does the producer capture, AT IMPORT, its own producing
code digest, the digest of its import closure under `live/`, and the
worktree HEAD -- or at EMIT time, which names the code that is on disk when
the receipt is written rather than the code that ran?

R-235: nothing here imports the modules it audits. Every finding is read
from the SOURCE by AST, at a named line, so a claim about code is checked
against the code. Read-only: this module opens no ledger, runs no producer
and takes no lock.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROTOCOL = "P003_DA_PROCESS_BUDGET_AND_RULE22_SWEEP_V1"
HERE = Path(__file__).resolve().parent
#: see da_root: importable as a script AND under `python3 -m`.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
#: THE TREE UNDER AUDIT. This seat executes from its own worktree, whose
#: copies of ANOTHER seat's files are whatever they were at the last sync --
#: auditing those would be a finding about the wrong copy of the file, the
#: R-601 class exactly (an address that names a different object). The
#: modules are read from the LEDGER tree, and every row carries the digest
#: it was read at plus whether that tree has it committed.
def _resolve_root() -> Path:
    """The tree under audit, through the ONE canonical predicate.

    `os.environ.get("PM_DATA_ROOT") or REPO` answered with THIS SEAT'S
    WORKTREE when the variable was unset -- this module's own probe caught
    it. A sweep of another seat's modules read from a worktree is a sweep
    of stale copies (DA 77 shipped exactly that)."""
    import da_root as _R                                      # noqa: PLC0415
    return Path(_R.require_canonical_root("the module sweep")["root"])


AUDIT_ROOT = _resolve_root()

#: THE MODULES UNDER AUDIT. BE's three heavy producers, the shared module
#: they all measure through, and this seat's own instruments. DE's runner is
#: carried as the KNOWN-BAD: it is the artifact that caused the finding, and
#: an instrument that cannot flag the case that motivated it is not an
#: instrument (rule 15).
BE_PRODUCERS = ("live/pm_research/be_gate1_fragment.py",
                "live/pm_research/be_gate1_state_tape.py",
                "live/pm_research/be_daybook_build.py")
BE_SHARED = ("live/pm_research/be_data_root.py",)
DA_INSTRUMENTS = ("live/pm_research/da_gate1_day_verdict.py",
                  "live/pm_research/da_race_read_verify.py",
                  "live/pm_research/da_book_verify.py",
                  "live/pm_research/da_accrual_report.py",
                  "live/mm_research/da_e2a_receipt_v2.py",
                  "live/mm_research/e2_a_runner.py",
                  "live/mm_research/e2_a_declare.py")
KNOWN_BAD = ("live/pm_research/de_multiday_gate1_runner.py",)

# ---------------------------------------------------------------- measures

#: A measurement's SCOPE is a property of where the number comes from, not
#: of what it is called. `peak_gb` is a name; `ru_maxrss` is a fact.
PROCESS_HIGHWATER = "PROCESS_HIGHWATER_never_falls"
PROCESS_CURRENT = "PROCESS_CURRENT_falls_on_release"
CGROUP_PEAK = "CGROUP_PEAK_never_falls"
CGROUP_CURRENT = "CGROUP_CURRENT_falls_on_release"


#: ONE entry, held BY IDENTITY. A dict keyed on `id(src)` would be a
#: correctness bug, not an optimisation: CPython reuses an id once the
#: string is collected, so a later module could read a previous module's
#: lines and every source segment in the receipt would name the wrong file.
_LINES_CACHE: list = [None, None]


def _seg(src: str, node: ast.AST) -> str:
    """The source of one node. `ast.get_source_segment` re-splits the whole
    file per call, which is 60 s on a 5,800-line runner; the lines are split
    ONCE here."""
    if _LINES_CACHE[0] is not src:
        _LINES_CACHE[0] = src
        _LINES_CACHE[1] = src.splitlines(keepends=True)
    lines = _LINES_CACHE[1]
    if lines is None:
        lines = src.splitlines(keepends=True)
    a = getattr(node, "lineno", None)
    b = getattr(node, "end_lineno", None)
    if a is None or b is None:
        return ""
    try:
        if a == b:
            return lines[a - 1][node.col_offset:node.end_col_offset]
        out = [lines[a - 1][node.col_offset:]]
        out += lines[a:b - 1]
        out.append(lines[b - 1][:node.end_col_offset])
        return "".join(out)
    except Exception:                                         # noqa: BLE001
        return ""


def measurement_sites(tree: ast.AST, src: str) -> list:
    """Every place this module ASKS THE KERNEL for memory, with its scope."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "ru_maxrss":
            out.append({"line": n.lineno, "kind": PROCESS_HIGHWATER,
                        "expr": _seg(src, n),
                        "why": ("getrusage(RUSAGE_SELF).ru_maxrss is the "
                                "high-water of the WHOLE PROCESS and never "
                                "falls")})
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            v = n.value
            if "statm" in v or "VmRSS" in v:
                out.append({"line": n.lineno, "kind": PROCESS_CURRENT,
                            "expr": repr(v),
                            "why": ("current resident set: it FALLS when "
                                    "memory is released")})
            elif "memory.peak" in v or "max_usage_in_bytes" in v:
                out.append({"line": n.lineno, "kind": CGROUP_PEAK,
                            "expr": repr(v),
                            "why": "the cgroup's high-water; never falls"})
            elif "memory.current" in v:
                out.append({"line": n.lineno, "kind": CGROUP_CURRENT,
                            "expr": repr(v), "why": "the cgroup's current"})
    return out


def measuring_functions(tree: ast.AST, src: str) -> dict:
    """fname -> the SCOPE of the number it hands back."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kinds = {m["kind"] for m in measurement_sites(n, src)}
            if kinds:
                #: a function reading both reports the WIDEST scope it can
                #: return -- the conservative reading, because a checker
                #: that under-states a scope under-states the risk.
                for k in (PROCESS_HIGHWATER, CGROUP_PEAK, PROCESS_CURRENT,
                          CGROUP_CURRENT):
                    if k in kinds:
                        out[n.name] = k
                        break
    return out


# ---------------------------------------------------------------- budgets

FIXTURE_RE = re.compile(r"fixture", re.I)
STAGE_RE = re.compile(r"stage", re.I)
PROCESS_CAP_RE = re.compile(r"(MEM|RSS|MEMORY|DAY)_?.*CAP|CAP_GI?B|"
                            r"MEMORY_?MAX", re.I)
BUDGET_WORD_RE = re.compile(r"budget|cap|limit|quota|ceiling|bar", re.I)
#: `growth_gb` and `delta_gb` are measured names too -- and they are the
#: names the REPAIR wears. Round 77's list had only the high-water words,
#: so when BE 60 landed `growth_gb = peak - baseline` the census could not
#: see the fix arrive: it read 0 mismatches AND 0 deltas, which says
#: nothing at all.
MEASURED_KEY_RE = re.compile(
    r"peak|rss|maxrss|highwater|mem|growth|delta|baseline", re.I)

FIXTURE_SCOPE = "FIXTURE_scope"
STAGE_SCOPE = "STAGE_scope"
PROCESS_SCOPE = "PROCESS_scope"
UNKNOWN_SCOPE = "SCOPE_UNDETERMINED"


def module_constants(tree: ast.AST) -> dict:
    """Module-level names -> their literal value (numbers and dicts only)."""
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception:                                 # noqa: BLE001
                out[n.targets[0].id] = None
    return out


def fixture_selected_budgets(tree: ast.AST, src: str) -> list:
    """Budget values CHOSEN BY A FIXTURE FLAG.

    `_Stages(FIXTURE_STAGE_BUDGETS_GB if fixture else STAGE_BUDGETS_GB)` is
    the shape: the same comparison serves two scopes and only one of them is
    the process's."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.IfExp):
            t = _seg(src, n.test)
            branches = _seg(src, n.body) + "|" + _seg(src, n.orelse)
            if FIXTURE_RE.search(t or "") and BUDGET_WORD_RE.search(branches):
                out.append({"line": n.lineno, "expr": _seg(src, n),
                            "test": t})
    return out


def _resolve_scope(name: str, consts: dict, fixture_sel: list) -> str:
    if not name:
        return UNKNOWN_SCOPE
    if FIXTURE_RE.search(name):
        return FIXTURE_SCOPE
    if STAGE_RE.search(name):
        return STAGE_SCOPE
    if PROCESS_CAP_RE.search(name):
        return PROCESS_SCOPE
    return UNKNOWN_SCOPE


def _dict_key_scopes(fn: ast.AST, src: str, meas: dict,
                     assigns: dict | None = None) -> dict:
    """Keys BOUND TO A MEASUREMENT inside this function.

    `row = {"peak_gb": _rss_gb()}` binds `peak_gb` to a process high-water,
    so `row["peak_gb"] <= b` two lines later IS that measurement. The
    binding is FUNCTION-SCOPED on purpose: a key of the same name parsed out
    of a declaration file elsewhere is not this measurement, and resolving
    it as one would invent a finding."""
    out = {}
    for n in ast.walk(fn):
        if not isinstance(n, ast.Dict):
            continue
        for k, v in zip(n.keys, n.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                continue
            sc, dl, _ = _measured_scope_of(v, src, meas, assigns or {}, {})
            if sc and sc not in (UNKNOWN_SCOPE, "LOOKALIKE"):
                #: THE DELTA FLAG TRAVELS WITH THE KEY. Keeping only the
                #: scope made `{"peak_gb": _rss_gb() - base}` -- the delta
                #: form, which is the repair -- read as a raw high-water at
                #: the comparison, so the instrument would have reported the
                #: fix as the defect.
                out[k.value] = (sc, dl)
    return out


def _measured_scope_of(node: ast.AST, src: str, meas: dict,
                       assigns: dict, key_scopes: dict) -> tuple:
    """(scope, is_delta, how) for one side of a comparison."""
    if node is None:
        return None, False, ""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        l, _, _ = _measured_scope_of(node.left, src, meas, assigns,
                                     key_scopes)
        r, _, _ = _measured_scope_of(node.right, src, meas, assigns,
                                     key_scopes)
        if l or r:
            return (l or r), True, "a DELTA of two measurements"
        return None, False, ""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in meas:
            return meas[node.func.id], False, f"calls {node.func.id}()"
        if node.func.id in ("round", "float", "int", "abs") and node.args:
            #: a DELTA wrapped in `round(...)` is still a delta. BE 60's
            #: repair is literally `round(peak - self.baseline_gb, 3)`.
            return _measured_scope_of(node.args[0], src, meas, assigns,
                                      key_scopes)
        if node.func.id in ("max", "min", "sum") and node.args:
            a = node.args[0]
            inner = (a.elt if isinstance(a, (ast.GeneratorExp, ast.ListComp))
                     else a)
            sc, dl, how = _measured_scope_of(inner, src, meas, assigns,
                                             key_scopes)
            if sc and sc != "LOOKALIKE":
                return sc, dl, f"{node.func.id}() over {how}"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr in meas:
        return meas[node.func.attr], False, f"calls .{node.func.attr}()"
    if isinstance(node, ast.Name) and node.id in assigns:
        return assigns[node.id][0], assigns[node.id][1], \
            f"`{node.id}` assigned from {assigns[node.id][2]}"
    if isinstance(node, ast.Subscript) and isinstance(node.slice,
                                                     ast.Constant):
        key = node.slice.value
        if isinstance(key, str) and key in key_scopes:
            sc, dl = key_scopes[key]
            return sc, dl, (
                f"key {key!r}, BOUND to a "
                f"{'DELTA of two measurements' if dl else 'measurement'} "
                f"in this function")
        if isinstance(key, str) and MEASURED_KEY_RE.search(key):
            return "LOOKALIKE", False, (
                f"key {key!r} LOOKS measured but is bound to no measurement "
                f"in this function -- NOT resolved as one")
    return None, False, ""


_BUDGET_ASSIGN_CACHE: dict = {}


def _budget_scope_local(node: ast.AST, src: str, fn: ast.AST) -> str:
    """A budget NAME resolved through its LOCAL assignments, up to 3 hops.

    `b = budgets["A0"]`, `budgets = FIXTURE_STAGE_BUDGETS_GB if fixture else
    STAGE_BUDGETS_GB` -- neither line names a scope on its own, and the pair
    names both. This is the shape the smoke died on, and reading only the
    immediate name would have missed it."""
    if not isinstance(node, ast.Name):
        return ""
    if id(fn) not in _BUDGET_ASSIGN_CACHE:
        m = {}
        for a in ast.walk(fn):
            if isinstance(a, ast.Assign) and len(a.targets) == 1 \
                    and isinstance(a.targets[0], ast.Name):
                v = a.value
                names = [n2.id for n2 in ast.walk(v)
                         if isinstance(n2, ast.Name)]
                if isinstance(v, ast.IfExp):
                    names.append(_seg(src, v.test) or "")
                m.setdefault(a.targets[0].id, []).extend(names)
        _BUDGET_ASSIGN_CACHE[id(fn)] = m
    table = _BUDGET_ASSIGN_CACHE[id(fn)]
    seen, frontier, best = {node.id}, [node.id], ""
    for _hop in range(3):
        nxt = []
        for cur in frontier:
            for nm in table.get(cur, []):
                if FIXTURE_RE.search(nm or ""):
                    return FIXTURE_SCOPE
                if STAGE_RE.search(nm or ""):
                    best = best or STAGE_SCOPE
                elif PROCESS_CAP_RE.search(nm or "") and not best:
                    best = PROCESS_SCOPE
                if nm not in seen:
                    seen.add(nm)
                    nxt.append(nm)
        frontier = nxt
        if best == STAGE_SCOPE:
            #: keep walking: a stage budget CHOSEN BY A FIXTURE FLAG is
            #: narrower still, and stopping at the first hit would report
            #: the wider of the two scopes.
            continue
    if best:
        return best
    for arg in getattr(getattr(fn, "args", None), "args", []) or []:
        if arg.arg == node.id:
            #: a PARAMETER carries the CALLER's scope, which is not knowable
            #: here. Say so rather than pick one.
            return "SCOPE_FROM_CALLER_" + node.id
    return ""


GATES_REFUSAL = "GATES_A_REFUSAL"
INSIDE_ASSERTION = "INSIDE_AN_ASSERTION_it_gates_no_refusal"
GATES_A_FLAG = "GATES_A_RECORDED_FLAG"
CHECK_CALLS = ("ok", "ck", "check", "assert_", "expect")


def _what_it_gates(node: ast.AST, parents: dict, src: str) -> str:
    """A comparison is a DEFECT only if it can REFUSE.

    DE's own battery contains `ok(_hw_after > FIXTURE_DAY_PEAK_RSS_MB_BUDGET
    ...)` -- a process-wide high-water against a fixture budget, written on
    purpose to PROVE the old comparison would have refused. Reading that as
    the defect would report the fix as the bug."""
    cur, depth = parents.get(node), 0
    while cur is not None and depth < 12:
        depth += 1
        if isinstance(cur, ast.If):
            if any(isinstance(x, ast.Raise) for x in ast.walk(cur)):
                return GATES_REFUSAL
            return GATES_A_FLAG
        if isinstance(cur, ast.Call):
            f = (cur.func.id if isinstance(cur.func, ast.Name)
                 else cur.func.attr if isinstance(cur.func, ast.Attribute)
                 else "")
            if f in CHECK_CALLS or f.startswith("assert"):
                return INSIDE_ASSERTION
        if isinstance(cur, (ast.Assert,)):
            return INSIDE_ASSERTION
        if isinstance(cur, ast.Assign):
            #: ONE INDIRECTION. `row["within_budget"] = peak <= b` followed
            #: by `if not row["within_budget"]: raise` refuses just as
            #: surely as testing the comparison in place -- and reading it
            #: as a mere recorded flag would under-state every budget in
            #: this shape.
            tgt = cur.targets[0]
            nm = (tgt.id if isinstance(tgt, ast.Name)
                  else str(tgt.slice.value)
                  if isinstance(tgt, ast.Subscript)
                  and isinstance(tgt.slice, ast.Constant) else "")
            fn = cur
            while fn is not None and not isinstance(
                    fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = parents.get(fn)
            if nm and fn is not None:
                for st in ast.walk(fn):
                    if isinstance(st, ast.If) and any(
                            isinstance(x, ast.Raise) for x in ast.walk(st)) \
                            and nm in (_seg(src, st.test) or ""):
                        return GATES_REFUSAL
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.Module)):
            break
        cur = parents.get(cur)
    return GATES_A_FLAG


def _attr_budget_scope(attr: str, tree: ast.AST, src: str) -> str:
    """A budget held on an ATTRIBUTE, resolved to the SCOPE OF THE VALUE IT
    IS CONSTRUCTED WITH.

    `b = self.budgets.get(name)` says nothing on its own. `self.budgets =
    dict(budgets)` in `__init__`, constructed at
    `_Stages(FIXTURE_STAGE_BUDGETS_GB if fixture else STAGE_BUDGETS_GB)`,
    says everything: the budget on the other side of that comparison is
    per-STAGE and, on the fixture path, per-FIXTURE -- both NARROWER than
    the process-wide high-water it is compared against."""
    owners, params = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Attribute) \
                and n.targets[0].attr == attr:
            for x in ast.walk(n.value):
                if isinstance(x, ast.Name):
                    params.add(x.id)
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and any(
                    isinstance(a, ast.Assign)
                    and isinstance(a.targets[0], ast.Attribute)
                    and a.targets[0].attr == attr
                    for a in ast.walk(fn) if isinstance(a, ast.Assign)):
                owners.add(cls.name)
    best = ""
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in owners):
            continue
        blob = " ".join(filter(None, [_seg(src, a) for a in n.args]
                              + [_seg(src, k.value) for k in n.keywords]))
        if FIXTURE_RE.search(blob):
            return FIXTURE_SCOPE
        if STAGE_RE.search(blob):
            best = STAGE_SCOPE
        elif PROCESS_CAP_RE.search(blob) and not best:
            best = PROCESS_SCOPE
    if not owners and params:
        #: the attribute exists and nothing constructs it here: say so.
        return ""
    return best


def budget_comparisons(tree: ast.AST, src: str, meas: dict,
                       consts: dict, fixture_sel: list) -> tuple:
    """Every ordered comparison of a RESOLVED MEASUREMENT against a BUDGET.

    Returns (comparisons, lookalikes). A subscript that merely LOOKS like a
    measurement is not counted as a budget comparison -- it is reported
    SEPARATELY, because counting it would invent findings and dropping it
    would hide them (rule 11)."""
    parents = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parents[c] = n

    def enclosing_fn(n):
        cur = parents.get(n)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
            cur = parents.get(cur)
        return None

    out, looks, seen, cache = [], [], set(), {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Compare) or len(n.ops) != 1:
            continue
        if not isinstance(n.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
            continue
        fn = enclosing_fn(n)
        #: a closure's comparison belongs to the OUTERMOST function -- the
        #: process `_mark` measures is `run_day`'s.
        owner = fn
        while owner is not None:
            up = enclosing_fn(owner)
            if up is None:
                break
            owner = up
        scope_fn = owner or tree
        if id(scope_fn) not in cache:
            #: TWO PASSES, because the two maps feed each other. `peak =
            #: _rss_gb()` must be known before `{"growth_gb": round(peak -
            #: baseline, 3)}` can be read as a DELTA -- and BE 60's repair
            #: is exactly that shape. One pass left the repair invisible:
            #: the census reported 0 mismatches AND 0 deltas, which says
            #: nothing at all.
            def _assigns(ks):
                out = {}
                for a in ast.walk(scope_fn):
                    if isinstance(a, ast.Assign) and len(a.targets) == 1 \
                            and isinstance(a.targets[0], ast.Name):
                        sc, dl, how = _measured_scope_of(a.value, src, meas,
                                                        {}, ks)
                        if sc and sc != "LOOKALIKE":
                            out[a.targets[0].id] = (sc, dl, how)
                return out

            asg0 = _assigns({})
            ks = _dict_key_scopes(scope_fn, src, meas, asg0)
            asg = _assigns(ks)
            cache[id(scope_fn)] = (ks, asg)
        key_scopes, assigns = cache[id(scope_fn)]
        left, right = n.left, n.comparators[0]
        for m_node, b_node in ((left, right), (right, left)):
            scope, is_delta, how = _measured_scope_of(
                m_node, src, meas, assigns, key_scopes)
            if not scope:
                continue
            bname = ""
            if isinstance(b_node, ast.Name):
                bname = b_node.id
            elif isinstance(b_node, ast.Attribute):
                bname = b_node.attr
            elif isinstance(b_node, ast.Subscript) \
                    and isinstance(b_node.slice, ast.Constant):
                bname = str(b_node.slice.value)
            elif isinstance(b_node, ast.Call):
                bname = _seg(src, b_node)
            row = {"line": n.lineno, "expr": " ".join(_seg(src, n).split()),
                   "in_function": (fn.name if fn else "<module>"),
                   "measured_how": how, "budget_name": bname}
            if scope == "LOOKALIKE":
                row["status"] = "UNRESOLVED_MEASURE_LOOKALIKE"
                row["why"] = (
                    "the key reads like a measurement and is bound to none "
                    "in this function -- reported, never counted as a "
                    "budget comparison and never dropped")
                if (n.lineno, "L") not in seen:
                    seen.add((n.lineno, "L"))
                    looks.append(row)
                break
            b_scope = (_budget_scope_local(b_node, src, scope_fn)
                       or _resolve_scope(bname, consts, fixture_sel))
            if b_scope in ("", UNKNOWN_SCOPE) or b_scope.startswith(
                    "SCOPE_FROM_CALLER"):
                #: the budget may be held on an attribute filled at a
                #: construction site elsewhere in the module.
                attrs = [x.attr for x in ast.walk(b_node)
                         if isinstance(x, ast.Attribute)]
                for a2 in ast.walk(scope_fn):
                    if isinstance(a2, ast.Assign) and len(a2.targets) == 1 \
                            and isinstance(a2.targets[0], ast.Name) \
                            and isinstance(b_node, ast.Name) \
                            and a2.targets[0].id == b_node.id:
                        attrs += [x.attr for x in ast.walk(a2.value)
                                  if isinstance(x, ast.Attribute)]
                for at in attrs:
                    got = _attr_budget_scope(at, tree, src)
                    if got:
                        b_scope = got + f"_via_attribute_{at}"
                        break
            gates = _what_it_gates(n, parents, src)
            row.update({"measured_scope": scope, "is_delta": is_delta,
                        "budget_scope": b_scope or UNKNOWN_SCOPE,
                        "gates": gates,
                        "verdict": _compare_verdict(scope, is_delta,
                                                    b_scope or "")})
            #: a mismatch that gates NOTHING is context, never a finding.
            if row["verdict"] == SCOPE_MISMATCH and gates == INSIDE_ASSERTION:
                row["verdict"] = ("SCOPE_MISMATCH_INSIDE_AN_ASSERTION_"
                                  "not_a_refusal")
            if (n.lineno, "C") not in seen:
                seen.add((n.lineno, "C"))
                out.append(row)
            break
    return out, looks


SCOPE_MISMATCH = "SCOPE_MISMATCH_process_highwater_vs_narrower_budget"
SCOPE_MATCHED = "SCOPE_MATCHED_process_measure_process_budget"
DELTA_OK = "DELTA_measured_per_stage"
CURRENT_OK = "CURRENT_RSS_falls_so_it_cannot_inherit_a_freed_peak"
UNCLASSIFIED = "UNCLASSIFIED_budget_scope_undetermined"


def _compare_verdict(measured: str, is_delta: bool, b_scope: str) -> str:
    if is_delta:
        return DELTA_OK
    if measured in (PROCESS_CURRENT, CGROUP_CURRENT):
        return CURRENT_OK
    if measured in (PROCESS_HIGHWATER, CGROUP_PEAK):
        if b_scope.startswith(PROCESS_SCOPE):
            return SCOPE_MATCHED
        if b_scope.startswith(FIXTURE_SCOPE) or b_scope.startswith(
                STAGE_SCOPE):
            return SCOPE_MISMATCH
    return UNCLASSIFIED


# --------------------------------------- THE CLASS: ambient process state

#: REV 58 section 1.2. THE THREE QUESTIONS ABOVE ARE INSTANCES; THIS IS THE
#: CLASS. "Does an assertion's verdict depend on state this process happens
#: to be IN, rather than on the property it claims to test?" Memory was one
#: ambient and the sweep asked only about memory -- so the 26-second
#: refusal, a `refuses(lambda: run_day(...))` whose needle appears ONLY
#: when the calling process does NOT hold the heavy lock, was none of its
#: three questions. The lock, the cgroup, the cwd, the worktree and the
#: clock are ambients too, and an assertion that reads one DIRECTLY --
#: rather than being handed an OBSERVATION -- is judging the process it
#: happens to be running in.
AMBIENT_READERS = {
    "memory": ("getrusage", "ru_maxrss", "/proc/self/statm",
               "/proc/self/status", "memory.peak", "memory.current"),
    "lock": ("flock", "/proc/locks", "lockf", ".heavy_run.lock", "LOCK_EX",
             "LOCK_NB"),
    "cgroup": ("/proc/self/cgroup", "/sys/fs/cgroup", "systemd-run",
               "MemoryMax"),
    "cwd": ("getcwd", "cwd()", "Path.cwd"),
    "worktree": ("__file__", "parents[", "HERE"),
    "clock": ("datetime.now", "time.time", "time.monotonic", "utcnow",
              "date.today", "perf_counter"),
    "process": ("getpid", "os.environ", "sys.argv", "psutil"),
}
#: a parameter named like an OBSERVATION is the injection that makes a
#: check about the property instead of about the process.
OBSERVATION_PARAM_RE = re.compile(
    r"^(obs|observation|stages|peak|peak_[a-z_]*|rss|now|clock|when|"
    r"held|lock_held|root|data_root|cwd|env|measured|reading|sample|"
    r"snapshot|state|budget|declared|proof|capture)$", re.I)
CHECK_CALLS_AMBIENT = ("ok", "ck", "check", "assert_", "expect", "refuses",
                       "raises")


def _ambient_kinds(text: str) -> list:
    t = text or ""
    return sorted(k for k, needles in AMBIENT_READERS.items()
                  if any(n in t for n in needles))


def ambient_state_census(tree: ast.AST, src: str) -> dict:
    """Every ASSERTION whose verdict can turn on ambient process state.

    An assertion is a call to a check helper (`ok`, `ck`, `refuses`, ...)
    or a bare `assert`. Its operands are read for DIRECT ambient reads --
    including one hop into a `lambda` body, which is exactly how the
    26-second shape hides: `refuses(lambda: run_day(...))` reads no ambient
    ITSELF, and the call it wraps decides the verdict by whether the
    process holds the lock."""
    parents = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parents[c] = n

    def enclosing_fn(n):
        cur = parents.get(n)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
            cur = parents.get(cur)
        return None

    #: the module's own functions, so a wrapped callee can be opened ONE
    #: hop to ask whether ITS refusal can come from an ambient.
    fn_src = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_src[n.name] = _seg(src, n) or ""

    def _truth_text(node) -> str:
        """THE TRUTH-BEARING PART ONLY. An assertion's MESSAGE often names
        a path or a clock; the VERDICT is the comparison. Reading the whole
        call made every `ok(x, f"...{__file__}...")` an ambient finding --
        the too-wide watch list of REV 49 section 2.4, again."""
        out = []
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
                out.append(_seg(src, sub) or "")
        if not out and isinstance(node, ast.Call) and node.args:
            out.append(_seg(src, node.args[0]) or "")
        return " ".join(out)

    rows, n_assertions = [], 0
    for n in ast.walk(tree):
        is_assert = isinstance(n, ast.Assert)
        callee = ""
        if isinstance(n, ast.Call):
            callee = (n.func.id if isinstance(n.func, ast.Name)
                      else n.func.attr if isinstance(n.func, ast.Attribute)
                      else "")
            if callee not in CHECK_CALLS_AMBIENT \
                    and not callee.startswith("assert"):
                continue
        elif not is_assert:
            continue
        n_assertions += 1
        seg = _seg(src, n) or ""
        kinds = _ambient_kinds(_truth_text(n))
        #: ONE HOP for the lambda, TWO for what it calls.
        #: `refuses(lambda: run_day(...))` reads no ambient ITSELF -- and
        #: `run_day` does, which is where the verdict comes from. That is
        #: the 26-second shape, and it is invisible one hop up.
        wrapped, wrapped_ambient = [], []
        for sub in ast.walk(n):
            if not isinstance(sub, ast.Lambda):
                continue
            body = _seg(src, sub.body) or ""
            wrapped.append(body[:120])
            kinds = sorted(set(kinds) | set(_ambient_kinds(body)))
            for c2 in ast.walk(sub.body):
                if isinstance(c2, ast.Call) and isinstance(c2.func, ast.Name):
                    ik = _ambient_kinds(fn_src.get(c2.func.id, ""))
                    if ik:
                        wrapped_ambient.append(
                            {"callee": c2.func.id, "ambient_kinds": ik})
                        kinds = sorted(set(kinds) | set(ik))
        fn = enclosing_fn(n)
        params = [a.arg for a in
                  (getattr(getattr(fn, "args", None), "args", []) or [])]
        injected = sorted(a for a in params if OBSERVATION_PARAM_RE.match(a))
        expects_refusal = callee in ("refuses", "raises") or any(
            isinstance(sub, ast.Call)
            and ((isinstance(sub.func, ast.Name)
                  and sub.func.id in ("refuses", "raises"))
                 or (isinstance(sub.func, ast.Attribute)
                     and sub.func.attr in ("refuses", "raises")))
            for sub in ast.walk(n))
        if not kinds:
            continue
        if expects_refusal and wrapped_ambient:
            verdict = "REFUSAL_EXPECTATION_WHOSE_CAUSE_CAN_BE_AMBIENT"
        elif injected and not wrapped_ambient:
            verdict = "INJECTED_OBSERVATION"
        else:
            verdict = "AMBIENT_IN_THE_VERDICT"
        rows.append({
            "line": n.lineno,
            "in_function": (fn.name if fn else "<module>"),
            "check": callee or "assert",
            "ambient_kinds": kinds,
            "wrapped_callables": wrapped,
            "wrapped_callees_reading_an_ambient": wrapped_ambient,
            "observation_parameters": injected,
            "verdict": verdict,
            "expr": " ".join(seg.split())[:160],
        })
    by_kind = {}
    for r in rows:
        for k in r["ambient_kinds"]:
            by_kind[k] = by_kind.get(k, 0) + 1
    flagged = [r for r in rows if r["verdict"] != "INJECTED_OBSERVATION"]
    return {
        "question": ("does any assertion's verdict depend on state this "
                     "process happens to be IN, rather than on the "
                     "property?"),
        "n_assertions_seen": n_assertions,
        "n_assertions_touching_an_ambient": len(rows),
        "n_flagged": len(flagged),
        "by_ambient_kind": dict(sorted(by_kind.items())),
        "rows": rows[:60],
        "n_rows_listed": min(len(rows), 60),
        "ambients_tested": sorted(AMBIENT_READERS),
        "what_this_CANNOT_see_from_source": [
            "MAGNITUDE: that a budget is 700 MB and the process is at 2.4 "
            "GB is a runtime fact; the source shows only that the "
            "comparison reads an ambient",
            "a CALLABLE CHOSEN AT RUNTIME: `refuses(fn)` where `fn` comes "
            "from a variable, a registry or a parameter -- the wrapped "
            "call cannot be read here, and the row says `wrapped` with no "
            "body rather than claiming it is clean",
            "whether an ambient READ is also the ambient the verdict TURNS "
            "on: a check may read the clock only to stamp a message",
        ],
    }


# ------------------------------------------------- fixtures on a real path

BATTERY_RE = re.compile(r"^(selftest|fixture)$|battery|_checks$|"
                        r"^_day_path_checks$|^selftest_")
#: a CALLER whose own name says it is a battery: `fixture_run`, `selftest_x`.
#: The distinction matters -- a battery calling a battery is not the defect.
CALLER_BATTERY_RE = re.compile(r"^(selftest|fixture)(_|$)|battery|_checks$")
FIXTURE_KW = ("fixture", "offline", "synthetic", "fake", "dry_run")
REAL_ENTRY_HINT = re.compile(r"--day|--run|--build|--sealed")


def _innermost_fn_at(tree: ast.AST, line: int) -> str:
    """The function a line actually sits in, nested defs included."""
    best, name = None, "<module>"
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and n.lineno <= line <= (n.end_lineno or n.lineno):
            if best is None or n.lineno > best:
                best, name = n.lineno, n.name
    return name


def _passed_as_callback(tree: ast.AST, src: str, fname: str) -> list:
    """Every call that receives this function BY NAME as a keyword.

    A battery called inside a nested `_battery_first()` handed to the real
    path as `before_work=_battery_first` is still a battery on the real
    path -- and saying WHERE it is handed over is the difference between
    'the shape is there' and 'the shape is there and it fires last'."""
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        callee = (n.func.id if isinstance(n.func, ast.Name)
                  else n.func.attr if isinstance(n.func, ast.Attribute)
                  else "")
        for k in n.keywords:
            if k.arg and isinstance(k.value, ast.Name) \
                    and k.value.id == fname:
                out.append(f"{callee}({k.arg}={fname})")
    return out


def call_graph(tree: ast.AST, src: str) -> dict:
    """caller -> [ {callee, line, kwargs, guards} ]. Guards are the source
    of every `if` test enclosing the call, so a battery reached ONLY through
    the CLI's own `--selftest` branch is distinguishable from one reached on
    the real path."""
    parents = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parents[c] = n
    out = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        edges = []
        early = []
        #: ONLY A DOMINATING EARLY EXIT COUNTS: a TOP-LEVEL `if cond:` in
        #: the function body, with no `else`, whose LAST statement leaves
        #: the function. An `if` nested in a loop or a branch does not
        #: dominate the code below it, and treating it as a guard would
        #: excuse exactly the call this sweep exists to find.
        for st in fn.body:
            if not isinstance(st, ast.If) or st.orelse or not st.body:
                continue
            last = st.body[-1]
            leaves = isinstance(last, (ast.Return, ast.Raise)) or (
                isinstance(last, ast.Expr) and isinstance(last.value, ast.Call)
                and "error" in (_seg(src, last.value.func) or ""))
            if leaves:
                early.append((st.lineno, _seg(src, st.test) or ""))
        for n in ast.walk(fn):
            if not isinstance(n, ast.Call):
                continue
            callee = (n.func.id if isinstance(n.func, ast.Name)
                      else n.func.attr if isinstance(n.func, ast.Attribute)
                      else "")
            if not callee:
                continue
            guards, cur = [], parents.get(n)
            while cur is not None and cur is not fn:
                if isinstance(cur, ast.If):
                    guards.append(_seg(src, cur.test))
                cur = parents.get(cur)
            #: AN EARLY EXIT IS A GUARD TOO. `if not a.selftest: ap.error(...)`
            #: above the call dominates it exactly as an enclosing `if` does,
            #: and a checker that only sees enclosing blocks would call a
            #: correctly guarded CLI a finding. Precomputed per function --
            #: recomputing it per call is quadratic on a 5,800-line runner.
            guards += [f"EARLY_EXIT: {t}" for ln, t in early if ln < n.lineno]
            kw = {k.arg: _seg(src, k.value) for k in n.keywords if k.arg}
            edges.append({"callee": callee, "line": n.lineno,
                          "kwargs": kw, "guards": guards})
        out[fn.name] = edges
    return out


#: A GUARD IS A DISPATCH ON THE BATTERY SWITCH, not a test that happens to
#: MENTION it. DE's `_main_day` exits early on
#: `not fixture and not payload[...]["producing_code_is_the_committed_bytes"]`
#: -- a compound test that does NOT dominate the battery call for the
#: fixture case. Matching the word alone excused the exact call this sweep
#: exists to find, and the sweep reported the runner CLEAN.
BATTERY_SWITCH_RE = re.compile(
    r"^(a\.|args\.|self\.)?(selftest|fixture|is_fixture)$", re.I)


def _operands(text: str) -> list:
    """Split `A or B` at the TOP LEVEL only.

    A regex split found the ` or ` inside `"--selftest" in (argv or [])`
    and tore one operand into two, after which a correctly guarded CLI read
    as unguarded. Depth and quotes are tracked."""
    t = " ".join((text or "").split())
    out, buf, depth, quote, i = [], [], 0, "", 0
    while i < len(t):
        c = t[i]
        if quote:
            buf.append(c)
            if c == quote:
                quote = ""
            i += 1
            continue
        if c in "\"'":
            quote = c
            buf.append(c)
            i += 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if depth == 0:
            for op in (" or ", " and "):
                if t[i:i + len(op)] == op:
                    out.append("".join(buf).strip())
                    buf = []
                    i += len(op)
                    break
            else:
                buf.append(c)
                i += 1
            continue
        buf.append(c)
        i += 1
    out.append("".join(buf).strip())
    return [x for x in out if x]


def _switch(t: str, negated: bool) -> bool:
    t = t.strip()
    if ("--selftest" in t or "--fixture" in t):
        #: `"--selftest" in argv` is the POSITIVE dispatch; its negation is
        #: spelled with `not`.
        return negated == t.startswith("not ")
    if t.startswith("not "):
        return negated and bool(BATTERY_SWITCH_RE.match(t[4:].strip()))
    return (not negated) and bool(BATTERY_SWITCH_RE.match(t))


def _is_battery_switch(text: str, negated: bool = False) -> bool:
    """`a.selftest` is a dispatch; so is `a.selftest or a.fixture`. A test
    that merely MENTIONS the word beside something else is not."""
    t = " ".join((text or "").replace("EARLY_EXIT: ", "").split())
    ops = _operands(t)
    return bool(ops) and all(_switch(o, negated) for o in ops)


def _guard_admits(text: str) -> bool:
    """Does this guard put the call ON the battery branch?

    THE TWO FORMS ARE OPPOSITE. An ENCLOSING `if a.selftest:` puts the code
    inside it on the battery branch. A dominating EARLY EXIT does the
    reverse: after `if a.selftest: return`, everything below runs on the
    REAL path -- so only a NEGATED switch (`if not a.selftest: error()`)
    leaves the battery branch below it. Reading both the same way excused a
    fixture-mode call sitting on a real path, which is the whole finding
    this sweep exists to make."""
    if (text or "").startswith("EARLY_EXIT: "):
        return _is_battery_switch(text, negated=True)
    return _is_battery_switch(text, negated=False)


#: DA 155: THREE STATES, NOT A BOOL. `git status --porcelain -- <path>`
#: exits 128 for a path OUTSIDE the tree, and collapsing that into False
#: reports "this file is uncommitted" for a file the tree has never heard
#: of -- every scratch module this battery audits was reported that way.
#: Absence read as a negative, which is the class this seat has been
#: finding all night, here in its own instrument.
IN_TREE_CLEAN = "COMMITTED_IN_THIS_TREE"
IN_TREE_DIRTY = "PRESENT_BUT_NOT_COMMITTED"
NOT_IN_TREE = "NOT_IN_THIS_TREE"


def _tree_containing(path: Path):
    """The git top-level the FILE lives in, or None if it lives in none.

    DA 156 / REV 131. `committed_state` asked AUDIT_ROOT -- the MAIN tree --
    about whatever path it was handed, so the auditor's own file read
    NOT_IN_THIS_TREE from every worktree and the cell asserting a real
    in-tree state was RED FROM EVERY WORKTREE. That is not the tree
    disagreeing; it is the question being asked of the wrong tree. The
    committed state of a file is a fact about the file's OWN repository,
    so that is what is asked.
    """
    try:
        r = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=20)
    except Exception:                                         # noqa: BLE001
        return None
    return Path(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() \
        else None


def committed_state(path: Path, root=None) -> str:
    """CLEAN / DIRTY / NOT-IN-THIS-TREE, distinguished by name.

    `root` defaults to the tree the PATH lives in -- not to AUDIT_ROOT --
    so the answer is a fact about the file rather than about which tree the
    caller happens to treat as canonical. Pass `root` explicitly to ask a
    NAMED tree about a path, which is a different and also legitimate
    question; the two disagree for a worktree file and that disagreement
    was the whole of REV 131's red.
    """
    tree = Path(root) if root is not None else _tree_containing(path)
    if tree is None:
        return NOT_IN_TREE
    r = subprocess.run(["git", "-C", str(tree), "status", "--porcelain",
                        "--", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        return NOT_IN_TREE
    return IN_TREE_CLEAN if r.stdout.strip() == "" else IN_TREE_DIRTY


def _is_committed(path: Path) -> bool:
    """Kept for callers that want the bool; TRUE only for CLEAN."""
    return committed_state(path) == IN_TREE_CLEAN


def fixture_on_real_path(tree: ast.AST, src: str) -> dict:
    """(c) A BATTERY INVOKED FROM THE REAL PATH.

    The predicate: a call to a battery whose caller is NOT itself a battery
    and where NO enclosing guard names the battery switch. DE's
    `_main_day: selftest(quiet=True, offline=fixture)` is the case; a CLI's
    `if "--selftest" in argv: return selftest()` is not."""
    g = call_graph(tree, src)
    findings, benign = [], []
    for caller, edges in g.items():
        caller_is_battery = bool(CALLER_BATTERY_RE.search(caller))
        for e in edges:
            if not BATTERY_RE.search(e["callee"]):
                continue
            guarded = any(_guard_admits(gtxt) for gtxt in e["guards"])
            inner = _innermost_fn_at(tree, e["line"])
            row = {"caller": caller, "callee": e["callee"], "line": e["line"],
                   "innermost_caller": inner,
                   "handed_to": _passed_as_callback(tree, src, inner),
                   "kwargs": e["kwargs"], "guards": e["guards"],
                   "guarded_by_the_battery_switch": guarded,
                   "caller_is_itself_a_battery": caller_is_battery}
            if guarded or caller_is_battery:
                benign.append(row)
            else:
                row["why"] = (
                    "a battery is called from a function that is not itself "
                    "a battery and no enclosing guard names the battery "
                    "switch -- so a REAL run executes it IN THE SAME "
                    "PROCESS, where every process-wide number the fixture "
                    "is judged against has already been raised by the real "
                    "work")
                findings.append(row)
    #: and the second half of (c): a real function called with a fixture
    #: argument from a caller that is not a battery.
    for caller, edges in g.items():
        if CALLER_BATTERY_RE.search(caller):
            continue
        for e in edges:
            #: ONLY A LITERAL `True`. `fixture=fixture` PROPAGATES the
            #: caller's own mode, which is how a mode is meant to travel;
            #: flagging it would report every honest plumb as a defect.
            hits = {k: v for k, v in e["kwargs"].items()
                    if k in FIXTURE_KW and v == "True"}
            #: THE GUARD RULE APPLIES HERE TOO. The first half checked
            #: guards and this half did not, so `assert_source_unchanged(
            #: ..., fixture=True)` inside a CLI's own `if a.selftest:`
            #: branch read as a fixture on the real path. A rule that holds
            #: on one half of a check and not the other is two rules.
            guarded2 = any(_guard_admits(g) for g in e["guards"])
            if hits and not guarded2 and not BATTERY_RE.search(e["callee"]):
                findings.append({
                    "caller": caller, "callee": e["callee"],
                    "line": e["line"], "kwargs": e["kwargs"],
                    "guards": e["guards"],
                    "guarded_by_the_battery_switch": False,
                    "caller_is_itself_a_battery": False,
                    "why": ("a fixture-mode argument is passed on a path "
                            "that is not a battery")})
    return {"findings": findings, "benign_battery_calls": benign,
            "n_findings": len(findings)}


# ------------------------------------------------------ rule 22 (R-605)

AT_IMPORT = "AT_IMPORT_the_code_that_ran"
AT_EMIT = "AT_EMIT_the_code_on_disk_when_the_receipt_was_written"
TYPED = "TYPED_LITERAL_not_derived_at_all"
ABSENT = "ABSENT"


def _import_time_functions(tree: ast.AST, src: str) -> set:
    """Functions that RUN AT IMPORT.

    The question rule 22 asks is not where a line of code SITS but WHEN THE
    CAPTURE HAPPENS. DE captures its closure inside `_digest_module`, called
    by `_capture_closure`, called at module level -- code that sits in a
    function and runs at import. A checker reading only the enclosing block
    would call that an emit-time stamp and mark the one runner that has the
    property as lacking it.

    The `if __name__ == "__main__":` guard is EXCLUDED: `main()` there runs
    after import, and counting it would make every function in the module
    import-time."""
    g = call_graph(tree, src)
    seeds = set()
    for st in tree.body:
        #: A `def` in the module body is a DEFINITION, not a call. Walking
        #: into it collected every call in every function and made the whole
        #: module read as import-time.
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef,
                           ast.ClassDef)):
            continue
        if isinstance(st, ast.If) and "__name__" in (_seg(src, st.test) or ""):
            continue
        for n in ast.walk(st):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                seeds.add(n.func.id)
    #: DEPTH 2, and no further. A function called at import may call
    #: helpers that the EMIT path also calls; expanding without bound makes
    #: every function in the module read as import-time and the answer
    #: becomes "yes" for everybody, which is the same as no answer.
    seen, frontier, depth = set(seeds), set(seeds), 0
    while frontier and depth < 1:
        depth += 1
        nxt = set()
        for f in frontier:
            for e in g.get(f, []):
                if e["callee"] not in seen:
                    seen.add(e["callee"])
                    nxt.add(e["callee"])
        frontier = nxt
    return seen


SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")


def _module_level_imports(tree: ast.AST) -> dict:
    """alias -> module name, for imports at MODULE level only."""
    out = {}
    for st in tree.body:
        if isinstance(st, ast.Import):
            for a in st.names:
                out[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(st, ast.ImportFrom) and st.module and not st.level:
            out[st.module.split(".")[0]] = st.module
    return out


def _sibling_source(mod: str) -> tuple:
    for pkg in ("live/pm_research", "live/mm_research"):
        f = AUDIT_ROOT / pkg / f"{mod}.py"
        if f.is_file():
            try:
                return f, f.read_text()
            except OSError:
                return None, None
    return None, None


def rule22_via_a_shared_module(tree: ast.AST, src: str) -> dict:
    """THE CAPTURE MAY LIVE IN A SIBLING, AND IT STILL COUNTS.

    BE 60 put the launch capture in `be_rule22.py` and the three producers
    import it. A per-FILE detector reports ABSENT for all three and that is
    a FALSE NEGATIVE about another seat's work -- the same 'half a
    predicate reported as the predicate' this sweep exists to catch. One
    hop: a module-level import of a sibling that HAS the construct, plus a
    call through that alias, is the property PRESENT VIA that sibling."""
    aliases = _module_level_imports(tree)
    used = {n.func.value.id for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)}
    #: WHEN the shared capture runs is decided HERE, in the importer: an
    #: alias called at MODULE LEVEL captures at import. BE's three
    #: producers each call `_R22.init(...)` at module level, and a checker
    #: that stopped at "the construct lives in another file" would report
    #: the timing it could not see.
    at_module_level = set()
    for st in tree.body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef,
                           ast.ClassDef)):
            continue
        if isinstance(st, ast.If) and "__name__" in (_seg(src, st.test) or ""):
            continue
        for n in ast.walk(st):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and isinstance(n.func.value, ast.Name):
                at_module_level.add(n.func.value.id)
    out = {}
    for alias, mod in aliases.items():
        if alias not in used or not mod.startswith(("be_", "de_", "da_",
                                                    "pm_", "e2_")):
            continue
        f, txt = _sibling_source(mod)
        if not txt:
            continue
        try:
            sub = ast.parse(txt)
        except SyntaxError:
            continue
        st = rule22_stamp(sub, txt, _via=False)["status"]
        when = (AT_IMPORT if alias in at_module_level else AT_EMIT)
        for k, v in st.items():
            if v != ABSENT and out.get(k) in (None, ABSENT):
                out[k] = f"{when}__VIA_{mod}"
    return out


def rule22_stamp(tree: ast.AST, src: str, _via: bool = True) -> dict:
    """Rule 22 / R-605, computed: the producing-code digest, the import
    closure and HEAD -- and WHEN each is captured.

    The distinction is the whole rule. A digest read at EMIT time names the
    file as it is when the receipt is written; if the file moved during the
    run that is not the code that ran, and the committed-bytes guard PASSES
    because the replacement is committed."""
    fn_lines = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_lines.append((n.lineno, n.end_lineno or n.lineno, n.name))
    at_import = _import_time_functions(tree, src)

    def where(line: int) -> tuple:
        host, best = "<module>", None
        for a, b, nm in fn_lines:
            if a <= line <= b and (best is None or a > best):
                best, host = a, nm
        if host == "<module>":
            return AT_IMPORT, host
        return (AT_IMPORT if host in at_import else AT_EMIT), host

    out = {"producing_code_digest": [], "import_closure": [],
           "head_sha": [], "typed_literals": [], "digests_of_other_files": []}
    for n in ast.walk(tree):
        seg = _seg(src, n)
        if isinstance(n, ast.Call):
            f = (n.func.attr if isinstance(n.func, ast.Attribute)
                 else n.func.id if isinstance(n.func, ast.Name) else "")
            if f in ("sha256", "file_digest") and (
                    "__file__" in seg or "read_bytes" in seg):
                w, host = where(n.lineno)
                #: WHOSE BYTES? A producing-code digest is a digest of THE
                #: MODULE'S OWN FILE. Round 77 counted every
                #: `sha256(<path>.read_bytes())` reached from an emit path,
                #: so a book's READ-BACK, a receipt's read-back, an INPUT
                #: fragment and BE's own battery control for this very
                #: property all read as producing-code digests -- and all
                #: three BE producers then sat falsely in
                #: `rule22_binds_and_incomplete`. The property is whose
                #: bytes are hashed, not where the call sits.
                own = ("__file__" in seg
                       or re.search(r"\b(me|self_src|_src|src_path|"
                                    r"MY_PATH|THIS_FILE)\b", seg)
                       is not None)
                row = {"line": n.lineno, "when": w, "in": host,
                       "expr": " ".join(seg.split())[:120],
                       "hashes_this_modules_own_file": bool(own)}
                if own:
                    out["producing_code_digest"].append(row)
                else:
                    out.setdefault("digests_of_other_files", []).append(row)
            if "rev-parse" in seg and "HEAD" in seg:
                w, host = where(n.lineno)
                out["head_sha"].append(
                    {"line": n.lineno, "when": w, "in": host,
                     "expr": " ".join(seg.split())[:120]})
        if isinstance(n, ast.Attribute) and n.attr == "modules" \
                and isinstance(n.value, ast.Name) and n.value.id == "sys":
            w, host = where(n.lineno)
            out["import_closure"].append(
                {"line": n.lineno, "when": w, "in": host,
                 "expr": " ".join(seg.split())[:120]})
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if not (isinstance(k, ast.Constant)
                        and isinstance(k.value, str)
                        and k.value in ("carrying_commit", "commit",
                                        "producing_commit",
                                        "producing_code_sha256")
                        and isinstance(v, ast.Constant)
                        and isinstance(v.value, str)):
                    continue
                blob = _seg(src, n) or ""
                historical = bool(re.search(r"supersed|chain|v\d+ ", blob,
                                            re.I))
                out["typed_literals"].append({
                    "line": k.lineno, "key": k.value, "value": v.value,
                    "looks_like_a_commit_sha": bool(SHA_RE.match(v.value)),
                    "kind": ("HISTORICAL_CITATION_of_a_superseded_artifact"
                             if historical else
                             "TYPED_STAMP_of_this_run"
                             if SHA_RE.match(v.value) else
                             "PLACEHOLDER_not_a_digest"),
                    "why": ("a provenance field TYPED as a literal is "
                            "derived from nothing and cannot move when the "
                            "code does -- unless it is CITING a past "
                            "artifact, where a literal is the only honest "
                            "form")})
    status = {}
    for k in ("producing_code_digest", "import_closure", "head_sha"):
        rows = out[k]
        status[k] = (ABSENT if not rows
                     else AT_IMPORT if any(r["when"] == AT_IMPORT
                                           for r in rows)
                     else AT_EMIT)
    if _via:
        via = rule22_via_a_shared_module(tree, src)
        for k, v in via.items():
            if status.get(k) == ABSENT:
                status[k] = v
        out["present_via_a_shared_module"] = via
    #: THE CLOSURE COVERS THE PRODUCER'S OWN FILE. Rule 22 asks for "the
    #: digest of every module in their import closure under `live/`" -- and
    #: the producing module is one of them. A separate own-file hash is a
    #: second statement of a fact the closure already carries, so a module
    #: whose closure is captured AT IMPORT satisfies the producing-code
    #: construct THROUGH IT, and the receipt says by which route.
    if status["import_closure"].startswith(AT_IMPORT) \
            and not status["producing_code_digest"].startswith(AT_IMPORT):
        status["producing_code_digest"] = (
            AT_IMPORT + "__COVERED_BY_THE_IMPORT_CLOSURE")
        out["producing_code_covered_by_the_closure"] = True
    out["status"] = status
    out["rule22_complete"] = all(v.startswith(AT_IMPORT)
                                 for v in status.values())
    out["typed_stamps_of_this_run"] = [t for t in out["typed_literals"]
                                       if t["kind"].startswith("TYPED_STAMP")]
    out["n_typed_stamps_of_this_run"] = len(out["typed_stamps_of_this_run"])
    return out


# ------------------------------------------------------------ the module

UNPARSEABLE = "SOURCE_DID_NOT_PARSE"


class AuditRefused(RuntimeError):
    """This module cannot be audited from the bytes it was given."""


def audit_module(path: Path, role: str) -> dict:
    #: DA 155: AN UNPARSEABLE INPUT IS A NAMED REFUSAL, NOT A TRACEBACK.
    #: `ast.parse` raised a bare SyntaxError here, and this battery feeds
    #: `audit_module` the stdout of a `git show` and text written to
    #: scratch -- so ANY transient that yields partial source killed the
    #: whole run with rc 1 and stdout truncated mid-cell, which is exactly
    #: the signature observed twice during a landing window and never
    #: since. The verdict now depends on the SUBJECT's bytes: unusable
    #: bytes refuse by name and the caller decides, rather than the
    #: process dying and the battery reading as a failure of the module
    #: it happened to be examining.
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise AuditRefused(
            f"REFUSED {UNPARSEABLE} for {path.name}: {e.msg} at line "
            f"{e.lineno}. {len(src)} bytes were read. This audit reads "
            f"source it is HANDED -- from `git show`, from scratch, from "
            f"the tree -- and text that does not parse is an unusable "
            f"input, not a finding about the module.") from None
    meas = measuring_functions(tree, src)
    consts = module_constants(tree)
    fx_sel = fixture_selected_budgets(tree, src)
    cmps, looks = budget_comparisons(tree, src, meas, consts, fx_sel)
    frp = fixture_on_real_path(tree, src)
    amb = ambient_state_census(tree, src)
    r22 = rule22_stamp(tree, src)
    mism = [c for c in cmps if c["verdict"] == SCOPE_MISMATCH]
    #: THE PREFIX IS "SCOPE_MISMATCH", not the whole refusing verdict
    #: string. Testing `startswith(SCOPE_MISMATCH)` -- the full constant --
    #: matched nothing, so the assertion-only mismatches counted ZERO while
    #: the module rows carried them. A count pinned to a whole string rather
    #: than the property is the defect this seat keeps shipping.
    mism_assert = [c for c in cmps
                   if c["verdict"].startswith("SCOPE_MISMATCH")
                   and c["verdict"] != SCOPE_MISMATCH]
    unc = [c for c in cmps if c["verdict"] == UNCLASSIFIED]
    return {
        "path": (str(path.relative_to(AUDIT_ROOT))
                 if path.is_relative_to(AUDIT_ROOT) else str(path)),
        "read_from_tree": str(AUDIT_ROOT),
        "committed_in_that_tree": _is_committed(path),
        #: DA 155: the STATE, so a scratch module outside the tree is not
        #: reported as an uncommitted one.
        #: asked of the AUDIT tree ON PURPOSE -- this census is about what
        #: THAT tree holds -- while the auditor's own identity below asks
        #: the file's own tree. Two questions, both named (DA 156).
        "committed_state": committed_state(path, root=AUDIT_ROOT),
        "committed_state_asked_of": str(AUDIT_ROOT),
        "role": role,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "measurement_sites": measurement_sites(tree, src),
        "measuring_functions": meas,
        "fixture_selected_budgets": fx_sel,
        "census": {
            "n_measurement_sites": len(measurement_sites(tree, src)),
            "n_budget_comparisons": len(cmps),
            "n_deltas": sum(1 for c in cmps if c["is_delta"]),
            "n_scope_matched": sum(1 for c in cmps
                                   if c["verdict"] == SCOPE_MATCHED),
            "n_scope_mismatch": len(mism),
            "n_scope_mismatch_inside_an_assertion": len(mism_assert),
            "n_current_rss": sum(1 for c in cmps
                                 if c["verdict"] == CURRENT_OK),
            "n_unclassified": len(unc),
            "n_unresolved_measure_lookalikes": len(looks),
            "n_fixture_selected_budget_sites": len(fx_sel),
            "n_battery_calls_on_the_real_path": frp["n_findings"],
            "n_assertions_touching_an_ambient":
                amb["n_assertions_touching_an_ambient"],
            "n_assertions_whose_verdict_is_ambient": amb["n_flagged"],
        },
        "budget_comparisons": cmps,
        "unresolved_measure_lookalikes": looks,
        "fixture_on_real_path": frp,
        "ambient_state_census": amb,
        "rule22": r22,
        "verdict": ("FLAGGED" if (mism or frp["n_findings"]) else
                    "UNCLASSIFIED_PRESENT" if unc else "CLEAN"),
        "why": ("(a)+(b) a process-wide high-water is compared against a "
                "narrower budget, and/or (c) a battery runs on the real "
                "path" if (mism or frp["n_findings"]) else
                "no comparison pairs a process-wide high-water with a "
                "narrower budget, and no battery is invoked from the real "
                "path"),
    }


def sweep(paths=None) -> dict:
    rows = []
    for group, role in ((BE_PRODUCERS, "BE_heavy_producer"),
                        (BE_SHARED, "BE_shared_measurement_module"),
                        (DA_INSTRUMENTS, "DA_instrument"),
                        (KNOWN_BAD, "DE_runner_the_known_bad")):
        for rel in (paths or group):
            p = AUDIT_ROOT / rel
            if not p.is_file():
                rows.append({"path": rel, "role": role,
                             "verdict": "MODULE_ABSENT"})
                continue
            rows.append(audit_module(p, role))
        if paths:
            break
    return {"modules": rows}


def carrying_commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or "UNKNOWN"
    except Exception:                                         # noqa: BLE001
        return "UNKNOWN"


def auditor_identity() -> dict:
    src = Path(__file__).resolve()
    d = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                        "--", str(src)], capture_output=True, text=True)
    return {"path": "live/pm_research/da_process_budget_audit.py",
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "tree_head": carrying_commit(),
            "producing_code_is_the_committed_bytes":
                d.returncode == 0 and d.stdout.strip() == ""}


#: Rule 22 binds RUNNERS and HEAVY PRODUCERS by its own words ("the
#: fragment, tape and book builders included"). A light read-only checker is
#: not in that set, and reporting it as a violation would put a demand in
#: the rule that the rule does not make. The book verifier IS listed,
#: because its book tier runs under the wrapper on a 290 MB pickle.
RULE22_BINDS = {
    "live/pm_research/be_gate1_fragment.py": True,
    "live/pm_research/be_gate1_state_tape.py": True,
    "live/pm_research/be_daybook_build.py": True,
    "live/pm_research/be_data_root.py": False,
    "live/pm_research/da_book_verify.py": True,
    #: THIS SEAT'S OWN HEAVY RUNNER. The E2-A smoke ran 84 minutes at 2.4 GB
    #: under the wrapper, so it is a runner and a heavy producer by rule
    #: 22's own words. Leaving it False would have exempted my own module
    #: from the rule I am auditing three of BE's against.
    "live/mm_research/e2_a_runner.py": True,
    "live/pm_research/de_multiday_gate1_runner.py": True,
}


def supersession_block(prior: Path, current: dict) -> dict:
    """The R-608 PAIR for an in-band re-emission, plus WHAT MOVED.

    A re-run of a census is only worth landing if the thing censused
    changed; the receipt says which modules moved and which did not, so a
    reader never has to diff two files to find out."""
    pri = json.loads(prior.read_text())
    was = {m["path"]: m.get("sha256") for m in pri.get("modules", [])
           if m.get("sha256")}
    now = {m["path"]: m.get("sha256") for m in current.get("modules", [])
           if m.get("sha256")}
    moved = sorted(p for p in set(was) & set(now) if was[p] != now[p])
    #: THE CHAIN TRAVELS FORWARD COMPLETE, as the accrual report's block
    #: already did. A v4 whose chain named only the v3 would make a reader
    #: open three files to learn what a fourth supersedes -- and two
    #: supersession blocks with two behaviours is two rules (R-608).
    prior_chain = [list(e) for e in
                   ((pri.get("supersedes") or {}).get("chain") or [])
                   if isinstance(e, (list, tuple)) and len(e) == 2]
    return {
        "path": prior.name,
        "sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
        "chain": prior_chain + [[prior.name,
                                 hashlib.sha256(
                                     prior.read_bytes()).hexdigest()]],
        "v1_untouched": True,
        "why_re_emitted": ("the tree under audit moved: a census names the "
                           "bytes it read, so a census of other bytes is a "
                           "different statement"),
        "modules_that_moved_since": moved,
        "modules_added": sorted(set(now) - set(was)),
        "modules_removed": sorted(set(was) - set(now)),
        "n_modules_moved": len(moved),
        "totals_then": pri.get("totals"),
    }


def build_report(now: datetime.datetime | None = None,
                 supersedes: Path | None = None) -> dict:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    sw = sweep()
    for m in sw["modules"]:
        m["rule22_binds_this_module"] = RULE22_BINDS.get(m["path"], False)
    live = [m for m in sw["modules"]
            if m.get("census", {}).get("n_scope_mismatch")
            or m.get("census", {}).get("n_battery_calls_on_the_real_path")]
    binding = [m for m in sw["modules"]
               if m.get("rule22_binds_this_module")
               and not m.get("rule22", {}).get("rule22_complete")]
    return {
        "protocol": PROTOCOL,
        "as_of_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "read_from_tree": str(AUDIT_ROOT),
        "auditor": auditor_identity(),
        "what_this_answers": {
            "a": ("does a budget comparison read a PROCESS-WIDE high-water "
                  "(ru_maxrss, cgroup memory.peak) rather than a per-stage "
                  "or per-fixture DELTA"),
            "b": ("is the BUDGET's own scope narrower than the "
                  "MEASUREMENT's -- one stage's, one fixture's, against a "
                  "number the whole process set"),
            "c": ("is a battery invoked from the REAL path, in the same "
                  "process, so that (a) and (b) can meet"),
            "rule22": ("is the producing-code digest, the import closure "
                       "and HEAD captured AT IMPORT, at EMIT, or not at "
                       "all (R-605)"),
        },
        "modules": sw["modules"],
        "totals": {
            "n_modules": len(sw["modules"]),
            "n_budget_comparisons": sum(
                m.get("census", {}).get("n_budget_comparisons", 0)
                for m in sw["modules"]),
            "n_deltas": sum(m.get("census", {}).get("n_deltas", 0)
                            for m in sw["modules"]),
            "n_scope_mismatch_gating_a_refusal": sum(
                m.get("census", {}).get("n_scope_mismatch", 0)
                for m in sw["modules"]),
            "n_scope_mismatch_inside_an_assertion": sum(
                m.get("census", {}).get(
                    "n_scope_mismatch_inside_an_assertion", 0)
                for m in sw["modules"]),
            "n_assertions_whose_verdict_is_ambient": sum(
                m.get("ambient_state_census", {}).get("n_flagged", 0)
                for m in sw["modules"]),
            "n_battery_calls_on_the_real_path": sum(
                m.get("census", {}).get(
                    "n_battery_calls_on_the_real_path", 0)
                for m in sw["modules"]),
            "n_modules_rule22_complete": sum(
                1 for m in sw["modules"]
                if m.get("rule22", {}).get("rule22_complete")),
            "n_modules_rule22_binds_and_incomplete": len(binding),
        },
        "modules_with_a_live_finding": [m["path"] for m in live],
        "rule22_binds_and_incomplete": [
            {"path": m["path"], "status": m["rule22"]["status"]}
            for m in binding],
        "supersedes": (supersession_block(Path(supersedes), sw)
                       if supersedes else None),
        "decides_nothing": (
            "the owners fix: BE 59 for the builders, DE 90 for the runner. "
            "This sweep names sites and counts predicates"),
    }


# ------------------------------------------------------------- falsifiers

def _scratch(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(body)
    return p


def _audit_text(tmp: Path, name: str, body: str) -> dict:
    return audit_module(_scratch(tmp, name, body), "planted")


CLEAN_SRC = '''
import resource
MEM_CAP_GB = 8.0


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def build(day):
    peak = _rss_gb()
    if peak > MEM_CAP_GB:
        raise RuntimeError("over the cap")
    return {"day": day}


def selftest():
    return 0


def main(argv=None):
    if "--selftest" in (argv or []):
        return selftest()
    return build("20260903")
'''

PLANTED_BATTERY_SRC = CLEAN_SRC.replace(
    '''    peak = _rss_gb()''', '''    selftest()
    peak = _rss_gb()''')

GUARDED_FIXTURE_KW_SRC = '''
def emit(where, fixture=False):
    return {"where": where, "fixture": fixture}


def build(day):
    return emit("real", fixture=False)


def main(argv=None):
    if "--selftest" in (argv or []):
        return emit("the fixture receipt", fixture=True)
    return build("20260903")
'''

UNGUARDED_FIXTURE_KW_SRC = GUARDED_FIXTURE_KW_SRC.replace(
    '''    return build("20260903")''',
    '''    emit("on the real path", fixture=True)
    return build("20260903")''')

PLANTED_MISMATCH_SRC = '''
import resource
STAGE_BUDGETS_GB = {"A0": 3.0}
FIXTURE_STAGE_BUDGETS_GB = {"A0": 0.7}


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def build(day, fixture=False):
    budgets = FIXTURE_STAGE_BUDGETS_GB if fixture else STAGE_BUDGETS_GB
    b = budgets["A0"]
    row = {"peak_gb": _rss_gb()}
    row["within_budget"] = row["peak_gb"] <= b
    if not row["within_budget"]:
        raise RuntimeError("over its stage budget")
    return row
'''

PLANTED_DELTA_SRC = PLANTED_MISMATCH_SRC.replace(
    '''    row = {"peak_gb": _rss_gb()}''',
    '''    base = _rss_gb()
    row = {"peak_gb": _rss_gb() - base}''')

PLANTED_EMIT_DIGEST_SRC = '''
import hashlib
from pathlib import Path


def emit():
    return {"producing_code_sha256":
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
'''

PLANTED_IMPORT_DIGEST_SRC = '''
import hashlib
from pathlib import Path

LAUNCH_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def emit():
    return {"producing_code_sha256": LAUNCH_DIGEST}
'''

#: THE 26-SECOND SHAPE, planted. `refuses(...)` sees a needle ONLY when the
#: calling process does NOT hold the lock -- so the verdict is about the
#: process, not about `run_day`. Nothing in the assertion mentions a lock.
AMBIENT_26_SECOND_SRC = '''
import fcntl


def refuses(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def run_day(day, book):
    with open("/home/yuqing/ctaNew/data/.heavy_run.lock") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return {"day": day}


def ok(cond, label):
    assert cond, label


def selftest():
    ok(refuses(lambda: run_day("2026-09-03", "/b.pkl")),
       "a real day refuses")
'''

#: THE FIXED SHAPE: the observation is HANDED IN, so the assertion is about
#: the property and gives the same answer in any process.
AMBIENT_INJECTED_SRC = '''
def ok(cond, label):
    assert cond, label


def peak_stage_predicate(stages, declared):
    return {"holds": stages["S1"]["growth_mb"] <= declared}


def check(stages, declared, obs):
    ok(peak_stage_predicate(stages, declared)["holds"], "within budget")
    ok(obs["flock_modes_on_the_inode"] is not None, "the lock was observed")
'''

PLANTED_LOOKALIKE_SRC = '''
BUDGET = 700.0


def check(declaration):
    plan = declaration["memory_plan"]
    return plan["peak_rss_mb"] < BUDGET
'''


def selftest() -> tuple:                                      # noqa: C901
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

    tmp = Path(tempfile.mkdtemp(prefix="da77_", dir=os.environ.get(
        "DA_SCRATCH") or tempfile.gettempdir()))

    # -- A. THE TREE UNDER AUDIT -----------------------------------------
    ck("THE SWEEP READS THE LEDGER TREE, NOT THIS SEAT'S WORKTREE COPY. "
       "***An earlier run of this instrument reported the DE runner FLAGGED "
       "for a comparison DE had already removed -- it was reading my "
       "worktree's stale copy. A finding about the wrong copy of a file is "
       "the R-601 class: an address that names a different object***",
       str(AUDIT_ROOT) == "/home/yuqing/ctaNew",
       f"AUDIT_ROOT={AUDIT_ROOT} (PM_DATA_ROOT), auditor's own tree={REPO}")

    # -- B. THE REAL KNOWN-BAD: the bytes the smoke died on ---------------
    smoke_bytes = subprocess.run(
        ["git", "-C", str(AUDIT_ROOT), "show",
         "77c723e:live/pm_research/de_multiday_gate1_runner.py"],
        capture_output=True, text=True)
    if smoke_bytes.returncode == 0:
        #: DA 155: rc 0 IS NOT ENOUGH. This hands `git show`'s stdout
        #: straight to the auditor; under concurrent git activity that
        #: text can arrive unusable, and an uncaught SyntaxError killed
        #: the run rather than failing a cell.
        try:
            old = _audit_text(tmp, "runner_at_77c723e.py", smoke_bytes.stdout)
        except AuditRefused as _e:
            ck("THE KNOWN-BAD AT 77c723e IS REACHABLE", False,
               f"`git show` returned rc 0 and {len(smoke_bytes.stdout)} "
               f"bytes that do not parse -- {str(_e)[:110]}")
            old = None
        new = audit_module(
            AUDIT_ROOT / "live/pm_research/de_multiday_gate1_runner.py",
            "DE_runner")
        old_mm = [c for c in old["budget_comparisons"]
                  if c["verdict"] == SCOPE_MISMATCH
                  and c.get("gates") == GATES_REFUSAL]
        new_mm = [c for c in new["budget_comparisons"]
                  if c["verdict"] == SCOPE_MISMATCH
                  and c.get("gates") == GATES_REFUSAL]
        ck("THE KNOWN-BAD IS THE ARTIFACT THAT CAUSED THE FINDING, NOT A "
           "PLANT: the runner AT 77c723e -- the bytes the 09-03 smoke died "
           "on -- is FLAGGED with a process-wide high-water against a "
           "fixture budget GATING A REFUSAL, and the SAME module at the "
           "ledger tip is not, because DE 89 made the budget a growth. "
           "***An instrument that cannot flag the case that motivated it is "
           "not an instrument***",
           len(old_mm) >= 1 and len(new_mm) == 0,
           f"77c723e: {len(old_mm)} refusing mismatch(es) at line(s) "
           f"{[c['line'] for c in old_mm]}; tip: {len(new_mm)}")
        ck("AND (c) IS STILL LIVE AT THE TIP: the battery is invoked from "
           "the REAL day path in the same process, at BOTH revisions -- DE "
           "89 closed the measurement half and the seam itself is open. "
           "***A NEW narrow-scope budget anywhere inside that battery "
           "reproduces the class***",
           new["census"]["n_battery_calls_on_the_real_path"] >= 1
           and old["census"]["n_battery_calls_on_the_real_path"] >= 1,
           f"tip: {[ (f['caller'], f['callee'], f['line']) for f in new['fixture_on_real_path']['findings'] ]}")
    else:
        ck("THE KNOWN-BAD AT 77c723e IS REACHABLE", False,
           "git show failed: " + smoke_bytes.stderr[:120])

    # -- B2. DA 155: UNUSABLE INPUT REFUSES BY NAME, BOTH DIRECTIONS -----
    _ok_src = _scratch(tmp, "parses_fine.py", CLEAN_SRC)
    _bad_src = _scratch(tmp, "does_not_parse.py", "def f(:\n")
    _empty = _scratch(tmp, "empty.py", "")
    try:
        audit_module(_bad_src, "planted")
        _bad_verdict = "ADMITTED"
    except AuditRefused as _e:
        _bad_verdict = str(_e).split(":")[0].replace("REFUSED ", "")
    except SyntaxError:
        _bad_verdict = "RAISED_SyntaxError"
    _ok_verdict = audit_module(_ok_src, "planted")["verdict"]
    _empty_verdict = audit_module(_empty, "planted")["verdict"]
    ck("DA 155 -- UNPARSEABLE SOURCE IS A NAMED REFUSAL, NOT A TRACEBACK, "
       "AND REAL SOURCE STILL AUDITS. This battery hands `audit_module` the "
       "stdout of a `git show` and text written to scratch; a bare "
       "SyntaxError there killed the whole run with rc 1 and stdout "
       "truncated mid-cell -- the signature seen twice in a landing window "
       "and never since. ***The verdict now depends on the SUBJECT's bytes: "
       "unusable bytes refuse by name and the caller decides***",
       _bad_verdict.startswith(UNPARSEABLE)
       and "does_not_parse.py" in _bad_verdict
       and _ok_verdict != UNPARSEABLE and _empty_verdict != UNPARSEABLE,
       f"unparseable -> {_bad_verdict}; real source -> {_ok_verdict}; an "
       f"EMPTY file still audits ({_empty_verdict}) because empty is valid "
       f"Python and is a fact about the module, not an unusable input")
    ck("DA 155 -- AND THE COMMITTED STATE IS THREE STATES: a scratch module "
       "OUTSIDE the tree reads NOT_IN_THIS_TREE rather than 'uncommitted'. "
       "`git status --porcelain -- <path>` exits 128 there, and the bool "
       "collapsed that into False -- so every planted module this battery "
       "audits was reported as uncommitted code. Absence read as a "
       "negative, in my own instrument",
       committed_state(_ok_src, root=AUDIT_ROOT) == NOT_IN_TREE
       #: ITS OWN TREE, not AUDIT_ROOT -- this is REV 131's red: run from
       #: a worktree the auditor's file is not under AUDIT_ROOT, so asking
       #: the main tree about it returned NOT_IN_THIS_TREE and the cell
       #: failed from every worktree while the code was correct.
       and committed_state(Path(__file__).resolve()) in
           (IN_TREE_CLEAN, IN_TREE_DIRTY)
       #: AND THE DISAGREEMENT IS THE POINT, asserted so it cannot quietly
       #: go away: from a worktree the two questions differ, and from the
       #: main tree they agree. Either way the file IS in a tree.
       and (committed_state(Path(__file__).resolve(), root=AUDIT_ROOT)
            == NOT_IN_TREE) == (not Path(__file__).resolve()
                                .is_relative_to(AUDIT_ROOT))
       and _is_committed(_ok_src) is False,
       f"scratch module -> {committed_state(_ok_src)}; this auditor's own "
       f"file -> {committed_state(Path(__file__).resolve())}; the bool "
       f"still answers False for the scratch path, which is why it needed "
       f"a name beside it")

    # -- C. PLANTED FIXTURE-IN-REAL-PATH, both directions ----------------
    clean = _audit_text(tmp, "clean_mod.py", CLEAN_SRC)
    planted = _audit_text(tmp, "planted_battery.py", PLANTED_BATTERY_SRC)
    ck("A PLANTED FIXTURE-IN-REAL-PATH IS FLAGGED AND THE SAME MODULE "
       "WITHOUT IT ADMITS: `build()` calling `selftest()` is a finding; the "
       "CLI's own `if \"--selftest\" in argv: return selftest()` is not",
       planted["census"]["n_battery_calls_on_the_real_path"] == 1
       and planted["verdict"] == "FLAGGED"
       and clean["census"]["n_battery_calls_on_the_real_path"] == 0
       and clean["verdict"] == "CLEAN",
       f"planted -> {planted['verdict']} at "
       f"{[f['caller'] + '->' + f['callee'] for f in planted['fixture_on_real_path']['findings']]}; "
       f"clean -> {clean['verdict']}")

    guarded_kw = _audit_text(tmp, "guarded_kw.py", GUARDED_FIXTURE_KW_SRC)
    unguarded_kw = _audit_text(tmp, "unguarded_kw.py",
                               UNGUARDED_FIXTURE_KW_SRC)
    ck("AND THE GUARD RULE APPLIES TO THE FIXTURE-ARGUMENT HALF TOO: "
       "`emit(fixture=True)` inside a CLI's own `if a.selftest:` branch is "
       "NOT a finding; the same call on an unguarded path IS. ***The first "
       "half checked guards and this half did not, so this sweep flagged "
       "its own seat's book verifier for a call sitting inside the battery "
       "branch -- a rule that holds on one half of a check and not the "
       "other is two rules***",
       guarded_kw["census"]["n_battery_calls_on_the_real_path"] == 0
       and unguarded_kw["census"]["n_battery_calls_on_the_real_path"] == 1,
       f"guarded -> {guarded_kw['verdict']}; unguarded -> "
       f"{unguarded_kw['verdict']} at "
       f"{[f['caller'] for f in unguarded_kw['fixture_on_real_path']['findings']]}")

    # -- D. PLANTED SCOPE MISMATCH, and its DELTA repair -----------------
    mism = _audit_text(tmp, "planted_mismatch.py", PLANTED_MISMATCH_SRC)
    delta = _audit_text(tmp, "planted_delta.py", PLANTED_DELTA_SRC)
    m_rows = [c for c in mism["budget_comparisons"]
              if c["verdict"] == SCOPE_MISMATCH]
    ck("A PLANTED PER-STAGE/FIXTURE BUDGET AGAINST `ru_maxrss` IS FLAGGED "
       "AS A SCOPE MISMATCH THAT GATES A REFUSAL -- through the flag "
       "(`row['within_budget'] = ...` then `if not ...: raise`) -- AND THE "
       "DELTA FORM OF THE SAME CODE ADMITS. ***That delta is the repair REV "
       "53 names, so the instrument must be able to see it arrive***",
       len(m_rows) == 1 and m_rows[0]["gates"] == GATES_REFUSAL
       and mism["verdict"] == "FLAGGED"
       and delta["census"]["n_scope_mismatch"] == 0
       and delta["census"]["n_deltas"] == 1,
       f"mismatch -> {m_rows[0]['verdict'][:44]} / {m_rows[0]['gates']}; "
       f"delta -> {delta['census']['n_deltas']} delta(s), "
       f"{delta['census']['n_scope_mismatch']} mismatch(es)")

    # -- E. PLANTED EMIT-TIME DIGEST vs IMPORT-TIME ----------------------
    emit_d = _audit_text(tmp, "planted_emit_digest.py",
                         PLANTED_EMIT_DIGEST_SRC)
    imp_d = _audit_text(tmp, "planted_import_digest.py",
                        PLANTED_IMPORT_DIGEST_SRC)
    ck("A PLANTED EMIT-TIME DIGEST IS FLAGGED **AS EMIT-TIME**, AND THE "
       "SAME CAPTURE AT MODULE LEVEL READS AS IMPORT-TIME. ***This is the "
       "whole of rule 22: a digest read when the receipt is written names "
       "the file as it is THEN, and the committed-bytes guard passes "
       "because the replacement is committed***",
       emit_d["rule22"]["status"]["producing_code_digest"] == AT_EMIT
       and imp_d["rule22"]["status"]["producing_code_digest"] == AT_IMPORT,
       f"emit-time -> {emit_d['rule22']['status']['producing_code_digest']};"
       f" module-level -> "
       f"{imp_d['rule22']['status']['producing_code_digest']}")

    # -- E2. REV 58 section 1.2: THE CLASS, not its instances ------------
    amb_bad = _audit_text(tmp, "ambient_26s.py", AMBIENT_26_SECOND_SRC)
    amb_ok = _audit_text(tmp, "ambient_injected.py", AMBIENT_INJECTED_SRC)
    bad_rows = amb_bad["ambient_state_census"]["rows"]
    ok_rows = amb_ok["ambient_state_census"]["rows"]
    ck("REV 58 section 1.2 -- THE GENERAL QUESTION IS ASKED NOW: ***does an "
       "assertion's verdict depend on state this process happens to be IN, "
       "rather than on the property?*** Memory was ONE ambient and the "
       "sweep asked only about memory, so the 26-second refusal -- a "
       "`refuses(lambda: run_day(...))` whose needle appears only when the "
       "caller does NOT hold the lock -- was none of its three questions. "
       "The planted shape is FLAGGED, and it takes TWO HOPS to see: the "
       "assertion mentions no lock, the lambda mentions no lock, and "
       "`run_day` holds the flock",
       any(r["verdict"] == "REFUSAL_EXPECTATION_WHOSE_CAUSE_CAN_BE_AMBIENT"
           and any(w["callee"] == "run_day" and "lock" in w["ambient_kinds"]
                   for w in r["wrapped_callees_reading_an_ambient"])
           for r in bad_rows),
       f"planted 26-second shape -> "
       f"{[ (r['verdict'], r['ambient_kinds']) for r in bad_rows ]}")
    ck("AND THE FIXED SHAPE ADMITS: an assertion over an observation HANDED "
       "IN (`stages`, `obs`) is INJECTED_OBSERVATION -- it gives the same "
       "answer in any process, which is the whole difference. ***Six "
       "ambients are tested (memory, lock, cgroup, cwd, worktree, clock, "
       "process), and the truth-bearing OPERANDS are read rather than the "
       "whole call: reading the message text made every `ok(x, f\"...{"
       "__file__}...\")` a finding***",
       all(r["verdict"] == "INJECTED_OBSERVATION" for r in ok_rows)
       and amb_ok["ambient_state_census"]["n_flagged"] == 0
       and set(AMBIENT_READERS) >= {"memory", "lock", "cgroup", "cwd",
                                    "worktree", "clock"}
       and len(amb_bad["ambient_state_census"][
           "what_this_CANNOT_see_from_source"]) >= 3,
       f"injected -> {amb_ok['ambient_state_census']['n_flagged']} flagged "
       f"of {len(ok_rows)} ambient-touching row(s); "
       f"{len(AMBIENT_READERS)} ambients tested; the census states "
       f"{len(amb_bad['ambient_state_census']['what_this_CANNOT_see_from_source'])} "
       f"things it cannot see from source (magnitude, runtime-chosen "
       f"callables, read-but-not-decisive)")
    # -- F. THE LOOKALIKE IS REPORTED, NEVER COUNTED, NEVER DROPPED ------
    look = _audit_text(tmp, "planted_lookalike.py", PLANTED_LOOKALIKE_SRC)
    ck("A SUBSCRIPT THAT ONLY LOOKS LIKE A MEASUREMENT IS REPORTED AS A "
       "LOOKALIKE AND IS NOT COUNTED AS A BUDGET COMPARISON. ***DE's "
       "battery reads `plan['peak_rss_mb']` out of a DECLARATION file; "
       "counting it would invent a finding, dropping it would hide one "
       "(rule 11)***",
       look["census"]["n_unresolved_measure_lookalikes"] == 1
       and look["census"]["n_budget_comparisons"] == 0
       and look["verdict"] == "CLEAN",
       f"{look['census']['n_unresolved_measure_lookalikes']} lookalike(s), "
       f"{look['census']['n_budget_comparisons']} budget comparison(s)")

    # -- G. THE GUARD RULE, both directions ------------------------------
    ck("A GUARD IS A DISPATCH ON THE BATTERY SWITCH, NOT A TEST THAT "
       "MENTIONS IT: `a.selftest`, `a.selftest or a.fixture` and "
       "`--selftest` are guards; `not fixture and not payload[...]` -- DE's "
       "actual early exit above the battery call -- is NOT. ***Matching the "
       "word alone excused the exact call this sweep exists to find, and "
       "the sweep reported the runner CLEAN***",
       _guard_admits("a.selftest")
       and _guard_admits("a.selftest or a.fixture")
       and _guard_admits('"--selftest" in argv')
       and not _guard_admits(
           'not fixture and not payload["source_identity"]["x"]')
       and not _guard_admits("a.output is None")
       #: AND THE TWO FORMS ARE OPPOSITE: an ENCLOSING positive switch puts
       #: the code on the battery branch; a dominating EARLY EXIT on the
       #: same positive switch leaves the REAL path below it, and only its
       #: NEGATION guards what follows.
       and _guard_admits("EARLY_EXIT: not a.selftest")
       and not _guard_admits("EARLY_EXIT: a.selftest")
       and not _guard_admits('EARLY_EXIT: "--selftest" in argv')
       and _guard_admits('EARLY_EXIT: not "--selftest" in argv'),
       "three positives and two negatives on enclosing guards; on early "
       "exits the polarity is REVERSED and driven both ways")

    # -- H. THE REAL SWEEP, and its census --------------------------------
    rep = build_report()
    by = {m["path"]: m for m in rep["modules"]}
    frag = by["live/pm_research/be_gate1_fragment.py"]
    tape = by["live/pm_research/be_gate1_state_tape.py"]
    book = by["live/pm_research/be_daybook_build.py"]
    ck("BE's FRAGMENT AND TAPE ARE CLEAN, AND THE CENSUS SAYS WHY RATHER "
       "THAN CERTIFYING SILENCE: one budget comparison each, both reading "
       "`ru_maxrss` -- a PROCESS-WIDE high-water -- against `MEM_CAP_GB`, "
       "which is the rule-20 wrapper's own PROCESS-WIDE 8 GB. ***Same "
       "instrument, matched scopes: that pairing is correct, and it is the "
       "pairing, not the instrument, that the smoke got wrong***",
       frag["verdict"] == "CLEAN" and tape["verdict"] == "CLEAN"
       and frag["census"]["n_budget_comparisons"] == 1
       and tape["census"]["n_budget_comparisons"] == 1
       and frag["census"]["n_scope_matched"] == 1
       and tape["census"]["n_scope_matched"] == 1
       and frag["census"]["n_battery_calls_on_the_real_path"] == 0
       and tape["census"]["n_battery_calls_on_the_real_path"] == 0,
       f"fragment L{frag['budget_comparisons'][0]['line']} and tape "
       f"L{tape['budget_comparisons'][0]['line']}: "
       f"{frag['budget_comparisons'][0]['verdict']}")
    bm = [c for c in book["budget_comparisons"]
          if c["verdict"] == SCOPE_MISMATCH]
    deltas = [c for c in book["budget_comparisons"] if c["is_delta"]]
    #: TWO STATES, BOTH NAMED. Round 77 found the shape here and BE 60
    #: landed the growth budget; a check asserting `len(bm) == 1` was
    #: pinned to the FINDING rather than to the property, and it broke on
    #: the FIX -- rounds 69/72/74/75/77/78, the same defect again. What is
    #: asserted now is the disjunction, with the state SAID.
    open_state = bool(bm) and bm[0]["gates"] == GATES_REFUSAL \
        and bm[0]["budget_scope"].startswith(FIXTURE_SCOPE)
    closed_state = not bm and bool(deltas)
    ck("BE's DAYBOOK: THE SHAPE ROUND 77 FOUND IS EITHER STILL OPEN (a "
       "per-stage/fixture budget against `ru_maxrss`, GATING A REFUSAL) OR "
       "CLOSED BY A DELTA -- and the census says WHICH. ***Round 77's "
       "check asserted the FINDING (`len(bm) == 1`) and broke on BE 60's "
       "FIX: a check pinned to a defect fails when the defect is repaired, "
       "which is the sixth instance of this seat's own recurring class***",
       open_state or closed_state,
       (f"OPEN: be_daybook_build.py:{bm[0]['line']} `{bm[0]['expr']}` -- "
        f"{bm[0]['budget_scope']}, {bm[0]['gates']}"
        if open_state else
        f"CLOSED: 0 refusing scope mismatches and "
        f"{len(deltas)} delta-measured budget comparison(s) at "
        f"line(s) {[c['line'] for c in deltas]} -- BE 60 landed the growth "
        f"budget, and the census carries the change rather than a stale "
        f"assertion"))

    da = [m for m in rep["modules"] if m["role"] == "DA_instrument"]
    ck("AND THIS SEAT'S OWN INSTRUMENTS CARRY NEITHER HALF: no scope "
       "mismatch and no battery on a real path across all seven. The one "
       "memory refusal among them -- the book verifier's -- is measured on "
       "CURRENT rss, which FALLS, so it cannot inherit a freed peak",
       all(m["census"]["n_scope_mismatch"] == 0
           and m["census"]["n_battery_calls_on_the_real_path"] == 0
           for m in da)
       and by["live/pm_research/da_book_verify.py"][
           "census"]["n_current_rss"] == 1,
       f"{len(da)} DA instruments, "
       f"{sum(m['census']['n_budget_comparisons'] for m in da)} budget "
       f"comparison(s) among them, 0 mismatches, 0 batteries on a real "
       f"path")

    # -- I. RULE 22, the census REV 51 section 3 counted at zero ----------
    def _r22(path):
        st = by[path]["rule22"]["status"]
        return {k: v.split("__VIA_")[0] for k, v in st.items()}, st

    be_state = {Path(p2).name: _r22(p2) for p2 in BE_PRODUCERS}
    be_closure_at_import = all(
        v[0]["import_closure"] == AT_IMPORT for v in be_state.values())
    ck("RULE 22 / R-605, COMPUTED AT THE SOURCE -- AND THE CAPTURE MAY LIVE "
       "IN A SHARED MODULE. BE 60 put it in `be_rule22.py` and each "
       "producer calls `_R22.init(...)` AT MODULE LEVEL, so the closure is "
       "captured at IMPORT even though the construct is in another file. "
       "***A per-FILE detector reported ABSENT for all three, which is a "
       "FALSE NEGATIVE about another seat's work -- exactly the 'half a "
       "predicate reported as the predicate' this sweep exists to catch. "
       "One hop, and the WHEN decided by the IMPORTER's call site.*** DE's "
       "runner captures it directly, at module level, in its own file",
       be_closure_at_import
       and by["live/pm_research/de_multiday_gate1_runner.py"][
           "rule22"]["rule22_complete"] is True
       and rep["totals"]["n_modules_rule22_complete"] == sum(
           1 for m in rep["modules"]
           if m.get("rule22", {}).get("rule22_complete"))
       and rep["totals"]["n_modules_rule22_complete"] >= 1,
       "; ".join(f"{k}: closure={v[1]['import_closure'][:9]}"
                 f"{'(via be_rule22)' if '__VIA_' in v[1]['import_closure'] else ''}"
                 f", head={v[1]['head_sha'][:9]}"
                 for k, v in be_state.items()))
    _others = {Path(k).name: len(by[k]["rule22"].get(
        "digests_of_other_files") or []) for k in BE_PRODUCERS}
    _tape = by["live/pm_research/be_gate1_state_tape.py"]["rule22"]
    ck("REV 60 section 5 -- A PRODUCING-CODE DIGEST IS A DIGEST OF **THE "
       "MODULE'S OWN FILE**, and the classifier is keyed on WHOSE BYTES "
       "are hashed rather than on where the call sits. ***Round 77 counted "
       "every `sha256(<path>.read_bytes())` reached from an emit path -- so "
       "a book's READ-BACK, a receipt's read-back, an INPUT fragment and "
       "BE's own battery control FOR THIS VERY PROPERTY all read as "
       "producing-code digests, and all three BE producers then sat "
       "FALSELY in `rule22_binds_and_incomplete`.*** Those sites are now "
       "counted separately as digests of OTHER files, and the closure -- "
       "which necessarily contains the producing module -- covers the "
       "construct, by the route the receipt names",
       all(by[k]["rule22"]["status"]["producing_code_digest"].startswith(
           AT_IMPORT) for k in BE_PRODUCERS)
       and sum(_others.values()) >= 3
       and by["live/pm_research/be_gate1_fragment.py"][
           "rule22"]["rule22_complete"] is True
       and by["live/pm_research/be_daybook_build.py"][
           "rule22"]["rule22_complete"] is True,
       f"other-file digests no longer counted as producing code: {_others}; "
       f"fragment and daybook now COMPLETE")
    ck("AND THE ONE FACT THAT MAY BE GENUINELY INCOMPLETE SURVIVES THE "
       "CORRECTION: `be_gate1_state_tape`'s HEAD is captured at EMIT, not "
       "at import, and it is the only module left in "
       "`rule22_binds_and_incomplete`. ***A correction that cleared the "
       "false three AND the true one would have been a looser instrument, "
       "not a better one***",
       _tape["status"]["head_sha"].startswith(AT_EMIT)
       and _tape["rule22_complete"] is False
       and [x["path"] for x in rep["rule22_binds_and_incomplete"]]
       == ["live/pm_research/be_gate1_state_tape.py"],
       f"be_gate1_state_tape head={_tape['status']['head_sha'][:34]}; "
       f"binds-and-incomplete = "
       f"{[Path(x['path']).name for x in rep['rule22_binds_and_incomplete']]}")

    typed = by["live/pm_research/be_daybook_build.py"][
        "rule22"]["typed_stamps_of_this_run"]
    ck("AND THE DAYBOOK'S ONE PROVENANCE LITERAL IS TYPED, NOT DERIVED: "
       "`seam.commit` is the string \"6f134a6\" in the source. A TYPED "
       "provenance field cannot move when the code does. ***The historical "
       "citations in a `supersedes` block are NOT counted here -- citing a "
       "past artifact by literal is the only honest form, and a checker "
       "that conflated the two would report every correction as a defect***",
       len(typed) == 1 and typed[0]["key"] == "commit"
       and typed[0]["looks_like_a_commit_sha"] is True
       and any(t["kind"].startswith("HISTORICAL")
               for t in by["live/mm_research/e2_a_declare.py"][
                   "rule22"]["typed_literals"]),
       f"be_daybook_build.py:{typed[0]['line']} commit=\"{typed[0]['value']}\""
       f" -- and e2_a_declare's supersedes literal reads as "
       f"{by['live/mm_research/e2_a_declare.py']['rule22']['typed_literals'][0]['kind']}")

    _tot = sum(m["ambient_state_census"]["n_flagged"] for m in rep["modules"]
               if m.get("census"))
    ck("AND THE PROGRAMME'S CENSUS IS REPORTED PER MODULE AND PER AMBIENT "
       "KIND, as a count and not a verdict: an ambient READ is not by "
       "itself a defect, and this sweep names sites for their owners "
       "rather than ruling on them",
       _tot == rep["totals"]["n_assertions_whose_verdict_is_ambient"]
       and rep["totals"]["n_assertions_whose_verdict_is_ambient"] >= 1,
       f"{rep['totals']['n_assertions_whose_verdict_is_ambient']} "
       f"ambient-verdict assertions across "
       f"{sum(1 for m in rep['modules'] if m.get('census') and m['ambient_state_census']['n_flagged'])} "
       f"module(s); by kind: "
       f"{ {k: sum(m['ambient_state_census']['by_ambient_kind'].get(k, 0) for m in rep['modules'] if m.get('census')) for k in sorted(AMBIENT_READERS)} }")

    # -- I2. THE CENSUS AGREES WITH THE ROWS IT SUMMARISES ---------------
    bad = []
    for m in rep["modules"]:
        if not m.get("census"):
            continue
        rows = m["budget_comparisons"]
        exp = {
            "n_budget_comparisons": len(rows),
            "n_deltas": sum(1 for c in rows if c["is_delta"]),
            "n_scope_mismatch": sum(1 for c in rows
                                    if c["verdict"] == SCOPE_MISMATCH),
            "n_scope_mismatch_inside_an_assertion": sum(
                1 for c in rows if c["verdict"].startswith("SCOPE_MISMATCH")
                and c["verdict"] != SCOPE_MISMATCH),
            "n_unresolved_measure_lookalikes":
                len(m["unresolved_measure_lookalikes"]),
            "n_battery_calls_on_the_real_path":
                len(m["fixture_on_real_path"]["findings"])}
        for k, v in exp.items():
            if m["census"][k] != v:
                bad.append(f"{m['path']}.{k}: census {m['census'][k]} "
                           f"vs rows {v}")
    tot_ok = (rep["totals"]["n_scope_mismatch_inside_an_assertion"]
              == sum(m["census"]["n_scope_mismatch_inside_an_assertion"]
                     for m in rep["modules"] if m.get("census")))
    ck("EVERY COUNT IN THE CENSUS IS RECOMPUTED FROM THE ROWS IT SUMMARISES "
       "AND THE TOTALS FROM THE MODULES. ***The first emission of this "
       "receipt reported ZERO assertion-only mismatches while DE's row "
       "carried one: the count tested `startswith(<the whole refusing "
       "verdict string>)` instead of the prefix. A count pinned to how a "
       "verdict is SPELLED rather than what it IS -- rounds 69, 72, 74, 75, "
       "and here again in the summary of my own rows***",
       not bad and tot_ok,
       f"{sum(1 for m in rep['modules'] if m.get('census'))} module "
       f"censuses recomputed from their rows, 0 disagreements; the "
       f"assertion-only total is "
       f"{rep['totals']['n_scope_mismatch_inside_an_assertion']}"
       if not bad else "; ".join(bad[:3]))

    # -- J. the report decides nothing ------------------------------------
    ck("THE REPORT NAMES SITES AND COUNTS PREDICATES AND DECIDES NOTHING: "
       "every module row carries its file digest, the tree it was read "
       "from, and whether that tree has it committed -- the owners fix "
       "(BE 59, DE 90)",
       all("sha256" in m and "read_from_tree" in m
           for m in rep["modules"] if m.get("census"))
       and rep["totals"]["n_modules"] == 12
       and "decides_nothing" in rep,
       f"{rep['totals']['n_modules']} modules; "
       f"{rep['totals']['n_budget_comparisons']} budget comparisons, "
       f"{rep['totals']['n_deltas']} deltas, "
       f"{rep['totals']['n_scope_mismatch_gating_a_refusal']} refusing "
       f"mismatches, "
       f"{rep['totals']['n_battery_calls_on_the_real_path']} batteries on a "
       f"real path")

    print(f"\nSELFTEST {'OK' if not fails else 'FAILED'} -- "
          f"{len(checks)} checks, {fails} failure(s)")
    return checks, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--supersedes", type=Path, default=None,
                    help="a prior census this re-emission supersedes; the "
                         "R-608 PAIR and what moved are computed")
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        checks, n_fail = selftest()
        if a.output:
            #: an in-band re-emission carries the R-608 PAIR of the census
            #: it supersedes AND its own battery -- a superseding artifact
            #: that dropped the checks would be a weaker statement wearing
            #: a later name.
            rep = build_report(supersedes=a.supersedes)
            rep["checks"] = checks
            rep["n_checks"] = len(checks)
            rep["n_failed"] = n_fail
            rep["both_directions"] = True
            a.output.write_text(json.dumps(rep, indent=2, sort_keys=True,
                                           default=str) + "\n")
        return 1 if n_fail else 0
    if a.sweep:
        rep = build_report(supersedes=a.supersedes)
        if a.output:
            a.output.write_text(json.dumps(rep, indent=2, sort_keys=True,
                                           default=str) + "\n")
        print(json.dumps(rep["totals"], indent=1))
        return 0
    ap.error("--selftest [--output <path>] or --sweep [--output <path>]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
