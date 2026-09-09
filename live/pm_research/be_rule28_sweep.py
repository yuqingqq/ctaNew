"""RULE 28 ON THIS SEAT'S SURFACE: recorded evidence with the check switched off.

SEAT_PROTOCOL rule 28 (R-848, REV's naming of four instances in one night):
*this pipeline repeatedly records or publishes the right thing and leaves the
check that would make it load-bearing switched off.* Two shapes are swept
here, both COMPUTED from the AST rather than grepped, because the defect is a
property of what the code DOES with a value and not of any word in it:

  (a) A DEFAULT THAT ANSWERS FOR SOMETHING IT WAS NEVER GIVEN.
      `_era_or_refuse(fi, None, ...)` returned a module literal for a day it
      was never told about (BE 113). REVIEW 116's `read_ledger(path,
      expect_sha256=None)` makes the UNSAFE call the shorter one. The
      question is not "does this parameter have a default" -- most should --
      it is **what happens when the caller omits it**: does the function
      REFUSE, or does it silently stand something in?

      Computable: for a parameter defaulting to `None`, find the guard that
      tests it (`x is None`, `not x`, `x is not None`) and read what that
      branch DOES. A `raise` in the omission branch is a REFUSAL. Anything
      else is a SUBSTITUTION. A non-None default with no guard at all is a
      SUBSTITUTION with no branch to inspect.

  (b) A PRODUCER THAT RETURNS EVIDENCE THE CONSUMER DISCARDS.
      REVIEW 117's `x0, _ = read_at(...)`: the staleness of the sample that
      decides a winner is computed, returned and thrown away on one line.
      Two forms are found -- an explicit `_` target, and a name that is
      BOUND and never LOADED again in its function, which is the same defect
      wearing a real identifier.

WHAT THIS INSTRUMENT IS NOT. It does not decide. It enumerates candidates
and classifies them by a computed predicate; every ranked verdict in the
filing was reached by DRIVING the site, not by reading this output. A census
that graded itself would be the very thing rule 28 is about.

LIMITS, STATED RATHER THAN LEFT TO BE DISCOVERED -- an instrument whose
limits are not written down invites a clean reading it did not earn.

  * IT GRADES PRESENCE, NOT IDENTITY. `on_omission` answers "what happens
    when the caller passes nothing". It does NOT ask whether the value the
    caller DID pass is the right one. Two of BE 114's three fixes are on
    that second axis -- `day_slugs(supply=...)` and `mask_block(sup, ...)`
    never checked that the supply names the day they were used for -- and
    the census still classifies both as SUBSTITUTES, correctly and
    unhelpfully. A clean `on_omission` column is not a clean surface.
  * IT IS INTRAPROCEDURAL. A guard in the caller, or a refusal one frame
    down, is invisible here; so is a value that flows into a dict and is
    checked later.
  * `NO_GUARD_VALUE_USED_DIRECTLY` IS NOT A VERDICT. `coin=COIN`,
    `progress=True` and `argv=None` all land there and none is a defect.
    The column separates candidates from non-candidates; the ranking in the
    filing came from driving each one.
  * DISCARDED RETURNS ARE FLAGGED WITHOUT KNOWING WHAT WAS DISCARDED. A
    deliberate, documented discard (`rc, out, err = _run(text)` in
    `be_forward_day.mutation_audit`, whose receipt says STDOUT is never
    searched) looks identical here to REVIEW 117's defect. Only the drive
    tells them apart.

SCOPE, stated so the closure claim is checkable: the ten modules of this
seat's build path -- the eight that a `day_selector`/`selector_for` drive
actually imports, plus `be_cancel_axis_null` (the pinned cascade entry point)
and `be_generation_count_derivation` (this seat's census of the book). NOT
closed over the other thirty `be_*.py` in the package, nor over the ten
non-BE modules on the same path, which are other seats' surfaces.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

#: The sweep's scope. Eight measured at runtime from a live selector drive
#: (BE 114), plus two this seat owns that a day RUN reaches rather than a
#: build.
SCOPE = (
    "be_daybook_build.py", "be_gate1_fragment.py", "be_gate1_state_tape.py",
    "be_era_for_day.py", "be_score_coverage.py", "be_rule22.py",
    "be_data_root.py", "be_forward_day.py",
    "be_cancel_axis_null.py", "be_generation_count_derivation.py",
)

OMISSION_REFUSES = "REFUSES"
OMISSION_SUBSTITUTES = "SUBSTITUTES"
NO_GUARD = "NO_GUARD_VALUE_USED_DIRECTLY"


class SweepRefused(RuntimeError):
    """A named refusal."""


def _default_kind(node) -> str:
    if isinstance(node, ast.Constant):
        return "NONE" if node.value is None else f"LITERAL({node.value!r})"
    if isinstance(node, ast.Name):
        return f"MODULE_NAME({node.id})"
    if isinstance(node, ast.Attribute):
        return "ATTRIBUTE"
    if isinstance(node, (ast.Tuple, ast.List, ast.Dict, ast.Set)):
        return "CONTAINER"
    if isinstance(node, ast.Call):
        return "CALL"
    return type(node).__name__.upper()


def _tests_param(test, name: str) -> bool:
    """Does `test` interrogate `name`'s presence?"""
    for n in ast.walk(test):
        if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) \
                and n.left.id == name and any(
                    isinstance(o, (ast.Is, ast.IsNot)) for o in n.ops):
            return True
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) \
                and isinstance(n.operand, ast.Name) and n.operand.id == name:
            return True
        if isinstance(n, ast.Name) and n.id == name \
                and isinstance(getattr(n, "ctx", None), ast.Load) \
                and isinstance(test, ast.Name):
            return True
    return False


def _omission_branch(test, name: str, body, orelse):
    """The branch that runs when the caller OMITTED `name`.

    `if x is None: A else: B` -> A. `if x is not None: A else: B` -> B. The
    polarity matters: reading the wrong branch would call a refusal a
    substitution and the census would be backwards."""
    neg = False
    for n in ast.walk(test):
        if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) \
                and n.left.id == name:
            neg = any(isinstance(o, ast.IsNot) for o in n.ops)
            break
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) \
                and isinstance(n.operand, ast.Name) and n.operand.id == name:
            neg = False
            break
    return (orelse if neg else body)


def default_census(paths) -> list:
    out = []
    for p in paths:
        src = Path(p).read_text()
        tree = ast.parse(src)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = fn.args
            pairs = list(zip(args.args[len(args.args) - len(args.defaults):],
                             args.defaults))
            pairs += [(a, d) for a, d in zip(args.kwonlyargs,
                                             args.kw_defaults) if d is not None]
            for a, d in pairs:
                name = a.arg
                verdict, ev = NO_GUARD, None
                for node in ast.walk(fn):
                    if isinstance(node, ast.If) and _tests_param(node.test,
                                                                 name):
                        br = _omission_branch(node.test, name, node.body,
                                              node.orelse)
                        raises = any(isinstance(x, ast.Raise)
                                     for b in br for x in ast.walk(b))
                        verdict = (OMISSION_REFUSES if raises
                                   else OMISSION_SUBSTITUTES)
                        ev = f"guard at line {node.lineno}"
                        if raises:
                            break
                    if isinstance(node, ast.IfExp) and _tests_param(node.test,
                                                                    name):
                        verdict, ev = (OMISSION_SUBSTITUTES,
                                       f"ternary at line {node.lineno}")
                out.append({
                    "file": Path(p).name, "func": fn.name, "line": fn.lineno,
                    "param": name, "default": _default_kind(d),
                    "on_omission": verdict, "evidence": ev,
                })
    return out


def discarded_census(paths) -> list:
    """Values a producer RETURNED and the consumer dropped."""
    out = []
    for p in paths:
        tree = ast.parse(Path(p).read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.Module)):
                continue
            loaded = {n.id for n in ast.walk(fn)
                      if isinstance(n, ast.Name)
                      and isinstance(n.ctx, ast.Load)}
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign):
                    continue
                if not isinstance(node.value, ast.Call):
                    continue
                for t in node.targets:
                    if not isinstance(t, (ast.Tuple, ast.List)):
                        continue
                    for i, el in enumerate(t.elts):
                        if not isinstance(el, ast.Name):
                            continue
                        if el.id == "_":
                            form = "UNDERSCORE"
                        elif el.id not in loaded:
                            form = "BOUND_NEVER_READ"
                        else:
                            continue
                        out.append({
                            "file": Path(p).name,
                            "func": getattr(fn, "name", "<module>"),
                            "line": node.lineno, "position": i,
                            "name": el.id, "form": form,
                            "producer": ast.unparse(node.value.func)[:60],
                        })
    return out


def sweep(root: Path | None = None, scope=SCOPE) -> dict:
    root = Path(root) if root else HERE
    paths = [root / n for n in scope]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise SweepRefused(
            f"REFUSED -- SCOPE_FILE_ABSENT: {missing}. A sweep that silently "
            f"skips a file in its own declared scope reports a clean surface "
            f"it never looked at (rule 11).")
    dc = default_census(paths)
    ds = discarded_census(paths)
    by = {}
    for r in dc:
        by[r["on_omission"]] = by.get(r["on_omission"], 0) + 1
    return {
        "protocol": "BE_RULE28_SWEEP_V1",
        "scope": list(scope),
        "n_files": len(paths),
        "defaults": {
            "n": len(dc),
            "by_on_omission": by,
            "SILENT": [r for r in dc if r["on_omission"] != OMISSION_REFUSES],
            "REFUSING": [r for r in dc if r["on_omission"]
                         == OMISSION_REFUSES],
        },
        "discarded_returns": {"n": len(ds), "rows": ds},
        "decides_nothing": "an enumeration of CANDIDATES. Every verdict in "
                           "the filing was reached by driving the site "
                           "(rule 14).",
    }


# ---------------------------------------------------------------------------
# THE FALSIFIER: it must FLAG a planted instance and REFUSE a bad input.
# ---------------------------------------------------------------------------

EXPECTED_CHECKS = 10

_PLANT = '''
def unsafe(path, expect_sha256=None):
    """REVIEW 116's shape: the UNSAFE call is the shorter one."""
    if expect_sha256 is None:
        expect_sha256 = "whatever"          # SUBSTITUTES
    return path

def safe(path, expect_sha256=None):
    if expect_sha256 is None:
        raise ValueError("expect_sha256 is required")   # REFUSES
    return path

def inverted(path, expect=None):
    if expect is not None:
        raise ValueError("no")              # the RAISE is on the PRESENT
    return path                             # branch -- SUBSTITUTES

def era(fi, era=None):
    era = fi.ERA if era is None else era    # the BE 113 shape, a ternary
    return era

def consumer():
    x0, _ = read_at(1, 2)
    y0, stale = read_at(3, 4)
    z0, unread = read_at(5, 6)
    return x0 + y0 + z0 + stale
'''


def falsify() -> int:
    import tempfile
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    td = Path(tempfile.mkdtemp(prefix="be114_sweep_"))
    (td / "plant.py").write_text(_PLANT)
    dc = {(r["func"], r["param"]): r
          for r in default_census([td / "plant.py"])}
    ok(dc[("unsafe", "expect_sha256")]["on_omission"] == OMISSION_SUBSTITUTES,
       f"POSITIVE CONTROL (a): REVIEW 116's shape is FLAGGED -- "
       f"`unsafe(path, expect_sha256=None)` substitutes on omission "
       f"({dc[('unsafe', 'expect_sha256')]['on_omission']}), so the shorter "
       f"call is the unsafe one")
    ok(dc[("safe", "expect_sha256")]["on_omission"] == OMISSION_REFUSES,
       "and the SAFE twin -- identical signature, `raise` in the omission "
       "branch -- is classified REFUSES. The census reads what the branch "
       "DOES; a signature-only check could not tell these two apart")
    ok(dc[("inverted", "expect")]["on_omission"] == OMISSION_SUBSTITUTES,
       "KNOWN-BAD ON THE POLARITY: a `raise` under `if expect is not None` "
       "is a refusal on the PRESENT case, not the omitted one, and is "
       "classified SUBSTITUTES -- reading the wrong branch would invert the "
       "whole census")
    ok(dc[("era", "era")]["on_omission"] == OMISSION_SUBSTITUTES
       and dc[("era", "era")]["default"] == "NONE",
       "and BE 113's own defect shape -- `era = fi.ERA if era is None else "
       "era` -- is flagged through the TERNARY, which is where it actually "
       "lived")
    ds = discarded_census([td / "plant.py"])
    forms = {(r["name"], r["form"]) for r in ds}
    ok(("_", "UNDERSCORE") in forms,
       "POSITIVE CONTROL (b): REVIEW 117's `x0, _ = read_at(...)` is FLAGGED")
    ok(("unread", "BOUND_NEVER_READ") in forms,
       "and so is the same defect wearing a real identifier -- `z0, unread = "
       "read_at(...)` where `unread` is never loaded again")
    ok(("stale", "BOUND_NEVER_READ") not in forms
       and ("stale", "UNDERSCORE") not in forms,
       "while a name that IS read again is NOT flagged -- the census "
       "distinguishes carrying the evidence from discarding it, which is "
       "the whole distinction rule 28 draws")
    (td / "clean.py").write_text("def f(a, b=1):\n    return a + b\n")
    ok(not discarded_census([td / "clean.py"]),
       "and a file with no unpacking yields no discarded rows -- the "
       "instrument is not flagging everything")
    try:
        sweep(root=td, scope=("nope.py",))
        ok(False, "an absent scope file must refuse")
    except SweepRefused as e:
        ok("SCOPE_FILE_ABSENT" in str(e),
           "KNOWN-BAD: a scope file that is not there REFUSES -- a sweep "
           "that skipped it would report a clean surface it never looked at")
    real = sweep()
    ok(real["n_files"] == len(SCOPE) and real["defaults"]["n"] > 0,
       f"and the real sweep runs over all {real['n_files']} scope files, "
       f"{real['defaults']['n']} defaults and "
       f"{real['discarded_returns']['n']} discarded-return rows")

    print()
    if fails:
        print(f"{checks} cells, {len(fails)} failures")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} cells, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        print(f"{checks} cells, 1 failures")
        return 1
    print(f"{checks} cells, 0 failures")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--falsify" in argv or "--selftest" in argv:
        return falsify()
    if "--sweep" in argv:
        print(json.dumps(sweep(), indent=1))
        return 0
    print("usage: be_rule28_sweep.py --falsify | --sweep")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
