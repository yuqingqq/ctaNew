#!/usr/bin/env python3
"""Mutation audit: does a suite NOTICE when a refusal is deleted?

A refusal nothing can distinguish from `pass` is not a check -- it is a comment
that raises. This walks a module's AST for `raise` sites inside named checker
functions, deletes each one in turn, and reports whether the suite goes red.

THE HARNESS ITSELF SHIPS A FALSIFIER (rule 15), because a mutation harness has
a silent failure mode that looks exactly like success: if the mutant copy fails
to run for ANY unrelated reason, every site "dies" and the report is a PERFECT
KILL RATE. That is not hypothetical -- it is how a 53/53 was produced and
corrected in this program (R-347), and how this file came to exist.

  CONTROL A  the UNMUTATED copy must PASS, or no result is reported at all
  CONTROL B  a CANARY refusal nothing tests must be reported SURVIVED, proving
             the harness can report a survivor before its zeros mean anything
  CONTROL C  an UNPARSEABLE mutant must NOT be counted as a kill
  CONTROL D  every refusal-bearing function must be IN SCOPE or EXPLICITLY
             excluded. A target list is chosen by hand, so a function left out
             is audited by nobody while the report still says "no survivors" --
             the clean sweep that is silent about what it never looked at.

Mutants are written NEXT TO the original, never in a scratch directory: a module
whose derived paths are relative to its own location behaves differently
elsewhere, which is precisely what made CONTROL A fire the first time it ran.

A kill is classified by CAUSE. An assertion-kill means a selftest named the
refusal. A crash-kill means removing the guard made something downstream throw:
still a detection, still loud, but it does not prove the refusal is ASSERTED,
and a later defensive check elsewhere would silently delete that coverage.
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

ASSERT_MARKER = "selftest failed"
CANARY = ('\n\ndef _mutation_audit_canary():\n'
          '    raise ValueError("canary: no test reaches this")\n')


class HarnessRefused(Exception):
    """The harness will not report numbers it cannot stand behind."""


def raise_sites(src: str, targets: set[str]) -> list[tuple[str, int, int]]:
    out = []
    for fn in ast.walk(ast.parse(src)):
        if isinstance(fn, ast.FunctionDef) and fn.name in targets:
            for n in ast.walk(fn):
                if isinstance(n, ast.Raise):
                    out.append((fn.name, n.lineno, n.end_lineno or n.lineno))
    return sorted(out, key=lambda s: s[1])


def delete_site(src: str, lo: int, hi: int) -> str:
    lines = src.split("\n")
    indent = len(lines[lo - 1]) - len(lines[lo - 1].lstrip())
    for i in range(lo - 1, hi):
        lines[i] = ""
    lines[lo - 1] = " " * indent + "pass  # MUTANT"
    return "\n".join(lines)


def _run(module: Path, text: str, suite_arg: str) -> tuple[int, str]:
    """Run `text` as a sibling of `module` so relative paths behave the same."""
    mut = module.with_name(f".mutation_audit_{os.getpid()}_{module.name}")
    try:
        mut.write_text(text, encoding="utf-8")
        r = subprocess.run([sys.executable, str(mut), suite_arg],
                           capture_output=True, text=True, timeout=900)
        return r.returncode, r.stdout + r.stderr
    finally:
        mut.unlink(missing_ok=True)


def classify(rc: int, out: str, marker: str = ASSERT_MARKER) -> str:
    """Classify by the line that ENDED the run, not by the marker appearing
    anywhere in the output. A traceback ECHOES the offending source line, so a
    crash inside a suite whose source contains the marker prints the marker --
    and matching it anywhere reports that crash as an assertion-kill. Found by
    this module's own crash-kill selftest."""
    if rc == 0:
        return "SURVIVED"
    tail = [l for l in out.splitlines() if l.strip()]
    return ("killed-by-assertion"
            if tail and marker in tail[-1] else "killed-by-crash")


def refusal_bearing(src: str) -> set[str]:
    return {fn.name for fn in ast.walk(ast.parse(src))
            if isinstance(fn, ast.FunctionDef)
            and any(isinstance(n, ast.Raise) for n in ast.walk(fn))}


def audit(module: Path, targets: set[str], suite_arg: str = "--selftest",
          marker: str = ASSERT_MARKER, excluded: set[str] | None = None) -> dict:
    src = module.read_text(encoding="utf-8")

    excluded = excluded or set()
    unscoped = sorted(refusal_bearing(src) - targets - excluded)
    if unscoped:
        raise HarnessRefused(
            f"REFUSED: CONTROL D -- refusal-bearing function(s) {unscoped} are "
            f"neither in --targets nor explicitly excluded. They would be "
            f"audited by NOBODY while this report says 'no survivors'. Name "
            f"them or exclude them on purpose; silence is not a scope.")

    rc, out = _run(module, src, suite_arg)
    if rc != 0:
        raise HarnessRefused(
            f"REFUSED: CONTROL A -- the UNMUTATED copy of {module.name} does "
            f"not pass ({out.strip().splitlines()[-1][:160] if out.strip() else 'no output'}). "
            f"Every mutant would exit non-zero for that same reason and the "
            f"audit would report a PERFECT KILL RATE. No numbers are reported.")

    can = src + CANARY
    c_site = raise_sites(can, {"_mutation_audit_canary"})
    if not c_site:
        raise HarnessRefused("REFUSED: CONTROL B -- canary not found; the "
                             "harness cannot prove it detects a survivor")
    rc, out = _run(module, delete_site(can, c_site[0][1], c_site[0][2]), suite_arg)
    if classify(rc, out, marker) != "SURVIVED":
        raise HarnessRefused(
            f"REFUSED: CONTROL B -- a canary refusal that NO test reaches was "
            f"not reported as a survivor (got {classify(rc, out, marker)}). "
            f"Until the harness shows it CAN report a survivor, a zero from it "
            f"is not a result.")

    rc, out = _run(module, src + "\ndef _control_c(:\n", suite_arg)
    if classify(rc, out, marker) == "killed-by-assertion":
        raise HarnessRefused(
            "REFUSED: CONTROL C -- an UNPARSEABLE mutant was classified as an "
            "assertion-kill, so the marker cannot separate a failing test from "
            "a broken file")

    sites, results = raise_sites(src, targets), []
    for fname, lo, hi in sites:
        rc, out = _run(module, delete_site(src, lo, hi), suite_arg)
        results.append({"function": fname, "line": lo,
                        "verdict": classify(rc, out, marker)})
    tally: dict[str, int] = {}
    for r in results:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    killed = sum(v for k, v in tally.items() if k != "SURVIVED")
    seen = tally.get("killed-by-assertion", 0) > 0
    return {"module": str(module), "targets": sorted(targets),
            # A marker that never matches cannot be CONTROLLED for -- it does
            # not trip A, B or C; it silently turns every assertion-kill into a
            # crash-kill. So it is OBSERVED and reported instead of claimed.
            "marker_observed": seen,
            "marker_unverified": bool(killed) and not seen,
            "controls": {"A_unmutated_passes": True,
                         "B_canary_reported_survivor": True,
                         "C_syntax_error_not_a_kill": True},
            "sites": len(sites), "tally": tally, "results": results,
            "survivors": [r for r in results if r["verdict"] == "SURVIVED"]}


def _selftests() -> int:
    """The harness audited by its own standard: it must FIND a survivor it is
    given, and REFUSE the two inputs that would make it lie."""
    import tempfile
    checks = 0

    def ok(cond, label):
        nonlocal checks
        if not cond:
            raise AssertionError(f"selftest failed: {label}")
        checks += 1

    good = '''
import sys
def check(x):
    if x < 0:
        raise ValueError("negative")
    if x > 100:
        raise ValueError("too big")
    return x
def _s():
    try:
        check(-1); print("selftest failed: no refusal"); return 1
    except ValueError: pass
    print("2 checks passed"); return 0
if __name__ == "__main__":
    sys.exit(_s())
'''
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "m.py"
        m.write_text(good)
        r = audit(m, {"check"})
        ok(r["sites"] == 2, "finds every raise site in the named function")
        ok(len(r["survivors"]) == 1
           and r["survivors"][0]["verdict"] == "SURVIVED",
           "POSITIVE CONTROL: the untested refusal ('too big') is reported as "
           "a SURVIVOR -- the harness demonstrably CAN fire")
        ok(r["tally"].get("killed-by-assertion") == 1,
           "and the tested refusal is an ASSERTION-kill, named by its marker")

        m.write_text('''
import sys
def check(x):
    if x is None:
        raise ValueError("none")
    return x.bit_length()
def _s():
    try:
        check(None); print("selftest failed: no refusal"); return 1
    except ValueError: pass
    print("1 check passed"); return 0
if __name__ == "__main__":
    sys.exit(_s())
''')
        rc = audit(m, {"check"})
        ok(all(x["verdict"] == "killed-by-crash" for x in rc["results"])
           and rc["results"],
           "a refusal whose removal only makes something downstream throw is "
           "classified crash-kill, NOT assertion-kill -- it is a detection, "
           "but it does not prove the refusal is asserted. The traceback here "
       "ECHOES a source line containing the marker, so matching the marker "
       "anywhere in the output reports this crash as an assertion-kill")

        m.write_text(good.replace('print("2 checks passed"); return 0',
                                  'return 3'))
        refused = ""
        try:
            audit(m, {"check"})
        except HarnessRefused as e:
            refused = str(e)
        ok("CONTROL A" in refused and "PERFECT KILL RATE" in refused,
           "KNOWN-BAD: a module whose UNMUTATED suite fails is REFUSED, not "
           "reported as 100% killed -- this is the exact false green that "
           "produced a 53/53 in this program (R-347)")

        two = good.replace('def _s():',
                           'def other():\n    raise ValueError("unscoped")\ndef _s():')
        m.write_text(two)
        d_ref = ""
        try:
            audit(m, {"check"})
        except HarnessRefused as e:
            d_ref = str(e)
        ok("CONTROL D" in d_ref and "other" in d_ref,
           "KNOWN-BAD: a refusal-bearing function left out of --targets "
           "REFUSES. A target list is chosen by hand, so an omitted function "
           "is audited by nobody while the report still reads 'no survivors' "
           "-- the clean sweep that is silent about what it never looked at")
        ok(audit(m, {"check"}, excluded={"other"})["sites"] == 2,
           "and EXPLICITLY excluding it proceeds -- scope may be narrowed on "
           "purpose, never by omission")

        m.write_text(good)
        base = audit(m, {"check"})
        ok(base["marker_observed"] is True
           and base["marker_unverified"] is False,
           "a marker that DOES match is reported as observed")
        blind = audit(m, {"check"}, marker="a string no output contains")
        ok(blind["marker_unverified"] is True
           and blind["tally"].get("killed-by-assertion") is None,
           "KNOWN-BAD: an unmatchable marker turns every assertion-kill into a "
           "crash-kill and trips NO control -- A, B and C all still pass. It "
           "cannot be controlled for, so it is OBSERVED: marker_unverified "
           "says the assertion/crash split is not meaningful in this run")
    print(f"da_mutation_audit selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("module", nargs="?", type=Path)
    ap.add_argument("--targets", default="",
                    help="comma-separated function names to mutate")
    ap.add_argument("--suite-arg", default="--selftest")
    ap.add_argument("--marker", default=ASSERT_MARKER)
    ap.add_argument("--exclude", default="",
                    help="comma-separated refusal-bearing functions "
                         "DELIBERATELY out of scope (control D)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftests()
    if not a.module or not a.targets:
        ap.error("module and --targets are required")
    try:
        r = audit(a.module, set(a.targets.split(",")), a.suite_arg, a.marker,
                  {x for x in a.exclude.split(",") if x})
    except HarnessRefused as e:
        print(e, file=sys.stderr)
        return 2
    print(f"{r['module']}  sites={r['sites']}  {r['tally']}")
    print(f"  controls: A unmutated passes / B canary reported survivor / "
          f"C syntax error not a kill / D every refusal-bearing function in "
          f"scope -- all green")
    if r["marker_unverified"]:
        print(f"  WARNING: marker {a.marker!r} never matched a failure tail; "
              f"the assertion/crash split is NOT meaningful in this run")
    for x in r["results"]:
        if x["verdict"] != "killed-by-assertion":
            print(f"  {x['verdict']:<20} {x['function']} L{x['line']}")
    return 1 if r["survivors"] else 0


if __name__ == "__main__":
    sys.exit(main())
