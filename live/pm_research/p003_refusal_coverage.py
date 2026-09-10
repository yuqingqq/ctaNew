#!/usr/bin/env python3
"""SUPERSEDED by `p003_refusal_exercise_check.py` (DA 200, REV 168's spec).

KEPT AS PROVENANCE, NOT AS AN INSTRUMENT. Its numbers are cited in Q-DA-402
and Q-DA-403 and must remain resolvable, but it had two defects the successor
exists to close and it should not be used to gate anything:

  * IT SAW ONE IDIOM. Inline `REFUSED <NAME>` literals only; blind to
    `NAME = "SOME_REFUSAL"` referenced in a `raise`. MEASURED: its population
    was 59; the successor's is 168, split 86 inline / 82 named-constant.
  * IT SILENTLY EXCLUDED THE UNNAMED ONES on the grounds that they were
    "sentences, not tokens". MEASURED: 1,103 such sites, 106 in
    `de_multiday_gate1_runner.py` alone. A dropped population reads as a
    clean one.

Use `p003_refusal_exercise_check` instead.

EVERY `REFUSED <NAME>` MUST BE NAMED IN A TEST. A STANDING FLOOR.

REV specified it and the coordinator authorised it (DA 198). Rule 15 already
requires that every checker ship a falsifier; REV's sweep measured what its
absence costs -- 34 refusals introduced without one. A snapshot list ages the
moment it is written; a FLOOR does not, which is why this is a check and not
a table.

THE RULE: a `REFUSED <NAME>` token that appears in a raise or a refusal
message must also appear inside at least one TEST FUNCTION somewhere in the
bounded universe. If it does not, nobody has ever driven that refusal and it
is a branch that has never been shown to fire.

WHAT IT DOES NOT CLAIM: that the test is a GOOD one. It establishes that the
name is mentioned in a test body -- a floor, not a proof. Saying so is the
point: a check that overstates its own reach is the defect it exists to catch.

BOUNDING: the universe is enumerated and COUNTED before the scan; unparsable
files are NAMED, never skipped silently; nothing is read off a truncated
display (DA 196 -- an enumeration piped through `head -20` reported thirteen
carriers of a set of eighteen).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOTS = (HERE,)

#: A refusal token: `REFUSED` followed by a SCREAMING_SNAKE name. The 4-char
#: floor keeps `REFUSED DAY`-style prose out; those are sentences, not tokens.
TOKEN = re.compile(r"REFUSED\s+([A-Z][A-Z0-9_]{3,})")

#: A test function by NAME. Deliberately generous: this is a floor, and a
#: narrow definition would under-report coverage and cry wolf.
TEST_NAME = re.compile(r"(^|_)(selftest|test)([_0-9]|$)", re.I)


def is_test_function(name: str) -> bool:
    return bool(TEST_NAME.search(name or ""))


def scan(roots=DEFAULT_ROOTS) -> dict:
    files, unparsable = [], []
    for r in roots:
        r = Path(r)
        files += sorted(p for p in r.rglob("*.py") if p.is_file())
    declared: dict[str, list] = {}
    tested: set[str] = set()
    for p in files:
        try:
            src = p.read_text()
            tree = ast.parse(src)
        except Exception as e:
            unparsable.append((str(p), type(e).__name__))
            continue
        for m in TOKEN.finditer(src):
            declared.setdefault(m.group(1), []).append(p.name)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and is_test_function(node.name):
                seg = ast.get_source_segment(src, node) or ""
                for m in TOKEN.finditer(seg):
                    tested.add(m.group(1))
                # a test may name the token without the REFUSED prefix
                for name in list(declared):
                    if name in seg:
                        tested.add(name)
    # second pass: a token declared in file A may be tested in file B, so the
    # `declared` set must be complete before membership is decided.
    for p in files:
        try:
            src = p.read_text()
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and is_test_function(node.name):
                seg = ast.get_source_segment(src, node) or ""
                for name in declared:
                    if name in seg:
                        tested.add(name)
    untested = sorted(n for n in declared if n not in tested)
    return {"n_files_enumerated": len(files),
            "n_unparsable": len(unparsable), "unparsable": unparsable,
            "n_refusal_tokens": len(declared),
            "n_tested": len(declared) - len(untested),
            "n_untested": len(untested),
            "untested": [{"token": n, "declared_in": sorted(set(declared[n]))}
                         for n in untested],
            "verdict": "FLOOR_HELD" if not untested and not unparsable
                       else "REFUSALS_NEVER_DRIVEN" if untested
                       else "UNPARSABLE_FILES_PRESENT",
            "WHAT_THIS_DOES_NOT_CLAIM": (
                "that each test is a GOOD test. It establishes the token is "
                "NAMED in a test body -- a floor, not a proof.")}


def assert_floor(roots=DEFAULT_ROOTS) -> dict:
    """REFUSES if any refusal token has never been named in a test."""
    r = scan(roots)
    if r["untested"]:
        names = ", ".join(f"{u['token']} ({u['declared_in'][0]})"
                          for u in r["untested"][:8])
        raise AssertionError(
            f"REFUSED REFUSALS_NEVER_DRIVEN: {r['n_untested']} of "
            f"{r['n_refusal_tokens']} refusal tokens appear in NO test "
            f"function: {names}"
            + (" ..." if r["n_untested"] > 8 else ""))
    if r["unparsable"]:
        raise AssertionError(
            f"REFUSED UNPARSABLE_FILES_PRESENT: {r['n_unparsable']} file(s) "
            f"could not be parsed and their refusals are UNKNOWN, not clean: "
            f"{r['unparsable'][:3]}")
    return r


# ------------------------------------------------------------- selftest

_N = {"n": 0, "bad": 0}


def _ok(cond, label):
    _N["n"] += 1
    if not cond:
        _N["bad"] += 1
        print(f"  FAIL {label}")
    return cond


def selftest(quiet: bool = False) -> int:
    import tempfile

    # ---- THE FALSIFIER FIRST, because a check that has never failed is the
    # ---- exact thing this exists to prevent.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # (a) PROPERLY TESTED -> passes
        (tmp / "mod_ok.py").write_text(
            'def go(x):\n'
            '    if not x:\n'
            '        raise ValueError("REFUSED FIXTURE_ALPHA: nope")\n'
            '\n'
            'def selftest():\n'
            '    try:\n'
            '        go(0)\n'
            '    except ValueError as e:\n'
            '        assert "FIXTURE_ALPHA" in str(e)\n')
        r_ok = scan((tmp,))
        _ok(r_ok["n_untested"] == 0 and r_ok["verdict"] == "FLOOR_HELD"
            and r_ok["n_refusal_tokens"] == 1,
            f"POSITIVE CONTROL: a refusal named in its own selftest PASSES "
            f"({r_ok['verdict']}, {r_ok['n_refusal_tokens']} token)")

        # (b) DELIBERATELY UNTESTED -> must FAIL and NAME it
        (tmp / "mod_bad.py").write_text(
            'def go(x):\n'
            '    if not x:\n'
            '        raise ValueError("REFUSED FIXTURE_BETA: never driven")\n')
        r_bad = scan((tmp,))
        _ok(r_bad["n_untested"] == 1
            and r_bad["untested"][0]["token"] == "FIXTURE_BETA"
            and r_bad["verdict"] == "REFUSALS_NEVER_DRIVEN",
            f"KNOWN-BAD: an untested refusal is FOUND and NAMED "
            f"({r_bad['untested']})")
        try:
            assert_floor((tmp,))
            fired = False
            msg = ""
        except AssertionError as e:
            fired = True
            msg = str(e)
        _ok(fired and "FIXTURE_BETA" in msg,
            "KNOWN-BAD: assert_floor RAISES and names the untested token")

        # (c) CROSS-FILE: a token declared in A and tested in B still passes.
        (tmp / "mod_bad.py").unlink()
        (tmp / "mod_decl.py").write_text(
            'def go():\n    raise ValueError("REFUSED FIXTURE_GAMMA: x")\n')
        (tmp / "mod_test.py").write_text(
            'def test_gamma():\n    assert "FIXTURE_GAMMA"\n')
        r_x = scan((tmp,))
        _ok(r_x["n_untested"] == 0,
            "CROSS-FILE: a refusal declared in one module and driven in "
            "another counts as covered -- the check is not per-file")

        # (d) PARTIAL INPUT: an unparsable file is UNKNOWN, never clean.
        (tmp / "mod_broken.py").write_text('def (:\n')
        r_u = scan((tmp,))
        _ok(r_u["n_unparsable"] == 1
            and r_u["verdict"] == "UNPARSABLE_FILES_PRESENT",
            "PARTIAL INPUT: an unparsable file makes the verdict UNKNOWN and "
            "is NAMED -- absence of a parse is never read as coverage")
        try:
            assert_floor((tmp,))
            fired2 = False
        except AssertionError as e:
            fired2 = "UNPARSABLE" in str(e)
        _ok(fired2, "PARTIAL INPUT: assert_floor refuses on an unparsable file")

    # ---- and the real surface, REPORTED not asserted (see __main__).
    real = scan()
    _ok(real["n_files_enumerated"] > 100,
        f"the universe is bounded and COUNTED: "
        f"{real['n_files_enumerated']} .py files, "
        f"{real['n_unparsable']} unparsable (named, not skipped)")
    _ok(real["n_refusal_tokens"] > 50,
        f"the surface carries {real['n_refusal_tokens']} distinct REFUSED "
        f"tokens")

    if not quiet:
        print(f"[p003_refusal_coverage] {_N['n'] - _N['bad']}/{_N['n']} checks, "
              f"{_N['bad']} failures | real surface: "
              f"{real['n_refusal_tokens']} tokens, {real['n_tested']} named in "
              f"a test, {real['n_untested']} NEVER DRIVEN")
    return 1 if _N["bad"] else 0


if __name__ == "__main__":
    import json
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if "--assert" in sys.argv:
        try:
            assert_floor()
            print("FLOOR_HELD")
        except AssertionError as e:
            print(e)
            sys.exit(1)
    else:
        print(json.dumps(scan(), indent=1)[:4000])
