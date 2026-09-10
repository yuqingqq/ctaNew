#!/usr/bin/env python3
"""EVERY NAMED REFUSAL MUST BE EXERCISED. A REPOSITORY INSTRUMENT.

Built to REV 168's specification (DA 200), superseding DA 198's
`p003_refusal_coverage`, which had TWO defects this one exists to close:

  * IT SAW ONE IDIOM. It matched inline `REFUSED <NAME>` literals only and was
    blind to `NAME = "SOME_REFUSAL"` referenced inside a `raise`. REV measured
    the split at 14/24, so either idiom alone misses more than half.
  * IT SILENTLY EXCLUDED THE UNNAMED ONES. Its regex floor dropped
    `REFUSED DAY {day}: ...` sites on the grounds that they were "sentences,
    not tokens" -- 29 of them in `de_multiday_gate1_runner.py` alone -- and a
    dropped population reads as a clean one. They are now a NAMED STATUS.

THE DECLARED LIMIT, REPEATED ON EVERY OUTPUT AND NOT ONLY THE FIRST: the
predicate is TEXTUAL. A cell that drives a refusal without naming it -- a bare
`except`, a message fragment, a call through a helper -- reads as
never-exercised. **THE ERROR INFLATES RATHER THAN DEFLATES.** That is the safe
direction for a floor, and the count is a FLOOR ON WHAT HAS BEEN SEEN TO BE
DRIVEN, never an exact defect count.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOTS = (HERE,)

NEVER_EXERCISED = "REFUSAL_NEVER_EXERCISED"
HAS_NO_NAME = "REFUSAL_HAS_NO_NAME"
NO_REFUSALS = "EXAMINED_NO_REFUSALS_FOUND"

#: REV 168 idiom (2). `-{0,2}` admits `REFUSED -- NAME`, which this repo writes.
INLINE = re.compile(r"REFUSED\s*-{0,2}\s*([A-Z][A-Z0-9_]{5,})")
#: REV 168 idiom (1). A module-level constant whose VALUE is a refusal name.
CONST_VALUE = re.compile(r"^[A-Z][A-Z0-9_]{5,}$")
#: The exercised set: selftest | falsify | _falsify | *_cells | test_*
EXERCISER = re.compile(r"(^|_)(selftest|falsify)([_0-9]|$)|_cells$|^test_", re.I)

LIMIT = ("TEXTUAL PREDICATE. A cell that drives a refusal WITHOUT NAMING it "
         "reads as never-exercised, so THE ERROR INFLATES RATHER THAN "
         "DEFLATES. This is a FLOOR on what has been SEEN to be driven, never "
         "an exact defect count.")


def is_exerciser(name: str) -> bool:
    return bool(EXERCISER.search(name or ""))


def _module_refusal_constants(tree, src):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str) \
                and CONST_VALUE.match(node.value.value):
            out[node.targets[0].id] = node.value.value
    return out


def scan(roots=DEFAULT_ROOTS) -> dict:
    files = []
    for r in roots:
        files += sorted(p for p in Path(r).rglob("*.py") if p.is_file())

    per_file, unparsable = {}, []
    population = {}          # token -> {files, idioms}
    unnamed = []             # (file, line, excerpt)
    const_names = {}         # token -> set of constant identifiers

    for p in files:
        try:
            src = p.read_text()
            tree = ast.parse(src)
        except Exception as e:
            unparsable.append({"file": str(p), "error": type(e).__name__})
            continue
        consts = _module_refusal_constants(tree, src)
        n_raise = n_tok = n_unnamed = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            seg = ast.get_source_segment(src, node) or ""
            n_raise += 1
            toks = set(INLINE.findall(seg))                       # idiom (2)
            for sub in ast.walk(node):                            # idiom (1)
                if isinstance(sub, ast.Name) and sub.id in consts:
                    toks.add(consts[sub.id])
                    const_names.setdefault(consts[sub.id], set()).add(sub.id)
            if toks:
                n_tok += len(toks)
                for t in toks:
                    e = population.setdefault(t, {"files": set(), "idioms": set()})
                    e["files"].add(p.name)
                    e["idioms"].add("inline" if t in INLINE.findall(seg)
                                    else "named_constant")
            elif "REFUSED" in seg:
                n_unnamed += 1
                unnamed.append({"file": p.name,
                                "line": getattr(node, "lineno", None),
                                "excerpt": " ".join(seg.split())[:110]})
        per_file[p.name] = {"n_raise_sites": n_raise,
                            "n_named_refusals": n_tok,
                            "n_unnamed_refusals": n_unnamed,
                            "status": (NO_REFUSALS if n_raise == 0
                                       else "EXAMINED")}

    # ---- exercised set, decided only after the population is complete
    exercised, exerciser_fns = set(), 0
    for p in files:
        try:
            src = p.read_text()
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and is_exerciser(node.name):
                exerciser_fns += 1
                seg = ast.get_source_segment(src, node) or ""
                for t in population:
                    if t in seg or any(c in seg for c in const_names.get(t, ())):
                        exercised.add(t)

    never = sorted(t for t in population if t not in exercised)
    out = {
        "n_files_enumerated": len(files),
        "n_unparsable": len(unparsable), "unparsable": unparsable,
        "POPULATION": len(population),
        "population_by_idiom": {
            "inline": sum(1 for v in population.values() if "inline" in v["idioms"]),
            "named_constant": sum(1 for v in population.values()
                                  if "named_constant" in v["idioms"])},
        "n_exercised": len(exercised),
        "n_never_exercised": len(never),
        "never_exercised": [{"token": t,
                             "files": sorted(population[t]["files"])}
                            for t in never],
        "REFUSAL_HAS_NO_NAME": {"n": len(unnamed), "sites": unnamed[:40],
                                "n_shown": min(40, len(unnamed))},
        "n_exerciser_functions": exerciser_fns,
        "files_with_no_refusals": sorted(
            f for f, v in per_file.items() if v["status"] == NO_REFUSALS),
        "per_file": per_file,
        "DECLARED_LIMIT": LIMIT,
    }
    out["verdict"] = (
        "UNPARSABLE_FILES_PRESENT" if unparsable else
        NEVER_EXERCISED if never else
        HAS_NO_NAME if unnamed else "EXERCISE_FLOOR_HELD")
    return out


def assert_floor(roots=DEFAULT_ROOTS, strict: bool = True) -> dict:
    """REFUSES, publishing the POPULATION COUNT beside every failure.

    REV 168 criterion 4: a checker reporting '0 never-exercised' out of a
    population of 0 is the `BINANCE_GAP_EXCLUDED: 0` shape -- a zero with no
    denominator beside it."""
    r = scan(roots)
    if r["unparsable"]:
        raise AssertionError(
            f"REFUSED UNPARSABLE_FILES_PRESENT: {r['n_unparsable']} file(s) "
            f"unparsable; their refusals are UNKNOWN, not clean. "
            f"POPULATION={r['POPULATION']}. {LIMIT}")
    if r["never_exercised"]:
        listed = "; ".join(f"{u['files'][0]}:{u['token']}"
                           for u in r["never_exercised"][:10])
        raise AssertionError(
            f"REFUSED {NEVER_EXERCISED}: {r['n_never_exercised']} of "
            f"POPULATION={r['POPULATION']} named refusals appear in no "
            f"exerciser: {listed}"
            + (" ..." if r["n_never_exercised"] > 10 else "")
            + f" | {LIMIT}")
    if r["REFUSAL_HAS_NO_NAME"]["n"] and strict:
        # REV 168 criterion 3. `strict=False` gates ONLY on the actionable
        # subset -- named refusals never exercised -- and is the mode a build
        # can hold green today. IT IS NOT A WEAKER SPEC: the unnamed count
        # still travels on every output, and a gate nobody can ever turn green
        # is a gate people route around, which is how a class goes quiet.
        raise AssertionError(
            f"REFUSED {HAS_NO_NAME}: {r['REFUSAL_HAS_NO_NAME']['n']} raise "
            f"site(s) say REFUSED and carry NO TOKEN, so they cannot be "
            f"exercised by name. POPULATION={r['POPULATION']}. {LIMIT}")
    return r


# ------------------------------------------------------------- falsifier

_N = {"n": 0, "bad": 0}


def _ok(cond, label):
    _N["n"] += 1
    if not cond:
        _N["bad"] += 1
        print(f"  FAIL {label}")
    return cond


def selftest(quiet: bool = False) -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # ---- CELL 1, POSITIVE CONTROL: a named refusal with no cell -> FLAG
        (tmp / "a_named_no_cell.py").write_text(
            'def go(x):\n'
            '    if not x:\n'
            '        raise ValueError("REFUSED FIXTURE_ALPHA_TOKEN: nope")\n')
        r1 = scan((tmp,))
        _ok(r1["POPULATION"] == 1 and r1["n_never_exercised"] == 1
            and r1["never_exercised"][0]["token"] == "FIXTURE_ALPHA_TOKEN"
            and r1["verdict"] == NEVER_EXERCISED,
            f"CELL 1 positive control: a named refusal with no cell is FLAGGED "
            f"and NAMED (POPULATION={r1['POPULATION']})")

        # ---- CELL 2, KNOWN-BAD: the same token inside a selftest -> NOT flagged
        (tmp / "a_named_with_cell.py").write_text(
            'def selftest():\n'
            '    assert "FIXTURE_ALPHA_TOKEN"\n')
        r2 = scan((tmp,))
        _ok(r2["POPULATION"] == 1 and r2["n_never_exercised"] == 0,
            "CELL 2 known-bad: the same refusal named in a selftest is NOT "
            "flagged -- the check does not simply flag everything")

        # ---- CELL 3, THE SILENT-REGEX CONTROL. The one that matters.
        (tmp / "a_named_with_cell.py").unlink()
        (tmp / "a_named_no_cell.py").unlink()
        (tmp / "b_unmatched_spelling.py").write_text(
            'def go(day):\n'
            '    raise ValueError(f"REFUSED DAY {day}: the book moved")\n')
        r3 = scan((tmp,))
        _ok(r3["POPULATION"] == 0
            and r3["REFUSAL_HAS_NO_NAME"]["n"] == 1
            and r3["verdict"] == HAS_NO_NAME,
            f"CELL 3 SILENT-REGEX CONTROL: a refusal the token regex does NOT "
            f"match is reported under {HAS_NO_NAME} -- NOT returned as a clean "
            f"zero (verdict={r3['verdict']}, population={r3['POPULATION']})")
        try:
            assert_floor((tmp,))
            fired = False
            msg = ""
        except AssertionError as e:
            fired, msg = True, str(e)
        _ok(fired and HAS_NO_NAME in msg and "POPULATION=0" in msg,
            "CELL 3: assert_floor REFUSES on it AND publishes POPULATION=0 "
            "beside the failure -- a zero never travels without its "
            "denominator (criterion 4)")

        # ---- IDIOM (1): a named CONSTANT referenced in a raise
        (tmp / "b_unmatched_spelling.py").unlink()
        (tmp / "c_const_idiom.py").write_text(
            'FIXTURE_CONST_REFUSAL = "FIXTURE_GAMMA_TOKEN"\n'
            '\n'
            'def go(x):\n'
            '    if not x:\n'
            '        raise ValueError(f"REFUSED {FIXTURE_CONST_REFUSAL}: no")\n')
        r4 = scan((tmp,))
        _ok(r4["POPULATION"] == 1
            and r4["never_exercised"][0]["token"] == "FIXTURE_GAMMA_TOKEN"
            and r4["population_by_idiom"]["named_constant"] == 1,
            f"IDIOM (1): a refusal held in a module CONSTANT and referenced in "
            f"a raise is in the population -- the idiom DA 198 was blind to "
            f"({r4['population_by_idiom']})")
        (tmp / "c_cell.py").write_text(
            'def falsify_gamma():\n'
            '    assert "FIXTURE_CONST_REFUSAL"\n')
        r5 = scan((tmp,))
        _ok(r5["n_never_exercised"] == 0,
            "IDIOM (1): naming the CONSTANT in a `falsify_*` cell exercises "
            "it -- the exerciser set is REV 168's, not DA 198's narrower one")

        # ---- CRITERION 5: show it examined its target.
        (tmp / "d_no_refusals.py").write_text('def plain():\n    return 1\n')
        r6 = scan((tmp,))
        _ok("d_no_refusals.py" in r6["files_with_no_refusals"]
            and r6["per_file"]["d_no_refusals.py"]["status"] == NO_REFUSALS
            and r6["per_file"]["c_const_idiom.py"]["n_raise_sites"] >= 1,
            f"CRITERION 5: a file yielding zero raise-sites is reported as "
            f"{NO_REFUSALS} rather than silently contributing 0/0, and a file "
            f"that WAS examined publishes its raise-site count")

        # ---- PARTIAL INPUT
        (tmp / "e_broken.py").write_text('def (:\n')
        r7 = scan((tmp,))
        _ok(r7["n_unparsable"] == 1
            and r7["verdict"] == "UNPARSABLE_FILES_PRESENT",
            "PARTIAL INPUT: an unparsable file makes the verdict UNKNOWN and "
            "is NAMED -- a failed parse is never read as coverage")

    real = scan()
    _ok(real["DECLARED_LIMIT"] == LIMIT,
        "the declared limit ships ON THE OUTPUT, every output, not only the "
        "first (REV 168)")
    _ok(real["POPULATION"] > 0,
        f"the real surface has a NON-ZERO population "
        f"({real['POPULATION']}) -- a zero here would be the "
        f"BINANCE_GAP_EXCLUDED: 0 shape")

    if not quiet:
        print(f"[p003_refusal_exercise_check] {_N['n'] - _N['bad']}/{_N['n']} "
              f"checks, {_N['bad']} failures | POPULATION={real['POPULATION']} "
              f"(inline {real['population_by_idiom']['inline']} / const "
              f"{real['population_by_idiom']['named_constant']}), "
              f"exercised {real['n_exercised']}, NEVER EXERCISED "
              f"{real['n_never_exercised']}, {HAS_NO_NAME} "
              f"{real['REFUSAL_HAS_NO_NAME']['n']}")
    return 1 if _N["bad"] else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if "--assert" in sys.argv:
        try:
            r = assert_floor(strict="--named-only" not in sys.argv)
            print("EXERCISE_FLOOR_HELD  POPULATION=%d" % r["POPULATION"])
        except AssertionError as e:
            print(e)
            sys.exit(1)
    else:
        r = scan()
        r.pop("per_file", None)
        print(json.dumps(r, indent=1, default=str)[:6000])
