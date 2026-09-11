"""WHEN A TIGHTENING LANDS, RE-DRIVE WHAT TESTS THE TIGHTENED THING.

The rule (REVIEW 205, DE 372) earned over four instances in three rounds:

  * my gate 4 tightening (canonical population required) broke the
    FIXTURES of gates 5 and 6, which call `build_actions`;
  * DA's gate 4 tightening broke DA's own gate-4 probe;
  * DA's gate 6 probe constructs `ReplayInputs` without the now-required
    `action_keys_sha256`, crashes, and an unprobed row defaulted to
    SATISFIED -- a tightening that made a LEDGER read better than before.

Every one was invisible in the module that changed, because the module
that changed was green. The thing that broke was the thing TESTING it.

A rule nobody runs is prose beside a table, so this is the rule as an
instrument: name a module you tightened, and it finds every file that
imports it, drives each one's `--falsify`, and reports the ones that are
not green -- at a NAMED TREE, because a drive that does not name its tree
proves nothing about any other (DE_PROCEDURE §15).

Usage:  de_dependents_sweep.py <module.py> [--tree DIR]
        de_dependents_sweep.py --falsify
Exit:   0 every dependent green, 1 a dependent is red, 4 no dependents.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = "P003_DE_DEPENDENTS_SWEEP_V1"
RED = "DEPENDENT_FALSIFIER_IS_RED_AFTER_THE_TIGHTENING"
CELL = re.compile(r"^\s*\[(PASS|FAIL)\]")
TOTAL = re.compile(r"^\s*(\d+)/(\d+) cells pass")


def importers(module: str, tree: Path = HERE) -> list:
    """Every file in the tree that imports `module`, by NAME not by guess."""
    stem = Path(module).stem
    pat = re.compile(rf"^\s*(?:import|from)\s+{re.escape(stem)}\b", re.M)
    out = []
    for f in sorted(Path(tree).glob("*.py")):
        if f.name == Path(module).name:
            continue
        try:
            if pat.search(f.read_text()):
                out.append(f.name)
        except (OSError, UnicodeDecodeError):
            continue
    return out


def drive(name: str, tree: Path = HERE) -> dict:
    """Run one dependent's falsifier. NO falsifier is reported, not hidden."""
    f = Path(tree) / name
    r = subprocess.run([sys.executable, str(f), "--falsify"],
                       capture_output=True, text=True, cwd="/tmp",
                       env={"PATH": "/usr/bin:/bin", "HOME": "/home/yuqing",
                            "DE_VALUATION_PREFLIGHT_OFF": "1",
                            # A STALE .pyc DEFEATED THIS INSTRUMENT'S OWN
                            # FIXTURE: `VALUE = 1` and `VALUE = 2` are the
                            # same SIZE, and CPython reuses cached
                            # bytecode when mtime and size match -- so the
                            # "tightened" module still behaved like the
                            # old one and the dependent stayed green. A
                            # sweep that re-drives dependents must not
                            # read yesterday's bytecode.
                            "PYTHONDONTWRITEBYTECODE": "1"})
    lines = r.stdout.splitlines()
    cells = [ln for ln in lines if CELL.match(ln)]
    # THE VERDICT IS THE EXIT CODE, NOT A COUNT I PARSED OUT OF SOMEONE
    # ELSE'S OUTPUT. My first version took the LAST "N/M cells pass" line
    # -- and a module that DRIVES other modules echoes THEIR totals, so I
    # read a nested module's count and reported DA's ledger RED when it
    # exits 0 with no failing cell. Attributing another program's output
    # to the wrong subject is the defect this sweep exists to catch,
    # committed by the sweep.
    m = None
    for ln in reversed(lines[-3:]):
        m = TOTAL.match(ln) or m
    if not cells:
        return {"module": name, "has_falsifier": False, "rc": r.returncode,
                "green": None,
                "note": (r.stdout + r.stderr).strip().splitlines()[-1][:110]
                if (r.stdout + r.stderr).strip() else "no output"}
    ok, total = (int(m.group(1)), int(m.group(2))) if m else (
        sum(1 for c in cells if CELL.match(c).group(1) == "PASS"),
        len(cells))
    failed_cell = any(CELL.match(c).group(1) == "FAIL" for c in cells)
    return {"module": name, "has_falsifier": True, "rc": r.returncode,
            "cells": f"{ok}/{total}",
            "green": r.returncode == 0 and not failed_cell,
            "verdict_from": "exit code, with a failing cell as a "
                            "second veto -- never a parsed count alone",
            "first_red": next((c.strip()[:90] for c in cells
                               if "[FAIL]" in c), None)}


def sweep(module: str, tree: Path = HERE) -> dict:
    deps = importers(module, tree)
    rows = [drive(d, tree) for d in deps]
    red = [r for r in rows if r["green"] is False]
    return {"protocol": PROTOCOL, "tightened": Path(module).name,
            "tree": str(tree), "n_dependents": len(deps),
            "dependents": deps, "rows": rows,
            "red": [r["module"] for r in red],
            "all_green": not red,
            "refusal": (f"REFUSED {RED}: {[r['module'] for r in red]} after "
                        f"tightening {Path(module).name}" if red else None)}


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    deps = importers("de_fair_value_actions.py")
    ck("the sweep FINDS the dependents of a real module, by import not by "
       "guess",
       "de_fair_value_policy_seam.py" in deps
       and "de_fair_value_replay_seam.py" in deps,
       ", ".join(deps))
    ck("  and does not list the module itself",
       "de_fair_value_actions.py" not in deps)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "de_thing.py").write_text("VALUE = 1\n")
        (d / "de_tests_thing.py").write_text(
            "import de_thing\n"
            "import sys\n"
            "def falsify():\n"
            "    ok = de_thing.VALUE == 1\n"
            "    print(f'  [{\"PASS\" if ok else \"FAIL\"}] the value is 1')\n"
            "    print(f'\\n{int(ok)}/1 cells pass')\n"
            "    return 0 if ok else 1\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(falsify() if '--falsify' in sys.argv else 2)\n")
        before = sweep("de_thing.py", d)
        ck("a dependent that is GREEN before a tightening reports green",
           before["all_green"] and before["n_dependents"] == 1,
           before["rows"][0]["cells"])
        # THE TIGHTENING: the module changes, its own tests would still
        # pass, and the DEPENDENT's fixture breaks.
        # DIFFERENT SIZE AS WELL AS DIFFERENT CONTENT, belt and braces
        # beside PYTHONDONTWRITEBYTECODE above.
        (d / "de_thing.py").write_text("VALUE = 222\n")
        after = sweep("de_thing.py", d)
        ck("after the tightening the DEPENDENT is RED, and the sweep names "
           "it",
           not after["all_green"] and after["red"] == ["de_tests_thing.py"]
           and RED in (after["refusal"] or ""),
           after["rows"][0]["first_red"])
        (d / "de_no_falsifier.py").write_text("import de_thing\n")
        none = sweep("de_thing.py", d)
        ck("a dependent with NO falsifier is REPORTED, never counted green",
           any(r["module"] == "de_no_falsifier.py"
               and r["has_falsifier"] is False and r["green"] is None
               for r in none["rows"]),
           "has_falsifier: False, green: None")
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("module", nargs="?")
    ap.add_argument("--tree", default=str(HERE))
    ap.add_argument("--falsify", action="store_true")
    a = ap.parse_args(argv)
    if a.falsify:
        return falsify()
    if not a.module:
        ap.error("name the module you tightened")
    out = sweep(a.module, Path(a.tree))
    print(f"tightened {out['tightened']} in {out['tree']}: "
          f"{out['n_dependents']} dependent(s)")
    for r in out["rows"]:
        mark = ("green" if r["green"] else "RED" if r["green"] is False
                else "NO FALSIFIER")
        print(f"  {r['module']:38s} {mark:12s} "
              f"{r.get('cells') or r.get('note', '')}")
    if out["refusal"]:
        print(out["refusal"])
        return 1
    return 0 if out["n_dependents"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
