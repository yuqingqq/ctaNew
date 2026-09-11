"""INSTRUMENT: names a function LOADS that its module never BINDS.

Import cannot catch this class -- Python resolves a global at CALL time --
so a module with an unbound name imports clean, passes every cell that
calls its OTHER functions, and raises NameError only when the defective
line finally executes. Twice today that line was in a record-building
path that runs AFTER the expensive work: at REVIEW 175, in
`computing_module_provenance` (a sentinel left in one of three use sites)
and in `_declaration_pin` (a `chain` that was never bound, whose
NameError a blanket `except` reported as "this declaration names no pin").

Usage:  de_unbound_name_sweep.py [module ...]      -> one line per finding
        de_unbound_name_sweep.py --falsify         -> the instrument's own
Exit:   0 clean, 1 findings, 2 the sweep itself failed to run.
"""
from __future__ import annotations
import builtins
import dis
import importlib
import os
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# A STATIC SWEEP MUST RUN WHEN THE TREE IS MID-MOVE -- that is exactly when
# an unbound name gets introduced. The driver refuses AT IMPORT while the
# declaration names another commit, which would make this instrument
# unusable in the only window it is needed. Off for THIS process only; the
# production arm in the cells drives the entry point with the guard ON.
os.environ.setdefault("DE_VALUATION_PREFLIGHT_OFF", "1")

# The valuation's computing closure plus the emit it calls at the end --
# the end is exactly where a late NameError hides.
DEFAULT = ("de_forward_value_day", "de_settlement_control_run",
           "de_multiday_gate1_runner", "de_forward_evaluator",
           "de_revaluation_emit", "de_window_decomposition")


def undefined_globals(modules) -> list:
    """[`mod.func:line name`] for every global load the module never binds."""
    out: list = []
    for mod in modules:
        m = importlib.import_module(mod)
        bound = set(vars(m)) | set(dir(builtins))

        def walk(code, where):
            for ins in dis.get_instructions(code):
                if (ins.opname in ("LOAD_GLOBAL", "LOAD_NAME")
                        and ins.argval not in bound):
                    out.append(f"{mod}.{where}:{ins.positions.lineno} "
                               f"{ins.argval}")
            for c in code.co_consts:
                if isinstance(c, types.CodeType):
                    walk(c, f"{where}.{c.co_name}")

        for name, obj in vars(m).items():
            if isinstance(obj, types.FunctionType) and obj.__module__ == mod:
                walk(obj.__code__, name)
            elif isinstance(obj, type) and obj.__module__ == mod:
                for mn, mo in vars(obj).items():
                    if isinstance(mo, types.FunctionType):
                        walk(mo.__code__, f"{name}.{mn}")
    return out


def falsify() -> int:
    """POSITIVE CONTROL AND KNOWN-BAD (rule 15)."""
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "de_sweep_planted.py").write_text(
            "def clean():\n    x = 1\n    return x\n\n\n"
            "def defective():\n    return never_bound_anywhere\n\n\n"
            "class K:\n    def m(self):\n        return also_not_bound\n")
        (Path(td) / "de_sweep_clean.py").write_text(
            "import json\n\n\ndef f(a):\n"
            "    return json.dumps({'a': a, 'b': len(str(a))})\n")
        sys.path.insert(0, td)
        try:
            planted = undefined_globals(["de_sweep_planted"])
            clean = undefined_globals(["de_sweep_clean"])
        finally:
            sys.path.remove(td)

    ck("FLAGS a planted unbound name in a function",
       any(f.endswith("never_bound_anywhere") for f in planted),
       f"{len(planted)} finding(s)")
    ck("FLAGS one inside a METHOD too (the record build lives in both)",
       any(f.endswith("also_not_bound") for f in planted),
       next((f for f in planted if f.endswith("also_not_bound")), "MISSED"))
    ck("does NOT flag a clean module (no false positive)",
       clean == [], str(clean))
    try:
        undefined_globals(["de_sweep_not_a_module_at_all"])
        refused = False
    except ImportError:
        refused = True
    ck("REFUSES a module it cannot import, rather than reporting clean",
       refused)
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    mods = [a for a in argv if not a.startswith("-")] or list(DEFAULT)
    found = undefined_globals(mods)
    for f in found:
        print(f)
    print(f"{len(found)} unbound name(s) in {len(mods)} module(s)")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
