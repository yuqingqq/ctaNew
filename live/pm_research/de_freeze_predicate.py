"""IS ANY PINNED MODULE TUNED AFTER ITS PIN? COMPUTED, NOT DECLARED.

`NO_PARAMETER_OR_MODULE_IS_TUNED_AFTER_THIS_COMMIT` was a typed boolean.
It was true when typed and silently false from 04:47Z, with nothing in the
artifact to say so -- rule 10 exactly: a hardcoded verdict beside a table
has contradicted the table three times in this repo.

IT IS SCOPED PER PIN because the programme now has two: the BUILD pin
(7ed5a90) governs the modules that make a book, and the VALUATION pin
(da00220) governs the modules that read one. A single boolean cannot be
true of both when they are different commits.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TUNED = "A_PINNED_MODULE_WAS_TUNED_AFTER_ITS_PIN"


class FreezePredicateRefused(RuntimeError):
    """A named refusal."""


def _blob(tree: Path, commit: str, name: str):
    r = subprocess.run(["git", "-C", str(tree), "show",
                        f"{commit}:live/pm_research/{name}"],
                       capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def no_module_tuned_after(commit: str, modules, tree=None) -> dict:
    """TRUE iff every named module's bytes in `tree` equal its bytes at
    `commit`. Returns the evidence, never a bare boolean."""
    tree = Path(tree) if tree else HERE.parents[1]
    if not modules:
        raise FreezePredicateRefused(
            f"REFUSED {TUNED}: an empty module set makes the predicate "
            f"vacuously true. A scope with nothing in it is not a pass.")
    per, moved = {}, []
    for name in sorted(modules):
        f = tree / "live" / "pm_research" / name
        now = (hashlib.sha256(f.read_bytes()).hexdigest()
               if f.is_file() else None)
        pinned = _blob(tree, commit, name)
        same = (now is not None and now == pinned)
        if not same:
            moved.append(name)
        per[name] = {"now": (now or "ABSENT")[:16],
                     "at_pin": (pinned or "ABSENT_AT_PIN")[:16],
                     "unchanged": same}
    return {"commit": commit, "tree": str(tree), "n_modules": len(per),
            "modules": per, "moved": moved,
            "NO_MODULE_TUNED_AFTER_THIS_COMMIT": not moved,
            "computed_not_declared": True}


def no_module_tuned_between(pin: str, commit: str, modules,
                            tree=None) -> dict:
    """TRUE iff every named module is byte-identical AT TWO COMMITS.

    This is the form the freeze actually needs: a pin is compared against
    the commit IN FORCE, not against a working tree that may be checked
    out at either. A tree-scoped answer is about one machine's checkout; a
    commit-scoped answer is about the claim the receipt makes.
    """
    tree = Path(tree) if tree else HERE.parents[1]
    if not modules:
        raise FreezePredicateRefused(
            f"REFUSED {TUNED}: an empty module set makes the predicate "
            f"vacuously true. A scope with nothing in it is not a pass.")
    per, moved = {}, []
    for name in sorted(modules):
        a, b = _blob(tree, pin, name), _blob(tree, commit, name)
        same = (a is not None and a == b)
        if not same:
            moved.append(name)
        per[name] = {"at_pin": (a or "ABSENT_AT_PIN")[:16],
                     "at_commit": (b or "ABSENT_AT_COMMIT")[:16],
                     "unchanged": same}
    return {"pin": pin, "commit": commit, "n_modules": len(per),
            "modules": per, "moved": moved,
            "NO_MODULE_TUNED_BETWEEN_THESE_COMMITS": not moved,
            "computed_not_declared": True}


def falsify() -> int:
    """rule 15. The KNOWN-BAD is real: the valuation modules against the
    BUILD pin, which is exactly the pre-amendment state."""
    cells = ok = 0
    def ck(n, c):
        nonlocal cells, ok
        cells += 1; ok += bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")
    VAL = ["de_settlement_control_run.py", "de_forward_evaluator.py",
           "de_settlement_control_aggregate.py"]
    tree = Path("/home/yuqing/ctaNew-wt-de2")
    pre = no_module_tuned_between(
        "7ed5a9015f75de64feeeeaad21d97e4eecc2b15c",
        "da002209cbfc7ece0721ebbf2b80ea6bdb17db18", VAL, tree=tree)
    ck("PRE-AMENDMENT (the REAL known-bad): the valuation modules are NOT "
       "unchanged between 7ed5a90 and da00220 -- FALSE, and it names them",
       pre["NO_MODULE_TUNED_BETWEEN_THESE_COMMITS"] is False
       and sorted(pre["moved"]) == sorted(VAL))
    post = no_module_tuned_between(
        "da002209cbfc7ece0721ebbf2b80ea6bdb17db18",
        "da002209cbfc7ece0721ebbf2b80ea6bdb17db18", VAL, tree=tree)
    ck("POST-AMENDMENT: the same modules against the AMENDED pin -> TRUE",
       post["NO_MODULE_TUNED_BETWEEN_THESE_COMMITS"] is True)
    build = no_module_tuned_between(
        "7ed5a9015f75de64feeeeaad21d97e4eecc2b15c",
        "da002209cbfc7ece0721ebbf2b80ea6bdb17db18",
        ["de_head_scoring.py", "de_phase4_diag_runner.py",
         "be_daybook_build.py", "harmful_stateful_policy.py"], tree=tree)
    ck("BUILD-side modules are UNCHANGED across the amendment, so the "
       "build pin is untouched by it",
       build["NO_MODULE_TUNED_BETWEEN_THESE_COMMITS"] is True)
    post = no_module_tuned_after("7ed5a9015f75de64feeeeaad21d97e4eecc2b15c",
                                 ["de_multiday_gate1_runner.py"], tree=tree)
    ck("BUILD scope against the BUILD pin -> TRUE",
       post["NO_MODULE_TUNED_AFTER_THIS_COMMIT"] is True)
    try:
        no_module_tuned_after("7ed5a90", [], tree=tree)
        ck("an EMPTY scope refuses rather than passing vacuously", False)
    except FreezePredicateRefused as e:
        ck("an EMPTY scope refuses rather than passing vacuously",
           TUNED in str(e))
    miss = no_module_tuned_after("7ed5a9015f75de64feeeeaad21d97e4eecc2b15c",
                                 ["de_forward_value_day.py"], tree=tree)
    ck("a module ABSENT at the pin counts as MOVED, not as unchanged",
       miss["NO_MODULE_TUNED_AFTER_THIS_COMMIT"] is False)
    print(f"\n{ok}/{cells} cells pass")
    return 0 if ok == cells else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print(json.dumps(no_module_tuned_after(argv[0], argv[1:]), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
