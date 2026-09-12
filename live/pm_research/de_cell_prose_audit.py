"""A CELL THAT ASSERTS WORDING IS NOT TESTING THE PROPERTY.

Rule 10 at cell granularity, from three instances in one session -- all
mine, all the same shape, all corrected the same way.

THE MECHANICAL TEST: if every string in the artifact were reworded
without changing its meaning, would this cell still pass? If not, it is
testing prose. Such a cell passes until someone improves a sentence and
then fails for a reason that has nothing to do with correctness -- and,
far worse, it PASSES WHILE THE PROPERTY IS BROKEN so long as the
sentence survives.

WHAT COUNTS AS PROSE, and the line is drawable mechanically:

  A TOKEN is fine              "AMENDMENT_CHOSEN_AFTER_THE_CLOCK_STARTED",
                               "PROJECTED", "OK" -- an UPPER_SNAKE name is
                               an identifier the code also uses, so
                               rewording the prose cannot move it
  A FIELD NAME is fine         "rows", "planning_rate" -- a key is part of
                               the structure, not of the wording
  A SENTENCE IS PROSE          "the world did not change", "not a state
                               change" -- reword it and the cell breaks
                               while the property stands

This audits MY OWN falsifiers, because a failure mode nameable three
times in one night is one I will commit a fourth.

Usage:  de_cell_prose_audit.py --falsify
        de_cell_prose_audit.py --audit [module ...]
"""
from __future__ import annotations

CALL_SITE = {
    "kind": "DELIBERATE_INVOCATION",
    "by": "a person, over their own cells, before landing them",
    "gates": "NOTHING -- it reports; the discipline is the author's",
}

import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = "P003_DE_CELL_PROSE_AUDIT_V1"

TOKEN = re.compile(r"^[A-Z][A-Z0-9_]{3,}$")
IDENTIFIERISH = re.compile(r"^[a-z0-9_.\[\]]+$")
#: A string a cell may safely compare against even though it is lowercase:
#: a path fragment or a file name is structure, not wording.
STRUCTURAL = re.compile(r"^[\w./-]+\.(py|json|jsonl|sh|md|pkl)$")


class ProseAuditRefused(ValueError):
    """The audit cannot be run as declared."""


def _strings(node) -> list:
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            out.append(sub.value)
    return out


def _asserted_strings(cond) -> list:
    """ONLY THE STRINGS THE CELL TESTS AGAINST A VALUE.

    A string PASSED AS AN ARGUMENT is an input, not an assertion -- a
    fixture's source text or a call's parameter is not the cell claiming
    anything about wording. Flagging those would make the audit fire on
    its own fixtures, which is the false-positive shape this lane keeps
    finding.
    """
    out = []
    for sub in ast.walk(cond):
        if isinstance(sub, ast.Compare):
            for side in [sub.left] + list(sub.comparators):
                if (isinstance(side, ast.Constant)
                        and isinstance(side.value, str)):
                    out.append(side.value)
        elif (isinstance(sub, ast.Call)
              and isinstance(sub.func, ast.Attribute)
              and sub.func.attr in ("startswith", "endswith", "count",
                                    "index", "find")):
            for a in sub.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    out.append(a.value)
    return out


def classify(text: str) -> str:
    t = text.strip()
    if not t:
        return "empty"
    if TOKEN.match(t):
        return "token"
    if STRUCTURAL.match(t):
        return "structural"
    if IDENTIFIERISH.match(t) and " " not in t:
        return "field_name"
    if len(t.split()) >= 3:
        return "PROSE"
    return "short_literal"


def audit_module(path: Path) -> dict:
    src = path.read_text()
    tree = ast.parse(src)
    rows = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "ck"):
            continue
        if len(node.args) < 2:
            continue
        cond = node.args[1]
        for text in _asserted_strings(cond):
            kind = classify(text)
            if kind == "PROSE":
                rows.append({"line": node.lineno,
                             "asserted_string": text[:80],
                             "cell": (_strings(node.args[0])[:1]
                                      or [""])[0][:60]})
    return {"module": path.stem, "n_prose_assertions": len(rows),
            "rows": rows}


def audit(modules=None, root: Path = HERE) -> dict:
    paths = ([root / f"{m}.py" for m in modules] if modules
             else sorted(root.glob("de_*.py")))
    out = [audit_module(p) for p in paths if p.is_file()]
    total = sum(r["n_prose_assertions"] for r in out)
    return {"protocol": PROTOCOL, "n_modules": len(out),
            "n_prose_assertions": total,
            "offenders": [r for r in out if r["n_prose_assertions"]],
            "clean": [r["module"] for r in out
                      if not r["n_prose_assertions"]],
            "the_test":
                "if every string in the artifact were reworded without "
                "changing its meaning, would the cell still pass?"}


def falsify() -> int:
    import tempfile
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    print("== the classifier draws the line mechanically ==")
    ck("an UPPER_SNAKE refusal name is a TOKEN, not prose",
       classify("AMENDMENT_CHOSEN_AFTER_THE_CLOCK_STARTED") == "token")
    ck("a field name is structure",
       classify("planning_rate") == "field_name")
    ck("a file name is structure",
       classify("de_band_hazard.py") == "structural")
    ck("a SENTENCE is prose",
       classify("the world did not change") == "PROSE")

    print("== the audit finds a planted prose assertion ==")
    tmp = Path(tempfile.mkdtemp(prefix="de_prose_"))
    (tmp / "de_fixture.py").write_text(
        "def falsify():\n"
        "    ck('good', doc['status'] == 'REFUSED_SOMETHING')\n"
        "    ck('bad', 'the world did not change' in doc['why'])\n"
        "    ck('also good', doc['n'] == 3 and 'rows' in doc)\n"
        "    ck('arg is not an assertion', "
        "helper('a whole sentence passed in') is None)\n")
    got = audit_module(tmp / "de_fixture.py")
    ck("POSITIVE CONTROL: the prose-asserting cell is flagged",
       got["n_prose_assertions"] == 1,
       got["rows"][0]["asserted_string"] if got["rows"] else "MISSED IT")
    ck("  and the token-asserting cell is NOT flagged",
       all("REFUSED_SOMETHING" not in r["asserted_string"]
           for r in got["rows"]))
    ck("  and the field-name cell is not flagged either",
       all("rows" != r["asserted_string"] for r in got["rows"]))
    ck("  and a SENTENCE PASSED AS AN ARGUMENT is an input, not an "
       "assertion",
       all("passed in" not in r["asserted_string"] for r in got["rows"]),
       "otherwise the audit fires on its own fixtures")

    print("== my own falsifiers, audited ==")
    mine = audit(["de_band_decision", "de_band_hazard",
                  "de_fair_value_plumbing_run", "de_unit_verdict",
                  "de_call_site_audit", "de_fair_value_rehearsal"])
    ck("the audit runs over my modules and reports a COUNT, not a "
       "verdict",
       isinstance(mine["n_prose_assertions"], int),
       f"{mine['n_prose_assertions']} prose assertions across "
       f"{mine['n_modules']} modules")
    for r in mine["offenders"]:
        print(f"      {r['module']}: {r['n_prose_assertions']}")
        for row in r["rows"][:3]:
            print(f"        line {row['line']}: {row['asserted_string']}")
    ck("  and modules with none are listed as clean rather than absent",
       isinstance(mine["clean"], list))

    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    if "--audit" in argv:
        mods = [a for a in argv if not a.startswith("--")]
        print(json.dumps(audit(mods or None), indent=2, default=str))
        return 0
    print(json.dumps({"protocol": PROTOCOL}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
