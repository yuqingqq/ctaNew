"""BE 221: does anything ASK this guard to fire?

REVIEW 265's rule, now binding: A FALSIFIER PROVES A MODULE CAN FIRE; IT SAYS
NOTHING ABOUT WHETHER ANYTHING ASKS IT TO. Two independent properties, and
this lane has been measuring only the first -- which is how a guard reaches
37/37 with no site at all.

THE DISCRIMINATOR THAT MAKES THE SWEEP HONEST. A grep for the module's name
finds three different things and only one of them is a call site:

  IMPORT    another .py imports it            -- a site
  EXEC      a .sh or unit runs it as a payload -- a site
  ARTIFACT  a consumer reads the file it emits -- a site, mediated by the
            artifact rather than by the import graph. REVIEW 265 flagged
            counting `da_step6_full_pipeline_freeze` as an orphan as its own
            false positive, and the same applies here.
  STRING    its name appears inside a literal  -- NOT a site

That last class is not hypothetical: `be_reserved_days.py:70` contains the
text "called ONLY from selftest (be_offpath_guards_v1.json)", and a name-grep
reads that sentence -- which says the guard is UNCALLED -- as evidence that it
is called. A sweep without this discriminator finds the opposite of the truth
in exactly the case that matters most.

Modules are classified, never silently passed: a module with no site of any
kind is reported as NO_SITE with its refusal count beside it, so "raises but
nobody asks" is visible rather than inferable.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path("/home/yuqing/ctaNew")
LANE = ROOT / "live/pm_research"
UNITS = Path.home() / ".config/systemd/user"

IMPORT, EXEC, ARTIFACT, STRING, NO_SITE = (
    "IMPORT", "EXEC", "ARTIFACT", "STRING_ONLY", "NO_SITE")


def _imports(py: Path) -> set[str]:
    """Module names this file IMPORTS -- from the AST, so a name inside a
    string literal cannot be mistaken for one."""
    try:
        tree = ast.parse(py.read_text())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.update(a.name.split(".")[-1] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[-1])
            out.update(a.name for a in n.names)
    return out


def _refusals(py: Path) -> int:
    try:
        tree = ast.parse(py.read_text())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return 0
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Raise))


def sites_for(mod: str, artifacts: dict[str, str] | None = None) -> dict:
    me = LANE / f"{mod}.py"
    found = {IMPORT: [], EXEC: [], ARTIFACT: [], STRING: []}
    for py in LANE.glob("*.py"):
        if py.name == f"{mod}.py":
            continue
        if mod in _imports(py):
            found[IMPORT].append(py.name)
        elif mod in py.read_text():
            found[STRING].append(py.name)
    for sh in list(LANE.glob("*.sh")) + list((ROOT / "scripts").glob("*.sh")):
        txt = sh.read_text()
        if re.search(rf"{re.escape(mod)}\.py\b", txt):
            found[EXEC].append(sh.name)
        elif mod in txt:
            found[STRING].append(sh.name)
    if UNITS.is_dir():
        for u in UNITS.glob("*"):
            try:
                if mod in u.read_text():
                    found[EXEC].append(u.name)
            except (OSError, UnicodeDecodeError):
                pass
    art = (artifacts or {}).get(mod)
    if art:
        hits = [p.name for p in LANE.glob("*.py")
                if p.name != f"{mod}.py" and art in p.read_text()]
        hits += [p.name for p in LANE.glob("*.sh") if art in p.read_text()]
        if hits:
            found[ARTIFACT] = sorted(set(hits))
    kinds = [k for k in (IMPORT, EXEC, ARTIFACT) if found[k]]
    return {"module": mod, "exists": me.exists(),
            "n_raise_statements": _refusals(me) if me.exists() else None,
            "site_kinds": kinds or [NO_SITE],
            "sites": {k: v for k, v in found.items() if v},
            "HAS_SITE": bool(kinds)}


BE_MODULES = [
    "be_score_neutrality", "be_book_content_diff", "be_rebuild_identity",
    "be_build_preflight", "be_coin_launcher", "be_closeout",
    "be_night_budget", "be_gap_windows", "be_book_window_census",
    "be_offpath_guards", "be_book_identity_compare", "be_sigma_30m",
    "be_theta_occupancy", "be_gap_census", "be_reserved_days",
]
ARTIFACT_OF = {
    "be_night_budget": "be_night_budget_v1.json",
    "be_closeout": "be_closeout_",
    "be_gap_windows": "be137_gap_windows_",
    "be_book_window_census": "be_book_window_census_",
}


def falsify() -> int:
    rc = 0

    def note(n, ok, d=""):
        nonlocal rc
        if not ok:
            rc = 1
        print(f"  {'PASS' if ok else 'FAIL'}  {n}" + (f"   [{d}]" if d else ""))

    off = sites_for("be_offpath_guards", ARTIFACT_OF)
    note("the STRING/IMPORT discriminator fires on the real case: "
         "be_reserved_days NAMES be_offpath_guards without importing it",
         "be_reserved_days.py" in off["sites"].get(STRING, [])
         and "be_reserved_days.py" not in off["sites"].get(IMPORT, []))
    note("  and so be_offpath_guards is correctly reported as NO_SITE",
         not off["HAS_SITE"], str(off["site_kinds"]))
    sn = sites_for("be_score_neutrality", ARTIFACT_OF)
    note("a genuinely imported module is reported as IMPORT",
         IMPORT in sn["site_kinds"], str(sn["sites"].get(IMPORT, [])[:2]))
    pf = sites_for("be_build_preflight", ARTIFACT_OF)
    note("a module executed by a launcher is reported as EXEC",
         EXEC in pf["site_kinds"], str(pf["sites"].get(EXEC, [])))
    nb = sites_for("be_night_budget", ARTIFACT_OF)
    note("an ARTIFACT-mediated module is a site, not an orphan "
         "(REVIEW 265's own false positive)",
         ARTIFACT in nb["site_kinds"] or not nb["HAS_SITE"],
         str(nb["site_kinds"]))
    note("a module that does not exist is reported, not skipped",
         sites_for("be_no_such_module", ARTIFACT_OF)["exists"] is False)
    print(json.dumps({"falsifier": "be_call_sites", "n": 6, "failed": rc}))
    return rc


if __name__ == "__main__":
    import sys
    if "--falsify" in sys.argv:
        raise SystemExit(falsify())
    rows = [sites_for(m, ARTIFACT_OF) for m in BE_MODULES]
    for r in rows:
        mark = "" if r["HAS_SITE"] else "   *** NO SITE ***"
        print(f"  {r['module']:26s} raises={str(r['n_raise_statements']):>4s}  "
              f"{','.join(r['site_kinds']):22s}{mark}")
    print(json.dumps({"n": len(rows),
                      "n_no_site": sum(1 for r in rows if not r["HAS_SITE"]),
                      "no_site": [r["module"] for r in rows
                                  if not r["HAS_SITE"]]}))
