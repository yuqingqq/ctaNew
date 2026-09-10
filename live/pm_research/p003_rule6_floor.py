#!/usr/bin/env python3
"""THE ONE AUTHORITY FOR THE RULE-6 MINIMUM-DRAW FLOOR.

Plan v2 step 1: "bind the duplicated rule-6 floor to one authority."

It was not duplicated. It was in THIRTEEN places under FOUR code spellings
(`MIN_DRAWS`, `MIN_DRAWS_011`, `RULE_6_FLOOR`, `MIN_PERMUTATIONS`) and two
declaration spellings (`min_draws_enforced`, `rule_6_floor`). A search for
any one spelling finds a SUBSET -- SEAT_PROTOCOL rule 32, so the set here is
built by the OPERATION ("carries or enforces the minimum-draw floor") and
enumerated in the declaration so a fourteenth copy is a deliberate addition.

They all read 200 today. THE FAILURE MODE IS SILENT: the day two disagree,
whichever one the code happens to read wins and nothing says so. So this
module does two things and only two:

  FLOOR       -- the value, READ from the declaration, never typed here
  reconcile() -- reads every enumerated carrier and RAISES on divergence

The code carriers are read by AST rather than imported: a constant is a
top-level assignment and parsing it needs no heavy import, and it is
structural rather than a regex over text.

DA owns four of the code carriers and two declarations. BE's are RECONCILED,
not edited -- converting them to derive is BE's act, and a drifting BE copy
is now LOUD instead of silent, which is the point.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DECL = HERE / "declarations" / "p003_rule6_floor_v1.json"

DIVERGED = "RULE6_FLOOR_DIVERGED"
UNREADABLE = "CARRIER_UNREADABLE"


class FloorDiverged(Exception):
    pass


def declaration(path=None) -> dict:
    return json.loads(Path(path or DECL).read_text())


def floor(path=None) -> int:
    """THE number. Read from the authority; never a literal in this file."""
    n = declaration(path).get("THE_NUMBER")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise FloorDiverged(
            f"REFUSED {DIVERGED}: the authority's THE_NUMBER is {n!r}, which "
            f"is not a usable floor. Nothing may derive from it.")
    return n


#: Derived at import so a consumer writes `from p003_rule6_floor import FLOOR`
#: instead of typing 200 for the fourteenth time.
FLOOR = floor()


def _derives_from_authority(path: Path, name: str):
    """Is `name` IMPORTED from this authority rather than typed?

    Structural, by AST: an `from p003_rule6_floor import FLOOR as <name>`
    (or an assignment whose RHS names the authority module) cannot drift,
    because there is no second copy to drift. A DERIVED carrier that has
    quietly become a literal again is the regression this catches."""
    if not path.is_file():
        return False, f"{UNREADABLE}: no such file"
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        return False, f"{UNREADABLE}: unparsable ({e.__class__.__name__})"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "p003_rule6_floor":
            for a in node.names:
                if (a.asname or a.name) == name:
                    return True, "DERIVED"
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    src = ast.dump(node.value)
                    if "p003_rule6_floor" in src or "floor" in src.lower():
                        return True, "DERIVED"
                    return False, (f"{UNREADABLE}: {name} is a LITERAL again "
                                   f"-- it was declared DERIVED")
    return False, f"{UNREADABLE}: {name} neither imported nor assigned"


def _const_from_source(path: Path, name: str):
    """The value of a top-level `name = <literal>` assignment, by AST.

    Returns (value, status). A missing file, an unparsable file or an absent
    constant is a NAMED STATUS -- never silently read as agreement."""
    if not path.is_file():
        return None, f"{UNREADABLE}: no such file"
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        return None, f"{UNREADABLE}: unparsable ({e.__class__.__name__})"
    found = None
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign)
                   else [])
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                val = node.value
                try:
                    found = ast.literal_eval(val)
                except Exception:
                    return None, f"{UNREADABLE}: {name} is not a literal"
    if found is None:
        return None, f"{UNREADABLE}: {name} not assigned at module level"
    return found, "READ"


def _field_from_json(path: Path, field: str):
    """`field` is a DOTTED path. Four of the six declaration carriers live
    nested, and the first reconcile() run reported them CARRIER_UNREADABLE
    rather than agreeing -- the instrument catching its own carrier list."""
    if not path.is_file():
        return None, f"{UNREADABLE}: no such file"
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return None, f"{UNREADABLE}: unparsable json"
    cur = doc
    for part in field.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, f"{UNREADABLE}: field {field!r} absent"
        cur = cur[part]
    return cur, "READ"


def reconcile(decl: dict | None = None, root: Path | None = None,
              raise_on_divergence: bool = True) -> dict:
    """Read EVERY enumerated carrier and refuse if any disagrees.

    A report beside a number is what let thirteen copies accumulate, so this
    RAISES rather than returning a verdict somebody has to notice."""
    d = decl if decl is not None else declaration()
    base = Path(root) if root is not None else REPO
    want = d.get("THE_NUMBER")
    carriers = d.get("carriers") or {}
    rows, bad, unreadable = [], [], []
    for c in carriers.get("code") or []:
        if c.get("binding") == "DERIVED":
            okd, st = _derives_from_authority(base / c["path"], c["name"])
            rows.append({"kind": "code", "path": c["path"], "name": c["name"],
                         "owner": c.get("owner"), "binding": "DERIVED",
                         "value": want if okd else None, "status": st})
            continue
        v, st = _const_from_source(base / c["path"], c["name"])
        rows.append({"kind": "code", "path": c["path"], "name": c["name"],
                     "owner": c.get("owner"), "binding": "LITERAL",
                     "value": v, "status": st})
    for c in carriers.get("declarations") or []:
        v, st = _field_from_json(base / c["path"], c["field"])
        rows.append({"kind": "declaration", "path": c["path"],
                     "name": c["field"], "owner": c.get("owner"),
                     "value": v, "status": st})
    for r in rows:
        if r["status"] not in ("READ", "DERIVED"):
            unreadable.append(r)
        elif r["value"] != want:
            bad.append(r)
    out = {"authority": str(DECL.relative_to(REPO)), "floor": want,
           "n_carriers": len(rows), "n_agreeing": len(rows) - len(bad) - len(unreadable),
           "diverged": bad, "unreadable": unreadable, "carriers": rows,
           "verdict": ("AGREED" if not bad and not unreadable
                       else DIVERGED if bad else UNREADABLE)}
    if raise_on_divergence and (bad or unreadable):
        names = [f"{r['path']}:{r['name']}={r['value']!r}" for r in bad]
        miss = [f"{r['path']}:{r['name']} ({r['status']})" for r in unreadable]
        raise FloorDiverged(
            f"REFUSED {DIVERGED}: the rule-6 floor is {want} in "
            f"{DECL.name} and disagrees elsewhere. DIVERGED: {names or 'none'}. "
            f"UNREADABLE: {miss or 'none'}. A floor that differs between two "
            f"carriers means whichever the code happens to read wins.")
    return out


def _set(doc: dict, dotted: str, value):
    cur = doc
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


# ------------------------------------------------------------- selftest

_N = {"n": 0, "bad": 0}


def _ok(cond, label):
    _N["n"] += 1
    if not cond:
        _N["bad"] += 1
        print(f"  FAIL {label}")
    return cond


def selftest(quiet: bool = False) -> int:
    import shutil
    import tempfile
    d = declaration()

    _ok(FLOOR == d["THE_NUMBER"] and FLOOR == 200,
        f"the module's FLOOR is READ from the authority ({FLOOR}) and is not "
        f"a literal in this file")
    _ok(d.get("THIS_IS_THE_ONE_AUTHORITY") is True,
        "the authority says it is the authority")

    # POSITIVE CONTROL: the real tree reconciles.
    r = reconcile()
    _ok(r["verdict"] == "AGREED" and r["n_carriers"] == d["carriers"]["n_carriers"]
        and r["n_agreeing"] == r["n_carriers"],
        f"POSITIVE CONTROL: all {r['n_carriers']} enumerated carriers agree at "
        f"{r['floor']} ({r['verdict']})")
    _ok(len({c["name"] for c in r["carriers"]}) >= 4,
        f"the carriers use {len({c['name'] for c in r['carriers']})} distinct "
        f"names -- a single-spelling search would have found a SUBSET (rule 32)")

    # KNOWN-BAD: a divergent copy of EACH KIND must REFUSE, and name itself.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for c in (d["carriers"]["code"] + d["carriers"]["declarations"]):
            dst = tmp / c["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / c["path"], dst)
        # (a) a CODE carrier drifts to 199
        code0 = d["carriers"]["code"][0]
        p = tmp / code0["path"]
        p.write_text(p.read_text().replace(f"{code0['name']} = 200",
                                           f"{code0['name']} = 199", 1))
        try:
            reconcile(d, root=tmp)
            fired = False
            msg = ""
        except FloorDiverged as e:
            fired = True
            msg = str(e)
        _ok(fired and code0["path"] in msg and "199" in msg,
            f"KNOWN-BAD (code): {code0['path']}:{code0['name']} drifted to 199 "
            f"-> REFUSES and NAMES the carrier")
        # restore, then (b) a DECLARATION carrier drifts to 201
        shutil.copy2(REPO / code0["path"], p)
        dec0 = d["carriers"]["declarations"][0]
        dp = tmp / dec0["path"]
        doc = json.loads(dp.read_text())
        _set(doc, dec0["field"], 201)
        dp.write_text(json.dumps(doc))
        try:
            reconcile(d, root=tmp)
            fired2 = False
            msg2 = ""
        except FloorDiverged as e:
            fired2 = True
            msg2 = str(e)
        _ok(fired2 and dec0["path"] in msg2 and "201" in msg2,
            f"KNOWN-BAD (declaration): {dec0['path']}:{dec0['field']} drifted "
            f"to 201 -> REFUSES and NAMES the carrier")
        # (c) PARTIAL INPUT: a carrier that cannot be read is NOT agreement
        _set(doc, dec0["field"], 200)
        dp.write_text(json.dumps(doc))
        (tmp / code0["path"]).unlink()
        try:
            reconcile(d, root=tmp)
            fired3 = False
            msg3 = ""
        except FloorDiverged as e:
            fired3 = True
            msg3 = str(e)
        _ok(fired3 and UNREADABLE in msg3,
            "PARTIAL INPUT: a missing carrier is CARRIER_UNREADABLE and "
            "REFUSES -- absence is never read as agreement")
        # (d) and a carrier whose constant stopped being a literal
        shutil.copy2(REPO / code0["path"], tmp / code0["path"])
        p2 = tmp / code0["path"]
        p2.write_text(p2.read_text().replace(f"{code0['name']} = 200",
                                             f"{code0['name']} = _cfg()", 1))
        rep = reconcile(d, root=tmp, raise_on_divergence=False)
        _ok(rep["verdict"] == UNREADABLE and any(
            "not a literal" in x["status"] for x in rep["unreadable"]),
            "PARTIAL INPUT: a constant that is no longer a literal reports "
            "NOT A LITERAL rather than silently passing")

    # KNOWN-BAD: a DERIVED carrier that quietly becomes a literal again.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for c in (d["carriers"]["code"] + d["carriers"]["declarations"]):
            dst = tmp / c["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / c["path"], dst)
        derived = [c for c in d["carriers"]["code"]
                   if c.get("binding") == "DERIVED"]
        _ok(len(derived) >= 4,
            f"{len(derived)} carriers are DERIVED -- they cannot drift because "
            f"there is no second copy")
        dc = derived[0]
        dp2 = tmp / dc["path"]
        txt = dp2.read_text().replace(
            f"from p003_rule6_floor import FLOOR as {dc['name']}",
            f"{dc['name']} = 200", 1)
        dp2.write_text(txt)
        try:
            reconcile(d, root=tmp)
            fired5 = False
            msg5 = ""
        except FloorDiverged as e:
            fired5 = True
            msg5 = str(e)
        _ok(fired5 and "LITERAL again" in msg5,
            f"KNOWN-BAD: {dc['path']}:{dc['name']} reverted from DERIVED to a "
            f"literal 200 -> REFUSES even though the VALUE is still correct. "
            f"The binding is the property, not the number.")

    # A LOWERED AUTHORITY IS ITSELF REFUSED BY ITS CONSUMERS' FLOOR CHECK.
    bad_auth = dict(d)
    bad_auth["THE_NUMBER"] = 0
    try:
        import tempfile as _t
        with _t.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(bad_auth, fh)
            bp = fh.name
        floor(bp)
        fired4 = False
    except FloorDiverged:
        fired4 = True
    _ok(fired4, "KNOWN-BAD: an authority declaring a floor of 0 REFUSES -- "
                "nothing may derive from an unusable floor")

    if not quiet:
        print(f"[p003_rule6_floor] {_N['n'] - _N['bad']}/{_N['n']} checks, "
              f"{_N['bad']} failures | FLOOR={FLOOR} from "
              f"{DECL.name}, {r['n_carriers']} carriers reconciled")
    return 1 if _N["bad"] else 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(json.dumps(reconcile(raise_on_divergence=False), indent=1))
