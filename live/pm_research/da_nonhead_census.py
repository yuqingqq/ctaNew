#!/usr/bin/env python3
r"""DA -- THE NON-HEAD CENSUS (REV 70 section 5).

A declaration is superseded in band as `_v2`, `_v3`, ... and a literal in
the code pins ONE of them by name. Nothing checked that the pinned one was
the CHAIN HEAD, so a module could keep reading v1 while v2 sat beside it --
and three instances of exactly that were found in one day (v1 pinned while
v2 existed; v2 pinned while v3 existed; the runner still pinning v2).

TWO PARTS, both computed:

  (1) THE CHAINS. Every declaration whose `supersedes` names a PRESENT file
      must resolve to EXACTLY ONE head, pair-verified by R-608's rule
      ({path, sha256}, both halves landing on one present file). A family
      with two heads REFUSES: nothing can say which one a pin should name.

  (2) THE LITERALS. An AST census over every string constant in `live/`
      matching `declarations/.*_v\\d+\\.json`. A literal naming a NON-head
      is REFUSED BY NAME, with the head it should have named.

INFRASTRUCTURE, NOT A STATISTIC (R-235): this reads FILENAMES and the
`supersedes` links the seats themselves write. It re-derives no seat's
number, and the pair rule it applies is read from `da_root`, which reads it
from DE's design.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PROTOCOL = "P003_DA_NONHEAD_CENSUS_V1"
DECL_RE = re.compile(r"declarations/([A-Za-z0-9_\-]+?)_v(\d+)\.json$")
LITERAL_RE = re.compile(r"declarations/[A-Za-z0-9_\-]+_v\d+\.json")


def _root() -> Path:
    import da_root as R                                       # noqa: PLC0415
    return R.code_root("the non-head census")


def _family(name: str) -> tuple:
    """('be_r_survey_declaration', 3) from 'be_r_survey_declaration_v3'."""
    m = re.match(r"^(.*)_v(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


def declaration_chains(decl_dir: Path) -> dict:
    """Families, their heads, and the links that make them.

    A head is a member NO OTHER MEMBER supersedes. The link is the R-608
    PAIR: both halves must land on one present file, or it is not a link
    and the family is reported as unlinked rather than chained."""
    files = sorted(p for p in decl_dir.glob("*.json") if p.is_file())
    present = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in files}
    fams: dict = {}
    for p in files:
        fam, ver = _family(p.stem)
        fams.setdefault(fam, []).append((ver, p))
    out = {}
    for fam, members in sorted(fams.items()):
        members.sort(key=lambda t: (t[0] is None, t[0]))
        superseded, links, bad = set(), [], []
        for _, p in members:
            try:
                obj = json.loads(p.read_text())
            except (OSError, ValueError) as e:
                bad.append({"file": p.name, "status": "UNREADABLE",
                            "why": str(e)[:120]})
                continue
            blk = obj.get("supersedes")
            if isinstance(blk, str):
                blk = {"path": blk}
            if not isinstance(blk, dict):
                continue
            path, sha = blk.get("path"), blk.get("sha256")
            if not path or not sha:
                bad.append({"file": p.name,
                            "status": "SUPERSESSION_LINK_INCOMPLETE",
                            "has": [k for k in ("path", "sha256")
                                    if blk.get(k)]})
                continue
            target = Path(path).name
            if target not in present:
                bad.append({"file": p.name, "status": "TARGET_ABSENT",
                            "target": target})
                continue
            if present[target] != sha:
                bad.append({"file": p.name,
                            "status": "TARGET_DIGEST_MISMATCH",
                            "target": target})
                continue
            superseded.add(target)
            links.append({"from": p.name, "to": target})
        heads = [p.name for _, p in members if p.name not in superseded]
        out[fam] = {
            "members": [p.name for _, p in members],
            "n_members": len(members),
            "heads": heads, "n_heads": len(heads),
            "links": links, "unlinked_or_broken": bad,
            "status": ("ONE_HEAD" if len(heads) == 1
                       else "NO_HEAD" if not heads
                       else "MULTIPLE_HEADS_UNLINKED"),
            "why": ("a family with more than one head has no answer to "
                    "'which one should a pin name?' -- the versions exist "
                    "and nothing links them"
                    if len(heads) != 1 else
                    "one head: every other member is superseded by a "
                    "PAIR-VERIFIED link"),
        }
    return out


def literal_census(root: Path, chains: dict) -> dict:
    """Every `declarations/..._vN.json` literal in `live/`, judged."""
    head_of = {}
    for fam, blk in chains.items():
        if blk["n_heads"] == 1:
            head_of[fam] = blk["heads"][0]
    rows, non_heads = [], []
    for py in sorted((root / "live").rglob("*.py")):
        try:
            tree = ast.parse(py.read_text())
        except (OSError, SyntaxError):
            continue
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Constant)
                    and isinstance(n.value, str)):
                continue
            for m in LITERAL_RE.finditer(n.value):
                named = Path(m.group(0)).name
                fam, _ = _family(Path(named).stem)
                head = head_of.get(fam)
                row = {"file": str(py.relative_to(root)), "line": n.lineno,
                       "names": named, "family": fam, "head": head,
                       "is_head": (None if head is None else named == head)}
                rows.append(row)
                if row["is_head"] is False:
                    non_heads.append(row)
    return {"n_literals": len(rows), "literals": rows,
            "n_naming_a_non_head": len(non_heads),
            "naming_a_non_head": non_heads,
            "verdict": ("REFUSED_A_LITERAL_NAMES_A_NON_HEAD" if non_heads
                        else "EVERY_LITERAL_NAMES_ITS_CHAIN_HEAD"),
            "why": ("a pin that names a superseded declaration reads bars "
                    "nobody is running under -- and the superseding version "
                    "is right beside it on disk")}


def build_report(root: Path | None = None) -> dict:
    r = Path(root) if root else _root()
    decl = r / "live/pm_research/declarations"
    chains = declaration_chains(decl)
    lits = literal_census(r, chains)
    multi = {k: v for k, v in chains.items() if v["n_heads"] != 1}
    return {
        "protocol": PROTOCOL,
        "as_of_utc": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(r), "declarations_dir": str(decl),
        "n_families": len(chains),
        "families_without_exactly_one_head": {
            k: {"heads": v["heads"], "status": v["status"]}
            for k, v in multi.items()},
        "n_families_without_exactly_one_head": len(multi),
        "chains": chains,
        "literal_census": lits,
        "infrastructure_not_a_statistic": (
            "this reads FILENAMES and the seats' own `supersedes` links; it "
            "re-derives no seat's number (R-235)"),
        "decides_nothing": ("each literal belongs to the seat that wrote "
                            "it; this census names them"),
    }


def selftest() -> tuple:
    import tempfile
    checks, fails = [], 0

    def ck(label, cond, detail=""):
        nonlocal fails
        checks.append({"check": label, "pass": bool(cond), "detail": detail})
        if not cond:
            fails += 1
        print(("ok   " if cond else "FAIL ") + label)
        if detail:
            print("       " + detail)

    tmp = Path(tempfile.mkdtemp(prefix="da91_"))
    d = tmp / "live" / "pm_research" / "declarations"
    d.mkdir(parents=True)
    v1 = d / "x_declaration_v1.json"
    v1.write_text(json.dumps({"v": 1}))
    v1sha = hashlib.sha256(v1.read_bytes()).hexdigest()
    v2 = d / "x_declaration_v2.json"
    v2.write_text(json.dumps({"v": 2, "supersedes": {"path": v1.name,
                                                     "sha256": v1sha}}))
    v2sha = hashlib.sha256(v2.read_bytes()).hexdigest()
    v3 = d / "x_declaration_v3.json"
    v3.write_text(json.dumps({"v": 3, "supersedes": {"path": v2.name,
                                                     "sha256": v2sha}}))
    mod = tmp / "live" / "pm_research" / "pins_v2.py"
    mod.write_text('PIN = "live/pm_research/declarations/'
                   'x_declaration_v2.json"\n')
    ch = declaration_chains(d)
    lc = literal_census(tmp, ch)
    ck("THE CHAIN RESOLVES TO EXACTLY ONE HEAD, pair-verified: v1 <- v2 <- "
       "v3 leaves v3 as the only member nothing supersedes",
       ch["x_declaration"]["n_heads"] == 1
       and ch["x_declaration"]["heads"] == [v3.name]
       and ch["x_declaration"]["status"] == "ONE_HEAD"
       and len(ch["x_declaration"]["links"]) == 2,
       f"{ch['x_declaration']['n_members']} members -> head "
       f"{ch['x_declaration']['heads']}")
    ck("KNOWN-BAD: A LITERAL NAMING v2 WHILE v3 IS THE HEAD IS REFUSED BY "
       "NAME, with the head it should have named. ***This is the shape "
       "found three times in one day -- a module reading bars nobody is "
       "running under, with the superseding version right beside it on "
       "disk***",
       lc["verdict"] == "REFUSED_A_LITERAL_NAMES_A_NON_HEAD"
       and lc["n_naming_a_non_head"] == 1
       and lc["naming_a_non_head"][0]["names"] == v2.name
       and lc["naming_a_non_head"][0]["head"] == v3.name,
       f"{lc['naming_a_non_head'][0]['file']}:"
       f"{lc['naming_a_non_head'][0]['line']} names {v2.name}, head is "
       f"{v3.name}")
    mod.write_text('PIN = "live/pm_research/declarations/'
                   'x_declaration_v3.json"\n')
    lc2 = literal_census(tmp, declaration_chains(d))
    ck("AND THE SAME MODULE PINNING THE HEAD ADMITS -- the census answers "
       "about the PIN, not about the module",
       lc2["verdict"] == "EVERY_LITERAL_NAMES_ITS_CHAIN_HEAD"
       and lc2["n_literals"] == 1 and lc2["n_naming_a_non_head"] == 0,
       f"{lc2['n_literals']} literal(s), {lc2['n_naming_a_non_head']} "
       f"naming a non-head")
    orphan = d / "x_declaration_v4.json"
    orphan.write_text(json.dumps({"v": 4}))
    ch2 = declaration_chains(d)
    ck("KNOWN-BAD: TWO HEADS IN ONE FAMILY REFUSE -- an unlinked v4 beside "
       "the chained v3 leaves no answer to 'which one should a pin name?', "
       "and a census that picked the highest number would be inventing the "
       "link the seat did not write",
       ch2["x_declaration"]["n_heads"] == 2
       and ch2["x_declaration"]["status"] == "MULTIPLE_HEADS_UNLINKED"
       and set(ch2["x_declaration"]["heads"]) == {v3.name, orphan.name},
       f"heads {sorted(ch2['x_declaration']['heads'])}")
    half = d / "y_declaration_v2.json"
    (d / "y_declaration_v1.json").write_text(json.dumps({"v": 1}))
    half.write_text(json.dumps({"v": 2, "supersedes": {"path":
                                                       "y_declaration_v1.json"}}))
    ch3 = declaration_chains(d)
    ck("AND A HALF-WRITTEN LINK IS NOT A LINK (R-608): a `supersedes` "
       "carrying only a path leaves BOTH versions as heads and is reported "
       "as SUPERSESSION_LINK_INCOMPLETE, never silently followed",
       ch3["y_declaration"]["n_heads"] == 2
       and any(b["status"] == "SUPERSESSION_LINK_INCOMPLETE"
               for b in ch3["y_declaration"]["unlinked_or_broken"]),
       f"y_declaration heads {sorted(ch3['y_declaration']['heads'])}, "
       f"broken {[b['status'] for b in ch3['y_declaration']['unlinked_or_broken']]}")
    print(f"\n{'SELFTEST OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {fails} failure(s)")
    return checks, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        checks, n_fail = selftest()
        if a.output:
            rep = build_report()
            rep["checks"] = checks
            rep["n_checks"] = len(checks)
            rep["n_failed"] = n_fail
            rep["both_directions"] = True
            a.output.write_text(json.dumps(rep, indent=2, sort_keys=True,
                                           default=str) + "\n")
        return 1 if n_fail else 0
    if a.census:
        rep = build_report()
        if a.output:
            a.output.write_text(json.dumps(rep, indent=2, sort_keys=True,
                                           default=str) + "\n")
        print(json.dumps({
            "n_families": rep["n_families"],
            "families_without_exactly_one_head":
                rep["n_families_without_exactly_one_head"],
            "n_literals": rep["literal_census"]["n_literals"],
            "naming_a_non_head":
                rep["literal_census"]["n_naming_a_non_head"],
            "verdict": rep["literal_census"]["verdict"]}, indent=1))
        return 0
    ap.error("--selftest or --census [--output <path>]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
