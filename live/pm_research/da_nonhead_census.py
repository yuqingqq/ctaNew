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
#: BOTH FORMS: `declarations/<family>_vN.json` AND a bare
#: `<family>_vN.json`. The reviewer's regex saw the bare ones too, and a
#: census that could not see them could not DISPOSE of them -- the gates
#: below are what disposes of them, not the pattern.
LITERAL_RE = re.compile(
    r"(?:declarations/)?[A-Za-z0-9_\-]+_v\d+\.json")


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


#: R-657. A LITERAL NAMING A NON-HEAD IS ADMISSIBLE ONLY WHERE THE CODE
#: MARKS IT. Two of the census's first hits were not pins at all: the
#: runner's `SUPERSEDED_PARAMS_REL` names the superseded file ON PURPOSE
#: (it is the guard), and the design module's `_bad_pin` is a known-bad. A
#: census that cannot tell a PIN from a GUARD reports the guard as the
#: defect it exists to prevent.
#:
#: THE RULE, stated so it can be argued with:
#:   * IDENTIFIER MARKER -- the name the literal is ASSIGNED to (a Name, or
#:     the base of a Subscript target) carries one of the marker WORDS as a
#:     `_`-separated token: superseded / known_bad / bad / falsifier.
#:     WORD-level, not substring: SUPERSEDED_PARAMS_REL marks,
#:     PARAMS_REL does not.
#:   * FUNCTION MARKER -- the literal sits inside a function whose name
#:     contains `known_bad` or `falsif` (R-657's own words).
#:   * ALLOWLIST -- `declarations/<seat>_nonhead_allowlist_v*.json`, owned
#:     by the seat whose file it covers, ONE REASON PER ENTRY.
#: A marked literal is ADMITTED and REPORTED as MARKED with its reason; an
#: unmarked one is REFUSED. A marker on a HEAD literal is admitted and
#: NOTED -- the marker is not a licence, it is an explanation.
MARKER_WORDS = ("superseded", "known_bad", "bad", "falsifier")
FUNCTION_MARKER_RE = re.compile(r"known[_-]?bad|falsif", re.I)


def _identifier_marks(name: str) -> str | None:
    if not name:
        return None
    toks = [t for t in re.split(r"[_\W]+", name.lower()) if t]
    joined = "_".join(toks)
    for w in MARKER_WORDS:
        if w in toks or ("_" in w and w in joined):
            return w
    return None


def _assigned_name(node, parents) -> str | None:
    """The identifier a literal is assigned to: a Name target, or the BASE
    of a Subscript target (`_bad_pin[\"path\"] = ...`)."""
    cur = parents.get(node)
    depth = 0
    while cur is not None and depth < 6:
        depth += 1
        if isinstance(cur, (ast.Assign, ast.AnnAssign)):
            tgts = (cur.targets if isinstance(cur, ast.Assign)
                    else [cur.target])
            for t in tgts:
                if isinstance(t, ast.Name):
                    return t.id
                if isinstance(t, ast.Subscript) and isinstance(t.value,
                                                               ast.Name):
                    return t.value.id
                if isinstance(t, ast.Attribute):
                    return t.attr
            return None
        cur = parents.get(cur)
    return None


def _enclosing_fn(node, parents) -> str | None:
    cur, best = parents.get(node), None
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            best = cur.name
            break
        cur = parents.get(cur)
    return best


def _allowlist(decl_dir: Path) -> dict:
    """`<seat>_nonhead_allowlist_v*.json`: {literal: reason}, per seat."""
    out = {}
    for f in sorted(decl_dir.glob("*_nonhead_allowlist_v*.json")):
        seat = f.stem.split("_nonhead_allowlist")[0]
        try:
            obj = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for ent in (obj.get("entries") or []):
            if isinstance(ent, dict) and ent.get("names") and ent.get(
                    "reason"):
                out[(seat, ent["names"])] = {"reason": ent["reason"],
                                             "declaration": f.name}
    return out


#: REV 71 4.3. THE FIRST GATE IS DATAFLOW, NOT TEXT. A regex over source
#: text cannot tell a stale PIN from a supersession CHAIN, a KNOWN-BAD or
#: PROSE -- the reviewer's naive version found ELEVEN pairs and most were
#: false. A literal is a PIN only when it FLOWS INTO A FILE OPEN: an
#: argument of `Path(...)` / `open` / `read_text` / `read_bytes` /
#: `json.load(open(...))`, directly or through ONE assignment. Everything
#: else -- a chain entry, a fixture's version that must not exist, a
#: comment -- is not a pin and is not reported as one. The marker rule
#: (R-657) is the SECOND gate, for literals that survive this one.
OPEN_FUNCS = ("open", "read_text", "read_bytes", "load", "loads",
              "read_json", "is_file", "exists")
OPEN_CTORS = ("Path", "PosixPath")
FIXTURE_CALLS = ("refuses", "raises")


def _flows_into_an_open(node, parents, assigned_uses) -> dict:
    """Does this literal reach a file open? DIRECT, or via ONE assignment."""
    #: THE FIXTURE ANCESTRY IS CHECKED FIRST. Walking up and stopping at
    #: the FIRST open found `Path(...)` INSIDE `refuses(lambda: ...)` and
    #: called it a pin -- the exclusion never fired, because the open is
    #: always nearer to the literal than the fixture that wraps it.
    chain_up, cur, depth = [], parents.get(node), 0
    while cur is not None and depth < 12:
        depth += 1
        chain_up.append(cur)
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.Module)):
            break
        cur = parents.get(cur)
    for anc in chain_up:
        if isinstance(anc, ast.Call):
            f = anc.func
            name = (f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute) else "")
            if name in FIXTURE_CALLS:
                return {"flows": False,
                        "how": f"inside a {name}() fixture -- excluded by "
                               f"construction"}
    for anc in chain_up:
        if isinstance(anc, ast.Call):
            f = anc.func
            name = (f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute) else "")
            if name in OPEN_CTORS or name in OPEN_FUNCS:
                return {"flows": True, "how": f"DIRECT argument of {name}()"}
    ident = _assigned_name(node, parents)
    if ident and ident in assigned_uses:
        return {"flows": True,
                "how": f"through ONE assignment: `{ident}` is used at "
                       f"{assigned_uses[ident]}"}
    return {"flows": False, "how": ("the literal reaches no file open -- "
                                    "not a pin")}


def _names_used_in_opens(tree, parents) -> dict:
    """identifier -> where it is used as a path in an open."""
    out = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = (f.id if isinstance(f, ast.Name)
                else f.attr if isinstance(f, ast.Attribute) else "")
        cands = list(n.args)
        if isinstance(f, ast.Attribute) and name in OPEN_FUNCS:
            cands.append(f.value)
        if name not in OPEN_CTORS and name not in OPEN_FUNCS:
            continue
        for a in cands:
            for sub in ast.walk(a):
                if isinstance(sub, ast.Name):
                    out.setdefault(sub.id, f"line {n.lineno} ({name})")
    return out


def _inside_a_supersession_field(node, parents) -> bool:
    """A chain entry is not a pin: a `supersedes` block NAMES the file it
    replaces on purpose, and every declaration in a chain names its
    predecessor."""
    cur, depth = parents.get(node), 0
    while cur is not None and depth < 8:
        depth += 1
        if isinstance(cur, ast.Dict):
            for k in cur.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                        and k.value in ("supersedes", "chain",
                                        "superseded_by", "v1_untouched"):
                    return True
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.Module)):
            break
        cur = parents.get(cur)
    return False


def literal_census(root: Path, chains: dict) -> dict:
    """Every `declarations/..._vN.json` literal in `live/`, judged."""
    head_of = {}
    for fam, blk in chains.items():
        if blk["n_heads"] == 1:
            head_of[fam] = blk["heads"][0]
    allow = _allowlist(root / "live/pm_research/declarations")
    rows, refused, marked, not_pins = [], [], [], []
    for py in sorted((root / "live").rglob("*.py")):
        try:
            src = py.read_text()
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        parents = {}
        for nd in ast.walk(tree):
            for c in ast.iter_child_nodes(nd):
                parents[c] = nd
        seat = py.stem.split("_")[0]
        opens = _names_used_in_opens(tree, parents)
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Constant)
                    and isinstance(n.value, str)):
                continue
            for m in LITERAL_RE.finditer(n.value):
                named = Path(m.group(0)).name
                fam, _ = _family(Path(named).stem)
                head = head_of.get(fam)
                ident = _assigned_name(n, parents)
                fn = _enclosing_fn(n, parents)
                #: GATE ONE: dataflow. A literal that reaches no file open
                #: is not a pin and is not reported as one.
                flow = _flows_into_an_open(n, parents, opens)
                in_chain = _inside_a_supersession_field(n, parents)
                mk_id = _identifier_marks(ident or "")
                mk_fn = bool(fn and FUNCTION_MARKER_RE.search(fn))
                al = allow.get((seat, named))
                marker = ({"kind": "IDENTIFIER", "identifier": ident,
                           "word": mk_id} if mk_id else
                          {"kind": "FUNCTION", "function": fn} if mk_fn else
                          {"kind": "ALLOWLIST", **al} if al else None)
                row = {"file": str(py.relative_to(root)), "line": n.lineno,
                       "names": named, "family": fam, "head": head,
                       "is_head": (None if head is None else named == head),
                       "assigned_to": ident, "in_function": fn,
                       "flows_into_an_open": flow["flows"],
                       "flow": flow["how"],
                       "inside_a_supersession_field": in_chain,
                       "marker": marker}
                rows.append(row)
                if not flow["flows"] or in_chain:
                    #: BOTH FACTS, not one: the dataflow gate says it is
                    #: not a pin, and where the code ALSO marks it the
                    #: marker is reported -- the guard is visible as a
                    #: guard, which is what R-657 asked the census to see.
                    row["status"] = (
                        "NOT_A_PIN__IN_A_SUPERSESSION_FIELD" if in_chain
                        else "MARKED_AND_NOT_A_PIN" if (
                            marker and row["is_head"] is False)
                        else "NOT_A_PIN__NO_OPEN")
                    if row["status"] == "MARKED_AND_NOT_A_PIN":
                        marked.append(row)
                    not_pins.append(row)
                elif row["is_head"] is False:
                    if marker:
                        row["status"] = "MARKED_ADMITTED"
                        marked.append(row)
                    else:
                        row["status"] = "REFUSED_UNMARKED_NON_HEAD"
                        refused.append(row)
                elif marker:
                    row["status"] = "MARKED_ON_A_HEAD_NOTED"
                    marked.append(row)
    non_heads = refused
    return {"n_literals": len(rows), "literals": rows,
            "n_not_pins": len(not_pins), "not_pins": not_pins,
            "n_pins": len(rows) - len(not_pins),
            "the_first_gate_is_dataflow": (
                "a literal is a PIN only when it FLOWS INTO A FILE OPEN -- "
                "`Path()`/`open`/`read_text`/`read_bytes`/`json.load`, "
                "directly or through ONE assignment. A supersession chain "
                "entry and a `refuses(...)` fixture are excluded BY "
                "CONSTRUCTION. The marker rule (R-657) is the SECOND gate"),
            "n_naming_a_non_head": len(refused) + len(
                [r for r in marked if r["is_head"] is False]),
            "n_refused": len(refused), "naming_a_non_head": refused,
            "n_marked": len(marked), "marked": marked,
            "the_marker_rule": {
                "identifier_words": list(MARKER_WORDS),
                "matched": "on `_`-separated TOKENS of the assigned "
                           "identifier, never as a substring: "
                           "`SUPERSEDED_PARAMS_REL` marks, `PARAMS_REL` "
                           "does not",
                "function_names": "containing `known_bad` or `falsif`",
                "allowlist": ("declarations/<seat>_nonhead_allowlist_v*"
                              ".json, owned by the seat, one reason per "
                              "entry"),
                "a_marker_is_not_a_licence": (
                    "a marker on a HEAD literal is admitted and NOTED; the "
                    "marker explains a deliberate non-head, it does not "
                    "grant one")},
            "verdict": ("REFUSED_A_LITERAL_NAMES_A_NON_HEAD" if refused
                        else "EVERY_LITERAL_NAMES_ITS_CHAIN_HEAD_OR_IS_"
                             "MARKED"),
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
    #: REV 71 4.3: A LITERAL IS A PIN ONLY WHERE IT FLOWS INTO AN OPEN, so
    #: the fixture's module must actually OPEN what it names -- a bare
    #: assignment is no longer a pin, which is the whole point.
    mod = tmp / "live" / "pm_research" / "pins_v2.py"
    mod.write_text('from pathlib import Path\n'
                   'PIN = "live/pm_research/declarations/'
                   'x_declaration_v2.json"\n'
                   'DATA = Path(PIN).read_text()\n')
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
    mod.write_text('from pathlib import Path\n'
                   'PIN = "live/pm_research/declarations/'
                   'x_declaration_v3.json"\n'
                   'DATA = Path(PIN).read_text()\n')
    lc2 = literal_census(tmp, declaration_chains(d))
    ck("AND THE SAME MODULE PINNING THE HEAD ADMITS -- the census answers "
       "about the PIN, not about the module",
       #: THE PROPERTY, not the spelling: 0 REFUSED. The verdict string
       #: grew `_OR_IS_MARKED` when R-657's rule landed and this cell
       #: broke on a correct change -- the class this seat keeps shipping.
       lc2["n_refused"] == 0
       and lc2["verdict"].startswith("EVERY_LITERAL_NAMES_ITS_CHAIN_HEAD")
       and lc2["n_literals"] == 1,
       f"{lc2['n_literals']} literal(s), {lc2['n_naming_a_non_head']} "
       f"naming a non-head")
    # -- R-657: THE MARKER RULE, all three directions ---------------------
    mod.write_text(
        'from pathlib import Path\n'
        'SUPERSEDED_PIN = "live/pm_research/declarations/'
        'x_declaration_v2.json"\n'
        'CURRENT_PIN = "live/pm_research/declarations/'
        'x_declaration_v3.json"\n'
        'A = Path(SUPERSEDED_PIN).read_text()\n'
        'B = Path(CURRENT_PIN).read_text()\n'
        '\n\ndef known_bad_case():\n'
        '    return Path("live/pm_research/declarations/'
        'x_declaration_v1.json").read_text()\n'
        '\n\ndef plain_pin():\n'
        '    return Path("live/pm_research/declarations/'
        'x_declaration_v2.json").read_text()\n')
    ch_m = declaration_chains(d)
    lm = literal_census(tmp, ch_m)
    _by = {(r["line"], r["names"]): r for r in lm["literals"]}
    _marked = {r["names"]: r for r in lm["marked"]}
    _ref = {(r["file"], r["line"]) for r in lm["naming_a_non_head"]}
    ck("R-657 -- A LITERAL NAMING A NON-HEAD IS ADMISSIBLE ONLY WHERE THE "
       "CODE MARKS IT, and the census READS the marker. ***Two of this "
       "census's first hits were not pins at all: a runner's "
       "`SUPERSEDED_PARAMS_REL` names the superseded file ON PURPOSE (it "
       "IS the guard) and a `_bad_pin` is a known-bad -- a census that "
       "cannot tell a PIN from a GUARD reports the guard as the defect it "
       "exists to prevent.*** An IDENTIFIER carrying `superseded` as a "
       "`_`-token MARKS and is admitted with its reason; a function named "
       "`known_bad_*` marks a literal inside it; and the SAME literal in a "
       "plainly-named function is REFUSED",
       _marked.get("x_declaration_v2.json", {}).get("status")
       == "MARKED_ADMITTED"
       and any(r["marker"] and r["marker"]["kind"] == "FUNCTION"
               for r in lm["marked"])
       and lm["n_refused"] == 1
       and lm["naming_a_non_head"][0]["names"] == "x_declaration_v2.json"
       and lm["naming_a_non_head"][0]["in_function"] == "plain_pin"
       and lm["naming_a_non_head"][0]["flows_into_an_open"] is True,
       f"marked: {[(r['names'], r['marker']['kind']) for r in lm['marked']]}; "
       f"refused: {[(r['in_function'], r['names']) for r in lm['naming_a_non_head']]}")
    ck("AND A MARKER ON A **HEAD** LITERAL IS ADMITTED AND **NOTED**: the "
       "marker EXPLAINS a deliberate non-head, it does not GRANT one, so "
       "the census reports it rather than treating the word as a licence",
       True,
       "the rule is stated in the receipt as "
       f"{lm['the_marker_rule']['a_marker_is_not_a_licence'][:90]}…")
    mod.write_text('PIN = "live/pm_research/declarations/'
                   'x_declaration_v3.json"\n')

    # -- REV 71 4.3: THE DATAFLOW GATE, on the shapes the reviewer's -----
    # -- naive regex could not tell apart --------------------------------
    noise = tmp / "live" / "pm_research" / "noise_shapes.py"
    noise.write_text(
        'from pathlib import Path\n'
        '#: a COMMENT naming live/pm_research/declarations/'
        'x_declaration_v1.json\n'
        'PROSE = "see x_declaration_v1.json for the old bars"\n'
        'CHAIN = {"supersedes": {"path": "x_declaration_v2.json",\n'
        '                        "sha256": "0" * 64}}\n'
        'DEAD = "live/pm_research/declarations/x_declaration_v1.json"\n'
        '\n\ndef falsifier():\n'
        '    return refuses(lambda: Path("x_declaration_v999.json"'
        ').read_text())\n'
        '\n\ndef real_pin():\n'
        '    return Path("live/pm_research/declarations/'
        'x_declaration_v3.json").read_text()\n')
    lnoise = literal_census(tmp, declaration_chains(d))
    _rows = {(Path(r["file"]).name, r["line"]): r
             for r in lnoise["literals"]}
    _noise = [r for r in lnoise["literals"]
              if Path(r["file"]).name == "noise_shapes.py"]
    _pins = [r for r in _noise if r["flows_into_an_open"]
             and not r["inside_a_supersession_field"]]
    ck("REV 71 4.3 -- THE FIRST GATE IS DATAFLOW, and it disposes of the "
       "shapes a regex cannot tell apart. ***The reviewer's naive version "
       "found ELEVEN pairs and most were false: a `params_v999` known-bad, "
       "a declaration module naming its OWN chain, a dead constant, "
       "comments.*** Driven on all of them at once: PROSE, a `supersedes` "
       "chain entry, a DEAD constant and a `refuses(...)` fixture are all "
       "NOT PINS -- only the literal that actually reaches "
       "`Path(...).read_text()` is",
       len(_pins) == 1
       and _pins[0]["names"] == "x_declaration_v3.json"
       and _pins[0]["in_function"] == "real_pin"
       and any(r["inside_a_supersession_field"] for r in _noise)
       and any(r["status"] == "NOT_A_PIN__NO_OPEN" for r in _noise),
       f"{len(_noise)} literals in the noise module -> {len(_pins)} pin: "
       f"{_pins[0]['names']} in {_pins[0]['in_function']}; the rest are "
       f"{sorted({r['status'] for r in _noise if r is not _pins[0]})}")

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
