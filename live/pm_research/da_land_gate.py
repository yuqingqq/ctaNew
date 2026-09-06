#!/usr/bin/env python3
"""DA -- THE LANDING GATE (DA 94 section 5).

***TEST, THEN LAND.*** This seat pushed a FAILING selftest twice (DA 83,
repaired at 02f3227; DA 93, repaired at 97cb93c) for one mechanical
reason: the suite and the `git commit` went into ONE compound command, so
the commit ran whatever the suite said. A shell that reports the last
command's status cannot stop a commit that already ran.

So the step is written down and made EXECUTABLE:

    1. run this gate; READ ITS EXIT CODE
    2. only if it is 0, issue the commit command  (rule 21: `git -C`)
    3. push, and verify the pushed tip against the remote

A KNOWN-RED is not a suppression list. A module already failing at the
tip for a reason that is not this round's is named here WITH its reason,
its raw count stays visible, and a KNOWN-RED that has gone GREEN is
reported as STALE -- otherwise this map becomes the place a real failure
goes to be forgotten.

SCOPE. The default scope is the seat's ACTIVE instruments (seconds). The
full `da_*.py` sweep runs well past 60 s, which is rule 20's heavy
threshold, so `--all` says so and is for a round that holds the lock --
never for a light batch.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = "P003_DA_LANDING_GATE_V1"

#: The instruments this seat maintains and runs every round.
REFUSAL_EXIT = 3

ACTIVE = ("da_root", "da_nonhead_census", "da_gate1_day_verdict",
          "da_cross_venue_forensics", "da_book_verify", "da_accrual_report",
          "da_process_budget_audit", "da_resolver_probe",
          "da_arm_replay_verify", "da_land_gate")

#: module -> (reason, who fixes). PRE-EXISTING at the tip, verified by
#: running the SAME module from the ledger tree before this map was
#: written -- never "it was already broken" from memory.
KNOWN_RED: dict[str, tuple] = {
    "da_cite_audit": (
        "COVERAGE COMPLETE fails on ten identifiers discovered in BE's and "
        "DE's modules that no citation declares (be_forward_day_race_"
        "context.RACE, de_multiday_gate1_runner.REAL_DAY_BUDGET_DERIVATION "
        "and eight more). The instrument is doing its job: their code grew "
        "and the declarations did not follow.",
        "the owning seats declare; DA re-runs"),
    "da_iter011_contract_verify": (
        "18/21 at the tip -- a round-11 contract instrument, red before "
        "this round and not touched by it.",
        "DA, in a round that reopens iter-011"),
}


#: REV 73 S2(c). ***THE STEP IS ONLY REAL IF THE COMMAND HAS THE SHAPE.***
#: "Run the suite, then commit" is a sentence; what stopped it twice was
#: that the two lived in ONE command and the shell ran both. So the shape
#: is a PREDICATE over the command actually issued: the FIRST clause is
#: the gate, and every `git commit` after it is chained with `&&` -- which
#: is what makes the commit conditional on the exit code. `;` runs the
#: commit whatever happened; `||` runs it precisely when the suite FAILED.
GATE_CLAUSE_MARK = "da_land_gate.py --gate"


def command_is_gated(cmd: str) -> dict:
    """Does this landing command chain the commit onto the gate's rc?"""
    parts, buf, sep, seps = [], "", None, []
    i = 0
    while i < len(cmd):
        two = cmd[i:i + 2]
        if two in ("&&", "||"):
            parts.append((sep, buf.strip()))
            seps.append(two)
            sep, buf, i = two, "", i + 2
            continue
        if cmd[i] == ";":
            parts.append((sep, buf.strip()))
            seps.append(";")
            sep, buf, i = ";", "", i + 1
            continue
        buf += cmd[i]
        i += 1
    parts.append((sep, buf.strip()))
    first = parts[0][1] if parts else ""
    commits = [(k, (s, c)) for k, (s, c) in enumerate(parts)
               if re.search(r"\bgit\b[^|;&]*\bcommit\b", c)]
    gate_first = GATE_CLAUSE_MARK in first
    unchained = [c for _, (s, c) in commits if s != "&&"]
    before_gate = [c for k, (_, c) in commits
                   if not gate_first or k == 0]
    return {"n_clauses": len(parts), "first_clause": first[:120],
            "separators": seps,
            "gate_is_the_first_clause": gate_first,
            "n_commit_clauses": len(commits),
            "commit_clauses_not_chained_with_and": unchained,
            "commit_clauses_before_the_gate": before_gate,
            "gated": bool(gate_first and commits and not unchained
                          and not before_gate),
            "why": ("the commit must be CONDITIONAL on the gate's exit "
                    "code: `;` commits whatever happened and `||` commits "
                    "exactly when the suite failed")}


def land_command(*after_the_gate: str) -> str:
    """The landing command, built in the only shape that is checkable."""
    return " && ".join(
        [f"python3 live/pm_research/da_land_gate.py --gate"]
        + [c.strip() for c in after_the_gate if c.strip()])


#: R-661 / DA 106. ***THE HOLD LIVED IN THE COORDINATOR'S SCRIPT AND IN NO
#: SEAT'S CHAIN.*** DE 110's pathspec commit of COORDINATION.md carried my
#: own uncommitted Q-DA-331 row -- verbatim and disclosed, rule 21's third
#: form -- and nothing in MY landing path would have stopped me doing the
#: same to another seat. A register commit is refused unless the working
#: file's difference from HEAD is EXACTLY this seat's own new rows:
#:   * an ADDED line whose row id is not mine  -> FOREIGN_ROW_IN_REGISTER
#:   * any existing line CHANGED or REMOVED    -> REGISTER_EDITED
#: and the same predicate runs again AFTER the commit as a post-condition.
REGISTER_REL = ("orchestrator/PROGRAMS/P-2026-003-polymarket-5min/"
                "workspace/COORDINATION.md")
MY_ROW_PREFIX = "| Q-DA-"


def _register_lines(tree: Path, ref: str | None = None) -> list:
    if ref is None:
        return (tree / REGISTER_REL).read_text().split("\n")
    r = subprocess.run(["git", "-C", str(tree), "show",
                        f"{ref}:{REGISTER_REL}"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"REGISTER_UNREADABLE_AT_{ref}: "
                           f"{(r.stderr or '').strip()[:120]}")
    return r.stdout.split("\n")


def _row_id(line: str) -> str | None:
    if not line.startswith("| Q-"):
        return None
    return line.split("|")[1].strip()


def register_hold(tree: Path = Path("/home/yuqing/ctaNew"),
                  *, mine_prefix: str = MY_ROW_PREFIX,
                  ref: str = "HEAD") -> dict:
    """May this seat commit the register right now?

    THE DIFFERENCE MUST BE EXACTLY THIS SEAT'S OWN NEW ROWS."""
    now = _register_lines(Path(tree))
    was = _register_lines(Path(tree), ref)
    import difflib
    added, removed, changed = [], [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, was, now, autojunk=False).get_opcodes():
        if tag == "insert":
            added += now[j1:j2]
        elif tag == "delete":
            removed += was[i1:i2]
        elif tag == "replace":
            changed += [(was[i1:i2], now[j1:j2])]
    foreign = sorted({_row_id(l) for l in added
                      if l.startswith("| Q-")
                      and not l.startswith(mine_prefix)} - {None})
    mine = sorted({_row_id(l) for l in added
                   if l.startswith(mine_prefix)} - {None})
    other_added = [l for l in added
                   if l.strip() and not l.startswith("| Q-")]
    out = {"register": REGISTER_REL, "compared_against": ref,
           "n_added_lines": len(added), "n_removed_lines": len(removed),
           "n_changed_hunks": len(changed),
           "my_new_rows": mine, "foreign_new_rows": foreign,
           "added_lines_that_are_not_rows": other_added[:5],
           "may_commit": False, "refusal": None,
           "the_rule": ("a register commit may carry EXACTLY this seat's "
                        "own new rows: no foreign row, no edit or removal "
                        "of a landed line, and nothing that is not a row")}
    if foreign:
        out["refusal"] = (
            f"FOREIGN_ROW_IN_REGISTER: the working file adds {foreign}, "
            f"which is not this seat's. Committing it would land another "
            f"seat's row under this seat's commit -- rule 21's third form, "
            f"the shape that carried my own row into DE 110.")
    elif removed or changed:
        out["refusal"] = (
            f"REGISTER_EDITED: {len(removed)} line(s) removed and "
            f"{len(changed)} hunk(s) changed against {ref}. A register "
            f"commit adds rows; it never edits or removes a landed one.")
    elif other_added:
        out["refusal"] = (
            f"REGISTER_NON_ROW_LINES_ADDED: {other_added[:3]}. A row "
            f"landing carries rows.")
    else:
        out["may_commit"] = True
    return out


#: REV 83 S4 / R-717. ***THE CLOSURE IS THE POST-CONDITION ON THE
#: COMMIT'S OWN DIFF, MADE LEGIBLE BY A TRAILER*** -- and it lives in ONE
#: shared script, `scripts/land_register_row.sh`, not in each seat's copy
#: of a rule. This gate keeps its tests-before-landing and MY DA 106 hold
#: as the PRE-check, and then CALLS that script: this module never adds or
#: commits COORDINATION.md itself again.
REGISTER_SCRIPT = Path("/home/yuqing/ctaNew/scripts/land_register_row.sh")
TRAILER_PREFIX = "Landed-By: land_register_row.sh "


def land_register(ids_regex: str, msg_file, *, dry: bool = False,
                  tree: Path = Path("/home/yuqing/ctaNew")) -> dict:
    """PRE-check with this seat's hold, then hand the landing to the
    shared script and read its verdict -- never our own add/commit."""
    if not REGISTER_SCRIPT.is_file():
        return {"status": "REGISTER_SCRIPT_ABSENT",
                "path": str(REGISTER_SCRIPT),
                "why": ("the shared landing script is the closure; this "
                        "gate does not fall back to its own add/commit, "
                        "because a fallback is how one seat's rule "
                        "quietly becomes two")}
    hold = register_hold(tree)
    out = {"pre_check": hold,
           "script": str(REGISTER_SCRIPT),
           "script_sha256": hashlib.sha256(
               REGISTER_SCRIPT.read_bytes()).hexdigest(),
           "ids_regex": ids_regex, "dry": dry}
    if not hold["may_commit"]:
        out["status"] = "REFUSED_BY_MY_OWN_PRE_CHECK"
        out["refusal"] = hold["refusal"]
        return out
    argv = ["bash", str(REGISTER_SCRIPT), ids_regex, str(msg_file)]
    if dry:
        argv.append("--dry")
    r = subprocess.run(argv, capture_output=True, text=True, timeout=300,
                       cwd=str(tree))
    out["returncode"] = r.returncode
    out["stdout"] = (r.stdout or "").strip().splitlines()
    out["stderr"] = (r.stderr or "").strip().splitlines()[-3:]
    #: THE SCRIPT'S REFUSAL NAME SURFACES HERE, verbatim.
    ref = [l for l in out["stdout"] if l.startswith("REFUSED")
           or "FAILED" in l]
    out["script_refusal"] = ref[0] if ref else None
    out["status"] = ("LANDED" if r.returncode == 0 and not dry
                     else "DRY_OK" if r.returncode == 0
                     else "REFUSED_BY_THE_SCRIPT")
    if out["status"] == "LANDED":
        b = subprocess.run(["git", "-C", str(tree), "log", "-1",
                            "--format=%B"], capture_output=True, text=True,
                           timeout=60).stdout
        trailer = [l for l in b.splitlines()
                   if l.startswith(TRAILER_PREFIX)]
        out["trailer"] = trailer[0] if trailer else None
        out["trailer_names_the_script_that_ran"] = bool(
            trailer and out["script_sha256"] in trailer[0])
        if not out["trailer_names_the_script_that_ran"]:
            out["status"] = "LANDED_BUT_THE_TRAILER_DOES_NOT_MATCH"
    return out


def run_module(mod: str, *, timeout_s: int = 180) -> dict:
    """Run one selftest. Script form first; `-m` when the script form
    cannot import `live` (two legacy modules only run that way)."""
    #: A MODULE OR A PATH. A round that touches a module outside
    #: `pm_research` -- DA 96 repinned `mm_research/e2_a_episodes.py` --
    #: must be able to put THAT selftest in front of the commit too, or
    #: the gate is green about files the commit does not contain.
    if mod.endswith(".py") or "/" in mod:
        f = Path(mod) if Path(mod).is_absolute() else (
            HERE.parent.parent / mod)
        mod = f.stem
    else:
        f = HERE / f"{mod}.py"
    if not f.is_file():
        return {"module": mod, "status": "MODULE_ABSENT", "rc": None}
    if "--selftest" not in f.read_text():
        return {"module": mod, "status": "NO_SELFTEST", "rc": None}
    root = HERE.parent.parent
    t0 = time.time()
    r = subprocess.run([sys.executable, str(f), "--selftest"],
                       capture_output=True, text=True, timeout=timeout_s,
                       cwd=str(root))
    form = "script"
    if r.returncode != 0 and "No module named 'live'" in (r.stderr or ""):
        pkg = str(f.parent.relative_to(root)).replace("/", ".")
        r = subprocess.run(
            [sys.executable, "-m", f"{pkg}.{mod}", "--selftest"],
            capture_output=True, text=True, timeout=timeout_s, cwd=str(root))
        form = "-m"
    tail = [x for x in (r.stdout or "").splitlines() if x.strip()]
    return {"module": mod, "rc": r.returncode, "form": form,
            "seconds": round(time.time() - t0, 1),
            "status": "GREEN" if r.returncode == 0 else "RED",
            "last_line": (tail[-1][:200] if tail else
                          (r.stderr or "").strip().splitlines()[-1][:200]
                          if (r.stderr or "").strip() else "")}


def gate(modules=ACTIVE, *, timeout_s: int = 180) -> dict:
    #: EVERY DECLARED KNOWN-RED IS RUN, whatever the scope. A map entry
    #: that is never executed can never be found stale -- and a stale
    #: entry is exactly how a fixed failure becomes a permanent excuse.
    mods = list(dict.fromkeys(list(modules) + list(KNOWN_RED)))
    rows = [run_module(m, timeout_s=timeout_s) for m in mods]
    red = [r for r in rows if r["status"] == "RED"]
    unexpected = [r for r in red if r["module"] not in KNOWN_RED]
    known_hit = {r["module"] for r in red if r["module"] in KNOWN_RED}
    scanned = {r["module"] for r in rows}
    stale = sorted(m for m in KNOWN_RED
                   if m in scanned and m not in known_hit
                   and any(r["module"] == m and r["status"] == "GREEN"
                           for r in rows))
    not_scanned = sorted(m for m in KNOWN_RED if m not in scanned)
    return {
        "protocol": PROTOCOL, "n_modules": len(rows),
        "n_green": sum(1 for r in rows if r["status"] == "GREEN"),
        "n_red": len(red), "n_no_selftest":
            sum(1 for r in rows if r["status"] == "NO_SELFTEST"),
        "rows": rows,
        "unexpected_red": [r["module"] for r in unexpected],
        "known_red_hit": sorted(known_hit),
        "known_red_declared": {k: v[0] for k, v in KNOWN_RED.items()},
        "stale_known_red": stale,
        "known_red_not_scanned": not_scanned,
        "may_land": not unexpected and not stale,
        "the_step": ("run this gate, READ ITS EXIT CODE, and only then "
                     "issue the commit command -- never both in one "
                     "compound command"),
        "a_known_red_is_not_a_suppression": (
            "each carries its reason and its owner, the raw red count "
            "stays visible, and one that has gone GREEN is reported STALE"),
    }


def selftest() -> tuple:
    import tempfile
    checks, fails = [], 0

    def ck(label, cond, detail=""):
        nonlocal fails
        checks.append({"check": label, "pass": bool(cond)})
        if not cond:
            fails += 1
        print(("ok   " if cond else "FAIL ") + label)
        if detail:
            print("       " + detail)

    tmp = Path(tempfile.mkdtemp(prefix="da94gate_"))
    (tmp / "green.py").write_text(
        "import sys\nif '--selftest' in sys.argv:\n"
        "    print('SELFTEST OK'); sys.exit(0)\n")
    (tmp / "red.py").write_text(
        "import sys\nif '--selftest' in sys.argv:\n"
        "    print('SELFTEST FAILED'); sys.exit(1)\n")
    global HERE, KNOWN_RED
    real_here, real_known = HERE, KNOWN_RED
    try:
        HERE = tmp
        KNOWN_RED = {}
        g_ok = gate(("green",))
        g_red = gate(("green", "red"))
        KNOWN_RED = {"red": ("planted", "nobody")}
        g_known = gate(("green", "red"))
        #: the SAME declared entry, once the module passes
        (tmp / "red.py").write_text(
            "import sys\nif '--selftest' in sys.argv:\n"
            "    print('SELFTEST OK'); sys.exit(0)\n")
        g_stale = gate(("green",))
    finally:
        HERE, KNOWN_RED = real_here, real_known
    ck("***TEST, THEN LAND*** (DA 94 section 5, this seat's own words at "
       "DA 83 and DA 93): the gate RUNS the selftests and its exit code is "
       "the thing read before a commit is issued. A green module lands; "
       "***one red module stops the land***, and it is named",
       g_ok["may_land"] is True and g_ok["n_green"] == 1
       and g_red["may_land"] is False
       and g_red["unexpected_red"] == ["red"],
       f"green only -> may_land {g_ok['may_land']}; green+red -> "
       f"may_land {g_red['may_land']}, unexpected {g_red['unexpected_red']}")
    ck("AND A KNOWN-RED IS NOT A SUPPRESSION LIST: a module already "
       "failing at the tip for a reason that is not this round's is named "
       "WITH its reason and does not stop the land -- but a known-red that "
       "has gone GREEN is reported STALE and stops it, so the map cannot "
       "become the place a real failure goes to be forgotten",
       g_known["may_land"] is True and g_known["known_red_hit"] == ["red"]
       and g_known["n_red"] == 1
       and g_stale["may_land"] is False
       and g_stale["stale_known_red"] == ["red"],
       f"declared red -> may_land {g_known['may_land']} with n_red "
       f"{g_known['n_red']} still visible; the same entry when the module "
       f"passes -> stale {g_stale['stale_known_red']}, may_land "
       f"{g_stale['may_land']}")
    ck("AND A MODULE THAT IS NOT THERE, OR HAS NO SELFTEST, IS A NAMED "
       "STATUS -- never a silent pass (rule 11)",
       run_module("no_such_module")["status"] == "MODULE_ABSENT"
       and gate(("no_such_module",))["n_green"] == 0,
       f"absent -> {run_module('no_such_module')['status']}")
    (tmp / "sub").mkdir()
    (tmp / "sub" / "outside.py").write_text(
        "import sys\nif '--selftest' in sys.argv:\n"
        "    print('SELFTEST OK'); sys.exit(0)\n")
    _rh = HERE
    try:
        HERE = tmp
        _out = run_module(str(tmp / "sub" / "outside.py"))
    finally:
        HERE = _rh
    ck("AND THE GATE REACHES A MODULE OUTSIDE ITS OWN DIRECTORY: a round "
       "that touches `mm_research/e2_a_episodes.py` must put THAT selftest "
       "in front of the commit too, or ***the gate is green about files "
       "the commit does not contain***. A path is accepted where a module "
       "name is, and it is labelled by its stem",
       _out["status"] == "GREEN" and _out["module"] == "outside",
       f"a path outside the gate's own directory -> {_out['status']} as "
       f"`{_out['module']}`")

    # -- R-661 / DA 106: THE REGISTER HOLD, DRIVEN ON A FIXTURE TREE ----
    import subprocess as _sp
    reg = Path(tempfile.mkdtemp(prefix="da106reg_"))
    (reg / Path(REGISTER_REL).parent).mkdir(parents=True, exist_ok=True)
    rp = reg / REGISTER_REL
    base = ["| id | seat | note |", "|---|---|---|",
            "| Q-DA-330 | DA | mine, landed |",
            "| Q-BE-286 | BE | theirs, landed |"]
    rp.write_text("\n".join(base) + "\n")
    for c in (["init", "-q"], ["add", "-A"],
              ["-c", "user.email=t@t", "-c", "user.name=t",
               "commit", "-qm", "base"]):
        _sp.run(["git", "-C", str(reg)] + c, capture_output=True, text=True)

    def _write(extra_lines, edit=None):
        lines = list(base)
        if edit is not None:
            lines[edit[0]] = edit[1]
        rp.write_text("\n".join(lines + list(extra_lines)) + "\n")

    _write(["| Q-DA-331 | DA | mine, new |"])
    _ok = register_hold(reg)
    _write(["| Q-DE-110 | DE | THEIRS, uncommitted |"])
    _foreign = register_hold(reg)
    _write(["| Q-DA-331 | DA | mine, new |",
            "| Q-BE-300 | BE | theirs too |"])
    _both = register_hold(reg)
    _write([], edit=(3, "| Q-BE-286 | BE | theirs, EDITED |"))
    _edited = register_hold(reg)
    _write(["not a row at all"])
    _nonrow = register_hold(reg)
    ck("R-661 / DA 106 -- ***THE HOLD LIVED IN THE COORDINATOR'S SCRIPT "
       "AND IN NO SEAT'S CHAIN.*** DE 110's pathspec commit of the "
       "register carried MY OWN uncommitted row -- verbatim and disclosed, "
       "rule 21's third form -- and nothing in MY landing path would have "
       "stopped me doing the same to another seat. A register commit is "
       "refused unless the working file's difference from HEAD is EXACTLY "
       "this seat's own new rows: a foreign ADDED row is "
       "`FOREIGN_ROW_IN_REGISTER` **naming the row id**, and any landed "
       "line CHANGED or REMOVED is `REGISTER_EDITED` -- ***a row landing "
       "adds rows; it never edits one***",
       _ok["may_commit"] is True and _ok["my_new_rows"] == ["Q-DA-331"]
       and _foreign["may_commit"] is False
       and _foreign["foreign_new_rows"] == ["Q-DE-110"]
       and "FOREIGN_ROW_IN_REGISTER" in _foreign["refusal"]
       and "Q-DE-110" in _foreign["refusal"]
       and _both["may_commit"] is False
       and _both["foreign_new_rows"] == ["Q-BE-300"]
       and _edited["may_commit"] is False
       and "REGISTER_EDITED" in _edited["refusal"]
       and _nonrow["may_commit"] is False,
       f"my row only -> may_commit {_ok['may_commit']} ({_ok['my_new_rows']}"
       f"); another seat's row -> "
       f"{_foreign['refusal'].split(':')[0]} naming "
       f"{_foreign['foreign_new_rows']}; mine BESIDE theirs -> "
       f"{_both['refusal'].split(':')[0]} naming "
       f"{_both['foreign_new_rows']}; an edited landed row -> "
       f"{_edited['refusal'].split(':')[0]}; a non-row line -> "
       f"{_nonrow['refusal'].split(':')[0]}")

    # -- REV 83 S4 / R-717: THE REGISTER LANDING IS THE SHARED SCRIPT ---
    ck("REV 83 S4 -- ***THE CLOSURE IS THE POST-CONDITION ON THE COMMIT'S "
       "OWN DIFF, AND IT LIVES IN ONE SHARED SCRIPT.*** This gate keeps "
       "its tests-before-landing and my DA 106 hold as the PRE-check, and "
       "then CALLS `scripts/land_register_row.sh`: ***this module never "
       "adds or commits COORDINATION.md itself again***, and there is no "
       "fallback to its own add/commit -- a fallback is how one seat's "
       "rule quietly becomes two. The script's own refusal name surfaces "
       "in this gate's output, and after a landing the trailer is read "
       "back from `git log -1 --format=%B` and matched against the "
       "DIGEST of the script that actually ran",
       REGISTER_SCRIPT.is_file()
       and "land_register" in globals()
       and TRAILER_PREFIX.startswith("Landed-By: ")
       #: and this module no longer commits the register itself
       #: THE PROPERTY, BY AST: no subprocess argv in this module names
       #: BOTH `commit` and the register. A prose scan for the word
       #: "commit" flagged `may_commit` and the docstring -- an
       #: instrument reading its own explanation instead of its code.
       and not [c for c in ast.walk(ast.parse(
           Path(__file__).read_text()))
           if isinstance(c, ast.Call)
           and any(isinstance(a, ast.List) for a in c.args)
           for lst in [a for a in c.args if isinstance(a, ast.List)]
           if {"commit"} <= {e.value for e in lst.elts
                             if isinstance(e, ast.Constant)
                             and isinstance(e.value, str)}
           and any(isinstance(e, ast.Constant)
                   and isinstance(e.value, str)
                   and "COORDINATION.md" in e.value for e in lst.elts)],
       f"the shared script at {REGISTER_SCRIPT.name} "
       f"({hashlib.sha256(REGISTER_SCRIPT.read_bytes()).hexdigest()[:16]});"
       f" this gate calls it and asserts the trailer afterwards")

    good = land_command("git -C /repo commit -F - -- a.py",
                        "git -C /repo push -q origin HEAD")
    semi = good.replace(" && git -C /repo commit", " ; git -C /repo commit")
    orr = good.replace(" && git -C /repo commit", " || git -C /repo commit")
    flip = ("git -C /repo commit -F - -- a.py && "
            "python3 live/pm_research/da_land_gate.py --gate")
    ck("REV 73 S2(c) -- ***TEST-THEN-LAND IN THE CHECKABLE FORM.*** The "
       "rule is not a sentence about intent: it is a SHAPE the landing "
       "command either has or does not. The FIRST clause is the gate and "
       "every `git commit` is chained onto it with `&&`, so the commit is "
       "conditional on the exit code. Driven on the three ways to get it "
       "wrong: ***`;` commits whatever happened*** (which is how a failing "
       "selftest was pushed twice), ***`||` commits precisely when the "
       "suite FAILED***, and a commit BEFORE the gate is not gated at all",
       command_is_gated(good)["gated"] is True
       and command_is_gated(semi)["gated"] is False
       and command_is_gated(orr)["gated"] is False
       and command_is_gated(flip)["gated"] is False
       and command_is_gated(
           "python3 live/pm_research/da_land_gate.py --gate")["gated"]
       is False,
       f"`&&` -> gated; `;` -> {command_is_gated(semi)['gated']}; `||` -> "
       f"{command_is_gated(orr)['gated']}; commit first -> "
       f"{command_is_gated(flip)['gated']}; gate with no commit -> not a "
       f"landing command")

    print(f"\n{'SELFTEST OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {fails} failure(s)")
    return checks, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="every da_*.py -- runs past 60 s, which is rule "
                         "20's heavy threshold: for a round holding the "
                         "lock, not for a light batch")
    ap.add_argument("--land-register", metavar="IDS_REGEX", default=None,
                    help="hand the register landing to the shared script "
                         "after this seat's own pre-check")
    ap.add_argument("--msg", type=Path, default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--register", action="store_true",
                    help="check the register hold: may this seat commit "
                         "COORDINATION.md right now?")
    ap.add_argument("--register-post", action="store_true",
                    help="the POST-CONDITION: after the commit, the "
                         "register must differ from its parent by this "
                         "seat's rows only")
    ap.add_argument("--also", action="append", default=[],
                    help="an extra module or path to gate on -- the "
                         "modules THIS round touches, wherever they live")
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        checks, n_fail = selftest()
        if a.output:
            a.output.write_text(json.dumps(
                {"protocol": PROTOCOL, "checks": checks,
                 "n_failed": n_fail}, indent=2) + "\n")
        return 1 if n_fail else 0
    if a.land_register:
        if not a.msg:
            ap.error("--land-register needs --msg <commit message file>")
        res = land_register(a.land_register, a.msg, dry=a.dry)
        print(json.dumps({k: v for k, v in res.items()
                          if k != "pre_check"}, indent=2, sort_keys=True))
        print(f"PRE-CHECK: {'PASS' if res['pre_check']['may_commit'] else res['pre_check']['refusal']}")
        for line in res.get("stdout", []):
            print(f"  script: {line}")
        if res.get("trailer"):
            print(f"  trailer: {res['trailer']}")
        return 0 if res["status"] in ("LANDED", "DRY_OK") else REFUSAL_EXIT
    if a.register or a.register_post:
        h = register_hold(ref="HEAD~1" if a.register_post else "HEAD")
        print(json.dumps(h, indent=2, sort_keys=True))
        if h["may_commit"]:
            print("REGISTER HOLD: PASS -- "
                  f"{h['my_new_rows'] or 'no new row'}, 0 foreign")
            return 0
        print(f"REGISTER HOLD REFUSED: {h['refusal']}")
        return REFUSAL_EXIT
    if a.gate or a.all:
        mods = (tuple(sorted(p.stem for p in HERE.glob("da_*.py")))
                if a.all else ACTIVE) + tuple(a.also)
        if a.all:
            print("NOTE: the full sweep runs past 60 s -- rule 20 heavy.")
        g = gate(mods)
        for r in g["rows"]:
            print(f"{r['status']:>12}  {r['module']:<34} "
                  f"rc={r['rc']}  {r.get('last_line', '')[:80]}")
        print(f"\n{g['n_green']} green / {g['n_red']} red "
              f"({len(g['known_red_hit'])} declared) / "
              f"{g['n_no_selftest']} without a selftest")
        if g["unexpected_red"]:
            print(f"LAND REFUSED -- unexpected red: {g['unexpected_red']}")
        if g["stale_known_red"]:
            print(f"LAND REFUSED -- STALE known-red (now green): "
                  f"{g['stale_known_red']}")
        if a.output:
            a.output.write_text(json.dumps(g, indent=2, sort_keys=True) + "\n")
        return 0 if g["may_land"] else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
