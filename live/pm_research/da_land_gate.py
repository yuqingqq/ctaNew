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
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = "P003_DA_LANDING_GATE_V1"

#: The instruments this seat maintains and runs every round.
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


def run_module(mod: str, *, timeout_s: int = 180) -> dict:
    """Run one selftest. Script form first; `-m` when the script form
    cannot import `live` (two legacy modules only run that way)."""
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
        r = subprocess.run(
            [sys.executable, "-m", f"live.pm_research.{mod}", "--selftest"],
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
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        checks, n_fail = selftest()
        if a.output:
            a.output.write_text(json.dumps(
                {"protocol": PROTOCOL, "checks": checks,
                 "n_failed": n_fail}, indent=2) + "\n")
        return 1 if n_fail else 0
    if a.gate or a.all:
        mods = (tuple(sorted(p.stem for p in HERE.glob("da_*.py")))
                if a.all else ACTIVE)
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
