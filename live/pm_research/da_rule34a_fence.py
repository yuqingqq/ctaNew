"""RULE 34a AS A PREDICATE -- and an audit of who opens the gate it fences.

DA 307, built under runbook §7l.

THE RULE (USER, `abd4b07`): WRITING DOES NOT CONSUME A DAY, READING DOES.
`data/pm_5min/tier2/**/day=2026-09-08/` and any later day are off limits: do
not read, summarise, aggregate or quote them. Their existence is not
permission.

IS THE RULE CHECKABLE AT ALL? The scoping question was asked before building,
because an honest "unenforceable as written" beats a predicate that tests
something adjacent. IT IS CHECKABLE IN ONE HALF AND NOT THE OTHER:

  CHECKABLE   the artifact CLASS -- a path under `data/pm_5min/tier2/` whose
              `day=` partition is >= the floor. That is a pure function of a
              path string and a date, and it is what this module evaluates.

  NOT CHECKABLE FROM CODE  the ACT of "summarise, aggregate or quote". A human
              reading a number off a terminal leaves no artifact, and nothing
              in this repository can detect it. THIS FENCE DOES NOT CLAIM TO.

So the predicate covers the mechanical half -- a program opening a protected
path -- and the human half remains a discipline. Saying so is the point: a
fence that implied it covered the reading eye would be worse than none.

§7l.1 -- INPUTS BOUND TO ARTIFACTS, NOT ARGUMENTS. The floor is PARSED from
the procedure documents that state the rule (`DE_PROCEDURE.md`,
`REV_PROCEDURE.md`) -- documents this seat does not author. Ask who supplies
the input: not the constrained party, and not the caller. A caller supplies
only the path being tested, which is the thing under examination rather than
the standard it is examined against.

§7l.2 -- ORDERING. The protected-class test is FIRST and there is no cheap
term above it. A caller cannot short-circuit past it with a convenience check.

AND IT HAS A CALL SITE (`audit_unguarded_readers`), which is the thing fifteen
other guards in this lane do not have. The audit is called from
`da_scheduled_units_and_eth_inputs.build()`, so the gap ANNOUNCES ITSELF in a
landed declaration rather than waiting for someone to go and look.

Usage:  da_rule34a_fence.py [--falsify]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROTOCOL = "P003_DA_RULE34A_FENCE_V1"
PROTECTED_READ = "RULE_34A_PROTECTED_DAY_READ"
FLOOR_UNPARSEABLE = "RULE_34A_FLOOR_NOT_PARSEABLE_FROM_THE_PROCEDURES"

#: The documents that STATE the rule. This seat authors neither.
RULE_SOURCES = (
    "orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace/DE_PROCEDURE.md",
    "orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace/REV_PROCEDURE.md",
)
PROTECTED_ROOT = "data/pm_5min/tier2"
_DAY_IN_PATH = re.compile(r"day=(\d{4}-\d{2}-\d{2})")
_FLOOR_IN_DOC = re.compile(r"day=(\d{4}-\d{2}-\d{2})")


class Rule34aRefused(ValueError):
    """A protected day's artifact was about to be read."""


def _repo() -> Path:
    here = Path(__file__).resolve().parent
    for base in (here, Path.cwd()):
        r = subprocess.run(["git", "-C", str(base), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    raise RuntimeError("REFUSED NO_REPOSITORY_RESOLVABLE")


def protected_floor(sources=RULE_SOURCES) -> dict:
    """THE FLOOR, PARSED FROM THE DOCUMENTS THAT STATE THE RULE (§7l.1).

    Not a literal in this file -- a literal here would be a copy of the rule
    rather than a reading of it, which is exactly what §7l.1 forbids. If the
    sources disagree, or none states a floor, this REFUSES rather than picking
    one: two floors is not a floor.
    """
    repo = _repo()
    found = {}
    for rel in sources:
        p = repo / rel
        if not p.is_file():
            continue
        days = set(_FLOOR_IN_DOC.findall(p.read_text(errors="replace")))
        if days:
            found[rel] = sorted(days)
    floors = {d for v in found.values() for d in v}
    if len(floors) != 1:
        raise Rule34aRefused(
            f"REFUSED {FLOOR_UNPARSEABLE}: the procedures state "
            f"{sorted(floors) or 'no'} protected-day floor(s) {found}. Two "
            f"floors is not a floor, and none is not a rule.")
    return {"floor": floors.pop(), "stated_in": sorted(found),
            "authored_by": "DE and REV, not this seat"}


def is_protected(path: str, floor: str) -> dict:
    """THE STRONGEST CLAUSE, AND IT IS FIRST (§7l.2).

    Protected iff the path is under the tier2 root AND carries a `day=`
    partition at or after the floor. Both halves are computed from the path
    string; nothing is asserted by a caller.
    """
    norm = str(path).replace("\\\\", "/")
    under_root = PROTECTED_ROOT in norm
    m = _DAY_IN_PATH.search(norm)
    day = m.group(1) if m else None
    at_or_after = bool(day) and day >= floor
    return {"protected": bool(under_root and at_or_after),
            "under_protected_root": under_root, "day_in_path": day,
            "floor": floor, "at_or_after_floor": at_or_after}


def assert_not_protected(path: str) -> dict:
    """REFUSE a read of a protected day's artifact. The fence proper."""
    f = protected_floor()["floor"]
    v = is_protected(path, f)
    if v["protected"]:
        raise Rule34aRefused(
            f"REFUSED {PROTECTED_READ}: {path} is a tier2 artifact for "
            f"day {v['day_in_path']}, at or after the rule 34a floor {f}. "
            f"Writing does not consume a day; READING does. Its existence is "
            f"not permission.")
    return v


#: Lane modules that touch the protected root at all.
_OPENS = re.compile(r"(open\(|read_text|read_bytes|pq\.read|read_table|"
                    r"parquet|json\.load|glob)")


def audit_unguarded_readers(ref: str = "origin/de-freeze-chain-v2") -> dict:
    """WHO OPENS THE GATE THIS FENCE STANDS BESIDE?

    THE POINT OF THIS FUNCTION IS THAT THE GAP ANNOUNCES ITSELF. Fifteen
    guards in this lane were found to have no call site, and every one was
    found by a person going to look. This is called from a landed
    declaration's `build()`, so the absence is reported without anyone
    looking.
    """
    repo = _repo()
    # THE PATTERN IS THE COMPONENT, NOT THE LITERAL PATH. My first version
    # grepped for "data/pm_5min/tier2" and found ONE module -- my own, which
    # only mentions it in prose -- while five modules actually touch tier2,
    # because `evaluation_pipeline` builds the path as
    # `DEFAULT_OUTPUT_ROOT.parent / "tier2"`. A literal-string query returns a
    # clean absence for the wrong reason, which is the defect this whole audit
    # exists to catch, reproduced inside the audit.
    r = subprocess.run(["git", "-C", str(repo), "grep", "-lE", r"tier2",
                        ref, "--", "live/pm_research/*.py"],
                       capture_output=True, text=True)
    files = [l.split(":", 1)[1] for l in r.stdout.splitlines() if ":" in l]
    rows = []
    for rel in sorted(set(files)):
        src = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{rel}"],
                             capture_output=True, text=True).stdout
        touches = ("tier2" in src)
        opens = bool(_OPENS.search(src)) and touches
        guarded = ("assert_not_protected" in src) or ("Rule34aRefused" in src)
        rows.append({"module": rel.split("/")[-1], "touches_tier2": touches,
                     "has_open_like_call": opens, "calls_the_fence": guarded})
    unguarded = [x["module"] for x in rows
                 if x["has_open_like_call"] and not x["calls_the_fence"]]
    return {"ref": ref, "n_modules_touching_tier2": len(rows), "modules": rows,
            "unguarded_readers": unguarded, "n_unguarded": len(unguarded),
            "reading": ("a module that opens files and touches the protected "
                        "root without calling the fence is an open gate. This "
                        "list is the fence's own report on its coverage."),
            "THE_FENCE_DOES_NOT_CLAIM_TO_COVER": (
                "a human reading a number off a terminal. That leaves no "
                "artifact and no code can see it.")}


def build() -> dict:
    try:
        fl = protected_floor()
        err = None
    except Rule34aRefused as e:
        fl, err = None, str(e)
    return {"protocol": PROTOCOL,
            "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rule": ("rule 34a (USER, abd4b07): writing does not consume a "
                     "day, READING does"),
            "floor": fl, "floor_error": err,
            "protected_root": PROTECTED_ROOT,
            "checkable_half": ("a program opening a path under the protected "
                               "root with a day= partition at or after the "
                               "floor"),
            "uncheckable_half": ("the human act of summarising, aggregating or "
                                 "quoting -- no artifact, no detection, and "
                                 "this fence does not pretend otherwise"),
            "audit": audit_unguarded_readers(),
            "rule_10": "the floor is parsed from the procedures at run time"}


def falsify() -> int:
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    fl = protected_floor()
    ck("§7l.1 the floor is PARSED from documents this seat does not author",
       fl["floor"] == "2026-09-08" and len(fl["stated_in"]) >= 1,
       f"{fl['floor']} from {[Path(x).name for x in fl['stated_in']]}")
    # ---- BOTH DIRECTIONS. A fence that only refuses is as broken as one
    # ---- that only admits, and tonight produced one of each.
    prot = "data/pm_5min/tier2/calib_panel/day=2026-09-09/coin=btc/part-0.parquet"
    ck("REFUSES the protected class",
       _refuses(prot), "day=2026-09-09 under tier2")
    ck("...and at the floor exactly, not only after it",
       _refuses("data/pm_5min/tier2/markout_events/day=2026-09-08/coin=eth/p.parquet"))
    ck("ADMITS a day BEFORE the floor",
       not _refuses("data/pm_5min/tier2/calib_panel/day=2026-09-07/coin=btc/p.parquet"),
       "day=2026-09-07 passes")
    ck("ADMITS a non-tier2 path on a protected day",
       not _refuses("data/pm_5min/raw/20260909/btc-updown-5m-1.jsonl.gz"),
       "the raw archive is not what rule 34a names")
    ck("ADMITS tier1 on a protected day",
       not _refuses("data/pm_5min/tier1/coverage/day=2026-09-09/coin=btc/p.parquet"))
    ck("a path with NO day= partition is not protected by accident",
       not _refuses("data/pm_5min/tier2/_manifest.json"))
    ck("§7l.2 the protected-class test is computed from the PATH, not asserted",
       is_protected(prot, fl["floor"])["day_in_path"] == "2026-09-09"
       and is_protected(prot, fl["floor"])["under_protected_root"] is True)
    # ---- the call site, which is the thing fifteen guards lack -----------
    a = audit_unguarded_readers()
    ck("POSITIVE CONTROL: the audit finds the modules that DO touch tier2",
       a["n_modules_touching_tier2"] >= 4,
       f"{a['n_modules_touching_tier2']} modules -- a literal-path query found "
       f"only 1, which is why the pattern is the COMPONENT")
    ck("THE FENCE HAS A CALL SITE: the audit runs and reports coverage",
       isinstance(a["unguarded_readers"], list) and a["n_modules_touching_tier2"] > 0,
       f"{a['n_modules_touching_tier2']} modules touch tier2, "
       f"{a['n_unguarded']} unguarded")
    ck("...and it NAMES the open gates rather than reporting a bare count",
       all(isinstance(x, str) for x in a["unguarded_readers"]),
       str(a["unguarded_readers"][:4]))
    ck("the UNCHECKABLE half is declared, not quietly implied covered",
       "does not pretend otherwise" in build()["uncheckable_half"])
    print(f"\n  {'RULE-34A FENCE CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


def _refuses(path: str) -> bool:
    try:
        assert_not_protected(path)
        return False
    except Rule34aRefused:
        return True


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        sys.exit(1 if falsify() else 0)
    print(json.dumps(build(), indent=1))
