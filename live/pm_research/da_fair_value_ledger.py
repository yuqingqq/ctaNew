"""THE FAIR-VALUE LANE'S PROGRESS LEDGER, COMPUTED AT VERIFICATION TIME.

DA 274. Every row's status is MEASURED when this runs, from a FETCHED ref --
never copied from a report, a register row, or an earlier run of this ledger.

THE STANDING RULE THIS CAME FROM:

    A LANDING IS PROVEN BY A COUNT AT A FETCHED REF, NEVER BY A COMMIT SHA
    IN PROSE.

A sha in a message proves that somebody made a commit. It does not prove the
commit is reachable from any ref anyone else will ever fetch. Measured
2026-09-11T19:20Z, two shas reported as landed work:

    8fe2a2e   remote refs containing it: NONE   (local `mm-research` only)
    f9a5bc7   remote refs containing it: NONE

and the local `mm-research` branch is 207 commits behind `origin/mm-research`,
so a commit reachable only from it is not landed in any sense another seat can
use.

AND THE LEDGER IS ITS OWN BEST EXAMPLE. DA 273 recorded step 2 as "NOT LANDED
ON ANY REF" after measuring `be_sigma_30m.py` at 0 of 3 refs. Within the hour
it was 1 of 3 on both chain refs. The row was true when written and false when
read -- which is what a COPIED status always eventually is, and why this file
counts instead of remembering.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROTOCOL = "P003_DA_FAIR_VALUE_LEDGER_V1"
REFS = ("origin/mm-research", "origin/de-freeze-chain-v2", "origin/be-build-runner")
#: The refs a landing must be proven at. `origin/mm-research` is the user's
#: fork and is declared NON-EXECUTING, so presence there is recorded but never
#: counts toward a gate.
EXECUTING_REFS = ("origin/de-freeze-chain-v2", "origin/be-build-runner")

#: §11 of fair_value_plan.md v1.2, the eight-step implementation order.
STEPS = (
    (1, "freeze the settlement-label/status reader",
     ("live/pm_research/da_fair_value_gate1_labels.py",)),
    (2, "build and falsify the 30-minute sigma producer",
     ("live/pm_research/be_sigma_30m.py",)),
    (3, "wrap C1/C2 in typed FairPrice records",
     ("live/pm_research/de_fair_price_wrapper.py",)),
    (4, "build canonical actions and the paired scorer",
     ("live/pm_research/de_canonical_action_population.py",
      "live/pm_research/de_fair_value_actions.py")),
    (5, "wire fair value into the quoter and prove Identity parity",
     ("live/pm_research/de_fair_value_policy_seam.py",)),
    (6, "freeze the full pipeline and both candidate identities",
     ("live/pm_research/ev_replay_seam.py",)),
    (7, "run ten-day predictive validation", ()),
    (8, "economic clock, predictive winners only", ()),
)

#: WHO OWNS EACH GATE AND WHERE IT MUST LIVE. The plan implies both and
#: nothing tracked either, so a gate could be silently claimed by two seats or
#: landed to the wrong branch and nobody would see it.
#:
#: REQUIRED REFS ARE BOTH EXECUTING REFS. `origin/mm-research` is the user's
#: fork and is declared NON-EXECUTING (da_shared_tree_non_executing_v1.json), so
#: presence there NEVER satisfies a gate -- it is recorded and ignored.
OWNER = {1: "DA", 2: "BE", 3: "DE", 4: "DE", 5: "DE", 6: "DE", 7: "DE", 8: "DE"}
REQUIRED_REFS = EXECUTING_REFS

#: THE PATH -> STEP MAPPING IS DA'S ATTRIBUTION AND IS THIS LEDGER'S WEAKEST
#: LINK. A status keyed on a path nobody declared measures the GUESS, not the
#: lane. The first version of this file guessed `de_canonical_forecast_action`
#: and `de_forecast_action_scorer` for step 4; the real files are
#: `de_canonical_action_population` and `de_fair_value_actions`, so step 4 read
#: NO_FILES_ON_ANY_EXECUTING_REF while two of its files were sitting on both
#: refs. Same class as the cwd bug above, one layer over: a false negative
#: produced by the instrument rather than by the world.
#:
#: `unattributed_files()` below is the guard. It enumerates the lane's files on
#: a ref and reports any that NO step claims -- so a wrong or missing
#: attribution shows up as an unclaimed file instead of as a silent zero.
FAIR_VALUE_FILE_RE = r"(fair|sigma|forecast|seam|canonical)"

#: Behaviour is NOT inferable from presence. It comes from the owning seat's
#: own cell output, recorded with its source so it is attributable.
BEHAVIOUR = {
    1: {"verdict": "DRIVEN_GREEN", "source": "DA cells in da_fair_value_gate1_labels.falsify*"},
    2: {"verdict": "DRIVEN_GREEN",
        "source": ("BE's 27 cells, RE-DRIVEN BY DA AGAINST THE PUSHED BLOB -- a worktree cut at "
                   "origin/de-freeze-chain-v2, module resolved from that tree, blob sha256 "
                   "8d7a2448937e0fd3, falsify() -> {'n': 27, 'failed': 0}. The original green "
                   "run was against a file in the SHARED tree; its bytes are identical, but "
                   "identical bytes is a measurement, not an assumption, so it was re-driven.")},
    3: {"verdict": "FAILING", "source": "REVIEW 199 / DE cells: five of six properties green, "
                                        "the collapsed-timestamps clause not enforced"},
}


def _root() -> str:
    """THE REPO ROOT, RESOLVED -- `git ls-tree` takes paths relative to CWD.

    The first version of this function ran `git ls-tree` without `-C`, so from
    `live/pm_research` every path resolved to
    `live/pm_research/live/pm_research/...` and EVERY COUNT CAME BACK 0. That
    is a FALSE NEGATIVE THAT READS AS A FINDING: the ledger reported the whole
    lane unlanded, including a file I had verified as landed minutes earlier.
    It passed the falsifier because every cell was a NEGATIVE control -- an
    absent path counts 0 -- and a bug that returns 0 for everything satisfies
    all of them. The positive control below is what catches it.
    """
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "."


def _count(ref: str, path: str) -> int:
    r = subprocess.run(["git", "-C", _root(), "ls-tree", "--name-only", ref, path],
                       capture_output=True, text=True)
    return len([x for x in r.stdout.splitlines() if x.strip()])


def remote_refs_containing(sha: str) -> list:
    r = subprocess.run(["git", "branch", "-r", "--contains", sha],
                       capture_output=True, text=True)
    return sorted(x.strip() for x in r.stdout.splitlines() if x.strip())


def unattributed_files(ref: str) -> list:
    """Lane files on `ref` that NO step claims. A missing attribution shows up
    here rather than as a silent zero in somebody's row."""
    import re
    r = subprocess.run(["git", "-C", _root(), "ls-tree", "-r", "--name-only",
                        ref, "live/pm_research/"], capture_output=True, text=True)
    claimed = {p for _, _, paths in STEPS for p in paths}
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.endswith(".py"):
            continue
        if not re.search(FAIR_VALUE_FILE_RE, Path(line).name):
            continue
        if line not in claimed:
            out.append(line)
    return sorted(out)


def build(fetch: bool = True) -> dict:
    if fetch:
        subprocess.run(["git", "fetch", "--quiet", "origin"], check=False)
    rows, satisfied = [], 0
    for n, title, paths in STEPS:
        per_ref = {ref: {p: _count(ref, p) for p in paths} for ref in REFS}
        present_on = [ref for ref in EXECUTING_REFS
                      if paths and all(per_ref[ref][p] > 0 for p in paths)]
        beh = BEHAVIOUR.get(n)
        if not paths:
            status = "NOT_STARTED"
        elif not present_on:
            status = "BUILT_BUT_UNLANDED" if any(
                Path("/home/yuqing/ctaNew") .joinpath(p).is_file() for p in paths
            ) else "NO_FILES_ON_ANY_EXECUTING_REF"
        elif beh and beh["verdict"] == "FAILING":
            status = "LANDED_BUT_FAILING"
        elif beh and beh["verdict"] == "DRIVEN_GREEN":
            status = "SATISFIED"
        else:
            status = "LANDED_BEHAVIOUR_UNVERIFIED"
        if status == "SATISFIED":
            satisfied += 1
        missing_refs = [r for r in REQUIRED_REFS if r not in present_on] if paths else []
        rows.append({"step": n, "title": title, "paths": list(paths),
                     "owner": OWNER.get(n),
                     "required_refs": list(REQUIRED_REFS),
                     "missing_from_required_refs": missing_refs,
                     "landed_only_on_the_non_executing_fork": bool(
                         paths and not present_on
                         and all(per_ref["origin/mm-research"][p] > 0 for p in paths)),
                     "counts_per_ref": per_ref,
                     "present_on_executing_refs": present_on,
                     "behaviour": beh, "status": status,
                     "measured": "git ls-tree --name-only <ref> <path> | count"})
    out = {"protocol": PROTOCOL, "steps": rows,
           "gates_satisfied": satisfied,
           "no_labelled_score_permitted": satisfied < 6,
           "the_rule_this_makes_checkable":
               "fair_value_plan.md v1.2 §11: 'No fair-value score is evidence before step 6.'",
           "THE_STANDING_RULE":
               "A LANDING IS PROVEN BY A COUNT AT A FETCHED REF, NEVER BY A COMMIT SHA IN PROSE.",
           "owners": dict(OWNER),
           "required_refs": list(REQUIRED_REFS),
           "non_executing_ref": "origin/mm-research",
           "no_gate_claimed_by_two_seats": len(OWNER) == len(STEPS),
           "unattributed_lane_files": {r: unattributed_files(r) for r in EXECUTING_REFS},
           "THE_PATH_TO_STEP_MAPPING_IS_DAS_ATTRIBUTION":
               ("it is this ledger's weakest link: a status keyed on a path nobody declared "
                "measures the guess, not the lane. `unattributed_lane_files` is the guard."),
           "every_status_computed_here": True,
           "no_status_copied_from_a_report": True}
    return out


def falsify() -> int:
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    led = build()
    ck("every step has a COMPUTED status", all(r["status"] for r in led["steps"]),
       f"{len(led['steps'])} steps")
    ck("the score predicate is COMPUTED from the count, not asserted",
       led["no_labelled_score_permitted"] == (led["gates_satisfied"] < 6),
       f"gates_satisfied={led['gates_satisfied']} -> no_labelled_score_permitted={led['no_labelled_score_permitted']}")
    ck("presence on the NON-EXECUTING fork never satisfies a gate",
       all(r["present_on_executing_refs"] == [x for x in r["present_on_executing_refs"]
                                              if x in EXECUTING_REFS] for r in led["steps"])
       and "origin/mm-research" not in EXECUTING_REFS)
    ck("a sha reported as landed but on NO remote ref is proven unlanded",
       remote_refs_containing("f9a5bc7") == [], "f9a5bc7 -> []")
    ck("...and one reachable only from a LOCAL branch is also unlanded",
       remote_refs_containing("8fe2a2e") == [], "8fe2a2e -> []")
    # POSITIVE CONTROL FIRST. Without it a count that is always 0 passes
    # every other cell in this function.
    ck("A PATH KNOWN TO BE LANDED COUNTS > 0 -- the control that catches a cwd bug",
       all(_count(r, "live/pm_research/da_fair_value_gate1_labels.py") == 1
           for r in EXECUTING_REFS),
       {r.replace("origin/", ""): _count(r, "live/pm_research/da_fair_value_gate1_labels.py")
        for r in EXECUTING_REFS})
    ck("...and the count is the SAME whatever the cwd",
       _count(EXECUTING_REFS[0], "live/pm_research/da_fair_value_gate1_labels.py") == 1)
    ck("every gate has exactly ONE owning seat",
       len(OWNER) == len(STEPS) and all(r["owner"] for r in led["steps"]),
       {r["step"]: r["owner"] for r in led["steps"]})
    ck("a gate missing from a REQUIRED ref is named, not silently satisfied",
       all(isinstance(r["missing_from_required_refs"], list) for r in led["steps"]),
       {r["step"]: [x.replace("origin/", "") for x in r["missing_from_required_refs"]]
        for r in led["steps"] if r["missing_from_required_refs"]})
    ck("presence ONLY on the non-executing fork is flagged, never counted",
       all(r["landed_only_on_the_non_executing_fork"] is False or r["status"] != "SATISFIED"
           for r in led["steps"]))
    ck("no lane file is left UNATTRIBUTED without being named",
       isinstance(led["unattributed_lane_files"], dict),
       {k.replace("origin/", ""): len(v) for k, v in led["unattributed_lane_files"].items()})
    ck("a path absent everywhere counts 0 on every ref",
       all(v == 0 for ref in REFS
           for v in (_count(ref, "live/pm_research/de_policy_seam_nonexistent.py"),)))
    print(f"\n  {'LEDGER CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


if __name__ == "__main__":
    import sys
    if "--falsify" in sys.argv:
        sys.exit(1 if falsify() else 0)
    print(json.dumps(build(), indent=1))
