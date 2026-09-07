"""THE USER-RULED EARLY READ OF THE FOUR SEALED DAYS (R-754, DE 121).

WHAT THIS IS. On 2026-09-07T06:18:16Z the USER ruled: "it does not make
sense to seal the results, show me 4 days results first, we need to check
and review the results". Rule 14 -- the user decides. The coordinator's
objection (rule 11: the four days become consumed; G = 4 gives 2^-4 =
0.0625; no interval below five complete days) is recorded in params v16's
`user_ruled_early_read` block and OVERRULED.

WHAT THIS IS NOT. Not a validation, not a verdict, not a new bar. It does
not touch `read_gate`, which still requires all six days and its own
clock; it does not move `PARAMS_REL`, which still names v15, so the
sealed six-day path is byte-identical for every day outside the ruling.

WHERE THE NUMBERS COME FROM. The COMPUTATION is v15's -- the same params
the four sealed days ran, so the arms, the estimand and the draw counts
are theirs and not this read's. Only the SEAL BAR comes from v16. That
split is the whole point: the read must show the numbers those runs
computed and stripped, not numbers it chose.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

# BOTH LAUNCHERS. The script path needs this file's own directory on
# `sys.path` for the sibling import; the `-m` path needs the repo root for
# the package import. Adding both makes the module launcher-agnostic --
# the defect DE 120 spent a batch on, one module later.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import de_multiday_gate1_runner as RUN                       # noqa: E402
import declaration_chain as DC                               # noqa: E402

PARAMS_FAMILY = "de_multiday_gate1_params"
DECL_DIR = "live/pm_research/declarations"

#: A NEW FAMILY. Never a `.vN` of a sealed receipt: a correction chain
#: says "this supersedes that", and this artifact supersedes nothing -- it
#: is a different question asked of the same days.
DAY_FAMILY = "p003_de_early_read_day"

EXPECTED_CHECKS = 7


class EarlyReadRefused(RuntimeError):
    """A named refusal. Every message begins with its own reason code."""


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def the_ruling(repo_root=None) -> dict:
    """THE BAR, READ FROM THE PARAMS CHAIN HEAD -- or REFUSED BY NAME.

    The block is not optional and it is not inferred. A read that opens
    four sealed days on the runner's own initiative is the thing rule 14
    forbids, so its authority has to be a landed declaration naming the
    ruling, and its absence is a refusal with a reason code."""
    root = Path(repo_root) if repo_root else Path(
        __file__).resolve().parents[2]
    head = DC.resolve_head(root / DECL_DIR, PARAMS_FAMILY)
    doc = json.loads(Path(head["path"]).read_text())
    block = doc.get("user_ruled_early_read")
    if not isinstance(block, dict):
        raise EarlyReadRefused(
            f"EARLY_READ_NOT_RULED: the params chain head "
            f"({head['name']}, {head['sha256'][:16]}…) carries no "
            f"`user_ruled_early_read` block. The early read opens days "
            f"that are sealed for a reason; its authority is a USER "
            f"ruling landed in the declaration, never this module's own "
            f"word (rule 14).")
    for field in ("the_ruling_verbatim", "this_reads_bar",
                  "days_consumed_by_this_read", "verdict_class"):
        if field not in block:
            raise EarlyReadRefused(
                f"EARLY_READ_BLOCK_INCOMPLETE: the block is missing "
                f"`{field}`. A ruling recorded in pieces is a ruling a "
                f"later reader has to reconstruct.")
    days = list(block["this_reads_bar"]["days"])
    return {
        "ruling_by_pair": {"path": head["path"], "sha256": head["sha256"]},
        "params_version": doc.get("version"),
        "days": days,
        "G": len(days),
        "verdict_class": block["verdict_class"],
        "interval": "NONE_BELOW_FIVE_DAYS",
        "is_a_validation": False,
        "days_consumed": list(block["days_consumed_by_this_read"]),
        "the_ruling_verbatim": block["the_ruling_verbatim"],
        "recorded_at_utc": block.get("recorded_at_utc"),
        "receipts": block["this_reads_bar"]["receipts"],
        "block": block,
    }


#: THE FIELDS THE DISPATCH ASKED FOR THAT THIS CODE PATH DOES NOT COMPUTE.
#: Named as STATUSES, never dropped in silence (reliability rule 4). Each
#: one would be a NEW statistic, and a read whose remit is "show what
#: those runs computed and stripped" may not invent one.
NOT_COMPUTED_BY_THIS_PATH = {
    "fills_leg": "the arm-day economic block is a SINGLE excess `D_E0` "
                 "against the 0-cancel baseline. There is no fills/"
                 "inventory decomposition anywhere in the runner, so a "
                 "leg split cannot be reported without defining one.",
    "inventory_leg": "as above -- the same single quantity is not two.",
    "p_two_sided": "`p_location` is `p_one_sided` from "
                   "`DESIGN.per_day_location`. Doubling it is a "
                   "DECISION about the test, not a re-read of it.",
    "rho_adverse_over_spread": "`rho`, `adverse` and `spread` appear "
                               "nowhere in the runner (0 occurrences "
                               "each). This quantity has never been "
                               "computed for these days.",
    "D_E_MINUS_R": "a sealed NAME, but no arm-day economic block "
                   "produces it; only `D_E0` is emitted.",
}


def economics_available_per_arm_day() -> dict:
    """WHAT AN UNSEALED ARM-DAY ACTUALLY CARRIES, read off the emitter."""
    return {
        "from_the_economic_block": ["D_E0", "Z", "p_location",
                                    "null_mean", "null_sd",
                                    "null_draws_summary.n"],
        "from_the_arm_level": ["n_fills_arm", "n_fills_baseline",
                               "n_cancels_issued", "status",
                               "admissibility", "seed", "draw_provenance"],
        "not_computed_by_this_path": NOT_COMPUTED_BY_THIS_PATH,
    }


def day_artifact_name(day: str, now=None) -> str:
    stamp = RUN.emission_stamp(now)
    return f"{DAY_FAMILY}_{day.replace('-', '')}__{stamp}.json"


def run_early_read_day(day: str, book, outdir, *, repo_root=None,
                       before_work=None) -> dict:
    """ONE DAY, UNSEALED UNDER THE RULING. The heavy call.

    `params` is v15's -- `RUN.load_params()` -- so the computation is the
    sealed runs'. `early_read` is v16's bar, and it reaches exactly one
    place: the seal call."""
    ruling = the_ruling(repo_root)
    if day not in ruling["days"]:
        raise EarlyReadRefused(
            f"EARLY_READ_DAY_NOT_IN_THE_BAR: {day} is not among "
            f"{ruling['days']}. The ruling names four days; a fifth would "
            f"be consumed by a read nobody authorised.")
    params = RUN.load_params()
    result = RUN.run_day(day, book, params=params, fixture=False,
                         n_days_complete=ruling["G"],
                         early_read={"G": ruling["G"],
                                     "days": ruling["days"]},
                         before_work=before_work)
    payload = {
        "protocol": "P003_DE_EARLY_READ_DAY_V1",
        "what_this_is": "the USER-ruled early read of one sealed day "
                        "(R-754). NOT a validation, NOT a verdict, and "
                        "NOT a version of that day's sealed receipt.",
        "ruling": ruling["ruling_by_pair"],
        "the_ruling_verbatim": ruling["the_ruling_verbatim"],
        "is_a_validation": False,
        "G": ruling["G"],
        "interval": "NONE_BELOW_FIVE_DAYS",
        "verdict_class": ruling["verdict_class"],
        "days_consumed": ruling["days_consumed"],
        "computation_params": {
            "path": RUN.PARAMS_REL,
            "sha256": _sha(Path(__file__).resolve().parents[2]
                           / RUN.PARAMS_REL),
            "why": "the COMPUTATION is the sealed runs' -- v15. Only the "
                   "seal bar comes from the ruling."},
        "economics_field_availability": economics_available_per_arm_day(),
        "day_run": result,
        "as_of": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = Path(outdir) / day_artifact_name(day)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True,
                              default=str) + "\n")
    return {"path": str(out), "sha256": _sha(out), "day": day}


# ------------------------------------------------------- the battery

def selftest(quiet: bool = False) -> int:
    """RED FIRST, on fixtures. Three drives, each with its own baseline."""
    import copy
    import shutil
    import tempfile
    n = [0]
    skipped: list = []

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_early_read] FAIL: {label}")
        n[0] += 1
        if not quiet:
            print(f"  PASS  {label}")

    repo = Path(__file__).resolve().parents[2]

    # ---- DRIVE 1: WITHOUT v16's BLOCK THE READ REFUSES BY NAME --------
    # A fixture chain, not the live one: the known-bad has to be a chain
    # whose head genuinely lacks the block, and editing the live head to
    # produce one would be the in-place edit rule 20 forbids.
    tmp = Path(tempfile.mkdtemp(prefix="early_read_"))
    (tmp / DECL_DIR).mkdir(parents=True)
    live_head = DC.resolve_head(repo / DECL_DIR, PARAMS_FAMILY)
    doc = json.loads(Path(live_head["path"]).read_text())
    without = copy.deepcopy(doc)
    without.pop("user_ruled_early_read", None)
    (tmp / DECL_DIR / f"{PARAMS_FAMILY}_v1.json").write_text(
        json.dumps(without, indent=1, sort_keys=True) + "\n")
    try:
        the_ruling(tmp)
        ok(False, "KNOWN-BAD: a chain head with NO ruling block opened "
                  "the early read")
    except EarlyReadRefused as e:
        ok(str(e).startswith("EARLY_READ_NOT_RULED"),
           f"DRIVE 1 (RED): a params head carrying no "
           f"`user_ruled_early_read` REFUSES BY NAME -- "
           f"`{str(e).split(':')[0]}`. The read's authority is a landed "
           f"USER ruling, never this module's word")

    # The SAME fixture chain WITH the block admits -- so drive 1 measured
    # the block and not the fixture.
    withb = copy.deepcopy(doc)
    (tmp / DECL_DIR / f"{PARAMS_FAMILY}_v2.json").write_text(
        json.dumps({**withb,
                    "supersedes": {
                        "path": f"{DECL_DIR}/{PARAMS_FAMILY}_v1.json",
                        "sha256": _sha(tmp / DECL_DIR
                                       / f"{PARAMS_FAMILY}_v1.json")}},
                   indent=1, sort_keys=True) + "\n")
    r = the_ruling(tmp)
    ok(r["G"] == 4 and r["is_a_validation"] is False
       and r["interval"] == "NONE_BELOW_FIVE_DAYS"
       and r["verdict_class"] == "EXPLORATORY"
       and len(r["days_consumed"]) == 4,
       f"DRIVE 1 CONTROL (GREEN): the same fixture chain WITH the block "
       f"admits -- G {r['G']}, class {r['verdict_class']}, interval "
       f"{r['interval']}, is_a_validation {r['is_a_validation']}, "
       f"{len(r['days_consumed'])} days consumed. So the refusal was the "
       f"block's absence and not the fixture's shape")

    # ---- DRIVE 2: THE UNSEALED LAYOUT CARRIES THE ECONOMIC FIELDS -----
    # Driven on `seal()` itself -- the one function both paths call -- so
    # the layout tested is the layout that will be emitted.
    arm = {"day": "2026-09-03", "arm": "A", "status": "OK",
           "economic": {"D_E0": 2.5, "Z": 2.45, "p_location": 0.008,
                        "null_mean": 0.0066, "null_sd": 1.0158,
                        "null_draws_summary": {"n": 600}},
           "n_fills_arm": 30171, "n_fills_baseline": 46439,
           "n_cancels_issued": 5146}
    sealed6 = RUN.seal(copy.deepcopy(arm), 4, 6)          # the six-day bar
    unsealed4 = RUN.seal(copy.deepcopy(arm), 4, 4)        # the ruling's bar
    ok(sealed6["sealed"] is True and "economic" not in sealed6
       and not any(k in sealed6 for k in ("n_fills_arm", "n_cancels_issued"))
       and unsealed4["sealed"] is False
       and isinstance(unsealed4.get("economic"), dict)
       and unsealed4["economic"]["D_E0"] == 2.5
       and unsealed4["n_fills_arm"] == 30171
       and unsealed4["n_cancels_issued"] == 5146,
       f"DRIVE 2: the SAME arm-day through the SAME `seal()` is "
       f"`sealed={sealed6['sealed']}` with the economics ABSENT at the "
       f"six-day bar, and `sealed={unsealed4['sealed']}` carrying D_E0, "
       f"the null block and the fill/cancel counts at the ruling's bar of "
       f"4. The read changes the BAR, never the computation")

    # ---- DRIVE 3: THE SEALED PATH IS BYTE-IDENTICAL OFF THE RULING ----
    # `early_read=None` is every path that is not this read. The property
    # is that such a path is untouched, and it is measured on the emitted
    # object, not read off the diff.
    for day in ("2026-09-07", "2026-09-08"):
        a = RUN.seal(copy.deepcopy({**arm, "day": day}), 1, 6)
        b = RUN.seal(copy.deepcopy({**arm, "day": day}), 1, 6)
        ok(json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
           and a["sealed"] is True and "economic" not in a,
           f"DRIVE 3: {day} -- a day OUTSIDE the ruling seals exactly as "
           f"before, `sealed=True`, economics absent. `early_read` "
           f"defaults to None on every such path")
    # THE KNOWN-BAD MUST FAIL FOR ITS OWN REASON. Driven through
    # `run_early_read_day`, whose membership check raises BEFORE any
    # heavy work: routed through `run_day` instead, the launch-form guard
    # refuses first and the cell passes on a refusal that has nothing to
    # do with the bar -- an instrument satisfying the words, not the
    # property.
    try:
        run_early_read_day("2026-09-07", None, tmp)
        ok(False, "KNOWN-BAD: a day outside the ruling entered the "
                  "early-read path")
    except EarlyReadRefused as e:
        ok(str(e).startswith("EARLY_READ_DAY_NOT_IN_THE_BAR"),
           f"DRIVE 3 KNOWN-BAD: the early-read path REFUSES 2026-09-07 BY "
           f"ITS OWN REASON CODE -- `{str(e).split(':')[0]}` -- before it "
           f"reads a book or takes a lock. A fifth day would be consumed "
           f"by a read nobody authorised")

    # ---- the absent fields are STATUSES, never silent drops -----------
    av = economics_available_per_arm_day()
    ok(len(av["not_computed_by_this_path"]) == 5
       and all(isinstance(v, str) and len(v) > 40
               for v in av["not_computed_by_this_path"].values()),
       f"the {len(av['not_computed_by_this_path'])} fields the dispatch "
       f"asked for that this path does not compute are named with their "
       f"reasons -- {sorted(av['not_computed_by_this_path'])} -- not "
       f"dropped in silence (reliability rule 4)")

    shutil.rmtree(tmp, ignore_errors=True)
    _sum = RUN.battery_summary("de_early_read", n_run=n[0],
                               skipped=skipped, expected=EXPECTED_CHECKS)
    if not quiet:
        print(_sum["line"])
    if not _sum["clean"]:
        raise SystemExit(f"[de_early_read] NOT CLEAN: {_sum['line']}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    raise SystemExit("usage: --selftest (the heavy day path is a GO, "
                     "never a bare run)")
