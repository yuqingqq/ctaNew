#!/usr/bin/env python3
"""THE PRODUCER FOR EVERY REFUSAL AND STATUS THE FORWARD DECLARATION PROMISES.

REV found `QUALITY_DECISION_SAW_AN_OUTCOME` declared with no producer: the
governing document named a refusal the enforcing instrument raised under a
DIFFERENT name, so a reader trusting that name was trusting nothing. Rule 15 --
"a zero from an instrument that never proved it can fire is not a result."

ENUMERATED BY THE OPERATION rather than fixing the one REV named: of 24 names
my declarations promise, FIVE REFUSALS had no reachable raise and THREE
STATUSES were never emittable. All eight are promises about a RESULT-EMITTING
path that did not exist, because the forward result has not been produced yet.
This module is that path, so the promises become checkable before the first
result rather than after.

(Three audit passes were needed to get that number. The first counted "the name
appears in a .py file", which is mention and not production; the second walked
only the `raise` node and missed refusals assembled from a list; the third
missed refusals raised through a module CONSTANT -- the name/value trap again.
The honest figure is the fourth.)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ---- the five REFUSALS the declaration promises, as this module's constants
ONE_COMPARISON = "ADVANCEMENT_CLAIMED_ON_ONE_COMPARISON"
CELLS_INDEPENDENT = "CELLS_COUNTED_AS_INDEPENDENT"
CELL_VERDICT = "CELL_LEVEL_VERDICT_REPORTED"
QUALITY_SAW_OUTCOME = "QUALITY_DECISION_SAW_AN_OUTCOME"
NO_FORWARD_LIMITS = "RESULT_DOES_NOT_STATE_ITS_FORWARD_LIMITS"
# DA 218: v4 adds a refusal, so it gets a producer in the SAME commit.
NO_PIPELINE_LIMIT = "RESULT_DOES_NOT_STATE_ITS_PIPELINE_PROVENANCE_LIMIT"

# ---- the three STATUSES the declaration promises
MINORITY_DAYS = "ADVANCED_ON_A_MINORITY_OF_DAYS"
POOLED_MINORITY = "POOLED_PASS_DRIVEN_BY_A_MINORITY_OF_DAYS"
NINE_OH_SEVEN_SIGN = "THE_09_07_DAY_CHANGES_THE_SIGN"


class ForwardResultRefused(Exception):
    pass


def require_forward_result(result: dict, n_days: int = 7) -> dict:
    """Every refusal the forward declaration promises, raised BY THAT NAME."""
    # RESULT_DOES_NOT_STATE_ITS_FORWARD_LIMITS
    if not result.get("forward_limits"):
        raise ForwardResultRefused(
            f"REFUSED {NO_FORWARD_LIMITS}: the result carries no "
            f"`forward_limits` field. A pass on a second attempt, one coin, "
            f"one latency, one fill assumption must say so ON the result.")
    # RESULT_DOES_NOT_STATE_ITS_PIPELINE_PROVENANCE_LIMIT
    if not result.get("pipeline_provenance_limit"):
        raise ForwardResultRefused(
            f"REFUSED {NO_PIPELINE_LIMIT}: the result carries no "
            f"`pipeline_provenance_limit`. The whole pipeline runs on code the "
            f"development screen never ran on; a result that cannot say so is "
            f"not quotable.")
    # ADVANCEMENT_CLAIMED_ON_ONE_COMPARISON
    for arm, a in (result.get("arms") or {}).items():
        if a.get("advances") and not (a.get("beats_zero_cancel")
                                      and a.get("beats_matched_random")):
            raise ForwardResultRefused(
                f"REFUSED {ONE_COMPARISON}: {arm} is marked advancing with "
                f"beats_zero_cancel={a.get('beats_zero_cancel')} and "
                f"beats_matched_random={a.get('beats_matched_random')}. Both "
                f"are required; neither alone advances an arm.")
    # CELLS_COUNTED_AS_INDEPENDENT
    n = result.get("n_independent_units")
    if n is not None and n > n_days:
        raise ForwardResultRefused(
            f"REFUSED {CELLS_INDEPENDENT}: the result claims {n} independent "
            f"units on {n_days} days. Two arms share a day, a book, a "
            f"reference path and a baseline; the cluster unit is the UTC day.")
    # CELL_LEVEL_VERDICT_REPORTED
    for cell, c in (result.get("per_cell") or {}).items():
        if c.get("verdict") is not None:
            raise ForwardResultRefused(
                f"REFUSED {CELL_VERDICT}: cell {cell} carries a verdict. "
                f"Per-cell outcomes are DESCRIPTIVE; the verdict comes from "
                f"the two arm-level tests and from nothing else.")
    # QUALITY_DECISION_SAW_AN_OUTCOME
    for day, q in (result.get("quality_decisions") or {}).items():
        srcs = q.get("sources_read") or []
        bad = [s for s in srcs if not any(
            a in s for a in ("data/pm_5min/raw",
                             "data/pm_5min/collector_gaps.jsonl"))]
        if bad:
            raise ForwardResultRefused(
                f"REFUSED {QUALITY_SAW_OUTCOME}: the admissibility decision "
                f"for {day} read {bad}. A quality rule that can see an "
                f"outcome is a selection rule.")
    return {"status": "FORWARD_RESULT_FIELDS_PRESENT", "n_days": n_days}


def classify_forward_result(result: dict, n_days: int = 7) -> list:
    """The statuses the declaration promises, EMITTED by that name."""
    out = []
    for arm, a in (result.get("arms") or {}).items():
        days = a.get("per_day_delta") or []
        if a.get("advances") and sum(1 for d in days if d <= 0) >= 2:
            out.append({"status": MINORITY_DAYS, "arm": arm,
                        "n_non_positive_days": sum(1 for d in days if d <= 0),
                        "n_days": len(days),
                        "rule": "travels on the verdict; never omitted"})
        if a.get("pooled_advances") and sum(1 for d in days if d <= 0) >= 2:
            out.append({"status": POOLED_MINORITY, "arm": arm})
    s = result.get("sensitivity_09_07") or {}
    if s.get("primary_sign") is not None and s.get("without_sign") is not None \
            and s["primary_sign"] != s["without_sign"]:
        out.append({"status": NINE_OH_SEVEN_SIGN,
                    "rule": "promotion is BLOCKED; it is a finding about 09-07"})
    return out


# ------------------------------------------------------------- falsifier

_N = {"n": 0, "bad": 0}


def _ok(c, label):
    _N["n"] += 1
    if not c:
        _N["bad"] += 1
        print(f"  FAIL {label}")
    return c


def _fires(result, name, n_days=7):
    try:
        require_forward_result(result, n_days)
        return False
    except ForwardResultRefused as e:
        return name in str(e)


def selftest(quiet: bool = False) -> int:
    GOOD = {"forward_limits": "second attempt, btc only, L=250ms",
            "pipeline_provenance_limit": "build and valuation on 7ed5a90",
            "arms": {"CONDVALUE_X_SKEW": {"advances": True,
                                          "beats_zero_cancel": True,
                                          "beats_matched_random": True,
                                          "per_day_delta": [1, 1, 1, 1, 1, 1, 1]}},
            "n_independent_units": 7, "per_cell": {},
            "quality_decisions": {"2026-09-08": {"sources_read": [
                "data/pm_5min/raw", "data/pm_5min/collector_gaps.jsonl"]}}}
    _ok(require_forward_result(GOOD)["status"] == "FORWARD_RESULT_FIELDS_PRESENT",
        "POSITIVE CONTROL: a complete, well-formed forward result PASSES")

    import copy
    # every declared REFUSAL fires BY ITS DECLARED NAME
    b = copy.deepcopy(GOOD); b.pop("forward_limits")
    _ok(_fires(b, NO_FORWARD_LIMITS), f"{NO_FORWARD_LIMITS} FIRES")
    b = copy.deepcopy(GOOD); b.pop("pipeline_provenance_limit")
    _ok(_fires(b, NO_PIPELINE_LIMIT),
        f"{NO_PIPELINE_LIMIT} FIRES -- added by v4 and given a producer in the "
        f"SAME commit, so it is never a promise without one")
    b = copy.deepcopy(GOOD); b["arms"]["CONDVALUE_X_SKEW"]["beats_matched_random"] = False
    _ok(_fires(b, ONE_COMPARISON),
        f"{ONE_COMPARISON} FIRES -- one comparison is not an advancement")
    b = copy.deepcopy(GOOD); b["n_independent_units"] = 14
    _ok(_fires(b, CELLS_INDEPENDENT),
        f"{CELLS_INDEPENDENT} FIRES -- 14 cells are not 14 units")
    b = copy.deepcopy(GOOD); b["per_cell"] = {"2026-09-08|CONDVALUE_X_SKEW": {"verdict": "PASS"}}
    _ok(_fires(b, CELL_VERDICT), f"{CELL_VERDICT} FIRES")
    b = copy.deepcopy(GOOD)
    b["quality_decisions"]["2026-09-08"]["sources_read"].append(
        "data/pm_5min/derived/settle/de_settle_result_20260908.json")
    _ok(_fires(b, QUALITY_SAW_OUTCOME),
        f"{QUALITY_SAW_OUTCOME} FIRES -- THE NAME REV FOUND HAD NO PRODUCER, "
        f"driven here under the name the declaration promises")

    # every declared STATUS is EMITTED by its declared name
    b = copy.deepcopy(GOOD)
    b["arms"]["CONDVALUE_X_SKEW"]["per_day_delta"] = [1, 1, 1, 1, -1, -1, 1]
    st = {x["status"] for x in classify_forward_result(b)}
    _ok(MINORITY_DAYS in st, f"{MINORITY_DAYS} EMITTED on 2 non-positive days")
    b["arms"]["CONDVALUE_X_SKEW"]["pooled_advances"] = True
    _ok(POOLED_MINORITY in {x["status"] for x in classify_forward_result(b)},
        f"{POOLED_MINORITY} EMITTED")
    b = copy.deepcopy(GOOD)
    b["sensitivity_09_07"] = {"primary_sign": 1, "without_sign": -1}
    _ok(NINE_OH_SEVEN_SIGN in {x["status"] for x in classify_forward_result(b)},
        f"{NINE_OH_SEVEN_SIGN} EMITTED on a sign difference")
    # and NOT emitted when the signs agree -- it must not fire on everything
    b["sensitivity_09_07"] = {"primary_sign": 1, "without_sign": 1}
    _ok(NINE_OH_SEVEN_SIGN not in {x["status"] for x in classify_forward_result(b)},
        "and it does NOT fire when the two legs agree -- a status that fires "
        "always is not a status")

    if not quiet:
        print(f"[da_forward_result_guard] {_N['n'] - _N['bad']}/{_N['n']} "
              f"checks, {_N['bad']} failures | 5 refusals FIRE by their "
              f"declared names, 3 statuses EMIT by theirs")
    return 1 if _N["bad"] else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
