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
NO_POWER_STATED = "RESULT_DOES_NOT_STATE_THE_POWER_THAT_PRODUCED_IT"
NO_SELECTION_READ = "RESULT_DOES_NOT_CARRY_ITS_SELECTION_HISTORY_READING"
BAD_ARMS = "FORWARD_RESULT_ARMS_IS_NOT_AN_ARM_MAPPING"
NO_ROBUSTNESS = "RESULT_DOES_NOT_CARRY_THE_DECLARED_ROBUSTNESS_LEG"
ROBUSTNESS_REVERSAL = "PROMOTION_CLAIMED_DESPITE_ROBUSTNESS_SIGN_REVERSAL"

PASS_SELECTION_READING = (
    "This result raises the standing of two arms. It does not establish them, "
    "and it says nothing about the family the 69 came from.")
FAIL_SELECTION_READING = (
    "The verdict is NOT_ESTABLISHED_AT_THIS_POWER, never NO_EFFECT.")

# ---- the three STATUSES the declaration promises
MINORITY_DAYS = "ADVANCED_ON_A_MINORITY_OF_DAYS"
POOLED_MINORITY = "POOLED_PASS_DRIVEN_BY_A_MINORITY_OF_DAYS"
NINE_OH_SEVEN_SIGN = "THE_09_07_DAY_CHANGES_THE_SIGN"


class ForwardResultRefused(Exception):
    pass


def _sign(value: float) -> int:
    value = float(value)
    return 1 if value > 0 else -1 if value < 0 else 0


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
    floor = result.get("FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT")
    power = {k: result.get(k) for k in (
        "attainable_minimum_p", "tolerance_negative_days",
        "a_pass_was_possible_at_this_G")}
    if (not isinstance(floor, dict)
            or power["attainable_minimum_p"] is None
            or power["tolerance_negative_days"] is None
            or not isinstance(power["a_pass_was_possible_at_this_G"], bool)
            or power["attainable_minimum_p"]
               != floor.get("attainable_minimum_p")
            or power["tolerance_negative_days"]
               != floor.get("tolerance_negative_days")
            or power["a_pass_was_possible_at_this_G"]
               != floor.get("a_pass_was_possible_at_this_G")):
        raise ForwardResultRefused(
            f"REFUSED {NO_POWER_STATED}: the result must carry the "
            f"evaluator's attainable_minimum_p, tolerance_negative_days and "
            f"a_pass_was_possible_at_this_G beside its verdict, equal to the "
            f"values in FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT. A copied result "
            f"without its power is not quotable.")
    arms = result.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise ForwardResultRefused(
            f"REFUSED {BAD_ARMS}: expected a non-empty mapping of arm name "
            f"to arm result, got {type(arms).__name__}.")
    if not all(isinstance(a, dict) for a in arms.values()):
        raise ForwardResultRefused(
            f"REFUSED {BAD_ARMS}: every arm result must be a mapping.")
    any_advances = any(a.get("advances") is True for a in arms.values())
    expected_reading = (PASS_SELECTION_READING if any_advances
                        else FAIL_SELECTION_READING)
    if result.get("selection_history_reading") != expected_reading:
        raise ForwardResultRefused(
            f"REFUSED {NO_SELECTION_READ}: a result with "
            f"any_arm_advances={any_advances} must carry the applicable "
            f"pre-written REVIEW 174 reading exactly. Expected "
            f"{expected_reading!r}.")
    # ADVANCEMENT_CLAIMED_ON_ONE_COMPARISON
    for arm, a in arms.items():
        if a.get("advances") and not (a.get("beats_zero_cancel")
                                      and a.get("beats_matched_random")):
            raise ForwardResultRefused(
                f"REFUSED {ONE_COMPARISON}: {arm} is marked advancing with "
                f"beats_zero_cancel={a.get('beats_zero_cancel')} and "
                f"beats_matched_random={a.get('beats_matched_random')}. Both "
                f"are required; neither alone advances an arm.")
        robust = a.get("robustness_leg")
        if (not isinstance(robust, dict)
                or robust.get("label") != "NO_FILLS_UNTIL_NEXT_GENERATION"
                or robust.get("observed_D_cents") is None):
            raise ForwardResultRefused(
                f"REFUSED {NO_ROBUSTNESS}: {arm} does not carry the labelled "
                f"NO_FILLS_UNTIL_NEXT_GENERATION pooled delta.")
        primary = ((a.get("pooled") or {}).get("D_arm_cents"))
        if primary is None:
            raise ForwardResultRefused(
                f"REFUSED {BAD_ARMS}: {arm} carries no pooled "
                f"D_arm_cents for the primary REFERENCE_FILLS leg.")
        sign_reversal = (_sign(primary)
                         != _sign(robust["observed_D_cents"]))
        if robust.get("sign_reversal") != sign_reversal:
            raise ForwardResultRefused(
                f"REFUSED {NO_ROBUSTNESS}: {arm} reports "
                f"sign_reversal={robust.get('sign_reversal')!r}, but the "
                f"primary and robustness deltas compute to "
                f"{sign_reversal}.")
        if a.get("advances") and sign_reversal:
            raise ForwardResultRefused(
                f"REFUSED {ROBUSTNESS_REVERSAL}: {arm} advances with primary "
                f"D={primary} and robustness D={robust['observed_D_cents']}. "
                f"The declared consequence is to block promotion, never to "
                f"choose the favourable fill assumption.")
    # CELLS_COUNTED_AS_INDEPENDENT
    n = result.get("n_independent_units")
    if not isinstance(n, int) or n != n_days:
        raise ForwardResultRefused(
            f"REFUSED {CELLS_INDEPENDENT}: the result claims {n} independent "
            f"units on {n_days} days; it must claim exactly the number of "
            f"UTC days, no more and no fewer. Two arms share a day, a book, a "
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
    arms = result.get("arms") or {}
    if not isinstance(arms, dict):
        return out
    for arm, a in arms.items():
        days = a.get("per_day_delta") or []
        if a.get("advances") and sum(1 for d in days if d <= 0) >= 2:
            out.append({"status": MINORITY_DAYS, "arm": arm,
                        "n_non_positive_days": sum(1 for d in days if d <= 0),
                        "n_days": len(days),
                        "rule": "travels on the verdict; never omitted"})
        if (a.get("pooled_advances_before_robustness")
                and sum(1 for d in days if d <= 0) >= 2):
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
    _N.update({"n": 0, "bad": 0})
    floor = {"attainable_minimum_p": {"day_sign_component": 0.015625},
             "tolerance_negative_days": 0,
             "a_pass_was_possible_at_this_G": True}
    GOOD = {"forward_limits": "second attempt, btc only, L=250ms",
            "pipeline_provenance_limit": "build and valuation on 7ed5a90",
            "FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT": floor,
            "attainable_minimum_p": floor["attainable_minimum_p"],
            "tolerance_negative_days": 0,
            "a_pass_was_possible_at_this_G": True,
            "selection_history_reading": PASS_SELECTION_READING,
            "arms": {"CONDVALUE_X_SKEW": {"advances": True,
                                          "beats_zero_cancel": True,
                                          "beats_matched_random": True,
                                          "pooled": {"D_arm_cents": 7.0},
                                          "robustness_leg": {
                                              "label": "NO_FILLS_UNTIL_NEXT_GENERATION",
                                              "observed_D_cents": 6.0,
                                              "sign_reversal": False},
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
    b = copy.deepcopy(GOOD); b.pop("attainable_minimum_p")
    _ok(_fires(b, NO_POWER_STATED), f"{NO_POWER_STATED} FIRES")
    b = copy.deepcopy(GOOD); b["tolerance_negative_days"] = 1
    _ok(_fires(b, NO_POWER_STATED),
        f"{NO_POWER_STATED} FIRES when copied power disagrees with the floor")
    b = copy.deepcopy(GOOD); b.pop("selection_history_reading")
    _ok(_fires(b, NO_SELECTION_READ), f"{NO_SELECTION_READ} FIRES")
    b = copy.deepcopy(GOOD); b["selection_history_reading"] = FAIL_SELECTION_READING
    _ok(_fires(b, NO_SELECTION_READ),
        f"{NO_SELECTION_READ} FIRES on the FAIL sentence beside a PASS")
    b = copy.deepcopy(GOOD); b["arms"] = ["CONDVALUE_X_SKEW"]
    _ok(_fires(b, BAD_ARMS), f"{BAD_ARMS} FIRES instead of crashing")
    b = copy.deepcopy(GOOD); b["arms"]["CONDVALUE_X_SKEW"]["beats_matched_random"] = False
    _ok(_fires(b, ONE_COMPARISON),
        f"{ONE_COMPARISON} FIRES -- one comparison is not an advancement")
    b = copy.deepcopy(GOOD); b["n_independent_units"] = 14
    _ok(_fires(b, CELLS_INDEPENDENT),
        f"{CELLS_INDEPENDENT} FIRES -- 14 cells are not 14 units")
    b = copy.deepcopy(GOOD); b.pop("n_independent_units")
    _ok(_fires(b, CELLS_INDEPENDENT),
        f"{CELLS_INDEPENDENT} FIRES when the unit count is absent")
    b = copy.deepcopy(GOOD); b["per_cell"] = {"2026-09-08|CONDVALUE_X_SKEW": {"verdict": "PASS"}}
    _ok(_fires(b, CELL_VERDICT), f"{CELL_VERDICT} FIRES")
    b = copy.deepcopy(GOOD)
    b["quality_decisions"]["2026-09-08"]["sources_read"].append(
        "data/pm_5min/derived/settle/de_settle_result_20260908.json")
    _ok(_fires(b, QUALITY_SAW_OUTCOME),
        f"{QUALITY_SAW_OUTCOME} FIRES -- THE NAME REV FOUND HAD NO PRODUCER, "
        f"driven here under the name the declaration promises")
    b = copy.deepcopy(GOOD)
    b["arms"]["CONDVALUE_X_SKEW"].pop("robustness_leg")
    _ok(_fires(b, NO_ROBUSTNESS), f"{NO_ROBUSTNESS} FIRES")
    b = copy.deepcopy(GOOD)
    b["arms"]["CONDVALUE_X_SKEW"]["robustness_leg"].update(
        {"observed_D_cents": -1.0, "sign_reversal": True})
    _ok(_fires(b, ROBUSTNESS_REVERSAL),
        f"{ROBUSTNESS_REVERSAL} FIRES on an advancing sign reversal")
    b["arms"]["CONDVALUE_X_SKEW"]["advances"] = False
    b["selection_history_reading"] = FAIL_SELECTION_READING
    _ok(require_forward_result(b)["status"] == "FORWARD_RESULT_FIELDS_PRESENT",
        "a sign reversal ADMITS when promotion is blocked and reported")

    # every declared STATUS is EMITTED by its declared name
    b = copy.deepcopy(GOOD)
    b["arms"]["CONDVALUE_X_SKEW"]["per_day_delta"] = [1, 1, 1, 1, -1, -1, 1]
    st = {x["status"] for x in classify_forward_result(b)}
    _ok(MINORITY_DAYS in st, f"{MINORITY_DAYS} EMITTED on 2 non-positive days")
    b["arms"]["CONDVALUE_X_SKEW"][
        "pooled_advances_before_robustness"] = True
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
              f"checks, {_N['bad']} failures | every result-bearing refusal "
              f"fires and every declared status emits")
    return 1 if _N["bad"] else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
