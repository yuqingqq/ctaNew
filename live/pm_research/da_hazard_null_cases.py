#!/usr/bin/env python3
"""DA 165: DRIVEN CASES FOR HAZARD's NULL, BUILT BEFORE IT LANDS.

The first null since the retraction. These are written against the
PROPERTY, not against whatever the run happens to produce -- a case
written after seeing the artifact tests the artifact's shape, not its
correctness, and rule 11 says choosing after seeing voids the test.

Each case must PASS on the real thing, FAIL on a known-bad, and REFUSE a
partial input (rule 33). Every control below is one that COULD fail: the
known-bads are constructed so that a case which merely reads fields would
report agreement.

  python3 live/pm_research/da_hazard_null_cases.py --selftest
  python3 live/pm_research/da_hazard_null_cases.py --artifact <path>
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL, REFUSED = "PROPERTY_HOLDS", "PROPERTY_VIOLATED", "REFUSED"


class CaseRefused(RuntimeError):
    """The case cannot be evaluated on the input it was given."""


def _arms(artifact: dict) -> list:
    """Every per-arm block, from either artifact shape."""
    if not isinstance(artifact, dict):
        raise CaseRefused("PARTIAL INPUT: the artifact is not a mapping.")
    arms = (artifact.get("day_run") or artifact).get(
        "per_day_sealed_artifacts")
    if not arms:
        raise CaseRefused(
            "PARTIAL INPUT: no `per_day_sealed_artifacts`. A case that "
            "returned PROPERTY_HOLDS over an empty arm list would be "
            "certifying an empty set -- which is the exact shape this "
            "programme keeps paying for.")
    return list(arms)


# ---------------------------------------------------------------- case 1
NULL_UNITS = ("CANCELS", "DECISIONS")
NULL_KWARGS = ("matched_on", "arm_cancels", "control_set_path")


def _call_kwargs_at(func, callee: str) -> list:
    """Every keyword NAME passed to `callee` inside `func`'s own body.

    Reads the CALLER. The original defect was a branch that was unit
    tested while its ONLY call site passed neither argument, so a case
    that reads the callee's signature cannot see it.
    """
    src = inspect.getsource(func)
    tree = ast.parse(src[src.index("def "):])
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id",
                                                                     None)
            if nm == callee:
                out.append(sorted(k.arg for k in n.keywords if k.arg))
    return out


def case_1_matching_unit_declared_and_enforced(artifact, *, entry=None,
                                               refuser=None):
    """(1) Is the matching unit DECLARED in the artifact and ENFORCED in
    the code -- end to end from the entry point?

    Two halves, and neither substitutes for the other. The ARTIFACT must
    name `CANCELS`; the ENTRY POINT must pass the arguments that
    implement it; and the refusal must fire when a caller cannot name its
    unit, rather than the run falling back to another null.
    """
    arms = _arms(artifact)
    declared, drawn = {}, []
    for a in arms:
        prov = a.get("draw_provenance")
        nm = a.get("arm")
        if prov is None:
            declared[nm] = "NO_DRAWS_ON_THIS_ARM"
            continue
        drawn.append(nm)
        declared[nm] = (prov or {}).get("matched_on", "ABSENT")
    if not drawn:
        raise CaseRefused(
            "PARTIAL INPUT: no arm in this artifact carries a "
            "`draw_provenance`, so no null was drawn and there is no "
            "declaration to check. A point-estimate run is not a null run "
            "and this case must not report PROPERTY_HOLDS over it.")
    bad = {k: v for k, v in declared.items()
           if v not in ("NO_DRAWS_ON_THIS_ARM", "CANCELS")}

    # ---- the ENTRY POINT, read at the caller ---------------------------
    entry_sites, entry_reaching = None, None
    if entry is None:
        try:
            import de_multiday_gate1_runner as R
            entry = R.run_day
        except Exception:                                     # noqa: BLE001
            entry = None
    if entry is not None:
        entry_sites = _call_kwargs_at(entry, "null_draws_valued")
        entry_reaching = [c for c in entry_sites
                          if all(k in c for k in NULL_KWARGS)]

    # ---- the REFUSAL, DRIVEN rather than read --------------------------
    refusals = {}
    if refuser is None:
        try:
            import de_multiday_gate1_runner as R
            refuser = R.null_draws_valued
        except Exception:                                     # noqa: BLE001
            refuser = None
    if refuser is not None:
        for label, kw in (
                ("matched_on omitted", {}),
                ("matched_on None", {"matched_on": None}),
                ("matched_on 'ROWS'", {"matched_on": "ROWS"}),
                ("CANCELS with no arm_cancels",
                 {"matched_on": "CANCELS", "control_set_path": "x"}),
                ("CANCELS with no control_set_path",
                 {"matched_on": "CANCELS", "arm_cancels": []})):
            try:
                refuser(None, None, [], {}, n_draws=1, seed=0, **kw)
                refusals[label] = "DID_NOT_REFUSE"
            except Exception as e:                            # noqa: BLE001
                refusals[label] = type(e).__name__ + ": " + str(e)[:60]
    named = {k: v for k, v in refusals.items()
             if "MATCHING_UNIT" in str(v)}
    ok = (not bad
          and (entry_reaching is None or len(entry_reaching) >= 1)
          and (not refusals or len(named) == len(refusals)))
    return {"case": "matching_unit_declared_and_enforced",
            "verdict": PASS if ok else FAIL,
            "declared_per_arm": declared,
            "arms_that_drew": drawn,
            "arms_declaring_something_other_than_CANCELS": bad,
            "entry_point_call_sites": entry_sites,
            "entry_point_sites_passing_all_three": (
                None if entry_reaching is None else len(entry_reaching)),
            "refusals_driven": refusals,
            "refusals_that_named_the_matching_unit": len(named),
            "why": ("a declaration nobody enforces is prose, and an "
                    "enforcement nobody declares is unauditable. The unit "
                    "must be NAMED in the artifact, PASSED at the entry "
                    "point, and REFUSED when absent -- the original defect "
                    "was a tested branch whose only call site passed "
                    "nothing")}


# ---------------------------------------------------------------- case 2
# DA 195 / plan v2 step 1: DERIVED, never typed. The floor lived in
# THIRTEEN carriers under FOUR spellings; `p003_rule6_floor` is the one
# authority and this name now reads from it.
from p003_rule6_floor import FLOOR as RULE_6_FLOOR  # noqa: E402


def case_2_rule_6_declared_before_the_result(artifact, params, *,
                                             checker=None):
    """(2) Rule 6 as a PROPERTY: the design AND the minimum sample are
    declared BEFORE the result, the count clears 200, and a SHORT count
    REFUSES rather than reporting.

    The bar must come from a PRE-DECLARED artifact -- the params file --
    and never from the receipt's own claim about itself. A receipt that
    declares its own bar has declared it after seeing the draws.
    """
    if not isinstance(params, dict) or "min_draws_per_arm_day" not in params:
        raise CaseRefused(
            "PARTIAL INPUT: the params carry no `min_draws_per_arm_day`, "
            "so there is no PRE-DECLARED bar and rule 6 cannot be checked. "
            "Reading the bar off the receipt would be reading the claim, "
            "not the declaration.")
    bar = int(params["min_draws_per_arm_day"])
    arms = _arms(artifact)
    counts, short = {}, {}
    for a in arms:
        n = ((a.get("economic") or {}).get("null_draws_summary") or {}).get("n")
        counts[a.get("arm")] = n
        if isinstance(n, int) and n < bar:
            short[a.get("arm")] = n
    drew = {k: v for k, v in counts.items() if isinstance(v, int)}
    #: DA 165, found by driving this case against the 09-03 POINT ESTIMATE
    #: before the null landed: every arm carried the STRING
    #: `NULL_NOT_DRAWN_POINT_ESTIMATE_RUN` where an int belongs, so
    #: `drew` was empty, `short` was empty, and the case reported
    #: PROPERTY_HOLDS over a run that drew NOTHING. ***Absence reading as
    #: a pass, in the case built to catch absence reading as a pass.***
    #: Case 1 refused the same artifact; this one did not, and the
    #: difference was invisible until both were run on a real file.
    if not drew:
        raise CaseRefused(
            f"PARTIAL INPUT: no arm carries an integer draw count "
            f"(found {counts!r}), so no null was drawn and there is "
            f"nothing to hold against the bar. PROPERTY_HOLDS here would "
            f"certify rule 6 on a run that made no draws.")

    # the short-count refusal, DRIVEN
    refusal = None
    if checker is not None:
        try:
            checker([0.0] * (bar - 1), params)
            refusal = "DID_NOT_REFUSE"
        except Exception as e:                                # noqa: BLE001
            refusal = type(e).__name__ + ": " + str(e)[:80]
    ok = (bar >= RULE_6_FLOOR and not short
          and (refusal is None or "DID_NOT_REFUSE" not in refusal))
    return {"case": "rule_6_declared_before_the_result",
            "verdict": PASS if ok else FAIL,
            "bar_from_the_pre_declared_params": bar,
            "bar_clears_rule_6_floor": bar >= RULE_6_FLOOR,
            "rule_6_floor": RULE_6_FLOOR,
            "draws_per_arm": counts,
            "arms_that_drew": sorted(drew),
            "arms_below_the_bar": short,
            "short_count_refusal_driven": refusal,
            "why": ("rule 6: declare the null BEFORE the result, design AND "
                    "minimum sample. A bar read off the receipt is a bar "
                    "declared after the draws; a bar below 200 satisfies "
                    "'declared' and violates the rule; and a short count "
                    "must REFUSE, because an under-sampled correct null "
                    "flatters as much as a wrong one")}


# ---------------------------------------------------------------- case 3
EXC_BLOCK = "cancel_unit_exception"
EXC_NUMERATOR = "n_cancelled_more_than_once"
EXC_DENOMINATOR = "n_reference_generations_cancelled"
FAIL_CLOSED_STATUS = "NULL_FAIL_CLOSED_CANCEL_UNIT_EXCEPTION"


def case_3_the_exception_travels(artifact):
    """(3) Does the ruled exception travel WITH ITS DENOMINATOR, and is a
    fail-closed arm stated WITH ITS REASON?

    The dangerous case is not a wrong number, it is SILENCE: an artifact
    that simply omits the one case the null cannot handle reads as an
    artifact with no exceptions. So an ABSENT block is a violation here,
    never a pass.
    """
    arms = _arms(artifact)
    per = {}
    for a in arms:
        nm = a.get("arm")
        exc = a.get(EXC_BLOCK)
        row = {"exception_block_present": isinstance(exc, dict),
               "status": a.get("status")}
        if isinstance(exc, dict):
            num, den = exc.get(EXC_NUMERATOR), exc.get(EXC_DENOMINATOR)
            row.update({
                "numerator": num, "denominator": den,
                "rate": exc.get("rate"),
                "max_on_one_generation":
                    exc.get("max_cancels_on_one_reference_generation"),
                "premise_holds": exc.get("premise_holds"),
                "has_numerator": num is not None,
                "has_denominator": den is not None,
                "numerator_without_denominator": (num is not None
                                                  and den is None),
                "rate_recomputes": (
                    None if not (isinstance(num, int) and isinstance(den, int)
                                 and den)
                    else abs(round(num / den, 8)
                             - float(exc.get("rate") or -1)) < 1e-8)})
        if a.get("status") == FAIL_CLOSED_STATUS:
            why = a.get("why_no_null")
            row["fail_closed"] = True
            row["reason_present"] = bool(why)
            row["reason_names_the_counts"] = bool(
                why and isinstance(exc, dict)
                and str(exc.get(EXC_NUMERATOR)) in str(why)
                and str(exc.get(EXC_DENOMINATOR)) in str(why))
        per[nm] = row
    violations = []
    for nm, r in per.items():
        if not r["exception_block_present"]:
            violations.append(f"{nm}: {EXC_BLOCK} ABSENT -- silence about "
                              f"the one case the null cannot handle")
            continue
        if r.get("numerator_without_denominator"):
            violations.append(f"{nm}: numerator with no denominator")
        if not r.get("has_denominator"):
            violations.append(f"{nm}: no denominator")
        if r.get("rate_recomputes") is False:
            violations.append(f"{nm}: `rate` does not recompute from its "
                              f"own numerator and denominator")
        if r.get("fail_closed") and not r.get("reason_present"):
            violations.append(f"{nm}: fail-closed with NO reason")
        if r.get("fail_closed") and not r.get("reason_names_the_counts"):
            violations.append(f"{nm}: fail-closed reason does not carry the "
                              f"counts it rests on")
    return {"case": "the_exception_travels_with_its_denominator",
            "verdict": PASS if not violations else FAIL,
            "per_arm": per, "violations": violations,
            "n_violations": len(violations),
            "why": ("a null silent about the one case it cannot handle is "
                    "the shape this programme has spent the night fixing. "
                    "The exception needs its DENOMINATOR (a bare '1' is not "
                    "a rate) and a fail-closed arm needs its REASON, or a "
                    "reader meets a missing null with nothing to read")}


# ---------------------------------------------------------------- case 4
RULED_ENDPOINT_TOKENS = ("R-801", "SETTLEMENT", "TRADES", "RESIDUAL")
PROXY_TOKENS = ("HARM_SHARE", "HARM SHARE", "HARMFUL_SHARE",
                "HARMFUL_FRACTION", "MARKOUT", "HARM_RATE")
SETTLEMENT_BLOCK = "economic_settlement"
DIAGNOSTIC_BLOCK = "economic"


def case_4_comparison_is_on_the_decision_metric(artifact):
    """(4) Is the comparison on the DECISION metric, or on a proxy?

    Rule 7: compare on the decision metric -- net value, rho =
    adverse/spread -- never on a proxy like harm share. Two ways to fail,
    and the second is the one DA 164 just found in this seat's own module:
    naming a proxy outright, OR attaching the null to the DIAGNOSTIC block
    while the ruled endpoint sits beside it untouched.
    """
    arms = _arms(artifact)
    per, violations = {}, []
    for a in arms:
        nm = a.get("arm")
        es = a.get(SETTLEMENT_BLOCK)
        ec = a.get(DIAGNOSTIC_BLOCK) or {}
        endpoint = (es or {}).get("endpoint")
        up = str(endpoint or "").upper()
        proxies = [t for t in PROXY_TOKENS if t in up]
        ruled = [t for t in RULED_ENDPOINT_TOKENS if t in up]
        null_in_ruled = isinstance(es, dict) and "null_draws_summary" in es
        null_in_diag = "null_draws_summary" in ec
        per[nm] = {"endpoint_named": endpoint,
                   "names_the_ruled_endpoint": bool(ruled),
                   "ruled_tokens_found": ruled,
                   "names_a_proxy": bool(proxies),
                   "proxy_tokens_found": proxies,
                   "null_summary_in_the_ruled_block": null_in_ruled,
                   "null_summary_in_the_diagnostic_block": null_in_diag}
        if endpoint is None:
            violations.append(f"{nm}: NO endpoint named -- absence is not a "
                              f"pass, a comparison that will not say what it "
                              f"compared cannot be checked against rule 7")
        elif proxies:
            violations.append(f"{nm}: endpoint names a PROXY {proxies}")
        elif not ruled:
            violations.append(f"{nm}: endpoint {endpoint!r} names neither "
                              f"the ruled endpoint nor a known proxy")
        if null_in_diag and not null_in_ruled:
            violations.append(
                f"{nm}: the null is attached to the DIAGNOSTIC block only, "
                f"so it is a null on D_E0 while the ruled settlement "
                f"endpoint sits beside it -- a proxy by placement rather "
                f"than by name")
    return {"case": "comparison_is_on_the_decision_metric",
            "verdict": PASS if not violations else FAIL,
            "per_arm": per, "violations": violations,
            "n_violations": len(violations),
            "ruled_tokens": list(RULED_ENDPOINT_TOKENS),
            "proxy_tokens": list(PROXY_TOKENS),
            "why": ("rule 7: the comparison is on the DECISION metric -- "
                    "net value, rho = adverse/spread -- never on a proxy "
                    "like harm share. A proxy substitutes two ways: by NAME "
                    "and by PLACEMENT, and the second is what DA 164 found "
                    "in this seat's own verdict module")}


CASES = ("matching_unit_declared_and_enforced",
         "rule_6_declared_before_the_result",
         "the_exception_travels_with_its_denominator",
         "comparison_is_on_the_decision_metric")


# ======================================================================
# THE CASES' OWN FALSIFIERS. Every control below is one that COULD fail:
# each known-bad is built so a case that merely READS fields would report
# agreement, and each partial input is one where a silent PROPERTY_HOLDS
# would be certifying an empty set.
# ======================================================================
def _arm(name="HAZARD_OVER_SKEWED_REF", *, drew=True, unit="CANCELS",
         n=500, exc=True, num=1, den=3862, rate=None, fail_closed=False,
         why=None, endpoint="R-801 SETTLEMENT P&L (trades + residual)",
         null_in_ruled=True, null_in_diag=True):
    a = {"arm": name, "status": "OK"}
    a["draw_provenance"] = ({"matched_on": unit} if drew else None) \
        if unit != "__ABSENT__" else {}
    if drew:
        a[DIAGNOSTIC_BLOCK] = {"D_E0": 1.0}
        if null_in_diag:
            a[DIAGNOSTIC_BLOCK]["null_draws_summary"] = {"n": n}
        a[SETTLEMENT_BLOCK] = {"D_E_settle": 11191.24, "endpoint": endpoint}
        if null_in_ruled:
            a[SETTLEMENT_BLOCK]["null_draws_summary"] = {"n": n}
    else:
        a[SETTLEMENT_BLOCK] = {"D_E_settle": 11191.24, "endpoint": endpoint}
    if exc:
        blk = {"matching_unit": "CANCELS",
               "max_cancels_on_one_reference_generation": 2,
               "premise_holds": not fail_closed}
        if num is not None:
            blk[EXC_NUMERATOR] = num
        if den is not None:
            blk[EXC_DENOMINATOR] = den
        blk["rate"] = (rate if rate is not None
                       else (round(num / den, 8) if num is not None and den
                             else None))
        a[EXC_BLOCK] = blk
    if fail_closed:
        a["status"] = FAIL_CLOSED_STATUS
        if why is not False:
            a["why_no_null"] = why or (
                f"{num} of {den} cancelled reference generations carry MORE "
                f"THAN ONE cancel, so the premise does not hold")
    return a


def _art(*arms):
    return {"day_run": {"per_day_sealed_artifacts": list(arms)}}


def refuses(fn, label):
    try:
        fn()
        return f"{label}: DID NOT REFUSE"
    except CaseRefused as e:
        return f"REFUSED -- {str(e)[:44]}"
    except Exception as e:                                    # noqa: BLE001
        return f"{label}: WRONG EXCEPTION {type(e).__name__}"


def selftest() -> tuple:
    checks: list = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    #: a refuser standing in for `null_draws_valued`: refuses by the ruled
    #: names, so the case can be driven with no runner import.
    def _refuser(*a, matched_on=None, arm_cancels=None,
                 control_set_path=None, **k):
        if matched_on not in NULL_UNITS:
            raise RuntimeError(f"NULL_MATCHING_UNIT_NOT_DECLARED: "
                               f"{matched_on!r}")
        if matched_on == "CANCELS" and arm_cancels is None:
            raise RuntimeError("NULL_MATCHING_UNIT_CONTRADICTED_BY_ITS_INPUTS")
        if matched_on == "CANCELS" and control_set_path is None:
            raise RuntimeError("NULL_MATCHING_UNIT_CONTRADICTED_BY_ITS_INPUTS")
        return {"ok": True}

    def _refuser_falls_back(*a, matched_on=None, **k):
        return {"ok": True}          # the ORIGINAL defect: no refusal

    def _entry_good(bk):
        null_draws_valued(bk, matched_on="CANCELS", arm_cancels=[],
                          control_set_path="p", n_draws=1, seed=0)

    def _entry_defective(bk):
        null_draws_valued(bk, n_draws=1, seed=0)

    # ---- case 1 -------------------------------------------------------
    g1 = case_1_matching_unit_declared_and_enforced(
        _art(_arm()), entry=_entry_good, refuser=_refuser)
    b1_unit = case_1_matching_unit_declared_and_enforced(
        _art(_arm(unit="DECISIONS")), entry=_entry_good, refuser=_refuser)
    b1_entry = case_1_matching_unit_declared_and_enforced(
        _art(_arm()), entry=_entry_defective, refuser=_refuser)
    b1_fall = case_1_matching_unit_declared_and_enforced(
        _art(_arm()), entry=_entry_good, refuser=_refuser_falls_back)
    ck("CASE 1 holds when the artifact NAMES CANCELS, the entry point "
       "PASSES all three arguments, and the refusal FIRES five ways -- and "
       "***it fails on each of the three independently***: a unit of "
       "DECISIONS, an entry point passing nothing (the ORIGINAL defect, "
       "read at the CALLER), and a callee that falls back instead of "
       "refusing",
       g1["verdict"] == PASS and b1_unit["verdict"] == FAIL
       and b1_entry["verdict"] == FAIL and b1_fall["verdict"] == FAIL
       and g1["refusals_that_named_the_matching_unit"] == 5
       and b1_fall["refusals_that_named_the_matching_unit"] == 0,
       f"good -> {g1['verdict']} ({g1['entry_point_sites_passing_all_three']}"
       f" site(s) pass all three, {g1['refusals_that_named_the_matching_unit']}"
       f"/5 named refusals); DECISIONS -> {b1_unit['verdict']}; entry "
       f"passing nothing -> {b1_entry['verdict']}; callee falling back -> "
       f"{b1_fall['verdict']}")
    ck("CASE 1 REFUSES a point-estimate artifact -- no arm drew, so there "
       "is no declaration to check and ***PROPERTY_HOLDS there would be "
       "certifying a run that never made a null***",
       "REFUSED" in refuses(
           lambda: case_1_matching_unit_declared_and_enforced(
               _art(_arm(drew=False)), entry=_entry_good, refuser=_refuser),
           "point estimate")
       and "REFUSED" in refuses(
           lambda: case_1_matching_unit_declared_and_enforced(
               {"day_run": {}}, entry=_entry_good), "no arms"),
       refuses(lambda: case_1_matching_unit_declared_and_enforced(
           _art(_arm(drew=False)), entry=_entry_good, refuser=_refuser),
           "point estimate"))

    # ---- case 2 -------------------------------------------------------
    P = {"min_draws_per_arm_day": 500}
    def _bar_checker(draws, params):
        if len(draws) < params["min_draws_per_arm_day"]:
            raise RuntimeError("REFUSED_TOO_FEW_DRAWS")
        return True
    def _bar_checker_soft(draws, params):
        return True                  # reports instead of refusing
    g2 = case_2_rule_6_declared_before_the_result(
        _art(_arm(n=500)), P, checker=_bar_checker)
    b2_short = case_2_rule_6_declared_before_the_result(
        _art(_arm(n=499)), P, checker=_bar_checker)
    b2_floor = case_2_rule_6_declared_before_the_result(
        _art(_arm(n=150)), {"min_draws_per_arm_day": 150},
        checker=_bar_checker)
    b2_soft = case_2_rule_6_declared_before_the_result(
        _art(_arm(n=500)), P, checker=_bar_checker_soft)
    ck("CASE 2 holds at exactly the bar and fails ONE BELOW it -- and "
       "***fails a bar of 150 that is honestly declared and still violates "
       "rule 6's floor of 200***, and fails a checker that REPORTS a short "
       "count instead of refusing",
       g2["verdict"] == PASS and b2_short["verdict"] == FAIL
       and b2_floor["verdict"] == FAIL
       and b2_floor["bar_clears_rule_6_floor"] is False
       and b2_soft["verdict"] == FAIL,
       f"n=500 at bar 500 -> {g2['verdict']}; n=499 -> "
       f"{b2_short['verdict']} {b2_short['arms_below_the_bar']}; declared "
       f"bar 150 -> {b2_floor['verdict']} (clears floor "
       f"{b2_floor['bar_clears_rule_6_floor']}); non-refusing checker -> "
       f"{b2_soft['verdict']} ({b2_soft['short_count_refusal_driven']})")
    ck("CASE 2 REFUSES params carrying no bar -- ***reading the bar off the "
       "receipt would be reading a number declared AFTER the draws***, "
       "which is the thing rule 6 forbids",
       "REFUSED" in refuses(
           lambda: case_2_rule_6_declared_before_the_result(
               _art(_arm()), {}, checker=_bar_checker), "no bar"),
       refuses(lambda: case_2_rule_6_declared_before_the_result(
           _art(_arm()), {}, checker=_bar_checker), "no bar in params"))

    ck("CASE 2 REFUSES AN ARTIFACT THAT DREW NOTHING -- ***found by "
       "driving this case against the 09-03 POINT ESTIMATE before the "
       "null landed: every arm carried the STRING "
       "`NULL_NOT_DRAWN_POINT_ESTIMATE_RUN` where an int belongs, so the "
       "case reported PROPERTY_HOLDS over a run with no draws.*** Case 1 "
       "refused that same artifact and this one did not -- absence "
       "reading as a pass, inside the case built to catch it",
       "REFUSED" in refuses(
           lambda: case_2_rule_6_declared_before_the_result(
               _art(_arm(n="NULL_NOT_DRAWN_POINT_ESTIMATE_RUN")), P,
               checker=_bar_checker), "no draws")
       and case_2_rule_6_declared_before_the_result(
           _art(_arm(n=500)), P, checker=_bar_checker)["verdict"] == PASS,
       refuses(lambda: case_2_rule_6_declared_before_the_result(
           _art(_arm(n="NULL_NOT_DRAWN_POINT_ESTIMATE_RUN")), P,
           checker=_bar_checker), "a run that drew nothing"))

    # ---- case 3 -------------------------------------------------------
    g3 = case_3_the_exception_travels(_art(_arm()))
    b3_absent = case_3_the_exception_travels(_art(_arm(exc=False)))
    b3_noden = case_3_the_exception_travels(_art(_arm(den=None)))
    b3_rate = case_3_the_exception_travels(_art(_arm(rate=0.5)))
    b3_why = case_3_the_exception_travels(
        _art(_arm(fail_closed=True, why=False)))
    b3_thin = case_3_the_exception_travels(
        _art(_arm(fail_closed=True, why="the premise does not hold")))
    g3_fc = case_3_the_exception_travels(_art(_arm(fail_closed=True)))
    ck("CASE 3 holds on an exception carrying BOTH legs and a fail-closed "
       "arm whose reason CARRIES ITS COUNTS -- and fails four ways: the "
       "block ABSENT (***the dangerous one: silence reads as 'no "
       "exceptions'***), a numerator with no denominator, a `rate` that "
       "does not recompute, and a fail-closed arm with no reason",
       g3["verdict"] == PASS and g3_fc["verdict"] == PASS
       and b3_absent["verdict"] == FAIL and b3_noden["verdict"] == FAIL
       and b3_rate["verdict"] == FAIL and b3_why["verdict"] == FAIL,
       f"1 of 3862 with rate -> {g3['verdict']}; fail-closed with counts in "
       f"the reason -> {g3_fc['verdict']}; ABSENT -> "
       f"{b3_absent['verdict']} ({b3_absent['violations'][0][:58]}...); no "
       f"denominator -> {b3_noden['verdict']}; wrong rate -> "
       f"{b3_rate['verdict']}; no reason -> {b3_why['verdict']}")
    ck("CASE 3 ALSO fails a fail-closed reason that is TRUE BUT EMPTY -- "
       "***prose without the counts it rests on is exactly the caveat-in-"
       "prose the ruling replaced***",
       b3_thin["verdict"] == FAIL
       and any("does not carry the counts" in v
               for v in b3_thin["violations"]),
       f"reason 'the premise does not hold' -> {b3_thin['verdict']}: "
       f"{[v for v in b3_thin['violations'] if 'counts' in v]}")

    # ---- case 4 -------------------------------------------------------
    g4 = case_4_comparison_is_on_the_decision_metric(_art(_arm()))
    b4_proxy = case_4_comparison_is_on_the_decision_metric(
        _art(_arm(endpoint="harm share of cancelled generations")))
    b4_mark = case_4_comparison_is_on_the_decision_metric(
        _art(_arm(endpoint="the 5-second markout D_E0")))
    b4_none = case_4_comparison_is_on_the_decision_metric(
        _art(_arm(endpoint=None)))
    b4_place = case_4_comparison_is_on_the_decision_metric(
        _art(_arm(null_in_ruled=False)))
    ck("CASE 4 holds on the ruled endpoint and catches a proxy BY NAME -- "
       "harm share and the 5-second markout both -- and catches an "
       "endpoint named NOTHING, because ***a comparison that will not say "
       "what it compared cannot be checked at all and absence is not a "
       "pass***",
       g4["verdict"] == PASS and b4_proxy["verdict"] == FAIL
       and b4_mark["verdict"] == FAIL and b4_none["verdict"] == FAIL
       and b4_proxy["per_arm"]["HAZARD_OVER_SKEWED_REF"]["proxy_tokens_found"],
       f"R-801 endpoint -> {g4['verdict']}; harm share -> "
       f"{b4_proxy['verdict']} "
       f"{b4_proxy['per_arm']['HAZARD_OVER_SKEWED_REF']['proxy_tokens_found']}"
       f"; markout -> {b4_mark['verdict']}; unnamed -> {b4_none['verdict']}")
    ck("CASE 4 ALSO catches a proxy BY PLACEMENT: the null attached to the "
       "DIAGNOSTIC block while the ruled settlement endpoint sits beside "
       "it untouched. ***That is the sibling-block defect DA 164 found in "
       "this seat's OWN verdict module, and the endpoint STRING is "
       "perfectly correct in this case*** -- so a name-only check passes "
       "it",
       b4_place["verdict"] == FAIL
       and any("DIAGNOSTIC block only" in v
               for v in b4_place["violations"])
       and b4_place["per_arm"]["HAZARD_OVER_SKEWED_REF"][
           "names_the_ruled_endpoint"] is True,
       f"null in the diagnostic only, endpoint string still correct "
       f"(names_the_ruled_endpoint "
       f"{b4_place['per_arm']['HAZARD_OVER_SKEWED_REF']['names_the_ruled_endpoint']}"
       f") -> {b4_place['verdict']}")

    fails = sum(1 for c in checks if not c["passed"])
    for c in checks:
        print(("ok   " if c["passed"] else "FAIL ") + c["check"])
        print("       " + c["detail"])
    print(f"\n{'SELFTEST OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {fails} failure(s)")
    return checks, fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--artifact")
    ap.add_argument("--params")
    a = ap.parse_args(argv)
    if a.selftest:
        return 1 if selftest()[1] else 0
    if not a.artifact:
        ap.error("--selftest, or --artifact <path>")
    art = json.loads(Path(a.artifact).read_text())
    if a.params:
        params = json.loads(Path(a.params).read_text())
    else:
        d = Path(__file__).resolve().parent / "declarations"
        ps = sorted(d.glob("de_multiday_gate1_params_v*.json"),
                    key=lambda p: int(re.search(r"_v(\d+)\.json",
                                                p.name).group(1)))
        params = json.loads(ps[-1].read_text()) if ps else {}
    out = {}
    for nm, fn, args in (
            ("case_1", case_1_matching_unit_declared_and_enforced, (art,)),
            ("case_2", case_2_rule_6_declared_before_the_result,
             (art, params)),
            ("case_3", case_3_the_exception_travels, (art,)),
            ("case_4", case_4_comparison_is_on_the_decision_metric, (art,))):
        try:
            out[nm] = fn(*args)
        except CaseRefused as e:
            out[nm] = {"verdict": REFUSED, "why": str(e)}
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
