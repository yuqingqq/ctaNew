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
RULE_6_FLOOR = 200


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
