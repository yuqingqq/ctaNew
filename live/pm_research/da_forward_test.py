#!/usr/bin/env python3
"""THE FORWARD TEST, EVALUATED AS PREDICATES.

`declarations/da_forward_test_declaration_v1.json` fixes the forward test
before a single untouched day is touched. This module evaluates every clause
as a predicate, because a declaration that is only prose beside a table is the
defect this programme keeps paying for -- and because DA 192 found ELEVEN of
its own predicates testing spelling rather than content, so the falsifiers
here are WEAKENED CLAUSES, not deleted keys.

The power expectation is RECOMPUTED, not read: a projection that is typed is
a claim, and this one has to be a measurement or it cannot be held against
the result later.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from da_asymmetry_null import (  # noqa: E402
    HOLDS, FAILS, UNCHECKED, unconditional, decidable, holm, _verdict)
from p003_rule6_floor import FLOOR as RULE_6_FLOOR  # noqa: E402

DECL = HERE / "declarations" / "da_forward_test_declaration_v1.json"
_N = NormalDist()


def load(path=None) -> dict:
    return json.loads(Path(path or DECL).read_text())


def z_from_p2(p: float) -> float:
    return _N.inv_cdf(1 - p / 2)


def project(z0: float, n0: int, n: int) -> float:
    """sqrt-N scaling of a z statistic. The declaration's own method."""
    return z0 * math.sqrt(n / n0)


# ----------------------------------------------------------------- items

def item_1_population(d: dict) -> dict:
    p = d.get("POPULATION") or {}
    fq = p.get("a_failed_quality_day") or {}
    props = {
        "the_population_is_a_RULE_not_a_list": p.get("rule_not_a_list") is True,
        "the_rule_is_anchored_to_the_FREEZE_COMMIT":
            "freeze commit" in str(p.get("THE_RULE", "")).lower(),
        "an_absent_freeze_REFUSES": (p.get("freeze_commit") or {}).get(
            "status_if_absent") == "FORWARD_TEST_NO_FREEZE_COMMIT",
        "a_failed_day_is_a_NAMED_STATUS":
            fq.get("status_name") == "DAY_EXCLUDED_ON_QUALITY",
        "the_substitution_is_WITHOUT_INSPECTING_OUTCOMES":
            "WITHOUT INSPECTING" in str(fq.get("and_then", "")).upper(),
        "a_quality_decision_that_saw_an_outcome_REFUSES":
            fq.get("refusal_if_violated") == "QUALITY_DECISION_SAW_AN_OUTCOME",
        "N_is_fixed_before_any_data": p.get("N_is_fixed_before_any_data") is True,
        "completeness_is_read_from_METADATA_only":
            "metadata" in str(p.get("complete", "")).lower(),
    }
    return _verdict(props, [], item="population", N=p.get("N"))


def item_2_both_comparisons(d: dict) -> dict:
    b = d.get("BOTH_COMPARISONS_ARE_REQUIRED") or {}
    a, c = b.get("comparison_A") or {}, b.get("comparison_B") or {}
    props = {
        "two_comparisons_are_named": bool(a.get("name")) and bool(c.get("name")),
        "the_CONJUNCTION_is_the_test": "AND" in str(
            b.get("the_conjunction_is_the_test", "")),
        "one_comparison_alone_is_NOT_a_pass": "DOES NOT ADVANCE" in str(
            b.get("the_conjunction_is_the_test", "")).upper(),
        "claiming_a_pass_on_one_REFUSES":
            b.get("refusal_if_reported_as_a_pass_on_one")
            == "ADVANCEMENT_CLAIMED_ON_ONE_COMPARISON",
        "comparison_A_is_at_the_DAY_CLUSTER_level":
            "UTC-DAY CLUSTER" in str(a.get("requirement", "")).upper(),
        "comparison_B_matches_on_all_three_keys":
            tuple(c.get("matched_on") or ()) == (
                "distinct_reference_generation_count", "side", "hour"),
        "the_rule_clause_is_unconditional":
            unconditional(b.get("rule"))["unconditional"],
    }
    return _verdict(props, [], item="both_comparisons")


def item_3_replay_assumption(d: dict) -> dict:
    r = d.get("REPLAY_ASSUMPTION") or {}
    s = r.get("A_SIGN_REVERSAL_BLOCKS_PROMOTION") or {}
    props = {
        "the_primary_is_FIXED_and_named":
            r.get("PRIMARY_FIXED_BEFORE_ANY_DATA") == "REFERENCE_FILLS",
        "the_robustness_leg_is_named_and_labelled":
            r.get("ROBUSTNESS_LEG_LABELLED") == "NO_FILLS_UNTIL_NEXT_GENERATION",
        "the_robustness_leg_is_ALWAYS_reported":
            r.get("the_robustness_leg_is_always_reported") is True,
        "substitution_is_forbidden":
            r.get("it_may_never_be_substituted_for_the_primary") is True,
        "a_sign_reversal_BLOCKS_rather_than_invites_a_choice":
            "BLOCKED" in str(s.get("consequence", "")).upper(),
        "the_blocking_status_is_named":
            s.get("status") == "SIGN_REVERSES_UNDER_THE_ROBUSTNESS_LEG",
        "the_sign_predicate_is_DECIDABLE": bool(s.get("predicate"))
            and "sign(" in str(s.get("predicate")),
        "the_blocking_clause_is_unconditional": unconditional(
            s.get("why_it_blocks_rather_than_invites_a_choice"))["unconditional"],
    }
    return _verdict(props, [], item="replay_assumption")


def item_4_combination_rule(d: dict) -> dict:
    c = d.get("THE_COMBINATION_RULE") or {}
    m = c.get("mixed_cases_enumerated") or {}
    minority = m.get("an_arm_advances_while_a_MINORITY_of_days_are_negative") or {}
    pool = c.get("pooling") or {}
    space = ("both_arms_advance", "exactly_one_advances", "neither_advances",
             "both_advance_with_OPPOSITE_SIGNS")
    props = {
        "declared_before_any_day_is_touched":
            c.get("declared_before_any_day_is_touched") is True,
        "the_two_arm_outcome_space_is_COVERED": all(
            k in m and str(m[k]).strip() for k in space),
        "opposite_signs_OVERRIDE_an_advancement":
            "OVERRIDES" in str(m.get("both_advance_with_OPPOSITE_SIGNS", "")).upper(),
        "one_arm_advancing_is_NOT_a_programme_pass":
            "not a programme-level pass" in str(m.get("exactly_one_advances", "")),
        "the_minority_days_predicate_is_DECIDABLE":
            decidable(minority.get("predicate"))["decidable"],
        "the_minority_status_TRAVELS_unconditionally":
            unconditional(minority.get("rule"))["unconditional"],
        "counting_days_is_explicitly_NOT_a_verdict_input":
            bool(m.get("counting_days_is_not_a_verdict_input")),
        "the_weights_are_fixed_by_the_BASELINE":
            "BASELINE" in str(pool.get("weights", "")).upper(),
        "COMPUTED_the_floor_matches_the_AUTHORITY":
            pool.get("rule_6_floor") == RULE_6_FLOOR,
        "COMPUTED_n_draws_clears_the_floor":
            isinstance(pool.get("n_draws"), int)
            and pool["n_draws"] >= RULE_6_FLOOR,
        "every_forbidden_entry_is_unconditional": bool(c.get("forbidden"))
            and all(unconditional(x)["unconditional"] for x in c["forbidden"]),
    }
    return _verdict(props, [], item="combination_rule")


def item_5_multiplicity(d: dict) -> dict:
    m = d.get("THE_MULTIPLICITY_ACTUALLY_CONTROLLED") or {}
    try:
        holm({"a": 0.01, "b": 0.02, "c": 0.03}, 2)
        refused = False
    except ValueError:
        refused = True
    props = {
        "the_controlled_family_is_TWO": m.get("controlled_family_size") == 2,
        # DA 210: this read the word SEQUENTIAL out of the VALUE while the word
        # lived in the KEY -- rule 16, in the checker written to catch it. It
        # now tests the STRUCTURE: both bars stated, and a stop condition named.
        "the_TWO_BARS_are_both_stated": all(
            x in str(m.get("holm_is_SEQUENTIAL_and_this_matters", ""))
            for x in ("0.025", "0.05")),
        "the_first_step_failing_STOPS_holm":
            "STOPS" in str(m.get("holm_is_SEQUENTIAL_and_this_matters", "")).upper(),
        "per_day_p_values_are_DESCRIPTIVE": "DESCRIPTIVE" in str(
            m.get("per_day_and_per_cell_p_values_are", "")).upper(),
        "a_cell_level_verdict_REFUSES":
            m.get("refusal") == "CELL_LEVEL_VERDICT_REPORTED",
        "THE_SECOND_ATTEMPT_IS_RECORDED_AS_NOT_IN_THE_CORRECTION": bool(
            m.get("AND_THE_SECOND_ATTEMPT_IS_NOT_IN_THIS_NUMBER")),
        "the_family_closure_is_unconditional":
            unconditional(m.get("family_may_not_be_reopened"))["unconditional"],
        "EXECUTED_holm_refuses_a_family_smaller_than_the_tests": refused,
    }
    return _verdict(props, [], item="multiplicity")


def item_6_unit(d: dict) -> dict:
    u = d.get("THE_UNIT") or {}
    props = {
        "the_cluster_unit_is_the_UTC_day": "UTC day" in str(u.get("cluster_unit", "")),
        "COMPUTED_G_MEETS_the_rule_8_bar":
            isinstance(u.get("G"), int) and isinstance(u.get("rule_8_bar"), int)
            and u["G"] >= u["rule_8_bar"],
        "intervals_ARE_claimable_and_that_is_stated":
            u.get("intervals_claimable") is True,
        "the_cell_is_NOT_a_unit": bool(u.get("what_is_still_not_a_unit")),
        "counting_cells_as_independent_REFUSES":
            u.get("refusal") == "CELLS_COUNTED_AS_INDEPENDENT",
    }
    return _verdict(props, [], item="unit", G=u.get("G"))


def item_7_power(d: dict) -> dict:
    """The projection RECOMPUTED. A typed projection is a claim, not a
    measurement, and this one must be held against the result later."""
    pw = d.get("PRE_REGISTERED_POWER_EXPECTATION") or {}
    obs = pw.get("observed_z_at_3_days") or {}
    bars = pw.get("holm_bars") or {}
    proj = pw.get("projection") or {}
    need = pw.get("days_to_clear") or {}
    screen = d.get("THESE_ARMS_FAILED_THEIR_DEVELOPMENT_SCREEN") or {}
    prim = screen.get("on_the_PRIMARY_pool") or {}
    ok_z = {}
    for arm, key in (("CONDVALUE_X_SKEW", "CONDVALUE_X_SKEW_p_two_sided"),
                     ("HAZARD_OVER_SKEWED_REF", "HAZARD_OVER_SKEWED_REF_p_two_sided")):
        p = prim.get(key)
        ok_z[arm] = (p is not None
                     and abs(z_from_p2(p) - obs.get(arm, -1)) < 5e-4)
    ok_proj = True
    for nkey, cells in proj.items():
        n = int(str(nkey).split("=")[1])
        for arm, v in cells.items():
            if abs(project(obs[arm], 3, n) - v) > 0.01:
                ok_proj = False
    zb1 = bars.get("step_1_smaller_p_needs_z")
    zb2 = bars.get("step_2_larger_p_needs_z")
    props = {
        "RECOMPUTED_the_observed_z_matches_the_artifacts_p": all(ok_z.values()),
        "RECOMPUTED_the_holm_bars": (
            zb1 is not None and abs(zb1 - z_from_p2(0.025)) < 5e-4
            and zb2 is not None and abs(zb2 - z_from_p2(0.05)) < 5e-4),
        "RECOMPUTED_every_projection_cell": ok_proj,
        "RECOMPUTED_five_days_clears_NEITHER_arm": (
            project(obs["CONDVALUE_X_SKEW"], 3, 5) < zb1
            and project(obs["HAZARD_OVER_SKEWED_REF"], 3, 5) < zb1),
        "RECOMPUTED_days_to_clear_CONDVALUE": need.get(
            "CONDVALUE_X_SKEW_step_1") == math.ceil(
                3 * (zb1 / obs["CONDVALUE_X_SKEW"]) ** 2),
        "RECOMPUTED_days_to_clear_HAZARD_given_CONDVALUE": need.get(
            "HAZARD_OVER_SKEWED_REF_step_2_given_CONDVALUE_clears") == math.ceil(
                3 * (zb2 / obs["HAZARD_OVER_SKEWED_REF"]) ** 2),
        "the_declared_expectation_says_FIVE_IS_MORE_LIKELY_INCONCLUSIVE":
            "MORE LIKELY TO RETURN INCONCLUSIVE" in str(
                pw.get("THE_DECLARED_EXPECTATION", "")).upper(),
        "the_upper_bound_caveat_is_stated_with_its_evidence":
            "SIGN FLIP" in str(pw.get("THIS_IS_AN_UPPER_BOUND_ON_POWER", "")).upper(),
    }
    return _verdict(props, [], item="power_expectation",
                    recomputed_z={k: round(project(v, 3, 5), 4)
                                  for k, v in obs.items()})


def item_8_stopping_rule(d: dict) -> dict:
    s = d.get("STOPPING_RULE") or {}
    inc = s.get("AN_INCONCLUSIVE_RESULT_EXHAUSTS_THESE_ARMS_UNDER_THIS_DECLARATION") or {}
    route = s.get("THE_ONLY_LEGITIMATE_ROUTE_TO_MORE_DAYS") or {}
    props = {
        "there_is_ONE_LOOK": s.get("N_is_5_and_there_is_ONE_LOOK") is True,
        "no_interim_analysis": bool(s.get("no_interim_analysis")),
        "an_INCONCLUSIVE_result_EXHAUSTS_the_arms": inc.get("value") is True,
        "the_inconclusive_status_is_named":
            inc.get("status") == "INCONCLUSIVE_UNDERPOWERED_AS_PREDICTED",
        "OPTIONAL_STOPPING_is_named_as_the_reason":
            "OPTIONAL STOPPING" in str(inc.get("means", "")).upper(),
        "the_extension_route_requires_a_PRIOR_amendment":
            "BEFORE THE FIRST FORWARD DAY IS TOUCHED" in str(route.get("route", "")).upper(),
        "the_route_is_declared_UNAVAILABLE_LATER": bool(route.get("not_available_later")),
        "the_exhaustion_clause_is_unconditional":
            unconditional(inc.get("means"))["unconditional"],
        "the_live_choice_is_surfaced_with_its_arithmetic": bool(
            route.get("AND_THE_CHOICE_IS_LIVE_AT_THIS_MOMENT_ONLY")),
    }
    return _verdict(props, [], item="stopping_rule")


def item_9_second_attempt(d: dict) -> dict:
    s = d.get("THESE_ARMS_FAILED_THEIR_DEVELOPMENT_SCREEN") or {}
    props = {
        "the_failed_screen_is_recorded_FIRST": bool(s.get("why_this_is_first")),
        "the_step_2_verdict_is_quoted": s.get("step_2_verdict")
            == "NO_SETTLEMENT_SKILL_OVER_MATCHED_RANDOM",
        "THE_MULTIPLICITY_IS_NAMED_AS_A_SECOND_ATTEMPT":
            "SECOND ATTEMPT" in str(s.get("THE_MULTIPLICITY_IS_NOT_TWO_ARMS", "")).upper(),
        "the_companion_pool_trap_is_named": "UNCORRECTED" in str(
            (s.get("on_the_COMPANION_pool") or {}).get(
                "READ_THIS_BEFORE_QUOTING_IT", "")).upper(),
        "what_it_does_NOT_mean_is_stated": bool(s.get("what_it_does_NOT_mean")),
    }
    return _verdict(props, [], item="second_attempt")


def item_10_09_07_companion(d: dict) -> dict:
    c = d.get("THE_09_07_COMPANION") or {}
    dis = c.get("IF_THE_TWO_DISAGREE") or {}
    est = c.get("clean_on_the_ESTIMAND") or {}
    diag = c.get("consumed_on_the_DIAGNOSTIC") or {}
    props = {
        "the_two_endpoints_carry_DIFFERENT_statuses":
            est.get("status") != diag.get("status")
            and bool(est.get("status")) and bool(diag.get("status")),
        "the_estimand_is_NEVER_COMPUTED": est.get("status") == "NEVER COMPUTED",
        "the_diagnostic_is_COMPUTED_AND_SEEN":
            diag.get("status") == "COMPUTED AND SEEN",
        "the_NON_INDEPENDENCE_of_the_two_endpoints_is_stated":
            "CORRELATED" in str(c.get("AND_THE_TWO_ARE_NOT_INDEPENDENT", "")).upper(),
        "it_is_a_COMPANION_and_never_merged":
            c.get("reported_ALONGSIDE_and_never_merged") is True,
        "the_result_is_reported_BOTH_WAYS": bool(c.get("the_result_is_reported_BOTH_WAYS")),
        "a_disagreement_promotes_NEITHER":
            "NEITHER" in str(dis.get("rule", "")).upper(),
        "the_disagreement_has_a_NAME":
            dis.get("status") == "THE_09_07_COMPANION_CHANGES_THE_VERDICT",
        "cherry_picking_is_forbidden_unconditionally":
            unconditional(dis.get("forbidden"))["unconditional"],
        "the_COST_of_waiting_is_stated_as_a_trade":
            bool(c.get("the_cost_of_this_choice_stated")),
    }
    return _verdict(props, [], item="09_07_companion")


def item_11_rule_11_position(d: dict) -> dict:
    r = d.get("RULE_11_POSITION") or {}
    pop = (d.get("POPULATION") or {}).get("expected_to_resolve_to") or {}
    split = r.get("ENDPOINT_SPLIT") or {}
    props = {
        "the_PRISTINE_set_is_enumerated": len(
            r.get("PRISTINE_neither_endpoint_ever_computed_or_seen") or []) == 5,
        "COMPUTED_the_pristine_set_EQUALS_the_expected_population": set(
            r.get("PRISTINE_neither_endpoint_ever_computed_or_seen") or []) == set(
            pop.get("days") or []),
        "the_ENDPOINT_SPLIT_day_is_named_with_BOTH_endpoints": bool(split) and all(
            set(v) >= {"settlement_R801", "markout_5s_D_E0"} for v in split.values()),
        "COMPUTED_the_split_day_is_NOT_in_the_pristine_set": not (
            set(split) & set(r.get("PRISTINE_neither_endpoint_ever_computed_or_seen") or [])),
        "the_CONSUMED_development_days_are_enumerated":
            len(r.get("CONSUMED_development_data") or []) == 4,
        "COMPUTED_no_day_appears_in_two_categories": len(
            set(r.get("PRISTINE_neither_endpoint_ever_computed_or_seen") or [])
            & set(r.get("CONSUMED_development_data") or [])) == 0,
        "the_NO_SELECTION_claim_cites_its_evidence":
            "2026-08-24" in str(r.get("NO_SELECTION_WAS_MADE_ON_ANY_OF_THEM", "")),
        "the_no_selection_clause_is_unconditional": unconditional(
            r.get("NO_SELECTION_WAS_MADE_ON_ANY_OF_THEM"))["unconditional"],
    }
    return _verdict(props, [], item="rule_11_position")


ITEMS = (item_1_population, item_2_both_comparisons, item_3_replay_assumption,
         item_4_combination_rule, item_5_multiplicity, item_6_unit,
         item_7_power, item_8_stopping_rule, item_9_second_attempt,
         item_10_09_07_companion, item_11_rule_11_position)


def evaluate(d: dict | None = None) -> dict:
    d = d if d is not None else load()
    out = [f(d) for f in ITEMS]
    unchecked = [c for r in out for c in (r.get("unchecked_clauses") or [])]
    return {"protocol": d.get("protocol"), "status": d.get("STATUS"),
            "items": out, "n_items": len(out),
            "n_property_checks_driven": sum(
                r.get("n_properties_driven", 0) for r in out),
            "n_items_FAILING": sum(1 for r in out if r["verdict"] == FAILS),
            "unchecked_clauses": unchecked,
            "no_item_fails": all(r["verdict"] != FAILS for r in out)}


#: WEAKENED clauses -- semantically weaker, vocabulary intact. Each MUST FAIL.
WEAKENED = [
 ("item_1_population", "the population becomes a NAMED LIST",
  ["POPULATION", "rule_not_a_list"], False),
 ("item_1_population", "a failed day may be substituted after a look",
  ["POPULATION", "a_failed_quality_day", "and_then"],
  "the next chronological untouched day is added, after checking it looks usable"),
 ("item_2_both_comparisons", "the conjunction becomes a disjunction",
  ["BOTH_COMPARISONS_ARE_REQUIRED", "the_conjunction_is_the_test"],
  "A OR B. An arm that clears either is promoted."),
 ("item_2_both_comparisons", "a match key is dropped",
  ["BOTH_COMPARISONS_ARE_REQUIRED", "comparison_B", "matched_on"],
  ["distinct_reference_generation_count", "side"]),
 ("item_3_replay_assumption", "a sign reversal stops blocking",
  ["REPLAY_ASSUMPTION", "A_SIGN_REVERSAL_BLOCKS_PROMOTION", "consequence"],
  "the reversal is reported alongside the promotion"),
 ("item_3_replay_assumption", "the blocking clause gains an escape",
  ["REPLAY_ASSUMPTION", "A_SIGN_REVERSAL_BLOCKS_PROMOTION",
   "why_it_blocks_rather_than_invites_a_choice"],
  "a sign that depends on the fill assumption blocks promotion, unless the "
  "primary leg's margin is comfortable"),
 ("item_4_combination_rule", "a mixed case is dropped",
  ["THE_COMBINATION_RULE", "mixed_cases_enumerated",
   "both_advance_with_OPPOSITE_SIGNS"], ""),
 ("item_4_combination_rule", "the draw count drops below the authority's floor",
  ["THE_COMBINATION_RULE", "pooling", "n_draws"], 100),
 ("item_4_combination_rule", "the floor stops matching the authority",
  ["THE_COMBINATION_RULE", "pooling", "rule_6_floor"], 150),
 ("item_5_multiplicity", "the controlled family stops matching the arms",
  ["THE_MULTIPLICITY_ACTUALLY_CONTROLLED", "controlled_family_size"], 10),
 ("item_5_multiplicity", "the second-attempt record is removed",
  ["THE_MULTIPLICITY_ACTUALLY_CONTROLLED",
   "AND_THE_SECOND_ATTEMPT_IS_NOT_IN_THIS_NUMBER"], ""),
 ("item_6_unit", "G is claimed below the bar while intervals stay claimable",
  ["THE_UNIT", "G"], 3),
 ("item_7_power", "a projection cell is fudged upward",
  ["PRE_REGISTERED_POWER_EXPECTATION", "projection", "N=5"],
  {"CONDVALUE_X_SKEW": 2.30, "HAZARD_OVER_SKEWED_REF": 2.25}),
 ("item_7_power", "the declared expectation becomes optimistic",
  ["PRE_REGISTERED_POWER_EXPECTATION", "THE_DECLARED_EXPECTATION"],
  "FIVE DAYS IS EXPECTED TO CLEAR for CONDVALUE."),
 ("item_7_power", "the days-to-clear figure is typed rather than computed",
  ["PRE_REGISTERED_POWER_EXPECTATION", "days_to_clear",
   "CONDVALUE_X_SKEW_step_1"], 8),
 ("item_8_stopping_rule", "an inconclusive result licenses more days",
  ["STOPPING_RULE",
   "AN_INCONCLUSIVE_RESULT_EXHAUSTS_THESE_ARMS_UNDER_THIS_DECLARATION",
   "value"], False),
 ("item_8_stopping_rule", "the exhaustion clause gains an escape",
  ["STOPPING_RULE",
   "AN_INCONCLUSIVE_RESULT_EXHAUSTS_THESE_ARMS_UNDER_THIS_DECLARATION",
   "means"],
  "the arms are not promoted and no further day may be spent, unless the "
  "margin is close enough to justify one more"),
 ("item_8_stopping_rule", "the extension no longer needs a PRIOR amendment",
  ["STOPPING_RULE", "THE_ONLY_LEGITIMATE_ROUTE_TO_MORE_DAYS", "route"],
  "an amendment fixing a larger N, committed when the need becomes clear"),
 ("item_10_09_07_companion", "09-07 is merged into the primary",
  ["THE_09_07_COMPANION", "reported_ALONGSIDE_and_never_merged"], False),
 ("item_10_09_07_companion", "a disagreement promotes the larger pool",
  ["THE_09_07_COMPANION", "IF_THE_TWO_DISAGREE", "rule"],
  "the pool including 09-07 is promoted, being the larger sample"),
 ("item_10_09_07_companion", "cherry-picking gains an escape",
  ["THE_09_07_COMPANION", "IF_THE_TWO_DISAGREE", "forbidden"],
  "quoting whichever pool gives the better verdict, unless the other is thin"),
 ("item_10_09_07_companion", "the endpoint split is flattened to one status",
  ["THE_09_07_COMPANION", "consumed_on_the_DIAGNOSTIC", "status"],
  "NEVER COMPUTED"),
 ("item_11_rule_11_position", "a consumed day is listed as pristine",
  ["RULE_11_POSITION", "PRISTINE_neither_endpoint_ever_computed_or_seen"],
  ["2026-09-06", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"]),
 ("item_11_rule_11_position", "the no-selection claim gains an escape",
  ["RULE_11_POSITION", "NO_SELECTION_WAS_MADE_ON_ANY_OF_THEM"],
  "no threshold or parameter was selected using these days, except where the "
  "operator judged a day representative"),
 ("item_9_second_attempt", "the second attempt stops being named",
  ["THESE_ARMS_FAILED_THEIR_DEVELOPMENT_SCREEN",
   "THE_MULTIPLICITY_IS_NOT_TWO_ARMS"],
  "the multiplicity is the two arms tested here"),
]

_C = {"n": 0, "bad": 0}


def _ok(cond, label):
    _C["n"] += 1
    if not cond:
        _C["bad"] += 1
        print(f"  FAIL {label}")
    return cond


def _break(d, path, value):
    import copy
    c = copy.deepcopy(d)
    o = c
    for k in path[:-1]:
        o = o[k]
    o[path[-1]] = value
    return c


def selftest(quiet: bool = False) -> int:
    D = load()
    FN = {f.__name__: f for f in ITEMS}
    r = evaluate(D)
    _ok(r["n_items_FAILING"] == 0,
        f"the declaration fails no property "
        f"({r['n_property_checks_driven']} driven across {r['n_items']} items); "
        f"failures: {[i['item'] for i in r['items'] if i['verdict']==FAILS]}")
    _ok(D["STATUS"].startswith("DECLARED-BEFORE"),
        "STATUS says it was declared BEFORE any untouched day was touched")

    # the power projection is a MEASUREMENT: recompute it independently here
    obs = D["PRE_REGISTERED_POWER_EXPECTATION"]["observed_z_at_3_days"]
    z5c = project(obs["CONDVALUE_X_SKEW"], 3, 5)
    z5h = project(obs["HAZARD_OVER_SKEWED_REF"], 3, 5)
    bar = z_from_p2(0.025)
    _ok(z5c < bar and z5h < bar,
        f"RECOMPUTED: at N=5 CONDVALUE projects to z={z5c:.4f} and HAZARD to "
        f"z={z5h:.4f}, both BELOW the Holm step-1 bar of {bar:.4f} -- FIVE "
        f"DAYS CLEARS NEITHER ARM, declared before the data")
    _ok(math.ceil(3 * (bar / obs["CONDVALUE_X_SKEW"]) ** 2) == 6,
        "RECOMPUTED: CONDVALUE needs SIX days, not the eight the dispatch "
        "said -- the correction is computed, not asserted")

    for item, what, path, val in WEAKENED:
        v = FN[item](_break(D, path, val))["verdict"]
        _ok(v == FAILS,
            f"WEAKENED -- {item}: {what} -> {v} (must be {FAILS})")

    if not quiet:
        print(f"[da_forward_test] {_C['n'] - _C['bad']}/{_C['n']} checks, "
              f"{_C['bad']} failures | {r['n_property_checks_driven']} properties "
              f"driven, {len(WEAKENED)} weakened clauses each REFUSED | "
              f"N={D['POPULATION']['N']}, ONE LOOK, inconclusive EXHAUSTS")
    return 1 if _C["bad"] else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(json.dumps(evaluate(), indent=1, default=str))


# ==================================================================
# DECLARATION v2 (DA 211) -- the USER's ruling, folded in before any
# day is touched. v1 is UNEDITED (rule 13).
# ==================================================================

DECL_V2 = HERE / "declarations" / "da_forward_test_declaration_v2.json"


def load_v2(path=None) -> dict:
    return json.loads(Path(path or DECL_V2).read_text())


def item_v2_population(d: dict) -> dict:
    p = d.get("POPULATION") or {}
    w = p.get("THE_POST_FREEZE_CLAUSE_IS_WITHDRAWN") or {}
    m = w.get("MEASURED") or {}
    fq = p.get("a_failed_quality_day") or {}
    after = set(m.get("days_starting_strictly_after_the_freeze") or [])
    before = set(m.get("days_starting_BEFORE_the_freeze") or [])
    props = {
        "COMPUTED_N_equals_the_day_list": p.get("N") == len(p.get("DAYS") or []),
        "N_is_six": p.get("N") == 6,
        "the_post_freeze_clause_is_WITHDRAWN":
            w.get("under_this_ruling_that_is_FALSE") is True,
        "COMPUTED_the_before_and_after_sets_PARTITION_the_population":
            (after | before) == set(p.get("DAYS") or []) and not (after & before),
        "COMPUTED_five_of_six_predate_the_freeze": len(before) == 5 and len(after) == 1,
        "NO_SUBSTITUTION_on_a_failed_day": "NO SUBSTITUTION" in str(
            fq.get("and_then", "")).upper(),
        "a_quality_decision_that_saw_an_outcome_REFUSES":
            fq.get("refusal_if_violated") == "QUALITY_DECISION_SAW_AN_OUTCOME",
        "a_file_count_is_declared_a_FLOOR": (p.get("MEASURED_CALENDAR") or {}).get(
            "A_FILE_COUNT_IS_A_COVERAGE_FLOOR_NOT_A_QUALITY_PASS") is True,
    }
    return _verdict(props, [], item="v2_population", N=p.get("N"))


def item_v2_governing_limit(d: dict) -> dict:
    g = d.get("THE_GOVERNING_LIMIT_OF_THIS_WHOLE_TEST") or {}
    r = g.get("THE_RESIDUAL_REV_STATES_IS_PERMANENTLY_UNCLOSABLE") or {}
    props = {
        "the_five_of_six_fact_is_the_GOVERNING_limit":
            g.get("FIVE_OF_SIX_DAYS_PREDATE_THE_FREEZE") is True,
        "the_lost_structural_guarantee_is_stated":
            "STRUCTURAL GUARANTEE IS GONE" in str(g.get("consequence", "")).upper(),
        "what_the_protection_rests_on_is_ENUMERATED":
            len(g.get("what_the_protection_now_rests_on") or []) >= 3,
        "the_residual_is_named_PERMANENTLY_UNCLOSABLE":
            r.get("status") == "PERMANENTLY_UNCLOSABLE",
        "it_is_stated_that_NO_FUTURE_WORK_CLOSES_IT":
            "no future work closes it" in str(
                r.get("it_is_not_a_gap_to_be_closed_later", "")).lower(),
        "ATTESTED_not_GUARANTEED_is_the_declared_wording":
            "ATTESTED" in str(r.get("how_it_must_be_reported", "")).upper(),
        "the_residual_clause_is_unconditional":
            unconditional(r.get("what"))["unconditional"],
    }
    return _verdict(props, [], item="v2_governing_limit")


def item_v2_09_07_in_primary(d: dict) -> dict:
    c = d.get("THE_09_07_PROBLEM_NOW_INSIDE_THE_PRIMARY") or {}
    s = c.get("REQUIRED_SENSITIVITY_LEG") or {}
    props = {
        "the_merge_is_acknowledged": c.get("THE_RULING_MERGES_IT") is True,
        "the_primary_is_declared_to_contain_a_SEEN_day":
            c.get("so_the_primary_population_contains_a_day_whose_DIAGNOSTIC_WAS_SEEN") is True,
        "the_two_endpoints_carry_DIFFERENT_statuses":
            (c.get("clean_on_the_ESTIMAND") or {}).get("status")
            != (c.get("consumed_on_the_DIAGNOSTIC") or {}).get("status"),
        "the_NON_INDEPENDENCE_is_stated":
            "CORRELATED" in str(c.get("AND_THE_TWO_ARE_NOT_INDEPENDENT", "")).upper(),
        "the_inclusion_was_on_STATUS_not_OUTCOME":
            "STATUS, not its RESULT" in str(
                c.get("what_was_known_when_the_day_was_ruled_in", "")),
        "a_sensitivity_leg_is_REQUIRED": bool(s.get("what")),
        "THE_POWER_TRAP_IS_CLOSED_IN_ADVANCE":
            "NOT** EVIDENCE" in str(s.get("AND_THE_TRAP_THIS_CLOSES_IN_ADVANCE", ""))
            or "NOT EVIDENCE" in str(s.get("AND_THE_TRAP_THIS_CLOSES_IN_ADVANCE", "")).upper(),
        "only_a_SIGN_difference_counts_as_evidence":
            "SIGN DIFFERENCE" in str(s.get("AND_THE_TRAP_THIS_CLOSES_IN_ADVANCE", "")).upper(),
        "a_sign_difference_BLOCKS_promotion":
            "BLOCKED" in str(s.get("rule_if_the_signs_differ", "")).upper(),
    }
    return _verdict(props, [], item="v2_09_07_in_primary")


def item_v2_power_arm_by_arm(d: dict) -> dict:
    pw = d.get("PRE_REGISTERED_EXPECTATION_ARM_BY_ARM") or {}
    obs = pw.get("observed_z_at_3_days") or {}
    bars = pw.get("holm_bars") or {}
    cv = pw.get("CONDVALUE_X_SKEW") or {}
    hz = pw.get("HAZARD_OVER_SKEWED_REF") or {}
    belongs = pw.get("DOES_09_07_BELONG_IN_A_sqrt_N_SCALING") or {}
    b1, b2 = bars.get("step_1_smaller_p_needs_z"), bars.get("step_2_larger_p_needs_z")
    zc6 = project(obs.get("CONDVALUE_X_SKEW", 0), 3, 6)
    zh6 = project(obs.get("HAZARD_OVER_SKEWED_REF", 0), 3, 6)
    props = {
        "RECOMPUTED_CONDVALUE_projection": abs(cv.get("projected_z_at_N6", 0) - zc6) < 0.01,
        "RECOMPUTED_HAZARD_projection": abs(hz.get("projected_z_at_N6", 0) - zh6) < 0.01,
        "RECOMPUTED_CONDVALUE_clears_at_N6": zc6 >= b1,
        "RECOMPUTED_HAZARD_does_NOT_clear_at_N6": zh6 < b2,
        "RECOMPUTED_CONDVALUE_power": abs(
            cv.get("POWER_AT_ITS_OWN_POINT_ESTIMATE", 0)
            - (1 - _N.cdf(b1 - zc6))) < 0.005,
        "RECOMPUTED_HAZARD_power": abs(
            hz.get("POWER_AT_ITS_OWN_POINT_ESTIMATE", 0)
            - (1 - _N.cdf(b2 - zh6))) < 0.005,
        "COMPUTED_even_the_clearing_arm_is_under_60_percent_power":
            cv.get("POWER_AT_ITS_OWN_POINT_ESTIMATE", 1) < 0.60,
        "HAZARD_is_declared_UNDERPOWERED_BY_CONSTRUCTION":
            "UNDERPOWERED BY CONSTRUCTION" in str(
                hz.get("THE_SENTENCE_THAT_CAN_ONLY_BE_WRITTEN_NOW", "")).upper(),
        "HAZARDs_failure_is_declared_NOT_evidence_of_no_effect":
            "NOT EVIDENCE OF NO EFFECT" in str(
                hz.get("THE_SENTENCE_THAT_CAN_ONLY_BE_WRITTEN_NOW", "")
            ).upper().replace("**", ""),
        "HAZARD_has_a_REPORTING_NAME_that_is_not_NO_EFFECT":
            hz.get("how_HAZARD_must_be_reported_if_it_fails")
            == "NOT_TESTED_AT_ADEQUATE_POWER",
        "the_09_07_scaling_question_is_ANSWERED_BOTH_WAYS":
            "YES" in str(belongs.get("ANSWER", "")).upper()
            and "NO" in str(belongs.get("ANSWER", "")).upper(),
        "the_upper_bound_caveat_cites_the_SIGN_FLIP":
            "SIGN FLIP" in str(pw.get("THIS_IS_AN_UPPER_BOUND_ON_POWER", "")).upper(),
    }
    return _verdict(props, [], item="v2_power_arm_by_arm",
                    recomputed={"CONDVALUE_z6": round(zc6, 4),
                                "HAZARD_z6": round(zh6, 4)})


ITEMS_V2 = (item_v2_population, item_v2_governing_limit,
            item_v2_09_07_in_primary, item_v2_power_arm_by_arm)


def evaluate_v2(d: dict | None = None) -> dict:
    d = d if d is not None else load_v2()
    out = [f(d) for f in ITEMS_V2]
    return {"protocol": d.get("protocol"), "status": d.get("STATUS"),
            "items": out, "n_items": len(out),
            "n_property_checks_driven": sum(
                r.get("n_properties_driven", 0) for r in out),
            "n_items_FAILING": sum(1 for r in out if r["verdict"] == FAILS),
            "no_item_fails": all(r["verdict"] != FAILS for r in out)}


WEAKENED_V2 = [
 ("item_v2_population", "N stops matching the day list",
  ["POPULATION", "N"], 5),
 ("item_v2_population", "a failed day may be substituted after all",
  ["POPULATION", "a_failed_quality_day", "and_then"],
  "the next chronological untouched day is added in its place"),
 ("item_v2_population", "the post-freeze clause is quietly reinstated",
  ["POPULATION", "THE_POST_FREEZE_CLAUSE_IS_WITHDRAWN",
   "under_this_ruling_that_is_FALSE"], False),
 ("item_v2_governing_limit", "the unclosable residual becomes closable",
  ["THE_GOVERNING_LIMIT_OF_THIS_WHOLE_TEST",
   "THE_RESIDUAL_REV_STATES_IS_PERMANENTLY_UNCLOSABLE",
   "it_is_not_a_gap_to_be_closed_later"],
  "a future census could establish it"),
 ("item_v2_governing_limit", "the residual clause gains an escape",
  ["THE_GOVERNING_LIMIT_OF_THIS_WHOLE_TEST",
   "THE_RESIDUAL_REV_STATES_IS_PERMANENTLY_UNCLOSABLE", "what"],
  "a seat that read a day and wrote nothing consumed it invisibly, unless the "
  "seat reports having done so"),
 ("item_v2_09_07_in_primary", "the sensitivity leg stops being required",
  ["THE_09_07_PROBLEM_NOW_INSIDE_THE_PRIMARY", "REQUIRED_SENSITIVITY_LEG",
   "what"], ""),
 ("item_v2_09_07_in_primary", "a five-day fail becomes evidence 09-07 carried it",
  ["THE_09_07_PROBLEM_NOW_INSIDE_THE_PRIMARY", "REQUIRED_SENSITIVITY_LEG",
   "AND_THE_TRAP_THIS_CLOSES_IN_ADVANCE"],
  "a six-day pass beside a five-day fail shows 09-07 carried the answer"),
 ("item_v2_power_arm_by_arm", "HAZARD's failure becomes evidence of no effect",
  ["PRE_REGISTERED_EXPECTATION_ARM_BY_ARM", "HAZARD_OVER_SKEWED_REF",
   "THE_SENTENCE_THAT_CAN_ONLY_BE_WRITTEN_NOW"],
  "HAZARD is expected to fail and its failure is evidence of no effect"),
 ("item_v2_power_arm_by_arm", "HAZARD gets reported as NO_EFFECT",
  ["PRE_REGISTERED_EXPECTATION_ARM_BY_ARM", "HAZARD_OVER_SKEWED_REF",
   "how_HAZARD_must_be_reported_if_it_fails"], "NO_EFFECT"),
 ("item_v2_power_arm_by_arm", "the power figure is fudged upward",
  ["PRE_REGISTERED_EXPECTATION_ARM_BY_ARM", "CONDVALUE_X_SKEW",
   "POWER_AT_ITS_OWN_POINT_ESTIMATE"], 0.85),
]


def selftest_v2(quiet: bool = False) -> int:
    D2 = load_v2()
    FN = {f.__name__: f for f in ITEMS_V2}
    r = evaluate_v2(D2)
    _ok(r["n_items_FAILING"] == 0,
        f"v2 fails no property ({r['n_property_checks_driven']} driven); "
        f"failures {[i['item'] for i in r['items'] if i['verdict']==FAILS]}")
    obs = D2["PRE_REGISTERED_EXPECTATION_ARM_BY_ARM"]["observed_z_at_3_days"]
    zc6 = project(obs["CONDVALUE_X_SKEW"], 3, 6)
    zh6 = project(obs["HAZARD_OVER_SKEWED_REF"], 3, 6)
    _ok(zc6 >= z_from_p2(0.025) and zh6 < z_from_p2(0.05),
        f"RECOMPUTED at N=6: CONDVALUE z={zc6:.4f} CLEARS its bar, HAZARD "
        f"z={zh6:.4f} does NOT -- declared arm by arm before the data")
    _ok((1 - _N.cdf(z_from_p2(0.025) - zc6)) < 0.60,
        f"RECOMPUTED: even the arm expected to CLEAR has power "
        f"{1 - _N.cdf(z_from_p2(0.025) - zc6):.3f} -- a coin flip")
    for item, what, path, val in WEAKENED_V2:
        v = FN[item](_break(D2, path, val))["verdict"]
        _ok(v == FAILS, f"WEAKENED v2 -- {item}: {what} -> {v} (must be {FAILS})")
    if not quiet:
        print(f"[da_forward_test v2] {_C['n'] - _C['bad']}/{_C['n']} checks, "
              f"{_C['bad']} failures | {r['n_property_checks_driven']} properties "
              f"driven, {len(WEAKENED_V2)} weakened clauses REFUSED | N=6, "
              f"5 of 6 PREDATE the freeze, HAZARD underpowered by construction")
    return 1 if _C["bad"] else 0
