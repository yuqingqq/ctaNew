"""THE INTERIM DECLARATION — committed BEFORE 09-01 and 09-02 are opened.

The USER ruled a PRELIMINARY test on the data that exists now rather than a
wait for five days. What makes that valid rather than void is that it is
declared before anything is opened, and that — unlike the 08-29 read — it
carries a SPECIFIC PRE-SPECIFIED HYPOTHESIS rather than a survey: 08-29 found
the candidate LOSES AT MATCHED VOLUME in all three btc budgets, and 09-01 and
09-02 can confirm or refute that.

IT IS AN INTERIM ON AN INCOMPLETE RACE AND SAYS SO IN ITS OWN FIELDS.

It declares and it selects nothing (rule 14). Every number in it is read from
a committed declaration or a frozen module constant, or computed here.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import harmful_action_eval as AE       # noqa: E402
import phase2_declaration as PD        # noqa: E402
import phase2_iter011 as I11           # noqa: E402

DECL_DIR = HERE / "declarations"
REPO = HERE.parents[1]

DAYS = ("20260901", "20260902")
COINS = ("btc", "eth")
POPULATIONS = ("20260901", "20260902", "POOLED")

#: The three statistics. Exactly one is PRIMARY and the ruling says which.
STATISTICS = ("MATCHED_VOLUME", "BY_THRESHOLD", "BY_COUNT")
PRIMARY_STATISTIC = "MATCHED_VOLUME"


class InterimDeclarationRefused(RuntimeError):
    """A named refusal."""


def _git(*a):
    r = subprocess.run(["git", "-C", str(REPO), *a],
                       capture_output=True, text=True, timeout=60)
    return r.stdout.strip() if r.returncode == 0 else None


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def primary_statistic() -> dict:
    """(1) WHAT IS BEING MEASURED, and why this one is primary.

    The construction is the reviewer's on 08-29: the incumbent is given the
    CANDIDATE'S REALISED CANCELLATION COUNT by lowering the incumbent's own
    theta until it cancels the same number. That answers the question
    BY_THRESHOLD cannot, because BY_THRESHOLD does not hold the operating
    point constant across two models whose scores are on different scales --
    on 08-29 the candidate cancelled 2.33x to 3.30x as many btc actions at the
    same theta, so its apparent gain was bought with volume."""
    return {
        "name": PRIMARY_STATISTIC,
        "definition": (
            "net cents (candidate at its frozen theta) MINUS net cents "
            "(incumbent at the theta that makes the incumbent cancel the "
            "SAME NUMBER of actions the candidate actually cancelled)"),
        "construction": (
            "lower the INCUMBENT's own theta to the candidate's realised "
            "count -- the incumbent's ranking is unchanged, only its cutoff "
            "moves, so the comparison is between two policies spending the "
            "SAME cancellation budget"),
        "why_primary": (
            "CLAUDE.md rule 7: controls are matched on the DECISION VARIABLE "
            "(action count). BY_THRESHOLD is not matched on it and BY_COUNT "
            "matches it only by letting each arm pick its own cutoff from the "
            "data being scored, which is non-causal"),
        "reported_beside_it_and_neither_is_primary": [
            "BY_THRESHOLD", "BY_COUNT"],
        "attribution": {
            "source": "the reviewer's construction on the 08-29 artifacts",
            "adopted_by": "USER ruling, dispatch of BE round 26"},
    }


def decomposition() -> dict:
    """The identity the report must satisfy, stated BEFORE it is computed.

    On 08-29 it was exact, and an identity that is exact is a check rather
    than a narrative: if the two terms do not sum to the BY_THRESHOLD
    increment, the report is wrong and says so."""
    return {
        "identity": ("BY_THRESHOLD increment  ==  VOLUME term  +  QUALITY "
                     "term"),
        "volume_term": (
            "incumbent_net(at the candidate's count) MINUS incumbent_net(at "
            "the shared theta) -- what the INCUMBENT would have gained purely "
            "by acting as often as the candidate did"),
        "quality_term": (
            "candidate_net(at its count) MINUS incumbent_net(at the SAME "
            "count) -- which IS the matched-volume statistic, i.e. the part "
            "of the increment that is not volume"),
        "exactness": "algebraic; the two terms telescope, so it must hold to "
                     "floating-point tolerance and is CHECKED, not asserted",
        "why_it_matters": (
            "it splits the headline into the part a bigger budget would have "
            "bought anyone and the part that is the model"),
    }


def prespecified_direction() -> dict:
    """(2) THE PREDICTION, fixed before the data is opened."""
    return {
        "hypothesis_source": "the 08-29 read (Q-BE-250), already consumed",
        "prediction": (
            "the MATCHED_VOLUME increment on btc is NEGATIVE at all three "
            "budgets"),
        "on_eth": (
            "08-29 found eth null in both arms and its cancel-count ratio was "
            "1.01-1.05x, so eth carries NO directional prediction and is "
            "reported without one"),
        "what_would_REFUTE_it": (
            "a positive matched-volume increment on btc, one-sided p below "
            "the declared level after the declared multiplicity"),
        "what_would_CONFIRM_it": (
            "a negative matched-volume increment on btc reproduced on both "
            "days separately, not only pooled"),
        "this_is_a_test_not_a_survey": True,
    }


def nulls() -> dict:
    """(3) BOTH NULLS, EXACT AND ONE-SIDED, at the counts already declared."""
    return {
        "paired_incumbent_null": {
            "statistic": "window-level sign flip of the paired increment",
            "n_perm": I11.N_PERM_011, "seed": I11.PERM_SEED_011,
            "source": "phase2_iter011.N_PERM_011 / PERM_SEED_011",
            "floor_p": 1.0 / (I11.N_PERM_011 + 1),
            "a_p_at_the_floor_is_a_BOUND_not_a_measurement": True},
        "matched_random_null": {
            "n_random": AE.N_RANDOM,
            "source": "harmful_action_eval.N_RANDOM",
            "matching": "action count within (side x hour) strata",
            "compared_on": "net cents, NOT harm share"},
        "sidedness": "ONE-SIDED",
        "sidedness_source": "amendment A2, unchanged from the 08-29 read",
        "direction_is_prespecified_above": True,
        "counts_are_stated_not_ranged": True,
    }


def family() -> dict:
    """(4) THE FAMILY AND ITS DENOMINATOR, computed and committed HERE.

    The inferential family is the PRIMARY statistic on the POOLED population.
    The per-day cells are REPORTED (the USER ruled: never only pooled) but
    they are disaggregation of the same comparison, not six further tests --
    and saying so before the read is what stops the denominator being chosen
    afterwards."""
    budgets = list(PD.BUDGETS) if not isinstance(PD.BUDGETS, dict) else \
        sorted(PD.BUDGETS)
    budget_keys = [f"{int(float(b) * 100)}%" if float(b) < 1 else str(b)
                   for b in budgets]
    inferential = [f"{PRIMARY_STATISTIC}/POOLED/{c}/{b}"
                   for c in COINS for b in budget_keys]
    reported = [f"{s}/{p}/{c}/{b}" for s in STATISTICS for p in POPULATIONS
                for c in COINS for b in budget_keys
                if f"{s}/{p}/{c}/{b}" not in inferential]
    return {
        "budgets": budget_keys,
        "budgets_source": "phase2_declaration.BUDGETS",
        "coins": list(COINS),
        "populations": list(POPULATIONS),
        "statistics": list(STATISTICS),
        "inferential_cells": sorted(inferential),
        "holm_denominator": len(inferential),
        "reported_but_not_in_the_denominator": sorted(reported),
        "n_reported_cells": len(reported),
        "n_cells_total": len(inferential) + len(reported),
        "why_the_denominator_is_not_everything_reported": (
            "the per-day cells are the SAME comparison disaggregated and the "
            "BY_THRESHOLD / BY_COUNT cells are reported beside the primary "
            "rather than tested; inflating the denominator with them would "
            "make the primary impossible to reject for a reason that has "
            "nothing to do with evidence. Declared HERE, before the read, so "
            "it cannot be chosen after"),
    }


def cluster_disclosure() -> dict:
    """(5) G = 2. POINT ESTIMATE AND NO INTERVAL. Stated plainly."""
    return {
        "ruled_cluster_unit": "UTC day",
        "G_complete_days": len(DAYS),
        "bar_for_an_interval": 5,
        "intervals_claimable": False,
        "what_is_reported": "A POINT ESTIMATE AND NO INTERVAL",
        "authority": "CLAUDE.md rule 8 -- below G=5 complete days: point "
                     "estimate, no interval, and say so",
        "unit_actually_used_for_the_null": "window",
        "weaker_than_ruled": True,
        "why_the_p_is_OPTIMISTIC": (
            "the paired null flips WINDOW signs, and windows inside one UTC "
            "day share coin, regime and book state -- they are not "
            "exchangeable, so the null variance is understated and the p is "
            "smaller than a day-clustered p would be. With G=2 there is no "
            "usable day-level permutation either: two days give two sign "
            "assignments"),
        "not_softened": (
            "this is stated as a limit of the interim, not as a caveat to be "
            "read past. An interim that understated its own limits would be "
            "the one thing a result must never do"),
    }


def alpha_spent() -> dict:
    """(6) WHAT THIS LOOK COSTS, so the eventual full read is honest."""
    return {
        "this_is_look": 1,
        "anticipated_looks": 2,
        "the_second": "the full read at G >= 5, currently earliest ~2026-09-06",
        "what_an_interim_look_costs": (
            "an unplanned interim inflates the family-wise error of the "
            "programme's eventual claim: two opportunities to see a rejection "
            "where one was declared"),
        "why_the_cost_here_is_BOUNDED_and_not_open_ended": (
            "the no-choice clause below. Because NOTHING may be chosen on "
            "this read and the interim may only recommend STOPPING, this is a "
            "stopping-only look: it can end the race early but it cannot "
            "change what the race tests, so it cannot bias the estimand"),
        "the_accounting_is_the_USER_S_TO_DECLARE_not_mine": (
            "how much alpha the full read must give back for this look is a "
            "declaration, not a computation, and rule 14 puts it with the "
            "USER. What this artifact fixes is that the look HAPPENED, when, "
            "and on which days -- so the accounting cannot later be done as "
            "if it had not"),
        "recorded_before_the_read": True,
    }


def no_choice_clause() -> dict:
    """(7) THE CLAUSE THAT KEEPS THE REST OF THE RACE HONEST."""
    return {
        "nothing_may_be_chosen_on_what_this_shows": True,
        "frozen_against_this_read": [
            "threshold", "budget", "coin set", "candidate", "arm identity",
            "horizon", "operating point"],
        "the_interim_MAY": "recommend STOPPING",
        "the_interim_MAY_NEVER": (
            "trigger a change to what is being tested -- a stopping rule and "
            "a tuning rule look identical in a commit and are opposite in "
            "what they cost"),
        "days_consumed_by_this_read": list(DAYS),
        "consequence_stated_plainly": (
            "09-01 and 09-02 are CONSUMED. They are no longer untouched days "
            "and CANNOT serve as clean forward validation afterwards. That is "
            "the price of looking early and the USER is paying it knowingly"),
        "days_still_untouched_after_this": (
            "09-03 onward; the full read's clean population is what remains, "
            "not what this consumed"),
    }


def _conventions_differ() -> dict:
    """increment()'s primary convention vs iteration 011's, compared."""
    import be_forward_metric as FM
    primary = next(k for k, v in FM.PAIRING_CONVENTIONS.items()
                   if v["role"].startswith("PRIMARY"))
    bridge = next(k for k, v in FM.PAIRING_CONVENTIONS.items()
                  if not v["role"].startswith("PRIMARY"))
    return {"increment_primary_convention": primary,
            "iteration_011_convention": bridge,
            "they_differ": primary != bridge,
            "so_no_published_number_shares_this_estimand": primary != bridge}


def reconciliation_caveat() -> dict:
    """READ-R3, carried from the START rather than added afterwards."""
    return {
        "claim": ("this estimand has NEVER been reconciled against any "
                  "published number, and cannot be from existing artifacts"),
        "why_not": (
            "`increment()` computes a BY_THRESHOLD estimand and iteration "
            "011's published cells are BY_COUNT -- different estimands, so "
            "there is no published number to reconcile against. The primary "
            "statistic here, MATCHED_VOLUME, is newer still and has no "
            "published counterpart at all"),
        "what_the_36_of_36_did_validate": (
            "the BRIDGE arm and everything downstream of "
            "`increment_by_window`, not the primary estimand"),
        "for_the_reader": ("do not carry the reconciliation's authority onto "
                           "any number this read produces"),
        "required_in_the_receipt": True,
        # R39: the claim above rests on the two conventions DIFFERING, which
        # is comparable at the constants rather than asserted in prose.
        "conventions_differ_COMPUTED": _conventions_differ(),
        "carried_from_the_start_not_added_after": True,
    }


def build() -> dict:
    return {
        "protocol": "BE_INTERIM_DECLARATION_V1",
        "status": "INTERIM ON AN INCOMPLETE RACE",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "declared_before_anything_was_opened": True,
        "ruled_by": "USER, dispatch of BE round 26 (scope change)",
        "population": {
            "days": list(DAYS),
            "coins": list(COINS),
            "era": "clob_v4_1 (the race era) -- not the clob_v3_1 of 08-29",
            "each_day_reported_SEPARATELY_as_well_as_pooled": True,
            "never_only_pooled": True,
        },
        "primary_statistic": primary_statistic(),
        "decomposition": decomposition(),
        "prespecified_direction": prespecified_direction(),
        "nulls": nulls(),
        "family": family(),
        "cluster_disclosure": cluster_disclosure(),
        "alpha_spent": alpha_spent(),
        "no_choice_clause": no_choice_clause(),
        "reconciliation_caveat": reconciliation_caveat(),
        "latency_ms": PD.TARGET_LATENCY_MS,
        "declared_in_commit": _git("rev-parse", "HEAD"),
        "selects_nothing": True,
        "adjudicates": None,
        "who_decides": "the USER (rule 14)",
    }


EXPECTED_CHECKS = 21


def selftest() -> int:
    checks = 0
    fails = []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    d = build()
    ok(d["status"].startswith("INTERIM"),
       "the artifact SAYS it is an interim on an incomplete race, in a field")
    ok(d["declared_before_anything_was_opened"] is True,
       "and that it was declared before anything was opened")
    ok(d["population"]["days"] == ["20260901", "20260902"]
       and d["population"]["never_only_pooled"] is True,
       f"population: {d['population']['days']}, both coins, each day reported "
       f"separately AS WELL AS pooled")
    ok(d["primary_statistic"]["name"] == "MATCHED_VOLUME"
       and d["primary_statistic"]["reported_beside_it_and_neither_is_primary"]
       == ["BY_THRESHOLD", "BY_COUNT"],
       "PRIMARY is MATCHED_VOLUME; BY_THRESHOLD and BY_COUNT are reported "
       "beside it and neither is primary")
    ok("lower" in d["primary_statistic"]["construction"]
       and "SAME" in d["primary_statistic"]["definition"],
       "the construction is the reviewer's: the INCUMBENT's own theta is "
       "lowered to the candidate's realised count")
    ok("NEGATIVE" in d["prespecified_direction"]["prediction"]
       and "btc" in d["prespecified_direction"]["prediction"],
       f"the direction is PRE-SPECIFIED: {d['prespecified_direction']['prediction']}")
    ok(d["prespecified_direction"]["what_would_REFUTE_it"],
       "and what would REFUTE it is stated, which is what makes it a test")
    ok("no directional prediction" in d["prespecified_direction"]["on_eth"]
       or "NO directional prediction" in d["prespecified_direction"]["on_eth"],
       "eth carries NO directional prediction -- 08-29 found it null in both "
       "arms, so predicting there would be a fishing expedition")
    n = d["nulls"]
    ok(n["paired_incumbent_null"]["n_perm"] == 2000
       and n["matched_random_null"]["n_random"] == 200,
       f"both nulls EXACT: {n['paired_incumbent_null']['n_perm']} "
       f"permutations, {n['matched_random_null']['n_random']} draws")
    ok(n["sidedness"] == "ONE-SIDED" and n["counts_are_stated_not_ranged"],
       "ONE-SIDED, counts stated and not ranged")
    ok(abs(n["paired_incumbent_null"]["floor_p"] - 1 / 2001) < 1e-15,
       f"the floor is COMPUTED ({n['paired_incumbent_null']['floor_p']:.8f}), "
       f"so a p at it is reported as a bound")
    f = d["family"]
    ok(f["holm_denominator"] == len(f["inferential_cells"]) == 6,
       f"the family is COMPUTED: {f['holm_denominator']} inferential cells "
       f"(primary x 2 coins x 3 budgets on POOLED)")
    ok(f["n_cells_total"] == len(f["inferential_cells"])
       + len(f["reported_but_not_in_the_denominator"]),
       f"and the reported-but-not-tested cells are enumerated too "
       f"({f['n_reported_cells']}), so nothing is invisible")
    ok(f["n_cells_total"] == 54,
       f"total cells {f['n_cells_total']} = 3 statistics x 3 populations x 2 "
       f"coins x 3 budgets")
    c = d["cluster_disclosure"]
    ok(c["G_complete_days"] == 2 and c["intervals_claimable"] is False
       and c["what_is_reported"] == "A POINT ESTIMATE AND NO INTERVAL",
       f"G={c['G_complete_days']}: POINT ESTIMATE AND NO INTERVAL, under "
       f"rule 8")
    ok("not\nexchangeable" in c["why_the_p_is_OPTIMISTIC"].replace(" ", "\n")
       and "understated" in c["why_the_p_is_OPTIMISTIC"],
       "and WHY the p is optimistic is stated in the VALUE and not just "
       "promised by the key -- windows inside a day are NOT EXCHANGEABLE so "
       "the null variance is UNDERSTATED (READ-R3, carried from the start)")
    a = d["alpha_spent"]
    ok(a["this_is_look"] == 1 and a["anticipated_looks"] == 2,
       "the alpha this look spends is recorded: look 1 of 2")
    ok("STOPPING" in a["why_the_cost_here_is_BOUNDED_and_not_open_ended"],
       "and WHY it is bounded -- a stopping-only look cannot bias the "
       "estimand, because it cannot change what is tested")
    nc = d["no_choice_clause"]
    ok(nc["nothing_may_be_chosen_on_what_this_shows"] is True
       and set(nc["frozen_against_this_read"]) >= {
           "threshold", "budget", "coin set", "candidate", "arm identity",
           "horizon", "operating point"},
       f"NOTHING may be chosen on this read: {nc['frozen_against_this_read']}")
    ok(nc["days_consumed_by_this_read"] == ["20260901", "20260902"]
       and "CANNOT serve as clean forward validation" in
       nc["consequence_stated_plainly"],
       "and the artifact SAYS the two days are CONSUMED and cannot serve as "
       "clean forward validation afterwards")
    ok(d["reconciliation_caveat"]["required_in_the_receipt"] is True
       and d["reconciliation_caveat"]["carried_from_the_start_not_added_after"],
       "READ-R3's never-reconciled caveat is carried FROM THE START")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}.")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--declare" in argv:
        print(json.dumps(build(), indent=1, sort_keys=True, default=str))
        return 0
    print("usage: be_interim_declaration.py --selftest | --declare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
