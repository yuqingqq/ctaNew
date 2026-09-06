"""THE DESIGN DECLARATION for the ruled multi-day Gate-1 run (R-547).

NO DATA IS TOUCHED BY THIS MODULE. It emits a declaration and proves, on
synthetic fixtures, that the declared rule can FAIL an arm that should
fail, PASS an arm that should pass, and REFUSE a day whose reference book
does not match its pinned digest. The run itself is a later, separate act
that the reviewer files on first (R-547 item 6).

WHAT THE USER RULED (R-547(A), verbatim in the register): Gate 1's control
is the REPLAY NULL -- random decisions, same count and same side split as
the arm, drawn from the arm's own decision population at the arm's own
theta, replayed through the SAME stateful cascade. Five named days decide
the section-7 stopping rule. Design and null are committed BEFORE data.

    python3 live/pm_research/de_multiday_design_declaration.py --selftest
    python3 live/pm_research/de_multiday_design_declaration.py --emit --output PATH
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


PROTOCOL = "P003_DE_MULTIDAY_GATE1_DESIGN_DECLARATION_V1"
EXPECTED_CHECKS = 20

DAYS = ("2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05")
ARMS = ("CONDVALUE_X_SKEW", "HAZARD_OVER_SKEWED_REF")
G = len(DAYS)
MULTIPLICITY = 2
MIN_DRAWS = 500
ALPHA = 0.05

#: Fixed on the CONSUMED 2026-08-24 13:50-14:50Z hour and pinned in two
#: places that must agree: BE's null artifact (`cells[arm].arm_filed.theta`)
#: and DE's arms emission (`arms[arm].identity`). Neither is re-fitted on
#: any of the five days -- that is what makes the days unconsumed.
THETA = {"CONDVALUE_X_SKEW": 0.32450609461933483,
         "HAZARD_OVER_SKEWED_REF": 0.43525926488298716}
THETA_PINS = {
    "CONDVALUE_X_SKEW": {
        "artifact": "data/pm_5min/derived/be_cancel_axis_null_v1.json",
        "sha256": "6951f57d2b8a23bd2d51f24d25659ddffc671c5932aaba89246b025"
                  "182c2fa08",
        "json_path": "cells.CONDVALUE_X_SKEW.arm_filed.theta",
        "head": "q1_arrival_composed_lgbm",
        "model_artifacts": {"lgbm_haz_btc.txt": "ec52055214a01ed5",
                            "lgbm_thresholds_btc.json": "0fa2f1f7a5a4c58f"}},
    "HAZARD_OVER_SKEWED_REF": {
        "artifact": "data/pm_5min/derived/be_cancel_axis_null_v1.json",
        "sha256": "6951f57d2b8a23bd2d51f24d25659ddffc671c5932aaba89246b025"
                  "182c2fa08",
        "json_path": "cells.HAZARD_OVER_SKEWED_REF.arm_filed.theta",
        "head": "incumbent_linear_d",
        "model_artifacts": {"linear_d_btc.json": "18701008c2bd18c6"}},
}

CASCADE_MACHINERY = {
    "path": "live/pm_research/be_cancel_axis_null.py",
    "owner": "BE",
    "de_does_not_reimplement_it": True,
    "why": "two implementations of one cascade is two cascades; the null "
           "must run through the SAME stateful policy the arm ran through "
           "or it is not a control for it (R-547 item 1)",
}


class DesignRefused(RuntimeError):
    """The declared design cannot be honoured on the inputs given."""


# ---------------------------------------------------------------- the rule

def per_day_location(observed: float, null_draws: list) -> dict:
    """One-sided, LARGER IS BETTER: p = (1 + #{null >= observed}) / (1+K)."""
    k = len(null_draws)
    if k < MIN_DRAWS:
        raise DesignRefused(
            f"REFUSED: {k} draws is below the declared minimum {MIN_DRAWS} "
            f"(rule 6). An under-sampled correct null flatters as much as "
            f"a wrong one.")
    ge = sum(1 for v in null_draws if v >= observed)
    return {"n_draws": k, "n_null_ge_observed": ge,
            "p_one_sided": (1 + ge) / (1 + k), "floor": 1 / (1 + k)}


def per_day_standardised_excess(observed: float, null_draws: list) -> float:
    """Z = (observed - mean(null)) / sd(null). The per-day cluster value."""
    sd = statistics.pstdev(null_draws)
    if sd == 0:
        raise DesignRefused(
            "REFUSED: the null has zero dispersion, so a standardised "
            "excess is undefined. A degenerate null is a STATUS, never a "
            "large Z.")
    return (observed - statistics.fmean(null_draws)) / sd


def _smallest_G(alpha: float, m: int, cap: int = 40) -> int | None:
    """The smallest number of day clusters whose unanimous sign test clears
    Holm -- COMPUTED, because a hardcoded 7 was wrong by one and the whole
    point of the field is to price a decision in calendar days."""
    thr = alpha / m
    for g in range(1, cap + 1):
        if 2.0 ** (-g) <= thr:
            return g
    return None


def day_cluster_verdict(z_by_day: dict, *, alpha: float = ALPHA,
                        m: int = MULTIPLICITY) -> dict:
    """THE SECTION-7 PREDICATE, DECLARED BEFORE ANY DAY IS SEEN.

    Cluster unit is the UTC day (rule 8). The statistic is the mean of the
    per-day standardised excesses; the test is the EXACT SIGN TEST over the
    G days, whose one-sided p is 2^-G when every day agrees.

    THE ASYMMETRY IS DELIBERATE AND IS THE POINT. This is a STOPPING rule:
      FAIL  -- cheap, and needs no significance. An arm fails if its
               cluster mean is <= 0, or if its day signs are not unanimous.
               "Did not beat the null" is not a claim that needs power.
      PASS  -- expensive, and CAPPED BY ARITHMETIC AT THIS G. With G = 5 the
               smallest attainable one-sided sign-test p is 2^-5 = 0.03125,
               and Holm at m = 2 compares the smaller p against alpha/2 =
               0.025. 0.03125 > 0.025, SO NO ARM CAN CLEAR HOLM ON THIS RUN
               EVEN IF EVERY DAY GOES ITS WAY. A pass is therefore
               DIRECTIONAL AND CONSISTENT, NEVER SIGNIFICANCE-BEARING --
               the same limit R-529(A) ruled for the forward race, declared
               here BEFORE the run rather than discovered after it.
    COMPUTED, not asserted: the smallest G that clears Holm is 6 at m = 2
    (2^-6 = 0.015625 <= 0.025) and 5 at m = 1 (2^-5 = 0.03125 <= 0.05).
    So ONE MORE ADMISSIBLE DAY would make a unanimous pass
    significance-bearing at m = 2 -- which is a fact worth having before
    the run rather than after it."""
    days = sorted(z_by_day)
    if len(days) != G:
        raise DesignRefused(
            f"REFUSED: the cluster test is declared over exactly G = {G} "
            f"days; got {len(days)}. Dropping or adding a day after the "
            f"fact is choosing after seeing (rule 11).")
    zs = [z_by_day[d] for d in days]
    mean_z = statistics.fmean(zs)
    n_pos = sum(1 for z in zs if z > 0)
    unanimous = n_pos == G
    p_sign = 2.0 ** (-G) if unanimous else None
    holm_threshold = alpha / m
    clears_holm = bool(p_sign is not None and p_sign <= holm_threshold)
    fails = (mean_z <= 0) or (not unanimous)
    return {
        "days": days, "z_by_day": {d: z_by_day[d] for d in days},
        "cluster_unit": "UTC day", "G": G,
        "mean_standardised_excess": mean_z,
        "n_days_positive": n_pos, "signs_unanimous": unanimous,
        "p_one_sided_sign_test": p_sign,
        "multiplicity_m": m, "alpha": alpha,
        "holm_threshold_for_the_smaller_p": holm_threshold,
        "clears_holm": clears_holm,
        "best_attainable_p_at_this_G": 2.0 ** (-G),
        "a_pass_is_significance_bearing": clears_holm,
        "FAILS_THE_SECTION_7_PREDICATE": fails,
        "verdict": ("FAILS_TO_BEAT_THE_REPLAY_NULL" if fails
                    else "BEATS_DIRECTIONALLY_NOT_SIGNIFICANTLY"),
        "smallest_G_that_clears_holm": _smallest_G(alpha, m),
        "smallest_G_that_clears_holm_at_m_1": _smallest_G(alpha, 1),
        "why_a_pass_cannot_be_significant_here": (
            f"2^-{G} = {2.0 ** (-G)} against a Holm threshold of "
            f"{holm_threshold} at m = {m}; the smallest G that clears is "
            f"{_smallest_G(alpha, m)} at m = {m} and "
            f"{_smallest_G(alpha, 1)} at m = 1"),
    }


def verify_book_digest(day: str, book_path: str, declared_sha256: str,
                       actual_sha256: str) -> dict:
    """A day whose reference book does not match its pinned digest REFUSES
    -- the whole day, not the offending draw."""
    if actual_sha256 != declared_sha256:
        raise DesignRefused(
            f"REFUSED: day {day} reference book digest mismatch at "
            f"{book_path}: declared {declared_sha256}, actual "
            f"{actual_sha256}. A day whose book moved is not the day the "
            f"design declared, and no draw on it is admissible.")
    return {"day": day, "book": book_path, "sha256": actual_sha256,
            "verified": True}


# ------------------------------------------------------------ declaration

def declaration() -> dict:
    return {
        "protocol": PROTOCOL,
        "status": "DESIGN_DECLARATION_NO_DATA_TOUCHED",
        "declared_before_any_draw": True,
        "ruling": "R-547 (USER). Gate 1's control is the REPLAY NULL; the "
                  "five named days decide the section-7 stopping rule; "
                  "design and null committed before data.",
        "days": {
            "named": list(DAYS), "G": G,
            "why_these": "the only era-pure clob_v4_1 days in existence "
                         "(R-547(C)); 08-29/30/31 straddle era boundaries "
                         "and are inadmissible; 08-20..08-25 are consumed",
            "admissibility_source": "da_dayverdict_<YYYYMMDD>.json, the "
                                    "era-admission block -- NOT CLAUDE.md "
                                    "rule 5's mm_hf Binance boundary, "
                                    "which does not govern this tape "
                                    "(R-547(B))",
            "nothing_has_been_chosen_on_them": (
                "the thetas were fixed on the consumed 08-24 hour and the "
                "race scored a different object on these days, sealed and "
                "unread"),
        },
        "arms": list(ARMS),
        "theta": THETA,
        "theta_pins": THETA_PINS,
        "theta_is_not_refitted_on_any_of_the_five_days": True,
        "what_DE_needs_from_BE_per_day": {
            "object": "the day's reference book -- the same shape as the "
                      "08-24 arms cache's `fr`: reference (slug -> side -> "
                      "generations with tranches), statuses, population, "
                      "n_slugs, and terminal_marks",
            "why_terminal_marks": "the inventory leg is valued to each "
                                  "window's terminal mark; a book without "
                                  "them reads NO_TERMINAL_MARK on every "
                                  "fill",
            "digest_pinning": (
                "BE publishes, per day, the book path and its sha256 in a "
                "builder declaration. DE's per-day artifact carries "
                "`reference_book: {day, path, sha256}` and RECOMPUTES the "
                "digest at read time; a mismatch REFUSES that day"),
            "de_does_not_build_the_book_and_does_not_open_BEs_pickle": True,
        },
        "decision_population": {
            "definition": "the arm's above-threshold events on day d at "
                          "the arm's FIXED theta -- the set a cancel "
                          "decision is drawn from",
            "on_the_consumed_hour_it_was": {
                "CONDVALUE_X_SKEW": {"decisions": 1154,
                                     "by_side": {"BUY_UP": 586,
                                                 "SELL_UP": 568}},
                "HAZARD_OVER_SKEWED_REF": {"decisions": 106,
                                           "by_side": {"BUY_UP": 41,
                                                       "SELL_UP": 65}}},
            "per_day_values_are_UNKNOWN_and_are_an_OUTPUT": True,
            "a_day_with_too_few_decisions_is_a_STATUS": (
                "if an arm's decision count on a day is 0, that day is "
                "reported as a counted status for that arm and the arm "
                "cannot be aggregated over G = 5 -- it does not silently "
                "become a 4-day test"),
        },
        "null": {
            "design": "random decisions matched to the arm's OWN count and "
                      "side split on that day, drawn from that day's "
                      "decision population, replayed through the SAME "
                      "stateful cascade",
            "matched_on": ["decision count", "side split"],
            "explicitly_not_matched_on": [
                "realised cancel count", "realised cancel set",
                "fills lost -- a decision is not a cancel"],
            "min_draws_per_arm_per_day": MIN_DRAWS,
            "machinery": CASCADE_MACHINERY,
            "seed": {
                "rule": "seed = int(sha256(day_book_sha256 || arm || "
                        "'P003_GATE1_MULTIDAY')[:8], 16)",
                "why": "the seed PINS THE DATA: it is derived from the "
                       "day's book digest, so a book that moved cannot "
                       "reuse the same draw sequence, and the sequence is "
                       "reproducible from the artifact alone",
                "not_a_bare_integer": True},
        },
        "metric": {
            "primary": "D(E0) -- net value delta at maker fee zero (our "
                       "signed rate), arm minus QR_SKEW_ONLY, per day",
            "robustness": "D(E-R) at the rebate's identity value, reported "
                          "beside it and never substituted for it",
            "unchanged_from": "R-537 / the fee-endpoint receipt v1-v3",
        },
        "per_day_location": {
            "statistic": "p = (1 + #{null >= observed}) / (1 + K), "
                         "ONE-SIDED, larger D is better",
            "floor_at_500_draws": 1 / (1 + MIN_DRAWS),
        },
        "day_cluster_aggregation": {
            "cluster_unit": "UTC day (rule 8)",
            "per_day_value": "Z_d = (D_d - mean(null_d)) / sd(null_d)",
            "estimate": "mean of Z_d over the G = 5 days",
            "interval": "reported ONLY because G >= 5; the exact sign test "
                        "over the 5 day signs is the test, and the "
                        "interval is the 5 Z values themselves -- no "
                        "normal approximation is claimed at n = 5",
            "why_not_pooled_over_draws": "pooling 2,500 draws across days "
                                         "would treat the DRAW as the "
                                         "cluster unit and inflate by the "
                                         "number of draws, which is free",
        },
        "section_7_predicate": {
            "statement": "arm a FAILS iff mean_d Z_(a,d) <= 0 OR the five "
                         "day signs are not unanimous",
            "declared_before_any_day_is_seen": True,
            "multiplicity_m": MULTIPLICITY,
            "alpha": ALPHA,
            "THE_ASYMMETRY_IS_THE_POINT": (
                "FAIL is cheap and needs no significance -- 'did not beat "
                "the null' is not a claim that needs power. PASS is capped "
                "by arithmetic: at G = 5 the smallest attainable one-sided "
                "sign-test p is 2^-5 = 0.03125, and Holm at m = 2 compares "
                "it against 0.025. NO ARM CAN CLEAR HOLM ON THIS RUN even "
                "if every day goes its way"),
            "so_a_pass_means": "DIRECTIONAL AND CONSISTENT, NEVER "
                               "SIGNIFICANCE-BEARING -- the same limit "
                               "R-529(A) ruled for the forward race, "
                               "declared here BEFORE the run rather than "
                               "discovered after it",
            "smallest_G_that_would_clear_holm": {
                "m_2": _smallest_G(ALPHA, 2), "m_1": _smallest_G(ALPHA, 1)},
            "and_that_is_ONE_MORE_DAY": (
                "at m = 2 the smallest clearing G is 6, so a SIXTH "
                "admissible day would make a unanimous pass "
                "significance-bearing. There is none yet -- 09-06 is the "
                "next candidate and is not complete -- but the price of a "
                "significant answer is one day, and that is worth knowing "
                "before the run rather than after it"),
            "this_is_stated_now_so_nobody_reads_a_pass_as_a_validation":
                True,
        },
        "falsifiers": {
            "a_planted_arm_that_MUST_FAIL": "an arm whose D sits at the "
                                            "null's median on every day",
            "a_planted_arm_that_MUST_PASS": "an arm above every draw on "
                                            "every day -- and the pass is "
                                            "asserted to be DIRECTIONAL, "
                                            "with clears_holm FALSE",
            "a_planted_day_with_a_WRONG_BOOK_DIGEST": "refuses that day "
                                                      "outright",
            "an_under_sampled_null": "fewer than 500 draws refuses",
            "a_degenerate_null": "zero dispersion is a STATUS, not a large "
                                 "Z",
            "a_missing_day": "G != 5 refuses rather than testing on 4",
        },
        "what_would_REFUTE_this_design": [
            "if the per-day decision population for an arm is systematically "
            "empty or tiny on the admissible days, the matched null cannot "
            "be built and Gate 1's control fails for a reason that is not "
            "about the arm -- that refutes the DESIGN, not the arm",
            "if BE's cascade cannot be driven on a day's book without "
            "re-fitting anything, the 'same cascade' premise is false",
            "if the five days' books are not independent in the way the "
            "day-cluster unit assumes (a single market event spanning "
            "days), the interval is wrong even at G = 5",
            "if the arm's theta is found to have been fitted on any of "
            "these five days, they are consumed and the run is void "
            "(rule 11)",
        ],
        "resources": {
            "basis": "measured on the consumed 08-24 hour",
            "de_arms_replay_per_hour": {"wall_s": 47.0, "peak_gb": 0.61},
            "be_null_500_draws_one_hour_two_arms": {"wall_s": 290.9},
            "per_day_estimate": (
                "a UTC day is 288 five-minute windows against the 12 of "
                "the measured hour, so a linear extrapolation is 24x: "
                "~19 minutes of replay and ~2 hours of null per arm-day, "
                "which is an ESTIMATE from one hour and not a measurement"),
            "cap": "one CPU, MemoryMax=8G, never raised (R-174); if a day "
                   "exceeds the cap the day REFUSES rather than the cap "
                   "rising",
            "recommendation": "the first day is run alone as a smoke with "
                              "its resource observation published before "
                              "the remaining four",
        },
        "what_this_declaration_is_not": {
            "a_run": False, "a_result": False,
            "it_touches_no_data": True,
            "gates_2_to_6": "if an arm passes, Gates 2-6 need FIVE FURTHER "
                            "admissible days (09-06 onward, earliest "
                            "complete set 09-10, readable 09-11) -- these "
                            "five are consumed by this run",
        },
    }


# --------------------------------------------------------------- selftest

def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_multiday_design_declaration] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        try:
            fn()
        except DesignRefused as exc:
            if needle.lower() not in str(exc).lower():
                raise SystemExit(f"[de_multiday_design_declaration] FAIL: "
                                 f"{label} -- wrong reason: {exc}")
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_multiday_design_declaration] FAIL: {label} "
                         f"-- ADMITTED")

    d = declaration()
    ok(d["days"]["named"] == list(DAYS) and d["days"]["G"] == 5,
       f"the five days are NAMED in the declaration: {DAYS}")
    ok(d["theta"]["CONDVALUE_X_SKEW"] == 0.32450609461933483
       and d["theta"]["HAZARD_OVER_SKEWED_REF"] == 0.43525926488298716
       and all("sha256" in v for v in d["theta_pins"].values()),
       "both thetas are stated WITH the artifact and json path they are "
       "pinned at, so a reader can check they were not refitted")
    ok(d["null"]["min_draws_per_arm_per_day"] == 500
       and d["null"]["machinery"]["de_does_not_reimplement_it"] is True,
       "the null is >=500 draws per arm per day through BE's cascade, "
       "cited by path and NOT re-implemented here")
    ok("sha256(day_book_sha256" in d["null"]["seed"]["rule"],
       "the seed PINS THE DATA BY DIGEST -- derived from the day's book "
       "sha256, so a moved book cannot reuse the draw sequence")

    # ---- the rule, driven ----------------------------------------------
    null = [float(i) / 100 for i in range(500)]          # 0.00 .. 4.99
    loc = per_day_location(2.495, null)
    ok(loc["n_draws"] == 500 and abs(loc["floor"] - 1 / 501) < 1e-12
       and 0.0 < loc["p_one_sided"] < 1.0,
       f"per-day location is one-sided with floor 1/501 = "
       f"{loc['floor']:.6f}; a median arm reads p = "
       f"{loc['p_one_sided']:.4f}")
    refuses(lambda: per_day_location(1.0, null[:499]),
            "KNOWN-BAD: 499 draws is below the declared 500 and REFUSES",
            "below the declared minimum")
    refuses(lambda: per_day_standardised_excess(1.0, [3.0] * 500),
            "KNOWN-BAD: a null with zero dispersion is a STATUS, never a "
            "large Z", "zero dispersion")

    # A PLANTED ARM THAT MUST FAIL: sits at the null's mean every day.
    z_fail = {day: 0.0 for day in DAYS}
    v_fail = day_cluster_verdict(z_fail)
    ok(v_fail["FAILS_THE_SECTION_7_PREDICATE"] is True
       and v_fail["verdict"] == "FAILS_TO_BEAT_THE_REPLAY_NULL"
       and v_fail["signs_unanimous"] is False,
       "PLANTED ARM THAT MUST FAIL: an arm at the null's mean on every day "
       "FAILS -- mean Z is 0 and no day sign is positive")
    z_mixed = {DAYS[0]: 3.0, DAYS[1]: 3.0, DAYS[2]: 3.0, DAYS[3]: 3.0,
               DAYS[4]: -0.01}
    v_mixed = day_cluster_verdict(z_mixed)
    ok(v_mixed["FAILS_THE_SECTION_7_PREDICATE"] is True
       and v_mixed["mean_standardised_excess"] > 0
       and v_mixed["n_days_positive"] == 4,
       "AND A STRONG-BUT-NOT-UNANIMOUS ARM ALSO FAILS: mean Z is "
       "comfortably positive and ONE day disagrees. The stopping rule is "
       "declared on unanimity, not on the mean, so a four-of-five arm "
       "cannot be talked into a pass after the fact")

    # A PLANTED ARM THAT MUST PASS -- and the pass must be DIRECTIONAL.
    z_pass = {day: 4.0 for day in DAYS}
    v_pass = day_cluster_verdict(z_pass)
    ok(v_pass["FAILS_THE_SECTION_7_PREDICATE"] is False
       and v_pass["verdict"] == "BEATS_DIRECTIONALLY_NOT_SIGNIFICANTLY"
       and v_pass["signs_unanimous"] is True,
       "PLANTED ARM THAT MUST PASS: unanimous positive on all five days "
       "does NOT fail the section-7 predicate")
    ok(abs(v_pass["p_one_sided_sign_test"] - 0.03125) < 1e-12
       and abs(v_pass["holm_threshold_for_the_smaller_p"] - 0.025) < 1e-12
       and v_pass["clears_holm"] is False
       and v_pass["a_pass_is_significance_bearing"] is False,
       f"AND THE PASS IS CAPPED BY ARITHMETIC, DECLARED NOW: 2^-5 = "
       f"{v_pass['p_one_sided_sign_test']} against a Holm threshold of "
       f"{v_pass['holm_threshold_for_the_smaller_p']} at m = 2, so "
       f"clears_holm is FALSE even when every day goes the arm's way. A "
       f"pass is DIRECTIONAL, never significance-bearing")
    ok(_smallest_G(0.05, 2) == 6 and _smallest_G(0.05, 1) == 5
       and 2.0 ** -6 <= 0.025 and 2.0 ** -5 > 0.025
       and d["section_7_predicate"]["smallest_G_that_would_clear_holm"]
       == {"m_2": 6, "m_1": 5},
       "and the clearing G is COMPUTED, not asserted: 6 at m = 2 and 5 at "
       "m = 1. I first wrote 7 and 6 by hand and the check caught it -- "
       "so the price of a significance-bearing answer is ONE MORE DAY, "
       "not two, and that is known before anyone spends them")
    refuses(lambda: day_cluster_verdict({d: 1.0 for d in DAYS[:4]}),
            "KNOWN-BAD: four days REFUSE rather than quietly testing on a "
            "smaller G -- dropping a day after the fact is choosing after "
            "seeing", "exactly G = 5")

    # A PLANTED DAY WITH A WRONG BOOK DIGEST.
    ok(verify_book_digest("2026-09-01", "b.pkl", "a" * 64,
                          "a" * 64)["verified"] is True,
       "POSITIVE CONTROL, AND IT ADMITS: a day whose book digest matches "
       "its pin is verified")
    refuses(lambda: verify_book_digest("2026-09-01", "b.pkl", "a" * 64,
                                       "b" * 64),
            "PLANTED DAY WITH A WRONG BOOK DIGEST: the whole day REFUSES, "
            "not the offending draw", "digest mismatch")

    ok(d["section_7_predicate"]["multiplicity_m"] == 2
       and d["section_7_predicate"][
           "this_is_stated_now_so_nobody_reads_a_pass_as_a_validation"],
       "multiplicity m = 2 is declared in the document, not inferred at "
       "read time")
    ok(len(d["what_would_REFUTE_this_design"]) >= 4,
       f"the declaration says what would REFUTE IT, not only what would "
       f"refute the arms: {len(d['what_would_REFUTE_this_design'])} "
       f"conditions")
    ok(d["resources"]["cap"].startswith("one CPU, MemoryMax=8G")
       and "day REFUSES rather than the cap rising"
       in d["resources"]["cap"],
       "the resource declaration names the cap AND what happens when a "
       "day exceeds it -- the day refuses, the cap never rises (R-174)")
    ok(d["decision_population"]["per_day_values_are_UNKNOWN_and_are_an_"
                                "OUTPUT"] is True,
       "the per-day decision counts are declared as OUTPUTS, so no "
       "expectation about them can be quietly turned into a filter")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_multiday_design_declaration] PASS -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.emit or a.output is None:
        ap.error("choose --selftest or --emit --output PATH")
    me = Path(__file__).resolve()
    payload = declaration()
    payload["as_of"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    payload["source_identity"] = {
        "producing_code": me.name,
        "producing_code_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
    }
    selftest_rc = selftest()
    payload["battery"] = {"outcome": "PASS" if selftest_rc == 0 else "FAIL",
                          "n_checks": EXPECTED_CHECKS - 1,
                          "ran_in_the_emitting_process": True}
    if a.output.exists():
        raise DesignRefused(f"output already exists: {a.output}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"emitted": str(a.output), "status": payload["status"],
                      "G": G, "arms": list(ARMS),
                      "battery": payload["battery"]["outcome"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
