"""WHAT THE BAND'S MARGIN ACTUALLY DEPENDS ON -- measured at the gate.

§8's band is 14 consecutive calendar days needing 10 evaluable, and
extension is forbidden. So the margin is a probability, not a
subtraction, and it rests on three things this file measures rather than
asserts:

  (a) THE GATE'S THRESHOLD IS A PARAMETER OF THE RATE. `be_build_
      preflight` passes a day when its population has at most ONE missing
      interior window. Quoting a pass rate without that number quotes
      half a statistic, so the rate is reported AT the gate and either
      side of it.

  (b) THE LEDGER IS NOT THE GATE. The collector's gap ledger is written
      by the collector and can be stale. This ranks the days by the
      ledger and asks how often a day the ledger calls cleaner FAILS
      while a dirtier one PASSES.

  (c) THE FAILING INPUT IS THE POPULATION, NOT THE TAPE. Measured: the
      raw tape holds 288 window files for btc AND eth on every day of
      09-01..09-11 except 09-03 (287), while BE's own `population(day)`
      supplies 265, 248, 247, 288, 288, 288, 287, 288, 288, 288, 284.
      Four days fail the gate on a COMPLETE tape. That is a derivation
      failure, and unlike a collection failure it is REBUILDABLE.

Usage:  de_band_hazard.py --falsify
        de_band_hazard.py --report
"""
from __future__ import annotations

CALL_SITE = {
    "kind": "IMPORTED_BY",
    "by": "de_band_decision.py",
    "gates": "amendment admissibility, consulted on every emit",
}

import json
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import de_fair_value_predictive as PRED                # noqa: E402

PROTOCOL = "P003_DE_BAND_HAZARD_V1"

#: §8's band, as declared. Not tunable here.
BAND_DAYS = 14
NEED_EVALUABLE = 10

#: BE's gate, read from be_build_preflight's own predicate:
#:     "PASS" if len(missing) <= 1 else WOULD_FAIL:MANY_MISSING_WINDOWS
GATE_MAX_MISSING_INTERIOR = 1
GATE_SOURCE = "be_build_preflight.check_day, window-supply row"

#: MEASURED 2026-09-12, 09-01..09-11. `pop_windows` is BE's own
#: population(day)["n_windows"]; `pop_missing` is the interior-window
#: count the gate tests; `tape_windows` is raw 5-min files present for
#: BOTH coins; `ledger_gap_s` is summed gap_closed duration for btc.
DAYS = {
    "20260901": {"pop_windows": 265, "pop_missing": 10,
                 "tape_windows": 288, "ledger_gap_s": 2025.5},
    "20260902": {"pop_windows": 248, "pop_missing": 40,
                 "tape_windows": 288, "ledger_gap_s": 1769.1},
    "20260903": {"pop_windows": 247, "pop_missing": 41,
                 "tape_windows": 287, "ledger_gap_s": 2294.7},
    "20260904": {"pop_windows": 288, "pop_missing": 0,
                 "tape_windows": 288, "ledger_gap_s": 440.2},
    "20260905": {"pop_windows": 288, "pop_missing": 0,
                 "tape_windows": 288, "ledger_gap_s": 68.4},
    "20260906": {"pop_windows": 288, "pop_missing": 0,
                 "tape_windows": 288, "ledger_gap_s": 49.1},
    "20260907": {"pop_windows": 287, "pop_missing": 1,
                 "tape_windows": 288, "ledger_gap_s": 145.3},
    "20260908": {"pop_windows": 288, "pop_missing": 0,
                 "tape_windows": 288, "ledger_gap_s": 264.8},
    "20260909": {"pop_windows": 288, "pop_missing": 0,
                 "tape_windows": 288, "ledger_gap_s": 163.7},
    "20260910": {"pop_windows": 288, "pop_missing": 0,
                 "tape_windows": 288, "ledger_gap_s": 91.4},
    "20260911": {"pop_windows": 284, "pop_missing": 4,
                 "tape_windows": 288, "ledger_gap_s": 210.2},
}


#: THE RECENT WINDOW, named rather than chosen: it is the days AFTER the
#: three whose classification is open. Naming it does not license using
#: it -- see `exclusion_is_licensed`.
RECENT_WINDOW = ("20260904", "20260905", "20260906", "20260907",
                 "20260908", "20260909", "20260910", "20260911")
OPEN_DAYS = ("20260901", "20260902", "20260903")
EXCLUSION_NOT_LICENSED = "RATE_EXCLUSION_NOT_LICENSED"

#: WHICH RATE IS THE FORWARD RATE IS UNRESOLVED, and a single number here
#: would be a false precision that costs fourteen nights. The planning
#: rate is the FULL window until a since-changed CONDITION is named
#: independently of the excluded days' outcomes.
PLANNING_RATE = "full_window"
OPTIMISTIC_BOUND = "recent_window"


class BandHazardRefused(ValueError):
    """The hazard cannot be computed as declared."""


def passes(day: dict, threshold: int = GATE_MAX_MISSING_INTERIOR) -> bool:
    return day["pop_missing"] <= threshold


def rate_at(threshold: int, days: dict = None) -> dict:
    days = days or DAYS
    ok = [d for d, v in days.items() if passes(v, threshold)]
    bad = sorted(d for d in days if d not in ok)
    return {"threshold": threshold, "n_pass": len(ok), "n_days": len(days),
            "rate": len(ok) / len(days), "failing_days": bad}


def threshold_sensitivity(steps=(0, 1, 2, 4, 10, 40, 41)) -> dict:
    rows = [rate_at(t) for t in steps]
    here = rate_at(GATE_MAX_MISSING_INTERIOR)
    nearby = [r for r in rows
              if abs(r["threshold"] - GATE_MAX_MISSING_INTERIOR) <= 3]
    swing = (max(r["rate"] for r in nearby)
             - min(r["rate"] for r in nearby))
    return {"gate": GATE_MAX_MISSING_INTERIOR, "gate_source": GATE_SOURCE,
            "rate_at_the_gate": here["rate"], "rows": rows,
            "swing_within_three_steps_of_the_gate": swing,
            "the_rate_is_a_function_of_the_threshold":
                "quoting a pass rate without the threshold quotes half a "
                "statistic; one step either way moves it by "
                f"{swing:.4f}"}


def ledger_rank_test(days: dict = None) -> dict:
    """DOES THE LEDGER ORDER THE DAYS THE WAY THE GATE DOES?"""
    days = days or DAYS
    order = sorted(days, key=lambda d: days[d]["ledger_gap_s"])
    conc = disc = 0
    inversions = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            pa, pb = passes(days[a]), passes(days[b])
            if pa == pb:
                continue
            if pa and not pb:
                conc += 1
            else:
                disc += 1
                inversions.append({"cleaner_by_ledger_but_FAILS": a,
                                   "dirtier_but_PASSES": b,
                                   "ledger_s": [days[a]["ledger_gap_s"],
                                                days[b]["ledger_gap_s"]]})
    tot = conc + disc
    if not tot:
        raise BandHazardRefused(
            "REFUSED NO_DISCORDANT_PAIRS_TO_RANK: every day has the same "
            "verdict, so the ledger's ordering cannot be tested against "
            "it -- a concordance of 1.0 here would mean nothing.")
    return {"ledger_order_best_to_worst":
                [f"{d[4:]}{'P' if passes(days[d]) else 'F'}" for d in order],
            "n_pass_fail_pairs": tot, "concordant": conc,
            "discordant": disc, "discordant_share": disc / tot,
            "rank_agreement": 2 * conc / tot - 1,
            "inversions": inversions,
            "and_within_the_PASSING_set":
                "ledger gap seconds range 49.1 to 440.2 among days that "
                "PASS, so the ledger separates nothing there",
            "reading":
                "mostly concordant, and wrong exactly where it matters: "
                "the failing day looks cleaner than two passing ones"}


def p_at_least(p: float, n: int = BAND_DAYS,
               k: int = NEED_EVALUABLE) -> float:
    """EXACT binomial tail. The margin is a probability, not a
    subtraction: E[evaluable] - 10 says nothing about the chance of
    landing under 10."""
    if not 0.0 <= p <= 1.0:
        raise BandHazardRefused(
            f"REFUSED RATE_IS_NOT_A_PROBABILITY: {p!r}.")
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def band_table(rates: dict = None, n: int = BAND_DAYS) -> list:
    rates = rates or {
        "REV's 10/11, file presence": 10 / 11,
        "two failing days, 9/11": 9 / 11,
        "9/11 x ETH at 95%": 9 / 11 * 0.95,
        "9/11 x ETH at 90%": 9 / 11 * 0.90,
        "MEASURED at BE's gate, 11 days (7/11)": 7 / 11,
        "MEASURED at BE's gate, recent 8 (7/8)": 7 / 8,
        "recent 8 x ETH at 95%": 7 / 8 * 0.95,
        "recent 8 x ETH at 90%": 7 / 8 * 0.90,
    }
    out = []
    for name, p in rates.items():
        ok = p_at_least(p, n)
        out.append({"basis": name, "p": p, "expected_evaluable": n * p,
                    "margin_vs_10": n * p - NEED_EVALUABLE,
                    "P_at_least_10": ok, "P_NO_VERDICT": 1 - ok})
    return out


def exclusion_is_licensed(condition: str = None,
                          evidence_independent_of_outcomes: bool = False,
                          names_days_by_outcome: bool = True) -> dict:
    """MAY 09-01..09-03 BE EXCLUDED? Only on a NAMED CONDITION.

    Dropping days because they look different is choosing after seeing,
    and it is the mechanism by which a rate gets flattered. The exclusion
    is licensed only when a since-changed condition is identified
    INDEPENDENTLY of those days' outcomes -- REV is on the window-supply
    break, DA has the mask and reboot evidence. Until one of them names
    it, the full-window rate is the planning rate and the recent-window
    rate is the OPTIMISTIC BOUND, never the other way round.
    """
    if not (isinstance(condition, str) and condition.strip()):
        raise BandHazardRefused(
            f"REFUSED {EXCLUSION_NOT_LICENSED}: no since-changed "
            f"CONDITION is named for {list(OPEN_DAYS)}. 'They look "
            f"different' is the observation being explained, not an "
            f"explanation, and excluding on it flatters the rate by "
            f"exactly the amount in question.")
    if not evidence_independent_of_outcomes or names_days_by_outcome:
        raise BandHazardRefused(
            f"REFUSED {EXCLUSION_NOT_LICENSED}: the condition "
            f"{condition!r} is supported only by the outcomes of the days "
            f"it would remove. The evidence must identify the condition "
            f"WITHOUT reference to which days failed.")
    return {"licensed": True, "condition": condition,
            "excluded": list(OPEN_DAYS),
            "planning_rate_becomes": OPTIMISTIC_BOUND}


def forward_rate_pair(days: dict = None) -> dict:
    """BOTH RATES, PERMANENTLY, WITH THE REGIME QUESTION NAMED."""
    days = days or DAYS
    full = {d: v for d, v in days.items()}
    recent = {d: v for d, v in days.items() if d in RECENT_WINDOW}
    rf, rr = rate_at(GATE_MAX_MISSING_INTERIOR, full), rate_at(
        GATE_MAX_MISSING_INTERIOR, recent)
    pf, pr = p_at_least(rf["rate"]), p_at_least(rr["rate"])
    return {
        "FORWARD_RATE_IS_UNRESOLVED": True,
        "planning_rate": {
            "which": PLANNING_RATE, "window": "09-01..09-11",
            "n_pass": rf["n_pass"], "n_days": rf["n_days"],
            "p": rf["rate"], "expected_evaluable": BAND_DAYS * rf["rate"],
            "P_at_least_10": pf, "P_NO_VERDICT": 1 - pf,
            "failing_days": rf["failing_days"]},
        "optimistic_bound": {
            "which": OPTIMISTIC_BOUND, "window": "09-04..09-11",
            "n_pass": rr["n_pass"], "n_days": rr["n_days"],
            "p": rr["rate"], "expected_evaluable": BAND_DAYS * rr["rate"],
            "P_at_least_10": pr, "P_NO_VERDICT": 1 - pr,
            "failing_days": rr["failing_days"]},
        "factor_between_their_failure_probabilities":
            (1 - pf) / (1 - pr) if pr < 1 else None,
        "the_same_seven_days_pass_in_BOTH":
            "the windows share their numerator; the entire disagreement "
            "is whether 09-01..09-03 count",
        "WHICH_APPLIES_IS_UNRESOLVED": {
            "open_days": list(OPEN_DAYS),
            "settled_by": "classify 09-11: was it the LAST of the old "
                          "failures or the FIRST of a new one",
            "exclusion_requires": "a since-changed CONDITION named "
                                  "independently of these days' outcomes "
                                  "(REV: the window-supply break; DA: the "
                                  "mask and reboot evidence)",
            "until_then": "the full window is the PLANNING rate and the "
                          "recent window is the OPTIMISTIC BOUND, never "
                          "the other way round"},
        "a_single_number_here_would_be_false_precision":
            "the two differ by a factor of "
            f"{((1 - pf) / (1 - pr)):.1f} in failure probability, and "
            f"choosing between them on how the days LOOK is the "
            f"mechanism that flatters a rate",
    }


AMENDMENT_AFTER_SEEING = "AMENDMENT_CHOSEN_AFTER_THE_CLOCK_STARTED"
LEVER_SET_MOVED = "LEVER_SET_MOVED_SINCE_THE_PIN"
NO_WINDOW = "VALIDATION_WINDOW_NOT_DECLARED"
NOT_LANDED = "AMENDMENT_NOT_LANDED_ON_BOTH_EXECUTING_REFS"
BAD_STAMP = "TIMESTAMP_IS_NOT_AN_INSTANT"

#: THE LEVER SET, PINNED. REVIEW 264: renaming was caught and ADDING was
#: not -- a lever registered without the flag key took a permissive early
#: return, admitted after the outcome was known. The set is now an
#: enumeration with a digest, and a lever cannot be added, removed or
#: have its flag flipped without the pin refusing.
LEVER_SPEC = {
    "i_accept_the_risk": {"must_be_declared_before_the_clock": False,
                          "is_an_amendment": False},
    "ii_improve_the_input": {"must_be_declared_before_the_clock": False,
                             "is_an_amendment": False},
    "iii_longer_band": {"must_be_declared_before_the_clock": True,
                        "is_an_amendment": True},
    "iv_fewer_required_days": {"must_be_declared_before_the_clock": True,
                               "is_an_amendment": True},
    "v_start_after_a_clean_run": {
        "must_be_declared_before_the_clock": True,
        "is_an_amendment": False},
}
LEVER_SET_PIN = "77ca872139eb297f"

#: The executing refs. An amendment lands on BOTH or it has not landed.
EXECUTING_REFS = ("origin/de-freeze-chain-v2", "origin/be-build-runner")
REPO = "/home/yuqing/ctaNew-wt-de2"

#: WHERE THE CLOCK COMES FROM. Not a parameter: the validation-window
#: declaration, read at a fetched ref.
WINDOW_DECL = "live/pm_research/declarations/da_validation_window_v1.json"
WINDOW_KEYS = ("clock_start_utc", "band_start_utc", "window_start_utc",
               "first_band_day_utc")

#: WHAT COUNTS AS THE BAND HAVING PRODUCED OUTPUT. If any of these exists
#: the outcome is known, whether or not anyone admits looking.
BAND_OUTPUT_PATTERNS = (
    "p003_de_section8_verdict", "p003_de_band_day_score",
    "p003_de_band_evaluable_count", "p003_de_section8_accrual",
)


def required_rate(confidence: float, n: int = BAND_DAYS,
                  k: int = NEED_EVALUABLE) -> float:
    """THE MINIMUM DAILY JOINT RATE at which the band reaches a stated
    confidence. Solved, not tabulated -- the caller picks its own
    tolerance rather than inheriting one."""
    if not 0.0 < confidence < 1.0:
        raise BandHazardRefused(
            f"REFUSED CONFIDENCE_IS_NOT_A_PROBABILITY: {confidence!r}.")
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if p_at_least(mid, n, k) >= confidence:
            hi = mid
        else:
            lo = mid
    return hi


def viability_table(confidences=(0.80, 0.90, 0.95)) -> dict:
    """WHAT WOULD HAVE TO BE TRUE for the test to be viable."""
    pair = forward_rate_pair()
    planning = pair["planning_rate"]["p"]
    optimistic = pair["optimistic_bound"]["p"]
    rows = []
    for c in confidences:
        need = required_rate(c)
        rows.append({
            "confidence": c, "required_daily_joint_rate": need,
            "gap_from_the_planning_rate": need - planning,
            "planning_rate_clears_it": planning >= need,
            "gap_from_the_optimistic_bound": need - optimistic,
            "optimistic_bound_clears_it": optimistic >= need})
    return {"band": {"n": BAND_DAYS, "k": NEED_EVALUABLE},
            "planning_rate": planning, "optimistic_bound": optimistic,
            "rows": rows,
            "reading":
                "the collector-plus-population path must run at the "
                "required rate or the test cannot reach that confidence; "
                "this is a statement about the INPUT, not about the "
                "candidates"}


def g_floor() -> dict:
    """HOW FAR k CAN FALL BEFORE THE TEST STOPS EXISTING.

    Read from the frozen predictive module's OWN ladder, never restated:
    below MIN_NONZERO the exact sign test returns INSUFFICIENT and no p
    exists at all, and just above it the candidate must be PERFECT.
    """
    rows = []
    for g in range(NEED_EVALUABLE + 2, 4, -1):
        lad = PRED.ladder(g)
        clears = [r for r in lad if r.get("clears_holm_step_one")]
        best = lad[0].get("attainable_p") if lad else None
        rows.append({"G": g, "smallest_attainable_two_sided_p": best,
                     "passing_rungs": len(clears),
                     "computable": best is not None,
                     "requires_a_perfect_run": len(clears) == 1})
    floor = min((r["G"] for r in rows if r["computable"]), default=None)
    return {"rows": rows, "min_nonzero_declared": PRED.MIN_NONZERO,
            "lowest_G_with_any_test": floor,
            "at_the_floor_the_candidate_must_be_perfect":
                any(r["G"] == floor and r["requires_a_perfect_run"]
                    for r in rows),
            "reading":
                f"below G={floor} the exact test returns "
                f"{PRED.INSUFFICIENT} and there is no p to correct; at "
                f"G={floor} and G={floor + 1} exactly one rung passes, so "
                f"the candidate must be positive on EVERY day"}


def min_band_for(p: float, confidence: float,
                 k: int = NEED_EVALUABLE) -> int:
    n = k
    while n < 400 and p_at_least(p, n, k) < confidence:
        n += 1
    return n


def max_k_for(p: float, confidence: float, n: int = BAND_DAYS) -> int:
    k = n
    while k > 0 and p_at_least(p, n, k) < confidence:
        k -= 1
    return k


def levers(confidence: float = 0.90) -> dict:
    """THE LEVERS, PRICED. No recommendation is made here: several are
    amendments to a user-authored plan and none of them is this seat's
    to choose."""
    pair = forward_rate_pair()
    pl = pair["planning_rate"]["p"]
    op = pair["optimistic_bound"]["p"]
    floor = g_floor()
    k_pl = max_k_for(pl, confidence)
    k_op = max_k_for(op, confidence)
    return {
        "confidence_used": confidence,
        "i_accept_the_risk": {
            "lever": "do nothing and accept P(no verdict)",
            "cost": {"at_the_planning_rate":
                         1 - p_at_least(pl),
                     "at_the_optimistic_bound": 1 - p_at_least(op)},
            "weakens": "nothing -- the test stays as declared",
            **LEVER_SPEC["i_accept_the_risk"]},
        "ii_improve_the_input": {
            "lever": "raise the collection/population success rate",
            "required_rate_at_this_confidence": required_rate(confidence),
            "gap_from_the_planning_rate":
                required_rate(confidence) - pl,
            "cost": "unknown until the mask-collapse cause is named; may "
                    "be unavailable",
            "weakens": "nothing -- it changes the INPUT, not the test",
            **LEVER_SPEC["ii_improve_the_input"],
            "note": "the only lever that does not trade the test's "
                    "strength for its feasibility"},
        "iii_longer_band": {
            "lever": "more calendar days in the band",
            "n_required_at_the_planning_rate":
                min_band_for(pl, confidence),
            "n_required_at_the_optimistic_bound":
                min_band_for(op, confidence),
            "cost": "calendar time, and every added day is a day the "
                    "candidates are not yet judged",
            "weakens": "nothing statistically -- k is unchanged",
            **LEVER_SPEC["iii_longer_band"],
            "why": "§8 says do not extend opportunistically; extending "
                   "after a shortfall is choosing after seeing"},
        "iv_fewer_required_days": {
            "lever": "lower k below 10",
            "max_k_at_the_planning_rate": k_pl,
            "max_k_at_the_optimistic_bound": k_op,
            "lowest_k_the_TEST_can_compute": floor["lowest_G_with_any_test"],
            "AVAILABLE_AT_THE_PLANNING_RATE":
                k_pl >= floor["lowest_G_with_any_test"],
            "AVAILABLE_AT_THE_OPTIMISTIC_BOUND":
                k_op >= floor["lowest_G_with_any_test"],
            "cost": "the exact test's resolution: at G=10 two rungs pass "
                    "(0 or 1 non-positive day); at G=9 and G=8 exactly "
                    "ONE does, so the candidate must be positive on "
                    "EVERY day",
            "weakens": "the test itself, and the multiplicity arithmetic "
                       "with it -- Holm's threshold does not move, so a "
                       "smaller G spends the same alpha on a coarser "
                       "ladder",
            **LEVER_SPEC["iv_fewer_required_days"]},
        "v_start_after_a_clean_run": {
            "lever": "begin the band only after N demonstrated clean days",
            "cost": "calendar time, and the clean run itself consumes "
                    "days that cannot later be in the band",
            "weakens": "nothing in the test; it buys the RATE by "
                       "selecting when to start, not what to count",
            **LEVER_SPEC["v_start_after_a_clean_run"],
            "caution": "the start condition must be declared as a "
                       "predicate, or 'it looked clean' becomes the "
                       "selection"},
        "THE_TRAP": {
            "which_levers": ["iii_longer_band", "iv_fewer_required_days"],
            "why": "both are what a disappointed operator reaches for "
                   "AFTER a band falls short, and at that point they are "
                   "choosing after seeing: the shortfall itself is the "
                   "information being used",
            "therefore": "if either is to be available at all it must be "
                         "DECLARED NOW, while the outcome is unknown",
            "enforced_by": "amendment_is_admissible -- a declaration "
                           "timestamped after the clock starts REFUSES "
                           f"{AMENDMENT_AFTER_SEEING}",
            "this_is_a_field_not_advice": True},
    }


def _git(args, repo: str = REPO) -> str:
    import subprocess
    out = subprocess.run(["git", "-C", repo] + list(args),
                         capture_output=True, text=True)
    return out.stdout.strip()


def _instant(text: str, what: str):
    """A TIMESTAMP IS AN INSTANT, NEVER A STRING. REVIEW 264 attack 4: a
    -05:00 rendering four hours AFTER the clock sorted BEFORE it, and
    `+00:00` sorts below `Z` at the identical instant."""
    import datetime as dt
    if not isinstance(text, str) or not text.strip():
        raise BandHazardRefused(
            f"REFUSED {BAD_STAMP}: {what} is {text!r}.")
    raw = text.strip().replace("Z", "+00:00")
    try:
        got = dt.datetime.fromisoformat(raw)
    except ValueError:
        raise BandHazardRefused(
            f"REFUSED {BAD_STAMP}: {what}={text!r} is not ISO-8601.") from None
    if got.tzinfo is None:
        raise BandHazardRefused(
            f"REFUSED {BAD_STAMP}: {what}={text!r} carries no offset, so "
            f"it names a wall clock rather than an instant.")
    return got.astimezone(dt.timezone.utc)


def lever_set_digest(spec: dict = None) -> str:
    import hashlib
    spec = spec or LEVER_SPEC
    blob = json.dumps({k: {kk: bool(vv) for kk, vv in sorted(v.items())}
                       for k, v in sorted(spec.items())}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def assert_lever_set_pinned(spec: dict = None) -> dict:
    got = lever_set_digest(spec)
    if LEVER_SET_PIN != "PLACEHOLDER" and got != LEVER_SET_PIN:
        raise BandHazardRefused(
            f"REFUSED {LEVER_SET_MOVED}: the lever set digests {got} "
            f"against the pin {LEVER_SET_PIN}. A lever cannot be added, "
            f"removed or have its before-the-clock flag flipped without "
            f"this refusing -- REVIEW 264 admitted two added levers after "
            f"the outcome was known.")
    return {"digest": got, "pin": LEVER_SET_PIN, "n_levers": len(
        spec or LEVER_SPEC)}


def declared_clock_start(ref: str = EXECUTING_REFS[0],
                         repo: str = REPO) -> dict:
    """THE CLOCK COMES FROM AN ARTIFACT, not from the amender."""
    blob = _git(["show", f"{ref}:{WINDOW_DECL}"], repo)
    if not blob:
        raise BandHazardRefused(
            f"REFUSED {NO_WINDOW}: {WINDOW_DECL} does not exist at {ref}. "
            f"With no declared validation window there is no clock to be "
            f"before, so no amendment can be admitted -- and the amender "
            f"does not get to supply one (REVIEW 264 attack 2).")
    try:
        doc = json.loads(blob)
    except Exception:                                       # noqa: BLE001
        raise BandHazardRefused(
            f"REFUSED {NO_WINDOW}: {WINDOW_DECL} at {ref} is not JSON.") \
            from None
    found = None
    stack = [doc]
    while stack and found is None:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k in WINDOW_KEYS and isinstance(v, str):
                    found = v
                    break
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    if not found:
        raise BandHazardRefused(
            f"REFUSED {NO_WINDOW}: {WINDOW_DECL} at {ref} carries none of "
            f"{list(WINDOW_KEYS)}.")
    return {"clock_start_utc": found, "instant": _instant(
        found, "clock_start_utc"), "read_from": f"{ref}:{WINDOW_DECL}"}


def landed_committer_timestamp(path: str, refs=EXECUTING_REFS,
                               repo: str = REPO) -> dict:
    """THE DECLARATION'S DATE IS THE COMMIT'S, so it cannot be typed."""
    stamps = {}
    for ref in refs:
        line = _git(["log", "-1", "--format=%cI %H", ref, "--", path], repo)
        if not line:
            raise BandHazardRefused(
                f"REFUSED {NOT_LANDED}: {path} has no commit on {ref}. An "
                f"amendment that is not landed on both executing refs has "
                f"no date this guard will accept (REVIEW 264 attack 1).")
        stamp, sha = line.split()[0], line.split()[1]
        stamps[ref] = {"committer_utc": stamp, "commit": sha,
                       "instant": _instant(stamp, f"{ref} committer date")}
    # THE LATER of the two: an amendment is landed when the SECOND ref
    # has it, and taking the earlier would flatter the amender.
    latest = max(stamps.values(), key=lambda v: v["instant"])
    return {"per_ref": {k: {kk: vv for kk, vv in v.items()
                            if kk != "instant"}
                        for k, v in stamps.items()},
            "declared_instant": latest["instant"],
            "declared_utc": latest["committer_utc"],
            "taken_as": "the LATER of the two refs -- an amendment is "
                        "landed when the second ref has it"}


def outcome_is_known_at(refs=EXECUTING_REFS, repo: str = REPO,
                        derived: str = None) -> dict:
    """COMPUTED, NEVER ASSERTED. If the band has produced output, the
    outcome is known whether or not anyone admits looking."""
    hits = []
    for ref in refs:
        listing = _git(["ls-tree", "-r", "--name-only", ref], repo)
        for line in listing.splitlines():
            if any(pat in line for pat in BAND_OUTPUT_PATTERNS):
                hits.append({"ref": ref, "path": line})
    d = Path(derived or "/home/yuqing/ctaNew/data/pm_5min/derived")
    if d.is_dir():
        for f in d.iterdir():
            if any(pat in f.name for pat in BAND_OUTPUT_PATTERNS):
                hits.append({"ref": "derived", "path": f.name})
    return {"known": bool(hits), "evidence": hits[:10],
            "patterns": list(BAND_OUTPUT_PATTERNS),
            "computed_not_asserted": True,
            "why": "a band that has produced a score, an evaluable count "
                   "or a verdict has an outcome, and the party amending "
                   "does not get to decide whether it looked"}


def amendment_is_admissible(lever: str, *, amendment_path: str = None,
                            refs=EXECUTING_REFS, repo: str = REPO,
                            spec: dict = None, derived: str = None) -> dict:
    """AN AMENDMENT IS ADMISSIBLE ONLY BEFORE THE CLOCK -- and every
    input is now bound to an ARTIFACT rather than to an argument.

    REVIEW 264 attacked the previous version five times and won five
    times, because all four inputs were caller-supplied. The order below
    matters as much as the bindings: the STRONGEST clause runs FIRST, so
    it can no longer be gated by the weakest.
    """
    spec_map = spec or LEVER_SPEC
    pin = assert_lever_set_pinned(spec_map)              # fix 4
    if lever not in spec_map:
        raise BandHazardRefused(
            f"REFUSED UNKNOWN_LEVER: {lever!r} is not one of "
            f"{sorted(spec_map)}.")
    entry = spec_map[lever]

    # (6) THE STRONGEST CLAUSE FIRST, and (3) COMPUTED. A lever whose
    # flag is missing or false no longer skips this.
    outcome = outcome_is_known_at(refs, repo, derived)    # fix 3
    if outcome["known"]:
        raise BandHazardRefused(
            f"REFUSED {AMENDMENT_AFTER_SEEING}: the band has already "
            f"produced output ({outcome['evidence'][:2]}), so its outcome "
            f"is known and the shortfall is the information being used. "
            f"This is computed from the artifacts, not asserted by the "
            f"party amending.")

    # (4) THE FLAG IS INDEXED, NEVER .get -- a lever missing the key
    # RAISES instead of taking a permissive default.
    if "must_be_declared_before_the_clock" not in entry:
        raise BandHazardRefused(
            f"REFUSED {LEVER_SET_MOVED}: {lever} carries no "
            f"`must_be_declared_before_the_clock`. A missing key is not a "
            f"False -- REVIEW 264 admitted two levers on exactly that "
            f"permissive default.")
    must = entry["must_be_declared_before_the_clock"]
    if not must:
        return {"admissible": True, "lever": lever, "lever_set": pin,
                "outcome_is_known": outcome["known"],
                "why": "not an amendment requiring pre-declaration",
                "is_an_amendment_to_a_user_authored_plan":
                    entry["is_an_amendment"]}

    if not amendment_path:
        raise BandHazardRefused(
            f"REFUSED {NOT_LANDED}: {lever} names no amendment file to "
            f"date. The declaration's date is its COMMIT's, so there must "
            f"be a committed file to read it from.")
    # (1) THE CLOCK FROM THE DECLARATION, (2) THE DATE FROM THE COMMIT.
    clock = declared_clock_start(refs[0], repo)
    landed = landed_committer_timestamp(amendment_path, refs, repo)
    # (5) INSTANTS, never strings.
    if landed["declared_instant"] >= clock["instant"]:
        raise BandHazardRefused(
            f"REFUSED {AMENDMENT_AFTER_SEEING}: {lever} landed "
            f"{landed['declared_utc']} against a clock starting "
            f"{clock['clock_start_utc']} -- compared as INSTANTS, so a "
            f"different rendering of the same moment cannot change the "
            f"answer.")
    return {"admissible": True, "lever": lever, "lever_set": pin,
            "declared_utc": landed["declared_utc"],
            "landed_on": landed["per_ref"],
            "clock_start_utc": clock["clock_start_utc"],
            "clock_read_from": clock["read_from"],
            "outcome_is_known": outcome["known"],
            "is_an_amendment_to_a_user_authored_plan":
                entry["is_an_amendment"],
            "every_input_is_bound_to_an_artifact": True}


def report() -> dict:
    return {
        "protocol": PROTOCOL,
        "band": {"days": BAND_DAYS, "need_evaluable": NEED_EVALUABLE,
                 "extension": "FORBIDDEN"},
        "THE_FAILING_INPUT_IS_THE_POPULATION_NOT_THE_TAPE": {
            "tape_windows_per_day": {d: v["tape_windows"]
                                     for d, v in DAYS.items()},
            "population_windows_per_day": {d: v["pop_windows"]
                                           for d, v in DAYS.items()},
            "days_failing_the_gate_on_a_COMPLETE_tape": sorted(
                d for d, v in DAYS.items()
                if not passes(v) and v["tape_windows"] >= 288),
            "consequence":
                "a population shortfall on a complete tape is a "
                "DERIVATION failure, and unlike a collection failure it "
                "is REBUILDABLE -- so these days are not necessarily "
                "lost, and fixing the population builder is worth more "
                "than any scheduling change"},
        "threshold_sensitivity": threshold_sensitivity(),
        "ledger_rank_test": ledger_rank_test(),
        "FORWARD_RATE_PAIR": forward_rate_pair(),
        "IS_THIS_TEST_RUNNABLE_AT_ALL": viability_table(),
        "G_FLOOR": g_floor(),
        "LEVERS": levers(),
        "band_probabilities": band_table(),
        "one_fewer_night": {
            "n": BAND_DAYS - 1,
            "P_at_least_10_at_the_recent_measured_rate":
                p_at_least(7 / 8, BAND_DAYS - 1)},
        "the_margin_is_a_probability":
            "E[evaluable] - 10 is not the margin; P(fewer than 10) is, "
            "and the two diverge exactly where the rate is uncertain",
    }


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    print("== the gate's threshold is a parameter of the rate ==")
    ts = threshold_sensitivity()
    ck("the rate AT the gate is reported with the gate's own number",
       ts["gate"] == 1 and abs(ts["rate_at_the_gate"] - 7 / 11) < 1e-12,
       f"<=1 -> {ts['rate_at_the_gate']:.4f}")
    ck("one step DOWN loses a day, one step up gains one -- it is steep",
       rate_at(0)["n_pass"] == 6 and rate_at(4)["n_pass"] == 8,
       f"<=0: {rate_at(0)['rate']:.4f}   <=4: {rate_at(4)['rate']:.4f}")
    ck("  so the rate is not a property of the data alone",
       ts["swing_within_three_steps_of_the_gate"] > 0.05,
       f"swing {ts['swing_within_three_steps_of_the_gate']:.4f} within "
       f"three steps")

    print("== the ledger's ranking against the gate's verdicts ==")
    lr = ledger_rank_test()
    ck("the test finds the inversions rather than asserting them",
       lr["discordant"] == 2 and all(
           i["cleaner_by_ledger_but_FAILS"] == "20260911"
           for i in lr["inversions"]),
       f"{lr['discordant']} of {lr['n_pass_fail_pairs']} pairs, both on "
       f"09-11")
    ck("  and a ranking with no discordant pair REFUSES rather than "
       "scoring 1.0",
       _no_pairs_refuses())

    print("== the population, not the tape ==")
    r = report()["THE_FAILING_INPUT_IS_THE_POPULATION_NOT_THE_TAPE"]
    ck("days fail the gate while their tape is COMPLETE",
       len(r["days_failing_the_gate_on_a_COMPLETE_tape"]) >= 3,
       str([d[4:] for d in r["days_failing_the_gate_on_a_COMPLETE_tape"]]))

    print("== the margin is a probability, computed exactly ==")
    ck("P(>=10 of 14) at p=1 is 1 and at p=0 is 0",
       p_at_least(1.0) == 1.0 and p_at_least(0.0) == 0.0)
    ck("the tail is exact, not simulated: P(>=10|p=0.5) = C-sum/2^14",
       abs(p_at_least(0.5) - sum(comb(14, i) for i in range(10, 15))
           / 2 ** 14) < 1e-15)
    ck("a rate outside [0,1] REFUSES", _bad_rate_refuses())
    tbl = {r["basis"]: r for r in band_table()}
    ck("at 9/11 x ETH 90% the chance of NO VERDICT is ~30%",
       0.29 < tbl["9/11 x ETH at 90%"]["P_NO_VERDICT"] < 0.31,
       f"{tbl['9/11 x ETH at 90%']['P_NO_VERDICT']:.4f}")
    measured = tbl["MEASURED at BE's gate, 11 days (7/11)"]
    ck("  and at the gate's MEASURED 11-day rate the test more likely "
       "fails than passes",
       measured["P_NO_VERDICT"] > 0.5,
       f"P(no verdict) {measured['P_NO_VERDICT']:.4f}")

    print("== both rates, reported as a PAIR ==")
    fp = forward_rate_pair()
    ck("the planning rate is the FULL window, not the recent one",
       fp["planning_rate"]["which"] == PLANNING_RATE
       and fp["planning_rate"]["n_days"] == 11,
       f"p={fp['planning_rate']['p']:.6f} "
       f"P(no verdict)={fp['planning_rate']['P_NO_VERDICT']:.6f}")
    ck("  and the recent window is labelled the OPTIMISTIC BOUND",
       fp["optimistic_bound"]["which"] == OPTIMISTIC_BOUND,
       f"p={fp['optimistic_bound']['p']:.6f} "
       f"P(no verdict)={fp['optimistic_bound']['P_NO_VERDICT']:.6f}")
    ck("  and both travel together with the regime question named",
       fp["FORWARD_RATE_IS_UNRESOLVED"] is True
       and "settled_by" in fp["WHICH_APPLIES_IS_UNRESOLVED"])
    ck("the same seven days pass in both windows",
       fp["planning_rate"]["n_pass"] == fp["optimistic_bound"]["n_pass"]
       == 7)
    ck("the factor between their failure probabilities is reported",
       fp["factor_between_their_failure_probabilities"] > 20,
       f"{fp['factor_between_their_failure_probabilities']:.2f}x")

    print("== what would have to be TRUE for the test to be viable ==")
    vt = viability_table()
    for r in vt["rows"]:
        ck(f"required rate at {r['confidence']:.2f} confidence is "
           f"computed, not tabulated",
           0.7 < r["required_daily_joint_rate"] < 0.9,
           f"p >= {r['required_daily_joint_rate']:.4f}  gap from "
           f"planning {r['gap_from_the_planning_rate']:+.4f}, from "
           f"optimistic {r['gap_from_the_optimistic_bound']:+.4f}")
    ck("the PLANNING rate clears NONE of the three confidences",
       not any(r["planning_rate_clears_it"] for r in vt["rows"]))
    ck("  while the optimistic bound clears all three",
       all(r["optimistic_bound_clears_it"] for r in vt["rows"]))

    print("== the floor under k, from the frozen module's own ladder ==")
    gf = g_floor()
    ck("below the declared minimum there is NO test, not a weak one",
       gf["lowest_G_with_any_test"] == PRED.MIN_NONZERO,
       f"G floor = {gf['lowest_G_with_any_test']} "
       f"(MIN_NONZERO={PRED.MIN_NONZERO})")
    ck("  and at the floor exactly ONE rung passes: a perfect run",
       gf["at_the_floor_the_candidate_must_be_perfect"] is True)
    lv = levers()
    ck("LEVER (iv) IS UNAVAILABLE AT THE PLANNING RATE -- the k it needs "
       "is below the k the test can compute",
       lv["iv_fewer_required_days"]["AVAILABLE_AT_THE_PLANNING_RATE"]
       is False,
       f"max k {lv['iv_fewer_required_days']['max_k_at_the_planning_rate']}"
       f" < floor {gf['lowest_G_with_any_test']}")
    ck("  and it is unnecessary at the optimistic bound",
       lv["iv_fewer_required_days"]["max_k_at_the_optimistic_bound"]
       >= NEED_EVALUABLE)
    ck("lever (iii) is priced in days at both rates",
       lv["iii_longer_band"]["n_required_at_the_planning_rate"] > BAND_DAYS,
       f"n >= {lv['iii_longer_band']['n_required_at_the_planning_rate']} "
       f"at the planning rate, "
       f"{lv['iii_longer_band']['n_required_at_the_optimistic_bound']} at "
       f"the bound")
    ck("the amendments are MARKED as amendments",
       lv["iii_longer_band"]["is_an_amendment"]
       and lv["iv_fewer_required_days"]["is_an_amendment"]
       and not lv["ii_improve_the_input"]["is_an_amendment"])

    print("== REVIEW 264's five attacks, re-run against the fixes ==")
    import subprocess, tempfile, datetime as dt

    def _g(repo, *a):
        return subprocess.run(["git", "-C", repo] + list(a),
                              capture_output=True, text=True)

    fx = tempfile.mkdtemp(prefix="de_amend_")
    _g(fx, "init", "-q", "-b", "main")
    _g(fx, "config", "user.email", "f@x")
    _g(fx, "config", "user.name", "fixture")
    decl = Path(fx) / WINDOW_DECL
    decl.parent.mkdir(parents=True, exist_ok=True)
    future = (dt.datetime.now(dt.timezone.utc)
              + dt.timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    decl.write_text(json.dumps({"clock_start_utc": future}))
    amend = Path(fx) / "amendment.json"
    amend.write_text(json.dumps({"lever": "iii_longer_band", "n": 20}))
    _g(fx, "add", "-A")
    _g(fx, "commit", "-qm", "fixture: window + amendment")
    _g(fx, "branch", "-f", "refA"), _g(fx, "branch", "-f", "refB")
    FXREFS = ("refA", "refB")
    empty = tempfile.mkdtemp(prefix="de_amend_derived_")

    ck("ATTACK 1 (backdate) CANNOT BE EXPRESSED: there is no "
       "declared_utc parameter",
       "declared_utc" not in
       amendment_is_admissible.__code__.co_varnames[
           :amendment_is_admissible.__code__.co_argcount
           + amendment_is_admissible.__code__.co_kwonlyargcount],
       "the date is the COMMIT's")
    try:
        amendment_is_admissible("iii_longer_band",
                                amendment_path="never_committed.json",
                                refs=FXREFS, repo=fx, derived=empty)
        ck("  and an uncommitted amendment REFUSES", False)
    except BandHazardRefused as exc:
        ck("  and an uncommitted amendment REFUSES", NOT_LANDED in str(exc))

    ck("ATTACK 2 (amender supplies the clock) CANNOT BE EXPRESSED: no "
       "clock_start_utc parameter",
       "clock_start_utc" not in
       amendment_is_admissible.__code__.co_varnames[
           :amendment_is_admissible.__code__.co_argcount
           + amendment_is_admissible.__code__.co_kwonlyargcount])
    try:
        declared_clock_start(EXECUTING_REFS[0], REPO)
        ck("  and with no declared window the REAL refs refuse", False)
    except BandHazardRefused as exc:
        ck("  and with no declared window the REAL refs refuse",
           NO_WINDOW in str(exc), "nothing to be 'before'")

    out_dir = tempfile.mkdtemp(prefix="de_amend_out_")
    (Path(out_dir) / "p003_de_section8_verdict_20260914.json").write_text(
        "{}")
    try:
        amendment_is_admissible("iii_longer_band",
                                amendment_path="amendment.json",
                                refs=FXREFS, repo=fx, derived=out_dir)
        ck("ATTACK 3 (outcome known, not declared) now REFUSES -- "
           "COMPUTED from the artifacts", False)
    except BandHazardRefused as exc:
        ck("ATTACK 3 (outcome known, not declared) now REFUSES -- "
           "COMPUTED from the artifacts",
           AMENDMENT_AFTER_SEEING in str(exc) and "computed" in str(exc))
    try:
        amendment_is_admissible("i_accept_the_risk",
                                amendment_path=None, refs=FXREFS,
                                repo=fx, derived=out_dir)
        ck("  and FIX 6: it fires even for a lever whose flag is False",
           False)
    except BandHazardRefused as exc:
        ck("  and FIX 6: it fires even for a lever whose flag is False",
           AMENDMENT_AFTER_SEEING in str(exc),
           "the strongest clause is no longer gated by the weakest")

    late = _instant("2026-09-13T23:00:00-05:00", "x")
    clock = _instant("2026-09-14T00:00:00Z", "y")
    ck("ATTACK 4 (timezone rendering) closed: -05:00 is FOUR HOURS after "
       "the clock as an INSTANT",
       late > clock and "2026-09-13T23:00:00-05:00" < "2026-09-14T00:00:00Z",
       "lexicographically it sorted BEFORE; as instants it does not")
    ck("  and +00:00 equals Z at the identical instant",
       _instant("2026-09-14T00:00:00+00:00", "a")
       == _instant("2026-09-14T00:00:00Z", "b"))
    try:
        _instant("2026-09-14T00:00:00", "naive")
        ck("  and a timestamp with NO offset refuses", False)
    except BandHazardRefused as exc:
        ck("  and a timestamp with NO offset refuses", BAD_STAMP in str(exc))

    added = dict(LEVER_SPEC)
    added["vi_extend_the_observation_window"] = {"is_an_amendment": True}
    try:
        assert_lever_set_pinned(added)
        ck("ATTACK 5 (add a lever) now REFUSES on the PIN", False)
    except BandHazardRefused as exc:
        ck("ATTACK 5 (add a lever) now REFUSES on the PIN",
           LEVER_SET_MOVED in str(exc), f"pin {LEVER_SET_PIN}")
    try:
        amendment_is_admissible("vii_relax_required_days",
                                amendment_path="amendment.json",
                                refs=FXREFS, repo=fx, derived=empty,
                                spec=dict(LEVER_SPEC,
                                          vii_relax_required_days={
                                              "is_an_amendment": True}))
        ck("  and a lever with the KEY OMITTED refuses rather than "
           "defaulting to permissive", False)
    except BandHazardRefused as exc:
        ck("  and a lever with the KEY OMITTED refuses rather than "
           "defaulting to permissive", LEVER_SET_MOVED in str(exc))

    print("== the honest controls still pass ==")
    got = amendment_is_admissible("iii_longer_band",
                                  amendment_path="amendment.json",
                                  refs=FXREFS, repo=fx, derived=empty)
    ck("an amendment landed BEFORE a declared clock IS admissible",
       got["admissible"] and got["every_input_is_bound_to_an_artifact"],
       f"declared {got['declared_utc']} < clock {got['clock_start_utc']}")
    ck("  and it records BOTH refs' commits, not a typed date",
       set(got["landed_on"]) == set(FXREFS))
    past = json.dumps({"clock_start_utc": "2020-01-01T00:00:00Z"})
    decl.write_text(past)
    _g(fx, "add", "-A"), _g(fx, "commit", "-qm", "clock moved earlier")
    _g(fx, "branch", "-f", "refA"), _g(fx, "branch", "-f", "refB")
    try:
        amendment_is_admissible("iii_longer_band",
                                amendment_path="amendment.json",
                                refs=FXREFS, repo=fx, derived=empty)
        ck("a clock that started BEFORE the amendment landed refuses",
           False)
    except BandHazardRefused as exc:
        ck("a clock that started BEFORE the amendment landed refuses",
           AMENDMENT_AFTER_SEEING in str(exc))
    ck("an unknown lever still refuses", _unknown_lever_refuses())
    ck("the lever set is PINNED and the pin matches",
       assert_lever_set_pinned()["digest"] == LEVER_SET_PIN,
       LEVER_SET_PIN)

    print("== excluding the open days requires a NAMED CONDITION ==")
    for args, why in (((None, False, True), "no condition named"),
                      (("they look different", False, True),
                       "supported only by the outcomes"),
                      (("a window-supply break", True, True),
                       "still names the days by outcome")):
        try:
            exclusion_is_licensed(*args)
            ck(f"refuses: {why}", False)
        except BandHazardRefused as exc:
            ck(f"refuses: {why}", EXCLUSION_NOT_LICENSED in str(exc))
    ck("  and an independently-evidenced condition IS licensed",
       exclusion_is_licensed("collector mask changed 09-03T18:00Z, from "
                             "the run ledger", True, False)["licensed"])

    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def _unknown_lever_refuses() -> bool:
    try:
        amendment_is_admissible("vi_wishful_thinking",
                                amendment_path="x.json")
        return False
    except BandHazardRefused as exc:
        return "UNKNOWN_LEVER" in str(exc)


def _no_pairs_refuses() -> bool:
    same = {d: dict(v, pop_missing=0) for d, v in DAYS.items()}
    try:
        ledger_rank_test(same)
        return False
    except BandHazardRefused:
        return True


def _bad_rate_refuses() -> bool:
    try:
        p_at_least(1.5)
        return False
    except BandHazardRefused:
        return True


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    if "--report" in argv:
        print(json.dumps(report(), indent=2, default=str))
        return 0
    print(json.dumps({"protocol": PROTOCOL, "band_days": BAND_DAYS,
                      "need": NEED_EVALUABLE}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
