"""THE DECISION ARTIFACT: one document, every number attributed.

This is the state a person decides on before spending fourteen nights.
It is built here rather than assembled from four seats' filings, and it
carries one discipline the assembling would have lost:

  EVERY NUMBER CARRIES THE POPULATION AND THE CRITERION THAT PRODUCED
  IT, and that is ENFORCED -- `assert_attributed` walks the document and
  refuses any bare numeric leaf.

The reason is specific rather than general. Tonight's largest error was a
pass rate quoted WITHOUT its criterion: 10 of 11 is true of raw-tape
window-file presence and false of BE's own population gate, which scores
7 of 11 over the same days. It survived three retellings. In a document
someone decides on, that error would be permanent.

Nothing here recommends. Several levers are amendments to a user-authored
plan and none of them is this seat's to choose; the one factual property
worth placing beside them -- that improving the input is the only lever
which does not trade the test's strength for its feasibility -- is stated
as a property, not as advice.

Usage:  de_band_decision.py --falsify
        de_band_decision.py --emit [--out PATH]
"""
from __future__ import annotations

CALL_SITE = {
    "kind": "ARTIFACT_MEDIATED",
    "by": "the decision is read from the artifact this emits",
    "artifact": "band_decision.json",
    "gates": "its own emit -- assert_attributed refuses a bare number, so an unattributed rate cannot reach the artifact a person decides from",
}

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import de_band_hazard as HZ                            # noqa: E402

PROTOCOL = "P003_DE_BAND_DECISION_V1"
UNATTRIBUTED = "NUMBER_WITHOUT_ITS_POPULATION_AND_CRITERION"

#: The criteria, named once. A rate is meaningless without one of these
#: beside it, and two of them disagree on the same eleven days.
C_GATE = ("BE gate: population(day) interior-missing windows <= 1, btc "
          "(be_build_preflight.check_day window-supply row)")
C_TAPE = ("raw-tape 5-minute window FILES present, btc AND eth "
          "(data/pm_5min/raw/<day>/)")
C_UNVERIFIED = ("UNVERIFIED -- reported by another seat; its definition "
                "was not found in any landed artifact (COORDINATION.md, "
                "HANDOFF.md, RESULTS.md, reviews/)")
C_BINOMIAL = ("exact binomial P(X >= 10 | n = 14, p), no simulation")
C_LADDER = ("the frozen de_fair_value_predictive ladder and MIN_NONZERO")

P_11 = "09-01..09-11, 11 consecutive UTC days"
P_8 = "09-04..09-11, 8 consecutive UTC days"
P_BAND = "a hypothetical 14-day band, 10 evaluable required"


class DecisionRefused(ValueError):
    """The artifact cannot be emitted as declared."""


def M(value, *, population: str, criterion: str, source: str,
      as_of: str = None, note: str = None) -> dict:
    """ONE NUMBER, WITH THE TWO THINGS THAT GIVE IT MEANING."""
    out = {"value": value, "population": population,
           "criterion": criterion, "source": source}
    if as_of:
        out["as_of"] = as_of
    if note:
        out["note"] = note
    return out


def _is_measure(node) -> bool:
    return (isinstance(node, dict)
            and {"value", "population", "criterion"} <= set(node))


def assert_attributed(doc, path: str = "$") -> int:
    """EVERY NUMERIC LEAF SITS INSIDE A MEASURE, or this refuses.

    Booleans are not measurements and are allowed; a number is not.
    """
    n = 0
    if _is_measure(doc):
        return 1
    if isinstance(doc, dict):
        for k, v in doc.items():
            n += assert_attributed(v, f"{path}.{k}")
        return n
    if isinstance(doc, (list, tuple)):
        for i, v in enumerate(doc):
            n += assert_attributed(v, f"{path}[{i}]")
        return n
    if isinstance(doc, bool) or doc is None or isinstance(doc, str):
        return 0
    if isinstance(doc, (int, float)):
        raise DecisionRefused(
            f"REFUSED {UNATTRIBUTED}: {path} = {doc!r} is a bare number. "
            f"A rate without its criterion is the error that survived "
            f"three retellings tonight -- 10 of 11 is true of tape "
            f"window-file presence and false of BE's population gate on "
            f"the same eleven days.")
    return n


#: THE AMENDMENT DECLARATIONS this artifact expects, by lever. Naming
#: them here is what gives the guard something to date: an amendment is
#: dated by the COMMIT that lands its file, so the file must be named
#: before it can be dated.
AMENDMENT_FILES = {
    "iii_longer_band":
        "live/pm_research/declarations/de_amendment_longer_band_v1.json",
    "iv_fewer_required_days":
        "live/pm_research/declarations/de_amendment_fewer_days_v1.json",
    "v_start_after_a_clean_run":
        "live/pm_research/declarations/de_amendment_clean_start_v1.json",
}


#: Modules that MENTION tier2 but do not read it (a docstring naming the
#: rule is not a read). Named explicitly so the coverage figure is not
#: quietly flattered, and so the list itself is auditable.
TIER2_PROSE_ONLY = ("de_fair_value_plumbing_run.py",)


def rule34a_prerequisite(root: Path = None) -> dict:
    """RULE 34a's FENCE, AS A BLOCKING PREREQUISITE WITH A COMPUTED STATE.

    The coordinator ruled the fence is due against THE BAND, not the
    morning. So the gap is declared now and its state is RECOMPUTED on
    every emit: when the call sites land, this flips to met without
    anyone editing a sentence.

    The three properties are DA's, from their fence audit. Property 1 is
    already met by da_rule34a_fence (DA 307); what is open is property 2.
    """
    root = Path(root or HERE)
    touch, guarded = [], []
    for f in sorted(root.glob("*.py")):
        src = f.read_text()
        if "tier2" not in src:
            continue
        touch.append(f.name)
        if ("da_rule34a_fence" in src or "is_protected" in src
                or "assert_not_protected" in src):
            guarded.append(f.name)
    reads = [f for f in touch if f not in TIER2_PROSE_ONLY]
    unguarded = [f for f in reads if f not in guarded]
    fence_exists = (root / "da_rule34a_fence.py").is_file()
    return {
        "prerequisite": "RULE 34a MUST BE ENFORCED BEFORE THE BAND OPENS",
        "due_against": "the band, not the morning -- the rule exists to "
                       "protect the validation population, and the band "
                       "cannot start until the effect floor is set and "
                       "two-coin production is demonstrated",
        "BLOCKING": True,
        "met": bool(fence_exists and not unguarded),
        "properties": {
            "1_protected_set_derived_from_an_artifact": {
                "met": fence_exists,
                "by": "da_rule34a_fence parses the floor from the "
                      "procedure documents (DA 307)"},
            "2_a_call_site_on_every_tier2_read_path": {
                "met": not unguarded,
                "modules_touching_tier2": touch,
                "prose_mention_only": list(TIER2_PROSE_ONLY),
                "read_paths": reads,
                "guarded": guarded,
                "UNGUARDED": unguarded},
            "3_falsifier_refuses_a_protected_read_AND_admits_an_"
            "unprotected_one": {
                "met": fence_exists,
                "by": "da_rule34a_fence.is_protected / "
                      "assert_not_protected, both driven"},
        },
        "state": ("THE FENCE EXISTS AND IS SOUND; IT IS UNWIRED ON THE "
                  "READ PATHS" if fence_exists and unguarded else
                  "ENFORCED" if fence_exists else "NO FENCE EXISTS"),
        "recomputed_on_every_emit": True,
        "why_not_built_tonight":
            "a fence built hastily to guard a population that does not "
            "yet exist is how a guard acquires the defects this "
            "programme spent the night removing",
    }


def amendment_status() -> dict:
    """THE CALL SITE (REVIEW 264 fix 7).

    The guard is consulted HERE, in the document a person decides on, for
    every lever that must be declared before the clock -- so its verdict
    is a field of the decision rather than a function nobody calls. Today
    every one of them refuses, because no validation window is declared:
    there is no clock to be before.
    """
    out = {}
    for lever, path in AMENDMENT_FILES.items():
        try:
            got = HZ.amendment_is_admissible(lever, amendment_path=path)
            out[lever] = {"admissible": True,
                          "declared_utc": got["declared_utc"],
                          "clock_start_utc": got["clock_start_utc"],
                          "clock_read_from": got["clock_read_from"],
                          "expected_declaration": path}
        except HZ.BandHazardRefused as exc:
            head = str(exc).split(":")[0].replace("REFUSED ", "").strip()
            out[lever] = {"admissible": False, "refusal": head,
                          "expected_declaration": path,
                          "detail": str(exc)[:200]}
    return {"consulted": "de_band_hazard.amendment_is_admissible",
            "when": "every time this artifact is emitted",
            "per_lever": out,
            "why_this_is_here":
                "REVIEW 264: a guard with cells and no call site "
                "constrains nothing; its verdict now travels in the "
                "document the decision is made from"}


def decision_artifact(as_of: str) -> dict:
    pair = HZ.forward_rate_pair()
    via = HZ.viability_table()
    lv = HZ.levers()
    floor = HZ.g_floor()
    src = "de_band_hazard (this seat, measured 2026-09-12)"

    def rate(block, population):
        return {
            "pass_rate": M(block["p"], population=population,
                           criterion=C_GATE, source=src),
            "days_passing": M(block["n_pass"], population=population,
                              criterion=C_GATE, source=src),
            "days_in_window": M(block["n_days"], population=population,
                                criterion=C_GATE, source=src),
            "expected_evaluable_in_14": M(
                block["expected_evaluable"], population=P_BAND,
                criterion=C_BINOMIAL, source=src),
            "P_at_least_10": M(block["P_at_least_10"], population=P_BAND,
                               criterion=C_BINOMIAL, source=src),
            "P_NO_VERDICT": M(block["P_NO_VERDICT"], population=P_BAND,
                              criterion=C_BINOMIAL, source=src),
            "failing_days": block["failing_days"],
        }

    return {
        "protocol": PROTOCOL,
        "as_of": as_of,
        "what_this_is": "the state a person decides on before committing "
                        "fourteen nights; nothing here recommends",
        "EVERY_NUMBER_CARRIES_ITS_POPULATION_AND_CRITERION": True,
        "why_that_discipline":
            "a pass rate quoted without its criterion survived three "
            "retellings tonight: 10 of 11 is true of raw-tape window-file "
            "presence and false of BE's population gate, which scores 7 "
            "of 11 over the same days",

        "THE_TWO_RATES": {
            "FORWARD_RATE_IS_UNRESOLVED": True,
            "planning_rate": rate(pair["planning_rate"], P_11),
            "optimistic_bound": rate(pair["optimistic_bound"], P_8),
            "the_same_seven_days_pass_in_both": True,
            "factor_between_failure_probabilities": M(
                pair["factor_between_their_failure_probabilities"],
                population=P_BAND, criterion=C_BINOMIAL, source=src,
                note="like for like at the SAME criterion; pairing the "
                     "tape criterion against the gate criterion gives "
                     "4.08x instead, which is how a factor gets "
                     "misquoted"),
            "for_contrast_the_OTHER_criterion": {
                "tape_window_files_11_days": M(
                    9 / 11, population=P_11, criterion=C_TAPE, source=src,
                    note="09-03 is short by one window on both coins"),
                "reported_by_another_seat": M(
                    10 / 11, population=P_11, criterion=C_UNVERIFIED,
                    source="REV, via the coordinator",
                    note="reproduces the tape criterion at a threshold of "
                         "zero missing interior windows; its definition "
                         "was not found in a landed artifact"),
            },
            "THE_REGIME_QUESTION": {
                "open_days": ["20260901", "20260902", "20260903"],
                "population_windows": {
                    d: M(HZ.DAYS[d]["pop_windows"], population=f"{d}, btc",
                         criterion=C_GATE, source=src)
                    for d in ("20260901", "20260902", "20260903")},
                "their_tape_was_complete": True,
                "settled_by": "classify 09-11: the LAST of the old "
                              "failures, or the FIRST of a new one",
                "exclusion_status": "NO CONDITION LICENSES AN EXCLUSION",
                "REV_searched_and_found_none":
                    "instrument, era and supply searched; REV named what "
                    "it did NOT read -- the mask producer source, "
                    "da_content_liveness_rule's implementation, per-window "
                    "content measurements, host metrics -- and the "
                    "direction its bias would push",
                "DA_holds_those_residuals_and_is_looking": True,
                "therefore":
                    "unless DA names a since-changed condition "
                    "independently of these days' outcomes, the planning "
                    "rate is the number and the optimistic bound is a "
                    "bound",
            },
        },

        "IS_IT_VIABLE": {
            "requirement": {
                "band_days": M(HZ.BAND_DAYS, population=P_BAND,
                               criterion="§8 as declared", source=src),
                "evaluable_needed": M(HZ.NEED_EVALUABLE,
                                      population=P_BAND,
                                      criterion="§8 as declared",
                                      source=src),
                "extension": "FORBIDDEN by §8",
            },
            "minimum_daily_joint_rate": [
                {"confidence": M(r["confidence"], population=P_BAND,
                                 criterion=C_BINOMIAL, source=src),
                 "required_rate": M(r["required_daily_joint_rate"],
                                    population=P_BAND,
                                    criterion=C_BINOMIAL, source=src),
                 "gap_from_the_planning_rate": M(
                     r["gap_from_the_planning_rate"], population=P_11,
                     criterion=f"{C_BINOMIAL} vs {C_GATE}", source=src),
                 "planning_rate_clears_it": r["planning_rate_clears_it"],
                 "gap_from_the_optimistic_bound": M(
                     r["gap_from_the_optimistic_bound"], population=P_8,
                     criterion=f"{C_BINOMIAL} vs {C_GATE}", source=src),
                 "optimistic_bound_clears_it":
                     r["optimistic_bound_clears_it"]}
                for r in via["rows"]],
            "reading":
                "the planning rate clears none of the three "
                "confidences; the optimistic bound clears all three, so "
                "the viability question IS the regime question",
        },

        "THE_FLOOR_UNDER_k": {
            "lowest_G_with_any_test": M(
                floor["lowest_G_with_any_test"], population=P_BAND,
                criterion=C_LADDER, source=src),
            "below_it": "the exact sign test returns INSUFFICIENT_"
                        "EVIDENCE and there is no p to correct",
            "at_G_9_and_G_8": "exactly ONE rung passes, so the candidate "
                              "must be positive on EVERY day",
        },

        "THE_LEVERS": {
            "i_accept_the_risk": {
                "cost_at_the_planning_rate": M(
                    lv["i_accept_the_risk"]["cost"]["at_the_planning_rate"],
                    population=P_BAND, criterion=C_BINOMIAL, source=src),
                "cost_at_the_optimistic_bound": M(
                    lv["i_accept_the_risk"]["cost"][
                        "at_the_optimistic_bound"],
                    population=P_BAND, criterion=C_BINOMIAL, source=src),
                "weakens": "nothing -- the test stays as declared",
                "is_an_amendment_the_user_alone_may_make": False},
            "ii_improve_the_input": {
                "required_rate_at_0_90": M(
                    lv["ii_improve_the_input"][
                        "required_rate_at_this_confidence"],
                    population=P_BAND, criterion=C_BINOMIAL, source=src),
                "gap_from_the_planning_rate": M(
                    lv["ii_improve_the_input"]["gap_from_the_planning_rate"],
                    population=P_11, criterion=f"{C_BINOMIAL} vs {C_GATE}",
                    source=src),
                "cost": "unknown until the mask-collapse cause is named; "
                        "may be unavailable",
                "weakens": "nothing -- it changes the INPUT, not the test",
                "is_an_amendment_the_user_alone_may_make": False,
                "A_FACTUAL_PROPERTY_NOT_A_RECOMMENDATION":
                    "this is the only lever that does not trade the "
                    "test's strength for its feasibility"},
            "iii_longer_band": {
                "days_required_at_the_planning_rate": M(
                    lv["iii_longer_band"]["n_required_at_the_planning_rate"],
                    population=P_11, criterion=C_BINOMIAL, source=src),
                "days_required_at_the_optimistic_bound": M(
                    lv["iii_longer_band"][
                        "n_required_at_the_optimistic_bound"],
                    population=P_8, criterion=C_BINOMIAL, source=src),
                "cost": "calendar time; every added day is a day the "
                        "candidates are not yet judged",
                "weakens": "nothing statistically -- k is unchanged",
                "is_an_amendment_the_user_alone_may_make": True,
                "must_be_declared_BEFORE_the_clock": True},
            "iv_fewer_required_days": {
                "largest_k_at_the_planning_rate": M(
                    lv["iv_fewer_required_days"][
                        "max_k_at_the_planning_rate"],
                    population=P_BAND, criterion=C_BINOMIAL, source=src),
                "smallest_k_the_test_can_compute": M(
                    lv["iv_fewer_required_days"][
                        "lowest_k_the_TEST_can_compute"],
                    population=P_BAND, criterion=C_LADDER, source=src),
                "AVAILABLE_AT_THE_PLANNING_RATE": lv[
                    "iv_fewer_required_days"][
                    "AVAILABLE_AT_THE_PLANNING_RATE"],
                "cost": "the test's resolution: at G=10 two rungs pass, "
                        "at G=9 and G=8 exactly one",
                "weakens": "the test itself, and the multiplicity "
                           "arithmetic with it -- Holm's threshold does "
                           "not move, so a smaller G spends the same "
                           "alpha on a coarser ladder",
                "is_an_amendment_the_user_alone_may_make": True,
                "must_be_declared_BEFORE_the_clock": True},
            "v_start_after_a_clean_run": {
                "cost": "calendar time, and the clean run consumes days "
                        "that cannot later be in the band",
                "weakens": "nothing in the test; it buys the RATE by "
                           "choosing when to start",
                "is_an_amendment_the_user_alone_may_make": False,
                "must_be_declared_BEFORE_the_clock": True,
                "caution": "the start condition must be a declared "
                           "PREDICATE, or 'it looked clean' becomes the "
                           "selection"},
        },

        "AMENDMENT_ADMISSIBILITY_NOW": amendment_status(),
        "BLOCKING_PREREQUISITES": {
            "rule_34a_fence": rule34a_prerequisite()},

        "THE_TRAP": {
            "which_levers": ["iii_longer_band", "iv_fewer_required_days"],
            "why": "both are what a disappointed operator reaches for "
                   "AFTER a shortfall, when the shortfall itself is the "
                   "information being used",
            "therefore": "if either is to be available it must be "
                         "DECLARED NOW, while the outcome is unknown",
            "enforced_by": "de_band_hazard.amendment_is_admissible, which "
                           "REFUSES a declaration with no timestamps, one "
                           "timestamped after the clock start, and any "
                           "consideration once the outcome is known",
            "this_is_a_field_not_advice": True},
    }


def emit(path=None, as_of: str = None) -> dict:
    import datetime as dt
    as_of = as_of or dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = decision_artifact(as_of)
    n = assert_attributed(doc)
    doc["n_attributed_measures"] = M(
        n, population="this artifact", criterion="assert_attributed walk",
        source="de_band_decision")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(doc, indent=2, default=str))
    return doc


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    print("== every number carries its population and criterion ==")
    doc = decision_artifact("2026-09-12T00:00:00Z")
    count = assert_attributed(doc)
    ck("the artifact passes the attribution walk",
       count > 25, f"{count} attributed measures")
    ck("  and a PLANTED bare number is caught -- positive control",
       _planted_bare_number_refuses())
    ck("  while a boolean is not a measurement and is allowed",
       assert_attributed({"flag": True, "why": "text"}) == 0)

    print("== the two rates, as a pair, with the criterion attached ==")
    tr = doc["THE_TWO_RATES"]
    ck("both rates carry the SAME criterion, so they are comparable",
       tr["planning_rate"]["pass_rate"]["criterion"]
       == tr["optimistic_bound"]["pass_rate"]["criterion"] == C_GATE)
    ck("  and they name DIFFERENT populations",
       tr["planning_rate"]["pass_rate"]["population"] != tr[
           "optimistic_bound"]["pass_rate"]["population"])
    ck("the OTHER criterion is carried beside them, not instead",
       tr["for_contrast_the_OTHER_criterion"]["tape_window_files_11_days"][
           "criterion"] == C_TAPE)
    ck("  and the unverified outside number is MARKED unverified",
       "UNVERIFIED" in tr["for_contrast_the_OTHER_criterion"][
           "reported_by_another_seat"]["criterion"])
    ck("the regime question is named and UNRESOLVED",
       tr["FORWARD_RATE_IS_UNRESOLVED"] is True
       and tr["THE_REGIME_QUESTION"]["exclusion_status"].startswith("NO"))
    ck("  and REV's search and what it did NOT read are recorded",
       "did NOT read" in tr["THE_REGIME_QUESTION"][
           "REV_searched_and_found_none"])

    print("== THE CALL SITE (REVIEW 264 fix 7) ==")
    st = doc["AMENDMENT_ADMISSIBILITY_NOW"]
    ck("the guard is consulted from ANOTHER module, in the emitted "
       "artifact",
       st["consulted"] == "de_band_hazard.amendment_is_admissible"
       and set(st["per_lever"]) == set(AMENDMENT_FILES))
    ck("  and today every amendment lever REFUSES -- no clock exists yet",
       all(not v["admissible"] for v in st["per_lever"].values()),
       str(sorted({v["refusal"] for v in st["per_lever"].values()})))
    ck("  each naming the declaration file that would date it",
       all(v["expected_declaration"].endswith(".json")
           for v in st["per_lever"].values()))

    print("== rule 34a's fence as a BLOCKING prerequisite ==")
    pr = doc["BLOCKING_PREREQUISITES"]["rule_34a_fence"]
    ck("it is declared BLOCKING and its state is COMPUTED, not written",
       pr["BLOCKING"] is True and pr["recomputed_on_every_emit"] is True)
    ck("property 1 is MET -- the set is artifact-derived (DA 307)",
       pr["properties"]["1_protected_set_derived_from_an_artifact"]["met"]
       is True)
    p2 = pr["properties"]["2_a_call_site_on_every_tier2_read_path"]
    ck("property 2 is OPEN and NAMES the unguarded read paths",
       p2["met"] is False and len(p2["UNGUARDED"]) >= 1,
       str(p2["UNGUARDED"]))
    ck("  and a PROSE mention is not counted as a read path",
       set(p2["prose_mention_only"]) <= set(p2["modules_touching_tier2"])
       and not set(p2["prose_mention_only"]) & set(p2["read_paths"]),
       "a docstring naming the rule is not a read")
    ck("the prerequisite is NOT met while a read path is unguarded",
       pr["met"] is False, pr["state"])
    v = doc["IS_IT_VIABLE"]["minimum_daily_joint_rate"]
    ck("three confidences, each with both gaps",
       len(v) == 3 and all("gap_from_the_planning_rate" in r for r in v))
    ck("  the planning rate clears none of them",
       not any(r["planning_rate_clears_it"] for r in v))
    ck("lever (iv)'s unavailability is a FIELD, not a sentence",
       doc["THE_LEVERS"]["iv_fewer_required_days"][
           "AVAILABLE_AT_THE_PLANNING_RATE"] is False)
    ck("the amendments are marked as the user's alone",
       doc["THE_LEVERS"]["iii_longer_band"][
           "is_an_amendment_the_user_alone_may_make"]
       and doc["THE_LEVERS"]["iv_fewer_required_days"][
           "is_an_amendment_the_user_alone_may_make"]
       and not doc["THE_LEVERS"]["ii_improve_the_input"][
           "is_an_amendment_the_user_alone_may_make"])
    ck("  and (ii)'s property is stated as a PROPERTY, not advice",
       "NOT_A_RECOMMENDATION" in json.dumps(
           doc["THE_LEVERS"]["ii_improve_the_input"]))
    ck("the trap travels in the artifact and names its enforcement",
       doc["THE_TRAP"]["this_is_a_field_not_advice"] is True
       and "REFUSES" in doc["THE_TRAP"]["enforced_by"])

    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def _planted_bare_number_refuses() -> bool:
    doc = decision_artifact("2026-09-12T00:00:00Z")
    doc["THE_TWO_RATES"]["a_helpful_summary_rate"] = 0.75
    try:
        assert_attributed(doc)
        return False
    except DecisionRefused as exc:
        return UNATTRIBUTED in str(exc)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    if "--emit" in argv:
        out = argv[argv.index("--out") + 1] if "--out" in argv else None
        doc = emit(out)
        print(json.dumps(doc, indent=2, default=str))
        return 0
    print(json.dumps({"protocol": PROTOCOL}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
