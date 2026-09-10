#!/usr/bin/env python3
"""DA 191 -- THE ASYMMETRY NULL, EVALUATED AS PREDICATES.

`declarations/da_asymmetry_null_declaration_v1.json` declares the null that
separates TAIL-CLIPPING from DE-LEVERING. This module evaluates every clause
of it as a predicate, because a declaration that is only prose beside a table
is the defect this programme keeps paying for (rule 10).

It also implements the STATISTIC itself, so the property that makes the whole
design work -- that pure de-levering scores EXACTLY ZERO -- is driven rather
than asserted (rule 15).

Nothing here draws. `STATUS` is DECLARED-NOT-RUN and this module never runs
the null; DE does, after this file is committed.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECL = HERE / "declarations" / "da_asymmetry_null_declaration_v1.json"
PARAMS = HERE / "declarations" / "de_multiday_gate1_params_v29.json"

HOLDS = "HOLDS"
FAILS = "FAILS"

RULE_6_FLOOR = 200
MATCH_KEYS = ("cancel_count", "side", "hour")
FORBIDDEN_STATISTIC = ("mean", "day_mean", "net_value_cents")


def load(path=None) -> dict:
    return json.loads(Path(path or DECL).read_text())


def _v(ok: bool, **kw) -> dict:
    return {"verdict": HOLDS if ok else FAILS, **kw}


# --------------------------------------------------------------- statistic

def tail_masses(book: dict) -> tuple:
    """(positive mass, negative mass) of a {window: cents} map."""
    pos = sum(v for v in book.values() if v > 0)
    neg = sum(v for v in book.values() if v < 0)
    return pos, neg


def asymmetry(arm: dict, baseline: dict, mode: str = "OWN_SIGN") -> dict:
    """A = ret_pos - ret_neg. THE tested statistic.

    OWN_SIGN      -- each book partitioned by its own per-window sign.
    BASELINE_SIGN -- the partition fixed by the baseline, so the arm cannot
                     move windows between the buckets it is scored on.

    A zero denominator is a NAMED STATUS, never a zero and never a drop
    (rule 4)."""
    if mode not in ("OWN_SIGN", "BASELINE_SIGN"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "OWN_SIGN":
        pb, nb = tail_masses(baseline)
        pa, na = tail_masses(arm)
    else:
        pos_w = [w for w, v in baseline.items() if v > 0]
        neg_w = [w for w, v in baseline.items() if v < 0]
        pb = sum(baseline[w] for w in pos_w)
        nb = sum(baseline[w] for w in neg_w)
        pa = sum(arm.get(w, 0.0) for w in pos_w)
        na = sum(arm.get(w, 0.0) for w in neg_w)
    if pb == 0 or nb == 0:
        return {"status": "TAIL_MASS_DENOMINATOR_ZERO", "mode": mode,
                "baseline_pos_mass": pb, "baseline_neg_mass": nb,
                "A": None, "ret_pos": None, "ret_neg": None}
    ret_pos, ret_neg = pa / pb, na / nb
    return {"status": "OK", "mode": mode, "ret_pos": ret_pos,
            "ret_neg": ret_neg, "A": ret_pos - ret_neg,
            "baseline_pos_mass": pb, "baseline_neg_mass": nb,
            "arm_pos_mass": pa, "arm_neg_mass": na,
            "n_windows": len(baseline)}


# --------------------------------------------------------------- the items

def item_1_estimand(d: dict) -> dict:
    e = d.get("estimand") or {}
    prim = e.get("PRIMARY_definition") or {}
    comp = e.get("REQUIRED_COMPANION_definition") or {}
    ok = (e.get("name") == "A = ret_pos - ret_neg"
          and prim.get("id") == "OWN_SIGN"
          and comp.get("id") == "BASELINE_SIGN"
          and all(k in prim for k in ("ret_pos", "ret_neg", "A"))
          and all(k in comp for k in ("ret_pos", "ret_neg"))
          and bool(e.get("NOT_the_mean"))
          and (e.get("zero_denominator") or {}).get("status_name")
          == "TAIL_MASS_DENOMINATOR_ZERO")
    return _v(ok, item="estimand", name=e.get("name"),
              both_definitions_declared=bool(prim and comp))


def item_2_direction(d: dict) -> dict:
    dd = d.get("direction") or {}
    obs = (d.get("observed_at_declaration_time") or {}).get("cells") or {}
    mult = d.get("multiplicity") or {}
    ok = (dd.get("test_is", "").startswith("ONE-SIDED")
          and "ret_pos > ret_neg" in (dd.get("declared_direction") or "")
          and bool(dd.get("HONESTY_CLAUSE_THIS_DECLARATION_WILL_NOT_SOFTEN"))
          and dd.get("refusal_if_absent") == "P_TWO_SIDED_ABSENT"
          and len(obs) == mult.get("n_cells"))
    return _v(ok, item="direction",
              observed_cells_recorded=len(obs),
              two_sided_companion_required=bool(dd.get("consequence")))


def item_3_minimum_sample(d: dict, params: dict | None = None) -> dict:
    m = d.get("minimum_sample") or {}
    n = m.get("n_draws")
    bar = None
    if params is None and PARAMS.is_file():
        params = json.loads(PARAMS.read_text())
    if params:
        bar = params.get("min_draws_per_arm_day")
    ok = (isinstance(n, int) and n >= RULE_6_FLOOR
          and m.get("rule_6_floor") == RULE_6_FLOOR
          and m.get("refusal_name") == "NULL_UNDER_SAMPLED"
          and (bar is None or n == bar))
    return _v(ok, item="minimum_sample", n_draws=n, rule_6_floor=RULE_6_FLOOR,
              params_bar=bar, agrees_with_the_pre_declared_bar=(n == bar))


def item_4_multiplicity(d: dict) -> dict:
    m = d.get("multiplicity") or {}
    ok = (m.get("n_cells") == len(m.get("arms") or []) * len(m.get("days") or [])
          and m.get("n_arms") == len(m.get("arms") or [])
          and "HOLM" in (m.get("correction_for_cell_claims") or "").upper()
          and "HOLM" in (m.get("correction_for_arm_level_claims") or "").upper()
          and bool(m.get("recorded_at"))
          and bool(m.get("no_other_family_may_be_declared_later")))
    return _v(ok, item="multiplicity", n_cells=m.get("n_cells"),
              n_arms=m.get("n_arms"), computed_n_cells=(
                  len(m.get("arms") or []) * len(m.get("days") or [])))


def item_5_matching(d: dict) -> dict:
    m = d.get("matching") or {}
    enf = m.get("enforcement") or {}
    un = m.get("unmatchable_cell") or {}
    ok = (tuple(m.get("matched_on") or ()) == MATCH_KEYS
          and m.get("simultaneously") is True
          and all(k in enf for k in MATCH_KEYS)
          and all("EXACT" in (enf[k] or "").upper() for k in MATCH_KEYS)
          and "MATCH_INFEASIBLE_HOUR" in (enf.get("hour") or "")
          and un.get("status_name") == "MATCH_INFEASIBLE"
          and bool(un.get("forbidden")))
    return _v(ok, item="matching", matched_on=m.get("matched_on"),
              simultaneously=m.get("simultaneously"))


def item_6_sinkers(d: dict, observed: dict | None = None) -> dict:
    """A criterion with no failing outcome is not a criterion.

    Beyond checking the sinkers are declared, this EVALUATES SINK_2 against
    the observed cells: if no declared sinker can ever be true, the test
    cannot fail and the section is decoration."""
    s = d.get("what_would_sink_it") or {}
    named = [k for k in s if k.startswith("SINK_")]
    have_pred = [k for k in named if (s[k] or {}).get("predicate")
                 and (s[k] or {}).get("verdict_if_true")]
    obs = observed if observed is not None else (
        (d.get("observed_at_declaration_time") or {}).get("cells") or {})
    already = [c for c, v in obs.items()
               if ((v.get("A_own_sign") or {}).get("A") or 0) <= 0]
    ok = (len(named) >= 5 and len(have_pred) == len(named)
          and len(already) >= 1)
    return _v(ok, item="what_would_sink_it", n_sinkers=len(named),
              all_have_predicates=(len(have_pred) == len(named)),
              cells_already_failing_SINK_2=already,
              the_test_can_fail=bool(already))


def item_7_falsifiers(d: dict) -> dict:
    f = d.get("falsifiers") or {}
    pos = f.get("positive_control_the_null_MUST_flag") or {}
    bad = f.get("known_bad_the_null_MUST_refuse") or []
    names = {b.get("name") for b in bad if isinstance(b, dict)}
    ok = (pos.get("name") == "ORACLE_WORST_WINDOWS"
          and bool(pos.get("construction")) and bool(pos.get("must"))
          and len(bad) >= 5
          and all(isinstance(b, dict) and b.get("construction") and b.get("must")
                  for b in bad)
          and "UNDER_CANCELLING_CONTROL" in names
          and "MEAN_SUBSTITUTED_FOR_THE_ASYMMETRY" in names)
    return _v(ok, item="falsifiers", positive_control=pos.get("name"),
              n_known_bad=len(bad), known_bad=sorted(names))


def item_8_population(d: dict) -> dict:
    p = d.get("population") or {}
    marks = ((p.get("2026-09-03_carries_its_marks_and_stays_SEPARABLE") or {})
             .get("marks") or [])
    ok = (len(p.get("days") or []) == 4
          and p.get("all_four_are_CONSUMED") is True
          and p.get("costs_no_validation_day") is True
          and p.get("protected_from_utc_date") == "2026-09-08"
          and len(marks) >= 4
          and all(m.get("mark") and m.get("detail") for m in marks))
    return _v(ok, item="population", n_days=len(p.get("days") or []),
              n_09_03_marks=len(marks),
              marks=[m.get("mark") for m in marks])


def item_9_validation_limit(d: dict) -> dict:
    v = d.get("what_a_pass_does_NOT_establish") or {}
    ni = v.get("no_interval_is_claimable") or {}
    ok = ("CANNOT VALIDATE" in (v.get("REQUIRED_VALUE") or "").upper()
          and v.get("REQUIRED_FIELD_ON_EVERY_RESULT") == "validation_limit"
          and v.get("refusal_name") == "RESULT_DOES_NOT_STATE_ITS_VALIDATION_LIMIT"
          and isinstance(ni.get("G"), int) and ni["G"] < ni.get("bar", 5)
          and ni.get("refusal_name") == "INTERVAL_CLAIMED_BELOW_G5")
    return _v(ok, item="validation_limit", G=ni.get("G"), bar=ni.get("bar"),
              intervals_claimable=False)


def item_10_per_draw_reduction(d: dict) -> dict:
    r = d.get("per_draw_reduction") or {}
    ok = (r.get("n_floats_per_draw") == len(r.get("keep_per_draw") or [])
          and r.get("n_floats_per_draw") == 4
          and any("fills" in x for x in (r.get("FORBIDDEN") or []))
          and bool(r.get("baseline_denominators")))
    return _v(ok, item="per_draw_reduction",
              n_floats_per_draw=r.get("n_floats_per_draw"),
              forbidden=r.get("FORBIDDEN"))


def item_11_wall_clock(d: dict) -> dict:
    w = d.get("wall_clock") or {}
    ok = ("NOT ESTABLISHED" in (w.get("status") or "").upper()
          and w.get("DOES_IT_FIT_BEFORE_09_54_37Z") is False
          and bool(w.get("why_not"))
          and bool(w.get("DA_own_observation_disagrees"))
          and "MUST NOT BE CHOSEN TO FIT" in (
              w.get("what_fits_in_that_window") or "").upper())
    return _v(ok, item="wall_clock", status=w.get("status"),
              fits_before_the_window=w.get("DOES_IT_FIT_BEFORE_09_54_37Z"))


ITEMS = (item_1_estimand, item_2_direction, item_3_minimum_sample,
         item_4_multiplicity, item_5_matching, item_6_sinkers,
         item_7_falsifiers, item_8_population, item_9_validation_limit,
         item_10_per_draw_reduction, item_11_wall_clock)


def evaluate(d: dict | None = None) -> dict:
    d = d if d is not None else load()
    out = [f(d) for f in ITEMS]
    return {"protocol": d.get("protocol"), "status": d.get("STATUS"),
            "items": out,
            "all_hold": all(r["verdict"] == HOLDS for r in out),
            "n_items": len(out)}


# ------------------------------------------------- guards on a RESULT

def require_result_fields(result: dict) -> dict:
    """A limit that lives only in the declaration does not bind the result
    (rule 35). These are the fields a result MUST carry, checked as fields."""
    missing = []
    if not result.get("validation_limit"):
        missing.append("RESULT_DOES_NOT_STATE_ITS_VALIDATION_LIMIT")
    if result.get("p_two_sided") is None:
        missing.append("P_TWO_SIDED_ABSENT")
    if tuple(result.get("matched_on") or ()) != MATCH_KEYS:
        missing.append("MATCHED_ON_ABSENT_OR_VAGUE")
    n = result.get("n_draws")
    if not isinstance(n, int) or n < RULE_6_FLOOR:
        missing.append("NULL_UNDER_SAMPLED")
    stat = (result.get("statistic") or "").lower()
    if stat in FORBIDDEN_STATISTIC:
        missing.append("MEAN_SUBSTITUTED_FOR_THE_ASYMMETRY")
    if result.get("interval") is not None:
        missing.append("INTERVAL_CLAIMED_BELOW_G5")
    if missing:
        raise ValueError("REFUSED " + " ".join(missing))
    return {"status": "RESULT_FIELDS_PRESENT", "checked": len(MATCH_KEYS) + 5}


# ------------------------------------------------------------- selftest

def _break(d: dict, path: list, value):
    import copy
    c = copy.deepcopy(d)
    o = c
    for k in path[:-1]:
        o = o[k]
    if value is _DEL:
        o.pop(path[-1], None)
    else:
        o[path[-1]] = value
    return c


class _DELT:
    pass


_DEL = _DELT()
_N = {"n": 0, "bad": 0}


def ok(cond, label):
    _N["n"] += 1
    if not cond:
        _N["bad"] += 1
        print(f"  FAIL {label}")
    return cond


def selftest(quiet: bool = False) -> int:
    D = load()
    P = json.loads(PARAMS.read_text()) if PARAMS.is_file() else None

    # ---- every item HOLDS on the real declaration, and FAILS when broken.
    ok(item_1_estimand(D)["verdict"] == HOLDS
       and item_1_estimand(_break(D, ["estimand", "NOT_the_mean"], _DEL)
                           )["verdict"] == FAILS
       and item_1_estimand(_break(D, ["estimand", "REQUIRED_COMPANION_definition"],
                                  {}))["verdict"] == FAILS,
       "item 1 estimand: holds real, fails without the not-the-mean clause "
       "and without the companion definition")

    ok(item_2_direction(D)["verdict"] == HOLDS
       and item_2_direction(_break(D, ["direction", "test_is"], "TWO-SIDED")
                            )["verdict"] == FAILS
       and item_2_direction(
           _break(D, ["observed_at_declaration_time", "cells"], {})
       )["verdict"] == FAILS,
       "item 2 direction: holds real; fails if the test stops being one-sided "
       "and fails if the observed cells are not recorded (the honesty clause "
       "must be BACKED BY THE NUMBERS, not by prose)")

    ok(item_3_minimum_sample(D, P)["verdict"] == HOLDS
       and item_3_minimum_sample(_break(D, ["minimum_sample", "n_draws"], 199),
                                 P)["verdict"] == FAILS
       and item_3_minimum_sample(_break(D, ["minimum_sample", "n_draws"], 200),
                                 P)["verdict"] == FAILS,
       "item 3 minimum sample: 199 fails the rule-6 floor AND 200 fails "
       "because it disagrees with the PRE-DECLARED params bar of 500 -- the "
       "bar is the artifact's, never the run's convenience")

    ok(item_4_multiplicity(D)["verdict"] == HOLDS
       and item_4_multiplicity(_break(D, ["multiplicity", "n_cells"], 4)
                               )["verdict"] == FAILS
       and item_4_multiplicity(
           _break(D, ["multiplicity", "correction_for_cell_claims"], "none")
       )["verdict"] == FAILS,
       "item 4 multiplicity: n_cells must EQUAL arms x days (computed, not "
       "typed) and a correction must be named")

    ok(item_5_matching(D)["verdict"] == HOLDS
       and item_5_matching(_break(D, ["matching", "matched_on"],
                                  ["cancel_count", "side"]))["verdict"] == FAILS
       and item_5_matching(_break(D, ["matching", "simultaneously"], False)
                           )["verdict"] == FAILS
       and item_5_matching(
           _break(D, ["matching", "enforcement", "hour"],
                  "approximately per hour"))["verdict"] == FAILS,
       "item 5 matching: dropping hour, dropping simultaneity, or relaxing "
       "the hour enforcement all FAIL -- these are the three that null "
       "de-levering and they are not severable")

    ok(item_6_sinkers(D)["verdict"] == HOLDS
       and item_6_sinkers(D, observed={"x": {"A_own_sign": {"A": 0.5}}}
                          )["verdict"] == FAILS,
       "item 6 sinkers: FAILS when no observed cell could trip SINK_2 -- a "
       "criterion with no reachable failing outcome is decoration")

    ok(item_7_falsifiers(D)["verdict"] == HOLDS
       and item_7_falsifiers(
           _break(D, ["falsifiers", "known_bad_the_null_MUST_refuse"], [])
       )["verdict"] == FAILS
       and item_7_falsifiers(
           _break(D, ["falsifiers", "positive_control_the_null_MUST_flag"],
                  {"name": "something reasonable"}))["verdict"] == FAILS,
       "item 7 falsifiers: a vague positive control with no construction and "
       "no must-condition FAILS")

    ok(item_8_population(D)["verdict"] == HOLDS
       and item_8_population(
           _break(D, ["population", "2026-09-03_carries_its_marks_and_stays_"
                      "SEPARABLE", "marks"], []))["verdict"] == FAILS,
       "item 8 population: 09-03 stripped of its marks FAILS")

    ok(item_9_validation_limit(D)["verdict"] == HOLDS
       and item_9_validation_limit(
           _break(D, ["what_a_pass_does_NOT_establish", "no_interval_is_"
                      "claimable", "G"], 7))["verdict"] == FAILS,
       "item 9 validation limit: G=7 would clear the bar and the clause must "
       "then stop asserting no-interval -- the guard reads the COUNT, not "
       "the sentence (REV 149's llamas)")

    ok(item_10_per_draw_reduction(D)["verdict"] == HOLDS
       and item_10_per_draw_reduction(
           _break(D, ["per_draw_reduction", "n_floats_per_draw"], 288)
       )["verdict"] == FAILS,
       "item 10 per-draw reduction: the count must EQUAL the listed keys")

    ok(item_11_wall_clock(D)["verdict"] == HOLDS
       and item_11_wall_clock(
           _break(D, ["wall_clock", "DOES_IT_FIT_BEFORE_09_54_37Z"], True)
       )["verdict"] == FAILS,
       "item 11 wall clock: claiming it fits FAILS -- it does not")

    # ---- THE STATISTIC ITSELF. The property the whole design rests on:
    # ---- PURE DE-LEVERING MUST SCORE EXACTLY ZERO.
    base = {f"w{i}": v for i, v in enumerate(
        [500.0, -300.0, 120.0, -80.0, 900.0, -1500.0, 40.0, -12.0, 260.0])}
    for k in (0.9, 0.5, 0.25, 0.05):
        lev = {w: k * v for w, v in base.items()}
        for mode in ("OWN_SIGN", "BASELINE_SIGN"):
            a = asymmetry(lev, base, mode)
            ok(abs(a["A"]) < 1e-12 and abs(a["ret_pos"] - k) < 1e-12
               and abs(a["ret_neg"] - k) < 1e-12,
               f"POSITIVE CONTROL FOR THE ESTIMAND: pure de-levering at "
               f"k={k} scores A=0 exactly in {mode} "
               f"(got {a['A']!r}) -- this is the property that makes the "
               f"statistic a de-levering null at all")

    # An ORACLE clip -- remove the worst window only -- must score A > 0.
    worst = min(base, key=lambda w: base[w])
    clip = {w: (0.0 if w == worst else v) for w, v in base.items()}
    a = asymmetry(clip, base, "OWN_SIGN")
    ok(a["A"] > 0.3, f"ORACLE CONTROL: clipping only the worst window scores "
                     f"A={a['A']:.4f} > 0 -- the statistic detects asymmetry "
                     f"it is handed")

    # A REVERSE oracle -- remove the BEST window -- must score A < 0.
    best = max(base, key=lambda w: base[w])
    anti = {w: (0.0 if w == best else v) for w, v in base.items()}
    a = asymmetry(anti, base, "OWN_SIGN")
    ok(a["A"] < -0.3, f"KNOWN-BAD: clipping the BEST window scores "
                      f"A={a['A']:.4f} < 0 -- the statistic is SIGNED and "
                      f"does not reward any clipping whatsoever")

    z = asymmetry({"a": 1.0}, {"a": 1.0}, "OWN_SIGN")
    ok(z["status"] == "TAIL_MASS_DENOMINATOR_ZERO" and z["A"] is None,
       "KNOWN-BAD: a zero tail denominator returns a NAMED STATUS with A None "
       "-- never a 0 that reads as 'no asymmetry' (rule 4)")

    # ---- the RESULT guards
    good = {"validation_limit": "cannot validate", "p_two_sided": 0.3,
            "matched_on": list(MATCH_KEYS), "n_draws": 500,
            "statistic": "A = ret_pos - ret_neg", "interval": None}
    try:
        require_result_fields(good)
        good_ok = True
    except ValueError:
        good_ok = False
    ok(good_ok, "RESULT GUARD positive control: a complete result passes")

    for path, val, name in (("validation_limit", None,
                             "RESULT_DOES_NOT_STATE_ITS_VALIDATION_LIMIT"),
                            ("p_two_sided", None, "P_TWO_SIDED_ABSENT"),
                            ("matched_on", ["cancel_count"],
                             "MATCHED_ON_ABSENT_OR_VAGUE"),
                            ("n_draws", 100, "NULL_UNDER_SAMPLED"),
                            ("statistic", "mean",
                             "MEAN_SUBSTITUTED_FOR_THE_ASYMMETRY"),
                            ("interval", [0.1, 0.3],
                             "INTERVAL_CLAIMED_BELOW_G5")):
        bad = dict(good)
        bad[path] = val
        try:
            require_result_fields(bad)
            fired = False
        except ValueError as e:
            fired = name in str(e)
        ok(fired, f"RESULT GUARD known-bad: {path}={val!r} must refuse {name}")

    res = evaluate(D)
    ok(res["all_hold"] is True,
       f"the real declaration holds on all {res['n_items']} items")

    if not quiet:
        print(f"[da_asymmetry_null] {_N['n'] - _N['bad']}/{_N['n']} checks, "
              f"{_N['bad']} failures; declaration STATUS={D['STATUS']}")
    return 1 if _N["bad"] else 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(json.dumps(evaluate(), indent=1, default=str))
