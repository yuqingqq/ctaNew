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
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECL = HERE / "declarations" / "da_asymmetry_null_declaration_v1.json"
PARAMS = HERE / "declarations" / "de_multiday_gate1_params_v29.json"

HOLDS = "HOLDS"
FAILS = "FAILS"

RULE_6_FLOOR = 200
MATCH_KEYS = ("cancel_count", "side", "hour")
FORBIDDEN_STATISTIC = ("mean", "day_mean", "net_value_cents")


# ==================================================================
# DA 192 -- THE INSTRUMENTS THAT TEST PROPERTIES INSTEAD OF SPELLING
# ==================================================================
#
# REV 162 found that SIX of this module's item predicates checked the
# WORDING of a clause rather than the property it asserts, in a module
# whose own docstring says a declaration that is only prose is the defect
# this programme keeps paying for.
#
# DA 192 enumerated by THE OPERATION rather than by inspection -- build a
# semantically WEAKER clause that keeps the vocabulary, and see which
# predicates still pass -- and the answer was ELEVEN OF ELEVEN, not six.
# REV's list was a sample of the same class (SEAT_PROTOCOL rule 32).
#
# Three instruments replace the spelling:
#   * `unconditional()`  -- an ABSOLUTE clause ("refuses", "must not",
#     "cannot validate", "forbidden") is violated by an EXCEPTION, so the
#     property is the ABSENCE OF AN ESCAPE, not the presence of a word.
#   * `decidable()`      -- a clause that states a bar an instrument must
#     meet has to name a QUANTITY. "do well" is not a bar.
#   * execution          -- where behaviour exists (the statistic, the
#     result guards, Holm, the matcher) the predicate DRIVES it.
#
# And the verdict is now THREE-VALUED. A clause this module cannot check
# from v1's fields is NAMED and the item reports
# HOLDS_WITH_UNCHECKED_CLAUSES -- never HOLDS. An unverifiable clause must
# not read as a verified one; that is what "11/11 items HOLD" did.

UNCHECKED = "HOLDS_WITH_UNCHECKED_CLAUSES"

#: An ABSOLUTE clause admits no exception. These are the connectives that
#: introduce one. Driven both ways by the selftest (rule 15): a clean
#: absolute clause must pass, and each marker must be caught.
EXCEPTION_MARKERS = (
    r"\bunless\b", r"\bexcept\b", r"\botherwise\b", r"\bon its own\b",
    r"\bwhere feasible\b", r"\bprovided that\b", r"\bsubject to\b",
    r"\bat the operator's discretion\b", r"\bbeyond what\b",
    r"\bif the operator\b", r"\bwhere the operator\b",
    r"\bat their discretion\b", r"\bmay be relaxed\b",
)

_QUANT = re.compile(r"(\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?(?:st|nd|rd|th)?"
                    r"|<=|>=|<|>|percentile|1/\d+)")


def unconditional(text) -> dict:
    """Does this ABSOLUTE clause admit an exception?

    The property is structural: an absolute promise with an escape is a
    weaker promise wearing the same words. This is what makes a weakened
    v2 FAIL instead of passing unchanged."""
    t = str(text or "")
    hits = [m for m in EXCEPTION_MARKERS if re.search(m, t, re.I)]
    return {"unconditional": not hits, "escapes": hits, "n_chars": len(t)}


def decidable(text) -> dict:
    """Does this clause name a bar an instrument could actually meet?

    `must: "do well"` is not a criterion; `must: "one-sided p <= 1/501 or
    above the 99th percentile"` is."""
    t = str(text or "")
    q = _QUANT.findall(t)
    return {"decidable": bool(q), "quantities": q[:6]}


def sinker_bar(sink: dict, d: dict) -> dict:
    """A sinker must be DECIDABLE -- its predicate names a quantity, OR the
    bar it depends on RESOLVES TO A NUMBER elsewhere in the declaration.

    "the ORACLE does not land in the extreme right tail" is decidable only
    because the oracle's bar is declared; if that bar goes vague this
    sinker goes vague with it, and the predicate must say so."""
    pred = str(sink.get("predicate") or "")
    if decidable(pred)["decidable"]:
        return {"decidable": True, "bar_from": "its own predicate"}
    if re.search(r"\bholm\b", pred, re.I):
        a = (d.get("multiplicity") or {}).get("alpha")
        return {"decidable": isinstance(a, (int, float)) and not isinstance(a, bool),
                "bar_from": "multiplicity.alpha", "value": a}
    if re.search(r"\boracle\b", pred, re.I):
        must = ((d.get("falsifiers") or {})
                .get("positive_control_the_null_MUST_flag") or {}).get("must")
        return {"decidable": decidable(must)["decidable"],
                "bar_from": "falsifiers.positive_control_the_null_MUST_flag.must"}
    return {"decidable": False, "bar_from": None,
            "why": "no quantity in the predicate and no bar it could resolve to"}


def formula_roles(text, allow_baseline_in_numerator=False) -> dict:
    """Does the declared ratio put the ARM over the BASELINE?

    Structural, not textual: split the declared formula at its division
    and check WHICH BOOK each side references. A v2 that writes the arm
    over the arm keeps every word and fails this."""
    t = str(text or "")
    if "/" not in t:
        return {"well_formed": False, "why": "no division in the declared formula"}
    num, den = t.split("/", 1)
    def refs(x):
        return {"arm": bool(re.search(r"\barm\b", x, re.I)),
                "baseline": bool(re.search(r"\bbase(line)?\b", x, re.I))}
    n, d = refs(num), refs(den)
    ok = (n["arm"] and (allow_baseline_in_numerator or not n["baseline"])
          and d["baseline"] and not d["arm"])
    return {"well_formed": ok, "numerator": n, "denominator": d}


def tails_named(text) -> dict:
    """A one-sided test names ONE tail. A clause that also names a
    two-sided fallback names two, and that is a different rule."""
    t = str(text or "").upper()
    named = sorted({s for s in ("ONE-SIDED", "TWO-SIDED") if s in t})
    return {"n_tails_named": len(named), "named": named,
            "single_rule": len(named) == 1}


def family_size_named(text):
    """The integer family size a correction clause names, or None."""
    m = re.search(r"over the\s+(\d+)\s", str(text or ""))
    return int(m.group(1)) if m else None


def holm(pvals: dict, family_size: int) -> dict:
    """Holm-Bonferroni over a DECLARED family size, implemented here so
    item 4 can DRIVE the correction rather than read the word HOLM."""
    if family_size < len(pvals):
        raise ValueError("REFUSED HOLM_FAMILY_SMALLER_THAN_THE_TESTS: "
                         f"{family_size} < {len(pvals)}")
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(order):
        adj = min(1.0, max(running, (family_size - i) * p))
        running = adj
        out[k] = adj
    return out


def match_draw(arm_cancels: list, drawn: list) -> dict:
    """The declared control, EXECUTED: cancel count, side and hour must
    match SIMULTANEOUSLY and EXACTLY. Refuses by name with the deficit."""
    from collections import Counter
    if len(drawn) != len(arm_cancels):
        raise ValueError("REFUSED MATCH_INFEASIBLE_COUNT: drawn "
                         f"{len(drawn)} against {len(arm_cancels)}")
    for key, name in ((1, "MATCH_INFEASIBLE_SIDE"), (2, "MATCH_INFEASIBLE_HOUR")):
        a = Counter(c[key] for c in arm_cancels)
        b = Counter(c[key] for c in drawn)
        if a != b:
            deficit = {k: a[k] - b.get(k, 0) for k in a | b if a[k] != b.get(k, 0)}
            raise ValueError(f"REFUSED {name}: deficit {deficit}")
    return {"status": "MATCHED", "n": len(drawn),
            "matched_on": list(MATCH_KEYS)}


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
#
# DA 192: every item below either EXECUTES the behaviour its clause
# promises, or evaluates a STRUCTURAL property of the clause (its
# unconditionality, its decidability, the roles inside its formula, a
# count it names against a count computed). A clause that can be checked
# by neither is NAMED in `unchecked_clauses` and the item reports
# HOLDS_WITH_UNCHECKED_CLAUSES. Nothing here reports HOLDS on the
# presence of a word.


def _verdict(props: dict, unchecked: list, **extra) -> dict:
    failed = [k for k, v in props.items() if not v]
    if failed:
        v = FAILS
    elif unchecked:
        v = UNCHECKED
    else:
        v = HOLDS
    return {"verdict": v, "failed_properties": failed,
            "n_properties_driven": len(props),
            "unchecked_clauses": unchecked, **extra}


_FIX = {f"w{i}": v for i, v in enumerate(
    [500.0, -300.0, 120.0, -80.0, 900.0, -1500.0, 40.0, -12.0, 260.0])}


def _delever_property() -> bool:
    """EXECUTED: proportional scaling scores A = 0 in both partitions."""
    for k in (0.9, 0.5, 0.25, 0.05):
        lev = {w: k * v for w, v in _FIX.items()}
        for mode in ("OWN_SIGN", "BASELINE_SIGN"):
            a = asymmetry(lev, _FIX, mode)
            if a["A"] is None or abs(a["A"]) > 1e-12:
                return False
    return True


def _oracle_property() -> tuple:
    worst = min(_FIX, key=lambda w: _FIX[w])
    best = max(_FIX, key=lambda w: _FIX[w])
    clip = {w: (0.0 if w == worst else v) for w, v in _FIX.items()}
    anti = {w: (0.0 if w == best else v) for w, v in _FIX.items()}
    return (asymmetry(clip, _FIX, "OWN_SIGN")["A"],
            asymmetry(anti, _FIX, "OWN_SIGN")["A"])


def _guard_refuses(field, value, name) -> bool:
    """EXECUTED: does the RESULT guard actually refuse this?"""
    good = {"validation_limit": "cannot validate", "p_two_sided": 0.3,
            "matched_on": list(MATCH_KEYS), "n_draws": 500,
            "statistic": "A = ret_pos - ret_neg", "interval": None}
    good[field] = value
    try:
        require_result_fields(good)
        return False
    except ValueError as e:
        return name in str(e)


def item_1_estimand(d: dict) -> dict:
    e = d.get("estimand") or {}
    prim = e.get("PRIMARY_definition") or {}
    comp = e.get("REQUIRED_COMPANION_definition") or {}
    orc, anti = _oracle_property()
    props = {
        "two_partitions_declared_and_distinct": (
            bool(prim.get("id")) and bool(comp.get("id"))
            and prim.get("id") != comp.get("id")),
        "primary_ret_pos_is_arm_over_baseline":
            formula_roles(prim.get("ret_pos"))["well_formed"],
        "primary_ret_neg_is_arm_over_baseline":
            formula_roles(prim.get("ret_neg"))["well_formed"],
        "companion_denominators_are_the_baseline": (
            formula_roles(comp.get("ret_pos"), True)["well_formed"]
            and formula_roles(comp.get("ret_neg"), True)["well_formed"]),
        "EXECUTED_proportional_scaling_scores_zero": _delever_property(),
        "EXECUTED_oracle_clip_scores_positive": orc > 0.3,
        "EXECUTED_anti_oracle_clip_scores_negative": anti < -0.3,
        "EXECUTED_zero_denominator_is_a_named_status": (
            asymmetry({"a": 1.0}, {"a": 1.0}, "OWN_SIGN")["status"]
            == "TAIL_MASS_DENOMINATOR_ZERO"),
        "EXECUTED_a_mean_statistic_is_refused": _guard_refuses(
            "statistic", "mean", "MEAN_SUBSTITUTED_FOR_THE_ASYMMETRY"),
    }
    unchecked = ["estimand.NOT_the_mean -- prose. The BEHAVIOUR it promises "
                 "is driven above (a mean-valued result is refused); the "
                 "sentence itself is not machine-checkable from v1."]
    return _verdict(props, unchecked, item="estimand",
                    oracle_A=orc, anti_oracle_A=anti)


def item_2_direction(d: dict) -> dict:
    dd = d.get("direction") or {}
    obs = (d.get("observed_at_declaration_time") or {}).get("cells") or {}
    mult = d.get("multiplicity") or {}
    direction = str(dd.get("declared_direction") or "")
    m = re.search(r"ret_pos\s*([<>])\s*ret_neg", direction)
    props = {
        "exactly_one_tail_is_named": tails_named(dd.get("test_is"))["single_rule"],
        "direction_puts_ret_pos_above_ret_neg": bool(m) and m.group(1) == ">",
        "observed_cells_recorded_equals_the_family": (
            len(obs) == mult.get("n_cells")),
        "honesty_clause_is_unconditional": unconditional(
            dd.get("HONESTY_CLAUSE_THIS_DECLARATION_WILL_NOT_SOFTEN")
        )["unconditional"],
        "EXECUTED_a_missing_two_sided_p_is_refused": _guard_refuses(
            "p_two_sided", None, "P_TWO_SIDED_ABSENT"),
    }
    return _verdict(props, [], item="direction",
                    observed_cells_recorded=len(obs))


def item_3_minimum_sample(d: dict, params: dict | None = None) -> dict:
    m = d.get("minimum_sample") or {}
    n = m.get("n_draws")
    if params is None and PARAMS.is_file():
        params = json.loads(PARAMS.read_text())
    bar = (params or {}).get("min_draws_per_arm_day")
    props = {
        "n_draws_clears_the_rule_6_floor": isinstance(n, int) and n >= RULE_6_FLOOR,
        "n_draws_equals_the_PRE_DECLARED_params_bar": (n == bar),
        "short_count_clause_is_unconditional":
            unconditional(m.get("short_count"))["unconditional"],
        "EXECUTED_a_short_count_is_refused": _guard_refuses(
            "n_draws", 100, "NULL_UNDER_SAMPLED"),
    }
    return _verdict(props, [], item="minimum_sample", n_draws=n,
                    params_bar=bar)


def item_4_multiplicity(d: dict) -> dict:
    m = d.get("multiplicity") or {}
    n_cells, n_arms = m.get("n_cells"), m.get("n_arms")
    computed = len(m.get("arms") or []) * len(m.get("days") or [])
    try:
        holm({"a": 0.01}, 1)
        drove = True
    except Exception:
        drove = False
    try:
        holm({"a": 0.01, "b": 0.02}, 1)
        refused = False
    except ValueError:
        refused = True
    props = {
        "n_cells_equals_arms_times_days": n_cells == computed,
        "n_arms_equals_the_arm_list": n_arms == len(m.get("arms") or []),
        "cell_correction_names_the_CELL_family": (
            family_size_named(m.get("correction_for_cell_claims")) == n_cells),
        "arm_correction_names_the_ARM_family": (
            family_size_named(m.get("correction_for_arm_level_claims")) == n_arms),
        "family_closure_clause_is_unconditional": unconditional(
            m.get("no_other_family_may_be_declared_later"))["unconditional"],
        "EXECUTED_holm_runs": drove,
        "EXECUTED_holm_refuses_a_family_smaller_than_the_tests": refused,
    }
    return _verdict(props, [], item="multiplicity", n_cells=n_cells,
                    computed_n_cells=computed,
                    cell_family_named=family_size_named(
                        m.get("correction_for_cell_claims")))


def item_5_matching(d: dict) -> dict:
    m = d.get("matching") or {}
    enf = m.get("enforcement") or {}
    un = m.get("unmatchable_cell") or {}
    arm = [("s1", "BUY_UP", 3), ("s1", "SELL_UP", 3), ("s2", "BUY_UP", 9)]
    def refuses(drawn, name):
        try:
            match_draw(arm, drawn)
            return False
        except ValueError as e:
            return name in str(e)
    props = {
        "matched_on_is_exactly_the_three_keys":
            tuple(m.get("matched_on") or ()) == MATCH_KEYS,
        "simultaneously_is_true": m.get("simultaneously") is True,
        "every_enforcement_clause_is_unconditional": all(
            unconditional(enf.get(k))["unconditional"] for k in MATCH_KEYS),
        "relaxation_is_forbidden_unconditionally":
            unconditional(un.get("forbidden"))["unconditional"],
        "EXECUTED_an_exact_match_passes":
            match_draw(arm, list(arm))["status"] == "MATCHED",
        "EXECUTED_a_short_count_refuses": refuses(
            arm[:2], "MATCH_INFEASIBLE_COUNT"),
        "EXECUTED_a_wrong_side_mix_refuses": refuses(
            [("s1", "BUY_UP", 3), ("s1", "BUY_UP", 3), ("s2", "BUY_UP", 9)],
            "MATCH_INFEASIBLE_SIDE"),
        "EXECUTED_a_wrong_hour_mix_refuses": refuses(
            [("s1", "BUY_UP", 3), ("s1", "SELL_UP", 3), ("s2", "BUY_UP", 4)],
            "MATCH_INFEASIBLE_HOUR"),
    }
    return _verdict(props, [], item="matching", matched_on=m.get("matched_on"))


def item_6_sinkers(d: dict, observed: dict | None = None) -> dict:
    s = d.get("what_would_sink_it") or {}
    named = [k for k in s if k.startswith("SINK_")]
    obs = observed if observed is not None else (
        (d.get("observed_at_declaration_time") or {}).get("cells") or {})
    already = [c for c, v in obs.items()
               if ((v.get("A_own_sign") or {}).get("A") or 0) <= 0]
    props = {
        "at_least_five_sinkers": len(named) >= 5,
        "every_sinker_has_a_predicate_and_a_verdict": all(
            (s[k] or {}).get("predicate") and (s[k] or {}).get("verdict_if_true")
            for k in named),
        "every_sinker_predicate_is_unconditional": all(
            unconditional((s[k] or {}).get("predicate"))["unconditional"]
            for k in named),
        "every_sinker_BAR_RESOLVES_to_a_number": all(
            sinker_bar(s[k] or {}, d)["decidable"] for k in named),
        "EVALUATED_at_least_one_observed_cell_already_trips_a_sinker":
            bool(already),
    }
    return _verdict(props, [], item="what_would_sink_it",
                    n_sinkers=len(named),
                    bars={k: sinker_bar(s[k] or {}, d).get("bar_from")
                          for k in named},
                    cells_already_failing_SINK_2=already)


def item_7_falsifiers(d: dict) -> dict:
    f = d.get("falsifiers") or {}
    pos = f.get("positive_control_the_null_MUST_flag") or {}
    bad = f.get("known_bad_the_null_MUST_refuse") or []
    names = {b.get("name") for b in bad if isinstance(b, dict)}
    orc, anti = _oracle_property()
    props = {
        "positive_control_names_a_construction": bool(pos.get("construction")),
        "positive_control_states_a_DECIDABLE_bar":
            decidable(pos.get("must"))["decidable"],
        "at_least_five_known_bads": len(bad) >= 5,
        "every_known_bad_has_a_construction_and_a_must": all(
            isinstance(b, dict) and b.get("construction") and b.get("must")
            for b in bad),
        "every_known_bad_must_clause_is_unconditional": all(
            unconditional(b.get("must"))["unconditional"] for b in bad
            if isinstance(b, dict)),
        "the_under_cancelling_control_is_named":
            "UNDER_CANCELLING_CONTROL" in names,
        "EXECUTED_the_statistic_separates_oracle_from_anti_oracle":
            orc > 0 > anti,
    }
    return _verdict(props, [], item="falsifiers", n_known_bad=len(bad),
                    known_bad=sorted(n for n in names if n))


def item_8_population(d: dict, artifact: dict | None = None) -> dict:
    p = d.get("population") or {}
    blk = p.get("2026-09-03_carries_its_marks_and_stays_SEPARABLE") or {}
    marks = {m.get("mark"): m for m in (blk.get("marks") or [])}
    unchecked = []
    props = {
        "four_days_declared": len(p.get("days") or []) == 4,
        "all_four_marked_consumed": p.get("all_four_are_CONSUMED") is True,
        "protected_boundary_is_2026_09_08":
            p.get("protected_from_utc_date") == "2026-09-08",
        "at_least_four_marks": len(marks) >= 4,
        "every_mark_carries_a_detail": all(
            m.get("detail") for m in marks.values()),
    }
    if artifact is None:
        cand = sorted(Path("/home/yuqing/ctaNew/data/pm_5min/derived").glob(
            "p003_de_point_estimate_day_20260903_L250ms__20260910T05*.json"))
        if cand:
            artifact = json.loads(cand[-1].read_text())
    if artifact:
        cov = (((artifact.get("population_and_coverage") or {})
                .get("coverage") or {}).get("coverage"))
        declared_cov = (marks.get("coverage") or {}).get("value")
        props["VERIFIED_coverage_mark_matches_the_artifact"] = (
            isinstance(declared_cov, float) and cov is not None
            and abs(declared_cov - cov) < 1e-12)
        adm = (artifact.get("population_and_coverage") or {}).get("ADMITTED")
        wc = str((marks.get("window_count") or {}).get("value") or "")
        props["VERIFIED_window_count_mark_names_the_artifacts_ADMITTED"] = (
            adm is not None and str(adm) in wc)
    else:
        unchecked.append("population marks could not be verified against the "
                         "09-03 artifact -- it was not found on this host")
    return _verdict(props, unchecked, item="population",
                    marks=sorted(m for m in marks if m))


def item_9_validation_limit(d: dict) -> dict:
    v = d.get("what_a_pass_does_NOT_establish") or {}
    ni = v.get("no_interval_is_claimable") or {}
    G, bar = ni.get("G"), ni.get("bar")
    props = {
        "cannot_validate_clause_is_unconditional":
            unconditional(v.get("REQUIRED_VALUE"))["unconditional"],
        "the_required_field_is_named":
            v.get("REQUIRED_FIELD_ON_EVERY_RESULT") == "validation_limit",
        "COMPUTED_G_is_below_the_bar": (
            isinstance(G, int) and isinstance(bar, int) and G < bar),
        "EXECUTED_a_missing_validation_limit_is_refused": _guard_refuses(
            "validation_limit", None,
            "RESULT_DOES_NOT_STATE_ITS_VALIDATION_LIMIT"),
        "EXECUTED_an_interval_is_refused_below_G5": _guard_refuses(
            "interval", [0.1, 0.3], "INTERVAL_CLAIMED_BELOW_G5"),
    }
    return _verdict(props, [], item="validation_limit", G=G, bar=bar)


def item_10_per_draw_reduction(d: dict) -> dict:
    r = d.get("per_draw_reduction") or {}
    keep = r.get("keep_per_draw") or []
    forb = r.get("FORBIDDEN") or []
    props = {
        "the_float_count_equals_the_listed_keys":
            r.get("n_floats_per_draw") == len(keep),
        "the_budget_is_four_floats": r.get("n_floats_per_draw") == 4,
        "every_FORBIDDEN_entry_is_unconditional": bool(forb) and all(
            unconditional(x)["unconditional"] for x in forb),
        "per_draw_fills_are_forbidden": any(
            "fills" in str(x) for x in forb),
    }
    return _verdict(props, [], item="per_draw_reduction",
                    n_floats_per_draw=r.get("n_floats_per_draw"))


def item_11_wall_clock(d: dict) -> dict:
    w = d.get("wall_clock") or {}
    m = d.get("minimum_sample") or {}
    mult = d.get("multiplicity") or {}
    rc = d.get("run_constraints_for_DE") or {}
    s_per = w.get("read_estimate_s_per_draw")
    n, cells = m.get("n_draws"), mult.get("n_cells")
    conc = rc.get("concurrency")
    computed_s = computed_fits = None
    available_s = None
    if all(isinstance(x, (int, float)) and x for x in (s_per, n, cells, conc)):
        computed_s = cells * n * s_per / conc
        from datetime import datetime, timezone
        try:
            t0 = datetime.fromisoformat(
                str(d.get("declared_at_utc")).replace("Z", "+00:00"))
            t1 = t0.replace(hour=9, minute=54, second=37,
                            microsecond=0, tzinfo=timezone.utc)
            available_s = (t1 - t0).total_seconds()
            computed_fits = computed_s <= available_s
        except Exception:
            pass
    st = w.get("status")
    props = {
        "status_clause_is_unconditional": unconditional(st)["unconditional"],
        "a_NOT_ESTABLISHED_status_names_no_duration": not (
            "NOT ESTABLISHED" in str(st).upper()
            and decidable(st)["decidable"]),
        "the_no_trimming_clause_is_unconditional": unconditional(
            w.get("what_fits_in_that_window"))["unconditional"],
        "the_unreconciled_observation_is_carried":
            bool(w.get("DA_own_observation_disagrees")),
        "COMPUTED_fit_agrees_with_the_declared_boolean": (
            computed_fits is not None
            and computed_fits == w.get("DOES_IT_FIT_BEFORE_09_54_37Z")),
    }
    return _verdict(props, [], item="wall_clock",
                    computed_required_s=computed_s,
                    available_s=available_s,
                    computed_fits=computed_fits,
                    declared_fits=w.get("DOES_IT_FIT_BEFORE_09_54_37Z"))


ITEMS = (item_1_estimand, item_2_direction, item_3_minimum_sample,
         item_4_multiplicity, item_5_matching, item_6_sinkers,
         item_7_falsifiers, item_8_population, item_9_validation_limit,
         item_10_per_draw_reduction, item_11_wall_clock)


def evaluate(d: dict | None = None) -> dict:
    d = d if d is not None else load()
    out = [f(d) for f in ITEMS]
    unchecked = [c for r in out for c in (r.get("unchecked_clauses") or [])]
    return {"protocol": d.get("protocol"), "status": d.get("STATUS"),
            "items": out,
            "n_items": len(out),
            "n_property_checks_driven": sum(
                r.get("n_properties_driven", 0) for r in out),
            "n_items_FAILING": sum(1 for r in out if r["verdict"] == FAILS),
            "n_items_with_unchecked_clauses": sum(
                1 for r in out if r["verdict"] == UNCHECKED),
            "unchecked_clauses": unchecked,
            "no_item_fails": all(r["verdict"] != FAILS for r in out),
            "HOW_TO_READ_THIS": (
                "`no_item_fails` means every property this module can DRIVE "
                "holds. It does NOT mean the declaration is fully verified: "
                "`unchecked_clauses` names what could not be checked from "
                "v1's fields. DA 192 -- a module that reported 11/11 HOLD "
                "was verifying spelling.")}


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
#
# DA 192: the falsifiers below are WEAKENED v2 CLAUSES -- semantically
# weaker, vocabulary intact -- because that is the shape REV 162 found
# passing unchanged. Deleting a key was never the threat; softening one
# was. Every entry in WEAKENED_V2 must make its item FAIL.


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


#: (item, what was weakened, path, weaker value). Each MUST make the item FAIL.
WEAKENED_V2 = [
 ("item_1_estimand", "ret_neg becomes arm-over-arm",
  ["estimand", "PRIMARY_definition", "ret_neg"],
  "sum over windows of min(arm(w), 0) / sum over windows of min(arm(w), 0)"),
 ("item_1_estimand", "the companion denominator becomes the arm",
  ["estimand", "REQUIRED_COMPANION_definition", "ret_pos"],
  "sum over {w : baseline(w) > 0} of arm(w) / sum over the same windows of arm(w)"),
 ("item_2_direction", "one-sided gains a two-sided escape",
  ["direction", "test_is"],
  "ONE-SIDED, right tail of the null distribution -- and TWO-SIDED where "
  "the one-sided result does not reach alpha"),
 ("item_2_direction", "the direction is reversed",
  ["direction", "declared_direction"], "ret_pos < ret_neg, i.e. A < 0"),
 ("item_3_minimum_sample", "the short-count refusal gains an escape",
  ["minimum_sample", "short_count"],
  "REFUSES, never reports -- unless the operator records the shortfall"),
 ("item_3_minimum_sample", "the draw count drops to the bare floor",
  ["minimum_sample", "n_draws"], 200),
 ("item_4_multiplicity", "cell claims corrected over the ARM family",
  ["multiplicity", "correction_for_cell_claims"], "HOLM over the 2 arms"),
 ("item_4_multiplicity", "the family may be re-opened later",
  ["multiplicity", "no_other_family_may_be_declared_later"],
  "the family is fixed here, unless a later day is ruled in"),
 ("item_5_matching", "a short hour may be rebalanced",
  ["matching", "enforcement", "hour"],
  "EXACT per UTC hour where feasible; where an hour is short the deficit "
  "moves to the nearest hour and MATCH_INFEASIBLE_HOUR is reported"),
 ("item_5_matching", "relaxing the match becomes permitted",
  ["matching", "unmatchable_cell", "forbidden"],
  "relaxing a match to make a cell run, except where the operator judges "
  "the deficit immaterial"),
 ("item_6_sinkers", "the primary sinker gains an escape",
  ["what_would_sink_it", "SINK_1_PRIMARY", "predicate"],
  "Holm-corrected one-sided p > 0.05 on the ARM-LEVEL statistic, unless "
  "the mechanism is corroborated elsewhere"),
 ("item_6_sinkers", "the oracle bar goes vague, so SINK_5 goes with it",
  ["falsifiers", "positive_control_the_null_MUST_flag", "must"],
  "land clearly above the rest of the draws"),
 ("item_7_falsifiers", "the oracle states no bar",
  ["falsifiers", "positive_control_the_null_MUST_flag"],
  {"name": "ORACLE_WORST_WINDOWS", "construction": "an oracle arm",
   "must": "do well"}),
 ("item_8_population", "the coverage mark stops matching the artifact",
  ["population", "2026-09-03_carries_its_marks_and_stays_SEPARABLE", "marks"],
  [{"mark": "coverage", "value": 0.92, "detail": "coverage was fine"},
   {"mark": "window_count", "value": "288 of 288", "detail": "full"},
   {"mark": "silently_missing_windows", "value": 0, "detail": "none"},
   {"mark": "unadjudicable_settlement_window", "value": "none", "detail": "none"}]),
 ("item_9_validation_limit", "cannot-validate gains 'on its own'",
  ["what_a_pass_does_NOT_establish", "REQUIRED_VALUE"],
  "THIS NULL CANNOT VALIDATE ON ITS OWN, but taken with the forward days "
  "it establishes the cancellation line."),
 ("item_10_per_draw_reduction", "the forbidden list gains an exception",
  ["per_draw_reduction", "FORBIDDEN"],
  ["retaining per-draw fills beyond what the operator needs for diagnosis"]),
 ("item_11_wall_clock", "'must not be trimmed' gains an exception",
  ["wall_clock", "what_fits_in_that_window"],
  "8 cells is MARGINAL. THE DRAW COUNT MUST NOT BE CHOSEN TO FIT THE "
  "WINDOW, except where the operator judges a reduced count acceptable."),
 ("item_11_wall_clock", "an unestablished status quotes a duration",
  ["wall_clock", "status"],
  "NOT ESTABLISHED precisely -- approximately 2 hours at N=2"),
 ("item_11_wall_clock", "the fit boolean is flipped against the arithmetic",
  ["wall_clock", "DOES_IT_FIT_BEFORE_09_54_37Z"], True),
]


def selftest(quiet: bool = False) -> int:
    D = load()
    FN = {f.__name__: f for f in ITEMS}

    # ---- 1. THE INSTRUMENTS, driven both ways (rule 15).
    ok(unconditional("REFUSES, never reports")["unconditional"]
       and not unconditional("REFUSES -- unless the operator agrees")["unconditional"],
       "unconditional(): a clean absolute clause passes, an escaped one does not")
    caught = [m for m in EXCEPTION_MARKERS
              if not unconditional("the rule holds " + m.strip("\\b").replace("\\", ""))["unconditional"]]
    ok(len(caught) >= len(EXCEPTION_MARKERS) - 2,
       f"unconditional(): {len(caught)} of {len(EXCEPTION_MARKERS)} markers "
       f"fire on their own text -- the detector is not a single spelling")
    ok(decidable("one-sided p <= 1/501 or above the 99th percentile")["decidable"]
       and not decidable("do well")["decidable"],
       "decidable(): a bar with a quantity passes, 'do well' does not")
    ok(formula_roles("min(arm(w),0) / min(baseline(w),0)")["well_formed"]
       and not formula_roles("min(arm(w),0) / min(arm(w),0)")["well_formed"]
       and not formula_roles("min(baseline(w),0) / min(baseline(w),0)")["well_formed"],
       "formula_roles(): arm-over-baseline passes; arm-over-arm and "
       "baseline-over-baseline do not -- the ROLES are tested, not the words")
    ok(tails_named("ONE-SIDED, right tail")["single_rule"]
       and not tails_named("ONE-SIDED and TWO-SIDED where it fails")["single_rule"],
       "tails_named(): one rule passes, a rule with a fallback does not")
    ok(family_size_named("HOLM over the 8 cells") == 8
       and family_size_named("HOLM over the 2 arms") == 2
       and family_size_named("HOLM") is None,
       "family_size_named(): the family SIZE is extracted, so a correction "
       "over the wrong family is visible")
    h = holm({"a": 0.001, "b": 0.02, "c": 0.4}, 8)
    ok(h["a"] < h["b"] < h["c"] and h["a"] == 0.008,
       f"holm(): monotone and correct over the declared family ({h})")
    try:
        holm({"a": 0.01, "b": 0.02}, 1)
        fired = False
    except ValueError:
        fired = True
    ok(fired, "holm(): REFUSES a family smaller than the number of tests")
    arm = [("s1", "BUY_UP", 3), ("s1", "SELL_UP", 3), ("s2", "BUY_UP", 9)]
    ok(match_draw(arm, list(arm))["status"] == "MATCHED",
       "match_draw(): an exactly matched draw passes")
    for drawn, name in (
            (arm[:2], "MATCH_INFEASIBLE_COUNT"),
            ([("s1", "BUY_UP", 3), ("s1", "BUY_UP", 3), ("s2", "BUY_UP", 9)],
             "MATCH_INFEASIBLE_SIDE"),
            ([("s1", "BUY_UP", 3), ("s1", "SELL_UP", 3), ("s2", "BUY_UP", 4)],
             "MATCH_INFEASIBLE_HOUR")):
        try:
            match_draw(arm, drawn)
            fired = False
        except ValueError as e:
            fired = name in str(e)
        ok(fired, f"match_draw(): refuses {name} -- the three keys are "
                  f"enforced SIMULTANEOUSLY, not in turn")
    sb = sinker_bar({"predicate": "the ORACLE does not land in the extreme "
                                  "right tail"}, D)
    ok(sb["decidable"] and sb["bar_from"].startswith("falsifiers"),
       "sinker_bar(): a sinker with no quantity of its own RESOLVES to the "
       "declared bar it depends on")
    ok(not sinker_bar({"predicate": "the arm underperforms"}, D)["decidable"],
       "sinker_bar(): a sinker whose bar resolves nowhere is NOT decidable")

    # ---- 2. THE STATISTIC. The property the design rests on.
    for k in (0.9, 0.5, 0.25, 0.05):
        lev = {w: k * v for w, v in _FIX.items()}
        for mode in ("OWN_SIGN", "BASELINE_SIGN"):
            a = asymmetry(lev, _FIX, mode)
            ok(abs(a["A"]) < 1e-12 and abs(a["ret_pos"] - k) < 1e-12,
               f"PROPORTIONAL SCALING at k={k} scores A=0 in {mode} -- this "
               f"is the IDEALISED de-levering model, not real cancelling "
               f"(REV 162: the matched control removes the FIRST-ORDER "
               f"effect, not all of it)")
    orc, anti = _oracle_property()
    ok(orc > 0.3 and anti < -0.3,
       f"the statistic is SIGNED: clipping the worst window scores "
       f"{orc:.4f}, clipping the BEST scores {anti:.4f}")
    ok(asymmetry({"a": 1.0}, {"a": 1.0}, "OWN_SIGN")["status"]
       == "TAIL_MASS_DENOMINATOR_ZERO",
       "a zero tail denominator is a NAMED STATUS, never a 0 reading as "
       "'no asymmetry' (rule 4)")

    # ---- 3. THE RESULT GUARDS.
    good = {"validation_limit": "cannot validate", "p_two_sided": 0.3,
            "matched_on": list(MATCH_KEYS), "n_draws": 500,
            "statistic": "A = ret_pos - ret_neg", "interval": None}
    try:
        require_result_fields(good)
        g = True
    except ValueError:
        g = False
    ok(g, "RESULT GUARD positive control: a complete result passes")
    for field, val, name in (
            ("validation_limit", None, "RESULT_DOES_NOT_STATE_ITS_VALIDATION_LIMIT"),
            ("p_two_sided", None, "P_TWO_SIDED_ABSENT"),
            ("matched_on", ["cancel_count"], "MATCHED_ON_ABSENT_OR_VAGUE"),
            ("n_draws", 100, "NULL_UNDER_SAMPLED"),
            ("statistic", "mean", "MEAN_SUBSTITUTED_FOR_THE_ASYMMETRY"),
            ("interval", [0.1, 0.3], "INTERVAL_CLAIMED_BELOW_G5")):
        ok(_guard_refuses(field, val, name),
           f"RESULT GUARD known-bad: {field}={val!r} refuses {name}")

    # ---- 4. THE WEAKENED-v2 BATTERY. This is the fix REV 162 demanded.
    for item, what, path, val in WEAKENED_V2:
        v = FN[item](_break(D, path, val))["verdict"]
        ok(v == FAILS,
           f"WEAKENED v2 -- {item}: {what} -> {v} (must be {FAILS}). A "
           f"softened clause that keeps the vocabulary MUST NOT pass.")

    # ---- 5. and the real declaration still stands on its own properties.
    res = evaluate(D)
    ok(res["n_items_FAILING"] == 0,
       f"the real declaration fails no property "
       f"({res['n_property_checks_driven']} driven across {res['n_items']} items)")
    ok(res["n_items_with_unchecked_clauses"] >= 1
       and len(res["unchecked_clauses"]) >= 1,
       "the module NAMES what it cannot check rather than reporting HOLD -- "
       "an unverifiable clause must not read as a verified one")

    if not quiet:
        print(f"[da_asymmetry_null] {_N['n'] - _N['bad']}/{_N['n']} checks, "
              f"{_N['bad']} failures | "
              f"{res['n_property_checks_driven']} properties driven, "
              f"{len(WEAKENED_V2)} weakened-v2 clauses each REFUSED, "
              f"{len(res['unchecked_clauses'])} clause(s) NAMED AS UNCHECKED")
    return 1 if _N["bad"] else 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(json.dumps(evaluate(), indent=1, default=str))
