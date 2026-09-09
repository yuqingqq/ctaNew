#!/usr/bin/env python3
"""DA 175: THE NULL'S DECLARED DESIGN, EVALUATED AS PREDICATES.

CLAUDE.md rule 6: declare the null BEFORE the result -- design AND minimum
sample. Every arm result this programme has produced is retracted; the next
number published must have had its null declared in advance, in the repo,
with a commit ref. `declarations/da_null_design_v1.json` is that
declaration and this module is what makes it checkable.

WHY A MODULE AND NOT A DOCUMENT. Rule 10: compute predicates, never print
conclusions -- a hardcoded verdict beside a table has contradicted the
table three times in this programme. So each of the eight required items is
a FUNCTION that evaluates the declaration and can FAIL, and each ships a
falsifier (rule 15).

THE UNIT IS NOT OPEN. USER ruling R-870: a draw resamples the CANCELLABLE
GENERATION. This module checks that the declaration says so; it does not
re-argue it and offers no alternatives.

NOTHING HERE EXECUTES A DRAW. It reads a declaration and a book receipt.

  python3 live/pm_research/da_null_design.py --selftest
  python3 live/pm_research/da_null_design.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECL = HERE / "declarations" / "da_null_design_v1.json"
HOLDS, VIOLATED = "DECLARED_AND_CONSISTENT", "DECLARATION_VIOLATED"
RULE_6_FLOOR = 200
FORBIDDEN = ("harm_share", "harmful_share", "harmful_fraction",
             "markout", "D_E0")


class DesignRefused(RuntimeError):
    """The declaration cannot be evaluated on the input given."""


def load(path=None) -> dict:
    p = Path(path) if path else DECL
    if not p.is_file():
        raise DesignRefused(f"PARTIAL INPUT: no declaration at {p}. A null "
                            f"with no declaration is the thing rule 6 "
                            f"forbids; absence is not a pass.")
    return json.loads(p.read_text())


def _v(ok, **kw):
    return dict(verdict=HOLDS if ok else VIOLATED, **kw)


# ---------------------------------------------------------------- item 1
def item_1_permutation(d: dict) -> dict:
    """(1) THE EXACT PERMUTATION -- held fixed vs resampled, AT THE
    GENERATION."""
    p = d.get("permutation") or {}
    if not p:
        raise DesignRefused("PARTIAL INPUT: no `permutation` block.")
    unit_ok = d.get("unit") == "CANCELLABLE_GENERATION"
    key_ok = p.get("resampled_key") == ["slug", "side", "gen"]
    #: the resampled thing must be the CANCEL DECISION, and rows/fills must
    #: be explicitly NOT resampled -- silence there is what let a row-matched
    #: null run for four rounds.
    notr = set(p.get("NOT_resampled") or [])
    not_ok = {"rows", "fills"} <= notr
    fixed = set(p.get("held_fixed") or [])
    fixed_ok = any("action count" in f.lower() for f in fixed) and any(
        "theta" in f.lower() for f in fixed)
    return _v(unit_ok and key_ok and not_ok and fixed_ok,
              item="permutation", unit=d.get("unit"),
              resampled_key=p.get("resampled_key"),
              rows_and_fills_explicitly_not_resampled=not_ok,
              action_count_and_theta_held_fixed=fixed_ok,
              n_held_fixed=len(fixed))


# ---------------------------------------------------------------- item 2
def item_2_matching(d: dict) -> dict:
    """(2) THE MATCHING (rule 7): matched on the DECISION VARIABLE,
    compared on the DECISION METRIC, and every match has a stated failure."""
    m = d.get("matching") or {}
    if not m:
        raise DesignRefused("PARTIAL INPUT: no `matching` block.")
    need = {"action_count", "side", "hour"}
    on = set(m.get("matched_on") or [])
    enf = m.get("enforcement") or {}
    #: EVERY matched variable needs an ENFORCEMENT clause -- a variable
    #: named but unenforced is a claim, not a control.
    enforced = need <= set(enf)
    cmp_on = [c.lower() for c in (m.get("compared_on") or [])]
    metric_ok = any("net_value" in c for c in cmp_on) and any(
        "rho" in c for c in cmp_on)
    #: and the comparison must not name a PROXY, checked against the
    #: forbidden list rather than by eye.
    proxy = [f for f in FORBIDDEN if any(f.lower() in c for c in cmp_on)]
    fail_ok = "REFUSE" in str(m.get("on_failure", "")).upper()
    return _v(need <= on and enforced and metric_ok and not proxy and fail_ok,
              item="matching", matched_on=sorted(on),
              every_matched_variable_has_an_enforcement=enforced,
              compared_on=m.get("compared_on"),
              proxies_named=proxy,
              failure_is_a_refusal=fail_ok)


# ---------------------------------------------------------------- item 3
def item_3_minimum_sample(d: dict, params: dict | None = None) -> dict:
    """(3) THE MINIMUM SAMPLE (>=200) AND THE WALL CLOCK AT N=3."""
    s = d.get("minimum_sample") or {}
    w = d.get("wall_clock") or {}
    if "n_draws" not in s:
        raise DesignRefused("PARTIAL INPUT: no declared `n_draws`.")
    n = int(s["n_draws"])
    floor_ok = n >= RULE_6_FLOOR
    #: the bar must come from a PRE-DECLARED artifact, not the receipt.
    bar_ok = "params" in str(s.get("source_of_the_bar", "")).lower()
    if params is not None:
        bar_ok = bar_ok and n >= int(params.get("min_draws_per_arm_day", 0))
    short_ok = "REFUSE" in str(s.get("short_count", "")).upper()
    conc = w.get("concurrency")
    #: THE WALL CLOCK MUST BE STATED AT N=3 -- and if it is not
    #: ESTABLISHED, the declaration must SAY SO. An unstated cost is how a
    #: race that runs for days gets authorised in a sentence.
    est = w.get("read_estimate_s_per_day")
    stated = est is not None and conc == 3
    honest = "NOT ESTABLISHED" in str(w.get("status", "")).upper()
    serial_h = (est * (d.get("multiplicity") or {}).get("n_days_planned", 0)
                / 3600.0) if est else None
    waves = None
    if conc:
        nd = (d.get("multiplicity") or {}).get("n_days_planned", 0)
        waves = -(-nd // conc)
    return _v(floor_ok and bar_ok and short_ok and stated and honest,
              item="minimum_sample", n_draws=n, rule_6_floor=RULE_6_FLOOR,
              clears_floor=floor_ok, bar_is_pre_declared=bar_ok,
              short_count_refuses=short_ok, concurrency=conc,
              read_estimate_s_per_day=est,
              implied_serial_hours=(round(serial_h, 1) if serial_h else None),
              implied_waves_at_this_concurrency=waves,
              implied_wall_clock_hours=(round(est * waves / 3600.0, 1)
                                        if est and waves else None),
              cost_is_declared_unestablished=honest,
              why=("an under-sampled correct null flatters as much as a "
                   "wrong one, and an unpriced race is authorised in a "
                   "sentence and runs for days"))


# ---------------------------------------------------------------- item 4
def item_4_estimand(d: dict) -> dict:
    """(4) THE ESTIMAND IS R-801 AND THE MARKOUT IS NOT THE TESTED
    QUANTITY."""
    e = d.get("estimand") or {}
    if not e:
        raise DesignRefused("PARTIAL INPUT: no `estimand` block.")
    ruling_ok = e.get("ruling") == "R-801"
    defn = str(e.get("definition", "")).lower()
    defn_ok = "trades" in defn and "residual" in defn and "settlement" in defn
    #: the markout may appear ONLY under a key that names it a diagnostic.
    mk = str(e.get("the_5s_markout_is", "")).upper()
    mk_ok = "DIAGNOSTIC" in mk and "NEVER" in mk
    #: and it must not have leaked into the COMPARED-ON metric.
    cmp_on = [c.lower() for c in ((d.get("matching") or {})
                                  .get("compared_on") or [])]
    leak = [c for c in cmp_on if "markout" in c or "d_e0" in c]
    return _v(ruling_ok and defn_ok and mk_ok and not leak,
              item="estimand", ruling=e.get("ruling"),
              definition_names_both_legs=defn_ok,
              markout_is_declared_a_diagnostic=mk_ok,
              markout_leaked_into_the_metric=leak)


# ---------------------------------------------------------------- item 5
def item_5_latency(d: dict) -> dict:
    """(5) LATENCY IS IN THE ESTIMAND: value only tranches after t + L."""
    lat = d.get("latency") or {}
    if not lat:
        raise DesignRefused("PARTIAL INPUT: no `latency` block.")
    pred = str(lat.get("predicate", ""))
    pred_ok = ">=" in pred and "t0" in pred and "L" in pred
    #: L must be READ from the book, never typed into the declaration --
    #: a typed L is a claim about a book rather than a fact from it.
    src = str(lat.get("L_place_ms_source", "")).lower()
    src_ok = "receipt" in src and "never typed" in src
    return _v(pred_ok and src_ok, item="latency", predicate=pred,
              L_read_from_the_book=src_ok)


# ---------------------------------------------------------------- item 6
def item_6_exclusions(d: dict, receipt: dict | None = None) -> dict:
    """(6) EXCLUSIONS ARE STATUSES WITH COUNTS, AND THE POPULATION
    RECONCILES."""
    pop = d.get("population_09_04") or {}
    exc = d.get("exclusions_are_statuses") or {}
    if not pop or not exc:
        raise DesignRefused("PARTIAL INPUT: no population or exclusions.")
    n = pop.get("n_reference_generations")
    cov, unc = pop.get("n_covered"), pop.get("n_uncovered")
    #: THE RECONCILIATION, COMPUTED: covered + uncovered must BE the
    #: population, and the declared coverage must be their ratio.
    sums = (cov is not None and unc is not None and cov + unc == n)
    ratio_ok = (abs(cov / n - pop.get("coverage", -1)) < 1e-12) if n else False
    counted = all(isinstance(v, int) for k, v in exc.items()
                  if not k.startswith("note"))
    agrees = None
    if receipt is not None:
        rr = receipt.get("reference") or {}
        agrees = (rr.get("generations") == n)
    return _v(sums and ratio_ok and counted and (agrees is not False),
              item="exclusions", n_reference_generations=n,
              n_covered=cov, n_uncovered=unc,
              covered_plus_uncovered_equals_n=sums,
              declared_coverage=pop.get("coverage"),
              coverage_recomputes_from_the_counts=ratio_ok,
              every_exclusion_carries_a_count=counted,
              agrees_with_the_book_receipt=agrees)


# ---------------------------------------------------------------- item 7
def item_7_falsifiers(d: dict) -> dict:
    """(7) WHAT WOULD FALSIFY IT (rule 15): the positive control it must
    FLAG and the known-bad it must REFUSE."""
    f = d.get("falsifiers") or {}
    pos = str(f.get("positive_control_the_null_MUST_flag", ""))
    bad = str(f.get("known_bad_the_null_MUST_refuse", ""))
    #: a positive control is only a control if it is CONSTRUCTIBLE and
    #: OUTCOME-AWARE -- otherwise it cannot demonstrate power.
    pos_ok = bool(pos) and "oracle" in pos.lower() and "tail" in pos.lower()
    bad_ok = bool(bad) and ("match" in bad.lower()
                            and "matched_on" in bad)
    return _v(pos_ok and bad_ok, item="falsifiers",
              has_positive_control=pos_ok, has_known_bad=bad_ok,
              why=("a zero from an instrument that never proved it can "
                   "fire is not a result"))


# ---------------------------------------------------------------- item 8
def item_8_multiplicity(d: dict) -> dict:
    """(8) THE MULTIPLICITY, RECORDED NOW (rule 12) -- and the CONSUMED
    DAYS (rule 11)."""
    m = d.get("multiplicity") or {}
    c = d.get("consumed_days") or {}
    if not m:
        raise DesignRefused("PARTIAL INPUT: no `multiplicity` block.")
    n = m.get("n_candidates_in_this_race")
    #: the count must AGREE with the enumerated candidates -- a number
    #: typed beside a list is the shape rule 10 exists for.
    named = m.get("candidates") or []
    agrees = (n == len(named))
    now = "before any draw" in str(m.get("recorded_at", "")).lower()
    #: and the consumed days must be NAMED, and must include the days this
    #: race runs on -- a day used to choose cannot also validate.
    sept = set(c.get("september_days_consumed_by_this_programme") or [])
    days = set(m.get("days") or [])
    consumed_ok = bool(sept) and days <= sept
    val = "5" in str(c.get("validation_requires", ""))
    return _v(agrees and now and consumed_ok and val,
              item="multiplicity", n_candidates=n, candidates=named,
              count_agrees_with_the_list=agrees,
              recorded_before_any_draw=now,
              days_this_race_runs_on=sorted(days),
              all_of_them_already_consumed=consumed_ok,
              consumed_elsewhere=c.get("already_consumed"),
              validation_needs_untouched_days=val)


ITEMS = (item_1_permutation, item_2_matching, item_3_minimum_sample,
         item_4_estimand, item_5_latency, item_6_exclusions,
         item_7_falsifiers, item_8_multiplicity)
