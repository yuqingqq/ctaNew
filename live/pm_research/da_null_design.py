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
def item_8_multiplicity(d: dict, as_of: str = "2026-09-09") -> dict:
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
    #: REV 149: this was `"5" in <prose>` and "we have 5 llamas" passed it.
    #: It now COUNTS the untouched complete days and compares to the
    #: declared bar. `as_of` is fixed here so the cell tests the code
    #: rather than the hour.
    u = untouched_complete_days(d, as_of=as_of)
    val = (u["required"] >= 5 and not u["validation_is_possible"])
    return _v(agrees and now and consumed_ok and val,
              untouched=u,
              item="multiplicity", n_candidates=n, candidates=named,
              count_agrees_with_the_list=agrees,
              recorded_before_any_draw=now,
              days_this_race_runs_on=sorted(days),
              all_of_them_already_consumed=consumed_ok,
              consumed_elsewhere=c.get("already_consumed"),
              validation_needs_untouched_days=val)


# ---------------------------------------------------- rule 35, both halves
RESULT_LIMIT_FIELD = "validation_limit"
LIMIT_REFUSAL = "RESULT_DOES_NOT_STATE_ITS_VALIDATION_LIMIT"


def untouched_complete_days(d: dict, as_of: str) -> dict:
    """THE PROPERTY, COMPUTED -- how many COMPLETE untouched UTC days exist.

    REV 149. This clause was checked as `"5" in <prose>`, which **"we have
    5 llamas" satisfies**. That is rule 15's silent-checker failure wearing
    a different hat: an instrument that cannot fail is not an instrument.

    So it is counted now. A day is untouched when it is on or after the
    PROTECTED-FROM date (rule 34) and not in the consumed list; it is
    COMPLETE when it ends strictly before `as_of`. **`as_of` is a
    PARAMETER, not the wall clock**, so the battery can drive both sides of
    the predicate -- a cell whose verdict slides with the hour tests the
    clock, not the code (DA 150's lesson, in my own module)."""
    import datetime as _dt
    c = d.get("consumed_days") or {}
    prot = c.get("protected_from_utc_date")
    if not prot:
        raise DesignRefused(
            "PARTIAL INPUT: the declaration names no `protected_from_utc_"
            "date`, so the untouched set has no start and cannot be "
            "counted. Rule 34 makes that a hard boundary, not a default.")
    consumed = set(c.get("already_consumed") or []) | set(
        c.get("september_days_consumed_by_this_programme") or [])
    start = _dt.date.fromisoformat(prot)
    today = _dt.date.fromisoformat(as_of)
    days, cur = [], start
    while cur < today:                       # strictly before => COMPLETE
        if cur.isoformat() not in consumed:
            days.append(cur.isoformat())
        cur += _dt.timedelta(days=1)
    need = int(c.get("min_untouched_complete_days", 5))
    #: and the DATE the bar is reached, so the answer is a plan and not
    #: just a refusal.
    reach, k, probe = None, 0, start
    while k < need and (probe - start).days < 400:
        if probe.isoformat() not in consumed:
            k += 1
        probe += _dt.timedelta(days=1)
    if k >= need:
        reach = probe.isoformat()
    return {"as_of": as_of, "protected_from": prot,
            "n_untouched_complete_days": len(days),
            "untouched_complete_days": days,
            "required": need,
            "validation_is_possible": len(days) >= need,
            "date_the_bar_is_reached": reach,
            "why": ("counted, never matched as text -- REV 149's falsifier "
                    "for the old check was 'we have 5 llamas'")}


def require_validation_limit(result: dict, d: dict) -> dict:
    """RULE 35: THE LIMIT TRAVELS ON THE RESULT, OR THE RESULT DOES NOT EMIT.

    The cannot-validate limit lived in the declaration and nothing made the
    null's OUTPUT carry it, so a reader holding the number and not the
    document could take a null result as validation. **A reader resolves
    FIELDS; nobody reads the design doc beside the number** (rule 13's
    reasoning). This is the emitter guard: call it on every result before
    publishing, and a result that cannot state its own limit REFUSES."""
    v = (d.get("validation_limit") or {})
    want = v.get("REQUIRED_VALUE")
    if not want:
        raise DesignRefused(
            "PARTIAL INPUT: the declaration carries no REQUIRED_VALUE for "
            f"`{RESULT_LIMIT_FIELD}`, so there is no limit to enforce and "
            "this guard would admit everything.")
    if not isinstance(result, dict):
        raise DesignRefused("PARTIAL INPUT: the result is not a mapping.")
    got = result.get(RESULT_LIMIT_FIELD)
    if got is None:
        raise DesignRefused(
            f"REFUSED {LIMIT_REFUSAL}: this result carries no "
            f"`{RESULT_LIMIT_FIELD}`. It runs on CONSUMED days and cannot "
            f"validate anything; a number published without that field "
            f"reads as a validated result to anyone who does not also have "
            f"the declaration.")
    if str(got).strip() != str(want).strip():
        raise DesignRefused(
            f"REFUSED {LIMIT_REFUSAL}: this result's "
            f"`{RESULT_LIMIT_FIELD}` is not the declared limit. A limit "
            f"paraphrased at the emit is a limit that can be softened at "
            f"the emit.")
    return {"field": RESULT_LIMIT_FIELD, "carried": True,
            "value_matches_the_declaration": True}


def stamp_validation_limit(result: dict, d: dict) -> dict:
    """Attach the declared limit to a result, so producers can comply."""
    v = (d.get("validation_limit") or {})
    if not v.get("REQUIRED_VALUE"):
        raise DesignRefused("PARTIAL INPUT: no REQUIRED_VALUE to stamp.")
    out = dict(result)
    out[RESULT_LIMIT_FIELD] = v["REQUIRED_VALUE"]
    return out


ITEMS = (item_1_permutation, item_2_matching, item_3_minimum_sample,
         item_4_estimand, item_5_latency, item_6_exclusions,
         item_7_falsifiers, item_8_multiplicity)


# ======================================================================
# FALSIFIERS. Each item is driven on the REAL declaration (must hold), on
# a MUTATION that breaks exactly its clause (must fail), and on a partial
# input (must refuse). A mutation is built by DAMAGING THE REAL FILE, not
# by writing a toy -- a fixture written by the same hand as the checker
# proves nothing about the producer (DA 125).
# ======================================================================
import copy  # noqa: E402


def _break(d: dict, path: list, value) -> dict:
    m = copy.deepcopy(d)
    o = m
    for k in path[:-1]:
        o = o[k]
    if value is None:
        o.pop(path[-1], None)
    else:
        o[path[-1]] = value
    return m


def raises_named(fn, name) -> bool:
    """Did `fn` refuse with THIS refusal name, checked on the FULL message.

    DA 177: my first version of the rule-35 cell asserted the refusal name
    against `refuses()`, which TRUNCATES to 44 chars -- so the check was
    matching a cut string and failed on a guard that was working. A
    text-matching defect inside the round about text-matching defects; the
    name is compared against the whole exception now.
    """
    try:
        fn()
        return False
    except DesignRefused as e:
        return name in str(e)
    except Exception:                                         # noqa: BLE001
        return False


def refuses(fn, label):
    try:
        fn()
        return f"{label}: DID NOT REFUSE"
    except DesignRefused as e:
        return f"REFUSED -- {str(e)[:40]}"
    except Exception as e:                                    # noqa: BLE001
        return f"{label}: WRONG EXCEPTION {type(e).__name__}"


def selftest() -> tuple:
    checks: list = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    D = load()
    rp = (HERE.parent.parent / "data" / "pm_5min" / "derived"
          / "be_daybook_receipt_20260904_btc__L250ms__EV21.json")
    receipt = json.loads(rp.read_text()) if rp.is_file() else None

    ck("ITEM 1 THE PERMUTATION IS AT THE GENERATION and the declaration "
       "holds -- and it FAILS if the unit is anything else, if the key is "
       "not (slug, side, gen), or if ***rows and fills are not EXPLICITLY "
       "excluded***, which is the silence that let a row-matched null run "
       "for four register entries",
       item_1_permutation(D)["verdict"] == HOLDS
       and item_1_permutation(_break(D, ["unit"], "ROW"))["verdict"] == VIOLATED
       and item_1_permutation(_break(D, ["permutation", "resampled_key"],
                                     ["slug", "side"]))["verdict"] == VIOLATED
       and item_1_permutation(_break(D, ["permutation", "NOT_resampled"],
                                     ["tranches"]))["verdict"] == VIOLATED,
       f"real -> {item_1_permutation(D)['verdict']}; unit=ROW, short key, "
       f"and rows/fills not excluded each -> VIOLATED")

    ck("ITEM 2 THE MATCHING: matched on action count, side and hour, "
       "compared on net value and rho -- and it FAILS if a matched "
       "variable has NO ENFORCEMENT clause (named but uncontrolled), if a "
       "***PROXY like harm share is substituted***, or if a failed match "
       "is anything other than a refusal",
       item_2_matching(D)["verdict"] == HOLDS
       and item_2_matching(_break(D, ["matching", "enforcement"],
                                  {"side": "x"}))["verdict"] == VIOLATED
       and item_2_matching(_break(D, ["matching", "compared_on"],
                                  ["harm_share"]))["verdict"] == VIOLATED
       and item_2_matching(_break(D, ["matching", "on_failure"],
                                  "rebalance quietly"))["verdict"] == VIOLATED,
       f"real -> {item_2_matching(D)['verdict']}; enforcement stripped, "
       f"harm_share substituted, silent rebalance -> VIOLATED each")

    i3 = item_3_minimum_sample(D)
    ck("ITEM 3 THE MINIMUM SAMPLE clears rule 6's floor of 200 from a "
       "PRE-DECLARED params file -- and FAILS at 199, fails if the bar is "
       "read off the receipt, and ***fails if the wall clock is stated "
       "without being declared unestablished***, because an unpriced race "
       "is authorised in a sentence and runs for days",
       i3["verdict"] == HOLDS and i3["n_draws"] == 500
       and item_3_minimum_sample(
           _break(D, ["minimum_sample", "n_draws"], 199))["verdict"] == VIOLATED
       and item_3_minimum_sample(
           _break(D, ["minimum_sample", "source_of_the_bar"],
                  "the receipt's own claim"))["verdict"] == VIOLATED
       and item_3_minimum_sample(
           _break(D, ["wall_clock", "status"], "fine"))["verdict"] == VIOLATED,
       f"n=500 -> {i3['verdict']}; {i3['implied_waves_at_this_concurrency']} "
       f"wave(s) at concurrency {i3['concurrency']} -> "
       f"{i3['implied_wall_clock_hours']} h implied, cost declared "
       f"unestablished {i3['cost_is_declared_unestablished']}")

    ck("ITEM 4 THE ESTIMAND IS R-801 and the markout is a DIAGNOSTIC -- "
       "and it FAILS if the markout is promoted to the tested quantity, "
       "***including by LEAKING INTO compared_on while the estimand text "
       "stays perfectly correct***, which a text-only check would pass",
       item_4_estimand(D)["verdict"] == HOLDS
       and item_4_estimand(_break(D, ["estimand", "the_5s_markout_is"],
                                  "the tested quantity"))["verdict"] == VIOLATED
       and item_4_estimand(_break(D, ["matching", "compared_on"],
                                  ["net_value_cents", "rho_x",
                                   "markout_cents"]))["verdict"] == VIOLATED,
       f"real -> {item_4_estimand(D)['verdict']}; markout promoted, and "
       f"markout leaked into compared_on with the estimand text intact "
       f"-> VIOLATED both")

    ck("ITEM 5 LATENCY IS IN THE ESTIMAND and L is READ FROM THE BOOK -- "
       "it FAILS if L is typed into the declaration instead, because a "
       "typed L is a claim about a book rather than a fact from it",
       item_5_latency(D)["verdict"] == HOLDS
       and item_5_latency(_break(D, ["latency", "L_place_ms_source"],
                                 "250.0, typed here"))["verdict"] == VIOLATED
       and item_5_latency(_break(D, ["latency", "predicate"],
                                 "value everything"))["verdict"] == VIOLATED,
       f"real -> {item_5_latency(D)['verdict']}; typed L and a predicate "
       f"that values everything -> VIOLATED")

    i6 = item_6_exclusions(D, receipt)
    ck("ITEM 6 THE POPULATION RECONCILES AND EVERY EXCLUSION CARRIES ITS "
       "COUNT: 328,578 covered + 29,530 uncovered = 358,108, the declared "
       "coverage RECOMPUTES from those counts, and ***it AGREES WITH THE "
       "09-04 BOOK RECEIPT*** -- and it FAILS on a moved count even when "
       "the coverage figure is left untouched",
       i6["verdict"] == HOLDS
       and i6["covered_plus_uncovered_equals_n"] is True
       and i6["coverage_recomputes_from_the_counts"] is True
       and i6["agrees_with_the_book_receipt"] is True
       and item_6_exclusions(
           _break(D, ["population_09_04", "n_uncovered"], 29531),
           receipt)["verdict"] == VIOLATED,
       f"{i6['n_covered']} + {i6['n_uncovered']} = "
       f"{i6['n_reference_generations']}, coverage "
       f"{i6['declared_coverage']} recomputes; receipt agrees "
       f"{i6['agrees_with_the_book_receipt']}; one count moved -> VIOLATED")

    ck("ITEM 7 IT SHIPS BOTH FALSIFIERS -- a POSITIVE CONTROL the null "
       "must flag (an outcome-aware oracle arm landing in the tail, which "
       "is what demonstrates POWER) and a KNOWN-BAD it must refuse. It "
       "fails if either is missing, because ***a zero from an instrument "
       "that never proved it can fire is not a result***",
       item_7_falsifiers(D)["verdict"] == HOLDS
       and item_7_falsifiers(
           _break(D, ["falsifiers", "positive_control_the_null_MUST_flag"],
                  "it should look sensible"))["verdict"] == VIOLATED
       and item_7_falsifiers(
           _break(D, ["falsifiers", "known_bad_the_null_MUST_refuse"],
                  None))["verdict"] == VIOLATED,
       f"real -> {item_7_falsifiers(D)['verdict']}; a vague control and a "
       f"missing known-bad -> VIOLATED")

    i8 = item_8_multiplicity(D)
    ck("ITEM 8 THE MULTIPLICITY IS RECORDED NOW and the CONSUMED DAYS are "
       "named: 2 candidates matching the list, recorded before any draw, "
       "and ***all four days this race runs on are ALREADY CONSUMED***, so "
       "validation needs later untouched days. It fails if the count "
       "disagrees with the list -- a number typed beside a list",
       i8["verdict"] == HOLDS and i8["n_candidates"] == 2
       and i8["all_of_them_already_consumed"] is True
       and item_8_multiplicity(
           _break(D, ["multiplicity", "n_candidates_in_this_race"],
                  1))["verdict"] == VIOLATED
       and item_8_multiplicity(
           _break(D, ["consumed_days",
                      "september_days_consumed_by_this_programme"],
                  ["2026-09-03"]))["verdict"] == VIOLATED,
       f"{i8['n_candidates']} candidates {i8['candidates']}, all consumed "
       f"{i8['all_of_them_already_consumed']}; count typed as 1, and a day "
       f"dropped from the consumed list -> VIOLATED")

    # ---- REV 149 / rule 35: both halves of the defect ------------------
    llamas = _break(D, ["consumed_days", "validation_requires"],
                    "we have 5 llamas")
    u_now = untouched_complete_days(D, as_of="2026-09-09")
    u_then = untouched_complete_days(D, as_of="2026-09-13")
    u_early = untouched_complete_days(D, as_of="2026-09-09")
    ck("REV 149 (1) THE FIVE-DAY CLAUSE IS COUNTED, NOT MATCHED AS TEXT -- "
       "***REV's own falsifier, 'we have 5 llamas', satisfied the old "
       "`\"5\" in <prose>` check and is now INERT***, because the verdict "
       "no longer reads that field at all: it counts COMPLETE untouched "
       "UTC days from the protected-from date and compares to the declared "
       "bar",
       item_8_multiplicity(llamas, as_of="2026-09-09")["verdict"] == HOLDS
       and item_8_multiplicity(D, as_of="2026-09-09")["verdict"] == HOLDS
       and u_now["n_untouched_complete_days"] == 1
       and u_now["validation_is_possible"] is False,
       f"'we have 5 llamas' no longer moves the verdict; counted instead: "
       f"{u_now['n_untouched_complete_days']} untouched complete day(s) "
       f"{u_now['untouched_complete_days']} against a bar of "
       f"{u_now['required']} -> validation possible "
       f"{u_now['validation_is_possible']}")

    ck("REV 149 (1b) AND THE COUNT MOVES WITH THE EVIDENCE RATHER THAN "
       "WITH THE HOUR: driven at a FIXED `as_of`, the untouched set grows "
       "from 1 day to 5 and the verdict flips to validation-possible on "
       "the date rule 34 predicts. ***A consumed day inside the window is "
       "NOT counted***, which is the whole point of the boundary",
       u_then["n_untouched_complete_days"] == 5
       and u_then["validation_is_possible"] is True
       and u_now["date_the_bar_is_reached"] == "2026-09-13"
       and "2026-09-06" not in u_then["untouched_complete_days"],
       f"as_of 2026-09-13 -> {u_then['n_untouched_complete_days']} days "
       f"{u_then['untouched_complete_days']}, possible "
       f"{u_then['validation_is_possible']}; bar reached "
       f"{u_now['date_the_bar_is_reached']}")

    ck("REV 149 (1c) AND IT REFUSES A DECLARATION WITH NO PROTECTED-FROM "
       "DATE -- rule 34 is a hard boundary, so the untouched set having no "
       "start is a REFUSAL and never a default",
       "REFUSED" in refuses(
           lambda: untouched_complete_days(
               _break(D, ["consumed_days", "protected_from_utc_date"], None),
               as_of="2026-09-09"), "no protected-from"),
       refuses(lambda: untouched_complete_days(
           _break(D, ["consumed_days", "protected_from_utc_date"], None),
           as_of="2026-09-09"), "no protected-from date"))

    good = stamp_validation_limit({"D_E_settle": 1.0}, D)
    ck("REV 149 (2) THE LIMIT TRAVELS ON THE RESULT OR THE RESULT DOES NOT "
       "EMIT (rule 35). A stamped result ADMITS; ***a result with the "
       "number and no `validation_limit` REFUSES by name***, because a "
       "reader holding the number and not the document would take it as "
       "validated; and a PARAPHRASED limit refuses too, since a limit "
       "softenable at the emit is not a limit",
       require_validation_limit(good, D)["carried"] is True
       and raises_named(
           lambda: require_validation_limit({"D_E_settle": 1.0}, D),
           LIMIT_REFUSAL)
       and raises_named(
           lambda: require_validation_limit(
               {"D_E_settle": 1.0,
                RESULT_LIMIT_FIELD: "exploratory"}, D), LIMIT_REFUSAL),
       f"stamped -> carried; bare number -> "
       f"{refuses(lambda: require_validation_limit({'D_E_settle': 1.0}, D), 'x')[:52]}; "
       f"paraphrased -> refused")

    ck("REV 149 (2b) AND THE GUARD REFUSES WHEN THE DECLARATION ITSELF "
       "CARRIES NO REQUIRED VALUE -- ***otherwise it would admit "
       "everything while looking like a guard***, which is the failure "
       "mode this whole class is made of",
       "REFUSED" in refuses(
           lambda: require_validation_limit(
               good, _break(D, ["validation_limit", "REQUIRED_VALUE"], None)),
           "no required value")
       and "REFUSED" in refuses(
           lambda: require_validation_limit("not a dict", D), "not a dict"),
       refuses(lambda: require_validation_limit(
           good, _break(D, ["validation_limit", "REQUIRED_VALUE"], None)),
           "declaration with no REQUIRED_VALUE"))

    ck("AND THE WHOLE DECLARATION REFUSES WHEN ABSENT -- a null with no "
       "declaration is exactly what rule 6 forbids, so absence must be a "
       "REFUSAL and never an empty pass",
       "REFUSED" in refuses(lambda: load("/nonexistent/decl.json"), "absent")
       and "REFUSED" in refuses(lambda: item_1_permutation({}), "empty")
       and "REFUSED" in refuses(lambda: item_8_multiplicity({}), "empty"),
       refuses(lambda: load("/nonexistent/decl.json"), "absent"))

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
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--declaration")
    a = ap.parse_args(argv)
    if a.selftest:
        return 1 if selftest()[1] else 0
    if not a.check:
        ap.error("--selftest or --check")
    D = load(a.declaration)
    rp = (HERE.parent.parent / "data" / "pm_5min" / "derived"
          / "be_daybook_receipt_20260904_btc__L250ms__EV21.json")
    receipt = json.loads(rp.read_text()) if rp.is_file() else None
    out = {}
    for fn in ITEMS:
        nm = fn.__name__
        try:
            out[nm] = (fn(D, receipt) if fn is item_6_exclusions else fn(D))
        except DesignRefused as e:
            out[nm] = {"verdict": "REFUSED", "why": str(e)}
    out["ALL_ITEMS_HOLD"] = all(v.get("verdict") == HOLDS
                                for v in out.values() if isinstance(v, dict))
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0 if out["ALL_ITEMS_HOLD"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
