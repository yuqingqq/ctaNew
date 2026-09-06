"""NO_MID_AT_FILL on the REAL §8.1 population -- its rate, and whether it
is SELECTIVE the way DE53's exclusion turned out to be.

WHY THIS EXISTS. `de_phase4_diag_runner.maker_pnl` values a fill's spread
capture as `sgn * (mid_at_fill - level) * 100 * shares`, and `mid_at()`
RETURNS None BEFORE A WINDOW'S FIRST QUOTE. That makes NO_MID_AT_FILL a
real exclusion, not a rounding case -- and until this module ran, NOBODY
KNEW WHETHER IT FIRED ON 0.1% OR 20% OF THE POPULATION. A spread figure
computed over an unmeasured exclusion is a figure over an unknown
denominator.

AND WHY NOW. DA's fourth oracle found DE53's generation exclusion
(1309/31122) SELECTIVE ON DURATION at the permutation floor: long
exposures are preferentially missing. If NO_MID_AT_FILL is ALSO
duration-correlated, two selective filters stack on the same axis and the
spread leg measures a systematically short-exposure subset. That is a
measurement, not a guess, and this module makes it.

THE UNIT IS THE TRANCHE, NOT THE GENERATION. DE53's exclusion drops
GENERATIONS; NO_MID_AT_FILL drops TRANCHES. The two are different
populations and their counts are not comparable term-by-term. The
duration attribute a tranche carries is ITS PARENT GENERATION'S, which is
stated in the emission rather than assumed by a reader.

DECIDES NOTHING (rule 14). Whether a selective exclusion voids a §8.1
economic number is the policy layer's call.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import da_population_audit as PA

#: The duration thresholds DA's DE53 report used, so the two readings can
#: be laid beside each other without re-bucketing one of them.
DURATION_TAIL_S = (2.0, 4.0, 8.0, 16.0)

#: Rule 6: declared before the result, and above `da_population_audit`'s
#: own declared minimum.
N_PERMUTATIONS = 400
SEED = 0

#: THE CONSUMER-SIDE LIVENESS PROBE (round 65).
#: DA's mutation audit (`p003_da_population_mutation_audit__20260905T161926Z
#: .json`) applied four mutants to `da_population_audit` -- the statistic
#: returning zero, the statistic dropping `abs`, the null not permuting, and
#: NOTHING_EXCLUDED emitted unconditionally -- and DA's own suite caught
#: 4 of 4 while THIS MODULE caught 0 OF 4, GREEN EVERY TIME. Including the
#: fourth, whose emitted string is the exact certification this module's
#: artifact cites. "Imported unchanged, a second copy is a second test" is
#: the right rule for avoiding DIVERGENCE; it is not a consumer-side
#: falsifier, and the difference is what those four mutants measured.
#: So the census no longer certifies from an oracle it has not just seen
#: FIRE and seen STAY SILENT, on two populations whose answers are fixed by
#: construction rather than by a seed.
ORACLE_PROBE_SEED = 20260906
ORACLE_PROBE_N = 96
ORACLE_PROBE_SELECTIVE_HOUR = 3
ORACLE_PROBE_RETAINED_HOUR = 9
ORACLE_PROBE_BALANCED_HOURS = (3, 4, 5, 6)


class MidCensusRefused(RuntimeError):
    """Refused rather than reporting a rate over a population it cannot
    identify."""


def _hour(epoch: float) -> int:
    return _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc).hour


def window_epoch(slug: str) -> int:
    """The window's start epoch, from the slug's own last field -- the same
    parse `select_v2_era` uses (`int(slug.rsplit('-', 1)[1])`), never a
    nearby proxy (rule 3)."""
    try:
        return int(slug.rsplit("-", 1)[1])
    except (ValueError, IndexError) as e:
        raise MidCensusRefused(
            f"REFUSED: slug {slug!r} carries no parsable window epoch; an "
            f"hour attribute invented for it would be a proxy timestamp "
            f"(rule 3)") from e


def tranche_records(reference: dict) -> list[dict]:
    """One record per tranche of the reference, carrying the attributes the
    selectivity test compares.

    `duration` IS THE PARENT GENERATION'S EXPOSURE LENGTH (`t1 - t0`), not
    the tranche's own -- a tranche is an instant. This is the same
    quantity DA tested DE53 on, which is why it is named the same.
    """
    if not isinstance(reference, dict) or not reference:
        raise MidCensusRefused(
            "REFUSED: empty reference. A rate over no population is not a "
            "rate; the caller must say which population it means.")
    out: list[dict] = []
    for slug, sides in sorted(reference.items()):
        w0 = window_epoch(slug)
        for side, gens in sorted(sides.items()):
            for g in gens:
                t0, t1 = float(g["t0"]), float(g["t1"])
                dur = t1 - t0
                hr = _hour(w0 + t0)
                for t in g.get("tranches", ()):
                    mid = t.get("mid_at_fill")
                    lvl = t.get("level")
                    mk = t.get("markout_cents_per_share")
                    out.append({
                        "slug": slug, "side": side, "hour": hr,
                        "duration": dur, "gen": g["gen"],
                        "t": float(t["t"]), "shares": t.get("shares"),
                        # THE PREDICATE `maker_pnl` ACTUALLY APPLIES: it
                        # skips on `mid is None OR level is None`, so the
                        # census must test the same disjunction or it
                        # would count a different exclusion.
                        "has_mid": not (mid is None or lvl is None),
                        "has_markout": mk is not None,
                    })
    return out


def census(records: list[dict]) -> dict:
    """The rate, as counted statuses -- never a zero, never a default."""
    if not records:
        raise MidCensusRefused(
            "REFUSED: no tranche records. Reporting 0.0% from an empty "
            "population would be an instrument that cannot fire (rule 15).")
    n = len(records)
    no_mid = sum(1 for r in records if not r["has_mid"])
    no_mk = sum(1 for r in records if not r["has_markout"])
    both = sum(1 for r in records
               if not r["has_mid"] and not r["has_markout"])
    sh_all = sum(float(r["shares"] or 0.0) for r in records)
    sh_no_mid = sum(float(r["shares"] or 0.0)
                    for r in records if not r["has_mid"])
    return {
        "n_tranches": n,
        "NO_MID_AT_FILL": no_mid,
        "NO_MARKOUT": no_mk,
        "NO_MID_AND_NO_MARKOUT": both,
        "VALUED_BY_MAKER_PNL": n - no_mid - sum(
            1 for r in records if r["has_mid"] and not r["has_markout"]),
        "no_mid_rate": round(no_mid / n, 6),
        "no_markout_rate": round(no_mk / n, 6),
        "shares_total": sh_all,
        "shares_no_mid": sh_no_mid,
        "shares_no_mid_rate": (round(sh_no_mid / sh_all, 6)
                               if sh_all else None),
        "unit": "TRANCHE (not generation); DE53's exclusion counts "
                "GENERATIONS and the two are not comparable term-by-term",
        "why_not_a_zero": "a fill whose entry mid is unknown has an "
                          "UNKNOWN spread, not a zero one (rule 4)",
    }


def duration_tail(records: list[dict],
                  thresholds: tuple = DURATION_TAIL_S) -> dict:
    """Exclusion rate in the duration tail against the base rate -- the
    shape DA's DE53 reading turned on.

    A per-threshold rate ABOVE the base rate is the observation; whether
    it is significant is the permutation test's job, not this table's."""
    n = len(records)
    base = sum(1 for r in records if not r["has_mid"]) / n if n else None
    rows = []
    for th in thresholds:
        sub = [r for r in records if r["duration"] >= th]
        ex = sum(1 for r in sub if not r["has_mid"])
        rows.append({
            "threshold_s": th, "n_at_or_above": len(sub),
            "n_excluded": ex,
            "rate": (round(ex / len(sub), 6) if sub else None),
            "status": "OK" if sub else "NO_TRANCHES_AT_THIS_THRESHOLD",
        })
    med_e = _median([r["duration"] for r in records if not r["has_mid"]])
    med_r = _median([r["duration"] for r in records if r["has_mid"]])
    return {
        "base_rate": (round(base, 6) if base is not None else None),
        "median_duration_s_excluded": med_e,
        "median_duration_s_retained": med_r,
        "rows": rows,
        "reading_is_the_callers": "rates are REPORTED; whether the tail "
                                  "differs is the permutation test below "
                                  "(rule 10 -- no verdict is printed here)",
    }


def _median(xs: list[float]):
    if not xs:
        return None
    s = sorted(xs)
    m = len(s) // 2
    return round(s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m]), 6)


def _probe_rec(hour: int, i: int) -> dict:
    """One synthetic record for the liveness probe. Only `hour` varies; it
    is a small-cardinality int, which `_levels` treats CATEGORICALLY, so
    the two arms' answers do not depend on bucketing."""
    return {"slug": "probe", "side": "BUY_UP", "hour": hour,
            "duration": 1.0, "gen": i, "t": float(i), "shares": 1.0,
            "has_mid": False, "has_markout": True}


def oracle_probe(*, n_permutations: int = N_PERMUTATIONS,
                 seed: int = ORACLE_PROBE_SEED) -> dict:
    """Drive the IMPORTED oracle on two populations whose answers are fixed
    BY CONSTRUCTION, and report what it said. Decides nothing itself.

    * **SELECTIVE arm** -- every excluded record in hour 3, every retained
      record in hour 9. The two hour distributions are DISJOINT, so the
      total-variation distance is 1.0 and no permutation can reach it. A
      working oracle MUST flag `hour`.
    * **BALANCED arm** -- both sides carry the SAME hour multiset. The
      distance is 0.0 exactly, so a working oracle must flag NOTHING.

    Neither arm depends on a lucky seed: one is maximally separated and the
    other is exactly equal, which is why this is a control and not a draw."""
    n, half = ORACLE_PROBE_N, ORACLE_PROBE_N
    sel_ex = [_probe_rec(ORACLE_PROBE_SELECTIVE_HOUR, i) for i in range(half)]
    sel_re = [_probe_rec(ORACLE_PROBE_RETAINED_HOUR, i) for i in range(half)]
    hours = ORACLE_PROBE_BALANCED_HOURS
    bal_ex = [_probe_rec(hours[i % len(hours)], i) for i in range(n)]
    bal_re = [_probe_rec(hours[i % len(hours)], i) for i in range(n)]
    selective = PA.compare(sel_ex, sel_re, attrs=("hour",),
                           n_permutations=n_permutations, seed=seed)
    balanced = PA.compare(bal_ex, bal_re, attrs=("hour",),
                          n_permutations=n_permutations, seed=seed)
    sel_hour = (selective.get("attributes") or {}).get("hour") or {}
    bal_hour = (balanced.get("attributes") or {}).get("hour") or {}
    fired = (selective.get("status") == "COMPARED"
             and "hour" in (selective.get("selective_attributes") or [])
             and float(sel_hour.get("tvd") or 0.0) > 0.0)
    silent = (balanced.get("status") == "COMPARED"
              and not (balanced.get("selective_attributes") or []))
    return {
        "oracle": "da_population_audit.compare",
        "oracle_sha256_prefix": _oracle_sha16(),
        "n_permutations": n_permutations, "seed": seed,
        "selective_arm": {
            "construction": "excluded all in hour "
                            f"{ORACLE_PROBE_SELECTIVE_HOUR}, retained all in "
                            f"hour {ORACLE_PROBE_RETAINED_HOUR} -- DISJOINT, "
                            "so TVD is 1.0 by construction",
            "status": selective.get("status"),
            "selective_attributes": selective.get("selective_attributes"),
            "tvd": sel_hour.get("tvd"),
            "p_permutation": sel_hour.get("p_permutation"),
            "ORACLE_FIRED": fired},
        "balanced_arm": {
            "construction": "both sides carry the SAME hour multiset over "
                            f"{list(ORACLE_PROBE_BALANCED_HOURS)} -- TVD is "
                            "0.0 by construction",
            "status": balanced.get("status"),
            "selective_attributes": balanced.get("selective_attributes"),
            "tvd": bal_hour.get("tvd"),
            "p_permutation": bal_hour.get("p_permutation"),
            "ORACLE_STAYED_SILENT": silent},
        "ORACLE_IS_LIVE": bool(fired and silent),
        "why_this_exists": (
            "DA's mutation audit applied four mutants to this oracle; DA's "
            "suite caught 4 of 4 and THIS CONSUMER caught 0 of 4, green "
            "every time -- including the mutant that emits NOTHING_EXCLUDED "
            "unconditionally, the exact string this module's artifact "
            "cites. A certification taken from an instrument that has not "
            "been seen to fire is not a certification (rule 15)"),
        "decides_nothing": "REPORTED (rule 14).",
    }


def _oracle_sha16() -> str:
    import hashlib
    return hashlib.sha256(
        Path(PA.__file__).read_bytes()).hexdigest()[:16]


def assert_oracle_live(*, n_permutations: int = N_PERMUTATIONS,
                       seed: int = ORACLE_PROBE_SEED) -> dict:
    """REFUSE to certify from an oracle that has not just been seen to fire
    AND to stay silent. This is the consumer-side falsifier the four
    surviving mutants proved was missing."""
    probe = oracle_probe(n_permutations=n_permutations, seed=seed)
    if not probe["selective_arm"]["ORACLE_FIRED"]:
        raise MidCensusRefused(
            "REFUSED: the imported oracle DID NOT FLAG a maximally "
            "selective exclusion (hours disjoint, TVD 1.0 by "
            f"construction). It answered {probe['selective_arm']}. No "
            "NOTHING_EXCLUDED or INDISTINGUISHABLE certification may be "
            "taken from an instrument in this state.")
    if not probe["balanced_arm"]["ORACLE_STAYED_SILENT"]:
        raise MidCensusRefused(
            "REFUSED: the imported oracle FLAGGED an exactly balanced "
            "exclusion (identical hour multisets, TVD 0.0 by "
            f"construction). It answered {probe['balanced_arm']}. An "
            "oracle that flags everything certifies nothing.")
    return probe


def selectivity(records: list[dict], *, n_permutations: int = N_PERMUTATIONS,
                seed: int = SEED) -> dict:
    """DA's fourth oracle, pointed at MY exclusion.

    The instrument is `da_population_audit.compare` UNCHANGED and
    IMPORTED, never re-implemented here: a second copy of a test is a
    second test.

    ROUND 65: importing it unchanged avoids DIVERGENCE and does NOT give
    this module a falsifier -- DA's mutation audit measured exactly that
    gap, 0 of 4 caught here against 4 of 4 in DA's own suite. So every
    call now runs `assert_oracle_live()` FIRST and refuses if the oracle
    cannot be seen to fire on a planted selective exclusion and to stay
    silent on an exactly balanced one. The probe result travels with the
    answer, so a reader can see the instrument was live WHEN THIS RAN
    rather than when it was written."""
    probe = assert_oracle_live(n_permutations=n_permutations)
    ex = [r for r in records if not r["has_mid"]]
    re_ = [r for r in records if r["has_mid"]]
    out = PA.compare(ex, re_,
                     attrs=("slug", "side", "hour", "duration"),
                     n_permutations=n_permutations, seed=seed)
    out["oracle_liveness_probe"] = probe
    return out


def denominator_check(records: list[dict]) -> dict:
    """DO THE TWO LEGS CARRY THE SAME DENOMINATOR? -- the open question
    from round 57, answered from the population rather than from reading.

    `build_reference` ALREADY FILTERS tranches whose markout is None, so
    the markout leg arrives complete. `maker_pnl` then accumulates BOTH
    legs inside the SAME `mid is None` guard -- so the markout it reports
    is silently RESTRICTED to the mid-known subset even though every
    tranche has a markout. The two legs therefore share a denominator,
    but it is the SPREAD leg's, and the markout leg is truncated below
    what its inputs support."""
    have_mk = [r for r in records if r["has_markout"]]
    both = [r for r in records if r["has_markout"] and r["has_mid"]]
    return {
        "n_with_markout": len(have_mk),
        "n_with_markout_and_mid": len(both),
        "n_markout_dropped_by_the_mid_guard": len(have_mk) - len(both),
        "legs_share_a_denominator": len(have_mk) == len(both),
        "predicate": "the markout leg is complete iff no tranche with a "
                     "markout is dropped by the mid guard",
        "why_it_matters": "`reconcile_maker_pnl` compares the reference's "
                          "markout to the replay's received markout. If "
                          "the mid guard truncates the reference leg, the "
                          "directional predicate |replay| <= |reference| "
                          "can fail for a reason that is not the policy's",
    }


# ---------------------------------------------------------------- loading

def load_reference(path: Path) -> dict:
    """The reference from a §8.1 arms cache, or a refusal naming what is
    missing -- never a silently empty population."""
    if not path.exists():
        raise MidCensusRefused(f"REFUSED: no cache at {path}")
    obj = pickle.loads(path.read_bytes())
    if not isinstance(obj, dict) or "fr" not in obj:
        raise MidCensusRefused(
            f"REFUSED: {path} is not a §8.1 arms cache (no 'fr' key); its "
            f"keys are {sorted(obj) if isinstance(obj, dict) else type(obj)}")
    fr = obj["fr"]
    for k in ("reference", "statuses", "population"):
        if k not in fr:
            raise MidCensusRefused(
                f"REFUSED: cache 'fr' lacks {k!r}; the population it "
                f"describes cannot be named, so no rate over it can be "
                f"reported (rule 8)")
    return fr


# --------------------------------------------------------------- selftest

def _mk(dur, mid=0.5, lvl=0.5, mk=1.0, slug="btc-updown-5m-1787579400",
        side="BUY_UP", gen=1, t=0.0):
    return {"gen": gen, "t0": 0.0, "t1": dur, "level": lvl,
            "tranches": [{"t": t, "shares": 5.0,
                          "markout_cents_per_share": mk,
                          "mid_at_fill": mid, "level": lvl}]}


def selftest() -> int:
    """FALSIFIERS IN BOTH DIRECTIONS (rule 15): a positive control the
    census MUST flag, and a known-bad input it MUST refuse."""
    checks = 0
    fails = []

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(msg)

    # --- POSITIVE CONTROL 1: the counter fires on a None mid ------------
    ref = {"btc-updown-5m-1787579400": {
        "BUY_UP": [_mk(1.0, mid=None), _mk(1.0, mid=0.52)],
        "SELL_UP": []}}
    rec = tranche_records(ref)
    c = census(rec)
    ok(c["NO_MID_AT_FILL"] == 1 and c["n_tranches"] == 2
       and abs(c["no_mid_rate"] - 0.5) < 1e-9,
       f"FALSIFIER-1: a None mid must be COUNTED -- got {c}")

    # --- POSITIVE CONTROL 2: it fires on a None LEVEL too, because that
    # is the other half of the disjunction `maker_pnl` applies.
    ref2 = {"btc-updown-5m-1787579400": {
        "BUY_UP": [_mk(1.0, lvl=None)], "SELL_UP": []}}
    ok(census(tranche_records(ref2))["NO_MID_AT_FILL"] == 1,
       "FALSIFIER-2: `maker_pnl` skips on `mid is None OR level is None`; "
       "a census that only tested the mid would count a DIFFERENT "
       "exclusion from the one the producer applies")

    # --- NEGATIVE CONTROL: a clean population must read exactly 0, and
    # that 0 is only meaningful because 1 and 2 proved the counter fires.
    ref3 = {"btc-updown-5m-1787579400": {
        "BUY_UP": [_mk(1.0), _mk(2.0)], "SELL_UP": [_mk(3.0)]}}
    ok(census(tranche_records(ref3))["NO_MID_AT_FILL"] == 0,
       "NEGATIVE CONTROL: a fully-mid'd population must read 0")

    # --- KNOWN-BAD INPUTS IT MUST REFUSE --------------------------------
    for bad, why in (({}, "empty reference"),
                     ("not a dict", "non-dict reference")):
        try:
            tranche_records(bad)
            ok(False, f"REFUSAL: must refuse a {why}")
        except MidCensusRefused:
            ok(True, "")
    try:
        census([])
        ok(False, "REFUSAL: must refuse an empty record list rather than "
                  "reporting 0.0% from nothing")
    except MidCensusRefused:
        ok(True, "")
    try:
        tranche_records({"no-epoch-here": {"BUY_UP": [_mk(1.0)],
                                           "SELL_UP": []}})
        ok(False, "REFUSAL: must refuse a slug with no parsable epoch "
                  "rather than inventing an hour (rule 3)")
    except MidCensusRefused:
        ok(True, "")

    # --- the duration tail must SEPARATE, not merely run -----------------
    ref4 = {"btc-updown-5m-1787579400": {
        "BUY_UP": [_mk(0.05), _mk(0.05), _mk(0.05), _mk(8.0, mid=None)],
        "SELL_UP": []}}
    dt = duration_tail(tranche_records(ref4), thresholds=(4.0,))
    ok(dt["rows"][0]["rate"] == 1.0 and abs(dt["base_rate"] - 0.25) < 1e-9,
       f"FALSIFIER-3: a tail-only exclusion must show a tail rate ABOVE "
       f"the base rate -- got {dt['rows'][0]['rate']} vs {dt['base_rate']}")
    ok(duration_tail(tranche_records(ref4),
                     thresholds=(1e9,))["rows"][0]["status"]
       == "NO_TRANCHES_AT_THIS_THRESHOLD",
       "a threshold no tranche reaches must be a STATUS, never a 0.0 rate")

    # --- the denominator check must be able to say NO -------------------
    ok(denominator_check(tranche_records(ref))
       ["n_markout_dropped_by_the_mid_guard"] == 1,
       "FALSIFIER-4: a tranche WITH a markout dropped by the mid guard "
       "must be counted -- this is the truncation the check exists for")
    ok(denominator_check(tranche_records(ref3))
       ["legs_share_a_denominator"] is True,
       "NEGATIVE CONTROL: with no mid gap the legs must agree")

    # --- selectivity must refuse an under-sampled null (rule 6) ---------
    try:
        selectivity(tranche_records(ref), n_permutations=10)
        ok(False, "REFUSAL: must not accept 10 permutations")
    except PA.PopulationAuditRefused:
        ok(True, "")

    # ================= ROUND 65: THE CONSUMER-SIDE FALSIFIER =============
    # DA's mutation audit: 4 mutants on the imported oracle, DA's suite
    # 4 of 4 CAUGHT, this module 0 OF 4, green every time. Everything below
    # exists because that gap was real and this file could not see it.
    #
    # POSITIVE CONTROL, AND IT MUST ADMIT: on the SHIPPED oracle the probe
    # must fire on a maximally selective exclusion and stay silent on an
    # exactly balanced one. A probe that only ever refuses is rule 16's
    # control that cannot pass.
    probe = oracle_probe()
    ok(probe["selective_arm"]["ORACLE_FIRED"] is True
       and probe["selective_arm"]["tvd"] == 1.0
       and probe["balanced_arm"]["ORACLE_STAYED_SILENT"] is True
       and probe["balanced_arm"]["tvd"] == 0.0
       and probe["ORACLE_IS_LIVE"] is True,
       f"ORACLE LIVENESS, BOTH DIRECTIONS ON THE SHIPPED INSTRUMENT: the "
       f"oracle must FLAG a disjoint-hour exclusion (got tvd "
       f"{probe['selective_arm']['tvd']}, p "
       f"{probe['selective_arm']['p_permutation']}, fired "
       f"{probe['selective_arm']['ORACLE_FIRED']}) and STAY SILENT on an "
       f"identical-multiset one (got tvd {probe['balanced_arm']['tvd']}, "
       f"silent {probe['balanced_arm']['ORACLE_STAYED_SILENT']}). "
       f"ORACLE_IS_LIVE={probe['ORACLE_IS_LIVE']}")
    # DEFENSIVELY, so the suite reaches its summary and prints its FAIL
    # lines: a bare call here aborted the whole selftest with a traceback
    # under every mutant, which is rc=1 -- red, but not red BY NAME.
    try:
        ok(assert_oracle_live()["ORACLE_IS_LIVE"] is True,
           "assert_oracle_live must ADMIT the shipped oracle -- a guard "
           "shown only to refuse is not a guard (SEAT_PROTOCOL 16)")
    except MidCensusRefused as e:
        ok(False, f"assert_oracle_live REFUSED THE SHIPPED ORACLE: {e}")

    # THE FOUR MUTANT SHAPES, STUBBED AT THE CONSUMER BOUNDARY. Each is the
    # ANSWER the corresponding mutant makes the oracle give; the census
    # must REFUSE to certify from every one of them.
    _real_compare = PA.compare
    try:
        def _stub_nothing_excluded(*a, **k):
            return {"status": "NOTHING_EXCLUDED", "n_excluded": 0,
                    "attributes": {}, "selective_attributes": []}
        PA.compare = _stub_nothing_excluded
        try:
            selectivity(tranche_records(ref))
            ok(False, "MUTANT 4 (STATUS_always_nothing_excluded): the "
                      "census CERTIFIED from an oracle that emits "
                      "NOTHING_EXCLUDED unconditionally -- the exact string "
                      "the artifact cites")
        except MidCensusRefused as e:
            ok("DID NOT FLAG" in str(e),
               "MUTANT 4 CAUGHT BY NAME: an oracle emitting "
               "NOTHING_EXCLUDED unconditionally is REFUSED, because it "
               "never flagged the planted selective exclusion")

        def _stub_zero_statistic(*a, **k):
            return {"status": "COMPARED", "n_excluded": 1, "n_retained": 1,
                    "attributes": {"hour": {"status": "COMPARED", "tvd": 0.0,
                                            "p_permutation": 1.0,
                                            "verdict":
                                                "INDISTINGUISHABLE_AT_THIS_N"}},
                    "selective_attributes": []}
        PA.compare = _stub_zero_statistic
        try:
            selectivity(tranche_records(ref))
            ok(False, "MUTANTS 1-3 (statistic zero / abs dropped / null not "
                      "permuting): the census CERTIFIED from an oracle that "
                      "never flags anything")
        except MidCensusRefused as e:
            ok("DID NOT FLAG" in str(e),
               "MUTANTS 1-3 CAUGHT BY NAME: every always-PASS shape -- a "
               "zero statistic, a signed statistic that cancels, and a null "
               "that does not permute -- reaches the consumer as 'nothing "
               "is ever selective', and that is now REFUSED")

        def _stub_flags_everything(*a, **k):
            return {"status": "COMPARED", "n_excluded": 1, "n_retained": 1,
                    "attributes": {"hour": {"status": "COMPARED", "tvd": 1.0,
                                            "p_permutation": 0.0,
                                            "verdict":
                                                "DISTRIBUTION_DIFFERS"}},
                    "selective_attributes": ["hour"]}
        PA.compare = _stub_flags_everything
        try:
            selectivity(tranche_records(ref))
            ok(False, "THE OTHER DIRECTION: the census certified from an "
                      "oracle that flags everything")
        except MidCensusRefused as e:
            ok("FLAGGED an exactly balanced" in str(e),
               "AND THE OTHER DIRECTION IS CAUGHT TOO: an oracle that "
               "flags an exactly-balanced exclusion is REFUSED -- an "
               "instrument that always fires certifies nothing either")
    finally:
        PA.compare = _real_compare
    ok(PA.compare is _real_compare,
       "and the shipped oracle is RESTORED after the stubs -- a test that "
       "leaves a monkeypatch behind poisons every check after it")

    # RED BY NAME. Until round 65 a failure returned rc=1 with the reason
    # only inside a JSON list, so DA's mutation harness recorded
    # `named_failures: []` -- the suite was red and could not say what for.
    # A red that does not name its check is a red a reader has to guess at.
    for f in fails:
        print(f"FAIL: {f}")
    print(json.dumps({"selftest": "PASS" if not fails else "FAIL",
                      "checks": checks, "n_failures": len(fails),
                      "failures": fails}, indent=1))
    return 0 if not fails else 1


# ------------------------------------------------------------------- main

def run(cache: Path, *, n_permutations: int = N_PERMUTATIONS) -> dict:
    import time
    fr = load_reference(cache)
    rec = tranche_records(fr["reference"])
    gen_n = sum(len(s[x]) for s in fr["reference"].values() for x in s)
    return {
        "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_cache": str(cache),
        "population": {
            "name": fr["population"], "n_windows": fr["n_slugs"],
            "n_generations": gen_n, "n_tranches": len(rec),
            "build_reference_statuses": fr["statuses"],
            "scope_limit": "THIS IS THE 12-WINDOW ARMS FRAGMENT, not the "
                           "full §3 population; every rate below is over "
                           "it (rule 8: n and as-of travel with it)",
        },
        "census": census(rec),
        "duration_tail": duration_tail(rec),
        "selectivity": selectivity(rec, n_permutations=n_permutations),
        "denominators": denominator_check(rec),
        "decides_nothing": "REPORTED (rule 14).",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default=str(
        Path(__file__).resolve().parents[2]
        / "data/pm_5min/derived/de_section81_cache_12.pkl"))
    ap.add_argument("--out")
    ap.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        res = run(Path(a.cache), n_permutations=a.permutations)
    except (MidCensusRefused, PA.PopulationAuditRefused) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    txt = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
