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


def selectivity(records: list[dict], *, n_permutations: int = N_PERMUTATIONS,
                seed: int = SEED) -> dict:
    """DA's fourth oracle, pointed at MY exclusion.

    The instrument is `da_population_audit.compare` UNCHANGED and
    IMPORTED, never re-implemented here: a second copy of a test is a
    second test."""
    ex = [r for r in records if not r["has_mid"]]
    re_ = [r for r in records if r["has_mid"]]
    return PA.compare(ex, re_,
                      attrs=("slug", "side", "hour", "duration"),
                      n_permutations=n_permutations, seed=seed)


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

    print(json.dumps({"selftest": "PASS" if not fails else "FAIL",
                      "checks": checks, "failures": fails}, indent=1))
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
