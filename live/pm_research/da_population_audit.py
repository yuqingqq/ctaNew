#!/usr/bin/env python3
"""DA: the FOURTH ORACLE -- do the numbers DESCRIBE the population?

Three kinds of claim already have an instrument in this seat. A value's
PRODUCTION has the codomain check; ANOTHER DOCUMENT has the cite audit;
BEHAVIOUR has BE's execution check. A claim about a POPULATION had none.

THE LIVE INSTANCE IS AN HONEST STATUS. DE's "1,309 of 31,122 generations
(4.21%) excluded -- the feature pass dropped every row of these" satisfies
rule 4 completely: counted, named, reconciling. **And the claim a reader
actually takes from it is that the exclusion does not change the
population** -- which nothing checks. The token reads identically whether
the 1,309 are scattered at random or are every generation in one hour on one
side. Rule 4 says exclusions must be COUNTED; it does not say they must be
IGNORABLE, and only the second is what a reader assumes.

THE TEST IS CHEAP, and it is the reviewer's: compare the EXCLUDED set's
distribution against the RETAINED set's, attribute by attribute. Reported
with a declared null and a declared minimum sample (rule 6): a permutation
test over the labels, >=200 draws, with the observed total-variation
distance as the statistic.

REPORTS, NEVER DECIDES (rule 14). A selective exclusion may be perfectly
correct -- a feature pass that drops thin windows SHOULD drop them unevenly.
What must not happen is a population claim resting on a count that was never
compared.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

EXPECTED_CHECKS = 15
MIN_PERMUTATIONS = 200

#: Integer attributes with at most this many distinct values are compared
#: level-by-level rather than bucketed: `hour` (24), a side code, a small
#: count. Above it, an integer is treated as a measurement.
CATEGORICAL_INT_MAX = 32


class PopulationAuditRefused(RuntimeError):
    """Refused rather than reporting an exclusion is ignorable."""


def _levels(recs: list[dict], attr: str, buckets: int = 8) -> tuple:
    """Attribute values as categorical levels; numerics are BUCKETED.

    Bucketing is declared in the output, because a continuous attribute
    compared level-by-level would read as maximally different for reasons of
    resolution rather than of selection.
    """
    vals = [r.get(attr) for r in recs]
    missing = sum(1 for v in vals if v is None)
    nums = [v for v in vals if isinstance(v, (int, float))
            and not isinstance(v, bool)]
    # AN INTEGER WITH FEW LEVELS IS CATEGORICAL, NOT CONTINUOUS. `hour` has
    # 24 values and was being bucketed into eight ranges, which flagged the
    # right attribute while naming the wrong LEVEL -- the exclusion came
    # from hour 3 and the report said "[3,5.875)". A reader sent to the
    # wrong level is a reader sent to read the wrong thing.
    all_int = bool(nums) and all(isinstance(v, int) for v in nums)
    if (nums and len(nums) == len(vals) - missing
            and len(set(nums)) > buckets
            and not (all_int and len(set(nums)) <= CATEGORICAL_INT_MAX)):
        lo, hi = min(nums), max(nums)
        w = (hi - lo) / buckets if hi > lo else 1.0
        out = [None if v is None
               else f"[{lo + w * min(int((v - lo) / w), buckets - 1):.4g},"
                    f"{lo + w * (min(int((v - lo) / w), buckets - 1) + 1):.4g})"
               for v in vals]
        return out, missing, {"bucketed": True, "n_buckets": buckets,
                              "lo": lo, "hi": hi}
    return ([None if v is None else str(v) for v in vals], missing,
            {"bucketed": False})


def _tvd(a: collections.Counter, b: collections.Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return float("nan")
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)


def compare(excluded: list[dict], retained: list[dict],
            attrs: tuple = ("slug", "side", "hour", "duration"),
            n_permutations: int = MIN_PERMUTATIONS,
            seed: int = 0) -> dict[str, Any]:
    """Is the EXCLUDED set distributed like the RETAINED set, attribute by
    attribute? Declared null, declared minimum sample, reported statistic."""
    if n_permutations < MIN_PERMUTATIONS:
        raise PopulationAuditRefused(
            f"REFUSED: {n_permutations} permutations is below the declared "
            f"minimum of {MIN_PERMUTATIONS} (rule 6). An under-sampled "
            f"correct null flatters as much as a wrong one.")
    if not isinstance(excluded, list) or not isinstance(retained, list):
        raise PopulationAuditRefused(
            "REFUSED: both sets must be lists of records.")
    if not retained:
        raise PopulationAuditRefused(
            "REFUSED: the RETAINED set is empty, so there is no population "
            "to compare the exclusion against. An exclusion that removed "
            "everything is not an ignorable one.")
    if not excluded:
        return {"status": "NOTHING_EXCLUDED", "n_excluded": 0,
                "n_retained": len(retained), "attributes": {},
                "selective_attributes": [],
                "why": ("nothing was excluded, so no exclusion can be "
                        "selective. This is a STATUS, not a pass -- it says "
                        "the question did not arise"),
                "decides_nothing": "REPORTED (rule 14)."}

    rng = random.Random(seed)
    allrecs = excluded + retained
    n_e = len(excluded)
    out: dict[str, Any] = {}
    for attr in attrs:
        lv, missing, meta = _levels(allrecs, attr)
        if missing == len(lv):
            out[attr] = {"status": "ATTRIBUTE_ABSENT",
                         "why": "no record carries this attribute; absence "
                                "is reported, never treated as agreement"}
            continue
        if missing:
            out[attr] = {"status": "ATTRIBUTE_PARTIALLY_ABSENT",
                         "n_missing": missing,
                         "why": "a distribution over records that carry the "
                                "attribute would compare two different "
                                "populations"}
            continue
        e_lv, r_lv = lv[:n_e], lv[n_e:]
        ce, cr = collections.Counter(e_lv), collections.Counter(r_lv)
        obs = _tvd(ce, cr)
        ge = 0
        for _ in range(n_permutations):
            shuf = lv[:]
            rng.shuffle(shuf)
            if _tvd(collections.Counter(shuf[:n_e]),
                    collections.Counter(shuf[n_e:])) >= obs:
                ge += 1
        p = (1 + ge) / (n_permutations + 1)
        ne, nr = sum(ce.values()), sum(cr.values())
        top = sorted(
            ((k, ce[k], cr[k], ce[k] / ne - cr[k] / nr)
             for k in set(ce) | set(cr)),
            key=lambda t: -abs(t[3]))[:5]
        out[attr] = {
            "status": "COMPARED", "n_levels": len(set(lv)),
            "tvd": round(obs, 6), "p_permutation": round(p, 6),
            "n_permutations": n_permutations,
            "bucketing": meta,
            "largest_share_differences": [
                {"level": k, "n_excluded": a, "n_retained": b,
                 "share_difference": round(d, 6)} for k, a, b, d in top],
            "verdict": ("DISTRIBUTION_DIFFERS" if p <= 0.05
                        else "INDISTINGUISHABLE_AT_THIS_N"),
        }
    sel = sorted(k for k, v in out.items()
                 if v.get("verdict") == "DISTRIBUTION_DIFFERS")
    return {
        "status": "COMPARED",
        "n_excluded": n_e, "n_retained": len(retained),
        "excluded_fraction": round(n_e / (n_e + len(retained)), 6),
        "attributes": out,
        "selective_attributes": sel,
        "null": (f"labels permuted between the two sets, {n_permutations} "
                 f"draws (declared minimum {MIN_PERMUTATIONS}); statistic is "
                 f"the total-variation distance between the level "
                 f"distributions"),
        "reading": ("THE EXCLUSION IS SELECTIVE ON " + ", ".join(sel)
                    if sel else
                    "no attribute distinguishes the excluded set at this N"),
        "limits": ("INDISTINGUISHABLE AT THIS N is not IGNORABLE: a small "
                   "excluded set has little power, and an attribute nobody "
                   "listed is not tested. This compares the attributes it "
                   "was GIVEN"),
        "decides_nothing": ("REPORTED. Whether a selective exclusion voids a "
                            "population claim is the policy layer's "
                            "(rule 14)."),
    }


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            print(f"FAIL: {label}")
            raise SystemExit(1)
        print(f"PASS: {label}")

    rng = random.Random(7)
    pop = [{"slug": f"btc-{i % 40}", "side": rng.choice(["BUY_UP",
                                                         "SELL_UP"]),
            "hour": rng.randrange(24), "duration": rng.uniform(1, 300)}
           for i in range(3000)]

    # ---- direction 1: an exclusion drawn at random must NOT flag ---------
    idx = list(range(len(pop)))
    rng.shuffle(idx)
    cut = int(0.0421 * len(pop))          # DE53's rate, on purpose
    exc = [pop[i] for i in idx[:cut]]
    ret = [pop[i] for i in idx[cut:]]
    r1 = compare(exc, ret, seed=1)
    ok(r1["status"] == "COMPARED" and not r1["selective_attributes"],
       f"POP-1 ADMITS: an exclusion drawn AT RANDOM at DE53's own rate "
       f"({r1['excluded_fraction']:.4f}) flags nothing -- p per attribute "
       f"{ {k: v.get('p_permutation') for k, v in r1['attributes'].items()} }. "
       f"An instrument that flagged this would flag every honest exclusion")

    # ---- direction 2: an exclusion concentrated in ONE hour MUST flag ----
    h = [r for r in pop if r["hour"] == 3]
    exc2 = h[:cut] if len(h) >= cut else h
    ret2 = [r for r in pop if r not in exc2]
    r2 = compare(exc2, ret2, seed=1)
    ok("hour" in r2["selective_attributes"]
       and r2["attributes"]["hour"]["p_permutation"] <= 0.05,
       f"POP-2 FIRES: the SAME COUNT taken entirely from one hour is "
       f"selective on `hour` (tvd {r2['attributes']['hour']['tvd']}, "
       f"p {r2['attributes']['hour']['p_permutation']}) -- and the count, "
       f"the fraction and the rule-4 status are IDENTICAL to POP-1. That is "
       f"the whole point: the token reads the same either way")
    ok(r2["attributes"]["hour"]["largest_share_differences"][0]["level"]
       == "3",
       "POP-2b and it names WHERE: the level carrying the largest share "
       "difference is the hour the exclusion was taken from, with both "
       "counts beside it")

    # ---- direction 2b: selective on SIDE only, hour untouched -----------
    b = [r for r in pop if r["side"] == "BUY_UP"][:cut]
    r3 = compare(b, [r for r in pop if r not in b], seed=1)
    ok("side" in r3["selective_attributes"]
       and "hour" not in r3["selective_attributes"],
       "POP-3 the flag is ATTRIBUTE-LOCAL: an exclusion skewed on side "
       "alone flags side and not hour, so a reader is pointed at the axis "
       "that actually moved")

    # ---- statuses, never silent ----------------------------------------
    ok(compare([], pop, seed=1)["status"] == "NOTHING_EXCLUDED",
       "POP-4 an EMPTY exclusion is a STATUS, not a pass -- it says the "
       "question did not arise, which is different from an answer")
    try:
        compare(pop[:10], [], seed=1)
        ok(False, "POP-5 an empty RETAINED set must refuse")
    except PopulationAuditRefused as e:
        ok("removed everything" in str(e),
           "POP-5 an exclusion that removed EVERYTHING refuses: there is no "
           "population left to compare against")
    try:
        compare(pop[:10], pop[10:], n_permutations=50)
        ok(False, "POP-6 an under-sampled null must refuse")
    except PopulationAuditRefused as e:
        ok("below the declared minimum" in str(e),
           "POP-6 fewer than 200 permutations REFUSES (rule 6): an "
           "under-sampled correct null flatters as much as a wrong one")
    r4 = compare([{"side": "BUY_UP"}] * 20, [{"side": "SELL_UP"}] * 200,
                 attrs=("side", "hour"), seed=1)
    ok(r4["attributes"]["hour"]["status"] == "ATTRIBUTE_ABSENT"
       and r4["attributes"]["side"]["status"] == "COMPARED",
       "POP-7 an attribute NO record carries reports ATTRIBUTE_ABSENT -- "
       "absence is never reported as agreement, which would be the "
       "empty-set trap on the oracle itself")
    r5 = compare([{"side": "BUY_UP", "hour": 1}] * 20,
                 [{"side": "SELL_UP"}] * 200, attrs=("hour",), seed=1)
    ok(r5["attributes"]["hour"]["status"] == "ATTRIBUTE_PARTIALLY_ABSENT",
       "POP-8 an attribute only SOME records carry is refused per-attribute "
       "-- comparing the carriers would compare two different populations")

    # ---- bucketing is declared, and numerics are not maximally distinct --
    r6 = compare(exc, ret, attrs=("duration",), seed=1)
    ok(_levels(pop, "hour")[2]["bucketed"] is False
       and _levels(pop, "duration")[2]["bucketed"] is True,
       "POP-9a an INTEGER with few levels (hour, 24) is categorical while a "
       "float (duration) is bucketed -- bucketing hour flagged the right "
       "attribute and named the wrong LEVEL, `[3,5.875)` for an exclusion "
       "taken from hour 3")
    ok(r6["attributes"]["duration"]["bucketing"]["bucketed"] is True
       and r6["attributes"]["duration"]["verdict"]
       == "INDISTINGUISHABLE_AT_THIS_N",
       "POP-9 a CONTINUOUS attribute is bucketed and the bucketing is "
       "declared -- compared level-by-level, 3,000 distinct durations would "
       "read as maximally different for reasons of resolution")
    dur_hi = sorted(pop, key=lambda r: -r["duration"])[:cut]
    r7 = compare(dur_hi, [r for r in pop if r not in dur_hi],
                 attrs=("duration",), seed=1)
    ok("duration" in r7["selective_attributes"],
       "POP-10 and it still FIRES on a duration-selective exclusion: "
       "bucketing did not buy the admission by blunting the instrument")

    # ---- determinism and the declared null ------------------------------
    ok(compare(exc, ret, seed=1) == compare(exc, ret, seed=1),
       "POP-11 the same seed gives the same answer -- a permutation p that "
       "moves between runs cannot be quoted")
    ok(str(MIN_PERMUTATIONS) in r1["null"] and "total-variation" in r1["null"]
       and "INDISTINGUISHABLE AT THIS N is not IGNORABLE" in r1["limits"],
       "POP-12 the null is DECLARED IN THE OUTPUT with its design and its "
       "minimum sample, and the limits say that indistinguishable is not "
       "ignorable -- the reading a count invites is the one this exists to "
       "stop")
    ok(r1["decides_nothing"].startswith("REPORTED"),
       "POP-13 decides nothing: a selective exclusion may be entirely "
       "correct, and whether it voids a population claim is the policy "
       "layer's")
    print(f"\nda_population_audit selftest: {checks} checks PASSED")
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: EXPECTED_CHECKS={EXPECTED_CHECKS} but {checks} ran.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--excluded"); ap.add_argument("--retained")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.excluded and a.retained):
        raise SystemExit("REFUSED: --excluded and --retained JSON files")
    try:
        print(json.dumps(compare(json.loads(Path(a.excluded).read_text()),
                                 json.loads(Path(a.retained).read_text())),
                         indent=1, sort_keys=True))
    except PopulationAuditRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
