#!/usr/bin/env python3
"""BE 132 -- THE OCCUPANCY OF THE THETA NEIGHBOURHOOD, PER DAY.

THE QUESTION THIS ANSWERS, and why it is not the certification's question.
The score-neutrality certification is measured on 09-03. The forward test runs
on 09-07..09-13. REV's ruling is that a consumed-day comparison cannot license
a forward day, because THE DECIDING QUANTITY IS THAT DAY'S OWN OCCUPANCY: a
certification saying "no generation flipped, and the nearest sat m_min above
theta" is worth exactly as much as the forward day's own margin distribution
makes it worth. A day whose generations crowd theta puts the question to the
scoring code many times; a day whose generations sit far from it never puts
the question at all, and "nothing flipped" there is WEAK_DAY_NEVER_PUT_THE
QUESTION however many generations it holds.

SO THIS MEASURES, PER BOOK AND PER ARM, ON THE ARM'S OWN FROZEN THETA:
  * n_scored -- the population, because a count without its n is not a number
  * m_min -- the closest any generation comes to theta
  * an ABSOLUTE margin ladder, in score units, common to every day, so two
    days are comparable without either one's delta_max in the denominator
  * the same counts as a RATE PER 1,000 SCORED GENERATIONS, because the days
    differ in size and a raw count would read as density
  * if -- and only if -- a certification artifact is supplied, the ladder
    SCALED by that certification's measured delta_max, which is the ladder
    that answers "does 09-03's result transfer to this day"

REFUSALS, in the direction that errs loud (rule 41):
  * a scaled ladder REQUESTED with no delta_max available REFUSES. Licensing a
    day against a bound nobody measured is the failure `per_book_guard` was
    built to prevent and this instrument will not do it either.
  * a delta_max of zero (bit-identical books) makes the SCALED ladder
    degenerate -- every margin is "outside" it -- so it is reported as
    NOT_APPLICABLE_DELTA_MAX_IS_ZERO rather than as an empty neighbourhood.
  * a book that cannot be read, or an arm whose head is absent, refuses by the
    comparator's own names; this module never invents a reading.

WHAT IT DOES NOT DO. It computes no value, no P&L and no outcome. It reads
scores and a frozen threshold. It decides nothing (rule 14): it reports a
distribution and the policy layer -- or a human -- reads it.

Exit codes (75 is the wrapper's and is not among them, rule 20):
  0  measured
  2  usage
  3  an input refused
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from live.pm_research.be_score_neutrality import (  # noqa: E402
    NeutralityRefused, arm_heads, gen_census, gen_max, identity_of, load,
    n_generations_in_book,
)

EXIT_CODES = {0: "MEASURED", 2: "USAGE", 3: "INPUT_REFUSED"}
assert 75 not in EXIT_CODES, "75 is the wrapper's conflict code (rule 20)"

# Absolute, in score units. Fixed here so no day can choose its own ladder
# after seeing its own margins (rule 11: choosing after seeing voids the test).
ABS_LADDER = (1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SCALED_K = (1, 10, 100, 1000)
NO_DELTA = "SCALED_LADDER_REQUESTED_WITH_NO_DELTA_MAX_CERTIFIED"


def delta_max_from_certification(path: Path) -> dict:
    """{arm: delta_max} from a certification artifact, by arm, never typed."""
    doc = json.loads(Path(path).read_text())
    per_arm = doc.get("per_arm") or {}
    if not per_arm:
        raise NeutralityRefused(
            f"REFUSED {NO_DELTA}: {path} carries no per_arm block")
    out = {}
    for arm, row in per_arm.items():
        d = row.get("B_delta_max_abs")
        if d is None:
            raise NeutralityRefused(
                f"REFUSED {NO_DELTA}: arm {arm} in {path} has no "
                f"B_delta_max_abs")
        out[arm] = float(d)
    return out


def occupancy_of(book: dict, *, delta_max: dict | None = None) -> dict:
    """Per arm: the population, m_min, and the margin ladder."""
    arms = arm_heads()
    ref_gens = n_generations_in_book(book)
    out: dict = {"identity": identity_of(book),
                 "n_reference_generations": ref_gens, "per_arm": {}}
    for arm, spec in sorted(arms.items()):
        head, theta = spec["head"], spec["theta"]
        scores = gen_max(book, head)
        census = gen_census(book, head)
        if len(scores) != census["n_generations_with_a_score"]:
            raise NeutralityRefused(
                f"REFUSED: arm {arm} enumerated {len(scores)} generation(s) "
                f"against {census['n_generations_with_a_score']} carrying a "
                f"score -- this instrument dropped "
                f"{census['n_generations_with_a_score'] - len(scores)}.")
        margins = sorted(abs(v - theta) for v in scores.values())
        n = len(margins)
        if not n:
            raise NeutralityRefused(
                f"REFUSED: arm {arm} has no scored generation to measure")

        def below(edge):
            lo, hi = 0, n
            while lo < hi:                    # margins is sorted
                mid = (lo + hi) // 2
                if margins[mid] < edge:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        absolute = {f"within_{e:g}": below(e) for e in ABS_LADDER}
        rate = {k: round(1000.0 * v / n, 6) for k, v in absolute.items()}
        row = {
            "head": head, "theta": theta,
            "params_file": spec["params_file"],
            "n_scored_generations": n,
            "n_reference_generations": ref_gens,
            "fraction_of_the_reference_scored": n / ref_gens if ref_gens else None,
            "m_min": margins[0],
            "m_p001": margins[max(0, n // 1000)],
            "m_p01": margins[max(0, n // 100)],
            "m_p10": margins[max(0, n // 10)],
            "m_median": margins[n // 2],
            "n_exactly_at_theta": sum(1 for v in scores.values() if v == theta),
            "n_at_or_above_theta": sum(1 for v in scores.values() if v >= theta),
            "absolute_ladder_counts": absolute,
            "absolute_ladder_per_1000_generations": rate,
            "LADDER_IS_FIXED": (
                "declared in the module, identical for every day, so no day "
                "can pick a ladder after seeing its own margins (rule 11)"),
        }
        if delta_max is not None:
            d = delta_max.get(arm)
            if d is None:
                raise NeutralityRefused(
                    f"REFUSED {NO_DELTA}: arm {arm} has no certified "
                    f"delta_max, and a day is never licensed against a bound "
                    f"nobody measured.")
            if d == 0.0:
                row["scaled_ladder"] = "NOT_APPLICABLE_DELTA_MAX_IS_ZERO"
            else:
                sc = {f"within_{k}x_delta_max": below(k * d) for k in SCALED_K}
                row["scaled_ladder_counts"] = sc
                row["scaled_ladder_per_1000_generations"] = {
                    k: round(1000.0 * v / n, 6) for k, v in sc.items()}
                row["delta_max_certified"] = d
                row["m_min_over_delta_max"] = margins[0] / d
        out["per_arm"][arm] = row
    return out


def falsify() -> int:
    """Rule 15: a positive control it must flag and a known-bad it refuses."""
    checks = []

    def note(n, ok):
        checks.append((n, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")

    AH = arm_heads()
    arm = sorted(AH)[0]
    head, theta = AH[arm]["head"], AH[arm]["theta"]
    other = sorted(AH)[1]
    ohead, otheta = AH[other]["head"], AH[other]["theta"]

    def book(gens, ogens, n_ref=None):
        """The reference carries EXACT generation identities -- the shape
        `reference_generation_keys` requires. A fixture built the old way
        (a list of None) refuses, which is the checker working."""
        def entries(g):
            return {(s, sd, float(i) / 100): {"score": v, "gen": gg, "t0": i}
                    for i, ((s, sd, gg), v) in enumerate(g.items())}
        ref = {}
        for slug, side, generation in sorted({(s, sd, gg) for s, sd, gg in gens}):
            ref.setdefault(slug, {}).setdefault(side, []).append(
                {"gen": generation})
        # extra UNSCORED reference generations, to make the scored set partial
        extra = 0 if n_ref is None else n_ref - len(gens)
        for i in range(extra):
            ref["s1"]["BUY_UP"].append({"gen": 1000 + i})
        return {"header": {"day": "20260907", "coin": "btc",
                           "placement_latency": {"placement_latency_ms": 250.0}},
                "fr": {"reference": ref},
                "asm": {"by_arm": {("btc", head): (entries(gens),),
                                   ("btc", ohead): (entries(ogens),)}}}

    # POSITIVE CONTROL: a generation placed 1e-9 from theta must be COUNTED at
    # every rung at or above 1e-9 and NOT counted below it. A ladder that
    # counted everything, or nothing, would pass a vacuous check.
    g = {("s1", "BUY_UP", 1): theta + 1e-9,
         ("s1", "BUY_UP", 2): theta + 0.5,
         ("s1", "BUY_UP", 3): theta - 2.0}
    og = {("s1", "BUY_UP", 1): otheta + 1.0,
          ("s1", "BUY_UP", 2): otheta + 0.5,
          ("s1", "BUY_UP", 3): otheta - 2.0}
    r = occupancy_of(book(g, og))["per_arm"][arm]
    note("a generation 1e-9 from theta is INSIDE the 1e-6 rung",
         r["absolute_ladder_counts"]["within_1e-06"] == 1)
    note("and OUTSIDE the 1e-12 rung -- the ladder is not counting everything",
         r["absolute_ladder_counts"]["within_1e-12"] == 0)
    note("the 1.0 rung holds the 1e-9 and the 0.5, not the 2.0",
         r["absolute_ladder_counts"]["within_1"] == 2)
    # RELATIVE, not absolute. theta is ~0.32, so (theta + 1e-9) - theta is
    # 1.0000000827e-09 and not 1e-9 -- the last-ulp error at that magnitude is
    # ~5.6e-17, which an absolute 1e-18 bar cannot pass. The cell was wrong,
    # not the code; a tolerance has to be stated in the units of the quantity.
    note("m_min is the 1e-9, not the median",
         abs(r["m_min"] / 1e-9 - 1.0) < 1e-6)
    note("the rate is per 1,000 scored generations, not a count",
         abs(r["absolute_ladder_per_1000_generations"]["within_1e-06"]
             - 1000.0 / 3) < 1e-3)
    note("the population and the reference are BOTH reported",
         r["n_scored_generations"] == 3 and r["n_reference_generations"] == 3)

    # A PARTIALLY SCORED reference is the real shape and must be MEASURED,
    # with the scored fraction reported -- not refused (BE 130's class).
    r2 = occupancy_of(book(g, og, n_ref=99))["per_arm"][arm]
    note("a reference wider than the scored set is MEASURED, not refused",
         r2["n_scored_generations"] == 3
         and r2["n_reference_generations"] == 99
         and abs(r2["fraction_of_the_reference_scored"] - 3 / 99) < 1e-12)

    # KNOWN-BAD: a scaled ladder asked for with no delta_max for the arm.
    try:
        occupancy_of(book(g, og), delta_max={other: 1e-9})
        note("a scaled ladder with no delta_max for an arm REFUSES", False)
    except NeutralityRefused as e:
        note("a scaled ladder with no delta_max for an arm REFUSES",
             NO_DELTA in str(e))

    # A delta_max of ZERO is degenerate, and says so rather than reading as
    # an empty neighbourhood.
    r3 = occupancy_of(book(g, og), delta_max={arm: 0.0, other: 0.0})
    note("delta_max of zero is NOT_APPLICABLE, never an empty neighbourhood",
         r3["per_arm"][arm]["scaled_ladder"]
         == "NOT_APPLICABLE_DELTA_MAX_IS_ZERO")

    # The scaled ladder counts what it should when a delta_max exists.
    r4 = occupancy_of(book(g, og), delta_max={arm: 1e-9, other: 1e-9})
    note("the 10x rung holds the generation at 1e-9? no -- strictly inside",
         r4["per_arm"][arm]["scaled_ladder_counts"]["within_1x_delta_max"] == 0
         and r4["per_arm"][arm]["scaled_ladder_counts"]["within_10x_delta_max"]
         == 1)
    note("m_min / delta_max is reported",
         abs(r4["per_arm"][arm]["m_min_over_delta_max"] - 1.0) < 1e-6)

    # A certification with no delta_max field REFUSES rather than defaulting.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cert.json"
        p.write_text(json.dumps({"per_arm": {arm: {}, other: {}}}))
        try:
            delta_max_from_certification(p)
            note("a certification without B_delta_max_abs REFUSES", False)
        except NeutralityRefused as e:
            note("a certification without B_delta_max_abs REFUSES",
                 NO_DELTA in str(e))

    bad = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_theta_occupancy", "n": len(checks),
                      "n_failed": len(bad), "failed": bad}))
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return falsify()
    cert = None
    if "--certification" in argv:
        i = argv.index("--certification")
        cert = argv[i + 1]
        del argv[i:i + 2]
    out_path = None
    if "--out" in argv:
        i = argv.index("--out")
        out_path = argv[i + 1]
        del argv[i:i + 2]
    if not argv:
        print("usage: be_theta_occupancy.py <book.pkl> [book.pkl ...] "
              "[--certification <cert.json>] [--out <file.json>] | --selftest")
        return 2
    try:
        dm = delta_max_from_certification(Path(cert)) if cert else None
        days = {}
        for b in argv:
            book = load(b)
            days[Path(b).name] = occupancy_of(book, delta_max=dm)
            del book
    except NeutralityRefused as e:
        print(json.dumps({"refused": str(e)}, indent=1))
        return 3
    result = {
        "protocol": "BE_THETA_OCCUPANCY_V1",
        "what_this_measures": (
            "how often each day PUTS THE QUESTION to the scoring code: the "
            "distribution of |generation_max_score - theta| at each arm's "
            "frozen theta"),
        "WHAT_THIS_DOES_NOT_LICENSE": (
            "This is not a certification and it transfers none. A day's "
            "occupancy says how much a certification measured elsewhere COULD "
            "be worth on this day; it is never itself evidence that the "
            "scoring code is decision-equivalent here (REV's ruling: the "
            "deciding quantity is the day's own occupancy)."),
        "certification": cert,
        "delta_max_certified": dm,
        "absolute_ladder": list(ABS_LADDER),
        "days": days,
    }
    text = json.dumps(result, indent=1, default=str)
    if out_path:
        Path(out_path).write_text(text)
        print(f"wrote {out_path}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
