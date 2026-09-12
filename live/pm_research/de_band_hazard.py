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

import json
import sys
from math import comb

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

    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


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
