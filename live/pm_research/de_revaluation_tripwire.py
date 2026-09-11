"""THE RE-VALUATION'S TRIPWIRE: per window, not just in aggregate.

An aggregate-only comparison is defeated by OFFSETTING MOVES -- two
windows shifting +200c and -200c read as zero. The gap time is also not
uniform: the 20:45 window (36.2 s, 6 gaps) holds about a quarter of it,
so the per-window view is where cents-per-gap-second is legible at all.

WHY THE THRESHOLD IS 110c AND NOT 2,754c: even the whole 20:45 window at
Q-DA-58 concentration reaches only ~316c. A bar set at the aggregate
scale would never fire on the thing it is watching for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROTOCOL = "P003_DE_REVALUATION_TRIPWIRE_V1"
CONCENTRATION_BAR_CENTS = 110.0
REF_INTERVAL_SCOPED_CENTS = 18.4
REF_WINDOW_SCOPED_CENTS = 1036.5
NO_TABLE = "REVALUATION_TRIPWIRE_HAS_NO_PER_WINDOW_TABLE"


class TripwireRefused(RuntimeError):
    """A named refusal."""


def concentration_finding(delta_D: float, per_window: dict) -> dict:
    """COMPUTED, never printed (rule 10). Two arms, either fires."""
    if not per_window:
        raise TripwireRefused(
            f"REFUSED {NO_TABLE}: an aggregate-only comparison is defeated "
            f"by offsetting moves. An empty table cannot exonerate a day.")
    agg = abs(delta_D) > CONCENTRATION_BAR_CENTS
    worst = max(per_window.items(), key=lambda kv: abs(kv[1]["delta_cents"]))
    per = abs(worst[1]["delta_cents"]) > CONCENTRATION_BAR_CENTS
    return {"CONCENTRATION_FINDING": bool(agg or per),
            "fired_on_aggregate": bool(agg),
            "fired_on_a_single_window": bool(per),
            "bar_cents": CONCENTRATION_BAR_CENTS,
            "aggregate_abs_delta_cents": abs(delta_D),
            "worst_window": worst[0],
            "worst_window_abs_delta_cents": abs(worst[1]["delta_cents"]),
            "why_this_bar": ("even the whole 20:45 window at Q-DA-58 "
                             "concentration reaches ~316c; a bar at the "
                             "aggregate scale would never fire")}


def tripwire(day_one: dict, revalued: dict, per_window_by_arm: dict) -> dict:
    """The whole emit: delta, table, references, predicates, residual."""
    out = {"protocol": PROTOCOL, "arms": {},
           "reference_levels_cents": {
               "interval_scoped": REF_INTERVAL_SCOPED_CENTS,
               "window_scoped": REF_WINDOW_SCOPED_CENTS},
           "SIGN_CHANGE_HALT": False, "halted_arms": []}
    for arm, d1 in day_one.items():
        d2 = revalued[arm]
        delta = d2 - d1
        table = per_window_by_arm.get(arm, {})
        summed = sum(w["delta_cents"] for w in table.values())
        residual = delta - summed
        sign_change = (d1 > 0) != (d2 > 0)
        if sign_change:
            out["SIGN_CHANGE_HALT"] = True
            out["halted_arms"].append(arm)
        out["arms"][arm] = {
            "D_day_one_cents": d1, "D_revalued_cents": d2,
            "delta_D_cents": delta,
            "per_window": table,
            "n_windows": len(table),
            "table_sums_to_cents": summed,
            "residual_cents": residual,
            "RESIDUAL_IS_A_FINDING_NOT_ROUNDING": residual != 0.0,
            "sign_change": sign_change,
            **concentration_finding(delta, table)}
    return out


def falsify() -> int:
    cells = ok = 0

    def ck(n, c):
        nonlocal cells, ok
        cells += 1; ok += bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")

    one_big = {f"w{i}": {"delta_cents": (111.0 if i == 0 else 0.0),
                         "gap_seconds": 36.2 if i == 0 else 1.0}
               for i in range(27)}
    r = concentration_finding(111.0, one_big)
    ck("ONE 111c window FIRES", r["CONCENTRATION_FINDING"]
       and r["fired_on_a_single_window"])
    flat_small = {f"w{i}": {"delta_cents": 4.0, "gap_seconds": 1.0}
                  for i in range(27)}
    r2 = concentration_finding(108.0, flat_small)
    ck("27 x 4c = 108c aggregate STAYS SILENT",
       r2["CONCENTRATION_FINDING"] is False)
    flat_big = {f"w{i}": {"delta_cents": 111.0 / 27, "gap_seconds": 1.0}
                for i in range(27)}
    r3 = concentration_finding(111.0, flat_big)
    ck("111c spread FLAT fires on the AGGREGATE arm",
       r3["CONCENTRATION_FINDING"] and r3["fired_on_aggregate"]
       and not r3["fired_on_a_single_window"])
    try:
        concentration_finding(500.0, {})
        ck("an EMPTY table REFUSES rather than exonerating", False)
    except TripwireRefused as e:
        ck("an EMPTY table REFUSES rather than exonerating",
           NO_TABLE in str(e))
    t = tripwire({"A": -11017.71}, {"A": +50.0},
                 {"A": {"w": {"delta_cents": 11067.71, "gap_seconds": 2.0}}})
    ck("a SIGN CHANGE halts", t["SIGN_CHANGE_HALT"]
       and t["halted_arms"] == ["A"])
    t2 = tripwire({"A": -100.0}, {"A": -50.0},
                  {"A": {"w": {"delta_cents": 40.0, "gap_seconds": 1.0}}})
    ck("a table that does NOT sum to delta emits the residual as a finding",
       t2["arms"]["A"]["residual_cents"] == 10.0
       and t2["arms"]["A"]["RESIDUAL_IS_A_FINDING_NOT_ROUNDING"])
    print(f"\n{ok}/{cells} cells pass")
    return 0 if ok == cells else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print("usage: de_revaluation_tripwire.py --falsify")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
