"""THE 09-07 RE-VALUATION EMIT. A POST-PROCESSOR, NOT A VALUATION.

It values nothing, so it needs no pin. It reads day one's V1 result, the
V2 result, and the two books' window decompositions, and computes the
table DA's forward-test declaration pre-commits.

IT IS COMMITTED BEFORE IT RUNS ON REAL INPUTS. A rule that exists only in
scratch when the number lands is indistinguishable from one written after
it -- which is the whole reason the declaration pre-commits the gap-seconds
column rather than letting it be chosen later.

FIELD NAMES ARE THE DECLARATION'S, NOT MINE: DELTA_D, delta_D_cents,
window_start, utc, gap_seconds, n_gap_intervals, share_of_day_gap_time,
CONCENTRATION_FINDING, SIGN_CHANGE_HALT, and the refusal
PER_WINDOW_TABLE_DOES_NOT_SUM_TO_THE_REPORTED_DELTA_D. A reader checks
IDENTITY, not vocabulary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROTOCOL = "P003_DE_REVALUATION_EMIT_V1"
CONCENTRATION_BAR_CENTS = 110.0
REFERENCE_INTERVAL_SCOPED_CENTS = 18.4
REFERENCE_WINDOW_SCOPED_CENTS = 1036.5
ROUNDING_BAND_CENTS = 1e-6
HALT_BAND_CENTS = 1.0

NO_TABLE = "REVALUATION_EMIT_HAS_NO_PER_WINDOW_TABLE"
BAD_SUM = "PER_WINDOW_TABLE_DOES_NOT_SUM_TO_THE_REPORTED_DELTA_D"
UNREADABLE = "REVALUATION_EMIT_CANNOT_READ_A_REQUIRED_INPUT"


class RevaluationEmitRefused(RuntimeError):
    """A named refusal."""


def read_json(path, what: str) -> dict:
    """An unreadable input REFUSES by name and emits no table."""
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:                        # noqa: BLE001
        raise RevaluationEmitRefused(
            f"REFUSED {UNREADABLE}: {what} at {path} -- {exc}. No table is "
            f"emitted: a table computed from inputs one of which could not "
            f"be read is not a partial result, it is a wrong one.")


def declared_windows(declaration) -> list:
    """The window spine, from the declaration that PRE-COMMITS it."""
    d = read_json(declaration, "the forward-test declaration")
    for _, v in _walk(d):
        pass
    node = _find_key(d, "THE_GAP_SECONDS_COLUMN_IS_PRE_COMMITTED_HERE_"
                        "SO_IT_CANNOT_BE_CHOSEN_LATER")
    if not node or "windows" not in node:
        raise RevaluationEmitRefused(
            f"REFUSED {NO_TABLE}: the declaration carries no pre-committed "
            f"window list, so the spine would have to be chosen now.")
    return node["windows"]


def _walk(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, p + "/" + k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, p + f"[{i}]")
    else:
        yield p, o


def _find_key(o, key):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def concentration_finding(delta_D: float, rows: list) -> dict:
    """COMPUTED (rule 10). Aggregate OR any single window; either fires."""
    if not rows:
        raise RevaluationEmitRefused(
            f"REFUSED {NO_TABLE}: aggregate-only is defeated by offsetting "
            f"moves -- the largest window holds 24.9% of the day's gap "
            f"time. An empty table cannot exonerate a day.")
    agg = abs(delta_D) > CONCENTRATION_BAR_CENTS
    worst = max(rows, key=lambda r: abs(r["delta_D_cents"]))
    per = abs(worst["delta_D_cents"]) > CONCENTRATION_BAR_CENTS
    return {"CONCENTRATION_FINDING": bool(agg or per),
            "fired_on_aggregate": bool(agg),
            "fired_on_a_single_window": bool(per),
            "bar_cents": CONCENTRATION_BAR_CENTS,
            "aggregate_abs_DELTA_D_cents": abs(delta_D),
            "worst_window_start": worst["window_start"],
            "worst_window_utc": worst.get("utc"),
            "worst_window_abs_delta_D_cents": abs(worst["delta_D_cents"]),
            "meaning_if_it_fires": ("a FINDING, not a new D -- the "
                                    "population's gap treatment is the "
                                    "story and it goes to the user")}


def residual_band(residual: float) -> dict:
    """<1e-6c rounding, 1e-6..1c report, >=1c HALT with the refusal name."""
    a = abs(residual)
    if a < ROUNDING_BAND_CENTS:
        band, halt = "ROUNDING", False
    elif a < HALT_BAND_CENTS:
        band, halt = "REPORT", False
    else:
        band, halt = "HALT", True
    return {"residual_cents": residual, "band": band,
            "HALT": halt,
            "refusal_name_if_halt": BAD_SUM,
            "bands": {"rounding_below_cents": ROUNDING_BAND_CENTS,
                      "halt_at_or_above_cents": HALT_BAND_CENTS}}


def emit(day_one_D: dict, revalued_D: dict, per_window_by_arm: dict,
         windows: list) -> dict:
    """The whole emit: (a) through (f)."""
    out = {"protocol": PROTOCOL,
           "reference_levels_cents": {
               "interval_scoped": REFERENCE_INTERVAL_SCOPED_CENTS,
               "window_scoped": REFERENCE_WINDOW_SCOPED_CENTS},
           "n_declared_windows": len(windows),
           "SIGN_CHANGE_HALT": False, "halted_arms": [], "arms": {}}
    spine = {int(w["window_start"]): w for w in windows}
    for arm, d1 in day_one_D.items():
        d2 = revalued_D[arm]
        DELTA_D = d2 - d1
        contrib = per_window_by_arm.get(arm, {})
        rows = []
        for start, w in sorted(spine.items()):
            rows.append({
                "window_start": start, "utc": w.get("utc"),
                "gap_seconds": w.get("gap_seconds"),
                "n_gap_intervals": w.get("n_gap_intervals"),
                "share_of_day_gap_time": w.get("share_of_day_gap_time"),
                "delta_D_cents": float(contrib.get(start, 0.0))})
        summed = sum(r["delta_D_cents"] for r in rows)
        res = residual_band(summed - DELTA_D)
        sign_change = (d1 > 0) != (d2 > 0)
        if sign_change:
            out["SIGN_CHANGE_HALT"] = True
            out["halted_arms"].append(arm)
        out["arms"][arm] = {
            "D_day_one_cents": d1, "D_revalued_cents": d2,
            "DELTA_D_cents": DELTA_D,
            "per_window_table": rows, "n_rows": len(rows),
            "table_sums_to_cents": summed,
            **res, "sign_change": sign_change,
            **concentration_finding(DELTA_D, rows)}
    return out


def falsify() -> int:
    cells = ok = 0

    def ck(n, c):
        nonlocal cells, ok
        cells += 1
        ok += bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")

    W = [{"window_start": 1788813900 + i * 300, "utc": f"w{i}",
          "gap_seconds": 36.187 if i == 0 else 1.0,
          "n_gap_intervals": 6 if i == 0 else 1,
          "share_of_day_gap_time": 0.249 if i == 0 else 0.01}
         for i in range(27)]
    starts = [w["window_start"] for w in W]

    one = {starts[0]: 111.0}
    r = emit({"A": -100.0}, {"A": 11.0}, {"A": one}, W)["arms"]["A"]
    ck("one window at 111c FIRES on the WINDOW arm",
       r["CONCENTRATION_FINDING"] and r["fired_on_a_single_window"])
    flat_small = {s: 4.0 for s in starts}
    r2 = emit({"A": -100.0}, {"A": 8.0}, {"A": flat_small}, W)["arms"]["A"]
    ck("27 x 4c = 108c aggregate STAYS SILENT",
       r2["CONCENTRATION_FINDING"] is False)
    flat_big = {s: 111.0 / 27 for s in starts}
    r3 = emit({"A": -100.0}, {"A": 11.0}, {"A": flat_big}, W)["arms"]["A"]
    ck("111c spread FLAT fires on the AGGREGATE arm only",
       r3["CONCENTRATION_FINDING"] and r3["fired_on_aggregate"]
       and not r3["fired_on_a_single_window"])
    try:
        read_json("/nonexistent/book.json", "a book")
        ck("an unreadable input REFUSES by name, no table", False)
    except RevaluationEmitRefused as e:
        ck("an unreadable input REFUSES by name, no table",
           UNREADABLE in str(e))
    ck("a residual >= 1c is a HALT carrying the declaration's refusal name",
       residual_band(1.5)["HALT"] and residual_band(1.5)
       ["refusal_name_if_halt"] == BAD_SUM)
    ck("a residual under 1e-6c is ROUNDING, not a finding",
       residual_band(1e-9)["band"] == "ROUNDING")
    t = emit({"A": -11017.71}, {"A": 5.0}, {"A": {starts[0]: 11022.71}}, W)
    ck("a SIGN CHANGE halts and names the arm",
       t["SIGN_CHANGE_HALT"] and t["halted_arms"] == ["A"])
    ck("the table carries the DECLARED spine, not the observed keys",
       len(t["arms"]["A"]["per_window_table"]) == 27)
    print(f"\n{ok}/{cells} cells pass")
    return 0 if ok == cells else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print("usage: de_revaluation_emit.py --falsify")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
