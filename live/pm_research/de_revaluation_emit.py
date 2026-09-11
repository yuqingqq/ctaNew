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
ROW_COUNT = "PER_WINDOW_TABLE_ROW_COUNT_DOES_NOT_MATCH_THE_DECLARED_SPINE"
BAD_SUM = "PER_WINDOW_TABLE_DOES_NOT_SUM_TO_THE_REPORTED_DELTA_D"
NO_DECOMP = "PER_WINDOW_TABLE_EMPTY_NO_DECOMPOSITION"
OUTSIDE = "PER_WINDOW_RESIDUAL_IS_A_CHANGE_OUTSIDE_THE_DECLARED_WINDOWS"
RESIDUAL_SIGN_CONVENTION = (
    "residual = SUM(declared rows) - DELTA_D = -(sum over the windows "
    "OUTSIDE the declared spine). A +50c change in an undeclared window "
    "therefore appears as residual_cents: -50. It is NOT 50c missing from "
    "the table.")
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
    return table_rows(node["windows"])


def table_rows(windows: list) -> list:
    """ONLY role == TABLE enters the spine.

    REVIEW 189: the declared list carries 27 TABLE rows AND a CENSUS_ONLY
    row (15:55, which BE 138 drove as BYTE-IDENTICAL across eras). Taking
    the list entire produces 28 rows while the declaration's own predicate
    says 27 -- the table contradicting the field beside it.

    A CENSUS_ONLY window contributes NO rows to the replay, so any
    non-zero delta attributed to it would be a fabrication, not a
    measurement. It is excluded from the table and counted separately.
    """
    return [w for w in windows if str(w.get("role", "TABLE")) == "TABLE"]


def spine_for_day(day: str, declaration=None, derived=None) -> list:
    """THE DAY'S OWN SPINE. 09-07 is 27; 09-08's fixed era is 43 (BE 133).

    A spine hardcoded to one day's count is a fixture tautology: it can
    only pass. The count comes from the day's own artifact, and the table
    is then ASSERTED against it."""
    derived = Path(derived or "/home/yuqing/ctaNew/data/pm_5min/derived")
    art = derived / f"be137_gap_windows_{day.replace('-', '')}.json"
    if art.is_file():
        d = read_json(art, f"{day}'s gap-window artifact")
        w = _find_key(d, "windows")
        if w:
            return table_rows(w)
        starts = d.get("gap_bearing_window_starts") or []
        return [{"window_start": s, "role": "TABLE"} for s in starts]
    if declaration:
        return declared_windows(declaration)
    raise RevaluationEmitRefused(
        f"REFUSED {NO_TABLE}: no gap-window artifact for {day} and no "
        f"declaration given. The spine must come from the day, never from "
        f"a constant carried over from another day.")


def assert_table_matches_spine(rows: list, spine: list) -> dict:
    """The row count is checked against THE DAY'S spine, not a literal."""
    if len(rows) != len(spine):
        raise RevaluationEmitRefused(
            f"REFUSED {ROW_COUNT}: the table has {len(rows)} rows and the "
            f"declared spine has {len(spine)}. A table shorter than its "
            f"spine has silently dropped a window; a longer one has "
            f"admitted a row the declaration excluded.")
    return {"n_rows": len(rows), "n_spine": len(spine), "matches": True}


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


def residual_band(residual: float, rows=None, delta_D=None) -> dict:
    """<1e-6c rounding, 1e-6..1c report, >=1c HALT -- WITH THE RIGHT NAME.

    REVIEW 150: two HALTs with opposite readings were raising one name.
      - every declared row 0.0 and residual == DELTA_D  -> the instrument
        is INCOMPLETE. Rules 1-3 have not run; this is not the tripwire
        firing.
      - a populated table and |residual| >= 1c -> the era fix moved a
        window OUTSIDE the declared spine. Rules 1-3 are retired until it
        is located.
    A reader must tell these apart BY NAME, because the second is a
    finding about the data and the first is a finding about us.
    """
    a = abs(residual)
    if a < ROUNDING_BAND_CENTS:
        band, halt = "ROUNDING", False
    elif a < HALT_BAND_CENTS:
        band, halt = "REPORT", False
    else:
        band, halt = "HALT", True
    empty = bool(rows) and all(r.get("delta_D_cents", 0.0) == 0.0
                               for r in rows)
    residual_is_all_of_delta = (delta_D is not None
                                and abs(residual + delta_D)
                                < ROUNDING_BAND_CENTS)
    if halt and empty and residual_is_all_of_delta:
        name, reading = NO_DECOMP, (
            "THE INSTRUMENT IS INCOMPLETE, not the day. Every declared row "
            "is 0.0 and the residual is the whole of DELTA_D, so no "
            "decomposition stands behind the table. Rules 1-3 have not run.")
    elif halt:
        name, reading = OUTSIDE, (
            "A CHANGE OUTSIDE THE DECLARED SPINE. The table is populated "
            "and cents remain unaccounted, so the era fix moved a window "
            "that carried no gap. Rules 1-3 are retired until it is located.")
    else:
        name, reading = None, "within band"
    return {"residual_cents": residual, "band": band, "HALT": halt,
            "refusal_name_if_halt": name,
            "what_a_halt_here_means": reading,
            "table_is_all_zero": empty,
            "residual_sign_convention": RESIDUAL_SIGN_CONVENTION,
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
        res = residual_band(summed - DELTA_D, rows, DELTA_D)
        # A D that lands EXACTLY on zero is neither positive nor
        # negative, and `(d1 > 0) != (d2 > 0)` lets it through. Zero is a
        # sign change from either side: the day's direction has gone.
        def _sgn(x):
            return 0 if x == 0.0 else (1 if x > 0 else -1)
        sign_change = _sgn(d1) != _sgn(d2)
        if sign_change:
            out["SIGN_CHANGE_HALT"] = True
            out["halted_arms"].append(arm)
        out["arms"][arm] = {
            "D_day_one_cents": d1, "D_revalued_cents": d2,
            "DELTA_D_cents": DELTA_D,
            "per_window_table": rows, "n_rows": len(rows),
            **assert_table_matches_spine(rows, windows),
            "table_sums_to_cents": summed,
            **res, "sign_change": sign_change,
            **concentration_finding(DELTA_D, rows)}
    return out


SUPERSEDED_GLOB = "be_daybook_{day}_btc__L250ms__*superseded*.pkl"
NOT_APPLICABLE = "NOT_APPLICABLE_SINGLE_BOOK_NO_SUPERSEDED_PAIR"


def superseded_pair_for(day: str, derived=None) -> dict:
    """Is there a PAIR to difference? A PREDICATE ON THE PAIR'S PRESENCE.

    The delta-D decomposition is a DAY-ONE instrument only by accident of
    history: 09-07 is the day that got rebuilt. What actually decides it is
    whether a SUPERSEDED book sits beside the live one. If a later day ever
    gets rebuilt, the pair exists and the decomposition runs -- so the test
    is the glob, never the date.
    """
    derived = Path(derived or "/home/yuqing/ctaNew/data/pm_5min/derived")
    compact = day.replace("-", "")
    pattern = SUPERSEDED_GLOB.format(day=compact)
    old = sorted(derived.glob(pattern))
    live = derived / f"be_daybook_{compact}_btc__L250ms__FWD1.pkl"
    have = bool(old) and live.is_file()
    return {"day": day, "glob_searched": str(derived / pattern),
            "n_superseded_found": len(old),
            "superseded": [str(x) for x in old],
            "live_book": str(live) if live.is_file() else None,
            "PAIR_EXISTS": have,
            "TRIPWIRE_STATUS": ("DELTA_D_DECOMPOSITION_RUNS" if have
                                else NOT_APPLICABLE),
            "why": ("a superseded book sits beside the live one, so the "
                    "change between them is measurable"
                    if have else
                    "one book and no superseded pair: there is no DELTA_D "
                    "to decompose. An empty table here is NOT a finding -- "
                    "it is the instrument asked a question it cannot "
                    "answer, so it reports NOT_APPLICABLE and never HALTs.")}


def unconditional_window_table(day: str, derived=None,
                               declaration=None) -> list:
    """THE DAY'S GAP WINDOWS WITH THEIR SECONDS -- no pair required.

    This part is about the day's TAPE, not about a pair of books, and it is
    what lets a reader see where the day's gaps sit even when no delta-D
    exists."""
    spine = spine_for_day(day, declaration=declaration, derived=derived)
    return [{"window_start": w.get("window_start"), "utc": w.get("utc"),
             "gap_seconds": w.get("gap_seconds"),
             "n_gap_intervals": w.get("n_gap_intervals"),
             "share_of_day_gap_time": w.get("share_of_day_gap_time")}
            for w in spine]


def running_tally(per_day_D: dict, n_declared: int, sided: int = 2) -> dict:
    """The day-cluster position so far -- COMPUTED, never typed (rule 10)."""
    import de_forward_evaluator as EV
    G = len(per_day_D)
    thr = EV.ALPHA / EV.M_FAMILY
    return {"G_so_far": G, "G_declared": n_declared,
            "days_remaining": n_declared - G,
            "negative_or_zero_days": [d for d, v in per_day_D.items()
                                      if v <= 0],
            "attainable_minimum_p_at_G_declared":
                EV.day_sign_p(n_declared, n_declared, sided),
            "tolerance_negative_days_at_G_declared":
                EV.tolerance(n_declared, thr, sided),
            "computed_not_typed": True}


def emit_single_book_day(day: str, per_arm_D: dict, all_days_D: dict,
                         n_declared: int, derived=None,
                         declaration=None) -> dict:
    """The emit for a day with ONE book: (a) through (d) of DE 274."""
    pair = superseded_pair_for(day, derived)
    if pair["PAIR_EXISTS"]:
        raise RevaluationEmitRefused(
            f"REFUSED {NO_TABLE}: {day} HAS a superseded pair, so the "
            f"delta-D decomposition applies and this single-book emit "
            f"would silently skip it.")
    return {"protocol": PROTOCOL, "day": day,
            "per_window_table_unconditional":
                unconditional_window_table(day, derived, declaration),
            "TRIPWIRE_STATUS": pair["TRIPWIRE_STATUS"],
            "tripwire_pair_search": pair,
            "per_arm_D_cents": per_arm_D,
            "running_tally": running_tally(all_days_D, n_declared),
            "reference_levels_cents": {
                "interval_scoped": REFERENCE_INTERVAL_SCOPED_CENTS,
                "window_scoped": REFERENCE_WINDOW_SCOPED_CENTS}}


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
    # --- REVIEW 150: the two HALTs must be told apart BY NAME ----------
    zero_rows = [{"window_start": s_, "delta_D_cents": 0.0} for s_ in starts]
    rz = emit({"A": -100.0}, {"A": -60.0}, {"A": {}}, W)["arms"]["A"]
    ck("an ALL-ZERO table halts as PER_WINDOW_TABLE_EMPTY_NO_DECOMPOSITION",
       rz["HALT"] and rz["refusal_name_if_halt"] == NO_DECOMP
       and rz["table_is_all_zero"])
    # a populated table plus 50c in an UNDECLARED window
    full = {s_: 10.0 for s_ in starts}
    delta_with_outside = sum(full.values()) + 50.0
    ro = emit({"A": 0.0}, {"A": delta_with_outside}, {"A": full},
              W)["arms"]["A"]
    ck("a POPULATED table with 50c outside the spine halts as "
       "PER_WINDOW_RESIDUAL_IS_A_CHANGE_OUTSIDE_THE_DECLARED_WINDOWS",
       ro["HALT"] and ro["refusal_name_if_halt"] == OUTSIDE)
    ck("  and the residual reads -50, per the stated sign convention",
       abs(ro["residual_cents"] + 50.0) < 1e-9
       and "-50" in ro["residual_sign_convention"])
    ck("  the two HALTs do NOT share a name",
       rz["refusal_name_if_halt"] != ro["refusal_name_if_halt"])
    ck("a residual under 1e-6c is ROUNDING, not a finding",
       residual_band(1e-9)["band"] == "ROUNDING")
    t = emit({"A": -11017.71}, {"A": 5.0}, {"A": {starts[0]: 11022.71}}, W)
    ck("a SIGN CHANGE halts and names the arm",
       t["SIGN_CHANGE_HALT"] and t["halted_arms"] == ["A"])
    ck("the table carries the DECLARED spine, not the observed keys",
       len(t["arms"]["A"]["per_window_table"]) == 27)

    # --- REVIEW 189: a CENSUS_ONLY row must NEVER enter the table --------
    mixed = [dict(w, role="TABLE") for w in W] + [
        {"window_start": 1788806100, "utc": "15:55:00Z", "gap_seconds": 3.0,
         "n_gap_intervals": 1, "role": "CENSUS_ONLY"}]
    ck("27 TABLE + 1 CENSUS_ONLY -> the spine is 27, not 28",
       len(table_rows(mixed)) == 27 and len(mixed) == 28)
    r4 = emit({"A": -100.0}, {"A": 11.0}, {"A": one}, table_rows(mixed))
    ck("  and the emitted table has 27 rows matching its spine",
       r4["arms"]["A"]["n_rows"] == 27 and r4["arms"]["A"]["matches"])
    # The KNOWN-BAD, demonstrated rather than asserted away: taking the
    # list ENTIRE admits the CENSUS row and yields 28 rows. That is the
    # defect REVIEW 189 found, and the filter is what removes it.
    unfiltered = emit({"A": -100.0}, {"A": 11.0}, {"A": one},
                      mixed)["arms"]["A"]["per_window_table"]
    filtered = r4["arms"]["A"]["per_window_table"]
    ck("  UNFILTERED admits the CENSUS row and yields 28",
       len(unfiltered) == 28
       and 1788806100 in [r["window_start"] for r in unfiltered])
    ck("  FILTERED excludes it and yields 27",
       len(filtered) == 27
       and 1788806100 not in [r["window_start"] for r in filtered])

    # --- the d2 == 0.0 sign edge ----------------------------------------
    z = emit({"A": -11017.71}, {"A": 0.0}, {"A": {starts[0]: 11017.71}}, W)
    ck("D_v1 < 0 and D_v2 == 0.0 -> SIGN_CHANGE_HALT names the arm",
       z["SIGN_CHANGE_HALT"] and z["halted_arms"] == ["A"])

    # --- per-day spine: 27 and 43 both pass; a short table REFUSES -------
    W43 = [{"window_start": 1788800000 + i * 300, "utc": f"x{i}",
            "gap_seconds": 1.0, "role": "TABLE"} for i in range(43)]
    r43 = emit({"A": -1.0}, {"A": -2.0}, {"A": {}}, W43)["arms"]["A"]
    ck("a 43-window spine yields 43 rows (09-08's fixed era)",
       r43["n_rows"] == 43 and r43["matches"])
    try:
        assert_table_matches_spine(W43[:42], W43)
        ck("a table one row SHORT of its spine REFUSES", False)
    except RevaluationEmitRefused as e:
        ck("a table one row SHORT of its spine REFUSES", ROW_COUNT in str(e))
    # --- DE 274: the pair PREDICATE, not the date ----------------------
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        d = Path(td)
        (d / "be_daybook_20260908_btc__L250ms__FWD1.pkl").touch()
        (d / "be137_gap_windows_20260908.json").write_text(json.dumps(
            {"gap_bearing_window_starts": [1788900000 + i * 300
                                           for i in range(43)]}))
        absent = superseded_pair_for("2026-09-08", d)
        ck("09-08 shape: NO superseded pair -> NOT_APPLICABLE, not a HALT",
           absent["TRIPWIRE_STATUS"] == NOT_APPLICABLE
           and absent["PAIR_EXISTS"] is False)
        ck("  and it records the glob it searched",
           "superseded" in absent["glob_searched"])
        e = emit_single_book_day("2026-09-08", {"A": 5.0},
                                 {"2026-09-07": -11017.71,
                                  "2026-09-08": 5.0}, 7, derived=d)
        ck("  the emit carries the UNCONDITIONAL window table anyway",
           len(e["per_window_table_unconditional"]) == 43)
        ck("  and a running tally computed at G=2 of 7, tolerance 0",
           e["running_tally"]["G_so_far"] == 2
           and e["running_tally"]["days_remaining"] == 5
           and e["running_tally"]["tolerance_negative_days_at_G_declared"]
           == 0)
        # STRUCTURAL, not a substring: the first version of this cell
        # matched the word HALT inside my own explanatory prose and failed
        # a correct emit. The property is "no HALT field is set", not "the
        # letters do not appear".
        def _halts(o):
            if isinstance(o, dict):
                if o.get("HALT") is True:
                    return True
                return any(_halts(v) for v in o.values())
            if isinstance(o, list):
                return any(_halts(v) for v in o)
            return False
        ck("  and never HALTs (no HALT field set anywhere)", not _halts(e))
        # a FAKE superseded book makes the pair exist -> decomposition runs
        (d / "be_daybook_20260908_btc__L250ms__FWD1.superseded_X.pkl").touch()
        present = superseded_pair_for("2026-09-08", d)
        ck("a superseded book APPEARS -> the decomposition runs",
           present["PAIR_EXISTS"]
           and present["TRIPWIRE_STATUS"] == "DELTA_D_DECOMPOSITION_RUNS")
        try:
            emit_single_book_day("2026-09-08", {"A": 5.0}, {"d": 1.0}, 7,
                                 derived=d)
            ck("  and the single-book emit REFUSES rather than skipping it",
               False)
        except RevaluationEmitRefused:
            ck("  and the single-book emit REFUSES rather than skipping it",
               True)

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
