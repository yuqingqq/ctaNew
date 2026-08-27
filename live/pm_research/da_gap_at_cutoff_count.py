"""Count GAP_AT_CUTOFF rows under the R-191 definition. Independent of BE.

SURFACE AUTHORISATION (R-126, in-file): R-191(3) orders BOTH seats to
implement the ruled definition INDEPENDENTLY and exchange counts; agreement is
required before the rebuild.

THE DEFINITION, transcribed from R-191(2) and implemented from that text alone
(this file does not consult BE's implementation):

    A row is GAP_AT_CUTOFF iff its ABSOLUTE decision instant
        T = t0 + t_start
    lies in  [g_start, g_end)  -- LOWER-INCLUSIVE, UPPER-EXCLUSIVE --
    for ANY recorded gap of THAT COIN in the collector-gaps ledger.

    Basis: ABSOLUTE on both sides.
    Gap scope: COIN-LEVEL, never per-slug. A gap is a FEED event; scoping it to
      the slug it was logged against is a lossy projection that drops warm-up
      and boundary overlaps -- rows with NEGATIVE t_start sit in windows whose
      per-slug gap list cannot see a gap logged against the neighbouring
      window. That projection is exactly what DA's earlier cross did, which is
      why its 152 is expected to move.
    Universe: ALL tape rows.  Unit: ROWS.  Score-side reported beside.

ONE AMBIGUITY THE RULING DOES NOT SETTLE, so it is reported rather than
decided: the ledger carries `collector_version` eras. "ANY recorded gap" is
implemented LITERALLY (no era filter) as the primary number, with the
era-filtered count published beside it so a seat comparing figures can see
whether an era assumption explains a difference.

    python3 live/pm_research/da_gap_at_cutoff_count.py --selftest
    python3 live/pm_research/da_gap_at_cutoff_count.py count --tape PATH
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import da_state_tape_verify as G

REPO = Path(__file__).resolve().parents[2]
PM_GAPS = REPO / "data/pm_5min/collector_gaps.jsonl"
ERA = "clob_v3_1"


def coin_gaps(path: Path | None = None, era: str | None = None
              ) -> tuple[dict[str, list[tuple[float, float]]], dict[str, int]]:
    """COIN -> sorted absolute [g_start, g_end) intervals, from the ledger.

    The ledger is the physical source of truth (R-191(2)). Malformed lines are
    COUNTED, never silently dropped.
    """
    # RESOLVED AT CALL TIME, deliberately. `path: Path = PM_GAPS` binds the
    # default when the function is DEFINED, so reassigning the module-level
    # PM_GAPS afterwards does nothing -- a test that believed it had injected a
    # fixture silently measured the PRODUCTION ledger instead. Caught on my own
    # new cross-tab test, whose synthetic gap never reached the counter.
    path = PM_GAPS if path is None else path
    out: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    diag = collections.Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        diag["lines"] += 1
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            diag["unparseable"] += 1
            continue
        if r.get("event") != "gap_closed":
            continue
        if era is not None and r.get("collector_version") != era:
            diag["era_filtered_out"] += 1
            continue
        s, e, slug = r.get("gap_start_ns"), r.get("gap_end_ns"), r.get("slug")
        if not (s and e and slug):
            diag["incomplete_gap_record"] += 1
            continue
        coin = str(slug).split("-", 1)[0]
        out[coin].append((s / 1e9, e / 1e9))
        diag["gaps"] += 1
    merged: dict[str, list[tuple[float, float]]] = {}
    for coin, iv in out.items():
        iv.sort()
        acc: list[tuple[float, float]] = []
        for a, b in iv:
            if acc and a <= acc[-1][1]:
                acc[-1] = (acc[-1][0], max(acc[-1][1], b))
            else:
                acc.append((a, b))
        merged[coin] = acc
        diag[f"merged_{coin}"] = len(acc)
    return merged, dict(diag)


def at_upper_edge(intervals: list[tuple[float, float]], t: float
                  ) -> tuple[bool, float]:
    """(is t EXACTLY at some g_end, distance to the nearest g_end).

    THE ONLY PLACE the ruled `[g0, g1)` and BE's builder `[g0, g1]` can
    disagree (R-194). Under closed containment a row exactly at g1 is flagged;
    under the ruled half-open it is not. So the difference set is precisely
    {rows with T == g_end}. If that set is EMPTY the two conventions are
    indistinguishable on this tape and the diagnostic build is equivalent to
    fixed-builder output; if it is non-empty the containment bug bites and the
    rebuild is forced.

    The distance is returned as well, because "zero hits" is only reassuring
    beside how CLOSE anything got: zero hits with rows landing microseconds
    away is a coincidence, zero hits with the nearest approach seconds away is
    a structural fact.
    """
    if not intervals:
        return False, float("inf")
    best = min(abs(t - b) for _, b in intervals)
    return best == 0.0, best


def in_gap(intervals: list[tuple[float, float]], t: float) -> bool:
    """[g_start, g_end): lower-inclusive, upper-EXCLUSIVE (R-191(2))."""
    if not intervals:
        return False
    i = bisect.bisect_right(intervals, (t, float("inf"))) - 1
    if i < 0:
        return False
    a, b = intervals[i]
    return a <= t < b


def count(tape: Path, era: str | None = None) -> dict[str, Any]:
    gaps, diag = coin_gaps(era=era)
    total = collections.Counter()
    by_coin = collections.Counter()
    by_split = collections.Counter()
    by_coin_split: collections.Counter = collections.Counter()
    flagged_ids: list[dict[str, Any]] = []
    edge_ids: list[dict[str, Any]] = []
    min_edge_dist = float("inf")
    n_rows = 0
    max_epoch_err = 0.0
    epoch_checked = 0
    missing_fields = collections.Counter()
    pred_true_status: collections.Counter = collections.Counter()
    unflagged_rows: list[dict[str, Any]] = []
    # THE OTHER DIRECTION. Reporting only predicate-true-but-unflagged makes a
    # two-sided disagreement look one-sided: 289 vs 286 reads as "3 rows" while
    # it can be 4 one way and 1 the other. A diff that can only see one
    # direction is how a net difference conceals its own size.
    flagged_not_pred_rows: list[dict[str, Any]] = []
    tape_flag_total = 0
    for r in G.iter_tape(tape):
        n_rows += 1
        coin, t0, ts = r.get("coin"), r.get("t0"), r.get("t_start")
        if coin is None or t0 is None or ts is None:
            missing_fields["coin_t0_or_t_start"] += 1
            continue
        T = float(t0) + float(ts)
        # cross-check the tape's own absolute field against the ruled formula
        dte = r.get("decision_time_epoch")
        if dte is not None:
            epoch_checked += 1
            max_epoch_err = max(max_epoch_err, abs(float(dte) - T))
        iv = gaps.get(coin, [])
        _tape_flag = str(r.get("state_status", "")) == "GAP_AT_CUTOFF"
        if _tape_flag:
            tape_flag_total += 1
            if not in_gap(iv, T):
                _lo_d = min((abs(T - a) for a, _b in iv), default=None)
                _hi_d = min((abs(T - b) for _a, b in iv), default=None)
                flagged_not_pred_rows.append({
                    "slug": r.get("slug"), "coin": coin, "side": r.get("side"),
                    "gen": r.get("gen"), "t0": t0, "t_start": ts,
                    "T_absolute_full": repr(T),   # NOT rounded: a rounded T
                    # printed next to a boundary is how a 0.0000000s distance
                    # reads as "no gap contains this row"
                    "split": r.get("split"),
                    "dist_to_nearest_g_start": _lo_d,
                    "dist_to_nearest_g_end": _hi_d})
        exact, dist = at_upper_edge(iv, T)
        if exact:
            total["at_g1_exact"] += 1
            # THE POSITIVE ASSERTION (R-196/R-199): present is only half of it.
            # A tape can show the right COUNT because half-open landed, or
            # because the gaps never arrived at all. What distinguishes them is
            # that the at-g1 rows are PRESENT **and carry a status other than
            # GAP_AT_CUTOFF**. If they are absent entirely, the gaps did not
            # reach the builder; if they are flagged, containment is still
            # closed.
            st = str(r.get("state_status", "?"))
            total[f"at_g1_status_{st}"] += 1
            if st == "GAP_AT_CUTOFF":
                total["at_g1_FLAGGED"] += 1
            if len(edge_ids) < 10:
                edge_ids.append({"slug": r.get("slug"), "t_start": ts,
                                 "T_absolute": round(T, 9),
                                 "split": r.get("split")})
        if dist < min_edge_dist:
            min_edge_dist = dist
        for band, lab in ((1e-6, "within_1us_of_g1"),
                          (1e-3, "within_1ms_of_g1"),
                          (1.0, "within_1s_of_g1")):
            if dist <= band:
                total[lab] += 1
        if in_gap(iv, T):
            total["flagged"] += 1
            # R-212 ROW-LEVEL DIFF. "flagged" here means PREDICATE-TRUE under
            # the ruled definition -- it is NOT the tape's status, and keeping
            # the two words apart is the whole point of this cross-tab. If the
            # two implementations disagree by 3, those 3 rows carry SOME tape
            # status; naming it is the difference between a diff and a guess.
            _st = str(r.get("state_status", "__ABSENT__"))
            pred_true_status[_st] += 1
            if _st != "GAP_AT_CUTOFF":
                # every one of them, not a sample: a discrepancy of 3 is not
                # something to report the first 10 of.
                unflagged_rows.append({
                    "slug": r.get("slug"), "coin": coin, "side": r.get("side"),
                    "gen": r.get("gen"), "t0": t0, "t_start": ts,
                    "T_absolute": round(T, 6),
                    "T_absolute_full": repr(T),
                    "dist_to_nearest_g_start": min(
                        (abs(T - a) for a, _b in iv), default=None),
                    "dist_to_nearest_g_end": min(
                        (abs(T - b) for _a, b in iv), default=None),
                    "split": r.get("split"),
                    "state_status": _st,
                    "decision_time_epoch": r.get("decision_time_epoch")})
            by_coin[coin] += 1
            sp = str(r.get("split", "?"))
            by_split[sp] += 1
            by_coin_split[(coin, sp)] += 1
            if len(flagged_ids) < 10:
                flagged_ids.append({
                    "slug": r.get("slug"), "side": r.get("side"),
                    "gen": r.get("gen"), "t_start": ts, "t0": t0,
                    "T_absolute": round(T, 6), "split": r.get("split"),
                    "state_status": r.get("state_status")})
    return {
        "instrument": "da_gap_at_cutoff_count_v1",
        "predicate_true_by_tape_status": dict(pred_true_status),
        "tape_flagged_total": tape_flag_total,
        "flagged_but_not_predicate_true": flagged_not_pred_rows,
        "n_flagged_but_not_predicate_true": len(flagged_not_pred_rows),
        "symmetric_diff_note": (
            "Two-sided. n_predicate_true_not_flagged = ruled-in / tape-out; "
            "n_flagged_but_not_predicate_true = tape-in / ruled-out. The NET "
            "count difference is their difference, so a net of 3 can be 4 and "
            "1. Distances to the nearest g_start/g_end are reported UNROUNDED "
            "because a boundary case rounds away."),
        "predicate_true_not_flagged": unflagged_rows,
        "n_predicate_true_not_flagged": len(unflagged_rows),
        "diff_note": ("`flagged` = PREDICATE-TRUE under the ruled definition, "
                      "NOT the tape's status. predicate_true_by_tape_status "
                      "cross-tabs the two: a row can satisfy the predicate and "
                      "carry a different status if the builder assigns one "
                      "status per row by precedence. Precedence between "
                      "GAP_AT_CUTOFF and other statuses has NOT been ruled -- "
                      "this instrument reports the rows and does not decide "
                      "(rule 14)."),
        "definition": "T = t0 + t_start in [g_start, g_end) of ANY gap of that "
                      "COIN in the collector-gaps ledger (R-191(2))",
        "era_filter": era,
        "tape": str(tape),
        "n_rows": n_rows,
        # NAME CAUTION (kept, not renamed -- 289 was exchanged under this key
        # and moving it mid-dispute would be its own name-drift): this is the
        # count of rows the RULED PREDICATE is true of, NOT the count of rows
        # the tape STATUSES as GAP_AT_CUTOFF. Those are the two numbers now in
        # dispute, so read predicate_true_by_tape_status beside it.
        "GAP_AT_CUTOFF_total": total["flagged"],
        "by_coin": dict(by_coin),
        "by_split": dict(by_split),
        "by_coin_split": {f"{c}/{s}": v for (c, s), v in by_coin_split.items()},
        "ledger": diag,
        "rows_missing_required_fields": dict(missing_fields),
        "epoch_crosscheck": {
            "rows_checked": epoch_checked,
            "max_abs_error_vs_t0_plus_t_start": round(max_epoch_err, 9)},
        "first_10_flagged": flagged_ids,
        # R-194 edge probe: the ONLY divergence between the ruled [g0,g1) and
        # the builder's [g0,g1] is rows landing exactly on g1.
        "containment_edge_probe": {
            "at_g1_exact": total.get("at_g1_exact", 0),
            "within_1us_of_g1": total.get("within_1us_of_g1", 0),
            "within_1ms_of_g1": total.get("within_1ms_of_g1", 0),
            "within_1s_of_g1": total.get("within_1s_of_g1", 0),
            "min_abs_distance_to_any_g1": (
                None if min_edge_dist == float("inf")
                else round(min_edge_dist, 9)),
            "at_g1_flagged": total.get("at_g1_FLAGGED", 0),
            "at_g1_status_breakdown": {k[len("at_g1_status_"):]: v
                                       for k, v in total.items()
                                       if k.startswith("at_g1_status_")},
            "half_open_landed": (total.get("at_g1_exact", 0) > 0
                                 and total.get("at_g1_FLAGGED", 0) == 0),
            "half_open_note": (
                "half_open_landed requires at-g1 rows to be PRESENT *and* "
                "UNFLAGGED. Present-but-flagged means containment is still "
                "closed; absent entirely means the gaps never reached the "
                "builder. The COUNT alone cannot separate those two."),
            "first_10_at_edge": edge_ids,
            "verdict": ("EDGE IMMATERIAL on this tape: no row lands exactly on "
                        "a g1, so closed and half-open containment are "
                        "indistinguishable here"
                        if total.get("at_g1_exact", 0) == 0 else
                        "EDGE BITES: rows land exactly on a g1, so the "
                        "builder's closed containment flags rows the ruled "
                        "half-open does not -- rebuild forced"),
        },
    }


def _selftests() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            raise AssertionError(f"selftest failed: {label}")

    # ---- R-212 cross-tab: it must SEPARATE predicate-true from tape status --
    import tempfile as _tf
    global PM_GAPS
    _realg = PM_GAPS
    try:
        with _tf.TemporaryDirectory() as _td:
            _g = Path(_td) / "gaps.jsonl"
            _g.write_text(json.dumps({
                "event": "gap_closed", "coin": "btc", "slug": "btc-1000",
                "gap_start_ns": int(1000.0 * 1e9),
                "gap_end_ns": int(2000.0 * 1e9)}), encoding="utf-8")
            PM_GAPS = _g
            _t = Path(_td) / "tape.json"
            # three rows INSIDE the gap: one correctly flagged, two masked by
            # some other status -- exactly the shape a 3-row discrepancy takes
            _rows = [
                {"slug": "btc-1", "coin": "btc", "t0": 1500.0, "t_start": 0.0,
                 "state_status": "GAP_AT_CUTOFF"},
                {"slug": "btc-2", "coin": "btc", "t0": 1500.0, "t_start": 1.0,
                 "state_status": "PRE_WINDOW"},
                {"slug": "btc-3", "coin": "btc", "t0": 1500.0, "t_start": 2.0,
                 "state_status": "NO_LEVEL_HISTORY"},
                {"slug": "btc-4", "coin": "btc", "t0": 9000.0, "t_start": 0.0,
                 "state_status": "OK"},          # outside the gap entirely
            ]
            _t.write_text(json.dumps({"rows": _rows}), encoding="utf-8")
            _rep = count(_t)
            ok(_rep["GAP_AT_CUTOFF_total"] == 3,
               "three rows satisfy the ruled predicate (the fourth is outside)")
            ok(_rep["predicate_true_by_tape_status"]
               == {"GAP_AT_CUTOFF": 1, "PRE_WINDOW": 1, "NO_LEVEL_HISTORY": 1},
               "the cross-tab SEPARATES predicate-true from the tape's status "
               "-- a count alone cannot tell 3 flagged from 1 flagged + 2 masked")
            ok(_rep["n_predicate_true_not_flagged"] == 2,
               "the masked rows are counted")
            _slugs = sorted(x["slug"] for x in _rep["predicate_true_not_flagged"])
            ok(_slugs == ["btc-2", "btc-3"],
               "and NAMED individually -- a 3-row discrepancy is not something "
               "to report the first ten of")
            # the OTHER direction must be seen too: a row the TAPE flags that
            # the ruled predicate does not. Without it a 4-vs-1 disagreement
            # reports as "3".
            _rows3 = [
                {"slug": "btc-9", "coin": "btc", "t0": 9000.0, "t_start": 0.0,
                 "state_status": "GAP_AT_CUTOFF"},        # outside every gap
                {"slug": "btc-1", "coin": "btc", "t0": 1500.0, "t_start": 0.0,
                 "state_status": "GAP_AT_CUTOFF"},        # inside, agreed
            ]
            _t.write_text(json.dumps({"rows": _rows3}), encoding="utf-8")
            _rep3 = count(_t)
            ok(_rep3["n_flagged_but_not_predicate_true"] == 1
               and _rep3["flagged_but_not_predicate_true"][0]["slug"] == "btc-9",
               "a row the TAPE flags but the ruled predicate does not is caught "
               "and NAMED -- the direction a one-sided diff cannot see")
            ok(_rep3["tape_flagged_total"] == 2,
               "the tape's own flag total is reported beside the predicate's, "
               "so the two numbers in dispute are both in one artifact")
            ok(_rep3["flagged_but_not_predicate_true"][0]
               ["dist_to_nearest_g_start"] is not None,
               "with an UNROUNDED distance to the nearest boundary -- a "
               "rounded T beside a boundary reads as 'no gap contains this row'")

            # FALSIFIER: with every row correctly flagged there must be NO
            # masked rows, or the instrument would 'find' masking anywhere
            _rows2 = [dict(r, state_status="GAP_AT_CUTOFF") for r in _rows[:3]]
            _t.write_text(json.dumps({"rows": _rows2}), encoding="utf-8")
            _rep2 = count(_t)
            ok(_rep2["n_predicate_true_not_flagged"] == 0
               and _rep2["GAP_AT_CUTOFF_total"] == 3,
               "a fully-flagged tape reports ZERO masked rows -- the cross-tab "
               "can come back empty, so a non-empty answer means something")
    finally:
        PM_GAPS = _realg

    iv = [(100.0, 200.0), (300.0, 400.0)]
    ok(in_gap(iv, 100.0), "LOWER bound is INCLUSIVE")
    ok(not in_gap(iv, 200.0), "UPPER bound is EXCLUSIVE -- [g_start, g_end)")
    ok(in_gap(iv, 199.999), "just inside the upper bound is in")
    ok(not in_gap(iv, 99.999), "just below the lower bound is out")
    ok(in_gap(iv, 350.0), "a later interval is found (not just the first)")
    ok(not in_gap(iv, 250.0), "between intervals is out")
    ok(not in_gap([], 150.0), "no gaps for a coin means nothing is flagged")
    ok(in_gap(iv, 399.999) and not in_gap(iv, 400.0),
       "the exclusive upper bound holds on the LAST interval too")

    # merging must not create membership that neither original had
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "g.jsonl"
        recs = [
            {"event": "gap_closed", "slug": "btc-updown-5m-1", "collector_version": ERA,
             "gap_start_ns": int(100e9), "gap_end_ns": int(200e9)},
            {"event": "gap_closed", "slug": "btc-updown-5m-2", "collector_version": ERA,
             "gap_start_ns": int(150e9), "gap_end_ns": int(250e9)},
            {"event": "gap_closed", "slug": "eth-updown-5m-9", "collector_version": ERA,
             "gap_start_ns": int(100e9), "gap_end_ns": int(200e9)},
            {"event": "collector_start", "recv_ns": 1},
        ]
        f.write_text("\n".join(json.dumps(r) for r in recs) + "\nnot json\n",
                     encoding="utf-8")
        g, d = coin_gaps(f)
        ok(d["unparseable"] == 1, "an unreadable ledger line is COUNTED")
        ok(d["gaps"] == 3, "non-gap events are not counted as gaps")
        ok(g["btc"] == [(100.0, 250.0)],
           "overlapping COIN-level gaps merge into one interval")
        ok(g["eth"] == [(100.0, 200.0)], "coins are kept separate")
        ok(in_gap(g["btc"], 220.0) and not in_gap(g["eth"], 220.0),
           "COIN-LEVEL scope: an instant in btc's gap is NOT in eth's -- the "
           "same instant gets different answers per coin, so a coin-blind "
           "rule cannot pass (R-42 mirror)")
        # the per-slug projection the ruling calls lossy
        ok(in_gap(g["btc"], 210.0),
           "an instant covered by the gap logged against slug-2 is in scope "
           "for ANY btc row -- this is precisely what per-slug scoping drops")
        gera, _ = coin_gaps(f, era="other_era")
        ok(gera == {}, "an era filter that matches nothing yields no gaps")

    # --- R-194 containment-edge probe ------------------------------------
    e, d = at_upper_edge(iv, 200.0)
    ok(e and d == 0.0, "a row EXACTLY at g1 is detected as an edge hit")
    ok(not in_gap(iv, 200.0),
       "...and the RULED containment does NOT flag it -- which is exactly the "
       "divergence from the builder's closed form")
    e2, d2 = at_upper_edge(iv, 199.999)
    ok(not e2 and abs(d2 - 0.001) < 1e-9,
       "a row just inside reports its distance, not an edge hit")
    ok(in_gap(iv, 199.999), "...and IS flagged by the ruled containment")
    e3, d3 = at_upper_edge(iv, 500.0)
    ok(not e3 and d3 == 100.0, "distance is to the NEAREST g1, across intervals")
    ok(at_upper_edge([], 1.0) == (False, float("inf")),
       "no intervals means no edge and infinite distance, not a false hit")

    print(f"da_gap_at_cutoff_count selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["count"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tape", default=None)
    ap.add_argument("--era", default=None,
                    help="optional collector_version filter (diagnostic)")
    a = ap.parse_args()
    if a.selftest or not a.cmd:
        return _selftests()
    if not a.tape:
        raise SystemExit("--tape PATH required; refusing to guess a tape")
    rep = count(Path(a.tape), era=a.era)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
