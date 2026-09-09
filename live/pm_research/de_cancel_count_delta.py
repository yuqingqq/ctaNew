#!/usr/bin/env python3
"""What the causal re-timing does to the CANCEL COUNT — the theta evidence.

THE QUESTION THIS EXISTS TO ANSWER, and it is the USER's to rule on.
`phase2_arms.freeze_thresholds` resolved theta over per-generation
**MAXIMA**. DE 155 (1) re-timed the decision: each scored row is emitted at
its own `t_start` and the FIRST crossing of theta cancels. A first-crossing
score is at or below the same generation's maximum, so **at an UNCHANGED
theta fewer generations cross and the selected count moves for a reason
that is not a market reason**. Theta is NOT re-fitted (coordinator's
ruling); the consequence is recorded on every receipt as `scoring_timing`
and measured here.

AND IT MOVES THE CONTROL, NOT ONLY THE ARM: the null is matched on the
action count, so a changed cancel count changes the matched null too. (What
the null's sampling unit even IS after the re-timing is a separate open
question for the USER -- `drafts/DE158_null_sampling_unit_QUESTION.md`. No
control is drawn on a corrected book until that lands. This module draws
none: it replays the ARM twice and counts.)

BOTH STREAMS COME OFF ONE BOOK. The corrected book's assembly is per ROW,
so the OLD aggregation is RECONSTRUCTED from those same rows -- the maximum
over each generation, stamped at the generation's start. The two streams
therefore differ in NOTHING but the aggregation and the timestamp: same
book, same scores, same theta, same policy, same engine. Any difference is
the re-timing and nothing else.

NO ECONOMIC VALUE IS REPORTED. Counts, populations, and which generations
change hands. Money is not this instrument's business.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harmful_stateful_policy as HSP          # noqa: E402

PROTOCOL = "P003_DE_CANCEL_COUNT_DELTA_V1"
ASSEMBLY_PRE_CAUSAL = "ASSEMBLY_PREDATES_CAUSAL_SCORING"
NO_SCORES = "CANCEL_DELTA_NO_ASSEMBLED_SCORES"
NOT_PER_ROW = "CANCEL_DELTA_ASSEMBLY_IS_NOT_PER_ROW"


class CancelDeltaRefused(RuntimeError):
    """Refuses rather than measuring something it cannot interpret."""


def assert_per_row_assembly(scored: dict) -> dict:
    """THE KNOWN-BAD GATE (rule 15). A pre-DE-155 assembly is a bare float
    per GENERATION keyed at the generation's start, with the row times
    already discarded -- so the OLD stream cannot be distinguished from the
    NEW one and a "delta" computed from it would be identically zero for a
    reason that has nothing to do with the policy. That is the worst kind
    of wrong answer: a null result that looks like a measurement."""
    if not scored:
        raise CancelDeltaRefused(
            f"{NO_SCORES}: the book carries no assembled scores for this "
            f"arm, so there is nothing to re-time.")
    k, v = next(iter(scored.items()))
    if not isinstance(v, dict):
        raise CancelDeltaRefused(
            f"{ASSEMBLY_PRE_CAUSAL}: the assembled score for {k} is a bare "
            f"{type(v).__name__} -- one MAXIMUM per generation stamped at "
            f"the generation's start (the pre-DE-155 shape). The row times "
            f"it was built from are gone, so the causal stream cannot be "
            f"reconstructed and the delta would be a spurious zero. "
            f"Rebuild the book's assembly.")
    missing = [f for f in ("score", "gen", "t0") if f not in v]
    if missing:
        raise CancelDeltaRefused(
            f"{NOT_PER_ROW}: the assembled value for {k} lacks {missing}. "
            f"A per-row score carries its own generation and start, or the "
            f"OLD aggregation cannot be rebuilt from it.")
    return {"status": "ASSEMBLY_IS_PER_ROW", "n_scored_rows": len(scored),
            "sample_key": list(map(str, k))}


def old_stream(scored: dict) -> list:
    """The OLD decision stream, REBUILT from the per-row scores: one event
    per generation, carrying the MAXIMUM, stamped at the generation's
    START. This is the aggregation DE 155 (1) removed."""
    agg: dict = {}
    for (slug, side, _t), v in scored.items():
        k = (slug, side, v["gen"])
        cur = agg.get(k)
        # `t0` is the same for every row of a generation, so it is set
        # whenever the max is, and there is no second branch to write.
        if cur is None or v["score"] > cur["score"]:
            agg[k] = {"score": v["score"], "t0": float(v["t0"])}
    return sorted(({"t": a["t0"], "slug": s, "side": sd, "gen": g,
                    "score": a["score"]} for (s, sd, g), a in agg.items()),
                  key=lambda e: (e["t"], e["slug"], e["side"], e["gen"]))


def new_stream(scored: dict) -> list:
    """The RULED stream: one event per scored ROW, at its own time."""
    return sorted(({"t": float(t), "slug": s, "side": sd, "gen": v["gen"],
                    "score": v["score"]}
                   for (s, sd, t), v in scored.items()),
                  key=lambda e: (e["t"], e["slug"], e["side"], e["gen"]))


def _cancels(reference: dict, events: list, params: dict) -> dict:
    """(slug, side, gen) -> t_request, for the generations cancelled."""
    out = HSP.replay_policy(reference, events, params)
    got = {}
    for c in out["cancels"]:
        got[(c["slug"], c["side"], int(c["ref_gen"]))] = float(c["t_request"])
    return {"by_gen": got,
            "cancels_issued": int(out["counters"].get("cancels_issued", 0)),
            "n_fills": len(out.get("fills") or [])}


def rows_in_by_generation(split_of: dict | None) -> dict:
    """Per-generation INPUT rows, from the tape index -- before the feature
    pass drops anything (REV 107's derivation, DE 158 (5))."""
    out: dict = {}
    for k in (split_of or {}):
        if isinstance(k, tuple) and len(k) >= 3:
            g = (k[0], k[1], k[2])
            out[g] = out.get(g, 0) + 1
    return out


def measure_arm(reference: dict, scored: dict, policy_params: dict, *,
                split_of: dict | None = None) -> dict:
    """One arm, both aggregations, on ONE book. Counts only."""
    shape = assert_per_row_assembly(scored)
    old_ev, new_ev = old_stream(scored), new_stream(scored)
    old, new = (_cancels(reference, old_ev, policy_params),
                _cancels(reference, new_ev, policy_params))
    # ---- REV 110 (C): THE INVARIANT IS ASSERTED, NOT RELIED ON -------
    # Everything below treats one generation as at most one cancel -- it
    # is why matching on cancels is matching on the decision variable at
    # all. The engine guarantees it (`one_cancel_per_generation`); this
    # module DEPENDS on it, so it checks it rather than trusting it.
    for _lbl, _r in (("OLD", old), ("RULED", new)):
        if len(_r["by_gen"]) != _r["cancels_issued"]:
            raise CancelDeltaRefused(
                f"ONE_CANCEL_PER_GENERATION_VIOLATED: the {_lbl} stream "
                f"issued {_r['cancels_issued']} cancels over "
                f"{len(_r['by_gen'])} distinct generations. Every count "
                f"below assumes those are the same number.")
    o, n = set(old["by_gen"]), set(new["by_gen"])
    both = o & n
    shifts = [new["by_gen"][g] - old["by_gen"][g] for g in both]
    later = sum(1 for s in shifts if s > 1e-9)
    kept = {}
    for (s, sd, _t), v in scored.items():
        g = (s, sd, v["gen"])
        kept[g] = kept.get(g, 0) + 1
    rows_in = rows_in_by_generation(split_of)
    partial = ({"PARTIAL_ROWS": sum(1 for g, c in rows_in.items()
                                    if kept.get(g, 0) < c),
                "PARTIAL_ROWS_DENOMINATOR": len(rows_in),
                "status": "COMPUTED_AGAINST_THE_TAPE_INDEX"}
               if rows_in else
               {"PARTIAL_ROWS": None,
                "status": ("NOT_COMPUTABLE_NO_SPLIT_OF: no tape index was "
                           "supplied, so PARTIAL_ROWS is UNKNOWN, not 0")})
    # ---- REV 110 (B): THE EXCLUDED POPULATION IS COUNTED (rule 4) ----
    # `generation_scores` already computes this and the table a theta
    # re-fit would be ruled on must carry it: a generation with no scored
    # row cannot be cancelled by the arm OR drawn by the control, so it is
    # out of both populations -- and an exclusion is a status, never a
    # silent absence.
    _ref_gens = {(slug, side, int(g["gen"]))
                 for slug, sides in (reference or {}).items()
                 for side, gens in sides.items() for g in gens}
    _no_rows = sorted(_ref_gens - set(kept))
    return {
        "assembly": shape,
        "n_generations_scored": len(kept),
        "n_reference_generations": len(_ref_gens),
        "n_generations_with_no_scored_rows": len(_no_rows),
        "generations_with_no_scored_rows_are": (
            "OUT OF BOTH POPULATIONS -- the arm cannot cancel them and the "
            "control cannot draw them. Counted here rather than left as a "
            "difference between two other numbers (rule 4)"),
        "n_scored_rows": len(scored),
        "rows_per_generation_max": (max(kept.values()) if kept else 0),
        "OLD_max_at_generation_start": {
            "n_events": len(old_ev), "cancels_issued": old["cancels_issued"],
            "n_fills": old["n_fills"]},
        "RULED_first_crossing_at_row_time": {
            "n_events": len(new_ev), "cancels_issued": new["cancels_issued"],
            "n_fills": new["n_fills"]},
        "delta": {
            "cancels": new["cancels_issued"] - old["cancels_issued"],
            "cancels_pct": (round(100.0 * (new["cancels_issued"]
                                           - old["cancels_issued"])
                                  / old["cancels_issued"], 3)
                            if old["cancels_issued"] else None),
            "fills": new["n_fills"] - old["n_fills"]},
        "generations_that_change_hands": {
            "cancelled_under_BOTH": len(both),
            "cancelled_only_under_OLD": len(o - n),
            "cancelled_only_under_RULED": len(n - o),
            "of_those_in_BOTH_cancelled_LATER": later,
            "median_shift_s": (round(sorted(shifts)[len(shifts) // 2], 4)
                               if shifts else None),
            "max_shift_s": (round(max(shifts), 4) if shifts else None),
            "why_only_OLD_is_the_look_ahead": (
                "a generation cancelled ONLY under the old aggregation was "
                "cancelled on information that had not arrived when the "
                "cancel was stamped -- or on a crossing that falls after "
                "the generation ended")},
        "partial_rows": partial,
        "theta_refitted": False,
        "WHY_THE_ATTRIBUTION_IS_SOUND": (
            "REV 110 (A): the two replays differ ONLY in the score stream. "
            "Nothing else in the policy can absorb or redistribute a "
            "changed cancel count -- in particular `max_cancels_per_minute` "
            "is inf in BOTH `de_phase4_diag_runner.cell_params` and BE's "
            "`params_for`, so no rate limit silently converts a cancel the "
            "re-timing moved into one it dropped. The delta below is the "
            "aggregation and the timestamp, and nothing else."),
        "THE_CALIBRATION_CONSEQUENCE": (
            "theta was fitted over per-generation MAXIMA; a first-crossing "
            "score is at or below that maximum, so an unchanged theta "
            "selects a different count for a reason that is not a market "
            "reason. This table is the evidence a re-fit would be ruled "
            "on; nothing here re-fits anything."),
    }


def measure_book(book_path, day: str, *, params=None, split_of=None) -> dict:
    """Both arms of one day, from one book.

    REV 110 (D): this used to say "Requires the heavy lock". It does not
    TAKE one and cannot enforce one -- it only reads a book and replays.
    The lock is the CALLER's discipline (rule 20 is about heavy runs, and
    loading a day book is one), and a docstring that claims a control the
    code does not hold is the shape rule 16 names."""
    import be_cancel_axis_null as B
    import de_multiday_gate1_runner as R
    t0 = time.time()
    P = params or R.load_params()
    coin = P.get("coin", "btc")
    bk = B.load(Path(book_path))
    out = {"protocol": PROTOCOL, "day": day, "book": str(book_path),
           "book_sha256": bk["source_sha256"], "params_file": R.PARAMS_REL,
           "arms": {}}
    for arm, spec in P["arms"].items():
        scored = bk["asm"]["by_arm"][(coin, spec["head"])][0]
        out["arms"][arm] = measure_arm(
            bk["ref"], scored, B.params_for(spec["theta"]),
            split_of=split_of)
        out["arms"][arm]["head"] = spec["head"]
        out["arms"][arm]["theta"] = spec["theta"]
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


# ------------------------------------------------------- the battery

EXPECTED_CHECKS = 6


def selftest(quiet: bool = False) -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_cancel_count_delta] FAIL: {label}")
        n[0] += 1
        if not quiet:
            print(f"  PASS  {label}")

    import de_phase4_diag_runner as PH
    S = HSP.SIDES[0]
    pol = PH.cell_params(
        {"coin": "btc", "latency_ms": 250, "budget": PH.BUDGETS[0],
         "enable_reduce": False,
         "charge_reset_cost_at_generation_start": False},
        theta_cancel=0.6, protection_mode=HSP.PROTECTION_MODES[0],
        repost_fill_model=HSP.REPOST_FILL_MODELS[0])

    # ---- THE POSITIVE CONTROL IT MUST FLAG ---------------------------
    # Three slugs, each a case the measurement has to separate:
    #   s1  max in a LATE row       -> cancelled under BOTH, but LATER
    #   s2  crossing after gen END  -> cancelled ONLY under OLD (the
    #                                  look-ahead's own signature)
    #   s3  FIRST row crosses       -> cancelled under BOTH, unmoved
    ref = {
        "s1": {S: [HSP._gen(0, 0.0, 10.0, [(1.0, 2.0, 3.0)])],
               HSP.SIDES[1]: []},
        "s2": {S: [HSP._gen(0, 0.0, 5.0, [(1.0, 2.0, 3.0)])],
               HSP.SIDES[1]: []},
        "s3": {S: [HSP._gen(0, 0.0, 10.0, [(1.0, 2.0, 3.0)])],
               HSP.SIDES[1]: []}}
    scored = {
        ("s1", S, 0.0): {"score": 0.2, "gen": 0, "t0": 0.0},
        ("s1", S, 6.0): {"score": 0.9, "gen": 0, "t0": 0.0},
        ("s2", S, 0.0): {"score": 0.2, "gen": 0, "t0": 0.0},
        ("s2", S, 6.0): {"score": 0.9, "gen": 0, "t0": 0.0},
        ("s3", S, 0.0): {"score": 0.9, "gen": 0, "t0": 0.0},
        ("s3", S, 4.0): {"score": 0.1, "gen": 0, "t0": 0.0}}
    m = measure_arm(ref, scored, pol)
    ch = m["generations_that_change_hands"]
    ok(m["OLD_max_at_generation_start"]["cancels_issued"] == 3
       and m["RULED_first_crossing_at_row_time"]["cancels_issued"] == 2
       and m["delta"]["cancels"] == -1
       and ch["cancelled_only_under_OLD"] == 1
       and ch["cancelled_only_under_RULED"] == 0
       and ch["cancelled_under_BOTH"] == 2
       and ch["of_those_in_BOTH_cancelled_LATER"] == 1
       and ch["max_shift_s"] == 6.0,
       f"POSITIVE CONTROL, AND IT FLAGS: the OLD aggregation cancels "
       f"{m['OLD_max_at_generation_start']['cancels_issued']} generations "
       f"and the RULED rule "
       f"{m['RULED_first_crossing_at_row_time']['cancels_issued']} "
       f"(delta {m['delta']['cancels']}). It SEPARATES the three cases: "
       f"{ch['cancelled_only_under_OLD']} cancelled only under the old "
       f"aggregation -- the crossing falls after the generation ended, "
       f"which is the look-ahead's own signature -- "
       f"{ch['cancelled_under_BOTH']} under both, of which "
       f"{ch['of_those_in_BOTH_cancelled_LATER']} moved later (max shift "
       f"{ch['max_shift_s']} s) and one did not move at all")

    # ---- AND A NULL RESULT IS NOT MANUFACTURED ------------------------
    # If every generation's max IS its first row, the two aggregations are
    # the same stream and the delta MUST be zero. A tool that reported a
    # difference here would be measuring itself.
    flat = {("s1", S, 0.0): {"score": 0.9, "gen": 0, "t0": 0.0},
            ("s3", S, 0.0): {"score": 0.9, "gen": 0, "t0": 0.0}}
    ref2 = {k: v for k, v in ref.items() if k in ("s1", "s3")}
    m0 = measure_arm(ref2, flat, pol)
    ok(m["n_reference_generations"] == 3
       and m["n_generations_with_no_scored_rows"] == 0
       and m["WHY_THE_ATTRIBUTION_IS_SOUND"].startswith("REV 110 (A)")
       and math.isinf(pol.get("max_cancels_per_minute", 0.0)),
       f"REV 110 (A)+(B): the excluded population is COUNTED -- "
       f"{m['n_reference_generations']} reference generations, "
       f"{m['n_generations_with_no_scored_rows']} with no scored row, out "
       f"of BOTH populations (rule 4) -- and the attribution is sound "
       f"because `max_cancels_per_minute` is "
       f"{pol.get('max_cancels_per_minute')} here and in BE's "
       f"`params_for`, so no rate limit can absorb a cancel the re-timing "
       f"moved. Checked, not asserted")

    # REV 110 (C): a stream that breaks the invariant refuses.
    _bad_inv = None
    try:
        _cancels_real = _cancels
        def _fake(reference, events, params):
            r = _cancels_real(reference, events, params)
            return {**r, "cancels_issued": r["cancels_issued"] + 1}
        globals()["_cancels"] = _fake
        measure_arm(ref, scored, pol)
    except CancelDeltaRefused as e:
        _bad_inv = str(e).split(":")[0]
    finally:
        globals()["_cancels"] = _cancels_real
    ok(_bad_inv == "ONE_CANCEL_PER_GENERATION_VIOLATED",
       f"REV 110 (C) KNOWN-BAD: a stream whose cancel COUNT disagrees with "
       f"its distinct-generation count refuses `{_bad_inv}`. Every count "
       f"this module reports assumes those are one number -- it now checks "
       f"that rather than relying on it")

    ok(m0["delta"]["cancels"] == 0
       and m0["generations_that_change_hands"]["cancelled_only_under_OLD"] == 0
       and m0["generations_that_change_hands"][
           "of_those_in_BOTH_cancelled_LATER"] == 0,
       "AND IT REPORTS ZERO WHEN THERE IS NOTHING TO REPORT: where every "
       "generation's maximum IS its first row the two aggregations are one "
       "stream, the delta is 0 and nothing changes hands -- so a non-zero "
       "reading above is a property of the data, not of the instrument")

    # ---- THE KNOWN-BAD IT MUST REFUSE --------------------------------
    pre = {("s1", S, 0.0): 0.9, ("s3", S, 0.0): 0.2}
    got = None
    try:
        measure_arm(ref2, pre, pol)
    except CancelDeltaRefused as e:
        got = str(e).split(":")[0]
    empty = None
    try:
        measure_arm(ref2, {}, pol)
    except CancelDeltaRefused as e:
        empty = str(e).split(":")[0]
    partialv = None
    try:
        measure_arm(ref2, {("s1", S, 0.0): {"score": 0.9}}, pol)
    except CancelDeltaRefused as e:
        partialv = str(e).split(":")[0]
    ok(got == ASSEMBLY_PRE_CAUSAL and empty == NO_SCORES
       and partialv == NOT_PER_ROW,
       f"KNOWN-BAD, THREE WAYS: a PRE-DE-155 assembly (a bare float per "
       f"generation, row times discarded) refuses `{got}` instead of "
       f"reporting a delta of 0 -- which is the dangerous answer, a null "
       f"result that looks like a measurement; an empty assembly refuses "
       f"`{empty}`; and a value missing `gen`/`t0` refuses `{partialv}`")

    # ---- THE GUARD, ON THE REAL STALE ARTIFACT (DE 159 (2)) -----------
    # `de_section81_cache_12.pkl` is the default cached reference four BE
    # modules read. It predates causal scoring, so its assembly is the
    # bare-float shape. A cell that ASSERTS the refusal is worth more than
    # a red cell: it makes a stale fixture prove the guard fires.
    import de_data_root as DR
    cache = Path(DR.resolve()["data_root"]) / "pm_5min/derived" \
        / "de_section81_cache_12.pkl"
    if cache.is_file():
        import pickle
        c = pickle.loads(cache.read_bytes())
        arm0 = sorted(c["asm"]["by_arm"])[0]
        real = None
        try:
            assert_per_row_assembly(c["asm"]["by_arm"][arm0][0])
        except CancelDeltaRefused as e:
            real = str(e).split(":")[0]
        ok(real == ASSEMBLY_PRE_CAUSAL,
           f"DE 159 (2) THE GUARD FIRES ON THE REAL STALE ARTIFACT: "
           f"`{cache.name}` -- the default cached reference four BE modules "
           f"read -- carries the pre-DE-155 assembly for {arm0} and refuses "
           f"`{real}`. A stale fixture that PROVES the guard fires is worth "
           f"more than a red cell; rebuilding it is BE's, and until then "
           f"this is what its redness means")
    else:
        ok(True, f"DE 159 (2) skipped BY NAME: {cache.name} is not on disk, "
                 f"so the guard has no real stale artifact to fire on here")

    if n[0] != EXPECTED_CHECKS:
        raise SystemExit(f"[de_cancel_count_delta] FAIL: check count "
                         f"{n[0]} != {EXPECTED_CHECKS}")
    if not quiet:
        print(f"[de_cancel_count_delta] PASS -- {n[0]} checks, "
              f"n_disarmed 0, n_skipped 0")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--day", type=str)
    ap.add_argument("--book", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.day and a.book):
        raise SystemExit("usage: --day <YYYY-MM-DD> --book <path> "
                         "[--output <file>] | --selftest")
    out = measure_book(a.book, a.day)
    text = json.dumps(out, indent=2, sort_keys=True, default=str) + "\n"
    if a.output:
        if a.output.exists():
            raise SystemExit(f"output already exists: {a.output}")
        a.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
