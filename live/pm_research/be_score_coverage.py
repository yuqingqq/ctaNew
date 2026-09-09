"""GENERATION COVERAGE OF AN ASSEMBLED SCORE MAP, IN BOTH OF ITS SHAPES.

WHY THIS MODULE EXISTS (R-841 [1] and [2], BE 112). `asm["by_arm"][(coin,
head)][0]` changed shape at DE 155. It used to be ONE FLOAT PER GENERATION
keyed at that generation's `t0`; it is now ONE ENTRY PER SCORED ROW keyed at
the ROW's own time, the value carrying `{"score", "gen", "t0"}` -- the
generation's identity moved OUT of the key and INTO the value.

Two of this seat's sites went on asking the OLD question of the NEW map:

    (slug, side, float(g["t0"])) in gen_scores

Under `PER_ROW_SCORES` that is a FIRST-ROW-ONLY test. It admits a generation
only when some scored row happens to sit exactly at the generation's start,
and it drops every generation whose first scored row does not -- silently,
as a coverage shortfall rather than as a status. It is the same defect BE
107 fixed in `be_cancel_axis_null.load()` and did not look for anywhere
else.

The second half is worse because it looks like arithmetic. Both sites
reported `len(gen_scores)` in the same block as the covered count, and the
builder then computed `n_uncovered = n_reference_generations - n_covered`.
Under the new shape `len(gen_scores)` is a count of ROWS while the other two
are counts of GENERATIONS. Three numbers in one block, two units, presented
as commensurable -- the exact class `_assembly_evidence` was corrected for
in round 60 (fragment ROWS against reference GENERATIONS), returning through
a different door.

WHAT THIS MODULE COMPUTES, and what it refuses to do. It resolves an
assembled score map to GENERATION coverage under either shape, names the
shape it found, gives every count its UNIT, and counts each exclusion as a
status (rule 4). It also recomputes the PRE-FIX test's own answer beside the
corrected one, so the delta between them is a MEASUREMENT in every receipt
rather than something a later reader has to infer.

It decides nothing. `assert_coverage` in `be_daybook_build` adjudicates;
this reports (rule 14).

TWO IMPLEMENTATIONS EXIST, DELIBERATELY, AND THE SEAM IS DRIVEN.
`be_cancel_axis_null.load()` carries its own shape handling and is NOT
converted to import this module: it is the ENTRY POINT of the cascade pinned
by `de_multiday_gate1_params.be_module` and `be_cascade.modules`, so editing
it re-pins params AND design and blocks every day run until both land
(R-835: that cost three pin pairs in one day). Rather than leave two
implementations to drift, `--falsify` drives BOTH over ONE synthetic per-row
book and asserts they agree on the shape, on the exclusion counts and on the
generation population. A divergence fails this module's battery.

THE FIXTURE IS SYNTHETIC AND THAT IS THE POINT (rule 15). No PER_ROW book
exists on disk -- every landed book predates the repair -- so the predicate
is falsified against an assembly built to have the property: generations
whose first scored row is NOT at their start, generations with several rows,
and a generation with no scored row at all. A fixture proves the membership
test; a book would prove nothing the fixture cannot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

#: The two shapes an assembled score map can have, plus the third answer
#: that is neither: an EMPTY map has no shape, and calling it
#: PER_GENERATION (which `any(...)` over no values does) would be a guess
#: reported as a reading.
SHAPE_PER_ROW = "PER_ROW_SCORES"
SHAPE_PER_GENERATION = "PER_GENERATION_SCORES"
SHAPE_EMPTY = "EMPTY_SCORE_MAP"
SHAPES = (SHAPE_PER_ROW, SHAPE_PER_GENERATION, SHAPE_EMPTY)

SHAPE_MEANING = {
    SHAPE_PER_ROW:
        "one entry per DECISION ROW, keyed (slug, side, ROW TIME), the value "
        "a dict carrying `score`, `gen` and `t0`. The generation is named by "
        "the VALUE. Books built through the corrected scorer (DE 155).",
    SHAPE_PER_GENERATION:
        "one bare float per GENERATION, keyed (slug, side, GENERATION t0). "
        "The generation is named by the KEY and the row times are already "
        "discarded. Every book and cache built before the look-ahead repair.",
    SHAPE_EMPTY:
        "no entries at all, so neither shape is observable. Reported as its "
        "own answer rather than defaulted, because `any(...)` over an empty "
        "map returns False and would publish PER_GENERATION as a reading.",
}

#: The membership test these two sites used before this module, kept as a
#: string so a receipt can say which test produced which number.
PRE_FIX_TEST = "(slug, side, float(g['t0'])) in gen_scores"


class CoverageRefused(RuntimeError):
    """A named refusal."""


def score_shape(gen_scores) -> str:
    """The shape of one assembled score map, or a REFUSAL by name.

    A map whose values are not all of one shape is refused rather than
    coerced: `be_cancel_axis_null.load()` would take the per-row branch on
    it and raise `TypeError`/`KeyError` at the first bare float, which is a
    refusal without a name. Same verdict, said out loud."""
    if not isinstance(gen_scores, dict):
        raise CoverageRefused(
            f"SCORES_NOT_A_MAP: the assembled scores are a "
            f"{type(gen_scores).__name__}, not a dict. There is no key to "
            f"resolve a generation from.")
    if not gen_scores:
        return SHAPE_EMPTY
    dicts = [k for k, v in gen_scores.items()
             if isinstance(v, dict) and "gen" in v]
    if not dicts:
        return SHAPE_PER_GENERATION
    if len(dicts) != len(gen_scores):
        raise CoverageRefused(
            f"MIXED_SCORE_SHAPES: {len(dicts)} of {len(gen_scores)} scored "
            f"values are per-row dicts carrying `gen` and the rest are not. "
            f"A map that is half one shape and half the other names its "
            f"generations two ways, and no single coverage question can be "
            f"asked of it.")
    return SHAPE_PER_ROW


def generation_coverage(ref: dict, gen_scores: dict, *, sides) -> dict:
    """GENERATION coverage of `gen_scores` over the reference `ref`.

    `sides` is supplied by the caller and never guessed here: the builder
    names its two sides as literals and the derivation reads
    `harmful_stateful_policy.SIDES`, and a module that picked one of those
    for them could walk a side the caller does not have and report coverage
    of a population the caller never built.

    STRUCTURAL faults refuse by name -- a malformed key, a generation with
    no `t0` or no `gen`, a mixed map: they are things this function cannot
    compute over. POPULATION facts are counted and returned -- uncovered
    generations, keys naming no generation, duplicate identities: they are
    things the BOOK must not have, and `assert_coverage` adjudicates them
    (rule 14)."""
    sides = tuple(sides)
    if not sides:
        raise CoverageRefused(
            "NO_SIDES: an empty side list would walk no generation and "
            "report perfect coverage of nothing. The caller names its "
            "sides.")
    shape = score_shape(gen_scores)
    per_row = shape == SHAPE_PER_ROW

    # ---- STAGE 1: THE REFERENCE, WALKED ONCE ---------------------------
    n_gen = 0
    ref_gens: set = set()
    t0_of: dict = {}
    span_of: dict = {}
    gen_at_t0: dict = {}
    dup_gen_id = 0
    shared_t0 = 0
    no_t1 = 0
    for slug in sorted(ref):
        by_side = ref[slug] or {}
        for side in sides:
            for g in (by_side.get(side) or ()):
                n_gen += 1
                if "t0" not in g:
                    raise CoverageRefused(
                        f"GENERATION_WITHOUT_T0: a reference generation of "
                        f"({slug!r}, {side!r}) carries no `t0`, so neither "
                        f"the pre-fix test nor this one can name it.")
                if "gen" not in g:
                    raise CoverageRefused(
                        f"GENERATION_WITHOUT_ID: a reference generation of "
                        f"({slug!r}, {side!r}) at t0={g['t0']} carries no "
                        f"`gen`. Under PER_ROW_SCORES the generation is "
                        f"named by the score's `gen` and there would be "
                        f"nothing in the reference to match it to.")
                t0 = float(g["t0"])
                k = (slug, side, g["gen"])
                if k in ref_gens:
                    dup_gen_id += 1
                ref_gens.add(k)
                t0_of[k] = t0
                if (slug, side, t0) in gen_at_t0:
                    shared_t0 += 1
                else:
                    gen_at_t0[(slug, side, t0)] = k
                t1 = g.get("t1")
                if t1 is None:
                    no_t1 += 1
                else:
                    span_of[k] = (t0, float(t1))

    # ---- THE PRE-FIX TEST, RECOMPUTED SO THE DELTA IS MEASURED ---------
    # This is the OLD expression, character for character in what it does:
    # walk the reference and ask whether a scored key sits at each
    # generation's t0. Keeping it is what turns "how wrong was it" from a
    # guess into a field.
    covered_pre_fix = sum(
        1 for slug in sorted(ref) for side in sides
        for g in ((ref[slug] or {}).get(side) or ())
        if (slug, side, float(g["t0"])) in gen_scores)

    # ---- STAGE 2: THE SCORES, WALKED ONCE ------------------------------
    covered: set = set()
    unnamed = 0
    n_rows_named = 0
    first_t: dict = {}
    for key, val in gen_scores.items():
        if not (isinstance(key, (tuple, list)) and len(key) == 3):
            raise CoverageRefused(
                f"SCORED_KEY_MALFORMED: a scored key is {key!r}, not a "
                f"(slug, side, t) triple. Nothing here can resolve it to a "
                f"generation.")
        slug, side, t = key[0], key[1], float(key[2])
        if per_row:
            gid = val.get("gen")
            k = None if gid is None else (slug, side, gid)
        else:
            k = gen_at_t0.get((slug, side, t))
        if k is None or k not in ref_gens:
            unnamed += 1
            continue
        n_rows_named += 1
        covered.add(k)
        if k not in first_t or t < first_t[k]:
            first_t[k] = t

    # ---- THE SECOND TRAVERSAL, so the two CAN disagree -----------------
    # `len(covered)` counts DISTINCT generations, reached from the scores.
    # This walk counts reference ENTRIES whose generation is covered. On a
    # well-formed reference they are the same number; they diverge exactly
    # when one (slug, side) carries the same `gen` twice, which is the one
    # way a set-based count and an entry-based count can differ. An
    # agreement that could not fail would be no agreement at all (rule 16).
    covered_entries = 0
    for slug in sorted(ref):
        by_side = ref[slug] or {}
        for side in sides:
            for g in (by_side.get(side) or ()):
                if (slug, side, g["gen"]) in covered:
                    covered_entries += 1

    late = sum(1 for k, t in first_t.items()
               if k in t0_of and float(t) != t0_of[k])

    return {
        "protocol": "BE_SCORE_COVERAGE_V1",
        "score_shape": shape,
        "score_shape_meaning": SHAPE_MEANING[shape],
        # ---- THE UNITS, BECAUSE THE UNITS ARE THE DEFECT ---------------
        "n_scored_keys": len(gen_scores),
        "n_scored_keys_unit": ("ROWS" if per_row else
                               "GENERATIONS" if shape == SHAPE_PER_GENERATION
                               else "NONE"),
        "n_scored_keys_is_commensurable_with_n_covered":
            shape == SHAPE_PER_GENERATION,
        "why_the_unit_is_published": (
            "under PER_ROW_SCORES `len(gen_scores)` counts ROWS while "
            "`n_covered` and `n_reference_generations` count GENERATIONS. "
            "The pre-fix blocks printed all three side by side and "
            "subtracted across them."),
        # ---- THE POPULATION --------------------------------------------
        "n_reference_generations": n_gen,
        "n_distinct_reference_generations": len(ref_gens),
        "n_covered": len(covered),
        # ---- REV 121's NUMBER, EMITTED (BE 116) ------------------------
        # `n_covered` counts generations SOME scored row names. This counts
        # the generations that have a scored key AT THEIR OWN t0, and the
        # gap between the two IS the exposure for every consumer that looks
        # a generation up at its start. It covers BOTH mechanisms at once --
        # a generation whose first scored row is LATER than its t0, and one
        # whose t0 row the feature pass dropped while a later row survived --
        # because it asks only whether the t0 key is there, not why it is
        # not. NO ARTIFACT ON DISK CAN ANSWER IT: every landed book predates
        # the causal scoring and is keyed at t0 BY CONSTRUCTION, so on those
        # the two numbers are equal by construction rather than by
        # measurement. The first corrected book is the artifact that settles
        # it, and this is the field it will settle it in.
        "n_generations_with_a_key_at_their_own_t0": covered_pre_fix,
        "n_generations_covered_WITHOUT_a_key_at_their_own_t0":
            len(covered) - covered_pre_fix,
        "what_that_gap_is": (
            "generations a consumer keying at `t0` CANNOT find although the "
            "book scored them. Two mechanisms, indistinguishable here and "
            "both counted: the first scored row is later than the "
            "generation's start, or the t0 row was dropped by the feature "
            "pass while a later one survived. On a PER_GENERATION book it is "
            "0 BY CONSTRUCTION -- the keys ARE the t0s -- so a zero from a "
            "landed book is not evidence"),
        "n_covered_reference_entries": covered_entries,
        "n_uncovered": n_gen - len(covered),
        "coverage": (len(covered) / n_gen) if n_gen else None,
        "n_scored_rows_naming_a_generation": n_rows_named,
        "rows_per_covered_generation": (round(n_rows_named / len(covered), 6)
                                        if covered else None),
        # ---- WHAT THE PRE-FIX TEST WOULD HAVE SAID ---------------------
        "pre_fix": {
            "test": PRE_FIX_TEST,
            "SAME_NUMBER_AS": "n_generations_with_a_key_at_their_own_t0 -- "
                              "one computation, published twice because it "
                              "answers two questions: what the old test "
                              "measured, and what a t0-keyed consumer can "
                              "still find",
            "n_covered": covered_pre_fix,
            "understated_coverage_by": len(covered) - covered_pre_fix,
            "what_it_measured": (
                "under PER_GENERATION_SCORES it is the same question and the "
                "two agree by construction. Under PER_ROW_SCORES it admits a "
                "generation only if a scored row sits exactly at its start, "
                "so the difference IS the population the old block dropped."),
        },
        # ---- EXCLUSIONS, EACH A STATUS WITH A COUNT (rule 4) -----------
        "exclusions": {
            "GENERATION_NOT_SCORED": len(ref_gens) - len(covered),
            "GENERATION_NOT_SCORED_meaning":
                "a reference generation NO scored key names -- the assembly "
                "produced nothing for it",
            "FIRST_SCORED_ROW_NOT_AT_GENERATION_START": late,
            "FIRST_SCORED_ROW_NOT_AT_GENERATION_START_meaning":
                "a covered generation whose earliest scored row is not at "
                "its `t0`. The PRE-FIX test dropped every one of these; they "
                "are covered here and counted so the change is visible",
            "SCORED_KEY_NAMES_NO_REFERENCE_GENERATION": unnamed,
            "SCORED_KEY_NAMES_NO_REFERENCE_GENERATION_meaning":
                "a scored key whose generation is not in the reference at "
                "all -- under PER_ROW its `gen`, under PER_GENERATION its "
                "key sitting at no generation's t0",
            "DUPLICATE_GENERATION_ID": dup_gen_id,
            "DUPLICATE_GENERATION_ID_meaning":
                "one (slug, side) carrying the same `gen` twice; it makes "
                "the set-based and entry-based covered counts disagree",
            "TWO_GENERATIONS_SHARE_A_T0": shared_t0,
            "TWO_GENERATIONS_SHARE_A_T0_meaning":
                "two generations of one (slug, side) starting at the same "
                "instant; the PRE-FIX test cannot tell them apart",
        },
        # ---- Q-DA-361's NECESSARY CONDITION, COUNTED HERE --------------
        "key_collision_hazard": _collision_hazard(span_of, no_t1, per_row,
                                                  gen_scores),
        "decides_nothing": "REPORTED. `assert_coverage` adjudicates (rule 14).",
    }


def _collision_hazard(span_of: dict, no_t1: int, per_row: bool,
                      gen_scores: dict) -> dict:
    """DA's Q-DA-361 necessary condition, computed from the reference.

    Under PER_ROW_SCORES the key is `(slug, side, t)` and does NOT name the
    generation, so two generations of one (slug, side) holding a row at the
    same `t` overwrite -- last writer wins, and since the reference is
    iterated in generation order the EARLIER generation loses its row with
    every status reporting health. The defect is DE's
    (`de_phase4_diag_runner.generation_scores`) and the repair is DE's.

    What is computable HERE, from what the book already carries, is the
    NECESSARY condition: a generation of one (slug, side) that starts before
    an earlier generation of the same (slug, side) has ended. If that count
    is zero, no key time can belong to two generations and the hazard is
    absent for this book -- which turns "reachability NOT established" into
    a number the receipt carries. A non-zero count is not the defect firing;
    it is the door being open."""
    out = {
        "why": "Q-DA-361 (DA 138): the PER_ROW key (slug, side, t) does not "
               "name the generation, so two generations of one (slug, side) "
               "with a row at the same t overwrite, earlier generation "
               "losing, with every status reporting health.",
        "whose_defect": "DE -- de_phase4_diag_runner.generation_scores. This "
                        "is a WITNESS computed on BE's surface, never a fix.",
        "applies_to_this_shape": per_row,
    }
    if no_t1:
        out["status"] = "NOT_COMPUTABLE_GENERATION_WITHOUT_T1"
        out["n_generations_without_t1"] = no_t1
        out["n_generations_starting_inside_an_earlier_one"] = None
        out["why_not_computable"] = (
            "the condition is an interval overlap and these generations "
            "carry no end. Reported as NOT COMPUTABLE, never as zero -- an "
            "absent measurement must not read as a clean one (rule 11).")
        return out
    by_ss: dict = {}
    for (slug, side, _g), span in span_of.items():
        by_ss.setdefault((slug, side), []).append(span)
    overlapping = 0
    overlapping_ss = set()
    for ss, spans in by_ss.items():
        spans.sort()
        end = None
        for a, b in spans:
            if end is not None and a < end - 1e-9:
                overlapping += 1
                overlapping_ss.add(ss)
            end = b if end is None else max(end, b)
    keys_in_overlapping = 0
    if overlapping_ss:
        keys_in_overlapping = sum(
            1 for k in gen_scores if (k[0], k[1]) in overlapping_ss)
    out["status"] = "COMPUTED"
    out["n_generations_starting_inside_an_earlier_one"] = overlapping
    out["n_slug_side_pairs_with_an_overlap"] = len(overlapping_ss)
    out["n_scored_keys_under_an_overlapping_pair"] = keys_in_overlapping
    out["reading"] = (
        "ZERO: no two generations of one (slug, side) overlap, so no key "
        "time can belong to two and the collision cannot occur in this book"
        if overlapping == 0 else
        f"NON-ZERO: {overlapping} generation(s) start before an earlier "
        f"generation of the same (slug, side) has ended, so the collision "
        f"is POSSIBLE here. Whether it FIRED needs the row-level t_start "
        f"population, which no artifact carries (DA 138).")
    return out


# ---------------------------------------------------------------------------
# THE FALSIFIER
# ---------------------------------------------------------------------------

def _fixture(per_row: bool) -> tuple:
    """ONE synthetic assembly with the properties the real books lack.

    Built to have exactly what the pre-fix test mishandles:

      s1/BUY_UP gen 0  t0=100.0  rows at 100.0, 103.0, 107.0
                       -- three ROWS for ONE generation, first AT the start
      s1/BUY_UP gen 1  t0=200.0  rows at 203.0, 209.0
                       -- FIRST ROW NOT AT THE START: the pre-fix test looks
                          at 200.0, finds nothing, and drops the generation
      s1/SELL_UP gen 2 t0=300.0  rows at 305.0
                       -- the same, on the other side
      s2/BUY_UP gen 3  t0=400.0  NO ROWS AT ALL
                       -- GENERATION_NOT_SCORED under either test

    So: 4 reference generations, 6 scored rows, 3 truly covered, and the
    pre-fix test finds ONE. The per-generation twin keys the same four
    generations at their own t0 with bare floats, which is what every book
    on disk holds."""
    ref = {
        "s1": {"BUY_UP": [{"gen": 0, "t0": 100.0, "t1": 150.0, "tranches": []},
                          {"gen": 1, "t0": 200.0, "t1": 250.0,
                           "tranches": [{"t": 210.0}]}],
               "SELL_UP": [{"gen": 2, "t0": 300.0, "t1": 350.0,
                            "tranches": []}]},
        "s2": {"BUY_UP": [{"gen": 3, "t0": 400.0, "t1": 450.0,
                           "tranches": []}],
               "SELL_UP": []},
    }
    if per_row:
        gs = {
            ("s1", "BUY_UP", 100.0): {"score": 0.11, "gen": 0, "t0": 100.0},
            ("s1", "BUY_UP", 103.0): {"score": 0.12, "gen": 0, "t0": 100.0},
            ("s1", "BUY_UP", 107.0): {"score": 0.13, "gen": 0, "t0": 100.0},
            ("s1", "BUY_UP", 203.0): {"score": 0.21, "gen": 1, "t0": 200.0},
            ("s1", "BUY_UP", 209.0): {"score": 0.22, "gen": 1, "t0": 200.0},
            ("s1", "SELL_UP", 305.0): {"score": 0.31, "gen": 2, "t0": 300.0},
        }
    else:
        gs = {("s1", "BUY_UP", 100.0): 0.13,
              ("s1", "BUY_UP", 200.0): 0.22,
              ("s1", "SELL_UP", 300.0): 0.31}
    return ref, gs


def _pre_fix_block(ref, gs, sides, n_gen):
    """THE PRE-FIX BLOCK, reproduced exactly as the two sites computed it.

    `be_daybook_build.py:821-826` and `be_generation_count_derivation.py:
    72-79` at `6377f20`. It is here so the known-bad drives the ACTUAL old
    code rather than a description of it."""
    scored = sum(1 for s in sorted(ref) for side in sides
                 for g in ((ref[s] or {}).get(side) or ())
                 if (s, side, float(g["t0"])) in gs)
    return {"n_scored_keys": len(gs), "n_reference_generations": n_gen,
            "n_covered": scored, "n_uncovered": n_gen - scored,
            "coverage": scored / n_gen if n_gen else None}


EXPECTED_CHECKS = 28


def falsify() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    SIDES = ("BUY_UP", "SELL_UP")

    # ---- 1. THE KNOWN-BAD: the PRE-FIX block on a PER_ROW assembly ------
    ref, gs_row = _fixture(per_row=True)
    old = _pre_fix_block(ref, gs_row, SIDES, 4)
    new = generation_coverage(ref, gs_row, sides=SIDES)
    ok(old["n_covered"] == 1 and old["n_uncovered"] == 3,
       f"KNOWN-BAD (the defect, DRIVEN): the PRE-FIX membership test covers "
       f"{old['n_covered']} of 4 generations and reports "
       f"{old['n_uncovered']} UNCOVERED, because gens 1 and 2 have no scored "
       f"row at their own t0")
    ok(new["n_covered"] == 3 and new["n_uncovered"] == 1,
       f"FIXED: the corrected test covers {new['n_covered']} of 4 -- every "
       f"generation some scored row NAMES -- and reports "
       f"{new['n_uncovered']} uncovered, which is gen 3, the one with no "
       f"rows at all")
    ok(old["n_scored_keys"] == 6 and old["n_reference_generations"] == 4
       and old["n_uncovered"] == 3,
       "KNOWN-BAD (the units, DRIVEN): the pre-fix block puts n_scored_keys="
       "6 (ROWS) beside n_reference_generations=4 (GENERATIONS) and "
       "subtracts across them -- three numbers, two units, no field saying so")
    ok(new["n_scored_keys"] == 6 and new["n_scored_keys_unit"] == "ROWS"
       and new["n_scored_keys_is_commensurable_with_n_covered"] is False,
       "FIXED: the same 6 is published as ROWS and explicitly marked NOT "
       "commensurable with the covered count")
    ok(new["n_generations_with_a_key_at_their_own_t0"] == 1
       and new["n_generations_covered_WITHOUT_a_key_at_their_own_t0"] == 2
       and new["n_generations_with_a_key_at_their_own_t0"]
       == new["pre_fix"]["n_covered"],
       f"REV 121's NUMBER, EMITTED: "
       f"{new['n_generations_with_a_key_at_their_own_t0']} of the fixture's "
       f"3 covered generations have a scored key AT THEIR OWN t0, so "
       f"{new['n_generations_covered_WITHOUT_a_key_at_their_own_t0']} are "
       f"covered and INVISIBLE to a consumer keying at t0 -- the exposure, "
       f"as a field. It is the same computation as `pre_fix.n_covered` and "
       f"the two are asserted equal so they can never drift apart")
    ok(new["pre_fix"]["n_covered"] == 1
       and new["pre_fix"]["understated_coverage_by"] == 2,
       f"AND THE DELTA IS A FIELD, not an inference: the corrected block "
       f"carries the pre-fix answer ({new['pre_fix']['n_covered']}) and the "
       f"{new['pre_fix']['understated_coverage_by']} generations it "
       f"understated by")
    ok(new["exclusions"]["FIRST_SCORED_ROW_NOT_AT_GENERATION_START"] == 2
       and new["exclusions"]["GENERATION_NOT_SCORED"] == 1
       and new["exclusions"]["SCORED_KEY_NAMES_NO_REFERENCE_GENERATION"] == 0,
       f"and the exclusions are STATUSES WITH COUNTS: 2 generations whose "
       f"first scored row is not at their start, 1 scored by nothing, 0 keys "
       f"naming no generation (rule 4)")
    ok(new["score_shape"] == SHAPE_PER_ROW
       and new["rows_per_covered_generation"] == 2.0,
       f"shape PER_ROW_SCORES, {new['rows_per_covered_generation']} rows per "
       f"covered generation -- the ratio that makes the row/generation "
       f"confusion visible at a glance")

    # ---- 2. THE POSITIVE CONTROL THAT MUST ADMIT (rule 16) -------------
    ref2, gs_gen = _fixture(per_row=False)
    old2 = _pre_fix_block(ref2, gs_gen, SIDES, 4)
    new2 = generation_coverage(ref2, gs_gen, sides=SIDES)
    ok(new2["score_shape"] == SHAPE_PER_GENERATION
       and new2["n_covered"] == old2["n_covered"] == 3
       and new2["n_uncovered"] == old2["n_uncovered"] == 1
       and new2["coverage"] == old2["coverage"],
       f"POSITIVE CONTROL: on a PER_GENERATION assembly the corrected test "
       f"and the pre-fix test give the SAME answer ({new2['n_covered']} "
       f"covered, {new2['n_uncovered']} uncovered). The fix changes nothing "
       f"about any book on disk -- which is what makes it a fix and not a "
       f"re-specification")
    ok(new2["n_generations_with_a_key_at_their_own_t0"] == new2["n_covered"]
       and new2["n_generations_covered_WITHOUT_a_key_at_their_own_t0"] == 0,
       f"and on a PER_GENERATION assembly the gap is 0 BY CONSTRUCTION "
       f"({new2['n_generations_with_a_key_at_their_own_t0']} == "
       f"{new2['n_covered']}) -- which is exactly why no book on disk can "
       f"answer REV 121's question and the first corrected one must")
    ok(new2["n_scored_keys_unit"] == "GENERATIONS"
       and new2["n_scored_keys_is_commensurable_with_n_covered"] is True
       and new2["pre_fix"]["understated_coverage_by"] == 0,
       "and on that shape the key count IS commensurable and the pre-fix "
       "delta is 0 -- the block says so rather than leaving it read off")
    ok(new2["exclusions"]["FIRST_SCORED_ROW_NOT_AT_GENERATION_START"] == 0,
       "with no late-first-row exclusions, because under PER_GENERATION "
       "every key IS at a generation's start by construction")

    # ---- 3. STRUCTURAL REFUSALS, EACH BY NAME --------------------------
    def refuses(fn, needle, label):
        nonlocal checks
        checks += 1
        try:
            fn()
            print("FAIL: " + label + " (did not refuse)")
            fails.append(label)
            return
        except CoverageRefused as e:
            hit = needle in str(e)
            print(("PASS: " if hit else "FAIL: ") + label)
            if not hit:
                fails.append(f"{label} (wrong name: {e})")

    refuses(lambda: score_shape([1, 2]), "SCORES_NOT_A_MAP",
            "KNOWN-BAD: a non-dict score map REFUSES by name")
    refuses(lambda: score_shape({("a", "b", 1.0): 0.5,
                                 ("a", "b", 2.0): {"gen": 1}}),
            "MIXED_SCORE_SHAPES",
            "KNOWN-BAD: a map half per-row and half per-generation REFUSES "
            "-- it names its generations two ways")
    refuses(lambda: generation_coverage(ref, gs_row, sides=()), "NO_SIDES",
            "KNOWN-BAD: an empty side list REFUSES rather than reporting "
            "perfect coverage of nothing")
    refuses(lambda: generation_coverage(
        {"s1": {"BUY_UP": [{"gen": 0}]}}, gs_row, sides=SIDES),
        "GENERATION_WITHOUT_T0",
        "KNOWN-BAD: a generation with no `t0` REFUSES by name")
    refuses(lambda: generation_coverage(
        {"s1": {"BUY_UP": [{"t0": 1.0}]}}, gs_row, sides=SIDES),
        "GENERATION_WITHOUT_ID",
        "KNOWN-BAD: a generation with no `gen` REFUSES by name -- under "
        "PER_ROW there would be nothing to match the score's `gen` to")
    refuses(lambda: generation_coverage(ref, {("s1", "BUY_UP"): 0.1},
                                        sides=SIDES),
            "SCORED_KEY_MALFORMED",
            "KNOWN-BAD: a scored key that is not a (slug, side, t) triple "
            "REFUSES by name")
    ok(score_shape({}) == SHAPE_EMPTY,
       "and an EMPTY map is its own answer, EMPTY_SCORE_MAP -- not "
       "PER_GENERATION, which is what `any(...)` over nothing would publish")

    # ---- 4. THE POPULATION FAULTS ARE COUNTED, NOT REFUSED ------------
    dup = {"s1": {"BUY_UP": [{"gen": 0, "t0": 100.0, "t1": 150.0},
                             {"gen": 0, "t0": 120.0, "t1": 160.0}],
                  "SELL_UP": []}}
    dcov = generation_coverage(dup, {("s1", "BUY_UP", 100.0): 0.5},
                               sides=SIDES)
    ok(dcov["exclusions"]["DUPLICATE_GENERATION_ID"] == 1
       and dcov["n_covered"] == 1 and dcov["n_covered_reference_entries"] == 2,
       f"a DUPLICATE generation id is COUNTED (1) and it makes the two "
       f"covered counts disagree -- {dcov['n_covered']} distinct against "
       f"{dcov['n_covered_reference_entries']} reference entries. The "
       f"agreement check has something to catch, which is why it is not an "
       f"agreement that cannot fail")
    shared = {"s1": {"BUY_UP": [{"gen": 0, "t0": 100.0, "t1": 150.0},
                                {"gen": 1, "t0": 100.0, "t1": 160.0}],
                     "SELL_UP": []}}
    scov = generation_coverage(shared, {("s1", "BUY_UP", 100.0): 0.5},
                               sides=SIDES)
    ok(scov["exclusions"]["TWO_GENERATIONS_SHARE_A_T0"] == 1,
       "two generations sharing a t0 are COUNTED -- the pre-fix test could "
       "not tell them apart and reported one of them covered")
    ucov = generation_coverage(
        ref, {**gs_row, ("s9", "BUY_UP", 1.0): {"score": 0.1, "gen": 77,
                                                "t0": 1.0}}, sides=SIDES)
    ok(ucov["exclusions"]["SCORED_KEY_NAMES_NO_REFERENCE_GENERATION"] == 1,
       "a scored key naming a generation the reference does not have is "
       "COUNTED, never silently dropped from the denominator")

    # ---- 5. Q-DA-361's NECESSARY CONDITION, BOTH DIRECTIONS -----------
    ok(new["key_collision_hazard"]["status"] == "COMPUTED"
       and new["key_collision_hazard"][
           "n_generations_starting_inside_an_earlier_one"] == 0,
       "Q-DA-361 WITNESS, clean case: no generation of one (slug, side) "
       "starts inside an earlier one, so no key time can belong to two")
    lap = {"s1": {"BUY_UP": [{"gen": 0, "t0": 100.0, "t1": 250.0},
                             {"gen": 1, "t0": 200.0, "t1": 300.0}],
                  "SELL_UP": []}}
    lcov = generation_coverage(
        lap, {("s1", "BUY_UP", 210.0): {"score": 0.1, "gen": 1, "t0": 200.0}},
        sides=SIDES)
    ok(lcov["key_collision_hazard"][
           "n_generations_starting_inside_an_earlier_one"] == 1
       and lcov["key_collision_hazard"][
           "n_scored_keys_under_an_overlapping_pair"] == 1
       and "NON-ZERO" in lcov["key_collision_hazard"]["reading"],
       "POSITIVE CONTROL for the witness: an overlapping pair is FOUND and "
       "the keys under it counted -- the statistic can fire, so its zero "
       "above is a reading and not an instrument that never proved it works")
    not1 = {"s1": {"BUY_UP": [{"gen": 0, "t0": 100.0}], "SELL_UP": []}}
    ncov = generation_coverage(not1, {("s1", "BUY_UP", 100.0): 0.5},
                               sides=SIDES)
    ok(ncov["key_collision_hazard"]["status"]
       == "NOT_COMPUTABLE_GENERATION_WITHOUT_T1"
       and ncov["key_collision_hazard"][
           "n_generations_starting_inside_an_earlier_one"] is None,
       "and a generation with no end makes the witness NOT COMPUTABLE and "
       "None -- never zero, which would read as a clean measurement (rule 11)")

    # ---- 6. THE SEAM: this module against the PINNED implementation ---
    # `be_cancel_axis_null.load()` keeps its own shape handling because it
    # is the pinned cascade entry point and converting it would re-pin
    # params AND design. Two implementations that never meet drift; these
    # two meet here, over ONE synthetic per-row BOOK on disk.
    import hashlib
    import pickle
    import tempfile
    seam = {"ran": False}
    try:
        import be_cancel_axis_null as CAN
        bref = {s: {sd: list(v) for sd, v in by.items()}
                for s, by in ref.items()}
        book = {"fr": {"reference": bref},
                "asm": {"by_arm": {("btc", "q1_arrival_composed_lgbm"):
                                   (gs_row, {})}}}
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "per_row_fixture_book.pkl"
            fp.write_bytes(pickle.dumps(book, protocol=pickle.HIGHEST_PROTOCOL))
            bk = CAN.load(fp)
        ex = bk["exclusions"]
        mine = generation_coverage(bref, gs_row, sides=SIDES)
        seam["ran"] = True
        ok(ex["score_shape"] == mine["score_shape"] == SHAPE_PER_ROW,
           f"SEAM: `be_cancel_axis_null.load()` and this module read the "
           f"SAME synthetic per-row book and agree on the shape "
           f"({ex['score_shape']})")
        ok(ex["FIRST_SCORED_ROW_NOT_AT_GENERATION_START"]
           == mine["exclusions"]["FIRST_SCORED_ROW_NOT_AT_GENERATION_START"]
           == 2
           and ex["GENERATION_NOT_SCORED"]
           == mine["exclusions"]["GENERATION_NOT_SCORED"] == 1
           and ex["n_scored_generations"] == mine["n_covered"] == 3
           and ex["n_reference_generations"]
           == mine["n_distinct_reference_generations"] == 4,
           f"and on every count they both compute: late-first-row "
           f"{ex['FIRST_SCORED_ROW_NOT_AT_GENERATION_START']}, unscored "
           f"{ex['GENERATION_NOT_SCORED']}, covered "
           f"{ex['n_scored_generations']}, reference "
           f"{ex['n_reference_generations']}. Two implementations, one "
           f"answer, DRIVEN -- a divergence fails this battery")
        ok(ex["n_rows"] == mine["n_scored_rows_naming_a_generation"] == 6,
           f"including the ROW count the null draws over "
           f"({ex['n_rows']}), which is the number the pre-fix block "
           f"published as though it were a generation count")
    except Exception as e:                                   # noqa: BLE001
        for _ in range(3):
            ok(False, f"SEAM cell could not run: {type(e).__name__}: {e}")

    print()
    # THE SUMMARY LINE IS `<n> cells, <k> failures` because every importer
    # runs this module through `be_rule22.shared_falsifier`, which reads
    # exactly that shape (REV 84 3.2). A module whose summary the reader
    # cannot parse reports ok=False, which is the safe direction, but it
    # would be a cell that can only fail.
    if fails:
        print(f"{checks} cells, {len(fails)} failures")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} cells, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        print(f"{checks} cells, 1 failures")
        return 1
    print(f"{checks} cells, 0 failures")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--falsify" in argv or "--selftest" in argv:
        return falsify()
    if "--fixture-json" in argv:
        ref, gs = _fixture(per_row="--per-generation" not in argv)
        print(json.dumps(generation_coverage(
            ref, gs, sides=("BUY_UP", "SELL_UP")), indent=1, default=str))
        return 0
    print("usage: be_score_coverage.py --falsify | --fixture-json "
          "[--per-generation]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
