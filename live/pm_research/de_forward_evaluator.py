"""THE FORWARD TEST'S EVALUATOR: per-day reporting, and the three numbers
that tell a reader whether the verdict could ever have been a pass.

Rehearsed on CONSUMED days (09-03..09-06) before the forward books land,
because a failure discovered on a forward book is not free: a run that
partially values a protected day has exposed it.

WHAT IS NEW HERE AND WHY EACH EXISTS
------------------------------------
`G` IS TAKEN FROM THE DATA, NEVER FROM A CONSTANT. The aggregator this
supersedes carries `PRIMARY = ("2026-09-04","2026-09-05","2026-09-06")` as
a MODULE LITERAL -- a 3-day constant from the development era. At N=7 that
literal would silently value a short population and report a verdict for
days nobody asked about. Here `days` is a REQUIRED argument and `G` is
`len(days)`; there is no default to fall back to.

ATTAINABLE MINIMUM p and TOLERANCE, computed per result, pass or fail:

    "A test that cannot attain its own threshold has measured nothing
     about the arms -- it has measured the calendar."

The minimum p says whether a pass was POSSIBLE at the G actually achieved.
The tolerance says HOW CLOSE TO PERFECTION it demanded -- how many negative
days the test could have survived. A reader cannot tell an INCONCLUSIVE
from a demanding test apart from one from an impossible test without both.

FUTILITY. At G=7 conjunct (a) requires a unanimous run: the moment ANY day
comes back negative for an arm, that arm CANNOT pass, whatever the
remaining days do. Stopping then costs NOTHING statistically -- it can only
ever reduce the chance of declaring success, never inflate it. Unlike an
early SUCCESS call, early futility is free, so it is computed after every
day and reported.

BOTH COMPONENTS ARE CARRIED, because the design has two conjuncts with
different floors and they are not interchangeable:
  (a) the day-sign component over G day-clusters      -> floor sided/2**G
  (b) the matched-control draw permutation, pooled    -> floor 1/(1+n)
(b) does not depend on G at all; (a) depends on nothing else. Reporting one
as if it were the test's floor understates or overstates it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb
from pathlib import Path
from time import time_ns

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import de_settlement_control_aggregate as AGG      # noqa: E402

PROTOCOL = "P003_DE_FORWARD_EVALUATOR_V1"
ARMS = AGG.ARMS
ALPHA = AGG.ALPHA
M_FAMILY = 2          # Holm: one p per ARM. 69 is selection multiplicity.
SIDED = 2             # two-sided; the direction was chosen after seeing

NO_DAYS = "FORWARD_EVALUATOR_CALLED_WITHOUT_AN_EXPLICIT_DAY_SET"
DUP_DAY = "FORWARD_EVALUATOR_DAY_SET_REPEATS_A_DAY"
MISSING = "FORWARD_EVALUATOR_BOOK_MISSING_FOR_A_DECLARED_DAY"
SHORT = "FORWARD_EVALUATOR_POPULATION_SHORTER_THAN_DECLARED"
NO_ROBUSTNESS = "FORWARD_EVALUATOR_DECLARED_ROBUSTNESS_LEG_MISSING"
OUTPUT_EXISTS = "FORWARD_EVALUATOR_OUTPUT_ALREADY_EXISTS"
BAD_METADATA = "FORWARD_EVALUATOR_METADATA_MALFORMED"
MISSING_METADATA = "FORWARD_EVALUATOR_REQUIRED_METADATA_MISSING"
CELL_COHORT = "FORWARD_ARM_CELLS_DO_NOT_SHARE_ONE_DAY_INPUT_COHORT"

VERDICT_BOTH = "FORWARD_SUPPORT_FOR_BOTH_ARMS"
VERDICT_ONE = "FORWARD_SUPPORT_FOR_THAT_ARM_ONLY"
VERDICT_NONE = "NO_FORWARD_SUPPORT"
VERDICT_SPLIT = "NOT_ONE_MECHANISM"


class EvaluatorRefused(RuntimeError):
    """A named refusal."""


# ---------------------------------------------------------------- floors --
def day_sign_p(n_pos: int, G: int, sided: int = SIDED) -> float:
    """P[>= n_pos of G positive] under the day-cluster sign-flip null."""
    one = sum(comb(G, j) for j in range(n_pos, G + 1)) / 2 ** G
    return min(1.0, sided * one)


def attainable_min_p(G: int, n_draws: int, sided: int = SIDED) -> dict:
    """The most extreme p EACH component can attain. Computed, not quoted."""
    return {
        "day_sign_component": day_sign_p(G, G, sided),
        "day_sign_holm_adjusted": min(1.0, day_sign_p(G, G, sided) * M_FAMILY),
        "draw_permutation_component": 1.0 / (1 + n_draws),
        "draw_permutation_holm_adjusted":
            min(1.0, M_FAMILY / (1 + n_draws)),
        "G": G, "n_draws": n_draws, "sided": sided,
        "note": "the draw component does not depend on G; the day-sign "
                "component depends on nothing else",
    }


def tolerance(G: int, threshold: float, sided: int = SIDED) -> int:
    """How many NEGATIVE days the day-sign test could survive and clear."""
    t = -1
    for neg in range(G + 1):
        if day_sign_p(G - neg, G, sided) <= threshold:
            t = neg
        else:
            break
    return t


def floor_block(G: int, n_draws: int, sided: int = SIDED) -> dict:
    """What must ride beside EVERY verdict, pass or fail."""
    thr = ALPHA / M_FAMILY
    mn = attainable_min_p(G, n_draws, sided)
    tol = tolerance(G, thr, sided)
    return {
        "attainable_minimum_p": mn,
        "holm_threshold_for_the_smaller_p": thr,
        "a_pass_was_possible_at_this_G": mn["day_sign_component"] <= thr,
        "tolerance_negative_days": tol,
        "requires_unanimity": tol == 0,
        "the_sentence": "A test that cannot attain its own threshold has "
                        "measured nothing about the arms -- it has measured "
                        "the calendar.",
        "read_this_as": (
            "the minimum p says whether a pass was POSSIBLE; the tolerance "
            "says how close to perfection it demanded"),
    }


def _sign(value: float) -> int:
    value = float(value)
    return 1 if value > 0 else -1 if value < 0 else 0


def robustness_pool(cells: dict, days, arm: str,
                    primary_D: float) -> dict:
    """Pool the declared fill-assumption leg; absence is not a zero."""
    per_day = {}
    for day in days:
        robust = (cells[(day, arm)].get("result") or {}).get(
            "robustness_leg")
        if (not isinstance(robust, dict)
                or robust.get("label") != "NO_FILLS_UNTIL_NEXT_GENERATION"
                or robust.get("observed_D_cents") is None):
            raise EvaluatorRefused(
                f"REFUSED {NO_ROBUSTNESS}: {day}/{arm} does not carry the "
                f"labelled NO_FILLS_UNTIL_NEXT_GENERATION delta. The leg is "
                f"always reported and may never be substituted with zero.")
        per_day[day] = float(robust["observed_D_cents"])
    pooled_D = sum(per_day.values())
    return {
        "label": "NO_FILLS_UNTIL_NEXT_GENERATION",
        "pooling": "SUM_OF_PER_DAY_CASH_DELTAS",
        "observed_D_cents": pooled_D,
        "per_day_D_cents": per_day,
        "primary_label": "REFERENCE_FILLS",
        "primary_observed_D_cents": primary_D,
        "sign_reversal": _sign(primary_D) != _sign(pooled_D),
        "a_sign_reversal_blocks_promotion": True,
    }


def verdict_for_forward(arms: dict) -> dict:
    advancing = [name for name, row in arms.items() if row["advances"]]
    pooled_passing = [name for name, row in arms.items()
                      if row["pooled_advances_before_robustness"]]
    signs = {_sign(arms[name]["pooled"]["D_arm_cents"])
             for name in pooled_passing}
    if len(pooled_passing) == 2 and len(signs) > 1:
        verdict = VERDICT_SPLIT
        why = "two opposite significant effects are not one mechanism"
        advancing = []
    elif len(advancing) == 2:
        verdict = VERDICT_BOTH
        why = "both arms clear both comparisons and the robustness block"
    elif len(advancing) == 1:
        verdict = VERDICT_ONE
        why = "one arm clears; this is not a programme-level pass"
    else:
        verdict = VERDICT_NONE
        why = "neither arm clears both comparisons and the robustness block"
    return {"verdict": verdict, "why": why,
            "n_arms_advancing": len(advancing),
            "arms_advancing": advancing,
            "triggers_no_extension_rule": not advancing}


# --------------------------------------------------------------- futility --
def futility(per_day_D: dict, G_declared: int, sided: int = SIDED) -> dict:
    """Is the test ALREADY DEAD for this arm? Computed after every day.

    Free to act on: it can only ever reduce the chance of declaring
    success, so unlike an early SUCCESS call it cannot inflate anything.
    """
    seen = list(per_day_D.items())
    if len(seen) > G_declared:
        raise EvaluatorRefused(
            f"REFUSED {SHORT}: {len(seen)} days scored against a declared "
            f"G={G_declared}. Futility is evaluated against the DECLARED "
            f"population; more days than declared means the population "
            f"moved and the floor with it.")
    neg = [d for d, v in seen if v <= 0]
    thr = ALPHA / M_FAMILY
    best_possible = day_sign_p(G_declared - len(neg), G_declared, sided)
    dead = best_possible > thr
    # TWO DIFFERENT DEATHS, and a reader must not confuse them: a test the
    # ARMS killed by printing a negative day, and a test the CALENDAR killed
    # because G was never large enough to attain the threshold.
    impossible = day_sign_p(G_declared, G_declared, sided) > thr
    cause = ("TEST_IMPOSSIBLE_AT_THIS_G" if impossible else
             "KILLED_BY_NEGATIVE_DAYS" if dead else "ALIVE")
    return {
        "cause": cause,
        "cause_means": {
            "TEST_IMPOSSIBLE_AT_THIS_G":
                "even a UNANIMOUS run at this G cannot attain the "
                "threshold. This says nothing about the arms -- it is the "
                "calendar. NEVER read it as evidence against them.",
            "KILLED_BY_NEGATIVE_DAYS":
                "the test COULD have passed at this G; the arm's own "
                "negative day(s) ended it. This IS about the arms.",
            "ALIVE": "still attainable",
        }[cause],
        "days_scored": len(seen),
        "days_declared": G_declared,
        "negative_or_zero_days": neg,
        "n_negative": len(neg),
        "best_attainable_p_given_what_is_already_seen": best_possible,
        "threshold": thr,
        "FUTILE": dead,
        "why": (
            f"{len(neg)} of {G_declared} day(s) already non-positive; even a "
            f"perfect run on the {G_declared - len(seen)} remaining day(s) "
            f"attains at best p={best_possible:.6f} > {thr:.4f}"
            if dead else
            f"{len(neg)} non-positive day(s); a unanimous remainder still "
            f"attains p={best_possible:.6f} <= {thr:.4f}"),
        "stopping_is_free": (
            "futility stopping only ever reduces the chance of declaring "
            "success; it cannot inflate a positive result"),
    }


# ------------------------------------------------------------ resolution --
def resolve_books(root: Path, days, revision: str, coin: str = "btc") -> dict:
    """Map day -> book path for a revision. REFUSES on ANY missing day.

    A short population is the failure mode this exists to prevent: valuing
    5 of 7 days and reporting it as the test is how a G-dependent floor
    gets quoted against the wrong G.
    """
    days = list(days)
    if not days:
        raise EvaluatorRefused(
            f"REFUSED {NO_DAYS}: the day set is the population and must be "
            f"passed explicitly. There is no default.")
    if len(set(days)) != len(days):
        raise EvaluatorRefused(f"REFUSED {DUP_DAY}: {days}")
    suffix = f"__{revision}" if revision else ""
    out, missing = {}, []
    for d in days:
        p = Path(root) / f"be_daybook_{d.replace('-', '')}_{coin}{suffix}.pkl"
        (out.__setitem__(d, p) if p.exists() else missing.append(d))
    if missing:
        raise EvaluatorRefused(
            f"REFUSED {MISSING}: revision {revision!r} has no book for "
            f"{missing}. Declared G={len(days)}, resolved {len(out)}. A "
            f"short population is NOT a smaller test -- it is a test "
            f"against a different floor.")
    return out


# ------------------------------------------------------------- reporting --
def standing_warning(G: int, sided: int = SIDED) -> str:
    """Said UNPROMPTED on every emit, so it is never something someone had
    to remember. At G=7 tolerance is 0: the FIRST negative day ends an arm."""
    tol = tolerance(G, ALPHA / M_FAMILY, sided)
    if tol < 0:
        return (f"| G={G} A PASS IS NOT ATTAINABLE AT THIS G "
                f"(min p={day_sign_p(G, G, sided):.4f} > "
                f"{ALPHA / M_FAMILY:.3f}) -- THIS MEASURES THE CALENDAR")
    if tol == 0:
        return (f"| G={G} tol=0 UNANIMITY REQUIRED -- the first negative "
                f"day ends that arm")
    return f"| G={G} tol={tol} negative day(s) survivable"


def per_day_line(cell: dict, pool: dict, G: int = None,
                 sided: int = SIDED) -> str:
    """The line the user reads, one per (day, arm), carrying the warning."""
    head = (f"DONE {cell['day']} {cell['arm']} "
            f"D={cell['D']:+.2f} p={pool['p_two_sided']:.6f} "
            f"n={cell['n']} base={cell['baseline_total_cents']:+.2f} "
            f"arm={cell['arm_total_cents']:+.2f}")
    return head if G is None else f"{head} {standing_warning(G, sided)}"


# --------------------------------------------- settlement-source finality --
NO_RECEIPT = "FORWARD_EVALUATOR_NO_VERIFIED_WINNER_RECEIPT_FOR_A_DAY"


def _winner_source_blocks(value):
    if isinstance(value, dict):
        if (isinstance(value.get("chainlink_verification"), dict)
                and isinstance(value.get("sha256"), str)):
            yield value
        for child in value.values():
            yield from _winner_source_blocks(child)
    elif isinstance(value, list):
        for child in value:
            yield from _winner_source_blocks(child)


def settlement_source_disclosure(days, derived: Path,
                                 revision: str = "L250ms",
                                 winner_sources=None) -> dict:
    """R-810's finality, carried on every published cent figure.

    Step 2 called `winner_source` WITHOUT `verification=`, so every cent it
    published is `VENUE_RECORD_NOT_VERIFIED_AGAINST_CHAINLINK` and
    `is_final_for_quotation` is False. This reads the per-day verification
    the day pipeline already computed and carries it.

    WHY A DAY-LEVEL READ IS SUFFICIENT FOR A CELL-LEVEL CLAIM: a cell's
    fills name a SUBSET of the day's slugs. If no slug on the day
    disagrees, no subset of them disagrees. The dominance runs one way and
    only one way -- it licenses `DISAGREE = 0`, never a finality claim for
    a day carrying unreachable slugs.
    """
    import glob as _glob
    import hashlib
    out, all_final = {}, True
    winner_sources = winner_sources or {}
    for d in days:
        c = d.replace("-", "")
        fs = sorted(_glob.glob(str(Path(derived) /
                    f"p003_de_point_estimate_day_{c}_{revision}__*.json")))
        if not fs:
            raise EvaluatorRefused(
                f"REFUSED {NO_RECEIPT}: {d} has no point-estimate receipt "
                f"under {derived}. A cent figure whose winner source was "
                f"never verified must say so, and saying so requires the "
                f"receipt that did the verifying.")
        expected_sha = winner_sources.get(d)
        matched = None
        for candidate in reversed(fs):
            path = Path(candidate)
            try:
                payload = path.read_bytes()
                rec = json.loads(payload)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EvaluatorRefused(
                    f"REFUSED {NO_RECEIPT}: unreadable verification receipt "
                    f"{path}: {type(exc).__name__}: {exc}") from None
            if str(rec.get("day") or "").replace("-", "") != c:
                continue
            for block in _winner_source_blocks(rec):
                if expected_sha is None or block.get("sha256") == expected_sha:
                    matched = (path, payload, block)
                    break
            if matched is not None:
                break
        if matched is None:
            raise EvaluatorRefused(
                f"REFUSED {NO_RECEIPT}: {d} has no verification receipt for "
                f"the winner-source digest used by its settlement cells, "
                f"{expected_sha}.")
        path, payload, winner_source = matched
        verification = winner_source["chainlink_verification"]
        per_slug = verification.get("per_slug") or {}
        statuses = [row.get("status") for row in per_slug.values()
                    if isinstance(row, dict)]
        names = ("VERIFIED_AGREE", "DISAGREE", "BOUNDARY_NOT_IN_CAPTURE",
                 "CHAINLINK_UNAVAILABLE", "VENUE_UNRESOLVED")
        counts = {name: statuses.count(name) for name in names}
        unknown = sorted(set(statuses) - set(names))
        finality = verification.get("finality") or {}
        final = counts.get("DISAGREE", 1) == 0 and all(
            v == 0 for k, v in counts.items()
            if k not in ("VERIFIED_AGREE", "DISAGREE")) \
            and bool(per_slug) and not unknown \
            and counts["VERIFIED_AGREE"] == len(per_slug) \
            and finality.get("is_final") is True \
            and winner_source.get("is_final_for_quotation") is True
        all_final &= final
        out[d] = {"counts": counts, "receipt": path.name,
                  "receipt_sha256": hashlib.sha256(payload).hexdigest(),
                  "winner_source_sha256": winner_source.get("sha256"),
                  "matches_settlement_cells": (
                      expected_sha is None
                      or winner_source.get("sha256") == expected_sha),
                  "no_slug_disagrees": counts.get("DISAGREE", None) == 0,
                  "is_final_for_quotation": final,
                  "unknown_statuses": unknown,
                  "why_not_final": (None if final else
                                    "slugs the Chainlink stream cannot "
                                    "reach keep finality gated")}
    n_disagree = sum(row["counts"]["DISAGREE"] for row in out.values())
    if all_final:
        reading = (
            "Every settlement cell's exact winner-source bytes have an "
            "identity-matched verification, and every slug agrees. The "
            "verification changes no cent figure.")
    elif n_disagree:
        reading = (
            f"{n_disagree} slug disagreement(s) are present. Settlement cents "
            "using the venue winner record are not final for quotation.")
    else:
        reading = (
            "No slug disagreement is recorded, but at least one winner is not "
            "verifiable from the captured Chainlink boundary. The cent figures "
            "remain venue-record values and are not final for quotation.")
    return {
        "per_day": out,
        "every_day_final_for_quotation": all_final,
        "n_slug_disagreements": n_disagree,
        "THE_LIMIT_THESE_NUMBERS_CARRY": reading,
    }


def _load_forward_cells(root: Path, days, arms) -> dict:
    """Load only cells that reconcile to the V2 runner checkpoints."""
    try:
        cells = {
            (day, arm): AGG.load_cell(
                Path(root), day, arm, strict_forward=True)
            for day in days for arm in arms
        }
    except AGG.AggregateRefused as exc:
        raise EvaluatorRefused(str(exc)) from None
    for day in days:
        shared = {}
        for arm in arms:
            result = cells[(day, arm)]["result"]
            robust = result["robustness_leg"]
            shared[arm] = {
                "book_sha256": result.get("book_sha256"),
                "winner_source": result.get("winner_source"),
                "zero_model_cancel_baseline_total_cents": result.get(
                    "zero_model_cancel_baseline_total_cents"),
                "robust_zero_model_cancel_baseline_total_cents": robust.get(
                    "zero_model_cancel_baseline_total_cents"),
                "book_receipt": result.get("book_receipt"),
                "params_pin": result.get("params_pin"),
                "input_verification": result.get("input_verification"),
                "score_neutrality_certifications": result.get(
                    "score_neutrality_certifications"),
                "score_delta_max_certified": result.get(
                    "score_delta_max_certified"),
                "forward_book_margin_guard": result.get(
                    "forward_book_margin_guard"),
                "producer": result.get("producer"),
            }
        canonical = {
            arm: json.dumps(value, sort_keys=True, separators=(",", ":"))
            for arm, value in shared.items()
        }
        if len(set(canonical.values())) != 1:
            raise EvaluatorRefused(
                f"REFUSED {CELL_COHORT}: {day} arm cells disagree on their "
                f"book, zero-cancel baselines or guard inputs.")
    return cells


def evaluate(root: Path, days, *, n_expected: int = None,
             arms=ARMS, sided: int = SIDED, derived: Path = None,
             revision: str = "L250ms") -> dict:
    """The whole path. `days` REQUIRED; G comes from it, not a constant."""
    days = list(days)
    if not days:
        raise EvaluatorRefused(f"REFUSED {NO_DAYS}: pass the population.")
    if len(set(days)) != len(days):
        raise EvaluatorRefused(f"REFUSED {DUP_DAY}: {days}")
    G = len(days)
    if n_expected is not None and G != n_expected:
        raise EvaluatorRefused(
            f"REFUSED {SHORT}: declared N={n_expected}, given G={G}. The "
            f"floor moves with G; a short population is a different test.")
    cells = _load_forward_cells(Path(root), days, arms)
    lines, pools, per_arm = [], [], {}
    for a in arms:
        for d in days:
            one = AGG.pooled(cells, (d,), a)
            lines.append(per_day_line(cells[(d, a)], one, G, sided))
        pool = AGG.pooled(cells, days, a)
        pools.append(pool)
        day_p = day_sign_p(
            sum(1 for value in pool["per_day_D_cents"].values()
                if value > 0), G, sided)
        per_arm[a] = {
            "pooled": pool,
            "primary_fill_assumption": "REFERENCE_FILLS",
            "day_sign_test": {
                "p_two_sided": day_p,
                "n_positive_days": sum(
                    1 for value in pool["per_day_D_cents"].values()
                    if value > 0),
                "n_non_positive_days": sum(
                    1 for value in pool["per_day_D_cents"].values()
                    if value <= 0),
                "unit": "UTC_DAY"},
            "matched_random_test": {
                "p_two_sided": pool["p_two_sided"],
                "n_draws": pool["n_draws"]},
            "intersection_union_p": max(day_p, pool["p_two_sided"]),
            "futility": futility(pool["per_day_D_cents"], G, sided),
            "robustness_leg": robustness_pool(
                cells, days, a, pool["D_arm_cents"])}
    holms = AGG.holm(
        [per_arm[a]["intersection_union_p"] for a in arms])
    arm_results = {}
    for a, h in zip(arms, holms):
        row = per_arm[a]
        positive = row["pooled"]["D_arm_cents"] > 0
        beats_zero = (positive
                      and row["day_sign_test"]["p_two_sided"]
                      <= h["threshold"])
        beats_random = (positive
                         and row["matched_random_test"]["p_two_sided"]
                         <= h["threshold"])
        pooled_advances = h["passes"] and beats_zero and beats_random
        advances = (pooled_advances
                    and not row["robustness_leg"]["sign_reversal"])
        arm_results[a] = {
            **row, "holm": h,
            "beats_zero_cancel": beats_zero,
            "beats_matched_random": beats_random,
            "pooled_advances_before_robustness": pooled_advances,
            "advances": advances,
            "per_day_delta": list(row["pooled"]["per_day_D_cents"].values())}
    v = verdict_for_forward(arm_results)
    n = min(p["n_draws"] for p in pools)
    floor = floor_block(G, n, sided)
    from da_forward_result_guard import (             # local: no import cycle
        FAIL_SELECTION_READING, PASS_SELECTION_READING)
    selection_reading = (PASS_SELECTION_READING
                         if any(a["advances"] for a in arm_results.values())
                         else FAIL_SELECTION_READING)
    return {"protocol": PROTOCOL, "G": G, "days": days,
            "arm_order": list(arms), "arms": arm_results,
            "per_day_lines": lines,
            "per_arm": arm_results,
            "verdict": v,
            "FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT": floor,
            "attainable_minimum_p": floor["attainable_minimum_p"],
            "tolerance_negative_days": floor["tolerance_negative_days"],
            "a_pass_was_possible_at_this_G":
                floor["a_pass_was_possible_at_this_G"],
            "selection_history_reading": selection_reading,
            "n_independent_units": G,
            "per_cell": {
                f"{d}|{a}": {
                    "D_cents": cells[(d, a)]["D"],
                    "p_two_sided_descriptive_only":
                        cells[(d, a)]["result"].get("p_two_sided"),
                    "n_draws": cells[(d, a)]["n"],
                    "input_validation":
                        cells[(d, a)]["cell_validation"],
                    "result_source": cells[(d, a)]["result_source"],
                    "checkpoint_source":
                        cells[(d, a)]["checkpoint_source"]}
                for d in days for a in arms},
            "ANY_ARM_ALREADY_FUTILE":
                any(arm_results[a]["futility"]["FUTILE"] for a in arms),
            "futile_arms": [a for a in arms
                            if arm_results[a]["futility"]["FUTILE"]],
            "standing_warning": standing_warning(G, sided),
            "settlement_source": (
                settlement_source_disclosure(
                    days, derived, revision,
                    winner_sources={
                        d: cells[(d, arms[0])]["result"]
                            ["winner_source"]["sha256"]
                        for d in days})
                if derived is not None else
                {"NOT_COMPUTED": "pass derived= to carry R-810 finality"})}


def _write_json_exclusive(path: Path, payload: dict) -> None:
    """Publish atomically without ever replacing an existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time_ns()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=1, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise EvaluatorRefused(
                f"REFUSED {OUTPUT_EXISTS}: {path}. A final result is "
                f"immutable; use a new revision rather than replacing it.") \
                from None
    finally:
        temporary.unlink(missing_ok=True)


def finalize_result(computed: dict, metadata: dict, output: Path) -> dict:
    """Bind required human-readable limits, guard, then publish once."""
    from da_forward_result_guard import (             # local: no import cycle
        classify_forward_result, require_forward_result)

    if not isinstance(metadata, dict):
        raise EvaluatorRefused(
            f"REFUSED {BAD_METADATA}: expected an object, got "
            f"{type(metadata).__name__}.")
    overlap = sorted(set(computed) & set(metadata))
    if overlap:
        raise EvaluatorRefused(
            f"REFUSED {BAD_METADATA}: metadata may not replace computed "
            f"fields: {overlap}.")
    required = ("forward_limits", "pipeline_provenance_limit",
                "quality_decisions")
    missing = [name for name in required if not metadata.get(name)]
    if missing:
        raise EvaluatorRefused(
            f"REFUSED {MISSING_METADATA}: {missing}. These fields must be "
            f"bound before the result is written, not added later.")
    result = {**computed, **metadata}
    n_days = int(result.get("G", len(result.get("days") or [])))
    result["result_guard"] = require_forward_result(result, n_days=n_days)
    result["declared_statuses"] = classify_forward_result(
        result, n_days=n_days)
    _write_json_exclusive(Path(output), result)
    return result


def progress_emit(root: Path, days_scored, *, n_declared: int,
                  arms=ARMS, sided: int = SIDED, derived: Path = None,
                  revision: str = "L250ms") -> dict:
    """WHAT IS EMITTED AFTER EVERY DAY, UNPROMPTED. All four travel together.

    Nothing here is remembered at the moment it matters: the per-day line,
    the futility verdict WITH the day that killed it and its cause, the
    attainable minimum p AT THE G ACHIEVED SO FAR, and the tolerance at
    that G. At the ruled G=7 the tolerance is 0, so this says on DAY ONE
    that the first negative day ends an arm -- not on day seven.
    """
    days_scored = list(days_scored)
    if not days_scored:
        raise EvaluatorRefused(f"REFUSED {NO_DAYS}: nothing scored yet.")
    if len(set(days_scored)) != len(days_scored):
        raise EvaluatorRefused(f"REFUSED {DUP_DAY}: {days_scored}")
    G_so_far = len(days_scored)
    cells = _load_forward_cells(Path(root), days_scored, arms)
    lines, per_arm = [], {}
    for a in arms:
        for d in days_scored:
            lines.append(per_day_line(cells[(d, a)],
                                      AGG.pooled(cells, (d,), a),
                                      n_declared, sided))
        pool = AGG.pooled(cells, days_scored, a)
        per_arm[a] = {"pooled_so_far": pool,
                      "futility": futility(pool["per_day_D_cents"],
                                           n_declared, sided)}
    n = min(p["pooled_so_far"]["n_draws"] for p in per_arm.values())
    out = {
        "protocol": PROTOCOL, "emit": "AFTER_EVERY_DAY",
        "days_scored": days_scored, "G_so_far": G_so_far,
        "G_declared": n_declared, "days_remaining": n_declared - G_so_far,
        "per_day_lines": lines,
        "floor_at_the_G_ACHIEVED_SO_FAR": floor_block(G_so_far, n, sided),
        "floor_at_the_G_DECLARED": floor_block(n_declared, n, sided),
        "futility": {a: per_arm[a]["futility"] for a in arms},
        "ANY_ARM_ALREADY_DEAD": any(per_arm[a]["futility"]["FUTILE"]
                                    for a in arms),
        "standing_warning": standing_warning(n_declared, sided),
    }
    if derived is not None:
        out["settlement_source"] = settlement_source_disclosure(
            days_scored, derived, revision,
            winner_sources={
                d: cells[(d, arms[0])]["result"]
                    ["winner_source"]["sha256"]
                for d in days_scored})
    dead = [a for a in arms if per_arm[a]["futility"]["FUTILE"]]
    out["STOP_ADVICE"] = (
        "NOT FUTILE -- continue" if not dead else
        "FUTILE for " + ", ".join(
            f"{a} ({per_arm[a]['futility']['cause']}"
            + (f", killed by {per_arm[a]['futility']['negative_or_zero_days']}"
               if per_arm[a]["futility"]["negative_or_zero_days"] else "")
            + ")" for a in dead)
        + ". Stopping now is FREE: it can only reduce the chance of "
          "declaring success, never inflate one.")
    return out


# ------------------------------------------------------------- falsifier --
def falsify() -> int:                                        # noqa: C901
    """rule 15: a positive control it MUST flag, a known-bad it must
    REFUSE, and a partial input it must not silently accept."""
    import tempfile
    import hashlib
    import de_multiday_gate1_runner as R
    import de_settlement_control_run as SC
    cells, n = 0, 0

    def ck(name, cond):
        nonlocal cells, n
        cells += 1
        n += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # --- FUTILITY: the positive control it MUST fire on ------------------
    dead = futility({"d1": +100.0, "d2": -1.0, "d3": +100.0}, 7)
    ck("futility FIRES on a constructed dead case (1 negative day at G=7)",
       dead["FUTILE"] and dead["n_negative"] == 1)
    ck("  and names the day that killed it", dead["negative_or_zero_days"]
       == ["d2"])
    ck("  and its best-attainable p exceeds the threshold",
       dead["best_attainable_p_given_what_is_already_seen"] > 0.025)

    # --- FUTILITY: the known-bad it must STAY SILENT on ------------------
    live = futility({"d1": +100.0, "d2": +5.0, "d3": +1.0}, 7)
    ck("futility STAYS SILENT on a live partial run (3 of 7, all positive)",
       not live["FUTILE"])
    full = futility({f"d{i}": +1.0 for i in range(7)}, 7)
    ck("futility STAYS SILENT on a unanimous complete run", not full["FUTILE"])
    ck("  a unanimous G=7 attains exactly 2/128",
       abs(full["best_attainable_p_given_what_is_already_seen"]
           - 2 / 128) < 1e-12)

    # --- FUTILITY: a ZERO day is non-positive and must kill it -----------
    zero = futility({"d1": +100.0, "d2": 0.0}, 7)
    ck("futility treats a ZERO delta as non-positive (not a pass)",
       zero["FUTILE"])

    # --- FUTILITY: partial input it must REFUSE --------------------------
    try:
        futility({f"d{i}": +1.0 for i in range(9)}, 7)
        ck("futility REFUSES more days than declared", False)
    except EvaluatorRefused as e:
        ck("futility REFUSES more days than declared", SHORT in str(e))

    # --- FUTILITY: the two deaths must be TOLD APART ---------------------
    ck("a G=4 run with NO negative day is futile as TEST_IMPOSSIBLE_AT_THIS_G",
       futility({f"d{i}": +1.0 for i in range(4)}, 4)["cause"]
       == "TEST_IMPOSSIBLE_AT_THIS_G")
    ck("a G=7 run with a negative day is KILLED_BY_NEGATIVE_DAYS",
       dead["cause"] == "KILLED_BY_NEGATIVE_DAYS")
    ck("a live G=7 partial run is ALIVE", live["cause"] == "ALIVE")

    # --- FLOORS: two-sided, and NOT the one-sided constant ---------------
    ck("G=7 two-sided attainable min p == 2/128",
       abs(attainable_min_p(7, 500)["day_sign_component"] - 2 / 128) < 1e-12)
    ck("G=7 two-sided tolerance is 0 (unanimity)", tolerance(7, 0.025) == 0)
    ck("G=10 is the first N tolerating one negative day two-sided",
       tolerance(9, 0.025) == 0 and tolerance(10, 0.025) == 1)
    ck("G=6 two-sided could NOT have passed (0.03125 > 0.025)",
       not floor_block(6, 500)["a_pass_was_possible_at_this_G"])
    ck("G=7 two-sided COULD pass", floor_block(7, 500)
       ["a_pass_was_possible_at_this_G"])
    ck("the draw component does not move with G",
       attainable_min_p(4, 500)["draw_permutation_component"]
       == attainable_min_p(7, 500)["draw_permutation_component"] == 1 / 501)

    # --- STANDING WARNING: said unprompted, and correct at each G --------
    ck("G=7 emit warns UNANIMITY REQUIRED", "UNANIMITY REQUIRED"
       in standing_warning(7))
    ck("G=4 emit warns the pass is NOT ATTAINABLE",
       "NOT ATTAINABLE AT THIS G" in standing_warning(4)
       and "MEASURES THE CALENDAR" in standing_warning(4))
    ck("G=10 emit reports a survivable day", "tol=1" in standing_warning(10))
    ck("every per-day line carries the warning when G is given",
       standing_warning(7) in per_day_line(
           {"day": "d", "arm": "a", "D": 1.0, "n": 500,
            "baseline_total_cents": 0.0, "arm_total_cents": 1.0},
           {"p_two_sided": 0.5}, 7))

    # --- FINALITY: refuses a day it cannot evidence ----------------------
    with tempfile.TemporaryDirectory() as td:
        try:
            settlement_source_disclosure(["2026-09-04"], Path(td))
            ck("finality REFUSES a day with no receipt", False)
        except EvaluatorRefused as e:
            ck("finality REFUSES a day with no receipt", NO_RECEIPT in str(e))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        day, winner_sha = "2026-09-07", "a" * 64
        receipt = root / (
            "p003_de_point_estimate_day_20260907_L250ms__fixture.json")
        verified = {
            "day": day,
            "nested": {"winner_source": {
                "sha256": winner_sha, "is_final_for_quotation": True,
                "chainlink_verification": {
                    "per_slug": {"slug": {"status": "VERIFIED_AGREE"}},
                    "finality": {"is_final": True}}}}}
        receipt.write_text(json.dumps(verified))
        disclosure = settlement_source_disclosure(
            [day], root, winner_sources={day: winner_sha})
        ck("finality binds to the exact winner-source digest used for P&L",
           disclosure["per_day"][day]["matches_settlement_cells"] is True
           and disclosure["every_day_final_for_quotation"] is True)
        try:
            settlement_source_disclosure(
                [day], root, winner_sources={day: "b" * 64})
            ck("finality REFUSES a receipt for different winner bytes", False)
        except EvaluatorRefused as exc:
            ck("finality REFUSES a receipt for different winner bytes",
               NO_RECEIPT in str(exc))
        verified["nested"]["winner_source"]["chainlink_verification"][
            "per_slug"]["slug"]["status"] = "DISAGREE"
        receipt.write_text(json.dumps(verified))
        disclosure = settlement_source_disclosure(
            [day], root, winner_sources={day: winner_sha})
        ck("a disagreement cannot be labelled final",
           disclosure["every_day_final_for_quotation"] is False)

    # --- G FROM THE DATA: driven on a 6-day AND an 8-day fixture ---------
    def _fixture(td, days, D_by_day):
        root = Path(td)
        for d in days:
            c = d.replace("-", "")
            for arm in ARMS:
                D = D_by_day[d]
                book_sha = hashlib.sha256(d.encode()).hexdigest()
                seed = R.seed_for(book_sha, arm)
                winner_sha = "f" * 64
                checkpoint = root / f"de_settle_ckpt_{d}_{arm}.jsonl"
                identity = SC.run_identity(
                    d, arm, book_sha, SC.DECLARED_N, seed, winner_sha)
                with checkpoint.open("w") as handle:
                    handle.write(json.dumps({
                        "kind": "HEADER", "identity": identity,
                        "protocol": SC.PROTOCOL,
                        "n_draws": SC.DECLARED_N, "seed": seed,
                        "day": d, "arm": arm,
                        "winner_source_sha256": winner_sha,
                        "baseline_total_cents": 0.0}) + "\n")
                    for index in range(SC.DECLARED_N):
                        handle.write(json.dumps({
                            "i": index, "seed": seed + index,
                            "settled_total_cents": 0.0,
                            "D": 0.0}) + "\n")
                result = {
                    "protocol": SC.PROTOCOL, "day": d, "arm": arm,
                    "book_sha256": book_sha,
                    "winner_source": {"path": "resolutions.jsonl",
                                      "sha256": winner_sha},
                    "book_receipt": {
                        "sha256": "b" * 64, "book_sha256": book_sha,
                        "builder_commit": SC.PIPELINE_COMMIT,
                        "book_scoring_code": {}},
                    "params_pin": {
                        "path": "params.json", "sha256": "c" * 64},
                    "input_verification": {
                        "be_module": {}, "models": {}, "thetas": {}},
                    "score_neutrality_certifications": [
                        {"path": "cert.json", "sha256": "d" * 64}],
                    "score_delta_max_certified": {
                        frozen_arm: 0.25
                        for frozen_arm in SC.BEN.arm_heads()},
                    "forward_book_margin_guard": {"passes": True},
                    "primary_fill_assumption": "REFERENCE_FILLS",
                    "zero_model_cancel_baseline_total_cents": 0.0,
                    "arm_settled_total_cents": D,
                    "observed_D_cents": D,
                    "robustness_leg": {
                        "label": "NO_FILLS_UNTIL_NEXT_GENERATION",
                        "zero_model_cancel_baseline_total_cents": 0.0,
                        "arm_settled_total_cents": D,
                        "observed_D_cents": D,
                        "primary_label": "REFERENCE_FILLS",
                        "primary_observed_D_cents": D,
                        "sign_reversal": False,
                        "a_sign_reversal_blocks_promotion": True},
                    "n_draws": SC.DECLARED_N, "seed": seed,
                    "null": {"n": SC.DECLARED_N, "min_D": 0.0,
                             "max_D": 0.0, "mean_D": 0.0,
                             "n_at_or_beyond_two_sided": 0},
                    "p_two_sided": 1 / (1 + SC.DECLARED_N),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256":
                        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "be_module": "e" * 64,
                    "producer": {
                        "path": str(Path(SC.__file__).resolve()),
                        "sha256": hashlib.sha256(
                            Path(SC.__file__).read_bytes()).hexdigest(),
                    },
                }
                (root / f"de_settle_result_{c}_{arm}.json").write_text(
                    json.dumps(result))
        return root

    with tempfile.TemporaryDirectory() as td:
        d6 = [f"2026-09-{7 + i:02d}" for i in range(6)]
        r6 = evaluate(_fixture(td, d6, {d: +1.0 for d in d6}), d6)
        ck("a 6-day fixture yields G=6 FROM THE DATA", r6["G"] == 6)
        ck("  and its floor is the G=6 floor, not a constant",
           abs(r6["FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT"]
               ["attainable_minimum_p"]["day_sign_component"]
               - 2 / 64) < 1e-12)
        ck("  and a G=6 pass is NOT attainable two-sided",
           not r6["FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT"]
           ["a_pass_was_possible_at_this_G"])
    with tempfile.TemporaryDirectory() as td:
        d8 = [f"2026-09-{7 + i:02d}" for i in range(8)]
        r8 = evaluate(_fixture(td, d8, {d: +1.0 for d in d8}), d8)
        ck("an 8-day fixture yields G=8 FROM THE DATA", r8["G"] == 8)
        ck("  and its floor is the G=8 floor", abs(
            r8["FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT"]
            ["attainable_minimum_p"]["day_sign_component"] - 2 / 256) < 1e-12)
        ck("  and a G=8 pass IS attainable",
           r8["FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT"]
           ["a_pass_was_possible_at_this_G"])
        ck("  the two fixtures give DIFFERENT floors, so G is not pinned",
           r6["FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT"]["attainable_minimum_p"]
           ["day_sign_component"] != r8["FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT"]
           ["attainable_minimum_p"]["day_sign_component"])
        # and the per-day emit, on day 1 of 8, must already warn
        p1 = progress_emit(Path(td), d8[:1], n_declared=8)
        ck("progress_emit works on DAY ONE", p1["G_so_far"] == 1
           and p1["days_remaining"] == 7)
        ck("  and carries the floor at the G ACHIEVED and at the G DECLARED",
           "floor_at_the_G_ACHIEVED_SO_FAR" in p1
           and "floor_at_the_G_DECLARED" in p1)
        ck("  and says STOP_ADVICE unprompted", "STOP_ADVICE" in p1)
        d8bad = dict({d: +1.0 for d in d8}, **{d8[0]: -5.0})
        with tempfile.TemporaryDirectory() as td2:
            p2 = progress_emit(_fixture(td2, d8, d8bad), d8[:1], n_declared=8)
            ck("  a negative DAY ONE is called dead immediately",
               p2["ANY_ARM_ALREADY_DEAD"] and "FUTILE for " in p2["STOP_ADVICE"])
            ck("  and the STOP_ADVICE names the day that killed it",
               d8[0] in p2["STOP_ADVICE"])

    # --- BOTH CONJUNCTS: a low draw p cannot hide a negative day --------
    with tempfile.TemporaryDirectory() as td:
        d7 = [f"2026-09-{7 + i:02d}" for i in range(7)]
        one_negative = {day: 1.0 for day in d7}
        one_negative[d7[-1]] = -1.0
        r7 = evaluate(_fixture(td, d7, one_negative), d7)
        ck("a tiny matched-random p cannot pass a 6-of-7 day-sign run",
           r7["verdict"]["verdict"] == VERDICT_NONE
           and not any(row["advances"] for row in r7["arms"].values()))
        ck("  the day-sign p, not a warning, is the blocking conjunct",
           all(abs(row["day_sign_test"]["p_two_sided"] - 0.125) < 1e-12
               and row["matched_random_test"]["p_two_sided"] == 1 / 501
               for row in r7["arms"].values()))
    with tempfile.TemporaryDirectory() as td:
        d7 = [f"2026-09-{7 + i:02d}" for i in range(7)]
        unanimous = evaluate(
            _fixture(td, d7, {day: 1.0 for day in d7}), d7)
        ck("a unanimous positive fixture can clear both conjuncts",
           unanimous["verdict"]["verdict"] == VERDICT_BOTH
           and all(row["advances"] for row in unanimous["arms"].values()))
        ck("  the evaluator emits an arm MAPPING for the result guard",
           isinstance(unanimous["arms"], dict)
           and set(unanimous["arms"]) == set(ARMS))

        # The result cannot be published without its interpretive limits,
        # and cannot be silently replaced after it is published once.
        final_path = Path(td) / "forward_result.json"
        metadata = {
            "forward_limits": "second attempt, btc only, L=250ms",
            "pipeline_provenance_limit": "build and valuation moved together",
            "quality_decisions": {
                day: {"sources_read": ["data/pm_5min/raw"]}
                for day in d7},
        }
        final = finalize_result(unanimous, metadata, final_path)
        ck("finalization runs the result guard before exclusive publication",
           final_path.is_file()
           and final["result_guard"]["status"]
           == "FORWARD_RESULT_FIELDS_PRESENT")
        try:
            finalize_result(unanimous, metadata, final_path)
            ck("finalization REFUSES to replace an existing result", False)
        except EvaluatorRefused as exc:
            ck("finalization REFUSES to replace an existing result",
               OUTPUT_EXISTS in str(exc))
        try:
            finalize_result(unanimous,
                            {k: v for k, v in metadata.items()
                             if k != "quality_decisions"},
                            Path(td) / "missing.json")
            ck("finalization REFUSES missing required metadata", False)
        except EvaluatorRefused as exc:
            ck("finalization REFUSES missing required metadata",
               MISSING_METADATA in str(exc))

    # The robustness leg is mandatory and its sign can only block.
    with tempfile.TemporaryDirectory() as td:
        d7 = [f"2026-09-{7 + i:02d}" for i in range(7)]
        root = _fixture(td, d7, {day: 1.0 for day in d7})
        cell = root / f"de_settle_result_{d7[0].replace('-', '')}_{ARMS[0]}.json"
        payload = json.loads(cell.read_text())
        payload.pop("robustness_leg")
        cell.write_text(json.dumps(payload))
        try:
            evaluate(root, d7)
            ck("evaluate REFUSES a missing robustness leg", False)
        except EvaluatorRefused as exc:
            ck("evaluate REFUSES a missing robustness leg",
               (NO_ROBUSTNESS in str(exc)
                or AGG.BAD_RECONCILIATION in str(exc)))

    with tempfile.TemporaryDirectory() as td:
        day = "2026-09-07"
        root = _fixture(td, [day], {day: 1.0})
        cell = root / (
            f"de_settle_result_{day.replace('-', '')}_{ARMS[0]}.json")
        payload = json.loads(cell.read_text())
        payload["params_pin"]["sha256"] = "f" * 64
        cell.write_text(json.dumps(payload))
        try:
            evaluate(root, [day])
            ck("evaluate REFUSES arm cells from different input cohorts", False)
        except EvaluatorRefused as exc:
            ck("evaluate REFUSES arm cells from different input cohorts",
               CELL_COHORT in str(exc))

    # --- HOLM: m = 2, and the step-down thresholds -----------------------
    h = AGG.holm([0.02, 0.03])
    ck("Holm m=2: smaller p tested at 0.025", abs(h[0]["threshold"] - 0.025)
       < 1e-12)
    ck("Holm m=2: larger p tested at 0.05", abs(h[1]["threshold"] - 0.05)
       < 1e-12)
    hf = AGG.holm([0.026, 0.030])
    ck("Holm step-down: the SMALLEST p failing 0.025 fails BOTH arms",
       not hf[0]["passes"] and not hf[1]["passes"])
    hp = AGG.holm([0.001, 0.030])
    ck("  but a smallest p under 0.025 lets the larger be tested at 0.05",
       hp[0]["passes"] and hp[1]["passes"])

    # --- RESOLUTION: refuses a missing book, resolves a complete set -----
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        days = ["2026-09-07", "2026-09-08", "2026-09-09"]
        for d in days[:2]:
            (root / f"be_daybook_{d.replace('-', '')}_btc__FWD1.pkl").touch()
        try:
            resolve_books(root, days, "FWD1")
            ck("resolve_books REFUSES a missing book", False)
        except EvaluatorRefused as e:
            ck("resolve_books REFUSES a missing book",
               MISSING in str(e) and "2026-09-09" in str(e))
        (root / "be_daybook_20260909_btc__FWD1.pkl").touch()
        got = resolve_books(root, days, "FWD1")
        ck("resolve_books resolves a COMPLETE set", len(got) == 3)
        try:
            resolve_books(root, [], "FWD1")
            ck("resolve_books REFUSES an empty population", False)
        except EvaluatorRefused as e:
            ck("resolve_books REFUSES an empty population", NO_DAYS in str(e))
        try:
            resolve_books(root, days + days[:1], "FWD1")
            ck("resolve_books REFUSES a repeated day", False)
        except EvaluatorRefused as e:
            ck("resolve_books REFUSES a repeated day", DUP_DAY in str(e))

    # --- G COMES FROM THE DATA, and a short population is refused --------
    try:
        evaluate(Path("/nonexistent"), ["2026-09-04"], n_expected=7)
        ck("evaluate REFUSES G != declared N", False)
    except EvaluatorRefused as e:
        ck("evaluate REFUSES G != declared N", SHORT in str(e))
    try:
        evaluate(Path("/nonexistent"), [])
        ck("evaluate REFUSES an empty day set", False)
    except EvaluatorRefused as e:
        ck("evaluate REFUSES an empty day set", NO_DAYS in str(e))
    ck("there is NO module-level day constant to fall back on",
       not any(k for k, v in globals().items()
               if isinstance(v, tuple) and v and
               all(isinstance(x, str) and x.startswith("2026-") for x in v)))

    print(f"\n{n}/{cells} cells pass")
    return 0 if n == cells else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--falsify", action="store_true")
    parser.add_argument("--root")
    parser.add_argument("--days", nargs="+")
    parser.add_argument("--n-expected", type=int)
    parser.add_argument("--derived")
    parser.add_argument("--revision", default="L250ms")
    parser.add_argument("--metadata")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.falsify:
        return falsify()
    required = {"root": args.root, "days": args.days,
                "n_expected": args.n_expected, "derived": args.derived,
                "metadata": args.metadata, "output": args.output}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("the final evaluator requires " + ", ".join(missing))
    try:
        metadata = json.loads(Path(args.metadata).read_text())
        computed = evaluate(
            Path(args.root), args.days, n_expected=args.n_expected,
            derived=Path(args.derived), revision=args.revision)
        result = finalize_result(computed, metadata, Path(args.output))
    except (EvaluatorRefused, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"refused": str(exc)}, indent=1))
        return 3
    print(json.dumps({"written": args.output,
                      "verdict": result["verdict"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
