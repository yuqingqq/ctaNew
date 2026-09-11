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

import json
import sys
from math import comb
from pathlib import Path

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


def settlement_source_disclosure(days, derived: Path,
                                 revision: str = "L250ms") -> dict:
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
    out, all_final = {}, True
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
        rec = json.loads(Path(fs[-1]).read_text())

        def _walk(o, pre=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield from _walk(v, pre + "/" + k)
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    yield from _walk(v, pre + f"[{i}]")
            else:
                yield pre, o
        counts = {k.rsplit("/", 1)[1]: v for k, v in _walk(rec)
                  if "/chainlink_verification/counts/" in k}
        final = counts.get("DISAGREE", 1) == 0 and all(
            v == 0 for k, v in counts.items()
            if k not in ("VERIFIED_AGREE", "DISAGREE"))
        all_final &= final
        out[d] = {"counts": counts, "receipt": Path(fs[-1]).name,
                  "no_slug_disagrees": counts.get("DISAGREE", None) == 0,
                  "is_final_for_quotation": final,
                  "why_not_final": (None if final else
                                    "slugs the Chainlink stream cannot "
                                    "reach keep finality gated")}
    return {
        "per_day": out,
        "every_day_final_for_quotation": all_final,
        "THE_LIMIT_THESE_NUMBERS_CARRY": (
            "step 2 called `winner_source` without `verification=`, so its "
            "published cents are labelled VENUE_RECORD_NOT_VERIFIED_"
            "AGAINST_CHAINLINK. The check has since been read from the day "
            "receipts and NO SLUG DISAGREES on any day, so NO CENT FIGURE "
            "CHANGES -- what was missing was the label, not the money."),
    }


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
    cells = {(d, a): AGG.load_cell(Path(root), d, a)
             for d in days for a in arms}
    lines, pools, per_arm = [], [], {}
    for a in arms:
        for d in days:
            one = AGG.pooled(cells, (d,), a)
            lines.append(per_day_line(cells[(d, a)], one, G, sided))
        pool = AGG.pooled(cells, days, a)
        pools.append(pool)
        per_arm[a] = {
            "pooled": pool,
            "futility": futility(pool["per_day_D_cents"], G, sided)}
    holms = AGG.holm([p["p_two_sided"] for p in pools])
    v = AGG.verdict_for(pools, holms)
    n = min(p["n_draws"] for p in pools)
    return {"protocol": PROTOCOL, "G": G, "days": days, "arms": list(arms),
            "per_day_lines": lines,
            "per_arm": {a: {**per_arm[a], "holm": h}
                        for a, h in zip(arms, holms)},
            "verdict": v,
            "FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT": floor_block(G, n, sided),
            "ANY_ARM_ALREADY_FUTILE":
                any(per_arm[a]["futility"]["FUTILE"] for a in arms),
            "futile_arms": [a for a in arms
                            if per_arm[a]["futility"]["FUTILE"]],
            "standing_warning": standing_warning(G, sided),
            "settlement_source": (
                settlement_source_disclosure(days, derived, revision)
                if derived is not None else
                {"NOT_COMPUTED": "pass derived= to carry R-810 finality"})}


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
    cells = {(d, a): AGG.load_cell(Path(root), d, a)
             for d in days_scored for a in arms}
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
            days_scored, derived, revision)
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

    # --- G FROM THE DATA: driven on a 6-day AND an 8-day fixture ---------
    def _fixture(td, days, D_by_day):
        root = Path(td)
        for d in days:
            c = d.replace("-", "")
            for arm in ARMS:
                (root / f"de_settle_result_{c}_{arm}.json").write_text(
                    json.dumps({"observed_D_cents": D_by_day[d],
                                "zero_model_cancel_baseline_total_cents": 0.0,
                                "arm_settled_total_cents": D_by_day[d]}))
                with (root / f"de_settle_ckpt_{d}_{arm}.jsonl").open("w") as f:
                    f.write(json.dumps({"kind": "HEADER",
                                        "n_draws": 500}) + "\n")
                    for i in range(500):
                        f.write(json.dumps({"i": i, "D": 0.0}) + "\n")
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
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print("usage: de_forward_evaluator.py --falsify")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
