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
def per_day_line(cell: dict, pool: dict) -> str:
    """The line the user reads, one per (day, arm)."""
    return (f"DONE {cell['day']} {cell['arm']} "
            f"D={cell['D']:+.2f} p={pool['p_two_sided']:.6f} "
            f"n={cell['n']} base={cell['baseline_total_cents']:+.2f} "
            f"arm={cell['arm_total_cents']:+.2f}")


def evaluate(root: Path, days, *, n_expected: int = None,
             arms=ARMS, sided: int = SIDED) -> dict:
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
            lines.append(per_day_line(cells[(d, a)], one))
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
                            if per_arm[a]["futility"]["FUTILE"]]}


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
