"""DE-Matched-Random -- the ACTING control the Phase-4 null requires.

SURFACE AUTHORISATION: R-459 §3(iii).  `de_lane4_real_parity.ARM_RUNNABLE`
records `RANDOM_MATCHED: NO_CONTRACT_IDENTITY_FOR_AN_ACTING_CONTROL` -- the
null of the frozen protocol (`DE_PHASE4_PROTOCOL_DRAFT.md:147-176`) is a
control that CANCELS, and nothing in the lane could say what such a control
is.  This file declares that identity and implements it.

THE CONTRACT IDENTITY, DECLARED (this is DE's ASK A1/A2, answered here and
in the addendum):

    RANDOM_MATCHED is an ACTING arm that cancels generations chosen
    UNIFORMLY AT RANDOM inside each (side, hour) stratum, where the number
    it cancels in a stratum is DETERMINED BY THE TREATED ARM's action count
    in that stratum and is never a caller-chosen number (LANE4 B1.1).

Four rules make it a control rather than a coin flip, and each is a refusal:

  * the demand per stratum comes from the treated arm.  A caller-supplied
    count is refused: a control whose size is chosen is a control that can
    be tuned.
  * the pool is ordered into a TOTAL ORDER before the RNG touches it, so a
    draw is reproducible from its seed and does not inherit dict order.
  * a stratum whose eligible pool is SMALLER than the demand REFUSES.  It
    does not clamp: clamping silently answers an easier question than the
    one the treated arm poses (the protocol says so at `:157-159`).
  * the draw is RANDOM WITH RESPECT TO THE TREATED ARM.  A "control" that
    reproduces the treated arm's own actions is refused by identity, not
    by taste -- it is the arm under test wearing the null's name.

    python3 live/pm_research/de_matched_random_control.py --selftest
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

EXPECTED_CHECKS = 20

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harmful_stateful_policy import SIDES        # noqa: E402

ARM_NAME = "RANDOM_MATCHED"
CONTRACT_IDENTITY = (
    "an ACTING arm cancelling generations chosen uniformly at random "
    "within (side, hour) strata, with the per-stratum count DETERMINED by "
    "the treated arm and never caller-chosen; pool totally ordered before "
    "the RNG; a stratum short of the demand REFUSES rather than clamping"
)


class ControlRefused(RuntimeError):
    """The control refuses rather than becoming a weaker question."""


def stratum(gen: dict) -> tuple:
    """(side, hour) -- the strata the protocol matches on (`:150`)."""
    for k in ("slug", "side", "hour"):
        if k not in gen:
            # SITE: stratum#1
            raise ControlRefused(f"generation is missing {k!r}: "
                                 f"{sorted(gen)}")
    if gen["side"] not in SIDES:
        # SITE: stratum#2
        raise ControlRefused(f"side {gen['side']!r} not in {SIDES}")
    return (gen["side"], gen["hour"])


def demand_from_treated(treated_actions, pool) -> dict:
    """How many cancellations the control owes per stratum -- READ OFF the
    treated arm, never supplied."""
    slugs = {g["slug"] for g in pool}
    unknown = sorted({a["slug"] for a in treated_actions
                      if a["slug"] not in slugs})
    if unknown:
        # SITE: demand#1
        raise ControlRefused(
            f"the treated arm acted on {unknown} which are not in the "
            f"eligible pool: the control cannot match an action count it "
            f"cannot place in a stratum")
    d: dict = {}
    by_slug = {g["slug"]: g for g in pool}
    for a in treated_actions:
        d[stratum(by_slug[a["slug"]])] = d.get(
            stratum(by_slug[a["slug"]]), 0) + 1
    return d


def draw(pool, treated_actions, seed: int, *, n_per_stratum=None) -> list:
    """One matched draw: the slugs this control cancels."""
    if n_per_stratum is not None:
        # SITE: draw#1
        raise ControlRefused(
            "the per-stratum count is DETERMINED BY THE TREATED ARM and is "
            "never a caller-chosen number (LANE4 B1.1): a control whose "
            "size is chosen is a control that can be tuned")
    demand = demand_from_treated(treated_actions, pool)
    strata: dict = {}
    for g in pool:
        strata.setdefault(stratum(g), []).append(g["slug"])
    out = []
    rng = random.Random(seed)
    for st in sorted(demand):
        # TOTAL ORDER BEFORE THE RNG TOUCHES IT: sorted, not dict order.
        avail = sorted(strata.get(st, []))
        want = demand[st]
        if want > len(avail):
            # SITE: draw#2
            raise ControlRefused(
                f"stratum {st} needs {want} cancellations and has "
                f"{len(avail)} eligible: REFUSED rather than clamped -- a "
                f"clamp silently answers an easier question than the one "
                f"the treated arm poses (protocol :157-159)")
        out.extend(rng.sample(avail, want))
    return sorted(out)


def refuse_if_not_random(drawn, treated_actions, *, pool_size: int) -> None:
    """A draw identical to the treated arm's own actions is the arm under
    test wearing the null's name."""
    t = sorted({a["slug"] for a in treated_actions})
    if sorted(drawn) == t and pool_size > len(t):
        # SITE: identity#1
        raise ControlRefused(
            f"the 'control' cancelled exactly the treated arm's own "
            f"{len(t)} actions out of a pool of {pool_size}: that is the "
            f"arm under test wearing the null's name, refused by identity "
            f"rather than by taste")


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_matched_random_control] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle=None):
        try:
            fn()
        except ControlRefused as exc:
            if needle and needle not in str(exc):
                raise SystemExit(f"[de_matched_random_control] FAIL: {label}"
                                 f" -- refused for another reason ({exc})")
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_matched_random_control] FAIL (no refusal): "
                         f"{label}")

    pool = [{"slug": f"g{i}", "side": SIDES[i % 2], "hour": i % 3,
             "harm_cents": 0.0} for i in range(60)]
    treated = [{"slug": "g0"}, {"slug": "g1"}, {"slug": "g2"}, {"slug": "g6"}]

    d = demand_from_treated(treated, pool)
    ok(sum(d.values()) == len(treated) and len(d) >= 2,
       f"THE DEMAND IS READ OFF THE TREATED ARM: {d} -- "
       f"{sum(d.values())} cancellations across {len(d)} (side, hour) "
       f"strata, matching the treated arm's own action count stratum by "
       f"stratum rather than in total")
    refuses(lambda: draw(pool, treated, 1, n_per_stratum={("BUY_UP", 0): 5}),
            "KNOWN-BAD: a CALLER-CHOSEN per-stratum count REFUSES -- a "
            "control whose size is chosen is a control that can be tuned "
            "(LANE4 B1.1)", needle="never a caller-chosen number")

    a1 = draw(pool, treated, seed=7)
    a2 = draw(pool, treated, seed=7)
    b1 = draw(pool, treated, seed=8)
    ok(a1 == a2 and len(a1) == len(treated),
       f"THE DRAW IS REPRODUCIBLE FROM ITS SEED: seed 7 gives {a1} twice -- "
       f"the pool is ordered totally before the RNG touches it, so a draw "
       f"does not inherit dict order and a receipt can be re-run")
    ok(b1 != a1,
       f"and a different seed gives a different draw ({b1}), so the "
       f"reproducibility above is the seed's and not a constant")
    got = {}
    for g in pool:
        got[stratum(g)] = got.get(stratum(g), 0) + 1
    drawn_strata = {}
    by_slug = {g["slug"]: g for g in pool}
    for s in a1:
        drawn_strata[stratum(by_slug[s])] = \
            drawn_strata.get(stratum(by_slug[s]), 0) + 1
    ok(drawn_strata == d,
       f"AND THE MATCH HOLDS PER STRATUM, not just in total: the draw's "
       f"{drawn_strata} equals the treated arm's {d} -- matching on the "
       f"count alone would let the control act in easier hours")

    # ---- the pool must be able to pay ------------------------------------
    thin = [{"slug": "t0", "side": SIDES[0], "hour": 0},
            {"slug": "t1", "side": SIDES[0], "hour": 0}]
    heavy = [{"slug": "t0"}, {"slug": "t1"}]
    ok(len(draw(thin, heavy, seed=1)) == 2,
       "POSITIVE CONTROL: a pool exactly the size of the demand pays it in "
       "full -- the refusal below is about a SHORT pool, not a tight one")
    refuses(lambda: draw(thin, heavy + [{"slug": "t0"}], seed=1),
            "KNOWN-BAD: a stratum SHORT of the demand REFUSES rather than "
            "clamping -- a clamp silently answers an easier question than "
            "the one the treated arm poses", needle="rather than clamped")
    refuses(lambda: demand_from_treated([{"slug": "not_in_pool"}], pool),
            "KNOWN-BAD: a treated action on a slug OUTSIDE the eligible "
            "pool REFUSES -- the control cannot match a count it cannot "
            "place in a stratum", needle="not in the eligible pool")
    refuses(lambda: stratum({"slug": "x", "side": "LONG", "hour": 0}),
            "KNOWN-BAD: an unknown side REFUSES", needle="not in")
    refuses(lambda: stratum({"slug": "x", "side": SIDES[0]}),
            "KNOWN-BAD: a generation missing `hour` REFUSES by name -- the "
            "strata are the protocol's and cannot be silently collapsed",
            needle="missing 'hour'")

    # ---- IDENTITY: a 'control' that reproduces the treated arm -----------
    refuses(lambda: refuse_if_not_random([a["slug"] for a in treated],
                                         treated, pool_size=len(pool)),
            "KNOWN-BAD: a draw identical to the treated arm's OWN actions "
            "is refused BY IDENTITY -- it is the arm under test wearing the "
            "null's name", needle="wearing the null's name")
    refuse_if_not_random(a1, treated, pool_size=len(pool))
    ok(True,
       f"POSITIVE CONTROL: the real draw {a1} is not the treated arm's "
       f"{sorted(a['slug'] for a in treated)} and passes the identity "
       f"check, so the refusal above is a filter and not a wall")
    refuse_if_not_random(["t0", "t1"], heavy, pool_size=2)
    ok(True,
       "and when the pool IS the treated set -- two eligible, two acted -- "
       "the identity check does not fire: with no freedom left, equality "
       "is arithmetic rather than evidence of copying")

    # ---- THE PLANTED-HARM CONTROL (rule 15) ------------------------------
    # A treated arm that cancels the harmful generations must beat the
    # draws; the draws must not beat it, or the null is not a null.
    planted = [{"slug": f"p{i}", "side": SIDES[i % 2], "hour": i % 3,
                "harm_cents": 100.0 if i < 6 else 0.0} for i in range(60)]
    harm = {g["slug"]: g["harm_cents"] for g in planted}
    treated_smart = [{"slug": f"p{i}"} for i in range(6)]
    treated_value = sum(harm[a["slug"]] for a in treated_smart)
    draws = [draw(planted, treated_smart, seed=s) for s in range(200)]
    dv = [sum(harm[s] for s in dr) for dr in draws]
    beaten = sum(1 for v in dv if v >= treated_value)
    ok(treated_value == 600.0 and beaten == 0,
       f"POSITIVE CONTROL WITH PLANTED HARM: the treated arm cancels the "
       f"six harmful generations ({treated_value} cents avoided) and NONE "
       f"of 200 matched draws reaches it -- so the control is beatable, "
       f"which is what makes not beating it mean something")
    ok(max(dv) > 0 and sum(dv) / len(dv) < treated_value / 2,
       f"and the draws are not degenerate: their best is {max(dv)} and "
       f"their mean {sum(dv) / len(dv):.1f} against the treated "
       f"{treated_value} -- a control that never touched a harmful "
       f"generation would be matched on the wrong thing")
    ok(len(draws) == 200,
       "with 200 draws, the protocol's declared minimum (§6) -- the number "
       "is the protocol's and is not chosen here")
    flat = [{"slug": f"p{i}", "side": SIDES[i % 2], "hour": i % 3,
             "harm_cents": 0.0} for i in range(60)]
    fharm = {g["slug"]: g["harm_cents"] for g in flat}
    ftreated = [{"slug": f"p{i}"} for i in range(6)]
    fdv = [sum(fharm[s] for s in draw(flat, ftreated, seed=s))
           for s in range(200)]
    ok(max(fdv) == 0 and sum(fharm[a["slug"]] for a in ftreated) == 0,
       "KNOWN-BAD FOR THE OTHER DIRECTION: with NO planted harm the "
       "treated arm scores exactly what the draws do (zero) -- the "
       "separation above came from the planting, not from the machinery")

    ok(ARM_NAME == "RANDOM_MATCHED" and "DETERMINED by the treated arm"
       in CONTRACT_IDENTITY,
       f"and the contract identity is DECLARED in the module for the "
       f"lane's `ARM_RUNNABLE` to point at: {CONTRACT_IDENTITY[:72]}...")
    ok(SIDES == ("BUY_UP", "SELL_UP"),
       f"the side vocabulary is the policy's own object, imported: {SIDES}")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_matched_random_control] selftest OK -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
