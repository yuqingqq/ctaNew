"""EV-Gates mechanism — the first executable piece, under ruling R-36.

R-36 returned R-24 (the `STOP-MM-VIABLE` amendment) as **PROSE-ONLY** and named
the landing evidence required to put it in force:

    "a CHECK THAT FAILS when a sign-blind threshold is supplied. Until that
     check exists and is shown to fail on a sign-blind input, STOP is not
     amended, it is only described."

This module is that check. Before it existed, `EV_GATES_PLAN` had **zero** lines
of code behind twenty behavioural commitments, and BE had reported R-24 "applied"
when it was a sentence in a plan.

THE DESIGN CHOICE THAT MATTERS. Sign-blindness is **DETECTED, not declared.**
A rule could be asked to carry a `favourable_arm` field and assert its own
directionality — but a declaration is exactly what failed everywhere else in this
corpus: the name is not the definition, and an author who has not noticed the
defect will not declare it. Instead the detector **mirrors the evidence**: it
feeds a rule an input and that input's reflection, and refuses any rule that
returns the same verdict for both. A rule that cannot tell `-0.532` from
`+0.532` says so under this test whatever its author believed.

That makes the check a *falsifier* in the sense `EV_GATES_PLAN` §6.2 requires:
it has a concrete input under which it fails, and the failing input is on disk.

    python3 live/pm_research/ev_gates.py --selftest
    python3 live/pm_research/ev_gates.py stop      # the live ladder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import flow_intensity as fi

PM = fi.PM

# R-24, as ruled: both verdict coins, both directions, symmetric.
VERDICT_COINS: tuple[str, ...] = ("btc", "eth")

FIRE_SIDE = "FIRE_SIDE"
PASS_SIDE = "PASS_SIDE"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

# A rule maps {coin: (lo, hi)} -> verdict string.
Rule = Callable[[Mapping[str, tuple[float, float]]], str]


# --------------------------------------------------------------- the rules

def stop_verdict_r24(cells: Mapping[str, tuple[float, float]]) -> str:
    """R-24's amended `STOP` rule. Directional and symmetric.

    FIRE_SIDE   both verdict coins exclude zero FROM BELOW  (hi < 0)
    PASS_SIDE   both verdict coins exclude zero FROM ABOVE  (lo > 0)
    otherwise   INSUFFICIENT_EVIDENCE

    The coordinator tightened their own draft from *at least one coin* to
    **both**, in both directions, so the bar has no thumb on it.
    """
    missing = [c for c in VERDICT_COINS if c not in cells]
    if missing:
        raise ValueError(f"STOP needs every verdict coin; missing {missing}")
    lo_hi = [cells[c] for c in VERDICT_COINS]
    if all(hi < 0.0 for _lo, hi in lo_hi):
        return FIRE_SIDE
    if all(lo > 0.0 for lo, _hi in lo_hi):
        return PASS_SIDE
    return INSUFFICIENT


def stop_verdict_original_SIGN_BLIND(
        cells: Mapping[str, tuple[float, float]]) -> str:
    """The bar R-24 replaced, kept ONLY as the failing witness.

    *"the interval must exclude zero on at least one verdict coin"* — no
    direction, and `on_pass` said *"proceed to the DE build"*. So the measured
    btc `-0.532 [-0.797, -0.287]`, the evidence that the maker is destroyed,
    satisfied it and read as a pass.

    This function exists to be REFUSED by `assert_directional`. It must never be
    used to evaluate anything.
    """
    for c in VERDICT_COINS:
        lo, hi = cells[c]
        if hi < 0.0 or lo > 0.0:        # excludes zero, either way
            return "PASS"
    return INSUFFICIENT


# ------------------------------------------------------- the R-36 detector

def mirror(cells: Mapping[str, tuple[float, float]]
           ) -> dict[str, tuple[float, float]]:
    """Reflect every interval through zero. `[-0.8, -0.3]` -> `[+0.3, +0.8]`."""
    return {c: (-hi, -lo) for c, (lo, hi) in cells.items()}


class SignBlind(AssertionError):
    """A verdict rule returned the same answer for an input and its mirror."""


def assert_directional(rule: Rule,
                       probe: Mapping[str, tuple[float, float]] | None = None
                       ) -> dict[str, Any]:
    """R-36's landing evidence: REFUSE a rule that cannot tell a sign.

    Feeds `rule` a probe and the probe's reflection through zero. A directional
    rule must answer differently; a sign-blind one answers the same, and that is
    the whole defect that let the programme-killing evidence read as a pass.

    Raises `SignBlind` on failure. Returns the two verdicts on success, so a
    caller can record WHAT the rule did rather than merely that it passed.
    """
    if probe is None:
        # the real measured btc/eth h=5 cells; see `stop_ladder()`
        probe = {"btc": (-0.797, -0.287), "eth": (-1.726, -0.759)}
    forward = rule(probe)
    reflected = rule(mirror(probe))
    if forward == reflected:
        raise SignBlind(
            f"SIGN-BLIND RULE REFUSED: returned {forward!r} for both "
            f"{dict(probe)} and its mirror {mirror(probe)}. A rule that cannot "
            f"distinguish an interval from its reflection cannot distinguish "
            f"'the maker is paid' from 'the maker is destroyed'.")
    return {"probe_verdict": forward, "mirror_verdict": reflected,
            "directional": True}



# ---------------------------------------------------------------------------
# R-40's GENERAL LESSON, applied to this module's own guard.
# ---------------------------------------------------------------------------
# "Fixing a defect in one channel does not fix the defect, it RELOCATES it to
#  whichever channel has no guard. A guard bounds a CHANNEL, never a BEHAVIOUR."
#
# `assert_directional` bounds ONE channel: rule EVALUATION. Sign-blindness can
# re-enter through a channel it never sees -- the BAR SPECIFICATION. In
# EV_GATES_PLAN section 9 the `GateBar` variants are:
#
#     Scalar(value, unit)          <- carries NO side
#     TwoSided(interval)           <- carries no side, and correctly so
#     ExcludesZero(side)           <- carries a side
#     Conjunction/Disjunction/...  <- inherit from their parts
#
# So a gate whose verdicts are DIRECTIONAL but whose bar is `Scalar` is
# sign-blind BY SPECIFICATION, and the rule-level mirror test never runs on it
# because a bar is not a rule. G-FF3 is exactly this shape on disk:
# `threshold: 0.0, unit: cents_per_share`, a SIGN test written as a scalar.
#
# BE found this by applying R-40's lesson to BE's own guard, before a reviewer
# did. The fix bounds the second channel; it does not claim to bound the
# behaviour, and a third channel (the verdict_map) is named as still open.

DIRECTIONAL_VERDICTS = frozenset({FIRE_SIDE, PASS_SIDE, "POSITIVE", "NEGATIVE"})

SIDED_BAR_VARIANTS = frozenset({"ExcludesZero"})
SIDELESS_BAR_VARIANTS = frozenset({"Scalar", "TwoSided"})
COMPOSITE_BAR_VARIANTS = frozenset({"Conjunction", "Disjunction", "AtLeastK",
                                    "PerScope"})


def bar_can_express_a_side(bar: Mapping[str, Any]) -> bool:
    """Can this bar specification distinguish one direction from the other?

    A composite is sided iff at least one part is. `TwoSided` is legitimately
    sideless -- an equivalence bar has no direction to lose.
    """
    kind = bar.get("kind")
    if kind in SIDED_BAR_VARIANTS:
        return bar.get("side") in ("POSITIVE", "NEGATIVE")
    if kind in SIDELESS_BAR_VARIANTS:
        return False
    if kind in COMPOSITE_BAR_VARIANTS:
        return any(bar_can_express_a_side(p) for p in bar.get("parts", []))
    raise ValueError(f"unknown GateBar variant {kind!r}")


def assert_bar_directional(bar: Mapping[str, Any],
                           declared_verdicts: Sequence[str]) -> dict[str, Any]:
    """Second channel: REFUSE a directional gate whose BAR cannot carry a side.

    `assert_directional` catches a sign-blind RULE. This catches a sign-blind
    SPECIFICATION -- the same defect entering where the first guard does not look.
    """
    directional = sorted(set(declared_verdicts) & DIRECTIONAL_VERDICTS)
    if not directional:
        return {"directional_gate": False, "checked": False,
                "note": "gate declares no directional verdict; bar side not required"}
    if not bar_can_express_a_side(bar):
        raise SignBlind(
            f"SIGN-BLIND BAR REFUSED: gate declares directional verdicts "
            f"{directional} but its bar {bar.get('kind')!r} cannot express a "
            f"side. A scalar threshold cannot say which direction passes, so the "
            f"verdict's sign comes from the reader, not the bar.")
    return {"directional_gate": True, "checked": True, "bar_is_sided": True}



# ---------------------------------------------------------------------------
# DETECTED, NOT DECLARED -- channel 3: `estimand_is_idealised`
# ---------------------------------------------------------------------------
# R-42 asked for the pattern to be applied "wherever a rule currently declares
# its own properties". `estimand_is_idealised` is the sharpest remaining case:
# EV_GATES_PLAN section 2.4 says it is "declared by the protocol, not inferred
# from the shape of a number" -- i.e. an author asserts, after seeing the
# results, which of their own arms was generous.
#
# It is DETECTABLE. An idealisation is generous BY CONSTRUCTION, so the arm
# computed under it measures better in the favourable direction than the arm
# computed without it. Feed both arms and the favourable direction, and the
# arms reveal which was generous. No author is asked anything.
#
# AND THE DETECTOR CATCHES WHAT A DECLARATION CANNOT: a declared-conservative
# arm that MEASURES as the generous one. That is on disk. From
# `skew_bound_v1.json`, terminal |net| p95, where SMALLER is favourable:
#
#     btc   SKEW_UB 21.41   SKEW_LB 45.66   -> UB generous, as declared
#     eth   SKEW_UB 19.98   SKEW_LB 17.10   -> LB is BETTER: POLARITY VIOLATED
#
# `SKEW_BOUND_RESULTS` notes the eth crossing and attributes it to small-sample
# noise at n=25 -- which may well be right. The point is that the declaration
# could not represent it at all, and the detector states it as a finding rather
# than silently carrying a label its own data contradicts.

class PolarityViolated(AssertionError):
    """An arm declared conservative measured as the generous one."""


def detect_idealisation(arms: Mapping[str, float], better_is: str,
                        declared: str | None = None) -> dict[str, Any]:
    """Which arm is generous? Measured from the arms, never asked of the author.

    `arms`      two or more named arms and their measured values
    `better_is` "lower" or "higher" -- which direction is favourable
    `declared`  optional: the arm the protocol CLAIMS is the idealised one

    Raises `PolarityViolated` when `declared` contradicts the measurement.
    """
    if better_is not in ("lower", "higher"):
        raise ValueError("better_is must be 'lower' or 'higher'")
    if len(arms) < 2:
        raise ValueError("idealisation is a COMPARISON; one arm cannot show it")
    ranked = sorted(arms.items(), key=lambda kv: kv[1],
                    reverse=(better_is == "higher"))
    generous, generous_v = ranked[0]
    conservative, conservative_v = ranked[-1]
    out: dict[str, Any] = {
        "measured_generous_arm": generous,
        "measured_conservative_arm": conservative,
        "values": dict(arms),
        "better_is": better_is,
        "margin": abs(generous_v - conservative_v),
        "detected_not_declared": True,
    }
    if declared is not None:
        out["declared_idealised_arm"] = declared
        out["declaration_holds"] = (declared == generous)
        if declared != generous:
            raise PolarityViolated(
                f"POLARITY VIOLATED: protocol declares {declared!r} the "
                f"idealised (generous) arm, but measured {arms} with "
                f"better_is={better_is!r} makes {generous!r} the generous one. "
                f"A declaration its own data contradicts is not a property of "
                f"the estimand; it is a claim about it.")
    return out


# --------------------------------------------------------- the live ladder

def stop_ladder(receipt: Path | None = None) -> dict[str, Any]:
    """Apply the amended rule at every horizon of the Layer-1 receipt.

    Reports the FULL ladder because the verdict is horizon-dependent and the
    horizon is unpinned (`Q-BE-4`, with the coordinator recommending h=5 to the
    user). A single number here would be the tuning `EDGE_LAYER1_PROTOCOL` warns
    against; the ladder makes the dependence visible instead of hiding it.
    """
    path = receipt or (PM / "derived/edge_layer1_v1.json")
    d = json.loads(path.read_text())
    out: dict[str, Any] = {"receipt": path.name, "coins": list(VERDICT_COINS),
                           "rule": "R-24 (amended, directional, both coins)",
                           "by_horizon": {}}
    for h in d["horizons_s"]:
        key = str(int(h))
        cells = {c: tuple(d["coins"][c]["horizons"][key]["markout_ci95_cents"])
                 for c in VERDICT_COINS}
        out["by_horizon"][key] = {
            "cells": {c: list(v) for c, v in cells.items()},
            "verdict": stop_verdict_r24(cells),
        }
    verdicts = {k: v["verdict"] for k, v in out["by_horizon"].items()}
    out["distinct_verdicts"] = sorted(set(verdicts.values()))
    out["horizon_dependent"] = len(out["distinct_verdicts"]) > 1
    out["caveats"] = [
        "STOP's OWN metric has never been computed; this reads a PRECONDITION.",
        "edge_l1_v1 is not STOP's estimand: no fee subtracted, maker never "
        "cancels, and its protocol forbids combining layers into one PnL.",
        "The receipt's population is CONTRADICTED (a 4x day over-report; the "
        "sample is one UTC day).",
        "Horizon is UNPINNED -- Q-BE-4. Picking one after seeing this table is "
        "the tuning the protocol warns against.",
    ]
    out["provenance"] = fi.provenance()
    return out


# -------------------------------------------------------------- selftest

def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    def _expect(label: str, exc: type[BaseException], fn) -> None:
        nonlocal checks
        try:
            fn()
        except exc:
            checks += 1
            return
        raise AssertionError(f"expected {exc.__name__} for {label}")

    # 1-4: the amended rule is directional, symmetric, and needs BOTH coins.
    both_neg = {"btc": (-0.797, -0.287), "eth": (-1.726, -0.759)}
    both_pos = mirror(both_neg)
    ok(stop_verdict_r24(both_neg) == FIRE_SIDE, "both exclude below -> FIRE_SIDE")
    ok(stop_verdict_r24(both_pos) == PASS_SIDE, "both exclude above -> PASS_SIDE")
    ok(stop_verdict_r24({"btc": (-0.797, -0.287), "eth": (-1.284, 0.089)})
       == INSUFFICIENT, "one coin spanning zero -> INSUFFICIENT")
    ok(stop_verdict_r24({"btc": (-0.8, -0.3), "eth": (0.3, 0.8)})
       == INSUFFICIENT, "coins disagreeing in DIRECTION -> INSUFFICIENT")

    # 5: R-24's tightening. Under "at least one coin" the mixed case would
    # have resolved; under "both" it does not. This is the clause the
    # coordinator tightened in their own draft.
    ok(stop_verdict_r24({"btc": (-0.8, -0.3), "eth": (-1.0, 0.5)}) == INSUFFICIENT,
       "one coin is not enough -- R-24's own tightening holds")

    # 6-7: THE LANDING EVIDENCE R-36 NAMED.
    # The check must FAIL on a sign-blind input. Demonstrated, not asserted.
    blind_failed = False
    try:
        assert_directional(stop_verdict_original_SIGN_BLIND)
    except SignBlind as exc:
        blind_failed = True
        ok("SIGN-BLIND RULE REFUSED" in str(exc),
           "the refusal names the defect it caught")
    ok(blind_failed,
       "R-36 LANDING EVIDENCE: the check FAILS on the sign-blind bar R-24 replaced")

    # 8: and it passes the amended rule, so it is not merely refusing everything.
    res = assert_directional(stop_verdict_r24)
    ok(res["probe_verdict"] == FIRE_SIDE and res["mirror_verdict"] == PASS_SIDE,
       "the amended rule answers OPPOSITELY on an input and its mirror")

    # 9: a gate that cannot fire is not a gate -- the detector must be able to
    # fail on something other than the one witness it ships with.
    always_pass: Rule = lambda cells: "PASS"
    try:
        assert_directional(always_pass)
        ok(False, "unreachable")
    except SignBlind:
        ok(True, "a constant rule is also refused")

    # 10: the mirror is an involution, so the probe is not doing hidden work.
    ok(mirror(mirror(both_neg)) == {k: tuple(v) for k, v in both_neg.items()},
       "mirror is its own inverse")

    # 11: missing verdict coin is refused, not defaulted.
    try:
        stop_verdict_r24({"btc": (-0.8, -0.3)})
        ok(False, "unreachable")
    except ValueError:
        ok(True, "a missing verdict coin is refused, never defaulted")

    # 12-17: R-40's lesson turned on this module's own guard. The bar channel.
    ok(bar_can_express_a_side({"kind": "ExcludesZero", "side": "NEGATIVE"}),
       "ExcludesZero(side) carries a side")
    ok(not bar_can_express_a_side({"kind": "Scalar", "value": 0.0}),
       "Scalar carries NO side")
    ok(bar_can_express_a_side({"kind": "Conjunction", "parts": [
        {"kind": "Scalar", "value": 0.0},
        {"kind": "ExcludesZero", "side": "NEGATIVE"}]}),
       "a composite is sided iff a part is")

    # G-FF3's ACTUAL shape on disk: threshold 0.0, unit cents_per_share -- a
    # SIGN test written as a scalar. The bar-channel guard catches it.
    gff3_bar = {"kind": "Scalar", "value": 0.0, "unit": "cents_per_share"}
    caught = False
    try:
        assert_bar_directional(gff3_bar, ["POSITIVE", "NEGATIVE"])
    except SignBlind as exc:
        caught = True
        ok("SIGN-BLIND BAR REFUSED" in str(exc), "the bar refusal names its defect")
    ok(caught, "R-40 CHANNEL 2: a directional gate on a Scalar bar is REFUSED")

    ok(assert_bar_directional({"kind": "TwoSided"}, ["PASS"])["checked"] is False,
       "an equivalence gate is not asked for a side it does not need")

    # 18-23: channel 3 -- estimand_is_idealised, DETECTED. Real receipt values
    # from skew_bound_v1.json, terminal |net| p95, smaller is favourable.
    btc = detect_idealisation({"SKEW_UB": 21.405, "SKEW_LB": 45.656},
                              better_is="lower", declared="SKEW_UB")
    ok(btc["measured_generous_arm"] == "SKEW_UB" and btc["declaration_holds"],
       "btc: the declared-generous arm measures generous")

    # eth: the declaration its own data contradicts.
    violated = False
    try:
        detect_idealisation({"SKEW_UB": 19.976, "SKEW_LB": 17.100},
                            better_is="lower", declared="SKEW_UB")
    except PolarityViolated as exc:
        violated = True
        ok("POLARITY VIOLATED" in str(exc), "the violation names itself")
    ok(violated,
       "eth: a declared-conservative arm measuring GENEROUS is REFUSED -- the "
       "case a declaration cannot represent")

    # undeclared use is fine: the detector answers without being told.
    quiet = detect_idealisation({"a": 1.0, "b": 2.0}, better_is="higher")
    ok(quiet["measured_generous_arm"] == "b"
       and "declared_idealised_arm" not in quiet,
       "the detector needs no declaration at all")

    _expect("one arm cannot show idealisation", ValueError,
            lambda: detect_idealisation({"only": 1.0}, better_is="lower"))
    _expect("direction must be named", ValueError,
            lambda: detect_idealisation({"a": 1.0, "b": 2.0}, better_is="sideways"))

    print(f"ev_gates selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["stop"], default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "stop":
        res = stop_ladder()
        print(json.dumps(res, indent=1))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
