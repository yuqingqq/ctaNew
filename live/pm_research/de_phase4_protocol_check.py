"""Anti-transcription check for DE_PHASE4_PROTOCOL_DRAFT.md.

SURFACE AUTHORISATION (R-126, in-file): R-379 TASK 3 (DE seat).
RESEARCH-ONLY, OFFLINE. It reads the DRAFT and evaluates every number the
draft prints, so a printed arithmetic and a printed answer cannot disagree.

WHY THIS EXISTS.  LANE4 B4.3 records a multiplicity derivation that was
internally consistent and externally unreproducible -- a "full space" of 21
that no reader could recompute -- produced BY the instrument written to
prevent exactly that.  The lesson is that a derivation must be RECOMPUTED
from its recorded inputs, never transcribed.  This module recomputes:

  * the size of the Phase-4 cell space from the axis rungs the draft lists;
  * the minimum-draw table from `N >= 20m - 1`;
  * that the numbers appearing in the draft's prose are the ones the
    recomputation produces.

Both directions (rule 15/16): the real draft must PASS, and a doctored copy
with one digit changed must FAIL.  A checker that only ever passes on the
document it was written beside proves nothing.

    python3 live/pm_research/de_phase4_protocol_check.py --selftest
"""
from __future__ import annotations

import argparse
import pathlib
import re

DRAFT = (pathlib.Path(__file__).resolve().parents[2]
         / "live/pm_research/plans/DE_PHASE4_PROTOCOL_DRAFT.md")

# The axes, transcribed ONCE from the protocol's own §4 table and then
# CHECKED against the draft's prose rather than trusted.
AXES = {
    "latency_rungs": 9,          # 5,10,20,30,50,75,100,150,250 ms
    "cost_rungs": 8,             # 0.00 .. 1.50 c
    "budget_rungs": 3,           # 5, 10, 15 %
    "repost_fill_models": 2,
    "protection_modes": 2,
    "reset_cost_semantics": 2,
    "reduce_configs": 3,         # off / on+shared / on+separate
    "coins": 2,
}
LATENCY_MS = (5, 10, 20, 30, 50, 75, 100, 150, 250)
COST_C = (0.00, 0.01, 0.05, 0.10, 0.25, 0.40, 0.75, 1.50)
BUDGETS = (0.05, 0.10, 0.15)
ALPHA = 0.05


class DraftMismatch(RuntimeError):
    """The draft prints a number its own inputs do not produce."""


def space_size(axes: dict[str, int] | None = None) -> int:
    a = axes or AXES
    n = 1
    for v in a.values():
        n *= v
    return n


def min_draws(m: int, alpha: float = ALPHA) -> int:
    """Smallest N with m/(N+1) < alpha -- the family can CLEAR alpha at all.

    A Holm-corrected family whose smallest attainable p already exceeds alpha
    cannot produce a surviving cell whatever the effect: iteration 011's
    24 x 1/501 = 0.0479 was that shape with one cell of headroom left."""
    n = 1
    while m / (n + 1) >= alpha:
        n += 1
    return n


def floor_p(m: int, n_draws: int) -> float:
    return m / (n_draws + 1)


def check(text: str) -> list[str]:
    """Return the list of MISMATCHES between the draft's printed numbers and
    the recomputation (empty = consistent)."""
    bad: list[str] = []
    prod = " × ".join(str(v) for v in
                      (AXES["latency_rungs"], AXES["cost_rungs"],
                       AXES["budget_rungs"], AXES["repost_fill_models"],
                       AXES["protection_modes"],
                       AXES["reset_cost_semantics"],
                       AXES["reduce_configs"], AXES["coins"]))
    if prod not in text:
        bad.append(f"the printed product '{prod}' is absent -- a reader "
                   f"cannot recompute a space size from steps that are not "
                   f"shown")
    n = space_size()
    if f"{n:,}" not in text:
        bad.append(f"space size {n:,} absent from the draft")
    # the printed latency ladder and cost bracket must be the ones counted
    if ", ".join(str(x) for x in LATENCY_MS) not in text:
        bad.append("the latency ladder in the draft is not the one counted")
    if len(LATENCY_MS) != AXES["latency_rungs"]:
        bad.append("latency rung count disagrees with the ladder itself")
    if len(COST_C) != AXES["cost_rungs"]:
        bad.append("cost rung count disagrees with the bracket itself")
    if len(BUDGETS) != AXES["budget_rungs"]:
        bad.append("budget rung count disagrees with the grid itself")
    # the minimum-draw table
    for m, printed in ((1, 20), (2, 40), (54, 1080)):
        got = min_draws(m)
        if got != printed:
            bad.append(f"minimum draws for m={m}: draft prints {printed}, "
                       f"recomputation gives {got}")
        # the draft prints thousands with a separator; accept either form,
        # and REQUIRE one of them -- an absent value is the shape that reads
        # as agreement
        if str(printed) not in text and f"{printed:,}" not in text:
            bad.append(f"minimum-draw value {printed} (m={m}) absent")
    # the adjudicated-family size the draft claims for a selection-axis ladder
    if AXES["latency_rungs"] * AXES["budget_rungs"] * AXES["coins"] != 54:
        bad.append("the 54-cell family is not 9 x 3 x 2 on these axes")
    # the floors quoted at N = 200
    for m, printed in ((1, "0.00498"), (2, "0.00995"), (54, "0.269")):
        got = floor_p(m, 200)
        if abs(got - float(printed)) > 5e-4:
            bad.append(f"Holm floor at N=200 for m={m}: draft prints "
                       f"{printed}, recomputation gives {got:.5f}")
        if printed not in text:
            bad.append(f"Holm floor {printed} (m={m}) absent from the draft")
    # 011's own resolution figure, quoted as the precedent
    holm011 = 24 * (1 / 501)
    if f"{holm011:.4f}" not in text:
        bad.append(f"the 011 precedent figure {holm011:.4f} is quoted "
                   f"differently from 24 x 1/501")
    return bad


#: R-459: the scheduled diagnostic is declared in an ADDENDUM, never by
#: editing the frozen document (rule 13). The addendum is only worth
#: anything if it binds the bytes it claims to be an addendum TO -- so the
#: sha it carries is recomputed here from the frozen file, and the ruling
#: it stands on is named.
ADDENDUM = DRAFT.parent / "DE_PHASE4_DIAGNOSTIC_ADDENDUM_2026-09-02.md"

EXPECTED_CHECKS = 21


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_phase4_protocol_check] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    text = DRAFT.read_text()
    ok(space_size() == 10368,
       f"the cell space RECOMPUTES to {space_size():,} from the axis rungs, "
       f"never transcribed")
    ok(min_draws(1) == 20 and min_draws(2) == 40 and min_draws(54) == 1080,
       "the minimum-draw rule N >= 20m (STRICT: m/(N+1) < alpha) reproduces "
       "the draft's table -- the off-by-one that 20m-1 would have printed is "
       "exactly a family sitting AT alpha and reading as if it cleared")
    ok(abs(floor_p(54, 200) - 0.269) < 5e-4,
       "AND THE ROW THAT MATTERS: at N=200 a 54-cell family's Holm floor is "
       f"{floor_p(54, 200):.3f} > 0.05 -- no cell could survive whatever the "
       f"effect")
    ok(abs(24 * (1 / 501) - 0.0479) < 5e-5,
       "the 011 precedent recomputes: 24 x 1/501 = 0.0479, a family with one "
       "cell of headroom")
    real = check(text)
    ok(real == [], f"POSITIVE CONTROL: the real draft is internally "
                   f"consistent -- {real}")

    # KNOWN-BADS: each must be caught, so the clean pass above is not vacuous
    ok(check(text.replace("10,368", "10,360")) != [],
       "KNOWN-BAD: a doctored space size is caught")
    ok(check(text.replace("1,080", "1,000").replace("1080", "1000")) != [],
       "KNOWN-BAD: a doctored minimum-draw value is caught")
    ok(check(text.replace("0.269", "0.026")) != [],
       "KNOWN-BAD: a doctored Holm floor is caught -- this is the digit that "
       "would turn an impossible family into a passable-looking one")
    bad_axes = dict(AXES, latency_rungs=7)
    ok(space_size(bad_axes) != space_size(),
       "KNOWN-BAD: changing an axis changes the recomputed space, so the "
       "space is a function of the axes and not a constant")
    # ---- R-459: the addendum binds the frozen document by its bytes ----
    import hashlib as _hl
    _frozen_sha = _hl.sha256(DRAFT.read_bytes()).hexdigest()
    ok(ADDENDUM.exists(), f"the diagnostic addendum exists at "
                          f"{ADDENDUM.name}")
    _add = ADDENDUM.read_text()
    ok(_frozen_sha in _add,
       f"THE ADDENDUM BINDS THE FROZEN DOCUMENT BY ITS BYTES: sha256 "
       f"{_frozen_sha[:16]}... recomputed here from "
       f"{DRAFT.name} and found in the addendum -- so an addendum written "
       f"against a different version of the protocol is a red check, not a "
       f"reader's problem (rule 13: the frozen document is never edited)")
    ok("R-459" in _add and "seventh" in _add,
       "and it NAMES THE RULING it stands on -- R-459, the USER's seventh "
       "decision -- so the authority for running a protocol that is "
       "otherwise execution-gated is in the document rather than in a "
       "conversation")
    ok("DIAGNOSTIC_NEVER_EVIDENCE" in _add and "is_a_validation = false"
       in _add and "G = 0" in _add,
       "and it declares what every output says about itself before any "
       "cell exists: is_a_validation false, G = 0, "
       "DIAGNOSTIC_NEVER_EVIDENCE")
    ok(_hl.sha256(b"not the protocol").hexdigest() not in _add,
       "KNOWN-BAD/CONTROL on that binding: an unrelated sha is NOT in the "
       "addendum, so the check above is a comparison and not a substring "
       "that any hex would satisfy")

    # ---- the RUNNER's declared grid IS the addendum's -------------------
    # A grid widened in code and not in the declaration is the shape rule 11
    # exists for: a rung nobody declared, produced by a runner that would
    # happily compute it.
    import sys as _sys
    _sys.path.insert(0, str(DRAFT.resolve().parents[1]))
    import de_phase4_diag_runner as _RUN
    _rungs_in_add = [r for r in _RUN.LATENCY_RUNGS_MS
                     if f"{r}" in _add.split("| latency `L`")[1].split("|")[1]]
    ok(len(_rungs_in_add) == len(_RUN.LATENCY_RUNGS_MS) == 9,
       f"THE RUNNER'S LATENCY AXIS IS THE ADDENDUM'S: all "
       f"{len(_RUN.LATENCY_RUNGS_MS)} rungs {_RUN.LATENCY_RUNGS_MS} are "
       f"named in the addendum's own §b row -- a rung added in code and "
       f"not in the declaration would leave this check short")
    ok(all(f"{int(b * 100)}%" in _add for b in _RUN.BUDGETS)
       and len(_RUN.BUDGETS) == 3,
       f"and the budget axis matches too: {_RUN.BUDGETS} against the "
       f"addendum's 5% / 10% / 15%")
    ok(_RUN.PRIMARY["latency_ms"] == 250 and _RUN.PRIMARY["budget"] == 0.10
       and _RUN.PRIMARY["coin"] == "btc"
       and _RUN.PRIMARY["enable_reduce"] is False
       and _RUN.PRIMARY["charge_reset_cost_at_generation_start"] is False,
       f"and the PRIMARY cell is the frozen one: {_RUN.PRIMARY}")
    ok(set(_RUN.ARMS_NOT_RUN) == {"HAZARD_ONLY_NEUTRAL", "CONDVALUE_NEUTRAL",
                                  "CONDVALUE_X_SKEW_X_FAIRPRICE"}
       and all(a in _add for a in _RUN.ARMS_NOT_RUN),
       f"and the arms the runner refuses to run are the arms the addendum "
       f"declares unrunnable: {sorted(_RUN.ARMS_NOT_RUN)}")
    ok("CONDVALUE_OVER_SKEWED_REF" in _add
       and any("CONDVALUE_OVER_SKEWED_REF" in a for a in _RUN.ARMS),
       "and the arm NAMED for this diagnostic appears in both -- the "
       "resolution is in the declaration, not only in the code")
    _widened = list(_RUN.LATENCY_RUNGS_MS) + [200]
    ok(not all(f"{r}" in _add.split("| latency `L`")[1].split("|")[1]
               for r in _widened),
       "KNOWN-BAD: a WIDENED grid (a 200 ms rung added to the runner's "
       "axis) is not in the addendum, so this check goes red -- which is "
       "the point: a rung may not enter the code without entering the "
       "declaration first")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_phase4_protocol_check] selftest OK -- {n[0]} checks")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    print("\n".join(check(DRAFT.read_text())) or "consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
