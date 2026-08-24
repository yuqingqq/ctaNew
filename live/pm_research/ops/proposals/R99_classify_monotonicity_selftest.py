"""R-99 commissioned instrument: STATIC MONOTONICITY SELFTEST on `classify()`.

**OWNERSHIP: `replay_canary.py` is DA-owned (COORDINATION §1). This file is a
DROP-IN PROPOSAL, not an edit.** OPS wrote and verified it; DA applies it, or the
coordinator reassigns. OPS edited a DA-owned file once today under a stale role
sheet and reverted; not repeating that.

DE's reasoning, which is the whole point: **a rule-ordering property cannot drift
with data, but it CAN be silently reintroduced by a code change.** So this is a
STATIC test over the rule's input lattice, run with every canary — not a
statistic over observations.

The property, from R-94: pre-amendment the rule was NON-MONOTONE in its own
evidence — `disagreements == 0` with zero harm was INVALID, while `disagreements
== 5` with the SAME zero harm was fine. The strictly safer observation was
punished more harshly.

Run:  python3 live/pm_research/ops/proposals/R99_classify_monotonicity_selftest.py
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, "/home/yuqing/ctaNew")
from live.pm_research.replay_canary import classify   # noqa: E402

FATAL_PREFIX = "INVALID"
K = 12          # disagreement counts swept; the property is not size-dependent


def is_fatal(status: str) -> bool:
    return status.startswith(FATAL_PREFIX)


def check(rule) -> list[tuple[str, bool, str]]:
    """Every assertion is a PROPERTY over the lattice, not a spot value."""
    out: list[tuple[str, bool, str]] = []

    # P1 — THE R-94 PROPERTY. Zero harm and a wired guard must never be fatal,
    # at ANY disagreement count including zero. This is the exact defect.
    fatal_at = [d for d in range(K + 1) if is_fatal(rule(1, d, 0.0)[0])]
    out.append(("P1 zero-harm + wired guard is never fatal",
                not fatal_at, f"fatal at disagreements={fatal_at}"))

    # P2 — NO SAFER INPUT PUNISHED HARDER. With harm fixed at zero there must be
    # no pair d1 < d2 where the SMALLER count is fatal and the larger is not.
    bad = [(a, b) for a in range(K + 1) for b in range(a + 1, K + 1)
           if is_fatal(rule(1, a, 0.0)[0]) and not is_fatal(rule(1, b, 0.0)[0])]
    out.append(("P2 no smaller-disagreement-count punished harder",
                not bad, f"non-monotone pairs {bad[:4]}"))

    # P3 — the unwired arm is UNCHANGED by the amendment and stays fatal, for
    # every disagreement count and both harm cases. R-94 did not touch it.
    leaks = [(d, x) for d in range(K + 1) for x in (0.0, 0.25)
             if rule(0, d, x)[0] != "INVALID_UNWIRED_GUARD"]
    out.append(("P3 unwired guard stays fatal everywhere",
                not leaks, f"leaked at {leaks[:4]}"))

    # P4 — nonzero harm at zero disagreements stays fatal: the counters then
    # contradict each other and fail-closed is the right answer.
    out.append(("P4 zero disagreements + nonzero delta stays fatal",
                is_fatal(rule(1, 0, 0.25)[0]), rule(1, 0, 0.25)[0]))

    # P5 — reclassification is granted ONLY on a measured zero, never inferred.
    granted_nonzero = [d for d in range(K + 1) if rule(1, d, 0.25)[1]]
    out.append(("P5 reclassify only on a MEASURED zero delta",
                not granted_nonzero, f"granted at {granted_nonzero}"))
    return out


def pre_amendment(event_only: int, disagreements: int, delta: float):
    """The rule as it stood BEFORE the amendment — the negative control.

    A test that cannot fail on the defect it was written for is not a test.
    """
    if event_only == 0:
        return "INVALID_UNWIRED_GUARD", False
    if disagreements == 0:
        return "INVALID_UNBOUND_GUARD", False          # <- the non-monotonicity
    if math.isclose(delta, 0.0, abs_tol=1e-15):
        return "BOUND_ZERO_SCORE_DELTA", False
    return "VALID_GUARD_BITES", False


def main() -> int:
    print("  CURRENT rule (replay_canary.classify)")
    cur = check(classify)
    for name, ok, detail in cur:
        print(f"    {'PASS' if ok else 'FAIL'}  {name:52s}{'' if ok else '  ' + detail}")
    print("\n  NEGATIVE CONTROL — the pre-amendment rule MUST fail P1 and P2")
    pre = check(pre_amendment)
    for name, ok, detail in pre:
        print(f"    {'PASS' if ok else 'FAIL'}  {name:52s}{'' if ok else '  ' + detail}")

    cur_ok = all(ok for _, ok, _ in cur)
    control_catches = (not pre[0][1]) and (not pre[1][1])
    print(f"\n  current rule: {sum(ok for _,ok,_ in cur)}/{len(cur)} properties hold")
    print(f"  negative control detects the R-94 defect: {control_catches}")
    return 0 if (cur_ok and control_catches) else 1


if __name__ == "__main__":
    raise SystemExit(main())
