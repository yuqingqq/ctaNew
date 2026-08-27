"""Time-based train/score embargo. Slug identity is NOT an embargo.

AUTHORISATION (R-126, in-file): R-184(4)(iii), user audit 2026-08-27.

WHAT WENT WRONG, stated so the next reader does not repeat it. Phase 2 enforced
the split with `assert_disjoint` over SLUG SETS, and the name made it sound
covered. It was not: two windows can carry different slugs and still overlap in
WALL-CLOCK time, because a window's labels reach forward past its own slug
start (t_start + FILL_HORIZON + MARKOUT). Measured at the decision clock, the
split had a gap of **-8.134 s** against a declared 60 s embargo -- scoring began
BEFORE training ended.

THE RULE, per the user's sequence:
    label_exit_time + EMBARGO_S  <  first score feature time
enforced on TIMES, never on identities. A row is admissible to training only
if everything its label can see has finished at least EMBARGO_S before the
earliest scored feature.

THE SELFTEST FAILS ON A 59-SECOND GAP. A boundary that is not tested at the
boundary is a boundary nobody has checked.
"""
from __future__ import annotations

import sys

EMBARGO_S = 60.0            # manifest split_embargo_s
FILL_HORIZON_S = 1.0
MARKOUT_S = 5.0


class EmbargoViolation(RuntimeError):
    """Training and scoring are closer than the declared embargo."""


def label_exit_time(row: dict) -> float:
    """The latest wall-clock instant a row's label can depend on."""
    return float(row["t0"]) + float(row["t_start"]) + FILL_HORIZON_S + MARKOUT_S


def feature_time(row: dict) -> float:
    """The decision-clock instant a row's features are read at."""
    return float(row["t0"]) + float(row["t_start"])


def purge_training(train_rows, score_rows, embargo_s: float = EMBARGO_S):
    """Drop training rows that violate the embargo. Returns (kept, dropped).

    Purges the TRAINING side, because the scoring population is the declared
    holdout and must not be trimmed to fit -- shrinking the test set to rescue
    a contaminated train set is how a leak becomes a smaller, invisible leak."""
    if not score_rows:
        raise EmbargoViolation("no scoring rows: an embargo over an empty "
                               "holdout is vacuous, not satisfied")
    first_score = min(feature_time(r) for r in score_rows)
    limit = first_score - embargo_s
    kept, dropped = [], []
    for r in train_rows:
        (kept if label_exit_time(r) < limit else dropped).append(r)
    return kept, dropped


def assert_embargo(train_rows, score_rows, embargo_s: float = EMBARGO_S) -> dict:
    """REFUSE unless every training label closes embargo_s before scoring opens."""
    if not train_rows or not score_rows:
        raise EmbargoViolation("empty side: embargo is undefined")
    last_exit = max(label_exit_time(r) for r in train_rows)
    first_score = min(feature_time(r) for r in score_rows)
    gap = first_score - last_exit
    if gap < embargo_s:
        raise EmbargoViolation(
            f"embargo VIOLATED: gap {gap:.3f}s < required {embargo_s:.1f}s. "
            f"Last training label exit {last_exit:.3f}, first scoring feature "
            f"{first_score:.3f}. Slug-disjointness does not imply this and did "
            f"not catch it.")
    return {"gap_s": gap, "embargo_s": embargo_s,
            "last_train_label_exit": last_exit, "first_score_feature": first_score}


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    def R(t0, ts):
        return {"t0": t0, "t_start": ts, "slug": f"x-{t0}", "side": "BUY_UP", "gen": 1}

    base = 1_000_000.0
    # a training row whose label exits at base+6.0 (ts=0 -> +1 +5)
    tr = [R(base, 0.0)]

    # THE BOUNDARY ARMS. 61s passes, 59s FAILS. A boundary untested at the
    # boundary is a boundary nobody has checked.
    ok(assert_embargo(tr, [R(base + 6.0 + 61.0, 0.0)])["gap_s"] >= 60.0,
       "KNOWN-GOOD: a 61s gap satisfies the 60s embargo")
    try:
        assert_embargo(tr, [R(base + 6.0 + 59.0, 0.0)])
        ok(False, "a 59s gap MUST fail")
    except EmbargoViolation as e:
        ok("59.000s < required 60.0s" in str(e),
           "POSITIVE CONTROL: a 59s gap is REFUSED and the message names the "
           "actual gap -- the test fires exactly one second inside the boundary")
    try:
        assert_embargo(tr, [R(base + 6.0 + 60.0, 0.0)])
        ok(True, "exactly 60s is admitted (>= is the declared rule)")
    except EmbargoViolation:
        ok(False, "exactly 60s must be admitted")

    # THE REAL DEFECT REPRODUCED: distinct slugs, overlapping times
    a = R(1787650200, 251.0)      # fragment-side row
    b = R(1787650500, -56.0)      # top-up-side row, DIFFERENT slug
    ok(a["slug"] != b["slug"], "the two rows carry DIFFERENT slugs")
    try:
        assert_embargo([a], [b])
        ok(False, "different slugs must not be enough")
    except EmbargoViolation:
        ok(True, "POSITIVE CONTROL: two rows with DIFFERENT SLUGS are still "
                 "refused when their times overlap -- exactly what "
                 "assert_disjoint could not see")

    kept, dropped = purge_training([R(base, 0.0), R(base + 1000.0, 0.0)],
                                   [R(base + 200.0, 0.0)])
    ok(len(kept) == 1 and len(dropped) == 1,
       "purge_training drops only the violating training rows")
    ok(dropped[0]["t0"] == base + 1000.0,
       "and it drops the LATE one, not the early one")
    try:
        purge_training([R(base, 0.0)], [])
        ok(False, "an empty holdout must be refused")
    except EmbargoViolation:
        ok(True, "an empty holdout is REFUSED, not treated as satisfied")

    print(f"phase2_embargo selftest: {checks} checks OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else 0)
