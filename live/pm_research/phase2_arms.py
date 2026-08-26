"""PHASE 2 — three heads on the top-up, per the committed declaration.

AUTHORISATION (R-126, in-file): R-170/R-173. Governed by `phase2_declaration`,
which was committed BEFORE any Phase-2 number existed (rule 11). Nothing here
chooses a threshold, window, feature set or hyperparameter -- if this file and
the declaration ever disagree, the declaration wins and this file is wrong.

THE ARCHITECTURE, and the reason it is not the obvious one:
  FIT on the CONSUMED FRAGMENT   (already spent; fitting costs nothing new)
  SCORE on the TOP-UP            (held out; R-166(3) reserved it)
Fitting on the top-up would consume the only held-out tape Phase 2 has and
make all three heads in-sample at once.

  A  PM_PLUS_FINE       the FROZEN artifact APPLIED UNCHANGED. No refit, no
                        weighting. R-157(2): the incumbent is not rewritten
                        mid-comparison, so arm A does not even load a fitter.
  B  PLUS_PRED_STATE_V1 A's features + DA's 21 state features, fitted on the
                        fragment WITH w = 1/n_rows(generation).
  C  LGBM_PINNED        same features and weighting, capacity pinned in the
                        declaration, seed pinned, no grid, no early stopping.

Every arm scores the SAME top-up rows. A row any arm cannot feature is dropped
from ALL arms, so the comparison stays paired -- an arm scoring a different
population is not a comparison at all.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, "/home/yuqing/ctaNew/live/pm_research")
import phase2_declaration as D

DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
TOPUP = DERIVED / "harmful_exposure_rows_v3_topup.json"
FRAGMENT = DERIVED / "harmful_exposure_rows_v3_eraB.json"
FROZEN = DERIVED / "harmful_reduced_fine_candidate_v1.json"
OUT = DERIVED / "phase2_three_arm_v1.json"


class PopulationLeak(RuntimeError):
    """A scoring row came from the fitting population."""


def assert_disjoint(fit_slugs: set, score_slugs: set) -> None:
    """The two populations must not intersect. AT ALL.

    This is the check that would have caught the 808-window build before any
    arm ran: the test set contained every fitting slug, and nothing errored."""
    overlap = fit_slugs & score_slugs
    if overlap:
        raise PopulationLeak(
            f"{len(overlap)} slug(s) appear in BOTH the fitting and scoring "
            f"populations (e.g. {sorted(overlap)[:3]}). Arms B and C are fitted "
            f"on the fragment; scoring them on rows they were fitted on is "
            f"in-sample and FLATTERS them. Refusing.")


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    assert_disjoint({"a", "b"}, {"c", "d"})
    ok(True, "KNOWN-GOOD: disjoint populations pass")
    try:
        assert_disjoint({"a", "b"}, {"b", "c"})
        ok(False, "an overlapping slug must be refused")
    except PopulationLeak as e:
        ok("FLATTERS" in str(e),
           "POSITIVE CONTROL: a single shared slug is REFUSED, naming the "
           "direction of the harm -- this is the 808-window failure, caught "
           "before any arm runs rather than after")
    ok(D.ARMS[0] == "PM_PLUS_FINE" and "PM_PLUS_FINE" not in D.WEIGHTED_ARMS,
       "arm A is the unweighted frozen incumbent, per the declaration")
    ok(D.N_RANDOM >= 200 and D.DECISION_METRIC == "net_cents",
       "declared null and decision metric come from the declaration, not "
       "from this file")
    ok(D.MULTIPLICITY_AFTER == 4, "multiplicity 4 after scoring two candidates")
    ok(len(D.PRED_STATE_V1) == 21, "PRED_STATE_V1 is the declared 21 features")
    ok("early_stopping" not in D.LGBM_PARAMS,
       "arm C has no early stopping -- nothing about it is chosen after "
       "seeing the scoring side")
    print(f"phase2_arms selftest: {checks} checks OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    selftest()
    if not TOPUP.exists():
        raise SystemExit(f"REFUSED: {TOPUP.name} does not exist. Phase 2 has "
                         f"no test surface; build it before scoring.")
    print("  ready: declaration loaded, populations will be asserted disjoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
