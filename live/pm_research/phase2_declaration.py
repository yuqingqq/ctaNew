"""PHASE-2 DECLARATION. Written and committed BEFORE any Phase-2 number exists.

AUTHORISATION (R-126, in-file): R-170, Phase 2 on the declared top-up.

RULE 11: choosing after seeing voids the test. Everything a later result could
be accused of having been chosen to fit is fixed HERE, in a file whose commit
predates the scoring run. If a number later suggests a different threshold,
window, or hyperparameter would have been better, that is a finding for a
FRESH declaration, not an edit to this file.

THE THREE HEADS, reported separately, never pooled into a "best":
  A  PM_PLUS_FINE      the FROZEN incumbent, applied UNCHANGED and UNWEIGHTED.
                       R-157(2): the incumbent is not rewritten mid-comparison.
  B  +PRED_STATE_V1    A's features + DA's 21 predictor-state features,
                       fitted WITH R-157 weighting w = 1/n_rows(generation).
  C  LGBM_PINNED       one pinned-capacity gradient boosting model, same
                       features as B, same weighting.

THE PACKAGE CONFOUND, declared before it can be spun (R-157(3)): B and C carry
BOTH new features AND the weighting; A carries neither. So a B or C win is
`features + weighting` JOINTLY and is NOT decomposable from three arms. The
direction that matters: A is the incumbent a freeze already adopted, so a
candidate winning purely on weighting would cause new FEATURES to be adopted on
the strength of a reweighting that had nothing to do with them.

MULTIPLICITY: 2 frozen-race candidates already stand. Phase 2 scores TWO more
(B, C), so any clears-claim here is judged against a null accounting for FOUR.

BTC IS EXPECTED TO BE UNDERPOWERED and that is a NAMEABLE OUTCOME, declared
now so it cannot later be presented as a disappointment or quietly dropped:
DA's top-up gives btc 33 OK windows (19.1%, PM_GAP-dominated) against eth 171
(98.8%). If 33 windows cannot separate three heads, the result is
UNDERPOWERED-ON-BTC. eth is the informative coin.
"""
from __future__ import annotations

# ---- LGBM hyperparameters, PINNED BEFORE ANY SCORING ----------------------
# Chosen for CAPACITY CONTROL, not tuned: shallow, few leaves, strong
# min_child_samples, heavy subsampling. The point of arm C is to ask whether
# a nonlinear learner finds anything the linear arms miss -- not to win a
# tuning contest on a small consumed population, which is how a fragile
# positive gets manufactured. NO GRID, NO EARLY STOPPING ON THE TEST SIDE.
LGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 200,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 10.0,
    "n_jobs": 4,
    "verbose": -1,
    "random_state": 20260826,
}
LGBM_VALUE_PARAMS = dict(LGBM_PARAMS, objective="regression")

ARMS = ("PM_PLUS_FINE", "PLUS_PRED_STATE_V1", "LGBM_PINNED")
WEIGHTED_ARMS = ("PLUS_PRED_STATE_V1", "LGBM_PINNED")   # R-157(1); A stays unweighted
N_RANDOM = 200                                          # rule 6 minimum
BUDGETS = (0.05, 0.10, 0.15)
TARGET_LATENCY_MS = 50
DECISION_METRIC = "net_cents"                           # rule 7: never harm share
POPULATION = "da_development_topup"                     # era end from DA's v2 receipt
MULTIPLICITY_BEFORE = 2
MULTIPLICITY_AFTER = 4

PRED_STATE_V1 = (
    "time_remaining_s", "terminal_window", "gen_age_s", "gen_age_missing",
    "level_size", "queue_ahead_of_level", "queue_ahead_norm",
    "queue_ahead_missing", "remaining_size_frac", "remaining_size_missing",
    "touch_move_age_s", "touch_move_missing",
    "pm_feed_age_s", "pm_feed_stale", "pm_feed_missing",
    "bn_feed_age_s", "bn_feed_stale", "bn_feed_missing",
    "level_size_vel_50ms", "level_size_vel_250ms", "level_size_vel_1000ms",
)

DECLARED_OUTCOMES = (
    "SEPARATES: a candidate beats the incumbent on net_cents AND beats the "
    "matched-random max on NET at the same budget.",
    "NULL: no candidate separates; the incumbent stands.",
    "UNDERPOWERED-ON-BTC: btc's 33 windows cannot separate the heads. Named, "
    "not treated as a negative result.",
)


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    ok(len(PRED_STATE_V1) == 21, "PRED_STATE_V1 is 21 features, matching DA's "
                                 "emitted numeric surface")
    ok("PM_PLUS_FINE" not in WEIGHTED_ARMS,
       "the FROZEN incumbent is UNWEIGHTED -- R-157(2), the reference is not "
       "rewritten mid-comparison")
    ok(N_RANDOM >= 200, "declared null carries >=200 matched draws (rule 6)")
    ok(DECISION_METRIC == "net_cents",
       "the decision metric is NET, never harm share (rule 7)")
    ok(MULTIPLICITY_AFTER == MULTIPLICITY_BEFORE + len(WEIGHTED_ARMS),
       "multiplicity increments by one per SCORED candidate, 2 -> 4")
    ok("early_stopping" not in LGBM_PARAMS and "n_estimators" in LGBM_PARAMS,
       "LGBM capacity is PINNED with no early stopping -- nothing about arm C "
       "may be chosen after seeing the test side")
    ok(LGBM_PARAMS["random_state"] == 20260826,
       "the seed is pinned, so arm C is reproducible rather than a draw")
    ok(any("UNDERPOWERED" in o for o in DECLARED_OUTCOMES),
       "UNDERPOWERED-ON-BTC is declared as a NAMEABLE outcome before any "
       "number exists, so a thin btc result cannot be spun either way")
    print(f"phase2_declaration selftest: {checks} checks OK")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(selftest() if "--selftest" in sys.argv else 0)
