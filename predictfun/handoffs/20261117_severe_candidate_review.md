# Severe-probability regularization and calibration: review

Both proposed directions are justified as separate declared candidates. Start
with calibration as the simpler diagnostic of probability scale, and compare
one fixed smoother model separately. Neither has yet been fitted or shown to
improve economic value. Keep the existing controller and economic target fixed.

Reviewed `49d159e5` / `cbf48b89` and `a580eed4` / `2ec3e5c6`. Saved-head
reconstruction as of 2026-11-17T06:01:01.960Z covers 36,056 neutral actions
and 1,310 severe events across the four past-only Oct24–27 folds. All seven
probability bands reproduce the 05:01:32Z support report. Receipt:
`reviews/severe_candidate_review_20261117/receipt_v1.json`. The verifier checks
saved model hashes, populations, forward chronology, probability bounds and
positive/known-bad controls. No new calibration fit, regularization fit, support
distance recomputation or validation was performed in this review.

## Why both directions now make sense

Among 21,772 decisions with predicted severe probability below .02, there are
290 observed severe events versus 192.768 expected. Observed/expected is
1.41–1.54 across the four bands below .05. Above .1 it is .922 and .865
(n1,653 and n562). The error changes with predicted probability, so a single
multiplier is an inadequate description of the calibration problem.

The earlier running adjustment corrected aggregate counts but worsened severe
log loss on all four days. That is evidence against this particular noisy
hourly multiplication, not against calibration fitted on a separate earlier
sample. Strong in-sample separation and weaker forward calibration make
regularization a reasonable hypothesis; they do not prove that lack of
regularization is the only cause.

The support diagnostic improves on relative test-only quintiles, but its
conclusion remains limited to 15 model-selected features. The pooled kNN rate
being closer to the observed rate is not proof that kNN predicts better: it is
not a proper-score or economic comparison, and its day-specific biases differ
in direction. Preserve those qualifications rather than declaring coverage and
conditional drift eliminated.

## A small, interpretable candidate comparison

Use the unchanged baseline plus two separately declared candidates initially:

| Arm | Change | What it tests |
|---|---|---|
| Baseline | Existing severe head | Reference |
| Smoother | One fixed larger leaf minimum and smaller round count | Whether reduced complexity improves forward probabilities and useful ranking |
| Calibrated | Existing frozen severe head plus a small calibration map | Whether its scores can be mapped to more reliable probabilities |

Current parameters are `min_data_in_leaf=200`, 200 rounds, 31 leaves,
learning rate .05 and L2 regularization 10. A reasonable *proposed* single
smoother setting is 400 minimum rows per leaf and 100 rounds, with everything
else unchanged. This is an explicitly flagged parameter change, not a result-
selected winner or a grid search. The two changed parameters form one package;
this comparison would not separately attribute their effects.

For a first calibration-only diagnostic, use a low-parameter monotone logit map:

    calibrated_p = sigmoid(a + b * logit(raw_p)), b > 0

Declare numerical clipping and regularization of the map before fitting. Fit
with ordinary probability loss and the natural event prevalence; retain all
eligible no-fill actions. A positive-slope map preserves ranking by the severe
score itself. It can still change net-EV ranking because severity amounts and
other components vary across orders. It cannot create missing ranking signal.

Use a genuine chronological split: base model training, then calibration, then
later evaluation. The calibration day must be excluded from the base model's
training, and the same frozen base model must score calibration and evaluation.
If the base model is refitted, do not silently carry a map trained on different
scores; declare and evaluate that transfer protocol. All labels must be available
by the calibration cutoff. One held-out day may have unstable regime coverage;
more earlier chronological calibration predictions are an alternative to declare,
not a window to choose after seeing evaluation performance.

Do not combine smoothing and calibration in the first comparison unless the
combined arm is also declared in advance and counted in multiplicity. A later
combination needs fresh validation data if selected after this comparison.

## Additional constraint discovered in this review

The existing independently fitted heads violate
`P(loss > 30) <= P(loss > 0)` in 2,268 of 36,056 actions:

| Predicted day | n | Violations |
|---|---:|---:|
| Oct24 | 9,259 | 640 |
| Oct25 | 8,938 | 566 |
| Oct26 | 8,940 | 459 |
| Oct27 | 8,919 | 603 |

This is model-output incoherence, not a newly found label bug or an established
cause of the whole strategy loss. The probability calibrator or smoother cannot
be treated as a coherent joint risk estimate without addressing it. Declare
the consistency handling before the economic test and show its separate effect.
A nested/joint probability representation can enforce the subset relation;
post-hoc clipping must not be silently bundled with a claimed calibration gain.
The initial separate-head score comparison can still diagnose the options, but
passing log loss alone does not complete the economic-model repair.

## Acceptance and chronology

Assess probability-band observed/expected counts, ordinary log loss and Brier
score, with counts by date, side and action kind. Check preservation of useful
ranking, severe-loss amount calibration and full action-value calibration.
The downstream economic comparison must match market/side/action participation,
include missed gains and residual severe losses, and retain at least 200 declared
matched null draws. Better total event counts alone are not success.

Hold gain, ordinary-loss and severe-amount heads fixed for the initial severe-
probability ablation, together with clocks, size, inventory terms and controller;
any joint consistency correction is an additional explicit design component.
Oct24–27 are already consumed development dates. An exploratory screen on them
is not prospective validation. Freeze the candidates, calibration protocol and
acceptance before at least five later untouched complete days; inspect the
research ledger first because chronological later dates can also be consumed.
