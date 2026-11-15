# Entry normalization review — 2026-11-15

There are normalization and representation gaps worth testing, but this audit
found no simple unit/sign error or mismatch between saved training and test
preprocessing. The strongest finding is that liquidity-relative public sizes
have much closer train/test distributions. That is not yet evidence of better
entry prediction, and does not establish why the previous models failed.

This is the user's requested review before the fill/adversity/queue component
study. That study has not started. No model was refitted, target changed, outcome
filtered, or policy promoted. Entry remains WAIT versus POST; cancel/keep is a
separate model.

## Population and scope

As of **2026-11-15T04:33:04Z**, the audit covers 13,911 eligible action probes
on seven existing label dates: Oct 25/26/27/31 and Nov 1/2/3. Evaluation findings
use 9,871 probes on five consumed dates, Oct 27/31 and Nov 1/2/3. These are
development data, not untouched forward validation. Existing exclusions remain
in the source receipts. No intervals or significance claims are made.

Two distinct input paths are inspected:

- The latest information study: 325 raw LightGBM inputs, including depth and
  history quantities expressed in fixed 30-share lots. It does not use the
  cleanup preprocessor.
- The earlier CLEAN_TREE/CLEAN_RIDGE comparison: economic transforms followed
  by train-only clipping, imputation, median/IQR scaling and empirical
  constant/duplicate removal.

Cleanup defects cannot directly explain the raw information study's failure.
The older cleanup also changed several things together; its poor performance
does not isolate the effect of normalization.

## Findings

### 1. Public size normalization is incomplete in the raw information study

Dividing shares by a fixed order size changes units. It does not distinguish
the same queue size in a thin book from that size in a deep book. Existing
causal liquidity features permit a state-dependent comparison.

For 18 public size inputs across five evaluation folds and two sides (180
feature/side/fold comparisons), dividing by the known liquidity scale reduced
the marginal train/test Kolmogorov–Smirnov distance in **178/180** comparisons.
The median distance was **0.1322 raw versus 0.0484 normalized**. Comparisons
overlap and are correlated; they are not 180 independent experiments.

For four Binance return horizons across the same ten fold/side cells (40
comparisons), known-volatility scaling reduced the distance in 29/40, with a
much smaller median change: **0.0343 to 0.0329**.

These are marginal distribution diagnostics using the observed action rows,
not weighted prediction scores. A feature can have a more stable distribution
while losing useful information about the target. No full 325-feature
normalization-only model comparison has been run. Retain absolute liquidity
alongside ratios when designing that comparison, so regime information is not
automatically discarded.

### 2. Cleanup scaling uses a different population from weighted model fitting

The earlier processor estimates clipping bounds, medians and IQRs without
inclusion weights. Model fitting/evaluation uses weights 2 for initial
generations and 32 for later generations. Across the ten training fold/side
cells, initial generations account for **84.31–86.71% of rows** but only
**25.14–28.96% of weighted mass**.

Weighted empirical medians differ from the fitted medians by as much as
**2.117 fitted scale units** (Nov 1 ask fold, `other_exposure_lots`). This is a
population-alignment concern, not leakage or proof that unweighted scaling is
invalid. Weighted clipping/scaling is a testable alternative if the weighted
action population is the intended population. Raw economic labels must remain
unchanged. A fixed affine rescaling alone is not expected to solve tree-model
failures; state-dependent ratios and clipping affect different things.

### 3. The nominal 30-second liquidity scale often has shorter history

**7,815/9,871 evaluation actions (79.17%)** occur before 25 seconds. The loader
retains at most five seconds of pre-window books, so these actions cannot have
a full trailing 30 seconds. Other actions may also have incomplete history;
79.17% is a lower bound for this concern. The median weights frames equally,
not elapsed time.

The scale is causal, but its effective support changes over the window. Record
actual history duration and frame count, and distinguish partial-history
estimates. Longer pre-window support should only be used where the source
contains causally available data. The current audit did not build a replacement
scale or measure prediction effects from one.

### 4. Empirical column removal loses some later outlier information

Among the 9,871 evaluation actions in the earlier cleanup path:

| Condition | Affected actions |
| --- | ---: |
| A removed missingness indicator becomes informative | 0 |
| A removed extreme-value indicator changes later | 15 |
| Columns identical during training diverge later | 14 |

The 14 duplicate-divergence actions overlap the 15 extreme-indicator actions;
do not add these counts. Observed divergences mostly concern `_high`/`_low`
flags, not wholesale collapse of the raw feature set. Error or economic
contribution from the affected actions was not measured.

Synthetic controls establish that discarded missingness flags can hide later
missing values, but that did not occur in these evaluation rows. Preserving a
status that was constant in training supports monitoring and fallback; it does
not teach the estimator how to respond to a state it has never seen.

## Checks that passed

All audited unit, side-sign, positive-denominator, reference-mask, clock-unit,
fill-fraction and target-reconciliation checks passed on the 13,911 probes.
All **10/10** saved cleanup preprocessing states reproduced exactly from their
original chronological training folds. Applying each transform to later data
left its fitted statistics unchanged. These checks found no train/test scaler
mismatch; they are not a blanket proof that every upstream feature is correct.

All five synthetic checks passed, including planted missing/extreme values,
diverging training duplicates, weighted-population changes, transform-state
immutability and unit-versus-economic scaling. Known-bad schemas and weights
were refused. Source/input digests were frozen before the historical audit.
The supervised audit exited 0, using 376 MiB peak RSS and 2.84 seconds; the
older size supervisor was resumed afterward.

## Recommended next comparison

Before changing targets or starting the component study, isolate normalization
with fixed rows, folds, targets, sampling weights and pinned model parameters.
First compare raw inputs with causal liquidity-relative public-size inputs,
retaining raw scale information. Keep own inventory/exposure in meaningful
absolute risk units. Separately test weight-aware preprocessing and explicit
history/support statuses if the cleanup path is retained. Do not bundle these
changes into one result and call the difference a normalization effect.

Measure bid and ask prediction loss against the same baselines and retain raw
economic errors. Any repeated historical comparison remains development work;
it cannot turn consumed dates into validation. Better distribution alignment
alone is insufficient to claim a better timing model.

## Reproduction and evidence

- Source commit: `3db21aec`; pre-audit freeze commit: `0bb344c6`.
- Entry point: `research/pf_entry_normalization_audit.py` (`selftest`, `audit`),
  supervised by `research/model_entry_normalization_runner.py`.
- Current receipt: `reviews/entry_normalization_20261115/current.json`.
- Compact results: `reviews/entry_normalization_20261115/summary.json`.
- Per-comparison evidence: `distribution_shift.csv`, `scaling_stats.csv` and
  `discarded_information.csv` in that review directory.
- Frozen sources and inputs: `freeze.json`; full-output digests: `artifacts.json`.
- The 1,463,703-byte full result remains under ignored data at
  `/home/yuqing/ctaNew/data/predict_fun/research/entry_normalization_20261115_v1/audit/result.json`;
  its SHA-256 is `e088b5cfdb5f722a4bbbc7f7b5b6a3288b63d2ffe904fdc878beb1782d45bcd9`.

Zero new candidates, zero historical model refits, no forward-validation or
economic-selection claim, and no model/strategy changes in this review.
