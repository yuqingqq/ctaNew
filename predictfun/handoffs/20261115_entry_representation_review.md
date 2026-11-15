# Entry representation: conditions exposed, timing improvement remains one-sided

Implemented and tested a representation that preserves absolute liquidity and
own risk while exposing relative queue pressure, history coverage, volatility
and fair-price response. Bid timing improves modestly on every historical test
day; ask timing worsens on every day. The input representation is more explicit,
but it does not yet reliably separate economically good POST and WAIT actions.
No policy or model was promoted.

Completed **2026-11-15T05:04:52Z**. Evaluation: **9,871 action probes**, 4,990 bid
and 4,881 ask, across 1,434 windows on five consumed dates: Oct 27/31 and Nov
1/2/3. Training uses earlier dates only. The full label population is 13,911
probes / 2,010 windows on Oct 25/26/27/31 and Nov 1/2/3. Development only; no
untouched validation, confidence intervals or causal claim.

## What changed and what different conditions mean

| Condition to distinguish | Representation |
| --- | --- |
| Same queue size in thin versus deep liquidity | Public quantity divided by known liquidity, with absolute liquidity retained |
| Same relative queue in different absolute liquidity | Original log-liquidity and absolute top-size logs retained |
| Own large/small position or order in a changing public book | Own quantities, inventory and exposure stay in absolute risk units |
| Same Binance return under low/high volatility | Side-aligned return divided by known volatility and square root of horizon; raw return and volatility retained |
| Same return near/far from the contract threshold or at different expiry times | Existing implied-fair sensitivity translates recent return into price ticks; candidate edge and expiry context retained |
| Zero observed change versus missing candidate history | Missing values remain missing; explicit availability and zero-size statuses |
| Short, sparse or stale history versus a fully observed quiet period | Actual span, available duration, frame count, maximum gap, full-history status and last-frame age |
| Bid versus ask implications | Separate models and side-aligned price/return features |

The fair-price response is a transform of recent information under the existing
fair-value assumption, not an observed future movement. Book shrink remains a
public-size proxy; it does not identify trades versus cancellations.

Three fixed arms isolate the changes:

- **RAW, 325 features:** exact saved ALL models from the information study.
- **RELATIVE, 325 features:** 47 public-size fields re-expressed relative to
  causal liquidity. Each original size is recoverable from its ratio, the
  recorded divisor and retained scale. All other fields stay unchanged.
- **CONDITION, 351 features:** RELATIVE plus 26 explicit condition features.
  This bundle adds history support, volatility units and fair-response context;
  the experiment does not attribute its result to an individual added field.

No clipping, imputation, empirical constant/duplicate deletion or fitted scaler
is used. The old cleanup's unweighted scaler therefore does not enter this
comparison. The existing frame-weighted liquidity scale is unchanged; its
actual support is now visible. Raw labels, sampling weights, rows, chronological
folds and the original pinned LightGBM parameters/rounds are unchanged.

## WAIT-versus-POST performance

Primary target: POST-now minus WAIT under the frozen neutral-KEEP continuation.
Positive numbers mean percentage reduction in weighted squared error against
RAW; they are not R2 percentage-point changes.

| Representation | Bid error reduction | Bid days better | Ask error reduction | Ask days better |
| --- | ---: | ---: | ---: | ---: |
| RELATIVE | +0.36% | 3/5 | -0.73% | 1/5 |
| CONDITION | +1.43% | 5/5 | -0.95% | 0/5 |

CONDITION improves bid timing error by a further 1.08% relative to RELATIVE;
it worsens ask timing by a further 0.21%. This comparison concerns the complete
26-feature condition bundle. Smaller train/test marginal distribution shifts
from the earlier audit did not translate into a general improvement.

All three arms remain worse than both simple baselines on each side. CONDITION
has timing R2 **-0.07962 bid / -0.03756 ask** against the earlier training mean;
its error is **7.33% / 3.58% higher** than the price/rung baseline. A consistent
relative improvement on five consumed days is a development lead, not
validation or evidence of profitable timing.

For the secondary absolute-POST target, CONDITION worsens bid error by 0.38%
and improves ask error by 1.30%. This opposite pattern is why absolute POST
value cannot substitute for the WAIT-versus-POST comparison.

**0/6 local economic checks pass.** Every arm/side uses 500 draws matched on
day, side, hour, rung, initial/later generation, selected action count and
sampling-weight mass; Bonferroni adjustment covers six checks. All selected
absolute-POST means are negative. No portfolio simulation or promotion follows.
The separate cancel/keep model's benefits are not included in these labels.

## Data coverage and condition-level diagnostics

The following counts use the same 9,871 evaluation actions and as-of above.
They describe overlapping conditions, not exclusions:

| Observed condition | Actions | Fraction of evaluation rows |
| --- | ---: | ---: |
| Incomplete nominal 30-second liquidity history | 7,857 | 79.60% |
| Unknown start reference for fair-value features | 4,758 | 48.20% |
| Unavailable complete 250 ms candidate-price history | 2,031 | 20.58% |
| Later-generation entry | 1,323 | 13.40% |

The actual history measurement refines the previous lower bound of 7,815 rows
before 25 seconds. The additional 42 rows also lack full history. Short history
is now encoded; it has not been silently stretched into 30 seconds.

Later-generation rows carry **71.23% of inverse-inclusion weight mass** despite
being 13.40% of rows. Weighting corrects the sampling design but cannot create
more distinct examples. This remains a substantial support limitation.

All predefined condition cells are retained, including unfavorable ones:
initial/later entry, full/partial history, known/unknown reference and candidate
history, and training-weighted-median liquidity, volatility and fair-distance
bins. Cuts use earlier training data only. Each cell reports n, effective n,
weighted realized/predicted means and error. Bid improvements occur in both
liquidity bins and both history-coverage groups; ask deterioration also spans
both, so it is not confined to a single such condition.

These diagnostics expose different states and remaining calibration errors.
They do not prove stable separation of favorable versus unfavorable actions.
No condition was selected as an entry rule, and no target/outcome bins were
used to filter the population.

## Population accounting and verification

Across the seven label days, statuses remain: 13,911 eligible, 81,584 not
sampled, 344 probe-not-reached and 251 actions in six sealed-lookback windows.
Across the five evaluation dates: 9,871 eligible, 56,142 not sampled, 235
probe-not-reached and the same 251 sealed-lookback actions. No extra exclusion
was introduced by this study. Source evaluation receipts retain 8,361 untimed
legs / 105,159.20 shares; no missing feature-grid rows. Untimed legs are source
execution-data limitations, not silently discarded prediction rows.

- Seven synthetic checks pass, including planted regime/signal responses and
  invalid-schema/clock/denominator/draw refusals.
- 2,010 historical future-perturbation checks pass. Every original input and
  label is exactly preserved; liquidity support reconciles to each saved scale.
- Forty new model files and twenty saved RAW controls verified at the artifacts.
  RAW predictions reproduce exactly to 1e-12 absolute tolerance.
- All jobs use the Codex light slot; peak worker RSS 859 MiB. Completed service
  exits 0 at 05:04:53Z. The older size verification completed successfully; its
  supervisor resumed after the first attempt and has exited.
- First attempt stopped on a pandas comparison assertion in a synthetic checker,
  before any historical build or fit. It is preserved. The fix changed no
  representation, target or candidate definition.

## Implication and next research

Keep the explicit distinctions and data-support statuses available for research,
but do not adopt CONDITION as a shared bid/ask timing solution. The measured
bid improvement is small, ask timing worsens, and both still lose to simple
baselines. More intelligible representation alone has not solved prediction.

The next component study should use these distinctions to investigate fill
opportunity, adverse movement and queue advantage separately, with side-specific
results and the actual intended keeper accounted for when defining combined
entry value. Improving support for later-generation states also remains
necessary. The component study itself has not been started here.

## Evidence

Implementation: `research/pf_entry_representation.py`; runner:
`research/model_entry_representation_runner.py`. Initial source `ee9f0c1d`,
checker correction `a6ee5803`; authoritative freeze `2523293e` /
`reviews/entry_representation_20261115/freeze_v2.json`. Original freeze and first
attempt logs remain as provenance.

Current receipt: `reviews/entry_representation_20261115/current.json`.
Machine results: `result.json`, `scores.csv`, `fold_scores.csv`, `conditions.csv`
and `condition_summary.csv` in that directory. The condition summary aggregates
every predefined timing cell: weighted means use sum(mean * weight_mass) /
sum(weight_mass), SSEs sum, and effective n is mass squared divided by
sum(fold_mass squared / fold_effective_n). Counts sum across the five dates.

All 86 output artifacts are bound by `artifacts.json`; raw rows/models/predictions
remain under ignored data at
`data/predict_fun/research/entry_representation_20261115_v2/`. The full raw-source
and saved-control bindings are in the authoritative freeze. No large datasets
are added to git.
