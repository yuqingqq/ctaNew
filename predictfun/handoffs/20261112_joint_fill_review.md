# Coherent fill/value model — completed review

**The forecast-consistency gap is repaired. Earlier-day calibration improves aggregate fill estimates, but this candidate does not improve pull selection. No strategy is promoted.**

As of **2026-11-12T10:11:07.501416+00:00**, the evaluation covers **40,059 eligible neutral actions on six consumed UTC dates**: 20261026, 20261027, 20261031, 20261101, 20261102, 20261103. These come from 90,283 submitted generations; other statuses are ended_before_probe_or_unknown=38,038, postonly_refused=12,172 and missing_label_mark=14. The receipts preserve all statuses, including the earlier build days. There is no untouched-validation claim; the inherited LODO guard-feature caveat remains.

This follows the [independent-component study](20261112_fill_value_review.md). The objective remains default quoting to acquire and retain useful queue priority, with withdrawal when avoidable adverse fills outweigh staying. This study repairs the estimator while holding its existing short-horizon labels fixed. It does not restore an entry gate or change live trading.

## What changed

Fill probability, quantity, squared quantity, favorable value and adverse value now come from a **single shared distribution of paired KEEP and RESET outcomes**. Every eligible action, including no-fill outcomes, contributes. Both branches use the same nonnegative outcome weights; quantity and value are no longer independently fitted and then clipped into partial agreement.

The implementation uses the pinned scalar LightGBM model to form tree partitions. It averages empirical paired-outcome distributions within the reached leaves, with a fixed 1% training-prior mixture. The raw boosted regression score is not used as the component forecast. Probability/quantity/value bounds are therefore preserved by construction. Positive and negative value components describe the positive and negative parts of aggregate branch markout, not gross values of individual fills.

For each outer test date, the last preceding consumed day is reserved for calibration. The partition model and leaf distributions use only earlier core days. Calibration fits positive joint-event multipliers on that preceding day and applies them to the whole distribution, including both branches and all moments. The model is not refitted afterward. This prevents independently calibrated components from contradicting one another.

All **160,236 joint forecasts and 320,472 branch forecasts** passed the declared support checks, with **zero violations**. All 12 calibrators had supported outcomes and converged. Saved model, distribution and calibrator artifacts reconstructed the forecasts. This establishes internal mathematical consistency under the simulator labels, not live queue accuracy or profitability.

## Calibration helps the averages, but conditional errors remain

All rows below use the same 40,059 actions per branch and the as-of above. KEEP has 10,835 filled actions; RESET has 3,171.

| Quantity | Observed | New queue model, raw | New queue model, calibrated |
|---|---:|---:|---:|
| KEEP fill rate | 27.05% | 30.70% | 27.49% |
| RESET fill rate | 7.92% | 8.98% | 8.01% |
| KEEP total filled shares | 49,661.08 | 56,630.92 | 50,712.15 |
| RESET total filled shares | 14,698.70 | 16,559.17 | 14,771.68 |

The calibrated-versus-raw KEEP Brier difference is −0.000723, with a 95% day-bootstrap interval of [−0.001616, −0.000206]. RESET improves only slightly: −0.000018 [−0.000043, +0.000015], which does not establish a reliable RESET calibration gain. Raw and calibrated versions share the exact same core models, so this comparison isolates the calibration step.

Good aggregate rates do not imply calibration in individual states. In the calibrated queue KEEP model, the 10–20% bin has n=1,754, predicts 19.22% and observes 213/1,754=12.14%; the 30–40% bin has n=11,086, predicts 32.77% and observes 4,327/11,086=39.03%. These fixed-bin diagnostics show remaining conditional errors. The positive/negative value components have not been separately established as calibrated.

The new calibrated model's KEEP Brier is 0.192499 versus 0.185752 for the previous independent queue model: worse by +0.006747 [+0.004883, +0.008202]. RESET improves from 0.073523 to 0.072806: difference −0.000717 [−0.001145, −0.000281]. This is a mixed result, not a general predictive improvement.

**Those older controls trained on one additional preceding day**, which the new model reserves for calibration. Comparisons with them assess the available candidates, but cannot attribute the difference solely to coherent versus independent architecture. A same-core-data comparison would be needed for that attribution. The coherence constraint itself is not evidence that weaker KEEP discrimination is unavoidable.

Within the new model, adding queue proxies improves calibrated KEEP Brier versus market-only features by −0.001177 [−0.001505, −0.000928]. The RESET difference is −0.000003 [−0.000009, +0.000004]. The queue proxies remain useful for existing-order fill opportunity in this simulator; they are observed depth/progress features, not exact venue queue positions.

## Pull decisions have not improved

The calibrated queue policy selects 34,769 pulls and 5,290 keeps among the 40,059 neutral actions. This is an action-sample diagnostic, not a cancellation rate measured in a sequential live run.

Comparisons below use the same pull count within day, side, hour, age and inventory-sign strata. Values are summed local counterfactual utility in dollars equivalent, **not portfolio P&L**; their horizons can overlap.

| Comparator at matched pull counts | New calibrated queue increment | 95% day interval |
|---|---:|---:|
| Matched random pulls, analytic expectation | +23.05 | [+10.09, +33.54] |
| Original direct queue model | −6.15 | [−20.40, +7.98] |
| Previous independent components | −0.94 | [−14.14, +11.94] |
| Saved guard value-margin ranking | −11.93 | [−18.72, −4.95] |
| New raw queue model ranking | 0.00 | [0.00, 0.00] |

The guard comparator is its value-margin ranking, not the complete guard strategy. Intervals resample the six UTC days 1,000 times. The separately declared random null has 500 matched shuffles. Its finite-sample mean differs slightly from the analytic expectation used above. These conditional comparisons do not remove historical selection or create untouched validation.

On the 5,290 kept actions, predicted KEEP-minus-RESET value totals **+19.89 dollars equivalent**, while realized local value is **−27.62**. Overall action-value RMSE is 8.6453 cents, versus 8.6332 for the original direct model. Calibration corrects much of the fill-rate bias but supplies no measured gain in matched pull-selection value. Predicting the amount likely to fill is still insufficient to identify which fills are worth keeping.

The KEEP/RESET fill-rate difference is not a pure queue-priority effect: RESET also waits one second before placement and has less time to fill within the local horizon. This experiment does not separate that delay cost from priority loss.

## Gap register and next work

| Gap | Current state | Remaining work |
|---|---|---|
| Own-order fill opportunity and expected quantity | Implemented and evaluated on neutral actions | Improve conditional accuracy, especially after re-entry; useful signal is not yet live-calibrated. |
| Probability/quantity/value dependence | Structural consistency repaired and checked | Preserve this consistency in subsequent estimators without assuming this leaf-mixture architecture is optimal. |
| Calibration | Earlier-day joint calibration implemented | Aggregate bias improves; state-dependent fill errors and favorable/adverse value calibration remain. |
| Fill timing and incoming executable flow | Not implemented in this stage | Model opportunity over time and price-level flow/depletion, using observable information and explicit censoring. |
| Value of retaining priority beyond the local horizon | Open | Extend the paired continuation target with a predeclared horizon, or validate a terminal value covering only the tail. Do not double-count queue value. |
| Re-entry semantics | Open | Align label and sequential replay price, recovery conditions and latency; current same-price reset labels differ from current-touch replay re-entry. |
| Complete trading value and inventory risk | Not tested for this candidate | Replay the final aligned policy; report trading, inventory and settlement separately. Local utility is not portfolio P&L. |
| Live queue/fill calibration | Open | Reconcile against a new logging-build capture containing wallet events. Mathematical consistency does not close this data gap. |

The next target-level study should align KEEP/PULL continuation and re-entry semantics before judging whether early resting pays through later good fills. Declare the horizon, price/recovery rules, matched controls and complete action population before generating labels. Reconcile fill tranches with the engine and include cancellation latency, unfilled actions and censored endpoints. Keep architecture, features and horizon comparisons separate so their effects are interpretable. No further variant or threshold was selected from this result.

## Verification and provenance

Source freeze: `e8693557`; declaration commit: `460ff94f`, declared at 2026-11-12T10:10:00.995869+00:00 before fitting. The experiment contains 12 partition models, 12 calibrators, two feature sets and four fixed zero-threshold local policies. There are no new sequential strategy variants. All dates are consumed research data.

Controls cover valid paired-outcome reconstruction, convex mixtures, known calibration target movement, unseen-event refusal, impossible quantity/value moments, negative weights, chronology, future features, corrupt saved forecasts, exact action identity, model reloads, day intervals and matched-count rankings. Source/input/model hashes and shared receipt copies are checked for publication. The run exited successfully in 66.58 seconds, with peak RSS 884 MiB.

Full metrics and fixed-bin calibration are in [result.json](../reviews/joint_fill_20261112/result.json) and the [numerical report](20261112_joint_fill_results.md). The publication receipt binds the review, reports, frozen source and data artifacts. Code and small receipts are on `research/predict-fill-value-20261112`; model/prediction files remain outside Git. Previous frozen results are preserved, rebates remain separate, and the loss-budget setting is unchanged.
