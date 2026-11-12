# Fill opportunity and resting quote value — completed first-stage review

**Result: retained-order fill opportunity has measurable predictive signal in this simulator, but the new component model does not improve keep-versus-repost valuation. No candidate is promoted.**

As of **2026-11-12T09:15:02.311650+00:00**, all 120 models, six expanding folds, saved-prediction checks and reports are complete. Population: 40,059 eligible neutral actions across 20261026, 20261027, 20261031, 20261101, 20261102, 20261103, out of 90,283 submitted neutral generations. Other statuses: ended_before_probe_or_unknown=38,038; postonly_refused=12,172; missing_label_mark=14. All statuses remain in the receipts. These are consumed historical dates, with inherited LODO guard features; there is no untouched validation or live fill-calibration claim.

The objective remains default quoting to acquire and retain useful priority. This test changes prediction architecture while keeping the existing labels fixed. It is not an entry-permission test and not a full portfolio replay. Source freeze `4ac8a3b6`, declaration `ebeedcbe`; original frozen strategy and previous results remain unchanged.

## Fill opportunity

| Branch | Model | Filled / eligible | Mean predicted fill probability | Observed fill probability | Brier error (lower is better) | Quantity RMSE, shares |
|---|---|---:|---:|---:|---:|---:|
| keep | baseline | 10,835 / 40,059 | 29.14% | 27.05% | 0.192991 | 2.0809 |
| keep | market | 10,835 / 40,059 | 29.84% | 27.05% | 0.188653 | 2.0574 |
| keep | queue | 10,835 / 40,059 | 29.30% | 27.05% | 0.185752 | 2.0429 |
| reset | baseline | 3,171 / 40,059 | 8.92% | 7.92% | 0.073005 | 1.2791 |
| reset | market | 3,171 / 40,059 | 8.46% | 7.92% | 0.073609 | 1.2826 |
| reset | queue | 3,171 / 40,059 | 8.42% | 7.92% | 0.073523 | 1.2819 |

For KEEP, the queue model lowers Brier error by 3.75% versus the training-only state baseline. Its Brier difference is −0.007239, with day-bootstrap interval [−0.009173, −0.005145]. It also improves over the same model without the four queue proxies: difference −0.002901 [−0.003539, −0.002197]. KEEP quantity error improves as well. This supports useful information in the observed queue/depth features for predicting **simulated fills**, conditional on this test design.

RESET is weaker: the queue model has worse Brier error than the baseline, difference +0.000518 [+0.000105, +0.000969], and worse quantity squared error. Extra queue features do not produce a clear RESET improvement over the market model. The two branches therefore do not yet provide equally reliable inputs to the action comparison.

The baseline uses only previous training days, with fixed shrinkage of state-stratified means; it does not use a test-day fill rate. State strata include side, order age, remaining quantity and displayed depth. All intervals resample six UTC days 1,000 times. Historical selection is not removed by these intervals.

The KEEP/RESET fill-rate gap is not a pure measure of queue priority. RESET sits out for one second before placement and loses priority; it has less available time inside the same short label horizon. Both effects contribute.

## Calibration still limits usefulness

KEEP predicts 52,344.27 shares versus 49,661.08 observed, a 5.40% overestimate. RESET predicts 14,489.26 versus 14,698.70. These are sums over the same 40,059 actions per branch, including unfilled actions.

Higher predicted probabilities are overconfident. For the queue KEEP model, the 60–70% bin has 636 actions: mean prediction 63.33%, observed 321/636 = 50.47%. For RESET, the 20–30% bin has 673 actions: mean prediction 22.75%, observed 63/673 = 9.36%. These are descriptive fixed bins, not a reason to choose a new threshold on the same data.

## Better fill prediction did not make better pull decisions

The component queue model chooses 29,577 pulls. At identical pull counts and matched day/side/hour/age/inventory-sign strata, its summed local utility is -9.07 dollars equivalent versus ranking with the original direct queue model, with day interval [-19.78, +2.45]. Versus the saved guard value-margin ranking it is -11.09 [-20.59, -2.12]. The latter is only the guard's value component, not its full entry/cancellation policy.

On the 10,482 actions it keeps, predicted KEEP-minus-RESET value totals +73.95 dollars equivalent, while realized local value is -43.31. Keep calibration is still poor. Overall value RMSE is 8.6518 cents per action versus 8.6332 for the original direct queue model; the paired squared-error difference is +0.321568 [+0.239935, +0.426527].

Both new component policies beat their matched random-pull nulls (500 draws, conditional p=1/501), but neither improves on the existing direct model at matched counts. Local utility includes the frozen inventory penalty and alternative fill markouts; it is not realizable portfolio P&L and must not be added to trading or settlement totals.

The queue model overpredicts the positive part of branch net markout: KEEP 451.18 versus 400.92 dollars equivalent; RESET 114.76 versus 86.13. These are positive parts of aggregate branch outcomes, not a count of individually favorable fills. Aggregate bias alone does not identify the cause of each wrong decision.

## Independent components are not yet a coherent joint forecast

The model fits probability, first/second quantity moments and positive/negative markout as separate regressions on all actions. The declared constraints enforce necessary probability/quantity moment bounds, but do not ensure joint consistency with the value heads. The supplementary audit found physically inconsistent rows:

| Model / branch | Actions | Any price/quantity/value support violation | Zero quantity with nonzero value |
|---|---:|---:|---:|
| baseline / keep | 40,059 | 11 | 0 |
| baseline / reset | 40,059 | 9 | 0 |
| market / keep | 40,059 | 228 | 205 |
| market / reset | 40,059 | 796 | 675 |
| queue / keep | 40,059 | 266 | 235 |
| queue / reset | 40,059 | 850 | 708 |

For example, at a known price and bounded binary payoff, expected positive/negative markout cannot exceed the corresponding maximum per-share movement times expected filled quantity. Predicting no quantity but positive markout value violates that requirement. These violations are reported without repairing the frozen forecasts. An artifact-integrity PASS means calculations and identities check out, not that forecast support or calibration passes.

Moment constraints were also frequently active: the queue fraction prediction was adjusted on 14,753/40,059 KEEP and 20,032/40,059 RESET actions; its second moment on 33,239/40,059 and 36,199/40,059. This is a limitation of independently fitted components. It does not invalidate the measured probability scores, but it prevents treating this collection of outputs as an established joint economic model.

## What to do with the finding

The useful result is that the observed state contains some information about when an existing resting order will fill. The unresolved parts are opportunity after re-entry, probability calibration, and the value/quantity dependence of those fills. Predicting a fill is not sufficient to identify a profitable fill.

The next model design should enforce coherent joint outcomes for no fill, partial/full fill and their value, and validate calibration using earlier training/validation data only. The later-queue-value study should then use a predeclared longer continuation horizon and re-entry semantics matching the replay, with price-level incoming flow and queue depletion evaluated where observable. Changes to architecture, features and horizon should have separate declared comparisons; do not tune a new threshold from this result.

This stage did not forecast raw incoming volume or exact fill time. It did not measure terminal queue value, repair the same-price-label/current-touch-re-entry mismatch, or validate simulation against a live capture. Those remain separate tasks. No extra numerical variant was launched after this readout.

## Verification and publication

Every head used all eligible actions, including zeros; no filled-only training, rebalancing or test-label calibration. All 120 saved models were reloaded and reconstructed; old direct scores matched on exact action identities. Original source/input hashes, chronology and label/accounting identities pass. Positive/refusal controls cover probabilities, quantities, duplicate/overfilled actions, future features, day intervals and matched rankings. The numerical run completed in about 305 seconds with peak RSS 602 MiB.

The supplementary physical-support check was prepared during fitting and declared before the numerical report completed or its values were inspected. `audit_declaration_v2.json` corrects only the original timing description: fit finished 09:14:50 UTC, audit declaration 09:14:55, report 09:15:02. The audit source, bounds, forecasts and results were unchanged. Original metadata is retained.

Canonical result: `../reviews/fill_value_20261112/result.json`, SHA-256 `1872956f050bd22e3e976db30db9172359c129825aaab4674152df3991dd5376`. Audit: `audit.json`; full metrics and calibration: [numerical report](20261112_fill_value_results.md). Source, declaration, receipts and this review are on `research/predict-fill-value-20261112`; large models and predictions remain outside Git.
