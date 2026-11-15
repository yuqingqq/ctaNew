# Entry repair results and remaining performance gap

The controlled coverage and training-history study finished at 2026-11-15T08:34:59Z.
More representative sampling improves prediction modestly, but none of the
models establishes useful economic entry selection. A short training window
alone is not supported as the explanation: on the same evaluation actions,
using only the latest two compatible training days slightly improves pooled
timing error relative to using all earlier compatible days.

Evaluation contains 9,871 unchanged actions: 4,990 bid and
4,881 ask on five consumed dates, October 27/31 and November 1/2/3. Earlier days
alone train each fold. This is development evidence, not untouched validation.
No confidence intervals, portfolio acceptance or model promotion are claimed.

## Coverage and timing performance

Eligible labeled actions increased from 13,911 to
27,669 across seven days and 2,010 windows. Every original
eligible action retains its target, features and probe timestamp. Initial
sampling stays 1/2; later sampling increases 1/32 to 1/4 while original probe
ages are preserved. Dense training uses weights 2/4; the unchanged evaluation
and sparse control retain weights 2/32.

Full population statuses: not_sampled: 65,889, eligible: 27,669, probe_not_reached: 2,281, sealed_lookback_window: 251.
Six sealed-lookback windows remain excluded. These row statuses are distinct
from untimed source-trade limitations recorded in the input receipts.

All arms use the same 351-input CONDITION representation, 200 boosting rounds
and pinned learner parameters. Bid and ask are separate. There is no learner
capacity change in this study.

| Training arm | Bid timing error reduction vs sparse | Ask timing error reduction vs sparse | Bid timing R2 vs prior mean | Ask timing R2 vs prior mean |
|---|---:|---:|---:|---:|
| Sparse, all available history | 0.00% | 0.00% | -0.0796 | -0.0376 |
| Dense, all available history | 2.27% | 2.38% | -0.0557 | -0.0134 |
| Dense, latest two dates | 2.76% | 3.49% | -0.0501 | -0.0004 |

Error reduction compares weighted squared errors on identical test actions.
Each R2 uses that arm's own training mean; it is not the same denominator across
arms. The sparse control exactly reproduces the earlier saved CONDITION model.
Denser all-history training improves timing error on four of five bid days and
two of five ask days, and remains worse than its price/rung baseline on both
sides in aggregate.

Latest-two-day training improves pooled timing MSE by 0.50% bid and 1.14% ask
versus dense all-history training. The first fold has identical histories; of
the four nontrivial folds, each side improves on two and worsens on two. The
pooled improvement therefore does not establish persistent drift or an optimal
window. All-history training is better for absolute POST value than recent-two
training. More distant, diverse calendar history remains untested.

## Economic selection still fails

Zero of six arm/side checks pass. Each uses 500 matched draws preserving day,
side, hour, rung, initial/later status, action count and sampling weight mass;
Bonferroni covers six checks. Every selected-action mean for absolute POST
value is negative. Recent-two ask timing nearly matches a constant mean and
slightly beats its price/rung baseline, but neither implies profitable entries.

For dense all-history training, the 854 selected bid actions have average
POST-minus-WAIT value -0.678 cents per posted share and absolute POST value
-1.623 cents; the 807 selected ask actions have +0.889 cents relative value and
-1.571 cents absolute value. These are inverse-probability weighted local
counterfactual values per submitted share, not realized portfolio P&L or the
per-filled-share metric in the sizing experiment. Ask relative value does not
beat its matched control. Favoring POST over WAIT can still favor one losing
action over another under this continuation.

## What explains the poor performance

The latest large-size losses came from neutral POST/KEEP: the entry callback
always posts when legal, and the selective keeper is absent. Larger size
retains order priority longer but also exposes more inventory and makes the
fixed cap bind much more often. Negative terminal marks are consistent with
adverse selection and inventory risk; this report does not separate their
short-horizon and end-of-window contributions. These losses do not measure the
combined learned entry plus selective-cancel strategy.

The prediction task also remains broad. POST and WAIT are valued through the
window terminal under a frozen bootstrap entry continuation and neutral KEEP.
The label includes future replacements and other slots' inventory/cap effects.
The prior first-fold audit found substantial variation from those external
components; the present density test does not newly identify a noise floor.
Market information that predicts a short-term price move need not predict this
whole continuation-value difference accurately.

The biggest remaining target-design gap is that the intended strategy uses
selective cancellation, while entry labels still value neutral KEEP. Holding
that continuation fixed was deliberate for the density/history comparison; it
must not be presented as alignment with the intended combined strategy.
Overfitting or insufficient regularization remains plausible from prior audits,
but this study did not change model capacity and cannot attribute the residual
error to that alone. Likewise, larger samples from the same dates do not create
new regimes, true queue observations or jointly waiting policy states.

The next useful model experiment should preserve this dense-data control,
price entry under an explicitly frozen intended keeper, and distinguish
near-term order outcomes from downstream continuation effects. Then test
regularization against simple baselines with a separate freeze. Do not remove
rare adverse outcomes or filled/unfilled actions to manufacture cleaner labels.
This is the proposed next experiment, not an implemented or promoted model.

## Verification and current state

All seven builds and five fits exited successfully. The completed report's
five receipt identities, all 60 saved model hashes, prediction hashes, training
receipt links and original feature/target reconciliation were checked at their
artifacts. Forty-nine frozen source identities still match. The runner reloads
models to verify predictions and checks exact sparse-control reproduction.
Worker peaks: 859 MiB across full validity/build/fit/report stages; all below
the 2 GiB self-guard. The old owned action-depth driver resumed after completion.

Canonical pointer: `reviews/entry_repair_20261115/current.json`.
Full evidence: `completion.json`, `full_result.json`, `full_scores.csv` in that
directory. Earlier running checkpoints and failed smoke attempts remain as
provenance. Existing clock reconstruction, fill timing and unobserved queue
qualifications remain open. No live behavior changed.
