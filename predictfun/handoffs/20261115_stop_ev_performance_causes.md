# Causes of the STOP EV prediction and value gap

The largest demonstrated problem is underestimating losses when an order fills.
Changing only the value-head loss confirms that Huber contributes to the
optimistic predictions, but correcting that objective alone does not produce
profitable selections. Heavy through-price losses, worse test-period outcomes,
and unresolved execution-time assumptions remain material.

## Controlled objective comparison

Frozen code `c69a6431`, declaration `d36e022b`. Five already-consumed chronological
test dates, Oct 27/31 and Nov 1–3; **137,703 actions**, including 13,923 with
fills. Results as of **2026-11-15T14:45:39Z**. This is a post hoc diagnostic,
not untouched validation. There are two declared variants: original Huber
control and one squared-error value-head ablation. All inputs, labels, training
windows, fill models, seeds, rounds and other parameters are identical.

The original fill probabilities, filled-value predictions and their products
were reproduced **exactly on all five folds** before measuring the alternative.
Every fitted head was saved and reloaded with identical predictions. Four
corrupted-input controls were rejected. No thresholds or parameters were tuned.

| Metric on the same 137,703 test actions | Original Huber | Squared-error value head |
|---|---:|---:|
| Predicted mean value, cents/action | −1.416 | −2.709 |
| Realized mean value, cents/action | −3.401 | −3.401 |
| Average overprediction, cents/action | +1.985 | +0.692 |
| Rank IC | +0.134 | +0.126 |
| Actions with positive prediction | 9,727 | 17,558 |
| Realized local value of each positive set | −$55.41 | −$82.76 |

The loss change reduces average optimism by **65.2%**, and improves it on all
five days. That is a controlled effect of the objective on these predictions,
conditional on the existing labels. It is not a claim that Huber explains 65%
of strategy losses. The comparison does not select squared error for deployment:
its own positive set still loses, and its ranking is slightly weaker.

On the **original top decile**, fixed before comparing heads (13,771 actions),
predicted mean changes +0.435 → +0.248 cents while realized mean stays −0.527.
Only **19.4%** of that subset's optimism is removed. The average improvement
is much larger than the improvement near the decision region.

## Where the prediction error occurs

For the original 9,727 kept actions, 1,412 fill. The model forecasts +3.74 cents
conditional value on those filled actions; their realized average is −3.92.
Predicted fill rate is 16.33% versus 14.52% observed.

The exact residual identity is
`p * v - y = (p - filled) * v + filled * (v - y)`, with zero labels for no fill.
In this kept set, the mean optimism is +1.191 cents/action: +1.112 from the
filled-value term and +0.078 from the probability-weighted term. Thus **93.4%**
of this accounting decomposition lies in value after fill. The identity is
not an independent causal attribution to two separable error sources.

The economic components show why missing tail risk matters. On these same
9,727 actions, fills stopping at the order price contribute **+$85.12** while
through-price fills contribute **−$140.54**, leaving **−$55.41**. Across the
full 137,703 actions, at-price value is +$42.62 and through-price value is
−$4,725.88. The worst 1% of actions account for **50.5% of gross losses**;
within the original kept set, the worst 1% account for **61.6%**. These are
local one-second values, not independent portfolio trades or realized live P&L.

Huber limits the influence of large residuals, whereas the requested EV must
include their full economic contribution. The controlled fit supports this
mechanism. It also shows that a mean-target objective alone does not make the
available features/model reliably identify positive conditional means.

## Training and test outcomes differ

Every test fold has a more negative mean value conditional on filling than its
own training population. Training filled-order means range −22.07 to −25.30
cents; corresponding test means range −25.81 to −37.24 cents. Training fold
sizes are 92,242–199,598 actions; the five test-day sizes are
30,698 / 19,869 / 21,361 / 33,190 / 32,585. Exact counts, tail quantiles and
components are retained per fold in the receipts.

Test fill rates are lower than their training rates in all five folds, but
the fills that do occur are worse. This is observed outcome-distribution
change, not proof that input features are outside training support. Changes in
market conditions, conditional relationships, and timestamp reconstruction are
not separated by these marginal statistics. It is an additional reason an
average historical value estimate can remain optimistic after changing loss.

## Timing materially changes the measured target

The completed E2SEP artifacts were independently audited at
**2026-11-15T14:46:05Z**; their source report is as of 14:40:31Z. Across nine
consumed days, shifting estimated timestamps −250 / 0 / +250 ms gives:

| Scenario | Eligible actions | Mean local value, cents/action |
|---|---:|---:|
| −250 ms | 268,756 | −4.159 |
| Nominal | 265,373 | −3.053 |
| +250 ms | 266,011 | −2.147 |

This is substantial sensitivity within the replay, but the action populations
change. These are not paired causal estimates, physical error bounds, or proof
that the true timestamp is +250 ms. E2SEP's future-WS group still permits prior
E2 queue effects, as the earlier synthetic review demonstrated.

All zero-shift actions match the primary labels by exact key/time/status/value.
All **18,911 strict-certified nominal actions** retain their action/time and
label value under both shifts; three planted audit corruptions are refused.
The strict certificate behaves as intended in this scenario test.

Among the five test days, the selected strict-certified subset (11,574 actions,
421 fills) has mean +0.080 cents/action and original-model IC +0.015. The
estimated-only certified subset (70,056 actions, 7,199 fills) has mean −3.894
and IC +0.136. The uncertified subset (56,073 actions, 6,303 fills) has mean
−3.503 and IC +0.143. These different populations are descriptive; they do not
establish that E2 timing caused all losses or that the strict subset is tradable.

## Implications and remaining work

The current rank metric rewards recognizing bad exposure even when every broad
prediction bucket has negative mean value. It does not certify a calibrated
zero threshold or profitable quote retention. The model's local one-second
target also does not price future queue value beyond that horizon; this study
does not measure the effect of that omission or the complete cancel/repost chain.

Prioritize a mean-preserving value target with explicit tail-loss diagnostics,
timing-qualified training/evaluation, and calibration checked near the keep
threshold by day and side. Test the full keep/pull policy only after these local
estimates are credible. Simply switching to squared error, adding quote size,
or forcing a lower pull rate is not supported as a solution by this experiment.

Evidence: `reviews/stop_ev_causes_20261115/report_v1.json`,
`interpretation_v1.json`, `timing_v1.json`, fold receipts, and run logs.
Versioned fitted heads and predictions are under
`data/predict_fun/research/stop_ev_causes_20261115_v1/` (excluded from git).
Peak RSS was 620 MiB for the 77-second controlled study and 415 MiB for the
two-second timing audit, both on the Codex slot. All source/input hashes were
checked before/after. The active model/policy pipeline and original artifacts
were not changed, and no raw or protected-day tape was opened.
