# Paired action-value error decomposition

The largest measured posting-value gap is the joint response of fills and
adverse price movement, especially severe losses. Correcting fill quantity,
endpoint price or inventory cost separately does not reliably repair it.
The failed state-conditioned mapper improves quantity error but worsens this
joint economic term. Keep SOFT_JOINT as the baseline.

A second, narrower cancellation issue is confirmed: some orders have already
filled but their notifications have not arrived. Valuing the full client-known
remainder as still exposed overstates what a cancellation can protect. This
needs causal uncertainty about remaining exposure, not hidden exchange state.

Diagnostic source f6b59100, freeze6743804b; supplementary state inspection
ff999de8. No new predictor, controller, trading rule or continuous replay.
All main audits passed, including archived labels/predictions and canonical
prefixes. This is consumed-development diagnostic evidence, not a model upgrade.

## What is measured

For every order i, let delta_q_i be fills with the action minus fills with
keep/wait, s_i its bid/ask sign, p_i its price, m0 the known decision-time mid
and mH the common one-second endpoint mid. The exact action-value identity is:

    edge   = sum_i delta_q_i * s_i * (m0 - p_i)
    motion = sum_i delta_q_i * s_i * (mH - m0)
    risk   = -(position_action^2 - position_base^2) / 360
    value  = edge + motion + risk

Both edge and motion are split into focal-order and other-order contributions.
Removing one order can redirect the same public volume to other orders; this
is preserved explicitly. These are endpoint wealth terms, not independently
measured spread income or a one-second markout after each fill.

For a post, affected focal quantity is q_post-q_wait; for a cancellation it is
q_keep-q_cancel. The previous quantity metric used the keep branch's total
focal fills after decision time, including fills that cancellation could not
avoid. The new quantity measure excludes the unavoidable shared prefix.

The joint term matters: expected fill-weighted movement is E[q * signed_move],
which in general differs from E[q] * E[signed_move]. The diagnostic measures
that covariance in the actual scenario distribution. It also replaces one
component at a time with its realized counterpart, without claiming those
future-information substitutions are implementable forecasts or coherent
strategy counterfactuals. Risk and the other components stay fixed during
these deliberately local substitutions.

## Population and controls

Exactly the archived neutral actions:418 on Oct26 and443 on Oct27,12 fixed
markets per date, with no-fill actions included. Re-run both canonical branches
for every selected action and reconcile archived economic labels, order identity,
prices, fills and historical/unavoidable prefixes. Recreate the same causal
client order state and reproduce all ordinary predictions from SOFT_JOINT,
EXPANDED_JOINT and STATE_CONDITIONAL. The Oct27 true-coarse-pair channel also
reproduces the previously archived diagnostic. The actual future path with
client-known state is an internal execution/target control, not a forecast.

Seven frozen diagnostic substitutions: true edge, true motion, true risk,
true other-order wealth, independent focal fill/move averages, true endpoint
applied to predicted fills, and true affected focal quantity while preserving
predicted fill-weighted movement. The last reports missing support explicitly
when predicted affected fill is zero but actual affected fill is positive.
All comparisons include their same-support original error.

Severe retained-exposure loss is defined before results as below-30 cents:
post action value below-30, or cancellation benefit above+30. No loss clipping;
all signs are action-relative and include risk/other-order effects. Tail strata
are descriptive partitions only, never a training or selection population.
Nulls use200 matched positive-action-count permutations within market/side/kind.
Signed component contributions to total squared error include cross terms;
they are accounting attributions and can be negative, not causal percentages.

The same estimated clocks, queue assumptions,30-share ladder, cap180 and1s
zero-continuation objective remain. Oct26 calibrated the inherited classifier;
both dates are already consumed research data. No confidence intervals,
untouched validation, new strategy P&L or promotion claim. Protected dates
remain unopened. Keep SOFT_JOINT pending evidence for a new model.

## Where the errors sit

Reports as of2026-11-17T00:43:16Z (Oct26) and00:47:54Z (Oct27). The following
tables use the retained SOFT_JOINT map on all205/213 cancel/post actions on
Oct26 and221/222 on Oct27, with zero ordinary-evaluation exclusions. All418/443
identities and source exclusions are retained in the report receipts. There
are182/213 and197/222 posts with zero actual affected fill; cancellation has
190/205 and194/221. These zeros remain meaningful action exposures.

| Date / action | n | Total value RMSE | Edge RMSE | Motion RMSE | Risk RMSE | Severe loss n | Severe loss share of squared error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1026 / cancel | 205 | 8.449 | 15.958 | 19.486 | 2.937 | 4 | 34.3% |
| 1026 / post | 213 | 36.111 | 17.416 | 43.523 | 4.111 | 12 | 88.3% |
| 1027 / cancel | 221 | 31.826 | 14.478 | 39.328 | 4.133 | 9 | 74.6% |
| 1027 / post | 222 | 24.829 | 14.783 | 30.518 | 3.373 | 11 | 86.5% |

Value errors are cents per whole action. Component RMSEs are not additive:
edge and motion errors partly offset. Motion has the largest component RMSE
in all four populations, but that does not mean replacing it independently
must improve the total. The Oct26 cancellation exception below illustrates
this. The severe-tail concentration persists across both posting samples;
it does not establish that these outcomes were predictable from current inputs.

Other-order wealth error is much smaller in these ordinary tests:1.885/0.899c
cancel/post on Oct26 and1.158/2.634c on Oct27. Redistribution is real and must
remain in valuation, but it is not the largest observed error term here.

## Which hypothetical correction helps

Same populations/as-of as above, all supported for these ordinary substitutions.
Each column replaces only the named term; all other predicted terms remain.
These use future information and are diagnostic calculations, not achievable
strategy performance or error lower bounds.

| Date / action | n | Original | Actual focal quantity | Actual endpoint with predicted fills | Actual joint motion term | Actual risk term |
|---|---:|---:|---:|---:|---:|---:|
| 1026 / cancel | 205 | 8.449 | 8.230 | 10.574 | 15.693 | 7.670 |
| 1026 / post | 213 | 36.111 | 38.305 | 30.614 | 16.285 | 34.530 |
| 1027 / cancel | 221 | 31.826 | 30.726 | 26.791 | 13.971 | 30.746 |
| 1027 / post | 222 | 24.829 | 21.271 | 26.047 | 13.423 | 23.599 |

For posting, the realized joint fill/motion term removes much more error than
risk correction. Actual quantity alone worsens Oct26; actual endpoint alone
worsens Oct27. The unresolved quantity is therefore not simply a mean price
forecast or average fill forecast. It is how much fills in the particular
states where price moves adversely, at the focal queue and latency.

On Oct26 cancellation, supplying actual edge raises RMSE8.449->20.231c and
supplying actual motion raises it to15.693c: independent substitutions break
compensating errors. This is a reason to preserve coherent joint scenarios or
moments when learning the new response, not to keep known errors deliberately.
Decorrelating focal fills from motion also worsens ordinary SOFT errors on
both dates (cancel9.020/33.256c, post36.374/25.952c), though the response's
existing dependence is still far from sufficient. Full200-draw matched nulls
for every substitution are in the reports; none is a new candidate policy.

## Why the state-conditioned revision failed

Oct27 true-coarse-pair diagnostic, same as-of00:47:54Z:420 shared supported
actions,208 cancellations and212 posts,23 unsupported actual pairs per map
explicitly retained. Knowing the coarse outcome does not remove the error.

| Action | n | SOFT affected-quantity MAE | Conditional affected-quantity MAE | SOFT motion RMSE | Conditional motion RMSE |
|---|---:|---:|---:|---:|---:|
| cancel | 208 | 2.497 | 2.187 | 32.954 | 36.508 |
| post | 212 | 3.875 | 3.403 | 29.452 | 31.718 |

Quantity MAE is shares; motion RMSE is cents per action. The new metric excludes
unavoidable fills and still shows better quantity estimates alongside worse
motion economics. Thus the earlier result was not solely an artifact of the
old quantity definition. A broader bank and this fixed similarity metric do
not correctly weight quantity together with price severity.

The true-quantity substitution in this true-pair channel has additional support
exclusions: SOFT1 cancel/2 post, expanded and conditional1 cancel/1 post each.
Those rows have actual affected fills despite zero predicted affected-fill
support. Their counts and same-support original errors are retained rather
than silently excluded. Ordinary and actual-path channels have no such exclusions.

## Concrete loss and state examples

The frozen largest-error artifact contains ten rows per channel/case/kind.
These are descriptive examples selected by error, not a training population.

Oct26 post UID `20261026:3817683:post:1:0:117`, in the213-post population above:
an ask at22c with18.6s left was valued at+1.619c; its realized action value was
-367.784c. Expected affected fills were5.130 shares versus25.211 realized.
Predicted fill-weighted signed motion was-0.217c/share versus-15c realized.
The loss involves both more fills and much worse movement. It cannot be fixed
by correcting the spread or inventory penalty alone. Another Oct27 largest
posting error occurs296.8s before expiry, so near-expiry filtering alone would
not cover the examples.

Actual future paths with unchanged client-known state reduce posting RMSE to
0.114c (Oct26,n213) and0.366c (Oct27,n222), with no exclusions. Cancellation
residuals remain5.498c (n205) and1.967c (n221). This supports internal alignment
of the local action accounting, while exposing state approximation; it neither
validates exchange realism nor proves future paths learnable.

A supplementary, explicitly post-hoc check selects the three largest actual-
path cancellation residuals per day (n6,asof2026-11-17T00:49:56Z). Exact neutral
order/fill timestamps reconstruct client-known and actual remaining shares.
Four selected orders have unreceived fills; all four predicted affected-fill
quantities exceed their actual remaining shares. Three were already fully
filled. The other two examples remain unresolved by this remainder check.

For UID `20261026:3804777:cancel:0:0:3`,10 remaining shares filled30ms before
the decision, but confirmation is scheduled70ms after it. The client still
knows10 remaining; the actual-path calculation predicts54.518c of cancellation
benefit, while the exact branch difference is zero. This is not permission to
feed true remaining quantity into the model. The causal solution must represent
uncertainty about unconfirmed fills, using only available information. These
six selected examples are not a population-wide attribution estimate.

## Next implementation target

The next response model should estimate a coherent order-conditioned pair:

    Q = E[quantity affected by the action | current information]
    M = E[affected quantity * signed endpoint price movement | current information]

For a focal post, expected wealth is known edge*Q + M. For cancellation the
sign reverses, unavoidable exposure is removed, and other-order redistribution
must still be accounted for. Queue, remaining size, depth, activation/cancel
latency, public flow dynamics, exact remaining contract time and price
sensitivity belong in that response conditioning. Do not replace M by a
product of independent means, or estimate severity from a fill-selected
training population. Keep no-fill exposures with zero realized contribution;
measure severe-value tails separately without deleting or clipping losses.

Predicted responses must share incoming volume across orders and preserve
cross-order/side dependence for inventory moments. Independent order heads
cannot each claim the same public trade. The existing common executor remains
the consistency reference. The pending-fill state also needs a causal belief
about remaining exposure; conservative reservation caps may continue using
the known upper bound while valuation integrates the uncertainty.

Before controller replay, freeze a training/test contract for these joint
responses and demonstrate ordinary action-value improvement on matched actions,
with tail calibration and exact probability/volume/accounting checks. The1s
zero-continuation and neutral-to-policy target mismatch remain separate limits.
This diagnostic has not trained that model or established its learnability.

## Artifacts and verification

* [Frozen contract](../reviews/value_decomposition_20261117/freeze_v1.json).
* [Main evidence index](../reviews/value_decomposition_20261117/index.json).
* [Main audit](../reviews/value_decomposition_20261117/audit_v1.json).
* [Timestamped state inspection](../reviews/value_decomposition_20261117/state_check_v1.json).
* `data/predict_fun/research/value_decomposition_20261117_v1/` contains ignored
  actual/predicted action parquets and exact largest-error rows.
* `predictfun/research/run_value_decomposition.sh` reproduces the declared
  sequential diagnostic, with existing outputs protected from overwrite.

All861 canonical action labels and archived ordinary predictions reproduce;
Oct27 true-pair predictions also reproduce. Main audit re-reads exact artifacts
and recomputes metrics, substitutions, nulls and examples. Positive controls
and known-bad refusals cover the joint term, action signs, unavoidable fills,
redistribution, zero-fill support, components and probability inputs; inherited
execution/queue/accounting checks also pass. Every join uses the exact UID/generation/time identity.

Main data stages257.37s/277.61s; audit6.75s; peak RSS1284MiB, Codex/core8 only.
State check5.26s/828MiB. All numerical work complete. No forecast, size, latency,
controller, cooldown, threshold or production changes. Results and negative
mapping evidence retained; no promotion or live profitability claim.
