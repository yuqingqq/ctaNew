# STOP EV model choice and target review

Keep the current LightGBM hurdle as the research benchmark. On the same data,
direct value regression, an at-price/through-price target decomposition and
one Extra Trees configuration do not improve the combined prediction and
equal-participation value results. The main demonstrated weakness is predicting
value conditional on a fill and calibrating it across days. These comparisons
do not establish an irreducible information limit or show that another model
family could never help.

The agreed one-second local EV target is mathematically coherent. Its use as
an immediate pull signal needs care: it values a longer neutral continuation,
whereas the intended policy reviews an order every 50 ms and can cancel later.
That is a policy/target alignment question, not evidence that LightGBM cannot
estimate an expected value. Changing the economic horizon was not tested here.

## Controlled comparison

As of **2026-11-15T15:31:29Z**, five consumed test days (Oct 27/31, Nov 1–3),
**81,630 certified-estimated actions**, including **7,620 filled actions**.
Same 83 inputs, chronological folds, original training matrices, 30-share
reference ladder, cancellation latency, labels and missing-value inputs.
There are four setups, each raw and with one predeclared calibration offset:
**eight reported variants**, including the saved original control. Three new
learned setups; no parameter search, horizon search, tail removal, selected
replacement or forward validation. The other session's T5v2 remains separate.

| Setup | Rank IC | Squared prediction error, cents squared | Value of same-count selected actions |
|---|---:|---:|---:|
| Current LightGBM P(fill) × mean value given fill | 0.1132 | 839.4 | −$148.73 |
| Direct LightGBM mean value on all actions | 0.1033 | 842.8 | −$233.01 |
| Separate LightGBM at-price and through-price values, summed | 0.1048 | 844.0 | −$262.55 |
| Extra Trees direct mean value | 0.1098 | 849.3 | −$155.98 |
| Existing order-only hurdle control | 0.0710 | 867.4 | −$328.35 |

Same-count selection takes the highest-predicted 20% within each
day/hour/side/rung/exact known-remainder group, rounded up: **17,585 actions
for every setup**, with identical group counts. These retrospective ranks are
diagnostics, not online thresholds. The matched random-selection expectation
is calculated analytically, not a sampled null or significance test. All
values are sums of local action labels, not whole-strategy P&L.

The original hurdle has lower MSE than every alternative on **all five days**.
It also has better IC and same-count value than both LightGBM alternatives
on all five days. Extra Trees beats its IC and same-count value on two days,
but is slightly worse pooled. The hurdle and Extra Trees both beat the
order-only control on same-count value on all five days. Thus market inputs
add useful information; the fixed order state is not the whole explanation.

This is one fixed Extra Trees configuration, not an exhaustive family search:
64 trees, maximum depth 10, minimum leaf 200, feature fraction 0.8, seed 1,
one thread, squared-error criterion and native missing inputs. LightGBM retains
the original parameters and 200 rounds; only the objective required for a
regression head changes. LightGBM's
[regression objective is squared error](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective),
and Extra Trees' squared-error criterion predicts leaf means
([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)).
Both are compatible with a mean-value target. Quantile, median or clipped
targets would estimate different quantities unless explicitly reconstructed
into the desired expected value.

## Where the current model works and fails

| Test day | Eligible / excluded from v1 population | Filled actions | Fill AUC | Filled-value rank IC | Filled-value R² |
|---|---:|---:|---:|---:|---:|
| Oct 27 | 12,842 / 17,856 | 1,229 | 0.841 | 0.248 | 0.0318 |
| Oct 31 | 14,278 / 5,591 | 1,224 | 0.825 | 0.181 | 0.0007 |
| Nov 1 | 14,851 / 6,510 | 1,273 | 0.823 | 0.214 | 0.0109 |
| Nov 2 | 20,079 / 13,111 | 1,958 | 0.809 | 0.237 | 0.0227 |
| Nov 3 | 19,580 / 13,005 | 1,936 | 0.825 | 0.235 | 0.0277 |

The fill head separates likely fills from no fills. The value head has some
rank information but explains only about **0–3%** of held-out filled-value
variance. Low R² alone is not proof of no economically useful conditional
mean; the relevant failure is that the predicted attractive orders do not
consistently have positive realized local mean. All four raw setups lose on
their own pooled positive-prediction selection. Both sides lose in each raw
setup. Exact side counts and value totals are in `interpretation_v1.json`.

There are only 5,808–11,267 filled training actions per fold across 3–7 dates.
Many actions share market conditions, so the number of rows overstates
independent regime coverage. Previous checks found worse test filled-value
distributions even within coarse matched state groups. This experiment does
not separate inadequate inputs, finite-sample noise, conditional regime
change and residual label uncertainty into causal percentages.

Predicting fill activity well does not justify cancelling whenever fill
probability rises: fills also deliver spread revenue. The missing quantity is
whether a likely fill has sufficiently adverse expected value for this order.
Likewise, timing-invariant membership is a hindsight label-validity subgroup,
not a feature that can define a live "quiet state".

## Calibration helps the average but does not solve participation

The only calibration tested adds the mean residual measured on the
intermediate reserved day. It uses no test outcomes. For the hurdle, mean
optimism falls **1.033 → 0.320 cents/action**, improving on every test day.
Its positive selection shrinks from **15,681 actions / −$117.57** to
**4,791 / −$41.37**, still losing on four of five days. The highest decile
remains optimistic: about **+1.04 predicted versus −0.81 realized**.

One variant is positive: calibrated Extra Trees selects **1,115 actions
(1.37%)** and makes **+$5.97** in summed local value. Per day its selected
counts are 479 / 413 / 161 / 45 / 17 and values are
+$3.93 / −$3.65 / +$0.61 / +$5.06 / +$0.02. This small, shrinking selection
does not establish a replacement for an active market-making strategy, and
action selection fraction is not measured quote uptime. It is one of eight
consumed-data variants. At the common 17,585-action selection, Extra Trees
still loses and does not beat the original hurdle pooled.

The current zero-EV cutoff is a stage-2 diagnostic. The agreed design instead
retains near-zero-value resting orders and pulls below a negative margin that
prices queue priority; that stage-3 policy has not been evaluated here. These
results should not be called its realized strategy performance.

## Target changes worth testing

The present label sums the focal order's tranches from **t + 22 ms to t + 1 s**
under the neutral engine continuation, with each tranche marked one second
after its own fill. Unavoidable fills before cancellation arrival are already
excluded. At-price plus through-price value exactly equals the total. The
component experiment changes the estimator representation only; it preserves
the original economic target and does not improve it.

For a policy that rechecks every **50 ms**, cancellation at the next check
could land near **t + 72 ms** under the same latency assumption. A loss from
a fill hundreds of milliseconds later may still be avoidable at that next
check. The one-second label does not distinguish immediate cancellation
urgency from risk that can be reassessed later. It also deliberately omits
replacement-order value and the future option value of keeping queue priority.

A useful next controlled target study would keep the per-fill markout and
economic units, but separate **additional exposure before the next possible
cancel**, longer-horizon risk, and good-fill value. The decision must still
price the value of retaining queue priority and use the declared continuation.
Simply shortening the label and making more actions zero does not show an
improvement. It must improve the repeated-decision replay at matched activity
and reduced churn. This is a proposed test, not a measured better target.

Keep all severe losses in the expected-value objective. An explicit tail
probability/severity decomposition could make calibration more inspectable,
but each rare-event head would have fewer examples. The failed economic
component split shows that adding heads alone is not sufficient.

## Timing evidence and verification

The newer E2IZE experiment (`2e4cdcb5`, source `ae83775a`, report as of
15:15:54Z) is a useful negative control: on 18,911 originally invariant
actions, 97.5% match after retiming known-WS trades, and their matched mean
changes only −0.014 cents; 2.2% of matched outcomes change. It does **not**
validate the timestamp errors of the WS-missing population: those E2 groups
are explicitly left unchanged by the intervention. The recorded error table
also has appreciable mass outside the assumed ±250 ms bound. Therefore the
new evidence narrows the generic-estimator hypothesis but cannot justify
calling all E2-dependent losses physically verified adverse flow. No raw
timestamp reconstruction was performed in this comparison.

Source `d32b0ca1`, freeze `266f3ee2`. All original saved control predictions,
training matrices, labels and action identities reproduce exactly. Every new
head saves and reloads with identical predictions; source hashes are stable.
Mean/tail, native-missing and matched-selection positive controls pass; three
corrupted prediction identities and nonfinite scores are refused. Peak RSS
519 MiB; one core, 176 seconds, using the Codex research slot. Models and
predictions stay in ignored `data/predict_fun/research/stop_ev_model_choice_20261115_v1/`.

Full evidence:
[`report_v1.json`](../reviews/stop_ev_model_choice_20261115/report_v1.json),
[`interpretation_v1.json`](../reviews/stop_ev_model_choice_20261115/interpretation_v1.json),
the frozen builder and run log. All source population/status receipts remain
available. No active model, guard, target, posting policy or T5v2 job changed.
