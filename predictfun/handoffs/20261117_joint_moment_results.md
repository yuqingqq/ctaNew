# Joint fill/price moment screen: failed; retain SOFT_JOINT

The direct joint-moment predictor did not consistently improve action values.
It improves ordinary value RMSE in only one of four date/action comparisons,
and beats the mechanical comparison in only one. Both frozen screens fail.
No controller rollout, profitability claim or promotion follows this result.

## Evidence and scope

- V2 source `5c87a239`, freeze `c659833d`; supplementary reconstruction source
  `5924b1a7`. Evidence: `reviews/joint_moment_20261117/v2/index.json`,
  `audit_v2.json`, `reconstruction_v1.json`, and `publication_v1.json`.
- Training Oct23–25: 27,236 eligible neutral-path actions in 864 markets,
  as of 2026-11-17 01:42:02 UTC. Of 30,673 selected actions, 3,435 proposed
  posts are unavailable in client state and two have stale future-book labels;
  all remain explicit statuses without replacement. No fill-only selection.
- Daily eligible counts: 9,039 / 9,259 / 8,938. All 232 deterministic sampled
  training labels reconcile with full canonical paired branches, including
  30 positive affected-fill actions. This is sampled training verification.
- Evaluation: exact archived 418 Oct26 actions (205 cancel / 213 post) and
  443 Oct27 actions (221 / 222), 12 markets per date; no new exclusions.
  Every evaluation label reconciles. Reports as of 2026-11-17
  01:42:03.845 / 01:42:05.464 UTC; reconstruction as of 01:42:19.745 UTC.
- All dates are consumed development dates; Oct26 also calibrated the older
  classifier. No untouched validation, confidence intervals or protected-date
  access. Results describe the simulator and selected neutral actions.

The failed V1 build stopped before fitting on an unavailable future-book path.
V2 preserves that row as a status. V1 source/freeze, logs and `aborted_v1.json`
remain provenance; the target, inputs, models and sample rule did not change.

## What was tested

Same 30-share orders, six rungs per side, cap180, risk1/360, one-second endpoint
and 58/37/100/50 ms place/cancel/notice/review clocks. Same paired action target:
post minus wait, or cancel minus keep; unavoidable cancellation fills excluded.

Fit pinned LGBM/200 rounds to all eligible actions, using 71 causal market
features plus16 order/depth/portfolio inputs. New inputs include displayed
depth at the focal price, cumulative displayed depth ahead, known own orders
ahead, queue estimate, exact expiry distance and client-known remaining shares.
True exchange state is label-only. Both sides share the fitted heads, with
side-oriented inputs and an explicit side feature; side metrics are separate.

Let `Q=E[affected focal shares]` and
`M=E[affected focal shares × signed endpoint price change]`.
Retained focal wealth is `edge × Q + M`. M is a joint product target, not the
product of separately predicted marginal means. Physical payoff bounds apply
to predictions; labels are not clipped or winsorized.

| Name | Difference from SOFT_JOINT |
|---|---|
| SOFT_JOINT | Existing scenario mapper; reference |
| JOINT_MOMENT | Replace focal wealth with directly predicted Q/M |
| MECHANICAL_MOMENT | Same Q/M targets; order, queue and basic book inputs only |
| DIRECT_WEALTH | Regress retained focal wealth directly |
| REMAINDER_ONLY | Scale cancellation focal wealth by predicted actual/known remainder |

All retain the reference other-order wealth and inventory-risk terms. The
remaining-exposure head is a separate ablation: Q/M labels already include
unconfirmed-fill uncertainty, so no double scaling is applied. Separate full
and mechanical severe-loss classifiers are scored but never used as a new
policy threshold. No parameter, horizon or threshold search was performed.

## Ordinary action-value prediction

RMSE in cents per action; lower is better. Populations and as-of times above.

| Day | Action | n | SOFT | Joint moment | Mechanical | Direct wealth | Zero prediction |
|---|---|---:|---:|---:|---:|---:|---:|
| Oct26 | Cancel | 205 | 8.449 | 10.104 | 9.742 | 8.961 | 7.397 |
| Oct26 | Post | 213 | 36.111 | 35.892 | 34.243 | 35.457 | 35.602 |
| Oct27 | Cancel | 221 | 31.826 | 32.975 | 31.508 | 32.245 | 31.636 |
| Oct27 | Post | 222 | 24.829 | 25.500 | 26.006 | 25.793 | 25.707 |

The direct-wealth alternative also fails to improve consistently. Joint moment
beats zero prediction only for Oct27 posts. Extra market dynamics do not show
a consistent increment over the mechanical model in this experiment; this is
not evidence that market information is intrinsically uninformative.

Bid/ask results are mixed. Posting RMSE SOFT→joint: Oct26 bid19.691→20.215
(n106), ask47.029→46.472 (n107); Oct27 bid32.065→31.847 (n111),
ask14.310→16.918 (n111). No side-specific winner is selected after observing this.

Phase checks also fail to show a broad improvement. First-minute posting
RMSE improves8.350→7.064 and44.975→43.557 (n48 each date), but Oct27
fifth-minute posting worsens11.860→19.438 (n37), and cancellation
12.342→24.778 (n33). Full phase tables are in `reconstruction_v1.json`.
These are prediction errors, not contract-minute strategy P&L.

## Severe losses and decisions

Severe means retaining the focal exposure has incremental total value below
−30c: negative post value or positive cancel benefit. The definition includes
the existing inventory/other-order terms and is not settlement loss.

| Day | Action | Severe n / all n | SOFT severe RMSE | Joint severe RMSE |
|---|---|---:|---:|---:|
| Oct26 | Cancel | 4 / 205 | 35.404 | 37.752 |
| Oct26 | Post | 12 / 213 | 142.956 | 142.662 |
| Oct27 | Cancel | 9 / 221 | 136.241 | 136.096 |
| Oct27 | Post | 11 / 222 | 103.765 | 102.189 |

Small improvements in some tail errors do not establish useful decisions.
At the unchanged positive-value threshold:

| Day | Posting model | Selected / all n | Selected severe posts / all severe | Sum of selected local values, c |
|---|---|---:|---:|---:|
| Oct26 | SOFT | 45 / 213 | 3 / 12 | −621.692 |
| Oct26 | Joint moment | 68 / 213 | 4 / 12 | −559.072 |
| Oct26 | Direct wealth | 60 / 213 | 0 / 12 | +126.972 |
| Oct27 | SOFT | 50 / 222 | 0 / 11 | −19.994 |
| Oct27 | Joint moment | 86 / 222 | 2 / 11 | −168.983 |
| Oct27 | Direct wealth | 56 / 222 | 2 / 11 | −157.196 |

These are overlapping neutral-path counterfactual sums, **not strategy P&L**.
Counts differ; each arm has200 market/side/action-kind count-matched null
draws in the reports. Direct wealth exceeds its Oct26 posting null maximum
(126.972 vs114.745c), but not Oct27 (−157.196 vs55.256c). That isolated result
does not survive across dates or justify selecting it as a successful strategy.

The severe-loss classifier has some ranking signal, but inconsistent incremental
skill/calibration. Posting AUC full/mechanical is .707/.684 on Oct26
(12 positives /213) and .785/.802 on Oct27 (11 /222). Brier scores are
.05301/.05244 and .04408/.04493. Cancellation AUC is .780/.672 and
.706/.670, but the full classifier's Oct27 Brier is worse. No tail threshold
or economic overlay was fitted on these outcomes.

## Pending remainder, representation and limits

Remaining-fraction RMSE improves from the known-upper-bound baseline
.10064→.08896 on Oct26 (n205, three unreceived-fill cases) and
.10979→.10813 on Oct27 (n221, four cases). Scaling cancellation focal wealth
improves total-value RMSE8.449→8.348 and31.826→31.793, but changes none of
the cancellation decisions. It does not solve the main value-prediction gap.

Training contains23,896 zero-affected-fill actions out of27,236 and1,054
severe retained-loss actions. This sparse, heavy-tailed target remains hard
for the tested mean regressions. The experiment does not isolate short
training history versus model misspecification as the cause.

Only one observed feature value per evaluation date lies outside its training
min/max: Oct26 `act_rel`, Oct27 `adv_pf_imb3`. There is no broad marginal range
extrapolation in this check. This does **not** rule out changed feature
combinations, different conditional relationships, or inadequate rare-event
coverage. Missing inputs remain NaN and are counted in the reports.

All saved predictions reproduce from model artifacts; all five economic value
reconstructions, moment/remainder/coverage metrics, selection identities and
reported nulls reconcile. Positive controls and planted-bad refusals pass.
Maximum job RSS1468MiB; all numerical jobs used Codex/core8 sequentially.

Keep SOFT_JOINT as the research comparison. This test does not support plugging
the new regressions into the controller. A useful next investigation is whether
an all-action distributional model can calibrate the probability **and size**
of fill-weighted losses, including the no-fill mass, and beat these mechanical
and direct-value controls. Merely adding features is not established as the fix.

Shared-volume multi-order allocation, portfolio moments, pending-state belief,
one-second continuation, neutral-to-policy transport and external simulator
realism remain unresolved. This screen held the controller fixed; it provides
no new churn, quoting-uptime, markout or settlement-P&L result.
