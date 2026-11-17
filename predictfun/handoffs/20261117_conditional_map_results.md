# Conditional response mapping with fixed event probabilities

The state-conditioned mapper **fails the frozen development gate**. Retain
`SOFT_JOINT`; no continuous replay ran and no new strategy P&L or churn result
is claimed. More scenario coverage and the fixed state-similarity kernel do not
resolve the conditional economic-value gap. Fill quantity improves somewhat,
but value error does not improve consistently.

All source, action identity, accounting, probability and metric audits pass.
The gate is a measured failure of this candidate, not a pipeline failure.

## Experiment

Source `f60e2144`, freeze `62201b95`. One candidate, `STATE_CONDITIONAL`, and
one `EXPANDED_JOINT` coverage ablation against the retained `SOFT_JOINT` baseline.
Existing Oct27 forecasts, policy, execution kernel and risk settings are fixed.
There is no fixed retry, minimum rest time or new action threshold.

The original bank compressed 19,198 selected training exposures into 1,929 raw
paired book/tape paths. The new bank contains 3,597 representatives, keeping
all 1,929 originals unchanged and allowing up to eight paths per original
phase/spread/coarse-event-pair cell. The same 27,875 eligible candidate times
and 19,198 deterministic selections are used, all eligible, with no missing
market probes. Representative population weights sum to 19,198. Selection
adds paths that cover different causal market states; realized P&L is never
used to choose them.

The candidate conditions within each event pair on 17 causal inputs: eight
cumulative depth coordinates, absolute W30 liquidity scale, spread, midpoint,
and six existing book/flow/Binance dynamics. Depth and scale are in log share
units relative to the fixed 30-share order size. Median/IQR comes only from
training states, with explicit missingness; zero IQR uses unit width. One fixed
Huber distance and kernel width are frozen, without a search.

For each current book, the original SOFT projection first produces an achieved
bid/ask coarse-event-pair distribution. Both new maps preserve that entire
225-cell distribution to numerical tolerance. This is the distribution achieved
by the finite-bank projection, not a claim that the original incompatible
forecast marginals have become exact. Only the conditional response inside
each event changes. The expanded-only ablation omits the state kernel.

All paths remain paired across sides, with raw trade volume and price/time
ordering intact. Current order queues and outstanding quantities are handled
by the existing common execution kernel. Whole-portfolio endpoint wealth and
inventory penalty define the action values. Settings remain 30 shares per
order, six rungs per side, cap 180, risk coefficient 1/360, a one-second horizon,
58ms placement, 37ms cancellation, 100ms notification and 50ms review.

## Evaluation

Reuse exactly the archived neutral post/wait and keep/cancel actions, labels
and generation/time identities on Oct26 and Oct27. Reconstruct current client
states from causal reference replays; reproduce SOFT predictions and canonical
reference fills, orders and cancellations. No-fill actions remain included.
Report action error, ranking, fill quantity and 200 count-matched null draws.
All economic values below are cents per whole action, not cents per share.

A separately labelled Oct27 retrospective diagnostic supplies the actual future
coarse pair to isolate within-category mapping quality. It never supplies future
information to ordinary predictions or policy. The frozen compute gate permits
a full 12-market policy replay only if STATE_CONDITIONAL improves or matches
true-pair value RMSE for both action kinds, strictly improving at least one,
on identical support. This is a development screen, not validation or promotion.

Oct26 calibrated the inherited classifier temperature. Both dates have already
been consumed for research; neither is untouched validation. No confidence
intervals or general profitability claim. Protected dates remain unopened.

## Ordinary forecast-to-value results

As of 2026-11-17T00:24:05Z for Oct26 and 00:26:40Z for Oct27. Each date uses
12 preselected full-contract markets. All 418 Oct26 actions (205 cancel,
213 post) and 443 Oct27 actions (221 cancel, 222 post) are supported by all
three maps; zero evaluation exclusions. Source action-generation statuses
remain in the linked reports: 13,450 generated / 9,423 eligible on Oct26;
14,480 / 9,588 on Oct27, before deterministic action selection. Statuses for
incomplete books, unavailable client orders, expiry and stale marks are kept.

| Date / action | n | SOFT RMSE | Expanded RMSE | Conditional RMSE | SOFT rank | Conditional rank |
|---|---:|---:|---:|---:|---:|---:|
| 1026 / cancel | 205 | 8.449 | 8.923 | 8.991 | 0.117 | 0.103 |
| 1026 / post | 213 | 36.111 | 36.101 | 36.097 | 0.067 | 0.037 |
| 1027 / cancel | 221 | 31.826 | 31.700 | 31.709 | 0.339 | 0.287 |
| 1027 / post | 222 | 24.829 | 25.260 | 25.342 | 0.224 | 0.207 |

There is no uniform economic improvement. Oct26 cancellation worsens and entry
error barely changes; Oct27 cancellation RMSE improves slightly but its rank
weakens, while entry RMSE and rank both worsen. On Oct27 the entry RMSE
increase occurs on both bid (n111) and ask (n111), so pooling the two sides is
not concealing an improvement on one side.

The positive-score selections have exactly the same aggregate realized values
as baseline on both dates despite changed action counts. For example, on Oct27
ordinary STATE_CONDITIONAL selects 48/222 posts with total realized advantage
-19.994 cents, mean -0.417 cents; SOFT selects 50/222 with the same total.
Candidate cancellation selections are 136/221, totaling +955.681 cents;
SOFT selects 137/221 with the same total. These are neutral one-action branch
values, not a portfolio replay. Candidate count-matched null means are -138.112
cents for posts and +654.202 for cancels; maxima +23.084 and +1,132.453.
Neither observed total exceeds its 200-draw matched-null maximum. This is
not a significance test across days; the full null draws' summaries are retained.

## True-category diagnostic and gate

As of 2026-11-17T00:26:40Z, the same 443 Oct27 actions. All maps share 420
supported identities (208 cancel, 212 post); 23 have an unsupported actual
coarse pair and remain explicitly reported. This table supplies future
category information only in the retrospective branch. It is not a strategy.

| Action | n | SOFT RMSE | Expanded RMSE | Conditional RMSE | SOFT fill-quantity MAE | Conditional fill-quantity MAE |
|---|---:|---:|---:|---:|---:|---:|
| cancel | 208 | 25.951 | 28.245 | 28.385 | 2.528 | 2.254 |
| post | 212 | 23.971 | 26.360 | 26.514 | 3.875 | 3.403 |

Quantity errors are shares; value errors are cents per action. Both action kinds
fail the predeclared no-worse RMSE requirement. The expanded-only ablation
already worsens value error, and state conditioning does not recover it.
The gate receipt therefore records `run_pilot: false`; audit verifies that
no pilot artifacts exist. Known-future predictions reproduce the old true-pair
baseline exactly; this is the full common420 population, not the common416
population from yesterday's richer-category comparison.

Fill-quantity MAE improves while value RMSE worsens. Conditional post-value MAE
also falls slightly (8.797 to 8.578 cents) while its RMSE rises (23.971 to
26.514); that pattern is consistent with larger residual value errors mattering
more. It does not identify their precise causal source. Category membership,
average fill quantity and generic current-state similarity are insufficient
checks of the conditional fill/price association. They do not prove the
market features have no predictive power.

## Interpretation and next boundary

Do not adopt this fixed distance kernel or expand/tune it on these same action
outcomes and call that validation. There are three separate unfinished parts:

1. Conditional price severity and its association with order-specific fills.
   A better fill-quantity prediction alone can still misvalue the selected fills.
   The next response test should separately reconcile expected fill shares,
   fill-weighted signed price movement and inventory-cost contribution on the
   same action identities, including severe-value tails and no-fill actions.
2. Relevant conditional state. This test has 17 coordinates, not all available
   information. It omits explicit volatility and exact remaining contract time
   within a phase; zero-IQR dynamics use unit width. Full-depth prices beyond
   the observed five levels and a learned joint response are also absent.
   Those are design limits, not newly demonstrated bugs or a license to tune
   against these outcomes. The coarse pair anchor also fixes the baseline's
   cross-side dependence as well as its marginals.
3. Sequential value consistency. The one-second endpoint still gives no
   continuation value to queue priority beyond that time. Neutral action labels
   do not include later learned cancellation/reposting. This experiment does
   not test or solve that policy/target mismatch or the earlier churn problem.

The limited conclusion is that this mapping revision fails even before testing
strategy feedback. The earlier forecast errors and external simulator realism
are not cleared by this result. Keep the baseline and the negative evidence.

## Reproduction and evidence

* Source commit `f60e2144`; freeze commit `62201b95`.
* [Frozen contract](../reviews/conditional_map_20261117/freeze_v1.json).
* [Evidence index](../reviews/conditional_map_20261117/index.json), including
  exact output hashes, action reports, source populations and statuses.
* [Gate](../reviews/conditional_map_20261117/gate.json) and
  [audit](../reviews/conditional_map_20261117/audit_v1.json).
* Ignored artifacts: `data/predict_fun/research/conditional_map_20261117_v1/`.
* Pipeline: `predictfun/research/run_conditional_map.sh`; numerical jobs use
  `PF_SLOT=codex` and the existing `run_light.sh` resource wrapper. Existing
  outputs are immutable; the launcher refuses overwriting completed stages.

Audit as of 2026-11-17T00:26:42Z: preserved all original raw paths and full
training exposure weight, exact action identities/labels and ordinary/true-pair
baseline reproduction, canonical reference parity, metric/null recomputation,
value decomposition, size bounds and coarse-pair mass preservation. The true-
pair maximum mass residual is 6.66e-16; all ordinary rows also pass 1e-12.
Positive and known-bad controls cover causal conditioning, queue preservation,
latency, shared volume, sign, ledger corruption and probability refusal. All
24 inherited nonempty compiled/canonical path comparisons pass.

Peak measured RSS 1,086 MiB; bank66.94s, Oct26 actions158.56s, Oct27 actions
154.42s, audit1.75s. No numeric job remains. No model-family, horizon, threshold,
latency, order size or controller tuning; one response candidate and one
coverage ablation. No new predictor fit, full-policy result or promotion.
