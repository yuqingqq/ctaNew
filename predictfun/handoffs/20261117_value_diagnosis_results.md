# Calibration, training coverage and current-controller diagnosis

Completed and audited as of 2026-11-17T02:49:32Z. Economic-value calibration
remains inadequate on these consumed samples. Market features contain useful
fill-event information; one to three days of training improves coverage without
consistent value gains. The current controller also has substantial sensitivity
to the bank's side representation. Neither finding establishes a profitable repair.

Source726a5267, freeze8fea0ef3. Inputs, targets, model settings and policy are
fixed. Output: `data/predict_fun/research/value_diagnosis_20261117_v1`.
Evidence: `reviews/value_diagnosis_20261117/index.json`, `audit_v2.json`
and `publication_v1.json`. Audit source 076daf1a; revision receipt ea4fe573.

## Populations and interpretation

Reuse all27,236 eligible neutral Oct23–25 actions. Original30,673 selected
actions include3,435 unavailable client-state posts and2 stale future-book
labels retained as statuses; no replacement, fill-only selection or clipping.
Evaluate the same418 Oct26 and443 Oct27 neutral actions,12 markets/date,
with no new exclusions. Kind populations are205/213 cancel/post on Oct26 and
221/222 on Oct27. Calibration/history report as of2026-11-17T02:33:52.742Z;
H3 daily reports02:33:42.588/02:33:43.545Z. All days are consumed research.
No day-cluster intervals, untouched validation or promotion.

Controller diagnostics are a separate own-policy population: SOFT_JOINT's
entries and first reviews on the archived12-market Oct27 pilot. Those rows
never enter training. Attribution uses8 identity-hash pairs per
market/side/model-cancel group, before attribution or outcome computation.
It measures predicted-value sensitivity, not realized cancellation regret.

## More history improves coverage, not consistently economic prediction

Pinned model parameters/200 rounds and87 inputs, same Q/M/direct-wealth/severe
targets. H3 uses the original saved models. Other windows refit without tuning.
The matched-count window samples by identity hash within kind/side/phase,
without looking at outcomes. Eight additional affected-fill classifiers
(full/mechanical per window) are diagnostic only and do not affect values.

| History | Training dates | Training n |
|---|---|---:|
| H1 | Oct25 | 8,938 |
| H2 | Oct24–25 | 18,197 |
| H3 | Oct23–25 | 27,236 |
| H3_MATCHED_N1 | Oct23–25, matched H1 stratum counts | 8,938 |

Joint-moment total-value RMSE in cents/action, on the fixed evaluation populations
and as-of times above:

| Day / action | n | H1 | H2 | H3 | H3 matched count |
|---|---:|---:|---:|---:|---:|
| Oct26 cancel | 205 | 11.850 | 11.143 | 10.104 | 10.279 |
| Oct26 post | 213 | 34.654 | 36.569 | 35.892 | 35.022 |
| Oct27 cancel | 221 | 32.954 | 33.227 | 32.975 | 33.014 |
| Oct27 post | 222 | 25.674 | 25.934 | 25.500 | 25.959 |

Only Oct26 cancellation improves monotonically. Broader history at matched
row count improves only that comparison. This is not evidence that substantially
longer history cannot help: the test spans at most3 days and mixes added coverage
with regime changes. No best window is selected or promoted.

Coverage uses16 predeclared state dimensions, full-training median/IQR,
asinh scaling and explicit missingness, with same-kind/side nearest neighbors.
The95th-percentile reference uses500 min-hash training queries per group and
excludes the query's entire market. It is a diagnostic distance, not a calibrated
probability of failure or an eligibility filter.

H1→H3 outside-reference counts fall75→45 of418 on Oct26 and42→25 of443 on
Oct27. Every H3 evaluation action has training support in the predeclared
kind/side/phase/depth/mid cell. Median cell counts186.5/219 and severe-event
counts6/7 show that nominal support can still contain few informative tails.

Coverage alone does not explain the errors. H3 posting RMSE for near/far groups
is26.612/82.981c on Oct26 (n193/20) but26.138/10.989c on Oct27 (n209/13).
The distance diagnostic identifies some difficult Oct26 states, but does not
consistently separate errors across dates. None of these rows are excluded.

## Market features predict fills more reliably than profitable fills

Affected-fill event means positive action-affected focal quantity; cancellation
excludes unavoidable fills. All zero-fill actions remain in fitting and scoring.

| Day / action | n / fill positives | Full fill AUC | Mechanical AUC |
|---|---:|---:|---:|
| Oct26 cancel | 205 / 15 | .812 | .727 |
| Oct26 post | 213 / 31 | .766 | .701 |
| Oct27 cancel | 221 / 27 | .884 | .836 |
| Oct27 post | 222 / 25 | .835 | .774 |

Full fill Brier and log loss also improve over mechanical in all four samples.
This supports incremental fill-event information in the market features on
these consumed actions. It does not establish profitable selection.

Posting expected affected quantity is reasonably close in aggregate:
2.759 predicted vs2.928 actual shares/action on Oct26 (n213), and2.670 vs2.599
on Oct27 (n222). Yet predicted versus realized fill-weighted signed endpoint
movement is−2.153 vs−3.123c/share and−2.911 vs−3.286. These are ratios of
summed joint moments to summed quantities, not independent price forecasts or
per-fill one-second markouts. The model understates adverse movement in both
posting samples, especially Oct26.

Severe-loss train AUC is.980 on both cancellation (455 positives/14,792)
and posting (599/12,444), as of02:33:40.730Z; these are explicitly in-sample.
Later posting AUC is.707/.785 (12/213 and11/222), with inconsistent incremental
skill over mechanical. This is a generalization gap, not a clean experiment
separating overfit, conditional distribution shift and outcome noise.

The auxiliary fill-probability model and existing quantity regression are not
mutually constrained. They violate `E[Q] <= P(Q>0) * known_remaining` for
46/213 and53/222 posts, and66/205 and77/221 cancellations. This describes
incompatibility of the newly combined diagnostic heads; the probability head
was not part of the previous controller, so it is not a discovered cause of
that controller's losses. A future joint model must enforce this consistency.

## Positive economic scores are not calibrated in these samples

The value bands were fixed before this diagnostic:−30,−5,−1,0,1,5,30c,
with unbounded end bins. The following contains every populated positive band
for H3 JOINT_MOMENT posting, not a selected threshold search.

| Day | Predicted-value band | n | Mean predicted, c | Mean actual, c |
|---|---|---:|---:|---:|
| Oct26 | (0,1] | 23 | +.474 | −7.959 |
| Oct26 | (1,5] | 34 | +2.597 | −5.881 |
| Oct26 | (5,30] | 11 | +8.069 | −16.006 |
| Oct27 | (0,1] | 41 | +.307 | −1.894 |
| Oct27 | (1,5] | 35 | +2.255 | −.337 |
| Oct27 | (5,30] | 10 | +16.089 | −7.955 |

All positive bands have negative realized means on both dates. These small,
heavy-tailed samples do not establish exact conditional expectations. They do
show why merely raising the posting threshold is not supported by this check.
All cases/windows retain200 market/side/kind action-count-matched nulls.
Sums of these overlapping counterfactual values are not strategy P&L.

Global realized-outcome RMSE is not a model-error confidence bound and must not
be inserted as an arbitrary trading margin. Uncertainty in an estimated mean
needs separate calibration and evaluation on later data.

## Current SOFT_JOINT controller

The completed replay exactly reproduces archived decisions and fills on all
12 Oct27 pilot markets, as of 2026-11-17T02:46:04.234Z. Of 10,847 selected
posts, 10,675 have a first-review pair, 118 are structurally cancelled before
that review, 51 fail activation and 3 lack an eligible review context. Among
the paired rows, 3,838 become model-cancel candidates and 6,837 remain
model-keeps; 116 of those model-keeps are nevertheless structurally removed
at that review. These are first-review counts, not total quoting uptime.

For the 3,838 rapid model cancellations, median predicted entry advantage is
0.212 cents/order. Best prices are unchanged in 1,887, but the entire top-five
book is unchanged in only 53. Known inventory is unchanged in 3,707 and the
queue estimate in 1,671. Only 27 have an unreceived focal fill. Unchanged
midpoint or best prices therefore cannot establish unchanged market conditions.

Attribution uses 384 identity-hash pairs: 192 model-cancels and 192 model-keeps,
96 per side in each group. This balanced sample is not a population-weighted
estimate. On the 192 sampled cancellations, mean keep-minus-cancel value falls
from +0.837 cents under frozen entry conditions to -1.396 cents at review.
The complete 16-cell decomposition assigns that -2.233-cent change as follows:

| Changed input group | Mean contribution, cents/order |
|---|---:|
| Market/book/scenario context | -1.402 |
| Forecast probabilities | -.559 |
| Portfolio and inventory | -.207 |
| Target queue estimate | -.064 |

This attributes changes in the controller's predictions, not realized economic
loss or the causal benefit of cancellation. Market context includes scenario
weights, relabelled paths and starting depth, not just price movement.

| Sensitivity on the same 192 model-cancels | Still predicts cancellation, n |
|---|---:|
| Keep the original absolute valuation endpoint | 170 |
| Extend the current horizon to two seconds | 182 |
| Restore the entry queue estimate | 189 |
| Restore the entry forecast probabilities | 109 |
| Restore the entry market/scenario context | 149 |
| Substitute true current state for delayed client state | 186 |

The last diagnostic uses latent queue, remaining quantity and inventory together;
it is not a pure fill-notice experiment and is never fed to the policy. In 147
of the 192 cancellations, wealth alone also favors cancellation without the
inventory-risk term. In 18, a hypothetical fresh post is positive while keeping
the current order is negative. Those actions have different activation and
cancellation latency, so this is not by itself an accounting contradiction.

The short endpoint, queue update and delayed state do not individually explain
most sampled reversals. These checks do not solve omitted continuation value or
show that the controller's response to changed conditions is economically right.
Bank identities change in only 19 of these 192 cancellations; sensitivity to
market/scenario context is therefore broader than discrete bank switching.

The predeclared 200-draw market/side-matched cancel-label null also finds larger
forecast changes among cancellations (mean L1 difference .08247 across all
10,675 pairs; null range -.01240 to +.01112). This is descriptive association
with the model decision, not evidence that those cancellations avoid losses.

## Bank representation is a concrete controller experiment

For all 384 sampled states, transforming both packed paths and order/portfolio
state by side swap and price complement preserves kernel value. This is an
accounting control. Transforming only the query against the original bank changes
the keep/cancel preference in 175 of 384 supported queries: 46/192 model-cancels
and 129/192 model-keeps. All four side/decision strata contain 96 observations;
their mean absolute value changes range from 1.506 to 2.333 cents/order.

The bank is built from bid/rung-zero probes. The measured sensitivity justifies
a controlled side-balanced or complement-paired bank experiment. It does not
show that the complemented choice is better, that the real market is symmetric,
or that bid conditioning causes the ask losses. Side balance must retain joint
flow/price dependence and use training data only. Compare it on the same neutral
actions and full replay, with every other policy setting fixed.

The failed direct-wealth head predicts focal value without using the scenario
bank. Its failure means that fixing bank representation alone cannot explain or
repair all prediction failures. Other-order and inventory terms still enter its
total action score; this is not a claim that the entire policy bypasses the bank.

## Next controlled work

1. Test a training-only bank representation correction with the existing model
   and policy held fixed. Report bid/ask value calibration, rapid reversals,
   queue retention and fixed-horizon fill markout, separately from settlement
   carry. Fewer cancels alone is not success.
2. Improve coherent joint quantity/value estimation and severe-loss calibration
   before introducing another policy threshold. Retain no-fill actions; do not
   turn this into a fill-selected training set. Measure incremental economic
   skill over the mechanical baseline and conditional tail amount, not only
   fill probability or severe-event classification AUC.
3. Evaluate any correction prospectively on at least five complete untouched
   days after freezing the full pipeline. The present Oct23–27 evidence is
   exploratory, not validation. A longer-history extension remains open; this
   three-day comparison cannot settle its usefulness.

Continuation value remains an open modeling issue, but the endpoint sensitivities
give no evidence that merely extending a timer fixes current reversals. No new
cooldown, minimum age, arbitrary RMSE margin, risk coefficient, controller rollout
or strategy P&L result is introduced by this diagnostic.

## Audit and reproduction record

The independent reconstruction checked all four histories, all saved heads,
feature/target identities, quantities and economic values, coverage metrics,
calibration reports and 200-draw economic nulls. It reconstructed all 384 sampled
states' 16-cell attributions and sensitivities. Positive controls cover calibration,
novelty, unchanged states, clock reversals and complement invariance; planted
metric, incomplete-cube, mirror and ledger corruption are refused.

The original auditor failed because UID was already the sensitivity table index
and it attempted to read UID as a column. The superseding auditor checks index
identity explicitly. `audit_revision_v2.json` binds the new source and original
failure log to unchanged calibration and controller receipts. No numeric result,
population, model or diagnostic algorithm changed. The original frozen source
and failure remain as provenance.

The calculation stages are `pf_value_diagnosis_calibration.py run` and
`pf_value_diagnosis_controller.py run`; the current auditor is
`pf_value_diagnosis_audit_v2.py`. The archived `run_value_diagnosis.sh` still names
the superseded auditor and is not the current audit entry point. All numerical
stages ran sequentially through `run_light.sh` with `PF_SLOT=codex`; publication
artifacts are immutable and reruns must not overwrite them. Calibration took
43 seconds/432 MiB peak RSS, controller reconstruction 732 seconds/1,045 MiB,
and the successful audit 130 seconds/412 MiB. Raw action and state tables remain
under the ignored data directory; only sources and compact evidence enter git.

Verify the published source chain, raw-artifact/model hashes and computed summary
predicates without overwriting evidence:

```bash
PF_SLOT=codex predictfun/research/run_light.sh value_diagnosis_verify -- \
  /home/yuqing/pricer-sol/venv/bin/python3 \
  predictfun/research/pf_value_diagnosis_record.py verify
```
