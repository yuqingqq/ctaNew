# Next priority: economically consistent entry and retention

User request, 2026-11-19: write clear next steps and evaluation metrics, then
commit them. The development session owns implementation. This commit is a plan,
not a model freeze, completed experiment or authorization to restart scheduling.

Basis: result commit **37e8b373d5ef29beb605573c82d0842dba320a89**, reviewed in
**1adf6a334306623bbd9d1af2332d21a9e0adcc55**. See the
[review](../reviews/automatic/20261119_37e8b373/report.md),
[reproduction](../reviews/automatic/20261119_37e8b373/receipt.json), and
[attribution](../reviews/automatic/20261119_37e8b373/followup.json).
Other-session implementation changes now in progress have not been reviewed by
this plan. Preserve them and existing result artifacts.

## Objective and decision

**Build an entry/retention policy that earns value while maintaining useful
resting quotes.** Lower loss obtained by immediately withdrawing almost all
orders is insufficient. Equally, lower cancellation frequency obtained by
keeping losing orders is insufficient.

The next priority is to make POST/WAIT and KEEP/PULL share an explicit
continuation policy, economic endpoint and order-state accounting. First measure
the current frozen entry-plus-cancellation combination; then build one-step
action advantages. Defer size, price-level, horizon and hyperparameter searches
until this comparison is reliable. No fixed cooldown, minimum-rest timer,
arbitrary queue bonus or cancellation-count penalty is introduced.

The known symptom is first-review reversal, not simply eventual cancellation:
the reviewed run sent 96–97% of its pulls at the first +100 ms review, roughly
42 ms after +58 ms activation. Most absolute loss preceded the earliest +137 ms
cancellation effect. Those are scenario-conditional findings on consumed
Oct26/27, as-of Nov19 07:39 UTC, with 30,242 / 29,820 original opportunities.
The existing POST label omits subsequent model pulls; a negative neutral-hold
label does not establish that entry followed by cancellation is negative.

## Fixed settings for this work

Keep the current 30-share best-price JOIN, market universe, side conventions,
existing 91 inputs, pinned LightGBM parameters, 50 ms decision grid, queue rule M,
and baseline A58/L37/N100 timing. Keep bid/ask models and reporting separate.
Use the same declared execution-scenario bank; change one component per stage.
Keep existing fee/rebate/cost assumptions explicit and unchanged. Missing or
unmodelled costs are not silently treated as measured zero-cost execution.

Start with consumed Oct26 and Oct27 diagnostics. Oct23–25 remain consumed
training material. Oct26 is not untouched because it was called a gate day;
Oct27 is also post hoc. Do not open protected dates to complete this plan.
Use PF_SLOT=codex or PF_SLOT=claude only for the owning session through
`predictfun/research/run_light.sh`, with absolute script paths. Checks precede
new fits and tape replays; this documentation session launches neither.

## Ordered implementation steps

### 0. Close the remaining correctness prerequisite

Owner: development session. Repair `Eq == 0, Eqm != 0` acceptance, including
positive and negative Eqm, zero probability, nonfinite values and tiny values
under an explicitly declared numerical tolerance. Validate every forecast's
moments and reconstructed value, not only sampled action agreement. An invalid
forecast has an explicit WAIT/KEEP fallback and remains counted.

Replay the existing falsifiers and scan both saved prediction artifacts. The
review found 23 / 15 affected RETAIN rows, all after the first model pull; the
expected historical result is zero changed first triggers and zero value change.
Publish the computed result even if it differs. Correct the report's attribution
between later pulls and never-pulled orders. Supersede receipts in-band.

**Exit gate G0:** original causality/refusal/gate cases and new coherence cases
pass with positive controls and known-bad refusals; hashes identify corrected
code and artifacts. No claim that this repair makes the model profitable.

### 1. Deliver a no-fit entry-plus-cancellation diagnostic

Use the existing chronological Oct23–25 fitted models after G0; retain their
weights and zero-value decision boundaries. On the original five-second neutral
opportunities, evaluate exactly these five named policies:

| ID | Entry | Subsequent cancellation |
|---|---|---|
| N0 | Never enter | None |
| A0 | Enter every original opportunity | First live opportunity, PULL-ALL |
| A1 | Enter every original opportunity | Corrected frozen MODEL-J |
| E0 | Enter iff coherent POST J_V > 0 | First live opportunity, PULL-ALL |
| E1 | Enter iff coherent POST J_V > 0 | Corrected frozen MODEL-J |

N0 has zero new exposure. E0/E1 use the identical admitted orders. A0/A1 use the
identical original opportunities. Preserve no-fill, post-only refusal, missing
feature and endpoint statuses. Count admitted, activated and rejected orders
separately. Inspect bid and ask, first-pull versus later-pull versus never-pull,
and fills before first decision / pending cancellation / after earliest effect.

This stage allows one attempted generation per original opportunity, with no
re-entry before its common END. Its alternative is named **NO_ENTRY**, not
adaptive WAIT. It can be reconstructed from saved fills where independent-order
assumptions hold; any new shared-state behavior requires actual replay. The
review's conditional E1 totals are reproduction anchors, not a new validation.
Regenerate policy ledgers and all metrics; do not publish only conditional means.

**Deliverable:** paired per-opportunity ledger, five-policy table, chronology and
population manifest, and falsifiers for branch identity, missed refusals and
double-counted fills. G1 passes when these reconcile, regardless of economic
sign. If all admitted-policy values remain negative, record that result; do not
search thresholds or sizes on these days to manufacture a pass.

### 2. Define genuine next-review actions and a frozen continuation

Before paired labels or the candidate fit, commit a complete reference policy `pi0` and its
artifact hashes. It must define absent, post-pending, live, cancel-pending and
terminal states, including entry, partial fills, re-entry and inventory limits.
Use the corrected E1 entry and cancellation rule as the initial reference;
preserve its score conventions explicitly. Defining pi0 is separate from
claiming its legacy scores estimate the new action values.

For this first adaptive experiment:

- Evaluate both actions from the same client-known state, on the existing 50 ms
  grid. Pending operations retain their actual activation/cancellation latency;
  a next grid tick does not imply a pending order is actionable.
- Use finite episodes rooted at the original opportunity, with a common absolute
  END = episode start + 5 seconds for every branch and descendant generation.
  A later post or review does not reset END. Stop new exposure at END and value
  existing inventory using the declared terminal convention. This is a bounded
  diagnostic; later continuous trading needs its own terminal treatment.
  Preserve live/pending orders and reservations in the END state; do not invent
  an instantaneous terminal cancellation. Local marked value at END is not a
  completed-cycle profit. A later claim about closed-cycle value must resolve
  pending fills through a common cleanup horizon or explicitly censor the pair.
- In an absent state, compare POST now against WAIT through the next grid tick;
  thereafter both branches follow pi0 from their respective states. WAIT can
  therefore lead to later entry. Do not replace its continuation by zero.
- In a live state, compare KEEP through the next grid tick against PULL now;
  thereafter both follow pi0. KEEP permits a later cancellation. PULL remains
  exposed until its effect and retains its inventory reservations while pending.
- KEEP preserves order identity, age, residual size and queue position. A new
  generation starts a new queue position. Changes to the public book do not
  magically move later volume ahead of an already resting order.
- Pi0 must be recomputed causally on each branch's own state. Do not attach cached
  neutral-path predictions to a changed generation, horizon or remaining size.
  Unsupported state/target combinations fail the build or receive a declared,
  counted fallback; they cannot silently become ordinary calibrated predictions.

Let `R_T(action; pi0)` be the branch's incremental cash plus marked inventory at
the common END, including the fixed cost convention and all subsequent actions
under pi0. The prediction targets are:

```
Delta_post = E[R_T(POST now; pi0) - R_T(WAIT one tick; pi0) | information at t]
Delta_keep = E[R_T(KEEP one tick; pi0) - R_T(PULL now; pi0) | information at t]
```

POST only if Delta_post > 0. PULL only if Delta_keep < 0. At a tie, preserve
absence/the existing order. Availability and inventory constraints still apply.
Prior fills and inventory common to both branches cancel in the comparison;
never write off inventory merely because its resting order was cancelled.

These are **paired action values**, not the old single-order removable-fill
label. Delayed entry and re-entry can have different prices and fills. Version
the target/contract accordingly; do not force signed branch differences into a
nonnegative-fill-quantity interface or count continuation value twice. Queue
benefit belongs in the realized branch difference, not an invented reward.

**Exit gate G2:** paired branches reconcile to the generating engine with common
endpoints, causal state and pending fills; tests cover delayed entry, preserved
queue, partial fill, cancel pending, re-entry after acknowledgement, both sides,
terminal censoring and branch-specific predictions. A counterexample where
neutral holding loses but KEEP-one-tick then PULL gains must be represented
correctly, as must the converse. Record common-END boundary effects separately.

### 3. Train one controlled candidate on the aligned action targets

Generate paired labels at neutral decision opportunities, including no fills,
no action and exclusions. Avoid outcome-selected training. Reachable-policy
states are an additional distribution check, not a replacement for the neutral
population. Keep all decisions of an episode together in splits and weighting;
report both per-decision diagnostics and one realized return per episode.

For a fully chronological first development fit, use **Oct23 as pi0 warm-up**,
freeze its weights, generate pi0-conditional labels on **Oct24–25**, and evaluate
on **Oct26 and Oct27**. Existing models fitted through Oct25 are valid for step 1,
but cannot be reused as point-in-time teachers of Oct24–25. Record the reduced
history explicitly. Compare against that same pi0 and same-history controls;
do not attribute differences from the older three-day fit solely to the target.

Declare **one primary learned candidate**, with separate bid/ask POST and KEEP advantage
heads, the existing rich causal inputs, pinned LightGBM parameters and zero
action boundaries. Direct paired-advantage supervision is an explicitly flagged
head/target change, not a claim that a new algorithm solves the problem. Retain
quantity/movement/severity decompositions as diagnostics and keep the old neutral
target model as a named reference. Fit two fixed diagnostic controls on the same
paired labels and history: fair-edge/time-only inputs, and the existing mechanical
input subset. Record all three input variants and their four side/action heads
in multiplicity; the rich-input variant remains primary regardless of rankings.
No leaf/round/feature/horizon sweep in this step. If the model/interface design changes, version and declare the candidate
before reading its outcomes; count the added candidate.

At evaluation, replay the learned policy on its own state. `pi0` is the frozen
label continuation; the learned policy is a policy-improvement candidate, not
necessarily the same trajectory. Report its full replay and distribution shift
explicitly. Do not update pi0 and refit silently until a favourable result appears.

**Exit gate G3:** chronological development predictions preserve chronology and
target identity, reconcile with value mapping, and are calibrated at actual
decision boundaries. Good fill AUC or neutral-population rank alone cannot pass.

### 4. Evaluate economic value and useful quoting together

Replay bid and ask jointly before claiming two-sided quote availability. Enforce
pending reservations, shared inventory and shared-volume conservation. Keep a
single-order result clearly labelled if those capabilities are unavailable.
For timing robustness, recompute own-state inputs, predictions and actions in
each scenario. A frozen-trigger sensitivity remains a separately named control.

Keep N0, first-opportunity PULL-ALL, pi0, and the current neutral-target rule as
named controls. Use identical episodes, market hours, terminal conventions and
costs. For entry selection, use at least **200 predeclared matched control draws**
with admission counts matched by day/side/hour and causal opportunity strata;
run the same frozen downstream controller. For retention, match the starting
admissions and risk sets, report action/latency distributions, and use the
existing exact STATES expectation only where its assumptions still apply.
Never combine outcomes from incompatible paths as a synthetic profitable ledger.

Publish all fixed candidates, statuses and trade-offs. A tiny profitable subset
is entry-selection evidence, not proof of a practical market-making policy.

## Required evaluation metrics

All tables carry source/model/target hashes, as-of, UTC days, original episode
counts, unique order counts and exclusions. Split by day, side and scenario.
Never count overlapping decision labels as independent strategy returns.

| Dimension | Exact measurement and denominator |
|---|---|
| Primary economics | Realized cash-plus-inventory change at common END, after declared costs; cents per original opportunity and total cents. Report each policy's absolute value and paired difference to N0, PULL-ALL and pi0. |
| Participation | Attempted posts / original opportunities, accepted posts / attempts, filled orders / accepted orders, shares filled / shares submitted. Include zero fills and post-only refusals. |
| Quote availability | Time-weighted fraction of the same valid market-clock intervals with an engine-active, executable quote: bid, ask, either side and both sides. Posts before activation count as absent; cancel-pending quotes remain exposed until effect. Report client-confirmed presence and cancel-pending time separately. Report at-best availability separately from any resting quote. |
| Order lifetime | Acceptance-to-full-fill/effective-cancel/END duration: p10, p50, p90 and cause counts. END survivors are censored, not cancellations. Separately report submission-to-cancel request and request-to-effect. |
| First-review reversals | Model pulls at an order's first eligible review / accepted orders that reach that review. Also report the fraction among all model pulls, so the two denominators are not confused. Separate structural exits. |
| Churn | Posts, effective cancellations and replacements per valid market hour and per 100 filled shares; lifetime and availability alongside them. Zero-fill denominators are NA, not zero. |
| Queue preservation | Age, known/simulated queue-ahead and remaining shares at KEEP and fill; new-generation/reset count and shares executed before each reset. Preserve missing/unknown queue states. Simulated rank is not verified venue rank. |
| Early-fill loss | Value of fills before first eligible review, during earliest cancellation latency, and after earliest effect. Also report actual cancel-pending fills. Reconcile the components to total value. |
| Decision attribution | Paired value versus PULL-ALL split into first-review pulls, later pulls, never-pulled orders and unavailable decisions; bid/ask and order counts for every group. |
| Prediction calibration | Predicted versus realized paired advantage by fixed bins and action sign; bias/MAE and sums at the zero boundary, on neutral and reached states separately. Include contribution of quantity, movement and continuation errors. |
| Adverse selection / markouts | For each actual fill, side*(mid_at_fill - fill_price) and side*(mid_at_fill+h - fill_price), quantity weighted, at h = 100 ms, 1 s and common END where observable. Identify execution clock, mid source and missing endpoints; do not call these settlement P&L. |
| Tail loss | Probability and amount of loss and severe excess beyond 30 cents per episode; p95/p99 loss and worst-day value. This retains the existing numerical severity threshold but declares the new episode aggregation explicitly; also retain per-order diagnostics. No clipping/winsorizing realized tails. Match predicted and realized aggregation units. |
| Inventory / capital | Mean and peak absolute inventory, time-integrated inventory, pending reservations, cap breaches and terminal inventory. Report actual size distribution if caps bind. |
| Robustness | Per-day/per-side values, timing-scenario range and each candidate's relative value. Never sum alternative scenarios into expected return; no probabilities are identified by the scenario bank. |

The markout horizons above are diagnostics, not additional fitted target or
threshold candidates. Compute only within declared available endpoints; an
unavailable 1 s markout cannot justify opening a protected next-day hour.

## Computed decision gates and stop rules

- **Correctness:** zero unexplained target/time/queue/accounting mismatches; every
  invalid forecast and exclusion counted; no future-dependent complete input.
  Exact accounting tolerance is declared in the checker before results.
- **Economic development gate:** positive absolute common-END value versus N0
  and positive paired value versus the same-admission PULL-ALL and pi0 controls
  on each of Oct26/27; report bid/ask separately. Negative sides are not removed
  post hoc. A failure remains a failure even if losses improve greatly versus A+0.
- **Prediction gate:** selected POST and KEEP advantages have positive realized
  mean, while selected PULL cases have negative realized Delta_keep, by day/side
  where supported. Compare positive and negative predicted-advantage totals
  against realized totals within each group: same sign and ratio in [0.8, 1.25].
  Zero realized totals or empty groups are NOT_EVALUABLE, never PASS. Fixed signed
  advantage-bin edges are [-infinity, -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10,
  infinity] cents, with exact zero reported separately. The candidate's realized
  value must exceed the matched-control mean on each development day; report
  the full control distribution without calling it a two-day confidence bound.
  Carry at least 200 matched draws or the justified exact expectation. These
  screening thresholds remain fixed; changing them creates a declared revision.
- **Practical development gate:** on the same admissions, both-side at-best
  availability and median live lifetime exceed first-opportunity PULL-ALL, while
  first-review reversal and cancellations per filled share decrease; the economic
  gate must pass simultaneously. Report the complete value/participation table
  rather than selecting a new threshold. If fewer than **200 distinct admitted
  orders per day/side**, call the practical result insufficient-support. This is
  a declared engineering screening floor, not a statistical confidence guarantee.
- **Bounded versus full strategy:** episode tests can pass a research gate only.
  Continuous two-sided replay must independently establish inventory/capital
  accounting, shared fills, replenishment and quote availability before a
  practical-strategy claim. No absolute venue service-level target is inferred
  from these two days or enforced by holding losing orders.
- **Forward gate:** after development, commit the entire selected pipeline,
  feature/target/continuation definitions, parameters, thresholds, controls,
  candidate count and source hashes. Then use at least five later untouched
  complete UTC days, with day-cluster intervals. Oct23–27 and previously accessed
  Oct28 cannot serve as that validation. This plan does not open protected data.

If G1 shows that existing entry forecasts remain negative with cancellation,
continue to the paired-target diagnosis, not an automatic larger-size test.
If action-value prediction is accurate but replay loses, inspect controller
state/continuation/inventory effects. If predicted-positive advantages realize
loss, inspect movement/tail calibration and distribution coverage before policy
threshold tuning. If value improves only by ceasing to quote, retain the result
as a risk-reduction finding; do not declare the market-making objective achieved.

## Handoff and next deliverable

**Immediate development deliverable: G0 repairs followed by the five-policy G1
ledger/table, with the complete metric schema and common END.** This is the
smallest useful next test and uses existing consumed artifacts where possible.
Then freeze the pi0/branch contract before building new labels and the one-step
candidate. Record every stage's commit, receipts, statuses and gate result in
the repository handoff. This review session will verify those artifacts; it does
not implement the other session's changes, push, merge or restart scheduling.
