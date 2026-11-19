# Predict.fun: complete adaptive entry and retention research plan, revision 2

User request, 2026-11-19: generate the complete plan, incorporating the latest
review and proposed metric refinements. Implementation belongs to the development
session; this session owns planning and independent review.

This is the prospective working plan. It supersedes the next steps and future
gates in [revision 1](20261119_entry_retention_priority.md). Earlier results keep
their original verdicts. This document is not a completed experiment or a model
freeze. The [parameter manifest](20261119_adaptive_mm_plan_v2.yml) records the
declared settings; executable builders, controls and artifacts must still be
committed before the experiment. No scheduler or live trading is introduced.

Basis: result **edfeda22dd72aacaee0c758e96e6e29fe2667422**, reviewed in
**0026ab85a124df2aa360e2cb647294a4eca8f6f9**. See the
[review and reproduction evidence](../reviews/automatic/20261119_edfeda22/report.md).
The prior zero-quantity coherence defect is resolved. The two new diagnostic
defects remain the immediate work. The user skipped revision 1's G1 baseline;
this plan does not reinstate it.

## 1. Objective, scope and order of work

Build a policy that creates positive economic value while maintaining useful
resting quotes. Evaluate each side and each order separately, with a controller
that respects shared inventory, pending exposure and queue priority.

The sequence is:

1. Correct the current diagnostic receipt without refitting.
2. Build and verify genuine POST/WAIT and KEEP/PULL branches.
3. Freeze a supported, causal continuation and generate representative labels.
4. Fit one declared prediction candidate and measure its action-value accuracy.
5. Replay that candidate on its own state; separate entry, retention and inventory
   effects using valid controls.
6. Change one model component only if the diagnosis identifies its failure.
7. Freeze the complete selected pipeline, then evaluate later untouched days.

Keep prediction, value mapping and action execution distinct. A good prediction
score is not a policy return; a lower cancellation rate is not an economic gain.
Do not reward quote age, penalize cancellation by an invented amount, enforce a
minimum resting time, or add a fixed retry cooldown. Reconsider eligible actions
on the decision grid using current state. Pending acknowledgements and hard risk
limits are execution constraints, not retry timers.

The first claim is a bounded research result. Multi-layer quoting, larger sizes,
price-level optimization and live execution are outside this first experiment.
Continuous portfolio replay is a later, separately measured gate.

## 2. Fixed settings and explicit changes

| Item | Revision 2 declaration |
|---|---|
| Market/universe | Same reviewed Predict.fun universe and side/instrument definitions; record instrument IDs and settlement convention, without substituting another venue's conventions. |
| Size and price | 30-share best-price JOIN per new generation, using the decision-time book; no size/price sweep. A post-only rejection remains an outcome. |
| Actions | Reconsider every 50 ms; at most one live or pending generation per side in the first experiment. No additional quote layers. |
| Baseline timing | Existing M50 scenario, activation 58 ms, cancellation 37 ms, fill notice 100 ms; preserve their exact engine definitions and event ordering. |
| Queue | Existing corrected queue rule M; keep identity/age/remaining quantity, and preserve priority against later arrivals. |
| Bounded endpoint | Original episode start +5,000 ms, shared by every descendant generation and both branches. Never restart the endpoint on repost. |
| Inventory budget | New bounded research setting: initial inventory zero, absolute position cap 30 shares per instrument; reserve all outstanding bid and ask exposure. No fractional resizing to fit the cap. Apply identically to all new adaptive controls. |
| Prediction base | Current rich 91-input representation, preserving causal book/flow dynamics; add only mandatory action/portfolio context identified in section 5. Pin the exact final list. |
| Learner | Existing LightGBM parameter dictionary, seed, threads and 200 rounds. Record resolved values, not only an import path. No hyperparameter search in the first candidate. |
| Decision boundary | Positive POST-minus-WAIT -> POST; negative KEEP-minus-PULL -> PULL. Ties preserve absence/order. Structural/risk constraints are separately logged. |
| Costs | Copy and hash the current Predict.fun cost convention. List fees, rebates and unmodelled costs explicitly; no silent change or import of Binance retail assumptions. |

The inventory cap, new action-state coverage, genuine continuation and revised
gates make this a **new experiment**, not a target-only change to the old run.
Do not present differences from the old independent-order results as an isolated
benefit from a new target or model. All new controls share the new environment.

For the bounded cap, client-known position `I`, outstanding worst-case bid
exposure `B`, and ask exposure `A` must satisfy `I+B <= 30` and `-I+A <= 30`.
Reservations include submitted/pending orders and fills not yet acknowledged;
they are transferred on events rather than released on cancel request. Use an
independent engine-position check to catch double allocation. Opposite pending
quotes do not cancel each other's worst-case reservations.

## 3. Stage R0: repair the current evidence, with no new model

Deliver a versioned, superseding receipt for the existing restricted experiment:

- Intersect each head's reached RETAIN mask with that head's actual usable POST
  admissions, joined by day/episode/order identity. Stop at the first issued
  pull and distinguish observations while cancel-pending from new decisions.
- Rebase `pair_qm` to the current decision's reference mid, or retire that field
  as an economic component. For the existing fixed-price experiment verify
  `pair_v = sign*(mid_now-price)*pair_q + pair_qm_now` on every row.
- Name the current alternative **NO_ENTRY to END**. Do not call its zero value
  evidence about adaptive WAIT or later entry.
- Preserve fitted models, `pair_v`, action thresholds and old artifacts. Publish
  corrected counts, bins, gates and unchanged baseline value; investigate any
  unexpected change rather than silently accepting it.

Use the review's two independent counterexamples and positive controls. Re-run
the relevant G0 coherence cases to prevent regression. Validate source/model/
input hashes and status populations. Do not refit to repair reporting.

**R0 gate:** no unexplained component mismatch; no reached row without admission;
corrected subgroups and honest names in a new receipt; existing direct-value
baselines reproduce. The latest candidate remains failed under its original
rules. Output: correction report, machine receipt, tests and source commit.

## 4. Stage R1: the causal branch engine and value contract

### State, clocks and executable actions

Each state contains episode and generation IDs, instrument/side, common endpoint,
client-known and engine position, cash, known remaining shares, queue state/age,
pending post/cancel events, outstanding reservations and the last available
market inputs. Decision inputs cannot read future engine fills or acknowledgements.

| State at decision | Allowed behavior |
|---|---|
| Absent | POST a feasible order or WAIT to the next grid tick. |
| Post pending | Process events; no duplicate post or invented immediate cancellation. Follow the engine's declared pre-activation cancellation semantics. |
| Live, including partial fill | KEEP the same generation or request PULL for its remainder. |
| Cancel pending | Continue exposure and reservations until actual effect/notification; no replacement using supposedly freed capacity. |
| Absent after fill/cancel acknowledgement | Re-entry can be considered at the next eligible grid tick. |
| Terminal | No new exposure. Preserve cash, inventory and pending liabilities; record terminal censoring. |

Store source event/receipt, decision, submission, activation, cancel-request,
cancel-effect, match, fill-notice and endpoint timestamps distinctly. Keep the
scenario's tie ordering. A request is not its effect; a fill notice is not a
match. Tests must cover events immediately before/at/after a boundary. Timing
scenarios are assumptions, not recovered exact venue matching times.

Initially verify each side in isolation, then the joint book. In joint replay,
process sides in a deterministic alternating order based on episode/grid index.
After any issued command, update pending state/reservations and rebuild inputs
before considering the other side. Never act twice on stale marginal values.
Shared public volume can fill the simulated portfolio only once; use branch-
local books and queues for alternative histories.

### Actions and common continuation

At time `t`, fork the same client-known and engine state:

```
Y_post(t) = R_T(POST now, then pi0-50) - R_T(WAIT one tick, then pi0-50)
Y_keep(t) = R_T(KEEP one tick, then pi0-50) - R_T(PULL now, then pi0-50)
```

Both branches resume the same frozen `pi0-50` at the next grid tick from their
own states. It handles the other side and pending events too. WAIT can later
post; PULL can later re-enter after acknowledgement. KEEP retains the original
queue position; replacement receives a new identity and queue position.
Use the same exogenous tape within each pair, not the same cached order state.

`R_T` is cash change plus marked inventory change at the common END, less the
declared execution costs, across **all descendant generations and both sides**
in that branch. Prior inventory/cash common at the fork cancel in the difference;
subsequent inventory changes do not. This is a marginal portfolio action value,
not a promise to add simultaneous independent bid and ask values.

Cash, buys, sells and terminal inventory must reconcile independently. Include
the same immutable END mark for both branches. Per-fill quantities/prices differ
between generations; never apply a single original quote price to all fills.
Signed branch quantity differences are allowed. Do not force these targets into
the old nonnegative removable-quantity contract. The fixed-price `pair_qm` formula
in R0 is not a complete multi-generation value decomposition.

No free terminal cancel or forced liquidation is invented. Bounded mark-to-END
value must disclose live/pending orders and unresolved exposure. A later
closed-cycle claim requires a common, declared cleanup horizon and its costs;
missing cleanup data are censored, not silently zero-valued.

### Required falsifiers and R1 gate

Every check needs a valid case plus a known-bad case that it rejects:

| Behavior | Required case |
|---|---|
| Actual WAIT | WAIT posts on a later tick in at least one nondegenerate synthetic episode; POST and WAIT have different valid histories. |
| Re-entry | PULL remains exposed while pending, then can produce a new generation at a different price. |
| Queue preservation | Partial KEEP retains priority; later snapshots/arrivals cannot jump ahead; a replacement goes behind prior queued volume. |
| Economic identity | Fill ledger equals cash-plus-inventory value for every branch, both sides and the pair difference; no double-counting common fills. |
| Continuation matters | Genuine engine-generated cases in both directions: neutral hold versus keep-then-follow-pi0 can change value sign through differing subsequent actions. Hand-written label arithmetic is insufficient. |
| Causality | Changing only events after a decision cannot change its full feature vector or action; changing a relevant known event can. |
| Boundaries and capacity | Match around cancel effect, delayed fill notice, cap saturation, post-only refusal, near-END post, missing mark and unavailable feature. |
| Identity/support | Wrong generation/END/scenario/model horizon is rejected. A fallback has a reason and is counted. |
| Shared book | Bid/ask actions reconcile inventory and shared fills; processing the second action uses updated reservations. |

**R1 gate:** all controls pass before data fitting. There is an executable WAIT
branch, a versioned `paired_continuation_v2` contract, traceable lineage and no
unexplained accounting/causality mismatch. First use synthetic data, then a
deterministic bounded slice of already consumed material. Do not start a full
label build to discover elementary engine errors.

## 5. Stage R2: supported continuation, representation and labels

### Freeze a teacher that supports the new states

`pi0-50` uses the prior E1 convention as its starting rule: positive coherent
neutral continuation value permits POST; negative coherent value requests PULL.
Its meaning and artifact version are separate from the old `PI0` result.

The legacy POST head was trained on births with a five-second horizon. Do not
apply it unchanged to a later generation with, for example, 700 ms left. Build
neutral shadow-order probes on **Oct23 only**, including decision-grid entry
offsets, remaining horizons and partial/queue states that the new engine can
produce. These probes value their own neutral structural continuation to the
same END. Fit the same pinned J architecture on those warm-up targets, then
freeze its models, complete policy and support manifest before Oct24–27 labels.
This is necessary teacher preparation, not a candidate/model-search arm.

Teacher availability/risk rules use the full causal state even if the J head
does not model inventory utility. The cap is a hard constraint; there is no
learned or arbitrary inventory penalty in this first run. Count absent-post
opportunities blocked by capacity separately from negative model values.

Specify training-domain checks for horizon, side, price, quantity, queue/age,
generation and feature availability. Physical validity alone does not establish
training coverage. Report counts in predeclared support strata and out-of-range
values. Unsupported states use an explicit WAIT/KEEP fallback and remain in
performance counts; a test dominated by them cannot establish model quality.
If the teacher never reaches delayed entry on real consumed probes, report that
degenerate continuation and stop before claiming a genuine timing experiment.
Do not fix it by silently changing thresholds or resetting END.

### Features and target populations

Audit the existing inputs before adding speculative signals. Preserve book/flow
changes over trailing windows, imbalance and depth, volatility, fair-value
response, order age/queue/remaining size and missingness. Version units and
trailing-only scaling; fit any learned transform on training days only. Check
clipping, nonfinite values, near-expiry prices and regime coverage by day/side.

Add mandatory context where absent: time to common END, generation age, known
inventory, side-specific reservations and pending flags, plus other-side order
state used by the controller. Declare their exact names, units and handling
before fitting. Identity fields remain join keys, not predictive shortcuts.
The rich-input candidate and controls share the same availability masks and
mandatory action context; publish their final input-list hashes.

Build an outcome-independent opportunity/state census from the neutral reference
path. For delayed generations, extend that census with deterministic shadow
probes and chronological pi0 branch rollouts; label each source population
separately. Do not substitute only successful fills, only positive teacher
scores, or only states the learned candidate later visits for the census.
Unsupported or infeasible actions are statuses, not negative labels.

Row key: day, instrument, original episode, generation, decision time, side,
action pair, continuation version, scenario, END and reference-state hash.
Save both complete branch returns, their difference, terminal inventories,
costs and event-ledger hashes. Preserve zero/no-fill/rejection outcomes. Raw
cash/quantity components and all price references must round-trip to value.

Chronology remains **Oct23 teacher warm-up; Oct24–25 candidate training;
Oct26–27 consumed development evaluation**. Purge any episode whose target
endpoint crosses a split boundary. All descendant generations stay with their
original episode. Oct28 was already accessed and is not validation.

For compute control, pin a deterministic sampling manifest before labels:
training POST/WAIT and KEEP/PULL probes every fifth grid tick, rotating phase by
a fixed episode hash; retain all original-time POST probes. Evaluate every
eligible 50 ms tick. Record every sampled/unsampled status. Sampling changes
training volume, not action latency. Use equal total training weight per
episode/side/head across its included rows, so long-lived orders cannot dominate
merely by contributing many snapshots. This weighting is a declared revision;
apply it to all same-target controls. No outcome-based downsampling or tail clipping.

**R2 gate:** matched branch targets reconcile to the engine; full census/statuses,
chronology, support and feature causality pass; pi0-50 and the complete label
builder are committed. Every exported component passes its own check, not only
the total value. Synthetic or sampled checks are clearly distinguished from
whole-artifact checks. Output new target/version paths; retain old artifacts.

## 6. Stage R3: one prediction experiment

The primary candidate **ADV2-RICH** has separate bid/ask x POST/KEEP heads and
regresses `Y_post`/`Y_keep` directly. Its output is expected action advantage
under pi0-50, in cents, not a permission bit. The controller applies signs,
constraints and fallback behavior. Do not add continuation costs a second time
to an already complete branch-value target.

Fit fixed **ADV2-FAIR** and **ADV2-MECH** input controls on the identical targets,
rows, weights and history. Rich remains the primary regardless of rankings.
Twelve candidate heads are declared. Count teacher and any reference heads
separately; count all subsequent alternative candidates in the experiment ledger.
Retain a same-history neutral-target E1 reference with supported new-state inputs
as a separate target/control, rather than relabelling its scores as advantages.

Measure prediction against the declared pi0-50 paired labels in two populations:
the neutral census and the states actually reached by each candidate. Reach
requires admission, correct generation, current liveness and no already-pending
pull. On new states, regenerate paired labels from that state; never join a
cached neutral label solely by a time or original order ID.

Report fixed bins, selected POST, WAIT, KEEP, PULL and ties/invalid fallbacks.
Maintain raw row counts and episode-balanced means. Summed overlapping labels
are diagnostics, not policy value. Rank, fill AUC and low MAE alone cannot pass.

**R3 gate:** identity/chronology correct; the prospective prediction screen in
section 9 computed without dropping failed sides/groups; full support counts
and all candidates disclosed. A failed screen permits a bounded diagnostic
policy replay to explain the failure; it does not permit promotion or a silent
new fit. This separates an informative failed experiment from validation.

## 7. Stage R4: own-state replay and valid controls

Replay every policy with its own order state and current causal inputs. Start
all policies from the same episode state, end them at the same absolute END,
and use the identical cost/risk settings. Value each episode once. For the
timing bank, recompute actions under each scenario; keep frozen-trigger
sensitivity under a separate name. Never average scenarios without justified
probabilities or call their count an independent sample size.

| Control | Meaning and comparison |
|---|---|
| N0 | No new entry throughout the episode. Zero incremental value from zero initial inventory. |
| PI0-50 | Frozen warm-up continuation, including actual WAIT and re-entry. |
| NEU2 | Same-history neutral-target reference, with the new state/horizon support; legacy E1 remains a historical control only. |
| ADV2-RICH / FAIR / MECH | Identical controller mechanics; each uses its own frozen advantage predictions. |
| ENTRY-ONLY | RICH entry decisions, PI0-50 retention, recomputed on this hybrid's own state. |
| RETAIN-ONLY | PI0-50 entry, RICH retention, recomputed on this hybrid's own state. |
| ENTRY-PA | RICH entry rule and first-eligible-review cancellation, with real latency and own-state re-entry. |

The two hybrids are predeclared attribution controls, not new fitted candidates
to select after observing results. Publish their interactions: the combined
policy's value need not equal the sum of entry and retention gains.

**Re-entry changes matching.** ENTRY-PA can have different later admissions from
RICH; do not call it a same-admission policy. The old saved-fill A-versus-A-PA
comparison is exact only for the restricted single-generation experiment.
For a controlled retention contrast, fork at a common admitted state and apply
KEEP/PULL or first-review cancellation, then the same pi0-50 continuation on
both branches. Report these local paired contrasts separately from full-policy
returns. Do not force the candidate's later orders onto a control path where
they are infeasible.

Use a matched **local action-information control**: before reading labels,
select at most one deterministic scheduled anchor per original episode/side/
action pair, with unavailable states retained as statuses. Within day, side,
market-window and minute strata, randomize the selected action count across
eligible anchors; each selected branch follows the same frozen pi0-50. Report
at least 200 fixed-seed draws or a mathematically justified exact expectation
for this additive, disjoint-anchor estimand. Match on the decision, not filled
volume. No stratification or inclusion based on future profit or fill outcome.

This estimates local action-selection lift, not the return of a continuously
randomized strategy. Never substitute it for RICH's own-state replay or stitch
overlapping branch returns into portfolio P&L. Any whole-policy randomized
control requires full replay and a separately declared matching scheme.

**R4 gate:** predicted-target evidence and actual replay are reported separately;
all ledgers reconcile; economic and relative quoting screens in section 9 pass
before advancement. Explain first-review reversals by known changes in market,
queue/remaining state, predicted advantage and continuation—not just their count.
Where a cause is not identifiable, say so. No fabricated causal decomposition
from feature importance alone.

## 8. Metric contract

Every table includes hashes, as-of, universe/days, scenario, costs, original
episodes, unique generations, eligible/admitted/active/reached states, no-fills,
refusals and all exclusions. Show day/side splits and a joint portfolio total.
An unavailable metric is NA with a reason, never zero. Use the same valid market
clock for every policy; missing data cannot improve a policy's denominator.

| Dimension | Definition and interpretation |
|---|---|
| Primary economic value | Cash-plus-marked-inventory change less declared costs at END: total cents, cents per original episode and per valid market hour; paired differences versus N0, PI0-50, NEU2 and ENTRY-PA. Include unposted/no-fill episodes. |
| Prediction versus controller | Predicted/realized `Y` under pi0-50, separately from `R_T` under the candidate. A label-calibrated action score does not itself predict the result of repeatedly following a different policy. |
| Entry/retention contribution | Fixed hybrids plus local paired contrasts; report interaction and population changes. Repeated decision effects are not summed as trading returns. |
| Participation | Episodes with any attempt / original episodes; attempts per episode; accepted / attempted generations; filled / accepted generations; filled / submitted shares. Show first entry and re-entry separately. |
| Quote availability | Engine-active executable quote time / common valid clock, for bid/ask/either/both; separately at best and any price. Pending posts are absent, cancel-pending orders remain exposed until effect. Separate client-confirmed presence. Joint two-sided time requires joint replay. |
| Lifetime | Activation to full fill/effective cancellation/END: p10/p50/p90 by cause and overall; END survivors censored. Report submission-to-request and request-to-effect. Faster successful fills are not a failure. |
| First-review reversals | Model pulls at first eligible review / admitted generations actually live and actionable at that review; also among all model pulls. Separate structural exits and pending states. |
| Churn and queue | Posts, effective cancels, replacements per valid hour and per 100 filled shares; resets, shares filled before each reset, age/remaining/queue-ahead at KEEP and fill. Zero filled-share denominator -> NA. Simulated rank is not venue-verified rank. |
| Timing loss | Partition fill value before first actionable review, during earliest effect delay and after earliest effect; additionally actual cancel-pending fill value. Disjoint partitions reconcile; overlapping diagnostics are not added together. |
| Calibration | Episode-balanced mean prediction, mean label, bias, absolute error and fixed-bin reliability for each action group; raw counts and diagnostic sums separately. Show neutral and admission-aware reached populations. |
| Support | Distinct episodes, nonzero-label episodes, loss/severe episodes and UTC days; input/horizon/generation/inventory strata and missing/fallback rates. Thousands of repeated zero decisions are not thousands of informative outcomes. |
| Fill markouts | Quantity-weighted `sign*(mid_fill-price)` and `sign*(mid_(fill+h)-price)`, h=100 ms, 1 s, common END. Separate instantaneous spread from subsequent signed mid movement. Record clock/mid source and missing endpoints. No unsupported comparison to top makers using different populations or clocks. |
| Unconditional tail cost | For episode value V: mean `max(-V,0)`, P(V<0), P(V<-30 c), mean `max(-V-30,0)` over all original episodes; total loss and severe-loss contribution. Retain zeros and all tails. |
| Conditional tails | Additionally p95/p99 of nonnegative loss `max(-V,0)` among filled episodes, and among loss episodes, with separate denominators/counts; largest losses and share of loss from worst 1% episodes. These are diagnostics, not outcome-selected training. |
| Inventory/capital | Known and engine inventory, mean/peak absolute and integrated inventory, submitted/pending reservations, cap breaches and terminal inventory/pending liabilities. Report position duration separately from quote lifetime. |
| Robustness | Per-day/side values, named scenario range, source/clock/queue sensitivity. No probability interpretation of scenario frequencies. No confidence interval below five complete UTC days; report temporal dependence limitations. |

Markouts do not change the fitted target horizon. Do not open a protected
next-day hour to fill a missing 1 s endpoint. A zero unconditional p95 with sparse
fills does not demonstrate low tail risk. If severe-loss predictions are not
output by the model, report the realized costs and explicitly say that severe
probability calibration was not measured.

## 9. Prospective gates and support rules

These rules apply to revision 2 candidates only. Retain the earlier ratio
diagnostic and original verdicts for old runs. The new absolute calibration
tolerance is an explicit engineering screening choice, not a statistical bound
or an estimate selected from a successful result.

| Gate | Required evidence |
|---|---|
| Correctness | All phase-specific positive controls/falsifiers; no unexplained identity, causal, accounting, queue or reservation mismatch. Compare value identities at absolute 1e-6 cents; moment coherence retains its existing 1e-9 absolute/relative convention. |
| Prediction support | Per day/side/action-selected group: at least 200 distinct episodes and 20 distinct episodes with nonzero paired label. Otherwise INSUFFICIENT_SUPPORT, not PASS. Report realized severe-event count; no tail-calibration claim on sparse/unmodelled severe events. These floors do not establish statistical power. |
| Prediction sign | Supported selected POST/KEEP groups have positive mean paired advantage; selected WAIT/PULL groups have negative mean paired advantage, under pi0-50. Zero/empty group -> NOT_EVALUABLE. Ties and invalid fallbacks separate. |
| Prediction amount | For each supported group, `abs(mean_prediction - mean_label) <= 0.05 c + 0.20*abs(mean_label)`, using the declared episode-balanced weights. Report [0.8,1.25] ratio diagnostic where meaningful, but it is no longer the sole near-zero amount gate. No per-side tolerance fitting. |
| Incremental information | On the disjoint local anchors, realized selected-action advantage exceeds the matched-control mean on each development day. Report the full draw distribution or exact-mean derivation, not a two-day confidence claim. |
| Economic development | At baseline timing, RICH has positive absolute episode value and positive paired full-policy value versus PI0-50 and ENTRY-PA on each of Oct26/27. Report NEU2 and all sides; never remove a negative side post hoc. Local retention contrasts use their own matched states, not forced future admissions. |
| Relative quoting development | With economic gate passing and >=200 admitted generations from distinct episodes per day/side: joint at-best availability exceeds ENTRY-PA; first-review reversal and cancellations per 100 filled shares are lower. Compare PI0-50 coverage too. No mandatory increase in median lifetime. NA/support failure cannot pass. |
| Risk correctness | Zero unexplained cap/reservation/shared-fill violations, and complete tail/terminal exposure reporting. A return increase coupled to worse inventory or severe loss is a disclosed trade-off, not automatic risk-adjusted improvement. |

If a policy makes no PULLs or no WAITs, the corresponding calibration group is
not evaluable, not evidence of perfect discrimination. The operational policy
can still be described, but a claim of calibrated bid/ask timing is not earned.

The quoting gate establishes **relative research improvement**, not an absolute
venue service level. A 0.1 percentage-point uptime increase does not establish
practical market making. A practical-strategy claim additionally requires a
committed absolute coverage and capital/service budget, continuous joint replay
and terminal handling. No service-level percentage is reverse-engineered from
these two consumed days or enforced by keeping losing orders.

Every gate is computed from artifacts. Separate PASS, FAIL, NOT_EVALUABLE and
INSUFFICIENT_SUPPORT. No manual promotion because one attractive metric improved.

## 10. Stage R5: diagnose, then change one component if needed

| Observed failure | Next experiment, after correctness |
|---|---|
| Wrong value identity, clocks, admissions or lineage | Fix builder/evaluator and issue superseding receipts; no model tuning. |
| Delayed states or remaining horizons unsupported | Expand neutral probes on already permitted earlier material; preserve chronology. Compare history/coverage using unchanged targets and parameters. |
| Good quantity/flow prediction, poor economic sign | Audit joint fill/movement and branch continuation error. No claim that fill AUC establishes profit prediction. |
| Sparse heavy-tailed value dominates direct regression | One declared component candidate: predict unconditional expected gain, ordinary loss and severe excess on all decision exposures, then combine to expected advantage. Do not drop zero/no-fill actions or clip realized losses. |
| Stable rank but biased amount | One declared calibration map fit on an earlier held-out development block, chronologically before its evaluation; test sign/amount and replay separately. |
| Clear variance/overfit at unchanged target | One explicitly flagged smoother parameter candidate, such as larger leaves/fewer rounds. Predeclare the exact parameter set; keep the original pinned run as control. No grid search on Oct26/27. |
| Accurate pi0-conditional scores but poor repeated-action replay | Investigate policy/state distribution shift and continuation mismatch using the hybrids. Any policy-iteration round is a new teacher/target version and counted candidate. |
| Good local execution, poor inventory result | Test one declared inventory valuation/control extension in continuous joint replay. Do not conceal inventory losses by reporting fill markout alone. |
| Improvement exists only in optimistic timing | Diagnose delay-sensitive action value on recomputed scenario states; do not claim a forecast recreates zero latency without an actual delayed-policy comparison. |

Choose one row based on the diagnostic evidence and commit the candidate before
reading its new outcomes. This plan does not launch all alternatives. Additional
calibration requires a distinct earlier fit block; Oct26 cannot calibrate and
validate itself. All accessed dates stay consumed regardless of their name.
Do not silently update the teacher until a favorable result appears.

For the optional calibration experiment on the existing dates, the concrete
chronology is teacher Oct23, candidate fit Oct24, calibration Oct25, evaluation
Oct26–27. Compare with an uncalibrated Oct24-only candidate. This changes training
history and is a separately counted experiment; fitting the candidate through
Oct25 and then calling Oct25 its held-out calibration set is forbidden.

For the component option, an exact additive target identity is
`Y = max(Y,0) - min(max(-Y,0),30) - max(-Y-30,0)` in cents per paired action.
Train those unconditional component amounts on the full action population.
Paired-action severe excess is a different unit from portfolio-episode severe
loss; never compare their probabilities or thresholds as if they were identical.
Probability/severity heads are additional declared outputs only when tested.

## 11. Stage R6: continuous replay and forward evaluation

Before a practical-strategy claim, run continuous joint replay with inventory
carried across episodes, replenishment, shared volume and pending events. The
bounded 30-share cap is an engineering baseline, not an asserted optimal capital
budget. Keep it as the initial continuous reference; any changed cap, size,
price ladder or inventory penalty is a new declared experiment. No independent
five-second inventory resets in a continuous ledger.

Use a calendar-anchored five-second valuation grid for the first continuous
extension, independent of post/repost times. Carry live generations, cash,
inventory and pending events across those boundaries. The next common END can
advance at a calendar boundary, but that is a new forecast target, not a new
order or an instantaneous cancellation. Probe and train on the resulting older
generations, nonzero opening inventory and new remaining horizons using earlier
permitted data. Version this continuous contract and candidate; never apply the
bounded zero-inventory training population to these states without checking
support. Freeze this extension before forward evaluation if the forward claim
is about continuous market making; a bounded-only freeze supports only bounded
claims.

After development, commit the full selected builder -> labels -> fit -> forecast
adapter -> controller -> replay -> metrics pipeline; pin resolved parameters,
features/transforms, teacher, sampling/weights, costs, scenarios, risk limits,
gates, candidate count, controls and output schema. A plan commit is insufficient.

Register at least five later complete UTC days that are genuinely untouched,
using the access ledger before reading their outcomes. Oct23–28 cannot serve.
Respect existing protected-data boundaries; this plan itself opens no protected
dates. Predeclare the exact days or consecutive-date eligibility rule, missing-
day handling, interval method and stop rule before access.

Default forward design: register exactly five consecutive eligible complete UTC
days and an eligible-day rule based only on collector/data integrity, with no
outcome-based exclusions or optional extension. Incomplete days get a status;
any replacement follows the registered calendar rule, not their observed value.
Primary statistical summaries are daily total value and paired daily differences
versus PI0-50 and ENTRY-PA. Use Student-t intervals on those three daily series,
with Bonferroni adjustment for a 95% simultaneous family (each two-sided interval
uses level `1 - 0.05/3`, degrees of freedom `G-1`). Promotion requires all three
lower bounds above zero, plus the locked prediction/quoting/risk gates. Report
individual days and the assumptions: this procedure is fragile at G=5 under
strong skew, serial dependence or regime change; it does not guarantee normality
or effective sample size. A larger predeclared sample or a different interval
method is a versioned design change before access, never a rescue after failure.

Use one frozen primary candidate. Apply the locked evaluator and report all
days, failures and controls. For forward mean value and policy differences,
use UTC-day units, never decision-row bootstrap intervals; at least five
days is a minimum and may still be inconclusive. No early stop after a favorable
day, extension to rescue failure, or refit on those days followed by calling
them untouched. Any revised model
must validate on a newly registered later set.

A completed forward report distinguishes value, quoting, risk and prediction
gates; it does not authorize live trading. An inconclusive interval is not PASS.
Deployment, exchange integration and monetary commitments remain out of scope.

## 12. Deliverables, execution and ownership

Each stage ships code commit, machine receipt, concise report, exact inputs and
output hashes, test logs, original/population/status counts and as-of. Corrected
receipts supersede old versions in-band; do not edit frozen artifacts in place.
Keep large data under ignored data paths; commit only small summaries/evidence.

| Deliverable | Owner | Dependency |
|---|---|---|
| R0 corrected restricted-result receipt | Development; review independently verifies | Start now; no fit |
| R1 branch engine/contract and synthetic traces | Development | R0; no full label run before falsifiers |
| R2 supported pi0-50, census, feature/label manifests | Development | R1 |
| R3 fixed candidate and prediction report | Development | R2 |
| R4 own-state replay, controls and complete metrics | Development | R3; failed candidate can receive bounded diagnostic replay |
| R5 one evidence-selected revision, if needed | Development; review verifies declared change | Diagnosed R3/R4 failure |
| R6 continuous/forward report | Development; independent review | Locked complete pipeline and proper data access |

Use the owning session's slot only through
`/home/yuqing/ctaNew/predictfun/research/run_light.sh`, with absolute script paths.
Codex reviews use `PF_SLOT=codex`; a busy slot means defer, not use the other
session's slot. Keep one light process per slot, existing process/disk limits,
and <=2 GiB working-set target. Chunk before expensive joins; measure peak RSS,
not just a check between large allocations. Persist only compact branch ledgers
and regenerate large traces on demand. Check correctness on small consumed
samples before estimating the full build's runtime/disk footprint.

Routine reviews do not launch full fitting or raw-tape experiments, implement
another session's fixes, push, merge or message externally. Preserve concurrent
changes. Review/plan commits carry `Predictfun-Review-Only: true`; automation
remains disabled. This planning session runs no research experiment.

**Immediate next priority: R0's superseding paired-diagnostic receipt.** Then
build the genuine branch engine and supported continuation; parameter tuning
follows a specific prediction failure, not a low quoting percentage alone.
