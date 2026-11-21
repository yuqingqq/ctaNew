# DEFAULT-POST-LOSS-MAP-v1 — executable development handoff

2026-11-21. USER-directed change of priority. Prepared against repository HEAD
**8a990d5f2c87de732206aef70eae78cb8a02257a**. This is a plan, not a new result or
a review of the unfinished coverage build. Development owns implementation;
Codex owns the combined review. No strategy or engine source is changed here.

## Decision and scope

**Next priority: simulate a continuously replenished, two-sided default-post
baseline, account for its gains and losses, and test whether dangerous exposure
was identifiable in time to avoid it.** The question is economic selection,
not whether a controller can reduce losses by almost never quoting.

The stop is now recorded in commit
**7a2478a004d9c251d85475a418d367134927e423** (partial build, no accepted result).
The USER has told development to stop the previous work. Keep its source,
manifests, logs and completed partitions; mark it stopped/superseded, without
accepting unfinished outputs or resuming it automatically. This instruction
supersedes the next-priority portion of review 216ff603 and the six-date
coverage launch priority. More data may be useful later, but it is not the
immediate experiment. No new dates are necessary for this round.

Proceed without another approval within this contract. First deliver the
baseline and loss map. A single conditional protection experiment is allowed
under the gate below; it is not mandatory when the evidence cannot support
one. Submit one combined receipt, including failures and limitations. Do not
insert a review after each routine repair or table.

## 1. One explicit baseline, with existing mechanics

Call the baseline **DEFAULT_POST_KEEP_v1**. Reuse `pf_branch_engine.POST_KEEP`
on both sides of ONE `Branch`, through the normal `pre_tick` / `decide` /
`step` / `finish` path. Do not copy or rewrite the engine.

| Client state | Baseline action |
|---|---|
| Absent and executable under the engine constraints | POST at that side's received best |
| Live | KEEP the same generation and queue position |
| Partially filled | KEEP the remaining quantity; do not replenish by cancelling it |
| Post/cancel pending, or an unacknowledged completion | Process notices; no duplicate order |
| Completion/refusal acknowledged and side eligible again | POST at the next normal decision tick |
| Invalid book, position/reservation constraint, or too near END | Existing constraint/fallback, explicitly counted |

Preserve size 30, cap 30, 50 ms decision grid, 58 ms activation, 37 ms cancel
effect, 100 ms fill notice/cancel acknowledgement, M50 trade timing, the queue
model, post-only refusals, original structural rule and original END=5,000 ms.
Use the actual pinned SETTINGS rather than separate literals in the runner.
There is no learned admission threshold, discretionary pull, cooldown, fixed
minimum lifetime, top-up, layer expansion or price chase in this baseline.
Its expected discretionary-pull count is exactly zero; structural pulls can
still occur. Track those separately.

An existing generation's priority must survive subsequent book snapshots.
New generations really join behind existing queue. Re-entry waits for the
existing acknowledgement/reservation rules, not an invented retry timer.
Size=cap means even a partial inventory imbalance can block a same-side new
30-share order; show cap-blocked time instead of calling it a model refusal.

Run frozen **PI0_50_v1** on a separate branch from the same initial world as a
reference. **NO_QUOTE** is the zero baseline. Do not call NO_QUOTE/N0 a posting
baseline. No need to fit/re-run RICH, SMOOTH, shrinkage or alternative sizes.

Earlier KEEP_INITIAL/KEEP_SELECTED experiments are not this experiment: they
overrode one generation while PI0 controlled the rest. Likewise `hold_from`
does not implement a replenishing two-sided policy. Use it only where its
local-label semantics are explicitly appropriate.

## 2. Population and execution sequence

Reuse the exact **384 world keys** in
`predictfun/reviews/entry_exposure_20261121/manifest.json`: four hash-ranked
worlds per UTC hour on consumed Oct 24–27. Deduplicate the two focal-side
forks into one world; simulate both sides jointly. Preserve every world key
and every missing-path/mark status. Do not require a successful historical
mandatory POST label to admit a world that the independent world loader can
evaluate; record any difference from the old fork population explicitly.

Discovery/training: Oct 24–25. Subsequent development measurement: Oct 26–27.
All four days are already consumed. This separation controls this round's
adaptation; it is not untouched validation. No confidence interval or claim of
four independent full-day P&Ls: these are sampled five-second episodes.

1. Freeze the baseline runner, world/input manifest, feature dictionary,
   labels, checks, sampling and reading rules before running any world.
2. Pilot the first eight canonical worlds of each discovery day (16 total),
   included in the full sample. Verify accounting and measured resource cost.
3. Run discovery worlds once per required policy, saving reusable traces and
   state features. Produce the loss map and candidate-gate decision.
4. If the conditional test is justified, commit its exact specification,
   fitting/selection code and fitted artifact before new Oct 26–27 output is
   inspected. Otherwise commit the no-candidate decision and its reason.
5. Run the later-day baseline/reference, and the one candidate if declared;
   summarize and check together. Do not tune again on Oct 26–27.

The existing day guards, era checks and input hashes remain binding. Do not
open Oct 28–30 or Nov 4 onward. Do not extend the stopped six-date build.

## 3. Save the evidence once; distinguish accounting from prediction

Keep a complete per-world ledger and generation lifecycle: post request,
activation/refusal, fills, client notices, pull request/reason, removal,
acknowledgement, remaining shares and queue, terminal inventory and orders.
Keep all eligible posts, including refusals and no-fill generations. Retain
full causal feature vectors at entry and live decision checkpoints, plus
client inventory/reservations and both sides' states. Hash identity and
feature order. No extra tape rereads solely to add a missing report column.

Reuse `pf_pipeline_export.real_features`, `known_steps` and
`pf_r2_shadow.rebuild_row` on the ACTUAL baseline/guard branch. Reuse the
existing 107 inputs where available; do not use PI0 support/prediction as an
eligibility gate for the default-post policy. Feature unavailability is a
diagnostic status, not permission to silently remove the order or sample.

The feature dictionary must state exact column, units, source and receipt
clock. Cover directional PF book imbalance/change, depth depletion/refill,
recent received trade flow, spread/edge, volatility and signed price changes,
Binance price/depth/flow, client-known queue/remaining quantity, generation
age, remaining episode horizon, inventory and pending exposure. Separate
side-signed pressure from unsigned volatility. Use existing causal dynamics
first; do not expand the feature family or invent public trade timestamps.
BookTicker is a quote source, not trade flow. Venue queue, future fills,
future midpoint and realized fill delay are outcome explanations only.

Repeated 50 ms checkpoints are a trajectory, not independent action samples.
For an action table, use one entry and one hash-selected eligible live
checkpoint per generation, with separate populations. Fix the selection seed
and denominators before labels; do not select the checkpoint preceding a loss.
Use the first live review additionally for churn reporting, not as another
independent copy of the same target. Retain world and generation clustering.

## 4. First answer: where does the money go?

For each counted fill, in cents with side sign s and quantity q, reconcile:

`END value = s*q*(mid_at_fill - fill_price)
           + s*q*(END_mark - mid_at_fill)`.

Sum once over fills to the episode's cash + inventory * END_mark. The first
term is execution edge against the fill-time mid; the second is subsequent
marked movement. This is an accounting split, not proof that movement is
caused by adverse selection. Keep bid and ask contributions and the joint
inventory path. Do not sum overlapping local labels as strategy P&L.

Report fill-time, +50/+100/+1,000 ms markouts, and common-END value. Every
horizon needs its available/missing fill and share denominators, frame age
and venue/receipt timestamp convention. Do not look past the episode or use
a future frame as a fill-time mid. Where the mid is missing, keep END value
and mark the decomposition unavailable; do not silently change populations.
Use common-coverage subsets for cross-horizon comparisons.

Show gains, ordinary losses and severe losses separately. Keep the existing
severe threshold **value < -30 cents per order opportunity** for comparability;
it is not -30 cents/share. Also show episode tails separately, worst values,
top-five generation/world loss shares, and every day x side contribution.
Give per-opportunity, per-filled-share and per-world denominators. Retain
zero outcomes. Removing severe losses after observing them is an oracle
description, never an implementable return.

The inherited cost convention is explicit zero costs/gross. END is a
five-second mark, not settlement; no free terminal liquidation. No fee,
rebate or settlement profitability claim. Inventory, cap and episode-end
effects must be visible before attributing every loss to the predictor.

## 5. Second answer: what was knowable, and what could an action prevent?

On discovery days, compare all gains/losses/no-fills by the causal state at
entry and at the sampled live checkpoint. Fix marginal bins from discovery
inputs (including zeros/missing flags), then carry the identical boundaries
to Oct 26–27. Include counts of distinct generations/worlds and weighted
exposure in every cell. Show both signs and both sides, not only worst cases.
Avoid an unrestricted feature/threshold/conjunction search.

For a live checkpoint at d, define **remaining local exposure** using only
that generation's default-path fills at/after the modeled cancel effect
(normally d+37 ms), marked at the same original END. Fills before that effect
cannot be saved by this cancellation. A cancel does not undo inventory
already filled. Entry exposure starts at activation. Give all eligible zero
outcomes their zero label. Report positive-value mass sacrificed as well as
negative-value mass avoided for any action mask.

This local cancellation accounting is a no-replacement selected-order
contrast, not full-policy action advantage. Re-entry and changed other-side
inventory/queue effects are evaluated only by the guarded branch replay.
Use the exact generation/side/price and actual structural termination; do not
borrow a different order's outcome. At a sampled checkpoint whose order has
already finished at the venue but is unnotified, future avoidable value is
zero and the client state remains pending/live as observed.

For severe cases, report a timeline: observations received, candidate signal
(if any), earliest executable intervention, fill, notification, later move.
Separate: danger visible at entry; danger appearing while live early enough
for cancellation; loss already locked in before intervention; and no usable
decision-time signal. Without a declared signal, the earliest possible
intervention is only a timing bound, not evidence the loss was predictable.
Post-fill movement may justify avoiding a fill if it was predictable before
the fill; otherwise it may require inventory management. Do not infer that
choice from the accounting split alone.

## 6. One conditional protection test, only when the loss map supports it

The baseline/loss map is the mandatory next deliverable. Do not spend this
round fitting a generic large value model. After discovery, development may
declare **one small interpretable protection candidate** if all are true:

- A stated, decision-known mechanism separates costly from worthwhile
  exposure on both discovery dates, with the table's actual support shown.
- It is actionable at the existing latency. There are at least 30 affected
  distinct generations in at least 10 worlds per side on which protection is
  enabled, including at least 10 losing and 10 positive exposures. Thin sides
  retain the baseline; this support rule is fixed before discovery.
- The intervention names its predicted/estimated loss AND foregone ordinary
  gain. A severe-event probability alone is insufficient to set a threshold.

If any requirement fails, report **NO_SUPPORTED_GUARD** with the offending
denominator/mechanism and finish the baseline experiment. Do not add data,
fit variants or force an artificial success to fill the round.

If it passes, freeze the complete selector before fitting it: exact inputs,
transformations, target, units, support/fallback rules, threshold-selection
algorithm, loss/gain trade-off and finite search count. Limit the resulting
rule to at most two decision-state conditions per side (or one depth-two
tree per side), no boosting/hyperparameter sweep. Count every examined
selector/threshold in multiplicity, not merely the final rule. All selection
uses Oct 24–25; the later days cannot choose features, sides or thresholds.

**Policy contract:** default POST/KEEP remains the fallback. A supported
danger state can withhold an absent quote or pull its affected resting side;
it must not indiscriminately pull the other side. The entry and live tests
use the same longer-exposure policy and original endpoint, with the correct
fresh/live queue and latency. Re-evaluate on the normal grid. Once danger
clears and the state is executable, posting resumes without a fixed timer.
Persist a partial order unless its own risk warrants removal. A state change
caused only by switching from POST to KEEP inputs must not be called a new
market event. A cold-start unsupported state is reported, not hidden by a
PI0 gate or an unannounced conservative fallback.

Evaluate DEFAULT_POST_KEEP, PI0, NO_QUOTE and this candidate on the exact same
later-day worlds, each with its OWN state and realistic acknowledgements.
No fitting of separate 50 ms POST-vs-WAIT continuation labels. Queue loss,
re-entry, fill and inventory changes must come from replay, not a penalty
constant that makes churn disappear by construction.

Use 500 predeclared matched draws of the local action masks within
day x side x hour x action-kind, preserving action counts and sampling
weights. Report movable support and degeneracy. These cheap controls test
selection versus random withholding/pulling on the SAME opportunity table;
their local value is not a simulated strategy return. Do not run 500 full
policy simulations or claim the local null validates the whole policy.

## 7. What the combined receipt must make clear

| Question | Mandatory evidence |
|---|---|
| Does default posting capture spread? | Fill count/shares; fill-time edge; each markout with coverage; common-END value |
| Do a few bad outcomes erase ordinary gains? | Positive/ordinary/severe contributions; no-fill count; world/day concentration |
| Is quoting actually maintained? | Bid/ask/either/both active and at-best time over ALL episode time; cap/pending/invalid-book time |
| Is there churn? | Policy vs structural requests; effective removals; generation lifetimes; first-review pulls; re-entries and queue lost |
| Is the danger identifiable beforehand? | Discovery/later-day state tables; causal inputs; timing eligibility; local avoided loss vs sacrificed gain; matched draws |
| Does protection help the strategy? | Paired per-world and per-day gross value versus default; shares, uptime, inventory and tails alongside it |
| Did it merely stop trading? | NO_QUOTE comparison, positive retained exposure and participation; local matched-selection test |

Report numbers for each date and side plus the joint policy, with n/statuses
and as-of. No sum of hybrid local components masquerading as total P&L.

For a candidate, distinguish three outcomes: improves selection (local net
benefit exceeds the matched p95 with adequate movable support); improves
strategy value over default (paired replay); and produces positive gross
value. None implies the others. A positive research readout requires all
three on BOTH later days, with nonzero retained fills and no unreported
participation collapse. Failure is an informative result, not permission
to lower the criterion. Passing is still adaptive development, not promotion.

End with ONE supported conclusion: actionable entry protection; actionable
live-order protection; predominantly non-actionable-at-current-latency loss;
inventory/execution constraint; or no demonstrated economic discrimination.
If evidence is mixed, identify it instead of selecting a favorable side/day.
An additional data request must name the specific missing cell or event
count; "more data" alone is not the next priority.

## 8. Efficient implementation and combined acceptance

Reuse accepted engine, loader, feature, ledger and replay metric helpers.
Keep new adapters outside frozen teacher sources. Baseline eligibility must
not require loading/fitting the teacher; load it once per day only for PI0.
Cache day inputs in-process by their frozen hashes. Stream partitions and
reuse recorded outcomes/features for tables and local masks. No full label
build, source extension, extra model family or new simulator implementation.

Budget: **at most two aggregate core-hours for this round**, also no more
than the unused portion of the previously approved eight-hour allocation.
Development records spent/remaining CPU from the stopped work before launch.
Use existing development slots, maximum four single-core jobs; preserve the
Codex slot. Pilot ceiling 0.25 core-hours; project total before expanding.
If even the baseline exceeds the remaining cap, stop with the measured
projection instead of silently thinning worlds or expanding compute.
Keep 1,536 MiB checkpoint / 2,048 MiB hard RSS limits, source-bound resume
records, immutable launchers and failure logs. No active job was stopped by
this planning session; the USER supplied the stop instruction separately.

Do only result-critical checks: two-side/post/refusal/partial-fill/queue and
acknowledgement behavior; cap and structural reasons; zero discretionary
pulls for the default policy; causal hidden-notice invariance with a delivered
positive control; ledger and fill-time decomposition identities; cancellation
effect boundary including a too-late-fill falsifier; all statuses/membership;
aggregate recomputation and stale/corrupted artifact refusal. Reuse existing
controls; add a small set for changed adapters. No requirement to rebuild
large mutation suites for untouched components or stop for cosmetic issues.

The receipt directory should contain the declaration and manifest, checks,
source/input hashes, status/resource records, baseline/generation/feature
tables, loss-attribution and timing tables, and any candidate freeze/fit and
later-day comparisons. A thin or negative result is still a complete round.
Commit the freeze(s) before their outputs and the final combined receipt.
The development session then requests ONE review. This plan authorizes the
bounded work; no further USER confirmation is needed unless scope/resources
expand, dates change, or untouched validation/live trading is proposed.
