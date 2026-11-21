# DEFAULT-POST-LOSS-MAP-v2 — duration, size and queue persistence

2026-11-21, USER-directed amendment to
[v1](../20261121_default_post_plan/plan.md), committed in
**e9e710fd606d96b8ce45f92a219bb4efcc9fac37**. Source inspected at that HEAD.
The USER notes that five seconds and 30 shares may miss the value of an
order that remains at the front of the queue through several partial fills.
This document supersedes v1's fixed duration/size, population and immediate
candidate-fitting instructions. Its accounting, causality, batching and
artifact requirements otherwise continue to apply. Development owns the
implementation; this planning session changes no engine or strategy code.

## Immediate next priority

**Measure the default-post policy over a continuous minute, with larger
resting quantities, before deciding what predictor or guard to build.**
Keep five seconds as a compatibility control and an observation checkpoint.
It is not the lifetime limit of the main experiment. Do not finish the old
384-world five-second-only package as a prerequisite, or restart the stopped
coverage build. Preserve any useful work already completed.

Longer allowed lifetime is not a minimum-hold timer. The baseline still
posts when executable, preserves each live order, and replenishes only after
completion is known. Structural cancellations, post-only refusals, position
limits and notices still operate. No new discretionary pull model, cooldown,
price chase, layer, or partial-order top-up is introduced.

## Fixed comparison, rather than a search for the best size

Run these declared scenarios on the same selected market/start keys:

| Scenario | Replay duration | New order size | Absolute inventory/reservation cap | Purpose |
|---|---:|---:|---:|---|
| LEGACY_5S_Q30_C30 | 5 s | 30 shares | 30 shares | Reproduce the old mechanics |
| REST_60S_Q30_C30 | 60 s | 30 | 30 | Longer exposure under the old capacity |
| REST_60S_Q30_C100 | 60 s | 30 | 100 | Isolate capacity relaxation at the same size |
| REST_60S_Q50_C100 | 60 s | 50 | 100 | Moderate larger order, same capacity |
| REST_60S_Q100_C100 | 60 s | 100 | 100 | Larger order, same capacity |

These are sensitivity scenarios, not five fitted strategies. Report every
scenario/day/side. Do not choose a favorable size or extend the grid after
seeing returns. Holding the cap at 100 in the size comparison avoids calling
an increase in permitted inventory a queue benefit. Cap/size ratios still
differ and legitimately change replenishment constraints; measure that effect
explicitly. The 30/cap30 versus 30/cap100 contrast separates capacity effects.

The 60-second branches carry orders, partial fills, queue state, inventory,
reservations and pending notices continuously. Record marks and lifecycle
state at **5, 30 and 60 seconds**, without resetting or cancelling at 5/30 s.
Report orders still open at 60 s as censored survivors. Sixty seconds is a
bounded first measurement, not proof about an entire market/session. If many
orders remain active or inventory unresolved, report that limitation and
propose a subsequent full-market run; do not silently extend this one.

Do not attribute differences between the 5-second control and a 60-second
branch solely to queue age: the end-of-episode admission constraint differs.
Legacy parity applies to the identical 5-second configuration. A long-run
prefix must agree only while neither branch has encountered a different
endpoint-dependent constraint. Log that first divergence explicitly.

## Bounded population and execution order

Use the existing ENTRY-EXPOSURE world manifest on consumed Oct 24–27.
From its four selected worlds per day/hour choose exactly ONE, by the minimum
SHA-256 of `default_post_duration_size_v2|` plus canonical JSON containing
day, market and start time (canonical key breaks ties). Commit the resulting
**at most 96 keys** before execution, along with inclusion probabilities and
the parent manifest hash. Do not rank by fills, return, queue, availability
of a favorable outcome, or previous experiment results.

All selected keys remain in the census. Preserve expiry, missing-path,
stale-book, missing-mark and other statuses; do not backfill rejected keys.
The existing path builder rejects ends past the market's safe expiry bound
(`ws + 298000 ms`). Keep that rule. Do not cross markets, stretch stale
quotes, manufacture 60 s of data, or read protected day/hour files to fill a
gap. Run the legacy control where valid; compare longer scenarios on their
common 60-second evaluable population and report the difference in coverage.

Discovery Oct 24–25; later descriptive checks Oct 26–27. All are consumed.
There is no model fitting or outcome-selected rule in this amended round.
Four sampled dates do not support an independent-validation or full-day
profitability claim. Carry shared-market/world identity through every table;
overlapping tapes and generations are not independent observations.

Pilot: the first two canonical selected keys per date, eight in total,
included in the manifest. Run all five scenarios on these keys to test the
entire size/duration comparison and measure resource use. Freeze once before
the pilot; accept the full bounded batch in one review, not after each arm.

## Minimal mechanics extension, with a real compatibility control

The current engine is intentionally fixed to the old contract:
`pf_branch_engine.Branch.__init__` checks END=5000; `Engine.add`, reservation
checks and `Branch.view` read fixed SIZE/CAP; `run`, policy horizons and fork
validation bind that contract. `pf_r2_labels.world_bind` also extracts only a
five-second tape. **Changing a report's END or passing a larger requested size
does not change those executed mechanics.**

Development should make the smallest separately versioned configuration
adapter/extension that executes size, cap and duration consistently, reusing
the accepted matching/queue/timing logic. Preserve the old frozen teacher and
engine source closure, or keep a pinned legacy import for compatibility;
do not silently invalidate old artifacts. Avoid a wholesale simulator rewrite.
Every new branch binds its actual size, cap, horizon, timing, tape and mark
in its identity, saved state and result. Refuse mismatches on reuse. No global
mutation of a running job's scenario or bypass of checks masquerading as
support for a new configuration.

`pf_queue_place_test.build_path(W,t,end)` already takes an endpoint; reuse its
timestamp and validity conventions with an explicitly extended path. Build
the required decision features from received raw data at their true times.
The old archived 5-second feature/fair vector must not be repeated, clipped
or stretched to cover a minute. Features with no valid longer reconstruction
remain explicit unavailable diagnostics; they do not gate the no-model
default-post baseline.

PI0 was trained/supported for its original size, cap and horizon. Do not run
it on 60 seconds or 50/100 shares and call that an accepted model comparison.
It may remain a separately labelled legacy 5s/Q30/C30 reference if already
available or inexpensive; it is not needed to answer this size/duration test.
NO_QUOTE remains the zero control for all scenarios.

## What would demonstrate a useful queue benefit?

Keep one larger order's gid, price and place through partial fills. Its
remaining quantity stays where that order was; it does not gain priority
merely because it is larger. Once fully filled, its replacement joins the
queue anew. Do not assume a same-price size increase preserves priority, or
merge a replenishment into the old order.

Show the mechanism, not only larger gross dollars:

- Submission-to-activation and activation-to-first-fill times; queue-ahead
  paths, with client estimates and venue diagnostics clearly separated.
- Time spent live, at the best, away from the best, and at estimated queue
  front; unchanged structural-cancel reasons and exposure lost to them.
- Distinct partial-fill events per generation, total filled fraction, time
  to full depletion, and replacements. Show survivor counts at 5/30/60 s.
- For Q50/Q100, the quantity and economics after the first 30 cumulative
  shares of a generation. Split a crossing fill at the exact 30-share point.
  This describes additional capacity; it is not by itself a causal measure
  of queue advantage. The full Q30 branch provides the actual comparison.
- Bid/ask/both-side quote time; no-quote time attributable to cap, pending
  acknowledgements, structural rules and invalid book. A larger cap can
  improve these independently of a larger order.
- Gross value and tail loss per world, per submitted share and per filled
  share, alongside total quantity, time-integrated/peak inventory and terminal
  inventory. Larger total P&L alone is not greater efficiency.

Use the v1 loss accounting and decision-time state map, now with real
5/30/60-second observation marks. A live quote's later fill and the inventory
created by an earlier fill are distinct sources of longer-horizon value.
Do not close inventory for free at a checkpoint. Keep markout coverage and
frame-age denominators; late fills lacking a future mark remain censored.

Retain the legacy absolute severe threshold (< -30 cents per generation)
as a labelled comparison, but also show loss per submitted share and a fixed
size-normalized severe threshold (< -1 cent per submitted share). No-fill
orders remain zero in both. Separate episode inventory tails from order
tails; equal absolute thresholds alone are not comparable across quantities.

For larger size, audit public-volume conservation and volume consumed across
our orders within each branch; do not allocate the same traded shares twice.
Separate branches are alternative worlds, not competing orders in one world.
Report order size relative to displayed depth and subsequent traded volume.
Historical replay leaves external participants' responses unchanged, so an
apparent large-size improvement is conditional on that assumption. If the
chosen sizes are large relative to the market, report that limitation before
using the result to infer deployable size or capacity.

## Budget, checks and stopping

Keep the v1 **two aggregate core-hour ceiling**, also bounded by the unused
previous allocation, and existing slots/RSS limits. Replace the larger
five-second population with this bounded matrix; do not add both full studies.
Pilot CPU ceiling remains 0.25 core-hours. Project all scenarios, feature
extraction and checks before continuing. Cache each input tape once per
market/day and reuse it across scenarios; stream outputs. No new label
training build or 500 full-policy simulations.

Result-critical controls, bundled with the implementation:

1. Exact legacy 5s/Q30/C30 behavior and ledger compatibility on synthetic and
   fixed pilot tapes; all executed settings match the recorded scenario.
2. A long synthetic tape whose order only fills AFTER 5 s, with repeated
   partial fills and later same-price arrivals. The remaining old order
   retains priority; full depletion/re-entry loses it correctly. Include a
   deliberately resetting/truncating implementation that the check rejects.
3. Size=30/50/100 and cap=30/100 reservations, partial fills, hidden notices,
   cancel effects and shared trade-volume limits; the same boundary controls
   as the accepted engine, without rebuilding unrelated test suites.
4. Continuous-run versus serialized checkpoint/resume identity, including
   queue/inventory/pending state; wrong-size/horizon/tape reuse refused.
5. All world/status membership, finite/stale marks, cash/inventory accounting,
   per-generation/episode totals and the published size/duration contrasts
   recompute from stored evidence. Retain every failed attempt.

One combined receipt must answer whether later execution is material,
whether larger orders retain useful queue position, whether additional fills
earn or lose value, and how much of any change comes from capacity/inventory.
If structural pulls still terminate nearly all orders early, report that
constraint; do not disable it quietly to manufacture longer resting.

**Defer v1's optional protection fit until this exposure design is understood.**
The next model's target/controller must match the selected resting and size
policy; do not reuse a five-second/30-share fitted target as evidence for a
minute/100-share strategy. The USER has authorized investigating longer and
larger resting in simulation; no additional confirmation is needed within
this bounded matrix. New dates, further size/horizon searches or larger
compute remain separate scope decisions. No new simulation is claimed here.
