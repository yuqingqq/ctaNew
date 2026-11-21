# Current plan — large resting orders with inventory management, full market

2026-11-21. USER clarification: test large orders WITH inventory management,
avoid a large test matrix, and do not restrict quoting to the first minute.
Prepared at **40d11d2185f611616193c390065822b82c524281**.

**This is the single current execution plan.** It supersedes the strategy
matrices, short horizons, comparison arms and optional model-fitting stages in
the earlier default-post plans and inventory addendum. Keep those documents
and any completed work as provenance. Development owns implementation; Codex
reviews one combined result. This document contains no new simulated result.

## 1. One strategy, one run design

Run **LARGE_REST_INVENTORY_v1**: quote both sides by default with a maximum
new order of **100 shares**, a hard per-market inventory/reservation cap of
**100 shares**, and the inventory mechanism below. Keep partial orders in
their existing queue positions. Replenish after completion is acknowledged.

There is ONE primary strategy. Do not run the old seven-arm matrix, size
sweeps, alternate caps, a parallel no-inventory strategy, PI0 comparisons or
a new prediction model in this round. Zero/no-trading is an arithmetic
reference, requiring no additional replay. A small legacy engine reproduction
check remains a correctness check, not a second economic experiment.

Quote continuously from the market's opening/first causally available valid
book through its supported pre-expiry end. These are approximately five-minute
markets. **Neither 5 seconds nor 60 seconds ends the strategy or resets it.**
The existing usable-tape boundary is market start +298 seconds (two seconds
before expiry); preserve that explicit boundary for this run and label it
accurately. Do not claim to model settlement or the last two seconds.

Carry orders, queue, inventory, reservations, pending notices and inventory
mode continuously across every minute. Record minute-by-minute readings and
a final aggregate. No restart, forced cancellation or free liquidation at a
measurement checkpoint. Inventory remains marked and disclosed at the final
supported endpoint. Different contracts have distinct positions; this is
per-market research, not a cross-market portfolio capital simulation.

## 2. Inventory management is part of the strategy

Use client-known position I and outstanding bid/ask reservations B/A,
including unacknowledged fills and cancel-pending quantities. Let Q=C=100,
soft trigger S=50 and recovery band R=25 shares. Target position is zero.
These are declared starting settings, not fitted or claimed optimal.

For an absent, otherwise executable side, desired quantities are:

`bid_desired = Q * clip(1 - I/S, 0, 1)`
`ask_desired = Q * clip(1 + I/S, 0, 1)`.

Clip them to the hard reservation room:

`bid_new = min(bid_desired, max(0, C - I - B))`
`ask_new = min(ask_desired, max(0, C + I - A))`.

Apply the established admissible quantity increment/minimum, or explicitly
retain the research convention if venue metadata is unavailable. Round down,
never above the budget. Report actual posted quantities: 100 is a maximum;
the controller intentionally posts less on the inventory-increasing side.

After processing notices, update a persistent restriction mode:
- I >= +50: LONG_RESTRICTED; I <= -50: SHORT_RESTRICTED. Crossing the opposite
  trigger reverses the mode directly.
- Otherwise abs(I) <=25 clears the mode; between 25 and 50 retain the previous
  mode. Start unrestricted at I=0.
- LONG_RESTRICTED blocks new bids and requests removal of a live bid;
  SHORT_RESTRICTED blocks new asks and requests removal of a live ask.
- Preserve the reducing-side order. Outside those restrictions preserve
  existing live/partial orders even when the desired NEW size has changed.

This is inventory hysteresis, not a fixed wait. Requests take the existing
cancel latency, and reservations release only on the real acknowledgement.
No duplicate order while pending, no instant undo of a fill, no every-tick
cancel/replace to resize, and no increase of an existing order's quantity
while claiming its old queue priority. Resume executable quoting when the
inventory state allows it. A reducing fill can overshoot zero; log reversals
and their costs rather than claiming guaranteed flattening.

Keep the existing M50 timing, 50 ms decision grid, activation/cancel/notice
delays, post-only rule and structural cancellation rule. Log inventory pulls
separately from structural pulls. No discretionary prediction guard, quote
price skew, taker hedge or end-of-run liquidation is added.

## 3. Market population and complete market paths

Stay on consumed Oct 24–27. Use the unique (day, market) IDs in the existing
ENTRY-EXPOSURE manifest as the candidate universe; do not run multiple
overlapping starting points for the same contract. Read market opening times
from the identity-bound registry, rather than treating the old order's
submission time as the market opening.

Group unique markets by day and their UTC opening hour. Select at most ONE
per cell by the minimum SHA-256 of `large_rest_inventory_full_market_v1|`
plus canonical JSON of day and market ID (canonical key breaks ties).
Commit those **at most 96 markets**, all candidate-cell denominators, registry
and input hashes before simulation. No selection by fills, returns, queue,
future coverage or prior policy outcomes; no backfill after failures.

Load the whole supported contract, with the existing causal preroll and
receipt clocks. At opening, wait until a valid book has actually been
received; do not backdate the first quote or initialize from a future frame.
Distinguish a genuinely not-yet-quotable opening from missing capture. Record
first usable receipt, first post and coverage by market minute. Preserve
explicit missing-market, truncated, stale-book, invalid-mark and expiry
statuses. If a whole path is not supportable, report that market as incomplete;
do not quietly substitute a favorable five-second/minute slice or bridge gaps.

No protected day/hour access or market-to-market tape stitching. The old
five-second archived fair/features must not be repeated or stretched: rebuild
diagnostic features from available raw inputs on the true clock. Feature
unavailability is a reported diagnostic status, not an unannounced admission
model. The no-model strategy itself uses the book and known inventory state.

All four days are consumed, sampled development data. Report no independent
validation, full-day return or confidence interval. Do not tune these settings
on one day and present the rest as untouched evaluation.

## 4. What to measure in this one experiment

Save complete generation/fill/notice/pull ledgers and lightweight decision
state once. Cache each market's input path and stream partitions. Keep no-fill
orders, refusals, open survivors and every exclusion status. Avoid expensive
feature reconstruction at every unchanged tick: use the existing builders at
entries and fixed/hash-selected live checkpoints, retaining causal times and
sampling denominators; never select the checkpoint just before a known loss.

The receipt must answer four questions:

1. **Do large resting orders get useful repeated fills?** Queue/active-at-best
   time, partial-fill counts, time to first/full fill, actual posted/filled
   shares, replacements and remaining orders, by minute as well as total.
2. **Does the strategy earn value?** Fill-time execution edge, subsequent
   movement, +50/+100/+1,000 ms markouts with valid denominators, and final
   cash + inventory x mark. Reconcile to each fill and count each once. Keep
   bid/ask and joint values; no free liquidation or overlapping-label sums.
3. **What does inventory management do?** Position path, integrated/peak
   absolute inventory, time above 50, recovery/overshoot, terminal position,
   cap/restriction/pending time, actual order sizes, inventory versus structural
   removals, first-review pulls and queue lost. Measure costs as well as lower
   exposure. With no ablation, do not claim a causal improvement over having
   no inventory management.
4. **Where and why are losses concentrated?** Gains/ordinary losses/severe
   losses and no-fills by market minute, side and decision-known conditions;
   order flow, book dynamics, volatility, queue and inventory. Separate loss
   accrued before a cancel could arrive, later fills, and movement of already
   acquired inventory. State association is a lead for the next experiment,
   not proof that the loss was predictable or avoidable.

Report absolute value and value per submitted/filled share, exposure and
world counts. Show both the legacy absolute severe-loss threshold and the
fixed size-normalized threshold from v2. Markouts cannot borrow post-expiry
prices; missing horizons remain explicit. Final value is gross, zero-cost,
pre-expiry marked value, not settlement or deployable net profit. Large-size
historical replay also assumes external flow does not react to our quote;
report size relative to depth/volume and enforce public-volume conservation.

## 5. Efficient implementation, necessary checks, one review

Use the minimal versioned size/cap/horizon extension already needed by v2;
reuse accepted matching, queue and causal timing logic. Bind executed Q/C,
endpoint, input tape, mark and inventory mode into state and result identity.
Do not spend time making frozen PI0 work outside its supported domain.

Run an eight-market pilot (first two canonical selected markets per date),
included in the same sample. Preserve the two aggregate core-hour ceiling,
also bounded by the unused previous allocation, existing slots and memory
limits. Pilot ceiling 0.25 core-hours. Project the one-strategy full-market
run before expansion; if it does not fit, report the measured resource need.
Do not truncate each market to one minute to make the budget fit. No extra
arms, data expansion or repeated feature/model searches.

Only necessary correctness checks: a bounded legacy reproduction; a long
partial-fill/queue case; actual-size/cap/reservation and hidden-notice checks;
inventory mode boundaries/recovery and resume identity; cancel timing;
public-volume conservation; complete membership and independent ledger/metric
recomputation with applied bad-input refusals. Reuse existing checks and fix
only defects that invalidate the result. No giant new test matrix and no
intermediate review for routine repairs.

Commit the complete executable declaration before the pilot and retain the
final outcome, failures and resource records in ONE combined receipt. Then
request ONE review. End that review with the largest evidenced limitation and
one targeted next experiment. No new model fit or guard search is required
to finish this round. Development may proceed without another confirmation
within this consolidated scope; earlier comparison plans are historical.
