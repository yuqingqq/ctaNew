# Current plan — resting ladder with inventory management, full market

2026-11-21. USER direction: large resting orders WITH inventory management,
continuous quoting across the market, one primary experiment, and multiple
price levels to acquire queue age before those levels become the best quote.
The USER subsequently rejected 50 shares per level as too small; the declared
starting size is therefore 100 per level, with the shared budgets below.
Prepared at **60f43033fc4f54a8d3be77a18ea4b1bafbd5a7d8**. This standalone plan
incorporates and supersedes the earlier single-level full-market plan.

**This is the single current execution plan.** It supersedes the strategy
matrices, short horizons, comparison arms and optional model-fitting stages in
the earlier default-post plans and inventory addendum. Keep those documents
and any completed work as provenance. Development owns implementation; Codex
reviews one combined result. This document contains no new simulated result.

## 1. One strategy, one run design

Run **LADDER_REST_INVENTORY_v1**, one fixed two-level ladder per side:

| Side | New order prices, from the received book | Nominal quantity per level |
|---|---|---:|
| Bid | best bid; best bid minus one tick | 100 shares each |
| Ask | best ask; best ask plus one tick | 100 shares each |

The nominal total is **200 shares per side**, with hard per-market
position/reservation cap **C=200**. This explicitly doubles the previous
single-level 100-share budget to accommodate two 100-share orders. The cap
applies across the whole side, never separately to each level. Inventory
management can reduce new quantities; 100 is the nominal per-order maximum.
Spacing uses the market's verified tick convention. Invalid/boundary prices
are explicit unavailable slots; never clamp two levels to the same price.

An early deeper quote can acquire age and move toward the front of its
price's queue before that price becomes the best. Posting a ladder does not
put an order ahead of shares already there. It also exposes us to sweeps when
the market moves through several levels; measure that cost.

**Preserve orders, not a constantly re-centered drawing of a ladder.** Keep
each live order's gid, price, remaining quantity and queue. Do not move both
orders whenever the best changes. A book snapshot cannot overwrite an old
queue position with newly arriving shares. Partial fills leave the remainder
in place; never top it up while claiming unchanged priority.

At most two unclosed generations per side, including pending orders. On an
acknowledged vacancy, consider the current best then one tick behind; create
only a missing valid price with available capacity. If both slots are occupied
by older prices, KEEP them unless their own structural or inventory rule
requires removal. Do not evict a queue merely to restore at-best coverage;
report any resulting at-best gap. Reserve a pending price until its close is
known, so no duplicate generation at the same side/price can be submitted.
Do not submit a quote that crosses any client-known unclosed opposite-side
order, including a cancellation still awaiting acknowledgement. Count that
as `own_cross_blocked`; do not simulate an unpriced self-trade or assume the
other order disappeared before its notice.

Freeze each new order's birth rung (0 or 1) for the existing structural
distance rule; do not re-label it from its current ranking to evade a pull.
Structural requests apply to the affected gid; no whole-ladder flush for one
bad level. Inventory restrictions can affect multiple same-direction orders
when all add to the restricted exposure, but leave the reducing side alone.
Replacements are new queue entries after their true acknowledgements.

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
SUMMED OVER ALL unclosed generations on that side, including post-pending,
partial fills not yet notified and cancel-pending quantities. Let Qside=C=200,
nominal per-level quantity q0=100, soft trigger S=100 and recovery band R=50
shares. Target position is zero. These are declared settings, not fitted.

Side size multipliers and total desired exposures are:

`f_bid = clip(1 - I/S, 0, 1); D_bid = Qside * f_bid`
`f_ask = clip(1 + I/S, 0, 1); D_ask = Qside * f_ask`.

For EACH eligible vacant level, use:

`q_bid_new = min(q0 * f_bid, max(0, D_bid - B), max(0, C - I - B))`
`q_ask_new = min(q0 * f_ask, max(0, D_ask - A), max(0, C + I - A))`.

Recompute reservations after every command before considering another slot.
At zero position with no orders this permits 100+100 per side. At I=+40 with
no bids it permits two 60-share bids, not a fresh 120-share budget for each
level. If existing reservations exceed a lower desired exposure, no new
same-side order is added; do not resize existing queues merely to match D.
Hard inequalities I+B<=C and -I+A<=C always bind, even while cancellation is
pending. Bid and ask commitments are not netted against each other.

Apply the established admissible quantity increment/minimum, or explicitly
retain the research convention if venue metadata is unavailable. Round down,
never above the budget. Report actual posted quantities: 100 is the per-order
maximum, 200 the nominal side budget; inventory can reduce either.

After processing notices, update a persistent restriction mode:
- I >= +100: LONG_RESTRICTED; I <= -100: SHORT_RESTRICTED. Crossing the opposite
  trigger reverses the mode directly.
- Otherwise abs(I) <=50 clears the mode; between 50 and 100 retain the previous
  mode. Start unrestricted at I=0.
- LONG_RESTRICTED blocks new bids and requests removal per live bid gid;
  SHORT_RESTRICTED blocks new asks and requests removal per live ask gid.
- Preserve the reducing-side orders. Outside those restrictions preserve
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

**Cancellation rationale, following the USER's question:** the hard cap
itself does not require cancelling exposure that is already reserved within
the cap. Soft-trigger pulls are a declared inventory-policy choice; the
distance/no-book/crossing structural requests are inherited simulation-policy
rules, not all exchange requirements. No model-score cancellation is present.
A best-price change alone does not move or cancel an order. The inherited
distance threshold can nevertheless terminate old/deeper quotes after a larger
move: report that cost and its reason explicitly, rather than interpreting it
as demonstrated adverse-selection protection or necessary queue management.

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
   Break these out by birth rung and price. Count aged deeper orders that
   later reach the best, their queue ahead and subsequent fill value, plus
   multi-level sweep losses. Treat these as mechanism descriptions; with one
   policy there is no demonstrated causal advantage over a single level.
2. **Does the strategy earn value?** Fill-time execution edge, subsequent
   movement, +50/+100/+1,000 ms markouts with valid denominators, and final
   cash + inventory x mark. Reconcile to each fill and count each once. Keep
   bid/ask and joint values; no free liquidation or overlapping-label sums.
3. **What does inventory management do?** Position path, integrated/peak
   absolute inventory, time above 100, recovery/overshoot, terminal position,
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

The existing engine holds multiple orders and sorts matching by price/time,
but the client Branch stores only one `open[side]` gid and its reservations,
status and action guards assume one generation per side. Development must
extend the client explicitly to per-order states; changing only the number
of submitted orders is insufficient. Keep matching/timing logic, actual-size
accounting, side-wide inventory and per-gid structural/notice semantics.

Use a deterministic command order: process all due notices and structural
requests, update inventory mode, then the existing alternating side order;
within a side process live gid requests and then vacant target prices from
best outward. Refresh reservations and invariants after every command. Bind
all active/pending gids, prices, quantities, birth rungs, queue state and
inventory mode in fingerprints and resume records. Do not use the old
single-open-gid policy wrappers or silently overwrite an earlier open order.
Do not spend time making frozen PI0 work outside its supported domain.

Run an eight-market pilot (first two canonical selected markets per date),
included in the same sample. Preserve the two aggregate core-hour ceiling,
also bounded by the unused previous allocation, existing slots and memory
limits. Pilot ceiling 0.25 core-hours. Project the one-strategy full-market
run before expansion; if it does not fit, report the measured resource need.
Do not truncate each market to one minute to make the budget fit. No extra
arms, data expansion or repeated feature/model searches.

Only necessary correctness checks: a bounded legacy single-level reproduction;
a long partial-fill/queue case; actual-size/cap/reservation and hidden-notice
checks; inventory mode/recovery and resume identity; cancel timing; complete
membership and independent ledger/metric recomputation. Add focused ladder
fixtures: two levels hit by one sweep share the same public volume; price/time
priority is respected; the cap includes BOTH levels and pending removals; a
partial fill or notice of one gid cannot close/alter another; later snapshots
preserve earlier queue age; and one structural pull leaves unaffected orders
resting. Include duplicate-price and crossing-own-pending-order refusals.
Use applied bad-input refusals that catch these failures. Reuse existing checks and fix
only defects that invalidate the result. No giant new test matrix and no
intermediate review for routine repairs.

Commit the complete executable declaration before the pilot and retain the
final outcome, failures and resource records in ONE combined receipt. Then
request ONE review. End that review with the largest evidenced limitation and
one targeted next experiment. No new model fit or guard search is required
to finish this round. Development may proceed without another confirmation
within this consolidated scope; earlier comparison plans are historical.
