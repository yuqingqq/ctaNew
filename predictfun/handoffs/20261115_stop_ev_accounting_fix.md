# STOP-EV accounting contract and integration

This supersedes the aggregate label and approximate reconciliation in the
original STOP-EV design. The user authorized resolving the accounting issues.
The implementation is `research/pf_stop_ev_accounting.py`; its synthetic
instrument is `research/pf_stop_ev_accounting_check.py`. The test receipt in
`reviews/stop_ev_accounting_20261115/` records the tested source and as-of.
This is a label/accounting instrument, not a fitted model or a trading result.

## Target

For one client-resting order at decision time `t`, predict its own gross fill
value under the declared neutral continuation, up to `t + 1000 ms`, bounded by
the replay terminal time. Each actual fill tranche contributes:

```
quantity * side_sign * (mid_at(fill_time + 1000 ms) - order_price)
```

Prices and values are cents and cents times shares. Bid sign is +1, ask sign
is -1. Fees, rebates, subsequent inventory liquidation, replacement orders,
and changes to other orders after a focal cancellation are outside this local
label. It is not terminal P&L or the full portfolio KEEP-minus-PULL effect.

The cancel latency is the engine's 22 ms. Fills strictly before `t + 22 ms`
are common exposure and reduce remaining size. A fill exactly at cancellation
arrival is avoidable: the engine executes only while `fill_time < cancel_time`.
Fills at or before decision time have already occurred. There is no full-size
loss assigned just because price crosses the level. No remaining exposure is
assigned after a full fill. Every tranche uses its own markout, including when
the fill clock is shifted; marks do not stay at the unshifted anchor.
Marks come from actual valid raw bid/ask pairs, not `tf_mff` forward-filled
values. Freshness uses the last valid pair's timestamp; a later invalid frame
cannot make an old mid look fresh.

The one-second target assumes neutral continuation over that interval, subject
to the fixed STOP engine's cross/no-book/drift/end and declared inventory-cap
rules. It does **not** assume a learned cancellation at the next 50 ms decision.
A policy can reconsider every 50 ms; that is a separate closed-loop experiment.
The reference has no learned pull, posting gate or fixed retry cooldown.

## Where executions come from

`replay_neutral` calls `pf_stop_queue_activation_lane.sim_joint_g`. Queue
initialization uses only the activation snapshot. Later additions cannot
retroactively move ahead. All orders on both sides and all declared rungs are
present in the same replay. The existing engine allocates each public leg's
available quantity among those orders.

`Ledger` reconciles per-leg, per-order and flat fill records, activation/age
identity, price, through-price classification, order remainder and final
position. It refuses an old queue contract, mismatched clock/size/reference,
overused taker quantity, missing source identities and unreconciled source
counts. Epoch timestamps use an absolute 0.001 ms comparison tolerance, not
relative tolerance. Local tranche sums also reconcile to remaining quantity.

`at_price_value_cents` and `through_price_value_cents` classify the public leg
relative to our price. These are **not** transaction-level STOP/THROUGH labels.
Transaction classification can remain a descriptive component, but is not
needed to fabricate executions or dictate a fill's expected sign.

## Data adapter contract

The other session's developing `pf_stop_ev.py` can retain its market-flow and
hypothetical single-order tables as descriptive diagnostics. Primary order
labels must instead use this API on the complete neutral ladder. A CLEAN-only
subset does not validate SHARED or ENDED cases; this adapter handles both by
using the actual focal order's tranches, including its fixed-rule ending.

```python
from pf_stop_ev_accounting import (
    Coverage, LegInfo, MissingLeg, Ledger, Settings, replay_neutral, build_actions,
)

# W is the already quality-checked two-sided window from the source loader.
# Use fresh W per replay, because the engine has mutable book caches.
r = replay_neutral(W, size=30., shift_ms=50., queue='M',
                   unknown_queue='zero', nr=6, cap=None, mode='none')

# One provenance record per raw timed leg in W, keyed by (side, leg_index).
# IDs must identify raw source rows, unique across both sides, not taker IDs.
# WS timing remains an execution assumption; E2 uncertainty must be explicit.
provenance = {(side, i): LegInfo(source_row_id, 'ws')}
# E2 alternative: LegInfo(source_row_id, 'e2', uncertainty_ms=250.)

# These limits/counts come from the loader completeness receipt, not inferred
# from the last observed trade or from only the legs that filled an order.
coverage = Coverage(
    trade_start_ms=..., trade_end_ms=..., mark_end_ms=...,
    expected_legs=..., book_start_ms=..., book_end_ms=...,
    missing=(MissingLeg(source_id, earliest_ms, latest_ms, reason), ...),
    book_gaps=(MissingLeg(gap_id, earliest_ms, latest_ms, reason), ...),
)
l = Ledger(W, r, submitted_quantity=30., shift_ms=50.,
           leg_info=provenance, coverage=coverage)
rows, receipt = build_actions(l, day=day, asof=asof,
    ages_ms=(0., 250., 500., 1000., 2000.), settings=Settings())
```

Freeze the complete loader/feature/label pipeline, probe ages, clock, unknown
queue assumption, size, cap, loss and multiplicity before historical execution.
The API example is a wiring guide, not a claim that raw provenance is validated.
The supplied source counts must include excluded legs as statuses. Untimed
legs with no finite bound use `(-inf, +inf)` and therefore invalidate affected
actions rather than disappear. Out-of-range records can carry bounded intervals.
`reason='untimed'` invalidates the reference even if a caller supplies a finite
interval: empirical offset quantiles do not establish hard event-time bounds.
Duplicate source IDs fail; distinct IDs with duplicate raw contents still need
the upstream source-quality check. Preserve reserved-day pre-read filtering.

Strict timing refuses any E2 uncertainty or missing-leg interval overlapping
the relevant neutral-path prefix, not merely uncertain legs that filled the
focal order. Earlier events can change inventory, queue and order births.
`all_anchors_sensitivity` explicitly permits E2 point estimates, with qualified
statuses; it does not cure missing timing or establish exact executions.
Keep clocks and timing populations separate. WS-only omission can change the
reference path and is not an exact alternative for an E2-affected window.

Known book validity, trade/book coverage, mark coverage, freshness and declared
gaps can exclude an action. Invalid labels are `None` with a named status;
valid actions with no avoidable fill have value zero. A declared maximum mark
age (default 1000 ms) remains a modeling assumption to inspect on real data.

## Inputs and independent evaluation

`features` and `label` are separate objects. The order input allowlist is
`ORDER_FEATURES`. Queue estimates use only received (`kf`) frames; unknown
depth stays explicit, and later visible depth can bound it. Remaining shares
use submitted quantity minus acknowledged fills, under the engine's 100 ms
notice assumption. Neither true queue ahead nor true future remainder enters
the input dictionary. Join the existing market features by exact known-time
decision identity; audit that join before fitting. The pooled side-aligned
model with a side indicator remains the agreed model choice.

One deterministic, outcome-independent probe age is assigned per neutral order
generation. Unreached, already-filled, cancelled, rejected and invalid probes
stay in the receipt as statuses. There is no search for a surviving favorable
step and no q-grid duplication in the economic action population. Training on
more steps requires separately declared weighting; those steps cannot become
independent evaluation actions.

`action_metrics` checks unique actions and compatible reference/label settings,
then reports local kept value and local gain against keeping all. Its null
matches kept count within day, side, hour, rung, known remaining size and sample
weight, with at least 200 draws (default 500). Degenerate matching yields no
p-value. Carry the build receipt's full status counts alongside the metric.
No below-day interval is produced. Known edge/depth/queue/size baselines and
incremental model skill still belong in the model evaluation. Do not gate data
quality on top makers' positive mean or on sweep values being negative.

## Verification and remaining work

The synthetic suite checks chronological FIFO values, both sides, partial and
shared fills, cancellation arrival, 30/100-share sizing, client-known versus
true remainder, future-data isolation, late visibility, execution-clock marks,
missing/ambiguous timing, mark gaps, corrupted identities and ledgers,
independent probes and matched-control evaluation. Deliberately bad inputs
must be rejected; profitable through-price fills must be allowed.

The other session still needs to wire this adapter into its raw data build,
report coverage/status counts on consumed development dates, and freeze/train
the model on the corrected labels. The draft single-order tables are not a
substitute. Existing model artifacts and frozen running replays remain
historical and are not relabeled in place. Historical frequency of these
defects and the model's economic improvement are not measured by this fix.

Integration update: a developing `pf_stop_ev.py` was observed importing this
adapter and adding primary generation labels and E9 reconciliation. Its E9
must independently derive marks from **valid raw quotes**, using the same
execution-time target; the legacy forward-filled mid can be stale or, in a
synthetic fixture, inconsistent with raw bid/ask values. A fixture must supply
coherent raw quotes. An empirical 0.1–99.9% offset interval for an untimed leg
does not make its missing anchor safe. No historical data run was reviewed.
