# Queue-realistic cancel-and-hold × skew — offline diagnostic protocol

**Status: frozen before execution. Research only.** This protocol corrects the
meaning of `FRONT` in the skew experiments. It adds no live venue, order,
cancel, or execution port.

## Question

Does cancel-and-hold × skew retain its result when the inventory-reducing quote
cannot receive guaranteed front-of-queue priority at an already occupied best
price?

## Queue semantics

The public tape supplies aggregate price-level size, not our individual queue
rank. The corrected replay therefore permits only two executable placements:

1. `JOIN_EXISTING`: quote at the current same-side touch behind all displayed
   quantity at that price; and
2. `PRICE_IMPROVE_1T`: when spread is at least two ticks, improve the reducing
   side by exactly one tick without crossing. Because the improved price lies
   inside the observed spread, its initial displayed queue ahead is zero.

A one-tick spread can never price-improve without crossing. Both sides must
JOIN. The replay never assigns zero queue ahead at the same occupied touch and
never treats a book update alone as proof that our order won a new-level race.

The counterfactual order is not inserted into the recorded public book. A trade
whose limit reaches a price-improved quote consumes that quote first because it
is at a better price than the recorded touch. Replenishment at the still-empty
improved level has zero displayed queue ahead; JOIN replenishment uses current
displayed same-side size. Own impact and competing orders at the hypothetical
inside-spread price remain unavailable and are named limitations.

## Inventory and cancellation policy

The skew band remains five shares:

- near flat: both sides `JOIN_EXISTING`;
- long Up beyond the band: SELL_UP is reducing and may
  `PRICE_IMPROVE_1T` only when spread permits; BUY_UP joins; and
- short Up beyond the band: BUY_UP is reducing and may price-improve; SELL_UP
  joins.

The requested baseline is `QR_CANCEL_HOLD_X_SKEW`. It preserves the reducing
side and applies the frozen v5 `q > 0.5` harmful predicate only to an actual
joined, non-reducing order. A matching effective cancel holds that side out
until signal clear or until inventory makes the side reducing. Release applies
the current queue-realistic placement rule.

## Frozen population and arms

Data, split, model, horizon, assumed cancel latency, feature clock, quote size,
and signal threshold are unchanged:

| coin | fill horizon | assumed cancel-effective latency |
|---|---:|---:|
| BTC | 50 ms | 25 ms |
| ETH | 250 ms | 100 ms |

One BTC and one ETH exact-event window per day are replayed for 2026-08-20
through 2026-08-24. Days 20/21/22 trained v5; 23/24 are already-seen
development days. There is no independent forward holdout.

The first six arms reproduce the prior cancel-and-hold receipt exactly:

1. `JOIN_ONLY`;
2. `FRONT_ONLY` — retained only as the old same-price zero-queue upper bound;
3. `SKEW_ONLY` — old guaranteed-front skew comparator;
4. `CANCEL_ONLY`;
5. `CANCEL_X_SKEW` — old immediate-repost comparator; and
6. `CANCEL_HOLD_X_SKEW` — old guaranteed-front hold comparator.

Two corrected arms are added:

7. `QR_SKEW_ONLY`; and
8. `QR_CANCEL_HOLD_X_SKEW` — the new policy baseline.

No model, threshold, band, price distance, latency, or sample is selected after
seeing results. One tick is the minimal legal price improvement, not a tuned
offset.

## Primary comparisons and controls

Primary comparisons by coin/day:

- corrected baseline minus old `CANCEL_HOLD_X_SKEW` (queue correction cost);
- corrected baseline minus `QR_SKEW_ONLY` (incremental cancellation value); and
- corrected baseline minus old immediate `CANCEL_X_SKEW`.

Report PnL, spread, drift, shares, inventory/cash risk, cancel/hold traffic,
spread-in-ticks opportunity, JOIN/price-improve placements, and fills by actual
placement kind.

Required controls:

- all six prior arms reproduce the prior event loop exactly;
- all-false corrected cancel-hold equals `QR_SKEW_ONLY` exactly;
- one-tick spread forces reducing and increasing sides to JOIN behind displayed
  depth;
- a two-tick spread permits exactly one-tick, non-crossing price improvement on
  the reducing side only;
- no same-price order receives zero initial queue ahead;
- held quotes cannot fill or reappear before release;
- deterministic rerun, partial-fill/stale-generation safety, and inventory
  reconciliation pass.

The result is diagnostic regardless of PnL. The predictor failed its original
gate, latency is assumed, the development days are repeatedly visible, and the
inside-spread counterfactual lacks competitor response. Promotion requires an
unchanged candidate on new days followed by real queue/ACK measurement outside
this research repository.
