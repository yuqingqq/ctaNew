# Queue-realistic cancel-and-hold x skew — offline diagnostic result

**Status: `DEVELOPMENT_DIAGNOSTIC / NON_PROMOTABLE_QUEUE_CORRECTION`.**
Decision eligible: no. This is an offline replay with no live venue, order,
cancel, or execution port.

Protocol: `QUEUE_REALISTIC_CANCEL_SKEW_PROTOCOL.md`. Implementation:
`policy_optimizer_queue_realistic.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_queue_realistic_v1.json`. Artifact ID:
`42f56c10e3cc3cfb6b2846248189dde7917d61d759e66e1ff4330a3f01f965d3`.

**Population:** n=10 windows, one BTC and one ETH window per day on
2026-08-20 through 2026-08-24. August 20/21/22 trained the v5 harmful-flow
model; August 23/24 are repeatedly inspected development days. There is no
independent forward holdout. BTC uses H=50 ms and assumed cancel-effective
L=25 ms; ETH uses H=250 ms and assumed L=100 ms.

**Gate-state banner:** every result below is downstream of the v5 harmful-flow
signal whose original model gate failed. Latency is assumed, not measured, and
the inside-spread counterfactual cannot model competitor response or own
impact. Positive results describe diagnostic shape only.

## The queue fix

The old `FRONT` label meant zero queue ahead at the same occupied touch. That
is retained only as a historical upper bound. The corrected arms use:

- `JOIN_EXISTING`: quote at the touch behind all displayed size; and
- `PRICE_IMPROVE_1T`: only the inventory-reducing side, only when the spread is
  at least two ticks, one tick inside the spread, with zero initially displayed
  queue ahead.

At a one-tick spread both sides must join. The replay never grants zero queue
ahead at an occupied price and never assumes that a book update proves we won
a new-level race.

The requested baseline is `QR_CANCEL_HOLD_X_SKEW`, not skew-only and not the
old immediate-repost cancel arm. It cancels only a harmful, actually joined,
inventory-increasing order, holds that side out while the signal remains
harmful, and preserves or releases any side that becomes inventory reducing.

## Full performance comparison

Mean PnL is cents per five-minute window. `mean5` mixes three training and two
already-seen development days; `dev2` is the mean of August 23/24. Neither is a
forward estimate.

| arm | BTC mean5 | BTC dev2 | ETH mean5 | ETH dev2 |
|---|---:|---:|---:|---:|
| `JOIN_ONLY` | +257.37 | -52.87 | +99.16 | -74.93 |
| `FRONT_ONLY` (unexecutable upper bound) | +1,116.27 | +276.49 | +726.38 | +429.77 |
| old `SKEW_ONLY` (same-price front on reducing side) | +286.65 | +5.52 | +184.28 | +67.92 |
| `CANCEL_ONLY` | +143.64 | -6.23 | +98.62 | +23.28 |
| old immediate `CANCEL_X_SKEW` | +307.56 | +33.64 | +173.82 | +151.81 |
| old `CANCEL_HOLD_X_SKEW` | +130.36 | +71.96 | +257.79 | +114.19 |
| `QR_SKEW_ONLY` | +197.75 | -52.88 | +125.28 | -65.62 |
| **`QR_CANCEL_HOLD_X_SKEW` baseline** | **+185.05** | **-40.46** | **+196.24** | **+53.54** |

The enormous symmetric `FRONT_ONLY` result is accompanied by mean terminal
absolute inventory of 793.27 BTC shares and 127.38 ETH shares, versus 9.68 and
5.43 for the corrected cancel baseline. It is not an executable strategy
result.

Correcting old skew to queue-realistic skew lowers mean5 PnL by 88.90 c/window
for BTC and 59.00 c/window for ETH. On the two development days the reductions
are 58.40 c and 133.53 c. This confirms that guaranteed same-price front was a
material optimistic assumption.

## Incremental cancellation result

The primary comparison is corrected cancel-and-hold x skew against corrected
skew with no harmful-flow cancellation:

| coin | day | `QR_SKEW_ONLY` | corrected baseline | cancel delta |
|---|---|---:|---:|---:|
| BTC | 2026-08-20 | -353.06 | +359.16 | +712.22 |
| BTC | 2026-08-21 | +150.02 | +382.35 | +232.33 |
| BTC | 2026-08-22 | +1,297.55 | +264.64 | -1,032.91 |
| BTC | 2026-08-23 | -58.03 | -10.83 | +47.21 |
| BTC | 2026-08-24 | -47.72 | -70.09 | -22.37 |
| ETH | 2026-08-20 | +233.77 | +151.93 | -81.84 |
| ETH | 2026-08-21 | -184.89 | +274.50 | +459.38 |
| ETH | 2026-08-22 | +708.72 | +447.69 | -261.03 |
| ETH | 2026-08-23 | -45.53 | +159.67 | +205.20 |
| ETH | 2026-08-24 | -85.70 | -52.60 | +33.10 |

Across all five visible days, cancellation adds -12.71 c/window for BTC and
+70.96 c/window for ETH. On development days it adds +12.42 c/window for BTC
and +119.15 c/window for ETH. ETH is positive on both development comparisons;
BTC reverses and remains negative in absolute dev2 PnL. There is no common
coin-robust cancellation result.

On the two development days, cancellation reduces mean filled shares from
547.10 to 260.47 and mean terminal absolute inventory from 38.19 to 15.81 for
BTC. For ETH it reduces shares from 134.37 to 94.37 and terminal absolute
inventory from 3.51 to 1.25, while changing mean PnL from -65.62 c to +53.54 c.

## How often price improvement was actually available

Across five windows per coin for the corrected baseline:

| coin | mean eligible sync fraction | JOIN placements | improve placements | JOIN/improve fill events | effective cancels | mean held side-time |
|---|---:|---:|---:|---:|---:|---:|
| BTC | 1.55% | 5,741 | 901 | 570 / 47 | 2,739 | 40.94% |
| ETH | 6.86% | 11,207 | 1,979 | 149 / 93 | 2,337 | 31.45% |

Price-improved orders provide only 8.45% of corrected-baseline filled BTC
shares and 25.00% of ETH shares. Most placement and fill behavior therefore
comes from joining, not getting in front. The eligible-sync and placement
fractions have different denominators: a placement is counted only when an
order is actually created or repositioned.

## Conclusion

The queue correction is necessary and changes the interpretation:

- same-price `FRONT` is an upper bound, not an achievable arm;
- queue-realistic cancel-and-hold x skew is now the policy baseline;
- cancellation remains unstable for BTC; and
- ETH H=250 ms / assumed L=100 ms retains the strongest diagnostic shape,
  beating corrected skew on both visible development days and reducing
  inventory, but it is not promotable because the model gate failed and no
  independent days or measured latency exist.

If new data are collected, the only defensible next replay is the unchanged
queue-realistic baseline against unchanged `QR_SKEW_ONLY`. Real individual
queue position, order/cancel ACK timing, queue loss on repost, and competitor
response require measurement outside this research repository.

## Verification

- 16 queue-realistic lifecycle/schema checks pass.
- All six historical arms reproduce the prior hold receipt exactly, including
  fills and diagnostics.
- All-false cancellation equals `QR_SKEW_ONLY`; deterministic rerun and
  inventory reconciliation pass.
- 13 immediate-cancel, 12 cancel-all, and 17 hold lifecycle checks pass.
- All 14 contract self-tests, Python compilation, artifact self-hash,
  code/protocol provenance hashes, and diff checks pass.
