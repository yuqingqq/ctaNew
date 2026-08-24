# Harmful-flow cancel-and-hold × skew — offline diagnostic result

**Status: `DEVELOPMENT_DIAGNOSTIC / NON_PROMOTABLE_MODEL_GATE_FAILED`.**
Decision eligible: no. This is an offline stateful replay with no live venue,
order, cancel, or execution port.

Protocol: `CANCEL_HOLD_SKEW_PROTOCOL.md`. Implementation:
`policy_optimizer_cancel_hold_skew.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_cancel_hold_skew_v1.json`. Artifact ID:
`ff424e200cc2067820bf7421ffd67c662d806a171e8203fa21a0072a1b09e04f`.

**Queue-semantics correction:** this receipt remains the exact historical
result, but its reducing-side `FRONT` means zero queue ahead at an already
occupied touch. It is therefore an optimistic upper-bound experiment, not the
current executable-placement baseline. The corrected `JOIN_EXISTING` /
one-tick price-improvement replay and conclusions are in
`QUEUE_REALISTIC_CANCEL_SKEW_RESULTS.md`; its baseline is
`QR_CANCEL_HOLD_X_SKEW`.

**Authorizing ruling (R-126 in-file rule, applied by the R-127-ordered
discipline pass):** the coordinator's standing KEEP-vs-CANCEL harness
instruction (R-125 item 4), retroactively confirmed as the authorization
for this work by R-127.

**Population (n and as-of, from the receipt):** **n = 10 windows** — ONE
window per coin per day (`*-1787247900` era slugs), btc+eth ×
2026-08-20/21/22/23/24; **08-20/21/22 are the v5 signal's TRAINING days;
08-23/24 are DEVELOPMENT days (already-seen; inspected repeatedly)**.
Receipt created 2026-08-24T10:10:33Z. There are NO holdout days anywhere
in this file.

**GATE-STATE BANNER — reads with EVERY number below (the R-127-ordered
per-number discipline): every figure in every table is downstream of the
v5 harmful-flow signal whose MODEL GATE FAILED, on seen days, at n = 1
window/coin/day, with latency ASSUMED (L=100 ms). Nothing here is
holdout evidence; nothing qualifies Stage A's 120/120 (R-127's own
guard); positives are diagnostic shape, not performance.**

## Development result

True liquidity withdrawal changes the result materially, but not uniformly.
Against the existing immediate-repost cancel×skew arm:

*(v5 gate: FAILED · seen days · n=1 win/coin/day)*
| coin | day | immediate repost, c/window | cancel-and-hold | delta |
|---|---|---:|---:|---:|
| BTC | 2026-08-23 | -21.24 | -7.90 | **+13.34** |
| BTC | 2026-08-24 | +88.53 | +151.82 | **+63.29** |
| ETH | 2026-08-23 | +345.63 | +185.30 | **-160.33** |
| ETH | 2026-08-24 | -42.02 | +43.09 | **+85.11** |

BTC therefore supports the lifecycle change versus immediate repost on both
development days. ETH reverses by day because the Aug 23 immediate-repost path
captured unusually favorable drift that holding out removed.

The economically stricter comparison is against skew with no cancellation:

*(v5 gate: FAILED · seen days · n=1 win/coin/day)*
| coin | day | skew only, c/window | cancel-and-hold | delta |
|---|---|---:|---:|---:|
| BTC | 2026-08-23 | +47.92 | -7.90 | **-55.83** |
| BTC | 2026-08-24 | -36.88 | +151.82 | **+188.71** |
| ETH | 2026-08-23 | +173.00 | +185.30 | **+12.30** |
| ETH | 2026-08-24 | -37.17 | +43.09 | **+80.26** |

ETH cancel-and-hold beats skew on both development days. BTC still reverses,
so no common BTC/ETH cancellation policy passes a day-robust comparison.

Across all five visible days, mean PnL/window is:

*(v5 gate: FAILED · seen days · n=1 win/coin/day)*
| coin | skew only | immediate cancel×skew | cancel-and-hold |
|---|---:|---:|---:|
| BTC | +286.65 c | +307.56 c | **+130.36 c** |
| ETH | +184.28 c | +173.82 c | **+257.79 c** |

ETH cancel-and-hold is positive on all five visible days and beats each
comparator on four of five. BTC loses heavily on two model-training days,
especially Aug 20, and has a much lower all-day mean. These all-day figures mix
training and already-seen development data and have no promotion meaning.

## What the hold changed

*(v5 gate: FAILED · seen days · n=1 win/coin/day)*
| coin | day | held side-time | filled-share retention vs skew | terminal abs net: skew → hold |
|---|---|---:|---:|---:|
| BTC | 2026-08-23 | 69.0% | 37.3% | 5.59 → 4.57 |
| BTC | 2026-08-24 | 37.3% | 44.2% | 5.22 → 4.25 |
| ETH | 2026-08-23 | 42.2% | 80.5% | 3.40 → 1.21 |
| ETH | 2026-08-24 | 42.9% | 75.8% | 11.41 → 1.16 |

The side-time denominator is two quote sides × 300 seconds. BTC's signal holds
out too much liquidity and retains less than half of skew's filled shares on
both development days. ETH retains 76–81% while materially lowering terminal
inventory, which explains why ETH is the only credible forward diagnostic.

For BTC, the lifecycle behaves as intended relative to immediate repost: on Aug
23 it gives up 132.61 c of spread but improves drift by 145.94 c; on Aug 24 it
gives up 120.76 c of spread and improves drift by 184.06 c. For ETH Aug 24 it
gives up 24.41 c and improves drift by 109.52 c. ETH Aug 23 is the reversal: it
loses 48.41 c of spread and 111.92 c of favorable drift.

## Lifecycle and traffic

Across five windows per coin:

*(v5 gate: FAILED · seen days · n=1 win/coin/day)*
| coin | submitted | effective | hold entries | held side-time | cancel+repost actions/s |
|---|---:|---:|---:|---:|---:|
| BTC | 3,443 | 3,250 | 2,783 | 1,515.04 s | 4.46 |
| ETH | 3,803 | 2,498 | 1,598 | 970.36 s | 4.20 |

BTC has seven holds and ETH five holds still open at window end; all other hold
entries release. Almost all releases are signal clears. Inventory-role changes
release 77 BTC and 11 ETH holds across the five windows. Traffic is modestly
lower than immediate repost, but it remains high and excludes ordinary quote
repositioning, ACK behavior, and queue/rejoin cost.

## Conclusion

Immediate reposting was part of the cancellation problem: true withdrawal
improves the selected lifecycle comparison for BTC and produces a materially
better risk/PnL tradeoff for ETH. It does not rescue the common harmful-flow
predictor across both coins.

This historical queue-upper-bound result originally identified ETH H=250 ms /
assumed L=100 ms as the only shape worth freezing. The queue correction retains
ETH as the strongest diagnostic but replaces this arm with
`QR_CANCEL_HOLD_X_SKEW` and replaces its comparator with `QR_SKEW_ONLY`. BTC
still should not advance.

No live or decision-facing activation is supported. The v5 model failed its
original gate, latency is assumed, and the development days have been inspected
repeatedly.

## Verification

- 17 new lifecycle/schema checks pass.
- The new event loop reproduces all five prior arms exactly, including fills
  and diagnostics.
- All-false hold equals skew exactly; deterministic rerun and inventory
  reconciliation pass.
- Python compilation, artifact self-hash, provenance hashes, and diff checks
  pass.
