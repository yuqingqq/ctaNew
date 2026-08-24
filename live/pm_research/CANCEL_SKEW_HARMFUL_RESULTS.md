# Harmful-flow cancel × skew — offline development result

**Status: `DEVELOPMENT_DIAGNOSTIC / NON_PROMOTABLE_MODEL_GATE_FAILED`.**
Decision eligible: no. This is a stateful shadow replay with no live venue,
order, cancellation or execution port.

Protocol: `CANCEL_SKEW_HARMFUL_PROTOCOL.md`. Implementation:
`policy_optimizer_cancel_skew.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_cancel_skew_harmful_v1.json`. Artifact
ID: `b7a2f9c1e87e4ff4957a2f69937681f80df23ad4ae0f936ce17194a17c2117e8`.

## Experiment

The replay composes the two fixed v5 diagnostic cells with the stateful
pessimistic skew policy:

- BTC: H=50 ms, assumed cancel-effective L=25 ms;
- ETH: H=250 ms, assumed cancel-effective L=100 ms.

Five 5-share arms run on the same 10 exact-event windows: JOIN, symmetric
FRONT, `SKEW_LB`, cancel-only JOIN, and cancel×`SKEW_LB`. Cancellation uses the
fixed value-weighted `q > 0.5` rule. It acts only on an order actually placed as
JOIN; a fronted inventory-reducing skew order is outside the v5 action schema
and remains untouched.

The lifecycle is stateful. Fills before effective time remain, partial fills
are retained, pending cancels are bound to an order generation, and a matching
cancel immediately reposts a full 5-share order behind current displayed depth.
The signal must clear before another cancel can be submitted.

Disabled-cancel parity with the existing replay engine, all-false signal
parity, deterministic rerun and JOIN-only action-schema controls all pass.

## Development-day result

| coin | day | skew only, c/window | cancel×skew | delta |
|---|---|---:|---:|---:|
| BTC | 2026-08-23 | +47.92 | -21.24 | -69.16 |
| BTC | 2026-08-24 | -36.88 | +88.53 | +125.41 |
| ETH | 2026-08-23 | +173.00 | +345.63 | +172.63 |
| ETH | 2026-08-24 | -37.17 | -42.02 | -4.85 |

Cancellation therefore improves skew on only one of two development days for
each coin, with opposite reversal days. It does not establish a stable
cancel×skew edge.

Skew does improve cancel-only on both development days for both coins:

| coin | cancel×skew minus cancel-only, Aug 23 | Aug 24 |
|---|---:|---:|
| BTC | +53.17 c/window | +26.58 |
| ETH | +226.16 c/window | +30.89 |

This identifies skew as the reliable component of the composition. It does not
repair the harmful-flow predictor's day instability.

Across all five visible days, mean PnL/window is +307.56 c cancel×skew versus
+286.65 c skew-only for BTC, and +173.82 versus +184.28 for ETH. These one-window
per coin-day means mix in-sample and already-seen development data and have no
promotion meaning.

## Fill, queue and risk effects

Cancel×skew submitted/effected 3,546/3,342 cancels over five BTC windows and
4,158/2,753 over five ETH windows. Combined cancel-plus-repost traffic is about
4.6 actions/second BTC and 4.6 actions/second ETH before ordinary quote
repositions. There were 39 BTC and 9 ETH partial-fill cancellations; stale
generation counts were 204 and 1,405, respectively. ETH's larger stale share
is consistent with its longer assumed 100 ms effective latency.

Relative to skew-only, BTC retained 68–88% of filled shares and 69–88% of
spread capture by day. ETH retained 86–103% of shares and 91–105% of spread;
values above 100% can occur because cancellation changes the inventory path,
subsequent skew placement, and replenishes a full quote after a partial fill.

Inventory risk is not uniformly improved. On the two development days,
cancel×skew lowers cash-at-risk versus skew for both BTC days and ETH Aug 23,
but raises it for ETH Aug 24. This remains a path-dependent stateful result.

## Conclusion

Cancel×skew is now implemented and testable in offline replay. The composition
shows that skew adds value to a cancellation policy, but cancellation does not
add stable value to skew. High cancel/repost churn, unmeasured cancel-effective
latency, missing queue/rejoin costs, one window per coin-day, and the failed v5
model gate keep the result non-decision-eligible. No live cancellation layer or
real-latency measurement is authorized by this result.

A follow-on out-of-schema stress test also applied this JOIN-trained signal to
fronted skew orders. It lost to JOIN-only cancel×skew on all four development
coin-days; see `CANCEL_SKEW_ALL_RESULTS.md`. Front-order cancellation should
remain disabled until it has an action-conditioned model.

## Verification

- 144 adverse/skew/cancellation module checks pass, including 13 new lifecycle
  and composition checks.
- 14 contract self-tests pass and the canonical contract diff is empty.
- Disabled-path parity, all-false parity, deterministic replay, compilation,
  artifact self-hash and all provenance hashes pass.
