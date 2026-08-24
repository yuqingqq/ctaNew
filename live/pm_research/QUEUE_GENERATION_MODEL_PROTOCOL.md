# Iteration 007 — generation-deduplicated action model protocol

**Status: FROZEN BEFORE FIT OR REPLAY at 2026-08-24T12:17:00Z. Research only.**

## Hypothesis and one change

Iteration 005 trained on every eligible 10 ms decision. Overlapping rows from
the same resting order generation repeatedly carry closely related future-fill
labels, although a cancellation edge can consume that generation only once.
Fit and evaluate the action-value classifier on the first point-in-time
eligible observation of each `(slug, maker_side, order_generation)`.

The row is selected solely by timestamp and generation identity, before
reading its label. A generation whose first eligible row has no preventable
economic fill remains a noneconomic row; no later row may replace it.

## Everything held fixed

- candidate: `QR_CANCEL_QGEN_X_SKEW`;
- behavior trace, 69 features, same-generation label and markout: iteration 005;
- BTC H50/L25, ETH H250/L100, q>0.5, pinned LightGBM parameters;
- three training days, two visible development days, one window/coin/day;
- exact-event 10 ms inference and false signal in reference-ineligible states;
- queue placement, quote size, five-share skew band, hold/release lifecycle;
- no rebates/incentives and assumed, unmeasured cancel-effective latency;
- isolated one-arm replay from iteration 006 for candidate, incumbent and
  comparator; no multi-arm comparison is permitted.

The model is fit only on first-generation rows with a latency-preventable,
nonzero gross value. It scores every eligible exact-event row for the policy.

## Frozen gates

The iteration-005 model gates apply on first-generation development rows:
positive weighted Brier skill, positive selected gross value each day,
positive gain versus the train-selected always/never constant each day,
positive gain versus old v5 on the same rows each day, and q>0.5 fraction
strictly between 2% and 98%.

Stateful adoption additionally requires isolated candidate PnL above
`QR_CANCEL_HOLD_X_SKEW` on both development days, mean PnL above isolated
`QR_SKEW_ONLY`, no terminal-inventory increase, no effective-cancel or
cancel/repost-traffic increase, and every trace/model/replay control passing.

All results remain development-only and require new independent forward days
if they survive. No threshold, row-selection, H/L, feature, or model sweep is
allowed after execution.
