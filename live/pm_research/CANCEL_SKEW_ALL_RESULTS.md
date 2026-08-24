# Harmful-flow cancel-all × skew — offline diagnostic result

**Status: `DEVELOPMENT_DIAGNOSTIC / NON_PROMOTABLE_OUT_OF_ACTION_SCHEMA`.**
Decision eligible: no. This is an offline replay with no live venue, order,
cancel, or execution port.

Protocol: `CANCEL_SKEW_ALL_PROTOCOL.md`. Implementation:
`policy_optimizer_cancel_skew_all.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_cancel_skew_all_v2.json`. Artifact ID:
`f9eae837559ef35cfa38c3f1e3cbe59e5675e07969b0e900bb41ab97d8ba92d1`.

## Result

Allowing the JOIN-trained harmful-flow signal to cancel the fronted,
inventory-reducing skew order made the existing cancel×skew policy worse on
both development days for both coins.

| coin | day | JOIN-only cancel×skew, c/window | cancel-all×skew | delta |
|---|---|---:|---:|---:|
| BTC | 2026-08-23 | -21.24 | -76.47 | **-55.24** |
| BTC | 2026-08-24 | +88.53 | +70.22 | **-18.31** |
| ETH | 2026-08-23 | +345.63 | +290.63 | **-55.00** |
| ETH | 2026-08-24 | -42.02 | -69.52 | **-27.50** |

The mean development loss versus JOIN-only cancellation is -36.77 c/window
for BTC and -41.25 c/window for ETH. Across all five visible days, cancel-all
also loses: mean PnL falls from +307.56 c to +140.80 c/window for BTC and from
+173.82 c to +97.61 c/window for ETH. The all-day mean deltas are -166.76 c
and -76.21 c, respectively. These all-day figures mix model-training and
already-seen development days and are descriptive only.

## What changed

Development-day filled quantity changed little relative to JOIN-only
cancel×skew:

| coin | day | share retention | spread-capture delta | drift delta |
|---|---|---:|---:|---:|
| BTC | 2026-08-23 | 98.70% | -1.97 c | -53.27 c |
| BTC | 2026-08-24 | 98.84% | -0.07 c | -18.24 c |
| ETH | 2026-08-23 | 100.00% | 0.00 c | -55.00 c |
| ETH | 2026-08-24 | 95.85% | -2.50 c | -25.00 c |

The loss is therefore mostly worse post-fill drift rather than simple lost
spread or a large reduction in filled shares. This is consistent with action-
schema mismatch and the stateful path change: a signal trained to value
cancelling a joined quote is not reliable on the deliberately fronted
inventory-reducing quote, and cancelling that quote changes later inventory
and skew placement.

Inventory risk is not repaired. Relative to JOIN-only cancel×skew on the
development days, terminal absolute inventory is worse for BTC Aug 23, better
for BTC Aug 24, identical for ETH Aug 23, and worse for ETH Aug 24.

## Cancellation traffic

Across five windows per coin:

| coin | policy | submitted | effective | cancel+repost actions/s |
|---|---|---:|---:|---:|
| BTC | JOIN-only cancel×skew | 3,546 | 3,342 | 4.59 |
| BTC | cancel-all×skew | 3,720 | 3,497 | 4.81 |
| ETH | JOIN-only cancel×skew | 4,158 | 2,753 | 4.61 |
| ETH | cancel-all×skew | 4,719 | 3,083 | 5.20 |

Cancel-all raises submissions by 4.9% BTC and 13.5% ETH, and effective cancels
by 4.6% and 12.0%. These rates exclude ordinary quote repositioning and still
omit cancel/rejoin cost and live ACK behavior.

## Conclusion

Do not cancel every skewed order with the current predictor. The experiment
rejects the proposed generic rule on the visible development data: it is worse
than restricting cancellation to JOIN orders in 4/4 development comparisons,
adds churn, and does not consistently improve inventory risk.

This does not prove that front-order cancellation can never work. It shows that
the current JOIN-trained score should not control it. A valid successor needs
an action-conditioned FRONT target that includes the value of inventory
reduction, followed by a frozen test on new forward days.

## Verification

- The new module's 12 lifecycle/schema checks pass.
- Original disabled-path, all-false, determinism, and JOIN-action-schema
  controls pass unchanged.
- Cancel-all all-false parity and deterministic replay pass.
- Python compilation, artifact self-hash, provenance hashes, and inventory
  reconciliation pass.
