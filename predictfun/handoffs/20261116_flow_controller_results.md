# First coherent forecast-to-action controller pilot

Strategy readout as of **2026-11-16 11:00:34 UTC**. Test date **2026-10-27**,
12 preselected markets (first selected market in each two-hour block), full
five-minute contracts, 48 strategy/market replays, no missing selected market.
This is a small development pilot on one previously consumed day. It is not
untouched validation, a full-day return, or a promotion decision.

## What is now implemented

The pipeline connects the existing 15-class joint flow/price predictor to
paired historical training paths, conserved-volume execution, explicit
portfolio action comparisons, and continuous replay. Post/wait and keep/cancel
use the same future scenarios and valuation. Both sides share inventory risk
and reservation limits. Entry and cancellation have no fixed retry or minimum
rest timer. Price/size optimization and a complete long-term continuation
value remain outside this minimal controller.

The response layer replaces the independently fitted quantity/value/risk heads
for controller decisions. Each scenario produces fills, cash, inventory and
value through one accounting calculation; the original prediction artifacts
remain intact. The old heads were not silently clipped or declared calibrated.

Queue-state repairs preserve front priority after a confirmed partial fill,
defer the queue cache until placement activation, and prevent later snapshots
from increasing initially unknown depth under the declared zero-depth
assumption. Versions 1–2 remain preserved with superseding freezes; active
controller version 3 includes all three repairs.

## Identical controls and distinct strategies

All cases use six 30-share rungs per side, the corrected queueM simulator with
unknown depth zero, placement/cancel/notice/review clocks of 58/37/100/50ms,
95ms pending cancellation and M50 trade anchoring. Net worst-case reservation
is capped at 180 shares. Quoting shuts down in the final two seconds. This
constraint is shared by all cases. M50 is a timestamp assumption, not latency.

- **KEEP_ALL_CAP:** always-rest ladder under those constraints; no learned
  cancellation. It is not the earlier uncapped KEEP_ALL configuration.
- **EMPIRICAL:** one-second action controller with training-only empirical
  path weights.
- **JOINT:** the same one-second controller with weights adjusted using the
  frozen learned joint forecasts for both sides.
- **JOINT_H2:** predeclared two-second endpoint sensitivity, using the same
  learned forecasts and paired paths. It was not selected after seeing returns.

The controllers maximize expected endpoint wealth minus a quadratic net
inventory penalty. Its marginal penalty is one cent per share at the cap.
This is a fixed economic design choice, not a fitted optimum. Gross collateral,
dynamic liquidation and indefinite queue continuation are not fully modeled.

## Pilot outcomes

Values below are gross simulated accounting across the 12 markets. Markouts
are quantity-weighted cents per covered filled share; P&L is cash plus ending
inventory at the stated valuation. These are not top-maker or live comparisons.

| Strategy | Filled shares | Initial edge c/sh | 1s markout c/sh | 5s markout c/sh | Both sides quoting | Terminal marked P&L | Passive settlement P&L |
|---|---:|---:|---:|---:|---:|---:|---:|
| KEEP_ALL_CAP | 59,686 | +0.61 | −2.43 | −2.24 | 61.84% | −$1,531.07 | −$1,557.14 |
| EMPIRICAL | 11,870 | +1.12 | −2.59 | −0.98 | 26.58% | +$129.04 | +$102.15 |
| JOINT | 8,399 | +1.70 | −1.44 | +1.67 | 32.84% | +$246.76 | +$232.39 |
| JOINT_H2 | 7,009 | +1.57 | −0.79 | +0.17 | 12.41% | +$98.00 | +$72.75 |

All filled shares have one-second mark coverage. Five-second covered shares
are respectively 59,165.13 / 11,704.74 / 8,218.06 / 6,857.89; expiry exclusions
are retained. Quote availability uses the entire contract-time denominator.

JOINT exceeds EMPIRICAL by $117.72 in terminal marked value and $130.24 in
passive settlement value, with positive paired differences in 8 of 12 markets.
Both-sided quote time is higher in all 12, but filled volume is lower. Mean
absolute ending inventory is 65.46 versus 118.33 shares. Different exposures
and one consumed day prevent interpreting the portfolio gain as established
forecast skill or general profitability.

## Churn remains a material problem

JOINT places 11,974 orders and issues 10,896 guard pulls, versus 6,080 and
4,933 for EMPIRICAL. Excluding final forced shutdown, JOINT has 10,884 model
cancellation decisions: median decision age 92ms, 41.76% at the first review,
and 70.45% before 250ms. These are cancellation-decision ages, not actual order
lifetimes; cancellation latency and earlier fills remain separate.

The two-second variant still has 8,044 model cancellation decisions, median
age 42ms and 54.45% at the first review. A longer endpoint alone does not solve
the churn. No fixed timer was introduced to disguise it.

The one-second markout is negative for every strategy. JOINT's positive
five-second/terminal values therefore cannot be presented as evidence that it
already captures profitable one-second fills reliably.

## Action-value validation and remaining model gaps

The first action diagnostic is superseded: 169 of its 222 neutral states had
net inventory outside the controller's cap. Its original correlations and null
comparison must not be used as the current action-quality evidence. Strategy
replays and churn statistics above are unchanged.

The corrected diagnostic, as of **2026-11-16 11:11:06 UTC**, uses a neutral
reference with the same hard reservation cap in the reference and both action
branches. It generated 7,240 actions: 4,574 eligible with estimated timing,
15 eligible, 2,476 not client-resting, 93 near the shutdown horizon, 81 lacking
book coverage and one with stale potential marks. Hash selection yields
**221 evaluated actions**, all within the net inventory cap; maximum absolute
known position is 179.54 shares. Every branch has identical historical and
unavoidable-cancellation prefixes.

- Learned action-advantage rank correlation: **0.280**, versus **0.140** for
  empirical weights on exactly the same actions.
- Mean predicted/realized cancellation advantage: **2.47 / 4.14 cents per
  action**. For 137 positive predictions, **4.92 / 5.55 cents per action**.
- The count-matched 200-draw diagnostic gives **759.73 cents** of realized
  selected advantage versus **624.22 cents** for the null mean; the null spans
  −49.67 to 1,147.37 cents. This is descriptive, not a significant-profit claim.
- Expected own quantity averages **3.00 shares** versus **2.55 realized**.
  Predicted/realized mean wealth benefit of cancellation is **2.02 / 3.88
  cents**, with the inventory-penalty contribution reported separately.

These results support useful action ranking, with remaining calibration error.
They cover resting cancellation on a bounded neutral path, not fresh-entry
calibration or the learned policy's own state distribution. All original
outcomes remain preserved, and no model or strategy parameter was selected
using this diagnostic. Its independent audit passes artifact, identity,
status, decomposition and null recomputation checks and refuses a corrupted
realized value. Runtime 153.40s, peak 1,079MiB; audit 0.91s, peak 204MiB.

The [active result index](../reviews/flow_control_20261116/summary_v2_receipt.json)
and [bounded diagnostic](../reviews/flow_control_20261116/bounded_probe_v1/report.json)
record the supersession in machine-readable form.

The empirical path mapping is physically coherent but remains coarse. Across
64,103 eligible reviews, an average 4.67% of learned probability mass lacks a
matching path class; the projected marginals' average L1 discrepancy is 0.126.
These are probability-distribution diagnostics, not percentages of quoting
time. Another 80 reviews have no matching scenario bank; 7,349 have invalid
current books. All statuses are counted. The classifier's original AUC cannot
be treated as the accuracy of this translated action-value distribution.

The learned forecast improves bounded-neutral action ranking, but this does
not yet isolate the source of the remaining trading errors. Remaining limits
include within-class flow/price severity, empirical path
transport across market conditions, uncertainty in client-known queue/remainder,
finite-horizon continuation, and frequent decisions on small forecast changes.
No change of model family is justified by this pilot alone. The next priority
is action-value calibration and decision stability, including uncertainty in
small predicted advantages and the value of continuing to rest an order. Any
calibration must use earlier data, with the current pilot retained as consumed
development evidence; it must not become a fixed retry timer.

## Verification and artifacts

The compiled branch executor matches the canonical simulator on 24 nonempty
synthetic paths. Positive and known-bad controls cover conserved shared volume,
cash/inventory, side signs, common cancellation exposure, placement latency,
queue priority, pending-state causality and inventory bounds. All neutral
observer replays match the original simulator; completed strategy fills are
independently reconciled to raw books and portfolio summaries. Source/model/
binary/data artifact identities are pinned and checked.

Pilot: 964.62s, peak 1,280MiB on one numerical core. Additional readout:
10.04s, peak 745MiB. Protected forward dates remain unopened. No live code,
exchange integration or production deployment was added.

Active controller source 6f1742fc, freeze 7fb46122; implementation merged to
mm-research at de73ec47. Readout source 0f5932d2. The [protocol](20261116_flow_controller_protocol.md)
and [primary report](../reviews/flow_control_20261116/pilot_v3/report.json),
[interpretation](../reviews/flow_control_20261116/pilot_v3/interpretation.json),
[audit](../reviews/flow_control_20261116/pilot_v3/audit.json) and
[comparison chart](../reviews/flow_control_20261116/pilot_v3/comparison.png)
record the full evidence. Raw ledgers remain under ignored
`data/predict_fun/research/flow_control_20261116_v3/`.
