# EV latency gap: recorded medians and anticipation hypothesis

As of 2026-11-16T01:57:35.556172+00:00, 1,434 first-minute windows on five consumed dates
(2026-10-27, 10-31, 11-01, 11-02, 11-03); six October 31 boundary exclusions.
All fills retained. Research only, no new model fits or policy promotion.

## What was missing, and what this run establishes

The prior full-EV study used 40 ms placement and 22 ms cancellation. The
other session had tested the previously recorded 58/37 ms medians for its
QR/LDG/MD policies, but not this full-EV rule. This run fills that gap.
It reproduces every prior baseline window, then changes only placement and
cancellation to 58/37. This is a deterministic median sensitivity, not a
measurement of present latency or a replay of its distribution.

| Clock | Total marked trading value ($) | Cents/share | Filled shares | Positive dates |
|---|---:|---:|---:|---:|
| 40/22 ms placement/cancel | +397.10 | +0.9259 | 42,890 | 4/5 |
| 58/37 ms recorded medians | +159.56 | +0.3312 | 48,173 | 4/5 |
| Zero order/notice delay | +642.44 | +0.9252 | 69,439 | 5/5 |
| Zero modeled feed + order delay | +820.92 | +0.9612 | 85,402 | 3/5 |

The first two rows are the new paired run. The zero-delay rows are from
the prior frozen study, completed 2026-11-15 18:58:55 UTC, on the same windows,
heads and M50 tape assumptions. All four retain the 50 ms decision grid.
Value is cash plus inventory marked at 60.2 seconds; these are not whole-day
or executable-liquidation profits. Rewards and a new fee/cost model are not
included. Positive simulation value is not established live profitability.

| Date | 40/22 ms ($) | 58/37 ms ($) | Paired difference ($) |
|---|---:|---:|---:|
| 20261027 | +197.70 | +200.40 | +2.70 |
| 20261031 | -59.99 | +9.39 | +69.38 |
| 20261101 | +127.77 | +11.32 | -116.45 |
| 20261102 | +75.76 | -62.87 | -138.63 |
| 20261103 | +55.86 | +1.32 | -54.54 |

Pooled median-minus-baseline: $-237.55.
Full modeled zero-minus-medians: $+661.36.
Neither is a smooth per-millisecond cost curve. Exposure and later actions
change with the execution clock.

The recorded-median setting loses 59.82% of baseline value despite 12.32%
more filled shares; value per share falls from 0.9259c to 0.3312c. Its
positive pooled value is concentrated: October 27 contributes +$200.40,
while the other four dates sum to -$40.85. All dates remain in the primary
result; this arithmetic describes fragility, not a proposed date filter.
For the existing full-EV policy, sensitivity to execution timing is now a
directly measured weakness. The data do not identify how much comes from
later placement, later cancellation, review phase or their subsequent paths.
These calculations are reproduced in results_v1/interpretation_v1.json.

## Which gap is actually established?

At the same old clocks, the full-EV rule versus the fill-probability-only QR
rule differs by $+1215.49.
Removing order/notice delay from EV adds
$+245.34; additionally removing modeled
feed delay adds $+178.48. The policy comparison
has endogenous exposure; these are not independent, additive causal shares
of the performance gap. The other session's 250 ms QR replay, recalibrated
threshold and young-fill exclusion are also a different policy/population.

This evidence supports improving economic decisions and time-sensitive
execution; it does not identify LGBM architecture as the dominant bottleneck.
There is a concrete target limitation: the local 1 s keep-value estimate is
also used for fresh entry, and its sign test does not price relinquished
queue priority. A profitable closed-loop replay does not validate its static
calibration or establish that every cancellation has positive incremental
value. The contribution of that target mismatch remains unmeasured.

Correction to the earlier inventory-first prioritization: terminal value
minus +1 s marked value is $+46.50
at baseline, $-92.21 at the recorded
medians, and $-148.48 at full zero.
Selected losing dates did not establish inventory management as the largest
pooled lever. These are mark-horizon differences, not executable exit gains.

## Can 50/100 ms anticipation recover the zero-delay benefit?

In principle, anticipating the correct instruction sufficiently early can
make its arrival coincide with the desired action time. An order-specific
early-post or early-cancel forecast can therefore recover some latency loss.
The required horizon depends on feed age, waiting for a decision and the
instruction's own delay; it is not automatically the same for both actions.
Earlier posting also changes queue position and exposes the order to fills
before the later decision would have existed.

A zero-latency replay reveals the benefit of fresher information and faster
actions. It does not demonstrate that future information is predictable.
Generic 50/100 ms price-direction accuracy is insufficient: the relevant
forecast is whether this side, price and queue position benefits from acting
now rather than one or two review steps later. Severe wrong posts/cancels
must be weighted by value, not treated like cheap classification mistakes.

There is also a verified review-grid interaction. An order posted at t=0
with activation at 40 ms first faces its model guard at 50 ms (age 10 ms).
At activation 58 ms, it first faces that guard at 100 ms (age 42 ms).
At activation 0 ms, first review is 50 ms (age 50 ms). Pending orders do not
receive that EV guard review, although structural handling can cancel them.
The synthetic probe verifies this scheduling mechanism, not its empirical
share of PnL or a defect requiring automatic repair.

## Decisive next experiment, not yet performed

Freeze one realistic execution clock and compare the current policy with:

1. An explicitly nondeployable 50/100 ms lookahead diagnostic for individual
   post and cancel decisions, retaining actual activation, queue, fills and
   notification delays. It measures a limited anticipation opportunity;
   do not call shifted future features a causal forecast or a physical bound.
2. A causal forecast of the same bounded decision value using only information
   received at decision time. Separate post-now versus wait and cancel-now
   versus defer; use a common short continuation and terminal mark rather
   than an unlimited chain of future optimal decisions.

Build labels on neutral decision-time exposures, including no-fill actions;
carry ambiguous timing/support as explicit statuses and sensitivities. Report
value recovered, harmful early actions, missed good fills, queue retention
and tail losses. Match controls on side/rung/action intensity; freeze the null
with at least 200 draws before an inferential claim. Model selection on these
consumed dates is development only; validation requires later untouched days.
First establish the lookahead opportunity survives timing/queue checks, then
measure how much a causal forecast can recover. No forecast recovery is
measured in this receipt.

## Reproduction and checks

Source 06fb7d77; pre-outcome freeze 37d61a9d. Two fixed clock configurations,
five dates, no fit or threshold search. All runs launched serially through
run_light.sh in the Codex/core 8 slot. Same frozen v3 heads, fresh entry and
review every 50 ms, six rungs x 30 shares per side, M50 trade timing, imputed
untimed legs, activation queue M and unknown-depth queue zero; no inventory
cap or retry timer. Feed clocks, 95 ms pending cancellation and 100 ms fill
notification unchanged. Physical timing, queue and market impact remain
assumptions under the separate simulator review.

All 1434 baseline windows reproduce prior PnL,
shares, orders, pulls, fills, max position and rung exposure. Independent
known-state comparisons: 6188; cached/uncached
trajectory comparisons: 10. Prior source,
model and input identities, all lane/terminal accounting and output aggregate
reconciliations pass. Placement and cancellation toy controls fire; corrupted
baselines/aggregates and the constant-first-review assumption are refused.
The review-phase probe was added after source inspection during the run;
it is a mechanism check, not another empirical strategy candidate.

Evidence: reviews/ev_lag_gap_20261116/freeze_v1.json,
results_v1/report.json, results_v1/interpretation_v1.json, receipt_*_v1.json.
Source: research/pf_ev_lag_gap.py, research/lag_gap_report.py and
research/lag_gap_interpret.py. Window parquet files remain outside git under
data/predict_fun/research/ev_lag_gap_20261116_v1/. No inferential interval or
significance claim on these consumed observations.
