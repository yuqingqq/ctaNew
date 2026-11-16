# Strategy names and performance accounting

Use this naming scheme for new comparisons. Historical filenames and frozen
artifacts retain their original names. The completed comparison is in
[the strategy scorecard](handoffs/20261116_strategy_scorecard.md); its
[registry](reviews/strategy_scorecard_20261116/freeze_v1.json) pins the cases,
model versions, clock settings, metrics and source hashes.
Machine-readable results are in the
[strategy totals](reviews/strategy_scorecard_20261116/results_v1/strategy_totals.csv)
and [per-date table](reviews/strategy_scorecard_20261116/results_v1/strategy_by_day.csv).

## Strategy names

| ID | What decides entry? | What decides keep/cancel? | Historical alias |
|---|---|---|---|
| `KEEP_ALL` | Post whenever the ladder permits | No learned guard; ordinary price/end handling still applies | KA |
| `FILL_GUARD` | Withhold when predicted fill probability is high | Pull when predicted fill probability is high | QR in the 50ms latency comparison |
| `SWEEP_GUARD` | Withhold when predicted sweep risk is high | Pull individual orders according to sweep risk at their price | MD1000 |
| `ORDER_EV` | Post when the order-state-only EV is nonnegative | Pull when that EV is negative | B2 |
| `STOP_EV` | Post when full STOP-EV is nonnegative | Pull when full STOP-EV is negative | EV, full-EV, arrival-test baseline |
| `ARRIVAL_EV` | Post when the causal forecast of arrival-time STOP-EV is nonnegative | Original STOP_EV guard | forecast |
| `FUTURE_ENTRY` | Uses the actual future arrival-time teacher score | Original STOP_EV guard | arrival_oracle; noncausal diagnostic |

The STOP-EV **model** has two heads: fill probability and conditional mean
filled value. Their product is its expected-value estimate. The corrected v3
full model uses 71 market fields and 12 order fields. The order-only B2 is a
separate model. `FILL_GUARD` uses only the full model's fill head, whereas
`STOP_EV` uses the product of both heads. A negative local model-evaluation
result is not the P&L of either strategy.

`ARRIVAL_EV` has additional, separate bid/ask regressors that forecast the
teacher score for the **same absolute quote price** at activation 58ms later.
Its cancellation rule is unchanged. `FUTURE_ENTRY` supplies future information
to entry; it cannot be deployed and is not an optimal-policy profit bound.

Use `FILL_GUARD_250_RECALIBRATED` when referring to the distinct STOP-EV-PLAY v2
study: its 250ms decision schedule and recalibrated thresholds differ from the
50ms fixed-threshold FILL_GUARD comparison. Never shorten both to “STOP-EV.”
The original stale-gate STOP-EV-PLAY replay was superseded and is not a current
strategy result.

## A strategy name is not an execution scenario

Every quoted result must specify the strategy, model version, clock, review
cadence, fill-clock/queue assumption, order size and valuation endpoint.

| Clock ID | Place / cancel | Fill and cancel notification | Feed assumptions |
|---|---|---|---|
| `legacy40` | 40 / 22ms | 100ms | Original recorded observation clocks |
| `recorded58` | 58 / 37ms | 100ms | Same observations; deterministic previously recorded order medians |
| `zero_orders` | 0 / 0ms | 0ms | Original observations |
| `zero_all` | 0 / 0ms | 0ms | Zero modeled feed delay, recorded events only |

Pending-placement cancellation is 95ms in the first two scenarios and zero
in the latter two. All scorecard cases still review every 50ms. M50 means
the **trade anchor is shifted back by 50ms in replay**; it is not an order
latency or a forecast horizon. Queue M assumes displayed queue at activation,
with the inherited unknown-depth-zero assumption. Six 30-share rungs per
side, no cap, and no fixed repost cooldown are shared by these cases.

Example: **STOP_EV / v3 / recorded58 / review50 / M50 / queueM / 6×30 per side**.

## Metrics to quote together

Use gross values and filled shares, then report both dollars and cents per
filled share. These quantities describe different aspects of a strategy:

1. **Spread capture:** the side-signed execution edge against the last valid
   public midpoint strictly before each modeled fill. It is a proxy for
   spread capture, not a matched round-trip trading profit.
2. **Midpoint movement after filling:** side-signed movement from that midpoint
   to +100ms, +500ms, +1s or +5s. Negative movement is adverse selection.
3. **Markout:** spread capture plus the subsequent midpoint movement. A +1s
   markout is not an executed sale after one second.
4. **Terminal P&L:** cash plus inventory marked at the common 60.2s endpoint,
   after first-minute quoting and the cancellation tail.
5. **Settlement P&L:** cash plus the final inventory's actual recorded binary
   payout, assuming no more trading after the 60.2s endpoint.
6. **Settlement carry:** settlement P&L minus terminal P&L. This identifies
   how much the final inventory's move to resolution contributes.
7. **Activity and risk:** filled shares, submissions, guard pulls, filled-order
   fraction, any/both-side quote availability, mean/max absolute inventory,
   and the per-day distribution of P&L.

Prices and inventory use an Up/YES-equivalent convention: bid fills buy Up;
ask fills sell Up, economically equivalent to buying Down at the complementary
price. The settlement audit also calculates the separate Up/Down token payouts.

Submissions include rejected placements. The filled-order fraction counts
orders with at least one fill divided by submissions. The historical `pulls`
field counts guard-triggered pulls; it excludes ordinary price-management and
end-of-window cancellations. Quote availability measures physical exposure,
including the time an order remains executable while cancellation is pending.

New scorecard markouts start at the **modeled execution time**. Earlier latency
reports start at the unshifted recorded/imputed anchor, 50ms later in M50.
Both are retained and named separately. The old terminal values reproduce
exactly; changes in the displayed short-horizon markout convention are not
strategy improvements.

All fills remain in the headline. Show the young-order, WS/E2, imputed and
cancel-pending partitions as attribution, with population counts, as-of and
mark coverage. Keep gross trading P&L separate from unmodeled fees/rewards.
Do not mix local-label sums with portfolio P&L, total dollars with daily
averages, or first-minute trading with a full five-minute trading simulation.

Settlement results can be dominated by inventory outcomes. Assess them beside
spread capture, short-horizon markouts and inventory exposure. Existing queue,
timing and counterfactual-fill limitations still apply; these consumed-data
comparisons do not establish live profitability.

## Future comparison records

Keep entry and cancellation changes explicit in the strategy description.
Attach a clock ID to each result, even when the policy is unchanged. Use a
new dated registry and output directory for the next experiment; preserve
this frozen comparison and its historical aliases.

The shared accounting implementation is
[`pf_strategy_metrics.py`](research/pf_strategy_metrics.py). It produces
fill-level spread, markouts and settlement values plus window-level cash,
inventory and availability. Reconcile these against the engine before
aggregating. Present the same dates and windows for every comparator,
include all fills, and state exclusions and valuation endpoints alongside
the table. Dollar totals and cents per filled share belong together.
