# Post-run analysis for the persistent ladder

2026-11-21. Prepared at **cf4eee757747d9a6e54148e02aa5dfb00ea051c8**.
This specifies the measurements for the USER's current
[execution plan](../20261121_default_post_ladder/plan.md): default POST/KEEP,
two levels per side, nominal 100 shares per level, shared cap 200 and inventory
management, across the supported market lifetime. It supplements that plan;
it changes no policy, population, compute budget or implementation ownership.
No simulation or new result is claimed here.

## 1. The dashboard and the decisions it supports

| Question | Required measurements | What to investigate next |
|---|---|---|
| Are fills economically useful? | Execution edge at fill; markouts at +50, +100, +1,000, +5,000 and +30,000 ms; final marked value; totals and cents per filled share | Positive edge followed by negative movement points to adverse selection or exposure after filling; negative initial edge points to price/execution mechanics |
| Are large orders and queue age useful? | Actual submitted/filled shares, fill fraction, no-fill share, repeated partial fills per generation, time to first/full fill, queue ahead, age at fill, deep orders becoming best and their subsequent outcomes | Whether the intended queue mechanism is operating; which levels/sizes warrant a later comparison |
| Are we actually making a market? | Bid/ask/either/both-side active time and at-best time; active levels; absent time by cause | Separate weak opportunities from cap, inventory, pending, invalid-book, own-cross or stale-price restrictions |
| Are cancellations destroying participation? | Requests and effective cancellations by reason and level; lifetime, first-review pull rate, replacements and re-entry queue ahead | Identify unnecessary structural/inventory churn before adding a predictive guard |
| Is inventory being managed usefully? | Position/equity paths, peak and time-average absolute inventory, time beyond the soft trigger, recovery time, sign overshoot, terminal inventory, sizes and restricted-side time | Distinguish adverse new fills from losses on inventory already acquired; locate prolonged or ineffective inventory reduction |
| Do a few states account for losses? | Positive/ordinary/severe contributions, zero/no-fill counts, worst-market losses, top-five market loss share, side/level/minute and decision-state breakdowns | Choose one plausible, causally available discrimination or controller change for the next round |

The first four columns of the headline result should be **final marked value,
filled shares, both-side active time, and severe-loss contribution**. A lower
loss or better value per fill accompanied by near-zero participation is not
by itself better market making. Keep the full dashboard beside any later
candidate comparison; do not optimize a single ratio or cancellation count.

## 2. Economic accounting and longer horizons

Prices and midpoints are in cents per share, shares are quantities, and
values are cents. For fill f with sign s=+1 for a bid and -1 for an ask,
quantity q, execution price p and actual engine fill time t:

```
execution_edge(f) = s * q * (mid(t) - p)
movement(f, h)    = s * q * (mid(t+h) - mid(t))
markout(f, h)     = s * q * (mid(t+h) - p)
END_value(f)      = s * q * (common_END_mark - p)
```

On identical valid fills, markout = execution edge + movement. With zero
starting cash and inventory, sum END_value over every fill once must equal
cash_END + inventory_END * common_END_mark. Use the same END mark for both
sides and all generations of a market. These are gross values under the
declared zero-cost convention, not settlement proceeds or executable net P&L.

The 5s and 30s markouts are cheap outcome lookups on the already-built market
path, not extra branch experiments or holding constraints. Use the last valid
venue frame at or before the horizon, preserve the path's staleness rules,
and report midpoint age. Never use a horizon past the supported END, backfill
a missing mark, or count a missing observation as zero. For every horizon
show eligible fills, valid fills and valid filled shares. Compare horizons
on their common valid subset as well as showing each horizon's full coverage.
Outcome venue mids are not inputs available to the client before receipt.

Report the cumulative equity path and minute increments E(t)-E(t_previous),
carrying cash and inventory across boundaries. Those increments telescope
to final value. Do not add overlapping markouts or generation END values to
the minute equity increments: these are alternative views of the same fills.
Show within-market maximum drawdown as a marked-path diagnostic, with gaps
explicit. Terminal inventory is marked, never silently liquidated for free.

## 3. Denominators, queue, churn and inventory

- Keep every submitted generation, including refusals, no-fills and survivors
  at END. Show any-fill/submitted and any-fill/accepted separately; quantity
  fill fraction uses actual submitted shares, not nominal 100 for reduced
  orders. Separate all-submitted and accepted quantity denominators. Partial
  fills leave the same gid/queue; a replacement is a new submitted generation.
- First/full-fill delay runs from activation. Report fills and censoring at
  cancellation/END beside delay summaries; an average among filled orders
  alone cannot establish a shorter wait. Count repeated fills by distinct
  matching events, not arbitrary accounting fragments of one event.
- Quote uptime is duration, not number of decisions. Primary denominator is
  the full supported opening-to-END window; show valid-book coverage and a
  valid-book-only denominator separately. Post-pending orders are not live;
  cancel-pending orders remain exposed until removal/full fill. Count live
  prices, not duplicate requests. Keep both-side at-best distinct from merely
  having two distant live sides. Attribute downtime with a declared exclusive
  reason precedence and retain simultaneous reason flags for diagnostics.
- Keep birth rung and current price distance separate. Report queue ahead
  at activation, later sampled states and fill; time with model queue-ahead
  zero; and aged deeper orders that become best. These are simulator queue
  estimates. Do not claim that queue age guarantees fills or profitability.
- Separate cancel requests, accepted/effective removals, full-fill-before-
  removal, and pending-at-END. Report rates per submitted and per reviewed
  generation, plus first-review pulls/reviewed generations, by reason. Show
  replacement delay and queue-ahead change; these describe lost queue age,
  not its counterfactual monetary cost.
- Report actual venue inventory for exposure/value and client-known inventory
  for decisions, including their discrepancy during notice latency. Compute
  integral |I(t)| dt / duration, peak |I|, time |I| >= 100, and terminal I.
  Recovery starts when the controller enters a restriction and ends when its
  known inventory returns to |I| <= 50; report unrecovered/censored episodes
  and opposite-side overshoot. Show aggregate reservations/pending exposure
  and actual posted size alongside cap/restriction time.

## 4. Loss distribution and market conditions

For each generation's END value, show positive, exactly zero, ordinary loss
and severe loss contributions. Preserve both declared severe definitions as
separate views: value < -30 c/generation; value/submitted shares < -1 c/share.
Do not mix their denominators. Sum generation values back to the market.
Report worst markets and the top five markets' share of gross negative
market value, with the number of losing markets; zero loss gives an undefined
share, not zero evidence of tails. Also show the value excluding severe
generations as an **ex-post decomposition**, never an achievable strategy.

Break out results by consumed day, market minute, side and birth rung, with
n markets/generations/fills, actual shares, status counts and as-of. Related
fills and generations are not independent samples. Four sampled consumed
days support descriptive development readings, not confidence intervals,
full-day returns, or untouched validation. A large tail contribution does
not establish a particular heavy-tail distribution.

Use receipt-known states at submission and the plan's fixed/hash-selected
live checkpoints: spread, distance from best, displayed depth/queue ratio,
order age and remaining shares, recent signed trade flow, book imbalance and
its changes, price movement/volatility, source ages, remaining market time,
known inventory, reservations and pending flags. Reuse available builders
and record feature units/missingness. Keep ordinary, gain and no-fill states
beside loss states. Do not select training rows only because they filled or
pick a checkpoint because a subsequent loss is nearby. Fill-time and venue
conditions are explanatory outcome diagnostics, not available predictors.

For representative severe cases, inspect the received observations, earliest
decision using them, command time, cancellation-effective time, fills and
subsequent inventory path. Separate fills already unavoidable by that command
from later fills, and price movement on inventory already acquired. A cancel
cannot undo a fill. This can locate a timing limitation; it does not measure
the P&L an unrun counterfactual would have saved.

## 5. One combined receipt, then one next experiment

Implement these as summaries of the same saved run, with the existing
accounting/timestamp checks and focused positive/known-bad controls. No new
training, parameter grid, extra replay arm or large mutation project is
requested. Freeze definitions before outputs; disclose any unavailable
diagnostic rather than stopping useful work for cosmetic completeness.

One policy can show queue accumulation, inventory exposure and where value
is lost. It cannot prove that two levels beat one, that inventory control
improves P&L, or that observed cancels saved losses. After the combined review,
choose ONE change supported by the loss/participation mechanism. Any claim of
selection advantage then needs a declared comparison, with participation
matched on the relevant decision variable and the same economic metric.
The current run's purpose is to identify that change efficiently.
