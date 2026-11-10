# Preserve trading value while reducing inventory: design review

2026-11-10. User requested useful work while own-fill calibration awaits a new
capture. This is a source review and a proposed experiment, not a new PnL result
or permission to use the protected Nov9–13 data. Native source reviewed: main
79a05d3. Research reference: frozen pm_ladder_v3, with continuous-basis and
incremental execution differences kept explicit.

## Findings in the current implementation

1. **Entry quality and exit willingness use the same gate.** In native
   `crates/pf-strategy/src/ladder.rs::desired_ladder`, the adverse/PM90/GS check precedes
   the reduce-only branch. RO60 changes side and size, but does not change the
   economic hurdle for reducing inventory. The per-minute PM90 cutoff rises
   through the closing phase. This makes a cost-bounded exit objective worth
   investigating; it does not establish that the gate should simply be removed.
2. **CAP10 is an adding trigger on known net, not a hard 10-share portfolio
   bound.** It checks current known net before sending, rather than the possible
   position after the new order and other fills. The separate kernel backstop
   counts pending exposure. Entry admission should distinguish the marginal risk
   of another order from a uniform reduction in every order's size.
3. **Small is not always easier to close.** RO60 caps the closing order at known
   inventory; XMIN1 then rejects it if token price times shares is below $1.
   Hypothetically, 10 Up shares with a 2-cent Down quote permit only $0.20 of
   non-reversing Down purchases. Meeting $1 would require 50 shares and create
   reverse exposure. The refusal is correct. Entry/exit planning should expose
   this constraint instead of attempting to solve it by increasing inventory.
   This is a limitation of the current buy-complement route; it is not a claim
   that every possible venue liquidation route is unavailable.
4. **The guard score is not settlement probability or directly spendable edge.**
   Frozen params define it as model prediction minus a fitted percentile offset.
   Its target is short-horizon exposure value. Do not subtract dollar exit cost
   from that score or interpret its sign as expected terminal profit without a
   separately checked conversion and calibration.
5. **Queue preservation is already partly implemented.** Incremental sizing
   keeps the front order on same-price growth. Its request/coverage benefit does
   not establish better realized spread capture. Unrestricted soft retention
   also increases active exposure; it is not by itself an inventory reduction.

These are design trade-offs, not newly discovered violations of the frozen rules.

## What the existing experiments rule out

The consumed-data receipts already show why another blanket rule is unlikely to
resolve the objective:

- Uniformly smaller entries reduce turnover and short-term trading value along
  with inventory. Strict reservations can also starve closing liquidity.
- Smaller closing orders can leave more residual inventory, partly because of
  minimum notional. Keeping or enlarging eligible closing orders repairs some
  reducing flow relative to strict reservations, but did not beat the original
  baseline on both trading value and inventory.
- Letting the reducing side ignore the line was already tested as RX5. In the
  saved spread-capture report, reducing fills' +1s value per share is worse under
  RO60-240+RX5 than RO60-240 at both J and X. This is a historical simulator
  warning against unconditional relaxation, not calibrated live evidence.
- A one-tick late concession gave timing-dependent results. Better settlement
  PnL can reflect carry rather than improved execution. FIFO matching also
  includes drift between fills and is not pure captured spread.

The source snapshot lists the exact receipts checked. They remain consumed,
exploratory comparisons under a disputed fill model; this review does not
upgrade their statistical or economic status.

## Next design: price inventory risk and closing cost separately

Keep the current entry signal as the reference and evaluate a separate closing
objective. A closing trade may give up some trading value in return for less
uncertainty; make that cost explicit and bounded. Preserve fresh-data, book,
price, notional, exposure and stop checks. An exit exemption from a profit
threshold must not become a general exemption from adverse selection.

The underlying principle is an inventory-dependent valuation, as in
[Avellaneda and Stoikov](https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf).
Their stock-price model and terminal-midpoint assumptions do not directly apply
to binary settlement. In particular, approaching expiry does not automatically
remove the uncertainty of a still approximately 50/50 binary outcome.

For a fixed binary inventory held to settlement, with net Up exposure q and
probability p, payoff variance is q² p(1-p). The exact marginal variance of a
signed fill s of z shares, holding p fixed, is
`((q + s*z)^2 - q^2) * p * (1-p)`. This identity can help define an objective;
it is not a forecast of a dynamic strategy's variance. Market probability is a
reference estimate, with sensitivity checks, until an independent forecast
shows incremental value. The distribution-free fixed-inventory bound q²/4 is
also useful for exposure reporting; differences of such bounds are not estimates
of actual variance changes.

Pending orders require the full interval
`[known_net - possible_Down_fills, known_net + possible_Up_fills]`.
Include pending creates, resting remainders, pending cancels and unresolved late
fills once each. Opposite outstanding orders must not cancel out in a risk
calculation: either side can fill alone. Pending fills also consume closing
room, so an order described as reducing must be checked again at submission.

Do not anchor an exit to the purchase price. For accounting, buying Up at 70c
and the same quantity of Down at 27c locks 3c per pair before costs if both fills
occur. That is a payoff identity, not a fill forecast. Requiring every exit to
lock a profit can trap losing inventory; the decision should use current
executable alternatives and a declared maximum cost of reducing risk.

## Concrete work that can proceed without the new live run

**First, audit exit feasibility on the unchanged reference action path.** Keep
every neutral action or generation, including unfilled and withheld actions.
For each inventory episode, record whether closing is limited by the score,
Binance condition, missing/stale inputs, price/book constraints, minimum
notional, outstanding reservations, throttling, or the stop. Preserve all
applicable predicates as well as the first blocking reason. Count action
episodes and duration; repeated decisions are not independent observations.
The already saved native guard states cover only 0–65 seconds, so they cannot
answer the full 60–240 second exit question without an explicitly scoped export.

**Then test one cost-bounded exit change in isolation.** Keep pre-60 entry
requests identical to the reference. During 60–240, first evaluate same-price
closing quotes that narrowly fail the profit cutoff; any price concession is
a separate arm with its cash cost included once. Compare with both the original
baseline and an otherwise identical reservation/queue control, so safety and
exit-price effects are not conflated. The cost budget is a new research
parameter, not the live $150 loss limit and not replenished by rebates.

**Only after that, test marginal entry admission.** Demand more value from
orders that expand the full possible inventory range, while retaining ordinary
size where exposure is low. Account for minimum executable closing quantity.
Keep this separate from the exit experiment to identify which change trades
away capture. Report the trade-off frontier rather than selecting a favourable
inventory cap after seeing settlement outcomes.

Before any numeric run, commit the complete implementation and declaration:
fixed candidate count, consumed dates, population/status rules, baseline,
evaluation metrics, fill/latency stress scenarios, and at least 200 matched null
draws if making an inferential claim. Include positive controls and deliberately
bad identity, time, reservation and notional cases. Deterministic feasibility
checks need no statistical significance claim. Full replay must propagate each
variant's fills into its own inventory and future actions; deleting suspect
fills from an old report is invalid.

The comparison must report spread at fill, subsequent adverse move, +1s value,
mark at a common 240-second boundary, settlement carry, explicit exit costs,
paired/residual attribution, traded shares, and inventory magnitude, duration,
extremes and square-time. Use common windows and frozen day-level inference.
Report value per share and total value; neither alone proves preservation of
trading gains. Inventory-square is an exposure metric, not measured PnL variance.

These offline checks can reject bad mechanisms and expose their costs now.
Robustness to assumed fills is still conditional. Calibrated own-order replay
and at least five complete untouched days remain necessary before claiming
better live PnL with lower inventory risk.

## Status

Source/evidence review complete. Exit-feasibility audit and cost-bounded exit
experiment are specified but not implemented or run in this review. No new
strategy PnL estimate, model fit, live change or protected market-data read.
Calibration B1 remains open; this work can proceed alongside its data request.
