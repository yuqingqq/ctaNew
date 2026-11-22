# Two-sided turnover, inventory and the value of resting orders

2026-11-22. Adds to the governing contract committed in
**c590b083bb129b9a3f936831a72f773847981490**. USER expects long resting to
capture repeated bid/ask spreads with limited inventory and asks how to
estimate its economic value. This is a design clarification, not a new run.

## What should be estimated

Value an order's contribution to the **joint two-sided quoting process**.
The benefit may be repeated partial fills and inventory-reducing fills at
already-earned priority. The useful prediction is not just the probability
of another fill; it also concerns the opposite-side fill's timing/price and
the inventory carried until then.

Longer resting can improve turnover, but low inventory is a hypothesis to
check. Balanced expected buy/sell volume does not imply balanced realized
flow, and flat terminal inventory can conceal large intermediate exposure.
The current cap/hysteresis controls limit positions but do not guarantee
fast or profitable offsetting fills. Preserve those controls in the first
comparison; do not raise size/caps or change their parameters implicitly.

For each prospective action and its declared continuation to common T,
estimate paths of:

- bid and ask fill times, quantities and execution prices, including partials;
- queue/remaining size on both sides and levels, with real re-entry resets;
- inventory I(u) = I(t) + cumulative bid fills - cumulative ask fills;
- market price movement jointly with those fills, including one-sided flow
  and adverse sweeps, and any replacement/constraint decisions.

Do not multiply unconditional fill probability by unconditional movement,
assume independent buy/sell flow, or pair expected volumes as if that were
realized turnover. Conditional joint paths are needed to capture their
economic covariance and tail risk. A direct value predictor is also possible,
but must learn labels from the same full-ledger continuation and be checked
against these components. This note selects no new model family or estimator.

## Ledger value and useful decomposition

For actual/forecast paths, with signed fill quantity +q for bids and -q for
asks, retain the parent's primary value:

    profit(t,T) = net_cash(T) - net_cash(t)
                  + I(T)*M(T) - I(t)*M(t)

For explanation only, it can be decomposed exactly (with consistently
defined fill-time mids) into:

    execution_edge = sum_f signed_q_f * (mid_f - execution_price_f)
    inventory_movement = I(t)*(M(T)-M(t))
                         + sum_f signed_q_f*(M(T)-mid_f)
    profit = execution_edge + inventory_movement - net execution costs

Net execution costs must be counted once; the first expression uses net cash.
Per-fill markouts, inventory movement and realized round-trip profit are
alternative decompositions/diagnostics, not three independent profits/costs
to add together. Queue benefit is embedded in the additional future fills;
do not also add a queue bonus. Terminal marked inventory is disclosed and is
not equivalent to having completed a profitable round trip or liquidated.

Opposite-side fills are not automatically spread capture: selling after a
large adverse move may realize a loss despite reducing inventory. Report
actual matched prices and elapsed exposure, not merely the count of cycles.

## Target at the decision

The value of retaining a particular bid/ask generation is its **marginal
effect on the full quoting book's longer-horizon value**, compared with
cancelling/re-entering that generation from the same starting state. Other
orders follow the same specified policy but respond to their own branch's
fills, inventory and capacity. Do not sum independently valued bid/ask
orders and silently ignore their shared inventory and reservations.

At admission, include the early exposure needed to establish queue access.
At a later review, earlier cash losses are sunk; current inventory and
remaining queue value still matter. Do not pay for the same acquisition
cost twice. Predict from client-known state, including both-side queue and
remaining-size estimates, inventory/reservations, pending commands, recent
joint flow/price dynamics, price placement and time to the common end.
Venue hindsight may validate labels but must not enter decision inputs.

## Required evaluation for the same combined deliverable

Report the following from matched retain versus avoid/re-entry branches,
per day/side and in aggregate with complete counts/statuses:

1. Full marked ledger profit and paired difference per opportunity/time;
   retained/forgone good value and avoided bad value, net of costs.
2. Buy and sell filled shares and completed offsetting volume, with a declared
   lot-matching convention (including starting inventory and remaining lots).
   Matched-volume diagnostics must reconcile to the cash/inventory ledger.
3. Time-weighted absolute inventory, peak inventory, terminal inventory,
   inventory-lot holding times, and time constrained by risk/cap. Do not use
   small terminal inventory alone to claim low inventory risk.
4. Execution edge versus inventory movement; adverse-selection diagnostics
   across supported horizons/common cohorts; lower-tail outcome and one-sided
   flow concentration, without unsupported small-sample confidence claims.
5. Quote participation, priority retained/lost, re-entry delay, order lifetime,
   cancel/repost counts and short post-to-pull reversals. More gross volume
   or more completed pairs is not sufficient without positive net economics.

This tests the USER's intended mechanism: **more profitable two-sided turnover
per unit of inventory exposure**, not simply more fills or longer residence.
It is a forecasting/economic comparison, not a rule forcing orders to remain
until inventory offsets. Long value can change quickly when one-way flow
makes replenishment or exit unlikely; immediate risk action remains possible.

## Scope

Fold these accounting/projection requirements into the single longer-
continuation contract and pilot projection already requested. Finish the
bounded saved-data diagnostic as a diagnostic; it cannot forecast hypothetical
opposite-side/re-entry fills by deleting rows alone. Prepare the actual paired
continuation design before new labels/fits. No new variant, replay, data day,
model fit or live use is launched here, and the 2.50 core-hour cumulative
ceiling is unchanged. No duplicate series of small classification experiments.
