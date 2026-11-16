# Joint flow, price and order-action research plan

User request: estimate the distribution of incoming orders and associated price
movement, then use those estimates to choose market-making actions.

Status: **plan only, 2026-11-16**. No new data build, fit or policy replay has
started. This document is not an experiment freeze or a performance claim.
The first implementation milestone is the data/target audit and empirical
joint-distribution baseline below. Research only; no order placement code.

## 1. The problem and the proposed model

At each decision time, estimate a conditional distribution over future public
flow and price/book paths. Translate each shared scenario into our possible
fills using the order's price, size, queue estimate and action latency. Compare
the resulting portfolio values of keep, cancel, post, wait and replace.

The key object is `P(flow path, price/book path | information received by t)`.
It is conditional on the present market, not a single historical distribution.
Aggressive volume, sweep depth and subsequent price movement must remain
dependent. Multiplying average volume by average price movement loses exactly
the association that can make large fills costly. Correlation here is useful
for prediction; it does not establish that incoming flow causes the movement.

The execution mapping adds client-known order and portfolio state:

```
received market state -> joint future scenarios
                                   |
orders + queue uncertainty + latency + candidate action
                                   |
                      own fills and portfolio outcomes
                                   |
                   compare actions at a common endpoint
```

The corrected STOP_EV already estimates local quantity-weighted fill value.
This plan retains its accounting. The additions are an inspectable joint
distribution, explicit action comparisons, consistent entry/resting valuation,
and inventory/queue continuation. Renaming the existing scalar EV does not
establish an improvement.

Current motivation: the [presence diagnostic](../reviews/stop_ev_presence_20261116/stop_ev_presence.txt)
finds first-review cancellation and disagreement between fresh and resting
scores; the [continuous-minute study](20261116_stop_ev_minutes.md) finds phase
and timing-source sensitivity. Neither proves this proposed model will work.

## 2. Stage A — establish what we can observe and label

Build a source/feature/target manifest covering the entire five-minute contract.
For every field record units, source event, receipt time, missingness, and whether
it is observed or inferred. Preserve the existing accounting and clock contracts.

| Component | Required information | Treatment of uncertainty |
|---|---|---|
| Aggressive flow | Buy/sell trades, quantities, prices, event sequence and source | Keep WS, estimated and imputed timing identities; no invented exact arrival times |
| Book dynamics | Spread, depth, net depletion/replenishment, movement of best prices | A snapshot change alone cannot distinguish gross additions, cancellations and trades |
| External market | Available causal BTC returns, signed flow, volatility, cross-market dislocation | Join on received time; stale and unavailable inputs remain explicit |
| Contract | Time remaining, price relative to the round's starting reference, payout metadata | Use the contract's actual reference; distinguish an external-price proxy |
| Our orders | Price, original/remaining size, age, acknowledged fills, pending actions, queue estimate | No true simulator queue/remainder or unreceived fill used as an input |
| Portfolio | Both outcome holdings, cash, reserved capital, pending matched exposure | One reconciled portfolio; distinguish economic exposure from free collateral |

Predict's public feed consists of full book snapshots and can combine multiple
changes. YES and NO are complementary views of the same book. Consequently,
exact individual order arrivals and cancellations may be unidentifiable from
our archives. Use observed trade flow and net book transitions where necessary;
label inferred event paths as scenarios. Do not fabricate order-level event
truth. [Official book documentation](https://dev.predict.fun/concepts/order-book).

Use one outcome-independent probe per neutral order generation for primary
action labels, retaining no-fill actions. Market-transition observations can
support auxiliary forecasting, but repeated reviews, overlapping horizons and
hypothetical alternatives must not multiply the economic sample. Record action,
generation and market identities and declare their training weights.

Normalize side signs and price distances in ticks/cents. Retain absolute price,
depth and size alongside trailing liquidity ratios and log quantity features.
Preserve tail outcomes in dollars; do not winsorize away the loss being forecast.
Fit scales and bin boundaries on training data only. Unknown queue, zero depth,
stale feeds and missing marks are distinct states, not interchangeable zeros.

**Deliverable:** schema, source coverage by day/side/depth/contract phase, small
reconciled example trajectories and executable validity controls. Include
no-flow, partial-fill, sweep, expiry, delayed-notification and invalid-data cases.
Every checker must accept a valid fixture and catch a planted error. Changing
future events must not change any earlier feature or decision.

**Gate:** raw trades/statuses, engine tranches, quantities and cash/inventory
reconcile; all joins are causal. If event timing is insufficient, use a coarser
observable transition target or mark the proposed fine-scale target unsupported.

## 3. Stage B — measure the joint distribution before fitting a complex model

Proposed first exposure horizon: 1 second, with elapsed-time bins ending at
50, 100, 250, 500 and 1,000 ms. These are development design choices, not an
assertion that the tape resolves every bin. Separate the exposure window from
the 1-second/5-second post-fill markouts and contract-end valuation. Near expiry,
report the shorter supported horizon and censored marks explicitly.

Describe, conditional on current state:

- Probability of no aggressive flow; direction, quantity, timing and depth
  reached when it occurs. Preserve opposite-side and multi-level dependence.
- Book replenishment/depletion and price movement associated with each flow
  scenario, including continued moves and reversals.
- For neutral probe orders, filled quantity and time, initial execution edge,
  signed post-fill movement, and the resulting dollar-value distribution.

Start with training-only empirical distributions in a small declared set of
state groups: side, depth/queue, liquidity, flow pressure, volatility and contract
phase. Back off sparse groups to broader groups and expose effective support.
Do not discover a profitable subgroup on the evaluation days and call it skill.

Keep whole observed flow/price sequences paired when constructing an empirical
scenario baseline. An initial coarse joint histogram can diagnose dependence;
it is not yet a valid event simulator. Any later sampled path must also pass
book, volume and chronology checks before being given to the fill engine.

**Deliverable:** a baseline prediction report separating flow predictability,
fill predictability, price movement conditional on flow/fill, and severe-loss
calibration. Compare against a mechanical edge/queue/depth/size/phase baseline.
This identifies whether the missing skill is flow, price conditional on flow,
or the execution mapping, before changing the trading policy.

## 4. Stage C — fit one small probabilistic challenger

Use LightGBM with the existing pinned structural settings as the first learned
baseline. Change objectives only to match the declared distributions; record
those changes and the full recipe. Start pooled with side-aligned inputs and a
side indicator; report calibration separately for bids and asks. Separate
models per side are a later, counted candidate if development evidence warrants.

A practical factorization is:

`P(flow | X) * P(price/book path | flow, X)`.

The first component models no-flow mass plus direction, size, reach and timing.
The second models a conditional distribution, including large adverse moves,
rather than only a mean trend. In fitting it, realized future flow is a target
conditioning variable. At inference we must integrate over *forecast* flow
scenarios; actual future flow must never be supplied as a known feature.

Use small joint bins/conditional probability heads and training-only empirical
within-bin paths for the pilot. Preserve bid/ask, time and depth dependence when
sampling: independent per-rung volume draws would let one trade fill us several
times. Sparse-tail probabilities and uncertainty must be reported. A calibrated
tail probability alone is insufficient; retain severity and the raw dollar mean.
Quantiles are diagnostics, not interchangeable with expected value.

Validate probability calibration and distribution scores, volume errors, joint
tail frequency/severity, and predicted versus realized quantity-weighted value.
Add a diagnostic ablation that breaks flow/price dependence while preserving
their marginals. It tests whether joint modeling helps; it is not another
unrecorded policy search. Check incremental skill beyond today's edge and queue,
so the model cannot pass just by rediscovering a mechanically derived target.

**Gate:** the challenger adds chronological predictive value over Stage B and
the mechanical baseline, with acceptable calibration by side and phase under
predeclared tolerances. Choose tolerances from training/calibration only and
commit them before reading the relevant evaluation outcomes. If it predicts
fills but not their value, report that failure and retain the simpler baseline.

## 5. Stage D — translate scenarios into comparable actions

Reuse the corrected chronological queue/fill engine. Apply each future scenario
to the whole ladder once, conserving shared public volume and each order's
remaining quantity. Compare actions under the same scenario and clocks.
Replay current queue assumptions and conservative alternatives; queue position
is an uncertain estimate, not observed priority. Exogenous public flow is a
small-maker approximation: bigger displayed orders can affect other traders.
Do not extrapolate a size benefit beyond the approximation's support.

Begin at the current fixed ladder and size to isolate the decision change:

| State | Comparison |
|---|---|
| Existing order | Keep versus cancel, with unavoidable fills before cancellation landing in both branches |
| Empty slot | Post versus wait, accounting for placement delay and rejection |
| Existing quote to move | Retain versus cancel-and-replace, accounting for new queue position and downtime |

Use one valuation function for all branches. A finite-horizon action value is
expected portfolio wealth at the common endpoint, plus any explicitly modeled
continuation value of remaining orders, minus the starting wealth and a declared
risk penalty. Wealth comes from the canonical cash/holdings/pending-exposure
ledger. Costs recorded in cash must not be subtracted again.

All branches use the same declared continuation policy after the first action;
freeze its exact rules before building action labels. Start with a short,
transparent continuation, not an unbounded learned future policy. Orders left
at the horizon require an explicit queue-continuation approximation or a stated
truncation. Compare against zero continuation and a longer horizon to expose
dependence on that approximation; queue value is not automatically positive.
Do not add a second arbitrary queue-loss fee if it is already in branch values.

Keep local fill markouts as diagnostics, not terms added on top of portfolio
P&L. Cancellation cannot erase previously accumulated inventory. Couple both
sides through inventory, collateral and tail-risk limits. Quote changes can
favor reducing exposure without requiring both sides to have identical behavior.

Keep an existing order unless an alternative has a sufficiently supported
incremental advantage or a hard constraint binds. Post when its advantage over
waiting clears the declared threshold. Calibrate uncertainty/switching margins
on calibration data; do not impose a fixed retry or minimum-rest timer. A 50 ms
review remains a comparison control, distinct from any forecast horizon.

Record predicted components, uncertainty, winning action and reason, pending
states, post-to-first-review changes, queue/age, and per-side absence duration.
This makes a recurrence of the current entry/keep disagreement diagnosable.

Only after fixed-size controls pass, expand prices and sizes in a small declared
candidate set, including a larger-size arm. Keep total capital/worst-case exposure
comparable and show any unavoidable differences. Orders cannot be edited in
place: a changed quote requires removal and a new order under the venue's
[documented mechanics](https://dev.predict.fun/trading/market-making). Verify
historical behavior rather than silently importing current documentation into
old replay data.

## 6. Stage E — distinguish better prediction from better market making

Run continuous five-minute replays; preserve orders and inventory across minute
boundaries. Primary control sequence:

1. Reproduce frozen STOP_EV and KEEP_ALL in their original configurations.
2. Use the new controller with the empirical distribution baseline.
3. Use the same controller, constraints and action set with the learned joint
   distribution. This isolates the predictor's incremental contribution.
4. Add larger size/price choices only as a separate, explicitly counted study.

The current same-clock comparator is placement 58 ms, cancellation 37 ms,
notification 100 ms and review 50 ms. Preserve feed timing separately. M0/M50/M250
are trade-anchor uncertainty assumptions, not alternative network latencies.
Do not bundle a model change with faster clocks or a looser execution model.

Report whole-portfolio terminal marked value and settlement value, inventory
trajectory, capital usage, tail losses, and value per quoted time as well as per
filled share. Also report initial edge, 1s/5s markouts, volume, age, cancellations,
reposts and zero/one/two-sided quote availability. Higher uptime alone is not a
success criterion. Explain whether gains come from selection or simply less
exposure, using action/exposure-matched controls with declared nulls (at least
200 draws), not arbitrary withholding comparisons.

Apply FINDINGS C1–C18: all-source whole-portfolio results at M0/M50/M250; queue
M/L; age and cancel-pending attribution; without-young-fill and B1a-weighted
diagnostics; source/phase/side breakdowns; original frozen engine comparator;
causal decision-invariance controls. Subset accounting is not a counterfactual
replay. Top-maker fills remain a descriptive external reference, not verified
account profits or a causal policy comparator.

Development is chronological with purged overlapping outcome windows: earlier
training, next calibration, later evaluation. Consumed days remain development.
Do not open the protected Nov4–6 or Nov9–13 blocks. Before a forward evaluation,
commit the entire pipeline, date manifest, exact action set, objectives,
thresholds, nulls, acceptance criteria and candidate count. Validation requires
at least five later untouched complete UTC days; uncertainty is clustered by day.
No fixed profitability threshold is selected from a test result. Genuine model
improvement and positive net strategy value are separate gates; label gross
results clearly when fees/rewards/settlement failures are not established.

## 7. Build order, artifacts and stop conditions

| Order | Proposed artifact | Completion condition |
|---|---|---|
| A | `pf_flow_price_data.py` + schema/status/coverage receipt | Causal features and conserved labels reconcile, with positive/refusal controls |
| B | `pf_flow_price_baseline.py` + empirical forecast report | Joint distributions and support are visible; mechanical comparison complete |
| C | `pf_flow_price_model.py` + reloadable artifacts | Chronological incremental skill and calibration evaluated under frozen criteria |
| D | `pf_flow_price_policy.py` + decision trace | Paired actions share clocks/continuation; quantity, capital and queue accounting pass |
| E | `pf_flow_price_replay.py` + independent audit/report | Full-contract economics and realism sensitivities complete |

These are proposed files, not existing implementations. Reuse
`pf_stop_ev_accounting.py`, `pf_queue_activation.py`, the corrected ladder lane,
`pf_stop_ev_minutes_*` valuation/audit code, the feature representation work and
the presence diagnostic. Preserve old freezes; issue versioned receipts for
corrections. New large datasets stay under ignored `data/predict_fun/research/`.
Use the Codex slot via `run_light.sh`, one numerical worker, bounded memory;
estimate runtime from a small pilot before scheduling the full build.

If source validity fails, repair the labels before fitting. If joint prediction
fails, improve inputs/coverage or retain a simpler model. If predictions pass
but action outcomes fail, inspect execution/continuation and inventory control.
If a gain depends on optimistic timing or queue assumptions, it remains a
qualified simulation result. Do not advance just because a more elaborate model
raises quote availability.

The structural inspiration is a state-dependent order-flow model such as the
[queue-reactive framework](https://arxiv.org/abs/1312.0563), adapted to the actual
observability of this dataset. Its publication does not establish effectiveness
on Predict.fun. The immediate objective is to discover whether current market
state predicts **both executable quantity and its associated economic value**.
