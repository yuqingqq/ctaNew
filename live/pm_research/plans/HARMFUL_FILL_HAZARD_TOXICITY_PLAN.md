# Harmful-fill hazard × toxicity optimization plan

**Status:** DRAFT / RECORDED BEFORE IMPLEMENTATION / NOT FROZEN  
**Scope:** offline research only; no venue adapter, live cancellation path, or
execution server  
**Incentives:** maker rebates and liquidity rewards excluded from the primary
economics, by current user direction

## 1. Decision and motivation

The next optimization should not be another generic adverse-price forecast.
It should estimate the expected economic harm of leaving each currently
resting side exposed.

The corrected five-second concentration result motivates a selective policy:

| population | BTC fills required to carry the operative drift share | ETH fills required |
|---|---:|---:|
| real-maker tape, cash-ranked | 1.60% | 4.38% |
| fixed-five-share `JOIN_BBO` replay | **7.47%** | **13.46%** |

The replay population is the strategy-relevant one. Its worst 10% of fills
carry 53.2% of BTC and 52.9% of ETH total negative five-second drift. The
required tail is stable across the three sampled days: BTC
7.24%/7.66%/7.58%; ETH 13.71%/13.99%/12.81%.

These are perfect-foresight ceilings, not model results. They establish that
the harm is selective enough for a gate to be possible. They do not establish
that the tail is predictable, that every tail event is an informed sweep, or
that an implementable cancellation can arrive in time.

## 2. Target architecture

At decision time `t`, for each live order generation and each resting side,
estimate two separate objects:

1. **Fill hazard:** the probability that the same order generation receives a
   latency-preventable fill during the declared action horizon.
2. **Conditional toxicity:** the signed five-second economic value of
   cancelling, conditional on such a fill.

For side `s`, assumed cancel-effective latency `L`, and fill horizon `H`:

```text
p_fill(t,s,L,H) = P(same-generation fill in [t+L, t+H] | state at t)

V_cancel(fill)  = - maker five-second markout from the resting order level
                   × filled shares

tox(t,s)        = E[V_cancel | preventable fill, state at t]

expected_cancel_value(t,s) = p_fill(t,s,L,H) × tox(t,s)
```

Positive `V_cancel` means cancellation avoids a harmful fill; negative
`V_cancel` means cancellation sacrifices a profitable fill. This formulation
therefore prices both sides of the decision instead of labelling every fill as
something to prevent.

The prediction unit is a **decision-time quote-exposure row**, not a completed
fill. Training only on completed fills would condition on the event the policy
is trying to change and would not teach the model when an exposed order will
remain unfilled.

## 3. Dataset contract

Build rows from the unchanged queue-realistic `QR_SKEW_ONLY` reference path.
Each row represents `(decision time, order generation, side)` and records:

- whether an order is live and its remaining size;
- queue ahead, filled fraction, order age and quote level;
- inventory and whether the side increases or reduces inventory;
- whether a same-generation fill would occur after `t+L` and by `t+H`;
- filled shares and five-second markout when the horizon is observable;
- explicit exclusions for gaps, missing future mid, generation replacement,
  truncated horizon and unproven timing;
- point-in-time feature timestamps and source-receipt provenance.

No candidate policy may generate its own training population. The no-cancel
reference trajectory defines the shadow counterfactual for every candidate so
models are compared on identical opportunities.

Rows with fills before `t+L` are economically stale, not preventable. Rows
whose cancellation timing cannot be established remain unresolved and cannot
be promoted as proven preventions. A latency grid is a sensitivity analysis,
not a substitute for venue cancel acknowledgements and owned-fill timing.

## 4. Feature plan

### 4.1 PM-native event state — primary

Use event-time, side-signed features at approximately
25/50/100/250/500/1,000 ms horizons:

- aggressive buy/sell quantity and notional;
- same-direction trade run length and flow purity;
- distinct levels consumed and multi-level sweep progression;
- bid/ask depth depletion and replenishment rates;
- imbalance, imbalance change, microprice and microprice change;
- distance from the resting order to touch and recent trade-through pressure;
- spread state, spread transition and quote age;
- order-generation queue state and partial-fill state.

Changes and acceleration matter more than static book levels for a fast pull
decision. Features must be available at local knowledge time and must not use
the event that causes the labelled fill.

### 4.2 Fair-price and external-flow context — secondary

Add, without replacing PM-native flow:

- PM microprice minus the point-in-time fair-price estimate;
- fair-price change over the same multiscale horizons;
- Binance spot/perpetual signed flow, microprice displacement and burst state;
- cross-venue disagreement and convergence speed;
- remaining window time, distance from strike and settlement-risk state.

The fair-price module describes where price should move. PM flow and queue
state describe whether this particular order is about to be selected. The two
roles must remain identifiable in ablations.

### 4.3 Frequency requirement

One-second bars that exclude the current second can be roughly 250–1,250 ms
old and are not sufficient as the only fast-cancellation representation.
Build sub-second event windows from receipt-time data. The decision clock may
run faster than the feature state changes, but duplicate-state decisions must
be collapsed or weighted so they do not create fictitious sample size.

## 5. Model plan

Use a limited, predeclared model family rather than a broad hyperparameter
sweep:

1. calibrated logistic/ridge models as the linear reference;
2. fixed-capacity LightGBM classifiers for fill hazard and harmful-fill sign;
3. a fixed-capacity nonlinear conditional-value or magnitude head.

The nonlinear candidate is justified by interactions such as sweep intensity
× side × queue position × fair-price disagreement. Hyperparameters must be
pinned before forward scoring and shared across comparable cells.

The conditional-toxicity component should report both:

- `P(V_cancel > 0 | fill, x)` for tail discrimination; and
- an economically weighted magnitude estimate or value-weighted sign score.

The combined action score is evaluated in expected cents, but ranking quality
in the harmful tail is reported separately so a noisy magnitude head cannot
hide useful discrimination.

## 6. Policy composition

Inventory skew and harmful-flow protection remain separate modules:

- inventory skew determines desired exposure and quote placement;
- the harmful-flow score can override the threatened side;
- the opposite side remains live when safe;
- intermediate risk reduces size before full cancellation;
- high risk cancels every live order generation on the threatened side,
  including an inventory-skewed order;
- inventory-reducing protection remains explicit and is not silently removed;
- cancel and repost use different thresholds or a short confirmation rule to
  avoid flicker;
- repost occurs only after risk falls below the lower threshold and ordinary
  placement rules again permit the quote.

The economic decision rule is:

```text
cancel when expected avoided harm
            > lost spread capture + queue-reset/repost cost.
```

Because queue-reset cost is not yet directly measured, report results both
before it and across a declared sensitivity grid. Do not call a gross result
net-profitable.

## 7. Latency analysis

Evaluate assumed cancel-effective latencies of
`5, 10, 20, 30, 50, 75, 100, 150, 250 ms`. The grid answers sensitivity and
break-even questions. It cannot prove preventability from public market data.

For each latency report:

- fills already executed before the decision;
- fills received before assumed cancel effectiveness;
- fills definitely stale by event-time evidence;
- unresolved fills;
- economically preventable fills under the simulation;
- loss capture, profitable-fill sacrifice and cancel/repost traffic.

Real cancel-send, venue ACK/reject/effective timestamps and owned-fill receipts
remain necessary for causal validation. Those execution measurements belong
outside this research-only repository if they require a live adapter.

## 8. Evaluation and gates

Global ROC AUC, Brier score and average regression error are diagnostics, not
the primary economic verdict for a concentrated-tail problem.

Report at cancellation budgets of 5%, 10% and 15%, plus the unchanged q>0.5
action when applicable:

- share of total negative drift captured;
- precision and recall for economically harmful fills;
- PR AUC and lift over base rate;
- profitable spread capture sacrificed;
- fill and share retention;
- adverse-cost/spread-capture ratio `rho = A/S`;
- gross cancel value per decision and per retained fill;
- inventory, effective cancellations, reposts and queue resets;
- per-day results, not only pooled results.

Every scored policy must be compared with random abstention matched on the
same cancellation or fill-retention fraction. Merely quoting less is not
selection skill. A useful selection model must reduce `rho` relative to that
matched control; strategy viability requires reaching `rho < 1` at material
retention after declared costs.

The strategy baseline remains `QR_CANCEL_HOLD_X_SKEW`, with
`QR_SKEW_ONLY` mandatory. No candidate is promotable unless it improves the
baseline on independent days without worsening the declared inventory and
traffic limits.

## 9. Validation discipline

- Existing 2026-08-20/21/22 data is training context.
- Existing 2026-08-23/24 data is seen development context.
- Do not tune features, thresholds, horizons or model capacity on new forward
  outcomes after the candidate is frozen.
- Admit only complete UTC days whose earliest required receipts postdate the
  candidate freeze.
- Treat UTC day as the primary independence cluster; window-level intervals
  are descriptive when the day count is inadequate.
- Account for every candidate in the forward race when constructing nulls and
  interpreting a clearing result.
- Preserve exact candidate builder code, dependency hashes, feature schema,
  split receipt and candidate hash before the first forward score.

More forward data is necessary to test generalization. More data from the same
seen days does not repair model selection, timing ambiguity or an invalid
freeze.

## 10. Implementation order

1. **Repair provenance first.** The current v4 sweep artifact is not accepted
   as a durable candidate: its builder is uncommitted, absent from its named
   commit, and does not contain the complete data/target/fit/artifact pipeline.
   Either reconstruct and freeze a complete reproducible builder or mark that
   artifact void.
2. Build and self-test the side-specific decision/exposure dataset.
3. Add receipt-time multiscale PM flow/depletion features and feature-as-of
   audits.
4. Fit the linear hazard × toxicity reference.
5. Fit the fixed nonlinear candidate and run feature-family ablations.
6. Evaluate loss capture and matched-random controls before any policy replay.
7. Integrate only a model that passes those gates into cancel × skew replay.
8. Run the latency and queue-reset-cost grids.
9. Freeze the surviving candidate and score it unchanged on complete forward
   days.
10. Measure real execution latency separately before making any claim about
    actually preventable fills.

## 11. Expected outcome and stopping rule

The most likely performance gain comes from the decision-time target and
PM-native event features, not from nonlinear capacity alone. The nonlinear
model is useful only if it identifies the harmful 7–15% tail substantially
better than matched random cancellation.

Stop the route if, after complete forward day clusters, the best frozen model
cannot improve `rho` over matched random abstention or cannot approach
break-even at a material quote-retention level. A negative result closes this
feature/data design; it does not license continued tuning on the same forward
days.
