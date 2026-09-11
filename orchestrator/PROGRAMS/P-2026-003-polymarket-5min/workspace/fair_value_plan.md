# Fair value plan — v1.2

**Status: REVISED DRAFT FOR USER FREEZE. Nothing under this document has been
fitted, scored or run.** Revised 2026-09-11 after a second code-and-contract
review. This plan does not alter the completed cancellation forward test or any
frozen Route-A result.

## 0. Supersession and ownership

Upon user freeze, this document supersedes the unscored
`live/pm_research/plans/PHASE2B_CHALLENGER_PROTOCOL_DRAFT.md` as the scoring,
freeze and promotion plan for the fair-value lane. That draft remains provenance.
The typed record and ownership fence in
`live/pm_research/plans/LANE2_FAIR_PRICE_SUCCESSOR_INTERFACE.md` remain binding:

- `Identity` is always available as the mandatory baseline and fallback;
- fair value estimates unconditional `E[Y | state]`;
- harmful-flow models may estimate only the fill-conditional residual relative
  to that anchor;
- models emit estimates; the policy layer decides quotes.

The frozen `SIGMA_ROUTE_A_PROTOCOL.md` and
`SIGMA_ROUTE_A_V2_PROTOCOL.md` are not amended here. Their last committed result
has one OOS day and remains `PRICING HOLD`. They may be rerun unchanged or later
support a separately declared successor, but neither is silently treated as a
pricing-ready input in this race.

## 1. Question and isolation

The cancellation experiment could not answer whether the quoting anchor was
wrong because candidate and baseline used the same anchor. This plan asks two
ordered questions:

1. Does a fixed fair-value challenger predict the settled binary better than
   the PM executable-book midpoint, `Identity`?
2. If it does, does replacing only that anchor improve settlement P&L and
   reduce adverse selection under the same quoter?

The economic comparison disables every harmful-flow/cancel-prediction overlay
in both legs. Both legs retain the same frozen skew, sizing, position limits,
ordinary quote-maintenance lifecycle, tick handling and latency. The only
experimental input is `FairPrice.value`. A later fair-value × harmful-flow
composition is a new experiment and a new multiplicity family.

## 2. Settlement target and observable boundary

For one five-minute market `[t0,T]`, let `X60(u)` be the value published by the
Chainlink 60-second-TWAP relay at boundary `u`. The working convention is

    Y_UP = 1{ X60(T) >= X60(t0) }

with ties resolving UP. The official closed-market winner in
`resolutions.jsonl` is the label. The Chainlink reconstruction verifies that
label; it does not replace an absent official resolution.

The endpoint convention has reproduced the official winner at the registered
gates, but a small residual has existed near stale boundaries. Before scoring:

- a window is label-admissible only when the official resolution exists and
  the margin-aware boundary checker reads `VERIFIED_AGREE`;
- `NOT_IN_CAPTURE`, missing resolution, outage and stale-boundary cases are
  counted statuses, never inferred labels;
- any `VERIFIED_DISAGREE` places the lane on hold until a superseding receipt
  explains it.

`X60(t0)` is fixed in the world at the window open, but may not yet be locally
known: the relay has measured delivery lag. A challenger may consume the strike
only after its local-knowledge timestamp. Before then it emits `NOT_READY` and
the policy falls back to Identity.

### What the recorded streams do and do not reveal

`crypto_prices_twap_sixty` is the rolling 60-second statistic itself, not the
raw Chainlink aggregate price process inside that average. `crypto_prices` on
the same subscription is a Binance spot mirror, not the settlement source.
Therefore the recorded data does **not** expose an exact, model-free realized
integral of the underlying Chainlink path inside `[T-60,T]`.

Consequences:

- `X60(t0)`, current `X60(t)` and final `X60(T)` are directly observed when the
  relay is fresh;
- any Binance integral used inside the terminal minute is explicitly a
  cross-venue proxy, not a “settlement-feed partial”;
- settlement mechanics determine the target but do not determine its
  probability; a probability requires a declared model and volatility input;
- integrating the rolling `X60` stream and calling it the underlying partial
  would double-smooth the process and is a known-bad construction.

Scope is **BTC and ETH together**. The primary statistical unit is the combined
BTC+ETH UTC day. Per-coin tables are diagnostics and cannot authorize a
coin-specific deployment. A coin-specific decision requires a new family and
new multiplicity arithmetic.

## 3. Existing assets and actual blockers

Already built, but not yet authorized for this race:

- `Identity`: midpoint of an admissible PM best bid/ask;
- `pm_microprice`: size-weighted PM microprice with Identity's exact
  admissibility population;
- `bn_bookticker_s60_probability`: a build-only driftless-GBM,
  moment-matched-average transformation using Chainlink `X60(t0)` as reference
  and Binance bookTicker as the cross-venue path proxy;
- the typed `FairPrice` record with source-event time, local-knowledge time,
  freshness, estimator identity and status.

Still missing:

1. a fixed point-in-time sigma producer for the Binance transformation;
2. a wrapper that converts its output into the typed `FairPrice` record without
   collapsing source time and local-knowledge time;
3. a policy seam that consumes the value rather than merely writing
   `fairprice_estimator` metadata;
4. a scorer and a full replay comparison against Identity.

The current trajectory exporter explicitly refuses the fair-price composition.
An unused schema field is not a partial implementation.

## 4. Closed candidate family

`Identity` is the comparator, not a candidate. Exactly two challenger identities
are declared. No feature, calibration layer, alternate lookback or third
estimator may be added after a labelled score is read.

### C1 — `pm_microprice`

The already-built size-weighted PM microprice. It consumes the same book event,
timestamps, depth threshold and admissibility decision as Identity and differs
only in the probability value. It has no fitted parameter.

### C2 — `bn_bookticker_mid`

The existing ratified estimator identifier is retained; its candidate identity
also binds `model_version = s60_probability_v1` and the builder digest. It is a
structural cross-venue candidate with these inputs fixed before scoring:

- reference: Chainlink `X60(t0)`, consumed only after local receipt;
- spot/path proxy: matching Binance USDM bookTicker midpoint;
- model: driftless GBM with the existing moment-matched lognormal average;
- terminal partial: Binance midpoint integral over `[T-60,t]`, explicitly
  labelled as a proxy for the unobserved Chainlink aggregate path;
- sigma: trailing 30-minute realized volatility of one-second Binance midpoint
  log returns, shifted by one complete observation, computed as
  `sqrt(mean(r_1s^2))` in per-square-root-second units with no annualisation;
- each one-second grid value is the latest midpoint whose local-knowledge time
  is at or before that grid instant; no interpolation or later tick may fill it;
- sigma admissibility: at least 90% of the 1,800 expected one-second returns,
  no source gap above five seconds, finite positive result, and
  `sigma_local_knowledge_ns <= decision_recv_ns`;
- no basis correction, drift fit, probability calibration or caller-supplied
  sigma fallback.

If any required input is unavailable, stale, pre-era or malformed, C2 emits a
typed non-OK status. The estimator never substitutes Identity itself; the policy
wrapper performs the declared fallback and counts it.

The sub-second Binance era floor is applied per event:
`recv_ns >= 1787579334881534478` (2026-08-24 13:48:54 UTC). Earlier rows are not
admitted to C2.

Both candidates emit one UP probability. DOWN is mechanically `1 - p_UP`; it is
never fitted independently. A complement check, token/outcome identity check and
UP/DOWN sign-flip falsifier are mandatory.

The forward multiplicity remains **m = 2** even if one candidate fails a
development or availability gate. This conservatively accounts for the closed
family that was inspected. Historical cancellation candidates are reported as
programme provenance but are not members of this fair-value estimand's family.

## 5. Build gates — no labelled score

Before any real outcome score is computed, land one committed build containing:

1. **Settlement verifier:** official winner join, margin-aware Chainlink
   agreement status, gap/outage statuses and a head-resolving supersession rule.
2. **Sigma producer:** the exact C2 contract above, with scale, minimum-count,
   gap, zero-volatility, stale-input, pre-era and future-knowledge falsifiers.
3. **Estimator wrapper:** a valid `FairPrice` for C1/C2; source-event and
   local-knowledge timestamps remain distinct at every hop.
4. **Canonical forecast-action builder:** one row per actual fair-value
   consumption decision on the neutral Identity reference path. The key is
   `(coin, slug, generation_id, decision_recv_ns)`. If multiple quote sides
   consume the same UP probability at the same generation and timestamp, they
   remain one forecast action. Duplicate keys refuse the build.
5. **Policy seam:** Identity substitution yields bit-identical quotes and
   trajectories; a non-Identity value changes the quote anchor in a positive
   control; absent challenger falls back to Identity.
6. **Replay seam:** baseline and challenger share all non-fair-value parameters,
   input snapshot and initial state, while preserving their own resulting order
   paths.

Required two-way falsifiers:

- a legitimately point-in-time record passes; a future-knowledge record refuses;
- a fresh record passes; a stale/missing record emits its exact status;
- Identity versus itself has exactly zero predictive and economic increment;
- a deliberately inverted settlement/token mapping is detected;
- an exact Chainlink endpoint label passes; a rolling-TWAP-as-raw-integral
  fixture refuses;
- a synthetic informative probability improves log loss; a known-bad constant
  cannot receive a positive verdict;
- changing only `fairprice_estimator` metadata while leaving the consumed value
  unchanged is detected as a decorative seam.

## 6. Development use of consumed days

Every day ending before the full-pipeline freeze is development/consumed. The
08-24..27 settlement split and 09-03..09-09 P-003 populations are explicitly
consumed. They may be used for performance profiling, reconciliation and a
clearly labelled descriptive smoke, but never for a validation interval or
headline claim.

Neither candidate is fitted to settlement outcomes, so there is no random-row
or leave-one-day-out training. If a future candidate introduces fitted
parameters, it is outside this closed family and must use strictly-forward UTC
day folds; a past day may never train on a future day.

### Common predictive population

At every canonical Identity forecast action:

- read Identity and challenger strictly as of `decision_recv_ns`;
- score the UP probability once, with DOWN carried only as the complement;
- clip Identity and challenger by the same fixed `epsilon = 1e-6` for log loss;
- if a challenger is non-OK, score the policy's Identity fallback on that action
  and count the native challenger status;
- if Identity is non-OK, count the action but score neither side.

This fallback score on the full Identity-eligible action universe is primary. A
native-OK intersection score is diagnostic. Thus a challenger cannot improve by
being absent on difficult actions.

The descriptive receipt reports UTC days, actions, unique markets, native
coverage, fallback counts, every exclusion status, per-source freshness and an
as-of timestamp. No development sign may change C1, C2, their sigma contract,
the quote mapping or m=2.

## 7. Predictive statistic and full freeze

Primary loss is natural-log loss against the official settled outcome. Brier
loss, reliability curves, time-to-expiry cells, per-coin cells and equal-window
weighting are diagnostics only.

For candidate `c` and complete UTC day `g`:

    LL_g(estimator) = mean over canonical forecast actions
    delta_LL_g(c)   = LL_g(Identity) - LL_g(c)

Positive `delta_LL_g` is better. The BTC+ETH portfolio-day mean gives each coin
equal weight; action counts remain reported. No action, fill, market or coin is
treated as an independent day.

Before validation, freeze as one commit:

    immutable inputs -> labels/statuses -> actions -> sigma -> FairPrice
                     -> fallback -> score -> quote mapping -> replay -> P&L

The declaration records all file hashes, commit ref, candidate count, action
key, epsilon, status grammar, source manifests, initial inventory, tick rounding,
latency, fee rule, quote parameters, null and success predicates. Every verdict
is computed from artifact fields; no prose-only pass is permitted.

The placement parameter is the current settlement-replay design choice:
`placement_latency_ms = 250` for every new generation in both legs. It is a
simulation assumption, **not a measured live end-to-end latency**. The receipt
must describe that boundary exactly, count every fill removed before
`generation_start + 250 ms`, and never describe the parameter as empirical.
No challenger receives an instantaneous first placement. Harmful-flow
cancellation remains disabled; ordinary quote replacement follows one shared,
separately declared lifecycle in both legs.

### Quote mapping frozen before predictive validation

The existing reference quoter is run with harmful-flow cancellation disabled.
For each candidate, replace only its Identity anchor with the latest admissible
candidate value; on non-OK status use Identity. Preserve the same skew, spread
rule, size, position cap, quote-maintenance rules and placement/cancel latency.

- UP uses `p`; DOWN uses `1-p`.
- Bid rounding is downward and ask rounding upward to the legal tick.
- Prices are bounded to the legal binary range. A candidate quote that would
  cross and take liquidity emits the existing `PLACE_WITHHELD` event with reason
  `MARKETABLE_CROSS`; it is never silently clamped into an apparently passive
  order.
- A candidate-induced quote change follows the same cancel/replace lifecycle as
  any other anchor change; it receives no zero-latency privilege.

The exact mapping and all numeric parameters are frozen now. Predictive results
cannot be used to tune the later economic policy.

## 8. Prospective predictive validation

Validation starts with the first complete UTC day strictly after the full
pipeline freeze. The accrual rule is fixed before that day:

- observe the first **14 consecutive calendar days**;
- require the first **10 evaluable complete BTC+ETH UTC days** within that band;
- days that fail a predeclared data gate remain counted with statuses;
- if fewer than 10 are evaluable by day 14, verdict is
  `INSUFFICIENT_EVIDENCE`; do not extend opportunistically.

Day eligibility is resolved by candidate-blind inputs only: the frozen day/book
gate, official resolutions and settlement-verification coverage. Challenger
availability, score, fills or P&L can never remove a day or replace it with a
later one.

For each candidate, native-OK coverage must be at least 95% of Identity-eligible
actions for each coin over the validation population. The primary policy score
still includes Identity fallback; the coverage gate prevents a nominal winner
that contributes almost no independent estimate.

The primary null is `median_g(delta_LL_g) = 0`. Use the exact two-sided paired
day sign test and enumerate all `2^G` day-sign assignments. At G=10 this is 1,024
assignments, above the 200-null minimum. The smallest two-sided p-value is
`2/2^10 = 0.001953125`; with m=2 the multiplicity-adjusted minimum remains
attainable. Holm correction is across C1 and C2.

An exactly zero daily increment is a reported tie and is excluded from the sign
count, never silently assigned a favourable sign. At least eight nonzero
portfolio-day increments are required for the exact test (`2^8 = 256`); fewer
than eight yields `INSUFFICIENT_EVIDENCE`. Mean and median summaries still use
all ten evaluable days, including zeros.

A candidate passes predictive validation only if all are true:

1. Holm-corrected p < 0.05 on the primary log-loss increment;
2. mean and median `delta_LL_g` are positive;
3. the 95% native-coverage gate passes for BTC and ETH;
4. all population, timestamp, complement and reconciliation predicates pass.

Brier or a favourable coin/time cell cannot rescue a failed primary. All ten
days become consumed regardless of verdict.

## 9. Economic validation on a new clock

Only predictive winners proceed, without refitting, recalibration or quote-map
changes. Economic validation begins on the next untouched complete UTC day and
uses the same 14-calendar-day / 10-evaluable-day accrual rule. Economic
multiplicity remains m=2 even if only one candidate reaches this stage.

### Comparator and endpoint

The comparator is the Identity-anchored reference quoter rerun on the identical
external snapshot and starting state. Candidate and baseline retain their own
resulting order paths; changing fair value legitimately changes quotes, queue
position, fills and inventory.

For candidate `c` and portfolio day `g`:

    PnL_g = trade cash flow
            + remaining inventory valued at the official settlement outcome

    delta_PnL_g(c) = PnL_g(candidate) - PnL_g(Identity)

All orders become effective only after the frozen 250 ms simulated placement
latency. P&L includes every fill at its own time and price, residual settlement,
quote replacement and queue effects, and the verified maker fee applicable to
these markets. A zero fee is used only if the receipt identifies the supporting
market/account rule.

### Adverse-selection mechanism check

For each fill with signed position change `dq`, fill price `q` and that token's
binary settlement `Y`:

    settlement_edge = dq * (Y - q)

Aggregate this over fills and divide by total absolute filled shares. Report
fill count, filled shares, quote-active time, ending inventory and the cash and
settlement legs separately. The candidate must improve both portfolio P&L and
settlement edge per filled share; lower activity alone is reported as the
mechanism, never described as better prediction.

A day with zero absolute filled shares in either leg has no per-share edge. It
is `NOT_EVALUABLE_FOR_EDGE`, never assigned edge zero, and remains visible in
the calendar-day accrual ledger. It does not permit replacing that day; fewer
than eight pair-comparable edge days inside the fixed ten-day population yields
`INSUFFICIENT_EVIDENCE` for the mechanism gate.

Use exact two-sided UTC-day sign tests for `delta_PnL_g` and the settlement-edge
increment, with Holm correction across candidates. Each test requires at least
eight nonzero, evaluable portfolio-day increments; otherwise it returns
`INSUFFICIENT_EVIDENCE`. Both gates are required, so one metric cannot rescue
the other. Intervals, if emitted, resample UTC days only.

A candidate is adopted only if all are true:

1. Holm-corrected p < 0.05 and positive mean/median daily P&L increment;
2. Holm-corrected p < 0.05 and positive mean/median settlement-edge increment;
3. all reconciliation, latency, fee, inventory and status predicates pass;
4. the Identity-vs-Identity control remains exactly zero end to end.

Old cancellation or vanilla artifacts remain provenance, not comparators for
this new reference path.

## 10. Kill criteria

- Any unexplained `VERIFIED_DISAGREE` settlement label: **hold**.
- Sigma producer, typed timestamp seam or source identities cannot be made
  point-in-time: **stop C2** with an explicit unavailable status.
- Candidate is changed after a labelled score: **void that candidate** and start
  a new family/clock; never amend it in place.
- Predictive validation fails: **stop that candidate**; do not inspect economic
  performance as a rescue.
- Predictive validation passes but economic P&L or settlement-edge validation
  fails: report **predictively better, not economically established**.
- Any result requires sub-latency information, silent fallback, population
  shrinkage, independent DOWN fitting or a rolling-TWAP-as-raw-path integral:
  **refuse the result**.

## 11. Implementation order

1. Freeze the settlement-label/status reader.
2. Build and falsify the 30-minute sigma producer.
3. Wrap C1/C2 in typed `FairPrice` records.
4. Build canonical actions and the paired scorer.
5. Wire fair value into the quoter and prove Identity parity.
6. Freeze the full pipeline and both candidate identities.
7. Run ten-day predictive validation.
8. Only for predictive winners, start the new ten-day economic clock.

Consumed-day smoke runs may share existing archives but must not starve an
active heavy replay. No fair-value score is evidence before step 6.
