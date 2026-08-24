# BE-FlowAndFills adverse-movement extension

Status: **DESIGN / NOT DECISION-ELIGIBLE**

Scope: research only. This plan adds no venue adapter, order sender, cancellation
command or live execution path. `contracts/contracts.yaml` v24 is authoritative
for types and module ownership.

## 1. Placement and responsibility

Adverse-movement prediction is a capability inside `BE-FlowAndFills`, not a new
belief or decision module:

~~~text
DA-StateView                         knowledge-time Binance + PM state
    -> DA-FlowActionGrid             immutable AdverseFeatureRow + exact action
    -> BE-FillFit                    offline AdverseMoveFit
    -> EV-AdverseMove               independent forward validation
    -> BE-FlowAndFills.evaluate      complete KEEP/CANCEL ActionOutcome
    -> DE-DecisionScheme             chooses an action only from full economics
~~~

The fit answers what is likely to happen if an exact resting action remains.
It never chooses `CANCEL`. DE owns that choice, and no DE consumer receives raw
features or a bare toxicity score.

## 2. Estimand

For candidate action `a`, maker side `s`, state known at `t`, cancellation
latency `L`, prediction horizon `Delta`, and markout horizon `h`, estimate:

~~~text
P(toxic fill if a is kept | x_t, a)
E[maker-signed markout_h | fill, x_t, a]
E[(-markout_h)+ * prevented_fill_size
  * 1{counterfactual fill occurs after t + L} | x_t, a]
~~~

The last line is gross adverse damage potentially avoidable by cancellation.
It is non-negative and does not pretend that the canceled fill's spread,
rebate, or queue value is preserved. Those terms belong to `ActionOutcome` and
`WealthLedger` when KEEP and CANCEL are compared.

This is a joint action/fill/markout target. An unconditional direction forecast
multiplied by an independent fill probability is not admissible: fills are
endogenous and are most likely in the states whose movement is adverse.

## 3. Inputs

All inputs are admitted through `StateView` on local receipt/knowledge time.
`DA-FlowActionGrid` freezes them into `AdverseFeatureRow` objects tied to the
exact shadow action; unavailable rows are retained. The direct-event candidate
uses the existing Binance futures WebSocket capture:

- `bookTicker`: event-driven best bid/ask and sizes;
- `trade`: raw matched trades and aggressor direction;
- `depth20@100ms`: top-20 depth snapshots;
- Polymarket price changes, trades, displayed depth and the exact resting action;
- time remaining, moneyness, tick, spread and input staleness;
- the measured end-to-end cancellation-latency profile.

Initial feature families are multi-scale Binance returns, signed aggressive
volume, burst intensity, microprice/book imbalance, depth depletion, PM queue
and imbalance state, and Binance-implied-versus-PM price divergence. Exact
windows and transformations must be frozen in `feature_schema_hash` before a
scored period. The identical builder hash is required offline and at decision
time. No feature may include the event that reveals or causes the fill being
predicted.

The deployed 1 Hz relay may produce a labelled diagnostic fit with
`feed_class=SAMPLED_RELAY`; it cannot validate the direct-event-WS question.

## 4. Candidate and output

Start with an interpretable regularized logistic model for the joint toxic-fill
probability and a regularized conditional-markout model. A more complex learner
is a separate frozen candidate, not an after-the-fact replacement.

Every estimate binds:

- the exact action hash, maker side, level, size and state;
- `as_of`, prediction horizon and markout horizon;
- the model and source-profile hashes;
- the measured cancellation-latency profile;
- maximum feature knowledge time and input staleness.

If the model, source schema, latency profile or admitted state does not match,
`BE-FlowAndFills` returns `Unavailable` rather than extrapolating.

## 5. Evaluation and promotion

`EV-AdverseMove` evaluates a frozen candidate on forward UTC days. It reports:

- toxic-fill Brier score and calibration error;
- gross avoidable adverse damage;
- KEEP-versus-CANCEL PnL delta under the frozen policy and economic schedule;
- per-fill and share-weighted results, with a day-clustered interval;
- coverage, refusals, source staleness and the latency profile used.

Classification accuracy is diagnostic. Promotion requires the frozen
KEEP-versus-CANCEL replay to improve net share-weighted value after lost spread,
rebate, queue position and actuation latency, with its pre-registered uncertainty
bar satisfied. The existing ten-forward-day floor applies unless a separately
reviewed protocol imposes a stricter requirement.

Artifact states remain monotone:

~~~text
DEVELOPMENT -> CANDIDATE_FROZEN -> VALIDATED
                                  | INSUFFICIENT_EVIDENCE
                                  | MODEL_REFUTED
~~~

Until `AdverseMoveFit.status == VALIDATED` and its matching evaluation verdict
is `PASS`, the decision-facing capability is unavailable. Adding this structure
therefore does not reopen the current passive-MM or cancellation verdict; it
defines the evidence needed to test the remaining direct-WS route.

## 6. Pre-build review and development candidate — 2026-08-24

The architecture and ownership above survive review. Four details were not
specific enough to implement without silently choosing the answer, so the first
build pins them here. This is a **development protocol**, not a promotion
protocol, because the available 2026-08-20..24 tape was already visible when
these choices were made.

1. **There is no measured end-to-end cancellation profile yet.** The real
   `DE-Actuator` is deliberately unbuilt and venue acknowledgement is
   unobserved. Development evaluation therefore uses the declared
   counterfactual ladder `{0, 150, 250, 350, 500} ms`. Every receipt labels it
   `ASSUMED_COUNTERFACTUAL`; no rung may populate a `VALIDATED` artifact or be
   called `tau_operative`.
2. **The decision clock is fixed, not outcome-selected.** Rows are generated on
   a 100 ms knowledge-time grid. Selecting only times before fills, or only large
   Binance moves, is forbidden because either conditions the sample on the
   outcome. Each row binds a 5-share JOIN-at-touch action and the conservative
   displayed-back queue bound.
3. **The target is now exact.** Prediction horizon is 1 s; markout horizon is
   5 s; a toxic fill is positive filled quantity whose share-weighted gross
   maker markout is below 0 c/share. Fill tranches before cancellation becomes
   effective remain fills. The direct avoidable-damage and signed cancel-value
   targets are fitted directly; an independently fitted fill probability is
   never multiplied by an unconditional direction forecast.
4. **Economic completeness is a gate, not an imputation.** Development cancel
   value includes the fill's full gross markout (therefore spread capture) and
   the measured 0.168 c/share maker-rebate scale. Liquidity-reward opportunity
   cost and rejoin queue value are unavailable. Results must consequently remain
   `INSUFFICIENT_EVIDENCE`, even if classification or gross cancellation value
   looks favourable.

The initial interpretable candidate is regularized logistic regression for the
joint toxic-fill event, ridge regression for fill-conditional markout, and
direct ridge regressions for avoidable damage and signed cancel value at each
latency rung. Training days are 2026-08-20/21/22; 2026-08-23/24 are a
development holdout. Before reading that holdout, the development
regularization is pinned to standardized features, logistic `C=1.0` with
natural class prevalence (no reweighting), and ridge `alpha=10.0`; these values
are not tuned on either split. Ten complete days collected strictly after a
separately frozen candidate remain mandatory for promotion.

Direct-event features are binned in completed 100 ms receipt-time buckets so a
bucket containing events after `as_of` can never leak into the row. The first
schema includes maker-signed multi-scale Binance returns and trade imbalance,
book burst counts, BBO microprice/imbalance, depth-5/depth-20 imbalance and
depletion, plus the exact PM action/queue state. It deliberately does **not**
invent a Binance-implied binary probability: that mapping is a fitted object,
not an observable feature, and would otherwise hide a second model inside the
feature builder.

## 7. Development implementation and result — 2026-08-24

The research implementation now exists in `adverse_feature_rows.py` and
`adverse_move.py`. It materializes point-in-time features separately from
future action labels, fits the four pinned model families per coin, and writes
`data/pm_5min/derived/adverse_move_development_v1.json`. Both modules are
offline readers only. They contain no order, cancel, exchange or decision port.

The first recorded-data smoke evaluation used one immutable BTC and ETH window
per UTC day: three training days and the two development-holdout days above.
There were 48,934 action rows and 48,569 admitted feature/label pairs. Features
use the 250 ms knowledge-lagged PM view; future markouts use a separate,
unlagged receipt-time state tape. This is enough to test the implementation and
expose a weak candidate, but the
overlapping 100 ms rows come from only ten windows and five days.

At the 250 ms assumed cancellation rung:

| coin | toxic Brier (train-prevalence baseline) | conditional-markout R² | direct cancel-value R² | selection gain vs train-selected constant, c/decision |
|---|---:|---:|---:|---:|
| BTC | 0.0708 (0.0809; +12.5% skill) | -0.652 | -0.265 | -0.1512 [-0.2387, -0.0656] |
| ETH | 0.0532 (0.0515; -3.4% skill) | -0.011 | -0.014 | +0.0246 [-0.0190, +0.0618] |

The value comparison is the load-bearing result. The training-only constant
rule selected `ALWAYS_CANCEL` at every latency because the unconditional
JOIN-at-touch action was adverse even after the measured rebate. The fitted
policy canceled 60–71% of BTC rows and 78–84% of ETH rows. It gained versus
the deliberately bad `KEEP_EVERY_ROW` reference, but BTC underperformed
`ALWAYS_CANCEL` at all five latency rungs. ETH had positive point estimates at
all rungs after the corrected markout clock, but 0/150/250 ms intervals span
zero. The nominal two-day intervals at 350/500 ms are positive (+0.0238 and
+0.0132 c/decision), a weak lead worth testing—not promotion evidence. Most of
the apparent cancellation benefit still comes from the passive strategy's
negative base rate rather than learned selective avoidance.

This initial linear candidate is **not freeze-ready**. BTC's classifier carries
some probability information but not action value. ETH does not beat the
probability base rate and every direct magnitude fit has negative R², despite
the slow-latency threshold hint. The structural adverse-move seam remains
useful because it makes these distinctions possible, but no decision module may
consume this fit. The receipt remains `DEVELOPMENT / INSUFFICIENT_EVIDENCE`,
artifact `3bb6c75032f411de4141915ad6d1708ef0164770daee2d17f5e0f58f71793b36`,
with zero strictly forward days, assumed rather than measured cancellation
latency, and missing liquidity-reward and rejoin-queue costs. The two-day
clustered intervals are descriptive and materially understate regime
uncertainty.

## 8. Fast-path v2 development protocol — frozen before its run

The v1 smoke test is not a test of a claimed 50 ms latency advantage: its
100 ms decision grid alone adds 0–100 ms of quantization delay, and its PM
feature view deliberately inherited the older 250 ms replay lag. The following
v2 is a separate source/action schema. V1 remains unchanged and reproducible.

V2 decisions are event-driven on local receipt timestamps from Binance
`bookTicker`/`trade` and Polymarket book/trade events. The first event after a
decision opens a fixed 10 ms cooldown; events inside that cooldown are admitted
to the next feature state but do not create duplicate decisions. There is no
artificial PM lag. Exact rolling windows use only events with
`recv_ns <= as_of_ns`; no completed 100 ms bucket is used. Binance
`depth20@100ms` remains slow context and may not itself establish a fast-path
latency claim.

The frozen fast windows are `{10, 25, 50, 100, 250, 500} ms`. Features are
maker-signed Binance mid returns, signed/absolute aggressive volume, trade
count and imbalance, book-update count, current BBO imbalance/microprice,
depth-5/depth-20 imbalance and depletion, the exact PM touch/queue state, and
four trigger-source indicators. A stale/missing current book, depth or full
500 ms history makes the row unavailable; it is never imputed.

Each feature/action row receives nested fill labels at
`{50, 100, 250, 500, 1000} ms` and a maker-signed markout five seconds after
each fill. Cancellation becomes effective at the assumed counterfactual rungs
`{10, 25, 50, 75, 100, 150, 250} ms`; only rungs strictly below a fill horizon
are evaluated. The direct target includes full markout and the measured
0.168 c/share rebate exactly as in v1. No rung is measured end-to-end or called
`tau_operative`.

Per coin and horizon, v2 keeps the pinned standardized logistic `C=1.0` joint
toxic-fill model and direct ridge `alpha=10.0` signed-cancel-value models.
Every policy is compared with the constant action selected from training data
alone. The days remain 2026-08-20/21/22 train and 2026-08-23/24 development
holdout. Those days and the v1 result are already known, so v2 can diagnose
whether faster timing is promising but cannot freeze, validate or promote a
candidate regardless of its result.

## 9. Fast-path v2 development result — 2026-08-24

The exact-event implementation is in `adverse_feature_rows_fast.py` and
`adverse_move_fast.py`. On the same one-window-per-coin-per-day development
sample it produced 231,092 feature/action rows. The receipt is
`data/pm_5min/derived/adverse_move_fast_development_v2.json`.

Toxic-fill Brier skill versus the training-prevalence forecast was:

| fill horizon | BTC | ETH |
|---:|---:|---:|
| 50 ms | +7.2% | -0.4% |
| 100 ms | +6.5% | -4.4% |
| 250 ms | +12.3% | +4.7% |
| 500 ms | +16.1% | +4.6% |
| 1,000 ms | +16.5% | +0.8% |

The timing counterfactual confirms that sub-50 ms reaction matters. At the
50 ms fill horizon, the fraction of filled shares occurring after a 10 ms
cancel-effective time was 63.9% BTC / 65.0% ETH; after 25 ms only 40.0% /
45.2% remained. At the 100 ms horizon, a 50 ms cancellation retained 47.5% /
46.8%. These are replay survival fractions, not measured cancel-ACK success.

Faster sampling did **not** rescue the first economic model. Every BTC direct
cancel-value fit had negative holdout R². At the 50 ms horizon its selection
gain over the training-selected constant was indistinguishable from zero; from
100 ms onward it was negative. ETH had narrow development hints: 50 ms fill /
25 ms cancel produced +0.00153 c/decision with a nominal two-day interval
[+0.00051, +0.00232], and the 250 ms horizon had positive point estimates whose
intervals all crossed zero. ETH's 50 ms toxic classifier did not beat base rate,
the direct fit R² was negative, and its 500/1,000 ms policies lost to the
constant baseline.

Therefore v2 answers the engineering question—exact-event 10 ms research
scoring is feasible—and sharpens the latency requirement, but it remains
`DEVELOPMENT / INSUFFICIENT_EVIDENCE`. The current linear magnitude model is
not freeze-ready, no real end-to-end latency was measured, and no decision or
execution module may consume the result.

## 10. Nonlinear v3 development protocol — frozen before its run

User direction on 2026-08-24 fixes the sequence: test model quality first,
compose with the already measured skew mechanism only if the model carries an
economic selection signal, and measure real latency afterwards. Incentives are
out of scope for this experiment. In particular, v3 excludes both liquidity
rewards and the 0.168 c/share maker rebate from its target. Its signed target is
the gross markout value of preventing the fill; spread capture remains inside
the observed fill markout, so a cancel must overcome the spread it forgoes.

The source/action rows, exact-event 10 ms cooldown, feature schema, horizons,
latency ladder, window selection and already-seen split are identical to v2.
This is a model-family comparison, not a new data result: train is
2026-08-20/21/22 and development holdout is 2026-08-23/24. Nothing can promote
on those days. The linear comparator is refitted on the same incentive-free
target with the v2 standardized ridge `alpha=10.0`; no v2 metric with a rebate
in its target is reused as a baseline.

The nonlinear candidate is LightGBM with no early stopping or holdout-driven
tuning. Both the toxic-fill classifier and direct signed-value regressor pin:
128 trees, learning rate 0.05, `num_leaves=15`, `max_depth=4`,
`min_child_samples=200`, column fraction 0.8, row fraction 1.0,
`reg_alpha=0`, `reg_lambda=10`, seed 20260824, deterministic column-wise
training. The classifier uses binary log loss at natural prevalence; the
regressor uses squared error because the decision needs a conditional mean,
not a conditional median. The action remains `CANCEL iff predicted signed
gross cancel value > 0`; no threshold is selected from holdout.

Every horizon/latency cell reports nonlinear and linear R²/MAE/RMSE, selection
gain against the constant action selected on training only, cancellation
fraction, prevented-share fraction, per-day gains, and nonlinear-minus-linear
realized value. Toxic-fill Brier skill is diagnostic only. A cell receives the
development label `MODEL_SIGNAL_PRESENT` only if all of the following hold:

1. nonlinear direct-value holdout R² is positive;
2. nonlinear selection gain versus the training-selected constant is positive
   on each development day and in aggregate;
3. nonlinear realized value exceeds the linear policy on each development day
   and in aggregate; and
4. the nonlinear cancellation fraction is strictly between 2% and 98%.

The label is deliberately difficult and is still not validation. All cells are
reported; the headline is never the best cell. If no cell passes, model quality
has failed before real-latency work and a dynamic cancel×skew composition is not
licensed. Skew is nevertheless measured separately, without cancellation, by
the already declared policy-optimizer Stage-B grid using the pessimistic
`SKEW_LB` semantics: the reducing side may front only on genuine level
formation, and after a full lift it re-joins behind displayed depth. That
stateful replay is the inventory/risk result; overlapping 10 ms shadow rows are
never summed into a fictitious inventory path.

## 11. Nonlinear v3 and separate skew result — 2026-08-24

The v3 implementation is `adverse_move_nonlinear.py`; its local receipt is
`data/pm_5min/derived/adverse_move_nonlinear_development_v3.json`. It reused the
231,092 v2 feature/action rows and refitted both families on the incentive-free
gross target. **No cell received `MODEL_SIGNAL_PRESENT`.** All 52 nonlinear
direct-value fits had negative holdout R² (range -0.0961 to -0.0041), and the
nonlinear toxic-fill classifier beat the logistic comparator in only 2 of 10
coin/horizon cells.

There are narrow sign-selection diagnostics, but none changes the result. On
BTC at H=100 ms, L=10/25 ms, LightGBM beat ridge on both development days and
had aggregate gains of +0.00740/+0.00865 c per decision against the
training-selected constant; its gain against that constant was nevertheless
negative on 2026-08-23, and R² was -0.0514/-0.0448. On ETH at H=100 ms,
L=75 ms, selection improved on the constant and ridge on both days, but the
realized cancel delta remained negative on both days and R² was -0.0562.
These are useful directions for a future separately frozen sign model, not a
license to relax the v3 magnitude gate after reading it.

The separate stateful skew runner is `policy_optimizer_skew.py`; its receipt is
`data/pm_5min/derived/policy_optimizer_stageB_skew_v1.json`. It ran the six
incentive-free `SKEW_LB` cells on 300 BTC/ETH windows across five complete days.
All 60 coin-day cells were negative and no cell promoted. Skew lost less than
symmetric FRONT in 60/60 cells but beat conservative JOIN in only 15/60. At the
5-share/no-abstention cell, daily p95 cash-at-risk stayed in $7.00-$12.68 on BTC
and $3.55-$4.49 on ETH. This reinforces the existing interpretation: skew is a
strong inventory/damage controller, not a source of positive fill economics.

The nonlinear prerequisite failed, so no dynamic cancel×skew replay is run and
real-latency measurement is deferred. A future candidate that clears the model
gate must measure latency outside this research repository as the full chain
`event received -> features ready -> decision -> cancel submitted -> venue ACK
-> order no longer fillable`, with p50/p90/p99 and load/side stratification.
Inference latency alone may not be substituted for cancel-effective latency,
and this repository must not add the live order or cancel path needed to obtain
that measurement.

## 12. Action-conditioned hurdle v4 — protocol and development result

V3 established that direct squared-error regression on the zero-heavy signed
cancel target does not learn usable action value. V4 therefore preserves the
entire v3 population and decomposes each horizon/latency target without making
fill and price-direction forecasts independent. Both stages are conditioned on
the same JOIN-at-touch shadow action and point-in-time feature row:

1. a LightGBM classifier estimates the probability that positive filled
   quantity remains preventable after the assumed cancellation latency; and
2. a LightGBM regressor, fitted only on those preventable-fill rows, estimates
   their signed incentive-free gross cancellation value.

The unconditional decision score is exactly
`P(preventable fill | x, action) * E[gross cancel value | preventable fill,
x, action)`. This is a zero-inflated factorization of the action-bound target,
not the prohibited product of an independent direction forecast and a fill
model. Nonpreventable rows have gross target zero by construction; the runner
audits that identity in every cell. The cancellation rule remains `score > 0`.
No threshold, cancel fraction or hyperparameter is selected from holdout.

Both stages reuse the pinned v3 tree capacity (128 trees, learning rate 0.05,
15 leaves, depth 4, minimum child population 200, column fraction 0.8,
regularization 0/10, deterministic seed 20260824). Direct v3 LightGBM and
refitted ridge models are evaluated on the identical incentive-free target.
The split remains the already-seen 2026-08-20/21/22 train and 2026-08-23/24
development holdout, so v4 cannot freeze or promote regardless of its result.

A development signal requires positive preventable-fill Brier skill and
positive combined value R²; positive selection versus the training-selected
constant and direct v3 tree both in aggregate and on each day; and a 2–98%
cancel fraction. The implementation is `adverse_move_hurdle.py`; the receipt is
`data/pm_5min/derived/adverse_move_hurdle_development_v4.json`.

On the same 231,092 rows, **0/52 cells passed**. The decomposition audit held in
all cells. Preventable-fill Brier skill was positive in 41/52 cells (all 26 BTC
cells and 15/26 ETH cells), ranging from -0.0618 to +0.2469. This establishes a
real but incomplete fast fill-occurrence signal. The conditional-value stage
had positive holdout R² in 0/52 cells (range -0.9293 to -0.0042), and the
combined unconditional estimate also had positive R² in 0/52 cells (range
-0.3341 to -0.0006).

The hurdle policy beat the direct tree in aggregate in 30/52 cells, showing
that respecting the point mass at zero is better structured than direct
regression. It beat the training-selected constant in only 5/52 cells and on
both development days in only one cell. That one cell, ETH H=100 ms/L=50 ms,
still realized -0.0103 c/decision, had combined R² -0.0084, and lost to the
direct tree on one of two days. BTC H=250 ms/L=10/25/50 ms beat the direct tree
on both days and had tiny aggregate gains versus the constant, but all three
constant comparisons reversed sign across the two days and combined R² stayed
between -0.140 and -0.103.

The bottleneck is therefore no longer described as simply “no fast signal.”
Fast action-conditioned features can forecast whether a cancellation remains
able to prevent a fill, especially on BTC. They do not forecast the signed
economic value conditional on that fill. Dynamic cancel×skew composition and
real-latency measurement remain unlicensed; adding either would turn a useful
fill-occurrence forecast into an unsupported cancellation policy.

## 13. Value-weighted harmful-flow v5 development protocol

The next experiment changes the statistical loss, not the rows, features,
timing or economics. It directly tests the user's intended policy: prevent
harmful fills while retaining profitable spread capture. For each horizon and
assumed latency, v5 first keeps the v4 action-conditioned preventable-fill
classifier. On training rows where positive quantity is preventable, it then
fits two classifiers to `helpful_cancel = (gross_cancel_value_cents > 0)`:

1. an unweighted harmful-fill classifier, estimating event-count probability;
2. a value-weighted classifier with sample weight
   `abs(gross_cancel_value_cents)`, estimating the share of absolute economic
   mass on which cancellation helps.

Zero-value fills have no action preference and receive zero economic weight.
No outcome from a nonpreventable row enters the conditional classifier. If the
value-weighted probability is denoted `q(x)`, its population optimum is

```
q(x) = E[|V| 1(V > 0) | x, preventable] / E[|V| | x, preventable].
```

Therefore `q(x) > 0.5` exactly when conditional expected signed cancel value is
positive. The action score `P(preventable | x, action) * (2q(x)-1)` has the same
sign and provides occurrence-aware ranking; it is not claimed to be calibrated
in cents. This avoids asking squared-error regression to recover a noisy
magnitude before choosing the action sign.

Both conditional classifiers reuse the pinned v3/v4 LightGBM capacity and
natural rows: 128 trees, learning rate 0.05, 15 leaves, depth 4, minimum child
population 200, column fraction 0.8, regularization 0/10 and deterministic seed
20260824. Economic sample weights are the target definition, not class
rebalancing. There is no winsorization, early stopping, threshold selection or
holdout-driven tuning. The existing v4 hurdle and v3 direct LightGBM are refit
as comparators on the identical population.

The split remains 2026-08-20/21/22 train and the already-seen 2026-08-23/24
development holdout. V5 cannot freeze or promote. Each cell reports weighted
Brier skill against the training weighted prevalence, unweighted probability
diagnostics, economic-weight effective sample size, action value versus the
training-selected constant, the unweighted sign model, v4 and v3, plus
per-development-day reversals.

A `HARMFUL_FLOW_SIGNAL_PRESENT` development label requires all of:

1. positive value-weighted Brier skill;
2. positive realized cancellation value in aggregate and on each day;
3. positive selection gain versus the training-selected constant in aggregate
   and on each day;
4. positive value versus both the unweighted sign model and v4 hurdle in
   aggregate and on each day; and
5. a cancellation fraction strictly between 2% and 98%.

This gate deliberately does not require value R²: v5 is a predeclared
economically weighted sign/ranking experiment. It also cannot supply missing
cancel/rejoin queue cost. Passing would license a separately frozen forward
candidate, not dynamic cancel×skew or a decision-facing cancellation module.
