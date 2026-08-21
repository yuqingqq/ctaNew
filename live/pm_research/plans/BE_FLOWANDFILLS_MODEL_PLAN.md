# BE-FlowAndFills — canonical model plan, Revision 3

Status: **FROZEN FOR FORWARD EVALUATION**

Frozen: 2026-08-21

Design-data cutoff: 2026-08-21T23:59:59Z

Primary evaluation begins: 2026-08-22T00:00:00Z

Scope: research only; no venue adapter, order sender, or live execution path

This is the single current specification for the flow model. It supersedes the
flow-model sections of all earlier revisions, including
`FLOW_MODEL_SPEC_REV2.md`. Earlier documents remain an audit trail, not a second
source of truth. The first descriptive fit is in `FLOW_INTENSITY_RESULTS.md`.

## 0. Current decision

The next build is a **per-coin marked point-process baseline**. A Hawkes term is
not the baseline and is not fitted merely because raw arrivals cluster. It is an
optional residual layer, admitted only after the full baseline is time-changed
and still rejects a unit-rate Poisson process on forward data.

What is already established:

- the 0.02-share class is a separately labelled event type, `MICRO_002`;
- the remaining labelled subprocess is `MARKET` (also called ex-micro);
- the class mix varies too much by coin for pooled parameters to be primary;
- the empirical `f_r` count profile is not a dominant rising settlement clock;
- arrival count and USDC throughput have different shapes, so throughput is
  not a second arrival intensity;
- the earlier marginal `f_p` result used execution-price bins in the numerator
  and midpoint bins in the denominator. That result is withdrawn and must not
  be used as a baseline or Hawkes compensator.

The maker-edge sign is still **UNDETERMINED** at the current number of days.
Nothing in this flow plan turns the websocket's zero fee field into maker PnL.

## 1. The stochastic object

### 1.1 Arrival identity and complement fold

One `last_trade_price` aggregate is one arrival. `OrderFilled` legs are not
arrival identities because one taker order can generate several legs.

Both outcome tokens are represented in the Up frame:

```text
p_exec_up = native_price              for an Up-token trade
p_exec_up = 1 - native_price          for a Down-token trade
BUY(Down) = SELL(Up), and conversely
```

The monetary mark is always the amount actually exchanged:

```text
V_i = size_shares_i * native_execution_price_i       [USDC]
```

It is **not** `size * p_exec_up`; that would overstate a Down trade at native
price 0.30 by valuing it at the folded Up price 0.70.

### 1.2 State, types, and marks

The pre-arrival state is

```text
x(t) = (coin, r, p_state, r_band, tick, spread_ticks,
        touch_notional, imbalance_notional)
```

where `r` is seconds remaining and `p_state` is the Up-book midpoint known at
least 250 ms before the arrival. `side`, `event_type`, execution price, size,
and notional are marks of the next arrival. In particular, the realized side of
the next event is not a covariate of total arrival intensity.

Use either equivalent factorization:

```text
lambda_all(t | H_t, x_t)                         events / second
P(type | arrival, H_t, x_t)
P(side | arrival, type, H_t, x_t)
G(V, p_exec | arrival, type, side, H_t, x_t)     mark law
```

or the cause-specific form

```text
lambda_type(t | H_t, x_t)
lambda_all = lambda_MICRO_002 + lambda_MARKET
```

Both are valid without assuming the two types are independent. Deleting the
micro rows and estimating `lambda_MARKET` estimates a **labelled subprocess**.
Independence is needed only to claim that deleting micro history has no effect
or to interpret the two processes counterfactually as unrelated.

### 1.3 Notional is a conditional mark, not an arrival rate

For the same conditioning state `x`:

```text
m_V(x, type) = E[V | arrival, x, type]             USDC / arrival
q_V(x, type) = lambda_type(x) * m_V(x, type)        USDC / second
```

Only `lambda` has a point-process compensator. `q_V` is derived notional
throughput. We may report empirical `sum(V) / exposure` descriptively, but do
not fit it as a substitute point-process intensity.

If the count-plus-mark decomposition is underpowered for a coin/state band,
return `Unavailable(DECOMPOSITION_UNDERPOWERED)` and report descriptive USDC/s
separately. Do not silently redefine the model.

## 2. Baseline specification

### 2.1 Per coin, never a pooled primary

All parameters and pass/fail verdicts are per coin. A pooled fit may be shown as
a diagnostic only and must disclose its event weights. There are no tied
coefficients across BTC, ETH, SOL, XRP, DOGE, BNB, and HYPE in the primary model.

There are not two structural models called Tier A and Tier B. There is one
factorization at different resolution. Sparse coins receive coarser `r`/`p`
bands or an explicit unavailable result; they do not switch from events/s to
USDC/s as if the units were interchangeable.

### 2.2 Frozen baseline order

Fit the following in order, retaining each layer only by forward likelihood:

```text
B0  per-coin piecewise-constant f_r(r)
B1  B0 + f_p(p_state | BODY/TERMINAL)
B2  B1 + tick-tail interaction
B3  B2 + f_book(spread, touch_notional, imbalance_notional)
M1  conditional type law
M2  conditional side law given arrival and type
M3  conditional monetary-mark law G(V | arrival, x, type, side)
```

`BODY` is `r in [60, 300]`; `TERMINAL` is `r in [0, 60)`. This boundary is a
frozen modelling choice motivated by the 60-second settlement construction,
not a discovered causal breakpoint. The wall-clock minute and five-minute
window phase are perfectly collinear in current data, so their effects are not
separately identified.

`f_r` and `f_p` are offsets/functions in the log ground intensity, not separate
arrival rates:

```text
log lambda_0(t | x_t) = intercept_coin
                       + f_r_coin(r_t)
                       + f_p_coin(p_state_t, r_band_t)
                       + gamma_coin * book_state_t
```

The first fit found `f_r` roughly flat in the body and falling near settlement;
it did **not** establish a dominant increasing clock. The empirical profile is
still retained because the Hawkes residual must not absorb any clock shape.

### 2.3 `f_p`: one state for numerator and denominator

For each quote received at time `t_recv`, its midpoint becomes admissible at
`t_recv + 250 ms`. This produces half-open state intervals. Both:

- arrivals in the numerator, and
- dwell time in the denominator

are assigned from these exact intervals. Execution price is a mark and never
chooses the `f_p` bin. If there is no admitted state, the arrival is rejected
from `f_p` and counted in `n_trades_no_state`.

A collector gap kills the current state at the gap start. It is not carried
across the gap; a new quote must mature through the 250 ms lag before exposure
or arrivals can re-enter. Report the execution-state bin mismatch rate as a
diagnostic.

Price bins are frozen at:

```text
[0,.05), [.05,.15), [.15,.35), [.35,.65),
[.65,.85), [.85,.95), [.95,1]
```

A bin with less than 60 seconds of dwell in the estimation sample is `FENCED`,
with its dwell shown. `f_p` cannot identify a fee effect: the taker fee is a
deterministic function of price and is collinear with moneyness.

### 2.4 Book and side layers

Book covariates use the same 250 ms pre-arrival state convention. Imbalance is
notional-weighted; the count-weighted version is a diagnostic because the 0.02
class shifts count imbalance in a state-dependent way. Tick size is read from
both snapshots and `tick_size_change` events and enters only as a tail
interaction.

Total `lambda` never consumes the next event's realized side. Side is modelled
as `P(BUY | arrival, type, x)`; a later cause-specific BUY/SELL intensity is
allowed only if it is algebraically reconciled to `lambda_all`.

## 3. Micro/market dependence

Independence is a simplifying null, not a prerequisite for estimating either
labelled subprocess. Test it only after both cause-specific baselines are fitted.

Raw cross-correlations are forbidden as the primary test because both types can
share `f_r`, `f_p`, and book-state clocks. Time-change each type by its fitted
cause-specific baseline, then test cross-history effects. In a two-type Hawkes
representation, independence corresponds to both off-diagonal branching terms
being zero. If rejected, retain a multitype/cross-history model and continue to
label the subprocesses; do not abandon or relabel ex-micro flow.

Underpowered dependence tests return `INSUFFICIENT_EVIDENCE`, never an assumed
independence verdict.

## 4. Optional Hawkes residual

### 4.1 What lambda, alpha, and beta mean

For a one-type exponential Hawkes process in calendar time:

```text
lambda(t) = mu(t) + sum_{t_i<t} alpha * exp[-beta * (t-t_i)]
n = alpha / beta
```

| parameter | meaning | unit | effect when increased |
|---|---|---|---|
| `lambda(t)` | instantaneous conditional arrival rate | events/s | more expected arrivals in the next short interval |
| `mu(t)` | baseline rate from `f_r`, `f_p`, and book state | events/s | raises flow without implying excitation |
| `alpha` | immediate rate jump after an arrival | events/s | makes the initial response larger |
| `beta` | exponential decay speed | 1/s | makes the response disappear faster |
| `n=alpha/beta` | expected direct offspring per event | dimensionless | measures branching strength; stationarity requires `n<1` |

Example with `mu=2/s`, `alpha=3/s`, and `beta=2/s` is invalid as a stationary
model because `n=1.5`. Holding `n=0.3` and setting `beta=2/s` gives
`alpha=0.6/s`: immediately after one event the rate rises from 2.0 to 2.6/s,
after one second the added rate is only `0.6*exp(-2)=0.081/s`. With the same
`n=0.3` but `beta=0.2/s`, the initial jump is only 0.06/s and lasts much longer.
Thus `alpha` alone is not the persistence parameter; `alpha/beta` is the total
branching mass.

### 4.2 Fit in operational time

First transform time by the fitted baseline compensator:

```text
u(t) = integral_0^t lambda_0(s | x_s) ds
```

The frozen one-type candidate on operational time is

```text
tilde_lambda(u) = (1-n) + sum_{u_i<u} n*beta*exp[-beta*(u-u_i)]
0 <= n < 1
```

Its stationary transformed mean is one. If the cross-type test admits a
multitype candidate, do **not** apply the scalar `(1-n)` independently to each
type. Fit instead

```text
tilde_lambda_a(u) = nu_a(u)
                  + sum_b sum_{u_i^b<u} N_ab*beta_ab*exp[-beta_ab*(u-u_i^b)]
```

for `a,b in {MICRO_002, MARKET}`. `nu_a(u)` starts from the fitted conditional
type probabilities and is jointly refitted on training data subject to mean
total transformed rate one. `N` is a nonnegative 2x2 branching matrix and
stationarity requires spectral radius `rho(N) < 1`. Off-diagonal entries are
the measured cross-excitation terms. Side is not added as a covariate; BUY/SELL
can become event types only in a separately powered marked model.

Here `u` is dimensionless operational time: one unit is one expected baseline
arrival. Candidate operational-time half-lives are frozen at
`{0.25, 0.5, 1, 2, 5, 10}` expected arrivals and are selected on training data
only. They are not seconds; the calendar duration corresponding to one unit of
`u` varies with the baseline rate.

### 4.3 When Hawkes is allowed

Do not fit the residual Hawkes layer until all are true for a coin:

1. B0–B3 have a complete, forward-scored baseline and an admissible compensator.
2. At least 10 complete primary-evaluation UTC days exist in one compatible
   collector era.
3. On baseline operational time, a pre-registered short-gap or dependence test
   rejects unit-rate Poisson after Holm correction across coins and tests.
4. The rejection has the direction of residual clustering, not merely a generic
   distribution mismatch.

If these are false, the Hawkes result is `NOT_ADMITTED`, not a zero branching
ratio and not a request to fit anyway.

### 4.4 Boundaries, gaps, and warm-up

Build one continuous arrival history per coin across adjacent five-minute token
markets. Baseline state may jump at a market boundary, but recent participant
history is carried across it. For every scored interval, retain unscored Hawkes
warm-up covering **both** at least 60 calendar seconds and at least 60 units of
baseline operational time. The latter covers six half-lives at the largest
candidate operational half-life; the former prevents a high-rate instant from
making the history requirement vacuous.

A collector gap terminates both the risk interval and known excitation history.
After a gap, data are not scored until both warm-up requirements are satisfied.
A window without the required observable warm-up is excluded with a named
reason.

## 5. Frozen forward protocol and gates

All choices above use data through 2026-08-21 and are frozen before the primary
period. The immutable manifest is `FLOW_MODEL_PROTOCOL_V3.yaml`.

Evaluation is expanding-window, day held out:

```text
fit_data_through < start_of_evaluation_day
evaluation unit = complete UTC day
primary minimum = 10 complete days per coin
uncertainty unit = UTC-day block
```

No bin, covariate, lag, half-life, or threshold may be selected using primary
days. Any change starts a new protocol/version and a new primary period.

### 5.1 Baseline retention

Each added baseline layer is retained per coin only when the day-block bootstrap
95% upper confidence bound for its forward delta negative log likelihood versus
the immediately nested model is below zero. Otherwise retain the simpler model.

For the final baseline, report time-rescaled inter-arrivals and:

- KS maximum CDF deviation from Exp(1);
- short-gap excess at transformed gap `<0.25`;
- Ljung–Box dependence through lag 10;
- randomized quantile residuals for mark/type/side laws.

Holm-correct inferential p-values across seven coins and the frozen tests.
Failure to reject is not proof of equivalence; the diagnostics and confidence
intervals remain visible.

### 5.2 Hawkes retention

A Hawkes candidate is retained only if all hold on strictly forward days:

- day-block bootstrap 95% upper bound for delta NLL versus the full baseline is
  below zero;
- the bootstrap 95% upper bound for `n` (or `rho(N)`) is below one;
- KS deviation does not worsen by more than 0.01;
- the clustering diagnostic that admitted Hawkes no longer rejects after Holm
  correction;
- fitted parameters remain inside their bounds on every training fold.

Otherwise publish `MODEL_REFUTED` or `INSUFFICIENT_EVIDENCE` and use the
baseline. In-sample likelihood never licenses retention.

## 6. Current artifact status

| artifact | status | permitted use |
|---|---|---|
| empirical `f_r` | descriptive, measured on `<2` days | design evidence only |
| original `f_p` in Revision 2 | **WITHDRAWN: state mismatch** | none |
| corrected same-state `f_p` | descriptive, measured on `<2` days | design evidence only |
| B0–B3 per-coin baseline | not fitted | next implementation |
| type/side/mark laws | not fitted | after baseline exposure is frozen |
| Hawkes residual | not admitted; insufficient forward days | specification only |
| maker profitability/fill response | sign unresolved | do not optimize |

## 7. Build order

1. Regenerate corrected same-state `f_p` and publish join diagnostics.
2. Accumulate primary data without changing this specification.
3. Materialize per-coin risk intervals and conditional marks.
4. Fit and forward-score B0–B3, then M1–M3.
5. Run operational-time residual diagnostics.
6. Fit Hawkes only for coins that satisfy the admission gate.
7. Build action-conditional fill probability only after the exogenous flow and
   mark law are forward-valid. Until then, `BE-FlowAndFills` returns
   `Unavailable` to a decision consumer.

This sequence prevents three known failure modes: pooled parameters hiding
per-coin differences, notional being mistaken for event intensity, and Hawkes
absorbing an omitted clock/state baseline.
