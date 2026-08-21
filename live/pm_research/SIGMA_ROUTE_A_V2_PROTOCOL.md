# SIGMA Route-A v2 conditional-variance protocol — frozen before evaluation

Protocol version: `route_a_v2`. Frozen: 2026-08-21T01:54:47Z.
Status: **PRE-REGISTERED / NOT YET EXECUTED / PRICING HOLD**.

This protocol is a successor candidate for one narrow failure mode in
`route_a_v1`: a pooled residual variance that is not conditionally calibrated.
It does **not** replace, amend or rescue the frozen v1 verdict. Route A v1 must
still be rerun unchanged after ten OOS test days and reported on its own terms.

## Design provenance and honest evaluation boundary

The choice below was made after inspecting the descriptive 2026-08-20 v1
artifact, whose signed `x = S30-S60` terciles often showed lower residual
variance in the middle and higher variance in one or both tails. The inspected
artifact is pinned as:

- source digest:
  `97d3c2a2253dab9c8babf5c6580a4584881c692a6a588509cdbf65c21ad7aba0`;
- v1 protocol SHA-256:
  `79c0f6b1ece5f4b15d2ddebaade5603cbea51634a495820dfb1853413a91a65d`;
- v1 script SHA-256:
  `f4b8e289e91086eb1ed048d22d2b0cf90cf68498044bc927591fb26425dee537`.

That snapshot is **design data**, not validation data. Primary v2 evaluation
starts on the first complete UTC day whose full collection interval postdates
this freeze: **2026-08-22**. Rows from 2026-08-19 through 2026-08-21 may train a
fold but may not contribute to a v2 gate, confidence interval or headline
score. No result from the incomplete 2026-08-21 day was inspected to choose
this specification.

## Decision and scope

V2 keeps every v1 choice except the conditional-variance estimator:

- same seven independently fitted symbols: BTC, ETH, SOL, XRP, DOGE, BNB, HYPE;
- same horizons: `r = 30, 60, 120, 180, 240, 270` seconds;
- same immutable inputs, knowledge-time reads, coverage/freshness filters,
  settlement target, units and no-intercept conditional mean;
- same strictly-forward UTC-day folds;
- same per-symbol/per-horizon separation, with no pooling or shrinkage across
  instruments or horizons;
- same reduced-form route: no `k_law`, `v(r)` or `Omega` term is added.

The v1 conditional mean remains:

```text
y = 10,000 * (S60(T) - S60(t)) / S60(t0)
x = 10,000 * (S30(t) - S60(t)) / S60(t0)
m = 10,000 * (S60(t) - S60(t0)) / S60(t0)

E[y | x,m] = alpha(symbol,r) * x
```

There is still no intercept. V2 cannot pass if this mean law fails its own
gate; a variance model may not hide conditional-mean bias.

## Frozen variance law

The variance predictor uses only the signed `x` tercile already named in v1.
It does not add `m`, book state, realised volatility, coin pooling, time of day
or a searched interaction. The signed cells allow the two tails to differ; no
symmetry is imposed after seeing one asymmetric regime-day.

For a test day `d`, independently for one `(symbol,r)` fit:

1. Fit `alpha_d` on all admissible raw rows from UTC days `< d`, exactly as v1.
2. Build **historical strictly-forward residuals** for days `< d`: a residual
   on historical day `h` must have been predicted by an alpha fitted only on
   raw days `< h`. Residuals fitted on their own day are forbidden.
3. Estimate the 1/3 and 2/3 quantiles `q1_d,q2_d` from raw training `x` values
   on days `< d`. Apply those two cuts to the historical forward residuals.
4. Let `H_d` be those historical forward residual rows and
   `s2_d = mean_{j in H_d}(e_j^2)`.
5. For signed cell `c in {low,mid,high}`, with `n_c` historical residuals, set

```text
lambda = 30 rows                         # frozen shrinkage strength
v_c,d = (sum_{j in c}(e_j^2) + lambda*s2_d) / (n_c + lambda)
```

6. A test row receives `v_hat_i = v_c,d` according to its observed `x_i` and
   the training cuts. Both `x_i` and its bin are known at the decision time.

The shrinkage target is the same fit's pooled historical forward MSE, not
another symbol or horizon. It guarantees a positive finite estimate when the
pooled MSE is positive and prevents a thin tail cell from becoming an
unbounded variance multiplier. There is no fitted shrinkage weight and no
post-hoc cap/floor. A fold refuses if `s2_d` is non-positive/non-finite, if
fewer than 30 historical forward residuals exist in total, or if any emitted
cell variance is non-positive/non-finite.

The frozen v1 comparator for the same test row is `v_pool_i = s2_d`. Thus v1
and v2 use identical historical residuals and differ only in whether their
second moment is pooled or selected by the predeclared signed-x cell.

## Cross-fitting and deployment candidate

All evaluated predictions are strictly forward:

- the test day's alpha, cuts, pooled variance and three cell variances use only
  earlier UTC days;
- the historical residual training set is itself forward-predicted;
- no test-day outcome, residual, cell count or variance enters its own fit;
- the non-overlapping 300-second market grid retains v1's non-overlapping label
  support at day boundaries.

After all gates pass, a descriptive deployment candidate may fit alpha and the
cuts on all admissible rows and estimate the three variances from all available
strictly-forward historical residuals. Full-sample in-sample residuals may not
replace that residual ledger.

## Frozen diagnostic cells

Evaluation retains v1's seven cells:

1. all evaluation rows;
2. low/middle/high signed `x` terciles;
3. low/middle/high `m` terciles.

Both pairs of cuts are estimated from each fold's raw training rows and applied
unchanged to that fold's test rows. The variance law is allowed to consume the
signed-x cell only. The `m` cells remain a genuine omitted-condition check.

## Gates

The inference unit is a complete UTC evaluation day, never a decision row.
Every gate requires at least **10 primary evaluation days** (all on or after
2026-08-22) and at least **30 evaluation rows in every frozen cell**. Use 5,000
deterministic day-block bootstrap draws with seed `20260821`, recomputing the
maximum cell statistic within every draw. Use percentile intervals with
Bonferroni family coverage across the six horizons of one symbol
(`alpha = 0.05/6`).

Each of the 42 symbol/horizon fits has three gates, for 126 required `PASS`
verdicts.

### 1. Conditional-mean gate — unchanged estimand

Let `s2_oos = mean(e_i^2)` over the primary evaluation rows. In each frozen
cell compute `|mean(e_i)| / sqrt(s2_oos)` and take the maximum. Tolerance:
**0.10 residual sigma**, exactly the v1 economic tolerance.

- `PASS`: simultaneous upper bound `<= 0.10`;
- `MODEL_REFUTED`: simultaneous lower bound `> 0.10`;
- otherwise `INSUFFICIENT_EVIDENCE`.

### 2. Conditional-variance calibration gate

For every row define `q_i = e_i^2 / v_hat_i`. In each frozen cell compute
`|mean(q_i)-1|` and take the maximum. Tolerance: **0.25 relative variance**, the
same scale used to reject v1 pooling.

- `PASS`: simultaneous upper bound `<= 0.25`;
- `MODEL_REFUTED`: simultaneous lower bound `> 0.25`;
- otherwise `INSUFFICIENT_EVIDENCE`.

### 3. Incremental variance-score gate

V2 must improve on its paired pooled comparator, not merely fit the cells it
was designed around. For each row compute Gaussian quasi-log scores (constant
terms omitted):

```text
L_v2_i   = 0.5 * (log(v_hat_i)  + e_i^2/v_hat_i)
L_pool_i = 0.5 * (log(v_pool_i) + e_i^2/v_pool_i)
delta_i  = L_v2_i - L_pool_i                 # lower is better
```

Aggregate `delta_i` to a mean within each UTC day before inference.

- `PASS`: one-sided simultaneous upper bound `< 0`;
- `MODEL_REFUTED`: one-sided simultaneous lower bound `>= 0`;
- otherwise `INSUFFICIENT_EVIDENCE`.

This score is a comparison gate, not permission to assume Gaussian tails. The
link family remains a separate belief-layer choice.

## Programme verdict and anti-rescue rules

`route_a_v2` is pricing-ready only if all 126 gates pass and the Route-A input
selection audit is acceptable. Any `MODEL_REFUTED` gate yields
`MODEL REFUTED — PRICING HOLD`; any missing or overlapping interval yields
`INSUFFICIENT_EVIDENCE — PRICING HOLD`.

Before the primary verdict, do not:

- change `lambda`, the three cells, their cuts, the evaluation start, a
  tolerance or the bootstrap because of observed results;
- add `m`, realised volatility, book variables or interactions to v2;
- reclassify 2026-08-20 or 2026-08-21 as validation data;
- let v2 overwrite, reinterpret or suppress the frozen v1 result;
- use v2 at probability level merely because v1's pooled-variance gate fails.

A different variance family is `route_a_v3` and needs its own dated protocol.

## Required output

The future implementation must emit, for every OOS row, the source slug, fold
day, alpha, training cuts, historical-forward residual count, pooled variance,
three shrunken cell variances, selected cell, selected variance, residual and
paired score difference. It must pin source/protocol/script hashes and carry the
Route-A selection ledger described by `route_a_v1`'s audit output.

No v2 fitting or result file is produced by freezing this document.
