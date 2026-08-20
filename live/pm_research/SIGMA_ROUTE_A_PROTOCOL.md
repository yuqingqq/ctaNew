# SIGMA Route-A measurement protocol — frozen before the first fit

Protocol version: `route_a_v1`. Frozen: 2026-08-20, before running
`exp_sigma_route_a.py` on the regression target. Implements Phase 0A step 6 of
`SIGMA_PLAN.md` Revision 5.

## Decision and scope

This experiment measures the reduced-form conditional law of the observed
settlement mark. It does not estimate the structural decomposition and never
adds `k_law`, `v(r)` or `Omega` to the residual.

Symbols: BTC, ETH, SOL, XRP, DOGE, BNB and HYPE, each fitted independently.
Remaining-time horizons: `r = 30, 60, 120, 180, 240, 270` seconds.

The first run is `DESCRIPTIVE` because the immutable snapshot currently spans
two UTC days. A pricing-law verdict requires at least ten **OOS test-day**
clusters; with strictly forward folds, the first collected day is training-only.

## Frozen inputs and time semantics

Inputs:

- immutable rotated `*.csv.gz` files from
  `data/pm_5min/prices/crypto_prices_twap_thirty/` and
  `crypto_prices_twap_sixty/`;
- one byte snapshot each of `markets.jsonl` and `resolutions.jsonl` at process
  start;
- only resolutions with `closed=true` and a declared winner.

The active uncompressed hourly feed file is excluded. The result records the
source-file manifest and a combined SHA-256 digest.

For duplicate `(stream, symbol, payload_event_timestamp)` records, keep the
earliest `recv_ns`: that is the earliest time the value was knowable. Predictors
are sorted/read on `recv_ns`; outcomes are sorted/read on payload event time.

For a market `[t0,T]` and horizon `r`:

- decision time is `t = T-r`;
- `S30(t)` and `S60(t)` are the last observations with `recv_ns <= t`;
- neither predictor may be more than 5 seconds old at `t`;
- `S60(t0)` and `S60(T)` are read by payload event time because they define the
  ex-post settlement target, not information available to the forecast;
- each target boundary read must be within 5 seconds of its boundary;
- both streams must have at least 90% nominal one-Hz event-time coverage over
  `[t0-5s,T+5s]`, matching the existing E0 admissibility rule;
- windows must be exactly 300 seconds and the strike `S60(t0)` must be positive.

Final winner agreement is audited but is not an exclusion: dropping a mismatch
would condition on the realised target near the decision boundary.

## Estimand and fit

All values are normalized arithmetic returns relative to the settlement strike
and expressed in bps:

```text
y = 10,000 * (S60(T) - S60(t)) / S60(t0)
x = 10,000 * (S30(t) - S60(t)) / S60(t0)
m = 10,000 * (S60(t) - S60(t0)) / S60(t0)

y = alpha(symbol,r) * x + residual
```

There is no intercept. This enforces the translation-invariant family
`E[x_T] = S60 + alpha*(S30-S60)` rather than silently introducing a raw-price
level. Residual variance is `mean(residual^2)`, not mean-subtracted sample
variance; subtracting the OOS residual mean would hide mean-model bias inside
the reported variance.

`alpha = sum(x*y)/sum(x^2)` is fit independently for every symbol/horizon.
No pooling or shrinkage is allowed in this experiment.

## Cross-fitting

Folds are strictly forward by UTC market-start day:

- test day `d` trains on all admissible rows from days `< d`;
- a fold needs at least 30 training rows for that symbol/horizon;
- no random-row CV and no future-day leave-one-out fold;
- the day boundary exceeds the longest 270-second remaining-time label support,
  providing the label embargo; rows from the test day never enter its fit;
- every OOS row records source slug, timestamps, fold day, fitted alpha,
  prediction and residual.

The full-sample alpha is reported as a descriptive deployment candidate only.
All gates and the residual variance claim use OOS residuals.

## Frozen conditioning diagnostics

The conditional-mean and conditional-variance gates use seven predeclared
cells per symbol/horizon:

1. all OOS rows;
2. low/middle/high terciles of `x = S30-S60`;
3. low/middle/high terciles of `m = S60-strike`.

Tercile boundaries are estimated on the training portion of each fold and then
applied unchanged to that fold's test rows. Cells are marginal, not a post-hoc
cross-product. This tests the two conditioning directions named by Revision 5
without mining arbitrary bins.

Let `s2 = mean(e^2)` over all OOS rows for one symbol/horizon.

- mean-model effect:
  `max_cell |mean(e)/sqrt(s2)|`; tolerance **0.10 residual sigma**;
- variance-model effect:
  `max_cell |mean(e^2)/s2 - 1|`; tolerance **0.25 relative variance**
  (about 12% in sigma units).

Those tolerances are fixed here before inspecting the Route-A results. They
correspond to at most about four probability cents at the centre for the mean
gate and prevent a pooled variance from hiding economically material regime
variation.

## Inference and verdicts

Inference unit is the UTC OOS test day, never a decision row. At ten or more OOS
days, and with at least 30 OOS rows in every frozen cell, use 5,000 deterministic
day-block bootstrap draws (seed `20260820`). Each
draw resamples complete OOS days and recomputes the maximum cell effect.

The interval uses percentile bounds with Bonferroni family coverage across the
six horizons of one symbol (`alpha = 0.05/6`). For each gate:

- `PASS`: simultaneous upper bound is within the frozen tolerance;
- `MODEL_REFUTED`: simultaneous lower bound exceeds the tolerance;
- `INSUFFICIENT_EVIDENCE`: fewer than ten OOS days or the interval overlaps the
  tolerance.

The report must include the point effect, both bounds, tolerance, OOS row/day
counts and verdict. There are no p-value-only passes.

## Outputs and interpretation

The script writes:

- `data/pm_5min/derived/sigma_route_a_v1.json`: manifest, exclusions, all OOS
  rows and structured per-fit results (ignored by git);
- `live/pm_research/SIGMA_ROUTE_A_RESULTS_2026-08-20.md`: concise committed
  result table and interpretation.

The current run answers whether the pipeline executes and gives provisional
OOS coefficients/residual scales. It cannot authorize probability-level use.
The identical protocol must be rerun after ten OOS test days; changing a gate,
tolerance, exclusion or conditioning cell creates a new protocol version and
must be reported as such.
