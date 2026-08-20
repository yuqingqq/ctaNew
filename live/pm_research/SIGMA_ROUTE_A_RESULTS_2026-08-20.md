# SIGMA Route-A measurement — protocol route_a_v1

Run time: 2026-08-20T14:21:10.684309+00:00. Status: **DESCRIPTIVE — INSUFFICIENT EVIDENCE**.

This is the first real fit of the Revision-5 reduced-form law. It uses
only observed S30/S60 streams and the observed settlement target; no
structural `k/v/Omega` term is added.

## Snapshot and admissibility

- source digest: `97d3c2a2253dab9c8babf5c6580a4584881c692a6a588509cdbf65c21ad7aba0`; immutable stream files: 46
- UTC data days: 2026-08-19, 2026-08-20; OOS test days: 2026-08-20
- final resolutions: 1956; admissible windows: 1560; regression rows: 9332; OOS rows: 5796
- settlement-direction agreement: 1560/1560 (100.00%)
- S30/S60 knowledge-time read skew: p50 83 ms; p95 924 ms

## Strictly forward OOS results

`alpha train` is the coefficient fitted only on days before the OOS day;
`alpha all` is descriptive. Residual variance is OOS mean squared error
around zero, so mean bias is not subtracted away.

| coin | r | rows/days | OOS rows/days | alpha train | alpha all | mean resid bp | resid sd bp | mean effect | var effect | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bnb | 30 | 223/2 | 138/1 | 1.352 | 1.452 | 0.187 | 1.815 | 0.160 | 0.313 | INSUFFICIENT |
| bnb | 60 | 223/2 | 138/1 | 1.113 | 1.284 | -0.242 | 4.303 | 0.235 | 0.503 | INSUFFICIENT |
| bnb | 120 | 222/2 | 138/1 | 0.818 | 1.385 | 0.068 | 7.458 | 0.300 | 0.305 | INSUFFICIENT |
| bnb | 180 | 222/2 | 138/1 | 2.113 | 2.176 | 1.581 | 10.558 | 0.269 | 0.899 | INSUFFICIENT |
| bnb | 240 | 222/2 | 138/1 | 2.867 | 2.008 | 0.664 | 13.895 | 0.268 | 0.079 | INSUFFICIENT |
| bnb | 270 | 222/2 | 138/1 | 1.773 | 1.566 | 0.800 | 14.392 | 0.201 | 0.438 | INSUFFICIENT |
| btc | 30 | 223/2 | 138/1 | 1.175 | 1.484 | 0.296 | 2.935 | 0.323 | 1.180 | INSUFFICIENT |
| btc | 60 | 223/2 | 138/1 | 0.747 | 0.876 | 0.012 | 5.948 | 0.236 | 1.996 | INSUFFICIENT |
| btc | 120 | 222/2 | 138/1 | 0.194 | 1.152 | 0.700 | 9.284 | 0.382 | 0.577 | INSUFFICIENT |
| btc | 180 | 222/2 | 138/1 | 2.479 | 2.182 | 1.970 | 12.771 | 0.423 | 0.720 | INSUFFICIENT |
| btc | 240 | 222/2 | 138/1 | 0.793 | 0.766 | 1.450 | 16.167 | 0.131 | 0.625 | INSUFFICIENT |
| btc | 270 | 222/2 | 138/1 | 1.373 | 1.122 | 1.438 | 16.734 | 0.224 | 0.525 | INSUFFICIENT |
| doge | 30 | 223/2 | 138/1 | 1.319 | 1.343 | 0.027 | 2.872 | 0.240 | 0.804 | INSUFFICIENT |
| doge | 60 | 223/2 | 138/1 | 1.082 | 1.292 | -0.185 | 5.919 | 0.220 | 0.424 | INSUFFICIENT |
| doge | 120 | 222/2 | 138/1 | 0.932 | 1.423 | 0.622 | 11.290 | 0.326 | 0.344 | INSUFFICIENT |
| doge | 180 | 222/2 | 138/1 | 2.655 | 2.229 | 2.675 | 15.091 | 0.372 | 0.275 | INSUFFICIENT |
| doge | 240 | 222/2 | 138/1 | 3.036 | 2.462 | 1.993 | 18.805 | 0.358 | 0.260 | INSUFFICIENT |
| doge | 270 | 222/2 | 138/1 | 4.078 | 2.257 | 2.342 | 22.163 | 0.352 | 0.493 | INSUFFICIENT |
| eth | 30 | 223/2 | 138/1 | 1.003 | 1.139 | 0.030 | 2.788 | 0.341 | 0.614 | INSUFFICIENT |
| eth | 60 | 223/2 | 138/1 | 0.912 | 0.973 | -0.882 | 7.131 | 0.257 | 0.477 | INSUFFICIENT |
| eth | 120 | 222/2 | 138/1 | -0.001 | 0.313 | -0.149 | 11.196 | 0.617 | 0.303 | INSUFFICIENT |
| eth | 180 | 222/2 | 138/1 | 2.112 | 1.916 | 1.411 | 14.348 | 0.414 | 0.712 | INSUFFICIENT |
| eth | 240 | 222/2 | 138/1 | 3.331 | 2.902 | 0.743 | 19.912 | 0.353 | 0.386 | INSUFFICIENT |
| eth | 270 | 222/2 | 138/1 | 3.427 | 2.615 | 1.462 | 21.186 | 0.410 | 0.372 | INSUFFICIENT |
| hype | 30 | 223/2 | 138/1 | 1.429 | 1.416 | 0.075 | 5.148 | 0.337 | 0.670 | INSUFFICIENT |
| hype | 60 | 223/2 | 138/1 | 2.399 | 2.283 | -0.630 | 10.785 | 0.467 | 0.448 | INSUFFICIENT |
| hype | 120 | 222/2 | 138/1 | 2.692 | 2.086 | -1.732 | 19.664 | 0.529 | 0.385 | INSUFFICIENT |
| hype | 180 | 222/2 | 138/1 | -0.894 | -0.159 | 1.509 | 24.526 | 0.895 | 0.798 | INSUFFICIENT |
| hype | 240 | 222/2 | 138/1 | 1.223 | 1.349 | 1.704 | 27.561 | 0.173 | 0.657 | INSUFFICIENT |
| hype | 270 | 222/2 | 138/1 | 4.759 | 4.160 | 1.204 | 31.528 | 0.589 | 0.407 | INSUFFICIENT |
| sol | 30 | 223/2 | 138/1 | 1.595 | 1.472 | 0.214 | 2.377 | 0.354 | 0.619 | INSUFFICIENT |
| sol | 60 | 223/2 | 138/1 | 0.384 | 0.588 | -0.147 | 5.850 | 0.340 | 0.939 | INSUFFICIENT |
| sol | 120 | 222/2 | 138/1 | 0.399 | 1.233 | 0.194 | 10.179 | 0.633 | 0.296 | INSUFFICIENT |
| sol | 180 | 222/2 | 138/1 | 2.643 | 2.518 | 1.540 | 13.995 | 0.291 | 0.405 | INSUFFICIENT |
| sol | 240 | 222/2 | 138/1 | 2.857 | 2.117 | 1.185 | 18.058 | 0.509 | 0.479 | INSUFFICIENT |
| sol | 270 | 222/2 | 138/1 | 4.811 | 2.916 | 1.372 | 21.333 | 0.548 | 0.249 | INSUFFICIENT |
| xrp | 30 | 222/2 | 138/1 | 1.348 | 1.406 | 0.000 | 3.762 | 0.188 | 0.729 | INSUFFICIENT |
| xrp | 60 | 222/2 | 138/1 | 1.316 | 1.491 | -0.264 | 8.888 | 0.281 | 0.675 | INSUFFICIENT |
| xrp | 120 | 221/2 | 138/1 | 1.550 | 1.512 | 1.836 | 16.074 | 0.526 | 1.258 | INSUFFICIENT |
| xrp | 180 | 221/2 | 138/1 | 1.013 | 1.810 | 4.657 | 21.102 | 0.534 | 1.369 | INSUFFICIENT |
| xrp | 240 | 221/2 | 138/1 | 1.454 | 1.506 | 5.214 | 26.476 | 0.360 | 0.694 | INSUFFICIENT |
| xrp | 270 | 221/2 | 138/1 | 3.555 | 2.757 | 6.356 | 28.513 | 0.539 | 0.943 | INSUFFICIENT |

## Exclusions

| reason | count |
|---|---:|
| `r120:stale_predictor` | 7 |
| `r180:stale_predictor` | 7 |
| `r240:stale_predictor` | 7 |
| `r270:stale_predictor` | 7 |
| `s30_window_coverage` | 374 |
| `s60_window_coverage` | 1 |
| `stale_target_boundary` | 21 |

## Verdict

This snapshot has **1 OOS test-day cluster(s)**. The frozen
gate requires 10, so every fitted cell is
`INSUFFICIENT_EVIDENCE` regardless of its point estimate. The numbers
above are a valid descriptive, strictly forward pipeline measurement;
they are not a probability-law authorization.

The point diagnostics are an early warning, not a gate verdict: **42/42** mean effects exceed 0.10 residual sigma and **40/42** variance effects exceed 0.25. With one test day these can be a day/regime effect; they provide no early support for homoskedastic Route A, but cannot yet refute it.

No new sigma specification is warranted from sample size alone. Re-run
this identical protocol as the day count grows. Only an OOS residual
diagnostic that eventually reads `MODEL_REFUTED` should reopen the
Route-A functional form.

Protocol: `live/pm_research/SIGMA_ROUTE_A_PROTOCOL.md`.
