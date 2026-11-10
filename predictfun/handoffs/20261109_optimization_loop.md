# Optimization loop — 2026-11-09

Verified comparison as-of **2026-11-09T16:33:35.408108+00:00**. **220 consumed Nov8 windows**, **279,059 decisions**, **13,077 neutral touch-pull actions** per scenario/arm; zero excluded windows. One partial day; no confidence interval or independent validation.

The loop completed two predeclared rounds: inventory restrictions, then shorter grace and latency stress. These results concern conditional quote mechanics and exposure. They do not estimate trading P&L, realized inventory variance, or settlement gains.

All arms use the same 0.01-cent soft exit except the reference. `all100/50/25` allow an accepted small touch order up to the named milliseconds. `individual100` also requires that order alone to reduce recorded inventory without crossing zero. `reserved100` requires enough reducing room for **all** same-side reservations, including pending creates/cancels and their grace. Flat inventory is refused by both inventory variants.

The extra inventory observer captures partial/full fill notices at their native knowledge time. Original native decisions, requests, books and due-event probes remain exact. Hypothetical orders never consume captured fills.

| Latency | Policy | Requests | Request change | Original aligned time change | Off-target non-reducing share-time change | Active inventory-range expansion change |
|---|---|---:|---:|---:|---:|---:|
| base | reference | 65,211 | +0.00% | +0.00% | +0.00% | +0.00% |
| base | all100 | 60,651 | -6.99% | +2.65% | +15.69% | +3.44% |
| base | individual100 | 64,978 | -0.36% | +0.13% | +0.00% | +0.00% |
| base | reserved100 | 65,177 | -0.05% | +0.02% | +0.00% | +0.00% |
| base | all50 | 65,203 | -0.01% | +1.23% | +11.83% | +2.21% |
| base | all25 | 65,211 | +0.00% | -0.00% | +7.66% | +1.13% |
| faster | reference | 65,332 | +0.00% | +0.00% | +0.00% | +0.00% |
| faster | all100 | 60,744 | -7.02% | +1.24% | +42.23% | +3.26% |
| faster | individual100 | 65,099 | -0.36% | +0.06% | +0.00% | +0.00% |
| faster | reserved100 | 65,296 | -0.06% | +0.01% | +0.00% | +0.00% |
| faster | all50 | 65,326 | -0.01% | +0.50% | +32.25% | +2.20% |
| faster | all25 | 65,330 | -0.00% | -0.00% | +18.09% | +1.13% |
| slower | reference | 64,548 | +0.00% | +0.00% | +0.00% | +0.00% |
| slower | all100 | 61,292 | -5.04% | +3.27% | +2.65% | +1.96% |
| slower | individual100 | 64,349 | -0.31% | +0.17% | +0.00% | +0.00% |
| slower | reserved100 | 64,524 | -0.04% | +0.02% | +0.00% | +0.00% |
| slower | all50 | 64,542 | -0.01% | +1.99% | +1.73% | +1.19% |
| slower | all25 | 64,542 | -0.01% | +1.00% | +0.88% | +0.60% |

Latency assumptions (create / accepted-cancel / pending-create-cancel): base **40/22/95 ms**, faster **20/10/50 ms**, slower **80/50/160 ms**. Cancellation reservation grace remains 1 second. Each row compares with its own latency reference. These are stress scenarios, not fitted venue quantiles.

“Non-reducing” includes flat inventory. “Inventory-range expansion” is the maximum absolute endpoint of `[recorded net − down orders, recorded net + up orders]` above `abs(recorded net)`, integrated over time. It does not net opposite orders and is not a realized risk or variance estimate.

## Computed comparisons

- `individual100`: retains 4.86% of unrestricted100ms coverage gain at base latency; 208 retained admissions / 118 same-price reopens. Fewer requests and more original aligned time in all three scenarios: **True**.
- `reserved100`: retains 0.68% of unrestricted100ms coverage gain at base latency; 29 retained admissions / 17 same-price reopens. Fewer requests and more original aligned time in all three scenarios: **True**.
- `all50`: retains 46.54% of unrestricted100ms coverage gain at base latency; 4,772 retained admissions / 0 same-price reopens. Fewer requests and more original aligned time in all three scenarios: **True**.
- `all25`: retains -0.02% of unrestricted100ms coverage gain at base latency; 4,772 retained admissions / 0 same-price reopens. Fewer requests and more original aligned time in all three scenarios: **False**.

Full action statuses, refusal reasons, inventory-check identities, source hashes and artifact hashes are in `reviews/optimization_loop_20261109/{inventory,duration_latency}/`. The duration round reconciles all four earlier base-latency arms. Both old reference and unrestricted100ms policy match the preceding soft-pull study.

## Limits and next work

- The factual inventory path is exogenous. A changed fill path would change it. Positive exposure mechanics cannot establish lower realized inventory or preserve trading profits.
- The historical fill-clock gate remains closed; the observer uses exact notice time, never an inferred match timestamp. Protected Nov9–13 remain unread.
- The separately implemented Rust A2 paired-order engine is not the order manager in this test. B4 parity/integration remains necessary.
- Calibrated fills or a controlled forward experiment are needed before economic selection or deployment. No policy is promoted by this report.
- A softer inventory-range restriction is a possible next separately declared experiment if strict no-flip gating sacrifices too much quote continuity.

Research branch `research/predict-opt-loop-20261109`. Reproduction: `pf_native_inventory_events` observer, `pf_inventory_grace_study declare/run ROUND`, `pf_inventory_grace_verify ROUND`, and `pf_optimization_report`, serially under `run_light.sh`. Declarations are committed before outcomes; completed output paths refuse overwrite.
