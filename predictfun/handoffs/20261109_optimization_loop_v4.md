# Optimization loop: native latency sensitivity — 2026-11-09

As-of **2026-11-09T17:15:06.600749+00:00**. **220 consumed Nov8 windows**, **279,059 decisions per profile**, **13,077 neutral pull actions per arm/profile**; zero exclusions. One partial day, no confidence intervals or independent validation. Each latency profile has its own paired reference.

This fifth pass changes only assumed venue feedback delays. The policies, native A2 execution code, factual intent path, capacity and reservation grace are unchanged.

| Latency (create/cancel/pending ms) | Venue request change | Original aligned time change | Off-target time change | Active share-time change | Active range expansion change |
|---|---:|---:|---:|---:|---:|
| faster (20/10/50) | -5.35% | +2.94% | +71.76% | +3.97% | +4.02% |
| base (40/22/95) | -5.33% | +4.17% | +28.36% | +4.10% | +4.16% |
| slower (80/50/160) | -3.72% | +4.37% | +5.61% | +2.58% | +2.62% |

Computed predicate—fewer venue requests and more original-price quote time than reference, with less off-target time than100ms—in all three profiles: **True**. The same predicate separately in both captured modes and each profile: **True**.
These are assumed stress cases, not measured latency quantiles or additional days. The changes do not establish higher spread capture, settlement P&L, favorable inventory or lower variance.

| Profile | Reference coverage | 50 ms coverage | 100ms aligned gain retained | Additional off-target time removed vs100ms |
|---|---:|---:|---:|---:|
| faster | 83.22% | 85.67% | 97.48% | 22.86% |
| base | 73.85% | 76.93% | 94.43% | 22.96% |
| slower | 64.04% | 66.84% | 90.34% | 21.90% |

## All three policies

| Profile | Policy | Venue requests | Aligned lane-seconds | Off-target lane-seconds | Active share-seconds | Reserved share-seconds |
|---|---|---:|---:|---:|---:|---:|
| faster | reference | 62,908 | 4,550.25 | 302.16 | 51,847.10 | 393,574.31 |
| faster | all100 | 59,537 | 4,687.57 | 583.27 | 54,237.21 | 389,159.80 |
| faster | update_first50 | 59,541 | 4,684.10 | 519.01 | 53,903.79 | 388,840.15 |
| base | reference | 62,788 | 4,038.32 | 680.03 | 50,982.92 | 398,385.66 |
| base | all100 | 59,440 | 4,216.59 | 930.36 | 53,408.15 | 393,832.84 |
| base | update_first50 | 59,444 | 4,206.66 | 872.89 | 53,075.68 | 393,514.12 |
| slower | reference | 62,119 | 3,501.46 | 1,561.26 | 55,099.51 | 409,428.00 |
| slower | all100 | 59,811 | 3,670.96 | 1,673.50 | 56,721.85 | 405,856.54 |
| slower | update_first50 | 59,811 | 3,654.59 | 1,648.92 | 56,518.49 | 405,652.76 |

## Verification and limits

- The base native output is byte-identical to the previous saved experiment before either stress case is run. Every base metric is also exactly equal. This isolates the specified latency change.
- Nine Rust adapter controls and four Python checks pass. Controls exercise all three create/cancel clocks, pending cancellation, rejection/backoff, paired partial fills, same-clock reopening, later-reopen refusal, corrupt lifecycle/action identity, unknown profiles and changed base artifacts. Prior native library test evidence remains linked in findings_v3.
- Saved artifacts verify across1,980 window-arm-profile rows and117,693 action-arm-profile rows. Coverage and exposure are independently rebuilt from saved order lifecycles; neutral actions, commands, capacity and deadlines reconcile. Wrong receipt and population links are refused.
- All native venue tokens, 250–4000ms retries, three-unanswered-per-side limit,150-share external cap and1s external reservation grace are preserved. The share cap is distinct from the dollar loss budget.
- Quote targets, guards and recorded inventory remain factual/exogenous. No hypothetical fills, post-only rejection model, wallet/kernel I/O or shared cross-market throttle are introduced. Full B4 integration and real timer scheduling remain separate.
- No new policy parameters:21 cumulative distinct policy settings. This retests three existing policies under three assumed latency profiles. No live change, frozen-engine change, protected Nov9–13 read or fill-clock bypass.
- Sources75bf977e, declarationb1441503 were committed before outcomes. Earlier input-preparation correction and all previous receipts remain preserved.

The loop continues with B4 paired-order integration and synthetic differential checks. Calibrated fills or authorized forward evidence remain necessary before claiming preserved trading gains or lower inventory variance.

Reproduction: `python3 -m predictfun.research.pf_native_a2_latency {build,declare,run,verify}`, then `pf_native_latency_report`; run numeric stages serially under `run_light.sh`. Destinations refuse overwrites. Register and full receipts: `reviews/optimization_loop_20261109/plan.json`, `native_latency/`, and superseding `findings_v4.json`.
