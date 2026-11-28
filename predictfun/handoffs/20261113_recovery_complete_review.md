# Recovery portfolio report: the candidate fails all three scenarios

Full report as of **2026-11-13T18:01:17.599349+00:00**. All 18 full scenario-days are complete and individually reviewed. Final end-to-end integrity verification passed at **2026-11-13T18:18:37.057835+00:00**, with `objective_pass=false`. Its 876 named artifact hashes and all three receipt copies were checked again. The negative result is now verified under the frozen research assumptions.

The primary SELECTIVE10 model loses strict marked value to both the chronological guard and DIRECT10 in X, execution stress and the age500 sensitivity. Every corresponding day interval is negative, and actual inventory-square exposure rises in every comparison. It makes fewer learned pull decisions than the previous HISTORY10 model but has less entry quote presence in every scenario. All three declared improvement gates are false. No alternative arm is promoted after inspecting results.

## Paired value and measured inventory risk

Six consumed outer dates: Oct 26/27/31 and Nov 1/2/3. Each policy/scenario has **1,728 windows**. Marked differences below use common available-mark windows; the remaining windows stay in the missing count. Inventory-square exposure uses every window. Intervals use the original 1,000 resamples of the six UTC days, not trades or individual windows. These are exploratory historical estimates.

| Scenario | SELECTIVE10 versus | Common / missing windows | Marked difference, USD [95% day interval] | Inventory-square change |
|---|---|---:|---:|---:|
| X | causal_guard | 1,521 / 207 | -252.57 [-432.23, -72.31] | +59.56% |
| X | direct10 | 1,551 / 177 | -81.82 [-147.38, -34.92] | +9.64% |
| stress | causal_guard | 1,516 / 212 | -507.70 [-625.04, -398.12] | +38.45% |
| stress | direct10 | 1,549 / 179 | -86.38 [-129.91, -31.77] | +2.62% |
| age500 | causal_guard | 1,659 / 69 | -53.85 [-102.13, -10.01] | +246.64% |
| age500 | direct10 | 1,669 / 59 | -30.55 [-60.16, -5.02] | +286.30% |

The age500 path suppresses credit for young fills and evolves the entire policy path. It is a sensitivity assumption, not a measured venue rule, a live-fill calibration or a formal bound on profit. Its result reinforces the current rejection: the negative comparison does not disappear in that path. It does not establish a causal explanation for the difference.

The new candidate improves on old HISTORY10 in X and stress, but loses to it in age500. Thus even that narrower improvement is not consistent across the declared execution assumptions. The X/stress tables, daily values and paired comparisons exactly match the earlier partial report; the final report adds the remaining scenario without revising those numbers.

## Fewer pulls still coincide with less quote presence

All presence rows use all 1,728 windows, including zero-presence windows. Entry means the first 60 seconds. The full replay interval ends at 240.1 seconds, with the remainder reduce-only; these percentages are not 300-second live-round uptime. Pull counts are repeated policy decisions.

| Scenario | SELECTIVE10 entry any quote | Old HISTORY10 entry any quote | SELECTIVE10 entry both sides | SELECTIVE10 learned pulls | Old HISTORY10 learned pulls |
|---|---:|---:|---:|---:|---:|
| X | 19.57% | 25.82% | 1.69% | 90,479 | 184,052 |
| stress | 22.34% | 31.97% | 2.17% | 82,353 | 157,581 |
| age500 | 20.21% | 26.68% | 1.82% | 79,447 | 172,172 |

A lower cancellation count alone does not establish useful persistence. The [native absence diagnostic](20261113_recovery_absence_overlap.md) already shows frequent overlap between closed recovery gates and actual absence in X/stress. That overlap is descriptive; it does not establish that reopening sooner would earn money. The [DEFAULT comparison](20261113_recovery_primary_stress_review.md) also shows why maximizing presence alone is insufficient.

## What this means for the next model

The corrected avoidable-value targets and consistent recovery mechanism were not sufficient to make this forecast economically useful. Its recovery estimate is the short value of a neutral order, not identified fresh-entry value. Single-intervention labels also do not guarantee value under repeated policy decisions; the full replay provides the relevant negative test here.

The frozen five/50-share experiment remains the next controlled test of retained residual quantity. It has its own native KEEP/RESET/PULL targets, shared capacity limits and actual inventory-risk gates. Its scoring age and re-entry protocol differ, so the present result does not determine its outcome. The separately frozen adverse-depth forecast tests history against a static baseline, but a forecast gain would still need a priced policy and sequential risk evaluation. Neither experiment is retuned using this report.

The original goal remains active. All current dates are consumed; a forward-validation claim requires at least five later untouched complete UTC dates and a prior full freeze. Live fill calibration and top-maker parity remain unproven.

## Evidence and verification state

- [Final evidence check](../reviews/model_recovery_20261113/evidence_check.json), SHA `3bd21750832f5e65efbcf52b7d3ec94ab5b3db3e8493440ea0b80c25dcc8c6a7`: 13 build records, four smoke and 18 full replay records; 7,113 saved inference samples reconstructed, including 7,086 in the full runs. The frozen verifier reconstructs sampled capture targets, native populations, recovery events, forecast samples, portfolio comparisons, participation and final predicates, and checks fitted-artifact chronology. This review checks its named artifacts and scope; it does not claim a second full numerical replay.
- [Full portfolio report](../reviews/model_recovery_20261113/portfolio_result.json), SHA `24c84a764aa46c8d0d07fc380752689fc33e1f45747b87d27423bc78b2bd0d35`.
- [Final scenario artifact review](../reviews/model_recovery_20261113/run_age500_20261103_artifact_review.json): 25 named artifact hashes, three receipt copies, 311 runner-reconstructed inference samples and 11 refusal controls.
- [Recovery progress](20261112_model_recovery_progress.md) preserves all completed run checkpoints and the accounting correction provenance.

The portfolio report’s `status=PASS` is not an improvement verdict; its computed `objective_pass` is false. The final evidence check also has `status=PASS` and `objective_pass=false`. Its accounting correction links match the preserved declaration, controls, failed attempt and raw parity evidence. The recovery driver has exited; the frozen size experiment began at 18:18:56 UTC after verifying this receipt. No live changes.
