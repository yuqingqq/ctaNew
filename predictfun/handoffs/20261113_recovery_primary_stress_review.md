# Recovery candidate: completed primary/stress comparisons are negative

As of **2026-11-13T15:39:36.887969+00:00**. This partial report covers the two completed scenarios, X and execution stress, on six consumed dates: Oct 26/27/31 and Nov 1/2/3. Each policy/scenario has **1,728 windows**. Age500 runs and the final full evidence check remain pending. No model, threshold, target or task order changed.

The primary SELECTIVE10 candidate improves on the previous HISTORY10 model, but loses strict marked value against both the chronological guard and the direct-value alternative while increasing inventory-square exposure. It also has less quote presence than the previous model despite fewer learned pulls. The covered value/risk, uncertainty and participation gates are all false. On these current outputs, an all-scenario success cannot be established by a favorable age500 result. The remaining declared work still must finish.

## Fixed portfolio comparisons

Values below are paired differences for SELECTIVE10. Marked value uses common windows and retains missing counts. Inventory exposure uses every window. Intervals resample the six UTC days with the original 1,000 draws. They are exploratory consumed-history results, not live-profit or forward-validation estimates.

| Scenario | Comparator | Common / missing windows | Marked difference, USD [95% day interval] | Inventory-square change |
|---|---|---:|---:|---:|
| X | default | 1,559 / 169 | +2,153.30 [+1,914.59, +2,346.10] | -0.36% |
| X | causal_guard | 1,521 / 207 | -252.57 [-432.23, -72.31] | +59.56% |
| X | direct10 | 1,551 / 177 | -81.82 [-147.38, -34.92] | +9.64% |
| X | old_history10 | 1,518 / 210 | +392.47 [+304.23, +483.75] | -13.49% |
| stress | default | 1,554 / 174 | +1,994.99 [+1,640.73, +2,283.41] | +9.66% |
| stress | causal_guard | 1,516 / 212 | -507.70 [-625.04, -398.12] | +38.45% |
| stress | direct10 | 1,549 / 179 | -86.38 [-129.91, -31.77] | +2.62% |
| stress | old_history10 | 1,541 / 187 | +631.89 [+513.48, +760.04] | +5.97% |

The comparison against the old model is not an isolated test of one change: the value targets and re-entry protocol changed together. The chronological guard also differs in entry/decision timing. The direct-value comparison shares the new recovery environment and provides another fixed benchmark; SELECTIVE10 fails it too. No arm is promoted from this interim view.

## Fewer pulls did not preserve more quoting time

All presence figures below include all 1,728 windows per row, including zero-presence windows. Entry is the first 60 seconds; the whole simulated interval is 240.1 seconds, with the remainder reduce-only. It is not a full 300-second live-round uptime measure. Learned-pull counts are repeated policy decisions, not distinct profitable opportunities.

| Scenario | Policy | Entry any quote | Entry both sides | Whole interval any quote | Learned pulls |
|---|---|---:|---:|---:|---:|
| X | default | 89.01% | 45.63% | 26.26% | 0 |
| X | selective10 | 19.57% | 1.69% | 7.61% | 90,479 |
| X | direct10 | 12.85% | 0.93% | 5.02% | 111,304 |
| X | old_history10 | 25.82% | 2.83% | 9.70% | 184,052 |
| X | causal_guard | 13.79% | 0.04% | 4.17% | not a learned policy |
| stress | default | 88.60% | 43.18% | 26.84% | 0 |
| stress | selective10 | 22.34% | 2.17% | 8.38% | 82,353 |
| stress | direct10 | 16.93% | 1.51% | 6.52% | 101,296 |
| stress | old_history10 | 31.97% | 4.35% | 10.94% | 157,581 |
| stress | causal_guard | 15.18% | 0.06% | 4.72% | not a learned policy |

SELECTIVE10 has more presence than DIRECT10, but that additional exposure accompanies worse value and greater inventory risk. Against the old HISTORY10 policy, fewer pulls coincide with less presence. A smaller cancellation count therefore cannot be treated as evidence that useful queue positions are being retained. It can also accompany fewer accepted orders or longer periods away; this report does not identify the cause of every absence.

## Default presence does not establish useful exposure

The same published partial receipt also includes DEFAULT, which has no learned pulls but retains native price, inventory and execution rules. These observations have the original as-of time above and the same six consumed dates. Each row below has 1,728 total windows; marked-value totals use only the stated available-mark population. They are descriptive totals, not paired differences.

| Scenario | Policy | Marked / missing windows | Marked total, USD | Inventory RMS, shares | Terminal absolute inventory, mean shares |
|---|---|---:|---:|---:|---:|
| X | default | 1,656 / 72 | -2,596.57 | 3.971 | 0.355 |
| X | selective10 | 1,596 / 132 | -291.31 | 3.963 | 1.408 |
| stress | default | 1,660 / 68 | -2,948.67 | 3.858 | 0.306 |
| stress | selective10 | 1,589 / 139 | -742.23 | 4.040 | 1.074 |

SELECTIVE10 improves paired marked value against DEFAULT in both completed scenarios. However, its all-window inventory-square exposure changes by only -0.36% in X and increases by 9.66% under stress, despite much lower entry presence. Lower quote presence is therefore not equivalent to lower inventory exposure. The observed differences do not independently identify the contribution of entry selection, hedging or inventory reduction. The model must improve the value of retained exposure and its subsequent inventory path; uptime alone cannot establish that improvement.

This is an additional reading of the unchanged partial receipt, not a new experiment, a revised success criterion or a reason to promote the candidate. SELECTIVE10 still fails its guard and DIRECT10 comparisons.

## Subsequent native absence diagnostic

At 2026-11-13T16:00:57.453496+00:00, the [native overlap reconstruction](20261113_recovery_absence_overlap.md) checks the same twelve completed scenario-days, all four recovery policies and actual accepted lifetimes. On the 1,728 SELECTIVE10 windows per scenario, both recovery gates are closed during 76.66% of entry no-quote time in X and 69.69% in stress. Cancel-pending orders remain present until their actual lifetime ends. This describes concurrent constraints; it does not identify avoidable absence or the return from reopening sooner. The economic comparisons above are unchanged.

## What this changes for the goal

This candidate does not solve the original low-presence/poor-timing problem in the completed scenarios. The short-value condition and shared recovery gate were useful design corrections, but their implementation alone did not create a sufficiently useful forecast. The earlier local calibration weakness now has negative full-policy evidence beside it.

The queued five/50-share experiment remains necessary to test the user’s residual-size mechanism under native partial fills and actual inventory risk. Its initial scoring age and pressure-clear protocol differ from this recovery candidate, so this result does not determine its outcome. The queued depth experiment separately tests whether history predicts further adverse movement beyond a static baseline. Neither experiment is retuned or selected using this report.

## Evidence and scope

Instrument source **53a0097b** calls the frozen reporter’s portfolio reconciliation, aggregation and paired comparison functions. It executes the unchanged success function in an isolated namespace containing only X/stress, renaming the result to `covered_scenario_gate_pass`; the full scientific scenario constants are unchanged. Existing artifact/coverage and success/failure controls pass, as do subset increased-risk and missing-comparison refusals.

The declaration binds 329 completed-run/source-receipt identities before reading portfolio values. All named inputs, instrument source hashes and three receipt copies matched after execution. Empty-fill settlement handling uses the existing committed accounting correction. The bounded auxiliary review exited zero, with 331 MiB peak RSS, 196.96 seconds wall and 189.66 seconds user CPU. It performs no new policy replay.

[Partial report](../reviews/model_recovery_20261113/partial_primary_stress_v1.json), SHA
`20fea81c7ecb9aa1b9636cd3f2a654b73c9fb8d93b3be8181e2b2cc30ac1b5fb`.

[Partial-view declaration](../reviews/model_recovery_20261113/partial_primary_stress_v1_declaration.json), SHA
`4e91293ef4830bae52b8b0146c483a43fadc308fd3e59c1efeb489424ac4ed26`.

The complete portfolio report and final verifier remain pending. The partial receipt is explicitly `PARTIAL_REPORT`, cannot release the size/depth dependencies, and does not replace a final evidence receipt. All current dates are consumed; execution calibration, untouched validation and top-maker parity remain unproven. No live changes.
