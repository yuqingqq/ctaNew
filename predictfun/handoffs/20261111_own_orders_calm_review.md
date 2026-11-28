# Own-order continuation and calm posting: tested, not adopted

As of 2026-11-11 17:43 UTC. Six previously consumed test dates (Oct 26, 27, 31 and Nov 1, 2, 3), 1,728 market windows per policy/scenario. This is exploratory research, not fresh validation.

The tested variants do not meet the objective of preserving trading value while reducing inventory exposure. Keep the existing baseline for this comparison.

| Change from frozen v3 baseline | Common marked windows | Marked-value change | Full settlement-P&L change | Inventory-square-time change |
|---|---:|---:|---:|---:|
| Hold every eligible soft pull briefly | 1,577 | -$18.42 | -$17.51 | +0.34% |
| Learned continuation on admitted orders | 1,578 | -$5.60 | -$12.37 | +0.007% |
| Allow new increasing orders only when calm | 1,579 | -$142.14 | -$305.80 | -97.73% |
| Small early quotes when calm | 1,578 | -$15.77 | -$18.38 | -1.29% |
| Early calm quotes plus learned continuation | 1,577 | -$21.38 | -$30.76 | -1.28% |

Marked changes use paired windows with valid marks at the trading stop; exclusions are 151/150/149/150/151 windows respectively. Settlement comparisons use all 1,728 windows. Inventory-square time is a quantity-exposure proxy, not a measured reduction in dollar-P&L variance. Rebates are excluded.

The early-calm inventory reduction does not persist under the declared conservative timing/queue assumptions: exposure changes +0.013%, marked value -$1.48 on 1,558 common windows (170 excluded). Learned continuation loses $15.51 of marked value and raises exposure 0.85% on 1,559 common windows (169 excluded). These results do not support promotion.

What the order records show:

- **The continuation opportunity is sparse under the current guards.** On the reference neutral path, only 92 of 12,757 eligible test decisions produce any incremental fill in the bounded extension; 12,665 are zero-fill outcomes. All are retained in the model population. The queue model's matched-action null has p=0.735 (500 draws). The market-plus-age diagnostic has p=0.100; neither establishes a useful positive continuation rule. The market-only model was a neutral-action diagnostic, not a full portfolio candidate.
- **The strict calm gate removes most trading.** It has 75 fill records versus 3,095 for baseline, and 419.41 shares versus 18,211.59. Its lower exposure comes with a substantial loss of trading value.
- **Early calm quotes mostly remain unfilled.** Of 3,494 seeded order generations, 13 receive any fill, totaling 54.29 shares. Their own full-generation 1 s / 5 s markouts total about +$0.03 / +$0.23, with no missing marks. These are descriptive subsets, not incremental portfolio gains: early orders can change later admissions, cancellations and inventory. The full strategy results above include those interactions.
- **A 1 s improvement alone is insufficient.** Baseline's strict 1 s / 5 s generation markouts are +$250.42 / +$212.22; continuation's are +$250.48 / +$211.71. Each excludes 3 generations with missing marks (251,221 / 249,850 known generations). These are different policy populations, not a paired causal estimate. The small short-horizon change does not translate into improved marked portfolio value.

The implementation preserves frozen v3 admission, sizing, hard guards and request reservations. Continuation evaluates the first eligible pull of an accepted touch order with at most 5 client-known remaining shares, within 0.01c below the pull line, and can extend it for at most 1 s. Calm entry either gates fresh increasing orders or permits a 5-share early touch quote in the first 60 s within that same small band. The results do not test arbitrary calm-period quoting or an unrestricted replacement guard.

Training uses expanding earlier-date folds, with duplicate actions shared by the neutral reference and seed paths counted once. The existing admission guard uses cached LODO predictions, so the full strategy is not a historical out-of-sample test. Queue/fill timing remains uncalibrated; the current live incremental-size/soft-pull implementation is not this frozen v3 baseline.

Verification completed:

- 9/9 neutral baseline rebuilds match saved window/fill/request artifacts exactly. All primary/stress portfolios reconcile independently to fills, cash, settlement and inventory exposure.
- 21 original/deduplication controls pass, plus causal warm-up and clock falsifiers. A new-harness warm-up bug was corrected with a superseding wrapper. All 70,115 saved neutral/primary feature decisions were unaffected; a direct cache-row audit independently verifies every derivative and complete 50 ms grids from window start.
- All 12 saved models reproduce their stored predictions; altered-score falsifiers are refused. The report's NumPy-scalar JSON issue was corrected without changing numerical results. Original frozen sources and failed output are retained.

Canonical results: [complete tables](20261111_own_orders_calm_results_v4.md), [machine receipt](../reviews/own_orders_calm_20261111/comparison_v4.json), [order populations and model identities](../reviews/own_orders_calm_20261111/mechanism.json), [direct cache audit](../reviews/own_orders_calm_20261111/grid_check.json). Use comparison_v4.json; the original data-directory comparison.json is an intentionally preserved incomplete serialization attempt.

Source is on research/predict-own-orders-calm-20261111. Canonical stages are the original neutral builder, unique-action trainer v2, primary engine replays, causal-wrapper v3 stress replays, native-JSON report v4 and the direct grid checker. Original runners remain frozen provenance; output paths refuse overwriting. No live or frozen-engine changes were made.
