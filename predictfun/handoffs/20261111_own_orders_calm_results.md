# Own-order continuation and calm entry
As of 2026-11-11T17:40:25.033365+00:00. Six consumed dates: 20261026, 20261027, 20261031, 20261101, 20261102, 20261103. Exploratory, not untouched validation.

The neutral engine uses the frozen v3 admissions. Every build reproduces the saved reference win/fill/request artifacts exactly. All sent generations are traced, including unfilled and rejected orders. Continuation is trained chronologically on first soft-pull decisions; cached admission-guard predictions are LODO, so this is conditional chronology, not a historically deployable full-strategy backtest.

| Timing | Policy | Windows / valid marks | Settlement $ | Marked $ | Carry on marked windows $ | 1s markout $ | Inventory RMS sh | End abs mean sh | Places | Kept | Seeded |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| X | reference | 1728 / 1579 | 324.30 | 152.63 | 46.24 | 250.74 | 11.800 | 5.737 | 251224 | 0 | 0 |
| X | hold_rule | 1728 / 1580 | 306.79 | 122.30 | 42.30 | 245.67 | 11.820 | 5.770 | 246905 | 9202 | 0 |
| X | continuation | 1728 / 1579 | 311.92 | 133.38 | 37.32 | 250.80 | 11.800 | 5.748 | 249853 | 3013 | 0 |
| X | calm_filter | 1728 / 1722 | 18.49 | 10.49 | -1.31 | -0.14 | 1.777 | 0.183 | 15380 | 0 | 0 |
| X | calm_seed | 1728 / 1579 | 305.91 | 150.56 | 46.94 | 251.01 | 11.724 | 5.678 | 252714 | 0 | 3494 |
| X | calm_seed_continuation | 1728 / 1579 | 293.54 | 131.30 | 38.02 | 251.07 | 11.724 | 5.689 | 251378 | 2993 | 3464 |
| stress | reference | 1728 / 1559 | -220.66 | -161.00 | 43.09 | -412.95 | 13.671 | 6.725 | 252803 | 0 | 0 |
| stress | continuation | 1728 / 1559 | -233.60 | -176.51 | 45.26 | -421.51 | 13.729 | 6.751 | 251591 | 1748 | 0 |
| stress | calm_seed | 1728 / 1558 | -223.09 | -162.48 | 43.30 | -414.04 | 13.672 | 6.725 | 253954 | 0 | 3178 |
| stress | calm_seed_continuation | 1728 / 1558 | -239.68 | -179.89 | 43.72 | -422.30 | 13.728 | 6.751 | 252887 | 1734 | 3164 |

Marked totals have different missing-mark populations; use the paired common-window deltas below. Carry + marked equals settlement only on that marked population. Engine 1s/60s markouts retain its historical last-valid-book convention; strict 1s/5s generation markouts and missing counts are in comparison.json. Markouts are not additive to settlement P&L. Rebates excluded.

| Comparison | Common marks / windows | Marked delta $ [day interval] | Full settle delta $ | Inventory-square delta % |
|---|---:|---:|---:|---:|
| X:hold_rule-reference | 1577 / 1728 | -18.42 [-43.88, -2.48] | -17.51 | +0.338 |
| X:continuation-reference | 1578 / 1728 | -5.60 [-10.68, -1.25] | -12.37 | +0.007 |
| X:calm_filter-reference | 1579 / 1728 | -142.14 [-407.65, +110.08] | -305.80 | -97.733 |
| X:calm_seed-reference | 1578 / 1728 | -15.77 [-63.00, +16.60] | -18.38 | -1.288 |
| X:calm_seed_continuation-reference | 1577 / 1728 | -21.38 [-73.09, +13.59] | -30.76 | -1.281 |
| X:calm_seed_continuation-calm_seed | 1578 / 1728 | -5.60 [-10.68, -1.25] | -12.37 | +0.007 |
| stress:continuation-reference | 1559 / 1728 | -15.51 [-34.34, +1.25] | -12.94 | +0.849 |
| stress:calm_seed-reference | 1558 / 1728 | -1.48 [-19.10, +13.76] | -2.43 | +0.013 |
| stress:calm_seed_continuation-reference | 1558 / 1728 | -18.89 [-44.79, +8.06] | -19.02 | +0.836 |
| stress:calm_seed_continuation-calm_seed | 1558 / 1728 | -17.41 [-39.99, +1.25] | -16.59 | +0.822 |

Intervals resample six UTC days, 1,000 draws. They do not correct for historical strategy selection. Primary X: trade anchor -50ms, place40/cancel22/notice100ms, queue M. Stress: -250ms,80/100/250ms, queue L. No fill/queue calibration claim.

| Neutral-action diagnostic | n / keep | Observed value c | Matched null mean c | p >= observed |
|---|---:|---:|---:|---:|
| reference:queue | 12757 / 3689 | -123.72 | -89.51 | 0.735 |
| reference:market | 12757 / 3752 | -12.56 | -85.29 | 0.100 |
| calm_seed:queue | 12509 / 3645 | -123.72 | -89.68 | 0.695 |
| calm_seed:market | 12509 / 3694 | -9.31 | -85.99 | 0.086 |
| calm_entry_reference | 183696 / 6858 | -179.12 | -49.17 | 0.747 |

500 matched-count shuffles per diagnostic. Continuation matches day/side/hour/age bucket/inventory sign; the market control includes age. Calm entry matches day/side/hour/size. All include unfilled zero outcomes. Shadow labels are overlapping action counterfactuals, not portfolio P&L; existing competitors/pending cancels are represented, future reference admissions are not forked. Full replays above test interactions.

Computed predicates: {"calm_seed": {"marked_interval_positive": false, "primary_improves_trading_reduces_exposure": false, "stress_improves_trading_reduces_exposure": false}, "calm_seed_continuation": {"marked_interval_positive": false, "primary_improves_trading_reduces_exposure": false, "stress_improves_trading_reduces_exposure": false}, "continuation": {"marked_interval_positive": false, "primary_improves_trading_reduces_exposure": false, "stress_improves_trading_reduces_exposure": false}}

No promotion: these are consumed dates and the live incremental-size/soft-pull implementation is not the frozen v3 baseline. Source/declaration, per-day checks, complete statuses, model fold counts, artifact hashes and all comparisons are in reviews/own_orders_calm_20261111/ and data/predict_fun/research/own_orders_calm_20261111/.
