# Queue continuation model: controlled research comparison

As of 2026-11-11T15:49:34.781514+00:00. All dates were previously consumed; exploratory results.

**Conclusion: this candidate does not meet the objective of preserving trading value while reducing inventory risk.** Queue estimates improve some comparisons, but the benefit changes with timing assumptions and inventory exposure does not consistently fall. The primary action-matched null gives p=0.236 after adding age to the baseline (500 shuffles, three scenarios).

The fixed-entry policies all lose money. In the primary scenario over all1,728 test windows, settlement PnL is -$1,186.64 for market+age, -$1,047.29 for queue, and -$862.99 for cancelling at the scheduled decision. The last baseline still bears entry fills and cancellation latency. A continuation model cannot undo those already incurred fills. These are controlled fixed-entry results; full v3 integration remains untested.

Next experiment: preserve the current strategy's selective entry guard, generate neutral order lifecycles from its actual admissions, and compare learned continuation with unchanged keep/cancel rules at those same decisions. Keep age in every baseline, use conservative out-of-sample value estimates before choosing an action, and retain the live-fill calibration requirement. The current prototype is not promoted.

Built KEEP/CANCEL/same-price REJOIN value models using market conditions, known inventory and remaining size, with or without causal queue estimates. Orders are fixed at5 shares on a shared entry schedule. Cancellation/acknowledgment/fill-notice latency and pending exposure are included. Training expands chronologically from Oct23–25 to six test days: Oct26/27/31 and Nov1/2/3.

The additional market_age baseline includes order age, isolating queue-size/progress information from remaining lifetime. Its source and design were committed before inspecting portfolio returns. Frozen v3 and live code are unchanged.

## Primary conservative queue assumption (L50)

| Policy | Marked PnL, common windows | Settlement PnL, common windows | Inventory RMS, all windows | Mean terminal absolute inventory | Orders |
|---|---:|---:|---:|---:|---:|
| always_keep | $-1484.64 | $-1543.12 | 6.686 sh | 4.178 sh | 46,099 |
| always_cancel | $-705.90 | $-771.44 | 6.148 sh | 4.663 sh | 48,776 |
| legacy_score | $-699.60 | $-730.48 | 6.295 sh | 4.858 sh | 48,275 |
| market | $-1047.52 | $-1044.62 | 6.398 sh | 4.512 sh | 53,462 |
| market_age | $-990.83 | $-1004.60 | 6.387 sh | 4.513 sh | 53,320 |
| queue | $-931.70 | $-880.44 | 6.409 sh | 4.520 sh | 53,200 |

Population: 1,376 common observable stop-mark windows out of 1,728 scheduled test windows across 6 complete UTC days. Inventory and order counts cover all scheduled windows. Raw full-settlement totals, spread/markouts, all scenarios, and exclusion counts are in comparison.json and the original report.json.

## Queue model versus age-matched market baseline

| Assumption | Common marked delta | Full settlement delta | Inventory-square time change |
|---|---:|---:|---:|
| L50 | $+59.13 | $+139.35 | +0.69% |
| M50 | $+23.21 | $+102.90 | -0.41% |
| L250slow | $-11.50 | $-12.41 | -1.57% |

Computed evidence rule supports incremental queue improvement: **False**. See decision_checks in comparison.json; missing complete marked values prevent a positive primary decision.

## Checks and limits

15 synthetic controls passed. The superseding accounting verifier reconciled 31,104 portfolio windows and 264,795 selected fills; the age baseline independently reconciled 5,184 more windows. Nulls use500 action-matched shuffles; intervals are clustered on UTC day.

Accounting v2 supersedes a floating-point bug: a roughly1e-15-share residue made an economically flat portfolio lose its known cash mark when the book mark was missing. Stable summation and1e-9-share zero tolerance restore those marks. Original model decisions, labels, fills, failed verifier and report are preserved. All comparisons here use the superseding receipt.

This is a fixed-entry continuation experiment, not a full-v3 comparison. Queue position and fill timing remain uncalibrated against own live fills. RuleL credits trades only; ruleM additionally caps queue ahead by observed same-price size. Slower timings are a separate sensitivity. Maker fees were verified zero in input registry; rebates are excluded. Legacy-score baseline uses prior LODO predictions and is diagnostic only. Markout/spread sums with missing-mark groups are partial, not complete trading PnL.

Reproduction: source68769426; original freeze9f8b09a8; age-control source recorded in its declaration. Use run_queue_model.sh controls/declare, commit declaration, build each declared day, then fit. Run pf_queue_accounting_v2, pf_queue_age_control declare (commit)/run, and pf_queue_report through run_light.sh. The original all/verify command includes the superseded verifier; accounting_v2 is the authoritative verification. Existing output directories intentionally refuse overwrites.
