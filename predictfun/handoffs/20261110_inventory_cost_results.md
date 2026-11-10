# Inventory-cost experiments: capture versus exposure

**Complete and independently checked. None of the three new policies meets the declared objective against the original baseline under either J or X.**

As of 2026-11-10T04:22:23.315164+00:00. Main population: nine consumed UTC days, 2,592 markets per arm per timing; five arms, 18 main replays. Stress: one consumed day, 288 markets per arm. No excluded markets or missing valuations in these comparisons. These are conditional simulator results, not independent live validation.

## Findings

- Relaxing the exit-score cutoff recovers a little inventory reduction relative to the reservation-only control, but both exit variants leave **more** terminal inventory than the original baseline and reduce short-term trading value. These arms include the late-reservation restriction; their full difference from baseline is not caused solely by the relaxed cutoff.
- Marginal entry reduces mean terminal inventory by **5.4% / 8.6%** and inventory-square exposure by **14.0% / 20.3%** (J / X). It also loses **15.2% / 16.8%** of +1s value while reducing traded shares by **17.5% / 17.7%**. This does not preserve capture.
- Marginal entry raises observed settlement P&L by **$79.85 / $98.36**, but this is not extra spread capture. Under J, mark-at-stop falls $6.91 and carry improves $86.76. Under X, mark-at-stop improves $58.49 and carry improves $39.87, despite lower +1s value. Both settlement-difference intervals include zero.
- Average inventory reduction is not a uniform tail-risk reduction: the marginal-entry maximum rises from **75 to 83.74 shares under J**, while falling from 65 to 59.48 under X. Daily P&L standard deviation is lower in this sample, but nine consumed days do not establish a variance improvement.
- Budget exit commits **$91.82 / $80.23** of lifetime quote-concession allowance but executes only **$0.798 / $0.668** of concessions. Canceled/rejected/unfilled intents consume allowance without refunds; committed allowance is not a cash loss. This particular conservative budget barely exercises paid exits, so it does not rule out better budget accounting.

## Full main comparison

USD totals over all nine days. The signed +1s move is favorable when positive; +1s value = spread + signed move. Mark and carry sum to settlement P&L.

| Timing | Arm | Spread $ | Signed +1s move $ | +1s value $ | Mark at stop $ | Carry $ | Settlement $ |
|---|---|---:|---:|---:|---:|---:|---:|
| J | Baseline | 422.23 | 180.60 | 602.83 | 677.12 | -163.06 | 514.06 |
| J | Late reservation | 400.60 | 157.37 | 557.97 | 658.36 | -156.32 | 502.04 |
| J | Soft exit | 403.80 | 154.14 | 557.94 | 595.03 | -167.23 | 427.81 |
| J | Budget exit | 404.13 | 153.15 | 557.28 | 598.76 | -161.76 | 437.00 |
| J | Marginal entry | 353.54 | 157.53 | 511.07 | 670.20 | -76.30 | 593.91 |
| X | Baseline | 355.51 | 108.04 | 463.56 | 431.16 | -113.54 | 317.63 |
| X | Late reservation | 338.30 | 102.23 | 440.54 | 385.42 | -75.48 | 309.94 |
| X | Soft exit | 339.66 | 98.03 | 437.69 | 368.68 | -73.61 | 295.07 |
| X | Budget exit | 339.54 | 99.20 | 438.74 | 371.53 | -40.95 | 330.58 |
| X | Marginal entry | 292.02 | 93.89 | 385.91 | 489.66 | -73.67 | 415.99 |

| Timing | Arm | Filled shares | +1s $ / 100 shares | Mean terminal absolute shares | Mean inventory-square | Mean seconds above 10 | Maximum absolute shares | Daily P&L SD $ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| J | Baseline | 34,954.7 | 1.725 | 6.557 | 168.162 | 64.480 | 75.000 | 134.511 |
| J | Late reservation | 33,124.4 | 1.684 | 7.121 | 176.497 | 67.549 | 75.000 | 137.596 |
| J | Soft exit | 33,525.0 | 1.664 | 6.966 | 173.050 | 66.865 | 75.000 | 137.311 |
| J | Budget exit | 33,533.9 | 1.662 | 6.963 | 172.853 | 66.807 | 75.000 | 134.712 |
| J | Marginal entry | 28,843.0 | 1.772 | 6.206 | 144.619 | 59.459 | 83.740 | 104.649 |
| X | Baseline | 28,681.8 | 1.616 | 5.720 | 144.533 | 53.383 | 65.000 | 103.088 |
| X | Late reservation | 27,429.3 | 1.606 | 6.154 | 150.925 | 55.936 | 65.000 | 103.120 |
| X | Soft exit | 27,671.7 | 1.582 | 6.060 | 148.346 | 55.558 | 65.000 | 107.131 |
| X | Budget exit | 27,710.3 | 1.583 | 6.045 | 148.275 | 55.453 | 65.000 | 103.238 |
| X | Marginal entry | 23,599.4 | 1.635 | 5.227 | 115.241 | 48.741 | 59.480 | 85.825 |

FIFO attribution is included for completeness; it includes price drift and should not be called pure spread.

| Timing | Arm | FIFO paired P&L $ | Residual P&L $ | Paired shares | Places | Cancels |
|---|---|---:|---:|---:|---:|---:|
| J | Baseline | 134.02 | 380.04 | 8979.38 | 372302 | 371743 |
| J | Late reservation | 99.78 | 402.26 | 7333.47 | 355921 | 355395 |
| J | Soft exit | 70.09 | 357.72 | 7734.09 | 367304 | 366737 |
| J | Budget exit | 63.28 | 373.72 | 7742.99 | 367331 | 366745 |
| J | Marginal entry | 50.65 | 543.26 | 6378.66 | 368869 | 368310 |
| X | Baseline | 173.54 | 144.09 | 6927.75 | 371461 | 370904 |
| X | Late reservation | 133.41 | 176.52 | 5739.24 | 357258 | 356727 |
| X | Soft exit | 113.06 | 182.01 | 5981.65 | 367270 | 366709 |
| X | Budget exit | 111.22 | 219.35 | 6020.20 | 367270 | 366697 |
| X | Marginal entry | 94.17 | 321.82 | 5025.60 | 371227 | 370673 |

## Changes versus both controls

All money differences are nine-day totals. Inventory is the difference in mean terminal absolute shares. The objective requires preserved +1s value AND preserved mark-at-stop AND less terminal inventory.

| Timing | Candidate minus reference | Δ +1s $ | Δ mark $ | Δ settlement $ | Δ terminal shares | Objective met |
|---|---|---:|---:|---:|---:|---|
| J | Soft exit − Baseline | -44.89 | -82.08 | -86.25 | +0.4093 | No |
| J | Soft exit − Late reservation | -0.03 | -63.33 | -74.23 | -0.1546 | No |
| J | Budget exit − Baseline | -45.55 | -78.36 | -77.05 | +0.4058 | No |
| J | Budget exit − Late reservation | -0.68 | -59.60 | -65.04 | -0.1580 | No |
| J | Marginal entry − Baseline | -91.76 | -6.91 | +79.85 | -0.3512 | No |
| J | Marginal entry − Late reservation | -46.90 | +11.84 | +91.87 | -0.9151 | No |
| J | Budget exit − Soft exit | -0.66 | +3.72 | +9.19 | -0.0034 | No |
| X | Soft exit − Baseline | -25.86 | -62.48 | -22.55 | +0.3403 | No |
| X | Soft exit − Late reservation | -2.84 | -16.74 | -14.86 | -0.0935 | No |
| X | Budget exit − Baseline | -24.82 | -59.63 | +12.95 | +0.3255 | No |
| X | Budget exit − Late reservation | -1.79 | -13.89 | +20.64 | -0.1084 | No |
| X | Marginal entry − Baseline | -77.65 | +58.49 | +98.36 | -0.4931 | No |
| X | Marginal entry − Late reservation | -54.63 | +104.24 | +106.05 | -0.9269 | No |
| X | Budget exit − Soft exit | +1.05 | +2.85 | +35.50 | -0.0149 | Yes |

The only positive objective flag is budget exit versus soft exit under X; it does not beat either original control on the full objective, and the same contrast fails under J.

The reservation-only control already loses 5.0–7.4% of +1s value and leaves
7.6–8.6% more terminal inventory than baseline. Narrow score relief recovers
some closing flow from this control, while worsening mark-at-stop. This is a
strategy sizing comparison; it does not justify omitting pending exposure from
the separate safety accounting.

For marginal entry versus baseline, day-bootstrap 95% settlement intervals are **−$23.40 to +$39.22/day (J)** and **−$25.01 to +$43.28/day (X)**. Inventory-square differences are more consistent within this sample: the declared 42-comparison max-T p-values are 0.014 (J) and 0.006 (X). These do not turn previously consumed, partly training data into independent validation. Full intervals and all 42 comparisons are in the receipt.

## Exit-feasibility audit

This is the unchanged baseline during seconds 60–240 while inventory is nonzero. J: 223,649.27 sampled state-seconds, 783,824 active episodes; X: 196,431.75 seconds, 691,005 active episodes. Each timing contains 2,592 markets. Episodes and repeated decisions are not independent samples. As-of for the post-hoc intersection audit: 2026-11-10T04:24:32.435064+00:00.

Overlapping flags occur during approximately 86% of inventory time for the score cutoff, 47% for an off-touch quote, and 30–31% for minimum notional. Those percentages must not be added. The following mutually exclusive intersections are more useful:

| Remaining blockers after other feasibility checks | J share of inventory time | X share |
|---|---:|---:|
| None; legal desired quote possible | 1.25% | 1.25% |
| Off-touch price only | 1.38% | 1.38% |
| Score cutoff only | 28.29% | 28.42% |
| Score cutoff AND off-touch price | 27.96% | 27.91% |
| Other blockers, including combinations with minimum notional | 41.12% | 41.04% |

**A price-only repair exposes about 1.38%, not 47%, of inventory time.** About 28% is jointly blocked by score and price even when the other diagnostic checks permit an order. Removing either condition alone does not create an executable closing quote. These are eligibility diagnostics, not forecasts of added fills or P&L. The intersection breakdown is explicitly post hoc; no policy was selected or replayed on it.

## Slower-feedback sensitivity

November 3 only, 288 markets/arm, X transaction timing, place 80 ms / fill notice 300 ms; descriptive, no intervals. Main X on this same day has baseline +1s $46.62 and settlement −$19.84; slowing feedback materially changes the path.

| Arm | +1s $ | Mark $ | Settlement $ | Mean terminal absolute shares | Inventory-square |
|---|---:|---:|---:|---:|---:|
| Baseline | 22.722 | -132.046 | -102.351 | 4.599 | 108.237 |
| Late reservation | 20.947 | -147.302 | -117.514 | 4.813 | 110.386 |
| Soft exit | 21.187 | -150.107 | -125.234 | 4.792 | 109.812 |
| Budget exit | 21.187 | -150.107 | -125.234 | 4.792 | 109.812 |
| Marginal entry | 15.308 | -82.022 | -62.063 | 3.227 | 66.823 |

Marginal entry again gives up +1s value while reducing average exposure; the paid-exit arm executes no concession on this stress day. This is sensitivity evidence, not calibration.

## Decision and next research focus

Keep the original entry policy as the capture reference. None of these three settings earns promotion for the stated objective. The entry rule improves value per traded share slightly, but loses too much total flow; the observed settlement gain is insufficient justification to adopt it.

The useful follow-up is a jointly executable reducing quote, evaluated against its explicit cost and risk reduction. A blanket lower score line or a price-only change will miss overlapping constraints. Any further test should preserve entry decisions and keep stale-data, book, notional, stop and no-reversal checks.

The concession budget also needs a separate design comparison: reserve the worst possible cost of outstanding orders plus confirmed filled concessions, and release unused allowance only after cancellation and late-fill uncertainty are resolved. The current lifetime no-refund budget is deliberately stricter. This alternative is **not implemented or tested here** and depends on defensible fill-notice assumptions.

Actual own-fill/queue calibration and native incremental historical parity remain necessary before a live-performance claim. These results support rejecting the tested settings and narrowing the next experiment; they do not establish an improved live strategy.

## Completed checks

- 33 policy/adapter controls; eight-market smoke; all 18 main replays and the five-arm stress replay reconciled.
- Exact baseline and late-reservation artifact parity under both main timings; exit-only pre60 fills/requests unchanged; no new post60 reverse-opening fills; per-market concession limit and fill provenance verified.
- Reporting controls: read-only saved-check comparison plus four refusals; explicit pandas audit-column selection plus four refusals. Original failed logs/declarations retained.
- Independent verifier: all 90 main arm-day outputs, fill/inventory integrals, aggregate arithmetic and objective flags; positive controls and five refusals. Joint audit: positive intersection/duration controls and four refusals.
- Final report: `reviews/inventory_cost_20261110/receipt_v4.json`; independent evidence: `verification_v4.json`; descriptive follow-up: `joint_audit.json`.
- Corrected report peak RSS 504 MiB; independent verifier 116 MiB; joint audit 309 MiB. Full runs remained below the 4 GiB process cap. New v2 data uses approximately 139 MiB. All jobs finished; recheck the shared slot before starting new work.


## Scope and controls

Nine previously consumed dates: October 23–27, October 31, and November 1–3,
2026. Each main arm has 2,592 markets under each of two fill assumptions:
J uses frame-jitter transaction timing; X shifts transactions 50 ms earlier and
uses maximum activation queue. The separate November 3 X stress doubles create
delay to 80 ms and increases fill-notice delay to 300 ms. These are sensitivity
assumptions, not measurements of the current live system.

The baseline and the reservation-only control are freshly replayed and checked
exactly against the earlier artifacts. All exit variants preserve the baseline's
pre-60-second fills and requests. Each variant then follows its own complete
fill/inventory path. No protected November 9–13 data is used.

| Arm | Change |
|---|---|
| Baseline | Frozen v3 with the existing historical execution adapter |
| Late reservation | After 60 seconds, reserve closing room for outstanding orders and unresolved late fills |
| Soft exit | Late reservation plus a 0.01-score reduction in the reducing-side PM90 cutoff during seconds 60–240 |
| Budget exit | Soft exit plus at most one tick inside the spread during seconds 180–240 when absolute known inventory is at least 10 shares |
| Marginal entry | Late reservation after 60 seconds; before then, clip entries according to their expansion of the full possible inventory range and the available guard-score premium |

Budget exit has a lifetime **$0.25 submitted-concession budget per market**,
including canceled, rejected and still-unfilled submissions, without refunds.
The budget limits deliberately quoted price concession relative to the original
quote; it does not bound adverse selection or total loss. Filled concessions
are already included in trade prices and are not subtracted from P&L again.
The existing 150-share backstop is unchanged. As in the prior research, the
aggregate dollar loss cap is disabled and rebates are excluded.

The marginal-entry setting is one fixed score premium per increase in possible
inventory radius squared. It is neither a probability model nor a calibrated
dollar valuation. Pending Up and Down orders count separately, including
creates, cancellations and unresolved fill notifications.

The marginal-entry policy also retains a feasible same-price order remainder.
Its comparison therefore measures the combined sizing/retention heuristic, not
the isolated causal effect of the risk premium. This historical single-order
adapter does not reproduce the current native incremental top-up mechanism.

## Reading the comparison

Spread is measured against the strictly prior midpoint at each fill. The +1s
value includes the subsequent signed midpoint movement; a positive movement
term means favorable selection. Mark-at-stop values all
positions at the common 240.1-second boundary; it is not an executable exit
price. Carry is the further change from that mark to binary settlement.
FIFO paired profit includes price drift between fills and is not pure spread.

Inventory-square is the time average of squared net shares over each full
300-second market. It measures exposure, not realized P&L variance. Terminal
inventory is mean absolute net shares across all markets, including zero-fill
markets. The declared objective requires both +1s value and mark-at-stop to be
preserved while terminal absolute inventory falls.

The baseline exit audit retains every sampled reference state. Consecutive
identical states are compressed into episodes. Episode durations carry the
last observed state to the next decision; they are not continuous queue
observations or independent submitted-order generations. First blocking reasons
partition the diagnostic population; overlapping flags do not. Execution-stage
throttling is not independently attributed by this audit.

Day-level uncertainty uses 2,000 paired bootstrap samples and 2,000 common sign
flips over nine UTC days, with a declared family of 42 comparisons. The one-day
stress is descriptive only. All dates were consumed previously, and eight were
used in guard training; intervals cannot turn these into independent validation.

## Provenance and verification

Initial policy/source freeze: `cc109212`; initial declaration: `d5f43299`.
The first full-day replay refused a discrepancy in the diagnostic checker:
raw quote notional could pass $1 while lot-rounded submission notional failed.
Correction `542bb9e4` distinguishes these gates without changing any policy
parameter. Superseding declaration `42dfcadf` preserves the original refusal.
The corrected suite has 33 passing source controls and an exact eight-market
smoke replay. Independent saved-artifact verifier: `deb4ad6e` (v3 output path in
`dbb96f30`).

Before aggregate results, source review found that the original report called
the runner's verifier, which attempted an exclusive second creation of existing
`checks.json` files. Adapter `78d7a610` compares the recomputed payloads and hashes
read-only, ignoring only their new verification timestamp. It preserves the
original policy calculations and saved checks; its controls include a positive
comparison and four deliberately bad inputs. The reporting correction is
declared separately. The v3 report then exposed pandas `.flags` metadata
access instead of the audit column. Source `dbb96f30` and declaration `daca5b2b`
supersede that reporting instrument with explicit column access and regression
controls; successful outputs are v4. Economic calculations and replay artifacts
are unchanged. The post-hoc intersection audit is source `72461f72`.

Source: `predictfun/research/pf_inventory_cost_*.py` and
`predictfun/research/run_inventory_cost.sh` on branch
`research/predict-inventory-design-20261110`.
Data/logs: `data/predict_fun/research/inventory_cost_20261110_v2/`.
Compact declaration and eventual receipts:
`predictfun/reviews/inventory_cost_20261110/`.

The simulator still uses historical single-order execution, not current native
incremental order management. Actual fill timing and queue behavior await the
new wallet-event capture. No live strategy change has been made; the historical
comparison alone cannot establish a deployable improvement.
