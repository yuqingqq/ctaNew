# PRESSURE-ACTION-ATTRIBUTION-v1 — report

2026-11-21 20:26Z, session aea5bfb9. This is review `7fca0db4`'s ONE next priority: one 2 x 2 ablation of the original nominal-clock pressure rule (`793c706b`, results `40a3ee8d`).
- **Bits:** C = pressure PULL of a would-be-KEEP live order; W = pressure WAIT of a would-be-POST slot.
- **Population:** the same 19 complete markets of 96 (statuses: incomplete:missing_endpoint 54, incomplete:stale_future_book 23, complete 19).
- **Freeze and checker:** freeze `8f618ea6`, before any ablated replay; checks v2 `d4b843c3` supersedes v1 (see Validity).
- **Multiplicity:** two new real arms, PULL_ONLY_10 and WITHHOLD_ONLY_01. FULL_11 and BASELINE_00 are controls.
- **Status of the evidence:** consumed Oct 24-27 development days. There is no interval, winner, promotion, threshold, feature or further candidate, and no profitability claim.

## Arms and controls

- **FULL_11:** reproduces the accepted pressure arm on 19 of 19 markets: fingerprint, fills, orders, requests, value, log, command array, queue records.
- **BASELINE_00:** the pressure receipt's disabled arm, equal to the original ladder (checks v2 C). The selftest proves that the 00 class delegates exactly to LadderInventory.
- **PULL_ONLY_10 / WITHHOLD_ONLY_01:** each acts only through its own component, at ON ticks only. A pull-only arm may submit during pressure, as declared (checks v2 D).
- **Parts:** pilot (first two complete markets per date) and rest (the other 11; Oct 27's rest is DAY_EMPTY). They are disjoint and cover the 19 markets.

## Arm totals

Gross, zero cost, in cents. END = cash + inventory × the END mark. The +1 s figures are on the shared cutoff t <= END - 1 s.

| arm | END | filled shares | execution edge | +50 ms | +100 ms | eligible +1 s | movement to END | late / missing | valid +1 s shares | censored | +1 s per share | ask per share | bid per share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASELINE_00 | -209,557.7 c | 96,991.1 | +39,326.2 c | -91,748.0 c | -102,657.2 c | -203,083.9 c | -7,617.2 c | +1,143.4 c | 96,604.0 | 0 | -2.102 c / share | -2.409 c / share | -1.792 c / share |
| PULL_ONLY_10 | -132,118.6 c | 85,714.4 | +50,179.1 c | -39,366.7 c | -44,392.0 c | -113,644.6 c | -18,945.5 c | +471.5 c | 85,369.1 | 0 | -1.331 c / share | -1.351 c / share | -1.311 c / share |
| WITHHOLD_ONLY_01 | -191,208.9 c | 96,754.0 | +42,057.6 c | -88,144.4 c | -98,609.7 c | -192,679.1 c | +341.7 c | +1,128.6 c | 96,352.5 | 0 | -2.000 c / share | -2.242 c / share | -1.755 c / share |
| FULL_11 | -86,109.6 c | 79,399.5 | +50,544.4 c | -27,654.1 c | -28,710.7 c | -85,164.3 c | -1,416.8 c | +471.5 c | 79,054.1 | 0 | -1.077 c / share | -0.999 c / share | -1.157 c / share |

- **all_arms_negative:** True. By day: 20261024 True, 20261025 True, 20261026 True, 20261027 True. Every arm's END is below zero; the contrasts below are relative gains only.

## The declared contrasts

Each is classified against zero incremental value.

| contrast | END | END total | eligible +1 s | eligible +1 s total | +1 s per share | per-share total | markets better | worse | equal | top-3 share of the absolute END contrasts |
|---|---|---|---|---|---|---|---|---|---|---|
| cancellation added to withholding (R11 - R01) | POSITIVE_EVERY_DAY | +105,099.3 c | POSITIVE_EVERY_DAY | +107,514.8 c | POSITIVE_EVERY_DAY | +0.922 c / share | 18 | 1 | 0 | 41.48 % |
| withholding added to cancellation (R11 - R10) | POSITIVE_EVERY_DAY | +46,009.0 c | POSITIVE_EVERY_DAY | +28,480.3 c | POSITIVE_EVERY_DAY | +0.254 c / share | 14 | 5 | 0 | 69.22 % |
| standalone cancellation (R10 - R00) | POSITIVE_EVERY_DAY | +77,439.1 c | POSITIVE_EVERY_DAY | +89,439.3 c | POSITIVE_EVERY_DAY | +0.771 c / share | 17 | 2 | 0 | 35.20 % |
| standalone withholding (R01 - R00) | POSITIVE_EVERY_DAY | +18,348.8 c | POSITIVE_EVERY_DAY | +10,404.8 c | POSITIVE_EVERY_DAY | +0.103 c / share | 14 | 1 | 4 | 64.34 % |

- **Interaction** (R11 - R10 - R01 + R00):
  - END POSITIVE_EVERY_DAY, total +27,660.1 c; by day 20261024 +1,374.5 c, 20261025 +489.4 c, 20261026 +13,879.9 c, 20261027 +11,916.3 c;
  - eligible +1 s POSITIVE_EVERY_DAY, total +18,075.5 c.
- **Sums:** the standalone effects are never summed as the full effect without the interaction. FULL_11 - BASELINE_00 = cancel alone + withhold alone + interaction.

### END contrasts by day

| contrast | 20261024 | 20261025 | 20261026 | 20261027 |
|---|---|---|---|---|
| cancellation added to withholding (R11 - R01) | +26,944.4 c | +27,570.6 c | +33,358.4 c | +17,225.8 c |
| withholding added to cancellation (R11 - R10) | +5,706.9 c | +9,792.7 c | +16,646.5 c | +13,862.9 c |
| standalone cancellation (R10 - R00) | +25,569.9 c | +27,081.2 c | +19,478.6 c | +5,309.5 c |
| standalone withholding (R01 - R00) | +4,332.3 c | +9,303.3 c | +2,766.7 c | +1,946.5 c |
| interaction (R11 - R10 - R01 + R00) | +1,374.5 c | +489.4 c | +13,879.9 c | +11,916.3 c |

### +1 s per share contrasts by day

Each arm's pooled valid-share denominator is in the arm table.

| contrast | 20261024 | 20261025 | 20261026 | 20261027 |
|---|---|---|---|---|
| cancellation added to withholding (R11 - R01) | +0.669 c / share | +1.194 c / share | +0.626 c / share | +1.421 c / share |
| withholding added to cancellation (R11 - R10) | +0.162 c / share | +0.295 c / share | +0.085 c / share | +0.773 c / share |
| standalone cancellation (R10 - R00) | +0.604 c / share | +0.998 c / share | +0.619 c / share | +0.786 c / share |
| standalone withholding (R01 - R00) | +0.097 c / share | +0.100 c / share | +0.078 c / share | +0.139 c / share |

### Fills lost and added per contrast

Fills are matched by the multiset of (t, side, price, quantity); candidate - reference END = added - lost.

| contrast | lost fills | lost shares | lost +1 s > 0 | lost +1 s < 0 | lost END | added fills | added shares | added +1 s > 0 | added +1 s < 0 | added END | candidate - reference END |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cancellation added to withholding (R11 - R01) | 3,336 | 35,794.3 | +23,683.2 c | -153,978.7 c | -112,432.4 c | 1,594 | 18,439.7 | +12,064.7 c | -34,845.3 c | -7,333.1 c | +105,099.3 c |
| withholding added to cancellation (R11 - R10) | 1,374 | 14,990.2 | +9,747.0 c | -45,982.4 c | +2,222.4 c | 770 | 8,675.3 | +8,067.0 c | -15,822.0 c | +48,231.4 c | +46,009.0 c |
| standalone cancellation (R10 - R00) | 3,109 | 32,296.2 | +21,187.9 c | -151,473.4 c | -65,094.2 c | 1,902 | 21,019.6 | +13,038.1 c | -53,884.2 c | +12,345.0 c | +77,439.1 c |
| standalone withholding (R01 - R00) | 835 | 9,124.8 | +5,553.0 c | -32,960.3 c | +36,865.5 c | 766 | 8,887.7 | +7,341.7 c | -24,344.2 c | +55,214.3 c | +18,348.8 c |

## Breakouts per arm (all days)

`withheld proposals` count slot decisions, not orders. Inventory is exact from the fills; uptime is exact from the order intervals.

| arm | pressure pulls | withheld proposals | reducing-side | inventory PULL requests | structural requests | generations | reviewed | first-review pulls | bid mean life | ask mean life | good fills (+1 s > 0) | bad fills (+1 s < 0) | rung 0 shares | rung 0 END | rung 1 shares | rung 1 END |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASELINE_00 | 0 | 0 | 0 | 117 | 4,403 | 5,288 | 4,840 | 4 | 3.14 s | 3.42 s | +64,960.6 c | -268,044.5 c | 53,332.7 | -115,285.9 c | 43,658.4 | -94,271.8 c |
| PULL_ONLY_10 | 553 | 0 | 343 | 89 | 4,029 | 5,519 | 4,837 | 111 | 3.03 s | 3.33 s | +56,810.7 c | -170,455.3 c | 46,946.5 | -122,026.2 c | 38,767.9 | -10,092.4 c |
| WITHHOLD_ONLY_01 | 0 | 523 | 305 | 119 | 4,302 | 5,177 | 4,774 | 2 | 3.14 s | 3.43 s | +66,749.3 c | -259,428.4 c | 52,878.3 | -146,855.0 c | 43,875.7 | -44,353.8 c |
| FULL_11 | 433 | 1,289 | 980 | 84 | 3,613 | 4,811 | 4,422 | 25 | 3.36 s | 3.42 s | +55,130.7 c | -140,295.0 c | 43,127.9 | -90,178.0 c | 36,271.6 | +4,068.4 c |

| arm | mean |I| | time |I| >= 100 | mean terminal |I| | both-side active | bid active | ask active | activation-clock ahead | venue ahead at a pull | next-same-side pairs | same price | ahead difference next - pull |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BASELINE_00 | 83.8 | 36.69 % | 87.4 | 40.70 % | 65.99 % | 70.13 % | 228.9 | — | 0 | 0 | — |
| PULL_ONLY_10 | 83.1 | 33.11 % | 85.5 | 42.03 % | 66.06 % | 68.95 % | 208.6 | 119.3 | 552 | 128 | -75.0 |
| WITHHOLD_ONLY_01 | 83.7 | 36.99 % | 90.3 | 39.79 % | 64.97 % | 69.83 % | 233.0 | — | 0 | 0 | — |
| FULL_11 | 83.9 | 31.09 % | 90.7 | 40.19 % | 62.05 % | 63.83 % | 235.1 | 128.1 | 430 | 57 | -30.3 |

- The queue columns carry review 909c671e's labels. The next-same-side pairs are a diagnostic, not a matched replacement or a counterfactual queue cost.
- The pressure receipt's 500-draw local screen (SELECTION_BEYOND_EXPOSURE, percentile 0.00 %) is context only, not an entry-action or whole-policy null.

## Review 7fca0db4 finding

- The re-entry rows are exported again. Checks v2 H compares every exported field against the ledger, and its falsifier refuses a changed re-entry price. The lag receipt (frozen) is not edited.

## Validity

- **Checks v2** A-I: 17 / 17 true. Falsifier: 11 / 11 corruptions refused by their family.
- **Supersession of the checker:** the pre-run rehearsal on the pilot copy passed every v1 check, but its falsifier stopped at the record-rule case: a ** spread of a tuple-keyed dict (`logs/ga_rehearse_checks.log`). v2 (`d4b843c3`) fixes only that case and was committed before it ran.
- **v2 rehearsal:** 16 / 16 checks true; B_membership is false by design.
- **Mutants and selftest:** mutation suite 20 / 20 on the first run; selftest groups 1-5 pass.
- **Cost:** projection 0.309 core-hours (cap 0.35); ladder line projected 1.849 of 2.0. The round used 0.242 core-hours; at the checks the line had recorded 1.778 core-hours; peak 1,226 MiB.

## Limits

- **Complete policies.** These are complete feedback policies on 19 consumed-day markets. A favourable P&L contrast alone is not predictive skill, and the contrasts are not per-order avoided losses.
- **Replay assumptions.** External flow is unchanged and timing is M50; the Binance clock is the declared median-delay model. Values are gross and zero cost, and END is a pre-expiry mark.
- **Remaining budget.** The original 2-core-hour ladder allocation is nearly used; a further round of this size needs the USER's decision.
