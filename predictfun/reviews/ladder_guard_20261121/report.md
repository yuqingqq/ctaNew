# ASK-RESERVATION-GUARD-v1 — report

2026-11-21 18:32Z, session aea5bfb9. Review `f8c5e579`'s ONE next priority: one declared candidate against the frozen ladder baseline (`47d2eec1`, results `869fa85f`) on its 19 complete markets (all 96 statuses as in the baseline). Freeze `ee632bc2` before any replay; rule sha256 `ff657441dee88f9b`; edge 200 shares (the frozen Oct 24-25 q5 edge). One adaptive consumed-day candidate: no profitability, significance, validation or promotion claim.

## Rule and control

- On asks only, while the client-known bid reservation float32(B) >= 200: a live ask the baseline would KEEP is PULLed; a slot it would POST is WAITed. Bids, the inventory rule (precedence), sizes, structural rules, timing and paths are the baseline's; no timer.
- **Disabled control:** the disabled arm reproduces the accepted baseline ledger (fingerprint, fills, orders, requests, value) on 19 of 19 markets.

## Paired replay (gross, zero cost, cents; END = cash + inventory x the END mark)

| day | arm | markets | END value | filled shares | ask shares | bid shares | execution edge | +50 ms | +100 ms |
|---|---|---|---|---|---|---|---|---|---|
| 20261024 | baseline | 7 | -64,287.9 c | 35,761.4 | 18,038.1 | 17,723.3 | +18,835.1 c | -26,511.3 c | -29,157.6 c |
| 20261024 | guard | 7 | -40,398.4 c | 30,706.8 | 15,346.5 | 15,360.3 | +16,792.6 c | -22,871.6 c | -24,808.7 c |
| 20261025 | baseline | 5 | -66,495.8 c | 26,959.4 | 13,509.6 | 13,449.8 | +9,958.4 c | -30,250.5 c | -31,801.9 c |
| 20261025 | guard | 5 | -55,833.2 c | 25,079.9 | 12,548.3 | 12,531.6 | +12,690.8 c | -23,927.3 c | -25,020.1 c |
| 20261026 | baseline | 5 | -45,400.2 c | 24,370.3 | 12,214.6 | 12,155.7 | +7,668.7 c | -24,685.7 c | -29,762.7 c |
| 20261026 | guard | 5 | -48,894.0 c | 24,400.0 | 12,232.1 | 12,167.9 | +9,420.8 c | -24,403.8 c | -29,610.4 c |
| 20261027 | baseline | 2 | -33,373.9 c | 9,900.0 | 4,961.5 | 4,938.5 | +2,864.0 c | -10,300.5 c | -11,935.0 c |
| 20261027 | guard | 2 | -30,988.4 c | 9,584.4 | 4,801.3 | 4,783.1 | +2,555.6 c | -9,167.4 c | -10,713.2 c |
| all | baseline | 19 | -209,557.7 c | 96,991.1 | 48,723.9 | 48,267.2 | +39,326.2 c | -91,748.0 c | -102,657.2 c |
| all | guard | 19 | -176,114.1 c | 89,771.0 | 44,928.2 | 44,842.9 | +41,459.8 c | -80,370.1 c | -90,152.4 c |

### +1 s markout on the shared fill-time cutoff (t <= END - 1 s; late fills stay in END)

| day | arm | +1 s | valid shares | censored fills | per share | ask per share | bid per share | good fills (+1 s > 0) | bad fills (+1 s < 0) |
|---|---|---|---|---|---|---|---|---|---|
| 20261024 | baseline | -61,614.3 c | 35,728.2 | 0 | -1.725 c / share | -2.285 c / share | -1.154 c / share | +24,299.3 c | -85,913.6 c |
| 20261024 | guard | -48,473.1 c | 30,673.5 | 0 | -1.580 c / share | -1.780 c / share | -1.381 c / share | +20,644.7 c | -69,117.8 c |
| 20261025 | baseline | -54,432.5 c | 26,704.4 | 0 | -2.038 c / share | -1.560 c / share | -2.523 c / share | +20,419.2 c | -74,851.7 c |
| 20261025 | guard | -43,774.1 c | 24,860.3 | 0 | -1.761 c / share | -0.805 c / share | -2.730 c / share | +18,114.3 c | -61,888.4 c |
| 20261026 | baseline | -61,126.4 c | 24,271.5 | 0 | -2.518 c / share | -3.171 c / share | -1.867 c / share | +15,850.3 c | -76,976.7 c |
| 20261026 | guard | -53,750.5 c | 24,297.6 | 0 | -2.212 c / share | -2.513 c / share | -1.910 c / share | +16,115.9 c | -69,866.3 c |
| 20261027 | baseline | -25,910.7 c | 9,900.0 | 0 | -2.617 c / share | -3.298 c / share | -1.933 c / share | +4,391.8 c | -30,302.4 c |
| 20261027 | guard | -22,611.6 c | 9,584.4 | 0 | -2.359 c / share | -2.869 c / share | -1.847 c / share | +5,244.5 c | -27,856.1 c |
| all | baseline | -203,083.9 c | 96,604.0 | 0 | -2.102 c / share | -2.409 c / share | -1.792 c / share | +64,960.6 c | -268,044.5 c |
| all | guard | -168,609.2 c | 89,415.8 | 0 | -1.886 c / share | -1.823 c / share | -1.948 c / share | +60,119.4 c | -228,728.6 c |

### Participation, churn and inventory

| day | arm | both-side active (exact) | ask active | bid active | guard pulls | guard withholds | reducing-side suppressed | generations | structural requests | policy requests | mean |I| | time |I| >= 100 | mean terminal |I| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | baseline | 39.85 % | 72.07 % | 65.02 % | 0 | 0 | 0 | 1,919 | 1,576 | 47 | 82.4 | 37.85 % | 83.1 |
| 20261024 | guard | 29.24 % | 60.17 % | 60.96 % | 203 | 11,313 | 0 | 1,851 | 1,368 | 243 | 79.8 | 34.22 % | 71.4 |
| 20261025 | baseline | 41.16 % | 71.66 % | 66.82 % | 0 | 0 | 0 | 1,318 | 1,083 | 35 | 84.6 | 37.92 % | 80.9 |
| 20261025 | guard | 37.06 % | 67.28 % | 67.11 % | 140 | 8,837 | 0 | 1,397 | 1,033 | 166 | 72.0 | 29.99 % | 78.8 |
| 20261026 | baseline | 41.77 % | 66.21 % | 66.66 % | 0 | 0 | 0 | 1,502 | 1,291 | 26 | 86.0 | 33.69 % | 99.6 |
| 20261026 | guard | 32.41 % | 59.02 % | 70.18 % | 170 | 14,366 | 0 | 1,644 | 1,277 | 194 | 74.2 | 26.94 % | 51.2 |
| 20261027 | baseline | 39.84 % | 69.38 % | 65.61 % | 0 | 0 | 0 | 549 | 453 | 9 | 80.9 | 36.90 % | 88.1 |
| 20261027 | guard | 24.93 % | 60.67 % | 59.66 % | 75 | 3,473 | 0 | 574 | 405 | 83 | 78.3 | 34.46 % | 90.3 |
| all | baseline | 40.70 % | 70.13 % | 65.99 % | 0 | 0 | 0 | 5,288 | 4,403 | 117 | 83.8 | 36.67 % | 87.4 |
| all | guard | 31.68 % | 61.79 % | 64.87 % | 588 | 37,989 | 0 | 5,466 | 4,083 | 686 | 76.1 | 31.22 % | 70.0 |

`guard withholds` count slot decisions (one per vacant ask slot per 50 ms tick while the signal is on), not orders; `guard pulls` are cancel requests. Client-known position at the interventions: min -95.8, max 0.0 shares, so `reducing-side suppressed` counts interventions at a long position.

### Per market (END, cents; filled shares)

| day | market | baseline END | guard END | guard - baseline | baseline shares | guard shares | guard pulls | guard withholds |
|---|---|---|---|---|---|---|---|---|
| 20261024 | 3697902 | -8,538.1 c | +3,720.2 c | +12,258.3 c | 4,121.5 | 1,853.0 | 5 | 418 |
| 20261024 | 3699278 | -11,688.6 c | -9,440.5 c | +2,248.0 c | 5,497.7 | 5,563.3 | 42 | 2,754 |
| 20261024 | 3700070 | -3,276.4 c | -4,479.9 c | -1,203.5 c | 5,771.2 | 5,079.3 | 49 | 2,787 |
| 20261024 | 3702960 | -2,051.7 c | -2,012.1 c | +39.7 c | 2,503.5 | 2,411.7 | 7 | 380 |
| 20261024 | 3704004 | -10,711.1 c | -10,644.8 c | +66.2 c | 7,014.4 | 5,435.7 | 27 | 1,307 |
| 20261024 | 3712725 | -13,128.9 c | -5,779.4 c | +7,349.4 c | 5,619.2 | 5,262.7 | 30 | 1,487 |
| 20261024 | 3716727 | -14,893.2 c | -11,761.9 c | +3,131.3 c | 5,233.9 | 5,101.0 | 43 | 2,180 |
| 20261025 | 3723774 | -14,024.2 c | -12,344.5 c | +1,679.7 c | 5,877.3 | 5,581.0 | 51 | 2,543 |
| 20261025 | 3734864 | -19,392.9 c | -13,162.8 c | +6,230.2 c | 5,925.0 | 4,996.9 | 44 | 2,467 |
| 20261025 | 3737919 | -15,136.8 c | -13,161.2 c | +1,975.6 c | 5,403.9 | 5,193.8 | 16 | 1,906 |
| 20261025 | 3763401 | -10,489.6 c | -8,263.4 c | +2,226.2 c | 6,794.5 | 6,150.8 | 19 | 697 |
| 20261025 | 3776314 | -7,452.3 c | -8,901.5 c | -1,449.2 c | 2,958.7 | 3,157.4 | 10 | 1,224 |
| 20261026 | 3789484 | -4,568.8 c | -3,447.9 c | +1,120.9 c | 5,350.4 | 5,168.6 | 46 | 2,710 |
| 20261026 | 3803131 | -1,828.4 c | +61.2 c | +1,889.6 c | 4,204.5 | 3,940.2 | 47 | 5,376 |
| 20261026 | 3804696 | -17,833.7 c | -15,941.2 c | +1,892.5 c | 4,832.4 | 4,094.8 | 38 | 2,537 |
| 20261026 | 3838531 | -4,906.1 c | -15,685.1 c | -10,779.0 c | 2,930.9 | 5,155.0 | 17 | 2,094 |
| 20261026 | 3850428 | -16,263.2 c | -13,881.0 c | +2,382.2 c | 7,052.2 | 6,041.2 | 22 | 1,649 |
| 20261027 | 3877472 | -18,314.4 c | -14,027.4 c | +4,287.0 c | 5,635.9 | 5,368.4 | 47 | 1,982 |
| 20261027 | 3899224 | -15,059.5 c | -16,961.0 c | -1,901.5 c | 4,264.0 | 4,216.0 | 28 | 1,491 |

### Rung split (all days), removals and lifetimes (the last two computed by this report tool from the records and command arrays; descriptive)

| arm | rung 0 shares | rung 0 END | rung 1 shares | rung 1 END |
|---|---|---|---|---|
| baseline | 53,332.7 | -115,285.9 c | 43,658.4 | -94,271.8 c |
| guard | 50,097.4 | -84,085.0 c | 39,673.6 | -92,029.1 c |

| arm | side | policy PULL reason | requests |
|---|---|---|---|
| baseline | bid | inventory_pull | 64 |
| baseline | ask | inventory_pull | 53 |
| guard | bid | inventory_pull | 68 |
| guard | ask | guard_pull | 588 |
| guard | ask | inventory_pull | 30 |

| arm | side | generations (not refused) | mean live time | median live time |
|---|---|---|---|---|
| baseline | bid | 2,520 | 2.83 s | 1.38 s |
| baseline | ask | 2,463 | 3.08 s | 1.53 s |
| guard | bid | 2,426 | 2.88 s | 1.38 s |
| guard | ask | 2,747 | 2.44 s | 0.93 s |

Live time = min(remove, full fill, END) - activation, the interval pf_ladder_audit.durations uses.

## Selection control (saved audit states; 500 matched draws, seed 20261121; not a null for the full feedback policy)

- Selected q5 ask live states 517 of 2,413 uncensored (6 censored, 3 of them selected); strata 365, movable 104 with 345 selected states, degenerate 73.
- Selected mean window +1 s -58.0 c against the matched null mean -51.4 c (p05 -58.7 c, p95 -42.7 c); the selected value's null percentile 7.40 %. Per valid window share -4.550 c / share (null p05 -4.547 c / share).
- Quantities (review finding 1): valid window 6,592.8, total 6,716.5, excluded (censored) 123.7 shares.

## Computed readings (measure/audit.json predicates)

- **END_paired:** MIXED — by day 20261024 +23,889.4 c, 20261025 +10,662.5 c, 20261026 -3,493.8 c, 20261027 +2,385.5 c; markets better / worse / equal 15 / 4 / 0.
- **+1 s per share (shared cutoff):** all IMPROVES_EVERY_DAY, ask IMPROVES_EVERY_DAY, bid MIXED — all-side by day 20261024 +0.144 c / share, 20261025 +0.278 c / share, 20261026 +0.306 c / share, 20261027 +0.258 c / share.
- **Participation:** filled shares by day 20261024 -5,054.6, 20261025 -1,879.5, 20261026 29.7, 20261027 -315.6; both-side active -9.02 pp, ask active -8.34 pp; ask shares -3,795.7.
- **Inventory:** mean |I| -7.6 shares, time |I| >= 100 -5.46 pp, terminal |I| -17.4.
- **Selection control:** NO_SELECTION_BEYOND_EXPOSURE. **Interpretation (declared rule):** mixed.

## Corrected summaries and the audit re-derivation (review f8c5e579)

- Finding 3: the generation +1 s censor count is 13 (generations with fills whose lifetime +1 s is missing; no-fill generations excluded).
- Finding 2: `guard_checks.py` G recomputed the audit's bin means and censored counts from its state rows (G_bin_means True, G_bin_censored True), every lead flag and pooled contrast (G_lead_flags True) and the candidate (G_candidate True); the falsifier corrupts the stored q5 means and the censored counts and both are refused.

## Validity

- Checks A-I: 21 / 21 true; falsifier 7 / 7; disabled-arm mismatches 0; rule violations 0.
- Post-run falsifier supplement (`tools/falsify_supplement.py`, added after the run; no reading changed): 22 / 22 in-memory corruptions refused by their named check, including review f8c5e579's two probe corruptions (the eight q5 means set to q1 + 1 -> G_bin_means; the frozen audit censor field set to -999,999 -> the supplement's X_frozen_censor_field, which requires the stored -3,195 to equal its frozen formula); the checks it leaves uncovered (F_arms, G_bin_censored, H_valid_terms) are the frozen falsifier's cases.
- Statuses (all 96 selected markets, baseline run): incomplete:missing_endpoint 54, incomplete:stale_future_book 23, complete 19; the guard ran on the complete ones only.
- Mutation suite 16 / 16; selftest groups 1-7 pass.
- Cost 0.211 core-hours when the checks ran; 0.259 core-hours for the round through the supplement and reports (every run_light record under logs/), of the 0.75 cap; peak 1,136 MiB.

## Limits

- One adaptive candidate on the 19 complete consumed-day markets; the rule's edge came from the audit of the same days (Oct 24-25 bins, the lead found on all four days).
- Replay values assume unchanged external flow (our withheld / pulled asks do not move others) and M50 timing; gross, zero cost; END is a pre-expiry mark.
- The selection control conditions on day x market x remaining x age x inventory sign only; it tests the local association, not the feedback policy.
- Paired values pass through the path-coupled inventory rule (CLAUDE.md pitfall 4): the largest single-market change is -10,779.0 c (20261026 market 3838531, filled shares 2,930.9 -> 5,155.0) against that day's -3,493.8 c; the largest gain is +12,258.3 c (20261024 market 3697902).
- Same 19 markets as the audit that chose the rule: no day here is a validation day.
