# PRESSURE-LEAD-SENSITIVITY-v1 — report

2026-11-21 19:58Z, session aea5bfb9. This is review `909c671e`'s ONE next priority: the frozen BINANCE-PRESSURE-GUARD-v1 rule (`793c706b`, results `40a3ee8d`) with **one more 50 ms of signal delay**.
- **Population:** the same 19 complete markets of 96 (statuses: incomplete:missing_endpoint 54, incomplete:stale_future_book 23, complete 19).
- **Freeze:** `b19b7ca0`, before any lagged replay. Rule sha256 `f9b4bcd04e88168d`; parent rule `24f853dcfab0682e`.
- **Checker:** checks v2 `9cc6161a` supersedes v1. It was committed before it ran on any full-run output; see Validity.
- **Status of the evidence:** a timing-margin sensitivity on consumed Oct 24-27 development days. It chooses no lag or threshold and makes no validation, promotion or profitability claim.

## Rule and controls

- **Rule.** At action tick k the frozen pressure rule reads its signal and status at tick k - 1 on its own side, from the pressure receipt's hash-bound saved cache.
  - Tick 0 is warm-up.
  - PULL, withhold and clearing all read the delayed stream.
  - Inventory, structural and all other decisions stay current; there is no timer.
- **Zero-lag control:** it reproduces the pressure receipt on 19 of 19 markets (fingerprint, fills, orders, requests, value, commands, the parent's log fields, queue records).
- **Baseline:** the pressure receipt's disabled arm, which equals the original ladder (checks v2 C).
- **Lagged join:** 4,840 live states; 0 zero-lag mismatches; 0 off-tick; 0 at warm-up; 0 non-ok.

## Three arms

Gross, zero cost, in cents. END = cash + inventory × the END mark.

| day | arm | END value | filled shares | execution edge | +50 ms | valid shares | censored | +100 ms | valid shares | censored |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | baseline | -64,287.9 c | 35,761.4 | +18,835.1 c | -26,511.3 c | 35,761.4 | 0 | -29,157.6 c | 35,746.0 | 7 |
| 20261024 | lag0 | -33,011.1 c | 33,793.4 | +22,030.4 c | -14,362.2 c | 33,793.4 | 0 | -15,047.7 c | 33,778.0 | 7 |
| 20261024 | lag1 | -28,649.7 c | 31,082.6 | +20,367.6 c | -14,063.5 c | 31,082.6 | 0 | -15,064.3 c | 31,067.2 | 7 |
| 20261025 | baseline | -66,495.8 c | 26,959.4 | +9,958.4 c | -30,250.5 c | 26,944.4 | 1 | -31,801.9 c | 26,944.4 | 1 |
| 20261025 | lag0 | -29,621.9 c | 24,407.8 | +18,043.3 c | -6,919.9 c | 24,392.7 | 1 | -5,271.6 c | 24,392.7 | 1 |
| 20261025 | lag1 | -37,890.1 c | 25,012.1 | +19,607.1 c | -13,214.0 c | 25,012.1 | 0 | -12,689.0 c | 25,012.1 | 0 |
| 20261026 | baseline | -45,400.2 c | 24,370.3 | +7,668.7 c | -24,685.7 c | 24,370.3 | 0 | -29,762.7 c | 24,370.3 | 0 |
| 20261026 | lag0 | -9,275.1 c | 16,513.2 | +8,251.0 c | -5,129.8 c | 16,513.2 | 0 | -6,990.5 c | 16,513.2 | 0 |
| 20261026 | lag1 | -31,758.0 c | 19,813.7 | +10,084.7 c | -6,500.1 c | 19,813.7 | 0 | -9,397.3 c | 19,813.7 | 0 |
| 20261027 | baseline | -33,373.9 c | 9,900.0 | +2,864.0 c | -10,300.5 c | 9,900.0 | 0 | -11,935.0 c | 9,900.0 | 0 |
| 20261027 | lag0 | -14,201.6 c | 4,685.1 | +2,219.8 c | -1,242.2 c | 4,685.1 | 0 | -1,400.9 c | 4,685.1 | 0 |
| 20261027 | lag1 | -24,085.2 c | 7,273.7 | +3,643.9 c | -2,106.2 c | 7,273.7 | 0 | -2,754.2 c | 7,273.7 | 0 |
| all | baseline | -209,557.7 c | 96,991.1 | +39,326.2 c | -91,748.0 c | 96,976.0 | 1 | -102,657.2 c | 96,960.6 | 8 |
| all | lag0 | -86,109.6 c | 79,399.5 | +50,544.4 c | -27,654.1 c | 79,384.4 | 1 | -28,710.7 c | 79,369.0 | 8 |
| all | lag1 | -122,383.0 c | 83,182.2 | +53,703.3 c | -35,883.8 c | 83,182.2 | 0 | -39,904.8 c | 83,166.8 | 7 |

### +1 s on the shared cutoff and the END identity

The cutoff is t <= END - 1 s.

| day | arm | +1 s | valid shares | censored | per share | ask per share | bid per share | eligible +1 s | movement to END | late / missing |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | baseline | -61,614.3 c | 35,728.2 | 0 | -1.725 c / share | -2.285 c / share | -1.154 c / share | -61,614.3 c | -2,731.1 c | +57.6 c |
| 20261024 | lag0 | -32,357.5 c | 33,760.1 | 0 | -0.958 c / share | -0.892 c / share | -1.026 c / share | -32,357.5 c | -711.2 c | +57.6 c |
| 20261024 | lag1 | -31,632.8 c | 31,049.3 | 0 | -1.019 c / share | -0.987 c / share | -1.051 c / share | -31,632.8 c | +2,925.6 c | +57.6 c |
| 20261025 | baseline | -54,432.5 c | 26,704.4 | 0 | -2.038 c / share | -1.560 c / share | -2.523 c / share | -54,432.5 c | -12,551.4 c | +488.1 c |
| 20261025 | lag0 | -17,996.5 c | 24,152.8 | 0 | -0.745 c / share | -0.343 c / share | -1.154 c / share | -17,996.5 c | -12,113.5 c | +488.1 c |
| 20261025 | lag1 | -28,927.4 c | 24,811.0 | 0 | -1.166 c / share | -0.403 c / share | -1.944 c / share | -28,927.4 c | -9,164.7 c | +202.0 c |
| 20261026 | baseline | -61,126.4 c | 24,271.5 | 0 | -2.518 c / share | -3.171 c / share | -1.867 c / share | -61,126.4 c | +15,128.6 c | +597.7 c |
| 20261026 | lag0 | -29,854.1 c | 16,456.2 | 0 | -1.814 c / share | -2.065 c / share | -1.561 c / share | -29,854.1 c | +20,653.3 c | -74.2 c |
| 20261026 | lag1 | -37,301.6 c | 19,612.6 | 0 | -1.902 c / share | -2.246 c / share | -1.560 c / share | -37,301.6 c | +4,709.3 c | +834.4 c |
| 20261027 | baseline | -25,910.7 c | 9,900.0 | 0 | -2.617 c / share | -3.298 c / share | -1.933 c / share | -25,910.7 c | -7,463.3 c | +0.0 c |
| 20261027 | lag0 | -4,956.2 c | 4,685.1 | 0 | -1.058 c / share | -1.423 c / share | -0.693 c / share | -4,956.2 c | -9,245.4 c | +0.0 c |
| 20261027 | lag1 | -11,698.2 c | 7,273.7 | 0 | -1.608 c / share | -1.648 c / share | -1.571 c / share | -11,698.2 c | -12,387.0 c | +0.0 c |
| all | baseline | -203,083.9 c | 96,604.0 | 0 | -2.102 c / share | -2.409 c / share | -1.792 c / share | -203,083.9 c | -7,617.2 c | +1,143.4 c |
| all | lag0 | -85,164.3 c | 79,054.1 | 0 | -1.077 c / share | -0.999 c / share | -1.157 c / share | -85,164.3 c | -1,416.8 c | +471.5 c |
| all | lag1 | -109,560.1 c | 82,746.7 | 0 | -1.324 c / share | -1.164 c / share | -1.485 c / share | -109,560.1 c | -13,916.9 c | +1,093.9 c |

### Paired END, per share and the retained gain

The retained gain is (lag1 - baseline) / (lag0 - baseline). It is descriptive.

| day | lag0 - baseline END | lag1 - baseline END | lag1 - lag0 END | retained END | lag0 - baseline per share | lag1 - baseline per share | retained per share |
|---|---|---|---|---|---|---|---|
| 20261024 | +31,276.8 c | +35,638.2 c | +4,361.4 c | 113.94 % | +0.766 c / share | +0.706 c / share | 92.12 % |
| 20261025 | +36,873.9 c | +28,605.7 c | -8,268.3 c | 77.58 % | +1.293 c / share | +0.872 c / share | 67.46 % |
| 20261026 | +36,125.1 c | +13,642.2 c | -22,482.9 c | 37.76 % | +0.704 c / share | +0.617 c / share | 87.54 % |
| 20261027 | +19,172.3 c | +9,288.7 c | -9,883.6 c | 48.45 % | +1.559 c / share | +1.009 c / share | 64.70 % |
| all | +123,448.1 c | +87,174.7 c | -36,273.4 c | 70.62 % | — | — | 75.93 % |

### Fills lost and added

Fills are matched by the multiset of (t, side, price, quantity); candidate - reference END = added - lost.

| pair | day | lost fills | lost shares | lost +1 s > 0 | lost +1 s < 0 | lost END | added fills | added shares | added +1 s > 0 | added +1 s < 0 | added END | candidate - reference END |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lag1_vs_baseline | 20261024 | 1,219 | 11,174.6 | +6,915.0 c | -44,836.7 c | -50,999.6 c | 618 | 6,495.7 | +4,201.4 c | -12,141.6 c | -15,361.4 c | +35,638.2 c |
| lag1_vs_baseline | 20261025 | 668 | 7,078.9 | +4,945.2 c | -35,130.8 c | -41,322.6 c | 466 | 5,131.6 | +5,623.4 c | -10,303.9 c | -12,717.0 c | +28,605.7 c |
| lag1_vs_baseline | 20261026 | 1,036 | 11,461.4 | +9,954.8 c | -42,370.6 c | -4,704.7 c | 577 | 6,904.8 | +2,716.7 c | -11,307.7 c | +8,937.5 c | +13,642.2 c |
| lag1_vs_baseline | 20261027 | 551 | 5,664.6 | +2,571.5 c | -21,755.1 c | +18,308.9 c | 280 | 3,038.3 | +2,348.3 c | -7,319.4 c | +27,597.6 c | +9,288.7 c |
| lag1_vs_baseline | all | 3,474 | 35,379.4 | +24,386.5 c | -144,093.1 c | -78,718.0 c | 1,941 | 21,570.5 | +14,889.9 c | -41,072.6 c | +8,456.7 c | +87,174.7 c |
| lag1_vs_lag0 | 20261024 | 698 | 6,751.3 | +3,970.8 c | -13,397.7 c | +1,277.8 c | 375 | 4,040.5 | +1,653.3 c | -10,355.6 c | +5,639.2 c | +4,361.4 c |
| lag1_vs_lag0 | 20261025 | 475 | 4,914.5 | +6,458.6 c | -9,367.3 c | +262.3 c | 494 | 5,518.9 | +5,866.2 c | -19,705.9 c | -8,006.0 c | -8,268.3 c |
| lag1_vs_lag0 | 20261026 | 188 | 2,649.5 | +675.0 c | -4,652.3 c | +15,763.0 c | 496 | 5,950.0 | +2,573.8 c | -13,998.6 c | -6,719.9 c | -22,482.9 c |
| lag1_vs_lag0 | 20261027 | 61 | 813.0 | +456.8 c | -2,722.1 c | +28,418.9 c | 335 | 3,401.7 | +1,801.0 c | -10,808.4 c | +18,535.3 c | -9,883.6 c |
| lag1_vs_lag0 | all | 1,422 | 15,128.3 | +11,561.2 c | -30,139.5 c | +45,722.0 c | 1,700 | 18,911.1 | +11,894.3 c | -54,868.5 c | +9,448.6 c | -36,273.4 c |

### Per market

| day | market | baseline END | lag0 END | lag1 END | lag0 - baseline | lag1 - baseline | lag1 - lag0 |
|---|---|---|---|---|---|---|---|
| 20261024 | 3697902 | -8,538.1 c | -2,429.1 c | -2,399.0 c | +6,109.0 c | +6,139.1 c | +30.1 c |
| 20261024 | 3699278 | -11,688.6 c | -4,353.7 c | -4,913.7 c | +7,334.9 c | +6,774.8 c | -560.0 c |
| 20261024 | 3700070 | -3,276.4 c | -1,581.0 c | +235.1 c | +1,695.4 c | +3,511.5 c | +1,816.1 c |
| 20261024 | 3702960 | -2,051.7 c | -501.0 c | -290.9 c | +1,550.7 c | +1,760.9 c | +210.2 c |
| 20261024 | 3704004 | -10,711.1 c | -6,716.0 c | -6,492.4 c | +3,995.0 c | +4,218.6 c | +223.6 c |
| 20261024 | 3712725 | -13,128.9 c | -8,627.1 c | -9,478.5 c | +4,501.8 c | +3,650.4 c | -851.4 c |
| 20261024 | 3716727 | -14,893.2 c | -8,803.1 c | -5,310.3 c | +6,090.0 c | +9,582.9 c | +3,492.8 c |
| 20261025 | 3723774 | -14,024.2 c | +3,967.7 c | -419.4 c | +17,991.9 c | +13,604.8 c | -4,387.1 c |
| 20261025 | 3734864 | -19,392.9 c | -13,490.9 c | -12,978.7 c | +5,902.0 c | +6,414.2 c | +512.2 c |
| 20261025 | 3737919 | -15,136.8 c | -4,760.2 c | -9,356.9 c | +10,376.6 c | +5,779.8 c | -4,596.7 c |
| 20261025 | 3763401 | -10,489.6 c | -9,775.8 c | -9,743.3 c | +713.8 c | +746.3 c | +32.5 c |
| 20261025 | 3776314 | -7,452.3 c | -5,562.6 c | -5,391.9 c | +1,889.6 c | +2,060.4 c | +170.8 c |
| 20261026 | 3789484 | -4,568.8 c | -1,593.3 c | -1,551.0 c | +2,975.5 c | +3,017.8 c | +42.2 c |
| 20261026 | 3803131 | -1,828.4 c | +7,504.7 c | +5,253.1 c | +9,333.1 c | +7,081.5 c | -2,251.6 c |
| 20261026 | 3804696 | -17,833.7 c | +2,415.5 c | -13,185.0 c | +20,249.2 c | +4,648.7 c | -15,600.5 c |
| 20261026 | 3838531 | -4,906.1 c | -9,523.9 c | -12,181.9 c | -4,617.8 c | -7,275.8 c | -2,658.0 c |
| 20261026 | 3850428 | -16,263.2 c | -8,078.1 c | -10,093.2 c | +8,185.1 c | +6,170.0 c | -2,015.1 c |
| 20261027 | 3877472 | -18,314.4 c | -16,661.9 c | -14,341.6 c | +1,652.5 c | +3,972.9 c | +2,320.3 c |
| 20261027 | 3899224 | -15,059.5 c | +2,460.3 c | -9,743.6 c | +17,519.8 c | +5,315.8 c | -12,203.9 c |

- lag1 - baseline: 18 markets gain, 1 lose.
- lag1 - lag0: 10 markets gain, 9 lose.

### Interventions, removals, lifetimes, inventory and uptime

`withheld proposals` count slot decisions, not orders. `reducing-side` means bids at I < 0 and asks at I > 0. Inventory is exact from the fills; uptime is exact from the order intervals.

| arm (all days) | pressure pulls | withheld proposals | near END | reducing-side | inventory PULL requests | structural requests | generations | reviewed | first-review pulls | of them pressure | bid mean life | ask mean life | mean |I| | mean peak |I| | time |I| >= 100 | mean terminal |I| | both-side active | bid active | ask active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 0 | 0 | 0 | 117 | 4,403 | 5,288 | 4,840 | 4 | 0 | 3.14 s | 3.42 s | 83.8 | 183.1 | 36.69 % | 87.4 | 40.70 % | 65.99 % | 70.13 % |
| lag0 | 433 | 1,289 | 0 | 980 | 84 | 3,613 | 4,811 | 4,422 | 25 | 23 | 3.36 s | 3.42 s | 83.9 | 179.2 | 31.09 % | 90.7 | 40.19 % | 62.05 % | 63.83 % |
| lag1 | 472 | 1,402 | 0 | 1,049 | 85 | 3,648 | 4,895 | 4,523 | 34 | 33 | 3.16 s | 3.41 s | 80.9 | 181.9 | 30.42 % | 84.5 | 41.39 % | 62.89 % | 68.02 % |

### Queue diagnostics (relabelled, review 909c671e finding 2)

- The activation-clock snapshot is the engine's queue ahead from the venue frame at activation (submission + 58 ms), evaluated at submission.
- Next-same-side pairs pair each pressure pull with the same side's next submission at any price and rung. They are a diagnostic, not a matched replacement or a counterfactual queue cost.

| arm (all days, market means) | activation-clock ahead | venue ahead at first review | known estimate at first review | venue ahead at the pull | pulls | with a next order | same price | same rung | gap to the next | ahead difference next - pull | same-price ahead difference |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lag0 | 235.1 | 251.8 | 227.8 | 128.1 | 433 | 430 | 57 | 156 | 1.00 s | -30.3 | -102.5 |
| lag1 | 236.4 | 251.4 | 227.4 | 126.7 | 472 | 471 | 52 | 175 | 0.97 s | 14.6 | -88.9 |

## Lagged selection screen

The frozen screen runs with the lagged mask: strata, 500 draws, seed 20261121, minimum support and censoring are unchanged.

- **Pooled:** 110 selected of 4,831 uncensored states; 62 movable strata with 103 selected. Selected -90.2 c against null mean -31.5 c and p05 -57.9 c; percentile 0.20 %. Per valid share -9.232 c / share (null p05 -5.104 c / share).
- **Per side (descriptive):**
  - bids: 51 selected, -18.4 c against p05 -43.2 c; percentile 37.20 %;
  - asks: 59 selected, -152.2 c against p05 -90.5 c; percentile 0.20 %.

## Computed readings (measure/audit.json predicates)

- **lag1 vs baseline:**
  - END_paired IMPROVES_EVERY_DAY (total +87,174.7 c; 18 better, 1 worse);
  - per share all IMPROVES_EVERY_DAY, bid IMPROVES_EVERY_DAY, ask IMPROVES_EVERY_DAY;
  - exact active time: both +0.69 pp, bids -3.10 pp, asks -2.12 pp;
  - inventory: mean |I| -2.9; selection SELECTION_BEYOND_EXPOSURE.
- **lag0 vs baseline (the reference):** END IMPROVES_EVERY_DAY (total +123,448.1 c); per share IMPROVES_EVERY_DAY.
- **Retained gain:** END 70.62 % in total (by day 20261024 113.94 %, 20261025 77.58 %, 20261026 37.76 %, 20261027 48.45 %); per share 75.93 %.
- **Reading (declared rule):** `survives_50ms_screen`.

## Review 909c671e findings

1. **Checker coverage (finding 1).** Checks v2's F family rebuilds exact uptime from each order's activation, removal and full fill, and exact inventory from all fills. It also rebuilds the queue fields and pair summaries for all three arms, and compares the participation and inventory readings. The coordinated uptime case (+25 pp with the reading moved) and an inventory case are refused.
   - The pressure receipt's claim that every reading field was recomputed (`ladder_pressure_20261121/report.md:192`, frozen) was too broad. This receipt supersedes it in band; that receipt is not edited.
2. **Queue labels (finding 2).** They are applied above and in `reentry.csv`: both gids, prices, rungs and times, the same-price and same-rung flags, the activation-clock ahead and the ahead difference.

## Validity

- **Checks v2** A-I: 21 / 21 true. Falsifier: 14 / 14 corruptions refused by their family.
- **Supersession of the checker:** the pre-run rehearsal on the pilot copy stopped v1 at F. Its `dict(...)` of the frozen helper's defaultdict dropped zero counters (KeyError `all_edge_censored`, `logs/gl_rehearse_checks.log`). v2 (`9cc6161a`) fixes only that and was committed before it ran.
- **v2 rehearsal:** 20 / 20 checks true; B_membership is false by design.
- **Mutants and selftest:** mutation suite 21 / 21. The first run had one escape (`same_price` from rungs); the fixture was strengthened and those logs are kept. Selftest groups 1-5 pass.
- **Cost:** pilot projection 0.344 core-hours. The round used 0.259 core-hours (every run_light record under logs/) of the 0.50 cap; peak 1,199 MiB.

## Limits

- **One timing margin on consumed days.** This is a single +50 ms margin on the same 19 consumed-day markets. The feature, its edges and the zero-lag result all came from these markets, so no day here is a validation day.
- **Clock model.** The Binance availability clock is the declared Tokyo median-delay model, not per-message receive stamps (review 909c671e). The lag tests a modest margin; it does not measure real feed-tail latency.
- **Replay assumptions.** External flow is unchanged and timing is M50. Values are gross and zero cost; END is a pre-expiry mark. Coverage (19 / 96) and external-flow realism stay open.
