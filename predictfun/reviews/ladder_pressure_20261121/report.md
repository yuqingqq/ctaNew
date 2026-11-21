# BINANCE-PRESSURE-GUARD-v1 — report

2026-11-21 19:25Z, session aea5bfb9. This is review `2c5dbd53`'s ONE next priority: one external-market signal against the frozen ladder baseline (`47d2eec1`, results `869fa85f`).
- **Population:** the baseline's 19 complete markets of its 96 selected ones (statuses: incomplete:missing_endpoint 54, incomplete:stale_future_book 23, complete 19).
- **Freeze:** `793c706b`, before any replay. Rule sha256 `24f853dcfab0682e`.
- **Frozen q5 edges:** bids x >= 0.4666730940341952, asks x >= 0.285456120967865; edge file `ca688917`.
- **Status of the evidence:** this is the second guard candidate of this ladder line on consumed Oct 24-27 development days. The feature was suggested by the audit of these same markets, and its bins come from Oct 24-25. It makes no profitability, significance, validation or promotion claim.

## Rule and controls

- **Rule.** On side s while the builder's float32 adv_bn_r250 at the decision is finite, positive and in its frozen q5:
  - a live order the baseline would KEEP is PULLed;
  - a slot it would POST is WAITed (a withheld slot proposal).
- **Everything else is the baseline's:** inventory and structural precedence, sizes, timing and paths. There is no timer.
- **Disabled control.** The disabled arm reproduces the accepted baseline ledgers on 19 of 19 markets: fingerprint, fills, orders, requests, value, initial queue and receipt-known queue estimates.
- **Integration boundary**, computed in the run and re-derived by the checker:
  - snapshot parity: 14,968 of 14,968 saved baseline feature rows equal the cache bit for bit;
  - side sign: 0 mismatches;
  - causal checks: 64 decisions; 64 unchanged under the builder's future-only perturbation (64 with an unchanged signal); 64 flipped by the delivered adverse-move control; 64 equal to the cache;
  - END identity gap: at most 7.3e-12 c.
- **Cache statuses (ticks):** bid ok 113,240, bid off_grid 0, bid no_market_row 0, bid nan 0, ask ok 113,240, ask off_grid 0, ask no_market_row 0, ask nan 0.
- **Missing-signal policy evaluations:** bids 0, asks 0.

## Paired replay

Gross, zero cost, in cents. END = cash + inventory × the END mark.

| day | arm | markets | END value | filled shares | ask shares | bid shares | execution edge | valid shares | censored | +50 ms | valid shares | censored | +100 ms | valid shares | censored |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | baseline | 7 | -64,287.9 c | 35,761.4 | 18,038.1 | 17,723.3 | +18,835.1 c | 35,761.4 | 0 | -26,511.3 c | 35,761.4 | 0 | -29,157.6 c | 35,746.0 | 7 |
| 20261024 | pressure | 7 | -33,011.1 c | 33,793.4 | 17,042.1 | 16,751.3 | +22,030.4 c | 33,793.4 | 0 | -14,362.2 c | 33,793.4 | 0 | -15,047.7 c | 33,778.0 | 7 |
| 20261025 | baseline | 5 | -66,495.8 c | 26,959.4 | 13,509.6 | 13,449.8 | +9,958.4 c | 26,959.4 | 0 | -30,250.5 c | 26,944.4 | 1 | -31,801.9 c | 26,944.4 | 1 |
| 20261025 | pressure | 5 | -29,621.9 c | 24,407.8 | 12,241.0 | 12,166.8 | +18,043.3 c | 24,407.8 | 0 | -6,919.9 c | 24,392.7 | 1 | -5,271.6 c | 24,392.7 | 1 |
| 20261026 | baseline | 5 | -45,400.2 c | 24,370.3 | 12,214.6 | 12,155.7 | +7,668.7 c | 24,370.3 | 0 | -24,685.7 c | 24,370.3 | 0 | -29,762.7 c | 24,370.3 | 0 |
| 20261026 | pressure | 5 | -9,275.1 c | 16,513.2 | 8,319.0 | 8,194.2 | +8,251.0 c | 16,513.2 | 0 | -5,129.8 c | 16,513.2 | 0 | -6,990.5 c | 16,513.2 | 0 |
| 20261027 | baseline | 2 | -33,373.9 c | 9,900.0 | 4,961.5 | 4,938.5 | +2,864.0 c | 9,900.0 | 0 | -10,300.5 c | 9,900.0 | 0 | -11,935.0 c | 9,900.0 | 0 |
| 20261027 | pressure | 2 | -14,201.6 c | 4,685.1 | 2,342.8 | 2,342.3 | +2,219.8 c | 4,685.1 | 0 | -1,242.2 c | 4,685.1 | 0 | -1,400.9 c | 4,685.1 | 0 |
| all | baseline | 19 | -209,557.7 c | 96,991.1 | 48,723.9 | 48,267.2 | +39,326.2 c | 96,991.1 | 0 | -91,748.0 c | 96,976.0 | 1 | -102,657.2 c | 96,960.6 | 8 |
| all | pressure | 19 | -86,109.6 c | 79,399.5 | 39,944.9 | 39,454.6 | +50,544.4 c | 79,399.5 | 0 | -27,654.1 c | 79,384.4 | 1 | -28,710.7 c | 79,369.0 | 8 |

### +1 s markout on the shared fill-time cutoff

The cutoff is t <= END - 1 s; later fills stay in END.

| day | arm | +1 s | valid shares | censored fills | per share | ask per share | bid per share | good fills (+1 s > 0) | bad fills (+1 s < 0) |
|---|---|---|---|---|---|---|---|---|---|
| 20261024 | baseline | -61,614.3 c | 35,728.2 | 0 | -1.725 c / share | -2.285 c / share | -1.154 c / share | +24,299.3 c | -85,913.6 c |
| 20261024 | pressure | -32,357.5 c | 33,760.1 | 0 | -0.958 c / share | -0.892 c / share | -1.026 c / share | +23,903.1 c | -56,260.6 c |
| 20261025 | baseline | -54,432.5 c | 26,704.4 | 0 | -2.038 c / share | -1.560 c / share | -2.523 c / share | +20,419.2 c | -74,851.7 c |
| 20261025 | pressure | -17,996.5 c | 24,152.8 | 0 | -0.745 c / share | -0.343 c / share | -1.154 c / share | +21,689.8 c | -39,686.3 c |
| 20261026 | baseline | -61,126.4 c | 24,271.5 | 0 | -2.518 c / share | -3.171 c / share | -1.867 c / share | +15,850.3 c | -76,976.7 c |
| 20261026 | pressure | -29,854.1 c | 16,456.2 | 0 | -1.814 c / share | -2.065 c / share | -1.561 c / share | +6,713.4 c | -36,567.6 c |
| 20261027 | baseline | -25,910.7 c | 9,900.0 | 0 | -2.617 c / share | -3.298 c / share | -1.933 c / share | +4,391.8 c | -30,302.4 c |
| 20261027 | pressure | -4,956.2 c | 4,685.1 | 0 | -1.058 c / share | -1.423 c / share | -0.693 c / share | +2,824.3 c | -7,780.5 c |
| all | baseline | -203,083.9 c | 96,604.0 | 0 | -2.102 c / share | -2.409 c / share | -1.792 c / share | +64,960.6 c | -268,044.5 c |
| all | pressure | -85,164.3 c | 79,054.1 | 0 | -1.077 c / share | -0.999 c / share | -1.157 c / share | +55,130.7 c | -140,295.0 c |

### The END identity

END = eligible +1 s markout + movement from each eligible fill's +1 s mid to the END mark + late / missing-horizon END.

| day | arm | eligible +1 s | movement to END | late / missing | END |
|---|---|---|---|---|---|
| 20261024 | baseline | -61,614.3 c | -2,731.1 c | +57.6 c | -64,287.9 c |
| 20261024 | pressure | -32,357.5 c | -711.2 c | +57.6 c | -33,011.1 c |
| 20261025 | baseline | -54,432.5 c | -12,551.4 c | +488.1 c | -66,495.8 c |
| 20261025 | pressure | -17,996.5 c | -12,113.5 c | +488.1 c | -29,621.9 c |
| 20261026 | baseline | -61,126.4 c | +15,128.6 c | +597.7 c | -45,400.2 c |
| 20261026 | pressure | -29,854.1 c | +20,653.3 c | -74.2 c | -9,275.1 c |
| 20261027 | baseline | -25,910.7 c | -7,463.3 c | +0.0 c | -33,373.9 c |
| 20261027 | pressure | -4,956.2 c | -9,245.4 c | +0.0 c | -14,201.6 c |
| all | baseline | -203,083.9 c | -7,617.2 c | +1,143.4 c | -209,557.7 c |
| all | pressure | -85,164.3 c | -1,416.8 c | +471.5 c | -86,109.6 c |

### Fills lost and added

Fills are matched by the multiset of (t, side, price, quantity). Pressure − baseline END = added − lost.

| day | lost fills | lost shares | lost +1 s > 0 | lost +1 s < 0 | lost END | added fills | added shares | added +1 s > 0 | added +1 s < 0 | added END | pressure - baseline END |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | 927 | 8,852.1 | +5,034.1 c | -43,258.8 c | -38,319.4 c | 649 | 6,884.1 | +4,638.0 c | -13,605.8 c | -7,042.6 c | +31,276.8 c |
| 20261025 | 714 | 7,714.4 | +4,551.3 c | -44,408.1 c | -36,536.7 c | 493 | 5,162.8 | +5,821.9 c | -9,242.6 c | +337.2 c | +36,873.9 c |
| 20261026 | 1,216 | 13,607.6 | +10,982.1 c | -48,890.3 c | -33,402.8 c | 449 | 5,750.5 | +1,845.2 c | -8,481.1 c | +2,722.3 c | +36,125.1 c |
| 20261027 | 707 | 7,167.8 | +3,193.4 c | -26,103.9 c | +8,412.4 c | 162 | 1,952.9 | +1,625.9 c | -3,582.0 c | +27,584.8 c | +19,172.3 c |
| all | 3,564 | 37,341.9 | +23,760.8 c | -162,661.0 c | -99,846.4 c | 1,753 | 19,750.3 | +13,931.0 c | -34,911.5 c | +23,601.7 c | +123,448.1 c |

### Per market (END in cents; filled shares)

| day | market | baseline END | pressure END | pressure - baseline | baseline shares | pressure shares | pressure pulls | withheld slot proposals |
|---|---|---|---|---|---|---|---|---|
| 20261024 | 3697902 | -8,538.1 c | -2,429.1 c | +6,109.0 c | 4,121.5 | 4,710.2 | 8 | 12 |
| 20261024 | 3699278 | -11,688.6 c | -4,353.7 c | +7,334.9 c | 5,497.7 | 5,252.1 | 25 | 68 |
| 20261024 | 3700070 | -3,276.4 c | -1,581.0 c | +1,695.4 c | 5,771.2 | 4,951.8 | 6 | 18 |
| 20261024 | 3702960 | -2,051.7 c | -501.0 c | +1,550.7 c | 2,503.5 | 2,397.4 | 13 | 44 |
| 20261024 | 3704004 | -10,711.1 c | -6,716.0 c | +3,995.0 c | 7,014.4 | 6,939.6 | 9 | 33 |
| 20261024 | 3712725 | -13,128.9 c | -8,627.1 c | +4,501.8 c | 5,619.2 | 4,989.5 | 43 | 141 |
| 20261024 | 3716727 | -14,893.2 c | -8,803.1 c | +6,090.0 c | 5,233.9 | 4,552.8 | 18 | 38 |
| 20261025 | 3723774 | -14,024.2 c | +3,967.7 c | +17,991.9 c | 5,877.3 | 5,291.0 | 28 | 84 |
| 20261025 | 3734864 | -19,392.9 c | -13,490.9 c | +5,902.0 c | 5,925.0 | 5,537.4 | 10 | 20 |
| 20261025 | 3737919 | -15,136.8 c | -4,760.2 c | +10,376.6 c | 5,403.9 | 4,861.9 | 10 | 16 |
| 20261025 | 3763401 | -10,489.6 c | -9,775.8 c | +713.8 c | 6,794.5 | 6,404.6 | 24 | 66 |
| 20261025 | 3776314 | -7,452.3 c | -5,562.6 c | +1,889.6 c | 2,958.7 | 2,312.9 | 5 | 18 |
| 20261026 | 3789484 | -4,568.8 c | -1,593.3 c | +2,975.5 c | 5,350.4 | 1,922.1 | 83 | 288 |
| 20261026 | 3803131 | -1,828.4 c | +7,504.7 c | +9,333.1 c | 4,204.5 | 3,421.4 | 45 | 113 |
| 20261026 | 3804696 | -17,833.7 c | +2,415.5 c | +20,249.2 c | 4,832.4 | 1,342.1 | 17 | 39 |
| 20261026 | 3838531 | -4,906.1 c | -9,523.9 c | -4,617.8 c | 2,930.9 | 4,371.0 | 39 | 135 |
| 20261026 | 3850428 | -16,263.2 c | -8,078.1 c | +8,185.1 c | 7,052.2 | 5,456.5 | 9 | 26 |
| 20261027 | 3877472 | -18,314.4 c | -16,661.9 c | +1,652.5 c | 5,635.9 | 3,582.6 | 35 | 114 |
| 20261027 | 3899224 | -15,059.5 c | +2,460.3 c | +17,519.8 c | 4,264.0 | 1,102.5 | 6 | 16 |

- 18 markets gain and 1 lose; the worst is -4,617.8 c.
- The three largest gains carry 45.17 % of the total.

### Interventions, removals, lifetimes

`withheld slot proposals` count slot decisions, one per vacant slot per 50 ms tick while ON; they are not orders. `reducing-side` means bids at I < 0 and asks at I > 0.

| day | arm | pressure pulls | withheld proposals | of them near END | reducing-side | pressure PULL requests | inventory PULL requests | structural requests | generations | reviewed | first-review pulls | of them pressure | bid mean life | ask mean life | bid median life | ask median life |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | baseline | 0 | 0 | 0 | 0 | 0 | 47 | 1,576 | 1,919 | 1,759 | 1 | 0 | 3.11 s | 3.79 s | 1.08 s | 1.48 s |
| 20261024 | pressure | 122 | 354 | 0 | 288 | 122 | 39 | 1,538 | 2,004 | 1,858 | 5 | 5 | 3.24 s | 3.64 s | 1.13 s | 1.45 s |
| 20261025 | baseline | 0 | 0 | 0 | 0 | 0 | 35 | 1,083 | 1,318 | 1,210 | 0 | 0 | 3.41 s | 3.53 s | 1.54 s | 1.43 s |
| 20261025 | pressure | 77 | 204 | 0 | 165 | 77 | 22 | 1,094 | 1,398 | 1,271 | 2 | 1 | 3.32 s | 3.50 s | 1.44 s | 1.68 s |
| 20261026 | baseline | 0 | 0 | 0 | 0 | 0 | 26 | 1,291 | 1,502 | 1,385 | 3 | 0 | 2.94 s | 2.88 s | 1.38 s | 1.58 s |
| 20261026 | pressure | 193 | 601 | 0 | 432 | 193 | 18 | 774 | 1,104 | 1,023 | 18 | 17 | 2.96 s | 2.91 s | 1.28 s | 1.53 s |
| 20261027 | baseline | 0 | 0 | 0 | 0 | 0 | 9 | 453 | 549 | 486 | 0 | 0 | 3.03 s | 3.23 s | 1.62 s | 1.97 s |
| 20261027 | pressure | 41 | 130 | 0 | 95 | 41 | 5 | 207 | 305 | 270 | 0 | 0 | 4.82 s | 3.68 s | 2.82 s | 2.09 s |
| all | baseline | 0 | 0 | 0 | 0 | 0 | 117 | 4,403 | 5,288 | 4,840 | 4 | 0 | 3.14 s | 3.42 s | 1.48 s | 1.58 s |
| all | pressure | 433 | 1,289 | 0 | 980 | 433 | 84 | 3,613 | 4,811 | 4,422 | 25 | 23 | 3.36 s | 3.42 s | 1.35 s | 1.58 s |

### Inventory (exact, from the fills) and exact uptime

| day | arm | mean |I| | mean peak |I| | time |I| >= 100 | mean terminal |I| | both-side active | bid active | ask active |
|---|---|---|---|---|---|---|---|---|
| 20261024 | baseline | 82.4 | 181.0 | 37.85 % | 83.1 | 39.85 % | 65.02 % | 72.07 % |
| 20261024 | pressure | 82.8 | 181.0 | 37.58 % | 85.8 | 46.34 % | 68.76 % | 75.03 % |
| 20261025 | baseline | 84.6 | 181.8 | 37.93 % | 80.9 | 41.16 % | 66.82 % | 71.66 % |
| 20261025 | pressure | 82.4 | 179.1 | 33.44 % | 80.2 | 45.69 % | 68.10 % | 74.89 % |
| 20261026 | baseline | 86.0 | 186.5 | 33.71 % | 99.6 | 41.77 % | 66.66 % | 66.21 % |
| 20261026 | pressure | 84.9 | 180.6 | 25.58 % | 104.5 | 33.85 % | 57.17 % | 48.69 % |
| 20261027 | baseline | 80.9 | 185.6 | 36.94 % | 88.1 | 39.84 % | 65.61 % | 69.38 % |
| 20261027 | pressure | 88.9 | 170.0 | 16.31 % | 99.7 | 20.77 % | 35.58 % | 34.87 % |
| all | baseline | 83.8 | 183.1 | 36.69 % | 87.4 | 40.70 % | 65.99 % | 70.13 % |
| all | pressure | 83.9 | 179.2 | 31.09 % | 90.7 | 40.19 % | 62.05 % | 63.83 % |

### Queue diagnostics (review 2c5dbd53 finding 2)

Venue values are the engine's queue ahead in shares; the receipt-known estimate is kept apart. Market means.

| day | arm | venue ahead after submission | venue ahead at first review | known estimate at first review | pressure pulls | venue ahead at the pull | re-entries | gap to re-entry | re-entry venue ahead | re-entry - pull ahead | re-entry at the same price |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | baseline | 129.9 | 137.9 | 126.7 | 0 | — | 0 | — | — | — | — |
| 20261024 | pressure | 123.7 | 129.5 | 118.1 | 122 | 65.4 | 122 | 1.30 s | 68.6 | 3.2 | 5.09 % |
| 20261025 | baseline | 196.7 | 209.2 | 195.0 | 0 | — | 0 | — | — | — | — |
| 20261025 | pressure | 191.3 | 203.9 | 191.1 | 77 | 85.8 | 77 | 0.89 s | 26.4 | -59.4 | 7.48 % |
| 20261026 | baseline | 363.3 | 389.0 | 346.5 | 0 | — | 0 | — | — | — | — |
| 20261026 | pressure | 372.1 | 400.2 | 355.6 | 193 | 248.4 | 190 | 0.91 s | 206.8 | -41.3 | 13.53 % |
| 20261027 | baseline | 320.2 | 355.5 | 304.8 | 0 | — | 0 | — | — | — | — |
| 20261027 | pressure | 392.4 | 428.5 | 383.6 | 41 | 152.3 | 41 | 0.39 s | 104.9 | -47.5 | 19.52 % |
| all | baseline | 228.9 | 245.6 | 221.2 | 0 | — | 0 | — | — | — | — |
| all | pressure | 235.1 | 251.8 | 227.8 | 433 | 128.1 | 430 | 1.00 s | 97.7 | -30.3 | 9.46 % |

## Selection control

Saved audit live states of both sides; 500 matched draws, seed 20261121; strata include side and the B >= 200 indicator. This is not a null for the feedback policy.

- **Population:** 111 selected states of 4,831 uncensored (9 censored, 0 of them selected; 0 with a missing value).
- **Strata:** 887 in all; 59 movable with 104 selected states; 5 degenerate.
- **Pooled (the answer):** selected mean window +1 s -97.7 c against the matched null mean -32.1 c (p05 -58.8 c, p95 -8.1 c); null percentile 0.00 %.
- **Per valid window share:** -10.171 c / share (null p05 -5.372 c / share).
- **Window quantities:** valid 1,066.6, total 1,066.6, excluded 0.0 shares.
- **Per side (descriptive):**
  - bids: 51 selected, -26.4 c against null mean -8.6 c and p05 -35.9 c; percentile 13.20 %;
  - asks: 60 selected, -158.4 c against null mean -52.1 c and p05 -100.8 c; percentile 0.00 %.

## Computed readings (measure/audit.json predicates)

- **END_paired:** IMPROVES_EVERY_DAY.
  - Total +123,448.1 c; by day 20261024 +31,276.8 c, 20261025 +36,873.9 c, 20261026 +36,125.1 c, 20261027 +19,172.3 c.
  - Markets better / worse / equal: 18 / 1 / 0.
  - END by side: bid MIXED, ask MIXED.
- **+1 s per share (shared cutoff):** all IMPROVES_EVERY_DAY, bid IMPROVES_EVERY_DAY, ask IMPROVES_EVERY_DAY.
  - All sides by day: 20261024 +0.766 c / share, 20261025 +1.293 c / share, 20261026 +0.704 c / share, 20261027 +1.559 c / share.
- **Participation:**
  - filled shares by day: 20261024 -1,968.0, 20261025 -2,551.6, 20261026 -7,857.1, 20261027 -5,214.9;
  - exact active time: both sides -0.51 pp, bids -3.94 pp, asks -6.30 pp.
- **Inventory (exact):** mean |I| 0.1 shares; time at |I| >= 100 -5.59 pp; terminal |I| 3.3.
- **Selection control:** SELECTION_BEYOND_EXPOSURE.
- **Interpretation (declared rule):** better_selection.

## Review 2c5dbd53 findings

1. **Checker reconciliation (finding 1).** The checker's F family recomputes every field a reading uses from the fills, mids, orders and logs. It derives the readings, and the selection answer from the manifest's draws, seed and bands, itself. The falsifier's coordinated case (a markout field and the stored reading moved together) is refused.
2. **Queue diagnostics (finding 2).** Venue values are recorded after submission, at the first live review and at every pressure pull, apart from the receipt-known estimates; the re-entry is recorded too.
3. **Selftest coverage (finding 3).** The new selftest names only what it runs. It tests every run_day guard in isolation and asserts its message.
- **Interpretation details** (no repair cycle needed): inventory is now exact from the fills, and withholds are reported as withheld slot proposals with the near-END ones counted apart.

## Validity

- **Checks** A-I: 22 / 22 true. Falsifier: 14 / 14 corruptions refused by their family, including the coordinated markout-and-reading case.
- **Rehearsal** on the pilot copy before the full run: 21 / 21 checks true. B_membership is false there by design.
- **Selftest and mutants:** mutation suite 29 / 29; selftest groups 1-7 pass.
- **Cost:** pilot projection 0.537 core-hours. The round used 0.398 core-hours (every run_light record under logs/, the pre-freeze probe included) of the 0.75 cap; peak 1,276 MiB.

## Limits

- **Selection of the feature and its edge.** This is one external signal on the 19 complete consumed-day markets. The audit of these same markets suggested adv_bn_r250 (the largest pooled separation, which failed its all-eight-cells rule), and the edges are its Oct 24-25 bins. No day here is a validation day.
- **Replay assumptions.** External flow is unchanged: pulled and withheld orders move no one else. Timing is M50. Values are gross and zero cost, and END is a pre-expiry mark.
- **Selection control.** It matches on day × market × side × remaining × age × inventory sign × the reservation indicator only. It tests the local association, not the feedback policy.
- **Path coupling.** Paired values pass through the path-coupled inventory rule (CLAUDE.md pitfall 4); the three largest gains carry 45.17 % of the total.
