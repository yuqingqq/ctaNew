# TAKE-EXIT VENUE-MATCHED COMPARISON v1 — BASE vs FLOOR under venue rules (2026-11-25)

[declaration.md](declaration.md) (declared before any run; resource amendment R1 before stage 2). Review 4e67e0856. Five consumed v3 days; development evidence only.

## Stage 1 — the speed bump alone (408 complete markets, settlement, ties 50 / 50)

| | 58 ms (v1 harness) | 208 ms (+150 ms speed bump) | speed-bump cost |
|---|---:|---:|---:|
| BASE | +1,880.64 | -230.03 | -2,110.67 |
| FLOOR | +2,360.47 | +73.56 | -2,286.91 |

| 172 ms (R3: ~22 ms one way + 150 ms hold) | total | days positive | takes with the limit moved |
|---|---:|---:|---:|
| BASE | -293.52 | 0 / 5 | 85.8% of 8,228 sent |
| FLOOR | -64.33 | 2 / 5 | 85.9% of 8,042 sent |

Stage 1 checks: ties = 8 True; the 58 ms values reproduce the reviewer's independent recount {'BASE': True, 'FLOOR': True}.

## Stage 2 — venue-matched (1439 of 1440 scheduled markets replayed)

Path status of replayed markets: {'complete_no_mark': 546, 'complete': 408, 'truncated': 485}.

### VM208-M50 (primary)

| | BASE | FLOOR |
|---|---:|---:|
| Total at settlement | -1,118.03 | -444.85 |
| … per scheduled market | -0.78 | -0.31 |
| … lower bound (unresolved orders) | -1,130.28 | -469.90 |
| … upper bound | -1,063.85 | +205.17 |
| Round trips after fees | -1,114.80 | +54.29 |
| … closed by the maker exit | -737.80 | +1,698.57 |
| … closed by an opposite take | -377.00 | -1,644.28 |
| Held to settlement | -3.23 | -499.14 |
| Cost of tokens held at the end | +11,321.88 | +15,727.57 |
| C1: total without exit fills < 100 ms old | -1,001.35 | -349.03 |
| C3: cancel-pending exit fills ($) | -450.32 | -350.15 |
| Complete markets only (408) | -337.74 | -38.19 |
| Days positive | 1 | 2 |
| Taker shares | 143,918 | 137,315 |
| Tokens held at the end | 15,159 | 20,933 |
| Markets holding ≥ 1 token | 355 | 488 |
| Takes cut by the 1 USDT minimum (min_capped) | 1,026 | 1,905 |

By day (FLOOR / BASE / difference): 20261023 -170.46 / -387.86 / +217.40; 20261031 +68.56 / +10.03 / +58.54; 20261101 +16.18 / -45.31 / +61.49; 20261102 -271.57 / -308.92 / +37.35; 20261103 -87.56 / -385.97 / +298.40

Take outcomes (FLOOR): {'limit_moved': 24194, 'partial': 657, 'filled': 2601, 'strict_cancelled': 376, 'invalid_or_stale': 88, 'empty': 145}; signal statuses: {'sent': 28061, 'own_cross': 565, 'no_room': 1498, 'no_fresh_book': 2514, 'near_end': 21, 'pending_ioc': 340, 'below_min': 790}.

### VM258-M50

| | BASE | FLOOR |
|---|---:|---:|
| Total at settlement | -1,297.67 | -532.81 |
| … per scheduled market | -0.90 | -0.37 |
| … lower bound (unresolved orders) | -1,308.69 | -556.29 |
| … upper bound | -1,243.22 | +147.16 |
| Round trips after fees | -1,281.16 | -0.23 |
| … closed by the maker exit | -902.58 | +1,621.60 |
| … closed by an opposite take | -378.58 | -1,621.83 |
| Held to settlement | -16.51 | -532.57 |
| Cost of tokens held at the end | +11,086.57 | +14,865.36 |
| C1: total without exit fills < 100 ms old | -1,169.52 | -341.09 |
| C3: cancel-pending exit fills ($) | -479.35 | -377.35 |
| Complete markets only (408) | -324.82 | -8.50 |
| Days positive | 0 | 2 |
| Taker shares | 139,492 | 132,783 |
| Tokens held at the end | 15,233 | 20,333 |
| Markets holding ≥ 1 token | 348 | 473 |
| Takes cut by the 1 USDT minimum (min_capped) | 1,044 | 1,901 |

By day (FLOOR / BASE / difference): 20261023 -209.79 / -467.44 / +257.64; 20261031 +35.39 / -21.31 / +56.70; 20261101 +4.80 / -66.20 / +71.00; 20261102 -315.26 / -361.30 / +46.03; 20261103 -47.94 / -381.43 / +333.49

Take outcomes (FLOOR): {'limit_moved': 24450, 'partial': 579, 'filled': 2516, 'strict_cancelled': 221, 'invalid_or_stale': 113, 'empty': 79}; signal statuses: {'sent': 27958, 'own_cross': 545, 'no_room': 1391, 'no_fresh_book': 2569, 'near_end': 24, 'pending_ioc': 407, 'below_min': 895}.

### VM208-M250

| | BASE | FLOOR |
|---|---:|---:|
| Total at settlement | -1,694.28 | -1,055.85 |
| … per scheduled market | -1.18 | -0.73 |
| … lower bound (unresolved orders) | -1,703.50 | -1,069.40 |
| … upper bound | -1,648.31 | -455.00 |
| Round trips after fees | -1,672.33 | -513.23 |
| … closed by the maker exit | -1,275.72 | +1,060.22 |
| … closed by an opposite take | -396.60 | -1,573.46 |
| Held to settlement | -21.95 | -542.62 |
| Cost of tokens held at the end | +11,124.46 | +15,089.53 |
| C1: total without exit fills < 100 ms old | -1,374.44 | -852.02 |
| C3: cancel-pending exit fills ($) | -63.40 | +111.35 |
| Complete markets only (408) | -534.89 | -305.96 |
| Days positive | 0 | 0 |
| Taker shares | 145,273 | 138,589 |
| Tokens held at the end | 14,491 | 19,819 |
| Markets holding ≥ 1 token | 342 | 461 |
| Takes cut by the 1 USDT minimum (min_capped) | 914 | 1,699 |

By day (FLOOR / BASE / difference): 20261023 -386.17 / -572.97 / +186.80; 20261031 -10.29 / -121.60 / +111.31; 20261101 -14.31 / -74.76 / +60.45; 20261102 -348.18 / -464.61 / +116.43; 20261103 -296.91 / -460.34 / +163.43

Take outcomes (FLOOR): {'limit_moved': 24559, 'partial': 669, 'filled': 2627, 'strict_cancelled': 481, 'empty': 66, 'invalid_or_stale': 90}; signal statuses: {'sent': 28492, 'own_cross': 394, 'no_room': 1273, 'no_fresh_book': 2535, 'near_end': 21, 'pending_ioc': 350, 'below_min': 724}.

## Readings (computed)

- `survives_venue_match`: False
- `floor_beats_base_venue`: True
- `holds_at_258`: False
- `holds_at_m250`: False
- `base_survives_venue_match`: False

## Checks

- PASS `K9_census`: 1440 scheduled markets, statuses {'complete_no_mark': 546, 'complete': 408, 'truncated': 485, 'boundary_censored_forward_tail': 1}
- PASS `K1_path_parity`: 408 / 408 complete markets identical
- PASS `K2_off_parity`: 24 market-arms
- PASS `K3_accounting_rederived`: 8634 ledgers re-accounted, 0 mismatches
- PASS `K4_min_buy`: 0 sent takes with a buy part below 1 USDT
- PASS `K4_falsifier_v1_flagged`: 81 below-minimum takes in the first 40 complete v1 FLOOR ledgers
- PASS `K5_queue_and_native_selftests`: queue L vs M, native book, minimum, accounting bounds
- PASS `K6_ties`: 8 tie markets paid 50, payout mismatches 0
- PASS `K7_bounds`: 0 rows outside their bounds
- PASS `K8_delays`: take delay mismatches 0, exit activation mismatches 0

## Limits

- Consumed days. The speed bump applied on these days: the venue changelog (read 2026-11-25) dates removalLockedUntil for non-post-only orders on up/down markets with a speed bump to 2026-06-26 and the speedBump market field to 2026-11-10; the value on these days is not recorded (150 ms now).
- Our order is not in the recorded book: competitors' reactions to it are not modelled. The strict two-frame IOC rule and queue L are conservative; the 1 s staleness rule truncates paths at recorder gaps.
- A resting exit that would oversell after a take is dropped in the token accounting but not in the engine path (counted as oversell).
