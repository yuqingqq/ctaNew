# TAKE-EXIT FLOOR v1 — break-even floor on the maker exit, valued at settlement (2026-11-25)

[amendment.md](amendment.md) (declared before the run), code frozen at da209477c. 408 complete markets on 20261023, 20261031, 20261101, 20261102, 20261103 (the v3 days; consumed, development evidence). BASE = the saved v3 STRICT58 ledgers; FLOOR = STRICT58 with the exit never offered beyond break-even (average cost incl. taker fees). Shares open at END (298 s) settle on the market's resolved outcome.

## Money split (USD, all markets)

| | BASE | FLOOR | FLOOR − BASE |
|---|---:|---:|---:|
| Round trips (spread capture), after fees | +1,916.15 | +2,620.02 | +703.87 |
| … closed by the maker exit | +2,009.47 | +3,166.88 | +1,157.40 |
| … closed by an opposite take | -93.33 | -546.86 | -453.53 |
| Held to settlement, after fees | -34.43 | -258.24 | -223.81 |
| **Total at settlement** | +1,881.72 | +2,361.78 | +480.06 |
| Held, at the 298 s mark | -14.50 | -249.73 | -235.23 |
| Total at the 298 s mark | +1,901.65 | +2,370.29 | +468.64 |
| Fees (included above) | +444.68 | +429.44 | -15.23 |
| Settlement − mark | -19.93 | -8.51 | +11.42 |

Per market at settlement: BASE +4.61, FLOOR +5.79.

## By day, total at settlement (USD)

| Day | BASE | FLOOR | FLOOR − BASE |
|---|---:|---:|---:|
| 20261023 | +288.91 | +402.87 | +113.96 |
| 20261031 | +600.48 | +621.67 | +21.19 |
| 20261101 | +367.19 | +427.22 | +60.03 |
| 20261102 | +279.28 | +359.94 | +80.66 |
| 20261103 | +345.87 | +550.09 | +204.22 |
| Days positive | 5 / 5 | 5 / 5 | 5 / 5 |

## Risk carried to settlement

| | BASE | FLOOR |
|---|---:|---:|
| Markets with a position at END | 340 | 337 |
| … of which ≥ 1 share | 44 | 73 |
| Shares held to settlement | 1,242.8 | 2,284.6 |
| Share of shares taken | 1.6 % | 3.0 % |
| Shares per market with a position, p50 | 0.6 | 0.6 |
| … p90 | 7.1 | 29.2 |
| … max | 50.0 | 50.0 |
| Loss if every held lot settles against us (USD) | +798.90 | +1,495.18 |
| … largest one market (USD) | +49.18 | +49.51 |
| Held shares that won | 61.5 % | 54.1 % |
| Settlement P&L of held lots per market: mean (USD) | -0.10 | -0.77 |
| … std (USD) | +2.19 | +4.69 |
| … p5 (USD) | -0.78 | -5.90 |
| … worst (USD) | -22.95 | -48.17 |
| … best (USD) | +18.69 | +18.69 |

## What the floor changed

- FLOOR exit posts: 7,602; priced up to the floor (above the touch): 1,457; cancelled because the floor rose: 0.
- FLOOR structural cancels: distance 991, cross 1,988, no book 7, floor 0.
- Shares closed by the maker exit: BASE 59,504.7, FLOOR 49,738.7; by an opposite take: BASE 8,571.6, FLOOR 11,694.6; shares taken: BASE 77,780.7, FLOOR 75,195.3.
- Paired per market at settlement: FLOOR better in 200, worse in 66, within ½ ¢ in 142; the top 10 markets carry 19.6 % of |difference|.

## FLOOR round trips by how they closed ([fill_split.json](fill_split.json))

`touch`: the exit was posted at the best price (the floor did not bind at that post, including re-posts after the price came back); `floor`: the exit was raised above the touch to break-even and rested away from it, where the replay's optimistic queue rule matters most; `take`: an opposite-signal take closed the position.

| Closed by | Round-trip shares | Round trips after fees (USD) | ¢ / share | Exit fill age p50 / p90 (ms) |
|---|---:|---:|---:|---:|
| touch | 42,883.7 | +3,139.36 | 7.32 | 554 / 5,376 |
| floor | 6,855.0 | +27.52 | 0.40 | 1,780 / 8,570 |
| take | 11,694.6 | -546.86 | -4.68 | — |

## Readings (computed)

- `floor_better_at_settlement`: True
- `floor_better_at_mark`: True
- `floor_round_trips_higher`: True
- `floor_carries_more`: True
- `base_settlement_total_positive`: True
- `floor_settlement_total_positive`: True

## Checks

- PASS `K1_census_equals_v3`: 408 complete markets
- PASS `K1_base_equals_v3`: max |diff| 3.65e-11 c, pooled 1901.65 USD
- PASS `K2_parity`: 11 markets, 260 exit posts
- PASS `K3_floor_respected`: 7602 FLOOR exit orders, 0 below the floor, 0 without a basis
- PASS `K3_falsifier_base_flagged`: 2652 of 8883 BASE exit orders beyond break-even; first (3655842, (38, 75.0, 77.0))
- PASS `K4_logged_floor_rederived`: 7602 posts, 0 mismatches
- PASS `K5_accounting_rederived`: identities enforced inside account()
- PASS `K6_yup_delays_rung0_bound`: yup bad 0, delay 0, closing/rung0 0, floor-bound posts 1457
- First-day projection 684 CPU s against the 900 s cap (go = True).

## Limits

- Consumed days, one floor level fixed before the run; no validation claim. Five day clusters: no interval.
- The replay is path-coupled: the floor changes positions, so later takes (room, flips) differ too; per-market differences are noisy.
- Exit fills keep the replay's optimistic queue rule (shrinking shown size assumed ahead of us) and 10 ms exit latency; a cross cancels without a fill. A floor order resting away from the touch leans on the same queue rule when the price comes back.
- Settlement uses the venue's resolved outcome for each market; trading stops at 298 s, so what is held then is held to resolution.
