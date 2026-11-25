# TAKE-STREAM v1 — stream-decided takes and a book-shrinking trigger under venue rules (2026-11-25)

[declaration.md](declaration.md) (frozen before any run). 408 complete markets on the five consumed v3 days; development evidence. Every config uses the venue-matched harness (150 ms hold, 1 USDT minimum, fees in shares, queue L, exits 58 / 37 / 95 ms, settlement).

| Config | Total at settlement | lower / upper | days + | maker exits | reverse takes | held | taker shares | limit moved | take value 1 s / 10 s (¢/sh) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GRID-172 | -372.07 | -384.10 / -353.95 | 0 / 5 | -278.28 | -54.60 | -39.19 | 24,564 | 86% | -0.49 / +0.06 |
| STREAM-172 | -379.42 | -391.13 / -361.30 | 1 / 5 | -259.34 | -58.68 | -61.40 | 25,310 | 77% | -0.30 / +0.01 |
| STREAM-192 | -283.91 | -295.83 / -270.81 | 1 / 5 | -266.39 | -32.93 | +15.41 | 23,466 | 86% | -0.47 / +0.05 |
| STREAM+DEPL-172 | -5,562.75 | -5,622.90 / -5,368.95 | 0 / 5 | -1,869.60 | -3,339.39 | -353.76 | 371,025 | 49% | -0.81 / -0.55 |
| STREAM+DEPLS-172 | -2,319.53 | -2,330.83 / -2,243.47 | 0 / 5 | -1,968.62 | -236.99 | -113.92 | 105,508 | 63% | -1.17 / -0.76 |
| STREAM+DEPLS-192 | -2,030.47 | -2,042.34 / -1,947.22 | 0 / 5 | -1,777.54 | -217.97 | -34.96 | 97,152 | 72% | -1.16 / -0.72 |

## Differences (by day)

- STREAM-172 - GRID-172: -7.35, positive on 4 / 5 days (20261023 +5.96, 20261031 +10.66, 20261101 +17.18, 20261102 +1.02, 20261103 -42.17)
- STREAM+DEPL-172 - STREAM-172: -5,183.32, positive on 0 / 5 days (20261023 -1,211.80, 20261031 -1,149.22, 20261101 -434.29, 20261102 -1,052.60, 20261103 -1,335.41)
- STREAM+DEPLS-172 - STREAM-172: -1,940.11, positive on 0 / 5 days (20261023 -318.67, 20261031 -537.95, 20261101 -40.95, 20261102 -432.13, 20261103 -610.40)

## Triggers per day

- 20261023: {'depl': 28871, 'stream': 9283, 'depls': 4881}
- 20261031: {'depl': 14459, 'stream': 3830, 'depls': 2137}
- 20261101: {'depl': 10399, 'stream': 2923, 'depls': 1423}
- 20261102: {'depl': 28846, 'stream': 9922, 'depls': 6049}
- 20261103: {'depl': 28093, 'stream': 9681, 'depls': 6538}

## Readings (computed)

- `stream_beats_grid`: True
- `depletion_adds`: False
- `strict_depletion_adds`: False
- `best_trigger_set_172`: STREAM-172
- `profitable_172`: False
- `multiplicity`: 3
- `holds_at_192`: False

## Checks

- PASS `population`: 408 complete markets
- PASS `K1_grid_parity`: 12 markets
- PASS `K4_take_timing`: 0 takes off decision + network + 150 ms
- PASS `K5_accounting_rederived`: 2448 ledgers, 0 mismatches
- PASS `K5_bounds`: 0 rows outside their bounds
- PASS `K5_min_buy`: 0 sent takes with a buy part below 1 USDT
- PASS `K2_K3_selftests`: stream / depletion triggers, the venue harness

## Limits

- Consumed days; complete markets only (stage A). Our order is not in the recorded book: competitors and makers react to it in reality.
- Decisions are causal (state at the tick before the decision, book known at the decision); the stream rule ignores window rolls between messages.
- The depletion rules were not tuned on outcomes; the strict floor comes from a day outside the test (sizes only).
