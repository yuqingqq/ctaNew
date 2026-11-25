# TAKE-REPRICE v1 — takes only on large move-implied repricings (2026-11-25)

[declaration.md](declaration.md) (frozen before any run). 408 complete markets, five consumed days; thresholds informed by the winners on these days (optimistic by construction). Venue-matched harness; comparator STREAM-172.

| Config | Total | lower / upper | days + | takes sent | filled | limit moved | taker sh | exits | reverse | held | take value 1 s / 10 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| REPRICE-4-172 | -130.44 | -133.13 / -129.88 | 1 / 5 | 6,132 | 215 | 82% | 8,191 | -89.82 | -4.73 | -35.89 | +0.27 / +0.65 |
| REPRICE-8-172 | -47.29 | -49.98 / -44.49 | 2 / 5 | 3,789 | 99 | 85% | 3,680 | -23.23 | -2.32 | -21.74 | +1.22 / +0.31 |
| REPRICE-16-172 | -23.67 | -25.29 / -20.87 | 1 / 5 | 1,556 | 30 | 87% | 1,129 | +24.18 | +0.00 | -47.85 | +1.20 / +6.10 |
| REPRICE-8-192 | -58.95 | -61.62 / -58.95 | 3 / 5 | 3,786 | 83 | 93% | 3,174 | -53.23 | -1.91 | -3.80 | +0.56 / +0.39 |
| STREAM-172 | -379.42 | -391.13 / -361.30 | 1 / 5 | 8,124 | 639 | 77% | 25,310 | -259.34 | -58.68 | -61.40 | -0.30 / +0.01 |

## vs STREAM-172 (by day)

- REPRICE-4-172: +248.99, positive on 4 / 5 days
- REPRICE-8-172: +332.13, positive on 4 / 5 days
- REPRICE-16-172: +355.75, positive on 4 / 5 days
- REPRICE-8-192: +320.47, positive on 5 / 5 days

Triggers (all markets): {'4': 6280, '8': 3843, '16': 1573}

## Readings (computed)

- `beats_stream_4`: True
- `beats_stream_8`: True
- `beats_stream_16`: True
- `profitable_4`: False
- `profitable_8`: False
- `profitable_16`: False
- `best`: REPRICE-16-172
- `multiplicity`: 3
- `holds_at_192`: False

## Checks

- PASS `population`: 408 complete markets
- PASS `take_timing`: 0 takes off decision + network + 150 ms
- PASS `accounting_rederived`: 1632 ledgers, 0 mismatches
- PASS `min_buy`: 0 sent takes below 1 USDT
- PASS `bounds`: 0 rows outside their bounds
- PASS `selftest`: dFV trigger: near-strike fires, far does not, quiet suppresses
