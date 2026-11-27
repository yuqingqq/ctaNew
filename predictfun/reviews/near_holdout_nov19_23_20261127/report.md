# Frozen passive inventory rule on November 19–23, 2026

All five test dates were assigned before reading outcomes. The census includes 352 complete market paths from 1,440 scheduled windows.

## Five-day decision

- Nominal execution. Absolute profit screen failed. Comparative screen passed.
  Failed criteria were positive without best day, profitable on a majority of dates.
- Short execution. Absolute profit screen passed. Comparative screen failed.
  Failed criteria were settlement at least as high as original, higher settlement than the midpoint time-only control.

| Timing | Policy | FIFO | Residual at payout | Settlement | Positive dates | Mean abs end, shares | Worst 10% abs end, shares | Worst 10% mean loss | Matched shares |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | near | $551.42 | −$238.81 | $312.61 | 2/5 | 7.60 | 65.35 | $55.29 | 206,870 |
| nominal | original | $592.38 | −$331.36 | $261.02 | 2/5 | 8.93 | 75.05 | $61.36 | 222,912 |
| nominal | time_only | $542.78 | −$260.56 | $282.23 | 2/5 | 6.80 | 58.27 | $57.03 | 212,581 |
| short | near | $1,458.73 | −$306.28 | $1,152.45 | 4/5 | 8.94 | 74.59 | $51.82 | 190,511 |
| short | original | $1,653.06 | −$487.72 | $1,165.34 | 4/5 | 10.57 | 84.41 | $57.83 | 205,215 |
| short | time_only | $1,651.80 | −$388.30 | $1,263.49 | 4/5 | 8.38 | 69.29 | $54.36 | 195,291 |

| UTC date | Eligible / 288 | Original nominal | Time-only nominal | Near nominal | Original short | Time-only short | Near short |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20261119 | 76 | −$72.27 | −$103.29 | −$101.70 | $222.40 | $87.21 | $9.48 |
| 20261120 | 59 | $211.25 | $101.66 | $132.21 | $423.10 | $422.07 | $433.24 |
| 20261121 | 66 | −$548.95 | −$491.06 | −$503.07 | −$388.28 | −$336.46 | −$425.82 |
| 20261122 | 77 | $715.24 | $809.86 | $818.97 | $752.56 | $893.98 | $915.52 |
| 20261123 | 74 | −$44.25 | −$34.95 | −$33.79 | $155.56 | $196.68 | $220.02 |

Coverage screen passed. Overall predeclared decision: **did not pass**.

## Coverage, execution and verification

- 20261119 — missing_endpoint 168, complete 76, stale_future_book 44.
- 20261120 — missing_endpoint 176, complete 59, stale_future_book 53.
- 20261121 — missing_endpoint 193, complete 66, stale_future_book 29.
- 20261122 — missing_endpoint 164, complete 77, stale_future_book 47.
- 20261123 — missing_endpoint 171, complete 74, stale_future_book 42, missing_future_book 1.

Economics apply to eligible tapes and include zero-fill markets. Missing outcomes carry unknown P&L. Reported dollars use zero maker fees/rebates, fixed external flow, modeled queue and the recorded YES payout including ties. Nominal placement takes 58 ms with 37 ms live cancellation. Short eligibility suppresses matches after a cancel request while retaining nominal reservation and notice timing. Neither profile measures real wallet fills. The absolute screen removes the best day and best market separately. The tail mean uses the largest ceil(n/10) positions or losses in each pooled policy.

The October 26 control reproduced every census status, all 70 tape identities, timing calibration, BTC grid and intraday pressure series, and the original and selected policy ledgers on the first eligible market. The readback checks pair membership, the inventory cap, FIFO plus residual at settlement, and a corrupted residual. The UTC day is the unit for interpreting signs. The December 1–5 prospective test is separate.

[Frozen protocol](protocol.md) · [Per-day coverage and computed checks](result.json) · [Input hashes](inputs.json)
