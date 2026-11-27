# Passive inventory rule selected for forward evaluation

As of November 27, 2026 UTC, keep the near-focused inventory rule frozen for independent evaluation. The added October 26 replay improved nominal settlement by $40.01. Using the shorter fill lifetime changed that comparison to −$28.86. In both cases the selected policy settled positive.

| October 26, 70 eligible markets | Original | Selected near-focused rule |
|---|---:|---:|
| Nominal settlement | $65.44 | $105.45 |
| Nominal completed FIFO | $283.76 | $286.57 |
| Nominal residual at payout | −$218.31 | −$181.12 |
| Mean absolute ending inventory, shares | 10.38 | 9.73 |
| Short-lifetime settlement | $309.36 | $280.51 |

All 288 scheduled markets were accounted for. Of these, 124 lacked an endpoint and 94 had a stale book. Both policies ran on the 70 complete markets. Every original nominal ledger reproduced the archived comparison on the same tape, and every row reconciled filled shares, completed FIFO and residual at settlement. The independent file check rejected a deliberately altered settlement. October 26 was already used in development.

| Nine consumed dates, 689 complete of 2,592 scheduled | Original | Selected near-focused rule |
|---|---:|---:|
| Nominal settlement | $1,576.80 | $1,892.93 |
| Nominal positive dates | 7/9 | 6/9 |
| Nominal completed FIFO | $3,729.22 | $3,816.52 |
| Nominal residual at payout | −$2,152.43 | −$1,923.59 |
| Mean absolute ending inventory, shares | 13.04 | 11.19 |
| Short-lifetime settlement | $2,903.92 | $3,135.96 |

The increase of $316.13 in nominal settlement includes $228.84 of smaller residual losses and $87.29 of additional completed FIFO gains. External trade flow and queue progression are modeled as in the frozen simulator, with zero passive fees and rebates. Missing markets have unknown economics.

## Independent test and live decision

The default prospective block is December 1–5, 2026 UTC, preserving the existing November allocations. The five days and their feed tails become available for assessment from December 6 at 02:00 UTC, conditional on collector coverage. The November 27 filename preflight returned awaiting_close_and_tail on all five. The selected policy and the forward comparison criteria are pinned in the selection receipt. The new-date input adapter still needs to reproduce the historical input pipeline on a consumed day before scoring this block. No independent result exists yet.

The Tokyo passive-urgency branch was checked against its remote head d0440d63a190f627ae960a7e4cd7cf425fbde22e. Its Rust strategy uses a 100-share lot, a time-only inventory allowance and a separate operator risk configuration. A shadow build must first reproduce the selected research policy's 50-share lots, 150-share admission cap including retiring reservations, BTC input clock, and inventory-dependent quote decisions. The earlier wallet comparison credited 119.49 simulated closing shares on orders with no wallet fill. Shadow evidence can check decisions, time, market coverage and order lifetimes. Actual fill calibration ultimately requires comparable real orders. A successful forward replay and matching shadow would support considering a separately authorized small funded pilot.

[Selection and evaluation criteria](selection.md) · [October 26 ledger](markets.csv) · [Nine-date totals](nine_days.json) · [Original eight-date comparison](../urgency_far_extension_20261127/report.md) · [Tokyo wallet evidence](../tokyo_execution_comparison_20261126/report.md)
