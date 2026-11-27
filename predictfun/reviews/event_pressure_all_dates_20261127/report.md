# Event-driven pressure cancels across 14 historical dates

The event-driven strategy settled at **+$3,632.02** on 1,041 eligible five-minute markets. The 50 ms control settled at **+$2,205.54**. The paired difference was +$1,426.48, with an improvement on 13 of 14 dates.

| Block | Eligible markets | 50 ms settlement | Event settlement | Difference | Dates improved | Event profitable dates | Delta without best improvement day |
|---|---:|---:|---:|---:|---:|---:|---:|
| Additional 9 dates | 689 | +$1,892.93 | +$2,773.69 | +$880.76 | 8/9 | 7/9 | +$564.37 |
| Previously reported 5 | 352 | +$312.61 | +$858.34 | +$545.73 | 5/5 | 3/5 | +$377.00 |
| All 14 dates | 1041 | +$2,205.54 | +$3,632.02 | +$1,426.48 | 13/14 | 10/14 | +$1,110.09 |

| UTC date | Eligible / 288 | 50 ms settlement | Event settlement | Difference | Event completed FIFO | Event residual at payout |
|---|---:|---:|---:|---:|---:|---:|
| 2026-10-23 | 70 | +$358.92 | +$388.11 | +$29.20 | +$434.63 | −$46.51 |
| 2026-10-24 | 81 | −$52.42 | −$14.05 | +$38.37 | +$56.07 | −$70.12 |
| 2026-10-25 | 72 | +$360.21 | +$384.96 | +$24.75 | +$545.56 | −$160.60 |
| 2026-10-26 | 70 | +$105.45 | +$237.97 | +$132.52 | +$409.28 | −$171.31 |
| 2026-10-27 | 58 | +$160.74 | +$158.04 | −$2.70 | +$297.97 | −$139.93 |
| 2026-10-31 | 102 | −$69.51 | +$39.03 | +$108.54 | +$927.80 | −$888.77 |
| 2026-11-01 | 88 | +$629.22 | +$710.37 | +$81.15 | +$984.83 | −$274.46 |
| 2026-11-02 | 72 | −$292.79 | −$140.24 | +$152.55 | −$87.87 | −$52.37 |
| 2026-11-03 | 76 | +$693.11 | +$1,009.50 | +$316.39 | +$1,061.50 | −$52.00 |
| 2026-11-19 | 76 | −$101.70 | −$2.17 | +$99.53 | +$7.48 | −$9.65 |
| 2026-11-20 | 59 | +$132.21 | +$233.10 | +$100.89 | +$252.79 | −$19.69 |
| 2026-11-21 | 66 | −$503.07 | −$438.27 | +$64.81 | −$351.81 | −$86.45 |
| 2026-11-22 | 77 | +$818.97 | +$930.75 | +$111.77 | +$1,063.79 | −$133.04 |
| 2026-11-23 | 74 | −$33.79 | +$134.93 | +$168.72 | +$189.67 | −$54.74 |

## Turnover, risk and queue churn

| Pooled metric | 50 ms | Event-driven |
|---|---:|---:|
| Completed FIFO profit | +$4,367.94 | +$5,791.68 |
| Residual P&L at payout | −$2,162.40 | −$2,159.66 |
| Settlement excluding best day | +$1,386.57 | +$2,622.53 |
| Settlement excluding best market | +$2,059.07 | +$3,509.67 |
| Mean absolute inventory at payout, shares | 9.98 | 9.93 |
| One-second markout, cents per eligible share | -1.486 | -1.385 |
| Shares filled | 978,013 | 946,880 |
| Shares in completed round trips | 483,814 | 468,270 |
| Submissions | 164,142 | 166,393 |
| Cancel requests | 143,075 | 145,302 |
| Same-lane reposts following a cancel | 120,553 | 122,712 |

The event strategy won on 546 of 1041 markets and broke even on 8. Its per-market settlement quantiles are shown below.

| Min | P10 | P25 | Median | P75 | P90 | Max |
|---:|---:|---:|---:|---:|---:|---:|
| −$99.77 | −$35.78 | −$19.95 | +$2.43 | +$26.24 | +$45.55 | +$122.35 |

## Interpretation and scope

- Additional 9 dates. The directional screen passed. The absolute settlement screen for event cancels passed.
- All 14 dates. The directional screen passed. The absolute settlement screen for event cancels passed.

The registry accounted for 4,032 scheduled markets. 1 boundary censored forward tail, 1,041 complete, 1,934 missing endpoint, 1 missing future book, 1,055 stale future book. Excluded markets have unknown economics.

One candidate. Both arms used the frozen BTC pressure thresholds and modeled feed arrival at exchange time plus 4 ms. Live cancel effect took 37 ms, and submissions and reposts remained on the 50 ms grid. Feed updates with an unchanged mid were eligible for pressure review. Both arms generated their own subsequent orders and fills. Public flow, queue matching and zero maker fees/rebates follow the original replay.

All dates have been used in research. The full sample describes historical behavior. The nine additional dates are reported separately because the five-day event result was already known before this extension. The 50 ms policy reproduced its archived ledger fields for every eligible market. Results do not measure exchange-side cancellation or actual wallet fills.

[Protocol](protocol.md) · [Daily figures, checks, coverage and file hashes](result.json) · [Original five-date report](../event_pressure_cancel_20261127/report.md)
