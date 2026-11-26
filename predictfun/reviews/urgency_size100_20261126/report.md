# Order-size sensitivity in the passive urgency controller

The 100-share cap earned +$22,244.34 across 689 complete markets. The matched 220-share control earned +$18,260.40. The smaller size had positive settlement on 9/9 days and higher settlement than 220 on 6/9.

| Date | Markets | Settlement at 100 | Settlement at 220 | Difference | ¢ per matched share, 100 / 220 | Mean absolute ending shares, 100 / 220 |
|---|---:|---:|---:|---:|---:|---:|
| 20261023 | 70 | +$2,173.43 | +$1,642.56 | +$530.86 | 3.20 / 2.06 | 28.4 / 30.1 |
| 20261024 | 81 | +$1,476.88 | +$1,592.92 | −$116.03 | 2.61 / 2.39 | 30.9 / 31.5 |
| 20261025 | 72 | +$3,355.07 | +$3,524.46 | −$169.39 | 6.03 / 5.26 | 28.2 / 31.3 |
| 20261026 | 70 | +$2,636.08 | +$1,774.49 | +$861.58 | 4.12 / 2.16 | 47.1 / 38.3 |
| 20261027 | 58 | +$1,710.96 | +$903.99 | +$806.97 | 3.27 / 2.15 | 38.5 / 55.0 |
| 20261031 | 102 | +$2,278.78 | +$2,308.88 | −$30.10 | 7.74 / 6.88 | 122.6 / 132.2 |
| 20261101 | 88 | +$3,396.71 | +$2,664.97 | +$731.74 | 8.86 / 5.95 | 48.2 / 51.0 |
| 20261102 | 72 | +$2,146.66 | +$1,640.53 | +$506.13 | 2.92 / 1.36 | 28.4 / 24.8 |
| 20261103 | 76 | +$3,069.77 | +$2,207.59 | +$862.18 | 4.46 / 2.93 | 23.9 / 33.7 |

## Aggregate economics

| Population and maximum size | Completed FIFO | Residual | Settlement | Matched shares | ¢ / matched share | Mean / p90 abs. ending shares | Cancellation requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| All nine dates, 100 | +$28,515.68 | −$6,271.35 | +$22,244.34 | 627,323 | 4.546 | 47.38 / 202.34 | 90,546 |
| All nine dates, 220 | +$25,071.86 | −$6,811.46 | +$18,260.40 | 772,619 | 3.245 | 50.95 / 189.11 | 94,799 |
| Later four dates, 100 | +$14,601.63 | −$3,709.72 | +$10,891.92 | 258,397 | 5.651 | 60.94 / 231.51 | 39,269 |
| Later four dates, 220 | +$12,830.05 | −$4,008.08 | +$8,821.97 | 319,711 | 4.013 | 66.05 / 263.88 | 41,551 |

## Scope and checks

As of November 26, 2026. Nine previously examined dates: boundary_censored_forward_tail 1, complete 689, missing_endpoint 1062, stale_future_book 840. Each daily table uses every complete market in the frozen census. October 23 predates the October 24–25 capacity fit. October 26–27 were strategy development dates. October 31–November 3 are reported separately as a later slice previously used in other research.

Both sizes use two rungs, the fixed 1,000-share inventory ceiling, the same time-dependent allowance, a 440-share same-side reservation ceiling including pending cancellations, the same frozen capacity lookup and the same queue, latency and payout accounting. Orders may be smaller than the maximum. Positive passive settlement excludes rebates and assumes no maker fee or change to public flow. The smaller size can change later orders, fills and position through policy feedback. This replay measures no return on capital.

All 220-share controls are bound to saved committed outputs. The lowest-ID complete market on each date reproduced all nine checked market fields. Each candidate submission was checked against 100; a synthetic 101-share order was refused. An independent read of the paired market CSVs reconciled the daily and pooled summaries and rejected a corrupted residual. Freeze: 4ed4049f2.

[Protocol](protocol.md) · [All metrics and coverage by date](result.json)
