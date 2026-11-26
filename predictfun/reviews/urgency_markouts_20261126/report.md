# 100-share urgency strategy markouts

Descriptive recount on November 26, 2026. The historical sample comprises the original 689 complete markets on nine consumed dates. Tokyo covers 11 different markets in one hour of shadow trading on November 26. All prices are in YES cents. Each fill contributes its signed difference between the future book midpoint and execution price, weighted by shares. A positive value favors the maker.

| Population | At fill | 1 s | 5 s | 30 s |
|---|---:|---:|---:|---:|
| Historical backtest | +0.864 | -1.699 | -1.710 | -1.214 |
| Tokyo crossing estimate | -0.760 | -1.490 | -2.010 | -0.863 |
| Tokyo crossing plus queue decreases | +0.709 | +0.035 | -0.240 | +0.294 |

## Historical markout by date, cents per eligible filled share

| UTC date | Markets | At fill | 1 s | 5 s | 30 s |
|---|---:|---:|---:|---:|---:|
| 20261023 | 70 | +0.785 | -1.533 | -1.818 | -1.799 |
| 20261024 | 81 | +0.888 | -1.086 | -1.241 | -1.848 |
| 20261025 | 72 | +1.053 | -1.459 | -1.674 | -1.595 |
| 20261026 | 70 | +0.672 | -1.488 | -1.253 | -0.895 |
| 20261027 | 58 | +0.885 | -1.640 | -1.734 | -1.751 |
| 20261031 | 102 | +0.873 | -2.019 | -2.480 | -1.196 |
| 20261101 | 88 | +1.006 | -2.100 | -2.013 | +0.032 |
| 20261102 | 72 | +0.827 | -1.937 | -1.616 | -0.784 |
| 20261103 | 76 | +0.892 | -2.212 | -1.776 | -0.695 |

Days with a negative historical markout: 0/9 at 0 s, 9/9 at 1 s, 9/9 at 5 s, 8/9 at 30 s.

## Denominators and checks

| Population | Horizon | Eligible shares | After endpoint | Missing two-sided book | Markout USD |
|---|---:|---:|---:|---:|---:|
| Historical backtest | 0 s | 1,286,584.10 | 0.00 | 704.41 | +11,116.88 |
| Historical backtest | 1 s | 1,285,502.18 | 866.73 | 919.60 | -21,839.31 |
| Historical backtest | 5 s | 1,281,867.98 | 4,697.36 | 723.16 | -21,917.17 |
| Historical backtest | 30 s | 1,226,915.64 | 57,803.73 | 2,569.13 | -14,894.31 |
| Tokyo crossing estimate | 0 s | 58,305.98 | 0.00 | 0.00 | -443.35 |
| Tokyo crossing estimate | 1 s | 58,305.98 | 0.00 | 0.00 | -868.83 |
| Tokyo crossing estimate | 5 s | 58,285.98 | 0.00 | 20.00 | -1,171.57 |
| Tokyo crossing estimate | 30 s | 56,873.82 | 23.00 | 1,409.16 | -491.04 |
| Tokyo crossing plus queue decreases | 0 s | 127,789.53 | 0.00 | 163.00 | +906.20 |
| Tokyo crossing plus queue decreases | 1 s | 127,752.53 | 0.00 | 200.00 | +45.28 |
| Tokyo crossing plus queue decreases | 5 s | 127,012.66 | 0.00 | 939.87 | -304.90 |
| Tokyo crossing plus queue decreases | 30 s | 121,656.94 | 37.00 | 6,258.59 | +358.15 |

Every historical market reproduced all nine archived metrics. An independent calculation using the two pinned October 26 Rust parity fixtures reproduced the eligible quantities and markout sums at every horizon in 2 markets. The Tokyo fill legs reconcile to the committed estimator quantities in all 11 markets.

The midpoint is the latest two-sided top of book received by the target time, using the frame current at that time. A missing or invalid current book excludes the fill for that horizon. Both datasets exclude horizons beyond opening plus 298 s. The historical clock starts at the simulated match. The Tokyo clock starts at the frame used to infer the fill. Inventory feedback and the sources of fills differ between the two runs. No maker fees, rebates or own-market impact are included.

The earlier historical run reported +4.546 cents per FIFO-matched share and +$22,244.34 at settlement. Those figures measure completed turns and settlement economics. Each fixed-horizon markout measures a single fill. The different horizons on the same fills must be read separately.

[Original historical receipt](../urgency_size100_20261126/report.md) · [Tokyo receipt](https://github.com/yuqingqq/pmmmrust/blob/29741ec/docs/receipts/20261126_urgency100_shadow_review/README.md)
