# What changes at 50 shares per level?

Paired historical replay as of November 26, 2026. All three sizes use the same 689 complete markets. Every order follows its own queue, fill and cancellation history.

## All nine dates

| Max shares per level | Orders posted | Orders with fills | Fill events | Orders closing inventory | FIFO order pairs | Matched shares |
|---|---|---|---|---|---|---|
| 220 | 106,694 | 28,352 | 127,579 | 14,868 | 27,074 | 772,619 |
| 100 | 104,118 | 28,510 | 113,632 | 14,535 | 27,034 | 627,323 |
| 50 | 102,415 | 29,682 | 92,383 | 14,974 | 27,818 | 445,764 |

| Max shares per level | Gross completed gains | Gross completed losses | Completed FIFO | ¢ per matched share | Residual at payout | Settlement | Mean abs. end shares |
|---|---|---|---|---|---|---|---|
| 220 | +$73,027.99 | −$47,956.13 | +$25,071.86 | 3.245 | −$6,811.46 | +$18,260.40 | 50.95 |
| 100 | +$64,879.25 | −$36,363.57 | +$28,515.68 | 4.546 | −$6,271.35 | +$22,244.34 | 47.38 |
| 50 | +$50,608.79 | −$24,797.54 | +$25,811.25 | 5.790 | −$5,746.94 | +$20,064.31 | 46.87 |

## October 31 through November 3

| Max shares per level | Orders posted | Orders with fills | Fill events | Orders closing inventory | FIFO order pairs | Matched shares |
|---|---|---|---|---|---|---|
| 220 | 46,441 | 11,592 | 53,041 | 6,063 | 10,891 | 319,711 |
| 100 | 44,801 | 11,667 | 46,885 | 5,914 | 10,868 | 258,397 |
| 50 | 44,211 | 12,310 | 38,311 | 6,135 | 11,274 | 185,802 |

| Max shares per level | Gross completed gains | Gross completed losses | Completed FIFO | ¢ per matched share | Residual at payout | Settlement | Mean abs. end shares |
|---|---|---|---|---|---|---|---|
| 220 | +$33,653.82 | −$20,823.77 | +$12,830.05 | 4.013 | −$4,008.08 | +$8,821.97 | 66.05 |
| 100 | +$29,793.59 | −$15,191.95 | +$14,601.63 | 5.651 | −$3,709.72 | +$10,891.92 | 60.94 |
| 50 | +$22,948.35 | −$10,495.44 | +$12,452.91 | 6.702 | −$3,210.02 | +$9,242.89 | 56.98 |

## Settlement by date

| UTC date | Markets | 220 | 100 | 50 | 50 less 100 |
|---|---|---|---|---|---|
| 20261023 | 70 | +$1,642.56 | +$2,173.43 | +$2,180.99 | +$7.56 |
| 20261024 | 81 | +$1,592.92 | +$1,476.88 | +$1,436.76 | −$40.12 |
| 20261025 | 72 | +$3,524.46 | +$3,355.07 | +$3,080.12 | −$274.95 |
| 20261026 | 70 | +$1,774.49 | +$2,636.08 | +$2,489.09 | −$146.99 |
| 20261027 | 58 | +$903.99 | +$1,710.96 | +$1,634.46 | −$76.50 |
| 20261031 | 102 | +$2,308.88 | +$2,278.78 | +$1,577.52 | −$701.26 |
| 20261101 | 88 | +$2,664.97 | +$3,396.71 | +$2,467.22 | −$929.49 |
| 20261102 | 72 | +$1,640.53 | +$2,146.66 | +$2,336.52 | +$189.86 |
| 20261103 | 76 | +$2,207.59 | +$3,069.77 | +$2,861.63 | −$208.14 |

## Interpretation

Moving from 220 to 100 changed gross gains by −$8,148.74, gross losses by +$11,592.56 and residual value by +$540.11. Settlement changed by +$3,983.94. The 100-share arm exceeded 220 on 6/9 dates and exceeded zero on 9/9.

Moving from 100 to 50 changed gross gains by −$14,270.46, gross losses by +$11,566.03 and residual value by +$524.40. Settlement changed by −$2,180.03. The 50-share arm exceeded 100 on 2/9 dates and exceeded zero on 9/9.

Each order is counted once per market when it receives its first fill. A single sell can close several earlier buys. A FIFO order pair counts a distinct opening/closing order-ID combination, and matched shares count the completed quantity once. Multiple executions against the same order IDs can add volume without adding a pair. Gross gains and losses apply to matched quantities at the two realized prices. Remaining inventory is valued at the binary payout.

Of 2,592 scheduled markets, 1,062 lacked an endpoint, 840 had stale future books and one crossed the end of the available feed. October 23 precedes the capacity fit, October 24–25 supplied the fit, and October 26–27 were used in controller development. The four later dates were already examined. Results assume the same public order flow for each arm, zero maker fees and no rebates. No evidence here identifies live queue priority or an optimal size.

The 100- and 220-share paths reproduced the nine previously saved market fields on every market. Fill and FIFO accounting reconciled for each size, including a synthetic partial-fill/reversal and corrupted-count and settlement refusals. The runner and protocol were frozen in 8831ad669.
