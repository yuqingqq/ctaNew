# Binance best-level alert replay

The candidate failed the frozen development screen. November 28, 2026 UTC. The replay includes 218 eligible five-minute markets of 864 scheduled, on three previously studied dates. The original census excludes 465 missing endpoints and 181 stale books. Excluded markets have unknown P&L. The two arms use the same tape, zero maker fees and rebates, and independent simulated fills and inventory.

The ten-input logistic score was fit on 299,897 random and nearby points on October 23, October 31, November 1 and November 2. The cutoff of 0.246104 was fixed from the earlier out-of-fold random scores. Training excluded the top taker’s order moments. The replay uses the Binance exchange clock plus 4 ms, 37 ms to remove a live order, and the original 50 ms placement grid. There is no cooldown. Directional book alerts use the pre-move population of the original study.

| UTC day | Markets | Current USD | With alert USD | Difference USD | Current / alert market Sharpe |
|---|---:|---:|---:|---:|---:|
| 20261124 | 73 | +34.06 | +210.56 | +176.50 | 0.0131 / 0.0722 |
| 20261125 | 62 | -177.32 | -124.21 | +53.11 | -0.0716 / -0.0588 |
| 20261126 | 83 | +341.87 | +85.71 | -256.16 | 0.1143 / 0.0322 |
| Pooled | 218 | +198.61 | +172.06 | -26.55 | 0.0246 / 0.0223 |

The paired policy improved 113 markets, worsened 105, and tied on 0. The improvement excluding its best day was -203.05 USD. Sharpe is mean settlement divided by sample SD at fixed capital, per market, unannualized. No interval on three days.

| Measure | Current | With alert |
|---|---:|---:|
| Worst-decile mean market P&L, USD | -69.542 | -65.707 |
| Completed FIFO P&L, USD | 359.412 | 358.430 |
| Residual inventory P&L, USD | -160.801 | -186.367 |
| Filled shares | 246,111.374 | 259,386.718 |
| Cancel requests | 57,213.000 | 101,510.000 |
| Reposts | 49,335.000 | 92,079.000 |
| Net one-second markout, cents per valued share | -1.124 | -0.970 |
| Mean absolute inventory at expiry, shares | 6.356 | 7.072 |

| Fill cohort | Shares | Shares lacking +1s mark | Adverse +1s USD | Favorable +1s USD | Settlement USD |
|---|---:|---:|---:|---:|---:|
| baseline | 246,111.4 | 186.8 | -4,221.81 | +1,458.29 | +198.61 |
| candidate | 259,386.7 | 287.8 | -4,208.14 | +1,695.01 | +172.06 |
| lost | 98,148.8 | 114.9 | -1,751.91 | +613.96 | -49.21 |
| added | 111,424.2 | 215.9 | -1,738.25 | +850.67 | -75.76 |
| timely_baseline | 54,280.8 | 97.7 | -898.68 | +285.86 | +668.38 |

Lost and added shares compare entire feedback paths on the same public trade, side and own quote price. The timely baseline cohort asks whether an extra live-order pull could have reached the venue before an actual baseline fill. It is an exposure diagnostic. Keeping later orders and inventory fixed would give a different counterfactual.

The overlay issued 61,129 incremental off-grid pulls and 1,135 incremental grid pulls on the candidate path, and withheld 44,082 otherwise eligible grid proposals. Repeated proposals can refer to one slot. Of the pulled generations, 52,288 had a subsequent accepted order on the same side at the same price.
On those observations, the mean tracked external shares ahead at pull and at the next accepted activation were 711.3 and 664.6. Mean time from pull request to that activation was 13.75 seconds. A depth change between the two observations can change that difference.

Baseline fills with enough cancel lead had a net one-second markout of -1.131 cents per valued share. Remaining baseline fills averaged -1.122. The timely cohort contained 21.3% of baseline gross adverse dollars and 19.6% of gross favorable dollars.

The original grid and event controls reproduced their archived market panels. Earlier raw feature vectors matched the source vectors in both directions, and truncating future updates preserved earlier features. Removing one dollar from a lost-fill subtotal failed the reconciliation. Live execution, market impact and unobserved markets remain outside this replay.

## Daily source of the difference

| UTC date | Change in completed FIFO USD | Change in residual USD | Current / alert +1s cents per share |
|---|---:|---:|---:|
| 20261124 | +213.38 | -36.87 | -0.987 / -0.760 |
| 20261125 | +76.22 | -23.11 | -0.947 / -0.868 |
| 20261126 | -290.58 | +34.42 | -1.392 / -1.243 |

[Protocol](protocol.md) · [Paired markets](paired_markets.csv) · [Computed metrics](results.json)
