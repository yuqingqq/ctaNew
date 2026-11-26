# Which simulated closes did the wallet miss?

As of November 26, 2026, on the November 8 historical v3 fixed-order replay. All 119.49 primary closing shares on completely unfilled wallet orders reconcile to 20 simulated fills, 15 distinct order IDs and 13 markets. The trace includes every matching order in each involved market and reconciles the closing shares by ID to the hash-verified parent artifact. Model time uses the public trade-group anchor minus 50 ms.

| Where the modeled trade falls | Closing shares on wallet-unfilled IDs | Closing shares on wallet-filled IDs |
|---|---:|---:|
| After acceptance notice and before cancel request, at nominal alignment | 42.46 | 75.45 |
| At or after cancel request, at nominal alignment | 77.03 | 0 |
| Inside the acknowledged-to-cancel-request interval at every allowed run-clock offset | 0 | 25.00 |

The last row is a subset check. A cancel request does not date removal. The exposure through the cancel boundary is uncertain because the venue may still match the order and the bot and public tape do not have a proven common clock. The wallet-filled comparator covers 8 IDs in 6 markets. Positive wallet fills on those orders do not establish that all 75.45 simulated shares closed on the wallet.

Of the 119.49, 72.49 were credited to trades through the quote, for which this engine bypasses the queue ahead. The other 47.00 were at the order price, including 12.00 at an initial modeled queue of zero. All 47 had an observable activation-level depth under the model. Only 10.46 of the 119.49 were credited on trades with an imputed timestamp. Timing, price, and imputation are overlapping breakdowns.

Using acceptance through cancel request as the modeled lifetime leaves 37.46 of the original 119.49 closing shares on the same IDs. That number compares complete simulated inventory histories, allowing earlier fill changes to alter whether a subsequent fill closes inventory. The lower and upper run-clock endpoints retain 0 and 15.00 respectively. These are sensitivities, not identified venue times. Shortening the lifetime for every order by fiat can suppress genuine fills. The public tape also cannot identify the exact external queue each order faced.

Keep the 50-share urgency strategy in shadow. The next test should align the same public trade IDs, Tokyo receipt times and venue acknowledgment/cancel evidence independently of filling outcomes. Build order-level plausible lifetimes, retain uncertain credits as ranges, and score both false closing shares and missed actual closing shares on all acknowledged orders. The test must preserve genuinely filled exits as well as reduce false ones. Before funding, the strategy also needs positive settlement under the intended 150-share cap across later independent markets with conservative execution. This one-day trace diagnoses execution and does not establish the new strategy's profitability.

[Event-level evidence](events.csv) · [Quantities and sensitivity by cohort](summary.json) · [Earlier execution and cap comparison](../tokyo_execution_comparison_20261126/report.md)
