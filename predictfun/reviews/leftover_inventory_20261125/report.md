# One-day passive inventory test

The projected-leftover controller failed the declared criterion on consumed Oct 26, 2026. The criterion required positive settlement P&L and an improvement over the fixed inventory controller with the same pressure guard. Market-level data are in [markets_v2.csv](markets_v2.csv). All dollar totals sum separate simulated markets.

| Policy | Completed round trips | Inventory at settlement | Settlement P&L | END mid P&L | Filled shares | Both sides live | Markets positive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed inventory | -8,412.52 | -4,164.80 | -12,577.32 | -11,365.67 | 570,255 | 83.7% | 12 / 70 |
| Fixed inventory + pressure | -1,761.74 | -3,265.88 | -5,027.62 | -4,150.55 | 449,942 | 79.3% | 18 / 70 |
| Projected leftover + pressure | +370.00 | -3,064.42 | -2,694.41 | -1,587.85 | 487,153 | 75.3% | 29 / 70 |

All three policies waited for the same opening trigger, with a median first submission 6.4 s after opening. Each used two passive levels per side, up to 220 shares per level, and a 1,000-share hard limit. No active takes. The passive simulator used 58 ms activation, 37 ms live cancel, 100 ms fill notices, current queue at activation and no queue credit for external cancellations. Settlement uses the recorded outcome. Completed turnover is FIFO. Book marking is at the replay endpoint, opening +298 seconds. Both-sided participation is the mean per-market grid fraction.

## Where the difference came from

- Pressure contributed +7,549.69 USD, improving 61 markets and worsening 9. The fixed-rule inventory settings were identical for this comparison.
- Changing the inventory policy under the same pressure rule contributed +2,333.21 USD, improving 38 markets and worsening 32. The completed-turnover contribution was +2,131.74 USD and the settlement residual contribution was +201.47 USD. C changes both new-quote sizing and the old soft-limit inventory pulls.
- Median peak exposure rose from 497 to 982 shares. Mean absolute ending inventory rose from 385 to 587. The largest C inventory was 999.999 shares. C submitted 30,912 orders, issued 27,942 cancellation requests, of which 6,487 were pressure pulls.

## Can we count on the exits?

At 22,604 of 30,912 C submissions (73.1%), subsequent gross opposite-side passive fills met or exceeded the estimate. For strictly positive forecasts this was 22,505 / 30,813. The intended lower-quartile coverage was about 75%. Each submission is a repeated decision on a common market path. Using actual subsequent exit volume in the same all-reservations-fill calculation put 1,505 submissions (4.9%) above $50. That stress calculation assumes every reserved share fills. It is separate from realized inventory P&L.

At the regularly spaced 5 s anchors, C's volume estimate was covered 73.3% of the time. Coverage was 79.2% near the BTC opening price (|z| < 0.5, n=3,704) and 58.2% farther away (|z| ≥ 1.5, n=1,284). The latter descriptive split also mixes different remaining times. Normalized BTC drift did not set order sizes in this test.

A post-run inventory diagnostic distinguishes selling the original position from ending flat. There were 3,083 nonflat C clock states where projected exit volume covered the entire current holding. In 2,935 of those, the market actually supplied that many opposite fills. In 1,559 of those latter cases, the strategy still finished with inventory on the same side after subsequent trading. These repeated anchors show how gross turnover and ending exposure can diverge. They do not identify the value of a different quoting policy.

## Charted market 3789484

| Policy | Settlement P&L | Completed turnover | Ending YES-equivalent shares |
|---|---:|---:|---:|
| A | +160.07 | +67.72 | -130.0 |
| B | +328.94 | +292.65 | -51.1 |
| C | +428.58 | +283.76 | -226.9 |

The first order was submitted 3.8 s after opening in all three arms. An independent notice-by-notice reconstruction reproduced each arm's summary and proposals, and exactly reproduced C's fill list and order lifecycle on this market.

## Scope and reproducibility

As of the Nov 25 run, the consumed Oct 26 registry contained 288 markets. The paired comparison includes all 70 with complete books and endpoint marks. The other 124 lacked the required endpoint mark and 94 had a stale future book. The completeness filter uses future tape availability. These are point estimates for the replayable subset. The simulator assumes unchanged external trades and continuous access to complementary token buys, a $1 minimum buy and no maker fees or rebates. There is no untouched-day or capital-adjusted return estimate.

The fit was pinned from 46 markets on Oct 24–25. An explanatory timestamp join and a score-module import were repaired. Independent enumeration then exposed a floating-point discrepancy at exactly 220 shares of recent flow. The corrected fit and complete replay supersede the first pass. Settlement changed by more than one millionth of a dollar in 0 A, 0 B and 46 C markets. The primary criteria and economic parameters were fixed before either run. [Protocol](protocol.md) · [Superseding fit](fit_v2.json) · [Results and checksums](results_v2.json) · [Trace control](trace_check.json) · [Gross versus net diagnostic](turnover_check.json)
