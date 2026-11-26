# Early limit on new inventory-adding quotes

**Decision: keep the urgency controller.** Applying the upcoming 250-share allowance to new quotes with 90 to 60 seconds remaining missed the declared two-day goal. It improved settlement on October 26, alongside *higher* mean ending inventory. October 27 passed the daily rule by $1.55 and 0.23 shares per market.

| Metric | Oct 26 urgency | Oct 26 early limit | Oct 27 urgency | Oct 27 early limit |
|---|---:|---:|---:|---:|
| Settlement P&L | $1,774.49 | $2,103.02 | $903.99 | $905.54 |
| Completed FIFO | $2,259.82 | $2,694.92 | $1,769.09 | $1,766.19 |
| Unmatched inventory P&L | −$485.32 | −$591.89 | −$865.10 | −$860.65 |
| Matched shares | 104,752 | 99,450 | 82,332 | 77,791 |
| Cents per matched share | 2.157 | 2.710 | 2.149 | 2.270 |
| New submissions with 90 to 60 seconds left | 1,871 | 1,264 | 1,361 | 896 |
| Markets above 250 shares at 60 seconds | 35 | 24 | 33 | 24 |
| Mean absolute shares at 60 seconds | 255.4 | 258.9 | 282.6 | 287.2 |
| Mean absolute ending shares | 38.25 | 41.94 | 54.96 | 54.73 |
| 90th percentile absolute ending shares | 188.35 | 188.96 | 227.37 | 227.37 |

The comparison covers all eligible markets in the frozen census as of November 26. October 26 had 70 of 288 complete (124 missing endpoint, 94 stale book); October 27 had 58 of 288 (139 missing endpoint, 91 stale book). These are two previously examined dates, shared external flow, modeled queue and latency, and passive fills. Ending inventory is measured at the last supported tick two seconds before expiry and marked at the recorded settlement payout.

October 26's $328.53 gain consisted of $435.10 more completed turnover and $106.57 more residual loss. The gain was $207.05 after removing its largest market improvement. Of the 70 markets, 23 improved, 28 worsened and 19 tied. On October 27, 21 markets improved, 19 worsened and 18 tied; excluding the single largest gain changes the daily difference to −$111.80. Matched shares fell on each date even as earned cents per matched share rose.

Fewer markets exceeded 250 at 60 seconds but the average absolute position at that time rose on both days. The rule limits new risk and gives resting orders a chance to fill. It cannot cut inventory already held or guarantee a passive exit. A post-run paired count from the market rows found that 12 and 9 markets crossed downward through the threshold on the two dates, with 1 and 0 crossing upward; none of the downward crossings landed exactly at 250 (tolerance 1e-8 shares). This count does not identify why a specific ending position persisted. Before changing another threshold, attribute the largest remaining positions to inventory already held at 90 seconds, subsequent fills from previously resting or newly submitted orders, and available reducing-side fills.

Controls reproduced on 128/128 market paths. The ledger reconstruction checks every submitted candidate quote in the window using delivered fill quantities and each unclosed generation, including pending cancellations. Native order-entry tests cover both sides, the closing quantity and the 210 held + 30 reserved case. An independent CSV check reproduced the daily gates and refused damaged settlement. Initial freeze 60a496f32. The first run stopped before publishing any output when an input assertion rejected the standard 50¢ tie payout. Amendment b3f3b886b corrected only that assertion in the runner before the successful replay.

[Protocol](protocol.md) · [Input-check amendment](amendment.md) · [Daily criteria](result.json) · [Market pairs](markets.csv)
