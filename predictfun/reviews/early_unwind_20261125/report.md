# Earlier passive unwind on October 26

Earlier passive unwinding lowered mean absolute settlement holdings from 587.4 to 63.2 shares (89.2%). Median ending holdings were 623.9 and 0.97 shares. The scheduled policy finished within one share of flat in 39 of 70 markets, versus zero for the control. The largest remaining position under the schedule was 333.7 shares. Reduce-only stops new risk while a passive exit still needs a fill.

| Seconds until expiry | Prior mean absolute shares | Scheduled mean absolute shares |
|---:|---:|---:|
| 150 | 448.6 | 448.6 |
| 120 | 532.2 | 487.5 |
| 90 | 490.0 | 360.2 |
| 60 | 516.8 | 282.1 |
| 30 | 427.5 | 182.9 |
| 10 | 478.0 | 87.9 |
| 5 | 554.5 | 71.3 |
| Settlement | 587.4 | 63.2 |

| Simulated dollars across the 70 markets | Prior | Scheduled |
|---|---:|---:|
| Completed FIFO turnover | +78.70 | −2,480.69 |
| Open inventory at settlement | −2,973.04 | −795.40 |
| Total settlement P&L | −2,894.34 | −3,276.08 |

Total P&L fell $381.74. The open-inventory component improved $2,177.64 while completed turnover fell $2,559.38. Winding down changes which fills are paired by FIFO, so those components cannot isolate a cost of individual exit fills. P&L improved in 39 markets and worsened in 31. The scheduled policy filled 405,343 shares versus 487,784 and quoted both sides on 64.2% versus 74.5% of grid time. In a post-run descriptive check, the five largest market losses averaged $316 under the schedule versus $414 under the control, with each arm's markets ranked separately.

Both policies waited for the same opening information and used only passive orders with the same pressure guard, venue clocks and queue model. The control reproduced the prior market metrics individually. One untuned schedule started with 1,000 shares and tightened to 800/600/400/200/0 at 150/120/90/60/30 seconds to true expiry. Client-known inventory and same-side reservations govern risk. Existing orders can fill while a cancel is in transit. Trading ends two seconds before settlement. The schedule was chosen before replay. An initial test import collided with an upstream module and was fixed before any market run.

This is descriptive development evidence from one consumed day, as of 2026-11-25. All 70 complete paths from the 288-market census were included; 124 lacked an endpoint mark and 94 had stale books. Profitability and general tail-risk effects remain unvalidated. [Protocol](protocol.md) · [Per-market results](markets.csv) · [Machine-readable totals](result.json)
