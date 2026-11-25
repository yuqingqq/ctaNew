# Spread earned on the retained passive strategy

The branch is the buffered straight-line unwind with the existing pressure guard. This diagnostic replays the original fills on the 70 complete markets of consumed October 26. All markets reproduce the saved fill quantity, matched quantity, completed and residual P&L, settlement, ending position, posts and cancellations. The full registry contains 288 markets, including 124 missing the endpoint and 94 with stale books.

| FIFO attribution | ¢ per matched share with the relevant marks | USD |
|---|---:|---:|
| At opening execution versus the preceding mid | +0.531 | +1,029.15 |
| Mid movement while holding | -2.180 | -4,220.76 |
| At closing execution versus the preceding mid | +0.459 | +889.13 |
| Realized completed turnover, all matched shares | -1.189 | -2,302.71 |

Of 193,653.9 matched shares, 193,636.0 had valid marks at both legs. Residual realized turnover excluded from the three-component sum by missing marks was $-0.24 over 17.9 shares. The two fill edges sum to +0.991¢ on those marked shares, and net turnover on that cohort was -1.189¢. At market level, the two fill edges summed positive in 70/70 and realized completed turnover was negative in 42/70.

With an intermediate mark on 193,636.0 shares, the first second after opening (or the entire holding period if shorter) contributed -1.797¢ per share. The time from that mark to the close contributed -0.383¢. On all shares, 46.2% completed at a gain, 3.7% flat and 50.1% at a loss.

| Opening public trade relative to our limit | Matched shares | Net ¢ per matched share | Net USD |
|---|---:|---:|---:|
| same price | 70,045.2 | +0.275 | +192.97 |
| through price | 123,608.7 | -2.019 | -2,495.68 |

With all midpoints sampled 50 ms earlier, the opening and closing execution edges combined are +1.292¢, holding movement is -2.469¢, and both marks cover 193,563.4 shares. Fill prices and all-share completed turnover are identical.

## Interpretation and limits

On a long cycle the signed components are opening midpoint minus buy price, closing midpoint minus opening midpoint, and sell price minus closing midpoint. Signs reverse for a short. FIFO shares are counted once per completed cycle, including partial fills and trades crossing flat. Each mark uses the last venue frame strictly before that time, maximum age one second, and the modeled trade clock. Execution edge is a measurement against that snapshot. A displayed midpoint is not a guaranteed price for unwinding. Only completed turnover is in this decomposition; terminal inventory is accounted for separately.

The opening trade type is observed on execution. Grouping by it describes where money went on this replay and cannot identify a rule that knows whether a sweep is coming. The signed holding movement combines selection on opening and closing, exposure duration and subsequent market movement. This calculation alone does not identify which action would have improved P&L.

[Per-market decomposition](markets.csv) · [Totals](result.json) · [Source and synthetic controls](cycles.py)
