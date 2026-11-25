# Smooth unwind and completed turnover

Matched replay of 70 complete October 26 markets. The reference staircase was reproduced in every market. All arms use the same opening information, passive execution, queue, pressure signals, fees, $50 admission rule, 1,000-share hard cap and final 30-second reduce-only period. The live-order buffer lets an existing order keep its queue when exposure exceeds the time or state limit by at most 22 shares before reduce-only. New proposals always obey the stated limit.

| Policy | FIFO-matched shares | Fills closing any shares | Completed P&L | ¢ per matched share | Residual P&L | Settlement P&L |
|---|---:|---:|---:|---:|---:|---:|
| Original staircase | 200,460 | 15,134 | $-2,480.69 | -1.237 | $-795.40 | $-3,276.08 |
| Staircase, buffered | 200,446 | 15,110 | $-2,506.00 | -1.250 | $-793.93 | $-3,299.93 |
| Straight line, buffered | 193,654 | 14,556 | $-2,302.71 | -1.189 | $-456.62 | $-2,759.33 |
| Market-conditioned curve, buffered | 193,611 | 14,548 | $-2,525.33 | -1.304 | $-394.35 | $-2,919.68 |

| Mean absolute inventory (shares) | 60 s left | 30 s left | 10 s left | Settlement | Markets within 10 shares of flat | Total cancellation requests |
|---|---:|---:|---:|---:|---:|---:|
| Original staircase | 282.1 | 182.9 | 87.9 | 63.2 | 41/70 | 24,627 |
| Staircase, buffered | 284.7 | 182.2 | 87.9 | 63.2 | 41/70 | 24,635 |
| Straight line, buffered | 275.0 | 118.1 | 60.7 | 41.9 | 51/70 | 23,617 |
| Market-conditioned curve, buffered | 270.5 | 118.5 | 60.9 | 40.9 | 46/70 | 24,152 |

## Paired contrasts

The declared gate requires more matched shares, higher completed P&L, higher settlement P&L, and mean absolute ending holdings no more than 10 shares above the comparator. Completed turnover uses FIFO. A fill that both closes inventory and opens a position in the opposite direction contributes only the closing portion to matched shares. Changes in FIFO pairing can move value between completed and residual P&L.

- Staircase, buffered versus Original staircase: -14 matched shares, $-25.31 completed P&L, $-23.84 settlement, -0.0 mean absolute ending shares. Markets better/worse/equal on settlement: 5/4/61. Joint gate: fail.

- Straight line, buffered versus Staircase, buffered: -6,792 matched shares, $+203.29 completed P&L, $+540.60 settlement, -21.3 mean absolute ending shares. Markets better/worse/equal on settlement: 42/27/1. Joint gate: fail.

- Market-conditioned curve, buffered versus Straight line, buffered: -43 matched shares, $-222.62 completed P&L, $-160.35 settlement, -1.0 mean absolute ending shares. Markets better/worse/equal on settlement: 30/39/1. Joint gate: fail.

Relative to the original unbuffered staircase, Straight line, buffered changed closed quantity by -6,806 shares, close events by -578, settlement by $+516.75, and absolute ending inventory by -21.3 shares per market.

Relative to the original unbuffered staircase, Market-conditioned curve, buffered changed closed quantity by -6,849 shares, close events by -586, settlement by $+356.40, and absolute ending inventory by -22.3 shares per market.

## Market inputs and limitations

The exit-rate fit used 46 consumed Oct 24–25 markets. At each predetermined anchor it offered a fixed 440-share position for 15 seconds from a fresh queue on the reducing side only. Sell and buy rates were fitted separately. Of 552 sell probes, 58 closed zero shares; of 552 buy probes, 44 closed zero. The state inputs were causal BTC drift relative to remaining volatility, recent delivered reducing fills, and current pressure on the exit side. On the evaluation ramp the mean absolute deviation of the state multiplier from 1 was 19.4% over queried side–ticks. Only 5 sell probes and 2 buy probes began with a pressure veto on the exit side. Specific conclusions about behavior during that veto have thin support. Throughput adjusts the cap relative to the same side and time. It does not forecast per-fill profitability.

The fit was committed before Oct 26 scoring. An initial synthetic test caught the reference probes using a branch expecting a resting-review hook. The corrected probes and original B roll-in use their original branch; the synthetic checks passed before the fit. A separate inherited policy-selector collision stopped the first evaluation before any market was written. The superseding evaluator verifies every training definition against the committed fit source. Each matched evaluation used that same fit.

These are simulated outcomes on one previously used development day, as of Nov 25. The Oct 26 registry had 288 markets, including 124 missing the required endpoint and 94 with stale books. This is a conditional complete-path comparison. No pricing advantage, distribution of future days or live profitability follows from a passing implementation or a single-day gate.

[Protocol](protocol.md) · [Market-level results](markets.csv) · [Checked totals](result.json) · [Earlier-day fit](fit.json)
