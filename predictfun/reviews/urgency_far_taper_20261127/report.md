# Time-dependent far-from-open inventory cap

Lowering the far allowance reduced average ending inventory on each of the three consumed dates. The pooled nominal settlement improved $31.45, from $468.54 to $499.99. It saved $96.72 on leftover inventory and gave up $65.26 of completed FIFO profit. The alternative cancellation assumption improved $29.96. Both passed the prespecified comparison with the previous BTC policy.

| Remaining | Previous far | New far | Near |
|---|---:|---:|---:|
| 210 s | 150 | 150 | 150 |
| 150 s | 150 | 137.5 | 125 |
| 90 s | 150 | 125 | 100 |
| 60 s | 150 | 100 | 50 |
| 48 s | 150 | 90 | 30 |
| 30 s | 0 | 0 | 0 |

At every instant, the new far endpoint is the midpoint of the prior far and near curves. The old far endpoint begins declining below 150 at 48 seconds left. Near means within 0.5 estimated remaining standard deviations of the Binance opening mid; far means at least 1.5, in either direction. Interpolate between them. A missing feature retains the prior fallback. On 450,438 of 759,389 decision ticks between 210 and 30 seconds left, the unrounded candidate allowance was lower. Of these, 128,098 were far and 322,340 were intermediate. These are market ticks, not orders. The P&L is a whole-policy effect across both regimes and subsequent fills. Pending reservations, whole-share rounding, the original $1 minimum, passive exits, and the independent 150-share hard limit all continue to apply.

| 211 paired markets | Previous BTC nominal | Lower far nominal | Previous BTC short | Lower far short |
|---|---:|---:|---:|---:|
| Mean absolute ending inventory, shares | 8.03 | 6.14 | 8.30 | 6.58 |
| Largest 22 absolute ending positions, mean shares | 66.08 | 52.27 | 65.96 | 53.34 |
| Average loss among worst 22 market settlements | $48.07 | $47.05 | $46.88 | $44.80 |
| Matched shares | 85,024 | 81,662 | 80,955 | 77,488 |
| Completed P&L | +$839.37 | +$774.11 | +$1,060.69 | +$1,018.81 |
| Residual P&L at payout | −$370.84 | −$274.12 | −$373.42 | −$301.58 |
| Settlement P&L | +$468.54 | +$499.99 | +$687.27 | +$717.23 |
| Positive dates | 2/3 | 2/3 | 3/3 | 3/3 |

The nominal daily settlements moved from −$52.42 to −$60.37 on Oct 24, +$360.21 to +$382.35 on Oct 25, and +$160.74 to +$178.01 on Oct 27. Both policies still reached a 150-share intramarket peak. The pre-existing time-only control earned $357.68 nominal and $673.97 under short cancellation. The immediate comparator in the test was the previous BTC policy.

As of Nov 27, 2026, the three-date census was 864 markets. The paired population included all 211 complete tapes, with 392 missing endpoints and 261 stale books. This is a 50-shares-per-level, two-level offline replay with static public flow, simulated queue priority, zero maker fees and rebates. The short assumption ends further eligibility at a cancel request and leaves nominal notice delays in place. On three previously used dates, the inventory reductions are more informative than the small pooled profit difference for choosing the next research step.

The freeze was 5678e6934. The prior policy reproduced its archived results on the lowest market ID of each date under both timing assumptions. The independent read-back reconciled market/tape membership, both sets of record hashes, shares and settlement, all 24 daily and pooled panels, and both decisions. Research runtime was 7 minutes 3 seconds. Freeze this curve for a comparison on the other consumed benchmark dates once the same BTC inputs are available. Tokyo settings remain unchanged.

[Declaration](request.md) · [Daily data](comparison.csv) · [Decision and hashes](result.json)
