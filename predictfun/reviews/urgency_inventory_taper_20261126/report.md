# Earlier passive inventory reduction

**Keep the current 150-share controller.** The earlier taper reduced mean absolute ending inventory by 69% under nominal execution. It lost money on all eight dates under both execution assumptions. On the nominal path, it saved $1,510 on leftover inventory and gave up $8,474 on completed round trips. Those are differences between two complete strategy paths.

| Across 619 paired markets | Current nominal | Earlier taper nominal | Current short | Earlier taper short |
|---|---:|---:|---:|---:|
| Mean absolute ending shares | 13.35 | 4.08 | 13.86 | 4.39 |
| Mean of the 62 largest absolute ending positions, shares | 98.11 | 32.67 | 99.46 | 34.83 |
| Markets ending at the 150-share cap | 3 | 0 | 2 | 0 |
| Average peak absolute shares within a market | 136.06 | 130.25 | 135.03 | 129.38 |
| Maximum absolute shares during the replay | 150 | 150 | 150 | 150 |
| Completed shares, counted once per round trip | 270,807 | 300,719 | 252,407 | 278,379 |
| Completed FIFO P&L | +$3,445 | −$5,029 | +$4,688 | −$2,950 |
| Residual inventory P&L at payout | −$1,934 | −$424 | −$2,093 | −$438 |
| Settlement P&L | +$1,511 | −$5,454 | +$2,595 | −$3,387 |
| Profitable dates out of eight | 6 | 0 | 5 | 0 |
| Mean loss of the 62 worst markets | $59.55 | $59.56 | $57.98 | $55.70 |

The nominal policy completed 11% more shares under the taper. Completed value moved from +1.27¢ to −1.67¢ per matched share. The accounting locates the deterioration in completed trading. The intervention changed admission sizing, resting-order review and when distant exits could be repriced together, so it does not isolate the cost of any individual action. October 31 had the largest current average ending inventory. The taper lowered it from 33.35 to 12.02 shares across 102 markets, while completed P&L moved from +$1,005 to −$764 and residual P&L from −$966 to −$286.

The current research curve first reaches the 150-share Tokyo hard cap at 48 seconds remaining. The candidate preserves the first half of the market, then scales the existing curve to 150 shares, reaching a target of 75 with 90 seconds left and zero with 30 seconds left. The inherited 22-share resting-order review tolerance still applies to positive targets. Outstanding reservations count against the separate whole-order hard cap throughout. The target governs passive orders and cannot compel an execution.

As of November 26, 2026 the eight consumed UTC dates were October 23–25, 27 and 31, and November 1–3. Of 2,304 markets in the original census, 619 had complete tapes, 938 lacked endpoints, 746 had stale books and one was boundary-censored. Every complete tape was paired. Research assumptions include fixed public flow, simulated L queues, zero maker fees and rebates, and passive execution. The short assumption ends fill eligibility at cancel request and preserves notice delays. These are development estimates. The November 14–18 block remains reserved.

The freeze was 74b795b0d. The original arm reproduced its archived figures on the lowest market ID of each date under both execution assumptions. All new ledgers reconciled matched and unmatched shares, FIFO P&L and payout, and respected the hard cap including same-side reservations. Independent read-back verified source hashes, identities, and all metrics in 36 daily and pooled panels. Replay took 21 minutes 15 seconds on one core.

**Next research priority:** isolate a late veto on new orders that add exposure, while retaining the original exit pricing and queue retention. That comparison can tell whether limiting fresh accumulation preserves profitable closes. The current result gives no basis to change Tokyo settings.

[Daily and pooled comparisons](comparison.csv) · [Prespecified rule](request.md) · [Receipt and hashes](result.json) · [Independent read-back](verify.py)
