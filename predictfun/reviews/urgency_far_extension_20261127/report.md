# Five additional dates for the frozen far-from-open inventory rule

The inventory reduction repeated, but the profit gain reversed. On 408 additional markets, nominal ending inventory averaged 12.20 shares versus 13.07 for the previous BTC rule. Lowering the far cap recovered $104.47 of leftover inventory losses and gave up $260.68 on completed round trips. The five-date incremental decision was risk_tradeoff under both cancellation assumptions. Retain the previous BTC curve as the research comparator for a profit objective.

| Five-date extension | Original controller | Previous BTC rule | Lower far limit |
|---|---:|---:|---:|
| Nominal completed FIFO P&L | +$2,894.12 | +$2,690.57 | +$2,429.89 |
| Residual inventory at payout | −$1,515.41 | −$1,371.63 | −$1,267.15 |
| Nominal settlement | +$1,378.71 | +$1,318.94 | +$1,162.74 |
| Days with positive nominal settlement | 4/5 | 3/5 | 3/5 |
| Mean absolute ending shares | 15.52 | 13.07 | 12.20 |
| Mean of largest 41 ending positions | 110.42 | 100.89 | 94.59 |
| Mean settlement loss of worst 41 markets | $60.95 | $56.78 | $57.23 |
| Settlement with short cancel eligibility | +$2,113.46 | +$2,168.18 | +$1,949.93 |

| UTC date | Complete markets | Previous BTC, nominal | Lower far, nominal | Increment |
|---|---:|---:|---:|---:|
| Oct 23 | 70 | +$358.92 | +$235.38 | −$123.54 |
| Oct 31 | 102 | −$69.51 | −$102.53 | −$33.02 |
| Nov 1 | 88 | +$629.22 | +$661.51 | +$32.29 |
| Nov 2 | 72 | −$292.79 | −$206.31 | +$86.48 |
| Nov 3 | 76 | +$693.11 | +$574.69 | −$118.42 |

Nominal mean absolute ending inventory fell on four of five extension dates. On Nov 2 it rose from 5.71 to 5.96 shares while settlement improved. Oct 23 and Nov 3 account for the largest lost turnover. Directional attribution to particular fills cannot be obtained from these whole-policy comparisons.

Across all eight consumed dates (619 markets), nominal settlement was +$1,511.36 for the original, +$1,787.48 for previous BTC, and +$1,662.73 for the lower far limit. Average absolute ending inventory was 13.35, 11.35, and 10.13 shares respectively. The tighter rule gained $31.45 in the three-date pilot, then lost $156.20 relative to previous BTC in the five-date extension. Under short eligibility, its five-date deficit was $218.25 and its eight-date deficit $188.30. The original controller also out-earned both BTC rules on the additional dates under nominal execution, so promotion of a BTC policy needs independent evidence.

**Population and verification.** As of Nov 27, 2026, 408 of the 1,440 registered markets on the five dates had complete tapes; 546 lacked endpoints, 485 had stale books and one was boundary-censored. All complete markets were paired, including zero fills. The 864,000-row Oct 24 raw BTC grid was reconstructed exactly across every stored field against the pilot. The five grids used the same exchange-event-time calculation, stale/collector gap rules, one-second model lag and frozen volatility feature. The previous hour was used when part of the allowed dates; Oct 22 and Oct 30 were outside the input set. The fallback was used on 15,039 of 1,468,392 active market ticks (1.02%). Hashes of each raw file and grid are in the grid receipts. The original strategy reproduced its archived control on the smallest market ID on each date under each timing. Independent read-back checked 66 daily and pooled panels, tape identities, input receipts, cap and settlement accounting. No November 14–18 tape was used.

The research setup uses up to 50 shares per level, two levels, fixed public flow, modeled queues and zero maker fees and rebates. Short eligibility suppresses fills after a cancel request but retains nominal notice delays. The extra dates had been used for other strategy research. These results measure simulated profitability on those dates. No configuration was changed or fit. Freeze 267f6beb7.

[Daily and pooled figures](comparison.csv) · [Decision and hashes](result.json) · [Frozen extension protocol](request.md) · [Three-date pilot](../urgency_far_taper_20261127/report.md)
