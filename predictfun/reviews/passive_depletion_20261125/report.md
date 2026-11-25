# Passive Binance depth depletion guard

The frozen economic decision is **stop**. All 70 complete October 26 markets reproduced the reference before the candidate comparison. The 288-market registry also includes 124 missing the endpoint and 94 with stale books. October 26 was already consumed. Values are simulator point estimates as of November 25, with zero maker fees and rebates.

| Policy | Settlement | FIFO completed | ¢/matched share | Residual at settlement | Matched shares | Mean absolute ending shares | Both sides active |
|---|---:|---:|---:|---:|---:|---:|---:|
| Straight-line reference | −$2,759.33 | −$2,302.71 | -1.189 | −$456.62 | 193,654 | 41.9 | 61.7% |
| With the depletion guard | −$3,340.47 | −$2,960.33 | -1.586 | −$380.14 | 186,607 | 43.2 | 59.3% |

Delta in settlement −$581.14, completed turnover −$657.62, residual +$76.48. Settlement improved in 31 markets, worsened in 39 and tied in 0. Dropping the largest improvement (market 3790202) leaves −$809.40. Dropping the largest absolute change leaves −$303.52.

## Orders and queue exposure

The candidate issued 8,886 incremental live pulls. These orders had accumulated 12,023.7 order-seconds since activation (mean 1.35 s per pull). It withheld 13,714 proposed slots and clipped 675 proposals to their closing quantity. Withheld slots count decisions, so one vacancy can appear repeatedly. Total cancels from all reasons were 23,617 and 30,676; submissions were 25,746 and 32,860. The signal was unknown on 33,916 of 834,400 side-ticks.

On the reference path, 571,284 purely adding live decisions and 19,368 otherwise eligible adding or mixed slot decisions were examined. On 1,312 live callbacks and 682 slot callbacks the existing pressure rule already acted while depletion was on. The candidate protected mixed resting orders on 6,708 flagged callbacks. Age and remaining rest are durations; neither measures an executable price for queue priority.

## First incremental flag per purely adding reference generation

This attributes the baseline fills after a feasible cancellation or withheld entry. The same order can remain profitable later. Zero-fill generations stay in the denominator. The full-policy table above also includes mixed-size changes, losing queue position, re-entry, subsequent fills and other-side feedback. Dollar amounts below are signed fill contributions.

| Action at first flag | Generations | With a later eligible fill | Shares at/after effect | Eligible +1 s | Eligible +5 s | Eligible +30 s | At settlement |
|---|---:|---:|---:|---:|---:|---:|---:|
| live | 4,622 | 1,249 | 66,082.8 | −$905.61 | −$671.37 | −$473.34 | $3,008.61 |
| slot | 1,341 | 337 | 15,889.4 | −$175.26 | $18.41 | $319.70 | $489.01 |

On the flagged reference generations, 18,107.9 shares had already matched and another 4,483.7 matched in flight. Flagged live orders had 9,304.4 cumulative order-seconds of feasible remaining exposure. After the effect, positive and negative settlement contributions were $20,571.07 and −$17,073.45.

| Horizon | Valued shares | Censored shares | Favorable fill value | Adverse fill value |
|---|---:|---:|---:|---:|
| +1 s | 81,972.2 | 0.0 | $336.51 | −$1,417.38 |
| +5 s | 81,970.2 | 2.0 | $1,403.35 | −$2,056.31 |
| +30 s | 81,795.2 | 177.0 | $4,632.25 | −$4,785.89 |

The matched settlement comparison covered 3,786 of 5,963 first flags. Selected contributions were $2,118.58, against −$996.10 on average in 500 draws (5th percentile −$1,862.48; observed percentile 100.0%). This compares reference exposure, conditional on support. A single day's simulated queue, receipt clock and fixed external trades limit interpretation.

## Reproduction

Candidate and score frozen in f3b9a44e4, with pre-score reservation correction 89cbd0ff8. The first four partial records were excluded. The independent checker was committed before scoring. Its first invocation stopped on an incomplete synthetic branch stub before checking any outputs. Providing the empty order list on that stub let the original assertions run. The replay and scorer were unchanged. Re-run the test and replay using the Codex run_light slot with trial.py, then score, verify.py and report.py. Raw partitions and hourly Binance input hashes are bound by the receipts.

[Protocol](protocol.md) · [Per-market outcomes](markets.csv) · [Frozen readings](results.json) · [Verification and partition hashes](verification.json)
