# MOVE-CASES: what the top makers do, and what it costs them, in small moves, big trends and reversions (results, 2026-11-13)

Session ce68ebe0.

USER:
- "You mean they don't pull the trending side, when price pumps they still have their ask there, don't they have adverse selection in this
  case? ... if price pumps a lot, they will suffer from loss right? Can you give me the logic there, the data you show are clearly
  incorrect".
- "You can check different cases, if price pumps a lot, let me know their logic chains, or a series of actions ... When price reverts,
  there will also be a different series of actions".

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| Script `research/pf_move_cases.py` + `run_move_cases.sh` | 0dd3eb0c, before the first run | 16:09-16:12Z (one slot retry) |
| Post hoc per-side table (descriptive) | a8cb3141 | 16:14Z |

## Why this was needed: what was wrong with the earlier statement

ROBUST-CHECK's K1 ("the top 5 hold 51% of the volume at the new price in crossing trades") measured only the price where the move
STOPPED. That is the new best, by construction a level that held. Prices run over in a bigger move were neither measured nor valued.

So "their orders are already at the next price, at the front" was true but incomplete. It read as if being there avoids adverse selection.
It does not, as below. The USER's objection was right.

## Design

- **Events:** the protection-side events after a guard onset (PULL-DEPTH's), with calm fallbacks as baseline; 9 consumed days, first 60 s.
  - **INTO side:** the side the price moves into (asks in a pump).
  - **AWAY side:** the other side (bids in a pump).
- **Move path** of the INTO side's best:
  - M = the largest move within 2 s of the new best's birth;
  - F = what is kept 3 s after the peak.
- **Cases** and their share of the events after a signal:

| Case | Definition | Share |
|---|---|---|
| 1 tick, holds | M = 1, F >= 1 | 15.9% |
| 1 tick, reverts | M = 1, F <= 0 | 19.1% |
| 2 ticks | M = 2 | 20.9% |
| big, trends | M >= 3, F >= M - 1 | **25.8%** |
| big, partial | M >= 3, otherwise | 4.2% |
| big, reverts | M >= 3, F <= M / 2 | **14.1%** |

  Pumps and dumps are the same to within 0.5 points.
- **Fills:** real fills (WS-timed legs; maker and taker identities from ORDERS), valued 5 s after each fill.
  - INTO side: at the old best, at the prices run over, at the peak price, and after the peak.
  - AWAY side: quotes that followed the move, old quotes, and after the peak.
- **Takers:** by group, with the move and against it.
- **Ours:** the simulated guard ladder beside.
- **Controls passed:**
  - Events equal PULL-DEPTH's.
  - Our ladder equals FUTURE-BEST's row for row.
  - Synthetic cases with three planted errors caught.

## The series of actions, by case (top 5, real fills; $ / day at 5 s across these events)

**Big pump that keeps going (26% of signal events): they ARE run over on the ask side, and their bids pay for it.**

| Step | What the data shows |
|---|---|
| 1. Buyers lift the old best ask | Top 5 are 31.5% of the volume filled there (about their usual 34%); -5.9 c/sh |
| 2. The pump runs through the next prices | Top 5 are **42.9%** of the volume run over there, more than their usual share (R540, 9/9 days): they do not pull those asks in time; **-3.6 c/sh** |
| 3. It stops | at a price where they are thickest (58.8% of the volume there); about flat (-0.2 c/sh) |
| 4. Meanwhile on the bid side | sellers keep hitting bids during the pump. Top 5 take 44% of the fills at the old bids (**+7.4 c/sh**) and 29% at bids that followed the move (+4.7) |
| 5. As takers | they buy into big pumps a little more than usual: 4.9% of the buying vs 3.5% in 1-tick moves (R542, 9/9); +4.8 c/sh; small in volume |
| Result | ask side **-$5,924 / day**, bid side **+$6,168 / day**, net **+$245** (about break-even) |

- They lose on the ask side in big trends on 9 of 9 days (R541). They lose less per share than the other makers (-2.11 vs -2.85 c/sh over
  the ask side).

**Big pump that reverts (14%): the same resting asks now win.**

| Step | What the data shows |
|---|---|
| 1-3. The ask side is run over during the pump | now it wins: +0.9 c/sh at the old best, +3.3 at the prices run over, +5.4 at the peak |
| 4. Bid side | bids that followed the move up lose (-1.8 c/sh); bids hit on the way back lose (-0.9) |
| 5. As takers | they sell into the reversion a little more (4.2% of the selling); +3.3 c/sh |
| Result | ask side **+$4,394 / day**, bid side **-$1,102**, net **+$3,292** (their best case) |

**Small moves (56% of events): a little loss on one side, more gain on the other.**

| Case | Ask side | Bid side | Net |
|---|---|---|---|
| 1 tick, holds | -$486 (-0.54 c/sh) | +$1,451 (+2.03) | +$965 |
| 1 tick, reverts | +$2,443 (+2.53) | -$1,632 (-0.98) | +$811 |
| 2 ticks | +$1,138 (+0.66) | +$873 (+0.55) | +$2,011 |
| big, partial | +$26 | +$616 | +$642 |

**Over all cases:** ask side (the side the price moves into) +$1,590 / day (+0.20 c/sh); bid side +$6,375 (+0.87 c/sh); total **+$7,965 / day
(+0.52 c/sh), positive on 9 of 9 days (R544)**.

The field makes +$7,103 / day (+0.30 c/sh) on about 1.5 times the volume. Per share the top 5 are better in almost every segment, by about
0.2-0.7 c/sh.

## The logic

1. **Adverse selection on the side the price moves into is real and they take it.**
   - In big trends they are run over at every price on the way, more than their usual share.
   - Their fills there lose 2-6 c/sh. No data sign that they pull those asks faster than others.
2. **Their protection is being on both sides.**
   - The side the price moves away from keeps getting filled by sellers who are on the wrong side of the move, and those fills win.
   - In a big trend the two sides cancel (-$5.9k vs +$6.2k a day). In a reversion the resting asks win.
   - The away/into volume ratio is 0.58 in big trends and 0.90 overall.
3. **What they earn is the average over cases**, not avoidance of the bad case:
   - small moves and reversions pay (+$0.8k to +$3.3k a day per case);
   - big trends cost about nothing net.
4. **Their edge over other makers is per-share quality in every segment plus size.** It is not a different pattern: the field shows the
   same signs everywhere.

## Ours (simulated guard ladder, same events)

| | Ask side (into) | Bid side (away) | Away / into volume | Total |
|---|---|---|---|---|
| Ours, all cases | -$55 / day (-0.20 c/sh) | +$136 (+0.52) | 0.94 | +$81 (+0.15 c/sh) |
| Ours, big trends | -$211 (-2.59) | +$179 (+3.62) | 0.61 | -$33 |
| Top 5, all cases | +$1,590 (+0.20) | +$6,375 (+0.87) | 0.90 | +$7,965 (+0.52) |

- Our two-sidedness is similar to theirs (0.94 vs 0.90; in big trends 0.61 vs 0.58).
- Per share we are lower in most segments; the largest gaps are in 2-tick moves and big trends.
- Our volume is about 3% of theirs.

## What changes in the earlier conclusions

- **SIDE-MOVES / ROBUST-CHECK C1** ("already at the next price, at the front") is restated:
  - it is true at the price where the move stops;
  - at prices run over in bigger moves they are also heavily present and lose (43% of that volume, -3.6 c/sh in big trends).
  - Being at the next price is not an edge in itself. It wins when the move stops or reverts and loses when it trends, about as FUTURE-BEST's
    oracle showed.
- **The idea that the top makers avoid the trend by pulling** is not supported. They stay two-sided and the other side offsets.

## Limits

- Fills only: a low share means "not filled there", never proof of a pull. Pulled orders are invisible.
- Values are 5 s marks, not settlement. WS-timed legs only (C12). Taker addresses are 90% known.
- The events start from our guard. Consumed days (C8). Ours are replay rule M (C5, C9).

Outputs:
- `data/predict_fun/research/move_cases_20261113/`: move_cases.txt / .json, posthoc.txt, events_*, segs_*, ours_*, day_*.json, logs.
- Copies in `reviews/move_cases_20261113/`.
