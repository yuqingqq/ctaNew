# GAP: the remaining gap between us and the top makers (results, 2026-11-12)

> **Re-check (2026-11-12 11:41Z, handoffs/20261112_age_recheck_results.md, USER):** 73-77% of our simulated profit comes from fills on orders under 100 ms old, which B1a found only 8-12% real. Without them v3's per-share lead over the top 5 is MIXED (8 / 6 of 9 days) and Y+F30's holds (9 / 8); our volume without them is about 40% of the figures below (v3: 567 of 1,433 sh a day at M 50).

Session ce68ebe0. USER: "Yea we do the test to check the performance first, then check what's the remaining gap to top makers".

Script and run:
- Script `research/pf_gap_top_makers.py`, driver `research/run_gap.sh` (767924de, committed before the first run).
- Run 11:24-11:31Z in the claude slot, peak 715 MiB.
- 9 consumed replay days, first 60 s of each window.

What is compared:
- **Theirs:** real maker fills at the best of their side on the public tape, 1.16M legs, ranked per day by shares (B12's rule).
- **Ours:** QR2's simulated 5-sh order per side.
- Both valued the same way: 1 s markout, with the same mids and anchors.
- Exploratory. Theirs is real and ours is simulated; rule M is unvalidated (B1a).

Controls passed:
- Each day's legs, with signer and taker order, align exactly with the simulator's.
- Our re-run arms reproduce QR2's committed day totals.
- Synthetic classifier, matcher and ranking cases; three planted errors caught.
- Statuses: 1,573,290 first-minute legs, of which 1,158,421 were at the best, 379,942 behind it, 34,119 inside the spread and 808
  without a frame.

## Answer: per share we are not behind; the gap is how much and when they trade

| | Shares/day at the best | c/sh 1 s | $/day 1 s |
|---|---|---|---|
| Top 5 (real) | 493,663 (98,733 per maker) | +0.23 | 1,118 (224 per maker) |
| Ranks 6-20 (real) | 534,188 | +0.25 | 1,360 |
| Rest (real) | 595,531 | -0.49 | -2,907 |
| v3 (simulated, M 50) | 1,433 | +1.03 | 14.7 |
| Y+F30 (simulated, M 50) | 2,232 | +1.05 | 23.5 |

- **R300 OURS-HIGHER (9 of 9 days at M 0 and M 50):** per share, our simulated v3 and Y+F30 beat the top 5's real at-best fills
  every day.
- **R301:** the top 5 trade 345 times our shares and earn 76 times our dollars (per maker 69x and 15x, against v3).

**R302 OFF-DOMINANT: 60% of the top 5's money is made while v3's guard says stay out.**

| Top 5, split by v3's guard at the fill (lag 50 ms) | Shares/day | c/sh 1 s | $/day |
|---|---|---|---|
| Guard ON: the moments v3 quotes | 34,646 | +1.30 | 449 |
| Guard OFF: the moments v3 stays out | 459,006 | +0.15 | 668 |

- While our guard is ON they earn about what we earn per share (+1.30 against our +1.03 to +1.38). They earn it on 24 times our
  volume.
- While it is OFF (93% of their volume) they still make +0.15 c/sh. A resting order of ours in those moments loses in this replay:
  -0.6 to -0.9 c/sh joining new levels without a guard (QR), -1.5 never pulling (S4).
- The OFF share is 0.52-0.70 on 7 days; on 10-31 and 11-01 it is -0.03 and 0.17.

**R303 DIFFERENT-FLOW: only 16.6% of our filled shares come from taker orders that also filled a top-5 maker at our price**
(7-31% per day).
- On those shared trades they make +0.89 c/sh and we make +0.76. On the trades that do not fill us they make +0.22.
- Our fills on unshared trades are better (+1.08).
- We mostly trade in different moments from theirs.

## What it means: the remaining gap has two parts

1. **When our guard says quote, the gap is size and place, not value.**
   - The top 5 alone take 34,646 sh a day at the best in those moments, at +1.30 c/sh ($449 a day).
   - We take 1,433 sh with one 5-sh order at the back of a ~500 sh queue.
   - Joining new levels fast (QR2's F30) is part of the place.
   - Size is untested: whether a larger order keeps the per-share value is a replay question (order size 25 / 50 / 100 sh during
     guard-ON moments).
2. **When our guard says stay out, they still earn, and we cannot.**
   - This is 60% of their money: +0.15 c/sh on 459k sh a day.
   - Earlier work places this edge on Predict.fun itself, within 250 ms of each trade (B12c), plus their own taking (B12d, about a
     quarter of B12's gap).
   - None of our tested signals sees it: the guard, the pressure guard (B14) and the keep-or-pull model (S4) all failed to make
     resting in these moments pay. The 250 ms danger-pull design (bbf0112d) was never run.

## Limits

- Theirs is real and ours is simulated; rule M is not validated (B1a). Our per-share advantage is a replay number, and the
  fast-entry part depends on the young fills B1a found mostly phantom.
- At-best maker legs only. Their fills behind the best and their taker orders are not in this table.
- First minute of each window only. Consumed days.
- Top makers are ranked per day (signer ids are per day). Our own account cannot be identified on these days; if it traded, it is
  among the ranked makers.
- Fees and rebates are not included, for either side.

Next (proposals, the user's call):
- **SIZE:** v3 / Y+F30 with order sizes 5 / 25 / 50 / 100 sh in the replay. Same days and controls, readings first.
- **The guard-OFF edge:** what the top 5's fills while our guard is OFF have in common (the trade before, the book in the last
  250 ms, their own taking). A new signal study.
- **Live, with the A4b capture:** whether any of our replay per-share value is real.

Outputs:
- `data/predict_fun/research/gap_20261112/`: gap.json / gap.txt, day_*.json, logs.
- Copies in `reviews/gap_20261112/`.
