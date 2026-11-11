# B12e: what happens when the top makers take (2026-11-11)

As of 2026-11-11 05:48Z (session ce68ebe0). USER "can you check their take, what happens on Binance or what events
happen during then?". Script `research/pf_b12e_smart_takes.py`, committed and pushed before its run (04422a15; output
05:45Z). Output in `live_vs_replay/b12e_20261111/`, copies in `reviews/b12e_20261111/`.

Public tape, chain match records, the cohort's Binance and model grids. One consumed partial day (the 220 windows of
2026-11-08): point estimates only.

## Design

SMART = 2,491 taker orders at the best in the first minute sent by top-5 makers' addresses (in practice makers 48, 51,
661, B12d post hoc). Compared with RANDOM (12,455 times: 5 per SMART order, same window and side) and OTHER (34,909 taker
orders by everyone else). Moves are signed so that + is in the taker's favour. Null: SMART against RANDOM labels
permuted within window, 500 draws. Readings R70-R75 in the script header. Controls passed (B12b's, planted lead / no
lead through the null, taker sign, label rules; K0, our address join).

## Results

| At the moment of the take | SMART | OTHER | RANDOM |
|---|---|---|---|
| Predict mid, second before (c) | +0.61 (35% > 0) | +0.26 | +0.08 |
| Predict mid, 250 ms after (c) | +1.14 (median +1.0) | +0.40 | +0.01 |
| Predict mid, 1 s after (c) | +1.74 (64% > 0) | +0.54 | +0.05 |
| Our model's 1 s prediction (c) | +0.60 (81% their way) | +0.22 (56%) | +0.03 (53%) |
| Our guard value of the side hit | -0.52 (10% > 0) | -0.34 | -0.10 (41% > 0) |
| Size at the best, side hit (sh) | median 84 | 203 (mean) | median 228 |
| Size at the best, other side (sh) | median 312 (mean 344) | 242 (mean) | 235 (mean) |
| Spread (ticks) | 1.03 | 1.08 | 1.03 |
| Binance, second before (bps) | +0.09 (19% > 0) | +0.10 (13%) | +0.01 (5%) |
| Binance, 2 s after (bps) | +0.11 | +0.06 | +0.02 |
| Age of our last Binance update (ms) | 37 | 57 | 67 |

- 55.2% of SMART orders come within 250 ms of another SMART order on the same side (they take in runs).
- Their gain on the legs they hit: +1.23 c/sh at 1 s.
- The upward SMART minus RANDOM differences (Predict moves, our model's prediction, Binance moves) are above their
  null p99. The null was one-sided upward, so for the lower values (size on the side hit, our guard value, Binance
  age) only the difference itself is reported. The spread does not differ (p 0.38).

| Reading | Result |
|---|---|
| R70 Binance lead | NO-BINANCE-LEAD (+0.075 bps over random; threshold +0.3) |
| R71 Predict lag | n/a |
| R72 our model | OUR-MODEL-SEES (37.3% of SMART above random's 90th percentile; instrument corr +0.297) |
| R73 book | THIN-SIDE (side hit median 84 sh against 228) |
| R74 Binance after | NO-BINANCE-FOLLOW (+0.086 bps) |

## The Binance grid is coarse (check made after the run)

The Binance series used here and in B12c v2 (step5 rows_rl_d2.bn_mid) changes in bursts: 90% of 1 s intervals in the
first minute show no change; a window's first minute holds a median of 5 distinct values; changes come 50-100 ms apart
(p50 0.1 s) and then nothing for up to ~5 s (p90 4.7 s), with sizes of $0.3-8.5 (p50 $2.9). That may be a quiet Sunday
or a coarse recording; no independent Binance recording of 11-08 was found to check against. Until it is checked:

- R70 / R74 here say only that this recording shows little Binance movement around the takes, not that Binance did
  not move. Our feed was busier at the takes (last update 37 ms old against 67 ms).
- B12c v2's NOT-BINANCE (R55) has the same weakness and is read as not shown (the instrument could not see sub-second
  Binance moves). B12c's model-free split (the gap earned within 250 ms) and v1 (our guard value) stand.

## What it says

1. Their takes come at a recognisable moment on Predict: the side they hit is thin (median 84 sh against 228
   normally), the other side is heavy (312 against ~235), and Predict's price has already started moving their way
   (+0.61 c in the second before). They take out the thin level, and the price moves +1.1 c their way within 250 ms
   (median one tick) and +1.7 c within 1 s.
2. Our own model sees these moments: its 1 s prediction points their way at 81% of their takes, and our guard value
   for the side they hit is below zero at 90% of them, so v3 should already be out of that side.
3. They act in runs: more than half of their orders follow another of theirs within 250 ms.
4. Binance: this recording shows little movement, but it is too coarse to say Binance did not move.

## What this changes

- How to avoid their flow: not by knowing who they are, but by not being the thin side when the book is lopsided and
  our model points against that side, and by pulling that side right after the first smart hit (the follow-ups come
  within 250 ms). Our model already flags most of these moments; our losses are where we are still there (slow
  cancels, B12b).
- A guard input worth testing: book imbalance at the best (side size against the other side) with Predict's own recent
  move.
- The Binance data question needs a sub-second Binance source for the cohort (or the live engine's own feed, via the
  A4b capture run).

## Limits

One consumed Sunday; point estimates; three addresses. Trade times are reports, so "the second before" includes about
0.2 s after the match, and part of the move after is their own take clearing the level. The Binance grid is coarse
(above). Sizes and spreads are ~50 ms snapshots.
