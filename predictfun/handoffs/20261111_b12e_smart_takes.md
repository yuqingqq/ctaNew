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

## Binance from the raw stream (B12e v2, 06:00Z)

A post-run check first suspected the research Binance grid (step5 rows_rl_d2.bn_mid) of being coarse: 90% of 1 s
intervals in the first minute show no change. `research/pf_b12e2_binance_raw.py` (f0b19fc9, committed before its run)
recomputed the Binance features from the HF collectors' raw BTCUSDT bookTicker for 11-08 (best bid / ask, updates
every few ms). The raw stream shows the same: 90.3% of 1 s intervals at random times have an unchanged best-price
mid (grid 90.5%). The updates are size changes; the best prices sit still. So the grid is faithful and Binance's best
bid / ask was genuinely quiet. The declared resolution rule (R76: zero share <= 0.5) assumed the price moves in most
seconds; it reads BN-RESOLUTION-WEAK, so R70v2 / R71v2 / R74v2 are n/a by rule, because the market was quiet, not
because the recording was coarse.

| Binance best-price mid (raw), bps in the taker's favour | SMART | OTHER | RANDOM |
|---|---|---|---|
| second before: mean / median | +0.054 / 0 | +0.062 | +0.018 / 0 |
| second before: moved >= 1 bp their way / against | 4% / 2% | 4% | 1% / 1% |
| 2 s after: mean / moved >= 1 bp their way | +0.145 / 6% | +0.066 / 4% | +0.028 / 3% |

Near 50 c, with ~4.5 minutes left, a 5-minute binary moves roughly 4-5 c per bp of BTC, so even 1 bp matters; but in
94% of their takes Binance's best price had moved less than 1 bp in the second before (median 0). Their trigger is on
Predict for most of their takes; a few per cent follow a visible Binance move. B12c v2's NOT-BINANCE stands as a
statement about Binance's best bid / ask (the correction notes added at 05:49Z are superseded).

**Our own at-best fills by the book on our side** (frame before; thin = our side's size <= half the other side's,
heavy = >= twice; share of shares / net 1 s / legs):

| Makers | thin side | balanced | heavy side |
|---|---|---|---|
| top 5 | 27% / -0.45 (4,386) | 38% / +0.10 | 35% / +0.71 |
| ranks 6-20 | 21% / -0.52 | 37% / +0.02 | 42% / +0.58 |
| ranks 21+ | 39% / -1.00 | 33% / -0.10 | 29% / +0.52 |
| ours | 17% / -2.16 (14) | 11% / -0.50 (6) | 72% / -0.59 (37) |

Thin-side fills lose for everyone; heavy-side fills earn for everyone except us. Most of our fills are on the heavy
side, where we sit at the back of a long queue (B12b: OURS-BEHIND), so we are filled when the big level is being eaten.

## What it says

1. Their takes come at a recognisable moment on Predict: the side they hit is thin (median 84 sh against 228
   normally), the other side is heavy (312 against ~235), and Predict's price has already started moving their way
   (+0.61 c in the second before). They take out the thin level, and the price moves +1.1 c their way within 250 ms
   (median one tick) and +1.7 c within 1 s.
2. Our own model sees these moments: its 1 s prediction points their way at 81% of their takes, and our guard value
   for the side they hit is below zero at 90% of them, so v3 should already be out of that side.
3. They act in runs: more than half of their orders follow another of theirs within 250 ms.
4. Binance: its best bid / ask had not moved by 1 bp in the second before 94% of their takes (raw stream, v2).

## What this changes

- How to avoid their flow: not by knowing who they are, but by not being the thin side when the book is lopsided and
  our model points against that side, and by pulling that side right after the first smart hit (the follow-ups come
  within 250 ms). Our model already flags most of these moments; our losses are where we are still there (slow
  cancels, B12b).
- A guard input worth testing: book imbalance at the best (side size against the other side) with Predict's own recent
  move.
- Binance trade flow and top-of-book size (not only the price) at their takes are still unchecked.

## Limits

One consumed Sunday; point estimates; three addresses. Trade times are reports, so "the second before" includes about
0.2 s after the match, and part of the move after is their own take clearing the level. Binance is checked on its
best bid / ask only (not trades or sizes). Sizes and spreads are ~50 ms snapshots.
