# B12c: a better model, or faster reaction? (2026-11-11)

As of 2026-11-11 04:27Z (session ce68ebe0). USER "Can you check if they have a better model, or they just react
faster". v1 `research/pf_b12c_model_or_speed.py`, committed and pushed before its run (8bc18b8e; output 04:19Z). v2
`research/pf_b12c2_binance.py`, written after v1's result and committed and pushed before its own run (dc8dde4e;
output 04:25Z). Output in `live_vs_replay/b12c_20261111/`, copies in `reviews/b12c_20261111/`.

Public tape and the cohort's Binance grid. One consumed partial day (the 220 windows of 2026-11-08): point estimates.

**Correction (05:49Z, after B12e).** The Binance grid used by v2 (step5 rows_rl_d2.bn_mid) is coarse: 90% of 1 s
intervals in the first minute show no change, and a window's first minute holds a median of 5 distinct values
(handoffs/20261111_b12e_smart_takes.md). v2's NOT-BINANCE (R55) therefore reads as not shown: the instrument could not
see sub-second Binance moves. 'Not faster reaction to Binance' is withdrawn. The model-free split (the gap earned
within 250 ms of the report) and v1 stand.

## Question

B12b: the top makers' edge is not their queue place, and it sits while our guard's value is falling. Do they predict
the moves (a better model), or act sooner on the same moves (faster reaction)?

## Design

Same legs as B12b part A: 55,514 at-best fills that did not go through the level, guard ON / OFF at lag 50; the top 5
(10,656) against ranks 6 and below (44,858). The instrument gets extra time: the gap is decomposed within cells of
what the instrument shows up to L ms after the trade. If the gap goes away with what our side sees shortly after the
trade, it is speed; if only with what comes much later, a model; if never, the edge is not that information. X =
the share of the within-guard gap that the cells account for.

- v1: the instrument is our guard's value v (quintile of its level x decile of its change up to anc + L).
- v2: the instrument is the Binance price path (Tokyo-stamped 50 ms grid; signed returns before the report and up to
  anc + L, in quintiles, x terciles of the mid's distance from 50 c), plus a model-free split of when the gap builds.

Controls, all passed before any table: v1 B12b's controls, the value lookup, a planted 'speed' world (X(500) 0.89 ->
SPEED, MATCHED-AT-500MS) and a planted 'model' world (X(500) 0.00 -> MODEL), the label rules. v2 v1's controls, the
Binance lookup and sign, planted worlds 'visible' (X_past 0.88 -> REACT-FASTER), 'later' (X(500) 0.00, X(2000) 0.85 ->
BETTER-MODEL) and 'other' (X(2000) -0.01 -> OTHER-INFORMATION), the label rules. K0 the frame reproduces b12.json and
B12b's part A legs; K1 (v2) the rebuilt mids reproduce v1 and v5 on every leg (0 difference). No leg lacks a value in
v2; v1 dropped 299 legs without a guard value at one of its times.

## Results (declared)

**v1, our guard's value: INSTRUMENT-WEAK.** Two seconds after the trade, our value accounts for none of the 1 s gap
(X(2000) -0.02; past 0.11, 0 0.03, 250 -0.01, 500 -0.10, 1000 -0.03). By its rule v1 cannot tell a model from speed
(R50, R51 n/a).

**v2, the Binance path: NOT-BINANCE -> OTHER-INFORMATION.**

| Horizon | gap within guard | X past | X +0 | X +250 ms | X +500 ms | X +1 s | X +2 s |
|---|---|---|---|---|---|---|---|
| 1 s | +0.211 | 0.06 | 0.09 | 0.13 | 0.15 | 0.14 | 0.22 |
| 5 s | +0.172 | 0.07 | 0.15 | 0.19 | 0.24 | 0.26 | 0.59 |

At 1 s the Binance path from before the trade up to 2 s after it accounts for 22% of the gap (R55 NOT-BINANCE; R56
n/a; R57 OTHER-INFORMATION). Residual at +2 s: +0.165 c/sh against null p99 +0.040 (N5).

**When the gap builds** (model-free, within guard, c/sh): spread +0.008; first 250 ms after the report +0.196; 250-500
ms +0.012; 0.5-1 s -0.004; 1-5 s -0.039. Net +0.211 at 1 s, +0.172 at 5 s.

**Binance moves at their fills** (signed, bps; + is good for the maker): top 5 -0.040 over the 250 ms before, -0.083
over the second before, -0.021 over the next 0.5 s, -0.052 over the next 2 s; ranks 6+ -0.042, -0.028, -0.004, -0.020.

| Reading | Result |
|---|---|
| v1 R53 instrument | INSTRUMENT-WEAK (X(2000, 1 s) -0.02) |
| v1 R50 / R51 | n/a (instrument weak) |
| v2 R55 instrument | NOT-BINANCE (X(2000, 1 s) 0.22) |
| v2 R56 / R57 | n/a / OTHER-INFORMATION |

## What it says

1. Not a better trend model. The whole gap is earned in the first 250 ms after the trade report (+0.196 of +0.211
   c/sh); from 0.25 to 5 s nothing is added (-0.03). A model that predicts where the price goes over seconds would keep
   earning after the first quarter second.
2. Not faster reaction to Binance. The Binance path from before the trade to 2 s after it accounts for 22% of the
   1 s gap, and the top makers' fills come after Binance moves that are no better than everyone's (slightly worse).
   Our guard's value accounts for none of it either (v1).
3. So their edge is local to Predict.fun and immediate: their fills are followed, within a quarter second, by Predict
   prices that hold, others' by prices that move. Polymarket is an unlikely source: the research session's lead-lag
   study (2026-11-10, branch research/predict-polymarket-leadlag-20261110) finds Predict moves before Polymarket by
   100-200 ms, and Polymarket moves cut 1 s prediction error by 0.4%.
4. What that leaves, not tested: (a) they react to Predict.fun's own trades and book, for example pulling after the
   first taker order of a burst, before the follow-ups hit; (b) their own size and refills hold the price after their
   fills, a mechanical part of the markout.

## What this changes

- A better Binance model, or a faster Binance feed, would not close this gap. Guard work (B7) should look at Predict's
  own trades and book over the last few hundred ms.
- If (a) holds, the top makers' speed is on Predict's own data path; ours is measurable with B1 (live run with the
  A4b capture, USER D2).
- Next test (proposed, B12d): taker-order bursts on the public tape. Are the top makers filled by the first taker
  order of a burst and absent for the follow-ups? Do they refill at the same price after their fills (their effect on
  the mid)?

## Limits

One consumed Sunday; point estimates. Trade times are reports (the match precedes them by about 0.2 s for our own
legs), so "the first 250 ms after the report" is roughly 0.2-0.45 s after the match. The Binance path is summarized by
net returns in quintile cells (coarse; biases X down). At 5 s the Binance path up to +2 s accounts for 59% of the
smaller 5 s gap; that is descriptive, not a declared reading, and sits beside a 1-5 s mid split near zero; it would need
its own test. Delta is proxied by the distance of the mid from 50 c. The Polymarket result is the research session's
(partial days, no intervals).
