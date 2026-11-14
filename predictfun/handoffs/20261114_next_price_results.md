# NEXT-PRICE: where we are when the next price becomes the best, and who holds its front (results, 2026-11-14)

Session ce68ebe0.

USER: "For example, when best bid is 50c, their orders are 49c, and always there? Can we also put our orders early there, It's we are late
when we first there or we late because we repost".

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| Script `research/pf_next_price.py` + `run_next_price.sh`, readings R570-R572 declared | 89b45cac, before the first run | 03:26-03:32Z |
| `posthoc`: queue numbers paired within the same births (POST HOC) | 2626caae | 03:33Z |
| `posthoc2`: rungs 0-4 only, after posthoc showed the rung-5 defect below (POST HOC) | 0eaa311b | 03:35Z |

## Design

- **Simulator:** the guard ladder exactly as QUEUE-PLACE.
  - 30 sh per rung; rung d is placed d ticks behind the known best and kept while the best stays within 3 + d ticks.
  - Every rung is pulled on the guard (v <= -0.10) and re-placed 250 ms after it clears.
  - 9 consumed days, first 60 s, fill clocks M 50 / 0 / 250.
- **Births:** every moment a price becomes our side's best.
  - **Fallback:** the best goes, e.g. 50c, and the next price, e.g. 49c, becomes the best. These are the user's case.
  - **Improve:** a better price appears.
  - Statuses (outside the readings): guard value masked at the window start, or no decision step yet: 15,702 fallback births.
- **Our status at the birth, over all six rungs at that price:**

| Status | Meaning |
|---|---|
| LIVE | our order is there and not being cancelled |
| PULLED | in the 5 s before, we had an order there and the guard pulled it |
| MOVED | the ladder re-centred it |
| USED | used up by fills |
| NEVER | no order of ours at that price in the 5 s before |

- **Queue ahead (rule M):** at the birth for LIVE orders, and for PULLED orders had they been kept. Compared with our next order at that
  price, at its activation.
- **The top makers (real fills):** the first taker order at the new best.
  - SMALL first trades (<= 30 sh) fill only the front of the queue.
  - CROSSING ones are taker orders that took the old best and continued into the new price. The orders they met there were resting
    before the birth.
  - Baseline: the top 5's share of all exactly timed at-best volume, 33.3%.
- **Controls passed:**
  - the simulator equals sim_arm on every window-side, rung and clock;
  - synthetic cases with four planted errors caught;
  - QUEUE-PLACE's self-tests pass.

## Results

**1. When the next price becomes the best, is our order there? (fallback births, M 50, n 154,740)**

| LIVE | PULLED by the guard | MOVED | USED | OTHER | NEVER (no order there for 5 s) |
|---|---|---|---|---|---|
| **7.8%** | **59.9%** | 1.3% | 1.2% | 0.3% | 29.5% (guard on at the birth for 88% of these) |

- **R570 LATE-BY-REPOST (STABLE, 9 of 9 days).** Of the births where we were not there:
  - 66.4% we had an order at that price and cancelled it ourselves (PULLED + MOVED);
  - 32.0% we had none there.
  - Per day the repost share is 58-70%.
- The same at M 0 and M 250 (LIVE 7.8 / 7.9%, PULLED 60.1 / 58.5%).
- By fallback size:
  - 1-tick fallbacks: LIVE 8.7%, PULLED 60.2%;
  - larger ones: 4.9% / 58.9%.
- **When the guard pulled it:**
  - our cancel was decided a median 558 ms before the price became the best (p25 202 / p75 1,362);
  - the order had rested only 410 ms (110 / 1,060).
- **When we come back:** our next order at that price goes live a median 434 ms after the birth (311 / 741). In 72% of these births the
  level is gone before we are back.
- **When we were there (LIVE):**
  - still there 250 ms later 57%, 1 s later 34%;
  - within 1 s the guard pulled 57% of them.
- **Our order at the price is mostly rung 1** (one tick behind the best: 50% of LIVE, 33% of PULLED). We do put orders at 49c early; the
  guard takes them out.

**2. Where would we be in the queue? (rule M; rungs 0-4, post hoc pairing)**

| Our order | Queue ahead at the birth, median (p25 / p75) | Size shown there | At the front (<= 5 sh ahead) | Order age at the birth |
|---|---|---|---|---|
| LIVE (there at the birth) | 190 (100 / 326) sh | 232 sh | 2.1% | 0.83 s |
| PULLED, had it been kept | **80 (18 / 196) sh** | 125 sh | **15.3%** | 1.33 s |
| Our next order after the birth | **469 sh** (paired with kept 133 sh on the same 27,844 births) | | | |

- Keeping the order we already had saves a median 288 sh of queue against coming back (same births).
- Even then we would rarely be first. Our orders live about a second, while the queue at that price formed earlier, and the top makers
  are in it.
- **R571 EARLY-IS-AHEAD: NOT-SHOWN** (declared, all rungs). LIVE 189 and kept 66 against next 439 sh pooled, but both at most half on only
  2 of 9 days.

**3. Who holds the front of the new best? (real fills)**

| Birth | First trades at the new best (exact) | A top-5 maker filled | Top-5 volume share | x their baseline |
|---|---|---|---|---|
| Fallback, all | 74,727 | 50.2% | 50.9% | 1.53 |
| Fallback, SMALL (front only) | 59,967 | **43.4%** | **45.4%** | **1.36** |
| Fallback, CROSSING (resting before the birth) | 21,005 | **61.9%** | 51.9% | 1.56 |
| Improve, SMALL | 55,570 | 20.2% | 21.8% | 0.66 |

- **R572 THEY-HOLD-THE-FRONT:** x1.36, above their baseline on 9 of 9 days (x1.22-1.63).
- Statuses for fallback births: first trade not exactly timed 34,290; no trade at the price while it was the best 61,425.
- They hold the front of a price that comes from behind (fallback) about half the time, more than any other group. At new better prices
  they are under their usual share. Their front comes from resting at the next price, not from racing to new prices.

## What it means

1. **"Their orders at 49c, always there?"** Often, not always.
   - When 49c becomes the best, the first small trade there fills a top-5 maker 43% of the time, 1.36x their usual share.
   - When a seller runs through 50c into 49c, a top-5 maker is filled at 49c 62% of the time. Those orders were resting before 50c went.
2. **"Can we put our orders early there?"** We already do: the ladder keeps an order one tick behind the best.
3. **"Late at first arrival or because we repost?"** Mostly because we repost.
   - In 60% of births the guard pulled our order at that price about half a second before it became the best.
   - In another 26% of births we had no order at that price for 5 s and the guard was on at the birth (88% of the NEVER cases).
   - So in about 86% of births the guard had pulled us, or was on, when the price became the best. We come back ~0.4 s later behind
     ~440-470 sh.
4. **Keeping the order would put us much further forward but rarely first.**
   - Median 80 sh ahead instead of ~470, and 15% at the front.
   - Our orders rest about a second; the top makers' orders were in that queue before ours.
5. **What keeping costs is not measured here.** The guard pulls protect us from being run over.
   - FRONT-PULL-SIZE: a resting ladder lost even at the front.
   - MOVE-CASES: the top makers lose -3.6 c/sh at prices run over in big trends.
   - PULL-DEPTH / FUTURE-BEST, which keep a deeper order unless the move is likely to reach it, gained +$18-44 a day at M 0 / M 50 and flip
     at M 250.

## A simulator defect found (FINDINGS P18)

- An order placed at a price beyond the 5 shown levels gets 0 sh queue ahead, because sim2 / sim4's place() reads the unknown level as 0.
  It starts at the front.
- Here: rung-5 orders had 0 sh ahead at 97.8% of their LIVE / PULLED births, so rung 5 is left out of the queue numbers above.
- Reach in earlier ladder results: rung 5 is ~0.0% of the guard ladder's at-best fill shares (QUEUE-PLACE, M 50). Fills from trades that
  take the whole level do not depend on the queue. Not re-run.

## Limits

- Replay rule M: queue numbers are the simulator's model, not observed (B1a; only a live run validates).
- Our order is not in the public book.
- The rungs are independent orders.
- The top makers' entry times are not observable: CROSSING proves presence before the birth, not for how long.
- First trade only. The 5 s lookback and the 30 sh line are fixed. Consumed days (C8).
- The queue pairing and the rung-5 exclusion are post hoc.

Outputs:
- `data/predict_fun/research/next_price_20261114/`: next_price.txt / .json, next_price_posthoc.txt, next_price_posthoc2.txt, ours_*,
  births_*, day_*.json, logs.
- Copies (txt / json / logs) in `reviews/next_price_20261114/`.
