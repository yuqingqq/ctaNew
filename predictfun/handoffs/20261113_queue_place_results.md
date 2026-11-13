# QUEUE-PLACE: do we lose our place by re-placing, or are we behind from the start? (results, 2026-11-13)

Session ce68ebe0. USER: "How do we become front in the queue, joining existing early doesnot make much sense, as we will also be behind
them, or can you check if we join early, then replace to be back in the queue, or is it because we are already back from when we join and
lose the position from the beginning".

Scripts and runs, claude slot:

| Step | Script / driver | Commit | Run |
|---|---|---|---|
| Main | `research/pf_queue_place.py`, `run_queue_place.sh` | bad57131 + rename 2f7143ea, both before the run | 11:16-11:20Z, peak 894 MiB |
| Post hoc WHY-LATE | `research/pf_queue_place_why_late.py`, `run_queue_place_why_late.sh` | af81edf8, before its run | 11:22-11:24Z |

- **Name fix:** bad57131 first committed the script as `pf_queue_origin.py` and overwrote the 11-06 descriptive script of that name. That
  file is restored byte-identical (2f7143ea) and this study was renamed QUEUE-PLACE before any run.

## Design

- **Simulator:** the guard ladder (LDG: 30 sh kept through partial fills, rungs 0-5, every rung pulled on the guard v <= -0.10 and
  re-placed once clear for 250 ms).
  - Copied as sim4 with bookkeeping: each order's price and times, why it was cancelled, and the queue ahead before each fill.
  - 9 consumed replay days, first 60 s, fill clocks M 50 (primary) and M 0.
- **Population:** our fills at the best on exactly-timed trades, aged (the same fills as BEST-GAP's BASE for rung 0). Estimated-time
  trades are printed separately (C12).
- **Per fill:**
  - the run of book frames in which our price is the best, and its start tb (the level's "birth": a better price appeared = improve,
    the level in front went away = fallback);
  - our chain of earlier orders at that price;
  - the queue ahead just before the fill (A, rule M), split into parts that add up to it:

| Part | What it is |
|---|---|
| U | already at the level when it became the best |
| LAT | joined within 97 ms of the birth (our median feed + order path) |
| LATE | joined after that, before our first order at this price went live |
| PREA | ahead of our first order when it was resting at this price before the level became the best |
| REQ | joined between our first order and the one that filled: lost by our own re-places |
| RAISE | the simulator's next-snapshot allowance at activation |

- **KEPT:** the queue ahead if our first order at the price had been kept.
- **Top makers:** for their at-best fills on the same trades:
  - the run kind and the run's age at the fill;
  - PROOF = their order had a fill at the price more than 250 ms before the run began. Their orders are seen only through fills, so this
    is a lower bound.

Controls passed:
- sim4 equals the simulator (sim_arm) fill for fill on 5,184 window-sides x 6 rungs x 2 clocks.
- The decomposed queue ahead equals the simulator's on all 47,823 checked fills (0 differ, none shared).
- Rung 0's population equals BEST-GAP's BASE: 9,398 fills, 105,603 sh, +0.007 c/sh. The real rows equal BEST-GAP's.
- The parts add up to A on every row.
- Synthetic cases with three planted errors caught.

Statuses (at-best fills, all rungs, M 50):
- 24,289 decomposed;
- 449 whose frame on the simulator's clock no longer showed our price as the best;
- 25 censored.

## Answer: we are behind from the moment we join; re-places add a fifth

Queue ahead at the fill, rung 0, M 50, exact trades, aged fills (share-weighted mean sh; share of A):

| Part | sh | Share |
|---|---|---|
| U: there when the level became the best | 1.8 | 12% |
| LAT: joined within 97 ms | 1.6 | 11% |
| **LATE: joined after 97 ms, before our first order** | **5.8** | **40%** |
| PREA: ahead of us at the deeper level | 0.5 | 3% |
| **REQ: our own re-places** | **3.1** | **22%** (guard pulls 80%, orders used up by fills 20%) |
| RAISE: next-snapshot allowance | 1.7 | 11% |
| A: queue ahead at the fill | 14.5 | |
| KEPT: had our first order been kept | 9.7 | |

The same split appears at M 0 (re-places 23%, LATE 38%) and on estimated-time trades (18%, 41%; A 68 sh).

Readings (declared, evaluated in code):
- **R480 BEHIND-FROM-ENTRY, STABLE (7 of 9 days).**
  - Re-places cause 21.7% of the queue ahead. Per day: 28 / 22 / 21 / 14 / 24 / 32 / 21 / 19 / 23%.
  - 78% was ahead of us when our first order at that price went live.
- **R481 LATE-ENTRY, STABLE (8 of 9).** The largest part joined after the level's first ~100 ms and before us.

**We arrive late:**
- Our first order at a level goes live a median **359 ms** after the level becomes the best. Only 5% arrive within 97 ms; 53% at
  97-500 ms, 33% at 0.5-2 s, 9% later.
- When the level appears it is thin (U, 12% of what ends up ahead of us). The queue forms in the next few hundred ms, before we get there.

**Why we arrive late (post hoc, WHY-LATE, rung 0's at-best fills from orders that joined after the birth):**

| Our rung 0 at the level's birth | Share of shares | c/sh | STOP | Our first order live after the birth, ms median (p25 / p75) | What ended the earlier order |
|---|---|---|---|---|---|
| OUT: no order (pulled, or waiting the 250 ms clear) | 60% | +0.05 | 33% | 350 (274 / 706) | guard 81%, filled 8% |
| HELD: our order kept at the old price | 38% | -0.13 | 34% | 400 (130 / 1,251) | guard 63%, drift 23%, cross 13% |
| PEND | 2% | | | | |

- The guard fires when the price moves, which is when new best levels form. At the birth our order has usually just been pulled.
- It comes back only after the guard clears plus the 250 ms clear wait plus the 40 ms order path.
- When the best improves, rung 0 keeps its order at the old price, held within 3 ticks. It cannot be at the new level until that order
  goes.
- New levels (improve): HELD 61%. Fallback levels: mostly OUT.

## Being at the price before it becomes the best pays

Our at-best fills by origin (M 50, exact trades, aged):

| Origin | Share of shares, rung 0 / all rungs | c/sh, rung 0 / all | STOP share, rung 0 / all | Mean queue ahead |
|---|---|---|---|---|
| **PRE-KEPT**: our order rested at the price before the level became the best | 8% / 17% | **+0.287 / +0.203** | **63% / 57%** | 6 sh |
| PRE-REPLACED: we were there before, then re-placed | 8% / 7% | +0.043 | 37% | 11 |
| POST-FIRST: joined after the birth | 59% / 53% | +0.024 / +0.026 | 35% | 16 |
| POST-REPLACED: joined after, then re-placed | 25% / 23% | -0.129 | 33% | 15 |
| Top 5 (real), for reference | | +0.405 | 80% | |

- **R482 BEING-THERE-FIRST-PAYS:** PRE +0.156 against POST -0.020 c/sh. Higher on 7 of 9 days, lower on 2.
- Deep rungs reach the best this way: 92% of their at-best shares are PRE-KEPT (rung 1: +0.160 c/sh, 50% STOP). They are only 10% of
  our at-best shares, because the guard pulls every rung when the price moves toward them.

## The top makers (R483, descriptive)

| At-best fills, exact trades | Level kind: fallback / improve | Level's age as the best at the fill, median (p25 / p75) | PROOF (lower bound) | c/sh | STOP |
|---|---|---|---|---|---|
| Top 5 | 61% / 39% | **841 ms** (296 / 1,905) | 4% | +0.405 | 80% |
| Field | 44% / 55% | 839 ms (286 / 2,023) | 2% | +0.239 | 71% |
| Ours, rung 0 | 57% / 43% | **2,143 ms** (1,050 / 4,035) | 0% (truth: 16% PRE) | +0.007 | 37% |
| Ours, all rungs | 61% / 39% | 1,971 ms (904 / 3,802) | 0% (truth: 24% PRE) | +0.017 | 38% |

- They fill on the same kinds of level as we do (about 60% are levels that came from behind).
- Their fills come early in a level's life as the best: 0.84 s against our 2.1 s. They are in the level from early on.
- PROOF is too weak to show when they entered. Ours is 0% while our truth is 16-24%, because an order resting behind is rarely filled
  before its level becomes the best.

## What it means

1. **Not mainly the re-places.** About a fifth of the queue ahead of our filled orders is due to our own re-places, mostly guard pulls.
   Keeping our first order would cut the queue ahead from 14.5 to 9.7 sh (12.8 to 9.7 without the snapshot allowance). KEEP-PLACE
   showed the pulls themselves are needed.
2. **Mostly we are behind from the start because we arrive late.** The queue that ends up ahead of us forms in the first few hundred ms of
   a level's life. Our first order arrives a median 359 ms after the level appears.
   - Mostly because the guard has just pulled us and we wait for it to clear plus 250 ms.
   - Or because rung 0's order is still parked at the old price.
3. **Joining an existing level earlier is not the way to the front** (as you said). The front belongs to:
   - **orders already at the price when it becomes the best.** About 60% of at-best fills, theirs and ours, are on levels that came from
     behind. Our orders that were there first earn +0.20 to +0.29 c/sh at 57-63% STOP.
   - **orders in within the first ~100 ms of a new level.** At the birth only 12% of the eventual queue is there.
4. **The conflict to solve:** the guard pull (which we need) is what takes us out of the price at the moment the new best forms.
   KEEP-PLACE's immediate re-entry did not raise $/day (NOT-SHOWN). Re-entering faster at the new best alone is not enough: what pays
   is being at the price before it becomes the best.

## Next (proposals, the user's call)

**QUEUE-FRONT (replay), the guard ladder changed in two ways:**
- (a) **Keep-and-add.** When the best improves, the order at the old price stays (it is already in that queue) and a new order goes to
  the new best. When the price comes back, the kept order is at the front. Needs a multi-order lane.
- (b) **Re-enter behind, not at the best.** After a guard pull, re-enter first at the levels behind the best: we are early there, and
  they become the best when the price falls back.

Measure:
- the PRE share and STOP share at the best;
- c/sh on exact trades;
- $/day with the age and C12 splits.

The live probe after the churn-fix deploy remains the check of rule M.

## Limits

- Replay rule M (B1a). Every queue number is the simulator's model of the queue, not an observed position.
- Consumed days, exploratory.
- The six rungs are independent orders.
- The top makers' entry times are not observable: PROOF is a lower bound and too weak here.
- WHY-LATE is post hoc and descriptive.
- The 97 ms split is fixed (QR).

Outputs:
- `data/predict_fun/research/queue_place_20261113/`: queue_place.json / .txt, fills_*.parquet, real_*.parquet, links_*.parquet,
  day_*.json, logs, why_late/.
- Copies (json / txt / logs) in `reviews/queue_place_20261113/`.
