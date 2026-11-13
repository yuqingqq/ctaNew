# LEARN-PULL: what the top makers avoid, why they pull, and how our pulls differ (results, 2026-11-13)

Session ce68ebe0.

USER:
- "We learn from them first, check how do they avoid being hit, and why they keep quoting in some big moves, do they always quote in big
  moves? Can you tell why they pull, we pull in sometimes they pull as well, and we pull much more often than theirs, can you check the
  difference so that we can learn from them".
- During the run: "Yes keep checking our timing issue, it looks other makers can quote in right time even when there is pump/dump, they
  can still make profits by quoting both sides, why we avoid such periods, and what they avoid actually?"

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| Script `research/pf_learn_pull.py` + `run_learn_pull.sh` | e3ad37ff, before the first run | first day job stopped 16:43Z, no output |
| Fix 1: two trades with the same frame stamp tripped the look-ahead assertion; such ties are now a status | d4adcdef | stopped again, no output |
| Fix 2: a duplicated row key | 245fbc4c | 16:44-16:49Z, peak ~860 MiB |

Design and readings were unchanged by the fixes.

## Design

- **Population:** every front order at the best that we can prove from the public tape.
  - The proof is that a maker was the only one filled by a trade that stopped at that price, with the price still shown after (EXIT's
    proof).
  - We then follow it to the next trade at that price within 3 s. Under price-time priority a front order still resting is hit first.
  - **STAYED:** it is hit. **GONE:** another order is hit; it was pulled or moved.
  - 193,153 observations: top 5 78,294, field 114,859.
  - Statuses: 49,274 ambiguous because of estimated-time legs; 58,135 because the next trade shares the proof's frame stamp.
- **Ours on the same interval:** whether our guard (v <= -0.10) fired at a step whose cancel would land before the trade.
- **The next trade:** STOP or THROUGH, and the value of a front fill there (c/sh at 5 s, volume-weighted).
- **Features 100 ms before the trade:**
  - our guard value, FRONT-PULL's P, the 37 model inputs;
  - Binance and Predict moves "against the order";
  - the shown size at the price;
  - the maker's own position in the window.
- **Big moves (>= 3 ticks in 2 s after a signal):** whether the top 5 are filled at prices run over, by move speed and price level, and
  the inventory direction of those fills.

## Results

**R551, how often: we pull about twice as often as they do on the same intervals**

| Interval to the next trade | Top 5 GONE | Field GONE | Our guard pulls |
|---|---|---|---|
| 0-250 ms | 15.9% | 27.5% | 48.6% |
| 250-1000 ms | 30.7% | 41.7% | 64.1% |
| 1000-3000 ms | 49.2% | 62.3% | 76.6% |
| All | **27.8%** | 40.1% | **59.8%** |

**R550 AVOID-ADVERSE (9/9 days): what they avoid is specific trades, not periods**

| Next trade | Top 5 GONE | Our pulls | Value of a front fill there |
|---|---|---|---|
| Goes THROUGH the price | **47.0%** | 80.9% | +0.07 c/sh |
| STOPS at the price | **26.9%** | 58.9% | +0.50 c/sh |

- Their ratio is 1.74; ours is 1.37.
- When they stayed, the next fill was worth +0.47 c/sh; when they had pulled, +0.14.
- A front order is filled well even when the trade goes through (about break-even): being first in the queue is what makes staying cheap.

**R552 / R553: the 2 x 2 for the top 5's front orders**

| | Share of intervals | Next fill's value at 5 s | Share THROUGH |
|---|---|---|---|
| Both pull | 21.2% | +0.01 c/sh | 8.5% |
| **We pull, they stay** | **38.6%** | **+0.25** | 4.3% |
| They pull, we keep | 6.6% | +0.50 | 3.2% |
| Both keep | 33.6% | +0.72 | 1.8% |

- **R552 OUR-EXTRA-PULLS-COST** (positive on 8 of 9 days): where only we pull, the fill given up was worth having.
- R553 THEY-PULL-NEEDLESSLY (7 of 9) on this measure. Their extra pulls are few (6.6%) and may be moves to another price, which fills
  cannot tell apart.
- Where both pull, the trade was worth nothing (+0.01): that is the real danger both see.

**R554, why they pull** (pull share by quintile, Q1 -> Q5)

| Input | Top 5 | Ours | Reading |
|---|---|---|---|
| FRONT-PULL's P (the level goes soon) | 18% -> 47% | 19% -> 88% | both react; we react twice as hard |
| Shown size at their price | 42% (thin) -> 19% (thick) | 81% -> 36% | both pull a thin level |
| Own side's depth change (250 ms) | 39% (depleting) -> 22-27% | 76% -> 49-64% | both react |
| Our guard value v | 40% (risky) -> 22% | 89-93% -> 15% | at Q2 we pull 93%, they pull 30% |
| Binance imbalance against the order | 26% -> 33% (flat) | **35% -> 87%** | we react strongly, they barely do |
| Volatility (sigma) | 30% -> 27% (flat) | **48% -> 67%** | we pull more in volatile periods, they do not |
| Binance return against (250 ms / 1 s) | 23-34% (flat) | 56-72% | weak for both, we are always higher |
| Their own inventory | 25-32% (flat) | n/a | weak |

- A model of their pulls (leave one day out) reaches AUC 0.78 with all inputs, against 0.59 with v alone and 0.53 with inventory alone.
- The top inputs by gain are:
  - time since the proof (mechanical);
  - P;
  - the shown size;
  - pending bid settlements;
  - own-side depth change;
  - inventory;
  - volatility;
  - the Binance imbalance change.
- **R555 NOT-SHOWN:** inventory barely matters. Orders that would add to their position are pulled 28.8%, those that reduce it 26.1%
  (ratio 1.10, below the 1.2 line).

**R556, big moves: they keep quoting, and do not back off as the move extends**

| Speed (time to the peak) | Events with a top-5 fill at a price run over | Their share: old best / 1st / 2nd / 3rd+ price run over / peak |
|---|---|---|
| <= 250 ms | 28.5% of 7,867 | 26.5% / 35.9% / 42.4% / 43.3% / 56.8% |
| 250-1000 ms | 53.7% of 9,613 | 32.1% / 37.9% / 41.6% / 43.8% / 58.2% |
| > 1000 ms | **73.1%** of 13,108 | 32.1% / 41.0% / 43.6% / 44.1% / 58.6% |

- Their share rises with depth and with slower moves: no sign of pulling deeper prices as the move continues.
- Inventory direction of their fills at prices run over: reducing 41%, flat 9%, increasing 50%. The field's: 25 / 31 / 44.

## What it means

1. **They avoid trades, not periods.** Their pulls are local to their own price: the level is likely to go (P), the queue at their price is
   thin, or their own side is depleting. They pull 47% before trades that go through their price and 27% before trades that stop there.
   Volatility, Binance moves and their own inventory hardly change their pull rate.
2. **We avoid periods.** Our guard reacts to Binance imbalance (87% pulls in the top quintile vs their 33%) and to volatility (67% vs 27%).
   That is what a pump or dump looks like, so we leave both sides for the whole move.
3. **On the same intervals we pull about twice as often (60% vs 28%).** Where we pull and they stay (39% of intervals), the front fill we
   would skip is worth +0.25 c/sh.
4. **Why they can afford to stay, and why copying their pull rule alone is not enough for us.**
   - They are at the front, and a front order is filled first, by the start of a trade: +0.50 c/sh when it stops, about break-even even
     when it goes through.
   - Our orders join late and sit at the back (QUEUE-PLACE: 359 ms after the level forms). They are filled mostly by big trades, so our
     guard has more to protect.
   - FRONT-PULL already tested resting by default and pulling on P, close to their rule. With our queue position (rule M) it lost to our
     guard on 9 of 9 days. With nothing ahead (rule U) the same policy earned +$171-246 a day.
   - The queue position decides whether their timing works.
5. **In big moves they do quote throughout** (73% of slower big moves), take the adverse fills, and are paid on the other side (MOVE-CASES).

## Next (proposals, the user's call)

- **Queue position first.** Measure our real queue position and fills live (the live probe after the churn-fix deploy; the replay's queue
  rule is not validated, B1a, C10). Then a replay of "keep orders to reach the front, pull only on the level's own risk (P, thin queue,
  own side depleting), not on Binance imbalance or volatility alone", reported at rule M and rule U.
- **A narrower guard test.** In replay, stop the guard from pulling when only Binance imbalance / volatility drive it and the level itself
  is safe (low P, thick queue). Report $ a day and pulls at M 0 / 50 / 250, WS / E2.

## Limits

- Fills only. A pull is inferred from a front order not being hit; a maker with several orders at the price needs all gone.
- Price-time priority is assumed; B1a found a fifth of live fills contradicting the shown-size queue.
- The proof selects front orders after small stopping trades. Ambiguous intervals are excluded, which is a large status.
- GONE includes moves to another price.
- Values are 5 s marks of one fill. Our guard and P are our signals. Consumed days (C8).

Outputs:
- `data/predict_fun/research/learn_pull_20261113/`: learn_pull.txt / .json, obs_*, bigfills_*, bigevents_*, day_*.json, logs.
- Copies in `reviews/learn_pull_20261113/`.
