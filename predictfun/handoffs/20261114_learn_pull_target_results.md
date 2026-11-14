# LEARN-PULL-TARGET: what the top makers' pulls line up with, and our guard's own target (results, 2026-11-14)

Session ce68ebe0. POST HOC on LEARN-PULL's observation files (no raw data read, nothing simulated).

USER: "Why do they pull? You mean they pull because some trade will take the price level? Or what is their target to train the model to
pull".

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| Script `research/pf_learn_pull_target.py` + `run_learn_pull_target.sh`, readings R557-R560 declared | a325538c, before the first run | 02:25Z, peak 452 MiB |
| `check`: the same with every trade counted once (POST HOC after the first report, descriptive, no readings) | acbdb7f7 | 02:27Z |

## What can and cannot be known

- We cannot see their model or whether they train one. We can measure which outcome their pulls line up with.
- Two candidate targets:
  - **LEVEL TAKEN:** the next trade at their price goes THROUGH it.
  - **FILL LOSES:** a fill at their price on that trade loses money, whether the trade goes through or stops there.
- **Our guard's own target** (`pf_badfill_grid_tx.py`, the frozen guard's label): the 1 s markout of a fill at the known best IF that price
  is swept through within 0.5 s, else 0.
  - That is LEVEL TAKEN, counting only the loss.
  - A fill on a trade that stops at our price counts 0, good or bad.
  - We pull at v <= -0.10 c, the same line for an order at the front or at the back of the queue.

## Population

- LEARN-PULL's front orders at the best, proven by a sole-maker stop and followed to the next trade at that price within 3 s.
- Top 5: 78,294 observations; field 114,859. Nine consumed days, first 60 s, WS-timed tape.
- Statuses:
  - v5 = 0 (no sign): 224;
  - no v or P (left out of T3): 1,542;
  - interval of exactly 3000 ms (in the last bucket): 20.

## Results

**T1. When they are gone, by the next trade at their price (bucket-standardised over the interval to the trade)**

| Next trade | Fill at 1 s | Top 5 gone | Our guard pulls | n |
|---|---|---|---|---|
| THROUGH | loses | 47.4% | 83.5% | 2,239 |
| THROUGH | wins | 36.6% | 70.2% | 1,102 |
| STOP | loses | **39.9%** | 73.7% | 19,712 |
| STOP | wins | **22.5%** | 53.7% | 54,850 |

- Trades that go through are 4% of the next trades at their price.
- Among trades that stop, they are gone 1.77x as often before a fill that loses at 1 s as before one that wins. At 5 s the ratio is 1.26,
  and higher on 9 of 9 days (**R557 PULLS-TRACK-FILL-LOSS**). Our pulls: 1.37 at 1 s, 1.17 at 5 s.
- The field: 1.42 at 1 s, 1.17 at 5 s.

**T2. The value of the fills they kept and skipped (c per share)**

| Counting | Kept | Skipped | Gap | Part the trade's kind accounts for |
|---|---|---|---|---|
| Each trade once, 1 s (post hoc) | +0.56 | **-0.12** | +0.68 | 6% (within-kind on 9 / 9 days) |
| Each trade once, 5 s (post hoc) | +0.49 | -0.22 | +0.71 | 6% (9 / 9) |
| Volume-weighted, 5 s (declared) | +0.47 | +0.14 | +0.33 | -2% (**R558 GAP-MOSTLY-WITHIN-KIND**, 7 / 9; two days with gap <= 0) |

- Within trades that stop at their price, at 1 s: kept +0.58, skipped -0.06. Within trades that go through: kept -0.27, skipped -0.90.
- The volume-weighted rows understate the gap: the volume at their price depends on their own presence (a trade through the price takes less
  there once they have left).

**T3. At their own pull count, would our signals pick the same trades?**

- Pulled per day and interval bucket as many as they did (27.9%), chosen by:
  - their actual pulls;
  - our guard value v, read 100 ms before the trade (lowest first);
  - FRONT-PULL's P (highest first);
  - random (200 draws);
  - the bound (the fill's own value).
- Declared, volume-weighted ($ kept):
  - **R560 FAIL**: their pulls kept more than the random p95 on only 3 of 9 days, so **R559 / R559P read VOID**.
  - This is the same weighting problem as in T2.
- Post hoc, each trade once (c per trade kept):

| Horizon | Theirs | Random (mean) | Our guard v | P | Bound |
|---|---|---|---|---|---|
| 1 s | +0.55 (above the random p95 on 9 / 9) | +0.36 | **+0.58** (>= theirs 9 / 9) | **+0.61** (9 / 9) | +1.15 |
| 5 s | +0.47 (9 / 9) | +0.28 | +0.54 (9 / 9) | +0.57 (9 / 9) | +2.36 |

## What it means

1. **"Will a trade take my level" is only part of why they pull.**
   - Trades that go through their price are 4% of the next trades there; they are gone before 37-47% of them.
   - Most of what they avoid is a trade that stops at their price and is followed by the price moving against them within about a second:
     they are gone before 40% of those against 22.5% of the ones that win.
   - The trade's kind accounts for 6% of the value gap between the fills they keep and those they skip.
2. **Their behaviour lines up with a short-horizon fill-value target:** "would a fill at my price lose within about a second", counting
   both the gains and the losses.
   - The 1 s outcome separates their pulls more than the 5 s one (1.77 vs 1.26).
   - The state before their pulls is the level's own (LEARN-PULL: P, a thin queue, their own side depleting).
3. **Our guard's target covers only part of that:** the loss when the level is swept within 0.5 s.
   - It gives no credit for good fills and does not label the "stops, then moves away" losses.
   - Its line is the same at the front and at the back of the queue.
4. **Our signals are not worse at picking.**
   - At their pull count, our guard value and P keep at least as much per trade as their actual pulls on 9 of 9 days (post hoc).
   - The difference is how often we pull (59% of the same intervals) and where our orders sit, not a better signal on their side.
   - Caveat: v and P are read 100 ms before the trade. Their pulls happen at an unknown time in the interval, possibly earlier. This favours our
     signals.
5. **For a model of our own:**
   - The label that matches their behaviour is the net value of a fill at our price over the next ~1 s at our real queue position (gains and
     losses), with the pull line where it turns negative.
   - We trained that kind of label at our replay queue position (keep-or-pull S4: Delta = Q_keep - Q_pull, 1 s). It had skill on 9 of 9 days,
     but the policy lost to the guard.
   - FRONT-PULL's front-tier line added only +$1.3 a day at rule M; with nothing ahead (rule U) the same policies earned +$171-246 a day.
   - Queue position first, as before.

## Limits

- Fills only: GONE includes moves to another price.
- Their own pull can make the level break (they are about a third of the volume), so an outcome after a pull is not independent of it.
- Only the population of front orders after small stopping trades; E2-ambiguous and same-stamp intervals excluded.
- Values are marks of one fill at 1 s / 5 s. Consumed days (C8). No replay of our orders (C7, C9).
- T2 per trade and T3 per trade are POST HOC (chosen after the declared volume-weighted run).

Outputs:
- `data/predict_fun/research/learn_pull_20261113/target/`: learn_pull_target.txt / .json, learn_pull_target_check.txt, logs.
- Copies in `reviews/learn_pull_20261113/target/`.
