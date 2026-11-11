# Plan: match the top makers at the best (2026-11-11)

Session ce68ebe0. USER: "we want to match the top makers, at least we can pull when they pull to avoid adverse fills like
they do and reduce redundant pulls to get good fills". This plan is built on B12-B15 (one consumed partial day for the maker
comparisons; replay fill rule not validated). Every step below is a declared test or a live measurement, not a promise.

## Where we stand (11-08, first minute, net 1 s c/sh on at-best fills; B12d)

| Makers | net 1 s | level holds 250 ms after the fill | fills after the level thinned >= 20% |
|---|---|---|---|
| top 5 | +0.24 | 81.6% | 17.2% of fills, at -0.31 |
| ranks 6-20 | +0.26 | 81.9% | (ranks 6+) 19.6%, at -1.04 |
| ranks 21+ | -0.20 | 72.9% | |
| ours | -1.34 | 42.2% | 21.4%, at -5.83 |

- An ordinary maker at the best (ranks 6-20) earns what the top 5 earn there. Matching the field is the first goal.
- Why we lose:
  - We sit at the back of the queue. Churn re-places our orders at the back: they are cancelled a median 23-38 ms after
    going live, and 82.5% are re-placed at the same price.
  - We fill at a median queue reach of 0.93 on heavy levels; the field fills at 0.20.
  - 39% of our shares come from trades through the level; the field's figure is 14.5%.
  - Slow venue cancels at 50-60 s caused our first-minute loss.
- "Pull when they pull": their cancels are anonymous on the public tape, so we cannot copy their pulls directly. What we
  can observe:
  - The level thinning: its size falling without trades means others are leaving. Our fills after a 20% thinning lost
    5.83 c/sh.
  - The opposite side thinning.
  - Binance's best bid / ask imbalance shifting while its price is still (B12f).
  - Bursts of Predict trades.
- B13b found their fill share does not fall under our pressure signal: their protection is mostly being at the front of
  the queue (small benign trades fill them first), not pulling more.

## Steps

| # | Step | Owner | Measures | Status |
|---|---|---|---|---|
| 1 | Live run with the churn fix: current guard + soft pull (0.01 c, no cap) + incremental size + the A4b wallet capture | Tokyo session; live = USER (D2) | requests per window; cancels < 100 ms after going live; queue reach at fill; share of fills from trades through the level; markouts by order age; soft_ended mix | ready on pmmmrust main 0c3b115 (no tag) |
| 2 | B16 second-layer decision model: at v3's own decision times, score keep / post against pull for our order at its queue place | research (this session) | flip score against the baseline's decisions (one decision changed at a time), then the full v3 replay; benchmarks: the current guard and B14 | starting now |
| 3 | B17 G4 hysteresis: keep a resting touch order while the best-level size ratio stays within a band around 50% (the MICRO imbalance rule caused 14.7% of touch pulls) | research | B15's order-level replay: orders, fills, value against v3 and the soft pull | after B16's first table |
| 4 | B1: calibrate the fill rule on our own live fills (match clocks from A4b), then relabel B16 with it | research | live vs replay fills per order | needs step 1's data |
| 5 | Freeze the best candidate (commit, receipt, the number of candidates in the race), forward test on >= 5 untouched days in shadow | research | the declared rule, day-clustered | after 2-4 |
| 6 | Port to pmmmrust behind a config flag (off by default), with parity tests; the Tokyo session deploys | port + Tokyo | parity, live A/B | after 5; live = USER |
| later | Stay after being hit (the top 5 re-hit 59.5% against 38.4%); taking the thin side (a separate strategy, taker fees) | research | | not started |

## B16 design (to be declared in its script before the first run)

- Baseline: the frozen v3 engine with the current guard, through a generated copy that logs every decision (time, side,
  action, our order's state). Its orders, fills and P&L are checked identical to the v3 engine.
- Label for each touch decision: the value of keeping or posting our order over the interval to the next decision,
  against 0 for pulling.
  - The order sits at its queue place: the size ahead when it went live, reduced by trades at its price only; no credit
    for cancels ahead (conservative).
  - It fills when trades at its price use up the size ahead, or when the level is traded through.
  - Its value is the shares filled times the markout at 1 s (and 5 s).
- Inputs:
  - the old guard value v (so a model that learns nothing agrees with the old decision);
  - the 32 current features and the 13 pressure features;
  - our order's state: queue ahead, age;
  - the level's state: heavy or thin side, level age, thinning (size change without trades over 100 / 250 / 500 ms on
    both sides).
- Training and scoring: leave one day out on the 9 consumed days, with 11-03 out of sample.
  - At each baseline decision the model can flip the action (keep to pull, or pull to keep).
  - Score per day: the sum of realised value (new action) minus value (old action), with the counts of flips each way and
    the change in quoting time.
- Benchmark arms at the same decisions: the current guard (the baseline itself), the B14 pressure guard, B16 with the 32
  features only (target effect) and B16 with all features (target + features).
- Falsifier: B16 refit on labels shuffled within each day must not beat the baseline.
- Limits: consumed days only, so exploratory; the label inherits the unvalidated fill rule; decisions are only at the
  baseline's decision times (a faster reaction between them is invisible here, and the full replay checks it).
