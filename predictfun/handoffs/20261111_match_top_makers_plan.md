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

## Issues behind our queue position (2026-11-11, USER: "we note down these issues")

The churn fix (soft pull + incremental size, pmmmrust main 0c3b115) stops self-inflicted re-entries at the back. It does not move
us forward: every research replay still enters at the back (queue ahead = the size shown when our order goes live).

| Group | # | Issue | Evidence | Status |
|---|---|---|---|---|
| A. Entry | 1 | We join existing levels, behind everything already resting | Queue ahead when live: p50 ~360-400 sh at the v3 join (strategy review 6.3 / 6.5) | Inherent to joining the best |
| A. Entry | 2 | Lateness: ~100 ms from a book change to our order live (53 ms to see it + 44 ms to go live) | A new level grows from 52 to ~155 sh in 100 ms (228 in 200 ms), so we join behind 150+ even reacting at once (6.5) | Partly the venue's feed delay; not measured live |
| A. Entry | 3 | Entry only when the guard opens (top 10% of moments), after the queue formed | Born levels: 423 sh ahead at PM90 vs 229 unguarded (6.5) | By design |
| A. Entry | 4 | The MICRO rule quotes the heavy side, the longest queue | 72% of our fills on the heavy side at median reach 0.93 (B12e / B12f) | By design |
| B. Holding | 5 | Churn: cancel and re-place on guard flicker and size flips | Cancelled a median 23-38 ms after going live, 82.5% re-placed at the same price (B13c / B13d) | Fixed in code; not live; not in the research engine |
| B. Holding | 6 | Absent most of the time; each return after a guard pull joins at the back | We rest ~11% of side-time (6.3) | By design |
| B. Holding | 7 | Re-pricing: when the best moves we cancel and re-place at the new best, behind its resting queue | Every move of the best resets our place | By design |
| B. Holding | 8 | We do not stay after a fill; the top makers do | Re-hit after an absorbed fill: top 5 59.5% vs others 38.4% (B12d) | Not addressed |
| C. Position | 9 | Only one rung (1 tick behind), pulled by the guard; no deeper layers | 41% of the top 5's volume fills behind the best (+0.24 at 1 tick, +1.00 c/sh at 2+; B12) | Rung test MIXED (6.5); deeper untested |
| D. Speed | 10 | We see trades and book changes late, outside the window where the top makers earn | Their edge is earned within 250 ms of the report (B12c) | Same latency as #2 |
| D. Speed | 11 | Slow venue cancels late in the first minute | 11 fills inside cancels of 0.5-5 s = our 11-08 first-minute loss (B12b) | Live-session issue |
| E. Measurement | 12 | The fill clock (match vs report time) is unknown | It flips the sign of every layer arm (+0.28 at 50 ms, -3.15 at 250 ms; 6.5) | Needs B1 (live fills with A4b) |
| E. Measurement | 13 | The replay's queue / fill model always enters at the shown size and is too generous | Replay fills ~5x the tape's value (B12g); misses pressure losses (B18) | Known limit |

Even at zero latency, 1 + 3 + 6 put us behind everyone already resting. Latency (2) matters at new levels, about half the at-best
volume. Next measurement (proposed): the order-level replay with one change at a time - zero latency, always present without guard
pulls, no re-pricing, a rung resting continuously 1 and 2 ticks behind - reporting the queue ahead at fill and the value at all
three trade-time shifts.
