# B18: fills in pressure moments, top makers against our strategy, 9 days (2026-11-11)

Session ce68ebe0. USER: "can we check on other days as well ... do we now pull orders when top makers do in the cases like imbalance
happens, or flow bursts? Or we still have those bad fills". Script `research/pf_b18_pressure_fills.py` (c28f99d5, committed before its
first run). Output as of 2026-11-11T14:07Z. Descriptive. Readings: R180 SIMILAR, R181 SAME, R182 NOT-WORSE.

## Design (in the script's header)

- Pressure classes, from the maker's side, read at the fill time from the latest 50 ms row (live clock), fixed before the run:
  - thin: our side's best level at most half the other's;
  - thinning: our best level shrank 20% or more in the last 250 ms;
  - burst: Binance trades in the last 1 s at least 2x their 30 s mean, with net flow against our side;
  - bnshift: Binance's best bid / ask imbalance moved 0.3 or more against our side in 250 ms;
  - any: at least one of them.
- Base rate (share of live moments, both sides, mean over the days): thin 33%, thinning 24%, burst 7.9%, bnshift 4.7%, any 47.6%.
- Fills compared:
  - every timed maker leg on the public tape, 0-240 s, by the day's rank;
  - the v3 arm's fills in the replays with the live guard (REF, B14's b14rJ / X) and with B16's qb guard.

## Results (9 days pooled; share of filled shares in the class; c/sh at 1 s / 60 s in the class, and out of it)

| Makers | any | value in any | value outside | thin | thinning | burst | bnshift |
|---|---|---|---|---|---|---|---|
| top 5 (tape) | 62.9% | +0.04 / -0.15 | +0.76 / +0.63 | 43.1% | 36.6% | 20.1% | 11.4% |
| ranks 6-20 (tape) | 53.1% | -0.09 / -0.26 | +0.68 / +0.75 | 32.7% | 30.7% | 15.6% | 8.9% |
| rest (tape) | 59.3% | -0.75 / -0.95 | +0.43 / +0.44 | 38.2% | 31.9% | 22.1% | 13.2% |
| REF, v3 replay J | 61.0% | +1.57 / +1.51 | +1.38 / +1.65 | 13.1% | 49.6% | 12.5% | 13.7% |
| qb, v3 replay J | 63.4% | +1.57 / +1.48 | +1.56 / +0.92 | 14.2% | 51.8% | 14.3% | 12.9% |

Value in each class, c/sh at 1 s / 60 s:
- top 5: burst -0.47 / -0.70, bnshift -0.65 / -0.66.
- rest: burst -1.87 / -1.87, bnshift -2.04 / -2.20.

Per day, the share of filled shares in 'any' for top 5 / REF / qb:

| Day | top 5 | REF | qb |
|---|---|---|---|
| 10-23 | 0.614 | 0.658 | 0.641 |
| 10-24 | 0.588 | 0.563 | 0.610 |
| 10-25 | 0.574 | 0.601 | 0.644 |
| 10-26 | 0.633 | 0.732 | 0.680 |
| 10-27 | 0.601 | 0.675 | 0.671 |
| 10-31 | 0.639 | 0.450 | 0.531 |
| 11-01 | 0.619 | 0.483 | 0.542 |
| 11-02 | 0.680 | 0.597 | 0.593 |
| 11-03 | 0.706 | 0.611 | 0.701 |

## Reading

- **The top makers do not avoid pressure moments.**
  - 63% of their filled shares come in them, against 48% of the time.
  - Their pressure fills roughly break even: +0.04 at 1 s, -0.15 at 60 s.
  - Their fills outside pressure earn +0.76 / +0.63. That is where their edge is.
  - The losing makers lose heavily in pressure (-0.75 / -0.95; bursts -1.87). So the top makers' edge is being hurt little when
    pressure hits, not pulling more. This agrees with B13b on 11-08.
- **Our strategy in the replay is filled in pressure moments as often as the top makers.** REF 61% against the top 5's 63% (R180
  SIMILAR). The qb guard does not change that (63%, R181 SAME).
  - The mix differs. v3's imbalance (MICRO) rule keeps us off the thin side: 13% of our fills, against the top 5's 43%.
  - But 50% of our fills come while our level is thinning, against their 37%: others leave, then we are filled (B12d's 11-08
    pattern, now on 9 days).
- **The replay cannot value those fills.** It scores our pressure fills about as well as the rest (+1.57 / +1.51 against +1.38 /
  +1.65; R182 NOT-WORSE), while real makers' pressure fills are 0 to -2 c/sh.
  - Its fill model does not reproduce the adverse selection of pressure moments, so a replay cannot reward pulling there.
  - That limits B14 / B16 and any "pull on pressure" test run through it.
  - Our real 11-08 fills after thinning lost 5.83 c/sh (B12d).

## Limits

- Tape fills are whole-market, not orders we could place.
- The replay is our simulated behaviour under an unvalidated fill model.
- The thresholds are fixed choices, not fitted.
- The 9 days were read before.

Outputs `data/predict_fun/research/b18_20261111/` (b18.json / .txt); copies `reviews/b18_20261111/`.
