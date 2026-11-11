# B20: pre-positioned, held orders at the order level (2026-11-11)

Session ce68ebe0. USER: "Can we somehow post in advance to be in a good position ... put our orders there until our guard says to
withdraw"; "Test this, check if we solved the position issue somehow". Script `research/pf_b20_hold_layers.py` (2f6283f6, committed
before its first run): the strategy review 6.5 lane simulator, plus three hold lanes and our queue place recorded at every fill.
Output as of 2026-11-11T16:06Z. 11-08 consumed cohort, 220 windows, first 60 s, one 5-share order per lane and side. Point estimates.

Controls passed:
- all 80 stored 6.5 arm-config cells reproduced (relative 1e-9);
- the original self-test, its planted errors and its regressions;
- six synthetic hold cases, plus a falsifier: a hold lane that secretly re-prices must fail.

## Answer: not solved

- **With the guard, held orders still live ~82 ms.** That is the same as the v3 join (life p50 82 ms, resting 11.4% of the time).
  The guard withdraws them almost at once, so they never build time priority.
  - They also enter behind the queue already at their price: 355 sh (touch), 288 (1 tick behind), 304 (2 behind), against 363 for
    v3.
- **Without the guard they rest long and fill nearer the front, and lose.**
  - Life p50 is 1.0-1.3 s and they rest 97-98% of the time. At the fill only 16-17% of the level is ahead of them on average (v3:
    36%).
  - But those fills lose 1.39-1.94 c/sh at 1 s: the fills that reach an order which simply rests are adverse.
  - A looser guard (PM50) sits in between: it rests 61% of the time and loses 0.14 to 0.49 c/sh.
- **Value is MIXED for every guarded arm, and the fill clock decides it.**
  - One tick behind (H1-g90): +0.58 / +0.19 / -3.16 c/sh at 1 s for shifts 0 / 50 / 250 ms, the same as v3's rung (R-g90).
  - Two ticks behind (H2-g90): +1.00 / +1.23 / -4.33 on 33 fills, 126 sh, too small to rely on.
  - No arm beats v3 in all 6 $/day cells: H1-g90 and H2-g90 beat it in 3 of 6, H0-g90 in 1 of 6.
- **At the front of the queue (rule U, the upper bound) the guarded arms earn +0.75 to +1.32 c/sh at 1 s.** The edge exists only
  where we cannot get. Reaching the front needs presence, and presence through weaker guard states brings the adverse fills (as 6.5's
  R17, "residence costs").

Readings: R210 NOT-IMPROVED for all six guarded H arms; R211 NEGATIVE for the unguarded and PM50 arms (H2-g50 MIXED) and MIXED for
the PM90 arms; R212 0-3 of 6.

## Table (rule M, trade-time shift 50 ms)

| Arm | orders / side-min | life p50 ms | rest % | ahead when live p50 | place at fill mean (0 front) | sweep share | fills | c/sh 1 s / 5 s / 60 s | $/day 1 s / 5 s |
|---|---|---|---|---|---|---|---|---|---|
| J-v3 (live rule) | 38.35 | 82 | 11.2 | 362.5 | 0.36 | 0.59 | 274 | -0.05 / +0.50 / -1.17 | -0.58 / +6.23 |
| R-g90 (v3's rung) | 34.22 | 82 | 11.4 | 291.0 | 0.54 | 0.88 | 75 | +0.28 / +0.34 / +1.97 | +1.05 / +1.27 |
| H0-g90 (touch, held) | 32.60 | 82 | 11.4 | 354.7 | 0.32 | 0.56 | 206 | -0.29 / +0.31 / -0.28 | -2.84 / +3.00 |
| H1-g90 (1 behind, held) | 33.14 | 82 | 11.5 | 288.2 | 0.52 | 0.89 | 71 | +0.19 / +0.30 / +1.88 | +0.66 / +1.06 |
| H2-g90 (2 behind, held) | 34.18 | 82 | 11.4 | 304.3 | 0.44 | 0.71 | 33 | +1.23 / +1.26 / +6.83 | +2.03 / +2.08 |
| H0-g50 | 56.41 | 132 | 60.9 | 193.7 | 0.20 | 0.42 | 1,489 | -0.49 / -0.69 / -0.64 | -36.13 / -50.28 |
| H1-g50 | 56.45 | 132 | 61.3 | 226.8 | 0.29 | 0.59 | 437 | -0.48 / -0.46 / +1.26 | -10.95 / -10.47 |
| H2-g50 | 58.54 | 132 | 61.3 | 262.8 | 0.34 | 0.64 | 177 | -0.14 / -0.37 / +1.97 | -1.23 / -3.28 |
| H0-none | 20.70 | 1,032 | 97.1 | 129.0 | 0.16 | 0.50 | 6,143 | -1.50 / -1.87 / -1.66 | -409.17 / -510.34 |
| H1-none | 17.79 | 1,275 | 98.0 | 161.0 | 0.17 | 0.54 | 4,140 | -1.79 / -2.11 / -2.10 | -331.28 / -391.47 |
| H2-none | 17.98 | 1,132 | 98.4 | 187.0 | 0.17 | 0.54 | 3,130 | -1.94 / -2.30 / -2.30 | -271.72 / -321.89 |

c/sh at 1 s by config:

| Arm | M 0 | M 50 | M 250 | L 50 | U 50 |
|---|---|---|---|---|---|
| J-v3 | +0.31 | -0.05 | -1.75 | -0.06 | +0.90 |
| H0-g90 | +0.08 | -0.29 | -2.04 | -0.32 | +0.86 |
| H1-g90 | +0.58 | +0.19 | -3.16 | +0.19 | +0.75 |
| H2-g90 | +1.00 | +1.23 | -4.33 | +1.21 | +1.32 |
| H0-none | -1.39 | -1.50 | -1.92 | -1.85 | -0.35 |

## Reading and limits

- "Place at fill" (the share of the level ahead of us just before the trade) cannot tell an early arrival from a level that others
  just left (the thinning fills of B12d / B18). The cleaner position signs are the order's life and the queue ahead when it went
  live: with the guard, neither moved.
- Hold lanes are separate single orders; a real ladder's inventory and caps are not combined.
- One consumed partial day, first minute only. The fill rule is not validated (B1a). The fill clock decides every guarded arm's sign
  (+ at 0 / 50 ms, - at 250 ms).

## What follows

- Holding at our guard's rhythm cannot give us time priority. The guard is on in short bursts (~11% of the time) and withdraws
  within ~80 ms.
- Resting long enough to reach the front brings fills we lose on (-1.4 to -1.9 c/sh). The top makers survive those same fills at
  break-even (B18), which takes speed and selection we do not have at ~100 ms.
- The two unknowns that decide what to do next both need the live run with the A4b wallet capture:
  - the fill clock (B1: which of the 0 / 50 / 250 ms worlds is real; at 0-50 ms the held 1-2 tick layers are positive);
  - our real queue place when we stop churning.

Outputs `data/predict_fun/research/b20_20261111/` (b20.json / .txt, level_birth_b20.json, run.log); copies `reviews/b20_20261111/`.
