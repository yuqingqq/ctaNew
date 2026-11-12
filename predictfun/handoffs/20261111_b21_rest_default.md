# B21: rest at the touch by default, pull only on a signal (2026-11-11, run 11-12)

Session ce68ebe0. USER: "So we need to build something to rest orders there as default, only pull when real signals happen".
Script `research/pf_b21_rest_default.py` (8391a783, committed before its first run): B20's order-level simulator plus a rest lane.
Output as of 2026-11-12T02:25Z. 11-08 consumed cohort, 220 windows, first 60 s, one 5-share order per lane and side. Point estimates.

Design (in the script's header):
- Rest lane: a 5-sh order at the known best whenever the book allows, with no guard gate. Held: not re-priced while the best is
  within 3 ticks. Cancelled when a signal fires (or when > 3 ticks behind, or crossing). Re-placed only after the signal has been
  clear for 250 ms.
- Signals, read every 50 ms from the live-clock rows; B18's thresholds, fixed before the run:
  - thinning: our best level <= 0.8x its size 250 ms earlier;
  - thin side: our side's imbalance <= -1/3;
  - burst: a Binance trade burst with net flow against us;
  - bshift: Binance's best bid / ask imbalance moved >= 0.3 against us in 250 ms;
  - guard: the live guard's value v <= -0.10 c.
- Arms: Z-none (never pulled by a signal), each signal alone, Z-events (the four event signals), Z-all (all five).

Controls passed:
- B20's controls: all 80 stored 6.5 cells; the self-test with its planted errors; the regressions; the hold cases and their
  falsifier. B20's table is reproduced byte for byte.
- Synthetic rest cases: placed without a guard; pulled at a signal; re-placed only after 250 ms clear; held when the best moves one
  tick. Their falsifier passed: with no cooldown, the re-entry case breaks.
- Every one of the 264,000 grid steps found its live row.
- Z-none equals B20's unguarded touch hold (H0-none) in every cell, as it must.

## Answer: resting by default fixes presence, not the losses

- **The orders stay, and fill nearer the front.**
  - Never pulled (Z-none):
    - orders live a median 1.0 s and rest 97% of the time (v3: 82 ms, 11%);
    - they join behind 129 sh (v3 363);
    - at the fill, 16% of the level is ahead of them on average (v3 36%).
  - The burst, bshift, guard and thin-side signals keep them resting 56-92% of the time with a life of 0.6-0.9 s (R222 PRESENT).
  - The thinning signal fires so often that orders live 382 ms. They churn more than v3: 48.6 orders per side-minute against 38.4.
    Thinning, events and all are ABSENT.
- **Every arm loses, at every fill clock (R220 NEGATIVE for all eight).**
  - At 1 s and the 50 ms clock, the arms lose 0.51 (guard) to 1.55 (thinning) c/sh. v3 is at -0.05 (+0.31 at 0 ms).
  - No arm beats v3 in any of the 6 $/day cells (R221 0 of 6).
  - Per share, the guard signal alone loses least (-0.51 c/sh).
  - Per day, all five signals together lose least: -17.70 $/day against -0.58 for v3. They keep the order out 66% of the time and
    trade least.
- **The signals help, but none is enough.**
  - Gains in c/sh at 1 s against never pulling, at the 0 / 50 / 250 ms clocks (R223 HELPS):
    - guard: +1.04 / +0.99 / +0.37;
    - Binance burst: +0.58 / +0.60 / +0.25;
    - Binance book shift: +0.58 / +0.55 / +0.25.
  - Our level thinning and our side turning thin help only at the 0 ms clock and hurt at 250 ms (R223 MIXED). Both are visible in
    our book only once the move has started. That is a reading, not a measurement.
  - Combining the signals (events, all) gains +0.67..+0.77 at 0-50 ms but loses 0.32-0.40 at 250 ms.
- **Even at the front of the queue, resting by default earns less than v3's gated quotes.**
  - Rule U (nothing ahead of us; the upper bound): never pulled -0.35 c/sh at 1 s, all five signals +0.58, v3 +0.90.
  - So queue position does not explain all of our losses. An order that simply rests collects the trades that v3's gate avoids by
    being absent.
  - The top makers earn +0.24 c/sh at the best (11-08, B12d), so they avoid those trades in time. Our signals, at our latency, do not.

## Table (rule M, trade-time shift 50 ms)

| Arm | orders / side-min | life p50 ms | rest % | ahead when live p50 | place at fill mean (0 front) | sweep share | fills | c/sh 1 s / 5 s / 60 s | $/day 1 s / 5 s |
|---|---|---|---|---|---|---|---|---|---|
| J-v3 (live rule) | 38.35 | 82 | 11.2 | 362.5 | 0.36 | 0.59 | 274 | -0.05 / +0.50 / -1.17 | -0.58 / +6.23 |
| H0-g90 (B20 touch hold) | 32.60 | 82 | 11.4 | 354.7 | 0.32 | 0.56 | 206 | -0.29 / +0.31 / -0.28 | -2.84 / +3.00 |
| Z-none | 20.70 | 1,032 | 97.1 | 129.0 | 0.16 | 0.50 | 6,143 | -1.50 / -1.87 / -1.66 | -409.17 / -510.34 |
| Z-thinning | 48.56 | 382 | 58.2 | 189.0 | 0.22 | 0.61 | 2,114 | -1.55 / -1.87 / -1.47 | -151.63 / -182.97 |
| Z-side | 25.80 | 582 | 55.7 | 234.0 | 0.25 | 0.56 | 1,432 | -1.47 / -1.70 / -0.83 | -98.02 / -113.43 |
| Z-burst | 25.81 | 926 | 88.6 | 162.3 | 0.16 | 0.45 | 4,626 | -0.90 / -1.25 / -0.75 | -190.88 / -263.71 |
| Z-bshift | 23.05 | 832 | 92.3 | 146.0 | 0.15 | 0.44 | 4,822 | -0.95 / -1.35 / -1.16 | -206.68 / -293.87 |
| Z-guard | 23.21 | 582 | 63.3 | 226.0 | 0.17 | 0.38 | 1,681 | -0.51 / -0.70 / -0.06 | -40.96 / -56.27 |
| Z-events | 33.92 | 382 | 39.0 | 261.7 | 0.26 | 0.53 | 611 | -0.83 / -1.01 / -0.89 | -24.90 / -30.44 |
| Z-all | 29.73 | 382 | 33.9 | 279.1 | 0.25 | 0.51 | 458 | -0.78 / -0.83 / +0.01 | -17.70 / -18.82 |

c/sh at 1 s by config, and each signal's gain against Z-none:

| Arm | M 0 | M 50 | M 250 | L 50 | U 50 (front) | gain vs Z-none, M 0 / 50 / 250 |
|---|---|---|---|---|---|---|
| J-v3 | +0.31 | -0.05 | -1.75 | -0.06 | +0.90 | |
| Z-none | -1.39 | -1.50 | -1.92 | -1.85 | -0.35 | |
| Z-thinning | -1.13 | -1.55 | -2.22 | -1.60 | +0.22 | +0.26 / -0.05 / -0.30 |
| Z-side | -1.01 | -1.47 | -2.16 | -1.67 | +0.34 | +0.38 / +0.03 / -0.24 |
| Z-burst | -0.81 | -0.90 | -1.67 | -1.16 | +0.00 | +0.58 / +0.60 / +0.25 |
| Z-bshift | -0.81 | -0.95 | -1.67 | -1.20 | -0.04 | +0.58 / +0.55 / +0.25 |
| Z-guard | -0.35 | -0.51 | -1.54 | -0.70 | +0.39 | +1.04 / +0.99 / +0.37 |
| Z-events | -0.67 | -0.83 | -2.31 | -0.91 | +0.56 | +0.72 / +0.67 / -0.40 |
| Z-all | -0.62 | -0.78 | -2.23 | -0.84 | +0.58 | +0.77 / +0.72 / -0.32 |

Readings:
- R220: NEGATIVE for all eight Z arms (every 1 s and 5 s cell is below zero at 0 / 50 / 250 ms).
- R221: 0 of 6 for every arm.
- R222: PRESENT for none, side, burst, bshift and guard; ABSENT for thinning, events and all (life p50 382 ms).
- R223: HELPS for burst, bshift and guard; MIXED for thinning and side.

## Reading and limits

- The fill clock sets the size of the loss, not its sign. Every Z arm is negative at 0, 50 and 250 ms.
- Rule M credits every shrink ahead of us. When others leave a level, it moves us to the front, which is where a sweep arrives
  first. "Place at fill" cannot tell an early arrival from a level that others just left.
- The signal thresholds are B18's fixed choices, not tuned. A tuned signal would need fresh days.
- One consumed partial day, first minute only.
- One 5-sh order per lane. Binance's hard guards (G1 / G2), CAP10 / RO60 and sizes are not modelled.
- The fill rule is not validated (B1a).

## What follows

- Not a candidate: resting by default with these five signals loses at every clock.
- The open question is whether the signals are too late or not informative.
  - Proposed B21b (not run): the same arms with each signal taken at the time its event happens on the venue and an instant cancel.
    That is the zero-latency bound.
  - If resting turns positive there, speed is the lever. B12c found the top makers' edge is earned within 250 ms of the report.
  - If it does not, we need a better pull signal: a model of the next adverse fill for an order that is resting, trained at that
    order's queue place rather than at the back.
- The live run with the A4b capture still decides the fill clock and our real queue place.

Outputs `data/predict_fun/research/b21_20261111/` (b21.json / .txt, b20.json / .txt, level_birth_b21.json, run.log); copies
`reviews/b21_20261111/`.
