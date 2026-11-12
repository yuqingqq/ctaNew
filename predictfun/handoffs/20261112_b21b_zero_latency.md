# B21b: B21's arms with our own latency at zero, the optimistic bound (2026-11-12)

Session ce68ebe0. USER: "Yea we can, so that will be the optimistic result right?". Script `research/pf_b21b_zero_latency.py`
(3b3e85c2, committed before its first run): B21's simulator with a second world. J-v3, H0-g90 and the eight Z arms run a second
time (named @0) with:
- the order live at its decision step (replay +40 ms);
- a cancel effective at once (+22 ms, or 95 ms after the create while it is pending);
- our own fill known at once (+100 ms).

The information is unchanged: Predict's book ~55 ms after the venue stamp at the median, Binance ~3.5 ms, the 50 ms grid. Showing us
the book earlier than its trades' report times would let the replay see a trade's effect before the trade.

Output as of 2026-11-12T02:56Z, run in the claude slot (core 9) beside a Codex job. 11-08 consumed cohort, 220 windows, first 60 s.
Point estimates.

Controls passed:
- B21's controls, all of B20's included.
- The real-latency arms reproduce the committed B21 table (770 of 770 cells), and this report's real cells equal it.
- Three synthetic cases flip between the worlds as declared:
  - a pull 10 ms before a sell: filled while the cancel is pending / not filled;
  - a sell 20 ms after the decision: not live yet / filled;
  - a second sell 20 ms after a full fill: no order yet / filled.
- The latencies are restored afterwards.

## Answer: zero latency does not rescue resting by default; it helps the gated v3

- **Every Z arm still loses at every fill clock.**
  - R230 NEGATIVE for all eight; R232 0 of 6 against v3 at the same zero latency; R233 NOT-RESCUED.
  - At 1 s and the 50 ms clock: -0.46 (guard) to -1.81 (never pulled) c/sh.
- **An instant cancel barely changes the signal arms**: -0.14 to +0.22 c/sh at the 0 and 50 ms clocks.
  - When our signals fire, the trades that hurt a resting order have mostly happened, or come anyway. The signals are late or not
    informative relative to those trades; our order path is not what makes them late.
  - The remaining information delay (the feed and the 50 ms grid) is part of "late". It cannot be removed in this replay without
    the match-time clock (B1).
- **For a resting order, being faster is worse.**
  - Instant activation and instant re-placement after a fill bring more of the same adverse fills. Never pulled: 6,143 -> 7,402 fills,
    -1.50 -> -1.81 c/sh at 1 s.
  - At the 250 ms clock every Z arm is worse at zero latency: -0.28 to -0.58 c/sh. R231 is SPEED-HURTS for none, bshift and events,
    and MIXED for the rest.
- **For the gated strategy, being faster helps at every fill clock (R231 SPEED-HELPS for v3 and the guarded hold).**
  - v3's c/sh at 1 s: +0.31 -> +0.57 (0 ms clock), -0.05 -> +0.23 (50 ms), -1.75 -> -1.53 (250 ms); +0.22..+0.27 c/sh in all.
  - $/day at 1 s / 5 s at the 50 ms clock: -0.58 / +6.23 -> +3.53 / +13.33, from 274 to 339 fills.
  - At 250 ms v3 still loses. The fill clock decides the sign.

## Table (rule M, trade-time shift 50 ms; replay latency -> @0)

| Arm | life p50 ms | rest % | ahead when live p50 | place at fill | fills | c/sh 1 s | c/sh 5 s | $/day 1 s | $/day 5 s |
|---|---|---|---|---|---|---|---|---|---|
| J-v3 | 82 -> 100 | 11.2 -> 12.4 | 363 -> 348 | 0.36 -> 0.34 | 274 -> 339 | -0.05 -> +0.23 | +0.50 -> +0.85 | -0.58 -> +3.53 | +6.23 -> +13.33 |
| H0-g90 | 82 -> 100 | 11.4 -> 12.4 | 355 -> 347 | 0.32 -> 0.28 | 206 -> 252 | -0.29 -> +0.13 | +0.31 -> +0.77 | -2.84 -> +1.53 | +3.00 -> +8.91 |
| Z-none | 1,032 -> 500 | 97.1 -> 99.3 | 129 -> 102 | 0.16 -> 0.20 | 6,143 -> 7,402 | -1.50 -> -1.81 | -1.87 -> -2.25 | -409.17 -> -597.69 | -510.34 -> -745.18 |
| Z-thinning | 382 -> 350 | 58.2 -> 60.1 | 189 -> 185 | 0.22 -> 0.23 | 2,114 -> 2,244 | -1.55 -> -1.53 | -1.87 -> -1.76 | -151.63 -> -160.20 | -182.97 -> -183.84 |
| Z-side | 582 -> 566 | 55.7 -> 56.7 | 234 -> 232 | 0.25 -> 0.26 | 1,432 -> 1,483 | -1.47 -> -1.41 | -1.70 -> -1.64 | -98.02 -> -98.62 | -113.43 -> -114.77 |
| Z-burst | 926 -> 622 | 88.6 -> 90.7 | 162 -> 140 | 0.16 -> 0.18 | 4,626 -> 5,179 | -0.90 -> -1.00 | -1.25 -> -1.44 | -190.88 -> -238.90 | -263.71 -> -342.99 |
| Z-bshift | 832 -> 550 | 92.3 -> 94.4 | 146 -> 123 | 0.15 -> 0.18 | 4,822 -> 5,644 | -0.95 -> -1.09 | -1.35 -> -1.59 | -206.68 -> -278.37 | -293.87 -> -405.08 |
| Z-guard | 582 -> 550 | 63.3 -> 64.5 | 226 -> 216 | 0.17 -> 0.18 | 1,681 -> 1,803 | -0.51 -> -0.46 | -0.70 -> -0.64 | -40.96 -> -40.08 | -56.27 -> -55.42 |
| Z-events | 382 -> 352 | 39.0 -> 40.2 | 262 -> 262 | 0.26 -> 0.25 | 611 -> 631 | -0.83 -> -0.88 | -1.01 -> -1.18 | -24.90 -> -27.52 | -30.44 -> -36.99 |
| Z-all | 382 -> 350 | 33.9 -> 34.9 | 279 -> 279 | 0.25 -> 0.24 | 458 -> 460 | -0.78 -> -0.77 | -0.83 -> -0.96 | -17.70 -> -17.69 | -18.82 -> -22.11 |

c/sh at 1 s by fill clock (replay latency -> @0):

| Arm | M 0 | M 50 | M 250 | U 50 (front) |
|---|---|---|---|---|
| J-v3 | +0.31 -> +0.57 | -0.05 -> +0.23 | -1.75 -> -1.53 | +0.90 -> +0.95 |
| H0-g90 | +0.08 -> +0.28 | -0.29 -> +0.13 | -2.04 -> -1.88 | +0.86 -> +0.92 |
| Z-none | -1.39 -> -1.49 | -1.50 -> -1.81 | -1.92 -> -2.50 | -0.35 -> -0.60 |
| Z-thinning | -1.13 -> -1.10 | -1.55 -> -1.53 | -2.22 -> -2.59 | +0.22 -> +0.20 |
| Z-side | -1.01 -> -0.79 | -1.47 -> -1.41 | -2.16 -> -2.73 | +0.34 -> +0.34 |
| Z-burst | -0.81 -> -0.81 | -0.90 -> -1.00 | -1.67 -> -2.03 | +0.00 -> -0.09 |
| Z-bshift | -0.81 -> -0.87 | -0.95 -> -1.09 | -1.67 -> -2.06 | -0.04 -> -0.17 |
| Z-guard | -0.35 -> -0.26 | -0.51 -> -0.46 | -1.54 -> -1.91 | +0.39 -> +0.40 |
| Z-events | -0.67 -> -0.68 | -0.83 -> -0.88 | -2.31 -> -2.59 | +0.56 -> +0.54 |
| Z-all | -0.62 -> -0.62 | -0.78 -> -0.77 | -2.23 -> -2.52 | +0.58 -> +0.56 |

$/day at 1 s / 5 s at zero latency, M 0 / M 50 / M 250:
- J-v3@0: +9.78 / +18.31, +3.53 / +13.33, -29.37 / -20.95.
- Z-all@0, the least bad Z arm: -12.76 / -14.33, -17.69 / -22.11, -126.21 / -127.23.

## Reading and limits

- Zero latency is not reachable. It bounds what a faster order path could give, with the information we receive.
- The feed delay (~55 ms median for Predict's book) and the 50 ms grid stay. A co-located maker sees the book sooner. Whether that
  is where the top makers' edge comes from (B12c: earned within 250 ms of the report) cannot be tested here without the match-time
  clock.
- Same limits as B21:
  - one consumed partial day, first minute only;
  - one 5-sh order per lane;
  - Binance's hard guards, CAP10 / RO60 and sizes are not modelled;
  - rule M is not validated (B1a).

## What follows

- Resting by default with these five signals is closed at the replay level (B21, B21b). Neither speed nor these signals turn it
  positive. It would need a pull signal that sees the harmful trade coming before it shows in our book or Binance data.
- For the gated strategy, the order path is worth about +0.25 c/sh at every fill clock on this day. That is a reason to keep the
  execution path fast, and it is the place where speed pays.
- The live run with the A4b capture measures what replays cannot: our real place / cancel latencies, the fill clock and our real
  queue place.

Outputs `data/predict_fun/research/b21b_20261112/` (b21b.json / .txt, b21.json / .txt, b20.json / .txt, level_birth_b21b.json,
run.log); copies `reviews/b21b_20261112/`.
