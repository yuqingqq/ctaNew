# QR, the queue race: why our orders sit at the back, and how fast we would need to be (plan, 2026-11-12)

Session ce68ebe0. USER: "it seems we quote at bad timings, not sure if it is because we requote slower than others, or we requote
and we lose the position. And yes, if we lose the latency game, we need to check how fast we need to be".
Script `research/pf_qr_queue_race.py`, driver `research/run_qr.sh`. Both are committed before the first run. The readings
(R280-R286), controls and limits are in the script's docstring, evaluated in code. Exploratory, on consumed days only.

## A. Where the queue ahead of our order comes from (9 replay days; v3, rest with the guard, rest never pulled)

Every order that is at the best when it goes live gets its queue ahead split into parts that add up to it:

| Part | Meaning | Answers |
|---|---|---|
| U | already at the level when it became the best | no speed beats this (a level that falls back brings the queue it had as second) |
| LAT | joined in the first 97 ms (our median delay: 53 ms feed + 44 ms) | "slower than others" |
| LATE | joined after we could have been there, before our first order went live | we came later than our speed allows (guard, grid, decisions) |
| REQ | joined between our first order and this one at the same price | "we requote and lose the position" |
| RAISE | the simulator's allowance for the next snapshot | measurement |

Each part comes from a ghost order placed at that moment and carried with the simulator's queue rule. R280 names the largest.

- Fills are also split by how they reached the front: near the front at activation (< 20 sh ahead), a sweep through our price,
  queue removed mostly by others cancelling, or queue mostly traded through. R282 tests whether arriving early pays; R283 whether
  being at the front because others left is worse.

## B. How fast (9 replay days)

- At every new best level (an improvement on the venue's clock), our order goes live d ms after the level's birth stamp.
  - d = 0 (bound), 15 (the venue's smallest feed delay alone), 30, 55 (that feed delay plus our 40 ms order path), 97 (our median), 150.
  - Without a guard and with v3's guard. Exits as today.
- References:
  - lane B (today's clock), with its realised delay;
  - random-time joins at 30 ms (speed without the birth);
  - v3.
- Readings:
  - R284: how much queue speed buys;
  - R285: whether 30 ms beats 97 ms;
  - R286: the slowest speed at which joining new levels is positive, and the slowest at which it beats v3.
- Values at the 0 / 50 / 250 ms fill clocks, rule L and the front bound U.
- Sensitivity: the same without the next-snapshot raise. The raise counts later joiners as ahead, which weighs most on fast
  entries.

## C. The live engine (11-08, B13c's data)

- Part A's split for every accepted live order at the best (R281).
- Plus how long after a level's birth our first order was accepted.

## Controls and cost

- The copied simulator equals S4's order for order on every window-side.
- Lane B's own schedule through the new lane equals lane B.
- The regression can fail (cancel latency 30 ms).
- Each ghost equals the simulator's queue at activation and at every fill.
- The removed parts add up exactly.
- Synthetic cases with known answers; four planted errors, each caught.
- About 1-1.5 h in the claude slot.

## Limits

- Consumed days: a pass makes a candidate, not a result.
- Rule M is not validated (B1a).
- One 5-sh order per lane, and our order is not in the public book.
- Part B speeds up only the entry; exits and the guard keep today's clock.
- Snapshot resolution.
- Part C is one day.
