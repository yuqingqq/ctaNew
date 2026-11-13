# DEEP-PULL: the ladder with the pull at every rung (results, 2026-11-13)

Session ce68ebe0. After DEPTH-TIMING (a3e3317c: our deep rungs lose because half their fills come at high P). USER: "Check if it still
the timing issue" / "try to test and check if these next steps can bridge the gaps".

Script `research/pf_deep_pull.py` + `run_deep_pull.sh` (ffc7e475, committed before the first run). 07:49-08:32Z, claude slot.

## Setup

- FRONT-PULL-SIZE's simulator and signal: S4's rest lane, 9 consumed replay days, first 60 s.
- 30-sh orders kept through fills, 6 rungs per side (0-5 ticks behind the best).
- Every rung is pulled, and kept out, while the trigger holds; it is re-placed once the trigger has been clear for 250 ms:
  - **LD90:** P >= the 90% line (primary);
  - **LD75:** P >= the 75% line;
  - **LDG:** the guard trigger v <= -0.10.
- References: FRONT-PULL-SIZE's ladders, where deep rungs are pulled only at the best (LFPQ-30, LZ-guard-30, LZ-none-30), and J-v3.

Controls passed:
- The self-tests.
- C1 at every rung.
- P aligned on all MOVE rows.
- The age bins add up.

## Answer

$/day without young fills at 1 s, pooled over 9 days (c/sh without young in brackets):

| Arm | M 0 | M 50 | M 250 | U 50 (front bound) | Deep rungs 1-5 at M 50 | Pulls a day |
|---|---|---|---|---|---|---|
| LZ-none-30 (never pulled) | -7,742 | -8,548 (-1.70) | -10,572 | -6,691 | -6,261 (-1.82) | 0 |
| LFPQ-30 (pull at the best only) | -4,384 | -5,095 (-1.99) | -7,069 | -4,149 | -4,861 (-2.05) | 27,437 |
| **LD90** | -737 | **-1,062 (-0.83)** | -4,567 | +134 | -640 (-1.09) | 72,729 |
| LD75 | -680 | -891 (-1.19) | -3,132 | +268 | -564 (-1.49) | 106,884 |
| **LDG** | **-16** | **-137 (-0.33)** | -1,785 | **+781** | **-47 (-0.33)** | 77,513 |
| J-v3 (5 sh) | +4.94 | +3.70 (+0.65) | -4.13 | +58 | | 0 |

Deep rungs' fills at high P (above the 0.424 tercile line of DEPTH-TIMING), M 50:

| Arm | High-P share of deep fills |
|---|---|
| LZ-none-30 | 54% |
| LFPQ-30 | 63% |
| LD90 | 13% |
| LD75 | 12% |
| LDG | 12% |
| Top makers' real deep fills (DEPTH-TIMING) | 19% |

The deep rungs' swept share is still 61-69% at M 50. The top makers' is 36%.

Readings (declared, evaluated in code):
- **R430 DEEP-PULL-HELPS.** LD90 beats LFPQ-30 by +$4,033 a day at M 50 (95% interval [+3,587, +4,480]), 9 of 9 days at both clocks.
- **R431 0 of 3** for LD90, LD75 and LDG: none is positive at M 0 / 50 / 250.
- **R432 0 of 6.** LD90 minus J-v3 is -$1,065 a day at M 50.
- **R433 MECHANISM.** The high-P share of LD90's deep fills fell from 54% to 13%, against 19% for the top makers.

Printed, not a reading:
- LDG minus J-v3 is -$21 at M 0 (interval [-96, +54], 3 of 9 days better) and -$140 at M 50 (1 of 9).
- At the front bound it is +$723 (9 of 9).

## What it means

1. **Timing is a large part of the gap, and the fix works in the expected direction.** Pulling every rung when risk is high moves our deep
   fills' risk mix to the top makers' (13% at high P against 19%). It recovers about $4,000 a day of the ladder's loss on every day.
2. **The guard value is again the better trigger.** For about the same number of pulls (77k against 73k a day), LDG loses -0.33 c/sh
   against LD90's -0.83. Its deep rungs are about break-even: -$47 a day at M 50, +$9 at M 0.
3. **What is left is queue place at the deep levels.**
   - Even with the right timing, 61-69% of our deep shares come from trades that sweep through the level, against 36% for the top
     makers.
   - Each pull sends a rung to the back of its level: about 75,000 pulls a day.
   - The top makers keep their deep orders for seconds (ORDERS: 5.9 s median between orders at a price) and are first in those queues
     (64% of their deep fills are trades that stop at their level).
   - With nothing ahead of us (U) the guard ladder earns +$781 a day. The gap between that and -$137 is the value of the queue place
     we lose.
4. **The 250 ms fill clock is still negative for every resting arm** (C2). The value depends on how late our information arrives
   relative to the trade.

## Next (proposals, the user's call)

- **KEEP-PLACE:** the guard ladder (LDG) with fewer, stickier pulls, so the rungs keep their place:
  - pull a deep rung only on a strong signal (the guard AND P >= the 90% line), or only when it is within 1 tick of the best;
  - after a pull, re-enter at once when the trigger clears (no 250 ms wait), or not before the level changes.
  - These are tuning choices on consumed days. Any pass needs a forward check.
- **The live probe** remains the test of the queue place itself.

## Limits

- Replay rule M (B1a). Rule L was not run. U is a bound. Volumes are inflated (C4).
- Rungs are independent orders without inventory control.
- Consumed days; this design came after DEPTH-TIMING; three candidate arms.
- The age table is not split by anchor source (C12).

Outputs: `data/predict_fun/research/deep_pull_20261113/` (deep_pull.json / .txt, eval_*.json, logs); copies in `reviews/deep_pull_20261113/`.
