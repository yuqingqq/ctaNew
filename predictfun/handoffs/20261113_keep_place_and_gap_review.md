# KEEP-PLACE results, and a review of our gaps to the top makers (2026-11-13)

Session ce68ebe0. USER: "Yes go test, check the performance result, then review our gaps to top makers".

## 1. KEEP-PLACE (replay)

Script `research/pf_keep_place.py` + `run_keep_place.sh` (0a5a6ee6, committed before the first run). 08:37-09:05Z, claude slot.

Setup: DEEP-PULL's guard ladder (LDG: 30 sh x 6 rungs per side, the guard trigger on every rung) with fewer, stickier pulls:
- **KPS:** pull only while the guard AND P >= the 90% line hold.
- **KPN:** entry blocked on the guard; a resting rung is pulled only within 1 tick of the best.
- **KPF:** immediate re-entry.

Controls passed: self-tests, C1 at every rung, P aligned, age bins, the clearance restored after KPF.

$/day without young fills at 1 s (pooled, 9 days):

| Arm | M 0 | M 50 | M 250 | U 50 (front bound) | Pulls a day | Deep rungs swept / at high P (M 50) |
|---|---|---|---|---|---|---|
| LDG (reference) | -16 | -137 | -1,785 | +781 | 77,513 | 63% / 12% |
| KPS (strong signal only) | -753 | -1,092 | -4,651 | +88 | 69,097 | 60% / 13% |
| KPN (pull near the best only) | -1,641 | -2,181 | -3,737 | -1,319 | 25,756 | 68% / 67% |
| KPF (immediate re-entry) | -16 | -170 | -2,113 | +976 | 179,126 | 64% / 13% |
| J-v3 (5 sh) | +4.94 | +3.70 | -4.13 | +58 | 0 | |

Readings (declared, evaluated in code):
- **R440:** KPS HURTS (9 of 9 days at both clocks); KPN HURTS (9 of 9); KPF NOT-SHOWN (-$0.07 at M 0, -$33 at M 50).
- **R441:** 0 of 3 for every arm.
- **R442:** 0 of 6 for every arm.

What it means:
- **The pulls are needed.** Pulling less often or at fewer rungs lets the high-risk fills back in (KPN: 67% of deep fills at high P) and
  loses more. A stronger condition (KPS) weakens the guard.
- **Our swept share is not caused by the pulls.** With a third of LDG's pulls, KPN's deep rungs are still 68% swept. Our deep orders sit
  behind queues that already exist when we place them.
- **The guard ladder (LDG) stays our best shape:** -$16 a day at M 0, -$137 at M 50, +$781 at the front bound.

## 2. Our gaps to the top makers

Theirs are real fills on the 9 consumed days, first minute (ORDERS, EXIT, DEPTH-TIMING); ours are the replay (rule M 50) unless stated.

| Dimension | Top makers | Us (best replay shape: LDG) | Status |
|---|---|---|---|
| Order size | median 30 sh filled per order, p90 100 | 30 sh | **closed** |
| Keep orders through fills | 25.8% of orders filled 3+ times | 41% at 30 sh (FRONT-PULL-SIZE) | **closed** (in replay) |
| Price levels | 6 per side per minute (p90 11) | 6 rungs per side | **closed** |
| Exposure at high risk (deep fills at high P) | 19% | 12% | **closed** (DEEP-PULL) |
| Pull on the signal | gone 29% at low P, 79% at high P (EXIT-v2) | guard trigger on every rung | **closed** in kind; their trigger is not known exactly |
| Two-sided, end near flat | 89% two-sided, \|net\| / gross 27% | ladders 26-29% \|net\| / gross | about **closed** |
| **Which trades fill us at a level** | at the best 20% level-taking; behind it 36% | rung 0 48%; behind it 63% | **OPEN** - the largest gap |
| Value per share at the best / behind it | +0.40 / +0.89 c/sh | -0.33 / -0.33 c/sh | **OPEN** (follows from the row above) |
| Fill clock / speed | decide 100-1000 ms before the trade | 29% of level moves happen before our cancel can land (MOVE); everything loses at the 250 ms clock | **OPEN** |
| Taking | 12.6% of their maker volume, mostly adding to their position | none | **OPEN**, untested |
| Scale | about 460k sh a day in the first minute (B12, 11-08) | about 47k sh a day | follows the rest |

### What the open gaps say

1. **Queue place at each level is the gap that decides the sign.**
   - The top makers' fills mostly come from trades that stop at their level (64-80%), which means they are near the front.
   - Ours mostly come from trades that take the level (48-63%), which means we are behind.
   - With nothing ahead of us (U) the same shape earns +$781 a day; under rule M it loses $137.
   - KEEP-PLACE shows this is not caused by our pulls. A new order placed at a level joins behind everything already there.
   - The top makers' orders behind the best are more likely orders that were placed earlier and kept as the price moved away, so they
     are first in those queues. They are not fresh orders posted behind other people's.
2. **Speed / fill clock.**
   - 29% of the level moves are over before our cancel can land.
   - At the 250 ms clock every resting arm loses heavily.
   - Their exits are decided earlier (100-1000 ms ahead). Ours react to the same signal later.
3. **Taking is a part of their edge we have not modelled.** About a quarter of their edge (B12) and 12.6% of their volume.

## 3. Next (proposals, the user's call)

- **KEEP-AND-ADD ladder (replay):**
  - place only at the best;
  - when the price moves away, keep the old orders (they become the deeper rungs, first in their queues) and add a new order at the new
    best, up to 6 per side;
  - pull every order on the guard trigger.
  - This tests whether early placement plus keeping gives our deep orders the top makers' queue place. It needs a multi-order lane in
    the simulator.
- **The live probe** of our real queue place with the same shape.
- **A taking study:** when the top makers take (signal, inventory), and whether a small directional take adds value for us.

## Limits

- Replay rule M (B1a); U is a bound. Consumed days; KEEP-PLACE's three variants are tuning on seen days.
- Theirs and ours differ in population: theirs are real legs; ours are simulated fills, and DEPTH-TIMING compares only WS-timed,
  complete trades.

Outputs: `data/predict_fun/research/keep_place_20261113/` (keep_place.json / .txt, eval_*.json, logs); copies in `reviews/keep_place_20261113/`.
