# FRONT-PULL-SIZE: the top makers' order shape in our replay (results, 2026-11-13)

Session ce68ebe0. USER: "it looks we need to increase our size and price levels for testing? Yes can you try to test and check if these
next steps can bridge the gaps".

Script `research/pf_front_pull_size.py` + `run_front_pull_size.sh` (b8e7592a, committed before the first run); 06:46-07:22Z, claude slot.

## Setup

- **Simulator:** S4's rest lane, 9 consumed replay days, first 60 s.
- **Order handling:** an order is kept through partial fills and re-placed only after a known full fill.
- **Signal and lines:** FRONT-PULL's (per-day P models, 50 / 90% lines).
- **Arms:**
  - one order per side at the best at 5 / 30 / 100 sh: never pulled (Z-none), the guard trigger v <= -0.10 (Z-guard), FPQ;
  - a 6-rung ladder per side at 30 sh (L*): rungs 0-5 ticks behind the best; the deeper rungs have no entry block and are pulled only
    while they are at the best;
  - J-v3 (5 sh) as the reference.
- **Two inert switches added to pf_kp_s4.py** (order size, rung offset). Both are off by default.

Controls passed:
- The self-tests.
- Switch cases with a falsifier: a 30-sh order filled twice and kept; a 5-sh order filled once; offset 1 rests one tick behind;
  the defaults equal the plain lane.
- C1 at 5 and 30 sh.
- P aligned on all MOVE rows.
- The age bins add up.

## Answer

$/day without young fills at 1 s, pooled over 9 days (replay rule M 50; c/sh without young in brackets):

| Arm | 5 sh | 30 sh | 100 sh |
|---|---|---|---|
| Never pulled (Z-none) | -459 (-1.38) | -2,287 (-1.44) | -5,542 (-1.51) |
| Guard trigger (Z-guard) | -20 (-0.37) | -89 (-0.33) | -173 (-0.26) |
| FPQ | -48 (-1.18) | -234 (-1.19) | -562 (-1.18) |
| Ladder, 6 rungs (LZ-none / LZ-guard / LFPQ) | | -8,548 / -4,854 / -5,095 (-1.70 / -1.89 / -1.99) | |
| J-v3 (5 sh) | +3.70 (+0.65) | | |

| Arm | Swept share | Orders filled >= 3 times | Filled shares with <= 5 sh ahead |
|---|---|---|---|
| FPQ-5 | 61% | 3.2% | 60% |
| FPQ-30 | 60% | 41.1% | 77% |
| FPQ-100 | 62% | 60.6% | 88% |
| LFPQ-30 | 68% | 48.5% | 83% |
| Top makers' real at-best volume (ORDERS / strategy review) | about 16% | 25.8% | - |

Other clocks:
- **M 0 and M 250:** the same picture. Everything except J-v3 at M 0 loses; J-v3 also loses at M 250 (-$4.13).
- **Front bound (rule U, nothing ahead of us; an upper bound):**
  - single-level resting orders scale up: Z-guard-30 +$689, Z-guard-100 +$1,203, FPQ-30 +$491 a day w/o young;
  - the ladder loses even there (LFPQ-30 -$4,149).

Readings (declared, evaluated in code):
- **R410 SIZE NOT-SHOWN.** FPQ-30 against FPQ-5 per share: higher on 5 / 4 days at M 0 / M 50. Per share the three sizes are the same
  (-1.18 / -1.19 / -1.18 c/sh at M 50).
- **R411 LADDER-HURTS, 9 of 9 days at both clocks.** LFPQ-30 -$5,095 against FPQ-30 -$234 a day.
- **R412 0 of 6** against J-v3: LFPQ-30 minus J-v3 -$5,099 a day at M 50 (95% interval [-5,781, -4,416], 0 of 9 days).
- **R413:** LFPQ-30 and FPQ-30 are positive at 0 of 3 clocks.

## What it means

1. **In the replay, the top makers' order shape does not make our fills look like theirs.**
   - Our 30-100 sh orders are kept and filled repeatedly: 41-61% of filled orders are filled 3+ times, against the top makers' 26%.
   - But 60% of their shares still come from trades that take the whole level. The top makers' real at-best volume is about 16% such
     trades.
   - Per share, size changes nothing (-1.18 c/sh at 5, 30 and 100 sh), so the loss grows with size.
2. **The deeper rungs are the worst part.** A rung 1-5 ticks behind the best is mostly reached by trades that sweep several levels (68%
   swept). It loses even with nothing ahead of it. B12 found the same for real makers: depth does not pay by itself (ranks 21+ lose
   behind the best).
3. **The guard trigger stays the best pull** at every size (-0.26 to -0.37 c/sh, against FPQ's -1.18).
4. **The queue model still decides the sign.**
   - With nothing ahead (U), single-level resting orders with the guard trigger earn +$213 / +$689 / +$1,203 a day at 5 / 30 / 100 sh.
   - Under rule M they lose.
   - Rule M credits every shrink in the shown size as queue ahead of us. A resting order then reaches the front just as others leave,
     before the level is taken. That is the pattern the top makers' exits create (EXIT).
   - Which is true for our real orders can only be measured live.
5. **Not a candidate in replay.** J-v3 stays the only positive arm. Whether the top makers' shape works for us depends on a queue place
   the replay cannot give.

## Next (proposals, the user's call)

- **The live probe** after the churn-fix deploy decides it:
  - one 30-sh order per side resting at the best, kept through fills, pulled on the guard trigger;
  - measure our real queue place (fills in standing vs level-taking trades) against rule M's 60% swept and the top makers' 16%.
- **Drop the deep ladder from the probe.** It loses even at the front bound.

## Limits

- Replay rule M is not validated (B1a). Rule L was not run. U is a bound.
- Rungs are independent orders: there is no inventory control and no CAP / RO; larger orders carry more inventory to settlement.
- Our orders are not in the public book, so the replay cannot show others reacting to a larger order.
- Consumed days; the design came after ORDERS (exploratory).
- Volumes are inflated (C4). The age table is not split by anchor source (C12).

Outputs: `data/predict_fun/research/front_pull_size_20261113/` (front_pull_size.json / .txt, eval_*.json, logs); copies in
`reviews/front_pull_size_20261113/`.
