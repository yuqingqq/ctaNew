# FRONT-PULL: rest at the best by default, pull on the chance the level goes (results, 2026-11-13)

Session ce68ebe0. USER: "How we can do front pull" / "You mean we use imbalance, level change to do the action? We can try".

Scripts and runs, all in the claude slot:

| Step | Script / driver | Commit | Run |
|---|---|---|---|
| Feed split (post hoc) | `research/pf_move_avoid.py feeds` | 50227d6e, committed before its run | 02:25-02:36Z |
| FRONT-PULL | `research/pf_front_pull.py` + `run_front_pull.sh`; `pf_kp_s4.py` gains one inert switch | df320ba3, committed before the first run | 02:57-03:54Z |

Setup (the script header has the full design and the declared readings):
- **Simulator:** S4's on the 9 consumed replay days, first 60 s. One 5-sh order per side rests at the known best by default (B21's rest
  lane: held within 3 ticks, re-placed once its trigger has been clear for 250 ms). B21 / S4 clocks, trade times and latencies.
- **Signal P:** MOVE-AVOID's fit of "the side's best moves worse in (t + 24, t + 274] ms", one model per held-out day fitted on the other
  8, computed at every 50 ms step. Inputs are the simulator's own (the guard's 32 features, v, our / the other side's level size change
  over 250 ms, Binance activity and imbalance change) and the side.
- **Lines:** quantiles of P over the training days' rows: 50% about 0.04, 75% about 0.10, 90% about 0.25.
- **Arms:**
  - Z-none (never pulled), Z-guard (pull at v <= -0.10, B21's best signal), J-v3 (the reference);
  - FP10 / FP25 / FP50: pull when P is in the top 10 / 25 / 50% of moments;
  - FPQ: the line set by queue place - orders with <= 25 sh known ahead keep until the 90% line, entry and orders further back use the
    50% line;
  - FPQ-shuf: FPQ with P shuffled in time (falsifier);
  - random pulls matched to FPQ's count (200 draws).

Controls passed:
- S4 / B21 self-tests and the mutation check.
- The new idle-only switch is falsified both ways. The tier choice rejects a planted swap.
- Each day's model reproduces MOVE-AVOID's per-day AUC exactly (0.815-0.859).
- P from the simulator equals the fit's prediction on all 591,868 MOVE rows.
- The never-pull path equals Z-none fill for fill. Age bins add up.
- P undefined (no known best or guard value) on 4.5% of steps: kept, no entry block.

## Answer

$/day at 1 s, pooled over 9 days, replay rule M 50 (primary):

| Arm | All | Without young fills | B1a M 0 / M 50 | c/sh w/o young | Shares/day | Pulls/day | Swept | Pending (its $) |
|---|---|---|---|---|---|---|---|---|
| Z-none | -610.95 | -458.95 | -240.72 / -205.45 | -1.38 | 39,921 | 0 | 60% | 3% (-32.35) |
| Z-guard | -21.63 | -20.20 | -10.04 / -8.69 | -0.37 | 6,273 | 12,667 | 52% | 7% (-6.67) |
| J-v3 | +14.73 | +3.70 | +2.92 / +2.46 | +0.65 | 1,433 | 0 | 75% | 21% (+2.13) |
| FP10 | -114.33 | -90.22 | -47.85 / -37.26 | -0.64 | 16,420 | 11,647 | 48% | 6% (-18.25) |
| FP25 | -85.16 | -68.24 | -36.13 / -27.92 | -0.91 | 9,061 | 17,431 | 55% | 7% (-11.70) |
| FP50 | -63.58 | -49.45 | -26.54 / -19.99 | -1.30 | 4,862 | 16,989 | 63% | 7% (-6.39) |
| FPQ | -62.31 | -48.18 | -25.90 / -19.48 | -1.18 | 5,140 | 16,679 | 61% | 6% (-6.52) |
| FPQ-shuf | -45.04 | -20.01 | -12.80 / -9.37 | -1.28 | 2,830 | 14,312 | 73% | 13% (-6.28) |
| random (matched to FPQ) | | -332 to -573 by day (mean of 200 draws) | | | | | | |

Without young fills, $/day at other clocks and at the front bound (U: nothing ahead of us; an upper bound, not a queue model):

| Arm | M 0 | M 250 | U 50 | U 50 c/sh w/o young |
|---|---|---|---|---|
| Z-none | -421.08 | -550.17 | -158.98 | -0.21 |
| Z-guard | -7.05 | -122.00 | +212.96 | +0.59 |
| J-v3 | +4.94 | -4.13 | +58.24 | +1.14 |
| FP25 | -50.60 | -209.17 | +245.65 | +0.49 |
| FPQ | -40.83 | -112.25 | +171.36 | +0.59 |

Readings (declared, evaluated in code):
- **R380 PULL-NOT-SHOWN.**
  - FPQ beats the matched random pulls on 9 of 9 days at M 0 and M 50.
  - It loses to the guard trigger on 0 of 9 days: -$28.0 a day at M 50 (95% interval [-34.0, -22.0]) and -$33.8 at M 0.
- **R381 0 of 6** against J-v3: -$51.9 a day at M 50 w/o young, 0 of 9 days better.
- **R382 FRONT-LINE-HELPS, but small.** Keeping front orders until the 90% line beats FP50 on 7 / 8 of 9 days at M 0 / M 50,
  +$1.3 a day. At M 250 it reverses (-$1.4, 1 of 9).
- **R383 0 of 3.** FPQ loses per share at every clock: -1.05 / -1.18 / -1.89 c/sh w/o young at M 0 / 50 / 250.
- **R384 the falsifier fails as required.** The shuffled P does not pass R380, so the evaluation stands.

## What it means

1. **Pulling on "the level goes" is not the pull we need.** It removes 90% of the never-pull loss, and it is far better than random
   pulls. But the guard's own value v, used as B21's trigger, loses less with fewer pulls: -$20 against -$48 a day w/o young, -0.37
   against -1.18 c/sh. The level going and our fill being bad are not the same event.
2. **More pulling makes each remaining fill worse.** Going from the top-10% line to the top-50% line, the loss per share grows from
   -0.64 to -1.30 c/sh and the swept share from 48% to 63%.
   - The moments P flags are the active ones. They carry the standing fills that pay as well as the level losses.
   - What survives the pull is the sweeps that come without warning.
   - Per share, the real P is only 0.1 c better than the same policy with P shuffled in time (-1.18 against -1.28).
3. **The queue model decides the sign.** With nothing ahead of us (rule U), resting plus a pull earns +$171 (FPQ) to +$246 (FP25) a day
   w/o young, 5% of shares swept. Under rule M the same policies lose, with 61% swept.
   - Under M, 60% of FPQ's filled shares had at most 5 sh ahead just before the fill. We reach the front when the queue ahead leaves,
     which is when the level is about to be taken (B21's note: rule M credits every shrink ahead of us).
   - That is the top makers' pull from the other side: when they leave, we move up, and we inherit the sweep.
4. **Speed is a small part here.** Fills while our cancel was in flight are 6-7% of FPQ's shares and -$6.5 a day at M 50, about 13% of
   its loss.
5. **Not a candidate.** v3's gated quoting (J-v3) stays the only positive arm. The guard value v stays the best pull trigger we have.

## Next (proposals, the user's call)

- **The live A4b capture decides this question.** Between rule M and rule U the resting policies move by $220-310 a day. Only live
  data shows where our orders really are in the queue, and who leaves before a sweep.
- **A queue-priced line on v instead of P:** Z-guard with a front tier, about 1 h. R382 says the front line helps a little; v is the
  better trigger.
- **The tape check of the top makers' exits:** are they in the level-taking trades, or gone just before? Public tape, about 1 h. This
  tests reading 3's mechanism directly.

## Limits

- Replay fill rule M is not validated (B1a, C10). Fill rule L was not run (C5). U is a bound, not a model. Volumes are inflated (C4).
- The age table is not split by trade-anchor source (C12).
- One 5-sh order per side. Our order is not in the public book. v3's CAP10 / RO60 / MICRO / G4 and Binance hard guards are not in the
  rest lane. This is J-v3, not the frozen v3 (C11).
- Consumed days. The signal's form was chosen after MOVE / AVOID on these days (exploratory).
- MOVE's rows, not our resting steps, are the training population. The AV fit's v comes from guard models whose fits included the
  held-out day (S4's indirect leak).

Outputs: `data/predict_fun/research/front_pull_20261113/` (front_pull.json / .txt, fit.json, eval_*.json, models, p_rows_*.npy, logs);
copies (json / txt) in `reviews/front_pull_20261113/`.
