# MOVE-DOWN + WHY-GONE: can we reach the requote side's new best faster, and why does our other-side order leave? (results, 2026-11-13)

Session ce68ebe0. USER: "The requote side, you mean their fill lands around 300ms but we repost 350 ms later, can we do faster. Bid side,
why we left?"

Script `research/pf_move_down.py` + `run_move_down.sh` (3695d17c, before the first run), claude slot, 14:53-14:58Z, peak 750 MiB.

## Design

- **Lane:** the guard ladder's rung 0 (30 sh, kept through partial fills). 9 consumed replay days, first 60 s.
- **Part 1, protection side** (the side the price moves into: asks in a pump):

| Arm | Rule |
|---|---|
| LDG | today: pulled while the guard is on; re-enters at the best once the guard has been clear 250 ms |
| FAST | today's rule with the fast path (5 ms order path, book known at stamp + 15 ms) |
| MD | move down: once the best has moved against us since the guard came on, enter at the new best at once (no guard-clear or 250 ms wait), keep it through that guard episode |
| MDP | MD, pulled while the guard is on if P(move >= 1 more) >= 0.25 |
| MDF | MD with the fast path |

  Measured at the protection side's new best after a signal (PULL-DEPTH's 75,914 levels) and over all trades.
- **Part 2, the other side** (bids in a pump): our held order at the old best when the other side's price improves, and why it ended
  within 1 s.
- **Controls passed:**
  - MD with early entry off equals LDG, and equals the FAST lane, fill for fill on every window-side.
  - The simulator copy equals the simulator.
  - Every order is mapped to its placement step.
  - Clocks are restored after each run.
  - Synthetic cases with three planted errors caught.

## Part 1: we can get there much faster, but arriving faster after the move loses

Our rung 0 at the protection side's new best after a signal (M 50):

| Arm | Present | Live after the birth, median | Within 100 ms | Queue ahead at go-live (shown, median) | Fills there, aged c/sh | STOP |
|---|---|---|---|---|---|---|
| LDG (today) | 35% | 347 ms | 10% | 436 sh | -0.12 | 24% |
| FAST | 36% | 313 ms | 12% | 430 sh | -0.05 | 26% |
| MD | 60% | **128 ms** | 25% | 222 sh | **-0.70** | 31% |
| MDF | 78% | **46 ms** | 85% | 161 sh | **-0.66** | 37% |
| Top 5 (real, for reference) | | fill at 302 ms | | 0 sh filled ahead (lower bound) | +0.30 | 82% |

- **Where today's 347 ms goes:** at the birth the guard is still on 83% of the time. It clears a median 236 ms later, then we wait 250 ms,
  then the 40 ms order path.
- **R520 FASTER:** MD, MDF and FAST all go live sooner, on 9 of 9 days.
- **R522 LOWER:** MD and MDF are worse per share at the new best on 9 of 9 days. MDP is not shown (4/5).

Rung-0 $/day, WS-timed trades, orders >= 100 ms old:

| Arm | M 0 | M 50 | M 250 | All trades M 50 |
|---|---|---|---|---|
| LDG | +27.7 | +6.3 | -313 | -89.5 |
| **FAST** | **+46.4** | **+28.0** | **-301** | **-45.9** |
| MD | -374.9 | -437.7 | -802 | -911.8 |
| MDP | +28.7 | +6.2 | -350 | -107.7 |
| MDF | -324.1 | -392.1 | -812 | -862.4 |

Readings:
- **R521 MD / MDF HURT:** -$444 / -$398 a day at M 50, 0 of 9 days higher at every clock.
- **R521 FAST HELPS:** +$19 / +$22 / +$12 a day at M 0 / 50 / 250, higher on 9 of 9 days at every clock. All trades +$44 at M 50.
  - This is the first gain in these studies that also holds at M 250.
- **R521 MDP NOT-SHOWN** (+$0.9 / -$0.1). Its entry did not check the model, so it re-entered right after each pull: 44k pulls a day vs
  12.8k, and -$286 a day on orders < 100 ms old. That is a design flaw, so MDP is not informative.
- **Age split (C1):** MD's losses are on orders of every age (-$337 / -$311 / -$127 a day, < 100 / 100-500 / >= 500 ms, WS).

**Why faster entry loses:**
- The new best already holds its queue when it forms (PULL-DEPTH: 128 sh at the birth).
- Even at 46 ms we are behind 161 sh. An order at the back of that queue is filled mainly when the move continues through it (31-37%
  STOP), at -0.66 to -0.70 c/sh.
- The top makers' 302 ms is a fill time, not an arrival time. They are first in line (0 sh filled ahead), so they were in the queue at or
  before the birth.
- Speed after the birth cannot reach that place. What reaches it is being at the price before it becomes the best (QUEUE-PLACE,
  FUTURE-BEST).

## Part 2: why our other-side order leaves: our own guard pulls it

Other side (bids in a pump), LDG rung 0, at the moment the other side's best improves (78,800 events):

| State at the birth | Share | What happened next |
|---|---|---|
| HELD-OLD: our order still at the old best | 40% | within 1 s: **pulled by our guard on that side 52%**, drift (the best ran > 3 ticks away) 11%, still live 34%, filled 1%; median 259 ms after the birth |
| OUT: no order | 41% | the last order was pulled by our guard 79% (a median 815 ms before), first order not placed yet 17% |
| HELD-OTHER / PEND / AT-NEW | 13% / 4% / 1% | |

- The guard on that side is on at the birth only 15% of the time. It mostly comes on in the next few hundred ms, and every rung is pulled
  when it does.
- The top makers' old orders stay: 2.6 times as often as the field's (SIDE-MOVES).
- So we leave because the guard ladder pulls every order on that side whenever its guard fires. That includes an order sitting behind the
  new best, where the price is moving away from it.

## What it means

1. **Faster after the move: no.** Getting to the new best 46-128 ms after it forms puts us behind 160-220 sh. We are filled when the move
   continues, and we lose $400-440 a day.
2. **Faster in general: yes.** The fast order path and feed help with today's rule at every fill clock (+$22 a day at M 50, 9/9; also at
   M 250 and over all trades).
3. **The other side:** our own guard takes our held order away. The top makers keep theirs.

## Next (proposals, the user's call)

- **FAST as an engine requirement:** a 5 ms order path and the fastest feed. The only arm here that helps at every clock. A live latency
  measurement on Tokyo would tell how close we can get.
- **KEEP-BEHIND (other side):** while the price moves away from an order that sits behind the best, do not let the guard pull it (or use
  the depth rule). Measure its value against LDG at M 0 / 50 / 250, WS / E2.
- **The front of the new best:** only by being there before it forms. That needs FUTURE-BEST's "will it stop here" prediction.

## Limits

- Replay rule M (B1a, C5). Volumes inflated (C4). Consumed days (C8). Rung 0 only. Ours simulated (C9).
- The fast path is hypothetical. MDP's entry design is flawed (above).
- The lane is the guard ladder, not the frozen v3 (C11). No order-level validation (C10).

Outputs:
- `data/predict_fun/research/move_down_20261113/`: move_down.txt / .json, fills_*, levfills_*, presence_*, whygone_*, day_*.json, logs.
- Copies (txt / json / logs) in `reviews/move_down_20261113/`.
