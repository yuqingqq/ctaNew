# QR, the queue race: why our orders sit at the back, and how fast we would need to be (results, 2026-11-12)

Session ce68ebe0. USER: "it seems we quote at bad timings, not sure if it is because we requote slower than others, or we requote
and we lose the position. And yes, if we lose the latency game, we need to check how fast we need to be".

Script and run:
- Script `research/pf_qr_queue_race.py` and driver `research/run_qr.sh` (plan `20261112_queue_race_plan.md`, bd71adbb), committed
  before the first run.
- Part C fix 4756b27d after its first run looped. The live re-place match can pair two orders both ways when a cancel is sent
  before its ack; the readings are unchanged.
- Run 09:58-10:31Z in the claude slot. Peak 906 MiB in the looping first live run; at most 787 MiB elsewhere.

Data:
- 9 consumed replay days (10-23..27, 10-31, 11-01..03), first 60 s of each window, one 5-sh order per lane.
- Part C: the live engine's orders in the 220 consumed 11-08 windows (one day).
- Exploratory: nothing here is validated.

Controls passed:
- The copied simulator equals S4's fill for fill and order for order on all 5,184 window-sides: v3, rest with the guard and rest
  never pulled at M 0 / M 50 / L 50, and lane B at all five configs (0 differ).
- Lane B's own entry steps run through the new lane equal lane B (0 differ).
- The regression can fail: with a 30 ms cancel it differs on 40 of the 40 window-sides tested.
- The ghost equals the simulator's queue at activation for every decomposed order (0 differ), and at the fill trade for 122,493
  fills (0 differ, none skipped).
- The removed parts add up to the queue ahead for every fill (0 differ).
- Synthetic cases with known answers; four planted errors caught. A planted live re-place cycle was added with the fix.

Statuses:
- 187,457 level births (improvements) and 187,373 random joins; 1,730 entries fell before a window's first grid step.
- Orders not at the best when they went live are not decomposed. For v3: 44,279 such orders, against 133,683 decomposed.

## 1. Why we sit at the back: mostly neither. We join the level late

Queue ahead when our order goes live, split into parts that add up to it:

| Part | v3 in the replay (M 50), mean sh | Live engine 11-08 (rule M), mean sh |
|---|---|---|
| Already at the level when it became the best | 92 (17%) | 31 (7%) |
| Joined within our 97 ms: "slower than others" | 88 (16%) | 115 (27%) |
| Joined after we could have been there: "late" | 227 (41%) | 191 (44%) |
| Lost by our own re-places: "requote" | 80 (15%) | 62 (14%) |
| The simulator's allowance for the next snapshot | 60 (11%) | 32 (7%) |
| Queue ahead at go-live | 548 | 431 |

Readings:
- **R280 LATE-ENTRY, STABLE:** the largest part on 9 of 9 days.
- **R281 LATE-ENTRY** for the live engine under rule L and rule M.
- At M 0 and L 50 the shares move by at most 4.1 points (rule L: U 20%, LATE 42%, REQ 10%).

Why we are late:
- **v3's first order at a level goes live a median 177 ms after the level is born** (p90 2.3 s). Only 14.5% arrive within 97 ms.
- The live engine: a median 386 ms (p90 5.0 s), 14.2% within 97 ms.
- When our order goes live the level is a median 597 ms old for v3 and 1.44 s for the live engine.
- v3 joins when its guard turns on, or again after a fill or a pull. It does not aim at the birth.
- 43% of v3's at-best orders are re-places (live 59.5%).

The rest arms:
- Resting with the guard has the same pattern (late 48%).
- Never pulling has no re-places; its largest part is the queue already there (41%), on levels that fall back.

**The fills say the same thing** (v3, c/sh at 1 s, fills). The table was declared before the run; the per-day counts below it are
post hoc:

| Our order went live when the level was | M 0 | M 50 | L 50 |
|---|---|---|---|
| < 100 ms old | +1.76 (758) | +1.59 (667) | +1.60 (640) |
| 100-500 ms old | +1.37 (1,666) | +1.06 (1,520) | +1.05 (1,471) |
| 0.5-2 s old | +0.59 (477) | +0.03 (495) | -0.02 (467) |
| >= 2 s old | -0.14 (221) | -0.14 (238) | -0.22 (213) |
| First order at the level | +1.47 (2,453) | +1.16 (2,223) | +1.17 (2,138) |
| Re-placed order | +0.37 (669) | +0.07 (697) | +0.01 (653) |

- Post hoc: young levels (< 0.5 s) beat old ones on 9 of 9 days, and first orders beat re-places on 9 of 9 days, at M 0 and at
  M 50.
- So v3 earns on the orders that joined a level early. Its late joins and its re-places earn about nothing.

How the front was reached (declared readings):
- **R282 EARLY, v3: EARLY-DOES-NOT-PAY under the declared rule, but only because of sample size.**
  - Fills that went live with < 20 sh ahead pool to +1.54 / +0.91 c/sh at M 0 / M 50, positive on 8 of 9 days at each clock.
  - Only 6 and 4 days had the 30 fills the rule requires.
- **R282 EARLY, never pulling: DOES-NOT-PAY.** Fills near the front without a guard lose -1.89 / -2.00 c/sh, 0 of 9 days positive.
  - A thin best level with no guard is a level about to be swept.
- **R283 ATTRITION-WORSE** for never pulling, 9 of 9 days at both clocks. At M 50:
  - a front reached because others cancelled: -0.19 c/sh;
  - a front reached because the queue traded through: -0.04;
  - sweeps through our price: -2.41.

## 2. How fast we would need to be

**The venue's book comes in snapshots about 50 ms apart.** The next snapshot after a birth comes p10 / p50 / p90 41 / 50 / 58 ms
later. Every entry under ~50 ms therefore lands in the same place: behind the shares shown in the birth snapshot.

Median queue ahead at go-live, joining every new level with no guard (M 50):

| Total delay from the birth | 0 | 15 | 30 | 55 | 97 (ours) | 150 |
|---|---|---|---|---|---|---|
| With the simulator's next-snapshot allowance | 122 | 122 | 123 | 200 | 250 | 333 |
| Without it | 51 | 51 | 51 | 104 | 143 | 260 |

**R284 SPEED-BUYS-QUEUE:** at 30 ms the queue ahead is half of ours at 97 ms (123 against 250).

Value of joining every new level at that delay (c/sh at 1 s; $/day at M 50):

| Entry | v3's guard: M 0 / M 50 / M 250 | v3's guard: $/day | No guard: M 50 | No guard: $/day |
|---|---|---|---|---|
| 0 ms (a bound) | +1.51 / +0.97 / -0.14 | +16.8 | -0.59 | -165 |
| 15 ms | +1.17 / +1.04 / -0.17 | +15.7 | -0.61 | -169 |
| 30 ms | +1.39 / +1.11 / -0.16 | +18.8 | -0.59 | -166 |
| 55 ms | +1.18 / +1.05 / -0.33 | +11.7 | -0.78 | -192 |
| 97 ms (our median) | +1.40 / +1.16 / -0.39 | +12.9 | -0.83 | -197 |
| 150 ms | +1.69 / +1.43 / -0.33 | +8.6 | -0.93 | -196 |
| Lane B (today's clock, realised ~110 ms) | +1.86 / +1.64 / +0.07 | +11.2 | -0.87 | -195 |
| Random-time joins at 30 ms | +0.97 / +0.69 / -0.53 | +1.7 | -1.27 | -280 |
| v3 | +1.38 / +1.03 / (S4: -0.74) | +14.7 | | |

Readings:
- **R285, no guard: SPEED-PAYS** (30 ms beats 97 ms on 9 of 9 days at M 0 and M 50). **R286, no guard: NONE**: it loses at every
  speed. Speed alone never makes it pay.
- **R285, v3's guard: MIXED per share** (5 vs 4 days). **R286, v3's guard: positive at every delay in the sweep at M 0 and M 50**
  (POS = 150 ms). Per share, only the 150 ms entry beats v3 (V3 = 150).

Post hoc, in $/day with the guard:
- 30 ms beats 97 ms on 9 of 9 days at M 0 and at M 50: about +6 $/day at M 50, from more fills at the same per-share value.
- 30 ms beats v3 on 8 of 9 days at M 50 and 7 of 9 at M 0.
- At the 250 ms fill clock every delay loses (-2.5 against -3.7 $/day; 30 ms better on 5 of 9 days).
- Lane B enters only once the stale clock still shows the level, which skips levels that vanish within ~50 ms. It has the best
  per-share value and is the only arm not negative at M 250 (+0.07). The levels that vanish fast are the bad ones to join.
- At the same speed, joining at a birth beats joining at a random time: +1.11 against +0.69 c/sh guarded, -0.59 against -1.27
  unguarded. Timing the entry to the birth matters more than the speed.

## What it means

1. **Our orders are behind mainly because we come to the level late.** Being slower than others and losing place on re-places are
   each about 15% of the queue ahead in the replay (live 27% and 14%). The largest part (41-44%) joins after we could have been
   there.
2. **The speed that matters is one book snapshot, about 50 ms.**
   - Under that, our order joins behind only the shares in the birth snapshot (~51 sh). At our 97 ms it is behind 2-3 times as
     many (51 against 143 sh without the allowance, 123 against 250 with it).
   - With the guard, 30 ms instead of 97 ms adds fills at the same per-share value. That is useful but does not change the sign.
   - Without the guard, no speed pays.
3. **The larger lever in this replay is when we enter, not how fast.**
   - Join new levels at their birth while the guard is on, and avoid levels that vanish at once.
   - Re-places and late joins earn about nothing.
   - Whether keeping the order instead of re-placing it would earn more is not tested here: B15's live soft pull left the value
     about unchanged.
4. **The fill clock still decides the sign.** At 250 ms only lane B is not negative.

## Limits

- Consumed days, exploratory: a pass makes a candidate, not a result.
- Rule M is not validated (B1a). Young orders, exactly the < 100 ms class and the fast entries here, were mostly phantom fills in
  B1a's live check.
- Part B speeds up only the entry; exits and the guard keep today's clock.
- The book's 50 ms snapshots set the resolution of "at the birth".
- One 5-sh order per lane, and our order is not in the public book.
- The 97 ms split point is fixed.
- Part C is one day; its trade times are unshifted (as B13c).
- Post hoc items are labelled.

## Next (proposals, the user's call)

- **QR2: v3 plus birth entry, on the same days with the same controls.**
  - v3 also joins a new level within ~30-50 ms of its birth while its guard is on.
  - It skips levels the stale clock no longer shows (lane B's filter).
  - It keeps its order through guard flicker instead of re-placing it.
  - Readings declared first; a pass leads to a freeze and a forward test on >= 5 untouched days after 11-13.
- **Live, with the A4b capture:** our real queue place and the fill clock (M 0 against M 250) decide whether any of this survives.

Outputs:
- `data/predict_fun/research/qr_20261112/`: qr.json / qr.txt, day_*.json, arr_*.npz, live.json, logs.
- Copies (without the npz) in `reviews/qr_20261112/`.
