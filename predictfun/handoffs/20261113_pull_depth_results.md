# PULL-DEPTH: pull only the prices at risk; the two sides after a danger signal; how the top makers handle them (results, 2026-11-13)

Session ce68ebe0. USER: "I mean if we can pull the quotes only at risk, for example if current price is 0.5, and we fired some danger signal,
it will probably go to like 0.48, we can pull 0.5, 0.49 and 0.48, keep our best bid at 0.47, so we still keep our position, then the other
side needs to repost. Can we check 2 sides as well ... in this case, I think latency is more important? ... can we also check different side
behavior in these cases for top makers, how do they handle".

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| Script `research/pf_pull_depth.py` + `run_pull_depth.sh` | 6fa5182e, before the first run | First job (rows 10-23) failed 11:55Z on a duplicate column name; no output written |
| Fix | 7f5f6aa1: column renamed, rows built as arrays (the failed job peaked at 1.67 GiB); design unchanged | 11:56-12:16Z, peak 871 MiB |
| Post hoc split (descriptive) | 0fb1fa22 | 12:17Z |

- 11:55-11:56Z another session's short job held core 9 and the driver waited one retry.

## Design

- **Data:** 9 consumed replay days, first 60 s.
- **Ladder:** the guard ladder (LDG: 30 sh, rungs 0-5, every rung pulled on the guard v <= -0.10). "Danger signal" = the guard.
- **Part A:**
  - At every 50 ms step with the guard on: how far the side's best moves against us within 1 s.
  - Model P(move >= j ticks), j = 1..5: LightGBM, fit on the other 8 days for each day (out-of-day).
- **Part B, the USER's rule:** while the guard is on, pull a resting rung d ticks behind the best if P(move >= d) >= tau. The best is always
  pulled; deeper rungs stay in their queue.

| Arm | Rule |
|---|---|
| LDG | pull every rung (today) |
| PD10 / PD25 | predicted depth, tau = 0.10 / 0.25 |
| PDF2 / PDF3 | fixed depth: always pull the top 2 / 3 prices, keep the rest |

- **Part C:**
  - Event = the guard coming on (an onset).
  - Hit side: the first new best by fallback (the level in front went away) within 1 s.
  - Other side: the first new best by improvement (the repost race).
  - Each against "calm" births with no onset nearby.
  - Per level: queue growth, real fills by group and by time since the price became the best, the first fill (the front), PROOF that
    the front order was resting at that price before the signal (a lower bound), and our fills.
- **Controls passed:**
  - tau = 0 equals LDG fill for fill on every window-side, rung and clock (the override run, every rung pulled).
  - PD25 differs from LDG.
  - LDG equals TIMING-SPLIT's rows.
  - P aligned; clocks restored after FAST.
  - The held-out day never in its fit.
  - Synthetic cases with three planted errors caught.

## Part A: most danger signals are false alarms, and the depth is predictable

- The guard is on at 42% of the 50 ms steps (2,290,802 steps, 275,122 onsets over 9 days).
- After an onset, within 1 s the price moves against us:

| Ticks | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| Share of onsets | **60%** | 20% | 8% | 4% | 2% | 5% |

  Within 250 ms: 80% no move.
- **R490 PREDICTABLE (j = 1, 2, 3):** out-of-day AUC 0.753-0.757 (days 0.735-0.775; j = 4, 5 the same).

## Part B: pulling only the prices at risk beats pulling every layer

$/day at 1 s, fills on orders >= 100 ms old:

| Arm | WS-timed trades, M 0 | WS-timed, M 50 | E2-timed, M 50 | All trades, M 50 | All trades, M 0 | Pulls a day |
|---|---|---|---|---|---|---|
| LDG (pull all) | +86.1 | +53.5 | -190.1 | -136.6 | -15.9 | 77,513 |
| **PD10** | **+114.6** | **+76.5** | -196.0 | **-119.5** | **+13.8** | 47,038 |
| **PD25** | **+146.3** | **+97.5** | -271.8 | -174.3 | +9.6 | 31,939 |
| PDF2 (fixed 2) | -749.7 | -1,028.6 | -1,152.3 | -2,180.9 | -1,641.2 | 25,756 |
| PDF3 (fixed 3) | -394.7 | -591.6 | -705.3 | -1,296.9 | -876.8 | 37,630 |

- **R491 HELPS, PD10:** +28.5 / +23.0 $/day on WS-timed trades, higher on 9 of 9 days at M 0 and M 50.
- **R491 HELPS, PD25:** +60.3 / +43.9, higher on 9 / 8 of 9.
- PDF2 reproduces KEEP-PLACE's KPN (-2,181 a day, all trades, M 50).
- With all ages counted (post hoc), the WS M 50 figures are +59.1 / +81.0 / +100.3.

Where the gain comes from (post hoc, WS M 50, w/o young):

| Arm | At the best | Behind the best |
|---|---|---|
| LDG | +0.3 | +62.4 |
| PD10 | +1.5 | +83.3 |
| PD25 | +2.5 | +102.5 |

- At-best fills from orders that were there first (PRE): 17% / 16% / 19% of at-best shares. Deep fills at high P: 5% / 7% / 10% (R492).

So:
- **The prediction is what makes it work.** Always keeping the layers 2-3 ticks back loses $600-1,000 a day.
- **The gain is not mainly "front of the new best".** It comes from keeping our deeper orders through the 60% of signals that are false
  alarms. Their queue place is kept, and their fills behind the best earn more.
- **On estimated-time trades PD25 loses more** (-$272 vs -$190). Over all trades PD10 still beats LDG; PD25 does not.

## Part C: the two sides after a danger signal

Queue at the new best (median shown sh at the birth / +100 / +250 / +500 / +1000 ms):

| Level | Count | Queue | Life, ms median |
|---|---|---|---|
| Hit side, after a signal | 75,914 | **128** / 145 / 236 / 277 / 310 | 388 |
| Hit side, calm | 33,318 | 101 / 118 / 189 / 233 / 265 | 250 |
| Other side, after a signal | 78,800 | **54 / 167 / 349** / 377 / 359 | 391 |
| Other side, calm | 35,412 | 51 / 155 / 334 / 364 / 351 | 253 |

The two sides differ as the USER expected:
- On the hit side the queue is already there when the price becomes the best. It was built while that price was deeper.
- On the other side the new price starts thin and fills within ~250 ms: a race.

Real fills (WS-timed):

| Level | Group | Share of fill shares | First-fill share | First fill after the birth, median | PROOF (rested there before the signal) | c/sh | STOP | c/sh < 100 ms / 100-500 / >= 500 ms after the birth |
|---|---|---|---|---|---|---|---|---|
| Hit side | **top 5** | **42%** | 37% | 250 ms | **18%** | **+0.30** | 82% | +0.05 / +0.25 / +0.39 |
| Hit side | field | 58% | 63% | 204 ms | 10% | -0.11 | 71% | -0.98 / -0.23 / +0.17 |
| Other side | top 5 | 27% | **20%** | 340 ms | 6% | +0.53 | 79% | **+0.98** / +0.73 / +0.36 |
| Other side | field | 73% | 80% | 301 ms | 3% | +0.58 | 73% | **+0.96** / +0.66 / +0.48 |

The top 5's share of all at-best fill shares is 33.9%.

- **Hit side (R493 PROPORTIONAL):**
  - The top 5 take 42% of the fill shares at the new best (34% elsewhere) and earn +0.30 c/sh while the field loses -0.11.
  - They are first in its queue slightly more than their usual share: ratio 1.10, higher on 8 of 9 days, below the 1.2 line.
  - Their first fills are proven to have rested at that price before the signal nearly twice as often as the field's (18% vs 10%, a
    lower bound).
  - They keep orders at deeper prices through moves more than others do.
- **Other side (R494 EARLY-PAYS, 9 of 9):** fills in the first 100 ms earn +0.96 c/sh against +0.45 after 500 ms, so speed matters there.
  The top makers do not win that race: 20% of first fills, first fill at 340 ms against the field's 301 ms.

Ours at those levels (M 50, WS):

| Level | Arm | Shares | c/sh (aged) | STOP | Our order live after the birth, median | Live within 100 ms |
|---|---|---|---|---|---|---|
| Hit side | LDG / PD10 / PD25 | 23,332 / 22,709 / 21,562 | -0.17 / -0.18 / -0.22 | 25-27% | ~320 ms | 18-22% |
| Other side | LDG rung 0 | 11,905 | +0.37 | 33% | 207 ms | 20% |
| Other side | FAST rung 0 | 15,519 (+30%) | +0.42 | 42% | 113 ms | 44% |

- **Hit side:** we still join the new best late after a real move, with or without PD. The rule pulls the price the move reaches (0.48 in
  the USER's example). The kept 0.47 is one price behind the new best, not in it.
- **Other side (R495 NOT-SHOWN):** the fast path gets us in earlier with 30% more shares and slightly better c/sh, but per share it is
  higher on only 6 of 9 days. Our rung 0 sits at the old price when the other side's best improves; it does not jump to the new price.

## What it means

1. **Keeping quotes beyond the predicted move works in replay**, mainly because most danger signals are false alarms. PD10 is the safer
   candidate:
   - better than pulling everything on exactly-timed trades (9 of 9 days at both clocks) and on all trades (pooled);
   - 39% fewer pulls.
2. **It does not by itself put us at the front of the hit side's new best.** That queue is built before the price becomes the best. The
   top makers are there more often (42% of fills, proven resting before the signal nearly twice as often as the field).
3. **On the other side it is a speed race, and early fills pay double.** Nobody dominates it, including the top makers. A faster path helps
   us; a lane that moves our order to the new price at once is not tested.

## Next (proposals, the user's call)

- **PD10 as a candidate:** freeze it (model + tau), then a forward replay on untouched days after 11-13, or the live probe with the churn
  fix. Consumed days here: a pass is a candidate, not a result.
- **REPOST lane on the other side:** when the side's best improves after the other side's signal, move rung 0 to the new price at once (the
  old order stays as a deeper rung), today's path and FAST. Measure c/sh, the share of fills in the first 100 ms and $/day.

## Limits

- Replay rule M (B1a). Consumed days, exploratory, five candidate arms. M 250 not run.
- The guard and P are our signals; the top makers' signals are unknown.
- Their orders are seen only through fills: PROOF is a lower bound. The first leg of a taker order is taken as the front.
- The rungs are independent orders. Estimated-time trades remain uncertain (C12).

Outputs:
- `data/predict_fun/research/pull_depth_20261113/`: rows_*, pred_*, mv models, fit.json, fills_*, events_*, levelfills_*, day_*.json,
  pull_depth.json / .txt, posthoc.txt, logs.
- Copies (json / txt / logs) in `reviews/pull_depth_20261113/`.
