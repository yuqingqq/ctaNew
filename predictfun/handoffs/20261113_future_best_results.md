# FUTURE-BEST: pull the orders at risk, keep the protection side at the future best price (results, 2026-11-13)

Session ce68ebe0. USER: "Pull the risk ones, and keep our protection side at the future best price, it should be ideally much better. Yes two
sides will be very different. It looks the repost side they are similar to us".

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| Script `research/pf_future_best.py` + `run_future_best.sh` | 037ed9de, before the first run | 13:44-13:57Z, peak 779 MiB |
| Post hoc readout (descriptive) | 6f13eb38 | 13:58Z |
| Review check (after the other session's source review) | 7b5eced4 | 14:01Z |

## Design

- **Data:** 9 consumed replay days, first 60 s. The guard ladder: 30 sh orders, rungs 0-5 ticks behind the best, guard v <= -0.10.
- **The USER's rule (FB):** on the side the guard protects, while the guard is on, an order d ticks behind the best is pulled only if
  the move is likely to go **through** its price: P(move >= d + 1) >= tau.
  - So the order at the predicted landing price (the future best) stays in its queue. PULL-DEPTH (PD) pulled it, using P(move >= d).
  - An order placed behind the best that the market has come to stays unless P(move >= 1) >= tau. Rung 0 at the best is pulled as today.
  - P is PULL-DEPTH's out-of-day model, unchanged; tau = 0.10 / 0.25 as before.
  - The other side's guard is mostly off, so the repost side is unchanged.
- **OFB:** the same rule with the realized move instead of P: a perfect-foresight diagnostic of this fixed rule. It is not a candidate and
  not a bound on what any policy could earn. It knows the next second, which is also the horizon the fills are valued at.
- **Clocks:** M 0 / M 50 / M 250. Splits by order age and trade timing (WS exact / E2 estimated).
- **New best after a signal:** PULL-DEPTH's adverse levels (the hit side's first new best within 1 s of a guard onset, 75,914 levels).
- **Controls passed:**
  - tau = 0 equals LDG fill for fill on every window-side, rung and clock.
  - LDG / PD10 / PD25 equal PULL-DEPTH row for row.
  - FB25 differs from PD25.
  - P and the move are aligned.
  - Synthetic cases with three planted errors caught.

## Result 1: on exactly-timed trades it beats PULL-DEPTH and pulling everything, at M 0 / M 50

$/day at 1 s, orders >= 100 ms old:

| Arm | WS M 0 | WS M 50 | WS M 250 | E2 M 50 | All trades M 50 | Pulls a day |
|---|---|---|---|---|---|---|
| LDG (pull all) | +86.1 | +53.5 | -936 | -190.1 | -136.6 | 77,513 |
| PD10 | +114.6 | +76.5 | -1,109 | -196.0 | **-119.5** | 47,038 |
| PD25 | +146.3 | +97.5 | -1,389 | -271.8 | -174.3 | 31,939 |
| **FB10** | **+135.9** | **+94.1** | -1,239 | -226.5 | -132.4 | 37,655 |
| **FB25** | **+194.8** | **+124.3** | -1,670 | -395.1 | -270.8 | 23,391 |
| OFB (perfect foresight) | +468.8 | +431.8 | -613 | -66.0 | +365.9 | 17,749 |

Readings:
- **R500 BETTER:**
  - FB10 vs PD10: +$21 / +$18 a day at M 0 / M 50, higher on 9 / 8 of 9 days.
  - FB25 vs PD25: +$48 / +$27, higher on 9 / 7.
- **R501 HELPS:** FB10 / FB25 vs LDG +$41 / +$71 at M 50, 8 of 9 days.

But it does not hold everywhere:
- **M 250 (C2): the sign flips.** Every arm that keeps more orders is lower than LDG on 9 of 9 days. This applies to PULL-DEPTH too (post hoc;
  M 250 was not run there):

  | Arm minus LDG, WS $/day | M 0 | M 50 | M 250 |
  |---|---|---|---|
  | PD10 | +28.5 (9/9) | +23.0 (9/9) | -172 (0/9) |
  | PD25 | +60.3 (9/9) | +43.9 (8/9) | -453 (0/9) |
  | FB10 | +49.8 (9/9) | +40.6 (8/9) | -302 (0/9) |
  | FB25 | +108.7 (9/9) | +70.8 (8/9) | -733 (0/9) |

- **R503, all trades M 50:**
  - FB10 is -$13 vs PD10 (2/9) and +$4 vs LDG (3/9).
  - FB25 is -$97 vs PD25 and -$134 vs LDG.
  - On estimated-time trades FB loses more (FB25 -$395 vs LDG -$190).
- **Age split (C1):** the gain is on orders >= 500 ms old (FB10 +77.6 vs LDG +37.8 on WS). Young fills are not the source.
- The extra money is again mostly behind the best (FB10 +97.9 vs PD10 +83.3 a day). At the best it is +3.5 vs +1.5.

## Result 2: we are now at the new best first, but it does not pay yet

Our fills at the hit side's new best after a signal (M 50, WS, aged):

| Arm | Shares | c/sh | STOP | Share of our shares from orders there before it became the best (PRE) | PRE c/sh |
|---|---|---|---|---|---|
| LDG | 23,332 | -0.174 | 25% | 16% | -0.616 |
| PD10 | 22,709 | -0.184 | 25% | 15% | -0.631 |
| PD25 | 21,562 | -0.221 | 27% | 20% | -0.669 |
| FB10 | 21,801 | -0.185 | 27% | 18% | -0.515 |
| **FB25** | **34,303** | -0.182 | 42% | **64%** | **-0.216** |
| OFB | 82,575 | **+0.747** | **78%** | 85% | **+0.901** |
| Top 5 (real, PULL-DEPTH) | | +0.30 | 82% | | |

- **R502 NOT-SHOWN** for both pairs (FB10 vs PD10 higher on 6 of 9 days; FB25 vs PD25 on 4 of 9).
- FB25 does what the USER asked: 64% of our shares at the new best now come from orders that were there before it became the best (16%
  under LDG). PRE fills lose much less (-0.22 vs -0.62 c/sh), but the level as a whole is still negative.
- **The problem is which landing prices we keep** (post hoc):
  - FB25's PRE fills: 50% STOP at +0.24 c/sh, 50% THROUGH at -0.67.
  - OFB's PRE fills: 88% STOP at +1.01.
  - The top 5's fills there: 82% STOP.
  - Half the time the move goes past the price we kept, and the order is run over.
- FB10 barely changes the new best (PRE 18%): at tau = 0.10 the model rarely says "the move stops here".

## What it means

1. **"Ideally much better" is right, in the ideal.**
   - With perfect knowledge of how far the move goes, keeping the order at the landing price would make about +$380 a day more than
     pulling everything on exact trades.
   - It would earn +0.90 c/sh at 88% STOP on the new best. The top makers show 82% STOP there (+0.30 c/sh).
   - So the rule is limited by the depth prediction. How much a realistic predictor can recover is unknown: the oracle also sees the
     next second, which is the horizon the fills are valued at.
2. **With today's model (AUC 0.75) the gain is modest and fragile.**
   - +$18-27 a day over PULL-DEPTH on exact trades at M 0 / M 50.
   - It does not hold over all trades or at M 250.
   - At the new best our kept orders are there first but are run over half the time.
3. **The decision that matters is "will the price stop here?"** That is the question at the moment the market reaches our kept order.
   The top makers' fills there are 82% STOP; ours 50%.
4. **Repost side (the USER's remark):**
   - The top makers do not win the speed race there either: first fill a median 340 ms after the new price vs the field's 301 ms. Ours
     goes live at 207 ms (113 ms on the fast path).
   - They still earn more per share there (+0.53 vs our +0.37 to +0.42; theirs are all real fills, ours simulated and aged).
   - Unchanged by this rule.

## Next (proposals, the user's call)

- **STOP-HERE model:** at the moment the market reaches a kept order (and at guard onset for the landing price), predict whether the level
  holds. Train on the oracle's labels: the level is not traded through within 1 s.
  - Add inputs the depth model lacks: the queue shown at the landing price and behind it, and how fast the queue in front is being eaten.
  - Add Binance sub-second moves (from the 08-24 boundary only).
  - Measure how much of OFB's gap it closes, with the M 250 and E2 splits.
- **Hold the freeze.** Do not freeze PD10 / FB10 yet: neither holds at M 250. Over all trades only PD10 (-$120 vs LDG -$137) and FB10
  (-$132) beat LDG, by a little.

## Limits

- Replay rule M (B1a, C5). Replay volumes inflated (C4). No order-level validation (C10).
- Consumed days, exploratory (C8). Tau from PULL-DEPTH.
- OFB uses the future: a diagnostic, not a bound.
- The rungs are independent orders. The lane is the guard ladder, not the frozen pm_ladder v3 (C11).
- Cancel-pending fills: 6% of WS shares at M 50 (FB25 -$37 a day of them, LDG -$12; C3).

## Review notes (added after the report)

The other session's source reviews (`handoffs/20261113_future_best_model_integration_review.md`, 13:53Z, written while this ran; and
`handoffs/20261113_pull_depth_model_integration_review.md`) raised these points:

1. **R500 / R501 (and PULL-DEPTH's R491) did not apply the declared 30-fill day minimum.** Checked (7b5eced4): the smallest cell has 657
   fills, so no day is excluded and every label is unchanged.
2. **Uncounted statuses:** 0 of 1,383,411 fills lack a 5 s value; 0 of 26,285 level fills lack a 1 s value.
3. **Inherited from PULL-DEPTH's P model.** This affects the PD and FB arms, not LDG. OFB shares the move definition.
   - **Nested-fold leak:** each training row's FRONT-PULL score comes from a model fitted on every other day, which includes the held-out
     day. The magnitude and direction are not measured.
   - **Target clock:** the move is counted from the last known book frame, not from the decision or the cancel taking effect. It is part
     nowcast.
   - **Missing frames:** a valid known frame whose later frames are all invalid counts as "no move". The count is not measured.
   - **Fold order:** leave-one-day-out trains on later days too. R490's AUC is the mean of daily AUCs, not pooled.

   So the PD / FB gains on exact trades are not clean out-of-sample numbers. The M 250 and all-trade conclusions (not robust) stand either
   way.
4. **Independent rungs:** the rungs are independent orders with separate trade budgets. This was already a limit.

A STOP-HERE model should use earlier-only training through the whole chain, targets that start at the action deadline, and explicit
statuses for missing future frames.

Outputs:
- `data/predict_fun/research/future_best_20261113/`: future_best.txt / .json, posthoc.txt, mincheck.txt, fills_*, levfills_*, day_*.json, logs.
- Copies (txt / json / logs) in `reviews/future_best_20261113/`.
