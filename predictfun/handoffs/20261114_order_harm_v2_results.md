# ORDER-HARM v2: probability and severity of harm per resting order, after the target-construction repairs (results, 2026-11-14)

Session ce68ebe0.

USER:
- "the next target should be the probability and severity of harm to each order after cancellation could take effect, with queue value
  included in the keep/pull decision".
- Then the review of v1: "Confirmed look-ahead in queue features ... Overlapping labels reuse the same fill ... Economic evaluation must
  count each physical fill once ... Keep and pull branches have different ending conditions ... repairing target construction before
  interpreting" / "Fixed the issues first".

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| v1 `research/pf_order_harm.py` | 67a6e6fb / f8f1db6b | stopped at 06:1xZ before any result after the USER's review; its two days of rows were deleted unread |
| v2 `research/pf_order_harm_v2.py` + `run_order_harm_v2.sh`, readings R620 / R621 declared | abf5638e, before its first run | 06:12-06:29Z |

## The repairs (v1 -> v2)

1. **Causal queue inputs.**
   - v1 fed the simulator's queue to the model: the start of the queue raised to the next frame up to 100 ms after activation, then capped by
     frames selected by stamp.
   - v2 builds the order's queue inputs only from frames known by the decision time (`queue_known`).
   - **Falsifier on real data:** on every 20th decision row, every frame not yet known at t was perturbed and the inputs recomputed:
     114,383 rows, 0 differ.
   - Synthetic cases with the two leaks planted (selection by stamp; the next-frame raise) fail as they must.
2. **No summed economics over overlapping rows.**
   - v1's offline reading R622 summed labels whose 2 s windows share fills; it is dropped.
   - Only ranking diagnostics are reported, per day.
   - The economic test is the joint replay, where each physical fill counts once.
3. **Identical terminal rules.**
   - Both branches are cut at t + H.
   - Rows whose horizon passes the lane's end (60 s, where the reference lane stops and trades are not loaded) are censored: 64,182 at 2 s,
     110,575 at 5 s.
   - The pull branch never runs past the end.

## Design (unchanged apart from the repairs)

- **Rows:** every resting order of the guard ladder, rungs 0-5 including the order at the best, both sides, on the never-pull reference path.
  Taken every 250 ms while its side's guard is on; M 50; 9 consumed days, first 60 s. 2,287,617 rows.
- **Keep branch:** the reference path.
- **Pull branch:** cancel at t; nothing placed while the guard stays on; re-entry behind the queue 250 ms after it clears.
- **Label over (t + 22 ms, t + 2 s],** per share of the 30-sh order, at 1 s marks:
  - HARM: negative keep fills (severity);
  - GAIN;
  - PULLV: the re-entered order's fills;
  - DELTA = keep - pull.

## Results

**The label** (c per share of the 30-sh order):

| Rung | Rows | P(harm) | Severity when harmed | Gain | Pull branch value | DELTA (keep - pull) |
|---|---|---|---|---|---|---|
| 0 (at the best) | 372,022 | 32.0% | 3.22 | +0.21 | -0.14 | **-0.68** |
| 1 (one tick behind) | 379,067 | 23.8% | 3.21 | +0.14 | -0.10 | **-0.52** |
| 2 | 382,144 | 18.7% | 3.14 | +0.11 | -0.07 | -0.41 |
| 3 | 383,941 | 15.4% | 3.02 | +0.09 | -0.05 | -0.33 |
| 4 | 385,158 | 12.7% | 2.94 | +0.07 | -0.03 | -0.27 |
| 5 | 385,285 | 11.1% | 2.81 | +0.08 | -0.02 | -0.21 |
| All | 2,287,617 | 18.9% | 3.11 | +0.12 | -0.07 | -0.40 |

- DELTA gets worse the longer the guard has been on: -0.27 at the first step, -0.48 after more than 20 steps.

**The models** (out of day):
- **R620 DELTA-SKILL: SKILL.** Rank correlation +0.151 pooled, +0.121 to +0.166 on 9 of 9 days.
- **R621 HARM-AUC: PREDICTABLE.** 0.753 pooled, 0.731-0.772 on 9 of 9 days.
- The severity model's rank correlation on harmed rows is +0.22.
- **Calibration:**
  - P(harm) is calibrated in every decile.
  - DELTA is slightly optimistic: the best decile is predicted -0.01 and realized **-0.06**, so it is still negative.
- **Top inputs:** the fair-price move (250 ms and 1 s), the Binance imbalance change and short returns against us, the order's depth, the
  sensitivity, volatility, and the fair price against the Predict mid.

## What it means

1. **The harm is predictable.** AUC 0.75 and a rank correlation of 0.15 on every day, from known information only. The fair-price inputs
   matter here, unlike in the move-depth models.
2. **Kept orders lose against pulling and staying out, at every rung and in every predicted decile.**
   - Even at rung 5 the average is -0.21 c/sh.
   - The best tenth of predicted states still realizes -0.06 c/sh.
   - In this replay (rule M, our orders mostly behind the queue) a resting order exposed to a signal is harmed often enough (19-32% within
     2 s, -3 c/sh each) that keeping does not pay on average.
3. **The label prices keeping blind for 2 s against leaving now.** A policy that re-decides every step can keep now and leave later, so the
   label understates adaptive keeping.
4. **The economic answer comes only from the policy replay** on the repaired joint engine: hard cap, each fill once.

## Limits

- Replay rule M (B1a; the queue model is unvalidated, and the label's harm depends on it).
- The keep branch never pulls later.
- Overlapping rows: per-day diagnostics only.
- Rungs are independent in the label; the joint replay couples them.
- Consumed days (C8).

Outputs:
- `data/predict_fun/research/order_harm_v2_20261114/`: order_harm_v2.txt / .json, rows_*, pred_*, fit_* / rows_* json, models/, logs.
- Copies (txt / json / logs) in `reviews/order_harm_v2_20261114/`.
