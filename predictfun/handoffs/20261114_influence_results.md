# INFLUENCE: the target reset, and the influence of a signal measured in this market's price units (results, 2026-11-14)

Session ce68ebe0. USER: "We need to reset the target and rebuild the model, to measure the influence effects".

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| Script `research/pf_influence.py` + `run_influence.sh`, readings R590-R594 declared | 19e52dfb, before the first run | 04:40-04:56Z, peak 1.1 GiB |

## What was reset

- **The signal is an event.** The rows are our guard's onsets (guard on, off at the step before, neither masked), not every guard-on step:
  269,605 onsets on 9 consumed days, first 60 s.
- **The target is the whole move.**
  - The farthest the side's best goes against us within 2 s, counted after t + 22 ms (1 s and 3 s printed).
  - Where it settles at 3 s.
  - The time to the peak.
  - A move already made by t + 22 ms is past saving (status).
- **The signal's size is in this market's price units.** Step 5's Binance-implied fair price p_bn, basis-adjusted, is used three ways,
  all signed against our side:
  - its change over 250 ms / 1 s;
  - the fair price minus the Predict mid;
  - the sensitivity (cents per BTC bp).
- **The influence on the market is measured without any model.** After the fair price moves at least 0.5 c against a side in 250 ms, how
  far and how fast the Predict price follows: 34,813 events.

## Results

**A. How the market follows a fair-price move** (R590 FOLLOWS: 9 of 9 days)

| Fair-price move in 250 ms | Events | Mean | Predict mid moved: 250 ms / 1 s / 3 s (ticks) | Ticks per cent at 1 s | Best moves >= 1 / 2 / 3 ticks in 2 s | Our guard on |
|---|---|---|---|---|---|---|
| 0.5-1 c | 10,123 | 0.72 c | 0.64 / 0.90 / 0.91 | 1.25 | 63% / 44% / 31% | 64% |
| 1-2 c | 10,455 | 1.45 c | 1.33 / 1.82 / 1.73 | 1.25 | 79% / 62% / 47% | 81% |
| 2-4 c | 9,234 | 2.81 c | 2.32 / 3.01 / 2.92 | 1.07 | 89% / 78% / 65% | 91% |
| >= 4 c | 5,001 | 6.55 c | 4.55 / 5.82 / 5.67 | 0.89 | 95% / 91% / 85% | 97% |

- The Predict price follows the fair price about one tick per cent.
- 70-80% of the 1 s move happens within 250 ms.
- It does not come back within 3 s.
- Our guard is on at most of these events.

**B. The move after our signal** (onsets with nothing moved by t + 22 ms, n 257,451)

| Farthest against us within | 0 | 1 tick | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| 1 s | 63.4% | 18.7% | 7.4% | 3.9% | 2.2% | 4.4% |
| **2 s (the target)** | **50.0%** | **21.0%** | 10.2% | 6.0% | 3.8% | 9.1% |
| 3 s | 42.1% | 20.9% | 11.5% | 7.2% | 4.8% | 13.5% |

- Median time to the peak: 1 tick 618 ms, 2 ticks 962 ms, 3 ticks 1,091 ms. A 1 s window cuts many moves short.
- At 3 s the price is below its 2 s peak at 46.5% of onsets (partial reversion); mean 0.47 ticks against us.

**By the fair-price move in the last 250 ms** (against us):

| Fair-price move | Share of onsets | No move in 2 s | Mean ticks | Moves that stop at 1 tick | 5+ ticks |
|---|---|---|---|---|---|
| in our favour | 42.7% | 50.7% | 1.43 | 41.4% | 9.0% |
| 0 to 0.5 c | **54.4%** | 50.8% | 1.34 | 44.4% | 8.0% |
| 0.5-1 c | 0.8% | 34.0% | 2.12 | 31.7% | 15.2% |
| 1-2 c | 0.9% | 25.2% | 2.75 | 23.7% | 21.7% |
| >= 2 c | 1.2% | **12.6%** | **5.85** | **10.2%** | **51.7%** |

- **Only 2.9% of our guard's onsets come with a fair-price move of at least 0.5 c.** There the move is large and follows it.
- At the other 97% the move does not depend on the fair price: half the time nothing moves in 2 s, and a move that starts stops at 1 tick
  41-44% of the time.
- The level of the fair-price gap hardly matters. Mean peak 1.38-1.63 ticks across its bins. Its rank correlation with the peak is positive
  on 9 of 9 days but tiny (+0.04 to +0.08; R591 FOLLOWS).

**The rebuilt model at onsets** (out of day; farthest within 2 s >= j)

| AUC | FULL (with influence) | NOINF | INFONLY (influence + side) | STEP (STOP-DEPTH's model) |
|---|---|---|---|---|
| j = 1 | 0.701 | 0.700 | 0.577 | 0.697 |
| j = 2 | 0.685 | 0.684 | 0.589 | 0.679 |
| j = 3 | 0.675 | 0.674 | 0.594 | 0.667 |
| j = 4 / 5 | 0.669 / 0.669 | | | 0.660 / 0.654 |

- **R593 RESET-HELPS: BETTER** than the step model on the new target (8 of 9 days, j = 1-3): +0.004 to +0.015 AUC.
- **R592 INFLUENCE-HELPS: NOT-SHOWN.** The influence inputs add nothing beyond the Binance features already there (6, 6, 5 of 9 days). The
  model does use the sensitivity and the fair-price change: both are in the top inputs on 9 of 9 days.
- FULL is well calibrated.
- **R594 KEEP-QUALITY: FEWER-RUN-OVERS** (rung 1 on 9 of 9 days, rung 2 on 7 of 9). The size is negligible.

| Keep the order one tick behind (rung 1), same count | Kept | Not reached | Stopped there | Run over |
|---|---|---|---|---|
| STEP (FB25 line) | 81.4% | 54.0% | 21.8% | 24.1% |
| FULL | 81.4% | 54.3% | 21.8% | **24.0%** |
| Realized move | 70.3% | 68.3% | 31.7% | 0.0% |

## What it means

1. **The influence of a Binance move is large, fast and one-for-one.** A fair-price move of X cents moves the Predict price about X ticks,
   mostly within 250 ms, and it stays. Where the fair price has moved, we know roughly where the move stops: about X ticks from the best.
2. **Our guard mostly fires without such a move.** 97% of its onsets have no fair-price move of 0.5 c. Those are Predict-local signals
   (book, flow, volatility):
   - half are followed by no move within 2 s;
   - when a move starts it stops at 1 tick 41-44% of the time;
   - nothing in our inputs separates the one-tick moves from the bigger ones much better (AUC ~0.70).
3. **The reset target helps a little; the influence inputs do not add to the Binance inputs.** The keep decision for the order one tick
   behind is about where it was: 24.0% vs 24.1% of kept onsets run over (lower on 9/9 days).

## Next (proposals, the user's call)

- **An influence rule in the replay**, the USER's strategy written in these units:
  - When the fair price has moved X >= 0.5 c against us, pull the prices within about X ticks of the best and keep the deeper ones.
  - When it has not, which is 97% of onsets: pull the best and keep the next price.
  - M 0 / 50 / 250, rule M and U, WS / E2, age split, against the guard ladder and FUTURE-BEST.
- **Step 2's ask side as planned:** keep the old order and add one at the new best. The fair price says where the new best should be.

## Limits

- p_bn's basis and volatility inputs are step 5's. The basis drifted in late October, which may be why the gap's level is uninformative.
- First minute only (240-300 s left).
- Our guard's onsets; consumed days; exploratory.
- AUC ranks onsets; only the replay values them.
- PART A's pass-through compares the move after t with the fair-price move before t; later fair-price moves are not removed.

Outputs:
- `data/predict_fun/research/influence_20261114/`: influence.txt / .json, events_*, onsets_*, pred_*, fit_* / rows_* json, models/, logs.
- Copies (txt / json / logs) in `reviews/influence_20261114/`.
