# STOP-DEPTH: how far a danger move goes, and how well we can predict where it stops (results, 2026-11-14)

Session ce68ebe0. Step 1 of the plan agreed with the USER:
- "Ok, so we need to check how large is the influence of the signal, then we just need to pull 50c bid, if we estimate it will stop 1c, we
  can rest 49c there and repost 50c ask".
- "Makes sense. Proceed".

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| Script `research/pf_stop_depth.py` + `run_stop_depth.sh`, readings R580-R583 declared | ebec8cd1, before the first run | 04:03-04:22Z, peak ~0.9 GiB |
| `posthoc`: value bins for the Binance return; where moves stop by signal strength (POST HOC) | 4adf3d32 | 04:24Z |

## Why

FUTURE-BEST (the USER's rule: keep an order unless the move is likely to go through its price) is limited by predicting where the move
stops:
- with the realized move it made +$378 a day more than the guard ladder on exactly timed trades;
- with PULL-DEPTH's model +$41-71, and not robust.

## Design

- **Rows:** PULL-DEPTH's guard-on steps (v <= -0.10, valid known book), 9 consumed days, first 60 s: 2,015,322 rows.
  - Statuses: guard value masked (start price unknown, P16 look-ahead) 275,520; invalid book 0; quiet (no new frame in the horizon) 543.
- **Clean target** (fixes the known-frame target and missing-frame zeros flagged in the other session's review), on the venue frames:
  - D0 = how far the price had already moved by t + 22 ms, before our cancel could land. Rows past saving for a threshold are a status.
  - D1 = how far it moves from then to t + 1 s; no new frame means no change.
- **Inputs without fitted models' outputs:**
  - FRONT-PULL's P is dropped (nested-fold leak), and so is the guard value (its fits include 8 of the 9 days).
  - BASE = PULL-DEPTH's other inputs.
  - NEW = BASE + directional features signed against our side, own / opposite L1 size and pending settlements, size within 1-5 ticks of
    the best, and recent best moves on both sides.
- **Models:** LightGBM per threshold, PULL-DEPTH's pinned parameters, leave one day out.
  - OLD = PULL-DEPTH's saved predictions (the model FUTURE-BEST used), on the same rows.
- **Controls passed:**
  - synthetic target and feature cases with three planted errors caught;
  - the feature list order;
  - no held-out row in a fit;
  - every row matched an OLD prediction.

## Results

**1. How far the move goes within 1 s after a guard-on step** (1.83 M steps where nothing had moved yet)

| Move | 0 | 1 tick | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| All | 59.2% | 19.7% | 8.4% | 4.6% | 2.8% | 5.3% |
| Strongest guard values (quintile 1) | 36.0% | 23.3% | 14.1% | 9.0% | 5.7% | 12.0% |
| Weakest guard values (quintile 5) | 71.4% | 15.9% | 5.4% | 2.8% | 1.5% | 3.0% |

A move of at least one tick stops at 1 tick 48.2% of the time, at 2 ticks 20.5%, and goes 3+ ticks 31.2%.

**2. The influence of the signal (post hoc): the Binance move against us sets the size**

| Binance 250 ms return against us | Share of guard-on steps | No move within 1 s | Moves that stop at 1 tick | Moves of 3+ ticks |
|---|---|---|---|---|
| In our favour | 0.8% | 63.0% | 43.4% | 33.0% |
| Zero | **93.2%** | 61.5% | **51.3%** | 28.2% |
| Against, low tercile | 2.0% | 33.4% | 36.5% | 40.3% |
| Against, mid tercile | 2.0% | 22.5% | 27.4% | 50.2% |
| Against, high tercile | 2.0% | **11.5%** | **15.2%** | **68.1%** |

- On the 1 s Binance return the same pattern is weaker. Moves stop at 1 tick 29.6% of the time in the high tercile against us, 52.6% when
  there is no Binance move.
- By guard-value quintile, moves stop at 1 tick 36.3% in the strongest quintile and 51.7-55.6% in the others.

**3. Can a better model tell where it stops?** (out of day; rows not past saving)

| Move at least | OLD | BASE | NEW |
|---|---|---|---|
| 1 tick | 0.721 | 0.719 | **0.725** |
| 2 ticks | 0.731 | 0.728 | **0.733** |
| 3 ticks | 0.736 | 0.732 | **0.738** |
| 4 / 5 ticks | 0.739 / 0.741 | | 0.742 / 0.745 |

- Readings:
  - **R580 PREDICTABLE** (1, 2, 3 ticks);
  - **R581 BETTER** than OLD (each on >= 7 of 9 days);
  - **R582 FEATURES-HELP** (NEW above BASE on >= 7 of 9).
- The gains are tiny: +0.002 to +0.004 AUC.
- NEW is well calibrated: predicted and realized rates agree in every decile.
- The top inputs are the Binance 100-500 ms returns and imbalance change against us, volatility, the L1 / L3 imbalance against us, the
  size at the best, and the size within 2 ticks.

**4. Does it change the decision to keep the order one tick behind (49c when 50c is the best)?**

Kept steps at FUTURE-BEST's FB25 keep count (rung 1, 1.96 M steps):

| | Kept | Not reached | Stopped at 49c (the good case) | Run over |
|---|---|---|---|---|
| OLD | 71.9% | 65.9% | 19.4% | 14.7% |
| NEW (same count) | 71.9% | 65.8% | 19.6% | **14.6%** |
| Realized move (perfect foresight) | 77.1% | 71.7% | **28.3%** | **0.0%** |

- **R583 NOT-SHOWN:** NEW runs over less on 6 of 9 days at rung 1 and 5 of 9 at rung 2. No real difference.

## What it means

1. **After our guard fires, most of the time nothing moves.** When something does, it is a one-tick move about half the time. Keeping the
   order one tick behind is right more often than not.
2. **The size of the move follows the Binance move against us.** But in 93% of guard-on steps Binance has not moved in the last 250 ms.
   There the outcome depends on the Predict book itself, and our inputs separate it only moderately well (AUC ~0.73).
3. **A cleaner, richer model does not move the decision:** 14.6% against 14.7% of kept orders run over. The information that would tell a
   one-tick move from a bigger one is not in the inputs we have. With these inputs, the replay gain of keeping the next price stays at
   FUTURE-BEST's (+$41-71 a day at M 50 on exact trades, negative at M 250).

## Next (proposals, the user's call)

- **Step 2 as planned:** the whole strategy in one replay.
  - Bid side: pull the price the move goes through, keep the next one with NEW.
  - Ask side: keep the old order and add one at the new best with the fast path.
  - Step 1 says the bid half will look like FUTURE-BEST. What is new is the ask half.
- **Or first, more information for the stop:**
  - features of the move as it unfolds: the first taker order's size, how much of the next level it took, the trades in the last 100 ms;
  - another venue's lead (Polymarket);
  - then re-run this check.

## Limits

- Our guard's population.
- PULL-DEPTH's 1 s horizon.
- Consumed days; the inputs were chosen after PULL-DEPTH / FUTURE-BEST on these days.
- AUC ranks rows; only the replay values them.
- The Binance bins and the stop-by-strength tables are post hoc and descriptive.

Outputs:
- `data/predict_fun/research/stop_depth_20261114/`: stop_depth.txt / .json, stop_depth_posthoc.txt, rows_* / pred_* parquet, fit_* / rows_*
  json, models/, logs.
- Copies (txt / json / logs) in `reviews/stop_depth_20261114/`.
