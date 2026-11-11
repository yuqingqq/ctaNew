# B13a: a guard model that emphasises book pressure and flow (2026-11-11)

As of 2026-11-11 07:08Z (session ce68ebe0). USER "We pick some days data to train and validate on other days, to use
the imbalance features and flow features ... need to emphasize on these features". Script
`research/pf_b13a_pressure_model.py`, committed and pushed before its run (4b98b043; output 07:08Z). Output and the six
fold models in `data/predict_fun/research/b13_20261111/`, report copies in `reviews/b13a_20261111/`.

## Design

The guard's price model: LightGBM predicting Predict's mid change over the next 1 s (y1000, clipped to +-5 c), with the
pinned step-5 settings (pf_model_spec: 300 rounds, 31 leaves, min 500 rows per leaf). BASE = the current 32 features.
EMPH = BASE + 13 engineered pressure features from trailing data: changes of Predict's L1 imbalance and of Binance's
best bid / ask imbalance over 250 ms and 1 s; a thin-side flag; each best level's size against its trailing 30 s mean,
and the thinner side's; Binance trades in the last second against their trailing mean; Predict imbalance x Binance
imbalance, imbalance x Predict's last 250 ms move, Binance imbalance x activity, Binance flow x imbalance.

Days: the 9 allowed days with step-5 rows (10-23, 10-24, 10-25, 10-26, 10-27, 10-31, 11-01, 11-02, 11-03). No protected
or later day; 11-08 (where the idea came from) is not used. Blocked 3-fold cross-validation by day: each fold trains on
six days (every 4th row, t < 240 s) and predicts the other three (every 2nd row), so every day is out of sample once.
Readings R90-R92 in the script header; 95% intervals over the 9 days (t, 8 df). Controls passed (features on a planted
frame, fold split, a planted signal only in an engineered feature -> IMPROVES, none -> NO-CHANGE, label rules).

## Results

| Day | fold | R2 BASE | R2 EMPH | dR2 | R2 on pressure rows BASE / EMPH |
|---|---|---|---|---|---|
| 10-23 | A | 0.213 | 0.245 | +0.032 | 0.312 / 0.341 |
| 10-24 | A | 0.172 | 0.202 | +0.029 | 0.254 / 0.281 |
| 10-25 | A | 0.168 | 0.204 | +0.035 | 0.261 / 0.295 |
| 10-26 | B | 0.208 | 0.234 | +0.026 | 0.306 / 0.329 |
| 10-27 | B | 0.219 | 0.248 | +0.029 | 0.333 / 0.359 |
| 10-31 | B | 0.175 | 0.189 | +0.015 | 0.299 / 0.316 |
| 11-01 | C | 0.155 | 0.177 | +0.022 | 0.226 / 0.249 |
| 11-02 | C | 0.179 | 0.205 | +0.026 | 0.249 / 0.273 |
| 11-03 | C | 0.185 | 0.213 | +0.029 | 0.267 / 0.293 |

- dR2, all rows: +0.027 [+0.023, +0.032], 9 of 9 days positive. Pressure rows (|L1 imbalance| >= 0.6 or Binance trades
  in the day's top decile, about half the rows): +0.025 [+0.022, +0.029], 9 of 9.
- Correlation of the prediction error with Predict's L1 imbalance: 0.015 -> 0.011; with Binance's imbalance: 0.020 ->
  0.015 (mean absolute over days).
- Gain: pressure features carry 52-54% of BASE's gain; the 13 engineered ones 26-31% of EMPH's. The largest is the
  change of Binance's best bid / ask imbalance over the last 250 ms (16.7% of EMPH's gain in fold C), then its 1 s change
  (4.2%), Binance activity against normal (2.6%) and Binance imbalance x activity (2.1%).

| Reading | Result |
|---|---|
| R90 skill | IMPROVES (+0.027 [+0.023, +0.032], 9/9) |
| R91 pressure rows | IMPROVES-UNDER-PRESSURE (+0.025 [+0.022, +0.029], 9/9) |
| R92 residual | NOT-ABSORBED (the error's link to imbalance falls by about a quarter, not half) |

## What it says

1. The current model under-uses pressure. Adding how the books' imbalance is changing (not only its level), how
   depleted the best levels are and how busy Binance is raises out-of-sample R2 from about 0.18 to 0.21 on every one of
   the 9 days, about 15% more explained variance.
2. The most useful single addition is the change of Binance's top-of-book imbalance over the last 250 ms, the same
   signal that leans the top makers' way before they take (B12f).
3. Some dependence on imbalance remains in the errors (R92): the emphasis helps but does not exhaust it.

## What this changes

- A better 1 s prediction is not yet a better guard: the guard maps predictions to per-side values against the PM90
  line. B13b reads the EMPH predictions against the top makers' behaviour on 11-08; turning EMPH into a guard needs its
  own value mapping and a test on untouched days (at least 5) before any live use.
- The validation days here (all 9) were read before by earlier research; neither model is tuned on them here, and the
  comparison is like for like (same folds, same settings).

## Limits

One target (1 s); every 4th row for training on one core; 9 days with a day-clustered interval; the days were in-sample
for the current model's design (both models are refit per fold).
