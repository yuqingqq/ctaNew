# TARGET-TEST: is the move target the right one, and do the rebuilt models help in the replay? (results, 2026-11-14)

Session ce68ebe0. USER: "Yea we rebuild the model, make sure we are on the right target, proceed to test the model and check the performance".

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| Script `research/pf_target_test.py` + `run_target_test.sh`, readings R600-R604 declared | 54408bcd, before the first run | 05:17-05:37Z, ~2 min a day |

## Design

- **Harness:** FUTURE-BEST's, unchanged.
  - The guard ladder: 30 sh, rungs 0-5, each rung an independent order.
  - While the guard is on, an order d ticks behind the best is pulled if P(move >= d + 1) >= tau.
  - An order at the best placed as rung 0 is always pulled, as today.
  - 9 consumed days, first 60 s, fill clocks M 0 / 50 / 250.
- **The right target:** oracle arms with the realized move at 1 s (OFB1 = FUTURE-BEST's OFB), 2 s (OFB2) and 3 s (OFB3).
- **Models, out of day:**

| Arm | Model |
|---|---|
| FB25 | PULL-DEPTH's model (the incumbent) |
| FBN25 | STOP-DEPTH's clean 1 s step model |
| FBF25 / FBF10 | INFLUENCE's onset-trained 2 s model, applied at every guard step |
| IR | The influence rule: X = ceil(fair-price move) ticks if >= 0.5 c against us, else 1 |
| LDG | The reference: pull every rung on the guard |

- **Controls passed:**
  - tau = 0 equals LDG everywhere;
  - LDG, FB25 and OFB1 equal FUTURE-BEST's fills row for row at all three clocks;
  - INFLUENCE's model recomputed at every step equals its saved onset predictions (max difference 0);
  - planted errors caught.
- **Statuses:**
  - 2,290,842 guard steps;
  - steps without a NEW / FULL prediction: 275,520 / 277,568, masked steps mostly; pulled.

## Results ($/day w/o young, 1 s, WS-timed trades)

| Arm | M 0 | M 50 | M 250 | All trades M 50 | Pulls / day | WS c/sh M 50 |
|---|---|---|---|---|---|---|
| LDG | +86 | +54 | -936 | -137 | 77,513 | +0.24 |
| FB25 (old model) | +195 | +124 | -1,670 | -271 | 23,391 | +0.36 |
| FBN25 | +189 | +120 | -1,686 | -273 | 23,318 | +0.34 |
| FBF25 (rebuilt) | +161 | +109 | -1,427 | -185 | 30,497 | +0.40 |
| FBF10 | +101 | +64 | -1,031 | -127 | 56,759 | +0.29 |
| IR | -36 | -219 | -2,257 | -1,011 | 17,417 | -0.32 |
| OFB1 (perfect, 1 s) | +469 | +432 | -613 | +366 | 17,749 | +0.88 |
| OFB2 (perfect, 2 s) | +467 | +435 | -585 | +378 | 20,833 | +0.98 |
| OFB3 (perfect, 3 s) | +449 | +415 | -610 | +351 | 23,641 | +0.99 |

Readings:
- **R600 RIGHT-TARGET: NOT-SHOWN.** OFB2 vs OFB1: -1 at M 0 (3 of 9 days higher), +4 at M 50 (6 of 9).
- **R600b: 2S-BETTER.** The 3 s oracle is below the 2 s one on 0 of 9 days higher.
- **R601: FBF25 vs FB25 NOT-SHOWN.** -34 at M 0 (0 of 9 days higher), -15 at M 50 (3 of 9); +243 at M 250 (9 of 9), because it pulls more.
- **R602: FBF25 vs LDG HELPS** at M 0 / M 50: +75 / +56 (9 / 8 of 9 days).
- **R603: IR HURTS** against LDG and FB25 (0-2 of 9 days higher).
- **R604: every keep arm FLIPS-AT-M250** against LDG (0 of 9 days higher).
- **C1:** the gains are on orders 500 ms or older; young fills are a few dollars.
- **C12:** over all trades at M 50 only FBF10 beats LDG (+$10, 7 of 9). The others lose $49-136 a day on estimated-time trades.

## What it means

1. **The horizon is not the problem.** Knowing the move perfectly over 1 s or 2 s is worth the same, about $380 a day over LDG at M 50. 3 s is worse.
2. **The rebuilt onset model does not improve the replay** over the old model, and the influence rule loses.
   - Kept orders behind the best are run over when Predict-local moves continue.
   - IR's kept orders reached the best and earned -0.36 c/sh there.
3. **Every keep arm turns negative at M 250**, and over all trades. The replay's sign still depends on the fill clock and the trade timing (C2, C12).
4. The USER's three gaps apply here. This replay:
   - pulls the order at the best whenever the guard fires;
   - applies onset-trained models at every guard step;
   - treats the rungs as independent orders, with no shared position.

## Next (agreed with the USER, 2026-11-14)

- **Joint ladder replay:** both sides in one simulation, a shared position, cap = K x order size (v3's soft K = 2; 5; 10; none), cancel latency, re-entry behind the queue.
- **ORDER-HARM label:** per resting order at each decision state, with rung 0 included. Keep vs pull from the moment a cancel would land, where pull follows the policy (stay out, re-join behind the queue). Reported as the probability and severity of harm, good-fill value and queue value.
- **The policy** evaluated in the joint replay.

Outputs:
- `data/predict_fun/research/target_test_20261114/`: target_test.txt / .json, fills_*, score_*, day_*.json, logs.
- Copies (txt / json / logs) in `reviews/target_test_20261114/`.
