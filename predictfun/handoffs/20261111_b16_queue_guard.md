# B16 stage 1: the guard trained on what our own order earns (2026-11-11)

Session ce68ebe0. USER: "the model may not predict the right action to take ... we haven't built the strong model". Plan:
`handoffs/20261111_match_top_makers_plan.md` (step 2). Scripts committed before the first full run: `research/pf_b16_queue_label.py`,
`research/pf_b16_queue_guard.py`, `research/run_b16.sh` (18515190). The report was rerun after one fix that came before any table
was printed (167fdfe8: a column-name clash in the report's join). Output as of 2026-11-11T13:37Z. Exploratory, 9 consumed days.

## Answer

- **The new target alone changes nothing.** Arm qa (the current 32 features, trained on our order's value instead of sweeps only)
  picks moments of the same quality as the live guard (R160 NO-CHANGE, -0.0001 c/sh). In the replay its sums are slightly positive
  at every horizon, inside the noise (R161 / R162 MIXED). This was expected from the label: even counting fills from trades at our
  price, our 5-share order at the back of a ~400-share queue is filled almost only by sweeps. The new label fills 13% of moments
  (10% by sweeps), and its mean is -0.225 against the old label's -0.228 c/sh.
- **The new target with the pressure features (arm qb) is better at 1 s and worse at 60 s and at settlement.**
  - At +1 s it is +$72 over 9 days at J (better on all 9 days; per-day interval [+3.9, +12.1]) and +$19 at X (7 / 9): R162 BETTER,
    the first variant to pass a declared replay rule.
  - Settlement $ is -$148 (J) and -$183 (X), and +60 s $ is -$67 / -$191 (4 / 9 and 3 / 9 days better). It trades 7% more (fills
    5,278 to 5,662 at J).
  - The loss is not inventory. Mean |net| at the stop is the same (6.56 against 6.67 sh), and so is the carry from the stop to
    settlement. It is the trading itself: mark-to-market at the stop is $622 against $758 (J) and $309 against $495 (X).
  - Its extra fills are good 1 s later and worth less by 60 s. The model learned the label's 1 s horizon.
- **Not promoted.** Keep the live guard. The finding to carry forward is that the training horizon decides the sign: a 1 s target
  optimises 1 s fills at the expense of the window's P&L.

## Design (declared in `pf_b16_queue_guard.py`)

- **Label** (`grid_ql.parquet`, 1,071,151 grid rows).
  - What a 5-share order of ours, RESTING at the back of the queue at the known best price, earns from tau + 22 to tau + 522 ms.
  - Fill rule M, the v3 engine's own: the size ahead is capped by the size shown in every frame, so a thinning level moves us up.
    Trades at our price use up the queue ahead, then fill us; a trade through our price fills the rest.
  - Value: the 1 s markout (the 5 s markout is kept beside it). The resting order under rule L (no credit for cancels ahead) and a
    fresh post are kept beside as well.
  - The design was set after three one-day checks on 11-02, with no guard fit. A fresh post would score fast moves through our price
    as refused posts instead of bad fills; rule L cannot see the fills that follow thinning.
- **Guards:** the live guard's recipe and B14's verified pipeline (LODO on the 8 read days, 11-03 from the fit on all 8, threshold
  = the 60% quantile, lines = per-minute p90 / p95 mapped onto the frozen lines, frozen v3 engine at J and X).
- **Benchmark (REF):** B14's replays of the live guard, which reproduced the stored v3 replays exactly.

Controls, all passed:
- Label: K1, the new label equals the live guard's label on all 438,951 sweep rows; K2, 10 synthetic cases; K3, three planted
  errors (no queue, cancels always credited, frame cap ignored) each caught.
- Guards: every grid row aligned; the falsifier's permutation keeps each day's label multiset; the map's C2 / C3 (frozen lines
  reproduced; each arm's map cuts exactly at its own lines; planted p90 + 0.001 caught); every replay's env as declared.
- Falsifier: arm qb refit on labels shuffled within each day reads WORSE (-0.474 c/sh [-0.737, -0.212]), so the referee can see a
  worse guard and R160 is not void.

## Tables

### R160 / R163: the moments each guard lets quote (grid rows, out of fold; c/sh, bid + ask pooled)

| Day | qa - REF (new label) | qb - REF (new label) | falsifier - REF |
|---|---|---|---|
| 10-23 | +0.0008 | +0.0023 | -0.660 |
| 10-24 | -0.0010 | +0.0030 | -0.283 |
| 10-25 | +0.0045 | +0.0181 | -0.205 |
| 10-26 | -0.0039 | -0.0079 | -0.870 |
| 10-27 | +0.0030 | +0.0103 | -0.774 |
| 10-31 | -0.0012 | +0.0036 | -0.125 |
| 11-01 | -0.0024 | +0.0041 | -0.141 |
| 11-02 | -0.0009 | -0.0068 | -0.738 |
| mean [95%, t(7)] | -0.0001 [-0.0025, +0.0022] | +0.0033 [-0.0037, +0.0104] | -0.474 [-0.737, -0.212] |
| reading | R160 NO-CHANGE | R160 NO-CHANGE (6 / 8 positive) | WORSE |

- Old label as referee (R163): qa -0.0017 [-0.0038, +0.0004] KEEPS; qb +0.0018 [-0.0046, +0.0082] KEEPS.
- 5 s new label: qa +0.0001, qb -0.0001.
- Quoted rows filled: REF 4.57%, qa 5.18%, qb 5.05%; by a sweep 3.21% / 3.58% / 3.44%.

### R161 / R162: v3 replay, arm - REF over 9 days ($; days better)

| Arm | settle J | settle X | +1 s J | +1 s X | +60 s J | +60 s X |
|---|---|---|---|---|---|---|
| qa | +93.03 (6) | +12.60 (5) | +9.12 (5) | +14.56 (7) | +111.43 (7) | +17.54 (7) |
| qb | -148.08 (4) | -182.88 (4) | +71.97 (9) | +18.55 (7) | -66.61 (4) | -191.41 (3) |
| readings | qa R161 MIXED, qb R161 MIXED | | qa R162 MIXED, qb R162 BETTER | | descriptive | |

9-day totals:

| | settle J | settle X | +1 s J | +1 s X | +60 s J | +60 s X |
|---|---|---|---|---|---|---|
| REF | 906 | 593 | 541 | 409 | 565 | 513 |
| qa | 999 | 605 | 550 | 424 | 676 | 531 |
| qb | 758 | 410 | 613 | 428 | 498 | 322 |

- Fills: REF 5,278 / 5,025 (J / X), qa 5,369 / 5,102, qb 5,662 / 5,302.
- Mark-to-market at the stop (J / X): REF 758 / 495, qa 752 / 507, qb 622 / 309.
- Carry from the stop to settlement: REF 148 / 98, qa 247 / 99, qb 136 / 101.
- Mean |net| at the stop: REF 6.67 / 5.94, qa 6.73 / 6.12, qb 6.56 / 5.83 sh.

## Limits

- Inherited from B14: the 9 days were read before, and the 13 features were chosen after 11-08 and tested on these days. This is
  exploratory, not a validation.
- The replay's fill model is not validated. The label uses the same fill rule M, which credits every shrink of the level ahead of
  us (an upper bound on how fast we move up).
- Every grid row is an order at the back; an order that rests longer is nearer the front.
- R160 scores moments on the grid, while the replay's value comes through the engine's own path (queue place, inventory, stops).

## Next (proposals, each a new declared candidate; not run)

- **B16c: the same pipeline with the label's horizon matched to the window's P&L.** The 60 s markout, or the 1 s value plus the
  inventory's value at the stop. B16 shows a 1 s target buys 1 s fills at the expense of 60 s and settlement. This is a choice made
  after seeing B16, so it is in-sample by construction and needs untouched days before any use.
- **Stage 2:** the second layer at v3's own decision times, with our order's state among its inputs.
- **Either way:** a candidate that wins here must be frozen and tested on >= 5 untouched days before live.

Outputs:
- `data/predict_fun/research/b16_20261111/`: report.json / .txt, build / map reports, oof_qa / oof_qb.parquet,
  replay_days_qa / qb.csv, guard_qa / qb_bid / ask.txt.
- `step4/badfill_info_tx_tk/grid_ql.parquet` and its report.
- step5/<day>/preds_qa60(k, km) and preds_qb60(k, km).
- step4 ladder dirs `..._preds_qa60km_b16aJ / X` and `..._preds_qb60km_b16bJ / X`.

Review copies: `reviews/b16_20261111/` (SHA256SUMS).
