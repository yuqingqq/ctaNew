# STOP-MODEL input, target and coverage review, input influence, regularization (2026-11-15, session aea5bfb9)

USER: "Before model play, can we first review the model inputs, training target, data coverage, etc, then check the model inputs influence
on our target, check their influence first, then proceed to build the model" / "And check if any inputs need regularization etc".

Scope: the STOP-MODEL (per side and ladder level j = d + 1: is the order d ticks behind the known best run over within H after a cancel
could land; every 50 ms step of each window's first 60 s; H = 72 / 250 / 1000 ms). The v4 build was stopped (03:29:53Z, two folds done) so
that this review comes first. Every decision below uses the development days 10-23..26 only; the five fold test days (10-27, 10-31,
11-01..03) are kept for the refit's evaluation. Coverage is printed for all nine days as counts.

Instruments (all committed before their run; outputs copied to reviews/stop_model_20261114/):

| Step | Code / commit | What |
|---|---|---|
| Data-level review | pf_stop_model_review.py run (d367a28f) | coverage, target, input distributions, univariate and model influence, declared drop rules |
| Independent code review | (agent, read-only) | catalogue of the 85 inputs with clocks and sign conventions; defects D1-D13 |
| Pipeline checks | pf_stop_model_review.py s5check (d1ff4fce) | K1 step-5 recomputation from raw data, K2 future / stale Binance, K3 two clocks |
| Regularization | pf_stop_model_review.py reg (d1ff4fce) | R1 causal normalization of drifting inputs, R2 tails, R3 LightGBM regularization |
| Within-side copies | pf_stop_model_review.py dup (d1965a2b) | the 20 raw copies of adverse-signed inputs; R1's final comparison |
| Loader fix | pf_kp_s4.py + pf_stop_model.py (6340dd32), d12cmp (cafd8cfe) | never read the adjacent days' boundary-hour books (D12); rows regression |

## 1. Inputs (85 in v3.1 / v4 fold 1)

| Group | n | What (clock) |
|---|---|---|
| Binance | 26 | returns over 25 ms-3 s, top-of-book imbalance and its 250 ms change, event age, taker flows over 250 ms-5 s, trade count; raw and adverse-signed (adv_ = x for asks, -x for bids) copies (event time + 3.5 ms) |
| Book sizes | 15 | own levels 1-5 sizes, cumulative size to level 1-3, size within j - 1 ticks of the best, own / opposite L1 log size; all relative to the causal W30 liquidity scale (frames known in the last 30 s) |
| Predict book state | 18 | spread (two clocks), mid, microprice deviation, L1 / L1-3 imbalance (raw and adverse), level gaps g2-g5, settlement-pending flows (known frame) |
| Predict dynamics | 17 | known frame's age, frames in the last 1 s, mid change over 250 ms-5 s (raw and adverse), own / opposite L1 size change over 250 ms, own / opposite best move over 250 ms / 1 s, age of the current own / opposite best level, act_rel (Binance trade activity relative to the window so far - Binance-derived, code review D4) |
| Fair price | 5 | fair-price change over 250 ms / 1 s, gap of the fair price to the Predict mid (adverse), sensitivity, distance from 50 c; NaN until the start price is known (~2.5 s) |
| Time / vol / side | 4 | time in window, time left (= 300 - time in window), sigma (Binance 1 s vol over the 30 min before the window), side |

Causality and quality:
- Model-side inputs (25): future-perturbation falsifier on real rows every rows day (7,094 checks, 0 differences; STOP-MODEL v3).
- Step-5 inputs (60) had no such instrument (code review D1). K1 recomputed 23 of them (13 Binance, 10 Predict) from the raw Binance hour
  files and the day's Predict frames with only events known by t, at 1,200 sampled steps on 10-24 and 10-26: 23 / 23 match the stored rows
  on 100.00% of the steps, NaN agreement 100.00%. Both planted look-aheads fail the same test (Binance known 50 ms early: bn_age 0% match;
  Predict frames without feed delay: pf_age 0%, pf_imb3 21-24%). The rest of step 5 (fair price, pending flows, sigma) uses the same clocks.
- K2: no step uses a Binance event after t (bn_age < 0: 0 on all nine days); no Binance bookTicker / trade hour file is missing (the zero-fill
  path of D2 is never taken); steps with Binance older than 1 s: 0-348 a day; older than 60 s: 0.
- K3: step 5's spread equals the model side's on 100% of rows; its L1 size differs on 0.003-0.12% (two copies of the known-frame clock, D9).
- Integrity (diag Q1): window-side blocks contiguous, labels non-negative and monotone in H, no infinite input, every dev day.
- Missing values: the fair-price inputs 4.7% (masked by design until the start price is known), act_rel 1.6% (the window's first second).
- Day-to-day drift (largest daily-median shift in pooled sd): sens 0.66, pf_nfr1000 0.52, sigma 0.52; every other input <= 0.20.
- Duplicates: spread = pf_spread, time left = 300 - time in window, r_c1 = r_ct1 = r_s1 (exact) -> dropped by the declared rule (|r| > 0.999).
  20 within-side copies (the 18 raw directional inputs with an adv_ copy; pf_pend_bid / ask1000 = own / opp_pend) are invisible to a pooled
  correlation (~0) - tested by refit (section 6).

## 2. Target

y^H_j = [D1_H >= j] on every usable row (v4's population U, adopted by the diag 2026-11-15: v3.1 trained and scored only rows where the
best had not already moved j ticks before the cancel could land; a policy cannot know that, and on every decision state v3.1 predicted 3.8%
against a 7.2% rate at H72 j1). D1_H = ticks the side's best moved against the side beyond the known best in the state at t + 22 ms or any
frame in (t + 22, t + H]. The code review re-derived the target from the code (boundaries inclusive at t + 22 and t + H, sides correct,
invalid frames skipped, "no frame" = no change; GAP = > 500 ms without any frame in the 1 s window).

| Head | rate 10-23 / 24 / 25 / 26 | 10-26 by time in window 0-5 / 5-15 / 15-30 / 30-60 s |
|---|---|---|
| H72 j1 | 0.071 / 0.061 / 0.060 / 0.072 | 0.104 / 0.074 / 0.068 / 0.068 |
| H250 j1 | 0.139 / 0.122 / 0.119 / 0.143 | 0.194 / 0.146 / 0.136 / 0.137 |
| H1000 j1 | 0.318 / 0.286 / 0.276 / 0.327 | 0.387 / 0.330 / 0.320 / 0.320 |
| H1000 j3 | 0.094 / 0.081 / 0.078 / 0.105 | 0.165 / 0.104 / 0.104 / 0.097 |
| H1000 j6 | 0.024 / 0.023 / 0.024 / 0.027 | 0.055 / 0.027 / 0.026 / 0.022 |

Bids and asks have the same rates (10-26 H1000 j1 0.334 / 0.321). The first 5 s of a window are the riskiest. A move inside the cancel
latency that reverts before t + 22 (D0 >= j, D1_H < j) is 0.003-0.08% of rows, so max(D0, D1_H) (the oracle's label) and D1_H agree.

## 3. Data coverage

| Day | windows used / registered | rows | invalid book | GAP | masked (start price unknown) | boundary windows | sides < 1,200 rows (min) |
|---|---|---|---|---|---|---|---|
| 10-23 | 288 / 288 | 690,552 | 648 | 326 | 32,644 | 0 | 62 (1,144) |
| 10-24 | 288 / 288 | 690,664 | 536 | 206 | 30,508 | 0 | 26 (1,150) |
| 10-25 | 288 / 288 | 690,928 | 272 | 286 | 32,180 | 0 | 36 (1,179) |
| 10-26 | 288 / 288 | 690,478 | 722 | 558 | 29,218 | 0 | 32 (1,065) |
| 10-27 | 288 / 288 | 691,128 | 72 | 1,334 | 30,738 | 0 | 32 (1,184) |
| 10-31 | 288 / 288 | 676,788 | 12 | 3,316 | 31,370 | 6 | 2 (1,194) |
| 11-01 | 288 / 288 | 690,790 | 410 | 2,356 | 30,514 | 0 | 22 (1,149) |
| 11-02 | 288 / 288 | 690,948 | 252 | 650 | 28,864 | 0 | 16 (1,163) |
| 11-03 | 288 / 288 | 691,062 | 138 | 294 | 28,908 | 0 | 18 (1,173) |

No UTC hour without rows on any day. Exclusions are statuses (counted): invalid known book, GAP, 10-31's six windows before 00:30 (sigma's
30 min look back reaches the sealed 10-30). Training draws uniformly per day (every day has equal weight).

## 4. Influence of the inputs on the target (v4 fold-10-27 models: trained on 10-23..25, scored on 10-26)

| Group | SHAP share H72 / H250 / H1000 j1 | log-loss increase when shuffled (x1e-3) H72 / H250 / H1000 | refit without the group |
|---|---|---|---|
| Binance | 27% / 35% / 35% | +59 / +150 / +128 | +10.3% worse, 0 / 9 heads better |
| Book sizes | 31% / 25% / 19% | +39 / +32 / +18 | +0.7% worse, 0 / 9 |
| Predict book state | 16% / 19% / 23% | +16 / +26 / +37 | +0.7% worse, 0 / 9 |
| Predict dynamics | 17% / 12% / 13% | +12 / +12 / +7 | +1.6% worse, 0 / 9 |
| Fair price | 5% / 6% / 7% | +3 / +6 / +4 | +0.6% worse, 0 / 9 |
| Time / vol / side | 4% / 3% / 2% | +0.4 / +1.3 / +1.1 | +0.2% worse, 0 / 9 |

Every group earns its place: removing any one makes every one of the 9 heads (j = 1..3 at every H) worse. The strongest inputs and how they
move the risk (SHAP by input decile, monotone in each case):
- r_s1, the own best level's size relative to the day's liquidity (22% of SHAP at H72 j1; AUC alone 0.84 at H72, 0.70 at 1 s): a thin best
  is run over, a thick one holds.
- adv_d_bnimb250, Binance's top-of-book imbalance turning against our side in the last 250 ms (13-19%): the strongest input at 250 ms and 1 s.
- adv_pf_imb3 / adv_pf_imb1, Predict's own depth imbalance against our side (8-11%).
- own_lvl_age: a best level that has just formed is riskier than an old one.
- adv_bn_imb, adv_bn_r250 / r500 (Binance moving against us), own_d250 (our L1 shrinking), gap_adv (the fair price says the price has further
  to go against us).

No input flips the sign of its relation to the target between dev days (per-day AUC > 0.52 and < 0.48 never at the same head).

Printed flags (no automatic change; the expected sign is a prior, the SHAP direction is conditional on the other inputs):
- Predict's own past mid moves against us (adv_pf_dm250 / 1000 / 5000) LOWER the predicted risk given the book state - mean reversion of
  Predict's mid; alone they raise it (AUC 0.53-0.56). Kept.
- sigma pushes the risk down at H72 / H250 given the rest (alone it raises it, AUC 0.55): it is collinear with sens (which contains 1 / sigma)
  and it drifts by day. Kept raw (section 5).
- The 25 / 50 ms Binance returns lower the 72 ms risk given the 250 ms return (a move too fresh to have reached Predict within 72 ms).

## 5. Regularization

Inputs (R1, causal normalization: an input divided by the median of its |value| over the day's previous <= 12 windows):

| Candidate (drift or audit-v2 PSI flag) | shape dispersion across dev days, pp, raw -> normalized (H72 j1 / H250 j1 / H1000 j1 / H1000 j3) | AUC raw / normalized (H72 j1) | selected |
|---|---|---|---|
| sens | 0.65 -> 0.27 / 1.18 -> 0.55 / 2.25 -> 1.13 / 1.12 -> 0.59 | 0.442 / 0.475 | yes (4 / 4) |
| bn_ntr1000 | 0.68 -> 0.59 / 1.03 -> 0.74 / 1.31 -> 0.70 / 0.75 -> 0.42 | 0.643 / 0.630 | yes (3 / 4) |
| d_bnimb250 | 0.80 -> 0.69 / 1.31 -> 1.05 / 1.29 -> 0.55 / 0.66 -> 0.37 | 0.496 / 0.496 | no (2 / 4) |
| adv_d_bnimb250, bn_r3000, adv_bn_r3000, pf_nfr1000 | normalized is less stable | | no |

Normalization makes sens and bn_ntr1000 more stable across days but weaker; the refit on the final inputs decides (section 6). sigma, the
most drifting input, was left out of R1 by a grouping error in the rule (it sat with the time / side inputs); the v2 dev-day probe had
already compared it raw and day-normalized and found the raw relation more stable (1.4 vs 2.2 pp), so it stays raw.

Tails (R2, 10-26): flows, fair-price changes and pending flows are heavy-tailed (dfv1000 26% of rows beyond 10 IQR, the 250 ms flows 21-22%;
the short Binance returns are mostly exactly 0). Trees are invariant to monotone transforms, so tails matter only through drift (R1): no
clipping.

Model (R3, fold 10-27, 180k rows a day, every head, 10-26 log loss against the pinned parameters):

| Variant | mean log-loss gain | heads better | H72 j6 | H250 j6 | H1000 j1 |
|---|---|---|---|---|---|
| lambda_l2 = 10 | +2.72% | 16 / 18 | 0.00423 -> 0.00360 | 0.01380 -> 0.01254 | 0.50333 -> 0.50346 |
| min_data_in_leaf = 1,000 | +2.12% | 16 / 18 | -> 0.00360 | -> 0.01278 | -> 0.50347 |
| num_leaves = 15 | -0.54% | 3 / 18 | -> 0.00429 | -> 0.01376 | -> 0.50537 |

The pinned parameters (since PULL-DEPTH) overfit the rare deep heads; an L2 penalty fixes that (-9% to -15% log loss at the deep heads) and
leaves the level-1 heads unchanged. ADOPTED by the declared rule: lambda_l2 = 10 (a change of the pinned parameters, flagged).

## 6. Within-side copies and the final normalization decision

Fold 10-27 refits (60k rows a day, target U, the 9 heads j = 1..3 at every H, v3.1 seeds), 10-26 log loss:
- Without the 20 within-side copies (bn_r25..3000, bn_imb, bn_flow250 / 1000 / 5000, pf_micro_dev, pf_imb1 / 3, pf_dm250 / 1000 / 5000,
  d_bnimb250 - each kept as its adverse-signed adv_ version with fside - and pf_pend_bid / ask1000, kept as own / opp_pend): +0.14% mean,
  8 / 9 heads better -> DROPPED (81 -> 61 inputs). The copies only split the same information.
- sens and bn_ntr1000 normalized (R1's selection) against raw on the final 61 inputs: -0.08% mean, 1 / 9 better -> not normalized.

## 7. Code review findings and their status

| # | Finding | Severity | Status |
|---|---|---|---|
| D1 | the future-perturbation falsifier covers only the 25 model-side inputs | material (control) | K1: 23 step-5 inputs recomputed from raw data known by t, 100% match, planted look-aheads fail |
| D2 | a missing Binance hour file would zero returns / flows and stale the fair price | material if a file is missing | K2: no file missing on the nine days, no future event, stale > 1 s on 0-348 steps a day |
| D3 | exact duplicates (r_c1, r_ct1 = r_s1; time left; spread) and 20 within-side copies | minor | exact ones dropped; copies tested (section 6) |
| D4 | act_rel is Binance-derived but grouped with Predict | minor | BN_F fixed (6340dd32); the diag's staleness check and the review's Binance ablation ran without it |
| D5 | inputs change meaning at window start (W30 sees <= 5 s at t_rel 0; act_rel NaN for 1 s; 250 ms differences NaN for 250 ms) | minor | NOT fixed: needs a rows rebuild with a 30 s pre-roll; the model has t_rel, and its early-window calibration is fine (diag Q7) |
| D6 | one 1 s GAP rule for every horizon | minor | NOT fixed: GAP rows are 0.03-0.5% |
| D7 | an empty own side reads as "no move" | minor | 0 such rows on every day |
| D8 | boundary conventions (a frame exactly at t + 22; young orders' cancel lands later) | minor | documented |
| D9 | two copies of the known-frame clock | minor | K3: agree on 99.9-100% of rows |
| D10 | pending flows in venue units (not scaled); log1p for the L1 log sizes | minor | NOT changed (pending flows: drift 0.13 sd, low influence) |
| D11 | the review's group-drop rule could drop fside with the time inputs; EXPECT incomplete | minor | no group was dropped; flags are printed only |
| D12 | the loader read every part file in a day folder, incl. the adjacent days' boundary hours: sealed 10-28 00Z (in 10-27's folder), sealed 10-30 23Z (10-31), forward-block 11-04 00Z (11-03); their frames were filtered out before any row of an included window | process (P11, forward-block rule) | fixed for this model (opt-in filter, skips counted); rows regression on 10-27 / 10-31 / 11-03 (section 8); other scripts still read them (default unchanged); step 5's per-day clock shift was computed over the folder's frames incl. those hours (not rebuilt) |
| D13 | Binance arrays concatenated across hour files without re-sorting | minor (unverified) | K2: no future event on any step; lookbacks can only be stale near hour boundaries |

## 8. Decisions for the build (v4) and what follows

(pending: build)
