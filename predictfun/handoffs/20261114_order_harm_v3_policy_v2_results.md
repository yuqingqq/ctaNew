# ORDER-HARM v3 + POLICY v2: keep-vs-cancel with a scheduled re-post (2026-11-14, session ce68ebe0)

**Verdict (computed by the report code): NOT-ACCEPTED for both arms.**

- The model ranks keep vs pull out of sample (R661 SKILL: Spearman IC > 0 on 5 / 5 test days, pooled +0.137).
- That ranking does not carry over to the whole portfolio (R652 REJECT for both arms at every cap):
  - OHQ0 (keep iff A_hat > 0) beats the FB25 keep rule at M 50 and M 250, but not at M 0.
  - It never beats pulling everything (LDG), apart from M 50 when young fills are counted, and it loses $482-667 a day to LDG at M 250 (0 / 5 days).
  - At FB25's keep share (OHQM), the model's choice of which orders to keep is worse than FB25's in all 12 conditions.
- More good at-best fills (R655 MORE), but fewer older orders kept through the guard (R654 FEWER), so R656 GAP-EVIDENCE is NOT-YET.

## Setup

| Item | Value |
|---|---|
| Code | c5f4557c: `pf_order_harm_v3.py` (labels, nested chronological fit), `pf_order_harm_policy_v2.py` (parity, replay, report), `pf_order_harm_client_lane.py`; drivers `run_order_harm_v3.sh`, `run_order_harm_policy_v2.sh`; post hoc `pf_order_harm_policy_v2_posthoc.py` (9ee5cc1b) |
| Run | claude slot; 12:04:15Z start, step 1 ALL DONE 13:19:08Z, step 2 ALL DONE 14:27:35Z; post hoc 14:32Z |
| Contract | `order-harm-client-50ms-terminal-tail-v2`. Decisions every 50 ms. Re-post at the first 50 ms step after the guard clears (repost_ms 0, no timer). Nothing is posted while the guard is on |
| Strategy (C11) | The JOINT-LADDER v3 client lane: both sides, rungs 0-5 x 30 sh, one shared position, the guard trigger on every rung. This is NOT the frozen pm_ladder v3 |
| Days (C8) | Labels on the 9 consumed replay days (10-23..27, 10-31, 11-01..03). Chronological folds test on 10-27, 10-31, 11-01, 11-02, 11-03, all consumed: development only |
| Arms | LDG = pull every rung at the guard. FB25 = the PULL-DEPTH keep rule. FB25C = FB25 at 50 ms. OFB2C = realized-move ceiling. OHQ0 = keep iff A_hat > 0. OHQM = keep share matched to FB25 on the calibration day |
| Caveats | Ours is simulated (C9). Replay rule M; rule L was not run (C5). Volumes and $ are inflated (C4). No number is validated at the order level (C10) |

## Controls

All passed. A failure would have stopped the output.

- **Synthetic checks:**
  - 9 client50 unit tests.
  - ORDER-HARM v3: 6 / 6 planted errors caught, including the shared-chain leak and the fixed timer.
  - POLICY v2: 7 / 7 planted errors caught.
  - Every earlier suite passed.
- **Labels (9 days):**
  - 9,068,348 decision rows.
  - Look-ahead: 0 of 453,421 checks differ.
  - The pull branch is constant per episode: 0 of 43,932 checks differ.
  - Recorder 0 differ, partition 0 bad.
  - No input at onset: 0. Forced pulls: 0.
  - Statuses: 62,136 of 1,518,411 rung-episodes censored with the 2 s tail, 78,938 with the 5 s tail.
- **Replay (5 test days):**
  - Input parity: 2,769,679 matched decisions (of 4,159,895 whose key exists), with 0 inputs and 0 predictions differing.
  - Look-ahead falsifier: 0 of 251,744 differ.
  - Engine regression: LDG with the default 250 ms timer equals frozen v3's saved fills row for row.
  - FB25 = FB25C at 50 ms: R651 is 0.0 in every cell.
  - Conservation: 0 violations on 103,680 runs.
  - Terminal cash / inventory reconciliation: 103,680 of 103,680 runs match.
  - Max true exposure is at or below the cap: 300 / 600 / 1,500 sh at hard 10 / 20 / 50; 1,823 sh with no cap.
  - Fills without a 1 s value: 0.
- **Post hoc reading:** 72 / 72 PART 1 cells reconcile; the falsifier (young fills kept) differs on 72.
- **Memory:** peak 1,769 MiB (fit, test 11-03).

## Labels and folds (step 1)

**Labels by rung**, in cents per share of the 30-sh order, first decision of each rung-episode:

| Rung | Rows | Mean R | Keep-all minus pull (KAV - Pe) | Share KAV > Pe | Mean P_e | Continuation term (R660, fold mean) |
|---|---|---|---|---|---|---|
| 0 | 250,760 | -0.184 | -0.106 | 0.182 | -0.496 | +0.034 |
| 1 | 252,407 | -0.140 | -0.119 | 0.120 | -0.356 | +0.019 |
| 2 | 253,205 | -0.110 | -0.125 | 0.083 | -0.253 | +0.010 |
| 3 | 253,623 | -0.086 | -0.119 | 0.063 | -0.182 | +0.004 |
| 4 | 254,117 | -0.069 | -0.113 | 0.050 | -0.132 | +0.000 |
| 5 | 254,299 | -0.050 | -0.089 | 0.056 | -0.088 | +0.001 |

- Keeping to the end of the episode loses to pulling at every rung on average. It is better in only 5-18% of first decisions.
- The option to cancel later is worth most at the front rung.
- With a 5 s tail the keep-all gap narrows at the front (rung 0: -0.066) and widens at the back (rung 5: -0.102).

**Folds (R661)** — train on the earlier days, calibrate on the day before the test day, test:

| Test day | Train | Cal | IC final | IC iteration 0 | Keep share (A_hat > 0) | Flip vs previous iteration | theta_M (FB25 keep share) |
|---|---|---|---|---|---|---|---|
| 10-27 | 10-23..25 | 10-26 | +0.140 | +0.160 | 0.500 | 0.247 | -0.021 (0.785) |
| 10-31 | 10-23..26 | 10-27 | +0.137 | +0.126 | 0.549 | 0.232 | -0.016 (0.807) |
| 11-01 | 10-23..27 | 10-31 | +0.142 | +0.137 | 0.593 | 0.230 | -0.017 (0.832) |
| 11-02 | 10-23..27, 10-31 | 11-01 | +0.128 | +0.146 | 0.605 | 0.183 | -0.017 (0.843) |
| 11-03 | 10-23..27, 10-31, 11-01 | 11-02 | +0.144 | +0.154 | 0.582 | 0.181 | -0.016 (0.803) |

R661: IC > 0 on 5 / 5 days, pooled +0.137 -> SKILL. Watch out:

- **Not converged:** 18-25% of decisions still change sign between iterations 2 and 3 (mean 0.215). Three iterations do not reach a fixed point.
- **Benchmark favours iteration 0:** R661's realized benchmark is the keep-all advantage, which is iteration 0's own target. Iteration 0 scores higher on it in 3 of 5 folds.

**Where the model keeps** (final models, test days pooled):

- **Inputs:** the top input is dfv250, 0.186 of the gain. It is the 250 ms change of the Binance-implied fair price, signed against the order's side. Next:

  | Input | Gain share |
  |---|---|
  | adv_d_bnimb250 | 0.052 |
  | adv_bn_imb | 0.043 |
  | adv_bn_r100 | 0.036 |
  | shown_k | 0.031 |
  | sens | 0.030 |
  | size_better | 0.027 |
  | activity | 0.025 |
  | depth_now | 0.024 |
  | rung | 0.015 |
  | sigma (volatility) | 0.014 |

- **Keep share by level and size ahead of the order:** the decision follows the size at better prices (the cushion in front of the order) and the level:

  | Level | Thin cushion | Mid cushion | Thick cushion |
  |---|---|---|---|
  | 0 | 0.60-0.62 | 0.91-0.94 | 0.92-0.93 |
  | 2 | 0.17-0.18 | 0.54-0.59 | 0.88-0.90 |
  | 5 | 0.07-0.10 | 0.24-0.28 | 0.72-0.74 |

- **Volatility:** the volatility tercile moves any cell by at most 0.07.

## Replay (step 2), headline cap hard 50, 5 test days

**Whole portfolio**: all trades, cash + inventory at 60.2 s, $ a day.

| Clock | Ages | LDG | FB25C | OFB2C (ceiling) | OHQ0 | OHQM |
|---|---|---|---|---|---|---|
| M 0 | all | +0.9 | -4.4 | +581.9 | +3.5 | -54.2 |
| M 0 | w/o young | -8.7 | -49.4 | +524.1 | -49.4 | -119.9 |
| M 50 | all | -171.5 | -295.1 | +358.3 | -165.2 | -355.1 |
| M 50 | w/o young | -176.9 | -297.2 | +376.5 | -197.3 | -383.4 |
| M 250 | all | -2,161.9 | -3,127.1 | -1,704.3 | -2,643.7 | -3,307.2 |
| M 250 | w/o young | -1,709.5 | -2,904.2 | -1,276.7 | -2,376.8 | -3,144.1 |

**R652 conditions for OHQ0** ($ a day difference, days higher of 5; a condition passes when it is higher pooled AND on at least 4 / 5 days):

| Against | Clock | All ages | w/o young |
|---|---|---|---|
| LDG | M 0 | +2.6 (3 / 5) fail | -40.6 (1 / 5) fail |
| LDG | M 50 | +6.4 (4 / 5) pass | -20.4 (3 / 5) fail |
| LDG | M 250 | -481.8 (0 / 5) fail | -667.3 (0 / 5) fail |
| FB25C | M 0 | +7.9 (3 / 5) fail | +0.0 (3 / 5) fail |
| FB25C | M 50 | +129.9 (4 / 5) pass | +99.9 (4 / 5) pass |
| FB25C | M 250 | +483.4 (5 / 5) pass | +527.4 (5 / 5) pass |

- **OHQ0:** 5 of 12 conditions pass -> REJECT.
- **OHQM:** 0 of 12 pass -> REJECT. It is -$55 to -$1,435 a day against LDG and -$50 to -$240 against FB25C, higher on 0-2 of 5 days.
- **Other caps:** the same verdicts at hard 10 / 20 / none.

**R650 WS-timed trades** (a diagnostic only):

- OHQ0 vs LDG: PAYS. +$29.9 (M 0, 5 / 5), +$13.7 (M 50, 5 / 5), -$310.6 (M 250, 0 / 5).
- OHQ0 vs FB25C: WORSE. -$30.6 (M 0, 1 / 5), -$16.1 (M 50, 1 / 5), +$322.4 (M 250, 5 / 5).
- On these trades the comparison with FB25C has the opposite sign from the whole portfolio at M 50 (C15).

**Quotes, preservation and good at-best fills** (R653-R655, M 50, hard 50):

| Arm | Live-quote share of 50 ms steps: all rungs | during the guard | at the best | at the best during the guard | Onset orders alive at the episode's last step | At-best fills >= 100 ms, sh / day | STOP share | STOP c/sh 1 s | THROUGH c/sh 1 s |
|---|---|---|---|---|---|---|---|---|---|
| LDG | 0.608 | 0.098 | 0.418 | 0.076 | 0.000 | 26,555 | 0.436 | +0.359 | -1.120 |
| FB25C | 0.747 | 0.507 | 0.317 | 0.059 | 0.718 | 24,530 | 0.486 | +0.325 | -1.243 |
| OFB2C | 0.733 | 0.505 | 0.381 | 0.151 | 0.728 | 38,666 | 0.658 | +0.877 | -1.182 |
| OHQ0 | 0.694 | 0.326 | 0.336 | 0.100 | 0.489 | 29,293 | 0.506 | +0.342 | -1.207 |
| OHQM | 0.755 | 0.485 | 0.366 | 0.157 | 0.701 | 39,067 | 0.523 | +0.270 | -1.272 |

- **R654 FEWER:** onset-order survival vs FB25C is OHQ0 -0.229 and OHQM -0.017, lower on 0 / 5 days.
- **R655 MORE:** at-best STOP fills are worth more than FB25C's (OHQ0 +$12.0, OHQM +$16.4 a day at M 50) and more than LDG's (+$9.1 / +$13.6).
- **Time at the best:** pulling everything and re-posting at once (LDG) spends the most time at the best (0.418). Kept orders end up behind the moved best: FB25C 0.317, OHQ0 0.336.

**Keep share in the replay**, decisions while resting with the guard on, M 50, by rung 0-5:

| Arm | Overall | Rung 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| FB25 | 0.960 | 0.76 | 0.95 | 0.97 | 0.98 | 0.99 | 0.99 |
| OFB2C | 0.958 | 0.79 | 0.94 | 0.97 | 0.98 | 0.98 | 0.99 |
| OHQ0 | 0.867 | 0.85 | 0.86 | 0.83 | 0.88 | 0.87 | 0.90 |
| OHQM | 0.956 | 0.92 | 0.95 | 0.95 | 0.96 | 0.97 | 0.97 |

The model keeps the front rung more often than FB25 and pulls the deeper rungs more often. The ceiling keeps nearly FB25's share at every rung but chooses different orders.

**Order age and trade source** (C1 / C12), $ a day at 1 s, M 50, hard 50:

| Arm | WS < 100 ms | WS 100-500 ms | WS >= 500 ms | E2 < 100 ms | E2 100-500 ms | E2 >= 500 ms |
|---|---|---|---|---|---|---|
| LDG | +10.1 | +19.0 | +22.6 | -41.8 | -67.2 | -146.3 |
| FB25C | +1.5 | +17.5 | +53.9 | -16.9 | -60.1 | -317.3 |
| OFB2C | -1.2 | +12.7 | +307.0 | -33.2 | -57.9 | -33.3 |
| OHQ0 | +6.6 | +19.6 | +35.8 | -25.8 | -62.1 | -226.4 |
| OHQM | +7.5 | +6.5 | +28.1 | -17.8 | -74.1 | -344.6 |

The losses sit on E2-timed trades against orders >= 500 ms old, i.e. the orders kept through a move. The ceiling keeps those orders only when it is safe (-$33).

**Cancel-pending fills** (C3, post hoc, all trades, 1 s, M 50, hard 50):

| Arm | Share of filled shares | $ a day (all ages) | $ a day (w/o young) |
|---|---|---|---|
| LDG | 9.6% | -77 | -52 |
| FB25C | 10.0% | -141 | -120 |
| OHQ0 | 8.2% | -98 | -81 |
| OHQM | 7.5% | -134 | -122 |
| OFB2C | 6.6% | -81 | -51 |

All caps and clocks are in `order_harm_policy_v2_posthoc.txt`.

**Per day**, whole portfolio all ages, $ a day, M 50, hard 50:

| Day | LDG | FB25C | OFB2C | OHQ0 | OHQM |
|---|---|---|---|---|---|
| 10-27 | +30.3 | -148.0 | +720.5 | +83.5 | -257.7 |
| 10-31 | -205.2 | -169.9 | -4.4 | -148.5 | -325.4 |
| 11-01 | +120.9 | +25.6 | +464.4 | +290.8 | +128.1 |
| 11-02 | -275.6 | -453.8 | +465.5 | -197.0 | -363.2 |
| 11-03 | -528.1 | -729.3 | +145.5 | -854.7 | -957.4 |

OHQ0 beats LDG on 4 of 5 days. On 11-03 it loses $327 to LDG, which makes the pooled margin +$6.

## Reading

1. **What the model learned:** it ranks the right thing (IC +0.137, 5 / 5 days).
   - Its keep share follows the level and the cushion in front of the order.
   - Its top input is the 250 ms fair-price move against the side. The direction of that input's effect is not printed.
   - Volatility adds little.
2. **Not enough to accept:** the ranking does not make the whole portfolio better than pulling everything.
   - Against FB25 it gains at M 50 / M 250 (+$100 to +$527 a day, at least 4 / 5 days). It keeps fewer orders than FB25 at the deeper rungs (0.83-0.90 vs 0.97-0.99).
   - It loses to LDG at M 0 without young fills (-$40.6, 1 / 5) and at M 250 (-$482 to -$667, 0 / 5).
   - At FB25's keep share it picks worse orders than FB25 (OHQM 0 / 12).
3. **Room left:** every non-ceiling arm loses on the whole portfolio at M 50 (LDG -$172 a day) and loses heavily at M 250. The realized-move ceiling makes +$358 at M 50: there is room, and the model captures little of it.
4. **Gap evidence (C17):** more good at-best fills (+$9-16 a day), but less time at the best than LDG, fewer older orders kept through the guard, and no portfolio acceptance. R656 NOT-YET.

**Possible reasons, not tested:**

- The label horizon (the 2 s tail after the guard clears) is short of the 60.2 s portfolio mark.
- The policy iteration is not converged (21% of decisions flip at the last iteration).
- The labels come from the keep-all lane, but the replay's states (positions, caps, re-posts) follow the policy.
- There is no inventory or remaining-size input.

## Files

- **Outputs:**
  - Step 1: `data/predict_fun/research/order_harm_v3_client50_repost0_20261114/` (`order_harm_v3.txt`).
  - Step 2: `data/predict_fun/research/order_harm_policy_v2_client50_repost0_20261114/` (`order_harm_policy_v2.txt`, `order_harm_policy_v2_posthoc.txt`).
- **Review copies:** `predictfun/reviews/order_harm_v3_policy_v2_20261114/` (reports, fold and day receipts, logs, SHA256SUMS).
