# B14 / B15: the pressure guard in v3, and the live soft pull at the order level (2026-11-11)

Session ce68ebe0. USER: "run the prediction guard"; "We are testing the soft pull and new model right?". Both tests were
declared in their scripts and committed before their first runs. Every control passed. Exploratory, on consumed days only.

| Test | Question | Script (commit) | Output as of | Readings |
|---|---|---|---|---|
| B14 | Does B13a's pressure model make a better guard for v3? | `research/pf_b14_pressure_guard.py`, `research/run_b14.sh` (0851282b; two pre-result fixes f602a2b2, 17d30722, see below) | 2026-11-11T11:28Z | R120 NO-CHANGE, R121 MIXED, R122 MIXED |
| B15 | What does the live soft pull (0.01 c band, no cap) do at the order level? | `research/pf_b15_soft_pull_nocap.py` (19691045) | 2026-11-11T11:28:41Z | R130 MIXED, R131 MIXED, R132 n/a |

## Answer

- **Pressure guard: no improvement; keep the current guard.**
  - It does not pick better moments to quote. On the grid rows it lets quote (the top 10% of each minute), the realised
    keep value is the same as the current guard's: +0.0002 c/sh [-0.0057, +0.0061], better on 4 of 8 days.
  - Its overall ranking of moments is slightly better (rank correlation with the keep value 0.230 against 0.213). The
    gain sits away from the pull line, so it does not change the quote decision.
  - In the v3 replay, held-to-settlement $ is +$162 over 9 days at J (6/9 days better) and +$11 at X (5/9). At +1 s
    it is +$24 (7/9) at J and -$27 (5/9) at X. Neither timing clears the declared rule; the daily swings come from
    inventory held to settlement.
  - Beside (not declared): at +60 s the pressure guard is lower at both timings (-$38 at J, -$175 at X over 9 days).
- **Live soft pull (no cap): fewer orders, about the same value.**
  - Orders per side-minute: 38.4 (v3) / 30.3 (100 ms buffer) / 29.5 (no cap) with the JOIN price; 34.1 / 26.9 / 26.2
    with v3's price rule. Dropping the cap removes only about 2.5% more orders than the 100 ms version.
  - Value moves by about a dollar a day either way, with signs that depend on the cell (1 s against 5 s, trade-time
    shift). Fills while kept in the band are few (29 / 18 in 220 windows, first minute), too few to read: v leaves a
    0.01 c band within a step or two, so a keep rarely lasts.
  - So, in this one-day replay, no cap neither hurts nor helps P&L measurably and does what it was built for (churn).

## B14 design (in the script's header)

- The live guard's recipe (`pf_step5_guard_preds.py`): per side LightGBM on the guard grid (8 read days, 1,071,151
  rows; label = the value of keeping an order at the known best through the next exposure), threshold = the 60% quantile
  of in-sample predictions, LODO on read days, 11-03 from the fit on all 8. Lines: per-minute p90 (pull and rung) and
  p95 (size) of the start-known v over the 8 read days.
- EMPH: the same recipe and code path with ALL + the 13 engineered pressure features, computed on `rows_rl_d2_pk` (the
  live clock the grid's features come from; B13a had used `rows`, another clock). Lines by the same rule; the values
  are mapped per minute so that the frozen engine's frozen lines cut exactly at EMPH's own p90 / p95.
- Replay: the frozen v3 engine (`pf_step4_ladder_v3.py`, arm `pf_ladder_v3_build.ARM`), `run_forward_v3.sh`'s env,
  timings J and X, 9 days (10-23..27, 10-31, 11-01..03), PF_PREDS = preds_t60k (REF) or preds_e60km (EMPH).

Controls: C1 the recipe through B14's own code reproduces the live guard's v exactly (11-03 and 10-23) and a threshold
moved by 1e-6 is caught; C2 the line rule reproduces the frozen lines (4 dp); C3 the mapping cuts exactly at EMPH's own
lines on all 9 days and a planted p90 + 0.001 is caught; C4 B13a's planted-frame feature check, every grid row has its
features; C5 row alignment; C6 the REF replays reproduce the stored v3 replays i2J / i2X on all 18 day-timings; C7 the
reading rules on synthetic inputs. R120's falsifier (EMPH refit with the 13 features' rows permuted within each day)
reads NO-CHANGE, so R120 is not void.

Two bugs in B14's own code were caught before any result was written:
- The grid's column list repeated `t_rel` (one of the 32 features). The training matrix was 33 wide and dropped `gap`.
  C1 failed and the run refused at 10:13Z.
- The fix then dropped `t_rel` with the other features and crashed after C1 had passed (10:19Z).

The failed runs' logs are kept in `data/predict_fun/research/logs/b14_failed_C1_20261111T1009Z` and
`b14_failed_crash_20261111T1015Z`. The run that produced the tables started 10:20:53Z at 17d30722. The build's peak RSS
was 1,961 MiB against the 2,048 MiB guard: a rerun with more features needs less memory.

### B14 R120: the moments each guard lets quote (grid rows, out of fold; c/sh, bid + ask pooled)

| Day | BASE (live guard) | EMPH | EMPH - BASE | PERM - BASE (falsifier) |
|---|---|---|---|---|
| 10-23 | +0.011 | +0.016 | +0.0050 | +0.0024 |
| 10-24 | -0.000 | -0.001 | -0.0005 | -0.0032 |
| 10-25 | -0.014 | -0.004 | +0.0099 | +0.0047 |
| 10-26 | +0.036 | +0.022 | -0.0138 | -0.0014 |
| 10-27 | +0.025 | +0.024 | -0.0017 | -0.0035 |
| 10-31 | -0.020 | -0.017 | +0.0027 | -0.0001 |
| 11-01 | -0.018 | -0.014 | +0.0034 | +0.0014 |
| 11-02 | -0.007 | -0.010 | -0.0035 | -0.0025 |
| mean [95%, t(7)] | | | +0.0002 [-0.0057, +0.0061] | -0.0003 [-0.0027, +0.0022] |

Share of grid rows quoted: 9.98% / 9.99% / 9.98% (matched by construction). Bad share (label <= -2 c) on them:
0.62% / 0.59% / 0.64%. Rank correlation of v with the label, mean over days: bid 0.213 / 0.231 / 0.213, ask 0.214 /
0.229 / 0.213.

### B14 R121 / R122: v3 replay, EMPH - REF per day ($)

| Day | settle J | settle X | +1 s J | +1 s X |
|---|---|---|---|---|
| 10-23 | +3.10 | +10.01 | +6.44 | +0.44 |
| 10-24 | +16.28 | -85.70 | -7.52 | -10.53 |
| 10-25 | -28.79 | +23.53 | +1.26 | -1.23 |
| 10-26 | +9.27 | -23.40 | +9.27 | -15.29 |
| 10-27 | +118.39 | +83.73 | +9.08 | +1.06 |
| 10-31 | -36.07 | -30.13 | +4.35 | +0.58 |
| 11-01 | -98.37 | -129.16 | +2.58 | +4.73 |
| 11-02 | +83.38 | +69.33 | +3.52 | +3.50 |
| 11-03 (out of sample) | +94.42 | +92.74 | -4.94 | -10.69 |
| sum | +161.61 | +10.95 | +24.04 | -27.43 |
| days > 0 | 6/9 | 5/9 | 7/9 | 5/9 |
| mean [95%, t(8)] | +17.96 [-35.81, +71.73] | +1.22 [-57.41, +59.85] | +2.67 [-1.76, +7.10] | -3.05 [-8.57, +2.48] |

9-day totals, REF then EMPH: settle J 905.89 / 1,067.50, X 592.88 / 603.83; +1 s J 541.08 / 565.12, X 409.19 / 381.76;
+60 s J 564.51 / 526.75, X 513.16 / 337.70. Fills J 5,278 / 5,498 (shares 36,148 / 37,964), X 5,025 / 5,002 (29,832 /
30,052). Windows positive J 731 / 734, X 630 / 626 of 2,592.

## B15 design (in the script's header)

`pf_review_queue_reach.py` (strategy review D-F, 4e48e570) with one arm added: exit line P90 - 0.01, no cap (the same
two-line state as the 100 ms buffer without its clock). One 5-share touch order per side, first 60 s, 220 windows of
11-08 (0.764 window-days, consumed). Queue rules M (the replay's fill model), L (no credit for cancels ahead) and U
(front of the queue, upper bound); trade time shifted 0 / 50 / 250 ms; prices JOIN (the known best) and V (v3's
MICRO target with G4 / G3). Control: all 60 stored arm-config cells of the 2026-11-10 run reproduced exactly; the
original's self-test, planted errors and real-data controls (i) / (ii) pass.

### B15: primary cells (rule M, trade shift 50 ms)

| Price | Arm | orders / side-min | life p50 / p90 ms | resting % | fills | shares | c/sh 1 s / 5 s | $/day 1 s / 5 s |
|---|---|---|---|---|---|---|---|---|
| JOIN | v3 | 38.35 | 82 / 332 | 11.23 | 274 | 960.9 | -0.046 / +0.495 | -0.58 / +6.23 |
| JOIN | buffer 100 ms | 30.25 | 132 / 532 | 12.99 | 297 | 1,067.9 | -0.062 / +0.439 | -0.86 / +6.14 |
| JOIN | no cap (live) | 29.47 | 132 / 582 | 13.26 | 304 | 1,102.9 | -0.071 / +0.514 | -1.03 / +7.42 |
| V | v3 | 34.07 | 82 / 382 | 10.01 | 183 | 633.2 | +0.101 / +0.894 | +0.84 / +7.41 |
| V | buffer 100 ms | 26.92 | 132 / 532 | 11.53 | 195 | 700.2 | +0.088 / +0.838 | +0.81 / +7.68 |
| V | no cap (live) | 26.23 | 132 / 582 | 11.76 | 199 | 720.2 | +0.069 / +0.940 | +0.65 / +8.86 |

- R130 (no cap - v3, $/day; JOIN then V; shifts 0 / 50 / 250; 1 s, 5 s): +0.42, +4.02, -0.45, +1.18, -2.32, -3.92;
  +0.32, +2.55, -0.19, +1.45, -1.37, -2.23 -> MIXED. At a 250 ms shift the no-cap arm is below v3.
- R131 (no cap - 100 ms buffer, the same cells): -0.13, +1.51, -0.16, +1.28, +1.02, +1.52; -0.13, +1.08, -0.16, +1.18,
  +0.98, +1.26 -> MIXED. It is below the buffer at 1 s for shifts 0 and 50, and above in every other cell.
- R132: hold-band fills of the no-cap arm, 29 (JOIN, -0.045 / +2.061 c/sh) and 18 (V, +0.224 / +2.657): n/a (< 50).
- Queue rules at shift 50, $/day at 1 s / 5 s (v3, buffer, no cap): L -0.70 / 5.46, -1.24 / 5.92, -1.41 / 7.19; U
  117.31 / 170.45, 128.93 / 185.72, 129.18 / 186.46; vL 0.71 / 7.03, 0.47 / 7.17, 0.30 / 8.34; vU 107.69 / 155.25,
  118.56 / 167.29, 118.79 / 169.35.

## Limits

- B14: the 13 features were designed after looking at 11-08 and tested by B13a on these 9 days. This is in-sample for
  the feature choice and is not a validation (that needs at least 5 untouched days). The lines are in-sample on read
  days for both guards (the same rule); only 11-03 is out of sample for both the fit and the lines.
- Both tests use the replay's fill model, which is not validated (B1a: it credits 1.5-2x live's shares), so only the
  differences under one model are read.
- B15 covers one consumed partial day, the first minute (the minute-0 line) and one 5-share order per side. It omits
  v3's sizes, the rung, the Binance guards and CAP10 / RO60. With the JOIN price, the hold band keeps an order one tick
  behind the best, where the live engine pulls once its relaxed quote moves.
- The live engine softens every minute's line; neither test covers minutes 1-3.
- Neither test models the live engine's incremental size.

## Next

- Live: keep the current guard; the soft pull stays as configured (no cap). The first live run measures what matters:
  requests per window, cancels within 100 ms of going live, the `soft_ended` mix, and fills on kept orders.
- Pressure model: not promoted as a guard. Its better mid prediction (B13a) does not move the 90th-percentile quote
  decision. A different use (for example a separate pull trigger on pressure) would be a new candidate, declared first
  and tested on untouched days.

Outputs: `data/predict_fun/research/b14_20261111/` (report.json / .txt, build / map / features reports, oof_grid.parquet,
replay_days.csv, guard_emph_bid / ask.txt); step5/<day>/preds_e60, preds_e60k, preds_e60km; step4 ladder dirs
`..._preds_t60k_b14rJ / b14rX`, `..._preds_e60km_b14eJ / b14eX`; `data/predict_fun/research/b15_20261111/`.
Review copies: `reviews/b14_20261111/`, `reviews/b15_20261111/` (SHA256SUMS).
