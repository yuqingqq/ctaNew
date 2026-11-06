# Freeze receipt: pm_ladder_v2 (P-2026-005, frozen 2026-11-06T09:23Z)

USER 2026-11-06: "it may need to update given a window" ... "Yes we can test this." v2 = pm_ladder_v1 with the guard models, thresholds and
per-minute lines refit every day on a trailing 7-day window. Everything else is v1's (predictfun/freeze/pm_ladder_v1/RECEIPT.md, unchanged
and still in force). The forward race that tests v1 and v2 is declared in predictfun/freeze/forward_20261104_08/RECEIPT.md.

## Identity

- Code commit: 4dee5c195bbe030203ace928a6021aa57ea7977e (branch mm-research), on top of v1's 8d5b02a0 / 38535aed.
- Files (sha256 at the commit):

| file | role | sha256 |
|---|---|---|
| pf_v2_refit.py | daily refit: window, models, c60, lines, guard values | 2e0127a08b3ecb4b3a6452164dd26f2fdc7fd19310dd35505e02c9b2587af61b |
| pf_v2_grid_day.py | training rows (label grid) for days after the 8 read days | d60efd0dd18ea4b143e718e3a01ee33602324f0d24a23cc9ed12f7f55901e756 |
| pf_ladder_v2_build.py | generates the v2 engine from the frozen v1 engine | fe1e518c149a366f8bb7c311a24c85b6455032466b49655935782d5443fc7af9 |
| pf_ladder_fwd_build.py | the shared FORWARD patch | b78fd0e2003ffb2a102f0cd0ea0e5c128b10b8def5392d96ac07782d60c68ea1 |
| pf_step4_ladder_v2.py | v2 replay engine (generated) | 9e1dc0ed441dd3355ba2e55418e29e7a0f849cd7a165e2bfde8acfc64d08688b |
| pf_step4_ladder.py | frozen v1 engine (the source; unchanged) | c682229e4fbe606b217cfa633816bcba67bce701c7e690b4cb8be9aa7ff6e37f |

## What v2 is

- Arm: v1's `MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95`, same rules, same clock.
- Engine: pf_step4_ladder_v2.py is the frozen engine with two anchored changes:
  - the per-minute lines come from PF_LINES_FILE (the day's own file) instead of the fixed step4/minute_lines_t60k.json;
  - the input-day bound FORWARD comes from PF_FORWARD (default the frozen '20261104').
  No strategy rule changes.
- Refit for day T, at T 00Z from complete days only (pf_v2_refit.py):
  - Window: the last 7 admissible UTC days before T. Days 10-28..30 are sealed and never used; the tape starts 10-23.
  - Models: one LightGBM per side on the window's grid rows. Fit, features, label, PARAMS and ROUNDS are v1's. c60[side] is the 60%
    quantile of the in-sample predictions.
  - Lines: v1's rule inside the window. Leave one window day out (fit on the other 6 days, with that fold's own c60) and predict the
    day's 50 ms rows: v = prediction - c60, with the start-known withhold. Per minute, bid and ask pooled: percentiles 90 (pull and
    ladder line) and 95 (size line), plus 50/60/70/80, rounded to 4 dp.
  - Guard values for T: v = the window models on T's rows - c60, with the start-known withhold.
- The window length 7 was set before any result: one weekday/weekend cycle (the 10-23..11-03 quote share at v1's fixed line was 6-8%
  on weekdays and 9-18% on weekends). No other length was run.

## Checks run at the commit (all PASS; each refuses on failure and has a falsifier)

- Fold parity: the refit code with v1's 8 read days and the fold without 10-23 reproduces step5/20261023/preds_t60.parquet exactly.
  Falsifier: c + 1e-6 fails.
- Line parity: the line rule on the 8 read days' out-of-fold values reproduces every key of minute_lines_t60k.json. Falsifier: 7 of
  the days differ.
- Start-known mask derived exactly from each day's preds_t60 / preds_t60k pair.
- Grid parity: pf_v2_grid_day rebuilds 10-31 (sealed-day cut) and 11-02 (preds_c3) equal to grid.parquet in every column. Falsifier:
  a 23 ms cancel latency differs.
- Engine parity: pf_step4_ladder_v2 with v1's lines and values is byte-identical to round i on 11-03 J (fills / win / requests,
  summary). Falsifier: the pm arm differs.
- Consistency: the 11-03 refit (window 10-24..11-02) has the same thresholds as the parity fold (the same 7 days).

## Rehearsal on the week-2 read days (in-sample; not a validation)

The walk-forward refits ran for 10-31 (5-day window), 11-01 (6), 11-02 (7) and 11-03 (7), with v2 replays at J and X
(step4/report_v2_rehearsal.json). Values are mark@stop $ per day, point estimates (4 days, no interval).

This is not a like-for-like race:
- v1's 10-31..11-02 values come from leave-one-day-out models trained on the other 7 read days, including later days.
- v1's lines were cut on all 8 days.
- v2 saw only earlier days.
- v2 itself was proposed after seeing that week 2 was weak.

| day | window | J v2 | J v1 | J control | X v2 | X v1 | X control |
|---|---|---|---|---|---|---|---|
| 10-31 | 5 d | +52.9 | +18.3 | -2.0 | +65.5 | +65.5 | +15.9 |
| 11-01 | 6 d | +48.9 | +29.7 | +12.3 | +10.9 | +4.6 | -16.7 |
| 11-02 | 7 d | +12.6 | -2.3 | -17.4 | -44.4 | -38.4 | -56.5 |
| 11-03 | 7 d | +17.4 | -12.2 | -8.1 | -23.6 | -33.1 | -6.2 |
| sum | | +131.8 | +33.5 | -15.2 | +8.4 | -1.4 | -63.5 |

v2 minus v1 by day:
- J: +34.6, +19.2, +14.9, +29.6.
- X: +0.0, +6.2, -6.0, +9.5. On 10-31, 65.55 vs 65.54 from different fills.

v2's minute-0 pull line and quote share (bid / ask):

| day | line | quote share |
|---|---|---|
| 10-31 | 0.0554 | 12.5% / 11.6% |
| 11-01 | 0.0490 | 14.6% / 11.6% |
| 11-02 | 0.0462 | 6.7% / 8.1% |
| 11-03 | 0.0465 | 6.7% / 6.0% |

v1's fixed line is 0.0492.

## Carried over / open

- v1 known issue (flow-noise splits on bn_flow features, |thr| < 1e-9) applies to every refit model too. Not changed (one change at a time).
- Live: v2 needs a daily refit on the research box (labels need the day's full tape) and a daily delivery of two model files + one lines
  file to the bot. Operational; not part of this freeze; going live is the user's decision.
- Any change to the procedure, window or rules after this receipt is a new version. This file is not edited.
