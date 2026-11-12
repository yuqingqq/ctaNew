# Keep-or-pull S1: the aligned label and its perfect-foresight ceiling on 11-08 (2026-11-12)

Session ce68ebe0. USER: "Go" (design `handoffs/20261112_keep_pull_model_design.md`, aa727160). Script `research/pf_kp_s1.py`
(b8603604, committed before its first run): B21's simulator plus a capture of our order's state at every step.
- Run 03:26-03:36Z in the claude slot: 574 s, peak 1.0 GiB.
- 11-08 consumed cohort, 220 windows, first 60 s, one 5-sh order per side. Point estimates.

Controls passed:
- K1: the never-pull path and v3 reproduce all 104 committed B21 cells.
- K2 / K3: the fork cases have the known answers, and all three planted errors break them.
- K4: on 20,587 sampled steps, the pull fork equals B21's rest lane pulled once at that step, fill for fill, and that path equals the
  never-pull path before the pull. 0 differ.
- Every window had frames and an outcome. Truncated rows are counted per horizon.

## Answer: GATE-PASS. The target can pay, and the prize is large

- **R251 GATE-PASS.** A perfect keep-or-pull decision makes resting by default positive under every fill clock:

  | Fill clock | c/sh at 1 s | $/day at 1 s |
  |---|---|---|
  | 0 ms | +0.82 | +137.84 |
  | 50 ms | +0.70 | +119.50 |
  | 250 ms | +0.42 | +72.86 |

  - That is against never pulling (-1.39 / -1.50 / -1.92 c/sh) and v3 (+0.31 / -0.05 / -1.75).
  - R252: it beats v3 in 6 of 6 $/day cells.
  - It pulls 8.9 times per side-minute and keeps 3,887 of never-pull's 6,143 fills (M 50).
- **Pulling is usually the wrong action.** At the 50 ms clock, H 5 s and 1 s markout:
  - 12% of resting moments favour pulling (Delta < 0, mean -1.52 c/sh);
  - 23% favour keeping (mean +1.51);
  - 65% are 0 (no fill on either path within 5 s).
  - On average keeping beats pulling (+0.16 c/sh). The pull path's re-entry at the back is worse than staying: Q_keep -1.10
    against Q_pull -1.25.
  - This holds under every fill clock and both queue rules (mean +0.13 to +0.16 at H 5 s).
- **The harm is concentrated, so a selective pull rule has a target worth learning.** The model must find the 12% of moments
  where pulling is right, without pulling in the other 88%.
- **The existing signals point the right way but are diluted** (post hoc, M 50; mean Delta / share of pull-moments):

  | Condition | mean Delta | share of moments where pulling is right |
  |---|---|---|
  | Binance book shift on | -0.50 | 24.5% |
  | Binance burst on | -0.20 | 18.3% |
  | Guard strongly negative on | +0.03 | 16.1% |
  | Thinning on | +0.11 | 15.6% |
  | Our side thin | +0.12 | 15.4% |
  | No signal (off) | +0.17 to +0.20 | 10.7% to 12.0% |
  | Guard value, lowest quintile | +0.01 | 16.9% |
  | Guard value, quintiles 3-5 | about +0.20 | 10% to 11% |
  | Order younger than 250 ms | -0.08 | 11.6% |
  | Order older than 1 s | about +0.20 | 12% to 13% |

  - Every signal raises the share of moments where pulling is right, but only Binance's two flip the mean.
  - That explains B21: pulling on any one of them pulls mostly in moments where keeping was right.

## What it means

- The aligned target is not empty. Perfect keep-or-pull decisions would earn about +$120/day on this partial day at the 50 ms
  clock, against v3's -$0.58.
- The ceiling uses perfect foresight. It bounds the prize; it does not show that our data contains the information. That is S4's
  test: an out-of-fold model on the 9 replay days, scored by simulating its policy.
- Next by the design: S2, port the simulator's inputs to the 9 replay days; then S3, labels and inputs there (pilot day first for
  memory and time); then S4.

## Limits

- One consumed partial day, first minute only.
- The replay fill model (rule M) is not validated (B1a).
- Our order is not in the public book.
- One order per side.
- CAP10 / RO60 / MICRO / G4 / Binance hard guards are not modelled.
- 65% of rows have Delta = 0. The model's decision matters only where fills happen within 5 s.

Outputs:
- `data/predict_fun/research/kp_s1_20261112/`: kp_s1.json / .txt, run.log, and labels_s1.parquet, 2.0 M rows over 4 configs, kept
  out of git.
- Copies of the json and txt in `reviews/kp_s1_20261112/`.
