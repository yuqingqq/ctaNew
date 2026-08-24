# BE-Belief — walk-forward recalibration of the venue book

**Status: DEVELOPMENT, not decision eligible.** Produced by
`be_belief.py`, implementing `plans/BE_BELIEF_PLAN.md` §12 steps 1–3.

## Receipt

- rows 8,908 over 1,784 windows, 4 UTC day(s): 2026-08-20, 2026-08-21, 2026-08-22, 2026-08-23
- sampling rule `WHOLE_COVERED_POPULATION`
- `days_sampled` ['20260820', '20260821', '20260822', '20260823'] (n=4); `days_declared` ['20260819', '20260820', '20260821', '20260822', '20260823']
- decision times, elapsed s: [30.0, 60.0, 120.0, 180.0, 240.0] (r = 270, 240, 180, 120, 60)
- up-rate 0.5043 per row, 0.5045 per window
- refused: {'no_state_at_decision_time': 12}

One outcome per window is shared by every decision time in it, so every n here is inflated ~5× and intervals cluster on window, then day.

## Calibration — at the mid AND at the executable prices

`spread` here is POOLED ACROSS COINS and is therefore a pooling artefact in exactly the way U8 named: ATM spread is **1 tick** on btc/eth and 3–7 ticks on the thin coins, so a pooled figure reports neither. Read it as a mixture, never as a venue spread.

| bucket | rows | windows | mid | bid | ask | realised | gap vs mid | buy Up at ask | sell Up at bid | spread | age s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0-0.1 | 1141 | 678 | 0.031 | 0.025 | 0.036 | 0.039 | +0.008 | +0.003 | -0.013 | 0.010 | 0.0 |
| 0.1-0.2 | 590 | 466 | 0.149 | 0.144 | 0.155 | 0.108 | -0.041 | -0.047 | +0.035 | 0.011 | 0.0 |
| 0.2-0.3 | 721 | 549 | 0.250 | 0.244 | 0.256 | 0.225 | -0.026 | -0.031 | +0.020 | 0.012 | 0.0 |
| 0.3-0.4 | 878 | 663 | 0.350 | 0.344 | 0.355 | 0.337 | -0.013 | -0.018 | +0.007 | 0.011 | 0.1 |
| 0.4-0.5 | 949 | 714 | 0.449 | 0.444 | 0.455 | 0.413 | -0.036 | -0.042 | +0.031 | 0.011 | 0.2 |
| 0.5-0.6 | 1008 | 730 | 0.547 | 0.542 | 0.553 | 0.534 | -0.014 | -0.019 | +0.008 | 0.011 | 1.7 |
| 0.6-0.7 | 891 | 683 | 0.649 | 0.644 | 0.655 | 0.622 | -0.028 | -0.033 | +0.022 | 0.012 | 0.0 |
| 0.7-0.8 | 756 | 572 | 0.750 | 0.744 | 0.756 | 0.751 | +0.001 | -0.004 | -0.007 | 0.011 | 0.0 |
| 0.8-0.9 | 653 | 508 | 0.849 | 0.843 | 0.855 | 0.896 | +0.047 | +0.041 | -0.053 | 0.012 | 0.0 |
| 0.9-1.0 | 1321 | 728 | 0.969 | 0.964 | 0.974 | 0.976 | +0.007 | +0.002 | -0.012 | 0.010 | 0.0 |

## In-sample `a` / `b` — DIAGNOSTIC, never deployed

`a` is the drift channel: estimated always, deployed never (§4.3).

| sample | n | a | b |
|---|---:|---:|---:|
| core |logit m| ≤ 3 | 7,123 | -0.062 | 1.083 |
| extreme | 1,785 | +0.192 | 1.106 |
| all rows | 8,908 | -0.059 | 1.087 |

| day | n | a | b |
|---|---:|---:|---:|
| 2026-08-20 | 607 | -0.002 | 0.827 |
| 2026-08-21 | 2,237 | +0.058 | 0.973 |
| 2026-08-22 | 2,337 | -0.119 | 1.147 |
| 2026-08-23 | 1,942 | -0.176 | 1.271 |

| r (s remaining) | n | a | b |
|---|---:|---:|---:|
| 270 | 1,781 | -0.014 | 1.158 |
| 240 | 1,777 | -0.043 | 1.199 |
| 180 | 1,665 | -0.081 | 1.060 |
| 120 | 1,279 | -0.143 | 1.037 |
| 60 | 621 | -0.121 | 0.980 |

## Walk-forward — fit on days < d, score day d

3 scored day(s) of 4 present.

| day | n test | model | log-loss | Δ | Brier | Δ |
|---|---:|---|---:|---:|---:|---:|
| 2026-08-21 | 2,237 | `raw_book` | 0.5667 | +0.0000 | 0.1938 | +0.0000 |
| 2026-08-21 | 2,237 | `anchored_b` | 0.5687 | +0.0020 | 0.1941 | +0.0002 |
| 2026-08-21 | 2,237 | `affine_ab` | 0.5687 | +0.0021 | 0.1941 | +0.0003 |
| 2026-08-21 | 2,237 | `two_slope` | 0.5693 | +0.0026 | 0.1944 | +0.0006 |
| 2026-08-21 | 2,237 | `isotonic10` | 0.5799 | +0.0132 | 0.1984 | +0.0045 |
| 2026-08-22 | 2,337 | `raw_book` | 0.5484 | +0.0000 | 0.1851 | +0.0000 |
| 2026-08-22 | 2,337 | `anchored_b` | 0.5500 | +0.0016 | 0.1856 | +0.0005 |
| 2026-08-22 | 2,337 | `affine_ab` | 0.5510 | +0.0026 | 0.1861 | +0.0010 |
| 2026-08-22 | 2,337 | `two_slope` | 0.5503 | +0.0019 | 0.1858 | +0.0007 |
| 2026-08-22 | 2,337 | `isotonic10` | 0.5544 | +0.0060 | 0.1876 | +0.0025 |
| 2026-08-23 | 1,942 | `raw_book` | 0.5166 | +0.0000 | 0.1712 | +0.0000 |
| 2026-08-23 | 1,942 | `anchored_b` | 0.5158 | -0.0008 | 0.1709 | -0.0002 |
| 2026-08-23 | 1,942 | `affine_ab` | 0.5151 | -0.0014 | 0.1706 | -0.0006 |
| 2026-08-23 | 1,942 | `two_slope` | 0.5148 | -0.0018 | 0.1705 | -0.0006 |
| 2026-08-23 | 1,942 | `isotonic10` | 0.5149 | -0.0017 | 0.1709 | -0.0002 |

### Pooled deltas vs the raw book, with intervals

| model | Δ log-loss | window-clustered 95% | day-clustered 95% |
|---|---:|---|---|
| `anchored_b` | +0.0011 | [-0.0006, +0.0026] (1634 win) | [-0.0008, +0.0020] (3 days) |
| `affine_ab` | +0.0012 | [-0.0005, +0.0028] (1634 win) | [-0.0014, +0.0026] (3 days) |
| `two_slope` | +0.0010 | [-0.0010, +0.0031] (1634 win) | [-0.0018, +0.0026] (3 days) |
| `isotonic10` | +0.0062 | [+0.0032, +0.0090] (1634 win) | [-0.0017, +0.0132] (3 days) |

## Monitor — reports, never promotes

**Role:** MONITOR — reports, never promotes. **Promotion:** a NEW FROZEN PROTOCOL with a calendar trigger; no bar lives here.

Population **VERDICT_COINS_ONLY [btc, eth] (FLOW_MODEL_PROTOCOL_V5:333)** · vintage **windows_total=1784 [btc:892,eth:892]** · 4 day(s) present, 3 scored · status **REPORTING**.

> The §12 step-5 gate that used to be rendered here is **DELETED**. It read *day-clustered CI excludes 0, else Identity* at 7 days and printed `would_ship_today`. The plan deleted it and this file kept enforcing it — so at 7 days a machine-generated receipt would have announced an automatic promotion the plan says cannot exist. **Deleting prose does not delete a rule that is implemented.**

Sign convention: Δ is challenger **minus** baseline on log-loss, so **negative beats the raw book**.

| model | Δ row-wtd | Δ **day-wtd** | day-clustered 95% | k | reading |
|---|---:|---:|---|---:|---|
| `anchored_b` | +0.00105 | +0.00097 | [-0.00076, +0.00203] | 3 | **INDISTINGUISHABLE_FROM_THE_BOOK** |
| `affine_ab` | +0.00121 | +0.00107 | [-0.00144, +0.00260] | 3 | **INDISTINGUISHABLE_FROM_THE_BOOK** |
| `two_slope` | +0.00104 | +0.00090 | [-0.00180, +0.00259] | 3 | **INDISTINGUISHABLE_FROM_THE_BOOK** |
| `isotonic10` | +0.00619 | +0.00585 | [-0.00171, +0.01321] | 3 | **INDISTINGUISHABLE_FROM_THE_BOOK** |

> **⚠ DETECTED, not assumed — the intervals above are NOT 95% intervals.** For `anchored_b`, `affine_ab`, `two_slope`, `isotonic10`: at k<=3 this interval IS [min,max] of the per-day deltas; 'excludes 0' means 'all days share a sign', a 25% event under a symmetric null Both columns are shown because the point estimate is a ROW average while the interval under it resamples DAYS, and the plan's primary unit is days — the two readings differed by 8.3× the last time only one was reported.

## What this does and does not license

- **Nothing here promotes anything.** Promotion requires a new frozen protocol with a calendar trigger. Where the table says `DAY_BLOCK_UNAVAILABLE`, the sample has one day block and no day-clustered interval exists — that is a refusal, not a small number.
- The deployed map pins `a = 0`. Any gain attributable to `a` is a bet that the observed drift continues, which is the directional claim this programme does not make.
- Nothing here is P&L. The unconditional gap is not harvestable: the measured selection haircut is 60–97 % and it is BE-FlowAndFills' term, not this module's.
