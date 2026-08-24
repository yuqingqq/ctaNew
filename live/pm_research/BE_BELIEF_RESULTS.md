# BE-Belief — walk-forward recalibration of the venue book

**Status: DEVELOPMENT, not decision eligible.** Produced by
`be_belief.py`, implementing `plans/BE_BELIEF_PLAN.md` §12 steps 1–3.

## Receipt

- rows 23,801 over 4,762 windows, 4 UTC day(s): 2026-08-20, 2026-08-21, 2026-08-22, 2026-08-23
- sampling rule `WHOLE_COVERED_POPULATION`
- `days_sampled` ['20260820', '20260821', '20260822', '20260823'] (n=4); `days_declared` ['20260819', '20260820', '20260821', '20260822', '20260823']
- decision times, elapsed s: [30.0, 60.0, 120.0, 180.0, 240.0] (r = 270, 240, 180, 120, 60)
- up-rate 0.5203 per row, 0.5204 per window
- refused: {'no_state_at_decision_time': 9}

One outcome per window is shared by every decision time in it, so every n here is inflated ~5× and intervals cluster on window, then day.

## Calibration — at the mid AND at the executable prices

`spread` here is POOLED ACROSS COINS and is therefore a pooling artefact in exactly the way U8 named: ATM spread is **1 tick** on btc/eth and 3–7 ticks on the thin coins, so a pooled figure reports neither. Read it as a mixture, never as a venue spread.

| bucket | rows | windows | mid | bid | ask | realised | gap vs mid | buy Up at ask | sell Up at bid | spread | age s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0-0.1 | 3128 | 1770 | 0.030 | 0.021 | 0.039 | 0.033 | +0.003 | -0.006 | -0.012 | 0.018 | 0.7 |
| 0.1-0.2 | 1550 | 1177 | 0.149 | 0.130 | 0.167 | 0.122 | -0.027 | -0.045 | +0.008 | 0.037 | 0.1 |
| 0.2-0.3 | 1803 | 1402 | 0.249 | 0.230 | 0.268 | 0.244 | -0.005 | -0.024 | -0.014 | 0.038 | 0.3 |
| 0.3-0.4 | 2129 | 1628 | 0.349 | 0.331 | 0.367 | 0.358 | +0.009 | -0.009 | -0.027 | 0.035 | 0.2 |
| 0.4-0.5 | 2524 | 1868 | 0.449 | 0.432 | 0.466 | 0.447 | -0.002 | -0.018 | -0.015 | 0.034 | 0.4 |
| 0.5-0.6 | 2648 | 1908 | 0.545 | 0.528 | 0.562 | 0.545 | -0.000 | -0.017 | -0.017 | 0.034 | 3.9 |
| 0.6-0.7 | 2356 | 1767 | 0.649 | 0.632 | 0.667 | 0.635 | -0.015 | -0.032 | -0.003 | 0.035 | 0.1 |
| 0.7-0.8 | 2064 | 1572 | 0.748 | 0.729 | 0.767 | 0.753 | +0.005 | -0.013 | -0.024 | 0.037 | 0.4 |
| 0.8-0.9 | 1712 | 1373 | 0.848 | 0.831 | 0.865 | 0.864 | +0.016 | -0.001 | -0.033 | 0.034 | 0.3 |
| 0.9-1.0 | 3887 | 2038 | 0.968 | 0.959 | 0.977 | 0.975 | +0.007 | -0.002 | -0.016 | 0.018 | 0.6 |

## In-sample `a` / `b` — DIAGNOSTIC, never deployed

`a` is the drift channel: estimated always, deployed never (§4.3).

| sample | n | a | b |
|---|---:|---:|---:|
| core |logit m| ≤ 3 | 18,755 | -0.006 | 1.037 |
| extreme | 5,046 | +0.271 | 1.055 |
| all rows | 23,801 | -0.001 | 1.040 |

| day | n | a | b |
|---|---:|---:|---:|
| 2026-08-20 | 2,082 | -0.069 | 0.989 |
| 2026-08-21 | 7,798 | +0.088 | 0.992 |
| 2026-08-22 | 8,004 | -0.093 | 1.120 |
| 2026-08-23 | 871 | +0.026 | 0.953 |

| r (s remaining) | n | a | b |
|---|---:|---:|---:|
| 270 | 4,759 | +0.046 | 1.035 |
| 240 | 4,750 | +0.020 | 1.133 |
| 180 | 4,335 | -0.035 | 1.073 |
| 120 | 3,283 | -0.061 | 0.956 |
| 60 | 1,628 | -0.120 | 0.975 |

## Walk-forward — fit on days < d, score day d

3 scored day(s) of 4 present.

| day | n test | model | log-loss | Δ | Brier | Δ |
|---|---:|---|---:|---:|---:|---:|
| 2026-08-21 | 7,798 | `raw_book` | 0.5611 | +0.0000 | 0.1913 | +0.0000 |
| 2026-08-21 | 7,798 | `anchored_b` | 0.5611 | -0.0000 | 0.1912 | -0.0000 |
| 2026-08-21 | 7,798 | `affine_ab` | 0.5627 | +0.0016 | 0.1919 | +0.0006 |
| 2026-08-21 | 7,798 | `two_slope` | 0.5627 | +0.0016 | 0.1920 | +0.0007 |
| 2026-08-21 | 7,798 | `isotonic10` | 0.5666 | +0.0055 | 0.1934 | +0.0022 |
| 2026-08-22 | 8,004 | `raw_book` | 0.5429 | +0.0000 | 0.1832 | +0.0000 |
| 2026-08-22 | 8,004 | `anchored_b` | 0.5431 | +0.0002 | 0.1832 | +0.0001 |
| 2026-08-22 | 8,004 | `affine_ab` | 0.5442 | +0.0013 | 0.1838 | +0.0006 |
| 2026-08-22 | 8,004 | `two_slope` | 0.5442 | +0.0013 | 0.1837 | +0.0005 |
| 2026-08-22 | 8,004 | `isotonic10` | 0.5471 | +0.0042 | 0.1848 | +0.0016 |
| 2026-08-23 | 871 | `raw_book` | 0.5503 | +0.0000 | 0.1844 | +0.0000 |
| 2026-08-23 | 871 | `anchored_b` | 0.5509 | +0.0006 | 0.1845 | +0.0001 |
| 2026-08-23 | 871 | `affine_ab` | 0.5510 | +0.0007 | 0.1845 | +0.0001 |
| 2026-08-23 | 871 | `two_slope` | 0.5513 | +0.0009 | 0.1847 | +0.0003 |
| 2026-08-23 | 871 | `isotonic10` | 0.5544 | +0.0040 | 0.1864 | +0.0020 |

### Pooled deltas vs the raw book, with intervals

| model | Δ log-loss | window-clustered 95% | day-clustered 95% |
|---|---:|---|---|
| `anchored_b` | +0.0001 | [-0.0000, +0.0003] (4237 win) | [-0.0000, +0.0006] (3 days) |
| `affine_ab` | +0.0014 | [+0.0006, +0.0022] (4237 win) | [+0.0007, +0.0016] (3 days) |
| `two_slope` | +0.0014 | [+0.0006, +0.0022] (4237 win) | [+0.0009, +0.0016] (3 days) |
| `isotonic10` | +0.0048 | [+0.0034, +0.0062] (4237 win) | [+0.0040, +0.0055] (3 days) |

## Monitor — reports, never promotes

**Role:** MONITOR — reports, never promotes. **Promotion:** a NEW FROZEN PROTOCOL with a calendar trigger; no bar lives here.

Population **7 COINS POOLED ['bnb', 'btc', 'doge', 'eth', 'hype', 'sol', 'xrp']** · vintage **windows_total=4762 [bnb:680,btc:680,doge:681,eth:681,hype:680,sol:680,xrp:680]** · 4 day(s) present, 3 scored · status **REPORTING**.

> The §12 step-5 gate that used to be rendered here is **DELETED**. It read *day-clustered CI excludes 0, else Identity* at 7 days and printed `would_ship_today`. The plan deleted it and this file kept enforcing it — so at 7 days a machine-generated receipt would have announced an automatic promotion the plan says cannot exist. **Deleting prose does not delete a rule that is implemented.**

Sign convention: Δ is challenger **minus** baseline on log-loss, so **negative beats the raw book**.

| model | Δ row-wtd | Δ **day-wtd** | day-clustered 95% | k | reading |
|---|---:|---:|---|---:|---|
| `anchored_b` | +0.00013 | +0.00027 | [-0.00000, +0.00060] | 3 | **INDISTINGUISHABLE_FROM_THE_BOOK** |
| `affine_ab` | +0.00142 | +0.00119 | [+0.00065, +0.00160] | 3 | **WORSE_THAN_THE_BOOK** |
| `two_slope` | +0.00141 | +0.00126 | [+0.00091, +0.00160] | 3 | **WORSE_THAN_THE_BOOK** |
| `isotonic10` | +0.00481 | +0.00458 | [+0.00400, +0.00554] | 3 | **WORSE_THAN_THE_BOOK** |

> **⚠ DETECTED, not assumed — the intervals above are NOT 95% intervals.** For `anchored_b`, `affine_ab`, `two_slope`, `isotonic10`: at k<=3 this interval IS [min,max] of the per-day deltas; 'excludes 0' means 'all days share a sign', a 25% event under a symmetric null Both columns are shown because the point estimate is a ROW average while the interval under it resamples DAYS, and the plan's primary unit is days — the two readings differed by 8.3× the last time only one was reported.

## What this does and does not license

- **Nothing here promotes anything.** Promotion requires a new frozen protocol with a calendar trigger. Where the table says `DAY_BLOCK_UNAVAILABLE`, the sample has one day block and no day-clustered interval exists — that is a refusal, not a small number.
- The deployed map pins `a = 0`. Any gain attributable to `a` is a bet that the observed drift continues, which is the directional claim this programme does not make.
- Nothing here is P&L. The unconditional gap is not harvestable: the measured selection haircut is 60–97 % and it is BE-FlowAndFills' term, not this module's.
