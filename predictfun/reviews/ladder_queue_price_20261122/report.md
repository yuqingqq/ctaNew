# QUEUE-PRICE-STATE-AUDIT-v1 — receipt

2026-11-22, session aea5bfb9.
- **Frozen:** `0e20331f`, before the run.
- **Specified by:** review `72a36de9`'s ONE next priority, with its four corrections bundled.
- **Run:** selftest → projection (fits) → run → readings → checks, through `launch_queue_price.sh`.
- **Coverage:** the same 4,840 neutral BASELINE live states, consumed Oct 24–27, 19 complete markets, saved data only.
- **Scope:** a descriptive diagnostic. It is not predictive skill, a guard or a validation, and it measures no model skill.
- **Statistics:** 4 UTC days < 5, so point estimates only.
- **Support rule:** a value counts only with ≥ 20 filled states valid for it in its own cohort.

## Integrity

| Item | Result |
|---|---|
| Join | 4,840 / 4,840 states joined at the exact key (day, m, gid, t, side) to one original live feature row each. fp_queue / fp_remaining / o_depth equal Q / R / o_depth on every state |
| Independent checks | `qp_checks.py` A / B / U / G / C / P / I all pass. U re-derives every reused field, including client_minus_venue from the audit first rows and the ledgers' venue records, with the ledger gzip bytes = their day records. C covers all 4,920 cell rows and every column. P recomputes every reading |
| Falsifier | 10 / 10 corruptions refused, including the review's two demonstrated escapes: a zeroed client_minus_venue and a negated h1000_per_share |
| Mutation and selftest | 23 / 23 mutants caught; selftest groups 1–5 pass |
| Resources | Round **0.012 core-hours** (cap 0.020, the write-once projection allowed it). Ladder line **1.974 of 2.0**. Peak 248 MiB |

## Occupancy census (supersedes the prior receipt's count; review finding 1)

| Price state | Display | States | front | within_R | beyond_R |
|---|---|---:|---:|---:|---:|
| at_touch | shown_positive | 1,633 | 133 | 841 | 659 |
| behind_best | shown_positive | 2,749 | 198 | 1,088 | 1,463 |
| behind_best | shown_zero | 145 | 145 | 0 | 0 |
| inside_best | shown_zero | 313 | 313 | 0 | 0 |

Of the 789 front states, **458 have zero displayed size (313 inside the best, 145 behind it) and 331 are occupied** (133 at the touch, 198 behind). Q > shown occurs in 0 states, as expected: the running minimum includes the current frame. No display is unknown or invalid.

## Principal comparison: at the touch, occupied (union, common cohort)

Per-share values are in c/share. The fill parts are the fill-level positive / negative per share at 1 s.

| Queue band | States | Filled (all) | Valued filled (1 s) | Valued shares | Censored from common | Median time to END | < 30 s to END | Fill-time | 1 s | 5 s | 30 s | Fill parts at 1 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| front | 133 | 64 | 39 | 1,577.7 | 25 | 47.6 s | 48 | +0.264 | **−0.951** | +1.007 | **+4.592** | +0.497 / −1.448 |
| within_R | 841 | 407 | 357 | 15,994.7 | 50 | 130.5 s | 113 | +0.338 | **−1.505** | −1.809 | **−1.267** | +0.468 / −1.973 |
| beyond_R | 659 | 349 | 322 | 13,873.3 | 27 | 152.1 s | 49 | +0.242 | **−2.274** | −2.856 | **−2.920** | +0.384 / −2.658 |

Computed readings:
- **Principal union order:** beyond_R < within_R < front, at 1 s and at 30 s.
- **The same order pooled over all states**, at the same support rule: front 129, within 609, beyond 715 valued filled. So **`q3_persists_at_touch_occupied` = true at 1 s and at 30 s.**
- **Days:** `days_evaluable` = 1 of 4, because front has 20 / 12 / 6 / 1 valued filled states on Oct 24 / 25 / 26 / 27.
  - At 1 s, Oct 24's order is within_R < beyond_R < front, so `days_same_order` = 0.
  - At 30 s it matches (1).
  - `same_order_all_4_days` = false at both horizons.
- **Sides:** side 0's front has 12 valued filled states, so it is not evaluable. Side 1's order at 1 s is beyond_R < front < within_R, so `sides_same_order` = false. At 30 s side 1 matches.
- **By interval:** only I0 is evaluable (front 21 valued filled).
  - 1 s: beyond_R −2.485 < within_R −1.658 < front −1.296.
  - 30 s: within_R −1.271 < beyond_R −0.332 < front +1.978.
  - I1–I3 are insufficient_support.

Read from the frozen per-day cells after the run, below the declared support on Oct 25–27 (so not a declared reading), front is the least negative at 1 s on every day:

| Day | front | within_R | beyond_R |
|---|---:|---:|---:|
| Oct 24 | −0.86 | −1.26 | −1.17 |
| Oct 25 | −0.98 | −2.00 | −2.31 |
| Oct 26 | −1.41 | −1.50 | −2.57 |
| Oct 27 | −0.50 | −0.93 | −3.33 |

**Composition the comparison does not control.** Front-at-touch states sit near the market's end:
- median 47.6 s to END, against 130.5 s and 152.1 s for within_R and beyond_R;
- 36 % of them (48 of 133) have < 30 s left;
- 19 % (25 of 133) drop out of the common cohort, against 6 % and 4 %.

## The pooled front by price state, and the exact-depth diagnostic

The pooled front's union-common value changes sign across its price states (1 s / 30 s, with valued filled states):

| Front price state | 1 s | 30 s | Valued filled |
|---|---:|---:|---:|
| at_touch, occupied | −0.951 | +4.592 | 39 |
| behind_best, occupied | +0.897 | −2.597 | 31 |
| behind_best, zero display (sparse) | −0.423 | −2.871 | 18 |
| inside_best, zero display | −3.998 | +0.005 | 41 |

The pooled front therefore averages states with different values. The inside-best front orders, priced inside the received spread, are the most adverse at 1 s.

Price-state totals (union common, 1 s / 30 s):

| Price state | States | Valued filled | 1 s | 30 s |
|---|---:|---:|---:|---:|
| at_touch, occupied | 1,633 | 718 | −1.816 | −1.702 |
| behind_best, occupied | 2,749 | 676 | −2.606 | −4.116 |

Behind-best occupied, exact depth 1 tick (1 s / 30 s, valued filled):

| Queue band | 1 s | 30 s | Valued filled |
|---|---:|---:|---:|
| front | −0.295 | −8.613 | 20 |
| within_R | −2.198 | −2.812 | 183 |
| beyond_R | −3.033 | −4.828 | 259 |

- Order: beyond_R < within_R < front at 1 s, and front < beyond_R < within_R at 30 s.
- Depths 2–4 are insufficient_support.

## Pressure census and queue quality

- **Pressure** (support census only, no contrast is evaluable: 0 of 6). Within the principal states, the positive-pressure cells have at most 13 valued filled states:

  | Queue band | positive_below_q5 | q5 |
  |---|---:|---:|
  | within_R | 13 | 11 |
  | beyond_R | 7 | 8 |
  | front | 0 | 1 |

- **Client queue estimate at the first review vs the simulator's venue queue.** This is explanatory, simulator-relative, and not exchange truth.

  | Group | Within 1 share | Client below by ≥ 1 | Client above by ≥ 1 |
  |---|---:|---:|---:|
  | at_touch, occupied | 900 / 1,633 | 532 | 201 |
  | behind_best, occupied | 1,654 / 2,749 | 939 | 156 |
  | Zero-display groups | 349 / 458 | — | — |

## Corrections to ARRIVAL-QUEUE-DECOMPOSITION-v1's receipt (superseding in band; its frozen files are not edited)

1. **Occupancy** (its report §R2). The count "656 of the 789 front states are alone at their price" is replaced by the census above: 458 zero-display, 331 occupied.
2. **Denominators** (its R2 / R4 tables). The "Filled states" and "Shares" columns beside the common-cohort values were all-state counts. For the union queue bands the values rest on:

   | Band | Common filled states | Common shares |
   |---|---:|---:|
   | front | 129 | 4,682.60 |
   | within_R | 609 | 29,435.52 |
   | beyond_R | 715 | 31,892.16 |

   Its sparse flags screened all-state fills. On the common cohort no pooled classification flips (review `support.csv`). Its reader's `r3_evaluable_negative_all_days` was misnamed (it required only one evaluable day); its value of 0 is unaffected. This receipt names the day counts and requires all four days for `same_order_all_4_days`.
3. **Checker coverage.** Its `aq_checks.py` let two corruptions through. The review verified the current values independently, and this receipt's checker re-derives every reused field.
4. **Timing** (its `report.md:68`). An arrival interval is not the latest actionable signal time.
   - A fill in I0 = [d + 37, d + 287) can still be excluded by a later 50 ms review: a request at d + 100 takes effect at d + 137.
   - "22.5 %" is the share of the union-common cohort's net 1 s loss arriving in I0. It is not a timing-unavoidable fraction.

## What this shows and does not show

On the pooled union, the queue ordering does point the same way at an occupied price level at the touch:
- front least negative, beyond_R most negative;
- the same order as pooled, at 1 s and at 30 s.

That is a descriptive point estimate, not an established ordering:
- front at the touch has 39 valued filled states;
- only 1 of 4 days and 1 of 2 sides meet the support minimum, and side 1's 1 s order differs;
- only I0 is evaluable among the intervals;
- front-at-touch states are concentrated near END (median 47.6 s), a difference the comparison does not control.

It is not a queue rule, a priority value or a keep/cancel valuation. The pressure-within-queue question remains unsupported.

**Largest evidenced limitation:** the comparable occupied cells are thin and time-to-END-skewed for the front. This saved population supports the direction on the pooled union but cannot establish it across days or sides at the declared minimum. By the review's rule, no tuned variant follows automatically. More data, or a counterfactual queue-value experiment, is a USER decision.

Files:
- `outputs/` holds the receipt copies (`states` and `cells` .csv.gz, `audit.json`), `readings.json` and the reading tables (t_principal, t_price_display, t_front_composition, t_behind_depth, t_pressure_census);
- `checks.json` and `checks_falsifier.json`;
- `logs/`, including `projection.json` and the review-verification log.

The pre-freeze rehearsal outputs (`data/predict_fun/research/ladder_queue_price_v1_20261122/rehearsal`) are superseded and unreported.
