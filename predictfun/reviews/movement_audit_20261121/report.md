# PI0-MOVEMENT-CALIBRATION-AUDIT-v1 — report

2026-11-21 12:01Z, session aea5bfb9. Review 5a8ee6f5's one next priority.
- Freeze `03243e5b3ddb304cf795c57010c1338181e4c478`: module, selftest, harness, checker, declaration and launcher, committed before any table existed.
- Results: the commit carrying this report.
- Scope: consumed Oct 24–27 only. No fit, no new branch, no threshold.

## The declared evidence (computed by `predicates()`; reading = the admitted entry, weighted)

| Question | Entry (declared) | First review | First pull |
|---|---|---|---|
| (i) broad or tail-concentrated | **concentrated** | concentrated | concentrated |
| (ii) decision state or fill timing | **both** | decision_state | decision_state |
| (iii) neutral vs admitted residual | **material** | material | material |
| (iv) cells populated | **populated** | populated | populated |
| recommendation | **representation_or_data_needed** | representation_or_data_needed | representation_or_data_needed |

The numbers behind each entry answer:
- **(i)** u = E[qm] − M is positive in 4 of the 4 day × side cells:
  - Oct 26 bid +6.14 c;
  - Oct 26 ask +34.23 c;
  - Oct 27 bid +0.46 c;
  - Oct 27 ask +16.39 c.

  The weighted top-5 world share of |Σ w u| is 0.591, at least 0.5, so the rule reads concentrated even though every cell is positive. Both facts stand. The unweighted share is 0.475.
- **(ii)** The decision-state ranges of c are:
  - horizon 33.6;
  - queue ratio 65.4;
  - ordinal 34.2.

  The delay-class range is 61.1 c. Neither is 1.5 times the other.
- **(iii)** c = mhat − M on fills is +38.57 c per filled opportunity on the admitted cohort (88 fills).
  - The forced Oct 26–27 reference has +3.11 c (153 fills).
  - The forced Oct 24–27 reference has -0.71 c.
- **(iv)** Cells with ≥ 30 opportunities and ≥ 10 fills: horizon 2, queue ratio
  3, ordinal 3.

## Validity

Checks and controls:
- Checks A–I: 9 / 9 true (`checks.json`):
  - A: identities and freeze;
  - B: membership and the exact-key weight / cell / part binding;
  - C: every decomposition recomputed;
  - D: strata from the client inputs;
  - E: every fill and reference recomputed;
  - F: every denominator;
  - G: predicates recomputed;
  - H: the review's independent numbers reproduced to their printed precision;
  - I: resources and the mutation suite.
- Falsifier: 13 / 13 corruptions of real copies are flagged by
  their check, with clean copies true (`checks_falsifier.json`).
- Mutation suite: 64 / 64 planted errors are caught by a selftest check.

Population:
- Membership: 329 admitted rows and 768 forced forks. Weights sum to 689.
  There are 0 weight mismatches.
- Fill-time mids exist for every admitted counted fill: 202 / 202 at entry. The median age is 24 ms
  and the maximum is 255 ms (venue time).
- Forced mids come from the retained edges: 379 / 379 for Oct 26–27. Their age is unavailable.
- There are no missing or out-of-time references.

Cost and the control:
- 148 run_light jobs, 0.275 core-hours, peak 1140 MiB. The mids high-water was 1,141 MiB.
  The write-once pilot projected 0.406 core-hours (run_rest true).
- Retained first-review control: PULL 233 / 299, percentile 0.806, 45 of
  91 strata free with 166 rows. All 299 reviews' own-time cells equal the admission cell.

## The decomposition (admitted entry, weighted; c per opportunity unless stated)

| Quantity | Value |
|---|---:|
| V − value (bias) | +11.099 |
| edge × (E[q] − q) | -0.882 |
| E[qm] − M | +11.981 |
| (P − z) mhat (probability factor) | +0.310 |
| z (mhat − M) (conditional head) | +11.671 |
| MSE(V) / zero / mechanical (c²) | 4,043.27 / 3,982.00 / 4,098.46 |
| movement MSE / zero (c²) | 4,336.46 / 4,395.20 |
| mean P / fill rate; Brier / pooled | 0.230 / 0.303; 0.152 / 0.211 |

P has useful probability discrimination with calibration error: its Brier is below the pooled rate, and it underpredicts the fill rate.
The movement head improves MSE over zero, and its mean error is +11.981 c.

## The movement head on fills: admitted versus the neutral reference

| Cohort (entry) | fills | mean mhat on fills (c / fill) | mean M on fills (c / fill) | c = mhat − M | mean E[qm] / mean M (c / opp.) |
|---|---:|---:|---:|---:|---:|
| admitted (weighted) | 88 | -6.90 | -45.47 | +38.57 | -1.78 / -13.76 |
| forced Oct 26–27 | 153 | -48.77 | -51.88 | +3.11 | -19.40 / -21.17 |
| forced Oct 24–27 | 304 | -46.22 | -45.51 | -0.71 | -19.14 / -18.32 |

Descriptive differences, admitted minus forced Oct 26–27 (no declared reading):
- The forecast differs by +41.87 c per fill.
- The realized movement differs by +6.41 c per fill.

Per filled share, the movement splits as follows (`prepost.csv`):

| Cohort | shares | total (c/share) | pre-fill (c/share) | post-fill (c/share) |
|---|---:|---:|---:|---:|
| admitted entry | 2,014.7 | -2.140 | -0.291 | -1.849 |
| forced entry, Oct 26–27 | 3,421.9 | -2.320 | -0.189 | -2.131 |

The post-fill share of the per-share movement is 0.864 for the admitted cohort and
0.919 for the forced cohort.

All 755 evaluable forced entries are decided at the episode start (t = 0 ms, 5,000 ms remaining, ordinal 1), and the forced decision-state strata have these cell counts: horizon 1,
queue ratio 3, ordinal 1. It cannot separate horizon or ordinal effects.

## Strata (admitted entry)

| stratum | cell | n | fills | mean E[qm] (c) | mean M (c) | u (c / opp.) | c = mhat − M (c / fill) | thin |
|---|---|---:|---:|---:|---:|---:|---:|---|
| hbin | (0,100] | 10 | 0 | -0.31 | +0.00 | -0.31 | — | thin |
| hbin | (100,500] | 90 | 9 | -0.62 | -0.22 | -0.40 | -7.87 | thin (cond.) |
| hbin | (2000,5000] | 131 | 56 | -2.61 | -26.60 | +23.99 | +51.53 |  |
| hbin | (500,2000] | 98 | 23 | -1.47 | -5.25 | +3.78 | +17.89 |  |
| qbin | (0,1] | 70 | 24 | -1.95 | -37.72 | +35.77 | +82.47 |  |
| qbin | (1,3] | 49 | 17 | -2.28 | -12.97 | +10.69 | +27.99 |  |
| qbin | >3 | 210 | 47 | -1.63 | -6.24 | +4.61 | +17.09 |  |
| obin | 1 | 161 | 48 | -2.53 | -16.84 | +14.31 | +39.10 |  |
| obin | 2 | 65 | 16 | -2.42 | -9.15 | +6.74 | +16.84 |  |
| obin | >=3 | 103 | 24 | -0.81 | -13.08 | +12.28 | +51.04 |  |
| delay_class | (100,500] | 31 | 31 | -2.30 | -66.03 | +63.73 | +59.43 |  |
| delay_class | (500,2000] | 28 | 28 | -3.30 | -41.93 | +38.64 | +33.76 | thin |
| delay_class | >2000 | 11 | 11 | -0.40 | -0.21 | -0.19 | -1.72 | thin |
| delay_class | [0,100] | 18 | 18 | -3.09 | -51.95 | +48.86 | +42.89 | thin |
| delay_class | no_fill | 241 | 0 | -1.45 | +0.00 | -1.45 | — | thin (cond.) |

Forecast bins and the first review / first pull are in `tables.md`.

## Concentration and contributions

| Quantity | weighting | top-5 generation share | top-5 world share | Σ |
|---|---|---:|---:|---:|
| D_EXPOSURE | weighted | 0.520 | 0.669 | -6,453.9 |
| D_EXPOSURE | unweighted | 0.365 | 0.563 | -2,499.9 |
| entry movement error | weighted | 0.461 | 0.591 | 8,255.1 |
| entry movement error | unweighted | 0.303 | 0.475 | 3,400.3 |

Signed day × side shares of the weighted sums:
- 20261026 side 0: D 0.152, entry u 0.135
- 20261026 side 1: D 0.684, entry u 0.568
- 20261027 side 0: D -0.034, entry u 0.013
- 20261027 side 1: D 0.199, entry u 0.284

## Observation sources, entry → first review (admitted, weighted; ages in ms)

These are counted with the bookTicker label (review 5a8ee6f5 finding 1):
- new PF frame: 0.969;
- new Binance bookTicker update: 0.928;
- new Binance depth snapshot: 0.917;
- new PF receipt: 0.969;
- any new observation: 1.000.

Transitions with no new observation: 0. So there is still no no-news comparison group. The direction of the decision-mid change: against 36,
unchanged 228, toward the order's value
35.

New observations alone do not attribute the POST-to-PULL reversal to changed danger.

## Review 5a8ee6f5 findings 1–4

1. **Finding 1.** The transitions table carries `new_bookticker` and no `new_bn_age` field. The three clocks and their units are documented in the module header.
2. **Finding 2.** The wording above follows the review. The units are explicit: c per opportunity, c per filled opportunity, c per share.
3. **Finding 3.** The weighted concentration reproduces the review's 0.520 (0.520, generation share of |w D|). It is shown beside the unweighted 0.365 and the
   world shares, and the signed day × side contributions are listed above.
4. **Finding 4.** Rows are bound to the manifest by exact key in three places: `adm_rows`, check B and falsifier case B_weight. The
   control derives each review's hour from its own decision time, and 299 / 299 reviews
   equal the admission cell.

## Limits

- These are related generations on two (admitted) and four (forced) consumed days. There is no interval, no validation and no
  profitability claim.
- The forced reference is differently selected. All 755 evaluable forced entries are decided at the episode start (t = 0 ms, 5,000 ms remaining, ordinal 1).
- The pre / post split is post-hoc attribution at the venue mid at the fill.
- Delay classes are realized outcomes, never features.
- Nothing is fitted. An oracle corrected with these outcomes is not a model score.
