# FEASIBLE-CANCEL-HEADROOM-v1 — superseding interpretation (v2)

2026-11-22, session aea5bfb9.
- **Supersedes:** the wording at `report.md:83, :100, :133, :141` and the results-commit headline, per review `32d102bd` finding 3.
- **Unchanged:** the frozen receipt (`2a2f78da` / `438ce3a8`), its numbers and its computed readings. `report.md` stays as provenance.
- **Check run:** none beyond `tools/review_answer_verify.py`, a descriptive cross-tab of the frozen outputs (`logs/ch_review_answer_verify.log`, 0.31 CPU s). It reproduces every count below.

## 1. Pure after the chosen cut versus pure at the first review (replaces `report.md:133`)

| Arm | Pure after the oracle's chosen cut | Of those, pure already at the first callback | Initially mixed at the first callback |
|---|---:|---:|---|
| CURRENT | 1,238 | 1,114 (cut at the first callback) | 124: 71 uncompensated, 53 compensated |
| FRONT | 1,560 | 1,241 | 319: 225 uncompensated, 93 compensated, 1 tie |

For the 124 CURRENT generations that start mixed, the oracle cuts **later**. It keeps their earlier good fills and removes the bad suffix.

Correct wording: "1,238 have only bad fills **after the chosen cut**; 1,114 of them are pure from the first review, 40 ms after activation." The 90 % first-cut share (1,129 of 1,254) stands. It supports examining admission and the first review; it does not make every economic decision an admission decision.

## 2. END wording (replaces `report.md:83, :141` and the commit headline's "bound")

**+13,067 c is the END gain of the +1 s-selected cuts**, not an END-optimized bound. The resulting CURRENT END is **−55,252 c** (recorded −68,320 c), and on Oct 25 the same cuts worsen END by 37,263 c. The cut is fixed across horizons. No END-optimized oracle is computed, and none is needed to state this correctly.

## 3. The bad-then-good denominator (replaces `report.md:100`)

The **629.11 c** of bad value and 891.57 c of good value are totals over the **27** bad-then-good generations that have a feasible cut before their first bad fill. Of the 29 flagged generations, 2 have no such cut. FRONT: 683.35 c over 44 of 47. These remain recorded-path, truncated-continuation counts. They do not test the whole-life queue hypothesis (`90e82f8e`).

## 4. Unchanged

- The readings, every number in `report.md` and the "does not show" list stand.
- Finding 4 (checker coverage narrower than the prose) stays deferred: the review independently reproduced every column and table, and the named comparisons will be extended when those fields next drive a decision.
