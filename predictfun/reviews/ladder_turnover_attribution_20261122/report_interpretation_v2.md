# PAIRED-TURNOVER-ATTRIBUTION-v1 — superseding interpretation (v2)

2026-11-22, session aea5bfb9.
- **Supersedes:** the wording at `report.md:41, :70, :79` and FINDINGS P114, per review `838f4be2` finding 1.
- **Unchanged:** the frozen receipt (`db044afb` / `51ed592d`) and every number. `report.md` stays as provenance.
- **Check run:** `review_answer_verify.py`, a descriptive pass on the frozen outputs (`logs/ta_review_answer_verify.log`, 0.28 CPU s). It reproduces each figure below.

## 1. Day signs and components (replaces `report.md:79` and P114's "on every sampled day")

**Total REVIEW Y is negative on all four sampled days.** Matched turnover explains its **pooled** deficit (ΔMatched −7.7k c), but not every day.

On **Oct 26** matched turnover *improves*: Y −2,562.82 c = ΔMatched +862.93 + ΔResidual −3,425.75. Residual inventory explains that day's loss.

Correct wording: "total REVIEW Y is negative on all four sampled days; matched turnover explains its pooled deficit, but improves on Oct 26, when residual inventory explains the loss."

A difference in total matched profit also changes matched quantity, lot mix and carried closures. It is **not** a matched-price causal comparison. Carried lots' closing profit is incremental since the fork, not lifetime round-trip profit.

## 2. Per matched share (qualifies `report.md`'s per-share table)

The table's figures are **means of per-arm ratios**. The ratio of sums is more negative:

| Cohort, arm | Mean of arm ratios | Ratio of sums | New lots only, ratio of sums |
|---|---:|---:|---:|
| REVIEW retain | −2.900 c | −3.401 c | −3.397 c |
| REVIEW alternative | −2.879 c | −3.181 c | −3.243 c |
| ADMISSION retain | −2.026 c | −2.828 c | −2.831 c |
| ADMISSION alternative | −2.153 c | −2.783 c | −2.792 c |

Newly opened lots alone lose about as much as all lots. The losing turnover is therefore not an artifact of carried bases or averaging.

That does not make inventory risk irrelevant: price movement while a lot is held becomes a realized loss when it closes.

## 3. "Largely independent" (replaces `report.md:70` and P114)

The own END sign and the matched round-trip profit are **not interchangeable**. No independence test was declared, so no independence claim is made.

The focal classes describe the matched portion; unmatched residual lies outside them. Neither a profitable matched portion nor a negative own-END sign shows that an early fill was necessary to obtain queue priority. The queue mechanism stays **unidentified**.

## 4. Unchanged

Every value, the corrected delays, the day-sign predicates and the "does not show" list stand. Finding 2 (ancillary checker coverage) is deferred: the review's `verify.py` reproduced every actual field.
