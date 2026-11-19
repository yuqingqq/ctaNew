# The four-part review, v2: review 92d8e52f answered — 2026-11-19

**USER:** "Check my latest commit, next, can you ask codex research session for next priority. continue the implementing and next priority
loop".

**Review addressed:** [92d8e52f](../automatic/20261119_0528e1a6/report.md), on the four-part review receipt
([v1](../pipeline_paired_review_20261119/report.md), commit 0528e1a6).

**How this receipt relates to v1.** It supersedes v1 in-band on the points below. v1 stays as provenance. Every number, reading and verdict
of v1 is unchanged; the comparison below shows this on both days.

## Each finding at the code

| Finding (92d8e52f) | At the code | Handled |
|---|---|---|
| **P2.** The plan v2 screen tests zero on the raw row sum, but decides sign and amount on episode-balanced means | Holds: `pf_pipeline_paired_review.py:240` against `:244-245` | Zero, sign and amount are now decided on the episode-balanced label mean. Nonfinite inputs give NOT_EVALUABLE. Your counterexample (100 episodes with two +1 c rows and 100 with one −2 c row: raw sum 0, episode-balanced mean −0.5 c) now gives PASS, and so does the same data with one row per episode. A genuine all-zero group and a zero episode-balanced mean both stay NOT_EVALUABLE. Mutants: the v1 raw-sum zero test, and nonfinite inputs admitted |
| **P2.** The receipt extends nesting to plan v2's ENTRY-PA | Holds | Correction 1 below |
| **P2.** The receipt reads B1a as establishing A's actual loss level | Holds | Correction 2 below |
| **P3.** V3's `np.isclose` has no `rtol=0`, so END could move 0.05 ms | Holds | An explicit absolute tolerance of 1 µs (1e-3 ms) with no relative term. The timestamps are epoch ms of about 1.79e12 (float64 ulp 2.4e-4 ms). New falsifier: END shifted by 0.01 ms fails V3. Mutant: the v1 relative tolerance |
| **P3.** The receipt says K7 "lets NaN through (`> 1e-6`)" | Holds: the statement was mine and wrong | `pf_pipeline_paired.py:309` is `if not (np.abs(idn) <= 1e-6).all():`, which rejects NaN. The `> 1e-6` on line 310 is only the printed count. I read that line as the condition. K7 remains an algebraic identity check; K4 carries the independent value and quantity evidence |

## Corrections in-band

1. **Nesting.** The nesting finding holds for this run's **same-admission A-PA** only: one generation, A's pulls never land before A-PA's,
   and the computed nesting predicate is true on both days.
   - It does not transfer to plan v2's ENTRY-PA, which has its own state and re-entry. There, pulling earlier can change inventory,
     capacity, later prices and admissions, so the quote histories need not nest.
   - v1's recommendation to replace ENTRY-PA, and the suggestion of never posting as a quoting comparator, are withdrawn. Uptime against
     never posting is trivial.
   - For R1 / R4, nesting is a computed predicate on the actual policies and states, never assumed. ENTRY-PA keeps its economic role,
     alongside PI0-50 / NEU2.
2. **B1a.** B1a is one day of live orders from an older, churning engine. It is not a validated per-fill probability for these two days or
   this order population.
   - The B1a-weighted values are a **sensitivity result**: down-weighting young fills changes the reported loss level, and A still trails
     PI0, NEU and E1 under both declared weightings.
   - They do not identify A's actual live loss or give a calibrated correction. Missing or spurious fills change later queue, partial-fill
     and action states, which a reweighting of saved fill values cannot reproduce.
3. **Two matched-control estimands, kept separate.**
   - The evaluation's declared reading, MATCHED CONTROL (A above the matched null's **mean**), **stands as PASS on both days**. It is
     literally true and is not changed retroactively.
   - The review's separately declared reading, MATCHED CONTROL WITH ITS SPREAD (one-sided draws p ≤ 0.05), is PASS on Oct 26 and FAIL on
     Oct 27.
   - Neither is a day-clustered interval or a forward result.
4. **Units.**
   - The "spread component" uses the **posting mid**, and the movement runs from that mid to END. It is not the measured spread at each
     fill, nor a 100 ms / 1 s markout.
   - The time-integrated inventory is summed over independent single-order episodes, not a netted continuous bid / ask portfolio.
   - Phase and spread associations do not by themselves justify a rule that refuses most of the market.
5. **K7.** As in the table above.

## The v2 run (the fixed module on the same inputs)

**Checks (before the run):**
- the v2 selftest passes, including the episode-balanced screen cases and the 0.01 ms END falsifier;
- the mutation suite catches **41 / 41** planted errors, among them the v1 raw-sum zero test, nonfinite screen inputs and the v1 relative tolerance.

**The v2 run** on Oct 26 and Oct 27 (consumed): 35 / 37 s; peak RSS 1,796 / 1,916 MiB against the 2,048 MiB guard. The buffer is thin; chunk
the export read before any larger run.

**[compare_v1_v2.py](compare_v1_v2.py)** ([log](logs/prv2_compare.log)), declared before v2 ran:
- **every reading, value and plan v2 screen verdict is identical** on both days (0 differences outside the module's own hash and outputs);
- 0 screen verdicts changed;
- the module source changed.

So the screen fix changes no saved verdict, as review 92d8e52f expected; that is now computed rather than asserted. Validity: PASS, no open
defect, on both days.

## Files

- **Module:** [`pf_pipeline_paired_review.py`](../../research/pf_pipeline_paired_review.py) v2 and its
  [mutation harness](../../research/pf_pipeline_paired_review_mutants.py).
- **Comparison:** [compare_v1_v2.py](compare_v1_v2.py), declared before the v2 run.
- **Logs and hashes:** [logs/](logs/); [sources.sha256](sources.sha256).
- **Data:** `data/predict_fun/research/pipeline_paired_review_v2_20261119`.
