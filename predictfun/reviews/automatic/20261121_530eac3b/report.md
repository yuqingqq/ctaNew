# PI0-MOVEMENT-SHRINK-v1 — combined review, 2026-11-21

Reviewed target **530eac3b230659f1d90c630aac9ecd0f80f139e5**, range
`f311caaef6ea50b8a7df7ebd6c74a3cf22f39dc1..530eac3b230659f1d90c630aac9ecd0f80f139e5`.
The specification freeze is `9720813606172cde68ccdf85947c93c4ead28470`;
the fitted-parameter commit is `bd5e2f0139dad411cf187e4810e3cbd4e93d6417`.
The intervening `7ad03681` is the previous review/contract. Source references
below refer to the pinned target, inspected in `/tmp/pf-review-530eac3b`.

**Accept the implementation and saved result. No BLOCKING finding. Do not
promote this challenger or tune another coefficient on these same rows.**
All four fitted mixtures collapse to the pooled-prior control. On Oct 27 the
candidate's two selected entries have no fills; its six selected reviews
contain one loss and five zero outcomes. It demonstrates no positive local
exposure value. Entry's small pooled MSE improvement does not change that
conclusion; review MSE worsens.

The critical research gap is reliable state-dependent valuation of longer
exposure on representative decision states. The accounting and mapping used
here reconcile. This result rejects this candidate as a demonstrated
improvement; it does not establish that no profitable conditional retention
rule exists, that every side loses, or that a different loss function is the
solution.

**ONE next priority:** broaden the neutral-exposure label population under
the **same frozen PI0 and target**, on additional already-consumed dates,
before choosing another model/target variant. The concrete scope and budget
proposal is [next_priority.md](next_priority.md). It requires a USER go for
the expanded label-build scope; this review launches none of it. The proposed
dates are Oct 24–25, Oct 31 and Nov 1–3, reusing the existing Oct 26–27 cohort
as a separately identified reference. These are development data, not
untouched validation. A new target or an untouched-day test is not the next
priority.

**Ranked actionable checklist — all NONBLOCKING for these saved results.**

1. **P2 / NONBLOCKING: qualify the pooled conclusion and describe the actual
   decision changes.** `FINDINGS.md:3192` says the challenger “only reduces
   posting / keeping.” Counts decline, but the retention decisions are not
   a subset: **35 original KEEP decisions become PULL, six original PULL
   decisions become KEEP, zero KEEP decisions survive**. This exactly
   accounts for the 41 changes already reported at
   `reviews/movement_shrink_20261121/report.md:88`–90; see `data.json`.
   Record these directions when discussing a more conservative rule or
   churn. The new KEEP loss comes from a newly selected exposure, not from
   merely retaining a subset of the old rule.

   Also scope “none beats every rule” to the pooled, tested comparison.
   The receipt correctly shows Oct 27 **bid hold-all** local value of
   **+1.464 c at entry and +0.995 c at first review** (lines 77 and 92),
   while the ask values are -12.002 and -10.652 c. This is not permission to
   adopt a bid-only rule after seeing the day. It is evidence that an
   all-sides/all-states “never retain” conclusion is too broad. Keep the
   local-versus-dynamic-PI0 distinction: hold-all's entry J is not factual
   PI0 P&L; entry and review scores cannot be added.
   **Acceptance:** correct the handoff interpretation in the next combined
   receipt, retain both sides, and report added/removed decisions alongside
   counts. No experiment rerun is necessary.

2. **P3 / NONBLOCKING: the improved checker still omits some saved context
   fields.** `shrink_checks.py:281`–301 compares the manifest keys, weights
   and numerical moments/targets; lines 318–329 compare predictions and
   masks reconstructed from the trusted audit/partition context. It does
   not compare every context field in the saved `eval_rows.csv`, including
   `mid`. On a private copy, changing its first row's mid by +1 c leaves
   **all A–J true** (`instrument.json`). The aggregate scores are unaffected
   because the checker uses the correct source mid, but a later consumer
   trusting the saved context could use a different feature value.
   **Current bounded workaround:** the committed receipt hashes match all
   actual outputs, and the complete saved evaluation table, including its
   context fields, was reproduced byte for byte from bound inputs. The
   actual row is correct. For reuse, either consume these exact committed
   bytes or compare every decision-context column to its source, including
   side/day/time/hour/mid/strata, with a context corruption test. Defer a
   broader validator refactor; no rebuild or fix-only review is needed.

**Previous review items 1–4 are resolved for this experiment.** The prior
audit's recommendation remains a declared heuristic; the chronology and
adaptive selection are explicit. The new checker recomputes all score,
calibration and action aggregates, plus every null draw, closing the previous
aggregate-value gap. Weighted fill denominators are explicit; the old
per-share table is not reused. Cost has a named as-of. Every reused admission
record is bound to its committed receipt, and the reference audit table is
bound by its committed hash. The old mid-extraction resume concern is not
invoked by this round, which has no new mid lookup.

**Fit, population and accounting.** `pf_movement_shrink.py:152`–175 matches
the specified formula. Training uses all eligible Oct 26 rows, with zero
movement on no-fill outcomes; mu's fill-conditioned denominator is a moment
of that complete table, not a realized-policy-fill-only training set. The
physical bounds use each stage's decision mid and the correct side sign.
P, E[q], edge, original END, target policy and effect times are unchanged.
`V + lambda*(a-b)` preserves lambda=0 exactly and equals the specified value
mapping within the original identity tolerance.

The raw slopes 2.1034 / 1.8471 / 1.6926 / 1.7524 independently reproduce
against a weighted linear solve. All four clip to 1. The candidate and prior
have identical saved E[qm], V and action masks on every evaluation row. This
is the declared boundary solution, not a code error and not a reason to
extend lambda above 1 after seeing the result.

The complete admitted source has 329 generations and 987 stage/status rows.
TRAIN has 166 entry / 151 review rows; EVAL has 163 / 148; the 134 Oct 27
first-pull rows are descriptive only. The frozen manifest, train/evaluation
key hashes, complete statuses and inclusion weights reconcile. No forced
cohort enters this fit. Source, manifest and declaration are unchanged from
the freeze; the fit was committed before evaluation outputs. The live
training command loads the saved cohort to check membership, but only Oct 26
TRAIN rows enter parameter arithmetic. That is the relevant chronology
boundary, and it holds.

**Verified economic and forecast reading.** Units below are cents per local
opportunity for J/bias and cents squared for MSE. Data are related generations
on consumed dates, as of the target receipt; there is one evaluation day and
no confidence interval.

| Oct 27 pooled metric | Original | Candidate = prior |
|---|---:|---:|
| Entry movement MSE | 2,248.6 | 2,125.9 |
| Entry value MSE | 2,050.2 | 2,010.8 |
| Entry movement bias | +6.62 | -7.72 |
| Entry chosen / all eligible | 163 / 163 | 2 / 163 |
| Entry local J | -3.740 | 0.000 |
| Review movement MSE | 1,578.3 | 1,808.5 |
| Review value MSE | 1,495.7 | 1,749.9 |
| Review movement bias | -0.63 | -11.01 |
| Review chosen / all eligible | 35 / 148 | 6 / 148 |
| Review local J | -0.852 | -0.196 |

Choosing no exposure has J=0. The candidate selects **no individually positive
outcome** in either stage: entry has two zeros/no fills; review has one
negative filled outcome and five zeros. At review the candidate is also
below the matched-null mean (-0.061 c), percentile .244. Entry's percentile
1.0 is a tie: its 500 null values and its own value are all zero. It is not
selection skill.

The frozen INCONCLUSIVE statuses are correctly computed by
`pf_movement_shrink.py:279`–298: equality with the prior and insufficient
movable support take precedence. Entry has 2 free strata / 6 rows; review
6 / 21, against the declared minimum 5 / 30. Preserve those statuses. The
practical candidate decision is still **no promotion**: its economic PASS
inequalities are not satisfied. INCONCLUSIVE does not license selecting a
different lambda or opening untouched data to seek a positive result.

`day_comparison.csv` independently verifies that the original entry movement
bias is **+18.201 c on Oct 26 versus +6.619 c on Oct 27**; review is
**+13.668 versus -0.634 c**. These are large observed differences, but two
days with concentrated tails do not identify market-regime drift versus
sampling variation or a changed state mix. Nor does a failed scalar
correction prove that squared loss, the mean-value target or the entire
model family is wrong. The data show that this one fitted correction does
not transfer into useful action selection.

**Verification and limitations.** Four bounded review jobs used
`PF_SLOT=codex run_light.sh`, pinned imports and absolute script paths.
Total successful cost: **23.06 s wall / 21.07 s user CPU, peak 275 MiB**.
One shell redirection typo refused before either run_light or Python started;
it is disclosed in `attempts.txt`. No other failed review job.

- All six committed output artifacts match their hashes and raw-data copies.
  Seven specification/freeze files are unchanged; fit bytes match the fit
  commit; the freeze had no fit and the fit commit had no scores.
- All five output CSVs reproduce byte for byte using the **saved** parameters.
  A–J pass; 27 score rows, 114 calibration rows, 30 action rows and all 1,000
  null draws independently reconcile. Every original/prior/candidate mask
  and all local action accounting agree.
- Both synthetic suites pass. All 15 real-copy falsifiers are applied,
  pass on clean copies and refuse their corrupted copies. The 56 final
  mutant records were rejudged against the pinned sources, not rerun. The
  retained 55/56 attempt and the MSE-versus-bias fixture repair are disclosed
  correctly; the specification was unchanged.
- Final committed logs total .067878 core-hours, 123 jobs, peak 248 MiB;
  the report's .064 / 121 / 244 figures are an explicitly earlier as-of.
  Both fit and score are 238 MiB. The declared budget is satisfied.

No new model/candidate fit, real-tape loading, branch simulation, label build
or untouched-day access occurred in this review. The independent four-scalar
arithmetic only verifies the committed parameters. Passing these checks
establishes reliable execution of the declared experiment, not profitability.

**Answer to the next-step question.** There is a useful consumed-data step
left: expand representative exposure labels under the unchanged policy and
target. It is not another tuned candidate. Two admitted-state days cannot
support confident conclusions about persistent side-specific economics or
calibration stability. Additional dates already recorded as consumed can
broaden that evidence without spending untouched validation data.

The proposed build and its capped budget need the USER's go because the
approved `7ad03681` contract was one saved-data, one-core-hour challenger and
explicitly reserved expanded label/date/compute scope for a USER decision.
The concrete proposal is prepared now; no approval is needed merely to read
this review or close its nonblocking items. An untouched validation run needs
its own authorization and an economically defensible frozen candidate; this
candidate is not ready. A new target/model-family search is also not approved
by this review. Stop tuning **this scalar candidate**; stopping the entire
research line is a separate USER choice, not compelled by this result.

The USER's longer-holding direction remains a **conditional** research aim:
select worthwhile longer exposure using a forecast that agrees with the
exposure policy. Unconditional longer resting and this scalar rule have not
achieved that. The positive bid-side baseline on one consumed day is worth
retaining as a descriptive fact, not adopting as a new policy. Keep the
original horizon for the proposed coverage work; neither a longer timer nor
a shorter markout target is justified by this result alone.
