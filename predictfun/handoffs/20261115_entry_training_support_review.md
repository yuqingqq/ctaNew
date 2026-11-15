# Entry training support: limited history is plausible, not established as the cause

Checked 2026-11-15T05:16:37.782981+00:00; data/model snapshot 2026-11-15T05:04:52Z. No new model fits or candidates.

The identified coordinate and missingness-handling gaps are addressed in the
CONDITION candidate. This does not clear the entire representation/data problem:
actual missing histories and reference information remain, ask timing worsens,
and the candidate has not been promoted. Explicit conditions do not guarantee
that a fitted model can separate economically favorable and unfavorable actions.

The current learner uses all available earlier labels: **2 to 6 labeled days**.
There is no deliberate rolling cap that discards a larger same-target history.
The final fold uses Oct 25/26/27/31 and Nov 1/2, with Oct 28–30 absent. The
earlier bootstrap label days have different continuation lineage; they must
not be appended as if they were the same labels.

| Evaluation date | Training days | Training rows bid / ask | Weight-only effective rows bid / ask | Later-entry rows bid / ask | Timing R2 bid / ask |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261027 | 2 | 2065 / 1975 | 566 / 523 | 324 / 284 | -0.0825 / -0.0305 |
| 20261031 | 3 | 3064 / 2968 | 826 / 788 | 461 / 430 | -0.0640 / 0.0005 |
| 20261101 | 4 | 4018 / 3894 | 1053 / 1017 | 564 / 542 | -0.0530 / -0.0288 |
| 20261102 | 5 | 5003 / 4859 | 1287 / 1248 | 668 / 646 | -0.1171 / -0.0809 |
| 20261103 | 6 | 6068 / 5850 | 1587 / 1506 | 847 / 783 | -0.0879 / -0.0473 |

These are overlapping expanding training sets, not disjoint samples. Effective
rows here equal `(sum w)^2 / sum(w^2)`, with initial/later weights 2/32. This
only discounts unequal weights; correlated actions/windows/days mean it is not
a count of independent observations. Uniform-weight positive controls and an
invalid-count refusal were checked. Counts reconcile to saved preprocessing
populations and the exact same input receipts used by the representation study.

Evaluation n = 9,871 actions (4,990 bid / 4,881 ask) across five consumed days.
Eligibility and source exclusions are unchanged; see the population accounting
in `20261115_entry_representation_review.md`. No row or outcome filtering.

This is limited support for a 351-column model, especially when only 284–847
later-generation training examples are available per side and fold. Column
count is not independent dimensionality, and no sample-size sufficiency rule is
asserted. More representative training data is a plausible improvement.

However, timing R2 does not improve monotonically as these folds grow, and
the evaluation day changes at each step. Therefore this table is not a learning
curve and cannot establish whether training history is the cause. Current
errors can also reflect target noise/continuation effects, weak conditional
signals, regime changes, model complexity and missing state information.

The nominal 30-second feature history is a separate issue. Partial-history
flags correctly expose the available observations, but adding more training
days does not restore the missing pre-window frames or unknown reference.

The discriminating next test is a **fixed-evaluation-date learning curve**:
freeze representation, target, learner and sampling; compare nested earlier
training histories on identical evaluation actions. Where only the final date
has enough earlier label days, call that a development diagnostic, not validation.
Expand comparable earlier labels to repeat the test across multiple dates.
Test denser later-generation sampling separately, so more days and improved
state coverage are not confounded. Keep all existing rows/rare losses and use
the actual frozen keeper when later aligning combined entry targets.

No such learning-curve comparison has been run in this review. Current status:
known representation changes implemented and checked; training support is a
clear limitation; the sole cause of poor prediction is not identified.

Evidence: `reviews/entry_training_support_20261115/result.json`; supporting
model/fit receipts remain under `reviews/entry_representation_20261115/`.
