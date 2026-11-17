# Review of the gain/loss diagnosis

Reviewed `fe325d18` / `3b8df784` (gain/loss model and results) and `9b987f65` /
`347acc7c` (diagnosis). The loss shortfall is real and the decomposition gives a
modest relative improvement. The stronger conclusion, “missing day-level loss
intensity, not coverage,” is not established by these diagnostics.

Review source `09c7a263`; evidence
`reviews/gain_loss_diagnosis_review_20261117/review_v1.json`. As of
2026-11-17T04:38:10.303Z, the review reproduces all saved full-model gain/loss
predictions on the 17,859 Oct26–27 actions, all three stored leave-day-out loss
ratios, the five market-grouped domain-classifier folds, AUC and familiarity
ratios. Source and input/model hashes reconcile. Positive controls and planted
numeric/chronology failures pass. The reconstructed domain scores and models
are retained under ignored `data/predict_fun/research/gain_loss_diagnosis_review_20261117`.
No new return model, policy, validation date or performance claim.

## Findings

### 1. The “next-day” diagnostic contains future-trained folds

`pf_gain_loss_diag.py:102` trains on every day except the held-out day:

| Held-out day | Training days | Evaluation n | Predicted / realized loss | Strictly forward? |
|---|---|---:|---:|---|
| Oct23 | Oct24, Oct25 | 9,039 | .825 | No |
| Oct24 | Oct23, Oct25 | 9,259 | 1.083 | No |
| Oct25 | Oct23, Oct24 | 8,938 | .748 | Yes |

This is a legitimate cross-day leave-one-out diagnostic, but two folds cannot
support the `NEXT-DAY-UNDERPREDICTION` interpretation. Only one of these three
folds is forward. The existing Oct26–27 model tests remain chronological; this
finding does not mean their original predictions used future training data.
Use explicitly past-only folds for the next-day claim and keep the old ratios
as descriptive leave-day-out evidence.

### 2. Relative familiarity does not rule out a coverage problem

`pf_gain_loss_diag.py:141` constructs quintiles entirely within the evaluation
set. Q1 means “less distinguishable than the other evaluation rows,” not “well
represented by training.” The same classifier strongly separates the 27,236
training and 17,859 evaluation states: reproduced AUC .927296.

Its training-score median is .0349. The median of the supposedly most familiar
evaluation quintile is .3837 (n3,572); Q2 spans .5359–.7231 (n3,572). These
scores are diagnostics, not calibrated overlap probabilities, but they expose
why relative quintiles cannot certify familiar support. The original P3 tests
only one monotonic relationship between score and aggregate loss ratio. Failure
of that predicate does not show no coverage gap.

Nor do broad one-variable strata hold all 87 inputs fixed. Their residuals
cannot establish “the same states lose differently” or identify an omitted
day-level factor. Finite-sample estimation, model error, covariate shift and
conditional drift remain possible. A recent-loss signal is a plausible
hypothesis, not a demonstrated cause or remedy. More training days are not
ruled out as useful by this experiment.

### 3. G2 matches total count, not market/side exposure

`pf_gain_loss.py:307` groups by day and action kind, then selects a global top
fraction and samples its null from that whole cell. Neither the comparator nor
the null holds market, side or time-of-day participation fixed. This is weaker
than the repository's matched-control requirement.

For example, Oct26 posts at 10% select 400 actions in each method, but their
counts differ in 324 of the 393 market/side cells selected by either method.
This may be a useful change in allocation; it does not isolate better order
selection under matched conditions. Recompute a comparison with common
market/side/kind budgets and the corresponding matched null before treating
G2 as a controlled economic improvement. Original G2 values remain valid as
global equal-count descriptive selections.

### 4. PASS is a limited relative screen, and label checks are sampled

The code intentionally defines PASS from G1 and G2, excluding G3. Thus PASS
coexists with failed loss calibration and negative selected-post value. It is
not evidence that the model is ready for a controller replay or profitable.
Set L scores focal wealth; the full action value including other-order/risk
terms is checked only on the separate 861-action set S.

The label receipts record 56 and 79 canonical checks on Oct26/27, including
3 and 9 positive-quantity checks, plus exact label agreement on 764 actions
shared with the previously audited set. They do not record 17,859 independent
canonical checks. There are 1,270 / 1,163 unavailable-client-state posts retained
as exclusions; eligible day populations are 8,940 / 8,919. Describe the zero
mismatches as occurring in the sampled checks.

The training receipt's `no_fill_rows` is actually `abs(W) <= 1e-9`: 23,971
zero-wealth actions. Actual zero affected quantity is 23,896; 75 filled actions
have zero focal wealth. This is a count-label issue; the full training population
was retained, so it does not by itself invalidate the fit.

## A more specific next diagnosis: event frequency versus amount

The saved component heads allow an exact aggregate accounting distinction.
For severe excess, write the predicted/realized total ratio as:

    (sum predicted severe probability / actual severe count)
      * ((sum predicted severe excess / sum predicted severe probability)
         / (actual severe excess / actual severe count))

On the same 17,859 actions and review timestamp above:

| Day / action | n | Severe-event count ratio | Excess per expected event ratio | Total severe-excess ratio |
|---|---:|---:|---:|---:|
| Oct26 cancel | 4,946 | .762 | 1.048 | .798 |
| Oct26 post | 3,994 | .835 | .955 | .798 |
| Oct27 cancel | 4,870 | .809 | .913 | .739 |
| Oct27 post | 4,049 | .739 | .890 | .658 |

The frequency head underpredicts severe-event counts in every cell. Conditional
amount accounting is closer, especially on Oct26. These are ratios of sums and
include composition effects, not causal attributions or proof the amount head
is conditionally calibrated. They show why a single unspecified “loss intensity”
multiplier or an amount-only repair is premature.

The next experiment should first use past-only predictions and labels to test
whether recent severe-event probability errors and severity errors persist into
later decisions. Compare a frozen training-only calibration with one predeclared
causal rolling correction; account for the component separately. Use only
neutral-reference labels whose endpoint and necessary observations have already
arrived. Never use the completed day's loss total, unreceived labels, or only
the strategy's selected fills as the adjustment population. Keep the controller
fixed and repair matched economic controls before claiming a strategy benefit.

If lagged errors carry incremental predictive value, a recent-loss feature or
running calibrator earns further testing. If they do not, day-to-day variation
alone is not a reason to add that input. Current dates are consumed development;
prospective validation still requires a frozen pipeline and at least five later
untouched complete days. No row-level confidence interval is warranted here.
