# Latest commits and next priority reviewed — 2026-11-19

Target: **0528e1a614d582d92cf3efbcd26d6afdf7ee8686**.
Range: **1785003084c8c17d64e6f89860ca42c35609e290..0528e1a614d582d92cf3efbcd26d6afdf7ee8686**,
covering R0 repair `dbe77f9c`, review-module declaration `6775eabd`, and its
results `0528e1a6`. Source was pinned in `/tmp/pf-review-0528e1a6`.

**R0 is complete. The prior two defects are fixed and the saved loss figures
reproduce. These commits review the restricted POST-versus-NO_ENTRY experiment;
they do not build the adaptive WAIT/re-entry pipeline. The next priority is R1's
branch engine, with the small review-screen correction below included in its
preflight work. There is no reason to refit the old candidate to repair these
diagnostics or to open validation data now.**

[Reproduction script](verify.py), [artifact receipt](receipt.json),
[synthetic cases](synthetic.json), [pinned selftest](selftest.log).

## Resolved versus open

- **Resolved:** `pair_qm` is rebased at the current decision's mid; all 5,723,835
  saved label rows across Oct24–27 satisfy the value identity, with finite
  residuals below 1e-6 cents. Other shared label columns are unchanged.
- **Resolved:** each head's reached mask applies its own POST admissions. A's
  saved reached counts are 67,776 / 76,731, matching the earlier independent
  review. The old never-posted contamination is gone.
- **No refit:** 44 actual old/new model files hash-identical; saved prediction
  fields, admissions and policy value tables unchanged. The restricted
  alternative is now accurately named NO_ENTRY.
- **Improved review:** matched-null dispersion, fill-age, cancel-pending,
  trade-source and tail tables are now present. Source hashes connect the
  pre-run declaration to both saved review outputs.
- **Still open:** genuine 50 ms WAIT, later entry, re-entry, branch-local
  continuation and shared inventory. No claim that R1–R4 of plan v2 is complete.

The previous coherence/round-trip correction remains present. No new
result-changing fill-accounting defect was found in these saved runs. The
validity PASS describes their declared checks; it is not venue fill validation
or proof that every future input is handled correctly.

## Findings, ranked by severity

### 1. P2 — the new episode-balanced screen tests zero in the wrong unit

`pf_pipeline_paired_review.py:240` returns NOT_EVALUABLE when the **raw row sum**
of labels is zero. Lines 244–245 then assess sign and bias using
**episode-balanced means**. Those statistics need not have the same zero or sign.

Independent supported counterexample:

- 100 episodes contribute two +1-cent rows each.
- Another 100 episodes contribute one −2-cent row each.
- The raw sum is zero, but the episode-balanced mean label is −0.5 cents.
- Every prediction is −0.5 cents. All 200 episodes have nonzero outcomes.
- The selected negative-action group satisfies the declared support, sign and
  amount rules, yet the function returns NOT_EVALUABLE.
- Giving each episode one row preserves the same episode means and predictions
  but changes the verdict to PASS. The true all-zero positive control remains
  NOT_EVALUABLE.

Use the declared episode-balanced label mean for the zero/sign/amount decisions,
with finite-input and support handling explicit. Add both representations and
the genuine-zero control to the screen tests. **None of the currently saved
prediction groups is affected**, so this does not change the present candidate's
FAIL or explain its losses. Repair it before reusing the screen for R3; no model
training or full replay is needed for this correction.

### 2. P2 — the report extends fixed-admission nesting to a different future control

The new result report at lines 70 and 99 says plan v2's ENTRY-PA comparator has
the same nesting property as this run's same-admission A-PA, and recommends
replacing it. That extension is not established.

The current single-generation A-PA really does keep a subset of A's fills/live
time, so longer resting versus it is largely mechanical. But plan v2 section 7
explicitly defines **ENTRY-PA with its own state and re-entry**, where pulling
earlier can change inventory, available capacity, later prices and admissions.
The two full policies need not have nested quote histories. The plan explicitly
forbids forcing A's future admissions onto the control's incompatible state.

Keep the current nesting caveat. For R1/R4, test actual policy/state identity
before asserting nesting; retain ENTRY-PA's economic role and the nontrivial
PI0-50/NEU2 comparisons. A quote-uptime comparison with never posting is trivial
and is not a substitute for a useful quoting comparator. Neither an uptime
increase nor non-nesting alone demonstrates predictive value.

No historical value needs changing. Correct this interpretation before changing
the prospective gate on that basis. The existing mean-only null comparison is
also a different declared estimand from the new p<=0.05 reading; keep their names
and prospective declarations separate instead of retroactively changing a PASS.

### 3. P2 — B1a sensitivity is described as establishing the actual loss level

The result report at line 35 says the historical 8–12% young-fill realism result
means this replay overstates **A's** loss. Repository FINDINGS identifies B1a's
evidence as one day of live orders from an older churning engine. It is not a
validated per-fill probability for these two days and this new order population.

The reproduced findings support a narrower statement: removing/downweighting
young fills changes the reported loss level, and A still trails the named
controls under the declared weightings. They do not identify actual live loss
or establish a calibrated correction. Missing/spurious fills can change later
queue, partial-fill and action states; multiplying saved fill values is not a
new causal replay. Keep B1a-weighted values labelled sensitivity results, with
the original population/engine limitation. This does not reverse the failed
candidate verdict or justify another model fit.

### Additional P3 verification/documentation points

- `pf_pipeline_paired_review.py:166` calls `np.isclose(..., atol=1e-6)` without
  setting `rtol=0`. A synthetic END shifted by **0.01 ms** is accepted while a
  1 ms shift is rejected. The routine therefore does not enforce a purely
  absolute microsecond-scale tolerance. Make the intended tolerance explicit;
  this counterexample does not demonstrate an erroneous saved timestamp.
- The report at line 73 says K7 still lets NaN pass through a `>1e-6` test.
  The pinned committed source at `pf_pipeline_paired.py:309` actually uses
  `if not (abs(idn) <= 1e-6).all()`, which rejects NaN. The `>1e-6` expression is
  only the printed mismatch count; it can undercount NaNs, but it does not let
  the run pass. Correct this stale description rather than reopening an already
  closed rejection path. K7 remains an algebraic identity check; K4's independent
  fill-value/quantity checks provide the additional evidence.

## What the new results establish

The new module is a richer review of the same frozen candidate, not evidence of
a newly improved predictive model. At B (M50, activation58/cancel37/notice100 ms,
30-share JOIN), local marked values are:

| Metric, cents unless stated | Oct26 | Oct27 |
|---|---:|---:|
| A | −2,032.12 | −11,865.45 |
| Same-admission A-PA | −1,053.02 | −6,943.81 |
| Additional value from A's retention | −979.11 | −4,921.64 |
| Matched-null mean, A cancellation held fixed | −11,025.73 | −13,431.47 |
| Exact matched-null standard deviation | 2,691.74 | 3,055.50 |
| One-sided conditional null p, 1,000 saved draws | 0.000999 | 0.304695 |

The null means/variances and saved-draw tail counts reproduce independently.
This is a conditional matched-selection diagnostic on consumed days, not a
day-cluster confidence interval or forward profitability result. The new
p<=0.05 reading is supported on Oct26 only; the old above-mean statement remains
literally true on both days.

The loss concentration is informative: there are 167 / 182 severe episodes,
and their largest negative fill-value component is after the earliest possible
cancellation effect (−12,645.64 / −18,384.76 c). This supports studying retained
exposure and continuation; it is not proof that the model could identify every
such loss in advance. Total unconditional loss amounts are 25,243.34 / 32,179.63 c;
these differ from net value because gains offset some losses. Spread/phase
associations alone do not justify a new rule that refuses most of the market.

The report's “spread component” is based on the posting mid, and movement runs
from that mid to END. It is not measured spread at each fill or the requested
100 ms/1 s fill markouts. Time-integrated inventory is summed over independent
order episodes, not a netted continuous bid/ask portfolio. Keep these units clear
when assessing model versus inventory performance.

## Checks, populations and limits

All checks used `PF_SLOT=codex` through `run_light.sh`, absolute script paths and
pinned committed imports. Three bounded checks completed: independent synthetic
cases, saved-artifact audit, and the new review module's synthetic selftest.
Total about 44 seconds, peak RSS 878 MiB. No model fitting, raw tape experiment,
full mutation-suite rerun or untouched-day access. Synthetic passes are
implementation evidence, not profitability evidence.

Verified actual 44 model pairs, all four saved label files, evaluation-day
prediction fields, receipt/source/input/output hashes and committed review
copies. Independent saved-fill accounting reproduces A at B; independent ledger
arithmetic reproduces tail totals, severe-window partitions and value components.
The 19 alternative scenarios were not rerun. Prior independent all-row value/
quantity reconciliation plus the unchanged label columns supports the R0 closure;
the new module's V1 itself samples orders, as its receipt correctly states.

Population: original opportunities **30,242 / 29,820**, with-path **29,832 /
29,338**, A reached rows **67,776 / 76,731**. These are not counts of independent
profitable trades. Review outputs as-of **2026-11-19 16:15:22 / 16:15:59 UTC**;
this artifact audit as-of **16:43:38 UTC**. Teacher Oct23, candidate fit Oct24–25,
evaluation Oct26–27; all consumed. There are two evaluation days, no forward
validation and no live-profit claim. Repository changes and unrelated data-file
deletions from other sessions were preserved.

## One concrete next priority

**Develop R1: an executable 50 ms WAIT/re-entry branch engine with common END.**
The first reviewable deliverable is code plus synthetic event/ledger traces—not
a new model fit. Require:

1. WAIT actually enters later in a valid test; PULL can re-enter only after its
   pending state permits it.
2. Both branches preserve the same absolute END and continue under a frozen
   causal policy from their own queue, cash, inventory and reservation states.
3. KEEP preserves queue identity; replacements reset it; partial fills and
   cancel-pending fills reconcile exactly without shared-volume duplication.
4. Both sides and timing boundaries have positive controls and known-bad
   refusals, followed by a bounded check on already consumed material.
5. Include the small weighted-zero screen fix and report clarifications in the
   preflight receipt; do not use them to justify another R0 model rerun.

Then proceed to R2's supported remaining-horizon teacher/labels, R3 prediction,
and R4 own-state evaluation under the committed plan. The user's instruction to
fix issues and run the pipeline already establishes that direction; a new
permission round merely to start R1 is unnecessary. The development session
owns implementation. This review commits evidence and review-owned state only;
it does not implement another session's changes, restart scheduling, push or merge.
