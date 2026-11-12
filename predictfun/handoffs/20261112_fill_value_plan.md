# Fill opportunity and quote value — first-stage plan

User authorized proceeding after the queue-priority clarification. Quote by
default and retain valuable priority remains the objective; restoring the old
entry gate is not the experiment.

This first stage tests whether fill opportunity is learnable from the existing
causal observations, and whether separating value components improves the
KEEP-versus-RESET decision. It holds the current short-horizon labels fixed so
that extending the horizon does not confound the model comparison.

## Frozen design

- Source freeze: `4ac8a3b6`; declaration: `ebeedcbe`.
- Neutral build dates: 20261023–27, 20261031, 20261101–03. Six expanding test
  folds: 20261026, 20261027, 20261031, 20261101–03. All dates are consumed.
- Original MARKET features versus MARKET plus the four observed queue proxies.
  Hidden simulated queue state is never an input.
- For each KEEP and RESET branch, fit five heads: any-fill probability, filled
  fraction, squared filled fraction, positive branch markout, negative branch
  markout. Every head trains on every eligible action, including zeros.
- Preserve the pinned model parameters and 300 rounds; only the fill-event
  objective changes to binary. Two feature sets x two branches x five heads x
  six folds = 120 models. No hyperparameter or threshold selection.
- Constrain probability and quantity moments to physical support. Save raw
  forecasts and every adjustment count. Predict the inventory-square penalty
  using the first and second quantity moments, not the square of mean quantity.
- Expected branch value equals expected positive net markout minus expected
  negative net markout minus the inventory penalty. The positive/negative split
  is of aggregate branch markout, not gross gains/losses on individual tranches.
  Fill probability is separately tested and constrains quantity support; there
  is no assumption that quantity and price movement are independent.

## Evaluation

Compare probability calibration, Brier/log loss, ranking and quantity errors
against training-only state-stratified means. Compare the two feature sets on
identical actions. Report fixed calibration bins, every upstream exclusion
status and exact zero-fill counts.

For local decisions, compare component-value forecasts with the saved original
direct-value predictions. Use the fixed zero threshold, 500 matched-count
random shuffles, and exact-count rankings from the old queue model and guard
value margin within day/side/hour/age/inventory-sign strata. Bootstrap only the
six UTC days, 1,000 draws. These sums are overlapping local counterfactual
utility, not portfolio P&L.

Every saved model is reloaded and its test predictions reconstructed. Verify
source, input, fold and action identities; exercise perfect/reversed forecasts,
invalid probability, overfill, duplicate action, chronology and feature leakage
refusals. Original direct-model scores must reproduce on the same test rows.

## What this stage does not establish

The saved labels describe fills within the existing approximately two-second
window, including early invalidation, under the primary simulation assumptions.
They do not contain time-to-fill targets or a raw future-volume forecast. The
experiment does not yet estimate later queue continuation value or fix the
same-price RESET versus current-touch re-entry mismatch. No actual-policy or
live capture calibration, sequential portfolio improvement, or untouched-day
validation is claimed.

The original frozen research and all earlier negative results remain intact.
Results and status are tracked in `reviews/fill_value_20261112/` and
`data/predict_fun/research/fill_value_20261112/`. Large models and predictions
remain outside Git. The Codex worker uses core 8 under the authorized global
two-process cap. No live strategy, loss budget or rebate accounting changes.

## Reproduce

From this research branch, with original saved neutral artifacts available:

```bash
python3 -m predictfun.research.pf_fill_value_spec
PF_SLOT=codex predictfun/research/run_light.sh fill_value -- \
  python3 -m predictfun.research.pf_fill_value
```

The historical worktree launcher has an older single-process guard; this run
uses the shared root's updated two-slot launcher and executes the module from
the isolated research worktree. Declaration and result paths refuse overwrites;
corrections require a superseding version. See the final human review before
using any result for a decision.
