# Training-only cleanup comparison — declared after first-fold diagnosis

User asks whether noise filtering is needed. Two new development candidates:
CLEAN_TREE and CLEAN_RIDGE. The existing three-arm comparison is unchanged.

Inputs: same repaired labels, neutral action population and inverse sampling
weights; no rows removed, no outcome clipping, no filled-only targets. Use the
prior signal study's fixed compact economic transformations, known-time public
liquidity scaling of public queue sizes, remove basis_bps (already unstable in
prior development audits) and redundant absolute book sizes, compact per-rung
client quantities/price/age/pending/queue/wait state. Train-only robust feature
bounds, median/IQR and missing/extreme indicators; remove training constants and
exact duplicates. No temporal smoothing across unrelated action rows.

CLEAN_TREE uses exactly pinned LightGBM settings. CLEAN_RIDGE uses fixed alpha100,
weighted fit, unpenalized intercept, all raw labels. Same separate side heads,
direct absolute POST and POST-minus-WAIT targets; POST only if absolute positive
and timing nonnegative. No threshold or hyperparameter tuning. The cleanup
changes inputs; the ridge arm additionally changes the learner. This does not
fix neutral KEEP/terminal target alignment and is not presented as doing so.

Five consumed chronological test dates and folds identical to the state study.
500-draw null matched day/side/hour/rung/generation group/action count/weighted
mass. Actual n and effective n; baseline errors and selection value; all arms
reported. Two new candidates, five in this repaired-state study including the
original three. No portfolio promotion or forward validation.

The supervisor yields only after starting the final label-day worker, lets it
finish, uses codex/core8 for tests and fits, then resumes the original supervisor
for its remaining fit/report. Identity-bound cleanup runs in finally and service
ExecStopPost. No source or output of the original experiment is overwritten.

Three synthetic checks: weighted linear positive control and bad weights/schema;
training-only preprocessing invariance plus a known-bad refit; public queue-scale
invariance while own quantities stay unchanged. The original signal processor's
existing synthetic checks also remain applicable. Source/freeze committed before
any cleanup fit. Current first-fold diagnosis makes this work post hoc.
