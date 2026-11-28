# Model upgrade goal — 2026-11-12

The user authorized building and evaluating a better model. The goal is early
default quoting, useful queue retention and selective withdrawal. No live code
or trading changes are included. Worktree: `/tmp/cta-model-upgrade-20261112`,
branch `research/predict-model-upgrade-20261112`.

## First experiment: isolate target and temporal information

The independently running continuation v3 study provides full-engine paired
KEEP/PULL labels on outcome-independent sampled neutral generations. Its source,
declaration and build receipts must pass identity/reconciliation checks before
consumption. Do not modify that running study. Its one-second re-entry rule is
an explicit first-stage comparison; this does not solve recovery-based re-entry.

Use the same eligible identities for all comparisons. Preserve the complete
population ledger, sampling, censored and missing statuses. All historical dates
are consumed: Oct 23–27, Oct 31, Nov 1–3. Outer tests are the last six dates;
the latest earlier date is calibration, with strictly earlier core training.

Predeclare four candidate score families, with pinned model parameters:

1. STATIC2: original observable features, excluding inherited learned guard
   values and guard margins, direct full-engine two-second value target.
2. STATIC10: identical inputs/model on the ten-second value target.
3. HISTORY10: STATIC10 plus fixed causal temporal and side-aligned features.
4. JOINT10: same HISTORY10 tree partition, coherent empirical paired-outcome
   distribution, with one-percent core prior and earlier-day joint calibration.

HISTORY10 is the primary candidate. This is not a search over hyperparameters.
STATIC2 versus STATIC10 isolates horizon; STATIC10 versus HISTORY10 isolates
temporal information; HISTORY10 versus JOINT10 isolates the estimator. The
joint version is diagnostic and cannot become primary after seeing outer tests.
All examples, including no-fill and quiet actions, enter training. No
oversampling of adverse fills, sign clipping or test-day threshold fitting.

Temporal features use current and trailing 100/250/1000 ms observations from
the existing declared decision-time grid, with explicit staleness and missing
history. They include signed flow/imbalance/microprice changes, return and flow
acceleration, quote adjustment relative to Binance, same-price touch depth
changes where observable, and interactions with known order state. Never use
latent simulator queue, future trade labels, inherited guard scores, or silently
interpret a new price level as depletion of the previous one.

Rebuild the original sweep-value guard benchmark on earlier days only, without
its inherited LODO predictions. Its target remains the legacy approximate sweep
target; it is a benchmark, not proof of action-level causal labels. Match its
ranked cancellation count on the new neutral population. The frozen historical
guard is a separate descriptive comparator with its existing caveats.

The primary local policy uses a fixed zero threshold, with value calibration
reported separately. Compare net action utility at matched cancellation counts
within day/side/hour/age/inventory sign. Declare 500 permutations and 1,000
UTC-day bootstrap draws. Report value calibration, favorable value forfeited,
adverse value avoided, all zero-outcome actions, horizon sensitivity and every
population/status count. Local utility sums are not portfolio P&L.

## Checks and subsequent stages

Before numerical output: commit the complete pipeline and declaration. Positive
and refusal controls cover feature availability/future mutation, same-price
depth, duplicate actions, chronological core/calibration/test boundaries,
artifact identity, saved-model reconstruction, matched controls and valuation.
No protected payloads are read. Use the existing Codex research slot only when
free; do not interrupt or duplicate the current continuation or QR2 runs.

Then freeze and run the primary candidate in full sequential replay, primary
and conservative execution assumptions, at unchanged size and inventory limits.
Report reference/default/matched guard and candidate results on common marked
windows plus missing statuses, full settlement, inventory-square time, favorable
and adverse markouts, union quote presence and cancellations. Frequent rescoring
must be tested explicitly: one neutral probe per generation does not by itself
validate repeated sequential decisions.

The next target extension is a 250 ms avoidable fill-value forecast and explicit
WAIT/recovery alternatives. It requires generation-linked fill events,
state-dependent effective cancellation and paired full-engine reconciliation;
the 250 ms fill-probability field alone is not a danger-value label. Do not invent
that label from aggregate continuation values or add arbitrary queue bonuses.

Fresh wallet-event capture remains necessary for live queue/fill/cancel
calibration. No simulated improvement is a live-validation claim. A successful
research candidate requires a full pipeline freeze and at least five later,
untouched complete days before any forward validation claim.

## Pre-fit correction, v2

Source inspection found that `rows_rl_d2_pk.gap` uses the registry start price
from the beginning of a window. The historical `preds_t60k` transformation
expresses unknown start price by writing -1 into the guard values; the default
policy checks their availability but ignores their sign. Thus dropping learned
guard columns alone does not establish contract-context causality.

Before any test, data build or fit of this upgrade, supersede its declaration.
Rebuild the gap feature from actual received `predictMarketChanged` startPrice
patches using the declared Tokyo availability clock, cached Binance mid, trailing
basis and volatility. Before a price is observed, gap stays missing. Later price
corrections affect only later feature rows. Do not substitute a final registry
price or infer an earlier REST value from the final record. The raw capture
hours and clock files are bound as inputs; missing captures are explicit.
Apply the same correction to the chronological guard's training examples.

Also bind that benchmark to `badfill_info_tx_tk/grid.parquet`, the target artifact
actually used for `preds_t60k`, rather than the superseded initial sweep grid.
The new model declaration is `reviews/model_upgrade_20261112_v2/declaration.json`.
The original declaration and stopped waiter logs remain intact. The driver now
respects the continuation driver's ownership between numerical workers.

## Additional execution sensitivity, declared before new model results

The B1a live-order audit and the20261112AGE attribution show that very young simulated fills are a material reliability risk. Add one separate full sequential stress, preserving the original two scenarios: primary execution clocks/queueM, but zero own fill credit before500ms of actual order age. Public prints can still deplete queue ahead; suppressed fills do not change own quantity, cash/inventory or send a fill notification. Later decisions and fills evolve in the engine. This is a sensitivity assumption, not a calibrated venue rule or formal PnL lower bound; real young fills do occur. Do not apply B1a's partial-day precision as a per-fill probability.

No model, threshold or policy is selected or changed. The same five frozen policies run on the same six consumed dates with unchanged size/inventory constraints. Before the stress, a zero-floor eight-window control must exactly match all five policies' original primary replay artifacts. Age500smoke/full runs must reconcile portfolios, show nonzero suppressed opportunities and contain no younger fills. Compare HISTORY10 to STATIC10 and the chronological guard at matched constraints; report whether the original objectives also survive this condition. Existing declarations/results remain intact.

Source freeze a36dec60; separate declaration `reviews/model_upgrade_age_20261112/declaration.json`; durable driver waits behind the original model/replay/component chain. Synthetic boundary/quantity/cash/refusal controls passed. Full no-op parity and age-sensitive model replays remain pending.

Settlement reporting correction: sourceac339f20 and `settlement_audit_declaration.json` preserve all frozen paths/targets and supersede final payout/carry fields using0.5for equal-price windows. Current final primary/age receipts will be `result_settlement_v3.json`; original `result.json` remains raw registry valuation. No primary marked-value or risk-predicate change. See execution record for controls and scheduling.

## Empty-fill age-check correction, v3

The age500 smoke simulation completed, but the post-run age validator failed on legitimate empty object-typed fill columns. Source dbffb3ac and `age_check_v3_declaration.json` preserve the numerical engine and all original scientific source files. The corrected checker accepts empty tables with an age column and retains zero counts/null minima; nonempty numeric, finite and minimum-age requirements remain enforced. The completed smoke artifacts were recovered without rerunning the engine, with an in-band superseding receipt.

Operational scheduler `model_eval_driver_v3.py` routes only the eight age run commands through `pf_model_age_v3.py`; all other tasks remain identical. Versioned settlement/presence waiters recognize that scheduler. Read the execution record for process identities, control evidence and the pending isolated entrypoint parity check. The correction changes no model, policy, threshold, fill-suppression assumption, inventory limit or success criterion.

The corrected age entrypoint integration gate is now complete: `reviews/model_upgrade_age_entry_parity_20261112/parity.json`, as of 2026-11-12T16:45:28.875122+00:00, verifies exact values and bytes for all22artifacts on five policies × eight windows, with identical generated engine source. This supersedes the earlier pending status of that implementation check; full age strategy results remain pending.
