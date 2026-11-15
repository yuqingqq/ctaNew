# Stop EV data and model prebuild review

The revised local-value objective is useful, but the proposed label still
discards the execution ordering that distinguishes front from back queue
positions. Resolve that before fitting. This review covers the design revision
`cae5c88f`, especially sections 7 and 8. It supplements the earlier
`20261115_stop_ev_review.md`; neither the owning session's design nor running
code was edited.

At the review snapshot, the shared workspace contains the revised design but
no STOP-EV-specific builder, dataset receipt, fitted model or evaluation
artifact. The running other-session worker is STOP-PLAY v6, not STOP-EV.
Consequently, the findings below distinguish confirmed specification issues
from implementation checks that remain unverified. No data-quality or model
performance pass is issued.

## Confirmed label and evaluation issues

1. **Window-average markouts erase the queue-dependent target.** Design lines
   104-108 form stopping fills from total flow minus q, then multiply by the
   average markout of all stopping flow. An order behind q executes a different
   subset of trades, not a representative slice of all trades in the window.
   Likewise, stopping flow after a through trade cannot be credited before it
   merely because both occur within the same horizon.

   A hypothetical counterexample, not a measured result: two consecutive
   stopping trades each execute 20 shares. The first trade's signed markout is
   +1 cent/share, the second -1 cent/share. For a 20-share order, q=0 gives
   +20 cents and q=20 gives -20 cents under the stated FIFO assumption. The
   window average is zero and the draft labels both queue positions zero.
   Thus the approximation can destroy the very distinction STOP-EV should
   learn, including its sign. Use chronological partial fills, each valued at
   its own future mark, and stop allocating once the order is filled.

2. **Any through trade still implies a full residual fill.** The new formula
   subtracts stopping fills from Z, which improves on the old double-charge,
   but then assigns all remaining shares to through value if any through
   trade occurs. Actual remaining taker volume may be smaller and is shared
   across inner and outer orders. A small through leg is not a full execution
   of every rung. Cap fills by remaining order size and the shared executable
   trade quantity. Require exact synthetic reconciliation; a broad
   'within the stated approximations' tolerance must not conceal sign errors.

3. **The exposure horizon and continuation are contradictory.** Lines 89-90
   describe one second of fills 'if kept until the next decision', which is
   only 50 ms later. A one-second potential keep-value target requires a
   declared continuation, such as keeping this one order until that horizon,
   with no replacements. It can be used as a rolling local surrogate, but
   is not the exact value of keeping only until the next decision. Fills
   before cancellation could land reduce remaining size/queue but are not
   avoidable losses. One second of fills with one-second post-fill marks also
   requires future mark coverage beyond the exposure horizon.

4. **Fixed submitted size does not cover remaining size.** The draft's label
   fixes Z=30 and stage 2 lists market inputs, level and q. A partially filled
   30-share order has less remaining quantity; 100-share testing adds another
   unsupported population. Include client-known remaining Z with declared
   label support and units, or restrict each model to its supported size/state.
   The execution ledger may use true quantities to construct outcomes, while
   features must use only acknowledged fills. Margins and EV units must agree
   when quantities differ.

5. **The proposed economic gate is still a ratio of unlike percentages.**
   Harm avoided and good value given up have different denominators; their
   ratio is not net economic value. Use signed value on neutral action units,
   matched controls and a declared null of at least 200 draws. Treat every
   50-ms step and q-grid expansion as correlated training samples, not fresh
   independent evaluation actions. Deduplicate identical q points when shown
   size is zero, or assign explicit augmentation weights. Day-only intervals
   and chronological splits remain required.

6. **Top-maker outcomes cannot be a label sanity pass condition.** Lines
   116-117 expect positive front stopping fills near the observed top-maker
   value and clearly negative sweep fills. Top makers select different
   states, queues and trades; through fills can also profit on reversals.
   Use those comparisons as descriptive diagnostics. Validate arithmetic,
   signs, timestamps, bounds and controlled synthetic cases without requiring
   the real dataset to resemble a selected maker population.

## Required checks when the builder appears

- **Use the corrected queue engine explicitly.** Import
  `pf_stop_queue_activation_lane.sim_joint_g`, or use the versioned corrected
  replay entrypoint. The old `pf_stop_model.sim_joint_g` intentionally still
  reproduces the running historical experiment. Merely merging the fix does
  not change old imports. Every dataset/replay receipt must name its queue
  contract; new references and labels must not inherit old queue identities.
- **Separate causal queue features from execution state.** Initialize feature
  estimates from client-known snapshots; later additions must not push an
  existing order backward. Do not feed simulator `ahead`, true remainder or
  delayed trade reports into features. Explicitly test frames stamped before
  a decision but received after it, plus unacknowledged fills.
- **Propagate incomplete timing to action validity.** Excluding untimed/E2
  legs as counted events is insufficient if affected windows then become
  zero-fill labels. Missing flow also changes queue depletion and the order
  of fills. Keep explicit incomplete/ambiguous statuses, compare WS-only and
  broader timing variants on declared populations, and account for each leg
  and each exclusion before deriving local value.
- **Carry forward corrected market data.** Verify the actual feature files
  use the corrected relay basis/fair price and 30-second causal prehistory.
  The v6 fixes do not automatically repair a new builder's independent reads
  of old shared step-5 data. Filter reserved/boundary files before reading;
  check mark staleness and end support per tranche.
- **Measure coverage on order decisions, not just trades.** Report zeros and
  exclusions by day, side, level, q, remaining size, time in window and market
  regime. Real client queue estimates, partial orders and unknown depth must
  have declared support; q=0/middle/back of current visible size is only a
  hypothetical grid. The smallest unit of later policy evaluation remains
  the actual neutral action, not each grid copy.
- **Remove mechanical explanations in baselines.** Compare prediction against
  a training-only baseline conditioned on known price/edge, level, q and size.
  Causal scaling must preserve units and handle missing/zero depth explicitly.
  Use sample weights appropriate to the intended action population. Any loss
  choice based on label distribution belongs to consumed development data;
  freeze it before the subsequent evaluation, recording all candidates.
- **Enforce the exposure cap jointly.** Cancel-pending orders still consume
  capacity and can fill. Independent per-level EV scores do not make joint
  inventory constraints independent. Preserve the existing hard-cap and
  notification-delay invariants when introducing EV-based allocation.

## Model architecture

The revised choice of one pooled model with side-aligned inputs and an explicit
side indicator is not itself an issue. It is an alternative to separate bid
and ask fits, consistent with the owning session's recorded user decision.
Similar sweep-classifier AUC on both sides does not prove local-value symmetry.
Validate bid/ask signs, calibration, support and net action value separately;
choose any split on development/calibration data and preserve later evaluation.

Direct local-value regression, keep-on-near-zero behavior and data checks
before fitting are improvements in the revised design. They do not compensate
for approximate labels that remove FIFO chronology. The first build should
produce a quantity-conserving chronological label ledger with synthetic
positive/known-bad controls, an action/leg reconciliation receipt and a coverage
report. Fit only after that instrument passes. No claim that the current
market-feature normalization is faulty, or that a stronger learner is needed,
is supported by this review.
