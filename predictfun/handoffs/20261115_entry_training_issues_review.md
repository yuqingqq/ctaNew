# Entry training: coverage repair and complete known-issue register

Review as of 2026-11-15T05:32:18.692064+00:00; model/data snapshot 2026-11-15T05:04:52Z.

The current coverage problem is primarily a **dataset-construction and sampling
problem**, with a deliberately limited date partition. It is not a short rolling
training-window parameter accidentally throwing away a large compatible label
history, and it does not by itself mean the raw recordings are corrupted.

It is fixable in part from existing admissible data: label more later-generation
opportunities and representative waiting states. More calendar coverage requires
additional comparable labels and a valid train/evaluation partition. Neither fix
can manufacture missing event observations or independently validate the label
simulator. Short history is not established as the sole cause of poor prediction.

No new labels, model fits, policy change or promotion were made in this review.
The deliverables are the source-bound issue register and a concrete versioned
density-repair specification. The latter is proposed, not implemented or tested.

## What limits coverage

1. The label builder samples initial generations at 1/2 and later generations at
   1/32. Large inverse weights correct that design but not the shortage of examples.
2. The learner already uses all earlier compatible label dates in its declared
   set: 2–6 training days. The excluded calendar gap is reserved by the loader;
   it is not simply missing raw data that we should load opportunistically.
3. Bootstrap label days use a different continuation. Appending them without
   reconstructing a causally valid same-policy label contract is not legitimate.
4. One delayed empty-generation probe under a neutral POST/KEEP reference does
   not guarantee coverage of repeated model-driven waits, joint emptiness or
   selective-cancel re-entry. More rows from the same narrow states are not enough.

The training-support snapshot contains 1,975–6,068 raw training rows per side/fold,
284–847 later-entry rows and weight-only effective counts of 522.6–1,587.1. These
overlapping expanding folds evaluate 9,871 actions (4,990 bid / 4,881 ask) across
five consumed dates, Oct 27/31 and Nov 1/2/3. Effective counts only account for
unequal weights, not correlation. Eligibility/source statuses remain as recorded
in the representation review; no further action or outcome filtering occurred.

## Known issues and repairs

This is the complete known-issue list for the entry task at this review, not a
claim that no undiscovered defect exists. Confirmed code paths, known scope limits
and unproven explanations are explicitly distinguished.

### C1: Later-entry opportunities are heavily thinned

Status: **confirmed**; priority: **high**.

Initial generation inclusion is 1/2; later generations 1/32. Fitting/evaluation weights are hardcoded to 2/32.

Weights correct the declared sampling probabilities but do not replace the missing examples; later-entry states have disproportionate weight and limited distinct support.

Repair: Version the sampler and per-row inclusion probabilities; label more later generations from the neutral reference, preserving existing probe ages and outcomes.

Evidence: `predictfun/research/pf_wait_post_state.py:335; predictfun/research/pf_wait_post_sampling.py:24`.

### C2: Few admissible, comparable labeled dates

Status: **confirmed**; priority: **high**.

The expanding learner has 2–6 earlier label days. Earlier bootstrap labels use a different continuation; sealed and forward dates are deliberately excluded.

This is a prepared-dataset/date-design limit, not a rolling-window parameter accidentally discarding a large compatible history.

Repair: Build more same-contract labels on admissible earlier/development data; define a separate later validation block. Do not append bootstrap labels or consume reserved dates automatically.

Evidence: `predictfun/research/pf_wait_post_v2.py:53; predictfun/research/pf_kp_s4.py:1570`.

### C3: Reference states and operational scope are narrow

Status: **confirmed_scope_limit**; priority: **high**.

One delayed probe per neutral POST/KEEP generation, primarily one empty rung while other slots follow the neutral prefix. Entry labels use 30 shares, six rungs and the first 60 seconds.

Repeated learned waits, simultaneous empty slots, selective-cancel re-entry, other order sizes and later contract phases are not established as representative.

Repair: Declare and sample additional joint-waiting/reference states without outcome selection; retain action identities and episode clusters. Separately test the intended size/phase.

Evidence: `predictfun/research/pf_wait_post_state.py:228; predictfun/research/pf_entry_value.py:69`.

### D1: Entry loader does not opt in to pre-read boundary filtering

Status: **confirmed_code_path**; priority: **high**.

The frozen dependency set contains SKIP_FOREIGN_PARTS=False and no assignment enabling it; all three entry builders call load_day directly.

Neighboring reserved files can be opened before row filtering. This is a read-boundary/lineage defect, not proof that their future prices entered eligible rows. Per-day clock estimates were also derived upstream and need separate lineage review.

Repair: Select allowed file hours before opening them in the entry path, audit upstream clock/source provenance, and require saved-row/label regression or superseding rebuilds.

Evidence: `predictfun/research/pf_kp_s4.py:1574; predictfun/research/pf_kp_s4.py:1678; predictfun/research/pf_wait_post_state.py:318; predictfun/research/pf_entry_information.py:155; predictfun/research/pf_entry_representation.py:115`.

### D2: Raw-to-feature causality verification is incomplete for this entry pipeline

Status: **confirmed_scope_limit**; priority: **high**.

Entry future checks perturb received histories and precomputed grid inputs. They cannot independently verify every upstream calculation. A separate stop-model audit recomputed a subset of upstream fields, not every entry field/population.

Passing downstream invariance is narrower than end-to-end verification of every input and nuisance clock.

Repair: Recompute upstream features from events known by the exact action time with planted early-clock failures; bind exact input definitions and source artifacts.

Evidence: `predictfun/research/pf_entry_information.py:132; predictfun/research/pf_wait_post_state.py:203; predictfun/handoffs/20261115_stop_model_input_review.md`.

### T1: Labels do not yet include the intended selective keeper

Status: **confirmed_target_scope_limit**; priority: **high**.

POST, WAIT and NEVER are terminal portfolio differences under neutral KEEP and one frozen bootstrap entry continuation.

They are coherent standalone entry labels, but do not estimate the entry value under the intended combined entry/cancel strategy. The continuation is not assumed optimal.

Repair: After keeper/continuation identity is frozen, relabel both action branches under the same downstream policy. Keep entry WAIT/POST separate from cancel/keep.

Evidence: `predictfun/research/pf_wait_post_state.py:273; predictfun/research/pf_wait_post_state.py:94`.

### T2: Sparse, heavy-tailed terminal targets include portfolio spillovers

Status: **confirmed_first_fold_diagnostic**; priority: **high**.

The existing first-fold noise audit used 4,040 training and 1,992 evaluation probes; terminal/external components contribute substantial variation. First-order decomposition keeps the terminal horizon.

Immediate entry signals may be weak relative to future policy/portfolio noise. It is not established that this noise is irreducible.

Repair: Measure fill opportunity, adverse movement and queue advantage separately and reconcile them to economic value; keep rare losses and no-fill actions. Do not train only on filled or ordinary-outcome rows.

Evidence: `predictfun/reviews/wait_post_state_20261115/noise/result.json; predictfun/research/pf_wait_post_state.py:260`.

### L1: Unknown visible-depth queue can become zero in the fill simulator

Status: **confirmed_code_path_impact_unmeasured**; priority: **high**.

lv marks prices beyond the visible fifth level NaN; place converts nonfinite queue to zero before the next-frame adjustment. If it remains unknown, zero persists.

Model inputs preserve unknown queue, but labels may depend on an optimistic queue assumption. Current affected-label/P&L magnitude has not been measured.

Repair: Record unknown-queue statuses in label paths; evaluate conservative/optimistic queue bounds or an empirically justified model, then rebuild affected labels. Do not silently filter the population.

Evidence: `predictfun/research/pf_wait_post_state_lane.py:127; predictfun/research/pf_wait_post_state_lane.py:140`.

### L2: Execution clocks and public-book fill rules are uncertain

Status: **documented_assumptions**; priority: **high**.

Current representation comparison reuses M50 labels; public snapshots, E2-estimated anchors and assumed Binance receipt latency limit timing precision. Untimed source legs remain explicitly excluded.

A deterministic learner can fit an assumed execution model whose fills differ from reality; missing executions may also affect source coverage.

Repair: Keep WS/E2 provenance, assess M0/M50/M250 and queue-rule sensitivity, and reconcile against recorded execution evidence before economic claims.

Evidence: `predictfun/research/pf_wait_post_state.py:94; predictfun/FINDINGS.md:C1-C15,P12,P15,P19`.

### F1: Missing observations remain after representation changes

Status: **confirmed**; priority: **high**.

Of 9,871 evaluation rows, 7,857 have partial liquidity history, 4,758 lack the start reference and 2,031 lack complete 250ms candidate history. Loader retains only five seconds of pre-window books.

Availability flags expose the missingness but cannot recreate known-time information. More training days do not restore those within-window observations.

Repair: Recover longer causal history/reference coverage only where captured and admissible; retain missingness/statuses otherwise. Audit the frame-weighted liquidity estimator separately.

Evidence: `predictfun/reviews/entry_representation_20261115/condition_summary.csv; predictfun/research/pf_kp_s4.py:1682`.

### F2: Available public signals are incomplete proxies

Status: **confirmed_information_limit**; priority: **medium**.

Top-five depth and pending/shrink increments do not identify exact public queue, cancellations versus trades, a causal venue execution stream, or deep Binance L2.

The feature catalog cannot guarantee sufficient information for fill/adversity timing, however well normalized.

Repair: Reconstruct additional causal flow only from receipt-timed records; quantify incremental contribution and observation limits instead of assuming a larger catalog fixes prediction.

Evidence: `predictfun/handoffs/20261115_entry_information_inventory.md`.

### M1: Learner complexity and regularization may be unsuitable for support

Status: **risk_not_isolated_cause**; priority: **medium**.

CONDITION has 351 named fields including aliases, with weight-only effective training counts about 523–1,587 per side/fold. The pinned learner is 200 boosted trees, 31-leaf maximum, no explicit L2.

Overfitting or poor conditional estimation is plausible; feature count is not independent dimensionality. Earlier compact/ridge candidates did not solve the task.

Repair: After label/support fixes, test a small predeclared regularization/complexity comparison on development data; flag changes from the pinned fairness baseline.

Evidence: `predictfun/reviews/entry_training_support_20261115/result.json; predictfun/reviews/entry_representation_20261115/fit_receipts/20261027.json`.

### M2: Stable timing and economic selection are not demonstrated

Status: **confirmed_performance_failure_cause_open**; priority: **high**.

CONDITION reduces bid timing error 1.43% but increases ask error 0.95%; both trail prior-mean/price-rung baselines. None of six economic checks passes. Some historical feature relationships change sign.

A lower distribution distance or better absolute-POST score is insufficient evidence of useful WAIT/POST timing; the cause of side/date differences remains open.

Repair: Retain separate bid/ask evaluation, decision calibration, per-condition support and matched economic controls. Do not select a common representation solely on pooled error.

Evidence: `predictfun/reviews/entry_representation_20261115/result.json; predictfun/reviews/entry_information_20261115/feature_summary.csv`.

### E1: Training-history effect and generalization have not been isolated

Status: **confirmed_validation_gap**; priority: **high**.

Growing folds test different days; no fixed-evaluation learning curve exists. These dates have been used repeatedly. Actions within market/day share information/outcomes; weight ESS does not remove dependence.

We cannot attribute failure to short history or call a selected historical improvement forward validation.

Repair: Compare nested histories on identical evaluation actions, separately vary later-state sampling, use day-level uncertainty where justified, then freeze for later untouched-day validation under the program rules.

Evidence: `predictfun/reviews/entry_training_support_20261115/result.json; CLAUDE.md:87`.

### E2: Combined strategy and net value are not established

Status: **confirmed_validation_gap**; priority: **medium**.

This representation study performs local prediction/selection checks only, with gross terminal labels. No combined selective-keeper portfolio, fees/rebates or measured runtime acceptance is shown.

Local scores cannot establish quote residence, inventory risk, full-policy profitability or operational speed.

Repair: After credible local signal, evaluate the actual combined strategy with its intended costs, clocks, order sizes and queue/age breakdowns; keep research and live execution separate.

Evidence: `predictfun/research/pf_entry_value.py:1; predictfun/reviews/entry_representation_20261115/result.json`.

## Concrete coverage repair

The proposed density-only comparison keeps the same dates, target, keeper and
evaluation actions. Change later-generation inclusion from 1/32 to **1/4**;
keep initial inclusion at 1/2. This is eight times the inclusion probability,
not a claim of eight times as much independent information or eligible labels.

Use the same generation hash so old selected actions remain a subset. Crucially,
retain each original requested probe age. The current code derives both selection
and age using the modulus; naively changing that modulus also changes the action
time and confounds the comparison. The new dataset must keep original ages.

Record sampling version and inclusion probability per action. New training weights
would be 2/4; the unchanged evaluation sample keeps its actual 2/32 weights. The
current generation-only hardcoded weight function cannot be reused unchanged.
Reject incompatible sampling metadata instead of silently mixing versions.

Write a new artifact namespace, preserve unique neutral action identities and
all exclusion statuses, and reconcile every new target. Keep no-fill actions
and rare adverse outcomes. Do not duplicate existing actions or select rows on
their realized outcome. The machine-readable specification lists the required
nested-sample, exact-age, invalid-probability and action/weight reconciliation
checks. Nothing in this review has run or qualified that new sampler.

## Repair order

1. Repair the entry pre-read boundary rule, audit upstream clock/source lineage,
   and quantify the unknown-depth queue fallback. If this changes labels or rows,
   issue superseding receipts; preserve all old results.
2. State whether the next experiment is standalone entry under neutral KEEP or
   entry under the frozen intended keeper. The latter requires relabeling both
   branches consistently; keep WAIT/POST and cancel/keep as separate models.
3. Build denser later-entry labels on admissible data and add representative joint
   waiting states with declared sampling. Longer-date and density comparisons
   should be separate so their effects can be identified.
4. Run a fixed-evaluation learning curve, then a small declared complexity or
   regularization comparison if adequate labels still fail. Keep bid/ask separate.
5. Only after credible local skill, test the combined portfolio and later untouched
   validation days under the program rules. No runtime or live-trading code here.

## What is already addressed

- Audited units, side signs and denominators passed.
- Start-price mask repaired in features and continuation; original labels rebuilt.
- Client-known ladder state and portfolio accounting checks implemented.
- Inverse inclusion weighting applied to fitting and scoring.
- New representation retains absolute liquidity/risk and missing/support statuses.
- Old CLEAN preprocessing removal/scaling defects do not enter RAW/RELATIVE/CONDITION.

These are scoped checks/fixes, not a universal data-cleanliness certificate. In
particular, a missing-queue feature does not repair a zero-queue fill assumption,
and perturbing downstream inputs does not reconstruct upstream source clocks.

## Evidence and read-boundary accounting

`reviews/entry_training_issues_20261115/current.json` resolves the source-bound
15-issue register and `coverage_repair_spec.json`. Source hashes identify the
exact entry implementations and prior results reviewed. No new performance
numbers were estimated. The first-fold noise evidence remains limited to 4,040
training and 1,992 evaluation actions, as of 2026-11-15T02:35:02Z.

The directory inventory exposed forward-family date names before the stricter
P9 rule was reread. No forward file contents or outcomes were opened in this
review. That metadata exposure is recorded; this audit does not certify those
dates as untouched. The existing entry loader read-boundary issue is recorded
separately as D1, and must not be confused with proven future-feature leakage.
