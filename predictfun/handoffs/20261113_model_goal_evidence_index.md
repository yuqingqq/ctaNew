# Better market-making model: evidence and completion requirements

Snapshot **2026-11-13T23:34:44.662483+00:00**. The goal remains active. There is no
verified improved model yet. This index links the evidence for the original
goal; it does not replace or revise any frozen experiment or success gate.

The goal is to build and evaluate a better Predict.fun BTC five-minute
market-making predictor for default quoting and selective withdrawal, using
causal action targets that account for avoidable adverse fills, retained
queue value and consistent re-entry. Controlled model comparisons must lead
to full sequential evaluation under execution stress and matched actual
inventory risk. Negative evidence and live-calibration limits remain visible.
No live trading changes are authorized by this research goal.

## Current experiments

| Experiment | Actual evidence | What remains |
|---|---|---|
| First value model: STATIC2 / STATIC10 / HISTORY10 / JOINT10 diagnostic | Complete local comparisons; primary HISTORY10 evaluated in 18 full scenario-days. Final integrity PASS at 2026-11-13 02:37:37 UTC; all three improvement predicates false. | This candidate failed the declared goal. Preserve its negative result; do not promote a different arm after seeing outcomes. |
| Avoidable-value plus recovery | Causal labels, six chronological value fits, local evaluation, native controls and 18/18 full scenario-days complete. The full report at 18:01:17 UTC fails value/risk, interval and participation gates in all three scenarios: SELECTIVE10 loses to guard/DIRECT10, increases inventory-square exposure and has less presence than old HISTORY10 despite fewer pulls. | Final integrity PASS at 18:18:37 UTC; all improvement gates false. This candidate failed the declared objective. |
| Five versus 50 shares | Full scientific pipeline frozen at `8a8f4c18`; exact computation adapter separately frozen at `96730882`. Controls exercise native partial fills, shared reservations and unchanged optimized outputs. | Execution started at 18:18:56 UTC. Four initial smoke stages and full references/captures for both profiles on Oct23, Oct24, Oct25 and Oct26 pass (20/169 stages). The latest Oct26/q50 capture at 23:34:32 UTC covers 288 windows and 2,225 actions: 2,212 eligible and 13 missing continuation marks; all short targets are eligible. Its neutral path exactly matches the independent reference, including two missing marked windows. The [stage ledger](20261113_size_experiment_progress.md) retains every earlier population, status and receipt. Action populations differ by profile. The Oct27/q5 reference is active. Remaining datasets, all fits and sequential comparisons are pending; no size/model advantage established. |
| Post-effect adverse-depth forecast | Full pipeline frozen at `0e246af6`, with 76 synthetic controls as of 14:41:07 UTC. Two feature families, five depth indicators each, pooled five/50-share profiles. | All 26 historical stages are queued. A forecast gain would still require a priced policy and full sequential value/risk evaluation. |

The native overlap diagnostic at 16:00:57 UTC reconstructs all 13,824 policy-windows
from the twelve completed X/stress runs. For SELECTIVE10, both recovery gates
are closed during 76.66% / 69.69% of entry no-quote time (1,728 windows per
scenario). This identifies concurrent restrictions, not profitable earlier
entry. See the linked overlap review for denominators, controls and limits.

The unchanged 15:39:36 UTC partial receipt also shows DEFAULT entry presence of 89.01% / 88.60% in X/stress (1,728 windows per scenario), with paired SELECTIVE10 marked improvements of +$2,153.30 / +$1,994.99 (1,559 / 1,554 common windows; 169 / 174 missing either mark). SELECTIVE10 inventory-square exposure changes by -0.36% / +9.66%. Thus much lower presence does not necessarily reduce inventory exposure. The linked primary/stress review now includes these original DEFAULT comparisons and their day intervals; no frozen result or criterion changed.

The first experiment's populations are six consumed outer dates × 288 windows
per policy/scenario, with five sequential policies and three scenarios. The
recovery local result has 9,977 sampled actions, 9,945 eligible targets and
32 missing continuation marks. The current recovery full runs each have four
policies × 288 windows. These populations must not be pooled across experiments
or interpreted as independent repeats of the same target.

Authoritative receipts and interpretation:

- First candidate: [final evidence](../reviews/model_upgrade_20261112_v2/final_evidence_check.json),
  [complete negative review](20261113_model_upgrade_complete_review.md).
- Recovery: [local result](../reviews/model_recovery_20261113/local_result.json),
  [local interpretation](20261113_recovery_local_review.md),
  [final integrity evidence](../reviews/model_recovery_20261113/evidence_check.json),
  [full portfolio report](../reviews/model_recovery_20261113/portfolio_result.json),
  [complete negative review](20261113_recovery_complete_review.md),
  [completed-scenario partial report](../reviews/model_recovery_20261113/partial_primary_stress_v1.json),
  [primary/stress interpretation](20261113_recovery_primary_stress_review.md),
  [native absence/gate overlap](20261113_recovery_absence_overlap.md),
  [latest completed run](../reviews/model_recovery_20261113/run_age500_20261103.json),
  [artifact review](../reviews/model_recovery_20261113/run_age500_20261103_artifact_review.json),
  [ongoing progress](20261112_model_recovery_progress.md).
- Size: [scientific declaration](../reviews/model_size_priority_20261113/declaration.json),
  [computation declaration](../reviews/model_size_priority_20261113/fast_execution_declaration.json),
  [design](20261113_size_experiment_frozen.md), [driver handoff](20261113_size_fast_handoff.md),
  [execution progress](20261113_size_experiment_progress.md).
- Depth: [declaration](../reviews/model_action_depth_forecast_20261113/declaration.json),
  [controls](../reviews/model_action_depth_20261113/experiment_controls_v2.json),
  [design](20261113_action_depth_forecast_frozen.md).

The [corrected maker-study review](20261113_robust_v2_model_integration_review.md)
at 16:09:38 UTC exactly reconstructs the latest K1–K7 report on nine consumed
dates. Top-maker residual-refill observations support the size hypothesis, but
observed pre-fill quantity does not identify original or remaining size;
future-volume-conditioned non-refills do not identify cancellation. The native
size comparison remains necessary and no current primary changes.

The [new maker next-trade audit](20261113_learn_pull_model_integration_review.md) at 17:39:41 UTC reconstructs the LEARN-PULL descriptive report on 193,153 observations (producer as of 16:48:15 UTC, nine consumed dates). Exact counterexamples show that full fill and cancellation can produce the same GONE label, and retained/replaced orders from one signer can produce the same STAYED label. Features anchored before the next trade can follow the hidden cancellation; the classifier uses future-dependent gap and nonchronological folds. Its reported AUC is reused, not independently refitted. These observations are unsuitable as causal cancellation labels; the native residual-size experiment remains necessary.

The [MOVE-CASES audit](20261113_move_cases_model_integration_review.md) at 17:46:07 UTC reconstructs its original report and post-hoc side table (producer as of 16:12:36 UTC; nine consumed dates; 100,719 events and 1,233,693 real segment records). Exact source controls show that overlapping event windows can count the same fill repeatedly; saved rows omit identities needed to deduplicate or attribute both sides to one maker. The endpoint selector also accepts missing future coverage in a synthetic case. Historical affected counts are unknown. Reported offsets are event-weighted fill marks, not verified portfolio P&L or an alternative model-success target.

## Requirement-by-requirement assessment

| Requirement | Evidence available | Completion assessment |
|---|---|---|
| Neutral, causal decision populations | Recovery's saved population/capture evidence; size generation assignment and all-status ledger; depth reuses those immutable captures. | Recovery construction checked on history. Both size profiles on Oct23, Oct24, Oct25 and Oct26 pass neutral parity, target, feature and provenance checks; remaining size/depth populations are pending. Fill-selected maker studies cannot substitute. |
| Avoidable danger after effective cancellation | Recovery's native short-value labels; size native timing and original-order target; depth excludes recovered pre-effect movement and counts persistent/post-effect movement. | Implementation controls exist. Reliable economic forecasting is not established. A cancel request is not its effective time. |
| Price retained priority and residual exposure | Native KEEP/PULL and size KEEP/RESET alternatives, unchanged partial order identifiers, taker identity and joint reservations. | Historical five/50 advantage pending. KEEP-minus-RESET is not an isolated causal queue premium. |
| Consistent withdrawal and re-entry | Recovery shares its recovery gate between label and replay; size shares pressure-clear conditions between PULL labels and policy. | Consistency is implemented and controlled. The recovery proxy is not identified fresh-entry value, and useful full-policy behavior remains unproven. |
| Controlled model comparison | Pinned parameters, earlier core / preceding calibration / later test dates; static/history variants with common actions. | First comparison negative; recovery comparisons negative in all three scenarios. Size/depth results pending. Guard-vs-model policy differences also include decision timing. |
| Default quoting with selective pulls | Native policies, recorded repeated decisions and recovery traces. | Executable research implementation exists. Better retained presence together with value/risk has not passed. |
| Full sequential execution stress | First and recovery experiments complete with negative results; size running its declared sequence. | Required size/model comparisons incomplete. The age500 path is a sensitivity assumption, not calibrated live execution. |
| Matched actual inventory risk | Frozen portfolio gates use all-window inventory-square exposure and common marked-value populations. | Configured equal caps alone are insufficient. No new candidate has passed the combined value/risk gates. |
| Preserve negative results and provenance | Original declarations, failed attempts, in-band accounting corrections, exact parity receipts and review commits remain. | Ongoing evidence requirement; current failures are not hidden by integrity PASS. |
| State live-calibration limits | Partial live matching evidence, source-clock checks and explicit replay assumptions. | Fresh execution calibration unresolved. No live profitability or top-maker parity claim. |
| No live changes | Goal work is in the isolated research worktree, with research artifacts and documentation mirrored to the shared tree. | Maintained within this work. A research result does not authorize deployment. |

## Venue documentation and remaining queue uncertainty

Official pages checked on **2026-11-13 at 19:18 UTC** confirm that an unfilled
limit-order remainder stays on the book ([Place orders](https://dev.predict.fun/trading/place-orders))
and that moving a quote requires cancelling and replacing it; orders cannot be
edited ([Market making](https://dev.predict.fun/trading/market-making)). Those
mechanics support comparing retained residuals with fresh replacement orders,
but do not establish a particular economic queue premium.

The reviewed order-book, placement, market-making and FAQ pages did not supply
an explicit FIFO or partial-fill priority guarantee. This is a limit of the
reviewed evidence, not a claim that no venue rule exists. The public book stream
can combine successive changes and omit intermediate states
([Prices and order book](https://dev.predict.fun/concepts/order-book)), so its
snapshots alone cannot establish complete order-level priority or hidden
cancellation timing. Empirical execution calibration remains necessary. This
documentation check changes no frozen model, target, policy or replay assumption.

## What must be true before an improvement claim

The current size primary must beat both its chronological guard and its
HISTORY comparator on common strict marked value, with nonincreasing
all-window inventory-square exposure and a positive lower day interval in
every declared scenario. Entry quote presence must not fall and learned pull
decisions must fall versus HISTORY. Five/50 comparisons use identical DEFAULT
and SELECTIVE policies and separately enforce the realized-risk requirement.
Missing marks and unknown takers remain in the population accounting.

The depth experiment's Brier screen is a forecast result only. Passing it does
not price remaining quantity, queue retention or replacement opportunities,
and includes no new cancellation policy. Any subsequent policy must be
specified and frozen before its evaluation, with consistent causal labels and
sequential replay. No post-hoc switch of the current primary is allowed.

All current dates—Oct 23–27/31 and Nov 1–3—are consumed. Any later claim of
forward validation needs at least five later untouched complete UTC dates and
a full prior freeze. A historical research improvement, if found, must still
carry its unresolved live-fill assumptions.

Larger partial orders and multiple layers are separate policy choices. This
size comparison is one layer under a joint inventory cap. The depth target
supports questions about multiple distances, but no complete joint-budget
multi-layer policy has been demonstrated. The maker studies give descriptive
evidence of some useful resting residuals; they do not identify the makers'
hidden policy or provide action-level training labels.

## Next executable dependency

Recovery has completed with final integrity PASS and improvement false. Size
driver **1576587** owns core8; four initial smokes and full DEFAULT
references/captures for both sizes on Oct23, Oct24, Oct25 and Oct26 passed.
The Oct27/q5 reference is active (worker **3519770**).
Depth driver **2900451** waits for the whole size
chain and final evidence. Do not restart live workers, steal
gaps between stages or alter scientific sources while they run.

All eighteen recovery receipts have passed the committed stage checker.
The full recovery report and final verifier are preserved with false improvement
predicates in all scenarios. Continue reviewing newly completed size captures and
fits, then the complete size report and depth forecast evidence. Revisit the full
goal only against those actual outputs; declarations and passing synthetic
tests alone do not prove completion.
