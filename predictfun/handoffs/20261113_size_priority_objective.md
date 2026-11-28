# Larger resting orders as a way to retain queue priority

User clarification, 2026-11-13: a larger resting order can survive several
partial fills and retain earned queue priority. A small exhausted order must
be replaced, which starts a new queue position. This is a mechanism the model
research must evaluate directly; the negative size study does not disprove it.

## Mechanism and existing support

A 50-share order that fills five shares has 45 shares left on the same order.
If those shares remain at the front, later small takers can fill them without
a cancel/repost cycle. Size does not itself give an earlier initial queue
position; it changes how long that position can survive incoming fills. A
five-share order can also survive a smaller partial fill; exhaustion, not any
fill, triggers the reset.

Predict's [order lifecycle](https://dev.predict.fun/trading/order-lifecycle)
and [order-book guide](https://docs.predict.fun/the-basics/understanding-the-order-book)
confirm that unfilled limit quantity remains open for further matches.
Its [market-making guide](https://dev.predict.fun/trading/market-making)
states that orders cannot be edited; changing a quote requires replacement.
Sources checked 2026-11-13. These pages do not explicitly specify a complete
FIFO priority contract. Retained time priority is the replay's queue mechanism;
its actual calibration remains a live-data question.

Our current default hook already retains a partially filled accepted order:
`pf_default_pull_hooks.Hook.targets` returns its existing price and initial
size while remaining quantity is positive, unless a price, data, inventory or
learned-pull condition withdraws it. New orders use `min(5, capacity)`, and the
explicit capacity is ten shares including reservations. The model includes
`remaining`, `queue_initial`, `queue_now`, `queue_min`, `queue_progress` and age,
but labels come from this five-share path. It cannot establish the value of a
50-share residual outside its training support.

The existing [size study review](20261113_front_pull_size_integration_review.md)
preserves its negative results. Similar negative value per share at larger
sizes can coexist with a real retention benefit: the benefit may be outweighed
by additional adverse fills. A count of fill records also cannot substitute
for repeated distinct taker-order fills or value earned after the first fill.

## Controlled comparison to prepare after the frozen replay

The current eighteen-run experiment remains unchanged. The next size study
must distinguish two questions rather than combine them in one P&L number:

1. **Value of keeping an earned position.** On a neutral 50-share reference
   path, define an action when a live order first has a client-known cumulative
   fill of at least five shares and still has positive remaining quantity.
   Fork KEEP the residual versus CANCEL and REPOST that residual at the same
   valid price, using actual notice/cancel/placement clocks and reservations.
   Subsequent fills while cancellation is pending must reduce the quantity
   available for replacement. Both branches share the same initial inventory,
   remaining quantity, external tape, danger rule and risk controls. This
   measures retaining priority versus losing it, including the real interruption.
   The population is defined at the observable decision, with every neutral
   generation represented and no future-fill selection. Unfilled, exhausted,
   censored and ineligible states remain explicit statuses.

2. **Practical small-versus-large policy value.** Compare five-share replenished
   orders and 50-share persistent orders under the same declared total exposure
   budget, fixed danger trigger, re-entry protocol and shared full engine.
   Fifty-share orders cannot fit the current ten-share reservation limit.
   A new common budget permitting 50 shares must be explicit for both arms;
   the original five-share/cap-ten strategy remains a separate context baseline.
   Equal caps do not imply equal realized risk. Report actual inventory-square
   exposure, adverse tails and value together, and require a risk-matched gain
   before treating a larger policy as an improvement. Do not achieve nominal
   matching by posting 50 while silently reserving only ten, or by dropping
   fills after the fact.

For both comparisons, record exhaustion-driven replacements separately from
danger-driven cancellations. Measure retained order identity/age, subsequent
**distinct taker events**, acquired and retained queue progress, fill value
before and after the initial partial fill, unavailable mark counts, replacement
latency, gross fills, and full inventory exposure. Latent replay queue position
is evaluation-only; the predictor may use only client-observable state.

The model target must depend on remaining quantity and earned priority. If
larger-order states are used for training, build their own neutral action labels
and chronological fits. Applying the five-share model to 50-share states and
calling it a test of queue value would be unsupported extrapolation. Keep the
same matching assumptions across arms and test execution sensitivity. Any
future numerical design needs its complete committed pipeline, control/null
specification and candidate count before execution; consumed dates remain
exploratory.

## Mechanism verified, with a control that removes the advantage

The committed checker ran at **2026-11-13T07:46:35.686593+00:00** on four
synthetic cases: sizes five/50, each under M and U, with two distinct taker
events and the same external book. Under M, both initial orders fill five
shares at 100 ms. The small order exhausts and its replacement activates at
240 ms behind 20 newly arrived shares. The large residual retains its initial
priority and takes another five-share fill at 300 ms. It uses one order
through both fills. The small replacement misses that second fill.

Under U, both sizes fill ten shares: removal of the modeled queue-ahead
penalty removes this fixture advantage. A checker asserting that U still
provides the larger-order gain refuses as required. This establishes the
mechanism in the reviewed simulator, not a historical profit or live estimate.

The exact size-report repeat-fill function was also tested with three fill
legs from one synthetic taker event versus three distinct events. It counts
three in both cases; the distinct-identity counts are one and three. A planted
claim of equivalence is rejected. This supports measuring distinct taker events
before comparing queue retention with the public-maker report; the historical
size of this measurement difference remains unknown.

[Corrected mechanism receipt](../reviews/model_recovery_20261113/queue_retention_mechanism_check_v2.json)
supersedes the original receipt's queue-field label only: S4 records displayed
depth as `a0`, but U models zero ahead. Version two names displayed and modeled
ahead separately; all order/fill paths and test outcomes are unchanged. Positive
schema and wrong-U/legacy refusals pass. The original receipt remains preserved.

Source checker: `research/pf_queue_retention_mechanism_check.py`, committed at
56855278 (executed from b57507bf); correction `pf_queue_retention_receipt_v2.py`
at d8debbdf. Wrapper rc=0, 125 MiB peak RSS, 0.44 seconds on the free auxiliary
slot. An earlier wrapper admission was refused because the other owner's core9
worker was active; no numerical work started then and the log is preserved.
No historical payload was read by this fixture and no live action was taken.


## Implementation checkpoint, 2026-11-13 07:58 UTC

`pf_size_priority_policy.py` (fcee3270) now supplies an explicit immutable order-
quantity/inventory-budget profile and an isolated subclass of the inherited
causal default hook. Exact source anchors parameterize new quantity, available
capacity and the hypothetical-entry feature state. Existing accepted order
size is retained after a partial fill; client-known reservations remain in the
capacity calculation. The original source/class is not monkey-patched.

It refuses a requested order larger than its declared inventory budget, direct
use of the inherited five-share learned modes, and the legacy age-probe builder.
A helper exposes only notified fill quantity and known residual; latent physical
remaining quantity or queue position is excluded. This is a preparation
component, not the complete size experiment or a historical runner.

Eight synthetic controls are committed in `tests/test_size_priority_policy.py`:
original-profile equivalence, retained large residual identity, late-fill/cancel
reservations, latent-state mutation invariance, common-budget constraints,
invalid profiles/old-model refusal, partial/exhausted statuses and strict source
anchor checks. **They have not run yet.** Pure launcher **3412927**, source
93ee8ef6, is live and waits for the other owner's `run_deep_pull.sh` chain to
finish and core9 to be free. It will run only this bounded unit suite through
the shared wrapper, without changing the primary replay. Its initial system-
Python import failure is preserved and superseded by the project-venv launch;
no test worker started during that failed import.

Next: inspect the actual control result, then connect the partial-fill action
ledger, native-engine residual reset, quantity-aware targets and full replay
accounting. The complete new pipeline and its experimental design still need
to be committed/frozen before historical execution. Current recovery worker
3178539 remains active on core8 under driver3159222; no current model/target,
policy threshold, execution criterion or live behavior changes.


## Native integration preparation, 2026-11-13 08:20 UTC

Source **18d25ce1** connects the first known partial-fill ledger to the native
execution path. The ledger was introduced at 8c6f7711/68a75622; this version
adds the native fill-notification callback before the engine clears an exhausted
known slot. A notification updates known quantity but cannot itself select an
action. The first eligible actual decision supplies the action, observed age
and causal features. Unfilled, refused, exhausted, pending, unavailable-state,
withdrawn and censored generations remain in the population. Quantity comparisons
use an explicit 1e-9 share roundoff tolerance.

An integration gap in the first quantity adapter was found before historical
execution: its exact target-body anchors do not exist on the installed
`ReservedHook` wrapper. It therefore refuses that installed stack. The new
`pf_size_priority_policy_v2.reserved_hook_class` composes the sized implementation
**below** the unchanged reservation methods. Cancellation continues reserving
client-known quantity until effective cancel plus the last possible fill-notice
delay; pending-to-accepted identity and notification updates are preserved.
No old wrapper or frozen source is bypassed/edited.

The observer is separate from policy: it calls the same targets first, then
records the client-visible state. New controls compare the observed and
unobserved native runs at both sizes, including orders, fills, generation
records and independent inventory accounting. A synthetic larger residual
should survive two taker events while the exhausted small order's replacement
queues behind newly arrived depth. These new native results are **not yet run**.

The original eight-test unit suite remains queued under **3412927**. The next
**23 tests** (ten installed-stack reservation cases, nine ledger cases and four
native-engine cases) are queued under **3597883**, after that suite passes and
the other owner's complete `run_deep_pull.sh` chain releases core9. Their
source hashes are bound in
[the queued native-controls receipt](../reviews/model_recovery_20261113/size_priority_native_controls_runner_started.json).
Syntax compilation passes; runtime correctness is pending. No duplicate worker
is launched and the primary replay keeps core8.

The residual CANCEL/REPOST fork, paired value targets, chronological new fits
and larger-order historical evaluation are still outstanding. This is
preparation, not a complete size experiment or evidence of a better model.


## 2026-11-13 08:48 UTC: observation verified; residual forks queued

The original8 quantity cases pass. The corrected installed-stack/ledger/native
observation and pure-reset groups pass30 cases; the combined36-case attempt
still failed because all six new fork cases refused their source template.
The template and nested notification indentation are now corrected; M/L source
compilation passes. Thirty-eight runtime cases are queued under3820900 behind
the active keep-place chain. The original failed sources/logs and the cancelled
pre-execution attempt are preserved. No passing full-fork result is claimed.

The strict continuation-value component is committed through61fb8eea, including
missing marks, retained abort statuses, complete inventory-square time and
peak risk in native fill order. Its seven analytic tests remain unrun. Read
[the native construction review](20261113_size_priority_native_controls.md)
for exact receipts, source versions, queue state and remaining pipeline work.


## 2026-11-13 09:07 UTC: native forks and saved labels pass

The 38-case native protocol suite passed at 09:06:38 UTC, followed by all
22 target/lifecycle/dataset/artifact controls at 09:07:29 UTC. The latter
receipt binds source commit95a4f8a3; its saved log and all committed source
hashes were checked, along with the three identical receipt copies. See
[the native result](../reviews/model_recovery_20261113/size_priority_native_controls_v4.json)
and [the dataset result](../reviews/model_recovery_20261113/size_priority_dataset_controls.json).
These are synthetic checks, not historical performance.

The new lifecycle ledger retains native fill notices after the decision cutoff
without inventing later actions. Saved artifacts include every neutral
generation and every window, including empty windows. Every paired label is
reconstructed from saved native fills and captures; corrupt quantities,
identities, marks, targets and coverage are refused. Missing marks and reset
aborts remain statuses. Previous failed attempts and their sources remain.

Next is a shared causal feature layer for larger orders, followed by a complete
committed historical builder, fitting and replay design. No size history has
been built and no larger-order model is fitted. The first-known-partial action
cannot protect an initial sweep that exhausts an order before such an action;
a broader timing policy still needs pre-fill decisions. The existing five-share
recovery experiment continues unchanged (two of eighteen full runs complete,
Oct31/X active at this checkpoint). No live changes.


## 2026-11-13 09:29 UTC: shared features and native manager verified

Three further bounded suites pass, with their log and committed source hashes
verified and three identical receipt copies checked:

- [Size features](../reviews/model_recovery_20261113/size_priority_feature_controls.json):
  ten tests, as of09:26:39.905673 UTC, sourcebe875b01.
- [Native managers](../reviews/model_recovery_20261113/size_priority_manager_controls.json):
  six tests, as of09:27:18.129400 UTC, source118b195f.
- [Saved feature artifacts](../reviews/model_recovery_20261113/size_priority_feature_artifact_controls.json):
  five tests, as of09:29:00.835805 UTC, sourced0bba02a.

`pf_size_priority_features.SizeHistory` uses the same corrected causal history
for batch training inputs and single replay states, reconstructs the inherited
float32 deltas, and adds initial quantity, notified filled quantity/fraction,
remaining fraction, declared inventory budget and signed inventory fraction.
Future market changes, latent fill/queue state and target/guard fields cannot
affect those features. Untouched orders are supported for later pre-fill work.

`pf_size_priority_manager` composes the installed reservation layer, sized
neutral policy, lifecycle ledger, native residual forks and artifact writer.
Its reference path has no observation/fork hooks. The complete synthetic
reference/capture comparison has identical requests, fills and independent
window accounting, and reconstructs the expected residual value after saving.
This is synthetic correctness, not a historical strategy result.

`pf_size_priority_feature_artifacts` attaches features to every captured action,
including unavailable targets, and supports an empty action population without
dropping its window. It reconstructs every saved feature plus all unchanged
metadata and targets; altered values are refused even after a file is rehashed.
The caller must independently verify the corrected-grid origin: the utility
checks the supplied identity and reconstructability, not the origin of an
in-memory grid. The construction receipt explicitly remains BUILT_UNVERIFIED.
The test log contains pandas fragmentation warnings from column attachment;
all checks pass. A batched concatenation could remove that overhead before a
future full historical build, with output equivalence checked if changed.

No controls remain queued or running at this checkpoint. The first direct
feature-suite admission was refused while core9 was occupied; pure launcher
4153363 subsequently waited for the entire timing-split chain4133953 to end
and finished normally. Its v2 scheduler respects any active research shell
chain, not just the previous named chains.

The main frozen recovery replay is three of eighteen full scenario-days
complete, with Nov1/X worker4143840 active under driver3159222. Oct31/X's
25 named artifacts, three receipt copies and correction links are verified in
`third_full_run_review.json`; its runner reconstructed300 inference samples
over four arms x288 windows. No multi-day policy verdict yet.

Next required work: complete and commit the historical runner, explicit input
and neutral-reference checks, chronological new fits, and a same-budget full
policy comparison. Preserve the distinction between the residual KEEP/RESET
queue-value experiment and a complete timing model: a partial-only action set
cannot protect the initial exhausting sweep. Broader pre-fill decisions and
a policy-consistent withdrawal/re-entry target remain necessary. No size
historical build, new fit, untouched-day validation or live change is claimed.


## 2026-11-13 10:02 UTC: pre-fill timing and three-way labels verified

The [size timing checkpoint](20261113_size_timing_pipeline.md) now documents
birth-assigned pre-fill/partial actions, native KEEP/RESET/PULL branches, shared
pressure-clear re-entry, strict order/quantity capture identities and linked
saved withdrawal labels. All declared bounded suites pass through09:59:29 UTC.
The original partial-only target remains a mechanism comparison; the new age
actions cover initial sweeps before a partial decision exists. No historical
size fitting or performance gain is claimed. Next: preserve original taker
identity/timing provenance, then complete and freeze the historical fitting
and full-policy pipeline. Recovery4/18 runs complete; Nov2/X remains active.


## 2026-11-13 10:21 UTC: taker provenance and fitting components verified

The [current size timing checkpoint](20261113_size_timing_pipeline.md) now records
original taker/clock metadata through all native branches, saved per-window
catalog verification, and an injected chronological fitting component with
profile-qualified identities. Seven provenance controls and four synthetic
fit controls pass at10:14:38/10:19:04 UTC. Native requests/fills are unchanged;
serialized forecasts reconstruct and corrupted identities/outputs refuse.
Historical size labels/fits, model policy integration and a complete committed
experiment declaration remain. These synthetic results do not establish a
larger-order performance benefit. Recovery4/18 full runs complete; Nov2/X remains
active at this checkpoint. No live changes.


## 2026-11-13 10:45 UTC: size-aware native model policy verified

The [new policy checkpoint](20261113_size_model_policy.md) documents strict
quantity/budget model binding, static/history/selective decisions, native partial
retention, shared withdrawal/re-entry, the earlier-only guard benchmark and exact
inference-sample coverage. Final controls pass through10:44:22 UTC; failed
preparation attempts remain preserved. Full-policy taker provenance and the
complete historical experiment runner/declaration remain. Recovery5/18 full runs
complete; Nov3/X is active. No historical larger-size fit or performance claim.
