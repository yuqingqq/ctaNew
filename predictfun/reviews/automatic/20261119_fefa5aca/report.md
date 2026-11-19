# R1 v2 review: execution repaired; saved-fork contract still open — 2026-11-19

Target: **fefa5aca63445e7b9d328b50b6f5f35d7702a2e0**.
Range: **1d72986ffa9cee8285bcef838fe2d054b16b2486..fefa5aca63445e7b9d328b50b6f5f35d7702a2e0**.
Development commits reviewed: **a9303c1f8b21054f3c90f0f22888a349c69deb02**
and **fefa5aca63445e7b9d328b50b6f5f35d7702a2e0**. Review-only 043ca403 is
context, not a new development change. Pinned source: `/tmp/pf-review-fefa5aca`.

**The prior P1 is fixed. The saved slice supports its reported execution and
accounting results. One P2 remains in the newly introduced saved-state entry
point: it can accept a different scenario, mark or focal side without validating
that identity. Close that small contract gap before R2 generates labels. There
is no evidence that it affected the saved slice, which constructs its own fork
state rather than supplying a cached one.**

Evidence: [script](verify.py), [independent synthetic cases](synthetic.json),
[saved-artifact audit](artifacts.json), [pinned selftest](selftest.log).
No implementation change, fit, label build or raw-tape replay was made here.

## Resolved and verified

| Previous requirement | Pinned source | Review conclusion |
|---|---|---|
| Hidden full fill must not suppress a client structural request | `pf_branch_engine.py:174–175,269–285` | Fixed. Client mode returns before the engine-truth guard; `Branch.structural` uses the received frame and client-open generations, without the hidden remainder. Independent zero/partial/full cases agree before the notice on both sides, including a pending-sensitive opposite side. |
| Named clocks and tie order | `:79–86,225–228,339–342` | Adopted: post notice 0 after activation, fill notice 100, cancel ACK 100; cancellation ACK cannot overtake fill notices in this contract. Trades, due notices, constraints, structural commands and alternating policy actions have a declared order. These remain simulation assumptions. |
| Same-fork continuation test | `:383–420,802–815` | F5 now creates one saved state and gives copies to both continuations. Both sign-direction controls pass. The identity-check weakness below does not invalidate the fact that this test passes the same actual object. |
| Numerical aggregate C parity, including END value | `:87,681–706` | Adopted, with separate share/cash/position/value tolerances. No longer represented as bit-for-bit/event parity. |
| Cap and pending state | `:253–267,299–323,339–358` | The current tick-level client path checks worst-case reservations and engine position; policy commands recheck them. Counts distinguish absent cap blocks, pending states and feasible actions. Notices are processed in a batch at the decision tick; this is not a separate asynchronous notice-event simulator. Structural requests do not release reservations. |
| Expanded bounded slice | `:538–650` | The frozen 128-per-side Oct24 selection, both policies' accounting, C totals and adaptive branch checks are present. The saved artifacts reconcile as detailed below. |

The original R1 receipt's per-public-trade versus per-time aggregation mismatch
is correctly treated as a representation issue. The archive records quantities
aggregated at each timestamp; the new engine preserves individual public-trade
fills. The revised comparison retains both time and quantity checks, and the
independent value check remains beside it. Earlier failed receipts are retained.

## Open finding, ranked by severity

### P2 — supplying `state=` bypasses the fork's scenario and identity contract

`pair` at **`pf_branch_engine.py:398–399`** deep-copies a supplied state and only
checks tick and episode. It bypasses `Branch.__init__`'s scenario/END guards.
The returned metadata at **`:420`** is built from the *requested* settings,
independently of the state actually executed. `fingerprint` at **`:345–349`**
omits the END mark, endpoint, scenario, tape identity and focal-side decision
phase. `fork_state` at **`:383–390`** may already have executed the opposite
side at this tick, but does not bind that fact to a target-side identity.

Independent synthetic counterexamples at the pinned source:

1. **Scenario mismatch:** a normal call with `cancel_effect_ms=50` is refused.
   With an otherwise valid `state=`, the same call returns `status=ok` and labels
   the scenario 50 ms, while its PULL at 200 ms still takes effect at **237 ms**.
2. **Valuation identity:** changing only the saved state's mark from 51 to 61
   leaves its fingerprint unchanged. The supplied original path still has mark
   51; `pair` accepts the state and returns an advantage of **110 cents instead
   of 10 cents** on the planted tape. This is a synthetic label-integrity
   example, not measured strategy performance.
3. **Focal-side identity:** a state prepared just before the ask decision at
   200 ms has already processed the bid. Passing it to a bid fork at the same
   tick/episode is accepted and records **two bid KEEP commands at 200 ms**.

Positive controls: a valid saved state works; the existing wrong-tick guard
refuses its known-bad input; direct construction refuses the wrong scenario.
The defect is specific to the saved-state contract, not failure of every guard.

This matters for R2 because a cache keyed only by episode/tick can reuse the
wrong side's state or attach the wrong target/scenario identity. All internal
cash/fill identities can still pass while the label describes another action.
The current slice calls `pair` without `state=` (`:603`), and F5 passes a matching
state. **No saved slice value is invalidated by these counterexamples.**

Required fix: make a saved fork an explicit, validated context. Bind at least
scenario/contract, tape and instrument identity, original start/absolute END,
END-mark provenance/value, episode, tick, focal side, generation and which sides
have already acted. Include those fields in the integrity fingerprint. Either
remove redundant path/settings arguments when a saved context is authoritative,
or reject any disagreement with them before executing either arm. Return
metadata from the validated executed context. A saved state for another side
must be refused, not repaired by executing that side again.

Add these three cases to the selftest/mutation coverage, plus a changed endpoint
and tape identity. Keep valid round-trip serialization/reload as the positive
control. This is a bounded interface repair; it does not require another full
tape experiment or any model fit.

## Do the slice artifacts support the receipt?

Yes, for the declared implementation experiment. The audit verified the actual
source/input/output hashes, committed CSV/JSON copies and the decompressed
histories against the original JSON hash. Input hashes agree with the archived
keep/pull receipt. Selection was independently rebuilt from the archived orders:
exactly the first **128 eligible orders per side**, **256 total**, all with paths.

The audit independently reconciled:

- all saved single-generation values to the archived M50 fills, for both KEEP
  and first-review PULL, with the original prices and marks;
- the reported 13 regression predicates and C-parity readings in the saved
  table (the C/raw-tape slice was not rerun here);
- **138 tapes**, **276 base runs**, **1,066 valid forks** and **38 not-live
  retention anchors**, with exact coverage/status counts;
- every saved arm's fill-ledger END value, every paired difference and common
  pre-fork fills; every saved request's effect delay against its own submission;
- later-entry/re-entry coverage and the base-run cap/pending counter totals.

The reported execution improvements therefore have evidence. They do not show
that a model chooses good timing: these are deterministic stand-ins, and their
WAIT arms post at the next tick by construction. Cap-forced inactivity remains
separate from a negative forecast. Costs are omitted in the current mechanical
value; the receipt discloses this, so these values must not be called net P&L.

One scope qualification for the timing checker: `pull_effect_equal` at `:577`
accepts any `why == 'structural'` without independently validating its timestamp.
The actual saved adaptive request histories pass the independent delay check in
this review. For the next guard-test update, make the structural branch of that
predicate test its declared time as well; a reason string alone is not timestamp
reconciliation. This does not change a saved reading and is not a second
execution defect.

## One concrete next priority

**Close the saved-fork identity contract and publish its narrow R1 closure
receipt.** Deliver the guard/fingerprint repair with the counterexamples above,
a valid serialization round trip, and unchanged valid-path ledgers. This is the
remaining prerequisite for R2 labels. The old fill-notice P1 stays closed; do not
reopen it or restart the engine investigation.

After that passes, R2's first reviewable deliverable should be a **versioned
shadow-probe/feature/support builder and its deterministic manifest**, verified
on synthetic fixtures and a fixed consumed Oct23 slice, before fitting PI0-50.
The following decisions are recorded now so development can prepare that work
without another design round. They do not declare that R2 or teacher fitting has
already passed a gate.

## R2 preparation contract to carry forward after closure

### Shadow probes and deterministic sampling

Use **Oct23 only** for teacher preparation. Each original episode retains its
original five-second absolute END. Keep every original opportunity/status in a
census; select neither on realized fills nor on target sign.

1. **Neutral entry shadows:** original entry at offset 0, plus offsets `50*k`
   for `k=1..98` where `k mod 5 == phase`. Use 30-share decision-time JOIN;
   preserve the current post-only, queue-M and timing rules. Offset 4950 is
   explicitly infeasible because activation would exceed END. One shadow path
   is an alternative world, not another simultaneous order on the same side.
2. **Neutral retained-order shadows:** record the live decisions of each probe
   through the same END, including naturally partial fills and the naturally
   evolved queue. Do not manufacture partial size or queue position to fill a
   training cell. Keep all statuses in the 50 ms census; sample labels every
   fifth global episode-grid tick, with the phase below. Also retain the first
   eligible live review of each probe so the first-review boundary is covered.
3. **Joint state coverage:** use the already fixed POST_KEEP and CHASE policies
   on Oct23 to supply deterministic additional states with nonzero inventory,
   other-side exposure and later generations. Label these source populations
   separately. At sampled feasible absent/live states, probe the focal order's
   neutral continuation from that exact state. No learned score selects rows.

The teacher's J targets remain **single-order neutral moments**, not signed
multi-generation paired quantities: POST includes that shadow order's fills
from activation; RETAIN includes its still-removable fills at/after the current
decision plus cancel latency. Hold under the frozen structural rule to original
END, with no replacement of the focal order inside this neutral target. Keep
zero/no-fill/refusal outcomes. Reconcile `q`, `q * signed_move`, value and severe
excess from individual fill prices/times. The later candidate's paired targets
remain the distinct `paired_continuation_v2` branch differences.

Manifest version `r2_shadow_sampling_v1`: canonical UTF-8 key
`r2_shadow_sampling_v1|day|instrument_id|original_episode_id|side|head`; phase is
the first eight SHA-256 digest bytes interpreted as unsigned big-endian, modulo
5. Use `head=post` for entry offsets and `head=retain` for retained rows. Do not
use Python's process-dependent `hash()`. Sort by stable original identity,
offset, generation and decision. Save the selected keys, unsampled statuses,
source hashes, exact phase function and all row counts before fitting.

Total training weight is equal per original episode/side/head across all of its
included probes and source families; descendants are not independent episodes.
For J's conditional quantity/movement fits, subset the same declared weights
to positive-quantity rows; do not resample fills. Sampled training does not
change the 50 ms evaluation/decision grid. Purge any target crossing a day split.

### Support and fallback status contract

Before labels/fit, declare fixed diagnostics for side and head, remaining
horizon `(0,100], (100,250], (250,500], (500,1000], (1000,2000], (2000,3500],
(3500,5000]` ms; remaining shares `0, (0,10], (10,20], (20,30]`; queue unknown,
zero, `(0,30], (30,150], >150` shares; generation `1, 2, >=3`; inventory sign;
other-side state; and price `[1,5], (5,20], (20,80], (80,95], (95,99]` cents.
Report distinct original episodes as well as rows and nonzero-label episodes.

For the initial support screen, require at least 200 distinct Oct23 episodes
per side/head/horizon cell and at least 20 contributing nonzero-quantity
episodes for its conditional J heads. These are engineering floors, not
confidence guarantees. Require that the applicable remaining-size, queue
missingness, generation, inventory-sign and other-state categories occur in
that side/head's training data, and report their finer cross-cell counts.
Declare observed training ranges for mandatory numeric context; outside-range
states are not silently clipped into support. Sparse fine cells remain explicit
diagnostics rather than an after-the-fact threshold search.

Keep the causes distinct: `no_path`, `missing_mark`, `off_grid`, `post_pending`,
`cancel_pending`, `cap_blocked`, `near_end`, `feature_unavailable`,
`unsupported_horizon`, `unsupported_state_category`, `outside_training_range`,
`insufficient_training_support`, `incoherent_forecast`. Unsupported absent
states WAIT; unsupported live states KEEP, subject to the independent structural
and risk rules. Pending states process events. Do not turn any of these into a
negative label. A post-only refusal discovered at activation is an outcome, not
a decision-time support filter. The queue-unknown convention retains its mask;
it is not automatically a missing-feature refusal.

If these predeclared screens leave little supported delayed entry, report that
coverage failure before candidate labels. Broaden the probe design in a new
manifest; do not silently lower floors after reading Oct26/27 outcomes.

### Mandatory context names and units

Retain the existing 91-input definitions. Append the following causal context
where not already present; keep raw units and hash the final ordered list.

| Field | Definition / unit |
|---|---|
| `t_to_end` (existing) | Original absolute END minus decision time, **ms**; never reset on a repost. |
| `ctx_has_generation` | 0/1, whether this side has an open generation known to the client. |
| `ctx_generation_age_ms` | Decision minus submission, **ms**, for an open generation; 0 with the absent mask otherwise. Existing activation-relative order-age inputs keep their original meaning. |
| `ctx_generation_ordinal` | Current generation number on this side, or next generation number when absent; integer starting at 1 within the original episode. |
| `ctx_inventory_shares` | Signed acknowledged inventory, **shares**, positive for bid-side accumulation. |
| `ctx_bid_reserved_shares`, `ctx_ask_reserved_shares` | Client worst-case outstanding exposure by side, **shares**, including submitted orders and unacknowledged fills. |
| `ctx_self_state_absent`, `ctx_self_state_post_pending`, `ctx_self_state_live`, `ctx_self_state_cancel_pending`, `ctx_self_state_terminal` | One-hot 0/1 client state for the evaluated side. |
| `ctx_other_state_absent`, `ctx_other_state_post_pending`, `ctx_other_state_live`, `ctx_other_state_cancel_pending`, `ctx_other_state_terminal` | One-hot 0/1 client state for the other side. |

Existing `fp_remaining`, `fp_queue`, queue-unknown mask and shown-depth fields
keep their established shares/normalization semantics. Portfolio context must
come from acknowledged/reserved state, never `Engine.rem`, the future activation
frame, final inventory or an END mark. Episode/generation IDs are join keys;
only the declared within-episode ordinal is a feature. Include the mandatory
context in rich, fair and mechanical comparisons consistently. Avoid introducing
a second duplicate time-to-END feature under a new name.

### Reuse boundaries and first R2 receipt

- Reuse `pf_pipeline_export.real_features` and its underlying market builder
  unchanged. Reuse `known_steps` and `order_inputs` calculations unchanged,
  supplying the correct generation's absolute submission/activation, price,
  fill-notice history, decision times and **original** terminal time. Do not
  reuse the old exported labels, initial-order indexing or future activation
  queue as a shortcut. The new adapter owns identities and portfolio context.
- Reuse the J architecture in `pf_pipeline_bench.fit/predict`: four moment/
  severe heads, side/kind split, pinned parameter dictionary, objectives,
  clipping, round-trip coherence and 200 rounds. New feature-list/weight
  plumbing must be explicitly versioned; the existing `fit_one` has no weight
  argument, so claiming byte-for-byte unchanged fitting would be false. Do not
  reuse `rows_of`'s old-day selection and fixed thinning for the new manifest.
- Resolve the cost convention **before teacher/target fitting**. R1's gross
  mechanical value is disclosed; do not defer a target-changing cost choice
  until R4. Record explicit zeros/unmodelled costs if retaining the current
  gross research convention, or version an actual cost change consistently.

First receipt: manifest and adapter code, all-row schema/status census for the
fixed first 128 original opportunities per side on consumed Oct23, no teacher
fit; synthetic later-birth/partial-fill/hidden-notice/queue/short-horizon cases;
full-feature causal perturbations including receipt times and private notices;
independent per-fill label reconciliation; support tables and memory footprint.
The population slice is fixed before outcomes. After this preparation passes,
build Oct23 teacher data, fit/freeze PI0-50 and its support contract, then build
Oct24–25 candidate labels and Oct26–27 consumed diagnostics. Existing early
access means none of those dates, nor Oct28, is forward validation.

## Verification limits and ownership

Five bounded codex-slot invocations completed, about seven seconds total, peak
RSS 304 MiB. These comprise the pinned synthetic selftest plus independent
synthetic/artifact checks; the latter two were repeated once after strengthening
the saved-mark counterexample. Full 40-mutant logs and the harness were inspected,
not rerun. No raw tape, full label build, fitting or untouched date was opened.

Slice as-of **2026-11-19 17:49:35 UTC**, 256 orders / 138 consumed Oct24 tapes;
final independent audit as-of **17:56:23 UTC**. This is implementation evidence,
not prediction quality or profitability. Further execution issues may exist
outside this bounded scope; the report does not claim venue-level fill validity.

Only review evidence and review-owned state are committed. Implementation,
other-session state edits and unrelated data deletions are preserved. No push,
external messaging or scheduler action.
