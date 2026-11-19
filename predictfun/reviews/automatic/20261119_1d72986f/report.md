# Review v2 closure and R1 design decisions — 2026-11-19

Pinned target: **1d72986ffa9cee8285bcef838fe2d054b16b2486**.
Range: **92d8e52fa35ad3d074e2a872b4842d6d98113287..1d72986ffa9cee8285bcef838fe2d054b16b2486**.
One development commit. Reviewed in detached `/tmp/pf-review-1d72986f`.
Request: development session aea5bfb9; review and next-priority loop authorized
by the user. Implementation remains with that session.

**The committed v2 corrections pass review; the prior findings are resolved.
No new defect was found in the changed committed screen/timing logic or its
saved correction receipts. R1 is not ready to pass: a separately captured,
uncommitted engine draft exposes an unacknowledged full fill through its
structural-cancellation state. The next priority is to close that causal
boundary and complete the R1 engine gate, before teacher fitting or labels.**

Evidence: [verification script](verify.py), [targeted cases](synthetic.json),
[saved-artifact audit](artifacts.json), [pinned selftest log](selftest.log),
[R1 design counterexample](wip_design.json). The captured
[draft](wip_engine_snapshot.py.txt) is review evidence, not implementation added
to the development module. Its hash is
`b7e7b5b979213f8885a70458da4851d5d44801c51c2c2340f4d02fa2d2412ce6`;
capture metadata is in [wip_snapshot.json](wip_snapshot.json). Conclusions about
that draft do **not** attribute its code to the pinned target commit.

While this review was running, development committed R1 as
**a9303c1f8b21054f3c90f0f22888a349c69deb02**. The relevant `Engine.advance` and
client status/notice/view/reservation functions are unchanged in that commit
([static identity check](head_advance.json)); the finding therefore remains
applicable there. This is not a full review or acceptance of that newer commit's
synthetic suite and slice results. The requested review range remains pinned.

## Prior findings: all resolved in the target

| Finding | Pinned evidence | Independent verification |
|---|---|---|
| Episode-balanced zero/sign/amount | `pf_pipeline_paired_review.py:242–258` computes the balanced means before the zero and sign/amount decisions | The unequal-row and equal-row cases both PASS; wrong sign and amount FAIL; both all-zero and weighted-zero/raw-nonzero cases are NOT_EVALUABLE; insufficient support is distinct. Six NaN/infinity prediction/label cases are refused. |
| V3 timing tolerance | `pf_pipeline_paired_review.py:170–173`, `rtol=0, atol=1e-3` in milliseconds | Absolute **1 microsecond**, not 1e-6 ms. At both relative and epoch-millisecond scales, a 0.0005 ms displacement is accepted and 0.002/0.01 ms displacements rejected in decision/effect/END fields. This accounts for float64 timestamp precision without a relative allowance. |
| Nesting overgeneralized to adaptive ENTRY-PA | v2 receipt `report.md:24–31` | Explicitly restricted to current same-admission A-PA; recommendation to replace future ENTRY-PA is withdrawn. Future own-state policies require an actual nesting test. |
| B1a described as the actual loss correction | v2 receipt `report.md:32–37` | Explicitly sensitivity only, with old-engine/day/population and noncausal reweighting limitations. |
| Matched-control estimands conflated | v2 receipt `report.md:38–43` | Original above-mean PASS on both days remains; the separately declared conditional draw test passes Oct26 only. Neither becomes a forward confidence claim. |
| K7 NaN description | v2 receipt `report.md:20,49`; `pf_pipeline_paired.py:309–310` | Correctly says the condition rejects NaN; only the printed mismatch count can omit it. No reopened code defect. |

The v2 source hash matches the committed module and both saved receipts. The
source/input/output manifest, actual data artifacts and committed copies match.
All numerical readings and the **60 saved prediction groups per day** are
unchanged. The v1/v2 ledgers and null-draw files are byte-identical. This is a
diagnostic repair, not a newly trained model or improved performance result.
The previous R0 label/admission repairs remain closed; no refit is warranted.

## Open finding, ranked by severity

### P1 — R1 draft: client structural decisions depend on an unacknowledged fill

Scope: the hash-pinned **uncommitted** draft only. This blocks the R1 causal
engine gate, not acceptance of the committed v2 correction.

At draft `pf_branch_engine.py:159`, `Engine.advance` skips structural processing
when engine-truth `o['rem'] <= EPS`. At lines 164–165 it otherwise creates a
client structural request. `Branch.status` at lines 226–231 exposes the existence
of that request as `cancel_pending`, even while `known_filled` correctly delays
the fill by NOTICE=100 ms at lines 213–214. The opposite side's view also reads
this state at lines 247–254.

Independent reproduction, all synthetic:

1. Submit a 30-share bid at 50 cents at 0 ms; activation is 58 ms.
2. A full match occurs at 80 ms. Its private notice is due at 180 ms.
3. At 100 ms the same received book in both cases has bid/ask 55/57, receipt
   time 95 ms; the order is more than three ticks behind. Known fills and known
   inventory remain zero.
4. With the unnoticed full fill, the draft issues **no structural request** and
   exposes `live`/KEEP. Without that fill, it issues a request at 100 ms and
   exposes `cancel_pending`. An unnoticed partial fill follows the latter path.

The observed inputs supplied to this structural rule are the same received book,
order identity and acknowledged quantities. Hidden engine remainder determines
the command/state. The test does not assume that arbitrary public trade-flow
inputs must be identical: the current rule does not consume those inputs or a
causal fill-estimation rule. It branches directly on private execution truth.

Controls: without the structural trigger, the two client views are identical
before the notice; after the notice, the position difference is legitimate.
The existing draft F6 at lines 584–592 perturbs events with **venue time after**
the decision, so it does not test this earlier-match/later-notice window.

Required resolution: issue client structural requests from client-known open
generation/remaining/reservation state. A request for an already filled order
can have no execution effect, but must still follow the declared client request
and acknowledgement path; do not suppress it using hidden fill completion.
Keep engine-truth fill allocation separate. Add before/at/after-notice controls
for zero, partial and full fills, both sides, and another-side policy sensitive
to pending state. Entire feature vectors, issued commands and reservation
transitions must remain causal. No measured economic impact on saved runs is
claimed here.

## Decisions on the five R1 questions

### (a) Acknowledgements: accepted as explicit baseline assumptions

For the first bounded R1 contract, use:

| Event | Assumed time | Client use |
|---|---|---|
| Submit at d | d | Reserve exposure immediately. |
| Activation and post acceptance/refusal notice | d + 58 ms | Client learns acceptance/refusal at that event; zero additional post-ACK delay is an explicit assumption. |
| Match at u | u | Execution accounting only until notice. |
| Fill notice | u + 100 ms | Transfer known fills from reservations into known inventory. |
| Live-order cancel effect after request c | c + 37 ms | Execution stops, but the client does not release reservations yet. |
| Pre-activation cancel effect | max(c + 37, d + 95) ms | Preserve the existing pending-cancel rule. |
| Cancellation acknowledgement | cancel effect + 100 ms | Release only the remaining acknowledged exposure; re-entry can be considered on the first grid at/after receipt. |

These are simulation assumptions, **not measurements of the exchange**. In
particular, “same private feed” is a modelling rationale, not evidence that post
and cancellation notices have those observed delays. Record activation,
post-notice delay (0), fill-notice delay (100) and cancel-ACK delay (100) as
separate named contract fields; equal numbers must not silently share semantics.

For a request at 100 ms: effect 137, ACK 237, next eligible grid 250 ms. There
is no additional cooldown. A complete fill notice may close an order earlier
than its cancel ACK once all its quantity is acknowledged. A fill at the removal
time is excluded; activation and END remain inclusive as declared by the engine.
At a decision tick, drain events/notices due by that tick before constructing
the client view, then process structural and policy decisions with an explicit
tie order. Do not remove existing pre-activation structural safeguards.

With these constant delays, every admissible pre-removal fill's notice precedes
the cancellation ACK. If future timing scenarios allow the ACK to overtake fill
notices, the contract must carry a final cumulative-fill watermark or retain
unresolved reservations. Do not release unknown exposure merely because a faster
ACK arrived. This does not require an ACK sweep or new data for R1.

### (b) Deterministic continuations: accepted for R1 mechanics

Hold-once, chase, reactive and post-keep are suitable frozen, causal test
policies. They do not establish the quality or support of PI0-50. No supported
teacher is needed to verify branch execution; that is R2.

For the “continuation matters” control, compare continuation policies from
**the same saved fork state and roll-in**, with the same END and tape. The
draft F5 at lines 570–579 invokes `pair` separately with HOLD_ONCE and CHASE;
`pair` at line 325 also uses that policy to generate the pre-fork history.
Assert that those fork states are identical, or explicitly fork one saved state
and vary only the continuation. A sign difference caused by a different roll-in
does not isolate continuation value. Demonstrate both sign directions with
engine-generated fills, then preserve the cases in the receipt.

### (c) Submission guard: accepted; parity alone is not a causal contract

Accept applying the structural guard only **after the generation's submission
tick**, since adaptive submissions occur after that tick's structural pass. The
old listed-order kernel otherwise processes future orders before they exist.
For zero-time submissions the compatibility behavior remains unchanged.

Keep a clearly named C-compatibility mode for regression, but distinguish it
from the client-aware adaptive mode. The P1 finding above means “one departure”
is not a sufficient description of the finished client engine: it also needs a
causal boundary between hidden fills and client structural commands. This can
leave original single-generation fills and values unchanged while changing
client event/state traces. Record the correction rather than reproducing a
noncausal client state just to preserve an old trace.

Also call current P1 **numerical aggregate parity**, not bit-for-bit/event
parity: draft lines 486 and 496 use tolerances on per-order quantity, cash and
position, while the C interface does not return individual fill timestamps.
Declare tolerances and compare END value too. The archived keep/pull ledger
provides the separate fill-time regression check; adaptive branches need their
own event/state falsifiers. Passing the same error through two implementations
does not establish causality.

### (d) Size 30, cap 30: intended for the bounded experiment

Yes. With I=+10 and no remaining bid reservation, a new 30-share bid would reach
+40, violating the +30 cap. It must wait until acknowledged inventory is <=0
or otherwise changes enough to make the full order feasible. The opposite ask
may reduce/offset inventory, subject to its own reservation and one-generation
constraints. The mirror holds for shorts. Existing resting remainder is not
automatically cancelled merely because a partial fill occurred.

Keep the cap and full-size/no-resize choice as declared. This is a deliberately
restrictive first research environment, not an optimized inventory policy.
Report feasible absent opportunities, cap-blocked time/opportunities by side,
pending-state time, model-selected WAIT and feature/support fallbacks separately.
Do not attribute cap-forced inactivity to poor prediction or add hidden resizing
to improve quoting. Both worst-case inequalities and engine inventory bounds
must be checked after every notice/command; pending asks never net away the
worst-case bid reservation.

### (e) Bounded slice: sufficient for baseline regression, not all of R1

Use the proposed two single-generation controls, but add adaptive checks on
the same bounded tapes. Freeze the selection before reading their results:
**Oct24, first 128 eligible archived opportunities per side, sorted by oid
(at most 256 total)**. This explicitly covers both sides without selecting on
fills or profit. Keep every selected identity and any reconstruction failure as
a status. Do not increase the prefix until a desired economic outcome appears.
This is a consumed-data implementation check, not a model sample-size claim.

If an earlier bounded prefix was already declared/run while this response was
in flight, retain that membership and its superseding receipts for baseline
regression. Use the fixed first-128-per-side subset of the available archived
opportunities for the additional adaptive check; this does not erase the prior
consumption or require repeating a larger raw-tape run solely to change n.

Required checks for that one bounded gate:

1. POST at 0/KEEP and POST at 0/first-review PULL: reconcile actual submitted
   JOIN price, activation/effect, all M50 fill times and quantities, common END
   mark, cash, inventory and **both** policies' END values to the archive.
   The draft slice at lines 430–434 checks the keep value explicitly but not
   the pull value, and its C check uses quantity only; complete those predicates.
2. For C-compatibility tapes, compare per-order quantities, cash, inventory and
   END value under declared numerical tolerances. Treat expected future-order
   submission-guard differences separately, never as silently ignored mismatch.
3. On the same deduplicated (market, start) tapes, run frozen POST_KEEP and CHASE
   jointly with genuine POST/WAIT and KEEP/PULL forks. Use predeclared anchor
   ticks (0 for entry, 200 ms for retention); preserve infeasible/not-live
   anchors as statuses. Check incremental versus batch **fill tuples**, common
   pre-fork history/END, value identity, reservations and client-event causality.
   These are invariant checks; no profitability threshold or policy selection.
4. Publish branch histories, later-post/re-entry counts, pending and cap blocks,
   refusal/no-path statuses and per-side coverage. Do not claim a transition
   was tested on real tape if it did not occur. Nondegenerate synthetic cases
   must exercise the missing transitions rather than extending data until it
   looks favorable.

The full synthetic table from plan v2 remains mandatory, including the new
unnoticed-fill refusal and the common-fork continuation test. No full tape run,
label build or fitting is needed to finish this gate.

## Checks, populations and limits

Checks used the codex slot through `run_light.sh`, with absolute script paths.
The shared process cap refused two attempts; they were deferred and retried,
without borrowing the development slot. Completed: targeted committed-source
cases, saved-artifact hashes/equality, the pinned module's synthetic selftest,
and the separate hash-pinned R1 synthetic counterexample. Logs are preserved.
The development receipt's 41/41 mutation result was inspected; the full mutation
suite was not rerun independently. The R1 draft's full selftest/slice was not
run by this review.
Four completed bounded checks took about 42 seconds in total, with peak RSS
719 MiB. The head-advance comparison was a static function-identity check only.

Saved receipt populations: Oct26 original 30,242, with path 29,832; Oct27 original
29,820, with path 29,338. Saved outputs as-of 2026-11-19 **17:26:07 / 17:26:44 UTC**;
independent artifact check **17:33:02 UTC**. Labels, exclusions, admissions and
timestamp-check readings are unchanged from the already reviewed artifacts.
The unchanged ledger hash links to the prior independent accounting audit;
this review did not rebuild labels or rerun economic replay.

Oct23 teacher, Oct24–25 fitting and Oct26–27 development belong to the existing
consumed experiment. No untouched day was accessed. Synthetic correctness and
receipt equality cannot establish predictive quality, quoting improvement or
profitability. No R1 label/model performance is claimed.

## One concrete next priority

**Finish R1's causal client/event boundary and publish its engine-gate receipt.**
First eliminate the unacknowledged-fill structural-state leak; adopt the five
decisions above, add the common-state continuation and boundary controls, then
run the fixed bounded regression/adaptive slice and commit the engine, contract,
tests and receipt for review. Do not refit the old model or start R2 labels yet.
After R1 passes, the next stage is the supported remaining-horizon PI0-50 teacher
and target construction in R2. No extra user permission round is needed.

This session changes review evidence and review-owned state only. Uncommitted
development code, existing state edits and unrelated data deletions are
preserved. No push, external message, scheduler action or implementation fix.
