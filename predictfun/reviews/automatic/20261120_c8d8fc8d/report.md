# R2-A full-day launch gate review — 2026-11-20

Pinned target: **c8d8fc8d52faeacb9413b33a679f99de451b3624**.
Range: **4cd7b45ff01c64d1473c695cbee21a8ce799526a..c8d8fc8d52faeacb9413b33a679f99de451b3624**.
Reviewed from the detached checkout `/tmp/pf-review-c8d8fc8d`.
[Development receipt](../../r2a_launch_gate_20261120/report.md),
[prior review](../20261120_b565f671/report.md),
[independent artifact audit](artifacts.json), [independent controls](controls.json),
[reproduction script](verify.py).

**PASS for the pinned full-day launch gate: no BLOCKING finding. All three prior
findings are closed, and the gate2 artifacts support the receipt. Continue the
authorized full Oct23 build; these findings do not require a restart. Two
nonblocking operational gaps remain, with explicit checks below for accepting
the full-day outputs. This is implementation/data-preparation acceptance, not
teacher support, prediction quality or profitability.**

The review is scoped to the committed sequential runner. Uncommitted changes to
the launcher or a proposed sharded runner are not covered by this approval.

## Ranked findings

### P2, NONBLOCKING — a resource-stop latch can disappear on restart

`pf_r2_runner.py:169–174` accepts existing records but initializes a fresh
`latch=None` without checking their resource breaches or the previous run status.
With all parts present, `:212–215` can replace STOP_RESOURCE with PARTS_DONE.
`finalize :243–248` chooses COMPLETE using the new process's high-water memory,
part status and label/causal verdicts; historical breach flags are reported but
do not enter the state decision.

Independent control, on a **temporary copy** of the saved four-part run: give
part 0 elapsed=901 seconds, set its seconds breach and write STOP_RESOURCE.
Resume with no pending parts returns PARTS_DONE; finalization returns COMPLETE
while still listing `breaches=[0]`. The unchanged copy completes normally, and a
copy with a failed reconciliation remains FAILED. No tape was rerun or archived
record changed. See [controls.json](controls.json).

This does not corrupt valid rows or support statistics. Gate2 has no recorded
breach, so it does not invalidate that demonstration. **For the full-day
acceptance, inspect every completion record and prior stop log, not only the
word COMPLETE.** Do not automatically restart through a resource stop. Record
its investigation and disposition; changing limits/membership/audit rules still
requires a new declared plan. Carrying the latch through resume and finalization
belongs in the next development round, without regenerating otherwise valid data.

The hard watchdog has a related scope limitation: it starts at `:174`, after
input loading at `:164`, and stops at `:210–211`, before finalization. The never-reset
VmHWM checks do cover those stages afterwards, and the launcher retains the
address-space limit, but the 2,048 MiB watchdog does not enforce that threshold
*during* those stages. Extend its lifetime in the next operational update;
retain whole-process peak measurements for this run.

### P2, NONBLOCKING under pinned finalization — standalone finalize omits source preflight

`run :158` verifies the actual runner/shadow/dependency hashes against the plan.
`finalize :222–230` does not. It accepts records and then calls the supplied
module's `support_merge`; the receipt at `:260` copies the plan's old source
hashes regardless of which finalizer is executing.

Independent control: a module wrapper with a different source-file hash, using
the unchanged support aggregator, can finalize the copied run as COMPLETE.
Its actual source map differs from the plan. This proves missing refusal, not
incorrect current support arithmetic. Gate2's pinned hashes and support
aggregates match independently; no stale finalization was found there.

**For this full build, finalize with the frozen source and assert its source map
equals the plan before invoking finalize.** Record that assertion and source
identity with the receipt. A changed finalizer must not silently inherit the old
provenance. Add the built-in preflight and a changed-source refusal in the next
round. This operational constraint permits the current build to continue.

## Prior findings and requested launch checks

| Prior item / request | Resolution and evidence at the pinned source |
|---|---|
| Whole-census materialization on return | **Closed.** `pf_r2_shadow.py:717–725` delegates to the runner; `pf_r2_runner.py:219–264` aggregates completion records and rehashes files without reading their rows. The only all-part reader is the explicit bounded helper at shadow `:606–618`. Independent re-finalization with both materializers replaced by raising sentinels completes with the same readings. |
| Unreceived venue-past frame blind spot | **Closed for the specified control.** `hidden_copy :406–424` perturbs the captured state's hidden book, unnotified fill quantities, engine remaining/queue and later trades without re-executing it. `observe :427–431` compares both sides and client context. `causal_rows :485–489` also compares the stored 107-vector to the hidden-state rebuild. The independent former leak fixture now fails visibility for absent decision 0 ms and live decision 150 ms; clean observations stay equal and delivered-book controls fire. |
| Same-partition duplicates masked by sets | **Closed.** Shadow `:648` rejects a repeated source pair; `:700–706` requires one status OR episode rows for each original, and refuses duplicate candidate identities. Guards run before weights/writes. Saved parts independently pass multiplicity, exact membership and canonical-key checks. Synthetic controls cover repeated pair/status/episode cases. |
| Resume identity and atomic checkpoints | Plan content binds declared sources/inputs/settings/schema/membership; `accept :140–151` checks plan/member/file identity. Temporary write and metadata check precede rename; the record is the completion marker (`:192–206`). Interrupted writes are rerun, conflicting records are refused, and failed records are retained. The remaining resource-latch/source-finalizer caveats are above. |
| Never-reset high-water measurement and watchdog | `rss :58–61` reads VmHWM without resetting it. Sampler reset changes only the per-part observation, preserving `/usr/bin/time` peak. Synthetic time/peak/loading stops and real subprocess watchdog exit pass; scope and restart caveats remain above. |
| Audit declaration | Shadow `:434–456` uses the declared compact JSON identity, SHA-256 ordering and per-original/side/source/head `ceil(n/64)` with minimum one. Independent row-based selection matches all 1,598 saved keys; seeds, draw order/statuses, retries and firing flags reconcile. Training sampling/weights are unchanged. |
| First gate FAIL and gate2 | First plans differ only in `tag`; the declared resume-identity predicate failed. `make_plan :105–108,121–137` now excludes location from content identity. Gate2 has identical plans, parts, non-resource record fields and aggregate support across uninterrupted and resumed runs. All eight receipt predicates are supported. Earlier FAIL evidence remains in-band. |
| R1 engine, neutral J target and feature builders | Unchanged in this commit. Source hashes match the accepted preparation; no target/model/cost change. |

The visibility control complements venue-preserving execution replay. It does
not undo legitimately received acceptance/refusal notices. Passing a finite
perturbation sample is evidence for the declared boundary checks, not a proof
that arbitrary future code cannot leak.

## Artifact verification

Consumed Oct23, **256 originals** (first 128 per side, all original statuses).
Saved A as-of **2026-11-20T02:14:22.611131+00:00**; resumed B as-of
**2026-11-20T02:26:31.363289+00:00**. Independent artifact audit as-of
**2026-11-20T03:04:37.550381+00:00**.

| Item | Independently verified reading |
|---|---|
| Source and saved-file identity | All 21 source/artifact manifest entries match; plan/source/settings/schema, committed A/B receipt/support/checkpoint copies and record/file hashes match |
| Restart | A COMPLETE; saved B1 stops after two records; resumed B COMPLETE. Four partition files equal A/B and accepted r2av3 **byte for byte**; records equal excluding elapsed/memory/as-of/breach fields, with no breach in either run |
| Population | Four disjoint parts of 64 originals; exact frozen selection; 213,349 rows, 137 columns; two missing-endpoint originals represented by one status each |
| Sample and weights | 44,959 sampled-ok rows; 508 original/side/head groups sum to one, other rows have zero weight |
| Labels and clocks | Saved per-fill reconciliation 44,959 checked / zero failed, including 32,902 no-fill outcomes. Independent saved-row q/value/qm/severe algebra, decision/END timestamps and original-END horizons reconcile |
| Inputs | Ordered 91 export + 16 context fields, float32; sampled-ok contexts present; all saved values preserved versus r2av3 |
| Causality | 1,598 saved audited checks, zero stored/rebuild, future-world, hidden-state, firing or not-evaluable errors; two second-seed retries. Independent key/seed/draw reconciliation passes |
| Refusals | Entry 312, joint_post_keep 6, joint_chase 12, retained in the unchanged saved parts/records |
| Support | Partition contributions, merged support and prior accepted r2av3 support agree; all 28 coarse cells remain insufficient at this slice size |
| Resources | Saved partition times 156.25 / 161.03 / 156.48 / 141.65 s; whole-process launcher peak 1,179 MiB; record high-water below 1,536 MiB, no breaches |

The audit did not reconstruct discarded fill lists or repeat the 1,598 real-tape
perturbations. Their per-fill/causality execution counts are verified saved
readings, supplemented by independent row algebra and bounded controls. Source
hashes cover the plan's declared dependencies/input list; this is not a claim
that every transitive source file or raw Predict.fun tape file is included in
that list. Preserve the execution checkout and input snapshot throughout the
run and disclose any drift at acceptance.

## Full-day decisions and acceleration

**Completion-record storage: approved.** Keep the approximately 210 KB records
beside their parts in `data/`; commit the immutable plan, final run status,
census/support JSONs, logs and a complete ordered `records.sha256` manifest.
Each hash entry must name its part record unambiguously. Verify those record
hashes, then their bound parquet hashes, before accepting the output. Keep the
actual records accessible. A hash manifest without the records is not enough.
This avoids putting approximately 100 MB of completion JSON in git without
weakening identity checks. No runner change is required for this storage choice.

**Reload on process start: accepted.** The measured approximately one-minute
day-input reload is not a reason to weaken provenance or introduce shared mutable
state. Retain hash/hour/offset checks at each start. Faster check code and cached
window features are reasonable later optimizations, provided the stored vectors,
audit keys/components and refusal controls remain equivalent on paired fixtures.

**A sparser audit is acceptable prospectively.** Declare
`r2_full_causal_v2_div256`: for each original/side/source/head stratum with n
sampled-ok rows, select `min(n, max(1, ceil(n/256)))` by the same canonical identity
encoding and SHA-256 ordering, with the new version in the key. Keep every
107-input component, hidden-state observation, firing control, retry/status rule,
synthetic falsifier and **100% sampled-ok per-fill reconciliation**. No training
row, weight, target or support threshold changes. Freeze source/version/plan and
independent count/key controls before using it; never switch or resample around
a failed audit. This is an engineering audit rate, not a statistical guarantee
on an undetected-leak frequency.

The saved 256-original slice would have **1,214 checks**, versus 1,598 under v1:
about 24% fewer, **not four times fewer**, because the per-stratum minimum often
binds. Using the supplied 0.32 s/check and 3.3 h base-work estimate, proportional
full-day arithmetic is approximately **144,475 checks / 16.1 one-core hours**,
versus about 20.3 h for v1. This is not a full-day benchmark.

**Prefer finishing an already-started v1 build over restarting just for this
saving.** A v1 plan remains valid. Do not mix v1 and v2 parts or weaken the current
plan in place. The pinned source would correctly refuse a stale-source resume.
Review of a later speed/sharding implementation is separate; this review neither
launches additional cores nor certifies uncommitted parallel code.

## ONE next priority

**Finish and accept the full Oct23 teacher-data build and its support assessment,
before any pi0-50 fit.** The launch gate is closed; do not spend another round
rerunning the same four-part demonstration absent a relevant source change or
new failure. The full-day receipt must pass the following output checks:

1. **Frozen population and identity.** All **30,466 originals / 15,233 per side**,
   all original statuses, exactly **477 deterministic parts of at most 64**.
   Match the committed plan's ordered IDs and member lists, no overlaps, missing
   originals, duplicate status/candidate identities or status-plus-episode mix.
   Bind source/settings/schema/cost/audit versions and input identity. Verify
   every actual record and parquet against the committed hashes. Assert the
   actual finalizer's source map equals the plan (finding 2 workaround).
2. **Streaming reconciliation.** In bounded per-part reads, independently
   recompute row/status/sample/refusal totals, canonical identities, weight
   groups and support contributions. Reconcile to records and census exactly;
   never concatenate the full day. Training exclusions remain named statuses.
   Every usable original/side/head weight group sums to one; other weights zero.
3. **Labels and clocks.** All sampled-ok rows counted in per-fill reconciliation,
   zero failures, no-fill rows retained. Verify saved q/value/qm/severe algebra,
   finite valid target fields, decision/effective/END clocks and the original END
   convention on the saved parts. Report statuses/counts by source/head/side,
   including no path, short horizon, pending/structural/refused states. No
   outcome-selected filtering or cost convention change; explicit zero fees and
   rebates continue to mean **gross** values with execution latency retained.
4. **Input and audit integrity.** Ordered 107-input float32 schema unchanged;
   independent canonical-key selection equals every recorded audit key under
   the frozen v1 or prospective v2 rule. Every selected check has its seed/draw
   history and firing controls; zero failures in each comparison component and
   zero not-evaluable checks. Exact audit counts and retry counts reconcile.
   A failure stays a failure, not a dropped row or an occasion to resample.
5. **Support assessment.** Reconcile coarse head/horizon cells, category and
   category-by-horizon counts, numeric ranges, delayed entries/later generations,
   partial fills, queue states, pending flags and other-side states. Apply the
   unchanged **200 distinct-original / 20 nonzero-original** support floors and
   declared categorical/range checks. List which intended teacher states are
   supported and which require absent-WAIT/live-KEEP fallback. Not every cell
   must pass to accept the *data*; sparse cells must never become an implicit
   authorization to extrapolate. No floor lowering to make a fit possible.
6. **Run disposition.** Every part has a passing data/audit record and all stages
   have measured elapsed/high-water evidence. Inspect all part breach flags and
   stop logs, including earlier processes, and independently enforce the declared
   900 s / 1,536 MiB checkpoint thresholds (finding 1 workaround). Any resource
   stop has an explicit investigated disposition before promotion. No silent
   reset of limits or history. COMPLETE alone is insufficient. Archive the
   pending full shadow mutation-suite result; investigate any unexpected survivor.

If these checks pass, the next decision can be a supported pi0-50 training
deliverable. If the data are reliable but intended states lack support, report
that coverage gap and the fallback population before fitting. Oct23 remains
consumed teacher/development data; no untouched day, forward validation or
profitability claim is earned by this census.

## Verification and limitations

The independent artifact audit and synthetic/saved-copy controls passed through
the codex slot with absolute scripts. Pinned runner selftest passes, including
resume/corruption/refusal, watchdog, short-write and materialization sentinels.
The committed runner mutation log reports 18/18 caught; the committed targeted
shadow logs report 18/18 caught. All 18 runner and 56 shadow mutation replacement
targets independently resolve. The full shadow suite was still running at the
target commit and is not represented here as a completed 56/56 run.

Pinned shadow selftest also passes, including the visibility and multiplicity
controls. Four successful bounded invocations took **141.22 seconds**, maximum
**398 MiB**: artifact audit 10.26 s / 397 MiB; independent controls 1.75 s /
269 MiB; runner selftest 28.78 s / 398 MiB; shadow selftest 100.43 s / 264 MiB.
Logs: [artifact](artifacts.log), [controls](controls.log),
[runner](runner_selftest_pass.log), [shadow](shadow_selftest.log).

Two earlier launcher refusals are retained ([first](runner_selftest.log),
[second](artifacts_attempt1.log)); no check ran in those attempts. Checks were
retried when the shared launcher allowed the codex slot, without bypassing its
limits or using a development slot. The development session changed the launcher
capacity during review; that launcher change is not part of the reviewed source.

No model fit, full-day label build, new real-tape experiment, untouched day,
implementation edit, push or external message was performed by this review.
