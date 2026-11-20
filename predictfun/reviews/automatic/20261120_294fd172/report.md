# R2-B full-day data review — 2026-11-20

Pinned target: **294fd172d7712bdfeb4c776a33ba63eaf1b7fc0a**.
Range: **e6b0efa481d203e8d9fca2397b930d606c0c378a..294fd172d7712bdfeb4c776a33ba63eaf1b7fc0a**.
Reviewed from the detached checkout `/tmp/pf-review-294fd172`.
[Development receipt](../../r2b_full_day_20261120/report.md),
[prior review](../20261120_c8d8fc8d/report.md),
[whole-build identity/aggregate audit](metadata.json),
[20-part row audit](sample.json), [acceptance/log audit](logs.json),
[support/partition CSV reconciliation](csv.json).

**ACCEPT the full Oct23 teacher-data build for the next supported pi0-50 fit.
No new BLOCKING finding.** The sharded implementation preserves the declared
population and partition-local computations; gate3 reproduces the accepted
slice. All 477 actual records and parquet files match their hashes, the complete
record aggregates reconcile, and the independently selected 20-part row sample
passes. The v1-to-v2 acceptance change is the log-instrument correction, without
changing the data predicates or support floors.

The two prior operational P2s remain **NONBLOCKING for this fit** because their
acceptance workarounds held. Runner v3 is outside this pinned review. Neither a
data rebuild nor another user approval is required to proceed with the existing
authorized research loop. This is acceptance of consumed training preparation;
it is not evidence that the teacher predicts value or makes money.

## Ranked findings and prior closure

### P2 — NONBLOCKING: historical resource stops still need the external check

The previously reported defect remains at `pf_r2_runner.py:183,221–225,253–258`:
resume initializes a fresh latch and finalization does not use historical part
breaches to select its completion state. The watchdog still starts after loading
at `:173,183` and ends before finalization at `:219–220`.

**The workaround held on this build.** Every one of the 477 records is PASS,
has no breach flag, and independently satisfies elapsed <=900 s and high-water
<=1,536 MiB. All four saved shard statuses match their exact modulo ownership,
have no pending parts or latch, and are PARTS_DONE. The 17 declared build/gate
logs have passing return codes and peaks within the limit. The maximum part time
is **171.453 s** and maximum shard high-water **1,421.438 MiB**. No stop/resume
disposition was needed on the saved launch chain.

Fold the planned v3 enforcement into the next development round. Do not rewrite
this build's plan, receipts or source identity to pretend v3 generated it. A
later code version does not invalidate these immutable training files; verify
them against their actual producing plan and commit. Re-finalizing the old
build, if necessary, must respect its frozen source identity.

### P2 — NONBLOCKING: standalone finalize still lacks its own source preflight

`pf_r2_runner.py:232–240` still enters record aggregation without asserting that
the executing source map equals the plan; the receipt copies the plan's sources.

**The workaround held.** `preflight.py:12–17` records actual and planned maps;
`launch5.sh:38–39` requires that process to succeed before full finalization.
The saved preflight at **09:43:07Z** precedes the census at **09:43:15Z**, names
the correct plan hash, and both maps independently equal the pinned source.
The run logs also support this order. No stale finalizer was found.

Carry the built-in check into v3 as planned. The reported prepared v3 and its
25-mutant suite are not committed at this target and are **not reviewed or
certified here**. The current workarounds satisfy this build's acceptance.

There is no new source defect demonstrated by this review. The prior
materialization, visibility and multiplicity findings stay closed; the shadow
builder and branch engine are unchanged. The previously pending shadow suite
is now archived with **56/56** caught and a passing unchanged control.

## Sharding and launcher review

`run :169–173` selects exactly `i % n == k` and loads only those partitions'
markets. `:181–182,221–225` uses that same set for pending work and writes a
distinct shard status. Partition membership, previous-original identity,
`process_partition`, weights, targets and audit selection are unchanged. Each
part still has one writer in the declared four-shard launch. `real_inputs`
(`pf_pipeline_export.py:285–299`) restricts market-specific fair data while the
day's tracked raw hours/depth inputs remain shared, explaining why the frozen
hour/hash assertion is valid for each shard.

The new synthetic shard control at runner `:440–452` passes on the pinned
source. All 20 runner mutation targets resolve, and the saved 20/20 log includes
the two new wrong-shard mutants. The entire mutation suite was not rerun here.

Gate3's five predicates are supported: four disjoint one-part shards completed;
the plans differ only in the runner hash; the **actual four parquet files** are
byte-identical to gate2A; non-resource record contents and support agree; saved
peaks are within the limit. This review checks actual bytes in addition to the
hash strings compared by `shard_check.py:30–33`.

`run_light.sh:18–26,34–35` maps codex to core 8 and Claude's four slots to
9/10/13/14, increases the process cap to five, and retains one-core affinity,
one-thread setting, the address-space cap and memory/disk preconditions. This
matches the user's four-core development authorization while preserving the
codex slot. No additional worker was launched by this review. No current-run
resource or ownership violation was found.

## Acceptance v1 → v2

**Confirmed: the change is confined to check (6)'s log predicate, its helper,
description and added log diagnostics.** The source from `TAG =` through the end
of checks (1)–(5) is byte-identical. The shard/record disposition predicates and
CSV-producing code outside the log predicate are unchanged.

The saved v1 failure names **`accept_full.log`**, the acceptance's then-current
output, with no footer. The retained file is now named `accept_full_v1.log`.
Every one of v1's 17 build-log footer tables equals v2's, and their return codes
and peaks pass. Saved results differ only in `6_run_disposition`; notes and
failing-part dictionaries agree. The v1 failure is preserved, not relabeled PASS.

`accept_logs.py:23–29` requires declared build logs to exist, checks every
non-acceptance log, and excludes acceptance-prefixed logs to avoid reading its
own unfinished output. The acceptance logs' own exit codes were checked
separately here: expected failure for v1, success for v2 and the helper checks.
All **11 falsifier cases** pass independently; rerunning the small mutation suite
catches **9/9** planted errors by a check rather than by a crash.

The statement about v1/v2 historical CSV byte equality is not independently
reproducible from two retained CSV snapshots: only the final copies are in this
commit. What is independently established is unchanged CSV/data-check code,
unchanged bound inputs, equal saved numerical readings/notes, and consistent
final files. This provenance limitation does not change acceptance or require
another full scan.

## Saved data and verification scope

Consumed **Oct23 only**. Census as-of **2026-11-20T09:43:15.569062+00:00**;
development acceptance v2 as-of **10:21:00.689582Z**. Independent metadata audit
as-of **10:28:25.209555Z**; row-sample audit as-of **10:29:43.409819Z**.

| Item | Verified result and scope |
|---|---|
| Immutable plan | `249161127c39d5a51cc7c374262073af4f38c9245ef08fb505f0ec3e646dd56b`; decompressed committed plan/census equal actual files; producing source and schema/settings/cost identities match |
| Population | All 30,466 originals, 15,233 per side; independent `(market, original)` partitioning gives 477 disjoint complete parts, 476 × 64 plus 2; original status counts match |
| File integrity | All 477 completion records and all 477 actual parquet hashes verified, approximately 1.08 GB total; actual record hashes match the committed ordered manifest |
| Whole-build aggregates | 25,463,127 census rows; 5,384,015 sampled-ok rows; 60,176 original/side/head weight groups; status/refusal/support contributions reconcile to the census |
| Saved label reconciliation | 5,384,015 counted, zero failures; 4,036,595 no-fill rows retained. These are the generating run's per-fill readings, not a new raw-tape reconstruction |
| Saved causal checks | 191,620; zero in every failure/not-evaluable component; 83 retries. Every record's keys, original/member identity, seeds, draw sequence/statuses and firing flags checked |
| Independent row sample | Fixed parts `0–3,116–119,236–239,356–359,473–476`: 1,218 originals, 1,040,282 rows, 219,999 sampled ok, 7,728 audit keys; membership/multiplicity, weights, schema, labels, clocks, support and independent audit selection pass |
| Sample falsifiers | Wrong value, duplicate identity and missing audit each refused; clean sampled parts pass |
| Resources | Four saved shard high-waters 1,124.594 / 1,421.438 / 1,265.129 / 1,210.168 MiB; per-part and process-log bounds pass |

This was **not** a second full 16-minute row scan. Whole-build numeric row
predicates are supported by the reviewed streaming acceptance, all actual file
hashes and record aggregates, plus the independent fixed sample above. Audit
selection was recomputed from rows on that sample; seed/draw and outcome-record
checks cover all 191,620 saved audits. No real-tape perturbations were rerun.

## Is the support contract adequate for pi0-50?

**Yes, for the declared first teacher fit, with the original inference guards.**
All 28 coarse side/head/horizon cells satisfy the unchanged 200 distinct-original
/ 20 nonzero-original floors. Their smallest distinct count is 2,945; the
smallest nonzero count is **39**, in bid POST with <=100 ms left. Therefore the
screen passes, but short-horizon conditional estimates still have little
nonzero evidence. Of **559 observed category-by-horizon diagnostic cells**, 62
have fewer than 200 distinct originals and 88 have fewer than 20 nonzero originals.
These finer counts were declared diagnostics, not extra fit thresholds. Do not
silently promote them to new selected thresholds or describe all joint states
as well supported.

The structural explanations in the receipt are sound under the engine's numeric
tolerances: POST creates 30 shares (`pf_branch_engine.py:138–140`); reservations
and the 30-share cap constrain inventory sign (`:31–32,261–275`); terminal means
at/after common END (`:244–247`). A zero known remainder is not a live retained
decision (`:254–258`). Queue-unknown and POST queue-zero are **observed absent**,
not proven impossible. Preserve that distinction and the missingness mask.

Keep the full original support contract from c68a072c, not only “28/28”:

- Applicable categories must have occurred for that side/head; the unobserved
  lists remain `unsupported_state_category`, including the observed-absent ones.
- Check mandatory numeric context against its observed per-side/head ranges;
  do not clip an unsupported state into the training domain. Retain the declared
  inventory-sign tolerance and the actual feature units/schema.
- Physical feasibility, pending/structural states and feature availability are
  separate from support. A post-only refusal revealed at activation remains an
  outcome; it is not an admission filter.
- Keep `unsupported_horizon`, `unsupported_state_category`,
  `outside_training_range`, `insufficient_training_support` and
  `incoherent_forecast` distinct. Unsupported absent states WAIT, live states KEEP,
  subject to structural/risk rules; pending states process notices.

The support JSON is adequate input to that contract. It is not yet an executable
support-enforcing teacher; implementing and testing that adapter is part of the
next deliverable. Stand-in POST_KEEP/CHASE coverage does not guarantee that the
learned teacher will remain in-domain or actually reach delayed entry.

## ONE concrete next priority

**Implement, fit and freeze the supported pi0-50 teacher on this accepted Oct23
dataset, with one bounded teacher-policy receipt.** This is already authorized
by the user's implement/review loop and established R2 plan; no additional USER
approval is needed. Do not start candidate paired-label production in this same
deliverable. Make the deliverable reviewable as follows:

1. **Training adapter and immutable population.** Consume these exact plan/part/
   record hashes. Use all sampled-ok Oct23 rows and their declared weights;
   retain zero/no-fill/refusal outcomes. Fit separately for bid/ask × POST/RETAIN.
   Add explicit weight and 107-feature plumbing to the existing J architecture
   in `pf_pipeline_bench.py:118–157`: p on all rows, conditional q and qm on
   q>1e-12 with the same weights subsetted, severe on all rows. Sixteen heads
   total, pinned parameter dictionary/objectives and **200 rounds**, no tuning
   or early stopping. Version that plumbing; do not reuse the legacy `rows_of`
   day selection/thinning or claim unchanged fit code when adding weights.
2. **Complete causal policy.** Reuse the feature/known-state builders unchanged
   on the **actual current branch state**, including later generations, other
   side, notices/reservations and the original END. Do not look up a convenient
   shadow row or reuse its stand-in-policy state at inference. Apply the support
   adapter above and the existing moment clipping/round-trip coherence. On the
   50 ms grid, supported coherent V>0 permits POST; V<0 requests PULL; zero means
   WAIT when absent / KEEP when live. Keep engine ordering, cap, timing and
   structural rules. No timer, severe-risk penalty or inventory-utility tuning
   is introduced in this first teacher.
3. **Checks and bounded training execution.** Before the full fit, test feature
   order/identity exclusion, whole-original weighting and conditional subsets,
   unsupported-category/range/coherence fallbacks, hidden-notice causality,
   action signs/ties, and saved-model reload equality with falsifiers. Stream
   narrow input columns or use a file-backed matrix; train one side/head group
   within the existing resource limits instead of materializing 25.5 million
   full census rows. No outcome-selected downsampling to solve memory pressure.
   Record peak/time and any stopped attempt. A resource failure changes the
   execution plan, not permission to silently change the training population.
4. **Freeze and receipt.** Commit the complete trainer/inference/support source,
   producing data identities, ordered features, weights, pinned parameters,
   library versions, model hashes and policy contract. Retain one teacher
   version, with multiplicity recorded. Report per-side/head and horizon
   in-sample diagnostics: fill calibration, conditional quantity/movement
   errors, severe error, coherent/incoherent fractions and predicted/realized
   value in cents. These are training diagnostics, not skill/selection gates.
   After freezing, use the fixed first 128 Oct23 originals per side for a bounded
   chronological teacher rollout: status/fallback/cap counts, actual delayed
   entries and later generations, and ledger invariants. If delayed entry is
   absent or fallback dominates, report that limitation before claiming a
   genuine adaptive timing experiment; do not tune on this readout.

Keep explicit zero fee/rebate **gross** values and 58/37/100 ms engine timing.
Land the prepared operational v3 fixes alongside this work with their own source
and controls, preserving this census's provenance. After the teacher freeze is
reviewed, proceed to the existing Oct24–25 candidate-label stage and Oct26–27
consumed diagnostics. Those days, and previously accessed Oct28, are not untouched
validation. No economic gate is passed by a successful training run.

## Checks, limits and ownership

Six successful bounded codex-slot invocations: pinned runner selftest, complete
identity/aggregate audit, fixed row sample, acceptance/log audit and the small
log mutation suite, plus exact reconciliation of seven committed support/part
CSV views to their frozen JSON data. **83.47 s total, maximum 737 MiB.** See
[runner](runner_selftest.log), [metadata](metadata.log), [sample](sample.log),
[log audit](logs_pass.log), [log mutants](log_mutants.log), [CSV audit](csv.log).

One review-instrument attempt failed (0.92 s) because it expected the archived
filename `accept_full_v1.log` inside v1's JSON; the JSON correctly records its
original filename `accept_full.log`. The failed [log](logs.log) is retained.
The corrected [script](verify.py) changes only that expectation/description;
the exact source used for metadata and sample evidence is preserved as
[verify_v1.py](verify_v1.py), matching their embedded script hashes. The correction
does not change their data checks. No implementation test was counted as passed
on a crash.

No model fit, new labels, raw-tape experiment, untouched-day access, implementation
edit, push or external message was performed by this review. Other-session work
is preserved; only review artifacts and review-owned state prefixes are committed.
