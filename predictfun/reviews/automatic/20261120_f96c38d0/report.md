# R2-D review — 2026-11-20

Pinned target: **f96c38d01a7f27ed4fd39ddf2ddadeecf162b84b**.
Requested range: **4584a3c2696625d27cf87f193a6260997348f85e..f96c38d01a7f27ed4fd39ddf2ddadeecf162b84b**.
Reviewed committed source in `/tmp/pf-review-f96c38d0`. The intervening
8cc1dc49 is the [prior review](../20261120_4584a3c2/report.md), not a development
change. [Development receipt](../../r2d_labels_20261120/report.md).

**Accept the arm-binding correction and the saved 128-pair integration
readings. Do not yet accept this runner for production label resume and
publication.** Two reliability gaps remain: the new orchestration/input
identity is not enforced, and saved records are admitted without checking
their contents. Independent checks find no such corruption in the delivered
A/B artifacts. These findings concern the next production step, not evidence
that the existing 128 labels are reversed or economically wrong.

Keep **pi0_50_v1** and the existing paired target. Replace exhaustive labelling
with the declared, weighted full-day sample below. The one next priority is
its production launch gate, including these acceptance repairs. No additional
USER approval is needed within the existing consumed-day research and four
development-slot authorization. No teacher refit or candidate fit is authorized
by this review's next deliverable.

## Ranked findings

### 1. P1 — BLOCKING for production resume: recorded execution sources and label-day inputs are not enforced

Evidence: `pf_r2_labels.py:88–105,239–242,359–363,379–399,446–449`.

The manifest records nine source hashes, but neither `pairs` nor `finalize`
compares them with the code executing the job. `teacher_assets` correctly
enforces the **frozen teacher** closure and library versions; the new label
adapter and runner are outside that closure. Consequently, a continuation
adapter change can be mixed with already-completed records under the old
manifest. A fork fingerprint does not establish which adapter executed the
continuation after that fork.

There is a separate input-identity omission in the same preflight:
`world_inputs` discards the hours and tracked input hashes returned by
`pf_pipeline_export.real_inputs` (`:285–299`). The manifest does not bind the
label day's keep/pull metadata, clock/offset inputs and market-feature inputs.
The engine fork binds its tape and state, not all external fair/depth inputs
used by future teacher decisions. Even an unchanged neutral fork can therefore
resume with changed continuation features.

[Independent reproduction](controls.json), using temporary copies only:
declare an all-zero label-source SHA in a copied manifest and update its copied
records' manifest links consistently. The executing source still has SHA
`42c40b16c0f084eef7639611825da6816d25326d984a30dc90fb63c67887c12f`.
The actual frozen-teacher preflight passes, `pairs` returns `PAIRS_DONE`, and
`finalize` exports 128 rows. This is a stale-source-manifest falsifier, not a
claim that the real source was changed during A/B. No new real pair was run.

**Required for production:** bind and enforce the label adapter/runner source
closure, teacher freeze/libraries, sampling plan and every producing input
identity before skipping records, running an arm, or publishing an accepted
dataset. Reuse the existing input identity machinery; do not discard its
metadata. Changes require a new plan/output namespace, never silent reuse.
Add positive resume and stale source/feature-input/clock/metadata refusals.
For a records-only finalizer, verify the producing records' immutable source
and input bindings; if it claims current-source/current-input equality, enforce
that as well. Old artifacts may remain readable as explicitly historical data.

### 2. P1 — BLOCKING for production acceptance: persisted records and labels can disagree while finalization succeeds

Evidence: `pf_r2_labels.py:115–116,400–411,430–434,446–463`;
`reviews/r2d_labels_20261120/r2d_checks.py:59–62`.

`valid_record` checks only `manifest_sha256` and the row key. It does not bind
the stored candidate, executed fork, arms or numeric results. Finalization
trusts stored `reconciled`/`common_prefork_identical` booleans and exports `Y`
without recomputing the identity. The receipt's check C also trusts those
booleans for its numerical reconciliation claim. Binding mismatches are
counted, but do not prevent record promotion. `finalize` does not inspect held
resource stops.

The positive control exports the intact 128 records. Four independent
temporary corruptions are then **all admitted and exported**: `Y += 1`, a
wrong result fork fingerprint, `reconciled = false`, and one arm's
`binding_bad = 1`. An independently detected, undisposed hard-RSS stop also
does not prevent export. See [controls.json](controls.json). This does not
invalidate the current receipt: the independent [record audit](records.json)
checks the actual values and identities and finds all 128 intact.

**Required for production:** one shared validator for resume, promotion and
accepted finalization. Validate candidate/executed context identity, status
schema, finite marks/values, exact arm membership and sign, `Y = R_a - R_b`,
and each `R = cash + position × mark`; check pre-fork, binding, causality,
parity and quantity invariants. Use the established floating-point tolerance,
not a looser repair tolerance. Bind promoted bytes in completion records and
the accepted dataset manifest. Known non-ok outcomes remain counted statuses;
failed invariants are failures, not legitimate censoring. Reconcile every
selected row/status, reject duplicates/extras, and refuse accepted publication
with an undisposed stop. Diagnostic exports may exist only with an explicit
unaccepted status. Include the four persisted-corruption falsifiers above.

### 3. P2 — NONBLOCKING for the valid labels: stop-event identity is fixed, but the watchdog can erase an earlier event

Evidence: `pf_r2_runner.py:94–98,172–198,207–234`;
`pf_r2_labels.py:380–384`.

Runner v4 correctly uses content-keyed history and plan-bound dispositions;
an old filename waiver no longer clears a new event. However, the watchdog
starts before history is archived. Its `poll` overwrites the status directly.
In the runner, loading can occur before `held_breaches` reaches the archive.
A subsequent hard stop can erase a prior unarchived stop in that interval.
The archive's claim that every writer archives first is therefore too broad.

The small synthetic control starts with an unarchived STOP_RESOURCE, invokes
the actual watchdog with a simulated 2,100 MiB RSS and a harmless exit recorder,
then archives: only the new hard-stop hash remains. The old hash is absent.
This is an operational provenance gap, not an observed A/B failure. Fold the
repair into the next round: archive before any status-overwriting writer can
run, including the watchdog and plan/finalize paths. Preserve the event and
plan binding; do not restore filename-based waivers. Meanwhile, retain and
independently review every failed-process log and disposition.

### 4. P2 — NONBLOCKING for the valid labels: resource gate D stays failed; the multi-day census retains live objects

Evidence: `pf_r2_labels.py:330–357,391–408,472`;
development receipt's resource table and logs.

The saved B census peaks at **1,652 MiB** and the three-day controls at
**1,631 MiB**, above the declared 1,536 MiB checkpoint. Being below the 2 GiB
hard ceiling does not turn D into PASS. No label arithmetic failure follows
from those peaks. Pair passes are below the checkpoint.

The diagnosis is not established as allocator retention alone. After
`del ... worlds ...`, the local `world` still references the last world and
its depth object while the next day's inputs load. A synthetic weak-reference
control through the actual census loop proves day one's depth remains alive
at day two's load, then is released after the function returns. The pairs loop
similarly retains `world` and the last teacher factory. Moreover, census itself
has neither the checkpoint guard nor the hard-RSS watchdog used by `pairs`.

Use isolated day/partition processes, release references, and cover census,
loading, pair generation and finalization with the declared resource rules.
Keep the never-reset process high-water reading. Do not silently raise limits.
The known CLI KeyError at `:472` is a further **P3 NONBLOCKING** issue: the
independently verified written census survives it, but fix the exit path for
production. These operational items belong in the same next round, without
requiring a rebuild of the valid bounded integration results.

## Prior findings and implementation review

| Prior item | Disposition at this target |
|---|---|
| P1 teacher bound to the wrong branch in a pair | **Closed in the new adapter.** `pair_bound :174–213` creates a fresh teacher for each executed arm; `bound_fork :148–157` binds roll-in. Forced action and remaining-side order match the engine. Every callback compares the bound branch's view. Saved binding checks: 40,374, zero mismatches. Promotion/acceptance of failed checks is the new finding 2. |
| P2 name-only disposition / lost history | **Partially closed.** Event hashes, reason and plan matching, and HELD behavior are correct. Watchdog overwrite window remains (finding 3). |
| P3 mixed value denominator | **Closed by diag2 and superseding report_v2.** Coherent-conditional means have coherent denominators; the separate all-row realized mean matches the actual training value/weight arrays. Models/freeze unchanged. |
| Enforce frozen library versions | **Closed.** Exact Python/LightGBM/numpy/pandas/pyarrow comparison is called by the adapter's teacher preflight. |

The five census sources, alternating-side ordering, original identity,
fixed END, timestamp scenario and POST/WAIT versus KEEP/PULL signs agree with
the previous contract. Selection is round-robin by source and independent of
label outcomes. Candidate rows contain no listed outcome fields; selected
pi0 forks agree with census roll-in. The final gzip/manifest are deterministic,
and the completed-manifest overwrite refusal works. Freeze SHA remains
`769dff779d584f913aa9c142fb852a47eddbe01bdd6c0930c87b9d28dfa73294`;
all 76 frozen sources, 16 model hashes and libraries match. The frozen engine,
teacher, shadow, export and mapping files are unchanged in this range.

The three excluded mutants are disclosed as equivalent under the neighboring
guards; they are not included in the 29 caught mutants. Archived 29/29 label,
29/29 runner and 4/4 diagnostic catches were inspected, not rerun in full.
Passing mutation tests does not establish exhaustive fault coverage, as the
persisted-record controls demonstrate.

## Saved artifacts and limits

As-of **2026-11-20 15:12:19 UTC** for completed independent artifact checks.
Evidence: [metadata](metadata.json), [records](records.json),
[diagnostics](diagnostics.json), [controls](controls.json),
[audit implementation](verify.py).

| Population/check | Independent result |
|---|---|
| Census | All 129,868 rows parsed; unique keys; decision grid and outcome exclusions checked; counts agree. 64 worlds, 27,417 ok candidates. |
| Manifest/selection | A, B and committed manifest/candidate bytes equal; independent round-robin selection reproduces all 128 rows. Manifest SHA `71a406caffe4f1fd83951e71208edb1eb806b48735e8d027c4273942b07bfc4a`. |
| Persisted pairs | All 128 A records equal committed copies; all 128 B hashes match the committed hash list. A/B agree after excluding declared runtime fields. Candidate, executed identity, arm arithmetic, teacher counts and CSVs reconcile. |
| Binding/visibility | 40,374 binding checks, zero bad; 5,875 hidden-state checks and 5,445 independent-rebuild comparisons, zero mismatches; 5,211 delivered-state positive changes. These are verified saved counters, not newly replayed tapes. |
| Real-row controls | Archived stale-binding controls flag 383/386 and 377/378 callbacks; four arm-order reversals and four Oct23 single-branch comparisons pass. Inspected, not rerun. |
| Receipt predicates | A/B/C/E/F remain true on the saved evidence; D remains false. The independent arithmetic check strengthens C beyond its stored flags. |
| Value diagnostic | 28 cells reconcile old numerators to corrected coherent denominators; four all-row means recomputed from actual training arrays. No prediction refit/recompute needed. |

**Only four of 128 Y values are nonzero, from two original opportunities.**
Three are the Oct24 bid original 0, KEEP at 100 ms, across neutral/joint-keep/
joint-chase sources: each -182.6433 cents. The other is Oct25 bid original 2,
neutral POST at zero: -45 cents. There are no positive Y values here. The three
source rows are not three independent opportunities.

The selected labels cover only **eight worlds and one market per day**:
Oct24 t0 is 00:00:05–00:01:25 UTC; Oct25 is 00:00:05–00:01:00 UTC. This is a
valid integration sample and an inadequate full-day learnability sample.
Neither the zero fraction nor absence of profitable actions should select a
different continuation, target, threshold or model from these observations.

The 6.06185 seconds/pair mean and 8.96296 maximum reproduce. The 9,139.74 /
12,708.46 core-hour exhaustive projections are correct **prefix extrapolation
arithmetic**, not measured full-day runtimes. Market/time and state coverage
are too narrow to treat the ratio as a precise forecast.

All market days here are consumed: Oct23 teacher training/diagnostics and
Oct24–25 candidate-training integration. There is no new untouched validation,
candidate-model skill estimate, strategy return or profitability claim.
Real arm fill/decision traces are hashed, not included in the saved records;
the independent audit can reconcile stored cash/position/mark/Y and identities,
but cannot independently sum every real fill from those hashes. Source and
synthetic engine checks supplement that limitation; no raw-tape replay was run.

## ONE next priority: production launch gate for `r2_label_sample_v2`

Deliver a single committed sampled-label plan, corrected production acceptance,
synthetic controls and an hourly distributed integration receipt. Keep the
teacher and economic estimand; do **not** shorten END, coarsen the decision grid,
replace continuation with open-loop actions, or select only eventual fills.
Those would create different targets, not just cheaper computation.

### Frozen population and deterministic selection

1. Start from **all original shared (day, market, t0) worlds on consumed
   Oct24 and Oct25**, as in the existing day metadata: 15,352 and 14,998 worlds.
   Both sides remain coupled in the same world. Preserve missing/no-path
   worlds as statuses, without replacement after outcomes or feasibility are
   inspected. Use the existing five roll-in sources and every-fifth-tick
   rotating phase; preserve original-time POST candidates.
2. Stratify worlds by day and **UTC hour of t0**. Select `min(64, N_hour)`
   worlds in each stratum by lowest SHA256 rank, then process in canonical
   original order. Maximum **1,536 worlds/day, 3,072 total**. Use seed/domain
   `r2_label_sample_v2_20261120`, domain-separated world/row hashes and canonical
   JSON keys (integer milliseconds/IDs, explicit nulls, sorted keys). Commit
   the exact algorithm, input hashes, complete world counts, selected keys and
   probabilities before generating paired outcomes. Hash ties break by full
   canonical key. Do not use the prefix order as the sampler.
3. Within each selected world's side, retain its **neutral original-time POST**
   row if it is eligible. Uniformly hash-select **one additional eligible POST**
   row from the rest of that side's five-source sampled census, and **one KEEP**
   row from its eligible KEEP census. If the mandatory POST is unavailable,
   select one POST from the available pool. Empty pools stay explicit statuses.
   Maximum **six pairs/world, 18,432 total**. Keep zeros and both signs. No
   realized-fill, nonzero-Y, forecast-score or severe-loss selection. Report
   selected source/horizon/generation/partial-queue/context coverage; do not
   fill an absent category with outcome-selected replacements.

This is an explicit sampling amendment, replacing the integration sample's
equal-source round robin. Its target population is the existing eligible
five-source census, averaged within original/side/head, then across originals.
It is not an equal-source estimand and not an estimate of live policy occupancy.

### Inclusion weights and output contract

For a selected world in hour h, `pi_h = min(64,N_h)/N_h`. For each
original/side/head, let N be its eligible base-census row count and q the
conditional row inclusion probability. Store raw weight
**`w = 1 / (pi_h * N * q)`**.

- Mandatory neutral POST: q=1. If there are N-1 other POST rows, the one drawn
  from those has q=1/(N-1). With no mandatory row, the one POST draw has q=1/N.
- The single KEEP draw has q=1/N. No replacement and no duplicate mandatory row.
- Each selected nonempty original/head has total weight 1/pi_h. Do not
  renormalize that total to one: doing so would over-weight sparse hours.
  Any later mean-one training rescale is declared separately per fitted head.

Persist N, q, pi_h, raw weight, seed/rank, source, status, original/world IDs and
all fork/arm identities. For eventual candidate learning, also persist the
**ordered 107 causal inputs at the common fork before the forced action**, using
the existing feature/support builders unchanged, with schema/hash and causality
checks. Do not recompute features later from a different roll-in or attach an
arm's post-action state to the label. Empty/invalid populations cannot silently
disappear from the status denominator.

The raw-weight total estimates the eligible-original population by inverse
probability weighting. Tests must verify inclusion probabilities and weighted
totals on enumerated tiny populations, including sparse hours, no mandatory
POST, singleton pools, absent heads and duplicates. Use actual full-day keys
to verify deterministic membership, stratification and byte-identical resume.

### First reviewable deliverable and budget

- Fix findings 1–2 before accepted production results. Include findings 3–4
  and the CLI correction in the same implementation round. Keep changes outside
  the frozen teacher closure. A changed source/input/plan gets a new namespace;
  do not rewrite the existing receipt or call its resource gate PASS.
- Freeze the full sample, then exercise a **nested first-ranked one world per
  observed hour/day** subset: at most 48 worlds and **288 pairs**, distributed
  over both entire days. This is the first bounded deliverable, with no fit.
  Label selection is the same as for those worlds in the full sample. Pilot
  results are diagnostic, not a representative training set; production
  weights refer to the frozen full sample. No outcome-driven sample amendments.
- Run each day/partition in its own process; use at most the already-authorized
  four development slots. Checkpoint at most 32 selected worlds; enforce
  1,536 MiB never-reset process HWM and 2,048 MiB hard watchdog over loading,
  census and pairs. Keep the 120 s/pair limit. Stop/hold at a breached checkpoint;
  preserve its record and exact event disposition. Predeclare a conservative
  partition runtime bound from the measured per-world/candidate workload,
  including loading, rather than bypassing the existing pair stop.
- Acceptance must include intact-versus-corrupt saved records, stale source and
  feature-input refusals, chronological arm binding/reversal/parity, exact
  statuses/populations/weights/feature order, numerical ledgers and Y, deterministic
  interruption/resume, complete input identity, zero unexplained causal/binding
  failures, and an accepted-output manifest. Preserve failed attempts. Verify
  any exact-computation speedup against both synthetic states and this distributed
  subset; caching by time/side alone is insufficient for branch-dependent inputs.
- At the prefix rate, 18,432 pairs cost about **31 core-hours**, plus about
  **4.5 core-hours** of census extrapolated from the saved A census and extra
  loading/checks. Use **40 core-hours as the initial planning ceiling**, not a
  promised runtime. Measure the distributed pilot before launching the remaining
  sample; report predicted total, process peaks and throughput by day/hour.
  If it exceeds the ceiling, pause at a partition boundary and propose exact
  computation savings or a separately frozen smaller probability sample. Do
  not choose the smaller sample based on Y or silently change the continuation.

Return this launch-gate receipt for review before the remaining production
sample. After production, the next acceptance review should report unique
originals, weight concentration/effective sample size, zero/positive/negative
Y by side/head/source/time, effect concentration and out-of-support states.
Do not infer adequate prediction signal from label construction passing. Any
candidate fit and forward evaluation remain subsequent stages of the pipeline.

## Checks executed and change scope

All six checks ran sequentially through **PF_SLOT=codex**, the repository's
`run_light.sh`, and absolute script paths against pinned source:

| Check | Result | Wall seconds | Peak MiB |
|---|---|---:|---:|
| Pinned label selftest | PASS: 320 stateless forks/116 ok pairs; 24 bound pairs; stale/reused falsifiers; nine coverage states | 143.00 | 260 |
| Pinned runner v4 selftest | PASS: resume/interruption, held stops, watchdog and source/input refusals | 32.83 | 396 |
| Independent metadata/selection audit | PASS | 2.11 | 317 |
| Independent persisted-record audit | PASS | 1.02 | 199 |
| diag2 selftest and saved-array audit | PASS | 0.98 | 252 |
| Production-admission and lifetime reproductions | PASS: the described defects reproduced; intact record positive control passes | 1.44 | 224 |

Total **181.38 s**, maximum **396 MiB**. Logs are beside this report. No full
mutation rerun (the label no-cap mutant alone was about 70 minutes), model fit,
new real-data label generation, full tape experiment or untouched-day access.
No implementation, strategy, controller or development-test edits. This commit
contains only review evidence and new review-owned state prefixes, preserving
the existing unstaged edits. No push or external message.
