# R2-E combined review and production decision — 2026-11-20

Pinned target: **714ce75978ab8dfec29c1fc41fd0dd800b982ad8**.
Range: `f96c38d01a7f27ed4fd39ddf2ddadeecf162b84b..714ce75978ab8dfec29c1fc41fd0dd800b982ad8`.
Producing freeze: **56e740826d480afd252c487e591678a0faed05f5**.
Reviewer: Codex; implementation owner: Claude aea5bfb9.
Evidence as of 2026-11-20 19:06 UTC; all line references below are to the pinned source.

## Decision

**Accept the saved 48-world / 288-pair pilot. Authorize the full declared sample as the next single development round.** Preserve the frozen teacher, target, population and weights. Include the small validation corrections below, optional speedups, bounded controls, pilot parity and the production measurement in that round; **no intermediate review after each repair**. No additional USER approval is needed within the existing four development slots and resource limits.

There is no demonstrated corruption of the delivered labels. The prior source/input and numerical-record blockers are substantially closed. One remaining acceptance gap can hide failed invariants behind exclusion statuses; it must be protected before production is called accepted. A separate boundary predicate falsely refuses legitimate zero-callback continuations. These do not justify rebuilding the accepted pilot or another fix-only review cycle. A verified all-record acceptance workaround is sufficient for the former if cheaper than refactoring.

Both proposed speedups are approved conditionally on the exact parity contract below. This review authorizes a label build, **not a candidate fit, an untouched-day evaluation or a profitability claim**.

## Consolidated actionable checklist

### 1. P2 — Exclusion statuses bypass pair validation; END is not checked — BLOCKING only for unprotected production acceptance

`predictfun/research/pf_r2_label_sample.py:545–548` calls `_pair_bad` only when the outer status is `ok`. The producer assigns statuses at `:364`, but the shared validator does not prove that assignment. On a copy of a real intact partition, changing a row to `feature_unavailable` and setting its result's `reconciled=false` is accepted. Changing an intact result's outer status to `pair_not_ok` is also accepted. These are producer-side validation controls, not evidence that saved file hashes were bypassed. A failure or unexpected exclusion could therefore disappear from the accepted training population.

Also, `_pair_bad` at `:446–448` checks kind, side, tick, settings and identity but not effective `result.end`. Changing END from 5,000 to 4,000 passes. The arm value formula uses each arm's own mark at `:452`; equality to the shared result mark is not enforced there. The latter is a static finding, while the END/status escapes are reproduced in [controls.json](controls.json).

**Same-round disposition:** validate status/reason consistency and applicable invariants for every selected outcome, including non-ok outcomes; a failed reconciliation/binding/causality assertion is a run failure, never an ordinary feature exclusion. Enforce END=5,000, the common finite mark for successful arms, and fill timestamps within the declared episode. An independent, fail-closed final acceptance audit over every production record is an acceptable bounded workaround; no need to redesign all schemas first. Keep legitimate missing-data statuses and counts. Preserve intact positives and add the three reproduced refusals.

**Why the current pilot is accepted:** every one of its 288 rows is `ok`; independently checked END, common marks, fill times and numerical accounting all agree. No affected exclusion was present. This acceptance gap does not invalidate those outputs.

### 2. P2 — A final-tick pair may legitimately have zero teacher callbacks — NONBLOCKING for pilot validity; handle inside the production round

`pf_r2_label_sample.py:464` requires `binding_checked > 0` in both arms. `pf_r2_labels.py:194–201` forces the focal action, calls the other side only if it remains to act, then runs later ticks. `pf_branch_engine.py:337–343` also omits policy calls in pending/terminal states. At the final tick, a valid forced action can leave no teacher call to check.

[boundary.py](boundary.py) uses the actual engine and arm-binding wrapper with a deterministic synthetic continuation. At k=99, side=0, both arms have zero callbacks and zero binding failures; the reconciled KEEP−PULL label is **+5 c**, because a trade arrives at 4,990 ms after the fork. The current binding predicate refuses it. Adjacent k=98 and the other-side k=99 cases pass. This is a synthetic integration control, not a measured real-market label.

The saved pilot pools contain **116 eligible bid KEEP rows at 4,950 ms** (57 Oct24, 59 Oct25). None was selected in this pilot. This identifies a reachable boundary, not a claim that all 116 have the synthetic outcome.

**Same-round disposition:** permit zero binding/probe counts only when the executed continuation has no eligible policy callback, and reconcile counters with actual calls. Keep binding checks on every callback and preserve stale/reused-arm refusal controls. Add the final-tick positive case and a deliberately skipped required-callback negative. Do not drop these draws, substitute other draws or turn their labels into zero. The unmodified runner may otherwise stop on a valid production draw; this is an operational false refusal, not evidence of bad saved labels.

### 3. P3 — Correct four interpretations in the next receipt — NONBLOCKING

- Receipt `predictfun/reviews/r2e_label_pilot_20261120/report.md:32–33`: the 12 nonzero labels arise from **10 distinct originals**, not 11. Seven positive-original and four negative-original sets overlap on one original. Use their union. Nine distinct worlds contain the effects; repeated rows are not independent opportunities.
- Receipt `:101–102`: **21** pairs have different complete fill lists; nine still have equal cash, inventory and Y=0. Twelve differ in fill economics and have nonzero Y. Distinguish path/timing/generation changes from economically different fills; zero Y does not prove identical execution paths.
- `r2e_checks.py:134–139`: `horizon_ms` bins `decision_ms`, i.e. elapsed episode time. Rename it `decision_ms` and separately report remaining horizon `5000 - decision_ms` before interpreting late-horizon support.
- Receipt `:129` / `r2e_checks.py` throughput fit: pairs=6×worlds in all 13 observations, so the three-column fit has rank two. The separate per-pair and per-world coefficients are not identifiable. The aggregate projection to the same ratio is still **26.9746 core-hours**; report that projection and the 46.1-hour slow-case sensitivity without claiming independently measured coefficients or guaranteed seven-hour completion.

The corrected counts and independent calculations are in [labels.json](labels.json). None changes Y, sampling weights or the launch decision. Keep the old receipt and supersede its readings in the production receipt.

### 4. P3 — Mutation summaries do not themselves enforce unique mutant identity — NONBLOCKING; defer if not cheap

`pf_mutant_summary.py:17–26` chooses the last JSON record; `:29–42` checks classifications and source sets but not the expected index/name inside each record. A copied successful record at another filename could inflate coverage. The receipt checker at `r2e_checks.py:109–123` consumes summary claims rather than revalidating every identity.

The review independently checked all **35 v1 and 100 v2** mutant records plus controls: one record each, expected index/name, matching source hashes, and the declared classifications. Thus the current claim is supported. Reuse that cheap identity audit with future suites, or add these checks to the summary when convenient. No mutation-suite rerun is required solely for this cleanup. Do not equate mutation coverage with an exhaustive proof of semantics.

## B1–B7 and prior findings

| Item | Review disposition |
|---|---|
| B1 source / input provenance | Closed for this delivery. The 81 producing-source hashes equal both the plan and producing freeze; teacher assets/libraries/schema and day input identities are enforced before skip/run/finalize. Independently hashed 221 unique input files (1,222,090,373 bytes), 16 models and the frozen source closure. |
| B2 persisted records / accounting / publication | Prior four corruptions are refused; A/B byte identity, population/picks/weights and all saved ledgers reconcile. Residual status/END admission gap is item 1; protect production acceptance in the same round. |
| B3 stop history | Closed for reviewed writer ordering: runner v5 archives before the watchdog writes (`pf_r2_runner.py:95–98`). The new runner's held-stop controls pass. Reviewed archived 30/30 runner result; did not rerun the whole mutation suite. |
| B4 resources / references | Pilot processes satisfy the declared bounds; highest recorded partition HWM 1,182.844 MiB, reported all-job maximum 1,271 MiB. Synthetic stage/checkpoint and lifetime controls pass. Historical v1 pairs lifetime sentinel remains deferred; production uses v2. |
| B5 census CLI | Fixed at `pf_r2_labels.py:494`; v1 receipt/control covers successful count printing. No new full census was run by this review. |
| B6 bounded mutants | Per-job tooling and preserved failures are appropriate. Final producing module is unchanged after freeze; selftest repairs are identified and final v2 selftest passes independently. Identity-audit limitation is item 4. |
| B7 distributed sample / inputs / readings | Sample construction and inverse-probability identity verified. Inputs and selected forks reconcile. Correct item 3's interpretations; preserve absent strata and the unchanged sampling design. |

The old R2-D v1 resume/finalize path is not certified as a production substitute. Its documented deferral is acceptable because v2 owns production admission. Attempts 1–3 and the failed suite/checker attempts remain visible; no acceptance is borrowed from those partial runs.

## Saved artifacts and meaning of the result

- Plan SHA-256: `a3f1e8975728800e728836c3c34574c151091991a2e5fe2903f22903223411ac`.
- Accepted A/B dataset digest: `0d5356401b5aee7ba872b3d8debd24d244cdb19989180867a1ac437fa6f8f5d9`.
- Source population: 15,352 Oct24 worlds and 14,998 Oct25 worlds; 1,536 selected/day, 64 per UTC hour. Pilot: first-ranked world/hour/day, 48 worlds, 96 originals, 288 paired rows. The pilot is nested and narrower than the full weighted estimand.
- All six A and six B parts/records checked, pools and A/B label bytes identical; checkpoint copies/control identities agree. Independent cash/position from stored six-field fills, generation quantities, common pre-fork fills, END value and Y all reconcile. Saved 288 reversal results are bound to the reviewed outputs and report equality; this review did not rerun 288 real pairs.
- 66,280 binding checks, 9,715 causal checks and 9,183 independent-input parity checks: **zero recorded failures**, consistent with the saved per-arm counters. Fork features have 107 float32 inputs with verified hashes.
- **276/288 exactly zero; eight positive, four negative; 10/96 distinct originals with nonzero Y; largest five effects account for about 88% of total absolute Y.** These are label-distribution observations on consumed Oct24–25, not model accuracy or settlement profits.
- The unselected pilot sources are present in the eligible pools: 63 pi0 KEEP rows and 68 joint_chase POST rows. Their absence among pilot picks is sampling sparsity, not demonstrated population absence. Do not redesign selection after observing Y; assess their realized full-sample support before choosing a candidate.

The freeze commit precedes the first attempt-4 launch. The final-source selftest differs from the frozen test, but the producing source does not. The retained v3 H=false is correct; v4.1 separates producing-source equality from final test identity. The v3→v4 change does not alter A–G, labels or descriptive outputs. The build-file parser correction is an instrument fix. This is legitimate superseding evidence, not a new successful forward experiment.

## ONE next priority: complete the declared production label sample and its support/readiness receipt

**Go**, in this sequence within one development round; submit one combined result review afterward:

1. Handle items 1–2, using the existing acceptance path or a documented fail-closed acceptance audit. Incorporate the cheap reporting corrections. Keep other cleanup deferred. Preserve `pi0_50_v1`, all 76 frozen teacher-source hashes, its models/features/support, END=5,000 ms, 50 ms decision grid, 58/37/100 ms timings, size/cap and explicit zero costs. No teacher or target change.
2. Implement either or both optional speedups outside the frozen closure, run the necessary synthetic controls, and freeze the amended orchestration/plan in a new namespace before new measurements. Preserve the sampling domain, selected worlds/rows and `w = 1/(pi*N*q)`. No stale-plan resume across source/input changes.
3. Reproduce the **288 accepted pilot pairs** with the final combination of enabled speedups. Compare canonical economic/decision payloads byte-for-byte: selected keys/statuses/weights, 107 float32 inputs and price, support decisions, fork/tape identity, arm fill lists and order sizes, request/decision hashes, cash/position/common mark/value and Y. Whitelist only declared provenance, timing/resource and causal-audit fields. Whole compressed records cannot be byte-identical when those audit counters change; do not exclude the economic `result` wholesale. Include the small boundary/status controls separately. One combined parity run is sufficient; do not require two full pilot reruns just because two exact speedups compose.
4. On passing those checks, proceed directly to the unchanged full sample: **up to 3,072 worlds / 18,432 pairs** on consumed Oct24–25, four already authorized development slots. Do not ask for another review between parity and launch. Validated old pilot rows may only be reused under explicit provenance/membership equivalence; do not relabel their old records as produced by a new source plan. Recomputing them as part of the new plan is also acceptable.
5. Publish one complete, independently accepted dataset and support/readiness receipt. Reconcile every planned world, picked row and status, all part hashes/identities, inclusion weights, arm accounting, actual END and marks, callback/audit denominators, and all stops. Show raw/weighted Y, sign/tails/concentration, distinct-original union counts, economical versus path-only differences, both arm inventories/fills, and coverage by day/side/head/source/elapsed and remaining horizon/generation/partial state. Preserve explicit absent categories. Report actual build/cache/parity time and peaks. No row-level intervals from two consumed days; no fit or claim of prediction skill in this deliverable.

### Decision (2): every 35th continuation audit — approved

Use the existing `causal_probe` rule at calls **1, 36, 71, ...**, with its unchanged deterministic per-call seed. This is a subset of the current calls 1, 8, 15, ... (`pf_r2_teacher.py:442–455`). Declare it explicitly in the new plan as an audit-rate change. Keep the initial sampled callback on short continuations, full binding checks on **every** callback, all fork hidden/rebuild checks, and all-record accounting. Do not alter the already distinct pi0 roll-in probe configuration, policy decisions or simulation clocks. Report actual audit denominators and state coverage, not a fixed expected ratio. Zero-call continuations use item 2's explicit logic.

The pilot retains the full-rate evidence. The production audit has less checking power; this is acceptable for the unchanged, parity-proven deterministic policy. Any causal/parity/binding mismatch is still a failure, never a tolerated sampling error. If parity fails, keep the slower verified implementation and report the failed speedup; do not change labels to obtain agreement.

### Decision (3): exact per-day input cache — approved, optional

Bind cached bytes and schema to the complete existing per-day input hashes, clocks/offsets, builder/source/library identities, market coverage and original ordering. Reuse unchanged builders, preserving dtypes, NaNs, float32 feature bytes and all timestamps. Load immutable cached inputs; do not cache mutable branches/teachers or bypass source/input preflight. A cache miss or mismatch must refuse or rebuild explicitly under the same identities, not silently substitute a different input set. Positive byte-parity and stale/wrong-day/truncated-cache refusals suffice.

Build/read the cache within the same watchdog/checkpoint rules, with isolated day processes or bounded/chunked storage. Do not materialize both full days to save loading time. Skip the cache if preserving these bounds makes it disproportionately expensive; the every-35 audit reduction alone is authorized.

### Resource and stopping contract

Retain 1,536 MiB checkpoint / 2,048 MiB hard RSS ceiling, never-reset HWM, 120 s per pair, existing load/world/partition bounds and durable held-stop dispositions. Keep the **40 core-hour planning ceiling**: measure throughput after early production checkpoints and stop/replan if completing the unchanged population would exceed it; do not silently trim the sample or select on outcomes. The pilot's 27-hour central estimate is not a guarantee, especially with 46.1 hours in its slow-case sensitivity. Speedups must be measured rather than credited in advance. No extra cores or untouched days are authorized here.

## Independent checks and limitations

All research execution used `PF_SLOT=codex /home/yuqing/ctaNew/predictfun/research/run_light.sh` with absolute script paths and imports from detached `/tmp/pf-review-714ce759`. No production implementation, training, real-pair build or untouched data was changed/run. Documentation and review evidence are the only changes.

| Check | Evidence | Wall / peak |
|---|---|---|
| Pinned v2 synthetic selftest | [selftest.log](selftest.log): two-part resume; 38 corruptions; ten identity refusals; resource/lifetime/stop/publication controls | 144.69 s / 263 MiB |
| Population, freeze, files, archived suite identities, checker revisions | [metadata.json](metadata.json), [metadata.log](metadata.log) | 6.70 s / 238 MiB |
| All saved parts, labels, accounting, weights and receipt readings | [labels.json](labels.json), [labels.log](labels.log) | 2.84 s / 240 MiB |
| Real-record intact positive and seven in-memory corruption controls; summary selftest | [controls.json](controls.json), [controls.log](controls.log) | 2.32 s / 238 MiB |
| Final-tick integration and actual pool census | [boundary.json](boundary.json), [boundary.log](boundary.log) | 0.96 s / 203 MiB |

Successful bounded checks total **157.51 seconds**, max 263 MiB. The initial independent label-audit script assumed five fill fields; the engine stores six. Its unpacking failure is retained in [labels_v1_schema_error.log](labels_v1_schema_error.log) with [verify_v1.py](verify_v1.py). Only that review-parser assumption changed in [verify.py](verify.py); the rerun passed without changing source/data. Metadata/controls name the v1 script hash; labels name the corrected script hash. This failure is not an engine defect.

The whole v1/v2/runner mutation suites and real reversal replays were not rerun: their saved artifacts and sources were checked, with independent bounded controls where useful. Passing implementation checks do not establish prediction quality. Production-scale support and whether this sparse paired target is learnable remain the next measurement's questions.
