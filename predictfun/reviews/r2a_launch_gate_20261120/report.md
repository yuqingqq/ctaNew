# R2-A full-day launch gate: a bounded, resumable runner, the three review findings closed (review 4cd7b45f answered) — 2026-11-20

**USER:** "Check my latest commit, next, can you ask codex research session for next priority. continue the implementing and next priority
loop".

**Review 4cd7b45f's next priority:** "Deliver the R2-A full-day launch gate: a genuinely bounded, resumable runner, with the three controls
above closed on synthetic fixtures and the same frozen 256-original partition artifacts. Preserve valid labels/inputs/weights and the explicit
joint-refusal correction. Demonstrate uninterrupted versus interrupted/resumed identity, checksums and aggregate readings; fail changed-input,
corrupt-part, overlap and stale-source resumes. A spy/sentinel must prove full finalization never materializes all parts. Add a stop-rule
firing fixture."

**Modules:**
- [`pf_r2_runner.py`](../../research/pf_r2_runner.py) (new), sha256 `5499248b...`.
- [`pf_r2_shadow.py`](../../research/pf_r2_shadow.py), sha256 `962cf7b4...`.
- Their mutation harnesses.

**No model is fitted or read; no full-day run was launched.** The target, J, the feature calculations, the manifest sampling and the training
population and weights are unchanged. The 256 slice's rows are byte-identical to the accepted r2av3 parts.

## The review's findings, at the code

| Finding | Holds? | Handled |
|---|---|---|
| **P2:** the final return re-materializes every part | Holds. `census` ended with `return read_parts(pdir)` | `census` now returns (directory, support, readings). **`materialize()` is the only all-part reader:** it reads its row count from the records before opening any file and refuses above 300,000 rows (the slice receipts' bound). The runner's `finalize` works from the **completion records alone**. Sentinels prove it: the runner selftest, the shadow selftest, and a re-finalization of the real gate run each replace every all-part reader with a sentinel that raises. The receipt is written last |
| **P2:** a venue-preserving check cannot see a context that reads an unreceived venue-past frame | Holds as a test gap; no leak. The client functions (`status`, `reserve`, `known_filled`, `view`, `structural`) read only acknowledged order fields, notified fills and the received frame | **The client-visibility control (`hidden_copy` / `observe`).** At the captured decision state, a copy perturbs in place, **with no execution re-run**: every book frame after the last received one, the unnotified own fills, every order's engine-truth remaining and queue ahead, and the public trades after the decision. The delivered public history and the own-order notices stay. The stored 107 inputs and the client's observation (both sides' views plus the context) must be unchanged. The positive control perturbs the last **received** frame, and the observation must change. Falsifiers: Codex's case as a fixture (a frame at venue 0 ms received at 500 ms; the absent POST at 0 ms and a live retain row: clean equal, a planted direct read refused on both, the delivered control fires), and a census whose context reads the engine-truth queue ahead (refused by the visibility component alone) |
| **P2:** the reconciliation compares sets, so duplicates within a partition pass | Holds | **`process_partition`** refuses a source pair yielded twice. It requires each original to carry exactly one status row **or** a set of episode rows (never both, never two status rows), unique canonical candidate identities, and represented originals equal to the partition's. Controls: a repeated pair, a repeated status row and a repeated episode row within a partition (each refused with its own message), plus the earlier dropped and cross-partition cases |

**Review 4cd7b45f's full-day decisions, adopted as declared:**
- **The audit sample `r2_full_causal_v1`:** per original × side × source × head stratum of n sampled ok rows, the min(n, max(1, ceil(n / 64))) rows with the smallest SHA-256 of the canonical key. The key is compact UTF-8 JSON: version, day, instrument, original, side, source, head, offset or null, generation, decision. Each check records its key, seeds and draw statuses.
- **The resume contract.**
- **The stop rule:** partition > 900 s or the high-water RSS > 1,536 MiB latches STOP_RESOURCE after a valid checkpoint; a watchdog enforces 2,048 MiB during work.

## The runner ([`pf_r2_runner.py`](../../research/pf_r2_runner.py))

| Stage | What it does |
|---|---|
| Plan (immutable) | Binds content only:<br>• every selected id and its hash, and the partition memberships;<br>• the source hashes of the shadow module actually used, the runner and 7 dependencies;<br>• the keep / pull receipt and output hashes, the input hours and their file hashes, the listing and the offset;<br>• the engine settings, both sampling versions, the ordered 107-input schema and dtype, the costs and the limits.<br>A different plan for a directory is refused |
| Run | Each pending part is processed, then written to a temporary file. It is validated: the parquet metadata's row count must equal the frame's, then the file is hashed. It is promoted by an atomic rename. Then its completion record is written atomically: plan hash, members, file hash, row / key / status / weight counts, support contribution, label and causal readings with one audit record per check, elapsed s and memory. **The record is the only completion marker** |
| Resume | A record is accepted only if it binds this plan's hash, the plan's members for the part and the actual file's hash. Anything else refuses the run; a conflict is never overwritten. A failed part keeps its failure record and is never rerun. A stale source or a changed input is refused before any work |
| Stop rule | Partition elapsed > 900 s, or the process high-water RSS > 1,536 MiB, latches STOP_RESOURCE: the part's record keeps the breach and no next part starts. The high-water is `VmHWM`, never reset, because resetting it would also reset run_light's `/usr/bin/time` peak; so a peak that recedes still counts. It is also checked after loading. A sampler records each part's own peak; as a watchdog it writes STOP_HARD_RSS and exits (3) above 2,048 MiB during work |
| Finalize | From the records: every part accepted and its file re-hashed, the population exactly once, readings and support summed. **COMPLETE only** if every part passed, the label and causal verdicts pass and finalization stayed within the limit. Otherwise FAILED or STOP_RESOURCE |

## Checks (synthetic)

- **Runner selftest** ([log](logs/runner_selftest_final.log)):
  - the watchdog first (a breach and a quiet control);
  - the plan binds content (two tags give the same plan);
  - uninterrupted = interrupted + resumed = an interrupted write + resumed;
  - refused resumes: a corrupt part, an overlapping record, another plan's record, a stale source, a changed input, a different plan, finalize with a record missing;
  - finalize refusing a part corrupted after the run, without publishing;
  - the stop rule: time, a receding peak above the limit, a high-water above the limit before the first part;
  - the watchdog end to end in a subprocess;
  - the sentinel and the bound;
  - failure records kept, never COMPLETE, never rerun;
  - an unbuildable future world reading NOT-EVALUABLE / FAILED;
  - audit keys independent of the partitioning;
  - a short write refused before promotion.
- **Shadow selftest** ([log](logs/shadow_selftest2.log)):
  - the earlier checks;
  - the visibility fixture and census;
  - the multiplicity controls;
  - the audit-sample key known answer, the per-stratum count and smallest hashes, and order invariance;
  - the sentinel.
- **Mutation suites:**
  - **Runner: 18 / 18** planted code errors caught by a check ([log](logs/runner_mutants_final.log)).
  - **Shadow:** every new and retargeted mutant (**18 / 18**) caught in targeted runs ([logs](logs/shadow_targeted/)). The full suite (56 mutants) was **still running at commit time**; its log lands with the next commit. The other 38 target code whose text is unchanged and passed the last full suite (46 / 46, R2-A v2).
- **Kept as provenance:**
  - [the first shadow selftest](logs/shadow_selftest1_KeyError_status_only_partition.log): the uniqueness guard on a status-only partition;
  - two runner suites with 2 unexpected outcomes each ([1](logs/runner_mutants1_two_unexpected.log), [2](logs/runner_mutants2_two_unexpected.log)), and their targeted reruns;
  - the [targeted shadow mutants](logs/shadow_targeted/), all caught (FINDINGS P78).

## The gate on the frozen 256 originals (consumed Oct 23)

**Population:** the first 128 originals per side, of any status (selected-id hash `4538b694...`, as in r2av3). Partitions 4 × 64, causal
sample `r2_full_causal_v1`. Driver [gate_run.py](gate_run.py) and checks [gate_checks.py](gate_checks.py) were declared before the runs.

**First run (label `gate`), kept as provenance:** 7 of 8 declared checks held. **(2) A = B FAILED as declared**
([log](logs/gate_checks_FAIL_plan_binds_tag.log)). The two plans differed in exactly one field, `tag` (`_gateA` / `_gateB`, the output
location), so every record differed only in its plan hash; the parts, support and readings were identical. **Fix:** the plan binds content
only, and finalize derives the receipt names from the directory. A selftest check (two tags, the same plan) and a mutant were added.

**Gate2 (the accepted demonstration on the final source)** ([checks log](logs/gate2_checks.log)): **ALL TRUE.**

| Check | Result |
|---|---|
| (1) States | A COMPLETE; B1 STOPPED after 2 records (the planted interruption); B resumed in a new process: COMPLETE |
| (2) **A = B** | The same plan hash `adf34d1e...`. All 4 part sha256 identical; every completion record identical except elapsed / memory / time stamps; the same support JSON; the same readings |
| (3) **= the accepted r2av3** | A's 4 parts are **frame-equal and byte-identical** to the r2av3 parts: valid labels, inputs, weights and the joint-refusal correction preserved |
| (4) **The audit sample** | An **independent** implementation of `r2_full_causal_v1` (its own key encoding and hashing) on the r2av3 rows selects **1,598** rows, review 4cd7b45f's count, and **exactly A's audited keys** ([audit CSV](audit_gate2A.csv)) |
| (5) Readings | Causality **PASS: 1,598 checked, every component 0**: stored vs rebuild, stored vs future-only world, stored vs visibility, firing, not evaluable. 2 checks needed a second seed. Labels **44,959 / 0** (32,902 no-fill). Refusals: entry 312, joint_post_keep 6, joint_chase 12. Weights: 508 groups summing to 1. Population: 256 / 256; the 2 no-path originals carry one status row each |
| (6) Resources | A's parts: 156 / 161 / 156 / 142 s (limit 900). High-water 1,180 MiB; run_light's measured peak 1,179 MiB (target 1,536; guard 2,048). B1 1,146 MiB, B2 1,138 MiB |
| (7) **Refused resumes** ([refuse JSON](gate2_refuse.json)) | Each was refused with its declared message, and B was unchanged by them (every file hashed before and after): a corrupt part, an overlapping record, another plan's record, a stale source, a changed input (a real input load with altered hashes), a different plan |
| (8) **Sentinel** | A re-finalized on a copy, with every all-part reader replaced by a raising sentinel: COMPLETE and identical readings ([JSON](gate2_refinalize.json)) |

**Throughput:** 615 s of partition time for 256 originals, i.e. 2.40 s per original including the audit checks. At one core that
extrapolates to about **20.3 h** for Oct 23's 30,466 originals, plus about 1 minute of input loading per process start. This is
arithmetic, not a full-day measurement; review 4cd7b45f's estimate was about 25 h against a 40 h ceiling.

## Open items for the full-day launch (not changed in this receipt)

1. **Record size:** each completion record is about 210 KB, mostly the audit records and support contribution. 477 parts would put
   about 100 MB in git, against the contract's "small checkpoint records". **Proposed, with no runner change:** the full day's records stay beside
   the parts in `data/`, as the parts do. The receipt commits the plan, the run status, the census and support JSONs and a `records.sha256`
   (477 hashes, as for gate2B here), which bind every record's content. A hash-bound sidecar remains possible if a reviewer requires the
   records themselves in git.
2. **Each process start reloads the day's inputs** (about 60 s and 1.1 GiB, inside the measured peaks), so a resume pays it again.
3. **The watchdog's real exit path** is tested end to end on a synthetic breach; no real run approached the hard limit.

## Limits

- **Consumed data only:** Oct 23 is the teacher day; no forward or validation claim.
- **Stand-in joint sources** (R1's deterministic policies).
- **Single-order neutral targets;** gross values with zero costs; timing scenario B.
- **The slice of 256** cannot carry teacher support: all 28 coarse cells are insufficient, as before.
- **The causality and visibility checks run on the audit sample**, not every row.

## Next (review 4cd7b45f; accelerated per the USER)

**USER, 2026-11-20:** "As long as the results are reliable, we can accelerate the process". So the full Oct 23 build **starts when the research
slot frees** (after the shadow suite), in parallel with this receipt's review:
- **Speculative:** nothing from it is accepted until the review passes without a blocking finding.
- **A blocking fix** means a rerun from scratch: the plan's source hashes refuse a stale resume.
- **Non-blocking findings** go into the next round's work.

1. The full Oct 23 build: all 30,466 originals in 477 partitions, under this runner and the decisions above.
2. The support assessment.
3. pi0-50.

All of it on consumed development data.

## Files

- **Hashes:** [sources.sha256](sources.sha256) (modules, harnesses, dependencies, gate scripts, the gate2A and r2av3 parts).
- **Checkpoints:**
  - gate2A: plan, the 4 completion records and the run status ([gate2A_checkpoint/](gate2A_checkpoint/));
  - gate2B: plan, run status and record hashes ([gate2B_checkpoint/](gate2B_checkpoint/));
  - the census and support JSONs of gate2A and gate2B;
  - the first run's census JSONs and plans ([first_run/](first_run/)).
- **The parts** (about 2 MB each) stay in `data/predict_fun/research/r2_shadow_20261119`, hashed.
- **Logs:** [logs/](logs/).
