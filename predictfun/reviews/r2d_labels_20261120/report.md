# R2-D: the arm-bound paired-label adapter and its bounded Oct 24–25 integration receipt (review 8cc1dc49's next priority) — 2026-11-20

**USER:** "... continue the implementing and next priority loop"; "As long as the results are reliable, we can accelerate the process"; "Yea 4
cores to make it faster".

**Review 8cc1dc49's next priority:** "R2-D: build the arm-bound candidate-label adapter and a bounded Oct24–25 integration receipt under the
existing frozen pi0_50_v1."

**Scope:** consumed Oct 24–25, the plan's candidate-training days. Frozen pi0_50_v1; no refit, no candidate fit, no tuning. Labels are gross
cents with zero costs, under timing scenario B.

## Review 8cc1dc49's findings

| Finding | What was done |
|---|---|
| **P1, BLOCKING:** the teacher is not bound to each executed arm | **The new adapter [`pf_r2_labels.py`](../../research/pf_r2_labels.py)**, outside the frozen closure (below). `pair_bound` mirrors `pf_branch_engine.pair` line for line. A policy factory binds a fresh teacher to each arm's own deep copy, before the forced action, before the remaining side's decision at the fork tick, and before every later tick. Every callback checks that the engine's view equals the bound branch's own view. The pi0 roll-in is bound the same way (`bound_fork`) |
| **P2, nonblocking:** a disposition matched by file name; stop history lost | **Runner v4** ([`pf_r2_runner.py`](../../research/pf_r2_runner.py) `0f56442c`). Stops are immutable events, keyed by the sha256 of the status bytes, in an append-only `stops.jsonl` archived before any process writes a status. A disposition covers only identities of its own plan, each with a reason; an older name-only disposition covers nothing. A held run writes `HELD`, not a new stop. **Suites:** selftest pass; **29 / 29** mutants, including 5 new ones ([log](logs/runner_v4_mutants.log)). The first v4 run had 2 crashes and is kept ([log](logs/runner_v4_mutants_run1_2crashed.log); FINDINGS P83) |
| **P3, nonblocking:** the value diagnostic's mixed denominator | **A superseding receipt,** [R2-C report_v2](../r2c_teacher_20261120/report_v2.md). It reports coherent weights, coherent-conditional means and the full-population realized mean ([`pf_r2_teacher_diag2.py`](../../research/pf_r2_teacher_diag2.py), outside the closure). Known-answer selftest; 4 / 4 mutants. The freeze and models are unchanged; the ">100 ms" wording is corrected there |
| Also asked | Library versions are now enforced against the freeze (`libraries_enforced`). The new orchestration sits outside the frozen 76-module closure. Label records carry explicit original / source / kind / fork / arm identities |

## The adapter and its suites

**[`pf_r2_labels.py`](../../research/pf_r2_labels.py)** (sha256 `42c40b16...`), declared in its header before any paired outcome was read:
- the contract: Y_post = R(POST) − R(WAIT); Y_keep = R(KEEP) − R(PULL); R = cash + inventory × END mark over both sides; 5,000 ms END;
  58 / 37 / 100 ms;
- the five sources: neutral, delayed, joint_post_keep, joint_chase, pi0;
- the round-robin selection: 16 per day × side × kind;
- the immutable manifest with every selected fork's fingerprint;
- records, resume, the stop rule and held stops.

**[Selftest](logs/labels_selftest.log)** (synthetic, 133 s):
- parity with `pf_branch_engine.pair` on 320 forks (116 ok pairs) under 5 stateless policies;
- 24 teacher-bound pairs, with binding, causal and reversal checks;
- the stale-bound and reused-arm falsifiers both caught;
- a single branch equal to the frozen teacher;
- fork identity: `bound_fork` = `fork_state`, and the pi0 roll-in = its census state;
- reconciliation falsifiers, each of which only one guard can catch;
- the selection order;
- end to end: an interrupted + resumed build equals the uninterrupted one; a tampered fork or candidates file is refused; a stop is held
  until its identity-bound disposition;
- the pins (freeze sha, libraries);
- computed coverage of all 9 required states: post / keep forks on both sides, a delayed post, a later generation, a partial known fill
  at a keep fork, a fill after a cancel request, and the remaining side deciding at the fork tick.

**[Mutation suite](logs/mutants/)** (29 planted errors, 2 shards): **29 / 29 caught by a check** (shard 0: 15 / 15, shard 1: 14 / 14), on
the adapter sources `42c40b16` / `3b643eb0` / `1a5c2f96`. The no-cap mutant labels every candidate, so shard 0 took 70 min.
- **Earlier runs, kept as provenance** (FINDINGS P82):
  - run 1 (3 shards; stopped): two refusals escaped as crashes, and one equivalent mutant passed;
  - run 2 (pre-streaming; stopped when the streaming fix changed the module).
- **Three equivalent mutants** are documented and excluded in the harness header.

## The bounded build (consumed Oct 24–25; [launch_r2d.sh](launch_r2d.sh), sources [sources_at_build.sha256](sources_at_build.sha256))

**Attempt 1 (kept: `logs/attempt1/`, `data/.../r2_labels_20261120_attempt1_manifest_in_memory`).**
- **What happened:** the manifest held all 129,868 candidate rows (70 MB JSON). The census peaked at 1,597 MiB, and B's resume pass
  latched **STOP_RESOURCE** at its first Oct 25 pair (1,553 MiB). The stop rule worked: it failed closed, and the record kept its breach
  flag. Build A was also refused once by the 5-process cap.
- **The fix:** candidates are streamed world by world to a deterministic gzip (`candidates.jsonl.gz`, hash in the manifest); only the ok
  rows stay in memory; the label pass never loads the candidates.

**The accepted builds** (A uninterrupted; B stopped after 40 pairs and resumed):
- **The prefix:** 2 days × 32 shared (market, t0) worlds, all 64 with an episode.
- **129,868 candidates** (manifest `71a406ca...`, candidates `bbec686a...`), by status: ok 27,417; unsampled 88,912; cap-blocked 6,606;
  post-pending 3,439; cancel-pending 3,259; near-END 235.
- **Rows / ok by source:**

  | Source | Oct 24 | Oct 25 |
  |---|---|---|
  | neutral | 2,652 / 590 | 4,707 / 1,021 |
  | delayed | 33,352 / 7,985 | 50,757 / 11,812 |
  | joint_post_keep | 6,400 / 786 | 6,400 / 1,026 |
  | joint_chase | 6,400 / 721 | 6,400 / 992 |
  | pi0 | 6,400 / 1,232 | 6,400 / 1,252 |

- **Selected: 128 pairs**, 16 per day × side × kind. Per cell that is 4 neutral and 3 each from delayed, joint_post_keep, joint_chase and
  pi0. Every selected pi0 fork equals its census rollout state (24 / 24).

**The census CLI defect.** The census CLI's final print still referenced the removed manifest field `candidates`, so it exited 1 **after**
writing its outputs. The module is unchanged here, so the suite's certification holds. Each census is instead verified independently by
[verify_census.py](verify_census.py) (rows, counts, outcome-free, all selected ok: [A](logs/r2d_A_census_verify.log),
[B](logs/r2d_B_census_verify.log)). The print is fixed next round.

## Contract checks ([r2d_checks.py](r2d_checks.py), declared before the outputs; [checks.json](checks.json)): ALL TRUE EXCEPT (D)

| Check | Reading | Holds |
|---|---|---|
| (A) Manifest | A and B have byte-identical manifests and candidate files. 64 prefix worlds; ≤ 16 per cell and 128 in all; every selected candidate ok; pi0 forks = census 24 / 24; no outcome field in any of the 129,868 candidate rows | TRUE |
| (B) Persisted / resumed identity | Every selected pair has a valid record in A and B. B's records equal A's except timing. B's first pass STOPPED and its resume reached PAIRS_DONE | TRUE |
| (C) Labels | 128 / 128 ok. Every pair reconciled (both arms' ledgers, Y = R_a − R_b, identical pre-fork history). Every arm teacher's binding was checked with 0 failures. The arm causal probes had 0 mismatches and 0 parity mismatches | TRUE |
| (D) Resources | Every pair ≤ 120 s (max 8.96 s); every pair's high-water ≤ 1,536 MiB; no undisposed stop. **But two multi-day processes exceeded the 1,536 MiB checkpoint:** B's census (1,652 MiB; A's identical census was 1,455) and the three-day controls run (1,631). All are under the 2 GiB rule. Pairs passes: A 1,440, B 1,319 / 1,413 | **FALSE** |
| (E) Real-row controls ([controls.json](controls.json)) | **Stale binding** caught on each day's first real pair (383 / 386 and 377 / 378 callbacks). **Reversal** equal on 2 real pairs per day, with its Y + 1 falsifier unequal. **Single branch** = the R2-C rollout's decisions on 4 Oct 23 episodes. v1 compared a live dict with its JSON round trip (integer keys became strings) and is kept ([v1](controls_v1.json)) | TRUE |
| (F) Suite on the build's sources | The adapter's mutation suite passed (29 caught = the harness's 29) on exactly the adapter sources the builds recorded. v1 of this check compared with a stale literal 26, written before the candidates file added 3 mutants, and is kept ([v1](checks_v1_stale_count.json)) | TRUE |

**On (D):** two-day processes peak 1,455–1,652 MiB for identical work, varying from run to run (glibc heap retention across days; the
system Arrow pool and `MALLOC_ARENA_MAX=2` are already set). **The production fix:** one process per day (or per partition) under the stop
rule, as the R2-B runner does. That fix is not applied here.

## Readings (declared; integration readout, not skill or value)

- **Fallback share:** 0.0 in every arm (POST / WAIT / KEEP / PULL; [arms.csv](arms.csv)).
- **Y (cents):** most cells are all zeros. Under the pi0 continuation both arms usually behave alike: the teacher mostly WAITs, and fills
  are rare. The exceptions: Oct 24 bid KEEP, mean −34.25 (min −182.64, 16 pairs); Oct 25 bid POST, mean −2.81 (min −45). Per pair:
  [labels_summary.csv](labels_summary.csv).
- **Cost:** 6.06 s per pair (mean; max 8.96 s), including the causal probe on every 7th arm decision.
- **Projection, labelling every ok candidate:**

  | Scope | Oct 24 | Oct 25 |
  |---|---|---|
  | The 32-world prefix (ok candidates, core-hours) | 11,314 → 19.1 h | 16,103 → 27.1 h |
  | A full day (worlds; core-hours) | 15,352 → **9,140 h** | 14,998 → **12,708 h** |

  **At this cost, plan v2's every-fifth-tick training population is infeasible.** The full label production needs a declared, much smaller
  sample, or a cheaper continuation. That is a decision for review (and possibly the USER).

## Limits

- **A bounded, source-balanced integration sample** of 128 pairs, not the training population.
- **Consumed days.** No forward, validation or value claim.
- **Sparse labels** under the frozen teacher: most Y are 0.
- **Census-pass memory** over the declared checkpoint, (D).
- **The census CLI print defect,** above.

## Next

Review 8cc1dc49's sequence was this adapter, then the full Oct 24–25 candidate labels. Before production, review must settle:
- the declared label sample size, given the projection;
- one process per day;
- the CLI print fix.

## Files

- **Data:** `data/predict_fun/research/r2_labels_20261120/{A,B}` (manifests, candidates, pair records, labels.csv).
- **Logs and launch chain:** [logs/](logs/), [launch_r2d.sh](launch_r2d.sh).
- **Scripts:** [r2d_checks.py](r2d_checks.py), [r2d_controls.py](r2d_controls.py), [verify_census.py](verify_census.py).
- **Copies:** [checkpoint/](checkpoint/) (A's manifest, labels, selected rows and manifest-derived counts).
