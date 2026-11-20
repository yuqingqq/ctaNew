# R2-B: the full Oct 23 teacher-data build on 4 cores, and its acceptance (review e6b0efa4's next priority) — 2026-11-20

**USER:** "As long as the results are reliable, we can accelerate the process"; "Yea 4 cores to make it faster".

**Review e6b0efa4's next priority:** "Finish and accept the full Oct23 teacher-data build and its support assessment, before any pi0-50 fit."
The full-day receipt must pass its six output checks, below.

**Consumed data only:** Oct 23, the teacher day. **No model is fitted or read.** The target, J, the features, the manifest sampling, the audit
rule (`r2_full_causal_v1`), the training population and weights are unchanged.

## What ran

| Step | Result |
|---|---|
| **Sharded runner** ([`pf_r2_runner.py`](../../research/pf_r2_runner.py), sha256 `65293187...`) | `run(shard=(k, n))` processes only the parts with i mod n = k, loads only their markets (the day's input hours and their hashes do not depend on the markets) and keeps its own status file. Records, parts and readings do not depend on the shard that produced them. New selftest: two shards run in turn each touch only their parts, and together equal the unsharded run. Two new mutants (a shard running every part; a shard running the others' parts) |
| **Four research slots** ([`run_light.sh`](../../research/run_light.sh)) | Three more single-core Claude slots: claude2 (core 10), claude3 (core 13), claude4 (core 14), beside claude (core 9). The process cap rose from 2 to 5, so Codex's slot stays free. Per USER approval |
| **The gated launch** ([launch5.sh](launch5.sh)) | Every gate is a computed predicate; a failed one stops the chain:<br>**Gate 1:** the shadow suite of c8d8fc8d, **56 / 56** caught ([log](../r2a_launch_gate_20261120/logs/shadow_mutants_final.log)).<br>**Gate 2:** the sharded runner's selftest and mutation suite, **20 / 20** ([log](logs/runner_mutants.log)).<br>**Gate 3:** the frozen 256 as 4 parallel shards reproduces gate2A ([check](shard_check.py), [log](logs/shard4_check.log), [checkpoint](shard4_checkpoint/)):<br>• parts byte-identical, records equal except timing;<br>• the same support and readings;<br>• plans differing only in the runner's own hash;<br>• shard peaks 1,086–1,281 MiB.<br>**Then:** the full plan, 4 shards, the finalize preflight ([preflight.py](preflight.py)) and finalize |
| **The full day** | **Plan** `24916112...`: all 30,466 originals in 477 partitions (476 of 64, the last of 2); loading peak 1,289 MiB, 77 s.<br>**Four shards**, 04:35 to 09:43 UTC: 5.1 h, against a one-core estimate of 20.3 h. Measured peaks **1,124 / 1,421 / 1,265 / 1,210 MiB**. The 64-original parts took 112.6–171.5 s, the 2-original part 4.9 s.<br>**Finalize preflight:** equal (the finalizer's actual sources = the plan's).<br>**Finalize:** COMPLETE from the records, peak 510 MiB, 8 s |

## Review e6b0efa4's findings

| Finding | For this build | The fix |
|---|---|---|
| **P2, nonblocking:** a resource-stop latch can disappear on resume, and the watchdog does not cover loading or finalization | **Workaround:** the acceptance (6) reads every record's breach flags and every stop log, and enforces 900 s / 1,536 MiB independently. No resource stop occurred, so none was resumed | **Prepared as runner v3:**<br>• `held_breaches` carries every breached record and earlier stop status through resume and finalization until a `disposition.json` records its investigation;<br>• the watchdog spans plan loading, the whole run and finalization.<br>Selftest passes, and **25 / 25** mutants are caught. It lands in the next round: this build's plan binds the sharded runner, and the preflight correctly refuses any other finalizer |
| **P2, nonblocking:** standalone `finalize` omits the source preflight | **Workaround:** preflight.py asserted before finalize that the finalizer's actual source map equals the plan's: equal, no differing entry ([JSON](preflight_full.json)) | **v3:** finalize refuses a stale source and records `finalizer_sources` |
| Decisions | Records stay beside the parts in `data/`. This receipt commits the plan, run status, census and support JSONs, logs and `records_full.sha256`. Reload per process start accepted. The v1 audit is kept, with no switch | — |

## The acceptance ([accept_full.py](accept_full.py), declared before the outputs existed; [v2 log](logs/accept_full_v2.log), [JSON](accept_full.json))

**v2: ALL TRUE** (rc 0, peak 671 MiB, 964 s; every part read alone, the full day never concatenated):

| Check | Computed reading | Holds |
|---|---|---|
| (1) Population and identity | **Population:** 30,466 originals, 15,233 per side, receipt-verified. 477 partitions = the independent recomputation, disjoint and complete.<br>**Records:** every record accepted (plan hash, members, parquet hash), with `records_full.sha256` written and re-verified. Finalize preflight equal.<br>**Bound in the plan:** `r2_full_causal_v1`, 107 float32 inputs, the engine settings, zero costs | TRUE |
| (2) Streaming reconciliation | Every part recomputed from its rows; totals = the census (25,463,127 rows, 5,384,015 sampled ok, 60,176 weight groups summing to one); merged support = the support JSON | TRUE |
| (3) Labels and clocks | Every part; labels 5,384,015 checked, 0 failed (4,036,595 without a fill) | TRUE |
| (4) Inputs and audit | Every part. The independent `r2_full_causal_v1` selection = the record keys, 191,620 checks = the census. Retried: 83. Zero failures in every component, zero not evaluable | TRUE |
| (5) Support reported | 28 / 28 cells SUPPORTED under 200 / 20; the floors applied exactly as declared | TRUE |
| (6) Run disposition | **Records:** every one PASS with no breach; elapsed max 171 s, high-water max 1,421 MiB.<br>**Shards:** all 4 PARTS_DONE with no latch; state COMPLETE.<br>**Logs:** 17 of 17 declared, all rc 0 and ≤ 1,536 MiB; none missing; the 11 falsifiers as expected. The shadow suite was archived as passed | TRUE |

**Dry run** on the 256 four-shard copy, writing its log into `logs/` like production:
- (2)–(6) TRUE;
- (1) FALSE, as it must be: the slice is not the full population;
- a planted rc 1 in a build log, and a planted missing declared log, each turn (6) FALSE and name the log.

**v1 → v2** (FINDINGS P79). The first full run (v1, [log](logs/accept_full_v1.log), [JSON](accept_full_v1.json), [script](accept_full_v1.py)) held
(1)–(5). Check (6) FAILED on the instrument, not the build:
- **The cause:** v1 globbed every log in `logs/`, and its own output was redirected into `logs/`. It read its own unfinished log, which gets
  run_light's footer only at exit. The dry run had written its log to the scratchpad, so it never met the condition.
- **Build logs:** in v1's own output, the only failing log is its own, and its 17 build-log footers equal v2's table exactly.
- **The fix:** v2 changes only (6)'s log predicate ([diff](accept_full_v1_to_v2.diff)). [accept_logs.py](accept_logs.py) requires the
  declared build logs and excludes the acceptance's own logs (`accept*.log`, whose disposition is their exit code) by name.
- **The fix's checks:** 11 falsifier cases run inside every acceptance ([selftest log](logs/accept_logs_selftest.log)). The mutation
  check catches **9 / 9** planted errors by a check ([accept_logs_mutants.py](accept_logs_mutants.py), [log](logs/accept_logs_mutants.log)).
- **v1 against v2:** the results differ only in (6). The notes are equal, and every CSV and `records_full.sha256` is byte-identical.

## Support assessment

**All 28 coarse cells are SUPPORTED** under the unchanged floors: at least 200 distinct originals and at least 20 with fills
([cells](support_cells_full.csv)). On the 256 slice, none were.

**Declared values never observed** (each would be `unsupported_state_category`; the fallback is absent → WAIT, live → KEEP):

| Side / head | Never observed |
|---|---|
| bid POST | remaining 0 / (0,10] / (10,20]; queue unknown / 0; inventory long; other side terminal |
| ask POST | the same, with inventory short |
| bid / ask retain | remaining 0; queue unknown; other side terminal |

**Structural** (impossible by construction; verified in [`pf_branch_engine.py`](../../research/pf_branch_engine.py)):
- a POST is a new 30-share JOIN generation (lines 33, 139), so remaining 0 / (0,10] / (10,20] cannot occur;
- a bid POST needs I + B + 30 ≤ 30, i.e. inventory I ≤ −B ≤ 0, and an ask POST needs −I + A + 30 ≤ 30, i.e. I ≥ A ≥ 0 (lines 31–32);
- `status()` returns `terminal` only when g ≥ END (line 246), and every decision is before END.

**Observed absent, not proven impossible:**
- queue `unknown`, on every side and head (a retained order's price could in principle leave the five shown levels);
- queue `0` for POST (a POST joins the best shown level, whose size was positive in every observed frame).

A state outside the observed categories falls back (`unsupported_state_category`).

**Coverage, separately** (rows / distinct originals / nonzero originals; [CSV](support_coverage_full.csv)):

| Category | bid | ask |
|---|---|---|
| Delayed entry (offset > 0), POST | 294,699 / 15,044 / 10,897 | 294,631 / 15,044 / 10,938 |
| Offset-zero entry, POST | 15,044 / 15,044 / 6,359 | 15,044 / 15,044 / 6,394 |
| Later generation, POST | 11,564 / 6,065 / 2,418 | 11,505 / 6,009 / 2,362 |
| Later generation, retain | 166,587 / 11,480 / 6,437 | 165,676 / 11,421 / 6,451 |

Categories, cross-cells, presence and per side / head ranges: [CSV copies](.).

## Limits

- **Consumed data only:** Oct 23 is the teacher day. No forward, validation or profitability claim.
- **Stand-in joint sources** (R1's deterministic policies).
- **Single-order neutral targets;** gross values with zero costs; timing scenario B.
- **The causality and visibility checks run on the audit sample:** 191,620 checks, not every row.
- **Coverage is not a fit:** support says where the teacher has data. Whether it learns there is the next stage.

## Next (review e6b0efa4's sequence)

1. The supported **pi0-50** fit and freeze, under its support contract: these cells and categories, with the fallbacks where unsupported.
2. Runner v3, landing with the next round.

## Files

- **Sources at launch:** [sources_at_launch.sha256](sources_at_launch.sha256) (sha256 prefixes):

  | File | Prefix |
  |---|---|
  | sharded runner | `65293187` |
  | its harness | `32e19044` |
  | `pf_r2_shadow.py` | `962cf7b4` |
  | `run_light.sh` | `e53f7f23` |

- **The full-day plan** (sha256 `249161127c39d5a51cc7c374262073af4f38c9245ef08fb505f0ec3e646dd56b`) and the four shard statuses:
  [full_checkpoint/](full_checkpoint/). The plan is committed as `plan.json.gz`.
- **The census and support JSONs:** [census_20261023_full.json.gz](census_20261023_full.json.gz), [support_20261023_full.json](support_20261023_full.json).
- **Compression:** the plan (1.87 MB) and the census (1.07 MB) are `gzip -9 -n` (248k + 124k lines otherwise). Their decompressed sha256 are
  in [checkpoint_gz.sha256](checkpoint_gz.sha256) and equal the originals in `data/` (computed); the plan's equals the plan hash above.
- **Gate 3's plan, statuses, records, census and support:** [shard4_checkpoint/](shard4_checkpoint/), [census_20261023_shard4.json](census_20261023_shard4.json),
  [support_20261023_shard4.json](support_20261023_shard4.json).
- **The 477 completion records** stay beside the parts in `data/predict_fun/research/r2_shadow_20261119/census_20261023_full/parts`
  (1.1 GB with the parts), hashed in order in [records_full.sha256](records_full.sha256).
- **Acceptance outputs:** [accept_full.json](accept_full.json), [parts_full.csv](parts_full.csv), [statuses_full.csv](statuses_full.csv),
  `support_{cells,categories,cross_cells,coverage,presence,ranges}_full.csv`.
- **The launch chain:** [launch5.sh](launch5.sh), [preflight.py](preflight.py) ([full](preflight_full.json), [256](preflight_shard4.json)),
  [shard_check.py](shard_check.py).
- **Logs:** [logs/](logs/).
