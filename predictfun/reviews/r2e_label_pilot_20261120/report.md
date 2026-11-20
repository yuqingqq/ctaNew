# R2-E: the r2_label_sample_v2 launch gate — the bundled B1–B7 round and the 48-world distributed pilot on consumed Oct 24–25 — 2026-11-20

**USER:**
- "... continue the implementing and next priority loop";
- "As long as the results are reliable, we can accelerate the process";
- "I have let the other session to include as many issues as possible in each round, so we don't waste time to fix and validate one by
  one";
- "We need to make the run more efficient".

**Reviews answered:**
- 4393ad19's next priority: "production launch gate for `r2_label_sample_v2`".
- 0ca01d1f's bundled round: B1–B7 together, with one combined review.
- The USER's standing rules in d21e38b6 / 0a5d1dbf: batch the fixes; prioritize reliable experiments; defer minor cleanup.

**Scope:**
- **Data:** consumed Oct 24–25 only, under the frozen pi0_50_v1 teacher; no refit, no candidate fit.
- **Labels:** gross cents with zero costs; END 5,000 ms; the 50 ms grid; 58 / 37 / 100 ms.
- **What this is:** a label-only readiness test. **It measures target variation and build reliability, not prediction skill.** The pilot
  is the first-ranked world of each (day, UTC hour) stratum, 48 worlds. It is not a representative training set, and its weights refer
  to the frozen full sample.

**Commits:**
- `56e740826d480afd252c487e591678a0faed05f5`: the freeze. Code, harnesses, the immutable plan and the declared checks, committed before any
  attempt-4 pair (review 0ca01d1f step 2).
- The result commit: this receipt, the corrected v2 selftest, and the state hunks.

## Verdict

**The declared checks are all true (v4.1, [checks.json](checks.json)).** Every pilot pair labels, validates, reverses and resumes
identically, inside the memory and time rules. **The full sample is projected at 27.0 core-hours**, under the 40 core-hour ceiling.

**The labels are sparse.** 276 of 288 paired labels are exactly 0 c. The 12 nonzero ones (8 positive, 4 negative) come from 11 of 96
originals, and the 5 largest carry 88 % of all |Y|.

## B1–B7 closure matrix

| ID | What was done (final sources at the freeze `56e74082`) | Closure evidence |
|---|---|---|
| **B1** execution provenance (BLOCKING) | **The runner:** [`pf_r2_label_sample.py`](../../research/pf_r2_label_sample.py) `3ae26a7c`. **The source closure** (`closure` :133): every research file reached by import statements, function-level ones included (81 files, the selftests excluded). **The plan binds:** the closure, the freeze and libraries, the sample, and per day the keep / pull receipt and outputs, the tape offset, 34 loader files (`real_files` :170: step1 / step5 / imputation offset / the flow-price receipt behind the market metadata) and real_inputs' 25 hours and 75 tracked files. **When it is enforced:** `verify_current` (:409) runs before a part is skipped or run, and before finalize: sources, teacher (`teacher_assets`), feature schema, recomputed input identity (`identity_now` :205, no arrays). After loading, real_inputs' own hours / tracked must equal the plan's. **What a change does:** it needs a new namespace; a plan is immutable, and its directory refuses a stop history | Selftest block 5. **Refused before a done part is skipped:** a stale source, changed loader files (with an inputs function that refuses to be read), changed tracked hours, a changed teacher, a changed feature schema, a changed tape offset, a changed plan (its records bind the old plan's sha). **Refused after loading:** real_inputs' own tracked hashes, a member world missing from the day, a changed keep / pull output. **On real data:** attempt 3's A4 was refused at promotion when `pf_r2_labels.py` changed mid-run (FINDINGS P87) |
| **B2** record and finalization integrity (BLOCKING) | One validator, `validate` (:475), shared by resume, promotion (`run` :590) and acceptance (`finalize` :633). **Bytes and records:** part / pool bytes against the record (checked before any byte is decoded, `accepted_part` :576); the record's plan / part / sources / inputs / counts / breach. **Membership:** every member world exactly once; the picks recomputed from the stored pool (key, N, q, w); world statuses complete. **Per ok pair** (`_pair_bad` :438, `_ledger_bad` :422): finite values; the context = the pick; Y = R_a − R_b; R = cash + position × mark; cash and position recomputed from the stored fills, every generation's fills ≤ its size, the pre-fork fills equal in both arms (reconcile_arm's 1e-6 / 1e-9); binding, causal, parity; the 107 inputs and their hash. **Publication:** held on an undisposed stop; the partitions must cover the run set once; output = `accepted.json` | Selftest block 4: **38 corruptions refused**, including the review's four (Y + 1, another fork, reconciled false, binding_bad 1), each check isolated (no record, so the counts cannot mask a row check). Blocks 7–8: promotion, resume and finalize refusals. **On real data:** the four corruptions refused and the intact parts valid on all 6 A partitions; finalize accepted A and B |
| **B3** durable stop history | Runner v5 (`pf_r2_runner.py` `5778c1a6`): the watchdog archives earlier stops before it writes its own (`Sampler.poll` :98). Every plan / run / finalize / control process archives before any status writer; content-keyed events; plan-bound reasoned dispositions; HELD | Runner v5 selftest; **30 / 30** mutants ([log](logs/runner_v5/runner_v5_mutants.log)). v2 selftest block 6: two successive stops at one status file both survive, and the first one's disposition leaves the second holding; a disposition of another plan holds |
| **B4** memory / process isolation | **v2:** one process per partition (one day; ≤ 8 pilot / ≤ 32 production worlds); member worlds only (`worlds_all` :211, keys); the 2,048 MiB watchdog from the first line of every process; the never-reset high-water ≤ 1,536 MiB, ≤ 120 s a pair and the partition bound checked after the loading, every census, every pair and every world (`Guard` :307); timing in every record (`timing` :322). **v1 (the R2-D path):** census / finalize under the watchdog with the checkpoint; the last world / fork / factory released between days | v2 selftest: the high-water latches at the loading / census / world (staged readings); the pair limit; the bound. **Lifetime sentinel:** one read of the day's inputs, every earlier world dead before the next is labelled; its retained-world positive control fires. v1 selftest: the census lifetime sentinel over two days, with its positive control; the census, pair and finalize checkpoints. **Real peaks:** every plan, run, control and finalize process peaked at 1,271 MiB or less (the plans; every run ≤ 1,183 MiB), under the 1,536 MiB checkpoint |
| **B5** census CLI | `pf_r2_labels.main` (:494): census prints the manifest's own `candidate_rows` | v1 selftest: the CLI census exits 0 with printed counts = manifest = candidates file; a second census refuses |
| **B6** bounded test execution | Fail-fast selftests: v1 exits before its end-to-end block if an earlier block failed; v2 at the first failed check. One mutant per `run_light` job ([`pf_mutant_one.py`](../../research/pf_mutant_one.py), [`pf_mutant_queue.sh`](../../research/pf_mutant_queue.sh), [`pf_mutant_summary.py`](../../research/pf_mutant_summary.py)), each classified CAUGHT / ESCAPED / HARNESS_FAILURE with its duration and source hashes. A missing record, escape, crash, failed control or mixed sources fails (the summary's own selftest) | **v1: 35 / 35** ([summary](logs/mutants_v1_summary.log); 72 CPU min; no-cap now seconds, was 70 min). **v2 run 1** (frozen selftest `e9399869`): **95 / 100**. Two escapes (a lazy-import closure gap, an identity check masked by the context check) and three crashes instead of failed checks, all kept ([run 1](logs/mutants_v2_run1_summary.log)); fixed in the selftest only (`02b79ba1`; the module unchanged). **v2 run 2: 100 / 100 ([summary](logs/mutants_v2_summary.log); 88 CPU min) on the module unchanged since the freeze** |
| **B7** representative sampling, features, receipt | Plan (`make_plan` :371): 15,352 / 14,998 worlds; 24 strata a day; 1,536 selected a day (64 an hour); pilot = each stratum's first-ranked world (48), partitions of 8. Picks / weights (`picks_of` :258): mandatory neutral POST + one extra POST + one KEEP, w = 1 / (pi N q). The 107 fork inputs with hidden-copy / delivered / rebuild checks (`fork_inputs` :274) | Selftest blocks 1–2, including the enumerated inverse-probability identity. Two independent plan builds byte-identical; the freeze committed before any attempt-4 pair. The pilot readings below |

## Attempts (all kept; FINDINGS P85–P87)

| Attempt | What happened | Fix |
|---|---|---|
| 1 ([logs](logs/attempt1/)) | The first plan step ran without the guard: `real_inputs` for every selected market of both days in one process, **2,132 MiB — over the 2 GiB rule**. No label existed | Every process starts the watchdog and the checkpoint on its first line. Hours / tracked come without arrays, and every run checks them against `real_inputs` |
| 2 ([logs](logs/attempt2/), [profile](logs/attempt2/load_profile_attempt2_pilot_A_part0.json)) | Every partition latched STOP_RESOURCE at loading (1,595–1,631 MiB). Nothing was promoted, the stops were archived, and the rest was HELD. A load profile found each partition building all 15,352 worlds of its day | A partition builds only its member worlds, and a missing member refuses |
| 3 ([logs](logs/attempt3/); measurement only) | Peaks fell to 1,119–1,164 MiB. A4's promotion was refused because I edited `pf_r2_labels.py` (in the closure) while it ran: the B1 binding fired on real data. Superseded by the bundle; its labels were never read | Never edit a source a running job reads |
| 4 (this receipt) | 12 partitions plus the planted interruption, all exit 0; no stop | — |

## Attempt 4: the declared checks

The checks were declared before any attempt-4 output existed ([declaration v3](logs/declaration_v3.txt)).
- **v3** kept as declared ([r2e_checks_v3.py](r2e_checks_v3.py) → [checks_v3.json](checks_v3.json)): A–G true, **H false**. The frozen v2
  selftest caught 95 / 100 and was corrected after the freeze, so the build-file hashes differ by design.
- **v4** ([declaration v4](logs/declaration_v4.txt)): only H changed. Producers must equal the freeze; the check files must equal the
  final suite run.
- **v4.1 / v3.1** ([declaration](logs/declaration_v4_1.txt)): one instrument fix in both. The build-source file was parsed hash → file,
  so H crashed before any reading was printed.

| Check | Value | What it requires |
|---|---|---|
| A plan | true | A = B byte-identical; 15,352 / 14,998 worlds; 24 strata a day, n_h = min(64, N_h); 1,536 a day; the pilot = each stratum's first-ranked world, in partitions of ≤ 8; the declared limits; strict UTC days; 107 features |
| B runs | true | All 12 partitions of A and B exit 0 and promote; the planted interruption printed STOPPED |
| C resume identity | true | Every part's label and pool bytes A = B (B part 0 resumed after its interruption) |
| D acceptance | true | finalize accepted A and B; dataset `0d535640…` equal; 288 rows, 144 pairs a day |
| E resources | true | **Every job ≤ 1,271 MiB** (the plans; runs ≤ 1,183); every pair ≤ 120 s (max 9.6 s); every partition within its bound; no stop |
| F labels | true | Revalidated now; **0** causal mismatches in 9,715 probe checks; **0** parity mismatches in 9,183; **0** binding failures in 66,280; the delivered positive control changed the fork inputs on 95 / 96 and 96 / 96 rows a day; 288 ≤ 288 pairs |
| G controls | true | Below |
| H suites | true (v4.1) | Producers = the freeze; the v1 / v2 selftests pass; v1 35 / 35, v2 100 / 100, runner v5 30 / 30 |

## Controls on real rows (all 6 A partitions; [r2e_controls.py](r2e_controls.py), outputs in `data/.../attempt4_pilot_A/controls/`)

- **Reversal:** all 288 ok pairs rerun with the arms in reverse order are equal in status, Y, every arm's value / cash / position / mark /
  fill and decision hashes / reconciliation, and fill lists and order sizes. The rebuilt fork fingerprints equal the stored ones.
- **Stale binding:** on each partition's first ok pair, arm teachers bound to a fork copy report binding failures (387–396 of 390–397
  callbacks).
- **Persisted corruption:** Y + 1, another fork, reconciled = false and binding_bad = 1 are each refused on a real row of every partition;
  the intact parts validate with their records and bytes.

## Readings (descriptive; the pilot's own populations; no intervals; [population](population.csv), [coverage](coverage.csv), [Y](y_summary.csv), [arms](arms.csv), [weights](weights.csv))

**Population:** 48 worlds (24 a day, one per UTC hour), 96 originals, 288 picks, 288 pairs run, 288 ok. As of the records, the last is
2026-11-20 17:36 UTC.

**Y (cents)**, ok pairs:

| Day | Head | n | Y = 0 | Y > 0 (originals) | Y < 0 (originals) | Mean | Min | Max | Top-5 \|Y\| share |
|---|---|---|---|---|---|---|---|---|---|
| Oct 24 | POST | 96 | 91 | 3 (3) | 2 (2) | +0.82 | −75.0 | +105.0 | 1.00 |
| Oct 24 | KEEP | 48 | 47 | 0 | 1 (1) | −0.54 | −25.7 | 0 | 1.00 |
| Oct 25 | POST | 96 | 92 | 3 (3) | 1 (1) | +0.95 | −165.0 | +225.8 | 1.00 |
| Oct 25 | KEEP | 48 | 46 | 2 (2) | 0 | +0.18 | 0 | +6.5 | 1.00 |
| Both | Both | 288 | 276 | 8 (7) | 4 (4) | +0.53 | −165.0 | +225.8 | 0.88 |

- Medians are 0 everywhere. Under the frozen continuation, the forced action changes the 5 s outcome only when it changes fills; that
  happens in 12 of 288 pairs.
- **Arms, both days:**
  - POST / WAIT: 67 / 46 fills, 581.5 / 398.8 shares filled, 833 / 656 generations; value sums −274.7 / −444.3 c.
  - KEEP / PULL: 43 / 44 fills, 318.0 / 328.5 shares; value sums −552.2 / −535.2 c.
  - Actions are ~95 % WAIT.
  - Teacher statuses are mostly supported model decisions. The fallbacks are cap_blocked 636–1,051, near_end 125–277 and
    incoherent_forecast 0–31 per arm and head.
- **Coverage of the ok rows (absent states explicit):**
  - KEEP sources: delayed 74, joint_post_keep 9, neutral 8, joint_chase 5, **pi0 0**.
  - POST sources: neutral 96 (the mandatory row), pi0 48, delayed 47, joint_post_keep 1, **joint_chase 0**.
  - POST at decision 0 ms: 97.
  - Generation ≥ 2: 12 POST and 7 KEEP rows.
  - A partial known fill at the KEEP fork: 3.
  - Every support status is supported; every UTC hour has ok rows.
- **Weights** (raw, the full-sample design; the pilot is not the sample):

  | Day | Head | Raw weight total | Weighted mean Y | ESS | Largest weight share |
  |---|---|---|---|---|---|
  | Oct 24 | POST / KEEP | 479.75 / 479.75 | +1.12 / −0.57 c | 50.4 / 47.9 | 0.022 |
  | Oct 25 | POST / KEEP | 468.69 / 468.69 | −3.25 / +0.18 c | 50.4 / 47.9 | 0.022 |

## Cost and efficiency (the USER: "We need to make the run more efficient")

**Measured in the pilot:**
- loading: 54.8 s a partition (max 58.4);
- census: 1.7 s a world;
- pairs: 5.0 s mean (9.6 s max).
- A fit of wall = a + b × pairs + c × worlds over the 13 runs gives 74 s a partition + 4.75 s a pair + 0.79 s a world.
- **Full sample:** 96 partitions of 32 worlds, 18,432 pairs (6 a world), **27.0 core-hours** (the same without the world term). The
  conservative bound (the slowest pilot partition's seconds a world × 3,072) is 46.1. **About 7 h on the four slots.** The pilot itself
  cost 1.05 core-hours.

**Where a pair's time goes** ([profile](logs/pair_profile.txt), one world under cProfile, profile-only namespace whose labels were never
read):
- **The frozen teacher's causal probe is about two thirds of `pair_bound`:** 44.5 of 65.3 s. At every 7th decision it re-decides on a
  hidden and a delivered copy, then rebuilds the 107 inputs from raw data (`feature_rows` → `pair_scales`, about 39 s).
- The model is cheap: LightGBM prediction takes 3.9 s and the ordinary inputs 3.2 s.
- Loading is about 55 s of every partition (`real_inputs`, mostly median computations).

**Two exact, label-preserving options for the full run.** Each needs the review's go and is verified byte-identical against this pilot's
accepted labels before use:
1. **Probe rate.** Sample the causal probe at every 35th decision instead of every 7th; roughly 2× faster per pair. The pilot already
   carries the full-rate probe on all 288 pairs, but this lowers the production run's checking power, so it is a decision, not a
   cleanup.
2. **Per-day input cache.** Build each day's inputs once and bind them by their hashes; about 10 %.

**What else changed this round:**
- one mutant per run_light job, so all four cores run mutants (the shards left one or two idle under the 5-process cap);
- fail-fast selftests: the v1 no-cap mutant went from 70 min to seconds;
- a freeze before every launch: restarts cost about 1.5 h today (FINDINGS P85–P88).

## Deferred (nonblocking; recorded per the batching rule)

- The v1 `pairs` pass releases its last world / fork / factory between days. This is coded but not sentinel-tested (the synthetic world
  has one day); documented in its harness.
- v1's resume / finalize remain the historical R2-D path. Production labels use v2's shared validator only; v1 now holds publication on
  an undisposed stop.
- Runner v5's suite ran before the per-mutant tooling, so its log has no per-mutant durations. Its code is unchanged.
- Equivalent, not-mutated guards are documented in each harness header.
- Attempt 3's partial outputs and the profile-only namespace stay in `data/` (the USER deletes, never this session).
- **Production-scale pool design:** the uniform extra POST and KEEP draws never reached pi0 KEEP or joint_chase POST in the pilot. That
  follows from the pool sizes under the declared uniform design, not from a defect. If those strata matter for a candidate, the
  production sample needs a declared stratification amendment, never an outcome-driven one.

## Next

The combined review of this round (one request). Then, on the review's go:
1. the remaining production sample, up to 18,432 pairs (~27 core-hours; with the optional exact speedups above);
2. the subsequent candidate stage.

The label sparsity (about 4 % nonzero) bears directly on that stage's design and sample size.
