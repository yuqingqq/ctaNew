# R2-A v2: stored-row causality, a no-drop population and the partitioned census (review d10d3698 answered) — 2026-11-19

**USER:** "Check my latest commit, next, can you ask codex research session for next priority. continue the implementing and next priority
loop".

**Review d10d3698's next priority:** "Publish an R2-A superseding acceptance receipt that closes the stored-row causality and no-drop
population checks, preserves the joint refusal outcome, and demonstrates partitioned output on the same frozen 256 originals. Include
unchanged valid labels/inputs/weights under canonical identities, firing counterexamples, complete status reconciliation and measured peak
memory. Complete the support diagnostics below in that preparation step. No full-day build or teacher fit should precede this closure."

**Supersession.** This receipt supersedes [R2-A](../r2a_shadow_census_20261119/report.md) (5432f636) in-band on five claims:
- the full-row causality claim (its check never compared the stored row);
- "14 context fields" (there are 16);
- "every other-side state" covered (terminal occurs in no sampled row);
- 312 refusals as the total (that was the entry rows only);
- the support tables (incomplete).

R2-A stays as provenance. Module: [`pf_r2_shadow.py`](../../research/pf_r2_shadow.py), sha256 `bbff8b19...`, with its
[mutation harness](../../research/pf_r2_shadow_mutants.py). **No model is fitted or read.** The target, the J architecture, the feature
calculations, the manifest sampling and every parameter are unchanged.

## The review's findings, at the code

| Finding | Holds? | Handled |
|---|---|---|
| **P2:** the "full-row" causality test never verified the stored row | Holds. `causal_rows` compared a fresh rebuild with a perturbed rebuild; only the context was compared with the row | **`rebuild_row`** rebuilds each checked row's 107 stored fields independently. It re-simulates the row's own source to its decision, then builds the market features at that grid index, `known_steps` → `order_inputs`, the `fp_` fields, `fair_edge` (from the archived fair path at the decision step) and `t_to_end`. **(i)** The stored row must equal that rebuild. **(ii)** It must also equal a rebuild in a future-only world (below). **(iii)** A same-time public input, a notified fill and the fair at the step must each change the rebuild. Declared tolerance: equality after the float32 cast (the stored dtype), NaN equal to NaN. **Planted corruptions** of the stored output (`fair_edge` from the END fair, `t_to_end` + 50, `fp_remaining` from every fill, a market field + 1) are each refused by (i). **Shared leaks** that both builders would read (the next step's fair; an engine-truth context) are each refused by (ii) |
| **P2:** a selected original can vanish when its source window is not yielded | Holds | Each partition emits `missing_source:market` for a market the source does not list and `missing_source:pair` for a listed market without a yielded window. An exact reconciliation refuses the run if any selected original is missing or appears twice, per partition and in total. Selftests: missing market, missing pair, all present, a dropped original refused, a duplicated original refused |
| **P2:** collection is not memory bounded | Holds | **Partitions** of at most 64 originals, in (market, original id) order. Each is processed, weighted, sorted, written (`part_NNNN.parquet`) and released. A manifest records membership and hashes. The support tables are computed per partition and added, so no partition's columns are kept. Selftest: one-original partitions equal one single partition (rows and support) |
| **P3:** joint POST refusals lose their outcome flag | Holds | Joint POST rows carry the engine's refusal outcome; **retain rows never carry the flag** (any source). Refusals are reconciled by source |
| **P3:** 14 context fields; every other-side state covered | Holds | **16** context fields: has_generation, generation age, ordinal, inventory, bid / ask reserved, and self / other state × 5. The terminal other-side state occurs in **no** sampled row on any side / head (disclosed below) |
| Support diagnostics incomplete | Holds | Per side / head: distinct-original and nonzero-original counts per category; side × head × horizon × category cross-cells; numeric ranges; inventory sign with a 1e-9 tolerance; delayed entry, offset-zero entry and later generations reported separately. Also all 28 declared cells (zeros included), each with its fallback (POST → WAIT, retain → KEEP), the declared values never observed (`unsupported_state_category`), and the fallback rule |

## The future-only world, and what the real-day rehearsal showed

Before the declared run, a rehearsal checked every row of the first 4 originals per side (a subset of the frozen 256; partitions of 2)
on Oct 23 ([script](rehearsal/rehearsal.py)). It found two instrument defects; neither is a census defect.

1. **A crash** ([log](logs/pr2b_rehearsal1_KeyError.log)). The rebuild called the real feature builder with only the focal side's key.
   The real builder reads both sides; the synthetic one did not. **Fix:** the rebuild passes both keys. A selftest wrapper records any
   call missing a side, and a mutant of it is caught.
2. **51 of 1,372 rows differed in the perturbed world** ([log](logs/pr2b_rehearsal2_receipt_clock_51_fail.log)). The
   [decomposition](rehearsal/decompose_perturbed.py) ([log](logs/pr2b_rehearsal2_decomposition.log)) applied each perturbation alone:

| Perturbation applied alone | Rows that differ |
|---|---:|
| The windows' future, for the market and order inputs (the export's own perturbation) | 0 |
| Unnotified fills, in the order inputs | 0 |
| The archived fair path after the decision step | 0 |
| **The engine's path, rebuilt from the perturbed windows** | **51** |

   The feature builder and `known_steps` are causal: the builder's boundary equals the last frame `known_steps` reads (`jk == jt`) on
   all 51. The path is not, and here the cause is the instrument. The export's perturbation keeps frames up to the last one *received*
   by the decision. The engine matches at *venue* time, and a frame received after the decision can precede it at the venue. Rewriting
   such a frame rewrote the venue's past. In every one of the 51, the generation was absent at the decision in the perturbed world: for
   example, a post-only refusal at activation, which the client learns at once (POST_NOTICE 0). The client's view reads the received
   frame (`E.f[E.known]`), so nothing leaked.

   **Fix: the engine's future-only world cuts on the venue clock** (`venue_windows`: the builder's own P5 / S5 / opp perturbation after
   the last frame at the venue by the decision). `venue_future` rebuilds the path from those windows and **halves every trade whose fill
   would be unnotified at the decision** (after decision − 100 ms). This checks R1's P1 property on real rows. The market and order
   inputs keep the export's receipt-clock perturbation. Falsifiers:
   - a unit test: a frame at the venue before the decision but received after it is kept, and later frames move;
   - a census with the planted engine-truth context, which the halved unnotified trades expose (stored_vs_perturbed > 0, same identity
     0).

   **Rerun** ([log](logs/pr2b_rehearsal3_venue_clock_pass.log)): **1,372 checked, 0 failures in every component.**

## The Oct 23 census on the frozen 256 originals

**Population:** Oct 23, the first 128 originals per side by id, of any status, unchanged: 256 selected (240 eligible, 14 `refused_post_only`,
2 `missing_endpoint`). The selected-id hash `4538b694c92c5b65...` equals v3's. Partitions: 4 × 64.

| Run | Source | Readings |
|---|---|---|
| [r2av2](census_20261023_r2av2.json) ([log](logs/pr2b_census_r2av2.log)) | before the draw retry | **The declared causality reading FAILED: 2 of 2,765** ([receipt checks](logs/pr2b_receipt_checks_r2av2.log): every other predicate true). Both are `perturbed tape without a path`: original 98 (ask), retain, decision 1,600 ms. [Diagnosis](rehearsal/diagnose_nopath.py) ([log](logs/pr2b_census_r2av2_nopath_diagnosis.log)): that draw (seed 95692) left the perturbed future with no valid END mark (`missing_endpoint`); the next 7 seeds all build. A draw that builds no world is an instrument gap, not a leak. The stored row equalled its rebuild and every firing control fired. **Fix:** up to 8 deterministic seeds per check. A check whose world builds in none is **not evaluable**, a counted status that is neither a pass nor a failure. The verdict is PASS only with zero failures **and** zero not-evaluable checks; otherwise NOT-EVALUABLE, which the receipt check refuses. Falsifiers: a planted always-unbuildable world must read NOT-EVALUABLE; a planted first-draw failure must read PASS with every check retried. Three mutants (no retry; an unbuildable world counted as passing; a verdict ignoring the status) are caught |
| **[r2av3](census_20261023_r2av3.json)** ([log](logs/pr2b_census_r2av3.log)) | the committed source `bbff8b19...` | Everything below |

**Readings of r2av3, each computed by the census or by a script declared before the run:**

| Reading | Result |
|---|---|
| [The declared comparison with v3](compare_v3_parts.py) ([log](logs/pr2b_compare_v3_parts_r2av3.log)) | **PASS.** The concatenated parts are in the canonical order. They equal v3 on **every column, exactly, dtypes included** (213,349 rows × 137 columns; float32 inputs), except `refused`. That column changed only on the 289 joint POST sampled ok rows, from missing to the engine's boolean outcome; joint retain rows carry none. Coarse support cells are identical; the selected-id hash is identical. (The same comparison also passes on r2av2: [log](logs/pr2b_compare_v3_parts.log).) |
| Stored-row causality, every 17th sampled ok row | **PASS: 2,765 checked, 0 failures.** Components: stored vs rebuild 0; stored vs the future-only world 0; firing 0; not evaluable 0. 2 checks needed a second seed (the two r2av2 rows) |
| Per-fill label reconciliation (q, value, severe excess, the value identity) | **PASS: 44,959 / 44,959 sampled ok rows.** 32,902 no-fill outcomes kept |
| Refusals at activation, by source | entry 312, joint_post_keep 6, joint_chase 12. The census reading equals the parquet's refused POST rows |
| [Receipt checks](receipt_checks.py) ([log](logs/pr2b_receipt_checks_r2av3.log)) | **All 9 true:** manifest (4 × 64, union = selection, no duplicate, hashes); population (256 / 256 represented; the 2 `no_path:missing_endpoint` originals have exactly one status row and no episode rows); weights (508 groups sum to 1, 0 elsewhere); inputs (91 + 16 in the export's order, float32, context on every sampled ok row); refusals; causality; labels; resources; support |
| **Memory and time** | **Measured peak 1,246 MiB** (run_light, the whole process, including the final writes), under the 1,536 MiB target and the 2,048 guard. RSS after each partition: about 308 MiB. Partitions took **327 / 353 / 304 / 253 s**, under the 900 s stop rule. Wall time 1,297 s |

**Throughput for the full-day plan** ([script](rehearsal/throughput_nocausal.py), [log](logs/pr2b_throughput_nocausal.log); the same 256,
causality off):

| Measurement | Value |
|---|---|
| Partitions, causality off | 26.0 / 26.7 / 24.3 / 23.0 s |
| Wall time, peak memory | 159 s (it includes about 60 s of fixed per-day loading), peak 1,102 MiB |
| The causality check's cost | about **0.41 s per checked row**: (1,237 − 100) s / 2,765 |
| Extrapolated to Oct 23 (30,466 originals, 477 partitions) | about **3.3 h** for the census; about **37.6 h** more for causality at every 17th row; about 41 h in all at one core. This is arithmetic, not a measured full-day run |

## Support (diagnostic at this size)

[Cells](support_cells_20261023_r2av3.csv), [categories](support_categories_20261023_r2av3.csv),
[cross-cells](support_cross_cells_20261023_r2av3.csv), [presence](support_presence_20261023_r2av3.csv),
[ranges](support_ranges_20261023_r2av3.csv), [coverage](support_coverage_20261023_r2av3.csv).

- **All 28 declared cells are INSUFFICIENT_TRAINING_SUPPORT.** At most 127 distinct originals, against the 200 floor. Each carries its
  fallback (POST → WAIT, retain → KEEP). The floor is not lowered; support is assessed on the full Oct 23 census.
- **Declared values never observed** (each would be `unsupported_state_category` at teacher time):

| Side / head | Never observed |
|---|---|
| bid POST | remaining 0 / (0,10] / (10,20]; queue unknown / 0; inventory long; other terminal; price (95,99] |
| ask POST | the same, with inventory short |
| bid / ask retain | remaining 0; queue unknown; other terminal; price (95,99] |

  Several of these are structural:
  - a POST has its full 30 shares at its decision;
  - with the 30-share cap, a 30-share bid POST requires inventory ≤ 0 (I + 30 ≤ 30), and an ask POST inventory ≥ 0;
  - the **terminal other-side state occurs in no sampled row** on any side / head.

  None is manufactured.
- **Observed ranges per side / head** (outside → `outside_training_range`):

| Field | POST | retain |
|---|---|---|
| Generation age | 0 ms (no generation yet; the pooled range would wrongly cover 0–4,950) | 100–4,950 ms |
| t_to_end | 100–5,000 ms | 50–4,900 ms |
| Inventory | bid −30–0, ask 0–30 (bounds carry ±4e-15 round-off, flat within the 1e-9 tolerance) | −30–30 |
| Generation ordinal | up to 13 | up to 13 |

- **Coverage, reported separately** (rows / distinct originals / nonzero originals):

| Category | bid POST | ask POST | bid retain | ask retain |
|---|---|---|---|---|
| Delayed entry (offset > 0) | 2,487 / 127 / 95 | 2,494 / 127 / 86 | — | — |
| Offset-zero entry | 127 / 127 / 56 | 127 / 127 / 58 | — | — |
| Later generation (≥ 2) | 101 / 49 / 20 | 100 / 50 / 20 | 1,465 / 91 / 53 | 1,408 / 96 / 50 |

## Checks

- **Selftest** (synthetic), on the committed source ([log](logs/pr2b_selftest6_final_source.log)). The earlier R2-A checks, plus:
  - the stored-row rebuild and its four planted corruptions;
  - the shared fair leak and the engine-truth context, each exposed only by the future-only world;
  - the venue-clock boundary;
  - the not-evaluable status (always-unbuildable and first-draw cases);
  - the builder's both-sides contract;
  - the population reconciliation (missing market, missing pair, dropped, duplicated);
  - one-original partitions equal to a single partition;
  - the joint refusal outcome (never on retain rows, checked on 38 joint retain rows);
  - the support cells, fallbacks, unsupported categories and inventory tolerance.
- **Mutation suite:** **46 / 46** planted code errors caught by a check ([log](logs/pr2b_mutants4.log)).
- **Kept as provenance:**
  - [The first suite](logs/pr2b_mutants1_four_unexpected.log) had **4 unexpected outcomes:**
    - a future-fair mutant survived, because both builders read the same fair; the shared-leak falsifier was added;
    - two missing-source mutants crashed the census instead of failing a check; the refusal is now a check failure;
    - a partition-reconciliation mutant survived; the duplicate falsifier was added.
  - [A suite on an intermediate source](logs/pr2b_mutants3_superseded_source.log): 39 / 39, before the rehearsal fixes.
  - The targeted reruns (`logs/targeted_*.log`).

## Limits

- **Consumed data only:** Oct 23 is the teacher day. No forward or validation claim.
- **Stand-in joint sources:** the R1 deterministic policies, not a teacher.
- **Single-order neutral targets:** never paired labels.
- **Gross values:** costs are explicit zeros, accepted by review d10d3698.
- **Timing:** scenario B only.
- **A fixed slice of 256 originals,** which cannot carry teacher support.
- **The causality check is a sample:** every 17th sampled ok row. Its engine world is our own venue-clock perturbation of the book frames
  plus halved unnotified and later trades. Trade times are not moved.
- **The full-day figures are arithmetic** from four 64-original partitions.

## Next (review d10d3698's full-day stage, after this receipt's review)

Before launch, the full Oct 23 build (all 30,466 originals, partitions ≤ 64: 477 partitions) still needs:
- a **declared causality sample** for the full day: every 17th row costs about 37.6 h of the budget;
- a **resume from committed partition manifests** without duplicating episodes;
- an **explicit stop** when a partition exceeds 15 minutes or 1.5 GiB.

After that come the support assessment, the pi0-50 fit and freeze, and candidate paired labels, all on consumed development data.

## Files

- **Module and harness hashes, plus the dependencies, receipt scripts and the r2av3 parts:** [sources.sha256](sources.sha256).
- **Census and support JSONs and the manifests** for r2av2 and r2av3: `census_*`, `support_*`, `manifest_*`. The parts (about 2 MB each)
  stay in `data/predict_fun/research/r2_shadow_20261119`, hashed in each census JSON and in sources.sha256.
- **Rehearsal and diagnostic scripts and readings:** [rehearsal/](rehearsal/).
- **Logs:** [logs/](logs/).
