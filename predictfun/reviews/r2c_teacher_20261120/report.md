# R2-C: the supported pi0-50 teacher, fitted, frozen, diagnosed and rolled out on consumed Oct 23 (review 3d7eae6a's next priority) — 2026-11-20

**USER:** "... continue the implementing and next priority loop"; "As long as the results are reliable, we can accelerate the process"; "Yea 4
cores to make it faster".

**Review 3d7eae6a's next priority:** "Implement, fit and freeze the supported pi0-50 teacher on this accepted Oct23 dataset, with one bounded
teacher-policy receipt."

**Scope:**
- **Consumed Oct 23 only**, the plan's teacher warm-up. No candidate paired labels, no tuning, no economic gate.
- **One teacher version, `pi0_50_v1`.** Every choice was declared in [`pf_r2_teacher.py`](../../research/pf_r2_teacher.py)'s header and in
  [teacher_checks.py](teacher_checks.py) before any fit or output was read.

## What was built

| Piece | Result |
|---|---|
| [`pf_r2_teacher.py`](../../research/pf_r2_teacher.py) (sha256 `c7d72d91...`) | **Extraction:** the accepted census, every sampled ok row, into per-group files.<br>**Streamed J fit:** 16 heads, the census weights, 107 inputs, 200 rounds.<br>**Freeze.**<br>**The causal teacher policy:** the inputs from `rebuild_row` on the actual branch, the support adapter, the unchanged mapping, the E1 rule.<br>**In-sample diagnostics**, **the bounded rollout** and **the causal / parity probe** |
| [`pf_r2_teacher_selftest.py`](../../research/pf_r2_teacher_selftest.py) | Synthetic only, 24 s ([log](logs/teacher_selftest.log)); see the next table |
| [`pf_r2_teacher_mutants.py`](../../research/pf_r2_teacher_mutants.py) | **44 / 44** planted errors caught by a check ([log](logs/teacher_mutants.log)). The first run caught 40 / 43 ([log](logs/teacher_mutants_run1_40of43.log); FINDINGS P80) |
| Runner v3, [`pf_r2_runner.py`](../../research/pf_r2_runner.py) (`516bf5f2`) | Review e6b0efa4's two P2s, landed now:<br>• the resource latch is held through resume and finalize until a `disposition.json` exists;<br>• the watchdog covers loading, the run and finalization;<br>• finalize runs its own source preflight.<br>Selftest pass; **25 / 25** mutants ([selftest](logs/runner_v3_selftest.log), [mutants](logs/runner_v3_mutants.log)). The R2-B census keeps its producing runner `65293187` at commit 294fd172 |
| The R2-B v1 outputs | Review 3d7eae6a's provenance note: the nine v1 files are now committed in [`../r2b_full_day_20261120/accept_v1_outputs/`](../r2b_full_day_20261120/accept_v1_outputs/README.md). They were copied 10:03:48Z, between v1's end and v2's start, and each equals its v2 namesake (`cmp`) |

**What the selftest checks:**
- **The inputs:** the census's 107, in order, with no identity field; stored exactly.
- **The construction:** the streamed fit equals the in-memory fit at ≤ 200,000 rows.
- **The weights:** whole-original weights, the filled-row subsets and the normalization constant, each to a known answer (1e-6).
- **Identity:** a changed plan, record, parquet, sampled-ok count or group is refused.
- **Support:** 10 cases. **The decision rule:** 10 cases, with signs, ties, incoherence and structural states.
- **A synthetic world, end to end:**
  - the adapter equals the census support;
  - the ledgers hold;
  - the honest teacher has 0 causal mismatches on live and absent states, and parity 0;
  - an engine-truth-remainder leak and an unreceived-frame leak are both caught;
  - the ledger and reconciliation instruments fail known-bad inputs.

## Declared choices (fixed before any fit)

| Choice | Declared | Why |
|---|---|---|
| Population | Every sampled ok row of the accepted census (all 477 parts hash-verified against the committed manifest): zero, no-fill and post-only-refusal outcomes kept; no downsampling | Review 3d7eae6a item 1 |
| Architecture | The J family of `pf_pipeline_bench.fit`: p binary on all rows, q and qm on q > 1e-12, severe on all rows; pinned parameters (sha `3826df94`), 200 rounds, no early stopping | Plan v2: "the same pinned J architecture" |
| Weights | The census weight (equal total per original × side × head), subsetted to the filled rows for q / qm, times one constant n / Σw per training set (mean 1; relative weights exact) | The pinned `lambda_l2 = 10` acts on hessian sums. At the raw scale (Σw = originals, about 1/157 of the retain rows) every leaf would shrink about 10× against the unweighted legacy fit, a parameter change in disguise. The selftest measures it: 10-round spread 3.93 normalized, 1.34 raw |
| Construction | File-backed `lgb.Sequence` (pread, 4,096-row batches) | The memory rule. Equal to an in-memory fit at ≤ 200,000 rows (150,000 and 199,999 measured equal); above that, a different deterministic bin sample (250,000 measured different; [check](construction_check.py), [logs](logs/)). All four groups exceed 200,000 rows; the q / qm subsets of POST (97,582 / 97,868) do not |
| Mapping | `pf_pipeline_bench.predict`, unchanged, on the 107 inputs: clipping, V = edge E[q] + E[q move], two-atom coherence | Review 3d7eae6a item 2 |
| Support | The contract's statuses, in order; fallback WAIT / KEEP:<br>• structural cap / near-END → WAIT;<br>• feature_unavailable;<br>• unsupported_horizon;<br>• insufficient_training_support;<br>• unsupported_state_category ('n/a' included);<br>• outside_training_range (± 1e-9; NaN outside; never clipped);<br>• incoherent_forecast | Reviews c68a072c / 3d7eae6a |
| Rule | Absent: POST iff supported, coherent and V > 0. Live: PULL iff supported, coherent and V < 0. Ties WAIT / KEEP. No timer, no severe penalty, no inventory utility. Engine timing 58 / 37 / 100 ms, cap 30, zero costs (gross) | Plan v2, E1 |

## The data and the fit

**Extraction:** 5,384,015 sampled ok rows, which equals the accepted census. The re-extraction with the final code is byte-identical to the
first, across all 96 files ([manifest](checkpoint/train_manifest.json), [log](logs/teacher_extract_v1.log)).

| Group | Rows | Filled | Fit (4 heads) | Peak RSS |
|---|---|---|---|---|
| bid POST | 327,341 | 97,582 | 40.8 s | 601 MiB |
| bid RETAIN | 2,353,692 | 573,171 | 205.9 s | **1,365 MiB** |
| ask POST | 327,264 | 97,868 | 40.4 s | 604 MiB |
| ask RETAIN | 2,375,718 | 578,799 | 208.6 s | **1,360 MiB** |

- **All 16 heads:** 200 iterations; every saved model reloads to identical predictions; rows equal the manifest
  ([fit records](checkpoint/)).
- **The retain fits' memory:** 1,360–1,365 MiB, under the 1,536 MiB guard and the 2 GiB cap, but with only about 170 MiB to spare. The
  streamed path's 200,000-row bin sample and the binned data dominate, not the raw matrix.

## The freeze

[freeze.json](checkpoint/freeze.json) (sha256 `769dff77...`, 11:48:25Z) binds:
- the plan `24916112`, the records manifest, the training manifest and the support JSON;
- the 107 ordered inputs (`a540a6f7`), the weight rule and the resolved parameters;
- LightGBM 4.6.0 / numpy 2.2.6 / pandas 3.0.1 / pyarrow 24.0.0 / Python 3.12.3;
- the 16 model hashes, the policy contract, timing and costs;
- the hashes of **all 76 research modules the freezing process imported**;
- multiplicity: **1 teacher, 0 candidates**.

`diag` and `rollout` refuse unless every hash still holds. The binding is deliberately broad: an edit to any of the 76 modules refuses
the teacher until a re-freeze. **The models** are committed as [checkpoint/models/*.txt.gz](checkpoint/models/), with decompressed hashes
in [checkpoint_gz.sha256](checkpoint_gz.sha256), equal to the freeze.

## Contract checks ([teacher_checks.py](teacher_checks.py), declared before the outputs; [checks.json](checks.json)): ALL TRUE

| Check | Reading | Holds |
|---|---|---|
| (A) Fits | 16 heads × 200 iterations; reload equal; rows equal the manifest | TRUE |
| (B) Freeze | Model hashes = the fit records = the files on disk; data, support, inputs and parameters bound; 1 teacher / 0 candidates | TRUE |
| (C) Adapter | Categories (80 keys) and cells (28), recomputed on all 5,384,015 training rows, equal the census support JSON exactly ([diag.json](checkpoint/diag.json)) | TRUE |
| (D) Rollout | 128 worlds, 127 with an episode (1 `no_path:missing_endpoint` status). Every ledger invariant true; every policy call logged. Causal checks 3,506 with 0 mismatches; the delivered control fired on 3,271; parity with an independent rebuild 3,384 / 0 mismatches | TRUE |

## In-sample diagnostics (training rows only; not skill and not gates; [CSV](checkpoint/diag_horizon.csv))

Weighted by the normalized census weights:

| Side / head | Rows | Predicted fill | Realized fill | Coherent | Incoherent rows | V predicted (c, weighted mean) | V realized |
|---|---|---|---|---|---|---|---|
| bid POST | 327,341 | 0.2991 | 0.2991 | 0.9997 | 113 | −9.17 | −9.35 |
| bid RETAIN | 2,353,692 | 0.2914 | 0.2914 | 0.9994 | 1,337 | −9.84 | −10.12 |
| ask POST | 327,264 | 0.2999 | 0.2999 | 0.9991 | 296 | −9.62 | −9.76 |
| ask RETAIN | 2,375,718 | 0.2935 | 0.2935 | 0.9982 | 4,235 | −10.49 | −10.78 |

**By horizon:**
- **Fill calibration** is close in every bin ≥ 100 ms.
- **In the shortest bin, (0, 100] ms,** the heads over-predict: fill 0.023 against 0.013, severe 1.41 against 0.00, and conditional movement
  −34.8 against −3.7 c (bid POST). This is where review 3d7eae6a found the least nonzero evidence: 39–181 nonzero originals per side and
  head.
- **Unweighted value sums** differ more for RETAIN (−20.1M against −14.8M c, bid). The fit weights each original equally, and long-lived
  orders dominate the unweighted sum.

## The bounded rollout (the frozen 256 = 128 worlds; [per side](checkpoint/per_side.csv), [worlds](checkpoint/worlds.csv), [decisions](checkpoint/decisions.csv.gz))

| Reading (declared) | bid | ask |
|---|---|---|
| Decisions: model / cap-blocked / near END | 11,929 / 365 / 91 | 11,746 / 325 / 85 |
| Actions: POST / WAIT / KEEP / PULL | 125 / 11,898 / 265 / 97 | 212 / 11,588 / 191 / 165 |
| First entry: delayed / offset 0 / none (originals) | **55** / 5 / 67 | **70** / 5 / 52 |

**Totals, both sides:**
- **Support / forecast fallbacks:** 0. Engine no-book fallbacks: 0. The declared fallback share is 0.0.
- **Later generations:** 202, in 80 originals.
- **PULLs:** 262. Post-only refusals: 28. Fills: **20**.
- **Peak RSS:** 1,234 MiB (the day's real inputs). Run time 593 s.

**The declared limitation flags:** delayed entry present on both sides (TRUE / TRUE); fallback dominant FALSE. The teacher reaches
delayed entry and later generations in its own rollout, so plan v2's degenerate-continuation stop does not apply.

**What the readout does not show:** with 20 fills in 127 episodes, nothing about value or profit. Mostly the teacher WAITs: 337 POSTs in
23,823 absent-state decisions (22,957 of them model decisions), and 262 PULLs against 456 KEEPs in 718 live decisions.

## Prior review findings

| Finding | Status |
|---|---|
| e6b0efa4 P2 ×2 (resource latch across resume / finalize; standalone finalize source preflight) | Fixed in runner v3, landed with its own controls: selftest and 25 / 25 mutants. The R2-B census keeps its producing source, plan and receipts |
| 3d7eae6a provenance note (v1 / v2 CSV equality not reproducible) | The v1 snapshots are committed beside the R2-B receipt; all nine equal their v2 namesakes |

## Limits

- **Consumed data, in-sample.** Oct 23 is the teacher's own training day; the rollout's 128 worlds are inside it. No forward,
  validation or profitability claim.
- **The rollout is a policy readout,** not an economic test: 20 fills.
- **Short horizons (≤ 100 ms) are poorly estimated,** as recorded above.
- **The streamed construction** differs from an in-memory fit above 200,000 rows (declared).
- **The causal probe samples** every 7th policy call, not every decision.

## Next (review 3d7eae6a's sequence)

After this freeze is reviewed: the Oct 24–25 candidate-label stage under the frozen `pi0_50_v1`, then Oct 26–27 consumed diagnostics. Those
days, and the previously accessed Oct 28, are not untouched validation.

## Files

- **Sources at the fit:** [sources_at_fit.sha256](sources_at_fit.sha256).
- **The launch chain:** [launch_teacher.sh](launch_teacher.sh).
- **Outputs:** [checkpoint/](checkpoint/) (freeze, fit records, training manifest, diagnostics, rollout JSON / CSVs, models gz), with
  [checkpoint_gz.sha256](checkpoint_gz.sha256).
- **Checks:** [checks.json](checks.json).
- **Logs:** [logs/](logs/).
- **The training files** (3.3 GB) stay in `data/predict_fun/research/r2_teacher_20261120/train`, hashed in the manifest. The first,
  byte-identical extraction is kept beside them as `train_v0_pre_group_guard`.
