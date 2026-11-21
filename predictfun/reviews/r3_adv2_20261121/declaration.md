# R3 ADV2-RICH v1 — freeze 1: the evaluation label sample, declared before any Oct 26–27 label exists

Review **315979d205e7788daf69e277aa1bd6e0cd881e0e** (pinned f68fd805) named one next priority: the fixed ADV2-RICH prediction experiment
with FAIR/MECH controls (Oct 24–25 training, consumed Oct 26–27 evaluation) and a bounded own-state R4 diagnostic, as one development
round ([report](../automatic/20261121_f68fd805/report.md)). This commit is the round's first freeze. It holds the evaluation sample only.

## What this commit freezes

- **The evaluation label sample** ([pf_adv2_sample.py](../../research/pf_adv2_sample.py)): the frozen runner `pf_r2_label_sample`
  (`4cdeff0c`, R2-F freeze d3d024fd), called unchanged with days Oct 26–27. Same domain, ranks, ties, picks, weights, audit every 35,
  limits and teacher (pi0_50_v1). Up to 64 worlds per (day, UTC hour), 3,072 worlds, at most 18,432 pairs.
- **The plan:** `8619eec52bf75b0c1164d5af26d5a6d86bc9fda8d39a6cab2bfbf6b94660e100` ([copy](plan/eval_plan.json.gz),
  [sidecar](plan/eval_sample.json)). 96 partitions; 1,536 worlds per day, sampled from 15,121 Oct 26 and 14,910 Oct 27 worlds.
- **The day rule:** the runner's identity function hashes every book part of a day directory. The Oct 27 directory also holds the Oct 28
  00h part. The wrapper filters the parts by the loader's own `part_allowed` before any file opens. Its selftest checks this with a recorder
  and a positive control. The mutation check catches 19 of 19 planted errors ([summary](logs/mutants_sample_summary.log); the first run's
  4 harness failures are kept in `logs/mutants_sample_run1*`). Oct 28 is never opened. One directory listing printed its file name during
  the audit; no Oct 28 content was read.
- **Budget:** the round's ceiling is 40 core-hours. The label build stops when its projection from finished partitions passes 28 core-hours
  ([launcher](launch_r3.sh)). R2-F's measured build took 15.46.

## Declared now, implemented and frozen in freeze 2, before any fit and before any Oct 26–27 label is read

These items restate review 315979d2's contract. Freeze 2 carries the code, its selftests and mutation checks.

- **Candidate:** ADV2-RICH-v1, four direct squared-error regressors (bid/ask × POST/KEEP) on Y. Controls ADV2-FAIR-v1 and ADV2-MECH-v1:
  the same rows, labels, weights and split. Twelve heads; RICH is primary regardless of rankings. The teacher's 16 heads are separate.
  - Inputs (ordered, sha256 of the sorted-key JSON list):
    - RICH: the stored 107, `43c9598dfb3bc2475d6b98c2b1107cfa57cfed1a8644992569e32c8a4d9ff1f9`;
    - FAIR: 18 (`fair_edge`, `t_to_end`, the 16 `pf_r2_shadow.CTX`), `c7e58a5ea0415656f694ad99805481750646b973caf0d168b79488bdc1ae85fc`;
    - MECH: 40 (`pf_flow_price_common.MECHANICAL`, `t_to_end`, the 16 CTX),
      `04bb6d81e06e6960e75dc8a7e02876d740df415963fffe6685f8e81a5868b8c8`.
  - Parameters: `pf_pipeline_bench.PARAMS` with objective `regression` (sha256 `ae93f1d4e35259c5b193fe15ff5d41ba9ea9e66bfdecc3f41a21579ccbf53cf4`),
    200 rounds, no early stopping or search. Native NaN handling.
- **Training:** every ok row of the accepted R2-F dataset (`2c62fc9baf696e6cd8913c5c294dea2c9513640714ae34b413dc37ef8c0acb2e`), Oct 24–25,
  zeros and tails included. Per side/head, the raw weights `w = 1/(pi N q)` times one constant `n / sum(w)`, identical in the three fits.
  Episodes crossing a day boundary are purged as a status.
- **Candidate support (outcome-independent, from Oct 24–25 only, common to the three inputs):**
  - a valid horizon/action state;
  - observed horizon bins and the six category dimensions per side/head;
  - the `pf_r2_shadow.NUMERIC` ranges;
  - at least 200 distinct training originals per side/head.

  An unsupported or unavailable state falls back to WAIT when absent and KEEP when live, with its own reason.
- **Controller:**
  - absent → POST iff the supported prediction is > 0, else WAIT;
  - live → PULL iff it is < 0, else KEEP;
  - ties keep the current state.

  Structural, risk, pending and terminal handling stay the engine's.
- **Offline prediction diagnostic** on this commit's sample:
  - all three inputs predict the same rows;
  - results are reported by day, side, head, selected action and plan v2's fixed bins;
  - zero and Oct 24–25 weighted-mean baselines;
  - d·Y and regret, missed positive and admitted negative value, and |Y| > 30 c tails;
  - the plan v2 sign/amount screens with the 200-original / 20-nonzero-original floors, and the exact and `abs(Y) > 1e-6 c` counts.

  No intervals on two days.
- **Local action-information control:**
  - anchors are the mandatory original-time POST anchor and the sampled KEEP anchor per original/side/head;
  - strata are day/side/head/source/instrument/minute;
  - 500 fixed-seed draws (seed 20261121) with a fixed all-anchor denominator.
- **Bounded R4 diagnostic:**
  - worlds: the nested first eight selected worlds per (day, UTC hour) by this plan's world rank, at most 384;
  - policies: N0, PI0-50, RICH, FAIR, MECH, ENTRY-ONLY, RETAIN-ONLY and ENTRY-PA, joint and own-state at every 50 ms;
  - reached-state labels: at most one reached POST and one reached KEEP decision per original/side/head of the RICH run, by the minimum
    sha256 rank under domain `adv2_rich_v1_reached_20261121`; each is relabelled from its exact snapshot under the frozen pi0, at most
    1,536 pairs;
  - each episode is valued once. This is a diagnostic, not the R4 gate.
- **Order of work:**
  1. Freeze 2: code, controls and gates.
  2. Fit, then a model-hash commit.
  3. Read and evaluate the Oct 26–27 labels.
  4. Replay.
  5. One combined receipt and one review request.

  The label build only writes. Its `accepted.json` holds statuses and weight totals, and no outcome is read before step 3.
