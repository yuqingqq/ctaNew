# Pipeline step 1: causal feature / label export on the simulator's JOIN orders — 2026-11-19

**USER:** "Can you check the new commit, proceed our pipeline model", after "We should work on the pipeline, a prediction model + value
mapping one".

## The new commit c2717dc8 checks out

The other session's prediction → value → controller contract (`pf_pipeline_contract.py`, `_check.py`; Codex) was reproduced on the claude
slot ([logs/contract_check_repro.json](logs/contract_check_repro.json)):
- the same input and receipt hashes;
- 16 synthetic tests;
- the same status counts, with 58,738 accounting identities per scenario at M0 / M50 / M250 (176,214 in all);
- a maximum discrepancy of 9.1e-13 c.

Reading the code found no defect. Its stated next step, which this receipt carries out, is "the causal feature/label exporter on the
corrected neutral identities".

## Instrument

`predictfun/research/pf_pipeline_export.py` (new; declared before any data was read). The contract's files are imported, never edited.

**Rows.** For every committed state of the keep / pull run (one 30-sh JOIN order at the best, END = +5 s):
- one POST row, whose affected fills are all of the order's fills;
- one RETAIN row per live 50 ms decision from +100 ms, whose affected fills are those a pull there could still remove (at or after its
  landing).

Exclusions keep the keep / pull run's statuses, plus `off_grid` and `feature_unavailable`; rows are never dropped.

**Inputs.** The pipeline's own builders, not re-implemented:
- the 71 market features (`pf_flow_price_data.feature_rows`);
- the 12 order inputs and 6 extras (`known_state` / `order_inputs`), with the contract's horizon to the common endpoint (exposure 5 s, not the
  old 1 s), computed vectorised and checked against `known_state`;
- the fair edge (A+0's input) and the time to END.

**Labels**, at M50, M0 and M250:
- the affected quantity q;
- the value at the END mark;
- qm = the contract's E[q · move] term, from the causal mid;
- the severe excess, no-fill, and whether the order is still live at that clock;
- at M50, q by trade-anchor source.

| Gate | Result |
|---|---|
| Selftest (synthetic tape, causal feature stub; `known_steps` = `known_state` at every synthetic decision; a brute recomputation of every label and input; refusals by their own control) | pass |
| Mutation suite (every mutant caught by a check) | **26 / 26** ([logs/px_mutants.log](logs/px_mutants.log)) |
| Smoke run on 2 real Oct 23 markets | pass |
| Every day: X1-X6 | pass on all five days |

**The first mutation run found two selftest defects** ([logs/px_mutants_run1_harness_defects.log](logs/px_mutants_run1_harness_defects.log)),
with 5 of 26 not caught by a check:
- The final "published day" check caught only ValueError and ran even after a refused first export. Four mutants that the checks did
  refuse escaped as crashes.
- The leaky feature stub ignored same-time information, so it was refused by the positive control instead of the future perturbation. The
  X5 message also named both conditions whatever failed.

Both were fixed (the check runs only after a published export and catches every refusal; the stub responds to same-time information; the
message names only the failing condition), and the suite was rerun: 26 / 26.

**Controls** (no output for a day unless all pass):
- **X1:** every committed state has one POST row; every eligible state has one RETAIN row per live step; identities are unique.
- **X2:** POST q / value = the committed order at M50, M0 and M250.
- **X3:** for all six declared score policies, the RETAIN values at each policy's executed pull steps sum to minus the committed keep / pull
  run's improvement for that day, with the same pull count.
- **X4:** sampled orders (oid % 97 == 0) re-simulated through the kernel give the exported fills exactly.
- **X5:** on sampled windows (m % 17 == 0), features at a decision are unchanged when every later input is perturbed and change when
  same-time information changes. The vectorised known state equals `known_state` on sampled orders.
- **X6:** no sealed or forward hour is read.

## Results ([days.csv](days.csv), computed by [days.py](days.py) from the receipts)

| Day | States | RETAIN rows | of them no causal mid | Kernel / causal / known-state checks | Hours read | A+0 pulls (X3) | s | MB |
|---|---:|---:|---:|---|---|---:|---:|---:|
| Oct 23 | 30,466 | 1,328,957 | 321 | 298 / 23 / 1,177 | 22_23 .. 23_23 | 14,155 | 338 | 332 |
| Oct 24 | 30,704 | 1,439,916 | 517 | 304 / 17 / 1,191 | 23_23 .. 24_23 | 14,290 | 340 | 340 |
| Oct 25 | 29,996 | 1,491,674 | 602 | 302 / 22 / 1,187 | 24_23 .. 25_23 | 13,961 | 331 | 345 |
| Oct 26 | 30,242 | 1,270,256 | 518 | 292 / 15 / 1,151 | 25_23 .. 26_23 | 14,106 | 342 | 330 |
| Oct 27 | 29,820 | 1,404,872 | 332 | 284 / 21 / 1,116 | 26_23 .. 27_23 | 13,861 | 340 | 355 |

- No row was off the engine grid.
- Peak memory was 1.36-1.45 GiB per day.
- Each day read the previous day's 23Z hour and its own 24 hours only. **Oct 27 read nothing of Oct 28.**
- Per-day receipts: [receipts/](receipts/). Outputs: `data/predict_fun/research/pipeline_export_20261119/<day>/rows_*.parquet`.

**Provenance note.** `pf_keep_pull_scenarios.py` was edited to v6.2 (the `triggers` parameter of `outcome()`, for the next step) at 04:22:23Z,
while the export was running.
- Oct 23-24 imported v6.1 (sha256 `7b8f0a62…`, committed).
- Oct 25-27 imported v6.2 (`d2995931…`, committed with the next step).

The export uses only `fills_g` from that module, which is identical in both. Editing a source while a chain that reads it is running breaks
the program's rule. It changed nothing here, but it is recorded.

## What this is not

No model was fitted, no policy was valued and no economic number is claimed. The labels are oracle accounting for training and evaluation,
never inputs. These are consumed days, queue rule M only (C5), with no order-level validation (C10).

## Next

The chronological prediction benchmark (`pf_pipeline_bench.py`): train Oct 23-25, evaluate Oct 27, with the mapping gate. Then the learned
keep / pull rule through the scenario bank on Oct 27 (`pf_pipeline_policy.py`): the USER's one-day test of the whole pipeline.
