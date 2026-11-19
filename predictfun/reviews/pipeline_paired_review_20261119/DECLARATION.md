# The four-part review of the paired run: declared before its data is read — 2026-11-19 16:14Z

**USER:** "After the run, Launch a loop: diagnosis first, if there are issues, clear the issues. After the run, we review: Validity ... Prediction ...
Controller ... Risk ...".

[`pf_pipeline_paired_review.py`](../../research/pf_pipeline_paired_review.py) is committed in this commit, before any run on data. Its docstring
declares every reading:
- **Validity:** V1–V5.
- **Prediction:** groups, gates and screens.
- **Controller:**
  - values against N0, PI0, NEU, E1 and the same-admission PULL-ALL;
  - nesting;
  - the matched null with its spread;
  - quoting;
  - first-review shares.
- **Risk:** tails, strata, windows, components, inventory, and FINDINGS C1 / C3 / C12.

The module is unchanged from here to its run. The run's `review_<day>.json` records the module's `source_sha256`.

| Item | Value |
|---|---|
| Module (sha256 prefix) | beada7650620fd3e |
| Mutation harness | bfd58ada37e7f6a6 |
| Receipt table generator [tables.py](tables.py) | b6b28d401ac9e583 |
| Run reviewed | R0 (commit dbe77f9c): paired v2 rerun from v1's models, evaluation v2 |
| Days | Oct 26 and Oct 27, consumed |
| Inputs | the paired v2 receipt and outputs; evaluation v2 JSON; benchmark v3 predictions; export v2; keep / pull and scenario day outputs |

## Checks before the run

- **Selftest:** pass ([logs/prv_selftest5.log](logs/prv_selftest5.log)).
- **Mutation suite:** **38 / 38** planted code errors caught by a check ([logs/prv_mutants2.log](logs/prv_mutants2.log)).
- **The first suite caught 37 of 38** ([kept](logs/prv_mutants2_one_escape_first_window.log)).
  - **Escape:** moving the ledger's first fill window from 100 ms to 50 ms passed. The synthetic tape has no kept fill before the first review.
  - **Fix:** a unit test of the ledger with planted fill times at 70 / 120 / 300 / 610 / 700 / 2,000 ms. It pins:
    - both window boundaries;
    - fill age from activation;
    - the cancel-pending window;
    - the four trade sources;
    - the cancellation;
    - the admissions.
  - **Escape confirmed caught:** [logs/prv_mut31.log](logs/prv_mut31.log).
- **One earlier selftest failure is kept as provenance** ([logs/prv_selftest2_export_provenance.log](logs/prv_selftest2_export_provenance.log)).
  - Falsifiers that swap the export also failed V5, because the new input-hash check correctly saw an evaluation that had read another export.
  - Those falsifiers now pair the swapped export with an evaluation copy that names it.

## Items added on the independent reviewer's findings (before this declaration)

- **P2, the practical predicates against A-PA:** labelled IMPLIED-BY-NESTING when the nesting predicate holds.
- **P2, a comparison that is not nested:** a quoting table for A against PI0, NEU and E1.
- **P2, the matched null's spread:**
  - its exact finite-population variance;
  - 1,000 seeded draws;
  - the reading MATCHED CONTROL WITH ITS SPREAD at a one-sided p of 0.05 or less.
- **P2:** the FINDINGS C1 / C3 / C12 tables per policy.
- **P3, validity hardening:**
  - V1 and V2 fail on NaN;
  - V4 counts the opportunities from two sources and counts the steps without a forecast (fallback decisions);
  - V5 compares at 1e-6 c absolute;
  - V5 also reproduces the evaluation's null mean and first-review shares;
  - V5 checks the evaluation's input hashes and source.
