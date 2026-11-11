# Queue study artifact versions

`report.json` is the original frozen receipt. Its verifier caught floating-point
residue on economically flat inventory. It is retained as provenance and is
superseded by `../queue_accounting_v2_20261111/report.json`, which records the
original hash, unchanged policies/fills, accounting correction and verification.

`comparison.json` uses that corrected receipt plus the separately declared
age-matched baseline in `../queue_age_control_20261111/`. Read
`../../handoffs/20261111_queue_policy_results.md` for the conclusion and limits.

Data and trained models remain under the ignored local data research directories;
no large data files are included in this branch. No live or frozen-v3 source changed.
