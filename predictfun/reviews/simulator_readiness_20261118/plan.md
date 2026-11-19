# Simulator readiness plan — P-2026-005 keep/pull simulator

USER goal (2026-11-18): **make the simulator ready for test runs that generate correct results**. A self-paced loop (session aea5bfb9)
works through this list. Each item records its status, the evidence that closes it, and its receipt. New reviews come first.

Statuses: OPEN, IN-PROGRESS, DONE (with receipt), USER (needs a decision or data only the USER can supply).

## Items

| # | Item | Status | Evidence that closes it |
|---|---|---|---|
| 1 | Review 8e63f178's P2s fixed in code: P2-1 window constituents per clock with feasibility (known v7); P2-2 exact identity / endpoint comparisons (case audit v4); P2-3 the c_hi bound typed by the answer's premise, N3 / N4 restated, the dependent lifecycle and K5 inferences evidence-only, K9 declared | DONE | suites case audit 40/40, census 17/17, exec-windows 127/127, scenarios 37/37 (every kill from a check); [receipt](../review_8e63f178_fixes_20261118/report.md) |
| 2 | Reruns on consumed data with the fixed code: case audit v3 receipt (3 legs), census v4, known v7, scenario replay v5 (Oct 23, the driver's `day` path), and compare.py R1-R4 (everything outside the fixes reproduces) | DONE | compare.py ALL true (R1-R4); readings in the receipt |
| 3 | Scenario replay meets the FINDINGS checklist items code can meet: C1 age split and C3 cancel-pending (v5, coded); C2 an explicit reading at M250; C12 the age split by trade-anchor source (WS / E2); C15 / C19 / C20 / C24 re-checked against the outputs | DONE | v6: C1 (age split, w/o young, B1a-weighted), C2 (M0 / M50 / M250 + readings), C3, C12 (by anchor source, control j); C13 met by the keep-pull base's decision-invariance control; C15 / C19 / C20 / C24 met by the outputs; C5 (rule L) and C17 (quote availability) are limits stated in the protocol, not computed; [receipt](../scenario_checklist_v6_20261118/report.md) |
| 4 | Per-cancel race study as review 8e63f178 specified: every cancel on Nov 8 (no-fill orders and retries included), request / response / match bounds / notice times kept apart, censored and unknown outcomes kept, the match-before-request evidence (K9-type) per cancel; never a cancel-failure model from notice-after-answer | DONE | v2: selftest, 25 / 25 mutants, C5 YES (86 / 86), C4 consistent; C3b NO (no match after a Canceled confirmation, K7, one-sided); [receipt](../cancel_race_study_20261118/report.md) |
| 5 | Oct 24-27 through the corrected chain (the driver's `all` phase: ledger, census-gated known ledger, scenario replay per day, pooled report) | DONE | the `all` phase on settlement times v2: every step rc 0, ledger V0 YES, scenario Z0 YES, Oct 23 reproduces v5 (R5a-d ALL true); [receipt](../scenario_checklist_v6_20261118/report.md) |
| 6 | Test-run protocol: what a run must output (checklist), what it may claim (scenario-conditional values, stated premises) and what it may not (P&L, order-level fill validity, match-vs-cancel order), the data rules and the gates | DONE | [test_run_protocol.md](test_run_protocol.md): gates, required outputs per checklist item, MAY / MAY NOT, data and padding rules |
| 7 | Order-level fill validation (FINDINGS C10: no replay number is validated at the order level; only a live run with capture can validate one) | USER | live run with capture is the USER's call |
| 8 | Untouched validation (C8 / C16: >= 5 untouched days after a freeze) | USER | the USER's choice of days and freeze |
| 9 | Found in item 5's first five-day run: settlement times v1 padded day 10-27 into the SEALED 10-28 00Z / 01Z REST hours (read 19:08-19:16Z). Fix: v2 refuses sealed / forward hours before opening, boundary legs status 5; void outputs set aside; reruns on v2 | DONE (fix) / USER (decisions) | v2 selftest + 15 / 15 mutants; compare_settlement.py S1-S3 ALL true (10-23..26 identical to v1; 18 boundary legs of 10-27 status 5); the five-day rerun clean; [receipt](../sealed_read_fix_20261118/report.md); USER: whether 10-28 stays sealed, deletion of the VOID outputs |

## Suite of record (every module on the committed sources; a kill counts only when a CHECK refused - the selftest's or run's own exit message or a named C.require control; a crash is unexpected)

| Module | Selftest | Mutation suite | Logs (suites/) |
|---|---|---|---|
| pf_settlement_times.py (v2) | pass | 15 / 15 | r1_pf_settlement_times_* |
| pf_keep_pull_exec_windows.py (known v7) | pass | 127 / 127 | r2_pf_keep_pull_exec_windows_* |
| pf_keep_pull_scenarios.py (v6.2, 2026-11-19: the policy parameter and external triggers) | pass | 49 / 49 | ../keep_pull_variants_20261119/v2/logs/sc62_* (v6.1: 46 / 46, ../keep_pull_variants_20261119/logs/sc61_*) |
| pf_keep_pull_variants.py (v2, 2026-11-19: policy comparisons; review e96321df's PULL-ALL label) | pass | 31 / 31 | ../keep_pull_variants_20261119/v2/logs/vr2_* (v1: 30 / 30) |
| pf_pipeline_export.py (v1, 2026-11-19: the pipeline's causal export) | pass | 26 / 26 | ../pipeline_export_20261119/logs/px_* - review 89caa74c P1 (POST inputs use a future arrival) OPEN: v2 next |
| pf_keep_pull.py (the committed base replay) | pass | 43 / 43 | r2_pf_keep_pull_selftest, r3_pf_keep_pull_mutants |
| pf_queue_place_ledger.py (loader, post-only mirror) | pass | 29 / 29 | r2_pf_queue_place_ledger_selftest, r3_pf_queue_place_ledger_mutants |
| pf_timing_case_audit.py | pass | 40 / 40 | r1_pf_timing_case_audit_* |
| pf_known_clock_census.py | pass | 17 / 17 | r1_pf_known_clock_census_* |
| pf_cancel_race_study.py | pass | 25 / 25 | r1_pf_cancel_race_study_* |

Every log was produced on the source hashes being committed (r1 / r2 / r3_sources.sha256, checked before the commit). Five harnesses (settlement,
exec windows, scenarios, keep-pull base, queue-place ledger) used to count any nonzero exit as a kill; they now require a check, and three
queue-place mutants that could only crash plant well-formed wrong output instead. The exec-window / scenario kills printed under the old rule
(r1_*_weak_harness.log) are kept as provenance.

## Rules every item follows

- Readings are declared before data is read; consumed days only (Oct 23-27, Nov 8 to 23:02Z); never 11-04..06, 11-09..13 or an
  untouched day; no orders.
- Jobs only via `run_light.sh` with PF_SLOT=claude; memory caps never raised; no edits to a source a running job reads.
- Every checker ships a falsifier; mutants must be caught by a check, not a crash.
- One commit per step, then fetch and an ahead-only push.

## Log

- 2026-11-18 17:16Z — plan created (tick 1). Items 1 and 3 in progress (code done, suites running); item 2 next.
- 2026-11-18 17:29Z — scenarios 37/37; exec-windows 127/128: the K9 test lost its book-bound case when strengthened - fixed (leg 4 now separates
  hi_book from hi_book - B); both K9 mutants caught; full suite rerunning.
- 2026-11-18 18:06Z — items 1-2 DONE: every suite passes with kills from checks; compare.py ALL true; the C1 / C3 splits computed on Oct 23
  (young fills 3.6-8%, E -769 c on them; cancel-pending 2.3% of gross). Committing this step; next item 3 (C2, C12).
- 2026-11-18 18:26Z — v6 (C12 + C2) selftest passes; scenario mutants 41 / 41 by checks; the Oct 23 v6 rerun running (compare.py R5a-d declared
  before it). Found: C1 also asks for the B1a-weighted headline (v5 skipped it) - patch ready for after the run. Item 4's instrument written.
- 2026-11-18 18:38Z — item 4 DONE (cancel race study v2; v1's retry count and C5 population fixed after v1 was read, FINDINGS P66). Item 3: Oct 23
  v6 reproduces v5 (compare ALL true); the B1a patch applied, its mutation suite and Oct 23-27 through the `all` phase running.
- 2026-11-18 19:48Z — the first five-day run finished (all rc 0, Oct 23 compare ALL true), but its receipts showed settlement times v1 had read the
  sealed 10-28 00Z / 01Z hours for day 10-27 (item 9). Its ledger and scenario outputs set aside as VOID; settlement v2 written (guard, status,
  falsifiers, 15 / 15 mutants); the five days rerunning on v2.
- 2026-11-18 21:56Z — STOP conditions met: review findings closed; every module's selftest and mutation suite pass with kills by checks (the suite
  of record above); the scenario replay meets every checklist item code can meet and reproduces its baselines; the protocol committed.
  What remains is the USER's: items 7, 8 and 9's decisions, plus C5 (rule L) and C17 (quote availability) if wanted.
- 2026-11-19 02:12Z — after the loop: the three VOID outputs deleted on the USER's instruction (item 9's second decision; manifest in
  sealed_read_fix_20261118/void_deleted_20261119.csv).
- 2026-11-19 02:49Z — the USER's one-day test ("Score variants vs A+0", Oct 23) through `pf_keep_pull_variants.py`: controls V0a-V0e YES; no
  variant survives the timing scenarios or is above A+0 in every one ([receipt](../keep_pull_variants_20261119/report.md)). The protocol's
  order description corrected (30-sh JOIN orders, not 5-sh). Next, the USER: the learned prediction + value-mapping pipeline.
- 2026-11-19 05:52Z — variants v2 closes review e96321df P2; the pipeline's first-day benchmark and keep / pull test ran, but review 89caa74c
  (P1 POST future-arrival inputs, P2 contract-refused forecasts acting, P2 a gate blind to the value's sign) holds: NOT ACCEPTED, fixes next.
