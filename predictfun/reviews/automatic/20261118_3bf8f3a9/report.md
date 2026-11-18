# Manual commit review — 2026-11-18

Target: **3bf8f3a9332de737514142bdf63b0b59ab272bb5**.
Range: `9a114cc12af265b066120bf9359a6d520e1b13c1..3bf8f3a9332de737514142bdf63b0b59ab272bb5`;
reviewed implementation commit `0e20fa8da48f7f7492722829ff534866132cd180`
and result commit `3bf8f3a9332de737514142bdf63b0b59ab272bb5`.
User request: “Check my latest commit.” Manual review; scheduler remains disabled.

**Verdict: no new blocking implementation or accounting defect found in this
scope. The fixes are sound, and the saved results reproduce. Execution timing
and the economic value of cancellation remain unvalidated.** This is acceptance
of the correction, not acceptance of the venue engine or a profitable strategy.

## Findings, in priority order

**Existing critical gate remains open: fill-versus-cancel ordering is not
identified.** On consumed Nov 8, 99 known maker legs / 86 orders / 131 candidate
rows, result as of `2026-11-18T12:55:18.229011+00:00`, S3 still calls three actual
fills avoided at some admissible K7 offset. The new code correctly keeps this
FAIL as the gate of record. Its conditional PASS uses the narrower UUID-derived
clock and requires UUID / REST-status clock alignment, which these data do not
independently establish. Source: `pf_keep_pull_exec_windows.py:1244–1272`;
case explanation: `reviews/settlement_lag_fix_check_20261118/report.md:34–68`.
The three failures occur below offsets approximately -210.3, -218.8 and -224.8 ms.
The UUID-based ST gate is correctly UNIDENTIFIED-UPPER-BOUND-ONLY.

On consumed Oct 23, the corrected ledger has 17,670 baseline fills, 8,862 removed
fills, and 50,553 linked candidate events; as of
`2026-11-18T12:55:54.945415+00:00`. Only 214 removed fills (1.5013% of removed
absolute local value) are classified CANCEL_FIRST under ST and the CP cancel
window. Every one depends on an unmapped leg's S3 fallback lower bound: 346
candidate links traced independently to their leg inputs. No mapped UUID leg
provides a finite ST lower endpoint. Thus even this small fraction is
assumption-dependent, not independent timestamp certification of avoided fills.
The corrected fixed-fill/identity excess-value bounds at 37 ms span
[-145,038, +168,157] cents. They are identification bounds for that ledger,
not statistical confidence intervals or bounds on all possible queue paths.
Evidence: [receipt.json](receipt.json), `ST_population`; result report lines
39–51 and the saved `exec_windows.json` bound fields.

**Minor reporting scope clarification; no new P1/P2:** the result report's
“the day's 50,553 events” (`reviews/keep_pull_corrected_day20261023/report.md:50`)
means candidate events linked to this baseline fill ledger. The scenario replay
contains 560,819 events. Likewise, the reported 10.1% replay-after-UUID rate
(lines 45, 71) uses 77,557 UUID-mapped candidate-event legs, from 90,682 candidate
legs, not all public legs or all baseline fills. It is conditional on creation
at/after matching and aligned clocks. This review records the exact denominators;
the underlying receipt values reconcile.

## Prior findings and the new fixes

| Issue | Status and evidence |
|---|---|
| Common-offset fit treated as universal per-event creation-lag bound (prior review 52c45d57) | Resolved by 9a114cc1; retained here. Independent 10/700 ms counterexample accepts both compatible worlds without a universal bound; impossible creation is refused. |
| Original K6 gate silently evaluated on the tighter K8 clock | Resolved by 0e20fa8d. K7, K8 shared-clock and zero-lag conditional readings have distinct keys and columns; `pf_keep_pull_exec_windows.py:1244–1275`. Planted FAIL/PASS separation is asserted at lines 1593–1605. Saved known readings exactly reproduce. |
| Module tests did not rehearse default upper-only production path | Resolved. Ledger `pf_keep_pull_exec_windows.py:1880–1896` and scenario `pf_keep_pull_scenarios.py:607–621` exercise receipt loading, unknown lag, upper-only bounds and the 19-scenario bank without STlo. Both complete selftests passed in this review. |
| Stale claims of cleared timestamps / universal lag bound | In-place supersession present in FINDINGS C23/P60/P61 and result rows, the STATUS superseded block, and the historical HANDOFF entry. Old receipts remain provenance. |
| Corrected empirical receipts pending | Resolved for consumed Oct 23 and known Nov 8. All 19 saved scenario rows and known-case readings reproduce; 18 non-ST scenario rows are unchanged from v3. STlo was removed; SThi changed. |
| Independent clock/creation-delay semantics; validation and model economics | Open. Corrected arithmetic cannot identify an unobserved match lower endpoint. No untouched validation or learned-model evaluation was performed. |

## What the economic results establish

These are **A+0 fair-edge keep/pull diagnostics**, not the learned
flow/price prediction → economic value mapping → controller pipeline
(`pf_keep_pull_scenarios.py:415`). No model was tuned in either new commit.

Scenario results are from one consumed Oct 23 day: 288 windows, 30,088 order
paths, 560,819 events; as of `2026-11-18T12:59:49.027151+00:00`.
The ledger's counted-state population is 30,466: 28,799 eligible, 1,289
post-only refusals, 354 missing endpoints, 24 stale future books.
SThi has 20,364 fills; baseline M50 has 17,670. These are different timing
scenarios and fill populations, not matched realized trading outcomes.

| Local END +5 s value, cents | Baseline M50 | Latest-match assumption SThi |
|---|---:|---:|
| Excess over the matched STATES null, E | +34,948 | +40,687 |
| KEEP plus the score's cancellation value | -174,961 | -135,043 |

Across the 19 declared scenarios E ranges from -13,342 to +40,687 cents. A
positive E means incremental value over the specified null, not profit. These
summed local marks are not settlement P&L or a self-financing inventory-managed
portfolio. There is no earliest-ST counterpart because the lower endpoint is
unknown. More consumed days can show repeatability, but cannot by themselves
resolve this timing ambiguity. No confidence interval is claimed from one day.

## Verification and limitations

Reviewed committed source in detached checkout `/tmp/predictfun-review-3bf8f3a9`,
preserving the dirty development checkout. Read repository AGENTS.md, CLAUDE.md
reliability rules, Predict.fun state and prior review/correction reports.

Reproduction command (bounded saved-table work and synthetic tests only):

```bash
PF_SLOT=codex /home/yuqing/ctaNew/predictfun/research/run_light.sh review_3bf8f3a9_verify -- /home/yuqing/pricer-sol/venv/bin/python3 /home/yuqing/ctaNew/predictfun/reviews/automatic/20261118_3bf8f3a9/verify.py /tmp/predictfun-review-3bf8f3a9
```

Successful run: 28.70 s, peak 720 MiB, rc 0, codex slot. Evidence:
[verify.py](verify.py), [run.txt](run.txt), [selftests.txt](selftests.txt),
[scenario_recompute.txt](scenario_recompute.txt), [receipt.json](receipt.json).
Verified 33 unique source/dependency/input/output hashes; committed result JSONs
match their saved data artifacts. Both complete pinned selftests pass, including
positive controls and known-bad refusals. Independently reconciled contradiction
counts and gross values, default-lag semantics, fallback provenance and all
19 saved economic rows. A corrupted scenario output receipt is refused.
Known K0–K8 readings recompute exactly from saved identity-linked case tables.

The initial harness attempt failed before tests because its source-list path
resolver doubled the repository prefix. Corrected only the review harness and
retained [run_attempt1.txt](run_attempt1.txt). This was not an implementation
failure. The other session's committed mutation logs report 109/109 execution
window and 33/33 scenario mutants; this review did not rerun those full suites.

No full tape/queue replay, training, untouched validation, external contact or
new capture. Used consumed Oct 23 and Nov 8 artifacts only. Synthetic passes
establish tested implementation behavior, not market predictability or profit.

## One concrete next priority

**Resolve the timing gate with an independent clock/identity case audit of the
three S3 false-avoidance legs.** For each, retain the exact raw identity-bearing
placement, match/status/UUID, first-notice and cancel-confirmation records;
document clock domains and derive only independently supported bounds. A
conclusive result must either justify the clock constraint that removes the
contradiction, or retain it and identify the missing observation needed to
resolve it. Passing under a narrower fitted clock is insufficient. Use the
same audit to distinguish evidence-backed match windows from assumed fallback
windows before using their economics as model targets. This is the next
development gate; model/value-mapping tuning remains premature.

Review-only artifacts and state update. No implementation edits, push, merge,
scheduled acknowledgement or scheduler restart.
