# Manual fix review — 2026-11-18

Pinned target: **c23e6f297c9b822cb3af8247da0b5e1ea99148f5**.
Reviewed the complete change from 24db60f14181fa159d1e4eef62aded19c81d5a6b,
including the ledger, mutation checks, committed logs and state changes, in
`/tmp/predictfun-review-c23e6f29`. Another session owns implementation.
Scheduled reviews remain removed.

**The earlier two findings are fixed in code, but this is not yet a validated
execution engine. Two P2 issues remain in the candidate-event diagnostics.**
The new module is an interval audit over fixed M50 fills; the queue engine is
unchanged. No empirical ledger or known-order result is included in this commit.

## Findings, ranked by severity

### P2 — Contradiction totals silently exclude multi-event fills

`predictfun/research/pf_keep_pull_exec_windows.py:375–381` restricts both
`fills_contradictory` and `replay_after_evidence` to `link_st == 1`. The report
prints these as contradictory fills and replay times after the evidence cap
without that population restriction (`:449–453`). A fill can have several
candidate events and nevertheless contradict **every** candidate window or
exceed **every** candidate evidence cap. Those definite contradictions are
currently absent from the headline totals and gross values.

Independent synthetic reproduction, as of 2026-11-18T06:00:08Z: 40 counted
orders, 38 eligible, 59 fills and 73 candidate events. Every candidate's S1/S2
window contradicts its evidence cap, and every replay fill is later than all
its candidate caps. The ledger retains these contradictions, but both summary
counts report **43 rather than 59**, omitting all 16 multi-event fills
(523.9004 synthetic cents gross). All five V0 controls still pass. This is a
diagnostic accounting error, not a measured incidence or loss on market data.

Requested correction: aggregate candidate predicates back to each fill;
distinguish all-candidate, some-candidate and unknown cases, reconcile counts
and gross values to the full population, and name restricted denominators.
Include a multi-event contradiction falsifier. The ledger's retained identities
make this correction possible without changing any strategy.

### P2 — Multi-event ranges are outer bounds, not exact joint event extrema

The headline claims exact extrema with one shared instant per event
(`pf_keep_pull_exec_windows.py:43–45`, `:51`, `:445`). This holds for the unique
event sweep. For ambiguous fills, `:345–361` independently optimizes each
fill/candidate pair, then adds its extrema to the unique-event extrema. The
same candidate event can therefore take different times in different terms.
The opening description at `:18–20` acknowledges independence, but the saved
bounds and report do not carry that distinction.

Independent synthetic case: two events, one unique fill at each, and a third
fill assigned to either event. Exhaustive enumeration of the common event
times and identity alternatives gives **[0, 10] synthetic cents**; the current
aggregation gives **[-10, 20]**. A positive control with two opposite-value
fills sharing one unique event correctly gives [0, 0].

These relaxed bounds can conservatively contain the coherent extrema; the
example does **not** establish a false profitability certificate. However,
their width can make a resolvable sign look unresolved, so it must not be
interpreted as the exact uncertainty left by the evidence. Either preserve
joint candidate consistency or label the mixed-population result explicitly
as an outer relaxation, including the affected fill/gross coverage in the
machine-readable receipt and report. Exact joint extrema need a separate claim.

## Prior findings: resolved versus open

- **Resolved in code:** removed/kept fills have their own cross-tabs and
  denominators (`:269–285`, `:341`); the prior FINDINGS/STATUS headline is
  qualified. The new self-test verifies conservation and rejects a lost fill.
- **Resolved in code, empirical receipt pending:** raw legs, candidate IDs,
  event windows, fills and every counted order/status are persisted
  (`:390–417`), with input/dependency/output hashes and identity reconciliation
  (`:200–213`). Synthetic output is exercised end to end. The real-data
  artifact has not yet established coverage or successful joins.
- **Implemented and synthetically checked:** REST Confirmed time is an upper
  bound rather than a match point; touching windows remain ties; cancellation
  uncertainty is represented; event legs intersect, with contradictions kept;
  unique-event contributions share a time. The multi-event caveats above remain.
- **Still open by design:** evidence/assumption calibration on known real
  lifecycles, clock alignment, queue/fill/remaining-size/eligibility
  re-simulation, and treatment of orders without an M50 fill. Activation and
  notice timing remain configured assumptions. These were not fixed by adding
  the audit and remain prerequisites for interpreting full strategy economics.

## Checks, artifacts and limits

Read the repository guide, CLAUDE reliability rules, HANDOFF, STATUS, FINDINGS,
the previous [review](../20261118_d1de7504/report.md) and
[execution-window addendum](../20261118_d1de7504/execution_windows_addendum.md).
The target source SHA256 is
`82906f5e2a38e587662b7e1229a5b92a47ba751072e0eddd1626e8f599fbb78e`, matching the
committed mutation run's source identity.

Bounded verification used only the Codex slot:

```bash
PF_SLOT=codex /home/yuqing/ctaNew/predictfun/research/run_light.sh review_c23e6f29 -- \
  /home/yuqing/pricer-sol/venv/bin/python3 \
  /home/yuqing/ctaNew/predictfun/reviews/automatic/20261118_c23e6f29/reproduce.py \
  /tmp/predictfun-review-c23e6f29
```

Exit 0; 2.63 s wall time; 212 MiB peak RSS. The unchanged target's complete
self-test passes, including its planted bad inputs and end-to-end rehearsal.
The independent reproductions above pass, and the independent conservation
check rejects a deliberately lost fill. Evidence: [receipt](receipt.json),
[reproduction](reproduce.py), [self-test output](selftest.txt),
[contradiction case](contradictions.txt), [run record](run.txt).
The committed 33/33 mutation log and mutation definitions were reviewed; that
suite was not rerun. No full training, tape replay, real ledger run or real
`known` run was launched.

K1–K3 must not be treated as a venue-engine acceptance gate. K1 compares a
match lifecycle whose upper bound already incorporates cancellation with the
cancel window; it is a consistency check. K4 checks the research anchor and
can report lifecycle misses/false avoidance while K0–K3 all say YES
(`:484–518`, demonstrated in the pinned synthetic self-test at `:666–678`).
Any subsequent real-data conclusion must inspect K4 and the ledger, not only
the YES readings. This limitation is separate from the two new findings.

All new measurements here are **synthetic**, with no market-data population or
profitability inference. Oct 23–27 and the Nov 8 own-order sample are already
consumed development evidence; neither can become untouched validation by
passing this instrument. No later validation day was read.

## One concrete next priority

**Close the candidate-event diagnostic accounting gate before the first real
ledger run:** include multi-event contradictions/cap violations, and identify
relaxed versus jointly exact bounds in the saved receipt, with the two small
reproductions above as acceptance cases. Then the existing ledger/known-order
check can test real coverage and window consistency; queue-aware scenario
replay remains a subsequent gate. Keep model tuning on hold until those
execution labels are defensible.

Only review evidence and review-owned state additions are included in this
review commit. Existing development changes and artifacts are preserved.
