# Testing-readiness review — 2026-11-19

Target: **a2f13f37d58d546b949bb4919725739978a47c76**.
Changes since review `8e63f178003ab98916f3467121c5d1d148344af4` include
`4a6bc58b7c4b6f28bbb8e2cdc1060236f1597f17`,
`fef8226fa5d9265ec7c06f4ed44771a38c78988f`, and the target.
User question: “Is it ready for testing now?”

**Yes: the current keep/pull replay is ready for controlled, scenario-conditional
research tests on consumed days. The three concrete findings from the previous
review are fixed. This is not order-level execution validation or evidence of
profitability.** No additional blocker to that limited test scope was established
by this targeted check. This is a readiness review, not an exhaustive audit of
every new instrument or every possible model/controller integration.

## Closure of the previous findings

| Previous issue | Independent check at the pinned source |
|---|---|
| P2-1, discarded constraints during clock conversion | The original planted case retains book lower 90,050 and REST lower 90,000. After offsets 0/-200 ms, it correctly flags the false avoidance at cancel receipt 90,100. Deliberately removing the REST constraint restores the old wrong result, demonstrating that the check can distinguish them. The implementation also tests feasibility of the converted window. |
| P2-2, permissive timestamp/identity comparison | A valid planted case passes; shifting its endpoint by 5,000 ms or its REST identity second by 1 s now rejects the A5 conclusion. |
| P2-3, cancellation ordering overclaim | CANCELED produces a CONDITIONAL acknowledgement bound, not EVIDENCE. The census no longer claims “then filled”; known-v7 uses first-notice lifecycle bounds and no acknowledgement upper bound on cancellation effect. K5 no longer certifies the cancel range from response timing. |

Known-v7's full saved readings reproduce from its small case tables: **99 legs /
86 orders / 131 candidate rows**, consumed Nov 8, as of
`2026-11-18T17:40:18.469938+00:00`. S3 remains PASS under the declared per-domain
constraints; S1/S2 fail and ST has no identified lower endpoint. The three
previous issue closures do not establish the assumed lower match bounds.

## Results and evidence gates

The corrected reference run contains **149,174 order paths over five consumed
days, Oct 23–27**, with 19 scenario rows; pooled artifact as of
`2026-11-18T20:46:28.060651+00:00`. Published scenario CSV values reconcile with
that artifact, including by-day sums. The current scenario input hashes point
to settlement-times **v2**, and the Oct 27 receipt excludes Oct 28 00Z/01Z.

The baseline's excess local value over its matched null is **+234,279 cents**;
at the M250 trade-clock assumption it is **-27,124 cents**. Baseline KEEP+score
is **-776,492 cents**. These are local END+5s values, not portfolio or settlement
P&L. The timing-stability gate is explicitly NOT-STABLE; the economic conclusion
cannot be treated as robust merely because the implementation gates pass.

The new per-cancel study covers **30,859 cancelled orders**, consumed Nov 8 to
23:02 UTC, as of `2026-11-18T18:35:02.945504+00:00`. It separates requests,
responses, match bounds and notices, includes no-fill orders and retries, and
retains censoring. Its saved result finds no match **provably** after a CANCELED
confirmation under K7; this is one-sided evidence, not proof that every cancel
took effect before its answer. Refused cancels remain outside the replay's
always-successful landing rule. No measured cancellation-effect distribution
or failure-probability target follows from this study.

The committed [test-run protocol](../../simulator_readiness_20261118/test_run_protocol.md)
defines the appropriate scope. Still outside acceptance:

- Order-level fill/queue validity; queue rule M is the only replayed rule.
- A policy acceptance decision: inventory/portfolio accounting, quote availability
  and guard survival are not established by this local keep/pull comparison.
- Generalization: the five reference days are consumed development data, not
  untouched validation.

The accidental Oct 28 REST-hour access is recorded in the
[sealed-read correction](../../sealed_read_fix_20261118/report.md). Corrected
outputs do not use those hours. Preserve the access history and exclude Oct 28
from claims of untouched validation; a later label or rerun cannot undo the read.
No sealed or forward data was read during this review.

## Bounded verification and limitations

Reviewed the state, correction report, per-cancel report, readiness plan,
protocol and final result artifacts against a detached checkout at
`/tmp/predictfun-review-a2f13f37`. Repository reliability rules apply. Original
development modifications were preserved.

Six pinned synthetic selftest suites passed: timing audit, clock census,
execution windows, scenario replay, settlement times and cancel race. The old
independent counterexamples were reapplied with positive controls. Final suite
source manifests match the pinned sources. Known/census/cancel-study output
hashes were checked; known readings were recomputed. The five-day CSV/JSON and
settlement-v2 provenance checks are accounting/provenance checks, not a fresh
five-day queue replay. Full mutation suites were not rerun in this review.

Evidence: [verify.py](verify.py), [receipt.json](receipt.json),
[selftests.txt](selftests.txt), [run.txt](run.txt).
All checks used `PF_SLOT=codex` through the absolute `run_light.sh` path.
The six suites completed in the initial attempts (peak 729 MiB). Two subsequent
review-harness receipt lookups needed correction: hours live under `R`, and the
scenario records the settlement table hash rather than its JSON receipt path.
Those failures are retained in [run_attempt1.txt](run_attempt1.txt) and
[run_attempt2.txt](run_attempt2.txt); neither was an implementation-test failure.
The final checks reused the already-passed unchanged-source selftests and
completed at rc 0 in 1.85 s / 211 MiB. For full reproduction, run:

```bash
PF_SLOT=codex /home/yuqing/ctaNew/predictfun/research/run_light.sh review_a2f13f37_ready -- /home/yuqing/pricer-sol/venv/bin/python3 /home/yuqing/ctaNew/predictfun/reviews/automatic/20261119_a2f13f37/verify.py /tmp/predictfun-review-a2f13f37
```

No training, new tape/queue experiment, live activity or untouched validation.

## One next priority

Run a **predeclared baseline-versus-candidate research comparison** through the
same consumed-day protocol and scenario bank. Preserve by-day values, fill
populations, age/source splits and timing sensitivity. Investigate changes that
survive the timing scenarios before treating a model improvement as an economic
improvement. Any model/controller integration needs its own action and accounting
reconciliation; the reference run currently evaluates the A+0 fair-edge rule.

Review-only evidence and owned state update; scheduler stays off, no push/merge.
