# Manual result review — 2026-11-18

Target: **541aeca9ad9cefddd6b6346146072ea05692e1ed**.
Range: `a50bc5292c14c2d26539fec7d0572198a7aef487..541aeca9ad9cefddd6b6346146072ea05692e1ed`.
Reviewed the code and artifacts in `d7591cb911966e7ffd9e3bd32594a0445ac77428`
and `541aeca9ad9cefddd6b6346146072ea05692e1ed`.
User: “Check my results.” Review only; scheduler remains disabled.

**The independent book-clock evidence is useful progress, and the three original
case conclusions survive exact checks. The new cancellation counts reproduce.
Three P2 correctness gaps remain in the instruments/receipt semantics; none is
shown here to reverse the saved three-case clock conclusion. Actual match versus
cancel-effect ordering, and strategy profitability, remain unidentified.**

## Findings, ordered by impact

### P2-1 — The per-domain split drops constraints before converting clocks

`predictfun/research/pf_keep_pull_exec_windows.py:1051–1058` assigns each leg's
already-clipped S3 lower endpoint to either book or status. However,
`sem_windows` constructs it from **both** `book_lower` and `REST_lower`
(`:344–375`). A constraint that loses before conversion can win after each
clock's offset is applied. Keeping only the initial winner makes the purported
per-domain gate incomplete (`:1227–1230`).

Independent planted case, all times in ms:

| Quantity | Value |
|---|---:|
| Raw book lower endpoint / offset | 90,050 / 0 |
| Raw REST lower endpoint / offset | 90,000 / -200 |
| Cancel-answer receipt | 90,100 |
| Correct mapped lower endpoint | max(90,050, 90,200) = 90,200 |

The current builder drops the REST constraint and reports zero false avoidances
(a passing gate). Retaining both constraints produces one false avoidance
(a failing gate). Both inputs satisfy the existing raw-endpoint reproduction
check. This is a functional false-pass counterexample, not a cosmetic label.

**Observed scope:** adding the missing REST lower constraints changes 47 of the
131 saved candidate rows, but leaves the saved S3 count at zero on 99 known
legs. This diagnostic holds the saved candidate set fixed; it does not validate
cross-clock candidate selection, upper-end clipping or contradiction statuses.
Retain constituent constraints, convert by domain, then form the hull/intersection
and determine feasibility. Do not discard a bound merely because it loses on
unconverted timestamps.

### P2-2 — Timestamp and identity checks accept materially wrong inputs

`predictfun/research/pf_timing_case_audit.py:582` uses default `np.isclose` on
epoch milliseconds for A3. Its relative tolerance is approximately **17.94
million ms** in the planted case. Changing the recomputed endpoint by **5,000 ms**
still yields `A3_reproduces=True` and
`CONTRADICTION-REMOVED-BY-INDEPENDENT-BOOK-CLOCK-BOUND`. A much larger perturbation
is refused, so this is a permissive check, not an unexercised path.

The same pattern at `:575–580` applies relative tolerance to content identity
fields including integer market IDs and epoch seconds. Changing the REST
maker's `exec_s` by **one whole second** still yields `SAME-SETTLEMENT` and the
passing A5 verdict. Use exact comparisons for IDs/discrete fields and explicit
absolute tolerances with `rtol=0` for numerical fields where rounding warrants
one. Add negative controls at the operational resolution.

**Observed scope:** independently recomputing the three actual binding endpoints
and using exact equality on their REST content keys confirms all three. This
finding invalidates the instrument's ability to reject certain bad inputs;
it does not demonstrate that these saved three identities/endpoints are wrong.

### P2-3 — Cancellation receipt labels disagree with their evidence

There are two linked semantic problems:

- `pf_known_clock_census.py:106–129` defines N3 from `CANCELED` plus eventual
  full quantity. It never compares the answer with a match time, and its order
  table omits the answer/notice timestamps from that predicate. A synthetic
  order with its fill notice **before** the answer receives the same “then
  filled” / `NO` verdict as one with its notice after it. The related
  `A1_filled_after_canceled_answer` flag at `pf_timing_case_audit.py:592` likewise
  tests only answer class and total filled quantity.
- More directly, `pf_timing_case_audit.py:652–653` writes
  **`match <= cancel answer receipt` as EVIDENCE for CANCELED**, while the same
  saved case says `A1_premise=NOT-ORDERED-BY-THE-ANSWER`. All three archived
  cases contain this contradiction. A prose caveat cannot supersede that
  machine-readable bound. Apply one consistent conditional/unknown treatment
  to this bound and any dependent lifecycle/latency inference.

The reported count of 63 CANCELED/full orders is numerically correct. An
independent check also confirms that all 63 had their first positive fill
**notice** after the first answer. That establishes delayed observation, not
matching after cancellation took effect. A match before cancel processing,
followed by a delayed notice, remains compatible. Neither cancellation failure
nor cancellation success is identified by the response-plus-final-quantity
count alone. The report's more cautious statement that ordering is unknown is
appropriate; its “answer does not stop a fill” headline should not become a
simulator failure-probability target.

For snapshot diagnostics, compare the snapshot with cumulative fills already
observed at that instant as well as with eventual fills; “after the first notice”
does not in general mean the whole final quantity was already known.

## What reproduces and what remains open

Population: consumed Nov 8, read only through 23:02 UTC; **99 legs / 86 filled
orders / 131 candidate rows**. Census v3 as of
`2026-11-18T16:16:08.017348+00:00`; known v6 as of
`2026-11-18T16:16:26.283013+00:00`.

- The saved census tables and N0–N5 readings reproduce exactly. Its book bounds
  cover 88/88 legs with a cancel confirmation under the declared content-identity,
  stamp-before-receipt and local clock premises.
- All saved known-v6 readings reproduce: per-domain S3 PASS, S1/S2 FAIL,
  ST UNIDENTIFIED-UPPER-BOUND-ONLY. Old K7-basis readings remain separate.
- For original cases 20/45/73, exact endpoint and settlement-identity checks pass;
  their saved independent book bounds exceed the respective critical offsets.
  The archived raw lifecycle records reproduce A0 and the cancel/notice records.
  This supports the previous review's requested clock explanation, under its
  stated premises. It does not identify S3's assumed 250 ms lower-match bound.
- Cancel-answer/final-fill counts: CANCELED 63 full + 6 partial; matched-or-gone
  1 full; refused 14 full + 1 partial; no answer 1 full. These are counts among
  **orders selected for having filled**, not rates per cancel or per attempt.
- Snapshot/alert counts reproduce: 74 orders have a terminal snapshot below
  eventual fill quantity; 62 of those snapshots follow the first notice;
  73 orders have a named late-fill or quantity-conflict alert. Alerts inherit
  their detector's timing premises and are not independent match timestamps.

The saved Oct 23 evidence-tier table also reproduces: **8,862 removed fills /
528,493 cents absolute local value**, baseline ledger as of
`2026-11-18T12:55:54.945415+00:00`. No CANCEL_FIRST classification is supported
by the EV-only window; S3/ST avoidances depend on assumed lower bounds. This
is development evidence. No economic replay or new prediction-model result is
contained in these two commits, and no untouched validation day was evaluated.

Prior review status: the requested independent three-case audit and extension
to known legs are delivered; the denominator clarification is corrected in
place. The clock explanation survives the independent checks above. The
instrument gaps in this report and uncertainty in cancel-effect/match timing
remain open. No profitability, engine acceptance or measured cancel latency
distribution follows from passing implementation checks.

## Verification

Read repository AGENTS.md, CLAUDE.md reliability rules, current Predict.fun
state and prior review. Pinned source was isolated at
`/tmp/predictfun-review-541aeca9`; dirty development files were preserved.

Both checks used the codex slot through the absolute `run_light.sh` path:

```bash
PF_SLOT=codex /home/yuqing/ctaNew/predictfun/research/run_light.sh review_541aeca9_verify -- /home/yuqing/pricer-sol/venv/bin/python3 /home/yuqing/ctaNew/predictfun/reviews/automatic/20261118_541aeca9/verify.py /tmp/predictfun-review-541aeca9
PF_SLOT=codex /home/yuqing/ctaNew/predictfun/research/run_light.sh review_541aeca9_scope -- /home/yuqing/pricer-sol/venv/bin/python3 /home/yuqing/ctaNew/predictfun/reviews/automatic/20261118_541aeca9/followup.py /tmp/predictfun-review-541aeca9
```

Evidence: [receipt.json](receipt.json), [followup.json](followup.json),
[run.txt](run.txt), [followup.txt](followup.txt), [selftests.txt](selftests.txt),
[verify.py](verify.py), [followup.py](followup.py).
Runs passed in 6.09 s / 215 MiB and 3.00 s / 434 MiB respectively.
Three pinned selftests passed. Seventeen latest source/output hashes checked;
additional saved input-table hashes checked before the scope reproduction.
Every review counterexample has a valid comparison or known-bad refusal control.
The committed full mutation-suite logs were inspected but not rerun.

No full journal scan, tape/queue replay, training or new capture. Saved Nov 8
tables and the three archived raw lifecycles were used, plus the consumed Oct 23
ledger. The full bot/research-frame matching census was not regenerated; the
saved book-bound values and their declared premises remain the basis for that
part of this bounded review.

## One concrete next priority

**Repair and supersede the timing-evidence receipt before expanding the
cancel-race census.** The acceptance check is that the three counterexamples
above are handled correctly—retain both clock constraints, reject timestamp/
identity mismatches at the required resolution, and keep CANCELED match ordering
unknown where evidence cannot establish it—while the exact saved-case checks
still reproduce. Then the proposed per-cancel study is useful if it separates
request, response, actual match bounds and notice times, includes no-fill orders
and retries, and preserves censoring/unknown outcomes. It must not train a cancel
failure model from “notice after answer.”

Review-only report/evidence and owned state prefixes; no implementation edits,
push, merge or scheduler acknowledgement.
