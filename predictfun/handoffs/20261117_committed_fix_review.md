# Review of committed fix b6eaa7c2

As of 2026-11-17 15:07:09 UTC. Reviewed code b6eaa7c2 and state update
792e2065. Review source 3cef4993; evidence
`reviews/committed_fix_review_20261117/receipt_v1.json`.
Code and synthetic review only; no market tape read or performance estimate.

The four prior areas are substantially repaired. Both original synthetic
selftests pass. One material acceptance issue remains in the random control.

## P1: unmatched draws still qualify for a matched economic claim

`pf_keep_pull.py:402` now samples only eligible live orders and tracks unmatched
pulls correctly. However, selecting a long-lived order at an early step can
leave too few orders for a later scheduled pull. Recording that shortfall does
not make the resulting comparison matched.

In `report()` around lines 502-508, every draw is sent to `k2` and `k3` regardless
of its unmatched count. The code verifies that executed null pulls were eligible,
but does not require their count to equal the score's. The output can still say
EARNS-OVER-MATCHED when the score simply pulls more orders. The new selftest
checks that shortages are reported; it does not check that shortages prevent a
matched-acceptance claim.

Reproduction uses 1,200 synthetic orders across five day cells and twenty market
strata per day, twelve same-side/same-phase orders per stratum. Six become
structurally unavailable at the second live decision; six remain available.
The score pulls the first six at +100 ms and the latter six at +150 ms. Every
order has one filled share worth exactly -1 cent at the common endpoint. All
branch quantities and values are built by the committed `variant()` helper.

Across 500 null draws:

| Quantity | Result |
|---|---:|
| Score's executed pulls | 1,200 |
| Mean null executed pulls | 899.768 |
| Mean unmatched pulls | 300.232 |
| Draws with unmatched pulls | 500/500 |
| Score and null value per removed share | -1 cent |
| K2 SCORE-EARNS | Pass |
| K3 SCORE-RANKS-VALUE | Fail |

K2's null p95 improvement is 914 cents, against the score's 1,200. Its five-day
direction rule also passes. There is no value-per-share discrimination in this
example. K3 correctly rejects it; this review does not claim a demonstrated
false positive in K3. With equal eligibility and late fills, the positive
control matches all 1,200 pulls on each of 200 draws and both K2/K3 fail.

Minimal correction: give a comparison with unmatched draws an explicit
UNMATCHED/NOT-EVALUABLE status, and prevent it from supporting matched acceptance.
Check action counts and timing per stratum/step, not just aggregate eligibility.
Preserve the unmatched draws and counts. Do not discard/retry draws until they
match, or use future survival to force a match; those introduce selection.
A different matched estimand can be declared if shortages make the current
trajectory-level design uninformative. Raw outcomes/oracle diagnostics remain
usable with their stated population and assumptions.

## Confirmed corrections

- **Ledger L5:** observed-range tipping points are separate from legal payoff
  bounds; one-order and paired-arm bounds are valid conservative bounds. The
  previous feasible sign-reversal example is no longer called worst-case robust.
- **Live versus pending:** live scores/outcomes/oracle start at +100 ms under
  the declared clock; +50-ms withdrawal is reported separately. At the live
  decision, zero and 37-ms cancellation land at +100 and +137 ms respectively.
- **Latency attribution:** numeric comparisons value the same zero-latency
  oracle decisions at both latencies. K1 failure is NOT-SHOWN; loss decomposition
  and the independent fill-window control address the previous timing issue.
- **Missing fair inputs:** orders stay in the score population. Missing input
  causes KEEP only at that step; a valid earlier trigger survives missing later
  data. A missing trigger input advances the action to the next valid trigger.
- **Initial null defect:** the old single-step 500-versus-250 example is fixed.
  The remaining problem is multi-step risk-set exhaustion and acceptance.

## Verification and next step

Ledger selftest passed in 14.50 s / 242 MiB; keep/pull selftest passed in 11.60 s /
298 MiB. The additional multi-step counterexample and matched/unmatched controls
passed in 1.08 s / 207 MiB. All sequential on Codex core8. Mutation suites were
reviewed, not rerun. The experiment sources were not modified by this review.

The ledger repair is ready for smoke/full real-data reconciliation; its original
2,054-state and 60-horizon-value gaps are still empirically unclosed. Add the
unmatched-acceptance guard before interpreting keep/pull as matched evidence.
No further model tuning follows from these synthetic checks. All old development
days remain consumed; there is no new validation or strategy-promotion claim.
