# Review of the ledger repair and proposed keep/pull measurement

As of 2026-11-17 13:58:04 UTC. Reviewed 7905b3c1/09fd69d9 and
360c2362/e14e064f. Review source ab597948 (initial ef4fc17b), evidence
`reviews/ledger_keep_pull_review_20261117/counterexamples_v2.json`.
This is a code/synthetic review; no market tape was read and no new economic
performance result was produced. Candidate sources remain unchanged.

The ledger repair addresses the original silent-skip defect in its control
flow: counted states survive failed paths, every arm/horizon has a status,
two-second values survive a failed five-second path, and saved v1 values are
reconciled. Original v1 outputs remain untouched. Both original synthetic
selftests pass on review. The actual 2,054-state gap and 60 missing two-second
statuses still need their market-data reconciliation; synthetic success does
not close that empirical check.

## Findings before interpreting the proposed full run

### P1: the ledger's L5 is an observed-range sensitivity, not a valid bound

`pf_queue_place_ledger.py:507` compares the missing mean needed to change sign
with the most extreme **observed** value, then report() calls the result
SIGN-ROBUST-TO-EXCLUDED. Unobserved orders can exceed the observed maximum.

Synthetic counterexample: two observed orders worth -1 cent each and one missing
order. L5 reports robust because the missing order must earn +2, beyond the
observed maximum -1. A missing +3-cent result flips the total positive and is
well inside the legal +1,500-cent bound for a 30-share bid at 50. The independent
positive control, two -1,000-cent observed orders and the same missing-order
bound, really cannot flip sign.

Fix: keep the tipping-point number but label it an observed-range comparison,
or use per-missing-order payoff/size bounds for a genuine worst-case claim.
Paired arm differences need valid bounds for both arms. This reporting defect
does not invalidate the repaired ledger construction or its L1-L3 identities.

### P1: the keep/pull null does not match actionable participation

`pf_keep_pull.py:374` permutes first-trigger steps within day/side/contract-minute
cells. `outcome()` subsequently rejects a trigger after the destination order's
endj. A permutation preserves raw trigger counts, but not executed pulls or
their distribution in expectation when triggers and order lifetime correlate.
The docstring's claim that pull counts match in expectation is false in general.
These cells also do not match UTC hour or market composition.

Synthetic counterexample: 1,000 orders across five synthetic day cells, 500
early actionable triggers, and 500 orders with an already-decided structural
cancel. Every removed filled share has the identical value -1 cent. The actual
score makes 500 actionable pulls; 500 null draws average 249.924 pulls, range
226-277. K2 passes on all five cells despite **no value-per-share ranking**.
K3 correctly fails in this example, so this is a defect in the claimed K2
matched comparison, not evidence that both gates necessarily falsely pass.
With identical eligibility for every destination, the positive control preserves
500 pulls in every one of 200 draws.

Fix: randomize over the causal risk set of eligible live orders and explicitly
match actionable counts and decision timing within appropriate side/time/market
strata. Keep one outcome per order and report any unmatched units. Do not match
using future fills or future lifetime. Check matching after applying eligibility;
retain economic value and removed-share diagnostics separately.

### P2: the oracle's latency interpretation is confounded by pending cancellation

The first allowed decision is +50 ms while activation is +58 ms. This is a
pending-order cancellation, not the declared live resting-order decision.
`landing(1, L)` is +95 ms for L=0, 25 and 37 because the fixed pending rule binds.
`outcome()` has an upper eligibility limit, but no lower live-order bound.

There is an additional mathematical simplification: with one fixed-price order
and one common endpoint, every share has the same signed value
`side_sign * (endpoint_mark - order_price)`. Cumulative value is monotone. For a
losing order the oracle prefers its first available pull; for a profitable one
it keeps. Thus the oracle's 0/25/37-ms outcomes coincide under the current
pending rule, regardless of the later tape. It measures perfect endpoint-sign
selection at the first opportunity, not ideal detection timing later in the
order's life. It remains a valid upper bound within this restricted setup.

Synthetic example: five orders lose 30 cents each, from fills of 20 shares at
+70 ms and 10 at +250 ms, each worth -1 cent/share. At each of L=0/25/37 the
oracle saves 10 cents/order. K1 fails even at zero cancellation latency; its
LATENCY-LIMITED label cannot identify incremental 37-ms latency as the cause.
K4's comparison of significance flags likewise does not establish a causal
latency effect or absence of ranking when both tests fail.

Fix: separate pending-post withdrawal from live-order keep/pull. Start the
primary comparison at the first eligible live-order decision under the declared
client-state clock, and compare zero versus actual cancellation latency at the
same decision state. Report the oracle curve numerically and call a failed
50%-avoidability threshold NOT-SHOWN; distinguish decision-grid, pending, and
cancellation delays. A score failing is not an information-limit finding.

### P2: whole-future fair availability selects the score population

`pf_keep_pull.py:250` requires finite fair values at every step through +4.9 s
before admitting an order to the score population. A missing late fair input
can exclude an order whose valid early score would already have cancelled it.
This is future-availability selection, not a demonstrated leak in score_steps().
Handle missing inputs at the decision where they are needed, with an explicit
status/fallback and retained order outcome. The future book/mark requirements
needed for labels remain a separate issue. This finding is from source review;
its frequency and economic effect on real tape were not measured.

## Next priority and run order

1. Correct L5's claim and the keep/pull eligibility/null/latency interpretations
   above. Add these counterexamples to the relevant selftests; passing mutation
   tests alone does not validate the economic meaning of the implemented rule.
2. Replay the ledger smoke and five consumed days. Require exact v1 reconciliation
   and explicit accounting of every missing state/horizon. Read the coverage
   effect rather than assuming it is negligible. No further model tuning yet.
3. Run the corrected same-state keep/pull measurement: fixed neutral live order,
   fixed endpoint, keep and pull branches, zero and actual cancellation latency.
   Report avoided ordinary/severe losses, foregone gains, retained participation,
   and the declared causal score against properly matched controls. Count the
   documented P47 structural-cancel asymmetry, keeping the reference engine
   frozen for the ledger reconciliation.
4. Use the result to choose the research direction: large avoidable loss but
   poor causal selection motivates better risk/value prediction; a material
   paired zero-versus-actual-latency gap motivates timing/latency work; little
   upside even under the restricted oracle motivates revisiting entry/exposure.
   These are future decision criteria, not new findings about current strategy
   performance. Prospective validation still requires later untouched days.

## Verification

Original ledger synthetic selftest: pass, 14.36 s / 241 MiB, including silent
skip reproduction, ten planted artifact errors and four known-bad refusals.
Original keep/pull synthetic selftest: pass, 15.68 s / 281 MiB, including kernel
reconciliation and its four planted/four known-bad controls. Review counterexamples
and independent positive controls: pass, 1 s / 200 MiB. All ran sequentially on
Codex core8. Original mutation suites were inspected but not rerun. No market
performance claim or model/strategy promotion; no full tape run launched.
