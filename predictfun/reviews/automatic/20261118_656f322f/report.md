# Scheduled ledger-results review — 2026-11-18

Review as of 2026-11-18 02:18 UTC. Request
`435567ff-031d-461b-8bf2-d182dcd666af`.

**Pinned target: `656f322fab161354f26032a43a2b917138c33a0e`.**
Reviewed the full range
`99d17541e05e8f6cdc7a2d1bd31c28bd64da5fe5..656f322fab161354f26032a43a2b917138c33a0e`.
It contains one results/state commit and no implementation changes. The pinned
source and committed receipts/logs were inspected in a detached checkout at
`/tmp/predictfun-review-656f322f`.

**Verdict: no new blocking finding. The prior 2,054-state and 60-horizon-value
accounting gaps are now empirically accounted for on the five consumed days.**
The repair preserves the original eligible-row results. It does not establish
profitability or validate the simulated fill model.

## Findings and prior issues, ordered by severity

### Prior P1 accounting gap — resolved for this recorded population

The saved ledger has 151,228 counted causal states across 1,440 markets on
Oct 23–27. Each counted state has exactly one JOIN, REST1 and IMPROVE arm row,
with explicit eligibility/exclusion status for each valuation.

Independent checks of the saved tables reproduce:

| Day | Counted states | Original saved states | Accounted gap | Original arm rows missing 2 s value |
|---|---:|---:|---:|---:|
| Oct 23 | 30,466 | 30,088 | 378 | 9 |
| Oct 24 | 30,704 | 30,358 | 346 | 14 |
| Oct 25 | 29,996 | 29,558 | 438 | 19 |
| Oct 26 | 30,242 | 29,832 | 410 | 14 |
| Oct 27 | 29,820 | 29,338 | 482 | 4 |
| Total | 151,228 | 149,174 | 2,054 | 60 |

The 2,054 exclusions comprise **1,920 missing endpoint marks and 134 stale
future books**. All 60 missing two-second arm values have `missing_endpoint`
status. Those 60 arm rows are not 60 distinct states: multiple arms share a
state. Contract minutes 4–5 contain 1,876 of the 2,054 five-second exclusions.

Evidence at the pinned target:
`predictfun/reviews/queue_place_ledger_20261117_v2/queue_place_ledger.txt:24`
through its reconciliation table, and the per-day receipts' `outputs` and
`reconciliation` fields. The responsible source checks are
`pf_queue_place_ledger.py:237` (conservation/status predicates), `:341`
(original-row reconciliation), and `:470` (saved-artifact hashes).

Saved state rows and tried event rows reconcile through the declared keys with
no value/status mismatch. The original `gone_before_first` records lack birth
identity; those reconcile by counts within market, side and birth kind. All 18
original readings reproduce, including their pooled means, not merely their
booleans. L1/L2/L3 therefore pass on this artifact
set. This supersedes the earlier reviews' **pending real-data reconciliation**
status; their old receipts remain unchanged as historical evidence.

### No new P0/P1/P2 defect — economic interpretation remains qualified

The source keeps observed-range sensitivity separate from legal payoff bounds
(`pf_queue_place_ledger.py:514`, `:526`, `:538`, `:598`). Recomputing both the
single-arm bounds and paired bounds agrees with the report. The former bound
labeling issue remains fixed.

As of the committed result receipt, **2026-11-18 02:09:46 UTC**, the eligible
five-second common-endpoint values are:

| Arm | Eligible orders | Mean value, cents/order |
|---|---:|---:|
| JOIN | 149,174 | -13.046 |
| REST1 | 146,483 | -13.889 |
| IMPROVE | 7,976 | -7.229 |

Including the additional states that can be valued at two seconds moves the
two-second means by only +0.0871 / +0.0457 / +0.0134 cents for JOIN / REST1 /
IMPROVE. That supports the declared L4 statement **at two seconds**. It does
not prove negligible exclusion bias at five seconds, where those marks remain
unavailable. The recovered JOIN subset's +0.561 cents is based on 1,430
two-second-valued excluded states, not all 2,054 exclusions.

REST1's robust sign is **negative**, not positive: even assigning the excluded
feasible orders their optimistic legal payoffs leaves the aggregate value
negative. JOIN and IMPROVE have bounds spanning zero. Their unknown five-second
values are not imputed as zero. Evidence:
`queue_place_ledger.txt:34-38` and `:46-48`; the pinned handoff at
`predictfun/HANDOFF.md:4-5` is consistent with those results.

The previously repaired unmatched keep/pull acceptance, live-versus-pending
clock separation and missing-fair handling are unchanged by this results-only
commit. STATES remains a local decision-state selection diagnostic; it cannot
be treated as an executable portfolio with repeated cancellations of one order.
No new keep/pull performance claim is present in this commit.

## Verification

`verify_artifacts.py` ran through the Codex slot with absolute paths, using
only existing derived tables and pinned source. It passed in **16.12 seconds,
1,279 MiB peak RSS**, within the wrapper's four-GiB address-space limit.

Checks included:

- All 18 committed result artifacts, including logs, match the live output
  copies. All 30 ledger parquet hashes (five full days plus smoke), five
  original row-table hashes, and recorded source hashes match their receipts.
- Every day contains the complete 288 consecutive five-minute UTC market
  windows; original counters and saved-table identities reconcile.
- Conservation and original-row reconciliation were recomputed on every day.
  Independent action-key counts, eligibility/value presence, finite values,
  and zero fill/value for post-only refusals pass.
- All 18 original readings and pooled means reproduce from the ledger. Reported
  exclusion totals, contract-minute counts, L4 mean shifts, L5 sensitivities
  and legal bounds agree with the stored tables.
- The clean first-day artifact passes, and all ten planted corruptions are
  refused by their respective instruments when rerun in memory. Legal-bound
  positive and counterexample controls pass. No artifact was modified.

The committed selftest/mutation logs show synthetic success and 29/29 planted
code errors detected. Those complete suites were inspected, not rerun because
the implementation is unchanged. Runtime tape-parity checks are recorded in
the day receipts; this review did not replay raw tape or independently
reconstruct every endpoint from raw book events.

Machine-readable evidence: `receipt_v1.json`, `verification_runs.json`.
The verifier is `verify_artifacts.py`. Evidence records the full target,
result/review timestamps, source/artifact identities, populations and controls.

## Limits

Oct 23–27 are consumed development days, not untouched validation. No model
was trained or promoted, no new tape experiment ran, and no confidence interval
or live-performance claim is made. Values are simulated cash plus inventory
marked at the common endpoint, not fill-relative markouts or settlement P&L.
Queue/fill validity, uncertain trade clocks, structural cancel asymmetry,
optimistic IMPROVE fills, inventory management and EARLY's ex-post selection
remain separate limitations. Accounting closure does not resolve them.

## One next priority for development

**Run the corrected keep-versus-pull measurement: smoke, then the five consumed
days using the pinned policy and controls.** Require KEEP to reproduce the
original eligible JOIN rows, retain every counted state's exclusion status,
report matched trajectory evaluability separately from STATES, and show the
same-decision 0/37-ms latency decomposition with the existing fill-clock and
young-fill splits. The goal is to measure how much loss is avoidable and whether
the causal score selects better cancellations at matched participation.

Development owns the run under its existing authorization boundary. This review
does not launch it, modify implementation, or select a new model/threshold.
