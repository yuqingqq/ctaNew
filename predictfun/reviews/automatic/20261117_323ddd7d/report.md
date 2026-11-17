# Scheduled review — 2026-11-17

As of 2026-11-17 15:38 UTC. Request
`a4588711-f930-43c6-9a8d-2f8e229e369f`.

**Pinned target: `323ddd7d224b7b155b0f8873271b0fdc8346cddb`.**
Reviewed the complete range
`38de5e66ccf6e4f3cde8819fc862ffdeb4256a6d..323ddd7d224b7b155b0f8873271b0fdc8346cddb`:
implementation `ebd5c6b333558d40c8202c908b100edbbe7970ef`, including the mutation-test
changes, and the target's HANDOFF/STATUS/FINDINGS update. Source and dependencies
were checked out at the target in `/tmp/predictfun-review-323ddd7d`; the shared
development checkout was preserved.

**Verdict: no new blocking defect found in the reviewed changes. The previous
P1 unmatched-acceptance finding is fixed.** This is an implementation review
with synthetic evidence, not evidence of market performance or profitability.

## Findings, ordered by severity

### P1 from the prior review — resolved: unmatched trajectories cannot pass acceptance

In the pinned `predictfun/research/pf_keep_pull.py:437-455`, trajectory draws
retain their shortages and verify counts per stratum and decision step. In
`:551-560`, any unmatched draw makes the trajectory comparison unevaluable and
sets both trajectory K2 and K3 to null. Report labels at `:571-575` distinguish
that refusal from the selected decision-state diagnostic. No retry-until-matched
or conditioning on future survival was introduced.

The independent reproduction exercises **the emitted report JSON**, not just
the helper that chooses the comparison. It repeats the prior review's 1,200
synthetic orders, five synthetic day cells, 100 strata, 500 draws. Every filled
share is worth -1 cent; the score pulls 1,200 orders. Trajectory draws still
average 899.768 pulls and 300.232 unmatched pulls, with shortages in 500/500
draws. The raw K2 calculation still passes, which preserves the falsifier.
The actual report now refuses that comparison: trajectory K2/K3 are null and
the decision-state K2/K3 are both false. The independent checker verified
dispatch and refusal for every reported policy/variant.

### No new P0/P1/P2 implementation finding — decision-state scope must remain explicit

The new sampler at `pf_keep_pull.py:459-475` matches state counts within each
day/market/side/minute and relative live decision step on the neutral KEEP
path. Eligibility is checked at the sampled step. `_sums` at `:389-411` values
the remaining fills after that decision's cancellation landing against KEEP.
This is consistent with the declared **local decision-state comparison**.

It is a different estimand from a feasible sequence of cancellations. The
source explicitly permits an order at multiple steps (`:56-59`); our fixture
has a mean 300.1 repeated-order states per 1,200 sampled states. Those suffix
values can include the same future fill more than once. Therefore STATES
totals, removed-share counts, and severe-event counts must not be presented
as an executable portfolio's P&L, distinct fills, or distinct severe events.
The `-STATES` readings and separate trajectory refusal preserve that distinction
in this fix. A passing STATES reading supports local selection only; portfolio
acceptance still requires one realized outcome per order. This is a documented
interpretation limit, not a newly demonstrated acceptance bug.

## Prior findings: resolved versus open

The prior report is
`predictfun/handoffs/20261117_committed_fix_review.md`, with evidence
`predictfun/reviews/committed_fix_review_20261117/receipt_v1.json`.

| Prior issue | Current status |
|---|---|
| Multi-step unmatched null could pass K2 | Resolved; independently checked at the generated artifact |
| Initial raw-trigger versus executed-pull mismatch | Remains fixed; counts and eligibility checked |
| Ledger observed-range sensitivity mislabeled as a bound | Remains fixed; ledger source hash unchanged and legal-bound controls pass |
| Pending withdrawal mixed with live cancellation | Remains fixed; live decisions start at +100 ms, land at +100/+137 ms for 0/37 ms cancellation latency |
| Incorrect latency attribution and future fair-availability selection | No regression found; pinned original selftest passes its paired-latency and per-step missing-input controls |
| 2,054 missing causal-state records and 60 missing 2 s values/statuses on real data | Still empirically open; no repaired tape receipts exist at the declared output locations as of this review |

The new range changes no market-data loader, valuation horizon, clock constant,
model, simulator, or action controller. The refactored trajectory `outcome`
still has at most one pull outcome per order. Full report generation was
exercised with constructed orders/fills; this independently checks reporting
and aggregation, not the validity of market inputs. Original kernel/timestamp
reconciliation is covered by the pinned synthetic selftest.

## Verification and limitations

All checks ran sequentially through `PF_SLOT=codex` and `run_light.sh`, using
absolute script paths, on core 8 under its memory limit:

| Check | Result |
|---|---|
| Pinned `pf_keep_pull.py selftest` | Pass; 14.72 s, 302 MiB peak RSS |
| Independent `reproduce.py` | Pass; 2.91 s, 249 MiB peak RSS |
| Equal-lifetime negative-ranking control, 200 draws | Every trajectory matched; K2/K3 false |
| Perfect loser-selection positive control, 200 draws per null | K2/K3 true for both nulls |
| Known-bad state after eligibility ended | Refused by its own live-decision control |
| Mirrored bid/ask losses | Identical quantities and economic values |

The original selftest counted 3,440 synthetic states, 3,151 eligible orders,
1,351 fill rows, 270 kernel checks and 290 invariance checks. Its full report
used STATES because 499/500 trajectory draws were incomplete. These counts and
its positive readings are implementation fixtures only.

The independent report's zero-action sensitivity produces NumPy empty/NaN
warnings for per-removed-share statistics; K3 stays false. No false acceptance
was observed. The development session's 43/43 mutation result is supported by
its local run log and inspected source, but the full suite was **not rerun**
in this bounded review. No full-size resource test was performed.

No market tape, training, or forward block was read. Oct 23–27 remain consumed
development dates. Dates in the synthetic report are fixture labels required
by the reporter, not newly observed data. No untouched validation day was used,
no day-confidence interval was estimated, and no strategy promotion follows.
Replay queue validity, young fills, trade-clock uncertainty, common-endpoint
valuation, structural-cancel asymmetry and inventory management remain outside
what these synthetic checks establish.

Evidence beside this report: `receipt_v1.json`, `verification_runs.json`,
`reproduce.py`, and `synthetic_report/keep_pull.json` / `keep_pull.txt`.
The receipt records the full target, source hashes and generated JSON hash.
It supersedes the prior finding only for this target; the old receipt remains
unchanged as evidence about its earlier target.

## One next priority for development

**Complete the queue-place ledger's real-data reconciliation gate on consumed
Oct 23–27: smoke first, then all five days.** Require L1 LEDGER-COMPLETE,
L2 V1-REPRODUCED and L3 GAP-ACCOUNTED, with explicit statuses accounting for
all 2,054 formerly absent states and all 60 missing 2 s arm values. Preserve
the original artifacts and publish a superseding receipt. The development
session owns this run; this review does not launch it or change its existing
run-authorization boundary. Further interpretation depends on that gate.
