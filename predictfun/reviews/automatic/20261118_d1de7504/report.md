# Manual timing-result review — 2026-11-18

Pinned target: **d1de75049fdf5a2477e279ca108114b5aef09b27**.
Reviewed the complete change from f0d75fc12ed3e4eef99983c8ab2228a76fcd6133 in
`/tmp/predictfun-review-d1de7504`, including source, result artifacts and state.
Another session owns development; scheduled reviews remain removed.

**The critical research block is execution ground truth: whether cancellation
could actually precede the fills it is credited with avoiding.** The audit
exposes sensitivity to that missing timing evidence. It does not establish
that a stronger model would help, that the model has no skill, or that all
possible timing evidence has been exhausted. The tested contribution/bound
arithmetic passes; two result-quality gaps below need qualification.

## Main unresolved research gate

Result as of **2026-11-18T04:19:44.800261+00:00**, consumed Oct 23–27:
151,228 counted states, 143,134 eligible orders, 6,040 post-only refusals and
2,054 declared path exclusions; 86,904 M50 fill rows. No untouched validation.

The baseline assigns a trade time and credits the cancellation with avoiding
fills at or after its assumed landing. The new audit allows that fill time to
vary. The reported selection-excess ranges are S1 [-426,829, +613,334] cents
and S2 [-638,264, +813,718] cents. Both include either sign, as do the reported
day/side/evidence splits. They are conditional scenario ranges, not confidence
intervals, absolute profits or calibrated bounds on live performance.

The distinction matters: public WS timing here refers to settlementsPending;
E2 is inferred from book changes; the fallback uses an executedAt second.
Neither greater timestamp precision nor a constant offset establishes the
venue's actual matching instant. Cancellation landing is itself still modeled
as decision +37 ms. Reliable ordering needs event semantics and aligned clocks
for both sides of that comparison.

The source explicitly retains M50 quantities and eligibility while varying
each fill's time independently. Its interval aggregation is a min/max hull of
candidate-leg intervals, not a jointly feasible sequence for every shared
trade. Such an envelope can be loose. Failure to certify a sign under it is
not proof that both endpoints are attainable market histories or that no
causal signal can work. Evidence: `predictfun/research/pf_keep_pull_timing_audit.py:14`,
`:67`, `:99`, `:108` and `:188`; result `timing_audit.txt:13`.

## Review findings, ranked by severity

### P2 — the removed-value headline mixes populations

`predictfun/FINDINGS.md`'s new FILL-VS-CANCEL row says the source split describes
value removed by the replay. The headline and reading in the new handoff also
refer to approximately half of removed value. But the calculation at
`pf_keep_pull_timing_audit.py:191-202` selects **all fills on pulled orders**,
including fills before the baseline cancellation landing.

Independent saved-table reconciliation finds 45,698 such fills. Their net
loss-sign value is 1,352,450.187 cents = replay-removed 1,169,671.170 plus
replay-kept 182,779.017. Both scenarios' AFTER/BEFORE/AMBIGUOUS buckets sum to
the larger population, not just the removed subset. Under S1, 10,693 / 45,698
fills are ambiguous (23.4%); they carry 36.6% of gross absolute value on that
population. The published 635,139-cent ambiguous *net* total cannot be treated
as an exact share of baseline removed value without the intersection table.

The detailed report's Part 1 heading correctly says "the score's pulled
orders." Correct the higher-level summary or report the baseline-removed ×
scenario-ordering cross-tab. This does not reverse the scenario sign result.

### P2 — the source-linked timing evidence is not retained for independent replay

`link()` joins candidate legs by market/side/anchor and executable price, then
collapses them to counts/class/min-max intervals (`:108-121`). `link_day()`
reads current loader data and timing-envelope summaries (`:124-132`), while
`run()` persists only aggregate results (`:228-230`). The result receipt hashes
the original keep/pull tables, but not the newly read leg/timing inputs or a
per-fill linked ledger. The shared provenance also omits the clock-audit and
loader dependencies used by this audit.

Consequently this review can check the formulas, original baseline and
aggregate conservation, but cannot reproduce the real S1/S2 linked intervals
from a saved audit ledger. "Every fill linked" means candidate-anchor coverage;
it is not a unique venue match identity or proof of the exact contributing leg.
Retain candidate trade/leg identities, interval provenance, inclusion statuses
and input hashes with the next evidence artifact. No raw-tape rerun was launched
to paper over this limitation.

## Verification and prior findings

Bounded `verify_timing.py` check passed in **2.32 s / 383 MiB** through
`PF_SLOT=codex predictfun/research/run_light.sh`. It verifies committed artifact
copies and recorded source/table hashes, reruns the pinned synthetic selftest,
and tests hand-computed cancellation/END boundaries and interval extrema. A
known-bad END exclusion is refused.

On all five days it reproduces collapsed per-fill contributions, the original
selection excess and score improvement; reconciles populations and all-fills
versus removed/kept values; verifies the reported S1/S2 aggregate conservation
and sign predicates. Current small timing-summary inputs equal the reported
envelope values; their hashes are recorded here, without claiming they were
frozen by the producer. Real-data linking/S1/S2 extrema were **not** rebuilt.

The prior request to inspect source timing has been addressed as a diagnostic,
with the retention limitation above. The clock/quantity qualification and
withdrawal of near-fill exclusion as an economic repair are recorded. The
old accounting fixes remain intact. No training, raw tape, new live capture,
strategy change, untouched validation or profitability claim in this review.

Evidence: `receipt_v1.json`, `verification.log`, `verify_timing.py`.

## One next priority

**Establish an order-level execution-ground-truth acceptance test, starting
with verification of the capture's event semantics.** Before relying on A4b
or any millisecond field, demonstrate that the evidence identifies the match
event or bounds it tightly enough relative to effective cancellation. Preserve
order/trade identities, raw event type/status, source timestamp, local receipt
time, placement/cancel requests and acknowledgments; retain unknowns and
unfilled orders. Then reconcile actual fills and remaining quantities with the
research replay on those same order lifecycles.

The immediate deliverable is a versioned example trace and a checker that
accepts genuine matching-event evidence and refuses a delayed status timestamp
substituted for it. If current captures cannot supply that, identify the
missing field/source before commissioning another capture; simply recording
more status messages with milliseconds is not sufficient. This gate belongs
to the development/execution owners. No live trading or deployment is part of
this review. Model tuning remains downstream of trustworthy economic labels.
