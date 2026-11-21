# PRESSURE-ACTION-ATTRIBUTION-v1 — combined independent review

2026-11-21, Codex. **Target: 54eab5455dd4c6e61c79d3543e3dd8a28e801ed0**.
Range: `7fca0db4cf34950936de23beaa8389f868ffb3a0..54eab5455dd4c6e61c79d3543e3dd8a28e801ed0`.
Reviewed freeze `8f618ea6`, checker supersession `d4b843c3`, and the pinned
results, using `/tmp/pf-ablation-review-54eab545`. Implementation and frozen
receipts were not changed. The other session's working changes were preserved.

**Accept this as a consumed-day action-attribution experiment. No blocking
finding or required replay.** Cancellation contributes more than withholding
alone, both increments help when added to the other component, and their
interaction is positive. All four policies still lose. The prior re-entry
checker finding is closed in this receipt.

Population, as of the receipt at **2026-11-21 20:26Z**: **19 complete markets**,
7/5/5/2 on Oct 24/25/26/27. Of the original 96, 54 missing-endpoint and 23
stale-future-book statuses remain excluded from paired economics. These are
four consumed development days, with signal/edges already selected using
these markets. No untouched validation, day interval, promotion or
profitability conclusion follows from these tests.

## What the evidence establishes

| Policy | END value, cents | Eligible +1 s value, cents | +1 s / valid share, cents | Mean exact both-side active time |
|---|---:|---:|---:|---:|
| BASELINE_00 | −209,557.7 | −203,083.9 | −2.102 | 40.70% |
| PULL_ONLY_10 | −132,118.6 | −113,644.6 | −1.331 | 42.03% |
| WITHHOLD_ONLY_01 | −191,208.9 | −192,679.1 | −2.000 | 39.79% |
| FULL_11 | −86,109.6 | −85,164.3 | −1.077 | 40.19% |

| Contrast | END increment, cents | Markets better / worse / equal |
|---|---:|---:|
| Cancellation alone, R10−R00 | +77,439.1 | 17 / 2 / 0 |
| Withholding alone, R01−R00 | +18,348.8 | 14 / 1 / 4 |
| Cancellation given withholding, R11−R01 | +105,099.3 | 18 / 1 / 0 |
| Withholding given cancellation, R11−R10 | +46,009.0 | 14 / 5 / 0 |
| Interaction, R11−R10−R01+R00 | +27,660.1 | — |

Each contrast is positive on all four days for END, eligible +1 s value,
and per-share value; the interaction is positive each day for END and +1 s.
The full END gain is **77,439.1 + 18,348.8 + 27,660.1 = 123,448.1 c**.
The interaction cannot be omitted. Top-three shares in the receipt use
**absolute market contrast mass**, not net gain; they are correctly defined.

These are complete-policy differences including re-entry, queue and inventory
feedback, not order-level avoided-loss estimates. Turning withholding off
allows posting into pressure, so PULL_ONLY's 553 pressure pulls and 111
first-review pulls are consistent with the declared arm. That does not justify
removing withholding: adding it to cancellation improves END by another
46,009.0 c. FULL has 433 pressure pulls and **25 / 4,422** first-review pulls
of any reason; the old near-universal first-review churn is not present in
this default-post ladder. There are still 3,613 structural requests, whose
purpose differs from the pressure guard.

The largest economic gap is the value of the exposure that remains:
FULL's +1 s value is **−85,164.3 c**, with **+55,130.7 c** positive and
**−140,295.0 c** negative contributions. Later movement from that horizon to
END is only **−1,416.8 c net**, plus **+471.5 c** late/missing-horizon value.
The arithmetic reconciles to END. Large offsetting later movements could
still exist; this net decomposition does not establish that inventory is
optimal. It does show that optimizing terminal inventory alone cannot
explain away the existing short-horizon loss. Positive execution edge
(+50,544.4 c) is followed by negative +50 ms value (−27,654.1 c).

## Ranked consolidated findings

1. **P3 — NONBLOCKING; defer or bundle cheaply: F does not check every
   reported lost/added component or the pooled +1 s contrast field.**
   `ablation_checks_v2.py:177–185` compares lost/added fill counts and END
   totals, but not quantities, signed END components, signed +1 s components,
   or the exported delta/gap fields. `:188–196` checks daily +1 s contrast
   values/classifications but omits its stored pooled `mo1s.total`.
   Adding 1,000 to `lost_q`, `lost_mo1s_neg`, or that pooled contrast total
   independently leaves family F passing; see `evidence.json`.
   **Verified workaround:** this review independently matched the fill
   multisets and recomputed **every field of all 76 lost/added rows**; every
   current value agrees. The arm ledgers and every stored predicate, including
   the pooled totals, also agree. Thus no current reading is invalidated.
   If reused in the next audit, extend these comparisons and add the three
   refusals in the same batch; otherwise retain this limitation. No separate
   repair round, label rebuild or replay is warranted.

No other issue materially invalidating this experiment was found across the
rule, engine integration, parts, controls, accounting and evidence identity.
The checker gap above is an acceptance-instrument limitation, not evidence
that the strategy's observed loss is a measurement error.

## Source and artifact review

**C/W gating and first-review bookkeeping.**
`pf_ladder_pressure_ablation.py:75–89` delegates to the frozen pressure
method only for the enabled component. With C off, it still records the
first live review before calling `LadderInventory.live`; with W off, slot
decisions call `LadderInventory.slot` directly. Neither disabled path reads
the pressure signal. The inherited pressure methods at
`pf_ladder_pressure.py:138–154` call inventory first, only overriding a
would-be KEEP/POST. Structural requests occur before policy callbacks
(`pf_branch_ladder.py:162–195`). No notice, reservation or timing change was
introduced. Current-tick cache selection and side sign are inherited from
the accepted parent. The four direct read-barrier tests, in addition to the
synthetic hand/random tapes, confirm disabled components cannot read signal.

All saved interventions match actual command arrays, current-side cached
float32 values/ages and client position after the +100 ms fill notice.
Pressure live pulls take effect at request +37 ms. All **14,033** non-fallback
first-live review entries across the three executed arms agree with the
queue observation times, including WITHHOLD_ONLY's disabled cancellation
branch. This observation bookkeeping has no policy effect.

**Controls and identity.** `:128–169` binds the frozen rule/source closure,
parent day record, and the original path/END mark before execution.
`pf_ladder_pressure_lag.parent_day:99–109` verifies parent committed record
bytes and gz/npz hashes. All receipt-copy, source, parent and current-output
hashes were independently checked. Every FULL ledger reproduces its accepted
parent: orders, fills, requests, value, fingerprint, guard logs and both queue
records. Additional comparisons of **152 arrays** (eight per market: commands,
decisions, venue/client inventory, reservations, cash, mid, nva) all match.
All four arms share the episode identity, tape id and mark. The inherited
baseline ledgers equal the original ladder on all 19 markets.

**Parts and membership.** `:106–110,139–145,209–234` implements a disjoint
8-market pilot and 11-market rest. The measurement takes their union and
rejects duplicate market keys, rather than counting the pilot twice.
`ablation_checks_v2.py:112–119` checks the complete baseline membership and
the pilot ordering. Oct 27 rest is DAY_EMPTY before parent/tape loading.
All eight part records agree with the manifest; 19 market records yield
76 arm rows and 76 contrast rows. The original incomplete statuses are
retained, not relabelled complete.

**Contrasts, accounting and lost/added.** `:178–205` uses the declared arm
ordering, strict daily signs, separate ratio-of-sums per-share metrics and
the correct interaction. Measurements use the shared `t <= END−1 s` cutoff
with valid-share denominators; missing marks are not counted as zero.
Independent calculations from all **76 fill ledgers** reproduce cash/END,
markouts and the END decomposition, exact inventory integrals, and exact
order-active interval unions/intersections. Lost/added multisets use
`(t, side, price, q)` with six-decimal keys, intentionally ignoring gid;
all current components and delta identities reproduce. They describe path
differences, not an avoided-trade causal effect or exchange-impact estimate.

**Prior P3 — closed.** `ablation_checks_v2.py:205–249` constructs next-same-side
rows from executed orders/queue records and checks all 17 exported data
fields, as well as the keys. Every field of all **986 current rows** matches.
The changed-next-price falsifier now fails H. The labels remain
activation-clock initial ahead and next same-side submission; a different
price/rung is not treated as a matched replacement or queue-cost estimate.
The frozen lag receipt was not edited.

**Checker supersession and tests.** The v1/v2 diff changes only the broken
tuple-keyed-dict corruption construction, the v2 committed-file membership,
documentation and output suffixes. No acceptance predicate was weakened.
The retained v1 rehearsal failed in its falsifier before the rest ran;
v2 was committed before its run. The current pinned selftest passes;
**17/17 checks and 11/11 falsifiers** reproduce. Saved mutation records were
rejudged against pinned source hashes, names and indices: control passes,
**20/20 caught**. Synthetic successes establish implementation checks, not
market profitability.

**Resources.** The write-once pilot projection is 0.309 core-hours for this
round and 1.849 cumulative, below 0.35/2.0 ceilings
(`tools/pilot_projection_ablation.py:35–52`). `run_day:137` requires it before
the rest. Recorded completed development use is **0.242586 core-hours**;
peak 1,226 MiB. Including available prior receipts/reviews and both review
attempts gives **1.798575 core-hours** before the negligible final accounting
job, leaving about **0.2014**. The receipt's 1.778 is correctly labelled
as-of its checks, rather than the final current balance. Resource figures
are the existing run_light recorded user-CPU convention. Do not reset this
cumulative budget when starting the next receipt.

## ONE next priority — PRESSURE-RESIDUAL-TIMING-AUDIT-v1

**Use saved FULL_11 evidence to distinguish loss during the cancellation
window from exposure the pressure rule did not flag in time.** This is one
targeted mechanism audit, not another threshold/feature/size candidate.
It directly determines whether the next change should address timing or
the danger signal/value mapping. There is no authorization here to change
the strategy or automatically promote a variant.

Preparation and acceptance contract for the development session:

1. Freeze the builder, exact input hashes, exhaustive timing categories,
   synthetic cases and computed reading rule before generating tables.
   Primary population: **all FULL_11 generations on the same 19 complete
   markets**, including refusals/no fills; preserve all 96 coverage statuses.
   Reuse saved order, command/request, fill/mid, cache and identity evidence.
   Baseline may provide the same descriptive table as context; do not pool
   policies or create another fitted candidate. No raw-tape load or replay.

2. Give each actual fill exactly one timing status, derived from engine
   event order. Separate (a) removal already requested but not effective,
   split pressure / inventory / structural; (b) no such pending request and
   a fill before the earliest feasible live pressure-cancel effect;
   (c) later exposure with no pressure ON early enough to remove it; and
   (d) non-evaluable/fallback or an apparent timely-ON action mismatch.
   Keep category precedence explicit. A timely-ON mismatch is investigated
   against actual callback eligibility and requests, not absorbed into
   “no signal.” Report signal-at-submission, first-live-review and latest
   feasible pre-fill cancel tick, with source age and actual actions.

   Use the native fill/activation/request/effect clocks. In particular,
   fills at a grid timestamp execute **before that grid's policy commands**;
   a same-timestamp request is not already pending for that fill. Removal
   is strict (`fill_t < remove`); live cancels have +37 ms and pending
   structural cancels have their separate engine rule. Derive feasible
   ticks from this ordering, not from the timestamp of a snapshot or from
   the first *future* callback. Any hypothetical earlier cancel is a timing
   eligibility diagnostic only; it is not a measured avoided loss.

3. For every status, by day/side/birth rung, report generations, no-fill
   generations, fills, shares and exposure duration; execution edge,
   +50/+100/+1,000 ms values, signed positive/negative contributions,
   valid quantities/censors, and END decomposition. Include all values,
   not only losing fills. Keep later inventory movement separate. Count
   partial fills once, retain generation identity, and show concentration
   by market so one path cannot masquerade as broad evidence. This is a
   diagnostic partition, not a fill-selected training population or a
   conditional-profit estimator.

4. Reconcile category sums exactly to the accepted FULL totals and its
   raw fills; prove duplicate and missing membership refusals. Synthetic
   cases must cover no fill, repeated partial fills, first-live timing,
   fill/request ties, fill/effect boundary, pressure switching ON after
   the last feasible cancel tick, absent/non-ok input, and all three
   removal reasons. If using the affected exported fields, include the
   small checker extension above in this batch. No full rerun is needed.

5. Predeclare a descriptive reading: compare the shares of **gross negative
   eligible +1 s value** in the timing-limited categories versus later
   unflagged exposure, with their offsetting positive value and quantities
   alongside. Report both fractions and all four day values; if one exceeds
   half on every day, call it the dominant observed mechanism; otherwise
   MIXED/THIN. Material unexplained event/action mismatches make this audit
   fail rather than manufacturing a strategy diagnosis. Do not call the
   partition predictive skill or infer a winning threshold from it.

**Budget and permission:** at most **0.12 additional core-hours** for the
whole batch (tests, audit and checker), and still at most **2.0 cumulative**.
Use the saved-input pilot to verify the bound before the full table; stop
explicitly if it cannot fit. This leaves room for review. **No additional
USER decision is needed within this existing scope.** More compute, another
replay/candidate outside this contract, new label days or untouched validation
requires the USER's decision. Do not quietly reduce the population to fit.

Largest evidenced limitation: even the best of these four *observed* policies
has materially negative short-horizon exposure value, and this ablation does
not locate its residual timing/signal cause. Its gains are conditional on
unchanged external flow, M50 matching, the modelled Binance delay, zero costs
and limited 19/96 coverage. The targeted saved-evidence audit above addresses
the first uncertainty efficiently; it cannot establish real-exchange impact
or out-of-sample profitability.

## Review evidence

- `reproduce.py`, `reproduce.log`, `evidence.json`: pinned controls, artifact
  hashes, independent ledgers/intervals, field checks and coverage probes.
- `independent_arms.csv`, `independent_lost_added.csv`: direct reconciliations.
- `resources.py`, `resources.json`, `resources.log`: cumulative usage and
  rejudged saved mutation records.
- `reproduce_attempt1.log` is retained: the first review script completed
  the saved-data checks but its added read-barrier fixture omitted the
  inventory configuration. Only that synthetic stub was completed; the
  rerun passed. No development source or result was changed.

All numerical verification ran through **PF_SLOT=codex run_light.sh**, on
absolute script paths. No training, real-tape replay, new label build or
untouched day was run. The report and review-owned state are committed
locally only, with `Predictfun-Review-Only: true`.
