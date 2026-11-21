# PRESSURE-RESIDUAL-TIMING-AUDIT-v1 — combined independent review

2026-11-21, Codex. **Target: 2063d31a91ab7e6124d7b8b7b7ec5df01ea66711**.
Range: `e40bd01d89c6446b1e9afdd15ed4e21ae355cd58..2063d31a91ab7e6124d7b8b7b7ec5df01ea66711`.
Reviewed freeze `a10eeea3e5dc4212c5ffc0a2c9e581b31201abfe` and the results
from `/tmp/pf-residual-review-2063d31a`. No implementation or frozen receipt
was edited. Verification used saved evidence and synthetic fixtures only.

**Accept the timing partition and its declared UNFLAGGED_EXPOSURE_DOMINANT
reading for these consumed days. No blocking finding or engine replay is
required.** Two supporting-instrument findings are bundled below; neither
changes the status counts, economics or reading. The next experiment should
test discrimination within the pressure rule's OFF region, rather than
assuming that faster cancellation alone solves the residual loss.

As of the development receipt at **2026-11-21 20:49Z**, the population is
**19 complete markets**, 7/5/5/2 on Oct 24/25/26/27, containing **7,395 fills
and 4,811 generations**. The original 96-market population retains 54
missing-endpoint and 23 stale-future-book exclusions upstream. All four
days are consumed adaptive development days, not untouched validation.
There are no day-level uncertainty intervals or profitability claims.

## Result and its practical meaning

The independent reconstruction agrees on every status:

| Status | Fills | Eligible +1 s value, cents | Gross negative +1 s, cents |
|---|---:|---:|---:|
| pending_pressure | 75 | −6,186.2 | −6,186.2 |
| pending_inventory | 5 | +12.1 | −31.2 |
| pending_structural | 321 | −9,321.2 | −11,037.1 |
| before_feasible_effect | 422 | −15,912.0 | −18,988.7 |
| unflagged_later | 6,572 | −53,757.0 | −104,051.8 |
| nonevaluable / event_or_action_mismatch | 0 / 0 | 0 | 0 |

Unflagged later exposure accounts for **74.17%** of gross negative eligible
+1 s value, and **71.67 / 73.73 / 80.46 / 64.83%** by day. All 19 markets
contribute negative value in this category; its top-three concentration is
about 25%. Thus the frozen dominance rule is met without one market deciding
the pooled result. All 4,811 generations are retained: 1,790 filled, 2,745
no-fill and 276 refused.

**Dominant loss mass is not dominant loss rate.** Unflagged exposure also
carries **90.34% of eligible filled shares** and **91.23% of positive +1 s
value**. It contains **+50,294.8 c** of positive value against −104,051.8 c
negative, or **−0.753 c per eligible share net**. Timing-limited fills are
only 9.66% of eligible shares but lose **−4.113 c/share net**. These are
descriptive ratios, not policy effects. The result identifies where most
remaining loss sits; it does not justify cancelling all OFF-region exposure,
establish that the missing danger was predictable, or quantify what zero
latency would save.

The END accounting remains **−85,164.3 c** eligible +1 s, **−1,416.8 c** net
later movement and **+471.5 c** late/missing, summing to **−86,109.6 c**.
Those small net later-movement totals hide substantial offsets: unflagged
fills contribute −69,006.5 c later movement, while other timing categories
offset much of it. Pending-structural fills indeed have −9,321.2 c at +1 s
but +29,210.1 c later movement. That is a horizon-dependent accounting fact,
not evidence that structural cancellation is wrong or that removing those
fills would improve END. The policies' inventory and subsequent actions
would also change.

## Consolidated ranked findings

1. **P2 — NONBLOCKING for the accepted partition; correct before using the
   pending-fill diagnostic columns for another decision.**
   `pf_ladder_pressure_residual.py:103–110` initializes feasible/evaluable/ON
   counts to zero and returns immediately for pending requests. Only
   `:114–118` actually computes those counts and the last feasible tick.
   Consequently **391 pending fills** with real feasible callbacks report
   zero: 313 structural, 73 pressure and 5 inventory. Their last-tick fields
   are blank as well. A hand case with 18 earlier feasible KEEP callbacks
   returns `pending_structural, n_feasible=0`; evidence is saved below.
   The checker repeats this early return at `residual_checks.py:90–96`, so
   agreement between the two implementations does not validate those zeros.
   The receipt's pending-group mean feasible/evaluable counts should not be
   read as evidence that no earlier decision opportunity existed.

   **Bounded workaround verified:** independently computed feasible ticks
   even for pending fills and found **zero timely evaluable ON ticks across
   all 7,395 fills**. The status precedence and every current status/value
   remain unchanged. Treat the skipped diagnostic counts as uncomputed,
   or compute them before the early returns in a superseding saved-data
   output; add a pending request with earlier callbacks to the fixture.
   No tape load/replay or repair-only round is required. Also label T1 as
   post-fill when appropriate: **242 fills precede their recorded T1**.
   Its signal is then a retrospective diagnostic, not an available input
   for that fill. Every recorded T1 in these data equals the native first
   live grid; there is no observed delayed-T1 misclassification.

2. **P3 — NONBLOCKING; retain or extend in the next batch: the checker does
   not cover all the supporting fields its header describes.**
   `residual_checks.py:137–162` verifies statuses, their three counters, core
   values, selected table sums and the dominance fractions/answer. It does
   not independently compare signal/action diagnostics, generation-status
   aggregates, concentration, or the positive-value/quantity components of
   the reading. It also omits several exposure, +50/+100 ms and censor
   table fields. Changing T1_x, the stored unflagged concentration, every
   generation-status count, or the stored unflagged positive total each
   leaves family F passing. File hashes protect the saved bytes, not the
   correctness of a coherently generated derived table.

   **Verified workaround:** this review independently checked raw fill and
   generation membership, statuses and values, then reconciled all current
   diagnostic fields to the inspected transform, all table fields, complete
   generation-status tables, concentration and every reading component.
   They agree, subject to finding 1's uncomputed counters. If the next round
   uses a supporting field, add the corresponding comparison and corruption
   case then; otherwise preserve this limitation. This is not a reason to
   repeat the experiment or claim its main reading is unreliable.

**Prior e40bd01d P3 remains explicitly deferred:** the ablation checker's
lost/added components and pooled contrast field are not used here. The
accepted FULL arm aggregates are used and hash-bound; this review again
checks them against fills. No earlier frozen receipt was changed.

## Review of clocks, labels and instruments

**Clocks and precedence.** `classify_fill:100–123` uses `req < fill_t`, so a
request at the fill's timestamp is not already pending. Actual fills satisfy
`act <= fill_t < remove`. The independent review derived the latest possible
grid as `50 * floor((fill_t−37)/50)` and intersected it with the order's live
callbacks, rather than using the production classifier. On the current
data it reproduces all statuses. All actual requests were reconciled with
the engine effect: +37 ms for a live request, otherwise
`max(request+37, submit+95)`. First-live callback times were checked against
`ceil(activation/50)*50`. No later callback was mistaken for an earlier
available cancellation opportunity in this population.

Pending-reason precedence is consistent with the question: an already
requested removal does not become “unflagged” merely because its latest
signal is OFF. On the other fills, feasible callbacks exclude any tick whose
+37 ms effect would arrive too late. Evaluability requires both a non-fallback
callback and an OK own-side cache status. An evaluable ON early enough is a
mismatch, never silently unflagged. The separate all-fill ON check above
also protects against an earlier missed ON being hidden by pending precedence.

**Reasons.** `request_reason:82–89` maps the engine's structural tag directly;
pressure/inventory require exactly one live PULL command at that gid's
request time. All current requested orders reconcile: 3,613 structural,
84 inventory and 433 pressure requests. No unknown reason was absorbed.

**Values and exact once.** `fill_values:129–134` uses each fill's own signed
quantity, price and saved markouts. The END−1 s cutoff is inclusive, with
missing horizons kept out of the eligible quantity denominator and in late
END accounting. Raw mids contain finite values or explicit nulls; there is
no NaN-as-zero ambiguity in this saved input. Independent row calculations
reproduce every fill's values and identity, including all partial fills.
There are **37 ineligible/censored +1 s fill rows**, kept as statuses rather
than silently dropped. The common END/mark and accepted arm totals agree.

`reconcile:157–168` checks each market's full fill-index and generation-id
membership and aggregate sums; checker B additionally checks global keys.
The independent review checks identities, fill price/side/gid, activation
and removal bounds, quantity conservation, and no missing mids. No fill
or generation is duplicated or lost. No-fill generations remain represented.

**Tables and reading.** `table:171–187` counts each generation once *within
each fill status*. A generation can appear in more than one status; its
full live time is then repeated. The receipt discloses this, so the “all”
exposure total is not a disjoint time allocation or a denominator for a
status hazard rate. Refused/no-fill generations are in the separate census.
Signed values and eligible quantities are additive; concentration uses
market gross negative value. `reading:190–204` gives mismatches priority,
uses the frozen 30-negative-fill daily minimum and requires >50% on every
day. The zero-nonevaluable result and all reading fields reproduce.

**Identity and execution.** `ablation_part:208–215`, `audit:233–250` and the
inherited `parent_day` bind committed part/day records to their raw hashes.
This review independently checked every gz/npz hash, source hash, edge file,
manifest and accepted arms.csv hash, receipt-copy byte equality, current
output hashes, and exact 8-pilot/19-full market membership. No research source
or checker changed between freeze and results. The audit loads saved
evidence only. The pilot/rehearsal/projection/full order is present in the
logs; the launcher refuses full without the write-once projection. The
audit function itself relies on that launcher for the resource gate, which
is sufficient for this recorded bounded run, not a general unguarded-CLI
authorization.

**Checks.** The pinned synthetic selftest passes. All **10 checks and seven
falsifiers** reproduce. The pilot has nine true checks with full-population
membership deliberately false; its `all_ok` and `falsifier_ok` are false as
saved, while all seven named corruptions are refused. This is the expected
rehearsal, not a full acceptance PASS. The saved mutation records were
rejudged against pinned source hashes, names and indices: control PASS,
**19/19 caught**. These implementation controls make no profitability claim.

**Resources.** The saved-input projection is 0.038436 core-hours, below 0.12,
with 1.837019 cumulative projected use below 2.0. Completed development logs
record **0.009617 core-hours**, peak 334 MiB. With available earlier receipts
and review logs, use is **1.811956 core-hours**, leaving approximately
**0.1880**, before the negligible accounting/commit verification jobs.
The receipt's 1.806 is explicitly as-of its checks, not the final balance.
Accounting follows the existing run_light recorded user-CPU convention.

## ONE next priority — POSITIVE-PRESSURE-GUARD-v1

Run **one declared economic challenger** to determine whether weaker,
same-direction Binance pressure can remove additional harmful exposure
without sacrificing too much useful trading. This is a test of broader
signal coverage, not a conclusion that a lower threshold will work.

- **One change only:** replace the current finite, positive, side-specific
  q5 predicate by **finite float32 `adv_bn_r250 > 0`** on both sides, at the
  same current tick. Zero ties remain OFF; non-OK input follows the unchanged
  fallback. Keep both pressure PULL and WAIT enabled, structural/inventory
  precedence, 58/37/100 ms clocks, two nominal 100-share levels, the full
  paths, sizing, no timer and all accounting unchanged. Count this as one
  new adaptive consumed-day candidate, frozen before any result. Do not
  try intermediate thresholds or choose one from the output.

- **Arms and population:** the same 19 complete markets and all 96 coverage
  statuses. Rerun Q5_CONTROL alongside the challenger on each market and
  require ledger/command/queue equality with accepted FULL_11. Reuse the
  original ladder only as a context baseline. Use the hash-bound current
  cache; no feature rebuild or new clock convention. Respect unchanged
  external-flow assumptions. No training, new days or validation claim.

- **Primary comparison:** challenger minus Q5_CONTROL, per market and day,
  for eligible +1 s **total** and END value; report per-share ratios with
  denominators separately. Preserve positive and negative contributions,
  lost/added fills, censoring, exact participation, inventory integrals,
  reducing-side suppression, structural/pressure pulls, first-review churn
  and generations. Report absolute values and all_arms_negative beside any
  improvements. The wider guard may reduce both harmful and good exposure;
  lower quantity or a better per-share ratio alone is not success.

- **Incremental local control in the same batch:** use the saved neutral
  baseline's one-live-snapshot-per-generation population, with its actual
  KEEP callback and unchanged inventory/structural eligibility, including
  no-future-fill rows. Restrict to finite, currently q5-OFF states. The
  challenger-only mask is `x > 0` within that population, decided from the
  snapshot's own clock. Reuse the accepted action-relative post-cancel
  window, +1 s labels, censoring, valid quantities and 500-draw matched
  design/seed/support minima. Match day/market/side, remaining size, age,
  known inventory sign and B>=200 as in the pressure receipt. Predeclare
  the mask and null; do not use fill-conditioned `k_last` or the future T1
  diagnostic as inputs. This checks incremental selection, not full-policy
  P&L significance. Thin support must remain INCONCLUSIVE.

- **Reading and safeguards:** freeze and compute daily/pooled signs for
  the two primary economic contrasts, the incremental matched-screen
  result, participation and churn. A positive economic reading requires
  both END and eligible +1 s totals to improve on **each** day; report the
  local screen separately. Any negative day remains MIXED/FAIL, and positive
  relative results never become promotion or validation. If the challenger
  merely suppresses exposure, fails the incremental screen, or worsens
  economics, record that result and stop this candidate; no same-round
  threshold search. Use synthetic boundary/side/non-OK/delegation controls,
  checked reproduction and a falsifier before the real run.

- **One combined batch:** fold findings 1–2 in only where their fields are
  reused; otherwise mark them explicitly uncomputed/deferred. If exporting
  lost/added fields, include the previously deferred ablation-field checks.
  No fix-only receipt or rerun of accepted evidence is needed.

**Budget:** at most **0.14 additional core-hours for the whole development
round**, with the original **2.0 cumulative ceiling** still binding. Use
all seven Oct 24 markets as the fixed first-date pilot, reuse those results,
then run the other three dates only if the write-once whole-round projection
fits. This avoids reloading/replaying the pilot date for a rest partition.
Reuse caches and do not rebuild the earlier feature/causality experiment.
If the measured projection cannot fit, stop before the other dates and seek
the USER's resource decision; do not trim controls/population or relax the
ceiling. About 0.048 core-hours remain for review at the proposed maximum.

**No additional USER decision is needed within this bounded continuing
experiment scope.** More compute than remains, new data days, untouched
validation or a broader model/threshold search requires the USER. This
review authorizes no implementation changes in the review thread.

Largest evidenced limitation: the existing q5 rule leaves a broad, mixed
OFF-region population with net adverse value; the partition cannot tell
which part is discriminable using information available at action time.
The single broader-pressure challenger above tests one concrete explanation
efficiently, while preserving the useful cancellation behavior already found.

## Evidence and ownership

`reproduce.py`, `reproduce.log`, `evidence.json`, `independent_status.csv`
and `independent_tick_counts.csv` contain the pinned controls, independent
classification/value checks and bounded diagnostic counterexamples.
`resources.py/json/log` contain budget and mutation-record verification.
All numerical checks ran through **PF_SLOT=codex run_light.sh** with absolute
script paths. No model fit, real-tape replay, new labels or untouched day was
run during review. Only review artifacts and review-owned state hunks are
committed locally with `Predictfun-Review-Only: true`; no push.
