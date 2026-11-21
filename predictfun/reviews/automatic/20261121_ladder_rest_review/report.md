# LADDER_REST_INVENTORY_v1 — combined review

2026-11-21. Pinned target **869fa85ffc158b070e63c1b89af4cea3c82c3abf**.
Range: a8f863af182989ee9696080458a5bb489aa1e521..869fa85ffc158b070e63c1b89af4cea3c82c3abf.
Reviewed the 47d2eec1 freeze, 42411ab3 checker supersession and 869fa85f results.
Source was inspected and exercised in detached worktree
`/tmp/pf-lr-review-869fa85f`; development code and outputs were not changed.

**Disposition: accept the saved execution/accounting baseline for the 19
complete markets, conditional on the declared historical replay and M50 timing.
Do not yet use the affected actionability, sweep and state-map diagnostics to
choose an optimization.** Correct those diagnostics in the next combined
measurement round; the evidence does not require rerunning this baseline.
Passing implementation checks does not demonstrate strategy profitability.

## Accepted evidence and its limits

- Independently reconciled saved fills to **-209,557.708650 c**, **96,991.071133
  filled shares**, on 19 markets. Every complete market loses. Dates have
  7/5/5/2 complete markets out of 24 selected each: this is consumed, selected
  development data, not four complete validation days.
- The receipt's edge +39,326.191 c, +50ms markout -91,748.009 c and +1s markout
  -203,083.922 c reproduce on their respective valid fill populations. On the
  identical common subset of **7,796 fills / 83,164.941 shares**, edge is
  **+35,625.169 c**, +50ms **-65,535.823 c**, and +1s **-161,824.281 c**.
  Thus the adverse movement observation survives equal-population comparison.
- Fills before any cancel request account for 8,647 fills / 92,497.573 shares,
  with +1s marked value -173,618.203 c on 8,603 valid fills. Cancel-pending
  fills account for 559 fills / 4,493.498 shares, with +1s value -29,465.719 c
  on 558 valid fills. This is a factual timing split, not a zero-latency
  counterfactual or a causal bound on improving the strategy.
- Recomputed the minimum-hash market selection and eight pilot members from
  the committed parent population and all four hash-verified registries.
  Saved membership/statuses and receipt copies agree. The 54 missing-endpoint
  and 23 stale-path cases remain excluded with explicit statuses, no backfill.
  Their coverage diagnostics reproduce from the saved diagnostic records:
  median endpoint mark lag 11.6485s for missing endpoints, first large gap
  274.7s after opening for stale paths. Raw tapes were not reloaded in review.
- All **36 saved-artifact checks and 10 falsifiers** pass when re-executed
  against the pinned source and the saved outputs. Receipt hashes and copied
  measurement CSVs agree with the raw artifacts. The 93 legacy reproductions,
  677 feature points and 14,968 reconstruction checks are saved run evidence;
  this review did not replay those real tapes. All 14,968 saved feature rows
  have status ok; fair_edge is NaN as declared.

Coverage is the largest population limitation: only 19/96 contracts support
the full endpoint. Requiring a valid two-sided book near expiry can select on
market behavior, not merely missing files (`pf_stop_ev_accounting.py:247-255`,
`pf_queue_place_test.py:65-89`). The code preserves this limitation honestly.
Do not extrapolate the loss distribution, average performance or bins to the
77 excluded markets, and do not weaken the endpoint/gap checks to increase n.

## Ranked consolidated findings

BLOCKING below applies to the specified diagnostic/optimization claim, not
to the independently reconciled total ledger. Bundle repairs and the next
measurement in one round. NONBLOCKING items do not hold up that measurement.

### 1. P1 — BLOCKING actionability inference: earliest-cancel boundary omits pending cancellation

`pf_ladder_rest.py:752-767` calls first-live-grid +37ms the earliest cancel
effect and partitions value into locked/later; `:977-980` carries it into
severe timelines. But `pf_branch_ladder.py:137-153` applies structural requests
to post-pending generations after their submission tick, and the reused
`pf_branch_engine.py:142-149` permits effect
`max(request+37, submit+95)` before activation has been observed.

Independent positive/refusal control: post at 0, activation 58, received
distance signal at tick 50, request 50, effect **95**. A 100ms fill is prevented;
without the request it fills 10 shares. The current diagnostic calls it locked
until **137**. In the real saved ledgers **235 fill rows / 93 generations /
2,175.724 shares** lie in [submit+95, submit+137). Their summed END value is
**+32,481.240 c**: this error does not explain away the net loss and correcting
the boundary does not establish a beneficial cancellation signal.

**Acceptance:** retain separate named boundaries for the first permissible
pending request, first live policy review, and a signal-specific effective
request. Recompute affected tables/timelines from saved fills with the engine's
strict removal boundary. A fill after the earliest theoretical effect is not
automatically avoidable: an eligible observed signal is still required.

### 2. P1 — BLOCKING sweep attribution: timestamp groups are not public-trade events

`pf_ladder_rest.py:736-757` and `:782-789` classify a sweep when multiple gids
fill at the same side/time, ignoring each fill's public trade index. Only
**388** fill rows belong to public-trade events filling multiple own levels;
**2,132** are timestamp-group flagged, including **1,744** rows outside those
events. The engine's actual shared-volume accounting uses the public trade
index and passes the saved volume checks. This is an attribution defect,
not demonstrated overfilling by the engine.

`sweep_value` at `:1032-1033` also sums every fill of any generation that ever
had a sweep flag, rather than the flagged event's fills.

**Acceptance:** distinguish same-timestamp multi-level activity from a shared
public-trade event; use day/market/trade index for the latter and sum its actual
tranches. Keep an explicitly named ever-exposed-generation view if useful.
Do not infer a multi-level sweep cost from the existing table.

### 3. P2 — BLOCKING use of state-map fill rates as future-fill probabilities

`pf_ladder_rest.py:912-914` correctly limits a live state's value to observed
fills after its hypothetical cancellation effect. But `:940-943` carries the
generation's full-lifetime filled quantity into that state, and `:964` uses it
for fill_rate. For the 4,840 sampled live states, **2,074** generations ever
fill but only **1,702** have a fill in the future value window. At the first
review these counts are 2,074 versus 1,986. This mixes an outcome window with
a different probability denominator, misleading fill/value discrimination.

**Acceptance:** use the same future tranche window for quantity, any-fill,
value and severe labels at each stage; keep lifetime fill statistics separately
named. Treat the per-generation hash sample as such; use its recorded inclusion
weights only if estimating a live-decision-time population. Entry, first-review
and sampled-live cohorts stay separate. These are baseline-path future
contributions, not keep-vs-pull causal values or neutral no-cancel labels.

### 4. P2 — NONBLOCKING ledger result: churn/recovery labels omit relevant events

- First-review pulls count only `live` policy commands (`pf_ladder_rest.py:739-742,
  :769`), after structural requests have already changed the state to pending
  (`pf_branch_ladder.py:163`, `:191`). The report's **4/4,840** is the policy-only
  first-review count. **147 accepted generations** were requested for removal
  no later than their first eligible live tick: **143 structural, 4 policy**.
  Report these distinct populations and reasons before claiming early churn
  is nearly absent. No new simulation is needed.
- `pf_ladder_rest.py:873-892` treats any restriction-mode exit as a completed
  recovery. There are **252 exits to the recovery band and 23 direct reversals**;
  the latter are not a return to |known I| <=50. Separate recovery, reversal
  and end-censoring, and recompute the recovery summary from the mode log.

### 5. P2 — NONBLOCKING baseline; required before making this resume API operational

The resume envelope is incompletely reconciled with the executable object
(`pf_branch_ladder.py:308-332`). Independent falsifiers are accepted for:
next_k 2 ->11; envelope ident changed; runtime B.cap 200 ->999 in the pickle;
and a changed external path/mark plus matching envelope tape_id while the
executed object retains the original tape/mark. The unchanged positive control
passes and a changed fingerprinted field is refused.

**Acceptance:** bind/validate next_k against saved tick/acted phase, envelope
ident/tape/mark against the executed object, and executable q0/cap/levels/inv/
own_cross against settings. Either fix before use or explicitly disable saved
branch resume for the next round. The baseline runner creates a fresh branch
per market (`pf_ladder_rest.py:402-416`); this defect did not alter these runs.

### 6. P2 — NONBLOCKING results: selftest guard fixtures depend on repository stage

The unmodified pinned selftest fails now at `pf_ladder_rest_selftest.py:353-365`:
it assumes the module is uncommitted and the pilot projection absent. Both
assumptions cease to hold after the run; its refusal sentinel is reached.
The first review log retains this failure. Isolating those two negative fixture
states in the review process makes the same suite pass; production files and
checks were unchanged. The saved pre-freeze mutation logs show client 24/24 and
runner 20/20, but were not rerun as a new mutation campaign.

**Acceptance:** make these fixtures explicitly provide uncommitted/no-projection
conditions and include their corresponding allowed conditions. This is an
instrument repair in the same round, not grounds to discard reconciled P&L.

### 7. P2 — NONBLOCKING arithmetic; constrain interpretation of the loss decomposition

`NEW_FILLS_DOMINATE` (`pf_ladder_rest.py:797-808`, `:1072-1074`) means fills
within a minute marked to its boundary dominate inventory carried into that
minute. It includes up to almost a minute of post-fill inventory movement.
It does not show that inventory management is unimportant or isolate immediate
adverse selection. The actual short markouts provide the latter evidence.

Likewise, per-generation END tails (`:751`, `:774-775`, `:1081`) attribute large
positive/negative values to opposite legs even after net inventory is flat.
A 100-share buy at 50 and sell at 51 has +100c joint value at any mark; at mark
80 the ask leg is -2,900c and classified severe. Keep the declared END table,
but do not read its -1,884,949.554c severe contribution as an independently
removable tail cost. Net-market loss concentration is 41.49% in the worst five
of 19 losing markets. Analyze short markouts and net equity alongside it.

### 8. P3 — NONBLOCKING; small measurement/presentation corrections can accompany the batch

- `part_rows` and inventory summaries use 50ms samples, not exact event
  durations (`pf_ladder_rest.py:851-865`, `:887-895`). Exact both-side active
  duration from saved activation/removal/fill intervals is **40.6981%** versus
  **40.3055%** reported. Here opening wait is zero, no inventory peak is missed,
  and maximum per-market mean-|I| difference is 0.06673 shares. Label the grid
  estimates or use event durations; they do not change the broad diagnosis.
- `tools/fill_ladder.py:103` formats any non-value column containing “share” as
  a percentage: the count 941 appears as **94100.0%**. The underlying count and
  v2 check are correct. Use an explicit count/percentage format map.

## Prior items and checker supersession

The planned multi-order client, aggregate reservations, per-gid structural
flags, fixed sampling and reconstruction are implemented. Tick order is
trades/notices/invariants/structural/mode/alternating sides, with reservations
recomputed between commands. The engine source is unchanged across this range.
Same-price/own-cross constraints, queued partials and shared public volume
are covered by the bounded controls and saved-ledger checks. Saved source
hashes, membership, actual quantities and common-mark accounting reconcile.

The v1 -> v2 checker diff changes CSV parsing to round_trip, adds its precision
selftest, includes the v2 source in the committed check, changes the output
names and documents supersession. No economic threshold or check predicate
was relaxed. The one severe-count discrepancy is correctly resolved. Other
diagnostic semantics above were outside those passing checks.

## ONE next priority: RECEIPT-LEAD-TIME-AUDIT-v1, with the corrections bundled

**Question:** on this default-post baseline, can information received before
an effective cancellation distinguish bad future fills from useful fills?
This is the missing evidence needed to choose a guard, price or entry change.
Most shares filled before any request; merely speeding up already-requested
cancels is not yet an explanation of the main loss bucket.

Development should freeze one measurement package, repair the affected
diagnostics above from the saved evidence, and answer this question on the
**same 19 complete markets / 5,288 submitted generations**, retaining all 96
membership/status rows. Reuse the existing entry/first/hash-live snapshots
and source identities; reconstruct causal inputs on the same consumed tapes
only where necessary. Keep each snapshot kind separate and retain no-fill
cases. Do not select only severe fills or pool overlapping state labels as P&L.

Use **+1s fill markout as the primary local economic outcome**, with +50ms/
+100ms and END as explicitly separate diagnostics. At live/pending decisions,
associate only observed future tranches at or after the engine-consistent
effect time; separate already-acquired inventory and later movement. Report
future-fill quantities and missing-horizon statuses on that same population.
Do not convert a censored horizon into a zero label.

Use the already declared dictionary and Oct24-25 bin edges, applied unchanged
to Oct26-27 (all consumed). For each populated state bin report opportunity
counts, quantities, positive/negative local contributions and side/day
consistency. Preserve receipt timestamps and show whether the relevant
observation precedes a feasible command/effect, distinguishing posting latency,
resting exposure, cancel-pending fills and subsequent inventory movement.
Audit a bounded sample of those reconstructed inputs with future-only
perturbations and delivered controls. M50 remains the declared timing
assumption; do not claim an observed real exchange fill timestamp from it.

This diagnostic may identify one candidate hypothesis; it must not claim
guard profit, an oracle avoided-loss total, or a new trained model. A later
candidate comparison must price lost good fills, queue/re-entry and inventory
effects. No broad model/threshold search or automatic longer-rest change.

Budget this package at **at most 0.5 aggregate core-hours, within the unused
existing allocation**, with existing memory/slot limits, one combined receipt
and no intermediate repair reviews. No baseline replay is needed for the
measurement corrections. Disable unused unsafe resume if fixing it would
delay this evidence. **No further USER permission is needed for this bounded
consumed-data measurement.** More days, untouched validation, or expanded
resource scope require a separate USER decision.

## Review checks and limitations

Evidence: [reproduce.py](reproduce.py), [evidence.json](evidence.json),
[diagnostics.py](diagnostics.py), [diagnostics.json](diagnostics.json) and logs
in this directory. All research checks used PF_SLOT=codex and absolute script
paths through the repository's run_light.sh. Successful combined check used
20.44 CPU-user seconds, peak 510 MiB; subsequent saved-data/registry checks
were below one second each. No fit, full label build, real-tape strategy
replay or untouched-day access occurred.

The initial unmodified selftest failure is retained. A separate first attempt
of the reviewer diagnostics script had a missing closing bracket and produced
no measurement; its log is retained, and the corrected script passed. Review
scripts modify no development implementation or receipt. Raw tape consistency
beyond committed identities and saved ledger evidence was not independently
remeasured. Full market-impact realism and the M50 execution-time assumption
remain outside this implementation review.
