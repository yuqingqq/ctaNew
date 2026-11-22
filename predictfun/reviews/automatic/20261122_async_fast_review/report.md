# ASYNC-EXECUTION-SENSITIVITY-v1 — combined review

2026-11-22. Full pinned target: **85127b3dfbdfb8689159fe8b9e76b447324b7cfc**.
Range: **b8ab077ab8a33c71778942279df9e8445acfe3e7..85127b3dfbdfb8689159fe8b9e76b447324b7cfc**.
Reviewed `7b55fc24`, freeze `09418aff`, and results `85127b3d` in detached
checkout `/tmp/pf-af-review-85127b3d`. No implementation change, historical
replay, fitting, new day, push or external messaging. Development owns fixes.

## Decision

**Accept the numerical sensitivity result. No blocking finding or required
rerun.** The faster concurrent execution contract is implemented consistently
with its declaration. Its numerical result is `assumptions_do_not_rescue`:
every tested arm loses on every consumed day, and every combined economic
contrast is MIXED. Correct the interpretation below in the next combined
receipt; do not start a minor-fix-only round or another tuned speed/queue arm.

Population: the same **19 complete markets on consumed Oct 24–27**, as of
this target. Original coverage remains 19 complete / 54 missing endpoint /
23 stale future book out of 96. These are selected supported paths, not a
full-day return estimate or untouched validation.

| Arm | END, c | Eligible +1 s, c | Eligible +1 s, c/share |
|---|---:|---:|---:|
| CURRENT_BASELINE | -86,109.61 | -85,164.28 | -1.08 |
| FRONT_BASELINE | -65,937.09 | -61,812.39 | -0.53 |
| CURRENT_ASYNC_FAST10 | -68,319.95 | -78,485.90 | -0.98 |
| FRONT_ASYNC_FAST10 | -86,960.60 | -61,051.63 | -0.48 |

EXEC_AT_CURRENT improves END by 17,789.66 c but +1 s by only 6,678.38 c;
EXEC_AT_FRONT worsens END by 21,023.51 c despite a 760.77 c +1 s improvement.
QUEUE_AT_FAST improves +1 s by 17,434.27 c on all four days, but worsens END
by 18,640.65 c. The full per-day predicates reproduce exactly. This narrows
the next question to economic selection and quoting decisions under the
tested assumptions; it does not establish that existing information lacks
predictive power.

## Consolidated findings and disposition

1. **P2 — NONBLOCKING for accepting the numbers; amend the interpretation.**
   `predictfun/reviews/ladder_async_fast_20261122/report.md:43` calls
   `END - eligible_1s` the movement from +1 s to END. That difference includes
   the END contribution of late/unsupported +1 s fills. The actual identity,
   independently rebuilt from saved fills and marks, is:

   | Arm | Eligible +1 s | Subsequent movement on that cohort | Late/unsupported +1 s fills at END | END |
   |---|---:|---:|---:|---:|
   | CURRENT_BASELINE | -85,164.28 | -1,416.84 | +471.51 | -86,109.61 |
   | FRONT_BASELINE | -61,812.39 | -5,175.38 | +1,050.68 | -65,937.09 |
   | CURRENT_ASYNC_FAST10 | -78,485.90 | +9,177.89 | +988.06 | -68,319.95 |
   | FRONT_ASYNC_FAST10 | -61,051.63 | -26,694.24 | +785.26 | -86,960.60 |

   Thus the END-versus-+1 s caveat is warranted, but the components must be
   named correctly. Neither this identity nor the END class flip identifies
   an inventory-management defect; no inventory-policy ablation was run.

   Report `:72` also needs “under this combined execution profile.” Requests
   are 27 ms faster for live cancellation and activation is 48 ms faster,
   **conditional on a fixed submission/request time**. Complete feedback
   changes the request times and states. A normally accepted post first
   becomes reviewable at +50 ms under ASYNC versus +100 ms under BASELINE
   on this grid (`pf_ladder_async_fast.py:85-94,137-141`;
   `pf_branch_ladder.py:108-120,223-228`). Concurrency changes too. There is
   no paired same-action estimate of a pure 27–48 ms benefit.

   Finally, report `:88` attributes the failure to “decision-time information”
   not anticipating sweeps. The experiment tests the **fixed Q5 pressure
   rule's use of that information**, not all predictive uses of the inputs.
   Replace that conclusion with: “This fixed pressure ladder leaves large
   adverse through-price exposure even with the tested faster execution and
   hypothetical front priority. It remains unknown how much of that exposure
   can be removed by timely, selective decisions without sacrificing the
   good fills and subsequent queue value.” Keep the report's qualifications
   at `:74-77`. No numerical recomputation or replay is needed.

2. **P3 — NONBLOCKING; inherited checker coverage remains incomplete, with a
   verified workaround for this receipt.** `fa_checks.py:253-281` checks only
   selected arm/class fields, and `:284-293` checks aggregate contrast values
   and verdicts rather than the entire predicate object. In-memory changes
   to a class's `mo30s`, an arm's `both_exact`, and a per-day absolute-negative
   flag still pass their named families. These are the same class of gaps
   deferred in the previous review, not evidence of wrong actual results.

   **Workaround completed here:** all 95 arm rows, 152 class rows including
   long-horizon values/censoring, treatment rows, block rows and the full
   predicate object were checked against the saved evidence. Independent
   economic reconstruction covers cash, position, END, markouts and their
   denominators; helper reproduction supplements it for unchanged ancillary
   arm metrics. Actual exported fields and committed copies agree. Extend
   the affected comparisons with the demonstrated bad-value cases only when
   they are reused for a consequential reading. No cleanup gate now.

   Family F checks saved trade metadata, not a separately reloaded public
   tape. The direct trade-row lookup is correct at pinned source, and the
   review checks event consistency across arms and volume/fill invariants.
   Do not describe it as an independent reconstruction of raw market data.

No additional material issue found at the reviewed integration boundaries.
The two findings do not invalidate the negative result or justify rerunning it.

## Implementation and integration review

- **Profile and clocks:** `pf_ladder_async_fast.py:79-94` routes new orders to
  `add9` with actual activation, and overrides the request clock for both
  pending and live orders. Structural and policy requests dispatch through
  that method. Strict removal, inclusive activation and trades-before-grid-
  commands remain inherited. The engine swap at `:100-106` precedes every
  order/trade; effective ASYNC settings are recorded. Source closure did not
  change between freeze and results.
- **Slots and risk:** `:108-123,137-166` frees desired capacity after a request
  while reservations still use every unclosed order. Price locking and
  own-cross see retiring orders. Inherited notices and `reserve`
  (`pf_branch_ladder.py:111-129`) release risk only when it becomes known.
  Position bounds and the two-desired-generation bound run through the
  inherited tick plus each command. Side order still alternates. There is
  no early fill information, early ACK, atomic replacement, or exemption
  from the 200-share reservation cap.
- **Information:** the saved pressure cache, side, grid and receipt clocks
  are unchanged. `PressureGuard.value` uses the current grid under both
  profiles. Different policy histories can see different states; this is
  not earlier access to an external event. The synthetic late-frame control
  confirms the wrapper does not advance the received book.
- **Parity and reuse:** PARITY_BASELINE equals accepted pressure on 8/8
  pilot markets, including fingerprints, orders, fills, requests, value,
  guard log, command arrays and queue records. Both reused queue arms are
  bound by manifest record hashes and gzip/npz hashes (`:200-210,330-342`).
  All arms use the same market identity and terminal mark. New arms use their
  own fills, notices, inventory and reservations.
- **Contract checker:** `fa_checks.py:99-130` reconstructs activation/removal
  and, at every POST, known fills, still-unclosed risk, slot capacity,
  duplicate price, own-cross and near-END. Zero violations on 76 measured
  ledgers. It depends on saved closures, so it complements source/synthetic
  tests rather than independently deriving every notice. The pre-freeze
  rehearsal's 38 baseline and six synthetic ledgers plus planted corruption
  controls are consistent with its code/log; the review also reran the
  current seven-group selftest and the receipt falsifiers.
- **Near END and blocks:** execution uses the profile's actual placement at
  `:160-161`. The declared frozen `withholds_near_end` statistic still uses
  58 ms (`pf_ladder_pressure.py:471`), but is **zero in every arm** under both
  the frozen and actual-profile window. No outcome impact here. `blocks`
  (`:296-309`) counts post-command desired occupancy and retiring *order-
  ticks*, not venue-active time or unique cancel-in-flight periods. The
  separate `both_exact` is venue-active time. Cap refusals really are zero;
  that does not mean the cap never constrains sizing/WAITs. The receipt
  correctly discloses this. Do not use the cap-refusal count as a risk-idle
  percentage.
- **Measurement and reading:** markouts retain their declared eligibility,
  censoring and units; class counts conserve fills and quantity. Every fill
  respects its order's side/price, activation, strict removal and size.
  Public-event metadata agrees across arms. All four contrasts and their
  interaction reproduce from separate arm totals. `assumptions_do_not_rescue`
  is the declared pooled-END test; the stronger every-day-negative statement
  also happens to hold. Through-price classification is an execution relation,
  not a claim that every such fill is informed. No lost/added attribution or
  fixed-state latency causal claim is supplied by this round.
- **Projection:** the 8-market pilot and 11-market rest are disjoint and cover
  19. Write-once projection reproduces to 0.223379h batch and 2.427526h line
  including its 0.03h review reserve, below approved caps. It is a cost gate;
  no economic winner was selected at the pilot. No Oct 27 rest job is needed.

## Prior findings, checks and resources

Previous finding 1 is substantively addressed by
`ladder_queue_front_20261122/report_interpretation_v2.md`: the original queue
receipt remains provenance and its numbers are unchanged. The interpretation
qualification needed above concerns the new ASYNC receipt. Previous finding 2
was explicitly deferred; coverage workarounds are extended here to the newly
exported data. The pre-freeze missed-amendment disclosure is retained; this
round now implements the approved asynchronous contract.

`verify.py` ran through **PF_SLOT=codex** and the prescribed `run_light.sh`:
seven-group synthetic selftest; 12 checker booleans; 12/12 falsifiers; all
24 saved mutant verdicts rejudged against pinned source identities; receipt
copies/source closure; 95 arm rows; 152 class rows; full readings and
projection. It used **11.05 CPU seconds**, peak **392 MiB**, no historical
replay/tape load. Evidence: `verify.log`, `evidence.json`,
`end_decomposition.csv`. Synthetic success establishes implementation
consistency, not predictive skill or profitability.

The receipt logs including its final checker sum to **2.375628h** before
this review. Adding 11.05 seconds gives approximately **2.378697h**, before
small staging/commit verification overhead. Review remains far below the
0.03h reserve and the approved line ceiling of 2.50h. New work must count
all later jobs and this review, not reuse the earlier 2.375 rounded total.

## ONE next priority — FEASIBLE-CANCEL-HEADROOM-v1

**One bounded saved-data audit of the loss a timely cancellation could remove,
and the good fills it would necessarily remove with it.** Do this before
another warning model or threshold. It addresses the user's open question
about bad fills that accompany good queue fills; the present experiment
does not quantify that trade-off. This is a diagnostic, not a candidate
strategy and not another speed/queue variant.

Freeze the builder, input manifest, deterministic choices, output checks and
reading map before producing the tables. Concrete preparation contract:

1. **Population:** every submitted generation, including refusals/no fills/
   no review, in CURRENT_ASYNC_FAST10 on these same 19 markets. Repeat the
   identical diagnostic for FRONT_ASYNC_FAST10 as a separately reported
   sensitivity, never pool the arms. Use only hash-bound saved ledgers,
   command arrays, fill marks and queue diagnostics. No raw tape load, new
   data, replay, model fit, threshold search or post-hoc market exclusion.
2. **Choices:** KEEP the recorded path, or replace **one actually recorded
   live KEEP** for that generation with PULL. Use the real command time and
   the profile's 10 ms effect. Only that generation's recorded fills at or
   after the hypothetical effect and before its recorded end are deleted.
   Fills at the request tick have already happened; a fill at the effect is
   removed. No cancellation before the generation is reviewable and no
   independent hindsight deletion of individual negative fills. Count each
   fill once, despite repeated reviews. Retain no-eligible-choice statuses.
3. **Hindsight diagnostic:** choose the single feasible cut maximizing the
   retained **eligible +1 s value** on that generation. KEEP wins ties with
   the recorded path; otherwise use the earliest equally optimal cut.
   This deliberately uses future outcomes and must be named an **oracle**,
   never a deployable signal. Compare with unchanged KEEP and pulling at
   the first eligible KEEP callback. No null/skill claim is appropriate for
   an oracle; these are accounting controls. END is recomputed for the same
   chosen cut, with late/unsupported +1 s values separate, not reoptimized.
4. **Required output:** actual versus oracle retained value, gross negative
   value removed, positive value/shares sacrificed, residual losses, action
   counts and concentration, by day/side/market and same-/through-price.
   Separate fills before the first feasible effect, oracle-deleted later
   fills and oracle-retained later fills. No eligible callback is a distinct
   status. Break out orders with good fills before a later bad suffix versus
   orders where the suffix mixes both. Carry entry/review queue information
   only where actually recorded; unknown is not front. Do not condition
   decision availability on hindsight knowledge of true remaining quantity.
5. **Limits and stop rule:** freeze all other orders and external flow. No
   replacement, queue re-entry cost or changed inventory feedback is modeled.
   Therefore this is an optimistic **fixed-path deletion accounting bound**,
   not an upper bound on all MM policies or a realizable profitability claim.
   Earlier fills are timing-constrained *given these admissions and allowed
   callbacks*, not universally unavoidable. If even the oracle leaves a
   negative +1 s total on a day, record `no_local_cancel_rescue` for that day;
   do not tune a cancellation-only variant to solve it. If headroom exists,
   report its good-fill cost and concentration; this permits a later question
   about causal discrimination, not automatic model fitting or promotion.
6. **Verification:** unchanged-mask reproduction of the accepted totals;
   per-generation prefix/suffix reconciliation and complete membership;
   synthetic fills just before/on/after effect, before first review, multiple
   partial fills, good-then-bad and bad-then-good suffixes, no review, no fill,
   late marks, and KEEP ties. Check selected masks and both good/bad economics
   independently with known-bad cases. Bundle the two nonblocking reporting/
   coverage items above where relevant; one receipt and one review.

**Budget/authorization:** cap this saved-data development round at **0.05
core-hours**, reserve **0.02h** for its combined review, and require the
updated cumulative ledger plus both to remain **≤2.50h**. It fits the roughly
0.12h remaining here; no additional USER permission is needed within this
existing consumed-data diagnostic scope and ceiling. Stop if that projection
does not fit; any extension, new data/days, untouched validation, live use or
new policy/model experiment needs a separate decision. POSITIVE-PRESSURE-
GUARD-v1 stays stopped.

**Largest evidenced limitation:** even favourable tested execution leaves a
negative short-horizon fill mix, dominated by through-price losses. We have
not yet measured how much can be cut at feasible decision times while
preserving the good fills, nor whether a causal signal can identify those
cuts. Faster execution alone did not answer either question.
