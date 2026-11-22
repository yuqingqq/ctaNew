# QUEUE-PRICE-STATE-AUDIT-v1 — combined review

2026-11-22. Reviewed target **012db7bbcdd861bbd19a984286935d1553c88db3**;
range **72a36de984fa2ace2b626bea6e8b6db5682fcb16..012db7bbcdd861bbd19a984286935d1553c88db3**.
Freeze **0e20331fc573f9b93485bc9fb4ed8bddbc334c9b**. Sources reviewed in
detached checkout `/tmp/pf-qp-review-012db7bb`; saved inputs read from the
shared data directory and verified against pinned records. No implementation
changes, replay, training, new day or external messaging.

## Decision

**Accept the descriptive economic result. No blocking finding, no required
rerun and no minor-fix round.** All 4,840 neutral states, 4,920 cell rows and
five exported reading tables reconcile. The principal common-cohort values
are correct: front −0.951 / +4.592 c/share at 1 / 30 seconds, within_R
−1.505 / −1.267, beyond_R −2.274 / −2.920. These are conditional descriptive
markouts, not strategy profit or the causal value of queue priority.

The conclusion must remain **insufficient evidence for a queue-based rule**:
only 39 valued filled front states, one evaluable day and one evaluable side;
the evaluable side's 1-second ordering differs from the pooled ordering.
Time to END and common-cohort censoring also differ materially across queue
groups. Thin evidence does not prove queue priority useless, but does not
justify another tuned queue variant or another round of marginal tables.

**USER priority, clarified during this review:** the objective is profitable
market making. Fix defects that could materially change economic decisions;
defer small issues when checked evidence already supports the decision. Stop
directions that reliable results reject. Passing software checks or merely
reducing a still-negative loss is not a profitability result.

## Consolidated actionable checklist

1. **P2 — NONBLOCKING: Q8 mixes the grouping clock with the error clock.**
   `pf_ladder_queue_price.py:128–145,200–201` obtains price/display from the
   sampled **live** row while carrying `client_minus_venue` from the earlier
   first review. `qp_checks.py:94–112` correctly reproduces that construction.
   `tools/read_queue_price.py:113–115` then groups the earlier error by the
   later live price/display. Receipt `report.md:111–117` should name both
   clocks; its 900/1,633 number does **not** measure accuracy conditional on
   being at the touch at the first review.

   Independent exact-key original-feature check: 4,314/4,840 live states are
   later than their first review; 1,240 change price/display group, including
   376 of the 1,633 later at-touch occupied rows. Grouping by the actual first
   review gives **957/1,922** within one share, rather than 900/1,633. See
   `evidence.json` and `group_clock_examples.csv`. This is simulator-relative
   error, never exchange-confirmed rank; “within” here is the code's strict
   `abs(error) < 1` partition.

   **Disposition:** relabel the existing table as “first-review queue error
   grouped by subsequent sampled-live price/display,” or use the first-review
   grouping if that conditional accuracy is needed later. Do not use it as
   current-live queue accuracy. This diagnostic is absent from the economic
   grouping and selection rules, so none of the principal values changes.
   Defer implementation; no extra measurement is required to close this round.

2. **P3 — NONBLOCKING: the checker covers fewer reading fields than claimed.**
   `qp_checks.py:253–266` checks summary orders and counts but omits the
   per-day/per-side order dictionaries, pooled numeric band values, detailed
   depth values/support flags and some metadata/counts. It also does not read
   the five condensed CSV tables produced at
   `tools/read_queue_price.py:121–125`. Thus receipt `report.md:17` and
   declaration's “every reading” wording is too broad.

   Two real-copy, in-memory reproductions leave frozen P passing: replace the
   Oct 24 principal order with a wrong order; replace the pooled front value
   with 999 c/share. No data files were altered. **All actual detailed values,
   orders and all five tables were separately verified correct in this review**
   using the independently reconstructed cells. The frozen falsifier still
   refuses all ten of its declared corruptions, including both prior escapes.

   **Disposition:** use the review evidence as the bounded workaround. Extend
   comparison to these fields/tables only when the reader/checker is next
   changed, with one detailed-JSON and one exported-table falsifier. No
   standalone repair cycle. Condensed tables also omit `valid_risk` while
   displaying the valid fill rate; its correct denominator remains explicitly
   available in full `cells.csv`, so carry it over when those tables next change.

No other outcome-changing defect found in the affected code and integration
boundaries. These two items are neither grounds to reject the existing
economic readings nor a requirement to repair every reporting surface.

## Prior findings: resolved versus open

| Finding from 72a36de9 | Review disposition |
|---|---|
| Q=0 confused with current empty display | Closed. Exact original `fp_shown`: 458 zero-display and 331 occupied front states; 198 behind-best occupied. No Q<=shown assumption in grouping. |
| All-state versus common counts/support; misleading all-days predicate | Closed for full cells and decision predicates. Own-cohort valued-filled support, explicit valid/risk/share denominators and all-four-day test verified. Compact export denominator omission is deferred above. |
| Derived queue/cell fields unchecked; ledger gzip identity | Closed for reused fields and all cell columns. U/C independently recompute them; actual ledger gzip hashes checked. New receipt's broader reading-coverage claim has the bounded limitation above. |
| Arrival bin mistaken for latest action time; 22.5% misinterpreted | Closed by receipt lines 132–134: a later review can prevent some I0 fills; 22.5% is a share of net 1-second loss, not an unavoidable fraction. |

Old frozen receipts/sources remain unchanged. The corrected receipt supersedes
their affected interpretations in band. No correctness finding requires
rebuilding the old arrival decomposition.

## Source and artifact checks

- **Join and labels:** builder lines 112–145 hash-bind source receipts and
  feature parquets; exact `(day,m,gid,t,side)` join; duplicate/missing/shifted
  key and conflicting Q/depth refusals covered in the pinned selftest. Q=0,
  unknown/invalid, price distance and displayed size are separate. Principal,
  exact-depth behind, outside and support-census views retain their scope.
- **Intervals and cohorts:** builder lines 149–196 preserves the accepted
  four intervals, union end-cause status and missing-value/common semantics.
  Current interval input bytes equal the previously reviewed artifact. All
  counts, fill/state positive and negative parts, rates, values, sparse flags,
  time-to-END and censoring columns recompute with the checker's separate
  pivot/groupby implementation (lines 118–166). No missing mark becomes a
  valid zero; no-fill rows remain. Timing/END semantics are inherited from
  accepted ARRIVAL-QUEUE evidence, not newly inferred from this selftest.
- **Readings:** the correct support minimum is applied to actual valued
  filled states. All-four-day and both-side consistency, pooled/principal
  persistence and pressure support census reproduce. No 0/6 pressure
  contrast is claimed evaluable. Post-run below-support day values are
  explicitly labelled descriptive/not declared at receipt lines 56–63.
  The main limitation is accurately stated at lines 142–150. Aside from
  the two items above, the wording stays within the computed predicates.
- **Identity/freeze:** builder and checker unchanged between freeze and
  result commit. Manifest source/input hashes, raw-versus-committed output
  copies, original parquet hashes and pressure ledger gzip bytes checked.
  The disclosed pre-freeze rehearsal is development work; it is not unseen
  validation. No untouched data were accessed.
- **Tests:** pinned selftest groups 1–5 pass; nine frozen check predicates
  pass; ten frozen falsifiers fire. All 23 saved final mutation records
  rejudged against the pinned source hashes, indices and names; no need to
  rerun 23 mutants. Five condensed tables (125 rows altogether) also checked.
  The remaining checker escapes are recorded, not silently counted as tests
  passed. Synthetic controls establish implementation behavior only.
- **Projection/resource gate:** `tools/projection_qp.py:21–30` is write-once,
  incorporates previous line/review usage and projects saved stage costs;
  launcher lines 13–14 refuses the run unless it fits. It projected 0.013166
  round hours. Recount including the development checker itself gives
  0.012628 round hours, 1.974781 cumulative before this review. The frozen
  receipt's 1.974 is its earlier at-check snapshot. No budget breach found.

Review compute: selftest 0.66 CPU seconds, complete verification 2.17 seconds,
resource summary 0.01 seconds; peak 208 MiB. Approximately **1.97557 of 2.0
core-hours** consumed, leaving **0.02443** before small commit verification.
Every numerical check used `PF_SLOT=codex` and absolute run_light/script paths.
No other session's slot, replay, model fit or full tape experiment was used.

## ONE next priority: finish the already-frozen economic challenger, subject to USER budget approval

**Complete POSITIVE-PRESSURE-GUARD-v1 under its existing specification and
stop rule.** This was frozen and piloted before the queue audit; it is not a
new queue rule selected from these thin cells. It asks the directly economic
question: does broader adverse-pressure coverage improve the complete policy
beyond Q5_CONTROL, without its apparent benefit being only reduced exposure?
Existing Q5 evidence justifies this single remaining comparison, not more
queue tables, a larger predictor, or a threshold search.

Concrete development deliverable after approval:

1. Keep the frozen `adv_bn_r250 > 0` rule, clocks, two 100-share levels,
   inventory mechanism, four consumed dates and 19 complete markets. Reuse
   the accepted seven-market Oct 24 pilot. Finish only Oct 25–27, the frozen
   incremental screen, full measurement and combined checks/receipt. Do not
   trim controls or sample, rerun the pilot, add thresholds, or read new days.
2. Compare POSITIVE versus Q5_CONTROL by day/market: **absolute and incremental
   END value**, eligible 1-second total value and c/share, positive/negative
   parts, lost/added fills, participation, inventory and first-review churn.
   Preserve the matched incremental selection screen. Report gross zero-cost
   economics and the common mark horizon; these remain simulation evidence,
   not live or settlement profitability. Queue findings remain limitations,
   not a new gating feature or a reason to block this existing comparison.
3. Honor the frozen `stop_candidate`: unless both primary economic contrasts
   improve every day **and** the incremental screen demonstrates selection
   beyond exposure, stop this broader-pressure candidate. A thin screen does
   not justify tuning more variants. Even if it passes, disclose whether
   absolute value remains negative; improvement alone does not complete the
   profitability goal. One combined review, without a prerequisite cleanup
   round for the two items above.
4. **USER decision required before resuming.** Proposed amendment: positive
   round cap **0.14 → 0.16 core-hours**, cumulative line **2.0 → 2.12**, with
   an updated write-once projection and review reserve. The existing pilot
   projects about **0.10066 additional core-hours**, reaching about **2.07623**
   before new review/overhead. This no longer fits the original line even
   though the round overrun alone was previously about 0.02. Preserve the
   original refused projection and add a superseding resource authorization;
   do not rewrite a frozen stop. These proposed ceilings are **not yet
   authorized**. This requirement comes from the explicit cap/approval terms
   in the current review request and the stopped candidate's declaration.

The clarification about profitability changes prioritization, not those
explicit caps. No new queue-value replay, data collection, untouched-day test
or live use is proposed or authorized by this review.

**Largest evidenced limitation:** the queue result is descriptive, thin and
skewed in remaining horizon; it cannot establish the economic benefit of
early posting or preserving priority. We have reliable evidence of losses
and of a useful pressure intervention, but no profitable complete strategy.
Spend the next authorized round on the frozen economic comparison and its
kill criterion, not on perfecting minor audit fields.
