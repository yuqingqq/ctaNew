# PI0-MOVEMENT-CALIBRATION-AUDIT-v1 — combined review, 2026-11-21

Reviewed target **f311caaef6ea50b8a7df7ebd6c74a3cf22f39dc1**, range
`6c2a992d8fc0e3e52b073c694687e674aa562f34..f311caaef6ea50b8a7df7ebd6c74a3cf22f39dc1`.
Development freeze: `03243e5b3ddb304cf795c57010c1338181e4c478`.
The intervening reviewer commit `5a8ee6f55639cc94854db7249d1f6ee4b849dce0`
was the acceptance contract, not a new development result.
Source and line references below are at the full target SHA. Review used the
detached checkout `/tmp/pf-review-f311caae`, preserving the shared working tree.

**Accept the saved audit for the stated descriptive use. No BLOCKING finding
for these results or the next bounded consumed-day experiment.** The three
accounting identities hold on all 3,083 eligible stage rows. All ten aggregate
and row-level CSVs reproduce byte for byte. The admitted movement forecast is
optimistic where PI0 chose to post; the value mapping itself reconciles.
The audit does not establish which representation will repair that forecast,
nor that a small calibration challenger cannot work.

**ONE next priority: PI0-MOVEMENT-SHRINK-v1**, specified below: a small, declared
movement calibration challenger fitted on the saved Oct 26 admitted states,
measured on Oct 27, with prediction and local action-value tests kept separate.
Both days are consumed. No additional USER decision is needed for this bounded
experiment. No new dates, larger label build, teacher refit, extended END or
untouched validation is authorized by this review.

**Consolidated actionable checklist — bundle with that experiment.**

1. **P2 / NONBLOCKING — preserve the declared diagnostic, qualify its inference.**
   `pf_movement_audit.py:507`–531 implements the frozen reading exactly. In
   particular, line 527 returns `representation_or_data_needed` whenever the
   top-five world share crosses 0.5, even with positive mean errors in every
   day/side cell and populated decision-state bins. `FINDINGS.md:3157` and
   `HANDOFF.md:936` turn this into a statement that more representative data or
   another representation is needed. This is a **declared research heuristic**,
   not a comparison of calibrated and uncalibrated predictions on another day.
   Tail concentration warrants caution about estimation; it does not identify
   whether a correction generalizes. Similarly, marginal residual ranges
   (`pf_movement_audit.py:515`) do not measure incremental predictive power or
   control for the other strata. The forced entries are all at t=0/ordinal 1;
   the admitted entries are selected at later states as well. Their different
   average errors do not isolate the causal effect of admission selection.
   The receipt correctly discloses this population difference at lines 107–108
   and 176; retain that qualification in state and subsequent conclusions.
   **Acceptance:** retain all original flags and thresholds; describe the
   nearly unbiased forced result as aggregate calibration on that cohort,
   not proof of conditional calibration or calibration of the admission
   population. Record the next candidate as a new, adaptive experiment, not
   a retroactive pass of this audit's recommendation gate.

2. **P2 / NONBLOCKING — A–I do not validate every reported aggregate value.**
   `movement_checks.py:264`–300 validates counts and weights for moments,
   pre/post and strata; H at lines 340–358 validates selected admitted-entry
   numbers. Neither validates all conditional forecast means, per-share
   ratios or stratum residual means. On private real-data copies, adding
   100 to one admitted-entry `mean_mhat_fill`, one `post_per_share`, and one
   stratum `c_m` leaves **all A–I true**. See `instrument.json` and the
   `instrument()` reproduction in `verify.py`.
   **Current bounded workaround:** receipt SHA256SUMS match all actual saved
   outputs; all ten CSVs were regenerated from the bound rows and references;
   direct independent arithmetic also checks all 91 moment views and 69
   pre/post views. Those changed copies fail the byte/hash comparison. Thus
   this is a checker coverage gap, not a discovered error in the actual tables.
   **Acceptance for reuse:** add independent checks of the aggregate values
   that drive the new candidate receipt, including conditional means, MSE,
   weighted denominators and selection values, with applied corruptions of
   those values. Do not rerun the current tape extraction to close this.

3. **P3 / NONBLOCKING — distinguish the displayed share count from the ratio
   denominator, and close out resource accounting.** `pf_movement_audit.py:432`
   computes the per-share denominator as `sum(weight*q)`, correctly. Lines
   434–438 store unweighted `q` beside weighted sums and ratios. The receipt's
   table at `report.md:99`–104 shows 2,014.7 shares next to the weighted
   -2.140 c/share ratio. The actual admitted denominator is **4,429.481269
   weighted shares**, with weighted movement -9,480.503630 c; the unweighted
   count is 2,014.733699. The ratio and pre/post split are correct. Store and
   label both denominators when this table is reused. Also, final committed
   logs contain **152 jobs / 0.282478 core-hours**, versus the report's
   earlier 148 / 0.275 snapshot (`report.md:66`); both are below the declared
   one-hour bound, peak 1,140 MiB. Use a named as-of or final totals in the next
   receipt. Neither detail changes the economic reading.

4. **P3 / NONBLOCKING — finish the reference-record binding if this extraction
   is reused.** `pf_movement_audit.py:344` records the exact admission record
   hash, but `measure()` at lines 548–550 does not compare that field to its
   current admission record; the existing-record return at line 314 checks
   sources only. The loader/checker also trusts local input records after
   checking their data hashes. For this pinned audit I additionally verified
   **all 24 admission/retention partition records against the earlier committed
   receipts**, all 12 mids records' admission hashes, part/day/key membership
   and fill-index coverage. Everything matches (`metadata.json`). Preserve
   those bindings in any reused adapter, or reuse these exact reviewed
   artifacts by committed hashes. No generalized resume rewrite or rerun is
   required for the next saved-data candidate.

**Resolved versus open previous findings.** All four items in `5a8ee6f5` are
closed for this receipt: bookTicker is correctly named; qm is explicitly
quantity-weighted decision-to-END movement, with conditional/unconditional
units separated and P's calibration error stated; weighted and unweighted
generation/world concentration are both present; and weights, cells, n_cell,
part and complete membership are bound by exact manifest keys. The retained
review control uses each review's decision hour; all 299 match their original
admission cells on these data. No prior material blocker remains. Forecast
quality and conditional-retention usefulness remain research questions.

**What the verified audit establishes.** As of the target receipt, the admitted
population is 329 sampled generations in 155 related worlds, weights summing
to the 689-generation census; entry/review/pull have 329/299/268 eligible rows.
The forced population is 768 forks, with every status kept; 755 entry rows
have usable model forecasts (756 forks have valid worlds). Stages and cohorts
are never pooled. The first pull is often the same observation as the first
review; it is not another independent sample.

- Entry V bias is +11.099 c per opportunity: edge/quantity -0.882 c plus
  movement +11.981 c. Movement error splits into probability +0.310 c and
  conditional-head +11.671 c. These are accounting decompositions, not causal
  assignments of model error.
- On the **88 filled opportunities**, admitted mhat averages -6.90 c versus
  -45.47 c realized, weighted. The overlapping forced reference has 153 filled
  opportunities, -48.77 versus -51.88 c. These are total movement cents per
  filled opportunity, not cents per tranche or per share. Entry has 202
  admitted fill tranches with reference mids; they are not 202 independent
  forecast observations.
- The post-fill component is -1.849 of the total -2.140 c per weighted filled
  share. Most of this realized loss happens after the fill. It does not
  prove those moves were predictable at entry, or that realized fill delay
  can be supplied as a decision feature.
- The movement head still slightly improves MSE against zero movement
  (4,336.46 versus 4,395.20 c²). The complete value forecast loses to zero
  (4,043.27 versus 3,982.00 c²). Deleting movement is not supported by those
  numbers. Squared error estimates the conditional mean appropriate for
  this expected-value accounting; sparse, heavy losses and selection can
  make that mean difficult to estimate. Aggregate MSE alone cannot establish
  useful POST/KEEP decisions.
- The weighted top-five world share of movement error is .591, and all four
  day/side mean errors are positive. These facts coexist. The 500-draw
  first-review PULL control remains at percentile .806, with 45/91 strata
  free (166 rows); no demonstrated selection advantage. Every observed
  entry-to-review transition has some new observation, so no no-news control
  identifies the cause of churn.

**Verification and limits.** `verify.py` imports only the pinned research
sources. Four bounded jobs ran through `PF_SLOT=codex run_light.sh`: metadata,
saved-data recomputation, synthetic/real-copy controls and the aggregate
checker probe. All passed: **41.51 s wall / 39.44 s user CPU total, peak
330 MiB**. Logs and JSON results are beside this report. All 23 receipt
artifacts match; the six freeze files checked are unchanged; 24 source records
and 12 reference records bind; A–I pass; all 13 real-copy falsifiers fire with
passing clean controls; both synthetic suites pass. The 64 final mutant
records were rejudged against the pinned sources (not rerun). The earlier
63/64 run is retained and its membership-fixture repair is appropriately
distinguished from a model change.

I did not reload the real tapes or rerun the 56-world extraction. Its clock
and identity code were inspected, its saved reference records and all 2,770
stage-counted fill rows reconcile, and the synthetic lookup controls passed.
Fill references use the latest venue frame at/before the fill; admitted
reference age reaches 255 ms, forced frame age is unavailable. This limits
the precision of the pre/post attribution, not total decision-to-END movement.
There was no fit, new label build, real branch simulation or untouched-day
access in this review. Passing controls establish implementation consistency,
not profitability or statistical validation.

**Next-round contract — PI0-MOVEMENT-SHRINK-v1 (one candidate).**

This is an explicit new experiment after seeing the audit, not a revision of
its frozen reading. A low-dimensional challenger is worthwhile because a
test on the saved rows is cheap and directly asks whether suppressing the
selected forecasts' optimism helps. The audit's concentration warning means
we cannot assume it will transfer. A new high-dimensional movement tree fitted
to a few dozen selected fills is not the next priority.

1. **Freeze, population and chronology.** Freeze the complete adapter, scalar
   fit, controls, evaluator, output names, manifest and reading rules before
   fitting; count one new candidate in multiplicity. Reuse the exact admitted
   artifacts verified here. Fit on **Oct 26 only**: 166 entry rows (42 filled)
   and 151 first-review rows (37 filled), as separate POST/RETAIN datasets.
   Measure on **Oct 27 only**: 163 entry rows (46 filled) and 148 first-review
   rows (42 filled), keeping original inclusion weights and every status.
   First-pull rows may be shown descriptively, never added to training or the
   main scores. Do not fit on the forced cohort or pool it with admissions.
   Commit the fitted parameter/hash artifact before producing candidate
   Oct 27 scores. Both dates have already informed this research: this is
   chronological **consumed-day development**, not unseen validation.

2. **Exact target and a single small correction.** Keep the original neutral
   `hold_from` targets, original END=5,000 ms, all timestamps/latencies,
   structural removals, price, size, cap and terminal mark. Let `z=1(q>1e-12)`,
   `M=qm` (zero without a fill), `b=E[qm]`, and `w` be the manifest weight.
   Keep the frozen P, E[q], edge and all support/forecast checks unchanged.
   Fit separately for each side × head; no hyperparameter search:

   - On Oct 26, `mu = sum(w*M) / sum(w*z)` within that group. This is a pooled
     conditional **quantity-weighted** movement per filled opportunity, not
     per-share drift. Derive it from the complete neutral-outcome table.
   - For each row construct the pooled prior
     `a = clip(P*mu, E[q]*L, E[q]*U)`. With decision-mid in cents,
     `(L,U)=(-mid,100-mid)` for bids and `(mid-100,mid)` for asks. These are
     known physical bounds, not fitted thresholds or a future mark. E[q]=0
     gives a=0. Report how often this projection binds.
   - Using **all** Oct 26 rows, including no fills, set `d=a-b` and
     `lambda=clip(sum(w*d*(M-b))/sum(w*d*d),0,1)`. A denominator at most
     1e-12 gives lambda=0 with an explicit status. Require at least 30 valid
     opportunities and 10 neutral filled opportunities in a training group;
     otherwise keep the original forecast with `insufficient_fit_support`.
     Those support rules are fixed now, not selected on Oct 27.
   - Predict `E[qm]_new = b + lambda*(a-b)` and
     `V_new = edge*E[q] + E[qm]_new`. This gives four mu/lambda pairs, no
     extra tree fit, no clipping of realized losses, no outcome-selected
     deletion and no additional factor of P. The original qm head already
     estimates conditional quantity-weighted movement; merely calling a
     replacement a “per-fill head” would not change that target.

3. **Prediction controls and scores.** Report original (lambda=0), the full
   pooled prior (lambda=1, a fixed control) and the fitted mixture on identical
   rows. Include zero and edge×E[q] value baselines. Report weighted movement
   and V MSE, mean bias, conditional-on-neutral-fill errors, Brier unchanged,
   calibrated-mean tables, selected counts, unique worlds, Kish n, severe
   value < -30 c totals, weighted/unweighted top-five generation/world
   concentration and side-specific results. MAE remains secondary. No
   trimming, loss winsorization, extra bin fit, threshold sweep or choice of
   shrinkage based on Oct 27. Declare identical old horizon/queue/ordinal
   bins as diagnostics only. A lower mean bias with worse MSE or worse
   economic selections must be visible; do not collapse these into “works”.

4. **Match the local action to the longer-exposure label.** At a saved entry,
   the local comparison is POST then keep that generation under the neutral
   structural rules to original END, versus no such exposure (local value
   zero). Use POST iff V_new>0; a tie WAITs. At the first live review, compare
   retaining that generation under the same longer-lived policy versus PULL
   at the existing cancellation effect time. Use KEEP iff V_new>=0. The
   targets already exclude fills before the respective action effect;
   common past/cancel-pending fills do not become avoidable losses. Keep
   structural removals. There is **no newly introduced fixed hold timer**,
   and no discretionarily cancelling the assessed KEEP 50 ms later while
   still claiming its neutral-hold value.

   This round scores those two **local** choices on existing neutral labels;
   it does not simulate a new combined strategy. Entry evaluation is a filter
   of PI0's original admissions, not evidence for posts PI0 rejected. Review
   evaluation is on PI0's reached live states, not the candidate's future
   reached states. Report each stage separately; never sum their selected
   values or claim a full PI0 P&L gain. Comparison against the original sign
   rule uses this same local exposure target, not its dynamically cancelling
   realized branch return.

5. **Economic controls and computed reading.** On Oct 27 show weighted
   `sum(w*chosen*value)/sum(w)` over all eligible opportunities and mean value
   among chosen opportunities, for each head and side. Compare with choosing
   none, choosing all, the original V sign rule and the pooled-prior sign
   rule. Freeze 500 uniform matched draws of the chosen action mask within
   day × side × decision-hour, preserving each stratum's action count and
   inclusion weights; publish free/degenerate strata, mean and p95. Labels
   stay on their original rows; only the mask changes. Require the null
   implementation to reproduce the weighted selected population. The
   claimed decision-advantage predicate needs nonempty selected exposure,
   positive local value, improvement over the original and pooled-prior
   controls, and value above matched p95. Degenerate controls or inadequate
   movable support give INCONCLUSIVE, not PASS. Compute prediction improvement
   separately (movement and V MSE versus original); show each side even when
   the pooled score improves. No confidence interval with one evaluation day,
   no claim that 500 draws create 500 independent market samples.

6. **One combined acceptance and budget.** Bundle checklist items 1–4 where
   reused, synthetic exact-mixture/zero-quantity/sign/boundary cases, a
   chronologically contaminated-fit refusal, wrong-weight/source/key/unit
   refusals, shuffled/adversarial action controls, and aggregate-value
   falsifiers with the fit and measurement. A lambda=0 control must reproduce
   the original vectors and local rule exactly; a hypothetical positive
   exposure must be selectable; all-PULL/WAIT cannot pass the economic gate.
   Bind full membership, every rowwise identity, train/evaluation keys and
   fitted parameters through the receipt. Reuse immutable artifacts or bind
   exact prior record hashes; no new raw mid lookup is necessary. Budget:
   **one aggregate core-hour**, existing authorized slots, 1,536 MiB checkpoint
   and 2,048 MiB hard RSS bound. Stop with explicit statuses if exceeded.
   Submit one freeze/fit/results package; no intermediate minor-fix review.

This leaves a consumed-day path for the USER's longer-holding direction:
test whether a calibrated longer-exposure value can select useful retention,
before changing the full controller. If it only reduces posting or fixes the
mean without selecting positive exposure, report that result and do not tune
another variant in the same round. Any favorable result would still need a
declared controller/reached-state replay before independent validation. More
label dates, untouched-day validation (at least five complete days), an
extended horizon, or an expanded compute/model search requires a separate
USER scope decision; none is necessary for the one challenger above.
