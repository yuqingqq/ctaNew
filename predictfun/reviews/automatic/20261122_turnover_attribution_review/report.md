# PAIRED-TURNOVER-ATTRIBUTION-v1 — combined review

2026-11-22 UTC. Review only; development owns implementation.

**Target: 51ed592d9ddf9e2968d12ffb9fc966763c0a8ce9**

Range: d799cb77c0a6de1b9c341ea6b1692f0c00cf273f..51ed592d9ddf9e2968d12ffb9fc966763c0a8ce9.
Reviewed freeze db044afba721b278a4c8ae11fcb416983709e8f1 and the results,
using a detached checkout. Other-session working files were not used as source.
Read the repository guide, reliability/batching rules, current Predict.fun
state, prior review and the superseding pilot interpretation.

**Accept the saved-data attribution. No result-blocking defect and no replay
required.** All 64 paired END values remain unchanged. All 128 arm
decompositions and 30,504 FIFO closes reproduce, including carried bases,
actual acquisition times, quantities and lineage assignments. This establishes
accounting, not predictive skill or profitable market making.

The useful result is that additional resting does not automatically earn back
its losses through repeated two-sided trading. Under this particular common
continuation, the REVIEW aggregate loses through completed turnover. The
ADMISSION gain is mainly residual inventory exposure. These are overlapping
counterfactuals on eight markets from four consumed days, not portfolio P&L.

## Consolidated findings

1. **P2 — NONBLOCKING for these values: the prose transfers the total-Y day
   sign to its turnover component, and overstates price attribution.**

   `predictfun/FINDINGS.md:3966` (P114) says retaining closed round trips worse
   on every sampled day. The receipt `report.md:79` similarly joins the
   every-day Y result to completed turnover and “worse prices.” The actual
   Oct 26 REVIEW decomposition is **Y −2,562.82 c = matched +862.93 c +
   residual −3,425.75 c**. The table and `report.md:41` already show this
   exception. Correct wording: **total REVIEW Y is negative on all four
   sampled days; matched turnover explains its pooled deficit, but improves
   on Oct 26, when residual inventory explains the loss.**

   A difference in total matched profit also changes matched quantity, lot
   mix and carried-inventory closures. It is not by itself a matched-price
   causal comparison. At `pf_ladder_turnover_attribution.py:86-94`, carried
   lots correctly use the fork mark, not their acquisition price. Their
   closing profit is incremental profit since the fork, not lifetime cash
   round-trip profit. The report's per-share table is explicitly a mean of
   arm ratios; keep that denominator when quoting it. Independent checks give:

   | Cohort | Retain: mean arm ratio / ratio of sums | Alternative: mean arm ratio / ratio of sums |
   |---|---:|---:|
   | REVIEW | −2.900 / −3.401 c | −2.879 / −3.181 c |
   | ADMISSION | −2.026 / −2.828 c | −2.153 / −2.783 c |

   Newly opened lots alone also lose: REVIEW pooled ratios are −3.397 /
   −3.243 c per matched share. Thus the broad finding of losing turnover is
   not an artifact of carried bases or averaging. It does **not** show that
   inventory risk is irrelevant: price movement while a lot is held can
   become a realized loss when it closes.

   `report.md:70` and P114's “largely independent” should mean only that own
   END sign and matched profit are **not interchangeable**. No independence
   test was declared. `pf_ladder_turnover_attribution.py:125-131` classifies
   the matched portion; unmatched residual remains outside that class.
   Neither a profitable matched portion nor a negative own-END sign proves
   the early fill was necessary to obtain queue priority.

   Disposition: supersede these sentences in the next combined receipt and
   state update. The numerical artifacts stand. No repair-only round.

2. **P3 — NONBLOCKING with this review's verified workaround: ancillary
   checker coverage remains partial.**

   `ta_checks.py:101-108` checks four FIFO totals and Q, not all arm/lot
   columns. `:121-124` checks only part of the event export; `:167-181`
   checks selected summary fields and the sign strings, not complete group
   membership or every statistic. The L check iterates replacement groups,
   so arms without a replacement need separate coverage. Hashes protect the
   frozen bytes but do not independently validate a builder's arithmetic.

   Review-only corruptions of `arms.hold_ms`, `arms.matched_lineage`,
   `lots.basis` and `summary.top5_share_abs_Y` each pass their corresponding
   semantic family. These are instrument coverage limits, not evidence that
   the saved fields are wrong. **Every actual field in those families was
   separately reproduced here**, including all 31 summary groups, every
   lot field, all event fields, all arm components and all 128 delay rows.

   Disposition: preserve `verify.py` as the bounded workaround for this
   receipt. Extend only checks for fields a later experiment actually uses,
   with a targeted known-bad control. Do not restart the accepted run or
   require general checker hardening before useful research.

## Prior findings and integration checks

- **Resolved:** the 12 WAIT-arm delay errors, including attempts versus
  accepted submissions. They are exactly the prior review's keys and all
  first attempts are +50 ms. The alternative mean is 869.841270 ms over
  63 arms with replacements. Arms without a replacement were also checked.
- **Resolved:** carried inventory retains its actual acquisition time while
  its incremental basis resets to the common fork mark. Pre-fork partial
  closes are processed before identifying the carried lots. All post-fork
  closes use FIFO and the correct long/short sign. Maximum observed
  `Q − matched − residual` discrepancy is 1.82e−12 c.
- **Resolved for interpretation:** `report_interpretation_v2.md` withdraws
  “wash,” the wrong total-Y day claim, and the END-sign sequence as evidence
  against two-sided compensation. The new positive fixture has fixed-price
  generations and genuinely profitable opposite-side turnover.
- **Open by design:** the queue-necessity mechanism. FIFO matching is an
  accounting convention; it does not establish that an early loss bought a
  later fill. Both policies change future fills, inventory and other orders.
- **Deferred, unchanged:** the older pilot's ancillary callback/trace checker
  gaps documented in d799cb77. This follow-up makes no new replay and does
  not claim those generic checks are repaired. The previously verified
  actual-input workaround still applies.

At `pf_ladder_turnover_attribution.py:139-148`, common events use a multiset
of execution time, public trade index, side, price and quantity, not gids
shared across branches. The two arms have identical starting inventory and
marks, so the common carried term cancels in Y. The event identity agrees
with Y on all pairs. If quantities differ, an event can appear in both
exclusive lists; counts are exact event counts, not net added-share counts.

Sampling, continuation, costs and terminal marks are unchanged. No outcome
selection or new timestamp lookup enters this attribution. The holdings and
focal classes are descriptive outcomes, not decision-time predictive inputs.

## Verification and resources

Evidence: `verify.py`, `evidence.json`, `checks.json`, `falsifier.json`, and
the two verification logs in this directory.

- Frozen source/declaration/manifest/checker bytes agree with db044afb.
  Parent day records agree with their committed copies; ledger hashes,
  parent measurement audit, every output hash and every committed compressed
  output copy agree. Exact 64-key membership is checked.
- Six-group synthetic selftest passes. The 22 final mutant records were
  re-judged against the pinned sources, indices and names. The mutation jobs
  were not rerun. Earlier development attempts remain provenance.
- Independent checker A/D/Y/L/F/S/I and all 10 falsifiers pass.
- An additional deque-based reconstruction verifies every emitted FIFO lot
  field and every arm component. Complete summary membership/statistics,
  focal checks, event exports, corrected delays and exact per-day readings
  reproduce. No historical simulation or model fitting was run.

All checks ran through `PF_SLOT=codex` and absolute script paths. The first
review harness attempt incorrectly redirected the checker's whole data root
while relocating its temporary resource falsifier; it stopped before any
verdict. Its log is retained. Only that temporary operation is redirected
in the corrected harness. CPU: **0.94 + 2.44 = 3.38 s**, peak 167 MiB.
Staging/commit verification also fits the **72 s / 0.02 h** review reserve.

Reading all completed development logs gives **0.004219 h** for this
follow-up, **2.663811 h** for the line before this review, and **0.277742 h**
for the pilot batch with both declared review reserves. These are slightly
later than the receipt's checker-time resource snapshot. Even the full
current reserve gives **2.683811 < 2.85 h**. No old classification cost was
charged again. No new data, untouched day, fit, replay or live use occurred.

## Largest limitation and ONE next priority

**The largest evidenced limitation is discrimination support, not another
accounting defect:** only 32 REVIEW decisions exist here, and the top five
carry 86% of their absolute Y. The 32 ADMISSION decisions are even more
concentrated (97%). We can measure long-continuation value reliably, but
cannot infer a useful conditional-retention rule from this pilot. No tested
model in this receipt predicts that target. Four consumed days supply no
untouched validation claim.

**Next priority: a declared LONG-VALUE-RETENTION-DISCRIMINATION-v1 pilot,
focused on REVIEW rather than another unconditional resting variant.** Its
question is: can information available before the review identify orders
whose *full joint continuation value* is better if retained than cancelled
and re-entered? The primary target remains the accepted full-book Y to the
same supported END; first-fill loss, 1 s markout and FIFO component signs
must not replace it.

Proposed single-round contract, to be frozen before new labels or fitting:

1. Use the same 19 supported consumed markets, Oct 24–27, current asynchronous
   timing, inventory/caps and common lineage continuation. Hash-sample up to
   **256 REVIEW forks**, 32 per day × side, from the unchanged eligible
   first-live-KEEP population. Keep every status; record inclusion weights,
   market coverage and exact old-label overlap. Select without outcomes or
   branch fills. Admission remains a separate, deferred question.
2. Save each fork's causal input vector and queue/age/rung, price relative
   to the received book, recent flow/pressure, known inventory, reservations,
   pending orders and remaining episode time. Reuse the existing causal
   feature builders and branch clocks; no venue-only queue or future fill
   information in the predictor. Bind the state to its paired-label context.
3. Declare **one conservative candidate**, its exact feature schema,
   transformations and fixed capacity/parameters before labels. Train only
   on Oct 24–25; score Oct 26–27 once. Count the package as a new adaptive
   development candidate. No threshold search, feature ablation race or
   tuning after scoring. If populated training/evaluation support is too
   small, return INSUFFICIENT instead of fitting more variants.
4. For `retain iff predicted Y > 0`, report selected **sum/mean of Y over
   all sampled opportunities** (zeros for alternatives), alongside always
   retain, always alternative, a training-only constant baseline, and at
   least 500 predeclared matched draws preserving action counts and weights.
   Give day/side/market contributions, action count, movable-null support,
   MSE versus zero/constant, calibration, negative tail and top-five
   concentration. All-arm inventory/turnover diagnostics remain secondary.
   Require positive selected value and improvement over those controls on
   both evaluation days to describe evidence of useful local discrimination;
   thin matched support is INSUFFICIENT. These independent fork decisions
   are not a composed policy replay or a profitability result.
5. Bundle the two findings above, final-source controls, identity checks and
   the measurement into one review. No extra cleanup gate. Do not expand to
   a live strategy or claim the selected fills establish queue necessity.

**USER decision required before that experiment:** authorize this new
label-and-discrimination scope with **up to 1.25 additional core-hours,
including a 0.05 h review reserve, and a ladder-line ceiling of 4.00 h**.
The observed 0.2235 h / 64-pair pilot suggests roughly 0.89 h for 256 pairs
before additional controls; this is planning arithmetic, not a measured
guarantee. A fixed prefix pilot must project the whole round inside 1.25 h,
otherwise stop for a revised resource decision. Existing approval is only
0.40 h for the current pilot batch and 2.85 h for the line; spare balance
does not cover the proposed round. No extra days or untouched validation
are requested. Preparing its declaration is possible now; no new labels,
fit or replay are authorized by this review alone.
