# R2-C teacher review — 2026-11-20

Pinned target: **4584a3c2696625d27cf87f193a6260997348f85e**.
Requested range: **294fd172d7712bdfeb4c776a33ba63eaf1b7fc0a..4584a3c2696625d27cf87f193a6260997348f85e**.
Reviewed the development commit from `/tmp/pf-review-4584a3c2`; the intervening
review-only commit 3d7eae6a was the prior contract, not a new development change.
[Development receipt](../../r2c_teacher_20261120/report.md),
[prior review](../20261120_294fd172/report.md).

**Accept the Oct23 training artifacts, frozen pi0_50_v1 models and bounded
single-branch rollout. One integration issue BLOCKS using the current teacher
directly for paired candidate-label production.** `Teacher` retains a bound
branch while `BE.pair` executes copies. The next deliverable is the branch-aware
paired-label adapter and its bounded Oct24–25 gate. No teacher refit, new
threshold or additional USER approval is needed for that work.

There are also two NONBLOCKING findings: resource dispositions match filenames
rather than individual stop events, and the value diagnostic mixes a coherent
numerator with an all-row denominator. Neither invalidates these saved models
or the reported single-branch actions. This review establishes implementation
and artifact validity, not forward prediction quality or profitability.

## Ranked findings

### P1 — BLOCKING for paired-label production: the teacher is not bound to each executed arm

Evidence: `pf_r2_teacher.py:389–390,398,400–402,425–428`;
`pf_branch_engine.py:449–462`.

`Teacher.fn(v)` calls `self.decide(v, self.B)`. Its feature builder reads the
generation, fills and client context from that stored `self.B`. `BE.pair`
deep-copies the saved state separately for POST/WAIT or KEEP/PULL and passes the
same policy object into both copies without rebinding it. A view from one arm
therefore does not identify the branch supplying its features. Binding once
to the saved fork, or to a prior rollout, is insufficient.

[Independent synthetic reproduction](pair_binding.py), [result](pair_binding.json):
the actual pinned `Teacher.fn/bind` and `BE.pair/step/decide`, with only the
predictor replaced by a state spy, give **98 branch/generation disagreements
in 397 callbacks**. At 100 ms, the POST arm's view has generation 0, but the
teacher's bound branch has no open bid. The engine nevertheless returns `ok`.
The positive single-branch control has **198 callbacks, zero disagreements**.
This is an interface reproduction, not an estimate of incorrect real labels;
no candidate labels exist in this receipt.

The existing Oct23 rollout is valid on this point: `pf_r2_teacher.py:568–572`
binds the actual branch it subsequently steps. Its inputs, support and action
checks are not invalidated by this finding. Resolve the integration in the
new label adapter, binding a teacher to each actual arm before *either side's*
continuation decision, including the remaining side at the fork tick. Keep
the saved fork immutable and preserve the engine's alternating order. The
frozen models, feature builders and engine need not change.

### P2 — NONBLOCKING: an old disposition clears a different stop with the same filename

Evidence: `pf_r2_runner.py:169–177,213–215,293–296`.

Runner v3 now carries earlier breaches and checks them at resume/finalize.
However, `held_breaches` subtracts `disposition.statuses` by basename only.
After `run_status.json` is cleared, a subsequent process can overwrite that
file with a new hard-RSS stop during loading. The old disposition clears the
new event too. [Reproduction](controls.json) verifies both clean behaviors
(unapproved stop held; that stop explicitly cleared), then changes the status
bytes from an earlier STOP_RESOURCE to a new STOP_HARD_RSS: `undisposed` still
equals `[]` despite different hashes.

Bind dispositions to immutable event identities/content hashes and the plan,
with a reason; preserve the stop history rather than treating a filename as a
permanent waiver. Fold this into the next development round. Current R2-B had
no breached records/stops and needed no disposition; this teacher does not use
the runner's resume path. No rebuild is required. Until corrected, independently
check every stopped attempt and its exact disposition before accepting any
resumed output. A new undisposed stop must not be accepted.

### P3 — NONBLOCKING: value diagnostics do not use a consistent mean population

Evidence: `pf_r2_teacher.py:495,501,504–510`.

`V_pred_wmean_c` and `V_real_wmean_c` sum only coherent rows but divide by the
weight of **all** rows. They are neither the conditional coherent-row means
nor ordinary all-row means; excluded rows implicitly contribute zero. There
are 5,981 incoherent rows across the four groups, so this is exercised.

[Frozen-model recomputation](short_diagnostic.json) reproduces the saved
<=100 ms fields on **137,738 rows**. For ask POST, the reported predicted mean
is -0.504842824 cents versus the coherent mean -0.505497755. The discrepancy
is small here, but the estimand should be explicit. Publish coherent weight
alongside means normalized by that weight; report the full-population realized
mean separately. Do not treat fallback KEEP as economic zero. Supersede the
diagnostic receipt if corrected; preserve the original freeze/models. This
reporting issue does not affect `Teacher.decide`, its predictions or actions.

## Training, mapping and freeze

The extraction at `pf_r2_teacher.py:165–226` checks the producing plan, ordered
records and parquet identities, keeps **every sampled ok row**, and rejects a
row outside the four groups. No zero/no-fill/refusal outcome filter was added.
`fit_group :294–315` correctly fits p and severe on all rows and q/qm on
q>1e-12 with the corresponding weights subsetted. All 16 saved boosters have
the expected 107 feature names and **200 iterations**.

The two declared plumbing choices are acceptable for this version:

- **Mean-one weights per training set.** `:277–287` scales by n/sum(w), preserving
  relative whole-original weights, including within each conditional subset.
  Scaling affects L2/hessian regularization, so this is a versioned choice,
  not an invariant transformation of a raw-weight fit. The recorded dictionary
  remains pinned; do not describe its effective fit as identical to raw census
  weighting. The header's approximate leaf-shrinkage explanation is illustrative,
  not a universal tenfold effect. Independent full-file checks find 15,044
  original weight groups per side/head, each summing to one within 8e-15.
- **File-backed Sequence.** `:235–274` preserves row/feature order; the synthetic
  selftest independently passes exact streamed/in-memory tree equality at its
  small size. Archived 150,000/199,999-row controls report equality; the declared
  250,000-row control reports different deterministic trees. Those larger
  construction controls were inspected, not rerun. Above the bin sample size,
  the chosen streamed construction is part of the model version. No claim of
  equivalence to the legacy large in-memory fit is warranted.

`predict_rows :377–380` uses the unchanged `pf_pipeline_bench.predict` mapping:
conditional filled quantity/movement, known-remaining clip, probability-weighted
value and two-atom coherence. The support adapter `:139–161` retains coarse
200/20 support, observed category presence, mandatory context ranges and the
inventory tolerance. `decide :406–423` applies physical guards then support,
coherence and E1 signs/ties in the declared order. Severe is a loss regression
diagnostic, not a newly selected cancellation threshold or severe-event
probability. Its score does not enter E1.

The **76-module source binding is appropriately conservative** for this
transitive import graph. All 76 actual pinned files, libraries, feature order,
parameters, data-manifest links, model hashes and decompressed committed model
bytes match. Freeze SHA:
`769dff779d584f913aa9c142fb852a47eddbe01bdd6c0930c87b9d28dfa73294`.
All committed checkpoint files also equal their actual data copies.

The JSON freeze was written at **11:48:25Z**, preceding diagnostic/rollout
completion at **11:50:42Z / 11:58:18Z**; the complete commit reviewed here is
the repository freeze under reliability rule 12. These pre-commit consumed-day
diagnostics are not prospective validation. The source/model hashes establish
which implementation produced the accepted artifacts; no alternative teacher
was selected in this receipt.

For labels, pin the exact freeze hash and model hashes, enforce its library
versions in the new runner, and record the label adapter's own source closure.
`frozen()` checks source/manifest/support bytes; libraries are recorded rather
than automatically enforced there. They match now. Add new orchestration outside
the frozen 76-file closure; do not silently rewrite the freeze to accommodate an
unrelated source edit. Any necessary changed behavior gets an explicit new
execution-contract identity, not a disguised replacement of pi0_50_v1.

## Saved artifacts and readings

Independent evidence: [metadata](metadata.json), [extraction sample](samples.json),
[output reconciliation](outputs.json), [short-horizon recomputation](short_diagnostic.json).
As-of **2026-11-20 12:14:26Z** for completed result checks; all market data used
by this review is saved **consumed Oct23** training/rollout data.

| Check | Result and scope |
|---|---|
| Training files | All 96 hashes, 3,473,420,460 bytes, match the manifest |
| Four groups, all / filled | Bid POST 327,341 / 97,582; bid RETAIN 2,353,692 / 573,171; ask POST 327,264 / 97,868; ask RETAIN 2,375,718 / 578,799 |
| Conditional subsets | Every conditional label/bookkeeping/weight array equals the exact positive-q subset of its all-row array |
| Extraction row sample | Fixed 20 parts `0–3,116–119,236–239,356–359,473–476`; 219,999 sampled-ok rows; all 107 input columns and label/weight fields match bit for bit; conditional X slices also match |
| Support diagnostic | 28 saved side/head/horizon counts match accepted census support; 80-category/28-cell reconciliation is PASS in the generating diagnostic |
| Models | All 16 actual model files and committed gzip contents agree with fit/freeze hashes; names and 200 iterations checked after load |
| Rollout | 128 worlds, 127 completed, one `no_path:missing_endpoint`; 254 side-episodes; 24,541 policy decisions |
| Actions | 337 POST, 262 PULL, 456 KEEP; all logged model signs/ties checked; 28 post-only refusals and 20 fills in saved side/world summaries |
| Entry | Bid/ask delayed first entries 55/70, offset-zero 5/5, none 67/52; 202 later generations across 80 side-originals |
| Guards/fallbacks | 690 cap-blocked and 176 near-END decisions; zero support/forecast fallback and zero engine fallback in this slice |
| Saved visibility/parity | 3,506 hidden-state checks, zero mismatches; 3,271 delivered changes; 3,384 cached/uncached rebuild comparisons, zero mismatches |

The rollout is not degenerate under the declared delayed-entry/fallback gate.
It still mostly WAITs; that is a descriptive reading, not evidence of good
timing. POST counts divided by absent decisions are not quoting-time fractions.
Short-horizon overprediction is real: bid POST fill probability is
**0.022671 vs 0.012797**, conditional q×movement **-34.771 vs -3.702 cents**,
and severe loss **1.41059 vs 0**, in the 3,175-row <=100 ms bin. There are only
39 distinct filled originals there. The larger-horizon aggregate calibration
is in-sample and does not prove ranking or tail discrimination. Use **>100 ms**
for the complementary bin wording; exactly 100 belongs to the short bin.

The saved visibility sample includes 112 live checks; the delivered mutation
changes no live check. Do not claim that positive control fires in each state.
The synthetic suite separately reaches live states and detects the known-hidden-
remainder leak (483 catches) and unreceived-frame leak (515 catches). Together
with the earlier feature-builder audits, these support the bounded causal claim.

`decisions.csv` lacks explicit world identity (`teacher :408,576`), but its
chronological blocks can be reconstructed from `worlds.logged`. I independently
reconciled every block to per-side statuses, POST/PULL counts and first-submit
times. This supports the receipt's aggregates. The next label artifact must
carry explicit original/fork/arm identities rather than rely on this ordering.
The saved CSVs do not contain per-fill ledgers; the 127 ledger PASS flags are
the generating run's instrument results, supported by reviewed source and
rerun synthetic falsifiers, not an independent real-tape fill replay here.

## Prior findings and instruments

The standalone-finalizer source preflight is **closed** at runner `:266`;
the receipt now records actual finalizer sources. Watchdogs cover loading,
run and finalize (`:116,186,267`). The former unconditional loss of resource
history is fixed, with the event-identity gap in P2 above still open. P80's
parquet/count/logging instrument fixes are present; their distinct falsifiers
and both sides' own-original routing are exercised by the teacher selftest.

The teacher and runner selftests pass again on pinned source. All 44/25 mutation
replacement sites match the declared counts; the archived suites record each
unchanged control passing and 44/44 and 25/25 errors caught. Entire mutation
suites were not rerun. The first teacher run's 40/43 result remains disclosed.

All nine newly retained R2-B v1 files equal their v2 counterparts byte for byte.
This closes the missing-snapshot comparison limitation. The claim about when
the scratch copies were made remains the development session's provenance;
git bytes alone do not independently prove that historical copy time.

## ONE concrete next priority

**R2-D: build the arm-bound candidate-label adapter and a bounded Oct24–25
integration receipt under the existing frozen pi0_50_v1.** This is part of the
already authorized implementation/review loop; **no additional USER decision
is required**. Acceptance of the teacher does not authorize treating an
unbound `BE.pair(..., teacher, ...)` result as a valid label.

1. **Execution adapter first.** Reuse the validated saved-fork context and
   existing engine primitives; bind a separate teacher to each executed arm.
   Keep each side's own original/world features, same common END, notices,
   reservations, alternating order and before-either-arm validation. On a
   single branch, reproduce the current teacher actions. With stateless
   stand-ins, reproduce `BE.pair` exactly. A deliberately stale-bound teacher
   must fail; reversing arm execution order must leave labeled outputs unchanged.
   Cover POST/WAIT and KEEP/PULL on both sides, delayed/re-entry/partial states,
   cancel-pending fills and the remaining-side decision at the fork tick.
2. **Unchanged economic contract.** Export whole-portfolio gross cents:
   `Y_post = R(POST now, then pi0) - R(WAIT one tick, then pi0)`;
   `Y_keep = R(KEEP one tick, then pi0) - R(PULL now, then pi0)`.
   Retain 5,000 ms original END, 50 ms decisions, 58/37/100 ms timing and zero
   fee/rebate convention. Save both returns, cash, inventory, END mark, open
   exposure and event/fill hashes; reconcile each component and identical
   pre-fork histories. Teacher RETAIN forecasts are continuation inputs; they
   are not the new KEEP-minus-PULL target. Missing or infeasible pairs are
   statuses, never zero labels.
3. **Population/identity before labels.** Preserve plan v2 section 5: neutral
   reference states plus the declared deterministic shadow sources and pi0
   rollout states, identified separately. Retain original-time POST probes,
   every-fifth-tick rotating-phase training sampling, zero/no-fill/refusal
   outcomes and equal total episode/side/head weights. No positive-score or
   successful-fill admission filter. Declare the immutable sampling/source
   manifest, full row key including scenario, END and fork fingerprint, and
   explicit arm/decision log identities before computing paired outcomes.
4. **First reviewable real receipt, bounded.** After synthetic controls pass,
   take the first 32 shared `(market,t0)` worlds of each consumed Oct24 and
   Oct25 in the committed original ordering. Within that prefix, label at most
   the first 16 feasible sampled pairs per side × action pair per day in a
   declared stable source/time/key order: **at most 64 pairs/day, 128 total**.
   Keep the prefix census and all excluded/unsampled statuses, source counts,
   head coverage and unavailable cells. This cap is an integration sample,
   not a replacement training manifest or a skill gate. Do not expand it
   after reading outcomes to make a missing cell appear. Synthetic controls
   must cover states absent from this prefix.
5. **Gate and resource receipt.** Require exact context/feature identity,
   zero ledger/target reconciliation failures, real-row arm binding checks,
   cached/independent feature parity, visibility controls and persisted/resumed
   output identity. Report support/fallbacks per arm/source/side, time and peak
   memory, plus measured seconds per pair before projecting full-day cost.
   Use the existing slots/limits and stop/disposition accounting. Fold the
   NONBLOCKING reporting/resource fixes into this round, outside the frozen
   teacher closure where possible. Finish this deliverable at its bounded
   receipt; review the adapter before full paired-label production. No teacher
   tuning, candidate fit, Oct26–27 evaluation or untouched-day access is needed.

The subsequent full training population remains consumed Oct24–25; Oct26–27
are consumed development evaluation and Oct28 was already seen. Only later
untouched complete days can serve validation under the existing plan.

## Review execution and ownership

Eight successful bounded codex-slot checks: teacher and runner selftests,
metadata, fixed extraction sample, saved-output reconciliation, independent
controls, actual pair-interface reproduction and short-horizon predictions.
**82.17 s total, maximum 1,068 MiB**. The only fits were tiny synthetic fixtures
inside the existing selftest; no real-data fit or label build ran.

One 5.48 s review-instrument attempt failed because the metadata checker assumed
both mutation harnesses named their table `MUTANTS`; runner calls it `MUT` and
includes an unchanged control. [Original script](verify_v1.py) and
[failed log](metadata_v1.log) are preserved. The [corrected script](verify.py)
handles each actual schema; all emitted evidence hashes bind that version.
This was not an implementation failure or a passing test.

No real tape experiment, untouched data, development edit, push or external
message was performed. Other-session dirty files and artifacts are preserved;
only this review's evidence and review-owned state prefixes are committed.
