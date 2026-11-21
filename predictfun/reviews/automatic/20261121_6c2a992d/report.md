# PI0-ADMISSION-EXPOSURE-v1 review — accept the comparison; diagnose movement calibration next

Review date: **2026-11-21 UTC**. Producing outputs as of **2026-11-21T11:12:01.503402+00:00**; independent checks completed at 11:21 UTC.

**Full target SHA: `6c2a992d8fc0e3e52b073c694687e674aa562f34`.**
Range: `a44f7dd807f7d8cf88d8967801652ef26ca93ad8..6c2a992d8fc0e3e52b073c694687e674aa562f34`.
Reviewed freeze `6b14953f19b3defae09f9d5cc13ce3d132efd3ff` and results at the pinned checkout `/tmp/pf-review-6c2a992d`. Own review 99fb165d is context. No implementation changes or push.

**No BLOCKING finding for these saved results or their reuse.** The accounting, selected population, stage targets, frozen predictions and corrected mean-score reading reproduce. Accept `admitted_longer_harmful` as a descriptive result on this sample. It does not establish that every early pull is necessary or that no conditional longer-resting policy can work.

**ONE next priority: PI0-MOVEMENT-CALIBRATION-AUDIT-v1**, a bounded diagnostic of the movement forecast on the existing admitted and forced populations, with the reporting/validator amendments below in the same round. Keep the target, teacher and controller frozen. **No further USER approval is needed for this consumed-day package.** New label dates, untouched validation, a longer episode end or expanded resources require a USER decision.

## What is now established

The committed manifest reproduces: **689 PI0 POST generations**, **329 selected**, **93 nonempty action-day × side × hour cells**, weights summing to **689**, from **155 worlds**. The source replay universe has 380 evaluable worlds and four explicit exclusions. All 329 selected generations are evaluable; all statuses remain in the stage tables: **329 entries, 299 first reviews, 268 first pulls**, plus 30 no-review and 61 no-pull statuses. All **896 saved model vectors**, including same-as-review duplicates, reproduce through the frozen predictor. Each selected gid, prefix, held-fill target, mark, END, weighted field and ledger decomposition checked in this review matches its declared source.

The factual reconstruction preserves the actual policy state and original absolute END. The intervention overrides only discretionary PULLs of the selected gid. It changes all such pulls until that generation closes; it is not a one-time delay of its first cancellation. Structural removal still applies. The hold-from target matches the intervention's selected-generation fills. There is no added minimum holding timer or reset inventory.

The reported weighted exposure differences reproduce: **−16.90 c on Oct26; −2.87 c on Oct27; −9.37 c pooled per admission**. The selected generation explains −9.27 c of the pooled difference. More exposure raises its weighted fill probability from 0.066 to 0.303, but reduces its END value from −0.287 to −9.557 c. This directly rejects unconditional longer retention of these admissions under the tested policy and endpoint.

The value mapping also reproduces: `V = edge × E[q] + E[qm]`. An independent decomposition on the same eligible rows locates the error more precisely than the fill-conditional residual alone:

| Stage | Model-valid n | Forecast minus realized value, c | Quantity/edge contribution, c | Movement contribution, c | MSE(V), c² | MSE(zero), c² |
|---|---:|---:|---:|---:|---:|---:|
| Admission | 329 | +11.099 | −0.882 | +11.981 | 4,043.27 | 3,982.00 |
| First review | 299 | +5.694 | −0.228 | +5.921 | 3,743.45 | 4,018.16 |
| First pull | 268 | +5.590 | −0.113 | +5.703 | 4,116.84 | 4,414.95 |

At admission, the movement prediction averages **−1.779 c**, versus **−13.760 c** realized. Its +11.981 c error decomposes exactly into **+0.310 c from the probability factor** and **+11.671 c from the conditional movement head**, using the identity specified below. Thus the request's suspicion about movement optimism is supported on this sample; there is no discovered multiplication/addition bug.

However, the movement head still improves MSE: its own score is **4,336.46 versus 4,395.20 c² for zero movement**, and including it improves value MSE from the mechanical baseline's **4,098.46 to 4,043.27 c²**. At first review the improvement is larger. “Not useful at all” or “remove movement” does not follow. The problem is economic calibration at policy-selected states, with uneven performance across sides/days, rather than a demonstrated total absence of movement information.

The first-review matched control reproduces: PULL 233/299, percentile **0.806**, 45/91 exchangeable strata containing 166 rows. It shows **no demonstrated selection advantage**, not proof of zero discrimination. The weights are constant within these sampling cells, so matching the per-cell counts also preserves the selected weight denominator. None of these 299 reviews crosses its admission's UTC-hour cell; the current count/side/hour comparison is valid despite using the admission hour in `stage_rows`.

These are related generations within two consumed days, without an interval or validation/profitability claim. Independent evidence: [data.json](data.json), [error_components.csv](error_components.csv), [interpretation.json](interpretation.json).

## Consolidated actionable findings

1. **P2 — NONBLOCKING: correct the source identity and keep observation arrival separate from the cause of churn.** `pf_pi0_admission.py:395-412` measures `new_bn_age`; `admission_tables.py:85-90` labels it “new BN trade,” repeated in the receipt's transition table. `pf_stop_model_review.py:1023-1042,1048-1058` obtains `bn_age` from the **bookTicker** clock; trades have their own timestamps used for flow windows. The 0.946 reading is therefore new quote/bookTicker observation, not new trade. The age arithmetic is a useful improvement over raw bit changes, but cannot identify a price shock or an adverse-flow shock. All 299 transitions having some new observation supplies no no-news comparison group. **Action:** rename this field in the next superseding table; document the three clock sources and units. Add an actual trade timestamp only if making a trade-arrival claim. Keep the distinction between arrival, direction/magnitude, elapsed time, order state and switching POST/RETAIN heads. No re-simulation is required for the rename; the short synthetic quote-only example and source trace are in `interpretation.json`/`verify.py`.

2. **P2 — NONBLOCKING: qualify the movement/probability interpretation and carry the correct units into the next diagnostic.** `pf_r2_shadow.py:173-181` defines `qm = value − sign × (decision_mid − order_price) × q`. This is **quantity-weighted decision-to-END movement**, including movement before a fill. It is not exclusively post-fill movement, and `qm | fill error = +38.568 c` is cents per filled generation, not cents per share or the all-opportunity value bias. `pf_pi0_admission.py:418-441` correctly computes the declared tables, but those tables alone do not isolate pre-fill versus post-fill deterioration or separate probability/conditional-head contributions. The independent identities above now verify the narrower movement-bias diagnosis. P has lower Brier error than the pooled-rate reference, yet underpredicts the weighted fill rate (**0.230 forecast versus 0.303 realized**); the highest bin forecasts 0.497 versus 0.712 realized. The observed bin fill rates are not strictly monotone (0.093 then 0.083 in adjacent entry bins). **Action:** say “useful probability discrimination with calibration error,” retain movement as an informative but biased component, and use explicit unconditional and conditional units. Add the exact component decomposition, calibration tables and clock-correct fill decomposition in the next package; do not change the valuation horizon merely to improve a score.

3. **P2 — NONBLOCKING scientific limitation: uncertainty is concentrated in a small number of weighted outcomes.** `pf_pi0_admission.py:377-390` and `admission_tables.py:65-69` explicitly report **unweighted** top-five concentration. The 0.37 figure is correct for that quantity. For the weighted economic estimand, the top five `|weight × D_EXPOSURE|` terms contribute **0.520** of absolute mass. Only **88/329** held generations fill, and 73 gain fills relative to PI0. The entry movement bias is strongly uneven: +34.23 c in Oct26 ask versus +0.46 c in Oct27 bid. **Action:** add weighted generation and world concentration beside the existing unweighted view, and report signed contributions by day/side. Treat this as a limit on the pooled calibration claim; do not delete large losses, select a winning subgroup, infer an interval from generations, or fit a global correction to the same admitted outcomes and call it validation. This is already covered by the next diagnostic, not a reason to rebuild the sample.

4. **P3 — NONBLOCKING for these artifacts; cheap validator amendment on reuse.** `admission_checks.py:163-180` checks selected keys, manifest weights and content hashes, but does not compare each result row's `weight`, `n_cell`, `cell` to its selected manifest row. `pf_pi0_admission.py:367-369,402,426,448` consumes those row fields in the weighted summaries/null. A rehashed row with doubled weight passes check A; reproduced in `interpretation.json`. Hashing a result is not a comparison to the declared sampling weight. **Current workaround verified:** this review compared all three fields on all 329 actual rows; every one matches. No current reading changes. **Action:** on the next adapter/checker touch, bind these fields by the exact generation key and include a known-bad row-weight/cell control. Derive a future first-review matching hour from its actual decision timestamp, or assert equality with the admission cell. No separate cleanup round or new mutation campaign is needed for these small checks.

There are **zero result-blocking items** in this checklist. Bundle these amendments with the next measurement. Deferred broad operational hardening is not a gate on that measurement.

## Prior findings and verification

The 99fb165d P1 is closed: the prior MAE-gated inference is superseded in this receipt and FINDINGS P97, with MSE, the exact-mean-mixture positive control and correct wording. The old receipt remains provenance. The forced-entry MSE is again 4,690.44 versus 4,785.28; first-review MSE is 4,794.91 versus 4,568.57. Both-modeled transitions are 717 with 34 flips. Model/fallback categories and all 12 old world-fork exclusions remain explicit in the cross-tab. The actual-admission cohort remedies the earlier forced-admission scope problem. The pilot is write-once and its saved arithmetic reproduces 0.597159 core-hours. Clock/source attribution is improved but needs the specific correction in finding 1. No old implementation blocker remains open.

The module, selftest, mutation harness, checker, declaration and manifest are unchanged from the pre-run freeze. Twenty-one committed output hashes match, compressed CSVs decompress exactly to the data outputs, all 12 raw part hashes match their committed records, and the prior cross-tab source tables match their accepted committed copies. The frozen teacher closure (76 research sources and model assets) verifies. Final-source mutation records judge 29/29 caught with a passing control; earlier suites remain provenance. Both synthetic suites and all 11 real-copy falsifiers pass. All saved table/predicate outputs reproduce.

All executable checks used `PF_SLOT=codex /home/yuqing/ctaNew/predictfun/research/run_light.sh` with the absolute path to [verify.py](verify.py), importing the pinned checkout:

| Check | Result | Wall seconds | Peak MiB |
|---|---|---:|---:|
| Metadata v1 | Reviewer path assumption failed; no result accepted | 1.52 | 230 |
| Metadata v2 | PASS: corrected location of prior committed exposure table | 0.99 | 221 |
| All saved parts, ledgers, targets, fresh archived-model inference, summaries | PASS | 6.03 | 452 |
| Synthetic suites and 11 falsifiers | PASS | 24.52 | 434 |
| Component identities, source semantics, weight-check control, pilot arithmetic | PASS | 2.29 | 423 |

Total **35.35 s wall**, including the retained reviewer path failure; four successful jobs, **33.83 s**; peak **452 MiB**. No fit, new labels, real tape simulation or untouched-day access. Producing-run CPU remains about 0.577 core-hours including the final report log; the receipt's slightly earlier 0.576 snapshot predates that final job.

Limitations: this review does not rerun the 155 real factual episodes. It checks their archived-source membership, saved prefix/target/ledger identities, effect times, forecast vectors and producer reconciliation flags, with real-engine synthetic controls for the reconstruction. Fill-time reference mids have not been independently re-extracted; they are a declared part of the next diagnostic if that decomposition is produced. These checks certify this bounded comparison, not every simulator path or a profitable strategy.

## ONE next priority — PI0-MOVEMENT-CALIBRATION-AUDIT-v1

Choose the **movement/adverse-selection calibration diagnostic**, using the saved cohorts to distinguish selection calibration, fill timing and queue/state effects. Do not first build another large absent-state label population or launch a new tree-model search. The current assets already contain the conditional moments and the matched economic outcomes needed to narrow that choice. There is no reason to declare the consumed days “exhausted” for diagnosis; they are exhausted as independent validation evidence.

**Population and immutable inputs.** Freeze the diagnostic code, source/output hashes, joins, strata, metrics, reading map and controls before computing its new tables. Use every one of the 329 admitted rows and all their stage statuses, with the existing inclusion weights. Entries, first reviews and first pulls remain separate estimands; same-as-review targets are not independent observations. Use all 768 previously declared forced forks (756 evaluable), with their 107 inputs and neutral held targets, as a **separate** unweighted reference cohort. Display the Oct26-27 overlap by date alongside the full Oct24-27 reference; never pool these differently selected samples. Separate shadow POST/WAIT/fallback. Keep all no-fill opportunities; a conditional-on-fill display is explanatory, not a training population. No new admission/absence census or new counterfactual branch is needed.

**Exact moment and value accounting.** At each stage define `z = 1(q > 0)`, `mhat = E[qm]/P`, `qhat = E[q]/P`, and observed `M = qm` (zero on no fill). Record explicit missing/zero-P statuses. Verify rowwise:

```
Vhat - Vreal = edge * (E[q] - q) + (E[qm] - M)
E[qm] - M = (P - z) * mhat + z * (mhat - M)
E[q] - q = (P - z) * qhat + z * (qhat - q)
```

These are accounting decompositions, not causal interventions that independently replace one head. They prevent attributing the entire conditional residual to all opportunities. Report mean/bias, MSE and calibration for each unconditional moment, conditional residuals separately, Brier for P, zero and mechanical baselines, weighted and unweighted counts, unique worlds and effective weight count. Keep MAE secondary. Preserve actual severe losses; include their signed contribution, weighted top-five generation/world shares and day/side breakdowns. No winsorization, selective exclusions or intervals.

**Timing and queue diagnosis, with no feature leakage.** Use decision-time `fp_queue` and `fp_remaining` (shares), `t_to_end` and generation age (ms), generation ordinal, side/stage and the existing P bins. Declare marginal strata in advance: remaining horizon `(0,100]`, `(100,500]`, `(500,2000]`, `(2000,5000]` ms; queue/remaining ratio `0`, `(0,1]`, `(1,3]`, `>3`, plus missing/invalid; ordinal `1`, `2`, `>=3`. Do not take their full Cartesian product on 88 filled admissions. Report side/day/stage totals plus each marginal view; cells below 30 opportunities or below 10 fills for a conditional statistic are descriptive/thin. Do not merge or tune bins after seeing outcomes. Venue-truth queue is a separate explanatory field and cannot be passed to a decision rule.

For filled tranches, classify actual delay from activation/effective review to fill into `[0,100]`, `(100,500]`, `(500,2000]`, `>2000` ms, with a no-fill class retained at opportunity level. This realized delay is an outcome, **never a feature or admission filter**. If separating pre-fill from post-fill movement, reuse the existing venue-mid lookup convention (`pf_adv2_replay.mid_at`) on the hash-bound tapes for these already selected worlds. For every fill verify:

```
sign*q*(END_mid - decision_mid)
  = sign*q*(fill_mid - decision_mid) + sign*q*(END_mid - fill_mid)
```

Record midpoint timestamps, age/staleness, missing-reference statuses and quantity denominators; do not move a missing reference to another clock or beyond original END. This is post-hoc economic attribution only. Reuse retained fill references where available; reconstruct only missing reference lookups, not policy episodes. The primary saved-target result remains usable if a bounded lookup cannot be completed; mark that attribution unavailable rather than silently dropping its fills.

**Selection, calibration and churn reading.** Show how movement residuals vary with the existing predicted value/movement and queue/horizon strata in admitted versus forced cohorts, without claiming that admission selection is random. Retain the existing 500-draw first-review control and report the exchangeable population. Count actual new bookTicker, depth and PF observations correctly; report direction/magnitude only where available from causally known inputs. New observations alone do not attribute the POST-to-PULL reversal to changed danger. The two heads and their effective windows differ.

The diagnostic must end with a computed evidence table answering: (i) is the mean bias concentrated in a few tail worlds or present across both days/sides; (ii) is it associated with decision-time state/remaining horizon or only with realized fill timing; (iii) does the conditional movement residual change materially between the neutral and admitted cohorts; (iv) are the available cells sufficiently populated to support a calibration candidate? Preserve inconclusive answers. An oracle corrected using the same realized outcomes is not a model score. This round fits no calibration map and selects no new trading threshold; it should specify whether a small **chronologically fitted** calibration candidate is defensible next, or whether more representative data/target representation is needed. Do not prescribe a new family from the pooled MSE alone.

**One acceptance gate and budget.** Bundle findings 1–4, the new measurement and its controls in one freeze/results round. Positive controls: exact calibrated mixture, independent probability versus conditional-head bias, equal total movement split into pre-fill versus post-fill paths, a quote-only update, and a clock-only transition. Refuse wrong row weights/cells, wrong units/signs, moved fill/effect/END times, substituted source parts, and future/venue-only information in decision-time strata. Reconcile complete membership, every decomposition and all output denominators. Keep the frozen teacher/features, latency, size, cap, structural rules, terminal mark and original END unchanged.

Budget: **one aggregate core-hour**, existing authorized development slots (at most four single-core workers), 1,536 MiB checkpoint / 2,048 MiB hard bound. A fixed first-part pilot precedes any additional raw-reference extraction; stop at the bound without outcome-dependent thinning. No new model fit, full-day label build, new date, extended END or hold timer. No intermediate repair-only review.

**The USER's longer-holding direction still has a consumed-day path:** conditional retention based on a calibrated value for the same longer-lived exposure, with the entry/retention target and controller sharing that policy. This diagnostic is preparation for that test; the present unconditional KEEP result does not reject it. It does reject using duration alone as the fix. Testing an extended horizon, obtaining more label days or claiming validation on at least five untouched complete days needs the USER's scope/data/compute decision. None is required to complete this one next diagnostic.
