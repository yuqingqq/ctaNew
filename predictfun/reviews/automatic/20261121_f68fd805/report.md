# R2-F combined review — accepted production labels; first candidate experiment next

Date: 2026-11-21. Pinned target: **f68fd805ca6ba52c411b7d570aaa791f10d17feb**.
Review range: `714ce75978ab8dfec29c1fc41fd0dd800b982ad8..f68fd805ca6ba52c411b7d570aaa791f10d17feb`.
Producing freeze: **d3d024fd2c5ccb1ca7e4d33e9b1162b91b50abd5**.
Prior review: `38febc0b076c1277c27c59e4847000b6615355e6`.
Reviewer: Codex; development owner: Claude aea5bfb9. Evidence as of 2026-11-21 00:38 UTC.

## Verdict

**Accept the full R2-F dataset. No blocking finding for proceeding to the candidate experiment.** The prior status/END admission gap and zero-callback false refusal are closed. All 96 production partitions, stored accounting and artifact identities pass this review's bounded checks. The every-35 audit preserves the 288 pilot economic/decision payloads; skipping the optional cache is appropriate.

There is no justification for another fix-only round or rebuilding these labels. The current limitation is sparse information about action value, particularly retention. Passing the engine and accounting checks establishes usable research labels, not a profitable strategy or a predictive model.

**ONE next priority:** implement, freeze and run the first **ADV2-RICH direct paired-value prediction experiment**, with its fixed FAIR/MECH controls, on consumed days; include a bounded R4 own-state diagnostic replay in the same round. The concrete contract below preserves Oct24–25 training and Oct26–27 evaluation. **No additional USER approval** is needed within the existing four development slots/resource limits. No untouched day, model search or target change is authorized.

## Consolidated actionable checklist

### 1. P2 — Retention and on-policy support are thin — NONBLOCKING for an honest diagnostic experiment

The receipt's pooled 543 nonzero rows conceal much smaller per-model populations. Independently counted from all saved rows:

| Day | Bid POST nonzero originals | Ask POST | Bid KEEP | Ask KEEP |
|---|---:|---:|---:|---:|
| Oct24 | 99 | 87 | 48 | 42 |
| Oct25 | 71 | 91 | 42 | 40 |

Each POST group has 3,042 rows from 1,521 originals; each KEEP group has 1,519–1,521 rows/originals. These are before splitting by predicted POST/WAIT/KEEP/PULL, remaining horizon, generation or tail severity. Four Oct24 counts each include one floating-point-sized effect (item 2).

There are only **36 pi0 KEEP rows** across both days (10/13 bid/ask Oct24; 9/4 Oct25), and only three have an effect exceeding 1e-6 c. Most KEEP training rows are delayed shadow states, not states the continuation commonly visits. This is a real coverage limitation, not evidence that the sampler failed. Source evidence: receipt `coverage.csv`, `population.csv`; `pf_r2_label_sample.py:351–380` preserves the five-source pools and sampling; [audit.json](audit.json), `group_support`, verifies per-model counts.

**Action in the candidate round:** report per-side/head and action-selected support, source/horizon/queue/partial/generation coverage and own-state exposure. Do not use the teacher's supported status as proof that the new advantage model is calibrated. Keep the existing 200-original / 20-nonzero-original evaluation floors; insufficient groups remain `INSUFFICIENT_SUPPORT`. Do not pool bid/ask to manufacture support or infer tail-probability skill from a mean-value regressor. This limits claims; it does not block fitting the declared model or diagnostic replay.

### 2. P3 — Exact nonzero counts include four arithmetic residues — NONBLOCKING; correct the support interpretation in the same round

`predictfun/reviews/r2f_production_20261120/r2f_checks.py:170–175` defines nonzero/sign by literal `Y != 0`, `Y > 0`, `Y < 0`. Four Oct24 labels have magnitudes from **1.42e-14 to 4.55e-13 c**. They are listed with keys in [audit.json](audit.json), `near_zero_nonzero`. Thus **543 exact nonzeros is reproducible, but only 539 exceed the existing 1e-6 c accounting tolerance**.

Preserve raw labels, weights, original tables and training targets. Add a separately named `abs(Y) > 1e-6 c` support diagnostic; do not count floating-point residues toward the 20 meaningful-event floor. Report exact counts alongside it. This is not tail clipping, outcome filtering, a new action threshold or a reason to rerun labels. The aggregate negative means and sparse-target conclusion are unchanged.

### 3. P3 — Top-level suite gate still reads summary text — NONBLOCKING; a verified workaround already covers this receipt

`r2f_checks.py:151–155` checks the 113/113 summary and success phrases; it does not itself join every mutant record to the final harness/selftest hashes. The revised `pf_mutant_summary.py:29–45,67–70` now checks index/name as requested. This review additionally verified **all 113 records plus the control**, their exact indices/names, classifications, one-record-per-file and final source hashes. No wrong identity was found.

Reuse that cheap per-record identity check in subsequent combined receipts, or call the existing summary against the final sources directly. No new mutation campaign is required solely for this integration cleanup. Older v1/runner suites remain historical passing evidence with unchanged producers, not new executions by this review.

No other concrete result-invalidating defect was found in the changed producer, its status/callback integration, sampling, ledger admission, parity comparison or streamed receipt. Historical deferrals remain nonblocking because they are outside this production path. This is bounded review coverage, not a claim of exhaustive correctness.

## Closure of 38febc0b

| Prior item | Finding at the pinned source | Disposition |
|---|---|---|
| Status hiding failed invariants | `pf_r2_label_sample.py:599–607`: outer status is checked against pair/fork status; only `missing_mark` is an allowed failed-pair exclusion; feature exclusions run `_pair_bad` with only the input requirement waived. `_pair_bad:522–523` requires the actual feature-unavailable reason. | Closed. The former corruptions are refused; both actual feature exclusions have valid Y=0 pairs and pass all applicable invariants. |
| Effective END/common mark/fill time | `:485–491`: END equals declared 5,000 ms and absolute identity duration; each arm uses the shared finite mark; fills lie in [0, END]. | Closed. Independently checked across all executed production pairs, not only synthetic fixtures. |
| Legitimate zero callbacks | `:308–318,453–475`: stores the executed decision log, binds its hash, counts only eligible unforced/non-fallback decisions, reconciles calls and binding checks. The engine omits pending/terminal states from this decision log (`pf_branch_engine.py:305–331`). | Closed. The real-engine synthetic final-tick +5 c case passes; all 71 zero-callback arms in ok rows reconcile. Missing required callbacks remain a failure. |
| Declared audit rate | `:115,353,374,513–517,674`: plan-bound every-35 continuation probes; checks equal ceil(callbacks/35). | Closed. Counts independently reconcile for both arms, including legitimate zero counts. Full binding checks and fork checks remain. |
| Reading corrections | Superseding pilot table uses nonzero-original union, path-only/economic differences, elapsed versus remaining horizon, and measured cost. | Closed. Pilot payload parity makes the corrections refer to the same labels. Item 2 above is an additional precision caveat. |
| Mutation identity | `pf_mutant_summary.py:29–45`: expected index/name enforced. | Closed in the summary; top-level integration limitation covered independently (item 3). |
| Optional day cache | Not implemented. | Accepted deferral; no missing gate. |

The removal of equivalent mutant #100 is justified by the remaining status-specific checks. The six unmasked/repaired test cases strengthen the test instrument; the producing module is unchanged between freeze and results. The retained 107/114 attempt is not hidden and is not counted as passing evidence. The v1.1 count change and v1.2 streaming change do not alter labels or select a model. All seven equivalence CSVs match byte-for-byte, and checks.json matches except `asof`; independent changed-count/control checks distinguish differences.

## Independent artifact review

- **Plan:** `0de151d1b980661d739e12c880f71eb1c81cca680ef6cef7ea97fcb36ce6c7fc`.
- **Dataset:** `2c62fc9baf696e6cd8913c5c294dea2c9513640714ae34b413dc37ef8c0acb2e`.
- Both committed compressed plans decompress to the saved plans. All 81 producing-source hashes match the freeze. The unchanged frozen teacher's 16 models, libraries and input schema match. All **221 input files / 1,222,090,373 bytes** named by the current identities were hashed; all **386 checkpoint entries** match the actual data files/statuses/plan.
- The full plan retains the prior selected worlds/probabilities: **3,072 worlds**, 1,536/day. **30 worlds** have no path, retained as 120 side/head status rows. Their four status placeholders account for the difference between the 18,432 maximum and 18,372 saved rows. There is one empty KEEP pool and two feature-unavailable KEEP rows. **18,251 pairs ran; 18,249 are ok.** No missing-mark failed pair occurs in this build. Exclusions are not silently converted to zero targets.
- All 96 partitions pass the pinned shared validator. Independently recomputed: cash/position from six-field fills, generation quantity limits, END/common-mark value, paired Y, fill time bounds, decision-log hashes, actual callback counts, action/status counter totals, audit denominators, 107-input float32 hashes, accepted manifest/dataset hash, sign/union counts and weighted means. No infinities in the stored inputs; sparse NaNs remain visible and should retain native missing-value handling in fitting.
- **4,435,006 binding checks; 145,045 causal checks; 138,385 input-parity checks; zero saved failures.** The two feature-excluded pairs were checked separately rather than silently omitted from validity accounting.
- **17,706 exact zeros**, 203 positive and 340 negative labels from 493 distinct originals in 468 worlds. Raw mean −0.410 c; weighted POST −0.476 c and KEEP −0.386 c. The receipt's tail/concentration readings are consistent with the saved outputs. These are counterfactual local labels under pi0, not a strategy's additive P&L.
- Fill paths: 16,776 identical, 926 path-only differences with equal cash/inventory, 543 literal nonzero-Y differences, and **four Y=0 cases with different cash/inventory**. Equal END value therefore does not imply equal inventory exposure. The receipt correctly retains that fourth category.
- Rechecked all **288 pilot rows / 192 pool lines** against the old accepted pilot. Canonical economic/decision payloads match with only the declared audit counters and newly added log fields excluded. The old decision hashes remain compared. A changed Y fails this comparison; changing only an excluded audit counter does not.
- All 96 production job logs show rc=0. CPU sum **15.4578 core-hours**, wall-time sum 15.6574 hours, no stops. The actual maximum production partition job RSS is **1,181 MiB**; the broader receipt's 1,271 MiB bound includes other jobs and is conservative. Resource projections were recorded during the run, not just after it.

## ONE next priority — ADV2-RICH v1, a fixed prediction test with bounded own-state attribution

This is **R3 prediction followed by an R4 diagnostic**, not an automatic R4 pass. Plan v2's R4 is the own-state replay stage; the prediction fit must precede it. Deliver code, freeze, readings and replay in **one combined round**, with no review between each adapter, fit and bounded measurement.

### Candidate, target and training

1. **Keep the declared primary:** `ADV2-RICH-v1`, four independent direct squared-error regressors: bid/ask × POST/KEEP. Labels are the existing signed cents, `R_POST − R_WAIT` and `R_KEEP − R_PULL`, under frozen pi0_50_v1. Use **all available ok Oct24–25 training rows**, including exact zeros and complete tails. Do not first shorten the history, train only nonzero rows, balance classes, winsorize, log-transform Y or replace mean regression with a classifier. Sparse Y alone has not demonstrated that the pinned regression model fails.
2. Fit `ADV2-FAIR-v1` and `ADV2-MECH-v1` as the already declared input controls, on exactly the same rows, labels, weights and split. RICH remains primary regardless of rankings. Declare **one primary plus two input controls / 12 fitted heads**; the existing teacher's 16 heads are separate. Do not add a hurdle/gain-loss/severe model, calibration map or smoother variant in this first measurement.
3. Inputs: RICH's stored **107** ordered float32 features; FAIR = ordered union of `fair_edge`, `t_to_end`, all 16 `pf_r2_shadow.CTX` fields; MECH = ordered union of `pf_flow_price_common.MECHANICAL`, `t_to_end`, those same context fields. The expected control counts are 18 and 40; assert actual names, uniqueness and subset membership, and freeze full lists/hashes. Preserve native NaNs and existing units. No IDs, source labels, day/time identities, hashes, chosen branch outcomes or teacher/future scores as model inputs. Keys remain joins. No extra learned normalization is needed for this tree baseline.
4. Preserve stored `w=1/(pi*N*q)`. For each training side/head use **one** multiplier `n/sum(w)` on its weights, identically in all three input fits, matching the teacher's declared L2 convention. Do not renormalize separately by label sign/source/outcome. Keep raw weights for reporting. Freeze the exact extraction masks, row keys, feature bytes, labels and raw/normalized weight hashes.
5. Resolve the pinned parameters, changing only objective to `regression`: learning_rate=.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, seed=1, num_threads=1, verbose=-1, lambda_l2=10; **200 rounds**, no early stopping/search. Preserve END=5,000 ms, 50 ms decisions, M50's 58/37/100 timing, size/cap 30, queue rules and explicit zero cost convention. New code stays outside the frozen teacher/label closure.

### Evaluation population and freeze

6. **Training: Oct24–25. Evaluation: consumed Oct26–27**, as plan v2 already states. No random train/test row split, no reuse of a training day as forward evidence, no Oct28 or protected-day access. Purge episodes crossing the date boundary. Commit the builder, candidate/controller rules, parameters, sample/null specification and diagnostic gates before fitting/evaluation; then record model hashes before evaluation outcomes are read. Both evaluation days remain consumed development, not validation.
7. For the **offline prediction diagnostic**, extend the reviewed deterministic sample contract to each evaluation day: up to 64 worlds per UTC hour, same canonical ranking/domain/ties, five-source census, mandatory POST + uniform extra POST + uniform KEEP, recomputing that day's inclusion probabilities. Maximum **3,072 worlds / 18,432 selected labels** across Oct26–27. Do not change membership based on predictions, Y or fills. Keep missing/unavailable/unsupported statuses. All three input models predict the identical rows.

   This is an explicit **sampled prediction diagnostic** using the existing census/weights, not a claim to have labeled every 50 ms opportunity. Actual controller decisions below remain every 50 ms. Avoid the previously infeasible full every-tick label build. In-sample training readings are allowed but must be separate.
8. Keep teacher support and candidate support distinct. For the first diagnostic controller, declare outcome-independent input eligibility from the training data: valid horizon/action state; observed horizon bins and categorical values; the existing `R2.NUMERIC` ranges; at least 200 distinct training originals in the side/head being used. Compute the manifest from Oct24–25 only, common to the input controls. Report finer horizon/source/partial/generation counts and nonzero counts separately. Do not reinterpret teacher support or this input-domain check as statistical calibration. Unsupported/unavailable candidate inputs use WAIT when absent, KEEP when live, with separate reasons and full denominator counts. No timer retry or fitted uncertainty band.

### Prediction metrics that cannot pass by predicting zero

9. Report by day, side, head, selected POST/WAIT/KEEP/PULL, and fixed plan-v2 prediction bins:
   - raw rows, distinct originals/worlds, exact nonzeros and `abs(Y)>1e-6` originals; support, source, remaining-horizon and fallback counts;
   - raw and sampling-weighted mean prediction/label/bias; fixed-bin calibration; weighted MSE/MAE **against zero and the Oct24–25 weighted mean for that side/head**, plus FAIR and MECH; rank is secondary;
   - signed local decision value with `d=1[prediction>0]`: `d*Y`, and regret `max(Y,0)-d*Y`, weighted on the same eligible population. For both heads d=1 means POST or KEEP. Include all zeros. These are local pi0-conditional diagnostics, not summed policy profit;
   - missed positive value and admitted negative value, overall and for |Y|>30 c; tail sums and event counts. The direct model does not output a severe-event probability, so do not claim probability calibration;
   - plan-v2 sign/amount screens unchanged: 200 distinct originals and 20 meaningful nonzero originals per day/side/action group, then the signed group-mean and `abs(mean_pred-mean_Y) <= .05 c + .20*abs(mean_Y)` checks. Report exact-count sensitivity for item 2. Empty/zero and insufficient groups cannot pass. No two-day confidence intervals.

10. Include the predeclared local action-information control. Use the mandatory original-time POST anchor (if available) and the one sampled KEEP anchor per original/side/head; do not substitute an outcome-selected anchor. In each day/side/head/source/instrument/minute stratum shuffle the candidate's selected-action count across eligible anchors for **500 fixed-seed draws** (seed 20261121, deterministic group ordering). Apply each draw to the same paired Y with a fixed all-anchor weight denominator. Keep singleton/no-variation strata visible; they supply no randomization information. Show lift over the matched mean and the full draw distribution, not a p-value/interval treated as independent-day proof. This control is additive local selection, not a continuously randomized policy replay.

### Bounded R4 diagnostic in the same round

11. On the **nested first eight selected worlds per UTC hour per evaluation day** (at most **384 worlds** total), run joint own-state policies at every 50 ms: N0, frozen PI0-50, RICH, FAIR, MECH, ENTRY-ONLY, RETAIN-ONLY, ENTRY-PA. Each recomputes causal inputs on its actual branch; no cached neutral score joined by timestamp. Hybrids use their own state. ENTRY-PA's future admissions are not forced to match RICH. A failed prediction screen still permits this explicitly diagnostic replay; it does not permit promotion or a silent refit.
12. Controller mapping stays in the policy layer: absent POST iff supported predicted POST advantage >0, otherwise WAIT; live PULL iff supported KEEP advantage <0, otherwise KEEP; exact ties preserve absence/order. Structural requests, risk cap, pending states and terminal handling remain the reviewed engine's. No extra hold timer or inventory penalty is introduced to rescue quoting metrics.
13. Save actual fork snapshots and, for RICH, deterministically select at most one reached POST and one reached KEEP decision per original/side/head before inspecting their paired outcomes. Use the minimum SHA-256 rank under domain `adv2_rich_v1_reached_20261121` over the canonical day/instrument/original/side/head/generation/decision-ms key, with the canonical key as tie-breaker; eligible decisions only, no score/fill/Y selection. Regenerate their paired labels with frozen pi0 continuation from that exact snapshot (at most **1,536 reached pairs** on the replay subset). Preserve fingerprints and inclusion counts. This tests continuation/state-shift errors; do not reuse a neutral/shadow label for a reached state. Repeated/overlapping effects are never added to episode P&L.
14. Report each episode once: END cash/marked inventory value and differences versus PI0 and ENTRY-PA, all excluded/no-entry/no-fill statuses, bid/ask/either/both executable quote time, first-review pulls, generations/cancels per filled quantity, cap/reservation validity, terminal inventory, loss/severe-loss costs and fill markouts with their proper clock/endpoint coverage. Compare neutral-sample and reached-state prediction errors separately. Explain observed churn using recorded market/queue/input/value changes; do not infer causes from feature importance alone.

   **Scope limit:** this small replay is diagnostic. Same-history NEU2 and the full economic/quoting R4 qualification remain deferred; PI0 must not be renamed NEU2. Thus this round cannot claim the complete R4 gate or practical/continuous profitability. It can answer whether RICH beats fixed inputs on local timing, whether its own-state actions reverse rapidly, and whether value calibration survives state change. This bounded scope is intentional acceleration, not a change to the eventual qualification criteria.

### One combined delivery and resource bounds

Use a frozen nested small process pilot to measure cost, then proceed through the already declared full prediction sample and bounded replay without another approval pause. A **40 core-hour planning ceiling for the combined round**, existing four development slots, 1,536 MiB checkpoints / 2,048 MiB hard RSS, per-pair limits and durable stop history remain. Stream extraction/evaluation; reuse immutable per-world market features only with identical causal builders. Continue at audit every 35 while preserving all-callback binding and all-record invariants. Do not alter accepted training files or the frozen teacher to make integration convenient.

Necessary controls are focused: exact extraction/weight/feature identity, model reload predictions, no split leakage, wrong-side/head/fork refusal, controller sign/tie/fallback cases, own-state binding and reconciled actual episode ledgers. Reuse existing engine controls rather than rerunning every unrelated historical suite. Fold items 1–3 above into this delivery or document their verified workaround. If cost exceeds the ceiling, retain the declared partial statuses and revise the compute plan explicitly; do not select a smaller successful population after inspecting outcomes.

**Success of the round means a reliable model diagnosis, including a negative result.** Do not tune until seeing which layer failed: lack of incremental prediction, poorly calibrated magnitude, unsupported states, or repeated-action continuation mismatch. A later smoother/calibration/decomposition candidate requires its own declared change and multiplicity entry; no need to pre-emptively build all of them now.

## Checks run and limitations

All research execution used **PF_SLOT=codex**, `/home/yuqing/ctaNew/predictfun/research/run_light.sh`, absolute script paths and imports pinned to detached `/tmp/pf-review-f68fd805`. No fit, new feature/label build, raw-tape replay or untouched day was run by the reviewer.

| Independent bounded check | Evidence | Wall / peak |
|---|---|---|
| Pinned producer synthetic selftest, including 50 corruptions and zero-callback boundary | [selftest.log](selftest.log) | 122.30 s / 321 MiB |
| Freeze/models/inputs/checkpoint hashes and all 113 mutant identities; summary falsifiers | [metadata.json](metadata.json), [metadata.log](metadata.log) | 3.17 s / 226 MiB |
| All production partitions, numerical/callback audit, status/weight/support counts and fresh corruption refusals | [audit.json](audit.json), [audit.log](audit.log) | 48.47 s / 315 MiB |
| All saved pilot parity payloads and changed-Y/neutral-audit controls | [parity.json](parity.json), [parity.log](parity.log) | 2.70 s / 231 MiB |
| Streamed-checker output equivalence and all 96 job resource logs | [receipts.json](receipts.json), [receipts.log](receipts.log) | 0.89 s / 194 MiB |

Successful bounded checks: **177.53 seconds**, maximum 321 MiB. Preserved reviewer-only failed attempts: `/usr/bin/python3` lacked NumPy, so the same test was rerun in the repository's existing `/home/yuqing/pricer-sol/venv/bin/python3`; the receipt-comparison helper initially expected eight CSVs instead of the documented seven. Neither changed producing code/data. Failed logs and script versions remain alongside the passing evidence. [verify_core.py](verify_core.py) has the exact hash used by metadata/audit/parity; [verify.py](verify.py) adds the final receipt comparison; [verify_receipts_v1.py](verify_receipts_v1.py) retains its initial assertion.

The full mutation suites and real causal/reversal simulations were not rerun. Their hashes, recorded counters, controls and identities were checked, supplemented by the pinned selftest and independent artifact arithmetic. Review changes are documentation/evidence only; no implementation fixes, model/strategy edits, push or merge.
