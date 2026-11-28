# Model upgrade execution state — 2026-11-12T11:34:15.519990+00:00

Goal remains active. This step made implementation and verification progress; the numerical training stage is now in a verified dependency wait. No fitted improvement or portfolio result is claimed.

The frozen candidate is HISTORY10: observed temporal pressure/depth/context plus actual order state, predicting full-engine KEEP-minus-PULL value over10seconds. STATIC2, STATIC10 and a coherent joint-outcome version are fixed diagnostic comparisons. Legacy learned score columns are excluded, and start-price context is reconstructed from received patches. Six chronological outer dates are consumed exploratory history. The complete design is in `20261112_model_upgrade_plan.md`.

The separately frozen sequential evaluation includes default quoting, the legacy guard, an earlier-only refitted guard, STATIC10 and HISTORY10. The default entry,5-share size, reservation accounting and inventory cap are shared. Primary and conservative execution assumptions are declared before results. Full engine reconciliation, inference reconstruction, common-window marked value, inventory-square exposure and actual quote-presence unions are required. Timer-based1second re-entry remains a stated limitation; recovery-conditioned entry and the separate250ms extension are not implemented by this experiment.

Verification completed: the model feature/chronology synthetic controls and replay cancellation/cadence/cooldown/reservation controls passed, including deliberately bad cases. Exact receipts are `reviews/model_upgrade_20261112_v2/controls.json` and `reviews/model_upgrade_eval_20261112/controls.json`.

The inherited clock audit inspected 15,552,000 feature keys on9declared consumed days, as of 2026-11-12T11:33:25.235376+00:00. Across 10,368 finite-score mutations, no pre-stop decision-time set changed after union with the existing50ms neutral wakeups. All 12,033,416 keys with a known mid before240seconds had finite original side scores. Missing scores occurred only outside that observable live mask. These checks do not prove duplicate event-loop processing invariance or validate every underlying feature; full replay parity remains required. Exact audit declarations and receipts are beside the model controls.

Workers verified at 2026-11-12T11:34:15.519990+00:00: upstream continuation driver339629 with its current day worker and audit finisher459264; our model waiter654408 and replay waiter674377. Our waiters have no numerical children. The upstream first day20261023 is complete, and20261024 is building. Data/fitting jobs wait for the upstream driver and finisher to exit; sequential runs wait for the model report. PIDs are snapshots and must be rechecked against command lines, not treated as permanent identities.

The brief synthetic checks and the read-only clock/availability audits used the idle second research slot while the upstream continuation worker held core8. The global two-worker cap was retained. All queued data, fitting and portfolio jobs remain on the Codex/core8 slot. The first replay waiter failed to import NumPy under the system interpreter before any stage; its log is preserved and `execution_v2.json` records restart under the project virtualenv. No model source or result was changed by that operational correction.

Branch `research/predict-model-upgrade-20261112`, worktree `/tmp/cta-model-upgrade-20261112`. Model source freeze35200a75; sequential source freeze262445db. Model outputs `data/predict_fun/research/model_upgrade_20261112_v2/`; sequential outputs `data/predict_fun/research/model_upgrade_eval_20261112/`. Next: inspect completed data receipts, chronological fits and matched-count local nulls, then primary/stress portfolio comparisons. Preserve failures and supersede corrections; do not declare success from local calibration or reduced cancellation alone. No live trading changes.

## 2026-11-12T11:46:21.267082+00:00 — first real-data integration and value components

The completed20261023 continuation day now has rebuilt features: 1924 eligible actions, original unchanged market fields reconciled with maximum absolute error 0.0, and 134282 chronological guard-target rows joined. This is data integrity evidence, not a model score. Its full population and context availability are in `reviews/model_upgrade_20261112_v2/data_20261023.json`. The remaining stages wait on the still-running upstream build; model waiter is now723465. Replay waiter674377 remains active.

Added read-only local value and sequential per-fill decompositions without changing the frozen candidate, thresholds or policy. Local reports separate branch aggregate loss avoided, gain forfeited and inventory penalty reduction; those are not fill-level components. Sequential reports value each actual simulated tranche at its own time/level and split favorable/adverse/zero/missing marks. Source cab5f017; `reviews/model_upgrade_eval_20261112/detail_declaration.json`. Synthetic controls passed. The real-artifact integration check, as of 2026-11-12T11:45:05.404914+00:00, exactly reconstructed both strict1s/5s generation valuations for 11369 fills and 19116 generations across 288 windows on20261026. This verifies the instrument, not the new model.

Detail finisher761197 waits for replay driver674377 before taking core8; it will skip the completed control receipt. Two earlier manual control launches were refused by the shared slot guard before Python started; their logs remain intact. The successful controls and existing-day integration check used the briefly idle core9 slot, without exceeding the global two-worker limit. The first-day feature integration used that slot for10.87seconds; remaining data/fits/replays stay queued for core8.

Completion review must retain the repository's B1a warning: simulated fills on young orders are poorly validated against real orders, including under M/L assumptions. An independent AGE study is examining that issue in the main checkout; do not interfere with it or interpret post-hoc removal of young fills as a sequential executable policy. Our fill-detail artifacts retain actual age and cancel-pending flags for attribution. Any apparent model improvement must be examined for dependence on those fills, alongside paired portfolio values and inventory exposure. No live-calibration claim is available.

## 2026-11-12T11:55:54.166608+00:00 — young-fill execution stress queued

Upstream20261023and20261024build receipts now pass;20261025is actively building. The original model comparison still has no fitted outer-test results. Added the separately frozen age500stress above before those results, motivated by the existing B1a/AGE evidence rather than new-model outcomes. No existing candidate, threshold, dataset or scenario changed.

Age driver860873 waits for detail finisher761197; replay driver674377 and model driver723465 remain in their dependency waits. Age controls passed in the idle second slot; future full runs remain serialized on core 8 with the shared cap. The new study will check a full sequential alternative, not a subtraction of young fills from an unchanged portfolio. Its source/declaration/controls/execution receipts are under `reviews/model_upgrade_age_20261112/`.

## 2026-11-12T12:07:20.262326+00:00 — forecast-horizon audit added before model results

Source review found that the initial report measures STATIC2 forecast calibration against the common10sdecision outcome, although that model predicts2s. The common10saction-policy comparisons remain intentional. A separately committed audit (81109594) will reconstruct all saved direct/joint/guard forecasts and emit `forecast_calibration_v2.json`: own-target errors and calibration for each family, with an earlier-core-only mean baseline. Its receipt will supersede the original forecast-calibration interpretation while explicitly retaining the cross-horizon diagnostics. No models, thresholds, policy comparisons or target magnitudes change.

Synthetic correct-horizon/cross-horizon/nonfinite/corrupt-output controls passed. Audit waiter998391 is bound to age driver860873 and runs after the complete execution study. Declaration, controls and execution receipt are under `reviews/model_upgrade_20261112_v2/forecast_audit_*`. Full numerical reconstruction still waits for fitted artifacts.

## 2026-11-12T12:53:19.427083+00:00 — first fold passed; replay precision correction queued

The frozen first fold completed on20261026: core20261023/24, calibration20261025, test2013 eligible neutral actions. Four data receipts pass. The original eight-window primary replay passed portfolio accounting, default-neutral parity and saved-score reconstruction at2026-11-12T12:42:30.103915+00:00. These are integration checks; no full six-day model-performance claim is available. First-fold preflight receipt records resumed model driver1320642.

Source review found a narrower inference-contract issue the original score reconstruction does not test: neutral training carries float32 subtraction for `d_bnimb250`/`d_pfimb250`, while promoting the corrected replay market matrix to float64 changes that subtraction. Precision audit source0ed68ad8 will compare both definitions against all eligible actions on the first four consumed days, plus the effect on fixed first-fold predictions. The audit has not run: the shared slot guard refused core9 while another study occupied it. No effect size or changed-decision count is claimed yet.

Correction sourcee9f88d7b retains the frozen models, corrected context, feature-history calculations, thresholds and three execution conditions. A versioned runtime adapter restores the two inherited delta arrays to the training definition and independently reconstructs every saved inference sample from the frozen grid. Synthetic actual-Window parity, planted float64 refusal, policy/cooldown controls and full accounting remain gates. Syntax compilation passed; numerical controls and real audit remain pending. Original source and smoke artifacts are preserved. Corrected outputs will be `model_upgrade_eval_20261112_v2` and `model_upgrade_age_20261112_v2`, with explicit superseding declarations.

Verified-idle own waiters674377/761197/860873/998391 were stopped with no children before replacing their downstream chain. Corrected driver1417682 waits for the continuing model driver1320642 and upstream continuation/audit workers, then runs the precision audit, declarations, controls, full primary/stress replay, component reports, sequential age stress and forecast reconstruction on core 8. It keeps the global two-worker cap and preserves failures without automatic retry. Execution and transition receipts bind the source, stopped identities and new PID. PIDs are snapshots, not permanent identities.

The current work is still exploratory consumed-history research. No better model is established, no live orders changed, and real-fill calibration remains unresolved.

## 2026-11-12T13:31:06.753174+00:00 — correction preflight waits for the occupied second slot

Six continuation days are reconciled; the Nov1 worker1642426 is active. No additional feature build, fit, precision audit or replay result has completed since the first fold. Both research slots remain occupied.

To run the already-frozen correction checks sooner, bounded operational driver1759312 (source10a00441) waits for the observed FV3 driver1320727 to exit, then separately checks core9 availability and the global cap. It will run the precision declaration/audit, corrected evaluation declarations/controls, and first-fold prepare/eight-window smoke. It preserves every failure and performs no automatic retry. This changes scheduling only: models, features, policies, scenarios and evaluation populations remain frozen.

The verified-idle full-evaluation waiter1417682 was stopped with no children. After the bounded preflight passes, the new driver automatically restarts the unchanged `model_eval_driver_v2.py`, which skips completed receipts and keeps full evaluation on core 8. Model driver1320642 continues unchanged. `precision_preflight_execution.json` binds source hashes, the other-study identity, stopped waiter and scope; `precision_preflight_launch.json` binds the new PID. The success receipt will record the resumed evaluation PID. All PIDs require current command-line verification.

No numerical correction check has run yet, and no model improvement is established.

## 2026-11-12T13:57:47.024249+00:00 — precision corrected and four chronological folds fitted

The precision audit passed at2026-11-12T13:48:29.562956+00:00 on7436 eligible neutral actions across20261023/24/25/26. Original double subtraction differed in2913rows, by at most5.960464477539063e-08 in either inherited imbalance delta. The corrected arithmetic matched every frozen training value exactly. On the2013actions in the first test fold, none of STATIC2/STATIC10/HISTORY10's saved-model predictions or signs changed. This is arithmetic evidence; it supplies no evidence that the correction improves trading performance. Exact declaration and result are `reviews/model_upgrade_20261112_v2/precision_audit_*.json` and `precision_audit.json`.

Corrected actual-Window and policy controls passed. The eight-window Oct26 replay passed at2026-11-12T13:50:55.240536+00:00, and all12saved inference samples were independently reconstructed from the frozen grid. All five policies' window/fill/request hashes and the independent portfolio hash equal the original smoke receipt. Generation feature metadata may differ by the corrected arithmetic. This narrow integration check is not a six-day performance comparison. Current sequential/age declarations explicitly supersede their original versions under `reviews/model_upgrade_eval_20261112_v2/` and `reviews/model_upgrade_age_20261112_v2/`.

The preflight automatically resumed corrected full-evaluation driver1944044. With core9 then idle, bounded driver1963584 built the already-reconciled20261027/31/20261101 features and fitted those three predeclared chronological folds; all six stages passed by2026-11-12T13:55:29.730888+00:00. Their eligible populations are1758/1224/1301actions, and unchanged market-feature reconciliation had maximum absolute error0 on all three days. Chronology and saved-model reconstruction pass. Four of six outer folds are now fitted, without changing any model definition or choosing a candidate from partial results.

The bounded fold batch stopped only the verified-idle model waiter1320642 and resumed the unchanged model driver as1983093 after success. Full-evaluation driver1944044 remains live. `ready_folds_execution.json`, `ready_folds_launch.json` and `ready_folds.json` bind scheduling, source and receipt identities. The last two input days and complete reports/portfolios are still pending. No live changes, untouched validation, live-fill calibration or improvement claim.

## 2026-11-12T14:07:05.316510+00:00 — fifth fold prepared; final input day active

Eight continuation-input days now pass full-day reconciliation. Nov2 completed at2026-11-12T14:00:46.929857+00:00 with288windows and1829paired forks:1811eligible actions and18missing-mark statuses. Its feature build reconciles unchanged market inputs exactly; the fifth chronological fit passes saved-model reconstruction and chronology (coreOct23/24/25/26/27/31; calibrationNov1; testNov2). Replay preparation also passed. All five fitted folds are prepared; no partial-fold performance selection was made.

The corrected replay's component and age-stress synthetic controls passed, and preparations forOct27/31/Nov1 were run early without changing the frozen pipeline. `early_preparation.json` and its execution receipt bind those stages. The fifth-fold batch stopped the verified-idle model waiter1983093 to prevent duplicate stage writes, then resumed the unchanged driver as2071865. Corrected full-evaluation waiter1944044 and final-day continuation worker2031750 are live. Core9 was used only for these short, already-declared stages under the shared cap.

Use `fifth_fold_v2.json` and `fifth_fold_execution_v2.json` for current fifth-fold orchestration provenance. They supersede an overbroad verified-hash field in the launcher: it copied all seven upstream hashes after directly rechecking four paths. The superseding receipt independently rehashes all seven exact artifacts, including window/fill/request files. All input bytes, stage outputs, model identities and results are unchanged. Original receipts remain intact.

The final day, sixth fold, complete local report and full sequential portfolios remain pending. No live changes or claim of improved performance.

## 2026-11-12T14:18:00Z — tie-settlement audit frozen and controlled

The completed FV3 review surfaced a registry valuation issue: equal start/end prices are recorded as Up. Predict's official resolution documentation confirms 50/50 payout for equal-price Chainlink rounds (https://dev.predict.fun/concepts/resolution, accessed2026-11-12). The independently read registry audit covers2592windows on the nine declared consumed dates and finds14ties, of which7are in the1728test windows. Exact day/market populations and as-of2026-11-12T14:16:48.848833+00:00 are in `reviews/model_upgrade_20261112_v2/settlement_audit_controls.json`.

Sourceac339f20 adds an immutable audit after the corrected replay chain. It independently reconstructs raw and corrected per-fill payouts and cash-plus-terminal-inventory; all frozen artifacts and engine parity remain intact. Complete `result_settlement_v3.json` receipts in the corrected primary and age roots will supersede settlement/carry fields and paired settlement intervals in-band. Primary book-marked comparisons, risk, decisions and continuation targets are unchanged: the target builder uses own-fill1s/5s marks and inventory penalty, not terminal payout. No correction magnitude is available before the full portfolios.

Synthetic controls passed for a tie with both trade sides, an unchanged non-tie, corrupted payout, missing/duplicate registry identity, reversed side and unknown outcome. Finisher2189202 waits for corrected evaluation driver1944044 to exit and the Codex/core8slot to be free; completed primary/age receipts are mandatory. Source, declaration, control and execution hashes are preserved in `settlement_audit_*` receipts. No numerical audit has run yet.

Five folds remain fitted/prepared. Model waiter2071865 and final continuation worker2031750 are active. No live trading changes or improvement claim.

## 2026-11-12T14:22:17.567989+00:00 — age-stress preparations advanced

The five fitted dates throughNov2 now also have the unchanged frozen preparation receipts in the corrected age-stress root. `age_preparation_early_execution.json` and `age_preparation_early.json` bind the five original preparation hashes and copied receipts; all stages passed on the idle second slot under the shared cap. The corrected evaluation driver will skip these completed stages after the full primary/stress runs. No age simulation or performance result has completed. Final input worker2031750 remains CPU-active; model2071865, evaluation1944044 and settlement2189202 waiters remain live.

## 2026-11-12T14:31:37.477832+00:00 — six folds and independent local audits complete

The final continuation day passed at2026-11-12T14:24:45.198032+00:00:288windows,1823paired actions,1810eligible and13missing marks. All seven exact artifacts were rehashed in `final_input_verified.json`. Upstream build/fit/audit workers exited normally; the original model driver automatically completed the final feature build, fit and full local report, then exited. All six corrected replay preparations are complete.

Read `20261112_model_upgrade_local_results.md` for the complete local interpretation: temporal features improve ranking versus STATIC10 but fail to establish superiority over the chronological guard; positive predicted keeps have negative aggregate observed advantage. The independent forecast reconstruction and existing local component audit were advanced on idlecore9 after all folds existed, with execution/result receipts. No fit, policy or threshold changed. The full driver skips these completed stages.

Corrected evaluation driver1944044 is running the first full primary replay with worker2277844; settlement finisher2189202 waits behind the chain. The goal remains active: complete primary/stress/age portfolios, per-fill diagnosis and corrected settlement receipts are still required. No live changes.

## 2026-11-12T14:56:48.606529+00:00 — bounded inference profiling and exact feature-container audit

The first full primary replay remains CPU-active under the original implementation. A separately committed read-only profile (source1eb6c151) reconstructs12saved smoke inference cases and profiles1200calls. It identifies repeated pandas construction as a material inference cost. Those profiling calls pass the larger saved audit row, so their absolute timings are not treated as the actual live input-container cost or whole-engine runtime.

The container-only implementation (sourcec9910e58) extracts the frozen `History.batch` NumPy body unchanged and substitutes only its one-row input/output containers. The feature audit passed at2026-11-12T14:52:28.077036+00:00: all15340eligible actions on all nine consumed dates match every frozen feature exactly;12saved smoke cases, future mutation, stale/invalid queries, corrupted feature and ambiguous transform controls also pass. The benchmark matches the actual state-container fields, with four unused guard/context fields held at0. Across two alternating rounds of1200calls, original feature construction took2.582/2.575seconds and the container substitute0.306/0.306seconds. This is a12-case repeated-call benchmark, not an end-to-end replay speedup.

The isolated parity replay source9b866f7c (initial pipeline5b0d51c1) is frozen under `model_upgrade_inference_parity_20261112`. Its eight-window smoke is running on idlecore9 while the original full primary worker2277844 continues oncore8. The gate requires exact equality of all five policies' win/fill/request/generation artifacts plus saved inference and independent portfolio records, with original accounting and source-identity checks. A complete288-window primary parity test is declared next. No main driver, candidate, model, feature arithmetic, policy threshold, execution clock or event population has changed. Adoption requires full parity and separately bound execution provenance.

Read `replay_profile_*`, `fast_feature_*` and `fast_replay_smoke_execution.json` in the model review root, and the new parity-root declaration. This is implementation work to complete the same frozen strategy experiment sooner, not a new model search or an improvement claim.

## 2026-11-12T15:18:30.321617+00:00 — first full primary day passes; runtime prototype remains unadopted

The complete Oct26 primary replay passed at2026-11-12T15:08:26.592219+00:00, with288windows per each of five policies. The default path is exact to the neutral reference. All398saved inference samples were independently rebuilt from the frozen grid at15:08:28.428557. A subsequent read-only checkpoint rehashed all22named artifacts and verified generation identities and counts. Exact receipts are `model_upgrade_eval_20261112_v2/run_X_20261026*.json`. Evaluation driver1944044 has moved toOct27 with worker2646745; settlement finisher2189202 still waits.

The one-day operational checkpoint (as of2026-11-12T15:15:33.718441+00:00;288windows, no interval) shows HISTORY10 with32839generations,28372model-pulled generations,55826scores, and132msmedian positive accepted lifetime. This is still frequent withdrawal under repeated decisions. It is not the six-day strategy comparison or a policy-selection result. The whole0–240.1s union-presence fraction is0.047629forHISTORY10 versus0.180143fordefault, but these denominators include the fixed60sreduce-only cutoff. Before interpreting occupancy against the user's earlier10%figure, add a descriptive presence decomposition at the existing60sboundary, requiring exact partition/reconciliation to the already-declared whole-window union. Do not tune a policy or change the frozen metric based on the checkpoint.

The fast-container eight-window parity gate passed at2026-11-12T14:57:25.349426+00:00: all22artifacts, five policies, all values and bytes exactly match the original corrected smoke. Its measured wrapper wall time was132.11seconds versus138.61seconds for the earlier original smoke. These are separate small runs, not a paired full-engine speedup estimate. The main runtime and driver remain unchanged. Full-day optimized parity is NOT queued at this point; it is a prerequisite only if that implementation is proposed for adoption.

An attempted nonblocking sampler attach to the existing worker was denied by the OS; no elevation or workaround against that process was attempted. A separately launched child under its own sampler then completed the same eight-window path with exact artifact parity at2026-11-12T15:06:04.775072+00:00. Sourcefc03928b, `whole_profile_*` receipts, and `model_upgrade_whole_profile_20261112/stacks.raw` bind that read-only instrument. The sampler reports2518samples and1error;5samples have an empty stack and must remain an explicit unknown category. The wrapper's peak-RSS/CPU figures measure the sampler parent, not the Python child; do not quote them as child resource usage.

Runtime stacks show that the actually installed order-state class is already `pf_default_pull_fast.FastWindow`, with cached observed depth minima. The uncached base `Window.features` source alone would have been the wrong explanation for current runtime. Loading, bulk chronological-guard inference and regular policy/state/lookups remain material in the sampled child. No further runtime optimization is adopted, and no whole-day speedup is claimed. The active full evaluation continues on the original frozen implementation.

## 2026-11-12T15:28:26.375275+00:00 — fixed-phase quote-presence audit passes first full day

Source 74147da4 adds a read-only partition at the existing 60-second reduce-only boundary. It separates side-time, any-side and simultaneous two-side exposure; overlapping generations are unioned and the first/second phases reconcile exactly to the original whole-window measure. Synthetic cross-boundary overlap and empty-window controls pass, while naive lifetime sums, missing clocks, duplicate generations and out-of-window intervals are refused. Full populations and source/receipt identities are checked. This is a descriptive supplement after the first whole-window checkpoint, not a new selection metric.

Oct26 primary presence audit passed at 2026-11-12T15:27:05.785448+00:00 on 288 windows per policy, one complete day with no interval. HISTORY10 has any-side presence of 25.88% in the first minute versus 12.83% for the chronological guard and 88.69% for default. Its two-side first-minute presence is 2.67%. The full table and interpretation are appended to the local-results review; no portfolio improvement or live claim follows from this checkpoint.

Bounded controls and the completed-day audit ran on idle core9 under the shared cap. Pure waiter2820757 waits behind evaluation driver1944044 and settlement finisher2189202, requires both corrected settlement results, then serializes remaining presence partitions on core 8 and produces the all-six-day report. It preserves failures and does not retry automatically. `presence_early_execution.json`, `presence_execution.json` and the declaration bind source, scheduling and process identity. Current Oct27 worker2646745 remains on the original frozen runtime; no strategy job was interrupted.

## 2026-11-12T15:44:46.578282+00:00 — final evidence checker controls pass

Source e120056d implements the completion integrity check, without changing models, policies, outcomes or success criteria. It binds exact local fit/features/model artifacts, all 18 full replay receipts and their actual parquet row counts, independent inference identities, three required smoke paths, corrected settlement reports, per-fill diagnosis and complete phase-presence receipts. It verifies exact population coverage and that settlement correction leaves nonsettlement evidence unchanged; the original objective predicates are recomputed from paired value and inventory deltas. A PASS means evidence integrity, not model superiority. Existing numerical audits remain the authority for their estimands.

Controls passed at 2026-11-12T15:43:54.050827+00:00 on idle core9 under the shared cap. Complete synthetic coverage and a positive strategy predicate are accepted; missing/duplicate coverage, altered artifact bytes and a planted contradictory improvement are refused. The same instrument directly verified the completed Oct26 primary files and all 398 saved inference identities. Full-chain checking remains unexecuted because remaining replay and audit results do not exist. Declaration/control/execution receipts are `evidence_check_*` in the model review root. No additional finisher was launched.

After evaluation1944044, settlement2189202 and presence2820757 finish, run `pf_model_evidence_check run` through the main checkout run_light wrapper in the free Codex slot, with PYTHONPATH set to this worktree and PYTHONSAFEPATH=1. It writes immutable `final_evidence_check.json`. Then inspect the paired strategy results, calibration, age dependence, phase presence and unresolved live limitations against the original objective before deciding goal completion. The checker does not itself complete or promote the goal.

## 2026-11-12T15:51:55.825489+00:00 — second primary day verified; third running

Oct27 full primary replay completed normally at 2026-11-12T15:48:59.742685+00:00, with five policies × 288 windows, exact default-neutral parity, portfolio reconciliation and 453 independently reconstructed saved inference rows. The original wrapper reports rc0 and 2433.18 seconds; its worker2646745 exited. Driver1944044 moved normally to Oct31 worker3013930. Settlement2189202 and presence2820757 remain dependency waiters.

The newly committed evidence checker directly rehashed 31 named artifacts and verified parquet populations in `checkpoint_X_20261027.json`. A bounded idle-core9 presence audit then passed at 2026-11-12T15:50:34.147706+00:00; the full phase table is appended to the local-results review. These descriptive checkpoints carry no interval and do not select a policy. The final presence finisher will skip this completed receipt. Two of 18 required full scenario-day replays are complete; 16 remain, including the currently running third primary day.

## 2026-11-12T15:58:35.982915+00:00 — local model diagnostic figure rendered and inspected

Source3b031673 renders the already-frozen HISTORY10 calibration bins and three matched-count local comparisons, using the original forecast/local receipts. It introduces no model, threshold, test population, metric or interval. The figure includes all 9917 eligible test actions, 60 paired missing-mark exclusions, source as-of dates, bin counts, correct ten-second horizon and the distinction from portfolio P&L. The positive-keep overprediction title is checked against the aggregate forecast/observed values in code.

PNG and SVG are stored under `reviews/model_upgrade_20261112_v2/local_diagnostic_v1.*`, linked from the local-results review. `local_diagnostic_figure.json` binds source commit, original report hashes and both artifact hashes. Rendering passed under the shared cap on idlecore9; the PNG was visually inspected for legibility and clipping, and all three artifact copies rehashed identically. This is presentation of completed local evidence, not a new performance result. Third primary worker3013930 continues unchanged.

## 2026-11-12T16:39:42.663145+00:00 — third primary day complete; age checker corrected at a scheduler boundary

Oct31 completed normally at 2026-11-12T16:32:06.746257+00:00, with288windows per each of five policies and631 saved inference reconstructions. The three completed primary dates contain1482 reconstructed samples in total. `checkpoint_X_20261031.json` rehashes31named artifacts and their actual parquet populations. Its complete phase-presence audit is appended to the local-results review; no policy is selected from these checkpoints.

The last age preparation and both predeclared eight-window smoke gates were advanced on idlecore9. The zero-floor gate passed exact reference parity. The age500 engine/replay accounting passed, but the old post-run validator raised TypeError on an empty object-typed age_ms Series for both guard modes. `age_smoke_early_failure.json` binds the failed log, completed raw run, exact fill hashes/counts/dtypes and original scheduling receipt. No simulation artifact was changed or rerun.

Correction sourcedbffb3ac is separately frozen in `age_check_v3_declaration.json`. Empty tables retain zero fills and null minimum age, provided the age column exists. Numeric boundaries pass; missing-column, below-floor, NaN, infinity and nonnumeric nonempty cases are refused. The corrected post-check passed at 2026-11-12T16:29:34.149152+00:00; its unchanged eight-window engine had 1101 suppressed opportunities, no below-floor credited fills, and eight reconstructed inference samples. Suppressed opportunities are engine events, not a measured number of real fills. `age_smoke_early_v3.json` supersedes the checker failure and verifies both smoke chains at exact named artifacts.

The original scheduler1944044 was SIGSTOP-held at the next boundary; its active Oct31 worker3013930 finished normally, wrapper rc0. Only after all descendants were terminal, the scheduler and idle settlement2189202/presence2820757 waiters were retired. New scheduler3436837 uses the original task generator with exactly eight age run commands routed through pf_model_age_v3; every other task is identical. New settlement3436838 and presence3436839 waiters recognize its exact script identity. Source/controls, held process identity, completed stage, retired processes and launched paths/PIDs are bound by `age_check_v3_scheduler_hold.json` and `age_check_v3_scheduler_execution.json`. The original scientific source/declarations remain unchanged.

Nov1 full primary worker3436856 is now active oncore8. Original corrected primary/stress runs continue unchanged. The remaining full age simulations will use the unchanged engine and the corrected post-check. Before those full runs, complete a bounded isolated zero-floor parity exercise through the new entrypoint itself; recovery has exercised its post-check, while the preceding zero-floor engine run used the original entrypoint. This is implementation validation, not another model trial.

Final review must read the new age_check_v3 declaration/controls, age_smoke_early_v3 and scheduler execution receipts alongside final_evidence_check.json. The earlier completion checker binds the age-check receipts and data but does not independently enumerate this new correction source declaration. Full primary/stress/age comparisons, corrected settlement, complete presence aggregation and final goal review remain pending. No live change or improvement claim.

## 2026-11-12T16:48:05.199163+00:00 — corrected age entrypoint passes isolated exact parity

Source1b32a54c and `age_entry_parity_declaration.json` freeze the bounded integration check required after the empty-fill validator correction. The corrected pf_model_age_v3 entrypoint independently replayed the first eight Oct26 windows at floor0 in `model_upgrade_age_entry_parity_20261112`; only original declaration/preparation bytes were copied, never reference output data. Positive DataFrame equality and an altered-cell refusal passed before execution.

Parity passed at 2026-11-12T16:45:28.875122+00:00: all five policies × eight windows, all 22 win/fill/request/generation/inference/independent artifacts exactly equal in values and bytes, and identical generated engine source. Existing default/no-op reference reconciliation and precision reconstruction also passed. A subsequent direct check rehashed every named data artifact against both namespaces and all mirrored receipt copies. Canonical receipt: `reviews/model_upgrade_age_entry_parity_20261112/parity.json`; scheduling receipt: `age_entry_parity_execution.json`.

This closes the previously pending corrected-entrypoint integration check. It does not add a model, policy, performance estimate or fill-age calibration. The preserved age500 smoke remains the engine-effect/empty-table example; it was recovered without rerunning that simulation. The original full primary worker3436856 continues under scheduler3436837. Complete primary/stress/age reports, settlement correction, presence aggregation and final goal evidence review remain pending.

Final review must include this parity receipt and its declaration alongside the age_check_v3 correction receipts and final_evidence_check.json; the latter predates these implementation supplements.

## 2026-11-12T17:20:38.366400+00:00 — fourth primary date verified; fifth running

Nov1 completed normally at 2026-11-12T17:15:55.442727+00:00; all five288-window policy populations reconcile, default-neutral parity holds, and 573 saved inference samples reconstruct. `checkpoint_X_20261101.json` directly verifies 31 named artifacts and actual parquet counts. The phase-presence audit passed at 2026-11-12T17:18:21.247548+00:00; all windows are retained and the phase unions reconcile exactly. Its descriptive table is appended to the local-results review, with no interval or policy selection.

There are now 4 completed primary dates and 2055 reconstructed inference samples across them. Worker3436856 exited normally with wrapper rc0. Scheduler 3436837 moved to Nov2 worker3807559 oncore8; settlement3436838 and presence3436839 remain dependency waiters. The corrected age-entrypoint parity gate is already complete. Full primary/stress/age reports, settlement correction, complete presence aggregation and final evidence/goal review remain pending. This turn was a verified wait followed by new completed-day evidence; no model or policy changed.

## 2026-11-12T17:57:49.230244+00:00 — fifth primary day complete; supplemental check awaits a free slot

Nov2 completed normally at 2026-11-12T17:55:48.210823+00:00; all five policy populations contain 288 windows, default-neutral parity holds, and 384 saved inference samples reconstruct at 2026-11-12T17:55:50.138509+00:00. The wrapper records rc=0 and 2,392.59 seconds; worker 3807559 exited normally. The original run and precision receipts are identical across data, main-review and worktree-review copies. Five primary days are complete, with 2,439 reconstructed inference samples in total. No complete primary/stress/age strategy conclusion is available.

Scheduler 3436837 moved normally to Nov3 worker 4175449 on core 8; settlement 3436838 and presence 3436839 remain dependency waiters. A bounded fifth-day evidence-check invocation was refused before execution because the second slot is occupied by live worker 3992421. No checkpoint file was created and no numerical job was interrupted. The additional direct artifact/count checkpoint and fifth-day phase-presence audit remain pending until a slot is free; the final full-chain evidence/presence audits also require this day. The refusal is a resource-cap wait, not a failed replay or permission request. No model, policy or source was changed.

## 2026-11-12T18:37:28.669253+00:00 — all six primary replay days complete; execution stress started

Nov3 completed normally at 2026-11-12T18:35:53.689595+00:00, with all five policies covering 288 windows, exact default-neutral parity, and 379 saved inference samples reconstructed at 2026-11-12T18:35:55.799913+00:00. The wrapper records rc=0, 2,405.53 seconds and 1,940 MiB peak RSS; worker 4175449 exited normally. All six primary run and precision receipts pass their own checks, have the declared policy/window populations, and are byte-identical across data and both review copies. They contain 2,818 reconstructed samples in total. This metadata inspection does not replace the pending direct artifact/count checks for Nov2 and Nov3.

Scheduler 3436837 has started the first conservative execution replay, Oct26 stress worker 340307 on core 8. Settlement 3436838 and presence 3436839 remain dependency waiters. The second slot is now occupied by live worker 340734 on core 9, so the two remaining primary artifact checkpoints and phase-presence audits are still pending. The final full-chain evidence and presence instruments also require those dates. Six of 18 full scenario-day runs are complete; the six stress and six age500 runs, final reports and goal evidence review remain required. No complete strategy comparison or improvement claim is available.

The original plan and model declaration were reread against the implemented scope. The current paired targets and sequential paths use consistent one-second timer re-entry. The 250 ms avoidable-value target and WAIT/recovery alternatives remain explicitly subsequent extensions; no recovery-model or live-validation claim is made. No source, model, threshold, policy, inventory constraint or success criterion changed during this turn.

## 2026-11-12T19:08:05.116416+00:00 — remaining primary artifact and presence audits complete

The occupied second research slot became free, confirmed at two successive process observations. The pending Nov2 and Nov3 direct evidence checks then passed under the shared wrapper on core 9, rehashing 31 named artifacts per day and verifying actual parquet populations. The two unchanged phase-presence audits completed at 2026-11-12T19:05:45.417653+00:00 and 2026-11-12T19:06:10.124475+00:00; all 288 windows per policy remain included and the phase unions reconcile. All four new receipts are identical across the data and both review copies and bind the exact declaration/run/precision artifacts. The local-results review contains the two new descriptive tables.

All six primary days now have the direct completion and phase-presence supplements. This closes the prior Nov2/Nov3 resource waits. The final presence finisher will skip these existing receipts and still produce the complete three-scenario aggregate. No complete strategy or live-performance claim follows from these checks. The first stress worker 340307 continues normally under scheduler 3436837; stress, age, value-detail, corrected settlement, aggregate presence and final evidence/goal review remain pending. No source, model, policy or threshold changed.

## 2026-11-12T19:19:12.519245+00:00 — first stress day and supplemental checks complete

Oct26 conservative execution completed normally at 2026-11-12T19:16:10.866135+00:00; all five 288-window policy populations pass portfolio reconciliation and all 353 sampled inference rows reconstruct at 2026-11-12T19:16:12.629413+00:00. The wrapper records rc=0, 2,416.70 seconds and 1,849 MiB peak RSS. Worker 340307 exited normally and scheduler 3436837 advanced to Oct27 stress worker 708796. The raw receipt's `default_neutral_exact` field is a scope flag set by `pf_model_eval_run.py` to `scenario == 'X'`. Its false value for stress is expected; no neutral-reference parity comparison is claimed for the changed execution scenario.

The existing evidence checker then verified 31 named artifacts and actual parquet populations in `checkpoint_stress_20261026.json`; the unchanged phase-presence audit passed at 2026-11-12T19:17:39.160165+00:00. Both bounded checks ran on free core 9 under the shared wrapper and returned rc=0. All new run, precision, checkpoint and presence receipts are identical across data and both review copies and bind the exact named artifacts. The local-results review includes the descriptive presence table, with no interval or policy selection.

Seven of 18 full scenario-day runs are complete: six primary and one stress, with 3,171 saved inference reconstructions across those paths. Five stress and six age500 days, the complete reports, per-fill diagnosis, corrected settlement, aggregate presence and final evidence/goal review remain pending. Settlement 3436838 and presence 3436839 remain dependency waiters. No source, model, policy, threshold or success criterion changed.

## 2026-11-12T19:57:28.408773+00:00 — second stress day and supplemental checks complete

Oct27 conservative execution completed normally at 2026-11-12T19:55:35.606817+00:00; all five 288-window policy populations reconcile and all 387 sampled inference rows reconstruct at 2026-11-12T19:55:37.500393+00:00. The wrapper records rc=0, 2,364.81 seconds and 1,924 MiB peak RSS. Worker 708796 exited normally and scheduler 3436837 advanced to Oct31 stress worker 1056363.

The existing evidence checker verified 31 named artifacts and actual parquet populations in `checkpoint_stress_20261027.json`. The unchanged phase-presence audit passed at 2026-11-12T19:56:29.381173+00:00. Both bounded checks ran on free core 9 under the shared wrapper and returned rc=0. The new run, precision, checkpoint and presence receipts are byte-identical across data and both review copies and bind the exact named artifacts. The local-results review contains the descriptive presence table, with no interval or policy selection.

Eight of 18 full scenario-day runs are complete: six primary and two stress, with 3,558 saved inference reconstructions across those paths. Four stress and six age500 days, complete reports, per-fill diagnosis, corrected settlement, aggregate presence and final evidence/goal review remain pending. Settlement 3436838 and presence 3436839 remain dependency waiters. No source, model, policy, threshold or success criterion changed.

## 2026-11-12T20:40:10.219899+00:00 — third stress day and supplemental checks complete

Oct31 conservative execution completed normally at 2026-11-12T20:37:24.094107+00:00; all five 288-window policy populations reconcile and all 558 sampled inference rows reconstruct at 2026-11-12T20:37:26.025211+00:00. The wrapper records rc=0, 2,508.44 seconds and 1,723 MiB peak RSS. Worker 1056363 exited normally and scheduler 3436837 advanced to Nov1 stress worker 1432957.

The existing evidence checker verified 31 named artifacts and actual parquet populations in `checkpoint_stress_20261031.json`. The unchanged phase-presence audit passed at 2026-11-12T20:38:55.952326+00:00. Both bounded checks ran on free core 9 under the shared wrapper and returned rc=0. The new run, precision, checkpoint and presence receipts are byte-identical across data and both review copies and bind the exact named artifacts. The local-results review contains the descriptive presence table, with no interval or policy selection.

Nine of 18 full scenario-day runs are complete: six primary and three stress, with 4,116 saved inference reconstructions across those paths. Three stress and six age500 days, complete reports, per-fill diagnosis, corrected settlement, aggregate presence and final evidence/goal review remain pending. Settlement 3436838 and presence 3436839 remain dependency waiters. No source, model, policy, threshold or success criterion changed.

## 2026-11-12T21:20:44.978857+00:00 — fourth stress day and supplemental checks complete

Nov1 conservative execution completed normally at 2026-11-12T21:18:23.549209+00:00; all five 288-window policy populations reconcile and all 527 sampled inference rows reconstruct at 2026-11-12T21:18:25.412859+00:00. The wrapper records rc=0, 2,459.33 seconds and 1,828 MiB peak RSS. Worker 1432957 exited normally and scheduler 3436837 advanced to Nov2 stress worker 1807996.

The existing evidence checker verified 31 named artifacts and actual parquet populations in `checkpoint_stress_20261101.json`. The unchanged phase-presence audit passed at 2026-11-12T21:19:41.619048+00:00. Both bounded checks ran on free core 9 under the shared wrapper and returned rc=0. The new run, precision, checkpoint and presence receipts are byte-identical across data and both review copies and bind the exact named artifacts. The local-results review contains the descriptive presence table, with no interval or policy selection.

Ten of 18 full scenario-day runs are complete: six primary and four stress, with 4,643 saved inference reconstructions across those paths. Two stress and six age500 days, complete reports, per-fill diagnosis, corrected settlement, aggregate presence and final evidence/goal review remain pending. Settlement 3436838 and presence 3436839 remain dependency waiters. No source, model, policy, threshold or success criterion changed.

## 2026-11-12T21:59:57.261453+00:00 — fifth stress day and supplemental checks complete

Nov2 conservative execution completed normally at 2026-11-12T21:57:00.316909+00:00; all five 288-window policy populations reconcile and all 332 sampled inference rows reconstruct at 2026-11-12T21:57:01.870861+00:00. The wrapper records rc=0, 2,316.34 seconds and 1,714 MiB peak RSS. Worker 1807996 exited normally and scheduler 3436837 advanced to Nov3 stress worker 2154336.

The existing evidence checker verified 31 named artifacts and actual parquet populations in `checkpoint_stress_20261102.json`. The unchanged phase-presence audit passed at 2026-11-12T21:58:54.067158+00:00. Both bounded checks ran on free core 9 under the shared wrapper and returned rc=0. The new run, precision, checkpoint and presence receipts are byte-identical across data and both review copies and bind the exact named artifacts. The local-results review contains the descriptive presence table, with no interval or policy selection.

Eleven of 18 full scenario-day runs are complete: six primary and five stress, with 4,975 saved inference reconstructions across those paths. One stress and six age500 days, complete reports, per-fill diagnosis, corrected settlement, aggregate presence and final evidence/goal review remain pending. The declared comparisons require all six dates; no partial five-day interval was added. Settlement 3436838 and presence 3436839 remain dependency waiters. No source, model, policy, threshold or success criterion changed.

## 2026-11-12T22:43:21.730005+00:00 — primary/stress comparison fails; declared settlement correction complete

Nov3 stress completed normally at 2026-11-12T22:35:55.238500+00:00; all five 288-window policy populations reconcile and 316 sampled inference rows reconstruct at 22:35:56.967089 UTC. The wrapper records rc=0, 2,335.06 seconds and 1,912 MiB peak RSS. Worker 2154336 exited normally. The existing checker verified 31 named artifacts and actual parquet populations; the unchanged phase-presence audit passed at 22:40:15.531042 UTC. All new receipts have identical data/main/worktree copies and exact run/precision links. Twelve full scenario-day runs contain 5,291 reconstructed inference samples.

The scheduler produced the complete primary/stress report at 22:36:17.206635 UTC and per-fill reconciliation at 22:37:52.961500 UTC, then started Oct26 age500 worker 2524771 on core 8. The frozen candidate fails the declared value-up/risk-not-up objective versus both chronological guard and STATIC10. Read `20261112_model_upgrade_primary_review.md` for paired values, day intervals, missing populations and interpretation; local utility is not substituted for portfolio P&L.

The already-declared primary settlement audit was advanced on idle core 9 after its complete report dependency existed. It passed at 22:40:51.901377 UTC (rc=0, 19.49 seconds, 270 MiB peak RSS), independently reconciling all 60 day/scenario/policy populations and correcting equal-price payout to 0.5. `settlement_primary_early_execution.json` binds the unchanged declaration, exact command/log, original/corrected report hashes and live driver identity. `result_settlement_v3.json` supersedes only settlement/carry fields; primary marked value, risk and failed predicates are unchanged. Scheduler and finishers were not edited or interrupted; the settlement finisher explicitly reuses an existing PASS correction.

Six full age500 runs, the age report and correction, final aggregate presence and final evidence/goal review remain pending. The new direct report is a negative intermediate research conclusion, not goal completion or promotion. No source, model, policy, threshold or criterion changed. The 250 ms avoidable-value and WAIT/recovery extension remains unimplemented.

## 2026-11-12T23:05:48.520542+00:00 — short-value and recovery primitives implemented while age replay runs

The preceding primary/stress result remains negative. Source commits 8ab3def1 and e1f3a436 add `pf_model_recovery_target.py` and `pf_model_recovery_policy.py`, with pure controls and synthetic full-engine clock tests. The target prices only the original generation's fills after an explicitly supplied effective-cancel clock, retains no-fill zeros and missing/censored statuses, and reconciles unavoidable own fills. The recovery state defaults to quoting and, after an explicit pull, requires a continuous safe interval; unsafe/unknown observations and gaps reset that interval. These primitives do not price complete recovery continuations or form a fitted model.

The combined 17 controls passed at 2026-11-12T23:02:11.407377+00:00 on idle core 9 under the shared wrapper (rc=0, 3.33 seconds, 137 MiB peak RSS). `recovery_primitive_controls_v1.json` binds the exact committed source, command and captured log. Its three copies, source hashes and log hash were directly verified. Positive engine examples establish the decision/cancel boundary convention; an inconsistent latency is refused. A trailing-whitespace issue in the first source commit was corrected before the combined run. No historical tape, new label build or model result was produced.

Read `20261112_model_recovery_progress.md` for the remaining integration requirements. Actual delayed/no-effect cancellation capture, the complete neutral ledger, paired full-engine recovery continuations, chronological fitting and a separately committed complete experiment declaration are still required. Current frozen model/replay source and success criteria remain unchanged. The Oct26 age500 worker 2524771 was directly confirmed CPU-active at roughly 27 minutes elapsed, within limits; its completion receipt is still pending.

## 2026-11-12T23:15:07.844108+00:00 — cancellation and tranche capture integrated with synthetic forks

Source df379b9c adds `pf_model_recovery_capture.py` and its integration controls. The isolated generated engine records cancellation request/effect clocks and whether the create was pending, and returns actual generation-linked tranches for each fork horizon. It preserves the inherited scheduling and policy. Five synthetic integration tests pass at the exact committed source: baseline/capture equality covers paired labels, the action population, independent windows and the full engine return (fills, requests, accounting and diagnostics). All captured economic branch fills reconcile to the raw engine at 2/10/30 seconds.

A synthetic delayed-cancel case proves that the recorder takes the engine's actual95ms effect, not the fixture row's22ms proxy. This is an intentionally varied control, not a newly discovered defect in the frozen primary/stress experiment. A pending-create case exercises the engine's separate effective-time formula. The first fixture moved only one tick and did not trigger pending cancellation, so its non-vacuity assertion failed; a three-tick price move during an80ms create fixes the fixture without weakening the assertion or changing capture behavior. The original failed and passing wrapper records are retained.

`recovery_capture_controls_v1.json` binds the exact source commit/hashes, command, observed5-test OK result and both wrapper records. The passing run used idle core9, rc=0,6.16seconds and145MiB peakRSS. The standalone unittest stdout was observed through the tool and was not separately saved; the receipt states that limitation. The instrument and tests are reproducible from the committed source. No historical tape or new prediction result was used.

Historical ledger/target integration, explicit no-effect handling, full KEEP/PULL/WAIT recovery continuations, chronological fitting and complete predeclared evaluation remain required. The original Oct26 age500 worker2524771 was directly confirmed active at36minutes elapsed with about1.7GiB RSS. The first experiment's sources and criteria remain unchanged.

## 2026-11-12T23:20:58.052765+00:00 — first full age500 day and supplemental audits complete

Oct26 age500 completed normally at 2026-11-12T23:18:50.735874+00:00; all five 288-window policy populations reconcile and 370 saved inference rows reconstruct at 2026-11-12T23:18:52.920636+00:00. The corrected age checker passed at 2026-11-12T23:18:50.956699+00:00, with no credited fills below 500 ms and 39,510 suppressed opportunity events. The wrapper records rc=0, 2,459.90 seconds and 1,835 MiB peak RSS. Worker 2524771 exited normally; scheduler 3436837 started Oct27 age500 worker 2902408 on core 8.

The existing evidence checker verifies 34 named artifacts and actual parquet counts in `checkpoint_age500_20261026.json`; the unchanged phase-presence audit passes at 2026-11-12T23:19:56.397901+00:00. Both bounded checks ran on idle core 9. All new data/main/worktree receipt copies are identical. The run/precision/presence hashes and corrected age-check declaration/run links were directly verified. Thirteen of 18 full scenario-day runs are complete, with 5,661 saved inference reconstructions in total. The local-results review includes the descriptive phase table; no partial age interval or policy selection is added.

Five full age500 dates, the complete age report and corrected settlement, final presence aggregation and first-experiment evidence review remain pending. The negative complete primary/stress gate is unchanged. Next-model primitives and actual cancellation capture are implemented and controlled, but historical target/recovery/model integration remains incomplete. The goal remains active; no live change or promotion.

## 2026-11-12T23:36:41.509362+00:00 — paired value decomposition and recovery continuation pass synthetic integration

Source6538eaf7 adds `pf_model_recovery_labels.py` and `pf_model_recovery_fork.py`. The label instrument independently reconstructs branch cash/net, strict fill value and inventory penalty, then defines the remainder as full KEEP-minus-PULL value minus the immediate original-order component. It retains no-effect zeros, unknown cancellation paths, missing marks and censored coverage as explicit statuses. Identical unavoidable fills must also carry identical strict marks.

Eight label controls pass, including equality to the original synthetic paired engine values. The strengthened common-prefix mark falsifier first exposed a shared dictionary in the synthetic fixture: changing one branch actually changed both. Deep-copying the fixture branches and asserting distinct identities corrected the test; the checker then refused the intended one-branch corruption. The failed and passing wrapper records remain available. This was not a historical label or model correction.

The new isolated PULL fork uses continuous recovery in place of the one-second timer. Six full-engine controls pass at the committed source, with saved stdout in `recovery_fork_final_controls.log`: exact original neutral/KEEP parity, current-touch re-entry after verified recovery, earlier re-entry for continuous clear evidence, no timer re-entry during persistent danger/unknown observations, reset after interruption, and refusal of early sends or corrupted recovery records. The source only received deterministic synthetic score callbacks. It is not a fitted or historically evaluated model.

`recovery_value_and_fork_controls_v1.json` binds the exact source, inherited control receipts, the eight-test label observation/wrapper record and the six-test fork log. The final fork control used idle core9, rc=0,9.21seconds and145MiB peakRSS. All previous component source hashes were rechecked against their receipts. `20261112_model_recovery_progress.md` now describes the integrated state and remaining full-pipeline work.

The active Oct27 age500 worker2902408 was directly confirmed CPU-active at about18minutes and within limits. Thirteen of18original full runs are complete; the five remaining age dates, complete reports/corrections/presence/evidence remain required. Historical target construction, causal recovery-score training, full new-policy replay and a complete scientific declaration also remain required. No goal completion, live change, model promotion or new performance claim.

## 2026-11-13T00:04:52.991083+00:00 — historical recovery pipeline and repeated-policy controls implemented

Source d64ec719e1c80fbf07128b47cc56f445f15ab285 adds the seven recovery-experiment modules and two control suites.
The neutral ledger/capture builder, corrected observable feature attachment,
fixed early-day recovery estimator, expanding value fits and full sequential
policy runner are implemented. Six pipeline controls and five policy/inference
controls pass with captured logs and exact committed source in
`recovery_pipeline_policy_controls_v1.json`. These are synthetic checks, not
historical model results. The policy checks include repeated withdrawal cycles,
an initial recovery observation at the pull clock, no timer re-entry on unknown
estimates, and rejection of corrupted inference or early placements.

The proposed primary requires both negative immediate own-fill value and
negative total keep value. Direct and component comparators remain explicit;
component and selective use the same two fitted heads. Recovery is planned to
be trained once on Oct23–24 and held fixed across all subsequent labels and
evaluation. The market-only recovery estimate is a neutral-order proxy, not a
validated fresh-order value model. The full evaluation report, declarations,
driver and evidence checks are still required before numerical history runs.
The implementation commit is not a scientific freeze.

Oct27 age500 also completed normally: run as-of 2026-11-12T23:59:22.006865+00:00, corrected age
check as-of 2026-11-12T23:59:22.072239+00:00, and 439 saved inference reconstructions
as-of 2026-11-12T23:59:24.045823+00:00. All five policies retain 288 windows. All credited
fills have actual age at least 500 ms; 36,706
suppressed opportunity events are not distinct real fills. The wrapper reports:
`2026-11-12T23:18:53Z tag=model_eval_v2_run_age500_20261027 rc=0 peak_rss_mib=1917 wall_s=2431.09 cpu_user_s=2427.11 memavail_start_gib=23 disk_free_gib=496->495 slot=codex`.

The independent checkpoint binds 34 named artifacts; phase-presence accounting
passes as of 2026-11-13T00:00:08.340510+00:00. Receipt mirrors and linked declaration/run/
precision hashes were checked directly. Fourteen of eighteen original full runs
are complete, with 6,100 reconstructed inference samples. Scheduler 3436837 has
started Oct31 age500 worker 3261711 on core 8. Four full age dates and complete
reports/corrections/presence/evidence remain. The primary/stress failure is unchanged.

## 2026-11-13T00:48:17.793445+00:00 — complete recovery experiment frozen; historical seed smoke passes

Source `78285fe02d00962abc5f7148e6001a8f39a0ddc6` completes local/portfolio reporting,
computed gates, target/artifact/inference/recovery-event verification, age500
sensitivity and serialized orchestration. Fifteen synthetic controls pass,
including the original/captured value identity, unknown statuses, chronological
fit refusals, full-engine zero-floor parity and actual 500 ms suppression, repeated
recovery cycles, independent inference and positive/negative report predicates.

The scientific declaration is as of 2026-11-13T00:32:13.765800+00:00 and binds 302 source
files, 836 inherited input identities, registry hashes and prior receipts.
The primary is fixed SELECTIVE10; recovery is trained once on Oct23–24 and reused
for all later labels/replays. The success gate includes value/risk and day
uncertainty versus chronological guard/direct, and entry presence/churn versus
the old temporal policy, under all three execution scenarios. No model or
threshold is selected from new outer outcomes.

The four-window historical seed smoke passed at 2026-11-13T00:36:27.200405+00:00: 279 generations
and 16 sampled actions, exact neutral paths and 16 reconciled KEEP/captured targets.
It used idle core 9 under the wrapper, rc=0, 46.30 seconds and 1,600 MiB peak RSS.
Feature attachment passed at 2026-11-13T00:38:09.391762+00:00, with all 16 rows retained; rc=0,
3.70 seconds and 1,234 MiB peak RSS. The pandas fragmentation warnings were nonfatal
performance messages; frozen source remains unchanged. The independent
`smoke_seed_evidence_20261023.json` reconstructs every target and binds 14 artifacts.
All data/main/worktree copies and linked hashes were directly checked.

The 60-stage pure driver started at 2026-11-13T00:35:11.662322+00:00 with PID 3592076, and was
directly confirmed alive/sleeping. It waits for the original workers and uses
only core 8 for full numerical jobs. Completed early-stage receipts are skipped;
partial/failed stages are not retried. The complete plan/progress documents
replace the earlier implementation-only status. No recovery/value fit or
new performance claim yet.

Oct31 age500 also completed normally at 2026-11-13T00:40:10.843422+00:00; the corrected age audit
passes at 2026-11-13T00:40:10.912519+00:00, with 20,654 suppressed opportunity events and no
credited fill below 500 ms. All five policies retain 288 windows, and 471 saved
inference rows reconstruct as of 2026-11-13T00:40:12.774511+00:00. The wrapper records:
`2026-11-12T23:59:24Z tag=model_eval_v2_run_age500_20261031 rc=0 peak_rss_mib=1726 wall_s=2448.63 cpu_user_s=2445.36 memavail_start_gib=23 disk_free_gib=495->495 slot=codex`.

The 34-artifact checkpoint and phase-presence audit pass (presence as-of
2026-11-13T00:42:53.486941+00:00); mirrors, declaration/run links and actual counts are
verified. The original experiment is 15/18 full runs complete, with 6,571 total
inference reconstructions. Worker 3631337 is running Nov1 age500; Nov2/3 and
complete age/report/settlement/presence/evidence work remain. The complete
primary/stress improvement gate remains failed.


## 2026-11-13T01:22:19.451204+00:00 — fourth full age500 date verified

Nov1 completed normally at 2026-11-13T01:19:32.963948+00:00; the corrected age audit
passes at 2026-11-13T01:19:33.032258+00:00 and all 454 saved inference rows reconstruct
at 2026-11-13T01:19:34.809999+00:00. Every policy retains 288 windows. There are
21,924 suppressed opportunity events and no credited fill below 500 ms.
The wrapper records rc=0, peak RSS 1,827 MiB, wall time 2,361.96 s and
CPU user time 2,360.07 s for the full run on core 8.

The 34-artifact checkpoint and phase-presence audit pass. Their data/main/worktree
mirrors, run/precision links and corrected-checker identity were checked directly.
Sixteen of eighteen original scenario-day runs are complete, with 7,025 saved
inference reconstructions. Nov2 worker 3990533 is confirmed running; Nov3,
complete age/report/settlement/presence and final evidence review remain.

Recovery driver 3592076 remains alive and waiting for the prior chain. The frozen
recovery source check passes at this checkpoint. No new recovery/value model has
been fitted, and no performance improvement has been established.


## 2026-11-13T01:58:57.077683+00:00 — pre-cancellation timing accounting complete

Source `31ae25eef29351b6c1b643c02373ccc6bd643257` freezes a descriptive instrument
for the already completed primary/stress fill-detail artifacts. Exact timing
boundaries, signed values and missing-value controls pass, with nine known-bad
refusals. The declaration precedes the historical breakdown; 60 full
day/scenario/policy populations reconcile to their existing fill-value totals.
The run returned rc=0 in 1.08 s with 143 MiB peak RSS on the spare slot.

The review binds the exact artifacts, checks mirrored receipts and computes
the negative-before-minimum-effect statements. Read
[the timing review](20261113_model_cancel_timing_review.md) for population,
value units and conditional-population limits. This changes no model, threshold,
next-experiment source, success criterion or protected-data boundary.


## 2026-11-13T02:01:27.523255+00:00 — fifth full age500 date verified

Nov2 completed normally at 2026-11-13T01:58:16.457165+00:00; the corrected age audit
passes at 2026-11-13T01:58:16.523156+00:00 and all 300 saved inference rows reconstruct
at 2026-11-13T01:58:18.018423+00:00. Every policy retains 288 windows. There are
27,122 suppressed opportunity events and no credited fill below 500 ms.
The wrapper records rc=0, peak RSS 1,692 MiB, wall time 2,323.08 s and
CPU user time 2,321.39 s for the full run on core 8.

The 34-artifact checkpoint and phase-presence audit pass; mirrored receipts,
run/precision links and corrected-checker identity were directly checked.
Seventeen of eighteen original runs are complete, with 7,325 reconstructed
inference samples. Nov3 worker 145469 is confirmed running. The final age
report, settlement correction, aggregate presence and final evidence review
remain before the queued recovery experiment starts full training.

Recovery driver 3592076 remains alive. Its frozen source check passes. The
new descriptive cancellation-timing review changes no fitted candidate or
criterion, and no new fitted-model improvement has been established.


## 2026-11-13T02:15:29.597599+00:00 — guard comparison source scope recorded

The deterministic Binance movement callback, its engine wiring, the default
targets and the learned-policy age gate were read at their actual definitions.
Six source files match the frozen recovery declaration. The source-review
receipt binds these identities and the new review document; it is not an
additional numerical or forward-validation claim. The benchmark's admission
and immediate reaction rules differ from the fitted value policies, so their
portfolio comparison cannot isolate prediction quality alone. The next
experiment and all frozen sources remain unchanged.


## 2026-11-13T02:37:37.167875+00:00 — original study closed; full recovery build started

All eighteen original scenario-day replays are complete. Nov3 age500 returned
rc=0 with 1,928 MiB peak RSS and 2,307.52 s wall time. Its corrected age audit
passes at 2026-11-13T02:36:43.859619+00:00, and all 300 saved inference samples
reconstruct. The complete age report, version-3 settlement correction and
three-scenario phase-presence aggregate passed in sequence.

The final integrity receipt passed at 2026-11-13T02:37:37.072135+00:00. It binds
1,011 artifacts, all eighteen full runs and 7,625 saved inference samples.
The primary/stress, age500 and combined improvement predicates are all false.
Read [the complete review](20261113_model_upgrade_complete_review.md) for
common/missing populations, day intervals, inventory risk, phase presence,
timing/guard scope and explicit corrected-age-entrypoint provenance.

After the predecessor workers exited, recovery driver 3592076 launched its
full Oct23 seed build, worker 508669, on the Codex/core 8 slot. Scientific source
`78285fe02d00962abc5f7148e6001a8f39a0ddc6` remains frozen. No new recovery/value
model has been fitted or shown a gain. The better-model goal remains active.


## 2026-11-13T03:03:26.769881+00:00 — first full recovery seed day complete

Oct23 seed build passed at 2026-11-13T03:03:23.002351+00:00, with 288 windows,
18,115 neutral generations and 1,939 sampled actions. All KEEP forks reconcile,
and the neutral window/fill/request paths and generation features exactly match
their saved reference. Target statuses retain 1,927 eligible and 12 missing
ten-second continuation marks; all 1,939 short targets are eligible. Feature
attachment passed at 03:03:26.601858 UTC, retaining every action in both files.

The wrapper records rc=0, 1,569 MiB peak RSS, 1,545.99 s wall and 1,544.85 s CPU
user time for the build; data attachment returned rc=0 in 3.44 s. Its pandas
fragmentation warnings are nonfatal and do not change the fixed feature code.
Full-day saved-capture reconstruction remains in the declared final audit;
the earlier independent four-window smoke remains passed. No numerical audit
was launched as a third worker while the shared spare slot was occupied.

The driver started Oct24 seed worker 744022 on core 8. No recovery/value model
has been fitted and no improvement is established. Source freeze and all
criteria remain unchanged. The plan now makes the endpoint-inventory training
penalty versus full-path inventory-exposure evaluation distinction explicit;
that source-reading clarification changes no label, coefficient or criterion.


## 2026-11-13T03:30:26.179371+00:00 — seed audit complete, recovery proxy fitted

Oct24 seed build passed at 03:27:14.925686 UTC: 288 windows, 15,372 neutral
generations and 1,802 sampled actions, with exact reference parity and all KEEP
forks reconciled. Its ten-second targets retain 1,795 eligible and seven missing
continuation marks; all short targets are available. Wrapper rc=0, 1,466 MiB
peak RSS, 1,428.31 s wall and 1,427.19 s CPU user time. Feature attachment passed
at 03:27:54.261506 UTC with all actions retained.

Only the pure driver was held at the build boundary; numerical workers were
uninterrupted. Both full seed days' saved captures were independently rebuilt
using the frozen verifier. `seed_evidence.json`, as of 03:28:09.487422 UTC,
binds 29 artifacts and reconstructs all 3,741 targets. Source `75c39780` supplies
the reproducible checkpoint entry point; the scientific freeze stays `78285fe0`.

The supplemental proxy forecast audit was declared at 03:28:16.740240 UTC,
before fitting, with seven positive/refusal control groups passed. It compares
later neutral short-value forecasts with the seed-only mean; no new model,
policy or threshold. Its historical diagnostic is still pending. Run it after
all recovery data builds; the unchanged primary driver does not run it.

The same driver was identity-checked and resumed at 03:29:14.429063 UTC. The
fixed recovery fit passed at 03:29:15.608728 UTC on all 3,741 eligible seed short
targets, with the pinned 300 rounds and saved-model prediction reconstruction.
The actual fitted-rule smoke passed at 03:30:23.657537 UTC (four windows,
279 neutral generations, 16 eligible/reconciled actions); feature attachment
passed at 03:30:26.011099 UTC. Smoke wrapper rc=0, 1,905 MiB peak RSS and 68.06 s.

Full Oct23 recovery-label worker 986996 is now active on core 8. The fixed
model's later forecast quality, six cancellation-value fits and all full policy
comparisons remain pending. This is integration and construction evidence;
no prediction-quality or strategy-improvement claim follows.


Training-support audit, 2026-11-13T03:54:16.600667+00:00: the exact fixed proxy
and its 3,741 seed actions pass five control groups. All 25–500 ms raw/signed
Binance return inputs have zero constructed bins under the unchanged settings,
despite valid varying values. The 300 actual trees have four or five leaves;
65 of 83 inputs are available and 64 receive splits. This describes a training
support limitation, not forecast performance or a reason to tune the frozen
model. See [the full descriptive review](20261113_recovery_training_support_review.md)
and `training_support_audit.json`. The primary full recovery-label build
continues; the later-day proxy forecast audit remains pending.


## 2026-11-13 03:57 UTC — first full recovery day and early capture verification

Full Oct23 recovery labels passed at 03:56:07.434849 UTC; data attachment passed
at 03:56:11.085029 UTC. Population: 288 windows, 18,115 neutral generations and
1,939 sampled actions. Ten-second status: 1,927 eligible / 12 missing marks;
all short targets eligible. Neutral reference parity is exact, all 1,939 KEEP
forks reconcile and both feature files preserve the sampled population. Build
wrapper rc=0, 1,918 MiB peak RSS, 1,541.45 seconds. Data wrapper also exits zero.

The frozen build verifier was advanced through committed entrypoint `2b5114b1`
on both the fitted-rule smoke and full Oct23. `first_recovery_evidence.json`
is as of 03:57:31.945411 UTC and binds 34 artifacts. All 16 smoke and 1,939
full-day captured targets reconstruct. Recovery permissions and placement
clocks reconstruct from recorded estimates; this checkpoint does not recompute
every recorded estimate from market inputs. Existing positive/refusal controls
and their logs are bound. Wrapper rc=0, 171 MiB peak RSS, 3.53 seconds.

The original driver advanced automatically to Oct24 at 03:56:11.262630 UTC,
worker 1220167 on core 8. Eight recovery-label days, six value fits, local and
full policy evaluations remain. The separately declared proxy forecast audit
must be run after all recovery data builds. No forecast or policy improvement
is established; no live change, model retuning or new policy variant.


## 2026-11-13 04:03 UTC — predeclared forecast audit launcher active

Committed `model_recovery_proxy_audit_driver.py` at `2f30cab9` and launched the
pure orchestrator as PID 1289091. Its started receipt is as of
04:03:24.911686 UTC. Two control groups check exact prerequisite identity with
six corrupt-receipt refusals, auxiliary-slot occupancy and the global numeric
cap. It waits for all nine data receipts and a free core 9, then invokes the
unchanged predeclared forecast audit through the shared wrapper. Complete or
partial audit outputs are never automatically rerun. Main driver 3592076 and
Oct24 worker 1220167 continue; no scientific source or policy changed.

The training-support review now explicitly records that `s_return_accel`
contains the 250 ms return and has 13 model splits on the same 3,741 training
actions (audit as of 03:54:16.600667 UTC). Excluded direct inputs do not imply
absence of all short-term information. The existing numeric receipt remains
correct and unchanged; this clarifies its interpretation.


## 2026-11-13 04:11 UTC — latest front-pull result integrated without changing the candidate

Read the separate front-pull report (as of 03:54:02 UTC), its saved JSON and
actual committed fit/report source. The published policy-improvement readings
fail despite beating random pulling. Its training conditions on future level
survival, uses other-date rather than earlier-only folds and inherits the guard
score's held-out-date exposure; its S4 rest lane differs from our full engine.
See [the source-grounded integration review](20261113_front_pull_integration_review.md)
for exact populations, artifact identities and the report's zero-days-better
wording correction. No independent rerun or statistical re-evaluation is
claimed. No score, threshold, queue rule or new variant is imported. The current
causal value/recovery experiment continues unchanged on Oct24.


## 2026-11-13 04:20 UTC — second full recovery-label day complete

Oct24 recovery build passed at 04:20:19.009313 UTC and feature attachment at
04:20:22.652841 UTC. All 288 windows, 15,372 neutral generations and 1,802
sampled actions remain accounted for. The ten-second status is 1,796 eligible
and six missing continuation marks; all short targets are eligible. Neutral
reference parity is exact and all 1,802 KEEP forks reconcile. This recovery
population's statuses differ from the seed run's 1,795 eligible / seven missing;
the seed receipts remain unchanged and are not substituted for recovery labels.
Independent saved-capture reconstruction for Oct24 remains in the final verifier.

Build wrapper rc=0, 1,899 MiB peak RSS, 1,447.93 s wall and 1,446.21 s CPU.
Data wrapper rc=0, 1,237 MiB, 3.47 s wall. The data and both review copies of each
receipt have identical hashes. Build receipt SHA:
`6bfb30ba04ae6c3a955cdaf7dbb2502bdbc6679275bd7ed2074a550870afe8dc`;
data receipt SHA:
`5c8ae9a2d6f9bb8dc844f56a857f8223d9fc6453df8a695dabdb80e526ce2461`.

The same driver advanced to Oct25 at 04:20:22.824324 UTC, worker 1443677.
The proxy-audit launcher is live and correctly reports seven missing data days
as of 04:20:25.732306 UTC. No restart, threshold change, model refit or policy
variant was introduced. Forecast and strategy improvement remain unproven.


## 2026-11-13 04:43 UTC — third full recovery-label day complete

Oct25 recovery build passed at 04:43:46.593936 UTC and feature attachment at
04:43:50.233156 UTC. All 288 windows, 14,973 neutral generations and 1,714
sampled actions are accounted for. Every sampled short and ten-second target
is eligible. The neutral population separately retains one `censored_30s`
generation, alongside 6,340 accepted/no-cancel, 5,030 not sampled, 1,888
post-only refusals and 1,714 forked. No generation is silently dropped.
Neutral parity is exact and all 1,714 KEEP forks reconcile. Independent saved
capture reconstruction for Oct25 remains in the final verifier.

Build wrapper rc=0, 1,753 MiB peak RSS, 1,403.95 s wall / 1,402.26 s CPU.
Data wrapper rc=0, 1,083 MiB, 3.45 s wall. Each receipt's data and two review
copies match. Build receipt SHA:
`c5f3b1a2212c855d7573e32f291fd613c599a963fb296585d4afcbc4c4ea3820`;
data receipt SHA:
`5776d699e5982c3b6a130677e56bba61eff788334b8b8adcc75afebce25da565`.

The same driver advanced to Oct26 at 04:43:50.385737 UTC, worker 1656807.
The proxy-audit launcher correctly reports six missing data days as of
04:44:06.877178 UTC. The read-only wait monitor ended on the completed data
receipt; it did not signal a numerical worker. No model, threshold or policy
changed. Three of nine recovery data days are complete; forecast quality and
strategy improvement remain unproven.


## 2026-11-13 05:10 UTC — fourth full recovery-label day complete

Oct26 recovery build passed at 05:10:42.456031 UTC and feature attachment at
05:10:46.161300 UTC. All 288 windows, 19,116 neutral generations and 2,033
sampled actions are accounted for. Ten-second targets: 2,020 eligible / 13
missing continuation marks; all short targets eligible. Neutral statuses:
8,527 accepted/no-cancel, 6,082 not sampled, 2,473 post-only refusals, 2,033
forked and one `censored_30s`. Neutral parity is exact; all KEEP forks reconcile.
Independent saved-capture reconstruction for Oct26 remains in the final verifier.

Build wrapper rc=0, 1,649 MiB peak RSS, 1,612.26 s wall / 1,610.47 s CPU.
Data wrapper rc=0, 1,237 MiB, 3.52 s wall. Data and both review copies of each
receipt match. Build receipt SHA:
`31ee083e941ff3be3c7106ab7f1ae831ea4b77828503b2339bbb6171d64b80b4`;
data receipt SHA:
`e0c86cbddeb0216db123861a509d2b3a4bbe52e5776d7ffcfabfaab1a236d010`.

The same driver advanced to Oct27 at 05:10:46.338516 UTC, worker 1905620.
The live proxy-audit launcher reports five missing data days as of
05:10:48.247811 UTC and correctly observes occupied cores 8 and 9. The other
owner's order extraction was left untouched. A stale tool-monitor session was
replaced only after directly confirming the same numerical worker remained
live; no numerical process was restarted. Forecast and policy evaluations are
still pending, with no new model, threshold or policy changes.


## 2026-11-13 05:34 UTC — fifth full recovery-label day complete

Oct27 recovery build passed at 05:34:34.440212 UTC and feature attachment at
05:34:38.126849 UTC. All 288 windows, 15,992 neutral generations and 1,762
sampled actions remain accounted for. Ten-second targets: 1,759 eligible /
three missing continuation marks; all short targets eligible. Neutral statuses:
6,805 accepted/no-cancel, 5,432 not sampled, 1,992 post-only refusals, 1,762
forked and one `censored_30s`. Neutral parity is exact and all KEEP forks
reconcile. Independent saved-capture reconstruction remains in the final verifier.

Build wrapper rc=0, 1,708 MiB peak RSS, 1,428.30 s wall / 1,425.85 s CPU.
Data wrapper rc=0, 1,237 MiB, 3.49 s wall. All three copies of each receipt
match. Build receipt SHA:
`cf35a9d1eb7d384749bb2c1f9a4eace1ce9179fcc4429f5bec74ffcf4b739e58`;
data receipt SHA:
`c08f9d7c13d9e0cb3a4447931d5e3f52b3f28eb2b184ae6047f24f9abec47a07`.

The same driver advanced to Oct31 at 05:34:38.303390 UTC, worker 2122355.
The proxy-audit launcher correctly reports four remaining data days as of
05:34:49.454301 UTC. Both numeric slots are occupied; the other owner's order
extraction is untouched. Five built dates include the seed/calibration dates,
so they are not five evaluated outer test folds. All forecast and full-policy
results remain pending. No model, threshold, policy or success criterion changed.


## 2026-11-13 05:53 UTC — sixth full recovery-label day complete

Oct31 recovery build passed at 05:53:32.680838 UTC and feature attachment at
05:53:36.246067 UTC. All 288 windows, 9,475 neutral generations and 1,225
sampled actions remain accounted for. Every sampled ten-second and short
target is eligible; no missing-target or neutral censoring status occurs.
Neutral statuses: 3,514 accepted/no-cancel, 3,710 not sampled, 1,026 post-only
refusals and 1,225 forked. Neutral parity is exact and all KEEP forks reconcile.
Independent saved-capture reconstruction remains in the final verifier.

Build wrapper rc=0, 1,487 MiB peak RSS, 1,134.54 s wall / 1,132.71 s CPU.
Data wrapper rc=0, 1,235 MiB, 3.41 s wall. All three copies of each receipt
match, as do all nine captured-build file hashes and both feature-file hashes.
These are identity checks, not independent target reconstruction. Build SHA:
`8b680001849f992ddb6440dbf5e3e2bc4be49bd1b0ffa1d5fb12e7fe2d34ad84`;
data SHA:
`8826aff9af4cd21a357a59c14978307dc18fb1bbfe4185afde1b09a411ca9b6e`.

The same driver advanced to Nov1 at 05:53:36.420872 UTC, worker 2296268.
The live proxy-audit launcher reports three remaining data days as of
05:53:50.426287 UTC. Both numeric slots are occupied; the other owner's public
order study is untouched. Six built dates include training/calibration data,
not six completed outer evaluation folds. Forecast and full-policy results
remain pending. No frozen model, threshold, policy or criterion changed.


## 2026-11-13 06:06 UTC — completed public-order study reviewed

The separate `orders_20261113/orders.json` report completed at 06:04:01 UTC
on nine consumed dates and 911,353 observed order records. Its source bytes
match `9f8ec33848123594cb153070f270e2e988789c67`; the inspected report SHA is
`7aee33e2649bd46f63ecae57a033e62264c5b86d607748ac572a70b3982cdbd6`.
The [source integration review](20261113_public_orders_integration_review.md)
distinguishes repeated fills from cancellation timing, filled-order/size
selection from a causal retention advantage, multiple filled prices from
simultaneous quoting layers, and observed within-window flows from account
inventory. It identifies unequal creator/at-best coverage and the summary's
default-zero `legs_unmapped` field, absent in all nine daily receipts.

This is a read-only interpretation of existing results, without an independent
rerun or a new performance claim. The other owner's artifacts are unchanged.
No score, timer, size, layer count, threshold or variant is imported. The same
Nov1 recovery-label worker 2296268 continues on core 8, with the primary and
proxy-audit drivers live. Forecast and full-policy evaluations remain pending.


## 2026-11-13 06:13 UTC — seventh full recovery-label day complete

Nov1 recovery build passed at 06:12:58.557880 UTC and feature attachment at
06:13:02.134922 UTC. All 288 windows, 10,653 neutral generations and 1,305
sampled actions remain accounted for. Ten-second targets: 1,301 eligible /
four missing continuation marks; all short targets eligible. Neutral statuses:
4,077 accepted/no-cancel, 3,943 not sampled, 1,328 post-only refusals and 1,305
forked; no neutral censoring. Neutral parity is exact and all KEEP forks
reconcile. Independent saved-capture reconstruction remains in the final verifier.

Build wrapper rc=0, 1,674 MiB peak RSS, 1,162.32 s wall / 1,160.36 s CPU.
Data wrapper rc=0, 1,235 MiB, 3.40 s wall. All three copies of each receipt
match, along with all nine captured-build and both feature-file hashes.
Build receipt SHA:
`79c4c3056dfae12a69896ac389db5b7eafffad7f6200930c8f88318b95ecadaa`;
data receipt SHA:
`bf85c162f0a0e4bd6d30690d697594c63b4bdbb9c3e6c78b850da92287994027`.

The same driver advanced to Nov2 at 06:13:02.307211 UTC, worker 2471516.
The live proxy-audit launcher reports two remaining data days as of
06:13:11.408356 UTC. Only core 8 is numerically occupied at this checkpoint;
the separate public-order study finished. Seven built dates include the
training/calibration dates, not seven evaluated outer folds. Forecast and
full-policy results remain pending. No model, threshold, policy or criterion
changed, and no live action was taken.


## 2026-11-13 06:25 UTC — public-order publication claims checked

The published `20261113_orders_results.md` is now inspected (SHA
`158b24cc764257d4a4d303b073888bf343e0058c4220246d827120e5672a3c14`), and its
machine review copy matches the original report. The existing integration
review now explicitly separates score/disappearance association from a
successful cancellation policy, and inferred-presence gaps from observed
cancel decision times. Inspection of the frozen fill allocator and current
default-quote hook confirms partial fills can leave a five-share order
resting; no empirical partial-fill frequency is inferred from source alone.
Larger size remains an untested change in this experiment and would require
its own risk-controlled comparison. No other owner's artifact is edited and
no scientific source, setting or criterion changes. Nov2 remains active.


## 2026-11-13 06:37 UTC — eighth full recovery-label day complete

Nov2 recovery build passed at 06:37:34.238168 UTC and feature attachment at
06:37:37.850608 UTC. All 288 windows, 17,944 neutral generations and 1,829
sampled actions remain accounted for. Ten-second targets: 1,823 eligible /
six missing continuation marks; all short targets eligible. Neutral statuses:
7,770 accepted/no-cancel, 5,662 not sampled, 2,683 post-only refusals and 1,829
forked; no neutral censoring. Neutral parity is exact and all KEEP forks
reconcile. Independent saved-capture reconstruction remains in the final verifier.

Build wrapper rc=0, 1,510 MiB peak RSS, 1,472.11 s wall / 1,469.09 s CPU.
Data wrapper rc=0, 1,236 MiB, 3.44 s wall. All three copies of each receipt
match, together with all nine captured-build and both feature-file hashes.
Build receipt SHA:
`8c716651ea33657bfe164de9c2d4a698e1ae83762974f96ef35b3b50c23ca9a5`;
data receipt SHA:
`0a061a2b120277ef92cba6e2669411589eda764da5a966ba7c8d0eaa216f9532`.

The same driver advanced to Nov3 at 06:37:38.021510 UTC, worker 2693862.
The live proxy-audit launcher reports only Nov3 data outstanding as of
06:37:52.603152 UTC. Eight built dates include training/calibration dates,
not eight evaluated outer folds. Forecast and full-policy results remain
pending. No model, threshold, policy or criterion changed; no live action.


## 2026-11-13 — prepare complete-data reconstruction checkpoint

While Nov3 is building, `pf_recovery_all_data_checkpoint.py` advances the
already required frozen capture verifier to the complete-data boundary. It
uses the same checks and bound positive/corrupt-target/population controls as
the first-day checkpoint, extended to all nine full recovery days. Commit the
entrypoint before execution; run once after all data and the queued proxy
forecast audit finish, on a genuinely free auxiliary slot through `run_light.sh`.
Output is the immutable `all_recovery_data_evidence.json`. No model, target,
metric, threshold, policy or success gate is added. The final full experiment
verifier remains required, including full replay inference and portfolio checks.


## 2026-11-13 07:23 UTC — all data, local fits, and accounting failure preserved

All nine full recovery data days completed at 07:02:13.217731 UTC: 2,592 windows,
138,743 neutral generations, 15,432 sampled actions, 15,382 ten-second eligible /
50 missing marks; all short targets eligible, four separate neutral censored
statuses. All six chronological value fits and saved-model prediction checks
completed by 07:02:38.076419 UTC. Local report as of 07:02:52.217695 covers
9,945 eligible / 32 missing test actions. SELECTIVE10 does not beat DIRECT10
at matched counts; its confidence interval crosses zero. See the local review.

All-data reconstruction passed at 07:08:13.800158 UTC on 15,432 captured targets,
125 artifacts (wrapper rc=0, 180 MiB, 23.46 seconds). This operational checkpoint
used temporarily free core8 while the pure driver was held; no worker was
interrupted. The same driver resumed at 07:08:59.659753 UTC, superseding the
hold receipt. X/stress/age0 integration smokes pass. The age500 smoke simulation
finished but settlement correction raised TypeError on its empty direct10 fill
table. Original driver 3592076 terminated at 07:12:56.478932 UTC. Its log and
partial output are preserved by empty_fill_age500_failure.json. No canonical
age500 success receipt exists. A versioned accounting adapter must prove raw
replay parity before a corrected receipt; no model or strategy change.

The separately queued fixed-proxy audit completed at 07:23:08.921632 UTC,
11,691 saved actions including 9,977 test actions across six dates; all short
labels available. Forecast RMSE 5.19342 vs constant 5.19597 c, equal-day MSE
improvement interval [-0.09024,+0.17810] c². Nonnegative predictions on 3,399
test actions sum to +1,480.99 c vs observed -527.27 c. No reliable forecast gain
is established. Audit and runner receipts, all their bound file hashes, and
three mirrored copies match. No live changes; full policy evaluation pending.


## 2026-11-13 07:31 UTC — corrected smoke exact; full recovery replay started

Supplemental source a4c190ec and declaration c38f6acd preserve the entire
scientific freeze and normalize numeric types only for empty fill tables.
Seven control groups passed at 07:28:02.056099 UTC, including the actual failed
empty case and nonempty original parity. The initial replacement launcher used
system Python and failed import before starting work; its log/failure receipt
are preserved and superseded by the project-venv launcher 3159222.

Corrected age500 smoke raw parity passed at 07:31:17.944617 UTC: 18 raw parquet
tables and the decompressed event stream exactly match the failed attempt.
All four policies retain eight windows; DIRECT10 still has zero fills. The
canonical smoke receipt carries correction provenance and supersedes the old
failure in-band. Wrapper rc=0, 1,893 MiB, 116.36 seconds. Mirrored receipts and
all parity/top-level/presence hashes match; corrupted provenance links refuse.

The same driver started full Oct26 primary replay at 07:31:18.208066 UTC on
core8, worker 3178539. Eighteen full scenario-day runs, report and final verifier
remain. Scientific settings and success predicates are unchanged. No model
improvement or live benefit has been demonstrated. See the correction review.


## 2026-11-13 — size/ladder results integrated without changing the freeze

Reviewed the separate report as of 07:22:49 UTC (nine consumed dates, 2,592
windows / 5,184 window-sides; 15,984 untimed first-minute legs retained as
statuses). Its size-per-share improvement is not shown; the tested ladder
loses on all nine dates. Source inspection identifies summed independent rung
allocation without shared trade volume/inventory, fill-record counts versus
the public study's distinct-event repeat counts, and post-hoc removal of young
fills rather than full-path suppression. R410's machine count is 5/3 dates
improving, not the published 5/4. Negative reported readings are preserved;
no corrected historical result is claimed. Read the size integration review.

Own full Oct26 primary recovery replay remains live under driver 3159222,
worker 3178539 on core8. No frozen model, policy, criterion or live change.


## 2026-11-13 07:46 UTC — user queue-retention mechanism verified

The user clarified that larger resting orders preserve earned priority through
partial fills, avoiding exhaustion-driven requeueing. The current five-share
training population cannot establish the value of a 50-share residual. Recorded
an explicit next comparison of residual KEEP versus cancel/repost, plus practical
size comparison under a common exposure budget that genuinely permits 50 shares.
The current cap-ten frozen experiment remains unchanged.

Synthetic checker as of 07:46:35.686593 UTC: four cases (five/50 shares, M/U),
two taker events each. Under M the small order exhausts and re-enters behind
new depth; the large residual keeps priority and takes the next fill. Under U,
both receive the same total quantity; a false advantage claim refuses. The
public report's exact repeat-fill counter also counts three same-event legs
as three fills, whereas distinct taker identities number one. The mismatch is
proved on fixtures, not measured historically. All tests pass, wrapper rc=0,
125 MiB, 0.44 seconds. Earlier admission refusal preserved; no work ran on a
busy core. Version-two receipt corrects displayed-versus-modeled queue metadata
in-band, with paths/outcomes unchanged and positive/refusal schema controls.

The full Oct26 primary recovery replay remains live on core8, worker 3178539,
under the single replacement driver 3159222. No model, criterion or live change.


## Implementation checkpoint, 2026-11-13 07:58 UTC

`pf_size_priority_policy.py` (fcee3270) now supplies an explicit immutable order-
quantity/inventory-budget profile and an isolated subclass of the inherited
causal default hook. Exact source anchors parameterize new quantity, available
capacity and the hypothetical-entry feature state. Existing accepted order
size is retained after a partial fill; client-known reservations remain in the
capacity calculation. The original source/class is not monkey-patched.

It refuses a requested order larger than its declared inventory budget, direct
use of the inherited five-share learned modes, and the legacy age-probe builder.
A helper exposes only notified fill quantity and known residual; latent physical
remaining quantity or queue position is excluded. This is a preparation
component, not the complete size experiment or a historical runner.

Eight synthetic controls are committed in `tests/test_size_priority_policy.py`:
original-profile equivalence, retained large residual identity, late-fill/cancel
reservations, latent-state mutation invariance, common-budget constraints,
invalid profiles/old-model refusal, partial/exhausted statuses and strict source
anchor checks. **They have not run yet.** Pure launcher **3412927**, source
93ee8ef6, is live and waits for the other owner's `run_deep_pull.sh` chain to
finish and core9 to be free. It will run only this bounded unit suite through
the shared wrapper, without changing the primary replay. Its initial system-
Python import failure is preserved and superseded by the project-venv launch;
no test worker started during that failed import.

Next: inspect the actual control result, then connect the partial-fill action
ledger, native-engine residual reset, quantity-aware targets and full replay
accounting. The complete new pipeline and its experimental design still need
to be committed/frozen before historical execution. Current recovery worker
3178539 remains active on core8 under driver3159222; no current model/target,
policy threshold, execution criterion or live behavior changes.


## Native integration preparation, 2026-11-13 08:20 UTC

Source **18d25ce1** connects the first known partial-fill ledger to the native
execution path. The ledger was introduced at 8c6f7711/68a75622; this version
adds the native fill-notification callback before the engine clears an exhausted
known slot. A notification updates known quantity but cannot itself select an
action. The first eligible actual decision supplies the action, observed age
and causal features. Unfilled, refused, exhausted, pending, unavailable-state,
withdrawn and censored generations remain in the population. Quantity comparisons
use an explicit 1e-9 share roundoff tolerance.

An integration gap in the first quantity adapter was found before historical
execution: its exact target-body anchors do not exist on the installed
`ReservedHook` wrapper. It therefore refuses that installed stack. The new
`pf_size_priority_policy_v2.reserved_hook_class` composes the sized implementation
**below** the unchanged reservation methods. Cancellation continues reserving
client-known quantity until effective cancel plus the last possible fill-notice
delay; pending-to-accepted identity and notification updates are preserved.
No old wrapper or frozen source is bypassed/edited.

The observer is separate from policy: it calls the same targets first, then
records the client-visible state. New controls compare the observed and
unobserved native runs at both sizes, including orders, fills, generation
records and independent inventory accounting. A synthetic larger residual
should survive two taker events while the exhausted small order's replacement
queues behind newly arrived depth. These new native results are **not yet run**.

The original eight-test unit suite remains queued under **3412927**. The next
**23 tests** (ten installed-stack reservation cases, nine ledger cases and four
native-engine cases) are queued under **3597883**, after that suite passes and
the other owner's complete `run_deep_pull.sh` chain releases core9. Their
source hashes are bound in
[the queued native-controls receipt](../reviews/model_recovery_20261113/size_priority_native_controls_runner_started.json).
Syntax compilation passes; runtime correctness is pending. No duplicate worker
is launched and the primary replay keeps core8.

The residual CANCEL/REPOST fork, paired value targets, chronological new fits
and larger-order historical evaluation are still outstanding. This is
preparation, not a complete size experiment or evidence of a better model.

Recovery full replay checkpoint: Oct26/X finished at 08:08:19.617350 UTC, four arms × 288 windows; 25 named artifacts and three receipt copies verified in `first_full_run_review.json`. Oct27/X worker3509979 is active. No frozen source, model, threshold or success criterion changed.


## 2026-11-13 08:48 UTC: observation verified; residual forks queued

The original8 quantity cases pass. The corrected installed-stack/ledger/native
observation and pure-reset groups pass30 cases; the combined36-case attempt
still failed because all six new fork cases refused their source template.
The template and nested notification indentation are now corrected; M/L source
compilation passes. Thirty-eight runtime cases are queued under3820900 behind
the active keep-place chain. The original failed sources/logs and the cancelled
pre-execution attempt are preserved. No passing full-fork result is claimed.

The strict continuation-value component is committed through61fb8eea, including
missing marks, retained abort statuses, complete inventory-square time and
peak risk in native fill order. Its seven analytic tests remain unrun. Read
[the native construction review](20261113_size_priority_native_controls.md)
for exact receipts, source versions, queue state and remaining pipeline work.
