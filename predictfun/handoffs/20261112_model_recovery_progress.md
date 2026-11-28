# Recovery full replay running — accounting correction verified

As of **2026-11-13T08:44:10.876717+00:00**, all nine recovery-data days,
six chronological value fits and both local forecast audits are complete.
All four integration scenarios pass. Full primary replays Oct26 and Oct27 are
complete, each with four policies ×288 windows. Oct27 reports exact default
parity, reconstructed recovery events and457 sampled inferences; all25 named
artifacts, three receipt copies and correction provenance were checked in
`second_full_run_review.json`. **Two of18** scenario-day runs are complete;
Oct31/X worker3839966 is active under driver3159222. Full portfolio reporting
and independent final verification remain; no multi-day improvement verdict.

The empty-fill settlement correction passed raw parity at 07:31:17.944617 UTC:
all 18 raw parquet tables and the decompressed recovery-event stream match
the original failed attempt exactly. The corrected smoke retains its zero-fill
arm and all eight windows per each of four policies. Original failed source,
artifacts and logs remain preserved. Read [the correction review](20261113_recovery_accounting_repair.md).
Scientific source **78285fe0**, models, targets, policy and success criteria
remain fixed; supplemental accounting source **a4c190ec** changes no strategy.

The [local action-model review](20261113_recovery_local_review.md), as of
07:02:52.217695 UTC, covers 9,945 eligible / 32 missing test actions across six
consumed dates. Reduced local cancellation does not establish a matched-count
advantage over the direct model. The [fixed-proxy audit](20261113_recovery_proxy_audit_review.md),
as of 07:23:08.921632 UTC, covers 9,977 test actions, all short-target eligible;
its day-level forecast-improvement interval includes zero and its nonnegative
prediction region has negative aggregate observed value. Neither finding is
portfolio P&L or untouched-day validation.

## Completed construction evidence

Full data as of 07:02:13.217731 UTC: nine days, 2,592 windows, 138,743 neutral
generations and 15,432 sampled actions. Ten-second labels retain 15,382 eligible /
50 missing marks; all short targets eligible, four separate neutral censored
statuses. All-data capture reconstruction passed at 07:08:13.800158 UTC on all
15,432 sampled actions, binding 125 artifacts. Recovery label-branch timing is
checked from recorded estimates, not an independent recomputation of each
estimate. The final verifier remains required. All six earlier-core fits and
saved-model prediction checks completed by 07:02:38.076419 UTC.

## Operational state

The original driver 3592076 exited at 07:12:56.478932 UTC on the empty-fill
settlement error. Its earlier all-data audit hold was already superseded by
the resume receipt at 07:08:59.659753 UTC; no hold remains. Replacement launcher
3149455 failed to import numpy under system Python before launching a worker.
The preserved import-failure receipt is superseded by the project-venv launcher
3159222. Only this replacement driver is live; no duplicate run is needed.
The queued fixed-proxy audit also finished normally; its launcher is terminal.

Seven accounting/parity control groups passed at 07:28:02.056099 UTC. Corrected
age500 smoke wrapper: rc=0, 1,893 MiB peak RSS, 116.36 seconds. Its canonical
receipt links the supplemental declaration, controls, original failure and raw
parity; a metadata review checks those identities and rejects corrupted links.
All full stages use the same adapter; the final evidence check binds this chain.

The original temporal experiment remains [complete and negative](20261113_model_upgrade_complete_review.md).
Earlier construction and operational checkpoints remain in
[the execution history](20261112_model_upgrade_execution.md). No protected
payload, live action, promoted model or demonstrated strategy improvement.

User priority-retention clarification and the passing four-case synthetic check are recorded in [the size-priority objective](20261113_size_priority_objective.md). The next comparison must price the remaining large order at its earned queue position; current five-share training cannot support extrapolation to 50 shares. No change to the running frozen replay.

Larger-order preparation (07:58 UTC): isolated quantity/budget adapter fcee3270 is committed, with eight unit controls queued under launcher 3412927 after the other owner releases core9. Tests remain pending; no historical larger-order runner or fitted model is claimed. Read the size-priority objective for remaining integration steps.


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


## 2026-11-13 12:50 UTC: nine full scenario-days verified

Recovery is **9/18** complete: all six primary X dates and stress dates
Oct 26, Oct 27 and Oct 31. The Oct 31/stress runner finished at
**2026-11-13T12:48:31.107964+00:00**, with four policies × 288 windows.
It reconstructed 268 sampled inference rows, recovery events and the exact
neutral default path. These are execution-integrity counts, not a performance
verdict. The wrapper exited zero after 2,020.43 seconds, using 1,740 MiB peak RSS.

The [completed-run receipt](../reviews/model_recovery_20261113/run_stress_20261031.json)
has SHA `7e73d9dda851b21d7586fbb8f9d7ca979bafecab81ad2b7d96549a77dcd46347`.
The [artifact review](../reviews/model_recovery_20261113/run_stress_20261031_artifact_review.json),
as of 12:49:29.822461 UTC, verifies all 25 named artifacts and three receipt
copies. Its SHA is
`303a87c1ca61e3781982a1402ec3bfc8fae805e42f873393ec7ddf31ed45de76`.

New reusable review source `model_recovery_stage_review.py`, committed at
**4379a7a3**, binds its own committed source and checks exact receipt-level
policy/window coverage. Positive controls accept the actual completed record;
11 corrupt-input controls reject broken accounting links, missing/duplicate
policies, missing artifacts, wrong dates, incomplete window counts, unchecked
inference and an altered artifact digest. Numerical reconstruction claims are
inherited from the frozen runner; the review does not perform a second replay.

Nov 1/stress worker **1867188** is active under recovery driver **3159222**.
Three stress dates and six age500 dates remain before the final portfolio
report and reconstruction. The optimized size driver **1576587** still waits
for the entire recovery chain and its verified final evidence. All 169 size
tasks remain; see [the completed size execution handoff](20261113_size_fast_handoff.md).
The 302 recovery, 385 size and six computational-adapter source identities
remain unchanged. No improved-model verdict or live change.

A recovery runtime switch was considered and left unadopted. The earlier
feature-container prototype has exact eight-window parity, but its separate
whole-engine smoke timings do not establish a material full-day speedup.
Those earlier receipts remain in `model_upgrade_inference_parity_20261112`;
this checkpoint starts no new optimization trial or replacement recovery driver.


## 2026-11-13 13:24 UTC: ten full scenario-days verified

Nov 1/stress finished at **2026-11-13T13:23:39.687265+00:00**, advancing the
frozen recovery replay to **10/18** scenario-days. Four policies each cover 288
windows; the runner reconstructs 508 sampled inference rows, recovery events
and the exact default neutral path. These are integrity counts, not an
improved-policy verdict. The wrapper exited zero with 1,848 MiB peak RSS,
2,108.43 seconds wall and 2,103.82 seconds user CPU.

The [run receipt](../reviews/model_recovery_20261113/run_stress_20261101.json)
has SHA `73a9ffffc68c6b9c164ea12fa6bb4bdb250eb45bf3f38642709dfad0e1c64412`.
The [artifact review](../reviews/model_recovery_20261113/run_stress_20261101_artifact_review.json),
as of **13:24:23.558524 UTC**, binds all 25 artifacts and three receipt copies,
with the existing positive and 11 corrupt-input controls. Its SHA is
`459df3de54758d7ce7ea73e84709f30423de9f8edf7a06ab43e55d7176ae92e9`.
Numerical reconstruction remains the frozen runner's check, not a second replay.

Nov 2/stress worker **2177940** started at 13:23:40.021595 UTC under driver
**3159222**. Two stress dates and six age500 dates remain, followed by the final
portfolio report/reconstruction. Size driver **1576587** still waits for that
complete evidence chain; all 169 size tasks remain. The 302 recovery, 385 size
and six computational-adapter source identities are unchanged. Separately,
22 synthetic action-depth target/native/artifact controls now pass; see
[the preparation record](20261113_action_depth_target_preparation.md).


## 2026-11-13 14:08 UTC: eleven full scenario-days verified

Nov 2/stress completed at **2026-11-13T13:58:51.388089+00:00**. The
[run receipt](../reviews/model_recovery_20261113/run_stress_20261102.json), SHA
`6efc190cc52c41c74af71e926e28805779a4d3b0b582484cf5d791f48c5879ca`,
covers four policies × 288 windows and 409 reconstructed inference rows.
The [artifact review](../reviews/model_recovery_20261113/run_stress_20261102_artifact_review.json)
binds all 25 artifacts and three copies with the existing positive and 11
refusal controls, as of **14:04:33.008063 UTC**, SHA
`1729216e30017e334f12a36b1099d331c937cd66bf737bd1b276561c9be2d30c`.

The frozen chain is now **11/18** full scenario-days. Nov 3/stress worker
**2500825** runs under driver **3159222**; one stress date and six age500
dates remain before the final report/reconstruction. Size driver **1576587**
continues waiting. The recovery, size and fast-adapter scientific/source
identities are unchanged. Separately, 46 synthetic depth integration tests
pass and the completed FUTURE-BEST aggregate audit is recorded in its own
[review](20261113_future_best_model_integration_review.md). No improved-model
verdict or live change.


## 2026-11-13 14:40 UTC: twelve full scenario-days verified

Nov 3/stress completed at **2026-11-13T14:33:35.271530+00:00**. The
[run receipt](../reviews/model_recovery_20261113/run_stress_20261103.json), SHA
`4f00d2c70d00ef20b0813b2ba710f136b63169fb1622e21495663b4d88f150fa`,
covers four policies × 288 windows and 379 reconstructed inference rows.
The [artifact review](../reviews/model_recovery_20261113/run_stress_20261103_artifact_review.json)
binds all 25 artifacts and three receipt copies, with the existing positive
and 11 refusal controls, as of **2026-11-13T14:35:01.910248+00:00**, SHA
`10718352a1a5de2f18c42106fe53cf6a50fa191719d05a664028d9a85fa55be4`.
These are integrity counts, not a performance verdict.

All six primary and six stress dates are now complete: **12/18** total.
Oct 26/age500 worker **2822379** is active under driver **3159222**. Six
fill-age sensitivity dates and final report reconstruction remain. Size driver
**1576587** still waits; its 169 stages and all scientific/computational
freezes are unchanged. The separate depth forecast pipeline is preparing its
full declaration after 76 synthetic tests; it will reuse immutable size
captures and wait for the complete size evidence chain. No live change.


## 2026-11-13 15:10 UTC: thirteen full scenario-days verified

Oct 26/age500 completed at **2026-11-13T15:09:14.365710+00:00**. The
[run receipt](../reviews/model_recovery_20261113/run_age500_20261026.json), SHA
`4a73223a75789ca46563331db9e1294d323b757594cf46ad9c42060e760b131b`,
covers four policies × 288 windows and 410 reconstructed inference rows.
The [artifact review](../reviews/model_recovery_20261113/run_age500_20261026_artifact_review.json)
binds all 25 artifacts and three receipt copies with the existing positive
and 11 refusal controls. Its as-of is **15:09:44.660504 UTC**, SHA
`e8f25bfa9d0fb29e293bdc45822cfe70c3e78ed76bf2c81892d221341a87ca25`.
These are integrity counts, not an improved-policy verdict.

The frozen recovery chain is **13/18** full scenario-days. Oct 27/age500
worker **3143408** is active under driver **3159222**. Five age500 dates and
the final report/reconstruction remain. Size driver **1576587** waits for
complete recovery evidence, and depth driver **2900451** waits for complete
size evidence. Their 169 and 26 stages have not started. Recovery's 302,
size's 385, fast adapter's six and depth's 416 source identities still match.

Separately, the [SIDE-MOVES review](20261113_side_moves_model_integration_review.md)
separates observed refills from inferred cancellations/priority, and the
[MOVE-DOWN review](20261113_move_down_model_integration_review.md) reconstructs
all its published numerical readings while distinguishing request time from
effective order ending. No new candidate was selected from those results,
and no live system was changed.


## 2026-11-13 15:16 UTC: verified wait; recovery optimization remains unadopted

The previous goal turn made progress: the thirteenth replay and the two maker-
report reviews were committed. This continuation is a verified wait. At
**15:15:21.331026 UTC**, worker **3143408** is running Oct 27/age500 with
36,278 user CPU ticks, under live driver **3159222**. Thirteen run receipts
are complete; the final recovery report/evidence do not yet exist. The live
size/depth driver identities are unchanged, and neither has started its
historical runs/fits. All four source freezes still match.

The existing exact-container feature optimization was reconsidered by reading
its implementation, the recovery call site and the prior benchmark artifacts.
The earlier eight-window, five-policy parity receipt is PASS as of
**2026-11-12T14:57:25.349426+00:00**, with 22 exact artifacts; SHA
`7c2b58cd03ae88f2fc7597ccfa23c7cbddd87f23af5cbf8a3e263eb3e8ee4e95`.
Its optimized smoke log SHA is
`4980c18f1f3cc3e0edad1346b0cbf9784a2011ba2b5f7d3e0dac2d92eceb98e6`.
That run took 132.11 seconds versus the earlier original smoke's 138.61,
as already recorded in the execution handoff. These are separate small
runs, not a paired full-recovery benchmark. A large component speedup cannot
be transferred to the whole replay. Recovery-specific adoption would also
need new parity/provenance and a safe driver handoff, while the size waiter
explicitly recognizes the original recovery drivers.

No new recovery optimization or parity replay is justified by this evidence;
leave the active recovery chain intact. Do not repeat this investigation
merely because the worker remains busy. The already frozen size optimization
stays in place. Wait for the next completed recovery artifact, then review it
with the existing stage checker. No candidate, target, model or live change.


## 2026-11-13 15:44 UTC: fourteen full scenario-days verified

Oct 27/age500 completed at **2026-11-13T15:43:16.084858+00:00**. The
[run receipt](../reviews/model_recovery_20261113/run_age500_20261027.json), SHA
`366da25be6b7950ccbcebcbed50c86b39fa32e1d3d57a5ca5f5dbb3969b52a8e`,
covers four policies × 288 windows and 374 reconstructed inference rows.
The [artifact review](../reviews/model_recovery_20261113/run_age500_20261027_artifact_review.json)
binds all 25 artifacts and three copies, with the existing positive and 11
refusal controls. Its as-of is **2026-11-13T15:44:12.197950+00:00**, SHA
`e6d292b706834664c5111204ca80677276bb4f4de23c3b0d7c24533254f4339e`.
These are integrity counts, not an improved-policy verdict.

The chain is now **14/18** full scenario-days. Oct 31/age500 worker
**3455831** runs under driver **3159222**. Four age500 dates plus final
report/reconstruction remain. Size driver **1576587** and depth driver
**2900451** wait for their whole predecessor chains. All four scientific/
computational source freezes still match; no live change.

The [completed primary/stress review](20261113_recovery_primary_stress_review.md)
now provides actual fixed portfolio comparisons on both complete scenarios,
as of 15:39:36.887969 UTC. SELECTIVE10 loses marked value to the chronological
guard and DIRECT10 while increasing inventory-square risk. It pulls less than
the previous HISTORY10 but has less quote presence. All covered value/risk,
uncertainty and participation gates fail. This is a source-bound partial view,
not a replacement final report; the remaining task sequence is unchanged.
The [goal evidence index](20261113_model_goal_evidence_index.md) keeps these
negative results and outstanding completion requirements together.


## 2026-11-13 16:15 UTC: fifteen full scenario-days verified

Oct 31/age500 completed at **2026-11-13T16:15:04.663395+00:00**. The
[run receipt](../reviews/model_recovery_20261113/run_age500_20261031.json), SHA
`192b683ca378df25cfae9c6dc8c99a31a72272cb0539865eb16d88b0dfc1da88`,
covers four policies × 288 windows and 219 runner-reconstructed inference
samples. The [artifact review](../reviews/model_recovery_20261113/run_age500_20261031_artifact_review.json)
at **2026-11-13T16:15:55.857203+00:00**, SHA
`cf7829757e1b57c5991ff71e880a78c0e7f4ea9e90bd10965210cf2856c2648c`,
checks 25 artifact identities, three receipt copies, accounting provenance
and the existing positive/11 refusal controls.

Recovery is now **15/18** full runs. The driver started Nov 1/age500 at
16:15:04.977527 UTC; worker **3739610** is live under **3159222**. Three
age500 dates and the final portfolio report/reconstruction remain. Size
**1576587** and depth **2900451** are live and waiting. Source hashes still
match all four freezes (302/385/6/416 files). Memory and disk remain above
the wrapper requirements. No workers were restarted and no frozen source
or live settings changed.

The [absence overlap](20261113_recovery_absence_overlap.md) and
[corrected maker-study interpretation](20261113_robust_v2_model_integration_review.md)
are now preserved alongside the negative primary/stress portfolio result.
They clarify the queue/re-entry problem but do not establish a better model.
The size driver explicitly waits for final integrity, not a successful
recovery objective, so verified negative recovery results can release the
unchanged size comparison. These checkpoint counts are not performance claims.


## 2026-11-13 16:49 UTC: sixteen full scenario-days verified

Nov 1/age500 completed at **2026-11-13T16:48:56.149943+00:00**. The
[run receipt](../reviews/model_recovery_20261113/run_age500_20261101.json), SHA
`824febf978806e0817b8ac6c5ded99d32c801218fc470ccb6803b3371dd3ba53`,
covers four policies × 288 windows and 387 runner-reconstructed inference
samples. The [artifact review](../reviews/model_recovery_20261113/run_age500_20261101_artifact_review.json)
at **2026-11-13T16:49:45.564260+00:00**, SHA
`0b56556195eb49780011681dd1601238028dd2fc5f9e9456ef85f3901c0afd96`,
checks all 25 named artifacts, three receipt copies, accounting provenance
and the existing positive/11 refusal controls.

Recovery is now **16/18** full runs. Nov 2/age500 started at
16:48:56.468972 UTC; worker **4049823** is live under driver **3159222**.
Two age500 dates and the final portfolio report/reconstruction remain.
Size **1576587** and depth **2900451** remain live and waiting. All four
source freezes match (302/385/6/416 files); memory/disk remain above the
wrapper requirements. Existing negative primary/stress performance and
queue/re-entry diagnostics are unchanged. This checkpoint establishes
artifact integrity, not improvement or completion of the model goal.


## 2026-11-13 17:23 UTC: seventeen full scenario-days verified

Nov 2/age500 completed at **2026-11-13T17:23:17.711665+00:00**. The
[run receipt](../reviews/model_recovery_20261113/run_age500_20261102.json), SHA
`4aadc2e571dd6788d4f28036076a746e03c071d46e04316175882843bd00a118`,
covers four policies × 288 windows and 375 runner-reconstructed inference
samples. The [artifact review](../reviews/model_recovery_20261113/run_age500_20261102_artifact_review.json)
at **2026-11-13T17:23:56.660912+00:00**, SHA
`a5f0721f94d5bc15d51d8dcd2dbf09b83d9dfe6abccfaca0662de6fdc2a7dcbd`,
checks all 25 named artifacts, three receipt copies, accounting provenance
and the existing positive/11 refusal controls.

Recovery is now **17/18** full runs. Nov 3/age500 started at
17:23:18.001794 UTC; worker **166333** is live under driver **3159222**.
One age500 date and the final portfolio report/reconstruction remain.
Size **1576587** and depth **2900451** remain live and waiting. All four
source freezes match (302/385/6/416 files), and resource checks remain above
the wrapper requirements. Existing negative primary/stress performance is
unchanged. Artifact integrity does not establish an improved model; the goal
remains active through the pending size/depth evidence and its assessment.


## 2026-11-13 17:57 UTC: all eighteen full scenario-days verified

Nov 3/age500 completed at **2026-11-13T17:56:43.098402+00:00**. The
[run receipt](../reviews/model_recovery_20261113/run_age500_20261103.json), SHA
`942d2bf384bd06b77df5355f02f9ad57731b9ee318b8348ae935b4a983fa818e`,
covers four policies × 288 windows and 311 runner-reconstructed inference
samples. The [artifact review](../reviews/model_recovery_20261113/run_age500_20261103_artifact_review.json)
at **2026-11-13T17:57:20.071955+00:00**, SHA
`05d16acfba148118ca9a919a2c68633d33bed70de086cbe372d4b993bb575271`,
checks all 25 named artifacts, three receipt copies, accounting provenance
and the existing positive/11 refusal controls.

Recovery is now **18/18** full runs. The complete portfolio report is running
as worker **466916** under driver **3159222**, followed by final evidence
reconstruction. Size driver **1576587** and depth driver **2900451** remain
live and waiting for full predecessor integrity. All four source freezes
match (302/385/6/416 files). The existing negative primary/stress results
remain; completed runs and artifact integrity do not establish improvement.
The goal remains active through final recovery assessment and the pending
size/depth experiments.


## 2026-11-13 18:18 UTC: full recovery evidence complete; size experiment released

The [full portfolio report](../reviews/model_recovery_20261113/portfolio_result.json)
as of **18:01:17.599349 UTC**, SHA
`24c84a764aa46c8d0d07fc380752689fc33e1f45747b87d27423bc78b2bd0d35`,
has all improvement gates false. SELECTIVE10 loses marked value to guard and
DIRECT10 with higher inventory-square exposure in all three scenarios, and
has less presence than old HISTORY10 despite fewer pulls. The completed
age500 scenario does not reverse the earlier negative result.

The [final verifier](../reviews/model_recovery_20261113/evidence_check.json)
passed at **2026-11-13T18:18:37.057835+00:00**, SHA
`3bd21750832f5e65efbcf52b7d3ec94ab5b3db3e8493440ea0b80c25dcc8c6a7`.
Its 876 named artifact hashes, three copies, report link and accounting
provenance match. Coverage is 13 build records and 22 replay records
(four smokes plus 18 full runs), with 7,113 saved inference samples
reconstructed. Integrity PASS is separate from `objective_pass=false`.
See the [complete review](20261113_recovery_complete_review.md).

Recovery driver 3159222 exited normally. Size driver **1576587** verified the
exact final predecessor receipt and started its frozen sequence at
**18:18:56.581145 UTC**. Its q5/q50 default runs and captures
passed on four-window smokes by 18:22:10 UTC; the first full q5 reference
run is active (worker 691425). Depth driver
**2900451** remains queued behind full size evidence. The research goal is
still active: neither this negative result nor the small size smokes prove
an improved prediction model.
