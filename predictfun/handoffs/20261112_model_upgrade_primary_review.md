# Model upgrade — primary and stress decision review

Completion update, 2026-11-13T02:37:37.072135+00:00: [the full three-scenario
review is complete](20261113_model_upgrade_complete_review.md), with all eighteen
runs, corrected reports, phase presence and final integrity passed. Age500 also
fails the declared improvement test. The negative primary/stress result below
is unchanged. The next frozen recovery experiment has started its full seed build.

As of 2026-11-12T22:43:21.730005+00:00. The corrected portfolio report is as of 2026-11-12T22:40:51.901377+00:00. Six consumed complete dates (Oct 26, Oct 27, Oct 31, Nov 1, Nov 2 and Nov 3), 1,728 windows per policy/scenario. Twelve of 18 full scenario-day runs are complete; the six age500 paths and final evidence review are pending.

The frozen HISTORY10 candidate fails the preset improvement test. It loses more marked value than the chronological guard and carries more inventory-square exposure under both execution assumptions. Its small point gain over STATIC10 is inconclusive and also requires more inventory risk. The result does not support promotion or a claim that a better quoting model has been built.

| Scenario | HISTORY10 compared with | Common / missing marked windows | Marked value difference, USD [95% day interval] | Inventory-square exposure change |
|---|---|---:|---:|---:|
| X | causal_guard | 1517 / 211 | -645.80 [-767.61, -517.42] | +84.45% |
| X | static10 | 1537 / 191 | +28.08 [-59.10, +117.42] | +18.19% |
| stress | causal_guard | 1538 / 190 | -1153.97 [-1292.14, -1016.82] | +30.65% |
| stress | static10 | 1572 / 156 | +17.27 [-104.93, +161.03] | +11.14% |

Marked value is cash plus terminal inventory at a strict book mark. Each difference uses the same windows for its two policies; missing counts remain explicit. Inventory-square exposure uses all 1,728 windows, with identical five-share order size and ten-share reserved-inventory limits. These are matched constraints, not equal realized risk. The intervals resample the six UTC days (1,000 draws), not fills or actions. Policy-dependent missing marks and the small, consumed date population remain limitations.

The negative guard comparison appears on every one of the six dates in each scenario. This is stronger evidence against this particular candidate than the local action comparison, whose continuations overlap and cannot be added into portfolio P&L. The earlier local advantage over STATIC10 does not establish an advantage for repeated live decisions.

| Scenario | HISTORY10 fills | Model-triggered pulls | Median positive accepted lifetime | Whole-window side-time presence |
|---|---:|---:|---:|---:|
| X | 13,168 | 184,052 | 132 ms | 5.20% |
| stress | 19,302 | 157,581 | 220 ms | 6.01% |

The candidate still cancels frequently. Whole-window side-time presence uses two sides over 240.1 seconds; it is not the fraction of time with any quote, and is not a 300-second live-round uptime measurement. New risk is allowed only during the first 60 seconds, followed by a reduce-only phase. Phase-specific per-day audits are in the local-results review; the final three-scenario presence aggregate remains pending. A slower cancellation can increase presence without retaining a useful order.

The declared per-fill audit is complete: it reconciles each fill to its actual generation, quantity, settlement and strict one-/five-second marks. Those descriptive conditional outcomes do not identify which fills a different policy would have avoided. The generation-based one-/five-second value fields and per-fill totals also have different missing-mark rules, so they must not be silently interchanged.

Interpretation: added temporal features have not fixed value calibration or the repeated-decision policy. The local kept region had negative observed advantage despite positive forecasts; full replay now shows that extra retained exposure is costly. These findings do not identify a single cause. The ten-second continuation target, short decision/cancellation times, timer re-entry and distribution of states visited by repeated decisions are separate issues to test. Increasing model complexity or widening the quote ladder has no demonstrated benefit here.

The next already-planned target extension is an explicit short-horizon avoidable-value estimate and WAIT/recovery alternatives, with generation-linked fills, effective cancel timing and full-engine paired reconciliation. It must distinguish damage that a cancellation can still prevent from value lost by surrendering queue position. Recovery must be part of the tested action path. Current one-second timer re-entry does not solve that problem. This extension has not been implemented or validated by the present result; no new threshold or model was selected after reading these dates.

The remaining age500 runs test dependence on fills credited before 500 ms of order age, with subsequent inventory and decisions evolving in the engine. This is a sensitivity assumption, not an observed venue rule or a P&L lower bound. Completing it cannot reverse the failed primary/stress promotion gate. Live queue/fill/cancel calibration and at least five later untouched complete dates remain necessary before a forward-validation claim. The one-layer JOIN experiment is not a performance comparison with top makers or a multilayer strategy.

The settlement audit independently reconciles all 60 day/scenario/policy populations and applies the declared 0.5 payout to equal-price rounds. Only settlement/carry fields are superseded; marked values, exposure, paths and predicates are unchanged. The generated raw sequential-results file retains the original registry settlement fields for provenance. Use the corrected receipt below for settlement statements.

Sources: [corrected portfolio report](../reviews/model_upgrade_eval_20261112_v2/result_settlement_v3.json), [raw portfolio report](../reviews/model_upgrade_eval_20261112_v2/result.json), [per-fill reconciliation](../reviews/model_upgrade_eval_20261112_v2/fill_value_detail.json), [early execution of the declared settlement audit](../reviews/model_upgrade_20261112_v2/settlement_primary_early_execution.json), [local and phase-presence review](20261112_model_upgrade_local_results.md). No live changes.

Related-work source review, 2026-11-12T22:46:20.744498+00:00: the separate MOVE study is relevant motivation for a short danger target, but is not a ready-made replacement score for this experiment. Its `pf_move_info.py::BASE_N` includes the inherited learned `v`; `lgb_oof` trains on every other declared day, including later days for earlier tests. `pf_move_avoid.py::report` fits and assesses the avoidable subset conditional on the level still existing at a future effective-cancel time; `ops` chooses recall operating points from its reported outcomes. These are exploratory public-level diagnostics, not a decision-time eligibility rule or a chronological own-order value policy. Do not import their model, operating threshold or claimed alarm-rate reduction as a validated quoting improvement. A future experiment must preserve all neutral action statuses, use causal contract context and earlier-day fits, and price cancellable own fills plus queue/re-entry loss in the engine. This source review changes no current result, policy, candidate or protected-data boundary.


Descriptive timing addendum, 2026-11-13T01:58:57.077683+00:00: [the committed age accounting](20261113_model_cancel_timing_review.md) separates fills before the earliest possible learned cancellation effect. HISTORY10's first two age bands have negative net fill-mark value in both completed scenarios. The comparison also includes an entry-policy difference: the chronological guard can withhold before placement, while the fitted value policies first score at 100 ms. This does not estimate counterfactual avoidability or select a new policy; full population and missing counts remain in the linked receipt.


Source-comparison addendum, 2026-11-13T02:15:29.597599+00:00: [the callback review](20261113_model_guard_comparison_review.md) confirms that the chronological guard applies deterministic trailing Binance movement rules and its learned score before the value policies' 100 ms accepted-order gate. The complete-policy benchmark remains valid, while prediction quality, admission and immediate responses are not isolated by that comparison. STATIC10/HISTORY10 share the scoring schedule. No policy or result changes.
