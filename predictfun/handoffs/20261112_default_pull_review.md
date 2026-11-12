# Default quoting with learned pulling — completed review

**Result: no candidate qualifies for promotion.** All six primary and six stress replay days finished, followed by the independent numerical audit and figure generation at 2026-11-12 05:43:57 UTC. The final result population is as of **2026-11-12 05:42:48 UTC**: six consumed test dates (20261026, 20261027, 20261031, 20261101–03), 1,728 windows per policy and scenario. These are exploratory historical comparisons, not untouched validation or calibrated live forecasts.

The experiment answers whether to retain an existing order in the new single-layer, up-to-5-share architecture. It does not optimize order size or a multilayer ladder. The matched guard has the same size, placement and reserved-inventory rules. The frozen ladder v3 remains a separate operational benchmark. All new arms reserve pending creates and cancellations against ±10 shares; the frozen reference has different sizing/ladder/reservation behavior, so its much larger inventory is not an equivalent risk control.

## What the completed comparison says

Against matched guard, each learned policy loses marked trading value on all six days under both execution scenarios. The simple pressure rule's favorable primary point estimate reverses under slower execution. Neither it nor the guard floor meets the declared requirement of more marked trading value with no increase in inventory-square time against both baselines under both scenarios.

| Policy vs matched guard | Primary common marks / missing | Primary paired marked change $ [day interval] | Primary inventory-area change | Stress common marks / missing | Stress paired marked change $ [day interval] | Stress inventory-area change |
|---|---:|---:|---:|---:|---:|---:|
| Pressure rules | 1,557 / 171 | +29.98 [−38.66, +120.36] | −12.381% | 1,514 / 214 | −105.76 [−265.99, +17.74] | +16.869% |
| Guard floor | 1,527 / 201 | −289.21 [−374.73, −188.79] | +117.940% | 1,530 / 198 | −663.57 [−803.49, −528.40] | +63.819% |
| Market + age model | 1,534 / 194 | −559.35 [−673.79, −451.66] | +74.271% | 1,517 / 211 | −1,110.16 [−1,260.76, −968.45] | +23.439% |
| Queue model | 1,508 / 220 | −592.27 [−694.89, −511.54] | +80.126% | 1,528 / 200 | −1,100.40 [−1,237.24, −952.70] | +26.321% |
| Queue model + pull margin | 1,516 / 212 | −563.77 [−657.41, −478.00] | +85.635% | 1,522 / 206 | −1,095.40 [−1,236.55, −967.01] | +28.115% |

Every row has 1,728 paired windows. Marked comparisons use the stated common-mark subset; intervals resample six UTC days 1,000 times. They do not correct for selection on consumed history. Pressure has three positive and three negative paired days in each scenario, and both of its intervals include zero. The audit independently reconstructs every comparison's marked total and common-mark counts by day.

Against frozen ladder v3, pressure loses $153.08 primary (1,533 common marks; 195 missing) and $122.70 stress (1,488 common marks; 240 missing), despite reducing inventory-area by 94.119% and 92.643%. All three learned policies also lose marked value against that reference in both scenarios. The exposure reduction largely reflects the different architecture and reservation rules; it does not establish that the learned cancellation policy preserves the ladder's trading gains.

Settlement is a separate measure, reported on all 1,728 windows per arm/scenario:

| Policy | Primary settlement $ | Stress settlement $ |
|---|---:|---:|
| Frozen ladder reference | +324.30 | −220.66 |
| Matched guard | +42.11 | −83.95 |
| Default without signal pull | −2,643.22 | −3,010.32 |
| Pressure rules | +60.10 | −331.37 |
| Guard floor | −121.77 | −796.12 |
| Market + age model | −686.00 | −1,445.74 |
| Queue model | −735.88 | −1,456.15 |
| Queue model + pull margin | −730.01 | −1,447.95 |

Rebates are excluded. Marked value plus carry equals settlement only on the same valid-mark population; it must not be reconciled to the all-window settlement total without the missing-mark windows. Full spread, markout, carry, lifecycle, inventory and status tables are in [the final report](20261111_default_pull_final.md) and its canonical JSON.

## What this teaches about the model

The direct regression estimates KEEP minus CANCEL+RESET value. Local fills over roughly two seconds are valued using the average of their 1s and 5s markouts, less an inventory-square penalty. RESET waits a fixed one second plus placement latency and loses queue priority. The one-second cooldown is a policy assumption; the model does not predict that danger clears after one second. Partial and zero fills are included, and quantity is valued per fill rather than assuming every posted share fills.

There is local predictive signal, but it does not translate into a good sequential policy. On 40,059 eligible neutral test actions, the observed local pull-benefit totals are 34,013.60 / 34,585.12 / 34,052.46 cents for market / queue / queue-with-margin, versus matched-null means of 29,493.19 / 30,004.03 / 29,434.23 cents. Each exceeds all 500 matched-count shuffles; the conditional permutation p-value is 1/501. This is a local, overlapping action diagnostic, not realizable portfolio P&L or independent-day validation. It also does not establish an incremental queue-feature advantage. Of those actions, 29,224 have no fills in either local branch.

The actual learned policies cancel most new generations: primary model pulls are 199,146 / 221,363 (market), 199,713 / 222,617 (queue), and 198,493 / 221,524 (queue-with-margin). Their median simulated order lifetime is 132 ms in primary and 220 ms in stress. This does not reproduce sustained queue residency. The policy can re-enter when its timer expires without requiring the hazard to clear. Pressure instead evaluates its trigger before new entry and keeps extending its cooldown while pressure remains. These mechanisms and the different neutral-versus-policy decision populations are plausible explanations to isolate, not proven causal attributions of the losses. Training RESET also uses the same price, while the sequential policy can re-enter at the then-current touch; future portfolio reactions are only approximated by the local label.

More instantaneous spread is not enough. Default quoting records +$687.66 primary spread but −$2,488.55 / −$2,593.34 strict 1s / 5s markout; stress gives +$748.51 spread and −$2,832.33 / −$3,031.60 markout. Primary spread-known/missing generation counts are 90,279/4 and both-horizon markout-known/missing 90,250/33; stress counts are 86,761/4 and 86,731/34. Unfilled generations remain represented. Spread and markout are alternative valuations and are never added to settlement P&L.

Lower ending inventory also does not guarantee the desired trade-off. The queue model ends with mean absolute inventory 0.959 shares primary and 0.572 stress, versus matched guard's 1.524 and 1.825, across 1,728 windows each; nevertheless it increases inventory-square time and loses substantial trading value. The declared area and marked valuation end at 240.1 seconds, after quoting stops at 240 seconds. Residual inventory is carried toward settlement. Area and terminal exposure are risk proxies; six consumed days do not establish future P&L variance.

Use the audit's union of live intervals for resting presence, not the report's summed order lifetimes: overlapping cancellations can double-count the latter. For example, the primary frozen reference's union presence is 2.279% of side-time versus 3.649% by summed order-time. The queue model's union presence is 3.947% primary and 5.127% stress (six days / 1,728 windows each). Neither is calibrated live venue uptime. Public maker fills alone cannot reveal their cancellation or quoting policy.

## Limits and next experiments

Primary execution uses a −50 ms trade anchor, 40/22/100 ms place/cancel/notice delays and queue rule M. Stress jointly changes the anchor to −250 ms, delays to 80/100/250 ms and queue rule L. This sensitivity does not separate which timing or queue assumption causes the result. Live fill/cancel clocks and queue behavior remain uncalibrated. The new model folds are chronological, but inherited guard predictions are LODO; the whole strategy is not a historically deployable OOS backtest. No protected raw dates were opened by this experiment.

The next research question should isolate re-entry: keep the five-share single-layer setup fixed and compare the existing timer with a predeclared rule for waiting until conditions recover. Its label must reflect the same re-entry rule, price and latency as its replay. A separate experiment can then compare feasible quantities or ladder configurations using action-specific fill/value estimates. Multiplying this model's score to choose larger sizes is unsupported. These are proposed experiments only; no extra variants were launched or tuned after this result.

For credible live ranking, the outstanding data requirement is a completed, identified run with inputs, outputs, wallet events, decision reasons, strict kernel journal export, clock diagnostics and the matching public book/trade tape, including unfilled and cancelled orders. See [the calibration capture specification](20261110_fill_calibration_request.md). Exact event identities and timestamp semantics must be reconciled before fitting fill probabilities or delays; unavailable execution time stays an interval/status. A scope decision is required before consuming any protected capture. After calibration and a new pipeline freeze, validation needs at least five later untouched complete UTC days. This negative consumed-data experiment is complete without claiming those future steps are complete.

## Completion evidence

| Required outcome | Inspected evidence and disposition |
|---|---|
| Preserve the completed own-order/calm study | Original canonical comparison and human review retained byte-for-byte against `d475eccb`; no prior candidate relabeled successful. Final integrity script rechecks both worktree and shared-root copies. |
| Quote by default and learn pulls on actual orders | Frozen hook/shadow/fit sources; all eight actual portfolio arms in both scenarios; saved generation scores, model decisions and effective cancellation times reconstructed by the numerical audit. |
| Account for inventory, delay, queue and reset costs | Effective-cancel snapshots, notified-prefix sizing, queue-loss/reset branch, inventory term and reservation fix inspected. Synthetic causal/quantity/prefix controls passed; approximations stated above. |
| Cover the authorized population | Nine neutral build days, six primary and six stress days, 288 windows per arm/day; smoke plus these runs total 22. Missing outcomes retained as statuses; paired marks checked by day. |
| Freeze training and avoid result-driven tuning | Source/declaration `36093438` / `531836a5`; six expanding folds, two pinned models each; 12 saved models and predictions verified. No changes to the 232 frozen numerical/research sources. |
| Compare complete portfolios | Sixteen policy/scenario aggregates, 42 paired comparisons, strict spread/markout and carry tables, terminal inventory, lifecycle and 96 day-level union-presence records inspected. |
| Quantify exploratory evidence correctly | Predeclared 500 matched-count shuffles and 1,000 day-cluster draws; counts, missingness, consumed-history and LODO limitations retained. Computed promotion criteria all fail. |
| Validate actual output | Numerical audit PASS: 22 runs, 1,189,652 feature identities, 1,127,825 saved policy-score reconstructions and 506 input hashes. Final closeout rehashes saved run artifacts, models, source and all 28 required stage logs. |
| Review presentation and scope | PNG inspected against the report; both PNG/PDF identities verified. Clustered point labels can overlap, so exact populations are in the tables/figure receipt. Diff restricted to research and documentation; no live strategy, loss-budget or rebate changes. |
| Track and publish conclusion | This review, canonical final report, chart, audit receipts and final status are saved in both worktree/shared root and committed/pushed on `research/predict-default-pull-20261111`. Execution-only change to two global research slots is recorded in `execution_two_slots_20261112.json`; the frozen strategy and completed results are unchanged. |

Canonical result: `reviews/default_pull_20261111_v4/comparison_final.json`, SHA-256 `e84214cd58030c8b510aaf3be4c6b9167a86e01778e8aa5a8f8dea9728649fba`. Independent audit: `completion_audit.json` and `completion_runtime_v4.json`. Final integrity receipt: `final_review_integrity.json`. The conclusion is a negative result for this tested architecture, not a rejection of all queue-aware or quote-default market making.
