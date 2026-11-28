# Future-best rule: model integration review

As of **2026-11-13 13:53 UTC**, the other owner's FUTURE-BEST run is still
evaluating its nine consumed dates. This review binds producer commit
**037ed9de8e21ed5ba8858cff552f54d22b720bfa** and the unchanged inputs from the
earlier [PULL-DEPTH review](20261113_pull_depth_model_integration_review.md).
It does not recompute or promote a historical performance result.

The [source review receipt](../reviews/model_recovery_20261113/future_best_integration_source_review.json),
as of **2026-11-13T13:53:11.071900+00:00**, has SHA
`4195cf5531379f38d7040723d82be9bf93c18ab27724eee14015e034996a1385`.
Instrument **48131569** verifies the two producer files at their exact commit,
43 inherited source/input artifacts, six synthetic decision cases and three
planted rule errors. It also exercises the exact report verdict loop on small
synthetic cells. No historical prediction, fit or portfolio was recomputed.

## What the change tests

For an order `d` ticks behind the currently known best, PULL-DEPTH withdraws
when the probability of a move of at least `d` ticks exceeds its threshold.
FUTURE-BEST generally uses `d + 1`: reaching the order's price may make it the
new best; moving beyond that price is a different event. This is a useful
geometric distinction for retaining an already-resting order.

The arrival exception applies to orders originally posted on a deeper rung.
An original rung-zero order at the best still pulls when its guard is on.
New entries remain blocked by that guard. Consequently this is a conditional
withdrawal-rule comparison inside the existing guard ladder; it is not a full
default-quoting replacement or a new fitted prediction model.

The exact source decision function passes bid/ask landing-price cases,
deeper-rung arrival, original-rung-zero behavior, guard-off and oracle cases.
Removing the one-tick shift, extending arrival retention to rung zero, or
reversing the oracle sign each breaks a positive case.

## The inherited model gaps remain

The producer loads the same PULL-DEPTH prediction, target, event and reference
fill artifacts. The source review verifies their identities against the prior
audit. It neither retrains the depth model nor repairs its training dependencies.
The earlier findings therefore still apply: the known-frame target can include
a recovered pre-intervention move, missing future observations can become zero,
and the upstream learned-feature chain can expose an outer test day's labels
to downstream training. Leave-one-day-out evaluation also uses later dates.
No new affected-row frequency or historical effect size is estimated here.

The six rungs are still simulated independently with separate trade budgets,
without the shared native inventory and reservation constraints. Selected
aged/WS fill marks are not a matched-risk portfolio result. The code counts
missing one-second values in the main fill table, but filters missing level-fill
values separately without an analogous count; its five-second marked totals
also do not require complete five-second marks. These limits remain relevant
even if row-level reference parity passes.

The OFB arm uses the realized recorded-book depth in the same fixed rule. That
is an oracle diagnostic for this proxy. It is not a proven upper bound on
monetary policy value: a price-depth oracle does not optimize fill selection,
inventory or continuation value, and a withdrawal may omit a later valuable
fill. Its difference from a forecast rule cannot by itself quantify the gain
available from a better economic model.

## Declared sample rule is missing from two verdicts

The producer header says a day with fewer than 30 fills in a compared cell does
not count. In the exact `report()` implementation, R500 and R501 count higher
or lower daily marked totals directly; that loop never references `MIN_FILLS`.
R502 applies the minimum explicitly.

The source control supplies one synthetic fill per arm/day and favorable
scalar cells on nine synthetic dates. The original R500/R501 loop emits
BETTER/HELPS despite every day being below the declared minimum. A corrected
in-memory control excludes those sparse days and emits NOT-SHOWN; it still
accepts a dense positive control with 30 fills per cell. No producer file was
edited. **Whether any historical verdict is affected remains unmeasured.**

## Implication for the active goal

Keep the distinction between reaching and passing a quote price when building
the next candidate. Require depth forecasts from action-deadline targets,
earlier-only training throughout the dependency chain, and a native joint
budget before treating that geometry as an economic improvement. Recorded
price depth remains distinct from the probability or monetary cost of filling
our specific order.

The separate [action-depth pipeline](20261113_action_depth_target_preparation.md)
now has a composed native source/capture/training adapter at **a50aa572**.
Its **46-test** combined suite is queued under **2422704**, behind the complete
FUTURE-BEST owner chain **2364799** on core9. The prior 38 component tests pass;
the newly composed path has not run yet. Recovery remains 10/18 complete with
Nov 2/stress worker 2177940 active; size driver 1576587 waits for final recovery
evidence. All frozen recovery/size/computational sources remain unchanged.


## Completed aggregate audit, 2026-11-13 14:08 UTC

The producer report is dated **2026-11-13T13:56:41Z**. The [independent
arithmetic/population receipt](../reviews/model_recovery_20261113/future_best_report_audit_v2.json)
passed at **2026-11-13T14:06:29.638543+00:00**, SHA
`3b45fe42933f1431d956e9e3f71784ffb9fbe1779ef3f54e832998ec11d233e6`.
It binds 30 inputs over the same nine consumed dates: Oct 23–27/31 and Nov 1–3,
2026. All 1,383,411 variant fill records were counted; these reuse trades
across variants and are not independent observations. The smallest aged WS
arm/clock/day cell has 657 fills. No compared cell falls below 30, so applying
the omitted minimum changes **none** of R500/R501's historical verdicts.
One-second and five-second main-fill missing counts and one-second level-fill
missing counts are all zero on these artifacts.

The initial audit refused changed producer bytes before reading historical
payloads. The producer had added post-hoc diagnostics. Version 2 binds the
original reviewed commit and verifies that all original definitions remain
unchanged (allowing only the added diagnostic functions, docstring and CLI).
The refused log is retained and linked by the superseding declaration. No
producer file or report was edited by this session.

For the report's M50 setting, aged one-second marked dollars per day are:

| Rule | WS records | WS $/day | Combined records | Combined $/day |
| --- | ---: | ---: | ---: | ---: |
| PD10 | 17,677 | +76.52 | 37,713 | −119.50 |
| FB10 | 18,287 | +94.09 | 40,052 | −132.40 |
| PD25 | 19,241 | +97.47 | 43,820 | −174.32 |
| FB25 | 25,544 | +124.34 | 57,648 | −270.78 |

These are selected-fill marked values over nine days, **not net joint-portfolio
P&L**. FB10 and FB25 improve the WS subset by +17.57 and +26.87 dollars/day
versus their depth counterparts, but their combined values worsen by −12.89
and −96.46. At M250, their WS incremental values reverse to −130.02 and
−280.69 dollars/day, lower on all nine dates. Thus retaining more protection
quotes is a useful geometry hypothesis, not established economic improvement.
The inherited chronology/target issues, independent rung budgets and live
fill calibration remain unresolved.

The composed action-depth adapter now passes all 46 synthetic controls; the
earlier queued state is superseded. Recovery is 11/18 complete, Nov 3/stress
worker 2500825 is active, and size driver 1576587 waits for final recovery
evidence. No depth historical model or live policy has been changed.
