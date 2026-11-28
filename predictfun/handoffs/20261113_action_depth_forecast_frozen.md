# Frozen action-depth forecast experiment

As of **2026-11-13T14:42:09.952078+00:00**. This is a prediction experiment
within the active economic-policy goal; it does not replace the required
sequential queue-value, execution-stress and inventory-risk evaluation.

The [declaration](../reviews/model_action_depth_forecast_20261113/declaration.json)
freezes commit **0e246af6b30230d16e410063ce374efbca574503**, **416 sources**
and **26 stages**, at **2026-11-13T14:41:54.521736+00:00**. Declaration SHA:
`b55263801d6bbed42a43e7e3cc6f17563146ca7fdd5af965580fb0afb17a35f1`.
No historical depth target, fit or forecast result exists yet.

## Population and targets

Reuse the frozen size experiment's original neutral captures and its separate
default reference, for five and 50 shares under cap50. The dates are the same
nine consumed UTC dates: Oct 23–27/31 and Nov 1–3, 2026. Every original
generation/status remains represented; at most one sampled action per
generation is fitted. The inherited assigned-age/partial design remains
unchanged. No completed-fill selection or newly simulated counterfactual is
introduced.

Before attaching depth, authenticate the original size run, fast-adapter
identity, all native artifacts, independent neutral parity, corrected causal
grid, source takers, settlement, exposure union and inventory accounting.
Reconstruct source best prices and client-known clocks from the exact book
shards and protected-boundary filters used by that native replay. The
collector/parser itself is not independently audited.

The target is the maximum recorded adverse depth from the last client-known
best, measured at and after the **native effective cancellation time** for
1,000 ms. A recovered pre-intervention move is excluded. An adverse move
persisting across that deadline counts. Maximum source gap and known-book
age are each fixed at 1,000 ms, matching the inherited native freshness
limit; these limits do not prove that every exchange update was recorded.
Observed labels cannot cross UTC boundaries or their native capture horizon.
Unknown and censored targets remain statuses, never zero.

## Fixed model and evaluation

Two feature families—STATIC plus size and HISTORY plus size—fit five binary
indicator heads for depth reaching 1–5 ticks. All ten heads use the same
observed neutral actions, including those with missing economic marks. The
existing 300 rounds and all other pinned parameters are retained; binary
objective matches the indicator target. There is no parameter search.

Each of the six outer dates trains only on the earlier core, reserving the
latest preceding date for calibration predictions without refitting or
selection. Profile core means provide a baseline. The causal allowlist omits
future labels/deadlines and learned guard outputs. A fixed decreasing
isotonic projection reconciles the five heads; this is not a calibration
claim. Unsupported depths above five ticks return unknown.

The primary forecast comparison is **q50 HISTORY versus STATIC**, using
mean Brier loss across the five fixed heads and equal UTC-day weighting.
The q5 comparison, profile-mean baselines and fixed decile calibration are
descriptive. Each table retains total/observed counts and depth/economic
statuses by day. Intervals use 1,000 day-bootstrap draws and require at least
five complete dates.

The 500-draw alignment diagnostic permutes whole forecast vectors within
date/side/hour/order-age/inventory-sign/assignment/partial-state/quote-distance/
static-risk strata. It neither assumes serial exchangeability nor estimates
a cancel policy's economic null. The computed forecast screen is exploratory
on consumed dates and cannot promote a trading strategy. No new cancel
policy or policy threshold is included in this declaration.

## Verification and execution

The [final combined controls](../reviews/model_action_depth_20261113/experiment_controls_v2.json)
passed **76 tests** at **2026-11-13T14:41:07.570994+00:00**, SHA
`7324e39bcf455b669b284c7e442cfb99e1a5f7661eeb403d17f6b67ee95f05e4`.
All 25 control source identities, prerequisites, start/log evidence and three
receipt copies were checked. Peak RSS was 317 MiB; wall time 65.16 seconds.
Controls exercise actual native captures, the saved-capture conversion,
immutable accounting, chronological fits, saved predictions, nulls and
complete-stage orchestration. Planted signals pass; constant forecasts and
corrupted artifacts/chronology/statuses refuse.

The first saved-capture attempt preserved two fixture errors caused by moving
path-bound artifacts without updating their test dependency paths. The fixed
fixture passes. The first complete freeze then refused a missing scheduling
helper in its source manifest before any declaration or historical payload
read. That refusal is retained in
[experiment_declaration_refusal_v1.json](../reviews/model_action_depth_20261113/experiment_declaration_refusal_v1.json).
The final source closure includes the helper, and all 76 controls passed again.

The [waiting driver](../reviews/model_action_depth_forecast_20261113/driver_started.json)
is **2900451**. It waits for the entire size chain and its verified final
evidence, then runs 18 immutable target builds, six chronological fits, the
forecast report and full source/label/model/report reconstruction. Numerical
work uses core8 through the existing shared wrapper and resource limits.
Partial stages/logs are preserved; no automatic restart or overwrite.

Recovery is **12/18** full scenario-days verified; Oct 26/age500 worker
**2822379** is active under **3159222**. Size driver **1576587** waits, with
all 169 tasks pending. The recovery, size and fast-adapter freezes remain
unchanged. Larger orders' historical economic benefit remains unproven.
No live trading code or settings changed.
