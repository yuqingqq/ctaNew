# Action-deadline depth target preparation

As of 2026-11-13 12:55 UTC, a new target primitive and eight synthetic tests are
implemented at commit **2f6df830**. This prepares a later model comparison; it
does not change the frozen recovery or size experiments. No historical target
table, model fit, prediction result or policy benefit has been produced.

The motivation is the [PULL-DEPTH integration review](20261113_pull_depth_model_integration_review.md):
the producer's known-frame target can count a move that recovered before a new
cancellation could act, and can turn unavailable future observations into zero.
Its upstream learned-feature dependencies also require a chronological rebuild.

## Target definition

`pf_action_depth_target.BookDepth` receives source and client-receipt clocks,
best bid/ask arrays, and explicitly supplied freshness/gap limits. An `Action`
names one neutral generation, its side, decision time, effective intervention
time, horizon and last client-visible frame. The helper refuses an inconsistent
known-frame index, impossible interval, bad clocks or duplicate action identity.

Let the reference be the best price in that known frame. The target is the
largest adverse distance from the reference in the recorded book: the state at
the effective intervention time, followed by frames strictly after that time
through the horizon endpoint. The book is interpreted as holding its recorded
state between frames. The last state at an equal-stamped deadline is the anchor;
all equal-stamped frames inside the interval remain in the depth calculation.

Separate fields retain adverse depth already present at the deadline and the
additional excursion beyond that depth. A persistent adverse move therefore
counts; a move that has recovered before the deadline does not. A future price
change strictly beyond the endpoint cannot change the depth value, although a
following frame can provide the required coverage bracket.

This measures **recorded-book price depth**, not continuous-market extrema,
the probability of our order filling, or an avoidable monetary loss. In
particular, reaching a resting price, clearing that price and filling our
specific order are different events. The helper makes no cancellation decision
and establishes no ladder-policy threshold.

## Explicit observation statuses

A complete label requires a valid client-known frame, a valid state at the
deadline, valid intervening frames, a right endpoint bracket, gaps within the
declared limit, and at least one observed frame in the future interval. These
are sampling requirements; they do not prove that the collector saw every
exchange update. No coverage limit is selected or tuned from historical results.

| Status | Meaning |
| --- | --- |
| `observed` | The declared recorded-book coverage requirements pass. |
| `missing_known_book` | No valid client-known reference exists. |
| `stale_known_book` | The known frame exceeds the supplied receipt-age limit. |
| `missing_deadline_book` | The recorded state at intervention is invalid. |
| `right_censored` | Recorded coverage does not bracket the horizon endpoint. |
| `invalid_future_book` | The coverage interval contains an invalid book frame. |
| `book_gap` | A required adjacent-frame gap exceeds the declared limit. |
| `no_post_effect_observation` | No valid frame is observed strictly after the deadline through the endpoint. |

Unknown labels remain `None`, with available partial evidence retained as a
diagnostic lower bound. A genuinely observed flat path has a finite zero label.
The batch helper returns every supplied action, including unknown statuses;
it does not select a training population from fills or outcomes.

## Synthetic evidence and remaining integration

The [controls receipt](../reviews/model_action_depth_20261113/target_controls_v1.json)
binds the committed implementation, tests, command, start receipt and actual
success log. Receipt SHA:
`4b31f9e5ec94ac281a860806671fb0b0fd38a57a2cf8839c7f473b6de59db0b1`.
All eight tests pass as of **2026-11-13T12:55:40.046611+00:00**.
The free auxiliary-core wrapper returned zero, with
28 MiB peak RSS and 0.08 seconds wall time. No historical payload was read.

Controls cover recovered past moves, future moves on both sides, reversed-sign
and stale-interval falsifiers, persistent versus additional depth, missing and
partially observed futures, endpoint censoring, gaps, a quiet interval without
future observations, last-known-frame visibility, staleness, impossible clocks,
bad shapes, immutable copied inputs, equal clocks and duplicate action refusal.

The target primitive checks clock consistency; the native adapter below now
adds in-memory engine reconciliation. A complete historical experiment still
needs authenticated artifact extraction, persisted neutral generations/statuses,
raw source-book identity, causal feature extraction, and earlier-only training
throughout any learned-feature dependency chain. Deadline/future diagnostic
fields must never enter decision-time features.

A later comparison must commit the complete data-to-fit-to-evaluation pipeline,
declare candidate multiplicity and matched controls before historical fitting,
and test whether the depth forecast adds economic value beyond the existing
short-value and queue-retention heads. The current goal remains active; this
primitive is preparation, not evidence of a stronger model.

## Native integration, 2026-11-13 13:07 UTC

Commit **a0456b5c** adds `pf_action_depth_native.py`. It reads a finished native
neutral window, verifies its KEEP/RESET/PULL captures, and reconstructs the
paired economic target before accepting the cancellation clock. The PULL and
RESET records must agree on cancellation request and effective times for the
same action. The depth target uses the recorded effective time, with no fixed
latency substituted. Order identity, side, price, quantity, activation, prefix
fills and withdrawal protocol remain subject to the existing capture checks.

The adapter creates its book context from the actual window's source clocks,
client-known clocks and best bid/ask arrays. Its array fingerprints are included
in the output. Action day, market and window identity must match, and the depth
helper's last-known frame must equal the native window's lookup. Records after
the UTC day are clipped before endpoint coverage is assessed. A target exceeding
the UTC day or captured native horizon remains an explicit censored row.

Every neutral generation remains in the returned population. Observation and
fork IDs must match their ledger statuses exactly; only the preassigned sampled
actions receive depth labels. Empty native windows are retained. Unknown depth
and missing economic-value marks remain separate statuses. The collector does
not change the neutral output or stored forks, as checked by the native fixture.

The [combined controls receipt](../reviews/model_action_depth_20261113/native_controls_v1.json),
as of **2026-11-13T13:07:48.482668+00:00**, records **13 passing tests**: the
original eight target tests, three capture/identity tests, and two native-engine
tests. It binds all four implementation/test sources, the prior target controls,
the recovery/size declarations, and the actual success log. SHA:
`3a2f97bd27257adc101dfc056765ba951fca44d7a8d0b03d0d9f5d7b4f518831`.
The auxiliary-core wrapper exited zero, with 224 MiB peak RSS and 4.40 seconds
wall time. Synthetic engine clocks use their actual epoch date, 1970-01-01;
no historical day payload was accessed.

Controls reject a changed PULL deadline even when its copied target field is
updated, a broken paired-capture hash, a wrong market/date, invalid horizons,
incomplete neutral generation coverage and duplicate observations. Two
untouched age-assigned orders receive labels before any notified fill. The
empty native window stays visible. Native population, forks, portfolio-window
records and raw result values remain unchanged after collection.

This is an **in-memory integration**. A persisted historical depth dataset,
raw-tape/input provenance, causal depth-feature builder, chronological fits,
matched controls and sequential policy evaluation are still outstanding.
Recovery remains 9/18 complete, with Nov 1/stress worker 1867188 active; size
driver 1576587 waits for final recovery evidence. The frozen experiments have
not adopted this target.


## Persisted artifact controls, 2026-11-13 13:24 UTC

Commit **5c4d2c28** adds `pf_action_depth_artifacts.py`. For every externally
declared day/market window, the writer saves the native source/receipt clocks,
best prices, complete neutral-generation ledger, depth labels and linked
KEEP/RESET/PULL captures. Original arrays are retained so UTC clipping and
unknown-book states can be reproduced. An empty window is a required record.
The writer accepts only the declared profile/configuration and exact window set.

Verification reconstructs each label from the saved arrays and captures, joins
every neutral generation to the original native population Parquet, and checks
each capture's exact digest against the original native capture streams. It
also verifies the original native artifact set, expected windows/configuration,
and supplied external input hashes. Construction alone is not a passing check.

The [combined artifact controls](../reviews/model_action_depth_20261113/artifact_controls_v1.json)
pass **22 tests** as of **2026-11-13T13:24:07.993442+00:00**: the prior 13
controls plus nine persisted-artifact cases. The receipt binds six source/test
files, three prerequisite receipts, the start record and the actual log. SHA:
`6bd428b088e028aa73a89e68b229b4f3ce5a0e6a2d85bf274fd3b0de3b68a50f`.
The auxiliary wrapper exited zero with 240 MiB peak RSS, 8.40 seconds wall
and 8.11 seconds user CPU. No historical day payload was read.

The new controls accept actual saved native captures and an empty native
window. They reject a rehashed wrong label, omitted generations with consistent
recomputed counts, omitted windows, duplicate evidence, rehashed altered books,
impossible clocks, bad shapes, wrong external identities/configuration, changed
input bytes, duplicate appends and incomplete closes. A self-consistently
rebuilt capture with an added field passes internal label reconstruction but
fails the independent exact join to the original capture. Read-only collection
leaves the native ledger, forks, window accounting and raw result unchanged.

This library authenticates the **supplied** input set; it does not prove that
set is complete or independently reconstruct its book arrays from raw tape.
That provenance, the historical loader, causal feature extraction, chronological
fits, declared matched controls and sequential economic evaluation remain
outstanding. No depth feature or model has entered the frozen recovery/size
experiments. Recovery is now 10/18 complete; the size experiment remains queued.


## Causal training-table integration, 2026-11-13 13:30 UTC

Commit **03cf41a3** adds `pf_action_depth_training.py`. It joins saved depth
labels to the same sampled neutral actions used for native withdrawal targets
and causal size features. Identity, side, decision, profile, notified quantity,
cancellation clock and economic statuses must agree. The full action population
is retained, with separate depth, economic and joint completeness statuses.
Observed depth remains usable when an economic mark is missing; unavailable
depth remains unknown even when economic value is complete.

The model-matrix helper uses only the existing explicit size-feature allowlist.
Future depth extrema, cancellation-effective diagnostics and economic targets
cannot be requested as model inputs. Saved features are reconstructed from the
supplied corrected grid; attaching target columns never changes them. The saved
training table is reconstructed exactly against native, depth and feature
artifacts. Grid/input hashes are checked against actual files, while their
completeness and independent reconstruction from raw tape remain caller duties.

The [combined training controls](../reviews/model_action_depth_20261113/training_controls_v1.json)
pass **30 tests** as of **2026-11-13T13:30:53.887887+00:00**: the previous 22
cases plus eight join/feature-isolation cases. Receipt SHA:
`99585b420368ba4e2170c6e5079c37e5037f1b714a6c690f504f54c0afa1f9af`.
All eight implementation/test hashes, three prerequisite receipts, start record,
actual log and three copies have been checked. The auxiliary wrapper exited
zero with 257 MiB peak RSS, 13.27 seconds wall and 12.85 seconds user CPU.

Controls accept the saved native feature/value/depth join and empty/reordered
sample populations. They reject missing/duplicate identities, wrong sides,
disagreeing statuses/deadlines, fabricated zero depth, missing observed depth,
future fields requested as features, rehashed altered targets/features, changed
grid values, external identities and profiles. Changing future target values
leaves the explicitly selected feature matrix unchanged.

This is a prepared data-to-training-table component, **not a fitted depth model**.
No historical day payload was read. Independent source-book provenance, a frozen
historical builder, chronological depth fits, local matched controls and full
sequential policy evaluation remain. The source-book provenance gap is the next
implementation step. Recovery remains 10/18 complete with Nov 2/stress worker
2177940 active; size driver 1576587 waits. Frozen sources remain unchanged.


## Independent cached-book provenance, 2026-11-13 13:38 UTC

Commit **d57977b8** adds `pf_action_depth_book_source.py`. It authenticates the
complete declared book-directory shard set, clock configuration and registry
files, then projects only the explicitly declared markets and source-time bounds
at the Parquet reader. It preserves the native stable source/receipt ordering,
including equal-stamped updates. The clock is rebuilt from shifted raw receipts,
the fixed 13.2 ms minimum delay, declared decision delay and cumulative
monotone correction. Registered market starts must match the requested windows.

The verifier compares every saved best-price/source-clock/known-clock array to
that independent input-file projection and requires exact declared market
coverage, including empty markets. Unknown book prices remain unknown. Input
hashes and the shard directory are checked before and after saved-file review.
This closes a cached-replay-input comparison capability; it does **not** audit
the upstream websocket collector or book parser.

The [combined source controls](../reviews/model_action_depth_20261113/book_source_controls_v1.json)
pass **38 tests** as of **2026-11-13T13:38:21.311976+00:00**: the prior 30
cases plus eight source-provenance cases. Receipt SHA:
`3d6238cc861f3b1d6dda0448f45730870db7f76a72c3893dacd016f43bfdb491`.
All ten implementation/test hashes, three prerequisites, start record, log and
three receipt copies were checked. The auxiliary wrapper exited zero with
255 MiB peak RSS, 13.28 seconds wall and 12.78 seconds user CPU.

Positive controls compare the projected arrays with the original clock helper
and actual native Window constructor on synthetic Parquet shards. They exercise
minimum-delay clipping, non-monotone receipts, equal source clocks, reader-level
half-open bounds, missing book values and an empty registered market. Known-bad
controls reject wrong delay/configuration, incomplete manifests, changed input
bytes, added shards, duplicate/missing markets, wrong registry starts/dates,
nonfinite raw clocks and rehashed altered saved prices/clocks.

The next step is to compose this source verifier with the native depth writer
and causal training-table builder. They currently pass their separate synthetic
interfaces; there is no complete historical depth builder or fitted depth model.
The complete pipeline, admissible bounds, chronological folds, model variants,
nulls and economic policy comparison must be committed before historical fitting.
Recovery remains 10/18 complete; Nov 2/stress worker 2177940 is live under
3159222, and size driver 1576587 waits. No historical payload was read by these
controls and no frozen experiment or live policy changed.


## Composed native bundle queued, 2026-11-13 13:55 UTC

Commit **a50aa572** composes the authenticated book projection, native
KEEP/RESET/PULL capture, depth writer, original-taker provenance and causal
training-table builder. It saves depth while the native trade catalog is still
available, before the original writer releases that catalog. It compares every
native input window with the declared source projection and reconstructs the
saved bundle at closure. Existing scientific and execution sources are unchanged.

Eight new synthetic cases cover the complete native bundle, an independent
original capture replay, exact requests/fills/economics/native-artifact parity,
empty windows/catalog lifetime, the five-share profile, changed training data,
self-consistently altered books, required external identities and input changes.
These cases are **not yet run**. Syntax compilation passed.

The immediate launch refused before any test/start/result artifact because the
other owner's FUTURE-BEST chain occupied core9. New pure launcher **fea64e88**
now waits for that entire chain, then a free auxiliary slot. Its own positive
and refusal controls cover test counts/exits, shell-chain identity and slot
availability. It binds the committed sources and preserves any failed attempt
without an automatic retry.

The [queued controls receipt](../reviews/model_action_depth_20261113/manager_controls_v1_started.json)
identifies driver **2422704**, 46 expected tests and the previous passing
38-test component receipt. The owner chain **2364799** remains active. Until
this new suite passes, only the separate component evidence is established.
Read the [FUTURE-BEST source review](20261113_future_best_model_integration_review.md)
for the reaching-versus-passing distinction and unchanged model limitations.


## Composed native bundle passed, 2026-11-13 14:08 UTC

The [46-test receipt](../reviews/model_action_depth_20261113/manager_controls_v1.json)
passed at **2026-11-13T13:57:42.640559+00:00**, superseding the queued state
above. SHA `e51a4d1a6a24682213619de980683fb151148b49b448b60cd29e21dca7fda2e4`.
All 15 source hashes, three prerequisites, start record, exact test count, log
and three receipt copies match. The wrapper used 258 MiB peak RSS and 26.35
seconds wall time. The complete native adapter preserves original requests,
fills, economics and capture artifacts in synthetic controls, including
partial fills and empty windows; altered source books and training labels fail.

This establishes the source-to-action-to-training composition. No historical
depth fit or policy benefit has been measured. Next is the chronological depth
model component, using only declared causal inputs and retaining missing
outcomes as statuses. The full historical builder, model and policy evaluation
still require a separate committed declaration. Recovery is 11/18 verified;
Nov 3/stress is running and the frozen size experiment remains queued.


## Chronological depth estimates passed, 2026-11-13 14:15 UTC

Commit **4d1558ec** adds `pf_action_depth_fit.py` and its synthetic controls.
Two fixed feature families compare existing STATIC plus size inputs with
HISTORY plus size inputs on the identical observed neutral-action population.
Each estimates the probability of recorded maximum post-cancellation depth
reaching 1–5 ticks. Ten binary heads retain the pinned learning rate, leaf size,
regularization, subsampling, seed and 300 rounds; the binary objective matches
the indicator target. There is no hyperparameter search.

All fitting uses the earlier core dates. The immediately preceding date and
outer date receive predictions without refitting or selecting a policy. The
profile-specific core base rates are saved as a baseline. An explicit causal
feature allowlist excludes depth labels, cancellation deadlines and learned
guard outputs. Depth observations remain usable when economic marks are
missing; missing depth never becomes zero, and all output actions/statuses
remain present. Observed label intervals cannot cross the UTC-day boundary.

A fixed equal-weight decreasing isotonic projection reconciles the five
independent heads. It does not establish probability calibration. Requests
for a deeper unsupported event return unknown instead of substituting the
five-tick estimate. These are recorded-book depth estimates, not the chance
of filling our order or an instruction to cancel.

The first combined suite ran 53 tests with one fixture error: a planted
corruption attempted to insert 1.01 into an integer column before the verifier
could run. Its failed receipt/log remain intact. Commit **8578cb34** changes
that mutation to an integer increment. The [superseding 53-test receipt](../reviews/model_action_depth_20261113/fit_controls_v2.json)
passed at **2026-11-13T14:13:44.905750+00:00**, SHA
`7fbce2f651ac2d44d7f1e09fe3d483d9950237437429531f2e48c2b873c6de77`.
All 17 source identities, three prerequisite identities, start/log evidence
and three receipt copies were checked. Peak RSS was 298 MiB, wall time 34.10
seconds. The positive model control uses over 1,500 synthetic core rows so
the pinned minimum leaf size can actually split on a planted signal. Corrupt
predictions, labels, causal inputs, model parameters, population metadata,
model files, chronology and missingness are refused.

No historical depth target population or fitted artifact exists yet. Next: a
complete authenticated historical builder with independent neutral reference
and native accounting; the chronological prediction report with frozen
comparisons/nulls; then a committed full-pipeline declaration before reading
historical labels. Any policy comparison must account for queue continuation,
latency and actual inventory risk. This component does not revise the queued
size experiment, any existing freeze, or live trading. Recovery remains
11/18 verified with Nov 3/stress running; size driver 1576587 waits.


## Complete historical forecast pipeline frozen, 2026-11-13 14:42 UTC

The missing historical interfaces are now implemented, tested and frozen.
The saved-native converter reproduces the actual native observer's depth
labels, book arrays and causal training table without rewriting upstream
artifacts. Upstream admission independently checks neutral reference parity,
accounting, original takers, source clocks and protected-date boundaries.

The intermediate suites pass 60 tests at 14:21:56.953582 UTC
(`saved_controls_v2`, SHA `9e677fb673c8e48528b05480ce05b91399a25edf6cbeea93f83ee0b631ef3e56`),
65 at 14:26:33.144932 (`upstream_controls_v1`, SHA
`c0fb9e7dda854d8656e0d85a5247d89660d356ee3505560840cff599aafea855`),
and 71 at 14:30:56.680607 (`metrics_controls_v1`, SHA
`954160f70a19be30315e5759bcf5f5c9a8ce7e518c0e66538b5bc2f6be4bb4da`).
These are overlapping combined suites, not additive independent counts.

The full pipeline then passed 76 controls and froze at commit **0e246af6**.
Read the [frozen experiment handoff](20261113_action_depth_forecast_frozen.md)
for its exact targets, population, nulls, 26 stages, source identities,
preserved failed/refused attempts and remaining economic-policy limits.
Driver **2900451** is live and waiting for the complete size evidence chain.
No historical depth fitting has started; forecast or strategy improvement is
not claimed. The earlier statements that the historical builder/report/freeze
were absent are superseded by this checkpoint.
