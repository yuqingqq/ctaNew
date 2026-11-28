# Size-aware model policy checkpoint, 2026-11-13 10:45 UTC

The goal remains active. This turn connected size-aware predictions to native
quote/cancel/re-entry decisions, integrated a chronological guard benchmark,
and verified saved policy evidence. All final bounded controls pass. These
are synthetic implementation results: no historical larger-order fit,
policy-performance gain, untouched-day validation or live change is established.
The full frozen recovery experiment remains unchanged, with5/18 runs complete.

## Implemented policy and model identity

`pf_size_priority_fit` now binds each training profile's original order quantity
and inventory limit in its receipt. A profile cannot silently change within the
pool. `pf_size_priority_predictor.Predictor` authenticates that receipt, all three
model files, feature order/target identity, pinned parameters, replay day and
core/calibration chronology. It refuses a different quantity/budget profile;
the old five-share/cap-ten model cannot be applied to50-share quotes through
this interface. Partial initial quantities remain valid within the declared
profile, because native capacity may reduce an individual accepted order.

`pf_size_priority_model_policy` composes beneath the existing reservation layer.
Its distinct modes are `default`, `guard` (legacy matched guard), `static`,
`history`, `selective`, and `causal_guard`. All three learned policies
are built on sized default quoting; the legacy guard deliberately retains its
existing entry-withholding behavior as a diagnostic benchmark.

Static/history pull at negative estimated KEEP-minus-PULL cents. Selective
pulls only when history value and estimated immediate original-order value are
both negative. Models emit numeric values; these zero-threshold rules live in
the policy layer. A retained partial order keeps its original price and initial
quantity, with notified filled quantity reducing its remaining exposure. The
policy never uses physical queue/remaining fields as features. Original price,
data, minimum-notional, pending/late-fill reservations and inventory constraints
still apply. There is no forced minimum resting-time lock.

Minimum score age, score cadence and deterministic audit sampling are explicit
`PolicyConfig` inputs. Controls used0ms/100ms/every score; those values are not
yet a full historical declaration. Pending creates are not scored as accepted
orders. At each eligible decision, the model reconstructs the same `SizeHistory`
features used by saved training actions, including retired reservations in the
client-visible state. A learned pull starts the exact same `Withdrawal` class
and pressure-clear configuration used by the three-way PULL target. Initial
quoting remains the default; pressure clearance applies after an explicit pull.
The fixture uses250ms clear observations with gaps no larger than50ms. Unknown,
unsafe or interrupted observations reset clearance.

The labels compare one intervention followed by their declared continuation.
The full policy can make later withdrawals after re-entry; this repeated-decision
distribution must be evaluated in sequential replay. Local labels alone do not
prove a portfolio gain or solve that distribution shift.

## Chronological guard benchmark

`pf_size_priority_guard` loads the earlier-only guard artifacts and the
training-only phase thresholds prepared by the existing model-upgrade study.
It verifies preparation/fit/model hashes, matching core days, calibration day
and target replay date, then caches both side estimates on the supplied causal
market grid. Decision lookup uses only the latest row at or before that clock,
with the same100ms age bound; future-grid mutation leaves earlier decisions
unchanged. The existing G1/G2 callback remains part of the guard predicate.
Saved guard scores/thresholds/actions reconstruct from the authenticated guard.

The `guard` mode is the legacy cache-based matched guard and must not be called
chronological. `causal_guard` is the proper earlier-only comparator under the
same explicit size/inventory profile. Reusing the old guard fits does not supply
new size-model training support. The historical runner still needs to bind their
actual preparation paths and all upstream receipts in its declaration.

## Full-policy evidence

`pf_size_priority_model_manager` writes native generation/window accounting and
one compressed evidence row per market window. Every score, learned pull,
pressure observation, place and cancellation is recorded. Cancellation evidence
carries its actual request and effective clocks, including pending-create status.
The checker binds order/side/quantity/age to accepted native generations,
reconstructs score decisions and repeated recovery cycles, refuses placements
before clearance, and verifies that every learned pull has its actual cancellation.
Empty-generation windows remain explicit and get an empty generation artifact.

`pf_size_priority_policy_evidence` reconstructs sampled client states against
the actual decision-state hash, rebuilds causal features from the supplied grid,
and recomputes forecasts from the predictor. The sample population itself must
match the declared hash rule exactly. Dropping a sample is refused even when the
saved counts and file hash are changed consistently. Actual inventory and
native accounting are retained; equal configured budgets still do not imply
equal realized risk.

## Controls and preserved failed attempts

| Receipt | Status | Expected / reported tests | As-of UTC | Source |
|---|---|---|---|---|
| `size_priority_model_policy_controls.json` | FAILED | 13 / 13 | 2026-11-13T10:32:46.312591+00:00 | 62dd9312 |
| `size_priority_model_policy_controls_v2.json` | PASS | 13 / 13 | 2026-11-13T10:35:10.853355+00:00 | 65b67c86 |
| `size_priority_guard_policy_controls.json` | PASS | 13 / 13 | 2026-11-13T10:39:23.159063+00:00 | 7c4be22f |
| `size_priority_policy_sampling_controls.json` | FAILED | 2 / 9 | 2026-11-13T10:41:58.390095+00:00 | 82f57871 |
| `size_priority_policy_sampling_controls_v2.json` | PASS | 11 / 11 | 2026-11-13T10:44:22.767727+00:00 | 98b6bab8 |

The first13-test attempt exposed a local `profiles` name shadowing the validator
in the new training-profile binding; no historical model was fitted. Source
65b67c86 corrects it and all13 controls pass. The first sampling-completeness
attempt reused `expected` for client fields, overwriting the expected sample set;
its fixture and three native cases failed. It also imported a TestCase into the
test module, causing unittest to discover9 tests rather than the declared2.
Source98b6bab8 uses distinct variable names and module-qualified fixture imports.
The final explicit11-test run includes both new sampling controls and all nine
model-policy tests; it passes. All original sources/logs/failed receipts remain.

Passing controls cover native positive-score/default request and fill parity,
partial retention and latent-field mutation invariance, negative-score cancel
and re-entry, native reservation precedence, saved model/profile identity,
chronological guard admission/withholding, empty windows, corrupted cancellation
and pressure state, changed/rehashed forecasts and missing audit samples.
The model-policy and guard suites overlap; their counts must not be added as
independent evidence. No historical payload was read by these tests.

Final sampling receipt SHA:
`4f06122d690cbc962def226b572f38a6d20e84bbbd797682c9f4f603ac76e91e`.
Guard receipt SHA:
`9badec14eced6331ae9faf219346b1087be74443cf2db027c83aa85d886f9acc`.
Model-policy v2 SHA:
`f39c916256b2d8dafc27c2aa58df250496955acc0703df7acee6dfca812b5e08`.
Logs, committed source identities, and three receipt copies were checked.
The302 original frozen recovery sources are unchanged. The final11-test wrapper
used324MiB and97.65seconds on the free auxiliary core9. No test job, queued
launcher or exec session remains pending at this checkpoint.

## Main recovery replay and next work

Nov2/X completed at2026-11-13T10:28:15.667677+00:00: four policies x288 windows,
exact neutral parity,454 reconstructed inference samples and reconstructed
recovery events. `fifth_full_run_review.json` verifies25 named artifact hashes,
three receipt copies and all accounting-correction links, with four corrupted-
link refusals. This is a receipt/artifact review, not an independent second replay.
Its SHA is`f733a1f5602a625545b2c782bcd4c334394e13e4ae716128fd6d5c87f71f576d`.
Five of18 full scenario-days are complete. Driver3159222 and Nov3/X worker591257
were confirmed live at this checkpoint. No final multi-day policy verdict yet.

Full-policy taker provenance is now implemented in source c9dcfb61.
`pf_size_priority_model_trades` wraps the actual model-policy callback, preserving
native matching and global fill order while recording original trade identity,
clock anchors, all generations (including unfilled), and strict markouts. Its
saved checker reconstructs cash, terminal inventory, fill counts, generation
quantity and native-order maximum inventory. Unknown taker identities leave
later-taker value unavailable. Multiple records of one taker do not count as
repeated distinct takers. Later-taker value is not a causal queue premium.

The five synthetic controls pass as of2026-11-13T10:52:25.536251+00:00:
positive policy request/fill parity; same-budget five/50 retention mechanism;
same-taker/unknown identity handling; rehashed corruption refusals; empty-window
coverage. Receipt `size_priority_full_policy_trades_controls.json` SHA
`6a63a5aa148ec4dcc557f7f497c03ec003f3d50b859ecc224f6eb7712f58a9e8`.
All66 committed source identities, log hash and three receipt copies were
checked; the302 frozen recovery sources remain unchanged. The wrapper used
235MiB/40.91s on the free auxiliary core9. No historical payload was read.

Then implement the historical runner and complete experiment specification:
verified corrected-grid/source/input identities, protected-date boundaries,
independent sized-neutral parity, feature/target joins, chronological fold
orchestration, guard/predictor artifact loading, full policy accounting/presence,
local matched-count nulls, day-cluster intervals and matched-risk comparisons.
Use an explicit shared budget permitting50 shares for both five/50 profiles;
retain the original five/cap-ten strategy as separate context. Native arm labels
and cap settings must identify the actual profile. Full execution sensitivity,
missing-mark statuses and all consumed-date limitations remain required.
The complete pipeline, null design and candidate/scenario multiplicity must be
committed/frozen before historical size execution. No complete historical size experiment
declaration exists yet. All full numerical
runs use core8; auxiliary capacity is for bounded controls/preparation only.


Runtime/accounting continuation: see `20261113_size_runtime.md` (11:10 UTC).
Actual size/cap descriptors, independent native target/reference parity, exact
delta arithmetic and full portfolio/presence accounting now pass controls.
The historical stage driver/full declaration remain. Recovery6/18 complete;
Oct26/stress is active. No historical size-model result or live change.
