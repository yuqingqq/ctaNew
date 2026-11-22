# Lower-latency amendment to the queue/sweep diagnostic

2026-11-22. USER: "We can use a lower latency assumption in the test, the
latency now is assumed." Source checked:
**1da57adb8de74616bbd56c638c6d2af5e64a008c**.

This supersedes the two-arm/timing/resource portions of the
[queue/sweep proposal](../20261122_queue_sweep_direction/plan.md), not its
population, economic objective, outcome definitions, ownership or materiality
rules. New recommended experiment name:
**QUEUE-PRIORITY-LATENCY-SENSITIVITY-v1**. Lower-latency sensitivity is explicitly
requested. No replay or implementation is performed by this amendment; no
previously unapproved compute ceiling is silently approved.

## Separate the clocks

The engine's effective settings are in `pf_branch_engine.py:83–92`, with the
causal event/notice order documented at lines 21–27. The baseline treats the
delays as fixed simulation parameters. Older documentation relates 58/37 ms
to placement/cancellation measurements; this does not establish their value
for a future deployment. Fill/cancel notification assumptions are separate.

Use exactly two predeclared profiles; no latency sweep or choice after results:

| Clock | BASELINE | FAST10 |
|---|---:|---:|
| Submission to activation | 58 ms | 10 ms |
| Live cancellation request to effect | 37 ms | 10 ms |
| Match to client fill notice | 100 ms | 10 ms |
| Cancel effect to client acknowledgement | 100 ms | 10 ms |
| Post acceptance/refusal notice after activation | 0 ms | 0 ms |
| Decision grid | 50 ms | 50 ms |
| Minimum pending-cancel effect relative to submission | 95 ms | 20 ms |

Pending removal remains `max(request + cancel_latency, submit + pending_min)`
when the request precedes activation. FAST10's 20 ms is the declared
10+10 ms analogue of the baseline's 58+37 ms, not the inherited 95 ms constant.
The matching fill-notice and cancel-ack latencies preserve the existing
condition that a cancel acknowledgement cannot overtake any earlier fill
notice. Post-only refusal, trades-before-commands ties, inclusive activation
and exclusive removal remain as declared by the current engine.

**FAST10 is an optimistic sensitivity assumption, not a measured achievable
latency or a claim of 10 ms end-to-end response.** It changes the combined
action/notice profile. A gain cannot be attributed solely to faster cancel
transport: entry timing, acknowledgements, inventory knowledge and re-entry
also change. This compact experiment does not separately identify those effects.

Keep Predict venue/receipt frame timestamps and Binance signal clocks/cache
unchanged. A decision uses only messages available by its tick. The 50 ms
decision grid is not a uniform feed delay; message age remains variable.
Even FAST10 waits for a decision tick and any delayed source information.

**M50 is a different assumption:** `pf_queue_place_test.py:65–86` constructs
the trade tape using its 50 ms event-clock shift. M0/M50/M250 are not order
submission speeds. Preserve M50 and the complete exogenous tape/END marks
in every arm. Moving recorded matches later, advancing future feed rows or
reusing a later signal at an earlier tick would confound execution timing
with trade-clock or information assumptions and is outside this experiment.

## Four cases, one experiment

| Arm | External queue on accepted new orders | Timing profile |
|---|---|---|
| CURRENT_BASELINE | Current initialization | BASELINE |
| FRONT_BASELINE | Zero ahead once at activation | BASELINE |
| CURRENT_FAST10 | Current initialization | FAST10 |
| FRONT_FAST10 | Zero ahead once at activation | FAST10 |

CURRENT_BASELINE reproduces the accepted FULL_11/Q5 ladder. The other three
are declared new sensitivity arms, recorded in multiplicity before replay.
The zero-ahead arms retain all hypothetical-execution limitations in plan.md.
Same market tape and strategy rules, genuine per-arm inventory/notice feedback;
never force the original action schedule onto changed fills.

Declare and compute the paired economic contrasts:
- queue effect at baseline: FRONT_BASELINE minus CURRENT_BASELINE;
- queue effect at FAST10: FRONT_FAST10 minus CURRENT_FAST10;
- combined latency-profile effect at current queues: CURRENT_FAST10 minus CURRENT_BASELINE;
- combined latency-profile effect at front queues: FRONT_FAST10 minus FRONT_BASELINE;
- interaction: FRONT_FAST10 - CURRENT_FAST10 - FRONT_BASELINE + CURRENT_BASELINE.

Report absolute END and eligible 1-second value alongside every improvement,
day/side/market support and the existing longer-horizon diagnostics, fill mix,
inventory and churn. If current-queue FAST10 works but current-queue baseline
does not, it motivates measuring achievable timing. If hypothetical priority
is needed as well, it motivates priority acquisition plus timing work, not an
assertion that the strategy is executable. If all remain negative, favorable
assumptions have not rescued this configuration. Do not tune intermediate
latencies or thresholds on the same results. Four consumed days remain
development evidence only; no confidence intervals or live promotion.

## Outcome-critical implementation checks, one combined review

Use a new declared configuration/wrapper, leaving frozen baseline sources and
receipts unchanged. Bind the ACTUAL effective timing, queue mode, source
hashes and input identity in all records and resume checks. Audit every
dependent use: submission feasibility near END, pending states, notices,
reservation release, structural/policy cancellation, client queue estimates,
observation snapshots and measurement. Changing only an engine constant while
helpers retain old delays is not a valid FAST10 run. The pressure market
cache may be reused only because source times, grid and feature builders are
unchanged; order/context observations use each arm's actual timing.

Reuse unaffected tests. Add small boundary fixtures for each changed timing,
pending cancellation, no release before acknowledgement, partial fill notices,
and a trade just before/at removal. Include a source event not delivered by
the decision: FAST10 must not expose it sooner. Check effective settings in
an actual trace and baseline parity. Bundle these with the queue fixtures and
ledger reconciliation; defer unrelated reporting cleanup.

## Resource proposal, still awaiting approval

Four arms replace two so the old proposed batch cap is superseded. Proposed
maximum **0.40 additional core-hours for the whole development batch**, with
**2.45 cumulative core-hours including review**. These caps are proposals,
not measured runtime predictions or existing authorizations. They replace,
not add to, the prior proposed 0.25/2.25 caps. The previously suggested
broader-pressure budget is unrelated and remains unapproved/stopped.

Retain the write-once pilot and projection gate: first two complete markets
per date, pilot reused in the final 19-market result, all remaining jobs,
verification and review reserve included. If it cannot fit, stop at the
checkpoint without changing the arms, population or ceilings. This USER
message authorizes changing the timing assumption in the proposed test;
the explicit cumulative-compute approval requirement is still outstanding.
