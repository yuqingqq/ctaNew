# Review of ORDER-HARM v3 / POLICY v2 fixes

Reviewed **f481b01b** and its parent design **9d38c57d**, 2026-11-14
around **09:45 UTC**. This is a code review with tiny synthetic controls,
not a historical experiment or model fit. The handoff records the user's
instruction not to run; neither research driver was launched. No top-level
label, model or result artifacts were present for these new versions at review.

The changes address several earlier design concerns, but two concrete defects
remain and the discussed 50 ms cadence has not been implemented.

## P1: labels still inherit decisions based on unobserved fills

`pf_order_harm_v3.py:262` builds the keep-all reference through `K.sim_arm`,
which calls the older `pf_kp_s4.sim2`. That engine's `rest_` predicate at
`pf_kp_s4.py:392` uses the venue's remaining quantity. With `idle_only=1`, an
unseen full fill makes `rest_` false, stops suppressing the guard, and causes
a cancellation timestamp to be recorded immediately.

The new `client_resting` predicate checks both notification-delayed full fills
and `o['tc']`. It therefore inherits the early cancellation through `tc`;
adding the fill-notification delay in that predicate does not repair the
generating reference path.

Actual-source synthetic case: an order is placed before a guard episode
starting at 200 ms. The venue fills it at 400 ms; notification is at 500 ms.

| Reference path | Recorded cancel | Label decisions before 500 ms |
|---|---:|---|
| No fill | Window end | 200, 450 ms |
| Fill at 400 ms, not known until 500 ms | 400 ms | 200 ms only |

The 450 ms state disappears even though the client does not yet know the fill.
This changes the training population and continuation chain relative to the
client-known policy lane. An in-memory repair of the generating engine's
resting predicate restores the 450 ms decision; making the notification
immediate legitimately removes it. Both controls are saved.

Required repair: generate reference paths and label decisions using the same
client-known semantics as JOINT-LADDER v3, and test decision-prefix invariance
against unobserved fills. The existing recorder parity compares two runs of
the same legacy engine, while feature parity skips unmatched orders/states;
neither certifies the missing decision population. The v3 source discloses
the old engine as a limitation; this is a concrete failure within that limit.

## P1: the new portfolio metric includes fills after its valuation time

`pf_order_harm_policy_v2.py:314` selects the mid at 60 seconds. The engine
cancels remaining orders at 60 seconds and continues matching trades while
cancellation is in flight. Lines 338-345 include those fills (anchors below
61 seconds) and value every included fill at the earlier 60-second mid.
R652 then treats this quantity as terminal portfolio PnL.

Actual engine plus the exact inclusion/mark expressions, synthetic M0 case:

- Buy order: 30 shares at 50c; cancel sent at 60.000 s, effective at 60.022 s.
- Fill occurs at 60.010 s. Mid at 60.000 s is 51c; later mid is 40c.
- Saved-metric calculation: **+$0.30**. Actual cash plus inventory at 60.000 s:
  **$0**, because the fill has not happened. At a common terminal after the
  cancellation tail, the filled position is **-$3.00** in this example.
- All seven engine accounting counters remain zero. A fill before 60 seconds
  passes the terminal-consistency control; one after cancellation cannot fill.

These are synthetic amounts, not historical impact estimates. The issue is
the inconsistent event/valuation horizon, not a claim about its empirical size.
The same mark convention existed previously, but this fix promotes it to the
acceptance criterion.

Required repair: finish the cancellation tail and mark all accumulated cash
and remaining inventory at one common terminal, or report an exact 60-second
snapshot and the subsequent tail separately. Keep post-stop fills accounted
for. Add a checker that refuses a purported terminal PnL containing executions
after that terminal, with cancellation-boundary cases.

## P2: model decisions are still every 250 ms

`pf_order_harm_v3.py:77` still sets `EVERY=5`; POLICY v2 imports that value.
Its cadence wrapper decides at the guard onset and every fifth 50 ms step.
The synthetic wrapper test scores at 50 and 300 ms, ignoring a changed decision
score at 100 ms. This fixes the earlier comparison confound by adding FB25C,
but it does not implement the subsequently discussed 50 ms model cadence.

Make decision cadence explicit and consistent between label construction and
policy evaluation; test the requested 50 ms case. Keep the separate 250 ms
re-entry cooldown and execution-latency assumptions unchanged for that test.
Rebuild or version cadence-dependent labels and controls rather than silently
reusing the prior outputs. An inference-runtime benchmark remains outstanding.

## Improvements verified in source, and remaining scope

- FB25C and model arms use the same cadence wrapper; FB25 at 50 ms remains
  separately available to measure the cadence effect.
- R652 includes WS and E2 trades, all three timing assumptions, both age
  treatments, and comparisons with both LDG and FB25C. The broader design is
  useful, but its terminal accounting needs the repair above.
- Outer model fits use earlier training days, a separate calibration day,
  and a later test day. Thresholds no longer pool all test dates. The guard
  and FRONT-PULL predictions still come from upstream leave-one-day-out fits,
  as explicitly disclosed; the full pipeline is not yet chronologically clean.
- The target now attempts continuation value through fitted Q iteration;
  its prediction diagnostic no longer scores against its own bootstrapped
  target. This does not establish that the fitted option value is correct,
  and the reference-path defect must be fixed before fitting.

Both new numerical programs are development studies on consumed dates.
There are no new performance results to endorse. This review does not reopen
the repaired JOINT-LADDER v3 future-rejection finding: the newly demonstrated
causality problem is in the legacy engine reused by the label builder.

Instrument: `research/model_order_harm_v3_fix_review.py`; immutable source
hashes and traces: `reviews/model_order_harm_v3_fix_review_20261114/`.
The instrument changes one predicate in memory for its positive repair
control. No research source, worker, or frozen artifact was modified.
