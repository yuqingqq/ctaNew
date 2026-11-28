# MOVE-DOWN review for the prediction-model goal

As of **2026-11-13T15:02:39.250143+00:00**; producer report as of
**2026-11-13T14:57:44Z**. Nine consumed dates: Oct 23–27/31 and Nov 1–3,
first 60 seconds of each market, independent 30-share rung 0.

The [review receipt](../reviews/model_recovery_20261113/move_down_integration_review_v1.json)
reconstructs R520, R521 and R522 from the saved outputs, with exact matches.
It binds producer **3695d17c**, nine producer/dependency sources and 65 input
artifacts; 1,111,298 saved fill records across reused variants/clocks are not
that many independent opportunities. Source controls execute the exact
producer functions, including its positive and three planted-error tests.
The bounded audit took 3.06 seconds and 320 MiB on the free auxiliary slot.
All bound hashes and the three receipt copies subsequently matched.

Receipt SHA: `d18fc3ea297f1444512f57146baf33bc755ba65f2d95e2ec82768eaffa99d5e4`.
Declaration SHA: `913ba905746d504802a370274d087ef4c80a07630dbedc6c354641057b605735`.
Review source commit: **9cae506c**. No model fit or policy replay was added.

## What the outputs establish

Entering at the new best while the danger signal remains active increases
participation, but its simulated fills have worse marks. At M50, there are
75,914 adverse-level events per arm: LDG is present in 26,265 (34.6%), MD in
45,162 (59.5%), and MD with the hypothetical fast path in 59,338 (78.2%).
Their conditional median go-live delays are 347, 128 and 46 ms respectively.
These medians condition on different participating subsets; they are not a
paired event-level latency effect.

One-second fill marks in dollars per day at M50, with each cell's fill-record
count in parentheses:

| Arm | WS, age >=100 ms | WS, all ages | All sources, age >=100 ms |
|---|---:|---:|---:|
| LDG, current guard | +6.26 (12,201) | +4.98 (13,515) | -89.48 (24,359) |
| FAST, same guard | +28.02 (12,709) | +27.42 (14,376) | -45.87 (26,048) |
| MD, immediate moved-price entry | -437.73 (43,021) | -775.01 (58,272) | -911.85 (76,734) |
| MDP, MD with depth-based pulls | +6.18 (13,020) | -279.46 (26,572) | -107.75 (26,285) |
| MDF, MD with fast path | -392.15 (46,385) | -971.59 (72,266) | -862.42 (84,665) |

These are fill-mark totals from the legacy independent-rung simulator,
not joint-budget portfolio P&L. All nine days are already consumed; no
untouched-day validation, new null or interval is claimed. Saved nonfinite
one-/five-second fill marks and one-second level marks are zero; this does
not establish timestamp freshness or endpoint coverage.

MDP is especially relevant to the re-entry design. Its entry ignores its
depth score, so a model-triggered pull can be followed by another entry under
the same market conditions. It averages 44,244 pulls/day versus LDG's 12,761.
Its near-baseline result after removing young fills hides substantial negative
young-fill marks. It also inherits the previously identified PULL-DEPTH model
leakage. It cannot establish that the learned withdrawal rule works.

## Two source findings and their practical limits

**R521 omits its declared 30-fill minimum.** The exact source loop labels a
synthetic sparse comparison HELPS even though the declared count gate would
exclude every day. However, every historical compared cell meets the minimum:
zero insufficient cells and no changed R521 readings. This source omission
does not explain the observed negative MD/MDF result.

**WHY-GONE measures cancellation requests, not effective order endings.**
Its live-state check uses `min(tr, tfill)`; its end reason/time use
`min(tc, tfill)`. In the underlying simulator, `tc` is the request and `tr`
the cancellation effect. Exact-code controls demonstrate a fill during the
pending interval classified as a guard cancellation, a request just before
the one-second boundary classified as ended despite exposure surviving past
it, and a previously requested cancellation producing a negative time after
birth. Among 31,744 other-side HELD-OLD rows, 137 actually have negative
reported durations. The reported 16,601 guard reasons (52.3%) therefore
describe request-time attribution, not a reconciled effective-removal share.
The full historical reason changes are unknown: saved tables omit the order
identities and native order timestamps needed to reconstruct them.

Neither this report nor SIDE-MOVES identifies the top makers' placement
times. Their inferred queue advantage cannot be attributed uniquely to
pre-birth resting, lower latency, larger residuals or a particular predictor.
FAST changes feed and order/cancel timings together and remains hypothetical;
its improvement does not isolate one latency component or calibrate live fills.

## Consequences for the active experiment

The size experiment's larger residual can preserve priority across partial
fills, but priority has value only when the remaining exposure is worth
keeping. The new results strengthen the reason to estimate further adverse
movement and to use consistent withdrawal/re-entry conditions. The frozen
depth experiment directly tests incremental history-based forecasts against
a static baseline, using causal neutral actions and post-cancellation-effect
targets. Its eventual forecast score still will not prove policy value.

Recovery, size and depth freezes remain unchanged. Recovery is 12/18 verified
at this review; the Oct 26/age500 worker is active. Size and depth remain
queued behind their declared predecessors. No live trading changes.
