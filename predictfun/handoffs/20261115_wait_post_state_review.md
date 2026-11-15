# Wait/post state and target review — 2026-11-15

Request: add missing inputs, review the model, add necessary information and
improve it. Source99173056, freeze42f61bf1. Entry-model changes only; the separate
stop-model and its cancellation-policy experiment remain separate.

## Implemented and reviewed

| Area | Finding and repair |
|---|---|
| Own ladder | Added client-live orders on both sides at every rung, including overlapping pending cancellations. Prices, sent/activation times, received fills and remaining sizes are observable; snapshots reconcile to client exposure. |
| Queue information | Public known-frame shown size and a causal queue proxy; unknown deep levels remain missing. True replay queue and future minimum size are excluded. |
| Joint waiting | Per-slot generation and time since its first eligible empty decision are available. This adds information for observed states; it does not create training examples for unvisited states. |
| Market context | Known-frame liquidity scale, normalized depth, best-price age, imbalance and feed age. Market-level age is separate from our order age. |
| Start-price availability | The earlier entry feature path read the start-price gap while its reference was unknown. New candidate inputs and the frozen continuation mask it. Labels are regenerated, not patched. |
| Economic attribution | First post-probe order contribution is separated from the rest of the ladder. WAIT may enter later under the same continuation. Replacement orders and displaced fills remain in the external component. The sum exactly equals the portfolio objective. |
| Target population | Neutral reference generations, independent hash samples, no fill-selected rows, raw economic labels including zeros and rare losses. Original unequal inclusion probabilities are weighted in fitting and scoring. |
| Same information | POST, WAIT and NEVER branch before the same decision with identical feature vectors and action prefixes. Prediction inputs do not include their future outcomes. |
| Inference | Saved model feature order/digest and chronology are checked on load. The runtime callback uses the same vector builder as labeling. Separate bid and ask heads; entry requires positive absolute value and nonnegative timing advantage. |
| Old evidence | Frozen v2 source and artifacts remain untouched; all new outputs use a new namespace. |

## Controls completed before historical runs

As of2026-11-15T02:02:49Z:13 synthetic tests passed. They cover execution parity at
all three declared fill clocks; pending reservations; delayed partial fills;
future venue-event invariance; queue causality; sensitivity to known prices,
ages and quantities; complete-vector future perturbations; the start-price mask;
economic signs/component sums; displaced profitable fills; no-fill outcomes;
runtime feature parity; and invalid feature-schema refusal. Known-bad quantities,
clocks, incomplete ladders and broken target sums are refused. The first test
attempt had two fixture errors: it forgot the M50 anchor shift. Those fixtures
were corrected to test the intended100ms notification delay; production clocks
were unchanged.

Every historical window also runs a future-feature perturbation control with a
known-input positive control. Every paired label checks shared decision identity
and portfolio accounting. Chronology and immutable source/input bindings are
checked by the supervisor before and after every worker.

## Remaining limitations — do not call these fixed

- No learned selective keeper is integrated. Training and runtime here use
  neutral KEEP plus engine safety; entry value will need relabeling under the
  intended frozen stop policy when that policy is defined.
- Per-order decomposition narrows attribution, not valuation time. Values still
  use the common60.2s terminal mark. It does not promise a learnable short-horizon
  signal or remove all noise.
- The reference population still has limited later-generation and jointly
  waiting support. Sampling weights cannot replace missing observations.
- Client queue inputs are estimates, not evidence of actual front position.
  Replay fill clocks, E2 anchors, young fills and Binance availability assumptions
  remain material. No local model test validates portfolio economics.
- The bootstrap continuation's weights remain frozen from v2; only its gap input
  availability is repaired. This isolates the new candidate comparison but does
  not claim the teacher itself is optimal.
- All evaluation dates are already consumed. Results are development evidence;
  acceptance needs a frozen combined policy and later untouched days.

Performance is recorded separately after all declared candidates finish. No
model is promoted based on code correctness, feature count or training fit.
