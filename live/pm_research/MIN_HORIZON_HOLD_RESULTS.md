# Optimization iteration 001 — minimum-horizon hold result

**Verdict: `REJECT` for BTC and ETH. Research only; decision eligible: no.**

Protocol: `MIN_HORIZON_HOLD_PROTOCOL.md`. Implementation:
`policy_optimizer_min_horizon_hold.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_min_horizon_hold_v1.json`. Artifact ID:
`aff9ced94067ef6370e2fe4b361f39e1d31bb3cb3fc8ba42a27183acab52c383`.

The candidate kept a harmful side out for at least one prediction horizon from
cancel-effective time: 50 ms BTC and 250 ms ETH. It was compared against the
queue-realistic `QR_CANCEL_HOLD_X_SKEW` incumbent on the same ten visible
windows. The v5 model gate remains failed and there are no forward days.

## Frozen development gates

| coin | Aug 23 delta vs incumbent | Aug 24 delta | dev2 candidate PnL | dev2 `QR_SKEW_ONLY` | terminal abs inventory: incumbent -> candidate | verdict |
|---|---:|---:|---:|---:|---:|---|
| BTC | -5.67 c | +15.00 c | -35.80 c | -52.88 c | 15.81 -> 16.73 | REJECT |
| ETH | -113.17 c | +121.35 c | +57.63 c | -65.62 c | 1.25 -> 2.36 | REJECT |

Both candidates beat corrected skew on dev2 and reduce cancel/repost traffic,
but both reverse by day and increase terminal inventory. They fail two frozen
bars and do not replace the incumbent.

Across all five visible days, the candidate changes mean PnL by -67.29 c/window
for BTC and +46.75 c/window for ETH. These means mix model-training and seen
development data and are not adoption evidence.

## Failure attribution

The minimum suppressed 1,326 early signal clears for BTC and 4,434 for ETH and
reduced effective cancels to 2,549 and 2,102 across five windows. However, the
event-driven implementation released only on the next recorded decision event.
Mean delay beyond the intended deadline ranged from about 279 to 748 ms by BTC
day and 171 to 916 ms by ETH day. That delay is larger than the intended 50/250
ms treatment and confounds the minimum-hold hypothesis with a coarse wake-up
clock.

Iteration 002 therefore tests one new structural change: schedule an internal
event at the exact deadline. It remains preregistered against the unchanged
queue-realistic incumbent; iteration 001 stays rejected.

## Verification

- 18 new lifecycle checks pass.
- All eight parent arms have exact fill/diagnostic parity.
- Zero-minimum equals the incumbent exactly; deterministic rerun passes.
- Artifact self-hash and code/protocol provenance pass.
