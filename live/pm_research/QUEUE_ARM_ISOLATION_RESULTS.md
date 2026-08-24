# Optimization iteration 006 — queue arm isolation result

**Verdict: `CORRECTED_BASELINE`. Research only; decision eligible: no.**

Protocol: `QUEUE_ARM_ISOLATION_PROTOCOL.md`. Implementation:
`policy_optimizer_queue_isolated.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_queue_isolated_v1.json`. Artifact ID:
`7a1091458fcac46b8bdcf8be269c88f57590ae49972d0ac75e63152899272556`.

All five frozen controls pass: cell-order invariance, other-cell presence
invariance, exact repeat determinism, all-false baseline/skew parity, and
source/code/protocol receipt presence.

The isolated `QR_CANCEL_HOLD_X_SKEW` and `QR_SKEW_ONLY` metrics reproduce the
stored queue-realistic parent exactly on every coin/day. Thus the historical
baseline numbers remain valid; the iteration-005 failure was caused by adding
a candidate with a different cancel-timer schedule to the same event loop,
which changed that run's comparator. Future comparisons must replay each arm
alone.

| coin | isolated baseline mean5 | isolated skew mean5 | delta | baseline dev2 | skew dev2 | dev2 delta |
|---|---:|---:|---:|---:|---:|---:|
| BTC | +185.05 c | +197.75 c | -12.71 c | -40.46 c | -52.88 c | +12.42 c |
| ETH | +196.24 c | +125.28 c | +70.96 c | +53.54 c | -65.62 c | +119.15 c |

No strategy was optimized in this iteration. The corrected invariant replay
wrapper and this receipt supersede multi-arm replay for all subsequent tests.
