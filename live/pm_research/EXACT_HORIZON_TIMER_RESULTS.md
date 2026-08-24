# Optimization iteration 002 — exact horizon timer result

**Verdict: `REJECT` for BTC and ETH. Research only; decision eligible: no.**

Protocol: `EXACT_HORIZON_TIMER_PROTOCOL.md`. Implementation:
`policy_optimizer_exact_horizon_timer.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_exact_horizon_timer_v1.json`. Artifact
ID: `df296faf442abc0135d421d43613b7865c499e05e0c0ad06ac5b9975d6693d08`.

The candidate changed iteration 001 only by scheduling an exact internal event
at cancel-effective time plus H. Parent controls were replayed on an independent
internal clock so the candidate's hypothetical timer could not wake them.

All nine parent arms have exact fill/diagnostic parity. The timer candidate has
exactly the same fills, PnL, inventory, and gate result as rejected iteration
001 on all ten windows. Its incremental timer value is 0.00 c/window on every
coin/day.

The timer did cause exact-deadline reposts when q was already clear, but no
historical trade reached those orders before the next public event. Releases
after a still-harmful deadline remain legitimately later, which is why the
all-release lateness diagnostic stays positive.

The incumbent remains `QR_CANCEL_HOLD_X_SKEW`. This result removes scheduler
granularity as the explanation for iteration 001's instability and routes the
loop to signal-state hysteresis.

Verification: 17 new checks, nine-arm parity, disabled-timer parity,
deterministic replay, artifact self-hash, and code/protocol provenance pass.
