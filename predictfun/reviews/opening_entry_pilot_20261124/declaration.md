# OPENING-ENTRY-v1 — initial quote batch only

USER,2026-11-24: "Yes we change only the market open entries first".
This corrects the assistant's earlier overbroad all-entry replay. Preserve
that receipt as a distinct experiment; its gain is not an opening-only result.

Before replay, define opening as the wait before the first actual submission
and all posting proposals on that submission's50ms decision tick. At the
next tick permanently restore the original PressureGuard slot policy. No
per-side unlock, later re-entry gate, timer, minimum fill or reset after an
order closes. A submitted-but-refused batch still ends this opening filter;
no future acceptance/fill is consulted to decide scope. If no order is ever
submitted, keep the explicit no-entry outcome. First-batch window covers both
sides/rungs consistently, avoiding side iteration asymmetry.

Reuse the exact previous unfitted directional gate only during that opening
phase: positive opposite-side q5 Binance pressure permits new risk; otherwise
only unreserved inventory reduction. Prices, size, levels, reservations,
inventory sizing, activation/cancel/notices, public-flow assumptions and
live-order handling remain the same. The sign-crossing inventory correction
is retained in both arms. Later entries use the same rule, not necessarily
the same actions: changed opening inventory/queues can change the later path.

Population: same complete consumed market paths Oct24/3702960,
Oct26/3789484, Oct27/3877472. No new days/fit/tuning. New candidate
OPENING_ONLY is counted separately from DIRECTIONAL_ENTRY (family total2).
Reuse the checked FIXED_DEFAULT baselines by file hash and unchanged source
closure. Rebuild the same tape with its exact saved fingerprint. Run one
disabled-wrapper full-market control on the first listed market and require
original FIXED_DEFAULT fingerprint/fills/orders/requests/commands/value parity.

Scope checks: first batch and both rungs, next-tick delegation even if signal
is flat, refused/full/closed orders do not rearm, unchanged live handling.
On every real later slot call independently recompute the original inventory
sizing/pressure action from the saved state and current cached signal. Reuse
the accepted accounting/quantity/notice/reservation checker; add deliberate
late-gating, action, cutoff and gate-log corruptions. No outcome is accepted
before scope and accounting checks pass.

Primary score: whole-market marked END value, split into completed FIFO
turnover and residual; first-minute value and matched shares are secondary.
Also report full filled shares and both-side quoting. Keep absolute values,
all3 markets, baseline deltas and predeclared source horizons. Point estimates
on selected examples only; no validation or selection-beyond-activity claim.
No automatic new variant follows a mixed result.

Budget: packet upper139.43 of180 CPU seconds, leaving40.57. Run hard cap28s;
prepare/check/publication within the remainder. Reuse existing baselines to
avoid redundant runs. Checkpoint outputs; a resource stop is not acceptance.
No cap increase or new data. Commit this declaration, manifest, source and
checker before the first new replay. Frozen historical files stay unchanged.
