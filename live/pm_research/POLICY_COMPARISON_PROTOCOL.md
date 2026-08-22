# POLICY_COMPARISON — new-BBO vs join-BBO, paired

Protocol `policy_v1`. **Frozen 2026-08-22 before measurement.** Coordinator
writes the rules; the research agent measures. Research only, not decision
eligible, no forward-day claim.

## Why this is answerable when the absolute edge is not

The pooled maker edge is `+0.173 ¢/share [−0.251, +0.596]` — useless, and it
needs ~25–30× current data to resolve. **This protocol does not measure it.**

Both policies are evaluated **on the same windows, at the same moments, against
the same flow**. Common variance — regime, day, coin, window phase — cancels in
the difference. A paired contrast can be well determined far below the interval
on either level, which is why the comparison is available now and the level is
not.

**Every headline here is a PAIRED DIFFERENCE.** Report levels for context;
never let a level carry the verdict.

## The two policies

Both quote **one side, 5 shares, on the unified Up book**, and differ only in
*when they join a price level*.

| | **NEW_BBO** | **JOIN_BBO** |
|---|---|---|
| trigger | the level becomes the touch (a new best appears) | the touch already exists when we arrive |
| queue position | **front** — nothing displayed ahead | **back** — behind all displayed depth |
| availability | only when a level forms | any decision time |
| `queue_ahead` | 0 | displayed size at that level |

These are the two ends of the FRONT/BACK bracket **expressed as policies**, which
is what they always were. Queue position is an OUTPUT of the placement rule, not
an assumed parameter.

**`NEW_BBO` is an UPPER BOUND on itself, not a guarantee.** Being first assumes
we win the race against other participants doing the same, which depends on our
latency and theirs and is **not observable in this tape**. Report it as a bound
and say so on every line.

## Estimands, per coin, paired

For each decision time where **both** policies are available:

1. `Δfill = P(fill | NEW_BBO) − P(fill | JOIN_BBO)`
2. `Δmarkout = E[outcome − ℓ | fill, NEW_BBO] − E[outcome − ℓ | fill, JOIN_BBO]`
3. `Δedge = Δ(fill-weighted realised edge)` — the product, which is what a maker
   actually earns per decision

Markout is measured **against settlement** (`S60(T)` vs `S60(t0)`, verified at
99.8 %), **not against a fair-value model**. That is what keeps this decoupled
from Route A and off the 10-day sigma clock.

**Only decision times where both are available may enter a paired comparison.**
A `NEW_BBO`-only sample against a `JOIN_BBO`-only sample is not paired and is
exactly the denominator/population defect this programme has hit six times.
Report the availability rates separately — if `NEW_BBO` is rarely available that
is itself a finding about the policy.

## Decision rule, written before the measurement

Verdict on **btc and eth only** (`verdict_coins`); the other five are descriptive.
Window-clustered bootstrap; day-clustered intervals are not computable at this
day count and must not be claimed.

- **NEW_BBO_DOMINATES** — `Δedge > 0` with the interval excluding zero on **both**
  btc and eth.
- **JOIN_BBO_DOMINATES** — `Δedge < 0`, interval excluding zero on both.
- **TRADE_OFF_CONFIRMED** — `Δfill > 0` and `Δmarkout < 0`, both intervals
  excluding zero, while the `Δedge` interval **spans** zero. The policies differ
  in mechanism and not in outcome; placement is then a **risk-shape choice, not
  a profit choice**, and that is a real finding.
- **NO_DIFFERENCE** — every interval spans zero, and the `Δedge` interval is
  **tight enough to exclude an effect worth acting on** (bound: `|Δedge|` CI
  within ±0.25 ¢/share). Placement does not matter, and the entire FRONT/BACK
  bracket was a distraction.
- **UNRESOLVED** — intervals span zero but are too wide to rule out an effect
  worth acting on. Say what n would settle it. **This is the default when
  underpowered; it must not be reported as NO_DIFFERENCE.**
- **VOID** — fewer than 200 paired decision times on either verdict coin.

## Pre-specified expectation, so a confirmation is not a surprise

`NEW_BBO` should win on fill (94.6 % vs 76.9 % at 15 s on btc, unpaired) and
**plausibly lose on markout**, because it quotes when the level is forming, thin,
and information is freshest. **A result showing `NEW_BBO` better on BOTH is a
reason to suspect the measurement**, not to celebrate — check the pairing and the
availability rates before believing it.

## Rules

1. Paired only; state the population of every denominator.
2. Per coin, never pooled — btc is ~64 % of any pooled denominator.
3. Read book state from `price_change.best_bid/ask`, never `book` snapshots.
4. Knowledge time is `recv_ns`; state read at the frozen 250 ms lag.
5. Gap-touched and tick-change-touched actions are `UNAVAILABLE`, reported not
   dropped.
6. R-DUAL: the micro class is 2–90 % of events by coin. Report both weightings.
7. A control per probe that fails if the test is vacuous.
8. Do not narrate a mechanism that was not measured.

## Out of scope

The absolute maker edge, the fair-price model, queue-position inference, and any
`ρ`-dependent quantity. This protocol compares two policies; it does not
establish that either is profitable.
