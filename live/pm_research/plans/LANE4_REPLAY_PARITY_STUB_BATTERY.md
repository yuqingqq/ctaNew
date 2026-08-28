# Lane 4 — seven-arm replay parity-stub battery (SPEC ONLY)

**Authorised by** the user's plan `d506a06` — *"common action-value interface
plus seven-arm offline replay"*, **build/preregister only, nothing scored**.
All code remains offline replay/research code.
**Status:** SPEC. **Owner:** DA (battery design). **Consumers:** DE, BE.

## 1. Why stubs first, and why they are typed

The battery is built and proved **before any predictor exists**. Every arm is a
**typed stub** returning declared-shape output with no model behind it. The
point is to establish that the harness itself is neutral: **if the arms differ
while every predictor is inert, the difference is the harness, and any later
result would inherit it invisibly.**

This ordering is not caution for its own sake. The programme's own history is
that path-coupled overlays amplified prediction noise 10–20x and produced large
replay deltas with zero ranking improvement. A battery that cannot first
demonstrate zero difference under zero signal cannot attribute a later
difference to signal.

## 2. THE ANCHOR TEST

> **A disabled predictor must be BIT-IDENTICAL to `QR_SKEW_ONLY`.**

Bit-identical, not "statistically indistinguishable", not "within tolerance":
identical fills, identical cancellations, identical inventory path, identical
per-window totals, byte-equal serialised trajectory. **A tolerance here would
hide exactly the coupling the test exists to find** — and after today's finding
that non-associative summation alone moves totals by ~1e-11 on identical terms,
"close" cannot be distinguished from "differently ordered but wrong".

Corollary anchors from the user's list, each equally bit-exact:
- **Infinite cancel threshold** ≡ `QR_SKEW_ONLY` (nothing ever crosses).
- **Zero repost threshold with permanent hold** ≡ cancel-and-hold.

## 3. The seven arms (the user's list, unchanged)

1. `QR_SKEW_ONLY`
2. `QR_CANCEL_HOLD_X_SKEW`
3. Fill-hazard-only cancel, neutral placement
4. Conditional-value cancel, neutral placement
5. Conditional-value cancel × frozen skew
6. Conditional-value cancel × frozen skew × fair-price residual
7. Random cancel, **matched on action count, side, hour and budget**

Run on the **same neutral opportunities and independent event clocks**. Arm 7 is
the matched control and inherits rule 7: matched on the decision variable,
compared on the decision metric, never on a proxy.

## 4. Lifecycle invariants the battery must enforce

From the user's list, each a behavioural test with a known-bad:
- One generation is cancelled **at most once**.
- Cancelled skewed orders **cannot fill after simulated effectiveness**.
- **Pre-effectiveness fills remain charged as stale** — the latency estimand on
  the replay side.
- Rate limits count **requested, effective and suppressed** cancellations
  separately.
- **No policy-generated trajectory is reused as its own training population.**

That last one is the outcome-selection rule (rule 1) in replay form, and it is
the one a battery is most likely to violate silently: a policy that generates
its own training data conditions on the event it exists to prevent.

## 5. Falsifiers the battery ships with (rule 15)

Each must FIRE on a known-bad, and each must have a **positive control** —
today's lesson: *the same battery that passes a correct harness must refuse a
broken one, or "all arms agree" is evidence of an unrun battery.*

1. **Anchor, both directions:** disabled predictor ≡ `QR_SKEW_ONLY` bit-exact
   (positive), **and** a deliberately perturbed stub — one extra cancel — must
   BREAK parity. If a one-cancel perturbation does not break it, the comparison
   is not bit-exact and the anchor is decorative.
2. **Determinism, cross-process:** two runs under **different `PYTHONHASHSEED`**
   produce byte-identical trajectories. Today's blocker-7 finding was exactly
   this class — a fixed RNG seed over a process-dependent iteration order is an
   independent draw, not a reproduction — so the battery must not inherit it.
3. **Matched control:** arm 7's action count, side and hour distribution equal
   the arm it is matched against, asserted rather than assumed.
4. **Double-cancel:** a stub attempting a second cancel on one generation is
   REFUSED.
5. **Stale charging:** a pre-effectiveness fill appears as stale, not as
   prevented; a post-effectiveness fill does not.
6. **Empty run refuses:** a battery over zero opportunities must NOT report
   seven passing arms. Zero difference under zero data is not parity.

## 6. What this document does NOT authorise

No scoring, no promotion, no forward clock. Any arm later evaluated on data
starts its own ≥5 complete-UTC-day clock on unconsumed days; consumed days stay
consumed. Whether any arm is adopted is a policy decision with its own priced
trade-offs (rule 14) — the battery estimates, it never decides.
