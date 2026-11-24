# DIRECTIONAL-ENTRY-PILOT-v1

2026-11-24, before replay. USER: "We set up the entry rule and replay, what do
you think". One exploratory candidate on exactly the three consumed dashboard
markets: Oct24/3702960, Oct26/3789484, Oct27/3877472. Keep their whole supported
paths to opening+298s. They were selected examples, not an unbiased validation
sample. No fit, new market/day, live use, parameter search or calibrated-EV claim.

The user's latest replay instruction authorizes this new replay scope. Reuse
only the unused part of the earlier180 CPU-second packet: its last conservative
upper is87.11s. The entire preparation/run/check/publication must fit92.89s;
run CPU hard stop75s including imports, reserve the rest for preparation and
checking. No task/line cap increase. If it cannot finish, retain checkpoints,
do not silently expand the budget, and report the specific resource needed.

## One admission rule

Reuse the accepted float32 `adv_bn_r250` cache and both frozen q5 boundaries.
Permit new bid exposure when ask-adverse pressure is in its positive q5
(upward return >=0.285456120967865 in the existing units); permit new ask
exposure when bid-adverse pressure is in its positive q5
(downward return magnitude >=0.4666730940341952). Use the unchanged bin helper
on float32 values, including its exact boundary rule. A valid flat signal is
not missing; it simply provides no directional permission in this candidate.
Missing/nonfinite/non-ok values provide no new-risk permission. No added age
threshold, rolling transformation, fair-value model or confidence claim.

Run the existing inventory/pressure slot policy first. A proposed POST keeps
its proposed quantity if the directional condition holds. Otherwise permit
only whole shares reducing known inventory after all unclosed same-side
reservations: bid room=max(0,-I-B), ask room=max(0,I-A), quantity=min(proposal,
floor(room)). Below one share WAIT. This rule cannot label an entire100-share
order as reducing when only10 shares of inventory need reducing. Existing
cap, own-cross, price locks, near-END and post-only refusal still apply.

This applies to every new posting proposal, not just tick0; it cannot unlock
an unrestricted opposite-side quote one tick after the first post. It never
adds a cancellation rule. Existing live orders retain the original pressure,
structural and inventory handling, including when the entry signal becomes
flat. It has no cooldown, fixed delay or minimum holding timer.

This tests a simple directional-entry hypothesis, not a claim that momentum
sign estimates conditional execution value or guarantees profitable prices.
Do not tune its window or thresholds using the outcomes.

## Controls and prerequisite bundled fix

Three CURRENT-queue arms (no idealized front treatment):

1. ORIGINAL: reproduce the saved CURRENT_ASYNC_FAST10 fills, orders, requests,
   fingerprint, value, notices and guard log exactly on each market.
2. FIXED_DEFAULT: the same rule with corrected inventory hysteresis.
3. DIRECTIONAL_ENTRY: the same corrected controller plus the admission rule.

The local wrapper corrects skipped-deadband sign crossing: long restriction
persists only while I>+50, short only while I<−50; crossing the opposite100
threshold enters the opposite restriction. Both tested arms get this fix.
Frozen source modules remain unchanged. Synthetic long-to-short and
short-to-long jumps must expose the original defect and pass the wrapper.

Use10ms activation/cancel,100ms notices,50ms grid, M50 public execution clock,
100-share maximum per level, two levels,200-share cap and existing inventory
sizing. Each arm has its own feedback state. The literal legacy tape loader
and path builder are reused; require the exact saved tape fingerprint before
either comparison arm. Inputs, source closure and checker are hashed and
committed before replay. No outcome-based membership/exclusion.

## Evaluation and acceptance

Primary paired contrast: DIRECTIONAL_ENTRY minus FIXED_DEFAULT over each whole
market, with completed FIFO turnover and residual value at the identical END
mark separated. ORIGINAL-to-FIXED_DEFAULT isolates the necessary controller
repair. Report absolute values against the zero value of never trading; lower
loss from lower participation is not profitability or predictive skill.

Also report filled shares, peak/mean inventory, quote participation (explicit
50ms grid sampling), generations, pulls, first entry time/side, first-minute
accounting as a secondary opening diagnostic, and50ms/1s/5s/30s markouts with
their valid shares and a shared END−30s fill cutoff. Do not promote a1s
improvement over losing completed turnover. Missing marks are not zero.

Independent checker: exact all-nine membership, saved-original parity,
cash/inventory/END reconciliation, quantity conservation, absence of wrong-sign
restrictions, signal at the decision's own tick, input known inventory from
notified fills, outstanding reservations from prior commands/closures, and
allowed entry sizes against that evidence. Include intact and known-bad cases.

Compute pooled/per-market delta signs and absolute profitability; report the
three-market point estimates without significance or confidence intervals.
No matched activity null is run, so no selection-beyond-exposure claim is
allowed. This is a small feedback-policy feasibility diagnostic. An all-WAIT
or low-activity loss reduction is not a positive-edge result. A mixed/negative
answer does not launch threshold tuning or extra variants.
