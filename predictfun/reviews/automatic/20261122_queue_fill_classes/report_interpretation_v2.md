# Superseding interpretation — early losses and retained queue value

2026-11-22. USER clarification after classification result
**c72f0d37c0debcefc92e3ef83283d62a05f86ef2**:
“The losing fills in the beginning may help us secure the front position in
the queue and [be] necessary for the following good fills.”

This governs interpretation of `report.md`; its frozen numbers and artifacts
are unchanged. No new measurement, replay or implementation is performed.

## Correction to the answer

The USER's hypothesis is the **whole-life cost of keeping an early queue
position**. Accepting early adverse partial fills while preserving the
surviving order may enable later profitable fills that a newly posted order
at the back cannot obtain. That is a plausible retention mechanism.

The classification did not test this mechanism. It starts at the first live
KEEP callback's cancel effect and separates earlier fills as a prefix. An
order labeled `GOOD_ONLY` after that anchor can have earlier losing fills.
The reported 11 compensated bad-then-good cases therefore **are not a count
of all whole-life cases of early loss followed by profitable queue use**.
Likewise, 1,114 `PURE_BAD` suffixes and the 92.85% loss share cannot establish
that early losses rarely pay for queue retention. They describe a different,
explicitly later conditional exposure.

The prior phrase “bad fills do not mechanically buy queue priority” should
not dismiss the USER's economic argument. The relevant choice is to remain
exposed and preserve the surviving order, or avoid early loss and potentially
give up later queue access. One can incur a cost to preserve a valuable
position without the cost itself being an exchange priority instruction.

## Required comparison for this hypothesis

Use the same pre-action state and a declared observation window through a
common end. Include the entire generation from admission, all partial fills,
and any replacement orders on their own identities.

- **Retain:** the early order remains under a declared resting/risk policy;
  partial fills preserve its remaining priority. Count early losses and later
  good fills together, as well as inventory/other-order effects.
- **Avoid and re-enter:** avoid the early exposure using an action actually
  available then, and re-enter by a fixed, causal rule. A replacement starts
  with its actual new queue, timing, size and risk reservations. Do not copy
  the retained arm's later fills into this arm.

If a loss happens before any feasible cancel, avoiding it requires an
admission/posting-time comparison. Do not invent an earlier cancellation or
pretend a later decision can remove an already-realized loss. Conversely,
past losses are sunk for a decision made later, even though they remain part
of the full entry/retention policy's economics. These are different decision
questions and require different accounting anchors.

For each matched opportunity, the economic question is whether **additional
later value enabled by staying, relative to re-entry, exceeds the additional
early cost and other changed costs**. Later profit alone is insufficient.
Distinguish later fills missed because the replacement is still behind at
the same price from fills missed while absent, at another price or blocked
by inventory. A complete-policy difference alone does not identify queue
as its sole cause.

The final categories should then be:

1. **Compensated queue continuity:** staying earns incremental later queue
   value that more than pays for its early adverse exposure, relative to the
   specified feasible alternative.
2. **Avoidable early loss:** re-entry avoids the early loss and preserves
   enough later value to beat continuous exposure.
3. **Uncompensated exposure:** later benefit exists but does not repay the
   early cost, or no useful later benefit exists; distinguish those subcases.
4. **Unidentified/censored:** the alternative, subsequent life or queue
   explanation is not observed reliably.

“Necessary” is conditional on that specified alternative, not all possible
trading policies. Cases must not be chosen because hindsight reveals a
favorable bad-then-good sequence. Use a declared action population and rule,
complete day/market/status accounting, and no horizon/threshold search.

## Handoff and scope

The pending saved-data headroom audit can still quantify suffix trade-offs;
it must not present itself as answering the whole-life queue hypothesis.
Link each first-anchor class to its full recorded prefix and end cause, and
make the anchor distinction explicit in the same combined receipt. This
cheap clarification is within its existing scope; do not build another
standalone classification variant merely to answer the USER's correction.

Actual queue necessity requires the matched re-entry/admission comparison
above. The development session should prepare its concrete rule, controls,
input coverage and runtime projection for the next combined review. This is
design preparation only: it does not silently expand the current saved-data
no-replay scope or approved 2.50 core-hour line. No new launch or resource
approval is inferred from this conceptual clarification. Other-session
implementation and frozen result artifacts remain untouched.
