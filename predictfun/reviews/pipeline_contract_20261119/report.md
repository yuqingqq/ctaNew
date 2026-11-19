# Prediction → value → controller contract

Implemented for the user's 2026-11-19 request to work on the learned pipeline.
This is development, not a review-only change and not a fitted-model result.
The existing simulator and score-variant implementations are unchanged.

## What changed

`pf_pipeline_contract.py` supplies three separate interfaces:

- An order-conditioned joint forecast of affected quantity and signed movement,
  with no-fill mass, explicit provenance and information timestamps. The
  distribution interface does not prescribe another historical scenario bank.
- An economic mapper retaining `E[q * movement]`, exact bid/ask signs, gains,
  ordinary losses and severe excess. Cancellation uses only removable fills;
  unavoidable fills do not become cancellation benefits. Identity, endpoint,
  reference price, execution scenario and continuation must match exactly.
- A local controller returning one post/wait or pull/keep action. Costs,
  continuation, other-order effects and inventory adjustments are explicit,
  target-bound economic terms. Pending actions cannot be issued twice; pulling
  releases no reservation. There is no fixed cooldown. Exact ties preserve the
  current state. The caller must revalue after each accepted portfolio action.

The old learned one-second/six-rung artifacts cannot be reused as predictions
for the corrected five-second JOIN target. There is no silent conversion of
their horizons, labels or populations. The saved checks use birth mid as a
causal algebraic reference, not an invented current book mid.

## Verification

Primary receipt: [saved_check_v3.json](saved_check_v3.json). Sources were developed
on `5578d96f684d472b04ed25325295a9a2a6989b2e`; exact new-source hashes are in the
receipt. The corrected scenario producer is pinned to
`a2f13f37d58d546b949bb4919725739978a47c76` and its saved source hash is verified.

- **16 synthetic tests pass**, including favorable/adverse joint dependence,
  complementary sides, no-fill probability, severe tails, action/cost signs,
  unavailable forecasts, pending reservations, capacity shared across successive
  posts, and exact identities/timestamps. Planted invalid distributions, future
  information, mismatched targets, corrupt saved artifacts and malformed fill
  ledgers are refused. Realized accounting forecasts are refused by the controller.
- **One consumed day, Oct23:** 30,466 original neutral orders. Per scenario:
  30,088 POST labels checked (including 1,289 refused posts with zero value),
  28,650 first-live KEEP-minus-PULL labels checked; 149 already structurally
  cancelling orders, 354 missing endpoints and 24 stale paths retained as
  statuses, alongside the refusals. Every original order is accounted for in
  each action population.
- **M0/M50/M250, A58/L37/N100:** 58,738 accounting identities per scenario,
  176,214 across the three scenarios. Maximum discrepancy from saved engine
  values is below `1e-7` cents in every scenario (observed about `9.1e-13`).
  POST agrees with full order value; prefix plus retained suffix agrees with
  KEEP, using the simulator's declared cancellation boundary convention.
- Source as-of: corrected scenario ledger **Nov18 20:28:18.245546 UTC**;
  underlying KEEP/PULL ledger **Nov18 02:53:31.517340 UTC**. M0/M250 use their
  already re-simulated branches in that underlying ledger, as the scenario
  report does; they are not re-timed M50 fills. Hashes of receipts and inputs
  are recorded. Final run: Codex slot, **17.62 s, 317 MiB**, exit 0.

Attempt 1 stopped on a checker wiring error: M0/M250 were sought in the scenario
file rather than the underlying clock branches. [attempt1.json](attempt1.json)
records it. V2 corrects that source mapping; V3 adds target-bound economic terms
and their refusals plus a saved-artifact corruption control. Earlier receipts
remain provenance; V3 is the current check.

## What this does not establish

No predictor was trained. These are oracle accounting identities and synthetic
controller tests, not forecast skill, a policy replay, profits or validation.
No raw tape, untouched day, full training or full tape experiment was opened.
No new performance/null comparison was made; no confidence interval is claimed.
This is an interface gate, not a substitute for the full simulator checklist.

The new controller is a single marginal decision interface, not yet connected to
the sequential replay. WAIT currently means no entry until the shared endpoint;
it is not an estimate of optimal later entry. Queue continuation, inventory costs,
client-state feature generation, interaction between orders and the learned
forecast adapter still need implementation/integration. A common endpoint does
not by itself eliminate churn. Real match-time uncertainty, cancel refusals and
order-level fill validation remain simulator limitations.

## One next priority

Build the **causal feature/label exporter on the corrected neutral identities**,
reusing the rich market/order features, preserving statuses and no-fill rows,
and reconciling sampled action labels against the kernel. Then freeze a
chronological consumed-day prediction benchmark before tuning. The complete
stage sequence and gates are in
[the pipeline handoff](../../handoffs/20261119_prediction_pipeline.md).
