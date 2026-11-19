# Prediction, value mapping and controller — 2026-11-19

User request: work on the prediction → value mapping → controller pipeline.
This session owns the new `pf_pipeline_contract*` files. Existing simulator,
score-variant work and other-session changes remain owned by that session.

## First deliverable: an executable shared contract

The corrected simulator's neutral JOIN experiment and the old learned pipeline
are different experiments. JOIN posts 30 shares at the known best, with an
absolute endpoint five seconds after submission. The old joint-moment/GAIN-LOSS
models use a rolling one-second endpoint, a six-rung portfolio and older timing
assumptions. Their fitted artifacts cannot be presented as forecasts for JOIN.

Build a small, independently testable interface before another model fit:

1. **Prediction:** an order-specific joint distribution of affected fill quantity
   and signed endpoint price movement. Quantity is ALL fills after activation for
   POST; for KEEP versus PULL it is only the fills a cancellation can still remove,
   after its scenario-assumed landing. Keep no-fill probability. A weighted set of
   joint outcomes is an interface, not a new historical donor-bank estimator.
2. **Mapping:** `E[value retained] = side*(reference_mid-price)*E[q] + E[q*move]`.
   Never replace the second term with `E[q]*E[move]`. Report expected gain,
   ordinary loss and severe excess without clipping the tail. Same absolute
   endpoint, continuation rule, order identity, decision time, reference price
   and execution scenario must agree from forecast to action. Cancellation
   benefit is the negative of the removable retained value, not the entire
   order's value. Costs and portfolio/continuation adjustments are explicit
   action-minus-baseline terms supplied by the economic layer.
3. **Controller:** local POST/WAIT or PULL/KEEP, side and order specific; zero
   advantage preserves the current state. No fixed cooldown. Pending orders
   cannot be resubmitted/cancelled twice and a pending pull releases no inventory
   reservation. Only one marginal decision is returned per call: the caller must
   update/revalue the joint portfolio before another action. This is not yet a
   sequential multi-order simulator integration.

Absolute endpoints fix inconsistent comparisons; they do not prove that five
seconds is the best horizon or solve continuation value. WAIT in this first
local adapter means no entry until the shared endpoint, not optimal later entry.
Continuation beyond that endpoint is absent only in the explicitly named JOIN
contract. A full controller needs a declared common continuation policy for both
branches and client-known queue/inventory state. No new arbitrary switching
penalty, confidence threshold or timer is introduced.

## Predeclared verification (before reading numerical results)

- Synthetic positive cases and known-bad refusals: bid/ask signs, adverse
  fill/price dependence, no-fill mass, severe tail, exact cancel boundary,
  unavoidable fills, future information, wrong identity/endpoint/scenario,
  pending cancellation reservation, duplicate actions and invalid probabilities.
- Bounded integration check on saved Oct23 neutral orders and corrected scenario
  v6 outputs at M0/M50/M250, A58/L37/N100. Verify receipt hashes first. Every row
  remains in the counts; refused posts have zero POST value; unavailable labels
  and unavailable cancellation decisions remain statuses. One first-live cancel
  decision per order (+100 ms); no duplicated-action performance estimate.
- Reconstruct POST and KEEP-minus-PULL components from the scenario's existing
  fills, compare to the saved order value and the direct prefix/suffix sum.
  Realized outcomes are **oracle accounting inputs only**, explicitly refused by
  the controller. They are not predictions, training results or policy returns.
- Existing receipt provenance is retained. No new raw tape read, full replay,
  model fit, untouched-day access, null experiment or profitability claim.

## Then build and test the learned system

1. Export causal market/order inputs on these exact neutral identities, with
   timestamps and status completeness; reconcile sampled labels to the kernel.
   Reuse the richer market representation (book/flow changes, scale, volatility,
   fair-price response) rather than silently shrinking it to the fair-edge score.
   Record queue age/ahead, remaining quantity, depth/rung, pending operations,
   inventory and time to the common endpoint. Audit every raw-data hour boundary.
2. Freeze a chronological consumed-day benchmark (Oct23–25 train; Oct26
   calibration only when required by a declared candidate; Oct27 development
   evaluation). Separate bid/ask estimates, pinned LightGBM parameters, explicit
   feature-only/mechanical and direct-value controls. Count all candidates.
   Do not reuse old one-second labels or fit economic cutoffs on Oct27.
3. Judge fill/quantity calibration, fill-weighted movement, severe loss amounts
   and net value separately, by side/day and matched participation (at least 200
   declared matched null draws). Gate mapping on amount calibration and economic
   ranking, not fill AUC alone. These days are consumed, not validation.
4. Integrate the frozen forecast adapter into a client-state sequential replay,
   including post/wait continuation, queue retention, partial fills, reservations,
   shared volume and inventory costs. Report posting-to-cancel reversals and their
   attribution, quote uptime per side/rung, spread/markouts, inventory and terminal
   value separately. Use the same timing scenario bank; do not average unidentified
   timing scenarios into a purported measured probability distribution.
5. Only after those gates, freeze the whole pipeline and evaluate at least five
   later untouched complete days. Oct28 is not untouched. No live orders here.

The immediate next priority after the contract check is the causal feature/label
exporter on the corrected neutral population, not hyperparameter tuning.
