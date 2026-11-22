# Maker trace reconstruction — evidence before imitation

**Governing USER clarification:** [coverage amendment](coverage_amendment.md).
Aim to explain the bulk of observed behavior using constrained estimates and
sensitivity checks. The eight-example packet is the initial check, not the
final deliverable or a reason to stop at missing exact timestamps.

2026-11-22. USER: simulate their whole queue trace, then inspect posting and
pulling decisions. Design-only handoff; no new data replay, fit or strategy
change has run. Source inspection at c57c6aa54ddb2d7a62e3a93cfbe0252cd683f971.
Development owns implementation. This supplements the current frozen
long-value discrimination experiment; it does not amend its inputs or gates.

## What we can learn

Reconstruct each maker's observed execution history and the **set of queue
histories consistent with it**. Use that to distinguish demonstrated order
persistence, possible entry/removal windows and unidentifiable actions.
Then connect the execution history to both sides' turnover and inventory.

This is an inverse problem. A simulator can produce a plausible history;
matching the fills does not establish that it is the maker's actual history.
The earlier study's next-trade GONE/STAYED labels must not be imported as
observed CANCEL/KEEP commands.

The prior review demonstrated the problem with the actual extractor:
`predictfun/handoffs/20261113_learn_pull_model_integration_review.md`.
A maker with 5 shares can fill completely before another maker trades.
Alternatively, it can fill 5 shares of a larger order and cancel its remainder
while another maker replaces the depth between snapshots. Aggregate depth
and the observed fill sequence can agree in both histories. They imply
different actions. A later fill by the same address also does not prove
the same order survived. Order hashes improve the evidence but do not by
themselves supply original quantity or command timestamps.

## First reviewable deliverable: a small trace-identifiability packet

Before implementing a broad imitation simulator, reuse the existing order-hash
maps and inspect the raw payload schema for **one already-consumed supported
market**, with up to eight order examples from two maker addresses. Include
repeated same-hash fills and an unresolved disappearance; this is a deliberately
small feasibility packet, not an economic sample or a coverage-rate estimate.
Record the deterministic selection and its purpose before inspecting outcomes.

1. **Inventory available fields.** Check the raw matches and book payloads,
   not only columns retained by `pf_orders.py`. Establish whether any field
   actually carries original signed quantity, creation/submission time,
   cancellation, expiry or remaining size. `executedAt`, receipt time and
   executed amount must not be renamed as those fields. Preserve exact source
   keys and timing precision. Use full identities where available.
2. **Construct the observed ledger.** Maker + market + side + price + order
   hash; each execution's quantity, trade identity, event-time evidence and
   receipt time. Repeated partial fills and observable opposite-side/taker
   fills belong to the same market ledger. Starting inventory, other accounts
   and hedges remain unknown unless independently observed.
3. **Construct conservative lifecycle/queue bounds.** An execution proves the
   order was executable then; its first fill is not its submission. Cumulative
   fills give a size lower bound, not necessarily original size. Last fill is
   not cancellation. Absence outside recorded book depth is unknown, not zero.
   Bound external queue and removal only where complete trade/depth evidence
   and declared priority assumptions allow it. Keep exhausted, cancelled,
   expired, repriced, missing evidence and end-censored as distinct possible
   explanations, without inventing exact timestamps.
4. **Keep clocks separate.** Venue activation/removal intervals are not the
   maker's private decision times. Decision times require an additional
   latency/information-delay assumption. Retain these as ranges/scenarios,
   not our strategy's 10/50/100 ms profile assigned to other makers as fact.
5. **Expose the ambiguity.** For each example, provide an event timeline with
   observed fills/depth, known facts, conditional bounds and unresolved
   alternatives. Where two action histories fit, save both. A compatible
   history is an illustration, not a uniquely reconstructed strategy. If the
   constraints contradict the tape, report it rather than assigning free
   front priority or changing the matching rules until it fits.

Small controls should include the two-history exhaustion/cancellation example,
same signer with a new hash, a genuine repeated partial fill, a price outside
shown depth, uncertain event timing, and a deliberately impossible overfill.
No full mutation campaign or general engine rewrite is needed for this packet.
Reconcile execution quantities and cash exactly; disclose unsupported bounds.

The deliverable is the trace table/timelines, source-field inventory,
identification limitations and a measured cost projection for extending to
the same supported market windows. No candidate fit or profitability verdict.
If action histories remain broadly ambiguous, stop trying to manufacture
exact imitation labels; keep the reconstruction as a descriptive benchmark.

## How it can subsequently inform the strategy

- Compare repeated-fill and two-sided turnover paths using common supported
  horizons, with incremental marked inventory separated from closed turnover.
  Our earlier maker studies mostly cover the first minute; do not claim a
  whole-market trace without reading and reconciling the longer interval.
- Analyze conditions before a bounded action window. Features 100 ms before
  a later trade may occur after the hidden action. Do not use future next-trade
  time, last-fill time or realized outcomes as decision inputs.
- Select any imitation identities from earlier history. The earlier “top
  five” rank is same-day volume, not an ex-ante profitable-maker identity.
- A later causal imitation must earn its own fills and queue priority. A
  factual replay using recorded executions is an accounting benchmark, not a
  tradeable policy. A counterfactual replacing a historical maker must also
  avoid counting that maker's old displayed liquidity and the simulated copy
  twice; a new-entrant test is a separate declared intervention.

The central question remains whether preserving a particular queue position
earns enough later joint value to offset the adverse fills it accepts. A
reconstructed sequence can reveal the mechanism to investigate; only a causal
policy comparison can establish what our strategy could obtain.

## Scope and ownership

Prepare this packet as a separately declared next task after the current
frozen experiment's combined receipt. The USER has requested the direction;
this memo does not cancel the current work, approve new days/live use, or
increase its compute allocation. Reuse saved data and bounded controls first.
Declare the pilot's resource needs before a new replay; do not silently charge
a broad maker reconstruction to the existing 1.25 h discrimination budget.
Bundle material issues and the packet into one development/review round.

Sources read: `pf_orders.py` extraction and `exit_obs`; `pf_learn_pull.py`
specification; ORDERS and LEARN-PULL receipts; the public-order integration
review and the next-trade identifiability review. No historical study was rerun.
