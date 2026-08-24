# Iteration 005 — queue-action-conditioned harmful-flow protocol

**Status: FROZEN BEFORE FIT OR REPLAY at 2026-08-24T11:50:00Z. Research only.**

## Hypothesis

The v5 harmful-flow classifier was trained on a hypothetical fresh five-share
order placed behind the displayed touch at every decision.  The cancellation
policy instead acts on an aged order generation whose displayed queue may have
been consumed, whose remaining size may be below five shares, and whose side
may increase or reduce the current inventory.  Training on the action actually
available to the queue-realistic policy should improve harmful-fill selection
without increasing inventory or cancel traffic.

## One structural change

Add `QR_CANCEL_QACT_X_SKEW`.  Its placement, five-share size, five-share skew
band, q>0.5 action threshold, assumed cancel-effective latency, false-to-true
edge arming, inventory-reducing protection, and cancel-and-hold lifecycle are
identical to `QR_CANCEL_HOLD_X_SKEW`.  Only the harmful-flow model and the
population on which it is learned change.

The behavior/reference trajectory is `QR_SKEW_ONLY`, replayed with the frozen
queue-realistic rules and no cancellation.  At the existing exact-event 10 ms
decision clock, a row is eligible only when that reference trajectory has:

- a live `JOIN_EXISTING` order;
- positive remaining resting size;
- an inventory-increasing (not inventory-reducing) side; and
- valid point-in-time PM and HF state under the existing staleness/gap rules.

This reference path is independent of both the old v5 model and the candidate.
It is an explicit shadow counterfactual, not a claim that candidate and
reference inventories remain identical after cancellation.

## Frozen feature and target contract

Retain the 63 v5 exact-event PM/HF features and append exactly six
point-in-time behavior-action fields:

1. `actual_queue_ahead_log1p`;
2. `actual_resting_fraction_of_quote`;
3. `actual_order_age_ms`;
4. `actual_filled_fraction_of_quote`;
5. `maker_signed_inventory_quotes`; and
6. `absolute_inventory_quotes`.

The only fitted cells remain BTC H=50/L=25 ms and ETH H=250/L=100 ms.  The
label follows the reference order's **current generation**.  A cancellation
can prevent only fills of that same generation at or after decision+L and no
later than decision+H.  A touch change, gap reset, or full-fill repost ends the
generation naturally.  Each prevented tranche is valued by the existing
unlagged receipt-time PM midpoint five seconds after that fill:

`gross_cancel_delta = -maker_sign * (mark_mid_5s - order_level) * shares`.

Maker rebates and liquidity incentives are zero.  Rejoin/hold opportunity cost
is deliberately outside this static label and is measured only by the stateful
policy replay.  Features never receive a future fill, markout, generation end,
or label field.

## Frozen fit and signal

- training days: August 20/21/22, 2026, one earliest window per coin/day;
- visible development days: August 23/24, one earliest window per coin/day;
- model: the pinned v5 LightGBM binary parameters, no early stopping or tuning;
- fit population: training rows with a latency-preventable, nonzero-value
  same-generation fill;
- label: gross cancel delta >0;
- weight: absolute gross cancel delta in cents;
- signal: q>0.5, with q evaluated on every eligible reference row;
- ineligible reference state: false signal for that side, so it cannot create
  a cancel and may clear an existing hold;
- no probability threshold, H/L, skew-band, feature, or hyperparameter sweep.

The candidate schedule is therefore a deterministic shadow-policy schedule.
It can be implemented later with a parallel no-cancel state tracker, but this
repository remains offline research and adds no venue or order interface.

## Frozen model gates

All must pass before the model is considered established:

- positive value-weighted Brier skill versus training weighted prevalence;
- positive realized gross cancel value on each visible development day;
- positive selection gain versus the train-selected always/never-cancel
  constant on each development day;
- positive value versus the old v5 q>0.5 model, rescored on the same eligible
  rows, on each development day; and
- eligible-row cancel fraction strictly between 2% and 98%.

The stateful replay runs even if a model gate fails, but failure makes the
candidate non-promotable.

## Frozen policy comparisons and adoption bars

The unchanged `QR_CANCEL_HOLD_X_SKEW` is the incumbent and unchanged
`QR_SKEW_ONLY` is mandatory.  For each coin, adoption additionally requires:

- candidate PnL delta versus incumbent >0 on both development days;
- candidate two-day mean PnL > `QR_SKEW_ONLY`;
- candidate two-day mean terminal absolute inventory <= incumbent;
- effective cancellations and cancel/repost traffic <= incumbent; and
- action-trace parity, parent parity, all-false parity, determinism,
  reconciliation, model receipt, feature-as-of, and artifact checks all pass.

All five days are development context.  No result is decision eligible.  A
candidate surviving these gates must be frozen unchanged for new independent
forward days; the two visible days can never be relabeled as forward.
