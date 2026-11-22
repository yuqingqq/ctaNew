# Queue priority and sweep exposure — next diagnostic proposal

2026-11-22. Prepared against **82f4a04c2a8219aa383134be31148f3155f8bb15**.
USER direction: earn repeated spread by acquiring and preserving queue priority,
tolerate ordinary adverse selection, and control persistent one-way exposure.
The USER suspects queue disadvantage explains most of our losses. This is a
testable hypothesis, not an established attribution.

This note updates research prioritization after review 58d20752. The single
recommended next experiment is the queue-priority sensitivity below. The
previous broader-pressure challenger remains preserved and stopped; it is
not to run in parallel or resume automatically. The later suggested generic
value-model rebuild is not the next task. This document is a proposal, not
an experiment freeze or authorization to exceed the existing compute cap.

## Why this test

`predictfun/research/pf_branch_engine.py:151–174` distinguishes same-price
prints from trades through our limit price. Same-price volume first consumes
the simulated external queue ahead. Through-price volume can fill our order
without that ahead-volume deduction. Thus improved priority might add ordinary
fills before a later sweep, but can also add adverse fills or change inventory
and subsequent quotes. Larger size preserves a partial order's priority while
exposing more quantity to that same sweep. Neither size nor early posting is
automatically profitable.

The queue/price audit does not identify this effect: only 39 valued filled
front-at-touch states meet its pooled common cohort, with remaining-time and
day/side composition differences. Comparing those observed groups cannot
establish the value of giving the SAME policy better initial queue priority.
The full pressure policy still loses in the accepted four consumed days.

## ONE experiment: QUEUE-PRIORITY-SENSITIVITY-v1

After a separately approved resource amendment, development should freeze
one paired sensitivity on the existing 19 complete Oct 24–27 markets, keeping
the original 96-market coverage/status census. No new day, fit, price ladder,
size sweep, pressure threshold, inventory parameter or trained sweep guard.

Two arms only:

1. **CURRENT_QUEUE:** the accepted FULL_11/Q5 pressure ladder, nominal two
   100-share levels per side, existing inventory/reservation controls. It must
   reproduce its accepted per-market ledger, commands and queue records.
2. **FRONT_AT_ACTIVATION:** the identical policy and latency conventions,
   except an accepted new order starts with zero external queue ahead at its
   activation. Apply this once per new generation, not every snapshot/tick.
   Preserve own-order price/time priority, partial remainder, public-volume
   conservation, post-only refusal and strict removal timing. Do not alter
   the frozen engine file; use a declared diagnostic wrapper and identify
   the mode in source/artifact identity. No future event or outcome selects
   which generations get this treatment.

FRONT_AT_ACTIVATION is hypothetical execution sensitivity, NOT an executable
policy, a guaranteed monetary upper bound, or exchange-ground-truth priority.
The unchanged-external-flow assumption is especially strong here: fills
diverted from existing makers may change real future book behavior, which
this simulator does not model.

Run both complete feedback policies. Keep decision rules and exogenous tape
fixed, **not** an identical pre-recorded submission schedule: different fills
must feed the actual notices, reservations, inventory, vacancies and later
decisions in each arm. Otherwise the counterfactual could violate capacity
or silently attribute inventory effects to queue rank. Report the total
policy effect; do not call it a pure per-order avoided loss.

## Measurement and safeguards that matter

- Primary: absolute END cash-plus-marked-inventory and FRONT minus CURRENT,
  per day/market and pooled. Mark remaining inventory; never score only
  completed winners or imply settlement/live profitability. Preserve the
  current declared cost convention and disclose it.
- Execution edge, positive/negative eligible 1-second markout, filled shares,
  c/share and total value with the existing common cutoff; report denominators.
  Retain 5/30-second markouts where supported as horizon diagnostics with
  explicit censoring, not alternative winning endpoints.
- Attribute fills to same-price versus through-price public events using the
  engine's exact predicate and event identity. Report each group's positive,
  negative and net values, quantities and repeated fills. These categories
  are explanatory after execution, never a future input or a claim that every
  same-price fill is benign/every through-price fill is harmful. A multi-level
  event must conserve and count public volume correctly.
- Inventory paths, pending reservations, two-sided participation, order
  lifetime, first-review churn and cancellation reasons. Separate the impact
  of changed fill mix from simply receiving more total fills.
- Reuse accepted timing/accounting controls. Add small fixtures showing
  same-price queue consumption, a through-price sweep, partial fills retaining
  priority, one-time initialization, delayed notices and the changed inventory
  feeding subsequent decisions. A changed control ledger, excess filled
  volume, illegal reservation or future-conditioned priority blocks results.
  Do not require fixes to the two deferred reporting issues from 58d20752.

## Interpretation and stopping

Declare the reading map before outputs. Show separate predicates for both
arms' absolute END signs, paired END and 1-second differences by day, and
changes in the same-price/through-price fill mix. Four consumed days support
point estimates only, not validation or uncertainty intervals.

- If front priority adds positive net exposure and materially improves
  complete-policy economics, it supports further work on how to ACQUIRE and
  PRESERVE priority. It does not prove an implementable method, nor that queue
  explains most of the observed deficit; report the measured change.
- If it adds fills but fails to improve economics, stop the hypothesis that
  simply obtaining front priority rescues this fixed strategy. Do not tune
  an intermediate queue advantage until it happens to win.
- If it improves results but the strategy still loses, state that queue
  helps but is insufficient. Remaining through-price and directional loss is
  the next economic limitation, not a reason to claim success.
- If mixed or unsupported, say so. No automatic additional variant, model
  fit, classifier or new data request.

An executable follow-up, only if justified, would preserve valuable resting
orders while responding to decision-time evidence of persistent harmful
flow. It must distinguish market movement from inventory-worsening exposure
and cannot promise avoidance of a sweep whose warning arrives after the
cancel deadline. Do not weaken emergency/data-integrity controls or delete
the structural rule solely because it accounts for many cancellations.

## Ownership and resources

Development implements; this thread reviews one bundled code/results receipt.
No replay was launched and no economics were newly measured for this note.
The original line is about 1.976/2.0 core-hours after the prior review; that
remaining budget is not sufficient authorization for this new counterfactual.
The earlier proposed 2.12-hour amendment concerned another experiment and
was not approved. Do not transfer it silently.

Before any run, obtain the USER's approval for this specific diagnostic and
its resource ceiling. Proposed maximum: **0.25 additional core-hours for the
whole development batch, cumulative line ceiling 2.25 including review**.
These are proposed caps, not measured runtime estimates. Freeze a write-once
pilot/projection gate: the first two complete markets per date, reuse them
in the full result, project all remaining work plus review reserve, and stop
if it cannot fit. No outcome-based pilot continuation or repeated pilot.
Keep all controls/population and use the authorized run_light slots.

Success for this round is a reliable answer about queue sensitivity that
changes the next economic decision. It is not a perfectly polished checker
or a declaration that the market-making strategy is now profitable.
