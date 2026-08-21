# QUEUE_AND_TYPE — two tests that can SHRINK the flow model

Protocol `queue_type_v1`. Frozen 2026-08-21 before measurement. Coordinator
writes the rules; the research agent measures. Research only; not decision
eligible; no forward-day claim.

**Both tests are designed so that a negative result deletes work.** Neither is a
validation of the flow model. C1 measures the thing that actually gates the
decision; C2 can remove a whole layer.

## C1 — do cancellations narrow the fill bracket? THE BINDING CONSTRAINT.

`BACK_DISPLAYED` grants **no cancellation credit**: it assumes every displayed
share ahead of a resting order *trades* before we do. Real queues shed depth to
cancels, and every cancel ahead of us moves us up. Measured bracket today:

```
15 s any-fill    front    back-displayed   width
btc              94.6%        76.9%        17.7pp
eth              84.8%        55.6%        29.2pp
hype             71.3%         2.4%        68.9pp
```

No arrival model narrows this — the uncertainty is queue position, not rate.

**Method.** Decompose per-level displayed-size decreases in `price_change` into
a trade-driven component (matched to `last_trade_price` at that level and time)
and a residual attributed to cancellation. Report at the touch, per coin, and by
moneyness. Then recompute the lower bound crediting cancellations ahead of the
action, and re-derive the bracket.

**MANDATORY VALIDATION, and the result is VOID without it.** The decomposition
must reconcile: for every level, `size(t+) - size(t-)` must equal
`new_or_replenished - traded - cancelled` to within **1 %** of gross level
turnover. Report the reconciliation residual per coin **before** any bracket
number. A decomposition that does not reconcile is measuring something else, and
the residual is the only thing standing between "cancellation" and "everything
we failed to model" — including hidden liquidity, which this tape cannot
identify.

**Decision rule.**

- **MATERIAL** — cancel-credited bracket width falls by **>= 50 %** on btc AND
  eth, with window-clustered intervals. The pessimistic bound is then not the
  binding constraint it appears to be, and queue modelling is worth continuing.
- **IMMATERIAL** — width falls by **< 20 %** on both. Then displayed depth ahead
  really does trade through, the bracket stands as measured, and **this is close
  to the programme-ending reading**: fill is not determinable from data we can
  collect. Say so plainly rather than softening it.
- **PARTIAL** — anything between, or a split between the two coins. Name which
  coin survives; authorise nothing beyond it.
- **VOID** — reconciliation residual exceeds 1 %.

Thin coins are reported but do not enter the verdict: at 90 % single-actor share
and a 69 pp bracket, hype cannot discriminate.

## C2 — is there ANY market self-excitation once micro is a type, not a deletion?

Branching currently selects 0.40-0.55 at a **censored** half-life. U1 established
one websocket event is one whole taker order, so this is not order slicing. But
the micro actor is 16.3 % of events (2 % btc to 90 % hype) and **A1 failed
bidirectionally at ~2x within 0.25 s** — the same timescale. So the apparent
self-excitation is plausibly micro<->market cross-excitation.

A1's failure is exactly why V4 models micro as a **type** rather than deleting
it. This test uses that.

**Method.** Bivariate Hawkes on types `{MICRO_002, MARKET}` in baseline
operational time, estimating the 2x2 branching matrix: `market->market`,
`micro->market`, `market->micro`, `micro->micro`. Use the extended half-life
grid already in code. Per coin; never pooled.

**Decision rule, on `market->market` specifically.**

- **DELETE_HAWKES_LAYER** — the `market->market` interval includes zero on btc
  AND eth. There is then no market self-excitation to model, and the Hawkes
  layer is removed from the specification rather than reserved for day 10.
- **RETAIN** — it excludes zero and exceeds 0.10 on at least one of btc/eth.
- **CENSORED** — the selected half-life sits on either grid boundary. Report as
  unresolved and state the grid needed; do NOT read a branching value off a
  censored fit.
- **UNRESOLVED** — anything else.

**Underpowered defaults to DELETE**, not to retain. A layer must earn its place;
the burden is on the layer, not on the reader.

## Rules for both

1. Per coin, never pooled — btc is ~64 % of any pooled denominator.
2. State the population every ratio's denominator is drawn from. Six instances
   of that defect so far, three of which read as findings.
3. Window-clustered intervals only, with the standing caveat that they miss
   day-level factors and therefore **understate** uncertainty at two days.
4. Every probe carries a control that fails if the test is vacuous.
5. `--selftest` green before any run; code must pass
   `assert_protocol_conformance()` if it touches declared constants.
6. Report the excluded set beside the retained one.
7. Do not narrate a mechanism that was not measured.

## Explicitly out of scope

Profitability, maker edge, adverse selection. The sign is measured and
UNDETERMINED at two days and no test here changes that.
