# EDGE_LAYER1 — fill quality at a fixed horizon, against book mid

Protocol `edge_l1_v1`. **Frozen 2026-08-22 before measurement.** Coordinator
writes the rules; the research agent measures. Research only, not decision
eligible, no forward-day claim.

## Why the current estimand is being replaced

Every edge figure in this corpus is markout against **settlement**, so a fill at
`t=30 s` is marked at `t=300 s`. That is **hold-to-expiry PnL, not spread
capture**, and on the marginal fills of the policy comparison it implied
**+10.31 ¢/share against a 0.50 ¢ half-spread** — a number dominated by
directional drift.

This protocol measures **Layer 1 only: the quality of the fill itself.** What
happens to the position afterwards is Layer 2 and belongs to
`plans/DA_INVENTORY_STATE_PLAN.md` and `plans/DE_PLACEMENT_POLICY_PLAN.md`.

## Estimand

For a maker fill at level `ℓ`, signed by **maker side**:

```
markout(h)  =  s · [ mid(t_fill + h) − ℓ ]          s = +1 maker BUY, −1 maker SELL

           =  s · [ mid(t_fill) − ℓ ]        ← spread captured at the moment of fill
            + s · [ mid(t_fill + h) − mid(t_fill) ]  ← post-fill drift (adverse selection)
```

**Report both components separately**, not just the sum. Spread capture is
mechanical and known; the drift term is the thing being measured.

`mid` is read from `price_change.best_bid/ask`, **never** `book` snapshots
(p90 6.2 s stale). Knowledge time is `recv_ns`. **No fair-value model** — this is
why Layer 1 stays off the sigma clock.

## Horizon ladder — report all of it

`h ∈ {5, 15, 30, 60}` seconds. **Every horizon is reported.** Adverse selection
grows with `h`, so selecting one after seeing results is tuning, and the shape of
`AS(h)` across the ladder is more informative than any single value.

## The truncation problem — pre-specified, because it is not benign

A fill at `r < h` has no `mid(t_fill + h)` inside the window. Two options and
both are wrong in different ways: **clamping to settlement** reintroduces exactly
the hold-to-expiry defect this protocol exists to remove; **excluding** drops
late fills.

**Excluding is chosen.** But the exclusion is **not random** — it removes
precisely the terminal minute, where `f_r`'s entire dynamic range lives and where
notional peaks then falls 9.5×. So:

- report the excluded count and the **`r`-distribution of exclusions** per coin
  and per horizon;
- state that `h = 60` cannot see the final minute **at all**, by construction;
- do **not** compare `AS(h)` across horizons without noting that each horizon
  sees a different `r`-population. That comparison is the denominator/population
  defect, which has hit this programme six times.

## Decision rule, written before the measurement

Verdict on **btc and eth**; the other five are descriptive. Window-clustered
intervals; day-clustered are not computable and must not be claimed.

- **EDGE_POSITIVE** — `markout(h) > 0` with the interval excluding zero at
  **every** horizon on both verdict coins.
- **EDGE_NEGATIVE** — `< 0`, interval excluding zero at every horizon on both.
- **HORIZON_DEPENDENT** — sign or significance changes across the ladder. Name
  the crossing horizon; that is a finding about how fast adverse selection
  arrives, not a failure.
- **UNDETERMINED** — intervals span zero. Report the width and what `n` would
  settle it. **This is the expected outcome given the pooled edge is
  `+0.173 [−0.251, +0.596]`, and it must not be dressed up as anything else.**
- **VOID** — fewer than 500 fills with a valid `mid(t_fill + h)` on either
  verdict coin at any reported horizon.

## What a positive result would and would not license

A positive Layer-1 markout means **the fill itself is good** — it does not mean
market making is profitable. Layer 2 (inventory carry, terminal residual, the
`r≈60` decision) is a separate accounting and can erase it. **Do not combine the
two layers into a single PnL figure in this protocol.**

## Rules

1. Per coin, never pooled — btc is ~64 % of any pooled denominator.
2. State the population of every denominator, including the per-horizon
   `r`-population created by truncation.
3. `R-DUAL`: report with and without the 0.02 micro class.
4. Gap-touched and tick-change-touched fills are `UNAVAILABLE`, reported not
   dropped.
5. Stamp `fi.provenance()` — the sample grew from three days to four on
   2026-08-22 and published numbers predate that.
6. A control per probe that fails if the test is vacuous: a synthetic fill with a
   known subsequent mid path must return the exact markout, and a zero-drift path
   must return exactly the spread captured.

## Out of scope

Inventory carry, terminal residual, the fair-price model, `ρ`, and any combined
PnL. This protocol measures one thing: **how good is a fill, `h` seconds later.**
