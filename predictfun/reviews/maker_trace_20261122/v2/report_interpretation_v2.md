# MAKER-TRACE — interpretation v2 (supersedes the interpretive statements of `../report.md` and `report.md`; numbers unchanged)

2026-11-23, session aea5bfb9. This answers the Codex review [`20261123_maker_trace_review`](../../automatic/20261123_maker_trace_review/report.md) (`567f741d9`), findings 1–2 (both hold).

**Rules for this file:**
- Frozen receipts are never edited. This file supersedes their wording only.
- **Horizon, clock, fee convention, cohort and denominator are named for every figure.**

## R4 — corrected reading (finding 1)

### What the headline figures are

- **Depth, sweep and window-phase figures** are **net P&L to resolution per filled share** (`net_res / sh`): the settlement value of each filled leg.
- **The taker "+1.21 c" figure** is a **gross 1 s markout** (`v_1s / sh_1s`). Its fees are not deducted.
- **Clock:** markouts start at the settlement-ID creation time (an upper bound of the match). Outside BOOK, "depth" is measured against the book at that time. Depth is therefore a historical proxy, not an observed posting decision.
- **"Sweep":** a record with several maker prices, not a size-matched comparison.

### Same fills, different horizons

Codex's post-hoc diagnostic (`economic_probe.json`, not a declared candidate) uses identical rows that have a valid 1 s mark:

| Cohort | Shares | Gross 1 s | Net 1 s (fees at that mark) | Resolution net, same rows |
|---|---|---|---|---|
| Top 10, maker at the best (d0) | 10.7 M | +0.36 c | +0.36 c | −0.16 c |
| Other makers, maker at d0 | 25.2 M | +0.16 c | +0.16 c | −0.14 c |
| Top 10, taker | 3.8 M | +1.21 c | +0.60 c | +0.72 c |
| Other makers, taker | 12.8 M | +0.16 c | **−0.42 c** | +1.01 c |

### Corrected statements

- **Withdrawn:** "top-of-book quoting does not pay / fills at the best lose after adverse selection".
  - At 1 s, d0 maker fills **gain** (+0.36 c for the top 10).
  - The loss appears between the short horizon and resolution, i.e. in inventory and resolution.
  - So the depth gradient is a statement about **settlement value**, not about short-horizon adverse selection at the fill.
- **Withdrawn:** "the field makes all of its profit on taker legs" as a short-horizon edge.
  - The field's taker legs are +0.16 c gross but **−0.42 c net of fees** at 1 s; they are positive only at resolution.
  - Role attribution is **not** independent strategy profit, because taker legs share inventory with the maker legs.
- **Stands:** the resolution totals.
  - Top 10: +35,586 = round trip +130,789 − residual 72,444 − fees 22,759.
  - Field: +110,136. Takers: −427,378.
- **Stands, as descriptive associations:** the sweep, early-window and depth associations at resolution. They are not the value of a quoting policy.

## R4 — "consistency comes from flat inventory" (finding 2): withdrawn

The code selected the 3 addresses positive on all 4 days (post-hoc) and reported their aggregate residual **P&L**. It never measured their **position path**.

Their recorded-fill positions reach about **152–504 shares** per address and day (review), and their end-of-day positions are nonzero.

**Corrected statement:** three pure makers had positive daily P&L and small aggregate residual contributions on these days. Why they were consistent is not identified.

The Bernoulli(END mid) null tests only the END-to-resolution residual, not inventory losses in general.

## Decision paths (`report.md`, decision section): labels corrected (review next-priority items 3–4)

- **Take rates** are **conditional on the address's active span**, which is built from its own realized first and last fills ± 30 s. That span is outcome-selected, so the rates are conditional estimands, not unconditional action rates.
- **Decision-time features** are taken at `match bound − L`, with L ∈ {0, 50, 250} ms. They can fall **after** the real decision when the match-to-ID lag is longer. Binance event time is a venue-state reference, not what the maker had received.
  - The associations are **assumption-sensitive**: momentum aligned with a take, inventory-reducing takes, and takes soon after own fills.
  - Rank 1's flip at L = 250 shows exactly this sensitivity.
- **"Posting at hi"** is the lifecycle's latest possible posting time, which can be the first fill. It is **fill-relative**, not a known posting-decision input, and the 2.4–4.5 c offset is not interpreted.
- **At-fill quote centring and the skew coefficients** describe filled orders only (censored). They are not the latent policy.
- **Pulls on adverse BTC moves** (ranks 1–2; p = 0.002 vs a phase-matched null) are an association of possible-cancel removal windows with BTC moves. Cancels are anonymous, and the windows are bounds.

**Pending (next-priority packet):**
- horizon-aligned FIFO completed-close attribution by role pair;
- an identifiability control with identical public fills and different hidden commands.

## Check semantics (review finding 3): v2 disposition

| Change | Review's view | Effect |
|---|---|---|
| Checker side: 0.01-share display tolerance, violated cells unknown, empty side = size 0 | matches its required dispositions | — |
| Lifecycle side: same-settle-ms groups are no longer voided (new `fifo_tie_conflict` status) | treats the v1 exclusion as a declared conservative rule | T2 is **5 / 16,433** in v2 vs 3 / 15,900 in v1; R3 for the top 10 is equal to 3 decimals; R4 rows are unchanged |

The v2 lifecycle change is disclosed here for the next review to rule on. Both readings stay `supported`, and the v1 receipt is unchanged.

## Latent receipt-earlier-than-UUID case (finding 4)

It is not present on current rows (`fill_hi_rule` passes). It will be fixed when the lifecycle is next touched; it does not justify a rerun on its own.
