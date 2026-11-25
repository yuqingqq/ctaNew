# TAKE-EXIT FLOOR v1 — a break-even price floor on the maker exit, valued at settlement (declared before any run)

USER (2026-11-25): "We test the limit price in the same days, check the spread capture and settlement pnl separately, and the risk we
carry to settlement".

## Days and population

- **Days:** the five v3 days 20261023, 20261031, 20261101, 20261102, 20261103 — the days the frozen rule was last run on
  (amendment_v3_other_days.md). All are consumed project days; this is development evidence, not validation.
- **Markets:** the 408 complete markets of the v3 census, through the same loader and path builder. The census is recomputed and
  must equal v3's exactly (every status counted). The 20261103 23:55 market stays boundary-censored. Nothing from the sealed block
  (10-28..10-30) or from 2026-11-04 on is read.
- **Binance input:** the v3 change cache, unchanged (no new Binance read).

## Arms

- **BASE:** the frozen rule under STRICT58 — the saved v3 ledgers, not re-run.
- **FLOOR:** STRICT58 with one change in the exit price (below). Everything else is the frozen code: triggers, take size and limit,
  the strict IOC rule, 58 ms take delay, cap ±50, exit size, rung 0 only, the pressure guard, the cross and no-book cancels,
  post-only, ASYNC_FAST10 timing, 100 ms fill notices, END 298 s.
- **PARITY (control):** the FLOOR code with the floor engaged but 1,000 ¢ out of reach, on every 40th complete market in census order.
  It must equal BASE exactly on fills, orders, commands, value, cash, position, signal_log and struct_log (floor-only keys removed).

## The floor (FLOOR arm)

- **Basis:** the client-known open position (fills known at fill time + 100 ms), average-cost accounting with taker fees included
  (maker fills pay 0). Adds average in; reductions keep the average; a flip starts a new basis at the flipping fill's price, with
  that fill's fee pro rata.
- **Long:** the exit ask is posted at max(best ask, ceil(average cost)). **Short:** the exit bid at min(best bid, floor(average net
  proceeds)). Whole cents (tick 1 ¢).
- **Cancels:** the distance rule (more than 3 ticks behind) is measured from that target, so an order resting at the floor is not
  cancelled for distance. A resting exit priced beyond the current floor (for example after an add raised it) is cancelled with
  reason `floor` and re-posted. Cross, no-book, guard and exit-not-needed cancels are unchanged.
- **Takes are not floored:** an opposite-signal take may still reduce or flip the position at a loss (frozen take rule).
- No time limit, no stop, no profit target above break-even. The level (break-even) is fixed here, before any run.

## Valuation and decomposition (per arm, computed, identities checked)

- **Settlement:** each market's resolved outcome from the loader (`yup` 1 = Up, 0 = Down; the loader already excludes markets without
  one; any other value is a status, never a drop). Shares open at END settle at 100 × yup ¢.
- **Mark:** the 298 s book mark as before, for continuity with v3.
- **Round trips ("spread capture"):** FIFO-matched pairs (take → maker exit, or take → opposite take), after the fees of both legs,
  split by how the pair closed (maker exit / opposite take).
- **Held to settlement:** FIFO lots open at END valued at settlement, after their own fees.
- **Total at settlement** = round trips + held to settlement = cash + END position × settlement − fees (identity).

## Risk carried to settlement (per arm)

Markets with a position at END (≥ 1 share and > 0); shares held (total, % of shares taken, per-market p50 / p90 / max); the loss if
every held lot settles against us (total, per-market max); settlement P&L of held lots per market (mean, std, p5, min) and per day;
the share of held shares that won; and mark versus settlement, Σ END position × (settlement − mark).

## Readings (predicates evaluated by the code)

- `floor_better_at_settlement`: FLOOR total at settlement > BASE, and the paired day difference > 0 on ≥ 4 of 5 days.
- `floor_better_at_mark`: the same on the mark valuation.
- `floor_round_trips_higher`: FLOOR round-trip P&L > BASE.
- `floor_carries_more`: FLOOR shares held to settlement > BASE.
- Paired per-market differences: markets better / worse / equal and the top-10 share of |difference| (the replay is path-coupled, so
  per-market differences are noisy). No interval: five day clusters.

## Checks (each with a falsifier where it is a checker)

1. Census equals v3's; BASE mark net per market equals the v3 summary exactly, pooled equals v3_readings.
2. PARITY equals BASE on the listed fields.
3. FLOOR: every maker exit is priced inside the floor recomputed from the ledger's own client-known fills at its submit time.
   Falsifier: the same check on the BASE ledgers must flag violations (it sells below break-even, e.g. the flow page's 46 ¢ exit).
4. FLOOR: the floor logged at each post equals the recomputed one.
5. Accounting identities per market and arm; FIFO gross and residual equal the frozen `fifo` on the same fills.
6. Every complete market has yup in {0, 1}; IOC delays are 58 ms in FLOOR; exits are on the closing side at rung 0; the floor binds
   at least once (positive control).
7. Synthetic selftests of the basis, the fee-carrying FIFO and the floor checker (a known-bad order must be flagged).

## Budget

Run ≤ 900 CPU s with a projection after the first day (cost × 5 × 1.1 ≤ 900, or stop); the round ≤ 0.25 core-h. RSS self-guard at
2,500 MiB under the 4 GiB slot. Outputs write-once.
