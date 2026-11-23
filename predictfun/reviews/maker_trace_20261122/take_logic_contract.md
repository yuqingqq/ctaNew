# TAKE-LOGIC v1 — contract (frozen before computation)

2026-11-23, session aea5bfb9.

**Mandate:** the USER, verbatim: "I don't want something unclear like they take when btc turns against their inventory, I need also some solid proof, like what's the inventory, and is it like the end of 5m market? I also want to check it happens one time like they sell when btc rises or it happens a lot. I want to see the clear logic not just description of what happened".

**Budget:** the USER approved +0.05 h. The task cap goes 1.85 → **1.90**; line 5.25 is unchanged.
- This measurement: ≤ 0.040 core-hours for build, tests and runs, plus a 0.010 share of the review.
- It is bundled with LOGIC-CHAIN CORRECTION v1 (one combined review).

**What this is:** a decision-time population (Reliability rule 1), not outcome-selected takes. Four consumed days (Oct 24–27), so:
- point estimates, no intervals (G < 5);
- p is a circular-shift tail fraction, not independent-day confidence;
- descriptive of these markets — no fit, not a validated rule.

## Population and event

**Unit:** maker-second. For each ex-ante top-3 maker (ranks 1–3; `chain_context.top3`) and each of the 19 BOOK markets (`chain_context.book_markets`), every instant `t = market_ws + k·1000 ms` from `market_ws` to `market_we − 1000`.
- This is the market's whole span, **never** the maker's active span.
- A pair with no fills enters with inventory 0.

**Event:** a take by that maker. These are the rows of `v2/chains/take/takes.csv` (all 4,147, any status), assigned to the instant t with `t < x_lo_hard_ms ≤ t + 1000`, i.e. the earliest possible match lies in the next second.
- An event-second counts once; its takes are also counted.
- **Event type**, from chain_take's saved class (the column that reproduces summary `cls_*`):

| Event type | Classes |
|---|---|
| reduce | reduce_partial / full_close / reversal_reducing / overshoot |
| add | add / from_flat |
| other | no_change / unclassified |
| timing_sensitive | its own type |

## State at t (known at t; primary D = 0, sensitivity: state at t − 1000 ms)

**Inventory q(t):** the YES-space position from the maker's status-ok legs with hi ≤ t (`position(..., clock='hi')`).
- `inv_timing_sensitive`: a leg's [x_lo_hard, hi] contains t, and including it changes the |q| bin or sign.
- `inv_ledger_conditional`: an excluded leg of that maker precedes t in the market.
- Bins of |q|: 0 | (0, 50] | (50, 200] | (200, 500] | > 500 shares; sign long / short.

**BTC aligned to inventory:** `a_h = sign(q) · r_h`, with h ∈ {1 s, 5 s, 30 s} (`btc_ret`).
- Category: with (a ≥ thr) / against (a ≤ −thr) / flat, with thr = `MOVE_THR` = 0.5 / 1.0 / 2.0 bp.
- Missing BTC is a status, counted.
- Dose bins for a_5s (bp): ≤ −5 | (−5, −2] | (−2, −1] | (−1, 1) | [1, 2) | [2, 5) | ≥ 5.

**Time left** (`time_left`), in seconds: > 300 (pre-window) | (240, 300] | (120, 240] | (60, 120] | (30, 60] | (10, 30] | [0, 10].

**Own fills in the last 5 s** (hi ≤ t): 0 | 1–2 | ≥ 3.

**Unwind edge** (cents): positive means unwinding now trades better than fair for the maker.
- Long: best bid − p_fair. Short: p_fair − best ask.
- Sources: `book_tob` at t; p_fair from the BTC fair-value model as used by chain_take.
- Bins: ≤ −2 | (−2, −0.5] | (−0.5, 0.5) | [0.5, 2) | ≥ 2.

## Measures

### M1 hazard

Event-seconds / maker-seconds per cell and per type, for each rank and top-3 pooled.
- Cells: |q| bin × sign × a_5s category × time-left bin.
- Marginal tables for every factor.

### M2 Mantel–Haenszel person-time rate ratios

Separately for reduce and add events; rank 1 and pooled. Instants with q ≠ 0 only.

| Ratio | Exposed / reference | Strata |
|---|---|---|
| RR_against | a_5s against / flat | |q| bin × time-left bin (× rank when pooled) |
| RR_with | a_5s with / flat | as RR_against |
| RR_size | |q| > 200 / |q| ∈ (0, 50] | BTC category × time-left bin |
| RR_end | time left ≤ 30 s / (60, 240] | |q| bin × BTC category |
| RR_burst | own fills 5 s ≥ 3 / 0 | |q| bin × time-left bin |
| RR_edge | edge ≥ 0.5 c / (−0.5, 0.5) | |q| bin × time-left bin |

RR_against and RR_with are also computed at 1 s and 30 s (secondary).

### M3 null (declared now)

- Each maker-market's event-seconds are circularly shifted by a uniform offset (mod its instant count): 500 shifts, seed 20261123. The state paths stay fixed.
- `p_hi = (1 + #{null ≥ obs}) / 501`, and p_lo likewise.
- The null tests timing alignment only: shifted takes no longer cause the inventory path.

### M4 one-off or repeated

**Market consistency:** for each M2 ratio, the within-market crude RR over supported markets (≥ 1 event, ≥ 30 exposed and ≥ 30 reference maker-seconds). Report n with RR > 1 / n supported, and the same per day.

**Declared patterns.** Each reports its event count, maker-days, markets with ≥ 1 and ≥ 5 events, and share of that maker's takes:

| Pattern | Definition |
|---|---|
| P1 | sell reducing a long while a_5s with — "sells when BTC rises" |
| P2 | sell reducing a long while a_5s against |
| P3 | buy reducing a short while a_5s with |
| P4 | buy reducing a short while a_5s against |
| P5 | reduce while a_5s flat |
| P6 | add while BTC 5 s moves with the take side |
| P7 | any take with time left ≤ 30 s |
| P8 | reduce with time left ≤ 30 s |

### M5 end of market

- Per maker-market: |q| at END, END − 30 s and END − 60 s.
- Share flattened: |q(END)| ≤ 0.1 · |q(END − 60 s)| among pairs with |q(END − 60 s)| > 0.
- Reduce hazard by time-left bin (from M1).

### M6 inventory at take

- |q| before each event, and the fraction unwound dq / |q|, by type and rank;
- beside it, the |q| distribution over all q ≠ 0 maker-seconds of the same rank.

### M7 price

- Unwind edge at reducing events vs at all q ≠ 0 maker-seconds;
- each take's price vs p_fair at t, by type.

## Reading (computed predicates, per rank 1 and pooled, every one reported)

Mechanisms and their ratios (for the reduce type unless stated):

| Mechanism | Ratio |
|---|---|
| inventory_size_driven | RR_size |
| stop_loss | RR_against |
| take_profit | RR_with |
| end_of_market_flattening | RR_end, AND share flattened ≥ 0.5 |
| burst_driven | RR_burst |
| edge_driven | RR_edge |

Also stated for add: RR_with = directional.

**Verdicts:**

| Verdict | Condition |
|---|---|
| `supported` | RR > 1, p_hi ≤ 0.05, and RR > 1 in ≥ 2/3 of ≥ 5 supported markets |
| `inconsistent` | p_hi ≤ 0.05 but the market share is < 2/3 |
| `insufficient_support` | < 5 supported markets or < 30 events |
| `not_supported` | otherwise |
| `timing_sensitive` | a verdict that flips at D = 1000 or with pending legs included |

**Multiplicity:** 7 mechanisms × 2 scopes, all listed, none selected.

## Outputs, tool, falsifiers

**Outputs:** `v2/chains/take_logic/` (write-once): `hazard_cells.csv`, `ratios.csv`, `consistency.csv`, `patterns.csv`, `end_of_market.csv`, `inventory_at_take.csv`, `edge.csv`, `claims.json`, and `receipt.json` (sha256 of source, inputs and outputs; CPU; RSS).

**Tool:** `tools/take_logic.py` + `tools/take_logic_selftest.py`. Synthetic fixtures:
1. a planted take-profit maker must read take_profit supported and stop_loss not;
2. a planted stop-loss maker must read stop_loss supported;
3. a random-time maker must read no mechanism supported on the fixture seed;
4. causality: moving BTC after t, or moving a later fill's arrival time, leaves every state at t unchanged;
5. the shift preserves event counts;
6. env mutants `MT_TL_MUT` ∈ {flip_align, leak_1s, active_span}, each caught.

**Cost:**
- builder ≤ 20 CPU s (≤ 2 selftest runs plus one pilot, Oct 24 all ranks, into `v2/chains/take_logic_pilot_20261024_r1`);
- full run projected ≤ 45 s;
- refuse to launch if the pilot projects > 60 s.
