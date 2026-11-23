# QUEUE-FRONT v1 — contract (frozen before computation)

2026-11-23, session aea5bfb9.

**Mandate:** the USER, verbatim:
- 2026-11-23: "we want to figure out the logic chains of top makers, so we need to understand why they do actions given market conditions and how do they get front in the queue";
- earlier: "when they get filled, is it some early posts resting there to get the front in queue".

The order-history estimator could not answer this from fills alone: 2,944 of 4,149 first-fill queue positions are unidentified and 889 conflict.

**Approach:** identify the front of the queue WITHOUT posting times.
- The venue enforces price-time priority (v2 report: T1 price-priority violations 1 / 18,433; T2 time-priority violations 5 / 16,433).
- So the first fill at a freshly created price level goes to the order at its front.
- That premise is stated, not re-tested here.

**Budget:** ≤ 0.06 core-hours all-in, including a 0.010 review share, under the USER's standing CPU authorization (2026-11-23).

**Data:**
- the 19 BOOK markets, Oct 24–27 (consumed);
- the usable full-depth frames and the maker fills, both as `chain_context` reads them (`frame_times`, `level_at_frames`, `level_series`, legs with hard clocks);
- the saved markouts (`v_1s` / `v_10s` / `v_60s`, gross from `settle_ms`, as chain_post_fill reads them);
- no new data, no replay.

## Units

**E1 fresh level:** a (market, side, tick) cell, in the book's YES-space convention, shown 0 or absent in usable frame f−1 and > 0 in frame f.
- No gap (> 1,000 ms) between the two frames.
- The creation window is (upd(f−1), upd(f)].
- A cell that is unknown (NaN) in f−1 is a status, not an event.

**E2 front window:** from creation until the first of:
- the cell's first matched fill;
- the cell shown 0 again;
- a gap;
- the market window end.

**Front holder:** the maker address of the FIRST fill at that cell inside the front window, ordered by the fill's hard window. Statuses:

| Status | Condition |
|---|---|
| `front_identified` | the shown size never decreased between creation and that fill except by matched fills, no gap, and the fill's [x_lo_hard, hi] starts after the creation window opens |
| `ambiguous_order` | another fill at the cell has an overlapping hard window |
| `unexplained_decrease` | a cancel ahead could change who is first |
| `no_fill` | the cell returned to 0 or the window ended without a fill: the creator is anonymous, counted in every denominator |
| `gap` / `unknown` | otherwise |

A fresh level's front holder is its creator, or an order accepted within the same frame interval. The distinction is not identified; the unit is "held the front of a fresh level".

## How the front was obtained (mechanism at creation, from frame f−1)

| Mechanism | Rule |
|---|---|
| `improve` | the tick is strictly better than the best price on its side in f−1, i.e. inside the spread: a new best price |
| `re_establish` | the cell was shown > 0 within the 5 s before creation and went to 0 through matched fills: re-posting a just-cleared price |
| `deep` | the tick is worse than the best price on its side in f−1: resting behind the touch before the market reaches it |
| `at_empty_touch` | the side's best level was absent in f−1 |
| `other` | none of the above |

**Joined existing queues:** top-3 fills at cells NOT fresh (the cell was > 0 before the order's lifecycle posting window) are counted beside, with the existing queue_class. That is the "joined a queue" mechanism, front only after clearing.

## Measures

**Who wins the front:**
- the share of `front_identified` fresh levels whose holder is rank 1 / 2 / 3 / another address;
- compared with that group's share of ALL maker fills at the same cells;
- per market and per day, with the denominators: all fresh levels, `no_fill`, ambiguous.

**How:** the mechanism mix of each group's front wins.

**Speed:**
- for `re_establish`: the time from the cell's emptying (last frame > 0 before 0) to the creation window;
- for `improve`: the time from the previous best-price change on that side;
- the distributions per group (rank 1–3 vs other identified holders).

**Whether front pays:**
- markouts (`v_1s`, `v_10s`, `v_60s`, gross, c/sh) of front-holder first fills vs later fills at the same fresh cells (behind the front);
- per group and per mechanism;
- with shares and fill counts.

**Null (declared):** within each market, permute the maker labels across all fills at fresh cells (front and behind): 500 permutations, seed 20261123.
- `p_hi` for "group's front share exceeds its fill share" = (1 + #{null ≥ obs}) / 501;
- market consistency: the share of supported markets (≥ 5 front_identified events) in which the group's front share exceeds its fill share.

**Readings, computed, per rank 1 and top-3 pooled:**

| Reading | Condition |
|---|---|
| `wins_front_more_than_fills` | front share > fill share, p_hi ≤ 0.05, in ≥ 2/3 of ≥ 5 supported markets |
| `main_mechanism` | the mechanism with the largest share of their front wins, with each share |
| `faster_than_others` | median `re_establish` speed below the other holders' median, in ≥ 2/3 of supported markets |
| `front_pays` | front-fill `v_10s` markout > behind-fill `v_10s` at the same cells, pooled and in ≥ 2/3 of supported markets |

Everything else is `unidentified` or `insufficient_support`. Point estimates over 4 days; no intervals.

## Outputs, tool, falsifiers

**Outputs:** `v2/chains/queue_front/` (write-once):
- `levels.csv` (fresh-level events with status and mechanism);
- `fronts.csv`, `shares.csv`, `mechanism.csv`, `speed.csv`, `economics.csv`, `consistency.csv`;
- `claims.json`;
- `receipt.json` (sha256 of source, inputs and outputs; CPU; RSS).

**Tool:** `tools/queue_front.py` + selftest. Synthetic book / fill fixtures:
1. a planted level creator filled first → `front_identified` with the right holder;
2. a cancel ahead (an unexplained decrease) → `unexplained_decrease`, never identified;
3. overlapping fill windows → `ambiguous_order`;
4. a level created and cancelled without a fill → `no_fill` in every denominator;
5. mechanism classification:
   - improve / re_establish / deep / at_empty_touch on hand-built frames;
6. causality: frames after the front fill do not change its status;
7. env mutants `MT_QF_MUT` ∈ {first_fill_by_settle (orders by hi only), ignore_decrease, drop_no_fill}, each caught.

**Cost:**
- builder ≤ 25 CPU s (≤ 2 selftests plus one Oct 24 pilot into `v2/chains/queue_front_pilot_20261024_r1`);
- full run projected ≤ 60 s;
- refuse to launch above 120 s.

## Next (not in this contract)

The posting decision logic — why they create, improve or re-establish levels given fair value, spread, inventory and BTC — on a decision-time population, after this result.

## Amendment 1 (before the full run; reason: the Oct 24 pilot's status counts, not its readings)

**Pilot (7 markets, 7,445 fresh levels):**

| Status | Count |
|---|---|
| no_fill | 3,299 |
| ambiguous_order | 2,921 |
| unknown | 973 |
| unexplained_decrease | 233 |
| gap | 11 |
| front_identified | **8** |

Every fill's match time is known only to a 2-s hard window. 72% of first fills arrive within 2 s of creation, so any second fill at the cell overlaps the first. The rule as written treated even overlaps between fills of the SAME maker as ambiguity, and 610 levels are that case. The question is WHO holds the front, not the order among one maker's own fills, so the rule changes. No other definition, bin, null or reading threshold changes.

1. **Earliest-overlap set:** the first fill at the cell (by hard-window start) plus every other fill at the cell whose hard window overlaps it.
   - If every fill in that set belongs to one maker address, the status is `front_identified` with that holder.
   - `ambiguous_order` remains only when the set spans ≥ 2 addresses. The address set is recorded.
2. **Bounds:** each group's front share is reported as bounds.
   - lower = `front_identified` levels held by the group;
   - upper = lower + `ambiguous_order` levels whose address set contains the group.
   - Both use the same denominator: identified + ambiguous.
3. **Readings:** `wins_front_more_than_fills` and the market-consistency count use the LOWER bound vs the fill share. The upper bound is reported beside. `main_mechanism`, `faster_than_others` and `front_pays` use `front_identified` levels only.
4. **Selftest additions:**
   - a same-maker overlap gives `front_identified`;
   - a two-maker overlap gives `ambiguous_order` with both addresses, and enters the upper bound only;
   - an env mutant `MT_QF_MUT=same_maker_ambiguous` (the old rule) must be caught.
