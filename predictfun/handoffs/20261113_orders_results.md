# ORDERS: how the top makers work, order by order (results, 2026-11-13)

Session ce68ebe0. USER: "Start the information collection, we fill the gaps and figure out how they work".

Scripts and runs, claude slot:

| Step | Script / driver | Commit | Run |
|---|---|---|---|
| Build v1 | `research/pf_orders.py extract` | 8fd16d97 | 10-23 stopped by its control: 86% mapped (no map written) |
| Build v2 + analysis | `pf_orders.py extract / day / report`, `run_orders.sh` | 1ffafa5e, 9f8ec338, committed before the v2 run | 04:59-06:04Z |

## Data

- **Source:** raw REST match records. Each maker leg carries the maker's address and order hash; each match carries the taker's
  address and hash and the transaction. Step 1 had dropped all of these.
- **Join (as pf_txmap):** to step 1's timed legs on the nine content keys plus rank within equal content, refined by the maker.
  - Hours read: step 1's padded hours minus the sealed days and the forward days. 10-27 skipped 10-28 00-01Z, 10-31 skipped 10-30 23Z,
    11-03 skipped 11-04 00-01Z.
- **Coverage of first-minute legs:**
  - maker 100% on every day;
  - order hash 95.6-98.7%;
  - exact taker order 84.4-93.3%.
- **Controls passed:**
  - our taker orders agree 100% with legs_tk's;
  - one maker, side and price per certain hash;
  - signer id and address one to one;
  - synthetic checks with two planted errors caught.
- **Status of the analysis:**
  - 1,603,252 legs in 0-61 s, none without a map row; 707 self-trade legs excluded.
  - 911,353 orders first filled in the first minute.
  - 187,633 new best levels, 116,966 of them filled.
  - 274,270 takes; 136,352 maker-windows.
- **Population limits:**
  - Event-level measures use only complete trades (every leg with a certain taker order).
  - Orders that never fill are invisible. Size and life are lower bounds from fills.

## Answer

| Measure (first minute, 9 days) | Top 5 | Ranks 6-20 | Rest |
|---|---|---|---|
| Orders | 173,778 | 283,317 | 454,258 |
| Orders filled by >= 3 taker orders | **25.8%** | 12.0% | 8.8% |
| Shares filled per order, median / p90 | **30 / 100** | 20 / 60 | 10 / 28 |
| Life of orders filled >= 2 times (first to last fill), median / p90 | **357 ms / 2.35 s** | 153 ms / 1.45 s | 109 ms / 1.50 s |
| Orders first in the queue at >= 1 / >= 2 trades | **37.1% / 15.7%** | 22.4% / 6.3% | 16.6% / 4.0% |
| Share of new-level creators / of at-best fills | 20.7% / 24.4% | 35.8% / 30.2% | 43.5% / 45.4% |
| Creator: birth to its first fill, median | 450 ms | 294 ms | 397 ms |
| Prices filled per maker and window-side, median / p90 | **6 / 11** | 3 / 6 | 1 / 5 |
| Fills with another price of theirs proven live | 19.7% | 11.2% | 9.0% |
| Maker-windows two-sided | **89.1%** | 84.1% | 43.7% |
| \|net\| / gross at the end of the minute | **26.8%** | 35.0% | 65.2% |
| Take shares / maker shares | 12.6% | 7.0% | 44.0% |
| Take shares reducing / from flat (the rest adds to the position) | 39.6% / 5.1% | 40.4% / 4.7% | 11.7% / 40.3% |
| Re-entry: next order at the same price after the last fill, median (p25 / p75) | 5.9 s (2.3 / 14.0) | 5.1 s | 4.8 s |

Readings (declared, evaluated in code; top 5 against ranks 6+):
- **R400 PERSIST-MORE:** orders filled >= 3 times, 25.8% against 10.0% (x2.6), higher on 9 of 9 days.
- **R401 FRONT-KEEP-MORE:** orders first in the queue at >= 2 trades, 15.7% against 4.9% (x3.2), 9 of 9 days.
- **R402 PROPORTIONAL**, leaning follower: 20.7% of new-level creators against 24.4% of at-best fills (ratio 0.85, lower on 7 of 9
  days, inside the 0.83 line).
- **R404 DIRECTIONAL**, at the line: 39.6% of their take shares reduce their position (line 40%). 55% add to it, 5% start from flat.

**EXIT-v2 (post hoc, EXIT on exact, complete taker orders):** EXIT holds.
- 10,020 observations.
- GONE share: top 5 50.7%, ranks 6-20 62.1%, rest 46.9%. Gap-matched top 5 minus field -1.6 points.
- From the bottom to the top quintile of our P: top 5 29% -> 79%, field 44% -> 64%. The top 5's effect is larger on 9 of 9 days.
- The one-leg 'taker orders' that K.load_day makes from unidentified legs did not drive EXIT's result.

## How they work, in our data

1. **They rest larger orders and keep them through fills.**
   - Median 30 sh filled per order (p90 100) against our 5-sh order.
   - A quarter of their orders are filled by 3 or more separate trades.
   - One in six is first in the queue at two or more trades.
   - A 5-sh order is used up by its first fill and goes back to the end of the queue. A 30-100 sh order keeps its place through many
     small trades.
2. **Their front place does not come from creating levels.** They are first at new levels slightly less often than their share of
   at-best fills, and not faster (450 ms from birth to the first fill). The place comes from keeping orders, with size, and from
   time priority accrued while others leave.
3. **They leave on the signal** (EXIT / EXIT-v2): 29% gone when our P is low, 79% when it is high, decided 100-1000 ms before the trade.
4. **They quote across many prices and both sides.**
   - 6 prices per side per minute (p90 11).
   - Two-sided in 89% of their windows.
   - The ladder is spread over time more than stacked: only 20% of their fills have another of their prices proven live at that
     moment. This is a lower bound, because unfilled orders are invisible.
5. **They end the minute fairly balanced** (|net| / gross 27%).
6. **Their takes mostly add to their position (55%) rather than rebalance it (40%).** Taking is directional, about an eighth of their
   maker volume (cf. B12: about a quarter of their edge is their own taking).
7. **Re-entry at a price takes seconds** (median 5.9 s between orders), about the same for every group. This gap is between fills, not
   between posting times.

## What it means for us

- **Size is not a multiplier here; it is what keeps the front.**
  - Our 5-sh order cannot hold its place through a single fill.
  - The replay's rule M put our resting order at the front mostly when the queue ahead left (FRONT-PULL).
  - A 30+ sh resting order that is not cancelled after partial fills keeps the place it has earned.
- **The pull signal is already the right one.** Their exits follow our P.
- **The ladder:** about 6 prices per side across the minute, two-sided, ending near flat.
- **Next (proposals, the user's call):**
  - FRONT-PULL at the top makers' order size: 30 and 100 sh resting orders, kept through partial fills, same pulls. About 1.5 h,
    replay. It is still rule M (B1a), but now with the order shape that keeps its place.
  - The live probe with the same shape after the churn-fix deploy.

## Limits

- Consumed days, first minute. Public tape: theirs real, nothing of ours.
- Orders are seen only through fills: sizes, lives and simultaneous prices are lower bounds; unfilled orders are invisible.
- Event-level measures use complete trades only (84-93% of legs).
- 'Creator' = the first filled order at a new level; a creator that cancelled before any fill is not seen.
- Taker identity per address: takes through other addresses are not linked.

Outputs: `data/predict_fun/research/orders_20261113/` (orders.json / .txt, build_*.json, day_*.json, map_*.parquet and per-day tables,
logs); copies (json / txt) in `reviews/orders_20261113/`.
