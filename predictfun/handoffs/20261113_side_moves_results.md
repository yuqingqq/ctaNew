# SIDE-MOVES: what the top makers do with their quotes on each side after a danger signal (results, 2026-11-13)

Session ce68ebe0. USER: "Can you still check both sides for top makers, on protection side, it looks we have similar latency performance,
but for the other quoting side, can you check what they do, do they also requote or what".

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| Script `research/pf_side_moves.py` + `run_side_moves.sh` | e2626a18, before the first run | 14:32-14:34Z, peak 730 MiB |
| Report fix | 36637b1f: the first report job failed (the maker table read DataFrame.join, a method, not the column); day outputs unchanged | 14:34Z |
| Post hoc readout (descriptive) + correction | 6e49d5ff: pump / dump split, proof of resting at the new price, KEPT by traded volume, value of kept fills; the readings reproduce exactly | 15:10-15:12Z |

**Corrected 2026-11-13 15:13Z (post hoc, section at the end):** two statements of the first version went beyond the data.
- "On the protection side they MOVE down to the new best": the post hoc shows their orders are mostly already at the next price before the
  move. A maker filled at both prices may have held both orders (a ladder); placement times are not public.
- "On the other side they mostly KEEP their old quote": they keep it more often than others (2.6x), but at least three quarters of their
  old orders are gone (cancelled or used up) even when 100+ sh trade at that price again.

## Design

- **Events:** PULL-DEPTH's, recomputed and checked row for row. A guard onset on one side, then within 1 s:
  - **protection side:** its best falls back to a new price (the landing price), 75,914 events;
  - **other side:** its best improves (the requote price), 78,800 events.
  - Calm births with no signal nearby are the baseline.
- **Data:** real maker legs with exact (WS) timing and the maker's address and order hash (ORDERS). 9 consumed days, first 60 s.
- **Groups:** top 5 by daily maker volume vs the field.
- **Measures** (orders are seen only through fills, so every share is a lower bound, the same for every group):
  - **A.** Orders filled at the new price: fill share, first-fill share, and shares filled ahead of each order (its queue place).
    - Each order's origin: RESTED (it had been at that price before), FROM-OLD (its maker was at the old best), OTHER (the maker was at
      another price on that side), FRESH.
  - **B.** Makers proven at the old best just before (a fill there within 2 s):
    - JOIN: they get a fill at the new price.
    - KEPT: their old order fills again at the old price within 10 s, among events where the old price trades again (OBS).
    - Split: MOVE (join, old order not seen again), ADD (join and kept), KEEP (kept only), NONE.
  - **C.** Protection side: who is filled at the old best in its last second (run over).
  - **D.** Ours: the guard ladder's rung 0, simulated with exact orders.
- **Controls passed:**
  - Events equal PULL-DEPTH's on every day.
  - The simulator copy equals the simulator on every window-side.
  - Synthetic cases with three planted errors caught.

## Other side (the side the price moves away from): the top makers keep their old quote more often than others and rarely requote

(Heading corrected 15:13Z; it said "mostly KEEP". See the post hoc section.)

B. Makers at the old best when the other side's price improves after a signal:

| Group | Maker-events | JOIN the new price | KEPT (old order fills again) | Among OBS: MOVE / ADD / KEEP / NONE |
|---|---|---|---|---|
| Top 5 | 42,159 | 6.2% (fill 498 ms after) | **18.5%** | 5% / 2% / **16%** / 76% |
| Field | 103,169 | 4.5% (588 ms) | 7.2% | 5% / 1% / 7% / 88% |
| Top 5, calm improvement | 8,886 | 7.3% | 19.2% | 5% / 3% / 16% / 75% |

Readings:
- **R511 TOP-KEEP-MORE:** their old order is still there and fills again 2.6 times as often as the field's (9 of 9 days).
- **R510 TOP-REQUOTE-MORE:** they put a quote at the new price a little more often (ratio 1.37, 8 of 9). Moving (cancel and chase) is the
  same as the field (5%).
- Their behaviour after a signal is the same as at any calm improvement. They do not react to the danger on this side.

A. At the new improved price:

| Group | Share of fill shares | First-fill share | Shares filled ahead, median (p75) | c/sh | Origin RESTED / FROM-OLD / OTHER / FRESH |
|---|---|---|---|---|---|
| Top 5 | 28.2% | 20.5% | **0** (22) | +0.53 | 2% / 18% / 31% / 48% |
| Field | 71.8% | 79.5% | 10 (64) | +0.57 | 1% / 7% / 21% / 71% |

- **R512 TOP-AHEAD (9 of 9):**
  - The top makers are at this price less often than their usual 34% share.
  - When they are, their order is usually the first one filled there (median 0 shares ahead).
  - Their orders there come from makers already quoting that side more often than the field's (FROM-OLD 18% vs 7%).

D. Ours (simulated) at these events:
- At the birth our rung 0 is HELD at the old price 40% of the time and has no order 41%.
- It has an order at the new price before the level ends in 22% of events, a median 315 ms after the birth.
- Our held old order is still live 1 s later only 34% of the time.

## Protection side: the top makers are at the front of the new best

(Heading corrected 15:13Z; it said "MOVE down ... fast". The post hoc shows their orders are mostly already at that price before the move.)

B. Makers at the old best when it falls away:

| Group | Maker-events | JOIN the landing price | Fill there, ms after the birth | KEPT | Among OBS: MOVE / ADD / KEEP / NONE |
|---|---|---|---|---|---|
| Top 5 | 32,230 | **15.4%** | **302** | 2.5% | **15%** / 1% / 2% / 83% |
| Field | 108,977 | 5.6% | 441 | 1.2% | 5% / 0% / 1% / 94% |

A. At the landing price:

| Group | Share of fill shares | First-fill share | Shares ahead, median (p75) | First fill after the birth | c/sh | Origin RESTED / FROM-OLD / OTHER / FRESH |
|---|---|---|---|---|---|---|
| Top 5 | **42.5%** | 37.3% | **0** (11) | 350 ms | **+0.30** | **10%** / **18%** / 26% / 46% |
| Field | 57.5% | 62.7% | 9 (71) | 450 ms | -0.11 | 3% / 9% / 19% / 68% |

C. Run over at the old best's last second:
- **R513 PROPORTIONAL:** the top 5 take 32.0% of the run-over shares against their 33.9% at-best share (ratio 0.94, lower on 7 of 9).
- Both groups lose about -1 c/sh there.

D. Ours (simulated): at the birth our rung 0 has no order 89% of the time (the guard pulled it). It has an order at the new price in 35%
of events, live a median 347 ms after the birth.

## What it means

1. **The two sides are handled oppositely.**
   - **Protection side** (corrected 15:13Z, see post hoc): their orders are mostly already at the next price before the move, and first in
     its queue. Top-5 makers quoting the old best are also filled at the new best 2.75 times as often as the field; that can be a ladder
     order rather than a requote.
   - **Other side** (corrected 15:13Z): they rarely requote. Their old quote stays more often than others' (2.6 times), but most of their
     old orders are gone. When one does stay it is usually filled later at a profit.
2. **Latency on the protection side is not similar to ours.**
   - Their fills at the new best (an upper bound on when their order arrived) come at 302 ms, at the front.
   - Our order goes live at a median 347 ms, in only 35% of events. It is behind the queue because we wait for the guard to clear plus
     250 ms (QUEUE-PLACE).
3. **On the other side we are closer to them.** We also keep our old order (held 40% at the birth). But ours is gone within 1 s two-thirds
   of the time (cause not measured here), while theirs stays.

## Next (proposals, the user's call)

- **MOVE-DOWN (protection side, replay):** at the guard onset, cancel rung 0 and place at the new best as soon as it forms, without the
  250 ms clear wait (FAST path optional). Then keep it.
  - Measure: shares ahead at the new best, STOP share, c/sh, $/day at M 0 / 50 / 250, WS / E2.
  - KEEP-PLACE's immediate re-entry at the same price did not help. This re-enters at the new price.
- **Why our held other-side order goes:** split by cause (guard / drift / fill / cross) before changing that side.

## Limits

- Orders are seen only through fills: every rate is a lower bound.
  - KEPT needs the old price to trade again with the order not used up, which favours larger orders (the top 5's are 30-100 sh).
  - JOIN times are fill times, not posting times.
- WS-timed legs only (C12). The guard is our signal; the top makers' own signals are unknown.
- Consumed days (C8). Ours are simulated with replay rule M (C5, C9).

Outputs:
- `data/predict_fun/research/side_moves_20261113/`: side_moves.txt / .json, orders_*, makers_*, runover_*, ours_*, day_*.json, logs.
- Copies (txt / json / logs) in `reviews/side_moves_20261113/`.

## Post hoc (added after the report, descriptive): pump vs dump, proof of being there first, how often the old order stays

Rerun with extended rows (6e49d5ff); the declared readings reproduce exactly. UP = a pump (protection side = asks, other side = bids);
DOWN = a dump. Fills are lower bounds: the public match records carry only order hash, signer, price and amount, with no creation or
cancel time (checked on a consumed day), so placement and cancel times are not observable.

**Protection side (the side the price moves into): their orders are already at the next price.**

| After a signal | Group | Share of fill shares at the new best | Of the group's shares there, from orders PROVEN there before it became the best | First-fill share | Shares ahead, median | c/sh | Share of the volume filled at the new price in the trade(s) that made it the best |
|---|---|---|---|---|---|---|---|
| UP (new best ask) | **top 5** | 43.5% | **16.7%** | 37.9% | **0** | **+0.30** | **48.9%** |
| UP | field | 56.5% | 8.6% | 62.1% | 10 | -0.05 | 51.1% |
| DOWN (new best bid) | **top 5** | 41.4% | **16.1%** | 36.7% | **0** | **+0.29** | **47.2%** |
| DOWN | field | 58.6% | 9.4% | 63.3% | 6 | -0.18 | 52.8% |

- **In the trade that pushes the price to the new level, half of the volume filled at that level is theirs.** That is 49% in a pump and
  47% in a dump, against their usual 34% of at-best fills.
  - Those orders were resting at the next price before the move: a taker reaches the front of the next level's queue first.
- At least 16-17% of their later fills there come from orders proven there before (field 9%, higher on 9 of 9 days). This is a lower
  bound.
- The same holds at calm fallbacks (17.5-18.4% vs 11.6-12.0%, 9 of 9 days). This is how they quote, not a reaction to a signal.

**Other side (the side the price moves away from): rarely requote; the old order stays more often than others', most are gone.**

Makers proven at the old best just before the move:

| After a signal | Group | Quote filled at the new price (JOIN) | Old order filled again within 10 s, when that price traded again | Same, by shares traded there after the move: < 30 / 30-100 / >= 100 sh | Kept fills: c/sh at 1 s, first one after the move |
|---|---|---|---|---|---|
| UP (old best bid) | **top 5** | 6.1% | **18.4%** | 14.7% / 18.2% / **23.7%** | **+0.52**, 1.2 s |
| UP | field | 4.4% | 7.4% | 6.6% / 7.2% / 8.6% | -0.04, 1.3 s |
| DOWN (old best ask) | **top 5** | 6.3% | **18.5%** | 13.6% / 19.4% / **23.5%** | **+0.64**, 1.4 s |
| DOWN | field | 4.6% | 6.9% | 5.2% / 7.4% / 8.6% | +0.30, 1.2 s |

- **Requoting to the better price is rare:** about 6% (field 4-5%).
- **The old order is still there more often than others':** 9 of 9 days.
  - When 100+ sh trade at the old price after the move, it would very likely be reached if it were still there (it was near the front:
    it had just been filled there).
  - Even then only 24% of the top makers' old orders are filled again (field 9%).
  - So at least about a quarter stay. The rest are gone, cancelled or used up by the fills before the move; public data cannot separate
    those two.
- **When it stays, waiting pays:** their kept fills come about 1.2-1.4 s after the move and earn +0.52 / +0.64 c/sh (field -0.04 /
  +0.30).
- The same pattern holds at calm improvements (kept 18.5-20.0% vs 7.1-7.2%).

**Ours (MOVE-DOWN, simulated):** our own guard on that side pulls our held old order within 1 s 52% of the time.

