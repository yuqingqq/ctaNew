# SIDE-MOVES: what the top makers do with their quotes on each side after a danger signal (results, 2026-11-13)

Session ce68ebe0. USER: "Can you still check both sides for top makers, on protection side, it looks we have similar latency performance,
but for the other quoting side, can you check what they do, do they also requote or what".

Scripts and runs, claude slot:

| Step | Commit | Run |
|---|---|---|
| Script `research/pf_side_moves.py` + `run_side_moves.sh` | e2626a18, before the first run | 14:32-14:34Z, peak 730 MiB |
| Report fix | 36637b1f: the first report job failed (the maker table read DataFrame.join, a method, not the column); day outputs unchanged | 14:34Z |

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

## Other side (the side that requotes): the top makers mostly KEEP their old quote

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

## Protection side: the top makers MOVE down to the new best, fast, and are at its front

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
   - **Protection side:** they follow the price down. Top-5 makers quoting the old best show up at the new best 2.75 times as often as
     the field. Their first fill there comes a median 302 ms after it forms, and their order is usually first in its queue. Another 10% of
     their orders there were already resting at that price.
   - **Other side:** they mostly do not requote. They keep their old quote in its queue (2.6 times the field) and only sometimes add one
     at the new price. When they do, it is usually first.
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
