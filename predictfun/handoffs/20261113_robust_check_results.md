# ROBUST-CHECK: are the conclusions about the two sides and about speed correct? (results, 2026-11-13)

Session ce68ebe0. USER: "Can you review then make sure your conclusion is correct then give me some robust results".

Steps, claude slot:

| Step | Commit | Run |
|---|---|---|
| v1: `research/pf_robust_check.py` + `run_robust_check.sh` | 1ae29bdb, before the first run | 15:27-15:36Z, peak 782 MiB |
| v1 post hoc matched comparison (descriptive) | aaa04b61 | 15:39Z (another session's job held the slot; nine retries) |
| Independent code review of `pf_side_moves.py` / `pf_move_down.py` (a fresh agent, given the code, not the conclusions) | none (read-only) | in parallel |
| v2 after the review: `research/pf_robust_check_v2.py` + `run_robust_check_v2.sh` | 9b218820, before its first run | 15:47-15:54Z |

v2 supersedes v1 where both measure the same thing. v1 stays as provenance.

## What was tested

The conclusions given to the USER after SIDE-MOVES (with its post hoc and correction) and MOVE-DOWN:

| | Conclusion |
|---|---|
| C1 | The side the price moves into (asks in a pump): the top makers' orders are mostly already at the next price before the move |
| C2 | The side the price moves away from (bids in a pump): they rarely requote to the better price |
| C3 | There, their old order stays more often than others', and the kept fills pay |
| C4 | Posting at the new best sooner after the move does not pay |
| C5 | The fast order path (5 ms + fastest feed) helps |
| C6 | The same in pumps and dumps |

## Code review: what it found and what was done

| # | Finding | Effect | Done |
|---|---|---|---|
| 1 | SIDE-MOVES' RESTED origin used fills before the onset, which can come after the birth | origin shares (descriptive) | not used in the robust checks; K2 uses margins before the birth |
| 2 | A trade stamped exactly at the birth (the first frame showing it) was outside the old-best proof window | small, B tables | v2 proof window includes it |
| 3 | Events born on an invalid frame (new price = old price or empty) | 0.1% of events (63 / 61 / 36 / 36 of ~76k / 79k / 33k / 35k) | v2 excludes them; no number moved |
| 4 | Windows past the loaded 61 s cut the 10 s kept window short | lowers KEPT for both groups | v2 uses only events with the full window (15,161 dropped) |
| 5 | KEPT per maker (more orders, more chances); unhashed legs; tie order within one trade | could bias top 5 vs field | v2: KEPT per order; hash coverage printed (top 5 99.8%, field 96.1%: no top-5 shortfall); K3b counts every order filled by the first trade |
| 6 | MOVE-DOWN's early-entry arm re-entered at once after a drift / cross kill, skipping the move test | MD numbers a little | v2 arms corrected (decision wrapper) |
| 7 | A fallback already known at the onset never counted as "moved" | dilutes MD towards today's lane | v2 measures the move from the best 100 ms before the onset |
| 8 | WHY-GONE took the old price from a time search, which can pick an earlier frame on equal stamps | small | not re-run; WHY-GONE's split is reported as before with this caveat |
| 9 | Cosmetic: cancel-pending orders counted as held; one day-rule omission in MOVE-DOWN's R521; docstring wording | none on conclusions | noted |

The review found these correct:
- the windows match their docstrings;
- the event recomputation matches PULL-DEPTH;
- the 250 ms wait emulation is exact by construction with early entry off (40 ms and 5 ms paths);
- order-price reconstruction is exact;
- clocks and switches are always restored;
- groups are one per maker per day.

## Results (v2; K8 from v1)

| Check | Result | Verdict |
|---|---|---|
| K1 strict sweep | In trades that cross from the old best into the next price, the top 5 hold **50.7%** of the volume at the next price (pump and dump alike) but only 28-29% at the old best being crossed. Their usual share is 33.9%. Above their usual share on 9 of 9 days | PASS |
| K2 proof with a 250 ms margin | 6.6-6.9% of their shares at the new best come from orders proven there at least 250 ms before it became the best, against 4.1-4.4% for the field (8 of 9 days, both directions) | PASS |
| K3 front | 37.3% of first-filled orders against 27.3% of all filled orders: ratio 1.37 | PASS, 9 of 9 days |
| K3b front, by first trade | 32.5% of orders filled by the first trade against 27.3%: ratio 1.19 | PASS, 9 of 9 days |
| K4 requote when visible | Where the new price trades >= 100 sh, **32.1% / 31.3%** of top makers at the old best are also filled at the new price, against 28.8% / 27.8% for the field | PASS by the declared 33% line, narrowly: "about one in three, a little more than others", not "rare" |
| K5 kept, per order | With >= 300 sh traded at the old price after the move, **25.0% / 24.2%** of their old orders fill again, against 8.3% / 8.7%, on 9 of 9 days. Orders that had filled < 10 sh before the move (not used up): **46.2% / 45.7%** against 11.1% / 11.8% | PASS |
| K6 kept fills' value | **+0.53 / +0.60** c/sh at 1 s (positive 9 of 9 days), +0.32 pooled at 5 s. The field's: -0.05 / +0.28 at 1 s | PASS |
| K7 early entry, corrected, against the same path | MD2 **-$384 / -$421 / -$463** a day (M 0 / 50 / 250, lower on 9 of 9 days each); MDF2 -$349 / -$394 / -$477. Gated by the model: -$0 to -$2 at M 0 / 50, -$32 / -$33 at M 250 with tau 0.25 (9 of 9 lower) | C4 ROBUST |
| K8 fast ladder (v1) | All six rungs with the fast path: WS **+$86 -> +$113** (M 0), **+$54 -> +$85** (M 50), -$936 -> -$914 (M 250); higher on 9 / 9 / 8 of 9 days. All trades +$70 / +$73 / +$46 a day | PASS |

v1 gave the same answers. Its K7 declared verdict "C4 REVISED" came from comparing every early-entry arm with today's slow lane: the fast arms inherited the fast path's gain. The matched comparison (v1 post hoc, then v2 as declared) shows no gain from early entry.

## Robust conclusions

1. **C1 holds.** On the side the price moves into, the top makers are thin at the price about to be crossed and thick at the next one.
   - They hold 28% of the volume at the old best but 51% at the next price.
   - Their orders lead the next level's queue: ratio 1.37 by first fill, 1.19 by first trade.
   - Some are proven there 250 ms or more before the move.
   - Pumps and dumps are the same; every check passed 8-9 of 9 days.
2. **C2 is restated.** It is not "rarely". Where the new price trades enough to see it, about one in three makers at the old best also quote the
   new price: top 5 32%, others 28%. The 6% seen over all events was a visibility artifact.
3. **C3 holds.**
   - Their old order stays about 3 times as often as others' (25% vs 8-9%; orders not used up 46% vs 11-12%), 9 of 9 days.
   - When it stays and is filled later, it earns +0.53 to +0.60 c/sh, positive every day.
4. **C4 holds.** Posting at the new best sooner after the move does not pay.
   - Ungated it loses about $400 a day at every clock.
   - Gated by our model it adds nothing at M 0 / 50 and loses $32 a day at M 250.
5. **C5 holds on the whole ladder.** The fast path adds about $30 a day on exact-timed trades and $70 over all trades (M 0 / 50); 9 of 9
   days.
6. **C6 holds.** Every check gives the same answer for pumps and dumps.

## Limits

- Public fills only: no order creation or cancel times.
- Inferences from queue order assume price-time priority; B1a found a fifth of live fills contradicting the shown-size queue.
- Consumed days (C8). Ours are replay with rule M (C5, C9).
- K4's line was declared before the run; 32% passes it narrowly.
- PULL-DEPTH's P (used by the gated arms) carries the nested-fold leak.
- WHY-GONE (our guard pulls our held order 52% of the time) was not re-run after review point 8.

Outputs:
- `data/predict_fun/research/robust_check_20261113/` (v1) and `.../v2/`: report txt / json, sweeps / orders / firsttrade / makers /
  oldorders / fills, day_*.json, logs.
- Copies (txt / json / logs) in `reviews/robust_check_20261113/` (v2 in `v2/`).
