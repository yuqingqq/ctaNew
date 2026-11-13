# P-2026-005 findings register

USER 2026-11-12: "Can we still keep all our findings tracked, otherwise these misfindings happen again and again".

The program's memory of what is known, what is broken and what every new result must be checked against. HANDOFF.md is the
chronological log; this file is the standing state.

**How to use it.**
1. Read it at session start, after HANDOFF.md.
2. Before reporting any number, go through §1. A report that skips an item says which one and why.
3. When a study finishes, add or update its line in §3 in the same commit as its report.
4. When a later check changes an earlier finding, change its status here and point to the check. Never delete a line. Also put a
   note at the top of the old report.

Statuses:
- **HOLDS:** no later check against it.
- **QUALIFIED:** holds with the stated limit.
- **VOID:** a defect invalidates it.
- **SUPERSEDED:** replaced by a later result.
- **OPEN:** not yet tested.

## 1. Checklist for every replay result

| # | Rule | Why (source) |
|---|---|---|
| C1 | Split profit by the order's age at the fill (< 100 / 100-500 / >= 500 ms). Report the headline without fills on orders < 100 ms old and B1a-weighted, beside the headline | Replay fills on orders < 100 ms old were 8-12% real on live's own orders (B1a). 73-77% of v3's and QR2's replay profit is there (AGE) |
| C2 | Report fill clocks M 0 / M 50 / M 250, and say whether the claim holds at M 250 | No constant trade-time shift is valid; signs flip between clocks (B1a, review 2026-11-10) |
| C3 | Report the cancel-pending share of fills, and its $ | The replay credits 2.5x live's cancel-pending shares (B1a); 42% of the frozen v3's replay shares are cancel-pending (FV3) |
| C4 | Treat replay volumes and $ totals as inflated | The replay credits 1.5-2x live's shares (B1a R24) |
| C5 | Rule M primary, L shown; U is a front bound only | M is unvalidated; a fifth of live's fills contradict the shown-size queue (B1a) |
| C6 | Every decision may use only information observable at its time. Write out observation, decision and placement clocks: feed >= 13.2 ms (median ~53), decision 2 ms, placement 40 ms. To model faster placement, change that leg explicitly. Never schedule a price-dependent decision before its information exists | QR's fast entries decided before the level existed (review 2026-11-12 10:42Z, 9053263d) |
| C7 | A whole-market markout is not our order's value: replay one order with queue position before recommending | Pro rata flow reversed the sign once (2026-11-10) |
| C8 | Name the days used. Parameters chosen on seen days are post hoc. A pass is a candidate; validation needs a freeze and >= 5 untouched days | Consumed days: 10-23..27, 10-31, 11-01..03 (9 replay days), 11-07, 11-08. Never read 11-04..06 or 11-09..13 (forward blocks) |
| C9 | When ours is simulated and theirs is real, say so in the table | GAP |
| C10 | No replay number is validated at the order level; only a live run with the A4b capture can validate one | B1a; USER D2 |
| C11 | Name the strategy. Lane J-v3 (one 5-sh order per side, v3's PM90 gate on the 50 ms grid) is NOT the frozen pm_ladder v3 (MICRO side choice, G1-G6 guards, LD2 rung, 25-sh size, CAP10, XMIN1, XV, RO60 / S240, event-driven). Test any candidate on the frozen engine (`pf_step4_ladder_v3.py`) too | Pipeline review 2026-11-12, S1 |
| C12 | Split C1's age table by trade-anchor source (WS-timed / E2-estimated); E2 anchors are uncertain by about +-250 ms | Pipeline review D4 |

## 2. Data and code pitfalls (each has bitten once)

| # | Pitfall | Fix |
|---|---|---|
| P1 | Step-1 signer ids are per day (order of first appearance), not global | Rank makers per day; a maker cannot be followed across days; our own account is unknown on the replay days |
| P2 | Live cancels sent before the ack can make B13c's re-place match pair two orders both ways | Walk chains only to strictly earlier accepts (QR part C, 4756b27d) |
| P3 | The venue's book arrives in ~50 ms snapshots (p10 / p50 / p90 41 / 50 / 58 ms after a birth frame) | "At the birth" has 50 ms resolution; the raise rule adds the next snapshot's joiners |
| P4 | Research hours padded by +-1 h reach sealed days | Enforce day rules in the loader; exclude boundary windows with a status |
| P5 | Proxy tuple keys merge distinct taker orders | Group by the event's own id (taker hash / legs_tk) |
| P6 | Cancel-pending orders still trade | Include them in price priority, self-cross checks and invariants |
| P7 | Results from before an engine fix | Void; report only fixed-engine numbers |
| P8 | STATUS.yml `updated:` must stay a block scalar | A quoted scalar broke the file (repaired 2026-11-12) |
| P9 | Forward data-day paths printed in a session count as reads | Aggregate at family level; never name the paths |
| P10 | Tie windows (start = end price, paid 50/50) are recorded as "Up" in the registry (`pf_step1.py:57-59`): 14 on the 9 days | Settlement-valued numbers need ties at 50 (fix proposed) |
| P11 | Step-1 day folders include the adjacent days' boundary hour files (sealed 10-28 / 10-30, forward 11-04); the loader keeps frames from ws - 5 s | Filter frames by stamp in the loader with a status; never list or read those files |
| P12 | Trade anchors: E2 estimates (about a third of first-minute shares) differ from WS times by IQR about -250 / +150 ms; a taker order's minimum time can come from a mismatched leg | See C12 |
| P13 | The sim2 family serves our own orders in time order (the frozen engine: best price first) and lets two of them take one trade leg | Matters when two of our orders overlap (CS fast paths: 1-7% of shares) |
| P14 | 10-31's taker ids: the step-1 and txmap signer indexes disagree (13% of first-minute shares unmapped) | Taker-order analyses on 10-31 are weaker |
| P15 | Binance is known at event time + 3.5 ms (a Tokyo-like constant); Binance-side delay spikes are not modelled | Sensitivity once a Tokyo busy-hour capture exists |
| P16 | Raw features (`gap`, `p_bn`) use the window's start price from tau = ws; only the guard value is masked (v = -1) until the price is known | Any model that reads raw features must apply the start-known mask itself |
| P17 | A rank AUC over rows that include NaN scores returns NaN without warning (FA's R341 on 5 of 9 days) | Drop or count NaN rows before any AUC / rank statistic; a NaN reading is an instrument failure, not a result |

## 3. Findings

### A. Is the replay right?

| Finding | Status | Source |
|---|---|---|
| The replay is not validated order by order. It credits 1.47-1.96x live's shares, at precision 0.03-0.25. Orders < 100 ms old are 8-12% real (M 0 / M 50); no constant trade shift is valid. Live's real young fills were not adverse (+0.05 c/sh) | HOLDS (one day, the old churning engine) | B1a, 20261110_b1a_live_orders.md |
| 73-77% of v3's and QR2's replay profit is on orders < 100 ms old. Without them v3 is $3.7 a day at M 50 ($2.5 B1a-weighted); orders >= 500 ms earn about 0; M 250 about 0 | HOLDS | AGE, 20261112_age_recheck_results.md |
| Replay results before round i (2026-11-05) | VOID (USER) | STATUS prefix_results |

### B. v3 and the guard

| Finding | Status | Source |
|---|---|---|
| The frozen pm_ladder v3 (its own engine, 9 replay days, whole window): $60 / $45 a day at 1 s (J / X timing), settlement $99 / $65 with ties at 50; 79-83% of the 1 s profit is on orders < 100 ms old (about $10 a day without them, $9-10 B1a-weighted); orders >= 500 ms lose slightly; 42% of its filled shares are on cancel-pending orders; at the 250 ms fill clock it loses $40 a day (1 of 9 days positive) | QUALIFIED: young and cancel-pending fills unvalidated (B1a); parameters chosen on these days | FV3, 20261112_fv3_results.md |
| A faster order path on the frozen v3 (20 / 10 ms against 40): +$2.0-2.6 / +$4.6-4.7 a day without young fills, 8-9 of 9 days at J and X; not at the 250 ms clock | HOLDS (replay; the speeds are settings, not measurements) | FV3 R332 / R333 |
| The frozen v3 has no inventory price skew (SK / RX / RL / A4 are not in its arm); its inventory logic only withholds quotes: CAP10 (10 sh net), G6 (50 sh, binds only without CAP10), RO60 (reduce-only from 60 s). Only CAP10 touches the first minute: it withholds about a sixth of v3's shares there; without it +$2.7 (J, 8 of 9 days) / +$1.2 (X, 4 of 9) a day w/o young fills, settlement -$20 / -$10 (4 of 9 days above: noise). The J-v3 timing studies carry no inventory logic. Timing base for the frozen engine: first 60 s, CAP10 and G6 off, venue rules kept | QUALIFIED: post hoc, in-sample (CAP10 / RO60 chosen on these days); no arm without G6 exists | INV-RULES, 20261112_inv_rules_results.md |
| v3 without its quoting limits (CAP10, G6, RO60 off; first minute): +$4.5 (J, 8 of 9 days) / +$3.4 (X, 7 of 9) a day w/o young fills, settlement lower (more inventory carried). Improving one tick inside a >= 2-tick spread (IM95) moves us toward the front (filled shares with <= 5 sh ahead 18.7% vs 5.0%, median queue ahead 208 vs 308 sh) but adds only +$1.6 / +$0.4 a day w/o young (6 of 9 days), $0 at the 250 ms clock: being first alone barely pays | QUALIFIED: replay rule M (B1a), consumed days | TB, 20261112_tb_results.md |
| Public tape (no fill model): no missing information in any feed we have. On top of the guard's inputs, Predict's trade tape (incl. a causal taker reputation), book cancels / adds / level age and Coinbase / OKX / Bybit add <= +0.002 AUC for 'the best moves against the side in (t + 24, t + 274] ms'. The guard's inputs fitted on that event reach AUC 0.87 (activity 0.60: they track harm when aimed at it); 29% of those moves happen before a cancel can land (speed lever); the avoidable rest is predicted at AUC 0.84 (9 of 9 days 0.82-0.86), mainly by the guard's own v (53% of gain); among trades that arrive, clearing vs standing at 0.82. Pulling in 11% / 23% of moments catches 50% / 70% of the avoidable level losses | HOLDS (descriptive, consumed days; AVOID / ops post hoc; AUC is not money: the pull's cost is the queue place) | MOVE, 20261112_move_results.md |
| Which live feed carries the avoidable-harm prediction (post hoc, MOVE-AVOID feeds; level still there at t + 24, goes in (t + 24, t + 274] ms): Binance alone AUC 0.762, Predict's book alone 0.767, both 0.841; the guard value v adds nothing on top (-0.0003; v alone 0.779); each feed adds about +0.08 to the other, so a pull needs both | HOLDS (consumed days, first minute, post hoc, public tape) | 20261112_move_results.md; 50227d6e |
| Rest at the best by default and pull on P (the level goes in 24-274 ms) (FRONT-PULL, J-v3 rest lane, 9 days, first minute): beats matched random pulls on 9 / 9 days but loses to the guard trigger v <= -0.10 on 0 / 9 (-$48 vs -$20 a day w/o young at M 50; -1.18 vs -0.37 c/sh) and to J-v3 (-$52 a day); more pulling worsens each remaining fill (top-10% line -0.64, top-50% -1.30 c/sh; swept 48% -> 63%); a front-tier line adds +$1.3 a day (8 / 9); with nothing ahead (rule U) the same policies earn +$171-246 a day: the queue model decides the sign; fills during our cancel 6% of shares (-$6.5 a day) | HOLDS (replay rule M, B1a; L not run; consumed days, design after MOVE) | 20261113_front_pull_results.md; df320ba3 |
| The top makers' order shape in our replay (FRONT-PULL-SIZE, S4 rest lane, 9 days, first minute): size does not change per-share value (FPQ -1.18 / -1.19 / -1.18 c/sh w/o young at 5 / 30 / 100 sh, M 50; R410 NOT-SHOWN), so losses scale (FPQ-30 -$234, FPQ-100 -$562 a day); a 6-rung ladder at 30 sh loses on 9 / 9 days (LFPQ-30 -$5,095 a day, R411 LADDER-HURTS; deep rungs 68% swept, negative even at rule U); our kept 30-100 sh orders are filled 3+ times (41-61%) but 60% of their shares come from level-taking trades against the top makers' ~16%; guard trigger stays the best pull; at rule U single-level resting + guard earns +$213 / +$689 / +$1,203 a day at 5 / 30 / 100 sh: the queue model decides | HOLDS (replay rule M, B1a; L not run; consumed days, design after ORDERS) | 20261113_front_pull_size_results.md; b8e7592a |
| Why real fills behind the best pay and our rungs lose (DEPTH-TIMING: same days, tape and value function; WS-timed exact complete trades): behind the best the top 5 earn +0.89 c/sh at 1 s and the field +0.58 (rising with depth: +0.58 / +0.98 / +1.77 at 1 / 2 / 3+ ticks), our never-pulled 30-sh rungs -1.10; in the same tercile of our P our fills are worth about theirs (low P STOP +1.41 vs +1.45), but half our deep shares fill at high P (-1.35 / -3.64 STOP / THROUGH) against 19% of theirs (R422 PRESENT-WHEN-SAFE; R420 WITHIN-TYPE 73%, 9 / 9 days; R421 64% vs 37% stops): it is the timing of presence / pulls at every rung, not the replay | HOLDS (replay rule M for ours; 62% of our fills excluded as not WS-timed / complete, same rule as real) | 20261113_depth_timing_results.md; 586efcba |
| Pulling every ladder rung when risk is high (DEEP-PULL, 30 sh x 6 rungs, 9 days, first minute): the high-P share of our deep fills falls from 54% to 13% (top makers 19%) and the ladder gains +$4,033 a day over pull-at-the-best (9 / 9 days, R430 DEEP-PULL-HELPS), but no arm is positive (R431 0 of 3; R432 0 of 6 vs J-v3): LD90 -$1,062, LDG (guard trigger on every rung) -$137 at M 50 (-$16 at M 0; deep rungs -0.33 c/sh), +$781 at the front bound; deep shares still 61-69% swept vs the top makers' 36% with ~75k pulls a day: the remaining gap is queue place at the deep levels (each pull re-queues) | HOLDS (replay rule M, B1a; consumed days, design after DEPTH-TIMING) | 20261113_deep_pull_results.md; ffc7e475 |
| Fewer / stickier pulls on the guard ladder (KEEP-PLACE: strong-signal-only, near-best-only, immediate re-entry): none helps (R440 KPS / KPN HURTS 9 / 9 days, KPF NOT-SHOWN; R441 0 of 3, R442 0 of 6); the pulls are needed (near-best-only lets deep fills at high P back to 67%), and our 60-68% swept share does not fall with a third of the pulls: our new orders join behind queues that already exist. Gap review: size, persistence, levels, high-risk exposure, two-sided closed in replay; open = which trades fill us at a level (rung 0 48% / deep 63% level-taking vs the top makers' 20% / 36%), speed / fill clock, taking | HOLDS (replay rule M; consumed days, tuning variants) | 20261113_keep_place_and_gap_review.md; 0a5a6ee6 |
| The guard ladder's loss is on estimated-time trades (TIMING-SPLIT, C12): LDG (30 sh x 6 rungs, guard on every rung) earns +$54 a day w/o young at M 50 (+0.24 c/sh) and +$86 at M 0 on WS-timed trades (7 / 9 days, R451 WS-POSITIVE) and loses -$190 on E2-timed ones (46% of its shares; R450 E2-DRIVEN); its WS-timed fills behind the best earn +0.92 c/sh against the top 5's +0.86; real makers' E2-timed fills are also much lower (top 5 at the best +0.38 -> -0.08); at the best our WS aged fills +0.05 vs +0.38; at M 250 even WS loses (-$936). Qualifies FRONT-PULL-SIZE / DEEP-PULL / KEEP-PLACE totals | HOLDS (replay rule M; consumed days; WS-only is a selection) | 20261113_timing_split_results.md; 298390f7 |
| The gap at the best price is queue place (BEST-GAP, exactly-timed trades, rung 0 with guard pulls, 30 sh): top 5 +0.405 c/sh, ours +0.007 (gap +0.398); their at-best fills are 80% trades that stop at their level, ours 37% (63% level-taking); within type ours are close (STOP +0.47 vs +0.53). Levers: queue place (rule U bound) +0.660 (166%, largest on 9 / 9 days; R464 MAIN: QUEUE), latency FAST +0.111 (28%; ZERO +0.133), size +0.003 (1%), timing -0.133 (we already sit at the best at lower risk than they do) | HOLDS (replay rule M; U / ZERO bounds; consumed days) | 20261113_best_gap_results.md; a4767fc1 |
| Our queue place at the best is lost at entry, not by re-places (QUEUE-PLACE, guard ladder rung 0, M 50, exact trades): of the queue ahead at our fills (14.5 sh) 22% is our own re-places (80% guard pulls; R480 BEHIND-FROM-ENTRY, 7 / 9 days) and 78% was ahead before our first order at that price went live - mostly LATE (40%; R481 LATE-ENTRY, 8 / 9): our first order arrives a median 359 ms after the level becomes the best (5% within 97 ms), when only 12% of the eventual queue was there at the birth. Post hoc: at the birth we are OUT (guard pull + 250 ms clear wait, 60%) or HELD at the old price (38%). Orders already at the price before it became the best (PRE-KEPT) earn +0.20 c/sh (rung 0 +0.29) at 57-63% STOP vs +0.03 / -0.13 and 33-35% for joins after the birth (R482 BEING-THERE-FIRST-PAYS, 7 / 9); only 17% of our at-best shares. Top 5 at-best fills come 0.84 s into a level's life vs ours 2.1 s, same ~60% fallback mix | HOLDS (replay rule M; consumed days; WHY-LATE post hoc) | 20261113_queue_place_results.md; bad57131 / 2f7143ea |
| Pull only the prices at risk (PULL-DEPTH, guard ladder): at guard-on steps the move within 1 s is predictable out of day (AUC 0.75 for >= 1-3 ticks; R490 PREDICTABLE) and most signals are false alarms (60% of onsets: no move in 1 s; 20% one tick). Pulling a rung only if P(move >= its depth) >= 0.10 / 0.25 beats pulling every rung on WS-timed trades: +$23 / +$44 a day at M 50 (+$29 / +$60 at M 0), 9/9 and 8/9 days (R491 HELPS), 39-59% fewer pulls; the gain is behind the best (+$62 -> +$83 / +$103), not at it; fixed-depth keeping loses $600-1,000 a day; on E2 trades PD25 loses more, over all trades only PD10 beats LDG. Two sides after a signal: the hit side's new best already holds 128 sh at its birth (the top 5 take 42% of its fills at +0.30 c/sh vs field -0.11; front proven resting before the signal 18% vs 10%; R493 PROPORTIONAL 1.10); the other side's new best starts at 54 sh and fills in ~250 ms, fills in its first 100 ms earn +0.96 vs +0.45 c/sh (R494 EARLY-PAYS 9/9); the top 5 do not win that race (20% of first fills); our FAST path gets 30% more shares there, per share 6/9 (R495 NOT-SHOWN) | HOLDS at M 0 / M 50 on WS trades; FLIPS at M 250 (FUTURE-BEST post hoc: PD10 -$172, PD25 -$453 a day vs LDG on WS trades, 0/9 days); P model inherits PULL-DEPTH's nested-fold leak (FRONT-PULL score fitted with the held-out day), known-frame move target and missing-frame zeros (other session's review 20261113_pull_depth_model_integration_review.md; magnitude unmeasured) (replay rule M; consumed days; five arms) | 20261113_pull_depth_results.md; 6fa5182e / 7f5f6aa1; 20261113_future_best_results.md |
| Keep the protection side at the future best (FUTURE-BEST, guard ladder, the USER's rule): while the guard is on, pull an order only if the move is likely to go THROUGH its price (P(move >= d + 1) >= tau, PULL-DEPTH's model and tau) and keep the order at the predicted landing price. On WS-timed trades it beats PULL-DEPTH (FB10 +$18, FB25 +$27 a day at M 50; +$21 / +$48 at M 0; 8/9 and 7/9 days, R500 BETTER) and pulling everything (+$41 / +$71, R501 HELPS); the gain is on orders >= 500 ms, mostly behind the best. It does not hold at M 250 (every keeping arm, PD included, below LDG on 9/9 days) or over all trades (FB10 +$4 vs LDG, -$13 vs PD10; FB25 -$134 vs LDG; E2 losses grow). At the hit side's new best FB25 puts us there first (64% of our shares from orders resting before it became the best vs 16%), but those orders are run over half the time (50% STOP, PRE -0.22 c/sh; R502 NOT-SHOWN). With the realized move (OFB, perfect foresight; a diagnostic, not a bound) the same rule makes +$378 a day more than LDG on WS trades and +0.90 c/sh at 88% STOP on the new best (top 5 +0.30, 82% STOP): the rule is limited by predicting where the move stops | FB10 / FB25 gain NOT robust (flips at M 250, all trades); P model inherits PULL-DEPTH's nested-fold leak (FRONT-PULL score fitted with the held-out day), known-frame move target and missing-frame zeros (other session's review 20261113_pull_depth_model_integration_review.md; magnitude unmeasured); 30-fill day rule checked after review (7b5eced4, min 657, labels unchanged); replay rule M; consumed days | 20261113_future_best_results.md; 037ed9de / 6f13eb38 |
| The top makers handle the two sides oppositely after a danger signal (SIDE-MOVES, real WS legs with maker / order hash, fills = lower bounds). Protection side: they MOVE down to the new best: top-5 makers quoting the old best are filled at the landing price 2.75x as often as the field (15.4% vs 5.6%), a median 302 ms after it forms (field 441), their orders there usually first (median 0 sh filled ahead vs 9), 10% already resting there; they take 42.5% of its fill shares at +0.30 c/sh (field -0.11); run over at the old best proportionally (32.0% vs 33.9% at-best share; R513 PROPORTIONAL). Other side: they mostly KEEP their old quote (old order fills again 18.5% vs 7.2%, R511 TOP-KEEP-MORE 9/9), join the new price a little more (6.2% vs 4.5%, R510 TOP-REQUOTE-MORE 8/9), move as often as the field (5%); when they join they are first (R512 TOP-AHEAD 9/9); same as at calm improvements. Ours (simulated LDG rung 0): protection side no order at the birth 89%, at the new best in 35% of events at 347 ms; other side held at the old price 40% but gone within 1 s two-thirds of the time | HOLDS (descriptive; consumed days; fills only; KEPT favours larger orders; JOIN times are fill times) | 20261113_side_moves_results.md; e2626a18 / 36637b1f |
| Reaching the protection side's new best faster after a move loses; a faster path in general helps (MOVE-DOWN, rung 0 of the guard ladder). Today our order is there in 35% of events, live 347 ms after the birth (guard still on at the birth 83%, clears 236 ms later, then 250 ms + 40 ms). Entering at once when the move happens (MD) gets us there in 60% at 128 ms, with the fast path (MDF) 78% at 46 ms (R520 FASTER 9/9), but still behind 161-222 sh shown, filled when the move continues (31-37% STOP, -0.66 to -0.70 c/sh; R522 LOWER 9/9): rung 0 -$444 / -$398 a day vs LDG at M 50, every clock (R521 HURTS). The top makers' place there (0 sh filled ahead) comes from being in the queue at or before the birth, not from speed after it. Today's rule with the fast path (FAST, 5 ms order path + fastest feed) helps at every clock: +$19 / +$22 / +$12 a day at M 0 / 50 / 250, 9/9 each, all trades +$44 (R521 HELPS; first gain here that holds at M 250). Other side: our held old-best order is pulled by our own guard on that side within 1 s 52% of the time (drift 11%, still live 34%); the top makers keep theirs | HOLDS (replay rule M; consumed days; rung 0; FAST path hypothetical; MDP arm uninformative: entry not gated, churned) | 20261113_move_down_results.md; 3695d17c |
| The move after a fill at the best: BTC explains a third (out-of-day R^2 Binance 0.34, other venues 0.38) but none of the top makers' edge (2-3%); their edge (+0.21 c at 250 ms) is that fewer of their fills are in trades that take the whole level (57% goes with the book's reaction); in fills that leave the level standing there is no gap (-0.005 c) | HOLDS (descriptive; WS-timed fills only) | MOVE Part A |
| Lane J-v3 (v3's quote line only, NOT the frozen pm_ladder v3, C11) is the only positive simple replay arm (+1.03 c/sh, $14.7 a day at M 50 on the 9 days). The frozen v3 is under a forward test with its own engine | QUALIFIED: 75% of J-v3's value is young-order fills (AGE); every "v3" in S4 / QR / QR2 / AGE / GAP / CS means J-v3 | S4, QR2, AGE, pipeline review S1 |
| The top makers' edge is not in our guard's states | HOLDS | B12 |
| A pressure-feature guard does not change the guard | HOLDS | B14 |
| A queue-aware guard target equals the live guard; adding pressure is better at 1 s, worse at 60 s; not promoted | HOLDS | B16 |
| The guard is on (stay out) in ~90% of resting moments; 94% of its alarms see no harm within its own 500 ms horizon, but after those "false" alarms a resting order loses -1.7 to -1.9 c per moment over the next 4.5 s (9 of 9 days), twice the loss after no-alarm moments: the alarms are mostly early, not false. It separates activity (AUC 0.67-0.72) better than harm (0.60-0.65) | HOLDS (replay, never-pull path, orders >= 100 ms old; the AUC result is post hoc after an instrument fix) | FA, 20261112_false_alarm_results.md |

### C. Resting by default and pulling

| Finding | Status | Source |
|---|---|---|
| Orders held in advance gain no priority (live ~82 ms with the guard, 288-355 sh ahead) | HOLDS | B20 |
| Resting at the touch by default and pulling on a signal loses at every fill clock (eight arms) | HOLDS (11-08) | B21 |
| Zero own latency does not rescue resting; it helps the gated v3 (+0.25 c/sh) | QUALIFIED: the v3 gain is young-order exposed (C1) | B21b |
| A perfect keep / pull decision makes resting pay on 11-08 | HOLDS as a perfect-foresight ceiling only | S1 |
| A learned keep / pull model has skill (9 of 9 days) but its policy loses to the guard trigger and v3; the label is mostly re-entry luck | HOLDS as a negative; its KP numbers are biased in KP's favour (start price before known, P16; fold leakage through v; no decision during the unknown-fill window) | S4, S4 diagnostic, pipeline review F1 / F3 / S3 |
| Default-pull loop (Codex): no candidate keeps trading value at no higher inventory; a learned pull beats random but not the guard margin; a joint fill model improves calibration, not selection | HOLDS | 20261112_default_pull_review.md, 20261112_fill_value_review.md, 20261112_joint_fill_review.md |
| 250 ms danger-pull design | OPEN, never run | bbf0112d |

### D. Queue priority and speed

| Finding | Status | Source |
|---|---|---|
| A new best level holds ~52 sh at birth and ~155 sh 100 ms later; no tested rule buys priority at our latency | HOLDS (11-08) | review 6.5 |
| The queue ahead of our orders mostly joins after we could have been there (v3 41%, live 44%); latency 16% / 27%, re-places 15% / 14% | QUALIFIED: a model-based split (fixed 97 ms boundary, ghost queues, snapshot allowance) | QR parts A / C; review 10:42Z |
| v3's fills on young levels / first orders earn more than on old levels / re-places (+1.59 / +1.16 against about 0, 9 of 9 days) | QUALIFIED: young levels mean young orders (C1); not re-checked by level age | QR |
| QR speed sweep at d = 0 / 15 / 30 ms | **VOID**: the decision is scheduled before the level exists (C6) | QR part B; review 10:42Z |
| QR speed sweep at d = 55 / 97 / 150 ms | QUALIFIED: causal (55 ms assumes the 15.2 ms minimum feed; its decision is 0.2 ms early, immaterial); young orders (C1) | QR part B |
| Joining earlier at today's speed (no old levels, keep through flicker) does not beat v3's $/day | HOLDS | QR2 R290 |
| Keeping orders through guard flicker is worse per share | HOLDS | QR2 K / K2 |
| The 500 ms age cut: same $/day, better per share, less M 250 loss | QUALIFIED: its gain is young-only (AGE R312 GAIN-IS-YOUNG) | QR2, AGE |
| v3 + joining new levels at 30 ms (F30, Y+F30, QR2-fast) beats v3 (+8.7 $/day; +2.6 without young fills) | **VOID** (C6); SUPERSEDED by CS's causal speed test | QR2, AGE R312 / R313; CS |
| With causal clocks, v3 plus event-driven joins at new levels gains only with a faster order path: 10-20 ms paths +1.5 to +4.4 $/day without young fills (M 50, 8-9 of 9 days); a faster feed with today's 40 ms path gains nothing; slowest gaining arm = today's feed + 20 ms path; at M 250 everything still loses; joining at births beats random times | HOLDS (replay, on J-v3 not the frozen v3 (C11); young fills about 70% of the headline; mechanism not established; pending-cancel window ruled out post hoc; own-order overlap fills (P13) are 1-7% of shares and excluding them shrinks the gains by up to 14%, still 8-9 of 9 days) | CS, 20261112_causal_speed_results.md, pf_cs_overlap.py |
| The live engine churns: at-best orders cancelled 23-38 ms after going live, 82.5% re-placed; guard flicker | HOLDS | B13c / B13d |
| The live soft pull is a churn control with value about unchanged; merged (pmmmrust 0c3b115), not deployed | HOLDS | B15 |

### E. The top makers

| Finding | Status | Source |
|---|---|---|
| Their edge is not queue place, not Binance and not trend. It is earned within 250 ms of the trade report, on Predict.fun itself; they stay after being hit; about a quarter is their own taking; it holds to 60 s and settlement | HOLDS (mostly 11-08; B12g 9 days); B12g's settlement values carry the tie error (P10, 14 windows) | B12-B12g, B13b, B18 |
| The top makers' exits before level-taking trades (EXIT, public tape, 9 days, first minute; front orders proven by a sole-maker stop <= 1 s before): they do not leave more often than other makers (GONE 51% vs 46-62%; gap-matched -0.8 points, R390 NOT-SHOWN) but they leave at the right moments - GONE 28% -> 80% from the bottom to the top quintile of our P against 42% -> 63% for the field (9 / 9 days, every gap bin); the choice is made 100-1000 ms before the trade, not in the last 100 ms; they are 42% of first-in-queue volume against 34% of level-taken volume, 52 sh per leg when a level is taken while they are in it | HOLDS (public tape, consumed days; R392 post hoc robust) | 20261113_exit_results.md; 11fb2adc / 5abc1d96 |
| How the top makers work, order by order (ORDERS: raw match records' order hashes joined to step 1's legs, 9 days, first minute): they rest larger orders and keep them through fills - 25.8% of their orders filled by >= 3 trades vs 10.0% (R400, 9 / 9 days), 15.7% first in the queue at >= 2 trades vs 4.9% (R401, 9 / 9), median 30 sh filled per order (p90 100) vs 20 / 10; they do NOT create new levels more than their share (20.7% of creators vs 24.4% of at-best fills, R402 PROPORTIONAL); 6 prices per side per minute (p90 11), two-sided in 89% of windows, end |net| / gross 27%; their takes (12.6% of maker volume) mostly add to their position (R404 DIRECTIONAL at the line, 39.6% reducing). EXIT holds on exact, complete taker orders (EXIT-v2: 29% -> 79% vs 44% -> 64%, 9 / 9) | HOLDS (public tape, consumed days; sizes and lives are lower bounds from fills) | 20261113_orders_results.md; 9f8ec338 |
| Per share, our simulated J-v3 beats their real at-best fills (+1.03 against +0.23) | QUALIFIED by AGE: MIXED without young fills (8 / 6 of 9 days); J-v3, not the frozen v3 (C11) | GAP, AGE R314 |
| They trade 345x our shares; 60% of their money is made while v3's guard is OFF (+0.15 c/sh on 459k sh a day); only 17% of our fills share a taker order with them | HOLDS (theirs real) | GAP |

### F. Live and execution

| Finding | Status | Source |
|---|---|---|
| Live is not running (USER 2026-11-10); live = the user's call | HOLDS | HANDOFF |
| Slow cancels (0.5-5 s) 50-60 s into the window caused 11-08's first-minute loss | HOLDS; handed to the live session | B12b |
| Maker fee 0 in public match records; rebate unverified | OPEN | STATUS maker_fee_public, rebate |

### G. Pipeline review (2026-11-12)

| Finding | Status | Source |
|---|---|---|
| Three independent stage reviews (data / time alignment, features / guard, simulator / valuation): two ERRORs (tie outcomes P10; S4's `gap` before the start price is known P16) and the framing risk that "v3" in the replays is J-v3 (C11); the rest are RISK / NOTE items now in §2 | HOLDS; fixes proposed (user's call) | 20261112_pipeline_review.md |

## 4. Only live can answer

- Whether our young orders really fill, and at what value (C1).
- The fill clock: when trades match relative to their report (C2).
- Our real place in the queue (C5).
- All three need a live run with the A4b capture (USER D2).
