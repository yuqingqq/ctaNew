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
