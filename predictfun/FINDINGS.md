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
| C3 | Report the cancel-pending share of fills | The replay credits 2.5x live's cancel-pending shares (B1a) |
| C4 | Treat replay volumes and $ totals as inflated | The replay credits 1.5-2x live's shares (B1a R24) |
| C5 | Rule M primary, L shown; U is a front bound only | M is unvalidated; a fifth of live's fills contradict the shown-size queue (B1a) |
| C6 | Every decision may use only information observable at its time. Write out observation, decision and placement clocks: feed >= 13.2 ms (median ~53), decision 2 ms, placement 40 ms. To model faster placement, change that leg explicitly. Never schedule a price-dependent decision before its information exists | QR's fast entries decided before the level existed (review 2026-11-12 10:42Z, 9053263d) |
| C7 | A whole-market markout is not our order's value: replay one order with queue position before recommending | Pro rata flow reversed the sign once (2026-11-10) |
| C8 | Name the days used. Parameters chosen on seen days are post hoc. A pass is a candidate; validation needs a freeze and >= 5 untouched days | Consumed days: 10-23..27, 10-31, 11-01..03 (9 replay days), 11-07, 11-08. Never read 11-04..06 or 11-09..13 (forward blocks) |
| C9 | When ours is simulated and theirs is real, say so in the table | GAP |
| C10 | No replay number is validated at the order level; only a live run with the A4b capture can validate one | B1a; USER D2 |

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
| v3 (pm_ladder v3) is frozen with a forward block and is the only positive replay arm (+1.03 c/sh, $14.7 a day at M 50 on the 9 days) | QUALIFIED by AGE: 75% of it is young-order fills | S4, QR2, AGE |
| The top makers' edge is not in our guard's states | HOLDS | B12 |
| A pressure-feature guard does not change the guard | HOLDS | B14 |
| A queue-aware guard target equals the live guard; adding pressure is better at 1 s, worse at 60 s; not promoted | HOLDS | B16 |

### C. Resting by default and pulling

| Finding | Status | Source |
|---|---|---|
| Orders held in advance gain no priority (live ~82 ms with the guard, 288-355 sh ahead) | HOLDS | B20 |
| Resting at the touch by default and pulling on a signal loses at every fill clock (eight arms) | HOLDS (11-08) | B21 |
| Zero own latency does not rescue resting; it helps the gated v3 (+0.25 c/sh) | QUALIFIED: the v3 gain is young-order exposed (C1) | B21b |
| A perfect keep / pull decision makes resting pay on 11-08 | HOLDS as a perfect-foresight ceiling only | S1 |
| A learned keep / pull model has skill (9 of 9 days) but its policy loses to the guard trigger and v3; the label is mostly re-entry luck | HOLDS | S4, S4 diagnostic |
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
| v3 + joining new levels at 30 ms (F30, Y+F30, QR2-fast) beats v3 (+8.7 $/day; +2.6 without young fills) | **VOID** pending a causal rerun: same schedule as QR's 30 ms arm (C6) | QR2, AGE R312 / R313 |
| The live engine churns: at-best orders cancelled 23-38 ms after going live, 82.5% re-placed; guard flicker | HOLDS | B13c / B13d |
| The live soft pull is a churn control with value about unchanged; merged (pmmmrust 0c3b115), not deployed | HOLDS | B15 |

### E. The top makers

| Finding | Status | Source |
|---|---|---|
| Their edge is not queue place, not Binance and not trend. It is earned within 250 ms of the trade report, on Predict.fun itself; they stay after being hit; about a quarter is their own taking; it holds to 60 s and settlement | HOLDS (mostly 11-08; B12g 9 days) | B12-B12g, B13b, B18 |
| Per share, our simulated v3 beats their real at-best fills (+1.03 against +0.23) | QUALIFIED by AGE: MIXED without young fills (8 / 6 of 9 days) | GAP, AGE R314 |
| They trade 345x our shares; 60% of their money is made while v3's guard is OFF (+0.15 c/sh on 459k sh a day); only 17% of our fills share a taker order with them | HOLDS (theirs real) | GAP |

### F. Live and execution

| Finding | Status | Source |
|---|---|---|
| Live is not running (USER 2026-11-10); live = the user's call | HOLDS | HANDOFF |
| Slow cancels (0.5-5 s) 50-60 s into the window caused 11-08's first-minute loss | HOLDS; handed to the live session | B12b |
| Maker fee 0 in public match records; rebate unverified | OPEN | STATUS maker_fee_public, rebate |

## 4. Only live can answer

- Whether our young orders really fill, and at what value (C1).
- The fill clock: when trades match relative to their report (C2).
- Our real place in the queue (C5).
- All three need a live run with the A4b capture (USER D2).
