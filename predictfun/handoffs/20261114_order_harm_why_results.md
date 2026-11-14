# ORDER-HARM WHY: why the keep-vs-cancel model loses, against the top makers (2026-11-14, session ce68ebe0)

USER: "yes check why it suffers loss and performs worse. Check against top makers, is it timing issue, then check the model why it picks the wrong time".

**POST HOC and descriptive.** Chosen after POLICY v2's NOT-ACCEPTED result; it changes no verdict there.

## Answers in one line each

1. **Why it loses**
   - At M 50, against pulling everything (LDG), the model swaps losses rather than removing them:
     - it saves $99 a day on orders re-posted right after the guard clears;
     - it loses $93 a day on orders it keeps during the guard.
   - Net +$6 a day.
   - At M 250 the kept orders lose $1,358 a day, which is far more than the re-post saving (+$929).
   - Most of the kept-order loss is on E2-timed trades (timing uncertain by about +-250 ms). At M 50 that is -$116 a day; the WS-timed part is +$23.
2. **Timing against the top makers**
   - It is NOT a question of when we are in the book. Our fills come in safer states than the top 5's (the MIX term is positive at M 0 / 50).
   - The whole gap is inside the same state (W2 WITHIN-STATE in all 24 cells).
   - A third to a half of that gap is fewer STOP fills (queue place): about a third at M 50 and M 250, about half at M 0. The rest is same-type fills that are still worse (WT: SAME-TYPE-WORSE in 11 / 12 cells).
   - The gap grows steeply with our reaction delay (LDG, WS: +0.37 / +0.18 / -1.96 c/sh at M 0 / 50 / 250, against +0.55 for the top 5). The timing problem is speed inside the moment, not presence.
3. **Why the model keeps at the wrong time**
   - On its own labels the keeps are right: they beat pull-and-re-post on 5 / 5 days (LABEL-OK).
   - But both choices lose money, and the label only ranks the two.
   - Its own misses come in three kinds:
     - **SEEN:** it keeps through moves it can see, when a thick cushion is in front (MISSES-MOVES 4 / 5 days; -$34 a day at M 50).
     - **LATER:** moves that arrive after its decision (-$34 a day at M 50, -$1,101 at M 250).
     - **NO-MOVE:** small losses with no fair-price move (-$35 a day at M 50).

## Setup

| Item | Value |
|---|---|
| Code | `pf_order_harm_why.py` + `run_order_harm_why.sh` (38995519, committed before the run); post hoc follow-up `pf_order_harm_why_posthoc.py` (a83af1f2) |
| Run | claude slot. 15:38:12Z-15:52:11Z; follow-up 15:54Z |
| Days | POLICY v2's five test days (10-27, 10-31, 11-01, 11-02, 11-03; consumed), first 60 s, hard 50, fill clocks M 0 / 50 / 250 |
| Ours (simulated) | LDG, FB25C, OHQ0, OHQM re-run with recording wrappers; the decisions are unchanged |
| Theirs (real) | Every maker leg on the tape in the same window-sides. Makers ranked per day: top 5 / 6-20 / rest |
| State of a trade (same function for both) | **phase:** our guard at the last step <= anchor - 50 ms; ON split by time since onset, AFTER split by time since the clear, else CALM. **move:** dfv250 against the maker side. **depth:** at best / behind / inside |
| Timing class of our fill | **PEND:** our cancel was in flight. **KEPT-ON:** kept by a decision in this guard episode. **KEPT-AFTER:** kept through an earlier episode. **REPOST:** placed <= 100 ms after a clear. **CALM:** everything else. **OTHER-ON:** guard on, no decision |

**Controls** (all passed; a failure means no output):

- Our 512,172 fill rows equal POLICY v2's saved fills row for row (4 arms x 3 clocks x 5 days).
- Legs are aligned on every window-side.
- Accounting violations: 0.
- Look-ahead falsifier: 0 of 231,895.
- Selftests caught all 4 planted errors (no lag, KEPT without a decision, PEND not first, MIX / WITHIN swapped).
- Follow-up: WITHIN reconciles on 12 / 12 cells, and the swapped-value falsifier differs on 12 / 12.
- Peak memory 1,253 MiB.

## 1. Where the loss is (W1)

Whole portfolio, $ a day at the 60.2 s mark, hard 50:

| Clock | Class | LDG | FB25C | OHQ0 | OHQM | OHQ0 - LDG |
|---|---|---|---|---|---|---|
| M 0 | CALM | +90.2 | +113.4 | +110.2 | +125.1 | +20.0 |
| M 0 | REPOST | -9.5 | +24.7 | +36.4 | +19.4 | +45.9 |
| M 0 | KEPT-ON | 0 | -112.8 | -101.1 | -177.4 | -101.1 |
| M 0 | KEPT-AFTER | 0 | +58.0 | +32.9 | +86.4 | +32.9 |
| M 0 | PEND | -79.7 | -80.6 | -70.4 | -105.7 | +9.3 |
| M 0 | total | +0.9 | -4.4 | +3.5 | -54.2 | +2.6 |
| M 50 | CALM | +43.1 | +49.8 | +43.7 | +58.6 | +0.5 |
| M 50 | REPOST | -118.9 | -26.5 | -20.1 | -15.3 | +98.8 |
| M 50 | KEPT-ON | 0 | -145.4 | -93.1 | -254.4 | -93.1 |
| M 50 | KEPT-AFTER | 0 | -15.3 | -6.0 | -1.5 | -6.0 |
| M 50 | PEND | -95.7 | -154.3 | -87.2 | -140.2 | +8.5 |
| M 50 | total | -171.5 | -295.1 | -165.2 | -355.1 | +6.4 |
| M 250 | CALM | -136.1 | -197.5 | -164.2 | -190.4 | -28.1 |
| M 250 | REPOST | -1,787.4 | -544.4 | -858.3 | -427.2 | +929.1 |
| M 250 | KEPT-ON | 0 | -1,085.7 | -657.0 | -1,301.5 | -657.0 |
| M 250 | KEPT-AFTER | 0 | -1,053.9 | -701.1 | -1,160.3 | -701.1 |
| M 250 | PEND | -238.4 | -219.5 | -255.8 | -215.4 | -17.4 |
| M 250 | total | -2,161.9 | -3,127.1 | -2,643.7 | -3,307.2 | -481.8 |

The class with the most negative OHQ0 - LDG (computed) is KEPT-ON at M 0 and M 50, and KEPT-AFTER at M 250. Against FB25C, OHQ0 gains at M 50 on KEPT-ON (+52.4) and PEND (+67.0).

**By trade source.** Per share at 1 s, M 50, all ages:

| Arm | Class | WS-timed (c/sh) | E2-timed (c/sh) |
|---|---|---|---|
| OHQ0 | KEPT-ON | -0.165 | -1.553 |
| OHQ0 | REPOST | +0.506 | -1.224 |
| OHQ0 | KEPT-AFTER | +0.454 | -0.970 |
| OHQ0 | CALM | +0.326 | -0.081 |
| LDG | REPOST | +0.295 | -1.224 |
| LDG | PEND | -1.023 | -2.146 |

The top 5 on E2-timed trades: CALM +0.423, AFTER +0.665, ON -0.426.

**Horizon (W5).** OHQ0's KEPT-ON fills at M 50 are -0.804 c/sh at 1 s, -1.033 at 5 s and -0.761 at the 60.2 s mark: adverse at every horizon, so the loss is not an artifact of the 1 s mark. 73% of the KEPT-ON shares are THROUGH fills: -1.15 c/sh at the best and -1.16 behind (KT).

## 2. Against the top makers: is it timing? (W2, WT)

**Decomposition.** Per share at 1 s, WS-timed legs in the first 60 s, ours without fills on orders < 100 ms old. The top 5 earn +0.553 c/sh.

| Clock | Arm | Ours | Gap | MIX (state mix) | WITHIN (same state) | of WITHIN: TYPE-MIX (STOP share) | of WITHIN: SAME-TYPE |
|---|---|---|---|---|---|---|---|
| M 0 | LDG | +0.370 | -0.182 | +0.307 | -0.489 | -0.248 | -0.240 |
| M 0 | OHQ0 | +0.401 | -0.152 | +0.212 | -0.364 | -0.169 | -0.189 |
| M 50 | LDG | +0.184 | -0.369 | +0.262 | -0.630 | -0.232 | -0.391 |
| M 50 | FB25C | +0.234 | -0.318 | +0.203 | -0.521 | -0.159 | -0.356 |
| M 50 | OHQ0 | +0.188 | -0.364 | +0.148 | -0.511 | -0.171 | -0.324 |
| M 50 | OHQM | +0.085 | -0.468 | +0.058 | -0.524 | -0.162 | -0.344 |
| M 250 | LDG | -1.961 | -2.514 | -0.394 | -2.097 | -0.748 | -1.185 |
| M 250 | OHQ0 | -1.973 | -2.525 | -0.427 | -2.072 | -0.656 | -1.271 |

- **W2 label:** WITHIN-STATE for every arm and clock, on WS-timed trades and on all trades (24 / 24).
- **The state mix favours us at M 0 / 50.** We are filled less while the guard is on and more in calm phases:

  | Phase | Top 5 share of WS shares | OHQ0 share (M 50) |
  |---|---|---|
  | ON | 47.6% | 26.3% |
  | CALM | 21.8% | 39.5% |

- **WT label:** SAME-TYPE-WORSE in 11 / 12 cells, QUEUE-PLACE in 1 (M 0 LDG, where the two parts are equal).

**Same state, different value.** WS-timed, 1 s, ours at M 50 without young fills:

| State | Top 5 | Ours |
|---|---|---|
| CALM | +0.870 | +0.29 to +0.43 |
| AFTER < 250 ms | +0.982 | +0.55 to +0.93 |
| AFTER 250-1000 ms | +0.873 | +0.12 to +0.41 |
| ON < 100 ms | +0.328 | -0.59 to -1.10 |
| Moves > 1.5 c against the side | -0.338 (8.2% of their shares) | -2.3 to -2.6 (0.9-4.6% of ours) |

**STOP share by state** (the queue place), WS-timed, M 50:

| State | Top 5 | Ours |
|---|---|---|
| CALM, at the best | 80% STOP | 50-62% STOP |
| AFTER, at the best | 83% | 29-52% |
| ON, at the best | 85% | 14-36% |
| ON, behind | 73% | 12-36% |

Within a type our fills are still worse. In CALM at the best:

| Type | Top 5 | Ours |
|---|---|---|
| STOP | +0.78 | +0.49 to +0.57 |
| THROUGH | +0.23 | -0.26 to -0.27 |

## 3. The model's keeps (W3, W4)

**What the model saw at its last keep before a kept fill** ($ a day at the 60.2 s mark | sh/day | mean A_hat | median ms from that decision to the fill | mean size at better prices):

| Clock | Arm | SEEN (> 0.5 c at the decision) | LATER (move after the decision) | NO-MOVE | Largest loss |
|---|---|---|---|---|---|
| M 0 | OHQ0 | -15.4 / 1,020 sh / +0.085 / 38 ms / 578 sh | -0.5 / 94 sh | -51.9 / 24,139 sh / +0.043 / 136 ms / 180 sh | NO-MOVE |
| M 50 | OHQ0 | -33.8 / 1,408 sh / +0.087 / 36 ms / 792 sh | -33.5 / 1,257 sh / +0.033 / 49 ms / 627 sh | -34.9 / 25,976 sh / +0.043 / 148 ms / 196 sh | NO-MOVE |
| M 50 | OHQM | -49.6 / 2,814 sh | -80.5 / 2,664 sh | -128.6 / 46,409 sh | NO-MOVE |
| M 250 | OHQ0 | -18.1 / 1,921 sh | -1,100.8 / 25,343 sh / +0.033 / 48 ms | -234.9 / 35,402 sh | LATER |
| M 250 | OHQM | -55.3 / 3,734 sh | -1,921.1 / 45,821 sh | -480.0 / 60,458 sh | LATER |

- **SEEN:** kept despite a visible move, with a thick cushion in front (792 sh at M 50). That loses about -2.4 c/sh: the cushion does not protect against a sweep.
- **LATER:** the move came after the decision. It dominates at M 250 (a reaction-time limit, not a ranking error).
- **NO-MOVE:** most of the kept volume, at small per-share losses with no fair-price move behind them.

**The model's view on its labels** (v3 test-day rows; keep share | mean A_hat | mean realized G under the final policy | mean keep-all advantage):

| Rung | Move <= 0 | 0-0.5 c | 0.5-1.5 c | > 1.5 c |
|---|---|---|---|---|
| 0 | 0.75, +0.047, +0.045, -0.205 | 0.68, +0.032, +0.026, -0.239 | 0.35, -0.067, -0.104, -0.597 | 0.09, -0.502, -0.703, -1.653 |
| 1-2 | 0.62, +0.016, +0.014, -0.235 | 0.54, +0.008, +0.006, -0.243 | 0.32, -0.049, -0.060, -0.454 | 0.12, -0.380, -0.511, -1.341 |
| 3-5 | 0.58, +0.002, -0.004, -0.203 | 0.52, -0.002, -0.008, -0.198 | 0.33, -0.030, -0.024, -0.296 | 0.20, -0.252, -0.301, -0.914 |

- **LABEL-OK:** kept decisions' mean G is +0.020 to +0.032 c/sh, positive on 5 / 5 days. The model is calibrated where there is no move and somewhat optimistic in moves.
- **MISSES-MOVES:** kept decisions in moves > 0.5 c have mean G of -0.021 to -0.048, negative on 4 / 5 days. 11-01 is the exception (+0.016).

**Both branches lose.** The pull branch (re-enter after the clear) averages -0.25 c/sh on the labels, and keep-all to the end is -0.112 below it. G > 0 means "keep beats pull-and-re-post-at-once", not "keep makes money". In the replay this is the M 50 swap: REPOST +$99 against KEPT-ON -$93. The option of not re-posting right away (wait) is not in this model; it belongs to the other session's post-vs-wait model.

## Reading

- **Not timing of presence.** We are filled in safer states than the top 5, not worse ones.
- **The gap is inside the same moments.**
  - A third to a half is fewer STOP fills (we get the sweeps, they get the stops: queue place, earlier finding QR, we join levels late).
  - The rest is same-type fills that are still worse.
  - The gap grows sharply with our reaction delay (M 0 -> 50 -> 250). That points at speed within the moment.
- **The keep / cancel model works on its own terms but cannot fix this.**
  - It chooses between two losing branches. Its keeps beat re-posting at once, by little.
  - Its fixable error is keeping through visible moves when the cushion is thick (MISSES-MOVES; -$34 a day at M 50).
  - Moves after its decision (LATER) dominate at M 250 and need speed or a leading signal.
- **Watch out: the losses sit on E2-timed trades**, whose anchors are uncertain by about +-250 ms (C12).
  - OHQ0's WS-timed classes are positive at M 50 apart from KEPT-ON (-0.165 c/sh), PEND and a few OTHER-ON fills.
  - The top 5 stay positive on E2-timed trades outside the guard.

**Possible next steps (USER's call):**

- (a) A visible-move override for the keep model: pull when dfv250 > 0.5 c whatever the cushion, or a monotone constraint on dfv250. This targets SEEN, about $34 a day at M 50.
- (b) The re-post timing: the other session's post-vs-wait model.
- (c) The dominant gap is same-state fill quality and queue place. It needs speed and earlier queue joining, not a better keep rule.
- (d) A check of our E2-timed fills, where most of the loss sits.

## Files

- **Outputs:** `data/predict_fun/research/order_harm_why_20261114/` (`order_harm_why.txt`, `order_harm_why_posthoc.txt`).
- **Review copies:** `predictfun/reviews/order_harm_why_20261114/`.
