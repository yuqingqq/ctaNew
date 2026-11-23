# TOP-3 MAKER LOGIC CHAINS v3 — take logic v2 + queue position; supersedes v2 §3 in-band

2026-11-23, session aea5bfb9.

**USER goal:** "establish the action chains and analyze to get their logics": why the top makers act given market conditions, and how they get to the front of the queue.

**Sources** (every number is read from the run outputs and their `claims.json`):

| Part | Contract | Frozen | Run |
|---|---|---|---|
| TAKE-LOGIC v2 | b298933fb | ba2833329 | `mt_x_tl2_v1` |
| QUEUE-FRONT v1 | 6436ee97c + amendments 5c8a4b1a4 / f4c7e5cbd | 90e3c848a | `mt_x_qf_v1` |

Review af916aba6 findings 2–5 are corrected here.

**Population and limits:**
- ex-ante top 3 × the 19 BOOK markets, Oct 24–27 (4 consumed days);
- point estimates, no intervals;
- p = circular-shift or permutation tail fraction, not independent-day confidence;
- descriptive of these markets.

**Clock wording:** the anchor is a *fixed market-second, pre-earliest-match proxy*, and q is reconstructed venue inventory, not client-known inventory.

## 1. Take logic (ambiguity-aware; `v2/chains/take_logic_v2/`)

**Outcome:** every raw take (4,124 in the panel) counted by its direction relative to the anchor's inventory — *opposing* (against the position) vs *same-direction* (adds) — plus any-take and the variant-grouped reduction class (O4). Ambiguous takes are no longer dropped.

The bias the review suspected is real. At |q| ≤ 50 shares, 294 of 858 takes reduce under some inventory variants but not all; above 500 shares it is 2 of 328. In the selftest, a simulated random taker produces a spurious size effect under the v1 outcome (1.92×) and none under the opposing outcome (1.01×).

| Logic | Pooled (D0, full panel) | Rank 1 | Across the 6 views (full / clean × D0 / D1000 / pending) | Reading |
|---|---|---|---|---|
| Inventory size → more opposing takes (> 200 vs ≤ 50 sh) | 0.91, p 0.675, 15/18 markets | 0.77, p 0.890 | 0.90–0.93, never supported | **WITHDRAWN as a maker rule** (declared decision) |
| Inventory size → fewer same-direction takes | **0.51** | **0.46** | 0.44–0.58, lower-tail p 0.002 in every view | Robust: large inventory halves adding takes (declared ratio, lower tail; the verdict rule tests only increases) |
| ≥ 3 own maker fills in 5 s → opposing takes | **1.42**, p 0.012, 18/19 markets | **1.45**, p 0.012, 16/19 | D0 and pending supported (1.42–1.63); D1000 1.25, p 0.12 | **Timing-sensitive**: needs the fills of the last second before the take's earliest match |
| BTC with the inventory → opposing takes | 1.21, p 0.052 | 1.34, p 0.018, only 4 markets with exposure | — | insufficient support (a lead) |
| BTC against the inventory → opposing takes | 0.73 | 0.80 | — | not supported; no BTC-direction association passed this declared screen |
| Last 30 s → takes | opposing **0.44**, same-direction **0.03** | 0.45 / 0.03 | every view, lower-tail p 0.002 | taking falls sharply in the last 30 s; adding almost stops |
| Unwind edge ≥ 0.5 c | 1.08 | 1.19 (1 market) | — | not supported |

**Collisions:** 2,765 raw takes share a second with another take by the same maker (904 seconds with ≥ 2; 54 mixed-side seconds). Outcomes are de-duplicated per second, and the shift preserves raw counts.

## 2. Queue position (`v2/chains/queue_front/`)

Identification uses the venue's time priority (v2 report: 5 violations / 16,433): the first fill at a freshly created price level belongs to the order at its front. Taker records are timed by the book decrease they caused (amendment 2), which gives frame resolution instead of the 2-s settlement window.

**Coverage:** 21,385 fresh levels.
- front holder identified: 888;
- ambiguous: 3,419 (a sweep across several makers, or an untimed earlier record);
- no fill / unknown / unexplained decrease / gap: the rest;
- record timing: 11,837 book-timed, 4,741 book-ambiguous, 26,801 unmatched, 201 settlement-only.

**Where their filled orders sat when posted.** This is the cell state before each order's posting window (4,149 filled orders, outcome-selected):

| | Price had no queue before the order's posting window | Joined an existing queue | Window unbounded |
|---|---|---|---|
| rank 1 | 904 (48%) | 915 (49%) | 49 |
| rank 2 | 454 (47%) | 479 (50%) | 26 |
| rank 3 | 908 (69%) | 389 (29%) | 25 |
| top 3 | **2,266 (55%)** | **1,783 (43%)** | 100 |

- "No queue before the window" does not prove the front: another order can arrive first inside the window.
- Among joined-queue orders, the position at the first fill is unidentified for 82%, behind for 8.5%, and front-after-clearing for < 1%.

**Do they win the front of fresh levels more than their share?**
- The top 3 hold 95 of 888 identified fronts.
- Their front share is bounded between **2.2%** (identified only) and **34.8%** (adding every ambiguous level that includes them), against **11.7%** of the fills at those cells.
- `wins_front_more_than_fills`: **unidentified** (p 0.87; the lower bound exceeds the fill share in 0 of 19 markets).
- Rank 1: 42 fronts, bounds 1.0–19.8% vs a fill share of 5.3%.

**How they got the fronts they hold** (identified, top 3; `main_mechanism` = improve):

| Mechanism | Share |
|---|---|
| improving the price | 48% |
| resting deeper, behind the touch | 26% |
| re-posting a just-cleared level | 25% |

**Speed when re-posting a cleared level:**
- top-3 median 129 ms (n = 24) vs 348 ms for other holders (n = 271);
- faster in 9 of 14 evaluable markets, short of the declared 2/3, so `faster_than_others` is **unidentified**;
- by rank: rank 2 100 ms (n 5), rank 3 81 ms (n 6), rank 1 406 ms (n 13).

**Does the front pay?** Descriptive, all identified holders, gross 10-s markouts:

| Fills | Front | Behind |
|---|---|---|
| all | **+2.59 c/sh** (919 fills) | −1.26 (926) |
| deep levels | **+9.06** (152) | +3.74 (28) |
| improve | +3.19 | −1.18 |
| re-establish | −1.88 | −2.00 |

- Rank 1's own fronts: −4.19 front vs −2.13 behind (42 fills each).
- `front_pays` for the top 3: unidentified (5 evaluable markets).

## 3. Corrections (review af916aba6 findings 2–5)

**Census:** the base populations are counted separately: 9,026 maker-fill rows and 4,147 take records. The 6,485 attribute incidences (mint 3,709, multi-record 2,722, merge 16, self-trade 38) overlap those rows and are not extra executions. The earlier figure of "19,658 observed executions" summed them.

**Withdrawn or corrected wording:**

| Earlier wording | Now |
|---|---|
| "fast reaction within ~1 s" | withdrawn: alignment sensitivity only |
| "BTC direction is not the trigger" | no BTC-direction association passed this declared screen |
| the 16/18 market count | crude within-market consistency |
| "adds stop" in the last 30 s | same-direction takes fall to 0.03× |
| "markets with ≥ 5" | counts event-seconds |
| "mostly carried into resolution" | some markets end with sizeable recorded positions (v2: 7 of 19 ≥ 50 sh) |

**Stated as data:**
- 46 reduce-classified takes (13 event-seconds) whose side equals the anchor inventory sign;
- 3 overshoot records.

## 4. The action chain as it stands (rank 1 dominates; evidence grade in brackets)

1. **Post** [observed fills, outcome-selected]. About half of their filled orders were posted at a price with no queue; the other half joined an existing queue.
   - Of the fresh-level fronts they verifiably hold, most come from improving the price.
   - They do NOT win fresh-level fronts more often than their share of fills [unidentified at the lower bound].
2. **Fill.** Their position at the first fill is mostly unidentified from public data.
3. **After fill bursts** → take the other side [supported at D0 and with pending legs, timing-sensitive].
   - When inventory is large they almost never add by taking [robust].
   - Size does not raise opposing takes [withdrawn].
4. **Near the end**, taking falls sharply and adding almost stops [robust].
5. **Pulls** [v2]: adverse BTC in the last second of cancel-compatible windows, under both references [timing-sensitive; pull vs exhaustion tied].

**Not identified:**
- private decision times;
- unfilled orders and unfilled takes;
- the front holder of 3,419 ambiguous fresh levels;
- queue position at the first fill for most orders.

**Next measurement for the USER goal:** the posting decision logic. On a decision-time population of market-seconds: when do fresh levels get created at the touch (improve) or re-established, given fair value vs the touch, spread, BTC and the holder's inventory? Identity is known only for levels that fill, so the anonymous creations stay in the denominators.
