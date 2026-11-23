# TOP-3 MAKER LOGIC CHAINS v4 — posting opportunities, queue-front corrections, six complete chains

2026-11-23, session aea5bfb9.

**What this does:** answers review 3f640ad9b's ONE next priority, for the USER goal "establish the action chains and analyze to get their logics".

**Sources:**
- contract `posting_chain_contract.md` (8e09f3391);
- tools frozen b9fde691c;
- runs `mt_x_qf2_v1` / `mt_x_po_v1` / `mt_x_ch3_v1` (all rc 0; 15.8 / 16.6 / 2.5 CPU s);
- outputs `v2/chains/{posting_opportunity,queue_front_v2,chains_v3}/`. Every number below is read from their `claims.json`.

**Supersedes in-band:** v3 §2 (queue) and v3 §1 wording per §6.

**Population and limits:**
- the 19 BOOK markets, Oct 24–27 (4 consumed days);
- point estimates, no intervals;
- p = circular-shift tail fraction (500 shifts);
- descriptive of these markets.

## 1. Posting opportunities — when new price levels appear (the market layer)

**Unit:** every market × side × second over the full 300-s windows: **11,400 rows, all with a usable book.**

**Events:** a visible fresh-level creation in the next second. These are market opportunities, not identified maker decision times.
- 19,991 creations;
- statuses: 1,159 straddling an anchor and 245 outside the population;
- types overlap: improve 11,804, deep 8,186, re-establish 3,996, empty touch 1.

**Verdict rule** (frozen): RR > 1, p_hi ≤ 0.05, and RR > 1 in ≥ 2/3 of ≥ 5 supported markets. "Timing-sensitive" means the verdict changes when the state is read 1 s earlier.

| Condition before the second (state at t) | Any new level | Re-posting a just-cleared level | Placing deeper (behind the touch) | Improving the price |
|---|---|---|---|---|
| spread ≥ 2 ticks vs 1 | **1.36×, 16/16 markets** | 1.49×, timing-sensitive | **1.38×, 15/16** | 3+ vs 2: 1.22×, only 4 markets: insufficient |
| thin own best level (≤ 50 vs > 200 sh) | **1.12×, 17/19** | **1.44×, 18/19** | **1.49×, 19/19** | insufficient |
| fair value > 1 c better than that side's best | 1.09×, timing-sensitive | **1.38×, 16/19** | 0.89×: not supported | 1.14×, 2/5: not supported |
| BTC 5 s against that side | 1.04×: not supported | **1.50×, 8/8** (partly mechanical: adverse moves cause the clearing that re-posting requires) | 1.50×, timing-sensitive | insufficient |
| BTC 5 s with that side | 1.13×, timing-sensitive | 0.86×: not supported | 1.46×, timing-sensitive | insufficient |
| last 30 s vs 60–240 s left | **2.04×, 19/19** | **2.70×, 17/19** | **7.60×, 19/19** | insufficient |
| recent taker flow (> 0 vs 0) | insufficient: flow is present in nearly every second | | | |

**Reading the table:**
- New levels appear more when the spread is wide and the side's best level is thin.
- A cleared level is re-posted more when fair value sits well inside that side's best price.
- Deep placements surge in the last 30 s.
- Improving the price is mechanically possible in only 1,116 rows (spread ≥ 2), too few for any verdict.

## 2. Who created them (the maker layer, separate)

**Owners:**
- **19,282 of 19,991** creations (96.5%) have no candidate owner.
- 709 have a candidate first-filler whose own posting window contains the creation: 706 are scenario-graded, 3 bound-constrained.
- Top 3: **73 candidate links** (70 scenario, 3 constrained).

**Type mix** (types overlap; candidate links, never identified creators):

| | Improve | Deep | Re-establish |
|---|---|---|---|
| top-3-linked creations | 73% | 27% | 26% |
| all creations | 59% | 41% | 20% |

**Inventory:** only at the 3 constrained links, so no reading is possible. The anonymous creations stay in every denominator; an unknown owner is not "no posting".

## 3. Queue-front corrections (queue_front_v2; review findings 1–6)

**Fresh levels:**

| Status | Count |
|---|---|
| front_candidate | 845 |
| ambiguous | 3,382 |
| unexplained decrease | 1,378 |
| no fill | 10,684 |
| unknown | 5,018 |
| gap | 78 |

**No holder is certified.** A cancellation and replacement inside one frame interval is invisible, so masked netting is possible at every level with an earliest record (5,605). For 233 of the 845 candidates (27.6%), another address could have been first under such a history.

**Straddle fix:** a settlement-only record must lie wholly inside the creation interval.

**Front-win:** **unidentified**. The fill share lies inside both the scenario bound and the masked-netting bound:
- top 3: scenario 2.1–35.3%, masked netting 2.7–38.9%, vs 11.5% fill share;
- rank 1: 0.9–20.0% vs 5.3%;
- p_hi 0.92.

The bounds hold within the declared reconstruction scenario; they are not certified real-world bounds.

**Value by role** (all makers, gross 10-s, c/sh; descriptive):

| Role | 10-s markout | Valid rows | Valid share of quantity |
|---|---|---|---|
| earliest set | +2.32 | 325 / 870 | 39% |
| later fills, other orders | −2.07 | 338 / 475 | 68% |
| later fills of the same order | +0.06 | 161 / 380 | 32% |

The valid shares differ, so the roles are not comparable. **"Does the front pay?" is withdrawn as a queue-effect inference.**

**Endpoint occupancy** (renamed). For rank 1's filled orders:
- `cell_occupied_at_window_lo` 915 / 1,868;
- `cell_empty_at_window_lo` 904 / 1,868.

These are read at the inferred posting window's LOWER endpoint, with post grades, and are never the queue state at the actual post. The v3 sentence "55% were posted with no queue" is withdrawn.

**Mechanism and speed:** each is the anonymous cell-creation context associated with a candidate first filler. Speed is the cell-event endpoint proxy, not a maker repost latency.

## 4. Six complete chains (`chains_v3/chains.md`)

The same 6 predetermined episodes: per rank, the first two in case order.

**In every chain:**
- **Tied:** pre-state, post, queue, ending and repost.
- **Agree:** takes (one take history, except chain 4 with no take in its window) and value (one history).

**The hidden actions that change the explanation:**
- the posting instant, inside windows up to ~78 s;
- the submitted size: exhausted at the filled size vs a cancelled remainder;
- the orders ahead at the first fill;
- pull vs exhaustion at the end;
- a new submission vs an already-resting order vs an unrelated successor.

**Known display limitation:** take / fill ids are shortened to 13 characters. Counts are unaffected.

## 5. Take wording (review finding 7) and process note (finding 6)

**Take wording:**
- "Large inventory roughly halves the **seconds containing same-direction takes**" — not trade counts or volume.
- D1000 lags the whole state by one second. It does not isolate the last second's fills or measure a reaction latency.
- 2,765 = extra raw records beyond the first per occupied second: 904 seconds hold 3,669 records.
- The lower-tail ratios are descriptive associations under a fixed-path shift null.

**Process note:** QUEUE-FRONT v1 amendments 1–2 were adaptive development on consumed days. The pilots also printed economics and readings. This round's tools were frozen before the full runs.

## 6. The logic chain as it stands (evidence grade in brackets)

1. **Market posting opportunities** [supported associations, market layer]:
   - new levels follow wide spreads and thin best levels;
   - cleared levels are re-posted when fair value sits inside the touch;
   - deep placements surge near the end.
2. **Who posts** [mostly unidentified]: 96.5% of new levels have no candidate owner. Top-3-linked ones lean toward price improvement (73% vs 59%) [scenario-graded candidate links].
3. **Queue front** [unidentified]: no holder is certified, and the front-win share is unidentified.
4. **Takes** [v3]:
   - large inventory halves seconds with adding takes [robust];
   - opposing takes follow own-fill bursts [timing-sensitive];
   - taking falls sharply in the last 30 s [robust];
   - the size trigger is withdrawn.
5. **Pulls** [v2]: adverse BTC in cancel-compatible windows [timing-sensitive; pull vs exhaustion tied].

**Not identified from public data:**
- who created most levels;
- who held the front;
- posting and cancel times;
- the size of unfilled orders.
