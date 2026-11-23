# TOP-3 MAKER LOGIC CHAINS v5 — touch transitions and quote responses; v4 corrections in-band

2026-11-23, session aea5bfb9.

**Mandate:** review 01a72b4e2's ONE next priority, for the USER goal "establish the action chains and analyze to get their logics".

**Sources:**
- contract `touch_response_contract.md` (668e560db);
- tools frozen b4fbf5de2, including the integrator's chains_v4 adapter alignment, checked against the real pilot;
- runs `mt_x_tt_v1` (15.9 CPU s) and `mt_x_ch4_v1` (2.6 s), both rc 0;
- outputs `v2/chains/{touch_transition,chains_v4}/`. Every number below is read from their `claims.json`.

**Population and limits:**
- the 19 BOOK markets, Oct 24–27 (4 consumed days);
- point estimates, no intervals;
- descriptive of these markets;
- frames are reconstructed venue-clock book snapshots, not private maker receipt times.

## 1. How the best prices move (native book frames)

**Census:** 117,888 adjacent usable-frame pairs; 10,095 lie outside the window.

| Class | Pairs |
|---|---|
| unchanged | 94,246 |
| two-side repricing (both bests move in one frame) | **4,897** (both down 2,255, both up 2,141, narrow 400, widen 101) |
| one-side improvement | bid 2,334, ask 2,168 |
| one-side retreat (spread opens) | bid 1,906, ask 1,890 |
| unknown | 352 |

**Where the 11,804 price-improvement creations happen:**
- **6,723 (57%) inside two-side repricing frames**, where the whole quote shifts at once;
- 2,678 in one-side bid improvements and 2,354 in ask improvements;
- 49 unsupported (crossed or empty side).

**Of the 8,821 the v4 anchor mask excluded:** 5,362 are two-side, 1,824 bid, 1,601 ask, 34 unsupported. That mask removed most improvement activity: 3,664 of 4,484 improvement event-seconds.

## 2. What follows a one-side retreat (3,796 response episodes)

**Retreat sizes:** 1 tick 2,667; 2 ticks 607; ≥ 3 ticks 522. Total exposure 292 s.

**Within the 1-s horizon, a retreat almost never stays unanswered:**

| What came next | Episodes | Share |
|---|---|---|
| the same side moves back toward the spread (response) | 999 | 26% |
| the **other side** moves first | 1,922 | 51% |
| both sides move together | 681 | 18% |
| the same side retreats further | 184 | 5% |
| a gap | 3 | — |
| **no change at all within 1 s** | **7** | — |

**The same-side responses are fast and mostly exact restorations:**
- median 54 ms (p25 48, p75 96, p90 141); 710 of 999 come in the very next book frame;
- types: **restoring 753 (75%)**, partial 139, overshoot 107;
- after a 1-tick retreat the response restores in 641 of 706 cases; after ≥ 3 ticks partial responses dominate (73 of 134).

**State conditions** (response hazard per exposure-second, MH in market × side × retreat-size × time-left strata, 500 within-stratum permutations).

The frozen support rule needs ≥ 30 exposed and ≥ 30 reference exposure-seconds per market, but these episodes last tens of milliseconds (3–14 s of exposure per market). So **every verdict is `insufficient_support`**. The pooled ratios are descriptive only:

| Contrast | Pooled ratio | p |
|---|---|---|
| BTC 5 s moved with the retreated side | 1.34 | p_hi 0.040 |
| BTC 5 s moved against it | 1.22 | p_hi 0.09 |
| fair value > 1 c inside the post-retreat best | 0.98 | — |
| thin own best level before the retreat | 0.72 | p_lo 0.018 |
| last 30 s | 1.24 | p_hi 0.004 |

The support floor was not changed after the pilot.

## 3. Who restores (the maker layer)

**Of the 999 responses:**
- **910 have no candidate owner**;
- 89 are scenario-graded candidates (0 constrained);
- of the 89 candidates, 77 are other makers and only **12 are top-3** (rank 1: 4, rank 2: 3, rank 3: 5).

Under masked netting, another first address is possible for 22 of the 89.

Continuations of the candidate orders (same-hash fills, endings, takes, markouts) are in `continuations.csv`. They are recorded rows of candidate orders, not an identified policy. The table never sums whole-book close values per candidate.

## 4. Six chains v4 (`chains_v4/chains.md`)

**The corrections hold** (every predicate in `claims.json` is true):
- the PRE-STATE link carries the post grade; no chain is ONE-HISTORY unless G2, and chain 3 (G1) is now TIED with `unresolved_outside_window`;
- not-evaluated alt-netting reads `not_evaluated`, never False;
- gross markouts are labelled "gross c/sh";
- the summary is recomputed from the graded records.

**Transition context:** each chain's post cell is matched to its frame-pair class. For example, chain 2's creation sits in a two-side both-down repricing frame, chain 4's in a both-up frame, and chain 6's in an unchanged frame. All transition links are TIED: the post could also have joined a shown cell, or been re-created masked inside the window.

**Overall:** pre-state, transition, post, queue, ending and repost are tied in all six chains. Takes and value have one history, except chain 4, which has no take in its window.

## 5. Corrections of v4 (review 01a72b4e2)

1. The v4 improve ratios describe **seconds starting with room to improve**. 3,664 of 4,484 improvement event-seconds and 8,821 of 11,804 creations were excluded. "Mechanically possible in only 1,116 rows" is **withdrawn**.
2. `re_establish` = **recent cell reappearance with print-compatible clearing**. "Emptied by matched fills" and "adverse moves cause the clearing" are **withdrawn**. `p_fair` is the existing trailing BTC reference, not a maker belief or a true value.
3. The v4 statement "every pre-state is TIED" was wrong for chain 3 in v3. v4 grades it (G1 → TIED, unresolved).
4. Not-evaluated queue rows and the gross labels are handled as in §4.

## 6. The logic chain as it stands (evidence grade in brackets)

1. **Quotes move mostly as a whole** [census]. 57% of price improvements happen while both best prices reprice in the same frame.
2. **A one-side retreat is answered within milliseconds** [census + episodes]. In half the cases the other side moves first, so the quote shifts. In a quarter the same side is restored, typically to the same price, in the very next frame (median 54 ms).
3. **The conditions that speed responses are untested** [`insufficient_support` under the frozen floor]. The descriptive ratios lean toward BTC movement and the last 30 s.
4. **Who does it is mostly unknown** [96.5% of creations and 91% of responses have no candidate owner]. Top-3 candidates are few (12 of 999 responses).
5. **Takes** [v3]:
   - large inventory halves seconds with adding takes;
   - opposing takes follow own-fill bursts [timing-sensitive];
   - taking falls sharply in the last 30 s.
6. **Pulls** [v2]: adverse BTC in cancel-compatible windows [timing-sensitive].

**Not identified:** private maker identities for most quote changes, their posting and cancel times, and the queue order behind fresh levels.
