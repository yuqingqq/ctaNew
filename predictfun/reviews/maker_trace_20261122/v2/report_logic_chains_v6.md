# TOP-3 MAKER LOGIC CHAINS v6 — competing transitions after a retreat, candidate continuation economics; v5 corrections in-band

2026-11-23, session aea5bfb9.

**Mandate:** review a91f6bb52's ONE next priority, for the USER goal "establish the action chains and analyze to get their logics".

**Sources:**
- contract `competing_transition_contract.md` (9909beacc);
- tools frozen bb4e64cf6;
- runs `mt_x_ct_v1` (0.5 CPU s) and `mt_x_ce_v1` (3.0 s), both rc 0;
- outputs `v2/chains/{competing_transition,continuation_econ}/`. Every number below is read from their `claims.json`.

**Population and limits:**
- the same 3,796 one-side retreat episodes as v5, on the 19 BOOK markets, Oct 24–27 (4 consumed days);
- descriptive;
- **no significance test, no interval, no validated rule**.

**Stop rule (declared before computation):** a thin or inconsistent answer ends this measurement line.

## 1. What happens next after a one-side retreat (all 3,796 reconciled; 0 class-direction violations)

| Next detected transition | Episodes |
|---|---|
| same side: restoring / partial / overshoot | 753 / 139 / 107 (999 = 26.3%) |
| other side: improves toward the spread | **1,853** |
| other side: retreats | 69 |
| both sides: both up / both down / narrow / widen / sub-label unknown | 227 / 220 / 226 / 2 / 6 |
| same side retreats further | 184 |
| no change within 1 s | 7 |
| censored (empty side) | 3 |

**Same-side response probability:** bounds 0.263–0.264, so censoring barely matters.

## 2. Conditions before the retreat vs same-side restoration (frozen contrasts, D0)

**Method:**
- harmonic-weighted risk difference (RD) over common-support strata (market × side × retreat size × time left);
- adequacy by counts; consistency = same sign at D0 and LAG, and on ≥ 3 of 4 days at both.

| Condition (exposed vs reference) | Restoration, all rows | Matched RD | Reading |
|---|---|---|---|
| **time left ≤ 30 s** vs 60–240 s | **34.1% vs 20.2%** (1,065 / 1,439 episodes) | **+14.8 pts** | **adequate, consistent** (4/4 days, both clocks) |
| **own best level thin (≤ 50 sh)** vs > 200 sh | **26.2% vs 31.7%** (2,835 / 240) | **−9.7 pts** | **adequate, consistent** (4/4 days, both clocks) |
| BTC 5 s moved against that side vs flat | 22.2% vs 26.6% (392 / 3,146) | +3.1 pts | adequate, consistent, but **fragile**: the unmatched difference has the opposite sign, and one market carries ~60% of the matched RD |
| BTC 5 s moved with that side vs flat | 29.9% vs 26.6% (244 / 3,146) | +5.3 pts | adequate, **inconsistent** (LAG: 2 of 4 days) |
| fair value > 1 c inside the post-retreat best vs ≤ 0 | 26.1% vs 27.2% (1,927 / 1,724) | −0.8 pts | adequate, **inconsistent** |

- Every contrast clears the frozen count floor (≥ 30 episodes, ≥ 10 responses and ≥ 10 competing transitions per group, ≥ 5 markets with ≥ 5 episodes), at both clocks.
- The answer is not thin: 3 consistent, 2 inconsistent.
- fair_gap uses the post-retreat best: it is known at the episode start, but it is not a wholly pre-retreat feature.

## 3. Timing (interval bounds, not latency)

**Response timing** (n = 999):

| Measure | Median | p90 |
|---|---|---|
| lower bound, max(0, response_lo − retreat_hi) | 0 ms | 91 ms |
| upper bound, response_hi − retreat_lo | 104 ms | 195 ms |
| observer endpoint difference | 54 ms | — |

710 of 999 responses (71%) arrive in the very next book frame. These are summaries of bounds, not a maker's reaction latency.

## 4. Candidate continuation economics (89 response links + 6 chains)

**Coverage:**
- 23 of the 89 linked orders have exact-leg accounting, including all 12 top-3 links;
- 66 are `unavailable` (makers outside the top-10 close accounting);
- no order is shared across links;
- all fill and close keys are unique (de-dup verified).

**Completed closes of the linked orders** (clean cohort, fees included, entry → close on the settlement clock):

| Window | Closes | Shares | Net |
|---|---|---|---|
| ≤ 10 s | 19 | 156.8 | +$2.04 |
| ≤ 60 s (= within the market window) | 23 | 213.9 | **−$9.65** |
| top-3 alone, within the window | — | — | −$12.10 |

**Residual:**
- closes of earlier inventory are kept apart: −$21.77;
- residual at the window END: −$7.10 for the available links;
- −$26.00 for the unavailable ones' recorded shares.

**Takes:** 375 distinct later takes, all inside END. Only 12 FIFO close rows (113.8 sh) are exits of these orders; a later take is not automatically an exit.

**Guards:**
- END bound, cohort, de-duplication and the tick-matched response context all hold;
- chain 3's response context is contextual-only (outside its conflicted post window).

**Reading:** restoring quotes that can be linked to a maker show no profit within the market window on this small, scenario-graded sample.

## 5. Corrections of v5 (review a91f6bb52)

1. "Quotes move mostly as a whole" is **withdrawn**.
   - Correct wording: 6,723 of 11,804 improvement creations (57%) fall in intervals where **both aggregate best prices changed within the same observed interval**.
   - Across all changed-touch pairs, two-side pairs (4,897) are fewer than one-side changed pairs (8,298).
   - "Other side first" means the next detected change was on the other side (1,853 improvements + 69 retreats), not a proven quote shift.
2. "Answered within milliseconds" is **withdrawn**.
   - Correct wording: **a subsequent touch change is usually detected within a few book frames**.
   - 54 ms is a frame endpoint difference, not a reaction latency (§3).
3. Exposure per market was 7.2–27.5 s, not 3–14 s. The frozen exposure-second floor failed structurally, and its `insufficient_support` verdicts stand.
4. Continuation guards: done in §4.

## 6. The logic chain as it stands (evidence grade in brackets)

1. **After one side of the touch retreats** [census + competing transitions]:
   - the next visible change is usually the other side improving (49%);
   - the same side is restored in 26% (mostly to the same price);
   - both sides change together in 18%.
2. **Restoration is more likely in the last 30 s** (+14.8 pts) and **less likely when the side's best level was thin** (−9.7 pts) [adequate, consistent across 4 days and both clocks; descriptive, not significance-tested].
   - BTC and fair-value conditions give no consistent direction.
3. **Who restores is mostly unknown** [91% of responses have no candidate owner]. The few linked restorations show no profit within the window [n = 23 exact].
4. **Takes** [v3]:
   - large inventory halves seconds with adding takes;
   - opposing takes follow own-fill bursts [timing-sensitive];
   - taking falls sharply in the last 30 s.
5. **Pulls** [v2]: adverse BTC in cancel-compatible windows [timing-sensitive].
6. **Queue front:** not identifiable from public data.

## 7. Where this line ends

The stop rule did not trigger, since three contrasts are adequate and consistent. What remains unidentified can't be resolved from public data:
- the makers behind most quote changes;
- their posting and cancel times;
- the queue order.

Any further step needs new evidence: untouched days for validation, or own-order data. Those are USER decisions, not another tuned variant.
