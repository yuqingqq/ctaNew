# TOUCH-TRANSITION-AND-RESPONSE v1 — contract (frozen before computation)

2026-11-23, session aea5bfb9.

**Mandate:** the ONE next priority of review 01a72b4e2 (`reviews/automatic/20261123_posting_chain_review/report.md`, findings 1–4).

**The question:** after the touch retreats, clears or moves, what observable market state precedes the replacement or improvement, which maker histories are compatible with it, and how does that quote's fill / turnover chain develop?

This serves the USER goal: action chains and their logic, including how quotes reach the front.

**Budget:** ≤ 0.040 core-hours all-in, including ≤ 0.010 for the review. The caps are 2.00 task / 5.30 line under the USER's standing CPU authorization; the ledger stood at 1.8733 before this round.

**Data:**
- the same 19 BOOK markets, Oct 24–27 (consumed);
- the stored frames / BTC / ledgers through `chain_context` and queue_front's frame handling;
- no new days, no fair-value model, no threshold search, no replay.

**Two tools:**

| Tool | Sections | Output (write-once) |
|---|---|---|
| `tools/touch_transition.py` | §1–§4 | `v2/chains/touch_transition/` |
| `tools/chains_v4.py` | §5 + corrections | `v2/chains/chains_v4/` |

Each tool has a selftest.

## §1 Native-frame transition census

**Unit:** each pair of adjacent usable depth frames (f−1, f) in a market, with the interval (upd(f−1), upd(f)].

**Class** (mutually exclusive, from the best bid b and best ask a in f−1 vs f, YES-space ticks):

| Class | Rule |
|---|---|
| `unchanged` | b and a unchanged; sizes may change |
| `retreat_bid` / `retreat_ask` | exactly one side's best moves AWAY from the spread (b down or a up); the other side unchanged. The spread opens. |
| `improve_bid` / `improve_ask` | exactly one side's best moves TOWARD the spread; the other side unchanged |
| `two_side` | both bests change, in any directions. Sub-labels: both up / both down / widen / narrow. |
| `gap` | upd(f) − upd(f−1) > 1,000 ms |
| `unknown` | either best is unknown or the side is empty |

**Joint timing:** a same-frame retreat-and-refill is never split into an invented order. A frame pair that both retreats and refills one side shows only its net change, and is classed by that net change.

**Cross-table:** the census × anonymous fresh-cell creations / reappearances (queue_front E1 cells), by frame pair.

**Reconciliation, required:** each of posting_opportunity's 11,804 `improve` creations maps to exactly one frame pair, giving its class or an explicit unsupported status. So do the 8,821 that the v1 anchor mask excluded.

## §2 Response episodes (sequential, one-side spread openings only)

**Episode:** one per `retreat_<side>` frame pair at f.
- **Exposure:** starts at upd(f).
- **Step size:** the retreat size in ticks, Δ = |best(f−1) − best(f)| on that side.
- **Ends at the first of:**

| End | Rule |
|---|---|
| (a) `response` | a frame where that side's best moves toward the spread: restoring (back to best(f−1)), partial (better than best(f), worse than best(f−1)) or overshooting (better than best(f−1)) |
| (b) `censored_other` | any other touch transition: the other side moves, or `two_side` |
| (c) `censored_gap` | a gap / unknown |
| (d) `no_response_1s` | the 1-s response horizon, upd(f) + 1,000 ms |

- **Exposure time** = end − upd(f) in ms. The full termination census is published.
- A response with no frame between upd(f) and the response frame is timed only to that interval; its bounds are saved.
- The `two_side` population is SEPARATE: a descriptive table of sub-labels × state, never a forced sequential response.

**State** (pre-transition frame f−1, the BTC grid ≤ upd(f−1), fills with hi ≤ upd(f−1); no next-frame feature):

| Feature | Definition |
|---|---|
| spread | before, in ticks |
| Δ | the available price gap |
| own-side touch depth | in f−1, posting_opportunity's bins |
| btc_5s_aligned | for the retreated side, posting_opportunity's definition |
| fair_gap | for the retreated side: the same existing `p_fair` vs the post-retreat best best(f), posting_opportunity's bins (room to restore) |
| time_left | take_logic bins |
| side | |

**Hazard comparisons:** response hazard = responses / exposure-seconds.

**Contrasts:**
- btc with vs flat, and against vs flat;
- fair_gap > 1 vs ≤ 0;
- depth ≤ 50 vs > 200;
- time-left ≤ 30 s vs 60–240 s.

**Strata:** market × side × Δ bin (1 / 2 / ≥ 3 ticks) × time-left bin. The time-left contrast omits its own bin.

**Null:** within-stratum permutation of the exposure label across episodes: 500 draws, seed 20261123, p_hi / p_lo = (1 + #) / 501.

**Market consistency** uses the within-market MH ratio. Verdicts follow posting_opportunity's rule (supported / supported_lower / not_supported / insufficient_support).

`timing_sensitive`: the verdict changes when the state is read at the frame before f−1. That is a state-lag sensitivity, not a reaction delay.

**Also reported:** the time-to-response distribution (ms) of uncensored responses, and the restoring / partial / overshoot mix.

## §3 Maker-compatible histories (separate layer)

**Candidate owner** of a response's new best cell: queue_front_v2's candidate first filler of that fresh cell, graded as in posting_opportunity §3 (window containment and post grade). Otherwise `owner_unknown`.
- An unknown owner stays in every denominator and is never a maker WAIT.
- The eventual filler is never stated as the creator.

**Continuation of a candidate-linked response** (keyed rows only):
- price / size and posting-time bounds;
- repeated same-hash fills;
- withdrawal / exhaustion and successor alternatives;
- subsequent recorded takes (take_logic_v2);
- the existing 10 / 60 s and END views;
- completed turnover / residual, from the existing accounting.

**Tables** use all supported linked cases, with shared valid cohorts and the attribution overlap. A whole-book close value is never summed once per candidate creation.

## §4 Combined gate (fixtures, each with a mutant caught)

1. A same-second spread-one → retreat → improvement sequence (the case the v1 anchor mask dropped): it must be counted as a retreat episode with a response.
2. Simultaneous best-price changes: they must be `two_side`, never sequential.
3. Masked cancellation / replenishment: owner candidates keep their alternatives.
4. A conflicted posting window: no identified pre-state (with §5).
5. Later same-hash fills: counted as one order.
6. A missing book: a status.
7. A later-frame perturbation: earlier episodes are unchanged.
8. Next-frame leakage in the state: the mutant must be caught.

## §5 Six chains v4 and corrections (chains_v4)

**The same 6 predetermined episodes.** Corrections:

| Finding | Correction |
|---|---|
| 3 | The PRE-STATE link carries the post grade. G1 / G3 / G0 windows, and G4 windows (earlier masked posting possible), get an `unresolved_outside_window` alternative and are never ONE-HISTORY unless the grade is G2. The summary is recomputed from the graded records. |
| 4 | An unknown alt-netting state stays `not_evaluated`, never False. |
| 4 | Gross markouts are labelled "gross c/sh". |

Each chain's post cell is matched to a §1 / §2 transition or episode where its creation interval allows. The response context and candidate grade are shown.

## Receipt corrections (in `v2/report_logic_chains_v5.md`)

**Finding 1:** the v4 improve ratios describe "seconds starting with room to improve". Publish the excluded counts: 3,664 of 4,484 improvement event-seconds; 8,821 of 11,804 creations. Withdraw "mechanically possible in only 1,116 rows".

**Finding 2:** `re_establish` = "recent cell reappearance with print-compatible clearing". Withdraw "emptied by matched fills" and "adverse moves cause the clearing". `p_fair` = the existing trailing BTC reference, not a maker belief.

**Findings 3–4:** as §5.

## Cost

| Builder | Budget (CPU s) |
|---|---|
| touch_transition | ≤ 30 (≤ 2 selftests + one Oct 24 pilot) |
| chains_v4 | ≤ 10 |

- Full runs projected ≤ 50 s; refuse above 120 s.
