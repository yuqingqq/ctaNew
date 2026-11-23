# POSTING-OPPORTUNITY-AND-CHAIN v1 — contract (frozen before computation)

2026-11-23, session aea5bfb9.

**Mandate:** the ONE next priority of review 3f640ad9b (`reviews/automatic/20261123_take_queue_review/report.md`, findings 1–7). It serves the USER goal: "establish the action chains and analyze to get their logics" — why the top makers act given market conditions, and how they get to the front of the queue.

**Budget:** ≤ 0.040 core-hours all-in, including ≤ 0.010 for the combined review. The cap was raised 1.90 → 1.95 under the USER's standing CPU authorization; the ledger stood at 1.8463 before this round.

**Data:**
- the same 19 BOOK markets, Oct 24–27 (consumed), with the existing book / BTC context builders and clocks;
- a pilot with a projection before the full packet;
- no new days, model family, policy test or validation.

**Three tools:**

| Tool | Sections | Output (write-once) |
|---|---|---|
| `tools/posting_opportunity.py` | §1–§3 | `v2/chains/posting_opportunity/` |
| `tools/queue_front_v2.py` | §5 (corrections) | `v2/chains/queue_front_v2/` |
| `tools/chains_v3.py` | §4 | `v2/chains/chains_v3/` |

Each tool has a selftest. Frozen v1 tools and outputs are never edited.

## §1 Neutral market-clock population (posting_opportunity)

**Unit:** market × side (bid / ask, in the book's YES-space convention) × second t, for every t over each market's full 300-s window: 19 × 300 × 2 rows. This includes seconds with no fill and no creation.

**Outcomes:** visible fresh-cell creations on that side with their creation interval (upd(f−1), upd(f)] inside (t, t + 1].
- A fresh cell is one shown 0 / absent in usable frame f−1 and > 0 in f; queue_front v1's E1 definition is reused.
- Types:
  - `improve` (strictly better than the side's best in f−1);
  - `re_establish` (the cell was shown > 0 within the prior 5 s and emptied by matched fills);
  - `deep` (worse than the side's best);
  - `at_empty_touch`.
- **Indicators per row:** any creation; each type separately. Types overlap and are never extra seconds. Exact cell-event keys and counts are saved beside each indicator.

**Statuses** (all counted):
- missing or stale book at t (no usable frame, or age > 1,000 ms);
- a creation interval straddling an anchor boundary (it cannot be assigned to one second);
- unknown cells.

**Wording:** an observed market opportunity, not an identified maker decision time. Adding size at an occupied price is not a fresh creation; its absence is never "no posting".

## §2 State before the anchor (frozen comparison set; pre-anchor observations only)

Each feature is computed at t from usable frames with upd ≤ t, the BTC grid ≤ t and fills with hi ≤ t. No feature's price level is chosen from the outcome.

| Feature | Definition | Bins |
|---|---|---|
| `spread` | in ticks | 1 / 2 / 3–4 / ≥ 5 |
| `depth_touch_own` | shown shares at the side's best price | 0 / (0, 50] / (50, 200] / > 200 |
| `depth_touch_opp` | shown shares at the opposite side's best price | same bins |
| `depth_2_own` | shares at best ± 1 tick, away from the spread | same bins |
| `btc_5s_aligned` | BTC 5-s return in YES direction for that side (bid: up = with; ask: down = with), `chain_context.MOVE_THR` | with / against / flat |
| `fair_gap` | the existing BTC fair value p_fair (`pf_mt_decision.btc_feats` 'p_fair', exactly as chain_take uses it; no refit). bid: p_fair − best bid (cents); ask: best ask − p_fair. Missing p_fair is a status. | ≤ 0 / (0, 1] / (1, 3] / > 3 |
| `flow_5s` | matched taker shares hitting this side's best in the last 5 s (records with hi ≤ t) | 0 / > 0 |
| `time_left` | take_logic's bins | — |

**Comparisons:** MH person-time rate ratios of each outcome type, with these exposed / reference contrasts:
- spread ≥ 2 vs 1;
- fair_gap > 1 vs ≤ 0;
- btc with vs flat, and against vs flat;
- own touch depth 0–50 vs > 200;
- flow > 0 vs 0;
- time-left ≤ 30 s vs 60–240 s.

**Strata:** market × side × time-left bin.
- For `improve`, opportunity availability is mechanical (an improvement needs spread ≥ 2), so `improve` ratios use only rows with spread ≥ 2, and the spread contrast for `improve` is 3+ vs 2.

**Null:** 500 circular shifts of each market-side's creation-event schedule (seed 20261123); p_hi / p_lo = (1 + #) / 501.

**Market consistency:** the share of supported markets (≥ 30 exposed and ≥ 30 reference seconds, ≥ 1 event) with RR > 1. Day readings and every support count are saved.

**Verdicts** (take_logic's rule):
- `supported` = RR > 1, p_hi ≤ 0.05, and ≥ 2/3 of ≥ 5 supported markets; the mirror rule with RR < 1 and p_lo gives `supported_lower`;
- otherwise `not_supported` / `insufficient_support`;
- `timing_sensitive` if the verdict flips when the state is read at t − 1 s.

## §3 Maker identity layer (separate; posting_opportunity)

**Anonymous creations:** they stay in every denominator.

**Candidate creator** of a creation: the address of a candidate first fill at that cell (queue_front_v2 §5, earliest one-address set), and only where that maker's lifecycle posting window for the filled hash contains the creation interval.
- The grade is the order's post grade from `correction_v1/post_regrade_orders.csv` (G2 bound_constrained → `candidate_constrained`; otherwise `candidate_scenario`).
- Otherwise the status is `owner_unknown`.
- The eventual first filler is not automatically the creator. This is a candidate link with its grade.

**Top-3 conditional context:**
- **Statistics:** the share of `improve` / `re_establish` / `deep` among candidate-linked top-3 creations vs among all creations, with counts per grade.
- **Inventory:** the candidate's reconstructed inventory at the creation (sign relative to the side: a bid creation while short = reducing; while long = adding), only for `candidate_constrained` links. It is reported as a conditional estimate with counts, and never pooled with unknown owners.

**Readings:** shares and counts only; no verdict on maker intent.

## §4 Six complete chains (chains_v3)

**Cases:** the same 6 predetermined episodes as `correction_v1/paths.md` (per rank, the first two in case order). No new selection.

**Each chain:**
1. **Pre-state:** §2 features at the post window.
2. **Post:** interval, price and size bounds. A filled-size lower bound is not a submitted size.
3. **Queue:** compatible queue histories, with repeated same-hash fills listed as the same order.
4. **Ending:** pull / exhaustion / repost alternatives with grades.
5. **Takes:** take_logic_v2 direction (opposing / same) at the anchor.
6. **Value:** completed turnover and residual value (matched_gap / closes accounting), with the 10 s / 60 s / END views and their valid / all denominators.

Each chain marks where its histories agree, and exactly which hidden action changes the explanation.

## §5 Queue-front corrections (queue_front_v2; review findings 1–6)

1. **No certification by net decreases.**
   - `book_timed` becomes `book_candidate`: a conditional timing candidate under a no-netting premise.
   - Every earliest-set holder is a `front_candidate` with timing basis (`book_candidate` / `settlement`), never "certified".
   - Alternative histories under masked cancellation / replenishment are counted wherever a non-fill decrease or addition could hide inside the candidate interval: any frame-interval change in the cell not fully explained by matched records, plus every interval of the cell's life before the candidate. Those levels carry `alt_netting_possible`.
2. **Straddle fix.** A settlement-only record joins the creation interval only if its WHOLE hard window lies inside (upd(f−1), upd(f)]; otherwise it is `untimed`. This fixes the v1 lower-endpoint rule at `queue_front.py:629`.
3. **Same-hash split.** Records after the earliest set are labelled:
   - `later_same_hash` (fills of an order hash in the earliest set);
   - `later_other`.
   Economics compares earliest-set vs later_other, with later_same_hash beside, valid / all counts and valid shares. "Does the front pay?" is withdrawn as a queue-effect inference.
4. **Endpoint occupancy.** v1's joined categories become `cell_occupied_at_window_lo` / `cell_empty_at_window_lo`, with the order's post grade and the timestamp. They are never an actual-post queue label.
5. **Mechanism and speed.** Mechanism = the anonymous cell-creation context associated with a candidate first filler. Speed = the cell-event endpoint proxy, reported with its lower / upper interval columns.
6. **Front-win reading.** Kept `unidentified`, with each denominator. The interval is described as a bound within the declared reconstruction scenario.

**Controls** (selftest, each with a mutant caught):
- the reviewer's masked-cancellation / replenishment counterexample: two histories with identical public data must both remain, with `alt_netting_possible`;
- a creation-window straddle;
- same-hash partial fills;
- no-fill / no-creation rows;
- a later-frame perturbation that leaves earlier labels unchanged.

## §6 Take wording (finding 7; in the receipt)

- "seconds containing same-direction takes" (not trade counts, not volume);
- D1000 = the whole state lagged one second;
- 2,765 = extra raw records beyond the first per occupied second (904 seconds hold 3,669 records);
- the lower-tail ratios are descriptive under a fixed-path shift null.

**Amendments 1–2 of QUEUE-FRONT v1** are described as adaptive consumed-day development (the pilots also printed economics and readings).

## Acceptance (one combined review)

- Every aggregate reconciles to saved keyed rows.
- Supported associations, contradicted hypotheses and unidentified links are published separately.
- If maker ownership stays thin, the packet finishes with the anonymous creation results and the bounded chains. No tuned variant is added to manufacture identification.

**Cost:**

| Builder | Budget (CPU s) |
|---|---|
| posting_opportunity | ≤ 25 (≤ 2 selftests + one Oct 24 pilot) |
| queue_front_v2 | ≤ 15 |
| chains_v3 | ≤ 10 |

- Full runs projected ≤ 60 s in total; refuse above 120 s.
