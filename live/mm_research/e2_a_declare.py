"""P-2026-002 E2-A — the declaration, emitted BEFORE any tape is read.

WHAT E2-A IS FOR. E1-A passed the overlay arithmetic on the aggTrades tape:
`eff_RT_sweep = 6.2645 [5.7561, 6.7539] <= 8` at T_p = 600 s over 12
XS-overlap symbols. **That number is a BRACKET on a tape that cannot see a
book.** `EXPERIMENT_PLAN.md` says so in its own words -- what aggTrades cannot
establish is *"queue position inside the bracket; depth-dependent sizing
(episodes are min-size, notional-free); requote policies; partial fills; own
impact"* -- and defers all five to E2-A. Section 2 then fixes E2-A: *"re-run
the section 1.5 episode design with real books: actual touch placement,
queue-bracketed fills, depth-aware sizes at the XS book's actual rebalance
notionals. Gate: eff_RT <= 8 bps under RiskAverse. Supersedes the E1-A
number."* And the sketch's standing rule: **a sign-flip across the queue
bracket is a FAILURE, never averaged.**

E2-A also owes an answer on ONE named cell. E1-A's results audit called ICP
*"fragile"* -- 72% episode skips, stale-sweep 11.0, ADV rank exactly 40 on
stale data -- and recorded it as **unresolved, not passed**. E2-A resolves it
or says why it cannot.

This file writes all of that as DATA before the tape is opened. It imports no
reader and takes no data path; the runner will verify its digest and refuse on
a mismatch, so the gate cannot be redefined after seeing.

FOUR THINGS THIS DECLARATION FIXES THAT THE PREREG COULD NOT HAVE KNOWN:

  1. **`hftbacktest` IS NOT INSTALLED ON THIS BOX.** The plan names it for the
     queue bracket. Installing a dependency is an environment change and not
     mine to make, so the two queue models are declared HERE, in closed form,
     and implemented in the runner -- with the consequence stated plainly:
     **I own their correctness, so each ships a falsifier that fires on a
     known-bad queue state.** This is a DEVIATION and it is flagged for the
     reviewer to rule on before any run, not absorbed.
  2. **THE WINDOW IS NOT E1-A's.** E1-A ran on Vision aggTrades
     2026-07-18..08-17; the L2 collector started 08-19. They do not intersect.
     So E2-A's number is a statement about 2026-08-20..09-05 and **supersedes
     E1-A's as the operative number without re-measuring E1-A's window** --
     the same limitation E2.0 carried, stated the same way.
  3. **DAY ADMISSION NOW NEEDS A THIRD STREAM.** E2.0 required bookTicker and
     trade. E2-A reads `depth20` for depth-aware sizing, so a day is
     admissible only if all THREE are complete. The admissible set is declared
     as an OUTPUT.
  4. **"DEPTH-AWARE SIZES AT THE XS BOOK'S ACTUAL REBALANCE NOTIONALS"
     REQUIRES A NOTIONAL THIS PROGRAMME HAS NEVER PINNED.** E1-A's episodes
     were explicitly *"min-size, notional-free"*. A per-symbol rebalance
     notional is a property of the P-2026-001 capstone book, not of this tape.
     It is declared as a REQUIRED INPUT with an explicit refusal if it cannot
     be sourced -- not silently defaulted, because the whole point of
     depth-aware sizing is that the answer depends on it.

    python3 live/mm_research/e2_a_declare.py --selftest
    python3 live/mm_research/e2_a_declare.py --emit
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECL_VERSION = 4
SUPERSEDES = {
    "path": "live/mm_research/declarations/p002_e2_a_declaration_v3.json",
    "sha256": "6383d781c7bbeaa65ecccf4c5af469539c2fac3ba060d7d3fbc87d557e960ea7",
    "carrying_commit": "39f3eca",
    "chain": ["v1 405ddb7ab10486c2 (367b800)",
              "v2 6567a25f04d7fb89 (0cbaba6)",
              "v3 6383d781c7bbeaa6 (39f3eca)"],
    "correction_is_in_band": (
        "rule 13: v1 is NOT edited and stands as provenance. v2 adds the "
        "data-root discipline the E2.0 RESULT review (section 6) requires of "
        "the P-002 surface. No tape was touched under either version."),
    "what_changed": [
        "v3, REVIEWER CONDITION (4e73cdd): the two queue models were pinned "
        "only at the BOUNDARIES. Each now carries an INTERIOR control -- a "
        "case inside the queue-position range whose outcome is derivable by "
        "hand from the model's own published definition -- with a declared "
        "tolerance and a refusal if it misses. The reference implementations "
        "live in this declaring module, so the declaration pins the "
        "semantics EXECUTABLY rather than in prose.",
        "AND THE INTERIOR CONTROL IMMEDIATELY CAUGHT A DEFECT IN v2's OWN "
        "FORMULA. v2 declared ProbQueue-f3's fill probability as "
        "f(front)/(f(front)+f(back)) with front = queue ahead. That makes "
        "the fill probability RISE as the queue ahead GROWS, which is "
        "backwards: hand-computed at front=30/back=70 it gives 0.072973 when "
        "the order is nearly at the head, and 0.927027 when it is nearly at "
        "the back. v3 states it as f(back)/(f(front)+f(back)) and the "
        "interior control pins 0.927027 at front=30/back=70, with the "
        "DIRECTION itself as a second control. No run had happened, so "
        "nothing is retracted -- but a boundary-only battery would have "
        "carried this into the first smoke.",
        "carried from v2: the data_root_discipline block.",

        "v4, THE RUNNER'S BOUNDARIES, DECLARED BEFORE THE RUNNER READS A "
        "TAPE. v3 declared the models; it did not declare what the runner "
        "does at the edges where a model has no answer. Three were ruled by "
        "the coordinator (R-570(C)) and one was left open by the reviewer "
        "(REVIEW_DA60 section 2): the queue-ahead-undefined status, the "
        "partial-fill pricing pair with its straddle rule, day admission "
        "over the twelve IN SCOPE, and the at-or-through-L direction. Each "
        "is now a field with a control, because a boundary settled in the "
        "runner's code and nowhere else is a boundary a reader cannot "
        "check.",

        "v4, THE OWED ATTRIBUTION (REVIEW_P002_E2A_DESIGN section A.2). v3 "
        "asserted RiskAverse's rule as hftbacktest's. The library is ABSENT "
        "and cannot be consulted, so which half is the published model and "
        "which is my tightening is now stated as fields rather than left "
        "for a reader to assume. The same treatment is given to every other "
        "place where the closed form needed a choice the library would "
        "otherwise have made: the observable definition of `back`, the "
        "fill quantity on a successful draw, and the at-L direction.",

        "v4, AND THE ORDERING PROPERTY IS SPLIT, BECAUSE ONE HALF OF IT WAS "
        "NOT TRUE AS WRITTEN. R-570(B) owes a falsifier reading `ProbQueue "
        "<= RiskAverse per EPISODE`. On COST that is false as finance, not "
        "as arithmetic: an episode whose mid runs AWAY over T_p is cheaper "
        "chased than filled, so filling more can cost more and a per-episode "
        "cost inversion is the winner's curse rather than a defect. What IS "
        "arithmetic per episode is the FILLED QUANTITY. v4 declares both: "
        "quantity ordering per episode (a violation is an implementation "
        "defect and REFUTES), cost ordering at the AGGREGATE gate row (a "
        "violation raises REFUTES_THE_BRACKET). Flagged rather than "
        "silently narrowed.",
    ],
}


def f_power3(x: float) -> float:
    """PowerProbQueueFunc3: f(x) = x**3, the plan's n = 3."""
    return float(x) ** 3


def riskaverse_filled_qty(queue_ahead: float, order_qty: float,
                          volume_through: float) -> float:
    """RiskAverse, in closed form. Everything ahead trades first.

    Filled quantity = clip(volume_through - queue_ahead, 0, order_qty).
    Deterministic, so its interior is exactly derivable: at queue_ahead 1000,
    order 10 and volume 1005, five units are filled -- not zero (the boundary
    a 'did it fill at all' test would check) and not ten.
    """
    return float(min(max(volume_through - queue_ahead, 0.0), order_qty))


def probqueue_f3_fill_prob(queue_ahead: float, depth_behind: float) -> float:
    """ProbQueue with PowerProbQueueFunc3, n = 3.

    p = f(back) / (f(front) + f(back)), f(x) = x**3.

    THE ORIENTATION IS THE POINT AND v2 HAD IT BACKWARDS. A fill probability
    must RISE as the queue AHEAD shrinks; v2's f(front)/(f(front)+f(back))
    fell instead. Both directions are driven in the battery below.
    """
    fa, fb = f_power3(queue_ahead), f_power3(depth_behind)
    if fa + fb <= 0:
        return 0.0
    return fb / (fa + fb)


#: The interior cases, with values computed BY HAND from the definitions above
#: and written here as literals so the code must match the arithmetic, not the
#: other way round. 27000/(27000+343000) = 0.0729729729...;
#: 343000/370000 = 0.9270270270...
INTERIOR_CONTROLS = {
    "tolerance": 1e-9,
    "RiskAverse": {
        "case": "queue_ahead 1000, order_qty 10, volume_through 1005",
        "hand_derivation": "clip(1005 - 1000, 0, 10) = 5",
        "expected_filled_qty": 5.0,
        "why_interior": "not 0 (no volume) and not 10 (fully through) -- the "
                        "order is HALF filled, which only a model that counts "
                        "the queue can produce",
    },
    "ProbQueue_f3": {
        "case": "queue_ahead 30, depth_behind 70",
        "hand_derivation": "f(70)/(f(30)+f(70)) = 343000/370000",
        "expected_prob": 343000.0 / 370000.0,
        "why_interior": "not 0 and not 1 -- a strictly interior probability "
                        "whose value follows from the published f(x) = x**3 "
                        "and nothing else",
    },
}
OUT = HERE / "declarations" / f"p002_e2_a_declaration_v{DECL_VERSION}.json"
PROTOCOL = f"P002_E2_A_OVERLAY_QUEUE_BRACKET_DECLARATION_V{DECL_VERSION}"

# ---- constants, each carried from a named source ---------------------------
CAPSTONE_THRESHOLD_BPS = 8.0      # EXPERIMENT_PLAN section 1.5, E1-A gate
FEE_MAKER_VIP0 = 1.8              # EXPERIMENT_PLAN section 0
FEE_TAKER_VIP0 = 4.5              # e1_markout_scan.FEE_TAKER_VIP0
TP_PRIMARY_S = 600                # the ONLY gate row; no patience shopping
TP_GRID_S = (60, 600, 3600)
GAP_FRACTION_MAX = 0.05
MIN_COMPLETE_DAYS = 14
DAY_GRID_PER_DAY = 24             # decision times: every hour on the hour
DIRECTIONS = ("buy", "sell")
ICP_SKIP_RATE_AUDIT = 0.72        # E1_CODE_REVIEW: ICP's episode-skip rate
SKIP_RATE_UNRESOLVED_BAR = 0.50   # declared here, see the field below

#: E1-A's published numbers, T_p = 600 s, 12 XS-overlap symbols, 31 d Vision
#: aggTrades. The runner must reproduce these with ITS OWN code on E1-A's own
#: data before any real-book number is reported.
E1A_REPRODUCTION_TARGET = {
    "window": "2026-07-18..2026-08-17 (31 d Vision tick aggTrades)",
    "tp_s": TP_PRIMARY_S,
    "n_symbols": 12,
    "symbols": ["AAVEUSDT", "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT",
                "DOGEUSDT", "ETHUSDT", "FILUSDT", "ICPUSDT", "LTCUSDT",
                "SOLUSDT", "XRPUSDT"],
    "eff_rt_touch_bps": 3.4485,
    "eff_rt_sweep_bps": 6.2645,
    "ci_touch": [3.1095, 3.7927],
    "ci_sweep": [5.7561, 6.7539],
    "tolerance_bps": 0.05,
    "source": "data/mm_hf/e1/e1a_gate_summary.csv, row tp_s=600",
    "why_required": (
        "E2-A supersedes E1-A's number. A superseding number produced by a "
        "DIFFERENT implementation of the same episode design would be "
        "measuring the implementation, not the book. E2.0 established this "
        "discipline and reproduced E1-B's ADA numbers to 0.0002 bps; E2-A "
        "inherits it on the E1-A side."),
}

#: THE TWO QUEUE MODELS, in closed form, because hftbacktest is absent.
QUEUE_MODELS = {
    "RiskAverse": {
        "role": "PESSIMISTIC -- the binding model, and the one the gate reads",
        "rule": (
            "a resting order of size q at level L is filled only when the "
            "cumulative opposite-side volume traded AT OR THROUGH L since the "
            "order was placed exceeds the depth that was resting at L when it "
            "was placed, PLUS q. That is: everything ahead must trade first, "
            "and the queue ahead is taken as the FULL depth observed at L at "
            "placement time -- never a fraction of it."),
        "queue_ahead_at_placement": "the depth20 snapshot's size at L at the "
                                    "last snapshot at or before t0",
        "filled_quantity": "clip(volume_at_or_through_L_since_t0 - "
                           "queue_ahead, 0, q) -- a QUANTITY, not a boolean; "
                           "full fill is the case where that clip saturates "
                           "at q, i.e. volume > queue_ahead + q",
        "ATTRIBUTION_the_library_is_absent_so_this_is_stated_not_assumed": {
            "why_this_field_exists": (
                "REVIEW_P002_E2A_DESIGN section A.2: v3 asserted 'this is "
                "hftbacktest's RiskAverse'. hftbacktest is NOT installed, so "
                "that claim cannot be checked here and must not be made "
                "flatly. Which half is the published model and which is my "
                "tightening is stated instead."),
            "the_published_model_s_half": (
                "everything ahead of the order must trade before the order "
                "trades, and the queue ahead is taken as the FULL depth "
                "observed at the level at placement. That is the RiskAverse "
                "idea and it is not mine."),
            "MY_tightening": (
                "the '+q' term -- requiring the cumulative volume to exceed "
                "the queue ahead PLUS the order's own size for a FULL fill, "
                "rather than filling the whole order the moment the queue "
                "ahead is exhausted. Whether the library carries that term I "
                "cannot verify with the library absent."),
            "its_direction_is_unambiguous_even_though_its_provenance_is_not": (
                "including '+q' makes fills rarer and eff_RT HIGHER, so it "
                "is CONSERVATIVE for a PASS gate: a PASS under this form is "
                "a PASS under a library that omits the term. A FAIL might "
                "not be, and that asymmetry is why the tightening is "
                "acceptable in a gate that only binds on PASS."),
            "what_would_settle_it": (
                "installing hftbacktest and comparing on the same episodes. "
                "That is an environment change and is not the seat's to "
                "make; it is named here so the open question is visible "
                "rather than resolved by silence."),
        },
        "at_or_through_L_direction": {
            "why_this_field_exists": (
                "REVIEW_DA60 section 2, the one residual the reviewer could "
                "not close: 'whether an opposite-side trade exactly AT L "
                "counts is a directional choice that changes the fill count, "
                "and no control pins it'. The interior control uses "
                "`volume_through`, which presupposes the answer."),
            "the_direction": (
                "a trade AT L COUNTS. For a resting BUY at L an aggressive "
                "sell printing at exactly L executes against resting bids AT "
                "L, which is the queue this order is standing in; a print "
                "strictly below L means the book was swept past L and "
                "everything at L traded too. So the admissible set is "
                "k_price <= k_L for a buy and k_price >= k_L for a sell, "
                "with the comparison on INTEGER TICK INDICES."),
            "why_integer_tick_indices": (
                "E1's D-i defect exists because on floats the touch (<=) and "
                "sweep-through (<) rules are indistinguishable. The same "
                "hazard reappears at every level comparison inside the queue "
                "simulation, so every one of them is done on integers."),
            "the_control": (
                "a single opposite-side trade at EXACTLY L, with volume "
                "greater than the queue ahead, must FILL under RiskAverse; "
                "the strict-through reading (k_price < k_L) is the known-bad "
                "and must NOT fill. Both directions are driven in the "
                "runner's battery, which is what makes this a choice with a "
                "check rather than a choice in a comment."),
        },
        "why_this_is_the_gate": (
            "EXPERIMENT_PLAN section 2: 'Gate: eff_RT <= 8 bps under "
            "RiskAverse.' A pass under the pessimistic model passes "
            "everywhere; a pass only under the optimistic one is not a pass."),
    },
    "ProbQueue_f3": {
        "role": "OPTIMISTIC bracket end -- reported, never the gate",
        "rule": (
            "ProbQueueModel with PowerProbQueueFunc3, n = 3: at "
            "each opposite-side trade the resting order fills with "
            "probability f(back)/(f(front)+f(back)) where f(x) = x**3, "
            "front = queue ahead remaining, back = depth behind -- so the "
            "probability RISES as the queue ahead shrinks. v2 had this "
            "inverted and the interior control caught it before any run. "
            "Implemented here in closed form because the library is not "
            "installed; the n = 3 exponent is the plan's, not chosen here."),
        "determinism": (
            "the fill draw is seeded per (symbol, day, hour, direction) from "
            "sha256 of those fields plus the declaration digest, so the "
            "optimistic end is REPRODUCIBLE and the seed pins the data."),
        "front_is_observable_back_is_CONSTRUCTED": {
            "front": "queue ahead REMAINING = max(queue_ahead_at_placement - "
                     "cumulative volume at-or-through L since t0, 0). Purely "
                     "observable from the two tapes.",
            "back": (
                "max(depth at L in the latest depth20 snapshot at or before "
                "the trade - front, 0). This is MY construction, not the "
                "library's: hftbacktest tracks `back` from the order feed, "
                "which this programme does not have and which no public tape "
                "carries. The observable analogue is 'what is standing at the "
                "level now, less what is still ahead of me' -- so `back` "
                "grows when other orders join at L, which is exactly the "
                "quantity the model wants."),
            "the_degenerate_case_is_DECLARED_not_left_to_a_zero_divide": (
                "front = 0 and back = 0 makes f(front)+f(back) = 0 and the "
                "ratio undefined. Declared: front = 0 fills with probability "
                "1 -- the queue ahead is exhausted, so there is nothing left "
                "to wait for. That branch is taken FIRST, so the ratio is "
                "only ever evaluated with a positive denominator. It agrees "
                "with the ratio by continuity wherever the ratio is defined "
                "(front = 0, back > 0 gives f(back)/f(back) = 1)."),
            "fill_quantity_on_a_successful_draw": (
                "the WHOLE remaining order fills. MY choice, and its "
                "direction is the model's role: ProbQueue-f3 is the "
                "OPTIMISTIC end, so on the event that the order is reached "
                "the optimistic reading is that the reaching sweep takes all "
                "of it. Filling only min(q, trade volume) instead would let "
                "a partial ProbQueue fill sit BELOW a full RiskAverse fill "
                "on the same episode, i.e. it would break the bracket "
                "arithmetically rather than empirically."),
        },
        "the_ordering_property_is_SPLIT_and_one_half_of_R570B_was_not_true": {
            "per_episode_ARITHMETIC": (
                "filled_qty(ProbQueue) >= filled_qty(RiskAverse) on EVERY "
                "episode. This holds by the models' own logic and not by a "
                "clamp: RiskAverse fills only once front reaches 0, and at "
                "front = 0 ProbQueue's probability is 1, so ProbQueue fills "
                "at that trade at the latest and possibly earlier. A "
                "violation cannot be finance; it is an implementation defect "
                "and REFUTES."),
            "per_episode_COST_is_NOT_an_ordering_property": (
                "R-570(B) owes 'ProbQueue <= RiskAverse per EPISODE'. On "
                "COST that is FALSE, and not because of a defect: an episode "
                "whose mid runs AWAY from the maker over T_p is CHEAPER "
                "chased than filled (the chase captures the favourable "
                "drift), so filling more can cost more. A per-episode cost "
                "inversion is the winner's curse in the sign that favours "
                "the maker. Narrowing this silently would have made a real "
                "check into a spurious refusal, so it is flagged."),
            "aggregate_COST": (
                "eff_RT(ProbQueue) <= eff_RT(RiskAverse) at the gate row is "
                "the bracket ordering the gate reads, and a violation there "
                "raises REFUTES_THE_BRACKET as v3 declared. It is an "
                "aggregate property because the maker's edge is an average "
                "over episodes, not a per-episode identity."),
        },
    },
}


#: R-570(C): three boundaries the coordinator ruled after the design review,
#: plus the arm size the min-size arm needs. Declared as data, each with the
#: control that pins it, because a boundary settled only inside the runner is
#: a boundary a reader cannot check.
RUNNER_BOUNDARIES = {
    "queue_ahead_undefined": {
        "ruling": "R-570(C)(1)",
        "when": "the level L taken from bookTicker at t0- is ABSENT from the "
                "relevant side of the last depth20 snapshot at or before t0 "
                "(the twenty levels do not reach it), or there is no depth20 "
                "snapshot at or before t0 at all",
        "rule": "the episode carries the counted status "
                "QUEUE_AHEAD_UNDEFINED, is EXCLUDED from BOTH fill "
                "simulations, and its count is reported with every table "
                "(CLAUDE.md rule 4)",
        "never": "a queue-ahead of ZERO. An absent level is not an empty "
                 "level: zero would make the order fill on the first trade "
                 "and would read as the most favourable queue position there "
                 "is, which is the opposite of what absence licenses.",
        "the_control": "an episode whose L is outside the snapshot's twenty "
                       "levels must carry the status and appear in NEITHER "
                       "model's population; the mirror control is an episode "
                       "whose L IS in the book, which must be admitted -- a "
                       "status that only ever fires is a filter, not a status",
    },
    "partial_fill_pricing": {
        "ruling": "R-570(C)(2)",
        "gate_bearing": "RESIDUAL_CHASED_AT_TP -- an episode filled to "
                        "fraction phi costs phi*c_fill + (1-phi)*c_chase, "
                        "because the unfilled part is still a working order "
                        "and is chased at T_p exactly as a wholly unfilled "
                        "episode is. Reduces to E1-A's formula exactly when "
                        "phi is 0 or 1, which is why the two remain "
                        "comparable.",
        "pessimistic_bracket": "WHOLE_LEG_CHARGED -- an episode that is not "
                               "FULLY filled costs c_chase entire. Reported "
                               "BESIDE the gate-bearing number, never "
                               "instead of it.",
        "straddle_is_a_FAIL": "if the two pricings fall on OPPOSITE sides of "
                              "the 8.0 bps threshold the verdict is "
                              "FAIL_PARTIAL_FILL_PRICING_STRADDLES. Never "
                              "averaged -- the same rule the queue bracket "
                              "carries (R-567), applied to the second "
                              "bracket the partial fills open.",
        "the_control": "a fabricated episode at phi = 0.5 must price "
                       "strictly between c_fill and c_chase under the "
                       "gate-bearing rule and exactly at c_chase under the "
                       "pessimistic one; and a straddling pair must return "
                       "the straddle verdict rather than a mean",
    },
    "day_admission_scope": {
        "ruling": "R-570(C)(3)",
        "rule": "depth20 completeness is required for the TWELVE SYMBOLS IN "
                "SCOPE -- the population E2-A supersedes E1-A on -- and not "
                "for all sixteen the collector happens to carry. A day is "
                "admissible FOR A SYMBOL; there is no cross-symbol day "
                "filter.",
        "why": "the four extra collected symbols (APT, ARB, ATOM, GMX) are "
               "not in E1-A's XS-overlap set. Requiring their depth20 would "
               "let a stream outage on a symbol E2-A does not measure delete "
               "days from one it does, which is a population change with no "
               "measurement behind it.",
        "the_control": "a symbol outside the twelve must be REFUSED by name "
                       "rather than measured; and a day complete on all "
                       "three streams for a symbol in scope must be ADMITTED "
                       "while the same day missing depth20 alone is EXCLUDED",
    },
    "the_min_size_arm_needs_a_q_and_it_is_MEASURED_not_chosen": {
        "why_this_field_exists": "the size-aware arm REFUSES for want of a "
                                 "declared rebalance notional (R-567(C)(b)), "
                                 "so the arm that RUNS is min-size -- and a "
                                 "queue model needs an order SIZE, which "
                                 "E1-A never had because its fills were "
                                 "binary.",
        "q": "the venue's quantity STEP for the symbol, estimated from the "
             "tape by mode-of-diffs over the day's distinct trade "
             "quantities -- the same estimator, and the same function, E1-A "
             "already uses for the PRICE tick and which the reproduction "
             "control reproduces to 4 dp.",
        "why_not_a_round_number": "a size chosen in this seat would make the "
                                  "gate a function of that choice, which is "
                                  "rule 11's shape and is exactly what the "
                                  "notional escalation refused to do. The "
                                  "quantity step is a property of the venue "
                                  "read off the venue's own prints.",
        "what_it_does_NOT_claim": "the step is the smallest ORDERABLE "
                                  "increment, which is a lower bound on the "
                                  "exchange minimum order size, not "
                                  "necessarily equal to it. The arm is "
                                  "labelled MIN_SIZE_NOT_THE_E2A_GATE either "
                                  "way, so the distinction changes no "
                                  "verdict; it is stated so the number is "
                                  "not read as an exchange filter.",
        "the_control": "a synthetic tape whose quantities are all multiples "
                       "of 0.25 must return 0.25; a tape with a single "
                       "off-grid quantity must NOT be dragged to that "
                       "quantity by one print",
    },
    "the_chase_crosses_the_REAL_touch": {
        "rule": "an unfilled or partly filled episode is chased at T_p by "
                "crossing to the actual best ask (buy) or best bid (sell) "
                "from bookTicker at T_p, plus the taker fee.",
        "why_this_is_not_a_change_of_estimand": "E1-A priced the chase at "
                                                "m(T_p) + sign*ES_day/2 "
                                                "because it had no book. "
                                                "E2-A removes ES_day from "
                                                "PLACEMENT as a sanctioned "
                                                "same-day look-ahead; "
                                                "leaving it in the CHASE "
                                                "would keep the look-ahead "
                                                "in half the episodes. "
                                                "ES_day is gone from E2-A "
                                                "entirely.",
        "the_cost_is_still_E1A_s": "shortfall against the DECISION mid plus "
                                   "the taker fee -- realised drift over T_p "
                                   "and the spread actually crossed, so the "
                                   "winner's curse is charged in full.",
    },
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def carrying_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(HERE))
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


# --------------------------------------------------------------------------
# THE PREDICATES, AS FUNCTIONS SO THEY CAN BE DRIVEN
# --------------------------------------------------------------------------
def gate_predicate(eff_rt_riskaverse: float | None,
                   eff_rt_probqueue: float | None,
                   ci_lo: float | None = None,
                   ci_hi: float | None = None,
                   interval_claimable: bool = True) -> dict:
    """The E2-A gate. RiskAverse decides; a bracket that disagrees FAILS.

    Carries E2.0's reviewer findings forward without being told twice: the
    verdict is named by the bar it crosses, and the interval is
    decision-bearing on the PASS side (a pass is the claim that needs power;
    a fail does not).
    """
    T = CAPSTONE_THRESHOLD_BPS
    if eff_rt_riskaverse is None:
        return {"decidable": False, "state": None, "threshold_bps": T,
                "why": "no admissible RiskAverse estimate exists"}
    ra, pq = eff_rt_riskaverse, eff_rt_probqueue
    ra_passes = ra <= T
    pq_passes = (pq is not None) and (pq <= T)
    bracket_disagrees = (pq is not None) and (ra_passes != pq_passes)
    ordering_inverted = (pq is not None) and (pq > ra)
    if ordering_inverted:
        state = "REFUTES_THE_BRACKET"
    elif bracket_disagrees:
        state = "FAIL_BRACKET_DISAGREES"
    elif not ra_passes:
        state = "FAIL"
    elif not interval_claimable:
        state = "INCONCLUSIVE_NO_INTERVAL"
    elif ci_hi is None or ci_hi > T:
        state = "INCONCLUSIVE_INTERVAL_DOES_NOT_CLEAR"
    else:
        state = "PASS"
    return {
        "decidable": True, "state": state, "threshold_bps": T,
        "eff_rt_riskaverse_bps": ra, "eff_rt_probqueue_bps": pq,
        "riskaverse_point_passes": bool(ra_passes),
        "probqueue_point_passes": (None if pq is None else bool(pq_passes)),
        "bracket_disagrees": bool(bracket_disagrees),
        "ordering_inverted": bool(ordering_inverted),
        "ci_lo_bps": ci_lo, "ci_hi_bps": ci_hi,
        "interval_claimable": bool(interval_claimable),
        "why": {
            "PASS": f"eff_RT under the PESSIMISTIC model is at or below "
                    f"{T} bps AND its 95% CI upper bound is too. A pass is "
                    f"the claim that needs power, so the interval binds here "
                    f"and not on the fail side.",
            "FAIL": f"eff_RT under RiskAverse exceeds {T} bps. A fail needs "
                    f"no interval: 'did not clear' is not a claim that needs "
                    f"power.",
            "FAIL_BRACKET_DISAGREES":
                "the two queue models straddle the threshold. The sketch's "
                "standing rule is that a sign-flip across the bracket is a "
                "FAILURE and is never averaged -- the bracket exists to "
                "express that the tape cannot pin queue position, and a "
                "bracket that does not decide has not decided.",
            "REFUTES_THE_BRACKET":
                "ProbQueue-f3 returned a HIGHER cost than RiskAverse. The "
                "optimistic model cannot be worse than the pessimistic one, "
                "so this refutes the INSTRUMENT, not the symbol, and no "
                "verdict about the overlay may be read from the run.",
            "INCONCLUSIVE_NO_INTERVAL":
                "the point clears but G < 5 complete days, so no interval is "
                "claimable (CLAUDE.md rule 8) and PASS cannot be asserted",
            "INCONCLUSIVE_INTERVAL_DOES_NOT_CLEAR":
                f"the point clears {T} bps but the 95% CI upper bound does "
                f"not -- a point that clears with an interval that does not "
                f"is INCONCLUSIVE",
        }[state],
    }


def partial_pricing_predicate(eff_rt_residual_chased: float | None,
                              eff_rt_whole_leg_charged: float | None) -> dict:
    """R-570(C)(2): the SECOND bracket, the one partial fills open.

    The gate-bearing pricing chases the residual at T_p; the pessimistic one
    charges the whole leg. If those two fall on opposite sides of the
    threshold the run has not decided, and the rule is the one the queue
    bracket already carries: a straddle is a FAIL, never an average.
    """
    T = CAPSTONE_THRESHOLD_BPS
    a, b = eff_rt_residual_chased, eff_rt_whole_leg_charged
    if a is None or b is None:
        return {"decidable": False, "straddles": None,
                "why": "both pricings are required to read the second bracket"}
    straddles = (a <= T) != (b <= T)
    return {
        "decidable": True,
        "straddles": bool(straddles),
        "state": ("FAIL_PARTIAL_FILL_PRICING_STRADDLES" if straddles
                  else "PARTIAL_FILL_PRICING_AGREES"),
        "threshold_bps": T,
        "eff_rt_residual_chased_bps": a,
        "eff_rt_whole_leg_charged_bps": b,
        "gate_bearing": "residual_chased",
        "why": (f"the two partial-fill pricings fall on opposite sides of "
                f"{T} bps, so the verdict depends on a convention rather "
                f"than on the book -- a FAIL, never the mean of the two"
                if straddles else
                f"both partial-fill pricings fall on the same side of {T} "
                f"bps, so the verdict does not depend on which is used"),
    }


def icp_predicate(skip_rate: float | None, in_aggregate: bool) -> dict:
    """The named cell E1-A left unresolved.

    A symbol is not silently passed or failed on a population that mostly is
    not there: above the declared skip bar the cell is UNRESOLVED and is
    EXCLUDED from the aggregate, with both facts reported.
    """
    if skip_rate is None:
        return {"decidable": False, "state": None,
                "why": "no episode census for this symbol"}
    bar = SKIP_RATE_UNRESOLVED_BAR
    if skip_rate > bar:
        return {"decidable": True, "state": "UNRESOLVED_TOO_FEW_EPISODES",
                "skip_rate": skip_rate, "bar": bar,
                "excluded_from_aggregate": True,
                "in_aggregate_as_run": in_aggregate,
                "why": f"{skip_rate:.0%} of episodes skipped, above the "
                       f"declared {bar:.0%} bar. E1-A measured ICP at "
                       f"{ICP_SKIP_RATE_AUDIT:.0%} and called the cell "
                       f"fragile; a verdict on a population that is mostly "
                       f"absent is not a verdict."}
    return {"decidable": True, "state": "RESOLVED", "skip_rate": skip_rate,
            "bar": bar, "excluded_from_aggregate": False,
            "in_aggregate_as_run": in_aggregate,
            "why": "enough episodes survive for the cell to be read on the "
                   "same footing as every other symbol"}


# --------------------------------------------------------------------------
def declaration() -> dict:
    return {
        "protocol": PROTOCOL,
        "status": "DECLARATION_NO_DATA_TOUCHED",
        "supersedes": SUPERSEDES,
        "program": "P-2026-002-hf-market-making",
        "step": "E2-A -- overlay bracket resolution on real books under the "
                "queue-model bracket",
        "carrying_commit": carrying_commit(),
        "declared_by": "DA seat (pm-da), single-seat program",
        "sources": {
            "prereg_episode_design": "live/mm_research/EXPERIMENT_PLAN.md "
                                     "section 1.5, Gate E1-A",
            "prereg_e2a": "live/mm_research/EXPERIMENT_PLAN.md section 2, E2-A",
            "bracket_rule": "live/mm_research/STRATEGY_SKETCH.md -- "
                            "'sign-flip between RiskAverse and ProbQueue = "
                            "failure, never averaged'",
            "icp_cell": "live/mm_research/E1_CODE_REVIEW.md and E1_RESULTS.md "
                        "-- ICP fragile, 72% episode skips, UNRESOLVED",
            "e1a_numbers": "data/mm_hf/e1/e1a_gate_summary.csv",
            "sibling_step": "live/mm_research/declarations/"
                            "p002_e2_0_declaration_v2.json",
        },

        "population": {
            "symbols": (
                "the XS-overlap set E1-A used, NAMED and inherited rather "
                "than recomputed: AAVE, ADA, AVAX, BNB, BTC, DOGE, ETH, FIL, "
                "ICP, LTC, SOL, XRP. Recomputing the top-40 ADV universe "
                "as-of a new date would change the population and the number "
                "together, and then E2-A would not supersede E1-A -- it would "
                "measure something else."),
            "symbol_list": E1A_REPRODUCTION_TARGET["symbols"],
            "day_admission_predicate": (
                "a UTC day is ADMISSIBLE iff 24 hour-files exist for ALL "
                "THREE streams -- bookTicker, trade AND depth20 -- and the "
                "intra-day bookTicker gap fraction is < 0.05. depth20 is new "
                "here: E2.0 needed two streams, E2-A needs three for "
                "depth-aware sizing."),
            "the_admissible_set_is_an_OUTPUT": (
                "declared as an output so no expectation about which days "
                "qualify can become a filter. Structural file-count as-of "
                "2026-09-06T05:01Z: 16 complete days per stream per symbol "
                "(08-20..08-25, 08-27..09-05); the gap-fraction leg is NOT "
                "yet evaluated."),
            "day_admission_is_PER_SYMBOL_over_the_twelve_IN_SCOPE": (
                "R-570(C)(3): depth20 completeness is required for the "
                "TWELVE symbols in scope, not for all sixteen the collector "
                "carries. A day is admissible FOR A SYMBOL and there is no "
                "cross-symbol day filter -- otherwise a stream outage on "
                "APT, ARB, ATOM or GMX, none of which E2-A measures, would "
                "delete days from a symbol it does."),
            "min_complete_days": MIN_COMPLETE_DAYS,
            "refuses_below_min": True,
            "cluster_unit": "UTC day (CLAUDE.md rule 8)",
            "episode_unit": (
                f"symbol x admissible day x decision time on a fixed grid "
                f"({DAY_GRID_PER_DAY}/day, every hour on the hour) x "
                f"direction in {list(DIRECTIONS)} -- E1-A's grid exactly, "
                f"unchanged, so the two are comparable"),
            "window_is_not_E1A_s": (
                "E1-A ran 2026-07-18..08-17 on Vision aggTrades; the L2 "
                "collector started 08-19 and the windows DO NOT INTERSECT. "
                "E2-A's number supersedes E1-A's as the operative number "
                "WITHOUT re-measuring E1-A's window. A reader must not "
                "difference the two and call it a queue-model effect."),
        },

        "what_changes_from_E1A_and_what_does_not": {
            "UNCHANGED": [
                "the episode grid, the patience ladder, the primary "
                f"T_p = {TP_PRIMARY_S} s and the rule that the gate reads "
                "ONLY that row (no patience shopping)",
                "the shortfall accounting against the decision mid, and the "
                "chase branch at the taker fee",
                f"the {CAPSTONE_THRESHOLD_BPS} bps capstone threshold",
                "equal weighting across symbols, day-clustered mean, block "
                "bootstrap interval",
            ],
            "CHANGED_BECAUSE_A_REAL_BOOK_EXISTS": [
                "PLACEMENT: the touch is the ACTUAL best bid/ask from "
                "bookTicker at t0-, not `m0 - sign*ES_day/2` from a "
                "same-day median flip-bounce. E1-A's own review logged that "
                "proxy as a sanctioned same-day look-ahead; it is gone.",
                "FILLS: the touch/sweep-through bracket is replaced by the "
                "QUEUE bracket (RiskAverse / ProbQueue-f3) against real "
                "depth. E1-A's bracket expressed 'we cannot see the queue'; "
                "this one expresses 'we can see the depth but not our place "
                "in it', which is a strictly narrower uncertainty.",
                "SIZE: episodes carry the XS book's rebalance notional "
                "instead of being min-size and notional-free.",
                "PARTIAL FILLS: admitted and reported, since depth makes "
                "them expressible; E1-A could not.",
            ],
        },

        "the_required_input_that_does_not_exist_yet": {
            "what": "a per-symbol rebalance NOTIONAL for the XS book",
            "why_it_matters": (
                "'depth-aware sizes at the XS book's actual rebalance "
                "notionals' is the whole of the size change. E1-A's episodes "
                "were explicitly min-size and notional-free, so this "
                "programme has never pinned the number."),
            "where_it_must_come_from": (
                "the P-2026-001 capstone book's own construction, not from "
                "this tape and not from a round number chosen here"),
            "if_it_cannot_be_sourced": (
                "the runner REFUSES the size-aware arm and reports the "
                "min-size arm ONLY, labelled as NOT the E2-A gate -- because "
                "a min-size answer is exactly the E1-A answer with a better "
                "fill model, and calling it E2-A would quietly drop the "
                "third of the three things section 2 asks for. It is a "
                "refusal with a named status, never a default."),
            "escalated": True,
        },

        "the_queue_bracket": QUEUE_MODELS,
        "interior_controls": INTERIOR_CONTROLS,
        "runner_boundaries": RUNNER_BOUNDARIES,
        "hftbacktest_is_absent": {
            "checked": "import hftbacktest raises ModuleNotFoundError on this "
                       "interpreter, as-of 2026-09-06T05:01Z",
            "consequence": (
                "the plan names hftbacktest for the bracket. Installing a "
                "dependency is an environment change and is not the seat's "
                "to make, so both models are declared in closed form above "
                "and implemented in the runner."),
            "what_that_costs": (
                "the models' CORRECTNESS becomes mine rather than a "
                "library's. Each therefore ships a falsifier that fires on a "
                "known-bad queue state, and the ordering property "
                "(optimistic <= pessimistic) is a COMPUTED predicate whose "
                "violation refutes the instrument rather than the symbol."),
            "ruling_requested": (
                "this is a DEVIATION from a pre-registered plan and is "
                "flagged for the reviewer and the coordinator BEFORE any run "
                "-- either the direct implementation is accepted, or the "
                "dependency is authorised, or E2-A waits. It is not absorbed."),
        },

        "the_gate_quantity": {
            "primary": f"eff_RT under RiskAverse at T_p = {TP_PRIMARY_S} s, "
                       f"day-clustered mean, bps of the decision mid",
            "formula": "eff_leg = fill_rate*E[cost|fill] + "
                       "(1-fill_rate)*E[cost|chase]; eff_RT = 2*eff_leg "
                       "(entry and exit legs symmetric by convention) -- "
                       "EXPERIMENT_PLAN section 1.5, unchanged",
            "fees": {"maker_bps": FEE_MAKER_VIP0, "taker_bps": FEE_TAKER_VIP0,
                     "tier": "VIP0 + BNB"},
            "threshold_bps": CAPSTONE_THRESHOLD_BPS,
            "also_reported": [
                f"the full T_p ladder {list(TP_GRID_S)} (reported, never "
                f"gated -- no patience shopping)",
                "ProbQueue-f3 as the optimistic bracket end",
                "per-symbol eff_RT, so no single name can carry the aggregate",
                "fill rate, partial-fill share and episode-skip rate per "
                "symbol",
                "the min-size arm beside the notional-aware arm, so the size "
                "effect is visible rather than assumed",
            ],
        },

        "what_settles_and_what_fails": {
            "PASS": f"eff_RT(RiskAverse) <= {CAPSTONE_THRESHOLD_BPS} bps AND "
                    f"its 95% CI upper bound <= {CAPSTONE_THRESHOLD_BPS}",
            "FAIL": f"eff_RT(RiskAverse) > {CAPSTONE_THRESHOLD_BPS} bps",
            "FAIL_BRACKET_DISAGREES": "the two queue models straddle the "
                                      "threshold -- never averaged",
            "REFUTES_THE_BRACKET": "ProbQueue-f3 > RiskAverse: the optimistic "
                                   "model cannot cost more than the "
                                   "pessimistic one, so the INSTRUMENT is "
                                   "refuted and no overlay verdict is read",
            "FAIL_PARTIAL_FILL_PRICING_STRADDLES":
                "the residual-chased and whole-leg-charged pricings fall on "
                "opposite sides of the threshold -- R-570(C)(2), never "
                "averaged",
            "the_order_the_states_are_checked_in": [
                "1. REFUTES_THE_BRACKET -- the instrument, before any verdict",
                "2. FAIL_PARTIAL_FILL_PRICING_STRADDLES -- the second "
                "bracket, before the gate reads either pricing",
                "3. the gate, on the GATE-BEARING pricing (residual chased "
                "at T_p) under RiskAverse",
            ],
            "INCONCLUSIVE_NO_INTERVAL": "point clears, G < 5, no interval",
            "INCONCLUSIVE_INTERVAL_DOES_NOT_CLEAR":
                "point clears, CI upper bound does not",
            "why_the_interval_binds_only_on_PASS": (
                "a PASS is the claim that carries weight -- it is what would "
                "let the overlay proceed -- and a claim that needs power gets "
                "an interval. A FAIL is 'did not clear', which needs none. "
                "This is E2.0's reviewer finding 3 applied before it had to "
                "be filed twice."),
            "what_this_supersedes": (
                "E1-A's eff_RT_sweep 6.2645 [5.7561, 6.7539] as the operative "
                "overlay number -- per EXPERIMENT_PLAN section 2, 'Supersedes "
                "the E1-A number'. E1-A stays as provenance and is not edited."),
        },

        "the_ICP_cell": {
            "why_it_is_named": (
                "E1-A's results audit recorded ICP as UNRESOLVED, not passed: "
                f"{ICP_SKIP_RATE_AUDIT:.0%} episode skips, stale-sweep 11.0, "
                "ADV rank exactly 40 on stale data, per-symbol eff_RT 7.53 -- "
                "the highest of the twelve and inside 0.5 bps of the "
                "threshold. E2-A owes it an answer."),
            "skip_rate_bar": SKIP_RATE_UNRESOLVED_BAR,
            "rule": (
                "a symbol whose episode-skip rate exceeds the bar is reported "
                "UNRESOLVED_TOO_FEW_EPISODES and EXCLUDED from the aggregate, "
                "with both the rate and the exclusion reported. A verdict on "
                "a population that is mostly absent is not a verdict."),
            "the_bar_is_declared_now_and_applies_to_every_symbol": (
                "0.50 is set before any E2-A episode census exists. It is not "
                "an ICP-shaped hole: it applies to all twelve, and if ICP "
                "clears it, ICP is read on the same footing as the rest."),
            "the_aggregate_is_reported_BOTH_WAYS": (
                "with and without any excluded symbol, as E1-A did when it "
                "reported 6.15 excluding ICP beside 6.26 including it -- so "
                "the exclusion's effect is visible and not just its result."),
        },

        "reproduction_control_inherited": E1A_REPRODUCTION_TARGET,

        "falsifiers": {
            "both_directions_required": True,
            "positive_controls": [
                "a synthetic book where the resting order is ALONE at the "
                "level fills on the first opposite-side trade under BOTH "
                "queue models -- with no queue ahead the two models must "
                "agree exactly",
                "a synthetic episode that fills at the touch with zero "
                "adverse drift must return cost = fee_maker - half_spread, "
                "the closed form, to floating tolerance",
                "an unfilled episode must return the chase cost with the "
                "TAKER fee and the realised drift over T_p, so the winner's "
                "curse is charged in full",
                "the E1-A reproduction control passing on E1-A's own data",
                "a day with all three streams complete and no gaps is "
                "ADMITTED",
                "an episode whose L IS inside the depth20 snapshot's twenty "
                "levels is ADMITTED to both models -- the undefined status "
                "must be able NOT to fire, or it is a filter wearing a "
                "status's name",
                "an episode filled to phi = 0.5 must price STRICTLY between "
                "c_fill and c_chase under the gate-bearing rule -- partial "
                "pricing must be able to produce an interior number",
                "a synthetic tape whose quantities are all multiples of 0.25 "
                "must return a quantity step of 0.25",
            ],
            "known_bads": [
                "a resting order behind depth larger than all subsequent "
                "volume must NOT fill under RiskAverse -- if it does, the "
                "queue is not being counted",
                "ProbQueue-f3 returning a HIGHER cost than RiskAverse on the "
                "same episodes must REFUTE the run, not be averaged away",
                "a day missing depth20 alone must be EXCLUDED even though "
                "bookTicker and trade are complete -- the third stream is a "
                "real requirement, not decoration",
                "a symbol above the skip bar must be UNRESOLVED and excluded "
                "from the aggregate, and the aggregate reported both ways",
                "an absent rebalance notional must REFUSE the size-aware arm "
                "rather than defaulting to min-size and calling it E2-A",
                "the runner must REFUSE if this declaration's sha256 differs",
                "the runner must REFUSE a result-bearing emission off the "
                "canonical ledger root (de_data_root.require_canonical, "
                "adopted from P-003 and imported, not copied)",
                "the runner must REFUSE if the E1-A reproduction control "
                f"misses the published numbers by more than "
                f"{E1A_REPRODUCTION_TARGET['tolerance_bps']} bps"
                "an episode whose level L is ABSENT from the depth20 "
                "snapshot must carry QUEUE_AHEAD_UNDEFINED and appear in "
                "NEITHER model's population -- a queue-ahead of zero there "
                "would read as the most favourable queue position there is",
                "an opposite-side trade at EXACTLY L must FILL under "
                "RiskAverse when it carries enough volume; the "
                "strict-through reading is the known-bad and must not",
                "the two partial-fill pricings straddling the threshold must "
                "return FAIL_PARTIAL_FILL_PRICING_STRADDLES, never their "
                "mean",
                "a symbol outside the twelve in scope must be REFUSED by "
                "name rather than measured",
                "filled_qty(ProbQueue) < filled_qty(RiskAverse) on ANY "
                "single episode must REFUTE -- that ordering is arithmetic, "
                "unlike the per-episode COST ordering, which the winner's "
                "curse can invert with no defect present",
            ],
        },

        "resources": {
            "cap": "one CPU, MemoryMax=8G, under the rule-20 wrapper "
                   "(flock + systemd-run). NEVER RAISED: a symbol that "
                   "exceeds the cap REFUSES.",
            "measured_input_sizes_as_of_2026-09-06T05:01Z": {
                "depth20 per day (gz)": {"ADAUSDT": "30 MB",
                                         "ICPUSDT": "14 MB",
                                         "BTCUSDT": "40 MB"},
                "note": "depth20 is 100 ms snapshots and is far more even "
                        "across symbols than bookTicker, where BTC is 480 MB "
                        "a day against ADA's 41. The heavy stream for E2-A is "
                        "still bookTicker.",
                "E2_0_measured_cost": "67 s wall / 657 MiB RSS for one symbol "
                                      "x 16 days on two streams",
            },
            "smoke": "ONE symbol first with its resource observation, before "
                     "any fan-out -- and it will be ICP, because ICP is the "
                     "cell E2-A owes an answer on and the one most likely to "
                     "refuse.",
            "data_root_discipline": {
                "why_this_field_exists": (
                    "the E2.0 RESULT review, section 6: `e2_0_true_mid.py:46` "
                    "was `ROOT = HERE.parents[1]` and there was NO data-root "
                    "resolver anywhere on the P-002 surface. On a partial "
                    "root the existing check fires only for a WHOLLY missing "
                    "directory, so a shell would have produced a silently "
                    "SMALLER population rather than a refusal."),
                "the_rule": (
                    "every result-bearing P-002 run resolves through the "
                    "SAME imported resolver P-003 uses -- "
                    "de_data_root.require_canonical, which itself delegates "
                    "to pm_tape_density._resolve_data_root. IMPORTED, NOT "
                    "COPIED: a second implementation of 'where is the "
                    "ledger' is exactly the thing that drifts apart."),
                "recorded_in_every_receipt": [
                    "repo_root", "data_root", "data_root_resolved", "branch",
                    "PM_DATA_ROOT_env", "tape_present", "is_canonical",
                    "refusal"],
                "refusal": (
                    "a result-bearing emission whose resolved data root is "
                    "not the canonical ledger REFUSES. Not a warning: a "
                    "worktree shell resolves, reads and holds a DIFFERENT "
                    "ledger."),
                "the_falsifier_is_the_reviewer_s_exact_case": (
                    "a PARTIAL root -- a real mm_hf tape holding 2 days of "
                    "19, every directory present, every file readable -- "
                    "must REFUSE before any day is read, and must NOT be "
                    "reported as a 2-day census. Driven through the runner's "
                    "own entry point, not through the resolver alone: the "
                    "child prints the census it WOULD have produced (2) and "
                    "then the refusal."),
                "already_in_force": (
                    "landed for E2.0 before this declaration was written, so "
                    "E2-A inherits it rather than promising it"),
            },
            "estimate_is_labelled_an_estimate": (
                "adding a third stream and a queue simulation to E2.0's "
                "measured 67 s is not linear and no wall-clock figure is "
                "offered before the smoke measures one."),
        },

        "what_would_refute_this_DESIGN_rather_than_a_symbol": [
            "the E1-A reproduction control failing -- then E2-A is not "
            "superseding E1-A, it is measuring a different estimator",
            "ProbQueue-f3 costing more than RiskAverse -- the bracket's "
            "ordering is arithmetic, so a violation is an implementation "
            "defect and no verdict may be read",
            "depth20 snapshots being too sparse at the touch to establish a "
            "queue-ahead at placement -- then 'depth-aware' is a name and not "
            "a property, and the honest report is that the bracket could not "
            "be narrowed",
            "a systematically empty admissible-day set once all three streams "
            "are required -- that refutes the population predicate",
            "the rebalance notional proving unsourceable -- then E2-A can "
            "resolve the queue question but NOT the size question, and must "
            "say which half it answered",
        ],
    }


def selftest() -> int:                                        # noqa: C901
    fails: list[str] = []

    def ok(c, m):
        print(("ok   " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    T = CAPSTONE_THRESHOLD_BPS
    # --- the gate, at every state, both directions ---
    ok(gate_predicate(6.0, 5.0, ci_lo=5.0, ci_hi=7.0)["state"] == "PASS",
       "GATE POSITIVE CONTROL: RiskAverse 6.0 with a CI upper bound of 7.0 "
       "PASSES -- the rule can admit, so it is not a fail wearing a gate's "
       "name")
    ok(gate_predicate(9.0, 7.0, ci_lo=8.0, ci_hi=10.0)["state"]
       == "FAIL_BRACKET_DISAGREES",
       "GATE KNOWN-BAD: RiskAverse 9.0 fails while ProbQueue 7.0 passes -- "
       "the bracket STRADDLES the threshold and that is a FAILURE, never an "
       "average of 8.0")
    ok(gate_predicate(9.0, 9.5, ci_lo=8.0, ci_hi=10.0)["state"]
       == "REFUTES_THE_BRACKET",
       "GATE KNOWN-BAD: the OPTIMISTIC model costing MORE than the "
       "pessimistic one refutes the INSTRUMENT, not the symbol -- checked "
       "before the pass/fail branches, so a broken bracket can never emit a "
       "verdict about the overlay")
    ok(gate_predicate(10.0, 9.0, ci_lo=9.0, ci_hi=11.0)["state"] == "FAIL",
       "GATE KNOWN-BAD: both models above the threshold is a plain FAIL")
    ok(gate_predicate(6.0, 5.0, ci_lo=5.0, ci_hi=9.0)["state"]
       == "INCONCLUSIVE_INTERVAL_DOES_NOT_CLEAR",
       "GATE KNOWN-BAD: a point of 6.0 whose CI reaches 9.0 is INCONCLUSIVE "
       "-- E2.0's reviewer finding 3 applied before it had to be filed twice")
    ok(gate_predicate(6.0, 5.0, ci_lo=5.0, ci_hi=7.0,
                      interval_claimable=False)["state"]
       == "INCONCLUSIVE_NO_INTERVAL",
       "GATE: below G = 5 no interval is claimable, so PASS cannot be "
       "asserted from a point (CLAUDE.md rule 8)")
    ok(gate_predicate(T, 5.0, ci_lo=5.0, ci_hi=T)["state"] == "PASS",
       f"GATE BOUNDARY: exactly {T} bps passes -- the plan says '<= 8', and "
       f"the boundary is read the way it is written")
    ok(gate_predicate(None, None)["decidable"] is False,
       "GATE: no admissible RiskAverse estimate is UNDECIDABLE, never a pass "
       "and never a fail")
    ok(gate_predicate(6.0, None, ci_lo=5.0, ci_hi=7.0)["state"] == "PASS"
       and gate_predicate(6.0, None, ci_lo=5.0,
                          ci_hi=7.0)["probqueue_point_passes"] is None,
       "GATE: a missing ProbQueue end does not silently become agreement -- "
       "it is None, and the bracket checks are skipped rather than passed")

    # --- the ICP cell, both directions ---
    ok(icp_predicate(0.72, True)["state"] == "UNRESOLVED_TOO_FEW_EPISODES"
       and icp_predicate(0.72, True)["excluded_from_aggregate"] is True,
       f"ICP KNOWN-BAD: E1-A's own measured "
       f"{ICP_SKIP_RATE_AUDIT:.0%} skip rate is above the "
       f"{SKIP_RATE_UNRESOLVED_BAR:.0%} bar -> UNRESOLVED and excluded from "
       f"the aggregate")
    ok(icp_predicate(0.10, True)["state"] == "RESOLVED"
       and icp_predicate(0.10, True)["excluded_from_aggregate"] is False,
       "ICP POSITIVE CONTROL: a 10% skip rate is RESOLVED and stays in the "
       "aggregate -- the bar can admit, and it is not an ICP-shaped hole")
    ok(icp_predicate(SKIP_RATE_UNRESOLVED_BAR, True)["state"] == "RESOLVED",
       "ICP BOUNDARY: exactly at the bar is RESOLVED (the rule is '> bar')")
    ok(icp_predicate(None, True)["decidable"] is False,
       "ICP: no episode census is UNDECIDABLE, never a silent pass")

    # --- THE INTERIOR CONTROLS (reviewer condition 4e73cdd) ---
    tol = INTERIOR_CONTROLS["tolerance"]
    ra = INTERIOR_CONTROLS["RiskAverse"]
    got = riskaverse_filled_qty(1000.0, 10.0, 1005.0)
    ok(abs(got - ra["expected_filled_qty"]) <= tol,
       f"RISKAVERSE INTERIOR: queue_ahead 1000, order 10, volume 1005 fills "
       f"{got} -- the hand-derived clip(1005-1000, 0, 10) = 5. Strictly "
       f"between the two boundaries a fill/no-fill test would check")
    ok(riskaverse_filled_qty(1000.0, 10.0, 999.0) == 0.0
       and riskaverse_filled_qty(1000.0, 10.0, 2000.0) == 10.0,
       "RISKAVERSE BOUNDARIES still hold: below the queue nothing fills, far "
       "beyond it the whole order does -- the interior control is an "
       "ADDITION, not a replacement")
    ok(riskaverse_filled_qty(1000.0, 10.0, 1005.0)
       > riskaverse_filled_qty(1000.0, 10.0, 1002.0),
       "RISKAVERSE MONOTONE in volume: more volume through never fills less")

    pq = INTERIOR_CONTROLS["ProbQueue_f3"]
    gotp = probqueue_f3_fill_prob(30.0, 70.0)
    ok(abs(gotp - pq["expected_prob"]) <= tol,
       f"PROBQUEUE INTERIOR: front 30 / back 70 gives {gotp:.9f} -- the "
       f"hand-derived 343000/370000 = {pq['expected_prob']:.9f} from "
       f"f(x) = x**3 and nothing else. Strictly interior: not 0, not 1")
    ok(probqueue_f3_fill_prob(30.0, 70.0)
       > probqueue_f3_fill_prob(70.0, 30.0),
       "PROBQUEUE ORIENTATION -- THE DEFECT v2 CARRIED: the fill probability "
       "must RISE as the queue AHEAD shrinks. v2's f(front)/(f(front)+f(back)) "
       "fell instead, giving 0.073 near the head and 0.927 near the back. A "
       "boundary-only battery would have carried that into the first smoke")
    ok(abs(probqueue_f3_fill_prob(0.0, 100.0) - 1.0) <= tol
       and abs(probqueue_f3_fill_prob(100.0, 0.0) - 0.0) <= tol,
       "PROBQUEUE BOUNDARIES still hold: nothing ahead fills with certainty, "
       "nothing behind never fills")
    ok(all(probqueue_f3_fill_prob(a, 100.0 - a)
           < probqueue_f3_fill_prob(a - 10.0, 110.0 - a)
           for a in (90.0, 70.0, 50.0, 30.0)),
       "PROBQUEUE MONOTONE across the whole interior, not just at one point")

    # --- the declaration cannot read data (AST, not substring) ---
    tree = ast.parse(Path(__file__).read_text())
    imported: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            if n.level == 0 and n.module:
                imported.add(n.module.split(".")[0])
    SAFE = {"__future__", "argparse", "ast", "hashlib", "json", "math",
            "subprocess", "pathlib"}
    ok(imported <= SAFE,
       f"NO READER IMPORTED: imports are {sorted(imported)}, a subset of the "
       f"safe set -- this module CANNOT open the tape whatever its prose says")
    io_calls = sorted({
        n.func.attr for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"open", "read_text", "read_bytes", "write_text",
                            "read_csv", "read_parquet", "glob", "iterdir"}
    } | {n.func.id for n in ast.walk(tree)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
         and n.func.id == "open"})
    ok(io_calls == ["read_bytes", "read_text", "write_text"],
       f"NO TAPE ACCESS: every file operation is enumerated from the AST and "
       f"is one of {io_calls}")

    d = declaration()
    states = set(d["what_settles_and_what_fails"]) - {
        "why_the_interval_binds_only_on_PASS", "what_this_supersedes",
        "the_order_the_states_are_checked_in"}
    produced = {partial_pricing_predicate(7.0, 9.0)["state"]} | {
        gate_predicate(*a, **k)["state"] for a, k in (
        ((6.0, 5.0), {"ci_lo": 5.0, "ci_hi": 7.0}),
        ((10.0, 9.0), {"ci_lo": 9.0, "ci_hi": 11.0}),
        ((9.0, 7.0), {"ci_lo": 8.0, "ci_hi": 10.0}),
        ((9.0, 9.5), {"ci_lo": 8.0, "ci_hi": 10.0}),
        ((6.0, 5.0), {"ci_lo": 5.0, "ci_hi": 9.0}),
        ((6.0, 5.0), {"ci_lo": 5.0, "ci_hi": 7.0,
                      "interval_claimable": False}))}
    ok(states == produced,
       f"THE EMITTED TABLE AND THE CODE CANNOT DISAGREE: the declared states "
       f"are exactly those the predicate produces ({sorted(states)})")
    ok(d["status"] == "DECLARATION_NO_DATA_TOUCHED"
       and d["hftbacktest_is_absent"]["ruling_requested"],
       "THE DEVIATION IS DECLARED, NOT ABSORBED: hftbacktest is absent, both "
       "models are written in closed form, and a ruling is requested BEFORE "
       "any run")
    ok(d["the_required_input_that_does_not_exist_yet"]["escalated"] is True,
       "THE MISSING INPUT IS ESCALATED: the XS rebalance notional does not "
       "exist in this programme, and its absence REFUSES the size-aware arm "
       "rather than defaulting to min-size and calling it E2-A")
    ok(len(d["falsifiers"]["known_bads"]) >= 6
       and len(d["falsifiers"]["positive_controls"]) >= 4,
       f"FALSIFIERS BOTH DIRECTIONS: "
       f"{len(d['falsifiers']['positive_controls'])} positive controls, "
       f"{len(d['falsifiers']['known_bads'])} known-bads")
    ok(d["population"]["window_is_not_E1A_s"]
       and d["what_would_refute_this_DESIGN_rather_than_a_symbol"],
       "THE LIMITS ARE IN THE DECLARATION: the window is not E1-A's, and "
       "five conditions are named that would refute the DESIGN rather than a "
       "symbol")
    ok(d["interior_controls"]["ProbQueue_f3"]["expected_prob"]
       == probqueue_f3_fill_prob(30.0, 70.0)
       and d["interior_controls"]["RiskAverse"]["expected_filled_qty"]
       == riskaverse_filled_qty(1000.0, 10.0, 1005.0),
       "THE DECLARED INTERIOR VALUES ARE THE ONES THE CODE PRODUCES -- the "
       "hand-derived literals and the implementations cannot drift apart")
    ok(d["reproduction_control_inherited"]["eff_rt_sweep_bps"] == 6.2645,
       "THE INHERITED CONTROL pins E1-A's published T_p=600 numbers, so a "
       "superseding number cannot come from a different estimator unnoticed")

    # ---- v4: the three ruled boundaries and the owed attribution --------
    ok(partial_pricing_predicate(7.0, 9.0)["state"]
       == "FAIL_PARTIAL_FILL_PRICING_STRADDLES",
       "PARTIAL-PRICING KNOWN-BAD: 7.0 residual-chased against 9.0 "
       "whole-leg-charged STRADDLES 8.0 and is a FAIL -- not the mean 8.0, "
       "which would have passed (R-570(C)(2))")
    ok(partial_pricing_predicate(6.0, 7.0)["state"]
       == "PARTIAL_FILL_PRICING_AGREES"
       and partial_pricing_predicate(9.0, 11.0)["state"]
       == "PARTIAL_FILL_PRICING_AGREES",
       "PARTIAL-PRICING POSITIVE CONTROL: two pricings on the SAME side of "
       "the threshold AGREE, on both sides of it -- the straddle rule can "
       "not fire, so it is a rule and not a veto")
    ok(partial_pricing_predicate(None, 7.0)["decidable"] is False,
       "PARTIAL-PRICING REFUSES a half-populated bracket rather than "
       "reading the half it has")

    rb = d["runner_boundaries"]
    ok(rb["queue_ahead_undefined"]["ruling"] == "R-570(C)(1)"
       and "never" in rb["queue_ahead_undefined"]
       and "ZERO" in rb["queue_ahead_undefined"]["never"],
       "BOUNDARY 1 DECLARED: an absent level is QUEUE_AHEAD_UNDEFINED and "
       "excluded from both models -- explicitly NOT a queue-ahead of zero, "
       "which would read as the best queue position there is")
    ok(rb["partial_fill_pricing"]["gate_bearing"].startswith(
           "RESIDUAL_CHASED_AT_TP")
       and "never" in rb["partial_fill_pricing"]["straddle_is_a_FAIL"].lower(),
       "BOUNDARY 2 DECLARED: the residual chased at T_p is gate-bearing, "
       "the whole-leg charge is the pessimistic bracket, and a straddle is a "
       "FAIL rather than an average")
    ok("TWELVE" in rb["day_admission_scope"]["rule"]
       and d["population"][
           "day_admission_is_PER_SYMBOL_over_the_twelve_IN_SCOPE"],
       "BOUNDARY 3 DECLARED: day admission is per symbol over the twelve in "
       "scope, so an outage on a symbol E2-A does not measure cannot delete "
       "days from one it does")

    att = QUEUE_MODELS["RiskAverse"][
        "ATTRIBUTION_the_library_is_absent_so_this_is_stated_not_assumed"]
    ok("+q" in att["MY_tightening"]
       and "HIGHER" in att[
           "its_direction_is_unambiguous_even_though_its_provenance_is_not"],
       "THE OWED ATTRIBUTION IS A FIELD, NOT A HOPE: the '+q' term is named "
       "as MY tightening, with its direction (fills rarer, eff_RT higher, "
       "conservative for a PASS gate) stated because its provenance cannot "
       "be checked with the library absent")
    ok(QUEUE_MODELS["RiskAverse"]["at_or_through_L_direction"][
           "the_direction"].count("k_price") == 2,
       "THE REVIEWER'S RESIDUAL IS CLOSED IN THE DECLARATION: the at-L "
       "direction is stated for both sides on INTEGER TICK INDICES, with a "
       "control that a trade exactly AT L fills and the strict-through "
       "reading is the known-bad")
    ordr = QUEUE_MODELS["ProbQueue_f3"][
        "the_ordering_property_is_SPLIT_and_one_half_of_R570B_was_not_true"]
    ok("FALSE" in ordr["per_episode_COST_is_NOT_an_ordering_property"]
       and "ARITHMETIC" in "".join(ordr),
       "THE ORDERING PROPERTY IS SPLIT AND SAYS SO: quantity per episode is "
       "arithmetic; COST per episode is not an ordering property at all, "
       "because favourable drift makes a chase cheaper than a fill")

    print(f"\n{'selftest OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(fails)} failure(s)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.emit:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(declaration(), indent=2, sort_keys=True)
                       + "\n")
        print(f"{OUT}  sha256 {sha256_file(OUT)}")
        return 0
    ap.error("choose --selftest or --emit")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
