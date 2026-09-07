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
import datetime
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: THE ONE CHAIN IMPLEMENTATION (R-711, REV 81 section 1.4). v8 is the first
#: version of this family written through it: the compare-and-swap that
#: refuses an in-place edit AND a head that moved under the caller lives
#: THERE, imported, never re-typed here. It opens nothing outside a
#: `declarations/` directory, so importing it leaves this module's
#: cannot-read-the-tape property intact -- and that property is CHECKED from
#: the AST in the selftest below, not asserted in this comment.
_PM_RESEARCH = HERE.parent / "pm_research"
if str(_PM_RESEARCH) not in sys.path:
    sys.path.insert(0, str(_PM_RESEARCH))
import declaration_chain as CHAIN                             # noqa: E402

DECL_VERSION = 8
SUPERSEDES = {
    "path": "live/mm_research/declarations/p002_e2_a_declaration_v7.json",
    "sha256": "57c92c9e899eb6912c659de3bb84f31994b8143bc79ba3f4fd7131d82d5bcfa0",
    "carrying_commit": "7a1e582dace24fec608481d5ea81bb3536ec0c83",
    "chain": ["v1 405ddb7ab10486c2 (367b800)",
              "v2 6567a25f04d7fb89 (0cbaba6)",
              "v3 6383d781c7bbeaa6 (39f3eca)",
              "v4 756ca9a31b89cd74 (0718fea)",
              "v5 90a9a99f6cb2a2cc (8e6b753)",
              "v6 127e0a56ddee775e (e29f984)",
              "v7 57c92c9e899eb691 (7a1e582)"],
    "why_v8_and_not_v7_amended": (
        "CLAUDE.md rule 13 and SEAT_PROTOCOL rule 20's immutability clause. "
        "v7 is committed AND CITED BY THE PAIR from a landed receipt -- "
        "data/mm_hf/e1/p002_e2a_sealed_smoke_BTCUSDT__20260906T071809Z.json "
        "carries `declaration: {path: ..._v7.json, sha256: 57c92c9e...}` -- "
        "so editing v7 would rewrite the declaration a landed run says it "
        "ran under. v8 supersedes in band through the shared "
        "compare-and-swap; v7 stays as provenance and that receipt keeps "
        "resolving. THE VERSION NUMBER WAS VERIFIED AT THE CHAIN, not taken "
        "from a dispatch: the DA 116 dispatch said 'v6 superseding v5' and "
        "the head was already v7."),
    "why_v7_and_not_v6_amended": (
        "R-580(C)(3). v6 is committed AND CITED BY A LANDED RECEIPT "
        "(p002_e2a_v6_admission__20260906T062536Z.json, sha "
        "11f12371e88c9d7d), so CLAUDE.md rule 13 forbids editing it. v7 "
        "supersedes in band; v6 stays as provenance and its receipt keeps "
        "resolving."),
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

        "v5, TWO CORRECTIONS THE RUNNER'S OWN FIXTURE FORCED, BOTH BEFORE "
        "ANY TAPE WAS OPENED. (i) v4 declared the min-size arm's q as 'the "
        "same estimator, and the same function, E1-A already uses for the "
        "price tick'. Driven, that is WRONG: tick_size/tick_mode keep a GCD "
        "fallback that fires when fewer than 99.9% of the diffs are integer "
        "multiples, and the GCD of a set containing ONE off-grid value "
        "collapses to the representation floor -- measured, a tape of 0.25 "
        "multiples plus a single 3.14159 returns 1e-5. q is therefore the "
        "MODAL positive diff with NO fallback, in the runner's own "
        "`qty_step_mode`, and the PRICE tick is left untouched because it is "
        "pinned by the E1-A reproduction control. (ii) a depth20 row with "
        "FEWER than twenty levels a side is NaN-PADDED by the CSV reader "
        "rather than rejected, and would have entered the simulation as a "
        "book with zero-size levels -- an invented queue position, the same "
        "defect class as reading an absent level as an empty one. Ragged "
        "rows are now detected, EXCLUDED and COUNTED in both directions.",

        "v5, AND THE SAME MECHANISM SHARPENS R-570(D)'s RECORD DEFECT. The "
        "modal-diff fix E1_RESULTS calls 'FIXED post-audit' IS present in "
        "the committed tick_size; what returns 1e-6 for FIL is the GCD "
        "FALLBACK the same docstring says was 'kept', firing on exactly the "
        "input the fix was written for. 'Designed and not landed' is not the "
        "whole account: it was landed, and it is overridden.",

        "v6, THE ADMISSION LEG IS RE-DECLARED AS AN OUTAGE DETECTOR "
        "(coordinator addendum to DA 62, on DA 61's census8 finding). v1-v5 "
        "gated a day on the intra-day bookTicker GAP FRACTION, inherited "
        "from E2.0 where it guarded against collector outage. MEASURED OVER "
        "EIGHT SYMBOLS, that leg selects on HOW OFTEN THE BEST QUOTE "
        "CHANGES: admissible days ran 16/16/16/14/14/13/11/1 against median "
        "decision-time quote ages of 68/102/116/169/194/222/261/517 ms -- "
        "monotone in activity -- and ICP, with 16 structurally complete days "
        "and ZERO of its missing seconds in runs of a minute or more, was "
        "cut to ONE. E2-A was dropping precisely the thin cell it exists to "
        "resolve. ADMISSIBILITY IS NOW A PROPERTY OF THE COLLECTOR BEING "
        "LIVE, never of how often a quiet book moves.",

        "v6, AND THE NEW LEG HAS NO THRESHOLD CHOSEN ON THIS DATA. The "
        "collector emits a heartbeat line carrying per-stream counters; its "
        "cadence is MEASURED as the modal inter-heartbeat interval (60 s) "
        "and the bar is 2x that. Measured separation on the real ledger: "
        "every clean day's largest heartbeat gap is 61 s -- one cadence -- "
        "while 2026-08-24 (the hf_ws_v2 era-boundary restart) is 158 s and "
        "2026-08-26 (the reboot) is 4,656 s. The bar sits between them by "
        "construction rather than by tuning, and the separation is a factor "
        "of 2.6 at its tightest.",

        "v6, AND WHAT MEASURING IT HAS CONSUMED, STATED RATHER THAN "
        "GLOSSED. The DECISION to replace the leg was informed by seeing "
        "which days v5 excluded, so eight symbols over 2026-08-20..09-05 "
        "are CONSUMED for any further re-choice of an admission predicate "
        "(rule 11). What is NOT tuned on them is the new leg itself: it "
        "reads a different quantity (the collector's heartbeat, not the "
        "book) and its bar comes from the collector's own cadence. A THIRD "
        "predicate proposed on this evidence would need days not yet "
        "examined.",

        "v6, AND THE PARTIAL-FILL BRACKET IS SHOWN TO FIRE. DA 61 reported "
        "that at partial_share = 0.000 the two R-570(C)(2) pricings "
        "COINCIDE and the straddle rule cannot fire -- rule 16's shape in "
        "my own work. The falsifiers now carry an episode built to be "
        "partial (queue 100, order 10, 104 units through -> filled 4 of "
        "10), on which the two pricings give eff_RT 6.0 and 10.0 and the "
        "straddle rule FIRES, their mean 8.0 being exactly the pass the "
        "rule exists to refuse. And the R-570(B) ordering falsifier is "
        "split in the battery as it already was in prose: QUANTITY per "
        "episode, COST at the aggregate gate row.",

        "v7, THE ORDERING PROPERTY AGAIN -- AND THE HALF v4 KEPT WAS ALSO "
        "FALSE. REVIEW_DA61_E2A section A.5 constructed the case v4's "
        "reasoning missed: RiskAverse fills clip(total - queue_ahead, 0, q), "
        "which is positive as soon as the CUMULATIVE volume passes the "
        "queue, while ProbQueue fills WITH CERTAINTY only once some trade "
        "arrives with front = 0. Between those two conditions is a MARGINAL "
        "REGIME where RiskAverse fills a sliver and ProbQueue is still a "
        "coin flip. Driven at queue_ahead 100, order 10, one trade of 105, "
        "depth 200: RiskAverse 5.0, ProbQueue 0.0 at max probability 0.5, "
        "993 of 2,000 seeds (49.6%) violating the per-episode quantity "
        "ordering. And the runner's response to a violation was to declare "
        "its own instrument REFUTED and read no gate -- so a CORRECT model "
        "disagreement would have suppressed the gate. v7 restates the "
        "property as E[filled_ProbQueue] >= filled_RiskAverse, makes "
        "per-episode testability the computable predicate `some trade has "
        "front = 0`, carries the marginal episodes as a COUNTED STATUS, and "
        "narrows REFUTES_THE_BRACKET to the regime where the ordering really "
        "is arithmetic.",

        "v7, AND MEASURING THE REVIEWER'S OWN PROPOSED RESTATEMENT SHOWS IT "
        "IS NOT SUFFICIENT ON ITS OWN -- THE front = 0 PREDICATE IS DOING "
        "ALL THE WORK. A.5 proposes the expectation as the fix. Constructed "
        "and driven: queue_ahead 100, order 10, ONE trade of 110, depth at L "
        "1000 -- RiskAverse fills 10 (the whole order), while front = 100 "
        "and back = 900 give p = 0.998630137 and hence E[filled_ProbQueue] = "
        "9.986301370 < 10. THE EXPECTATION ORDERING IS VIOLATED TOO, in the "
        "same marginal regime, with no defect anywhere. So the expectation "
        "is not a weaker-but-true version of the property; it is true "
        "exactly where the realisation ordering is true, which is the "
        "testable set. This is why v7 makes the predicate -- not the moment "
        "-- the thing that decides testability, and it is a strengthening of "
        "A.5 rather than an implementation of it.",

        "v7, THE LIVENESS LEG'S CONTROLS ARE THE RULED ONES (R-580(C)(1)). "
        "REVIEW_DA61_E2A section B.2 named 2026-08-29/30 as the positive "
        "control the leg must FLAG -- a common-mode event across all eight "
        "symbols. The collector's own ledger refutes that on every channel "
        "it has: zero restarts (all four are on 08-24 and 08-26), 1,439 of "
        "1,440 heartbeats each day with a maximum gap of 61 s (one cadence) "
        "against 158 s on 08-24 and 4,656 s on 08-26, zero WebSocket drops "
        "(08-28 has three, 08-31 has one), and -- decisively -- the "
        "throughput COMPOSITION: the event-driven streams fell about 40% "
        "while the FIXED-CADENCE depth20 stream held at 92-93%. An outage "
        "suppresses every stream; a quiet market suppresses only the "
        "event-driven ones. 08-29/30 is a quiet weekend. Requiring the leg "
        "to flag it would rebuild, through the CONTROL, exactly the "
        "activity-selecting defect v6 removed from the PREDICATE. RULED: "
        "the positive controls are the 08-24 and 08-26 collector events; "
        "08-29/30 is the NEGATIVE control and must ADMIT.",

        "v7, THE RULE-5 ERA PREDICATE BECOMES AN ADMISSION LEG OF ITS OWN "
        "(A.5's companion, REVIEW_DA61_E2A section B.3). E2-A's estimand is "
        "a queue simulation on sub-second arrival order. CLAUDE.md rule 5 "
        "fixes sub-second-reliable Binance data at recv_ns >= "
        "1787579334881534478 (2026-08-24 13:48:54 UTC, the hf_ws_v2 stamp "
        "boundary); before it, rows were stamped POST-PARSE and p99 carries "
        "up to ~0.6 s of parse-backlog error concentrated in bursts -- "
        "exactly when queue position matters. The reviewer measured 36 of "
        "101 admissible symbol-days as legacy-stamped with NO RECEIPT SAYING "
        "SO. v7 makes era purity a per-symbol-day admission leg, measured "
        "row-wise on recv_ns rather than assumed from the date, with the "
        "legacy share reported beside every table.",

        "v7, AND THE ERA LEG COLLAPSES THE POPULATION BELOW THE DECLARED "
        "MINIMUM -- STATED, NOT SOLVED BY MOVING THE BAR (R-580(C)(2)). "
        "Post-boundary complete days are 2026-08-25..09-05 = 12, of which "
        "08-26 is out on the collector reboot, leaving AT MOST 11 admissible "
        "post-boundary days per symbol against the declared minimum of 14. "
        "E2-A would refuse on the ENTIRE population. NO THRESHOLD IS CHANGED "
        "AFTER SEEING (CLAUDE.md rule 11): the mechanism runs on the "
        "post-boundary days as a SEALED SMOKE -- economics sealed and never "
        "read, resources and statuses published -- and the GATE IS READ ONLY "
        "AT G >= 14 post-boundary complete days, approximately 2026-09-09. "
        "Reversible by the USER; recorded as USER-visible.",

        "v7, DECISION-TIME QUOTE AGE IS A STATUS AT TWO LEVELS AND THE GATE "
        "IS READ BOTH WAYS (section B.4.2). Age is reported per symbol-day "
        "AND per episode, never gated -- filtering on it would select on "
        "activity, which is the defect v6 removed. The gate is read twice: "
        "over ALL resolved episodes, and over the subset whose decision-time "
        "quote is no older than a DECLARED staleness bar of 1,000 ms. The "
        "bar is NOT tuned on ICP: it is CLAUDE.md rule 5's own dividing line "
        "-- 'pre-boundary mm_hf tape is usable for >= 1 s bars only' -- the "
        "granularity at which this programme already says a stamp is "
        "trustworthy. A straddle of the 8 bps threshold between the two "
        "readings is a STATED SENSITIVITY, never averaged.",

        "v7, THE ICP CELL IS LABELLED RATHER THAN LEFT UNRESOLVED. E1-A left "
        "ICP unresolved at 72% episode skips; UNRESOLVED_TOO_FEW_EPISODES is "
        "not the honest end here because about 736 gate-row episodes exist "
        "over sixteen days. The honest end is a STATED one: run it, and "
        "label the cell NOT_COMPARABLE_ON_PLACEMENT_QUALITY with the "
        "mechanism written down -- on a quiet book the resting order sits at "
        "a STALE touch and later prints sweep through it, which inflates "
        "both the fill rate and the apparent capture. It is the "
        "E1-proxy-mid failure family in a new place. On ICP the bracket is "
        "already near-degenerate (fill rates 0.935 / 0.978), which is what a "
        "stale-touch placement looks like from the inside.",

        "v7, AND R-584 (USER RULING, MID-BATCH): SCOPE IS BTC FOR NOW. The "
        "sealed smoke runs on BTCUSDT; the gate, once 14 post-boundary days "
        "exist, is read on BTC; the twelve-symbol census stays as CONTEXT "
        "and the thin-name (ICP) cell is DEFERRED, NOT RESOLVED -- its "
        "declared handling stands and travels with the cell, but E2-A under "
        "this scope does not answer it. A reader must not take a BTC number "
        "as an answer for the thin names: BTC is the most active symbol in "
        "the set, so the placement-quality problem the ICP label names is at "
        "its WEAKEST there.",

        "v7, AND R-584's RESOURCE BOUNDARY, WHICH IS A MEASUREMENT DECISION "
        "AND NOT AN OPTIMISATION. BTC's bookTicker is 621 MB gzipped and "
        "about 60 M rows on a single day. The day read now STREAMS hour-file "
        "by hour-file and keeps only the EPISODE GRID -- the 24 "
        "decision-time quotes and their T_p ends -- because holding the "
        "other 60 million rows buys nothing and is the entire memory "
        "problem. A day whose residency exceeds the declared cap REFUSES: "
        "the cap is never raised and the population is never made smaller to "
        "fit it, since both would turn a resource limit into a silent change "
        "of what was measured. The cap is 75% of the rule-20 wrapper's own "
        "8 GiB MemoryMax, because a cgroup kill is silent and leaves nothing "
        "written -- the guard must fire while there is still room to write "
        "the refusal.",

        "v7, THREE REPORTING CORRECTIONS THAT RIDE THE SAME EDIT (sections "
        "A.6, A.2, A.4.1). (i) partial_share = 0.000 is STRUCTURAL, not "
        "rare: in the min-size arm a partial needs the cumulative volume to "
        "land inside a ONE-QUANTITY-STEP window, so the two R-570(C)(2) "
        "pricings bracket NOTHING there; they become live only in the "
        "size-aware arm, which refuses for want of a declared notional. "
        "(ii) --no-repro turns the gating control off and the receipt went "
        "SILENT -- key absent, not false, the P-003 asymmetry class in "
        "P-002. It now writes {reproduced: null, gates_the_run: false, "
        "why_skipped}. (iii) the quantity step is UNDERDETERMINED below "
        "three distinct quantities and now carries the status "
        "QTY_STEP_UNDERDETERMINED rather than a modal diff computed from "
        "one or two values.",
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
FAMILY = "p002_e2_a_declaration"
OUT = HERE / "declarations" / f"{FAMILY}_v{DECL_VERSION}.json"
PROTOCOL = f"P002_E2_A_OVERLAY_QUEUE_BRACKET_DECLARATION_V{DECL_VERSION}"

# ---- constants, each carried from a named source ---------------------------
CAPSTONE_THRESHOLD_BPS = 8.0      # EXPERIMENT_PLAN section 1.5, E1-A gate
FEE_MAKER_VIP0 = 1.8              # EXPERIMENT_PLAN section 0
FEE_TAKER_VIP0 = 4.5              # e1_markout_scan.FEE_TAKER_VIP0
TP_PRIMARY_S = 600                # the ONLY gate row; no patience shopping
TP_GRID_S = (60, 600, 3600)
#: v8 RETIRES `GAP_FRACTION_MAX` FROM THIS MODULE. It was a dead constant
#: here -- nothing in this declaration read it after v6 withdrew the
#: fraction bar from admission -- and a loose 0.05 beside an admission leg
#: invites its re-use as one. The day-level gap fraction survives as a
#: REPORTED status in the runner (`GAP_REPORTING_REFERENCE`), gating
#: nothing. E2.0's own bar is untouched and still lives in e2_0_declare.
MIN_COMPLETE_DAYS = 14
DAY_GRID_PER_DAY = 24             # decision times: every hour on the hour
DIRECTIONS = ("buy", "sell")
ICP_SKIP_RATE_AUDIT = 0.72        # E1_CODE_REVIEW: ICP's episode-skip rate
SKIP_RATE_UNRESOLVED_BAR = 0.50   # declared here, see the field below

#: ---- v7 / R-584 (USER ruling 2026-09-06): SCOPE IS BTC FOR NOW ----------
#: The sealed smoke runs on BTCUSDT and the gate, once 14 post-boundary days
#: exist, is READ on BTC. The twelve-symbol census stays as CONTEXT. The
#: thin-name (ICP) cell is DEFERRED, not resolved -- E2-A does not answer it
#: under this scope and must not be read as having done so.
GATE_SYMBOL = "BTCUSDT"

#: R-584's resource boundary. BTC's bookTicker is the 8.6 GB input DA 61's
#: census queued by measured size: 621 MB gzipped and about 60 M rows on a
#: single day. The day read STREAMS per hour-file and the run REFUSES if a
#: day exceeds the cap -- the cap is never raised and the population is never
#: made smaller to fit it.
#: WHERE THE CAP COMES FROM, so it is not a number chosen to fit BTC: the
#: rule-20 wrapper's own MemoryMax is 8 GiB, and a cgroup kill is SILENT --
#: it leaves nothing written, so a run killed at the cap would look like a
#: run that never happened. The guard fires at 75% of the wrapper's cap,
#: while there is still room to write the refusal.
DAY_RSS_CAP_GIB = 6.0

#: ---- v7: rule 5's era boundary, as a NUMBER ------------------------------
#: CLAUDE.md rule 5, quoted: "Sub-second-reliable Binance data exists ONLY
#: from 2026-08-24 13:48:54 UTC (recv_ns >= 1787579334881534478, the hf_ws_v2
#: stamp boundary in the ledger)." E2-A's estimand IS sub-second arrival
#: order, so this binds here in a way it did not bind E2.0 (which reads the
#: exchange stamp T). Carried as an integer so the leg compares row stamps,
#: never dates.
ERA_BOUNDARY_RECV_NS = 1787579334881534478
ERA_BOUNDARY_UTC = "2026-08-24T13:48:54Z"

#: The declared staleness bar, and WHERE IT COMES FROM. Not chosen on ICP's
#: measured 517 ms median -- that would be a threshold picked after seeing.
#: It is rule 5's own dividing line: pre-boundary tape is "usable for >= 1 s
#: bars only", i.e. one second is the granularity at which this programme
#: already says a stamp can be trusted. The gate is read BOTH ways; the bar
#: partitions the second reading, it never excludes an episode.
STALENESS_BAR_MS = 1000.0

#: v7 / REVIEW_DA61_E2A A.4.1: a modal positive diff computed from one or two
#: distinct quantities is not an estimate of a step, it is an echo of the
#: sample. Below this the step is UNDERDETERMINED and the day says so.
MIN_DISTINCT_QUANTITIES = 3


# --------------------------------------------------------------------------
# v8 -- THE OUTAGE PREDICATE, THE WINDOW IT MAY BE APPLIED IN, AND THE DAYS
# IT MAY NEVER BE APPLIED TO.
#
# R-745(4), a COORDINATOR ruling on the USER's instruction, disclosed for
# overrule by one line. What the ruling says, in its own words: "a day is
# inadmissible only if any bookTicker gap run >= 60 s on the symbol",
# "PRE-DECLARED ... for days from 2026-09-06 onward and NEVER re-applied to
# the consumed window", "the depth20 and 24-hour-file legs stay as they
# are", and decision-time quote age "REPORTED per day beside every result,
# never an admission criterion".
# --------------------------------------------------------------------------
#: A gap RUN is a maximal block of contiguous whole seconds in which no valid
#: bookTicker quote arrives -- the same run-length `e2_a_runner.gap_profile`
#: counts, on the same clock (`read_book`'s T) and behind the same
#: valid-quote filter. The leg is therefore read with the very instrument
#: whose published reading motivated the ruling, not with a second one that
#: would have to be reconciled to it.
OUTAGE_RUN_S = 60

#: WHY A RUN AND NOT A FRACTION, stated where the number lives. A day-level
#: gap FRACTION cannot separate a collector outage from a book that did not
#: move, and measured over eight symbols it excluded on ACTIVITY: ICP, 16
#: structurally complete days with zero of its missing seconds in runs of a
#: minute or more, was cut to one. A run of a minute is missing DATA; a
#: scatter of one-second holes is present data about a still book.
WHY_A_RUN_NOT_A_FRACTION = (
    "an outage is a few LONG contiguous runs; quietness is many one-second "
    "holes. The fraction sums them and cannot tell them apart, so it selects "
    "on how often the best quote changes -- which is what a thin-name cell "
    "exists to measure, not a reason to exclude the day.")

#: THE CONSUMED WINDOW, NAMED FROM ITS SOURCES rather than from a date range
#: in prose (CLAUDE.md rule 11; SEAT_PROTOCOL "a cited artifact is
#: locatable"). Every day listed here was SEEN on the quantity the ruled
#: predicate reads, or was simulated under a previous version, BEFORE the
#: predicate was chosen. The lists are transcribed from the artifacts at the
#: digests below -- this module reads no data path, so a reader re-derives
#: them from the named files rather than trusting the transcription.
CONSUMED_SOURCES = (
    {
        "path": "data/mm_hf/e1/p002_e2a_census8__20260906T055936Z.json",
        "sha256": ("9fa3e218a929ab8afc407cbe24c78b617f3f65955"
                   "43e0128f01c9025dea50e69"),
        "protocol": "P002_E2_A_OVERLAY_RUNNER_V1_CENSUS",
        "declaration_it_cited": {
            "path": "live/mm_research/declarations/"
                    "p002_e2_a_declaration_v5.json",
            "sha256": ("90a9a99f6cb2a2cc4c8f45f211177695355859e15"
                       "7643e854a34ef77ccc7b593")},
        "what_it_consumed": (
            "THE GAP-RUN READING ITSELF. This census published `gap_profile` "
            "per symbol-day -- `n_runs_ge_60s`, `max_gap_run_s`, "
            "`share_of_missing_seconds_in_runs_ge_60s` -- and the ruled "
            "predicate was chosen having seen it. Rule 11: those symbol-days "
            "are spent, and the predicate is never re-applied to them, not "
            "even to ADMIT one."),
        "symbols": ("AAVEUSDT", "ADAUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT",
                    "FILUSDT", "ICPUSDT", "LTCUSDT"),
        "days": ("20260820", "20260821", "20260822", "20260823", "20260824",
                 "20260825", "20260827", "20260828", "20260829", "20260830",
                 "20260831", "20260901", "20260902", "20260903", "20260904",
                 "20260905"),
        "the_days_this_census_SAW_but_never_JUDGED": {
            "days": ("20260819", "20260826", "20260906"),
            "measured": (
                "each carries `gap_fraction: null` and `gap_profile: null` "
                "for all eight symbols -- the census computes a profile only "
                "where bookTicker has its 24 hour-files, and these had 12, 0 "
                "and 6-7. 2026-09-06 was an INCOMPLETE day at 05:59Z when "
                "the census ran, six hours after it started."),
            "why_it_matters": (
                "the forward window opens ON 2026-09-06, and it can only do "
                "so honestly if nothing about that day's gap structure had "
                "been read when the predicate was chosen. That is a "
                "measurement at the artifact, not a convenience: the day was "
                "seen as a FILE COUNT and never as a gap reading."),
        },
    },
    {
        "path": "data/mm_hf/e1/p002_e2a_census__20260906T055752Z.json",
        "sha256": ("cb57b45c50fd83b8818b00e63a1db657dbbfd86d6"
                   "0afe75f7ef338f44db443cf"),
        "note": (
            "the two-symbol run that preceded the eight-symbol census; its "
            "judged set is a SUBSET and adds nothing to the union. Carried "
            "because it exists and a consumed window named from one artifact "
            "while another consumed the same days would be a window named "
            "from a convenience."),
        "symbols": ("ADAUSDT", "ICPUSDT"),
        "days": ("20260820", "20260821", "20260822", "20260823", "20260824",
                 "20260825", "20260827", "20260828", "20260829", "20260830",
                 "20260831", "20260901", "20260902", "20260903", "20260904",
                 "20260905"),
    },
    {
        "path": ("data/mm_hf/e1/"
                 "p002_e2a_sealed_smoke_BTCUSDT__20260906T071809Z.json"),
        "sha256": ("ec50c88809bdaa3f54a0d0e3f6e275801a0c258172"
                   "fc646c5b8d2b7df5627bd0"),
        "superseded_in_band_by": {
            "path": ("data/mm_hf/e1/p002_e2a_sealed_smoke_BTCUSDT__"
                     "20260906T071809Z.v2.json"),
            "sha256": ("7fa7540fa97e31ad033bda669e3ddcae43899e2a5"
                       "054c523274171c607168a05"),
            "nothing_was_re_run": True},
        "what_it_consumed": (
            "THE EPISODES THEMSELVES. The sealed smoke SIMULATED these BTC "
            "days under v7. BTCUSDT is not in the census above, so without "
            "this source the gate symbol would have had an EMPTY consumed "
            "set -- which is how a window opens on days that were already "
            "run."),
        "symbols": ("BTCUSDT",),
        "days": ("20260825", "20260827", "20260828", "20260829", "20260830",
                 "20260831", "20260901", "20260902", "20260903", "20260904",
                 "20260905"),
    },
)


def consumed_symbol_days() -> dict:
    """{symbol: (day, ...)} -- the union over every consumed source.

    PER SYMBOL-DAY, never per date range: the census consumed eight symbols
    and the sealed smoke consumed BTC, and a day consumed for one symbol
    says nothing about another.
    """
    out: dict = {}
    for s in CONSUMED_SOURCES:
        for sym in s["symbols"]:
            out.setdefault(sym, set()).update(s["days"])
    return {k: tuple(sorted(v)) for k, v in out.items()}


def _day_plus(day: str, n: int) -> str:
    """`20260905` + 1 -> `20260906`. Calendar arithmetic, never string math."""
    d = datetime.date(int(day[:4]), int(day[4:6]), int(day[6:8]))
    return (d + datetime.timedelta(days=n)).strftime("%Y%m%d")


#: COMPUTED, NEVER TYPED. The window opens the day after the LATEST day any
#: source consumed. It lands on 2026-09-06 because the last consumed day is
#: 2026-09-05 -- and if a later day is ever consumed, this MOVES with it
#: instead of contradicting a literal that nobody re-derives. The ruling's
#: own date is thereby corroborated by the artifacts rather than restated.
LATEST_CONSUMED_DAY = max(d for s in CONSUMED_SOURCES for d in s["days"])
FORWARD_WINDOW_START_DAY = _day_plus(LATEST_CONSUMED_DAY, 1)

#: The earliest DATE on which the gate could be read, as a floor and never a
#: prediction: the window's 14th day is FORWARD_WINDOW_START_DAY + 13 and it
#: closes at the following midnight, so no read is possible before then. It
#: happens only if all 14 are ADMISSIBLE, which no artifact can say yet.
EARLIEST_READ_DAY = _day_plus(FORWARD_WINDOW_START_DAY, MIN_COMPLETE_DAYS)
EARLIEST_READ_DATE = (f"{EARLIEST_READ_DAY[:4]}-{EARLIEST_READ_DAY[4:6]}-"
                      f"{EARLIEST_READ_DAY[6:8]}")

#: Every state leg (d) can produce, enumerated so the emitted table and the
#: code cannot disagree -- the discipline `gate_predicate` already carries.
OUTAGE_LEG_STATES = (
    "ADMITTED_NO_OUTAGE_RUN",
    "REFUSED_OUTAGE_RUN",
    "REFUSED_CONSUMED_NEVER_REJUDGED",
    "REFUSED_BEFORE_THE_FORWARD_WINDOW",
    "REFUSED_OUTAGE_LEG_NOT_EVALUATED",
)


def outage_leg(symbol: str, day: str,
               max_gap_run_s: float | None = None) -> dict:
    """Leg (d) on ONE symbol-day. THE ORDER OF THE CHECKS IS THE RULE.

    1. CONSUMED first, so a spent symbol-day is refused BY NAME as consumed
       even when the measurement would admit it. A day that is refused for
       the right reason and admitted for the wrong one is the rule-11
       failure this leg exists to make impossible.
    2. Then the window: forward-only. Days before it are refused as OUTSIDE,
       which is a different fact from CONSUMED and is said differently -- a
       symbol nobody ever measured still has no admissible history here.
    3. Then absence: an unmeasured leg REFUSES with its own status. Absence
       must never read as a pass (rule 11 / protocol rule 11), and the era
       leg above refuses the same way for the same reason.
    4. Only then the predicate itself, at OUTAGE_RUN_S with `>=`.
    """
    consumed = day in consumed_symbol_days().get(symbol, ())
    base = {"leg": "d", "symbol": symbol, "day": day,
            "outage_run_s": OUTAGE_RUN_S,
            "max_gap_run_s": max_gap_run_s,
            "forward_window_start_day": FORWARD_WINDOW_START_DAY}
    if consumed:
        return dict(base, admissible=False,
                    state="REFUSED_CONSUMED_NEVER_REJUDGED",
                    why=(f"{symbol} {day} is in the CONSUMED window: it was "
                         f"seen on the gap quantity, or simulated, before "
                         f"this predicate was chosen (R-745(4), CLAUDE.md "
                         f"rule 11). It is not re-judged -- not even to "
                         f"admit it."))
    if day < FORWARD_WINDOW_START_DAY:
        return dict(base, admissible=False,
                    state="REFUSED_BEFORE_THE_FORWARD_WINDOW",
                    why=(f"{day} falls before {FORWARD_WINDOW_START_DAY}, the "
                         f"day after the latest consumed day "
                         f"({LATEST_CONSUMED_DAY}). This leg is declared "
                         f"FORWARD-ONLY and has no reading to give for "
                         f"earlier days."))
    if max_gap_run_s is None:
        return dict(base, admissible=False,
                    state="REFUSED_OUTAGE_LEG_NOT_EVALUATED",
                    why=(f"the longest bookTicker gap run was not measured "
                         f"for {symbol} {day}, so this leg cannot admit it. "
                         f"An unevaluated leg is not a passed one."))
    return dict(
        base, admissible=bool(float(max_gap_run_s) < OUTAGE_RUN_S),
        state=("REFUSED_OUTAGE_RUN" if float(max_gap_run_s) >= OUTAGE_RUN_S
               else "ADMITTED_NO_OUTAGE_RUN"),
        why=(f"longest bookTicker gap run {float(max_gap_run_s):g} s "
             f"{'>=' if float(max_gap_run_s) >= OUTAGE_RUN_S else '<'} "
             f"{OUTAGE_RUN_S} s"))


def expected_filled_probqueue(probs, order_qty: float) -> float:
    """E[filled quantity] under ProbQueue-f3 on one episode.

    The order fills the WHOLE remaining quantity on the first successful
    draw (the declared branch), so the only random event is whether ANY draw
    succeeds: E[filled] = q * (1 - prod_i (1 - p_i)). Exact, not sampled --
    an expectation estimated by re-running the same seeded draw would be the
    realisation wearing a different name.
    """
    surv = 1.0
    for pr in probs:
        surv *= (1.0 - float(pr))
    return float(order_qty) * (1.0 - surv)


def ordering_is_testable_per_episode(front) -> bool:
    """v7's testability predicate: does SOME trade arrive with front = 0?

    THE WHOLE PROPERTY TURNS ON THIS. RiskAverse fills
    clip(total - queue_ahead, 0, q), which is positive as soon as the
    CUMULATIVE volume passes the queue. ProbQueue fills with certainty only
    once a trade arrives with the queue already cleared -- front = 0, where
    f(front) = 0 makes p = 1. Where that happens, ProbQueue fills the whole
    order at that trade at the latest, so filled_PQ = q >= filled_RA and BOTH
    the realisation and the expectation orderings hold ARITHMETICALLY.
    Where it does not, neither holds, and a violation is the model
    disagreeing with itself in the regime it was never claimed for.
    """
    for f in front:
        if float(f) <= 0.0:
            return True
    return False


#: The three regimes, with every value hand-derivable from the two published
#: model definitions and RE-DERIVED in the battery. These are the controls
#: that decide what may refute the instrument, so they are declared as data.
ORDERING_REGIMES = {
    "TESTABLE_full_overshoot": {
        "case": "queue_ahead 50, order_qty 10, two trades of 60, depth 500",
        "front_reaches_zero": True,
        "hand_derivation": (
            "v_before = [0, 60]; front = [50, 0]. At the second trade "
            "front = 0 so p = 1 and ProbQueue fills all 10. RiskAverse fills "
            "clip(120 - 50, 0, 10) = 10."),
        "expected_filled_RiskAverse": 10.0,
        "expected_E_filled_ProbQueue": 10.0,
        "realisation_violations_expected": 0,
        "may_refute": True,
    },
    "MARGINAL_reviewer_A5": {
        "case": "queue_ahead 100, order_qty 10, one trade of 105, depth 200",
        "front_reaches_zero": False,
        "hand_derivation": (
            "v_before = [0]; front = 100, back = 100, so p = 100^3/(100^3 + "
            "100^3) = 0.5 exactly. RiskAverse fills clip(105 - 100, 0, 10) = "
            "5. ProbQueue fills 10 or 0 on a coin flip."),
        "expected_filled_RiskAverse": 5.0,
        "expected_E_filled_ProbQueue": 5.0,
        "realisation_violations_expected_share": 0.5,
        "measured_by_the_reviewer": "993 / 2000 seeds = 0.4965",
        "may_refute": False,
        "why_not": (
            "the models genuinely disagree here and neither is wrong. "
            "Refuting on this would let a correct disagreement suppress the "
            "gate."),
    },
    "MARGINAL_expectation_counterexample_DA63": {
        "case": "queue_ahead 100, order_qty 10, one trade of 110, depth 1000",
        "front_reaches_zero": False,
        "hand_derivation": (
            "v_before = [0]; front = 100, back = 900, so p = 900^3/(100^3 + "
            "900^3) = 729000000/730000000 = 0.998630136986... RiskAverse "
            "fills clip(110 - 100, 0, 10) = 10, the WHOLE order. "
            "E[filled_ProbQueue] = 10 * 0.998630136986 = 9.98630136986 < 10."),
        "expected_filled_RiskAverse": 10.0,
        "expected_E_filled_ProbQueue": 10.0 * (900.0 ** 3)
                                       / (100.0 ** 3 + 900.0 ** 3),
        "expectation_ordering_holds": False,
        "may_refute": False,
        "why_this_case_exists": (
            "REVIEW_DA61_E2A A.5 proposes the EXPECTATION as the corrected "
            "property. Driven, the expectation is violated too, in the same "
            "marginal regime, with no defect anywhere. So the expectation is "
            "not a weaker-but-always-true restatement: it is true exactly "
            "where the realisation ordering is true, i.e. on the testable "
            "set. The front = 0 predicate is doing all the work, and this "
            "control is what keeps that from being taken on trust."),
    },
}


def ordering_verdict(n_testable: int, n_violations_testable: int,
                     n_marginal: int, n_violations_marginal: int,
                     e_filled_probqueue: float | None = None,
                     filled_riskaverse: float | None = None) -> dict:
    """v7's narrowed trigger. ONLY a testable-regime violation may refute.

    A violation in the marginal regime is a counted status and the gate is
    still read; a violation in the testable regime is arithmetic and can only
    be an implementation defect, so it replaces the verdict.
    """
    refutes = n_violations_testable > 0
    exp_ok = None
    if e_filled_probqueue is not None and filled_riskaverse is not None:
        exp_ok = bool(e_filled_probqueue >= filled_riskaverse - 1e-9)
    return {
        "state": ("REFUTES_THE_BRACKET" if refutes
                  else "ORDERING_HOLDS_WHERE_TESTABLE"),
        "property": "E[filled_ProbQueue_f3] >= filled_RiskAverse",
        "testability_predicate": "some trade in the episode has front = 0",
        "n_episodes_testable": int(n_testable),
        "n_violations_in_the_testable_regime": int(n_violations_testable),
        "n_episodes_marginal_ORDERING_NOT_TESTABLE":  int(n_marginal),
        "n_violations_in_the_marginal_regime": int(n_violations_marginal),
        "marginal_violations_do_NOT_silence_the_gate": True,
        "population_expectation_ordering_holds": exp_ok,
        "population_expectation_is_REPORTED_not_refuting": (
            "the expectation ordering is violable in the marginal regime "
            "with no defect present (ORDERING_REGIMES, the DA-63 "
            "counterexample), so a population-level inversion is reported "
            "beside the marginal share and never raises REFUTES"),
        "why": ("a violation where some trade cleared the queue is "
                "arithmetic and can only be an implementation defect; a "
                "violation where none did is the two models disagreeing in "
                "the regime neither claims"),
    }


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
        "q": "the venue's quantity STEP for the symbol: the MODAL positive "
             "diff over the day's distinct trade quantities, with NO GCD "
             "fallback, in the runner's own `qty_step_mode`.",
        "why_it_is_NOT_E1A_s_tick_function_although_v4_said_it_was": (
            "v4 declared it as the same function E1-A uses for the price "
            "tick. Driven in the fixture, that is wrong: tick_size / "
            "tick_mode keep a GCD fallback that fires when fewer than 99.9% "
            "of diffs are integer multiples of the modal one, and the GCD of "
            "a set containing ONE off-grid value collapses to the "
            "representation floor -- measured, 0.25 multiples plus a single "
            "3.14159 returns 1e-5. The PRICE tick is NOT changed: it is "
            "pinned by the E1-A reproduction control and must not move. The "
            "quantity step is a new quantity and takes the robust form. Both "
            "directions are driven: the robust estimator returns 0.25 and "
            "the fallback version is shown returning 1e-5."),
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
    "ragged_depth20_rows": {
        "when": "a depth20 line does not carry exactly twenty levels a side",
        "rule": "the row is EXCLUDED and COUNTED, in both directions "
                "(over-long rows the CSV reader rejects, and SHORT rows it "
                "silently NaN-pads), with the counts reported per day",
        "never": "padded with zeros. A zero-size level is a queue position, "
                 "and inventing one is the same defect class as reading an "
                 "ABSENT level as an EMPTY one -- which boundary 1 refuses.",
        "the_control": "a two-row fixture, one complete and one truncated, "
                       "must yield exactly one snapshot and one counted "
                       "ragged row",
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


def emitter_identity() -> dict:
    """WHO EMITTED THIS DECLARATION, in a form a rebase cannot invalidate.

    DA 63 landed the v7 emitter as `cd212f9`, emitted v7 against it, and then
    another seat's landing REBASED that commit to `acd393a` -- byte-identical
    content, a commit id that no longer exists on any branch. A declaration
    carrying only a commit id is an address that can be rewritten out from
    under it, which is the same class as an address travelling without its
    object.

    So the DURABLE citation is the emitter's CONTENT DIGEST: the sha256 of
    this module's bytes, which no rebase can change. The commit id is kept
    beside it as a convenience and LABELLED best-effort.
    """
    src = Path(__file__).resolve()
    r = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    last = r.stdout.strip() if r.returncode == 0 else ""
    return {
        "emitter_path": "live/mm_research/e2_a_declare.py",
        "emitter_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "emitter_sha256_is_the_durable_citation": (
            "a rebase rewrites commit ids and leaves the bytes alone. The "
            "digest identifies the code that produced this declaration "
            "whatever happens to the history above it."),
        "emitter_commit_best_effort": last or None,
        "emitter_commit_caveat": (
            "the last commit touching the emitter AT EMIT TIME. A later "
            "rebase can rewrite it; the digest cannot be rewritten. Resolve "
            "the digest first and treat a missing commit id as a rewritten "
            "history, not as a missing declaration."),
        "tree_head_at_emit": carrying_commit(),
    }


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
        "emitter_identity": emitter_identity(),
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
            "the_gate_symbol_under_R584": GATE_SYMBOL,
            "the_twelve_are_CONTEXT_under_R584": (
                "the XS-overlap set remains the declared population of the "
                "EXPERIMENT; R-584 narrows what is READ to BTC. The census "
                "over the twelve stays as context and is not re-run."),
            "day_admission_predicate": (
                "v6: a UTC day is ADMISSIBLE FOR A SYMBOL iff (a) 24 "
                "hour-files exist for ALL THREE streams -- bookTicker, trade "
                "AND depth20 -- and (b) THE COLLECTOR WAS LIVE for the whole "
                "day: no gap between consecutive heartbeats exceeding 2x the "
                "collector's own MEASURED modal cadence, and no collector "
                "restart inside the day. Nothing about how often the book "
                "moves enters admission."),
            "what_is_REPORTED_and_NOT_gated": [
                "the intra-day bookTicker gap fraction -- the quantity v5 "
                "gated on",
                "decision-time quote age per symbol-day (p50/p90/max at the "
                "24 decision times), which is the staleness a thin name "
                "actually carries into placement and which a reader must see",
                "the gap run-length profile, which is what shows a quiet book "
                "and an outage apart",
            ],
            "why_the_v5_leg_was_WRONG_and_how_that_was_established": (
                "it was inherited from E2.0 as an OUTAGE guard and, measured "
                "over eight symbols, turned out to select on ACTIVITY: "
                "admissible days 16/16/16/14/14/13/11/1 against median "
                "decision-time ages 68..517 ms, monotone. ICP's missing "
                "seconds are 2,853-11,997 one-second holes a day with "
                "0.0000 of them in runs of 60 s or more -- a quiet book, not "
                "missing data -- and the leg cut it from 16 days to 1. The "
                "control that made this readable is ADA at 16 of 16 under "
                "the same code and the same bar, reproducing E2.0's own "
                "admissible set."),
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
                "a day with all three streams complete and a LIVE "
                "collector is ADMITTED -- including a book so quiet that 99% "
                "of its seconds carry no message, which is the whole point of "
                "v6 and the case v5 excluded",
                "the partial-fill bracket must be able to produce two "
                "DIFFERENT numbers: an episode filled 4 of 10 prices at "
                "eff_RT 6.0 residual-chased against 10.0 whole-leg-charged",
                "an episode whose L IS inside the depth20 snapshot's twenty "
                "levels is ADMITTED to both models -- the undefined status "
                "must be able NOT to fire, or it is a filter wearing a "
                "status's name",
                "an episode filled to phi = 0.5 must price STRICTLY between "
                "c_fill and c_chase under the gate-bearing rule -- partial "
                "pricing must be able to produce an interior number",
                "a synthetic tape whose quantities are all multiples of 0.25 "
                "must return a quantity step of 0.25",
                "v8 leg (d) MUST BE ABLE NOT TO FIRE: a day inside the "
                "forward window whose missing seconds are scattered "
                "one-second holes, with no run of a minute, is ADMITTED -- "
                "otherwise the leg is the activity filter v6 removed wearing "
                "a new name",
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
                "a day whose COLLECTOR heartbeat shows a gap beyond twice its "
                "measured cadence, or a restart inside the day, must be "
                "EXCLUDED even with 24 hour-files on all three streams",
                "a partial episode's two pricings straddling the threshold "
                "must return FAIL_PARTIAL_FILL_PRICING_STRADDLES rather than "
                "their mean -- driven on an episode that is ACTUALLY partial, "
                "because at partial_share = 0.000 the rule cannot fire at all",
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
                f"v8 leg (d): a day inside the forward window carrying one "
                f"{OUTAGE_RUN_S} s bookTicker gap run must be REFUSED, and "
                f"the boundary is `>=` -- a longest run of "
                f"{OUTAGE_RUN_S - 1} s must ADMIT and one of "
                f"{OUTAGE_RUN_S} s must not",
                "v8 leg (d): a CONSUMED symbol-day must be refused BY NAME "
                "as consumed even when its measurement would admit it -- a "
                "day refused for the right reason and admitted for the wrong "
                "one is the rule-11 failure this leg exists to prevent",
                "v8 leg (d): a symbol-day whose longest gap run was NOT "
                "measured must be refused as NOT EVALUATED, never admitted "
                "quietly",
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

        #: ---- v7 / R-584 ---------------------------------------------
        "scope_R584_BTC_ONLY": {
            "ruling": "R-584, USER, 2026-09-06 -- 'SCOPE IS BTC FOR NOW'",
            "the_gate_symbol": GATE_SYMBOL,
            "what_it_means_here": [
                "the SEALED SMOKE runs on BTCUSDT, not on ADA or DOGE",
                "the E2-A GATE, once 14 post-boundary complete days exist, "
                "is read on BTC",
                "the twelve-symbol census (DA 61's census8 and the v6 "
                "admission over the twelve) stays as CONTEXT and is not "
                "re-run as a population",
                "the thin-name / ICP cell is DEFERRED, NOT RESOLVED",
            ],
            "the_ICP_cell_is_DEFERRED_not_answered": (
                "E1-A left ICP unresolved and v7 declares HOW it would be "
                "reported when it is taken up -- the label, its mechanism "
                "and both gate readings. Under R-584 that cell is not run "
                "and E2-A must not be read as having resolved it. A deferral "
                "stated is not a result withheld: the declared handling "
                "stands and travels with the cell."),
            "why_a_narrower_scope_does_not_weaken_the_gate": (
                "the gate was always a per-symbol read at T_p = 600 s over "
                "that symbol's admissible days -- not a cross-symbol "
                "average. Narrowing the SCOPE changes which cells are read, "
                "never what a cell means."),
            "what_a_reader_must_not_do": (
                "read a BTC number as an answer for the thin names. BTC is "
                "the most active symbol in the set and its decision-time "
                "quote age is the shortest; the placement-quality problem "
                "the ICP label names is at its WEAKEST there."),
        },

        "the_resource_boundary_R584": {
            "why_it_is_declared": (
                "BTC's bookTicker is 8.6 GB, 621 MB gzipped on a single day "
                "and about 60 M rows -- measured, and the reason DA 61's "
                "census queued BTC, ETH, SOL and XRP by input size. A "
                "whole-day read of it does not fit beside the trade and "
                "depth20 tapes under the rule-20 wrapper."),
            "the_rule": (
                "the bookTicker day is STREAMED hour-file by hour-file and "
                "only the EPISODE GRID is kept -- 24 decision-time quotes "
                "and their T_p-end quotes. Everything else in a 60 M row day "
                "is read, used to answer those, and released."),
            "the_cap": DAY_RSS_CAP_GIB,
            "the_cap_units": "GiB resident",
            "where_the_cap_comes_from": (
                "the rule-20 wrapper's own MemoryMax of 8 GiB. A cgroup kill "
                "is SILENT and leaves nothing written, so the guard must "
                "fire while there is still room to write the refusal: the "
                "bar is 75% of the wrapper's cap. It is NOT tuned on BTC."),
            "on_a_breach": (
                "the day REFUSES with its measured residency named. THE CAP "
                "IS NEVER RAISED AND THE POPULATION IS NEVER MADE SMALLER TO "
                "FIT IT -- both would turn a resource limit into a silent "
                "change of what was measured."),
            "published_either_way": (
                "per-day resident and peak memory at each stage, per-stage "
                "wall clock, the number of hour-files streamed and the peak "
                "residency of a single hour-file -- a resource observation, "
                "reported whether or not the guard fires."),
            "the_falsifier_runs_both_ways": (
                "a cap set below the measured residency must REFUSE the day "
                "with nothing emitted, and the real cap must ADMIT the same "
                "day -- otherwise the guard is a wall rather than a bound."),
        },

        #: ---- v7 ------------------------------------------------------
        "the_ordering_property_v7": {
            "supersedes": (
                "the v4 split. v4 kept 'filled_qty(ProbQueue) >= "
                "filled_qty(RiskAverse) on EVERY episode' as arithmetic. "
                "REVIEW_DA61_E2A A.5 constructed the case where it is not, "
                "and the runner's response to a violation was to declare "
                "itself refuted -- so a correct model disagreement would "
                "have suppressed the gate."),
            "property": "E[filled_ProbQueue_f3] >= filled_RiskAverse",
            "testability_predicate": (
                "PER EPISODE, testable iff SOME opposite-side trade arrives "
                "with front = 0, i.e. the cumulative volume BEFORE that "
                "trade already reaches queue_ahead. There p = 1 by f(0) = 0, "
                "ProbQueue fills the whole order at that trade at the "
                "latest, and both the realisation and the expectation "
                "orderings follow arithmetically."),
            "marginal_regime": (
                "no trade reaches front = 0: RiskAverse can fill a sliver "
                "off the CUMULATIVE volume while ProbQueue is still a draw. "
                "Carried as the counted status "
                "ORDERING_NOT_TESTABLE_MARGINAL, reported with every table "
                "(rule 4), and NEVER a refutation."),
            "refutation_trigger_NARROWED": (
                "REFUTES_THE_BRACKET is raised by a violation IN THE "
                "TESTABLE REGIME ONLY. A marginal-regime violation does not "
                "silence the gate."),
            "the_expectation_alone_is_NOT_the_fix": (
                "measured: at queue_ahead 100, order 10, one trade of 110 "
                "and depth 1000, RiskAverse fills the whole order while "
                "E[filled_ProbQueue] = 9.98630136986 < 10. The expectation "
                "ordering is violated in the marginal regime with no defect "
                "present, so it is true exactly on the testable set and the "
                "predicate -- not the moment -- is what decides."),
            "population_reading": (
                "the mean E[filled_ProbQueue] against the mean "
                "filled_RiskAverse over all resolved episodes is REPORTED "
                "beside the marginal share; an inversion there is a stated "
                "fact about the population, never a refutation."),
            "regimes_with_hand_derivations": ORDERING_REGIMES,
        },

        "admission_legs_v8": {
            "WHAT_v8_CHANGES": (
                "leg (d) is ADDED. Legs (a), (b) and (c) are carried "
                "UNCHANGED and the selftest asserts they are byte-identical "
                "to v7's, read from v7 on disk at the pair this version "
                "supersedes -- not re-typed and compared by eye."),
            "leg_d_outage_v8": {
                "ruling": (
                    "R-745(4), coordinator, on the USER's instruction, "
                    "DISCLOSED FOR OVERRULE BY ONE LINE."),
                "predicate": (
                    "a symbol-day is inadmissible on this leg iff the "
                    f"bookTicker tape carries a gap RUN of at least "
                    f"{OUTAGE_RUN_S} s -- a maximal block of contiguous "
                    "whole seconds with no valid bookTicker quote, counted "
                    "by `e2_a_runner.gap_profile` on `read_book`'s T behind "
                    "the same valid-quote filter. The comparison is `>=`."),
                "why_a_run_and_not_a_fraction": WHY_A_RUN_NOT_A_FRACTION,
                "states": list(OUTAGE_LEG_STATES),
                "the_order_of_the_checks_IS_the_rule": (
                    "consumed, then the window, then absence, then the "
                    "measurement. A consumed symbol-day is refused BY NAME "
                    "as consumed even when the measurement would admit it; "
                    "an unmeasured leg refuses with its own status, because "
                    "absence must never read as a pass."),
                "FORWARD_ONLY": (
                    f"applies to days from {FORWARD_WINDOW_START_DAY} "
                    f"onward and to no earlier day. The start is COMPUTED as "
                    f"the day after {LATEST_CONSUMED_DAY}, the latest day any "
                    f"consumed source names -- not typed from the ruling."),
                "what_it_does_NOT_change": (
                    "legs (a), (b) and (c). The 24-hour-file and depth20 "
                    "requirements stay exactly as they are (R-745(4)'s own "
                    "words), the collector-liveness leg keeps its ruled "
                    "controls, and rule 5's era leg is untouched."),
                "reference_implementation": (
                    "e2_a_declare.outage_leg(symbol, day, max_gap_run_s) -- "
                    "in this declaring module, so the semantics are pinned "
                    "EXECUTABLY and the runner's wiring is checked against "
                    "them rather than against prose."),
                "positive_control_RULED": (
                    "a day of scattered one-second holes and no run of a "
                    "minute MUST ADMIT -- the leg has to be able NOT to "
                    "fire, or it is the activity filter v6 removed wearing a "
                    "new name."),
                "known_bad_RULED": (
                    "a day carrying one 60 s run MUST REFUSE, and a "
                    "CONSUMED day MUST REFUSE AS CONSUMED even with a clean "
                    "measurement."),
            },
            "leg_a_streams": (
                "24 hour-files on ALL THREE streams -- bookTicker, trade, "
                "depth20. Unchanged since v4."),
            "leg_b_collector_liveness": {
                "predicate": (
                    "no gap between consecutive heartbeats exceeding 2x the "
                    "collector's own MEASURED modal cadence, and no "
                    "collector restart inside the day. A property of the "
                    "COLLECTOR, never of how often a quiet book moves."),
                "positive_controls_RULED": [
                    "2026-08-24 -- the hf_ws_v2 era-boundary deploy: max "
                    "heartbeat gap 158 s against a 120 s bar, 2 restarts. "
                    "MUST REFUSE.",
                    "2026-08-26 -- the reboot: max heartbeat gap 4,656 s, 2 "
                    "restarts. MUST REFUSE.",
                ],
                "negative_control_RULED": (
                    "2026-08-29 and 2026-08-30 MUST ADMIT: 1,439 of 1,440 "
                    "heartbeats, max gap 61 s (one cadence), zero restarts, "
                    "zero WebSocket drops. A quiet weekend with a live "
                    "collector."),
                "the_withdrawn_control": (
                    "REVIEW_DA61_E2A B.2 posed 08-29/30 as the POSITIVE "
                    "control -- a common-mode event the leg must flag. "
                    "R-580(C)(1) WITHDREW it on the collector's own ledger: "
                    "the event-driven streams fell about 40% while the "
                    "FIXED-CADENCE depth20 held at 92-93%. An outage "
                    "suppresses every stream; a quiet market suppresses only "
                    "the event-driven ones. Flagging it would have rebuilt "
                    "the activity-selecting defect v6 removed -- through the "
                    "control instead of the predicate."),
                "both_directions": (
                    "the leg is not accepted on a refusal alone: a real "
                    "outage must REFUSE and a quiet-but-live day must "
                    "ADMIT, and both are driven."),
            },
            "leg_c_rule5_era_purity": {
                "ruling": "R-580(C)(2); REVIEW_DA61_E2A B.3",
                "why_it_binds_here_and_not_in_E2_0": (
                    "E2.0 reads the exchange stamp T. E2-A's estimand IS "
                    "sub-second arrival order on recv_ns, so a legacy-"
                    "stamped row carries up to ~0.6 s of parse-backlog error "
                    "into the queue simulation, concentrated in bursts -- "
                    "exactly when queue position matters."),
                "predicate": (
                    "a symbol-day is era-admissible iff EVERY bookTicker row "
                    "in the day carries recv_ns >= "
                    f"{ERA_BOUNDARY_RECV_NS} ({ERA_BOUNDARY_UTC}). MEASURED "
                    "row-wise, never inferred from the date."),
                "legacy_share_reported_beside_every_table": True,
                "the_population_collapse_STATED": (
                    "post-boundary complete days are 2026-08-25..09-05 = 12, "
                    "of which 08-26 is out on the collector reboot: AT MOST "
                    "11 admissible post-boundary days per symbol against the "
                    "declared minimum of 14. E2-A refuses on the entire "
                    "population under the declared bar."),
                "NO_THRESHOLD_CHANGE_AFTER_SEEING": (
                    "CLAUDE.md rule 11. The minimum stays 14. What changes "
                    "is what may be READ, not what is required."),
                "the_sealed_smoke_regime": {
                    "what_runs": "the mechanism, on the post-boundary days",
                    "what_is_published": (
                        "resources, populations, every status count, the "
                        "legacy share, the ordering regimes, the quote-age "
                        "distributions -- and the sha256 of the sealed "
                        "payload"),
                    "what_is_SEALED": (
                        "every economic quantity: eff_RT under either model "
                        "and either pricing, the per-episode costs, the fill "
                        "rates, the bootstrap intervals and the overlay "
                        "verdict. Written once, digested, and NOT READ."),
                    "when_the_gate_may_be_read": (
                        "at G >= 14 admissible post-boundary complete days, "
                        "approximately 2026-09-09"),
                    "reversible_by": "the USER; recorded as USER-visible",
                },
            },
        },

        "the_forward_window_v8": {
            "start_day": FORWARD_WINDOW_START_DAY,
            "start_is_COMPUTED_not_typed": (
                f"the day after {LATEST_CONSUMED_DAY}, the latest day any "
                f"consumed source names. If a later day is ever consumed the "
                f"start MOVES with it; a literal date would have gone on "
                f"disagreeing with the artifacts silently."),
            "min_complete_days": MIN_COMPLETE_DAYS,
            "NO_THRESHOLD_CHANGE_AFTER_SEEING": (
                "CLAUDE.md rule 11. The minimum stays 14. What v8 changes is "
                "WHICH DAYS may be read, never how many are required."),
            "earliest_read_date": EARLIEST_READ_DATE,
            "earliest_read_date_is_a_FLOOR": (
                f"the window's 14th day is {FORWARD_WINDOW_START_DAY} + 13 "
                f"and it closes at the following midnight, so no read is "
                f"possible before {EARLIEST_READ_DATE}. It happens then only "
                f"if all 14 are ADMISSIBLE, which no artifact can say yet -- "
                f"this is a floor on the date, never a prediction."),
            "the_14_day_minimum_RESTARTS_here": (
                "the count begins at the window's start. No day before it "
                "counts toward the minimum, whatever its quality."),
        },

        "the_consumed_window_v8": {
            "rule": (
                "CLAUDE.md rule 11: days seen while the predicate was being "
                "chosen are SPENT. The predicate is never re-applied to "
                "them -- not even to admit one."),
            "sources": [dict(s) for s in CONSUMED_SOURCES],
            "symbol_days": {k: list(v)
                            for k, v in consumed_symbol_days().items()},
            "named_from_its_sources_not_from_a_date_range": (
                "R-745(4) quotes the window as '08-20..09-05'. The artifacts "
                "carry more than that sentence: the eight-symbol census "
                "ENUMERATES 2026-08-19..09-06 and JUDGES 16 of those days on "
                "the gap quantity, and the BTC sealed smoke -- a symbol the "
                "census does not cover at all -- simulated 11 days. This "
                "block carries the symbol-days themselves, so a reader "
                "resolves membership instead of parsing a range."),
            "the_two_kinds_of_consumption": (
                "SEEN ON THE QUANTITY (the censuses published each day's "
                "gap-run profile before the predicate was chosen) and RUN "
                "(the sealed smoke simulated the episodes). Both spend a "
                "symbol-day; they are listed per source rather than merged "
                "into one undifferentiated set."),
            "2026-09-06_IS_NOT_CONSUMED_and_this_is_MEASURED": (
                "the eight-symbol census enumerates 20260906 and carries "
                "`gap_fraction: null` and `gap_profile: null` for all eight "
                "symbols: it ran at 05:59Z with 6-7 of 24 hour-files "
                "present, so the day was seen as a FILE COUNT and never as a "
                "gap reading. The window can therefore open on it. THE "
                "STRICTEST ALTERNATIVE IS STATED, NOT HIDDEN: a reader who "
                "holds that enumerating a day spends it would start the "
                "window at 20260907 and read no earlier than 2026-09-21. "
                "That reading costs exactly one day and the USER or the "
                "coordinator may take it by one line; this declaration "
                "implements the ruling as written."),
        },

        "what_the_ruling_replaces_read_at_the_HEAD_v8": {
            "status": "DISCLOSED_NOT_ABSORBED",
            "what_R745_4_says": (
                "'the inherited bookTicker gap-fraction leg' is REPLACED by "
                "the outage predicate."),
            "what_the_head_ALREADY_said": (
                "v6 had already withdrawn the gap-fraction bar from "
                "admission (R-580(C)(1)) and replaced it with leg (b), "
                "COLLECTOR LIVENESS -- a property of the collector's own "
                "heartbeat ledger, not of the tape. At v7 there was no "
                "gap-fraction bar left to replace: the runner carries the "
                "fraction as `GAP_REPORTING_REFERENCE`, gating nothing, and "
                "`GAP_FRACTION_MAX` in this module was dead."),
            "what_v8_therefore_DOES": (
                "it ADDS the ruled predicate as leg (d) rather than removing "
                "a leg. The ruling's own words are a rule about when a day "
                "is inadmissible, and they are implemented exactly; what is "
                "NOT done is deleting leg (b), which carries three RULED "
                "controls (08-24 and 08-26 must refuse, 08-29/30 must "
                "admit). Adding a leg can only exclude more days, never "
                "admit one the head would have refused."),
            "what_a_reader_should_do_with_this": (
                "treat it as an overrulable reading. If the ruling intended "
                "leg (b) to be REPLACED by leg (d) -- both guard against "
                "outage, one from the collector's ledger and one from the "
                "tape -- that is one line from the coordinator or the USER, "
                "and it is a REMOVAL of a control, which is why this seat "
                "did not make it silently."),
            "the_quote_age_clause_was_already_in_force": (
                "R-745(4) also rules decision-time quote age a REPORTED "
                "field and never a criterion. v7's "
                "`decision_time_quote_age_v7` block already said exactly "
                "that, and the runner already reports p50/p90/max over the "
                "24 decision times per symbol-day. v8 changes nothing there "
                "and says so rather than re-announcing it as new."),
        },

        "decision_time_quote_age_v7": {
            "status_not_gate": (
                "reported per symbol-day (p50/p90/max over the 24 decision "
                "times) AND per episode (the age of the quote the order was "
                "placed from). Filtering on it would select on activity, "
                "which is the defect v6 removed from admission."),
            "declared_staleness_bar_ms": STALENESS_BAR_MS,
            "where_the_bar_COMES_FROM": (
                "CLAUDE.md rule 5's own dividing line -- pre-boundary tape "
                "is 'usable for >= 1 s bars only'. It is the granularity at "
                "which this programme already says a stamp is trustworthy. "
                "It is NOT ICP's measured 517 ms median, which would be a "
                "threshold chosen after seeing."),
            "the_gate_is_read_BOTH_WAYS": (
                "reading A over ALL resolved episodes; reading B over the "
                "subset whose decision-time quote age <= the bar. Both are "
                "published with their n."),
            "a_straddle_is_a_stated_sensitivity": (
                "if A and B fall on opposite sides of the 8.0 bps threshold "
                "the cell is reported STRADDLES_THE_STALENESS_BAR and the "
                "two readings stand side by side. Never averaged -- the same "
                "rule the queue bracket carries."),
        },

        "the_ICP_cell_label_v7": {
            "status_under_R584": "DEFERRED_NOT_RESOLVED",
            "what_deferred_means": (
                "R-584 puts the scope on BTC, so the ICP cell is NOT RUN "
                "under this declaration. What follows is the DECLARED "
                "HANDLING for when it is taken up -- written before the "
                "data, so the label cannot be chosen after seeing the "
                "number. E2-A must not be read as having resolved ICP."),
            "label": "NOT_COMPARABLE_ON_PLACEMENT_QUALITY",
            "not_this": (
                "UNRESOLVED_TOO_FEW_EPISODES is not the honest end: roughly "
                "736 gate-row episodes exist over sixteen days. The cell has "
                "a population; what it does not have is a comparable one."),
            "the_mechanism": (
                "on a quiet book the resting order sits at a STALE touch -- "
                "ICP's decision-time quote is 206-902 ms old at the median "
                "and up to 7.2 s at the maximum, 3-10x ADA -- and later "
                "prints sweep through a level the book has already left. "
                "That inflates BOTH the fill rate and the apparent capture. "
                "It is the E1 proxy-mid failure family in a new place: a "
                "price that stands in for the market rather than being it."),
            "the_symptom_already_measured": (
                "on ICP's one v5-admissible day the bracket is near "
                "degenerate -- fill rates 0.935 RiskAverse against 0.978 "
                "ProbQueue-f3 -- which is what stale-touch placement looks "
                "like from the inside: nearly everything fills under both "
                "models, so the queue bracket stops discriminating."),
            "what_is_still_reported": (
                "the cell RUNS and publishes its statuses, its quote-age "
                "distribution and both gate readings; the label travels with "
                "the number so no reader can compare it to ADA's without "
                "seeing why they are not comparable."),
        },

        "the_partial_fill_bracket_is_STRUCTURAL_in_the_min_size_arm": (
            "REVIEW_DA61_E2A A.6. partial_share = 0.000 is not rarity, it is "
            "structure: with q = one quantity step against a queue of "
            "hundreds, a partial requires the cumulative volume to land "
            "inside a ONE-STEP window, so phi is 0 or 1 almost surely and "
            "the two R-570(C)(2) pricings coincide. In the min-size arm the "
            "pricings bracket NOTHING. They become live only in the "
            "size-aware arm, which REFUSES for want of a declared rebalance "
            "notional -- so the machinery is controlled in the fixture (an "
            "episode built partial: queue 100, order 10, 104 through, filled "
            "4 of 10, eff_RT 6.0 against 10.0, straddle FIRES) and reported "
            "as inert on the arm that runs."),

        "reporting_corrections_v7": {
            "no_repro_receipt_field": {
                "finding": (
                    "REVIEW_DA61_E2A A.2: --no-repro turned the gating "
                    "control off and the receipt went SILENT -- the key was "
                    "absent, not false. The P-003 asymmetry class, in P-002."),
                "the_rule": (
                    "the field is ALWAYS present. With --no-repro it reads "
                    "{reproduced: null, gates_the_run: false, why_skipped: "
                    "<the reason>} -- null is not false: the control did not "
                    "run, so it neither passed nor failed."),
            },
            "qty_step_underdetermined": {
                "finding": "REVIEW_DA61_E2A A.4.1",
                "the_rule": (
                    f"below {MIN_DISTINCT_QUANTITIES} distinct trade "
                    "quantities in the day the modal positive diff is an "
                    "echo of the sample rather than an estimate of a step. "
                    "The day carries the status QTY_STEP_UNDERDETERMINED, "
                    "produces no episodes, and is counted -- never a step "
                    "guessed from one or two values."),
            },
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
    SAFE = {"__future__", "argparse", "ast", "datetime", "hashlib", "json",
            "math", "subprocess", "sys", "pathlib",
            #: DA 120: `tempfile`, imported INSIDE the battery to build the
            #: disarmed COPY the shared falsifier's known-bad runs against.
            #: It reaches no tape, and the cell below pins every write in
            #: this module to the battery.
            "tempfile",
            #: v8: the ONE chain implementation (R-711). It is stdlib-only
            #: and touches nothing outside a `declarations/` directory, so
            #: it cannot reach the tape either -- and the cell below proves
            #: the stronger property directly rather than by import list.
            "declaration_chain"}
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
       f"NO TAPE ACCESS: every file operation is enumerated from the AST "
       f"and is one of {io_calls}")

    #: AND THE PROPERTY THAT ACTUALLY MATTERS, STATED DIRECTLY (DA 120).
    #: "This module never writes" was v8's form of it, and it broke the
    #: moment the battery had to write a DISARMED COPY of the shared module
    #: into a temp dir for the falsifier cell below. The guarantee is not
    #: "no write" -- it is that THE EMIT goes through the compare-and-swap:
    #: every `write_text` here is inside the SELFTEST, and `main`, the only
    #: path that emits a declaration, holds none.
    _fn_ranges = {f.name: (f.lineno, getattr(f, "end_lineno", f.lineno))
                  for f in ast.walk(tree)
                  if isinstance(f, ast.FunctionDef)}
    _writes = [w.lineno for w in ast.walk(tree)
               if isinstance(w, ast.Call)
               and isinstance(w.func, ast.Attribute)
               and w.func.attr == "write_text"]
    _st_lo, _st_hi = _fn_ranges.get("selftest", (0, 0))
    _mn_lo, _mn_hi = _fn_ranges.get("main", (0, 0))
    ok(bool(_writes)
       and all(_st_lo <= ln <= _st_hi for ln in _writes)
       and not [ln for ln in _writes if _mn_lo <= ln <= _mn_hi],
       f"THE EMIT CANNOT BYPASS THE COMPARE-AND-SWAP: every `write_text` in "
       f"this module is inside the SELFTEST (line(s) {_writes}, selftest "
       f"{_st_lo}-{_st_hi}), and `main` ({_mn_lo}-{_mn_hi}) -- the only path "
       f"that emits a declaration -- holds none. The v8 emit goes through "
       f"declaration_chain.write_next_version or it does not happen")

    # --- v8: a data-path literal must never reach a call ------------------
    def _data_literal_reaching_a_call(t) -> list:
        """Every `data/...` string constant that is an argument of a call.

        The import list says what this module COULD reach; this says what it
        DOES with the one data path it names. The consumed-source citations
        are data paths ON PURPOSE -- a cited artifact is locatable -- and
        the property that matters is that none of them is ever opened.
        """
        #: OPENERS ONLY. An earlier draft counted EVERY call and flagged
        #: this checker's own `startswith("data/")` and the list its
        #: known-bad compares against -- a check that fires on its own text
        #: is not measuring the module.
        openers = {"open", "read_text", "read_bytes", "read_csv",
                   "read_parquet", "glob", "iterdir", "Path"}
        hits = []
        for c in ast.walk(t):
            if not isinstance(c, ast.Call):
                continue
            fn = getattr(c.func, "id", None) or getattr(c.func, "attr", None)
            if fn not in openers:
                continue
            for a in list(c.args) + [k.value for k in c.keywords]:
                for sn in ast.walk(a):
                    if isinstance(sn, ast.Constant) \
                            and isinstance(sn.value, str) \
                            and sn.value.startswith("data/"):
                        hits.append(sn.value)
        return hits

    ok(_data_literal_reaching_a_call(tree) == [],
       "NO DATA PATH IS OPENED: the consumed-source citations name files "
       "under data/ so a reader can locate them, and not one of those "
       "literals is an argument to any call in this module")
    _bad_tree = ast.parse('from pathlib import Path\n'
                          'x = Path("data/mm_hf/e1/tape.json").read_text()\n')
    ok(_data_literal_reaching_a_call(_bad_tree)
       == ["data/mm_hf/e1/tape.json"],
       "KNOWN-BAD, DRIVEN: the same check FIRES on a module that opens a "
       "data path -- a checker that has never been shown to fire is not "
       "what is passing above")

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
    # ---- v8: leg (d), the window, and the consumed set ----------------
    _pw = FORWARD_WINDOW_START_DAY
    _clean = outage_leg("BTCUSDT", _pw, 0.0)
    _hole = outage_leg("BTCUSDT", _pw, float(OUTAGE_RUN_S - 1))
    _run = outage_leg("BTCUSDT", _pw, float(OUTAGE_RUN_S))
    _long = outage_leg("BTCUSDT", _pw, 4656.0)
    ok(_clean["admissible"] and _clean["state"] == "ADMITTED_NO_OUTAGE_RUN"
       and _hole["admissible"]
       and not _run["admissible"] and _run["state"] == "REFUSED_OUTAGE_RUN"
       and not _long["admissible"],
       f"LEG (d) BOTH DIRECTIONS AT THE BOUNDARY: a day of scattered holes "
       f"whose longest run is {OUTAGE_RUN_S - 1} s ADMITS and one of "
       f"{OUTAGE_RUN_S} s REFUSES -- `>=`, and the leg can NOT fire, which "
       f"is what separates it from the activity filter v6 removed")

    _consumed_day = consumed_symbol_days()["BTCUSDT"][-1]
    _c = outage_leg("BTCUSDT", _consumed_day, 0.0)
    ok(not _c["admissible"]
       and _c["state"] == "REFUSED_CONSUMED_NEVER_REJUDGED"
       and _consumed_day in str(_c["why"]),
       f"KNOWN-BAD, DRIVEN: {_consumed_day} is CONSUMED for BTCUSDT and is "
       f"refused BY NAME as consumed on a measurement (0 s) that would "
       f"otherwise admit it -- the same input admits at {_pw} above, so the "
       f"refusal is the consumed set and not the number")

    _never = outage_leg("ETHUSDT", _consumed_day, 0.0)
    ok(not _never["admissible"]
       and _never["state"] == "REFUSED_BEFORE_THE_FORWARD_WINDOW",
       f"AND THE TWO REFUSALS ARE DIFFERENT FACTS: ETHUSDT was never "
       f"measured on the gap quantity, so {_consumed_day} is not CONSUMED "
       f"for it -- it is refused as OUTSIDE THE WINDOW, which is what "
       f"forward-only means for a symbol with no consumed history")

    _unmeasured = outage_leg("BTCUSDT", _pw, None)
    ok(not _unmeasured["admissible"]
       and _unmeasured["state"] == "REFUSED_OUTAGE_LEG_NOT_EVALUATED",
       "ABSENCE IS NOT A PASS: a symbol-day whose longest gap run was not "
       "measured is refused with its own status, exactly as the era leg "
       "refuses ERA_LEG_NOT_EVALUATED (rule 11)")

    _states = {outage_leg(*a)["state"] for a in (
        ("BTCUSDT", _pw, 0.0), ("BTCUSDT", _pw, 60.0),
        ("BTCUSDT", _consumed_day, 0.0), ("ETHUSDT", "20260901", 0.0),
        ("BTCUSDT", _pw, None))}
    ok(_states == set(OUTAGE_LEG_STATES)
       and set(d["admission_legs_v8"]["leg_d_outage_v8"]["states"])
       == _states,
       f"THE EMITTED STATES AND THE CODE CANNOT DISAGREE: the five declared "
       f"states are exactly the five the predicate produces "
       f"({sorted(_states)})")

    ok(all(day < FORWARD_WINDOW_START_DAY
           for s in CONSUMED_SOURCES for day in s["days"])
       and FORWARD_WINDOW_START_DAY == _day_plus(LATEST_CONSUMED_DAY, 1)
       and _day_plus(FORWARD_WINDOW_START_DAY, MIN_COMPLETE_DAYS - 1)
       == _day_plus(EARLIEST_READ_DAY, -1),
       f"THE WINDOW IS COMPUTED FROM THE ARTIFACTS, NOT TYPED: no consumed "
       f"day falls on or after {FORWARD_WINDOW_START_DAY}, the start is the "
       f"day after {LATEST_CONSUMED_DAY}, and the {MIN_COMPLETE_DAYS}th day "
       f"of the window closes the midnight before {EARLIEST_READ_DATE}")

    _bad_start = _day_plus(LATEST_CONSUMED_DAY, 0)
    ok(any(day >= _bad_start
           for s in CONSUMED_SOURCES for day in s["days"]),
       f"KNOWN-BAD, DRIVEN: a window starting one day earlier "
       f"({_bad_start}) WOULD contain a consumed day, so the cell above is "
       f"not a tautology about any start date")

    # ---- v8: legs (a), (b), (c) are carried, proven at v7's own bytes ----
    _v7 = json.loads((HERE.parent.parent / SUPERSEDES["path"]).read_text())
    _v7_sha = hashlib.sha256(
        (HERE.parent.parent / SUPERSEDES["path"]).read_bytes()).hexdigest()
    _carried = ("leg_a_streams", "leg_b_collector_liveness",
                "leg_c_rule5_era_purity")
    ok(_v7_sha == SUPERSEDES["sha256"]
       and all(d["admission_legs_v8"][k] == _v7["admission_legs_v7"][k]
               for k in _carried),
       f"LEGS (a), (b), (c) ARE CARRIED UNCHANGED, CHECKED AT v7's OWN "
       f"BYTES: v7 resolves at the pair this version supersedes "
       f"({_v7_sha[:16]}) and all three legs compare EQUAL -- v8 ADDS leg "
       f"(d) and removes no control")
    ok(_v7["admission_legs_v7"]["leg_b_collector_liveness"]
       != dict(d["admission_legs_v8"]["leg_d_outage_v8"]),
       "AND THE COMPARISON CAN FAIL: leg (d) is not equal to leg (b), so "
       "the cell above is comparing content and not an identity that holds "
       "for any two keys")

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

    # ---- v7: the ordering property, its three regimes, both directions ----
    def _episode(queue_ahead, order_qty, vol, depth):
        """The runner's own arithmetic, re-derived here from the published
        definitions so the declaration pins the semantics executably."""
        v_before, run = [], 0.0
        for v in vol:
            v_before.append(run)
            run += v
        front = [max(queue_ahead - vb, 0.0) for vb in v_before]
        back = [max(d - f, 0.0) for d, f in zip(depth, front)]
        probs = [probqueue_f3_fill_prob(f, b) for f, b in zip(front, back)]
        return {
            "front": front,
            "probs": probs,
            "testable": ordering_is_testable_per_episode(front),
            "filled_RiskAverse": riskaverse_filled_qty(queue_ahead, order_qty,
                                                       run),
            "E_filled_ProbQueue": expected_filled_probqueue(probs, order_qty),
        }

    TOL = 1e-9
    reg = ORDERING_REGIMES

    e_t = _episode(50.0, 10.0, [60.0, 60.0], [500.0, 500.0])
    ok(e_t["testable"] is True
       and abs(e_t["filled_RiskAverse"] - 10.0) <= TOL
       and abs(e_t["E_filled_ProbQueue"] - 10.0) <= TOL
       and reg["TESTABLE_full_overshoot"]["front_reaches_zero"] is True,
       "v7 ORDERING, TESTABLE REGIME (front reaches 0): RiskAverse "
       f"{e_t['filled_RiskAverse']} and E[ProbQueue] "
       f"{e_t['E_filled_ProbQueue']} -- once a trade arrives with the queue "
       "already cleared, p = 1 and ProbQueue fills the whole order, so the "
       "ordering is ARITHMETIC and a violation here can only be a defect")

    e_m = _episode(100.0, 10.0, [105.0], [200.0])
    ok(e_m["testable"] is False
       and abs(e_m["probs"][0] - 0.5) <= TOL
       and abs(e_m["filled_RiskAverse"] - 5.0) <= TOL,
       "v7 ORDERING, MARGINAL REGIME (the reviewer's A.5 case): front never "
       f"reaches 0, p = {e_m['probs'][0]} exactly, RiskAverse fills "
       f"{e_m['filled_RiskAverse']} off the CUMULATIVE volume while "
       "ProbQueue is a coin flip -- 993 of 2,000 seeds violate the "
       "realisation ordering and NOT ONE of them is a defect")

    e_c = _episode(100.0, 10.0, [110.0], [1000.0])
    exp_ce = 10.0 * (900.0 ** 3) / (100.0 ** 3 + 900.0 ** 3)
    ok(e_c["testable"] is False
       and abs(e_c["filled_RiskAverse"] - 10.0) <= TOL
       and abs(e_c["E_filled_ProbQueue"] - exp_ce) <= TOL
       and e_c["E_filled_ProbQueue"] < e_c["filled_RiskAverse"] - TOL,
       "v7 THE EXPECTATION ALONE IS NOT THE FIX -- KNOWN-BAD FOR A.5's OWN "
       f"PROPOSAL: RiskAverse fills {e_c['filled_RiskAverse']} (the whole "
       f"order) while E[ProbQueue] = {e_c['E_filled_ProbQueue']:.11f} < 10, "
       "in the marginal regime, with no defect present. The expectation is "
       "true exactly on the TESTABLE set, so the front = 0 predicate is "
       "doing all the work and this control is what stops that being taken "
       "on trust")

    v_ref = ordering_verdict(n_testable=100, n_violations_testable=1,
                             n_marginal=0, n_violations_marginal=0)
    v_marg = ordering_verdict(n_testable=100, n_violations_testable=0,
                              n_marginal=40, n_violations_marginal=19)
    ok(v_ref["state"] == "REFUTES_THE_BRACKET",
       "v7 REFUTATION TRIGGER, POSITIVE CONTROL -- IT CAN STILL FIRE: one "
       "violation in the TESTABLE regime raises REFUTES_THE_BRACKET, so "
       "narrowing the trigger has not disarmed it")
    ok(v_marg["state"] == "ORDERING_HOLDS_WHERE_TESTABLE"
       and v_marg["n_violations_in_the_marginal_regime"] == 19
       and v_marg["marginal_violations_do_NOT_silence_the_gate"] is True,
       "v7 REFUTATION TRIGGER, KNOWN-BAD FOR THE OLD CODE: 19 violations in "
       "the MARGINAL regime and none in the testable one leaves the gate "
       "readable and counts them as a status. The pre-v7 runner would have "
       "declared its own instrument refuted and read no gate")

    ok(reg["MARGINAL_expectation_counterexample_DA63"]["may_refute"] is False
       and reg["MARGINAL_reviewer_A5"]["may_refute"] is False
       and reg["TESTABLE_full_overshoot"]["may_refute"] is True,
       "v7 EXACTLY ONE REGIME MAY REFUTE, and it is the arithmetic one -- "
       "declared as data, so a reader can check which disagreements are "
       "allowed to silence a gate")

    # ---- v7: rule 5's era leg ---------------------------------------------
    era = declaration()["admission_legs_v8"]["leg_c_rule5_era_purity"]
    ok(ERA_BOUNDARY_RECV_NS == 1787579334881534478
       and ERA_BOUNDARY_UTC == "2026-08-24T13:48:54Z"
       and str(ERA_BOUNDARY_RECV_NS) in era["predicate"]
       and "row-wise" in era["predicate"],
       "v7 ERA LEG carries rule 5's OWN number and measures it ROW-WISE: a "
       "symbol-day is era-admissible iff every bookTicker row carries "
       f"recv_ns >= {ERA_BOUNDARY_RECV_NS} -- never inferred from the date, "
       "because the boundary falls at 13:48:54 inside a day")
    ok("11" in era["the_population_collapse_STATED"]
       and "14" in era["the_population_collapse_STATED"]
       and "rule 11" in era["NO_THRESHOLD_CHANGE_AFTER_SEEING"],
       "v7 THE POPULATION COLLAPSE IS STATED AND THE BAR IS NOT MOVED: at "
       "most 11 admissible post-boundary days against a declared minimum of "
       "14, and the minimum stays 14 -- what changes is what may be READ")
    seal = era["the_sealed_smoke_regime"]
    ok("eff_RT" in seal["what_is_SEALED"]
       and "verdict" in seal["what_is_SEALED"]
       and "NOT READ" in seal["what_is_SEALED"]
       and "14" in seal["when_the_gate_may_be_read"],
       "v7 THE SEALED SMOKE NAMES WHAT IS SEALED (eff_RT, the per-episode "
       "costs, the intervals and the verdict) and when it may be opened "
       "(G >= 14 post-boundary days) -- a seal that did not say what it "
       "covers would be a seal over whatever was convenient")

    # ---- v7: the liveness controls, and the WITHDRAWN one ------------------
    live = declaration()["admission_legs_v8"]["leg_b_collector_liveness"]
    pos = " ".join(live["positive_controls_RULED"])
    ok("2026-08-24" in pos and "2026-08-26" in pos
       and "4,656" in pos and "158 s" in pos,
       "v7 LIVENESS POSITIVE CONTROLS are the two MEASURED collector events "
       "-- 08-24 at 158 s with 2 restarts and 08-26 at 4,656 s -- so the leg "
       "is pinned by days it must REFUSE")
    ok("2026-08-29" not in pos and "2026-08-30" not in pos
       and "MUST ADMIT" in live["negative_control_RULED"]
       and "quiet weekend" in live["negative_control_RULED"],
       "v7 THE WITHDRAWN CONTROL CANNOT CREEP BACK: 08-29/30 appears ONLY as "
       "the NEGATIVE control that must ADMIT. Requiring the leg to flag a "
       "quiet weekend would rebuild the activity-selecting defect v6 removed "
       "-- through the control instead of the predicate")
    ok("92-93%" in live["the_withdrawn_control"]
       and "fixed-cadence" in live["the_withdrawn_control"].lower(),
       "v7 AND THE WITHDRAWAL CARRIES ITS EVIDENCE: the event-driven streams "
       "fell about 40% while the FIXED-CADENCE depth20 held at 92-93% -- an "
       "outage suppresses every stream, a quiet market only the event-driven "
       "ones")

    # ---- v7: the staleness bar, and where it does NOT come from -----------
    qa = declaration()["decision_time_quote_age_v7"]
    ok(STALENESS_BAR_MS == 1000.0
       and "rule 5" in qa["where_the_bar_COMES_FROM"]
       and "517" in qa["where_the_bar_COMES_FROM"]
       and "after seeing" in qa["where_the_bar_COMES_FROM"],
       "v7 THE STALENESS BAR IS 1,000 ms FROM RULE 5's OWN '>= 1 s bars "
       "only' LINE, and the declaration names the number it is NOT (ICP's "
       "measured 517 ms median) so a reader can see the bar was not tuned")
    ok("the_gate_is_read_BOTH_WAYS" in qa
       and "reading A over ALL resolved" in qa["the_gate_is_read_BOTH_WAYS"]
       and "Never averaged" in qa["a_straddle_is_a_stated_sensitivity"],
       "v7 THE GATE IS READ BOTH WAYS and a straddle is STATED, never "
       "averaged -- the same rule the queue bracket already carries")

    # ---- v7: the ICP label, the structural partial, the two corrections ----
    icp7 = declaration()["the_ICP_cell_label_v7"]
    ok(icp7["label"] == "NOT_COMPARABLE_ON_PLACEMENT_QUALITY"
       and "stale" in icp7["the_mechanism"].lower()
       and "sweep through" in icp7["the_mechanism"]
       and "736" in icp7["not_this"],
       "v7 THE ICP CELL IS LABELLED WITH ITS MECHANISM: a resting order at a "
       "STALE touch swept by later prints inflates fill rate and capture -- "
       "and the label is not 'too few episodes', because about 736 gate-row "
       "episodes exist")
    PF_KEY = "the_partial_fill_bracket_is_STRUCTURAL_in_the_min_size_arm"
    ok(PF_KEY in declaration()
       and "bracket NOTHING" in declaration()[PF_KEY]
       and "size-aware arm, which REFUSES" in declaration()[PF_KEY],
       "v7 THE PARTIAL-FILL PRICINGS BRACKET NOTHING IN THE MIN-SIZE ARM: "
       "phi is 0 or 1 almost surely at q = one step, so the pair is inert on "
       "the arm that runs and live only on the arm that refuses")
    rc = declaration()["reporting_corrections_v7"]
    ok("null is not false" in rc["no_repro_receipt_field"]["the_rule"]
       and "gates_the_run: false" in rc["no_repro_receipt_field"]["the_rule"],
       "v7 --no-repro WRITES THE FIELD RATHER THAN GOING SILENT: "
       "{reproduced: null, gates_the_run: false, why_skipped} -- null is not "
       "false, because the control neither passed nor failed")
    ok(MIN_DISTINCT_QUANTITIES == 3
       and "QTY_STEP_UNDERDETERMINED" in rc["qty_step_underdetermined"][
           "the_rule"],
       "v7 QTY_STEP_UNDERDETERMINED below three distinct quantities: a modal "
       "diff over one or two values is an echo of the sample, not a step")

    # ---- v7 / R-584: the BTC scope and the resource boundary -------------
    sc = declaration()["scope_R584_BTC_ONLY"]
    ok(sc["the_gate_symbol"] == "BTCUSDT" == GATE_SYMBOL
       and "DEFERRED, NOT RESOLVED" in " ".join(sc["what_it_means_here"])
       and "must not be read as having resolved" in sc[
           "the_ICP_cell_is_DEFERRED_not_answered"],
       "R-584 SCOPE IS BTC and the thin-name cell is DEFERRED, NOT RESOLVED "
       "-- the declared handling for ICP stands and travels with the cell, "
       "but E2-A under this scope does not answer it")
    ok(declaration()["the_ICP_cell_label_v7"]["status_under_R584"]
       == "DEFERRED_NOT_RESOLVED",
       "AND THE ICP BLOCK ITSELF CARRIES THE DEFERRAL, so a reader who "
       "reaches the label without reading the scope block still cannot take "
       "it for a result")
    ok("WEAKEST" in sc["what_a_reader_must_not_do"],
       "R-584 NAMES THE MISREADING IT INVITES: BTC is the most active symbol "
       "in the set, so the placement-quality problem the ICP label describes "
       "is at its WEAKEST there -- a BTC number is not an answer for a thin "
       "name")
    rb = declaration()["the_resource_boundary_R584"]
    ok(rb["the_cap"] == DAY_RSS_CAP_GIB == 6.0
       and "8 GiB" in rb["where_the_cap_comes_from"]
       and "NOT tuned on BTC" in rb["where_the_cap_comes_from"]
       and "NEVER RAISED" in rb["on_a_breach"]
       and "NEVER MADE SMALLER" in rb["on_a_breach"],
       f"R-584 RESOURCE BOUNDARY: the cap is {DAY_RSS_CAP_GIB} GiB = 75% of "
       f"the rule-20 wrapper's own 8 GiB MemoryMax, because a cgroup kill is "
       f"silent -- and on a breach the day REFUSES rather than the cap being "
       f"raised or the population made smaller")
    ok("STREAMED" in rb["the_rule"] and "EPISODE GRID" in rb["the_rule"]
       and "must REFUSE" in rb["the_falsifier_runs_both_ways"]
       and "must ADMIT" in rb["the_falsifier_runs_both_ways"],
       "AND THE STREAM IS DECLARED AS A RULE WITH A TWO-WAY FALSIFIER: only "
       "the episode grid is kept, and a cap below the measured residency "
       "must REFUSE while the real cap ADMITS the same day")

    # ---- the emitter is cited by CONTENT, which a rebase cannot rewrite --
    eid = declaration()["emitter_identity"]
    import hashlib as _h
    live_digest = _h.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    ok(eid["emitter_sha256"] == live_digest and len(live_digest) == 64
       and "rebase" in eid["emitter_sha256_is_the_durable_citation"]
       and "best_effort" in "".join(eid.keys()),
       "THE EMITTER IS CITED BY CONTENT DIGEST, NOT ONLY BY COMMIT: DA 63 "
       "landed the v7 emitter, emitted against it, and another seat's "
       "landing REBASED that commit to a new id with byte-identical content "
       "-- a declaration carrying only a commit id had an address that was "
       "rewritten out from under it. The digest cannot be rewritten")

    # ---- rule 20's clause (REV 84 S3.2 / REV 85 S3, R-726) -------------
    #: THE SHARED MODULE'S OWN FALSIFIER RUNS AS ONE CELL OF THIS BATTERY.
    #: This emitter imports `declaration_chain` for the compare-and-swap,
    #: so a regression in the one implementation is this battery's problem
    #: too. SPAWNED AS A PROCESS, not called: a broken `__main__`, a syntax
    #: error under an edit, or a falsifier that no longer runs at all is
    #: then a failure HERE rather than something an in-process call routes
    #: around. It replaces nothing this module's own verdicts rest on.
    def _dc_falsify(_prog):
        _r = subprocess.run([sys.executable, str(_prog), "--falsify"],
                            capture_output=True, text=True, timeout=300)
        _ls = [x for x in (_r.stdout or "").strip().splitlines() if x.strip()]
        return (_r.returncode, _ls[-1] if _ls else "",
                [x for x in _ls if x.startswith("FAIL")])

    _DC_PATH = _PM_RESEARCH / "declaration_chain.py"
    _dc_rc, _dc_sum, _dc_bad = _dc_falsify(_DC_PATH)
    ok(_dc_rc == 0 and _dc_sum.endswith("0 failures") and not _dc_bad,
       f"REV 84 S3.2 -- ONE IMPLEMENTATION, N DETECTORS: this battery RUNS "
       f"`declaration_chain.py --falsify` AS A SUBPROCESS -> rc {_dc_rc}: "
       f"{_dc_sum!r}. A regression in the shared chain module fails every "
       f"importer at once, and no importer re-implements its logic")

    #: RED FIRST. A cell that only ever runs the GOOD module has never been
    #: shown to fire, so the same cell is driven against a COPY with ONE
    #: falsifier disarmed -- VERSION_PATH_EXISTS, the refusal that makes a
    #: landed version immutable, which is the guarantee THIS emitter rests
    #: on when it writes v8 through the compare-and-swap.
    import tempfile                                           # noqa: PLC0415
    with tempfile.TemporaryDirectory() as _dc_td:
        _dc_copy = Path(_dc_td) / "declaration_chain.py"
        _dc_src = _DC_PATH.read_text()
        _dc_dis = _dc_src.replace("    if dst.exists():",
                                  "    if False and dst.exists():")
        _dc_copy.write_text(_dc_dis)
        _bad_rc, _bad_sum, _bad_fails = _dc_falsify(_dc_copy)
    ok(_dc_dis != _dc_src and _bad_rc != 0 and "1 failures" in _bad_sum
       and _bad_fails,
       f"KNOWN-BAD, DRIVEN: the SAME cell against a COPY with one falsifier "
       f"disarmed FAILS -> rc {_bad_rc}: {_bad_sum!r}; "
       f"{(_bad_fails or [''])[0][:70]}")

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
        #: R-711 / SEAT_PROTOCOL rule 20: the closure is AT THE MOMENT OF THE
        #: WRITE. The head is resolved HERE, the payload composed from it,
        #: and the shared implementation refuses VERSION_PATH_EXISTS (a
        #: landed version is immutable), HEAD_MOVED (another writer landed
        #: between the read and the write) or PAIR_MISMATCH (the payload
        #: claims a chain that does not exist). No branch of this emitter
        #: writes a declaration any other way.
        head = CHAIN.resolve_head(OUT.parent, FAMILY)
        if head["sha256"] != SUPERSEDES["sha256"]:
            print(f"REFUSED: the chain head is {head['name']} at "
                  f"{head['sha256'][:16]} but this module supersedes "
                  f"{Path(SUPERSEDES['path']).name} at "
                  f"{SUPERSEDES['sha256'][:16]}. Re-read the head and "
                  f"recompose -- a version number is never taken from a "
                  f"dispatch.")
            return 3
        res = CHAIN.write_next_version(OUT.parent, FAMILY, declaration(),
                                       head["pair"])
        if Path(res["path"]) != OUT:
            print(f"REFUSED: the chain wrote {res['path']} while this "
                  f"module's DECL_VERSION names {OUT}")
            return 4
        print(f"{res['path']}  sha256 {res['sha256']}  supersedes "
              f"{Path(res['superseded']['path']).name} "
              f"{res['superseded']['sha256'][:16]}")
        return 0
    ap.error("choose --selftest or --emit")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
