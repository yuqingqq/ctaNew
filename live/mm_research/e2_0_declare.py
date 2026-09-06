"""P-2026-002 E2.0 — the declaration, emitted BEFORE any tape is read.

WHAT E2.0 IS FOR, in the programme's own words. E1-B found no standalone
Binance market-making candidate except **ADA +2.44 bps**, and the results audit
called that ARTIFACT-LIKELY: ADA is a fat-tick name (tick 5.64 bps, pinned
31/31) whose eq-weighted realized half-spread flips to **notional-weighted
−0.322** — small prints capture the half-tick, the DOLLARS are adversely
selected. Two binding checks were pre-registered before any L2 existed:

  (i)  `EXPERIMENT_PLAN.md` §2: recompute §1.3(a)-(c) with TRUE bookTicker
       mids and report `Δrs(τ*) = rs_proxy − rs_true`. **Δrs > +1.0 bps on an
       E1-B passer VOIDS its pass.**
  (ii) `E1_CODE_REVIEW.md` pre-read amendment: **the gate quantity is the
       NOTIONAL-WEIGHTED rs on true mids; if it is below fee at VIP0 on the L2
       window the ADA cell dies regardless of eq numbers.**

This file writes those two, and everything they depend on, as DATA — before
the tape is opened. It cannot read the tape: it imports no reader and takes no
data path. The runner (`e2_0_true_mid.py`) verifies this declaration's digest
before it computes anything, so the gate cannot be redefined after seeing.

THREE THINGS THIS DECLARATION FIXES THAT THE PREREG COULD NOT HAVE KNOWN, each
stated as a decision with its reason rather than absorbed:

  1. **THERE IS NO OVERLAP WINDOW.** E1 ran on Vision tick aggTrades
     2026-07-18→08-17. The L2 collector started 2026-08-19. The windows DO NOT
     INTERSECT, so `Δrs = rs_proxy − rs_true` cannot be computed the way §2
     imagined — across the two windows it would confound METHOD with PERIOD.
     So Δrs is computed with BOTH mids on the SAME collected days, from the
     SAME sweep events, differing only in the mid. Δrs is then method-only,
     and `rs_true` is a statement about 2026-08-20..09-05 and NOT about E1's
     window. **The ADA settlement is a settlement on a new period; that is a
     limitation of the settlement, not a licence to compare across windows.**
  2. **THE PROXY IMPLEMENTATION MUST BE PROVED AGAINST E1's OWN NUMBERS.** A
     reimplemented proxy that quietly differs from E1's would make Δrs a
     statement about two codebases. The Vision aggTrades are still on disk, so
     the runner REPRODUCES E1's published ADA numbers on E1's own window
     (eq +2.443 / notional −0.322 at τ*=30) as a REQUIRED control. Failure to
     reproduce refuses the whole run.
  3. **THE MID ALIGNMENT USES EXCHANGE `T`, NEVER `recv_ns`.** CLAUDE.md rule 5
     makes sub-second `recv_ns` reliable only from 2026-08-24T13:48:54Z, and
     this window starts 08-20 — so a recv_ns alignment would make a third of
     the days inadmissible AND would fold our own ~74 ms latency and clock
     offset into the markout. Exchange transact_time is the stamp the event
     itself carries (CLAUDE.md rule 3). The boundary is therefore not binding
     on any gate quantity — and because "not binding" is a claim, the run
     ALSO reports every gate on the post-boundary subset as a declared
     robustness split.

    python3 live/mm_research/e2_0_declare.py --selftest
    python3 live/mm_research/e2_0_declare.py --emit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECL_VERSION = 2
OUT = HERE / "declarations" / f"p002_e2_0_declaration_v{DECL_VERSION}.json"
PROTOCOL = f"P002_E2_0_TRUE_MID_DECLARATION_V{DECL_VERSION}"
SUPERSEDES = {
    "path": "live/mm_research/declarations/p002_e2_0_declaration_v1.json",
    "sha256": "95aeaf1bc35fd8fba1eb96af54429ec1bf7c366afc9e87f0b5be1248504d442a",
    "carrying_commit": "a7b5534152e83b59dda196ab6fa07fc9a0131c99",
    "correction_is_in_band": (
        "rule 13: v1 is NOT edited and stands as provenance. v2 supersedes it "
        "on the reviewer's three findings (46030b1, filed 2026-09-06T04:33Z, "
        "BEFORE the smoke ran and BEFORE any result was read). The smoke's "
        "receipt is emitted under v2; the v1 run's numbers were never read, "
        "never cited and its unread receipt was removed."),
    "what_changed": [
        "FINDING 1: Delta_rs is now declared on the INTERSECTION of sweeps "
        "where BOTH mids are defined, with the events dropped for want of a "
        "proxy counted as a status. v1 said 'the SAME sweep events' -- but "
        "the true mid has no validity window and the proxy needs two prints "
        "within 10 s, so v1 would have let a POPULATION difference leak into "
        "what is supposed to be a MID difference.",
        "FINDING 2: the settle leg is relabelled by the GATE it clears "
        "(fee + c_safe = 2.3, EXPERIMENT_PLAN section 1.5 gate 1) and killed "
        "by the DEATH bar (fee = 1.8, the amendment). v1's SETTLED_ALIVE was "
        "labelled at 1.8, so a cell at 2.0 would have read ALIVE while "
        "FAILING gate 1. The 1.8-2.3 band is real and now has its own name.",
        "FINDING 3: the interval is DECISION-BEARING on the alive side. "
        "ALIVE requires the point >= 2.3 AND the bootstrap CI lower bound "
        ">= 1.8 (the plan's gate 2). A point that clears with an interval "
        "that does not is INCONCLUSIVE, not alive. The KILL stays a point "
        "rule because that is the pre-registered text and a kill must not be "
        "weakened after seeing -- but its interval robustness is reported.",
    ],
}

# ---- constants, every one carried from a named source, none invented here ---
FEE_MAKER_VIP0 = 1.8          # bps, +BNB — EXPERIMENT_PLAN §0 / e1_markout_scan FEE_MAKER
FEE_MAKER_VIP1 = 1.44
C_SAFE = 0.5                  # EXPERIMENT_PLAN §1.5 gate 1 (fee + c_safe = 2.3 at VIP0)
VOID_THRESHOLD_BPS = 1.0      # EXPERIMENT_PLAN §2 E2.0
TAUS_S = (0.1, 0.5, 1, 5, 15, 30, 60, 300)   # §1.3b grid + the §2 sub-second points
TAU_STAR_DEFAULT_S = 30       # §1.5, the pre-registered gate point
VALID_MS = 10_000             # §1.2 two-sided last-print mid validity (proxy leg only)
IDENTITY_TOL_BPS = 0.2        # §1.3c: |mean es − mean Λ − mean MO| > 0.2 → proxy_incoherent
GAP_FRACTION_MAX = 0.05       # §2 data prerequisite
MIN_COMPLETE_DAYS = 14        # §2 data prerequisite
DAYS_POSITIVE_FRACTION = 0.70  # §1.5 gate 3 ("> 0 on ≥ 70% of days")
TAU_STAR_DAY_FRACTION = 24 / 31  # §1.5 τ* rule, expressed as the fraction the plan gives
ERA_BOUNDARY_NS = 1787579334881534478   # CLAUDE.md rule 5 / collector_runs.jsonl
BOOT_B = 2000
BOOT_SEED = 20260906

#: E1's published ADA numbers, τ*=30, 31 days 2026-07-18..08-17, from
#: E1_RESULTS.md and E1_CODE_REVIEW.md §(b). The runner must reproduce these on
#: E1's own data before it is allowed to report anything about the new window.
E1_ADA_REPRODUCTION_TARGET = {
    "window": "2026-07-18..2026-08-17 (31 d Vision tick aggTrades)",
    "tau_star_s": 30,
    "rs_eq_bps": 2.443,
    "rs_notional_bps": -0.322,
    "days_eq_positive": 31,
    "days_notional_positive": 7,
    "n_days": 31,
    "tolerance_bps": 0.05,
    "source": "live/mm_research/E1_RESULTS.md + E1_CODE_REVIEW.md section (b)",
    "why_required": (
        "Delta_rs is a difference between two mids. If the proxy leg is a "
        "reimplementation that does not match E1's, Delta_rs is a difference "
        "between two CODEBASES and the voiding rule means nothing."),
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def carrying_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(HERE))
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


def days_threshold(n_days: int, fraction: float) -> int:
    """The plan states its day counts for 31 days. Re-express, never re-choose."""
    return math.ceil(fraction * n_days)


# --------------------------------------------------------------------------
# THE TWO PRE-REGISTERED PREDICATES, AS FUNCTIONS SO THEY CAN BE DRIVEN
# --------------------------------------------------------------------------
def void_predicate(delta_rs_bps: float | None) -> dict:
    """(i) EXPERIMENT_PLAN §2: Delta_rs > +1.0 bps VOIDS the E1 pass.

    Delta_rs = rs_proxy - rs_true, EQ-WEIGHTED, both on the SAME collected
    days. Eq-weighted deliberately: that is the weighting E1 actually gated on,
    so this asks "was the number E1 read inflated by its mid?" -- a different
    question from (ii), which asks "is the right quantity positive?".
    """
    if delta_rs_bps is None:
        return {"decidable": False, "voids": None,
                "why": "Delta_rs is undefined (no admissible overlap of "
                       "proxy-valid and true-mid-valid events); the void leg "
                       "cannot be read and says so"}
    return {"decidable": True, "voids": bool(delta_rs_bps > VOID_THRESHOLD_BPS),
            "delta_rs_bps": delta_rs_bps, "threshold_bps": VOID_THRESHOLD_BPS,
            "why": ("the proxy mid was optimistic by more than the "
                    "pre-registered tolerance, so E1's number is not readable"
                    if delta_rs_bps > VOID_THRESHOLD_BPS else
                    "the proxy mid is within tolerance; E1's number is "
                    "readable as a measurement, whatever its sign")}


def settle_predicate(rs_true_notional_bps: float | None,
                     tier: str = "vip0",
                     ci_lo_bps: float | None = None,
                     ci_hi_bps: float | None = None,
                     interval_claimable: bool = True) -> dict:
    """(ii) The pre-read amendment: notional-weighted rs on TRUE mids.

    "if notional-weighted rs(tau*) < fee at VIP0 on the L2 window, the ADA cell
    dies regardless of eq numbers" -- E1_CODE_REVIEW.md.
    """
    fee = FEE_MAKER_VIP0 if tier == "vip0" else FEE_MAKER_VIP1
    gate = fee + C_SAFE                     # section 1.5 gate 1: 2.3 at VIP0
    if rs_true_notional_bps is None:
        return {"decidable": False, "state": None, "dies": None,
                "death_bar_bps": fee, "gate1_bar_bps": gate,
                "why": "no admissible notional-weighted estimate exists"}
    x = rs_true_notional_bps
    dies = x < fee                          # the amendment's rule, UNCHANGED
    point_clears_gate1 = x >= gate          # the plan's gate 1
    interval_clears_gate2 = (ci_lo_bps is not None and ci_lo_bps >= fee)
    if dies:
        state = "DEAD"
    elif not point_clears_gate1:
        state = "NOT_KILLED_PENDING_GATE_1"
    elif not interval_claimable:
        state = "INCONCLUSIVE_NO_INTERVAL"
    elif not interval_clears_gate2:
        state = "INCONCLUSIVE_INTERVAL_DOES_NOT_CLEAR"
    else:
        state = "ALIVE_CLEARS_GATE_1_AND_2"
    return {
        "decidable": True, "state": state, "dies": bool(dies),
        "death_bar_bps": fee, "gate1_bar_bps": gate,
        "rs_true_notional_bps": x,
        "point_clears_gate1": bool(point_clears_gate1),
        "ci_lo_bps": ci_lo_bps, "ci_hi_bps": ci_hi_bps,
        "interval_claimable": bool(interval_claimable),
        "interval_clears_gate2": bool(interval_clears_gate2),
        "kill_is_interval_robust": (
            None if (dies is False or ci_hi_bps is None)
            else bool(ci_hi_bps < fee)),
        "why": {
            "DEAD": "per DOLLAR of maker fill the realized half-spread is "
                    "below the VIP0 maker fee -- the amendment's kill, a "
                    "POINT rule because that is its pre-registered text and a "
                    "kill is not weakened after seeing. Its interval "
                    "robustness is reported beside it, never instead of it.",
            "NOT_KILLED_PENDING_GATE_1":
                f"between the death bar {fee} and the plan's own gate {gate}: "
                f"NOT dead, and NOT a pass. v1 called this band ALIVE, which "
                f"would have labelled a cell that FAILS section 1.5 gate 1 as "
                f"surviving.",
            "INCONCLUSIVE_NO_INTERVAL":
                "the point clears the gate but G < 5 complete days, so no "
                "interval is claimable (CLAUDE.md rule 8) and ALIVE cannot be "
                "asserted from a point",
            "INCONCLUSIVE_INTERVAL_DOES_NOT_CLEAR":
                f"the point clears {gate} but the 95% CI lower bound is below "
                f"the fee {fee}, failing the plan's gate 2 -- a point that "
                f"clears with an interval that does not is INCONCLUSIVE",
            "ALIVE_CLEARS_GATE_1_AND_2":
                "the point clears gate 1 AND the interval clears gate 2. This "
                "is the outcome the E1 audit did NOT expect and it carries "
                "the heavier burden: the remaining section 1.5 conditions "
                "still apply.",
        }[state],
    }


def ada_verdict(void: dict, settle: dict) -> dict:
    """The 2x2. Deliberately has no 'unknown' cell when both legs are decidable.

    The two legs are INDEPENDENT and both can fire: one asks whether E1's
    measurement is readable, the other whether the right quantity pays.
    """
    if not void["decidable"] or not settle["decidable"]:
        return {"verdict": "UNDECIDABLE",
                "why": "at least one leg had no admissible estimate; reported "
                       "as a status, never resolved by assumption (rule 4)"}
    v, st = void["voids"], settle["state"]
    return {
        "verdict": ("VOIDED+" if v else "NOT_VOIDED+") + st,
        "voids_e1_pass": bool(v),
        "settle_state": st,
        "cell_dies": bool(st == "DEAD"),
        "cell_passes_gate1": bool(st == "ALIVE_CLEARS_GATE_1_AND_2"),
        "why_void": void["why"],
        "why_settle": settle["why"],
        "the_two_legs_are_independent": (
            "one asks whether E1's MEASUREMENT is readable, the other whether "
            "the RIGHT quantity pays. Both can fire; neither implies the "
            "other; and the settle state is named by the bar it actually "
            "crosses, never by one boolean doing duty for two bars."),
    }


# --------------------------------------------------------------------------
def declaration(n_days_expected: int = 16) -> dict:
    return {
        "protocol": PROTOCOL,
        "status": "DECLARATION_NO_DATA_TOUCHED",
        "supersedes": SUPERSEDES,
        "program": "P-2026-002-hf-market-making",
        "step": "E2.0 -- notional-weighted true-mid recompute of the E1 screen",
        "carrying_commit": carrying_commit(),
        "declared_by": "DA seat (pm-da), single-seat program",
        "sources": {
            "prereg": "live/mm_research/EXPERIMENT_PLAN.md sections 1.3, 1.5, 2",
            "amendment": "live/mm_research/E1_CODE_REVIEW.md -- 'What E1x/E2.0 "
                         "must check', item (ii)",
            "e1_verdict": "live/mm_research/E1_RESULTS.md",
            "e1_code": "live/mm_research/e1_markout_scan.py",
        },

        "the_true_mid": {
            "definition": "m_true(u) = (best_bid + best_ask) / 2 taken from "
                          "the LAST bookTicker message with exchange "
                          "transact_time T <= u; for m(t-) the search is "
                          "STRICTLY BEFORE the sweep's own T.",
            "source_stream": "data/mm_hf/raw/bookTicker/<SYM>/<YYYYMMDD>_<HH>.csv[.gz]",
            "schema": "recv_ns,E,T,u,bid,bid_qty,ask,ask_qty",
            "clock": "exchange transact_time T (milliseconds)",
            "why_not_recv_ns": (
                "recv_ns is OUR stamp. CLAUDE.md rule 5 makes sub-second "
                "recv_ns reliable only from 2026-08-24T13:48:54Z "
                f"(ns {ERA_BOUNDARY_NS}), which would make the first third of "
                "this window inadmissible; and it carries ~74 ms of measured "
                "one-way latency plus clock offset straight into the markout. "
                "Exchange T is the stamp the event itself carries "
                "(CLAUDE.md rule 3)."),
            "validity_rule": "NONE. The book is continuous: a bookTicker "
                             "message stands until the next one. The 10 s "
                             "validity window belongs to the PROXY mid, which "
                             "is built from prints and goes stale.",
            "staleness_is_reported_not_filtered": (
                "age of the standing quote at each query time is reported "
                "(p50/p99 per symbol-day) so a reader can see whether 'true' "
                "meant 'fresh'. It does not filter events, because filtering "
                "on quote age would select on activity."),
        },

        "the_proxy_mid": {
            "definition": "EXPERIMENT_PLAN section 1.2: two-sided last-print "
                          "mid, (last taker-buy sweep price + last taker-sell "
                          "sweep price)/2, both required within "
                          f"{VALID_MS} ms of the query time.",
            "why_recomputed_here": "so that Delta_rs isolates the MID and not "
                                   "the period -- see the reproduction control",
        },

        "population": {
            "symbols": "ALL 16 collected symbols, not a subset. The plan asks "
                       "for 'E1-B passers + 3 fixed negative controls + "
                       "XS-overlap'; running all 16 is a superset, drops "
                       "nothing, and removes every question about which three "
                       "controls were meant.",
            "symbol_list": ["AAVEUSDT", "ADAUSDT", "APTUSDT", "ARBUSDT",
                            "ATOMUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT",
                            "DOGEUSDT", "ETHUSDT", "FILUSDT", "GMXUSDT",
                            "ICPUSDT", "LTCUSDT", "SOLUSDT", "XRPUSDT"],
            "day_admission_predicate": (
                "a UTC day is ADMISSIBLE iff (a) 24 hour-files exist for BOTH "
                "bookTicker and trade, and (b) the intra-day gap fraction is "
                f"< {GAP_FRACTION_MAX} measured as the share of the day's "
                "seconds with no bookTicker message. Both computed at run "
                "time."),
            "the_admissible_set_is_an_OUTPUT": (
                "declared as an output, never as an expectation, so no belief "
                "about which days should qualify can become a filter. "
                "Structural file-count as-of 2026-09-06T04:20Z: 16 complete "
                "days for every symbol (08-20..08-25, 08-27..09-05); 08-19 is "
                "a 12-hour partial, 08-26 a 23-hour partial (reboot), 09-06 "
                "in progress. The gap-fraction leg has NOT been evaluated."),
            "min_complete_days": MIN_COMPLETE_DAYS,
            "refuses_below_min": True,
            "cluster_unit": "UTC day (CLAUDE.md rule 8)",
            "event_unit": (
                "the SWEEP, not the print: prints are collapsed on "
                "(transact_time, is_buyer_maker) exactly as EXPERIMENT_PLAN "
                "section 1.1 and e1_markout_scan.sweeps do, with the "
                "qty-weighted sweep price. The collected `trade` stream is "
                "per-match and FINER than E1's aggTrades, so without this "
                "collapse the event populations would not be comparable."),
            "sign_convention": (
                "q_j = -1 when is_buyer_maker (taker sold, the maker's BID was "
                "filled), +1 otherwise. MO_j(tau) = -q_j * (m(t_j+tau) - p_j) "
                "/ p_j * 1e4 bps -- a maker-bid fill followed by a rising mid "
                "is POSITIVE. Driven in both directions by the falsifiers."),
        },

        "which_population_each_leg_is_computed_over": {
            "why_this_field_exists": (
                "REVIEWER FINDING 1 (46030b1): v1 said Delta_rs used 'the "
                "SAME sweep events', but the two mids DO NOT EXIST on the "
                "same sweeps. The true mid needs only a prior bookTicker "
                "message; the proxy needs a taker-buy AND a taker-sell print "
                "within 10 s. In a print drought the proxy is undefined and "
                "the true mid is not. Unrestricted, Delta_rs would mix a MID "
                "difference with a POPULATION difference -- the one thing it "
                "exists not to do."),
            "VOID_leg_population": (
                "the INTERSECTION: sweeps of an admissible day where BOTH "
                "mids are defined at t- AND at t+tau*, i.e. "
                "v0_true & v1_true & v0_proxy & v1_proxy. Delta_rs is "
                "computed over exactly this set for both mids, so the only "
                "thing that differs between the two terms is the mid."),
            "SETTLE_leg_population": (
                "ALL sweeps of an admissible day with a valid TRUE mid at t- "
                "and t+tau*. NOT the intersection: the amendment says "
                "'notional-weighted rs on true mids', and restricting the "
                "economics to the events the PROXY happens to see would let "
                "the proxy's limitations select the settle population."),
            "membership_weighting": (
                "membership is a property of the EVENT and is identical under "
                "eq and notional weighting; the weighting changes only how "
                "members are averaged. So the two legs' weightings (eq for "
                "the void, notional for the settle) do not move who is in."),
            "counted_as_a_status": [
                "n_sweeps_true_valid", "n_sweeps_proxy_valid",
                "n_sweeps_intersection",
                "n_true_valid_without_proxy (the events the proxy could not "
                "see -- reported as a count AND a share, rule 4)",
                "the same counts for E1's own window via the reproduction "
                "control, so the two windows' proxy coverage is comparable",
            ],
            "what_a_population_difference_would_do": (
                "Delta_rs on the intersection licenses a statement about the "
                "MID on the events both mids see. It does NOT license "
                "transporting that statement to the settle population, which "
                "is larger. If n_true_valid_without_proxy is a large share, "
                "then E1's proxy was blind to that share of its own window "
                "and the VOID leg's reach is correspondingly narrower -- that "
                "is a limit on the void reading, reported with it, never a "
                "reason to widen either population after seeing."),
        },

        "the_gate_quantity": {
            "primary": "notional-weighted rs(tau*) on TRUE mids, per symbol",
            "rs_definition": "rs(tau) == mean MO(tau) (EXPERIMENT_PLAN "
                             "section 1.3c: es = rs + Lambda, so MO already IS "
                             "rs; the 'ES - 2|MO|' form double-counts)",
            "weights": "Q * p, the USD notional of the sweep",
            "why_notional": (
                "economics are per DOLLAR; an eq-weighted markout is a "
                "per-EVENT statistic. E1's own audit measured the gap at "
                "eq - notional ~ 2.8 bps on ADA, twenty times ADA's 0.14 bps "
                "margin over fee."),
            "also_reported": ["eq-weighted (so Delta_rs is comparable to what "
                              "E1 gated on)", "per side (maker-bid / "
                              "maker-ask)", "Lambda(tau) and es",
                             "size-bucketed rs by notional quintile "
                              "(amendment item ii says 'and size-bucketed')"],
            "tau_grid_s": list(TAUS_S),
            "tau_star_rule": (
                f"tau* = {TAU_STAR_DEFAULT_S} s if the day-median "
                "time-to-next-opposite-sweep <= 30 s on at least "
                f"ceil({TAU_STAR_DAY_FRACTION:.6f} * G) days (the plan's "
                "'>= 24 of 31', re-expressed as its own fraction for a "
                f"G-day window; G=16 -> {days_threshold(16, TAU_STAR_DAY_FRACTION)}); "
                "else min(300, 2 x median) rounded up to {60, 300}. Stated ex "
                "ante to forbid horizon shopping."),
            "identity_check": (
                f"|mean es - mean Lambda(tau) - mean MO(tau)| > "
                f"{IDENTITY_TOL_BPS} bps on a (symbol, day) flags that day "
                "proxy_incoherent and EXCLUDES it from gates; the excluded "
                "fraction is reported with every table (rule 4)."),
        },

        "what_settles_and_what_voids": {
            "leg_i_VOID": {
                "quantity": "Delta_rs(tau*) = rs_proxy - rs_true, EQ-WEIGHTED, "
                            "both computed on the SAME admissible days over "
                            "the INTERSECTION population defined below",
                "threshold_bps": VOID_THRESHOLD_BPS,
                "consequence": "Delta_rs > +1.0 bps VOIDS E1's ADA pass",
                "source": "EXPERIMENT_PLAN.md section 2, E2.0",
            },
            "leg_ii_SETTLE": {
                "quantity": "notional-weighted rs(tau*) on true mids",
                "threshold_bps": FEE_MAKER_VIP0,
                "consequence": "below the VIP0 maker fee KILLS the ADA cell "
                               "regardless of eq numbers",
                "source": "E1_CODE_REVIEW.md pre-read amendment item (ii)",
            },
            "two_bars_not_one": {
                "why_this_field_exists": (
                    "REVIEWER FINDING 2 (46030b1): v1's table called anything "
                    "at or above 1.8 SETTLED_ALIVE. 1.8 is the DEATH bar (the "
                    "amendment's kill: below the VIP0 maker fee). The PLAN's "
                    "own pass bar is fee + c_safe = 2.3 (EXPERIMENT_PLAN "
                    "section 1.5 gate 1). A cell at 2.0 would have read ALIVE "
                    "while FAILING gate 1. One boolean was doing duty for two "
                    "bars."),
                "death_bar_bps": FEE_MAKER_VIP0,
                "gate1_bar_bps": FEE_MAKER_VIP0 + C_SAFE,
                "gate2_rule": "the plan's gate 2: block-bootstrap 95% CI "
                              "lower bound >= fee_maker (no margin)",
            },
            "interval_is_decision_bearing": {
                "why_this_field_exists": (
                    "REVIEWER FINDING 3 (46030b1): v1 compared a POINT to a "
                    "threshold while declaring an interval that G = 16 makes "
                    "claimable. A point that clears with an interval that "
                    "does not is not a pass."),
                "on_the_alive_side": "ALIVE requires the point >= 2.3 AND the "
                                     "95% CI lower bound >= 1.8. Otherwise "
                                     "INCONCLUSIVE.",
                "on_the_kill_side": "the kill stays the amendment's POINT "
                                    "rule, because weakening a pre-registered "
                                    "kill after seeing is exactly what rule "
                                    "11 forbids. `kill_is_interval_robust` "
                                    "(CI upper bound < 1.8) is REPORTED "
                                    "beside it, never instead of it.",
                "below_G_5": "no interval is claimable, so ALIVE cannot be "
                             "asserted at all: the state is "
                             "INCONCLUSIVE_NO_INTERVAL (CLAUDE.md rule 8)",
            },
            "outcome_table": {
                "settle_states": {
                    "DEAD": "notional rs_true < 1.8 (the death bar)",
                    "NOT_KILLED_PENDING_GATE_1":
                        "1.8 <= notional rs_true < 2.3 -- neither dead nor "
                        "passing; the band v1 mislabelled ALIVE",
                    "INCONCLUSIVE_NO_INTERVAL":
                        "point >= 2.3 but G < 5, no interval claimable",
                    "INCONCLUSIVE_INTERVAL_DOES_NOT_CLEAR":
                        "point >= 2.3 but CI-lo < 1.8 (fails gate 2)",
                    "ALIVE_CLEARS_GATE_1_AND_2":
                        "point >= 2.3 AND CI-lo >= 1.8",
                },
                "void_states": {"VOIDED": "Delta_rs > 1.0",
                                "NOT_VOIDED": "Delta_rs <= 1.0"},
                "verdict": "the two are reported as one string "
                           "'<void>+<settle>' and as separate fields; "
                           "UNDECIDABLE if either leg has no admissible "
                           "estimate -- a STATUS, never resolved by "
                           "assumption",
            },
            "the_expectation_is_recorded_so_it_can_be_wrong": (
                "the E1 results audit expects SETTLED_DEAD. Recording the "
                "expectation is not a prediction the result must meet: "
                "SETTLED_ALIVE is a live outcome and carries the heavier "
                "burden, which is why the full section 1.5 gate is computed "
                "for every symbol whatever the verdict."),
            "what_this_CANNOT_settle": [
                "the window is 2026-08-20..09-05, NOT E1's 07-18..08-17. A "
                "dead cell here is dead ON THIS PERIOD; it does not "
                "retroactively re-measure E1's window, and a live cell here "
                "would not either.",
                "rs is still a POPULATION statistic over all sweeps -- hazard "
                "H1 (population vs marginal) is measured by the notional "
                "weighting, not removed by it. Only E2's queue replay puts a "
                "marginal order at the back of the queue.",
                "no queue position, no own impact, no fees beyond the maker "
                "fee comparison. E2.0 cannot establish economic sign; it can "
                "only kill or fail to kill.",
            ],
        },

        "falsifiers": {
            "both_directions_required": True,
            "positive_controls": [
                "a synthetic book where the mid always moves IN the maker's "
                "favour by a known number of bps must return rs = that number "
                "(both maker-bid and maker-ask events)",
                "a synthetic population of many small POSITIVE events and few "
                "large NEGATIVE ones must give eq > 0 AND notional < 0 -- the "
                "H1 shape itself, so the weighting code is shown to "
                "discriminate rather than merely to run",
                "a clean day with no gaps must be ADMITTED",
                "the identity mean es - mean Lambda == mean MO must hold "
                "exactly on synthetic data built from a known mid path",
            ],
            "known_bads": [
                "a synthetic book where the mid always moves AGAINST the "
                "maker must return rs NEGATIVE at the known magnitude",
                "a day with an injected gap above the threshold must be "
                "EXCLUDED with a named status and counted, never dropped",
                "a sweep whose t+tau falls past the end of available book "
                "must be EXCLUDED and counted, never valued at the last "
                "known mid",
                "reversing the is_buyer_maker convention must FLIP the sign "
                "of rs -- if it does not, the sign convention is not wired",
                "the runner must REFUSE if this declaration's sha256 differs "
                "from the one it was built against",
                "the runner must REFUSE a symbol with fewer than "
                f"{MIN_COMPLETE_DAYS} admissible days",
                "the runner must REFUSE if the E1 reproduction control misses "
                "E1's published ADA numbers by more than "
                f"{E1_ADA_REPRODUCTION_TARGET['tolerance_bps']} bps",
            ],
        },

        "e1_reproduction_control": E1_ADA_REPRODUCTION_TARGET,

        "robustness_splits_declared_now": {
            "post_era_boundary": (
                f"every gate recomputed on days entirely at or after "
                f"{ERA_BOUNDARY_NS} (2026-08-24T13:48:54Z). Declared BEFORE "
                "the run so it cannot become a rescue: the gate quantities do "
                "not use recv_ns, and this split is the check on that claim. "
                "Expected admissible post-boundary days: 08-25 and "
                "08-27..09-05 = 11, above the G>=5 floor."),
            "per_side": "maker-bid and maker-ask separately (section 1.5 gate 4)",
            "size_buckets": "notional quintiles, per the amendment's "
                            "'size-bucketed'",
            "tau_sensitivity": "the full tau grid is reported; the GATE is "
                               "read only at tau*",
        },

        "statistics": {
            "day_clustered_mean": "gates 1 and 4 are day-clustered means over "
                                  "admissible days",
            "interval": "stationary block bootstrap of the ratio-of-sums "
                        f"(B={BOOT_B}, seed={BOOT_SEED}, 30-min bins, 4h "
                        "blocks) per EXPERIMENT_PLAN section 1.4",
            "interval_floor": "CLAUDE.md rule 8: below G=5 complete days, "
                              "point estimate and NO interval, said so",
            "days_positive_threshold": (
                f"gate 3 is '> 0 on >= {DAYS_POSITIVE_FRACTION:.0%} of days'; "
                f"for G=16 that is {days_threshold(16, DAYS_POSITIVE_FRACTION)}"),
        },

        "resources": {
            "cap": "one CPU, MemoryMax=8G, under the rule-20 wrapper "
                   "(flock + systemd-run --scope --slice=research.slice). "
                   "NEVER RAISED: a symbol that exceeds the cap REFUSES, the "
                   "cap does not rise.",
            "measured_input_sizes_as_of_2026-09-06T04:21Z": {
                "ADAUSDT bookTicker per day (gz)": "41 MB",
                "ADAUSDT trade per day (gz)": "4.3 MB",
                "BTCUSDT bookTicker per day (gz)": "480 MB",
                "BTCUSDT trade per day (gz)": "39 MB",
                "note": "BTC is ~12x ADA. This is why the smoke is ONE symbol "
                        "with its resource observation before any fan-out, "
                        "and why the reader is day-at-a-time and streaming.",
            },
            "smoke": "ADAUSDT alone, all admissible days, RSS and wall "
                     "recorded, BEFORE any other symbol runs. ADA is also the "
                     "cell this step exists to settle.",
            "estimate_is_labelled_an_estimate": (
                "no wall-clock estimate is offered before the smoke measures "
                "one. The smoke IS the estimate."),
        },

        "what_would_refute_this_DESIGN_rather_than_a_symbol": [
            "the E1 reproduction control failing -- then the proxy leg is a "
            "different estimator and Delta_rs is not the pre-registered "
            "quantity",
            "the true-mid quote age being routinely large (a book that is not "
            "actually continuous in our capture) -- then 'true mid' is a "
            "second proxy and must be called one",
            "a systematically empty admissible-day set -- that refutes the "
            "population predicate, not the symbols",
            "the identity check failing on true mids across most symbol-days "
            "-- es = rs + Lambda is arithmetic, so a broad failure means the "
            "reader is wrong, not the market",
        ],
    }


def selftest() -> int:
    fails: list[str] = []

    def ok(c, m):
        print(("ok   " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    # --- the VOID leg, both directions ---
    ok(void_predicate(1.5)["voids"] is True,
       "VOID leg KNOWN-BAD: Delta_rs = +1.5 bps > 1.0 VOIDS the E1 pass")
    ok(void_predicate(0.9)["voids"] is False,
       "VOID leg POSITIVE CONTROL: Delta_rs = +0.9 bps does NOT void -- a "
       "rule that voided everything would settle nothing")
    ok(void_predicate(1.0)["voids"] is False,
       "VOID leg BOUNDARY: exactly +1.0 does NOT void (the plan says '> 1.0')")
    ok(void_predicate(-3.0)["voids"] is False,
       "VOID leg: a NEGATIVE Delta_rs (proxy pessimistic) does not void -- "
       "the rule is one-sided by design and that is stated, not assumed")
    ok(void_predicate(None)["decidable"] is False,
       "VOID leg: an undefined Delta_rs is UNDECIDABLE, never a pass")

    # --- the SETTLE leg: THREE bars, driven at every one (reviewer 2 and 3) ---
    ok(settle_predicate(-0.32, ci_lo_bps=-1.0, ci_hi_bps=0.4)["state"] == "DEAD",
       "SETTLE KNOWN-BAD: E1's own notional ADA number (-0.322) is below the "
       "1.8 bps death bar -> DEAD")
    ok(settle_predicate(-0.32, ci_lo_bps=-1.0,
                        ci_hi_bps=0.4)["kill_is_interval_robust"] is True,
       "SETTLE: a kill whose CI upper bound is also below the fee is reported "
       "INTERVAL-ROBUST -- beside the point rule, never instead of it")
    ok(settle_predicate(1.0, ci_lo_bps=-1.0,
                        ci_hi_bps=3.0)["kill_is_interval_robust"] is False,
       "SETTLE: a kill whose interval STRADDLES the fee is reported NOT "
       "interval-robust -- and still kills, because weakening a "
       "pre-registered kill after seeing is what rule 11 forbids")
    ok(settle_predicate(2.0, ci_lo_bps=1.9, ci_hi_bps=2.1)["state"]
       == "NOT_KILLED_PENDING_GATE_1",
       "SETTLE KNOWN-BAD (reviewer finding 2): 2.0 bps sits BETWEEN the 1.8 "
       "death bar and the plan's 2.3 gate -- v1 called this ALIVE, which "
       "would have labelled a cell that FAILS section 1.5 gate 1 as surviving")
    ok(settle_predicate(2.5, ci_lo_bps=1.0, ci_hi_bps=4.0)["state"]
       == "INCONCLUSIVE_INTERVAL_DOES_NOT_CLEAR",
       "SETTLE KNOWN-BAD (reviewer finding 3): a POINT of 2.5 clears the gate "
       "but a CI lower bound of 1.0 fails gate 2 -> INCONCLUSIVE, not alive")
    ok(settle_predicate(2.5, ci_lo_bps=2.0, ci_hi_bps=3.0,
                        interval_claimable=False)["state"]
       == "INCONCLUSIVE_NO_INTERVAL",
       "SETTLE: below G = 5 no interval is claimable, so ALIVE cannot be "
       "asserted from a point at all (CLAUDE.md rule 8)")
    ok(settle_predicate(2.5, ci_lo_bps=2.0, ci_hi_bps=3.0)["state"]
       == "ALIVE_CLEARS_GATE_1_AND_2",
       "SETTLE POSITIVE CONTROL: point 2.5 >= 2.3 AND CI-lo 2.0 >= 1.8 is the "
       "ONLY way to ALIVE -- the rule can admit, so it is not a kill wearing "
       "a gate's name")
    ok(settle_predicate(1.8, ci_lo_bps=1.8, ci_hi_bps=2.0)["dies"] is False
       and settle_predicate(1.8, ci_lo_bps=1.8, ci_hi_bps=2.0)["state"]
       == "NOT_KILLED_PENDING_GATE_1",
       "SETTLE BOUNDARY: exactly at the death bar does NOT die and does NOT "
       "pass -- the two bars are 1.8 and 2.3 and neither is the other")
    ok(settle_predicate(1.5, "vip1", ci_lo_bps=1.5, ci_hi_bps=1.6)["dies"]
       is False
       and settle_predicate(1.5, "vip0", ci_lo_bps=1.5,
                            ci_hi_bps=1.6)["dies"] is True,
       "SETTLE: the tier is a parameter and changes the answer at 1.5 bps -- "
       "VIP0 kills, VIP1 does not")
    ok(settle_predicate(None)["decidable"] is False,
       "SETTLE: no admissible estimate is UNDECIDABLE, never a kill")

    # --- the joint verdict: every settle state, both void states ---
    seen = set()
    for dr in (1.5, 0.5):
        for rs, lo, hi in ((-0.3, -1.0, 0.4), (2.0, 1.9, 2.1),
                           (2.5, 1.0, 4.0), (2.5, 2.0, 3.0)):
            seen.add(ada_verdict(void_predicate(dr),
                                 settle_predicate(rs, ci_lo_bps=lo,
                                                  ci_hi_bps=hi))["verdict"])
    ok(len(seen) == 8,
       f"THE JOINT TABLE IS TOTAL: 2 void states x 4 reachable settle states "
       f"= {len(seen)} distinct verdicts, every one reachable")
    ok(ada_verdict(void_predicate(None),
                   settle_predicate(2.0, ci_lo_bps=1.9,
                                    ci_hi_bps=2.1))["verdict"] == "UNDECIDABLE",
       "JOINT: an undecidable leg yields UNDECIDABLE, not a default pass and "
       "not a default kill")
    vj = ada_verdict(void_predicate(1.5),
                     settle_predicate(2.0, ci_lo_bps=1.9, ci_hi_bps=2.1))
    ok(vj["cell_dies"] is False and vj["cell_passes_gate1"] is False
       and vj["settle_state"] == "NOT_KILLED_PENDING_GATE_1",
       "JOINT: `cell_dies` and `cell_passes_gate1` are SEPARATE fields and "
       "both are False in the 1.8-2.3 band -- no single boolean carries two "
       "bars, which is the whole of reviewer finding 2")

    # --- day-count re-expression, not re-choosing ---
    ok(days_threshold(31, DAYS_POSITIVE_FRACTION) == 22,
       "DAY COUNTS: the plan's own '>= 22 of 31' is reproduced by the "
       "fraction, so the G=16 threshold is a re-expression and not a new "
       f"choice (G=16 -> {days_threshold(16, DAYS_POSITIVE_FRACTION)})")
    ok(days_threshold(31, TAU_STAR_DAY_FRACTION) == 24,
       "DAY COUNTS: the plan's own tau* '>= 24 of 31' likewise reproduces "
       f"(G=16 -> {days_threshold(16, TAU_STAR_DAY_FRACTION)})")

    # --- the declaration cannot read data ---
    # CHECKED AT THE AST, NOT BY SUBSTRING. My first version scanned this
    # file's own source for "gzip"/"pandas"/"data/mm_hf/raw" -- and those
    # strings are IN THE CHECK ITSELF, so it could never pass: a control that
    # cannot fire, in the mirror image of the one SEAT_PROTOCOL 16 names. The
    # property is about IMPORTS and CALLS, so it is read from the syntax tree.
    import ast as _ast
    tree = _ast.parse(Path(__file__).read_text())
    imported: set[str] = set()
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, _ast.ImportFrom):
            if n.level == 0 and n.module:
                imported.add(n.module.split(".")[0])
    SAFE = {"__future__", "argparse", "hashlib", "json", "math",
            "subprocess", "pathlib", "ast"}
    ok(imported <= SAFE,
       f"NO READER IMPORTED: this module's imports are {sorted(imported)}, a "
       f"subset of the declared safe set -- no gzip, csv, pandas or pyarrow, "
       f"so it CANNOT open the tape whatever its prose says")
    io_calls = sorted({
        n.func.attr for n in _ast.walk(tree)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
        and n.func.attr in {"open", "read_text", "read_bytes", "write_text",
                            "read_csv", "read_parquet", "glob", "iterdir"}
    } | {n.func.id for n in _ast.walk(tree)
         if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)
         and n.func.id == "open"})
    ok(io_calls == ["read_bytes", "read_text", "write_text"],
       f"NO TAPE ACCESS: every file operation in this module is enumerated "
       f"from the AST and is one of {io_calls} -- read_bytes digests the "
       f"emitted declaration, read_text reads THIS file for the checks above, "
       f"write_text writes the declaration. No open(), no glob, no reader.")

    d = declaration()
    tbl = set(d["what_settles_and_what_voids"]["outcome_table"]["settle_states"])
    produced = {settle_predicate(x, ci_lo_bps=lo, ci_hi_bps=hi,
                                 interval_claimable=ic)["state"]
                for x, lo, hi, ic in ((-0.3, -1.0, 0.4, True),
                                      (2.0, 1.9, 2.1, True),
                                      (2.5, 1.0, 4.0, True),
                                      (2.5, 2.0, 3.0, False),
                                      (2.5, 2.0, 3.0, True))}
    ok(d["status"] == "DECLARATION_NO_DATA_TOUCHED" and tbl == produced,
       f"THE EMITTED OBJECT and the CODE cannot disagree: the declared "
       f"settle_states are exactly the states the predicate produces "
       f"({sorted(tbl)})")
    ok(d["supersedes"]["sha256"] == SUPERSEDES["sha256"]
       and len(d["supersedes"]["what_changed"]) == 3,
       "RULE 13: v2 carries a supersedes block naming v1 by path and sha256, "
       "with one entry per reviewer finding; v1 is not edited")
    ok("which_population_each_leg_is_computed_over" in d
       and "INTERSECTION" in d["which_population_each_leg_is_computed_over"]
       ["VOID_leg_population"],
       "REVIEWER FINDING 1 is a FIELD: the void leg's population is the "
       "intersection, the settle leg's is all true-mid-valid events, and the "
       "difference is counted as a status")
    ok(len(d["falsifiers"]["known_bads"]) >= 5
       and len(d["falsifiers"]["positive_controls"]) >= 3,
       f"FALSIFIERS BOTH DIRECTIONS declared: "
       f"{len(d['falsifiers']['positive_controls'])} positive controls, "
       f"{len(d['falsifiers']['known_bads'])} known-bads")
    ok(d["what_settles_and_what_voids"]["what_this_CANNOT_settle"],
       "THE LIMITS ARE IN THE DECLARATION, not left for the reader: the "
       "window is not E1's, H1 is measured not removed, no queue position")

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
