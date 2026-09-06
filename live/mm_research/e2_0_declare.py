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
OUT = HERE / "declarations" / "p002_e2_0_declaration_v1.json"
PROTOCOL = "P002_E2_0_TRUE_MID_DECLARATION_V1"

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
                     tier: str = "vip0") -> dict:
    """(ii) The pre-read amendment: notional-weighted rs on TRUE mids.

    "if notional-weighted rs(tau*) < fee at VIP0 on the L2 window, the ADA cell
    dies regardless of eq numbers" -- E1_CODE_REVIEW.md.
    """
    fee = FEE_MAKER_VIP0 if tier == "vip0" else FEE_MAKER_VIP1
    if rs_true_notional_bps is None:
        return {"decidable": False, "dies": None, "fee_bps": fee,
                "why": "no admissible notional-weighted estimate exists"}
    dies = rs_true_notional_bps < fee
    return {"decidable": True, "dies": bool(dies), "fee_bps": fee,
            "rs_true_notional_bps": rs_true_notional_bps,
            "clears_gate1_margin": bool(rs_true_notional_bps >= fee + C_SAFE),
            "why": ("per DOLLAR of maker fill the realized half-spread does "
                    "not cover the VIP0 maker fee, which is the whole of "
                    "standalone viability" if dies else
                    "the dollars clear the fee; the cell survives this leg "
                    "and must then meet the full section 1.5 gate")}


def ada_verdict(void: dict, settle: dict) -> dict:
    """The 2x2. Deliberately has no 'unknown' cell when both legs are decidable.

    The two legs are INDEPENDENT and both can fire: one asks whether E1's
    measurement is readable, the other whether the right quantity pays.
    """
    if not void["decidable"] or not settle["decidable"]:
        return {"verdict": "UNDECIDABLE",
                "why": "at least one leg had no admissible estimate; reported "
                       "as a status, never resolved by assumption (rule 4)"}
    v, d = void["voids"], settle["dies"]
    if v and d:
        return {"verdict": "VOID_AND_DEAD", "voids_e1_pass": True,
                "cell_dies": True,
                "why": "E1's ADA number is not readable AND the true-mid "
                       "dollars are below fee: the pass was an artifact and "
                       "the cell is dead on the new window"}
    if v and not d:
        return {"verdict": "VOID_ONLY", "voids_e1_pass": True,
                "cell_dies": False,
                "why": "E1's ADA number is not readable, but the true-mid "
                       "dollars clear the fee: the cell is RE-OPENED on true "
                       "mids and must meet the full section 1.5 gate. It is "
                       "not a pass and must never be reported as one"}
    if (not v) and d:
        return {"verdict": "SETTLED_DEAD", "voids_e1_pass": False,
                "cell_dies": True,
                "why": "the proxy mid was within tolerance -- E1 measured "
                       "correctly -- and the dollars are still below fee. The "
                       "cell is settled dead, which is the audit's expectation"}
    return {"verdict": "SETTLED_ALIVE", "voids_e1_pass": False,
            "cell_dies": False,
            "why": "proxy within tolerance and the dollars clear the fee: ADA "
                   "survives E2.0 and proceeds to the full section 1.5 gate "
                   "and E2. This is the outcome the audit did NOT expect, so "
                   "it carries the heavier burden of proof"}


# --------------------------------------------------------------------------
def declaration(n_days_expected: int = 16) -> dict:
    return {
        "protocol": PROTOCOL,
        "status": "DECLARATION_NO_DATA_TOUCHED",
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
                            "both computed on the SAME admissible days from "
                            "the SAME sweep events",
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
            "outcome_table": {
                "VOID_AND_DEAD": "Delta_rs > 1.0 AND notional rs_true < 1.8",
                "VOID_ONLY": "Delta_rs > 1.0 AND notional rs_true >= 1.8 -- "
                             "cell RE-OPENED, not passed",
                "SETTLED_DEAD": "Delta_rs <= 1.0 AND notional rs_true < 1.8",
                "SETTLED_ALIVE": "Delta_rs <= 1.0 AND notional rs_true >= 1.8",
                "UNDECIDABLE": "either leg has no admissible estimate -- a "
                               "STATUS, never resolved by assumption",
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

    # --- the SETTLE leg, both directions ---
    ok(settle_predicate(-0.32)["dies"] is True,
       "SETTLE leg KNOWN-BAD: E1's own notional ADA number (-0.322) is below "
       "the 1.8 bps VIP0 fee and KILLS the cell")
    ok(settle_predicate(2.5)["dies"] is False
       and settle_predicate(2.5)["clears_gate1_margin"] is True,
       "SETTLE leg POSITIVE CONTROL: 2.5 bps clears the fee AND the "
       "fee + c_safe margin -- the rule can admit")
    ok(settle_predicate(1.8)["dies"] is False
       and settle_predicate(1.8)["clears_gate1_margin"] is False,
       "SETTLE leg BOUNDARY: exactly at the fee does not die but does NOT "
       "clear gate 1's margin -- two different thresholds, not collapsed")
    ok(settle_predicate(1.5, "vip1")["dies"] is False
       and settle_predicate(1.5, "vip0")["dies"] is True,
       "SETTLE leg: the tier is a parameter and changes the answer at 1.5 "
       "bps -- VIP0 kills, VIP1 does not (a 'conditional pass, tier-gated')")
    ok(settle_predicate(None)["decidable"] is False,
       "SETTLE leg: no admissible estimate is UNDECIDABLE, never a kill")

    # --- the 2x2, every cell reachable ---
    cells = {}
    for dr in (1.5, 0.5):
        for rs in (-0.3, 2.5):
            v = ada_verdict(void_predicate(dr), settle_predicate(rs))
            cells[(dr > 1.0, rs < 1.8)] = v["verdict"]
    ok(cells == {(True, True): "VOID_AND_DEAD", (True, False): "VOID_ONLY",
                 (False, True): "SETTLED_DEAD", (False, False): "SETTLED_ALIVE"},
       f"THE 2x2 IS TOTAL: all four cells reachable and distinct -- {cells}")
    ok(ada_verdict(void_predicate(None), settle_predicate(2.0))["verdict"]
       == "UNDECIDABLE",
       "THE 2x2: an undecidable leg yields UNDECIDABLE, not a default pass "
       "and not a default kill")
    ok(ada_verdict(void_predicate(1.5), settle_predicate(2.5))["cell_dies"]
       is False
       and ada_verdict(void_predicate(1.5),
                       settle_predicate(2.5))["voids_e1_pass"] is True,
       "VOID_ONLY is NOT a pass: the cell is re-opened, and the verdict says "
       "so in the same field a reader would key on")

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
    ok(d["status"] == "DECLARATION_NO_DATA_TOUCHED"
       and set(d["what_settles_and_what_voids"]["outcome_table"]) ==
       {"VOID_AND_DEAD", "VOID_ONLY", "SETTLED_DEAD", "SETTLED_ALIVE",
        "UNDECIDABLE"},
       "THE EMITTED OBJECT carries the same five outcomes the predicates "
       "produce -- the table and the code cannot disagree")
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
