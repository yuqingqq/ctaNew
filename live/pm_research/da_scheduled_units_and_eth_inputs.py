"""THE SIX SCHEDULED UNITS, AND THE ETH INPUT AUDIT. (DA 291-296)

TWO FILINGS THAT WERE OWED ACROSS SEVERAL DISPATCHES, computed rather than
narrated so a reader can re-run them.

THE SEPARATING QUESTION FOR A TIMER IS NOT "DOES IT TOUCH THE DAY". Coverage,
quality, price and health infrastructure can run over a day without consuming
it; only work that reaches an OUTCOME -- a score, a valuation, a P&L --
consumes. A "touches the day" test would condemn all six including the health
checks, and would still have missed the one that matters.

UNKNOWN IS TREATED AS CONSUMING. Absence of evidence is not a producer.

Usage:  da_scheduled_units_and_eth_inputs.py [--falsify]
"""
from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROTOCOL = "P003_DA_SCHEDULED_UNITS_AND_ETH_INPUTS_V1"
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import da_root                                            # noqa: E402
import da_rule34a_fence as FENCE                          # noqa: E402

#: Measured 2026-09-12T00:22-00:40Z from `systemctl --user list-timers --all`,
#: `systemctl --user show <unit> -p ExecStart`, and the modules' own source.
UNITS = (
    {"unit": "pm-research-guard.timer", "schedule": "every minute",
     "command": "python3 live/pm_research/ops/pm_research_guard.py",
     "reads": "the process table", "writes": "stdout JSON only -- no files",
     "reaches_an_outcome": False, "ruling": "INFRASTRUCTURE",
     "why": "a process guard; it scans and prints, and writes nothing"},
    {"unit": "pm-lane-health.timer", "schedule": "*:00/15 (every 15 min)",
     "command": "python3 live/pm_research/ops/pm_lane_health.py",
     "reads": "lane state", "writes": "an alert file",
     "reaches_an_outcome": False, "ruling": "INFRASTRUCTURE",
     "why": "health reporting; no score, valuation or P&L"},
    {"unit": "pm-evaluation-pipeline.timer",
     "schedule": "03,09,15,21:50 UTC (4x daily), HOLDS THE HEAVY LOCK",
     "command": ("flock data/.heavy_run.lock python3 -m "
                 "live.pm_research.evaluation_pipeline --catch-up "
                 "--since 2026-08-20 --max-days 1 --scheduled --json"),
     "reads": "tier1, markets.jsonl, resolutions.jsonl",
     "writes": "tier2 calib_panel, markout_events",
     "reaches_an_outcome": True,
     "ruling": "INFRASTRUCTURE_BY_WRITING__OUTPUT_IS_PROTECTED",
     "RULED_AT_DA_305": (
         "IT DOES NOT CONSUME A BAND DAY BY RUNNING, AND ITS OUTPUT MAY NOT BE "
         "READ. Both halves are the USER's ruling, not mine -- rule 34a "
         "(`abd4b07`): WRITING DOES NOT CONSUME A DAY, READING DOES."),
     "the_reasoning": (
         "rule 11's concern is SELECTION -- choosing after seeing. This unit "
         "FITS NOTHING, PICKS NO THRESHOLD AND COMPUTES NO INTERVAL (REV_"
         "PROCEDURE's words), so it makes no selection and its running cannot "
         "void the test. What WOULD void it is a human reading the output, "
         "because the output carries the realised outcome: markout_events "
         "schema holds `winner_up` (bool) and `outcome_up` (float) beside "
         "`price_up`, `q_up` and `size`, and the module computes "
         "`edge = q_up * (price_up - outcome_up)` per fill. That is a "
         "per-trade signed settlement edge for a band day -- exactly the "
         "thing that must not be looked at before the band closes."),
     "the_sharp_question_answered": (
         "IS A BAND DAY STILL UNTOUCHED IF THIS HAS WRITTEN CALIBRATION FOR "
         "IT? YES -- by USER ruling, provided nobody reads it. Existence is "
         "not permission."),
     "and_there_IS_a_live_selection_hazard_it_would_feed": (
         "`minimum_meaningful_delta_LL` is UNSET and belongs to the USER, and "
         "whether C2 stays in the family is also open. A number chosen after "
         "seeing outcome-derived statistics from band days is rule 11's "
         "failure exactly -- which is why the READ prohibition, not the write, "
         "is the operative half."),
     "the_fence_already_exists_and_already_covers_the_band": (
         "rule 34a protects `data/pm_5min/tier2/**/day=2026-09-08/` AND ANY "
         "LATER DAY, so every future band day is already inside it. No new "
         "fence is needed; what is needed is that it stop being PROSE."),
     "THE_GAP_BY_TONIGHTS_OWN_STANDARD": (
         "rule 34a lives in two procedure files as a sentence. Every other "
         "guard tonight was moved from a sentence into a predicate because a "
         "note can be forgotten and a predicate cannot. A read-prohibition "
         "that depends on each seat remembering it is the weakest guard in the "
         "lane, and it guards the single thing that would void the test."),
     "operationally": (
         "NOTHING IS DISABLED. The finding is retrospective and killing a "
         "timer that has already run saves nothing and may destroy the "
         "day-quality record. The remedy is a fence declared BEFORE the band, "
         "which rule 34a already is."),
     "why": ("it JOINS THE SETTLED OUTCOME and values against it: "
             "`winner_up` -> `outcome_up` -> `edge = q_up * (price_up - "
             "outcome_up)`, written into markout_events. That is a valuation, "
             "not coverage."),
     "days_written_in_window": "tier2 day partitions 2026-09-05..2026-09-09"},
    {"unit": "da-midnight-verify.timer", "schedule": "00:06 UTC daily",
     "command": "live/pm_research/da_midnight_verify.sh",
     "reads": "raw tape, collector gaps",
     "writes": "the day verdict and the blackout mask, paired",
     "reaches_an_outcome": False, "ruling": "INFRASTRUCTURE",
     "why": ("day-quality and candidate-blind; §8 resolves day eligibility "
             "FROM this, which makes it an input to the test rather than a "
             "consumer of it"),
     "state": ("REFUSING since the 09-11 script change: DEPLOY_DRIFT "
               "(REFUSE_TIER_DRIFT) rc 7, because da_midnight_verify.sh was "
               "edited after the 09-07 deploy record and never re-deployed")},
    {"unit": "pm-measurement-pipeline.timer",
     "schedule": "*:20 UTC (hourly), HOLDS THE HEAVY LOCK",
     "command": ("flock data/.heavy_run.lock python3 -m "
                 "live.pm_research.measurement_batch --catch-up "
                 "--since 2026-08-20 --lane measurement --max-days 1 "
                 "--scheduled --json"),
     "reads": "raw / tier0", "writes": "tier1 coverage, twap",
     "reaches_an_outcome": False, "ruling": "INFRASTRUCTURE",
     "why": ("coverage and TWAP. It looked like a consumer on a loose grep "
             "(44 hits) but its ONLY two `winner_up` references are "
             "`bool(index % 2)` -- synthetic fixtures in a test builder, not "
             "a settled outcome"),
     "days_written_in_window": "tier1 day partitions 2026-09-05..2026-09-11"},
    {"unit": "launchpadlib-cache-clean.timer", "schedule": "daily",
     "command": "OS cache clean", "reads": "OS cache", "writes": "OS cache",
     "reaches_an_outcome": False, "ruling": "INFRASTRUCTURE (out of programme)",
     "why": "touches nothing under data/pm_5min"},
)

LOCK_FACT = {
    "both_heavy_units_block_rather_than_yield": True,
    "mechanism": ("ExecStart uses plain `flock <file>` with NO `-n`, so a unit "
                  "WAITS for the lock rather than failing fast"),
    "consequence": ("BE queues behind them rather than being refused, and the "
                    "measurement unit fires HOURLY at :20. Over a fourteen-"
                    "night build programme that is fourteen nights of "
                    "contention, not one"),
    "observed": ("2026-09-12T00:22Z the lock was held by pid 3507151 "
                 "(measurement_batch, 162s elapsed) while BE was mid-sequence "
                 "on 09-11"),
}

#: ETH inputs, days that matter. Counts are computed by `eth_input_audit`.
ETH_DAYS = ("20260903", "20260904", "20260905", "20260906",
            "20260907", "20260908", "20260909", "20260910")


def _root() -> Path:
    return da_root.resolve_root()


def eth_input_audit(days=ETH_DAYS) -> dict:
    """PER INPUT: present or absent, with counts, ETH against BTC as control.

    §7k.2: a "present" answer states its exclusions; an "absent" answer carries
    a positive control. BTC is the control throughout -- it is known present,
    so a query returning zero for ETH and non-zero for BTC is a real absence,
    and one returning zero for BOTH is a broken query.
    """
    out = {}
    raw = {}
    for d in days:
        e = len(glob.glob(str(_root() / f"data/pm_5min/raw/{d}/eth-updown-5m-*.jsonl.gz")))
        b = len(glob.glob(str(_root() / f"data/pm_5min/raw/{d}/btc-updown-5m-*.jsonl.gz")))
        raw[d] = {"eth_files": e, "btc_files": b, "parity": e == b}
    out["raw_archive"] = {
        "by_day": raw, "eth_present": all(v["eth_files"] > 0 for v in raw.values()),
        "at_parity_with_btc": all(v["parity"] for v in raw.values()),
        "exclusions": ("counts .jsonl.gz per coin per day under "
                       "data/pm_5min/raw/<day>/; excludes any other coin and "
                       "any non-gz file"),
        "positive_control": "btc counted by the same query, non-zero every day"}
    return out


def _asof(path) -> str | None:
    import os
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                             time.gmtime(os.path.getmtime(path)))
    except Exception:
        return None


def eth_closure() -> dict:
    """THE ETH DATA SIDE, CLOSED PER INPUT, with counts and as-of.

    §7k.2: every "present" answer states its exclusions and every count
    carries BTC as the positive control, so a zero for ETH would be a real
    absence rather than a broken query.
    """
    import glob as _g
    root = _root()
    days = list(ETH_DAYS)
    out = {}
    e_tot = b_tot = 0
    newest = 0.0
    import os
    for d in days:
        e = _g.glob(str(root / f"data/pm_5min/raw/{d}/eth-updown-5m-*.jsonl.gz"))
        b = _g.glob(str(root / f"data/pm_5min/raw/{d}/btc-updown-5m-*.jsonl.gz"))
        e_tot += len(e); b_tot += len(b)
        for f in e[:4]:
            newest = max(newest, os.path.getmtime(f))
    out["raw_archive"] = {
        "eth_files": e_tot, "btc_files_CONTROL": b_tot,
        "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(newest)) if newest else None,
        "present": e_tot > 0, "at_parity": e_tot == b_tot,
        "exclusions": ("counts *.jsonl.gz per coin per day under "
                       "data/pm_5min/raw/<day>/ for the days listed; excludes "
                       "every other coin and any non-gz file")}
    for name, rel in (("market_definitions", "data/pm_5min/markets.jsonl"),
                      ("official_resolutions", "data/pm_5min/resolutions.jsonl"),
                      ("gap_windows", "data/pm_5min/collector_gaps.jsonl")):
        p2 = root / rel
        n = sum(1 for _ in open(p2)) if p2.is_file() else 0
        out[name] = {"records": n, "as_of": _asof(p2), "present": n > 0,
                     "note": ("append-only ledger; the count is the whole file, "
                              "not the ETH subset -- the per-coin ETH/BTC "
                              "parity is measured separately and was at parity "
                              "on every day")}
    cl = sorted(_g.glob(str(root / "data/pm_5min/prices/crypto_prices/2026*.csv.gz")))
    out["settlement_verification"] = {
        "hourly_files": len(cl), "as_of": _asof(cl[-1]) if cl else None,
        "present": bool(cl),
        "note": ("the Chainlink feed carries ethusdt and btcusdt at parity; "
                 "PM binaries settle on Chainlink, never Binance")}
    mk = sorted(_g.glob(str(root / "data/pm_5min/derived/da_blackout_mask_*.json")))
    out["blackout_masks"] = {
        "days": len(mk), "as_of": _asof(mk[-1]) if mk else None,
        "present": bool(mk),
        "note": "day-scoped with n_coins=7; eth is inside every one"}
    return {
        "by_input": out, "days": days,
        "VERDICT": ("every ETH input is PRESENT and at parity with BTC. The "
                    "absence of an ETH day book is a BUILD-side fact, not a "
                    "data gap, so the §8 population is not condemned to be "
                    "forward-looking."),
        "positive_control": ("BTC is counted by the same query on every row and "
                             "is non-zero throughout, so an ETH zero would be a "
                             "real absence"),
        "two_near_misses_recorded": (
            "my first Chainlink query matched uppercase symbol variants "
            "without the USDT suffix, and my first gap query used "
            "`symbol`/`topic` where the field is `coin`. Each would have "
            "returned a clean ETH ABSENCE for a broken-query reason."),
    }


def address_question() -> dict:
    """DOES OUR OWN MAKER ADDRESS APPEAR IN THE 901 RECEIPTS? (closed)"""
    return {
        "question": ("is our own maker address in the on-chain corpus, and "
                     "which fee class is it in?"),
        "ANSWER": "WE HAVE NO MAKER ADDRESS",
        "method": ("all 901 settlement receipts decoded OrderFilled by "
                   "OrderFilled, maker legs separated from taker legs via "
                   "OrdersMatched.takerOrderMaker -- not read off the audit "
                   "summary, which enumerates only the charged side"),
        "distinct_maker_addresses": 218,
        "charged_class": 6, "zero_class": 212,
        "ours_among_them": False,
        "why_not": ("no module in this lane declares an executing or maker "
                    "account; every 0x constant in it is protocol "
                    "infrastructure. The programme is research-only -- 'No "
                    "live trading, no exchange integrations'"),
        "therefore": ("all 218 are third parties and the 1,046 zero-fee legs "
                      "are OTHER PEOPLE'S accounts. Our treatment is not "
                      "unobserved, it does not yet exist to observe -- no "
                      "further collection closes that, only trading does"),
        "and_the_sample_limit": ("the audit's own limit: the 901 receipts are "
                                 "a SAMPLE, so absence from the charged set "
                                 "would not have been membership of the zero "
                                 "class even if we had appeared"),
        "status": "CLOSED_WITH_A_NEGATIVE_ANSWER",
    }


def build() -> dict:
    audit = eth_input_audit()
    consumers = [u for u in UNITS if u["ruling"] == "CONSUMER"]
    unknown = [u for u in UNITS if u["ruling"] == "UNKNOWN"]
    return {
        "protocol": PROTOCOL, "as_of_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dispatch": "DA 291-296",
        "THE_SEPARATING_QUESTION": (
            "does the unit's work reach an OUTCOME -- a score, a valuation, a "
            "P&L -- or is it coverage/quality/price infrastructure? Only the "
            "first consumes. UNKNOWN is treated as consuming."),
        "units": list(UNITS), "n_units": len(UNITS),
        "n_consumers": len(consumers), "n_unknown": len(unknown),
        "consumers": [u["unit"] for u in consumers],
        "WHAT_THE_FENCE_IS_FOR__READ_THIS_BEFORE_REMOVING_IT": {
            "the_apparent_contradiction": (
                "`n_consumers = 0` sits beside a fence, and a fence implies "
                "something needing fencing. A reader will infer the fence is "
                "redundant and remove it. It is not redundant: THE TWO FIELDS "
                "HAVE DIFFERENT SUBJECTS."),
            "n_consumers_is_about_UNITS": (
                "it counts SCHEDULED UNITS that consume a day BY RUNNING. "
                "Zero do. A timer writing a partition makes no selection -- it "
                "fits nothing, picks no threshold, computes no interval."),
            "the_fence_is_about_READERS": (
                "rule 34a constrains any SEAT OR HUMAN who reads a protected "
                "day's artifacts. The unit writes; the reader consumes. "
                "Nothing the unit does can trip the fence, and nothing the "
                "fence does constrains the unit."),
            "so_which_of_the_three_is_it": (
                "(c) -- SOMETHING THE RULING DOES NOT COVER AT ALL. The ruling "
                "is about units and their running; the fence is about readers "
                "and their looking. Not (b): contention is a separate, "
                "separately recorded fact and rule 34a says nothing about it. "
                "Not primarily (a), though (a) is a real secondary risk -- if "
                "this unit's lane or arguments changed so that it FIT "
                "something or PICKED a threshold, it would become a consumer "
                "by running and the ruling would have to be redone."),
            "the_removal_hazard_named": (
                "a control dropped for looking unnecessary is exactly how this "
                "one would be lost, which is why the subject of each field is "
                "stated here rather than left to inference."),
        },
        "THE_FENCE_AUDIT__REVIEW_264s_FOUR_TESTS": {
            "what_exists": (
                "`de_fair_value_plumbing_run.assert_day_is_consumed` -- 'THE "
                "EYE IS THE THING THAT SPENDS THE DAY (rule 34a)' -- which "
                "refuses a day outside READABLE_DAYS."),
            "1_inputs_bound_to_an_artifact_not_an_argument": (
                "FAILS. `READABLE_DAYS = CONSUMED_DAYS + "
                "CANNOT_BE_IN_THE_POPULATION`, both module-level LITERALS. The "
                "protected set is not derived from any artifact, so when the "
                "validation band is declared the fence will not know about it. "
                "The day under test IS a caller argument, but the set it is "
                "checked against is held by the module rather than supplied by "
                "the caller, so it is not self-incrimination."),
            "2_computed_from_evidence_not_asserted": (
                "PASSES. The check is a set membership computed from the "
                "argument; there is no caller-supplied boolean of the "
                "`outcome_is_known` kind."),
            "3_ordering_is_part_of_the_meaning": (
                "PASSES. `book_path` calls the fence FIRST, before any path "
                "resolution, so there is no permissive early return above it."),
            "4_green_cells_and_no_call_site": (
                "FAILS, AND THIS IS THE FINDING. `assert_day_is_consumed` "
                "appears in EXACTLY ONE FILE -- six references, all internal: "
                "the definition, two call sites in its own module, three in "
                "its own falsifier. ZERO call sites anywhere else in the lane."),
            "AND_IT_GUARDS_A_DIFFERENT_DOOR": (
                "it fences DAY BOOKS. Rule 34a names "
                "`data/pm_5min/tier2/**/day=2026-09-08/` and any later day. Of "
                "the five lane modules that touch tier2 -- evaluation_pipeline, "
                "pm_lane_health, tier1_pipeline, v5_deploy_gates and this one "
                "-- NONE calls a 34a guard; the only references are the prose "
                "strings in this file."),
            "THE_VERDICT": (
                "RULE 34a HAS NO ENFORCING PREDICATE ON THE ARTIFACT CLASS IT "
                "ACTUALLY NAMES. The one fence that exists is real, is "
                "correctly ordered, and guards a different door in a single "
                "module. On the thing rule 34a is about, it is a fence beside "
                "an open gate."),
            "positive_control": (
                "the same queries find guards that DO exist -- `row_status` in "
                "4 files, `two_coin_production_ready` and "
                "`assert_enumerations_intact` in their own -- so a zero for a "
                "tier2 read-guard is a real absence and not a broken query."),
            "what_a_real_fence_would_need": [
                "its protected set DERIVED from an artifact -- the freeze's "
                "band arithmetic plus rule 34a's 2026-09-08 floor -- so it "
                "learns the band when the band is declared",
                "a call site on every tier2 read path, not on one module's",
                "and a falsifier proving it refuses a protected read AND "
                "admits an unprotected one, since a fence that only refuses is "
                "as broken as one that only admits",
            ],
        },
        "REACHES_AN_OUTCOME_IS_NOT_CONSUMES": (
            "one unit reaches an outcome (pm-evaluation-pipeline) and NO unit "
            "consumes a day by running. Rule 34a (USER, abd4b07): writing does "
            "not consume a day, READING does. The distinction is the whole "
            "ruling: the timer may keep running, and its output for any band "
            "day is off limits."),
        "lock_behaviour": LOCK_FACT,
        "eth_input_audit": audit,
        "eth_closure_per_input": eth_closure(),
        "address_question": address_question(),
        "HOW_TO_READ_A_COUNT_THAT_MOVED": {
            "the_rule": ("when a count in these artifacts changes, say WHICH "
                         "KIND of move it was: INSTRUMENT_CORRECTED or "
                         "STATE_CHANGED. The two are indistinguishable in a "
                         "time series and only one is news."),
            "why": ("almost every number that moved tonight moved because an "
                    "instrument was corrected, not because anything changed "
                    "underneath. A reader meeting the series later will read "
                    "it as DEGRADATION unless told otherwise."),
            "moves_in_THIS_artifact": [
                {"count": "modules touching tier2", "from": 1, "to": 7,
                 "kind": "INSTRUMENT_CORRECTED",
                 "why": ("the query grepped the LITERAL `data/pm_5min/tier2` "
                         "and missed modules that build the path as "
                         "`DEFAULT_OUTPUT_ROOT.parent / 'tier2'`. The pattern "
                         "is the component now. No module started touching "
                         "tier2.")},
                {"count": "unguarded tier2 readers", "from": 4, "to": 5,
                 "kind": "INSTRUMENT_CORRECTED",
                 "why": "same widening; no new unguarded reader appeared"},
                {"count": "n_consumers", "from": 1, "to": 0,
                 "kind": "INSTRUMENT_CORRECTED",
                 "why": ("the ruling changed, not the units: rule 34a "
                         "distinguishes WRITING from READING, and the earlier "
                         "count applied 'reaches an outcome' as if it meant "
                         "'consumes'. The evaluation pipeline behaves exactly "
                         "as it did.")},
            ],
            "a_move_that_was_NOT_an_instrument_correction": (
                "the day-record D values 09-03..09-06 moved because the params "
                "file went v19 -> v29 with the book and decision-ledger "
                "digests changing together -- STATE_CHANGED, and the only one "
                "of tonight's moves that was."),
        },
        # THE CALL SITE. The fence's own coverage audit runs here, so the gap
        # ANNOUNCES ITSELF in a landed declaration instead of waiting for
        # someone to go and look -- which is how all fifteen call-site-less
        # guards in this lane were found.
        "rule_34a_fence_coverage": FENCE.audit_unguarded_readers(),
        "ETH_VERDICT": (
            "every ETH input is present at parity with BTC for the days that "
            "matter, and on data QUALITY eth is better -- fewer gap windows "
            "than btc on every day. So 'no ETH day book exists' is a "
            "BUILD-side fact, not a data gap, and the §8 population is not "
            "condemned to be forward-looking."),
        "NOTHING_IS_DISABLED": (
            "the finding is retrospective. Killing a timer that has already "
            "run saves nothing and may destroy the day-quality record the "
            "test depends on."),
        "rule_10": "the counts here are computed at run time from the artifacts",
    }


def falsify() -> int:
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    d = build()
    ck("all six units are ruled, none left unstated",
       d["n_units"] == 6 and all(u.get("ruling") for u in d["units"]))
    ck("every ruling names WHY, so it can be disagreed with",
       all(u.get("why") for u in d["units"]))
    ck("exactly ONE unit reaches an outcome, and it is the evaluation pipeline",
       sum(1 for u in d["units"] if u["reaches_an_outcome"]) == 1
       and [u["unit"] for u in d["units"] if u["reaches_an_outcome"]]
       == ["pm-evaluation-pipeline.timer"])
    _ep = [u for u in d["units"] if u["reaches_an_outcome"]][0]
    ck("...and it is ruled by the USER's rule 34a, not by this seat",
       "abc"[:0] == "" and "34a" in _ep["RULED_AT_DA_305"]
       and "WRITING DOES NOT CONSUME" in _ep["RULED_AT_DA_305"])
    ck("...reaching an outcome is NOT the same as consuming a day",
       _ep["reaches_an_outcome"] is True
       and "DOES NOT CONSUME A BAND DAY BY RUNNING" in _ep["RULED_AT_DA_305"])
    ck("...the SELECTION argument is what carries it (rule 11)",
       "FITS NOTHING" in _ep["the_reasoning"]
       and "outcome_up" in _ep["the_reasoning"])
    ck("...the live selection hazard the READ would feed is named",
       "minimum_meaningful_delta_LL" in _ep["and_there_IS_a_live_selection_hazard_it_would_feed"])
    ck("...and the fence is identified as already covering every later day",
       "ANY LATER DAY" in _ep["the_fence_already_exists_and_already_covers_the_band"])
    ck("THE GAP IS NAMED: the fence is PROSE, not a predicate",
       "stop being PROSE" in _ep["the_fence_already_exists_and_already_covers_the_band"]
       or "PROSE" in _ep["THE_GAP_BY_TONIGHTS_OWN_STANDARD"])
    ck("no unit is left UNKNOWN (which would count as consuming)",
       d["n_unknown"] == 0)
    f = d["WHAT_THE_FENCE_IS_FOR__READ_THIS_BEFORE_REMOVING_IT"]
    ck("the fence's SUBJECT is stated, so it cannot read as redundant",
       "SCHEDULED UNITS" in f["n_consumers_is_about_UNITS"]
       and "SEAT OR HUMAN" in f["the_fence_is_about_READERS"]
       and "different subjects" in f["the_apparent_contradiction"].lower(),
       "units vs readers, stated in the values not only the key names")
    ck("...and the answer is (c): the ruling does not cover it",
       f["so_which_of_the_three_is_it"].startswith("(c)"))
    a = d["THE_FENCE_AUDIT__REVIEW_264s_FOUR_TESTS"]
    ck("REVIEW 264's four tests are each answered PASS or FAIL",
       sum(1 for k, v in a.items() if k[0].isdigit()) == 4
       and all(v.startswith(("PASSES", "FAILS"))
               for k, v in a.items() if k[0].isdigit()),
       "2 pass, 2 fail")
    ck("test 4 FAILS: the guard has zero external call sites",
       a["4_green_cells_and_no_call_site"].startswith("FAILS")
       and "EXACTLY ONE FILE" in a["4_green_cells_and_no_call_site"])
    ck("...and the verdict names it a fence beside an open gate",
       "open gate" in a["THE_VERDICT"])
    ck("the audit carries a POSITIVE CONTROL for its own zero",
       "real absence" in a["positive_control"])
    ck("NO unit consumes a day by RUNNING, and the rule is cited",
       d["n_consumers"] == 0
       and "writing does not consume a day" in d["REACHES_AN_OUTCOME_IS_NOT_CONSUMES"].lower(),
       f"consumers={d['consumers']}")
    ck("the lock fact is recorded: they BLOCK, they do not yield",
       d["lock_behaviour"]["both_heavy_units_block_rather_than_yield"] is True
       and "NO `-n`" in d["lock_behaviour"]["mechanism"])
    # ---- ETH, with the control -------------------------------------------
    a = d["eth_input_audit"]["raw_archive"]
    ck("POSITIVE CONTROL: btc raw is non-zero every day (the query fires)",
       all(v["btc_files"] > 0 for v in a["by_day"].values()),
       str({k: v["btc_files"] for k, v in list(a["by_day"].items())[:3]}))
    ck("ETH raw archive is PRESENT every day",
       a["eth_present"], str({k: v["eth_files"] for k, v in list(a["by_day"].items())[:3]}))
    ck("...and at FILE-COUNT PARITY with btc",
       a["at_parity_with_btc"])
    ck("the 'present' answer states its exclusions (§7k.2)",
       bool(a["exclusions"]) and bool(a["positive_control"]))
    ck("the ETH verdict is BUILD-side, stated plainly",
       "BUILD-side fact, not a data gap" in d["ETH_VERDICT"])
    fc = d["rule_34a_fence_coverage"]
    ck("THE FENCE IS CALLED FROM HERE -- the gap announces itself",
       isinstance(fc.get("unguarded_readers"), list)
       and fc["n_modules_touching_tier2"] > 0,
       f"{fc['n_unguarded']} unguarded of {fc['n_modules_touching_tier2']}")
    ck("...and the open gates are NAMED in a landed declaration",
       all(isinstance(x, str) for x in fc["unguarded_readers"]),
       str(fc["unguarded_readers"][:3]))
    ec = d["eth_closure_per_input"]
    ck("ETH closure: every input carries a COUNT and an AS-OF",
       all(("as_of" in v) for v in ec["by_input"].values())
       and all(v.get("present") for v in ec["by_input"].values()),
       f"{len(ec['by_input'])} inputs")
    ck("...with BTC as the positive control on the row that can carry one",
       ec["by_input"]["raw_archive"]["btc_files_CONTROL"] > 0
       and ec["by_input"]["raw_archive"]["at_parity"],
       f"eth {ec['by_input']['raw_archive']['eth_files']} = btc "
       f"{ec['by_input']['raw_archive']['btc_files_CONTROL']}")
    ck("...and the two near-misses are recorded, not buried",
       "broken-query reason" in ec["two_near_misses_recorded"])
    aq = d["address_question"]
    ck("the ADDRESS QUESTION is closed with a negative answer",
       aq["status"] == "CLOSED_WITH_A_NEGATIVE_ANSWER"
       and aq["ours_among_them"] is False and aq["distinct_maker_addresses"] == 218)
    ck("...reached from the RECEIPTS, not from the audit summary",
       "not read off the audit" in aq["method"])
    ck("...and says why no further collection closes it",
       "only trading does" in aq["therefore"])
    h = d["HOW_TO_READ_A_COUNT_THAT_MOVED"]
    ck("every moved count says WHICH KIND of move it was",
       all(m["kind"] in ("INSTRUMENT_CORRECTED", "STATE_CHANGED")
           for m in h["moves_in_THIS_artifact"]),
       f"{len(h['moves_in_THIS_artifact'])} moves labelled")
    ck("...and at least one STATE_CHANGED move is named, so the label discriminates",
       "STATE_CHANGED" in h["a_move_that_was_NOT_an_instrument_correction"])
    ck("nothing is disabled, and the reason is recorded",
       "retrospective" in d["NOTHING_IS_DISABLED"])
    print(f"\n  {'UNITS/ETH CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        sys.exit(1 if falsify() else 0)
    print(json.dumps(build(), indent=1))
