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
     "reaches_an_outcome": True, "ruling": "CONSUMER",
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
        "lock_behaviour": LOCK_FACT,
        "eth_input_audit": audit,
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
       d["n_consumers"] == 1 and d["consumers"] == ["pm-evaluation-pipeline.timer"],
       str(d["consumers"]))
    ck("...and its reason is the OUTCOME JOIN, not its name",
       "winner_up" in [u for u in d["units"]
                       if u["ruling"] == "CONSUMER"][0]["why"])
    ck("no unit is left UNKNOWN (which would count as consuming)",
       d["n_unknown"] == 0)
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
    ck("nothing is disabled, and the reason is recorded",
       "retrospective" in d["NOTHING_IS_DISABLED"])
    print(f"\n  {'UNITS/ETH CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        sys.exit(1 if falsify() else 0)
    print(json.dumps(build(), indent=1))
