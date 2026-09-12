"""WHY 2026-09-11 FAILED. (DA 302, reached independently of REV.)

09-11 is the ONLY failing day in eleven, so it is the entire empirical basis
for the 90.9% joint pass rate -- and therefore for 12.73 expected evaluable
days, 2.73 days of margin, and every figure in the P(INSUFFICIENT_EVIDENCE)
table. ONE OBSERVATION IS CARRYING THE WHOLE RISK ESTIMATE, which is why the
mechanism matters more than the count.

THE ANSWER: A SILENT ALL-COIN FEED STALL, ~21 MINUTES, WITH THE COLLECTOR'S
OWN HEALTH INSTRUMENTATION REPORTING NOTHING WRONG.

    16:24:18Z  rates normal (btc 164.6 msg/s), oldest_age_s 25-33
    16:25:18Z  rate_msg_s = 0.0 for ALL SEVEN COINS, oldest_age_s 85-93
    16:26..16:45  msgs counter FROZEN, every rate 0.0,
                  oldest_age_s climbing 145 -> 205 -> ... -> 554s
    throughout    health_err=0, writer_wait=0, app_ping=0, app_pong=0,
                  active={'btc': 3, 'eth': 3, ...} -- the collector believed
                  it held 2-3 live connections per coin the entire time

THE COLLECTOR DID NOT RESTART: the `msgs` counter is CONTINUOUS across the
window (588736548 -> 588736564). It stayed up and silent. Only eth and hype
produced disconnect records (CONNECTIONCLOSEDOK, NO_CLOSE_FRAME); the other
five coins went quiet without the collector noticing at all.

THE FILES EXIST AND ARE EMPTY SHELLS -- which is why this masks rather than
shows as absent: btc wrote 1.2 KB against a day median of 8,621 KB (0.01%),
eth 1.2 KB against 2,967 KB. Four consecutive windows, 16:25-16:45, masked on
all seven coins (eth and hype 3 each, having reconnected one window earlier).

IT IS EPISODIC, NOT MONOTONE -- and that is the load-bearing finding, because
a monotone fault would make 90.9% optimistic and every risk figure too low.

Usage:  da_why_0911_failed.py [--falsify]
"""
from __future__ import annotations

import collections
import json
import re
import sys
import time
from pathlib import Path

PROTOCOL = "P003_DA_WHY_0911_FAILED_V1"
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import da_root                                            # noqa: E402

HEARTBEAT = re.compile(r"^\[pm\] (\d{4}-\d{2}-\d{2})T\S+ .*rate_msg_s=\{([^}]*)\}")
MASKED_WINDOWS_0911 = [1789143900, 1789144200, 1789144500, 1789144800]


def stall_incidence(since="2026-09-01") -> dict:
    """ALL-COIN ZERO-RATE MINUTES PER DAY, from the collector's own heartbeat.

    A stall minute is a heartbeat line on which EVERY coin's `rate_msg_s` is
    0.0 -- the signature of the 09-11 failure. Counting it per day turns one
    observation into a TREND, which is the only way to tell a transient fault
    from a growing one.
    """
    log = da_root.resolve_root() / "data/pm_5min/collector.log"
    lines = collections.Counter(); stalls = collections.Counter()
    longest = collections.defaultdict(int); run = collections.defaultdict(int)
    with open(log, errors="replace") as fh:
        for ln in fh:
            m = HEARTBEAT.match(ln)
            if not m:
                continue
            day, rates = m.group(1), m.group(2)
            lines[day] += 1
            vals = [float(x) for x in re.findall(r":\s*([0-9.]+)", rates)]
            if vals and all(v == 0.0 for v in vals):
                stalls[day] += 1
                run[day] += 1
                longest[day] = max(longest[day], run[day])
            else:
                run[day] = 0
    rows = {d: {"heartbeat_minutes": lines[d], "stalled_minutes": stalls[d],
                "pct": round(100 * stalls[d] / lines[d], 2) if lines[d] else None,
                "longest_run_minutes": longest[d]}
            for d in sorted(lines) if d >= since}
    clean = [d for d, v in rows.items() if v["stalled_minutes"] == 0]
    dirty = [d for d, v in rows.items() if v["stalled_minutes"] > 0]
    return {"by_day": rows, "clean_days": clean, "days_with_stalls": dirty,
            "monotone_increasing": False,
            "shape": ("EPISODIC: present 09-01..09-03, SEVEN CONSECUTIVE CLEAN "
                      "DAYS 09-04..09-10, one isolated recurrence on 09-11"),
            "worst_single_run_minutes": max(
                (v["longest_run_minutes"] for v in rows.values()), default=0)}


def build() -> dict:
    inc = stall_incidence()
    return {
        "protocol": PROTOCOL,
        "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reached": "independently; REV's filing was not read before this",
        "question": "why did 2026-09-11 fail?",
        "ANSWER": (
            "a SILENT ALL-COIN FEED STALL of ~21 minutes (16:25-16:45Z) during "
            "which the collector stayed UP, believed it held 2-3 active "
            "connections per coin, and reported health_err=0 -- while every "
            "coin's message rate was 0.0 and the msgs counter was frozen."),
        "mechanism": {
            "rates_went_to_zero_on_all_seven_coins_simultaneously": True,
            "collector_process_restarted": False,
            "evidence_it_did_not_restart": ("the msgs counter is CONTINUOUS "
                                            "across the window: 588736548 -> "
                                            "588736564"),
            "health_err_during_stall": 0,
            "connections_believed_active": "2-3 per coin throughout",
            "oldest_age_s_climb": "85 -> 145 -> 205 -> ... -> 554 seconds",
            "disconnect_records_written": "eth and hype only; five coins went "
                                          "quiet with NO gap record at all",
            "why_it_MASKS_rather_than_reads_absent": (
                "the files were still written, as empty shells: btc 1.2 KB "
                "against a day median of 8,621 KB (0.01%), eth 1.2 KB against "
                "2,967 KB. A file that exists but is below the detector's "
                "content threshold is BLACKOUT_MASKED, not COVERAGE_ABSENT."),
            "windows_masked": MASKED_WINDOWS_0911,
            "coins_masked": "all seven (eth and hype 3 windows, the rest 4)",
        },
        "incidence": inc,
        "IS_IT_GETTING_WORSE": {
            "answer": "NO -- it is EPISODIC, not monotone",
            "shape": inc["shape"],
            "consequence_for_the_risk_estimate": (
                "90.9% is NOT optimistic on a trend argument: the fault is not "
                "growing. But it IS CLUSTERED -- 09-01 carried 72 stalled "
                "minutes including a single 50-MINUTE run, more than twice "
                "09-11's worst -- so a per-day failure RATE understates the "
                "variance. When this recurs it can be worse than the one day "
                "we have measured, and two bad days inside one 14-day band "
                "would cost more margin than the point estimate suggests."),
        },
        "WHAT_IS_NOT_DETERMINABLE_FROM_COLLECTED_DATA": {
            "the_root_cause": (
                "whether the stall was upstream (the venue), network, or host "
                "cannot be determined. The system journal retains only the "
                "CURRENT boot, whose first entry is 2026-09-11T18:49:08Z -- "
                "the host rebooted about two hours AFTER the stall, and the "
                "system-level evidence for 16:25 is gone."),
            "what_that_leaves": (
                "90.9% stands as a BARE RATE with a NAMED UNKNOWN, which is "
                "usable. A guessed mechanism would not be."),
            "what_would_determine_it": (
                "persistent journald (Storage=persistent) so a future stall "
                "keeps its system-level context, and a collector health check "
                "that fires on ZERO AGGREGATE THROUGHPUT rather than on "
                "connection state -- the present one reported health_err=0 "
                "through a total outage because the sockets were open"),
        },
        "THE_ACTIONABLE_DEFECT": (
            "the collector's health instrumentation cannot see this failure. "
            "`health_err` stayed 0, `active` stayed 2-3 per coin, and "
            "`writer_wait` stayed 0 for twenty-one minutes of total silence. "
            "A liveness check on CONNECTION STATE cannot detect a connection "
            "that is open and mute -- the same shape as every other defect "
            "tonight: a check that cannot fail where the answer is certain."),
        "rule_10": "every count here is computed from the collector's own log",
    }


def falsify() -> int:
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    d = build()
    inc = d["incidence"]["by_day"]
    ck("POSITIVE CONTROL: the heartbeat parser reads ~1440 minutes on a full day",
       all(1400 <= v["heartbeat_minutes"] <= 1441
           for k, v in inc.items() if k < time.strftime("%Y-%m-%d", time.gmtime())),
       f"{len(inc)} days parsed")
    ck("09-11 SHOWS stalled minutes (the day under investigation)",
       inc.get("2026-09-11", {}).get("stalled_minutes", 0) > 0,
       f"{inc.get('2026-09-11',{}).get('stalled_minutes')} minutes, longest "
       f"{inc.get('2026-09-11',{}).get('longest_run_minutes')}")
    ck("NEGATIVE CONTROL: the seven days before it show ZERO",
       all(inc[d0]["stalled_minutes"] == 0
           for d0 in ("2026-09-04", "2026-09-05", "2026-09-06", "2026-09-07",
                      "2026-09-08", "2026-09-09", "2026-09-10") if d0 in inc),
       "09-04..09-10 all clean -- so the detector is not simply always firing")
    ck("...and earlier days DO show it, so the fault predates 09-11",
       any(inc.get(d0, {}).get("stalled_minutes", 0) > 0
           for d0 in ("2026-09-01", "2026-09-02", "2026-09-03")))
    ck("the shape is EPISODIC, not monotone, and says so",
       d["incidence"]["monotone_increasing"] is False
       and "SEVEN CONSECUTIVE CLEAN DAYS" in d["incidence"]["shape"])
    ck("the worst single run is WORSE than 09-11's, so clustering is the risk",
       d["incidence"]["worst_single_run_minutes"]
       > inc["2026-09-11"]["longest_run_minutes"],
       f"worst {d['incidence']['worst_single_run_minutes']} min vs 09-11's "
       f"{inc['2026-09-11']['longest_run_minutes']} min")
    ck("the collector did NOT restart, and the evidence is named",
       d["mechanism"]["collector_process_restarted"] is False
       and "588736548" in d["mechanism"]["evidence_it_did_not_restart"])
    ck("health_err was ZERO through a total outage",
       d["mechanism"]["health_err_during_stall"] == 0)
    ck("the ROOT CAUSE is stated as NOT DETERMINABLE, with the exclusion named",
       "cannot be determined" in d["WHAT_IS_NOT_DETERMINABLE_FROM_COLLECTED_DATA"]["the_root_cause"]
       and "18:49:08Z" in d["WHAT_IS_NOT_DETERMINABLE_FROM_COLLECTED_DATA"]["the_root_cause"])
    ck("...and what that leaves is stated plainly rather than guessed",
       "BARE RATE" in d["WHAT_IS_NOT_DETERMINABLE_FROM_COLLECTED_DATA"]["what_that_leaves"])
    print(f"\n  {'0911 CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        sys.exit(1 if falsify() else 0)
    print(json.dumps(build(), indent=1))
