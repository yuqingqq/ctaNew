"""THE THIRD STATUS: an EMPTY SHELL is neither a blackout nor an absence.

DA 303. A 1.2 KB file where the day median is 8,621 KB EXISTS, so a presence
check says yes; it carries no content, so a coverage check should say no; and
the status it currently receives -- BLACKOUT_MASKED -- is the one that reads
as EXPLAINED rather than MISSING. That is a value that is neither, resolving
toward permission, and it is the sixth costume of tonight's defect:

    the gate computing None; the required field simply absent; the refusal
    that cannot tell "cannot check" from "missing"; the trailing
    `else: SATISFIED`; two fields carrying one bit; and now a file that
    exists but says nothing.

SO THERE ARE THREE STATUSES, NOT TWO, AND THE THRESHOLD IS EXPLICIT:

    COVERAGE_ABSENT  no file at all
    EMPTY_SHELL      a file exists and carries < SHELL_FRACTION of the day's
                     median size -- present, and empty
    BLACKOUT_MASKED  a file exists with real but sub-threshold content: a
                     genuinely quiet window, which is a different fact
    PRESENT          at or above the content threshold

WHY THE MEDIAN AND NOT A BYTE COUNT. A fixed byte floor cannot separate "the
feed stopped" from "this coin is thin" -- hype trades a fraction of btc's
volume and a 60 KB hype window may be healthy. The day's own median per coin
is the only scale that travels across coins and across days.

Usage:  da_window_content_status.py [--falsify] [--day YYYYMMDD]
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

PROTOCOL = "P003_DA_WINDOW_CONTENT_STATUS_V1"

#: A file below this fraction of the day's median for its own coin is an
#: EMPTY SHELL. 0.01 is two orders of magnitude below typical; the 09-11 stall
#: produced files at 0.0001-0.011 of median, and the quietest healthy windows
#: measured on clean days sit far above it.
SHELL_FRACTION = 0.01
#: Below this, a window is sub-threshold but has real content -- a quiet
#: window, not a dead feed.
QUIET_FRACTION = 0.25

COVERAGE_ABSENT = "COVERAGE_ABSENT"
EMPTY_SHELL = "EMPTY_SHELL"
BLACKOUT_MASKED = "BLACKOUT_MASKED"
PRESENT = "PRESENT"

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import da_root                                            # noqa: E402


def classify(size: int | None, median: float,
             shell=SHELL_FRACTION, quiet=QUIET_FRACTION) -> dict:
    """THE THREE-WAY STATUS, with the threshold REPORTED, not implicit."""
    if size is None:
        return {"status": COVERAGE_ABSENT, "fraction_of_median": None,
                "why": "no file at all"}
    frac = (size / median) if median else None
    if frac is not None and frac < shell:
        s, why = EMPTY_SHELL, (f"the file EXISTS and carries {frac:.4%} of the "
                               f"day median -- present, and empty")
    elif frac is not None and frac < quiet:
        s, why = BLACKOUT_MASKED, (f"real but sub-threshold content "
                                   f"({frac:.1%} of median): a quiet window")
    else:
        s, why = PRESENT, f"{frac:.1%} of median"
    return {"status": s, "fraction_of_median": frac, "why": why,
            "thresholds": {"shell_fraction": shell, "quiet_fraction": quiet}}


def day_statuses(day: str, coins=("btc", "eth")) -> dict:
    root = da_root.resolve_root()
    out = {}
    for coin in coins:
        sizes = {}
        for f in glob.glob(str(root / f"data/pm_5min/raw/{day}/{coin}-updown-5m-*.jsonl.gz")):
            ws = int(f.rsplit("-", 1)[1].split(".")[0])
            sizes[ws] = os.path.getsize(f)
        if not sizes:
            out[coin] = {"n_windows": 0, "median_bytes": None, "counts": {}}
            continue
        med = st.median(sizes.values())
        counts, shells = {}, []
        for ws, sz in sorted(sizes.items()):
            c = classify(sz, med)
            counts[c["status"]] = counts.get(c["status"], 0) + 1
            if c["status"] == EMPTY_SHELL:
                shells.append({"window": ws,
                               "utc": time.strftime("%H:%M", time.gmtime(ws)),
                               "bytes": sz,
                               "fraction_of_median": round(c["fraction_of_median"], 6)})
        out[coin] = {"n_windows": len(sizes), "median_bytes": med,
                     "counts": counts, "empty_shells": shells}
    return out


def build(day: str = "20260911") -> dict:
    return {
        "protocol": PROTOCOL, "day": day,
        "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "statuses": [COVERAGE_ABSENT, EMPTY_SHELL, BLACKOUT_MASKED, PRESENT],
        "thresholds": {"shell_fraction": SHELL_FRACTION,
                       "quiet_fraction": QUIET_FRACTION,
                       "scale": "the day's MEDIAN window size FOR THAT COIN"},
        "why_three_not_two": (
            "an EMPTY SHELL is present and empty. Classed as BLACKOUT_MASKED "
            "it reads as EXPLAINED; classed as COVERAGE_ABSENT it would read "
            "as a collection hole the archive could not fill. It is neither, "
            "and a status that is neither resolves toward permission."),
        "why_the_median_and_not_a_byte_floor": (
            "a fixed floor cannot separate 'the feed stopped' from 'this coin "
            "is thin'. hype trades a fraction of btc's volume, so the only "
            "scale that travels across coins and days is the day's own median "
            "for that coin."),
        "by_coin": day_statuses(day),
        "NOT_YET_WIRED_INTO_THE_MASK_PRODUCER": (
            "`da_blackout_mask.py` still emits the two-status split. Wiring "
            "this in changes a producer that is under the deploy pin, and that "
            "deploy is currently blocked by 71 untracked lane files -- a "
            "tree-state question that is not this seat's to clear. So the "
            "classifier and its threshold are landed and driveable now, and "
            "the producer adopts them when the deploy question is resolved."),
        "rule_10": "the counts and the median are computed from the archive",
    }


def falsify() -> int:
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    med = 8_621_000.0
    ck("a 1.2 KB file against an 8,621 KB median is an EMPTY SHELL",
       classify(1_200, med)["status"] == EMPTY_SHELL,
       f"{classify(1_200, med)['fraction_of_median']:.6%} of median")
    ck("NO FILE is COVERAGE_ABSENT, a different status",
       classify(None, med)["status"] == COVERAGE_ABSENT)
    ck("a genuinely QUIET window is BLACKOUT_MASKED, not a shell",
       classify(int(0.10 * med), med)["status"] == BLACKOUT_MASKED,
       "10% of median: real content, below threshold")
    ck("a healthy window is PRESENT",
       classify(int(0.9 * med), med)["status"] == PRESENT)
    ck("the three exclusions are DISTINCT -- no two share a status",
       len({classify(None, med)["status"], classify(1_200, med)["status"],
            classify(int(0.10 * med), med)["status"]}) == 3)
    ck("the THRESHOLD is reported beside every verdict, not implicit",
       classify(1_200, med)["thresholds"]["shell_fraction"] == SHELL_FRACTION)
    d = build("20260911")
    btc = d["by_coin"]["btc"]
    ck("POSITIVE CONTROL: 09-11 btc has the expected ~288 windows",
       280 <= btc["n_windows"] <= 288, f"{btc['n_windows']} windows")
    ck("DRIVEN: the 09-11 stall windows classify as EMPTY_SHELL",
       btc["counts"].get(EMPTY_SHELL, 0) >= 3,
       f"{btc['counts'].get(EMPTY_SHELL)} shells: "
       f"{[s['utc'] for s in btc['empty_shells']]}")
    clean = build("20260910")["by_coin"]["btc"]
    ck("NEGATIVE CONTROL: a CLEAN day (09-10) has ZERO empty shells",
       clean["counts"].get(EMPTY_SHELL, 0) == 0,
       f"09-10 counts: {clean['counts']}")
    ck("...so the classifier is not simply flagging everything",
       clean["counts"].get(PRESENT, 0) > 200)
    print(f"\n  {'CONTENT-STATUS CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        sys.exit(1 if falsify() else 0)
    day = "20260911"
    if "--day" in sys.argv:
        day = sys.argv[sys.argv.index("--day") + 1]
    print(json.dumps(build(day), indent=1))
