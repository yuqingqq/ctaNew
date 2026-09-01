#!/usr/bin/env python3
"""CONTENT-LIVENESS RULE — DRAFT FOR USER FREEZE. It governs NOTHING today.

R-370 left this open: *"a content-liveness status rule must be frozen BEFORE
the next day is judged, not after."* 08-31 still stands as
`CONTENT_LIVENESS_UNRESOLVED`.

THE HOLE IT ADDRESSES, in one sentence: a feed that THINS without
disconnecting writes no gap row, leaves full window coverage, and passes
P1/P2/P3 — so the day passes carrying a fraction of its data. Measured, not
supposed: 08-31 held the feed at 0.51% of normal rate for ~4.1 h with NO gap
rows (R-368/R-369), and 668 windows across 7 of 13 days hold near-zero data
invisibly (R-362).

WHAT THIS FILE IS
    a status vocabulary, a detector, two proposed bars, their provenance, and
    controls in both directions.
WHAT IT IS NOT
    a gate. `governs()` returns False for every day until a USER freeze flips
    `FROZEN_BY_USER`, and no other module consumes it. Models estimate; policy
    decides (rule 14). Nobody amends a design after seeing it (rule 4 of the
    seat protocol) — so this is DRAFT-FOR-USER-FREEZE and I do not ratify it.

RULE 11 IS ENFORCED MECHANICALLY, NOT PROMISED
    `calibrate()` REFUSES any day after `CALIBRATION_MAX_DAY` ("20260831").
    Every calibration day is already consumed/seen, which is exactly why they
    may set a bar that is applied PROSPECTIVELY and never used to validate it.
    **2026-09-01 DID NOT INFORM THESE THRESHOLDS AND CANNOT**: the refusal is
    a check with a falsifier, so the claim is verifiable rather than asserted.
    The rule takes effect from `EFFECTIVE_FROM_DAY` = 2026-09-02; 09-01 is
    measured and REPORTED, never judged, by it.

    python3 live/pm_research/da_content_liveness_rule.py --selftest
    python3 live/pm_research/da_content_liveness_rule.py --calibrate
    python3 live/pm_research/da_content_liveness_rule.py --day 20260831
    python3 live/pm_research/da_content_liveness_rule.py --draft
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_tape_density as TD                                  # noqa: E402

WINDOW_S = TD.WINDOW_S
WINDOWS_PER_DAY = 288

#: THE FREEZE SWITCH. One constant, so ratifying the rule is a one-line USER
#: act with a commit behind it -- and so nothing can start governing by
#: accident. While this is False every consumer must read `governs: false`.
FROZEN_BY_USER = True  # USER ruling 2026-09-01 ("Yea proceed" on the coordinator's
#: recommended course, R-386); frozen by coordinator commit on that authority.
#: EFFECTIVE_FROM_DAY unchanged — first governed day is 2026-09-02.

#: Prospective from here. NOT 09-01: the rule is drafted while 09-01 is in
#: flight, so applying it to that day is choosing after seeing (rule 11).
EFFECTIVE_FROM_DAY = "20260902"

#: Rule 11's hard edge, enforced by a refusal rather than a comment.
CALIBRATION_MAX_DAY = "20260831"

#: Inherited from `pm_tape_density`, NOT re-chosen here. The DAY-LEVEL
#: classification below is invariant across thin_frac in [0.01, 0.25] -- a 25x
#: range -- which is the property that matters; the window COUNT is not.
THIN_FRAC = TD.THIN_FRAC
MIN_WINDOWS_FOR_MEDIAN = TD.MIN_WINDOWS_FOR_MEDIAN

#: ---------------------------------------------------------------------------
#: THE TWO PROPOSED BARS
#: ---------------------------------------------------------------------------
#: L1 SEVERITY -- invisible content shortfall as a fraction of the content the
#: coin-day's PRESENT windows should have held. Denominator is present windows,
#: not 288, because ABSENT windows are `complete_tape`'s question and counting
#: them here would charge one loss to two predicates.
#:
#: L2 CONCENTRATION -- the longest run of consecutive invisible-thin windows.
#: NOT redundant with L1, and the arithmetic says so rather than the prose: a
#: total one-hour blackout is 12 empty windows of 288, i.e. L1 = 4.17%, which
#: PASSES an 8% severity bar. L1 sees mass, L2 sees continuity, and the 08-26
#: and 08-31 events are both continuous. The selftest drives exactly this case.
L1_SEVERITY_MAX = 0.08
L2_RUN_WINDOWS_MAX = 12          # 12 x 300s = 60 minutes

#: PROVENANCE -- the reading these bars were chosen from, pinned with its
#: as-of so drift in the tape is visible instead of silent. Worst coin per day.
#: Recomputable with --calibrate; the selftest asserts the PROPERTIES of this
#: table (the empty band, the invariance), not just its presence.
CALIBRATION_AS_OF = "2026-09-01T13:57:00Z"
CALIBRATION_READING = {
    # day        (L1 severity, L2 longest run, windows present on worst coin)
    "20260819": (0.04290, 2, 115),      # PARTIAL day: collector started mid-day
    "20260820": (0.00346, 1, 288),
    "20260821": (0.00000, 0, 288),
    "20260822": (0.01718, 5, 288),
    "20260823": (0.00000, 0, 288),
    "20260824": (0.00000, 0, 288),
    "20260825": (0.00000, 0, 288),
    "20260826": (0.14300, 40, 279),     # R-362: contiguous 3h20m, all coins
    "20260827": (0.00000, 0, 288),
    "20260828": (0.00000, 0, 288),
    "20260829": (0.01041, 1, 288),      # R-362's 10 invisible windows
    "20260830": (0.01723, 5, 288),
    "20260831": (0.21280, 48, 288),     # R-368/R-369: the ~4h quiet event
}
#: The two days the rule exists for, named so a reader can check the detector
#: fires on the events that motivated it rather than on something else.
CALIBRATION_KNOWN_EVENTS = ("20260826", "20260831")

DRAFT = f"""\
CONTENT-LIVENESS RULE — DRAFT FOR USER FREEZE (DA, {CALIBRATION_AS_OF})

1. QUESTION.  Did this UTC day's tape lose CONTENT that no gap row records?
   Distinct from every existing predicate: `complete_tape` asks whether the
   WINDOW is present, P1/P2/P3 ask how long the feed was KNOWN to be down.
   A thinned-but-connected feed answers all three favourably.

2. STATUS VOCABULARY.  Statuses, never a silent anything (rule 4).
     CONTENT_LIVE                    both bars hold, with their measured values
     CONTENT_THIN                    a bar fails; the failing bar is NAMED
     CONTENT_LIVENESS_UNJUDGEABLE    no coin has enough windows for a median
     CONTENT_LIVENESS_UNRESOLVED     the detector cannot reach the day at all
   Every status carries n (windows judged), the coin it is worst on, and the
   as-of of the scan. UNJUDGEABLE and UNRESOLVED are NOT passes.

3. DETECTOR.  `pm_tape_density`'s measure, unchanged: uncompressed bytes per
   (coin, window) from the gzip trailer; THIN = below {THIN_FRAC:.2f} x that
   (day, coin) MEDIAN; INVISIBLE = thin AND not overlapped by a gap-ledger
   interval for that coin. Relative to the coin's own median because coin
   activity spans two orders of magnitude.

4. BARS (proposed).  Per coin-day, then worst-coin for the day:
     L1  severity    sum(median - bytes) over INVISIBLE windows,
                     / (windows_present x median)          <= {L1_SEVERITY_MAX:.2%}
     L2  concentration  longest run of consecutive INVISIBLE windows
                                                           <= {L2_RUN_WINDOWS_MAX} windows (60 min)

5. PROVENANCE, and why these are not fitted numbers.  On the 13 calibration
   days the two statistics are BIMODAL with an empty band between them:
     non-event days   L1 <= 4.29%   L2 <= 5 windows
     08-26 event      L1 = 14.30%   L2 = 40 windows
     08-31 event      L1 = 21.28%   L2 = 48 windows
   ANY L1 bar in (4.29%, 14.30%) and ANY L2 bar in (5, 40) classifies all 13
   days identically. 8% is near the geometric middle of the L1 band; 12
   windows is the round hour inside the L2 band. The day classification is
   also invariant across thin_frac in [0.01, 0.25].

6. RULE 11.  All 13 calibration days are CONSUMED/SEEN — which is what makes
   them admissible for setting a prospective bar and inadmissible for
   validating one. **2026-09-01 did not inform these thresholds**; the
   calibrator REFUSES any day after {CALIBRATION_MAX_DAY}, with a falsifier.
   The rule applies from {EFFECTIVE_FROM_DAY}. It does not re-judge any day
   before that — 08-29 keeps its verdict, and 08-31 keeps its UNRESOLVED
   status as the record of what was true when it was judged.

7. GOVERNING POWER: NONE until the USER freezes it (rule 14). This file
   REPORTS. `FROZEN_BY_USER` is False, `governs()` is False for every day, and
   no other module consumes this one. Adopting it means flipping one constant
   in a commit the USER authorises — and at that point a day it fails becomes
   a day the COORDINATOR excludes with a stated reason, not a day this
   instrument rejects.

8. WHAT THE COORDINATOR/USER STILL HAS TO DECIDE, listed because a draft that
   hides its open questions is not a draft:
     (a) whether L1/L2 join the governing set or stay REPORTED beside it
         (the `tape_density` disposition under R-362);
     (b) worst-coin vs per-coin-day granularity — R-211(3) makes coin-days
         independent, and this draft computes BOTH;
     (c) whether a CONTENT_THIN day is inadmissible or merely disclosed;
     (d) 08-31's standing status, which this rule does not retroactively move.
"""


class Refused(Exception):
    """A population this rule must not summarise."""


def governs(day_token: str) -> bool:
    """False for every day until a USER freeze. Deliberately the only place
    the two conditions meet, so a consumer cannot satisfy one and forget the
    other."""
    return bool(FROZEN_BY_USER) and day_token >= EFFECTIVE_FROM_DAY


def measure_day(day_token: str, gaps=None, raw_root: Path | None = None,
                thin_frac: float = THIN_FRAC) -> dict[str, Any]:
    """L1/L2 per coin and worst-coin, with a STATUS. Decides nothing."""
    root = TD.RAW if raw_root is None else raw_root
    try:
        agg = TD.scan_day(day_token, root)
    except TD.Refused as e:
        return {"day": day_token, "status": "CONTENT_LIVENESS_UNRESOLVED",
                "why": str(e), "governs": governs(day_token),
                "frozen_by_user": FROZEN_BY_USER}
    gaps = TD.load_gaps() if gaps is None else gaps
    per: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for (c, w), b in agg.items():
        per[c].append((w, b))

    coins: dict[str, Any] = {}
    for c, wins in sorted(per.items()):
        wins.sort()
        if len(wins) < MIN_WINDOWS_FOR_MEDIAN:
            coins[c] = {"status": "CONTENT_LIVENESS_UNJUDGEABLE",
                        "n_windows": len(wins),
                        "why": f"{len(wins)} windows < {MIN_WINDOWS_FOR_MEDIAN}"}
            continue
        med = statistics.median([b for _, b in wins])
        if med <= 0:
            coins[c] = {"status": "CONTENT_LIVENESS_UNJUDGEABLE",
                        "n_windows": len(wins), "why": "median window is empty"}
            continue
        invis = [(w, b) for w, b in wins
                 if b < med * thin_frac
                 and not TD.gap_overlaps(gaps, c, w, w + WINDOW_S)]
        sev = sum(med - b for _, b in invis) / (len(wins) * med)
        run = best = 0
        prev = None
        for w, _ in invis:
            run = run + 1 if prev is not None and w - prev == WINDOW_S else 1
            best = max(best, run)
            prev = w
        l1 = sev <= L1_SEVERITY_MAX
        l2 = best <= L2_RUN_WINDOWS_MAX
        coins[c] = {
            "status": "CONTENT_LIVE" if (l1 and l2) else "CONTENT_THIN",
            "n_windows": len(wins), "median_bytes": int(med),
            "n_invisible_thin": len(invis),
            "L1_severity": round(sev, 5), "L1_pass": l1,
            "L2_longest_run_windows": best, "L2_pass": l2,
            "failing_bars": [n for n, p in (("L1_severity", l1),
                                            ("L2_concentration", l2)) if not p],
        }

    judged = {c: v for c, v in coins.items()
              if v["status"] in ("CONTENT_LIVE", "CONTENT_THIN")}
    if not judged:
        # 0 of 0 passing is the empty-set trap; an unjudgeable day is a status.
        return {"day": day_token, "coins": coins,
                "status": "CONTENT_LIVENESS_UNJUDGEABLE",
                "why": "no coin had enough windows for a median",
                "n_coins_judged": 0, "governs": governs(day_token),
                "frozen_by_user": FROZEN_BY_USER}
    worst_c = max(judged, key=lambda c: (judged[c]["L1_severity"],
                                         judged[c]["L2_longest_run_windows"]))
    worst_r = max(judged, key=lambda c: judged[c]["L2_longest_run_windows"])
    thin = [c for c, v in judged.items() if v["status"] == "CONTENT_THIN"]
    return {
        "day": day_token,
        "status": "CONTENT_THIN" if thin else "CONTENT_LIVE",
        "coins": coins,
        "n_coins_judged": len(judged),
        "n_coins_unjudgeable": len(coins) - len(judged),
        "worst_L1_coin": worst_c,
        "worst_L1_severity": judged[worst_c]["L1_severity"],
        "worst_L2_coin": worst_r,
        "worst_L2_longest_run_windows": judged[worst_r]["L2_longest_run_windows"],
        "coins_thin": sorted(thin),
        "bars": {"L1_severity_max": L1_SEVERITY_MAX,
                 "L2_run_windows_max": L2_RUN_WINDOWS_MAX,
                 "thin_frac": thin_frac},
        # THE TWO FIELDS EVERY CONSUMER MUST READ FIRST.
        "governs": governs(day_token),
        "frozen_by_user": FROZEN_BY_USER,
        "effective_from_day": EFFECTIVE_FROM_DAY,
        "status_is_a_measurement_not_a_verdict": (
            "CONTENT_THIN reports that a day carries invisible content loss "
            "beyond the proposed bars. Whether that makes the day "
            "inadmissible is the coordinator's ruling (rule 14), and until "
            "FROZEN_BY_USER this rule has no bars in force at all."),
    }


def calibrate(days: list[str], gaps=None,
              raw_root: Path | None = None) -> dict[str, Any]:
    """The provenance table. REFUSES a day after CALIBRATION_MAX_DAY.

    Rule 11 as a check rather than a promise: a bar informed by a day it will
    later judge is not a bar. The refusal names the day, so a caller that
    tries cannot mistake the failure for a data problem.
    """
    late = sorted(d for d in days if d > CALIBRATION_MAX_DAY)
    if late:
        raise Refused(
            f"REFUSED to calibrate on {late}: every calibration day must be "
            f"<= {CALIBRATION_MAX_DAY}. A threshold informed by a day it is "
            f"meant to judge prospectively is chosen after seeing (rule 11), "
            f"and 2026-09-01 is the first forward day.")
    if not days:
        raise Refused("REFUSED: an empty calibration set. A bar derived from "
                      "no days is the empty-set trap, not a conservative bar.")
    gaps = TD.load_gaps() if gaps is None else gaps
    out = {}
    for d in days:
        r = measure_day(d, gaps=gaps, raw_root=raw_root)
        if r["status"] in ("CONTENT_LIVENESS_UNRESOLVED",
                           "CONTENT_LIVENESS_UNJUDGEABLE"):
            out[d] = {"status": r["status"]}
            continue
        out[d] = {"L1_severity": r["worst_L1_severity"],
                  "L2_longest_run_windows": r["worst_L2_longest_run_windows"],
                  "status": r["status"]}
    return {"days": out, "max_day_allowed": CALIBRATION_MAX_DAY,
            "n_days": len(out)}


def band(reading: dict[str, tuple], events: tuple) -> dict[str, Any]:
    """The empty band the bars sit in — computed, never asserted (rule 10)."""
    ev = [reading[d] for d in events]
    non = [v for d, v in reading.items() if d not in events]
    return {
        "non_event_max_L1": max(v[0] for v in non),
        "event_min_L1": min(v[0] for v in ev),
        "non_event_max_L2": max(v[1] for v in non),
        "event_min_L2": min(v[1] for v in ev),
        "L1_bar_inside_band": (max(v[0] for v in non) < L1_SEVERITY_MAX
                               < min(v[0] for v in ev)),
        "L2_bar_inside_band": (max(v[1] for v in non) < L2_RUN_WINDOWS_MAX
                               < min(v[1] for v in ev)),
        "n_non_event_days": len(non), "n_event_days": len(ev),
    }


# --------------------------------------------------------------------------
def selftest() -> int:
    import gzip
    import datetime as _dt
    import tempfile
    checks = 0
    positive_controls_run = 0

    def ok(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            print(f"FAIL: {label}")
            raise SystemExit(1)

    # ---- the pinned provenance must actually SUPPORT the bars -------------
    b = band(CALIBRATION_READING, CALIBRATION_KNOWN_EVENTS)
    ok(b["L1_bar_inside_band"] and b["L2_bar_inside_band"],
       "PROVENANCE: both bars sit strictly inside the empty band between the "
       "non-event days and the two known events — computed from the pinned "
       "reading, not asserted in prose")
    ok(b["n_non_event_days"] == 11 and b["n_event_days"] == 2
       and len(CALIBRATION_READING) == 13,
       "PROVENANCE: the band is computed over all 13 calibration days, so it "
       "cannot be a statement about a subset nobody counted")
    # A BAR IS ONLY UNFITTED IF A RANGE OF BARS AGREES. Executed.
    for cand in (0.05, 0.06, 0.08, 0.10, 0.12):
        cls = {d: (v[0] <= cand) for d, v in CALIBRATION_READING.items()}
        ok(sorted(d for d, p in cls.items() if not p)
           == sorted(CALIBRATION_KNOWN_EVENTS),
           f"PROVENANCE: an L1 bar of {cand} classifies all 13 days "
           f"identically — the answer does not depend on choosing 0.08")
    for cand in (6, 8, 12, 20, 39):
        cls = {d: (v[1] <= cand) for d, v in CALIBRATION_READING.items()}
        ok(sorted(d for d, p in cls.items() if not p)
           == sorted(CALIBRATION_KNOWN_EVENTS),
           f"PROVENANCE: an L2 bar of {cand} windows classifies all 13 days "
           f"identically")

    # ---- rule 11 is a REFUSAL, and it has a falsifier ---------------------
    try:
        calibrate(["20260830", "20260901"])
        ok(False, "RULE 11: calibrating on 09-01 must REFUSE")
    except Refused as e:
        ok("20260901" in str(e) and "after seeing" in str(e),
           "RULE 11 KNOWN-BAD: a calibration set containing 2026-09-01 "
           "REFUSES BY NAME — the claim that 09-01 did not inform the "
           "thresholds is a check, not a promise")
    try:
        calibrate([])
        ok(False, "an empty calibration set must REFUSE")
    except Refused as e:
        ok("empty-set trap" in str(e),
           "KNOWN-BAD: an EMPTY calibration set refuses — a bar from no days "
           "is not a conservative bar")
    ok(CALIBRATION_MAX_DAY < EFFECTIVE_FROM_DAY
       and "20260901" > CALIBRATION_MAX_DAY
       and "20260901" < EFFECTIVE_FROM_DAY,
       "RULE 11: 2026-09-01 lies strictly BETWEEN the last calibration day "
       "and the first governed day — it neither informs the bars nor is "
       "judged by them")

    # ---- governance follows the switch, in both conditions ----------------
    # (Originally asserted the DRAFT state, FROZEN_BY_USER is False. The USER
    # froze the rule 2026-09-01 (R-386), so the check now tests the INVARIANT
    # on both sides of the switch instead of pinning one side as the state.)
    _saved = globals()["FROZEN_BY_USER"]
    try:
        globals()["FROZEN_BY_USER"] = False
        ok(governs("20260902") is False and governs("20260901") is False,
           "GOVERNANCE: nothing governs while FROZEN_BY_USER is False")
        globals()["FROZEN_BY_USER"] = True
        ok(governs("20260902") is True and governs("20260901") is False
           and governs("20260831") is False,
           "GOVERNANCE POSITIVE CONTROL: with the freeze flipped the rule "
           "governs from 09-02 and STILL never reaches 09-01 or earlier — a "
           "switch that turns nothing on would prove nothing about the "
           "switch, and one that reaches backwards would void rule 11")
    finally:
        globals()["FROZEN_BY_USER"] = _saved
    ok(FROZEN_BY_USER is _saved,
       "GOVERNANCE: the freeze flag is restored to its committed state after "
       "the control (True since the R-386 USER freeze; the control never "
       "leaks its toggle)")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "raw"

        def build(day: str, spec: dict[int, int], coin: str = "btc"):
            """spec: window index -> lines. 5000 lines is a healthy window."""
            (root / day).mkdir(parents=True, exist_ok=True)
            base = int(_dt.datetime.strptime(day, "%Y%m%d").replace(
                tzinfo=_dt.timezone.utc).timestamp())
            for i, n in spec.items():
                w = base + i * WINDOW_S
                with gzip.open(root / day / f"{coin}-updown-5m-{w}.jsonl.gz",
                               "wb") as fh:
                    fh.write(b'{"x":1}\n' * n)
            return base

        # FALSIFIER: a SYNTHETIC QUIET DAY the rule MUST flag. 60 consecutive
        # windows (5 h) at 0.2% of median, no gap rows -- the 08-31 shape.
        quiet = {i: 5000 for i in range(288)}
        quiet.update({i: 10 for i in range(100, 160)})
        base_q = build("20260910", quiet)
        rq = measure_day("20260910", gaps={}, raw_root=root)
        ok(rq["status"] == "CONTENT_THIN"
           and rq["coins"]["btc"]["L1_pass"] is False
           and rq["coins"]["btc"]["L2_pass"] is False
           and rq["coins"]["btc"]["L2_longest_run_windows"] == 60,
           "FALSIFIER: a synthetic 5-hour quiet spell with NO gap rows is "
           "flagged CONTENT_THIN on BOTH bars — a detector that has never "
           "fired is not a detector (rule 15)")

        # POSITIVE CONTROL, synthetic: the same tape with NO quiet spell must
        # come back LIVE, or the flag above is the code shouting at everything.
        build("20260911", {i: 5000 for i in range(288)})
        rh = measure_day("20260911", gaps={}, raw_root=root)
        ok(rh["status"] == "CONTENT_LIVE"
           and rh["worst_L1_severity"] == 0.0
           and rh["worst_L2_longest_run_windows"] == 0,
           "POSITIVE CONTROL: a healthy synthetic day reads CONTENT_LIVE")
        positive_controls_run += 1

        # DISCRIMINATION: the IDENTICAL quiet day, once gap rows COVER the
        # spell, is ACCOUNTED loss and passes. The rule measures what the
        # ledger MISSED, not merely what was small.
        g = {"btc": [(base_q + 100 * WINDOW_S - 1,
                      base_q + 160 * WINDOW_S + 1)]}
        rq2 = measure_day("20260910", gaps=g, raw_root=root)
        ok(rq2["status"] == "CONTENT_LIVE"
           and rq2["coins"]["btc"]["n_invisible_thin"] == 0,
           "DISCRIMINATION: the same 5-hour spell WITH gap rows covering it "
           "is accounted loss and passes — P1/P2/P3 already charge for it, "
           "and charging twice would be a different defect")

        # L2 IS NOT REDUNDANT, driven rather than argued: a 1-hour TOTAL
        # blackout is 12 empty windows of 288 -> L1 4.17% (PASSES 8%) while
        # L2 = 12. At 13 windows L2 fails and L1 still passes.
        for n_out, want_l2 in ((12, True), (13, False)):
            spec = {i: 5000 for i in range(288)}
            spec.update({i: 1 for i in range(50, 50 + n_out)})
            day = f"2026092{n_out - 10}"
            build(day, spec)
            r = measure_day(day, gaps={}, raw_root=root)
            ok(r["coins"]["btc"]["L1_pass"] is True
               and r["coins"]["btc"]["L2_pass"] is want_l2,
               f"L2 IS LOAD-BEARING: a {n_out}-window total blackout passes "
               f"L1 (severity {r['coins']['btc']['L1_severity']:.4f} <= "
               f"{L1_SEVERITY_MAX}) and L2 is {want_l2} — the concentration "
               f"bar catches a shape the severity bar cannot")

        # REFUSALS ARE STATUSES (rule 4), and neither is a pass.
        r_absent = measure_day("20991231", gaps={}, raw_root=root)
        # governs=False here was a DRAFT-state pin (the fixture day is past
        # EFFECTIVE_FROM_DAY, so under the R-386 freeze it truthfully governs).
        # The invariant is the STATUS plus an honest governs field: an
        # UNRESOLVED status under a governing rule is a refusal, never a pass.
        ok(r_absent["status"] == "CONTENT_LIVENESS_UNRESOLVED"
           and r_absent.get("governs") == governs("20991231"),
           "KNOWN-BAD: an ABSENT day is UNRESOLVED, never CONTENT_LIVE — and "
           "its governs field reports the switch honestly")
        build("20260912", {i: 5000 for i in range(5)})
        r_few = measure_day("20260912", gaps={}, raw_root=root)
        ok(r_few["status"] == "CONTENT_LIVENESS_UNJUDGEABLE"
           and r_few["n_coins_judged"] == 0,
           "KNOWN-BAD: 5 windows is too few for a median — UNJUDGEABLE, not "
           "a clean day. 0 thin of 0 judged is the empty-set trap")

        # THE STATUS MUST NAME THE FAILING BAR, or CONTENT_THIN is a mood.
        ok(rq["coins"]["btc"]["failing_bars"] == ["L1_severity",
                                                  "L2_concentration"]
           and rh["coins"]["btc"]["failing_bars"] == [],
           "STATUS: a THIN coin names WHICH bars failed and a LIVE one names "
           "none — a status without its reason is a silent drop wearing a "
           "label")

    # ---- POSITIVE CONTROL ON THE REAL TAPE, when it is present ------------
    real = "20260827"
    if (TD.RAW / real).is_dir():
        rr = measure_day(real)
        ok(rr["status"] == "CONTENT_LIVE"
           and rr["worst_L1_severity"] <= L1_SEVERITY_MAX,
           f"POSITIVE CONTROL ON REAL DATA: {real}, a real busy day with 288 "
           f"windows on 7 coins, reads CONTENT_LIVE (worst L1 "
           f"{rr['worst_L1_severity']}) — the bars do not reject ordinary days")
        positive_controls_run += 1
        # THE PIN IS CHECKED ON A **PARTIAL** DAY, deliberately. Every other
        # real and synthetic leg holds 288 windows, where `len(wins)` and 288
        # are the same number -- so none of them can tell which denominator L1
        # uses. 08-19 has 115 windows, and its pinned 0.04290 is the value
        # ONLY the present-windows denominator produces (288 gives 0.01714).
        # Without this leg a denominator mutation survives the whole suite.
        if (TD.RAW / "20260819").is_dir():
            r19 = measure_day("20260819")
            ok(abs(r19["worst_L1_severity"]
                   - CALIBRATION_READING["20260819"][0]) <= 5e-5
               and r19["worst_L2_longest_run_windows"]
               == CALIBRATION_READING["20260819"][1],
               "PROVENANCE PINNED ON A PARTIAL DAY: 08-19 (115 windows) "
               "reproduces its pinned L1/L2 exactly, which fixes the "
               "present-windows denominator and would catch tape drift under "
               "the reading the bars were chosen from")
        ev = measure_day("20260831")
        ok(ev["status"] == "CONTENT_THIN" and ev["coins_thin"],
           "FALSIFIER ON REAL DATA: 2026-08-31, the ~4h quiet event this rule "
           "exists for, is flagged CONTENT_THIN on the real tape")
    ok(positive_controls_run >= 1,
       "a suite whose positive controls were all skipped proves nothing: at "
       "least one ADMIT case must actually have run")

    print(f"da_content_liveness_rule selftests: {checks} checks passed "
          f"({positive_controls_run} positive control(s) executed)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--day")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.draft:
        print(DRAFT)
        return 0
    if a.calibrate:
        days = [d for d in TD.all_days() if d <= CALIBRATION_MAX_DAY]
        c = calibrate(days)
        print(json.dumps(c, indent=1))
        live = {d: (v.get("L1_severity"), v.get("L2_longest_run_windows"))
                for d, v in c["days"].items() if "L1_severity" in v}
        drift = {d: (live[d], CALIBRATION_READING.get(d))
                 for d in live
                 if d in CALIBRATION_READING
                 and (abs(live[d][0] - CALIBRATION_READING[d][0]) > 5e-5
                      or live[d][1] != CALIBRATION_READING[d][1])}
        print(f"\npinned as-of {CALIBRATION_AS_OF}; "
              f"{len(drift)} day(s) DRIFTED from the pin: {drift}")
        print(json.dumps(band(CALIBRATION_READING, CALIBRATION_KNOWN_EVENTS),
                         indent=1))
        return 1 if drift else 0
    if not a.day:
        print(DRAFT)
        return 0
    r = measure_day(a.day)
    print(json.dumps(r, indent=1) if a.json else
          json.dumps({k: v for k, v in r.items() if k != "coins"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
