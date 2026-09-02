#!/usr/bin/env python3
"""CONTENT-LIVENESS RULE v2 — the proposed ABSOLUTE FLOOR, and its checker.

DRAFT-FOR-USER-FREEZE. See `plans/DA_CONTENT_LIVENESS_RULE_V2_AMENDMENT.md`.
**This file governs nothing and wires into nothing.** It does not import or
modify `da_content_liveness_rule`, and no verdict path calls it. Wiring
follows the freeze, never precedes it (rule 14; and the dispatch says the
draft is a document plus its checker).

WHAT IT CLOSES (reviewer RR6-1, quoted verbatim in the amendment): v1
classifies a window as thin RELATIVE TO THE SAME DAY'S MEDIAN. Past roughly
60% dark the median itself crosses into the dark regime, every dark window
stops being "thin", and **a day 100% dark reads CONTENT_LIVE with L1 = 0 and
L2 = 0.** The only absolute floor in v1 is `median <= 0`.

THE AMENDMENT, in one sentence: keep v1 exactly as frozen, and add ONE
predicate that asks the same question against a reference THE DAY UNDER TEST
CANNOT MOVE -- the trailing median of prior complete days.

IT INTRODUCES NO NEW MEASUREMENT THRESHOLD. `V2_DARK_FRAC` IS v1's
`THIN_FRAC` and `V2_RUN_MAX` IS v1's `L2_RUN_WINDOWS_MAX`. Only the
DENOMINATOR changes. The two genuinely new numbers are structural (how far
back the reference looks, and how many prior days it needs), and both are
calibrated on days <= 2026-08-31 only.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_tape_density as TD                                   # noqa: E402

WINDOW_S = TD.WINDOW_S

#: REUSED FROM v1, NOT RE-CHOSEN. Same fraction, same run bar; the amendment
#: changes what the fraction is OF.
V2_DARK_FRAC = TD.THIN_FRAC                 # 0.05
V2_RUN_MAX = 12                             # v1's L2_RUN_WINDOWS_MAX

#: THE TWO NEW NUMBERS, both structural and both calibrated on <= 08-31.
#: K=7: with a median-of-priors reference, the reference only turns dark once
#: MORE THAN HALF the trailing window is dark, so v2 is robust to up to THREE
#: consecutive fully dark days and degrades on the fourth. That bound is
#: computable, is stated in the amendment, and is NOT guarded -- a guard that
#: cannot fire is not a guard.
V2_TRAILING_DAYS = 7
#: 3 priors is where the reference stops moving: measured on <= 08-31, the
#: day-median / trailing-reference ratio spans 0.396..1.360 over 70 coin-days
#: once at least three priors exist.
V2_MIN_REFERENCE_DAYS = 3

CALIBRATION_MAX_DAY = "20260831"

#: Statuses EXTEND v1's vocabulary; none is replaced. A day carries its v1
#: status AND its v2 status, and the composite is the more severe.
STATUS_DARK = "CONTENT_DARK"
STATUS_NO_REF = "CONTENT_LIVENESS_NO_REFERENCE"


class Refused(Exception):
    """A population this checker must not summarise."""


def _runs(sorted_windows: list[int]) -> int:
    best = run = 0
    prev = None
    for w in sorted_windows:
        run = run + 1 if prev is not None and w - prev == WINDOW_S else 1
        best = max(best, run)
        prev = w
    return best


def day_medians(days: list[str], raw_root: Path | None = None
                ) -> dict[tuple[str, str], float]:
    out = {}
    for d in days:
        try:
            agg = TD.scan_day(d, TD.RAW if raw_root is None else raw_root)
        except TD.Refused:
            continue
        per = collections.defaultdict(list)
        for (c, w), b in agg.items():
            per[c].append(b)
        for c, v in per.items():
            if len(v) >= TD.MIN_WINDOWS_FOR_MEDIAN:
                out[(d, c)] = statistics.median(v)
    return out


def trailing_reference(day: str, coin: str, all_days: list[str],
                       medians: dict) -> float | None:
    """Median of `coin`'s daily medians over the prior complete days.

    POINT-IN-TIME BY CONSTRUCTION: only days STRICTLY BEFORE `day` are read,
    so the day under test cannot move its own reference. That is the whole
    amendment -- v1's denominator is the day itself, which a blackout drags
    down with it.
    """
    i = all_days.index(day) if day in all_days else len(all_days)
    prior = [medians[(d, coin)] for d in all_days[max(0, i - V2_TRAILING_DAYS):i]
             if (d, coin) in medians]
    if len(prior) < V2_MIN_REFERENCE_DAYS:
        return None
    return statistics.median(prior)


def measure_v2(day: str, all_days: list[str], medians: dict, gaps=None,
               raw_root: Path | None = None) -> dict[str, Any]:
    """v1's L1/L2 AND the new absolute-floor L3, side by side."""
    gaps = TD.load_gaps() if gaps is None else gaps
    try:
        agg = TD.scan_day(day, TD.RAW if raw_root is None else raw_root)
    except TD.Refused as e:
        return {"day": day, "status_v2": "CONTENT_LIVENESS_UNRESOLVED",
                "why": str(e)}
    per = collections.defaultdict(list)
    for (c, w), b in agg.items():
        per[c].append((w, b))
    coins: dict[str, Any] = {}
    for c, wins in sorted(per.items()):
        wins.sort()
        if len(wins) < TD.MIN_WINDOWS_FOR_MEDIAN:
            coins[c] = {"status_v1": "CONTENT_LIVENESS_UNJUDGEABLE",
                        "status_v2": "CONTENT_LIVENESS_UNJUDGEABLE",
                        "n_windows": len(wins)}
            continue
        med = statistics.median([b for _, b in wins])
        ref = trailing_reference(day, c, all_days, medians)
        vis = [(w, b) for w, b in wins
               if not TD.gap_overlaps(gaps, c, w, w + WINDOW_S)]
        # v1: relative to THIS day's median.
        l2 = _runs([w for w, b in vis if med > 0 and b < med * V2_DARK_FRAC])
        # v2: relative to a reference this day cannot move.
        if ref is None:
            coins[c] = {"status_v1": "CONTENT_THIN" if l2 > V2_RUN_MAX
                        else "CONTENT_LIVE",
                        "status_v2": STATUS_NO_REF, "L2_v1_run": l2,
                        "median_bytes": int(med), "reference_bytes": None,
                        "why": f"fewer than {V2_MIN_REFERENCE_DAYS} prior "
                               f"complete days; NO_REFERENCE is not a pass"}
            continue
        l3 = _runs([w for w, b in vis if b < ref * V2_DARK_FRAC])
        dark_share = sum(1 for _, b in vis if b < ref * V2_DARK_FRAC) / len(vis)
        coins[c] = {
            "status_v1": "CONTENT_THIN" if l2 > V2_RUN_MAX else "CONTENT_LIVE",
            "status_v2": STATUS_DARK if l3 > V2_RUN_MAX else "CONTENT_LIVE",
            "median_bytes": int(med), "reference_bytes": int(ref),
            "median_over_reference": round(med / ref, 4) if ref else None,
            "L2_v1_run": l2, "L3_v2_run": l3,
            "L3_v2_dark_share": round(dark_share, 4),
            "v2_catches_what_v1_misses": l3 > V2_RUN_MAX >= l2,
        }
    judged = {c: v for c, v in coins.items() if "L3_v2_run" in v}
    return {
        "day": day, "coins": coins, "n_coins_judged": len(judged),
        "worst_L2_v1": max([v["L2_v1_run"] for v in judged.values()] or [0]),
        "worst_L3_v2": max([v["L3_v2_run"] for v in judged.values()] or [0]),
        "status_v1": ("CONTENT_THIN"
                      if any(v["status_v1"] == "CONTENT_THIN"
                             for v in coins.values()) else "CONTENT_LIVE"),
        "status_v2": (STATUS_DARK
                      if any(v.get("status_v2") == STATUS_DARK
                             for v in coins.values()) else
                      STATUS_NO_REF
                      if judged == {} and coins else "CONTENT_LIVE"),
        "amendment_changes_this_day": any(
            v.get("v2_catches_what_v1_misses") for v in judged.values()),
        "governs": False,
        "frozen_by_user": False,
        "note": ("v2 is a DRAFT. It governs nothing, wires into nothing, and "
                 "does not alter v1's reading of any day -- it reports a "
                 "SECOND status beside it."),
    }


def calibrate(days: list[str], raw_root: Path | None = None,
              gaps=None) -> dict[str, Any]:
    """REFUSES any day after CALIBRATION_MAX_DAY (rule 11).

    09-01 and 09-02 are SEEN days: 09-02 is the event that motivated this
    amendment, and calibrating on the day that motivated the rule is the
    error the whole structure exists to avoid. E1 (08-26) and E2 (08-31) are
    the anchors and both predate the boundary.
    """
    late = sorted(d for d in days if d > CALIBRATION_MAX_DAY)
    if late:
        raise Refused(
            f"REFUSED to calibrate on {late}: every calibration day must be "
            f"<= {CALIBRATION_MAX_DAY}. 09-02 is the event that MOTIVATED "
            f"this amendment and may be cited, never calibrated on -- "
            f"choosing a threshold on the day that prompted it is rule 11 in "
            f"one move.")
    if not days:
        raise Refused("REFUSED: an empty calibration set is the empty-set "
                      "trap, not a conservative bar.")
    med = day_medians(days, raw_root)
    out = {}
    for d in days:
        r = measure_v2(d, days, med, gaps=gaps, raw_root=raw_root)
        out[d] = {"worst_L2_v1": r.get("worst_L2_v1"),
                  "worst_L3_v2": r.get("worst_L3_v2"),
                  "agree": r.get("worst_L2_v1") == r.get("worst_L3_v2"),
                  "amendment_changes_this_day":
                      r.get("amendment_changes_this_day")}
    ratios = []
    for d in days:
        for c in {c for (dd, c) in med if dd == d}:
            ref = trailing_reference(d, c, days, med)
            if ref:
                ratios.append(round(med[(d, c)] / ref, 4))
    return {"days": out, "n_days": len(out),
            "n_days_the_amendment_changes": sum(
                1 for v in out.values() if v["amendment_changes_this_day"]),
            "legitimate_ratio_n": len(ratios),
            "legitimate_ratio_min": min(ratios) if ratios else None,
            "legitimate_ratio_median": statistics.median(ratios) if ratios
            else None,
            "legitimate_ratio_max": max(ratios) if ratios else None,
            "headroom_min_ratio_over_dark_frac": round(
                min(ratios) / V2_DARK_FRAC, 1) if ratios else None}


# --------------------------------------------------------------------------
def selftest() -> int:
    import gzip
    import tempfile
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            print(f"FAIL: {label}")
            raise SystemExit(1)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "raw"

        def mk(day, per_window):
            (root / day).mkdir(parents=True, exist_ok=True)
            base = int(dt.datetime.strptime(day, "%Y%m%d").replace(
                tzinfo=dt.timezone.utc).timestamp())
            for i in range(288):
                with gzip.open(root / day /
                               f"btc-updown-5m-{base + i * WINDOW_S}.jsonl.gz",
                               "wb") as fh:
                    fh.write(b'{"x":1}\n' * per_window(i))

        prior = ["2026080%d" % i for i in range(1, 6)]
        for d in prior:
            mk(d, lambda i: 5000)

        # ---- THE AMENDMENT, IN ONE PAIR OF DAYS --------------------------
        # (1) A 100% DARK day. v1 sees nothing; v2 must see it.
        dark = "20260806"
        mk(dark, lambda i: 3)
        days = prior + [dark]
        med = day_medians(days, root)
        r = measure_v2(dark, days, med, gaps={}, raw_root=root)
        b = r["coins"]["btc"]
        ok(b["status_v1"] == "CONTENT_LIVE" and b["L2_v1_run"] == 0,
           "FALSIFIER, v1 HALF: a 100% DARK day reads CONTENT_LIVE under v1 "
           "with run 0 -- the blind spot RR6-1 names, reproduced rather than "
           "quoted")
        ok(b["status_v2"] == STATUS_DARK and b["L3_v2_run"] == 288
           and b["v2_catches_what_v1_misses"] is True,
           "FALSIFIER, v2 HALF: the SAME day reads CONTENT_DARK under v2 with "
           "a 288-window run. THAT DIFFERENCE IS THE AMENDMENT")
        ok(b["median_over_reference"] is not None
           and b["median_over_reference"] < 0.01,
           "and the day's own median is <1% of its trailing reference, which "
           "is exactly why a same-day denominator cannot see it")

        # (2) A GENUINELY QUIET but HONEST day must stay LIVE under BOTH.
        # 40% of normal everywhere -- below the 0.396 quietest coin-day
        # measured on the calibration set, so this is a harder case than any
        # real quiet day in the record.
        quiet = "20260807"
        mk(quiet, lambda i: 2000)
        days2 = prior + [quiet]
        med2 = day_medians(days2, root)
        r2 = measure_v2(quiet, days2, med2, gaps={}, raw_root=root)
        q = r2["coins"]["btc"]
        ok(q["status_v1"] == "CONTENT_LIVE" and q["status_v2"] == "CONTENT_LIVE"
           and q["L3_v2_run"] == 0,
           "FALSIFIER, THE OTHER WAY: a genuinely quiet day at 40% of normal "
           "stays CONTENT_LIVE under BOTH -- a slow venue hour is not a "
           "blackout, and an amendment that failed honest quiet days would "
           "be worse than the blind spot")
        ok(0.35 < q["median_over_reference"] < 0.45,
           "and its median/reference ratio lands where it was built to "
           "(~0.40), below the quietest real coin-day in the calibration set")

        # (3) A PARTIAL blackout: BOTH must agree, or the amendment is not
        # additive -- it would be re-judging days v1 already rules on.
        part = "20260808"
        mk(part, lambda i: 3 if 100 <= i < 130 else 5000)
        days3 = prior + [part]
        med3 = day_medians(days3, root)
        r3 = measure_v2(part, days3, med3, gaps={}, raw_root=root)
        p = r3["coins"]["btc"]
        ok(p["status_v1"] == "CONTENT_THIN" and p["status_v2"] == STATUS_DARK
           and p["L2_v1_run"] == 30 and p["L3_v2_run"] == 30
           and p["v2_catches_what_v1_misses"] is False,
           "ADDITIVITY: on a PARTIAL blackout v1 and v2 agree exactly (both "
           "run 30) -- v2 EXTENDS coverage into the case v1 cannot see and "
           "does not re-judge the ones it can")

        # (4) NO REFERENCE is a status, never a pass.
        solo = "20260801"
        r4 = measure_v2(solo, [solo], day_medians([solo], root), gaps={},
                        raw_root=root)
        ok(r4["coins"]["btc"]["status_v2"] == STATUS_NO_REF,
           "KNOWN-BAD: with too few prior days v2 reports NO_REFERENCE -- an "
           "absent reference is a status, never a clean day")

        # (5) RULE 11 IS A REFUSAL, with a falsifier.
        try:
            calibrate(["20260830", "20260902"], raw_root=root)
            ok(False, "calibrating on 09-02 must REFUSE")
        except Refused as e:
            ok("20260902" in str(e) and "MOTIVATED" in str(e),
               "RULE 11 KNOWN-BAD: a calibration set containing 09-02 -- the "
               "day that motivated this amendment -- REFUSES BY NAME")
        try:
            calibrate([], raw_root=root)
            ok(False, "an empty calibration set must REFUSE")
        except Refused as e:
            ok("empty-set trap" in str(e), "KNOWN-BAD: empty set refuses")

        # (6) THE THRESHOLDS ARE REUSED, not re-chosen -- asserted, so a
        # future edit that quietly forks them fails here.
        ok(V2_DARK_FRAC == TD.THIN_FRAC,
           "the dark fraction IS v1's thin fraction; the amendment changes "
           "the DENOMINATOR, not the bar")
        ok(V2_RUN_MAX == 12,
           "the run bar IS v1's frozen L2_RUN_WINDOWS_MAX")

        # (7) THE REFERENCE IS POINT-IN-TIME: a LATER day cannot move it.
        ref_before = trailing_reference(dark, "btc", days, med)
        days_more = days + ["20260809"]
        mk("20260809", lambda i: 1)
        med_more = day_medians(days_more, root)
        ok(trailing_reference(dark, "btc", days_more, med_more) == ref_before,
           "POINT-IN-TIME: adding a LATER (and fully dark) day does not move "
           "an earlier day's reference -- the reference reads only days "
           "strictly before the day under test")

    print(f"da_content_liveness_v2_check selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--day")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    days = [d for d in TD.all_days() if d <= CALIBRATION_MAX_DAY]
    if a.calibrate:
        print(json.dumps(calibrate(days), indent=1))
        return 0
    if a.day:
        all_d = [d for d in TD.all_days() if d <= a.day]
        print(json.dumps(measure_v2(a.day, all_d, day_medians(all_d)), indent=1))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
