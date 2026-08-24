"""Will the NEXT day fit its memory envelope? Answer BEFORE running it.

From `OP_PLANE_PLAN` §8e's never-attempted audit. Today's cap-adjacent failure
cost 162 minutes and a blocked lane to discover something a pre-flight would have
said in a second. `LANE_PROGRESS` reports what happened; nothing predicted.

**This is a WEAK instrument and it says so.** It rests on TWO measured coin-days,
one of which is CENSORED at `memory.max`, so the memory exponent is a LOWER
BOUND and the true peak for 08-21 is unknown. It therefore reports a RANGE and a
three-way verdict, and it will not return FITS on a censored extrapolation.
Overstating here would be worse than not predicting at all.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

DATA = Path("/home/yuqing/ctaNew/data/pm_5min")
MEMORY_MAX_BYTES = 16 * 2**30

# The two measured Tier-2-only runs. Both single-phase, on a pre-built full
# batch, so they are like-for-like. Provenance is systemd's own accounting.
#   08-20: Consumed 2min 51.457s CPU, 13.6G peak   UNTHROTTLED, complete
#   08-21: Consumed 3min 18.903s CPU, 16.0G peak   UNTHROTTLED, complete,
#          but peak PINNED AT memory.max -> CENSORED, true peak unknown/higher
# AS-OF 2026-08-24 (R-104(3): every cited population carries its n AND its as-of).
# n = 3 Tier-2-only runs, all UNTHROTTLED. RECALIBRATED after 08-22 landed clean:
# the fit now rests on the two UNCENSORED points. The previous calibration used
# the 08-21 pair whose peak was PINNED AT THE CAP, which read a censored
# measurement as a scaling law and over-stated the memory exponent (>=1.52 vs the
# 1.05 the uncensored pair gives).
POINTS = [
    {"day": "20260822", "bytes": None, "cpu_s": 139.0, "peak_gb": 11.87, "censored": False},
    {"day": "20260820", "bytes": None, "cpu_s": 171.5, "peak_gb": 13.60, "censored": False},
]
# Held out, NOT fitted: its peak is censored at the 16 GB cap so it is a LOWER
# bound. The fit is checked against it and inflated if it under-predicts -- a
# censored point cannot calibrate, but it can still falsify.
CENSORED_CHECK = {"day": "20260821", "bytes": None, "peak_gb_at_least": 16.0}
AS_OF = "2026-08-24"


def day_bytes(day: str) -> int:
    d = DATA / "raw" / day
    if not d.is_dir():
        return 0
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file())


def fit_exponent(x0: float, y0: float, x1: float, y1: float) -> float:
    if x0 <= 0 or x1 <= 0 or y0 <= 0 or y1 <= 0 or x0 == x1:
        return float("nan")
    return math.log(y1 / y0) / math.log(x1 / x0)


def verdict_for(target: float, ref_lo: dict, ref_hi: dict) -> tuple[str, float, float]:
    """Pure verdict logic, so it can be tested without touching disk."""
    k_cpu = fit_exponent(ref_lo["bytes"], ref_lo["cpu_s"], ref_hi["bytes"], ref_hi["cpu_s"])
    k_mem = fit_exponent(ref_lo["bytes"], ref_lo["peak_gb"], ref_hi["bytes"], ref_hi["peak_gb"])
    ratio = target / ref_lo["bytes"]
    cpu = ref_lo["cpu_s"] * ratio ** k_cpu
    mem = ref_lo["peak_gb"] * ratio ** k_mem
    cap = MEMORY_MAX_BYTES / 2**30
    if mem >= cap:
        return "WILL_NOT_FIT", cpu, mem
    if target > ref_hi["bytes"] and ref_hi["censored"]:
        return "MARGINAL", cpu, mem
    if mem > 0.85 * cap:
        return "MARGINAL", cpu, mem
    return "FITS", cpu, mem


def selftest() -> int:
    """BE's lesson, applied here: verifying a tool's FORM is not verifying it
    WORKS. This tool's landing-evidence claim used to be `grep -c VERDICT` — it
    would have passed on output reading "VERDICT: banana". These drive the
    verdict LOGIC instead, including the safety rule that matters most: it must
    REFUSE `FITS` on any upward extrapolation from a censored reference."""
    lo = {"bytes": 5.60 * 2**30, "cpu_s": 171.5, "peak_gb": 13.6, "censored": False}
    hi = {"bytes": 6.23 * 2**30, "cpu_s": 198.9, "peak_gb": 16.0, "censored": True}
    cases = [
        ("smaller day fits",                    lo, hi, 4.92 * 2**30, "FITS"),
        ("at the censored reference",           lo, hi, 6.23 * 2**30, "WILL_NOT_FIT"),
        ("above the censored reference",        lo, hi, 6.50 * 2**30, "WILL_NOT_FIT"),
        ("far above",                           lo, hi, 9.00 * 2**30, "WILL_NOT_FIT"),
    ]
    # The censored-refusal branch is UNREACHABLE with the real references (see
    # the note below), so it is exercised against synthetic ones. A branch that
    # cannot fire is not a guard -- this proves the logic works IF the
    # calibration ever makes it reachable.
    lo2 = {"bytes": 5.0 * 2**30, "cpu_s": 100.0, "peak_gb": 8.0, "censored": False}
    hi2 = {"bytes": 6.0 * 2**30, "cpu_s": 120.0, "peak_gb": 9.0, "censored": True}
    cases.append(("censored-refusal branch (synthetic refs)", lo2, hi2, 6.3 * 2**30, "MARGINAL"))
    ok = 0
    for name, a, b, target, want in cases:
        got, _, mem = verdict_for(target, a, b)
        good = got == want
        ok += good
        print(f"  {'PASS' if good else 'FAIL'}  {name:42s} {got:13s} (want {want}) peak~{mem:.1f}GB")
    print(f"  {ok}/{len(cases)} verdict-logic cases")
    print("  NOTE: with the REAL references the censored reference already sits AT")
    print("        the 16 GB cap, so every upward extrapolation returns WILL_NOT_FIT")
    print("        and the MARGINAL censored-refusal branch is UNREACHABLE today.")
    return 0 if ok == len(cases) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", help="UTC day as YYYYMMDD")
    ap.add_argument("--selftest", action="store_true",
                    help="drive the verdict logic, not just its output shape")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.day:
        print("  FAIL  --day or --selftest required"); return 2
    if not re.fullmatch(r"\d{8}", args.day):
        print("  FAIL  --day must be YYYYMMDD")
        return 2

    for p in POINTS:
        p["bytes"] = day_bytes(p["day"])
    CENSORED_CHECK["bytes"] = day_bytes(CENSORED_CHECK["day"])
    a, b = POINTS
    if not a["bytes"] or not b["bytes"]:
        print("  FAIL  reference days missing from raw/ — cannot calibrate")
        return 2

    target = day_bytes(args.day)
    if not target:
        print(f"  FAIL  raw/{args.day} absent or empty — nothing to predict")
        return 2

    k_cpu = fit_exponent(a["bytes"], a["cpu_s"], b["bytes"], b["cpu_s"])
    k_mem = fit_exponent(a["bytes"], a["peak_gb"], b["bytes"], b["peak_gb"])
    ratio = target / a["bytes"]
    cpu_pred = a["cpu_s"] * ratio ** k_cpu
    mem_raw = a["peak_gb"] * ratio ** k_mem
    # Falsify the fit against the held-out CENSORED point. Its true peak is at
    # least 16 GB, so if the fit predicts less it under-predicts a day we have
    # actually seen -- inflate every prediction by that shortfall. A fit that is
    # known to be low must not be used low.
    cc_ratio = CENSORED_CHECK["bytes"] / a["bytes"] if CENSORED_CHECK["bytes"] else None
    margin = 1.0
    if cc_ratio:
        cc_pred = a["peak_gb"] * cc_ratio ** k_mem
        if cc_pred < CENSORED_CHECK["peak_gb_at_least"]:
            margin = CENSORED_CHECK["peak_gb_at_least"] / cc_pred
    mem_pred = mem_raw * margin
    cap_gb = MEMORY_MAX_BYTES / 2**30

    print(f"  target day        {args.day}   {target/2**30:.2f} GB raw tape")
    print(f"  reference         {a['day']} {a['bytes']/2**30:.2f} GB -> {a['cpu_s']:.1f}s / {a['peak_gb']:.1f} GB")
    print(f"                    {b['day']} {b['bytes']/2**30:.2f} GB -> {b['cpu_s']:.1f}s / {b['peak_gb']:.1f} GB"
          f"{'  [CENSORED at memory.max]' if b['censored'] else ''}")
    print(f"  exponent  cpu     {k_cpu:.2f}   (upper bound: reclaim CPU is included)")
    print(f"  exponent  memory  {k_mem:.2f}   {'LOWER BOUND — reference peak is censored' if b['censored'] else ''}")
    print(f"  as-of / basis     {AS_OF}   n=2 uncensored fit + 1 censored held-out check")
    print(f"  predicted cpu     {cpu_pred:.0f} s")
    print(f"  predicted peak    {mem_pred:.1f} GB   (raw fit {mem_raw:.1f} x margin {margin:.3f})   vs cap {cap_gb:.0f} GB")

    # Three-way verdict. A censored memory exponent can only ever UNDERSTATE the
    # peak, so FITS is refused whenever the extrapolation is upward from the
    # censored reference -- the direction in which being wrong is expensive.
    extrapolating_up = target > b["bytes"]
    if mem_pred >= cap_gb:
        verdict, why = "WILL_NOT_FIT", "predicted peak is at or above the cap"
    elif extrapolating_up and b["censored"]:
        verdict, why = ("MARGINAL",
                        "larger than the censored reference; the memory exponent "
                        "is a lower bound, so the true peak may exceed this")
    elif mem_pred > 0.85 * cap_gb:
        verdict, why = "MARGINAL", "within 15% of the cap"
    else:
        verdict, why = "FITS", "predicted peak is comfortably below the cap"
    print(f"  VERDICT           {verdict}   ({why})")
    print("  NOTE              two reference points, one censored. This is a")
    print("                    weak instrument; treat MARGINAL as 'measure it', "
          "not 'probably fine'.")
    return 0 if verdict == "FITS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
