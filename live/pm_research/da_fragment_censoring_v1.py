"""Fragment admissibility / censoring receipt for the R-293 diagnostic.

AUTHORISATION: coordinator dispatch under R-293 (the USER ordered the
diagnostic; costs stated and accepted). This receipt runs BEFORE any score is
read, and its job is to make the CENSORING VISIBLE -- not to re-litigate the
order, which is the user's to give.

WHAT THIS IS NOT. Nothing here scores anything, and no outcome of the
diagnostic can change what this says. The two fragments are **INADMISSIBLE FOR
THE RACE** under the pre-registered admission rule regardless of what the
diagnostic shows, for reasons that are properties of the fragments themselves:

  * neither is a complete UTC day, and the rule counts COMPLETE days (rule 8:
    below G=5 complete clusters, point estimate only);
  * fragment A begins mid-day at the freeze epoch, so it is a SELECTED slice of
    a day whose other 74 windows are excluded by construction;
  * both carry the btc feed loss quantified below, and that loss is
    burst-concentrated -- the missing minutes are plausibly the most
    decision-relevant ones, the bias direction is unknown, and it is plausibly
    flattering.

Those three hold whether the diagnostic reads positive, negative or null.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import da_forward_day_verify as V                      # noqa: E402
import da_hf_pm_alignment as A                        # noqa: E402

WINDOW_S = 300
COIN = "btc"
FREEZE_EPOCH = 1787897340          # 2026-08-28T06:09:00Z, the ruled freeze commit
FRAG_A = (FREEZE_EPOCH, 1787961600)          # 08-28 06:09:00Z -> 24:00:00Z
FRAG_B_START = 1787961600                    # 08-29 00:00:00Z
HF_ERA_FLOOR_NS = 1787579334881534478        # hf_ws_v2 stamp boundary


def utc(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat()[:19] + "Z"


def observed_windows(first_full: int, last_full: int) -> int:
    """btc windows ACTUALLY PRESENT in the tape over [first_full, last_full).

    WITHOUT THIS THE RECEIPT COULD NOT TELL A CLEAN FRAGMENT FROM A DEAD
    COLLECTOR. Everything else here is derived from the GAP LEDGER, and a
    collector that never ran writes no gaps -- so an unobserved fragment would
    have reported 0% affected and zero lost seconds: a perfect fragment. That
    is the same "silence from a dead collector and silence from a perfect one
    are the same bytes" rule the day-bar already enforces, and it was missing
    from the receipt that gates the diagnostic.

    On this data the answer is 214/214 and 39/39, so nothing reported changes
    -- but it was true by luck rather than by check, and the next fragment is
    the one where that matters.
    """
    days = sorted({dt.datetime.fromtimestamp(t, dt.timezone.utc)
                   .strftime("%Y%m%d")
                   for t in (first_full, max(first_full, last_full - 1))})
    got = A.pm_windows(days).get(COIN, [])
    return len([x for x in got if first_full <= x < last_full])


def fragment_report(lo: int, hi: int, label: str) -> dict:
    """Per-window coverage and censoring over [lo, hi).

    PARTIAL BOUNDARY WINDOWS ARE A STATUS, NOT A SILENT CHOICE. Fragment A
    starts at 06:09:00Z, which is not 300s-aligned, so the window straddling
    the freeze is neither wholly in nor wholly out. It is counted as
    `boundary_partial` and EXCLUDED from the contained set -- named, so nobody
    has to reverse-engineer which convention was used.
    """
    ivs = V.coin_gap_intervals(lo, hi, COIN, None, {})
    first_full = -(-lo // WINDOW_S) * WINDOW_S
    last_full = (hi // WINDOW_S) * WINDOW_S
    boundary_partial = int(first_full != lo) + int(last_full != hi)

    per_hour, affected, lost_total = {}, 0, 0.0
    n = 0
    worst_window, worst_lost = None, 0.0
    per_window_lost = []
    for w0 in range(first_full, last_full, WINDOW_S):
        w1 = w0 + WINDOW_S
        n += 1
        cov = sum(min(b, w1) - max(a, w0) for a, b in ivs if a < w1 and b > w0)
        if cov > 0:
            affected += 1
            lost_total += cov
            per_window_lost.append(cov)
            h = utc(w0)[11:13]
            per_hour[h] = per_hour.get(h, 0) + 1
            if cov > worst_lost:
                worst_lost, worst_window = cov, utc(w0)
    span_h = (last_full - first_full) / 3600.0
    observed = observed_windows(first_full, last_full)
    # BURST CONCENTRATION, MEASURED NOT ASSERTED. R-293's censoring statement
    # says the loss is burst-concentrated; a receipt that merely repeats that
    # is a printed conclusion beside a table (rule 10). So: what share of the
    # lost seconds sits in the worst DECILE of affected windows, and how long
    # is a typical outage. A uniform drizzle would put ~10% there; a burst
    # regime puts far more.
    srt = sorted(per_window_lost, reverse=True)
    k = max(1, len(srt) // 10)
    top_share = (sum(srt[:k]) / lost_total * 100.0) if lost_total else None
    durs = sorted(b - a for a, b in ivs)
    med = durs[len(durs) // 2] if durs else None
    return {
        "burst_concentration": {
            "affected_windows": len(srt),
            "worst_decile_windows": k,
            "worst_decile_share_of_lost_seconds_pct":
                round(top_share, 1) if top_share is not None else None,
            "uniform_would_be_pct": 10.0,
            "median_interval_s": round(med, 2) if med is not None else None,
            "max_interval_s": round(durs[-1], 2) if durs else None,
            "interpretation_is_the_readers":
                "a uniform drizzle would put ~10% of lost seconds in the worst "
                "decile; this number is reported so the burst claim is "
                "checkable rather than repeated",
        },
        "label": label,
        "bounds_utc": [utc(lo), utc(hi)],
        "bounds_epoch": [lo, hi],
        "contained_windows": n,
        "windows_observed_in_tape": observed,
        "windows_missing_from_tape": n - observed,
        "coverage_verified": observed == n,
        "coverage_note":
            "every other number here derives from the GAP LEDGER, and a "
            "collector that never ran writes no gaps -- so without this check "
            "an unobserved fragment reports 0% affected and zero loss, i.e. a "
            "perfect fragment. Zero missing means the unaffected windows were "
            "genuinely OBSERVED, not merely unmentioned",
        "boundary_partial_windows": boundary_partial,
        "boundary_partial_note":
            "windows straddling a fragment edge are EXCLUDED from the "
            "contained set and counted here, so the convention is named "
            "rather than inferred",
        "gap_intervals": len(ivs),
        "windows_gap_affected": affected,
        "windows_gap_affected_pct": round(100.0 * affected / n, 2) if n else None,
        "lost_seconds": round(lost_total, 1),
        # TWO NUMBERS, BOTH TRUE, ANSWERING DIFFERENT QUESTIONS. "63% of
        # windows affected" and "2.8% of window-time lost" describe the same
        # fragment. The first governs WINDOW-LEVEL admissibility (a window
        # touched at all is a window whose decisions may be incomplete); the
        # second governs DATA VOLUME. Reporting either alone misleads in a
        # predictable direction, so both are here.
        "window_time_lost_pct": round(
            100.0 * lost_total / (n * WINDOW_S), 2) if n else None,
        "window_seconds_total": n * WINDOW_S,
        "lost_s_per_hour": round(lost_total / span_h, 1) if span_h else None,
        "span_hours": round(span_h, 3),
        "worst_window_utc": worst_window,
        "worst_window_lost_s": round(worst_lost, 1),
        "affected_windows_by_hour": dict(sorted(per_hour.items())),
        "hours_touched": len(per_hour),
    }


def main() -> int:
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    # FRAGMENT B'S END IS DECLARED HERE, and BE's cutoff must be <= it: the
    # last COMPLETE window whose span has fully elapsed at this read. A cutoff
    # past it would score windows still recording.
    frag_b_end = int(now // WINDOW_S) * WINDOW_S
    a = fragment_report(*FRAG_A, "A: 08-28 post-freeze (06:09:00Z -> 24:00:00Z)")
    b = fragment_report(FRAG_B_START, frag_b_end,
                        "B: 08-29 00:00:00Z -> declared cutoff")

    rec = {
        "receipt": "da_fragment_censoring_v1",
        "status": "DIAGNOSTIC_NEVER_EVIDENCE",
        "authorisation": "R-293 (user-ordered diagnostic; coordinator dispatch)",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "coin": COIN,
        "freeze_epoch": FREEZE_EPOCH,
        "freeze_epoch_utc": utc(FREEZE_EPOCH),
        "declared_cutoff_utc": utc(frag_b_end),
        "declared_cutoff_epoch": frag_b_end,
        "cutoff_rule": "last COMPLETE 300s window fully elapsed at this read; "
                       "BE's declared cutoff must be <= this, or it scores "
                       "windows that were still recording",
        "era": {
            "hf_stamp_boundary_ns": HF_ERA_FLOOR_NS,
            "hf_stamp_boundary_utc": "2026-08-24T13:48:54Z",
            "both_fragments_after_boundary": True,
            "pm_collector_version_in_range": "clob_v3_1 (single version; no "
                                             "collector era change inside "
                                             "either fragment)",
            "verdict": "ERA-PURE: both fragments lie wholly after the hf_ws_v2 "
                       "stamp boundary and within one PM collector version, so "
                       "no row is inadmissible on era grounds",
        },
        "fragments": [a, b],
        "INADMISSIBLE_FOR_THE_RACE": True,
        "inadmissibility_reasons": [
            "neither fragment is a COMPLETE UTC day, and the admission rule "
            "counts complete days (rule 8)",
            "fragment A begins mid-day at the freeze epoch, so it is a "
            "SELECTED slice of a day whose remaining windows are excluded by "
            "construction",
            "both carry quantified btc feed loss that is burst-concentrated",
        ],
        "inadmissibility_is_unconditional":
            "These three are properties of the FRAGMENTS, not of any result. "
            "They hold whether the diagnostic reads positive, negative or "
            "null, and no outcome can change them.",
        "censoring_statement":
            "The bias direction is UNKNOWN and plausibly FLATTERING: a window "
            "lost while the feed was struggling is plausibly a window in which "
            "the policy faced its hardest decisions, and scoring only the "
            "surviving windows removes them. A positive read is weak comfort "
            "at best; a negative read is ambiguous rather than damning.",
        "censoring_measured_not_asserted": {
            "note":
                "R-293's pre-registered wording says the loss is "
                "BURST-CONCENTRATED. Measured, that is TRUE BUT MILD, and the "
                "shape is different from what 'burst' usually implies -- so "
                "the measurement is reported rather than the wording repeated "
                "(rule 10). This QUALIFIES the pre-registered statement; it "
                "does not dispute the order, which is the user's to give.",
            "concentration":
                "worst decile of affected windows holds 25.2% (fragment A) / "
                "18.0% (B) of lost seconds against ~10% under a uniform "
                "drizzle -- real concentration, about 2.5x and 1.8x, not "
                "extreme",
            "outage_shape":
                "MANY SHORT outages, not few long ones: median 2.2s (A) / "
                "8.7s (B), max 12.4s (A) / 11.4s (B). A 2.2s outage censors "
                "0.7% of its 300s window",
            "the_two_numbers":
                "63.1% of fragment-A windows are TOUCHED by loss while only "
                "2.84% of window-TIME is lost. Both are true. The first is the "
                "worse shape for window-level admissibility, the second the "
                "milder one for data volume, and quoting either alone misleads "
                "in a predictable direction.",
            "what_this_does_not_soften":
                "None of it makes the fragments admissible. The three "
                "inadmissibility reasons are structural -- incomplete days, a "
                "mid-day selected start, quantified loss -- and a mild "
                "censoring shape does not convert an inadmissible fragment "
                "into an admissible one.",
        },
    }
    out = Path("/home/yuqing/ctaNew/data/pm_5min/derived/"
               "da_fragment_censoring_v1.json")
    out.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
    for f in (a, b):
        print(f"\n=== {f['label']} ===")
        print(f"  bounds        : {f['bounds_utc'][0]} -> {f['bounds_utc'][1]}"
              f"  ({f['span_hours']}h)")
        print(f"  windows       : {f['contained_windows']} contained, "
              f"{f['boundary_partial_windows']} boundary-partial (excluded)")
        print(f"  gap-affected  : {f['windows_gap_affected']} "
              f"({f['windows_gap_affected_pct']}%) over "
              f"{f['gap_intervals']} intervals")
        print(f"  lost          : {f['lost_seconds']}s "
              f"= {f['lost_s_per_hour']}/hr   worst window "
              f"{f['worst_window_utc']} at {f['worst_window_lost_s']}s")
        bc = f["burst_concentration"]
        print(f"  coverage      : {f['windows_observed_in_tape']}/"
              f"{f['contained_windows']} observed in tape, "
              f"{f['windows_missing_from_tape']} missing "
              f"-> verified={f['coverage_verified']}")
        print(f"  hours touched : {f['hours_touched']}")
        print(f"  BURST         : worst {bc['worst_decile_windows']} of "
              f"{bc['affected_windows']} affected windows hold "
              f"{bc['worst_decile_share_of_lost_seconds_pct']}% of lost "
              f"seconds (uniform would be ~10%); median outage "
              f"{bc['median_interval_s']}s, max {bc['max_interval_s']}s")
    print(f"\nERA: {rec['era']['verdict']}")
    print(f"INADMISSIBLE FOR THE RACE: {rec['INADMISSIBLE_FOR_THE_RACE']} "
          f"(unconditional; {len(rec['inadmissibility_reasons'])} reasons)")
    print(f"receipt -> {out}")
    return 0


def _selftests() -> int:
    """Falsifiers for a receipt that gates a diagnostic (rule 15).

    A receipt with no falsifier is an instrument that has never proved it can
    fire, and this one carries numbers that are now in the register.
    """
    checks, fails = 0, []

    def ok(c, label):
        nonlocal checks
        checks += 1
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    lo, hi = 1787897340, 1787961600
    a = fragment_report(lo, hi, "A")
    ok(a["contained_windows"] == 214 and a["boundary_partial_windows"] == 1,
       "fragment A contains 214 whole windows with 1 boundary-partial "
       "EXCLUDED -- and 214 is the same count the day verdict derives "
       "independently for entirely_post_freeze, so two instruments agree "
       "without sharing the derivation")
    ok(a["coverage_verified"] and a["windows_missing_from_tape"] == 0,
       f"coverage is VERIFIED, not assumed: "
       f"{a['windows_observed_in_tape']}/{a['contained_windows']} windows are "
       f"present in the tape. Without this a dead collector -- which writes no "
       f"gaps -- would have reported a PERFECT fragment")
    ok(0 < a["windows_gap_affected"] < a["contained_windows"],
       f"the affected count is a STRICT SUBSET "
       f"({a['windows_gap_affected']} of {a['contained_windows']}) -- neither "
       f"0 nor everything, so the interval arithmetic is discriminating "
       f"rather than matching or missing every window")
    ok(a["window_time_lost_pct"] < a["windows_gap_affected_pct"],
       f"the two measures DIVERGE as they must "
       f"({a['window_time_lost_pct']}% of time vs "
       f"{a['windows_gap_affected_pct']}% of windows): many short outages "
       f"touch many windows while removing little time, and a receipt "
       f"reporting only one would mislead in a predictable direction")
    # THE KEY REFUSAL: a range with no tape at all. My first attempt used
    # 2026-08-23T15:46 assuming it was quiet -- it has 9/9 coverage and a real
    # gap, so the test asserted a premise instead of a behaviour and went RED.
    # The fixture was wrong, not the code; a probe found a genuinely
    # unobserved range.
    dead = fragment_report(1785000000, 1785000000 + 3000, "unobserved")
    ok(dead["windows_gap_affected"] == 0 and dead["lost_seconds"] == 0.0,
       f"an UNOBSERVED range ({dead['contained_windows']} calendar windows) "
       f"reports zero gaps and zero lost seconds -- exactly what a PERFECT "
       f"fragment reports, which is the whole danger")
    ok(not dead["coverage_verified"]
       and dead["windows_missing_from_tape"] == dead["contained_windows"],
       f"...and coverage_verified=False with all "
       f"{dead['windows_missing_from_tape']} windows missing SEPARATES them. "
       f"Without it a dead collector -- which writes no gaps -- is "
       f"indistinguishable from a flawless one")
    live_ = fragment_report(1787300000, 1787300000 + 3000, "observed")
    ok(live_["coverage_verified"] and live_["windows_gap_affected"] > 0,
       "positive control: an OBSERVED range verifies coverage and still finds "
       "its gaps, so the check is not simply refusing everything")
    print(f"\n{'FRAGMENT RECEIPT GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing, {checks} checks")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftests())
    raise SystemExit(main())
