#!/usr/bin/env python3
"""Blackout mask + complement quality — R-409's instruments.

USER RULING R-409, verbatim: *"If the data quality is good over the
non-blackout time, we should use that data."* A day with a blackout is NOT
excluded; it accrues if the frozen bars pass on the non-blackout windows, and
the blackout windows are MASKED as accounted loss — counted, reported, and
excluded from that day's forward score.

TWO ARTIFACTS, AND NEITHER DECIDES ANYTHING.

  * `build_mask()` EXPORTS the window list the FROZEN v1 detector already
    counts. It does not define a second population: it recomputes the list
    with v1's identical definition and then **REFUSES unless the count equals
    the frozen detector's own `n_invisible_thin` for every coin.** A mask that
    disagrees with L1's numerator is a defect, so the disagreement is a
    refusal rather than a number.
  * `complement_quality()` re-evaluates the frozen P1/P2/P3 over the UNMASKED
    windows and REPORTS. `race_accrual_eligible` is untouched; the disposition
    is the coordinator's act with R-409 as its stated reason (rule 14).

NO NEW THRESHOLD IS CHOSEN HERE. Where the evaluation needs a constant this
file does not have — a minimum complement size above which the complement bars
mean anything — it ESCALATES in the artifact rather than inventing one.

THE v2 SEAM IS PRESENT AND REFUSES. v2's absolute floor is drafted and NOT
frozen, so `v2_mask_windows()` raises by name. That is the R-402 class made
explicit: a capability that is built but not authorised must refuse out loud,
never sit silently unwired.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_tape_density as TD                                   # noqa: E402
import da_content_liveness_rule as CLR                         # noqa: E402

REPO = Path("/home/yuqing/ctaNew")
DERIVED = REPO / "data/pm_5min/derived"
WINDOW_S = TD.WINDOW_S
WINDOWS_PER_DAY = 288

ARTIFACT_KIND = "da_blackout_mask_v1"
DISPOSITION_RULE = "R-409"


class MaskRefused(Exception):
    """The mask cannot be produced honestly."""


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def day_bounds(day: str) -> tuple[int, int]:
    d = dt.datetime.strptime(day, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp()), int(d.timestamp()) + 86400


def _runs(ws: list[int]) -> list[dict]:
    out, cur = [], []
    for w in ws:
        if cur and w - cur[-1] == WINDOW_S:
            cur.append(w)
        else:
            if cur:
                out.append(cur)
            cur = [w]
    if cur:
        out.append(cur)
    return [{"start_utc": _iso(r[0]), "end_utc": _iso(r[-1] + WINDOW_S),
             "n_windows": len(r)} for r in out]


def v2_mask_windows(*_a, **_k):
    """The v2 absolute-floor seam. REFUSES until the USER freezes v2.

    Built now so the seam is visible and its absence is a REFUSAL rather than
    a silence — R-402 was a governing rule that sat wired-to-nothing without
    saying so. When v2 is frozen this becomes the second mask source for the
    days v1 cannot see (a mostly-dark day, where the median collapses).
    """
    raise MaskRefused(
        "REFUSED: the v2 absolute-floor mask is DRAFT and NOT USER-FROZEN "
        "(see plans/DA_CONTENT_LIVENESS_RULE_V2_AMENDMENT.md). v1 cannot "
        "identify the non-blackout complement on a mostly-dark day, so on "
        "such a day this seam must refuse rather than hand back a mask v1 "
        "could not see. Freeze v2 to enable it.")


def build_mask(day: str, raw_root: Path | None = None, gaps=None
               ) -> dict[str, Any]:
    """The window-level mask, EXPORTED from the frozen v1 detector."""
    gaps = TD.load_gaps() if gaps is None else gaps
    frozen = CLR.measure_day(day, gaps=gaps, raw_root=raw_root)
    if frozen.get("status") in ("CONTENT_LIVENESS_UNRESOLVED",
                                "CONTENT_LIVENESS_UNJUDGEABLE"):
        raise MaskRefused(
            f"REFUSED: the frozen detector reports {frozen['status']} for "
            f"{day} ({frozen.get('why')}). A day the detector cannot judge "
            f"has no mask, and an empty mask there would read as 'nothing "
            f"was dark' — the empty-set trap on the artifact the scorer "
            f"consumes.")
    try:
        agg = TD.scan_day(day, TD.RAW if raw_root is None else raw_root)
    except TD.Refused as e:
        raise MaskRefused(str(e)) from None
    per: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for (c, w), b in agg.items():
        per[c].append((w, b))

    coins: dict[str, Any] = {}
    for c, wins in sorted(per.items()):
        wins.sort()
        fz = frozen["coins"].get(c, {})
        if "n_invisible_thin" not in fz:
            coins[c] = {"status": fz.get("status", "UNJUDGEABLE"),
                        "n_windows_total": len(wins), "n_masked": None,
                        "why": "the frozen detector did not judge this coin, "
                               "so there is no numerator to export"}
            continue
        med = statistics.median([b for _, b in wins])
        # v1's DEFINITION, character for character in intent: below
        # thin_frac x the SAME-DAY median, and not overlapped by a gap row.
        masked = [w for w, b in wins
                  if b < med * CLR.THIN_FRAC
                  and not TD.gap_overlaps(gaps, c, w, w + WINDOW_S)]
        # THE EQUALITY IS THE CONTRACT. A mask that disagrees with L1's
        # numerator is not a mask, it is a second population wearing the
        # name of the first.
        if len(masked) != fz["n_invisible_thin"]:
            raise MaskRefused(
                f"REFUSED: mask/L1 disagreement for {day} {c} — the mask "
                f"lists {len(masked)} windows while the frozen detector "
                f"counts {fz['n_invisible_thin']}. The mask MUST export the "
                f"population L1 already counts; a different one would mask "
                f"windows the bars still charge for, or charge for windows "
                f"the scorer has dropped.")
        coins[c] = {
            "status": fz.get("status"),
            "n_windows_total": len(wins),
            "n_masked": len(masked),
            "masked_fraction": round(len(masked) / len(wins), 4),
            "masked_fraction_of_288": round(len(masked) / WINDOWS_PER_DAY, 4),
            "frozen_n_invisible_thin": fz["n_invisible_thin"],
            "agrees_with_frozen_L1_numerator": True,
            "longest_run_windows": max([r["n_windows"] for r in _runs(masked)]
                                       or [0]),
            "runs": _runs(masked),
            "masked_windows": masked,
        }
    return {
        "artifact": ARTIFACT_KIND,
        "day": day,
        "as_of_utc": _iso(dt.datetime.now(dt.timezone.utc).timestamp()),
        "disposition_rule": DISPOSITION_RULE,
        "disposition_text": (
            "USER R-409: a day with a blackout is not excluded; it accrues if "
            "the frozen bars pass on the non-blackout windows, and the "
            "blackout windows are masked as accounted loss. This artifact "
            "REPORTS the mask; it decides nothing (rule 14)."),
        "detector": {
            "module": "da_content_liveness_rule",
            "version": "v1_FROZEN",
            "authority": "USER ruling 2026-09-01, R-386",
            "thin_frac": CLR.THIN_FRAC,
            "module_sha256_prefix": hashlib.sha256(
                Path(CLR.__file__).read_bytes()).hexdigest()[:16],
            "definition": "below thin_frac x the SAME-DAY (day, coin) median "
                          "AND not overlapped by a gap-ledger interval",
        },
        "v2_seam": {
            "status": "INERT_PENDING_USER_FREEZE",
            "frozen_by_user": False,
            "refuses": True,
            "why": ("v2's absolute floor is DRAFT. On a mostly-dark day v1 "
                    "cannot identify the complement at all, so this seam "
                    "refuses by name rather than returning a mask v1 could "
                    "not see. Built and refusing, never silently absent."),
        },
        # A CONSUMER MUST BE ABLE TO REFUSE A PARTIAL MASK. A mask built
        # mid-day lists only the windows that exist so far; scoring a day off
        # one would score the complement of a day that had not finished.
        "day_closed_calendar": dt.datetime.now(dt.timezone.utc).timestamp()
        >= day_bounds(day)[1],
        "consumer_note": ("REFUSE this artifact for final scoring unless "
                          "`day_closed_calendar` is true -- a mid-day mask is "
                          "a diagnostic, not a scoring input"),
        "day_status_frozen": frozen.get("status"),
        "n_coins": len(coins),
        "total_masked_windows": sum(v.get("n_masked") or 0
                                    for v in coins.values()),
        "coins": coins,
    }


def complement_quality(day: str, mask: dict, gaps_path: Path | None = None,
                       raw_root: Path | None = None) -> dict[str, Any]:
    """The FROZEN P1/P2/P3 re-evaluated over the UNMASKED windows. REPORTED.

    EVERY DENOMINATOR IS STATED, because this block's whole risk is that a
    rate computed over a smaller denominator reads like the frozen bar and is
    not it:

      * `P1_lost_s_per_UNMASKED_hour` divides by the unmasked hours actually
        observed. This is the honest 'loss per hour of usable feed'.
      * `P1_lost_s_per_CALENDAR_24h` divides by 24, which is what the FROZEN
        bar does (`day_bar_v2`'s `lost / 24.0`). Carried so the frozen number
        and the complement number can never be confused for each other.
      * P2's share is given over the UNMASKED count AND over 288.
      * P3's rolling hour is CALENDAR time, 3600 s wide. A rolling hour may
        span masked windows; loss inside them is excluded from the sum while
        the hour keeps its calendar width. Stated because the alternative
        (compressing time) would invent a statistic the frozen bar has no
        counterpart for.

    L1 OVER THE COMPLEMENT IS TAUTOLOGICALLY ZERO and is reported as such
    (rule 9): the complement is DEFINED as the windows v1 did not call thin,
    so v1 finds no thin windows in it. That is arithmetic, not evidence, and
    it must never be quoted as 'the complement is clean'.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import da_forward_day_verify as D
    lo, hi = day_bounds(day)
    # THE COMPLEMENT IS OVER WINDOWS THAT EXIST, NOT OVER THE CALENDAR.
    # Deriving it as `range(288) - masked` credited an OPEN day with every
    # window that had not happened yet: on 09-02 at 09:5xZ that read 248
    # unmasked windows out of 119 present, i.e. 169 windows of "clean
    # complement" that produced no data at all. That is the empty-set trap
    # inside the complement, and it would have inflated exactly the quantity
    # R-409 tells the scorer to trust.
    try:
        agg = TD.scan_day(day, TD.RAW if raw_root is None else raw_root)
    except TD.Refused as e:
        return {"block": "complement_quality", "status": "UNRESOLVED",
                "governs": False, "why": str(e)}
    present: dict[str, set] = collections.defaultdict(set)
    for (c, w), _ in agg.items():
        present[c].add(w)
    out: dict[str, Any] = {}
    for c, m in sorted(mask["coins"].items()):
        if not m.get("n_masked") and m.get("n_masked") != 0:
            out[c] = {"status": "UNJUDGEABLE", "why": m.get("why")}
            continue
        masked = set(m["masked_windows"])
        starts = sorted(present.get(c, set()))
        stray = masked - present.get(c, set())
        if stray:
            # A masked window that is not present is a contradiction: the
            # detector flagged content in a window with no file.
            out[c] = {"status": "REFUSED",
                      "why": f"{len(stray)} masked window(s) are not present "
                             f"in the raw scan; the mask and the tape "
                             f"disagree about which windows exist"}
            continue
        unmasked = [w for w in starts if w not in masked]
        iv = D.coin_gap_intervals(lo, hi, c, gaps_path)
        # loss charged ONLY inside unmasked window spans
        lost = 0.0
        material = 0
        for w in unmasked:
            cov = sum(min(b, w + WINDOW_S) - max(a, w)
                      for a, b in iv if a < w + WINDOW_S and b > w)
            lost += cov
            if cov >= D.P2_MATERIAL_SPAN_S:
                material += 1
        unmasked_h = len(unmasked) * WINDOW_S / 3600.0
        # P3 over CALENDAR hours, counting only unmasked loss
        worst = 0.0
        um = set(unmasked)
        for h0 in [lo + i * WINDOW_S for i in range(WINDOWS_PER_DAY)]:
            if h0 + 3600 > hi:
                break
            s = 0.0
            for w in range(h0, h0 + 3600, WINDOW_S):
                if w in um:
                    s += sum(min(b, w + WINDOW_S) - max(a, w)
                             for a, b in iv if a < w + WINDOW_S and b > w)
            worst = max(worst, s)
        out[c] = {
            "n_windows_masked": len(masked),
            "n_windows_present": len(starts),
            "n_windows_unmasked": len(unmasked),
            "day_is_complete": len(starts) == WINDOWS_PER_DAY,
            "complement_fraction_of_PRESENT": round(
                len(unmasked) / len(starts), 4) if starts else None,
            "complement_fraction_of_288": round(
                len(unmasked) / WINDOWS_PER_DAY, 4),
            "unmasked_hours": round(unmasked_h, 3),
            "lost_seconds_in_complement": round(lost, 1),
            "P1_lost_s_per_UNMASKED_hour": round(lost / unmasked_h, 2)
            if unmasked_h else None,
            "P1_lost_s_per_CALENDAR_24h": round(lost / 24.0, 2),
            "P1_frozen_bar": D.P1_LOST_S_PER_HR_MAX,
            "P2_material_windows_in_complement": material,
            "P2_share_of_UNMASKED": round(material / len(unmasked), 4)
            if unmasked else None,
            "P2_share_of_288": round(material / WINDOWS_PER_DAY, 4),
            "P2_frozen_bar_share": D.P2_MATERIAL_SHARE_MAX,
            "P3_worst_rolling_60min_lost_s": round(worst, 1),
            "P3_frozen_bar": D.P3_ROLLING_60MIN_LOST_S_MAX,
            "L1_over_complement": 0.0,
            "L1_over_complement_is_TAUTOLOGICAL": True,
            "L1_tautology_note": (
                "the complement is DEFINED as the windows v1 did not call "
                "thin, so v1 finds none in it. Rule 9: this is arithmetic, "
                "not evidence, and must never be read as 'the complement is "
                "clean'."),
        }
    return {
        "block": "complement_quality",
        "governs": False,
        "role": "REPORTED — the disposition is the coordinator's act (rule 14)",
        "disposition_rule": DISPOSITION_RULE,
        "bar_regime_note": (
            "P1/P2/P3 are the day_bar_v2 statistics. On a day whose GOVERNING "
            "regime is count_bar_v1_frozen (before 2026-08-29) these numbers "
            "are computed but do not govern that day -- read `bar_regime` in "
            "the verdict beside them."),
        "denominator_note": (
            "P1 is carried over BOTH denominators: per UNMASKED hour (loss per "
            "hour of usable feed) and per CALENDAR 24 h (what the FROZEN bar "
            "divides by). P2's share is given over the unmasked count AND over "
            "288. P3's rolling hour stays CALENDAR-wide and excludes loss "
            "inside masked windows."),
        "ESCALATION_no_minimum_complement_size": (
            "NO minimum complement size is set here and none is chosen. The "
            "frozen bars were pre-registered against a 288-window day; applied "
            "to a small complement they are being read on a population they "
            "were not registered for, and below some size they mean nothing. "
            "That constant is the USER's (rule 14 / rule 11). Until it is "
            "ruled, read `complement_fraction` beside every number here."),
        "coins": out,
    }


def write_mask(day: str, outdir: Path | None = None, **kw) -> Path:
    m = build_mask(day, **kw)
    d = DERIVED if outdir is None else outdir
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"da_blackout_mask_{day}.json"
    p.write_text(json.dumps(m, indent=1, sort_keys=True), encoding="utf-8")
    return p


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

    # ---- the v2 seam REFUSES, by name and out loud ------------------------
    try:
        v2_mask_windows("20260902")
        ok(False, "the v2 seam must REFUSE while v2 is unfrozen")
    except MaskRefused as e:
        ok("NOT USER-FROZEN" in str(e) and "mostly-dark" in str(e),
           "v2 SEAM: present and REFUSING by name — a built-but-unauthorised "
           "capability must refuse out loud, never sit silently unwired "
           "(the R-402 class, made explicit)")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "raw"

        def mk(day, thin_idx=()):
            (root / day).mkdir(parents=True, exist_ok=True)
            base = day_bounds(day)[0]
            for i in range(288):
                n = 3 if i in thin_idx else 5000
                with gzip.open(root / day /
                               f"btc-updown-5m-{base + i * WINDOW_S}.jsonl.gz",
                               "wb") as fh:
                    fh.write(b'{"x":1}\n' * n)
            return base

        # POSITIVE CONTROL: two masked windows, complement bars by hand.
        base = mk("20260905", thin_idx={100, 101})
        m = build_mask("20260905", raw_root=root, gaps={})
        b = m["coins"]["btc"]
        ok(b["n_masked"] == 2 and b["masked_windows"] ==
           [base + 100 * WINDOW_S, base + 101 * WINDOW_S]
           and b["agrees_with_frozen_L1_numerator"] is True
           and b["longest_run_windows"] == 2 and len(b["runs"]) == 1,
           "POSITIVE CONTROL: a synthetic day with exactly two thin windows "
           "yields a 2-window mask with the right starts, one run of 2, and "
           "an equality-checked agreement with the frozen L1 numerator")
        ok(m["detector"]["version"] == "v1_FROZEN"
           and len(m["detector"]["module_sha256_prefix"]) == 16
           and m["disposition_rule"] == "R-409",
           "the mask names the detector it came from, by version and content "
           "hash, and the ruling it serves")

        cq = complement_quality("20260905", m, raw_root=root)
        c = cq["coins"]["btc"]
        # BY HAND: 288 - 2 = 286 unmasked windows = 23.8333 h, no gaps at all.
        ok(c["n_windows_unmasked"] == 286
           and c["unmasked_hours"] == round(286 * 300 / 3600.0, 3)
           and c["lost_seconds_in_complement"] == 0.0
           and c["P1_lost_s_per_UNMASKED_hour"] == 0.0
           and c["P1_lost_s_per_CALENDAR_24h"] == 0.0,
           "POSITIVE CONTROL: the complement's bars reproduce by hand — 286 "
           "unmasked windows, 23.8333 unmasked hours, zero charged loss")
        ok(c["complement_fraction_of_PRESENT"] == round(286 / 288, 4)
           and c["n_windows_present"] == 288 and c["day_is_complete"] is True,
           "and the complement fraction is stated over PRESENT windows, with "
           "the present count and a completeness flag beside it")
        ok(c["L1_over_complement"] == 0.0
           and c["L1_over_complement_is_TAUTOLOGICAL"] is True
           and "not evidence" in c["L1_tautology_note"],
           "L1 over the complement is reported AS TAUTOLOGICAL (rule 9): the "
           "complement is defined as the windows v1 did not flag, so a zero "
           "there is arithmetic and must never read as 'clean'")
        ok(cq["governs"] is False
           and "ESCALATION_no_minimum_complement_size" in cq,
           "the block REPORTS and ESCALATES the one constant it would need "
           "rather than choosing it")
        ok(c["P1_lost_s_per_UNMASKED_hour"] is not None
           and c["P1_lost_s_per_CALENDAR_24h"] is not None
           and c["P2_share_of_UNMASKED"] is not None
           and c["P2_share_of_288"] is not None,
           "EVERY denominator is carried in pairs, so a complement rate can "
           "never be mistaken for the frozen bar's rate")

        # EMPTY MASK: a day with no thin windows at all.
        mk("20260906")
        m0 = build_mask("20260906", raw_root=root, gaps={})
        ok(m0["coins"]["btc"]["n_masked"] == 0
           and m0["coins"]["btc"]["masked_windows"] == []
           and m0["coins"]["btc"]["runs"] == []
           and m0["total_masked_windows"] == 0,
           "EMPTY-MASK CONTROL: a clean day emits an EMPTY mask with "
           "n_masked=0 — present and empty, which is not the same artifact as "
           "absent")
        cq0 = complement_quality("20260906", m0, raw_root=root)
        ok(cq0["coins"]["btc"]["n_windows_unmasked"] == 288
           and cq0["coins"]["btc"]["complement_fraction_of_PRESENT"] == 1.0,
           "and its complement is the WHOLE day, so the complement bars and "
           "the frozen bars are the same numbers by construction")

        # BOTH DIRECTIONS on the closed flag: the 20260905 fixture is a
        # FUTURE day, so it must read False; a past-dated one must read True.
        # A flag asserted in one direction only proves nothing about the
        # other, which is the direction a scorer depends on.
        mk("20260401", thin_idx={5})
        m_past = build_mask("20260401", raw_root=root, gaps={})
        ok(m["day_closed_calendar"] is False
           and m_past["day_closed_calendar"] is True
           and "REFUSE this artifact" in m["consumer_note"],
           "the mask states whether its day is CLOSED -- False on a future "
           "fixture and True on a past one -- and carries the instruction to "
           "refuse a mid-day mask for scoring, because a partial mask would "
           "score the complement of a day that had not finished")

        # OPEN-DAY CONTROL: the complement is PRESENT minus masked, never
        # CALENDAR minus masked. Red-first for the defect this caught on the
        # real 09-02, where the calendar form reported 248 unmasked windows
        # out of 119 that existed.
        (root / "20260908").mkdir(parents=True)
        base8 = day_bounds("20260908")[0]
        for i in range(119):
            n = 3 if 100 <= i < 110 else 5000
            with gzip.open(root / "20260908" /
                           f"btc-updown-5m-{base8 + i * WINDOW_S}.jsonl.gz",
                           "wb") as fh:
                fh.write(b'{"x":1}\n' * n)
        m8 = build_mask("20260908", raw_root=root, gaps={})
        cq8 = complement_quality("20260908", m8, raw_root=root)
        c8 = cq8["coins"]["btc"]
        ok(c8["n_windows_present"] == 119 and c8["n_windows_masked"] == 10
           and c8["n_windows_unmasked"] == 109
           and c8["day_is_complete"] is False,
           "OPEN-DAY CONTROL: with 119 windows present and 10 masked the "
           "complement is 109 -- NOT 278. Crediting an open day with windows "
           "that have not happened is the empty-set trap inside the very "
           "quantity R-409 tells the scorer to trust")
        ok(c8["complement_fraction_of_PRESENT"] == round(109 / 119, 4)
           and c8["complement_fraction_of_288"] == round(109 / 288, 4)
           and c8["complement_fraction_of_PRESENT"]
           != c8["complement_fraction_of_288"],
           "and BOTH complement fractions are carried and DIFFER on an open "
           "day, so neither can be read as the other")
        ok(c8["unmasked_hours"] == round(109 * WINDOW_S / 3600.0, 3),
           "and the unmasked-hours denominator follows the present windows")

        # KNOWN-BAD, RED-FIRST: a mask that disagrees with L1's count REFUSES.
        _real = CLR.measure_day
        try:
            def _lying(day, **kw):
                r = _real(day, **kw)
                r["coins"]["btc"]["n_invisible_thin"] += 1
                return r
            CLR.measure_day = _lying
            try:
                build_mask("20260905", raw_root=root, gaps={})
                ok(False, "a mask/L1 disagreement must REFUSE")
            except MaskRefused as e:
                ok("mask/L1 disagreement" in str(e) and "btc" in str(e),
                   "KNOWN-BAD: when the exported list and the frozen "
                   "numerator disagree by even ONE window the mask REFUSES "
                   "and names the coin — a mask that disagrees with L1 would "
                   "mask windows the bars still charge for")
        finally:
            CLR.measure_day = _real
        ok(CLR.measure_day is _real, "the detector is restored after the control")

        # KNOWN-BAD: an unjudgeable day has NO mask, not an empty one.
        (root / "20260907").mkdir(parents=True)
        base7 = day_bounds("20260907")[0]
        for i in range(5):
            with gzip.open(root / "20260907" /
                           f"btc-updown-5m-{base7 + i * WINDOW_S}.jsonl.gz",
                           "wb") as fh:
                fh.write(b'{"x":1}\n' * 5000)
        try:
            build_mask("20260907", raw_root=root, gaps={})
            ok(False, "an UNJUDGEABLE day must refuse a mask")
        except MaskRefused as e:
            ok("empty-set trap" in str(e),
               "KNOWN-BAD: a day the frozen detector cannot judge gets NO "
               "mask — an empty mask there would tell the scorer 'nothing was "
               "dark' about a day nobody measured")

    print(f"da_blackout_mask selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--day")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--outdir")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.day:
        raise SystemExit("REFUSED: --day YYYYMMDD")
    m = build_mask(a.day)
    if a.write:
        p = write_mask(a.day, Path(a.outdir) if a.outdir else None)
        print(f"wrote {p}")
    print(json.dumps({k: v for k, v in m.items() if k != "coins"}, indent=1))
    for c, v in sorted(m["coins"].items()):
        print(f"  {c}: masked {v.get('n_masked')}/{v.get('n_windows_total')} "
              f"({v.get('masked_fraction')}) longest run "
              f"{v.get('longest_run_windows')}")
    print(json.dumps(complement_quality(a.day, m), indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
