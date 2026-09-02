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
import os
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_tape_density as TD                                   # noqa: E402
import da_content_liveness_rule as CLR                         # noqa: E402

#: RR12-1 -- THE SPLIT. These are two different roots and conflating them is
#: the defect: a single hardcoded REPO made a run FROM A WORKTREE record the
#: MAIN tree's commit, so the artifact named a tree that did not execute.
#:
#:   CODE_ROOT -- the tree the RUNNING FILE lives in, derived from __file__.
#:                Everything about WHICH CODE RAN (carrying_commit, the dirty
#:                flag) is asked of this tree and no other.
#:   DATA_ROOT -- where the tape and the derived artifacts live. It does NOT
#:                follow the code: a worktree has no `data/` (it is gitignored
#:                and lives once), so deriving this from __file__ would make a
#:                worktree run read an empty directory and report a clean day.
#:                Overridable for a rehearsal; canonical by default.
CODE_ROOT = Path(__file__).resolve().parents[2]
#: ONE definition, imported. `pm_tape_density` is the lowest-level reader
#: of this tape, so the resolution lives there and everything above it
#: agrees by construction rather than by two copies of a rule.
DATA_ROOT = TD.DATA_ROOT
#: Kept as the DATA root under its old name so no consumer silently changes
#: meaning; every git question below asks CODE_ROOT instead.
REPO = DATA_ROOT
DERIVED = DATA_ROOT / "data/pm_5min/derived"

#: ---------------------------------------------------------------------------
#: R-411(i) and R-411(ii) -- USER-RULED at R-424, named ONCE, here.
#: ---------------------------------------------------------------------------
#: R-424 §4, quoted: "for G-COUNTING only, a coin-day counts toward the >=5 bar
#: only if its unmasked complement covers >= 50% of the calendar day -- >= 144
#: of 288 windows; every good window is scored regardless."
#:
#: READ THE SECOND CLAUSE. This is a G-COUNTING floor and NOTHING ELSE. It does
#: not gate scoring, it does not shrink a population, and it does not make a
#: short-complement day bad: every window in the complement is scored whatever
#: this says. It decides only whether the coin-day counts toward the >=5-day
#: bar (rule 8).
G_MIN_COMPLEMENT_WINDOWS = 144
G_MIN_COMPLEMENT_RULING = "R-424 §4 (USER, 2026-09-02), applying R-411(i)"

#: R-424 §4, quoted: "the P1 bar on a complement reads per UNMASKED hour (loss
#: per hour of usable feed); the calendar-24h form stays reported beside it."
P1_GOVERNING_DENOMINATOR = "per_unmasked_hour"
P1_DENOMINATOR_RULING = "R-424 §4 (USER, 2026-09-02), applying R-411(ii)"
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


def uncompressed_for(p: Path) -> int:
    """Thin wrapper so the fixture can assert its own window sizes."""
    return TD.uncompressed_size(p)


def _head_commit() -> str | None:
    """The commit this producer ran at (R-387's `carrying_commit`)."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:                                        # pragma: no cover
        return None


def _tree_dirty() -> bool | None:
    """True when a PRODUCING file differs from the commit named above.

    A `carrying_commit` recorded over a dirty tree points at bytes that did
    not run -- R-306's standing rule, one artifact down.
    """
    import subprocess
    files = ["live/pm_research/da_blackout_mask.py",
             "live/pm_research/da_content_liveness_rule.py",
             "live/pm_research/pm_tape_density.py"]
    try:
        r = subprocess.run(["git", "-C", str(CODE_ROOT), "status", "--porcelain",
                            "--"] + files, capture_output=True, text=True,
                           timeout=20)
        return bool(r.stdout.strip())
    except Exception:                                        # pragma: no cover
        return None


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
        # R-387/R-412: an artifact that names no code cannot be reproduced.
        # `carrying_commit` is the HEAD the producer ran at; `dirty` says
        # whether the tree matched it, because a commit ref over a dirty tree
        # is a pointer to bytes that did not run.
        "producer": {
            "module": "da_blackout_mask",
            "module_sha256_prefix": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest()[:16],
            # RR12-1: the commit is asked of the tree the RUNNING FILE is in,
            # and that tree is NAMED. A run from a worktree records the
            # worktree's commit; previously it recorded the main tree's, so
            # the artifact named a tree that did not execute.
            "code_root": str(CODE_ROOT),
            "carrying_commit": _head_commit(),
            "tree_dirty_on_producing_files": _tree_dirty(),
            "data_root": str(DATA_ROOT),
            # DA10-R2: the BRANCH, so the pair is self-explaining.
            "data_root_branch": TD.DATA_ROOT_BRANCH,
            "roots_note": ("code_root is where the running file lives; "
                           "data_root is where the tape and artifacts live. "
                           "They differ under a worktree, and only the first "
                           "answers 'which code ran'."),
        },
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
            # R-411(ii), RULED at R-424 §4: the per-UNMASKED-hour form GOVERNS
            # and the calendar-24h form stays REPORTED beside it. Both are
            # still emitted; what changed is that the artifact now SAYS which
            # one the bar is read against, instead of leaving a reader to
            # choose between two numbers that differed by 3.6x on 09-02.
            "P1_governing_denominator": P1_GOVERNING_DENOMINATOR,
            "P1_governing_value": round(lost / unmasked_h, 2)
            if unmasked_h else None,
            "P1_governing_pass": (None if not unmasked_h
                                  else (lost / unmasked_h)
                                  <= D.P1_LOST_S_PER_HR_MAX),
            "P1_ruling": P1_DENOMINATOR_RULING,
            "P2_material_windows_in_complement": material,
            "P2_share_of_UNMASKED": round(material / len(unmasked), 4)
            if unmasked else None,
            "P2_share_of_288": round(material / WINDOWS_PER_DAY, 4),
            "P2_frozen_bar_share": D.P2_MATERIAL_SHARE_MAX,
            # RR9-2: the OTHER half of P2's definition. The block's thesis is
            # that every denominator is stated; the numerator's threshold --
            # which windows count as material at all -- has to be too, or a
            # reader of the artifact alone cannot reconstruct P2. READ from
            # the constant, never restated as a literal.
            "P2_material_span_s": D.P2_MATERIAL_SPAN_S,
            "P3_worst_rolling_60min_lost_s": round(worst, 1),
            "P3_frozen_bar": D.P3_ROLLING_60MIN_LOST_S_MAX,
            # R-411(i), RULED at R-424 §4. G-COUNTING ONLY: this decides
            # whether the coin-day counts toward the >=5-day bar and NOTHING
            # else. Every window in the complement is scored regardless --
            # the ruling says so in the same sentence, and it is repeated in
            # the payload because a bare boolean beside a complement invites
            # the other reading.
            "counts_toward_G": len(unmasked) >= G_MIN_COMPLEMENT_WINDOWS,
            "counts_toward_G_floor_windows": G_MIN_COMPLEMENT_WINDOWS,
            "counts_toward_G_ruling": G_MIN_COMPLEMENT_RULING,
            "counts_toward_G_scope": (
                "G-COUNTING ONLY. A false here does NOT exclude the day's "
                "data: every good window is scored regardless (R-424 §4). It "
                "means the coin-day does not count toward the >=5 "
                "complete-day bar."),
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
        # WAS `ESCALATION_no_minimum_complement_size`. The USER ruled it at
        # R-424; an escalation that has been answered must stop reading as
        # open, or the artifact keeps asking a settled question.
        "RULED_minimum_complement_size": (
            f"RULED at {G_MIN_COMPLEMENT_RULING}: a coin-day counts toward "
            f"the >=5-day bar only if its unmasked complement covers "
            f">= {G_MIN_COMPLEMENT_WINDOWS} of 288 windows (>= 50% of the "
            f"calendar day). G-COUNTING ONLY -- every good window is scored "
            f"regardless. Emitted per coin-day as `counts_toward_G`."),
        "RULED_P1_governing_denominator": (
            f"RULED at {P1_DENOMINATOR_RULING}: the P1 bar on a complement "
            f"reads {P1_GOVERNING_DENOMINATOR!r} (loss per hour of usable "
            f"feed); the calendar-24h form stays REPORTED beside it."),
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
        # PINNING THE RULING, NOT THE ESCALATION. This check asserted that an
        # ESCALATION key was present -- true while the question was open and
        # FALSE the moment the USER answered it. That is the draft-state-pin
        # class the R-386 freeze surfaced three times; it now pins the RULED
        # state and fails if either ruling is dropped or the block starts
        # governing.
        ok(cq["governs"] is False
           and "ESCALATION_no_minimum_complement_size" not in cq
           and G_MIN_COMPLEMENT_RULING in cq["RULED_minimum_complement_size"]
           and P1_GOVERNING_DENOMINATOR
           in cq["RULED_P1_governing_denominator"],
           "the block REPORTS, and both R-411 constants read as RULED (R-424) "
           "rather than as open escalations -- an answered question must stop "
           "being asked by the artifact")
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

        # ---- RR9-1: make the export contract STRUCTURAL, not data-dependent
        # The equality can only fire when the two definitions DISAGREE on
        # that day's data. Both redefinition mutants (drop the gap-overlap
        # exclusion; move thin_frac 0.05 -> 0.06) passed the whole suite,
        # and on real data one of them still BUILDS 08-26 cleanly -- so a
        # redefined mask ships silently on any day where the change happens
        # not to matter, which is the day nobody would look at. This fixture
        # makes BOTH changes matter by construction.
        #
        #   row (a): a thin window that OVERLAPS a gap-ledger interval, so
        #            dropping the exclusion adds it to the mask;
        #   row (b): a window between 0.05x and 0.06x the median, so moving
        #            the fraction to 0.06 adds THAT one.
        med_lines = 5000
        rr9 = "20260404"
        (root / rr9).mkdir(parents=True, exist_ok=True)
        b9 = day_bounds(rr9)[0]
        # 0.055 x median -> inside (0.05, 0.06)
        between = max(1, int(med_lines * 0.055))
        for i in range(288):
            n = med_lines
            if i == 50:            # (a) thin AND gap-covered
                n = 3
            elif i == 60:          # (b) between the two fractions
                n = between
            elif i == 70:          # a plain thin window, in the mask either way
                n = 3
            with gzip.open(root / rr9 /
                           f"btc-updown-5m-{b9 + i * WINDOW_S}.jsonl.gz",
                           "wb") as fh:
                fh.write(b'{"x":1}\n' * n)
        g9 = {"btc": [(b9 + 50 * WINDOW_S + 10, b9 + 50 * WINDOW_S + 20)]}
        m9 = build_mask(rr9, raw_root=root, gaps=g9)
        b9c = m9["coins"]["btc"]
        ok(b9c["n_masked"] == 1
           and b9c["masked_windows"] == [b9 + 70 * WINDOW_S],
           "RR9-1 STRUCTURAL FIXTURE: under v1's definition exactly ONE "
           "window is masked -- window 70. Window 50 is thin but GAP-COVERED "
           "(accounted loss, excluded) and window 60 sits at 0.055x the "
           "median, thin under 0.06 but not under 0.05")
        # The two rows exist and are what they claim to be -- otherwise the
        # fixture could pass while testing nothing.
        _sizes = {i: uncompressed_for(root / rr9 /
                                      f"btc-updown-5m-{b9 + i * WINDOW_S}.jsonl.gz")
                  for i in (50, 60, 70)}
        _med = sorted(_sizes.values())[1]
        ok(TD.gap_overlaps(g9, "btc", b9 + 50 * WINDOW_S,
                           b9 + 51 * WINDOW_S) is True
           and not TD.gap_overlaps(g9, "btc", b9 + 60 * WINDOW_S,
                                   b9 + 61 * WINDOW_S),
           "RR9-1: row (a) really is gap-covered and row (b) really is not -- "
           "a fixture that did not carry both shapes would kill neither mutant")

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

    # ---- R-411(i)/(ii) AS RULED AT R-424, with falsifiers on the edge ----
    # The G floor is a >= test on the COMPLEMENT, so the interesting fixtures
    # are the two windows either side of it. 288 - 144 = 144 masked gives a
    # complement of exactly 144 (counts); one more masked window gives 143
    # (does not).
    # THE EDGE IS BUILT BY VARYING THE PRESENT COUNT, NOT THE MASKED COUNT,
    # and the reason is a documented property of v1. Pushing masked past half
    # the day moves the day's OWN median into the dark regime, so the dark
    # windows stop being "thin relative to the median" and the mask comes back
    # EMPTY -- RR6-1's blind spot, which is exactly what the (now frozen) v2
    # absolute floor exists for. A 145-of-288 fixture therefore measured v1's
    # limit rather than the G floor. Holding masked at 56 of ~200 present
    # keeps the fixture inside v1's competence and lets the complement land
    # on 144 and 143.
    for _present, _want, _comp in ((200, True, 144), (199, False, 143)):
        _d = f"2026041{1 if _want else 2}"
        _nmask = _present - _comp
        (root / _d).mkdir(parents=True, exist_ok=True)
        _b = day_bounds(_d)[0]
        for _i in range(_present):
            with gzip.open(root / _d /
                           f"btc-updown-5m-{_b + _i * WINDOW_S}.jsonl.gz",
                           "wb") as fh:
                fh.write(b'{"x":1}\n' * (3 if _i < _nmask else 5000))
        _m = build_mask(_d, raw_root=root, gaps={})
        _c = complement_quality(_d, _m, raw_root=root)["coins"]["btc"]
        ok(_c["n_windows_unmasked"] == _comp
           and _c["counts_toward_G"] is _want,
           f"R-411(i) EDGE: a {_comp}-window complement counts_toward_G="
           f"{_want} against the ruled floor of {G_MIN_COMPLEMENT_WINDOWS}. "
           f"The two fixtures sit ONE WINDOW apart, so an off-by-one in the "
           f"comparison fails here rather than on a real day")
        ok("G-COUNTING ONLY" in _c["counts_toward_G_scope"]
           and str(G_MIN_COMPLEMENT_WINDOWS) in _c["counts_toward_G_ruling"]
           or _c["counts_toward_G_floor_windows"] == G_MIN_COMPLEMENT_WINDOWS,
           "R-411(i): the boolean travels with its SCOPE -- a false does not "
           "exclude the day's data, it means the coin-day does not count "
           "toward the >=5-day bar")

    # R-411(ii): the governing denominator must be able to DISAGREE with the
    # calendar one, or naming it would be decoration. A day with a large
    # masked block concentrates the same loss into fewer usable hours.
    mk("20260413", thin_idx=set(range(200)))
    _m2 = build_mask("20260413", raw_root=root, gaps={})
    _c2 = complement_quality("20260413", _m2, raw_root=root)["coins"]["btc"]
    ok(_c2["P1_governing_denominator"] == P1_GOVERNING_DENOMINATOR
       and _c2["P1_governing_value"] == _c2["P1_lost_s_per_UNMASKED_hour"]
       and _c2["P1_lost_s_per_CALENDAR_24h"] is not None,
       "R-411(ii): the block NAMES which denominator governs, carries that "
       "value, and keeps the calendar-24h figure reported beside it")

    # ---- RR12-1 CONTROL: a WORKTREE run records the WORKTREE's commit ----
    # The defect this closes: `REPO` was one hardcoded path, so a producer
    # running from a worktree stamped the MAIN tree's HEAD -- an artifact
    # naming a tree that did not execute. Proving it needs a second tree at a
    # DIFFERENT commit, so the two answers cannot coincide by accident.
    import subprocess as _sp
    import tempfile as _tf2
    _root_git = _sp.run(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"],
                        capture_output=True, text=True)
    _prev = _sp.run(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD~1"],
                    capture_output=True, text=True)
    if _root_git.returncode == 0 and _prev.returncode == 0:
        _here, _there = _root_git.stdout.strip(), _prev.stdout.strip()
        ok(_here != _there,
           "RR12-1 control precondition: HEAD and HEAD~1 differ, so the two "
           "trees cannot record the same commit by coincidence")
        _wt = Path(_tf2.mkdtemp(prefix="da_wt_control_"))
        _wt_path = _wt / "tree"
        _added = _sp.run(["git", "-C", str(CODE_ROOT), "worktree", "add",
                          "--detach", str(_wt_path), _there],
                         capture_output=True, text=True)
        try:
            if _added.returncode == 0:
                # THE WORKTREE SUPPLIES A DIFFERENT GIT HEAD; THE CODE UNDER
                # TEST IS THE CODE BEING SHIPPED. Left at HEAD~1 the tree
                # holds the PRE-FIX file, which reproduces the defect (it
                # stamps the main tree's commit) -- useful as a demonstration,
                # useless as a control over the fix. So the producing files
                # are copied in, and `tree_dirty` then reads True, which is
                # itself the correct answer for a tree whose files differ
                # from its own HEAD.
                for _f in ("da_blackout_mask.py", "pm_tape_density.py",
                           "da_content_liveness_rule.py",
                           "da_forward_day_verify.py"):
                    (_wt_path / "live/pm_research" / _f).write_bytes(
                        (CODE_ROOT / "live/pm_research" / _f).read_bytes())
                _prog = (
                    "import sys, json, gzip, datetime as dt\n"
                    f"sys.path.insert(0, {str(_wt_path / 'live/pm_research')!r})\n"
                    "import da_blackout_mask as BM\n"
                    "from pathlib import Path\n"
                    "raw = Path(sys.argv[1]); day='20260410'\n"
                    "(raw/day).mkdir(parents=True, exist_ok=True)\n"
                    "base = BM.day_bounds(day)[0]\n"
                    "for i in range(288):\n"
                    "    with gzip.open(raw/day/f'btc-updown-5m-{base+i*300}"
                    ".jsonl.gz','wb') as fh: fh.write(b'x\\n'*5000)\n"
                    "m = BM.build_mask(day, raw_root=raw, gaps={})\n"
                    "print(json.dumps(m['producer']))\n")
                _r = _sp.run([sys.executable, "-c", _prog, str(_wt / "raw")],
                             capture_output=True, text=True, timeout=300)
                ok(_r.returncode == 0,
                   f"RR12-1 control: the producer RUNS from a worktree at all "
                   f"-- which it could not before, because its data root came "
                   f"from __file__ and a worktree has no tape "
                   f"({_r.stderr.strip()[-160:]})")
                _prod = json.loads(_r.stdout.strip().splitlines()[-1])
                ok(_prod["carrying_commit"] == _there
                   and _prod["carrying_commit"] != _here,
                   f"RR12-1 CONTROL: a run from the worktree records the "
                   f"WORKTREE's commit ({_there[:12]}), NOT the main tree's "
                   f"({_here[:12]}). The artifact names the tree that "
                   f"executed")
                # THE FLAG MUST AGREE WITH THE CHILD TREE'S ACTUAL STATE,
                # which is COMPUTED here rather than assumed. Asserting `True`
                # encoded the fixture's arrangement: it holds only while the
                # copied files differ from the child's HEAD, so the control
                # went red the first time a commit touched none of them
                # (e384792 changed only the preflight). Third instance of the
                # DA10-R5 class, in the same control -- assert the property.
                _exp_dirty = bool(_sp.run(
                    ["git", "-C", str(_wt_path), "status", "--porcelain",
                     "--"] + [f"live/pm_research/{_f}" for _f in
                              ("da_blackout_mask.py", "pm_tape_density.py",
                               "da_content_liveness_rule.py",
                               "da_forward_day_verify.py")],
                    capture_output=True, text=True).stdout.strip())
                ok(_prod["tree_dirty_on_producing_files"] == _exp_dirty,
                   f"RR12-1 CONTROL: the dirty flag ({_prod['tree_dirty_on_producing_files']}) "
                   f"equals the CHILD tree's own measured state "
                   f"({_exp_dirty}) -- the flag answers about the tree that "
                   f"ran, and the expectation is computed from that tree "
                   f"rather than assumed from how the fixture happened to be "
                   f"arranged")
                # DA10-R5: ASSERT THE PROPERTY, NOT THE ENVIRONMENT. This
                # compared the child's data_root against the PARENT's, which
                # only holds when the parent's own root is canonical -- so the
                # control failed from any tree carrying the tape. The property
                # is that the CHILD resolved its own roots and they DIFFER:
                # the temp worktree carries no tape, so its data root cannot
                # be itself.
                ok(str(_wt_path) in _prod["code_root"]
                   and _prod["data_root"] != _prod["code_root"]
                   # MEMBERSHIP, not `!=`: a MISSING key satisfies `!=`, so
                   # deleting `data_root_branch` from the emission would have
                   # passed this control. The child carries no tape, so its
                   # own resolution must be branch 1 (an inherited
                   # PM_DATA_ROOT) or branch 3 (canonical) -- and it must be
                   # PRESENT.
                   and _prod.get("data_root_branch") in (
                       "1_env_PM_DATA_ROOT", "3_canonical"),
                   f"RR12-1 CONTROL: the child NAMES both roots and they "
                   f"DIFFER -- code_root is the throwaway worktree, and since "
                   f"that tree carries no tape its resolver CANNOT take "
                   f"branch 2 (got {_prod.get('data_root_branch')!r}). "
                   f"Written this way twice over: asserting '3_canonical' "
                   f"encoded the environment again -- a child inheriting "
                   f"PM_DATA_ROOT resolves by branch 1, which is equally "
                   f"correct. The property is that the child resolved its "
                   f"OWN roots and they differ")
        finally:
            _sp.run(["git", "-C", str(CODE_ROOT), "worktree", "remove",
                     "--force", str(_wt_path)], capture_output=True)
            _sp.run(["git", "-C", str(CODE_ROOT), "worktree", "prune"],
                    capture_output=True)
        ok(not _wt_path.exists(),
           "RR12-1 control: the throwaway worktree is removed -- an "
           "instrument that leaves trees behind is the untracked-drop-in "
           "class in another costume")

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
