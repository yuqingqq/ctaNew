#!/usr/bin/env python3
"""Find windows the tape KEPT but barely FILLED — loss no gap row records.

R-361 LIVE-1. Three halves of one hole:

  * the collector's steady-state receive path is a bare `await ws.recv()`
    with no timeout, so a socket whose feed degrades stays open;
  * `complete_tape` counts WINDOWS, not ROWS, so a window present at 1% of
    its normal content reads as complete;
  * `oldest_age_s` is computed, printed, and consumed by nothing.

A feed that thins without disconnecting therefore writes no gap row, leaves
full window coverage, and passes P1/P2/P3 — **the day PASSES carrying less
data.** That is not hypothetical. Executed on the tape as of 2026-08-31:

  2026-08-30 18:00-18:20Z  ALL SEVEN coins at ~0.5-2% of their day median,
                           ~20 minutes, ZERO gap rows.
                           btc 18:05Z holds 532 rows; a normal btc window
                           holds 70,284.
  2026-08-29 20:10-22:45Z  10 windows across five coins at ~0.2% of median,
                           ZERO gap rows — on the day whose verdict reads
                           all_pass: true.

WHAT THIS IS AND IS NOT. It is a DETECTOR over data already on disk: it
computes a predicate and reports a population, and it decides nothing (repo
rule 14). Whether a thinned window makes a day inadmissible is the day
verdict's call, not this file's. It does not touch the collector and needs
no deploy.

MEASURE. Uncompressed byte count, read from each member's gzip trailer
(ISIZE, the last 4 bytes) — exact and O(1) per file, so the whole 24k-file
tape is a stat-and-seek rather than 24k decompressions. Bytes, not rows,
because rows would cost a full decompress for the same ordering; the two are
near-perfectly monotone within a coin and the reported ratios are to that
coin's OWN median, never across coins.

THINNESS IS RELATIVE TO (day, coin). Market activity varies by day and by
coin over two orders of magnitude, so a global byte floor would flag every
hype window and no btc window.

Exclusions are STATUSES, never silent drops (rule 4): a day+coin cell with
too few windows to have a median is reported as UNJUDGEABLE, not skipped,
and a cell with no windows at all REFUSES rather than reporting zero
thinned — 0 of 0 passing is the empty-set trap.

THE THRESHOLD IS A CHOICE MADE AFTER SEEING (rule 11), so this instrument
REPORTS and does not VETO — DA `6fc539f` ruled it and the ruling is right.
See the in-band correction on THIN_FRAC below: an earlier comment here
claimed the finding was threshold-INSENSITIVE, that claim was never computed,
and it is false. The window COUNT moves with the threshold; the DAY count
(7) and the three named events do not.

    python3 live/pm_research/pm_tape_density.py
    python3 live/pm_research/pm_tape_density.py --day 20260829
    python3 live/pm_research/pm_tape_density.py --json      # + sensitivity
    python3 live/pm_research/pm_tape_density.py --selftest   # rule 15
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
import statistics
import struct
import sys
from pathlib import Path

#: RR12-1 -- CODE ROOT AND DATA ROOT ARE NOT THE SAME TREE, and this file
#: reads DATA. Deriving the data root from `__file__` is right for code and
#: wrong for data: `data/` is gitignored and exists ONCE, so a run from a
#: worktree pointed at an empty directory. `scan_day` refuses an absent day,
#: so it failed loudly rather than reporting a clean one -- but it made a
#: worktree unable to measure anything at all, which is why the split is a
#: precondition for proving a worktree run records its own commit.
#:
#: Resolution order, deterministic and stated: an explicit PM_DATA_ROOT wins;
#: otherwise the tree holding this file IF it actually carries the tape;
#: otherwise the canonical tree. A worktree therefore reads the real tape.
CODE_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DATA_ROOT = Path("/home/yuqing/ctaNew")


def _resolve_data_root() -> Path:
    env = os.environ.get("PM_DATA_ROOT")
    if env:
        return Path(env)
    # THE TEST IS FOR THE TAPE, NOT FOR A DIRECTORY. A worktree checks out
    # the TRACKED receipts under data/pm_5min/derived, so `data/pm_5min`
    # exists there while `raw/` -- the gitignored tape this module actually
    # reads -- does not. Testing the parent directory picked the worktree and
    # then failed on the ledger; testing for the tape itself is the property.
    if (CODE_ROOT / "data" / "pm_5min" / "raw").is_dir():
        return CODE_ROOT
    return CANONICAL_DATA_ROOT


DATA_ROOT = _resolve_data_root()
REPO = DATA_ROOT
RAW = DATA_ROOT / "data/pm_5min/raw"
GAP_LEDGER = DATA_ROOT / "data/pm_5min/collector_gaps.jsonl"

WINDOW_S = 300
# A window below this fraction of its (day, coin) median is THIN.
#
# IN-BAND CORRECTION (rule 13). This constant first shipped with the comment
# "every value in [0.03, 0.25] returns the same set". **That was asserted and
# never computed, and it is FALSE** -- a printed conclusion beside a number,
# which is exactly what rule 10 forbids, committed by the author of the rule's
# latest citation. Measured across the real tape:
#
#     frac    days_dirty   invisible_windows
#     0.005       6              145
#     0.01        7              249
#     0.05        7              668
#     0.10        7              749
#     0.25       10              782
#     0.50       13             1259
#
# and the SET differs from the 0.05 set by 65 windows at 0.03, 81 at 0.10,
# 114 at 0.25. So:
#
#   * the WINDOW COUNT is threshold-dependent and must never be quoted as a
#     measurement -- "668" is a reading at one setting, not a quantity;
#   * the DAY COUNT is stable at 7 across [0.01, 0.10], a tenfold range;
#   * the three named events (08-26 04:35-07:55Z, 08-30 18:00-18:20Z,
#     08-29 eth 20:10Z) sit at 0.045%-3.5% of median and are flagged at ANY
#     threshold >= 0.01, so they do not depend on this choice at all.
#
# DA (6fc539f) ruled this instrument REPORTS and does not VETO, because the
# threshold was picked after seeing which days fail it (rule 11) -- correct,
# and the sensitivity above is why the ruling matters rather than being
# procedural. A GOVERNING bar must be pre-registered against days not yet
# seen; that is a coordinator/user act, not this file's.
#
# A parameter-free criterion exists and is only PARTLY available. The
# collector's own `msg_by_coin` counters settle thinness with no threshold at
# all. Corrected after DA `5b6582f` measured it: the log DOES reach the recent
# days -- 1,438 usable intervals on 08-29 and 1,082 on 08-31 -- and only fails
# to reach 08-19/08-26. So the threshold-free measure is live for anything
# recent and prospective only for the old days, which matters because it means
# 09-01 is measurable from its first minute rather than after retention
# accumulates. DA's `content_liveness_for` is that measure; this file's
# fraction-of-median threshold remains the only way to reach the OLD days.
#
# From 2026-08-31 the collector also emits a `health_sample` row per 60s to
# `collector_health.jsonl` carrying per-coin counts and `oldest_age_s`, so
# future days need neither this threshold nor a dateless-log reconstruction.
THIN_FRAC = 0.05
MIN_WINDOWS_FOR_MEDIAN = 20

_FN = re.compile(r"^([a-z]+)-updown-5m-(\d+)\.jsonl(?:\.\d+)?\.gz$")


class Refused(Exception):
    """A population this instrument must not summarise."""


def uncompressed_size(path: Path) -> int:
    """Bytes the member expands to, from the gzip trailer.

    ISIZE is the last 4 bytes, little-endian, modulo 2**32. Our windows top
    out near 12 MB so the wrap is unreachable; a file too short to hold a
    trailer is reported as 0 rather than raising, because a truncated
    archive IS a thin window and must be counted as one.
    """
    try:
        sz = path.stat().st_size
        if sz < 18:                      # header+trailer minimum
            return 0
        with path.open("rb") as fh:
            fh.seek(-4, os.SEEK_END)
            return struct.unpack("<I", fh.read(4))[0]
    except OSError:
        return 0


def scan_day(day: str, raw_root: Path = RAW) -> dict:
    """Aggregate uncompressed bytes per (coin, window) for one UTC day."""
    d = raw_root / day
    if not d.is_dir():
        raise Refused(f"no raw directory for {day} — an absent day is not a "
                      f"clean day, and reporting 0 thinned windows for it "
                      f"would be the empty-set trap")
    agg: dict[tuple[str, int], int] = collections.defaultdict(int)
    for fn in os.listdir(d):
        m = _FN.match(fn)
        if m:
            agg[(m.group(1), int(m.group(2)))] += uncompressed_size(d / fn)
    if not agg:
        raise Refused(f"{day} has a raw directory but NO window files — "
                      f"refusing rather than reporting a clean day")
    return agg


def load_gaps(path: Path = GAP_LEDGER) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    if not path.exists():
        return out
    for ln in path.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("gap_start_ns") and r.get("gap_end_ns") and r.get("coin"):
            out[r["coin"]].append((r["gap_start_ns"] / 1e9,
                                   r["gap_end_ns"] / 1e9))
    return out


def gap_overlaps(gaps, coin: str, w0: float, w1: float) -> bool:
    return any(gs < w1 and ge > w0 for gs, ge in gaps.get(coin, ()))


def judge_day(day: str, gaps=None, raw_root: Path = RAW,
              thin_frac: float = THIN_FRAC) -> dict:
    """Per (day, coin): the thin windows, split by whether a gap row knows.

    The whole point is the second number. A thin window a gap row already
    covers is ACCOUNTED loss — the ledger saw it and the bars charge for it.
    A thin window with no gap row is INVISIBLE loss: no row, full coverage,
    clean bars.
    """
    agg = scan_day(day, raw_root)
    gaps = load_gaps() if gaps is None else gaps
    per: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for (c, w), b in agg.items():
        per[c].append((w, b))

    coins = {}
    for c, wins in sorted(per.items()):
        sizes = [b for _, b in wins]
        if len(wins) < MIN_WINDOWS_FOR_MEDIAN:
            coins[c] = {"status": "UNJUDGEABLE", "n_windows": len(wins),
                        "why": f"{len(wins)} windows < "
                               f"{MIN_WINDOWS_FOR_MEDIAN}: too few for a "
                               f"median to mean anything"}
            continue
        med = statistics.median(sizes)
        if med <= 0:
            coins[c] = {"status": "UNJUDGEABLE", "n_windows": len(wins),
                        "why": "median window is empty"}
            continue
        thin = [(w, b) for w, b in wins if b < med * thin_frac]
        acct = [(w, b) for w, b in thin
                if gap_overlaps(gaps, c, w, w + WINDOW_S)]
        invis = [(w, b) for w, b in thin if (w, b) not in acct]
        coins[c] = {
            "status": "JUDGED", "n_windows": len(wins),
            "median_bytes": int(med),
            "n_thin": len(thin), "n_thin_accounted": len(acct),
            "n_thin_invisible": len(invis),
            "invisible_windows": [
                {"window_start": w,
                 "utc": dt.datetime.fromtimestamp(
                     w, dt.timezone.utc).strftime("%H:%M:%SZ"),
                 "bytes": b, "frac_of_median": round(b / med, 5)}
                for w, b in sorted(invis)],
        }
    judged = [c for c, v in coins.items() if v["status"] == "JUDGED"]
    total_invis = sum(coins[c]["n_thin_invisible"] for c in judged)
    return {
        "day": day, "coins": coins,
        "n_coins_judged": len(judged),
        "n_coins_unjudgeable": len(coins) - len(judged),
        # the predicate, computed -- never a printed conclusion (rule 10)
        "clean": len(judged) > 0 and total_invis == 0,
        "total_thin_invisible": total_invis,
        "total_thin_accounted": sum(coins[c]["n_thin_accounted"]
                                    for c in judged),
        "threshold_frac_of_median": thin_frac,
    }


def sensitivity(days: list[str], gaps, raw_root: Path = RAW,
                fracs=(0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.25,
                       0.35, 0.50)) -> list[dict]:
    """The whole curve, because a single reading privileges one choice.

    The threshold was picked after seeing which days fail it, so the honest
    receipt carries what depends on that choice and what does not. Emitted
    alongside the results rather than on request: a sensitivity nobody runs
    is a sensitivity nobody sees.
    """
    out = []
    for f in fracs:
        dirty = tot = 0
        for d in days:
            try:
                r = judge_day(d, gaps=gaps, raw_root=raw_root, thin_frac=f)
            except Refused:
                continue
            dirty += 0 if r["clean"] else 1
            tot += r["total_thin_invisible"]
        out.append({"threshold_frac_of_median": f, "days_with_invisible_loss":
                    dirty, "invisible_windows": tot})
    return out


def all_days(raw_root: Path = RAW) -> list[str]:
    """Derived from disk, never hardcoded — the day list in this repo went
    stale four times in three days when it was a literal."""
    if not raw_root.is_dir():
        return []
    return sorted(d for d in os.listdir(raw_root)
                  if d.isdigit() and (raw_root / d).is_dir())


# ---------------------------------------------------------------- falsifier
def selftest() -> int:
    """Rule 15: a positive control it MUST flag, a bad input it MUST refuse.

    Built on a synthetic tape in a temp dir, because a detector proved only
    against the real tape cannot show it would fire on data that has not
    happened yet.
    """
    import gzip
    import tempfile
    checks = []

    def ok(cond, label):
        checks.append(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            print(f"SELFTEST FAILED at check {len(checks)}")
            raise SystemExit(1)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "raw"
        day = "20260901"
        (root / day).mkdir(parents=True)
        base = int(dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc).timestamp())
        # 40 healthy windows, then 2 deliberately thin ones
        for i in range(40):
            w = base + i * WINDOW_S
            with gzip.open(root / day / f"btc-updown-5m-{w}.jsonl.gz",
                           "wb") as fh:
                fh.write(b'{"x":1}\n' * 5000)
        thin_w = []
        for i in (40, 41):
            w = base + i * WINDOW_S
            thin_w.append(w)
            with gzip.open(root / day / f"btc-updown-5m-{w}.jsonl.gz",
                           "wb") as fh:
                fh.write(b'{"x":1}\n' * 3)

        r = judge_day(day, gaps={}, raw_root=root)
        ok(r["coins"]["btc"]["n_thin"] == 2,
           "POSITIVE CONTROL: two deliberately thin windows among 40 healthy "
           f"ones are FLAGGED (got {r['coins']['btc']['n_thin']}) — a "
           "detector that has never fired is not a detector")
        ok(r["total_thin_invisible"] == 2 and r["clean"] is False,
           "and with NO gap row covering them they count as INVISIBLE, so "
           "the day's computed `clean` predicate is False")

        # the SAME tape, now with a gap row covering both -> accounted, not
        # invisible. This is the leg that proves the detector distinguishes
        # loss the ledger KNOWS from loss it does not.
        g = {"btc": [(thin_w[0] - 1, thin_w[1] + WINDOW_S + 1)]}
        r2 = judge_day(day, gaps=g, raw_root=root)
        ok(r2["coins"]["btc"]["n_thin"] == 2
           and r2["total_thin_invisible"] == 0
           and r2["total_thin_accounted"] == 2 and r2["clean"] is True,
           "DISCRIMINATION: the identical thin windows, once a gap row "
           "COVERS them, are ACCOUNTED and not invisible — the instrument "
           "measures what the ledger MISSED, not merely what was small")

        # healthy-only tape must come back clean, or every alarm above is
        # just the detector shouting at everything
        r3 = judge_day(day, gaps={}, raw_root=root, thin_frac=1e-9)
        ok(r3["total_thin_invisible"] == 0 and r3["clean"] is True,
           "NEGATIVE CONTROL: with the threshold driven to ~0 nothing is "
           "flagged and the day reads clean — the flags above come from the "
           "data, not from the code flagging unconditionally")

        # refusals: an absent day and an empty day are STATUSES, never a
        # clean report (rule 4, and the empty-set-passes trap)
        for bad, why in ((lambda: judge_day("20991231", gaps={},
                                            raw_root=root),
                          "an ABSENT day REFUSES rather than reporting 0 "
                          "thinned windows"),
                         (None, None)):
            if bad is None:
                continue
            try:
                bad()
                ok(False, why)
            except Refused:
                ok(True, f"KNOWN-BAD: {why}")

        empty = "20260902"
        (root / empty).mkdir()
        try:
            judge_day(empty, gaps={}, raw_root=root)
            ok(False, "an EMPTY day directory must refuse")
        except Refused:
            ok(True, "KNOWN-BAD: a day directory with NO window files "
                     "REFUSES — 0 thinned of 0 windows is the empty-set trap "
                     "that already passed once on the 08-27 arm")

        # the trailer read must be exact, not approximate: assert it against
        # a real decompress, or every number above rests on an assumption
        p = root / day / f"btc-updown-5m-{base}.jsonl.gz"
        ok(uncompressed_size(p) == len(gzip.open(p, "rb").read()),
           "the gzip-trailer size EQUALS the true decompressed length — the "
           "O(1) measure this whole scan rests on is exact, not a proxy")

    print(f"pm_tape_density selftests: {len(checks)} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", help="one UTC day token (default: all on disk)")
    ap.add_argument("--thin-frac", type=float, default=THIN_FRAC)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    days = [a.day] if a.day else all_days()
    if not days:
        print("no days on disk")
        return 1
    gaps = load_gaps()
    out, dirty = [], 0
    for d in days:
        try:
            r = judge_day(d, gaps=gaps, thin_frac=a.thin_frac)
        except Refused as ex:
            print(f"{d}  REFUSED: {ex}")
            continue
        out.append(r)
        if not r["clean"]:
            dirty += 1
    if a.json:
        print(json.dumps({
            # A CONSUMER MUST BE ABLE TO TELL A RESHAPE FROM AN ABSENCE.
            # This receipt shipped as a bare LIST, then became an object; DA's
            # reader iterated it, got dict KEYS, matched no row, and reported
            # "no measurement for this day" WHILE THE MEASUREMENT SAT IN THE
            # FILE (312b121). It did not crash and did not refuse -- it
            # reported absence, which is the worse failure because absence is
            # a plausible answer. My defect: I reshaped a PUBLISHED artifact
            # without a version, one message after asking DA to rename a
            # field in it. A version makes the next reshape loud instead of
            # silent, and it is the producer's job to supply it, not the
            # consumer's to guess.
            "schema": "pm_tape_density/2",
            "schema_note": ("v1 was a bare JSON LIST of day objects. Any "
                            "reader that does not recognise this string must "
                            "report SCHEMA_UNRECOGNISED, never a clean zero "
                            "and never UNMEASURED."),
            "days": out,
            "threshold_sensitivity": sensitivity(days, gaps),
            "note": ("the per-day window COUNTS are readings at "
                     f"threshold_frac_of_median={a.thin_frac}, not "
                     "quantities: see threshold_sensitivity. The DAY count "
                     "is stable at 7 across [0.01, 0.10]. This instrument "
                     "REPORTS and does not VETO (DA 6fc539f): the threshold "
                     "was chosen after seeing which days fail it, so a "
                     "GOVERNING bar must be pre-registered against days not "
                     "yet seen."),
        }, indent=1))
        return 0

    print(f"thin = window below {a.thin_frac:.0%} of its (day, coin) median "
          f"uncompressed size\n")
    print(f"{'day':10} {'coins':>6} {'thin':>5} {'accounted':>10} "
          f"{'INVISIBLE':>10}  verdict")
    for r in out:
        tot_thin = r["total_thin_accounted"] + r["total_thin_invisible"]
        print(f"{r['day']:10} {r['n_coins_judged']:6d} {tot_thin:5d} "
              f"{r['total_thin_accounted']:10d} "
              f"{r['total_thin_invisible']:10d}  "
              f"{'clean' if r['clean'] else 'INVISIBLE LOSS'}")
    print()
    for r in out:
        if r["clean"]:
            continue
        print(f"{r['day']} — windows the gap ledger does NOT record:")
        for c, v in sorted(r["coins"].items()):
            if v["status"] != "JUDGED" or not v["invisible_windows"]:
                continue
            times = ", ".join(f"{w['utc']}({w['frac_of_median']:.3%})"
                              for w in v["invisible_windows"][:6])
            more = "" if len(v["invisible_windows"]) <= 6 \
                else f" +{len(v['invisible_windows']) - 6} more"
            print(f"  {c:5} {v['n_thin_invisible']:3d}  {times}{more}")
    print(f"\n{dirty} of {len(out)} days carry loss no gap row records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
