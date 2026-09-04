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


#: DA10-R2: WHICH BRANCH ANSWERED. A tree can satisfy the resolver's test and
#: still lack what a consumer needs -- the test is "carries data/pm_5min/raw"
#: while consumers also read `derived/` and `data/mm_hf/`. Recording the branch
#: makes a short run self-explaining instead of a smaller number.
DATA_ROOT_BRANCH: str = "unresolved"

#: DA11-R1: this suite's own total, asserted over ran + skipped so an empty
#: data root cannot produce the same summary line as a complete one.
EXPECTED_CHECKS = 21


def _resolve_data_root() -> Path:
    global DATA_ROOT_BRANCH
    env = os.environ.get("PM_DATA_ROOT")
    if env:
        DATA_ROOT_BRANCH = "1_env_PM_DATA_ROOT"
        return Path(env)
    # THE TEST IS FOR THE TAPE, NOT FOR A DIRECTORY. A worktree checks out
    # the TRACKED receipts under data/pm_5min/derived, so `data/pm_5min`
    # exists there while `raw/` -- the gitignored tape this module actually
    # reads -- does not. Testing the parent directory picked the worktree and
    # then failed on the ledger; testing for the tape itself is the property.
    if (CODE_ROOT / "data" / "pm_5min" / "raw").is_dir():
        DATA_ROOT_BRANCH = "2_code_tree_carries_the_tape"
        return CODE_ROOT
    DATA_ROOT_BRANCH = "3_canonical"
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


#: Files that could not be READ during the current scan. A truncated archive
#: is a thin window and belongs in the data; an unreadable one is a fact about
#: the reader and belongs here. Reset per `scan_day`.
class UnreadableMember(Refused):
    """A member whose size could not be READ. Outside the codomain of a byte
    count, deliberately: 0 would be a measurement and this is the absence of
    one (the DA32-R1 predicate)."""


#: Every member that refused during the current `scan_day`, so the reader is
#: owed all of them rather than the first. NOT the signal -- the raise is.
UNREADABLE: list[dict] = []


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
    except OSError as e:
        # PHANTOM FAILURE (named 2026-09-04), CLOSED PROPERLY 2026-09-05.
        # Returning 0 here is right for a TRUNCATED archive -- that genuinely
        # IS a thin window -- and wrong for a file that could not be READ: a
        # permission error, an I/O error or a race also land here, and 0
        # makes every consumer report a DARK WINDOW.
        #
        # ROUND 32 LEFT THE 0 IN PLACE and moved the signal into a module
        # census that `scan_day` refuses on. **That still fails the codomain
        # predicate (DA32-R1/R2):** 0 is inside the codomain of a byte count,
        # so the refusal lived at ONE call site while every other caller --
        # `da_blackout_mask.uncompressed_for` among them -- got a silent 0
        # through a module-level list, which is action at a distance. The
        # value now LEAVES THE CODOMAIN by raising, so no caller can consume
        # an unreadable file as a measurement whether or not it knew to ask.
        UNREADABLE.append({"path": str(path), "error": repr(e)})
        raise UnreadableMember(
            f"REFUSED: {path} could not be READ ({e!r}). Its size is not 0 "
            f"bytes -- it is unknown, and 0 would read as a DARK window "
            f"everywhere downstream. An unreadable file is a fact about the "
            f"reader and must not be summarised as data.") from e


def scan_day(day: str, raw_root: Path = RAW) -> dict:
    """Aggregate uncompressed bytes per (coin, window) for one UTC day."""
    d = raw_root / day
    if not d.is_dir():
        raise Refused(f"no raw directory for {day} — an absent day is not a "
                      f"clean day, and reporting 0 thinned windows for it "
                      f"would be the empty-set trap")
    UNREADABLE.clear()
    agg: dict[tuple[str, int], int] = collections.defaultdict(int)
    for fn in os.listdir(d):
        m = _FN.match(fn)
        if m:
            try:
                agg[(m.group(1), int(m.group(2)))] += uncompressed_size(d / fn)
            except UnreadableMember:
                # Caught so the census is COMPLETE -- a reader is owed every
                # unreadable file, not the first one -- and refused below.
                # Nothing is added to the window: an unreadable member makes
                # its window's total unknown, not smaller.
                continue
    if not agg:
        raise Refused(f"{day} has a raw directory but NO window files — "
                      f"refusing rather than reporting a clean day")
    if UNREADABLE:
        # A DARKNESS THAT IS OURS, NOT THE FEED'S. Refusing here rather than
        # returning zeros keeps an I/O problem from being reported as a
        # blackout by every consumer downstream.
        raise Refused(
            f"REFUSED for {day}: {len(UNREADABLE)} window file(s) could not "
            f"be READ ({[u['path'] for u in UNREADABLE[:3]]}). Their size "
            f"would read as 0 bytes and every consumer would report a DARK "
            f"window -- a blackout finding produced by an I/O error rather "
            f"than by the feed. An unreadable file is a fact about the "
            f"reader and must not be summarised as data.")
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


def data_root_resolution_audit(module_dir: Path | None = None
                               ) -> dict:
    """Who still derives a DATA path the way RR12-1 forbade, and who pins one?

    THE RULE THIS MODULE OWNS, stated once and then CHECKED rather than
    trusted: code paths come from `__file__`, data paths come from
    `_resolve_data_root()`. RR12-1 was that rule's first violation and it was
    repaired at `da_blackout_mask` and `da_forward_day_verify` by importing
    the resolution from here -- but a repair applied at the sites that were
    noticed is not a class closed, which is how `flow_intensity` kept the old
    expression until round 23 and quietly emptied the accrual selector for
    every worktree run.

    TWO CATEGORIES, DIFFERENT VERDICTS, and collapsing them would be wrong:

      * `derived_data_roots` -- a module-level CONSTANT that becomes a DATA
        path off a `__file__`-derived root. THIS is RR12-1's defect: from a
        worktree it resolves to a tree with no tape and the reader sees an
        EMPTY population, which is the failure that reads like a clean result.
      * `other_joins` -- the same join inside a function or a suite. A USE,
        not a declaration, and reported apart because counting them together
        makes the audit flag its own falsifier (it did).
      * `pinned_canonical_paths` -- a data path hardcoded to the canonical
        tree. NOT the same defect and mostly not a defect at all: it always
        reads the real data, so it fails safe. Its cost is different and is
        stated rather than scored -- such a path cannot be redirected, so a
        run under PM_DATA_ROOT redirects the readers that resolve and not
        these, giving a PARTIAL isolation. This repo has already paid for the
        general form of that (an isolation covering only the visible half
        reads as isolated).

    REPORTED, NEVER ENFORCED (rule 14). Several of the pinned paths are other
    seats' files, and DA's own two -- the era ledger and the collector log --
    are canonical deliberately.

    ONE THING THIS SCANNER GOT WRONG FIRST, AND ITS OWN FALSIFIER CAUGHT IT.
    The root-assignment pattern was anchored with `^` and compiled WITHOUT
    `re.M`, so `^` matched only the start of the FILE: every root declared on
    any line but the first was invisible and the scan returned a clean 0 over
    164 files. The planted-violation control refused to fire and that is the
    only reason the zero was not believed -- rule 15 exactly, on the
    instrument rather than on the code it audits.
    """
    d = (Path(__file__).resolve().parent if module_dir is None
         else Path(module_dir))
    if not d.is_dir():
        raise Refused(
            f"REFUSED: no module directory at {d}. An audit that cannot read "
            f"the source must refuse, never report a clean surface -- a "
            f"silent regex mismatch once reported exactly that.")
    _asg = re.compile(
        r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*Path\(__file__\)\.resolve\(\)"
        r"\.parents?\[?\d*\]?", re.M)
    _pin = re.compile(r"""Path\(\s*["']/home/yuqing/ctaNew/data/""")
    _JOIN = r"\b%s\s*/\s*[\"']data/"
    derived, other, pinned = [], [], []
    files = sorted(d.glob("*.py"))
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except OSError:                                      # pragma: no cover
            continue
        names = set(_asg.findall(txt))
        for n, line in enumerate(txt.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if _pin.search(line):
                pinned.append({"file": f.name, "line": n,
                               "text": line.strip()[:110]})
            for nm in names:
                if not re.search(_JOIN % nm, line):
                    continue
                rec = {"file": f.name, "line": n, "root": nm,
                       "text": line.strip()[:110]}
                # A DECLARATION is the class RR12-1 names: a module-level
                # constant that BECOMES a data path. A join inside a function
                # or a selftest is a USE -- reported separately, because
                # counting the two together makes an audit of its own suite
                # flag itself, which this one did.
                if re.match(r"^[A-Z_][A-Z0-9_]*\s*=", line):
                    derived.append(rec)
                else:
                    other.append(rec)
    return {
        "module_dir": str(d), "n_files_scanned": len(files),
        "derived_data_roots": derived,
        "n_derived_data_roots": len(derived),
        "derived_root_files": sorted({x["file"] for x in derived}),
        "other_joins": other,
        "n_other_joins": len(other),
        "pinned_canonical_paths": pinned,
        "n_pinned_canonical_paths": len(pinned),
        "pinned_files": sorted({x["file"] for x in pinned}),
        "rr12_1_class_closed": not derived,
        "role": "REPORTED_NOT_ENFORCED",
        "note": ("`rr12_1_class_closed` is about the FIRST category only. "
                 "The pinned paths are a different trade-off and are listed "
                 "so the partial-isolation cost is visible, not so they can "
                 "be counted as violations."),
    }


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
    skipped: list[tuple[str, str]] = []

    def skip(name, absent):
        """DA11-R1: a skip is a STATUS, never a pass.

        This module used to `checks.append(True)` after printing the SKIP, so
        a complete root and an EMPTY root produced the BYTE-IDENTICAL summary
        `9 checks passed` -- and that summary is exactly the one line
        `v5_deploy_gates` captures as the gate's result. The gate could not
        tell the two apart. Same shape as the verifier's fix (DA10-R1).
        """
        skipped.append((name, str(absent)))
        print(f"  SKIP {name}: absent input {absent}")

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

    # THE CARRIED ITEM, CLOSED. This suite passed 7/7 against an EMPTY data
    # root: every check above runs on a synthetic tape in a temp dir, so the
    # RESOLVED root was never exercised and `all_days() -> []` read as a clean
    # report over nothing. The resolver's answer is now asserted, and an empty
    # root is a NAMED status rather than a silent pass -- the same idiom as
    # the verifier's skips (DA10-R1).
    ok(DATA_ROOT_BRANCH in ("1_env_PM_DATA_ROOT",
                            "2_code_tree_carries_the_tape", "3_canonical"),
       f"the data root resolved by a NAMED branch ({DATA_ROOT_BRANCH}) -- a "
       f"reader of any artifact can say which tree answered")
    _days = all_days()
    if _days:
        ok(len(_days) > 0,
           f"and the resolved root {DATA_ROOT} carries {len(_days)} day(s), "
           f"so this suite ran against a tape that exists")
    else:
        skip("resolved-root-carries-days",
             f"{DATA_ROOT}/data/pm_5min/raw (branch {DATA_ROOT_BRANCH})")
    # ---- PHANTOM FAILURE: an unreadable file is not an empty one ---------
    import os as _os32
    import tempfile as _tf32
    with _tf32.TemporaryDirectory() as _t32:
        _r32 = Path(_t32) / "20300301"
        _r32.mkdir()
        import gzip as _gz32
        for _i in range(2):
            (_r32 / f"aaa-updown-5m-{1900000000 + _i * 300}.jsonl.gz"
             ).write_bytes(_gz32.compress(b"x" * 5000))
        _agg32 = scan_day("20300301", Path(_t32))
        ok(len(_agg32) == 2 and all(v > 0 for v in _agg32.values()),
           "PHANTOM-1 POSITIVE CONTROL: a readable day scans and every window "
           "carries a positive size")
        _bad32 = _r32 / "aaa-updown-5m-1900000600.jsonl.gz"
        _bad32.write_bytes(_gz32.compress(b"y" * 5000))
        _os32.chmod(_bad32, 0o000)
        try:
            scan_day("20300301", Path(_t32))
            _os32.chmod(_bad32, 0o644)
            ok(_os32.geteuid() == 0,
               "PHANTOM-2 an UNREADABLE window file must make scan_day REFUSE "
               "(skipped only when running as root, where chmod cannot deny)")
        except Refused as _e32:
            _os32.chmod(_bad32, 0o644)
            ok("could not be READ" in str(_e32)
               and "blackout finding produced by an I/O error" in str(_e32),
               f"PHANTOM-2 AN UNREADABLE FILE REFUSES BY NAME rather than "
               f"reading as 0 bytes. Before this it returned 0, and 0 is "
               f"DARK: `da_dark_interval_scan` would report an interval and "
               f"`da_blackout_mask` would mask it -- a BLACKOUT FINDING "
               f"produced by a permission error. A negative verdict from a "
               f"read that never happened ({str(_e32)[:60]}...)")

    # ---- DA32-R1/R2: the value must LEAVE THE CODOMAIN, not be censused --
    import tempfile as _tfc
    with _tfc.TemporaryDirectory() as _t:
        _d = Path(_t)
        _trunc = _d / "btc_0.jsonl.gz"
        _trunc.write_bytes(b"\x1f\x8b\x08")          # too short for ISIZE
        ok(uncompressed_size(_trunc) == 0,
           "CODOMAIN-1 a TRUNCATED archive still reads 0 and does NOT raise: "
           "a truncated file genuinely IS a thin window, and the fix must "
           "not turn a real measurement into a refusal")
        _bad = _d / "eth_0.jsonl.gz"
        _bad.write_bytes(b"\x1f\x8b\x08" + b"\x00" * 30)
        import os as _osc
        _osc.chmod(_bad, 0o000)
        if _osc.geteuid() == 0:
            _osc.chmod(_bad, 0o644)
            skip("CODOMAIN-2 needs a file chmod can deny (not root)")
            skip("CODOMAIN-3 needs a file chmod can deny (not root)")
        else:
            try:
                uncompressed_size(_bad)
                _osc.chmod(_bad, 0o644)
                ok(False, "CODOMAIN-2 an unreadable member must RAISE")
            except UnreadableMember as _e:
                ok("is not 0 bytes -- it is unknown" in str(_e),
                   "CODOMAIN-2 THE FIX ROUND 32 SHOULD HAVE MADE: an "
                   "unreadable member RAISES rather than returning 0 into a "
                   "module census. 0 is INSIDE the codomain of a byte count, "
                   "so the refusal only protected the one call site that "
                   "knew to ask; the raise protects every caller, including "
                   "the ones that do not know this module exists")
                _osc.chmod(_bad, 0o644)
            _osc.chmod(_bad, 0o000)
            try:
                import da_blackout_mask as _BMc
                # THE CLASS THE WRAPPER ACTUALLY RAISES. Under --selftest this
                # module is `__main__`, so `__main__.UnreadableMember` and the
                # freshly-imported `pm_tape_density.UnreadableMember` are two
                # different class objects and a bare `except UnreadableMember`
                # would MISS the raise -- and a missed raise here reads as
                # "the wrapper did not propagate", which is a phantom failure
                # inside the check that closes one.
                _BMc.uncompressed_for(_bad)
                _osc.chmod(_bad, 0o644)
                ok(False, "CODOMAIN-3 uncompressed_for must propagate")
            except _BMc.TD.UnreadableMember as _e3:
                ok("via uncompressed_for" in str(_e3),
                   "CODOMAIN-3 (DA32-R2) the wrapper that sat OUTSIDE the "
                   "refusal now propagates and names its own path -- it used "
                   "to hand its fixture a silent 0, a thin window that never "
                   "existed")
                _osc.chmod(_bad, 0o644)
            except ImportError:
                skip("CODOMAIN-3 da_blackout_mask not importable here")

    # ---------------------------------------------------------------- RR12-1
    # THE CLASS, NOT THE INSTANCE. The scanner is driven on a planted
    # violation before its zero on the real tree is allowed to mean anything
    # (rule 15: a zero from an instrument that never proved it can fire is not
    # a result -- a silent regex mismatch once reported a clean surface).
    import tempfile as _tfa
    with _tfa.TemporaryDirectory() as _t:
        _d = Path(_t)
        (_d / "clean.py").write_text(
            "from pathlib import Path\n"
            "CODE_ROOT = Path(__file__).resolve().parents[2]\n"
            "DOC = CODE_ROOT / 'docs/README.md'\n"
            "# COMMENTED = CODE_ROOT / \"data/pm_5min\"\n", encoding="utf-8")
        _a0 = data_root_resolution_audit(_d)
        ok(_a0["n_derived_data_roots"] == 0
           and _a0["n_pinned_canonical_paths"] == 0
           and _a0["rr12_1_class_closed"] is True,
           "RR12-1 AUDIT admits a clean file: a `__file__`-derived root is "
           "fine when what hangs off it is CODE, and a commented-out join is "
           "not a join")
        (_d / "guilty.py").write_text(
            "from pathlib import Path\n"
            "REPO = Path(__file__).resolve().parents[2]\n"
            "PM = REPO / \"data/pm_5min\"\n", encoding="utf-8")
        _a1 = data_root_resolution_audit(_d)
        # THE MESSAGE MUST NOT INDEX WHAT THE CHECK IS TESTING FOR. Written
        # the obvious way -- `_a1["derived_data_roots"][0]["root"]` inside the
        # label -- this check FAILED WITH AN IndexError instead of by name the
        # moment the scanner stopped finding anything, which is the shape
        # R-495 (D) named. Driven: re-introducing the anchor bug now goes red
        # BY NAME with the empty list shown.
        _hits = _a1["derived_data_roots"]
        ok(len(_hits) == 1 and _hits[0]["file"] == "guilty.py"
           and _hits[0]["root"] == "REPO"
           and _a1["rr12_1_class_closed"] is False,
           f"RR12-1 AUDIT FIRES on a planted DECLARATION and names the root "
           f"(got {_hits!r}) -- the exact expression "
           f"`flow_intensity` carried until this round. NOTE: the FIRST "
           f"version of this scanner anchored on `^` without re.M, so it saw "
           f"only line 1 of each file and returned a clean 0 across 164 "
           f"files. THIS control is the only reason that zero was not "
           f"believed")
        (_d / "uses.py").write_text(
            "from pathlib import Path\n"
            "ROOT = Path(__file__).resolve().parents[2]\n"
            "def f():\n"
            "    return ROOT / \"data/pm_5min/derived\"\n", encoding="utf-8")
        _a1b = data_root_resolution_audit(_d)
        ok(_a1b["n_derived_data_roots"] == 1 and _a1b["n_other_joins"] == 1
           and _a1b["other_joins"][0]["file"] == "uses.py",
           "RR12-1 AUDIT separates a DECLARATION from a USE: the same join "
           "inside a function is reported as an other-join and does not move "
           "the class count. Without that split an audit of its own suite "
           "flags its own falsifier, which this one did")
        (_d / "pinned.py").write_text(
            "from pathlib import Path\n"
            "DERIVED = Path(\"/home/yuqing/ctaNew/data/pm_5min/derived\")\n",
            encoding="utf-8")
        _a2 = data_root_resolution_audit(_d)
        ok(_a2["n_derived_data_roots"] == 1
           and _a2["n_pinned_canonical_paths"] == 1
           and _a2["pinned_files"] == ["pinned.py"],
           "RR12-1 AUDIT keeps the two categories APART: a hardcoded "
           "canonical path is counted as PINNED and not as a derived-root "
           "violation. They fail in opposite directions -- one reads an "
           "empty tree, the other always reads the real one -- so scoring "
           "them together would hide which is which")
    try:
        data_root_resolution_audit(Path("/nonexistent/pm_research"))
        ok(False, "RR12-1 AUDIT must refuse an absent module dir")
    except Refused as _e:
        ok("never report a clean surface" in str(_e),
           "RR12-1 AUDIT REFUSES an unreadable source tree -- 'I could not "
           "read it' must never come back as 'nothing to find'")
    _real = data_root_resolution_audit()
    _ACCRUAL = ("flow_intensity.py", "warning_window.py", "pm_tape_density.py",
                "da_forward_day_verify.py", "da_blackout_mask.py")
    _bad = [f for f in _ACCRUAL if f in _real["derived_root_files"]]
    ok(not _bad,
       f"RR12-1 THE ACCRUAL PATH IS CLEAN, and that is the claim this round "
       f"is entitled to make: none of {list(_ACCRUAL)} declares a data root "
       f"from `__file__` (offenders: {_bad}). `flow_intensity` was the last "
       f"one and it is repaired here; `da_blackout_mask` and "
       f"`da_forward_day_verify` already import this module's resolution, so "
       f"RR12-1's ORIGINAL SITE needs nothing further")
    ok(_real["rr12_1_class_closed"] is False
       and _real["n_derived_data_roots"] > 0,
       f"RR12-1 THE CLASS IS NOT CLOSED, MEASURED AND NOT ASSUMED: "
       f"{_real['n_derived_data_roots']} declaration(s) across "
       f"{len(_real['derived_root_files'])} of {_real['n_files_scanned']} "
       f"files still resolve a data path from `__file__`, plus "
       f"{_real['n_other_joins']} other join(s) and "
       f"{_real['n_pinned_canonical_paths']} canonical pin(s). "
       f"`pm_host_load_join.py` is among them -- DA20-R4, already filed, now "
       f"shown to be ONE MEMBER OF A CLASS rather than a lone nit. This "
       f"check asserts the state is REPORTED, not that it is zero: closing "
       f"26 files is a dispatch, not a side effect")

    if len(checks) + len(skipped) != EXPECTED_CHECKS:
        raise AssertionError(
            f"pm_tape_density selftest FAILED: {len(checks)} ran + "
            f"{len(skipped)} skipped = {len(checks) + len(skipped)}, expected "
            f"{EXPECTED_CHECKS}. A check that neither ran nor named itself as "
            f"a skip has VANISHED.")
    print(f"pm_tape_density selftests: {len(checks)} checks passed"
          + (f" ({len(skipped)} skipped; ran+skipped={EXPECTED_CHECKS})"
             if skipped else f" (0 skipped; ran+skipped={EXPECTED_CHECKS})"))
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


def data_root_provenance() -> dict:
    """The pair a reader needs to answer "which tree produced this?".

    DA10-R2: emitted by the mask, the verdict AND the preflight, so no reader
    has to know how the process was launched. The BRANCH is carried too: the
    resolver's test is "carries data/pm_5min/raw", and a tree can pass it while
    lacking `derived/` or `data/mm_hf/` -- which is exactly how a 238-check run
    happened with nothing saying why.
    """
    return {
        "code_root": str(CODE_ROOT),
        "data_root": str(DATA_ROOT),
        "data_root_branch": DATA_ROOT_BRANCH,
        "branches": ("1_env_PM_DATA_ROOT | 2_code_tree_carries_the_tape "
                     "| 3_canonical"),
        "resolver_predicate": "the tree carries data/pm_5min/raw",
        "predicate_caveat": ("consumers also read data/pm_5min/derived and "
                             "data/mm_hf; a tree can satisfy the predicate "
                             "and still lack those, so read the branch"),
    }
