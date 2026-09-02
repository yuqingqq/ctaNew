#!/usr/bin/env python3
"""Cross-venue discrimination of the blackout signature.

Implements `plans/DA_CROSS_VENUE_DISCRIMINATION_DESIGN.md`, which was
COMMITTED BEFORE any rate inside an event window was read (rule 6). The
thresholds, the outcome table and the window derivation in this file are the
declared ones; nothing here was chosen after seeing an answer.

Three collectors, ONE host, ONE network path, THREE venues. If all three thin
together the cause is host or path; if only Polymarket thins it is
venue-side. No amount of Polymarket-only evidence separates those.

    python3 live/pm_research/da_cross_venue_forensics.py --selftest
    python3 live/pm_research/da_cross_venue_forensics.py --events
    python3 live/pm_research/da_cross_venue_forensics.py --day 20260902
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_tape_density as TD                                   # noqa: E402

REPO = Path("/home/yuqing/ctaNew")
WINDOW_S = 300

#: DECLARED: the 10%-of-median cut is REUSED from `content_liveness_for`'s
#: `note_10pct`, not invented after seeing three events.
THIN_CUT = 0.10
#: DECLARED outcome thresholds.
THIN_MAJORITY = 0.50
THIN_QUIET = 0.10

VENUES: dict[str, dict] = {
    "polymarket": {
        "logs": [REPO / "data/pm_5min/collector.log"],
        "re": re.compile(
            r"^\[pm\]\s+(?:(\d{4})-(\d{2})-(\d{2})T)?(\d{2}):(\d{2}):(\d{2})Z"
            r".*?\bmsgs=(\d+)"),
        "counters": 1,
        "sentinel": re.compile(r"^\[pm\]"),
    },
    "binance_hf": {
        "logs": [REPO / "data/mm_hf/collector.log.pre-reboot-20260826",
                 REPO / "data/mm_hf/collector.log"],
        "re": re.compile(
            r"^\[hf\]\s+(?:(\d{4})-(\d{2})-(\d{2})T)?(\d{2}):(\d{2}):(\d{2})Z"
            r"\s+bookTicker=(\d+)\s+depth\d+=(\d+)\s+trade=(\d+)"),
        "counters": 3,
        "sentinel": re.compile(r"^\[hf\]"),
        "lat": re.compile(
            r"^\[hf\]\s+(?:\d{4}-\d{2}-\d{2}T)?(\d{2}):(\d{2}):(\d{2})Z"
            r".*recv-lat~(\d+)ms"),
    },
    "hyperliquid": {
        "logs": [REPO / "data/mm_hf/hl_collector.log.pre-reboot-20260826",
                 REPO / "data/mm_hf/hl_collector.log"],
        "re": re.compile(
            r"^\[hl\]\s+(?:(\d{4})-(\d{2})-(\d{2})T)?(\d{2}):(\d{2}):(\d{2})Z"
            r"\s+bbo=(\d+)\s+l2Book=(\d+)\s+trades=(\d+)"),
        "counters": 3,
        "sentinel": re.compile(r"^\[hl\]"),
    },
}


class Unmeasured(Exception):
    """A population this instrument must not summarise (rule 4)."""


def dated_heartbeats(text: str, rx: re.Pattern, n_counters: int,
                     sentinel: re.Pattern, anchor_epoch: float
                     ) -> list[tuple[float, int]]:
    """(epoch, summed cumulative counter) — ONE backward walk, file order.

    Generalised from `content_liveness_for`'s walker, which two bugs were
    already found in (Q-DA-196): anchoring a dateless block to the FILE rather
    than to the entry after it, and a `sorted()` that made the monotonicity
    check vacuous. Both fixes are carried here: one walk in file order (an
    append-only log's order IS time order) and a monotonicity REFUSAL.

    A >24 h silent gap is NOT detectable from dateless stamps and is declared
    rather than guarded -- a guard that cannot fire is not a guard.
    """
    entries, saw = [], False
    for ln in text.splitlines():
        if sentinel.match(ln):
            saw = True
        m = rx.match(ln)
        if not m:
            continue
        g = m.groups()
        y, mo, d, hh, mm, ss = g[:6]
        counters = [int(x) for x in g[6:6 + n_counters]]
        sod = int(hh) * 3600 + int(mm) * 60 + int(ss)
        exact = (dt.datetime(int(y), int(mo), int(d), int(hh), int(mm),
                             int(ss), tzinfo=dt.timezone.utc).timestamp()
                 if y else None)
        entries.append((sod, sum(counters), exact))
    if not entries:
        if saw:
            raise Unmeasured(
                "REFUSED: the log carries its own prefix lines but NOT ONE "
                "matches the heartbeat shape. That is a FORMAT CHANGE, not an "
                "absence of history -- reporting it as 'no data' would read a "
                "rename as a missing venue, which is exactly the alibi shape "
                "this design refuses.")
        return []
    out, cur = [], None
    for sod, tot, exact in reversed(entries):
        if exact is not None:
            ts = exact
        else:
            ref = anchor_epoch if cur is None else cur
            ts = int(ref // 86400) * 86400 + sod
            if ts > ref:
                ts -= 86400
        out.append((ts, tot))
        cur = ts
    out.reverse()
    for i in range(1, len(out)):
        if out[i][0] <= out[i - 1][0]:
            raise Unmeasured(
                "REFUSED: heartbeat dates do not reconstruct monotonically; "
                "two stamps resolve to the same instant or step backward, so "
                "one window's traffic could be attributed to another.")
    return out


def rate_series(venue: str, logs=None) -> list[tuple[float, float]]:
    """(interval-end epoch, messages/second) for one venue.

    Counters are CUMULATIVE on all three collectors (verified on the first
    2,000 lines of each log, which predate every event window). A counter
    RESET (restart) yields a negative delta and is DROPPED as a status rather
    than counted as zero traffic -- a restart is not a blackout, and calling
    it one would manufacture the very signature under investigation.
    """
    spec = VENUES[venue]
    paths = spec["logs"] if logs is None else logs
    pts: list[tuple[float, int]] = []
    seen_any = False
    for p in paths:
        if not Path(p).exists():
            continue
        seen_any = True
        anchor = Path(p).stat().st_mtime
        pts += dated_heartbeats(Path(p).read_text(errors="replace"),
                                spec["re"], spec["counters"],
                                spec["sentinel"], anchor)
    if not seen_any:
        raise Unmeasured(f"no log file present for {venue}")
    pts.sort()
    out = []
    for i in range(1, len(pts)):
        dt_s = pts[i][0] - pts[i - 1][0]
        dv = pts[i][1] - pts[i - 1][1]
        if dt_s <= 0 or dv < 0:          # restart / reset: a STATUS, not a 0
            continue
        out.append((pts[i][0], dv / dt_s))
    return out


def day_bounds(day: str) -> tuple[int, int]:
    d = dt.datetime.strptime(day, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp()), int(d.timestamp()) + 86400


def thin_fraction(series, day: str, w0: float, w1: float) -> dict[str, Any]:
    """Share of one-minute intervals inside [w0, w1) below 10% of the DAY's
    median rate. The denominator is the day, so a venue is compared against
    its own normal rather than against another venue's."""
    lo, hi = day_bounds(day)
    day_pts = [r for t, r in series if lo <= t < hi]
    win = [(t, r) for t, r in series if w0 <= t < w1]
    if len(day_pts) < 60:
        return {"status": "UNMEASURED",
                "why": f"only {len(day_pts)} one-minute intervals on {day}; "
                       f"the log does not cover this day densely enough to "
                       f"have a median. UNMEASURED is not 'normal'.",
                "n_day_intervals": len(day_pts), "n_window_intervals": len(win)}
    if not win:
        return {"status": "UNMEASURED",
                "why": "no heartbeat interval falls inside the window; the "
                       "log does not reach it. UNMEASURED is not 'normal'.",
                "n_day_intervals": len(day_pts), "n_window_intervals": 0}
    med = statistics.median(day_pts)
    if med <= 0:
        return {"status": "UNMEASURED", "why": "the day's median rate is 0",
                "n_day_intervals": len(day_pts),
                "n_window_intervals": len(win)}
    thin = [(t, r) for t, r in win if r < med * THIN_CUT]
    return {"status": "MEASURED",
            "n_day_intervals": len(day_pts),
            "n_window_intervals": len(win),
            "day_median_msgs_per_s": round(med, 3),
            "window_median_msgs_per_s": round(
                statistics.median([r for _, r in win]), 3),
            "window_min_msgs_per_s": round(min(r for _, r in win), 4),
            "n_thin": len(thin),
            "thin_fraction": round(len(thin) / len(win), 4),
            "first_thin_utc": _iso(thin[0][0]) if thin else None,
            "last_thin_utc": _iso(thin[-1][0]) if thin else None}


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def verdict(pm: dict, hf: dict, hl: dict) -> dict[str, Any]:
    """THE DECLARED OUTCOME TABLE. No result is forced into a bucket."""
    if any(x["status"] != "MEASURED" for x in (pm, hf, hl)):
        return {"verdict": "UNRESOLVED-UNMEASURED",
                "why": "at least one venue is UNMEASURED in this window, and "
                       "an unmeasured venue is not an alibi (rule 4)",
                "unmeasured": [n for n, x in (("polymarket", pm),
                                              ("binance_hf", hf),
                                              ("hyperliquid", hl))
                               if x["status"] != "MEASURED"]}
    p, f, l = (x["thin_fraction"] for x in (pm, hf, hl))
    if p >= THIN_MAJORITY and f < THIN_QUIET and l < THIN_QUIET:
        v, why = "H1-POLYMARKET-SIDE", (
            "Polymarket thinned while BOTH other venues on the same host and "
            "the same network path stayed at normal rate")
    elif min(p, f, l) >= THIN_MAJORITY:
        v, why = "H2-HOST-OR-PATH", (
            "all three venues thinned together, which no venue-side cause can "
            "produce")
    elif p >= THIN_MAJORITY and (f >= THIN_MAJORITY) != (l >= THIN_MAJORITY):
        v, why = "UNRESOLVED-ASYMMETRIC", (
            "Polymarket and exactly one other venue thinned; that pattern "
            "points at neither hypothesis as declared")
    else:
        v, why = "UNRESOLVED-OTHER", "the declared conditions do not apply"
    return {"verdict": v, "why": why,
            "thin_fractions": {"polymarket": p, "binance_hf": f,
                               "hyperliquid": l},
            "declared_thresholds": {"majority": THIN_MAJORITY,
                                    "quiet": THIN_QUIET, "cut": THIN_CUT}}


def derive_window(day: str, raw_root: Path | None = None,
                  gaps=None) -> dict[str, Any]:
    """W = the longest contiguous invisible-thin run on `day`, unioned across
    coins. DERIVED by the frozen detector, never hand-picked."""
    # raw_root/gaps are PARAMETERS, not module lookups: `scan_day`'s default
    # binds `TD.RAW` at def time, so patching the module constant does
    # nothing and the fixture below would have silently scanned the real
    # tape -- a test that reads production data is not a fixture.
    gaps = TD.load_gaps() if gaps is None else gaps
    agg = TD.scan_day(day, TD.RAW if raw_root is None else raw_root)
    per = collections.defaultdict(list)
    for (c, w), b in agg.items():
        per[c].append((w, b))
    best = None
    per_coin = {}
    for c, wins in sorted(per.items()):
        wins.sort()
        if len(wins) < TD.MIN_WINDOWS_FOR_MEDIAN:
            continue
        med = statistics.median([b for _, b in wins])
        if med <= 0:
            continue
        inv = [w for w, b in wins
               if b < med * TD.THIN_FRAC
               and not TD.gap_overlaps(gaps, c, w, w + WINDOW_S)]
        run, cur = [], []
        for w in inv:
            if cur and w - cur[-1] == WINDOW_S:
                cur.append(w)
            else:
                if cur:
                    run.append(cur)
                cur = [w]
        if cur:
            run.append(cur)
        if not run:
            continue
        longest = max(run, key=len)
        per_coin[c] = {"n_windows": len(longest),
                       "start_utc": _iso(longest[0]),
                       "end_utc": _iso(longest[-1] + WINDOW_S)}
        # COMPARE LIKE WITH LIKE. This compared a window COUNT against a
        # DURATION IN SECONDS (`best[1] - best[0]`), so after the first coin
        # set `best` no later coin could ever beat it and the ALPHABETICALLY
        # FIRST coin won. On 08-26 and 08-31 bnb happened to hold the maximum
        # so the windows were right by luck; on 09-02 it returned bnb's 14
        # windows instead of btc's 40 -- a real event reported at a third of
        # its length. Caught by the derived window disagreeing with the run
        # structure already measured in Q-DA-202.
        if best is None or len(longest) > best[2]:
            best = (longest[0], longest[-1] + WINDOW_S, len(longest))
    if best is None:
        raise Unmeasured(f"{day} has no invisible-thin run to derive a window "
                         f"from")
    return {"day": day, "w0": best[0], "w1": best[1],
            "n_windows": best[2], "start_utc": _iso(best[0]),
            "end_utc": _iso(best[1]), "per_coin_longest_run": per_coin,
            "offsets_utc": sorted({v["end_utc"] for v in per_coin.values()})}


def host_window(w0: float, w1: float) -> dict[str, Any]:
    """The R-163 resource-monitor CSV over [w0, w1), from the journal."""
    try:
        out = subprocess.run(
            ["journalctl", "--user", "-u", "resource-monitor", "--no-pager",
             "-o", "cat", "--since",
             dt.datetime.fromtimestamp(w0 - 3600, dt.timezone.utc).strftime(
                 "%Y-%m-%d %H:%M:%S"), "--until",
             dt.datetime.fromtimestamp(w1 + 3600, dt.timezone.utc).strftime(
                 "%Y-%m-%d %H:%M:%S"), "--utc"],
            capture_output=True, text=True, timeout=120)
    except Exception as e:                                  # pragma: no cover
        return {"status": "UNMEASURED", "why": f"journalctl failed: {e!r}"}
    rows, alerts = [], []
    for ln in out.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("ALERT"):
            alerts.append(ln)
            continue
        f = ln.split(",")
        if len(f) != 9 or f[0] == "ts":
            continue
        try:
            ts = dt.datetime.strptime(f[0], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=dt.timezone.utc).timestamp()
            rows.append((ts, int(f[1]), int(f[2]), float(f[6]), int(f[4])))
        except ValueError:
            continue
    if not rows:
        return {"status": "UNMEASURED",
                "why": "the resource-monitor journal does not reach this "
                       "window. An absent host record is NOT evidence the "
                       "host was healthy (R-366).",
                "n_rows": 0}
    inw = [r for r in rows if w0 <= r[0] < w1]
    outw = [r for r in rows if not (w0 <= r[0] < w1)]
    if not inw:
        return {"status": "UNMEASURED",
                "why": "rows exist nearby but none inside the window",
                "n_rows": len(rows), "n_in_window": 0}
    def stat(v, i):
        return {"min": min(x[i] for x in v), "median": statistics.median(
            [x[i] for x in v]), "max": max(x[i] for x in v)}
    return {"status": "MEASURED", "n_rows": len(rows), "n_in_window": len(inw),
            "n_outside_window_context": len(outw),
            "mem_avail_mib": stat(inw, 1), "swap_used_mib": stat(inw, 2),
            "load1": stat(inw, 3), "collectors_mib": stat(inw, 4),
            "alerts_in_range": alerts[:10], "n_alerts_in_range": len(alerts),
            # DECLARED: an excursion is what would SUPPORT an H2-host cause.
            "excursion_mem_below_alert_floor": min(x[1] for x in inw) < 4096,
            "excursion_swap_in_use": max(x[2] for x in inw) > 512,
            "load1_median_in_vs_out": (
                statistics.median([x[3] for x in inw]),
                statistics.median([x[3] for x in outw]) if outw else None),
            "absence_note": ("no excursion REMOVES ONE H2 MECHANISM; it does "
                             "not prove H1 (R-366)")}


def hf_latency(w0: float, w1: float) -> dict[str, Any]:
    """HF `recv-lat~NNms` inside the window — the free path signal."""
    spec = VENUES["binance_hf"]
    pts = []
    for p in spec["logs"]:
        if not Path(p).exists():
            continue
        anchor = Path(p).stat().st_mtime
        cur = None
        for ln in reversed(Path(p).read_text(errors="replace").splitlines()):
            m = spec["lat"].match(ln)
            if not m:
                continue
            hh, mm, ss, ms = m.groups()
            sod = int(hh) * 3600 + int(mm) * 60 + int(ss)
            ref = anchor if cur is None else cur
            ts = int(ref // 86400) * 86400 + sod
            if ts > ref:
                ts -= 86400
            cur = ts
            pts.append((ts, int(ms)))
    inw = [ms for t, ms in pts if w0 <= t < w1]
    outw = [ms for t, ms in pts if not (w0 <= t < w1)]
    if not inw:
        return {"status": "UNMEASURED", "n_in_window": 0,
                "why": "no HF latency sample inside the window"}
    return {"status": "MEASURED", "n_in_window": len(inw),
            "median_ms_in_window": statistics.median(inw),
            "max_ms_in_window": max(inw),
            "median_ms_outside": statistics.median(outw) if outw else None,
            "note": ("a degraded PATH should raise this; flat latency with "
                     "normal rates is positive evidence the path was healthy")}


def analyse(day: str, w0=None, w1=None, label="") -> dict[str, Any]:
    if w0 is None:
        d = derive_window(day)
        w0, w1 = d["w0"], d["w1"]
    else:
        d = {"day": day, "w0": w0, "w1": w1, "start_utc": _iso(w0),
             "end_utc": _iso(w1), "derived": False}
    legs = {}
    for v in VENUES:
        try:
            legs[v] = thin_fraction(rate_series(v), day, w0, w1)
        except Unmeasured as e:
            legs[v] = {"status": "UNMEASURED", "why": str(e)}
    return {"label": label, "window": d, "venues": legs,
            "verdict": verdict(legs["polymarket"], legs["binance_hf"],
                               legs["hyperliquid"]),
            "host": host_window(w0, w1),
            "hf_recv_latency": hf_latency(w0, w1),
            "as_of_utc": _iso(dt.datetime.now(dt.timezone.utc).timestamp())}


# --------------------------------------------------------------------------
def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            print(f"FAIL: {label}")
            raise SystemExit(1)

    base = 1787000000 - (1787000000 % 86400)
    def hb(tag, sod_list, vals, dated=False):
        out = []
        for sod, v in zip(sod_list, vals):
            t = dt.datetime.fromtimestamp(base + sod, dt.timezone.utc)
            stamp = (t.strftime("%Y-%m-%dT%H:%M:%SZ") if dated
                     else t.strftime("%H:%M:%SZ"))
            out.append(f"[{tag}] {stamp} bbo={v} l2Book=0 trades=0")
        return "\n".join(out)

    spec = VENUES["hyperliquid"]
    txt = hb("hl", [0, 60, 120, 180], [0, 600, 1200, 1800])
    got = dated_heartbeats(txt, spec["re"], spec["counters"], spec["sentinel"],
                           base + 200)
    ok(len(got) == 4 and got[0][0] == base and got[-1][0] == base + 180,
       "POSITIVE CONTROL: four dateless heartbeats date correctly by the "
       "backward walk")
    ok([round((got[i][1] - got[i-1][1]) / (got[i][0] - got[i-1][0]), 3)
        for i in range(1, 4)] == [10.0, 10.0, 10.0],
       "POSITIVE CONTROL: cumulative counters difference to a constant rate")

    # KNOWN-BAD: a format change must REFUSE, never read as an absent venue.
    try:
        dated_heartbeats("[hl] subscribed 16 coins\n[hl] 12:00:00Z bbo_v2=5",
                         spec["re"], spec["counters"], spec["sentinel"],
                         base)
        ok(False, "a format change must REFUSE")
    except Unmeasured as e:
        ok("FORMAT CHANGE" in str(e),
           "KNOWN-BAD: prefix lines with no matching heartbeat REFUSE as a "
           "FORMAT CHANGE -- the alibi shape, where a renamed field would let "
           "a venue read as 'no data' and then as 'not affected'")
    ok(dated_heartbeats("nothing here", spec["re"], spec["counters"],
                        spec["sentinel"], base) == [],
       "and a log with NO lines of this venue at all returns empty rather "
       "than refusing -- absence of the venue differs from a format change")

    # KNOWN-BAD: non-monotonic reconstruction must refuse.
    try:
        dated_heartbeats(hb("hl", [0, 0], [0, 1]), spec["re"],
                         spec["counters"], spec["sentinel"], base + 10)
        ok(False, "duplicate stamps must REFUSE")
    except Unmeasured as e:
        ok("monotonically" in str(e),
           "KNOWN-BAD: two stamps resolving to one instant REFUSE")

    # A RESTART IS NOT A BLACKOUT.
    class _S:
        pass
    pts = [(base, 0), (base + 60, 600), (base + 120, 5)]   # counter reset
    ser = []
    for i in range(1, len(pts)):
        d_t = pts[i][0] - pts[i-1][0]
        d_v = pts[i][1] - pts[i-1][1]
        if d_t > 0 and d_v >= 0:
            ser.append((pts[i][0], d_v / d_t))
    ok(len(ser) == 1,
       "a counter RESET is dropped as a status, not counted as zero traffic "
       "-- counting a restart as a blackout would manufacture the signature "
       "under investigation")

    # THE DECLARED OUTCOME TABLE, driven in every branch (rule 16).
    M = lambda f: {"status": "MEASURED", "thin_fraction": f}
    ok(verdict(M(0.9), M(0.0), M(0.0))["verdict"] == "H1-POLYMARKET-SIDE",
       "outcome table: PM alone thin -> H1")
    ok(verdict(M(0.9), M(0.9), M(0.9))["verdict"] == "H2-HOST-OR-PATH",
       "outcome table: all three thin -> H2")
    ok(verdict(M(0.9), M(0.9), M(0.0))["verdict"] == "UNRESOLVED-ASYMMETRIC",
       "outcome table: PM + exactly one other -> UNRESOLVED-ASYMMETRIC")
    ok(verdict(M(0.1), M(0.0), M(0.0))["verdict"] == "UNRESOLVED-OTHER",
       "outcome table: nothing thin -> UNRESOLVED-OTHER, never H1 by default")
    ok(verdict({"status": "UNMEASURED"}, M(0.0), M(0.0))["verdict"]
       == "UNRESOLVED-UNMEASURED",
       "outcome table: an UNMEASURED venue is NOT an alibi -- it cannot "
       "contribute a 'normal' reading to an H1 verdict")

    # thin_fraction refuses a day it cannot see, and admits one it can.
    ser = [(base + i * 60, 100.0) for i in range(120)]
    day = dt.datetime.fromtimestamp(base, dt.timezone.utc).strftime("%Y%m%d")
    r = thin_fraction(ser, day, base, base + 3600)
    ok(r["status"] == "MEASURED" and r["thin_fraction"] == 0.0,
       "POSITIVE CONTROL: a flat healthy series reads 0.0 thin")
    ser2 = [(base + i * 60, 100.0 if i < 60 else 1.0) for i in range(120)]
    r2 = thin_fraction(ser2, day, base + 3600, base + 7200)
    ok(r2["status"] == "MEASURED" and r2["thin_fraction"] == 1.0,
       "POSITIVE CONTROL: a series that collapses to 1% of median reads 1.0 "
       "thin -- the detector fires")
    ok(thin_fraction(ser[:10], day, base, base + 3600)["status"]
       == "UNMEASURED",
       "KNOWN-BAD: too few intervals to have a day median is UNMEASURED, "
       "never 'normal'")
    ok(thin_fraction(ser, day, base + 200000, base + 203600)["status"]
       == "UNMEASURED",
       "KNOWN-BAD: a window the log does not reach is UNMEASURED")

    # KNOWN-BAD for the window-derivation defect above: a LATER coin holding
    # the LONGER run must win. Under the count-vs-seconds comparison the
    # alphabetically first coin always won, so this fixture is the falsifier.
    import tempfile as _tf, gzip as _gz
    with _tf.TemporaryDirectory() as _td:
        _root = Path(_td) / "raw"
        _day = "20260910"
        (_root / _day).mkdir(parents=True)
        _b = day_bounds(_day)[0]
        for coin, thin_at, thin_len in (("aaa", 10, 3), ("zzz", 100, 30)):
            for i in range(288):
                n = 3 if thin_at <= i < thin_at + thin_len else 5000
                with _gz.open(_root / _day /
                              f"{coin}-updown-5m-{_b + i * WINDOW_S}.jsonl.gz",
                              "wb") as fh:
                    fh.write(b'{"x":1}\n' * n)
        if True:
            w = derive_window(_day, raw_root=_root, gaps={})
            ok(w["n_windows"] == 30
               and w["start_utc"] == _iso(_b + 100 * WINDOW_S),
               "KNOWN-BAD (window derivation): with a 3-window run on the "
               "alphabetically FIRST coin and a 30-window run on the LAST, "
               "the derived window is the 30-window one. The original "
               "comparison put a window COUNT against a DURATION IN SECONDS, "
               "so the first coin always won and a 40-window event was "
               "reported as 14")
            ok(len(w["per_coin_longest_run"]) == 2,
               "and both coins' longest runs are reported beside it, so the "
               "choice is visible rather than implicit")

    # ---- RR7-1: the SHIPPED regex of EVERY venue, on a REAL line --------
    # The fixtures above are all built with the `hyperliquid` spec, so the
    # binance_hf and polymarket regexes were never matched against a real
    # line -- a mutation to either left the suite green. The protection was
    # in the product (the verdict refuses) and not in the suite; this is the
    # suite half. Each venue must (a) match at least one real line from its
    # OWN log and (b) yield the declared number of counters.
    import re as _re
    for _v, _spec in VENUES.items():
        _hit = None
        for _p in _spec["logs"]:
            if not Path(_p).exists():
                continue
            for _ln in Path(_p).read_text(errors="replace").splitlines():
                _m = _spec["re"].match(_ln)
                if _m:
                    _hit = (_ln, _m)
                    break
            if _hit:
                break
        ok(_hit is not None,
           f"RR7-1 ({_v}): the SHIPPED regex matches a REAL line from that "
           f"venue's OWN log -- otherwise a broken parser reads as an absent "
           f"venue and then as an unaffected one")
        _ln, _m = _hit
        _g = _m.groups()
        ok(len([x for x in _g[6:6 + _spec["counters"]] if x is not None])
           == _spec["counters"],
           f"RR7-1 ({_v}): that real line yields all {_spec['counters']} "
           f"declared counter group(s), so the spec's `counters` matches the "
           f"line it actually parses")
    # AND THE CONTROL IN THE OTHER DIRECTION: a venue's regex must NOT match
    # another venue's line, or 'matches a real line' proves nothing.
    _pm_line = "[pm] 12:00:00Z markets=5 msgs=100"
    ok(VENUES["polymarket"]["re"].match(_pm_line) is not None
       and VENUES["binance_hf"]["re"].match(_pm_line) is None
       and VENUES["hyperliquid"]["re"].match(_pm_line) is None,
       "RR7-1 DISCRIMINATION: each venue's regex matches its own shape and "
       "REJECTS another venue's -- three parsers that all matched everything "
       "would pass the check above and discriminate nothing")

    print(f"da_cross_venue_forensics selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--events", action="store_true")
    ap.add_argument("--day")
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    days = ["20260826", "20260831", "20260902"] if a.events else [a.day]
    res = []
    for d in days:
        try:
            res.append(analyse(d, label=f"event {d}"))
        except Unmeasured as e:
            res.append({"label": f"event {d}", "status": "UNMEASURED",
                        "why": str(e)})
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
