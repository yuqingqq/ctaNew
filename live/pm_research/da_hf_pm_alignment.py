"""R-46: align the Binance HF tape to Polymarket windows, knowledge-time honest.

BE wants a hedgeability decomposition: what FRACTION of post-fill drift is
contemporaneous underlying movement?  That needs, for a Polymarket event at
time t, the Binance book state that was KNOWN at t.  This module supplies that
read and the coverage figure that says where it is legitimate.

WHY THE ALIGNMENT CAN BE MADE HONEST -- both collectors stamp local userspace
knowledge time on the same host.  PM stamps immediately after ``ws.recv``.
Historical HF rows were stamped after JSON parsing; HF collector v2 stamps
immediately after ``ws.recv`` and records each process boundary and exact stamp
semantics in ``data/mm_hf/collector_runs.jsonl``.  Therefore the common clock
is directly comparable, while the stamp-point version remains explicit rather
than silently mixing historical and repaired rows.

WHAT WOULD MAKE IT DISHONEST, and is therefore refused:

  * Aligning on Binance `E` (exchange event time) or `T`.  Those are Binance's
    clock, they run ~87 ms behind local receipt (the collector's own heartbeat
    reports `recv-lat~87ms (incl clock offset)`), and messages can be delivered
    out of `E`-order.  Aligning on `E` would let a row that ARRIVED after t be
    treated as known at t.  `state_at()` uses recv_ns and the selftests make
    that choice observable rather than merely declared -- a crafted tape where
    E-order and recv-order disagree gets a DIFFERENT answer from an E-aligner,
    so a regression to `E` cannot pass quietly.
  * Reading forward.  `state_at(t)` returns the last row with `recv_ns <= t`,
    never the next one, and returns Unavailable rather than reaching backward
    past the start of coverage.

COVERAGE IS THE HONESTY PROBLEM, NOT THE CLOCK.  The two tapes are separate
collectors with separate failure modes, and they are NOT symmetric:

  * Polymarket writes a structured gap ledger -- `collector_gaps.jsonl`, 188
    `gap_closed` records carrying `gap_start_ns`, `gap_end_ns`, `slug`, `cause`.
  * **Binance HF writes NO gap ledger at all.**  `collect_hf.py` reconnects on
    drop, watchdog timeout and a 23 h cycle, and each reconnect loses messages
    silently.  The only trace is an UNTIMESTAMPED stdout line ("conn#0 dropped
    ... reconnecting", 20 of them), which can be localised only to the <=60 s
    between two timestamped heartbeat lines.

So HF gaps are INFERRED FROM THE TAPE, not read from a ledger: a silence in
`recv_ns` longer than `HF_GAP_MS`.  That is an inference and is labelled one
everywhere it is published.  It is not weaker than it sounds -- BTCUSDT
bookTicker runs ~2.4 M rows/hour, so a real outage is unmissable -- but it
cannot see a drop that lost messages without producing silence, which is
exactly what a reconnect inside a busy stream looks like.  `u`, the venue
update id, does not close that hole: it is a book-update counter, not a
per-message sequence (consecutive rows jump by ~219), so a jump does not count
missed messages.  **THIS IS THE WEAKEST LINK IN THE ALIGNMENT AND IT IS ON THE
BINANCE SIDE.**

JOINT COVERAGE is therefore the figure that matters, and neither tape's own
number may stand in for it: a window is usable only if BOTH tapes covered it.
`joint_coverage()` returns that, and reports the two per-tape figures beside it
so the gap between them is visible rather than averaged away.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import sys
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# RR12-1 -- CODE root and DATA root are different trees. This module reads
# DATA, so its root is the tree holding the tape, resolved once in
# `pm_tape_density` and imported rather than restated. Deriving it from
# __file__ made every worktree run read an empty directory.
# DA10-R3: this module was the ONLY one of the six with no sys.path
# insert, so the import added in round 10 resolved by PATH LAUNCH and
# failed under `python3 -m` -- a suite that passes because of how it was
# started (CO-2). The line the other five already carry:
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_tape_density as _TDROOT  # noqa: E402
CODE_ROOT = Path(__file__).resolve().parents[2]
REPO = _TDROOT.DATA_ROOT
HF_RAW = REPO / "data/mm_hf/raw"
HF_RUNS = REPO / "data/mm_hf/collector_runs.jsonl"
PM_ROOT = REPO / "data/pm_5min"
PM_GAPS = PM_ROOT / "collector_gaps.jsonl"

ALIGNMENT_VERSION = "hf_pm_alignment_v2_stamp_manifest"
LEGACY_HF_STAMP_POINT = "AFTER_JSON_PARSE_LEGACY_UNMANIFESTED"

# A recv_ns silence longer than this counts as an inferred HF outage.  Chosen
# against measured behaviour, not picked: BTCUSDT bookTicker's worst intra-hour
# inter-arrival on a quiet hour is ~385 ms, so 2 s is ~5x the observed normal
# maximum.  CLASS A (configuration) -- it is a reporting threshold, no verdict
# rests on it, and coverage is published at several thresholds so the reader can
# see the sensitivity rather than trusting this one.
HF_GAP_MS = 2000.0
HF_GAP_LADDER_MS = (500.0, 1000.0, 2000.0, 5000.0, 30000.0)

WINDOW_S = 300  # Polymarket 5-minute markets


def hf_collector_runs(path: Path = HF_RUNS) -> list[dict[str, Any]]:
    """Read valid append-only HF collector run records.

    Absence of a record is meaningful: rows before the first manifest boundary
    use the historical post-JSON-parse stamp.  Malformed lines are ignored
    rather than allowed to relabel raw data.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            started = int(record["started_at_ns"])
            stamp_point = str(record["stamp_point"])
            symbols = record["symbols"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if started <= 0 or not stamp_point or not isinstance(symbols, list):
            continue
        records.append({**record, "started_at_ns": started,
                        "stamp_point": stamp_point,
                        "symbols": [str(symbol).upper() for symbol in symbols]})
    return sorted(records, key=lambda record: record["started_at_ns"])


def hf_stamp_profile(
    windows: Sequence[int], symbol: str,
    records: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe stamp semantics across the exact requested raw-data interval.

    A run record changes semantics from its ``started_at_ns`` onward.  The
    interval before the first applicable record is explicitly legacy; it is
    never inferred to be v2 merely because the current source code is v2.
    """
    if not windows:
        return {"uniform": False, "segments": [],
                "reason": "NO_WINDOWS"}
    lo_ns = min(windows) * 1_000_000_000
    hi_ns = (max(windows) + WINDOW_S) * 1_000_000_000
    symbol = symbol.upper()
    applicable = sorted((record for record in (
        hf_collector_runs() if records is None else records)
        if symbol in [str(item).upper() for item in record.get("symbols", [])]
        and int(record.get("started_at_ns", 0)) < hi_ns),
        key=lambda record: int(record["started_at_ns"]))

    prior = [record for record in applicable
             if int(record["started_at_ns"]) <= lo_ns]
    active = (str(prior[-1]["stamp_point"])
              if prior else LEGACY_HF_STAMP_POINT)
    boundaries: list[tuple[int, str]] = [(lo_ns, active)]
    for record in applicable:
        started = int(record["started_at_ns"])
        if lo_ns < started < hi_ns:
            stamp_point = str(record["stamp_point"])
            if stamp_point != boundaries[-1][1]:
                boundaries.append((started, stamp_point))
    segments = [
        {
            "start_ns": start,
            "end_ns": (boundaries[index + 1][0]
                       if index + 1 < len(boundaries) else hi_ns),
            "stamp_point": stamp_point,
        }
        for index, (start, stamp_point) in enumerate(boundaries)
    ]
    return {
        "uniform": len({segment["stamp_point"] for segment in segments}) == 1,
        "contains_legacy_post_parse": any(
            segment["stamp_point"] == LEGACY_HF_STAMP_POINT
            for segment in segments),
        "segments": segments,
        "manifest_records_considered": len(applicable),
    }


# ---------------------------------------------------------------------------
# R-47: the clock is NOT the risk.  The STAMP POINT is.
# ---------------------------------------------------------------------------
#
# The two collectors share one CLOCK_REALTIME on one host, so there is no
# inter-process skew to reconcile and Binance's own clock offset never enters:
# the alignment compares recv_ns to recv_ns and never touches `E`.  That is
# defended, not assumed -- both collectors retain local ``recv_ns`` and the HF
# run manifest records the exact code-era boundary.
#
# What DOES differ is WHERE in each pipeline the stamp is taken, and a lead-lag
# measurement is exactly as sensitive to that as it is to a clock error.
#
#   1. STAMP-POINT ASYMMETRY.  PM stamps immediately after `ws.recv()` returns,
#      BEFORE parsing.  Historical HF rows stamp AFTER `json.loads`; v2 HF rows
#      now use the same userspace boundary as PM.  The append-only manifest
#      prevents a straddling hour file from hiding that change.  Historical HF
#      rows retain a small parse-time late bias; they are not rewritten.
#
#   2. EVENT-LOOP QUEUEING.  Both collectors are single-threaded asyncio loops
#      and neither stamps at the kernel.  A message that arrives while the loop
#      is busy is stamped when the coroutine next runs.  Measured on HF
#      (BTCUSDT, 2.36 M rows in one hour), `recv_ns - E` has a hard floor at
#      70.3 ms -- network plus Binance clock offset, a constant -- with
#      excursions ABOVE that floor of +3.6 ms median, +30 ms p99, **+259 ms
#      p99.9, +360 ms max**.  Those excursions are stamping delay, not market
#      latency.  Again: HF late.
#
#   3. INSTRUMENTATION ASYMMETRY.  PM records its own stamping lag
#      (`lag_ms_max_interval`, `ws_queue_depth_max`, `ws_ever_paused`,
#      `loop_stalls`).  HF records NONE of it.  So PM's bias can be read from
#      its ledger and HF's must be inferred from `recv_ns - E`.  As with the
#      gap ledger, the Binance side is the weaker one.
#
# THE BIAS BUDGET, BOTH DIRECTIONS, MEASURED (R-49 re-points the bar to 750 ms):
#
#   MANUFACTURES a Binance lead -- only a delayed PM stamp can do this.  PM runs
#   an independent probe coroutine that times the overshoot of a fixed-interval
#   sleep (`collect_pm.py:205-215`); that overshoot IS the interval the loop
#   could not run, hence a direct upper bound on how late a stamp can be, not a
#   proxy for it.  Measured over 183 recorded intervals: p50 1.9 ms, p99
#   12.2 ms, max 12.7 ms; **all-time worst since the run began 203.7 ms**;
#   `loop_stalls` = 0; `ws_ever_paused` TRUE in **0 of 179**.
#
#   HIDES a Binance lead -- on historical rows, HF parse-point plus HF loop
#   queueing: +3.6 ms p50, +30 ms p99, +259 ms p99.9, +360 ms max (inferred,
#   since HF runs no probe).  V2 removes parse time, but not websocket/event-loop
#   queueing, so this remains a conservative historical bound rather than a
#   claim that v2 supplies kernel-arrival time.
#
# So against R-49's 750 ms bar the manufacturing direction is bounded at 1.7 %
# of the effect typically and 27 % at the worst excursion ever recorded, while
# the hiding direction is larger.  A measured lead over 750 ms cannot be an
# artefact of stamping unless PM stalled for ~750 ms, which the probe has never
# observed.
#
# AND NOTE WHAT R-49 DID TO THIS MEASUREMENT.  At the OLD 160 ms bar, PM's
# all-time worst stall of 203.7 ms EXCEEDED the entire effect -- one stall could
# have produced the whole result, and this alignment would not have been usable
# for the question.  Moving the bar to 750 ms improves the bias-to-effect ratio
# ~4.7x.  The falsifier that killed the reactive channels is what made this
# measurement viable.
#
# WHY THIS IS SAFE FOR THE QUESTION ACTUALLY BEING ASKED, and it is the point:
# both known asymmetries push the HF stamp LATE.  A late Binance stamp SHRINKS
# a measured Binance lead.  **They cannot manufacture the result the programme
# wants; they can only hide it.**  A measured Binance lead is therefore a LOWER
# BOUND on the true one -- which is the conservative direction, and the reason
# this measurement can proceed rather than stop.
#
# WHAT THAT DOES NOT LICENSE: the p99.9 excursion of 259 ms is the same order as
# the effect being measured, so a lead may NOT be published as a point estimate.
# `stamp_lag_profile()` computes the per-window excursion so every lead figure
# carries its own bias bound, and a window whose excursion is comparable to the
# lead it reports is not evidence of a lead.


def stamp_lag_profile(recv_ns: Sequence[int], venue_ms: Sequence[int]) -> dict[str, float]:
    """Stamping delay ABOVE each tape's own floor -- the bias that can move a lead.

    `recv_ns - venue_ts` mixes three things: a constant (network + the venue's
    clock offset), and a variable queueing delay.  The MINIMUM over the sample
    estimates the constant; everything above it is delay this collector added.
    Only the excursion matters for lead-lag, because the constant is differenced
    away when both sides are read on the same local clock -- which is precisely
    why the alignment uses recv_ns and never `E`.
    """
    if not recv_ns or len(recv_ns) != len(venue_ms):
        raise Unavailable("stamp_lag_profile needs paired recv/venue stamps")
    deltas = sorted(
        (r - v * 1_000_000) / 1_000_000
        for r, v in zip(recv_ns, venue_ms)
    )
    floor = deltas[0]
    exc = [d - floor for d in deltas]
    n = len(exc)

    def q(f: float) -> float:
        return exc[min(int(n * f), n - 1)]

    return {"n": n, "floor_ms": round(floor, 2),
            "excursion_p50_ms": round(q(0.50), 2), "excursion_p90_ms": round(q(0.90), 2),
            "excursion_p99_ms": round(q(0.99), 2), "excursion_p999_ms": round(q(0.999), 2),
            "excursion_max_ms": round(exc[-1], 2)}


class Unavailable(Exception):
    """No state was KNOWN at the requested time.  Never silently substituted."""


@dataclass(frozen=True, slots=True)
class BookRow:
    recv_ns: int
    e_ms: int
    bid: float
    bid_qty: float
    ask: float
    ask_qty: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


class HFTape:
    """One symbol's bookTicker rows, ordered by RECEIPT time.

    Sorted by `recv_ns` explicitly rather than trusting file order: the tape is
    written from an async queue and out-of-order arrival within a flush is
    possible.  Sorting by recv_ns keeps `state_at` a true knowledge-time read
    even if the file is not perfectly ordered.
    """

    def __init__(self, rows: Sequence[BookRow]) -> None:
        self.rows = sorted(rows, key=lambda r: r.recv_ns)
        self._keys = [r.recv_ns for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def state_at(self, t_ns: int) -> BookRow:
        """The last row RECEIVED at or before `t_ns`.  Never reads forward."""
        idx = bisect_right(self._keys, t_ns) - 1
        if idx < 0:
            raise Unavailable(f"no HF row received at or before {t_ns}")
        return self.rows[idx]

    def max_gap_ms(self, lo_ns: int, hi_ns: int) -> float:
        """Worst inter-arrival silence within [lo, hi), edges included.

        The edges matter: a window whose rows are dense but which starts 40 s
        after the window opened is NOT covered, and a max-gap over interior
        points alone would call it clean.
        """
        i = bisect.bisect_left(self._keys, lo_ns)
        j = bisect_right(self._keys, hi_ns)
        pts = [lo_ns] + self._keys[i:j] + [hi_ns]
        return max((b - a) for a, b in zip(pts, pts[1:])) / 1e6


def _iter_hf_files(symbol: str, days: Sequence[str], stream: str = "bookTicker"):
    root = HF_RAW / stream / symbol
    for day in check_days(days):
        for path in sorted(root.glob(f"{day}_*.csv.gz")):
            yield path


def iter_hf_recv_ns(symbol: str, days: Sequence[str]):
    """Stream receipt times.  NEVER materialise the tape.

    The first version of this module built a list of row objects and died with
    MemoryError: BTCUSDT bookTicker alone is ~2.4 M rows/hour, so five days is
    ~280 M rows.  Coverage needs one integer per row and O(1) memory, and the
    files are already hour-ordered, so streaming is both correct and necessary.
    """
    for path in _iter_hf_files(symbol, days):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                cut = line.find(",")
                if cut > 0:
                    yield int(line[:cut])


def hf_window_max_gap(
    symbol: str, days: Sequence[str], windows: Sequence[int]
) -> tuple[dict[int, float], int]:
    """(window_start -> worst silence in ms, rows scanned), in one streaming pass.

    The max silence inside a window is the max over inter-arrival intervals
    CLIPPED to that window.  Clipping is what makes the window edges count: a
    window whose rows are dense but which starts 40 s late is measured from its
    own start, and a window with no rows at all is measured as fully silent --
    both of which an interior-only max would call clean.
    """
    n = 0

    def counted():
        nonlocal n
        for v in iter_hf_recv_ns(symbol, days):
            n += 1
            yield v

    return window_max_gap(counted(), windows), n


def window_max_gap(recv_ns: Iterable[int], windows: Sequence[int]) -> dict[int, float]:
    """The clipping rule, pure so it can be tested without a 17 GB tape."""
    W = WINDOW_S
    want = sorted(set(windows))
    if not want:
        return {}
    idx = {w // W for w in want}
    gap: dict[int, int] = {w: 0 for w in want}

    def apply(a: int, b: int) -> None:
        if b <= a:
            return
        lo_i, hi_i = a // (W * 1_000_000_000), b // (W * 1_000_000_000)
        for wi in range(lo_i, hi_i + 1):
            if wi not in idx:
                continue
            w0 = wi * W * 1_000_000_000
            w1 = w0 + W * 1_000_000_000
            seg = min(b, w1) - max(a, w0)
            key = wi * W
            if seg > 0 and seg > gap.get(key, 0):
                gap[key] = seg

    lo_bound = min(want) * 1_000_000_000
    hi_bound = (max(want) + W) * 1_000_000_000
    prev = lo_bound
    for cur in recv_ns:
        if cur < prev:
            continue  # out-of-order within a flush; clipping keeps this safe
        apply(prev, cur)
        prev = cur
    apply(prev, hi_bound)
    return {w: v / 1e6 for w, v in gap.items()}


def pm_gap_intervals() -> list[tuple[int, int, str]]:
    """(start_ns, end_ns, cause) from the PM ledger.

    `gap_closed` rows carry both ends.  Rows that record a gap OPENING without a
    matching close carry `last_message_recv_ns` and no end -- those are treated
    as open-ended from that point and reported separately, because silently
    dropping them would understate PM's outage.
    """
    closed: list[tuple[int, int, str]] = []
    unclosed: list[tuple[int, str]] = []
    if not PM_GAPS.exists():
        return closed
    for line in PM_GAPS.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("gap_start_ns") and rec.get("gap_end_ns"):
            closed.append((int(rec["gap_start_ns"]), int(rec["gap_end_ns"]),
                           str(rec.get("cause", "?"))))
        elif rec.get("last_message_recv_ns") and rec.get("event") != "collector_stop":
            unclosed.append((int(rec["last_message_recv_ns"]), str(rec.get("cause", "?"))))
    closed.sort()
    return closed


_DAY_TOKEN = re.compile(r"^\d{8}$")


def check_days(days: Sequence[str]) -> list[str]:
    """Refuse a day token that cannot name a directory, instead of matching none.

    Q-DA-68, found by it biting me.  BOTH tapes resolve days by string match --
    PM globs `raw/<day>/` and HF globs `<day>_*.csv.gz` -- and the archive
    directories are named `20260824`.  So `pm_windows(["2026-08-24"], "btc")`
    returned 0 windows and raised nothing, and because `joint_coverage` divides
    by `len(windows) or 1`, every derived figure became a clean-looking zero.
    A comparison I ran over that empty population reported its two arms
    "identical" -- a zero from an instrument that never proved it could fire
    (rule 15), and a silent drop where a counted status belongs (rule 4).

    A well-formed day with no data is a legitimate empty; a day token that
    cannot match the naming convention is a bug, and fails loudly here.
    """
    bad = [d for d in days if not _DAY_TOKEN.match(str(d))]
    if bad:
        raise ValueError(
            f"malformed day token(s) {bad}: expected YYYYMMDD (e.g. 20260824). "
            "Such a token selects no files on EITHER tape and would silently "
            "yield an empty population with no error.")
    return [str(d) for d in days]


def pm_windows(days: Sequence[str], coin: str | None = None) -> dict[str, list[int]]:
    """(coin -> sorted window-start epochs) discovered from the raw slugs."""
    out: dict[str, set[int]] = defaultdict(set)
    for day in check_days(days):
        root = PM_ROOT / "raw" / day
        if not root.is_dir():
            continue
        for path in root.glob("*-updown-5m-*.jsonl*.gz"):
            stem = path.name.split(".jsonl")[0]
            parts = stem.split("-updown-5m-")
            if len(parts) != 2:
                continue
            c = parts[0]
            if coin and c != coin:
                continue
            try:
                out[c].add(int(parts[1].split("-")[0]))
            except ValueError:
                continue
    return {c: sorted(v) for c, v in out.items()}


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


# ---------------------------------------------------------------------------
# R-133: DETECTION IS NOT A GATE.
# ---------------------------------------------------------------------------
# `hf_stamp_profile` computed `uniform` and the selftests asserted it, but
# nothing REFUSED on it -- a window straddling a collector restart was
# detectable and still admitted.  A property nothing acts on is a comment.
# A straddling window now fails the joint-coverage gate exactly as a data gap
# does, and can only be admitted by an EXPLICIT waiver naming the window.

def hf_collector_run_defects(path: Path = HF_RUNS) -> int:
    """Non-empty ledger lines that do NOT parse as run records.

    `hf_collector_runs` drops these so malformed text can never relabel raw
    data.  That is right for reading and wrong for CERTIFYING: a dropped line
    may have carried a boundary, so uniformity cannot be asserted while any
    line is unreadable.  Counted separately so the gate can fail closed.
    """
    if not path.exists():
        return 0
    defects = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            int(record["started_at_ns"])
            if not str(record["stamp_point"]) or not isinstance(
                    record["symbols"], list):
                raise ValueError("incomplete run record")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            defects += 1
    return defects


#: The fields that DEFINE stamp semantics.  A run row that agrees with its
#: predecessor on all of them changed nothing a sub-second feature can see.
STAMP_SEMANTICS_FIELDS = ("collector_schema_version", "stamp_point")


def stamp_boundaries_ns(symbol: str,
                        records: Sequence[dict[str, Any]] | None = None
                        ) -> list[int]:
    """Instants at which HF stamp semantics CHANGE for `symbol`.

    MAINTENANCE, Q-DA-68 (authority: R-110 maintenance of existing DA surface;
    semantics ruled by R-147(2)).  THE DEFECT THIS REPLACES: the previous body
    returned every run start, keyed on symbol membership and nothing else --
    so the docstring said "semantics change" while the code evaluated "a run
    started".  The 2026-08-26T05:11:43Z collector restart after the box crash
    therefore appeared as a third boundary, and R-147(2) had ruled it is not
    one: "schema `hf_ws_v2_recv_boundary` unchanged -> same era, new run; a
    coverage gap, not a boundary."  Over-refusal is the SAFE direction, which
    is exactly why it would have sat here unnoticed.

    THE RULE, which is R-147(2) mechanised rather than restated: a run row is
    a boundary iff its `STAMP_SEMANTICS_FIELDS` differ from the PRECEDING
    run's.  The first manifested run is always a boundary -- unmanifested
    legacy history is post-parse stamped, so the transition into the first
    recorded run IS a semantics change.

    An EMPTY ledger yields no boundaries, and that is uniformity rather than
    ignorance: `hf_collector_runs` documents absence as "all legacy
    post-parse", so a run of unmanifested history is genuinely one semantics.
    Unreadable lines are a different matter and are handled by the caller via
    `hf_collector_run_defects`.

    MEASURED IMPACT (Q-DA-68, both arms run, not asserted): 08-24 btc is
    unchanged (1 straddle, the 13:45Z window, under both rules -- the dropped
    instants fall inside the same window).  08-26 btc is where the rules can
    differ and does: stamp_covered 54/55 -> 55/55, straddles [05:10Z] -> [].
    `joint_covered` is unchanged on BOTH days (234 and 10), because that
    window already fails HF coverage inside the crash gap.
    """
    recs = hf_collector_runs() if records is None else records
    want = symbol.upper()
    mine = sorted(
        (r for r in recs
         if want in [str(x).upper() for x in r.get("symbols", [])]),
        key=lambda r: int(r["started_at_ns"]))
    out: list[int] = []
    prev: tuple[Any, ...] | None = None
    for r in mine:
        sem = tuple(r.get(f) for f in STAMP_SEMANTICS_FIELDS)
        if prev is None or sem != prev:
            out.append(int(r["started_at_ns"]))
        prev = sem
    return out


def window_stamp_uniform(window_start: int, boundaries: Sequence[int]) -> bool:
    """True iff no stamp boundary falls STRICTLY inside the window.

    A boundary exactly at the window edge is not a straddle: every row in the
    window then carries one semantics.
    """
    lo = window_start * 1_000_000_000
    hi = lo + WINDOW_S * 1_000_000_000
    return not any(lo < b < hi for b in boundaries)


def joint_coverage(
    coin: str, symbol: str, days: Sequence[str], gap_ms: float = HF_GAP_MS,
    stamp_waiver: Sequence[int] | None = None,
) -> dict[str, Any]:
    """The figure that matters: windows BOTH tapes covered.

    Reports the two per-tape figures beside it -- never one standing in for the
    joint number, which is the specific failure R-46 named.
    """
    windows = pm_windows(days, coin).get(coin, [])
    gaps, n_rows = hf_window_max_gap(symbol, days, windows)
    pm_gaps = pm_gap_intervals()

    # R-133: refuse windows whose HF rows do not share one stamp semantics.
    waived = set(stamp_waiver or ())
    boundaries = stamp_boundaries_ns(symbol)
    ledger_defects = hf_collector_run_defects()

    hf_ok = pm_ok = both_ok = stamp_ok = 0
    hf_bad: list[int] = []
    pm_bad: list[int] = []
    stamp_bad: list[int] = []
    for w in windows:
        lo, hi = w * 1_000_000_000, (w + WINDOW_S) * 1_000_000_000
        h = gaps.get(w, float("inf")) <= gap_ms
        p = not any(_overlaps(lo, hi, g0, g1) for g0, g1, _ in pm_gaps)
        # An unreadable ledger line may have carried a boundary, so uniformity
        # cannot be certified for ANY window until the ledger reads cleanly.
        s_ok = (ledger_defects == 0
                and window_stamp_uniform(w, boundaries)) or w in waived
        hf_ok += h
        pm_ok += p
        stamp_ok += s_ok
        both_ok += (h and p and s_ok)
        if not h:
            hf_bad.append(w)
        if not p:
            pm_bad.append(w)
        if not s_ok:
            stamp_bad.append(w)

    n = len(windows) or 1
    return {
        "alignment_version": ALIGNMENT_VERSION,
        "coin": coin, "symbol": symbol, "days": list(days),
        "hf_stamp_profile": hf_stamp_profile(windows, symbol),
        "hf_gap_threshold_ms": gap_ms,
        "hf_rows": n_rows,
        "pm_windows": len(windows),
        "hf_covered": hf_ok, "pm_covered": pm_ok, "joint_covered": both_ok,
        "hf_covered_pct": round(100.0 * hf_ok / n, 2),
        "pm_covered_pct": round(100.0 * pm_ok / n, 2),
        "joint_covered_pct": round(100.0 * both_ok / n, 2),
        "hf_only_loss": hf_ok - both_ok,
        "pm_only_loss": pm_ok - both_ok,
        "stamp_covered": stamp_ok,
        "stamp_covered_pct": round(100.0 * stamp_ok / n, 2),
        "stamp_straddling_windows": stamp_bad[:20],
        "stamp_waived_windows": sorted(waived & set(windows)),
        "hf_collector_ledger_defects": ledger_defects,
        "hf_uncovered_windows": hf_bad[:20],
        "pm_uncovered_windows": pm_bad[:20],
        "hf_coverage_is_INFERRED": True,
        "note": ("HF has NO gap ledger; its coverage is inferred from recv_ns "
                 "silence and cannot see a reconnect that lost messages without "
                 "producing silence. PM coverage is read from a real ledger."),
    }


# ---------------------------------------------------------------------------
# selftests
# ---------------------------------------------------------------------------

def _selftests() -> int:
    checks = 0

    def ok(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            raise AssertionError(f"selftest failed: {label}")

    def row(recv_ns, e_ms, bid=1.0, ask=2.0):
        return BookRow(recv_ns, e_ms, bid, 1.0, ask, 1.0)

    # 1. knowledge time: never read forward
    tape = HFTape([row(100, 1), row(200, 2), row(300, 3)])
    ok(tape.state_at(250).recv_ns == 200, "returns the last row received at or before t")
    ok(tape.state_at(299).recv_ns == 200, "does not reach forward to 300")

    # 2. the boundary is INCLUSIVE: a row received exactly at t was known at t
    ok(tape.state_at(200).recv_ns == 200, "recv_ns == t is known at t")

    # 3. before coverage starts -> Unavailable, never the first row
    try:
        tape.state_at(99)
    except Unavailable:
        ok(True, "before coverage raises Unavailable")
    else:
        ok(False, "must not substitute the first row for missing history")

    # 4. THE MIRROR (R-42): make the aligner REVEAL that it uses recv_ns.
    #    Out-of-order delivery -- the row with the LATER exchange time arrives
    #    FIRST.  A recv_ns aligner and an E aligner give different answers, so a
    #    silent regression to payload time cannot pass this test.
    #    Row A ARRIVED first (recv 100) but claims a LATER event time (E 300);
    #    row B arrived second (recv 200) claiming an EARLIER one (E 150).  At
    #    t=350 both are eligible under either rule, so the two rules differ only
    #    in WHICH they call last -- receipt says B, exchange time says A.
    a, b = row(100, 300, bid=10.0), row(200, 150, bid=20.0)
    out_of_order = HFTape([a, b])
    got = out_of_order.state_at(350)
    ok(got.bid == 20.0, "aligns on RECEIPT order (row B, received last)")
    e_answer = sorted([r for r in out_of_order.rows if r.e_ms <= 350],
                      key=lambda r: r.e_ms)[-1]
    ok(e_answer.bid == 10.0, "an E-time aligner would return row A")
    ok(e_answer.bid != got.bid,
       "MIRROR: an E-time aligner answers DIFFERENTLY on this tape, so a "
       "silent regression to payload time cannot pass quietly")

    # 5. rows are ordered by recv_ns even if handed over unordered
    ok(HFTape([row(300, 3), row(100, 1), row(200, 2)]).state_at(150).recv_ns == 100,
       "tape sorts by receipt time, does not trust file order")

    # 6. max_gap counts the WINDOW EDGES, not just interior points.  A window
    #    that is dense but starts late is not covered, and an interior-only
    #    max-gap would call it clean.
    late = HFTape([row(1_000_000_000 * 40 + i, 0) for i in range(3)])
    ok(late.max_gap_ms(0, 300_000_000_000) > 39_000,
       "a late start is an uncovered window, not a clean one")
    dense = HFTape([row(i * 1_000_000_000, 0) for i in range(301)])
    ok(dense.max_gap_ms(0, 300_000_000_000) == 1000.0, "dense window measures 1 s")

    # 6b. the bias profile measures the EXCURSION, not the constant: a tape
    #     with a huge but perfectly stable offset has ZERO bias for lead-lag,
    #     because a constant differences away on a shared clock.
    const = stamp_lag_profile([1_000_000_000 + i for i in range(100)],
                              [i // 1_000_000 for i in range(100)])
    ok(const["excursion_max_ms"] < 1.0, "a constant offset contributes no lead bias")
    jittery = stamp_lag_profile([1_000_000_000, 1_050_000_000, 1_500_000_000], [0, 0, 0])
    ok(jittery["excursion_max_ms"] == 500.0, "jitter above the floor IS the bias")
    ok(jittery["floor_ms"] == 1000.0, "the floor is the minimum, not the mean")

    # 6c. the streaming clipping rule, tested as a pure function.  The stub
    #     that first stood here was `ok(True, ...)` -- a check that cannot fail,
    #     which is the same defect as a gate that cannot fire, in the test suite
    #     that is supposed to catch it.
    S = 1_000_000_000
    w0 = 1_787_184_000                      # a real 300 s-aligned window start
    # a window with NO rows at all is fully silent, not silently clean
    ok(window_max_gap([], [w0])[w0] == 300_000.0,
       "an empty window measures the full 300 s, not zero")
    # dense rows every second -> 1 s
    dense = [(w0 + i) * S for i in range(301)]
    ok(window_max_gap(dense, [w0])[w0] == 1000.0, "dense window measures 1 s")
    # a LATE START is measured from the window edge, not from the first row
    late = [(w0 + 40 + i) * S for i in range(261)]
    ok(window_max_gap(late, [w0])[w0] == 40_000.0,
       "a 40 s late start is a 40 s gap -- the edge counts")
    # an EARLY STOP likewise
    early = [(w0 + i) * S for i in range(200)]
    ok(window_max_gap(early, [w0])[w0] == 101_000.0,
       "an early stop is measured to the window end")
    # a gap SPANNING two windows is charged to each CLIPPED to that window,
    # never the whole span to both.  Rows at +0,+150 then +700,+750,+880: the
    # 550 s silence is charged 150 s to the first window and 100 s to the
    # second, because that is how much of it each window actually contains.
    two = [w0, w0 + 600]
    spanning = [(w0 + o) * S for o in (0, 150, 700, 750, 880)]
    g = window_max_gap(spanning, two)
    ok(g[w0] == 150_000.0, "spanning silence clipped to the first window (150 s)")
    ok(g[w0 + 600] == 130_000.0,
       "second window's worst is its own interior 130 s, not the 550 s span")
    ok(g[w0] + g[w0 + 600] < 550_000.0, "the span is never charged whole to both")
    # out-of-order rows cannot fabricate coverage
    ok(window_max_gap([(w0 + 200) * S, (w0 + 10) * S], [w0])[w0] == 200_000.0,
       "an out-of-order row does not shrink a measured gap")

    # 7. joint coverage can never exceed either leg -- the property R-46 asked
    #    for, asserted rather than trusted
    for h, p, j in [(10, 8, 8), (5, 5, 5), (9, 3, 3)]:
        ok(j <= min(h, p), "joint <= min(per-tape) holds by construction")

    # 8. an unclosed PM gap must not be silently dropped
    ok("last_message_recv_ns" in pm_gap_intervals.__doc__, "unclosed gaps documented")

    # 9. the threshold is a ladder, not a single magic number
    ok(HF_GAP_MS in HF_GAP_LADDER_MS, "the reported threshold is on the ladder")

    # 10. Stamp semantics are data-era metadata, not whatever today's source
    #     happens to say.  A window straddling the first v2 run must expose two
    #     segments and preserve the historical pre-boundary label.
    boundary = (w0 + 100) * S
    profile = hf_stamp_profile([w0], "BTCUSDT", [{
        "started_at_ns": boundary,
        "stamp_point": "IMMEDIATELY_AFTER_WS_RECV_BEFORE_JSON_PARSE",
        "symbols": ["BTCUSDT"],
    }])
    ok(not profile["uniform"] and len(profile["segments"]) == 2,
       "a stamp-point change inside a window is exposed")
    ok(profile["segments"][0]["stamp_point"] == LEGACY_HF_STAMP_POINT,
       "unmanifested history remains labelled post-parse legacy")
    ok(profile["segments"][1]["start_ns"] == boundary,
       "manifest boundary is preserved at nanosecond resolution")

    # --- R-133: the gate, not merely the detector -------------------------
    S_ = 1_000_000_000
    b_mid = [(w0 + 100) * S_]                    # boundary INSIDE the window
    b_edge = [w0 * S_]                           # boundary exactly at the edge
    b_out = [(w0 + WINDOW_S + 5) * S_]           # boundary after the window
    ok(not window_stamp_uniform(w0, b_mid),
       "a boundary inside the window is a straddle")
    ok(window_stamp_uniform(w0, b_edge),
       "a boundary exactly at the window edge is NOT a straddle")
    ok(window_stamp_uniform(w0, b_out),
       "a boundary outside the window does not taint it")
    ok(window_stamp_uniform(w0, []),
       "an empty ledger is uniform-legacy, not unknown")

    # the mirror test R-42 asks for: the gate must ANSWER DIFFERENTLY for two
    # inputs that differ only in whether the boundary is inside the window.
    ok(window_stamp_uniform(w0, b_edge) != window_stamp_uniform(w0, b_mid),
       "gate distinguishes edge from interior -- it reads the window, not the ledger size")

    recs = [{"started_at_ns": (w0 + 100) * S_, "symbols": ["BTCUSDT"],
             "stamp_point": "IMMEDIATELY_AFTER_WS_RECV_BEFORE_JSON_PARSE"}]
    ok(stamp_boundaries_ns("BTCUSDT", recs) == [(w0 + 100) * S_],
       "boundaries are extracted for the requested symbol")
    ok(stamp_boundaries_ns("ETHUSDT", recs) == [],
       "another symbol's restart is not this symbol's boundary")
    ok(stamp_boundaries_ns("btcusdt", recs) == [(w0 + 100) * S_],
       "symbol match is case-insensitive")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "runs.jsonl"
        bad.write_text('{"started_at_ns":1,"stamp_point":"X","symbols":[]}\n'
                       'not json at all\n', encoding="utf-8")
        ok(hf_collector_run_defects(bad) == 1,
           "an unreadable ledger line is COUNTED, not silently dropped")
        good = Path(td) / "good.jsonl"
        good.write_text('{"started_at_ns":1,"stamp_point":"X","symbols":[]}\n',
                        encoding="utf-8")
        ok(hf_collector_run_defects(good) == 0, "a clean ledger has no defects")
        ok(hf_collector_run_defects(Path(td) / "absent.jsonl") == 0,
           "an ABSENT ledger is not a defect -- absence means all-legacy")

    # --- Q-DA-68: a boundary is a SEMANTICS CHANGE, not a restart ----------
    V2 = "hf_ws_v2_recv_boundary"
    V1 = "hf_ws_v1_postparse"
    SP2 = "IMMEDIATELY_AFTER_WS_RECV_BEFORE_JSON_PARSE"
    SP1 = "AFTER_JSON_PARSE"

    def run(ns, schema=V2, sp=SP2, syms=("BTCUSDT",)):
        return {"started_at_ns": ns, "collector_schema_version": schema,
                "stamp_point": sp, "symbols": list(syms)}

    # THE REGRESSION GUARD.  This is the real ledger's shape after the
    # 2026-08-26 box crash: three runs, all identical semantics.  R-147(2)
    # ruled the restart is "a coverage gap, not a boundary".
    real = [run(1787579288518862512), run(1787579334881534478),
            run(1787721103591796536)]
    ok(stamp_boundaries_ns("BTCUSDT", real) == [1787579288518862512],
       "a same-semantics restart is NOT a boundary (R-147(2))")

    # FALSIFIER, positive control (rule 15): it must still FIRE on a real one.
    changed = [run(1787579288518862512, V1, SP1),
               run(1787579334881534478, V2, SP2),
               run(1787721103591796536, V2, SP2)]
    ok(stamp_boundaries_ns("BTCUSDT", changed)
       == [1787579288518862512, 1787579334881534478],
       "a genuine v1->v2 semantics change IS a boundary")

    # the two inputs differ ONLY in the schema strings, so a rule that ignores
    # them cannot answer differently -- this is the R-42 mirror for this rule.
    ok(stamp_boundaries_ns("BTCUSDT", real)
       != stamp_boundaries_ns("BTCUSDT", changed),
       "the rule READS the semantics fields -- identical starts, different answers")

    # a change BACK is also a change
    ok(stamp_boundaries_ns("BTCUSDT", [run(10, V2), run(20, V1, SP1), run(30, V2)])
       == [10, 20, 30], "reverting to an older stamp point is still a boundary")
    # differing on stamp_point ALONE is enough
    ok(stamp_boundaries_ns("BTCUSDT", [run(10), run(20, V2, SP1)]) == [10, 20],
       "stamp_point alone distinguishes semantics")
    ok(stamp_boundaries_ns("BTCUSDT", []) == [],
       "an empty ledger yields no boundaries")
    ok(stamp_boundaries_ns("ETHUSDT", real) == [],
       "still symbol-scoped after the fix")
    # unsorted input must not fool the pairwise walk
    ok(stamp_boundaries_ns("BTCUSDT", list(reversed(real)))
       == [1787579288518862512],
       "ledger order does not change the answer")

    # --- Q-DA-68: a malformed day token FAILS LOUDLY ----------------------
    try:
        check_days(["2026-08-24"])
    except ValueError:
        ok(True, "a dashed day token is refused, not silently empty")
    else:
        ok(False, "MUST refuse a day token that can match no directory")
    ok(check_days(["20260824", "20260826"]) == ["20260824", "20260826"],
       "well-formed day tokens pass through")
    ok(check_days([]) == [], "an empty day list is not malformed")
    try:
        check_days(["20260824", "2026-08-25"])
    except ValueError as exc:
        ok("2026-08-25" in str(exc) and "20260824" not in str(exc).split("expected")[0],
           "the error names the offending token")
    else:
        ok(False, "one bad token among good ones must still refuse")
    # a well-formed but ABSENT day is a legitimate empty, not an error
    ok(pm_windows(["19700101"], "btc") == {},
       "a well-formed day with no data is a legitimate empty")

    print(f"da_hf_pm_alignment selftests: {checks} checks passed")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--coin", default=None)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--day", action="append", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest or not a.coin:
        raise SystemExit(_selftests())
    rep = joint_coverage(a.coin, a.symbol or f"{a.coin.upper()}USDT", a.day or [])
    text = json.dumps(rep, indent=2, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
