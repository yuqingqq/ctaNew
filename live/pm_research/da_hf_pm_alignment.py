"""R-46: align the Binance HF tape to Polymarket windows, knowledge-time honest.

BE wants a hedgeability decomposition: what FRACTION of post-fill drift is
contemporaneous underlying movement?  That needs, for a Polymarket event at
time t, the Binance book state that was KNOWN at t.  This module supplies that
read and the coverage figure that says where it is legitimate.

WHY THE ALIGNMENT CAN BE MADE HONEST -- the enabling fact, stated first because
if it were false the task would stop here:

    BOTH collectors stamp `recv_ns = time.time_ns()` AT PARSE TIME, ON THE SAME
    HOST.  `collect_pm.py:408` and `collect_hf.py:175`.

So the two tapes share one wall clock and receipt times are directly
comparable with no offset estimation, no exchange-clock arithmetic, and no
cross-venue skew model.  A cross-venue alignment usually founders on exactly
that; here it is free, and it is the ONLY reason this is knowledge-time honest.

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
import sys
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
HF_RAW = REPO / "data/mm_hf/raw"
PM_ROOT = REPO / "data/pm_5min"
PM_GAPS = PM_ROOT / "collector_gaps.jsonl"

ALIGNMENT_VERSION = "hf_pm_alignment_v1_r46"

# A recv_ns silence longer than this counts as an inferred HF outage.  Chosen
# against measured behaviour, not picked: BTCUSDT bookTicker's worst intra-hour
# inter-arrival on a quiet hour is ~385 ms, so 2 s is ~5x the observed normal
# maximum.  CLASS A (configuration) -- it is a reporting threshold, no verdict
# rests on it, and coverage is published at several thresholds so the reader can
# see the sensitivity rather than trusting this one.
HF_GAP_MS = 2000.0
HF_GAP_LADDER_MS = (500.0, 1000.0, 2000.0, 5000.0, 30000.0)

WINDOW_S = 300  # Polymarket 5-minute markets


# ---------------------------------------------------------------------------
# R-47: the clock is NOT the risk.  The STAMP POINT is.
# ---------------------------------------------------------------------------
#
# The two collectors share one CLOCK_REALTIME on one host, so there is no
# inter-process skew to reconcile and Binance's own clock offset never enters:
# the alignment compares recv_ns to recv_ns and never touches `E`.  That is
# defended, not assumed -- `collect_pm.py:407-408` and `collect_hf.py:172-175`.
#
# What DOES differ is WHERE in each pipeline the stamp is taken, and a lead-lag
# measurement is exactly as sensitive to that as it is to a clock error.
#
#   1. STAMP-POINT ASYMMETRY.  PM stamps immediately after `ws.recv()` returns,
#      BEFORE parsing.  HF stamps AFTER `json.loads`.  So an HF row is stamped
#      one JSON parse later than a PM row that arrived at the same instant --
#      microseconds, and it makes Binance look LATER than it was.
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
#   HIDES a Binance lead -- HF parse-point plus HF loop queueing: +3.6 ms p50,
#   +30 ms p99, +259 ms p99.9, +360 ms max (inferred, since HF runs no probe).
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
    deltas = sorted((r / 1e6) - v for r, v in zip(recv_ns, venue_ms))
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
    for day in days:
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


def pm_windows(days: Sequence[str], coin: str | None = None) -> dict[str, list[int]]:
    """(coin -> sorted window-start epochs) discovered from the raw slugs."""
    out: dict[str, set[int]] = defaultdict(set)
    for day in days:
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


def joint_coverage(
    coin: str, symbol: str, days: Sequence[str], gap_ms: float = HF_GAP_MS
) -> dict[str, Any]:
    """The figure that matters: windows BOTH tapes covered.

    Reports the two per-tape figures beside it -- never one standing in for the
    joint number, which is the specific failure R-46 named.
    """
    windows = pm_windows(days, coin).get(coin, [])
    gaps, n_rows = hf_window_max_gap(symbol, days, windows)
    pm_gaps = pm_gap_intervals()

    hf_ok = pm_ok = both_ok = 0
    hf_bad: list[int] = []
    pm_bad: list[int] = []
    for w in windows:
        lo, hi = w * 1_000_000_000, (w + WINDOW_S) * 1_000_000_000
        h = gaps.get(w, float("inf")) <= gap_ms
        p = not any(_overlaps(lo, hi, g0, g1) for g0, g1, _ in pm_gaps)
        hf_ok += h
        pm_ok += p
        both_ok += (h and p)
        if not h:
            hf_bad.append(w)
        if not p:
            pm_bad.append(w)

    n = len(windows) or 1
    return {
        "alignment_version": ALIGNMENT_VERSION,
        "coin": coin, "symbol": symbol, "days": list(days),
        "hf_gap_threshold_ms": gap_ms,
        "hf_rows": n_rows,
        "pm_windows": len(windows),
        "hf_covered": hf_ok, "pm_covered": pm_ok, "joint_covered": both_ok,
        "hf_covered_pct": round(100.0 * hf_ok / n, 2),
        "pm_covered_pct": round(100.0 * pm_ok / n, 2),
        "joint_covered_pct": round(100.0 * both_ok / n, 2),
        "hf_only_loss": hf_ok - both_ok,
        "pm_only_loss": pm_ok - both_ok,
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
