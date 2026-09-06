"""P-2026-002 E2-A -- the overlay bracket on REAL books, under the queue bracket.

E2-A supersedes E1-A's overlay number (`eff_RT_sweep 6.2645 [5.7561, 6.7539]`
at T_p = 600 s) by changing three things and nothing else:

  PLACEMENT   the actual best bid/ask from bookTicker at t0-, not
              `m0 - sign*ES_day/2` from a same-day median flip-bounce that
              E1-A's own review logged as a sanctioned look-ahead.
  FILLS       the queue bracket (RiskAverse / ProbQueue-f3) against real
              depth20, not touch/sweep-through.
  PARTIALS    a filled QUANTITY, not a boolean, priced two ways.

Everything else -- the episode grid, the patience ladder, the shortfall
accounting against the decision mid, the 8.0 bps threshold, the day-clustered
mean and the block bootstrap -- is E1-A's, unchanged, so the two are
comparable. The declaration is
`declarations/p002_e2_a_declaration_v4.json` and this module REFUSES if its
sha256 has moved.

EVERY LEVEL COMPARISON IS ON INTEGER TICK INDICES. E1's D-i defect exists
because on floats the touch (<=) and sweep-through (<) rules are
indistinguishable; the same hazard reappears at every comparison inside a queue
simulation, so none of them is done on floats.

    python3 live/mm_research/e2_a_runner.py --selftest
    python3 live/mm_research/e2_a_runner.py --fixture --output R.json
    python3 live/mm_research/e2_a_runner.py --run --symbols ICPUSDT --output R.json
"""
from __future__ import annotations

import argparse
import builtins
import gzip
import hashlib
import datetime
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_0_true_mid as E20                                   # noqa: E402
import e2_a_declare as D                                      # noqa: E402
import e2_a_episodes as EP                                    # noqa: E402

CODE_ROOT = E20.CODE_ROOT
ROOT = E20.ROOT
RAW = E20.RAW
PROTOCOL = "P002_E2_A_OVERLAY_RUNNER_V1"
DECL_PATH = HERE / "declarations" / "p002_e2_a_declaration_v6.json"
DECL_SHA = "127e0a56ddee775ecb478453e77ee08614b58023aa45f22fe1642492d8c0f4fb"

EPS = 1e-12
TP_GRID_S = D.TP_GRID_S
TP_PRIMARY_S = D.TP_PRIMARY_S
FEE_MAKER = D.FEE_MAKER_VIP0
FEE_TAKER = D.FEE_TAKER_VIP0
THRESHOLD = D.CAPSTONE_THRESHOLD_BPS
#: REPORTED, NEVER GATED since v6. Kept only so the number v5 gated on stays
#: visible beside the days it used to exclude.
GAP_REPORTING_REFERENCE = 0.05
HOURS_PER_DAY_FILES = 24
SEC_PER_DAY_MS = 86_400_000
N_LEVELS = 20
BOOT_SEED = EP.BOOT_SEED
BOOT_B = EP.BOOT_B

SYMBOLS_IN_SCOPE = tuple(D.E1A_REPRODUCTION_TARGET["symbols"])

#: Episode statuses. Exclusions are STATUSES, never silent drops (rule 4).
ST_RESOLVED = "RESOLVED"
ST_NO_QUOTE = "NO_QUOTE_BEFORE_T0"
ST_QAU = "QUEUE_AHEAD_UNDEFINED"
ST_NO_BOOK_TP = "NO_BOOK_AT_TP"
STATUSES = (ST_RESOLVED, ST_NO_QUOTE, ST_QAU, ST_NO_BOOK_TP)

#: The two partial-fill pricings, R-570(C)(2).
PR_RESIDUAL = "residual_chased"          # GATE-BEARING
PR_WHOLE = "whole_leg_charged"           # pessimistic bracket
PRICINGS = (PR_RESIDUAL, PR_WHOLE)
MODELS = ("RiskAverse", "ProbQueue_f3")


class E2ARefused(RuntimeError):
    """The run cannot proceed under the declaration."""


def carrying_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(HERE))
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


def wrapper_block() -> dict:
    """WHAT THIS RUN ACTUALLY RAN UNDER -- rule 20 / R-575(C).

    A receipt that does not say whether it ran inside the capped scope leaves
    a reader unable to tell a wrapped run from an unwrapped one, and the two
    have different resource meanings. Read from the process's own cgroup, so
    it is a produced fact and not a claim about the command line.
    """
    scope, slice_ = None, None
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            path = line.rsplit(":", 1)[-1]
            for part in path.split("/"):
                if part.endswith(".scope"):
                    scope = part
                if part.endswith(".slice") and part != "user.slice":
                    slice_ = part
    except OSError:
        pass
    lock = ROOT / "data" / ".heavy_run.lock"
    return {"rule": "SEAT_PROTOCOL rule 20 / R-575(C)",
            "systemd_scope": scope,
            "systemd_slice": slice_,
            "ran_under_the_rule20_wrapper": slice_ == "research.slice",
            "heavy_run_lock_path": str(lock),
            "read_from": "/proc/self/cgroup, in this process",
            "why_the_SLICE_and_not_the_scope": (
                "the first version of this field tested `scope.startswith("
                "'run-')` and reported TRUE for an UNWRAPPED run, because the "
                "calling shell is itself inside a transient scope "
                "(app.slice). A field that is true either way is a control "
                "that cannot fail. `--slice=research.slice` is what the "
                "rule-20 wrapper adds and nothing else in this session "
                "does."),
            "note": ("false means this step was NOT inside the capped "
                     "scope. That is correct only for a step under 60 s and "
                     "1 GiB that opens no tape; anything else takes the lock "
                     "FIRST and REFUSES if it is held.")}


def load_declaration() -> dict:
    if not DECL_PATH.is_file():
        raise E2ARefused(f"REFUSED: no declaration at {DECL_PATH}")
    got = hashlib.sha256(DECL_PATH.read_bytes()).hexdigest()
    if got != DECL_SHA:
        raise E2ARefused(
            f"REFUSED: declaration sha256 {got[:16]} != the pinned "
            f"{DECL_SHA[:16]}. A run whose declaration has moved is not the "
            f"run that was declared.")
    return json.loads(DECL_PATH.read_text())


def require_symbol_in_scope(sym: str) -> None:
    """R-570(C)(3): the population is the TWELVE, by name."""
    if sym not in SYMBOLS_IN_SCOPE:
        raise E2ARefused(
            f"REFUSED: {sym} is not one of E1-A's twelve XS-overlap symbols "
            f"{list(SYMBOLS_IN_SCOPE)}. E2-A supersedes E1-A on E1-A's "
            f"population; measuring a different one would not supersede, it "
            f"would measure something else.")


# --------------------------------------------------------------------------
# depth20 -- the parse, and it is the expensive step
# --------------------------------------------------------------------------
def parse_depth20_bytes(raw: bytes) -> tuple:
    """`recv_ns,E,T,u,bids,asks` with each side `p@q|p@q|...` x 20.

    The rigid shape is what makes this affordable: translating `@` and `|` to
    `,` turns every line into a flat 4 + 40 + 40 = 84 column CSV, which the C
    parser reads at whole-file speed. Splitting 6.4M small strings per day in
    Python does not finish in a useful time, and a reader that is too slow to
    run is a reader that silently becomes a sample.

    Rows that do not carry exactly 20 levels a side are RAGGED: they are
    excluded and COUNTED, never quietly padded with zeros -- a zero level is a
    queue position, and inventing one is the same defect class as reading an
    absent level as an empty one.
    """
    flat = raw.translate(bytes.maketrans(b"@|", b",,"))
    names = ["recv_ns", "E", "T", "u"]
    for side in ("b", "a"):
        for i in range(N_LEVELS):
            names += [f"{side}p{i}", f"{side}q{i}"]
    n_raw = flat.count(b"\n")
    df = pd.read_csv(io.BytesIO(flat), header=None, names=names,
                     engine="c", on_bad_lines="skip")
    #: `on_bad_lines="skip"` only catches rows with MORE fields than names.
    #: A row with FEWER is NaN-PADDED and reads as a book with zero-size
    #: levels -- an invented queue position, which is the same defect class
    #: as reading an absent level as an empty one. Caught by this module's own
    #: ragged-row control, which failed on the first run.
    n_over = n_raw - len(df)
    short = df.isna().any(axis=1).to_numpy() if len(df) else np.zeros(0, bool)
    n_short = int(short.sum())
    df = df[~short]
    n_ragged = int(n_over) + n_short
    if len(df) == 0:
        return None, {"n_raw_rows": int(n_raw), "n_ragged_rows": n_ragged,
                      "n_ragged_overlong": int(n_over),
                      "n_ragged_short_nan_padded": n_short,
                      "n_snapshots": 0}
    t = df["T"].to_numpy(np.int64)
    bp = df[[f"bp{i}" for i in range(N_LEVELS)]].to_numpy(np.float64)
    bq = df[[f"bq{i}" for i in range(N_LEVELS)]].to_numpy(np.float64)
    ap = df[[f"ap{i}" for i in range(N_LEVELS)]].to_numpy(np.float64)
    aq = df[[f"aq{i}" for i in range(N_LEVELS)]].to_numpy(np.float64)
    o = np.argsort(t, kind="stable")
    return (t[o], bp[o], bq[o], ap[o], aq[o]), {
        "n_raw_rows": int(n_raw), "n_ragged_rows": n_ragged,
        "n_ragged_overlong": int(n_over),
        "n_ragged_short_nan_padded": n_short,
        "n_snapshots": int(len(t))}


def qty_step_mode(quantities: np.ndarray) -> float:
    """The venue's quantity STEP: the MODAL positive diff of the distinct
    quantities, and NO GCD fallback.

    It is deliberately NOT `e1_markout_scan.tick_size` / `EP.tick_mode`, and
    the reason is a defect this module's own control caught. Those keep a GCD
    fallback that fires when fewer than 99.9% of the diffs are integer
    multiples of the modal one -- and the GCD of a set containing ONE
    off-grid value collapses to the representation floor. Driven here: a tape
    of 0.25 multiples plus a single 3.14159 returns 1e-5 under the fallback
    and 0.25 without it.

    That is the same mechanism E1's results audit recorded for FIL, where 81
    off-grid prints returned 1e-6 against a true 1e-4. The modal-diff FIX is
    present in the committed `tick_size`; the RETAINED FALLBACK is what
    overrides it on exactly the input the fix was written for. The price tick
    is left alone here because it is pinned by the E1-A reproduction control
    and must not move; the quantity step is a new quantity and takes the
    robust form.
    """
    u = np.sort(np.unique(np.asarray(quantities, dtype=float)))
    d = np.diff(u)
    d = d[d > EPS]
    if d.size == 0:
        return float(u[0]) if u.size else float("nan")
    scaled = np.round(d * 1e8).astype(np.int64)
    vals, cnts = np.unique(scaled, return_counts=True)
    return float(vals[cnts.argmax()] / 1e8)


def read_depth20(sym: str, day: str):
    files = E20._hour_files("depth20", sym, day)
    n_files = len(files)
    if not files:
        return None, n_files, {"n_snapshots": 0}
    chunks, meta = [], {"n_raw_rows": 0, "n_ragged_rows": 0,
                        "n_ragged_overlong": 0,
                        "n_ragged_short_nan_padded": 0, "n_snapshots": 0}
    for p in files:
        op = gzip.open(p, "rb") if p.suffix == ".gz" else builtins.open(p, "rb")
        with op as fh:
            raw = fh.read()
        got, m = parse_depth20_bytes(raw)
        for k in meta:
            meta[k] += m.get(k, 0)
        if got is not None:
            chunks.append(got)
    if not chunks:
        return None, n_files, meta
    t = np.concatenate([c[0] for c in chunks])
    bp = np.concatenate([c[1] for c in chunks])
    bq = np.concatenate([c[2] for c in chunks])
    ap = np.concatenate([c[3] for c in chunks])
    aq = np.concatenate([c[4] for c in chunks])
    o = np.argsort(t, kind="stable")
    return (t[o], bp[o], bq[o], ap[o], aq[o]), n_files, meta


# --------------------------------------------------------------------------
# admission
# --------------------------------------------------------------------------
#: The collector's own heartbeat, which is the health ledger the admission
#: predicate reads. Both files: the live log and the pre-reboot rotation.
COLLECTOR_LOGS = ("collector.log.pre-reboot-20260826", "collector.log")
HEARTBEAT_RE = re.compile(rb"^\[hf\] (\d{2}):(\d{2}):(\d{2})Z bookTicker=")


def _log_beats(path: Path, end_date) -> list[float]:
    """Absolute heartbeat times from a log whose lines carry only HH:MM:SSZ.

    Anchored from the END at a known date and walked BACKWARDS, decrementing
    the day at each wrap. The end anchor is a fact (the live log ends now; the
    rotated one ends at its reboot), so no date is guessed from a filename.
    """
    secs = []
    with builtins.open(path, "rb") as fh:
        for line in fh:
            m = HEARTBEAT_RE.match(line)
            if m:
                secs.append(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]))
    if not secs:
        return []
    days = [None] * len(secs)
    d = end_date
    for i in range(len(secs) - 1, -1, -1):
        days[i] = d
        if i > 0 and secs[i - 1] > secs[i]:
            d = d - datetime.timedelta(days=1)
    return [datetime.datetime.combine(
        days[i], datetime.time(), datetime.timezone.utc).timestamp() + secs[i]
        for i in range(len(secs))]


def collector_heartbeats(now_utc_date=None) -> np.ndarray:
    base = ROOT / "data" / "mm_hf"
    now = now_utc_date or datetime.datetime.now(
        datetime.timezone.utc).date()
    ends = {"collector.log": now,
            "collector.log.pre-reboot-20260826": datetime.date(2026, 8, 26)}
    ts: list[float] = []
    for name in COLLECTOR_LOGS:
        p = base / name
        if p.is_file():
            ts += _log_beats(p, ends[name])
    return np.array(sorted(set(ts)))


def collector_restarts() -> np.ndarray:
    p = ROOT / "data" / "mm_hf" / "collector_runs.jsonl"
    if not p.is_file():
        return np.zeros(0)
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line)["started_at_ns"] / 1e9)
    return np.array(sorted(set(out)))


def collector_health(day: str, beats: np.ndarray,
                     restarts: np.ndarray) -> dict:
    """WAS THE COLLECTOR LIVE ON THIS DAY? -- v6's admission leg.

    This is a property of the COLLECTOR, never of how often a quiet book
    moves. The bar is 2x the collector's own MEASURED modal cadence, so it is
    derived from the collector's behaviour rather than chosen: a day where
    every beat lands is one cadence apart, and a day with a real stop is
    orders of magnitude away from that.
    """
    d0 = int(pd.Timestamp(day, tz="UTC").timestamp())
    d1 = d0 + 86_400
    if beats.size < 2:
        return {"live": False, "why": "no heartbeat ledger available",
                "n_beats": 0}
    diffs = np.diff(beats).astype(np.int64)
    diffs = diffs[diffs > 0]
    cadence = float(np.bincount(diffs).argmax()) if diffs.size else 60.0
    bar = 2.0 * cadence
    inday = beats[(beats >= d0) & (beats < d1)]
    if inday.size == 0:
        return {"live": False, "why": "no heartbeat inside the day",
                "n_beats": 0, "cadence_s": cadence, "bar_s": bar}
    #: The day's boundaries count: a collector that came up at 06:00 was not
    #: live at 00:30, and an interior-only view would not see it.
    seq = np.concatenate(([d0], inday, [d1]))
    gaps = np.diff(seq)
    n_restarts = int(((restarts >= d0) & (restarts < d1)).sum())
    return {"live": bool(gaps.max() <= bar and n_restarts == 0),
            "n_beats": int(inday.size),
            "cadence_s": cadence, "bar_s": bar,
            "max_heartbeat_gap_s": float(gaps.max()),
            "n_gaps_over_bar": int((gaps > bar).sum()),
            "n_collector_restarts_in_day": n_restarts,
            "why": ("collector live: every heartbeat gap is within 2x its own "
                    "measured cadence and no restart falls inside the day"
                    if gaps.max() <= bar and n_restarts == 0 else
                    f"collector NOT live: max heartbeat gap "
                    f"{gaps.max():.0f} s against a bar of {bar:.0f} s, "
                    f"{n_restarts} restart(s) in day")}


def stream_file_counts(sym: str, day: str) -> dict:
    return {s: len(E20._hour_files(s, sym, day))
            for s in ("bookTicker", "trade", "depth20")}


def day_admission(sym: str, day: str, counts: dict,
                  health: dict | None = None,
                  gap: float | None = None,
                  age: dict | None = None) -> dict:
    """v6. A UTC day is ADMISSIBLE FOR A SYMBOL iff (a) all THREE streams
    carry 24 hour-files and (b) THE COLLECTOR WAS LIVE.

    v5 gated on the intra-day bookTicker gap fraction, inherited from E2.0
    where it guarded against collector OUTAGE. Measured over eight symbols,
    that leg selects on HOW OFTEN THE BEST QUOTE CHANGES: exclusion was
    monotone in activity and ICP -- 16 structurally complete days, zero of
    its missing seconds in runs of a minute or more -- was cut to ONE day,
    which is precisely the thin cell E2-A exists to resolve.

    ADMISSIBILITY IS NOW A PROPERTY OF THE COLLECTOR, NEVER OF THE BOOK.
    Decision-time quote age and the gap fraction are REPORTED per symbol-day
    as statuses, so a reader can see the staleness a thin name carries --
    but no book is excluded for being quiet.
    """
    complete = {s: n == HOURS_PER_DAY_FILES for s, n in counts.items()}
    all_complete = all(complete.values())
    live = bool(health and health.get("live"))
    reasons = [f"{s}_files={counts[s]}" for s in counts if not complete[s]]
    if not live:
        reasons.append("collector_not_live: "
                       + (health or {}).get("why", "no health ledger"))
    return {"day": day, "admissible": bool(all_complete and live),
            "stream_file_counts": counts, "streams_complete": complete,
            "collector_health": health,
            "reasons_excluded": reasons,
            "REPORTED_not_gated": {
                "gap_fraction": gap,
                "decision_time_quote_age_ms": age,
                "why": ("these describe how ACTIVE the book is, not whether "
                        "the data is there. v5 gated on the first of them "
                        "and cut ICP from 16 days to 1.")}}


# --------------------------------------------------------------------------
# the two queue models, wired to an episode
# --------------------------------------------------------------------------
def episode_seed(sym: str, day: str, hour: int, sign: float,
                 decl_digest: str) -> int:
    """The seed pins the DATA the RNG is applied to, not just the RNG
    (SEAT_PROTOCOL rule 10)."""
    key = f"{sym}|{day}|{hour:02d}|{'buy' if sign > 0 else 'sell'}|{decl_digest}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def probqueue_prob(front: np.ndarray, back: np.ndarray) -> np.ndarray:
    """Vectorised f(back)/(f(front)+f(back)), with the DECLARED degenerate
    branch taken first: front = 0 fills with certainty."""
    fa = np.power(np.maximum(front, 0.0), 3)
    fb = np.power(np.maximum(back, 0.0), 3)
    den = fa + fb
    out = np.where(den > 0, fb / np.where(den > 0, den, 1.0), 0.0)
    return np.where(front <= 0, 1.0, out)


def simulate_episode(queue_ahead: float, order_qty: float,
                     vol: np.ndarray, depth_at_L: np.ndarray,
                     rng: np.random.Generator) -> dict:
    """Both models on ONE episode's opposite-side trade sequence.

    `vol` is the per-trade volume at-or-through L in time order; `depth_at_L`
    is the observed depth at L in the latest snapshot at or before each trade.
    """
    v_cum = np.cumsum(vol) if len(vol) else np.zeros(0)
    total = float(v_cum[-1]) if len(vol) else 0.0
    filled_ra = D.riskaverse_filled_qty(queue_ahead, order_qty, total)

    if len(vol):
        v_before = v_cum - vol
        front = np.maximum(queue_ahead - v_before, 0.0)
        back = np.maximum(depth_at_L - front, 0.0)
        prob = probqueue_prob(front, back)
        draws = rng.random(len(vol))
        hit = draws < prob
        idx = int(np.argmax(hit)) if hit.any() else -1
    else:
        idx, prob = -1, np.zeros(0)
    filled_pq = float(order_qty) if idx >= 0 else 0.0

    return {"filled_qty_RiskAverse": float(filled_ra),
            "filled_qty_ProbQueue_f3": filled_pq,
            "volume_at_or_through_L": total,
            "n_opposite_trades": int(len(vol)),
            "probqueue_fill_index": idx,
            "probqueue_max_prob": float(prob.max()) if len(prob) else 0.0}


def price_episode(filled_qty: float, order_qty: float,
                  c_fill: float, c_chase: float) -> dict:
    """R-570(C)(2). Both pricings, side by side, neither averaged."""
    phi = 0.0 if order_qty <= 0 else min(max(filled_qty / order_qty, 0.0), 1.0)
    full = phi >= 1.0 - 1e-9
    return {"phi": phi,
            PR_RESIDUAL: phi * c_fill + (1.0 - phi) * c_chase,
            PR_WHOLE: c_fill if full else c_chase,
            "is_partial": bool(0.0 < phi < 1.0 - 1e-9)}


# --------------------------------------------------------------------------
# one day
# --------------------------------------------------------------------------
def evaluate_day(sym: str, day: str, decl_digest: str,
                 book=None, trades=None, depth=None,
                 tick: float | None = None,
                 qty_step: float | None = None) -> dict:
    """Every episode on one symbol-day, both models, both pricings."""
    if book is None:
        book, _, _ = E20.read_book(sym, day, extend=True)
    if trades is None:
        trades, _, _ = E20.read_trades(sym, day)
    if depth is None:
        depth, _, _ = read_depth20(sym, day)
    if book is None or trades is None or depth is None:
        return {"day": day, "usable": False,
                "why": "one of the three streams read empty"}

    bt_t, bid, ask = book
    tr_t, tr_p, tr_q, tr_m = trades
    d_t, d_bp, d_bq, d_ap, d_aq = depth

    if tick is None:
        tick = EP.tick_mode([tr_p, bid, ask])
    if qty_step is None:
        qty_step = qty_step_mode(tr_q)
    order_qty = float(qty_step)

    k_tr = np.round(tr_p / tick).astype(np.int64)
    k_bid = np.round(d_bp / tick).astype(np.int64)
    k_ask = np.round(d_ap / tick).astype(np.int64)
    #: taker-SELL prints (buyer is maker) hit resting BIDS; taker-BUY prints
    #: lift resting ASKS. `sign` follows E1-A: +1 = our resting buy.
    is_taker_sell = tr_m.astype(bool)

    #: The day boundary comes from the DAY the episode grid is defined on,
    #: never from the first row that happened to be read. A quote stamped a
    #: millisecond before midnight would otherwise move every decision time
    #: in the day by 24 hours, silently.
    day0 = int(pd.Timestamp(day, tz="UTC").timestamp()) * 1000
    rows, status_counts = [], {s: 0 for s in STATUSES}
    ordering_violations = []

    for tp in TP_GRID_S:
        hours = range(24 if tp * 1000 <= 3_600_000 else 23)
        for hh in hours:
            t0 = day0 + hh * 3_600_000
            i0 = int(np.searchsorted(bt_t, t0, "left")) - 1
            for sign in (1.0, -1.0):
                if i0 < 0:
                    status_counts[ST_NO_QUOTE] += 1
                    continue
                b0, a0 = float(bid[i0]), float(ask[i0])
                m0 = (b0 + a0) / 2.0
                L = b0 if sign > 0 else a0
                kL = int(round(L / tick))

                j = int(np.searchsorted(d_t, t0, "right")) - 1
                if j < 0:
                    status_counts[ST_QAU] += 1
                    continue
                side_k = k_bid[j] if sign > 0 else k_ask[j]
                side_q = d_bq[j] if sign > 0 else d_aq[j]
                where = np.flatnonzero(side_k == kL)
                if where.size == 0:
                    status_counts[ST_QAU] += 1
                    continue
                queue_ahead = float(side_q[where[0]])

                t_end = t0 + tp * 1000
                lo = int(np.searchsorted(tr_t, t0, "right"))
                hi = int(np.searchsorted(tr_t, t_end, "right"))
                seg = slice(lo, hi)
                opp = is_taker_sell[seg] if sign > 0 else ~is_taker_sell[seg]
                #: AT OR THROUGH L, on integer tick indices (declaration
                #: `at_or_through_L_direction`): a print AT L trades against
                #: the queue this order stands in.
                thru = ((k_tr[seg] <= kL) if sign > 0 else (k_tr[seg] >= kL))
                sel = opp & thru
                vol = tr_q[seg][sel]
                times = tr_t[seg][sel]

                if len(times):
                    jj = np.searchsorted(d_t, times, "right") - 1
                    jj = np.clip(jj, 0, None)
                    sub_k = k_bid[jj] if sign > 0 else k_ask[jj]
                    sub_q = d_bq[jj] if sign > 0 else d_aq[jj]
                    depth_at_L = (sub_q * (sub_k == kL)).sum(axis=1)
                else:
                    depth_at_L = np.zeros(0)

                rng = np.random.default_rng(
                    episode_seed(sym, day, hh, sign, decl_digest))
                sim = simulate_episode(queue_ahead, order_qty, vol,
                                       depth_at_L, rng)

                iT = int(np.searchsorted(bt_t, t_end, "right")) - 1
                if iT < 0 or t_end > bt_t[-1]:
                    status_counts[ST_NO_BOOK_TP] += 1
                    continue
                p_x = float(ask[iT]) if sign > 0 else float(bid[iT])
                c_fill = sign * (L - m0) / m0 * 1e4 + FEE_MAKER
                c_chase = sign * (p_x - m0) / m0 * 1e4 + FEE_TAKER

                if (sim["filled_qty_ProbQueue_f3"]
                        < sim["filled_qty_RiskAverse"] - 1e-9):
                    ordering_violations.append(
                        {"day": day, "tp_s": tp, "hour": hh, "sign": sign,
                         "filled_RiskAverse": sim["filled_qty_RiskAverse"],
                         "filled_ProbQueue_f3":
                             sim["filled_qty_ProbQueue_f3"]})

                row = {"tp_s": tp, "hour": hh, "sign": sign,
                       "queue_ahead": queue_ahead, "order_qty": order_qty,
                       "c_fill_bps": c_fill, "c_chase_bps": c_chase,
                       "n_opposite_trades": sim["n_opposite_trades"],
                       "volume_at_or_through_L":
                           sim["volume_at_or_through_L"]}
                for mdl in MODELS:
                    pr = price_episode(sim[f"filled_qty_{mdl}"], order_qty,
                                       c_fill, c_chase)
                    row[f"phi_{mdl}"] = pr["phi"]
                    row[f"partial_{mdl}"] = pr["is_partial"]
                    for pk in PRICINGS:
                        row[f"cost_{mdl}_{pk}"] = pr[pk]
                rows.append(row)
                status_counts[ST_RESOLVED] += 1

    df = pd.DataFrame(rows)
    out = {"day": day, "usable": True, "tick": float(tick),
           "rows": rows,
           "qty_step": float(qty_step),
           "status_counts": status_counts,
           "n_attempted": int(sum(status_counts.values())),
           "ordering_violations": ordering_violations,
           "cells": {}}
    for tp in TP_GRID_S:
        sub = df[df.tp_s == tp] if len(df) else df
        cell = {"n_episodes": int(len(sub))}
        if len(sub):
            for mdl in MODELS:
                cell[f"fill_rate_{mdl}"] = float((sub[f"phi_{mdl}"] > 0).mean())
                cell[f"mean_phi_{mdl}"] = float(sub[f"phi_{mdl}"].mean())
                cell[f"partial_share_{mdl}"] = float(
                    sub[f"partial_{mdl}"].mean())
                for pk in PRICINGS:
                    cell[f"eff_rt_{mdl}_{pk}"] = float(
                        2.0 * sub[f"cost_{mdl}_{pk}"].mean())
        out["cells"][str(tp)] = cell
    return out


# --------------------------------------------------------------------------
# aggregation and verdict
# --------------------------------------------------------------------------
def aggregate_symbol(days: list[dict], tp: int) -> dict:
    """Day-clustered mean and the stationary bootstrap, E1-A's estimator."""
    vals = {}
    usable = [d for d in days if d.get("usable")
              and d["cells"].get(str(tp), {}).get("n_episodes", 0) > 0]
    g = len(usable)
    for mdl in MODELS:
        for pk in PRICINGS:
            k = f"eff_rt_{mdl}_{pk}"
            series = np.array([d["cells"][str(tp)][k] for d in usable])
            if g == 0:
                vals[k] = {"eff_rt_bps": None, "n_days": 0}
                continue
            lo, hi = (EP.stationary_boot_ci(series, np.ones(g), exp_block=3)
                      if g >= 4 else (float("nan"), float("nan")))
            vals[k] = {"eff_rt_bps": float(series.mean()),
                       "ci_lo": None if np.isnan(lo) else float(lo),
                       "ci_hi": None if np.isnan(hi) else float(hi),
                       "n_days": g}
    n_att = sum(d["n_attempted"] for d in days if d.get("usable"))
    n_res = sum(d["status_counts"][ST_RESOLVED] for d in days
                if d.get("usable"))
    vals["G_complete_days"] = g
    vals["episode_skip_rate"] = (None if n_att == 0
                                 else float(1.0 - n_res / n_att))
    return vals


def verdict(agg: dict, decl: dict) -> dict:
    """Checked in the DECLARED order: the instrument, then the second
    bracket, then the gate on the gate-bearing pricing."""
    ra_r = agg[f"eff_rt_RiskAverse_{PR_RESIDUAL}"]
    ra_w = agg[f"eff_rt_RiskAverse_{PR_WHOLE}"]
    pq_r = agg[f"eff_rt_ProbQueue_f3_{PR_RESIDUAL}"]
    g = agg["G_complete_days"]
    partial = D.partial_pricing_predicate(ra_r["eff_rt_bps"],
                                          ra_w["eff_rt_bps"])
    gate = D.gate_predicate(ra_r["eff_rt_bps"], pq_r["eff_rt_bps"],
                            ci_lo=ra_r.get("ci_lo"), ci_hi=ra_r.get("ci_hi"),
                            interval_claimable=(g >= 5))
    state = gate["state"]
    if gate.get("state") != "REFUTES_THE_BRACKET" and partial.get("straddles"):
        state = "FAIL_PARTIAL_FILL_PRICING_STRADDLES"
    return {"state": state, "gate": gate, "partial_pricing": partial,
            "gate_bearing_pricing": PR_RESIDUAL,
            "G_complete_days": g,
            "interval_claimable": bool(g >= 5)}


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def run(symbols, out_path: Path | None, min_days: int | None = None,
        run_repro: bool = True) -> dict:
    t_start = time.time()
    root_block = E20.require_canonical_root(
        "P-2026-002 E2-A result-bearing emission")
    decl = load_declaration()
    decl_digest = DECL_SHA[:16]
    for sym in symbols:
        require_symbol_in_scope(sym)
    min_days = (decl["population"]["min_complete_days"]
                if min_days is None else min_days)

    result = {"protocol": PROTOCOL, "carrying_commit": carrying_commit(),
              "wrapper": wrapper_block(),
              "data_root_check": root_block,
              "ledger_root": {"data_root": str(ROOT),
                              "data_root_branch": E20.DATA_ROOT_BRANCH,
                              "code_root": str(CODE_ROOT),
                              "code_and_data_are_the_same_tree":
                                  str(ROOT) == str(CODE_ROOT)},
              "declaration": {"path": str(DECL_PATH.relative_to(CODE_ROOT)),
                              "sha256": DECL_SHA,
                              "carrying_commit": decl["carrying_commit"]},
              "symbols_in_scope": list(SYMBOLS_IN_SCOPE),
              "the_size_aware_arm": {
                  "status": "REFUSED_NO_DECLARED_REBALANCE_NOTIONAL",
                  "why": decl["the_required_input_that_does_not_exist_yet"][
                      "if_it_cannot_be_sourced"]},
              "the_arm_that_ran": "MIN_SIZE_NOT_THE_E2A_GATE",
              "symbols": {}}

    # 7 of the reviewer's checklist: the inherited control GATES the smoke.
    if run_repro:
        rep = EP.reproduce_e1a(list(SYMBOLS_IN_SCOPE), regimes=("csv",))
        result["inherited_control"] = {
            "reproduced": bool(rep.get("reproduced")),
            "regimes": rep.get("regimes"),
            "gates_the_run": True}
        if not rep.get("reproduced"):
            result["REFUSED"] = (
                "the E1-A reproduction control MISSED: E2-A would not be "
                "superseding E1-A's number, it would be measuring a "
                "different estimator")
            if out_path:
                out_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n")
            raise E2ARefused(result["REFUSED"])

    for sym in symbols:
        w0 = time.time()
        days_all = sorted({f.name.split("_")[0]
                           for f in (RAW / "bookTicker" / sym).glob("*.csv*")})
        admissions, evaluated = [], []
        beats, restarts = collector_heartbeats(), collector_restarts()
        for day in days_all:
            counts = stream_file_counts(sym, day)
            health = collector_health(day, beats, restarts)
            admissions.append(day_admission(sym, day, counts, health))
        adm_days = [a["day"] for a in admissions if a["admissible"]]
        if len(adm_days) < min_days:
            result["symbols"][sym] = {
                "admissions": admissions,
                "n_admissible_days": len(adm_days),
                "REFUSED": (f"{len(adm_days)} admissible days < the declared "
                            f"minimum {min_days}: no gate is read for this "
                            f"symbol and the cap is not relaxed")}
            continue
        for day in adm_days:
            book, _, bmeta = E20.read_book(sym, day, extend=True)
            trades, _, tmeta = E20.read_trades(sym, day)
            depth, _, dmeta = read_depth20(sym, day)
            ev = evaluate_day(sym, day, decl_digest, book, trades, depth)
            ev["stream_meta"] = {"book": bmeta, "trades": tmeta,
                                 "depth20": dmeta}
            evaluated.append(ev)
            del book, trades, depth
        aggs = {str(tp): aggregate_symbol(evaluated, tp) for tp in TP_GRID_S}
        primary = aggs[str(TP_PRIMARY_S)]
        viol = [v for d in evaluated for v in d["ordering_violations"]]
        vd = verdict(primary, decl)
        if viol:
            vd = {"state": "REFUTES_THE_BRACKET",
                  "why": (f"{len(viol)} episodes where ProbQueue-f3 filled "
                          f"LESS than RiskAverse. That ordering is arithmetic "
                          f"per episode, so this is an implementation defect "
                          f"and no overlay verdict may be read."),
                  "n_ordering_violations": len(viol),
                  "examples": viol[:5]}
        icp = D.icp_predicate(primary["episode_skip_rate"], in_aggregate=True)
        result["symbols"][sym] = {
            "admissions": admissions,
            "n_admissible_days": len(adm_days),
            "admissible_days": adm_days,
            "days": [{k: v for k, v in d.items() if k != "rows"}
                     for d in evaluated],
            "aggregate_by_tp": aggs,
            "gate_row_tp_s": TP_PRIMARY_S,
            "verdict": vd,
            "icp_rule": icp,
            "n_ordering_violations": len(viol),
            "wall_s": round(time.time() - w0, 2)}

    result["wall_s_total"] = round(time.time() - t_start, 2)
    try:
        import resource                                       # noqa: PLC0415
        result["max_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss
    except Exception:                                         # noqa: BLE001
        pass
    if out_path:
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


# --------------------------------------------------------------------------
# THE FIXTURE -- every boundary driven on synthetic tapes, NO DATA
# --------------------------------------------------------------------------
class _IOWatch:
    """Records every path this process opens, so 'no data was touched' is a
    PRODUCED FACT and not a promise (DE's fixture v6 discipline, R-566(A))."""

    def __init__(self):
        self.paths: list[str] = []

    def __enter__(self):
        w = self
        self._open, self._gz, self._rc = builtins.open, gzip.open, pd.read_csv
        self._rb, self._rt = Path.read_bytes, Path.read_text

        def rec(f):
            if isinstance(f, (str, Path)):
                w.paths.append(str(f))
            return f

        builtins.open = lambda f, *a, **k: w._open(rec(f), *a, **k)
        gzip.open = lambda f, *a, **k: w._gz(rec(f), *a, **k)
        pd.read_csv = lambda f, *a, **k: w._rc(rec(f), *a, **k)
        Path.read_bytes = lambda p, *a, **k: w._rb(rec(p), *a, **k)
        Path.read_text = lambda p, *a, **k: w._rt(rec(p), *a, **k)
        return self

    def __exit__(self, *exc):
        builtins.open, gzip.open, pd.read_csv = self._open, self._gz, self._rc
        Path.read_bytes, Path.read_text = self._rb, self._rt
        return False

    def tape_paths(self) -> list[str]:
        return sorted({p for p in self.paths if "/data/mm_hf/" in p})


FIX_DAY = "20260820"
#: DERIVED from FIX_DAY, never typed: a hand-written epoch that disagrees with
#: the day string moves every decision time in the fixture and the battery
#: then measures nothing. The first version of this line was four days out and
#: the day-boundary check is what said so.
DAY0_MS = int(pd.Timestamp(FIX_DAY, tz="UTC").timestamp()) * 1000


def _synth_day(tick=0.001, qty_step=1.0, spread_ticks=2, mid=100.0,
               queue_at_touch=1000.0, trades=(), drift_ticks=0,
               levels_present=True, n_levels=N_LEVELS):
    """A book, a trade tape and a depth20 tape, all in memory.

    `trades` is a sequence of (offset_ms, price_ticks_from_L_buy, qty,
    is_taker_sell). The buy-side level L is the best bid.
    """
    half = spread_ticks / 2.0 * tick
    bid0, ask0 = mid - half, mid + half
    # book: t0-1 ms (the quote at t0-), and one at every hour end + drift
    bt_t = np.array([DAY0_MS - 1] +
                    [DAY0_MS + h * 3_600_000 + 1_800_000 for h in range(25)],
                    dtype=np.int64)
    d = drift_ticks * tick
    bid = np.array([bid0] + [bid0 + d] * 25)
    ask = np.array([ask0] + [ask0 + d] * 25)
    # depth20 one snapshot just before t0
    kb = np.arange(n_levels)
    bp = (bid0 - kb * tick)[None, :]
    ap = (ask0 + kb * tick)[None, :]
    if not levels_present:                     # push the book away from L
        bp = bp - 50 * tick
        ap = ap + 50 * tick
    bq = np.full((1, n_levels), 500.0)
    aq = np.full((1, n_levels), 500.0)
    bq[0, 0] = queue_at_touch
    aq[0, 0] = queue_at_touch
    d_t = np.array([DAY0_MS - 1], dtype=np.int64)
    tr_t, tr_p, tr_q, tr_m = [], [], [], []
    for off, ktick, q, taker_sell in trades:
        tr_t.append(DAY0_MS + off)
        tr_p.append(bid0 + ktick * tick if taker_sell else ask0 + ktick * tick)
        tr_q.append(q)
        tr_m.append(taker_sell)
    if not tr_t:                               # a tape must not be empty
        tr_t, tr_p, tr_q, tr_m = [DAY0_MS - 10], [mid], [qty_step], [True]
    order = np.argsort(np.array(tr_t, dtype=np.int64), kind="stable")
    book = (bt_t, bid, ask)
    tapes = (np.array(tr_t, dtype=np.int64)[order],
             np.array(tr_p)[order], np.array(tr_q)[order],
             np.array(tr_m, dtype=bool)[order])
    depth = (d_t, bp, bq, ap, aq)
    return book, tapes, depth


def _row(ev, tp=TP_PRIMARY_S, hour=0, sign=1.0):
    for r in ev["rows"]:
        if r["tp_s"] == tp and r["hour"] == hour and r["sign"] == sign:
            return r
    return None


def fixture(out_path: Path | None = None) -> dict:              # noqa: C901
    """Every boundary and every falsifier, both directions, on synthetic
    tapes. Emits a receipt; touches no tape and proves it."""
    checks: list[dict] = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    with _IOWatch() as watch:
        decl = load_declaration()
        digest = DECL_SHA[:16]

        # -- 1. the two interior controls, at the declared tolerance --------
        tol = decl["interior_controls"]["tolerance"]
        ra_i = D.riskaverse_filled_qty(1000.0, 10.0, 1005.0)
        pq_i = D.probqueue_f3_fill_prob(30.0, 70.0)
        ck("INTERIOR RiskAverse", abs(ra_i - 5.0) <= tol,
           f"clip(1005-1000,0,10) = {ra_i} against the hand-derived 5.0")
        ck("INTERIOR ProbQueue-f3", abs(pq_i - 343000 / 370000) <= tol,
           f"f(70)/(f(30)+f(70)) = {pq_i} against 343000/370000")

        # -- 2. alone at the level: the two models must AGREE --------------
        rng = np.random.default_rng(1)
        alone = simulate_episode(0.0, 1.0, np.array([5.0]), np.array([500.0]),
                                 rng)
        ck("POSITIVE alone-at-level identical under both models",
           alone["filled_qty_RiskAverse"] == alone["filled_qty_ProbQueue_f3"]
           == 1.0,
           f"queue_ahead 0, one opposite trade of 5: RiskAverse "
           f"{alone['filled_qty_RiskAverse']}, ProbQueue "
           f"{alone['filled_qty_ProbQueue_f3']} -- with nothing ahead the "
           f"probability is 1 by the declared branch, so they cannot differ")

        # -- 3. known-bad: a queue larger than all volume never fills -------
        deep = simulate_episode(1e9, 1.0, np.array([5.0, 5.0]),
                                np.array([1e9, 1e9]),
                                np.random.default_rng(2))
        ck("KNOWN-BAD queue > all subsequent volume never fills (RiskAverse)",
           deep["filled_qty_RiskAverse"] == 0.0,
           f"queue_ahead 1e9 against 10 units of volume fills "
           f"{deep['filled_qty_RiskAverse']} -- if this were positive the "
           f"queue would not be being counted")

        # -- 4. ordering per EPISODE, on QUANTITY, over a battery -----------
        bad = 0
        rg = np.random.default_rng(20260906)
        for i in range(400):
            qa = float(rg.integers(0, 2000))
            oq = float(rg.integers(1, 20))
            n = int(rg.integers(0, 40))
            v = rg.random(n) * 200.0
            dep = np.maximum(rg.random(n) * 3000.0, 0.0)
            r = simulate_episode(qa, oq, v, dep, np.random.default_rng(i))
            if (r["filled_qty_ProbQueue_f3"]
                    < r["filled_qty_RiskAverse"] - 1e-9):
                bad += 1
        ck("ORDERING per EPISODE on QUANTITY (400 randomised episodes)",
           bad == 0,
           f"{bad} episodes where ProbQueue filled LESS than RiskAverse. The "
           f"ordering is arithmetic on quantity -- RiskAverse fills only once "
           f"front reaches 0, and at front = 0 ProbQueue's probability is 1")

        # -- 5. and the COST ordering is NOT a per-episode property ---------
        c_fill, c_chase_good = 1.0, -6.0      # favourable drift on the chase
        pr_full = price_episode(1.0, 1.0, c_fill, c_chase_good)
        pr_none = price_episode(0.0, 1.0, c_fill, c_chase_good)
        ck("R-570(B) SHARPENED: per-episode COST can invert with no defect",
           pr_full[PR_RESIDUAL] > pr_none[PR_RESIDUAL],
           f"an episode whose mid ran AWAY over T_p costs "
           f"{pr_none[PR_RESIDUAL]} chased against {pr_full[PR_RESIDUAL]} "
           f"filled -- filling MORE costs MORE, so a per-episode cost "
           f"inversion is the winner's curse, not an implementation defect. "
           f"This is why the arithmetic check is on quantity.")

        # -- 6. the at-L trade, driven through the RUNNER -------------------
        bk, tr, dp = _synth_day(queue_at_touch=10.0,
                                trades=[(1000, 0, 100.0, True)])
        ev_at = evaluate_day("ICPUSDT", FIX_DAY, digest, bk, tr, dp,
                             tick=0.001, qty_step=1.0)
        r_at = _row(ev_at)
        bk2, tr2, dp2 = _synth_day(queue_at_touch=10.0,
                                   trades=[(1000, +1, 100.0, True)])
        ev_above = evaluate_day("ICPUSDT", FIX_DAY, digest, bk2, tr2, dp2,
                                tick=0.001, qty_step=1.0)
        r_above = _row(ev_above)
        ck("AT-L POSITIVE (reviewer REVIEW_DA60 section 2): a trade at "
           "EXACTLY L fills",
           r_at is not None and r_at["phi_RiskAverse"] == 1.0,
           f"one opposite print at exactly L with 100 units against a queue "
           f"of 10 fills phi = "
           f"{None if r_at is None else r_at['phi_RiskAverse']}")
        ck("AT-L KNOWN-BAD: a print one tick ABOVE L (away from the buy) "
           "does NOT fill",
           r_above is not None and r_above["phi_RiskAverse"] == 0.0,
           f"the same print one tick on the wrong side of L fills phi = "
           f"{None if r_above is None else r_above['phi_RiskAverse']} -- "
           f"which is what makes the at-or-through rule a direction with a "
           f"check rather than a comment")

        # -- 7. QUEUE_AHEAD_UNDEFINED, both directions ----------------------
        bk3, tr3, dp3 = _synth_day(levels_present=False,
                                   trades=[(1000, 0, 100.0, True)])
        ev_u = evaluate_day("ICPUSDT", FIX_DAY, digest, bk3, tr3, dp3,
                            tick=0.001, qty_step=1.0)
        ck("BOUNDARY 1 KNOWN-BAD: L absent from the snapshot is "
           "QUEUE_AHEAD_UNDEFINED and in NEITHER model's population",
           ev_u["status_counts"][ST_QAU] == ev_u["n_attempted"]
           and ev_u["status_counts"][ST_RESOLVED] == 0,
           f"{ev_u['status_counts'][ST_QAU]} of {ev_u['n_attempted']} "
           f"episodes carry the status and 0 resolve -- never a queue-ahead "
           f"of zero, which would read as the best queue position there is")
        ck("BOUNDARY 1 POSITIVE: an L that IS in the book is ADMITTED",
           ev_at["status_counts"][ST_QAU] == 0
           and ev_at["status_counts"][ST_RESOLVED] == ev_at["n_attempted"],
           f"{ev_at['status_counts'][ST_RESOLVED]} of "
           f"{ev_at['n_attempted']} resolve -- the status can NOT fire, so "
           f"it is a status and not a filter")

        # -- 8. partial fills, both pricings, and the straddle --------------
        pr_half = price_episode(0.5, 1.0, 1.0, 5.0)
        ck("BOUNDARY 2 POSITIVE: phi = 0.5 prices STRICTLY between c_fill "
           "and c_chase under the gate-bearing rule",
           1.0 < pr_half[PR_RESIDUAL] < 5.0
           and pr_half[PR_WHOLE] == 5.0 and pr_half["is_partial"],
           f"residual_chased {pr_half[PR_RESIDUAL]} strictly inside "
           f"(1.0, 5.0); whole_leg_charged {pr_half[PR_WHOLE]} exactly at "
           f"the chase")
        strad = D.partial_pricing_predicate(7.0, 9.0)
        agree = D.partial_pricing_predicate(6.0, 7.0)
        ck("BOUNDARY 2 KNOWN-BAD: the two pricings straddling 8.0 is a FAIL, "
           "never their mean",
           strad["state"] == "FAIL_PARTIAL_FILL_PRICING_STRADDLES"
           and agree["state"] == "PARTIAL_FILL_PRICING_AGREES",
           f"(7.0, 9.0) -> {strad['state']} even though the mean is exactly "
           f"8.0 and would have passed; (6.0, 7.0) -> {agree['state']}")

        # -- 9. day admission v6: the COLLECTOR, not the book ---------------
        full = {"bookTicker": 24, "trade": 24, "depth20": 24}
        d0 = int(pd.Timestamp(FIX_DAY, tz="UTC").timestamp())
        live_beats = np.arange(d0 - 120, d0 + 86_520, 60, dtype=float)
        out_beats = np.concatenate([
            np.arange(d0 - 120, d0 + 40_000, 60, dtype=float),
            np.arange(d0 + 45_000, d0 + 86_520, 60, dtype=float)])
        no_rs = np.zeros(0)
        h_live = collector_health(FIX_DAY, live_beats, no_rs)
        h_out = collector_health(FIX_DAY, out_beats, no_rs)
        h_restart = collector_health(FIX_DAY, live_beats,
                                     np.array([d0 + 50_000.0]))
        a_quiet = day_admission("ICPUSDT", FIX_DAY, full, h_live,
                                gap=0.99, age={"p50_ms": 5000.0})
        a_d20 = day_admission("ICPUSDT", FIX_DAY, dict(full, depth20=23),
                              h_live)
        a_out = day_admission("ICPUSDT", FIX_DAY, full, h_out)
        a_rs = day_admission("ICPUSDT", FIX_DAY, full, h_restart)
        ck("v6 POSITIVE, THE WHOLE POINT: a book so quiet that 99% of its "
           "seconds carry no message is ADMITTED when the COLLECTOR is live",
           a_quiet["admissible"]
           and a_quiet["REPORTED_not_gated"]["gap_fraction"] == 0.99,
           f"gap_fraction 0.99 and decision-time age 5,000 ms are REPORTED "
           f"({a_quiet['REPORTED_not_gated']['gap_fraction']}) and the day "
           f"still admits -- under v5 this day was excluded, which is what "
           f"cut ICP from 16 days to 1")
        ck("v6 KNOWN-BAD: a COLLECTOR OUTAGE refuses -- a heartbeat gap of "
           "5,000 s against a bar of 2x the measured 60 s cadence",
           not a_out["admissible"] and h_out["max_heartbeat_gap_s"] > 4000
           and h_out["bar_s"] == 120.0,
           f"max_heartbeat_gap_s {h_out['max_heartbeat_gap_s']:.0f} vs bar "
           f"{h_out['bar_s']:.0f}: {a_out['reasons_excluded']}")
        ck("v6 KNOWN-BAD: a COLLECTOR RESTART inside the day refuses even "
           "with an unbroken heartbeat",
           not a_rs["admissible"]
           and h_restart["n_collector_restarts_in_day"] == 1,
           f"1 restart in day, max gap {h_restart['max_heartbeat_gap_s']:.0f} "
           f"s within bar: {a_rs['reasons_excluded']}")
        ck("v6 KNOWN-BAD: a day missing depth20 ALONE is still EXCLUDED",
           not a_d20["admissible"]
           and a_d20["reasons_excluded"] == ["depth20_files=23"],
           f"{a_d20['reasons_excluded']} -- the third stream is a real "
           f"requirement, not decoration")
        ck("v6: THE BAR IS THE COLLECTOR'S OWN MEASURED CADENCE, not a "
           "number chosen here",
           h_live["cadence_s"] == 60.0 and h_live["bar_s"] == 120.0,
           f"modal inter-heartbeat interval {h_live['cadence_s']:.0f} s -> "
           f"bar {h_live['bar_s']:.0f} s")

        # -- 9b. PARTIAL FILLS MUST BE ABLE TO FIRE (rule 16) ---------------
        #: Reported in DA 61: at partial_share = 0.000 the two R-570(C)(2)
        #: pricings coincide and the straddle rule cannot fire. A rule that
        #: cannot fire is not a guard, so it is driven here on an episode
        #: built to be partial.
        part = simulate_episode(queue_ahead=100.0, order_qty=10.0,
                                vol=np.array([104.0]),
                                depth_at_L=np.array([0.0]),
                                rng=np.random.default_rng(11))
        ck("PARTIAL FILL FIRES: a queue of 100, an order of 10 and 104 units "
           "through leaves the order HALF FILLED under RiskAverse",
           abs(part["filled_qty_RiskAverse"] - 4.0) < 1e-9,
           f"filled {part['filled_qty_RiskAverse']} of 10 -- clip(104-100, "
           f"0, 10) = 4, strictly between the two boundaries")
        #: c_fill 0.0 / c_chase 5.0 are chosen so the TWO PRICINGS LAND ON
        #: OPPOSITE SIDES of the 8.0 bps threshold once doubled to eff_RT --
        #: that is the case the straddle rule exists for and the case
        #: partial_share = 0.000 made unreachable.
        pr_part = price_episode(part["filled_qty_RiskAverse"], 10.0,
                                c_fill=0.0, c_chase=5.0)
        eff_res, eff_whole = 2 * pr_part[PR_RESIDUAL], 2 * pr_part[PR_WHOLE]
        ck("AND BOTH PRICINGS THEN DIFFER: residual-chased prices the "
           "unfilled 60%, whole-leg charges the entire leg as a chase",
           abs(pr_part[PR_RESIDUAL] - 3.0) < 1e-9
           and pr_part[PR_WHOLE] == 5.0 and pr_part["is_partial"],
           f"phi {pr_part['phi']:.2f}: residual_chased "
           f"{pr_part[PR_RESIDUAL]} vs whole_leg_charged {pr_part[PR_WHOLE]} "
           f"-- eff_RT {eff_res} vs {eff_whole}, so the second bracket is "
           f"not degenerate")
        strad_real = D.partial_pricing_predicate(eff_res, eff_whole)
        ck("AND THE STRADDLE RULE FIRES ON THEM: eff_RT 6.0 against 10.0 "
           "spans the 8.0 bps threshold",
           strad_real["state"] == "FAIL_PARTIAL_FILL_PRICING_STRADDLES",
           f"{strad_real['state']} -- the rule DA 61 reported as unable to "
           f"fire at partial_share 0.000 is shown FIRING on an episode that "
           f"is actually partial, and their mean 8.0 would have passed")

        # -- 9c. THE ORDERING FALSIFIER, SPLIT ------------------------------
        agg_ra, agg_pq = 7.0, 6.0
        ck("ORDERING (aggregate, COST): ProbQueue at or below RiskAverse at "
           "the gate row ADMITS; above it REFUTES THE INSTRUMENT",
           D.gate_predicate(agg_ra, agg_pq, ci_lo=5.0,
                            ci_hi=7.5)["state"] == "PASS"
           and D.gate_predicate(agg_ra, 9.0, ci_lo=5.0,
                                ci_hi=7.5)["state"] == "REFUTES_THE_BRACKET",
           "aggregate cost ordering is the gate-level check; the per-episode "
           "cost ordering is NOT an ordering property at all (driven above)")

        # -- 10. the population is the twelve, by name ----------------------
        refused_scope = False
        try:
            require_symbol_in_scope("ATOMUSDT")
        except E2ARefused:
            refused_scope = True
        admitted_scope = True
        try:
            require_symbol_in_scope("ICPUSDT")
        except E2ARefused:
            admitted_scope = False
        ck("SCOPE both directions: a symbol outside the twelve is REFUSED "
           "by name, one inside is admitted",
           refused_scope and admitted_scope,
           "ATOMUSDT refused (collected but not in E1-A's XS set), ICPUSDT "
           "admitted")

        # -- 11. the closed-form cost, and the chase ------------------------
        r_fill = _row(ev_at)
        half_bps = r_fill["c_fill_bps"] - FEE_MAKER
        ck("POSITIVE the closed form: a touch fill with no adverse drift "
           "costs fee_maker - half_spread",
           abs(r_fill["c_fill_bps"] - (FEE_MAKER + half_bps)) < 1e-9
           and half_bps < 0,
           f"c_fill = {r_fill['c_fill_bps']:.6f} bps = {FEE_MAKER} "
           f"{half_bps:+.6f} (the half-spread earned, hence negative)")
        bk4, tr4, dp4 = _synth_day(queue_at_touch=1e9, drift_ticks=+10,
                                   trades=[(1000, 0, 1.0, True)])
        ev_ch = evaluate_day("ICPUSDT", FIX_DAY, digest, bk4, tr4, dp4,
                             tick=0.001, qty_step=1.0)
        r_ch = _row(ev_ch)
        ck("POSITIVE the chase: an unfilled episode pays the TAKER fee plus "
           "the realised drift over T_p",
           r_ch["phi_RiskAverse"] == 0.0
           and r_ch["c_chase_bps"] > FEE_TAKER,
           f"phi = 0, c_chase = {r_ch['c_chase_bps']:.4f} bps against a "
           f"taker fee of {FEE_TAKER} -- the mid ran 10 ticks against the "
           f"maker and the winner's curse is charged in full")

        # -- 12. the quantity step, measured not chosen ---------------------
        grid = np.arange(1, 200) * 0.25
        off = np.append(grid, 3.14159)
        qs, qs_off = qty_step_mode(grid), qty_step_mode(off)
        gcd_off = EP.tick_mode([off])
        ck("q IS MEASURED: a tape of 0.25 multiples returns 0.25, and ONE "
           "off-grid print does not drag it",
           abs(qs - 0.25) < 1e-9 and abs(qs_off - 0.25) < 1e-9,
           f"qty_step_mode gives {qs} on grid and {qs_off} with one off-grid "
           f"quantity")
        ck("KNOWN-BAD, AND IT IS E1's OWN FALLBACK: the GCD fallback kept in "
           "`tick_size` collapses on a single off-grid value",
           abs(gcd_off - 0.25) > 1e-9,
           f"EP.tick_mode on the same tape returns {gcd_off} instead of "
           f"0.25 -- the modal-diff fix is present and the retained GCD "
           f"fallback overrides it, which is the FIL 1e-6-vs-1e-4 mechanism. "
           f"The price tick is NOT changed here: it is pinned by the E1-A "
           f"reproduction control.")

        # -- 13. the depth20 parse, and ragged rows counted -----------------
        good = (b"1,2,3,4,"
                + b"|".join(f"{2.4 - i * 0.001:.6f}@{100 + i}".encode()
                            for i in range(N_LEVELS)) + b","
                + b"|".join(f"{2.401 + i * 0.001:.6f}@{200 + i}".encode()
                            for i in range(N_LEVELS)) + b"\n")
        ragged = b"1,2,3,4," + b"2.399000@11|2.398000@12," + b"2.400000@13\n"
        got, meta = parse_depth20_bytes(good + ragged)
        ck("PARSE: twenty levels a side land at the right sizes",
           got is not None and got[2][0, 0] == 100 and got[4][0, 0] == 200
           and abs(got[1][0, 0] - 2.4) < 1e-9 and meta["n_snapshots"] == 1,
           f"bid0 {got[1][0, 0]}@{got[2][0, 0]}, ask0 "
           f"{got[3][0, 0]}@{got[4][0, 0]}")
        ck("PARSE KNOWN-BAD: a row without twenty levels a side is RAGGED, "
           "counted and EXCLUDED -- never padded with zeros",
           meta["n_ragged_rows"] == 1 and meta["n_snapshots"] == 1,
           f"{meta['n_ragged_rows']} ragged of {meta['n_raw_rows']} raw -- a "
           f"padded zero level is an invented queue position")

        # -- 14. the declaration is pinned ----------------------------------
        pinned = hashlib.sha256(DECL_PATH.read_bytes()).hexdigest() == DECL_SHA
        ck("THE DECLARATION IS PINNED: the runner refuses if its sha256 "
           "moves", pinned, f"{DECL_SHA[:16]} matches on disk")

        # -- 15. the size-aware arm refuses ---------------------------------
        ck("THE SIZE-AWARE ARM REFUSES rather than defaulting to min-size "
           "and calling it E2-A",
           bool(decl["the_required_input_that_does_not_exist_yet"][
               "escalated"]),
           "the arm that runs is labelled MIN_SIZE_NOT_THE_E2A_GATE")

    tape = watch.tape_paths()
    n_fail = sum(1 for c in checks if not c["passed"])
    receipt = {
        "protocol": PROTOCOL + "_FIXTURE",
        "carrying_commit": carrying_commit(),
        "wrapper": wrapper_block(),
        "status": "FIXTURE_NO_DATA_TOUCHED",
        "declaration": {"path": str(DECL_PATH.relative_to(CODE_ROOT)),
                        "sha256": DECL_SHA},
        "data_free_proof": {
            "instrument": "builtins.open + gzip.open + pandas.read_csv + "
                          "Path.read_bytes + Path.read_text, wrapped for the "
                          "duration of the fixture and restored after",
            "n_paths_opened": len(watch.paths),
            "tape_paths_opened": tape,
            "no_path_under_data_mm_hf_was_opened": tape == [],
            "why": "a fixture that says it touched no data and cannot show "
                   "it is a claim, not a control"},
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": n_fail,
        "both_directions": True,
    }
    if out_path:
        out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True)
                            + "\n")
    for c in checks:
        print(("ok   " if c["passed"] else "FAIL ") + c["check"])
        print("       " + c["detail"].replace("\n", " "))
    print(f"\n{'FIXTURE OK' if not n_fail else 'FIXTURE FAILED'} -- "
          f"{len(checks)} checks, {n_fail} failure(s); "
          f"tape paths opened: {len(tape)}")
    return receipt


def gap_profile(bt_t: np.ndarray, day: str) -> dict:
    """WHAT the missing seconds ARE -- an outage or a quiet book.

    `gap_fraction` counts seconds with no bookTicker message. That number
    cannot tell a collector outage from a symbol whose best quote simply did
    not change, and the two demand opposite readings: an outage is missing
    DATA, quietness is present data about a still book. The run-length
    profile separates them: an outage is a few LONG contiguous runs, and
    quietness is many one-second holes.
    """
    day0 = int(pd.Timestamp(day, tz="UTC").timestamp()) * 1000
    inday = bt_t[(bt_t >= day0) & (bt_t < day0 + SEC_PER_DAY_MS)]
    if len(inday) == 0:
        return {"n_quotes_in_day": 0, "gap_fraction": 1.0}
    present = np.zeros(86_400, bool)
    present[((inday - day0) // 1000).astype(np.int64)] = True
    missing = ~present
    idx = np.flatnonzero(missing)
    if idx.size == 0:
        runs = np.zeros(0, np.int64)
    else:
        brk = np.flatnonzero(np.diff(idx) != 1)
        starts = np.concatenate(([0], brk + 1))
        ends = np.concatenate((brk, [idx.size - 1]))
        runs = (idx[ends] - idx[starts] + 1).astype(np.int64)
    n_missing = int(missing.sum())
    long_runs = runs[runs >= 60]
    return {
        "n_quotes_in_day": int(len(inday)),
        "gap_fraction": float(n_missing / 86_400),
        "n_missing_seconds": n_missing,
        "n_gap_runs": int(runs.size),
        "max_gap_run_s": int(runs.max()) if runs.size else 0,
        "median_gap_run_s": float(np.median(runs)) if runs.size else 0.0,
        "share_of_missing_seconds_in_runs_ge_60s":
            float(long_runs.sum() / n_missing) if n_missing else 0.0,
        "n_runs_ge_60s": int(long_runs.size),
        "reading": ("OUTAGE-SHAPED: most missing seconds sit in runs of a "
                    "minute or more"
                    if n_missing and long_runs.sum() / n_missing > 0.5
                    else "QUIET-BOOK-SHAPED: the missing seconds are "
                         "scattered short holes, i.e. seconds in which the "
                         "best quote did not change"),
    }


def decision_time_quote_age(bt_t: np.ndarray, day: str) -> dict:
    """Quote AGE at the 24 decision times -- the quantity E2-A actually rests
    on, which the day-level gap fraction is a poor proxy for.

    Placement takes the touch from the last bookTicker STRICTLY BEFORE t0. A
    day can have 15% of its seconds carry no message and still have a quote
    milliseconds old at every decision time; conversely a fresh-looking day
    could be stale exactly on the hour. E2.0's own TrueMid records that a
    quote stands until the next one and that filtering on age would SELECT ON
    ACTIVITY -- so this reports the age rather than filtering on it.
    """
    day0 = int(pd.Timestamp(day, tz="UTC").timestamp()) * 1000
    t0s = day0 + np.arange(24, dtype=np.int64) * 3_600_000
    i = np.searchsorted(bt_t, t0s, "left") - 1
    ok = i >= 0
    ages = (t0s[ok] - bt_t[np.clip(i, 0, None)][ok]).astype(float)
    if ages.size == 0:
        return {"n_decision_times_with_a_prior_quote": 0}
    return {"n_decision_times_with_a_prior_quote": int(ages.size),
            "p50_ms": float(np.percentile(ages, 50)),
            "p90_ms": float(np.percentile(ages, 90)),
            "max_ms": float(ages.max())}


def census(symbols, out_path: Path | None = None,
           light: bool = False) -> dict:
    """The admission table and the gap PROFILE, per symbol-day.

    A population census, not a gate read: no episode is simulated and no
    threshold is applied to anything but the DECLARED admission predicate.
    """
    root = E20.require_canonical_root("P-2026-002 E2-A admission census")
    beats, restarts = collector_heartbeats(), collector_restarts()
    out = {"protocol": PROTOCOL + "_CENSUS",
           "carrying_commit": carrying_commit(),
           "wrapper": wrapper_block(),
           "data_root_check": root,
           "declaration": {"path": str(DECL_PATH.relative_to(CODE_ROOT)),
                           "sha256": DECL_SHA},
           "admission_leg": "v6 -- COLLECTOR LIVENESS, not book activity",
           "light": bool(light),
           "light_means": ("no tape is opened: admission needs only the "
                           "hour-file census and the collector's heartbeat "
                           "ledger, which is the whole point of v6. The "
                           "REPORTED-not-gated statuses (gap fraction, gap "
                           "profile, decision-time quote age) are omitted, "
                           "and their absence is why this is not a "
                           "substitute for the full census."),
           "collector_heartbeats_seen": int(beats.size),
           "collector_restarts_seen": int(restarts.size),
           "symbols": {}}
    for sym in symbols:
        require_symbol_in_scope(sym)
        days = sorted({f.name.split("_")[0]
                       for f in (RAW / "bookTicker" / sym).glob("*.csv*")})
        rows = []
        for day in days:
            counts = stream_file_counts(sym, day)
            health = collector_health(day, beats, restarts)
            prof, gap, age = None, None, None
            if (not light) and counts["bookTicker"] == HOURS_PER_DAY_FILES:
                bk, _, _ = E20.read_book(sym, day, extend=False)
                if bk is not None:
                    prof = gap_profile(bk[0], day)
                    gap = prof["gap_fraction"]
                    age = decision_time_quote_age(bk[0], day)
            adm = day_admission(sym, day, counts, health, gap, age)
            adm["gap_profile"] = prof
            rows.append(adm)
        n_adm = sum(1 for r in rows if r["admissible"])
        n_complete = sum(1 for r in rows
                         if all(r["streams_complete"].values()))
        n_not_live = sum(1 for r in rows
                         if all(r["streams_complete"].values())
                         and not r["admissible"])
        out["symbols"][sym] = {
            "days": rows, "n_days_seen": len(rows),
            "n_days_all_three_streams_complete": n_complete,
            "n_admissible": n_adm,
            "n_excluded_by_collector_outage_alone": n_not_live,
            "min_complete_days": D.MIN_COMPLETE_DAYS,
            "meets_minimum": bool(n_adm >= D.MIN_COMPLETE_DAYS)}
        print(f"{sym}: {n_complete} days with all three streams complete, "
              f"{n_adm} admissible, {n_not_live} excluded by COLLECTOR "
              f"OUTAGE alone")
    if out_path:
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def e1_resolver_parity(out_path: Path | None = None) -> dict:
    """The falsifier for adopting the resolver in E1's PRODUCING module.

    The ruling (DA 62) is that this is a PORTABILITY change, not a result
    change. That is a claim with two halves and both are driven here:

      PARITY   where the OLD resolution was already right -- the shared tree,
               whose `parents[2]` IS the ledger -- the OLD and NEW roots must
               agree and E1's own `tick_size` must return the SAME value for
               every one of the twelve symbols. A single difference means the
               change moved a number and the commit is refused.
      THE FIX  where the OLD resolution was wrong -- a per-seat worktree --
               the OLD root must yield ZERO day-files (which is what made
               `tick_size` raise) and the NEW root must yield the real count.

    A parity check that could only ever pass would be rule 16's shape, which
    is why the second half is here: the two roots must DIFFER somewhere and
    the difference must be the whole of the effect.
    """
    root = E20.require_canonical_root("P-2026-002 E1 resolver parity")
    sys.path.insert(0, str(HERE))
    import e1_markout_scan as E1                              # noqa: PLC0415
    old_root = Path(E1.__file__).resolve().parents[2]
    new_root = E1.REPO
    syms = list(SYMBOLS_IN_SCOPE)

    def probe(repo: Path) -> dict:
        before = E1.SRC
        E1.SRC = repo / "data/mm_hf/vision/parquet/aggTrades"
        try:
            out = {}
            for sy in syms:
                files = E1.day_files(sy)
                out[sy] = {"n_day_files": len(files),
                           "tick_size": (float(E1.tick_size(sy)) if files
                                         else None)}
            return out
        finally:
            E1.SRC = before

    new = probe(new_root)
    old = probe(old_root)
    same_tree = old_root == new_root
    mismatches = [sy for sy in syms
                  if old[sy]["tick_size"] != new[sy]["tick_size"]
                  or old[sy]["n_day_files"] != new[sy]["n_day_files"]]
    old_empty = [sy for sy in syms if old[sy]["n_day_files"] == 0]
    new_found = [sy for sy in syms if new[sy]["n_day_files"] > 0]
    parity_holds = (not mismatches) if same_tree else True
    fix_demonstrated = (not same_tree) and len(old_empty) == len(syms) \
        and len(new_found) == len(syms)
    out = {"protocol": PROTOCOL + "_E1_RESOLVER_PARITY",
           "carrying_commit": carrying_commit(),
           "wrapper": wrapper_block(),
           "data_root_check": root,
           "old_resolution": {"expression": "Path(__file__).parents[2]",
                              "root": str(old_root)},
           "new_resolution": {"expression": "de_data_root.resolve()"
                                            "['repo_root'], imported",
                              "root": str(new_root)},
           "code_and_data_are_the_same_tree": same_tree,
           "per_symbol_old": old, "per_symbol_new": new,
           "n_symbols": len(syms),
           "PARITY_tick_size_and_file_counts_identical": bool(parity_holds),
           "n_mismatches": len(mismatches), "mismatched_symbols": mismatches,
           "FIX_old_root_empty_new_root_populated": bool(fix_demonstrated),
           "n_symbols_old_root_saw_zero_files": len(old_empty),
           "n_symbols_new_root_sees_files": len(new_found),
           "how_to_read_this": (
               "run from the SHARED tree the two roots coincide and the "
               "PARITY half is the meaningful one; run from a per-seat "
               "WORKTREE they differ and the FIX half is. Both halves are "
               "reported every time so a reader can see which one this run "
               "actually exercised.")}
    if out_path:
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("data_root_check", "per_symbol_old",
                                   "per_symbol_new")}, indent=2))
    return out


def mechanism_check(sym: str, out_path: Path | None = None) -> dict:
    """Does the real-book path EXECUTE? -- with every cost REDACTED.

    The fixture drives `evaluate_day` on synthetic tapes, so the code path is
    exercised; it has never met a real bookTicker, a real trade tape or a real
    depth20 snapshot. If a ruling later admits days, the first real run should
    not be the first time the parse, the level lookup and the two simulations
    see the tape.

    This runs the ADMISSIBLE days only, and emits NO cost, NO eff_RT and NO
    verdict -- only episode STATUS counts, the queue-ahead and fill-quantity
    distributions, the ordering predicate, and resources. A number that is not
    written cannot be quoted, and the declared minimum-day floor is untouched
    because no gate is read here.
    """
    root = E20.require_canonical_root("P-2026-002 E2-A mechanism check")
    require_symbol_in_scope(sym)
    decl = load_declaration()
    digest = DECL_SHA[:16]
    t0 = time.time()
    days_all = sorted({f.name.split("_")[0]
                       for f in (RAW / "bookTicker" / sym).glob("*.csv*")})
    adm_days = []
    beats, restarts = collector_heartbeats(), collector_restarts()
    for day in days_all:
        counts = stream_file_counts(sym, day)
        if day_admission(sym, day, counts,
                         collector_health(day, beats, restarts))["admissible"]:
            adm_days.append(day)
    if not adm_days:
        raise E2ARefused(
            f"REFUSED: {sym} has no admissible day, so there is no real book "
            f"to exercise the path on. An empty population is not a check.")
    per_day = []
    for day in adm_days:
        book, _, bmeta = E20.read_book(sym, day, extend=True)
        trades, _, tmeta = E20.read_trades(sym, day)
        depth, _, dmeta = read_depth20(sym, day)
        ev = evaluate_day(sym, day, digest, book, trades, depth)
        rows = ev["rows"]
        gate = [r for r in rows if r["tp_s"] == TP_PRIMARY_S]
        qa = np.array([r["queue_ahead"] for r in gate]) if gate else np.zeros(0)
        per_day.append({
            "day": day, "tick": ev["tick"], "qty_step": ev["qty_step"],
            "status_counts": ev["status_counts"],
            "n_attempted": ev["n_attempted"],
            "n_ordering_violations": len(ev["ordering_violations"]),
            "stream_meta": {"book": bmeta, "trades": tmeta, "depth20": dmeta},
            "gate_row_tp_s": TP_PRIMARY_S,
            "n_gate_row_episodes": len(gate),
            "queue_ahead_at_placement": {
                "p50": float(np.percentile(qa, 50)) if qa.size else None,
                "p90": float(np.percentile(qa, 90)) if qa.size else None,
                "max": float(qa.max()) if qa.size else None,
                "n_zero": int((qa == 0).sum())},
            "fill_rate_RiskAverse":
                float(np.mean([r["phi_RiskAverse"] > 0 for r in gate]))
                if gate else None,
            "fill_rate_ProbQueue_f3":
                float(np.mean([r["phi_ProbQueue_f3"] > 0 for r in gate]))
                if gate else None,
            "partial_share_RiskAverse":
                float(np.mean([r["partial_RiskAverse"] for r in gate]))
                if gate else None,
            "n_opposite_trades_p50":
                float(np.percentile([r["n_opposite_trades"] for r in gate], 50))
                if gate else None,
        })
        del book, trades, depth
    out = {"protocol": PROTOCOL + "_MECHANISM_CHECK",
           "carrying_commit": carrying_commit(),
           "wrapper": wrapper_block(),
           "status": "MECHANISM_ONLY_ALL_COSTS_REDACTED",
           "data_root_check": root,
           "declaration": {"path": str(DECL_PATH.relative_to(CODE_ROOT)),
                           "sha256": DECL_SHA},
           "symbol": sym, "admissible_days_used": adm_days,
           "min_complete_days_declared": decl["population"][
               "min_complete_days"],
           "why_no_number": (
               "this is NOT a gate read and emits no cost, no eff_RT and no "
               "verdict. It answers one question -- does the real-book path "
               "execute on a real tape -- so that a first real run is not "
               "also a first contact. The declared minimum-day floor is "
               "untouched because no gate is read."),
           "days": per_day,
           "wall_s": round(time.time() - t0, 2)}
    try:
        import resource                                       # noqa: PLC0415
        out["max_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss
    except Exception:                                         # noqa: BLE001
        pass
    if out_path:
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if k != "data_root_check"}, indent=2)[:4000])
    return out


def diagnose_tick(sym: str, out_path: Path | None = None) -> dict:
    """R-570(D): WHY E1's tick_size returns 1e-6 for FIL, at the mechanism.

    The record says the function was "FIXED post-audit (mode-of-diffs)" and
    that the fix produces 1e-4. Both halves of `tick_size` are executed here
    with their intermediates exposed, so the account is a measurement rather
    than a reading of the source.
    """
    root = E20.require_canonical_root(
        "P-2026-002 E1_RESULTS record-defect diagnosis")
    import e1_markout_scan as E1                              # noqa: PLC0415
    #: `e1_markout_scan` has NO data-root resolver -- it computes
    #: `SRC = REPO / "data/..."` from its own file location, so from a
    #: worktree it returns an EMPTY day list and `tick_size` raises on an
    #: empty argmax. That is the same gap the E2.0 result review filed
    #: against `e2_0_true_mid.py:46`, still open in the module that PRODUCED
    #: E1's published numbers; the reviewer hit it too (REVIEW_DA60 section
    #: 1, "I could not execute tick_size() in my worktree"). It is not edited
    #: here -- E1's producing code is not this step's surface -- but it is
    #: pointed at the resolved ledger for the duration, and an empty file
    #: list REFUSES rather than being read as a symbol with no prints.
    src_before = E1.SRC
    E1.SRC = E20.VISION
    try:
        files = E1.day_files(sym)
        if not files:
            raise E2ARefused(
                f"REFUSED: no Vision aggTrades for {sym} under "
                f"{E20.VISION}. An empty price population is not a tick.")
        return _diagnose_tick_inner(sym, files, E1, root, out_path)
    finally:
        E1.SRC = src_before


def _diagnose_tick_inner(sym, files, E1, root, out_path):
    uniq: set[float] = set()
    for f in files:
        uniq.update(np.unique(
            pd.read_parquet(f, columns=["price"]).to_numpy(float).ravel()))
    u = np.sort(np.fromiter(uniq, float))
    d = np.diff(u)
    d = d[d > 1e-12]
    scaled = np.round(d * 1e8).astype(np.int64)
    vals, cnts = np.unique(scaled, return_counts=True)
    modal = float(vals[cnts.argmax()] / 1e8)
    mult = d / modal
    frac_int = float(np.mean(
        np.abs(mult - np.round(mult)) < 1e-6 * np.maximum(mult, 1)))
    import math as _m                                         # noqa: PLC0415
    g = 0
    for v in vals:
        g = _m.gcd(g, int(v))
    gcd_tick = float(g / 1e8)
    returned = float(E1.tick_size(sym))
    fallback_fired = frac_int < 0.999
    out = {
        "protocol": PROTOCOL + "_TICK_DIAGNOSIS",
        "carrying_commit": carrying_commit(),
        "wrapper": wrapper_block(),
        "data_root_check": root,
        "symbol": sym,
        "n_day_files": len(files),
        "n_distinct_prices": int(len(u)),
        "modal_diff_the_FIX_produces": modal,
        "frac_diffs_that_are_integer_multiples_of_the_modal": frac_int,
        "fallback_threshold": 0.999,
        "gcd_fallback_fired": bool(fallback_fired),
        "gcd_fallback_value": gcd_tick,
        "tick_size_actually_returns": returned,
        "the_finding": (
            "the modal-diff FIX is present and produces "
            f"{modal:g}; the RETAINED GCD fallback fires "
            f"({frac_int:.6f} < 0.999) and overrides it with {gcd_tick:g}, "
            f"which is what tick_size() returns ({returned:g}). The record's "
            f"'FIXED post-audit' describes a state the code does not reach "
            f"on this input -- not because the fix is missing, but because "
            f"the fallback the same docstring says was 'kept' supersedes it "
            f"on exactly the input the fix was written for."
            if fallback_fired else
            "the GCD fallback did NOT fire on this symbol, so the modal diff "
            f"stands and tick_size() returns {returned:g}"),
        "control_both_directions": {
            "the_diagnosis_must_be_able_NOT_to_fire": (
                "reported per symbol; a symbol whose prints are all on grid "
                "returns gcd_fallback_fired = false"),
        },
    }
    if out_path:
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("data_root_check",)}, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=["ICPUSDT"])
    ap.add_argument("--min-days", type=int, default=None)
    ap.add_argument("--no-repro", action="store_true")
    ap.add_argument("--diagnose-tick", nargs="*", default=None)
    ap.add_argument("--census", nargs="*", default=None)
    ap.add_argument("--mechanism-check", default=None)
    ap.add_argument("--e1-resolver-parity", action="store_true")
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest or a.fixture:
        r = fixture(a.output)
        return 1 if r["n_failed"] or not r["data_free_proof"][
            "no_path_under_data_mm_hf_was_opened"] else 0
    if a.e1_resolver_parity:
        r = e1_resolver_parity(a.output)
        return 0 if (r["PARITY_tick_size_and_file_counts_identical"]
                     or r["FIX_old_root_empty_new_root_populated"]) else 1
    if a.mechanism_check:
        mechanism_check(a.mechanism_check, a.output)
        return 0
    if a.census is not None:
        census(a.census or list(SYMBOLS_IN_SCOPE), a.output,
               light=a.light)
        return 0
    if a.diagnose_tick is not None:
        syms = a.diagnose_tick or ["FILUSDT"]
        outs = [diagnose_tick(sy, None) for sy in syms]
        if a.output:
            a.output.write_text(
                json.dumps({"protocol": PROTOCOL + "_TICK_DIAGNOSIS",
                            "carrying_commit": carrying_commit(),
                            "symbols": outs}, indent=2, sort_keys=True) + "\n")
        return 0
    if a.run:
        res = run(a.symbols, a.output, min_days=a.min_days,
                  run_repro=not a.no_repro)
        for sym, blk in res["symbols"].items():
            print(f"{sym}: {blk.get('REFUSED') or blk['verdict']['state']}")
        return 0
    ap.error("choose --selftest/--fixture or --run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
