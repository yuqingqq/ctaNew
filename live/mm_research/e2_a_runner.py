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
`declarations/p002_e2_a_declaration_v8.json` and this module REFUSES if its
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
import tempfile
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
DECL_PATH = HERE / "declarations" / "p002_e2_a_declaration_v8.json"
DECL_SHA = "929039c9fdc35d5946de2ed0b535eebc3a08307e74780651f0a6d5dbb02d3a36"

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

#: R-584 (USER ruling, 2026-09-06): SCOPE IS BTC FOR NOW. The smoke runs on
#: BTCUSDT and the gate, when 14 post-boundary days exist, is read on BTC.
#: The twelve-symbol census stays as CONTEXT; the thin-name (ICP) cell is
#: DEFERRED, not resolved. Any other symbol may still be executed as a
#: diagnostic, and is LABELLED as not the gate rather than quietly counted.
GATE_SYMBOL = D.GATE_SYMBOL

#: R-584. BTC's bookTicker is the 8.6 GB input the census queued by size --
#: 621 MB gz and 60 M rows on a single day, measured. The day read STREAMS
#: per hour-file and the run REFUSES if a day exceeds the cap. The cap is
#: NEVER raised and the population is NEVER made smaller to fit it.
#: WHERE THE CAP COMES FROM: the rule-20 wrapper's own MemoryMax is 8 GiB and
#: a cgroup kill is SILENT -- it leaves nothing written. The guard must fire
#: while there is still room to write the refusal, so the bar is 75% of the
#: wrapper's cap. It is not tuned on BTC.
DAY_RSS_CAP_GIB = D.DAY_RSS_CAP_GIB


def rss_now_gib() -> float:
    """Current RSS, which FALLS when a day is released -- unlike the
    getrusage high-water mark, which never does."""
    try:
        with builtins.open("/proc/self/statm", "rb") as fh:
            return int(fh.read().split()[1]) * 4096 / 1073741824
    except Exception:                                         # noqa: BLE001
        return float("nan")


def rss_peak_gib() -> float:
    import resource                                           # noqa: PLC0415
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


def memory_guard(where: str, day: str, cap: float | None = None) -> dict:
    """R-584. REFUSE rather than raise the cap or shrink the population.

    `cap` is injectable ONLY so the falsifier can drive the refusing
    direction; every production call takes the declared cap.
    """
    cap_gib = DAY_RSS_CAP_GIB if cap is None else float(cap)
    now, peak = rss_now_gib(), rss_peak_gib()
    obs = {"where": where, "day": day,
           "rss_now_gib": round(now, 3), "rss_peak_gib": round(peak, 3),
           "cap_gib": cap_gib}
    if now > cap_gib:
        raise E2ARefused(
            f"MEMORY CAP: {now:.2f} GiB resident at {where} on {day} exceeds "
            f"the declared {cap_gib} GiB. The cap is NOT raised and "
            f"the population is NOT reduced -- the day is refused and said "
            f"so.")
    return obs


# --------------------------------------------------------------------------
# the book, two ways: whole (fixtures, small symbols) and STREAMED (R-584)
# --------------------------------------------------------------------------
def episode_grid(day: str):
    """The decision times and their T_p ends. ONE definition, so the
    streamed reader and `evaluate_day` cannot drift apart."""
    day0 = int(pd.Timestamp(day, tz="UTC").timestamp()) * 1000
    lefts, rights = set(), set()
    for tp in TP_GRID_S:
        for hh in range(24 if tp * 1000 <= 3_600_000 else 23):
            t0 = day0 + hh * 3_600_000
            lefts.add(t0)
            rights.add(t0 + tp * 1000)
    return day0, np.array(sorted(lefts)), np.array(sorted(rights))


class BookAccess:
    """The three things `evaluate_day` asks a book for."""

    def before(self, t):                                      # noqa: D102
        raise NotImplementedError

    def at_or_before(self, t):                                # noqa: D102
        raise NotImplementedError


class FullBook(BookAccess):
    """The whole day in memory -- unchanged behaviour, used by the fixture
    and by any symbol small enough not to need streaming."""

    def __init__(self, bt_t, bid, ask):
        self.t, self.bid, self.ask = bt_t, bid, ask
        self.t_max = int(bt_t[-1]) if len(bt_t) else None
        self.n_rows = int(len(bt_t))
        self.streamed = False

    def before(self, t):
        i = int(np.searchsorted(self.t, t, "left")) - 1
        return None if i < 0 else (int(self.t[i]), float(self.bid[i]),
                                   float(self.ask[i]))

    def at_or_before(self, t):
        i = int(np.searchsorted(self.t, t, "right")) - 1
        return None if i < 0 else (int(self.t[i]), float(self.bid[i]),
                                   float(self.ask[i]))


class StreamedBook(BookAccess):
    """R-584. One hour-file at a time, keeping ONLY the quotes the episode
    grid asks for.

    The whole-day read is 60 M rows on BTC; the grid needs 24 decision-time
    quotes and 72 T_p-end quotes. Holding the rest is the entire memory
    problem, and it buys nothing.
    """

    def __init__(self, lefts, rights):
        self._l = {int(t): None for t in lefts}
        self._r = {int(t): None for t in rights}
        self._lk = np.array(sorted(self._l))
        self._rk = np.array(sorted(self._r))
        self.t_max = None
        self.n_rows = 0
        self.n_bad_quotes = 0
        self.n_unparsed = 0
        self.n_files = 0
        self.peak_file_rss_gib = 0.0
        self.streamed = True

    def ingest(self, t, bid, ask):
        if len(t) == 0:
            return
        o = np.argsort(t, kind="stable")
        t, bid, ask = t[o], bid[o], ask[o]
        self.n_rows += int(len(t))
        tm = int(t[-1])
        self.t_max = tm if self.t_max is None else max(self.t_max, tm)
        for keys, store, side in ((self._lk, self._l, "left"),
                                  (self._rk, self._r, "right")):
            if not len(keys):
                continue
            idx = np.searchsorted(t, keys, side) - 1
            ok = idx >= 0
            for k, i in zip(keys[ok], idx[ok]):
                cand_t = int(t[i])
                cur = store[int(k)]
                if cur is None or cand_t > cur[0]:
                    store[int(k)] = (cand_t, float(bid[i]), float(ask[i]))

    def _get(self, store, t, which):
        k = int(t)
        if k not in store:
            #: a target the stream was never built for would silently read
            #: as "no quote", i.e. a wrong EXCLUSION. Refuse instead.
            raise E2ARefused(
                f"streamed book asked for a {which} quote at {k}, which is "
                f"not on the episode grid it was built from")
        return store[k]

    def before(self, t):
        return self._get(self._l, t, "before")

    def at_or_before(self, t):
        return self._get(self._r, t, "at-or-before")


def stream_book(sym: str, day: str, extend: bool = True) -> StreamedBook:
    """R-584: read bookTicker hour-file by hour-file, keep the grid only."""
    _, lefts, rights = episode_grid(day)
    sb = StreamedBook(lefts, rights)
    files = E20._hour_files("bookTicker", sym, day)
    if extend:
        nx = E20._next_hour_file("bookTicker", sym, day)
        if nx is not None:
            files = files + [nx]
    for f in files:
        df = pd.read_csv(f, header=None, usecols=[2, 4, 6],
                         names=["T", "bid", "ask"],
                         dtype={2: "int64", 4: "float64", 6: "float64"},
                         on_bad_lines="skip")
        t = df["T"].to_numpy()
        bid, ask = df["bid"].to_numpy(), df["ask"].to_numpy()
        del df
        #: A QUOTE MUST BE A QUOTE -- E2.0's rule, applied per file so the
        #: streamed and the whole-day reads cannot disagree.
        good = (bid > 0) & (ask > 0) & (ask >= bid)
        sb.n_bad_quotes += int((~good).sum())
        sb.ingest(t[good], bid[good], ask[good])
        sb.n_files += 1
        sb.peak_file_rss_gib = max(sb.peak_file_rss_gib, rss_now_gib())
        del t, bid, ask, good
    return sb




#: Episode statuses. Exclusions are STATUSES, never silent drops (rule 4).
ST_RESOLVED = "RESOLVED"
ST_NO_QUOTE = "NO_QUOTE_BEFORE_T0"
ST_QAU = "QUEUE_AHEAD_UNDEFINED"
ST_NO_BOOK_TP = "NO_BOOK_AT_TP"
STATUSES = (ST_RESOLVED, ST_NO_QUOTE, ST_QAU, ST_NO_BOOK_TP)

#: v7 LABELS. Distinct from STATUSES on purpose: a status is EXCLUSIVE and
#: says why an episode produced no row; a label describes a RESOLVED episode
#: and never removes it. Filtering on either of these would select on
#: activity, which is the defect v6 took out of admission.
LB_MARGINAL = "ORDERING_NOT_TESTABLE_MARGINAL"
LB_STALE = "QUOTE_STALE_AT_DECISION"
LABELS = (LB_MARGINAL, LB_STALE)

#: v7 day-level status: below three distinct trade quantities the modal
#: positive diff is an echo of the sample, not an estimate of a step.
ST_QTY_UNDET = "QTY_STEP_UNDERDETERMINED"

#: v7 era leg (CLAUDE.md rule 5), carried from the declaration so the number
#: has ONE home.
ERA_BOUNDARY_RECV_NS = D.ERA_BOUNDARY_RECV_NS
ERA_BOUNDARY_UTC = D.ERA_BOUNDARY_UTC
STALENESS_BAR_MS = D.STALENESS_BAR_MS
MIN_DISTINCT_QUANTITIES = D.MIN_DISTINCT_QUANTITIES

#: v7 SEALED SMOKE. Every key whose value is an ECONOMIC quantity. The open
#: receipt is scanned for these names recursively and REFUSES if one appears,
#: so "sealed" is a checked property of the artifact rather than a promise
#: about which branch was taken.
#: `_bps` and `staleness` are here because DA 63's FIRST marker list MISSED
#: `staleness_sensitivity.reading_A_all_episodes_bps` -- an eff_RT in bps
#: under a name none of the other markers matched. The leak scan uses this
#: same list, so it would have reported ZERO leaks beside two gate numbers
#: sitting in the open receipt: a control blind in exactly the place it was
#: guarding. Caught by enumerating the emitted keys against the list rather
#: than trusting the list, and BEFORE any sealed run emitted.
SEALED_KEY_MARKERS = ("eff_rt", "cost_", "c_fill", "c_chase", "ci_lo",
                      "ci_hi", "verdict", "gate", "phi_", "fill_rate",
                      "mean_phi", "aggregate_by_tp", "partial_pricing",
                      "_bps", "staleness")

#: the list AS IT WAS when it missed the leak, kept so the regression control
#: can show the miss rather than assert it.
SEALED_KEY_MARKERS_PRE_FIX = ("eff_rt", "cost_", "c_fill", "c_chase", "ci_lo",
                              "ci_hi", "verdict", "gate", "phi_", "fill_rate",
                              "mean_phi", "aggregate_by_tp",
                              "partial_pricing")

#: The two partial-fill pricings, R-570(C)(2).
PR_RESIDUAL = "residual_chased"          # GATE-BEARING
PR_WHOLE = "whole_leg_charged"           # pessimistic bracket
PRICINGS = (PR_RESIDUAL, PR_WHOLE)
MODELS = ("RiskAverse", "ProbQueue_f3")


class E2ARefused(RuntimeError):
    """The run cannot proceed under the declaration."""


class LaunchCaptureRefused(E2ARefused):
    """The code that ran is not the code the receipt would name."""


# ---------------------------------------------------------- RULE 22 / R-605
#: THE LAUNCH CAPTURE. Rule 22 as amended: a runner and every heavy producer
#: capture AT IMPORT the digest of every module of their import closure
#: under `live/`, plus the worktree's HEAD and whether it was dirty, and
#: REFUSE THE EMIT BY NAME if any of it moved. DA 77's own sweep found this
#: seat's two runners lacking it -- and found that this seat's binding map
#: had exempted them, which is worse than the gap.
#:
#: R-235: DE's `source_identity_at_launch` / `assert_source_unchanged` were
#: read AS A DOCUMENT and re-implemented here. Nothing of DE's is imported.
#: The property is not "the file I am is unchanged" -- that is one module of
#: many; it is "every module that RAN is still the bytes that ran".
LIVE_DIR = str(Path(__file__).resolve().parents[1])
LAUNCH_TIME_UTC = datetime.datetime.now(
    datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
LAUNCH_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).resolve().read_bytes()).hexdigest()
#: path -> digest OF THE BYTES SEEN WHEN THE MODULE FIRST ENTERED THIS RUN.
#: Never a second read: an import can return a module already in
#: `sys.modules` whose file has since moved, and the second read would
#: record the mover's bytes as the runner's.
LAUNCH_CLOSURE: dict = {}
LAUNCH_CAPTURE_POINTS: list = []


def _digest_module(mod) -> None:
    f = getattr(mod, "__file__", None)
    if not f:
        return
    try:
        p = Path(f).resolve()
    except OSError:
        return
    if not str(p).startswith(LIVE_DIR) or str(p) in LAUNCH_CLOSURE:
        return
    try:
        LAUNCH_CLOSURE[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        LAUNCH_CLOSURE[str(p)] = None


def capture_closure(where: str = "module import") -> int:
    """Digest every `live/` module loaded so far. Callable again AFTER a
    lazy import, because a module imported inside a function was not in
    `sys.modules` at import time and would otherwise be outside the
    closure -- the exact gap REV 53 section 1.1 found in DE's."""
    before = len(LAUNCH_CLOSURE)
    for m in list(sys.modules.values()):
        _digest_module(m)
    LAUNCH_CAPTURE_POINTS.append(
        {"where": where, "n_modules_after": len(LAUNCH_CLOSURE),
         "n_added": len(LAUNCH_CLOSURE) - before})
    return len(LAUNCH_CLOSURE)


def _porcelain(stdout: str) -> dict:
    """The programme's ONE porcelain parser (REV 64), imported."""
    _pm = str(Path(__file__).resolve().parents[1] / "pm_research")
    if _pm not in sys.path:
        sys.path.insert(0, _pm)
    import da_root as _R                                       # noqa: PLC0415
    return _R.parse_porcelain(stdout)


def _is_the_ledger_symlink(root: str, rel: str) -> bool:
    """Is `<root>/<rel>` R-553's symlink to the canonical data root?

    Three facts, all about the object: it is a SYMLINK, and it RESOLVES to
    the canonical data root, which is read from DE's module rather than
    typed here (R-235). The caller has already established that git calls
    it untracked."""
    try:
        p = Path(root) / rel.rstrip("/")
        if not p.is_symlink():
            return False
        #: `da_root` lives in the sibling package; this module is in
        #: live/mm_research. The canonical constant is still READ from DE's
        #: module through the one predicate rather than typed here (R-235).
        _pm = str(Path(__file__).resolve().parents[1] / "pm_research")
        if _pm not in sys.path:
            sys.path.insert(0, _pm)
        import da_root as _R                                   # noqa: PLC0415
        canon = (_R.canonical_from_DEs_source() or {}).get("data")
        if not canon:
            return False
        return str(p.resolve()) == str(Path(canon).resolve())
    except OSError:
        return False


def _head_state() -> dict:
    """The worktree's HEAD and whether it was dirty, AT IMPORT."""
    root = str(Path(__file__).resolve().parents[2])

    def _g(*a):
        try:
            r = subprocess.run(["git", "-C", root, *a], capture_output=True,
                               text=True, timeout=60)
        except Exception:                                     # noqa: BLE001
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    #: RAW, NEVER STRIPPED (REV 64). `_g` strips, and a stripped block
    #: loses the LEADING SPACE of its FIRST line -- which is the read half
    #: of the porcelain defect. The status call bypasses it.
    try:
        _r = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                            capture_output=True, text=True, timeout=60)
        st = _r.stdout if _r.returncode == 0 else None
    except Exception:                                         # noqa: BLE001
        st = None
    if st is None:
        return {"worktree": root, "head": _g("rev-parse", "HEAD"),
                "dirty": None, "dirty_paths": []}
    #: THE LEDGER SYMLINK IS NOT DIRT, AND IT IS EXEMPTED AS A PROPERTY
    #: (DE 94). A seat worktree's `data/` is R-553's symlink to the ledger:
    #: it shows as `?? data`, and a name-matched exemption would let any
    #: file called `data` through. The predicate is UNTRACKED **and** a
    #: SYMLINK **and** resolving to the canonical data root -- three facts
    #: about the object, none about its name.
    parsed = _porcelain(st)
    exempt, dirt = [], []
    for row in parsed["rows"]:
        if row["untracked"] and _is_the_ledger_symlink(root, row["path"]):
            exempt.append(row["path"])
        else:
            dirt.append(row["path"])
    #: a line the parser could not read is DIRT, not silence.
    dirt += parsed["malformed"]
    return {"worktree": root, "head": _g("rev-parse", "HEAD"),
            "dirty": bool(dirt),
            "dirty_paths": dirt[:20],
            "exempt_ledger_symlinks": exempt,
            "why_exempt": (
                "untracked AND a symlink AND resolving to the canonical "
                "data root -- a property of the object, never its name")}


def closure_drift(closure: dict | None = None) -> list:
    """Every module of the closure whose FILE no longer holds the bytes it
    held when it entered this run. `closure` is injectable ONLY so the
    falsifier can drive both directions on copies; every production call
    takes the launch capture."""
    src = LAUNCH_CLOSURE if closure is None else closure
    out = []
    for path, at_launch in sorted(src.items()):
        try:
            now = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            now = None
        if now != at_launch:
            out.append({"module": Path(path).name, "path": path,
                        "at_launch": at_launch, "now": now,
                        "gone": now is None})
    return out


def source_identity_at_launch() -> dict:
    """THE BYTES THAT RAN -- all of them -- and whether they still hold."""
    me = Path(__file__).resolve()
    try:
        now = hashlib.sha256(me.read_bytes()).hexdigest()
    except OSError:
        now = None
    drift = closure_drift()
    head_now = _head_state()
    return {
        "producing_code": me.name,
        "producing_code_sha256": LAUNCH_SOURCE_SHA256,
        "digest_taken_at": "MODULE IMPORT, before any work",
        "launch_time_utc": LAUNCH_TIME_UTC,
        "on_disk_sha256_at_emit": now,
        "source_unchanged_during_the_run": now == LAUNCH_SOURCE_SHA256,
        "import_closure": {
            "n_modules": len(LAUNCH_CLOSURE),
            "root": LIVE_DIR,
            #: A LIST OF [name, digest] PAIRS, not a dict keyed by
            #: filename: a module name must never sit where a key-scanning
            #: net looks for economic keys (this programme owns a module
            #: called `e1_markout_scan.py`). The shape removes the
            #: interaction instead of exempting the block from the scan.
            "modules": [[Path(k).name, v]
                        for k, v in sorted(LAUNCH_CLOSURE.items())],
            "digested": ("from the bytes each module held when it FIRST "
                         "entered this run -- never re-read"),
            "capture_points": LAUNCH_CAPTURE_POINTS,
        },
        "closure_drift": drift,
        "closure_unchanged_during_the_run": not drift,
        "head_at_import": LAUNCH_HEAD,
        "head_at_emit": head_now,
        "head_unchanged_during_the_run": (
            LAUNCH_HEAD.get("head") == head_now.get("head")),
        "worktree_was_dirty_at_import": LAUNCH_HEAD.get("dirty"),
    }


def assert_source_unchanged(where: str, *, fixture: bool = True) -> dict:
    """REFUSE THE EMIT if any of the code that ran changed under it.

    Not the RUN -- the modules are in memory and the run is unaffected.
    What is not honest is a receipt naming bytes that did not produce it,
    and `producing_code_is_the_committed_bytes` PASSES when the replacement
    is itself committed (R-603)."""
    idy = source_identity_at_launch()
    if idy["closure_drift"]:
        raise LaunchCaptureRefused(
            f"REFUSED at {where}: A MODULE OF THIS RUN'S IMPORT CLOSURE "
            f"CHANGED UNDER IT -- "
            f"{[d['module'] for d in idy['closure_drift']]}. The run is "
            f"unaffected; a receipt stamped from the files would name code "
            f"that DID NOT RUN. Rule 22 as amended (R-605): the capture is "
            f"the CLOSURE, not one file.")
    if not idy["source_unchanged_during_the_run"]:
        raise LaunchCaptureRefused(
            f"REFUSED at {where}: THE SOURCE CHANGED UNDER THIS RUN. This "
            f"process is executing {LAUNCH_SOURCE_SHA256[:16]} (read at "
            f"import) and the file now holds "
            f"{str(idy['on_disk_sha256_at_emit'])[:16]}.")
    if not idy["head_unchanged_during_the_run"]:
        raise LaunchCaptureRefused(
            f"REFUSED at {where}: THE WORKTREE'S HEAD MOVED UNDER THIS RUN "
            f"-- {str(idy['head_at_import'].get('head'))[:12]} -> "
            f"{str(idy['head_at_emit'].get('head'))[:12]}. A "
            f"carrying_commit would name a commit this run did not execute "
            f"from.")
    if not fixture and idy["worktree_was_dirty_at_import"]:
        raise LaunchCaptureRefused(
            f"REFUSED at {where}: THE WORKTREE WAS DIRTY AT IMPORT "
            f"({idy['head_at_import'].get('dirty_paths')}). For a REAL "
            f"artifact the producing code must be locatable in a commit; "
            f"uncommitted bytes are locatable nowhere. A fact for a "
            f"fixture, a refusal for a real run.")
    return idy


capture_closure("module import")
LAUNCH_HEAD = _head_state()



def carrying_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(HERE))
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


def runner_identity() -> dict:
    """WHICH CODE PRODUCED THIS RECEIPT, in a form a rebase cannot rewrite.

    `carrying_commit` is the tree's HEAD, and in a per-seat worktree that is
    whatever commit the worktree was last detached at -- NOT necessarily a
    commit carrying the code that ran. DA 63 measured its own worktree three
    landings behind while running this module. And even a correct commit id
    can be rewritten: this batch's emitter commit was rebased to a new id
    with byte-identical content between landing and emitting.

    So the durable citation is the CONTENT DIGEST of the module that ran,
    with the commit ids beside it and labelled for what they are.
    """
    src = Path(__file__).resolve()
    r = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    last = r.stdout.strip() if r.returncode == 0 else ""
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(src)],
                           capture_output=True, text=True,
                           cwd=str(src.parent))
    return {
        "runner_path": "live/mm_research/e2_a_runner.py",
        "runner_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "runner_sha256_is_the_durable_citation": (
            "the digest identifies the code that produced this receipt "
            "whatever happens to the history above it. Resolve it first."),
        "runner_commit_best_effort": last or None,
        "tree_head_at_run": carrying_commit(),
        "producing_code_is_the_committed_bytes": (
            dirty.returncode == 0 and dirty.stdout.strip() == ""),
        "why_that_field": (
            "false means the module had uncommitted edits when it ran, so "
            "the commit ids below it describe a DIFFERENT file than the one "
            "that produced these numbers. Stated, never inferred."),
    }


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



def verify_consumed_sources(sources=None, root=None) -> dict:
    """v8: every artifact the declaration's CONSUMED window names, resolved.

    THE SEAM THIS CLOSES. `e2_a_declare` cannot read a data path -- its own
    AST cell proves it -- so the consumed symbol-days are TRANSCRIBED into
    it from the censuses and the sealed smoke. A transcription is exactly
    where a digit turns over, and the declaration would go on naming an
    artifact that no longer says what it claims. This locates each source in
    the ledger and digests it. A source that is ABSENT fails; it never
    skips.
    """
    srcs = list(D.CONSUMED_SOURCES if sources is None else sources)
    base = ROOT if root is None else Path(root)
    rows, absent, mismatched = [], 0, 0
    for src in srcs:
        p = base / src["path"]
        if not p.is_file():
            absent += 1
            rows.append({"path": src["path"], "status": "ABSENT",
                         "why": "a digest without a resolvable location is "
                                "a pin to nothing"})
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != src["sha256"]:
            mismatched += 1
        rows.append({"path": src["path"],
                     "status": "OK" if got == src["sha256"]
                               else "DIGEST_MOVED",
                     "sha256": got, "declared_sha256": src["sha256"],
                     "n_symbols": len(src["symbols"]),
                     "n_days": len(src["days"])})
    return {"ok": absent == 0 and mismatched == 0,
            "n_sources": len(srcs), "n_absent": absent,
            "n_mismatched": mismatched, "sources": rows,
            "consumed_symbol_days_n": {k: len(v) for k, v in
                                       D.consumed_symbol_days().items()}}


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


def era_leg(sym: str, day: str) -> dict:
    """v7 / R-580(C)(2): CLAUDE.md rule 5 as a per-symbol-day admission leg.

    MEASURED ROW-WISE on `recv_ns`, never inferred from the date -- the
    boundary falls at 13:48:54 INSIDE 2026-08-24, so a date test would
    admit a day that is 56% legacy-stamped.

    Why it binds E2-A and did not bind E2.0: E2.0 reads the exchange stamp
    `T`. E2-A's estimand IS sub-second arrival order on `recv_ns`, and a
    legacy-stamped row carries up to ~0.6 s of parse-backlog error,
    concentrated in bursts -- exactly when queue position matters.
    """
    files = E20._hour_files("bookTicker", sym, day)
    n_rows = n_legacy = 0
    mn = mx = None
    for f in files:
        a = pd.read_csv(f, header=None, usecols=[0],
                        dtype={0: "int64"})[0].to_numpy()
        if a.size == 0:
            continue
        n_rows += int(a.size)
        n_legacy += int((a < ERA_BOUNDARY_RECV_NS).sum())
        lo, hi = int(a.min()), int(a.max())
        mn = lo if mn is None else min(mn, lo)
        mx = hi if mx is None else max(mx, hi)
    share = (n_legacy / n_rows) if n_rows else None
    post = bool(n_rows > 0 and n_legacy == 0)
    return {
        "n_rows": n_rows,
        "n_legacy_stamped": n_legacy,
        "legacy_share": share,
        "post_boundary": post,
        "min_recv_ns": mn,
        "max_recv_ns": mx,
        "boundary_recv_ns": ERA_BOUNDARY_RECV_NS,
        "boundary_utc": ERA_BOUNDARY_UTC,
        "measured_row_wise": True,
        "why": ("every bookTicker row carries recv_ns at or after the "
                "hf_ws_v2 stamp boundary" if post else
                f"{n_legacy} of {n_rows} bookTicker rows are LEGACY-STAMPED "
                f"(recv_ns < {ERA_BOUNDARY_RECV_NS}); a queue simulation on "
                f"recv_ns cannot use them"),
    }


def stream_file_counts(sym: str, day: str) -> dict:
    return {s: len(E20._hour_files(s, sym, day))
            for s in ("bookTicker", "trade", "depth20")}


def day_admission(sym: str, day: str, counts: dict,
                  health: dict | None = None,
                  gap: float | None = None,
                  age: dict | None = None,
                  era: dict | None = None,
                  require_era: bool = True,
                  outage: dict | None = None,
                  require_outage: bool = True) -> dict:
    """v8. A UTC day is ADMISSIBLE FOR A SYMBOL iff (a) all THREE streams
    carry 24 hour-files, (b) THE COLLECTOR WAS LIVE, (c) every bookTicker
    row is post-boundary, and (d) THE TAPE CARRIES NO OUTAGE RUN.

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
    #: v7 leg (c). ABSENCE MUST NOT READ AS A PASS (rule 11 / protocol rule
    #: 11): a caller that does not supply the era measurement gets a REFUSAL
    #: with a named status, never a quiet admission. `require_era=False` is
    #: for the census, which says in its own receipt that the leg was not
    #: evaluated.
    if require_era:
        if era is None:
            era_ok = False
            reasons.append("ERA_LEG_NOT_EVALUATED: rule 5 admissibility was "
                           "not measured for this symbol-day, so it cannot "
                           "be admitted")
        else:
            era_ok = bool(era.get("post_boundary"))
            if not era_ok:
                reasons.append(f"rule5_legacy_stamped: {era.get('why')}")
    else:
        era_ok = True
    #: v8 leg (d). R-745(4): forward-only from the day after the last
    #: consumed day, the consumed window never re-judged, and an
    #: UNMEASURED leg refuses with its own status -- the predicate itself
    #: lives in the declaring module so the runner cannot drift from what
    #: was declared.
    if require_outage:
        d_leg = D.outage_leg(sym, day,
                             None if outage is None
                             else outage.get("max_gap_run_s"))
        outage_ok = bool(d_leg["admissible"])
        if not outage_ok:
            reasons.append(f"{d_leg['state']}: {d_leg['why']}")
    else:
        outage_ok = True
        d_leg = {"evaluated": False,
                 "why": ("this caller did not evaluate leg (d); "
                         "admissibility here is NOT the E2-A gate "
                         "population")}
    return {"day": day,
            "admissible": bool(all_complete and live and era_ok and outage_ok),
            "stream_file_counts": counts, "streams_complete": complete,
            "collector_health": health,
            "outage_leg_d": d_leg,
            "outage_leg_enforced": bool(require_outage),
            "outage_profile": outage,
            "era_rule5": era if era is not None else {
                "evaluated": False,
                "why": ("this caller did not measure the era leg; "
                        "admissibility here is NOT the E2-A gate population")},
            "era_leg_enforced": bool(require_era),
            "reasons_excluded": reasons,
            "REPORTED_not_gated": {
                "gap_fraction": gap,
                "decision_time_quote_age_ms": age,
                "why": ("these describe how ACTIVE the book is, not whether "
                        "the data is there. v5 gated on the first of them "
                        "and cut ICP from 16 days to 1. v8 gates on the "
                        "gap RUN instead -- missing DATA -- and still "
                        "reports these.")}}


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
        idx, prob, front = -1, np.zeros(0), np.zeros(0)
    filled_pq = float(order_qty) if idx >= 0 else 0.0

    #: v7. THE TESTABILITY PREDICATE, computed per episode. The ordering
    #: E[filled_ProbQueue] >= filled_RiskAverse is ARITHMETIC exactly where
    #: some trade arrives with front = 0 -- there f(0) = 0 makes p = 1, so
    #: ProbQueue takes the whole order at that trade at the latest while
    #: RiskAverse can never exceed it. Where no trade reaches front = 0,
    #: RiskAverse can fill a sliver off the CUMULATIVE volume while ProbQueue
    #: is still a draw, and NEITHER the realisation nor the expectation
    #: ordering holds -- measured, both, in the declaration's regimes.
    testable = bool((front <= 0.0).any()) if len(front) else False
    e_filled_pq = D.expected_filled_probqueue(prob, order_qty)

    return {"filled_qty_RiskAverse": float(filled_ra),
            "filled_qty_ProbQueue_f3": filled_pq,
            "E_filled_qty_ProbQueue_f3": float(e_filled_pq),
            "ordering_testable": testable,
            "front_min": float(front.min()) if len(front) else None,
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

    #: R-584: `book` is either the whole day (a 3-tuple, as the fixture and
    #: E2.0's reader give it) or a StreamedBook that holds only the episode
    #: grid. Both answer the same three questions.
    bk = FullBook(*book) if isinstance(book, tuple) else book
    tr_t, tr_p, tr_q, tr_m = trades
    d_t, d_bp, d_bq, d_ap, d_aq = depth

    if tick is None:
        #: the streamed book carries no price array, so the tick comes from
        #: the trade tape alone there. Stated, not silently different: on the
        #: whole-day path the quote prices are included exactly as before.
        tick = (EP.tick_mode([tr_p, book[1], book[2]])
                if isinstance(book, tuple) else EP.tick_mode([tr_p]))
    #: v7 / REVIEW_DA61_E2A A.4.1. A modal positive diff over fewer than
    #: three distinct quantities is an echo of the sample, not an estimate of
    #: a step. The day REFUSES with a counted status rather than guessing.
    #: The status is about ESTIMATING the step: a caller that SUPPLIES one
    #: (the fixture pins q = 1.0) is not estimating anything and is not
    #: subject to it.
    n_distinct_q = int(np.unique(tr_q).size)
    if qty_step is None and n_distinct_q < MIN_DISTINCT_QUANTITIES:
        return {"day": day, "usable": False, "why": ST_QTY_UNDET,
                "status_day": ST_QTY_UNDET,
                "n_distinct_quantities": n_distinct_q,
                "min_distinct_quantities_required": MIN_DISTINCT_QUANTITIES,
                "detail": ("the quantity step is UNDERDETERMINED: a modal "
                           "positive diff computed from fewer than three "
                           "distinct quantities is a property of the sample "
                           "rather than of the instrument")}
    if qty_step is None:
        qty_step = qty_step_mode(tr_q)
    order_qty = float(qty_step)

    k_tr = np.round(tr_p / tick).astype(np.int64)
    t_max_book = bk.t_max
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
    label_counts = {lb: 0 for lb in LABELS}
    viol_testable, viol_marginal = [], []

    for tp in TP_GRID_S:
        hours = range(24 if tp * 1000 <= 3_600_000 else 23)
        for hh in hours:
            t0 = day0 + hh * 3_600_000
            q0 = bk.before(t0)
            for sign in (1.0, -1.0):
                if q0 is None:
                    status_counts[ST_NO_QUOTE] += 1
                    continue
                tq0, b0, a0 = q0
                #: v7 / B.4.2. The age of the quote this order is placed
                #: FROM. Reported, never gated -- but it partitions the
                #: second gate reading at the declared staleness bar.
                quote_age_ms = float(t0 - tq0)
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

                qT = bk.at_or_before(t_end)
                if qT is None or t_max_book is None or t_end > t_max_book:
                    status_counts[ST_NO_BOOK_TP] += 1
                    continue
                p_x = qT[2] if sign > 0 else qT[1]
                c_fill = sign * (L - m0) / m0 * 1e4 + FEE_MAKER
                c_chase = sign * (p_x - m0) / m0 * 1e4 + FEE_TAKER

                #: v7. The violation is recorded in the regime it happened
                #: in. ONLY the testable regime may refute (A.5): there the
                #: ordering is arithmetic, so a violation can only be an
                #: implementation defect. A marginal-regime violation is the
                #: two models disagreeing where neither claims an ordering,
                #: and it MUST NOT silence the gate.
                if (sim["filled_qty_ProbQueue_f3"]
                        < sim["filled_qty_RiskAverse"] - 1e-9):
                    rec = {"day": day, "tp_s": tp, "hour": hh, "sign": sign,
                           "filled_RiskAverse": sim["filled_qty_RiskAverse"],
                           "filled_ProbQueue_f3":
                               sim["filled_qty_ProbQueue_f3"],
                           "front_min": sim["front_min"],
                           "ordering_testable": sim["ordering_testable"]}
                    (viol_testable if sim["ordering_testable"]
                     else viol_marginal).append(rec)
                if not sim["ordering_testable"]:
                    label_counts[LB_MARGINAL] += 1
                if quote_age_ms > STALENESS_BAR_MS:
                    label_counts[LB_STALE] += 1

                row = {"tp_s": tp, "hour": hh, "sign": sign,
                       "queue_ahead": queue_ahead, "order_qty": order_qty,
                       "quote_age_ms": quote_age_ms,
                       "ordering_testable": sim["ordering_testable"],
                       "E_filled_ProbQueue_f3":
                           sim["E_filled_qty_ProbQueue_f3"],
                       "filled_RiskAverse": sim["filled_qty_RiskAverse"],
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
    ages = df["quote_age_ms"].to_numpy() if len(df) else np.zeros(0)
    out = {"day": day, "usable": True, "tick": float(tick),
           "rows": rows,
           "qty_step": float(qty_step),
           "n_distinct_quantities": n_distinct_q,
           "status_counts": status_counts,
           "label_counts": label_counts,
           "n_attempted": int(sum(status_counts.values())),
           #: v7: the violations, split by the regime they occurred in.
           "ordering_violations_testable": viol_testable,
           "ordering_violations_marginal": viol_marginal,
           "n_ordering_testable": int(
               sum(1 for r in rows if r["ordering_testable"])),
           "n_ordering_marginal": label_counts[LB_MARGINAL],
           #: v7 / B.4.2: the staleness a placement actually carried,
           #: REPORTED per symbol-day. Never a filter on admission.
           "decision_time_quote_age_ms": (
               {"n": int(ages.size),
                "p50": float(np.percentile(ages, 50)),
                "p90": float(np.percentile(ages, 90)),
                "max": float(ages.max()),
                "n_over_declared_bar": label_counts[LB_STALE],
                "declared_bar_ms": STALENESS_BAR_MS}
               if ages.size else {"n": 0}),
           "cells": {}}
    for tp in TP_GRID_S:
        sub = df[df.tp_s == tp] if len(df) else df
        cell = {"n_episodes": int(len(sub))}
        if len(sub):
            fresh = sub[sub.quote_age_ms <= STALENESS_BAR_MS]
            cell["n_episodes_fresh_quote"] = int(len(fresh))
            cell["n_ordering_testable"] = int(sub["ordering_testable"].sum())
            cell["mean_filled_RiskAverse"] = float(
                sub["filled_RiskAverse"].mean())
            cell["mean_E_filled_ProbQueue_f3"] = float(
                sub["E_filled_ProbQueue_f3"].mean())
            for mdl in MODELS:
                cell[f"fill_rate_{mdl}"] = float((sub[f"phi_{mdl}"] > 0).mean())
                cell[f"mean_phi_{mdl}"] = float(sub[f"phi_{mdl}"].mean())
                cell[f"partial_share_{mdl}"] = float(
                    sub[f"partial_{mdl}"].mean())
                for pk in PRICINGS:
                    cell[f"eff_rt_{mdl}_{pk}"] = float(
                        2.0 * sub[f"cost_{mdl}_{pk}"].mean())
                    #: READING B, at the declared staleness bar. Both
                    #: readings are published with their n; a straddle of
                    #: the threshold is STATED, never averaged.
                    cell[f"eff_rt_{mdl}_{pk}_fresh"] = (
                        float(2.0 * fresh[f"cost_{mdl}_{pk}"].mean())
                        if len(fresh) else None)
        out["cells"][str(tp)] = cell
    return out


# --------------------------------------------------------------------------
# aggregation and verdict
# --------------------------------------------------------------------------
def aggregate_symbol(days: list[dict], tp: int, fresh: bool = False) -> dict:
    """Day-clustered mean and the stationary bootstrap, E1-A's estimator.

    `fresh=True` is v7's READING B: the same estimator over the subset of
    episodes whose decision-time quote is no older than the DECLARED
    staleness bar. It is a second reading, never a filter on admission --
    filtering the population on quote age would select on activity, which is
    the defect v6 removed.
    """
    vals = {}
    suffix = "_fresh" if fresh else ""
    nkey = "n_episodes_fresh_quote" if fresh else "n_episodes"
    usable = [d for d in days if d.get("usable")
              and (d["cells"].get(str(tp), {}).get(nkey) or 0) > 0]
    g = len(usable)
    for mdl in MODELS:
        for pk in PRICINGS:
            k = f"eff_rt_{mdl}_{pk}"
            if g == 0:
                vals[k] = {"eff_rt_bps": None, "n_days": 0}
                continue
            series = np.array([d["cells"][str(tp)][k + suffix]
                               for d in usable], dtype=float)
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
    vals["reading"] = ("B_fresh_quotes_only" if fresh else "A_all_episodes")
    vals["staleness_bar_ms"] = STALENESS_BAR_MS if fresh else None
    vals["n_episodes"] = int(sum(
        (d["cells"].get(str(tp), {}).get(nkey) or 0) for d in usable))
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


def staleness_sensitivity(agg_a: dict, agg_b: dict) -> dict:
    """v7 / B.4.2: the gate read BOTH ways, and a straddle STATED.

    Reading A is every resolved episode; reading B is the subset whose
    decision-time quote is no older than the declared staleness bar. If the
    two fall on opposite sides of the 8.0 bps threshold the cell says so.
    THE TWO ARE NEVER AVERAGED -- the same rule the queue bracket carries.
    """
    a = agg_a[f"eff_rt_RiskAverse_{PR_RESIDUAL}"]["eff_rt_bps"]
    b = agg_b[f"eff_rt_RiskAverse_{PR_RESIDUAL}"]["eff_rt_bps"]
    if a is None or b is None:
        return {"state": "NOT_BOTH_READABLE",
                "reading_A_all_episodes_bps": a,
                "reading_B_fresh_quotes_bps": b,
                "staleness_bar_ms": STALENESS_BAR_MS}
    pa, pb = a <= THRESHOLD, b <= THRESHOLD
    return {"state": ("STRADDLES_THE_STALENESS_BAR" if pa != pb
                      else "AGREES_ACROSS_THE_STALENESS_BAR"),
            "reading_A_all_episodes_bps": a,
            "reading_B_fresh_quotes_bps": b,
            "reading_A_n_days": agg_a[
                f"eff_rt_RiskAverse_{PR_RESIDUAL}"]["n_days"],
            "reading_B_n_days": agg_b[
                f"eff_rt_RiskAverse_{PR_RESIDUAL}"]["n_days"],
            "reading_A_n_episodes": agg_a.get("n_episodes"),
            "reading_B_n_episodes": agg_b.get("n_episodes"),
            "staleness_bar_ms": STALENESS_BAR_MS,
            "never_averaged": True}


#: v7 SEALING. Structural keys that merely NAME the gate rather than carrying
#: a number; without this list the marker scan would strip the metadata a
#: reader needs to know WHICH gate is sealed.
SEALED_KEY_ALLOW = ("gate_row_tp_s", "gates_the_run", "gate_read",
                    "when_the_gate_may_be_read", "why_no_gate_is_read",
                    "gate_bearing_pricing")


def _is_sealed_key(k: str, markers=SEALED_KEY_MARKERS) -> bool:
    if k in SEALED_KEY_ALLOW:
        return False
    kl = k.lower()
    return any(m in kl for m in markers)


#: A SECOND NET, WITH A DIFFERENT SHAPE FROM THE FIRST. The marker list and
#: the leak scan share a list, so they share a blind spot -- DA 63's own
#: `staleness_sensitivity.reading_A_all_episodes_bps` passed both. This net
#: does not read the marker list at all: it censuses every NUMERIC LEAF that
#: survived and refuses on any whose name is economic-SHAPED. Two nets
#: cut from the same cloth catch the same things; these are cut differently
#: on purpose.
ECONOMIC_SHAPE = ("bps", "eff_rt", "eff_", "cost", "ci_", "capture", "pnl",
                  "spread", "fee", "rebate", "markout", "cents", "profit",
                  "edge", "revenue")


def numeric_key_census(obj, key=""):
    """Every key name in the object that carries a NUMERIC leaf.

    Published in the sealed receipt so a reader can audit what survived
    instead of taking the seal on trust."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out |= numeric_key_census(v, str(k))
    elif isinstance(obj, list):
        for v in obj:
            out |= numeric_key_census(v, key)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(key)
    return out


def find_economic_shaped_leaks(obj):
    """Numeric leaves whose NAME is economic-shaped, whatever the marker
    list thinks. Fails closed: an unforeseen economic field refuses the
    emission rather than riding out in a receipt labelled sealed."""
    return sorted(k for k in numeric_key_census(obj)
                  if any(m in k.lower() for m in ECONOMIC_SHAPE))


#: RULE 22 MEETS THE SEAL, and the answer is the SHAPE, not an exemption.
#: A closure stamped as a dict keyed by FILENAME puts module names where
#: the seal's nets look for economic KEYS -- and this programme owns a
#: module called `e1_markout_scan.py`. The first draft of this fix carved
#: the closure out of both nets, which is a HOLE in a seal to solve a
#: problem the seal does not have. The closure is stamped as a LIST OF
#: [name, digest] PAIRS instead: no module name is ever a key, the values
#: are strings so the numeric census never sees them, and NEITHER NET IS
#: TOUCHED.
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def closure_name_collides_with_the_seal(name: str) -> bool:
    """Would this module's NAME be read as an economic key if the closure
    were ever stamped as a dict again? Reported, never relied on."""
    n = name.lower()
    return (any(m in n for m in SEALED_KEY_MARKERS)
            or any(m in n for m in ECONOMIC_SHAPE))


def redact_sealed(obj, path="", dropped=None):
    """Remove every economic quantity, recording WHICH keys were removed.

    The open receipt keeps its shape and its statuses; what it loses is
    named. A seal that did not say what it covered would be a seal over
    whatever happened to be convenient.
    """
    if dropped is None:
        dropped = []
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            if _is_sealed_key(str(k)):
                dropped.append(here)
                continue
            out[k] = redact_sealed(v, f"{path}.{k}" if path else str(k),
                                   dropped)
        return out
    if isinstance(obj, list):
        return [redact_sealed(v, f"{path}[{i}]", dropped)
                for i, v in enumerate(obj)]
    return obj


def find_sealed_leaks(obj, path="", markers=SEALED_KEY_MARKERS):
    """The falsifier for the seal: any economic key surviving in the open
    receipt is a LEAK, named. Run on every sealed emission.

    `markers` is a parameter ONLY so the regression control can drive the
    pre-fix list and show the miss. Every production call takes the default.
    """
    leaks = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            if _is_sealed_key(str(k), markers):
                leaks.append(here)
            leaks += find_sealed_leaks(v, here, markers)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaks += find_sealed_leaks(v, f"{path}[{i}]", markers)
    return leaks


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def inherited_control_block(run_repro: bool, rep: dict | None = None) -> dict:
    """v7 / REVIEW_DA61_E2A A.2. THE FIELD IS ALWAYS PRESENT.

    --no-repro turned the gating control off and the receipt went SILENT --
    the key ABSENT rather than false. That is the P-003 asymmetry class in
    P-002: a reader cannot tell "the control passed" from "the control was
    never asked". `null` is not `false`: the control neither passed nor
    failed, because it did not run.
    """
    if not run_repro:
        return {"reproduced": None,
                "gates_the_run": False,
                "why_skipped": ("--no-repro: the E1-A reproduction control "
                                "was NOT RUN, so this run is not gated on "
                                "E2-A superseding E1-A's estimator rather "
                                "than measuring a different one. A receipt "
                                "produced this way is not gate-bearing."),
                "regimes": None}
    rep = rep or {}
    return {"reproduced": bool(rep.get("reproduced")),
            "regimes": rep.get("regimes"),
            "gates_the_run": True,
            "why_skipped": None}


def run(symbols, out_path: Path | None, min_days: int | None = None,
        run_repro: bool = True, sealed: bool = False,
        sealed_out: Path | None = None) -> dict:
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
              "runner_identity": runner_identity(),
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
              "scope_R584": {
                  "ruling": "R-584 (USER, 2026-09-06): scope is BTC for now",
                  "gate_symbol": GATE_SYMBOL,
                  "the_thin_name_cell": "DEFERRED_NOT_RESOLVED",
                  "the_twelve": "CONTEXT, not re-run as a population"},
              "memory_cap_gib": DAY_RSS_CAP_GIB,
              "symbols": {}}

    #: v8: THE CONSUMED WINDOW'S SOURCES ARE RESOLVED BEFORE ANY DAY IS
    #: ADMITTED. Leg (d) refuses a symbol-day by consulting a list that was
    #: TRANSCRIBED into the declaring module (which cannot read a data
    #: path). If one of those artifacts is gone or its bytes moved, the
    #: window the run is about to apply is a claim nobody can check -- so
    #: the run REFUSES here rather than admitting days against it.
    consumed = verify_consumed_sources()
    result["consumed_window"] = consumed
    if not consumed["ok"]:
        result["REFUSED"] = (
            f"the CONSUMED WINDOW's sources do not resolve: "
            f"{consumed['n_absent']} absent, {consumed['n_mismatched']} at a "
            f"digest other than the declared one. Leg (d) would be applied "
            f"against a window that cannot be verified.")
        if out_path:
            out_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise E2ARefused(result["REFUSED"])

    #: v7 / REVIEW_DA61_E2A A.2: THE FIELD IS ALWAYS PRESENT. --no-repro
    #: turned the gating control off and the receipt went SILENT -- the key
    #: absent rather than false. `null` is not `false`: the control neither
    #: passed nor failed, because it did not run.
    if not run_repro:
        result["inherited_control"] = inherited_control_block(False)
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
            #: v7 leg (c). The era leg is measured only where the cheap legs
            #: already admit -- it reads recv_ns off every bookTicker row, so
            #: running it on a day already excluded on hour-file counts would
            #: buy nothing. A day it is not run for is NOT silently admitted:
            #: `day_admission` refuses on ERA_LEG_NOT_EVALUATED.
            pre = all(n == HOURS_PER_DAY_FILES for n in counts.values()) \
                and bool(health.get("live"))
            era = era_leg(sym, day) if pre else None
            #: v8 leg (d), read the same way: the leg refuses a CONSUMED or
            #: pre-window day with no measurement at all, so the tape is
            #: streamed only where the reading can change the answer.
            needs = D.outage_leg(sym, day)["state"] \
                == "REFUSED_OUTAGE_LEG_NOT_EVALUATED"
            outage = outage_measure(sym, day) if (pre and needs) else None
            admissions.append(day_admission(sym, day, counts, health,
                                            era=era, require_era=True,
                                            outage=outage,
                                            require_outage=True))
        adm_days = [a["day"] for a in admissions if a["admissible"]]
        n_pre_era = sum(1 for a in admissions
                        if all(a["streams_complete"].values())
                        and bool((a["collector_health"] or {}).get("live")))
        era_block = {
            "boundary_recv_ns": ERA_BOUNDARY_RECV_NS,
            "boundary_utc": ERA_BOUNDARY_UTC,
            "n_days_admissible_before_the_era_leg": n_pre_era,
            "n_days_admissible_after_the_era_leg": len(adm_days),
            "n_days_lost_to_the_era_leg": n_pre_era - len(adm_days),
            "legacy_share_by_day": {
                a["day"]: (a["era_rule5"] or {}).get("legacy_share")
                for a in admissions if isinstance(a.get("era_rule5"), dict)
                and a["era_rule5"].get("measured_row_wise")},
            "why": ("CLAUDE.md rule 5: E2-A's estimand is sub-second arrival "
                    "order on recv_ns, so a legacy-stamped row carries up to "
                    "~0.6 s of parse-backlog error into the queue "
                    "simulation"),
        }
        #: R-580(C)(2). Under the era leg no symbol reaches the declared
        #: minimum of 14 until ~2026-09-09. THE BAR IS NOT MOVED. In SEALED
        #: mode the mechanism still runs on the admissible post-boundary days
        #: -- economics sealed and never read, statuses and resources
        #: published -- and NO GATE IS READ.
        refusal = None
        if len(adm_days) < min_days:
            refusal = (f"{len(adm_days)} admissible days < the declared "
                       f"minimum {min_days}: no gate is read for this symbol "
                       f"and the cap is not relaxed")
            if not sealed:
                result["symbols"][sym] = {
                    "admissions": admissions,
                    "n_admissible_days": len(adm_days),
                    "admissible_days": adm_days,
                    "era_leg": era_block,
                    "REFUSED": refusal,
                    "the_population_collapse_is_the_ERA_LEG": (
                        era_block["n_days_lost_to_the_era_leg"] > 0)}
                continue
        resources = []
        for day in adm_days:
            #: R-584. The bookTicker day is STREAMED hour-file by hour-file
            #: and only the episode grid is kept. Every stage is guarded:
            #: a day over the declared cap REFUSES, the cap is not raised
            #: and the population is not made smaller to fit it.
            st = {"day": day}
            w = time.time()
            book = stream_book(sym, day, extend=True)
            st["book"] = {"wall_s": round(time.time() - w, 2),
                          "n_hour_files": book.n_files,
                          "n_rows_streamed": book.n_rows,
                          "n_bad_quotes": book.n_bad_quotes,
                          "peak_single_hour_file_rss_gib":
                              round(book.peak_file_rss_gib, 3),
                          "kept": "the episode grid only"}
            st["after_book"] = memory_guard("after the streamed book", day)
            w = time.time()
            trades, _, tmeta = E20.read_trades(sym, day)
            st["trades"] = {"wall_s": round(time.time() - w, 2), **tmeta}
            st["after_trades"] = memory_guard("after the trade tape", day)
            w = time.time()
            depth, _, dmeta = read_depth20(sym, day)
            st["depth20"] = {"wall_s": round(time.time() - w, 2), **dmeta}
            st["after_depth20"] = memory_guard("after depth20", day)
            w = time.time()
            ev = evaluate_day(sym, day, decl_digest, book, trades, depth)
            st["evaluate"] = {"wall_s": round(time.time() - w, 2)}
            st["after_evaluate"] = memory_guard("after the episodes", day)
            ev["stream_meta"] = {
                "book": {"streamed": True, "n_rows": book.n_rows,
                         "n_hour_files": book.n_files,
                         "n_bad_quotes": book.n_bad_quotes},
                "trades": tmeta, "depth20": dmeta}
            evaluated.append(ev)
            resources.append(st)
            del book, trades, depth
        aggs = {str(tp): aggregate_symbol(evaluated, tp) for tp in TP_GRID_S}
        aggs_fresh = {str(tp): aggregate_symbol(evaluated, tp, fresh=True)
                      for tp in TP_GRID_S}
        primary = aggs[str(TP_PRIMARY_S)]
        #: v7 / A.5. The trigger is narrowed to the arithmetic regime. A
        #: marginal-regime violation is COUNTED and the gate is still read;
        #: only a testable-regime violation can be an implementation defect.
        vt = [v for d in evaluated
              for v in d.get("ordering_violations_testable", [])]
        vm = [v for d in evaluated
              for v in d.get("ordering_violations_marginal", [])]
        n_test = sum(d.get("n_ordering_testable", 0) for d in evaluated)
        n_marg = sum(d.get("n_ordering_marginal", 0) for d in evaluated)
        pc = [d["cells"][str(TP_PRIMARY_S)] for d in evaluated
              if d.get("usable") and d["cells"][str(TP_PRIMARY_S)]["n_episodes"]]
        e_pq = (float(np.mean([c["mean_E_filled_ProbQueue_f3"] for c in pc]))
                if pc else None)
        f_ra = (float(np.mean([c["mean_filled_RiskAverse"] for c in pc]))
                if pc else None)
        ordering = D.ordering_verdict(n_test, len(vt), n_marg, len(vm),
                                      e_pq, f_ra)
        ordering["mean_E_filled_ProbQueue_f3"] = e_pq
        ordering["mean_filled_RiskAverse"] = f_ra
        vd = verdict(primary, decl)
        if ordering["state"] == "REFUTES_THE_BRACKET":
            vd = {"state": "REFUTES_THE_BRACKET",
                  "why": (f"{len(vt)} episodes IN THE TESTABLE REGIME where "
                          f"ProbQueue-f3 filled LESS than RiskAverse. Some "
                          f"trade in each reached front = 0, where the "
                          f"ordering is arithmetic, so this can only be an "
                          f"implementation defect and no overlay verdict may "
                          f"be read."),
                  "n_ordering_violations_testable": len(vt),
                  "examples": vt[:5]}
        stale = staleness_sensitivity(primary, aggs_fresh[str(TP_PRIMARY_S)])
        icp = D.icp_predicate(primary["episode_skip_rate"], in_aggregate=True)
        ages = [d["decision_time_quote_age_ms"] for d in evaluated
                if d.get("usable") and d["decision_time_quote_age_ms"].get("n")]
        placement = {
            "quote_age_p50_ms_median_over_days": (
                float(np.median([a["p50"] for a in ages])) if ages else None),
            "quote_age_p90_ms_median_over_days": (
                float(np.median([a["p90"] for a in ages])) if ages else None),
            "quote_age_max_ms": (float(max(a["max"] for a in ages))
                                 if ages else None),
            "n_episodes_over_the_declared_bar": sum(
                d["label_counts"][LB_STALE] for d in evaluated
                if d.get("usable")),
            "declared_bar_ms": STALENESS_BAR_MS,
            "REPORTED_not_gated": True,
        }
        cell_label = None
        if sym == "ICPUSDT":
            lab = decl["the_ICP_cell_label_v7"]
            cell_label = {"label": lab["label"],
                          "mechanism": lab["the_mechanism"],
                          "scope": ("the DECLARED ICP cell -- the label is "
                                    "not a computed threshold applied to "
                                    "every symbol, which would be a bar "
                                    "chosen after seeing. The evidence "
                                    "beside it IS computed."),
                          "evidence_computed_here": placement}
        result["symbols"][sym] = {
            "admissions": admissions,
            "n_admissible_days": len(adm_days),
            "admissible_days": adm_days,
            "era_leg": era_block,
            "days": [{k: v for k, v in d.items() if k != "rows"}
                     for d in evaluated],
            "aggregate_by_tp": aggs,
            "aggregate_by_tp_fresh_quotes_only": aggs_fresh,
            "gate_row_tp_s": TP_PRIMARY_S,
            "verdict": vd,
            "ordering_property": ordering,
            "staleness_sensitivity": stale,
            "placement_quality": placement,
            "cell_label": cell_label,
            "icp_rule": icp,
            "n_ordering_violations_testable": len(vt),
            "n_ordering_violations_marginal": len(vm),
            "REFUSED": refusal,
            #: R-584: the gate is read on BTC. Another symbol may be executed
            #: as a diagnostic and is LABELLED rather than quietly counted.
            "is_the_gate_symbol_under_R584": bool(sym == GATE_SYMBOL),
            "scope_label": (None if sym == GATE_SYMBOL
                            else "NOT_THE_GATE_SYMBOL_UNDER_R584"),
            "resource_observation": resources,
            "memory_cap_gib": DAY_RSS_CAP_GIB,
            "gate_read": bool(refusal is None and not sealed
                              and sym == GATE_SYMBOL),
            "why_no_gate_is_read": (
                None if (refusal is None and not sealed
                         and sym == GATE_SYMBOL) else
                (refusal or "")
                + ("" if sym == GATE_SYMBOL else
                   f" | R-584: {sym} is not the gate symbol "
                   f"({GATE_SYMBOL}); this cell is a diagnostic")
                + (
                    " | SEALED SMOKE (R-580(C)(2)): the mechanism ran on the "
                    "admissible post-boundary days, every economic quantity "
                    "is sealed and NOT read, and the gate is read only at "
                    "G >= 14 post-boundary complete days (~2026-09-09)"
                    if sealed else "")),
            "the_population_collapse_is_the_ERA_LEG": (
                era_block["n_days_lost_to_the_era_leg"] > 0),
            "wall_s": round(time.time() - w0, 2)}

    result["wall_s_total"] = round(time.time() - t_start, 2)
    #: RULE 22 / R-605. The closure, HEAD and the dirty state, captured at
    #: IMPORT and stamped here; the emit REFUSES BY NAME if any moved. A
    #: real run's bar is the strict one: a dirty worktree is locatable in no
    #: commit.
    result["source_identity"] = assert_source_unchanged(
        "the E2-A run's emit", fixture=False)
    try:
        import resource                                       # noqa: PLC0415
        result["max_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss
    except Exception:                                         # noqa: BLE001
        pass

    if not sealed:
        if out_path:
            out_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    #: ---- THE SEALED SMOKE ------------------------------------------------
    #: The full payload is written ONCE, digested, and NOT READ. The open
    #: receipt is the same object with every economic key REMOVED -- and the
    #: removal is CHECKED, not asserted: a marker scan over the open receipt
    #: must find zero leaks or the emission refuses.
    if sealed_out is None:
        raise E2ARefused("sealed run without a sealed output path: the "
                         "economics would have nowhere to go but the open "
                         "receipt")
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    sealed_out.parent.mkdir(parents=True, exist_ok=True)
    sealed_out.write_text(payload)
    dropped: list[str] = []
    openr = redact_sealed(result, "", dropped)
    leaks = find_sealed_leaks(openr)
    econ_leaks = find_economic_shaped_leaks(openr)
    census = sorted(numeric_key_census(openr))
    openr["sealed_payload"] = {
        "path": str(sealed_out.relative_to(ROOT))
                if str(sealed_out).startswith(str(ROOT)) else str(sealed_out),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "n_bytes": len(payload.encode()),
        "n_keys_removed_from_the_open_receipt": len(dropped),
        "keys_removed": sorted(set(
            k.split(".")[-1] for k in dropped)),
        "what_is_sealed": decl["admission_legs_v8"][
            "leg_c_rule5_era_purity"]["the_sealed_smoke_regime"][
                "what_is_SEALED"],
        "when_the_gate_may_be_read": decl["admission_legs_v8"][
            "leg_c_rule5_era_purity"]["the_sealed_smoke_regime"][
                "when_the_gate_may_be_read"],
        "NOT_READ_BY_THIS_RUN": True,
        "leak_scan": {"n_leaks": len(leaks), "leaks": leaks[:20]},
        "second_net_economic_shaped_leak_scan": {
            "n_leaks": len(econ_leaks), "leaks": econ_leaks[:20],
            "why_a_second_net": (
                "the marker scan and the redactor share a list, so they "
                "share a blind spot. This net censuses numeric leaves by "
                "NAME SHAPE instead and refuses on anything economic-"
                "shaped, including a field nobody thought to mark."),
        },
        "numeric_keys_that_survived": census,
        "n_numeric_keys_that_survived": len(census),
    }
    if leaks or econ_leaks:
        openr["REFUSED"] = (
            f"{len(leaks)} marker leak(s) and {len(econ_leaks)} "
            f"economic-shaped leak(s) survived the redaction: the receipt is "
            f"not sealed")
        if out_path:
            out_path.write_text(
                json.dumps(openr, indent=2, sort_keys=True) + "\n")
        raise E2ARefused(openr["REFUSED"])
    if out_path:
        out_path.write_text(json.dumps(openr, indent=2, sort_keys=True) + "\n")
    return openr


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
               levels_present=True, n_levels=N_LEVELS,
               depth_at_touch=None):
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
    #: `queue_at_touch` is the size AHEAD at placement; `depth_at_touch` is
    #: the total depth the snapshot shows at L. They are the same field read
    #: at the same instant, so they default equal -- but the declaration's
    #: marginal counterexample needs them to differ (depth 1000 against a
    #: queue of 100), which a real tape produces whenever the book at L
    #: changes between placement and a trade.
    bq[0, 0] = queue_at_touch if depth_at_touch is None else depth_at_touch
    aq[0, 0] = queue_at_touch if depth_at_touch is None else depth_at_touch
    q_ahead_override = queue_at_touch if depth_at_touch is not None else None
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
    if q_ahead_override is not None:
        #: TWO snapshots: the one at t0- carries the queue AHEAD at
        #: placement, and a later one -- before the first trade -- carries
        #: the total depth at L. That is what a live book does between a
        #: placement and the print that trades against it, and it is what
        #: the declaration's marginal counterexample needs (queue 100
        #: against depth 1000).
        bq0 = bq.copy()
        aq0 = aq.copy()
        bq0[0, 0] = q_ahead_override
        aq0[0, 0] = q_ahead_override
        depth = (np.array([DAY0_MS - 1, DAY0_MS + 500], dtype=np.int64),
                 np.vstack([bp, bp]), np.vstack([bq0, bq]),
                 np.vstack([ap, ap]), np.vstack([aq0, aq]))
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
        POST = {"post_boundary": True, "legacy_share": 0.0, "n_rows": 10,
                "n_legacy_stamped": 0, "measured_row_wise": True,
                "why": "synthetic: every row post-boundary"}
        #: v8: FIX_DAY (20260820) is a CONSUMED day, so leg (d) refuses it
        #: whatever the tape says. These four cells are about legs (a)-(c)
        #: and opt OUT of leg (d) EXPLICITLY -- and the cell below proves
        #: the opt-out is not the default.
        a_quiet = day_admission("ICPUSDT", FIX_DAY, full, h_live,
                                gap=0.99, age={"p50_ms": 5000.0}, era=POST,
                                require_outage=False)
        a_d20 = day_admission("ICPUSDT", FIX_DAY, dict(full, depth20=23),
                              h_live, era=POST, require_outage=False)
        a_out = day_admission("ICPUSDT", FIX_DAY, full, h_out, era=POST,
                              require_outage=False)
        a_rs = day_admission("ICPUSDT", FIX_DAY, full, h_restart, era=POST,
                             require_outage=False)
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

        # ==== v7 ==========================================================
        # -- 16. THE ERA LEG (rule 5), both directions and the absence case --
        POSTB = {"post_boundary": True, "legacy_share": 0.0, "n_rows": 100,
                 "n_legacy_stamped": 0, "measured_row_wise": True,
                 "why": "synthetic: every row post-boundary"}
        LEGACY = {"post_boundary": False, "legacy_share": 0.56, "n_rows": 100,
                  "n_legacy_stamped": 56, "measured_row_wise": True,
                  "why": "synthetic: 56 of 100 rows legacy-stamped"}
        a_post = day_admission("ICPUSDT", FIX_DAY, full, h_live, era=POSTB,
                               require_outage=False)
        a_leg = day_admission("ICPUSDT", FIX_DAY, full, h_live, era=LEGACY,
                              require_outage=False)
        a_none = day_admission("ICPUSDT", FIX_DAY, full, h_live, era=None,
                               require_outage=False)
        ck("v7 ERA LEG POSITIVE CONTROL: a post-boundary day with a live "
           "collector ADMITS -- the leg is not a guard shown only refusing",
           a_post["admissible"] and a_post["era_leg_enforced"],
           f"legacy_share 0.0 -> admissible {a_post['admissible']}")
        ck("v7 ERA LEG KNOWN-BAD: a LEGACY-STAMPED day REFUSES with the "
           "share named, however complete its streams are",
           (not a_leg["admissible"])
           and any("rule5_legacy_stamped" in r
                   for r in a_leg["reasons_excluded"]),
           f"legacy_share 0.56 -> {a_leg['reasons_excluded'][-1][:90]} -- a "
           f"queue simulation on recv_ns cannot use post-parse stamps")
        ck("v7 ERA LEG, ABSENCE IS NOT A PASS: a day whose era leg was NOT "
           "MEASURED is REFUSED, not quietly admitted",
           (not a_none["admissible"])
           and any("ERA_LEG_NOT_EVALUATED" in r
                   for r in a_none["reasons_excluded"]),
           "era=None with require_era=True -> ERA_LEG_NOT_EVALUATED. "
           "Rule 11: absence must never read as a pass")

        # -- v8 LEG (d): THE OUTAGE PREDICATE, WIRED --------------------
        #: A declaration nobody applies is rule 17's defect: the predicate
        #: is proven in `e2_a_declare`'s own battery, and these cells prove
        #: THE RUNNER'S ADMISSION PATH consumes it -- the seam, not the
        #: module's invariant a second time.
        FWD = D.FORWARD_WINDOW_START_DAY
        d_clean = day_admission("ICPUSDT", FWD, full, h_live, era=POSTB,
                                outage={"max_gap_run_s": 3})
        d_run = day_admission("ICPUSDT", FWD, full, h_live, era=POSTB,
                              outage={"max_gap_run_s": D.OUTAGE_RUN_S})
        d_none = day_admission("ICPUSDT", FWD, full, h_live, era=POSTB)
        d_consumed = day_admission("ICPUSDT", FIX_DAY, full, h_live,
                                   era=POSTB, outage={"max_gap_run_s": 0})
        ck("v8 LEG (d) POSITIVE CONTROL, IN THE RUNNER'S OWN ADMISSION "
           "PATH: a day inside the forward window whose missing seconds are "
           "scattered holes ADMITS -- the leg must be able NOT to fire",
           d_clean["admissible"] and d_clean["outage_leg_enforced"]
           and d_clean["outage_leg_d"]["state"] == "ADMITTED_NO_OUTAGE_RUN",
           f"{FWD}, longest run 3 s -> admissible "
           f"{d_clean['admissible']}")
        ck("v8 LEG (d) KNOWN-BAD: the same day with ONE run of "
           f"{D.OUTAGE_RUN_S} s REFUSES, and the refusal is named in "
           "`reasons_excluded` where the receipt carries it",
           (not d_run["admissible"])
           and d_run["outage_leg_d"]["state"] == "REFUSED_OUTAGE_RUN"
           and any("REFUSED_OUTAGE_RUN" in r
                   for r in d_run["reasons_excluded"]),
           f"{D.OUTAGE_RUN_S} s -> {d_run['reasons_excluded'][-1][:80]}")
        ck("v8 LEG (d), ABSENCE IS NOT A PASS: a day whose gap run was not "
           "measured REFUSES with its own status, exactly as the era leg "
           "does",
           (not d_none["admissible"])
           and any("REFUSED_OUTAGE_LEG_NOT_EVALUATED" in r
                   for r in d_none["reasons_excluded"]),
           "outage=None with require_outage=True -> "
           "REFUSED_OUTAGE_LEG_NOT_EVALUATED")
        ck("v8 THE CONSUMED WINDOW IS NEVER RE-JUDGED, AND THE DEFAULT IS "
           "ON: FIX_DAY is consumed, so the same clean measurement that "
           f"admits at {FWD} REFUSES here BY NAME as consumed -- which is "
           "also what makes the four legacy cells' explicit "
           "`require_outage=False` visible rather than assumed",
           (not d_consumed["admissible"])
           and d_consumed["outage_leg_d"]["state"]
           == "REFUSED_CONSUMED_NEVER_REJUDGED"
           and a_post["admissible"] and not a_post["outage_leg_enforced"],
           f"{FIX_DAY} with max_gap_run_s 0 -> "
           f"{d_consumed['outage_leg_d']['state']}")

        # -- v8: ONE run-length implementation, two callers ---------------
        _pres = np.ones(86_400, bool)
        _pres[1000:1000 + D.OUTAGE_RUN_S] = False          # one 60 s run
        _prof_run = gap_runs(_pres, 86_400 - D.OUTAGE_RUN_S)
        _scatter = np.ones(86_400, bool)
        _scatter[::3] = False                              # 1 s holes only
        _prof_scatter = gap_runs(_scatter, 57_600)
        _t = (np.flatnonzero(_pres).astype(np.int64) * 1000
              + int(pd.Timestamp(FIX_DAY, tz="UTC").timestamp()) * 1000)
        ck("v8 THE RUN LENGTH HAS ONE IMPLEMENTATION: `gap_profile` (over an "
           "in-memory day) and `gap_runs` (what the streamed measurement "
           "calls) return the SAME max_gap_run_s on the same seconds, and "
           "the two shapes the leg must separate come out different",
           gap_profile(_t, FIX_DAY)["max_gap_run_s"]
           == _prof_run["max_gap_run_s"] == D.OUTAGE_RUN_S
           and _prof_scatter["max_gap_run_s"] == 1
           and _prof_scatter["n_runs_ge_60s"] == 0,
           f"one {D.OUTAGE_RUN_S} s run -> max "
           f"{_prof_run['max_gap_run_s']} s; 28,800 one-second holes -> max "
           f"{_prof_scatter['max_gap_run_s']} s, "
           f"{_prof_scatter['n_runs_ge_60s']} runs >= "
           f"{D.OUTAGE_RUN_S} s")

        # -- v8: the consumed-source seam, driven on a SYNTHETIC root -----
        #: ON A TEMP ROOT ON PURPOSE. The real check runs in `run()`'s
        #: preflight, where a moved consumed source must stop a gate run;
        #: driving it here against the ledger would open three paths under
        #: data/mm_hf and cost this fixture the data-free property it
        #: exists to prove. The BEHAVIOUR is what is proven here, on files
        #: this cell writes itself.
        with tempfile.TemporaryDirectory() as _td:
            _tr = Path(_td)
            #: NOT under a `data/mm_hf/` path, deliberately: this fixture's
            #: data-free proof matches that substring, and a synthetic file
            #: shaped like a ledger path would spend the very property the
            #: proof exists to show. The check resolves a path and a digest;
            #: the path's SHAPE is nothing to it.
            (_tr / "fixture_artifacts").mkdir(parents=True)
            _art = _tr / "fixture_artifacts" / "consumed.json"
            _art.write_text('{"a": 1}\n')
            _real = hashlib.sha256(_art.read_bytes()).hexdigest()
            _tmpl = {"path": "fixture_artifacts/consumed.json",
                     "symbols": ("BTCUSDT",), "days": ("20260901",)}
            _cs = verify_consumed_sources(
                sources=[dict(_tmpl, sha256=_real)], root=_tr)
            _bad_cs = verify_consumed_sources(
                sources=[dict(_tmpl, sha256="0" * 64)], root=_tr)
            _absent_cs = verify_consumed_sources(
                sources=[dict(_tmpl, sha256=_real,
                              path="fixture_artifacts/no_such.json")],
                root=_tr)
        ck("v8 THE CONSUMED-SOURCE SEAM ADMITS: a source that is present at "
           "the digest the declaration names RESOLVES -- the transcription "
           "in a module that cannot read data is checked where the ledger "
           "can be read",
           _cs["ok"] and _cs["n_sources"] == 1 and _cs["n_mismatched"] == 0
           and _cs["n_absent"] == 0,
           f"present at its digest -> ok {_cs['ok']}")
        ck("KNOWN-BAD, DRIVEN, BOTH WAYS: a MOVED digest is refused and an "
           "ABSENT artifact is refused -- the check FAILS when the file is "
           "gone rather than skipping, which is how a skipped check reads "
           "as a passed one (R-649)",
           (not _bad_cs["ok"]) and _bad_cs["n_mismatched"] == 1
           and (not _absent_cs["ok"]) and _absent_cs["n_absent"] == 1,
           f"perturbed digest -> mismatched {_bad_cs['n_mismatched']}; "
           f"absent path -> absent {_absent_cs['n_absent']}")

        # -- 17-19. THE ORDERING PROPERTY, END TO END IN THE RUNNER ---------
        #: Driven through evaluate_day, not through the model functions:
        #: 'the wiring is where the sign conventions can invert' (R-570(B)).
        #: THE REVIEWER'S A.5 CASE, at its own numbers: queue_ahead 100,
        #: order 10, one opposite trade of 105, depth at L 200.
        bkM, trM, dpM = _synth_day(queue_at_touch=100.0, depth_at_touch=200.0,
                                   trades=[(1000, 0, 105.0, True)])
        evM = evaluate_day("ICPUSDT", FIX_DAY, digest, bkM, trM, dpM,
                           tick=0.001, qty_step=10.0)
        rM = _row(evM, hour=0)
        ck("v7 ORDERING, MARGINAL REGIME REPRODUCED IN THE RUNNER'S OWN "
           "WIRING at the reviewer's numbers: no trade reaches front = 0, so "
           "the episode is classified NOT TESTABLE and any violation is "
           "counted as MARGINAL rather than raising a refutation",
           rM is not None and rM["ordering_testable"] is False
           and len(evM["ordering_violations_testable"]) == 0
           and abs(rM["filled_RiskAverse"] - 5.0) < 1e-9
           and abs(rM["E_filled_ProbQueue_f3"] - 5.0) < 1e-9,
           f"queue_ahead {rM['queue_ahead']}, order 10, one trade of 105 at "
           f"depth 200: RiskAverse fills {rM['filled_RiskAverse']} off the "
           f"CUMULATIVE volume, E[ProbQueue] {rM['E_filled_ProbQueue_f3']}, "
           f"testable={rM['ordering_testable']}. THE PRE-v7 RUNNER WOULD "
           f"HAVE READ NO GATE FOR THIS SYMBOL")
        vio_m = sum(
            1 for sd in range(2000)
            if simulate_episode(100.0, 10.0, np.array([105.0]),
                                np.array([200.0]),
                                np.random.default_rng(sd)
                                )["filled_qty_ProbQueue_f3"] < 5.0 - 1e-9)
        ck("v7 AND THE REVIEWER'S MEASUREMENT REPRODUCES IN THIS CODE: about "
           "half of all seeds violate the per-episode REALISATION ordering "
           "in the marginal regime, with no defect present",
           800 <= vio_m <= 1200,
           f"{vio_m} of 2,000 seeds = {vio_m / 2000:.4f} (REVIEW_DA61_E2A "
           f"A.5 measured 993 / 2000 = 0.4965). A per-episode realisation "
           f"test on a stochastic model is a coin flip with a veto attached")
        bkT, trT, dpT = _synth_day(queue_at_touch=50.0,
                                   trades=[(1000, 0, 60.0, True),
                                           (2000, 0, 60.0, True)])
        evT = evaluate_day("ICPUSDT", FIX_DAY, digest, bkT, trT, dpT,
                           tick=0.001, qty_step=10.0)
        rT = _row(evT)
        vio_t = sum(
            1 for sd in range(2000)
            if simulate_episode(50.0, 10.0, np.array([60.0, 60.0]),
                                np.array([50.0, 50.0]),
                                np.random.default_rng(sd)
                                )["filled_qty_ProbQueue_f3"] < 10.0 - 1e-9)
        ck("v7 ORDERING, TESTABLE REGIME: a trade arrives with the queue "
           "already cleared (front = 0), so ProbQueue fills the whole order "
           "and the ordering holds ARITHMETICALLY -- zero violations over "
           "2,000 seeds, which is what makes a violation here a DEFECT",
           rT is not None and rT["ordering_testable"] is True
           and len(evT["ordering_violations_testable"]) == 0
           and len(evT["ordering_violations_marginal"]) == 0
           and vio_t == 0,
           f"queue_ahead 50, two trades of 60: testable="
           f"{rT['ordering_testable']}, E[ProbQueue] "
           f"{rT['E_filled_ProbQueue_f3']} against RiskAverse "
           f"{rT['filled_RiskAverse']}; {vio_t} / 2,000 seeds violate")
        #: THE EXPECTATION COUNTEREXAMPLE, END TO END. queue_ahead 100,
        #: order 10, ONE trade of 110, depth at L 1000.
        bkC, trC, dpC = _synth_day(queue_at_touch=100.0,
                                   depth_at_touch=1000.0,
                                   trades=[(1000, 0, 110.0, True)])
        evC = evaluate_day("ICPUSDT", FIX_DAY, digest, bkC, trC, dpC,
                           tick=0.001, qty_step=10.0)
        rC = _row(evC, hour=0)
        hand = 10.0 * (900.0 ** 3) / (100.0 ** 3 + 900.0 ** 3)
        ck("v7 THE EXPECTATION ALONE IS NOT THE FIX -- KNOWN-BAD FOR THE "
           "REVIEWER'S OWN PROPOSAL, driven through the RUNNER: RiskAverse "
           "fills the whole order while E[ProbQueue] falls short, in the "
           "marginal regime, with no defect present",
           rC is not None and rC["ordering_testable"] is False
           and abs(rC["filled_RiskAverse"] - 10.0) < 1e-9
           and abs(rC["E_filled_ProbQueue_f3"] - hand) < 1e-12
           and rC["E_filled_ProbQueue_f3"] < rC["filled_RiskAverse"] - 1e-9,
           f"RiskAverse {rC['filled_RiskAverse']}, E[ProbQueue] "
           f"{rC['E_filled_ProbQueue_f3']:.15f} against the hand derivation "
           f"{hand:.15f} (900^3/(100^3+900^3) x 10). A.5 proposes the "
           f"expectation as the corrected property; measured, it is true "
           f"EXACTLY on the testable set, so the front = 0 predicate is what "
           f"decides and this control is what stops that being taken on "
           f"trust")
        ovr = D.ordering_verdict(50, 1, 10, 0)
        ovm = D.ordering_verdict(50, 0, 10, 7)
        ck("v7 THE NARROWED TRIGGER, BOTH DIRECTIONS: a TESTABLE-regime "
           "violation still REFUTES; a marginal-regime violation counts and "
           "the gate stays readable",
           ovr["state"] == "REFUTES_THE_BRACKET"
           and ovm["state"] == "ORDERING_HOLDS_WHERE_TESTABLE"
           and ovm["n_violations_in_the_marginal_regime"] == 7,
           "narrowing has not disarmed the falsifier -- it has stopped a "
           "correct model disagreement from silencing a gate")

        # -- 20. DECISION-TIME QUOTE AGE: a label, never an exclusion -------
        r_fresh = _row(evT, hour=0)
        r_stale = _row(evT, hour=3)
        ck("v7 QUOTE AGE is carried PER EPISODE and labels rather than "
           "excludes: a 1 ms quote and a 30-minute quote are BOTH RESOLVED, "
           "and only the second is labelled stale",
           r_fresh["quote_age_ms"] <= STALENESS_BAR_MS
           and r_stale["quote_age_ms"] > STALENESS_BAR_MS
           and evT["label_counts"][LB_STALE] > 0
           and evT["status_counts"][ST_RESOLVED] == len(evT["rows"]),
           f"hour 0 age {r_fresh['quote_age_ms']:.0f} ms, hour 3 age "
           f"{r_stale['quote_age_ms']:.0f} ms, "
           f"{evT['label_counts'][LB_STALE]} labelled of "
           f"{evT['status_counts'][ST_RESOLVED]} resolved -- filtering on "
           f"age would select on ACTIVITY, the defect v6 removed")

        # -- 21. THE GATE READ BOTH WAYS, and the straddle ------------------
        aggA = aggregate_symbol([evT], TP_PRIMARY_S)
        aggB = aggregate_symbol([evT], TP_PRIMARY_S, fresh=True)
        ck("v7 READING B IS A DIFFERENT POPULATION, not a re-label: the "
           "fresh-quote reading carries fewer episodes than reading A",
           (aggB["n_episodes"] or 0) < (aggA["n_episodes"] or 0)
           and aggB["staleness_bar_ms"] == STALENESS_BAR_MS,
           f"reading A {aggA['n_episodes']} episodes, reading B "
           f"{aggB['n_episodes']} at a {STALENESS_BAR_MS:.0f} ms bar")
        st_str = staleness_sensitivity(
            {f"eff_rt_RiskAverse_{PR_RESIDUAL}":
                {"eff_rt_bps": 7.0, "n_days": 1}},
            {f"eff_rt_RiskAverse_{PR_RESIDUAL}":
                {"eff_rt_bps": 9.0, "n_days": 1}})
        st_agr = staleness_sensitivity(
            {f"eff_rt_RiskAverse_{PR_RESIDUAL}":
                {"eff_rt_bps": 7.0, "n_days": 1}},
            {f"eff_rt_RiskAverse_{PR_RESIDUAL}":
                {"eff_rt_bps": 7.5, "n_days": 1}})
        ck("v7 THE STALENESS STRADDLE FIRES AND CAN ALSO ADMIT: 7.0 against "
           "9.0 STRADDLES the 8.0 threshold; 7.0 against 7.5 AGREES",
           st_str["state"] == "STRADDLES_THE_STALENESS_BAR"
           and st_agr["state"] == "AGREES_ACROSS_THE_STALENESS_BAR"
           and st_str["never_averaged"],
           "a rule that could only fire would be a veto, not a rule -- and "
           "the two readings are never averaged")

        # -- 22. QTY_STEP_UNDERDETERMINED, both directions ------------------
        bkQ, trQ, dpQ = _synth_day(trades=[(1000, 0, 5.0, True),
                                           (2000, 0, 5.0, True)])
        evQ = evaluate_day("ICPUSDT", FIX_DAY, digest, bkQ, trQ, dpQ,
                           tick=0.001)
        bkQ3, trQ3, dpQ3 = _synth_day(trades=[(1000, 0, 5.0, True),
                                              (2000, 0, 7.0, True),
                                              (3000, 0, 9.0, True)])
        evQ3 = evaluate_day("ICPUSDT", FIX_DAY, digest, bkQ3, trQ3, dpQ3,
                            tick=0.001)
        ck("v7 QTY_STEP_UNDERDETERMINED KNOWN-BAD: one distinct quantity "
           "REFUSES the day rather than returning a modal diff computed "
           "from it",
           (not evQ["usable"]) and evQ["why"] == ST_QTY_UNDET
           and evQ["n_distinct_quantities"] < MIN_DISTINCT_QUANTITIES,
           f"{evQ['n_distinct_quantities']} distinct quantities < "
           f"{MIN_DISTINCT_QUANTITIES} -> {evQ['why']}")
        ck("v7 QTY_STEP POSITIVE CONTROL: three distinct quantities ESTIMATE "
           "the step and the day runs -- the status is about the estimator, "
           "not a wall",
           evQ3["usable"]
           and evQ3["n_distinct_quantities"] >= MIN_DISTINCT_QUANTITIES,
           f"{evQ3['n_distinct_quantities']} distinct quantities -> "
           f"qty_step {evQ3['qty_step']}")

        # -- 23. THE SEAL is a CHECKED property, not a promise --------------
        sample = {"symbols": {"ADAUSDT": {
            "n_admissible_days": 11,
            "aggregate_by_tp": {"600": {"eff_rt_RiskAverse_residual_chased":
                                        {"eff_rt_bps": 6.1}}},
            "verdict": {"state": "PASS"},
            "status_counts": {"RESOLVED": 500},
            "gate_row_tp_s": 600}}}
        dropped = []
        red = redact_sealed(sample, "", dropped)
        ck("v7 REDACTION removes every economic key and KEEPS the statuses "
           "and the structural metadata",
           "aggregate_by_tp" not in red["symbols"]["ADAUSDT"]
           and "verdict" not in red["symbols"]["ADAUSDT"]
           and red["symbols"]["ADAUSDT"]["status_counts"]["RESOLVED"] == 500
           and red["symbols"]["ADAUSDT"]["gate_row_tp_s"] == 600
           and len(dropped) >= 2,
           f"{len(dropped)} key(s) removed, named in the receipt; "
           f"gate_row_tp_s survives because it names the gate rather than "
           f"carrying a number")
        ck("v7 THE LEAK SCAN FIRES ON A PLANTED ECONOMIC KEY and is CLEAN "
           "on the redacted object -- a seal asserted is not a seal",
           len(find_sealed_leaks(sample)) > 0
           and find_sealed_leaks(red) == [],
           f"{len(find_sealed_leaks(sample))} leak(s) in the raw payload, "
           f"{len(find_sealed_leaks(red))} after redaction. The sealed "
           f"emission REFUSES if this scan is non-empty")

        # -- 24. --no-repro writes the field -------------------------------
        ic_off = inherited_control_block(False)
        ic_on = inherited_control_block(True, {"reproduced": True,
                                               "regimes": ["csv"]})
        ck("v7 --no-repro WRITES {reproduced: null, gates_the_run: false, "
           "why_skipped} instead of going SILENT -- null is not false",
           ic_off["reproduced"] is None
           and ic_off["gates_the_run"] is False
           and bool(ic_off["why_skipped"])
           and ic_on["reproduced"] is True and ic_on["gates_the_run"] is True,
           "the key is ALWAYS present, so a reader can tell 'the control "
           "passed' from 'the control was never asked'")

        # -- 25. R-584: THE STREAMED BOOK ANSWERS WHAT THE WHOLE ONE DOES --
        _, g_l, g_r = episode_grid(FIX_DAY)
        rq = np.random.default_rng(584)
        q_t = np.sort(DAY0_MS - 5_000 + rq.integers(
            0, 90_000_000, size=40_000).astype(np.int64))
        q_b = 100.0 + rq.random(q_t.size)
        q_a = q_b + 0.01
        whole = FullBook(q_t, q_b, q_a)
        chunked = StreamedBook(g_l, g_r)
        for lo in range(0, q_t.size, 4_000):          # 10 "hour files"
            sl = slice(lo, lo + 4_000)
            chunked.ingest(q_t[sl], q_b[sl], q_a[sl])
        same_l = all(whole.before(int(t)) == chunked.before(int(t))
                     for t in g_l)
        same_r = all(whole.at_or_before(int(t)) == chunked.at_or_before(int(t))
                     for t in g_r)
        ck("R-584 STREAMING PARITY: the streamed book returns the IDENTICAL "
           "quote to the whole-day book at every one of the grid's decision "
           "times and T_p ends -- a different reader giving a different "
           "answer is the whole risk of the change",
           same_l and same_r and chunked.t_max == whole.t_max
           and chunked.n_rows == whole.n_rows,
           f"{len(g_l)} decision times and {len(g_r)} T_p ends over 40,000 "
           f"quotes fed in 10 chunks: identical on both rules, t_max "
           f"{chunked.t_max} == {whole.t_max}, {chunked.n_rows} rows")
        off_grid = False
        try:
            chunked.before(int(g_l[0]) + 7)
        except E2ARefused:
            off_grid = True
        ck("R-584 STREAMING KNOWN-BAD: a target the stream was NOT built for "
           "REFUSES instead of returning nothing -- an absent target would "
           "read as NO_QUOTE, which is a wrong EXCLUSION, not a wrong number",
           off_grid,
           "an off-grid timestamp raises E2ARefused; silence there would "
           "have deleted episodes and looked like a quiet tape")

        # -- 26. R-584: THE MEMORY GUARD, BOTH DIRECTIONS ------------------
        admitted = memory_guard("fixture", FIX_DAY)
        refused = False
        try:
            memory_guard("fixture", FIX_DAY, cap=0.0001)
        except E2ARefused:
            refused = True
        ck("R-584 MEMORY GUARD BOTH WAYS: the declared cap ADMITS this "
           "process and a cap below its residency REFUSES -- a guard only "
           "ever shown refusing is a wall, not a bound",
           admitted["cap_gib"] == DAY_RSS_CAP_GIB and refused
           and admitted["rss_now_gib"] < DAY_RSS_CAP_GIB,
           f"resident {admitted['rss_now_gib']} GiB under the declared "
           f"{DAY_RSS_CAP_GIB} GiB cap admits; a 0.0001 GiB cap refuses. THE "
           f"CAP IS NEVER RAISED AND THE POPULATION IS NEVER MADE SMALLER")

        # -- 27. R-584: the gate symbol, both directions -------------------
        ck("R-584 SCOPE both directions: BTCUSDT is the gate symbol and any "
           "other symbol is LABELLED a diagnostic rather than quietly "
           "counted toward a gate",
           GATE_SYMBOL == "BTCUSDT"
           and decl["scope_R584_BTC_ONLY"]["the_gate_symbol"] == GATE_SYMBOL
           and decl["the_ICP_cell_label_v7"]["status_under_R584"]
           == "DEFERRED_NOT_RESOLVED",
           "the thin-name cell is DEFERRED, NOT RESOLVED -- its declared "
           "handling stands, and E2-A under this scope does not answer it")

        # -- 27b. THE SEAL's OWN BLIND SPOT, pinned as a regression --------
        leaky = {"symbols": {"BTCUSDT": {"staleness_sensitivity": {
            "state": "AGREES_ACROSS_THE_STALENESS_BAR",
            "reading_A_all_episodes_bps": 6.1,
            "reading_B_fresh_quotes_bps": 6.4,
            "reading_A_n_episodes": 1200}}}}
        missed = find_sealed_leaks(leaky, "", SEALED_KEY_MARKERS_PRE_FIX)
        caught = find_sealed_leaks(leaky)
        red_leaky = redact_sealed(leaky, "", [])
        ck("THE SEAL's OWN BLIND SPOT, PINNED: the pre-fix marker list saw "
           "ZERO leaks in a block holding two eff_RT values in bps; the "
           "current list catches them and the redaction removes the block",
           missed == [] and len(caught) > 0
           and find_sealed_leaks(red_leaky) == []
           and "staleness_sensitivity" not in red_leaky["symbols"]["BTCUSDT"],
           f"pre-fix scan: {len(missed)} leaks -- it would have certified a "
           f"receipt carrying reading_A_all_episodes_bps. Current scan: "
           f"{len(caught)} leaks ({', '.join(k.split('.')[-1] for k in caught[:3])}). "
           f"A leak scan that shares its blind spot with the redactor "
           f"certifies exactly what it fails to see")

        # -- 27c. THE SECOND NET, cut differently from the first -----------
        unforeseen = {"symbols": {"BTCUSDT": {"diagnostics": {
            "realised_capture_per_episode": 2.4,
            "markout_cents_at_tp": -1.1,
            "n_episodes": 1200}}}}
        net1 = find_sealed_leaks(unforeseen)
        net2 = find_economic_shaped_leaks(unforeseen)
        ck("THE SECOND NET CATCHES WHAT THE MARKER LIST WAS NEVER TOLD "
           "ABOUT: two economic fields under names no marker matches are "
           "caught by NAME SHAPE over numeric leaves, and the emission "
           "refuses on either net",
           net1 == [] and len(net2) == 2
           and find_economic_shaped_leaks(
               {"n_episodes": 5, "wall_s": 1.2, "p50": 29.0}) == [],
           f"marker scan: {len(net1)} leaks. Second net: {len(net2)} "
           f"({', '.join(net2)}). And it does NOT fire on mechanism fields "
           f"(n_episodes, wall_s, p50) -- a net that flagged everything "
           f"would be as useless as one that flagged nothing")

        # -- 28. THE RECEIPT NAMES THE CODE BY CONTENT, NOT ONLY BY COMMIT --
        rid = runner_identity()
        live_digest = hashlib.sha256(
            Path(__file__).resolve().read_bytes()).hexdigest()
        ck("THE RECEIPT CITES THE RUNNER BY CONTENT DIGEST: a per-seat "
           "worktree's HEAD is whatever it was last detached at, and a "
           "rebase can rewrite even a correct commit id -- the digest can do "
           "neither",
           rid["runner_sha256"] == live_digest and len(live_digest) == 64
           and rid["runner_path"].endswith("e2_a_runner.py")
           and isinstance(rid["producing_code_is_the_committed_bytes"], bool),
           f"runner_sha256 {live_digest[:16]}, tree head "
           f"{(rid['tree_head_at_run'] or '')[:7]}, committed-bytes "
           f"{rid['producing_code_is_the_committed_bytes']} -- DA 63 measured "
           f"its own worktree THREE LANDINGS behind while running this "
           f"module, so `carrying_commit` alone would have named code that "
           f"did not run")

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

    # -- 16. RULE 22 / R-605: THE LAUNCH CAPTURE ------------------------
    _idy = source_identity_at_launch()
    ck("RULE 22 / R-605 -- THE LAUNCH CAPTURE IS THE IMPORT CLOSURE, NOT "
       "ONE FILE: at import this run digested every `live/` module in "
       "`sys.modules` plus HEAD and the dirty state, and it re-captures "
       "after a LAZY import so a module that entered mid-run is not left "
       "outside. ***DA 77's sweep found this runner without it -- and "
       "found that this seat's own binding map had EXEMPTED it***",
       _idy["import_closure"]["n_modules"] >= 4
       and _idy["producing_code_sha256"] == LAUNCH_SOURCE_SHA256
       and _idy["closure_unchanged_during_the_run"] is True
       and any("lazy" in c["where"] or "import" in c["where"]
               for c in _idy["import_closure"]["capture_points"]),
       f"{_idy['import_closure']['n_modules']} modules "
       f"{sorted(_idy['import_closure']['modules'])}, HEAD "
       f"{str(_idy['head_at_import']['head'])[:8]}, dirty "
       f"{_idy['head_at_import']['dirty']}")

    _sd = Path(tempfile.mkdtemp(prefix="da78_e2a_closure_"))
    _a, _b = _sd / "sib_one.py", _sd / "sib_two.py"
    _a.write_text("X = 1\n")
    _b.write_text("X = 2\n")
    _syn = {str(_a): hashlib.sha256(_a.read_bytes()).hexdigest(),
            str(_b): hashlib.sha256(_b.read_bytes()).hexdigest()}
    _none = closure_drift(_syn)
    _b.write_text("X = 2  # landed mid-run\n")
    _one = closure_drift(_syn)
    _hold = dict(LAUNCH_CLOSURE)
    LAUNCH_CLOSURE.clear()
    LAUNCH_CLOSURE.update(_syn)
    _msg = ""
    try:
        assert_source_unchanged("a driven emit", fixture=True)
    except LaunchCaptureRefused as _e:
        _msg = str(_e)
    LAUNCH_CLOSURE.clear()
    LAUNCH_CLOSURE.update(_hold)
    _ok = assert_source_unchanged("a driven emit", fixture=True)
    ck("AND IT REFUSES BY MODULE NAME WHEN A SIBLING IS REWRITTEN MID-RUN "
       "AND ADMITS WHEN NOTHING MOVED -- driven on COPIES, never on a "
       "module another seat is working in. ***The run is unaffected; a "
       "receipt stamped from the files would name code that DID NOT RUN, "
       "and `producing_code_is_the_committed_bytes` PASSES when the "
       "replacement is itself committed (R-603)***",
       _none == [] and [d["module"] for d in _one] == ["sib_two.py"]
       and "IMPORT CLOSURE" in _msg and "sib_two.py" in _msg
       and _ok["closure_unchanged_during_the_run"] is True,
       f"nothing moved -> {len(_none)} drift; one rewritten -> "
       f"{[d['module'] for d in _one]} and the emit refuses naming it")

    # -- REV 64: THE PORCELAIN DEFECT HAS TWO NECESSARY HALVES ----------
    _pmp = str(Path(__file__).resolve().parents[1] / "pm_research")
    if _pmp not in sys.path:
        sys.path.insert(0, _pmp)
    import da_root as _PR                                     # noqa: PLC0415
    BLOCK = " M live/x.py\n?? data\nR  a -> b\n"
    good = _PR.parse_porcelain(BLOCK)
    #: HALF ONE, the READ: strip the block first, then parse it correctly.
    strip_first = _PR.parse_porcelain(BLOCK.strip())
    #: HALF TWO, the SLICE: parse the RAW block with the old rule.
    old_slice_raw = [ln[3:] for ln in BLOCK.split("\n") if ln]
    #: BOTH: strip AND the fixed slice -- the defect as it shipped.
    both = [ln[3:] for ln in BLOCK.strip().split("\n") if ln]
    ck("REV 64 -- THE PORCELAIN DEFECT IS ONE DEFECT WITH TWO NECESSARY "
       "HALVES, and the parser closes both. RAW read, the XY code by FIXED "
       "WIDTH, the path from COLUMN 4, and `R  old -> new` split on the "
       "arrow: every path in the reviewer's block is recovered EXACTLY -- "
       "`live/x.py` (leading space), `data`, and `b` from `a`. ***No seat "
       "had both halves right: DE and BE are safe by their READ, this seat "
       "was safe by its SLICE***",
       [r["path"] for r in good["rows"]] == ["live/x.py", "data", "b"]
       and good["rows"][0]["xy"] == " M"
       and good["rows"][2]["renamed_from"] == "a"
       and good["rows"][1]["untracked"] is True
       and good["n_malformed"] == 0,
       f"paths {[r['path'] for r in good['rows']]}; first XY "
       f"{good['rows'][0]['xy']!r}; rename b<-a")
    ck("AND EACH HALF IS DRIVEN RED FIRST, SEPARATELY: ***the READ half "
       "alone*** shifts the FIRST line so a correct parser cannot read it "
       "at all -- its path is LOST, not merely mistyped; "
       "***the SLICE half alone*** is right on a raw line and wrong the "
       "moment one is shifted; and ***the two together*** return "
       "`ive/x.py` for `live/x.py`, which is the character this seat's "
       "receipts actually lost",
       strip_first["n_malformed"] == 1
       and strip_first["malformed"] == ["M live/x.py"]
       and [r["path"] for r in strip_first["rows"]] == ["data", "b"]
       and old_slice_raw[0] == "live/x.py"
       and both[0] == "ive/x.py"
       and both[2] == "a -> b",
       f"strip alone -> the first line is UNREADABLE and its path is LOST "
       f"({strip_first['malformed']}), leaving "
       f"{[r['path'] for r in strip_first['rows']]}; slice alone on a raw "
       f"line -> {old_slice_raw[0]!r}; BOTH -> {both[0]!r}, and the rename "
       f"unsplit as {both[2]!r}")
    ck("AND A LINE THE PARSER CANNOT READ IS NAMED, NOT DROPPED: a "
       "truncated status line lands in `malformed` and is counted as DIRT "
       "by the caller -- an unreadable status is not a clean tree (rule 11)",
       _PR.parse_porcelain("M\n?? ok\n")["n_malformed"] == 1
       and _PR.parse_porcelain("M\n?? ok\n")["rows"][0]["path"] == "ok",
       "a 1-character line is malformed and the good line beside it still "
       "parses")

    # -- 17. the capture meets the SEAL, by SHAPE ------------------------
    _mods = _idy["import_closure"]["modules"]
    _names = [m[0] for m in _mods]
    _collide = [n for n in _names if closure_name_collides_with_the_seal(n)]
    _stamped = {"source_identity": _idy}
    _red = redact_sealed(_stamped, "", [])
    _after = ((_red.get("source_identity") or {}).get("import_closure")
              or {}).get("modules")
    _leaks = find_sealed_leaks(_stamped)
    _econ = find_economic_shaped_leaks(_stamped)
    _planted = {"source_identity": {"import_closure": {
        "modules": _mods, "cost_per_fill": 12.5}}}
    _p_leaks = find_sealed_leaks(_planted)
    _p_econ = find_economic_shaped_leaks(_planted)
    ck("THE CLOSURE IS STAMPED AS [name, digest] PAIRS, SO NO MODULE NAME "
       "IS EVER A KEY -- and the seal's two nets are UNTOUCHED. ***The "
       "first draft of this fix carved the closure out of both nets: a "
       "HOLE in a seal, to solve a problem the seal does not have. The "
       "shape removes the interaction instead.*** The stamp survives "
       "redaction intact, both nets find nothing in it, and an economic "
       "key planted BESIDE it is still caught by name",
       isinstance(_mods, list) and all(len(m) == 2 for m in _mods)
       and _after == _mods and _leaks == [] and _econ == []
       and any("cost_per_fill" in x for x in _p_leaks + _p_econ),
       f"{len(_mods)} pairs survive redaction unchanged; 0 leaks from "
       f"either net; a planted `cost_per_fill` beside them -> "
       f"{sorted(set(_p_leaks + _p_econ))}; module names that WOULD have "
       f"collided as keys: {_collide or 'none today'}")

    tape = watch.tape_paths()
    n_fail = sum(1 for c in checks if not c["passed"])
    receipt = {
        "protocol": PROTOCOL + "_FIXTURE",
        "carrying_commit": carrying_commit(),
        "runner_identity": runner_identity(),
        #: RULE 22 / R-605: the closure, HEAD and dirty state captured at
        #: IMPORT. A fixture RECORDS the dirty state; a real run refuses on
        #: it.
        "source_identity": assert_source_unchanged(
            "the fixture receipt's emit", fixture=True),
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
    return gap_runs(present, int(len(inday)))


def gap_runs(present: np.ndarray, n_quotes_in_day: int) -> dict:
    """The run-length profile of a day's MISSING seconds. ONE implementation.

    v8 leg (d) reads `max_gap_run_s` off this, and so does the census's
    reported profile. Two implementations of a run length -- one streaming
    the hour-files, one over an in-memory day -- would be two definitions of
    the quantity the ruled admission predicate compares against, and nothing
    would notice when they drifted. `outage_measure` builds the same
    presence array from the same clock and the same valid-quote filter and
    calls THIS function.
    """
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
    long_runs = runs[runs >= D.OUTAGE_RUN_S]
    return {
        "n_quotes_in_day": int(n_quotes_in_day),
        "gap_fraction": float(n_missing / 86_400),
        "n_missing_seconds": n_missing,
        "n_gap_runs": int(runs.size),
        "max_gap_run_s": int(runs.max()) if runs.size else 0,
        "median_gap_run_s": float(np.median(runs)) if runs.size else 0.0,
        "share_of_missing_seconds_in_runs_ge_60s":
            float(long_runs.sum() / n_missing) if n_missing else 0.0,
        "n_runs_ge_60s": int(long_runs.size),
        "outage_run_s": int(D.OUTAGE_RUN_S),
        "reading": ("OUTAGE-SHAPED: most missing seconds sit in runs of a "
                    "minute or more"
                    if n_missing and long_runs.sum() / n_missing > 0.5
                    else "QUIET-BOOK-SHAPED: the missing seconds are "
                         "scattered short holes, i.e. seconds in which the "
                         "best quote did not change"),
    }


def outage_measure(sym: str, day: str) -> dict:
    """v8 leg (d)'s input: the day's gap-run profile, STREAMED hour by hour.

    Admission must be cheap. `read_book` concatenates a whole day -- 60 M
    rows on BTC -- and the gate run reads the book again for episodes, so
    measuring the leg that way would read the heaviest day twice inside one
    capped unit. This accumulates only an 86,400-slot presence array, one
    hour-file at a time, off the SAME column (`T`) behind the SAME
    valid-quote filter `read_book` applies, and hands it to `gap_runs`.
    """
    day0 = int(pd.Timestamp(day, tz="UTC").timestamp()) * 1000
    present = np.zeros(86_400, bool)
    n_quotes = 0
    for f in E20._hour_files("bookTicker", sym, day):
        df, _ = E20._read_csv([f], [2, 4, 6], ["T", "bid", "ask"],
                              {2: "int64", 4: "float64", 6: "float64"})
        if df is None or len(df) == 0:
            continue
        t = df["T"].to_numpy()
        bid, ask = df["bid"].to_numpy(), df["ask"].to_numpy()
        t = t[(bid > 0) & (ask > 0) & (ask >= bid)]
        s = (t - day0) // 1000
        s = s[(s >= 0) & (s < 86_400)]
        if s.size:
            present[s.astype(np.int64)] = True
        n_quotes += int(s.size)
    if n_quotes == 0:
        return {"n_quotes_in_day": 0, "gap_fraction": 1.0,
                "max_gap_run_s": 86_400, "streamed": True}
    return dict(gap_runs(present, n_quotes), streamed=True)


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
           "admission_leg": ("v8 -- COLLECTOR LIVENESS (leg b) and the "
                             "TAPE OUTAGE RUN (leg d), neither of them book "
                             "activity"),
           "outage_leg_d": {
               "outage_run_s": D.OUTAGE_RUN_S,
               "forward_window_start_day": D.FORWARD_WINDOW_START_DAY,
               "evaluated_here": not bool(light),
               "why_not_when_light": (
                   "a light census opens no tape, so it has no gap-run "
                   "measurement; the rows say the leg was not evaluated "
                   "rather than admitting without it")},
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
            #: the census is a DIAGNOSTIC table, not the gate population:
            #: the era leg reads recv_ns off every row and is measured only
            #: in `run()`. `require_era=False` makes that explicit in every
            #: row (era_leg_enforced: false) rather than silent.
            #: v8: the census evaluates leg (d) from the SAME profile it
            #: reports, and a LIGHT census -- which opens no tape -- says
            #: the leg was not evaluated instead of admitting without it.
            adm = day_admission(sym, day, counts, health, gap, age,
                                require_era=False,
                                outage=prof, require_outage=not light)
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
    #: RULE 22: a LAZY import was not in `sys.modules` at launch, so it was
    #: outside the closure. REV 53 section 1.1 found exactly that hole in
    #: DE's -- the two modules that do the replaying were outside it.
    capture_closure("after a lazy import of e1_markout_scan")
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
    #: PARITY IS NOT APPLICABLE WHEN THE ROOTS DIFFER, and saying `true`
    #: there would be a verdict that cannot fail sitting beside a table of
    #: mismatches -- which is exactly what the first version of this field
    #: printed: `PARITY ...: true` with `n_mismatches: 12`. None, not True.
    parity_holds = (not mismatches) if same_tree else None
    fix_demonstrated = ((not same_tree) and len(old_empty) == len(syms)
                        and len(new_found) == len(syms)) or None \
        if not same_tree else None
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
           "half_exercised": ("PARITY (roots coincide)" if same_tree
                              else "THE FIX (roots differ)"),
           "PARITY_tick_size_and_file_counts_identical":
               (None if parity_holds is None else bool(parity_holds)),
           "PARITY_not_applicable_here": (not same_tree),
           "n_mismatches": len(mismatches), "mismatched_symbols": mismatches,
           "what_the_mismatches_ARE": (
               "the OLD root seeing nothing where the NEW root sees the "
               "tape -- that IS the fix, not a parity failure"
               if not same_tree else
               "genuine disagreements between old and new resolution on the "
               "same tree; any at all refuse the change"),
           "FIX_old_root_empty_new_root_populated":
               (None if fix_demonstrated is None else bool(fix_demonstrated)),
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
        #: the mechanism check exercises the CODE PATH on a real book and
        #: reads no gate, so it runs on the legs that do not need a tape
        #: read; its receipt already says the era leg was not enforced and
        #: now says the same of leg (d).
        if day_admission(sym, day, counts,
                         collector_health(day, beats, restarts),
                         require_era=False,
                         require_outage=False)["admissible"]:
            adm_days.append(day)
    if not adm_days:
        raise E2ARefused(
            f"REFUSED: {sym} has no admissible day, so there is no real book "
            f"to exercise the path on. An empty population is not a check.")
    per_day = []
    for day in adm_days:
        #: R-584: the mechanism check reads the book the same way the gate
        #: run does. Two book paths in one module is how they drift apart.
        book = stream_book(sym, day, extend=True)
        bmeta = {"streamed": True, "n_rows": book.n_rows,
                 "n_hour_files": book.n_files,
                 "n_bad_quotes": book.n_bad_quotes}
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
            "n_ordering_violations_testable": len(
                ev["ordering_violations_testable"]),
            "n_ordering_violations_marginal": len(
                ev["ordering_violations_marginal"]),
            "n_ordering_testable": ev["n_ordering_testable"],
            "n_ordering_marginal": ev["n_ordering_marginal"],
            "decision_time_quote_age_ms": ev["decision_time_quote_age_ms"],
            "era_leg_enforced_here": False,
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
    #: RULE 22: a LAZY import was not in `sys.modules` at launch, so it was
    #: outside the closure. REV 53 section 1.1 found exactly that hole in
    #: DE's -- the two modules that do the replaying were outside it.
    capture_closure("after a lazy import of e1_markout_scan")
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
    ap.add_argument("--symbols", nargs="*", default=[GATE_SYMBOL])
    ap.add_argument("--min-days", type=int, default=None)
    ap.add_argument("--no-repro", action="store_true")
    ap.add_argument("--sealed", action="store_true",
                    help="R-580(C)(2): run the mechanism, seal every "
                         "economic quantity, read no gate")
    ap.add_argument("--sealed-output", type=Path, default=None)
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
        good = (r["PARITY_tick_size_and_file_counts_identical"] is True
                or r["FIX_old_root_empty_new_root_populated"] is True)
        return 0 if good else 1
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
                  run_repro=not a.no_repro, sealed=a.sealed,
                  sealed_out=a.sealed_output)
        for sym, blk in res["symbols"].items():
            #: NOTHING ECONOMIC IS PRINTED IN A SEALED RUN. The verdict state
            #: is itself a reading of the gate.
            if a.sealed:
                print(f"{sym}: SEALED -- "
                      f"{blk.get('n_admissible_days')} admissible days, "
                      f"gate_read={blk.get('gate_read')}")
            else:
                print(f"{sym}: "
                      f"{blk.get('REFUSED') or blk['verdict']['state']}")
        return 0
    ap.error("choose --selftest/--fixture or --run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
