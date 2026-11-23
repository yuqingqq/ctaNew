"""queue_front.py - QUEUE-FRONT v1 (P-2026-005 MAKER-TRACE; session aea5bfb9, 2026-11-23): WHO holds the front of freshly created price
levels on the 19 BOOK markets (Oct 24-27), HOW they got there (improve / re_establish / deep / at_empty_touch), HOW FAST, and WHETHER front
fills pay - built EXACTLY to predictfun/reviews/maker_trace_20261122/queue_front_contract.md (frozen before computation, commit 6436ee97c).
USER (verbatim in the contract): "... how do they get front in the queue". DESCRIPTIVE on four consumed days (point estimates, no intervals:
G = 4 < 5); p_hi is a within-market label-permutation tail fraction, not independent-day confidence; no fit, no replay, no new data, no model
skill, no profitability claim. PREMISE (stated, not re-tested): price-time priority (v2 report T1 1 / 18,433, T2 5 / 16,433), so the first
fill at a fresh level goes to the order at its front; the holder is the level's creator or an order accepted within the same frame interval
(not distinguished): the unit is "held the front of a fresh level".
INPUTS per BOOK market (MT_RUN=v2; nothing else is opened): chain_context.load_market - usable depth frames (frame_times, level_at_frames;
  NaN = unknown cell), the per-frame best ticks book() uses (M.depth best), the lifecycle fills / orders tables. Every MAKER leg comes from
  the lifecycle fills table with pf_mt_lifecycle's cell convention (YES side y_side, tick y_ptick; only placeable on-grid legs enter a cell)
  and its hard window [x_lo_hard_ms, hi_ms] (bounded windows are exactly [hi - 2 s, hi]: checked; -1 = unbounded, a receipt-hi fill). Its
  maker address and saved markouts come from econ/fills_valued.parquet maker rows joined by (sid, mk_idx) (the lifecycle orders' mk_signer
  when no row exists: counted). chain_post_fill's saved orders.csv (posting windows, the existing queue_class), bound to its receipt.
E1 FRESH LEVEL a candidate is a cell (side, tick) shown > 0 in usable frame f >= 1 and not > 0 in f - 1, whose creation window (upd(f-1),
  upd(f)] opens before the window end we (later candidates are counted, never rows). Across a gap (> 1,000 ms between f - 1 and f) it is a
  status row 'gap', with the cell unknown (NaN) in f - 1 a status row 'unknown' (unit status_only: statuses, not events); otherwise (shown 0
  in f - 1) it is an E1 event.
FRONT WINDOW the level is observed from f to its boundary b, the first later frame not shown > 0 or after a gap: zero (shown 0 again: the
  level ended by upd(b)), gap, unknown (NaN cell) or book_end. MEMBERS = the cell's fills that could have matched inside the level: hi >
  upd(f-1) and x_lo_hard <= T_last (T_last = min(upd(b), we) at a zero boundary, else we); an unbounded window always qualifies. The FIRST
  fill j1 = the member whose hard window starts first (Amendment 1, 5c8a4b1a4: an unbounded window starts first; then the smallest hi, sid,
  mk_idx); the EARLIEST-OVERLAP SET = j1 + every member whose hard window overlaps j1's (x_lo_hard_k <= hi_1; an unbounded member always).
  STATUS, first that applies: no_fill (no member: returned to zero / the window ended without a fill; in every denominator) |
  ambiguous_order (the earliest-overlap set spans >= 2 maker addresses; the address set and its groups are recorded; one maker's own
  overlapping fills are NOT ambiguous) | gap (a gap or the book end between creation and j1's reflection, or a gap after the level's zero frame before hi_1) |
  unknown (an unknown cell before j1; j1 unbounded; j1's window does not start after the creation window opens, x_lo_hard_1 <= upd(f-1);
  j1 straddles the zero frame while a re-creation at the cell opens before hi_1) | unexplained_decrease (a drop in frames (f, E], E =
  min(fa_1, b) at a zero boundary else fa_1, fa_1 = the first frame with upd >= hi_1, not explained by prints: drop - prints > 0.01 sh +
  1e-9; prints = the q of every on-grid fill that may be reflected in that frame, pf_mt_lifecycle._cells' [fb_h + 1, fa] rule) |
  front_identified (holder = the set's one maker address = j1's; group rank1 / rank2 / rank3 / other by the ex-ante top 3).
  READING (contract silent): j1 may straddle the level's zero frame (hi_1 > upd(b), the fill that emptied the level): attributed when the
  cell stays shown 0 with no gap from b until hi_1 (a match after upd(b) would need an order posted and consumed between two frames, never
  shown), else unknown / gap; counted (straddles_end).
AMENDMENT 2 (f4c7e5cbd; the production status rule; the settlement-window rule above survives only as the selftest's settle_only_timing
  mutant): every taker RECORD at a cell (its maker legs there) is timed by the book decrease it caused: candidate frames in its
  [hi - 2 s, hi] (the lifecycle's [fb_h + 1, fa]; never across a gap) whose shown-size decrease equals its cell quantity within 0.01 sh ->
  book_timed (one candidate, claimed by no other record) / book_ambiguous / untimed (book_unmatched, or settlement_only once assigned to a
  creation interval: no visible decrease, window reaching (upd(f-1), upd(f)], x_lo_hard > upd(f-1)); an unbounded window is never timed.
  A level's member records split into OLD life (book-timed at or before f - 1, or every candidate there: excluded, counted), NEW life
  (book-timed in (f, k_end], k_end = b at a zero boundary else b - 1, intervals opening before we; or settlement-only) and unplaced. The
  front = the new life's earliest record (settlement-only first, by hard window; overlapping settlement-only records join it). Records
  that could precede it are competitors (book_ambiguous with an earlier new-life candidate; untimed with a window reaching
  (upd(f-1), upd(k*)]). One maker address over the earliest set and its competitors -> front_identified; several -> ambiguous_order
  (address set kept; a one-record sweep by several makers included); a decrease before k* claimed by no record -> unexplained_decrease;
  no new-life record -> gap / unknown (the observed span ends before we with records after it), unknown (untimed members), else no_fill.
  Behind = later new-life records (and book_ambiguous ones whose every candidate follows k*). READING: records that could precede the
  front make the level ambiguous / unknown, never identified, unless they share the one address (conservative).
BEHIND fills of an identified level = the later members certainly in it: hi <= min(T_last + 2 s, the last frame still 0 before a
  re-creation or gap after b) at a zero boundary, hi <= min(upd(b-1), we + 2 s) otherwise (unattributable members counted). Each fill
  belongs to at most one level. The holder's other fills in the earliest-overlap set are behind fills in window order (fronts.csv in_overlap).
MECHANISM at creation from frame f - 1 (exclusive, precedence re_establish > improve > deep > at_empty_touch > other; the non-exclusive
  flags are kept): re_establish = the cell's last positive frame p < f lies within 5 s of upd(f-1), frames p+1..f-1 are known 0 without a
  gap, and the drop to 0 at p+1 is explained by prints (> 0); improve / deep = strictly better / worse than the own side's best tick in
  f - 1; at_empty_touch = the own side was empty in f - 1; other = the own side's touch was unknown (an unknown cell beyond the best known
  level). READING (contract silent on precedence): re-posting a just-cleared best price is also an improve, so re_establish (the specific
  one) goes first; with the table order it could never occur when the cleared price was the best.
SPEED re_establish: upd(f-1) - upd(p) (bounds [upd(f-1) - upd(p+1), upd(f) - upd(p)] beside); improve: upd(f-1) - upd(g*), g* = the
  previous frame where the side's best changed (known on both sides, no gap; else NaN).
SHARES per scope (pooled / day / market), Amendment 1 bounds over one denominator, identified + ambiguous_order levels: front share
  LOWER = identified levels held by the group, UPPER = LOWER + ambiguous levels whose address set contains the group; fill share = the
  group's share of the fills at the identified levels (front + behind). NULL (declared): within each market the maker labels of those fills are
  permuted, 500 permutations, Generator(PCG64(SeedSequence([20261123, day, mid]))) per market (a pilot market carries the full run's
  draws); p_hi = (1 + #{null front count >= observed}) / 501. Supported market: >= 5 front_identified levels.
ECONOMICS GROSS markouts v_1s / v_10s / v_60s from settle_ms (c/sh = 100 sum v / sum q) on chain_post_fill's PNL cohort per fill (status
  ok, not a self-trade, st_res in pf_mt_econ.RES_VALID, st_1s / st_10s / st_60s ok; excluded by first failing reason, counted): the front
  fill vs the behind fills of the same levels, per holder group and mechanism (by='holder'), and per filling maker's group (by='filler').
JOINED (counted beside): every top-3 order of chain_post_fill's orders.csv: joined_existing_queue when the cell is shown > 0 in the frame
  in force at its union posting lower end post_lo_union_ms (chain_context.level_size, fresh <= 1 s), cell_empty_before_window when shown 0,
  cell_unknown / window_unbounded otherwise; with the existing queue_class and whether the hash is an identified front holder.
READINGS (claims.json, computed; rank1 and top3 pooled; else unidentified / insufficient_support): wins_front_more_than_fills (front share
  LOWER bound > fill share, p_hi <= 0.05, lower > fill in >= 2/3 of >= 5 supported markets; the upper bound beside); main_mechanism,
  faster_than_others and front_pays on front_identified levels only: main_mechanism (largest share of the group's front wins,
  every share listed; >= 5 wins and >= 5 supported markets; a tie is unidentified); faster_than_others (pooled median re_establish speed
  below the other identified holders' (outside the group), and in >= 2/3 of the >= 5 supported markets where both medians exist);
  front_pays (pooled front v_10s c/sh > behind v_10s c/sh at the group's levels, and in >= 2/3 of the >= 5 supported markets with both
  cohorts). READING: markets lacking a group's re_establish fronts or a cohort are not evaluable and leave the 2/3 denominator.
OUTPUTS write-once (a partial directory renamed at the end): levels.csv, fronts.csv, shares.csv, mechanism.csv (front wins, the fresh-level
  mix by status, joined), speed.csv, economics.csv, consistency.csv, claims.json, receipt.json (sha256 of sources, inputs and outputs; CPU
  by stage; VmHWM). run -> v2/chains/queue_front; run --pilot <day> --out <dir> -> the pilot (prints the full-run CPU projection).
GUARDS MT_RUN=v2; measurement days only (refused before any path is built); MT_QF_MUT (the selftest's planted errors first_fill_by_settle /
  ignore_decrease / drop_no_fill / same_maker_ambiguous / settle_only_timing / ignore_old_life) is refused in production; the full run
  refuses unless the tool, its selftest, the contract and chain_context.py are tracked and unmodified (the freeze) and the recorded pilot
  projection (the amended tool's pilot, queue_front_pilot_20261024_r3) is <= 120 CPU s; fills_valued sha256 must
  equal the accepted binding and chain_post_fill's receipt, orders.csv its receipt's sha256; VmHWM self-guard 1,450 MiB (never raised);
  CPU self-guard.
Usage (run_light, MT_RUN=v2): queue_front.py run | queue_front.py run --pilot 20261024 --out <dir>      selftest: queue_front_selftest.py"""
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pf_mt_common as C  # noqa: E402
import pf_mt_econ as E  # noqa: E402
import pf_mt_lifecycle as LC  # noqa: E402
import chain_context as CX  # noqa: E402

need, Forbidden = C.need, C.Forbidden
NAN = float('nan')
VERSION = 'queue_front_v1'
CONTRACT_FILE = C.RECEIPT_V1 / 'queue_front_contract.md'
CONTRACT_COMMIT = '6436ee97c'
AMENDMENT_COMMIT = ('5c8a4b1a4', 'f4c7e5cbd')    # 1: one-address front sets, front-share bounds; 2: records timed by the book, old / new life
PILOT_RUN = 'r3'                                 # the pilot of the amended tool (r1 = the original rule, r2 = amendment 1; both kept)
TOOL_FILES = (HERE / 'queue_front.py', HERE / 'queue_front_selftest.py', CONTRACT_FILE, HERE / 'chain_context.py')
DEP_FILES = (RESEARCH / 'pf_mt_common.py', RESEARCH / 'pf_mt_econ.py', RESEARCH / 'pf_mt_lifecycle.py', RESEARCH / 'pf_mt_decision.py',
             HERE / 'chain_post_fill.py')
TOP3 = tuple(CX.TOP3)
TOP3_L = tuple(a.lower() for a in TOP3)
N_BOOK = 19
PILOT_DAY = '20261024'
FV_SHA256 = '2a874707cc7801d6760e2de36075870700cc0be8e72cd9a6df0213424221cde6'   # the accepted binding (chain_post_fill / matched_gap_r1)

# ------------------------------------------------------------------ declared constants (the contract's; none fitted) ------------------------------------------------------------------
NT = int(CX.NT)                                  # 101 ticks
NC = 2 * NT                                      # cells: side * 101 + tick (side 0 bid / 1 ask, YES space)
GAP_MS = int(CX.DEPTH_GAP_MS)                    # 1,000 ms
MAX_LAG_MS = int(CX.MAX_LAG_MS)                  # 2,000 ms: bounded hard windows are [hi - 2 s, hi]
TOL = float(LC.TOL_SH) + 1e-9                    # an unexplained drop exceeds the prints by > 0.01 sh (+ float slack, as LC._tol)
REEST_MS = 5_000                                 # re_establish: the cell was > 0 within 5 s before creation
SEED, N_PERM = 20261123, 500
MIN_FRONTS, MIN_MARKETS, MIN_WINS = 5, 5, 5      # supported market; supported markets per reading; a group's wins for main_mechanism
SHARE_NUM, SHARE_DEN = 2, 3
P_MAX = 0.05
GROUPS = ('rank1', 'rank2', 'rank3', 'other')
SCOPE_GROUPS = GROUPS + ('top3',)
GNAMES = dict(rank1=('rank1',), rank2=('rank2',), rank3=('rank3',), other=('other',), top3=('rank1', 'rank2', 'rank3'))
GCODES = dict(rank1=(0,), rank2=(1,), rank3=(2,), other=(3,), top3=(0, 1, 2))
READ_SCOPES = ('rank1', 'top3')
READINGS = ('wins_front_more_than_fills', 'main_mechanism', 'faster_than_others', 'front_pays')
STATUSES = ('front_identified', 'ambiguous_order', 'unexplained_decrease', 'no_fill', 'gap', 'unknown')
MECHS = ('improve', 're_establish', 'deep', 'at_empty_touch', 'other')
FLAGS = ('is_improve', 'is_re_establish', 'is_deep', 'is_at_empty_touch', 'touch_unknown')
JOINED_ST = ('joined_existing_queue', 'cell_empty_before_window', 'cell_unknown', 'window_unbounded', 'cell_not_placeable')
HORIZONS = ('1s', '10s', '60s')
MUTANTS = ('first_fill_by_settle', 'ignore_decrease', 'drop_no_fill', 'same_maker_ambiguous', 'settle_only_timing', 'ignore_old_life')
REC_ST = ('book_timed', 'book_ambiguous', 'book_unmatched', 'settlement_only')   # amendment 2: record timing statuses
GBIT = dict(rank1=1, rank2=2, rank3=4, other=8, top3=7)   # amb_gmask bits: the groups in an ambiguous level's address set
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
CPU_CAP_RUN_S, CPU_CAP_PILOT_S = 150., 16.
PROJ_MAX_S = 120.
CSV_FILES = ('levels.csv', 'fronts.csv', 'shares.csv', 'mechanism.csv', 'speed.csv', 'economics.csv', 'consistency.csv')
OUT_FILES = CSV_FILES + ('claims.json', 'receipt.json')
LIFE_COLS = ('sid', 'mk_idx', 'mk_hash', 'y_side', 'y_ptick', 'shares', 'hi_ms', 'x_lo_hard_ms', 'status')
FV_COLS = ('day', 'mid', 'sid', 'mk_idx', 'role', 'addr', 'hash', 'y_side', 's', 'q', 'p', 'self_trade', 'settle_ms', 'status', 'st_res',
           'st_1s', 'v_1s', 'st_10s', 'v_10s', 'st_60s', 'v_60s')
PF_DTYPES = dict(day=str, mid=np.int64, rank=np.int64, addr=str, mk_hash=str, y_side=str, tick=np.int64, n_fills=np.int64,
                 post_lo_union_ms=np.int64, post_hi_ms=np.int64, window_status=str, queue_class=str)
KEY = np.int64(1) << np.int64(40)                # (cell, ms) search keys: cell * 2^40 + (ms - base)
BIG = np.int64(1) << np.int64(60)
ZERO_B, GAP_B, UNK_B, END_B = 0, 1, 2, 3
END_NAMES = np.array(['zero', 'gap', 'unknown', 'book_end'], dtype=object)
LEVEL_COLS = ('day', 'mid', 'level_id', 'side', 'tick', 'frame', 'fid', 't_open_ms', 't_create_ms', 'phase', 'unit', 'status', 'reason',
              'end_type', 'end_ms', 'size_created', 'n_members', 'n_unbounded', 'holder', 'holder_group', 'front_sid', 'front_mk_idx',
              'front_hash', 'front_hi_ms', 'front_lo_ms', 'straddles_end', 'n_behind', 'n_unattributed', 'n_overlap', 'overlap_one_address',
              'amb_addresses', 'amb_gmask', 'n_unexplained', 'front_timing', 'front_book_ms', 'n_old_life', 'mechanism', 'is_improve',
              'is_re_establish', 'is_deep', 'is_at_empty_touch', 'touch_unknown', 'best_prev_tick', 'speed_ms', 'speed_lo_ms', 'speed_hi_ms')
FRONT_COLS = ('day', 'mid', 'level_id', 'side', 'tick', 'role', 'k', 'sid', 'mk_idx', 'mk_hash', 'addr', 'group', 'holder_group', 'mechanism',
              'q', 'hi_ms', 'lo_ms', 'in_overlap', 'null_front', 'cohort', 'excl_reason', 'q_c', 'v_1s', 'v_10s', 'v_60s')
FILL_JOIN_KEYS = ('legs', 'legs_grid', 'legs_offgrid', 'legs_unbounded', 'legs_with_fv_row', 'legs_addr_from_orders', 'legs_no_address',
                  'cell_fv_vs_life_mismatch', 'q_fv_vs_life_mismatch', 'addr_fv_vs_orders_mismatch', 'fv_rows_without_life_leg',
                  'fv_top3_rows_without_life_leg')
JOIN_COLS = ('day', 'mid', 'group', 'addr', 'mk_hash', 'y_side', 'tick', 'n_fills', 'post_lo_union_ms', 'post_hi_ms', 'window_status',
             'queue_class', 'joined_status', 'size_before', 'is_front_holder')


# ------------------------------------------------------------------ small helpers ------------------------------------------------------------------
def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(C.REPO))
    except ValueError:
        return str(p)


def _sha(path):
    return E.sha256(Path(path))


def _clean(o):
    """A JSON-safe copy (NaN / inf -> None, numpy scalars -> Python, sets -> sorted lists, keys -> str)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return sorted(_clean(v) for v in o)
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def _json(path, obj):
    Path(path).write_text(json.dumps(_clean(obj), indent=1, sort_keys=True, allow_nan=False) + '\n')


def hwm():
    return float(LC.vmhwm_mib())


def guard(where, lim=C.GUARD_MIB):
    m = hwm()
    need(m < lim, f'memory self-guard: VmHWM {m:.0f} MiB >= {lim:.0f} MiB at {where} (refused)')
    return m


def _git(files):
    """HEAD and whether the tool files are tracked and unmodified (the freeze)."""
    try:
        g = ['git', '-C', str(C.REPO)]
        head = subprocess.run(g + ['rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30).stdout.strip() or None
        rel = [str(Path(f).resolve().relative_to(C.REPO)) for f in files]
        tracked = subprocess.run(g + ['ls-files', '--error-unmatch', '--'] + rel, capture_output=True, text=True, timeout=30).returncode == 0
        st = subprocess.run(g + ['status', '--porcelain', '--'] + rel, capture_output=True, text=True, timeout=30)
        return dict(head=head, tool_files=rel, tracked=tracked, unmodified=bool(tracked and st.returncode == 0 and not st.stdout.strip()))
    except (OSError, ValueError, subprocess.SubprocessError):
        return dict(head=None, tool_files=None, tracked=None, unmodified=None)


def mutant():
    """The selftest's planted-error switch MT_QF_MUT, read at call time; production refuses any value."""
    m = os.environ.get('MT_QF_MUT', '')
    need(m == '' or m in MUTANTS, f'MT_QF_MUT={m!r} is not a declared mutant {MUTANTS} (refused)')
    return m


def paths():
    """Run-time paths on the MT_RUN roots (v2: C.RECEIPT = the receipt's v2/ directory)."""
    ch = C.RECEIPT / 'chains'
    return dict(out=ch / 'queue_front', pilot_receipt=ch / f'queue_front_pilot_{PILOT_DAY}_{PILOT_RUN}' / 'receipt.json', post_fill=ch / 'post_fill')


def pilot_out(day):
    return C.RECEIPT / 'chains' / f'queue_front_pilot_{day}_{PILOT_RUN}'


def _first(conds, names, default, n):
    """The first condition that holds wins (object array of names)."""
    out = np.full(n, default, dtype=object)
    for cnd, nm in reversed(list(zip(conds, names))):
        out[np.asarray(cnd, bool)] = nm
    return out


def _acc_next(mask, F):
    """(F, K) bool -> per column the first row index >= r where mask holds (F when none)."""
    ar = np.arange(F, dtype=np.int32).reshape((F,) + (1,) * (mask.ndim - 1))
    x = np.where(mask, ar, np.int32(F))
    return np.minimum.accumulate(x[::-1], axis=0)[::-1]


def _acc_last(mask, F):
    """(F, ...) bool -> per column the last row index <= r where mask holds (-1 when none)."""
    ar = np.arange(F, dtype=np.int32).reshape((F,) + (1,) * (mask.ndim - 1))
    x = np.where(mask, ar, np.int32(-1))
    return np.maximum.accumulate(x, axis=0)


def _med(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else NAN


def _qd(x, pre):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    out = {f'{pre}_n': int(len(x)), f'{pre}_mean': float(x.mean()) if len(x) else NAN}
    for q in QS:
        out[f'{pre}_p{int(round(q * 100)):02d}'] = float(np.quantile(x, q)) if len(x) else NAN
    return out


def _cat(frames, cols):
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame({c: [] for c in cols})
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ inputs ------------------------------------------------------------------
def read_fv_makers(path, markets, cnt):
    """fills_valued MAKER rows (every address, every status) of the markets [(day, mid)], read through the row-group mid statistics
    (chain_post_fill.read_fv's pattern); every row must carry its market's day and (mid, sid, mk_idx) must be unique, else refused."""
    path = Path(path)
    need(path.is_file(), f'input missing: {path} (refused)')
    mids = np.array(sorted({int(m) for _, m in markets}), np.int64)
    day_of = {int(m): str(d) for d, m in markets}
    pf = pq.ParquetFile(path)
    md = pf.metadata
    need(md.num_row_groups > 0, 'fills_valued: no row group (refused)')
    names = [md.row_group(0).column(k).path_in_schema for k in range(md.num_columns)]
    miss = [c for c in FV_COLS if c not in names]
    need(not miss, f'fills_valued lacks {miss} (refused)')
    jm = names.index('mid')
    vm = pa.array(mids)
    parts = []
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(jm).statistics
        if st is not None and st.has_min_max:
            k = int(np.searchsorted(mids, int(st.min), 'left'))
            if k >= len(mids) or mids[k] > int(st.max):
                cnt['fv_groups_pruned'] += 1
                continue
        tb = pf.read_row_group(i, columns=list(FV_COLS))
        cnt['fv_groups_read'] += 1
        cnt['fv_rows_scanned'] += tb.num_rows
        tb = pa.table({c: (tb.column(c).cast(tb.column(c).type.value_type) if pa.types.is_dictionary(tb.column(c).type) else tb.column(c))
                       for c in FV_COLS})
        msk = pc.fill_null(pc.and_(pc.is_in(tb.column('mid'), value_set=vm), pc.equal(tb.column('role'), 'maker')), False)
        if pc.any(msk).as_py():
            parts.append(tb.filter(msk))
        del tb
    if parts:
        df = pa.concat_tables(parts).to_pandas()
    else:
        df = pd.DataFrame({c: pd.Series(dtype=object) for c in FV_COLS})
    if len(df):
        exp = np.array([day_of[int(m)] for m in df['mid'].to_numpy(np.int64)], dtype=object)
        got = np.array([str(x) for x in df['day'].to_numpy(dtype=object)], dtype=object)
        need(bool((exp == got).all()), f'fills_valued: {int((exp != got).sum())} maker rows of a BOOK market under another day (refused)')
        key = pd.MultiIndex.from_arrays([df['mid'].to_numpy(np.int64), df['sid'].to_numpy(dtype=object), df['mk_idx'].to_numpy(np.int64)])
        need(key.is_unique, 'fills_valued: a maker leg (mid, sid, mk_idx) twice (refused)')
    df['addr_l'] = np.array([str(a).lower() for a in df['addr'].to_numpy(dtype=object)], dtype=object)
    cnt['fv_maker_rows'] = int(len(df))
    return df


def read_post_fill(pf_dir, markets, hashes, fv_sha):
    """chain_post_fill's saved orders.csv (every top-3 filled order) bound to its receipt: the file's sha256 must be the receipt's output, the
    receipt's fills_valued the accepted binding, the row count the receipt's, every rank's address the ex-ante top 3 -> the run's rows."""
    pf_dir = Path(pf_dir)
    po, pr = pf_dir / 'orders.csv', pf_dir / 'receipt.json'
    need(po.is_file() and pr.is_file(), f'input missing: {po} / {pr} (refused)')
    R = json.loads(pr.read_text())
    so = _sha(po)
    hashes[_rel(po)] = so
    hashes[_rel(pr)] = _sha(pr)
    need(((R.get('outputs') or {}).get('orders.csv') or {}).get('sha256') == so,
         "chain_post_fill orders.csv differs from its receipt's sha256 (refused: mismatched artifacts)")
    rfv = ((R.get('inputs') or {}).get('econ/fills_valued.parquet') or {}).get('sha256')
    need(rfv == fv_sha, f"chain_post_fill's fills_valued {str(rfv)[:12]} differs from the binding {str(fv_sha)[:12]} (refused: mismatched artifacts)")
    hdr = pd.read_csv(po, nrows=0).columns
    miss = [c for c in PF_DTYPES if c not in hdr]
    need(not miss, f'chain_post_fill orders.csv lacks {miss} (refused)')
    D = pd.read_csv(po, usecols=list(PF_DTYPES), dtype=PF_DTYPES)
    need(len(D) == int((R.get('rows') or {}).get('orders', -1)), "chain_post_fill orders.csv rows differ from its receipt (refused)")
    rk = D['rank'].to_numpy(np.int64)
    ad = np.array([str(a).lower() for a in D['addr'].to_numpy(dtype=object)], dtype=object)
    need(bool(np.isin(rk, (1, 2, 3)).all()), 'chain_post_fill orders.csv: a rank outside 1-3 (refused)')
    need(all(bool((ad[rk == r] == TOP3_L[r - 1]).all()) for r in (1, 2, 3)), "chain_post_fill orders.csv: an address differs from its rank's top-3 address (refused)")
    keep = {(str(d), int(m)) for d, m in markets}
    sel = np.array([(str(d), int(m)) in keep for d, m in zip(D['day'].to_numpy(dtype=object), D['mid'].to_numpy(np.int64))], bool)
    return D[sel].reset_index(drop=True), dict(rows=int(len(D)), rows_run=int(sel.sum()), fills_valued=rfv)


# ------------------------------------------------------------------ one market ------------------------------------------------------------------
def market_book(M):
    """The usable depth frames of one BOOK market as (F, 202) cells (cell = side * 101 + tick, side 0 bid / 1 ask: pf_mt_lifecycle._book's
    YES-space convention) through chain_context.frame_times / level_at_frames (checked equal to the lifecycle book), the per-frame best ticks
    chain_context.book() reads, and the gap flags (> 1,000 ms; checked equal to the lifecycle's)."""
    need(M.depth is not None, f'market {M.day} {M.mid}: no depth (not a BOOK market; refused)')
    B = M.depth['B']
    t = np.asarray(CX.frame_times(M), np.int64)
    F = int(len(t))
    if F:
        fr = np.repeat(np.arange(F, dtype=np.int64), NC)
        sd = np.tile(np.repeat(np.array([0, 1], np.int64), NT), F)
        tk = np.tile(np.arange(NT, dtype=np.int64), 2 * F)
        S = np.asarray(CX.level_at_frames(M, fr, sd, tk), np.float64).reshape(F, NC)
        del fr, sd, tk
    else:
        S = np.zeros((0, NC))
    need(np.array_equal(S.reshape(F, 2, NT), np.asarray(B['shown'], np.float64), equal_nan=True),
         f'market {M.day} {M.mid}: level_at_frames differs from the lifecycle book (refused: cell convention)')
    gap = np.zeros(F, bool)
    if F > 1:
        gap[1:] = np.diff(t) > GAP_MS
    need(np.array_equal(gap, np.asarray(B['gap'], bool)), f'market {M.day} {M.mid}: gap flags differ from the lifecycle (refused)')
    best = np.vstack([np.asarray(M.depth['best']['bid'], np.float64), np.asarray(M.depth['best']['ask'], np.float64)]).reshape(2, F)
    return dict(t=t, S=S, gap=gap, best=best, fid=np.asarray(B['fid'], np.int64), F=F)


def orders_signer(O, hashes):
    """mk_hash -> the lifecycle orders' mk_signer (S_NOPRIO rows; lowercase; '' when absent)."""
    out = np.full(len(hashes), '', dtype=object)
    if O is None or not len(O) or not len(hashes):
        return out
    sel = O['scenario'].to_numpy(dtype=object) == 'S_NOPRIO'
    hs = O['mk_hash'].to_numpy(dtype=object)[sel]
    if not len(hs):
        return out
    sg = np.array([str(x).lower() for x in O['mk_signer'].to_numpy(dtype=object)[sel]], dtype=object)
    ix = pd.Index(hs)
    need(ix.is_unique, 'lifecycle orders: a hash twice among the S_NOPRIO rows (refused)')
    j = np.asarray(ix.get_indexer(np.asarray(hashes, dtype=object)), np.int64)
    out[j >= 0] = sg[j[j >= 0]]
    return out


def market_fills(M, fvm):
    """Every maker leg of the market (the lifecycle fills table: all makers) with its cell, hard window, maker address, top-3 group and
    saved markouts (fills_valued maker row by (sid, mk_idx); chain_post_fill's PNL cohort rule) -> (fills dict, counts)."""
    L = M.life_fills
    need(L is not None, f'market {M.day} {M.mid}: no lifecycle fills table (refused)')
    for c_ in LIFE_COLS:
        need(c_ in L.columns, f'lifecycle fills {M.day} {M.mid}: column {c_} missing (refused)')
    n = int(len(L))
    sid = L['sid'].to_numpy(dtype=object)
    mk_idx = L['mk_idx'].to_numpy(dtype=np.int64)
    mk_hash = L['mk_hash'].to_numpy(dtype=object)
    ys = L['y_side'].to_numpy(dtype=object)
    tick = L['y_ptick'].to_numpy(dtype=np.int64)
    q = L['shares'].to_numpy(dtype=np.float64)
    hi = L['hi_ms'].to_numpy(dtype=np.int64)
    lo = L['x_lo_hard_ms'].to_numpy(dtype=np.int64)
    fst = [set(str(x).split('+')) for x in L['status'].to_numpy(dtype=object)]
    side = np.where(ys == 'bid', 0, np.where(ys == 'ask', 1, -1)).astype(np.int64)
    offgrid = np.array([('unplaceable' in s_) or ('subcent_price' in s_) for s_ in fst], bool) if n else np.zeros(0, bool)
    grid = (side >= 0) & (tick >= 0) & (tick < NT) & np.isfinite(q) & (q > 0) & ~offgrid
    bnd = lo >= 0
    need(bool((hi > 0).all()), f'lifecycle fills {M.day} {M.mid}: a leg without a positive hi_ms (refused)')
    need(bool((lo[bnd] == hi[bnd] - MAX_LAG_MS).all()), f'lifecycle fills {M.day} {M.mid}: a bounded window is not [hi - 2 s, hi] (refused)')
    cnt = Counter(legs=n, legs_grid=int(grid.sum()), legs_offgrid=int((~grid).sum()), legs_unbounded=int((~bnd).sum()))
    pos = np.full(n, -1, np.int64)
    if fvm is not None and len(fvm) and n:
        kf = pd.MultiIndex.from_arrays([fvm['sid'].to_numpy(dtype=object), fvm['mk_idx'].to_numpy(dtype=np.int64)])
        need(kf.is_unique, f'fills_valued {M.day} {M.mid}: a maker leg twice (refused)')
        pos = np.asarray(kf.get_indexer(pd.MultiIndex.from_arrays([sid, mk_idx])), np.int64)
    has = pos >= 0
    osig = orders_signer(M.orders, mk_hash)
    addr = osig.copy()
    reason = np.full(n, 'no_fv_row', dtype=object)
    v = {h: np.full(n, NAN) for h in HORIZONS}
    qf = np.full(n, NAN)
    if has.any():
        G = fvm.iloc[pos[has]]
        addr[has] = G['addr_l'].to_numpy(dtype=object)
        st_ok = G['status'].to_numpy(dtype=object) == 'ok'
        self_ = G['self_trade'].to_numpy(dtype=bool)
        res = G['st_res'].isin(list(E.RES_VALID)).to_numpy(dtype=bool)
        mh = {h: G[f'st_{h}'].to_numpy(dtype=object) == 'ok' for h in HORIZONS}
        reason[has] = np.where(~st_ok, 'status', np.where(self_, 'self_trade', np.where(~res, 'st_res', np.where(
            ~mh['1s'], 'st_1s', np.where(~mh['10s'], 'st_10s', np.where(~mh['60s'], 'st_60s', 'ok'))))))
        for h in HORIZONS:
            v[h][has] = G[f'v_{h}'].to_numpy(dtype=np.float64)
        qf[has] = G['q'].to_numpy(dtype=np.float64)
        fp = G['p'].to_numpy(dtype=np.float64)
        with np.errstate(invalid='ignore'):
            ftk = np.where(np.isfinite(fp), np.rint(fp * 100.), -1.).astype(np.int64)
        fys = G['y_side'].to_numpy(dtype=object)
        cnt['cell_fv_vs_life_mismatch'] = int(((fys != ys[has]) | (ftk != tick[has])).sum())
        cnt['q_fv_vs_life_mismatch'] = int((np.abs(qf[has] - q[has]) > 1e-9).sum())
        both = has & (osig != '')
        cnt['addr_fv_vs_orders_mismatch'] = int((addr[both] != osig[both]).sum())
    cnt['legs_with_fv_row'] = int(has.sum())
    cnt['legs_addr_from_orders'] = int((~has & (osig != '')).sum())
    cnt['legs_no_address'] = int((addr == '').sum())
    if fvm is not None and len(fvm):
        cnt['fv_rows_without_life_leg'] = int(len(fvm) - has.sum())
        cnt['fv_top3_rows_without_life_leg'] = int(np.isin(fvm['addr_l'].to_numpy(dtype=object), TOP3_L).sum()
                                                   - np.isin(addr[has], TOP3_L).sum())
    grp = np.full(n, 3, np.int8)
    for k, a in enumerate(TOP3_L):
        grp[addr == a] = k
    coh = reason == 'ok'
    fl = dict(n=n, sid=sid, mk_idx=mk_idx, mk_hash=mk_hash, side=side, tick=tick, q=q, hi=hi, lo=lo, grid=grid, addr=addr, grp=grp,
              reason=reason, cohort=coh, q_c=np.where(coh, qf, 0.), **{f'v_{h}': np.where(coh, v[h], NAN) for h in HORIZONS})
    return fl, cnt


def _prints(t, cell, q, lo, hi, F):
    """pf_mt_lifecycle._cells' print rule: a fill may be reflected in frames [max(fb_h + 1, 1), min(fa, F - 1)], fb_h = the last frame before
    x_lo_hard (-1 unbounded), fa = the first frame with upd >= hi -> (F, NC) summed shares."""
    d = np.zeros((F + 1, NC))
    if len(cell) and F:
        fb = np.where(lo >= 0, np.searchsorted(t, lo, 'left') - 1, -1)
        fa = np.searchsorted(t, hi, 'left')
        st_, en_ = np.maximum(fb + 1, 1), np.minimum(fa, F - 1)
        m = st_ <= en_
        np.add.at(d, (st_[m], cell[m]), q[m])
        np.add.at(d, (en_[m] + 1, cell[m]), -q[m])
    return np.cumsum(d[:F], axis=0)


def _records(t, S, gap, F, fl, g, cellA, sidc, acode, base):
    """Amendment 2: every taker RECORD at a cell (sid x cell: its maker legs there; one shared hard window) timed by the book decrease it
    caused: candidate frames k in [max(fb_h + 1, 1), min(fa, F - 1)] (the record's [hi - 2 s, hi], no lag past hi; not across a gap) whose
    shown-size decrease equals the record's cell quantity within 0.01 sh -> book_timed (one candidate, claimed by no other record) /
    book_ambiguous (several candidates, or several records claiming one decrease) / untimed (none; an unbounded window is never timed).
    Also: claimed (F, NC), the cumulative count of decreases claimed by no record (cumUX), the record lists sorted by (cell, hi)."""
    hi, lo = fl['hi'], fl['lo']
    gi = np.flatnonzero(g)
    if len(gi):
        ruq, rinv = np.unique(sidc[gi] * NC + cellA[gi], return_inverse=True)
        rinv = np.asarray(rinv, np.int64).ravel()
    else:
        ruq, rinv = np.zeros(0, np.int64), np.zeros(0, np.int64)
    NR = int(len(ruq))
    rc, rsid = (ruq % NC).astype(np.int64), (ruq // NC).astype(np.int64)
    legs = gi[np.lexsort((fl['mk_idx'][gi], rinv))] if len(gi) else np.zeros(0, np.int64)
    roff = np.r_[0, np.cumsum(np.bincount(rinv, minlength=NR))].astype(np.int64)
    rq = np.bincount(rinv, weights=fl['q'][gi], minlength=NR).astype(np.float64) if NR else np.zeros(0)
    rhi, rhmin = np.full(NR, -1, np.int64), np.full(NR, np.iinfo(np.int64).max, np.int64)
    rbnd, rgb = np.ones(NR, bool), np.zeros(NR, np.int64)
    if NR:
        np.maximum.at(rhi, rinv, hi[gi])
        np.minimum.at(rhmin, rinv, hi[gi])
        rbnd[rinv[lo[gi] < 0]] = False
        np.bitwise_or.at(rgb, rinv, np.left_shift(np.int64(1), fl['grp'][gi].astype(np.int64)))
    rlo = np.where(rbnd, rhmin - MAX_LAG_MS, -1)
    fb = np.where(rbnd, np.searchsorted(t, rlo, 'left') - 1, -1)
    ks, ke = np.maximum(fb + 1, 1), np.minimum(np.searchsorted(t, rhi, 'left'), F - 1)
    rr = np.flatnonzero(rbnd & (ks <= ke))
    Lr = (ke[rr] - ks[rr] + 1).astype(np.int64)
    tot = int(Lr.sum())
    rep = np.repeat(rr, Lr)
    kk = np.repeat(ks[rr], Lr) + (np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(Lr) - Lr, Lr))
    cc = rc[rep]
    with np.errstate(invalid='ignore'):
        dv = S[kk - 1, cc] - S[kk, cc]
        cand = (dv > 0) & (np.abs(dv - rq[rep]) <= TOL) & ~gap[kk]
    cr, ck = rep[cand], kk[cand]
    ncand = np.bincount(cr, minlength=NR) if NR else np.zeros(0, np.int64)
    coff = np.r_[0, np.cumsum(ncand)].astype(np.int64)
    claimed = np.zeros((F, NC), bool)
    rtimed, rk = np.zeros(NR, bool), np.full(NR, -1, np.int64)
    if len(ck):
        claimed[ck, rc[cr]] = True
        _, cinv, ccnt = np.unique(ck * NC + rc[cr], return_inverse=True, return_counts=True)
        ncl = ccnt[np.asarray(cinv, np.int64).ravel()]
        fi = np.minimum(coff[:-1], len(ck) - 1)
        rtimed = (ncand == 1) & (ncl[fi] == 1)
        rk = np.where(rtimed, ck[fi], -1)
    ramb = (ncand >= 2) | ((ncand == 1) & ~rtimed)
    Dm = np.zeros((F, NC))
    if F > 1:
        Dm[1:] = S[:-1] - S[1:]
    with np.errstate(invalid='ignore'):
        UX = (Dm > TOL) & ~claimed
    cumUX = np.zeros((F + 1, NC), np.int32)
    cumUX[1:] = np.cumsum(UX, axis=0, dtype=np.int32)
    rb = np.flatnonzero(rbnd)
    rb = rb[np.lexsort((rsid[rb], rhi[rb], rc[rb]))]
    ru = np.flatnonzero(~rbnd)
    ru = ru[np.lexsort((rsid[ru], rhi[ru], rc[ru]))]
    return dict(NR=NR, rc=rc, rq=rq, rhi=rhi, rlo=rlo, rbnd=rbnd, rgb=rgb, legs=legs, roff=roff, ncand=ncand, coff=coff, ck=ck, rtimed=rtimed,
                rk=rk, ramb=ramb, cumUX=cumUX, rb=rb, rbk=rc[rb] * KEY + (rhi[rb] - base), ru=ru, ruk=rc[ru] * KEY + (rhi[ru] - base),
                hi_mismatch=int((rhi != rhmin).sum()) if NR else 0)


def _new_life(R, nE, f, c, To, kend, b, btype, zero, zbw, tb, t, we, J0, J1, V0, V1, acode, addr, mut):
    """Amendment 2 per fresh level. Members = records with hi > upd(f-1) and x_lo_hard <= T_last (unbounded: hi > upd(f-1)): OLD life
    (book-timed at or before f - 1, or every candidate there) excluded; NEW life = book-timed in (f, k_end] or settlement-only (no visible
    decrease, window inside the creation interval reach, x_lo_hard > upd(f-1)); book_ambiguous with a new-life candidate (CA) and untimed
    records are not placed. The front = the new life's earliest record (a settlement-only record precedes every book-timed one; among
    settlement-only records the hard window orders, overlapping ones join the earliest set). Competitors: CA records with a candidate before
    the front's interval and untimed records whose window reaches into (upd(f-1), upd(k*)]. One maker address over the earliest set and
    its competitors -> front_identified; several -> ambiguous_order (address set kept). The v1 unexplained-decrease rule on the frames
    before k* (a decrease claimed by no record)."""
    ffs, ig_old, sma = mut == 'first_fill_by_settle', mut == 'ignore_old_life', mut == 'same_maker_ambiguous'
    rtimed, rk, ramb, rbnd = R['rtimed'].tolist(), R['rk'].tolist(), R['ramb'].tolist(), R['rbnd'].tolist()
    rlo, rhi, rgb = R['rlo'].tolist(), R['rhi'].tolist(), R['rgb'].tolist()
    coff, ck, roff = R['coff'].tolist(), R['ck'].tolist(), R['roff'].tolist()
    legs, cumUX = R['legs'], R['cumUX']
    rb, ru = R['rb'].tolist(), R['ru'].tolist()
    nleg = [roff[r + 1] - roff[r] for r in range(R['NR'])]
    tl = t.tolist()
    so_used = np.zeros(R['NR'], bool)
    NEG = -(1 << 62)
    o = dict(st=np.full(nE, 'no_fill', dtype=object), rs=np.full(nE, '', dtype=object), lead=np.full(nE, -1, np.int64), fhi=np.full(nE, -1, np.int64),
             flo=np.full(nE, -1, np.int64), nset=np.zeros(nE, np.int64), one=np.zeros(nE, bool), amb=np.full(nE, '', dtype=object),
             gm=np.zeros(nE, np.int64), unx=np.zeros(nE, np.int64), old=np.zeros(nE, np.int64), timing=np.full(nE, '', dtype=object),
             bookms=np.full(nE, -1, np.int64), strad=np.zeros(nE, bool), nbeh=np.zeros(nE, np.int64), nun=np.zeros(nE, np.int64))
    ae, al, ar, ak, ai, an = [], [], [], [], [], []
    J0, J1, V0, V1 = J0.tolist(), J1.tolist(), V0.tolist(), V1.tolist()
    fL, cL, ToL, keL, bL, btL = f.tolist(), c.tolist(), To.tolist(), kend.tolist(), b.tolist(), btype.tolist()
    reasons = {GAP_B: 'gap_before_fill', UNK_B: 'unknown_cell_before_fill', END_B: 'book_end_before_fill'}
    for e in range(nE):
        fe, ce, Toe, ke = fL[e], cL[e], ToL[e], keL[e]
        tfe = tl[fe]
        NLr, SO, CA, U = [], [], [], []
        n_old = n_after = old_legs = mem_legs = 0
        for r in rb[J0[e]:J1[e]] + ru[V0[e]:V1[e]]:
            mem_legs += nleg[r]
            if rtimed[r]:
                k = rk[r]
                if k <= fe - 1 and not ig_old:
                    n_old += 1
                    old_legs += nleg[r]
                elif k <= ke:
                    NLr.append((k, r))
                else:
                    n_after += 1
            elif ramb[r]:
                cs = ck[coff[r]:coff[r + 1]]
                if cs[-1] <= fe - 1 and not ig_old:
                    n_old += 1
                    old_legs += nleg[r]
                elif any(fe < k <= ke for k in cs):
                    CA.append(r)
                else:
                    n_after += 1
            elif rbnd[r] and rlo[r] <= tfe and (rlo[r] > Toe or ffs):
                SO.append(r)
                so_used[r] = True
            else:
                U.append(r)
        o['old'][e] = n_old
        rs_ = None
        if SO:
            SO.sort(key=lambda r: (rhi[r], r))
            rs_ = SO[0]
            E_ = [rs_] if ffs else [rs_] + [r for r in SO[1:] if rlo[r] <= rhi[rs_]]
            kst, timing = fe, 'settlement'
            later = [r for r in SO[1:] if r not in E_] + [r for _, r in sorted(NLr)]
        elif NLr:
            NLr.sort()
            kst, rs_ = NLr[0]
            E_, timing = [rs_], 'book'
            later = [r for _, r in NLr[1:]]
        if rs_ is None:
            if btL[e] != ZERO_B and tl[bL[e] - 1] < we and n_after > 0:
                o['st'][e], o['rs'][e] = ('unknown' if btL[e] == UNK_B else 'gap'), reasons[btL[e]]
            elif CA or U:
                o['st'][e], o['rs'][e] = 'unknown', 'untimed_record_only'
            else:
                o['rs'][e] = 'returned_to_zero' if zbw[e] else 'window_end_without_fill'
            o['nun'][e] = mem_legs - old_legs
            continue
        CP = [r for r in CA if any(fe < k < kst for k in ck[coff[r]:coff[r + 1]])]
        tks = tl[kst] if kst >= 0 else tfe
        CP += [r for r in U if (rlo[r] if rbnd[r] else NEG) <= tks]
        Fset = E_ + CP
        codes = set()
        for r in Fset:
            codes.update(acode[legs[roff[r]:roff[r + 1]]].tolist())
        ambig = (len(Fset) >= 2 or len(codes) >= 2) if sma else len(codes) >= 2
        unx = int(cumUX[kst, ce] - cumUX[fe + 1, ce]) if kst > fe + 1 else 0
        if mut == 'ignore_decrease':
            unx = 0
        o['lead'][e] = -1
        o['fhi'][e], o['flo'][e] = rhi[rs_], rlo[rs_]
        o['nset'][e], o['one'][e] = len(Fset), len(Fset) >= 2 and len(codes) == 1
        o['timing'][e], o['bookms'][e], o['unx'][e] = timing, (tl[kst] if timing == 'book' else -1), unx
        if ambig:
            o['st'][e] = 'ambiguous_order'
            o['rs'][e] = 'untimed_record_could_be_first' if CP else ('overlapping_settlement_windows' if len(E_) >= 2 else 'sweep_several_makers')
            o['amb'][e] = '|'.join(sorted({addr[x] for r in Fset for x in legs[roff[r]:roff[r + 1]].tolist()}))
            gm = 0
            for r in Fset:
                gm |= rgb[r]
            o['gm'][e] = gm
            o['nun'][e] = mem_legs - old_legs
            continue
        if unx > 0:
            o['st'][e], o['rs'][e] = 'unexplained_decrease', 'unexplained_drop_before_fill'
            o['nun'][e] = mem_legs - old_legs
            continue
        o['st'][e], o['rs'][e] = 'front_identified', 'ok'
        lead = int(legs[roff[rs_]])
        o['lead'][e] = lead
        o['strad'][e] = bool(zero[e] and rhi[rs_] > tb[e])
        n_att = 0
        for r in E_:
            for x in legs[roff[r]:roff[r + 1]].tolist():
                ae.append(e); al.append(x); ar.append('front'); ak.append(0); ai.append(True); an.append(x == lead)
                n_att += 1
        beh = later + [r for r in CA if r not in CP and all(kst < k <= ke for k in ck[coff[r]:coff[r + 1]])]
        for j, r in enumerate(beh, start=1):
            for x in legs[roff[r]:roff[r + 1]].tolist():
                ae.append(e); al.append(x); ar.append('behind'); ak.append(j); ai.append(False); an.append(False)
                n_att += 1
                o['nbeh'][e] += 1
        o['nun'][e] = mem_legs - old_legs - n_att
    o['att'] = (np.asarray(ae, np.int64), np.asarray(al, np.int64), np.asarray(ar, dtype=object), np.asarray(ak, np.int64), np.asarray(ai, bool),
                np.asarray(an, bool))
    o['so_used'] = so_used
    return o


def _fronts_df(day, mid, col, ei, att, fl, hi, lo):
    """The attributed fills (front / behind) of identified levels -> fronts table rows, the null's labels and designated front fills."""
    le_e, fix, role, kk, inset, nullf = att
    tot = int(len(fix))
    if not tot:
        return None, np.zeros(0, np.int8), np.zeros(0, bool)
    le = ei[le_e]
    fr = {k_: None for k_ in FRONT_COLS}
    fr.update(day=np.full(tot, day, dtype=object), mid=np.full(tot, mid, np.int64), level_id=col['level_id'][le], side=col['side'][le],
              tick=col['tick'][le], role=role, k=kk, sid=fl['sid'][fix], mk_idx=fl['mk_idx'][fix], mk_hash=fl['mk_hash'][fix], addr=fl['addr'][fix],
              group=np.array(GROUPS, dtype=object)[fl['grp'][fix]], holder_group=col['holder_group'][le], mechanism=col['mechanism'][le],
              q=fl['q'][fix], hi_ms=hi[fix], lo_ms=lo[fix], in_overlap=inset, null_front=nullf, cohort=fl['cohort'][fix], excl_reason=fl['reason'][fix],
              q_c=fl['q_c'][fix], **{f'v_{h}': fl[f'v_{h}'][fix] for h in HORIZONS})
    isfr = role == 'front'
    need(bool((fl['addr'][fix][isfr] == col['holder'][le][isfr]).all()), 'internal: a front row is not its level holder')
    need(int(nullf.sum()) == int(len(np.unique(le_e))), 'internal: not one designated front fill per identified level')
    return pd.DataFrame({k_: fr[k_] for k_ in FRONT_COLS}), fl['grp'][fix].astype(np.int8), nullf.copy()


def _empty_core(day, mid):
    return dict(levels=pd.DataFrame({c: [] for c in LEVEL_COLS}), fronts=pd.DataFrame({c: [] for c in FRONT_COLS}), lab=np.zeros(0, np.int8),
                front=np.zeros(0, bool), counts=Counter(), checks={})


def market_core(day, mid, ws, we, bk, fl):
    """Fresh levels, their front status / holder / mechanism / speed and the attributed front + behind fills of one market (module rules).
    Pure: arrays in, tables out; each status reads only frames up to the first fill's reflection frame (and, at a zero boundary, up to the
    first later frame not shown 0)."""
    mut = mutant()
    day, mid, ws, we = str(day), int(mid), int(ws), int(we)
    t, S, gap, best, F = bk['t'], bk['S'], bk['gap'], bk['best'], int(bk['F'])
    cnt = Counter()
    n = int(fl['n'])
    hi, lo, g = fl['hi'], fl['lo'], fl['grid']
    cellA = np.where(g, fl['side'] * NT + fl['tick'], -1).astype(np.int64)
    sidc = pd.factorize(pd.Series(fl['sid'], dtype=object), sort=True)[0].astype(np.int64) if n else np.zeros(0, np.int64)
    ib = np.flatnonzero(g & (lo >= 0))
    ib = ib[np.lexsort((fl['mk_idx'][ib], sidc[ib], hi[ib], cellA[ib]))]
    iu = np.flatnonzero(g & (lo < 0))
    iu = iu[np.lexsort((fl['mk_idx'][iu], sidc[iu], hi[iu], cellA[iu]))]
    base = np.int64(min([ws - 10 ** 7] + ([int(t[0])] if F else []) + ([int(hi.min())] if n else [])) - 1)

    def off(x):
        return np.clip(np.asarray(x, np.int64) - base, 0, KEY - 1)
    keys_b = cellA[ib] * KEY + (hi[ib] - base)
    keys_u = cellA[iu] * KEY + (hi[iu] - base)
    hb_s, lb_s, hu_s = hi[ib], lo[ib], hi[iu]
    nb_tot, nu_tot = int(len(ib)), int(len(iu))
    if F == 0:
        return _empty_core(day, mid)
    P = _prints(t, cellA[g], fl['q'][g], lo[g], hi[g], F)
    U = np.zeros((F, NC), bool)
    if F > 1:
        with np.errstate(invalid='ignore'):
            U[1:] = (np.maximum(S[:-1] - S[1:], 0.) - P[1:]) > TOL
    cumU = np.zeros((F + 1, NC), np.int32)
    cumU[1:] = np.cumsum(U, axis=0, dtype=np.int32)
    pos = S > 0
    nanS = np.isnan(S)
    k0 = S == 0
    lpos = _acc_last(pos, F)
    # ---- fill-to-cell matching (lifecycle convention): shown > 0 in a frame at or before hi / within [fb_h, hi] ----
    fle = np.searchsorted(t, hi, 'right') - 1
    okf = g & (fle >= 0)
    lp_f = np.full(n, -1, np.int64)
    if okf.any():
        lp_f[okf] = lpos[fle[okf], cellA[okf]]
    fbh = np.where(lo >= 0, np.searchsorted(t, lo, 'left') - 1, -1)
    shown_le_hi, shown_win = lp_f >= 0, lp_f >= np.maximum(fbh, 0)
    top = g & (fl['grp'] < 3)
    checks = dict(grid_fills=int(g.sum()), grid_fills_not_shown_le_hi=int((g & ~shown_le_hi).sum()),
                  grid_fills_not_shown_in_window=int((g & ~shown_win).sum()), top3_grid_fills=int(top.sum()),
                  top3_not_shown_le_hi=int((top & ~shown_le_hi).sum()), top3_not_shown_in_window=int((top & ~shown_win).sum()),
                  top3_offgrid_fills=int((~g & (fl['grp'] < 3)).sum()), frames=F, gaps=int(gap.sum()), unknown_cells=int(nanS.sum()))
    # ---- candidates ----
    if F >= 2:
        ff, cc = np.nonzero(pos[1:] & ~pos[:-1])
        ff, cc = ff.astype(np.int64) + 1, cc.astype(np.int64)
    else:
        ff = cc = np.zeros(0, np.int64)
    inpop = t[ff - 1] < we if len(ff) else np.zeros(0, bool)
    cnt['candidates_total'] = int(len(ff))
    cnt['candidates_after_window_end'] = int((~inpop).sum())
    ff, cc = ff[inpop], cc[inpop]
    nC = int(len(ff))
    isg = gap[ff] if nC else np.zeros(0, bool)
    pnan = (nanS[ff - 1, cc] & ~isg) if nC else np.zeros(0, bool)
    fresh = ~isg & ~pnan
    need(bool(k0[ff[fresh] - 1, cc[fresh]].all()), 'internal: a fresh level not shown 0 in f - 1')
    ei = np.flatnonzero(fresh)
    f, c = ff[ei], cc[ei]
    nE = int(len(ei))
    col = {k: None for k in LEVEL_COLS}
    col.update(day=np.full(nC, day, dtype=object), mid=np.full(nC, mid, np.int64), level_id=np.arange(nC, dtype=np.int64),
               side=np.where(cc // NT == 0, 'bid', 'ask').astype(object), tick=cc % NT, frame=ff, fid=bk['fid'][ff], t_open_ms=t[ff - 1],
               t_create_ms=t[ff], phase=np.where(t[ff] < ws, 'pre', 'live').astype(object), unit=np.where(fresh, 'E1', 'status_only').astype(object),
               status=np.where(isg, 'gap', 'unknown').astype(object), reason=np.where(isg, 'creation_across_gap', 'prev_cell_unknown').astype(object),
               size_created=S[ff, cc])
    for k_, dv in (('end_type', ''), ('holder', ''), ('holder_group', ''), ('front_sid', ''), ('front_hash', ''), ('mechanism', ''),
                   ('amb_addresses', ''), ('front_timing', '')):
        col[k_] = np.full(nC, dv, dtype=object)
    for k_ in ('end_ms', 'front_hi_ms', 'front_lo_ms', 'front_mk_idx', 'n_members', 'n_unbounded', 'n_behind', 'n_unattributed', 'n_unexplained',
               'n_overlap', 'front_book_ms'):
        col[k_] = np.full(nC, -1, np.int64)
    col['amb_gmask'] = np.zeros(nC, np.int64)
    col['n_old_life'] = np.zeros(nC, np.int64)
    for k_ in ('straddles_end', 'overlap_one_address') + FLAGS:
        col[k_] = np.zeros(nC, bool)
    for k_ in ('best_prev_tick', 'speed_ms', 'speed_lo_ms', 'speed_hi_ms'):
        col[k_] = np.full(nC, NAN)
    fr_rows, lab, front = [], np.zeros(0, np.int8), np.zeros(0, bool)
    acode = pd.factorize(pd.Series(fl['addr'], dtype=object))[0].astype(np.int64) if n else np.zeros(0, np.int64)
    R = _records(t, S, gap, F, fl, g, cellA, sidc, acode, base)         # amendment 2: every record at a cell timed by the book
    so_used = np.zeros(R['NR'], bool)
    cnt['record_hi_mismatch'] = R['hi_mismatch']
    if nE:
        nbrk = _acc_next(~pos | gap[:, None], F)
        nnk = _acc_next(~k0 | gap[:, None], F)
        lnan = _acc_last(nanS, F)
        lgap = _acc_last(gap, F)
        To = t[f - 1]
        b = np.where(f + 1 < F, nbrk[np.minimum(f + 1, F - 1), c], F).astype(np.int64)
        inb = b < F
        bc = np.minimum(b, F - 1)
        btype = np.where(~inb, END_B, np.where(gap[bc], GAP_B, np.where(nanS[bc, c], UNK_B, ZERO_B)))
        zero = btype == ZERO_B
        tb = np.where(inb, t[bc], BIG)
        tb1 = t[b - 1]
        zbw = zero & (tb <= we)
        T_last = np.where(zero, np.minimum(tb, we), we)
        cK = c * KEY
        i0 = np.searchsorted(keys_b, cK + off(To), 'right').astype(np.int64)
        i1 = np.maximum(np.searchsorted(keys_b, cK + off(T_last + MAX_LAG_MS), 'right').astype(np.int64), i0)
        nbm = i1 - i0
        u0 = np.searchsorted(keys_u, cK + off(To), 'right').astype(np.int64)
        nu = np.maximum(np.searchsorted(keys_u, cK + (KEY - 1), 'right').astype(np.int64) - u0, 0)
        has_b, has_u = nbm > 0, nu > 0
        i0c = np.minimum(i0, max(nb_tot - 1, 0))
        u0c = np.minimum(u0, max(nu_tot - 1, 0))
        hb1 = np.where(has_b, hb_s[i0c], BIG) if nb_tot else np.full(nE, BIG)
        lb1 = np.where(has_b, lb_s[i0c], BIG) if nb_tot else np.full(nE, BIG)
        hu1 = np.where(has_u, hu_s[u0c], BIG) if nu_tot else np.full(nE, BIG)
        j1u = has_u                                # amendment 1: the first fill by hard-window start (an unbounded window starts first)
        j1b = has_b & ~has_u
        none = ~has_b & ~has_u
        hi1 = np.where(j1b, hb1, np.where(j1u, hu1, BIG))
        # ---- the earliest-overlap set (amendment 1): j1 + every member whose hard window overlaps j1's (bounded: x_lo_hard_k <= hi_1;
        #      unbounded: always) -> its size, whether it is one maker address, its groups (bits rank1 1 / rank2 2 / rank3 4 / other 8) ----
        i_ov = np.where(none, i0, np.minimum(np.searchsorted(keys_b, cK + off(hi1 + MAX_LAG_MS), 'right').astype(np.int64), i1))
        i_ov = np.maximum(i_ov, i0)
        u_ov = np.where(j1u, u0 + nu, u0)
        n_set = (i_ov - i0) + (u_ov - u0)
        mn, mx =np.full(nE, np.iinfo(np.int64).max, np.int64), np.full(nE, -1, np.int64)
        gmask = np.zeros(nE, np.int64)
        for srt, s0, s1 in ((ib, i0, i_ov), (iu, u0, u_ov)):
            sel = s1 > s0
            if sel.any():
                ac = np.r_[acode[srt], 0]
                ix = np.empty(2 * int(sel.sum()), np.int64)
                ix[0::2], ix[1::2] = s0[sel], s1[sel]
                mn[sel] = np.minimum(mn[sel], np.minimum.reduceat(ac, ix)[0::2])
                mx[sel] = np.maximum(mx[sel], np.maximum.reduceat(ac, ix)[0::2])
            gs_ = fl['grp'][srt] if len(srt) else np.zeros(0, np.int8)
            for g_ in range(4):
                cs_ = np.r_[0, np.cumsum(gs_ == g_)]
                gmask |= np.where((cs_[s1] - cs_[s0]) > 0, 1 << g_, 0)
        one_addr = (n_set >= 1) & (mn == mx)
        multi = (n_set >= 2) & ~one_addr
        amb_addr = np.full(nE, '', dtype=object)
        if multi.any():
            ab, au = fl['addr'][ib], fl['addr'][iu]
            for r_ in np.flatnonzero(multi):
                amb_addr[r_] = '|'.join(sorted(set(ab[i0[r_]:i_ov[r_]].tolist()) | set(au[u0[r_]:u_ov[r_]].tolist())))
        amb = multi.copy()                         # ambiguous only when the earliest-overlap set spans >= 2 addresses
        if mut == 'same_maker_ambiguous':          # planted error: the original rule (any overlap is ambiguous, even one maker's own fills)
            amb = n_set >= 2
        cre_ok = j1b & (lb1 > To)
        if mut == 'first_fill_by_settle':          # planted error: fills ordered by hi only (the window's lower end ignored)
            amb = np.zeros(nE, bool)
            cre_ok = j1b & (hi1 > To)
        obs_bad = ~zero & ~none & (hi1 > tb1)
        strad = zero & ~none & (hi1 > tb)
        nz = np.where(zero & (b + 1 < F), nnk[np.minimum(b + 1, F - 1), c], F).astype(np.int64)
        in_nz = nz < F
        t_nz1 = np.where(in_nz, t[np.maximum(nz - 1, 0)], t[F - 1])
        gap_nz = in_nz & gap[np.minimum(nz, F - 1)]
        late = strad & (hi1 > t_nz1)
        fa1 = np.searchsorted(t, hi1, 'left').astype(np.int64)
        Eix = np.maximum(np.where(zero, np.minimum(fa1, b), np.minimum(fa1, F - 1)), f)
        nU = (cumU[Eix + 1, c] - cumU[f + 1, c]).astype(np.int64)
        dec = nU > 0
        if mut == 'ignore_decrease':               # planted error: a cancel ahead is not looked for
            dec = np.zeros(nE, bool)
        conds = [none & zbw, none, amb, obs_bad & (btype == UNK_B), obs_bad & (btype == GAP_B), obs_bad, late & gap_nz, late & in_nz,
                 late, j1u & ~cre_ok, ~cre_ok, dec]
        st = _first(conds, ['no_fill', 'no_fill', 'ambiguous_order', 'unknown', 'gap', 'gap', 'gap', 'unknown', 'gap', 'unknown', 'unknown',
                            'unexplained_decrease'], 'front_identified', nE)
        rs = _first(conds, ['returned_to_zero', 'window_end_without_fill', 'overlapping_fill_windows', 'unknown_cell_before_fill',
                            'gap_before_fill', 'book_end_before_fill', 'gap_after_level_end', 'fill_spans_two_levels', 'book_end_before_fill',
                            'first_fill_unbounded', 'fill_window_before_creation', 'unexplained_drop_before_fill'], 'ok', nE)
        hold = np.full(nE, -1, np.int64)
        if nb_tot:
            hold[j1b] = ib[i0c[j1b]]
        if nu_tot:
            hold[j1u] = iu[u0c[j1u]]
        isf = st == 'front_identified'
        # ---- the r2 (settlement-clock) attribution; used only under the settle_only_timing mutant ----
        idl = isf & j1b
        H_cut = np.where(zero, np.minimum(T_last + MAX_LAG_MS, t_nz1), np.minimum(tb1, we + MAX_LAG_MS))
        i_end = np.minimum(np.searchsorted(keys_b, cK + off(H_cut), 'right').astype(np.int64), i1)
        cnt['attribution_invariant_violations'] = int((idl & (i_end < i0 + 1)).sum())
        i_end = np.where(idl, np.maximum(i_end, i0 + 1), i0)
        # ---- amendment 2: the level's new life by book-timed records; the front = the new life's earliest record ----
        k_we = int(np.searchsorted(t, we, 'left'))                 # intervals (upd(k-1), upd(k)] opening before we: k <= k_we
        kend = np.minimum(np.where(zero, b, b - 1), k_we)
        J0 = np.searchsorted(R['rbk'], cK + off(To), 'right').astype(np.int64)
        J1 = np.maximum(np.searchsorted(R['rbk'], cK + off(T_last + MAX_LAG_MS), 'right').astype(np.int64), J0)
        V0 = np.searchsorted(R['ruk'], cK + off(To), 'right').astype(np.int64)
        V1 = np.maximum(np.searchsorted(R['ruk'], cK + (KEY - 1), 'right').astype(np.int64), V0)
        A2 = _new_life(R, nE, f, c, To, kend, b, btype, zero, zbw, tb, t, we, J0, J1, V0, V1, acode, fl['addr'], mut)
        so_used = A2['so_used']
        if mut == 'settle_only_timing':             # planted error: the r2 rule (fills ordered by the 2-s settlement windows only)
            rows_ = np.flatnonzero(idl)
            lens = (i_end[rows_] - i0[rows_]).astype(np.int64)
            rep_ = np.repeat(rows_, lens)
            kk_ = np.arange(int(lens.sum()), dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
            att = (rep_, ib[np.repeat(i0[rows_], lens) + kk_] if len(rep_) else np.zeros(0, np.int64), np.where(kk_ == 0, 'front', 'behind').astype(object),
                   kk_, kk_ < (i_ov - i0)[rep_], kk_ == 0)
            P_ = dict(st=st, rs=rs, lead=np.where(isf, hold, -1), fhi=np.where(hold >= 0, hi[np.maximum(hold, 0)] if n else -1, -1),
                      flo=np.where(hold >= 0, lo[np.maximum(hold, 0)] if n else -1, -1), nset=n_set, one=one_addr & (n_set >= 2), amb=amb_addr,
                      gm=gmask, unx=nU, old=np.zeros(nE, np.int64), timing=np.where(isf, 'settlement', '').astype(object),
                      bookms=np.full(nE, -1, np.int64), strad=isf & strad, nbeh=np.where(idl, i_end - i0 - 1, 0), nun=np.where(idl, i1 - i_end, 0),
                      att=att)
        else:
            P_ = A2
        st, rs = P_['st'], P_['rs']
        isf = st == 'front_identified'
        lead = P_['lead']
        okl = lead >= 0
        lc_ = np.maximum(lead, 0)

        def at(a, dv, dt=object):                  # the front fill's field on identified rows (a market without fills has none)
            return np.where(okl, np.asarray(a)[lc_], dv) if n else np.full(nE, dv, dtype=dt)
        # ---- mechanism at creation (frame f - 1) and speed ----
        sd, tk = c // NT, c % NT
        bp = best[sd, f - 1]
        fin = np.isfinite(bp)
        with np.errstate(invalid='ignore'):
            is_imp = fin & np.where(sd == 0, tk > bp, tk < bp)
            is_deep = fin & np.where(sd == 0, tk < bp, tk > bp)
        is_emp, t_unk = np.isinf(bp), np.isnan(bp)
        p = lpos[f - 1, c].astype(np.int64)
        pcl = np.maximum(p, 0)
        p1 = np.minimum(pcl + 1, F - 1)
        dt_re = t[f - 1] - t[pcl]
        is_re = ((p >= 0) & (lnan[f - 1, c] <= p) & (lgap[f - 1] <= p) & (dt_re <= REEST_MS) & k0[p1, c] & ~U[p1, c] & (P[p1, c] > 0))
        mech = _first([is_re, is_imp, is_deep, is_emp], ['re_establish', 'improve', 'deep', 'at_empty_touch'], 'other', nE)
        chg = np.zeros((2, F), bool)
        with np.errstate(invalid='ignore'):
            chg[:, 1:] = (best[:, 1:] != best[:, :-1]) | gap[None, 1:]
        lchg = _acc_last(chg.T, F).T
        gs = lchg[sd, f - 1].astype(np.int64)
        gsc = np.maximum(gs, 1)
        clean = (gs >= 1) & ~np.isnan(best[sd, gsc]) & ~np.isnan(best[sd, gsc - 1]) & ~gap[gsc]
        mre, mim = mech == 're_establish', mech == 'improve'
        speed = np.where(mre, dt_re.astype(np.float64), np.where(mim & clean, (t[f - 1] - t[gsc]).astype(np.float64), NAN))
        # ---- E1 columns ----
        col['status'][ei], col['reason'][ei] = st, rs
        col['end_type'][ei] = np.where(zero & ~zbw, 'we', END_NAMES[btype])
        col['end_ms'][ei] = np.where(zbw, tb, np.where(zero, we, tb1))
        col['n_members'][ei], col['n_unbounded'][ei] = nbm + nu, nu
        hg = np.array(GROUPS, dtype=object)[np.asarray(at(fl['grp'], 3, np.int64), np.int64)]
        col['holder'][ei] = np.where(isf, at(fl['addr'], ''), '')
        col['holder_group'][ei] = np.where(isf, hg, '')
        col['front_sid'][ei] = np.where(isf, at(fl['sid'], ''), '')
        col['front_hash'][ei] = np.where(isf, at(fl['mk_hash'], ''), '')
        col['front_mk_idx'][ei] = np.where(isf, at(fl['mk_idx'], -1, np.int64), -1)
        col['front_hi_ms'][ei], col['front_lo_ms'][ei] = P_['fhi'], P_['flo']
        col['straddles_end'][ei] = P_['strad']
        col['n_behind'][ei], col['n_unattributed'][ei] = P_['nbeh'], P_['nun']
        col['n_overlap'][ei], col['overlap_one_address'][ei] = P_['nset'], P_['one']
        amb_ = st == 'ambiguous_order'
        col['amb_addresses'][ei] = np.where(amb_, P_['amb'], '')
        col['amb_gmask'][ei] = np.where(amb_, P_['gm'], 0)
        col['n_unexplained'][ei] = P_['unx']
        col['front_timing'][ei] = np.where(isf, P_['timing'], '')
        col['front_book_ms'][ei] = np.where(isf, P_['bookms'], -1)
        col['n_old_life'][ei] = P_['old']
        col['mechanism'][ei] = mech
        for k_, v_ in (('is_improve', is_imp), ('is_re_establish', is_re), ('is_deep', is_deep), ('is_at_empty_touch', is_emp), ('touch_unknown', t_unk)):
            col[k_][ei] = v_
        col['best_prev_tick'][ei] = bp
        col['speed_ms'][ei] = speed
        col['speed_lo_ms'][ei] = np.where(mre, (t[f - 1] - t[p1]).astype(np.float64), NAN)
        col['speed_hi_ms'][ei] = np.where(mre, (t[f] - t[pcl]).astype(np.float64), NAN)
        cnt['straddling_fronts'] = int(np.asarray(P_['strad'], bool).sum())
        cnt['levels_old_life_excluded'] = int((np.asarray(P_['old']) > 0).sum())
        # ---- the attributed fills of identified levels (front record(s) k = 0, behind k >= 1; one designated front fill per level) ----
        fdf, lab, front = _fronts_df(day, mid, col, ei, P_['att'], fl, hi, lo)
        if fdf is not None:
            fr_rows.append(fdf)
    rst = np.where(R['rtimed'], 'book_timed', np.where(R['ramb'], 'book_ambiguous', np.where(so_used, 'settlement_only', 'book_unmatched')))
    for s_ in REC_ST:
        cnt[f'rec_{s_}'] = int((rst == s_).sum())
    cnt['rec_unbounded'] = int((~R['rbnd']).sum())
    levels = pd.DataFrame({k_: col[k_] for k_ in LEVEL_COLS})
    if mut == 'drop_no_fill':                     # planted error: no_fill levels leave every denominator
        levels = levels[levels['status'] != 'no_fill'].reset_index(drop=True)
    fronts = _cat(fr_rows, FRONT_COLS)
    cnt['levels_rows'] = int(len(levels))
    cnt['front_rows'] = int(len(fronts))
    return dict(levels=levels, fronts=fronts, lab=lab, front=front, counts=cnt, checks=checks)


def null_market(day, mid, lab, front, n_perm=N_PERM, seed=SEED):
    """The declared null of one market: the maker labels of every attributed fill (front and behind) permuted n_perm times,
    Generator(PCG64(SeedSequence([seed, day, mid]))) -> observed and null front counts per group code (0..3)."""
    lab, front = np.asarray(lab, np.int8), np.asarray(front, bool)
    k, n = int(front.sum()), int(len(lab))
    obs = np.array([int((lab[front] == g_).sum()) for g_ in range(4)], np.int64)
    nul = np.zeros((4, int(n_perm)), np.int64)
    if k and n:
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(day), int(mid)])))
        X = np.tile(lab, (int(n_perm), 1))
        rng.permuted(X, axis=1, out=X)
        Xf = X[:, front]
        for g_ in range(4):
            nul[g_] = (Xf == g_).sum(axis=1)
    return dict(obs=obs, null=nul, k=k, n=n, fills=np.bincount(lab.astype(np.int64), minlength=4)[:4] if n else np.zeros(4, np.int64))


def p_hi_of(obs, null):
    """(1 + #{null >= obs}) / (n + 1) on integer counts (exact ties count)."""
    null = np.asarray(null)
    return (1 + int((null >= obs).sum())) / (len(null) + 1)


def joined_rows(day, mid, M, pfm, front_hashes):
    """Every top-3 order of chain_post_fill's orders.csv in the market: the cell before its posting window (chain_context.level_size at
    post_lo_union_ms; fresh <= 1 s) -> joined_existing_queue / cell_empty_before_window / cell_unknown / window_unbounded /
    cell_not_placeable, with the existing queue_class and whether the hash holds an identified front."""
    if pfm is None or not len(pfm):
        return pd.DataFrame({c: [] for c in JOIN_COLS})
    plo = pfm['post_lo_union_ms'].to_numpy(np.int64)
    ys = pfm['y_side'].to_numpy(dtype=object)
    tk = pfm['tick'].to_numpy(np.int64)
    okc = ((ys == 'bid') | (ys == 'ask')) & (tk >= 0) & (tk < NT)
    tq = np.where(plo >= 0, plo, int(M.ws))
    ls = CX.level_size(M, tq, np.where(okc, ys, 'bid').astype(object), np.where(okc, tk, 0))
    lst, lsz = np.asarray(ls['status'], dtype=object), np.asarray(ls['size'], np.float64)
    js = _first([plo < 0, ~okc, lst != 'ok', lsz > 0], ['window_unbounded', 'cell_not_placeable', 'cell_unknown', 'joined_existing_queue'],
                'cell_empty_before_window', len(pfm))
    rk = pfm['rank'].to_numpy(np.int64)
    hs = pfm['mk_hash'].to_numpy(dtype=object)
    return pd.DataFrame(dict(day=np.full(len(pfm), str(day), dtype=object), mid=np.full(len(pfm), int(mid), np.int64),
                             group=np.array(GROUPS, dtype=object)[rk - 1], addr=pfm['addr'].to_numpy(dtype=object), mk_hash=hs, y_side=ys, tick=tk,
                             n_fills=pfm['n_fills'].to_numpy(np.int64), post_lo_union_ms=plo, post_hi_ms=pfm['post_hi_ms'].to_numpy(np.int64),
                             window_status=pfm['window_status'].to_numpy(dtype=object), queue_class=pfm['queue_class'].to_numpy(dtype=object),
                             joined_status=js, size_before=np.where((plo >= 0) & okc & (lst == 'ok'), lsz, NAN),
                             is_front_holder=np.array([h in front_hashes for h in hs], bool)))


def market_pass(day, mid, M, fvm, pfm):
    """One market: book, fills, core, joined orders, the declared null draws -> a part."""
    bk = market_book(M)
    fl, fcnt = market_fills(M, fvm)
    core = market_core(str(day), int(mid), int(M.ws), int(M.we), bk, fl)
    fr = core['fronts']
    fh = set(fr.loc[fr['role'] == 'front', 'mk_hash'].tolist()) if len(fr) else set()
    jn = joined_rows(day, mid, M, pfm, fh)
    nul = null_market(day, mid, core['lab'], core['front'])
    cnt = Counter(fcnt)
    cnt.update(core['counts'])
    return dict(day=str(day), mid=int(mid), ws=int(M.ws), we=int(M.we), levels=core['levels'], fronts=fr, joined=jn, null=nul,
                counts=cnt, checks=core['checks'])


# ------------------------------------------------------------------ tables ------------------------------------------------------------------
def _econ(X):
    """Fills -> counts, shares and the cohort's gross markouts in cents per share (100 sum v / sum q over cohort rows)."""
    n = int(len(X))
    if not n:
        return dict(n_fills=0, shares=0., cohort_rows=0, cohort_shares=0., v_1s_c_sh=NAN, v_10s_c_sh=NAN, v_60s_c_sh=NAN, excluded='')
    coh = X['cohort'].to_numpy(dtype=bool)
    qs = float(X['q_c'].to_numpy(np.float64)[coh].sum())
    out = dict(n_fills=n, shares=float(X['q'].to_numpy(np.float64).sum()), cohort_rows=int(coh.sum()), cohort_shares=qs)
    for h in HORIZONS:
        out[f'v_{h}_c_sh'] = 100. * float(X[f'v_{h}'].to_numpy(np.float64)[coh].sum()) / qs if qs > 0 else NAN
    ex = Counter(str(r) for r in X['excl_reason'].to_numpy(dtype=object)[~coh])
    out['excluded'] = '|'.join(f'{k}:{ex[k]}' for k in sorted(ex))
    return out


def _gmask(col_values, gname):
    return pd.Series(np.asarray(col_values, dtype=object), dtype=object).isin(GNAMES[gname]).to_numpy(dtype=bool)


def _null_sum(NL, keys, gname):
    """Observed and null front counts of a group summed over markets, and the front total."""
    obs, nul, k = 0, np.zeros(N_PERM, np.int64), 0
    for key in keys:
        x = NL[key]
        for g_ in GCODES[gname]:
            obs += int(x['obs'][g_])
            nul = nul + x['null'][g_]
        k += int(x['k'])
    return obs, nul, k


def shares_table(LV, FR, NL, scopes):
    rows = []
    for sc, d, m in scopes:
        keys = [k for k in NL if sc == 'pooled' or (k[0] == d and (sc == 'day' or k[1] == m))]
        Lm = LV if sc == 'pooled' else LV[(LV['day'] == d).to_numpy() & ((sc == 'day') | (LV['mid'] == m).to_numpy())]
        Fm = FR if sc == 'pooled' else FR[(FR['day'] == d).to_numpy() & ((sc == 'day') | (FR['mid'] == m).to_numpy())]
        E1 = Lm[Lm['unit'] == 'E1']
        so = Counter(Lm.loc[Lm['unit'] == 'status_only', 'status'].tolist())
        stc = Counter(E1['status'].tolist())
        nfi = stc['front_identified']
        FI = E1[E1['status'] == 'front_identified']
        base = dict(scope=sc, day=d, mid=m if sc == 'market' else -1, n_markets=len(keys), n_candidates=int(len(Lm)),
                    n_status_only_gap=so['gap'], n_status_only_unknown=so['unknown'], n_fresh=int(len(E1)),
                    **{f'n_{s}': stc[s] for s in STATUSES})
        qall = Fm['q'].to_numpy(np.float64) if len(Fm) else np.zeros(0)
        amv = _amb_gmasks(E1)
        den = nfi + len(amv)                       # amendment 1: one denominator for both bounds (identified + ambiguous_order)
        for gname in SCOPE_GROUPS:
            nfg = int(_gmask(FI['holder_group'], gname).sum())
            nag = int(((amv & GBIT[gname]) != 0).sum())
            nf = int(len(Fm))
            gm = _gmask(Fm['group'], gname)
            nfgf = int(gm.sum())
            obs, nul, k = _null_sum(NL, keys, gname)
            need(obs == nfg and k == nfi, f'internal: null counts ({obs}, {k}) differ from the level table ({nfg}, {nfi}) in {sc} {d} {m}')
            ok_ = bool(k and den)
            ns = nul / den if ok_ else np.full(N_PERM, NAN)
            r = dict(base, group=gname, n_front_group=nfg, n_ambiguous_with_group=nag, bound_denominator=den,
                     front_share_lower=nfg / den if den else NAN, front_share_upper=(nfg + nag) / den if den else NAN,
                     front_share_identified=nfg / nfi if nfi else NAN, front_share_of_fresh=nfg / len(E1) if len(E1) else NAN, n_fills=nf,
                     n_fills_group=nfgf, fill_share=nfgf / nf if nf else NAN,
                     fill_share_q=float(qall[gm].sum()) / float(qall.sum()) if float(qall.sum()) > 0 else NAN,
                     null_mean=float(np.mean(ns)) if ok_ else NAN, null_p05=float(np.quantile(ns, 0.05)) if ok_ else NAN,
                     null_p50=float(np.quantile(ns, 0.5)) if ok_ else NAN, null_p95=float(np.quantile(ns, 0.95)) if ok_ else NAN,
                     p_hi=p_hi_of(obs, nul) if k else NAN, n_perm=N_PERM)
            r['front_minus_fill'] = r['front_share_lower'] - r['fill_share'] if (den and nf) else NAN
            rows.append(r)
    return pd.DataFrame(rows)


def _amb_gmasks(E1):
    """The group bits (GBIT) of every ambiguous_order level's address set (amendment 1)."""
    if not len(E1):
        return np.zeros(0, np.int64)
    return E1.loc[(E1['status'] == 'ambiguous_order').to_numpy(), 'amb_gmask'].to_numpy(np.int64)


def mechanism_table(LV, JN, pscopes):
    rows = []
    E1 = LV[LV['unit'] == 'E1']
    for sc, d in pscopes:
        Es = E1 if sc == 'pooled' else E1[E1['day'] == d]
        FI = Es[Es['status'] == 'front_identified']
        for gname in ('all',) + SCOPE_GROUPS:
            Fg = FI if gname == 'all' else FI[_gmask(FI['holder_group'], gname)]
            nn = int(len(Fg))
            for mc in MECHS:
                k = int((Fg['mechanism'] == mc).sum())
                rows.append(dict(table='front_wins', scope=sc, day=d, group=gname, status='front_identified', category=mc, n=k, denom=nn,
                                 share=k / nn if nn else NAN))
            for fg in FLAGS:
                k = int(Fg[fg].to_numpy(dtype=bool).sum()) if nn else 0
                rows.append(dict(table='front_wins_flags', scope=sc, day=d, group=gname, status='front_identified', category=fg, n=k, denom=nn,
                                 share=k / nn if nn else NAN))
        for s_ in STATUSES:
            Ss = Es[Es['status'] == s_]
            nn = int(len(Ss))
            for mc in MECHS:
                k = int((Ss['mechanism'] == mc).sum())
                rows.append(dict(table='fresh_levels', scope=sc, day=d, group='any', status=s_, category=mc, n=k, denom=nn, share=k / nn if nn else NAN))
        Js = JN if sc == 'pooled' else JN[JN['day'] == d]
        for gname in ('rank1', 'rank2', 'rank3', 'top3'):
            Jg = Js[_gmask(Js['group'], gname)]
            nn = int(len(Jg))
            for js in JOINED_ST:
                Jj = Jg[Jg['joined_status'] == js]
                rows.append(dict(table='joined', scope=sc, day=d, group=gname, status=js, category='_all', n=int(len(Jj)), denom=nn,
                                 share=len(Jj) / nn if nn else NAN, n_fills=int(Jj['n_fills'].sum()) if len(Jj) else 0,
                                 n_front_holder=int(Jj['is_front_holder'].to_numpy(dtype=bool).sum()) if len(Jj) else 0))
                for qc in sorted(set(Jj['queue_class'].astype(str).tolist())):
                    Jq = Jj[Jj['queue_class'].astype(str) == qc]
                    rows.append(dict(table='joined', scope=sc, day=d, group=gname, status=js, category=qc, n=int(len(Jq)), denom=int(len(Jj)),
                                     share=len(Jq) / len(Jj), n_fills=int(Jq['n_fills'].sum()),
                                     n_front_holder=int(Jq['is_front_holder'].to_numpy(dtype=bool).sum())))
    return pd.DataFrame(rows)


def speed_table(LV, pscopes):
    rows = []
    E1 = LV[LV['unit'] == 'E1']
    for sc, d in pscopes:
        Es = E1 if sc == 'pooled' else E1[E1['day'] == d]
        for mc in ('re_establish', 'improve'):
            Em = Es[Es['mechanism'] == mc]
            FI = Em[Em['status'] == 'front_identified']
            for gname in ('all_fresh', 'all_identified') + SCOPE_GROUPS:
                X = Em if gname == 'all_fresh' else (FI if gname == 'all_identified' else FI[_gmask(FI['holder_group'], gname)])
                r = dict(scope=sc, day=d, mechanism=mc, group=gname, n_levels=int(len(X)), **_qd(X['speed_ms'], 'speed_ms'))
                r['speed_lo_ms_p50'] = _med(X['speed_lo_ms']) if mc == 're_establish' else NAN
                r['speed_hi_ms_p50'] = _med(X['speed_hi_ms']) if mc == 're_establish' else NAN
                rows.append(r)
    return pd.DataFrame(rows)


def econ_table(FR, pscopes):
    rows = []
    for sc, d in pscopes:
        Fs = FR if sc == 'pooled' else FR[FR['day'] == d]
        for by, gcol in (('holder', 'holder_group'), ('filler', 'group')):
            for gname in ('all',) + SCOPE_GROUPS:
                Fg = Fs if gname == 'all' else Fs[_gmask(Fs[gcol], gname)]
                for mc in (('all',) + MECHS if by == 'holder' else ('all',)):
                    Fmc = Fg if mc == 'all' else Fg[Fg['mechanism'] == mc]
                    for role in ('front', 'behind'):
                        rows.append(dict(scope=sc, day=d, by=by, group=gname, mechanism=mc, role=role, **_econ(Fmc[Fmc['role'] == role])))
    return pd.DataFrame(rows)


def consistency_rows(LV, FR, NL, markets):
    rows = []
    E1 = LV[LV['unit'] == 'E1']
    for d, m in markets:
        Lm = E1[(E1['day'] == d).to_numpy() & (E1['mid'] == m).to_numpy()]
        FI = Lm[Lm['status'] == 'front_identified']
        RE = FI[FI['mechanism'] == 're_establish']
        Fm = FR[(FR['day'] == d).to_numpy() & (FR['mid'] == m).to_numpy()]
        nfi, nf = int(len(FI)), int(len(Fm))
        amv = _amb_gmasks(Lm)
        den = nfi + len(amv)
        for gname in SCOPE_GROUPS:
            gi = _gmask(FI['holder_group'], gname)
            nfg = int(gi.sum())
            nag = int(((amv & GBIT[gname]) != 0).sum())
            nfgf = int(_gmask(Fm['group'], gname).sum())
            fs, fls = (nfg / den if den else NAN), (nfgf / nf if nf else NAN)          # fs = the LOWER bound (amendment 1)
            obs, nul, k = _null_sum(NL, [(d, m)], gname)
            rg = RE['speed_ms'].to_numpy(np.float64)[_gmask(RE['holder_group'], gname)]
            ro = RE['speed_ms'].to_numpy(np.float64)[~_gmask(RE['holder_group'], gname)]
            mg, mo = _med(rg), _med(ro)
            Fh = Fm[_gmask(Fm['holder_group'], gname)]
            fe, be = _econ(Fh[Fh['role'] == 'front']), _econ(Fh[Fh['role'] == 'behind'])
            fin_fb = np.isfinite(fe['v_10s_c_sh']) and np.isfinite(be['v_10s_c_sh'])
            rows.append(dict(day=d, mid=m, group=gname, n_fresh=int(len(Lm)), n_front_identified=nfi, supported=bool(nfi >= MIN_FRONTS),
                             n_ambiguous=int(len(amv)), n_front_group=nfg, n_ambiguous_with_group=nag, front_share_lower=fs,
                             front_share_upper=(nfg + nag) / den if den else NAN, front_share_identified=nfg / nfi if nfi else NAN, n_fills=nf,
                             n_fills_group=nfgf, fill_share=fls, front_gt_fill=bool(fs > fls) if (den and nf) else None,
                             p_hi=p_hi_of(obs, nul) if k else NAN,
                             n_reest_group=int(np.isfinite(rg).sum()), med_reest_group=mg, n_reest_others=int(np.isfinite(ro).sum()), med_reest_others=mo,
                             faster=bool(mg < mo) if (np.isfinite(mg) and np.isfinite(mo)) else None,
                             front_cohort_shares=fe['cohort_shares'], front_v10_c_sh=fe['v_10s_c_sh'], behind_cohort_shares=be['cohort_shares'],
                             behind_v10_c_sh=be['v_10s_c_sh'], front_pays=bool(fe['v_10s_c_sh'] > be['v_10s_c_sh']) if fin_fb else None))
    return rows


def readings(LV, FR, NL, cons, markets):
    """The contract's readings, computed, per rank1 and top3 pooled (verdict = the reading, unidentified or insufficient_support)."""
    E1 = LV[LV['unit'] == 'E1']
    FI = E1[E1['status'] == 'front_identified']
    sup = [(r['day'], r['mid']) for r in cons if r['group'] == 'top3' and r['supported']]
    n_sup = len(sup)
    out = []
    for scope in READ_SCOPES:
        R = {(r['day'], r['mid']): r for r in cons if r['group'] == scope}
        gi = _gmask(FI['holder_group'], scope)
        nfi, nfg = int(len(FI)), int(gi.sum())
        amv = _amb_gmasks(E1)
        den, nag = nfi + len(amv), int(((amv & GBIT[scope]) != 0).sum())
        nf, nfgf = int(len(FR)), int(_gmask(FR['group'], scope).sum())
        fs, fls = (nfg / den if den else NAN), (nfgf / nf if nf else NAN)        # amendment 1: the LOWER bound vs the fill share
        obs, nul, k = _null_sum(NL, list(NL), scope)
        p = p_hi_of(obs, nul) if k else NAN
        n_gt = sum(1 for key in sup if R[key]['front_gt_fill'] is True)
        pr = dict(front_share_lower_gt_fill_share=bool(fs > fls) if (den and nf) else False, p_hi_le_0_05=bool(p <= P_MAX) if np.isfinite(p) else False,
                  supported_markets_ge_5=n_sup >= MIN_MARKETS, market_share_ge_2_3=bool(n_sup and SHARE_DEN * n_gt >= SHARE_NUM * n_sup))
        v = 'insufficient_support' if (n_sup < MIN_MARKETS or not (nfi and nf)) else ('wins_front_more_than_fills' if all(pr.values()) else 'unidentified')
        out.append(dict(scope=scope, reading='wins_front_more_than_fills', verdict=v, predicates=pr,
                        values=dict(front_share_lower=fs, front_share_upper=(nfg + nag) / den if den else NAN, fill_share=fls, n_front_identified=nfi,
                                    n_ambiguous=int(len(amv)), n_ambiguous_with_group=nag, bound_denominator=den, n_front_group=nfg, n_fills=nf,
                                    n_fills_group=nfgf, p_hi=p, n_perm=N_PERM, n_supported_markets=n_sup, n_markets_front_gt_fill=n_gt)))
        Fg = FI[gi]
        tot = int(len(Fg))
        cn = {mc: int((Fg['mechanism'] == mc).sum()) for mc in MECHS}
        top = max(cn.values()) if tot else 0
        tops = [mc for mc in MECHS if tot and cn[mc] == top]
        pr = dict(group_front_wins_ge_5=tot >= MIN_WINS, supported_markets_ge_5=n_sup >= MIN_MARKETS, unique_largest=len(tops) == 1)
        v = 'insufficient_support' if not (pr['group_front_wins_ge_5'] and pr['supported_markets_ge_5']) else (tops[0] if pr['unique_largest'] else 'unidentified')
        out.append(dict(scope=scope, reading='main_mechanism', verdict=v, predicates=pr,
                        values=dict(n_front_wins=tot, counts=cn, shares={mc: (cn[mc] / tot if tot else NAN) for mc in MECHS}, n_supported_markets=n_sup)))
        RE = FI[FI['mechanism'] == 're_establish']
        rgi = _gmask(RE['holder_group'], scope)
        mg, mo = _med(RE['speed_ms'].to_numpy(np.float64)[rgi]), _med(RE['speed_ms'].to_numpy(np.float64)[~rgi])
        ev = [key for key in sup if R[key]['faster'] is not None]
        n_fast = sum(1 for key in ev if R[key]['faster'] is True)
        pr = dict(pooled_median_below_others=bool(mg < mo) if (np.isfinite(mg) and np.isfinite(mo)) else False,
                  evaluable_markets_ge_5=len(ev) >= MIN_MARKETS, market_share_ge_2_3=bool(ev and SHARE_DEN * n_fast >= SHARE_NUM * len(ev)))
        v = ('insufficient_support' if (len(ev) < MIN_MARKETS or not (np.isfinite(mg) and np.isfinite(mo))) else
             ('faster_than_others' if all(pr.values()) else 'unidentified'))
        out.append(dict(scope=scope, reading='faster_than_others', verdict=v, predicates=pr,
                        values=dict(median_speed_ms_group=mg, median_speed_ms_others=mo, n_group=int(np.isfinite(RE['speed_ms'].to_numpy(np.float64)[rgi]).sum()),
                                    n_others=int(np.isfinite(RE['speed_ms'].to_numpy(np.float64)[~rgi]).sum()), n_supported_markets=n_sup,
                                    n_evaluable_markets=len(ev), n_markets_faster=n_fast, others='identified holders outside the group')))
        Fh = FR[_gmask(FR['holder_group'], scope)]
        fe, be = _econ(Fh[Fh['role'] == 'front']), _econ(Fh[Fh['role'] == 'behind'])
        okp = np.isfinite(fe['v_10s_c_sh']) and np.isfinite(be['v_10s_c_sh'])
        ev = [key for key in sup if R[key]['front_pays'] is not None]
        n_pay = sum(1 for key in ev if R[key]['front_pays'] is True)
        pr = dict(pooled_front_v10_gt_behind=bool(fe['v_10s_c_sh'] > be['v_10s_c_sh']) if okp else False, evaluable_markets_ge_5=len(ev) >= MIN_MARKETS,
                  market_share_ge_2_3=bool(ev and SHARE_DEN * n_pay >= SHARE_NUM * len(ev)))
        v = 'insufficient_support' if (len(ev) < MIN_MARKETS or not okp) else ('front_pays' if all(pr.values()) else 'unidentified')
        out.append(dict(scope=scope, reading='front_pays', verdict=v, predicates=pr,
                        values=dict(front=fe, behind=be, n_supported_markets=n_sup, n_evaluable_markets=len(ev), n_markets_front_gt_behind=n_pay)))
    return out


def assemble(parts):
    LV = _cat([p['levels'] for p in parts], LEVEL_COLS)
    FR = _cat([p['fronts'] for p in parts], FRONT_COLS)
    JN = _cat([p['joined'] for p in parts], JOIN_COLS)
    NL = {(p['day'], p['mid']): p['null'] for p in parts}
    return LV, FR, JN, NL


def analyse(kind, days, markets, parts, cnt):
    """Parts -> (tables, claims, digest)."""
    LV, FR, JN, NL = assemble(parts)
    scopes = [('pooled', 'all', -1)] + [('day', d, -1) for d in days] + [('market', d, m) for d, m in markets]
    pscopes = [('pooled', 'all')] + [('day', d) for d in days]
    cons = consistency_rows(LV, FR, NL, markets)
    tables = {'levels.csv': LV, 'fronts.csv': FR, 'shares.csv': shares_table(LV, FR, NL, scopes), 'mechanism.csv': mechanism_table(LV, JN, pscopes),
              'speed.csv': speed_table(LV, pscopes), 'economics.csv': econ_table(FR, pscopes), 'consistency.csv': pd.DataFrame(cons)}
    rd = readings(LV, FR, NL, cons, markets)
    E1 = LV[LV['unit'] == 'E1']
    pop = dict(markets=len(markets), candidates_rows=int(len(LV)), candidates_after_window_end=int(cnt.get('candidates_after_window_end', 0)),
               status_only=dict(Counter(LV.loc[LV['unit'] == 'status_only', 'status'].tolist())), fresh_levels=int(len(E1)),
               fresh_by_status={s: int((E1['status'] == s).sum()) for s in STATUSES},
               fresh_by_reason=dict(Counter(E1['reason'].tolist())), straddling_fronts=int(cnt.get('straddling_fronts', 0)),
               attributed_fills=int(len(FR)), front_fills=int((FR['role'] == 'front').sum()) if len(FR) else 0,
               attribution_invariant_violations=int(cnt.get('attribution_invariant_violations', 0)),
               identified_by_one_address_overlap=int(((E1['status'] == 'front_identified').to_numpy() & E1['overlap_one_address'].to_numpy(dtype=bool)).sum())
               if len(E1) else 0,
               supported_markets=int(sum(1 for r in cons if r['group'] == 'top3' and r['supported'])))
    n_old = int((E1['n_old_life'].to_numpy(np.int64) > 0).sum()) if len(E1) else 0
    beside = dict(record_timing={s_: int(cnt.get(f'rec_{s_}', 0)) for s_ in REC_ST}, records_unbounded_never_timed=int(cnt.get('rec_unbounded', 0)),
                  fresh_levels=int(len(E1)), levels_old_life_excluded=n_old, old_life_excluded_share=n_old / len(E1) if len(E1) else None,
                  identified_front_timing=dict(Counter(E1.loc[(E1['status'] == 'front_identified').to_numpy(), 'front_timing'].astype(str).tolist()))
                  if len(E1) else {})
    pop.update(beside)
    claims = dict(version=VERSION, kind=kind, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, amendment=AMENDMENT_COMMIT), days=list(days),
                  markets=[f'{d}_{m}' for d, m in markets], population=pop, readings=rd, beside_readings=beside,
                  null=dict(design='within each market the maker labels of every attributed fill (front and behind) of the identified levels '
                                   'permuted; the front fills keep their positions', n_perm=N_PERM, seed=SEED,
                            draws='Generator(PCG64(SeedSequence([seed, day, mid]))) per market', p_hi='(1 + #{null front count >= observed}) / 501'),
                  rules=dict(min_front_identified_per_supported_market=MIN_FRONTS, min_supported_markets=MIN_MARKETS, min_group_wins=MIN_WINS,
                             market_share=f'{SHARE_NUM}/{SHARE_DEN}', p_max=P_MAX, reest_ms=REEST_MS, gap_ms=GAP_MS, max_lag_ms=MAX_LAG_MS, tol_sh=TOL,
                             mechanism_precedence='re_establish > improve > deep > at_empty_touch > other',
                             front_holder='amendment 2: records timed by the book decrease they caused; old-life records (book-timed at or before '
                                          'f - 1) excluded; the front = the new life earliest record; one maker address over it (and any untimed '
                                          'record that could precede it) -> front_identified, several -> ambiguous_order (set recorded)',
                             front_share='amendment 1: bounds over identified + ambiguous_order; lower = identified held by the group, upper = lower + '
                                         'ambiguous levels whose address set contains the group; wins_front_more_than_fills and its market count use '
                                         'the lower bound; main_mechanism / faster_than_others / front_pays use front_identified only'),
                  multiplicity=dict(readings=len(READINGS), scopes=len(READ_SCOPES), tests=len(READINGS) * len(READ_SCOPES), all_listed=True,
                                    selected=None),
                  fill_to_cell={k: int(v) for k, v in sorted(cnt.items()) if k.startswith('cellcheck_') or k in FILL_JOIN_KEYS},
                  descriptive_only=True, intervals=None,
                  labels=dict(unit='held the front of a fresh level (creator or an order accepted in the same frame interval: not distinguished)',
                              economics='GROSS markouts from settle_ms, chain_post_fill PNL cohort; a later outcome is not the value of the action',
                              days='four consumed days: point estimates, no intervals (G = 4 < 5)'))
    return tables, claims, digest(claims, tables)


def digest(claims, tables):
    """The pilot / run print: the key numbers (no verdict that the tables do not carry)."""
    SH = tables['shares.csv']
    pool = SH[SH['scope'] == 'pooled'] if len(SH) else SH
    sh = {r['group']: dict(n_identified=r['n_front_group'], n_ambiguous_with_group=r['n_ambiguous_with_group'], denominator=r['bound_denominator'],
                           front_share_lower=r['front_share_lower'], front_share_upper=r['front_share_upper'], fill_share=r['fill_share'],
                           n_fills_group=r['n_fills_group'], p_hi=r['p_hi']) for r in pool.to_dict('records')}
    MC = tables['mechanism.csv']
    mix = {}
    if len(MC):
        T3 = MC[(MC['table'] == 'front_wins') & (MC['scope'] == 'pooled') & (MC['group'] == 'top3')]
        mix = {r['category']: dict(n=r['n'], share=r['share']) for r in T3.to_dict('records')}
    SP = tables['speed.csv']
    spd = {}
    if len(SP):
        R = SP[(SP['scope'] == 'pooled') & (SP['mechanism'] == 're_establish')]
        spd = {r['group']: dict(n=r['speed_ms_n'], p50=r['speed_ms_p50']) for r in R.to_dict('records')}
    EC = tables['economics.csv']
    eco = {}
    if len(EC):
        R = EC[(EC['scope'] == 'pooled') & (EC['by'] == 'holder') & (EC['mechanism'] == 'all')]
        eco = {f"{r['group']}:{r['role']}": dict(n=r['n_fills'], cohort_shares=r['cohort_shares'], v_10s_c_sh=r['v_10s_c_sh']) for r in R.to_dict('records')}
    return dict(population=claims['population'], shares_pooled=sh, mechanism_top3=mix, reest_speed=spd, economics=eco,
                fill_to_cell=claims['fill_to_cell'], readings={f"{r['scope']}:{r['reading']}": r['verdict'] for r in claims['readings']})


# ------------------------------------------------------------------ claims, receipt, write-once ------------------------------------------------------------------
def receipt(kind, days, markets, hashes, cnt, stages, t0, asof0, outs, git, extra=None):
    srcs = {_rel(f): _sha(f) for f in TOOL_FILES + DEP_FILES if Path(f).exists()}
    rec = dict(version=VERSION, kind=kind, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, amendment=AMENDMENT_COMMIT,
                                                         sha256=srcs.get(_rel(CONTRACT_FILE))),
               module=str(Path(__file__).resolve()), argv=list(sys.argv), run=C.RUN, mutant=os.environ.get('MT_QF_MUT', ''), asof_start=asof0,
               asof_end=_now(), days=list(days), markets=[f'{d}_{m}' for d, m in markets], addresses=list(TOP3), inputs=dict(sorted(hashes.items())),
               sources=srcs, outputs=outs, git=git, counts=dict(sorted((str(k), int(v)) for k, v in cnt.items())),
               constants=dict(NT=NT, GAP_MS=GAP_MS, MAX_LAG_MS=MAX_LAG_MS, TOL=TOL, REEST_MS=REEST_MS, SEED=SEED, N_PERM=N_PERM, MIN_FRONTS=MIN_FRONTS,
                              MIN_MARKETS=MIN_MARKETS, MIN_WINS=MIN_WINS, SHARE=f'{SHARE_NUM}/{SHARE_DEN}', P_MAX=P_MAX, FV_SHA256=FV_SHA256,
                              PROJ_MAX_S=PROJ_MAX_S, MECHS=MECHS, STATUSES=STATUSES, MUTANTS=MUTANTS, GBIT=GBIT, PILOT_RUN=PILOT_RUN, REC_ST=REC_ST),
               cpu_s=dict(by_stage=dict(stages), process_total=round(time.process_time(), 3)), wall_s=round(time.time() - t0, 2),
               vmhwm_mib=round(hwm(), 1), peak_rss='VmHWM of this process', guard_mib=C.GUARD_MIB)
    rec.update(extra or {})
    return rec


def write_out(out, tables, claims, rec_fn):
    """Write-once: every file in a partial directory, the receipt last (it hashes the others), then one rename."""
    out = Path(out)
    need(not out.exists(), f'write-once: {out} exists (refused)')
    tmp = out.with_name(f'.{out.name}.partial{os.getpid()}')
    need(not tmp.exists(), f'partial directory {tmp} exists (refused)')
    tmp.mkdir(parents=True)
    try:
        outs = {}
        for nm, df in tables.items():
            df.to_csv(tmp / nm, index=False)
            outs[nm] = dict(rows=int(len(df)))
        _json(tmp / 'claims.json', claims)
        outs['claims.json'] = {}
        for nm in outs:
            outs[nm].update(sha256=_sha(tmp / nm), bytes=int((tmp / nm).stat().st_size))
        rec = rec_fn(outs)
        _json(tmp / 'receipt.json', rec)
        need(not out.exists(), f'write-once: {out} appeared during the run (refused)')
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return rec


def run_core(kind, days, markets, parts, out, hashes, cnt, stages, tick, git, asof0, t0, extra_fn=None):
    """Shared tail of the run / pilot (and the selftest's end-to-end): tables, claims, write-once."""
    for p in parts:
        for k_, v_ in p['checks'].items():
            cnt[f'cellcheck_{k_}'] += int(v_)
        cnt.update(p['counts'])
    tables, claims, dg = analyse(kind, days, markets, parts, cnt)
    tick('tables')
    guard('tables')

    def rec_fn(outs):
        tick('write')
        return receipt(kind, days, markets, hashes, cnt, stages, t0, asof0, outs, git, extra_fn(stages) if extra_fn else None)
    rec = write_out(out, tables, claims, rec_fn)
    return rec, claims, dg


# ------------------------------------------------------------------ run / pilot ------------------------------------------------------------------
def preflight_projection(path, hashes):
    """The full run launches only below the recorded pilot projection limit (<= 120 CPU s)."""
    path = Path(path)
    need(path.is_file(), f'no recorded pilot projection: {path} is missing (run the pilot first; refused before launch)')
    J = json.loads(path.read_text())
    pr = (J.get('projection') or {}).get('full_run_cpu_s')
    need(J.get('kind') == 'pilot' and isinstance(pr, (int, float)) and np.isfinite(pr),
         f'{path.name}: not a pilot receipt with a finite projection.full_run_cpu_s (refused before launch)')
    need(float(pr) <= PROJ_MAX_S, f'the recorded pilot projects {float(pr):.1f} CPU s for the full run > {PROJ_MAX_S:.0f} s (refused before launch)')
    hashes[_rel(path)] = _sha(path)
    return dict(pilot_receipt=_rel(path), full_run_cpu_s=float(pr), limit_s=PROJ_MAX_S)


def _load_default(day, mid, root, hc):
    return CX.load_market(day, int(mid), TOP3, root=root, hash_cache=hc)


def execute(kind, days, out, inp=None, pilot_receipt=None, require_freeze=True, log=print, loader=None, book=None, pf_dir=None,
            fv_sha=FV_SHA256):
    """kind 'run' (every measurement day -> v2/chains/queue_front; freeze + pilot projection required) or 'pilot' (its day, out given).
    loader / book / pf_dir / fv_sha are injected only by the selftest (production: chain_context.load_market, the 19-market BOOK manifest,
    v2/chains/post_fill, the accepted fills_valued binding)."""
    c_imp = time.process_time()
    t0, asof0 = time.time(), _now()
    need(mutant() == '', f"MT_QF_MUT={os.environ.get('MT_QF_MUT')!r} is set: production never runs a planted error (refused before any input is read)")
    need(C.RUN == 'v2', f'MT_RUN must be v2 (is {C.RUN!r}; refused)')
    need(kind in ('run', 'pilot'), f'kind {kind!r} (refused)')
    days = tuple(str(d) for d in days)
    need(len(days) >= 1 and len(set(days)) == len(days) and all(d in C.MEASURE_DAYS for d in days),
         'days must be distinct measurement days (refused before any path is built)')
    out = Path(out)
    need(not out.exists(), f'write-once: {out} exists (refused before any input is read)')
    need(out.parent.is_dir(), f'the output parent directory is missing: {out.parent} (refused)')
    PT = paths()
    hashes, cnt = {}, Counter()
    stages = dict(imports=round(c_imp, 3))
    last = [time.process_time()]
    cap = CPU_CAP_RUN_S if kind == 'run' else CPU_CAP_PILOT_S

    def tick(nm):
        now = time.process_time()
        stages[nm] = round(stages.get(nm, 0.) + now - last[0], 3)
        last[0] = now
        need(now < cap, f'CPU self-guard: {now:.1f} s >= {cap:.0f} s at {nm} (refused)')
    git = _git([f for f in TOOL_FILES if Path(f).exists()])
    pre = {}
    if kind == 'run':
        need(days == tuple(C.MEASURE_DAYS), 'the full run covers every measurement day (refused)')
        if require_freeze:
            need(all(Path(f).exists() for f in TOOL_FILES) and bool(git.get('unmodified')),
                 'freeze: queue_front.py, its selftest, the contract and chain_context.py must be tracked and unmodified (refused)')
        pre = preflight_projection(Path(pilot_receipt) if pilot_receipt else PT['pilot_receipt'], hashes)
    inp = Path(inp) if inp else C.OUT
    t3 = CX.top3(inp)
    need(tuple(a.lower() for a in t3) == TOP3_L, 'chain_context.top3 differs from the module top 3 (refused)')
    hashes[_rel(inp / 'econ' / 'identities.json')] = _sha(inp / 'econ' / 'identities.json')
    bm = CX.book_markets() if book is None else book
    need(all(d in bm for d in days), f'days {days} are not all BOOK manifest days (refused)')
    markets = [(d, int(m)) for d in days for m in sorted(int(x) for x in bm[d])]
    tick('preflight')
    PF, pfinfo = read_post_fill(PT['post_fill'] if pf_dir is None else pf_dir, markets, hashes, fv_sha)
    tick('post_fill')
    fvp = inp / 'econ' / 'fills_valued.parquet'
    FV = read_fv_makers(fvp, markets, cnt)
    fv_mid = FV['mid'].to_numpy(np.int64) if len(FV) else np.zeros(0, np.int64)
    pf_key = list(zip(PF['day'].to_numpy(dtype=object), PF['mid'].to_numpy(np.int64))) if len(PF) else []
    tick('fills_valued')
    loader = _load_default if loader is None else loader
    hc, parts = {}, []
    for d, m in markets:
        M = loader(d, m, inp, hc)
        need(str(M.day) == d and int(M.mid) == m and bool(M.book), f'loaded market {M.day} {M.mid} is not the BOOK market {d} {m} (refused)')
        for k_, v_ in (M.inputs or {}).items():
            if isinstance(v_, dict) and v_.get('sha256'):
                hashes.setdefault(k_, v_['sha256'])
        tick('context_load')
        fvm = FV[fv_mid == m].reset_index(drop=True)
        pfm = PF[np.array([k == (d, m) for k in pf_key], bool)].reset_index(drop=True) if len(PF) else PF
        parts.append(market_pass(d, m, M, fvm, pfm))
        del M
        cnt['markets'] += 1
        tick('market')
        guard(f'market {d} {m}')
    fk = str(fvp.resolve())
    fsha = hc.get(fk) or _sha(fvp)
    hashes['econ/fills_valued.parquet'] = fsha
    need(fsha == fv_sha, f'fills_valued sha256 {fsha[:12]} differs from the accepted binding {str(fv_sha)[:12]} (refused)')
    need(pfinfo['fills_valued'] == fsha, "fills_valued differs from chain_post_fill's receipt (refused: mismatched artifacts)")
    cnt['post_fill_rows'], cnt['post_fill_rows_run'] = pfinfo['rows'], pfinfo['rows_run']
    tick('bindings')
    n_mk = len(markets)

    def extra_fn(stg):
        ex = dict(preflight=pre, post_fill=pfinfo)
        if kind == 'pilot':
            r_mk = N_BOOK / max(1, n_mk)
            fixed = sum(stg.get(k, 0.) for k in ('imports', 'preflight', 'post_fill', 'bindings'))
            scal = sum(stg.get(k, 0.) for k in ('fills_valued', 'context_load', 'market', 'tables', 'write'))
            proj = fixed + scal * r_mk
            ex['projection'] = dict(full_run_cpu_s=round(proj, 2), pilot_cpu_s=round(time.process_time(), 3), limit_s=PROJ_MAX_S,
                                    launch_allowed=bool(proj <= PROJ_MAX_S),
                                    rule=f'fixed (imports + preflight + post_fill + bindings) {fixed:.2f} + (fills_valued + context_load + market + '
                                         f'tables + write) {scal:.2f} x {N_BOOK}/{n_mk}; the first load carries the one-off fills_valued sha256 '
                                         f'(scaled: conservative)')
        return ex
    rec, claims, dg = run_core(kind, days, markets, parts, out, hashes, cnt, stages, tick, git, asof0, t0, extra_fn=extra_fn)
    log(json.dumps(_clean(dict(out=str(out), digest=dg, projection=rec.get('projection'), cpu_s=rec['cpu_s'], vmhwm_mib=rec['vmhwm_mib'])),
                   indent=1, sort_keys=True, default=str))
    return rec


def main(argv):
    a = list(argv[1:])
    usage = 'usage: MT_RUN=v2 queue_front.py run | MT_RUN=v2 queue_front.py run --pilot <day> [--out <dir>]'
    if a == ['run']:
        execute('run', C.MEASURE_DAYS, paths()['out'])
        return 0
    if len(a) in (3, 5) and a[0] == 'run' and '--pilot' in a:
        opts = {}
        rest = a[1:]
        for i in range(0, len(rest), 2):
            need(rest[i] in ('--pilot', '--out') and rest[i] not in opts, usage)
            opts[rest[i]] = rest[i + 1]
        day = opts['--pilot']
        need(day in C.MEASURE_DAYS, f'pilot day {day} is not a measurement day (refused before any path is built)')
        execute('pilot', (day,), Path(opts['--out']) if '--out' in opts else pilot_out(day))
        return 0
    raise SystemExit(usage)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
