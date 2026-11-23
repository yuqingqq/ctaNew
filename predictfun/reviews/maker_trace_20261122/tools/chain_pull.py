"""PULL logic-chain census (P-2026-005 MAKER-TRACE logic-chain packet; contract order_estimator_contract.md section 7.2, which overrides the
development brief where they differ; session aea5bfb9 development, 2026-11-23). USER (binding): "... and when they pull, anything happens,
like reverse price trend from binance ... need to figure out the logics to get to know how they perform." DESCRIPTIVE ONLY, consumed days
(point estimates, no intervals: G = 4 < 5). NO ROW IS A PULL: exhaustion (EX) is feasible for every order (the lifecycle keeps 'exhausted'
in every removal set), so a cancel-possible removal is an EX / PL TIE (contract 4.1 / 4.5: a pull is identified only when EX and EC are
infeasible in every scenario, which public data never gives); every row carries identification 'tied' and its feasible end histories.
Market context comes ONLY from chain_context (chain_context_v1, contract 7.0: btc_ret, book, touch_distance, level_size, position,
pending, tau, matched_times, over_window, null_summary); this tool's declared rules follow (none fitted, none tuned after a result).
POPULATION the ex-ante top 3 (econ/identities.json ex_ante.top10_shares ranks 1-3 = pf_mt_decision.TOP3, else refuse) x the 19 BOOK markets
       (pf_mt_econ.load_book: exactly 19, measurement days, else refuse); every top-3 hash of life/<day>_<mid>_orders.parquet (MT_RUN=v2) with
       its three scenario rows (else refuse), its life fills and its x_links rows.
VIEWS  S_FIFO and S_NOPRIO reported separately (contract 7.2); S_FIFO_LO's removal_hi carried (x_rhi_S_FIFO_LO, lo_rhi_differs).
       removal_lo = the last fill's x_lo_hard (hi - 2,000 ms), scenario-free (refused if the rows disagree).
CLASS  exactly one per order and view, never pooled, first that applies: unidentified/order_invalid (an ORDER_INVALID flag, a side other
       than bid / ask, a tick outside 0..100, or a removal set that is not a non-empty subset of pf_mt_common.REMOVAL); exhausted_proven
       (S_FIFO view, set = {exhausted}; grade C); capacity_tight (x_size_ub <= cum + TOL_SH in all three scenario rows: EX only, grade B);
       capacity_tight_view (only this view's row is tight: EX only under the view, counted apart); unidentified/missing_evidence;
       end_censored_possible (end_censored in the set); cancel_possible (cancelled_or_expired in the set, PL feasible = x_size_ub NaN or
       > cum + TOL_SH, removal_lo >= 0, removal_hi >= 0); else unidentified/window_unbounded or no_cancel_member. end_histories per view =
       the feasible of EX (exhausted in the set), PL (cancelled_or_expired, UB not tight), EC (end_censored, UB not tight): a cancel_possible
       row is always 'EX|PL'. pulls.csv holds the cancel_possible rows (both views); the other classes get the same context beside.
WINDOW decision window under the declared delay profile D in DELAYS = {0, 250, 1000} ms (contract 7.0 / 7.2; shown separately, never an
       identified latency): [removal_lo - D, removal_hi] (chain_context.over_window(D_ms=D)). Nested: the union of the profiles is the D = 1,000
       window. WINDOW STATUS on [removal_lo, removal_hi], first that applies: order_invalid, window_unbounded (an end < 0), before_pre
       (removal_lo < ws - PRE_MS), straddles_ws, reaches_we (removal_hi >= we: market-end expiry possible), ok. Only ok windows get controls
       and enter a comparison; every row stays with its status.
INSTANTS chain_context.over_window's set: the lower end, every 100 ms grid instant strictly inside, the upper end (one instant when equal).
       Grid instants are read from per-market series built by the chain_context queries on the aligned grid; an end off the grid is a direct
       chain_context point query. Every feature at an instant uses data <= that instant (pending is the declared reconstruction exception:
       it is built from records created after the instant). Checked per market against chain_context (refused on a mismatch): every
       CHECK_EVERY-th grid instant of every series against point queries, and the first CHECK_K ok windows against over_window (category set,
       instant / valid counts, BTC min / max).
CONTEXT per instant c (s = +1 YES bid / -1 YES ask; aligned = s x value: negative = against the order): a_1s / a_5s / a_30s = s r_* (BTC bps);
       dist = own-side touch distance (ticks; + behind, 0 at, - inside; -inf an empty own side = inside); opp_tw5 = s (opp(c - 5 s) - opp(c))
       (ticks, + = the opposite touch moved toward the order); mid_a5 = s (mid(c) - mid(c - 5 s)) (cents); depth = shares at the order's
       (side, tick), ddepth5 its 5 s change; nf1 / nf5 = the address's legs with settle_ms in [c - 1 s, c) / [c - 5 s, c); dpos5 = s x the
       recorded-position change over [c - 5 s, c); spos = s x recorded position; spend = s x pending; tau (s).
MECH   per-instant conditions (contract 7.0 move bins): btc_against_1s a_1s <= -0.5; btc_against_5s a_5s <= -1 (contract
       pull_after_adverse_move); btc_against_30s a_30s <= -2; book_toward_5s opp_tw5 >= 1; book_away_5s opp_tw5 <= -1; mid_against_5s
       mid_a5 <= -1; behind_touch dist >= 1; own_fill_5s nf5 >= 1; inv_built_5s dpos5 > POS_EPS (contract inventory_driven_pull). Per window:
       compatible (true at every instant, none missing) / contradicted (false at every instant, none missing) / timing_sensitive (both occur)
       / unidentified (no valid instant, or the valid ones agree while some are missing); per D and over the union; d_sensitive = the per-D
       classes differ. most_supported = the compatible set, else the timing_sensitive set, else 'unidentified', else 'none'. A condition
       class describes the window; it never names a cause (the removal itself is an EX / PL tie).
FEATURES continuous per D: the value at the window's lower end of a_1s a_5s a_30s dist opp_tw5 mid_a5 depth ddepth5 nf1 nf5 dpos5 spos spend
       tau; over the window min / max of a_1s a_5s a_30s; binary: every MECH as all (compatible) and some (compatible or timing_sensitive).
CONTROL N_DRAWS = 200 width-matched draws per event and kind (>= chain_context.draws_per_target of any address-day; 200 replicates of the
       group statistic). presence (contract 7.2): chain_context.matched_times (seed 20261123, PCG64 per target, same market and phase pool:
       pre [ws - 300 s, ws), live [ws, we)) with mask = 'some valid same-side order of the address is inside its certain-presence span
       [persist_lo, persist_hi]' at every grid instant of the window (width ceil(w / 100) x 100 ms, w = removal_hi - removal_lo) and fallback
       = the address's active span (first to last status-ok leg, settle_ms; counted, not a not-removed instant); targets grouped by (view,
       address, side, width) with key 'chain_pull|<view>|<side>|w<width>'; both kinds' control features use the event order's own (side,
       tick).
       fill_anchored (SUPPLEMENTARY, declared here beyond contract 7.2): anchors = the non-last fills of the address's valid same-side orders,
       start = the first grid instant >= the fill's x_lo_hard (the event window's own geometry from its last fill), the window inside the
       phase and ending by that order's persist_hi (proven present afterwards: not removed); uniform with replacement over anchors, numpy
       Generator(PCG64([SEED, view, day, rank])) over the address-day's events in (market, view, removal_lo, hash) order; it removes the
       construction that every event window starts 2 s before a fill. A control decision window is [c - D, c + ceil(w / 100) x 100]: the
       upper end on the grid, as matched_times checks its mask (width matched to the 100 ms grid). Point-query dist uses touch_distance's
       formula on the same book() call (checked against touch_distance on the grid).
STATS  scope = view x group x control x address (rank1..3, pooled) x day (each, all) x D (0 / 250 / 1000; 'union' for MECH) x feature: n_events
       (ok window), n_with_controls; obs = mean over events with a finite value and >= 1 valid draw; replicate k = the mean over those events
       of their k-th draw where valid; chain_context.null_summary(obs, replicates): null mean, p05 / p50 / p95, p_upper = (1 + #rep >= obs) /
       (1 + n), p_lower = (1 + #rep <= obs) / (1 + n). Groups: cancel_possible, cp_successor_5s, cp_no_successor_5s, exhausted_proven,
       capacity_tight, capacity_tight_view, end_censored_possible, unidentified (unbounded windows have no controls).
SUCCESSOR x_links rows (status ok, refill / reprice) with prev_hash = the order and gap_ms <= SUCC_MS = 5,000 -> cp_successor_5s, else
       cp_no_successor_5s; successor_before_rhi = the successor filled before the order was proven gone (overlap possible: replacement versus
       a second ladder order is order_estimator.link_hypotheses' question, contract 4.6; never decided here).
OUTPUT write-once <RECEIPT>/chains/pull/ (pilot: chains/pull_pilot_<day>_r<rank>/), built in a partial directory renamed at the end:
       pulls.csv (one row per cancel_possible order x view: an EX / PL tie), controls.csv (per event x control: draw status, draws, headline
       control means and the event's mid-rank percentile among its own draws), summary.csv (STATS), census.csv (class / reason / window
       status / end histories with orders, fills, shares per view x address x day and pooled), receipt.json (inputs' sha256 via
       chain_context's hash cache, sources' sha256, git state, declared rules, counts, context checks, CPU per stage, VmHWM; the pilot adds
       the measured projection).
GUARDS days are measurement days only, checked on the string before any path; VmHWM self-guard pf_mt_common.GUARD_MIB (never raised); CPU
       self-guard CPU_CAP_S; write-once. The tool refuses when chain_context lacks a required function.
Usage (run_light, PF_SLOT=claude4, MT_RUN=v2): chain_pull.py run | chain_pull.py pilot <day> <rank 1-3>   selftest: chain_pull_selftest.py"""
import hashlib
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
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pf_mt_common as C  # noqa: E402
import pf_mt_econ as E  # noqa: E402
import pf_mt_decision as M  # noqa: E402   (constants only: TOP3, ORDER_INVALID)

need, OK = C.need, C.STATUS_OK
NAN = float('nan')
VERSION = 'chain_pull_v2'
CONTRACT_FILE = C.REPO / 'predictfun/reviews/maker_trace_20261122/order_estimator_contract.md'

# ------------------------------------------------------------------ declared constants (none fitted) ------------------------------------------------------------------
TOP3 = M.TOP3
LOW3 = tuple(a.lower() for a in TOP3)
RANKS = ('rank1', 'rank2', 'rank3')
VIEWS = ('S_FIFO', 'S_NOPRIO')
SCEN = tuple(C.SCENARIOS)                       # S_FIFO, S_FIFO_LO, S_NOPRIO
DELAYS = (0, 250, 1000)
ND = len(DELAYS)
STEP = 100
PRE_MS = 300_000
LOOKBACK = 5_000                                # the 5 s differences (opp_tw5, mid_a5, ddepth5, dpos5, nf5)
LAGROWS = LOOKBACK // STEP
GRID_PRE_MS = PRE_MS + max(DELAYS) + LOOKBACK + 2 * STEP
GRID_POST_MS = 60_000
N_DRAWS = 200
SEED = 20261123
SUCC_MS = 5_000
TOL_SH = 0.01
POS_EPS = 1e-6
CPU_CAP_S = 240.
CHECK_EVERY, CHECK_K = 50, 3
MOVE_BP = (('1s', 1_000, 0.5), ('5s', 5_000, 1.0), ('30s', 30_000, 2.0))   # contract 7.0: |r| >= threshold is a move
CLASSES = ('cancel_possible', 'exhausted_proven', 'capacity_tight', 'capacity_tight_view', 'end_censored_possible', 'unidentified')
WSTAT = ('ok', 'order_invalid', 'window_unbounded', 'before_pre', 'straddles_ws', 'reaches_we')
GROUPS = ('cancel_possible', 'cp_successor_5s', 'cp_no_successor_5s', 'exhausted_proven', 'capacity_tight', 'capacity_tight_view',
          'end_censored_possible', 'unidentified')
KINDS = ('presence', 'fill_anchored')
# mechanisms: (name, series, variant, op, threshold); variant: side / cell / addr / addr_side
MECHS = (('btc_against_1s', 'a_1s', 'side', 'le', -0.5), ('btc_against_5s', 'a_5s', 'side', 'le', -1.0),
         ('btc_against_30s', 'a_30s', 'side', 'le', -2.0), ('book_toward_5s', 'opp_tw5', 'side', 'ge', 1.0),
         ('book_away_5s', 'opp_tw5', 'side', 'le', -1.0), ('mid_against_5s', 'mid_a5', 'side', 'le', -1.0),
         ('behind_touch', 'dist', 'cell', 'ge', 1.0), ('own_fill_5s', 'nf5', 'addr', 'ge', 1.0), ('inv_built_5s', 'dpos5', 'addr_side', 'gt', POS_EPS))
MNAMES = tuple(m[0] for m in MECHS)
START_FEATS = (('a_1s', 'side'), ('a_5s', 'side'), ('a_30s', 'side'), ('dist', 'cell'), ('opp_tw5', 'side'), ('mid_a5', 'side'),
               ('depth', 'cell'), ('ddepth5', 'cell'), ('nf1', 'addr'), ('nf5', 'addr'), ('dpos5', 'addr_side'), ('spos', 'addr_side'),
               ('spend', 'addr_side'), ('tau', 'mkt'))
WIN_FEATS = ('a_1s', 'a_5s', 'a_30s')
END_NAMES = tuple(n for n, _ in START_FEATS) + tuple(f'r_{nm}' for nm, _, _ in MOVE_BP)
CONT = tuple(f'{n}_start' for n, _ in START_FEATS) + tuple(f'{n}_{w}' for n in WIN_FEATS for w in ('wmin', 'wmax'))
BINS = tuple(f'{m}_{q}' for m in MNAMES for q in ('all', 'some'))
FEATS = CONT + BINS
NF, NCONT = len(FEATS), len(CONT)
HEADLINE = ('a_1s_start', 'a_5s_start', 'a_1s_wmin', 'a_5s_wmin')
CLS_MECH = ('compatible', 'contradicted', 'timing_sensitive', 'unidentified')
REQUIRED_CX = ('load_market', 'btc_ret', 'book', 'touch_distance', 'level_size', 'position', 'pending', 'tau', 'over_window', 'matched_times',
               'null_summary', 'draws_per_target')
ORD_COLS = ('scenario', 'mid', 'mk_hash', 'mk_signer', 'y_side', 'y_ptick', 'n_fills', 'cum_shares', 'first_hi_ms', 'last_hi_ms',
            'persist_lo_ms', 'persist_hi_ms', 'ahead_lo', 'ahead_hi', 'removal_set', 'removal_lo_ms', 'removal_hi_ms', 'removal_class',
            'removal_method', 'status', 'x_size_ub')
FIL_COLS = ('mk_hash', 'sid', 'hi_ms', 'x_lo_hard_ms', 'x_rec_rank', 'shares', 'x_ahead_hi_S_FIFO')
LNK_COLS = ('mk_signer', 'y_side', 'kind', 'prev_hash', 'next_hash', 'prev_ptick', 'next_ptick', 'prev_last_hi_ms', 'next_first_hi_ms', 'gap_ms',
            'status')
MKT_COLS = ('mid', 'ws', 'we', 'status')
_MUT_LEAK_MS = 0          # planted-error switch (selftest only): every instant read this many ms LATER (a look-ahead the falsifier must catch)
_MUT_PHASE_FLIP = False   # planted-error switch (selftest only): presence controls drawn from the OTHER phase (a phase-mismatched control)


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def vmhwm_mib():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmHWM:'):
                    return int(line.split()[1]) / 1024.
    except OSError:
        pass
    return 0.


def guard(where, limit=C.GUARD_MIB, cpu0=None):
    m = vmhwm_mib()
    if m > limit:
        raise MemoryError(f'self-guard: VmHWM {m:.0f} MiB > {limit:.0f} MiB at {where}')
    if cpu0 is not None and time.process_time() - cpu0 > CPU_CAP_S:
        raise RuntimeError(f'self-guard: CPU {time.process_time() - cpu0:.1f} s > {CPU_CAP_S:.0f} s at {where}')
    return m


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_clean(v) for v in (sorted(o, key=str) if isinstance(o, (set, frozenset)) else o)]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o if isinstance(o, (str, int, type(None))) else str(o)


def _flags(st):
    return {x for x in str(st).split('+') if x and x != OK}


def parse_set(x):
    """removal_set text -> frozenset of members, or None when it is not a non-empty subset of pf_mt_common.REMOVAL."""
    if not isinstance(x, str) or not x:
        return None
    s = frozenset(x.split('|'))
    return s if s <= set(C.REMOVAL) and '' not in s else None


# ------------------------------------------------------------------ census (pure) ------------------------------------------------------------------
def end_histories(st, tight_v):
    """the feasible end histories of contract 4.1 under one view: EX (exhausted in the set), PL / EC (cancelled_or_expired / end_censored in
    the set and the view's size bound not tight) -> 'EX|PL', ... ('' for an unparseable set)."""
    if st is None:
        return ''
    h = [x for x, ok in (('EX', 'exhausted' in st), ('PL', 'cancelled_or_expired' in st and not tight_v),
                         ('EC', 'end_censored' in st and not tight_v)) if ok]
    return '|'.join(h)


def order_table(O, F, Lk, ws, we):
    """O: the top-3 lifecycle rows of one market (all scenarios; column 'a' = rank index), F: their fills, Lk: their x_links rows ->
    one row per hash with the per-scenario facts, the successor facts and per view class, reason, window status and end histories
    (columns class_<view>, why_<view>, wstat_<view>, hist_<view>). The pull chain's single class definition (chain_take imports it)."""
    need(set(O['scenario']) <= set(SCEN), 'lifecycle: an unknown scenario')
    want = tuple(sorted(SCEN))
    cnt = O.groupby('mk_hash')['scenario'].agg(lambda x: tuple(sorted(x)) == want)
    need(bool(cnt.all()), 'lifecycle: a hash without exactly one row per scenario (refused)')
    P = {s: O[O['scenario'] == s].set_index('mk_hash').sort_index() for s in SCEN}
    h = P[SCEN[0]].index
    for s in SCEN[1:]:
        need(P[s].index.equals(h), 'lifecycle: scenario rows of different hashes')
        for c in ('mk_signer', 'y_side', 'y_ptick', 'n_fills', 'cum_shares', 'first_hi_ms', 'last_hi_ms', 'persist_lo_ms', 'persist_hi_ms',
                  'removal_lo_ms', 'a'):
            need(bool((P[s][c].to_numpy() == P[SCEN[0]][c].to_numpy()).all()), f'lifecycle: {c} differs across scenario rows (refused)')
    B = P['S_FIFO']
    T = pd.DataFrame(dict(mk_hash=h.to_numpy(object), a=B['a'].to_numpy(np.int64), y_side=B['y_side'].to_numpy(object),
                          k=B['y_ptick'].to_numpy(np.int64), n_fills=B['n_fills'].to_numpy(np.int64), cum=B['cum_shares'].to_numpy(float),
                          first_hi=B['first_hi_ms'].to_numpy(np.int64), last_hi=B['last_hi_ms'].to_numpy(np.int64),
                          p_lo=B['persist_lo_ms'].to_numpy(np.int64), p_hi=B['persist_hi_ms'].to_numpy(np.int64),
                          rlo=B['removal_lo_ms'].to_numpy(np.int64), status=B['status'].to_numpy(object)))
    side_ok = (T['y_side'].isin(['bid', 'ask']) & T['k'].between(0, 100)).to_numpy(bool)      # else an invalid order (status, never dropped)
    T['s'] = np.where(T['y_side'] == 'bid', 1, -1).astype(np.int64)
    T['k'] = np.where(side_ok, T['k'], 0).astype(np.int64)
    T['valid'] = np.array([not (_flags(x) & M.ORDER_INVALID) for x in T['status']], bool) & side_ok
    for s in SCEN:
        T[f'set_{s}'] = P[s]['removal_set'].to_numpy(object)
        T[f'rhi_{s}'] = P[s]['removal_hi_ms'].to_numpy(np.int64)
        T[f'ub_{s}'] = P[s]['x_size_ub'].to_numpy(float)
        T[f'meth_{s}'] = P[s]['removal_method'].to_numpy(object)
        T[f'rcls_{s}'] = P[s]['removal_class'].to_numpy(object)
    T['ahead_lo_first'] = B['ahead_lo'].to_numpy(float)
    T['ahead_hi_first'] = B['ahead_hi'].to_numpy(float)
    F = F.sort_values(['mk_hash', 'hi_ms', 'x_rec_rank', 'sid'], kind='stable')
    last_ah = F.drop_duplicates('mk_hash', keep='last').set_index('mk_hash')['x_ahead_hi_S_FIFO']     # the last row (NaN kept)
    T['ahead_hi_last'] = last_ah.reindex(T['mk_hash']).to_numpy(float)
    nf_ = F.groupby('mk_hash').size().reindex(T['mk_hash']).fillna(0).to_numpy(np.int64)
    need(bool((nf_ == T['n_fills'].to_numpy()).all()), 'life fills: the fill count of a top-3 order differs from its orders row (refused)')
    L_ = Lk[(Lk['status'] == OK) & Lk['kind'].isin(['refill', 'reprice']) & (Lk['gap_ms'] <= SUCC_MS)]
    L_ = L_.sort_values(['prev_hash', 'gap_ms', 'next_hash'], kind='stable')
    first = L_.drop_duplicates('prev_hash', keep='first').set_index('prev_hash')                     # the nearest successor's own row
    kinds = L_.groupby('prev_hash')['kind'].agg(lambda x: '+'.join(sorted(set(x))))
    T['succ_kind'] = kinds.reindex(T['mk_hash']).fillna('none').to_numpy(object)
    T['succ_hash'] = first['next_hash'].reindex(T['mk_hash']).fillna('').to_numpy(object)
    T['succ_gap_ms'] = first['gap_ms'].reindex(T['mk_hash']).fillna(-1).to_numpy(np.int64)
    T['succ_first_hi'] = first['next_first_hi_ms'].reindex(T['mk_hash']).fillna(-1).to_numpy(np.int64)
    T['succ_ptick'] = first['next_ptick'].reindex(T['mk_hash']).fillna(-1).to_numpy(np.int64)
    tight_all = np.ones(len(T), bool)
    for s in SCEN:
        tight_all &= np.isfinite(T[f'ub_{s}'].to_numpy()) & (T[f'ub_{s}'].to_numpy() <= T['cum'].to_numpy() + TOL_SH)
    for v in VIEWS:
        cls, why, ws_, hist = [], [], [], []
        for i in range(len(T)):
            st = parse_set(T[f'set_{v}'].iat[i]); ub = T[f'ub_{v}'].iat[i]; cum = T['cum'].iat[i]
            rlo, rhi = int(T['rlo'].iat[i]), int(T[f'rhi_{v}'].iat[i])
            tight_v = bool(np.isfinite(ub) and ub <= cum + TOL_SH)
            if not T['valid'].iat[i] or st is None:
                c_, w_ = 'unidentified', 'order_invalid' if not T['valid'].iat[i] else 'bad_removal_set'
            elif v == 'S_FIFO' and st == frozenset(('exhausted',)):
                c_, w_ = 'exhausted_proven', ''
            elif tight_all[i]:
                c_, w_ = 'capacity_tight', ''
            elif tight_v:
                c_, w_ = 'capacity_tight_view', ''
            elif 'missing_evidence' in st:
                c_, w_ = 'unidentified', 'missing_evidence'
            elif 'end_censored' in st:
                c_, w_ = 'end_censored_possible', ''
            elif 'cancelled_or_expired' in st and rlo >= 0 and rhi >= 0:
                c_, w_ = 'cancel_possible', ''
            else:
                c_, w_ = 'unidentified', 'window_unbounded' if 'cancelled_or_expired' in st else 'no_cancel_member'
            if not T['valid'].iat[i]:
                wst = 'order_invalid'                  # no cell / priority use for an invalid order: never evaluated, counted
            elif rlo < 0 or rhi < 0:
                wst = 'window_unbounded'
            elif rlo < ws - PRE_MS:
                wst = 'before_pre'
            elif rlo < ws <= rhi:
                wst = 'straddles_ws'
            elif rhi >= we:
                wst = 'reaches_we'
            else:
                wst = 'ok'
            cls.append(c_); why.append(w_); ws_.append(wst); hist.append(end_histories(st, tight_v))
        T[f'class_{v}'] = cls
        T[f'why_{v}'] = why
        T[f'wstat_{v}'] = ws_
        T[f'hist_{v}'] = hist
    T['lo_rhi_differs'] = T['rhi_S_FIFO_LO'].to_numpy() != T['rhi_S_FIFO'].to_numpy()
    return T


def phase_of(lo, ws):
    """0 = pre [ws - PRE_MS, ws), 1 = live [ws, we) (an ok window lies in one phase)."""
    return (np.asarray(lo) >= ws).astype(np.int64)


def phase_bounds(ph, ws, we):
    """[start, end) of phase ph."""
    return (ws - PRE_MS, ws) if ph == 0 else (ws, we)


# ------------------------------------------------------------------ instants and windows (chain_context.over_window's set) ------------------------------------------------------------------
def interior(G, lo, hi):
    """grid rows strictly inside (lo, hi): a = lo // 100 + 1, b = (hi - 1) // 100 (over_window's g1 / g2), as rows of the market grid."""
    lo, hi = np.asarray(lo, np.int64) + _MUT_LEAK_MS, np.asarray(hi, np.int64) + _MUT_LEAK_MS
    a = np.floor_divide(lo, STEP) + 1 - G['g0']
    b = np.floor_divide(hi - 1, STEP) - G['g0']
    return a, b, b < a


def sparse(x, fn):
    """sparse table for range fn (np.fmin / np.fmax: NaN ignored) over a 1-D array."""
    T = [np.asarray(x, float)]
    j = 1
    while (1 << j) <= len(x):
        p = T[-1]
        T.append(fn(p[:len(p) - (1 << (j - 1))], p[(1 << (j - 1)):]))
        j += 1
    return T


def rq(T, i0, i1, fn):
    """range fn over [i0, i1] (inclusive; i0 <= i1, arrays)."""
    n = (i1 - i0 + 1).astype(np.int64)
    j = np.floor(np.log2(np.maximum(n, 1))).astype(np.int64)
    out = np.full(i0.shape, np.nan)
    for jj in np.unique(j):
        m = j == jj
        out[m] = fn(T[jj][i0[m]], T[jj][i1[m] - (1 << int(jj)) + 1])
    return out


def build_grid(S, ws, we):
    """S: per-instant series on the aligned market grid (context_series) -> G: variant arrays V, raw BTC returns R, prefix sums of every
    MECH condition and of its validity, sparse min / max tables of the BTC returns. Variant index: side 0 bid / 1 ask; cell = S['cells']
    index; addr 0..2; addr_side = 2 a + side; mkt 0."""
    T = len(S['t'])
    G = dict(g0=int(S['t'][0] // STEP), T=T, t=S['t'], ws=ws, we=we, cells=list(S['cells']), cell_ix={c: i for i, c in enumerate(S['cells'])},
             cell_tick=np.array([int(k) for _, k in S['cells']], np.int64))

    def lag(x, k):
        y = np.full(x.shape, np.nan)
        if x.shape[-1] > k:
            y[..., k:] = x[..., :-k]
        return y
    K5 = LAGROWS
    sg = np.array([1., -1.])
    V = {}
    for nm, _, _ in MOVE_BP:
        V[f'a_{nm}'] = sg[:, None] * S[f'r_{nm}'][None, :]                  # side x T
    opp = np.stack([S['ask'], S['bid']])                                    # the opposite touch of a bid is the ask
    with np.errstate(invalid='ignore'):
        ot = sg[:, None] * (lag(opp, K5) - opp)
        V['mid_a5'] = sg[:, None] * (S['mid'][None, :] - lag(S['mid'], K5)[None, :])
    V['opp_tw5'] = np.where(np.isfinite(ot), ot, np.nan)                    # an empty side at either end: unknown (chain_context book_change)
    V['dist'] = S['dist']                                                   # cell x T
    V['depth'] = S['depth']
    V['ddepth5'] = S['depth'] - lag(S['depth'], K5)
    V['nf1'] = S['nlegs'] - lag(S['nlegs'], 1_000 // STEP)                   # addr x T: legs with settle_ms in [c - 1 s, c)
    V['nf5'] = S['nlegs'] - lag(S['nlegs'], K5)
    pos = S['pos']
    V['dpos5'] = (sg[None, :, None] * (pos - lag(pos, K5))[:, None, :]).reshape(-1, T)       # addr_side x T
    V['spos'] = (sg[None, :, None] * pos[:, None, :]).reshape(-1, T)
    V['spend'] = (sg[None, :, None] * S['pend'][:, None, :]).reshape(-1, T)
    V['tau'] = S['tau'][None, :]
    G['V'] = V
    G['R'] = {nm: np.asarray(S[f'r_{nm}'], float) for nm, _, _ in MOVE_BP}
    G['P'], G['Q'] = {}, {}
    for name, ser, _, op, thr in MECHS:
        ok, c = cond_of(V[ser], op, thr)
        z = np.zeros(V[ser].shape[:-1] + (1,), np.int32)
        G['P'][name] = np.concatenate([z, np.cumsum(c, -1, dtype=np.int32)], -1)
        G['Q'][name] = np.concatenate([z, np.cumsum(ok, -1, dtype=np.int32)], -1)
    G['mn'] = {nm: sparse(G['R'][nm], np.fmin) for nm, _, _ in MOVE_BP}
    G['mx'] = {nm: sparse(G['R'][nm], np.fmax) for nm, _, _ in MOVE_BP}
    G['rv'] = {nm: np.r_[0, np.cumsum(np.isfinite(G['R'][nm]), dtype=np.int64)] for nm, _, _ in MOVE_BP}
    return G


def cond_of(x, op, thr):
    """(valid, condition) of a MECH series: valid = not NaN (dist -inf, an empty own side, is a valid 'inside'), condition false when invalid."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    with np.errstate(invalid='ignore'):
        c = (x <= thr) if op == 'le' else (x >= thr) if op == 'ge' else (x > thr)
    return ok, c & ok


def _variant(kind, side, cell, addr):
    return dict(side=side, cell=cell, addr=addr, addr_side=2 * addr + side, mkt=np.zeros_like(side))[kind]


def point_values(cx, M_, x, side, tick, addr):
    """every END_NAMES series at arbitrary instants x (int ms) through chain_context point queries (the same quantities context_series
    reads on the grid): order side 0 / 1, tick, address index. Data <= x only (pending: the declared reconstruction exception)."""
    x = np.asarray(x, np.int64).ravel() + _MUT_LEAK_MS
    side, tick, addr = (np.asarray(v, np.int64).ravel() for v in (side, tick, addr))
    n = len(x)
    out = {k: np.full(n, np.nan) for k in END_NAMES}
    if n == 0:
        return out
    sg = np.where(side == 0, 1., -1.)
    b = cx.btc_ret(M_, x)
    for nm, _, _ in MOVE_BP:
        r = np.asarray(b[f'r_{nm}'], float)
        out[f'r_{nm}'], out[f'a_{nm}'] = r, sg * r
    b1, b0 = cx.book(M_, x), cx.book(M_, x - LOOKBACK)
    is_bid = side == 0
    opp1 = np.where(is_bid, np.asarray(b1['ask_tick'], float), np.asarray(b1['bid_tick'], float))
    opp0 = np.where(is_bid, np.asarray(b0['ask_tick'], float), np.asarray(b0['bid_tick'], float))
    with np.errstate(invalid='ignore'):
        ot = sg * (opp0 - opp1)
        out['mid_a5'] = sg * (np.asarray(b1['mid_c'], float) - np.asarray(b0['mid_c'], float))
    out['opp_tw5'] = np.where(np.isfinite(ot), ot, np.nan)
    sides = np.where(is_bid, 'bid', 'ask').astype(object)
    best = np.where(is_bid, np.asarray(b1['bid_tick'], float), np.asarray(b1['ask_tick'], float))
    with np.errstate(invalid='ignore'):                      # chain_context.touch_distance's formula on the same book() call (checked
        out['dist'] = np.where(is_bid, best - tick, tick - best)    # per market against touch_distance on the grid; avoids its per-element side loop)
    d1 = np.asarray(cx.level_size(M_, x, sides, tick)['size'], float)
    d0 = np.asarray(cx.level_size(M_, x - LOOKBACK, sides, tick)['size'], float)
    out['depth'], out['ddepth5'] = d1, d1 - d0
    for a in np.unique(addr):
        m = addr == a
        xa, A = x[m], TOP3[int(a)]
        p1, p5, p01 = (cx.position(M_, A, xa - lagm, clock='settle') for lagm in (0, LOOKBACK, 1_000))
        r1, r5 = np.asarray(p1['recorded'], float), np.asarray(p5['recorded'], float)
        n1, n5, n01 = (np.asarray(p['n'], float) for p in (p1, p5, p01))
        out['nf1'][m], out['nf5'][m] = n1 - n01, n1 - n5
        out['dpos5'][m], out['spos'][m] = sg[m] * (r1 - r5), sg[m] * r1
        out['spend'][m] = sg[m] * np.asarray(cx.pending(M_, A, xa)['net'], float)
    out['tau'] = np.asarray(cx.tau(M_, x), float)
    return out


def endpoint_values(cx, M_, G, x, side, cell, addr):
    """END_NAMES at instants x: the grid value when x is an aligned grid instant past the lag rows (equal to the point query by
    construction; checked per market), else a chain_context point query."""
    x, side, cell, addr = (np.asarray(v, np.int64).ravel() for v in (x, side, cell, addr))
    n = len(x)
    rows = np.floor_divide(x, STEP) - G['g0']
    use_g = (x % STEP == 0) & (rows >= LAGROWS) & (rows < G['T']) & (_MUT_LEAK_MS == 0)
    out = {k: np.full(n, np.nan) for k in END_NAMES}
    if use_g.any():
        r_, sd, ce, ad = rows[use_g], side[use_g], cell[use_g], addr[use_g]
        for nm, kind in START_FEATS:
            out[nm][use_g] = G['V'][nm][_variant(kind, sd, ce, ad), r_]
        for nm, _, _ in MOVE_BP:
            out[f'r_{nm}'][use_g] = G['R'][nm][r_]
    if (~use_g).any():
        sel = ~use_g
        pv = point_values(cx, M_, x[sel], side[sel], G['cell_tick'][cell[sel]], addr[sel])
        for k in END_NAMES:
            out[k][sel] = pv[k]
    return out


def eval_windows(G, lo, hi, side, cell, addr, E0, E1):
    """Windows [lo, hi] (ms; hi >= lo) over over_window's instants {lo, grid instants strictly inside, hi}; E0 / E1 = END_NAMES values at
    lo / hi (endpoint_values). -> every CONT feature (start = the value at lo) and per MECH the counts (true, valid, n) over the instants;
    'inside' = the interior rows lie in the market grid (else NaN / zero counts)."""
    shp = np.shape(lo)
    lo, hi, side, cell, addr = (np.asarray(v, np.int64).ravel() for v in (lo, hi, side, cell, addr))
    E0 = {k: np.asarray(v, float).ravel() for k, v in E0.items()}
    E1 = {k: np.asarray(v, float).ravel() for k, v in E1.items()}
    a, b, none = interior(G, lo, hi)
    single = hi == lo
    inside = none | ((a >= 0) & (b < G['T']))
    ac = np.clip(a, 0, G['T'] - 1)
    bc = np.clip(b, 0, G['T'] - 1)
    use = inside & ~none                                         # windows with interior rows in the grid
    nin = np.where(use, bc - ac + 1, 0)
    n = np.where(single, 1, nin + 2)
    out = {}
    for nm, _ in START_FEATS:
        out[f'{nm}_start'] = np.where(inside, E0[nm], np.nan).reshape(shp)
    sg = np.where(side == 0, 1., -1.)
    for nm, _, _ in MOVE_BP:
        imn, imx = np.full(len(lo), np.nan), np.full(len(lo), np.nan)
        cntf = np.zeros(len(lo), np.int64)
        if use.any():
            imn[use] = rq(G['mn'][nm], ac[use], bc[use], np.fmin)
            imx[use] = rq(G['mx'][nm], ac[use], bc[use], np.fmax)
            cntf[use] = G['rv'][nm][bc[use] + 1] - G['rv'][nm][ac[use]]
        e0, e1 = E0[f'r_{nm}'], np.where(single, np.nan, E1[f'r_{nm}'])
        mn = np.fmin(np.fmin(imn, e0), e1)
        mx = np.fmax(np.fmax(imx, e0), e1)
        full = (cntf == nin) & np.isfinite(e0) & (single | np.isfinite(E1[f'r_{nm}']))
        a_mn, a_mx = np.where(sg > 0, mn, -mx), np.where(sg > 0, mx, -mn)
        out[f'a_{nm}_wmin'] = np.where(inside & full, a_mn, np.nan).reshape(shp)
        out[f'a_{nm}_wmax'] = np.where(inside & full, a_mx, np.nan).reshape(shp)
    for name, ser, kind, op, thr in MECHS:
        v = _variant(kind, side, cell, addr)
        Pp, Qq = G['P'][name], G['Q'][name]
        tr = np.where(use, Pp[v, bc + 1] - Pp[v, ac], 0).astype(np.int64)
        va = np.where(use, Qq[v, bc + 1] - Qq[v, ac], 0).astype(np.int64)
        o0, c0 = cond_of(E0[ser], op, thr)
        o1, c1 = cond_of(E1[ser], op, thr)
        tr = tr + c0 + np.where(single, False, c1)
        va = va + o0 + np.where(single, False, o1)
        out[name] = (np.where(inside, tr, 0).reshape(shp), np.where(inside, va, 0).reshape(shp), np.where(inside, n, 0).reshape(shp))
    out['inside'] = inside.reshape(shp)
    return out


def classify(tr, va, n):
    """per window: 0 compatible, 1 contradicted, 2 timing_sensitive, 3 unidentified (arrays of counts)."""
    full = (va == n) & (n > 0)
    c = np.full(np.shape(n), 3, np.int8)
    c[(tr > 0) & (tr < va)] = 2
    c[full & (tr == n)] = 0
    c[full & (tr == 0)] = 1
    return c


def windows_at(cx, M_, G, lo_list, hi, side, cell, addr):
    """per D profile the eval_windows dict of the windows [lo_d, hi] (one shared upper end) -> list."""
    E1 = endpoint_values(cx, M_, G, hi, side, cell, addr)
    return [eval_windows(G, lo_d, hi, side, cell, addr, endpoint_values(cx, M_, G, lo_d, side, cell, addr), E1) for lo_d in lo_list]


# ------------------------------------------------------------------ controls ------------------------------------------------------------------
def coverage_mask(lo, hi):
    """mask_fn(M, g) for 'some interval [lo_i, hi_i] holds g' (closed intervals)."""
    a, b = np.sort(np.asarray(lo, np.int64)), np.sort(np.asarray(hi, np.int64))

    def fn(M_, g):
        g = np.asarray(g, np.int64)
        return (np.searchsorted(a, g, 'right') - np.searchsorted(b, g, 'left')) > 0
    return fn


def presence_controls(cx, M_, D, T, act, cnt):
    """contract 7.2 controls through chain_context.matched_times: per ok event N_DRAWS aligned starts c (-1 none) -> (C0, status)."""
    n_ev = len(D)
    C0 = np.full((n_ev, N_DRAWS), -1, np.int64)
    st = np.full(n_ev, 'n/a', dtype=object)
    okw = (D['wst'] == 'ok').to_numpy()
    if not okw.any():
        return C0, st
    dpt = int(cx.draws_per_target(int(okw.sum())))                          # the helper's minimum for >= 200 pooled draws
    need(N_DRAWS >= dpt, f'N_DRAWS {N_DRAWS} below chain_context.draws_per_target {dpt}')
    cnt['draws_per_target_market_max'] = max(cnt.get('draws_per_target_market_max', 0), dpt)
    ev = D[okw]
    wb = -(-ev['w'].to_numpy(np.int64) // STEP)                             # ceil(w / 100): the mask covers the whole window
    for (vi, a_, s_, w_), idx in pd.DataFrame(dict(vi=ev['vi'].to_numpy(), a=ev['a'].to_numpy(), s=ev['s'].to_numpy(), w=wb),
                                              index=ev.index).groupby(['vi', 'a', 's', 'w']).groups.items():
        idx = np.asarray(idx)
        Tm = T[(T['a'].to_numpy() == a_) & (T['s'].to_numpy() == s_) & T['valid'].to_numpy(bool) & (T['p_lo'].to_numpy() >= 0)
               & (T['p_hi'].to_numpy() >= T['p_lo'].to_numpy())]
        mask = coverage_mask(Tm['p_lo'].to_numpy(), Tm['p_hi'].to_numpy())
        alo, ahi = act.get(int(a_), (-1, -1))
        fb = coverage_mask([alo], [ahi]) if alo >= 0 else None
        tt = D.loc[idx, 'rlo'].to_numpy(np.int64)
        if _MUT_PHASE_FLIP:
            po = np.where(D.loc[idx, 'ph'].to_numpy() == 1, 'pre', 'live').astype(object)
        else:
            po = tt
        side = 'bid' if s_ == 1 else 'ask'
        mt = cx.matched_times(M_, TOP3[int(a_)], n=N_DRAWS, seed=SEED, phase_of=po, mask_fn=mask, width_ms=int(w_) * STEP,
                              fallback_mask_fn=fb, key=f'chain_pull|{VIEWS[int(vi)]}|{side}|w{int(w_)}')
        tm = np.asarray(mt['t'], np.int64)
        need(tm.shape == (len(idx), N_DRAWS), 'matched_times returned an unexpected shape')
        need(bool((tm[tm >= 0] % STEP == 0).all()), 'matched_times returned starts off the aligned grid (refused)')
        pos_ = D.index.get_indexer(idx)
        C0[pos_] = tm
        st[pos_] = np.asarray(mt['status'], dtype=object)
        for x_ in np.asarray(mt['status'], dtype=object):
            cnt[f'controls:{VIEWS[int(vi)]}:presence:{x_}'] += 1
    return C0, st


def anchors_of(T, F, a, s, w, ph, ws, we):
    """fill_anchored candidates (aligned starts c = the first grid instant >= the fill's x_lo_hard) for the non-last fills of the address's
    valid same-side orders whose window [c, c + w] lies inside phase ph and ends by that order's persist_hi."""
    p0, p1 = phase_bounds(ph, ws, we)
    rows = np.flatnonzero((T['a'].to_numpy() == a) & (T['s'].to_numpy() == s) & T['valid'].to_numpy(bool) & (T['n_fills'].to_numpy() >= 2))
    cs = []
    for j in rows:
        fl = F.get(T['mk_hash'].iat[j])
        if fl is None or len(fl) < 2:
            continue
        c = fl[:-1]
        c = c[c >= 0]
        c = -(-c // STEP) * STEP                                            # the first grid instant at or after x_lo_hard
        ok = (c >= p0) & (c + w < p1) & (c + w <= int(T['p_hi'].iat[j]))
        cs.append(c[ok])
    return np.concatenate(cs) if cs else np.zeros(0, np.int64)


def anchored_controls(D, T, F_anchor, ws, we, rngs, cnt):
    """the supplementary fill_anchored controls (declared beyond contract 7.2): per ok event N_DRAWS anchors -> (C0, status)."""
    n_ev = len(D)
    C0 = np.full((n_ev, N_DRAWS), -1, np.int64)
    st = np.full(n_ev, 'n/a', dtype=object)
    okw = (D['wst'] == 'ok').to_numpy()
    hr = np.argsort(np.argsort(T['mk_hash'].to_numpy().astype(str), kind='stable'), kind='stable')
    order = np.lexsort((hr[D['o'].to_numpy()], D['rlo'].to_numpy(), D['vi'].to_numpy()))
    vi_, a_, s_, w_, ph_ = (D[c].to_numpy() for c in ('vi', 'a', 's', 'w', 'ph'))
    w_ = -(-np.maximum(w_, 0) // STEP) * STEP                               # the control width on the grid (as the presence controls)
    cache = {}
    for e in order:
        if not okw[e]:
            continue
        key = (int(a_[e]), int(s_[e]), int(w_[e]), int(ph_[e]))
        if key not in cache:
            cache[key] = anchors_of(T, F_anchor, *key, ws, we)
        ca = cache[key]
        if len(ca):
            C0[e] = ca[rngs[(int(vi_[e]), int(a_[e]))].integers(0, len(ca), N_DRAWS)]
            st[e] = 'anchors'
        else:
            st[e] = 'none'
        cnt[f'controls:{VIEWS[int(vi_[e])]}:fill_anchored:{st[e]}'] += 1
    return C0, st


# ------------------------------------------------------------------ accumulator and statistics ------------------------------------------------------------------
class Acc:
    """per (view, group, kind, addr, day, D(+union), feature): obs sum / n and per-draw replicate sums / n (dense)."""

    def __init__(self, ndays):
        sh = (len(VIEWS), len(GROUPS), len(KINDS), 3, ndays, ND + 1, NF)
        self.os, self.on = np.zeros(sh), np.zeros(sh)                 # counts kept in float64 (exact below 2**53; BLAS matmuls)
        self.rs, self.rn = np.zeros(sh + (N_DRAWS,)), np.zeros(sh + (N_DRAWS,))
        self.ne = np.zeros((len(VIEWS), len(GROUPS), len(KINDS), 3, ndays), np.int64)       # events with an ok window
        self.nc = np.zeros((len(VIEWS), len(GROUPS), len(KINDS), 3, ndays), np.int64)       # ... of them with >= 1 draw

    @staticmethod
    def masks(gmask, a):
        """[(address index, float [n_groups, n_ev] mask)] of the addresses present among the events."""
        out = []
        for aa in range(3):
            ma = gmask & (a == aa)[None, :]
            if ma.any():
                out.append((aa, ma.astype(float)))
        return out

    def add(self, vi, MA, ki, di, dj, fj, x_obs, x_ctl):
        """x_obs [n_ev], x_ctl [n_ev, N_DRAWS] (NaN = invalid); MA = Acc.masks(group mask, address index) of these events."""
        vo = np.isfinite(x_obs)
        vc = np.isfinite(x_ctl)
        use = vo & vc.any(1)
        uf = use.astype(float)
        xo = np.where(use, x_obs, 0.)
        vcu = vc & use[:, None]
        xc = np.where(vcu, x_ctl, 0.)
        vf = vcu.astype(float)
        for aa, mf in MA:
            self.os[vi, :, ki, aa, di, dj, fj] += mf @ xo
            self.on[vi, :, ki, aa, di, dj, fj] += mf @ uf
            self.rs[vi, :, ki, aa, di, dj, fj] += mf @ xc
            self.rn[vi, :, ki, aa, di, dj, fj] += mf @ vf


def summarize(acc, days, cx):
    """-> summary DataFrame over every scope with n_events > 0; obs against its N_DRAWS group-statistic replicates through
    chain_context.null_summary (null mean, p05 / p50 / p95, p_upper, p_lower)."""
    rows = []
    ndays = len(days)
    for vi, v in enumerate(VIEWS):
        for gi, g in enumerate(GROUPS):
            for ki, kd in enumerate(KINDS):
                for ai in range(4):
                    aa = slice(0, 3) if ai == 3 else slice(ai, ai + 1)
                    for di in range(ndays + 1):
                        dd = slice(0, ndays) if di == ndays else slice(di, di + 1)
                        ne = int(acc.ne[vi, gi, ki, aa, dd].sum())
                        if ne == 0:
                            continue
                        nc = int(acc.nc[vi, gi, ki, aa, dd].sum())
                        os_ = acc.os[vi, gi, ki, aa, dd].sum((0, 1)); on = acc.on[vi, gi, ki, aa, dd].sum((0, 1))
                        rs = acc.rs[vi, gi, ki, aa, dd].sum((0, 1)); rn = acc.rn[vi, gi, ki, aa, dd].sum((0, 1))
                        for dj in range(ND + 1):
                            for fj, f in enumerate(FEATS):
                                if dj == ND and fj < NCONT:
                                    continue                         # the union exists for the MECH classes only
                                n_o = int(on[dj, fj])
                                obs = float(os_[dj, fj] / n_o) if n_o else NAN
                                with np.errstate(invalid='ignore', divide='ignore'):
                                    rep = np.where(rn[dj, fj] > 0, rs[dj, fj] / np.maximum(rn[dj, fj], 1), np.nan)
                                ns = cx.null_summary(obs if n_o else None, rep)
                                rows.append((v, g, kd, (RANKS + ('pooled',))[ai], (tuple(days) + ('all',))[di],
                                             'union' if dj == ND else str(DELAYS[dj]), f, 'rate' if fj >= NCONT else 'mean', ne, nc, n_o,
                                             int(rn[dj, fj].sum()), int(ns['n']), ns['obs'], ns['mean'], ns['p05'], ns['p50'], ns['p95'],
                                             (ns['obs'] - ns['mean']) if np.isfinite(ns['mean']) else NAN, ns['p_upper'], ns['p_lower']))
    cols = ['view', 'group', 'control', 'scope', 'day', 'D', 'feature', 'kind', 'n_events', 'n_with_controls', 'n_obs', 'n_ctl_valid', 'n_reps',
            'obs', 'null_mean', 'null_p05', 'null_p50', 'null_p95', 'diff', 'p_upper', 'p_lower']
    return pd.DataFrame(rows, columns=cols)


# ------------------------------------------------------------------ one market ------------------------------------------------------------------
def market_events(T, ws, we):
    """event table: one row per (order, view)."""
    ev = []
    for vi, v in enumerate(VIEWS):
        for j in range(len(T)):
            ev.append(dict(o=j, vi=vi, cls=T[f'class_{v}'].iat[j], rlo=int(T['rlo'].iat[j]), rhi=int(T[f'rhi_{v}'].iat[j]),
                           wst=T[f'wstat_{v}'].iat[j], hist=T[f'hist_{v}'].iat[j], a=int(T['a'].iat[j]), s=int(T['s'].iat[j]),
                           side=0 if T['s'].iat[j] == 1 else 1, k=int(T['k'].iat[j]), succ=T['succ_kind'].iat[j] != 'none'))
    D = pd.DataFrame(ev)
    if not len(D):
        return D
    D['w'] = np.where(D['wst'] == 'ok', D['rhi'] - D['rlo'], -1)
    D['ph'] = np.where(D['wst'] == 'ok', phase_of(D['rlo'].to_numpy(), ws), -1)
    return D


def group_mask(D):
    gm = np.zeros((len(GROUPS), len(D)), bool)
    for gi, g in enumerate(GROUPS):
        if g == 'cp_successor_5s':
            gm[gi] = (D['cls'] == 'cancel_possible').to_numpy() & D['succ'].to_numpy(bool)
        elif g == 'cp_no_successor_5s':
            gm[gi] = (D['cls'] == 'cancel_possible').to_numpy() & ~D['succ'].to_numpy(bool)
        else:
            gm[gi] = (D['cls'] == g).to_numpy()
    return gm


def cell_matrix(G):
    """(side 0/1, tick 0..100) -> the cell index of the context series (-1 = not loaded)."""
    cm = np.full((2, 101), -1, np.int64)
    for (sd, k), i in G['cell_ix'].items():
        cm[int(sd), int(k)] = i
    return cm


def check_context(cx, M_, G, D, EVF, cell, cnt):
    """declared per-market equivalence checks against chain_context (refused on a mismatch): (1) every CHECK_EVERY-th grid instant past the
    30 s lookback, every cell x address: the grid series equal point queries; (2) the first CHECK_K ok events, D = 0 and 250: the MECH
    category sets, instant and valid counts and the BTC aligned min / max equal chain_context.over_window's."""
    js = np.arange(LAGROWS + 300, G['T'], CHECK_EVERY)
    if len(js) and G['cells']:
        nc_ = len(G['cells'])
        ci = np.repeat(np.arange(nc_), 3 * len(js))
        ai = np.tile(np.repeat(np.arange(3), len(js)), nc_)
        rr = np.tile(js, 3 * nc_)
        sd = np.array([G['cells'][c][0] for c in ci], np.int64)
        pv = point_values(cx, M_, G['t'][rr], sd, G['cell_tick'][ci], ai)
        for nm, kind in START_FEATS:
            g_ = G['V'][nm][_variant(kind, sd, ci, ai), rr]
            p_ = pv[nm]
            need(bool(np.array_equal(np.isnan(g_), np.isnan(p_)) and np.array_equal(g_[~np.isnan(g_)], p_[~np.isnan(p_)])),
                 f'context check: grid series {nm} differs from chain_context point queries')
        for nm, _, _ in MOVE_BP:
            g_, p_ = G['R'][nm][rr], pv[f'r_{nm}']
            need(bool(np.array_equal(np.isnan(g_), np.isnan(p_)) and np.array_equal(g_[~np.isnan(g_)], p_[~np.isnan(p_)])),
                 f'context check: grid BTC r_{nm} differs from chain_context.btc_ret')
        cnt['check_grid_points'] += int(len(rr))
    ok_e = np.flatnonzero((D['wst'] == 'ok').to_numpy())[:CHECK_K]
    for e in ok_e:
        lo, hi = int(D['rlo'].iat[e]), int(D['rhi'].iat[e])
        sd_, ce_, a_ = int(D['side'].iat[e]), int(cell[e]), int(D['a'].iat[e])
        tk_ = int(G['cell_tick'][ce_])
        for dj, Dm in ((0, 0), (1, 250)):
            for name, ser, _, op, thr in MECHS:
                def fn(M2, tt, ser=ser, op=op, thr=thr):
                    pv_ = point_values(cx, M2, tt, np.full(len(tt), sd_), np.full(len(tt), tk_), np.full(len(tt), a_))
                    ok_, c_ = cond_of(pv_[ser], op, thr)
                    return c_, ok_
                ow = cx.over_window(M_, fn, lo, hi, D_ms=Dm, mode='category')
                tr, va, n = (int(x[e]) for x in EVF[dj][name])
                mine = '|'.join(x for x, has in (('False', va - tr > 0), ('True', tr > 0)) if has)
                need(str(ow['cats'][0]) == mine and int(ow['n_points'][0]) == n and int(ow['n_valid'][0]) == va,
                     f'context check: {name} D={Dm} cats {mine!r} / n {n} / valid {va} differ from over_window '
                     f'{ow["cats"][0]!r} / {ow["n_points"][0]} / {ow["n_valid"][0]}')
            for nm, _, _ in MOVE_BP:
                def fr(M2, tt, nm=nm):
                    pv_ = point_values(cx, M2, tt, np.full(len(tt), sd_), np.full(len(tt), tk_), np.full(len(tt), a_))
                    return pv_[f'a_{nm}'], ~np.isnan(pv_[f'a_{nm}'])
                ow = cx.over_window(M_, fr, lo, hi, D_ms=Dm, mode='sign')
                for q in ('min', 'max'):
                    mine = float(EVF[dj][f'a_{nm}_w{q}'][e])
                    if np.isfinite(mine):
                        need(float(ow[q][0]) == mine, f'context check: a_{nm} {q} D={Dm} {mine!r} differs from over_window {ow[q][0]!r}')
            cnt['check_windows'] += 1


def run_market(cx, M_, G, T, F_anchor, act, ws, we, day_idx, acc, rngs, ctl_rows, pull_rows, mid, day, cnt):
    """Every event of one market (both views): windows, controls, statistics, pulls.csv / controls.csv rows, the context checks."""
    D = market_events(T, ws, we)
    if not len(D):
        return
    CM = cell_matrix(G)
    vv = T['valid'].to_numpy(bool)[D['o'].to_numpy()]                 # invalid orders: no cell, never evaluated (window status order_invalid)
    cell = np.where(vv, CM[D['side'].to_numpy(np.int64), np.clip(D['k'].to_numpy(np.int64), 0, 100)], 0)
    need(bool((cell[vv] >= 0).all()), 'grid: an event cell missing from the context series')
    cell = np.maximum(cell, 0)
    okw = (D['wst'] == 'ok').to_numpy()
    bnd = (D['rlo'] >= 0).to_numpy() & (D['rhi'] >= D['rlo']).to_numpy() & vv
    gm = group_mask(D)
    side, addr = D['side'].to_numpy(np.int64), D['a'].to_numpy(np.int64)
    rlo, rhi = np.where(bnd, D['rlo'].to_numpy(), 0), np.where(bnd, D['rhi'].to_numpy(), 0)
    # ---- the events' windows [removal_lo - D, removal_hi] (bounded only; ok ones enter the statistics) ----
    EVF = [dict() for _ in DELAYS]
    sb = np.flatnonzero(bnd)
    if len(sb):
        W = windows_at(cx, M_, G, [rlo[sb] - Dm for Dm in DELAYS], rhi[sb], side[sb], cell[sb], addr[sb])
        for dj in range(ND):
            for k_, v_ in W[dj].items():
                if isinstance(v_, tuple):
                    full = tuple(np.zeros(len(D), np.int64) for _ in v_)
                    for z in range(3):
                        full[z][sb] = v_[z]
                    EVF[dj][k_] = full
                else:
                    arr = np.zeros(len(D), bool) if k_ == 'inside' else np.full(len(D), np.nan)
                    arr[sb] = v_
                    EVF[dj][k_] = arr
    else:
        for dj in range(ND):
            EVF[dj] = {**{f: np.full(len(D), np.nan) for f in CONT}, **{m: tuple(np.zeros(len(D), np.int64) for _ in range(3)) for m in MNAMES},
                       'inside': np.zeros(len(D), bool)}
    check_context(cx, M_, G, D, EVF, cell, cnt)
    # ---- controls: presence (matched_times) and fill_anchored; control window [c - D, c + ceil(w / 100) x 100] (the upper end on the grid,
    #      as matched_times checks its mask over the window) ----
    w_ev = -(-np.maximum(D['w'].to_numpy(np.int64), 0) // STEP) * STEP
    strat = {}
    for ki, kd in enumerate(KINDS):
        if kd == 'presence':
            C0, st = presence_controls(cx, M_, D, T, act, cnt)
        else:
            C0, st = anchored_controls(D, T, F_anchor, ws, we, rngs, cnt)
        strat[kd] = st
        valid = okw[:, None] & (C0 >= 0)
        fi = np.flatnonzero(valid.ravel())
        er = fi // N_DRAWS
        CF = [dict() for _ in DELAYS]
        if len(fi):
            c_ = C0.ravel()[fi]
            Wc = windows_at(cx, M_, G, [c_ - Dm for Dm in DELAYS], c_ + w_ev[er], side[er], cell[er], addr[er])
        for dj in range(ND):
            for f in CONT:
                x = np.full(len(D) * N_DRAWS, np.nan)
                if len(fi):
                    x[fi] = Wc[dj][f]
                CF[dj][f] = x.reshape(len(D), N_DRAWS)
            for m in MNAMES:
                trio = []
                for z in range(3):
                    x = np.zeros(len(D) * N_DRAWS, np.int64)
                    if len(fi):
                        x[fi] = Wc[dj][m][z]
                    trio.append(x.reshape(len(D), N_DRAWS))
                CF[dj][m] = tuple(trio)
            ins = np.zeros(len(D) * N_DRAWS, bool)
            if len(fi):
                ins[fi] = Wc[dj]['inside']
            CF[dj]['inside'] = ins.reshape(len(D), N_DRAWS)
        has = valid.any(1)
        vi_arr = D['vi'].to_numpy()
        for vi in range(len(VIEWS)):
            sel = okw & (vi_arr == vi)
            if not sel.any():
                continue
            g_sel, a_sel = gm[:, sel], addr[sel]
            acc.ne[vi, :, ki, :, day_idx] += np.stack([(g_sel & (a_sel == aa)[None, :]).sum(1) for aa in range(3)], 1)
            acc.nc[vi, :, ki, :, day_idx] += np.stack([(g_sel & (a_sel == aa)[None, :] & has[sel][None, :]).sum(1) for aa in range(3)], 1)
            MA = Acc.masks(g_sel, a_sel)
            ucnt = {}
            for dj in range(ND):
                fe, fc = EVF[dj], CF[dj]
                for fj, nm in enumerate(CONT):
                    acc.add(vi, MA, ki, day_idx, dj, fj, np.where(fe['inside'][sel], fe[nm][sel], np.nan),
                            np.where(fc['inside'][sel], fc[nm][sel], np.nan))
                for mi, name in enumerate(MNAMES):
                    te, ve, ne_ = (x[sel] for x in fe[name]); tc, vc_, nc_ = (x[sel] for x in fc[name])
                    ce = np.where(fe['inside'][sel], classify(te, ve, ne_), 3); cc = np.where(fc['inside'][sel], classify(tc, vc_, nc_), 3)
                    for q, fj in ((0, NCONT + 2 * mi), (1, NCONT + 2 * mi + 1)):
                        yo = np.where(ce == 3, np.nan, (ce == 0) if q == 0 else ((ce == 0) | (ce == 2))).astype(float)
                        yc = np.where(cc == 3, np.nan, (cc == 0) if q == 0 else ((cc == 0) | (cc == 2))).astype(float)
                        acc.add(vi, MA, ki, day_idx, dj, fj, yo, yc)
                    u = ucnt.setdefault(name, [0, 0, 0, 0, 0, 0])
                    for z, x in enumerate((te, ve, ne_, tc, vc_, nc_)):
                        u[z] = u[z] + x
            ai_e = np.logical_and.reduce([EVF[dj]['inside'][sel] for dj in range(ND)])       # a profile off the grid -> union unidentified
            ai_c = np.logical_and.reduce([CF[dj]['inside'][sel] for dj in range(ND)])
            for mi, name in enumerate(MNAMES):
                te, ve, ne_, tc, vc_, nc_ = ucnt[name]
                ce, cc = np.where(ai_e, classify(te, ve, ne_), 3), np.where(ai_c, classify(tc, vc_, nc_), 3)
                for q, fj in ((0, NCONT + 2 * mi), (1, NCONT + 2 * mi + 1)):
                    yo = np.where(ce == 3, np.nan, (ce == 0) if q == 0 else ((ce == 0) | (ce == 2))).astype(float)
                    yc = np.where(cc == 3, np.nan, (cc == 0) if q == 0 else ((cc == 0) | (cc == 2))).astype(float)
                    acc.add(vi, MA, ki, day_idx, ND, fj, yo, yc)
        ok_i = np.flatnonzero(okw)                                         # controls.csv rows (ok events), built column-wise
        if len(ok_i):
            vk = valid[ok_i]
            cmx = np.where(vk, C0[ok_i], -1).max(1)
            cmn = np.where(vk, C0[ok_i], np.iinfo(np.int64).max).min(1)
            cols = dict(day=[day] * len(ok_i), mid=[mid] * len(ok_i), scope=[RANKS[int(x)] for x in addr[ok_i]],
                        mk_hash=T['mk_hash'].to_numpy(object)[D['o'].to_numpy()[ok_i]], view=[VIEWS[int(x)] for x in D['vi'].to_numpy()[ok_i]],
                        cls=D['cls'].to_numpy(object)[ok_i], control=[kd] * len(ok_i), draw_status=st[ok_i], n_draws=vk.sum(1),
                        draw_span_ms=np.where(vk.any(1), cmx - cmn, -1))
            for dj, Dm in enumerate(DELAYS):
                for nm in HEADLINE:
                    xo = np.where(EVF[dj]['inside'], EVF[dj][nm], np.nan)[ok_i]
                    X = np.where(CF[dj]['inside'], CF[dj][nm], np.nan)[ok_i]
                    fin = np.isfinite(X)
                    nfin = fin.sum(1)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        cols[f'{nm}_ctl_mean_D{Dm}'] = np.where(nfin > 0, np.where(fin, X, 0.).sum(1) / np.maximum(nfin, 1), np.nan)
                        lt = (fin & (X < xo[:, None])).sum(1)
                        eq = (fin & (X == xo[:, None])).sum(1)
                    cols[f'{nm}_pct_D{Dm}'] = np.where((nfin > 0) & np.isfinite(xo), (lt + 0.5 * eq) / np.maximum(nfin, 1), np.nan)
            ctl_rows.append(pd.DataFrame(cols))
    # ---- pulls.csv rows: every cancel_possible order x view (an EX / PL tie), every window status; classes precomputed per profile ----
    CLD = [{name: np.where(EVF[dj]['inside'], classify(*EVF[dj][name]), 3) for name in MNAMES} for dj in range(ND)]
    all_in_v = np.logical_and.reduce([EVF[dj]['inside'] for dj in range(ND)])
    CLU = {}
    for name in MNAMES:
        tu, vu, nu = (sum(EVF[dj][name][z] for dj in range(ND)) for z in range(3))
        CLU[name] = np.where(all_in_v & (nu > 0), classify(tu, vu, nu), 3)
    for e in np.flatnonzero((D['cls'] == 'cancel_possible').to_numpy()):
        j, v = int(D['o'].iat[e]), VIEWS[int(D['vi'].iat[e])]
        row = dict(day=day, mid=mid, scope=RANKS[int(addr[e])], addr=TOP3[int(addr[e])], mk_hash=T['mk_hash'].iat[j], view=v,
                   y_side=T['y_side'].iat[j], y_ptick=int(T['k'].iat[j]), n_fills=int(T['n_fills'].iat[j]), cum_shares=float(T['cum'].iat[j]),
                   first_hi_ms=int(T['first_hi'].iat[j]), last_hi_ms=int(T['last_hi'].iat[j]), persist_lo_ms=int(T['p_lo'].iat[j]),
                   persist_hi_ms=int(T['p_hi'].iat[j]), removal_set=T[f'set_{v}'].iat[j], removal_lo_ms=int(D['rlo'].iat[e]),
                   removal_hi_ms=int(D['rhi'].iat[e]), removal_method=T[f'meth_{v}'].iat[j], removal_class=T[f'rcls_{v}'].iat[j],
                   x_size_ub=float(T[f'ub_{v}'].iat[j]), x_rhi_S_FIFO_LO=int(T['rhi_S_FIFO_LO'].iat[j]), lo_rhi_differs=bool(T['lo_rhi_differs'].iat[j]),
                   end_histories=D['hist'].iat[e], identification='tied', pull_identified=False, window_status=D['wst'].iat[e],
                   width_ms=int(D['rhi'].iat[e] - D['rlo'].iat[e]) if bnd[e] else -1,
                   phase=('pre', 'live')[int(D['ph'].iat[e])] if D['ph'].iat[e] >= 0 else 'n/a',
                   tau_lo_s=(we - int(D['rlo'].iat[e])) / 1000. if D['rlo'].iat[e] >= 0 else NAN,
                   ahead_hi_first=float(T['ahead_hi_first'].iat[j]), ahead_hi_last_fill=float(T['ahead_hi_last'].iat[j]),
                   successor=T['succ_kind'].iat[j], successor_hash=T['succ_hash'].iat[j], successor_gap_ms=int(T['succ_gap_ms'].iat[j]),
                   successor_ptick=int(T['succ_ptick'].iat[j]),
                   successor_before_rhi=bool(T['succ_kind'].iat[j] != 'none' and 0 <= int(T['succ_first_hi'].iat[j]) <= int(D['rhi'].iat[e])),
                   draws_presence=strat['presence'][e], draws_fill_anchored=strat['fill_anchored'][e], status=T['status'].iat[j])
        for dj, Dm in enumerate(DELAYS):
            f = EVF[dj]
            for nm in CONT:
                row[f'{nm}_D{Dm}'] = float(f[nm][e]) if f['inside'][e] else NAN
            for name in MNAMES:
                row[f'{name}_D{Dm}'] = CLS_MECH[int(CLD[dj][name][e])]
        sup, dsens = {c: [] for c in CLS_MECH}, []
        for name in MNAMES:
            c = int(CLU[name][e])
            row[f'{name}_union'] = CLS_MECH[c]
            sup[CLS_MECH[c]].append(name)
            if len({int(CLD[dj][name][e]) for dj in range(ND)}) > 1:
                dsens.append(name)
        row['d_sensitive'] = '+'.join(dsens) if dsens else 'none'
        if sup['compatible']:
            row['most_supported'], row['support_grade'] = '+'.join(sup['compatible']), 'compatible'
        elif sup['timing_sensitive']:
            row['most_supported'], row['support_grade'] = '+'.join(sup['timing_sensitive']), 'timing_sensitive'
        elif sup['unidentified']:
            row['most_supported'], row['support_grade'] = 'unidentified', 'unidentified'
        else:
            row['most_supported'], row['support_grade'] = 'none', 'contradicted'
        pull_rows.append(row)


# ------------------------------------------------------------------ context series (the chain_context adapter) ------------------------------------------------------------------
def context_series(cx, M_, t, cells, addrs):
    """Per-instant market context at the aligned instants t from chain_context (the one definition, contract 7.0; API of chain_context_v1):
    btc_ret -> r_1s / r_5s / r_30s (bps, NaN unless ok), book -> bid_tick / ask_tick (-inf / +inf an empty side, NaN unknown or stale) and
    mid_c, touch_distance -> ticks, level_size -> size, position(clock='settle') -> recorded and n, pending -> net, tau -> s."""
    t = np.asarray(t, np.int64)
    S = dict(t=t, cells=list(cells))
    b = cx.btc_ret(M_, t)
    for nm, _, _ in MOVE_BP:
        S[f'r_{nm}'] = np.asarray(b[f'r_{nm}'], float)
    bk = cx.book(M_, t)
    S['bid'], S['ask'], S['mid'] = (np.asarray(bk[k], float) for k in ('bid_tick', 'ask_tick', 'mid_c'))
    S['dist'] = (np.stack([np.asarray(cx.touch_distance(M_, t, sd, int(k))['ticks'], float) for sd, k in cells]) if cells
                 else np.zeros((0, len(t))))
    S['depth'] = (np.stack([np.asarray(cx.level_size(M_, t, sd, int(k))['size'], float) for sd, k in cells]) if cells
                  else np.zeros((0, len(t))))
    pos, nl, pend = [], [], []
    for a in addrs:
        p = cx.position(M_, a, t, clock='settle')
        pos.append(np.asarray(p['recorded'], float))
        nl.append(np.asarray(p['n'], float))
        pend.append(np.asarray(cx.pending(M_, a, t)['net'], float))
    S['pos'], S['nlegs'], S['pend'] = np.stack(pos), np.stack(nl), np.stack(pend)
    S['tau'] = np.asarray(cx.tau(M_, t), float)
    return S


def active_spans(M_, addrs):
    """the address's first-to-last status-ok leg (settle_ms, any role) in the market (contract 7.2 fallback) -> {addr index: (lo, hi)}."""
    out = {}
    for i, a in enumerate(addrs):
        G = M_.legs.get(a.lower())
        st = np.asarray(G['settle_ms'], np.int64) if G is not None and len(G) else np.zeros(0, np.int64)
        out[i] = (int(st.min()), int(st.max())) if len(st) else (-1, -1)
    return out


def anchors_table(Fl):
    """life fills of the top-3 orders -> {hash: x_lo_hard of every fill in settle order (hi, record rank, sid)}."""
    Fl = Fl.sort_values(['mk_hash', 'hi_ms', 'x_rec_rank', 'sid'], kind='stable')
    return {h: g['x_lo_hard_ms'].to_numpy(np.int64) for h, g in Fl.groupby('mk_hash', sort=False)}


# ------------------------------------------------------------------ inputs ------------------------------------------------------------------
def check_top3(path):
    """econ/identities.json ex_ante.top10_shares ranks 1-3 must be pf_mt_decision.TOP3 (contract section 1)."""
    need(Path(path).is_file(), f'identities: input missing: {Path(path).name}')
    ea = json.loads(Path(path).read_text()).get('ex_ante') or {}
    need(ea.get('day') == C.ID_DAY, f'identities: ex_ante is not from the ID day {C.ID_DAY}')
    lst = ea.get('top10_shares') or []
    need(len(lst) >= 3 and [x.get('rank') for x in lst[:3]] == [1, 2, 3], 'identities: ex_ante.top10_shares does not start with ranks 1-3')
    addrs = tuple(str(x.get('addr')) for x in lst[:3])
    need(addrs == TOP3, 'identities: ex_ante ranks 1-3 differ from pf_mt_decision.TOP3 (refused)')
    return addrs


def book_markets(days, book=None):
    """the 19 BOOK markets (pf_mt_econ.load_book) restricted to `days` -> [(day, mid)] in (day, mid) order. `book` (selftest only) replaces
    the manifest; the 19-market rule then does not apply."""
    bk = E.load_book() if book is None else {d: sorted(int(m) for m in v) for d, v in book.items()}
    if book is None:
        need(sum(len(v) for v in bk.values()) == 19, f'BOOK manifest: {sum(len(v) for v in bk.values())} markets, expected 19 (refused)')
    need(set(bk) <= set(C.MEASURE_DAYS), 'BOOK manifest: a day outside the measurement days (refused)')
    return [(d, int(m)) for d in sorted(bk) if d in days for m in sorted(bk[d])]


def read_life(life, day, mid, low3):
    """the top-3 rows of one market's lifecycle tables -> (orders with 'a', fills of those hashes, x_links of those signers, paths)."""
    P = {k: life / f'{day}_{mid}_{k}.parquet' for k in ('orders', 'fills', 'x_links')}
    for k, p in P.items():
        need(p.is_file(), f'missing lifecycle output: life/{p.name}')
    O = pq.read_table(P['orders'], columns=list(ORD_COLS)).to_pandas()
    sg = O['mk_signer'].astype(str)
    low = sg.str.lower()
    for a in low3:
        sp = set(sg[low == a])
        need(len(sp) <= 1, f'lifecycle: a top-3 address under two spellings {sp} (refused)')
    O['a'] = low.map({a: i for i, a in enumerate(low3)}).fillna(-1).astype(np.int64)
    O = O[O['a'] >= 0].copy()
    hs = set(O['mk_hash'])
    Fl = pq.read_table(P['fills'], columns=list(FIL_COLS)).to_pandas()
    Fl = Fl[Fl['mk_hash'].isin(hs)].copy()
    Lk = pq.read_table(P['x_links'], columns=list(LNK_COLS)).to_pandas()
    Lk = Lk[Lk['mk_signer'].astype(str).str.lower().isin(low3)].copy()
    return O, Fl, Lk, P


def read_markets(inp, day, mids):
    p = inp / f'markets_{day}.parquet'
    need(p.is_file(), f'markets: input missing: {p.name}')
    T = pq.read_table(p, columns=list(MKT_COLS)).to_pandas()
    T = T[T['mid'].isin(mids)]
    need(T['mid'].is_unique and len(T) == len(mids), f'markets_{day}: a BOOK market missing or duplicated')
    return {int(m): (int(round(float(a) * 1000)), int(round(float(b) * 1000)), str(s)) for m, a, b, s in zip(T['mid'], T['ws'], T['we'], T['status'])}, p


# ------------------------------------------------------------------ census counts ------------------------------------------------------------------
def census_rows(T, day, mid, rows):
    for v in VIEWS:
        for (a, c, w, ws_, h_), g in T.groupby(['a', f'class_{v}', f'why_{v}', f'wstat_{v}', f'hist_{v}']):
            rows.append(dict(view=v, day=day, mid=mid, scope=RANKS[int(a)], cls=c, reason=w or '-', window_status=ws_, end_histories=h_ or '-',
                             orders=int(len(g)), fills=int(g['n_fills'].sum()), shares=float(g['cum'].sum())))


def census_table(rows):
    D = pd.DataFrame(rows)
    if not len(D):
        return D
    keys = ['view', 'day', 'scope', 'cls', 'reason', 'window_status', 'end_histories']
    out = [D.groupby(keys, as_index=False)[['orders', 'fills', 'shares']].sum()]
    for sc, dd in (('pooled', None), (None, 'all'), ('pooled', 'all')):
        X = D.copy()
        if sc:
            X['scope'] = sc
        if dd:
            X['day'] = dd
        out.append(X.groupby(keys, as_index=False)[['orders', 'fills', 'shares']].sum())
    R = pd.concat(out, ignore_index=True)
    R['share_of_orders'] = R['orders'] / R.groupby(['view', 'day', 'scope'])['orders'].transform('sum')
    return R.sort_values(keys, kind='stable').reset_index(drop=True)


# ------------------------------------------------------------------ run ------------------------------------------------------------------
def _git(files):
    try:
        g = ['git', '-C', str(C.REPO)]
        head = subprocess.run(g + ['rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30).stdout.strip() or None
        rel = [str(Path(f).resolve().relative_to(C.REPO)) for f in files]
        tracked = subprocess.run(g + ['ls-files', '--error-unmatch', '--'] + rel, capture_output=True, text=True, timeout=30).returncode == 0
        st = subprocess.run(g + ['status', '--porcelain', '--'] + rel, capture_output=True, text=True, timeout=30)
        return dict(head=head, files=rel, tracked=tracked, unmodified=bool(tracked and st.returncode == 0 and not st.stdout.strip()))
    except (OSError, ValueError, subprocess.SubprocessError):
        return dict(head=None, files=None, tracked=None, unmodified=None)


def run(cx, inp=None, out=None, days=None, ranks=None, pilot=None, guard_mib=C.GUARD_MIB, log=print, hash_inputs=True, book=None):
    """The census: every (day, BOOK market) of `days`, the address indices `ranks` (0..2; all for the full run). cx = the chain_context
    module (or a selftest stub with the same API). Writes the write-once output directory and returns the receipt dict."""
    cpu0, wall0, asof0 = time.process_time(), time.time(), _now()
    days = tuple(days) if days is not None else tuple(C.MEASURE_DAYS)
    need(len(days) >= 1 and len(set(days)) == len(days) and all(d in C.MEASURE_DAYS for d in days),
         'days must be distinct measurement days (pf_mt_common.MEASURE_DAYS; refused before any path is built)')
    days = tuple(sorted(days))
    ranks = tuple(sorted(set(ranks))) if ranks is not None else (0, 1, 2)
    need(all(r in (0, 1, 2) for r in ranks), 'ranks must be in 0..2')
    miss = [f for f in REQUIRED_CX if not callable(getattr(cx, f, None))]
    need(not miss, f'chain_context lacks {miss} (refused: every market-context feature comes from it)')
    inp = Path(inp) if inp is not None else C.OUT
    out = Path(out) if out is not None else (C.RECEIPT / 'chains' / ('pull' if pilot is None else f'pull_pilot_{pilot[0]}_r{pilot[1]}'))
    need(not out.exists(), f'write-once: {out} exists (refusing to overwrite)')
    tmp = out.with_name(f'.{out.name}.partial{os.getpid()}')
    need(not tmp.exists(), f'a partial output exists: {tmp}')
    stage_cpu, cnt, hashes = {}, Counter(), {}
    t_st = [time.process_time()]

    def stage(name, rec_=None):
        now = time.process_time()
        dlt = now - t_st[0]
        stage_cpu[name] = round(stage_cpu.get(name, 0.) + dlt, 4)
        if rec_ is not None:
            rec_[name] = round(rec_.get(name, 0.) + dlt, 4)
        t_st[0] = now
        guard(name, guard_mib, cpu0)
    idp = inp / 'econ' / 'identities.json'
    check_top3(idp)
    hashes['econ/identities.json'] = dict(sha256=_sha(idp), bytes=idp.stat().st_size)
    mk = book_markets(days, book)
    need(mk, 'no BOOK market on the requested days')
    stage('preflight')
    life = inp / 'life'
    acc = Acc(len(days))
    pull_rows, ctl_rows, cen_rows = [], [], []
    rngs, mkt_checks = {}, {}
    hash_cache = {} if hash_inputs else None          # chain_context hashes every input it reads once per run (fills_valued ~1 GB)
    fvp = inp / 'econ' / 'fills_valued.parquet'
    if hash_cache is not None and fvp.is_file():       # the one-time hash as its own (fixed) stage; load_market reuses the cache entry
        hash_cache[str(fvp.resolve())] = E.sha256(fvp)
    stage('hash_fills_valued')
    by_day = {}
    for d, m in mk:
        by_day.setdefault(d, []).append(m)
    market_cpu = []
    for di, d in enumerate(days):
        if d not in by_day:
            continue
        MK, mp = read_markets(inp, d, by_day[d])
        hashes[mp.name] = dict(sha256=_sha(mp), bytes=mp.stat().st_size)
        for vi in range(len(VIEWS)):
            for a in range(3):
                rngs[(vi, a)] = np.random.Generator(np.random.PCG64([SEED, vi, int(d), a]))
        stage('markets')
        for m in by_day[d]:
            c_m0, mrec = time.process_time(), {}
            ws, we, mst = MK[m]
            O, Fl, Lk, P = read_life(life, d, m, LOW3)
            for k, p in P.items():
                hashes[f'life/{p.name}'] = dict(sha256=_sha(p), bytes=p.stat().st_size)
            O = O[O['a'].isin(ranks)]
            Fl = Fl[Fl['mk_hash'].isin(set(O['mk_hash']))]
            cnt['orders_top3'] += int(O['mk_hash'].nunique())
            stage('life', mrec)
            if not len(O):
                cnt['markets_without_orders'] += 1
                continue
            T = order_table(O, Fl, Lk, ws, we)
            census_rows(T, d, m, cen_rows)
            stage('census', mrec)
            Tv = T[T['valid'].to_numpy(bool)]
            if not len(Tv):
                cnt['markets_without_valid_orders'] += 1
                continue
            ctx = cx.load_market(d, m, TOP3, hash_cache=hash_cache)
            need((int(ctx.ws), int(ctx.we)) == (ws, we), f'chain_context market times differ from markets_{d} for {m} (refused)')
            for k_, v_ in (getattr(ctx, 'inputs', None) or {}).items():
                hashes[str(k_)] = v_
            mkt_checks[f'{d}_{m}'] = dict(checks=getattr(ctx, 'checks', None), cost=getattr(ctx, 'cost', None))
            stage('context_load', mrec)
            t = np.arange((ws - GRID_PRE_MS) // STEP * STEP, (we + GRID_POST_MS) // STEP * STEP + STEP, STEP, dtype=np.int64)
            cells = sorted({(int(sd), int(k)) for sd, k in zip(np.where(Tv['s'] == 1, 0, 1), Tv['k'])})
            S = context_series(cx, ctx, t, [(('bid', 'ask')[sd], k) for sd, k in cells], TOP3)
            S['cells'] = cells
            G = build_grid(S, ws, we)
            act = active_spans(ctx, TOP3)
            del S
            stage('context_series', mrec)
            run_market(cx, ctx, G, T, anchors_table(Fl), act, ws, we, di, acc, rngs, ctl_rows, pull_rows, m, d, cnt)
            stage('events_controls', mrec)
            market_cpu.append(dict(day=d, mid=m, cpu_s=round(time.process_time() - c_m0, 4), orders=int(len(T)), **mrec))
            log(f'{d} {m}: orders {len(T)} cpu {time.process_time() - c_m0:.2f} s vmhwm {vmhwm_mib():.0f} MiB')
            del G, ctx
    SM = summarize(acc, days, cx)
    PL = pd.DataFrame(pull_rows)
    CT = pd.concat(ctl_rows, ignore_index=True) if ctl_rows else pd.DataFrame()
    CE = census_table(cen_rows)
    stage('summary')
    tmp.mkdir(parents=True)
    try:
        for name, df in (('pulls.csv', PL), ('controls.csv', CT), ('summary.csv', SM), ('census.csv', CE)):
            df.to_csv(tmp / name, index=False)
        srcs = [Path(__file__).resolve(), HERE / 'chain_pull_selftest.py']
        cxf = getattr(cx, '__file__', None)
        if cxf:
            srcs.append(Path(cxf).resolve())
        sources = {str(p.relative_to(C.REPO)) if str(p).startswith(str(C.REPO)) else str(p): _sha(p) for p in srcs if p.is_file()}
        stage('hash')
        rec = dict(version=VERSION, module='chain_pull', asof_start=asof0, asof_end=_now(), argv=sys.argv, mt_run=C.RUN, slot=os.environ.get('PF_SLOT'),
                   contract=str(CONTRACT_FILE.relative_to(C.REPO)), contract_sha256=_sha(CONTRACT_FILE) if CONTRACT_FILE.is_file() else None,
                   pilot=None if pilot is None else dict(day=pilot[0], rank=pilot[1]), days=list(days), ranks=[RANKS[r] for r in ranks],
                   markets=[f'{d}_{m}' for d, m in mk], inputs=hashes, sources=sources, git=_git(srcs),
                   chain_context=dict(module=str(cxf) if cxf else None, version=getattr(cx, 'VERSION', None), markets=mkt_checks,
                                      inputs_hashed_by='chain_context load_market(hash_cache): sha256 once per input per run'),
                   declared=dict(DELAYS=DELAYS, window='[removal_lo - D, removal_hi] (chain_context.over_window D_ms)', STEP=STEP, PRE_MS=PRE_MS,
                                 LOOKBACK=LOOKBACK, N_DRAWS=N_DRAWS, SEED=SEED, SUCC_MS=SUCC_MS, TOL_SH=TOL_SH, POS_EPS=POS_EPS, MOVE_BP=MOVE_BP,
                                 MECHS=MECHS, START_FEATS=START_FEATS, WIN_FEATS=WIN_FEATS, CLASSES=CLASSES, WSTAT=WSTAT, GROUPS=GROUPS, KINDS=KINDS,
                                 VIEWS=VIEWS, CHECK_EVERY=CHECK_EVERY, CHECK_K=CHECK_K,
                                 presence='chain_context.matched_times: same-side certain-presence union mask, active-span fallback, key '
                                          'chain_pull|<view>|<side>|w<width>',
                                 fill_anchored='supplementary (beyond contract 7.2): numpy Generator(PCG64([SEED, view, day, rank]))',
                                 stats='chain_context.null_summary(obs, N_DRAWS group-statistic replicates)',
                                 identification='no row is a pull: every cancel_possible row is an EX / PL tie (identification tied)'),
                   counts=dict(sorted(cnt.items())), rows=dict(pulls=len(PL), controls=len(CT), summary=len(SM), census=len(CE)),
                   cpu_stage_s=stage_cpu, market_cpu=market_cpu, cpu_total_s=round(time.process_time() - cpu0, 3),
                   wall_s=round(time.time() - wall0, 3), vmhwm_mib=round(vmhwm_mib(), 1), guard_mib=guard_mib)
        if pilot is not None:
            rec['projection'] = projection(rec)
        rec['outputs'] = {p.name: dict(sha256=_sha(p), bytes=p.stat().st_size) for p in sorted(tmp.iterdir())}
        (tmp / 'receipt.json').write_text(json.dumps(_clean(rec), indent=1, sort_keys=True, allow_nan=False) + '\n')
        need(not out.exists(), f'write-once: {out} appeared during the run (refusing to overwrite)')
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return rec


def projection(rec):
    """pilot (one address x one day) -> the full run (3 addresses x 4 days, 19 markets, 4,149 top-3 orders). fixed = preflight + the
    fills_valued hash + sources; per day = the first market's context load beyond the median + markets_<day>; per market = life + census +
    median context load + context series; events / controls scaled by orders (pilot orders -> 4,149) and, as the upper variant, by
    address-days (x 12); summary x 5. One point: no size-volume fit."""
    s, mc = rec['cpu_stage_s'], rec['market_cpu']
    fixed = sum(s.get(k, 0.) for k in ('preflight', 'hash_fills_valued', 'hash'))
    cl = [m.get('context_load', 0.) for m in mc]
    med = float(np.median(cl[1:])) if len(cl) > 1 else (cl[0] if cl else 0.)
    day_extra = max((cl[0] if cl else 0.) - med, 0.) + s.get('markets', 0.) / max(len(rec['days']), 1)
    per_m = float(np.mean([m.get('life', 0.) + m.get('census', 0.) + m.get('context_series', 0.) for m in mc])) + med if mc else 0.
    n_ord = max(rec['counts'].get('orders_top3', 0), 1)
    ev = s.get('events_controls', 0.)
    base = fixed + day_extra * 4 + per_m * 19 + s.get('summary', 0.) * 5
    by_orders, by_ad = base + ev * 4149. / n_ord, base + ev * 12. / max(len(rec['ranks']) * len(rec['days']), 1)
    return dict(fixed_s=round(fixed, 3), day_extra_s=round(day_extra, 3), per_market_s=round(per_m, 3), events_controls_s=round(ev, 3),
                pilot_orders=n_ord, summary_s=round(s.get('summary', 0.), 3), full_run_s_by_orders=round(by_orders, 2),
                full_run_s_by_address_days=round(by_ad, 2), full_run_core_h=round(by_orders / 3600., 5), pilot_cpu_s=rec.get('cpu_total_s'),
                note='one point (no size-volume fit): events / controls scale with orders; context with markets')


def main(argv):
    use = 'usage: chain_pull.py run | chain_pull.py pilot <day> <rank 1-3> [<folder under chains/, starting pull_pilot_>]'
    need(len(argv) >= 2 and argv[1] in ('run', 'pilot'), use)
    if argv[1] == 'pilot':
        need(len(argv) in (4, 5) and argv[2] in C.MEASURE_DAYS and argv[3] in ('1', '2', '3'), use)
        need(len(argv) == 4 or (argv[4].startswith('pull_pilot_') and '/' not in argv[4] and '..' not in argv[4]), use)
    import chain_context as cx            # the one definition of market context (contract 7.0); required
    if argv[1] == 'run':
        rec = run(cx)
    else:
        out = (C.RECEIPT / 'chains' / argv[4]) if len(argv) == 5 else None
        rec = run(cx, out=out, days=(argv[2],), ranks=(int(argv[3]) - 1,), pilot=(argv[2], int(argv[3])))
    print(json.dumps(_clean(dict(rows=rec['rows'], counts=rec['counts'], cpu_stage_s=rec['cpu_stage_s'], cpu_total_s=rec['cpu_total_s'],
                                 vmhwm_mib=rec['vmhwm_mib'], projection=rec.get('projection'))), indent=1))


if __name__ == '__main__':
    main(sys.argv)
