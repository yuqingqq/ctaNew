"""POST + FILL logic-chain census (P-2026-005 MAKER-TRACE logic-chain packet; contract order_estimator_contract.md sections 4.3, 4.4, 5, 7.0,
7.1, 11; development, 2026-11-23). USER (binding): "We focus on getting the logic chains from order history, like when they post, what's the
market condition like, ... and when they get filled, is it some early posts resting there to get the front in queue, need to figure out the
logics to get to know how they perform." DESCRIPTIVE ONLY, four consumed days (Oct 24-27; point estimates, no intervals: G = 4 < 5): no
strategy, imitation, fit, replay, validation or profitability claim; no model skill is measured. OUTCOME-SELECTED: only filled orders are
visible, so every POST statement is conditional on a later fill, never an unconditional "they post when X".
SINGLE DEFINITIONS reused, never redefined: order_estimator.order_core (union posting window, resting bound, window class, clock flags, fill
       arrays, front_after_partial count), order_estimator.cell_arrays / cell_nomask / order_book (contract 4.4 ahead-at-acceptance range
       and front class, time to touch, the P_nomask prior-driven representative with its joint rule over every order at the cell; the other
       makers' members as light_core, checked equal to order_core on every top-3 order), order_estimator.link_hypotheses (4.6), legs_frame
       (the address's legs; take |position| classes are chain_take's, joined by sid); chain_context (contract 7.0; injected: run(cx, ...))
       for every market-context value: book, touch_distance, level_size / level_at_frames, btc_ret, position, tau, matched_times,
       null_summary, over_window (the equivalence check below). This tool's own rules follow (declared before any data; none fitted).
POPULATION ranks 1-3 of econ/identities.json ex_ante.top10_shares (= pf_mt_decision.TOP3, else refuse) x the 19 BOOK markets (pf_mt_econ.load_book:
       exactly 19, measurement days only, else refuse). One row per (day, market, address, order_hash): every top-3 hash of the lifecycle
       orders (all three scenario rows, else refuse), whatever its status. A lifecycle order exists only through a fill: this is every
       filled top-3 order. Statuses are carried, never dropped.
WINDOW (contract 4.3, order_core) union (post_lo_u, post_hi]; post_hi = the first fill's hi; width = post_hi - post_lo_u (window_width_bin by
       chain_context.width_bin); post_lo_u = -1 -> window_unbounded (evaluated from max(ws - 300 s, the grid start); class unidentified).
       REST resting age at the first fill in [max(0, first x_lo_hard - post_hi), post_hi - post_lo_u]: the lower bound is 0 by construction
       (a first fill never proves an early post); rest_bin = chain_context.age_bin of the upper bound (le_1s / le_10s / gt_10s / unbounded).
INSTANTS chain_context.over_window's convention: the window's lower end, every 100 ms grid instant strictly inside, and its upper end. Values
       at the grid instants come from chain_context once per market (book, btc_ret, position, tau; the touch distance from that book's
       touch by touch_distance's rule; the depth = the Market's lifecycle book cell (M.depth B.shown) at that book's frame, NaN when stale /
       unknown: level_at_frames / level_size's rule without their per-point conversion cost); the ends are point queries
       (touch_distance, level_size, book, btc_ret, position). CHECKED every market (refused on a mismatch): touch distance and depth at every
       CHECK_EVERY-th grid instant against touch_distance / level_size, and the first CHECK_K orders' windows against over_window (POST
       category set; touch-distance, depth, BTC r_5s and recorded-position min / max; BTC r_5s aligned category set). Every context value
       at an instant uses data <= that instant (falsifier in the selftest); the reconstruction label E below uses LATER prints by design.
POST   IDENTIFICATION (contract 4.3 / 7.1; the estimator's finding): a first fill never proves an early post - no public posting upper
       bound exists before the first fill - so early queue building is never asserted. post_class (frozen, first that applies) is the
       window-width identification class: unidentified (an INVALID order, window unbounded, or a first-fill clock flag) / late_post (width <=
       LATE_MS: identified late post) / short_rest (LATE_MS < width <= EARLY_MS: posted within 10 s before the first fill's upper bound,
       early queue building excluded, a late post not certain) / early_or_late_unidentified (width > EARLY_MS: an early queue-building post
       and a late post are both compatible).
       POSITION per candidate acceptance instant c, first that applies: unknown (own-side distance or opposite touch unknown: excluded from
       the set, counted) / crossing (the price at or through the opposite touch) / early_queue_builder (E: the first print at its (side,
       tick) by anyone AT OR AFTER c, each print at its hi = the S_FIFO clock, is >= EARLY_MS after c: resting behind the touch or at a level
       not traded for those 10 s; a reconstruction label using later prints) / touch_joiner (at or inside the own-side touch; an empty own
       side is inside) / behind_touch (behind by >= 1 tick). post_position = the constant category over the known instants, else
       timing_sensitive (post_cat_set kept), none known -> unknown. Compatible alternatives (contract 7.1), never asserted classes:
       early_queue_builder_possible, touch_joiner_possible / _only (at), touch_improver_possible / _only (inside; _only needs every instant
       known), fresh_certain (= late_post), early_possible (rest_hi > EARLY_MS or unbounded). post_hi = the first fill's hi (checked), so the
       own first print bounds the late end and early_queue_builder never holds at every instant of a lifecycle window.
CONTEXT over the instants (contract 7.1): touch distance range and category (at / inside / behind / crossing); spread range (ticks) and tight
       (<= 1 tick) category; depth at its price range and empty (<= TOL) category (the FIFO ahead volume); BTC a_h = s r_h (s = +1 bid / -1
       ask) with / against / flat (thr 0.5 / 1 / 2 bp: chain_context.MOVE_THR) and its ranges; tau range; phase (pre: post_hi < ws, live:
       lo >= ws, straddle); inventory relation of s x position, offsetting (< -1e-6) / adding (> 1e-6) / flat, for the recorded, certain and
       possible (certain + pending) positions; |pending| max. Category: constant over the known instants -> that, several ->
       timing_sensitive (the set kept), none known -> unknown. D profiles (chain_context.D_PROFILES; submission never identified): the
       D_FAMILIES categories over [lo - D, hi]; d_sensitive when one differs from D = 0.
QUEUE  order_book (contract 4.4; S_FIFO): front_at_acceptance / front_reached_after_clearing / front_at_acceptance_or_reached /
       behind_at_first_fill / unidentified / conflicting; ahead-at-acceptance range; later fills front_after_partial (order_core rule); the
       S_FIFO_LO lifecycle ahead bounds beside, lo_clock_queue_differs when they cross a class threshold S_FIFO's do not.
FILL   what followed: later fills of the same hash (n_tied_first: the first fill's settle ms); the next same-side order (x_links children,
       the earliest first fill; refill = same tick, reprice = another) with its link_hypotheses status; the address's opposite-side own
       maker shares (ok, not self-trade) in (T, T + FOLLOW_MS] after the first fill and after any of its fills (union, each leg once), T =
       the leg's settle_ms; econ mq / rq of the order's own ok rows; own taker records (unique sid, role taker) with settle_ms in (T_last,
       T_last + FOLLOW_MS] and (T_first, T_first + FOLLOW_MS]: counts, the first one's sid (by (settle_ms, sid)) and delay, and the window
       starts follow_first_lo_ms / follow_last_lo_ms. Their |position| classes are NOT recomputed here (integrator 2026-11-23, budget):
       chain_take's frozen take classes join at integration by (day, mid, address, sid) - the first take's sid, or every take of the
       address whose settle_ms lies in a window (the bounds above). The dropped columns (TAKE_DROPPED) do not return under another name.
       (order_estimator.take_core / legs_frame: legs_frame only, for the address's legs.)
PNL    per order ONE cohort for the four metrics: its own maker rows in fills_valued (role maker, its hash, market, address) with status
       ok, not a self-trade, st_res in pf_mt_econ.RES_VALID and st_1s / st_10s / st_60s ok. net_res = P&L to resolution, recorded fees
       deducted; v_1s / v_10s / v_60s = GROSS markouts from settle_ms (fees not deducted; fee_fill beside). Excluded rows counted by first
       failing reason. A later outcome is not the value of the action.
CONTROL ("posted when X", contract 7.1): each order with a bounded window of width w in phase pre / live against N_DRAWS control windows
       [c, c + w] from chain_context.matched_times (same address, market and phase pool: pre [ws - 300 s, ws), live [ws, we); within the
       address's active span (its first to last fills_valued leg in the market, the start rounded up to the 100 ms grid); seed SEED, key =
       the order hash; no fallback: no_eligible / no_pool counted). The control's hypothetical order has the order's side and tick.
       Statistics CTRL: 'all' = the category at every instant (every instant known, else not eligible), 'some' = at >= 1 known instant;
       obs = share of eligible orders with controls; replicate k = share over the k-th draws; chain_context.null_summary (mean, p05 / p50 /
       p95, p_upper / p_lower). A descriptive association on an outcome-selected population, never a cause.
GUARDS MT_RUN=v2; measurement days only (refused before any path is built); fills_valued sha256 = FV_SHA256 (the accepted matched_gap_r1
       binding) else refuse; an address outside the top 3 or two spellings refuse; VmHWM self-guard pf_mt_common.GUARD_MIB (1,450 MiB,
       never raised); CPU self-guard CPU_CAP_S; outputs write-once (a partial directory renamed at the end).
COST   (identical results, checked): lifecycle rows3_of only for the top-3 rows; the other orders' cell members as vectorised light cores
       (checked equal to light_core in the selftest and to order_core on every top-3 order at run time); cell_arrays / order_book on the
       cell's own fills (both filter to the cell; an order with legs outside its cell gets the full table); order_core on fills sorted once
       (presorted); own maker rows grouped once per address. --compare DIR (pilot): every retained column of orders.csv / summary.csv
       against an earlier output of the same address-day, as text (the receipt's compare section).
OUTPUTS RECEIPT/chains/post_fill (run) or chains/post_fill_pilot_<day>_r<rank>[suffix] (pilot): orders.csv (one row per top-3 filled order),
       summary.csv (per address x day, per address, top3 per day and pooled: class counts with orders / executions / shares and the PNL
       cohort; execution-level queue classes; control statistics; status counts), receipt.json (inputs and sources with sha256, git,
       declared constants, labels, counts, checks, CPU per stage, VmHWM; the pilot adds the measured projection).
Usage (run_light, PF_SLOT=claude2, MT_RUN=v2): chain_post_fill.py run | chain_post_fill.py pilot <day> <rank 1-3> [--suffix S] [--compare DIR]
       (DIR: a directory name under RECEIPT/chains)      selftest: chain_post_fill_selftest.py"""
import hashlib
import json
import os
import re
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
import pf_mt_lifecycle as LC  # noqa: E402
import pf_mt_econ as E  # noqa: E402
import pf_mt_decision as DEC  # noqa: E402   (constants only: TOP3)
import order_estimator as OE  # noqa: E402   the single definitions of contract 4.3 / 4.4 / 4.6 / 5

need, OK = C.need, C.STATUS_OK
NAN = float('nan')
VERSION = 'chain_post_fill_v1'
CONTRACT_FILE = C.REPO / 'predictfun/reviews/maker_trace_20261122/order_estimator_contract.md'
FV_SHA256 = '2a874707cc7801d6760e2de36075870700cc0be8e72cd9a6df0213424221cde6'   # the accepted matched_gap_r1 receipt's binding
N_BOOK = 19

# ------------------------------------------------------------------ declared constants (none fitted, none tuned) ------------------------------------------------------------------
TOP3 = DEC.TOP3
LOW3 = tuple(a.lower() for a in TOP3)
RANKS = ('rank1', 'rank2', 'rank3')
STEP = 100
LATE_MS, EARLY_MS = OE.LATE_MS, OE.EARLY_MS          # 1,000 / 10,000 ms (contract 4.3; the estimator's constants)
FOLLOW_MS = 30_000                                  # contract 7.1: what followed within 30 s
PRE_MS = 300_000
TOL = LC.TOL_SH                                     # 0.01 sh
POS_EPS = 1e-6
MOVE_BP = (('1s', 0.5), ('5s', 1.0), ('30s', 2.0))  # contract 7.0 (chain_context.MOVE_THR, checked equal)
N_DRAWS, SEED = 200, 20261123
CHECK_EVERY, CHECK_K = 37, 8                        # the per-market equivalence checks against chain_context (declared sample)
CPU_CAP_S = 120.
SCEN = tuple(C.SCENARIOS)
INVALID = LC.INVALID
POST_CATS = ('early_queue_builder', 'touch_joiner', 'behind_touch', 'crossing')            # per-instant position codes 0..3; -1 unknown
POST_CLASSES = ('late_post', 'short_rest', 'early_or_late_unidentified', 'unidentified')    # the window-width identification class
TOUCH_CATS = ('at', 'inside', 'behind', 'crossing')
BTC_CATS = ('with', 'against', 'flat')
INV_CATS = ('offsetting', 'adding', 'flat')
EMPTY_CATS = ('empty', 'nonempty')
TIGHT_CATS = ('tight', 'wide')
REDUCING = OE.REDUCING
FAMILIES = (('post', 'cell', POST_CATS), ('touch', 'cell', TOUCH_CATS), ('depth0', 'cell', EMPTY_CATS), ('spread1', 'mkt', TIGHT_CATS),
            ('btc_1s', 'side', BTC_CATS), ('btc_5s', 'side', BTC_CATS), ('btc_30s', 'side', BTC_CATS), ('inv', 'addr_side', INV_CATS),
            ('invc', 'addr_side', INV_CATS), ('invp', 'addr_side', INV_CATS))
FAM_KIND = {f: k for f, k, _ in FAMILIES}
FAM_CATS = {f: c for f, _, c in FAMILIES}
RANGES = (('dist', 'cell'), ('depth', 'cell'), ('spread', 'mkt'), ('r_1s', 'mkt'), ('r_5s', 'mkt'), ('r_30s', 'mkt'), ('pos', 'addr'),
          ('pend', 'addr'), ('tau', 'mkt'))
D_FAMILIES = ('touch', 'btc_1s', 'btc_5s', 'btc_30s', 'inv')
# the declared control statistics: (name, family, category or categories, 'all' | 'some')
CTRL = tuple([(f'btc_{h}_{c}_{q}', f'btc_{h}', (c,), q) for h, _ in MOVE_BP for c, q in (('against', 'all'), ('with', 'all'),
                                                                                        ('against', 'some'), ('with', 'some'))] +
             [('touch_at_or_inside_all', 'touch', ('at', 'inside'), 'all'), ('touch_behind_all', 'touch', ('behind',), 'all'),
              ('touch_crossing_some', 'touch', ('crossing',), 'some'), ('depth_empty_all', 'depth0', ('empty',), 'all'),
              ('depth_empty_some', 'depth0', ('empty',), 'some'), ('spread_tight_all', 'spread1', ('tight',), 'all'),
              ('inv_offsetting_all', 'inv', ('offsetting',), 'all'), ('inv_adding_all', 'inv', ('adding',), 'all'),
              ('inv_flat_all', 'inv', ('flat',), 'all')])
CLASS_FIELDS = ('post_class', 'post_position', 'window_class', 'window_width_bin', 'rest_bin', 'queue_class', 'nomask_class',
                'early_queue_builder_possible', 'touch_joiner_possible', 'touch_joiner_only', 'touch_improver_possible', 'touch_improver_only',
                'fresh_certain', 'early_possible',
                'touch_cat', 'btc_5s_cat', 'inv_cat', 'inv_certain_cat', 'phase', 'n_fills_bin', 'next_kind', 'next_link_status', 'd_sensitive')
# the take-class columns of the first pilot, dropped (their classes are chain_take's: join on (day, mid, address, sid)); the summary fields
# built on them; ignored by --compare
TAKE_DROPPED = ('n_takes_30s_after_last_reducing', 'take_after_last_class', 'take_after_last_opposing', 'n_takes_30s_after_first_reducing',
                'take_after_first_class')
TAKE_DROPPED_FIELDS = ('take_after_last_class', 'take_after_last_opposing')
FV_COLS = tuple(dict.fromkeys(OE.FV_COLS + ('st_res', 'st_1s', 'v_1s', 'st_10s', 'v_10s', 'st_60s', 'v_60s')))
SUMMARY_COLS = ('table', 'rank', 'day', 'field', 'category', 'n_orders', 'n_executions', 'shares', 'pnl_rows', 'pnl_shares', 'net_res_c_sh',
                'v_1s_c_sh', 'v_10s_c_sh', 'v_60s_c_sh', 'fee_fill_c_sh', 'pnl_excluded_rows', 'obs', 'n_obs', 'null_mean', 'null_p05', 'null_p50',
                'null_p95', 'p_upper', 'p_lower', 'n_draws', 'n_no_control', 'note')
LABELS = dict(
    population='outcome-selected: filled orders only; unfilled orders, non-actions and the opportunity population are not visible',
    clocks='book upd_ms and Binance E_ms are venue state, not what a maker received; settle_ms is the UUIDv7 creation time (an upper bound of '
           'the match); submission is unidentified (no delay bound for other makers; D profiles are sensitivities)',
    post='post_class is the window-width identification class: only late_post (window <= 1 s) is an identified posting time; a first fill '
         'never proves an early post (early / late stay compatible alternatives); the position categories are compatible alternatives over '
         'the feasible instants (early_queue_builder uses LATER prints: a reconstruction label); every context value uses data <= its instant',
    queue='front classes hold under the S_FIFO price-time scenario only (S_NOPRIO has no queue); a first fill or zero bound never proves '
          'front at acceptance',
    nomask='P_nomask is a prior-driven representative (no-masking / fewest hidden commands), never evidence',
    inventory='recorded / certain / possible VENUE positions (chain_context), never client-known inventory',
    net_res='P&L to resolution of the maker legs, recorded fees deducted (pf_mt_econ); the same cohort as the markouts',
    v_h='GROSS markout s q (m(settle_ms + h) - p), h = 1 s / 10 s / 60 s, from settle_ms; fees not deducted (fee_fill beside)',
    controls='descriptive association against width-matched windows of the same address / market / phase; not a cause, not an action rate',
    days='four consumed days (Oct 24-27): point estimates, no intervals (G = 4 < 5)')
_MUT_LEAK_MS = 0          # planted-error switch (selftest only): read every grid instant's context this many ms LATER (a look-ahead)
_MUT_NO_E = False         # planted-error switch (selftest only): drop the E (later-print) rule from the POST category


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def vmhwm_mib():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) / 1024.
    return 0.


def guard(where, limit=C.GUARD_MIB, cpu0=None):
    m = vmhwm_mib()
    need(m < limit, f'memory self-guard: VmHWM {m:.0f} MiB >= {limit:.0f} MiB at {where}')
    if cpu0 is not None:
        c = time.process_time() - cpu0
        need(c <= CPU_CAP_S, f'CPU self-guard: {c:.1f} s > {CPU_CAP_S} s at {where}')
    return m


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (set, frozenset)):
        return sorted(_clean(v) for v in o)
    return o


def _flags(s):
    return '+'.join(sorted(s)) if s else OK


def _split(st):
    return {x for x in str(st).split('+') if x and x != OK}


# ------------------------------------------------------------------ per-instant category codes (pure; one rule for grid and point values) ------------------------------------------------------------------
def touch_codes(d, opp, is_bid, tick, nxt, x):
    """POST and touch codes at instants: d = own-side touch distance (+ behind, 0 at, - inside, -inf empty own side, NaN unknown), opp =
    the opposite touch tick (NaN unknown, +-inf empty), nxt = the first print at the cell at or after x (int64 max: none)."""
    d, opp = np.asarray(d, float), np.asarray(opp, float)
    known = ~np.isnan(d) & ~np.isnan(opp)
    with np.errstate(invalid='ignore'):
        cross = known & np.where(is_bid, opp <= tick, opp >= tick)
    early = ((np.asarray(nxt, np.int64) - np.asarray(x, np.int64)) >= EARLY_MS) & (not _MUT_NO_E)
    post = np.where(~known, -1, np.where(cross, 3, np.where(early, 0, np.where(d <= 0, 1, 2)))).astype(np.int8)
    touch = np.where(~known, -1, np.where(cross, 3, np.where(d == 0, 0, np.where(d < 0, 1, 2)))).astype(np.int8)
    return post, touch


def depth_code(dep):
    dep = np.asarray(dep, float)
    return np.where(np.isnan(dep), -1, np.where(dep <= TOL, 0, 1)).astype(np.int8)


def spread_code(sp):
    sp = np.asarray(sp, float)
    return np.where(np.isnan(sp), -1, np.where(sp <= 1 + 1e-9, 0, 1)).astype(np.int8)


def btc_code(a, thr):
    a = np.asarray(a, float)
    return np.where(np.isnan(a), -1, np.where(a >= thr, 0, np.where(a <= -thr, 1, 2))).astype(np.int8)


def inv_code(sp):
    sp = np.asarray(sp, float)
    return np.where(np.isnan(sp), -1, np.where(sp < -POS_EPS, 0, np.where(sp > POS_EPS, 1, 2))).astype(np.int8)


def next_print(prints, cell, x):
    """the first print at the cell AT OR AFTER each instant (a print in the instant's own ms counts: an order accepted at c is matched at m
    >= c, so its own first print at hi bounds the instant hi; int64 max = none)."""
    ph = prints.get(cell)
    x = np.asarray(x, np.int64)
    if ph is None or not len(ph):
        return np.full(len(x), np.iinfo(np.int64).max, np.int64)
    j = np.searchsorted(ph, x, 'left')
    return np.where(j < len(ph), ph[np.minimum(j, len(ph) - 1)], np.iinfo(np.int64).max).astype(np.int64)


# ------------------------------------------------------------------ the chain_context adapter: grid series, point queries, window counts ------------------------------------------------------------------
class Context:
    """One market's per-instant context through chain_context: the aligned 100 ms grid t (values once per market) and point queries at
    window ends. Variants: cell index (side, tick), side 0 bid / 1 ask, address index into addrs."""

    def __init__(self, cx, M, t, cells, addrs, prints):
        self.cx, self.M, self.prints = cx, M, prints
        t = np.asarray(t, np.int64)
        need(len(t) >= 1 and t[0] % STEP == 0 and (len(t) == 1 or bool((np.diff(t) == STEP).all())), 'Context: t must be the aligned 100 ms grid')
        self.t, self.T, self.g0 = t, len(t), int(t[0] // STEP)
        self.cells, self.cix, self.addrs = list(cells), {c: i for i, c in enumerate(cells)}, tuple(addrs)
        tq = t + _MUT_LEAK_MS
        b = cx.book(M, tq)
        bid, ask = np.asarray(b['bid_tick'], float), np.asarray(b['ask_tick'], float)
        frame, stale = np.asarray(b['frame'], np.int64), np.asarray(b['stale'], bool)
        use = (frame >= 0) & ~stale
        nc, T = len(self.cells), self.T
        S = dict(dist=np.full((nc, T), np.nan), depth=np.full((nc, T), np.nan))
        K = dict(post=np.full((nc, T), -1, np.int8), touch=np.full((nc, T), -1, np.int8), depth0=np.full((nc, T), -1, np.int8))
        Bk = M.depth['B'] if M.depth is not None else None             # the Market's lifecycle _book (documented attribute)
        for i, (sd, k) in enumerate(self.cells):
            own, opp = (bid, ask) if sd == 'bid' else (ask, bid)
            with np.errstate(invalid='ignore'):
                d = (own - k) if sd == 'bid' else (k - own)               # touch_distance's rule on the same book (checked)
            if Bk is not None and Bk['F'] and 0 <= k < Bk['shown'].shape[2]:
                raw = Bk['shown'][np.clip(frame, 0, Bk['F'] - 1), 0 if sd == 'bid' else 1, k]     # level_at_frames / level_size's cell (checked)
                dep = np.where(use, raw, np.nan)
            else:
                dep = np.full(T, np.nan)
            S['dist'][i], S['depth'][i] = d, dep
            K['post'][i], K['touch'][i] = touch_codes(d, opp, sd == 'bid', k, next_print(prints, (sd, k), t), t)
            K['depth0'][i] = depth_code(dep)
        sp = np.asarray(b['spread'], float)
        S['spread'], K['spread1'] = sp[None, :], spread_code(sp)[None, :]
        r = cx.btc_ret(M, tq)
        for nm, thr in MOVE_BP:
            x = np.asarray(r[f'r_{nm}'], float)
            S[f'r_{nm}'] = x[None, :]
            K[f'btc_{nm}'] = np.stack([btc_code(sg * x, thr) for sg in (1., -1.)])
        P = [cx.position(M, a, tq) for a in self.addrs]
        for fam, key in (('inv', 'recorded'), ('invc', 'certain'), ('invp', 'possible')):
            K[fam] = np.stack([inv_code(sg * np.asarray(p[key], float)) for p in P for sg in (1., -1.)])     # addr_side = 2 addr + side
        S['pos'] = np.stack([np.asarray(p['recorded'], float) for p in P])
        S['pend'] = np.stack([np.abs(np.asarray(p['pending'], float)) for p in P])
        S['tau'] = np.asarray(cx.tau(M, tq), float)[None, :]
        self.S, self.K = S, K
        self.P, self.Q = {}, {}
        for f, _, cats in FAMILIES:
            X = K[f]
            z = np.zeros(X.shape[:-1] + (1,), np.int32)
            self.P[f] = [np.concatenate([z, np.cumsum(X == c, -1, dtype=np.int32)], -1) for c in range(len(cats))]
            self.Q[f] = np.concatenate([z, np.cumsum(X >= 0, -1, dtype=np.int32)], -1)

    @staticmethod
    def variant(kind, ci, si, ai):
        return dict(cell=ci, side=si, addr=ai, addr_side=2 * ai + si, mkt=np.zeros_like(ci))[kind]

    def point(self, x, ci, si, ai):
        """codes of every family and the range values at arbitrary instants x for per-point variants (chain_context point queries)."""
        cx, M = self.cx, self.M
        x, ci, si, ai = (np.asarray(v, np.int64).ravel() for v in (x, ci, si, ai))
        n = len(x)
        sides = np.array([self.cells[c][0] for c in ci], object) if n else np.zeros(0, object)
        ticks = np.array([self.cells[c][1] for c in ci], np.int64) if n else np.zeros(0, np.int64)
        K, R = {}, {}
        if n == 0:
            for f, _, _ in FAMILIES:
                K[f] = np.zeros(0, np.int8)
            for k, _ in RANGES:
                R[k] = np.zeros(0)
            return K, R
        b = cx.book(M, x)
        td = cx.touch_distance(M, x, sides, ticks)
        ls = cx.level_size(M, x, sides, ticks)
        is_bid = sides == 'bid'
        d = np.asarray(td['ticks'], float)
        opp = np.where(is_bid, np.asarray(b['ask_tick'], float), np.asarray(b['bid_tick'], float))
        nxt = np.zeros(n, np.int64)
        for c in np.unique(ci):
            m = ci == c
            nxt[m] = next_print(self.prints, self.cells[int(c)], x[m])
        K['post'], K['touch'] = touch_codes(d, opp, is_bid, ticks, nxt, x)
        dep = np.asarray(ls['size'], float)
        K['depth0'] = depth_code(dep)
        sp = np.asarray(b['spread'], float)
        K['spread1'] = spread_code(sp)
        r = cx.btc_ret(M, x)
        sg = np.where(si == 0, 1., -1.)
        for nm, thr in MOVE_BP:
            R[f'r_{nm}'] = np.asarray(r[f'r_{nm}'], float)
            K[f'btc_{nm}'] = btc_code(sg * R[f'r_{nm}'], thr)
        pos = {k: np.full(n, np.nan) for k in ('recorded', 'certain', 'possible', 'pending')}
        for a in np.unique(ai):
            m = ai == a
            p = cx.position(M, self.addrs[int(a)], x[m])
            for k in pos:
                pos[k][m] = np.asarray(p[k], float)
        for fam, key in (('inv', 'recorded'), ('invc', 'certain'), ('invp', 'possible')):
            K[fam] = inv_code(sg * pos[key])
        R.update(dist=d, depth=dep, spread=sp, pos=pos['recorded'], pend=np.abs(pos['pending']), tau=np.asarray(cx.tau(M, x), float))
        return K, R

    def interior(self, lo, hi):
        """grid rows [a, b] strictly inside (lo, hi) (b < a: none), over_window's g1 / g2; inside = the rows lie in the grid."""
        a = (np.asarray(lo, np.int64) // STEP + 1) - self.g0
        b = ((np.asarray(hi, np.int64) - 1) // STEP) - self.g0
        none = b < a
        inside = none | ((a >= 0) & (b < self.T))
        return a, b, none, inside

    def windows(self, lo, hi, ci, si, ai, ends=None):
        """per family the category counts and the known count over the instants {lo, grid strictly inside, hi} of each window; ends =
        (codes at lo, codes at hi) when already known, else point queries. -> dict with n, inside."""
        lo, hi, ci, si, ai = (np.asarray(v, np.int64).ravel() for v in (lo, hi, ci, si, ai))
        need(bool((hi >= lo).all()), 'windows: hi < lo')
        if ends is None:
            ends = (self.point(lo, ci, si, ai)[0], self.point(hi, ci, si, ai)[0])
        K0, K1 = ends
        a, b, none, inside = self.interior(lo, hi)
        need(bool(inside.all()), f'windows: {int((~inside).sum())} window interiors outside the market grid (the grid must cover every window)')
        single = hi == lo
        ac, bc = np.clip(a, 0, self.T), np.clip(np.where(none, a - 1, b), -1, self.T - 1)
        nin = np.where(none, 0, bc - ac + 1)
        out = dict(n=np.where(single, 1, nin + 2), inside=inside)
        for f, kind, cats in FAMILIES:
            v = self.variant(kind, ci, si, ai)
            Q = self.Q[f]
            kn = np.where(none, 0, Q[v, bc + 1] - Q[v, ac])
            e0, e1 = K0[f], K1[f]
            out[f'{f}_known'] = kn + (e0 >= 0) + np.where(single, 0, e1 >= 0)
            out[f] = [np.where(none, 0, P[v, bc + 1] - P[v, ac]) + (e0 == c) + np.where(single, 0, e1 == c) for c, P in enumerate(self.P[f])]
        return out

    def ranges(self, lo, hi, ci, si, ai, R0, R1):
        """min / max of every range series over each window's instants (NaN ignored; NaN when none is finite)."""
        lo, hi, ci, si, ai = (np.asarray(v, np.int64).ravel() for v in (lo, hi, ci, si, ai))
        a, b, none, _ = self.interior(lo, hi)
        out = {f'{k}_{q}': np.full(len(lo), np.nan) for k, _ in RANGES for q in ('min', 'max')}
        for i in range(len(lo)):
            for k, kind in RANGES:
                X = self.S[k]
                v = int(self.variant(kind, ci[i:i + 1], si[i:i + 1], ai[i:i + 1])[0]) if X.shape[0] > 1 else 0
                seg = [X[v, max(a[i], 0):b[i] + 1]] if not none[i] else []
                vals = np.concatenate(seg + [np.atleast_1d(R0[k][i]), np.atleast_1d(R1[k][i]) if hi[i] != lo[i] else np.zeros(0)])
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    out[f'{k}_min'][i], out[f'{k}_max'][i] = float(vals.min()), float(vals.max())
        return out

    def grid_codes(self, rows, ci, si, ai):
        """the codes at grid rows (an instant on the aligned grid: its point value equals the grid value)."""
        rows, ci, si, ai = (np.asarray(v, np.int64).ravel() for v in (rows, ci, si, ai))
        return {f: self.K[f][self.variant(k, ci, si, ai), rows] for f, k, _ in FAMILIES}


def window_category(W, fam, i):
    """the contract-7.1 category of window i: constant over the known instants -> it, several -> timing_sensitive, none -> unknown; set."""
    cats = FAM_CATS[fam]
    present = [cats[c] for c in range(len(cats)) if int(W[fam][c][i]) > 0]
    if not present:
        return 'unknown', ''
    return (present[0] if len(present) == 1 else 'timing_sensitive'), '|'.join(sorted(present))


def indicator(W, fam, cats, q):
    """CTRL indicator arrays (value, eligible): 'all' = every instant known and in cats; 'some' = >= 1 known instant in cats."""
    names = FAM_CATS[fam]
    c = sum(W[fam][names.index(x)] for x in cats)
    kn, n = W[f'{fam}_known'], W['n']
    if q == 'all':
        el = (n > 0) & (kn == n)
        return el & (c == n), el
    el = kn > 0
    return el & (c > 0), el


def check_context(cx, ctx, rows_lo, rows_hi, ci, si, ai, W, R, cnt):
    """declared per-market equivalence checks against chain_context (refused on a mismatch): (1) touch distance and depth at every
    CHECK_EVERY-th grid instant of every cell vs touch_distance / level_size; (2) the first CHECK_K orders' windows vs over_window: the POST
    category set, touch-distance / depth / BTC r_5s / recorded-position min and max, the BTC r_5s aligned category set."""
    M, t = ctx.M, ctx.t
    js = np.arange(0, ctx.T, CHECK_EVERY)
    for i, (sd, k) in enumerate(ctx.cells):
        td = cx.touch_distance(M, t[js], sd, k)
        ls = cx.level_size(M, t[js], sd, k)
        for nm, a, b in (('touch distance', np.asarray(td['ticks'], float), ctx.S['dist'][i, js]),
                         ('depth', np.asarray(ls['size'], float), ctx.S['depth'][i, js])):
            need(bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])),
                 f'context check: {nm} at ({sd}, {k}) differs from chain_context')
        cnt['check_grid_points'] += 2 * len(js)
    for i in range(min(CHECK_K, len(rows_lo))):
        lo, hi, c_, s_, a_ = int(rows_lo[i]), int(rows_hi[i]), int(ci[i]), int(si[i]), int(ai[i])
        sd, k = ctx.cells[c_]
        sg = 1. if s_ == 0 else -1.

        def fpost(M_, tt, c_=c_):
            K_, _ = ctx.point(tt, np.full(len(tt), c_), np.full(len(tt), s_), np.full(len(tt), a_))
            v = np.array([POST_CATS[x] if x >= 0 else None for x in K_['post']], object)
            return v, K_['post'] >= 0
        ow = cx.over_window(M, fpost, lo, hi, mode='category')
        mine = window_category(W, 'post', i)[1]
        need(str(ow['cats'][0]) == mine, f'context check: POST category set {mine!r} differs from over_window {ow["cats"][0]!r}')
        for key, fn in (('dist', lambda M_, tt: (lambda d: (d['ticks'], ~np.isnan(d['ticks'])))(cx.touch_distance(M_, tt, sd, k))),
                        ('depth', cx.feat_level(sd, k)), ('r_5s', cx.feat_btc('r_5s')), ('pos', cx.feat_position(ctx.addrs[a_], 'recorded'))):
            ow = cx.over_window(M, fn, lo, hi, mode='sign')
            for q in ('min', 'max'):
                x, y = float(ow[q][0]), float(R[f'{key}_{q}'][i])
                need((np.isnan(x) and np.isnan(y)) or x == y, f'context check: {key} {q} {y!r} differs from over_window {x!r}')
        ow = cx.over_window(M, cx.feat_btc_align('r_5s', sg), lo, hi, mode='category')
        mine = window_category(W, 'btc_5s', i)[1]
        need(str(ow['cats'][0]) == mine, f'context check: BTC r_5s set {mine!r} differs from over_window {ow["cats"][0]!r}')
        cnt['check_windows'] += 1


# ------------------------------------------------------------------ per-order chains (pure) ------------------------------------------------------------------
def light_core(r3):
    """the fields order_estimator.cell_nomask reads from ANOTHER order at the cell (hash, cum, post_lo_u, post_hi, first_hi, conflict), from
    its three lifecycle rows by order_core's rules: conflict = the X_FLAGS of its statuses (the flags order_core adds from fills or clocks -
    fills_mismatch, hi_from_receipt, clock_unchecked, receipt_before_uuid - are not X_FLAGS; its no_feasible_history cannot arise: 'exhausted'
    is in every lifecycle removal set, checked). Checked equal to order_core on every top-3 order of the market (refused otherwise)."""
    b = r3['S_NOPRIO']
    fl = set()
    for s in SCEN:
        fl |= _split(r3[s]['status'])
        need('exhausted' in str(r3[s]['removal_set']).split('|'), f'lifecycle order {b["mk_hash"]}: a removal set without exhausted')
    plo = [int(r3[s]['post_lo_ms']) for s in SCEN]
    return dict(hash=b['mk_hash'], cum=float(b['cum_shares']), post_lo_u=-1 if min(plo) < 0 else min(plo), post_hi=int(b['post_hi_ms']),
                first_hi=int(b['first_hi_ms']), conflict=bool(fl & OE.X_FLAGS))


LIGHT_KEYS = ('hash', 'cum', 'post_lo_u', 'post_hi', 'first_hi', 'conflict')


def light_cores(O, hashes):
    """light_core for the hashes `hashes` of the market's lifecycle orders O, vectorised (the same rules, in O's first-appearance order;
    checked equal to light_core in the selftest and to order_core on every top-3 order at run time) -> {hash: dict}."""
    h = O.mk_hash.to_numpy(object)
    keep = np.isin(h, np.array(sorted(hashes), object)) if len(hashes) else np.zeros(len(O), bool)
    st, rs = O.status.astype(str).to_numpy(object)[keep], O.removal_set.astype(str).to_numpy(object)[keep]
    hk = h[keep]
    urs = {r: 'exhausted' in r.split('|') for r in set(rs.tolist())}
    bad = np.array([not urs[r] for r in rs], bool)
    if bad.any():
        need(False, f'lifecycle order {hk[np.flatnonzero(bad)[0]]}: a removal set without exhausted')
    ust = {s: bool(_split(s) & OE.X_FLAGS) for s in set(st.tolist())}
    df = pd.DataFrame(dict(h=hk, plo=O.post_lo_ms.to_numpy(np.int64)[keep], conf=np.array([ust[s] for s in st], bool)))
    agg = df.groupby('h', sort=False).agg(plo=('plo', 'min'), conf=('conf', 'max'))
    b = O[keep & (O.scenario.to_numpy(object) == 'S_NOPRIO')].set_index('mk_hash').reindex(agg.index)
    plo = agg.plo.to_numpy(np.int64)
    return {hh: dict(hash=hh, cum=float(c), post_lo_u=-1 if p < 0 else int(p), post_hi=int(ph), first_hi=int(fh), conflict=bool(cf))
            for hh, c, p, ph, fh, cf in zip(agg.index, b.cum_shares.to_numpy(float), plo, b.post_hi_ms.to_numpy(np.int64),
                                             b.first_hi_ms.to_numpy(np.int64), agg.conf.to_numpy(bool))}


def post_class(core, width):
    """the frozen window-width identification class (post_class, reason); width -1 = unbounded."""
    if core['flags'] & INVALID:
        return 'unidentified', 'invalid_order'
    if core['post_lo_u'] < 0 or width < 0:
        return 'unidentified', 'window_unbounded'
    if core['clock_bad']:
        return 'unidentified', 'first_fill_clock'
    if width <= LATE_MS:
        return 'late_post', 'width_le_late_ms'
    if width <= EARLY_MS:
        return 'short_rest', 'width_le_early_ms'
    return 'early_or_late_unidentified', 'width_gt_early_ms'


def post_position(W, i):
    """(post_position, category set) over the known candidate instants: constant -> it, several -> timing_sensitive, none -> unknown."""
    present = [POST_CATS[c] for c in range(len(POST_CATS)) if int(W['post'][c][i]) > 0]
    if not present:
        return 'unknown', ''
    return (present[0] if len(present) == 1 else 'timing_sensitive'), '|'.join(sorted(present))


def lo_clock_differs(r3):
    """S_FIFO_LO's lifecycle ahead bounds cross a class threshold (ahead_lo > TOL; ahead_hi <= TOL) that S_FIFO's do not (or the reverse)."""
    def k(s):
        a, b = OE._fin(r3[s]['ahead_lo']), OE._fin(r3[s]['ahead_hi'])
        return (bool(np.isfinite(a) and a > TOL), bool(np.isfinite(b) and b <= TOL))
    return k('S_FIFO') != k('S_FIFO_LO')


def pnl_rows(own_mk):
    """the PNL cohort of one order's own maker rows -> sums and the excluded count by first failing reason."""
    st = own_mk.status.to_numpy(object) == OK
    self_ = own_mk.self_trade.to_numpy(bool)
    res = np.isin(own_mk.st_res.to_numpy(object), list(E.RES_VALID))
    mk = {h: own_mk[f'st_{h}'].to_numpy(object) == OK for h in ('1s', '10s', '60s')}
    reason = np.where(~st, 'status', np.where(self_, 'self_trade', np.where(~res, 'st_res', np.where(~mk['1s'], 'st_1s',
                      np.where(~mk['10s'], 'st_10s', np.where(~mk['60s'], 'st_60s', 'ok'))))))
    c = reason == 'ok'
    q = own_mk.q.to_numpy(float)
    out = dict(pnl_rows=int(c.sum()), pnl_shares=float(q[c].sum()), net_res=float(own_mk.net_res.to_numpy(float)[c].sum()),
               fee_fill=float(own_mk.fee_fill.to_numpy(float)[c].sum()), pnl_excluded_rows=int((~c).sum()),
               pnl_excluded=dict(Counter(reason[~c].tolist())))
    for h in ('1s', '10s', '60s'):
        out[f'v_{h}'] = float(own_mk[f'v_{h}'].to_numpy(float)[c].sum())
    return out


def follow_opposite(T_fills, opp_t, opp_q):
    """opposite-side own maker shares in (T, T + FOLLOW_MS] after the first fill and after any fill (union; each leg once)."""
    if len(T_fills) == 0 or len(opp_t) == 0:
        return 0., 0., 0
    T = np.sort(np.asarray(T_fills, np.int64))
    first = (opp_t > T[0]) & (opp_t <= T[0] + FOLLOW_MS)
    j = np.searchsorted(T, opp_t, 'left') - 1                     # the latest fill strictly before the leg
    anyw = (j >= 0) & (opp_t - T[np.maximum(j, 0)] <= FOLLOW_MS)
    return float(opp_q[first].sum()), float(opp_q[anyw].sum()), int(anyw.sum())


def takes_in(T0, tk_t, tk_sid):
    """own taker records (unique sid) with settle_ms in (T0, T0 + FOLLOW_MS] -> (n, first delay ms or -1, first sid or ''), the first by
    (settle_ms, sid). Their |position| classes are chain_take's (join on (day, mid, address, sid))."""
    m = (tk_t > T0) & (tk_t <= T0 + FOLLOW_MS)
    if not m.any():
        return 0, -1, ''
    ix = np.flatnonzero(m)
    j0 = ix[np.lexsort((tk_sid[ix].astype(str), tk_t[ix]))[0]]
    return len(ix), int(tk_t[j0] - T0), str(tk_sid[j0])


def n_fills_bin(n):
    return '1' if n <= 1 else ('2' if n == 2 else ('3_5' if n <= 5 else '6+'))


def cell_ok(c):
    """a placeable YES cell (chain_context refuses any other side): context is evaluated only there."""
    return c[0] in ('bid', 'ask') and 0 <= int(c[1]) < LC.NT


# the context columns of an order whose cell is not placeable (status context_not_evaluated): blanked, never guessed
CONTEXT_BLANK = dict(post_position='unknown', post_cat_set='', early_queue_builder_possible=False, touch_joiner_possible=False,
                     touch_joiner_only=False, touch_improver_possible=False, touch_improver_only=False, touch_cat='unknown', touch_set='',
                     dist_min=NAN, dist_max=NAN, spread_min=NAN, spread_max=NAN, spread_cat='unknown', depth_min=NAN, depth_max=NAN,
                     depth_cat='unknown', **{f'btc_{nm}_cat': 'unknown' for nm, _ in MOVE_BP}, **{f'btc_{nm}_set': '' for nm, _ in MOVE_BP},
                     **{f'a_{nm}_{q}': NAN for nm, _ in MOVE_BP for q in ('min', 'max')}, inv_cat='unknown', inv_set='', inv_certain_cat='unknown',
                     inv_possible_cat='unknown', pos_min=NAN, pos_max=NAN, pend_abs_max=NAN, d_sensitive=False, d_sensitive_fields='')


def market_orders(O, Fl, Lk, fv_m, B, ws, addrs):
    """the estimator's single definitions for every order of the addresses `addrs` (lowercase) in one market -> (cores, books, links, cells,
    prints). O / Fl are the market's FULL lifecycle tables: every other order at a cell (any maker, the other top-3 addresses included) is a
    member of the joint P_nomask rule. Cost paths with identical results (COST in the module note)."""
    need(set(O.scenario.astype(str).unique()) <= set(SCEN), 'lifecycle orders: an undeclared scenario (refused)')
    need(not O.duplicated(['mk_hash', 'scenario']).any(), 'lifecycle orders: a hash twice under one scenario (refused)')
    need(bool((O.groupby('mk_hash', sort=False).scenario.nunique() == len(SCEN)).all()), 'lifecycle orders: a hash without its three scenario rows (refused)')
    sig = O.mk_signer.astype(str).str.lower()
    R3 = OE.rows3_of(O[sig.isin(addrs).to_numpy(bool)])             # the estimator's rows: the top-3 orders only
    top = set(R3)
    settle = {(s, h): int(t) for s, h, t, r in zip(fv_m.sid, fv_m.hash, fv_m.settle_ms, fv_m.role) if r == 'maker'}
    Fs = Fl.sort_values(['hi_ms', 'x_rec_rank', 'sid'], kind='stable')          # order_core's own (stable) order: presorted
    fb = {h: g for h, g in Fs[Fs.mk_hash.isin(top)].groupby('mk_hash', sort=False)}
    cc = OE.clip_counts(O[sig.isin(addrs).to_numpy(bool)])         # per signer: the top-3 signers' own rows suffice
    cores = {}
    for h in sorted(top):
        r3 = R3[h]
        g = str(r3['S_NOPRIO']['mk_signer']).lower()
        fl = fb.get(h, Fl.iloc[0:0])
        clip = OE.clip_for(cc.get(g, Counter()), OE._fin(r3['S_FIFO']['size_exact']))
        c = OE.order_core(r3, fl, settle={sid: settle.get((sid, h)) for sid in fl.sid}, clip=clip, presorted=True)
        c['clip'] = clip
        c['lo_clock_differs'] = lo_clock_differs(r3)
        cores[h] = c
    bN = O[O.scenario.to_numpy(object) == 'S_NOPRIO']
    cell_of = dict(zip(bN.mk_hash.to_numpy(object), zip(bN.y_side.astype(str).to_numpy(object).tolist(), bN.y_ptick.to_numpy(np.int64).tolist())))
    cells = sorted({cell_of[h] for h in top if cell_ok(cell_of[h])}) or [('bid', 1)]      # a placeholder cell when none is placeable
    cset = set(cells)
    mem_h = [h for h in pd.unique(O.mk_hash.to_numpy(object)) if cell_of[h] in cset]     # first-appearance order (as rows3_of's)
    LCV = light_cores(O, set(mem_h) | top)
    members = {c: [] for c in cells}
    for h in mem_h:
        members[cell_of[h]].append(cores[h] if h in cores else LCV[h])
    for h in top:
        lc_, fc = LCV[h], cores[h]
        need(all((abs(lc_[k] - fc[k]) < 1e-9) if k == 'cum' else (lc_[k] == fc[k]) for k in LIGHT_KEYS),
             f'light core of {h} differs from order_core (refused)')
        need(fc['post_hi'] == fc['first_hi'], f'order {h}: post_hi != first fill hi (lifecycle invariant; refused)')
    touch = OE.touch_arrays(B)
    cf = {(str(sd), int(k)): g for (sd, k), g in Fl.groupby(['y_side', 'y_ptick'], sort=False)}     # each cell's fills, in Fl order
    CA, NM, FFc = {}, {}, {}
    for c in cells:                                                  # P_nomask's joint rule once per cell (order_estimator.cell_nomask)
        si = LC.SIDE.get(c[0], -1)
        if si < 0 or B['F'] == 0:
            CA[c] = NM[c] = None
            continue
        Fc = cf.get(c, Fl.iloc[0:0])
        FFc[c] = OE.fill_frames(Fc, B)
        CA[c] = OE.cell_arrays(B, Fc, FFc[c], si, c[1])
        NM[c] = OE.cell_nomask(members[c], B, CA[c], ws)
    FF_all, books = None, {}
    for h in sorted(top):
        c, fo = cell_of[h], fb.get(h)
        at_cell = c in FFc and (fo is None or bool(((fo.y_side.astype(str).to_numpy(object) == c[0]) & (fo.y_ptick.to_numpy(np.int64) == c[1])).all()))
        if at_cell:                                                  # its own fills all at its cell: the cell's fills give order_book's result
            books[h] = OE.order_book(cores[h], B, touch, cf.get(c, Fl.iloc[0:0]), FFc[c], CA[c], NM[c], ws)
        else:
            if FF_all is None:
                FF_all = OE.fill_frames(Fl, B)
            books[h] = OE.order_book(cores[h], B, touch, Fl, FF_all, CA.get(c), NM.get(c), ws)
    sids = {h: set(g.sid) for h, g in fb.items()}
    links = {}
    for h in sorted(top):
        kids = Lk[(Lk.prev_hash == h) & Lk.next_hash.isin(top)]
        if not len(kids):
            links[h] = dict(next_kind='none', next_link_status='none', n_children=0)
            continue
        k = kids.sort_values(['next_first_hi_ms', 'next_hash'], kind='stable').iloc[0]
        P, Q = cores[h], cores[k.next_hash]
        res = OE.link_hypotheses(P, Q, shared_record=bool(sids.get(h, set()) & sids.get(k.next_hash, set())))
        links[h] = dict(next_kind=res['kind'], next_link_status=res['status'], n_children=int(len(kids)), next_hash=str(k.next_hash),
                        next_gap_lo_ms=res['gap_lo_ms'], next_gap_hi_ms=res['gap_hi_ms'], next_d_ticks=res['d_ticks'],
                        next_relations=res['relations'], next_first_hi_ms=int(Q['first_hi']))
    pr = Fl[Fl.y_side.isin(['bid', 'ask'])]
    bad = np.array([bool(_split(x) & {'unplaceable', 'subcent_price'}) for x in pr.status], bool) if len(pr) else np.zeros(0, bool)
    pr = pr[~bad]
    prints = {(sd, int(k)): np.sort(g.hi_ms.to_numpy(np.int64)) for (sd, k), g in pr.groupby(['y_side', 'y_ptick'], sort=False)}
    return cores, books, links, cells, prints


def order_rows(cx, ctx, day, mid, ws, we, cores, books, links, Lg, cnt, check=True):
    """one row per top-3 order: window, rest, POST class and flags, context, queue, fill chain, PNL; and the windows for the controls."""
    hs = sorted(cores, key=lambda h: (cores[h]['first_hi'], h))
    n = len(hs)
    if n == 0:
        return [], None
    rank = np.array([LOW3.index(cores[h]['signer'].lower()) for h in hs], np.int64)
    okc = np.array([cell_ok((cores[h]['y_side'], cores[h]['tick'])) for h in hs], bool)
    ci = np.array([ctx.cix.get((cores[h]['y_side'], cores[h]['tick']), 0) for h in hs], np.int64)    # placeholder 0: blanked below
    si = np.array([0 if cores[h]['y_side'] == 'bid' else 1 for h in hs], np.int64)
    hi = np.array([cores[h]['post_hi'] for h in hs], np.int64)
    plo = np.array([cores[h]['post_lo_u'] for h in hs], np.int64)
    lo = np.minimum(np.where(plo >= 0, plo, np.maximum(ws - PRE_MS, int(ctx.t[0]))), hi)
    K0, R0 = ctx.point(lo, ci, si, rank)
    K1, R1 = ctx.point(hi, ci, si, rank)
    W = ctx.windows(lo, hi, ci, si, rank, ends=(K0, K1))
    R = ctx.ranges(lo, hi, ci, si, rank, R0, R1)
    WD = {D: ctx.windows(lo - D, hi, ci, si, rank) for D in cx.D_PROFILES if D}
    if check:
        check_context(cx, ctx, lo, hi, ci, si, rank, W, R, cnt)
    wbin = cx.width_bin(np.where(plo >= 0, hi - plo, -1).astype(float))
    take_t, take_sid, opp, own_by = {}, {}, {}, {}
    for a in set(rank.tolist()):
        L = Lg[a]
        tk = L[(L.role == 'taker')].drop_duplicates('sid')
        take_t[a], take_sid[a] = tk.settle_ms.to_numpy(np.int64), tk.sid.to_numpy(object)
        mk = L[(L.role == 'maker') & L.lot]
        opp[a] = {sg: (mk.settle_ms.to_numpy(np.int64)[mk.s.to_numpy() == sg], mk.q.to_numpy(float)[mk.s.to_numpy() == sg]) for sg in (1, -1)}
        own_by[a] = {hh: g for hh, g in L[L.role == 'maker'].groupby('hash', sort=False)}     # each order's own maker rows, in L order
    rows = []
    for i, h in enumerate(hs):
        c, bk, a = cores[h], books[h], int(rank[i])
        width = int(c['post_hi'] - c['post_lo_u']) if c['post_lo_u'] >= 0 else -1
        pc_, why = post_class(c, width)
        ppos, cset = post_position(W, i)
        s_o = 1 if c['y_side'] == 'bid' else -1
        L = Lg[a]
        own = own_by[a].get(h, L.iloc[0:0])
        T_f = own.settle_ms.to_numpy(np.int64)
        o_t, o_q = opp[a][-s_o]
        opp_first, opp_any, n_opp_any = follow_opposite(T_f, o_t, o_q)
        okr = own.status.to_numpy(object) == OK
        T_first, T_last = (int(T_f.min()), int(T_f.max())) if len(T_f) else (int(c['first_hi']), int(c['last_hi']))
        tl = takes_in(T_last, take_t[a], take_sid[a])
        tf = takes_in(T_first, take_t[a], take_sid[a])
        pn = pnl_rows(own)
        rest_hi = float(c['rest_ub'])
        rest_lo = float(max(0, c['first_lo_hard'] - c['post_hi'])) if c['first_lo_hard'] >= 0 else 0.
        tcat, tset = window_category(W, 'touch', i)
        ln = links[h]
        n_i, tk_i = int(W['n'][i]), int(W['touch_known'][i])
        full = n_i > 0 and tk_i == n_i
        dchg = sorted({f for D in WD for f in D_FAMILIES if window_category(WD[D], f, i)[0] != window_category(W, f, i)[0]})
        row = dict(day=day, mid=int(mid), rank=a + 1, addr=TOP3[a], mk_hash=h, y_side=c['y_side'], tick=int(c['tick']), n_fills=int(c['n']),
                   cum_shares=float(c['cum']), first_hi_ms=int(c['first_hi']), last_hi_ms=int(c['last_hi']), ws_ms=int(ws), we_ms=int(we),
                   post_lo_union_ms=int(c['post_lo_u']), post_lo_fifo_ms=int(c['post_lo_fifo']), post_hi_ms=int(c['post_hi']),
                   window_width_ms=width, window_width_bin=str(wbin[i]), window_status='window_unbounded' if width < 0 else 'bounded',
                   eval_lo_ms=int(lo[i]), n_instants=n_i, window_class=c['post_class'], rest_lo_ms=rest_lo, rest_hi_ms=rest_hi,
                   rest_bin=str(cx.age_bin(np.array([rest_hi]))[0]), post_class=pc_, post_class_reason=why, post_position=ppos,
                   post_cat_set=cset, n_unknown_instants=int(n_i - W['post_known'][i]),
                   early_queue_builder_possible=bool(W['post'][0][i] > 0),
                   touch_joiner_possible=bool(W['touch'][0][i] > 0), touch_joiner_only=bool(full and W['touch'][0][i] == n_i),
                   touch_improver_possible=bool(W['touch'][1][i] > 0), touch_improver_only=bool(full and W['touch'][1][i] == n_i),
                   fresh_certain=bool(0 <= width <= LATE_MS), early_possible=bool(width < 0 or width > EARLY_MS),
                   touch_cat=tcat, touch_set=tset, dist_min=R['dist_min'][i], dist_max=R['dist_max'][i],
                   spread_min=R['spread_min'][i], spread_max=R['spread_max'][i], spread_cat=window_category(W, 'spread1', i)[0],
                   depth_min=R['depth_min'][i], depth_max=R['depth_max'][i], depth_cat=window_category(W, 'depth0', i)[0],
                   **{f'btc_{nm}_cat': window_category(W, f'btc_{nm}', i)[0] for nm, _ in MOVE_BP},
                   **{f'btc_{nm}_set': window_category(W, f'btc_{nm}', i)[1] for nm, _ in MOVE_BP},
                   **{f'a_{nm}_min': (s_o * R[f'r_{nm}_min'][i] if s_o > 0 else -R[f'r_{nm}_max'][i]) for nm, _ in MOVE_BP},
                   **{f'a_{nm}_max': (s_o * R[f'r_{nm}_max'][i] if s_o > 0 else -R[f'r_{nm}_min'][i]) for nm, _ in MOVE_BP},
                   n_btc_missing=int(n_i - W['btc_5s_known'][i]), tau_min_s=R['tau_min'][i], tau_max_s=R['tau_max'][i],
                   phase='pre' if c['post_hi'] < ws else ('live' if lo[i] >= ws else 'straddle'),
                   inv_cat=window_category(W, 'inv', i)[0], inv_set=window_category(W, 'inv', i)[1],
                   inv_certain_cat=window_category(W, 'invc', i)[0], inv_possible_cat=window_category(W, 'invp', i)[0],
                   pos_min=R['pos_min'][i], pos_max=R['pos_max'][i], pend_abs_max=R['pend_max'][i],
                   d_sensitive=bool(dchg), d_sensitive_fields='|'.join(dchg),
                   queue_class=bk['front_class'], ahead_acc_min=bk['ahead_lo'], ahead_acc_max=bk['ahead_hi'], ahead_lo_fifo=c['ahead_lo_fifo'],
                   ahead_hi_fifo=c['ahead_hi_fifo'], lo_clock_queue_differs=bool(c['lo_clock_differs']),
                   touch_first_ms=int(bk['touch_first_ms']), touch_at_start=bk['touch_at_start'],
                   nomask_class=bk['nomask_class'], nomask_status=bk['nomask_status'], nomask_n=int(bk['nomask_n']), nomask_rep_lo_ms=int(bk['nomask_rep_lo']),
                   nomask_rep_hi_ms=int(bk['nomask_rep_hi']), nomask_rest_lo_ms=bk['nomask_rest_lo'], nomask_rest_hi_ms=bk['nomask_rest_hi'],
                   n_fills_bin=n_fills_bin(int(c['n'])), n_later_fills=int(c['n'] - 1),
                   n_tied_first=int(np.sum(c['fill_hi'][1:] == c['fill_hi'][0])) if c['n'] > 1 else 0,
                   n_front_after_partial=int(c['n_front_after_partial']), size_status=c['size_status'], union_histories='|'.join(sorted(c['union'])),
                   next_kind=ln['next_kind'], next_link_status=ln['next_link_status'], n_children=ln['n_children'], next_hash=ln.get('next_hash', ''),
                   next_gap_lo_ms=ln.get('next_gap_lo_ms', NAN), next_gap_hi_ms=ln.get('next_gap_hi_ms', NAN), next_d_ticks=ln.get('next_d_ticks', 0),
                   opp_sh_30s_first=opp_first, opp_sh_30s_any=opp_any, n_opp_legs_30s_any=n_opp_any,
                   mq_sh=float(own.mq.to_numpy(float)[okr].sum()), rq_sh=float(own.rq.to_numpy(float)[okr].sum()),
                   n_takes_30s_after_last=tl[0], take_after_last_delay_ms=tl[1], take_after_last_sid=tl[2], n_takes_30s_after_first=tf[0],
                   take_after_first_sid=tf[2], follow_first_lo_ms=T_first, follow_last_lo_ms=T_last,
                   n_own_rows=int(len(own)), **{k: v for k, v in pn.items() if k != 'pnl_excluded'},
                   pnl_excluded_reasons=json.dumps(pn['pnl_excluded'], sort_keys=True), status=_flags(c['flags'] | bk['book_flags']))
        if not okc[i]:
            row.update(CONTEXT_BLANK, status=_flags(c['flags'] | bk['book_flags'] | {'context_not_evaluated'}))
        rows.append(row)
    cnt['position_early_queue_builder_at_every_instant'] += sum(r['post_position'] == 'early_queue_builder' for r in rows)   # 0 by construction
    ctl = dict(hs=hs, rank=rank, ci=ci, si=si, lo=lo, hi=hi, plo=plo, W=W, okc=okc, phase=np.array([r['phase'] for r in rows], object))
    return rows, ctl


# ------------------------------------------------------------------ controls ------------------------------------------------------------------
def control_draws(cx, ctx, ctl, spans, cnt):
    """chain_context.matched_times per order (bounded window, phase pre / live): N_DRAWS starts on the aligned grid -> (starts, status)."""
    n = len(ctl['hs'])
    starts = np.full((n, N_DRAWS), -1, np.int64)
    status = np.full(n, 'none', object)
    for i in range(n):
        ph, w, a = ctl['phase'][i], int(ctl['hi'][i] - ctl['lo'][i]), int(ctl['rank'][i])
        if not ctl['okc'][i]:
            status[i] = 'context_not_evaluated'; continue
        if ctl['plo'][i] < 0:
            status[i] = 'window_unbounded'; continue
        if ph not in ('pre', 'live'):
            status[i] = 'straddle'; continue
        lo_s, hi_s = spans[a]
        if lo_s < 0:
            status[i] = 'no_span'; continue
        mt = cx.matched_times(ctx.M, TOP3[a], n=N_DRAWS, phase_of=np.array([ph], object), width_ms=w,
                              within=(-(-lo_s // STEP) * STEP, hi_s), key=ctl['hs'][i])
        starts[i], status[i] = mt['t'][0], str(mt['status'][0])
        need(bool((mt['t'][0][mt['t'][0] >= 0] % STEP == 0).all()), 'matched_times returned starts off the aligned grid (refused)')
    cnt.update(Counter(f'control_status:{s}' for s in status))
    return starts, status


def control_eval(ctx, ctl, starts):
    """the control windows' counts: lower ends on the aligned grid (grid codes), upper ends by point queries."""
    n, nd = starts.shape
    ok = starts >= 0
    if not ok.any():
        return None, ok
    ii = np.repeat(np.arange(n), nd)[ok.ravel()]
    c0 = starts.ravel()[ok.ravel()]
    w = (ctl['hi'] - ctl['lo'])[ii]
    ci, si, ai = ctl['ci'][ii], ctl['si'][ii], ctl['rank'][ii]
    rows = c0 // STEP - ctx.g0
    need(bool(((rows >= 0) & (rows < ctx.T)).all()), 'control starts outside the market grid (refused)')
    K0 = ctx.grid_codes(rows, ci, si, ai)
    K1, _ = ctx.point(c0 + w, ci, si, ai)
    Wc = ctx.windows(c0, c0 + w, ci, si, ai, ends=(K0, K1))
    full = {}
    for k, v in Wc.items():
        if isinstance(v, list):
            full[k] = []
            for x in v:
                z = np.zeros(n * nd, np.int64); z[ok.ravel()] = x; full[k].append(z.reshape(n, nd))
        else:
            z = np.zeros(n * nd, v.dtype); z[ok.ravel()] = v; full[k] = z.reshape(n, nd)
    return full, ok


def control_stats(ctl, Wc, okd, day):
    """per CTRL statistic: the observed indicator / eligibility per order and the control indicators per draw."""
    rec = dict(rank=ctl['rank'], day=np.array([day] * len(ctl['hs']), object), has=okd.all(1))
    for nm, fam, cats, q in CTRL:
        v0, e0 = indicator(ctl['W'], fam, cats, q)
        if Wc is None:
            vc = ec = np.zeros(okd.shape, bool)
        else:
            vc, ec = indicator(Wc, fam, cats, q)
            ec = ec & okd
        rec[nm] = (v0, e0 & okd.all(1), vc, ec)
    return rec


def ctrl_table(cx, stats, scopes):
    """control rows per scope: obs share vs the N_DRAWS replicate shares (chain_context.null_summary)."""
    rows = []
    for sc_rank, sc_day in scopes:
        for nm, fam, cats, q in CTRL:
            num, den = 0., 0
            rn, rd = np.zeros(N_DRAWS), np.zeros(N_DRAWS)
            nnc = 0
            for r in stats:
                m = np.ones(len(r['rank']), bool)
                if sc_rank != 'top3':
                    m &= r['rank'] == RANKS.index(sc_rank)
                if sc_day != 'all':
                    m &= r['day'] == sc_day
                if not m.any():
                    continue
                v0, e0, vc, ec = r[nm]
                e = e0 & m
                num += float(v0[e].sum()); den += int(e.sum())
                rn += (vc & ec & e[:, None]).sum(0); rd += (ec & e[:, None]).sum(0)
                nnc += int((m & ~r['has']).sum())
            obs = num / den if den else NAN
            rep = np.where(rd > 0, rn / np.maximum(rd, 1), np.nan)
            ns = cx.null_summary(obs if den else None, rep)
            rows.append(dict(table='control', rank=sc_rank, day=sc_day, field=nm, category=f'{fam}:{"|".join(cats) if not isinstance(cats, str) else cats}:{q}',
                             obs=ns['obs'], n_obs=den, null_mean=ns['mean'], null_p05=ns['p05'], null_p50=ns['p50'], null_p95=ns['p95'],
                             p_upper=ns['p_upper'], p_lower=ns['p_lower'], n_draws=ns['n'], n_no_control=nnc,
                             note='outcome-selected population; descriptive association, not a cause'))
    return rows


# ------------------------------------------------------------------ summaries (pure) ------------------------------------------------------------------
def scopes_of(days):
    return [(r, d) for r in RANKS for d in tuple(days) + ('all',)] + [('top3', d) for d in tuple(days) + ('all',)]


def _scope_mask(D, sc_rank, sc_day):
    m = np.ones(len(D), bool)
    if sc_rank != 'top3':
        m &= D['rank'].to_numpy() == RANKS.index(sc_rank) + 1
    if sc_day != 'all':
        m &= D['day'].to_numpy(object) == sc_day
    return m


def class_table(OD, scopes):
    rows = []
    for sc_rank, sc_day in scopes:
        S = OD[_scope_mask(OD, sc_rank, sc_day)]
        for f in CLASS_FIELDS:
            for v, g in S.groupby(S[f].astype(str), sort=True):
                ps = float(g.pnl_shares.sum())
                cps = (lambda x: 100. * float(g[x].sum()) / ps if ps > 0 else NAN)  # noqa: E731
                rows.append(dict(table='classes', rank=sc_rank, day=sc_day, field=f, category=v, n_orders=int(len(g)), n_executions=int(g.n_fills.sum()),
                                 shares=float(g.cum_shares.sum()), pnl_rows=int(g.pnl_rows.sum()), pnl_shares=ps, net_res_c_sh=cps('net_res'),
                                 v_1s_c_sh=cps('v_1s'), v_10s_c_sh=cps('v_10s'), v_60s_c_sh=cps('v_60s'), fee_fill_c_sh=cps('fee_fill'),
                                 pnl_excluded_rows=int(g.pnl_excluded_rows.sum())))
            rows.append(dict(table='classes', rank=sc_rank, day=sc_day, field=f, category='_total', n_orders=int(len(S)),
                             n_executions=int(S.n_fills.sum()), shares=float(S.cum_shares.sum())))
    return rows


def exec_table(OD, FQ, scopes):
    """execution-level queue classes: first fills by queue_class; later fills front_after_partial / later_other (counts and shares)."""
    rows = []
    for sc_rank, sc_day in scopes:
        S, Fq = OD[_scope_mask(OD, sc_rank, sc_day)], FQ[_scope_mask(FQ, sc_rank, sc_day)]
        for v, g in Fq[Fq.kind == 'first'].groupby('queue_class', sort=True):
            rows.append(dict(table='executions', rank=sc_rank, day=sc_day, field='first_fill_queue_class', category=v, n_orders=int(len(g)),
                             n_executions=int(len(g)), shares=float(g.q.sum())))
        for v, g in Fq[Fq.kind != 'first'].groupby('kind', sort=True):
            rows.append(dict(table='executions', rank=sc_rank, day=sc_day, field='later_fill', category=v, n_orders=int(g.mk_hash.nunique()),
                             n_executions=int(len(g)), shares=float(g.q.sum())))
        rows.append(dict(table='executions', rank=sc_rank, day=sc_day, field='all_fills', category='_total', n_orders=int(len(S)),
                         n_executions=int(len(Fq)), shares=float(Fq.q.sum())))
    return rows


def status_table(OD, scopes):
    rows = []
    for sc_rank, sc_day in scopes:
        S = OD[_scope_mask(OD, sc_rank, sc_day)]
        cnt = Counter(x for st in S.status for x in str(st).split('+'))
        for k, v in sorted(cnt.items()):
            rows.append(dict(table='statuses', rank=sc_rank, day=sc_day, field='status_flag', category=k, n_orders=int(v)))
    return rows


def fill_queue_rows(day, mid, cores):
    rows = []
    for h, c in cores.items():
        a = LOW3.index(c['signer'].lower())
        q, ahi = c['fill_q'], c['fill_ahi_fifo']
        for j in range(len(q)):
            kind = 'first' if j == 0 else ('front_after_partial' if ahi[j] == 0. else 'later_other')
            rows.append(dict(day=day, mid=mid, rank=a + 1, mk_hash=h, kind=kind, q=float(q[j])))
    return rows


# ------------------------------------------------------------------ inputs ------------------------------------------------------------------
def check_top3(path):
    need(Path(path).is_file(), f'identities: input missing: {Path(path).name}')
    ea = json.loads(Path(path).read_text()).get('ex_ante') or {}
    need(ea.get('day') == C.ID_DAY, f'identities: ex_ante is not from the ID day {C.ID_DAY}')
    lst = ea.get('top10_shares') or []
    need(len(lst) >= 3 and [x.get('rank') for x in lst[:3]] == [1, 2, 3], 'identities: ex_ante.top10_shares does not start with ranks 1-3')
    addrs = tuple(str(x.get('addr')) for x in lst[:3])
    need(addrs == TOP3, 'identities: ex_ante ranks 1-3 differ from pf_mt_decision.TOP3 (refused)')
    return addrs


def book_markets(days):
    bk = E.load_book()
    need(sum(len(v) for v in bk.values()) == N_BOOK, f'BOOK manifest: {sum(len(v) for v in bk.values())} markets, expected {N_BOOK} (refused)')
    need(set(bk) <= set(C.MEASURE_DAYS), 'BOOK manifest: a day outside the measurement days (refused)')
    return [(d, int(m)) for d in sorted(bk) if d in days for m in sorted(bk[d])]


def read_fv(path, mk, cnt):
    """the top-3 rows of the BOOK markets `mk` [(day, mid)] through the row-group mid statistics (every status); every row's day must be its
    market's day."""
    mids = np.array(sorted({m for _, m in mk}), np.int64)
    day_of = {m: d for d, m in mk}
    pf = pq.ParquetFile(path); md = pf.metadata
    names = [md.row_group(0).column(k).path_in_schema for k in range(md.num_columns)] if md.num_row_groups else []
    need(all(c in names for c in FV_COLS), f'fills_valued lacks {[c for c in FV_COLS if c not in names]}')
    jm = names.index('mid')
    vm, va = pa.array(mids), pa.array(list(LOW3))
    parts = []
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(jm).statistics
        if st is not None and st.has_min_max:
            k = int(np.searchsorted(mids, int(st.min), 'left'))
            if k >= len(mids) or mids[k] > int(st.max):
                cnt['fv_groups_pruned'] += 1
                continue
        t = pf.read_row_group(i, columns=list(FV_COLS))
        cnt['fv_groups_read'] += 1; cnt['fv_rows_scanned'] += t.num_rows
        t = pa.table({c: (t.column(c).cast(t.column(c).type.value_type) if pa.types.is_dictionary(t.column(c).type) else t.column(c)) for c in FV_COLS})
        m = pc.fill_null(pc.and_(pc.is_in(t.column('mid'), value_set=vm), pc.is_in(pc.utf8_lower(t.column('addr')), value_set=va)), False)
        if pc.any(m).as_py():
            parts.append(t.filter(m))
        del t
    need(parts, 'fills_valued: no row of the top 3 in the BOOK markets')
    df = pa.concat_tables(parts).to_pandas()
    df['addr_l'] = df.addr.str.lower()
    sp = df.groupby('addr_l').addr.nunique()
    need((sp <= 1).all(), f'fills_valued: an address under two spellings {sp[sp > 1].to_dict()} (refused)')
    bad = df.day.to_numpy(object) != np.array([day_of[int(m)] for m in df.mid], object)
    need(not bad.any(), f'fills_valued: {int(bad.sum())} rows of a BOOK market under another day (refused)')
    need(set(df.role.unique()) <= {'maker', 'taker'}, 'fills_valued: a role other than maker / taker')
    cnt['fv_rows_top3_book'] = int(len(df))
    return df


def read_links(inp, day, mid):
    need(day in C.MEASURE_DAYS, f'day {day} is not a measurement day (never opened)')
    p = Path(inp) / 'life' / f'{day}_{int(mid)}_x_links.parquet'
    need(p.is_file(), f'missing lifecycle output: life/{p.name}')
    return pd.read_parquet(p), p


def _git(files):
    head = subprocess.run(['git', '-C', str(C.REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    st = {}
    for p in files:
        p = Path(p)
        try:
            rel = str(p.resolve().relative_to(C.REPO))
        except ValueError:
            continue
        tr = subprocess.run(['git', '-C', str(C.REPO), 'ls-files', '--error-unmatch', rel], capture_output=True).returncode == 0
        clean = tr and subprocess.run(['git', '-C', str(C.REPO), 'diff', '--quiet', 'HEAD', '--', rel]).returncode == 0
        st[rel] = dict(tracked=tr, clean=bool(clean), sha256=_sha(p) if p.is_file() else None)
    return dict(head=head, files=st)


def compare_outputs(new_dir, ref_dir):
    """every retained column of orders.csv / summary.csv against a reference output of the same scope, compared as TEXT (both read with
    dtype=str and no NA parsing), rows matched by key; the reference's TAKE_DROPPED columns and TAKE_DROPPED_FIELDS summary rows are the
    declared drops -> dict(identical, per file: row counts, same_row_keys, columns compared / missing / new, mismatched {column: rows})."""
    new_dir, ref_dir = Path(new_dir), Path(ref_dir)
    need((ref_dir / 'orders.csv').is_file() and (ref_dir / 'summary.csv').is_file(), f'compare: no reference outputs in {ref_dir.name} (refused)')
    out = dict(reference=ref_dir.name)
    for name, key, drop_rows in (('orders.csv', ['day', 'mid', 'mk_hash'], ()),
                                 ('summary.csv', ['table', 'rank', 'day', 'field', 'category'], TAKE_DROPPED_FIELDS)):
        a = pd.read_csv(ref_dir / name, dtype=str, keep_default_na=False)
        b = pd.read_csv(new_dir / name, dtype=str, keep_default_na=False)
        if len(drop_rows):
            a = a[~a.field.isin(drop_rows)]
        cols = [c for c in a.columns if c not in TAKE_DROPPED and c not in key]
        missing = [c for c in cols if c not in b.columns]
        new_cols = [c for c in b.columns if c not in a.columns]
        need(not a.duplicated(key).any() and not b.duplicated(key).any(), f'compare: {name} keys are not unique (refused)')
        a, b = a.set_index(key).sort_index(), b.set_index(key).sort_index()
        same = bool(a.index.equals(b.index))
        mism = {}
        if same:
            for c in cols:
                if c not in missing:
                    k = int((a[c].to_numpy(object) != b[c].to_numpy(object)).sum())
                    if k:
                        mism[c] = k
        out[name] = dict(rows_ref=int(len(a)), rows_new=int(len(b)), same_row_keys=same, columns_compared=len(cols) - len(missing),
                         missing_columns=missing, new_columns=new_cols, mismatched=mism)
    out['identical'] = all(out[n]['same_row_keys'] and not out[n]['missing_columns'] and not out[n]['mismatched'] for n in ('orders.csv', 'summary.csv'))
    return out


# ------------------------------------------------------------------ one market (the run's per-market pass; the selftest calls it directly) ------------------------------------------------------------------
def market_pass(cx, M, Lk, fv_m, ranks, cnt, check=True, stage=lambda name: None):
    """one loaded chain_context market M: estimator definitions, the context grid, the order rows, the controls -> (order rows, fill-queue
    rows, control statistics or None, info). fv_m = the market's fills_valued rows of the top 3 (every status); Lk = its x_links."""
    d, m, ws, we = str(M.day), int(M.mid), int(M.ws), int(M.we)
    want = tuple(LOW3[r] for r in ranks)
    O, Fl = M.orders, M.life_fills
    sig = O.mk_signer.astype(str).str.lower()
    n_top = int(O[sig.isin(want)].mk_hash.nunique())
    cnt['orders_top3'] += n_top
    if not n_top:
        cnt['markets_without_orders'] += 1
        return [], [], None, dict(orders=0)
    cores, books, links, cells, prints = market_orders(O, Fl, Lk, fv_m[fv_m.addr_l.isin(want)], M.depth['B'], ws, want)
    stage('estimator')
    spans = []
    for a in LOW3:
        st_ = fv_m.settle_ms[fv_m.addr_l == a].to_numpy(np.int64)
        spans.append((int(st_.min()), int(st_.max())) if len(st_) else (-1, -1))
    lo_all = [cores[h]['post_lo_u'] for h in cores if cores[h]['post_lo_u'] >= 0] + [s[0] for s in spans if s[0] >= 0]
    hi_all = [cores[h]['post_hi'] for h in cores] + [s[1] for s in spans if s[1] >= 0]
    g_lo = (min(lo_all + [ws - PRE_MS]) - max(cx.D_PROFILES) - STEP) // STEP * STEP
    g_hi = (max(hi_all + [we]) // STEP + 1) * STEP
    ctx = Context(cx, M, np.arange(g_lo, g_hi + STEP, STEP, dtype=np.int64), cells, TOP3, prints)
    stage('context_series')
    Wn = OE.windows_of(Fl)
    Lg = {a: OE.legs_frame(fv_m[fv_m.addr_l == LOW3[a]], Wn) for a in ranks}
    for a in ranks:
        cnt['legs_without_window'] += int((Lg[a].wst == 'no_window').sum())
    rows, ctl = order_rows(cx, ctx, d, m, ws, we, cores, books, links, Lg, cnt, check=check)
    fq = fill_queue_rows(d, m, cores)
    stage('orders')
    st, starts = None, None
    if ctl is not None:
        starts, _ = control_draws(cx, ctx, ctl, spans, cnt)
        stage('controls_draw')
        Wc, okd = control_eval(ctx, ctl, starts)
        st = control_stats(ctl, Wc, okd, d)
        stage('controls_eval')
    return rows, fq, st, dict(orders=len(rows), grid=int(ctx.T), cells=len(cells), ctx=ctx, ctl=ctl, starts=starts, cores=cores, spans=spans)


# ------------------------------------------------------------------ run ------------------------------------------------------------------
def run(cx, inp=None, out=None, days=None, ranks=None, pilot=None, guard_mib=C.GUARD_MIB, log=print, fv_sha=FV_SHA256, check=True,  # noqa: C901
        compare=None):
    """The census: every (day, BOOK market) of `days`, the address indices `ranks` (0..2). cx = the chain_context module (the selftest
    injects one whose load_market builds its synthetic markets). Writes the write-once output directory and returns the receipt dict."""
    cpu0, wall0, asof0 = time.process_time(), time.time(), _now()
    need(inp is not None or C.RUN == 'v2', f'MT_RUN must be v2 (is {C.RUN!r})')
    days = tuple(days) if days is not None else tuple(C.MEASURE_DAYS)
    need(len(days) >= 1 and len(set(days)) == len(days) and all(d in C.MEASURE_DAYS for d in days),
         'days must be distinct measurement days (pf_mt_common.MEASURE_DAYS; refused before any path is built)')
    days = tuple(sorted(days))
    ranks = tuple(sorted(set(ranks))) if ranks is not None else (0, 1, 2)
    need(all(r in (0, 1, 2) for r in ranks), 'ranks must be in 0..2')
    need(LATE_MS == 1_000 and EARLY_MS == 10_000, 'order_estimator LATE_MS / EARLY_MS differ from the contract (1,000 / 10,000 ms)')
    need(dict((f'r_{nm}', thr) for nm, thr in MOVE_BP) == dict(cx.MOVE_THR), 'MOVE_BP differs from chain_context.MOVE_THR')
    need(tuple(cx.D_PROFILES) == (0, 250, 1000), 'chain_context.D_PROFILES differ from the contract (0 / 250 / 1000 ms)')
    inp = Path(inp) if inp is not None else C.OUT
    out = Path(out) if out is not None else (C.RECEIPT / 'chains' / ('post_fill' if pilot is None else f'post_fill_pilot_{pilot[0]}_r{pilot[1]}'))
    need(not out.exists(), f'write-once: {out} exists (refusing to overwrite)')
    tmp = out.with_name(f'.{out.name}.partial{os.getpid()}')
    need(not tmp.exists(), f'a partial output exists: {tmp}')
    stage_cpu, cnt, hashes, hc = {}, Counter(), {}, {}
    t_st = [time.process_time()]

    def stage(name):
        now = time.process_time()
        stage_cpu[name] = round(stage_cpu.get(name, 0.) + now - t_st[0], 4)
        t_st[0] = now
        guard(name, guard_mib, cpu0)
    idp = inp / 'econ' / 'identities.json'
    check_top3(idp)
    hashes['econ/identities.json'] = dict(sha256=_sha(idp), bytes=idp.stat().st_size)
    mk = book_markets(days)
    need(mk, 'no BOOK market on the requested days')
    fvp = inp / 'econ' / 'fills_valued.parquet'
    hc[str(fvp.resolve())] = fsha = _sha(fvp)                        # shared with chain_context.load_market: one hash per file per run
    need(fv_sha is None or fsha == fv_sha, f'fills_valued sha256 {fsha} differs from the accepted binding {fv_sha} (refused)')
    hashes['econ/fills_valued.parquet'] = dict(sha256=fsha, bytes=fvp.stat().st_size)
    stage('preflight')
    fv = read_fv(fvp, mk, cnt)
    stage('fills_valued')
    rows, fq_rows, stats, market_cpu = [], [], [], []
    for d, m in mk:
        c_m0 = time.process_time()
        M = cx.load_market(d, m, TOP3, root=inp, hash_cache=hc)
        need(str(M.day) == d and int(M.mid) == m, f'chain_context returned market {M.day} {M.mid} for {d} {m} (refused)')
        for k, v in M.inputs.items():
            hashes[k] = v
        Lk, pl = read_links(inp, d, m)
        hashes[f'life/{pl.name}'] = dict(sha256=_sha(pl), bytes=pl.stat().st_size)
        stage('context_load')
        r_, fq_, st_, info = market_pass(cx, M, Lk, fv[fv.mid == m], ranks, cnt, check=check, stage=stage)
        rows += r_
        fq_rows += fq_
        if st_ is not None:
            stats.append(st_)
        market_cpu.append(dict(day=d, mid=m, cpu_s=round(time.process_time() - c_m0, 4), orders=len(r_), grid=info.get('grid', 0),
                               cells=info.get('cells', 0)))
        log(f'{d} {m}: orders {len(r_)} grid {info.get("grid", 0)} cells {info.get("cells", 0)} cpu {time.process_time() - c_m0:.2f} s '
            f'vmhwm {vmhwm_mib():.0f} MiB')
        del M, info
    OD = pd.DataFrame(rows)
    FQ = pd.DataFrame(fq_rows, columns=['day', 'mid', 'rank', 'mk_hash', 'kind', 'q'])
    if len(OD):
        FQ = FQ.merge(OD[['mk_hash', 'mid', 'queue_class']], on=['mk_hash', 'mid'], how='left')
    sc = scopes_of(days)
    SM = pd.DataFrame(class_table(OD, sc) + exec_table(OD, FQ, sc) + ctrl_table(cx, stats, sc) + status_table(OD, sc), columns=list(SUMMARY_COLS)) \
        if len(OD) else pd.DataFrame(columns=list(SUMMARY_COLS))
    stage('summary')
    tmp.mkdir(parents=True)
    try:
        OD.to_csv(tmp / 'orders.csv', index=False)
        SM.to_csv(tmp / 'summary.csv', index=False)
        srcs = [Path(__file__).resolve(), HERE / 'chain_post_fill_selftest.py', HERE / 'order_estimator.py', CONTRACT_FILE]
        cxf = getattr(cx, '__file__', None)
        if cxf:
            srcs.append(Path(cxf).resolve())
        sources = {(str(Path(p).relative_to(C.REPO)) if str(p).startswith(str(C.REPO)) else str(p)): _sha(p) for p in srcs if Path(p).is_file()}
        stage('hash')
        rec = dict(version=VERSION, module='chain_post_fill', asof_start=asof0, asof_end=_now(), argv=sys.argv, mt_run=C.RUN, slot=os.environ.get('PF_SLOT'),
                   contract=str(CONTRACT_FILE.relative_to(C.REPO)), contract_sha256=_sha(CONTRACT_FILE) if CONTRACT_FILE.is_file() else None,
                   pilot=None if pilot is None else dict(day=pilot[0], rank=pilot[1]), days=list(days), ranks=[RANKS[r] for r in ranks],
                   markets=[f'{d}_{m}' for d, m in mk], inputs=hashes, sources=sources, git=_git(srcs),
                   chain_context=dict(module=str(cxf) if cxf else None, version=getattr(cx, 'VERSION', None)),
                   declared=dict(STEP=STEP, LATE_MS=LATE_MS, EARLY_MS=EARLY_MS, FOLLOW_MS=FOLLOW_MS, PRE_MS=PRE_MS, TOL=TOL,
                                 POS_EPS=POS_EPS, MOVE_BP=MOVE_BP, N_DRAWS=N_DRAWS, SEED=SEED, D_PROFILES=list(cx.D_PROFILES), D_FAMILIES=D_FAMILIES,
                                 CHECK_EVERY=CHECK_EVERY, CHECK_K=CHECK_K, POST_CATS=POST_CATS, POST_CLASSES=POST_CLASSES, TOUCH_CATS=TOUCH_CATS,
                                 BTC_CATS=BTC_CATS, INV_CATS=INV_CATS, CTRL=[[nm, fam, list(cats), q] for nm, fam, cats, q in CTRL],
                                 CLASS_FIELDS=CLASS_FIELDS, REDUCING=sorted(REDUCING), RES_VALID=list(E.RES_VALID),
                                 controls='chain_context.matched_times(n=N_DRAWS, phase pool, within = the active span, key = hash); no fallback',
                                 p='chain_context.null_summary: p_upper = (1 + #rep >= obs) / (1 + reps); p_lower = (1 + #rep <= obs) / (1 + reps)',
                                 instants='chain_context.over_window convention: lo, every 100 ms grid instant strictly inside, hi',
                                 post_hi_note='post_hi = the first fill hi (lifecycle, checked): resting lower bound 0; an early post is never '
                                              'identified (early_queue_builder only as _possible); post_class = the width identification class'),
                   labels=LABELS, counts=dict(sorted(cnt.items())), rows=dict(orders=len(OD), summary=len(SM)),
                   cpu_stage_s=stage_cpu, market_cpu=market_cpu, cpu_total_s=round(time.process_time() - cpu0, 3),
                   wall_s=round(time.time() - wall0, 3), vmhwm_mib=round(vmhwm_mib(), 1), guard_mib=guard_mib)
        if pilot is not None:
            rec['projection'] = projection(rec, inp)
        if compare is not None:
            rec['compare'] = compare_outputs(tmp, compare)
        rec['declared']['take_dropped'] = dict(columns=TAKE_DROPPED, summary_fields=TAKE_DROPPED_FIELDS,
                                               join='chain_take take classes on (day, mid, address, sid): take_after_last_sid / '
                                                    'take_after_first_sid, or every own take with settle_ms in (follow_*_lo_ms, + FOLLOW_MS]')
        rec['outputs'] = {p.name: dict(sha256=_sha(p), bytes=p.stat().st_size) for p in sorted(tmp.iterdir())}
        (tmp / 'receipt.json').write_text(json.dumps(_clean(rec), indent=1, sort_keys=True, allow_nan=False) + '\n')
        need(not out.exists(), f'write-once: {out} appeared during the run (refusing to overwrite)')
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return rec


def projection(rec, inp):
    """pilot -> the full run: fixed (preflight + fills_valued + summary + hash) + per market (context load + series) x 19 + per order
    (estimator + orders + controls) x the top-3 orders of the 19 markets (S_NOPRIO key columns counted)."""
    s = rec['cpu_stage_s']
    nm = max(len(rec['market_cpu']), 1)
    no = max(sum(x['orders'] for x in rec['market_cpu']), 1)
    n_full = 0
    for d, v in E.load_book().items():
        for m in v:
            o = pd.read_parquet(Path(inp) / 'life' / f'{d}_{int(m)}_orders.parquet', columns=['scenario', 'mk_signer'])
            n_full += int(o[o.scenario == 'S_NOPRIO'].mk_signer.astype(str).str.lower().isin(LOW3).sum())
    fixed = sum(s.get(k, 0.) for k in ('preflight', 'fills_valued', 'summary', 'hash'))
    per_m = sum(s.get(k, 0.) for k in ('context_load', 'context_series')) / nm
    per_o = sum(s.get(k, 0.) for k in ('estimator', 'orders', 'controls_draw', 'controls_eval')) / no
    full = fixed + per_m * N_BOOK + per_o * n_full
    return dict(fixed_s=round(fixed, 3), per_market_s=round(per_m, 4), per_order_s=round(per_o, 6), markets_in_pilot=nm, orders_in_pilot=no,
                orders_full=n_full, full_run_s=round(full, 2), full_run_core_h=round(full / 3600., 5),
                note='one address-day: the per-market context cost is measured on its markets (each loads all three addresses); the per-order '
                     'cost assumes the other addresses cost alike (one point, not a size-volume fit)')


def main(argv):
    usage = 'usage: chain_post_fill.py run | chain_post_fill.py pilot <measurement day> <rank 1-3> [--suffix S] [--compare DIR under chains/]'
    need(len(argv) >= 2 and argv[1] in ('run', 'pilot'), usage)
    opts = {}
    if argv[1] == 'pilot':
        need(len(argv) >= 4 and argv[2] in C.MEASURE_DAYS and argv[3] in ('1', '2', '3'), usage)
        rest = argv[4:]
        need(len(rest) % 2 == 0 and all(k in ('--suffix', '--compare') for k in rest[0::2]), usage)
        opts = dict(zip(rest[0::2], rest[1::2]))
        need(all(re.fullmatch(r'[A-Za-z0-9_]+', v) for v in opts.values()), 'pilot options: letters, digits and _ only (names under chains/)')
    else:
        need(len(argv) == 2, usage)
    import chain_context as cx            # the one definition of market context (contract 7.0); required
    if argv[1] == 'run':
        rec = run(cx)
    else:
        d, r = argv[2], int(argv[3])
        rec = run(cx, days=(d,), ranks=(r - 1,), pilot=(d, r), out=C.RECEIPT / 'chains' / f'post_fill_pilot_{d}_r{r}{opts.get("--suffix", "")}',
                  compare=(C.RECEIPT / 'chains' / opts['--compare']) if '--compare' in opts else None)
    print(json.dumps(_clean(dict(rows=rec['rows'], counts=rec['counts'], cpu_stage_s=rec['cpu_stage_s'], cpu_total_s=rec['cpu_total_s'],
                                 vmhwm_mib=rec['vmhwm_mib'], projection=rec.get('projection'), compare=rec.get('compare'))), indent=1))


if __name__ == '__main__':
    main(sys.argv)
