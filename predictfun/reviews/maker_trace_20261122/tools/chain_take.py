"""TAKE CHAIN + ACTION-TYPE ENUMERATION (P-2026-005 MAKER-TRACE logic-chain packet, census tool 7.3; session aea5bfb9 development,
2026-11-23). USER (binding): "And they also actively take to reduce inventory, also need to check the market condition. I have not listed
all actions here. You will analyze the actions and corresponding market context or logics." Spec: order_estimator_contract.md sections 2
(clocks, statuses), 5 (takes), 6 (TAKE logic rows), 7.0 (chain_context), 7.3 (this tool); plans automatic/20261123_maker_decision_timing,
_decision_logic, _order_estimator; v2/report_interpretation_v3.md (actual |position| reduction, never the opposing-side test).
DESCRIPTIVE ONLY on consumed days (Oct 24-27: four point estimates, no intervals, G = 4 < 5). Not a fitted model, an imitation strategy, a
replay, a validation, a calibrated probability or a profitability claim; it measures no model skill. Every rule below is declared before
any result of this tool exists; synthetic proof (hand-computed classes, a recovered and a null association, action counts, a look-ahead
falsifier, refusals, every writer end to end): chain_take_selftest.py.
POPULATION ex-ante top 3 (econ/identities.json ex_ante.top10_shares ranks 1-3 = closes.TOP3, else refuse) x the 19 BOOK markets
       (pf_mt_econ.BOOK_MANIFEST 'complete'; exactly 19, every day a measurement day, else refuse). Per market everything comes from
       chain_context.load_market (the one definition, contract 7.0): ws / we, BTC, the depth book, the lifecycle orders / fills and the 3
       addresses' fills_valued legs of every status with their lifecycle windows (M.legs ok, M.excluded not ok); fills_valued sha256 must
       equal the accepted matched_gap_r1 binding (else refuse). x_links: life/<day>_<mid>_x_links.parquet (lifecycle, reused as is).
CLOCKS a leg's match window is chain_context's [lo_hard_ms, hi_ms] (lifecycle x_lo_hard = hi - MAX_LAG_MS, verified; -1 = unbounded; a leg
       without a lifecycle row gets the settle-based window). A take with hi < settle_ms is receipt_before_uuid, with lo_hard -1
       window_unbounded: both give no bound to any claim (kept, counted). D PROFILES (decision delay alternatives) D in (0, 50, 250, 1000)
       ms = the contract's {0, 250, 1000} UNION the dispatch's {0, 50, 250}: decision window [lo_hard - D, hi] (D shifts the lower end
       down, contract 7.0); D = 0 primary; every D reported. INSTANTS of a window = its two ends and every 100 ms grid instant strictly
       inside (chain_context.over_window's set); every feature at an instant reads data <= that instant only.
TAKE   unit = one record (sid) whose taker is the address (role taker), whatever its status; tk_hash = its taker order (records of one
       tk_hash listed together: tk_n_records, tk_seq). dq = s x (its shares less self-trade shares), YES space, every leg status (excluded
       legs flagged excluded_row; contract 5: pos_after = pos_before + this record's legs).
INVENTORY chain_context.position (status-ok legs; self-trade rows cancel): recorded = position at the take's settle_ms (legs with settle <
       it: strictly earlier), certain(D) = 'certain' at lo_hard - D (legs with hi < it), reconciled against this tool's own action sum
       (refused on a mismatch); simultaneous = own actions (sid, role; self-trades out) at the take's settle ms of another sid, evaluated
       excluded (= recorded) and included; pending(D) = own non-self legs sequenced at or before the take (settle <= its settle, another
       sid) whose window meets [lo_hard - D, hi].
CLASS  of (before, dq), eps = 1e-6 sh (contract 5; dispatch names in DISPATCH_NAMES; equal to order_estimator.inv_class, selftest-checked):
       no_change |dq| <= eps; from_flat |before| <= eps; add same direction; opposing: full_close |after| <= eps, reduce_partial same sign,
       reversal_reducing sign flip with |after| < |before| - eps, overshoot otherwise. Actual reduction = reduce_partial | full_close |
       reversal_reducing; exposure-adding = add | overshoot | from_flat. take_class_D = the common class of {recorded, certain(D),
       simultaneous-included}, else timing_sensitive (= the dispatch's ambiguous_pending); take_class = take_class_0.
CONTEXT per D over the decision window's instants: BTC r_1s / r_5s / r_30s per instant with / against (s x r >= / <= +- chain_context.MOVE_THR
       0.5 / 1 / 2 bp) / flat -> window category with | against | flat (constant, none missing) | mixed (two categories) | gap (constant on
       the valid instants, some missing) | missing (none valid); p_fair edge 100 s (p_fair - p_avg) cents min / max; spread min / max. At
       the last usable frame <= the window start (chain_context.book, status ok): the touch on the taken side (buy: best ask, sell: best
       bid), spread, depth at that touch (level_size), ticks through (worst: buy max leg tick - ask, sell bid - min leg tick; avg likewise).
       tau and phase at hi (pre < ws <= live); |position| bin (chain_context POS_EDGES) and sign of the recorded before; preceding own maker
       fill (last own maker leg with settle < the take's): delay bounds [max(0, lo_hard - prev.hi), max(0, hi - prev.lo_hard)]. Fill burst
       at the window start lo: certain = own maker legs with lo_hard >= lo - 5 s and hi < lo; possible = windows meeting [lo - 5 s, lo),
       sequenced at or before the take; burst3 = certain >= 3; recorded counts settle in [lo - 1 s, lo) / [lo - 5 s, lo) and the offsetting
       shares (s_fill = -s) in [lo - 5 s, lo).
MECHANISMS per take and D (contract 6 TAKE rows): inventory_unwind compatible iff every variant actually reduces, contradicted iff every
       variant is add / overshoot, unidentified iff every variant is from_flat / no_change (or one is unclassified), else timing_sensitive;
       directional (r_5s window category) with -> compatible, against -> contradicted, flat -> insufficient_comparison, mixed ->
       timing_sensitive, gap / missing -> unidentified; after_fill_burst certain >= 3 -> compatible, possible == 0 -> contradicted, else
       timing_sensitive. A take flagged NO_CLAIM is unidentified for all three and enters no comparison (counted).
CONTROLS (i) matched no-take instants: grid instants c of the address's active span in the market (first to last own leg settle, clipped to
       [ws - 300 s, we)), no own take record settle within +- 2 s; stratum (phase(c), sign and |position| bin of the recorded position at c)
       = the take's (phase(hi), recorded before); empty -> fallback1 (phase, bin) -> fallback2 (phase) -> no_control (counted). Control
       window [c - 2,000 - D, c] (the take's width in ms); every feature by the same functions with the take's side; 'reduction_if_taken' =
       the take's dq at the recorded position at c actually reduces. N_DRAWS = 500 draws (one instant per take per draw) from
       Generator(PCG64(SeedSequence([20261123, day, mid, rank]))); per group: observed mean, null mean, p05 / p50 / p95, p_hi = (1 + #null
       >= obs) / (1 + n), p_lo likewise; binary features add the odds ratio (the take-rate ratio between instants with and without the
       feature). (ii) reducing vs adding takes (take_class) on the same features: labels permuted within address x market (N_PERM = 500,
       SeedSequence([20261123, day, mid, rank, 1])); statistic mean(reducing) - mean(adding). Groups: address x day, address, pooled.
ACTIONS (contract 7.3) per address x day: maker_fill (base), post_new_level (the first filled order at that side and tick in the market),
       post_same_level, refill / reprice (x_links pairs through order_estimator.link_hypotheses, the single definition: feasible sets and
       identified subsets; churn bound [0, #replace feasible]), ladder_overlap (same-side pairs at different ticks with link_hypotheses
       overlap_certain), keep_after_fill (every non-last fill), pull_possible:<view>:<class> (chain_pull.order_table, S_NOPRIO and S_FIFO,
       never pooled), take_<class>, two_sided (an opposite-side order whose feasible life [post_lo_union, end_hi or we] overlaps; certain
       when an opposite fill window lies inside the order's certain-presence span; quoted width in ticks), and the types the data show
       (take_self_trade, take_sweep, take_multi_record_order, take_kind_mint / merge). Each: counts and denominators, population (filled
       orders = OUTCOME-SELECTED: unfilled orders are invisible; take records are complete public records), context at its declared anchor
       (tau, live share, spread, BTC move / with share, offsetting-position share, |position| >= 200 share), the next own event after it,
       mechanism-status counts and one logic-table row whose status is the most frequent mechanism status (ties -> unidentified).
OUTPUTS C.RECEIPT / chains / take (pilot: chains / take_pilot_<day>_r<rank>); write-once, built in a partial directory renamed at the end:
       takes.csv, take_controls.csv, actions.csv, summary.csv, receipt.json (inputs' and sources' sha256, git, counts, statuses, checks, CPU
       by stage, VmHWM; the pilot adds the projection).
GUARDS MT_RUN=v2; measurement days only (chain_context refuses another day before any path is built; x_links paths are built for manifest
       days only); VmHWM self-guard pf_mt_common.GUARD_MIB (1,450 MiB, never raised); CPU self-guard CPU_CAP_S; run refuses unless this tool,
       its selftest, chain_context.py and the contract are tracked and unmodified (the freeze).
Usage (run_light): MT_RUN=v2 chain_take.py run | MT_RUN=v2 chain_take.py pilot <day> <rank>      selftest: chain_take_selftest.py"""
import collections
import json
import os
import shutil
import sys
import time
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
import closes as X  # noqa: E402

need, OK = C.need, C.STATUS_OK
NAN = float('nan')
VERSION = 'chain_take_v1'
CONTRACT = 'predictfun/reviews/maker_trace_20261122/order_estimator_contract.md (sections 2, 5, 6 TAKE rows, 7.0, 7.3)'
CONTRACT_FILE = C.RECEIPT_V1 / 'order_estimator_contract.md'
TOOL_FILES = (HERE / 'chain_take.py', HERE / 'chain_take_selftest.py', HERE / 'chain_context.py', CONTRACT_FILE)
DEP_FILES = (HERE / 'order_estimator.py', HERE / 'chain_pull.py', HERE / 'closes.py', RESEARCH / 'pf_mt_common.py', RESEARCH / 'pf_mt_econ.py',
             RESEARCH / 'pf_mt_lifecycle.py', RESEARCH / 'pf_mt_decision.py')
FV_RECEIPT = C.RECEIPT_V1 / 'v2' / 'matched_gap_r1' / 'receipt.json'
TOP3 = X.TOP3
RANKS = ('rank1', 'rank2', 'rank3')
N_BOOK = 19

# ------------------------------------------------------------------ declared constants (none fitted) ------------------------------------------------------------------
STEP = 100                                       # ms: the instant grid (contract 7.0 window evaluation)
MAX_LAG = C.MAX_LAG_MS                           # 2,000: lo_hard = hi - MAX_LAG (lifecycle premise; verified per leg)
D_CONTRACT, D_DISPATCH = (0, 250, 1000), (0, 50, 250)
D_PROFILES = tuple(sorted(set(D_CONTRACT) | set(D_DISPATCH)))
D_PRIMARY = 0
POS_EPS = 1e-6
POS_BINS = (50., 200., 1000.)                    # contract 7.0 = chain_context.POS_EDGES (checked at load)
MOVES = (('r_1s', 0.5), ('r_5s', 1.0), ('r_30s', 2.0))   # contract 7.0 = chain_context.MOVE_THR (checked at load)
BURST_N, BURST_MS, REC_MS = 3, 5_000, (1_000, 5_000)
EXCL_MS = 2_000                                  # no own take within +- 2 s of a control instant
PRE_MS = 300_000                                 # active span clipped at ws - 300 s
N_DRAWS, N_PERM, MIN_DRAWS, SEED = 500, 500, 200, 20261123
KEEP_REACT_MS = 1_000                            # contract KEEP rule: no time to react
TOL_SH = 0.01
REC_TOL = 1e-6                                   # recorded-position reconciliation (shares)
CPU_CAP_S = 120.
GRID_BACK_MS = BURST_MS + MAX_LAG + max(D_PROFILES) + 2 * STEP
N_PAIRS_FULL = 49                                # top-3 x BOOK pairs with rows (mt_oe_inspect: rank 2 absent on 8 of 19 markets); projection only
_MUT_LEAK_MS = 0                                 # planted-error switch (selftest only): evaluate each window's upper end this many ms LATER

CLS = ('from_flat', 'add', 'reduce_partial', 'full_close', 'reversal_reducing', 'overshoot', 'no_change', 'unclassified', 'timing_sensitive')
CI = {c: i for i, c in enumerate(CLS)}
REDUCING = ('reduce_partial', 'full_close', 'reversal_reducing')
ADDING = ('add', 'overshoot', 'from_flat')
DISPATCH_NAMES = dict(reduce_partial='reduce', full_close='full_close', reversal_reducing='reversal_overshoot', overshoot='opposing_not_reducing',
                      add='add', from_flat='add (from flat)', timing_sensitive='ambiguous_pending', no_change='no_change', unclassified='unclassified')
MECH = ('inventory_unwind', 'directional', 'after_fill_burst')
MST = ('compatible', 'contradicted', 'timing_sensitive', 'insufficient_comparison', 'unidentified')
DIRCAT = ('with', 'against', 'flat', 'mixed', 'gap', 'missing')
NO_CLAIM = frozenset(('window_unbounded', 'receipt_before_uuid', 'no_life', 'life_window_mismatch', 'side_mixed'))
POOL_LEVELS = ('stratum', 'fallback1', 'fallback2', 'no_control', 'no_claim')
FEATS = (('burst3_certain', 'binary'), ('burst_possible_any', 'binary'), ('n_rec_1s', 'count'), ('n_rec_5s', 'count'), ('offset_sh_rec_5s', 'continuous'),
         ('r_1s_with', 'binary'), ('r_1s_against', 'binary'), ('r_5s_with', 'binary'), ('r_5s_against', 'binary'), ('r_5s_flat', 'binary'),
         ('r_5s_mixed', 'binary'), ('r_30s_with', 'binary'), ('r_30s_against', 'binary'), ('spread_start', 'continuous'),
         ('depth_taken_start', 'continuous'), ('tau_s', 'continuous'), ('pending_n', 'count'), ('reduction_if_taken', 'binary'))
CONTRAST_FEATS = tuple(f for f, _ in FEATS if f != 'reduction_if_taken')
LEG_COLS = ('sid', 'mk_idx', 'role', 'addr', 'hash', 'kind', 's', 'q', 'p', 'self_trade', 'settle_ms', 'status', 'hi_ms', 'lo_hard_ms', 'has_life')
ORD_COLS = ('scenario', 'mid', 'mk_hash', 'mk_signer', 'y_side', 'y_ptick', 'n_fills', 'cum_shares', 'first_hi_ms', 'last_hi_ms', 'post_lo_ms',
            'post_hi_ms', 'persist_lo_ms', 'persist_hi_ms', 'ahead_lo', 'ahead_hi', 'removal_set', 'removal_lo_ms', 'removal_hi_ms', 'removal_class',
            'removal_method', 'status', 'x_size_ub')
LF_COLS = ('mk_hash', 'sid', 'hi_ms', 'x_lo_hard_ms', 'x_rec_rank', 'shares', 'x_ahead_hi_S_FIFO')
LNK_COLS = ('mid', 'mk_signer', 'y_side', 'kind', 'prev_hash', 'next_hash', 'prev_ptick', 'next_ptick', 'prev_last_hi_ms', 'next_first_hi_ms',
            'gap_ms', 'x_latest', 'status')
SCEN = tuple(C.SCENARIOS)


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def hwm():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) / 1024.
    return 0.


def guard(where, lim=C.GUARD_MIB):
    h = hwm()
    need(h < lim, f'memory self-guard: VmHWM {h:.0f} MiB >= {lim:.0f} MiB at {where}')
    return h


def _sha(path):
    return X._sha(path)


# ------------------------------------------------------------------ the market-context bridge (chain_context, contract 7.0) ------------------------------------------------------------------
CC_API = ('load_market', 'btc_ret', 'book', 'level_size', 'position')


class Ctx:
    """The ONLY bridge to the market context: chain_context.py's API (or a selftest stub with the same surface). Every BTC / book / inventory
    value of this tool comes through these methods, vectorised over venue-ms instants."""

    def __init__(self, mod, root=None):
        for nm in CC_API:
            need(callable(getattr(mod, nm, None)), f'chain_context: {nm} missing (refused)')
        mt = getattr(mod, 'MOVE_THR', None)
        need(mt is None or all(abs(float(mt[k]) - v) < 1e-12 for k, v in MOVES), f'chain_context MOVE_THR {mt} differs from this tool\'s {MOVES}')
        pe = getattr(mod, 'POS_EDGES', None)
        need(pe is None or tuple(float(x) for x in pe) == POS_BINS, f'chain_context POS_EDGES {pe} differ from this tool\'s {POS_BINS}')
        self.mod, self.root, self.hash_cache = mod, root, {}
        self.name = f"{getattr(mod, '__name__', type(mod).__name__)} {getattr(mod, 'VERSION', '')}".strip()

    def market(self, day, mid):
        return self.mod.load_market(day, int(mid), TOP3, root=self.root, hash_cache=self.hash_cache)

    @staticmethod
    def times(M):
        return int(M.ws), int(M.we)

    def btc(self, M, x):
        """-> dict(code (0 = ok), r_1s, r_5s, r_30s (bp), pf (p_fair)); NaN unless code 0 (pf_mt_decision.btc_feats unchanged)."""
        r = self.mod.btc_ret(M, np.asarray(x, np.int64))
        return dict(code=np.asarray(r['code']), r_1s=np.asarray(r['r_1s'], float), r_5s=np.asarray(r['r_5s'], float), r_30s=np.asarray(r['r_30s'], float),
                    pf=np.asarray(r['p_fair'], float))

    def book(self, M, x):
        """-> dict(code (0 = book status ok: a fresh usable frame, both touches levels, not crossed), bid, ask (ticks), spread (ticks))."""
        b = self.mod.book(M, np.asarray(x, np.int64))
        ok = np.asarray(b['status'], dtype=object) == 'ok'
        return dict(code=np.where(ok, 0, 1).astype(np.int8), bid=np.where(ok, np.asarray(b['bid_tick'], float), NAN),
                    ask=np.where(ok, np.asarray(b['ask_tick'], float), NAN), spread=np.where(ok, np.asarray(b['spread'], float), NAN))

    def depth(self, M, x, side, tick):
        """-> shown shares at (side 0 bid / 1 ask, tick) in the frame book() uses (NaN unless its status is ok)."""
        x = np.asarray(x, np.int64)
        tk = np.clip(np.nan_to_num(np.asarray(tick, float), nan=-1., posinf=-1., neginf=-1.), -1, 101).astype(np.int64)
        return np.asarray(self.mod.level_size(M, x, np.broadcast_to(np.asarray(side, np.int64), x.shape), np.broadcast_to(tk, x.shape))['size'], float)

    def position(self, M, addr, x):
        p = self.mod.position(M, addr, np.asarray(x, np.int64))
        return np.asarray(p['recorded'], float), np.asarray(p['certain'], float)

    @staticmethod
    def rows(M, addr):
        """Every fills_valued row of the address in the market (chain_context M.legs ok + M.excluded not ok) with its lifecycle window."""
        a = addr.lower()
        parts = [M.legs.get(a), M.excluded.get(a)]
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            return None
        R = pd.concat(parts, ignore_index=True)
        for c in LEG_COLS:
            need(c in R.columns, f'chain_context legs lack {c} (refused)')
        return R


def make_ctx(mod=None, root=None):
    if mod is None:
        need((HERE / 'chain_context.py').is_file(), 'chain_context.py missing: every market-context feature comes from it (refused)')
        import chain_context as mod  # noqa: E402
    return Ctx(mod, root)


# ------------------------------------------------------------------ pure core: classes and mechanisms ------------------------------------------------------------------
def classify(before, dq, eps=POS_EPS):
    """Class codes (CLS) of a take of signed size dq at position `before` (actual |position|, contract section 5)."""
    b, d = np.broadcast_arrays(np.asarray(before, float), np.asarray(dq, float))
    a = b + d
    ab, aa = np.abs(b), np.abs(a)
    out = np.full(b.shape, CI['overshoot'], np.int8)
    opp = (np.sign(b) * np.sign(d) < 0) & (ab > eps)
    out[(ab > eps) & ~opp] = CI['add']
    out[ab <= eps] = CI['from_flat']
    out[opp & (aa <= eps)] = CI['full_close']
    out[opp & (aa > eps) & (np.sign(a) == np.sign(b))] = CI['reduce_partial']
    out[opp & (aa > eps) & (np.sign(a) == -np.sign(b)) & (aa < ab - eps)] = CI['reversal_reducing']
    out[(np.abs(d) <= eps) & (ab > eps)] = CI['no_change']
    out[~(np.isfinite(b) & np.isfinite(d))] = CI['unclassified']
    return out


RED_CODES = np.array([CI[c] for c in REDUCING], np.int8)


def combine(*codes):
    """Per take: the common class of the present variants (code -1 = absent), else timing_sensitive."""
    M_ = np.vstack([np.asarray(c, np.int16) for c in codes])
    pres = M_ >= 0
    first = np.where(pres.any(0), M_[np.argmax(pres, 0), np.arange(M_.shape[1])], CI['unclassified'])
    return np.where(np.all(~pres | (M_ == first), 0), first, CI['timing_sensitive']).astype(np.int8)


def mech_inventory(*codes):
    """inventory_unwind status codes (MST) from the variant class codes (-1 = absent)."""
    M_ = np.vstack([np.asarray(c, np.int16) for c in codes])
    pres = M_ >= 0
    any_p = pres.any(0)
    out = np.full(M_.shape[1], MST.index('timing_sensitive'), np.int8)
    out[(np.isin(M_, RED_CODES) | ~pres).all(0) & any_p] = MST.index('compatible')
    out[(np.isin(M_, (CI['add'], CI['overshoot'])) | ~pres).all(0) & any_p] = MST.index('contradicted')
    out[(np.isin(M_, (CI['from_flat'], CI['no_change'])) | ~pres).all(0) | ~any_p | (M_ == CI['unclassified']).any(0)] = MST.index('unidentified')
    return out


def dir_category(n_exp, n_ok, n_with, n_against):
    """Window category code (DIRCAT) from instant counts (chain_context.over_window statuses: missing / timing_sensitive (mixed) /
    partially_missing (gap) / stable (the category))."""
    n_exp, n_ok, n_with, n_against = (np.asarray(v, np.int64) for v in (n_exp, n_ok, n_with, n_against))
    flat = n_ok - n_with - n_against
    const = (n_with == n_ok) | (n_against == n_ok) | (flat == n_ok)
    cat = np.where(n_with == n_ok, 0, np.where(n_against == n_ok, 1, 2))
    return np.where(n_ok <= 0, DIRCAT.index('missing'), np.where(~const, DIRCAT.index('mixed'),
                    np.where(n_ok < n_exp, DIRCAT.index('gap'), cat))).astype(np.int8)


_DIR_MST = np.array([MST.index(m) for m in ('compatible', 'contradicted', 'insufficient_comparison', 'timing_sensitive', 'unidentified',
                                            'unidentified')], np.int8)


def mech_directional(cat):
    return _DIR_MST[np.asarray(cat, np.int64)]


def mech_burst(n_cert, n_poss):
    n_cert, n_poss = np.asarray(n_cert), np.asarray(n_poss)
    return np.where(n_cert >= BURST_N, MST.index('compatible'), np.where(n_poss == 0, MST.index('contradicted'),
                    MST.index('timing_sensitive'))).astype(np.int8)


def pos_bin(p):
    return np.searchsorted(np.asarray(POS_BINS), np.abs(np.asarray(p, float)), side='right').astype(np.int8)


def pos_sign(p):
    p = np.asarray(p, float)
    return np.where(p > POS_EPS, 1, np.where(p < -POS_EPS, -1, 0)).astype(np.int8)


# ------------------------------------------------------------------ pure core: range helpers ------------------------------------------------------------------
def count_in(sorted_t, lo, hi):
    """# of sorted_t values in [lo, hi) per query."""
    return (np.searchsorted(sorted_t, hi, 'left') - np.searchsorted(sorted_t, lo, 'left')).astype(np.int64)


def _cs(v):
    return np.r_[0., np.cumsum(np.asarray(v, float))]


def range_sum(cum, i0, i1):
    """sum over grid indices [i0, i1] (inclusive, clipped to the grid) from a leading-0 cumulative array; empty ranges give 0."""
    n = len(cum) - 1
    i0c, i1c = np.clip(i0, 0, n), np.clip(np.asarray(i1) + 1, 0, n)
    return np.where(i1c > i0c, cum[np.maximum(i1c, i0c)] - cum[i0c], 0.)


def range_reduce(a, i0, i1, ufunc):
    """ufunc (np.fmin / np.fmax: NaN-ignoring) over a[i0..i1] inclusive (clipped) per query; NaN for an empty range."""
    n = len(a)
    if n == 0 or len(i0) == 0:
        return np.full(len(i0), NAN)
    lo, hi = np.clip(i0, 0, n), np.clip(np.asarray(i1) + 1, 0, n)
    ok = hi > lo
    ext = np.r_[np.asarray(a, float), NAN]
    idx = np.empty(2 * len(lo), np.int64)
    idx[0::2], idx[1::2] = np.where(ok, lo, n), np.where(ok, hi, n)
    return np.where(ok, ufunc.reduceat(ext, idx)[0::2], NAN)


# ------------------------------------------------------------------ inputs ------------------------------------------------------------------
def population(ident, manifest, hashes):
    addrs = X.load_top10(ident)[:3]
    need(addrs == TOP3, 'identities: ex_ante ranks 1-3 differ from the contract top 3 (refused)')
    hashes['econ/identities.json'] = _sha(ident)
    book = E.load_book(manifest)
    hashes['manifest'] = _sha(manifest)
    need(set(book) <= set(C.MEASURE_DAYS), f'BOOK manifest: a day outside the measurement days {sorted(set(book) - set(C.MEASURE_DAYS))} (refused)')
    need(sum(len(v) for v in book.values()) == N_BOOK, f'BOOK manifest: {sum(len(v) for v in book.values())} markets, not {N_BOOK} (refused)')
    return addrs, book


def links_path(inp, day, mid):
    need(day in C.MEASURE_DAYS, f'lifecycle day {day} is not a measurement day (never opened)')
    return Path(inp) / 'life' / f'{day}_{int(mid)}_x_links.parquet'


def read_links(inp, day, mid, hashes):
    p = links_path(inp, day, mid)
    need(p.is_file(), f'life x_links: input missing: {p.name}')
    sch = pq.read_schema(p)
    for c in LNK_COLS:
        need(c in sch.names, f'life x_links: required column {c} missing ({p.name})')
    T = pq.read_table(p, columns=list(LNK_COLS)).to_pandas()
    low = {a.lower() for a in TOP3}
    T = T[T['mk_signer'].astype(str).str.lower().isin(low).to_numpy()].reset_index(drop=True)
    hashes[f'life/{p.name}'] = _sha(p)
    return T


def market_tables(M):
    """The lifecycle orders (top-3 signers, every scenario row) and the fills of those orders, from chain_context's M."""
    O = M.orders
    if O is None or not len(O):
        return None, None
    for c in ORD_COLS:
        need(c in O.columns, f'lifecycle orders lack {c} (refused)')
    low = {a.lower() for a in TOP3}
    O = O[O['mk_signer'].astype(str).str.lower().isin(low).to_numpy()][list(ORD_COLS)].reset_index(drop=True)
    for c in O.columns:
        if isinstance(O[c].dtype, pd.CategoricalDtype):
            O[c] = O[c].astype(str)
    sp = {}
    for x in O['mk_signer'].astype(str).unique():
        sp.setdefault(x.lower(), set()).add(x)
    need(all(len(v) == 1 for v in sp.values()), 'lifecycle orders: a top-3 address under two spellings (refused)')
    LF = M.life_fills
    for c in LF_COLS:
        need(c in LF.columns, f'lifecycle fills lack {c} (refused)')
    LF = LF[LF['mk_hash'].isin(set(O['mk_hash']))][list(LF_COLS)].reset_index(drop=True)
    return O, LF


# ------------------------------------------------------------------ one address in one market: legs, actions, takes ------------------------------------------------------------------
def pair_inputs(R):
    """Every fills_valued row R of ONE address in ONE market (chain_context legs: every status, with the lifecycle window) -> legs (ok,
    non-self), actions, takes (every status), the settle times of excluded own rows and the active span."""
    R = R.reset_index(drop=True)
    ok = (R['status'].astype(str) == OK).to_numpy()
    selft = R['self_trade'].fillna(False).to_numpy(bool)
    tkr = (R['role'].astype(str) == 'taker').to_numpy()
    settle, hi, lh = (R[c].to_numpy(np.int64) for c in ('settle_ms', 'hi_ms', 'lo_hard_ms'))
    has = R['has_life'].to_numpy(bool)
    sid = R['sid'].astype(str).to_numpy(object)
    s, q, p = R['s'].to_numpy(np.int64), R['q'].to_numpy(float), R['p'].to_numpy(float)
    lg = ok & ~selft
    need(bool(((lh[lg] < 0) | (hi[lg] - lh[lg] == MAX_LAG)).all()), 'a status-ok leg whose window is neither unbounded nor MAX_LAG_MS wide (refused)')
    legs = dict(sid=sid[lg], role=tkr[lg].astype(np.int8), s=s[lg], q=q[lg], dq=(s * q)[lg], settle=settle[lg], hi=hi[lg], lh=lh[lg], has_life=has[lg])
    A = pd.DataFrame({'sid': legs['sid'], 'role': legs['role'], 'dq': legs['dq'], 'settle': legs['settle']}).groupby(
        ['sid', 'role'], sort=False).agg(dq=('dq', 'sum'), settle=('settle', 'first')).reset_index()
    A = A.sort_values(['settle', 'sid', 'role'], kind='mergesort').reset_index(drop=True)
    excl_settle = np.sort(settle[~ok & (settle > 0)])
    T = pd.DataFrame({'sid': sid, 's': s, 'q': q, 'qp': q * p, 'p': p, 'self': selft, 'ok': ok, 'settle': settle, 'hash': R['hash'].astype(str).to_numpy(object),
                      'kind': R['kind'].astype(str).to_numpy(object), 'st': R['status'].astype(str).to_numpy(object), 'hi': hi, 'lh': lh, 'has': has})[tkr]
    cols = (('sid', 'object'), ('s_min', 'int64'), ('s_max', 'int64'), ('q_all', 'float64'), ('q_self', 'float64'), ('qp', 'float64'),
            ('p_min', 'float64'), ('p_max', 'float64'), ('n_legs', 'int64'), ('n_px', 'int64'), ('settle', 'int64'), ('settle2', 'int64'),
            ('tk_hash', 'object'), ('n_hash', 'int64'), ('kinds', 'object'), ('n_ok', 'int64'), ('sts', 'object'), ('hi', 'int64'), ('hi2', 'int64'),
            ('lh', 'int64'), ('lh2', 'int64'), ('has', 'bool'), ('self', 'bool'))
    if len(T):
        T = T.assign(q_self=np.where(T['self'], T['q'], 0.), pr=np.round(T['p'], 6))
        g = T.groupby('sid', sort=False)
        K = pd.DataFrame({'s_min': g['s'].min(), 's_max': g['s'].max(), 'q_all': g['q'].sum(), 'q_self': g['q_self'].sum(), 'qp': g['qp'].sum(),
                          'p_min': g['p'].min(), 'p_max': g['p'].max(), 'n_legs': g.size(), 'n_px': g['pr'].nunique(), 'settle': g['settle'].min(),
                          'settle2': g['settle'].max(), 'tk_hash': g['hash'].first(), 'n_hash': g['hash'].nunique(),
                          'kinds': g['kind'].agg(lambda v: '+'.join(sorted(set(v)))), 'n_ok': g['ok'].sum(),
                          'sts': g['st'].agg(lambda v: '+'.join(sorted(set(v) - {OK}))), 'hi': g['hi'].min(), 'hi2': g['hi'].max(), 'lh': g['lh'].min(),
                          'lh2': g['lh'].max(), 'has': g['has'].all(), 'self': g['self'].any()}).reset_index()
        K = K[[c for c, _ in cols]]
    else:
        K = pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols})
    K = K.sort_values(['settle', 'sid'], kind='mergesort').reset_index(drop=True)
    K['n_bad'] = K['n_legs'].to_numpy(np.int64) - K['n_ok'].to_numpy(np.int64)
    K['cons'] = (K['hi'].to_numpy() == K['hi2'].to_numpy()) & (K['lh'].to_numpy() == K['lh2'].to_numpy())
    pos = settle[settle > 0]
    span = (int(pos.min()), int(pos.max())) if len(pos) else None
    return legs, A, K, excl_settle, span


def take_core(legs, A, K, excl_settle, pos_fn=None):
    """Per take: dq, the inventory variants (recorded / simultaneous / certain(D)), classes, mechanisms, pending, preceding maker fill and
    burst counts. pos_fn(x) -> (recorded, certain) = chain_context.position (None: this tool's own sums, the selftest's reference)."""
    n = len(K)
    out = dict(n=n)
    if n == 0:
        return out
    s = K['s_min'].to_numpy(np.int64)
    side_mixed = (K['s_min'].to_numpy() != K['s_max'].to_numpy()) | (K['n_hash'].to_numpy() != 1) | (K['settle'].to_numpy() != K['settle2'].to_numpy())
    q_net = K['q_all'].to_numpy(float) - K['q_self'].to_numpy(float)
    dq = s * q_net
    settle, hi, lh = K['settle'].to_numpy(np.int64), K['hi'].to_numpy(np.int64), K['lh'].to_numpy(np.int64)
    sidk = K['sid'].astype(str).to_numpy(object)
    a_set, a_dq, a_sid = A['settle'].to_numpy(np.int64), A['dq'].to_numpy(float), A['sid'].astype(str).to_numpy(object)
    a_cs = _cs(a_dq)
    i_lt, i_le = np.searchsorted(a_set, settle, 'left'), np.searchsorted(a_set, settle, 'right')
    rec_own = a_cs[i_lt]
    tie, n_tie = np.zeros(n), np.zeros(n, np.int64)
    for k in np.flatnonzero(i_le > i_lt):                 # own actions at the take's settle ms other than the take itself
        sl = slice(i_lt[k], i_le[k])
        other = a_sid[sl] != sidk[k]
        tie[k], n_tie[k] = float(a_dq[sl][other].sum()), int(other.sum())
    oh = np.argsort(legs['hi'], kind='mergesort')
    h_s, h_cs = legs['hi'][oh], _cs(legs['dq'][oh])
    own_cert = lambda x: h_cs[np.searchsorted(h_s, x, 'left')]  # noqa: E731   legs with hi < x
    if pos_fn is None:
        rec = rec_own
    else:
        rec = pos_fn(settle)[0]
        dev = float(np.max(np.abs(rec - rec_own))) if n else 0.
        need(dev <= REC_TOL, f'chain_context.position recorded differs from the own action sum by {dev:.3g} sh (refused)')
    c_rec = classify(rec, dq)
    c_sim = np.where(n_tie > 0, classify(rec + tie, dq), -1).astype(np.int16)
    out.update(s=s, dq=dq, q_net=q_net, settle=settle, hi=hi, lh=lh, rec=rec, tie=tie, n_tie=n_tie, side_mixed=side_mixed, c_rec=c_rec, c_sim=c_sim)
    bounded = lh >= 0
    for D in D_PROFILES:
        lo = np.where(bounded, lh - D, settle - MAX_LAG - D)
        if pos_fn is None:
            cert = own_cert(lo)
        else:
            cert = pos_fn(lo)[1]
            dev = float(np.max(np.abs(cert - own_cert(lo))))
            need(dev <= REC_TOL, f'chain_context.position certain differs from the own sum by {dev:.3g} sh (refused)')
        cert = np.where(bounded, cert, NAN)
        c_cert = np.where(bounded, classify(np.nan_to_num(cert), dq), -1).astype(np.int16)
        out[f'cert_{D}'], out[f'c_cert_{D}'] = cert, c_cert
        out[f'cls_{D}'] = combine(c_rec, c_cert, c_sim)
        out[f'm_inv_{D}'] = mech_inventory(c_rec, c_cert, c_sim)
        out[f'pend_n_{D}'], out[f'pend_dq_{D}'], out[f'pend_abs_{D}'] = pending_counts(legs, lo, settle, True, sidk)
    mk = legs['role'] == 0
    om = np.argsort(legs['settle'][mk], kind='mergesort')
    m_set, m_hi, m_lh, m_s, m_q = (legs[k][mk][om] for k in ('settle', 'hi', 'lh', 's', 'q'))
    j = np.searchsorted(m_set, settle, 'left') - 1
    has_prev = j >= 0
    jc = np.maximum(j, 0)
    p_hi = np.where(has_prev, m_hi[jc] if len(m_hi) else 0, -1)
    p_lh = np.where(has_prev, m_lh[jc] if len(m_lh) else -1, -1)
    out['prev_delay_lo'] = np.where(has_prev & bounded, np.maximum(0, lh - p_hi), NAN)
    out['prev_delay_hi'] = np.where(has_prev & (p_lh >= 0), np.maximum(0, hi - p_lh), NAN)
    out['prev_settle_gap'] = np.where(has_prev, settle - (m_set[jc] if len(m_set) else 0), NAN)
    for D in D_PROFILES:
        lo = np.where(bounded, lh - D, settle - MAX_LAG - D)
        for k_, v_ in burst_counts(m_set, m_hi, m_lh, m_s, m_q, lo, s, settle).items():
            out[f'{k_}_{D}'] = v_
    out['n_excl_before'] = np.searchsorted(excl_settle, settle, 'right').astype(np.int64)
    return out


def burst_counts(m_set, m_hi, m_lh, m_s, m_q, lo, s, ref):
    """Own maker legs (settle-sorted) before a window start lo: certain (lo_hard >= lo - 5 s and hi < lo) / possible (window meets
    [lo - 5 s, lo); sequenced at or before ref = the take's settle / the control instant - 1) counts, recorded counts settle in
    [lo - 1 s, lo) / [lo - 5 s, lo) and the offsetting shares (s_fill = -s) in [lo - 5 s, lo)."""
    lo, s, ref = np.asarray(lo, np.int64), np.asarray(s, np.int64), np.asarray(ref, np.int64)
    bnd = m_lh >= 0
    hb, lb = np.sort(m_hi[bnd]), np.sort(m_lh[bnd])
    su = np.sort(m_hi[~bnd])                         # unbounded (no UUID): its receipt hi
    # certain: bounded with lo_hard >= lo - 5 s and hi < lo; bounded windows are exactly MAX_LAG wide (pair_inputs), so lo_hard >= lo - 5 s
    # <=> hi >= lo - 5 s + MAX_LAG
    cert = count_in(hb, lo - BURST_MS + MAX_LAG, lo)
    # possible: hi >= lo - 5 s and lo_hard < lo: bounded lo_hard in [lo - 5 s - MAX_LAG, lo) (its hi < lo + MAX_LAG <= ref); unbounded
    # (lo_hard -1: no lower end) hi >= lo - 5 s, sequenced at or before ref (settle in [lo - 5 s, ref]; hi <= settle)
    poss = count_in(lb, lo - BURST_MS - MAX_LAG, lo) + count_in(su, lo - BURST_MS, ref + 1)
    r1 = count_in(m_set, lo - REC_MS[0], lo)
    r5 = count_in(m_set, lo - REC_MS[1], lo)
    off = np.zeros(len(lo))
    for sd in (1, -1):
        mm = m_s == sd
        st_, cs_ = m_set[mm], _cs(m_q[mm])
        sel = s == -sd                                   # a take of side s offsets the own maker fills of side -s
        if sel.any():
            off[sel] = cs_[np.searchsorted(st_, lo[sel], 'left')] - cs_[np.searchsorted(st_, lo[sel] - REC_MS[1], 'left')]
    return dict(n_cert=cert, n_poss=poss, n_rec_1s=r1, n_rec_5s=r5, offset_sh_rec_5s=off)


def pending_counts(legs, lo, ref, inclusive, own_sid=None):
    """Own legs (ok, non-self) sequenced before ref (settle < ref; settle <= ref and another sid when inclusive) whose window [lo_hard, hi_leg]
    meets the decision window [lo, ref']: hi_leg >= lo (a leg sequenced before ref has hi_leg <= settle <= ref, so its lower end lies before
    the window's upper end; receipt-skewed takes are NO_CLAIM). Exact range counts: #(settle <= ref) - #(hi < lo) + #(hi < lo and settle >
    ref) - the take's own legs, the third term only possible for a leg with settle - hi > MAX_LAG (counted directly). -> (count, dq, |dq|)."""
    lo, ref = np.asarray(lo, np.int64), np.asarray(ref, np.int64)
    st, hi, dq, sid = legs['settle'], legs['hi'], legs['dq'], legs['sid']
    os_, oh = np.argsort(st, kind='mergesort'), np.argsort(hi, kind='mergesort')
    ss, hs = st[os_], hi[oh]
    side = 'right' if inclusive else 'left'
    out = []
    for v in (np.ones(len(dq)), dq, np.abs(dq)):
        cs_s, cs_h = _cs(v[os_]), _cs(v[oh])
        out.append(cs_s[np.searchsorted(ss, ref, side)] - cs_h[np.searchsorted(hs, lo, 'left')])
    rare = np.flatnonzero(st - hi > MAX_LAG)
    for j in rare:                                       # hi < lo and settle beyond ref: subtracted above but never pending
        m_ = (hi[j] < lo) & ((st[j] > ref) if inclusive else (st[j] >= ref))
        for i_, v in enumerate((1., dq[j], abs(dq[j]))):
            out[i_] = out[i_] + np.where(m_, v, 0.)
    if own_sid is not None and len(sid):                 # the take's own legs (settle = ref): remove exactly those the terms above counted
        by = {}
        for j, x in enumerate(sid):
            by.setdefault(x, []).append(j)
        for k, x in enumerate(own_sid):
            for j in by.get(x, ()):
                if hi[j] >= lo[k] and (st[j] <= ref[k] if inclusive else st[j] < ref[k]):
                    for i_, v in enumerate((1., dq[j], abs(dq[j]))):
                        out[i_][k] -= v
    return np.rint(out[0]).astype(np.int64), out[1], out[2]


# ------------------------------------------------------------------ the market grid and window reductions ------------------------------------------------------------------
def market_grid(ctx, M, lo_ms, hi_ms, cnt):
    """Per-instant venue context on the 100 ms grid [lo_ms, hi_ms] through the bridge -> arrays and leading-0 cumulative counts."""
    G0 = (int(lo_ms) // STEP) * STEP
    n = max(1, (int(hi_ms) - G0) // STEP + 1)
    g = G0 + STEP * np.arange(n, dtype=np.int64)
    B, K = ctx.btc(M, g), ctx.book(M, g)
    bok, kok = B['code'] == 0, K['code'] == 0
    G = dict(G0=G0, n=n, g=g, pf=np.where(bok, B['pf'], NAN), spread=np.where(kok, K['spread'], NAN), cum_ok=_cs(bok))
    for nm, thr in MOVES:
        r = np.where(bok, B[nm], NAN)
        G[f'cum_up_{nm}'], G[f'cum_dn_{nm}'] = _cs(bok & (r >= thr)), _cs(bok & (r <= -thr))
    cnt['grid_instants'] += n
    cnt['grid_btc_not_ok'] += int((~bok).sum())
    cnt['grid_book_not_ok'] += int((~kok).sum())
    return G


def window_feats(ctx, M, G, lo, hi, s):
    """Over the instants of [lo, hi] (both ends + every grid instant strictly inside: chain_context.over_window's set): the counts and
    window category (DIRCAT) of every move for side s, p_fair and spread min / max."""
    lo, hi, s = np.asarray(lo, np.int64), np.asarray(hi, np.int64) + _MUT_LEAK_MS, np.asarray(s, np.int64)
    i0 = (lo - G['G0']) // STEP + 1                       # first grid instant strictly after lo
    i1 = -((G['G0'] - hi) // STEP) - 1                     # last grid instant strictly before hi: ceil((hi - G0) / STEP) - 1
    n_in = np.maximum(i1 - i0 + 1, 0)
    n_exp = n_in + 2
    Bl, Bh = ctx.btc(M, lo), ctx.btc(M, hi)
    okl, okh = Bl['code'] == 0, Bh['code'] == 0
    n_ok = np.rint(range_sum(G['cum_ok'], i0, i1)).astype(np.int64) + okl + okh
    out = dict(n_exp=n_exp, n_ok=n_ok)
    for nm, thr in MOVES:
        up = np.rint(range_sum(G[f'cum_up_{nm}'], i0, i1)).astype(np.int64) + (okl & (Bl[nm] >= thr)) + (okh & (Bh[nm] >= thr))
        dn = np.rint(range_sum(G[f'cum_dn_{nm}'], i0, i1)).astype(np.int64) + (okl & (Bl[nm] <= -thr)) + (okh & (Bh[nm] <= -thr))
        out[f'cat_{nm}'] = dir_category(n_exp, n_ok, np.where(s > 0, up, dn), np.where(s > 0, dn, up))
    pl, ph = np.where(okl, Bl['pf'], NAN), np.where(okh, Bh['pf'], NAN)
    out['pf_min'] = np.fmin(range_reduce(G['pf'], i0, i1, np.fmin), np.fmin(pl, ph))
    out['pf_max'] = np.fmax(range_reduce(G['pf'], i0, i1, np.fmax), np.fmax(pl, ph))
    Kl, Kh = ctx.book(M, lo), ctx.book(M, hi)
    out['spread_min'] = np.fmin(range_reduce(G['spread'], i0, i1, np.fmin), np.fmin(Kl['spread'], Kh['spread']))
    out['spread_max'] = np.fmax(range_reduce(G['spread'], i0, i1, np.fmax), np.fmax(Kl['spread'], Kh['spread']))
    return out


def start_state(ctx, M, x, s):
    """The last usable frame <= x (the window start): taken-side touch (buy -> ask, sell -> bid), spread, depth at that touch."""
    K = ctx.book(M, x)
    ok = K['code'] == 0
    s = np.asarray(s)
    touch = np.where(ok, np.where(s > 0, K['ask'], K['bid']), NAN)
    dep = ctx.depth(M, x, np.where(s > 0, 1, 0), touch)
    return dict(touch=touch, spread=np.where(ok, K['spread'], NAN), depth=np.where(ok & np.isfinite(touch), dep, NAN))


def feats_of(W, b, st, tau, pend_n):
    """The compared features (FEATS) from window categories W, burst counts b, the start state st, tau and the pending count: one
    definition for takes and controls."""
    f = dict(burst3_certain=(b['n_cert'] >= BURST_N).astype(float), burst_possible_any=(b['n_poss'] > 0).astype(float),
             n_rec_1s=np.asarray(b['n_rec_1s'], float), n_rec_5s=np.asarray(b['n_rec_5s'], float), offset_sh_rec_5s=np.asarray(b['offset_sh_rec_5s'], float),
             spread_start=st['spread'], depth_taken_start=st['depth'], tau_s=np.asarray(tau, float), pending_n=np.asarray(pend_n, float))
    for nm, _ in MOVES:
        f[f'{nm}_with'] = (W[f'cat_{nm}'] == DIRCAT.index('with')).astype(float)
        f[f'{nm}_against'] = (W[f'cat_{nm}'] == DIRCAT.index('against')).astype(float)
    f['r_5s_flat'] = (W['cat_r_5s'] == DIRCAT.index('flat')).astype(float)
    f['r_5s_mixed'] = (W['cat_r_5s'] == DIRCAT.index('mixed')).astype(float)
    return f


# ------------------------------------------------------------------ controls ------------------------------------------------------------------
def candidates(G, span, take_settle, ws, we, A):
    """Control instants of one pair: grid instants in the active span ([first, last own leg settle] clipped to [ws - 300 s, we)) with no own
    take record settle within +- EXCL_MS -> dict(c ms, phase, sign, bin, pos) or None."""
    if span is None:
        return None
    lo, hi = max(span[0], ws - PRE_MS), min(span[1], we - 1)
    if hi < lo:
        return None
    c = G['g'][(G['g'] >= lo) & (G['g'] <= hi)]
    near = count_in(np.sort(np.asarray(take_settle, np.int64)), c - EXCL_MS, c + EXCL_MS + 1) > 0
    c = c[~near]
    pos = _cs(A['dq'].to_numpy(float))[np.searchsorted(A['settle'].to_numpy(np.int64), c, 'left')]
    return dict(c=c, phase=(c >= ws).astype(np.int8), sign=pos_sign(pos), bin=pos_bin(pos), pos=pos, n_near=int(near.sum()))


def control_feats(ctx, M, G, cand, legs, D, we):
    """The compared features of the control windows [c - MAX_LAG - D, c] for both sides -> {+1: {feature: array}, -1: {...}}."""
    c = cand['c']
    lo = c - MAX_LAG - D
    mk = legs['role'] == 0
    om = np.argsort(legs['settle'][mk], kind='mergesort')
    m_set, m_hi, m_lh, m_s, m_q = (legs[k][mk][om] for k in ('settle', 'hi', 'lh', 's', 'q'))
    pend = pending_counts(legs, lo, c, False)[0].astype(float)
    K = ctx.book(M, lo)
    ok = K['code'] == 0
    out = {}
    for sd in (1, -1):
        sv = np.full(len(c), sd, np.int64)
        W = window_feats(ctx, M, G, lo, c, sv)
        b = burst_counts(m_set, m_hi, m_lh, m_s, m_q, lo, sv, c - 1)
        touch = np.where(ok, K['ask'] if sd > 0 else K['bid'], NAN)
        dep = ctx.depth(M, lo, np.full(len(c), 1 if sd > 0 else 0), touch)
        st = dict(spread=np.where(ok, K['spread'], NAN), depth=np.where(ok & np.isfinite(touch), dep, NAN))
        out[sd] = feats_of(W, b, st, (we - c) / 1000., pend)
    return out


def pools(cand, t_phase, t_sign, t_bin):
    """Per take: candidate indices of its stratum (phase, sign, bin), else fallback1 (phase, bin), else fallback2 (phase); level 0 / 1 / 2,
    3 = no control."""
    n = len(t_phase)
    res, lvl = [None] * n, np.full(n, 3, np.int8)
    if cand is None or len(cand['c']) == 0:
        return res, lvl
    ph = cand['phase'].astype(np.int64)
    ckeys = [ph * 100 + (cand['sign'].astype(np.int64) + 1) * 10 + cand['bin'], ph * 100 + cand['bin'], ph]
    tp, ts_, tb = (np.asarray(v, np.int64) for v in (t_phase, t_sign, t_bin))
    tkeys = [tp * 100 + (ts_ + 1) * 10 + tb, tp * 100 + tb, tp]
    idx = []
    for kk in ckeys:
        o = np.argsort(kk, kind='mergesort')
        u, st = np.unique(kk[o], return_index=True)
        idx.append({int(a): o[b:e] for a, b, e in zip(u, st, np.r_[st[1:], len(o)])})
    for j in range(n):
        for L_ in range(3):
            p = idx[L_].get(int(tkeys[L_][j]))
            if p is not None and len(p):
                res[j], lvl[j] = p, L_
                break
    return res, lvl


def draw(pool_list, rng, R):
    """One pooled instant per take per draw -> int64 [n, R] of candidate indices (-1 = no pool); takes drawn in their fixed order."""
    out = np.full((len(pool_list), R), -1, np.int64)
    for j, p in enumerate(pool_list):
        if p is not None and len(p):
            out[j] = p[rng.integers(0, len(p), size=R)]
    return out


def null_stats(obs, null):
    """obs and the null replicates -> null mean, p05 / p50 / p95, p_hi = (1 + #null >= obs) / (1 + n), p_lo (contract 7)."""
    v = np.asarray(null, float)
    v = v[np.isfinite(v)]
    if not np.isfinite(obs) or len(v) == 0:
        return dict(null_mean=NAN, p05=NAN, p50=NAN, p95=NAN, p_hi=NAN, p_lo=NAN, n_draws=int(len(v)))
    q = np.quantile(v, [.05, .5, .95])
    return dict(null_mean=float(v.mean()), p05=float(q[0]), p50=float(q[1]), p95=float(q[2]), p_hi=float((1 + (v >= obs).sum()) / (1 + len(v))),
                p_lo=float((1 + (v <= obs).sum()) / (1 + len(v))), n_draws=int(len(v)))


# ------------------------------------------------------------------ actions ------------------------------------------------------------------
def order_rows(O, LF):
    """One row dict per own order hash (its three scenario rows + its lifecycle fills): the fields order_estimator.link_hypotheses reads
    (its caller spelling: post_lo_union, post_hi, first_hi, last_lh = removal_lo, removal_lo, end_hi, ex_any / pl_any / ec_any, tick,
    records) and the two-sided test's (certain-presence span, fill windows)."""
    if O is None or not len(O):
        return []
    fb = {h: g for h, g in LF.groupby('mk_hash', sort=False)} if LF is not None and len(LF) else {}
    rows = []
    for h, G_ in O.groupby('mk_hash', sort=True):
        by = {r['scenario']: r for r in G_.to_dict('records')}
        need(set(by) == set(SCEN) and len(G_) == len(SCEN), f'life orders: hash {h} lacks its three scenario rows (refused)')
        b = by['S_FIFO']
        cum = float(b['cum_shares'])
        ex, pl, ec = {}, {}, {}
        for sc, r in by.items():
            toks = set(str(r['removal_set']).split('|'))
            ub = float(r['x_size_ub'])
            room = (not np.isfinite(ub)) or (ub - cum > TOL_SH)
            ex[sc], pl[sc], ec[sc] = 'exhausted' in toks, ('cancelled_or_expired' in toks) and room, ('end_censored' in toks) and room
        plo = [int(by[sc]['post_lo_ms']) for sc in SCEN]
        rhi = [int(by[sc]['removal_hi_ms']) for sc in SCEN]
        f = fb.get(h)
        rows.append(dict(mk_hash=h, side=0 if str(b['y_side']) == 'bid' else 1, tick=int(b['y_ptick']), n_fills=int(b['n_fills']), cum=cum,
                         first_hi=int(b['first_hi_ms']), last_hi=int(b['last_hi_ms']), post_lo_union=-1 if min(plo) < 0 else min(plo),
                         post_hi=int(b['post_hi_ms']), removal_lo=int(b['removal_lo_ms']), last_lh=int(b['removal_lo_ms']),
                         end_hi=-1 if min(rhi) < 0 else max(rhi), ex_any=any(ex.values()), pl_any=any(pl.values()), ec_any=any(ec.values()),
                         persist_lo=int(b['persist_lo_ms']), persist_hi=int(b['persist_hi_ms']), status=str(b['status']),
                         records=frozenset(f['sid'].astype(str)) if f is not None else frozenset(),
                         f_lh=f['x_lo_hard_ms'].to_numpy(np.int64) if f is not None else np.zeros(0, np.int64),
                         f_hi=f['hi_ms'].to_numpy(np.int64) if f is not None else np.zeros(0, np.int64)))
    return rows


def two_sided_certain(x, y):
    """Contract 7.3: certain when a fill window of the opposite-side order lies inside the other order's certain-presence span (either way)."""
    for a, b in ((x, y), (y, x)):
        if a['persist_lo'] < 0 or a['persist_hi'] < a['persist_lo'] or not len(b['f_hi']):
            continue
        if bool(((b['f_lh'] >= 0) & (b['f_lh'] >= a['persist_lo']) & (b['f_hi'] <= a['persist_hi'])).any()):
            return True
    return False


def feasible_overlap(a, b, ws, we):
    """Feasible lives [post_lo_union (open -> ws - 300 s), end_hi (open -> we)] intersect."""
    la = a['post_lo_union'] if a['post_lo_union'] >= 0 else ws - PRE_MS
    lb = b['post_lo_union'] if b['post_lo_union'] >= 0 else ws - PRE_MS
    ea = a['end_hi'] if a['end_hi'] >= 0 else we
    eb = b['end_hi'] if b['end_hi'] >= 0 else we
    return max(la, lb) <= min(ea, eb)


LOGIC = {   # (state/change, alternatives, proposed trade-off, observed action, competing explanation, what would falsify) - declared text
    'maker_fill': ('a resting own order at its price', 'keep / cancel / reprice / take', 'earn the spread while bearing adverse selection',
                   'a passive fill (outcome-selected)', 'none declared: the descriptive base type', 'markouts and queue analysis (chain_post_fill)'),
    'post_new_level': ('no earlier filled own order at this side and tick', 'join an existing own level / wait / take',
                       'queue priority at a new price before demand reaches it vs adverse selection there', 'first evidence = its first fill (never the command)',
                       'late post, reaction to a price change, inventory offset', 'chain_post_fill POST mechanisms (posting window vs touch / BTC)'),
    'post_same_level': ('an earlier filled own order at this side and tick', 'leave the level / reprice / wait', 'keep presence at a known price',
                        'first evidence = its first fill', 'a second resting (ladder) order vs a replacement', 'link_hypotheses on the x_links pair'),
    'reprice': ('an earlier own order at another tick on the same side', 'keep the old price / cancel / take', 'follow the market to keep queue value',
                'next order first filled at another tick', 'overlapping ladder order, replenishment, unrelated order', 'overlap_identified contradicts a replacement'),
    'refill': ('an earlier own order at the same tick and side', 'leave / reprice / take', 'replenish the level after exhaustion or a pull',
               'next order first filled at the same tick', 'a second resting order, unrelated order', 'overlap_identified (the next order rested meanwhile)'),
    'ladder_overlap': ('two own orders on one side at different ticks', 'a single order / pull one', 'hold priority at several prices at once',
                       'certain co-presence (link_hypotheses overlap_certain)', 'the purpose is open; the co-presence itself is observed', 'no certain co-presence'),
    'keep_after_fill': ('an own order just filled (partially)', 'cancel the remainder / reprice / keep', 'retain queue priority for later fills vs holding risk',
                        'the same hash fills again later (observed persistence)', 'no time to react (feasible gap <= 1 s) is equally compatible',
                        'nothing falsifies the persistence; only the motive is open'),
    'pull_possible': ('a resting own order whose remainder may have been withdrawn', 'keep / reprice', 'avoid adverse fills vs lose queue value',
                      'disappearance compatible with a cancel', 'exhaustion or expiry (EX / EC feasible)', 'EX and EC infeasible in every scenario'),
    'take': ('own inventory and the market over the decision window', 'wait / quote the reducing side passively / take',
             'pay spread and fee to reduce exposure (inventory_unwind) or capture a move (directional)', 'an aggressive record (complete public record)',
             'inventory_unwind vs directional vs after_fill_burst (non-exclusive)',
             'add / overshoot in every variant contradicts unwind; BTC against at every instant contradicts directional'),
    'two_sided': ('own orders on both sides', 'one-sided quoting', 'earn the spread on both sides / offset inventory with the other side',
                  'an opposite-side order with overlapping feasible life', 'sequential one-sided quoting within the timing ambiguity', 'no certain co-presence'),
}


def logic_key(t):
    if t.startswith('take'):
        return 'take'
    if t.startswith('pull_possible'):
        return 'pull_possible'
    return t if t in LOGIC else 'maker_fill'


def next_events(legs, rows, ref, sv):
    """The first own event with settle strictly after ref: a maker leg (same / opposite side as sv) or a take record (reducing / adding /
    other by take_class)."""
    mk = legs['role'] == 0
    t, side, code = [legs['settle'][mk]], [np.where(legs['s'][mk] > 0, 1, -1)], [np.zeros(int(mk.sum()), np.int64)]
    if rows is not None and len(rows['sid']):
        c = rows['take_class']
        t.append(rows['settle_ms'])
        side.append(np.zeros(len(c), np.int64))
        code.append(np.where(np.isin(c, REDUCING), 1, np.where(np.isin(c, ADDING), 2, 3)))
    t, side, code = np.concatenate(t), np.concatenate(side), np.concatenate(code)
    o = np.argsort(t, kind='mergesort')
    t, side, code = t[o], side[o], code[o]
    out = np.full(len(ref), 'none', object)
    if not len(t):
        return out
    j = np.searchsorted(t, ref, 'right')
    has = j < len(t)
    jc = np.minimum(j, len(t) - 1)
    mkr = has & (code[jc] == 0)
    out[mkr & (side[jc] == sv)] = 'maker_same'
    out[mkr & (side[jc] != sv)] = 'maker_opp'
    out[has & (code[jc] == 1)] = 'take_reducing'
    out[has & (code[jc] == 2)] = 'take_adding'
    out[has & (code[jc] == 3)] = 'take_other'
    return out


# ------------------------------------------------------------------ the build ------------------------------------------------------------------
class Build:
    """Per-pair results of a run (or pilot): take rows, control draws and features, action events; then the tables."""

    def __init__(self, ctx, link_fn, pull_fn, cnt, use_cc_position=True):
        self.ctx, self.link_fn, self.pull_fn, self.cnt, self.use_pos = ctx, link_fn, pull_fn, cnt, use_cc_position
        self.take_rows, self.ctrl, self.events = [], [], []

    def market(self, day, mid, M, LK, ranks):
        ctx, cnt = self.ctx, self.cnt
        ws, we = ctx.times(M)
        O, LF = market_tables(M)
        P = []
        for ri in ranks:
            R = ctx.rows(M, TOP3[ri])
            if R is None:
                cnt[f'pairs_absent:{RANKS[ri]}'] += 1
                continue
            legs, A, K, excl, span = pair_inputs(R)
            P.append((ri, legs, A, K, excl, span))
        if not P:
            return
        los = [ws - PRE_MS] + [x[5][0] for x in P if x[5] is not None] + [int(x[3]['lh'][x[3]['lh'] >= 0].min()) for x in P if (x[3]['lh'] >= 0).any()]
        his = [we] + [x[5][1] for x in P if x[5] is not None] + [int(x[3]['hi'].max()) for x in P if len(x[3])]
        G = market_grid(ctx, M, min(los) - GRID_BACK_MS, max(his), cnt)
        guard(f'grid {day} {mid}')
        pull = self.pull_classes(O, LF, LK, ws, we, ranks)
        for ri, legs, A, K, excl, span in P:
            self.pair(day, mid, ri, legs, A, K, excl, span, G, M, ws, we, O, LF, LK, pull)

    def pull_classes(self, O, LF, LK, ws, we, ranks):
        """chain_pull.order_table classes per own hash (the pull chain's single definition), or None when unavailable."""
        if self.pull_fn is None or O is None or not len(O):
            return None
        O2 = O.copy()
        O2['a'] = O2['mk_signer'].astype(str).str.lower().map({a.lower(): i for i, a in enumerate(TOP3)}).astype(np.int64)
        O2 = O2[O2['a'].isin(list(ranks))]
        if not len(O2):
            return None
        T = self.pull_fn(O2, LF[LF['mk_hash'].isin(set(O2['mk_hash']))], LK, ws, we)
        return T.set_index('mk_hash')[['class_S_FIFO', 'class_S_NOPRIO']]

    def pair(self, day, mid, ri, legs, A, K, excl, span, G, M, ws, we, O, LF, LK, pull):
        ctx, cnt = self.ctx, self.cnt
        a = TOP3[ri]
        cnt['pairs'] += 1
        cnt['legs_ok_nonself'] += len(legs['sid'])
        cnt['legs_no_life'] += int((~legs['has_life']).sum())
        cnt['legs_lo_unbounded'] += int((legs['lh'] < 0).sum())
        cnt['legs_receipt_before_uuid'] += int((legs['hi'] < legs['settle']).sum())
        pos_fn = (lambda x: ctx.position(M, a, x)) if self.use_pos else None
        T = take_core(legs, A, K, excl, pos_fn)
        n = T['n']
        cnt['takes'] += n
        rows = None
        if n:
            s, hi, lh, settle = T['s'], T['hi'], T['lh'], T['settle']
            qa = K['q_all'].to_numpy(float)
            p_avg = np.where(qa > 0, K['qp'].to_numpy(float) / np.where(qa > 0, qa, 1.), NAN)
            fl = []
            for k in range(n):
                f = set()
                if lh[k] < 0:
                    f.add('window_unbounded')
                if not bool(K['has'].iat[k]):
                    f.add('no_life')
                elif not bool(K['cons'].iat[k]):
                    f.add('life_window_mismatch')
                if hi[k] < settle[k]:
                    f.add('receipt_before_uuid')
                if T['side_mixed'][k]:
                    f.add('side_mixed')
                if T['n_tie'][k] > 0:
                    f.add('simultaneous_action')
                if bool(K['self'].iat[k]):
                    f.add('self_trade')
                if int(K['n_bad'].iat[k]) > 0:
                    f.add('excluded_row')
                if T['n_excl_before'][k] > 0:
                    f.add('excluded_before')
                if hi[k] >= we:
                    f.add('after_we')
                fl.append(f)
            claim = np.array([not (f & NO_CLAIM) for f in fl], bool)
            tkh = K['tk_hash'].astype(str).to_numpy(object)
            rows = dict(day=np.full(n, day, object), mid=np.full(n, int(mid)), rank=np.full(n, ri + 1), addr=np.full(n, a, object),
                        sid=K['sid'].astype(str).to_numpy(object), tk_hash=tkh, s=s, q_all=qa, q_self=K['q_self'].to_numpy(float), dq=T['dq'],
                        n_legs=K['n_legs'].to_numpy(np.int64), n_px=K['n_px'].to_numpy(np.int64), p_avg=p_avg, p_min=K['p_min'].to_numpy(float),
                        p_max=K['p_max'].to_numpy(float), kinds=K['kinds'].to_numpy(object), settle_ms=settle, hi_ms=hi, x_lo_hard_ms=lh,
                        leg_status=K['sts'].to_numpy(object), pos_rec=T['rec'], tie_dq=T['tie'], n_tie=T['n_tie'],
                        cls_rec=np.array(CLS, object)[T['c_rec']], cls_sim=np.where(T['c_sim'] >= 0, np.array(CLS, object)[np.maximum(T['c_sim'], 0)], ''),
                        phase=np.where(hi >= ws, 'live', 'pre').astype(object), tau_s=(we - hi) / 1000., pos_sign=pos_sign(T['rec']), pos_bin=pos_bin(T['rec']),
                        prev_delay_lo_ms=T['prev_delay_lo'], prev_delay_hi_ms=T['prev_delay_hi'], prev_settle_gap_ms=T['prev_settle_gap'],
                        n_excl_before=T['n_excl_before'], status=np.array(['+'.join(sorted(f)) or OK for f in fl], object), claim=claim)
            rows['tk_n_records'] = pd.Series(tkh).map(pd.Series(tkh).value_counts()).to_numpy(np.int64)
            rows['tk_seq'] = pd.DataFrame({'h': tkh, 't': settle}).groupby('h')['t'].rank(method='first').to_numpy(np.int64)
            unid = MST.index('unidentified')
            for D in D_PROFILES:
                lo = np.where(lh >= 0, lh - D, settle - MAX_LAG - D)
                Wf = window_feats(ctx, M, G, lo, hi, s)
                st = start_state(ctx, M, lo, s)
                rows[f'pos_cert_{D}'] = T[f'cert_{D}']
                rows[f'cls_cert_{D}'] = np.where(T[f'c_cert_{D}'] >= 0, np.array(CLS, object)[np.maximum(T[f'c_cert_{D}'], 0)], '')
                rows[f'take_class_{D}'] = np.array(CLS, object)[T[f'cls_{D}']]
                for k_ in ('pend_n', 'pend_dq', 'pend_abs', 'n_cert', 'n_poss', 'n_rec_1s', 'n_rec_5s', 'offset_sh_rec_5s'):
                    rows[f'{k_}_{D}'] = T[f'{k_}_{D}']
                rows[f'n_inst_{D}'], rows[f'n_inst_ok_{D}'] = Wf['n_exp'], Wf['n_ok']
                for nm, _ in MOVES:
                    rows[f'cat_{nm}_{D}'] = np.array(DIRCAT, object)[Wf[f'cat_{nm}']]
                rows[f'edge_pf_min_c_{D}'] = 100. * np.where(s > 0, Wf['pf_min'] - p_avg, p_avg - Wf['pf_max'])
                rows[f'edge_pf_max_c_{D}'] = 100. * np.where(s > 0, Wf['pf_max'] - p_avg, p_avg - Wf['pf_min'])
                rows[f'spread_min_{D}'], rows[f'spread_max_{D}'] = Wf['spread_min'], Wf['spread_max']
                rows[f'touch_start_{D}'], rows[f'spread_start_{D}'], rows[f'depth_taken_start_{D}'] = st['touch'], st['spread'], st['depth']
                worst = np.where(s > 0, 100. * rows['p_max'], 100. * rows['p_min'])
                rows[f'through_worst_tk_{D}'] = np.where(s > 0, worst - st['touch'], st['touch'] - worst)
                rows[f'through_avg_tk_{D}'] = np.where(s > 0, 100. * p_avg - st['touch'], st['touch'] - 100. * p_avg)
                for nm_, v_ in (('inventory_unwind', T[f'm_inv_{D}']), ('directional', mech_directional(Wf['cat_r_5s'])),
                                ('after_fill_burst', mech_burst(T[f'n_cert_{D}'], T[f'n_poss_{D}']))):
                    rows[f'{nm_}_{D}'] = np.array(MST, object)[np.where(claim, v_, unid)]
                b = {k_: T[f'{k_}_{D}'] for k_ in ('n_cert', 'n_poss', 'n_rec_1s', 'n_rec_5s', 'offset_sh_rec_5s')}
                ft = feats_of(Wf, b, st, (we - hi) / 1000., T[f'pend_n_{D}'])
                ft['reduction_if_taken'] = np.isin(T['c_rec'], RED_CODES).astype(float)
                rows[f'_feats_{D}'] = ft
            rows['take_class'] = rows[f'take_class_{D_PRIMARY}']
            self.take_rows.append(rows)
            cand = candidates(G, span, settle, ws, we, A)
            pl, lvl = pools(cand, (hi >= ws).astype(np.int8), pos_sign(T['rec']), pos_bin(T['rec']))
            pl = [p if claim[j] else None for j, p in enumerate(pl)]
            lvl = np.where(claim, lvl, 4).astype(np.int8)
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, int(day), int(mid), ri + 1])))
            dr = draw(pl, rng, N_DRAWS)
            cf = {D: control_feats(ctx, M, G, cand, legs, D, we) for D in D_PROFILES} if cand is not None and len(cand['c']) else None
            self.ctrl.append(dict(draws=dr, level=lvl, cf=cf, cand_pos=None if cand is None else cand['pos'], sides=s, dq=T['dq']))
            cnt['control_candidates'] += 0 if cand is None else len(cand['c'])
            cnt['control_candidates_near_take_excluded'] += 0 if cand is None else cand['n_near']
        self.actions_pair(day, mid, ri, legs, A, rows, O, LF, LK, M, ws, we, pull)
        guard(f'pair {day} {mid} {ri + 1}')

    def actions_pair(self, day, mid, ri, legs, A, rows, O, LF, LK, M, ws, we, pull):
        """Every action instance of one pair -> self.events."""
        low, ev = TOP3[ri].lower(), []
        for j in np.flatnonzero(legs['role'] == 0):
            ev.append(dict(type='maker_fill', anchor=int(legs['hi'][j]), ref=int(legs['settle'][j]), s=int(legs['s'][j]), sh=float(legs['q'][j]),
                           mst='unidentified', ts=False, pop='outcome_selected_filled_orders'))
        Oa = O[O['mk_signer'].astype(str).str.lower() == low] if O is not None else None
        recs = order_rows(Oa, LF[LF['mk_hash'].isin(set(Oa['mk_hash']))]) if Oa is not None and len(Oa) else []
        recs.sort(key=lambda o: (o['first_hi'], o['mk_hash']))
        seen = set()
        for o in recs:
            sd = 1 if o['side'] == 0 else -1
            t = 'post_new_level' if (o['side'], o['tick']) not in seen else 'post_same_level'
            seen.add((o['side'], o['tick']))
            wid = (o['post_hi'] - o['post_lo_union']) if o['post_lo_union'] >= 0 else -1
            ev.append(dict(type=t, anchor=o['first_hi'], ref=o['first_hi'], s=sd, sh=o['cum'], mst='unidentified', ts=bool(wid < 0 or wid > 1000),
                           pop='outcome_selected_filled_orders', extra=dict(window_ms=wid)))
            if len(o['f_hi']) >= 2:
                oo = np.argsort(o['f_hi'], kind='mergesort')
                f_lh, f_hi = o['f_lh'][oo], o['f_hi'][oo]
                for j in range(len(f_hi) - 1):
                    gap = f_hi[j + 1] - f_lh[j] if f_lh[j] >= 0 else -1
                    tsens = bool(0 <= gap <= KEEP_REACT_MS)
                    ev.append(dict(type='keep_after_fill', anchor=int(f_hi[j + 1]), ref=int(f_hi[j + 1]), s=sd, sh=NAN, ts=tsens,
                                   mst='timing_sensitive' if tsens else 'compatible', pop='outcome_selected_filled_orders'))
            if pull is not None and o['mk_hash'] in pull.index:
                for sc in ('S_NOPRIO', 'S_FIFO'):
                    pc_ = str(pull.at[o['mk_hash'], f'class_{sc}'])
                    ident = pc_ == 'cancel_possible' and not (o['ex_any'] or o['ec_any'])
                    ev.append(dict(type=f'pull_possible:{sc}:{pc_}', anchor=int(o['end_hi'] if o['end_hi'] >= 0 else o['last_hi']), ref=int(o['last_hi']),
                                   s=sd, sh=NAN, ts=False, mst='compatible' if ident else 'unidentified', pop='outcome_selected_filled_orders'))
            else:
                self.cnt['pull_classes_unavailable'] += 1
        for i_, x in enumerate(recs):                                   # ladder overlap: same side, different ticks, overlap certain
            for y in recs[i_ + 1:]:
                if x['side'] != y['side'] or x['tick'] == y['tick'] or y['first_hi'] > x['last_hi']:
                    continue                                            # y first filled after x's last fill: overlap cannot be certain
                if self.link_fn is None:
                    self.cnt['ladder_link_unavailable'] += 1
                    continue
                if bool(self.link_fn(x, y)['overlap_certain']):
                    ev.append(dict(type='ladder_overlap', anchor=y['first_hi'], ref=y['first_hi'], s=1 if x['side'] == 0 else -1, sh=NAN, mst='compatible',
                                   ts=False, pop='outcome_selected_filled_orders', extra=dict(ticks_apart=abs(x['tick'] - y['tick']))))
        for x in recs:                                                  # two-sided quoting
            opp = [y for y in recs if y['side'] != x['side'] and feasible_overlap(x, y, ws, we)]
            if not opp:
                continue
            cert = [y for y in opp if two_sided_certain(x, y)]
            width = [(y['tick'] - x['tick']) if x['side'] == 0 else (x['tick'] - y['tick']) for y in (cert or opp)]
            ev.append(dict(type='two_sided', anchor=x['first_hi'], ref=x['first_hi'], s=1 if x['side'] == 0 else -1, sh=NAN,
                           mst='compatible' if cert else 'timing_sensitive', ts=not cert, pop='outcome_selected_filled_orders',
                           extra=dict(width_tk=float(np.median(width)), certain=bool(cert))))
        Lk = LK[LK['mk_signer'].astype(str).str.lower() == low] if LK is not None and len(LK) else None
        if Lk is not None and len(Lk):
            by = {o['mk_hash']: o for o in recs}
            for r in Lk.itertuples():
                P_, Q_ = by.get(r.prev_hash), by.get(r.next_hash)
                st, fs = 'link_hypotheses_unavailable', ''
                if str(r.status) != OK:
                    st = f'link_{r.status}'
                elif P_ is None or Q_ is None:
                    st = 'order_missing'
                elif self.link_fn is not None:
                    res = self.link_fn(P_, Q_)
                    st, fs = str(res['status']), '+'.join(res['feasible'])
                mst = {'sequential_identified': 'compatible', 'overlap_identified': 'contradicted', 'tied': 'timing_sensitive'}.get(st, 'unidentified')
                ev.append(dict(type=str(r.kind), anchor=int(r.next_first_hi_ms), ref=int(r.next_first_hi_ms), s=1 if str(r.y_side) == 'bid' else -1, sh=NAN,
                               mst=mst, ts=st == 'tied', pop='outcome_selected_filled_orders',
                               extra=dict(link_status=st, feasible=fs, replace_feasible=int('replace' in fs.split('+')), x_latest=int(bool(r.x_latest)),
                                          tick_change=int(r.next_ptick) - int(r.prev_ptick), gap_ms=int(r.gap_ms))))
        if rows is not None:
            for k in range(len(rows['sid'])):
                c = rows['take_class'][k]
                base = dict(anchor=int(rows['hi_ms'][k]), ref=int(rows['settle_ms'][k]), s=int(rows['s'][k]), sh=float(rows['q_all'][k]),
                            pop='take_records_complete')
                ev.append(dict(type=f'take_{c}', mst=rows[f'inventory_unwind_{D_PRIMARY}'][k], ts=c == 'timing_sensitive', **base))
                extra_types = (['take_self_trade'] if 'self_trade' in rows['status'][k] else []) + (['take_sweep'] if rows['n_px'][k] >= 2 else []) + \
                    (['take_multi_record_order'] if rows['tk_n_records'][k] >= 2 else []) + \
                    [f'take_kind_{kd}' for kd in ('mint', 'merge') if kd in str(rows['kinds'][k]).split('+')]
                for t in extra_types:
                    ev.append(dict(type=t, mst='unidentified', ts=False, **base))
        if not ev:
            return
        anc = np.array([e['anchor'] for e in ev], np.int64)
        sv = np.array([e['s'] for e in ev], np.int64)
        B, Kb = self.ctx.btc(M, anc), self.ctx.book(M, anc)
        r5 = np.where(B['code'] == 0, B['r_5s'], NAN)
        pos = _cs(A['dq'].to_numpy(float))[np.searchsorted(A['settle'].to_numpy(np.int64), anc, 'left')]
        nx = next_events(legs, rows, np.array([e['ref'] for e in ev], np.int64), sv)
        for j, e in enumerate(ev):
            e.update(day=day, mid=int(mid), rank=ri + 1, tau=(we - e['anchor']) / 1000., live=bool(e['anchor'] >= ws), spread=Kb['spread'][j],
                     btc_ok=bool(B['code'][j] == 0), move=bool(np.isfinite(r5[j]) and abs(r5[j]) >= MOVES[1][1]),
                     with_=bool(np.isfinite(r5[j]) and sv[j] * r5[j] >= MOVES[1][1]), offset=bool(sv[j] * pos[j] < -POS_EPS),
                     big=bool(abs(pos[j]) >= POS_BINS[1]), next=nx[j])
        self.events.extend(ev)

    # ---- tables ----
    def takes_table(self):
        if not self.take_rows:
            return pd.DataFrame()
        cols = [k for k in self.take_rows[0] if not k.startswith('_')]
        return pd.concat([pd.DataFrame({k: r[k] for k in cols}) for r in self.take_rows], ignore_index=True)

    def controls_table(self, days):
        """take_vs_control (i) and reducing_vs_adding (ii) rows per group x D x feature, and the pool-level counts."""
        if not self.take_rows:
            return pd.DataFrame(), {}
        TR = self.take_rows
        rk = np.concatenate([r['rank'] for r in TR])
        dd = np.concatenate([r['day'] for r in TR]).astype(str)
        mids = np.concatenate([r['mid'] for r in TR])
        groups = [(f'rank{r}', d) for r in (1, 2, 3) for d in days] + [(f'rank{r}', 'all') for r in (1, 2, 3)] + [('pooled', 'all')]
        gm = [(np.ones(len(rk), bool) if sc == 'pooled' else (rk == int(sc[4:]))) & ((dd == d) if d != 'all' else True) for sc, d in groups]
        lvl = np.concatenate([c['level'] for c in self.ctrl])
        cls0 = np.concatenate([r['take_class'] for r in TR])
        claim = np.concatenate([r['claim'] for r in TR])
        info = dict(pool_level={nm: int((lvl == i).sum()) for i, nm in enumerate(POOL_LEVELS)})
        lab = np.where(np.isin(cls0, REDUCING) & claim, 1, np.where(np.isin(cls0, ADDING) & claim, 0, -1))
        perm = np.full((len(rk), N_PERM), -1, np.int8)
        for key in sorted(set(zip(dd.tolist(), mids.tolist(), rk.tolist()))):
            m = np.flatnonzero((dd == key[0]) & (mids == key[1]) & (rk == key[2]) & (lab >= 0))
            if len(m) < 2:                               # a single eligible take: its only permutation is itself
                perm[m, :] = lab[m][:, None]
                continue
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, int(key[0]), int(key[1]), int(key[2]), 1])))
            base = lab[m]
            for r in range(N_PERM):
                perm[m, r] = base[rng.permutation(len(m))]
        out = []
        for D in D_PROFILES:
            ft_all = {f: np.concatenate([r[f'_feats_{D}'][f] for r in TR]) for f, _ in FEATS}
            for f, kind in FEATS:
                parts = []
                for c in self.ctrl:
                    dr = c['draws']
                    if c['cf'] is None:
                        parts.append(np.full(dr.shape, NAN))
                    elif f == 'reduction_if_taken':
                        v = np.isin(classify(c['cand_pos'][np.maximum(dr, 0)], c['dq'][:, None]), RED_CODES).astype(float)
                        parts.append(np.where(dr >= 0, v, NAN))
                    else:
                        ap, am = c['cf'][D][1][f], c['cf'][D][-1][f]
                        v = np.where((c['sides'] > 0)[:, None], ap[np.maximum(dr, 0)], am[np.maximum(dr, 0)])
                        parts.append(np.where(dr >= 0, v, NAN))
                Fc = np.concatenate(parts, axis=0)
                ft = ft_all[f]
                for (sc, d), m in zip(groups, gm):
                    okt = m & claim & (lvl <= 2) & np.isfinite(ft)
                    row = dict(comparison='take_vs_control', scope=sc, day=d, D=D, feature=f, kind=kind, n_takes=int(okt.sum()),
                               n_fallback1=int((m & (lvl == 1)).sum()), n_fallback2=int((m & (lvl == 2)).sum()), n_no_control=int((m & (lvl == 3)).sum()),
                               n_no_claim=int((m & (lvl == 4)).sum()))
                    obs = float(ft[okt].mean()) if okt.any() else NAN
                    Fg = Fc[okt]
                    N_ = np.isfinite(Fg).sum(0)
                    null = np.where(N_ > 0, np.nansum(Fg, 0) / np.maximum(N_, 1), NAN) if len(Fg) else np.full(N_DRAWS, NAN)
                    row.update(obs=obs, **null_stats(obs, null))
                    if kind == 'binary' and np.isfinite(obs) and np.isfinite(row['null_mean']) and 0 < obs < 1 and 0 < row['null_mean'] < 1:
                        row['odds_ratio'] = (obs / (1 - obs)) / (row['null_mean'] / (1 - row['null_mean']))
                    out.append(row)
                if f not in CONTRAST_FEATS:
                    continue
                for (sc, d), m in zip(groups, gm):
                    mm = m & (lab >= 0) & np.isfinite(ft)
                    n1, n0 = int((mm & (lab == 1)).sum()), int((mm & (lab == 0)).sum())
                    row = dict(comparison='reducing_vs_adding', scope=sc, day=d, D=D, feature=f, kind=kind, n_reducing=n1, n_adding=n0, n_takes=n1 + n0,
                               n_excluded_class=int((m & (lab < 0)).sum()))
                    if n1 == 0 or n0 == 0:
                        out.append(dict(row, obs=NAN, mean_reducing=NAN, mean_adding=NAN, **null_stats(NAN, [])))
                        continue
                    m1, m0 = float(ft[mm & (lab == 1)].mean()), float(ft[mm & (lab == 0)].mean())
                    Pm, fv = perm[mm], ft[mm][:, None]
                    k1, k0 = (Pm == 1).sum(0), (Pm == 0).sum(0)
                    null = np.where((k1 > 0) & (k0 > 0), np.where(Pm == 1, fv, 0.).sum(0) / np.maximum(k1, 1) - np.where(Pm == 0, fv, 0.).sum(0) /
                                    np.maximum(k0, 1), NAN)
                    out.append(dict(row, obs=m1 - m0, mean_reducing=m1, mean_adding=m0, **null_stats(m1 - m0, null)))
        return pd.DataFrame(out), info

    def actions_table(self, days):
        """One row per action type x scope x day (each day and all): counts, denominators, context, next event, statuses, logic row."""
        if not self.events:
            return pd.DataFrame()
        Ev = pd.DataFrame(self.events)
        Ev['day'] = Ev['day'].astype(str)
        if 'extra' not in Ev:
            Ev['extra'] = None
        take_re = r'^take_(?:' + '|'.join(CLS) + r')$'
        rows = []
        for sc, mem in [(f'rank{r}', (r,)) for r in (1, 2, 3)] + [('pooled', (1, 2, 3))]:
            for d in list(days) + ['all']:
                E_ = Ev[Ev['rank'].isin(mem) & ((Ev['day'] == d) if d != 'all' else True)]
                if not len(E_):
                    continue
                den = dict(n_maker_fills=int((E_['type'] == 'maker_fill').sum()), n_orders=int(E_['type'].isin(['post_new_level', 'post_same_level']).sum()),
                           n_takes=int(E_['type'].str.match(take_re).sum()))
                for t, G_ in E_.groupby('type', sort=True):
                    lk = LOGIC[logic_key(t)]
                    mc = G_['mst'].value_counts()
                    top = int(mc.max())
                    best = [x_ for x_ in MST if int(mc.get(x_, 0)) == top]
                    nx = G_['next'].value_counts(normalize=True)
                    xs = {}
                    ex = [e for e in G_['extra'] if isinstance(e, dict)]
                    if ex:
                        X_ = pd.DataFrame(ex)
                        for c in X_.columns:
                            if X_[c].dtype.kind in 'fiu':
                                xs[f'{c}_median'] = float(np.nanmedian(X_[c].astype(float)))
                                xs[f'{c}_sum'] = float(np.nansum(X_[c].astype(float)))
                            else:
                                xs[c] = {str(k): int(v) for k, v in X_[c].astype(str).value_counts().head(8).items()}
                    sp = G_['spread'].astype(float)
                    rows.append(dict(action_type=t, scope=sc, day=d, n=int(len(G_)), shares=float(np.nansum(G_['sh'].astype(float))), **den,
                                     population=str(G_['pop'].iloc[0]), tau_med_s=float(np.nanmedian(G_['tau'])), live_share=float(G_['live'].mean()),
                                     spread_med_tk=float(np.nanmedian(sp)) if np.isfinite(sp).any() else NAN, btc_ok_share=float(G_['btc_ok'].mean()),
                                     btc_move_share=float(G_['move'].mean()), btc_with_share=float(G_['with_'].mean()),
                                     offset_pos_share=float(G_['offset'].mean()), pos_ge200_share=float(G_['big'].mean()),
                                     timing_sensitive_n=int(G_['ts'].sum()), next_modal=str(nx.index[0]), next_modal_share=float(nx.iloc[0]),
                                     next_dist='|'.join(f'{k}:{v:.3f}' for k, v in nx.items()), **{f'n_{x_}': int(mc.get(x_, 0)) for x_ in MST},
                                     status=best[0] if len(best) == 1 else 'unidentified',
                                     status_rule='the most frequent mechanism status of the type (ties -> unidentified)',
                                     decision_window=('[x_lo_hard - D, hi] (match window, D profiles)' if t.startswith('take') else
                                                      'first-fill evidence; posting window in extra' if t.startswith('post') else 'declared anchor (contract 7.3)'),
                                     state_change=lk[0], alternatives=lk[1], proposed_trade_off=lk[2], observed_action=lk[3],
                                     resulting_state=f'next own event {nx.index[0]} ({nx.iloc[0]:.3f})',
                                     supporting_evidence='; '.join(f'{x_} {int(mc.get(x_, 0))}' for x_ in MST), competing_explanation=lk[4],
                                     what_would_falsify=lk[5], extra=json.dumps(xs, sort_keys=True, default=str)))
        return pd.DataFrame(rows)


def summary_table(takes, controls, actions, days):
    """Per address x day and pooled: take-class counts (D = 0; the recorded variant beside), mechanism-status counts, key associations,
    action counts."""
    if takes is None or not len(takes):
        return pd.DataFrame()
    out = []
    for sc, mem in [(f'rank{r}', (r,)) for r in (1, 2, 3)] + [('pooled', (1, 2, 3))]:
        for d in list(days) + ['all']:
            T = takes[takes['rank'].isin(mem) & ((takes['day'].astype(str) == d) if d != 'all' else True)]
            row = dict(scope=sc, day=d, n_takes=int(len(T)), n_claim=int(T['claim'].sum()), **{f'cls_{c}': int((T['take_class'] == c).sum()) for c in CLS},
                       **{f'cls_rec_{c}': int((T['cls_rec'] == c).sum()) for c in CLS})
            row['actual_reduction'] = sum(row[f'cls_{c}'] for c in REDUCING)
            row['opposing'] = row['actual_reduction'] + row['cls_overshoot']
            row['actual_reduction_rec'] = sum(row[f'cls_rec_{c}'] for c in REDUCING)
            row['opposing_rec'] = row['actual_reduction_rec'] + row['cls_rec_overshoot']
            row['nonflat_rec'] = row['n_takes'] - row['cls_rec_from_flat'] - row['cls_rec_no_change'] - row['cls_rec_unclassified']
            for mech in MECH:
                for st in MST:
                    row[f'{mech}_{st}'] = int((T[f'{mech}_{D_PRIMARY}'] == st).sum())
            if controls is not None and len(controls):
                cv = controls[(controls['scope'] == sc) & (controls['day'] == d) & (controls['D'] == D_PRIMARY)]
                for f in ('burst3_certain', 'burst_possible_any', 'n_rec_5s', 'offset_sh_rec_5s', 'r_5s_with', 'r_5s_against', 'reduction_if_taken'):
                    r = cv[(cv['comparison'] == 'take_vs_control') & (cv['feature'] == f)]
                    if len(r):
                        for k in ('obs', 'null_mean', 'p_hi', 'p_lo', 'n_takes'):
                            row[f'tvc_{f}_{k}'] = r[k].iloc[0]
                    r = cv[(cv['comparison'] == 'reducing_vs_adding') & (cv['feature'] == f)]
                    if len(r):
                        for k in ('obs', 'p_hi', 'p_lo', 'n_reducing', 'n_adding'):
                            row[f'rva_{f}_{k}'] = r[k].iloc[0]
            if actions is not None and len(actions):
                av = actions[(actions['scope'] == sc) & (actions['day'] == d)]
                for t, nn in zip(av['action_type'], av['n']):
                    row[f'act_{t}'] = int(nn)
            out.append(row)
    return pd.DataFrame(out)


def status_counts(tk):
    c = collections.Counter()
    if tk is not None and len(tk):
        for s in tk['status']:
            for f in str(s).split('+'):
                c[f] += 1
    return dict(sorted(c.items()))


# ------------------------------------------------------------------ run / pilot ------------------------------------------------------------------
def _stager():
    c0 = time.process_time()
    st, last = {}, [c0]

    def tick(name):
        now = time.process_time()
        st[name] = round(st.get(name, 0.) + now - last[0], 3)
        last[0] = now
        need(now - c0 < CPU_CAP_S, f'CPU self-guard: {now - c0:.1f} s >= {CPU_CAP_S} s')
    return st, tick, c0


def deps(link_fn=None, pull_fn=None):
    """order_estimator.link_hypotheses (contract 4.6) and chain_pull.order_table (contract 7.2): imported, never redefined."""
    if link_fn is None and (HERE / 'order_estimator.py').is_file():
        import order_estimator as OE  # noqa: E402
        link_fn = getattr(OE, 'link_hypotheses', None)
    if pull_fn is None and (HERE / 'chain_pull.py').is_file():
        import chain_pull as CP  # noqa: E402
        pull_fn = getattr(CP, 'order_table', None)
    return link_fn, pull_fn


def build(inp, days, ranks, ctx, link_fn, pull_fn, cnt, hashes, tick, manifest=None, fv_sha=None, guard_mib=C.GUARD_MIB, use_cc_position=True):
    """The shared build of run and pilot: population, then every BOOK market of `days` for the address indices `ranks`."""
    addrs, book = population(Path(inp) / 'econ' / 'identities.json', manifest or E.BOOK_MANIFEST, hashes)
    need(all(d in book for d in days), f'days {days} are not all BOOK manifest days')
    exp = fv_sha or json.loads(FV_RECEIPT.read_text())['inputs']['econ/fills_valued.parquet']['sha256']
    fvk = str((Path(inp) / 'econ' / 'fills_valued.parquet').resolve())
    tick('population')
    B = Build(ctx, link_fn, pull_fn, cnt, use_cc_position)
    for d in days:
        for mid in book[d]:
            M = ctx.market(d, mid)
            if fvk in ctx.hash_cache and 'econ/fills_valued.parquet' not in hashes:
                need(ctx.hash_cache[fvk] == exp, f'fills_valued sha256 {ctx.hash_cache[fvk][:12]} differs from the accepted binding {exp[:12]} (refused)')
                hashes['econ/fills_valued.parquet'] = ctx.hash_cache[fvk]
            for k_, v_ in getattr(M, 'inputs', {}).items():
                if isinstance(v_, dict) and v_.get('sha256'):
                    hashes.setdefault(k_, v_['sha256'])
            tick('context_load')
            LK = read_links(inp, d, mid, hashes)
            tick('links')
            B.market(d, mid, M, LK, ranks)
            cnt['markets'] += 1
            tick('pairs')
            guard(f'market {d} {mid}', guard_mib)
    need('econ/fills_valued.parquet' in hashes, 'fills_valued was not hashed by chain_context (no binding check; refused)')
    return B


def tables(B, days):
    tk = B.takes_table()
    ctrl, info = B.controls_table(days)
    act = B.actions_table(days)
    return tk, ctrl, act, summary_table(tk, ctrl, act, days), info


def write_out(out, tk, ctrl, act, summ, rec):
    tmp = out.with_name(f'.{out.name}.partial{os.getpid()}')
    need(not tmp.exists(), f'partial directory {tmp} exists')
    tmp.mkdir(parents=True)
    try:
        for nm, df in (('takes', tk), ('take_controls', ctrl), ('actions', act), ('summary', summ)):
            df.to_csv(tmp / f'{nm}.csv', index=False)
        X._json(tmp / 'receipt.json', rec)
        need(not out.exists(), f'write-once: {out} appeared during the run (refused)')
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def receipt(kind, days, cnt, hashes, ctx, git, stages, c0, t0, asof0, tk, ctrl, act, summ, info, link_fn, pull_fn, extra=None):
    srcs = {str(Path(f).resolve().relative_to(C.REPO)): _sha(f) for f in TOOL_FILES + DEP_FILES if Path(f).exists()}
    rec = dict(version=VERSION, kind=kind, contract=CONTRACT, module=str(Path(__file__).resolve()), asof_start=asof0, asof_end=_now(), run=C.RUN,
               addresses=list(TOP3), days=list(days), context=ctx.name, link_hypotheses=getattr(link_fn, '__module__', None),
               pull_classes=getattr(pull_fn, '__module__', None), inputs=hashes, sources=srcs, git=git,
               constants=dict(STEP=STEP, MAX_LAG=MAX_LAG, D_PROFILES=D_PROFILES, D_CONTRACT=D_CONTRACT, D_DISPATCH=D_DISPATCH, D_PRIMARY=D_PRIMARY,
                              POS_EPS=POS_EPS, POS_BINS=POS_BINS, MOVES=MOVES, BURST_N=BURST_N, BURST_MS=BURST_MS, REC_MS=REC_MS, EXCL_MS=EXCL_MS,
                              PRE_MS=PRE_MS, N_DRAWS=N_DRAWS, N_PERM=N_PERM, SEED=SEED, KEEP_REACT_MS=KEEP_REACT_MS, NO_CLAIM=sorted(NO_CLAIM)),
               class_names=dict(contract=CLS, dispatch=DISPATCH_NAMES), counts={k: v for k, v in sorted(cnt.items())}, pools=info,
               statuses=status_counts(tk), outputs={nm: int(len(df)) for nm, df in (('takes', tk), ('take_controls', ctrl), ('actions', act), ('summary', summ))},
               population_labels=dict(orders='outcome_selected_filled_orders (unfilled orders are invisible)', takes='take_records_complete'),
               cpu_s=dict(by_stage=stages, total=round(time.process_time() - c0, 3), cap=CPU_CAP_S), wall_s=round(time.time() - t0, 2),
               vmhwm_mib=round(hwm(), 1), guard_mib=C.GUARD_MIB)
    rec.update(extra or {})
    return rec


def run(inp=None, out=None, cc=None, link_fn=None, pull_fn=None, require_freeze=True, manifest=None, fv_sha=None, days=None, log=print):
    need(C.RUN == 'v2', f'MT_RUN must be v2 (is {C.RUN!r})')
    inp = Path(inp) if inp else C.OUT
    out = Path(out) if out else C.RECEIPT / 'chains' / 'take'
    need(not out.exists(), f'write-once: {out} exists (refused before any input is read)')
    git = X._git([f for f in TOOL_FILES if Path(f).exists()])
    if require_freeze:
        need(all(Path(f).exists() for f in TOOL_FILES) and bool(git.get('unmodified')),
             'freeze: the tool, its selftest, chain_context.py and the contract must be tracked and unmodified before the run (refused)')
    ctx = make_ctx(cc, inp)
    link_fn, pull_fn = deps(link_fn, pull_fn)
    need(link_fn is not None and pull_fn is not None, 'order_estimator.link_hypotheses / chain_pull.order_table missing (refused)')
    t0, asof0 = time.time(), _now()
    stages, tick, c0 = _stager()
    cnt, hashes = collections.Counter(), {}
    days = tuple(days or C.MEASURE_DAYS)
    B = build(inp, days, (0, 1, 2), ctx, link_fn, pull_fn, cnt, hashes, tick, manifest, fv_sha)
    tk, ctrl, act, summ, info = tables(B, days)
    tick('tables')
    guard('tables')
    rec = receipt('run', days, cnt, hashes, ctx, git, stages, c0, t0, asof0, tk, ctrl, act, summ, info, link_fn, pull_fn)
    write_out(out, tk, ctrl, act, summ, rec)
    log(json.dumps(dict(out=str(out), outputs=rec['outputs'], cpu_s=rec['cpu_s'], vmhwm_mib=rec['vmhwm_mib']), indent=1))
    return rec


def pilot(day, rank, inp=None, out=None, cc=None, link_fn=None, pull_fn=None, manifest=None, fv_sha=None, log=print):
    """One address-day: the same build; writes chains/take_pilot_<day>_r<rank> (write-once) with the measured projection; prints a digest."""
    need(C.RUN == 'v2', f'MT_RUN must be v2 (is {C.RUN!r})')
    need(day in C.MEASURE_DAYS, f'pilot day {day} is not a measurement day (refused before any path is built)')
    need(rank in (1, 2, 3), f'pilot rank {rank} not in 1..3')
    inp = Path(inp) if inp else C.OUT
    out = Path(out) if out else C.RECEIPT / 'chains' / f'take_pilot_{day}_r{rank}'
    need(not out.exists(), f'write-once: {out} exists (refused before any input is read)')
    ctx = make_ctx(cc, inp)
    link_fn, pull_fn = deps(link_fn, pull_fn)
    t0, asof0 = time.time(), _now()
    stages, tick, c0 = _stager()
    cnt, hashes = collections.Counter(), {}
    B = build(inp, (day,), (rank - 1,), ctx, link_fn, pull_fn, cnt, hashes, tick, manifest, fv_sha)
    tk, ctrl, act, summ, info = tables(B, (day,))
    tick('tables')
    total = time.process_time() - c0
    nmk = max(1, int(cnt['markets']))
    per_market = stages.get('context_load', 0.) / nmk + stages.get('links', 0.) / nmk
    per_pair = (stages.get('pairs', 0.) + stages.get('tables', 0.)) / max(1, int(cnt['pairs']))
    proj = stages.get('population', 0.) + per_market * N_BOOK + per_pair * N_PAIRS_FULL
    projection = dict(full_run_cpu_s=round(proj, 2), pilot_cpu_s=round(total, 3),
                      rule=f"population {stages.get('population', 0.):.2f} + per-market (context + links) {per_market:.3f} x {N_BOOK} + per-pair "
                           f"(pairs + tables) {per_pair:.3f} x {N_PAIRS_FULL}; the first market's load includes the one-off fills_valued sha256")
    rec = receipt('pilot', (day,), cnt, hashes, ctx, X._git([f for f in TOOL_FILES if Path(f).exists()]), stages, c0, t0, asof0, tk, ctrl, act, summ,
                  info, link_fn, pull_fn, extra=dict(pilot=dict(day=day, rank=rank), projection=projection))
    write_out(out, tk, ctrl, act, summ, rec)
    sc = f'rank{rank}'
    key = ctrl[(ctrl['D'] == D_PRIMARY) & (ctrl['scope'] == sc) & (ctrl['day'] == day)] if len(ctrl) else ctrl
    cols = [c for c in ('comparison', 'feature', 'n_takes', 'obs', 'null_mean', 'p_hi', 'p_lo', 'mean_reducing', 'mean_adding') if c in key.columns]
    dig = dict(out=str(out), context=ctx.name, counts=dict(cnt), statuses=status_counts(tk), pools=info,
               summary=summ[(summ['scope'] == sc) & (summ['day'] == day)].to_dict('records') if len(summ) else [],
               key_controls=key[cols].to_dict('records') if len(key) else [],
               actions=act[(act['scope'] == sc) & (act['day'] == day)][['action_type', 'n', 'status', 'timing_sensitive_n', 'next_modal',
                                                                         'next_modal_share']].to_dict('records') if len(act) else [],
               cpu_s=rec['cpu_s'], projection=projection, vmhwm_mib=rec['vmhwm_mib'])
    log(json.dumps(X._clean(dig), indent=1, sort_keys=True, default=str))
    return rec, (tk, ctrl, act, summ)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['run']:
        run()
    elif len(a) == 3 and a[0] == 'pilot':
        pilot(a[1], int(a[2]))
    else:
        raise SystemExit('usage: MT_RUN=v2 chain_take.py run | MT_RUN=v2 chain_take.py pilot <day> <rank>')
