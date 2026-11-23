"""TAKE-LOGIC v1 (P-2026-005 MAKER-TRACE; session aea5bfb9, 2026-11-23): the DECISION-TIME TAKE HAZARD of the ex-ante top-3 makers on the
19 BOOK markets, built EXACTLY to predictfun/reviews/maker_trace_20261122/take_logic_contract.md (frozen before computation, commit 3d9c6b082).
USER mandate: solid proof of the takers' logic - what the inventory is, BTC with / against it, the end of the 5-minute market, one-off or
repeated. DESCRIPTIVE on four consumed days (Oct 24-27; G = 4 < 5 days: point estimates, no intervals); p is a circular-shift tail fraction,
not independent-day confidence; no fit, not a validated rule, no model skill, no profitability claim.

POPULATION every maker-second: each ex-ante top-3 address (chain_context.top3 = chain_take.TOP3) x each of the 19 BOOK markets
  (chain_context.book_markets), instants t = ws + k * 1000 ms, k = 0 .. (we - ws) // 1000 - 1 (ws to we - 1000: the market's whole span, never
  the maker's active span); a pair without fills enters with inventory 0. Each market is loaded ONCE (chain_context.load_market) for the three
  addresses. A population check (every pair holds every instant on the grid; no-fill pairs at q = 0) must pass before anything is written.
EVENT every row of v2/chains/take/takes.csv (chain_take v1, any status) at the instant t with t < x_lo_hard_ms <= t + 1000 (k =
  (x_lo_hard_ms - ws - 1) // 1000); an event-second counts once per type, its takes are counted beside it. Type from take_class, the column
  that reproduces chain_take's summary cls_* counts (asserted against the dispatch numbers and summary.csv pooled / all, else refused):
  reduce = reduce_partial / full_close / reversal_reducing / overshoot; add = add / from_flat; other = no_change / unclassified;
  timing_sensitive. Takes outside the instants are statuses (before_population: x_lo_hard_ms <= ws; after_population; unbounded_window;
  no_pair), counted, never dropped silently.
STATE known at t: variant D0 (primary); D1000 (EVERY read at t - 1000 ms: inventory, fills, BTC, book, time left, the end points); pend (pending
  legs included: q = position 'possible' = status-ok legs with x_lo_hard <= t, own fills + the pending own maker legs; all else as D0).
  q = chain_context.position(M, a, t + 1, 'hi')['pos'] (status-ok legs with hi <= t, YES space); |q| bins 0 | (0, 50] | (50, 200] | (200, 500] |
  > 500 sh (|q| <= 1e-6 is 0), sign long / short. inv_timing_sensitive: an end of the feasible range [pos_min, pos_max] of
  chain_context.pending at t + 1 (legs with x_lo_hard <= t < hi) has another |q| bin or sign (exact for any subset of the pending legs).
  inv_ledger_conditional: position status excluded_leg_possible (an excluded leg's window started by t). a_h = sign(q) r_h, r_h =
  chain_context.btc_ret r_1s / r_5s / r_30s (pf_mt_decision.btc_feats) -> chain_context.align_cat with / against / flat at MOVE_THR 0.5 / 1 /
  2 bp; missing BTC its own level (counted). a_5s dose bins <= -5 | (-5, -2] | (-2, -1] | (-1, 1) | [1, 2) | [2, 5) | >= 5 bp. Time left
  (chain_context.time_left) > 300 | (240, 300] | (120, 240] | (60, 120] | (30, 60] | (10, 30] | [0, 10] s. Own fills = own status-ok
  non-self-trade MAKER legs with hi in (t - 5 s, t] (per leg, chain_take's burst unit) -> 0 | 1-2 | >= 3. Unwind edge (cents): long
  100 (best bid - p_fair), short 100 (p_fair - best ask); chain_context.book_tob status ok and BTC ok, else missing; q = 0 -> no_inventory;
  p_fair = btc_ret p_fair (the source chain_take's Ctx.btc wraps); bins <= -2 | (-2, -0.5] | (-0.5, 0.5) | [0.5, 2) | >= 2.
M1 hazard = event-seconds / maker-seconds per cell (|q| bin x sign x a_5s category x time-left bin) and per factor (every state factor), per
  type (+ any), ranks 1-3 and pooled (D0).
M2 Mantel-Haenszel person-time RR = sum_s a_s T0_s / T_s / sum_s b_s T1_s / T_s over strata with exposed AND reference person-time, reduce and
  add, rank 1 and pooled, q != 0 only: RR_against / RR_with (a_5s against / with vs flat; strata |q| bin x time left), RR_size (|q| > 200 vs
  (0, 50]; a_5s category x time left), RR_end (time left <= 30 s vs (60, 240]; |q| bin x a_5s category), RR_burst (own fills >= 3 vs 0; |q| bin
  x time left), RR_edge (edge >= 0.5 c vs (-0.5, 0.5); |q| bin x time left); RR_against / RR_with at 1 s and 30 s (secondary). Every pooled
  stratum includes rank (the contract names rank for RR_against / RR_with; applied to every ratio: the conservative reading).
M3 null (declared in the contract): each maker-market's event-second vector circularly shifted by a uniform offset mod its instant count, 500
  shifts, seed 20261123 (offsets per pair from Generator(PCG64(SeedSequence([20261123, day, mid, rank]))): a pilot pair carries the full run's
  offsets); the state paths stay fixed. p_hi = (1 + #{null >= obs}) / 501, p_lo likewise; ties within 1e-12 relative count; a NaN replicate (no
  event in an informative stratum) counts on both sides (conservative).
M4 market consistency: the crude within-market RR (the scope's ranks, q != 0, exposed / reference) over supported markets (>= 1 event-second,
  >= 30 exposed and >= 30 reference maker-seconds): n RR > 1 / n supported, all days and per day. Patterns P1-P8: event-seconds (primary
  unit), takes, maker-days, markets with >= 1 / >= 5 event-seconds, share of that maker's takes on the run's days; P5 needs q != 0 as P1-P4;
  reduce takes with q = 0, missing BTC or a side equal to the inventory sign are status rows.
M5 q at END (we), END - 30 s, END - 60 s per pair (legs with hi <= the point); share flattened |q(END)| <= 0.1 |q(END - 60 s)| over the scope's
  pairs with |q(END - 60 s)| > 0, per variant; the reduce hazard by time-left bin (M1).
M6 |q(t)| at each take and the fraction unwound -sign(q) dq / |q| by type and rank; beside it |q| over every q != 0 maker-second of the scope.
M7 unwind edge at reduce event-seconds vs every q != 0 maker-second; each take's price vs p_fair at t: 100 s (p_fair - p_avg) cents
  (positive = traded better than fair for the maker), by type.
VERDICTS (computed; claims.json): insufficient_support (< 5 supported markets or < 30 event-seconds in strata with exposed and reference
  person-time) | supported (RR > 1, p_hi <= 0.05, RR > 1 in >= 2/3 of the supported markets) | inconsistent (p_hi <= 0.05, share < 2/3) |
  not_supported; timing_sensitive when the D0 / D1000 / pend verdicts differ. Mechanisms (reduce): inventory_size_driven RR_size, stop_loss
  RR_against_5s, take_profit RR_with_5s, end_of_market_flattening RR_end AND share flattened >= 0.5 (a supported RR_end with a smaller share
  reads not_supported, an undefined share insufficient_support), burst_driven RR_burst, edge_driven RR_edge; (add) directional RR_with_5s.
  7 mechanisms x 2 scopes, all listed, none selected.
OUTPUTS write-once (a partial directory renamed at the end): hazard_cells.csv, ratios.csv, consistency.csv, patterns.csv, end_of_market.csv,
  inventory_at_take.csv, edge.csv, claims.json, receipt.json (sha256 of the sources, of every input read and of every output written; CPU by
  stage; VmHWM = peak RSS). run -> v2/chains/take_logic; run --pilot <day> --out <dir> -> the pilot (prints the full-run CPU projection).
GUARDS MT_RUN=v2; measurement days only; MT_TL_MUT (the selftest's planted errors flip_align / leak_1s / active_span) is refused in
  production; the full run refuses unless the recorded pilot projection is <= 60 CPU s and the tool, its selftest, the contract,
  chain_context.py and chain_take.py are tracked and unmodified (the freeze); fills_valued sha256 must equal the accepted binding and
  chain_take's receipt, and every input both receipts name must agree (else refused: mismatched artifacts); VmHWM self-guard 1,450 MiB
  (pf_mt_common.GUARD_MIB, never raised); CPU self-guard.
Usage (run_light, MT_RUN=v2): take_logic.py run | take_logic.py run --pilot 20261024 --out <dir>      selftest: take_logic_selftest.py"""
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pf_mt_common as C  # noqa: E402
import chain_context as CX  # noqa: E402
import chain_take as CT  # noqa: E402

X = CT.X
need, Forbidden = C.need, C.Forbidden
NAN, INF = float('nan'), float('inf')
VERSION = 'take_logic_v1'
CONTRACT_FILE = C.RECEIPT_V1 / 'take_logic_contract.md'
CONTRACT_COMMIT = '3d9c6b082'
TOOL_FILES = (HERE / 'take_logic.py', HERE / 'take_logic_selftest.py', CONTRACT_FILE, HERE / 'chain_context.py', HERE / 'chain_take.py')
DEP_FILES = (HERE / 'closes.py', RESEARCH / 'pf_mt_common.py', RESEARCH / 'pf_mt_econ.py', RESEARCH / 'pf_mt_lifecycle.py', RESEARCH / 'pf_mt_decision.py')
TOP3 = CT.TOP3
RANKS = (1, 2, 3)
N_BOOK = CT.N_BOOK
PILOT_DAY = '20261024'

# ------------------------------------------------------------------ declared constants (the contract's; none fitted) ------------------------------------------------------------------
STEP_MS = 1_000                                  # one maker-second
D_SENS = 1_000                                   # sensitivity: the state at t - 1000 ms
VARIANTS = ('D0', 'D1000', 'pend')
SEED, N_SHIFT = 20261123, 500
POS_EPS = CT.POS_EPS                             # 1e-6 sh: |q| at or below is flat (chain_take's class epsilon)
Q_LAB = ('0', '(0,50]', '(50,200]', '(200,500]', '>500')
SIGN_LAB = ('flat', 'long', 'short')
CAT_LAB = ('with', 'against', 'flat', 'missing')
HZ = (('1s', 'r_1s'), ('5s', 'r_5s'), ('30s', 'r_30s'))
THR = {k: float(CX.MOVE_THR[k]) for _, k in HZ}  # 0.5 / 1.0 / 2.0 bp
DOSE_LAB = ('<=-5', '(-5,-2]', '(-2,-1]', '(-1,1)', '[1,2)', '[2,5)', '>=5', 'missing')
TL_LAB = ('>300', '(240,300]', '(120,240]', '(60,120]', '(30,60]', '(10,30]', '[0,10]')
F5_LAB = ('0', '1-2', '>=3')
EDGE_LAB = ('<=-2', '(-2,-0.5]', '(-0.5,0.5)', '[0.5,2)', '>=2', 'missing', 'no_inventory')
BTC_LAB = tuple(str(x) for x in CX.BTC_CODE)    # ok / btc_missing / btc_invalid / after_we
FILL_WIN_MS = 5_000
END_OFFS_MS = (0, 30_000, 60_000)                # END, END - 30 s, END - 60 s
FLAT_FRAC, FLAT_SHARE_MIN = 0.1, 0.5
MIN_EVENTS, MIN_MARKETS, MIN_MKT_S = 30, 5, 30
P_MAX = 0.05
C_HI_MAX = int(np.floor(P_MAX * (N_SHIFT + 1) + 1e-9))   # p_hi <= 0.05  <=>  1 + #{null >= obs} <= 25
SHARE_NUM, SHARE_DEN = 2, 3
TIE_REL = 1e-12
TYPES = ('reduce', 'add', 'other', 'timing_sensitive')
TYPES_ANY = TYPES + ('any',)
M2_TYPES = ('reduce', 'add')
CLASS_COL = 'take_class'
CLASS_TYPE = dict(reduce_partial='reduce', full_close='reduce', reversal_reducing='reduce', overshoot='reduce', add='add', from_flat='add',
                  no_change='other', unclassified='other', timing_sensitive='timing_sensitive')
EXPECT_CLS = dict(reduce_partial=2065, add=1185, timing_sensitive=838, no_change=38, reversal_reducing=8, overshoot=3, from_flat=10, full_close=0,
                  unclassified=0)
N_TAKES_ALL = 4147
TAKE_READ = ('day', 'mid', 'rank', 'addr', 'sid', 's', 'dq', 'p_avg', 'x_lo_hard_ms', 'hi_ms', 'settle_ms', 'status', CLASS_COL)
TAKE_NEED = ('day', 'mid', 'rank', 'addr', 's', 'dq', 'p_avg', 'x_lo_hard_ms', CLASS_COL)
DEFS = ('btc_1s', 'btc_5s', 'btc_30s', 'size', 'end', 'burst', 'edge')
STRATA_LAB = dict(btc_1s='|q| bin x time-left bin x rank', btc_5s='|q| bin x time-left bin x rank', btc_30s='|q| bin x time-left bin x rank',
                  size='a_5s category x time-left bin x rank', end='|q| bin x a_5s category x rank', burst='|q| bin x time-left bin x rank',
                  edge='|q| bin x time-left bin x rank')
RATIOS = (('RR_against_5s', 'btc_5s', 1, 'a_5s against', 'a_5s flat'), ('RR_with_5s', 'btc_5s', 2, 'a_5s with', 'a_5s flat'),
          ('RR_size', 'size', 1, '|q| > 200', '|q| in (0,50]'), ('RR_end', 'end', 1, 'time left <= 30 s', 'time left in (60,240] s'),
          ('RR_burst', 'burst', 1, 'own fills 5 s >= 3', 'own fills 5 s = 0'), ('RR_edge', 'edge', 1, 'edge >= 0.5 c', 'edge in (-0.5,0.5) c'),
          ('RR_against_1s', 'btc_1s', 1, 'a_1s against', 'a_1s flat'), ('RR_with_1s', 'btc_1s', 2, 'a_1s with', 'a_1s flat'),
          ('RR_against_30s', 'btc_30s', 1, 'a_30s against', 'a_30s flat'), ('RR_with_30s', 'btc_30s', 2, 'a_30s with', 'a_30s flat'))
PRIMARY = ('RR_against_5s', 'RR_with_5s', 'RR_size', 'RR_end', 'RR_burst', 'RR_edge')
MECHANISMS = (('inventory_size_driven', 'reduce', 'RR_size'), ('stop_loss', 'reduce', 'RR_against_5s'), ('take_profit', 'reduce', 'RR_with_5s'),
              ('end_of_market_flattening', 'reduce', 'RR_end'), ('burst_driven', 'reduce', 'RR_burst'), ('edge_driven', 'reduce', 'RR_edge'),
              ('directional', 'add', 'RR_with_5s'))
MECH_OF = {(t, r): m for m, t, r in MECHANISMS}
M2_SCOPES = ('rank1', 'pooled')
SCOPES = ('rank1', 'rank2', 'rank3', 'pooled')
PATTERNS = (('P1', 'sell reducing a long while a_5s with (sells when BTC rises)'), ('P2', 'sell reducing a long while a_5s against'),
            ('P3', 'buy reducing a short while a_5s with'), ('P4', 'buy reducing a short while a_5s against'), ('P5', 'reduce while a_5s flat (q != 0)'),
            ('P6', 'add while BTC 5 s moves with the take side'), ('P7', 'any take with time left <= 30 s'), ('P8', 'reduce with time left <= 30 s'))
QUANTS = (0.10, 0.25, 0.50, 0.75, 0.90)
MUTANTS = ('flip_align', 'leak_1s', 'active_span')
CPU_CAP_RUN_S, CPU_CAP_PILOT_S = 90., 25.
PROJ_MAX_S = 60.
CSV_FILES = ('hazard_cells.csv', 'ratios.csv', 'consistency.csv', 'patterns.csv', 'end_of_market.csv', 'inventory_at_take.csv', 'edge.csv')
OUT_FILES = CSV_FILES + ('claims.json', 'receipt.json')
MS_FIELDS = tuple(f'{nm}{d}' for d in '01' for nm in ('r_1s', 'r_5s', 'r_30s', 'pf', 'bok', 'bst', 'bid', 'ask', 'kok', 'tl'))
PS_FIELDS = ('q0', 'q1', 'fl0', 'fl1', 'qp', 'flp', 'ts', 'lg')


# ------------------------------------------------------------------ small helpers ------------------------------------------------------------------
def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(C.REPO))
    except ValueError:
        return str(p)


def paths():
    """Run-time paths on the MT_RUN roots (v2: C.RECEIPT = the receipt's v2/ directory)."""
    ch = C.RECEIPT / 'chains'
    tk = ch / 'take'
    return dict(takes=tk / 'takes.csv', summary=tk / 'summary.csv', take_receipt=tk / 'receipt.json', out=ch / 'take_logic',
                pilot_receipt=ch / f'take_logic_pilot_{PILOT_DAY}_r1' / 'receipt.json')


def pilot_out(day):
    return C.RECEIPT / 'chains' / f'take_logic_pilot_{day}_r1'


def mutant():
    """The selftest's planted-error switch MT_TL_MUT, read at call time; production refuses any value."""
    m = os.environ.get('MT_TL_MUT', '')
    need(m == '' or m in MUTANTS, f'MT_TL_MUT={m!r} is not a declared mutant {MUTANTS} (refused)')
    return m


def _scope_mask(rank, sc):
    rank = np.asarray(rank)
    return np.ones(len(rank), bool) if sc == 'pooled' else (rank == int(sc[4:]))


def _qs(x, pre):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    d = {f'{pre}_n': int(len(x)), f'{pre}_mean': float(x.mean()) if len(x) else NAN}
    qv = np.quantile(x, QUANTS) if len(x) else [NAN] * len(QUANTS)
    for q, v in zip(QUANTS, qv):
        d[f'{pre}_p{int(round(q * 100)):02d}'] = float(v)
    return d


def _shares(code, labs, pre):
    c = np.asarray(code, np.int64)
    b = np.bincount(c, minlength=len(labs)) if len(c) else np.zeros(len(labs))
    return {f'{pre}_{lb}': (float(b[i]) / len(c) if len(c) else NAN) for i, lb in enumerate(labs)}


# ------------------------------------------------------------------ bins and categories (contract 'State at t') ------------------------------------------------------------------
def qbin(q):
    a = np.abs(np.asarray(q, float))
    return np.select([~(a > POS_EPS), a <= 50., a <= 200., a <= 500.], [0, 1, 2, 3], 4).astype(np.int8)


def qsign(q):
    q = np.asarray(q, float)
    return np.where(q > POS_EPS, 1, np.where(q < -POS_EPS, -1, 0)).astype(np.int8)


def sign_code(sg):
    sg = np.asarray(sg)
    return np.where(sg > 0, 1, np.where(sg < 0, 2, 0)).astype(np.int8)


def tl_codes(tl_s):
    tl = np.asarray(tl_s, float)
    need(bool((tl >= 0).all()), 'time left < 0 at a population instant (refused)')
    return np.select([tl > 300, tl > 240, tl > 120, tl > 60, tl > 30, tl > 10], [0, 1, 2, 3, 4, 5], 6).astype(np.int8)


def f5_codes(n):
    n = np.asarray(n)
    return np.select([n <= 0, n <= 2], [0, 1], 2).astype(np.int8)


def dose_codes(a):
    a = np.asarray(a, float)
    with np.errstate(invalid='ignore'):
        return np.select([np.isnan(a), a <= -5, a <= -2, a <= -1, a < 1, a < 2, a < 5], [7, 0, 1, 2, 3, 4, 5], 6).astype(np.int8)


def align_codes(s, r, thr):
    """chain_context.align_cat(s, r, thr) -> 0 with / 1 against / 2 flat / 3 missing (unknown)."""
    s, r = np.asarray(s, float), np.asarray(r, float)
    if len(r) == 0:
        return np.zeros(0, np.int8)
    lab = np.asarray(CX.align_cat(s, r, thr), dtype=object)
    return np.select([lab == 'with', lab == 'against', lab == 'flat'], [0, 1, 2], 3).astype(np.int8)


def edge_values(sg, bid, ask, pf, kok, bok):
    """Unwind edge (cents): long 100 (bid - p_fair), short 100 (p_fair - ask); valid with book_tob ok and BTC ok; -> (edge, bin code)."""
    sg = np.asarray(sg)
    with np.errstate(invalid='ignore'):
        e = np.where(sg > 0, 100. * (np.asarray(bid, float) - np.asarray(pf, float)),
                     np.where(sg < 0, 100. * (np.asarray(pf, float) - np.asarray(ask, float)), NAN))
        ok = np.asarray(kok, bool) & np.asarray(bok, bool) & (sg != 0) & np.isfinite(e)
        code = np.select([sg == 0, ~ok, e <= -2, e <= -0.5, e < 0.5, e < 2], [6, 5, 0, 1, 2, 3], 4).astype(np.int8)
    return np.where(ok, e, NAN), code


# ------------------------------------------------------------------ states (one market, every address; chain_context only) ------------------------------------------------------------------
def pair_state(M, addr, Xs, cc=CX):
    """Per evaluation instant x: q (status-ok legs with hi <= x), qp ('possible': x_lo_hard <= x), fl (own maker fills, hi in (x - 5 s, x]),
    flp (fl + pending own maker legs), ts (inv_timing_sensitive), lg (inv_ledger_conditional)."""
    t1 = np.asarray(Xs, np.int64) + 1
    p = cc.position(M, addr, t1, clock='hi')
    pe = cc.pending(M, addr, t1)
    q = np.asarray(p['pos'], float)
    need(np.array_equal(np.asarray(pe['certain'], float), q), f'{addr}: chain_context.pending certain differs from position (refused)')
    G = M.legs[addr.lower()]
    if len(G):
        mk = (np.asarray(G['role'], dtype=object) == 'maker') & ~np.asarray(G['self_trade'].fillna(False), dtype=bool)
        hi = np.sort(G['hi_ms'].to_numpy(np.int64)[mk])
        lo = np.sort(G['lo_hard_ms'].to_numpy(np.int64)[mk])           # -1 (unbounded) sorts first: always started
    else:
        hi = lo = np.zeros(0, np.int64)
    x = np.asarray(Xs, np.int64)
    n_hi = np.searchsorted(hi, x, 'right')
    fl = n_hi - np.searchsorted(hi, x - FILL_WIN_MS, 'right')
    flp = fl + (np.searchsorted(lo, x, 'right') - n_hi)
    pmin, pmax = np.asarray(pe['pos_min'], float), np.asarray(pe['pos_max'], float)
    qb, sg = qbin(q), qsign(q)
    ts = (qbin(pmin) != qb) | (qbin(pmax) != qb) | (qsign(pmin) != sg) | (qsign(pmax) != sg)
    lg = np.asarray(p['status'], dtype=object) == 'excluded_leg_possible'
    return dict(q=q, qp=np.asarray(p['possible'], float), fl=fl.astype(np.int32), flp=flp.astype(np.int32), ts=np.asarray(ts, bool), lg=np.asarray(lg, bool))


def market_part(day, mid, M, ranks=RANKS, cc=CX):
    """One market's per-instant states for the addresses of `ranks`: the market's BTC / book / time left once, then every address."""
    mut = mutant()
    ws, we = int(cc.market_ws(M)), int(cc.market_we(M))
    K = (we - ws) // STEP_MS
    need(K >= 1, f'market {day} {mid}: window shorter than one second (refused)')
    T = ws + STEP_MS * np.arange(K, dtype=np.int64)
    ends = we - np.asarray(END_OFFS_MS, np.int64)
    lk = STEP_MS if mut == 'leak_1s' else 0          # planted error: every read one second LATER
    Xs = np.concatenate([T, T - D_SENS, ends, ends - D_SENS]).astype(np.int64) + lk
    S0, S1 = slice(0, K), slice(K, 2 * K)
    E0, E1 = slice(2 * K, 2 * K + 3), slice(2 * K + 3, 2 * K + 6)
    B = cc.btc_ret(M, Xs)
    Bk = cc.book_tob(M, Xs)
    tl = np.asarray(cc.time_left(M, Xs), np.int64) / 1000.
    code = np.asarray(B['code'], np.int64)
    kcode = np.asarray(Bk['code'], np.int64)
    ms = {}
    for d, sl in (('0', S0), ('1', S1)):
        for nm in ('r_1s', 'r_5s', 'r_30s'):
            ms[nm + d] = np.asarray(B[nm], float)[sl]
        ms['pf' + d] = np.asarray(B['p_fair'], float)[sl]
        ms['bok' + d] = (code == 0)[sl]
        ms['bst' + d] = code[sl].astype(np.int8)
        ms['bid' + d] = np.asarray(Bk['best_bid'], float)[sl]
        ms['ask' + d] = np.asarray(Bk['best_ask'], float)[sl]
        ms['kok' + d] = (kcode == 0)[sl]
        ms['tl' + d] = tl[sl]
    pairs = []
    for rk in ranks:
        a = TOP3[rk - 1]
        st = pair_state(M, a, Xs, cc)
        G = M.legs[a.lower()]
        present = np.ones(K, bool)
        if mut == 'active_span':                      # planted error: the maker's active span instead of the market's whole span
            h = G['hi_ms'].to_numpy(np.int64) if len(G) else np.zeros(0, np.int64)
            present = ((T >= h.min()) & (T <= h.max())) if len(h) else np.zeros(K, bool)
        ps = dict(q0=st['q'][S0], q1=st['q'][S1], fl0=st['fl'][S0], fl1=st['fl'][S1], qp=st['qp'][S0], flp=st['flp'][S0], ts=st['ts'][S0], lg=st['lg'][S0])
        end = dict(D0=st['q'][E0].copy(), D1000=st['q'][E1].copy(), pend=st['qp'][E0].copy())
        pairs.append(dict(rank=int(rk), addr=a, present=present, st=ps, end=end, n_legs=int(len(G))))
    return dict(market=dict(day=str(day), mid=int(mid), ws=ws, we=we, K=int(K), T=T), ms=ms, pairs=pairs)


def assemble(parts):
    """Market parts -> the panel P: per-instant arrays over every pair, pair / market metadata and the declared shift offsets."""
    keys = ('pair', 'rank', 'mkt', 'k', 't') + MS_FIELDS + PS_FIELDS
    cols = {k: [] for k in keys}
    pairs, mks, U, pn, pb = [], [], [], [], []
    base = 0
    for mi, part in enumerate(parts):
        mk = part['market']
        mks.append({k: v for k, v in mk.items() if k != 'T'})
        for pr in part['pairs']:
            pres = np.asarray(pr['present'], bool)
            n, pid = int(pres.sum()), len(pairs)
            gidx = np.full(mk['K'], -1, np.int64)
            gidx[pres] = base + np.arange(n, dtype=np.int64)
            cols['pair'].append(np.full(n, pid, np.int64))
            cols['rank'].append(np.full(n, pr['rank'], np.int64))
            cols['mkt'].append(np.full(n, mi, np.int64))
            cols['k'].append(np.arange(n, dtype=np.int64))
            cols['t'].append(mk['T'][pres])
            for f in MS_FIELDS:
                cols[f].append(part['ms'][f][pres])
            for f in PS_FIELDS:
                cols[f].append(pr['st'][f][pres])
            ss = np.random.SeedSequence([SEED, int(mk['day']), int(mk['mid']), int(pr['rank'])])
            U.append(np.random.Generator(np.random.PCG64(ss)).integers(0, n, size=N_SHIFT) if n else np.zeros(N_SHIFT, np.int64))
            pairs.append(dict(pid=pid, day=mk['day'], mid=mk['mid'], rank=pr['rank'], addr=pr['addr'], ws=mk['ws'], we=mk['we'], K=mk['K'], n=n,
                              base=base, gidx=gidx, end=pr['end'], n_legs=pr['n_legs'], mkt=mi))
            pn.append(n)
            pb.append(base)
            base += n
    need(len(pairs) > 0, 'no market part (refused)')
    P = {k: np.concatenate(v) for k, v in cols.items()}
    P.update(N=int(base), pairs=pairs, markets=mks, U=np.vstack(U).astype(np.int64), pair_n=np.asarray(pn, np.int64), pair_base=np.asarray(pb, np.int64))
    days = sorted({m['day'] for m in mks})
    P['days'] = days
    P['mday'] = np.array([days.index(m['day']) for m in mks], np.int64)
    return P


def population_checks(P, expect_pairs=None):
    """Computed predicates: every pair holds every instant ws .. we - 1000 on the 1 s grid; a pair without status-ok legs sits at q = 0."""
    K_tot = int(sum(pm['K'] for pm in P['pairs']))
    every = all(pm['n'] == pm['K'] for pm in P['pairs'])
    grid = every and all(np.array_equal(P['t'][pm['base']:pm['base'] + pm['n']], pm['ws'] + STEP_MS * np.arange(pm['K'], dtype=np.int64))
                         for pm in P['pairs'])
    nofill = [pm for pm in P['pairs'] if pm['n_legs'] == 0]
    nf_zero = all(pm['n'] == pm['K'] and not bool(np.any(np.abs(P['q0'][pm['base']:pm['base'] + pm['n']]) > POS_EPS)) for pm in nofill)
    pairs_ok = expect_pairs is None or len(P['pairs']) == int(expect_pairs)
    return dict(pairs=len(P['pairs']), expected_pairs=expect_pairs, maker_seconds=int(P['N']), expected_maker_seconds=K_tot,
                every_pair_every_instant=bool(every), instants_on_grid=bool(grid), no_fill_pairs=len(nofill), no_fill_pairs_at_zero=bool(nf_zero),
                complete=bool(every and grid and nf_zero and pairs_ok and int(P['N']) == K_tot))


# ------------------------------------------------------------------ takes -> events ------------------------------------------------------------------
def check_reproduction(T, S):
    """take_class must reproduce chain_take's summary counts (the dispatch numbers and summary.csv pooled / all), else refused."""
    need(len(T) == N_TAKES_ALL, f'takes.csv has {len(T)} rows, not {N_TAKES_ALL} (refused)')
    cnt = Counter(str(x) for x in np.asarray(T[CLASS_COL], dtype=object))
    extra = sorted(set(cnt) - set(CT.CLS))
    need(not extra, f'{CLASS_COL}: classes outside chain_take.CLS {extra} (refused)')
    got = {c: int(cnt.get(c, 0)) for c in CT.CLS}
    need(all(got[c] == v for c, v in EXPECT_CLS.items()), f'{CLASS_COL} does not reproduce the summary counts {EXPECT_CLS}: {got} (refused)')
    row = S[(S['scope'].astype(str) == 'pooled') & (S['day'].astype(str) == 'all')]
    need(len(row) == 1, 'summary.csv: no single pooled / all row (refused)')
    srow = {c: int(row[f'cls_{c}'].iat[0]) for c in CT.CLS}
    need(srow == got, f'{CLASS_COL} does not reproduce summary.csv pooled / all cls_*: {got} vs {srow} (refused)')
    return dict(class_column=CLASS_COL, counts=got, dispatch_expected=dict(EXPECT_CLS), summary_pooled_all=srow, reproduces=True)


def prepare_takes(T):
    """Validated takes with their event type (contract table)."""
    for c in TAKE_NEED:
        need(c in T.columns, f'takes: required column {c} missing (refused)')
    T = T.reset_index(drop=True).copy()
    T['day'] = [str(x) for x in np.asarray(T['day'], dtype=object)]
    cls = [str(x) for x in np.asarray(T[CLASS_COL], dtype=object)]
    bad = sorted(set(cls) - set(CLASS_TYPE))
    need(not bad, f'takes: unknown take class(es) {bad} (refused)')
    T['type'] = [CLASS_TYPE[c] for c in cls]
    rk = T['rank'].to_numpy(np.int64)
    need(bool(np.isin(rk, RANKS).all()), 'takes: a rank outside 1..3 (refused)')
    low = np.array([str(a).lower() for a in np.asarray(T['addr'], dtype=object)], dtype=object)
    want = np.array([TOP3[r - 1].lower() for r in rk], dtype=object)
    need(bool((low == want).all()), "takes: an address differs from its rank's top-3 address (refused)")
    need(bool(np.isin(T['s'].to_numpy(np.int64), (-1, 1)).all()), 'takes: a side outside +-1 (refused)')
    return T


def events(P, T):
    """Every take -> its instant (t < x_lo_hard_ms <= t + 1000) or a status; EV: E_<type> event-second indicators, N_<type> take counts."""
    T = T.reset_index(drop=True).copy()
    n = len(T)
    gi, st = np.full(n, -1, np.int64), np.full(n, 'no_pair', dtype=object)
    lut = {(pm['day'], pm['mid'], pm['rank']): pm for pm in P['pairs']}
    day = np.asarray(T['day'], dtype=object)
    mid, rk, xlo = T['mid'].to_numpy(np.int64), T['rank'].to_numpy(np.int64), T['x_lo_hard_ms'].to_numpy(np.int64)
    grp = {}
    for i in range(n):
        grp.setdefault((str(day[i]), int(mid[i]), int(rk[i])), []).append(i)
    for key, ix in grp.items():
        pm = lut.get(key)
        if pm is None:
            continue
        ix = np.asarray(ix, np.int64)
        x = xlo[ix]
        k = (x - pm['ws'] - 1) // STEP_MS
        unb = x < 0
        bef = ~unb & (k < 0)
        aft = ~unb & (k >= pm['K'])
        ins = ~unb & ~bef & ~aft
        g = np.full(len(ix), -1, np.int64)
        g[ins] = pm['gidx'][k[ins]]
        gi[ix] = g
        s_ = np.full(len(ix), 'outside_active_span', dtype=object)     # priority (last assignment wins): unbounded > before > after > assigned
        s_[g >= 0] = 'assigned'
        s_[aft] = 'after_population'
        s_[bef] = 'before_population'
        s_[unb] = 'unbounded_window'
        st[ix] = s_
    T['gi'], T['pop_status'] = gi, st
    asg = st == 'assigned'
    typ = np.asarray(T['type'], dtype=object)
    EV = {}
    for t_ in TYPES_ANY:
        m = asg & (typ == t_) if t_ != 'any' else asg
        cnt = np.bincount(gi[m], minlength=P['N']).astype(np.int64)
        need(len(cnt) == P['N'], 'an event outside the panel (refused)')
        EV['N_' + t_], EV['E_' + t_] = cnt, (cnt > 0).astype(np.int64)
    return T, EV


def shifted(P, idx):
    """The global instant of each event-second idx under every circular shift (its own pair's instants only) -> (N_SHIFT, n_e)."""
    idx = np.asarray(idx, np.int64)
    if len(idx) == 0:
        return np.zeros((N_SHIFT, 0), np.int64)
    p = P['pair'][idx]
    j = P['k'][idx]
    n = P['pair_n'][p]
    b = P['pair_base'][p]
    return np.ascontiguousarray((b[:, None] + (j[:, None] + P['U'][p]) % n[:, None]).T)


# ------------------------------------------------------------------ state codes per variant ------------------------------------------------------------------
def variant_codes(P, v):
    """Per-instant state codes of variant v: D0 (at t), D1000 (every read at t - 1000), pend (pending legs included)."""
    need(v in VARIANTS, f'variant {v!r} not in {VARIANTS}')
    d = '1' if v == 'D1000' else '0'
    q = P['qp'] if v == 'pend' else P['q' + d]
    fl = P['flp'] if v == 'pend' else P['fl' + d]
    sg = qsign(q)
    s_al = (-sg if mutant() == 'flip_align' else sg).astype(float)     # planted error: the alignment sign flipped
    out = dict(q=np.asarray(q, float), sg=sg, qb=qbin(q), sc=sign_code(sg), tl=tl_codes(P['tl' + d]), f5b=f5_codes(fl), bst=P['bst' + d].astype(np.int64))
    for h, nm in HZ:
        out['cat_' + h] = align_codes(s_al, P[nm + d], THR[nm])
    with np.errstate(invalid='ignore'):
        out['a5'] = s_al * P['r_5s' + d]
    out['dose'] = dose_codes(out['a5'])
    out['edge'], out['edgeb'] = edge_values(sg, P['bid' + d], P['ask' + d], P['pf' + d], P['kok' + d], P['bok' + d])
    return out


# ------------------------------------------------------------------ M2 / M3 / M4: Mantel-Haenszel ratios, the shift null, market consistency ------------------------------------------------------------------
def def_codes(rank, Cv, name):
    """Exposure group (-1 excluded; 0 = reference) and stratum (rank outermost) of definition `name` -> (code = stratum * G + group or -1,
    group, S strata, G groups). q != 0 only."""
    rk = np.asarray(rank, np.int64) - 1
    qb, tl, c5 = Cv['qb'].astype(np.int64), Cv['tl'].astype(np.int64), Cv['cat_5s'].astype(np.int64)
    if name.startswith('btc_'):
        cat = Cv['cat_' + name[4:]].astype(np.int64)
        grp = np.select([cat == 2, cat == 1, cat == 0], [0, 1, 2], -1)          # flat ref / against / with
        strat, S, G = (rk * 4 + (qb - 1)) * 7 + tl, 84, 3
    elif name == 'size':
        grp = np.select([qb >= 3, qb == 1], [1, 0], -1)
        strat, S, G = (rk * 4 + c5) * 7 + tl, 84, 2
    elif name == 'end':
        grp = np.select([tl >= 5, (tl == 2) | (tl == 3)], [1, 0], -1)
        strat, S, G = (rk * 4 + (qb - 1)) * 4 + c5, 48, 2
    elif name == 'burst':
        f = Cv['f5b'].astype(np.int64)
        grp = np.select([f == 2, f == 0], [1, 0], -1)
        strat, S, G = (rk * 4 + (qb - 1)) * 7 + tl, 84, 2
    elif name == 'edge':
        e = Cv['edgeb'].astype(np.int64)
        grp = np.select([(e == 3) | (e == 4), e == 2], [1, 0], -1)
        strat, S, G = (rk * 4 + (qb - 1)) * 7 + tl, 84, 2
    else:
        raise ValueError(f'unknown ratio definition {name!r}')
    grp = np.where(qb > 0, grp, -1).astype(np.int64)
    code = np.where(grp >= 0, strat * G + grp, -1).astype(np.int64)
    need(bool((code < S * G).all()), f'{name}: a stratum code outside its range (refused)')
    return code, grp, S, G


def stratum_counts(code, S, G, idx):
    """Person-time (S, G) of the coded instants and the event-seconds idx per (stratum, group)."""
    code = np.asarray(code, np.int64)
    pt = np.bincount(code[code >= 0], minlength=S * G).reshape(S, G).astype(float)
    co = code[np.asarray(idx, np.int64)]
    obs = np.bincount(co[co >= 0], minlength=S * G).reshape(S, G).astype(float)
    return pt, obs


def null_counts(code, S, G, SH):
    """Event-seconds per (replicate, stratum, group) under the shifts SH (N_SHIFT, n_e)."""
    cn = np.asarray(code, np.int64)[SH]
    okn = cn >= 0
    ri = np.nonzero(okn)[0].astype(np.int64)
    return np.bincount(ri * (S * G) + cn[okn], minlength=N_SHIFT * S * G).reshape(N_SHIFT, S, G).astype(float)


def mh_rr(A, B, T1, T0):
    """Mantel-Haenszel person-time RR per row: A, B (R, S) exposed / reference events, T1, T0 (S,) person-time; strata with T1 > 0 and T0 > 0
    only -> (RR (R,), num, den); +inf = exposed events with no reference events, NaN = no event."""
    A, B = np.atleast_2d(np.asarray(A, float)), np.atleast_2d(np.asarray(B, float))
    T1, T0 = np.asarray(T1, float), np.asarray(T0, float)
    inf_ = (T1 > 0) & (T0 > 0)
    Ts = np.where(inf_, T1 + T0, 1.)
    w1, w0 = np.where(inf_, T0 / Ts, 0.), np.where(inf_, T1 / Ts, 0.)
    num = np.ascontiguousarray(A * w1[None, :]).sum(1)
    den = np.ascontiguousarray(B * w0[None, :]).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        rr = np.where(den > 0, num / np.where(den > 0, den, 1.), np.where(num > 0, INF, NAN))
    return rr, num, den


def tail_counts(obs, null):
    """(1 + #null >= obs, 1 + #null <= obs); ties within TIE_REL count; NaN replicates count on both sides (conservative); (None, None) for a
    NaN observation."""
    null = np.asarray(null, float)
    if obs is None or np.isnan(obs):
        return None, None
    nan = int(np.isnan(null).sum())
    with np.errstate(invalid='ignore'):
        if np.isinf(obs):
            hi, lo = int((null == INF).sum()), int((~np.isnan(null)).sum())
        else:
            tol = TIE_REL * max(abs(obs), 1e-300)
            hi, lo = int((null >= obs - tol).sum()), int((null <= obs + tol).sum())
    return 1 + hi + nan, 1 + lo + nan


def verdict(rr, c_hi, n_events, n_sup, n_gt1):
    """The contract's verdict rule (computed, never printed beside a number)."""
    if n_sup < MIN_MARKETS or n_events < MIN_EVENTS:
        return 'insufficient_support'
    sig = c_hi is not None and int(c_hi) <= C_HI_MAX
    share_ok = SHARE_DEN * int(n_gt1) >= SHARE_NUM * int(n_sup)
    if bool(rr > 1) and sig and share_ok:
        return 'supported'
    if sig and not share_ok:
        return 'inconsistent'
    return 'not_supported'


def end_combine(v_rr, share):
    """end_of_market_flattening = RR_end AND share flattened >= 0.5."""
    if share is None or not np.isfinite(share):
        return 'insufficient_support'
    if v_rr == 'supported' and share < FLAT_SHARE_MIN:
        return 'not_supported'
    return v_rr


def market_rr(grp, g, rmask, mkt, n_mk, Ev):
    """Per market (the scope's ranks): crude exposed / reference event-seconds and person-time, the crude RR, supported (>= 1 event-second,
    >= 30 exposed and >= 30 reference maker-seconds) and RR > 1."""
    sel = rmask & ((grp == 0) | (grp == g))
    key = mkt[sel].astype(np.int64) * 2 + (grp[sel] == g).astype(np.int64)
    Tm = np.bincount(key, minlength=2 * n_mk).reshape(n_mk, 2).astype(float)
    Em = np.bincount(key, weights=np.asarray(Ev, float)[sel], minlength=2 * n_mk).reshape(n_mk, 2)
    T0, T1, b, a = Tm[:, 0], Tm[:, 1], Em[:, 0], Em[:, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        rr = np.where(b > 0, (a / np.where(T1 > 0, T1, 1.)) / (b / np.where(T0 > 0, T0, 1.)), np.where(a > 0, INF, NAN))
    sup = (a + b >= 1) & (T1 >= MIN_MKT_S) & (T0 >= MIN_MKT_S)
    with np.errstate(invalid='ignore'):
        gt1 = sup & (rr > 1)
    return dict(a=a, T1=T1, b=b, T0=T0, rr=rr, sup=sup, gt1=gt1)


def ratio_tables(P, codes, EV):
    """Every M2 ratio x type x scope x variant: the MH RR, its shift null (M3) and market consistency (M4) -> (row dicts keyed (variant, scope,
    type, ratio), consistency rows)."""
    ev_idx = {t: np.flatnonzero(EV['E_' + t]) for t in M2_TYPES}
    SH = {t: shifted(P, ev_idx[t]) for t in M2_TYPES}
    rank, mkt = P['rank'].astype(np.int64), P['mkt'].astype(np.int64)
    n_mk, mday, days = len(P['markets']), P['mday'], P['days']
    RK, cons = {}, []
    for v in VARIANTS:
        Cv = codes[v]
        for dn in DEFS:
            code, grp, S, G = def_codes(rank, Cv, dn)
            for ty in M2_TYPES:
                pt, obs = stratum_counts(code, S, G, ev_idx[ty])
                nul = null_counts(code, S, G, SH[ty])
                Ev = EV['E_' + ty].astype(float)
                for rn, dn_, g, exl, rfl in RATIOS:
                    if dn_ != dn:
                        continue
                    for sc in M2_SCOPES:
                        sm = (np.arange(S) < S // 3) if sc == 'rank1' else np.ones(S, bool)
                        T1, T0 = pt[sm, g], pt[sm, 0]
                        Aa = np.vstack([obs[sm, g][None, :], nul[:, sm, g]])
                        Bb = np.vstack([obs[sm, 0][None, :], nul[:, sm, 0]])
                        rr = mh_rr(Aa, Bb, T1, T0)[0]
                        o, nl = float(rr[0]), rr[1:]
                        c_hi, c_lo = tail_counts(o, nl)
                        inf_s = (T1 > 0) & (T0 > 0)
                        a_t, b_t, T1t, T0t = float(obs[sm, g].sum()), float(obs[sm, 0].sum()), float(T1.sum()), float(T0.sum())
                        n_ev = int(round(float((obs[sm, g] + obs[sm, 0])[inf_s].sum())))
                        rmask = _scope_mask(rank, sc)
                        mc = market_rr(grp, g, rmask, mkt, n_mk, Ev)
                        n_sup, n_gt1 = int(mc['sup'].sum()), int(mc['gt1'].sum())
                        with np.errstate(divide='ignore', invalid='ignore'):
                            crude = (a_t / T1t) / (b_t / T0t) if (T1t > 0 and T0t > 0 and b_t > 0) else (INF if (a_t > 0 and T1t > 0) else NAN)
                        fin = nl[np.isfinite(nl)]
                        qn = np.quantile(fin, [.05, .5, .95]) if len(fin) else [NAN] * 3
                        mech = MECH_OF.get((ty, rn), '')
                        RK[(v, sc, ty, rn)] = dict(
                            variant=v, scope=sc, type=ty, ratio=rn, definition=dn, primary=rn in PRIMARY, exposed=exl, reference=rfl, strata=STRATA_LAB[dn],
                            RR_MH=o, RR_crude=float(crude), a_exposed=a_t, T_exposed=T1t, b_reference=b_t, T_reference=T0t, n_strata=int(sm.sum()),
                            n_strata_informative=int(inf_s.sum()), n_events_informative=n_ev,
                            maker_seconds_q0_excluded=int((rmask & (Cv['qb'] == 0)).sum()),
                            maker_seconds_other_excluded=int((rmask & (Cv['qb'] > 0) & (grp != 0) & (grp != g)).sum()),
                            null_n=int(len(nl)), null_n_nan=int(np.isnan(nl).sum()), null_n_inf=int(np.isinf(nl).sum()),
                            null_mean=float(fin.mean()) if len(fin) else NAN, null_p05=float(qn[0]), null_p50=float(qn[1]), null_p95=float(qn[2]),
                            c_hi=-1 if c_hi is None else int(c_hi), c_lo=-1 if c_lo is None else int(c_lo),
                            p_hi=NAN if c_hi is None else c_hi / (N_SHIFT + 1), p_lo=NAN if c_lo is None else c_lo / (N_SHIFT + 1),
                            n_markets_supported=n_sup, n_markets_rr_gt1=n_gt1, market_share_rr_gt1=(n_gt1 / n_sup) if n_sup else NAN,
                            mechanism=mech, verdict_rule=verdict(o, c_hi, n_ev, n_sup, n_gt1) if mech else '', verdict='')
                        for mi in range(n_mk):
                            mm = P['markets'][mi]
                            cons.append(dict(row='market', variant=v, scope=sc, type=ty, ratio=rn, day=mm['day'], mid=mm['mid'], a=float(mc['a'][mi]),
                                             T1=float(mc['T1'][mi]), b=float(mc['b'][mi]), T0=float(mc['T0'][mi]), rr_crude=float(mc['rr'][mi]),
                                             supported=bool(mc['sup'][mi]), rr_gt1=bool(mc['gt1'][mi])))
                        for di, d in enumerate(days):
                            dm = mday == di
                            ns, ng = int(mc['sup'][dm].sum()), int(mc['gt1'][dm].sum())
                            cons.append(dict(row='summary', variant=v, scope=sc, type=ty, ratio=rn, day=d, n_supported=ns, n_rr_gt1=ng,
                                             share=(ng / ns) if ns else NAN))
                        cons.append(dict(row='summary', variant=v, scope=sc, type=ty, ratio=rn, day='all', n_supported=n_sup, n_rr_gt1=n_gt1,
                                         share=(n_gt1 / n_sup) if n_sup else NAN))
    return RK, cons


def flatten_shares(P):
    """M5 share flattened per scope and variant over the scope's pairs with |q(END - 60 s)| > 0."""
    out = {}
    for sc in SCOPES:
        prs = [pm for pm in P['pairs'] if sc == 'pooled' or pm['rank'] == int(sc[4:])]
        for v in VARIANTS:
            e = np.array([np.asarray(pm['end'][v], float) for pm in prs], float).reshape(-1, 3)
            base = np.abs(e[:, 2]) > POS_EPS
            fl = base & (np.abs(e[:, 0]) <= FLAT_FRAC * np.abs(e[:, 2]))
            n, k = int(base.sum()), int(fl.sum())
            out[(sc, v)] = dict(scope=sc, variant=v, n_pairs=len(prs), n_q60_nonzero=n, n_flattened=k, share=(k / n) if n else NAN)
    return out


def mechanism_claims(RK, flat):
    """7 mechanisms x 2 scopes: per variant the rule verdict and its predicates; the final verdict (timing_sensitive on any difference)."""
    out = []
    for mech, ty, rn in MECHANISMS:
        for sc in M2_SCOPES:
            ent = dict(mechanism=mech, scope=sc, type=ty, ratio=rn)
            vv = {}
            for v in VARIANTS:
                r = RK[(v, sc, ty, rn)]
                ns, ng, c_hi = r['n_markets_supported'], r['n_markets_rr_gt1'], r['c_hi']
                pred = dict(rr_gt_1=bool(r['RR_MH'] > 1), p_hi_le_0_05=bool(c_hi >= 0 and c_hi <= C_HI_MAX),
                            supported_markets_ge_5=bool(ns >= MIN_MARKETS), events_ge_30=bool(r['n_events_informative'] >= MIN_EVENTS),
                            share_rr_gt1_ge_2_3=bool(ns > 0 and SHARE_DEN * ng >= SHARE_NUM * ns))
                vm = r['verdict_rule']
                d = dict(RR_MH=r['RR_MH'], RR_crude=r['RR_crude'], p_hi=r['p_hi'], p_lo=r['p_lo'], n_events_informative=r['n_events_informative'],
                         n_markets_supported=ns, n_markets_rr_gt1=ng, market_share_rr_gt1=r['market_share_rr_gt1'], verdict_rule=vm, predicates=pred)
                if mech == 'end_of_market_flattening':
                    fs = flat[(sc, v)]
                    d.update(share_flattened=fs['share'], n_pairs_q60_nonzero=fs['n_q60_nonzero'], n_pairs_flattened=fs['n_flattened'])
                    pred['share_flattened_ge_0_5'] = bool(np.isfinite(fs['share']) and fs['share'] >= FLAT_SHARE_MIN)
                    vm = end_combine(vm, fs['share'])
                d['verdict'] = vm
                r['verdict'] = vm
                vv[v] = vm
                ent[v] = d
            ent['verdict'] = vv['D0'] if vv['D0'] == vv['D1000'] == vv['pend'] else 'timing_sensitive'
            ent['flips_at'] = [v for v in ('D1000', 'pend') if vv[v] != vv['D0']]
            out.append(ent)
    return out


# ------------------------------------------------------------------ M1, patterns, M5-M7 tables ------------------------------------------------------------------
def hazard_table(P, C0, EV):
    """M1: event-seconds / maker-seconds per cell and per factor marginal, per type (+ any), ranks 1-3 and pooled (D0)."""
    cell = ((C0['qb'].astype(np.int64) * 3 + C0['sc']) * 4 + C0['cat_5s']) * 7 + C0['tl']
    cell_lab = [f'{a}|{b}|{c}|{d}' for a in Q_LAB for b in SIGN_LAB for c in CAT_LAB for d in TL_LAB]
    G = [('cell', 'qbin|sign|a_5s|time_left', cell, cell_lab), ('marginal', 'qbin', C0['qb'], Q_LAB), ('marginal', 'sign', C0['sc'], SIGN_LAB),
         ('marginal', 'a_1s', C0['cat_1s'], CAT_LAB), ('marginal', 'a_5s', C0['cat_5s'], CAT_LAB), ('marginal', 'a_30s', C0['cat_30s'], CAT_LAB),
         ('marginal', 'a_5s_dose', C0['dose'], DOSE_LAB), ('marginal', 'time_left', C0['tl'], TL_LAB), ('marginal', 'fills_5s', C0['f5b'], F5_LAB),
         ('marginal', 'edge', C0['edgeb'], EDGE_LAB), ('marginal', 'inv_timing_sensitive', P['ts'].astype(np.int64), ('no', 'yes')),
         ('marginal', 'inv_ledger_conditional', P['lg'].astype(np.int64), ('no', 'yes')), ('marginal', 'btc_status', C0['bst'], BTC_LAB)]
    Ef = {t: EV['E_' + t].astype(float) for t in TYPES_ANY}
    Nf = {t: EV['N_' + t].astype(float) for t in TYPES_ANY}
    rows = []
    for sc in SCOPES:
        m = _scope_mask(P['rank'], sc)
        for table, factor, code, labs in G:
            c = np.asarray(code, np.int64)[m]
            L = len(labs)
            ms = np.bincount(c, minlength=L)
            need(len(ms) == L, f'hazard {factor}: a code outside its labels (refused)')
            es = {t: np.bincount(c, weights=Ef[t][m], minlength=L) for t in TYPES_ANY}
            tk = {t: np.bincount(c, weights=Nf[t][m], minlength=L) for t in TYPES_ANY}
            for i in np.flatnonzero(ms):
                r = dict(scope=sc, table=table, factor=factor, level=labs[i], maker_seconds=int(ms[i]))
                for t in TYPES_ANY:
                    r[f'event_seconds_{t}'] = int(round(es[t][i]))
                    r[f'takes_{t}'] = int(round(tk[t][i]))
                    r[f'hazard_{t}'] = float(es[t][i]) / float(ms[i])
                rows.append(r)
    return pd.DataFrame(rows)


def _assigned(T):
    return T[np.asarray(T['pop_status'], dtype=object) == 'assigned']


def patterns_table(P, C0, T):
    """M4 declared patterns P1-P8 (D0 state at each take's instant) and the reduce status rows."""
    A = _assigned(T)
    gi = A['gi'].to_numpy(np.int64)
    ty = np.asarray(A['type'], dtype=object)
    s, rk = A['s'].to_numpy(np.int64), A['rank'].to_numpy(np.int64)
    day = np.asarray(A['day'], dtype=object)
    sg, c5, tlb = C0['sg'][gi].astype(np.int64), C0['cat_5s'][gi].astype(np.int64), C0['tl'][gi].astype(np.int64)
    s_al = -s if mutant() == 'flip_align' else s
    side_with = align_codes(s_al, P['r_5s0'][gi], THR['r_5s']) == 0
    red, add = ty == 'reduce', ty == 'add'
    late = tlb >= 5
    MT = dict(P1=red & (s == -1) & (sg == 1) & (c5 == 0), P2=red & (s == -1) & (sg == 1) & (c5 == 1), P3=red & (s == 1) & (sg == -1) & (c5 == 0),
              P4=red & (s == 1) & (sg == -1) & (c5 == 1), P5=red & (sg != 0) & (c5 == 2), P6=add & side_with, P7=late, P8=red & late)
    STAT = (('status:reduce_q0_at_t', 'reduce take, q(t) = 0', red & (sg == 0)),
            ('status:reduce_btc_missing', 'reduce take, q != 0, BTC 5 s missing', red & (sg != 0) & (c5 == 3)),
            ('status:reduce_side_equals_inventory_sign', 'reduce take whose side equals the sign of q(t)', red & (sg != 0) & (s == sg)))
    n_mk = len(P['markets'])
    rkT = T['rank'].to_numpy(np.int64)
    rows = []
    for nm, desc, M_ in [(p, d_, MT[p]) for p, d_ in PATTERNS] + list(STAT):
        for sc in SCOPES:
            sm = _scope_mask(rk, sc)
            m = M_ & sm
            ues = np.unique(gi[m])
            per_mk = np.bincount(P['mkt'][ues], minlength=n_mk) if len(ues) else np.zeros(n_mk, np.int64)
            den = int(len(T)) if sc == 'pooled' else int((rkT == int(sc[4:])).sum())
            rows.append(dict(pattern=nm, definition=desc, scope=sc, n_event_seconds=int(len(ues)), n_takes=int(m.sum()),
                             maker_days=len({(int(a), str(b)) for a, b in zip(rk[m], day[m])}), markets_ge1=int((per_mk >= 1).sum()),
                             markets_ge5=int((per_mk >= 5).sum()), takes_denominator=den, share_of_takes=(int(m.sum()) / den) if den else NAN,
                             takes_in_population=int(sm.sum())))
    return pd.DataFrame(rows)


def end_table(P, flat, hz):
    """M5: per pair q at END / END - 30 s / END - 60 s (every variant), the share flattened, the reduce hazard by time left (M1)."""
    rows = []
    for pm in P['pairs']:
        r = dict(kind='pair', day=pm['day'], mid=pm['mid'], rank=pm['rank'], n_legs=pm['n_legs'])
        for v in VARIANTS:
            e = np.asarray(pm['end'][v], float)
            r[f'q_end_{v}'], r[f'q_end_m30s_{v}'], r[f'q_end_m60s_{v}'] = float(e[0]), float(e[1]), float(e[2])
            r[f'flattened_{v}'] = bool(abs(e[0]) <= FLAT_FRAC * abs(e[2])) if abs(e[2]) > POS_EPS else None
        rows.append(r)
    for d in flat.values():
        rows.append(dict(kind='share_flattened', **d))
    if len(hz):
        h = hz[(hz['table'] == 'marginal') & (hz['factor'] == 'time_left')]
        for rr in h.to_dict('records'):
            rows.append(dict(kind='reduce_hazard_time_left', scope=rr['scope'], level=rr['level'], maker_seconds=rr['maker_seconds'],
                             event_seconds=rr['event_seconds_reduce'], takes=rr['takes_reduce'], hazard=rr['hazard_reduce']))
    return pd.DataFrame(rows)


def inventory_table(P, C0, T):
    """M6: |q(t)| at each take and the fraction unwound -sign(q) dq / |q| by type and scope; beside it |q| over every q != 0 maker-second."""
    A = _assigned(T)
    gi = A['gi'].to_numpy(np.int64)
    q = P['q0'][gi]
    aq = np.abs(q)
    nz = aq > POS_EPS
    dq = A['dq'].to_numpy(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(nz, -np.sign(q) * dq / np.where(nz, aq, 1.), NAN)
    ty, rk = np.asarray(A['type'], dtype=object), A['rank'].to_numpy(np.int64)
    rows = []
    for sc in SCOPES:
        sm = _scope_mask(rk, sc)
        for t_ in TYPES_ANY:
            m = sm & (ty == t_) if t_ != 'any' else sm
            rows.append(dict(kind='take', scope=sc, type=t_, n=int(m.sum()), n_q_zero=int((m & ~nz).sum()), **_qs(aq[m & nz], 'absq'),
                             **_qs(frac[m], 'frac_unwound'), **_shares(qbin(q[m]), Q_LAB, 'qbin_share')))
        im = _scope_mask(P['rank'], sc) & (C0['qb'] > 0)
        rows.append(dict(kind='maker_seconds_q_nonzero', scope=sc, type='', n=int(im.sum()), n_q_zero=0, **_qs(np.abs(P['q0'][im]), 'absq'),
                         **_shares(C0['qb'][im], Q_LAB, 'qbin_share')))
    return pd.DataFrame(rows)


def edge_table(P, C0, T, EV):
    """M7: the unwind edge at reduce event-seconds vs every q != 0 maker-second; each take's price vs p_fair at t (100 s (p_fair - p_avg))."""
    rows = []
    red = np.flatnonzero(EV['E_reduce'])
    for sc in SCOPES:
        sm = _scope_mask(P['rank'], sc)
        for pop, idx in (('reduce_event_seconds', red[sm[red]]), ('q_nonzero_maker_seconds', np.flatnonzero(sm & (C0['qb'] > 0)))):
            e, eb = C0['edge'][idx], C0['edgeb'][idx]
            fin = np.isfinite(e)
            rows.append(dict(kind='unwind_edge', scope=sc, population=pop, type='', n=int(len(idx)), n_missing=int((~fin).sum()), **_qs(e, 'value_c'),
                             **_shares(eb[fin], EDGE_LAB[:5], 'bin_share')))
    A = _assigned(T)
    gi = A['gi'].to_numpy(np.int64)
    with np.errstate(invalid='ignore'):
        x = 100. * A['s'].to_numpy(float) * (P['pf0'][gi] - A['p_avg'].to_numpy(float))
    ty, rk = np.asarray(A['type'], dtype=object), A['rank'].to_numpy(np.int64)
    for sc in SCOPES:
        sm = _scope_mask(rk, sc)
        for t_ in TYPES_ANY:
            m = sm & (ty == t_) if t_ != 'any' else sm
            rows.append(dict(kind='take_vs_fair', scope=sc, population='takes', type=t_, n=int(m.sum()), n_missing=int((m & ~np.isfinite(x)).sum()),
                             **_qs(x[m], 'value_c')))
    return pd.DataFrame(rows)


def status_block(P, C0, EV):
    out = {}
    for sc in SCOPES:
        m = _scope_mask(P['rank'], sc)
        nz = m & (C0['qb'] > 0)
        out[sc] = dict(maker_seconds=int(m.sum()), q_nonzero=int(nz.sum()), inv_timing_sensitive=int((m & P['ts']).sum()),
                       inv_ledger_conditional=int((m & P['lg']).sum()), btc_status={BTC_LAB[i]: int(((C0['bst'] == i) & m).sum()) for i in range(len(BTC_LAB))},
                       edge_q_nonzero={EDGE_LAB[i]: int(((C0['edgeb'] == i) & nz).sum()) for i in range(len(EDGE_LAB))},
                       event_seconds_inv_timing_sensitive={t: int((EV['E_' + t].astype(bool) & m & P['ts']).sum()) for t in TYPES_ANY},
                       event_seconds_inv_ledger_conditional={t: int((EV['E_' + t].astype(bool) & m & P['lg']).sum()) for t in TYPES_ANY})
    return out


def analyse(P, T, EV, full=True):
    """Every measure on the panel; full=False -> the verdict path only (ratios, shares, mechanisms)."""
    codes = {v: variant_codes(P, v) for v in VARIANTS}
    RK, cons = ratio_tables(P, codes, EV)
    flat = flatten_shares(P)
    mech = mechanism_claims(RK, flat)
    out = dict(mechanisms=mech, flatten=flat, ratio_rows=RK)
    if not full:
        return out
    hz = hazard_table(P, codes['D0'], EV)
    out.update(ratios=pd.DataFrame(list(RK.values())), consistency=pd.DataFrame(cons), hazard=hz, patterns=patterns_table(P, codes['D0'], T),
               end=end_table(P, flat, hz), inventory=inventory_table(P, codes['D0'], T), edge=edge_table(P, codes['D0'], T, EV),
               statuses=status_block(P, codes['D0'], EV))
    return out


# ------------------------------------------------------------------ claims, receipt, write-once ------------------------------------------------------------------
def make_claims(kind, days, P, T, EV, R, popc, repro):
    st = Counter(str(x) for x in np.asarray(T['pop_status'], dtype=object))
    by_type = Counter(str(x) for x in np.asarray(T['type'], dtype=object))
    ms = {sc: int(_scope_mask(P['rank'], sc).sum()) for sc in SCOPES}
    ev = {t: dict(event_seconds=int(EV['E_' + t].sum()), takes=int(EV['N_' + t].sum())) for t in TYPES_ANY}
    return dict(version=VERSION, kind=kind, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT), days=list(days), reproduction=repro,
                population=popc, maker_seconds=ms, takes=dict(rows=int(len(T)), status=dict(st), by_type=dict(by_type)), events=ev,
                statuses=R['statuses'], mechanisms=R['mechanisms'], share_flattened=list(R['flatten'].values()),
                multiplicity=dict(mechanisms=len(MECHANISMS), scopes=len(M2_SCOPES), tests=len(MECHANISMS) * len(M2_SCOPES), all_listed=True, selected=None),
                null=dict(design='each maker-market event-second vector circularly shifted by a uniform offset mod its instant count; state paths fixed',
                          n_shifts=N_SHIFT, seed=SEED, offsets='Generator(PCG64(SeedSequence([seed, day, mid, rank])))',
                          p_hi='(1 + #{null >= obs}) / (n_shifts + 1); NaN replicates counted on both sides; ties within 1e-12 relative'),
                rules=dict(min_events=MIN_EVENTS, min_supported_markets=MIN_MARKETS, min_market_maker_seconds=MIN_MKT_S, p_max=P_MAX, c_hi_max=C_HI_MAX,
                           market_share=f'{SHARE_NUM}/{SHARE_DEN}', flatten_frac=FLAT_FRAC, flatten_share_min=FLAT_SHARE_MIN,
                           timing_sensitive='verdicts differ among D0 / D1000 / pend'),
                descriptive_only=True, intervals=None)


def receipt(kind, days, hashes, cnt, stages, t0, asof0, outs, git, extra=None):
    srcs = {_rel(f): X._sha(f) for f in TOOL_FILES + DEP_FILES if Path(f).exists()}
    rec = dict(version=VERSION, kind=kind, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, sha256=srcs.get(_rel(CONTRACT_FILE))),
               module=str(Path(__file__).resolve()), run=C.RUN, mutant=os.environ.get('MT_TL_MUT', ''), asof_start=asof0, asof_end=_now(),
               days=list(days), addresses=list(TOP3), inputs=dict(sorted(hashes.items())), sources=srcs, outputs=outs, git=git,
               counts=dict(sorted((str(k), v) for k, v in cnt.items())),
               constants=dict(STEP_MS=STEP_MS, D_SENS=D_SENS, VARIANTS=VARIANTS, SEED=SEED, N_SHIFT=N_SHIFT, POS_EPS=POS_EPS, Q_LAB=Q_LAB, THR=THR,
                              DOSE_LAB=DOSE_LAB, TL_LAB=TL_LAB, F5_LAB=F5_LAB, EDGE_LAB=EDGE_LAB, FILL_WIN_MS=FILL_WIN_MS, END_OFFS_MS=END_OFFS_MS,
                              FLAT_FRAC=FLAT_FRAC, FLAT_SHARE_MIN=FLAT_SHARE_MIN, MIN_EVENTS=MIN_EVENTS, MIN_MARKETS=MIN_MARKETS, MIN_MKT_S=MIN_MKT_S,
                              P_MAX=P_MAX, C_HI_MAX=C_HI_MAX, TIE_REL=TIE_REL, CLASS_COL=CLASS_COL, CLASS_TYPE=CLASS_TYPE),
               cpu_s=dict(by_stage=dict(stages), process_total=round(time.process_time(), 3)), wall_s=round(time.time() - t0, 2),
               vmhwm_mib=round(CT.hwm(), 1), peak_rss='VmHWM of this process', guard_mib=C.GUARD_MIB)
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
        X._json(tmp / 'claims.json', claims)
        outs['claims.json'] = {}
        for nm in outs:
            outs[nm].update(sha256=X._sha(tmp / nm), bytes=int((tmp / nm).stat().st_size))
        rec = rec_fn(outs)
        X._json(tmp / 'receipt.json', rec)
        need(not out.exists(), f'write-once: {out} appeared during the run (refused)')
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return rec


def run_core(kind, days, parts, T, out, hashes, cnt, stages, tick, git, asof0, t0, expect_pairs, repro=None, extra_fn=None):
    """Shared tail of the run / pilot (and the selftest's end-to-end): panel, population check, events, measures, claims, write-once."""
    P = assemble(parts)
    tick('assemble')
    popc = population_checks(P, expect_pairs)
    need(popc['complete'], f'population check failed (refused): {popc}')
    T2, EV = events(P, T)
    for k_, v_ in Counter(str(x) for x in T2['pop_status']).items():
        cnt[f'takes_{k_}'] += v_
    cnt['maker_seconds'] = int(P['N'])
    tick('events')
    R = analyse(P, T2, EV, full=True)
    tick('tables')
    CT.guard('tables')
    claims = make_claims(kind, days, P, T2, EV, R, popc, repro)
    tables = {'hazard_cells.csv': R['hazard'], 'ratios.csv': R['ratios'], 'consistency.csv': R['consistency'], 'patterns.csv': R['patterns'],
              'end_of_market.csv': R['end'], 'inventory_at_take.csv': R['inventory'], 'edge.csv': R['edge']}

    def rec_fn(outs):
        tick('write')
        return receipt(kind, days, hashes, cnt, stages, t0, asof0, outs, git, extra_fn(stages) if extra_fn else None)
    rec = write_out(out, tables, claims, rec_fn)
    return rec, claims, R


# ------------------------------------------------------------------ run / pilot ------------------------------------------------------------------
def preflight_projection(path, hashes):
    """The full run starts only below the recorded pilot projection limit (<= 60 CPU s)."""
    path = Path(path)
    need(path.is_file(), f'no recorded pilot projection: {path} is missing (run the pilot first; refused before launch)')
    J = json.loads(path.read_text())
    pr = (J.get('projection') or {}).get('full_run_cpu_s')
    need(J.get('kind') == 'pilot' and isinstance(pr, (int, float)) and np.isfinite(pr),
         f'{path.name}: not a pilot receipt with a finite projection.full_run_cpu_s (refused before launch)')
    need(float(pr) <= PROJ_MAX_S, f'the recorded pilot projects {float(pr):.1f} CPU s for the full run > {PROJ_MAX_S:.0f} s (refused before launch)')
    hashes[_rel(path)] = X._sha(path)
    return dict(pilot_receipt=_rel(path), full_run_cpu_s=float(pr), limit_s=PROJ_MAX_S, pilot_sources=J.get('sources'))


def read_takes(PT, hashes):
    for k in ('takes', 'summary', 'take_receipt'):
        need(PT[k].is_file(), f'input missing: {PT[k]} (refused)')
        hashes[_rel(PT[k])] = X._sha(PT[k])
    hdr = pd.read_csv(PT['takes'], nrows=0).columns
    for c in TAKE_READ:
        need(c in hdr, f'takes.csv: required column {c} missing (refused)')
    T = pd.read_csv(PT['takes'], usecols=list(TAKE_READ), float_precision='round_trip',
                    dtype=dict(day=str, addr=str, sid=str, status=str, take_class=str, mid=np.int64, rank=np.int64, s=np.int64, x_lo_hard_ms=np.int64,
                               hi_ms=np.int64, settle_ms=np.int64))
    S = pd.read_csv(PT['summary'], dtype={'scope': str, 'day': str})
    tkr = json.loads(PT['take_receipt'].read_text())
    need(int((tkr.get('outputs') or {}).get('takes', -1)) == len(T), "takes.csv rows differ from chain_take's receipt outputs (refused)")
    return T, check_reproduction(T, S), tkr


def execute(kind, days, out, inp=None, pilot_receipt=None, require_freeze=True, log=print):
    """kind 'run' (every measurement day -> v2/chains/take_logic, freeze + pilot projection required) or 'pilot' (its day, out given)."""
    c_imp = time.process_time()
    t0, asof0 = time.time(), _now()
    need(mutant() == '', f"MT_TL_MUT={os.environ.get('MT_TL_MUT')!r} is set: production never runs a planted error (refused before any input is read)")
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
    git = X._git([f for f in TOOL_FILES if Path(f).exists()])
    pre = {}
    if kind == 'run':
        need(days == tuple(C.MEASURE_DAYS), 'the full run covers every measurement day (refused)')
        if require_freeze:
            need(all(Path(f).exists() for f in TOOL_FILES) and bool(git.get('unmodified')),
                 'freeze: take_logic.py, its selftest, the contract, chain_context.py and chain_take.py must be tracked and unmodified (refused)')
        pre = preflight_projection(Path(pilot_receipt) if pilot_receipt else PT['pilot_receipt'], hashes)
    tick('preflight')
    inp = Path(inp) if inp else C.OUT
    T_all, repro, tkr = read_takes(PT, hashes)
    tick('inputs')
    addrs, book = CT.population(inp / 'econ' / 'identities.json', CT.E.BOOK_MANIFEST, hashes)
    need(tuple(a.lower() for a in CX.top3(inp)) == tuple(a.lower() for a in TOP3), 'chain_context.top3 differs from chain_take.TOP3 (refused)')
    bm = CX.book_markets()
    need({d: sorted(int(m) for m in v) for d, v in bm.items()} == {d: sorted(int(m) for m in v) for d, v in book.items()},
         'chain_context.book_markets differs from chain_take.population (refused)')
    need(all(d in book for d in days), f'days {days} are not all BOOK manifest days (refused)')
    T = prepare_takes(T_all[T_all['day'].astype(str).isin(days).to_numpy()])
    take_days = Counter(str(x) for x in T_all['day'])
    fv_exp = json.loads(CT.FV_RECEIPT.read_text())['inputs']['econ/fills_valued.parquet']['sha256']
    hashes[_rel(CT.FV_RECEIPT)] = X._sha(CT.FV_RECEIPT)
    tick('population')
    hc, parts = {}, []
    for d in days:
        for mid in book[d]:
            M = CX.load_market(d, int(mid), TOP3, root=inp, hash_cache=hc)
            for k_, v_ in M.inputs.items():
                if isinstance(v_, dict) and v_.get('sha256'):
                    hashes.setdefault(k_, v_['sha256'])
            tick('context_load')
            parts.append(market_part(d, int(mid), M))
            del M
            cnt['markets'] += 1
            tick('states')
            CT.guard(f'market {d} {mid}')
    fvk = str((inp / 'econ' / 'fills_valued.parquet').resolve())
    need(hc.get(fvk) == fv_exp, f'fills_valued sha256 {str(hc.get(fvk))[:12]} differs from the accepted binding {fv_exp[:12]} (refused)')
    need(tkr['inputs'].get('econ/fills_valued.parquet') == hc.get(fvk),
         "fills_valued sha256 differs from chain_take's receipt: the saved takes and this context are different data (refused)")
    both = [k for k in hashes if isinstance(tkr['inputs'].get(k), str)]
    mism = [k for k in both if tkr['inputs'][k] != hashes[k]]
    need(not mism, f"inputs differ from chain_take's receipt {mism[:5]} (refused: mismatched artifacts)")
    cnt['inputs_matching_take_receipt'] = len(both)
    n_mk = int(cnt['markets'])
    n_pilot_takes = sum(take_days.get(d, 0) for d in days)

    def extra_fn(stg):
        ex = dict(preflight=pre, take_rows_by_day=dict(take_days), class_reproduction=repro)
        if kind == 'pilot':
            r_mk = N_BOOK / max(1, n_mk)
            r_ev = max(r_mk, sum(take_days.get(d, 0) for d in C.MEASURE_DAYS) / max(1, n_pilot_takes))
            fixed = sum(stg.get(k, 0.) for k in ('imports', 'preflight', 'inputs', 'population'))
            scal_mk = stg.get('context_load', 0.) + stg.get('states', 0.)
            scal_ev = sum(stg.get(k, 0.) for k in ('assemble', 'events', 'tables', 'write'))
            proj = fixed + scal_mk * r_mk + scal_ev * r_ev
            ex['projection'] = dict(full_run_cpu_s=round(proj, 2), pilot_cpu_s=round(time.process_time(), 3), limit_s=PROJ_MAX_S, launch_allowed=bool(proj <= PROJ_MAX_S),
                                    rule=f'fixed (imports + inputs + population) {fixed:.2f} + (context_load + states) {scal_mk:.2f} x {N_BOOK}/{n_mk} + '
                                         f'(assemble + events + tables + write) {scal_ev:.2f} x max(market ratio, take ratio) {r_ev:.2f}; the first load '
                                         f'carries the one-off fills_valued sha256 (scaled: conservative)')
        return ex
    rec, claims, R = run_core(kind, days, parts, T, out, hashes, cnt, stages, tick, git, asof0, t0, 3 * n_mk, repro=repro, extra_fn=extra_fn)
    log(json.dumps(X._clean(digest(out, claims, R, rec)), indent=1, sort_keys=True, default=str))
    return rec


def digest(out, claims, R, rec):
    RK = R['ratio_rows']
    key = {}
    for rn in PRIMARY:
        r = RK[('D0', 'rank1', 'reduce', rn)]
        key[rn] = dict(RR_MH=r['RR_MH'], p_hi=r['p_hi'], p_lo=r['p_lo'], n_events=r['n_events_informative'], a=r['a_exposed'], T1=r['T_exposed'],
                       b=r['b_reference'], T0=r['T_reference'], markets_rr_gt1=f"{r['n_markets_rr_gt1']}/{r['n_markets_supported']}", null_p50=r['null_p50'])
    mech = {f"{m['mechanism']}:{m['scope']}": dict(final=m['verdict'], D0=m['D0']['verdict'], D1000=m['D1000']['verdict'], pend=m['pend']['verdict'])
            for m in claims['mechanisms']}
    return dict(out=str(out), kind=claims['kind'], days=claims['days'], maker_seconds=claims['maker_seconds'], events=claims['events'],
                take_status=claims['takes']['status'], reduce_rank1_D0=key, mechanisms=mech,
                share_flattened={f"{d['scope']}:{d['variant']}": d for d in claims['share_flattened'] if d['variant'] == 'D0'},
                projection=rec.get('projection'), cpu_s=rec['cpu_s'], vmhwm_mib=rec['vmhwm_mib'])


def main(argv):
    a = list(argv[1:])
    usage = 'usage: MT_RUN=v2 take_logic.py run | MT_RUN=v2 take_logic.py run --pilot <day> [--out <dir>]'
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
