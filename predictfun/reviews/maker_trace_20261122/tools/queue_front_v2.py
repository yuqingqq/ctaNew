"""queue_front_v2.py - QUEUE-FRONT v2 (P-2026-005 MAKER-TRACE; session aea5bfb9, 2026-11-23): section 5 of
predictfun/reviews/maker_trace_20261122/posting_chain_contract.md (POSTING-OPPORTUNITY-AND-CHAIN v1, frozen before computation, commit
8e09f3391) = the queue-front corrections of review 3f640ad9b (reviews/automatic/20261123_take_queue_review/report.md, findings 1-6).
It IMPORTS the frozen v1 tool queue_front.py (never edited) and overrides only what section 5 corrects: every input, definition, bin and
constant not named below is v1's (the 19 BOOK markets Oct 24-27, chain_context.load_market, the lifecycle fills / orders, fills_valued,
chain_post_fill orders.csv; QF._records' book timing, the v1 member / old-life / new-life / competitor rules, the creation context, the null).
One more input: correction_v1/post_regrade_orders.csv (the order post grade), bound to its receipt_post.json. DESCRIPTIVE on four consumed
days: no fit, no replay, no new data, no interval (G = 4 < 5), no model skill, no profitability claim.
(1) NO CERTIFICATION BY NET DECREASES. v1's book_timed record (the unique decrease equal to its quantity, claimed by no other record) is a
  book_candidate: a conditional timing candidate under a NO-NETTING premise. The earliest set is v1's (amendment 2) under that premise; a
  one-address set is a front_candidate (never certified) with timing_basis book_candidate / settlement. ALTERNATIVE HISTORIES (masked
  cancellation / replenishment) are counted per level with an earliest record: alt_sites_life = the frame intervals of the level's life
  from the creation interval through the candidate interval (an addition could mask an earlier fill there; the candidate's own decrease
  could be a cancel), alt_sites_unexplained = the intervals in [f + 1, k_hi] (k_hi = the first frame at or after the earliest record's hi,
  at most the level's last new-life frame) whose change no matched record explains (an increase, or a decrease claimed by no record);
  alt_netting_possible = alt_sites_life + alt_sites_unexplained > 0. READING: the creation interval and the candidate interval always
  count, so it holds for every level with an earliest record. The MASKED-NETTING first-filler set alt_first = every member record (old-life
  records by book included: their decrease could be a cancel) whose hard window starts no later than H1 = the earliest hi among the records
  that must lie in the new life (bounded, x_lo_hard > upd(f-1), hi <= upd(b) at a zero boundary else upd(b-1)); unbounded members always.
  alt_first_addresses keeps every first-filler address those windows allow; alt_first_other_address marks a level whose first-filler
  ADDRESS could differ from the earliest set's under such a history (READING, conservative: an exhaustive hidden-order solver is not run;
  the set may over-include).
(2) STRADDLE FIX. A record without a visible decrease is settlement_only (joins the creation interval) only if its WHOLE hard window
  [x_lo_hard, hi] lies inside (upd(f-1), upd(f)]; otherwise it is untimed (v1 checked the lower endpoint only, queue_front.py:629).
  READING: bounded windows are exactly 2,000 ms and an E1 creation interval is <= 1,000 ms (the gap rule), so the settlement basis cannot
  occur on real data; v1's settlement_only records become untimed (counted: rec_straddle_untimed, levels straddle_record).
(3) SAME-HASH SPLIT. Fills after the earliest set at a front_candidate level are later_same_hash (an order hash of the earliest set or its
  same-address competitors) or later_other. Economics = earliest_set vs later_other, later_same_hash beside, with all / valid rows and
  shares and the valid shares; "does the front pay?" is withdrawn as a queue-effect inference (no verdict).
(4) ENDPOINT OCCUPANCY. v1's joined categories become cell_occupied_at_window_lo / cell_empty_at_window_lo (+ cell_unknown /
  window_unbounded / cell_not_placeable), read at the inferred posting-window lower endpoint post_lo_union_ms with the frame time used and
  the order's post grade; never an actual-post queue label. Population: the top-3 orders that later filled (chain_post_fill).
(5) MECHANISM AND SPEED. creation_context = the anonymous cell-creation context (v1's rule, frame f - 1) associated with a candidate first
  filler. speed_ms = v1's cell-event endpoint proxy with speed_lo_ms / speed_hi_ms interval columns (re_establish: v1's; improve: lower
  upd(f-1) - upd(g*), upper upd(f) - upd(g* - 1)); no ownership of either endpoint, no 'faster' verdict.
(6) FRONT-WIN READING. Computed with every denominator; verdict 'wins_front_more_than_fills' needs v1's predicates AND no level carrying
  alternative histories, else unidentified / insufficient_support; [scenario_lower, scenario_upper] is a bound within the declared
  reconstruction scenario (no-netting book candidates, one-address earliest sets), alt_lower / alt_upper the masked-netting set beside; p_hi
  is v1's within-market label permutation of the attributed fills (a conditional shuffle after selection).
PARITY (in-process): v1's _new_life runs on the same inputs; every level without a straddle record must reproduce v1's decision exactly
  (status mapped, reason, front record, set, counts, attributed fills mapped front -> earliest_set / behind -> later_*): parity.n_violations
  (pilot: recorded; full run: refused unless 0, and the recorded pilot's must be 0).
OUTPUTS write-once (a partial directory renamed at the end): levels.csv (every fresh-level candidate), candidates.csv (keyed per E1 level:
  day, mid, level_id / side, tick, frame; the earliest one-address set candidate: candidate_addr (front_candidate only), candidate_hash,
  earliest_set_hashes, timing_basis, candidate interval, alt_netting_possible, alt_first_addresses ...; read by posting_opportunity),
  fills.csv (the keyed attributed fills: every economics aggregate reconciles to it), shares.csv, mechanism.csv, speed.csv, economics.csv,
  occupancy.csv (keyed per order), claims.json, receipt.json (sha256 of sources, inputs and outputs; CPU by stage; VmHWM).
  run -> v2/chains/queue_front_v2; run --pilot <day> --out <dir> -> the pilot (prints the full-run CPU projection).
GUARDS MT_RUN=v2; measurement days only (refused before any path is built); MT_QF2_MUT (the selftest's planted errors) and v1's MT_QF_MUT
  are refused in production; the full run refuses unless the tool, its selftest, the contract, queue_front.py and chain_context.py are
  tracked and unmodified (the freeze) and the recorded pilot projection is <= 120 CPU s with 0 parity violations; fills_valued sha256 =
  the accepted binding = chain_post_fill's; orders.csv = its receipt; post_regrade_orders.csv = receipt_post.json, which must name the same
  orders.csv; every post_fill order has exactly one regrade row with the same rank / cell / window; VmHWM self-guard 1,450 MiB (never
  raised); CPU self-guard (pilot 12 s, run 150 s).
Usage (run_light, MT_RUN=v2): queue_front_v2.py run | queue_front_v2.py run --pilot 20261024 --out <dir>   selftest: queue_front_v2_selftest.py"""
import bisect
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import queue_front as QF  # noqa: E402  (the frozen v1 tool: imported, never edited)

C, E, LC, CX = QF.C, QF.E, QF.LC, QF.CX
need = C.need
NAN = float('nan')
INF = float('inf')
VERSION = 'queue_front_v2'
CONTRACT_FILE = C.RECEIPT_V1 / 'posting_chain_contract.md'
CONTRACT_COMMIT = '8e09f3391'
REVIEW = dict(commit='3f640ad9b', file='predictfun/reviews/automatic/20261123_take_queue_review/report.md', findings='1-6')
PILOT_RUN = 'r1'
PILOT_DAY = QF.PILOT_DAY
TOOL_FILES = (HERE / 'queue_front_v2.py', HERE / 'queue_front_v2_selftest.py', CONTRACT_FILE, HERE / 'queue_front.py', HERE / 'chain_context.py')
DEP_FILES = QF.DEP_FILES + (QF.CONTRACT_FILE, HERE / 'queue_front_selftest.py')
TOP3, TOP3_L, N_BOOK, FV_SHA256 = QF.TOP3, QF.TOP3_L, QF.N_BOOK, QF.FV_SHA256
NT, NC, GAP_MS, MAX_LAG_MS, TOL, REEST_MS = QF.NT, QF.NC, QF.GAP_MS, QF.MAX_LAG_MS, QF.TOL, QF.REEST_MS
SEED, N_PERM = QF.SEED, QF.N_PERM
MIN_FRONTS, MIN_MARKETS, MIN_WINS, SHARE_NUM, SHARE_DEN, P_MAX = QF.MIN_FRONTS, QF.MIN_MARKETS, QF.MIN_WINS, QF.SHARE_NUM, QF.SHARE_DEN, QF.P_MAX
GROUPS, SCOPE_GROUPS, GNAMES, GBIT = QF.GROUPS, QF.SCOPE_GROUPS, QF.GNAMES, QF.GBIT
CONTEXTS, FLAGS, HORIZONS = QF.MECHS, QF.FLAGS, QF.HORIZONS
KEY, BIG = QF.KEY, QF.BIG
ZERO_B, GAP_B, UNK_B, END_B, END_NAMES = QF.ZERO_B, QF.GAP_B, QF.UNK_B, QF.END_B, QF.END_NAMES
FRONT = 'front_candidate'
STATUSES = (FRONT, 'ambiguous_order', 'unexplained_decrease', 'no_fill', 'gap', 'unknown')
V1_TO_V2 = dict(front_identified=FRONT)                     # every other status keeps its v1 name
REC_ST = ('book_candidate', 'book_ambiguous', 'book_unmatched', 'settlement_only')
BASES = ('book_candidate', 'settlement')
ROLES = ('earliest_set', 'later_other', 'later_same_hash')
OCC_ST = ('cell_occupied_at_window_lo', 'cell_empty_at_window_lo', 'cell_unknown', 'window_unbounded', 'cell_not_placeable')
MUT_ENV = 'MT_QF2_MUT'
MUTANTS = ('certify_net_decrease', 'lower_endpoint_settle', 'same_hash_behind', 'drop_no_fill', 'whole_life_unexplained')
READ_SCOPES = ('rank1', 'top3')
CPU_CAP_RUN_S, CPU_CAP_PILOT_S, PROJ_MAX_S = 150., 12., 120.
RG_DTYPES = dict(day=str, mid=np.int64, rank=np.int64, addr=str, mk_hash=str, y_side=str, tick=np.int64, post_lo_union_ms=np.int64,
                 post_hi_ms=np.int64, grade=str)
CSV_FILES = ('levels.csv', 'candidates.csv', 'fills.csv', 'shares.csv', 'mechanism.csv', 'speed.csv', 'economics.csv', 'occupancy.csv')
OUT_FILES = CSV_FILES + ('claims.json', 'receipt.json')
STR_COLS = ('end_type', 'candidate_addr', 'candidate_group', 'candidate_sid', 'candidate_hash', 'earliest_set_hashes', 'timing_basis',
            'earliest_set_addresses', 'alt_first_addresses', 'v1_timing', 'creation_context')
NEG1_COLS = ('end_ms', 'n_members', 'n_unbounded', 'candidate_mk_idx', 'candidate_lo_ms', 'candidate_hi_ms', 'candidate_book_ms',
             'candidate_interval_lo_ms', 'candidate_interval_hi_ms', 'n_earliest_set', 'n_unexplained', 'n_later_same_hash', 'n_later_other',
             'n_unattributed')
ZERO_COLS = ('amb_gmask', 'n_old_life', 'alt_sites_life', 'alt_sites_unexplained', 'alt_first_n_records', 'alt_first_n_addresses', 'alt_gmask')
BOOL_COLS = ('straddles_end', 'earliest_one_address', 'alt_netting_possible', 'alt_first_other_address', 'straddle_record') + FLAGS
FLOAT_COLS = ('best_prev_tick', 'speed_ms', 'speed_lo_ms', 'speed_hi_ms')
LEVEL_COLS = ('day', 'mid', 'level_id', 'side', 'tick', 'frame', 'fid', 't_open_ms', 't_create_ms', 'phase', 'unit', 'status', 'reason',
              'end_type', 'end_ms', 'size_created', 'n_members', 'n_unbounded', 'candidate_addr', 'candidate_group', 'candidate_sid',
              'candidate_mk_idx', 'candidate_hash', 'earliest_set_hashes', 'timing_basis', 'candidate_lo_ms', 'candidate_hi_ms',
              'candidate_book_ms', 'candidate_interval_lo_ms', 'candidate_interval_hi_ms', 'straddles_end', 'n_earliest_set',
              'earliest_one_address', 'earliest_set_addresses', 'amb_gmask', 'n_unexplained', 'n_old_life', 'n_later_same_hash', 'n_later_other',
              'n_unattributed', 'alt_netting_possible', 'alt_sites_life', 'alt_sites_unexplained', 'alt_first_n_records', 'alt_first_addresses',
              'alt_first_n_addresses', 'alt_first_other_address', 'alt_gmask', 'straddle_record', 'v1_status', 'v1_timing', 'creation_context',
              'is_improve', 'is_re_establish', 'is_deep', 'is_at_empty_touch', 'touch_unknown', 'best_prev_tick', 'speed_ms', 'speed_lo_ms',
              'speed_hi_ms')
CAND_COLS = ('day', 'mid', 'level_id', 'side', 'tick', 'frame', 'fid', 't_open_ms', 't_create_ms', 'phase', 'status', 'reason', 'candidate_addr',
             'candidate_group', 'candidate_hash', 'earliest_set_hashes', 'timing_basis', 'candidate_interval_lo_ms', 'candidate_interval_hi_ms',
             'candidate_lo_ms', 'candidate_hi_ms', 'alt_netting_possible', 'alt_sites_life', 'alt_sites_unexplained', 'alt_first_addresses',
             'alt_first_n_addresses', 'alt_first_other_address', 'earliest_set_addresses', 'n_earliest_set', 'earliest_one_address',
             'creation_context', 'n_members', 'n_old_life', 'straddle_record')
FILL_COLS = ('day', 'mid', 'level_id', 'side', 'tick', 'role', 'k', 'sid', 'mk_idx', 'mk_hash', 'addr', 'group', 'candidate_group',
             'creation_context', 'q', 'hi_ms', 'lo_ms', 'in_earliest_set', 'null_front', 'valid', 'excl_reason', 'q_valid', 'v_1s', 'v_10s', 'v_60s')
OCC_COLS = ('day', 'mid', 'group', 'addr', 'mk_hash', 'y_side', 'tick', 'n_fills', 'window_lo_ms', 'post_hi_ms', 'window_status', 'post_grade',
            'occupancy', 'size_at_window_lo', 'frame_ms_at_window_lo', 'frame_age_ms', 'lookup_status', 'is_front_candidate_hash')
PAR_KEYS = ('st', 'rs', 'lead', 'fhi', 'flo', 'nset', 'one', 'amb', 'gm', 'unx', 'old', 'timing', 'bookms', 'strad', 'nbeh', 'nun')


# ------------------------------------------------------------------ guards, paths ------------------------------------------------------------------
def mutant():
    """The selftest's planted-error switch MT_QF2_MUT, read at call time; v1's MT_QF_MUT must never be set here."""
    m = os.environ.get(MUT_ENV, '')
    need(m == '' or m in MUTANTS, f'{MUT_ENV}={m!r} is not a declared mutant {MUTANTS} (refused)')
    need(os.environ.get('MT_QF_MUT', '') == '', f"MT_QF_MUT={os.environ.get('MT_QF_MUT')!r} (a v1 planted error) is set (refused)")
    return m


def paths():
    ch = C.RECEIPT / 'chains'
    return dict(out=ch / 'queue_front_v2', pilot_receipt=ch / f'queue_front_v2_pilot_{PILOT_DAY}_{PILOT_RUN}' / 'receipt.json',
                post_fill=ch / 'post_fill', regrade=ch / 'correction_v1')


def pilot_out(day):
    return C.RECEIPT / 'chains' / f'queue_front_v2_pilot_{day}_{PILOT_RUN}'


def settle_inside(rlo, rhi, t_open, t_create, lower_endpoint=False):
    """Section 5.2: a record without a visible decrease joins the creation interval (t_open, t_create] only if its WHOLE hard window
    [rlo, rhi] lies inside; lower_endpoint=True is v1's rule (queue_front.py:629, the lower endpoint only: the lower_endpoint_settle mutant)."""
    if lower_endpoint:
        return t_open < rlo <= t_create
    return t_open < rlo and rhi <= t_create


# ------------------------------------------------------------------ inputs ------------------------------------------------------------------
def read_regrade(rg_dir, markets, hashes, post_orders_sha):
    """correction_v1/post_regrade_orders.csv (every top-3 filled order with its post grade) bound to receipt_post.json: the file's sha256 must
    be the receipt's output and the receipt's post_fill orders.csv the one read here; (day, mid, mk_hash) unique -> the run's rows."""
    rg_dir = Path(rg_dir)
    po, pr = rg_dir / 'post_regrade_orders.csv', rg_dir / 'receipt_post.json'
    need(po.is_file() and pr.is_file(), f'input missing: {po} / {pr} (refused)')
    R = json.loads(pr.read_text())
    so = QF._sha(po)
    hashes[QF._rel(po)] = so
    hashes[QF._rel(pr)] = QF._sha(pr)
    need(((R.get('outputs') or {}).get('post_regrade_orders.csv') or {}).get('sha256') == so,
         "post_regrade_orders.csv differs from its receipt's sha256 (refused: mismatched artifacts)")
    rpo = ((R.get('inputs') or {}).get('post_orders') or {}).get('sha256')
    need(rpo == post_orders_sha, f"the regrade's post_fill orders.csv {str(rpo)[:12]} differs from the one read {str(post_orders_sha)[:12]} "
                                 "(refused: mismatched artifacts)")
    hdr = pd.read_csv(po, nrows=0).columns
    miss = [c for c in RG_DTYPES if c not in hdr]
    need(not miss, f'post_regrade_orders.csv lacks {miss} (refused)')
    D = pd.read_csv(po, usecols=list(RG_DTYPES), dtype=RG_DTYPES)
    key = pd.MultiIndex.from_arrays([D['day'].to_numpy(dtype=object), D['mid'].to_numpy(np.int64), D['mk_hash'].to_numpy(dtype=object)])
    need(key.is_unique, 'post_regrade_orders.csv: an order (day, mid, mk_hash) twice (refused)')
    keep = {(str(d), int(m)) for d, m in markets}
    sel = np.array([(str(d), int(m)) in keep for d, m in zip(D['day'].to_numpy(dtype=object), D['mid'].to_numpy(np.int64))], bool)
    return D[sel].reset_index(drop=True), dict(rows=int(len(D)), rows_run=int(sel.sum()), sha256=so)


def join_grades(PF, RG):
    """Every post_fill order of the run gets its regrade row's grade; the two tables must hold the same orders with the same rank, cell and
    window (refused otherwise: mismatched artifacts)."""
    def k(D):
        return pd.MultiIndex.from_arrays([D['day'].astype(str).to_numpy(dtype=object), D['mid'].to_numpy(np.int64),
                                          D['mk_hash'].astype(str).to_numpy(dtype=object)])
    kp, kr = k(PF), k(RG)
    need(kp.is_unique, 'chain_post_fill orders.csv: an order (day, mid, mk_hash) twice (refused)')
    j = np.asarray(kr.get_indexer(kp), np.int64)
    need(bool((j >= 0).all()), f'{int((j < 0).sum())} post_fill orders without a regrade row (refused: mismatched artifacts)')
    need(len(RG) == len(PF), f'the regrade holds {len(RG)} orders of the run markets, post_fill {len(PF)} (refused: mismatched artifacts)')
    G = RG.iloc[j].reset_index(drop=True)
    for c_ in ('rank', 'tick', 'post_lo_union_ms', 'post_hi_ms'):
        need(bool((G[c_].to_numpy(np.int64) == PF[c_].to_numpy(np.int64)).all()), f'regrade {c_} differs from post_fill (refused: mismatched artifacts)')
    need(bool((G['y_side'].astype(str).to_numpy(dtype=object) == PF['y_side'].astype(str).to_numpy(dtype=object)).all()),
         'regrade y_side differs from post_fill (refused: mismatched artifacts)')
    return PF.assign(post_grade=G['grade'].astype(str).to_numpy(dtype=object))


# ------------------------------------------------------------------ one market ------------------------------------------------------------------
def _new_life_v2(R, nE, f, c, To, kend, b, btype, zero, zbw, tb, tb1, t, we, J0, J1, V0, V1, acode, addr, mkh, cumX, mut):
    """v1's _new_life (amendment 2) with section 5's corrections: the straddle fix (settle_inside), statuses front_candidate, timing
    book_candidate, the same-hash split of later records, the alternative-history sites and the masked-netting first-filler set. Every
    other branch is v1's line for line (checked in process: parity)."""
    cert, low_end, same_beh, whole = (mut == 'certify_net_decrease', mut == 'lower_endpoint_settle', mut == 'same_hash_behind',
                                      mut == 'whole_life_unexplained')
    NR = int(R['NR'])
    rtimed, rk, ramb, rbnd = R['rtimed'].tolist(), R['rk'].tolist(), R['ramb'].tolist(), R['rbnd'].tolist()
    rlo, rhi, rgb = R['rlo'].tolist(), R['rhi'].tolist(), R['rgb'].tolist()
    coff, ck, roff = R['coff'].tolist(), R['ck'].tolist(), R['roff'].tolist()
    legs = R['legs'].tolist()
    cumUX = R['cumUX']
    rb, ru = R['rb'].tolist(), R['ru'].tolist()
    nleg = [roff[r + 1] - roff[r] for r in range(NR)]
    tl = t.tolist()
    F = len(tl)
    adL, mkL, acL = list(addr), list(mkh), acode.tolist()
    so_used, member = np.zeros(NR, bool), np.zeros(NR, bool)
    NEG = -(1 << 62)
    o = dict(st=np.full(nE, 'no_fill', dtype=object), rs=np.full(nE, '', dtype=object), lead=np.full(nE, -1, np.int64), fhi=np.full(nE, -1, np.int64),
             flo=np.full(nE, -1, np.int64), nset=np.zeros(nE, np.int64), one=np.zeros(nE, bool), amb=np.full(nE, '', dtype=object),
             gm=np.zeros(nE, np.int64), unx=np.zeros(nE, np.int64), old=np.zeros(nE, np.int64), timing=np.full(nE, '', dtype=object),
             bookms=np.full(nE, -1, np.int64), strad=np.zeros(nE, bool), nbeh=np.zeros(nE, np.int64), nun=np.zeros(nE, np.int64),
             kst=np.full(nE, -1, np.int64), ivlo=np.full(nE, -1, np.int64), ivhi=np.full(nE, -1, np.int64), esa=np.full(nE, '', dtype=object),
             esh=np.full(nE, '', dtype=object), nsame=np.zeros(nE, np.int64), nother=np.zeros(nE, np.int64), altp=np.zeros(nE, bool),
             slife=np.zeros(nE, np.int64), sun=np.zeros(nE, np.int64), altn=np.zeros(nE, np.int64), alta=np.full(nE, '', dtype=object),
             altna=np.zeros(nE, np.int64), alto=np.zeros(nE, bool), altg=np.zeros(nE, np.int64), strad_rec=np.zeros(nE, bool))
    ae, al, ar, ak, ai, an = [], [], [], [], [], []
    J0, J1, V0, V1 = J0.tolist(), J1.tolist(), V0.tolist(), V1.tolist()
    fL, cL, ToL, keL, bL, btL = f.tolist(), c.tolist(), To.tolist(), kend.tolist(), b.tolist(), btype.tolist()
    zL, zbwL, tbL, tb1L = zero.tolist(), zbw.tolist(), tb.tolist(), tb1.tolist()
    reasons = {GAP_B: 'gap_before_fill', UNK_B: 'unknown_cell_before_fill', END_B: 'book_end_before_fill'}

    def alt_out(e, alt, esa):
        aa, ag = set(), 0
        for r in alt:
            ag |= rgb[r]
            for x in legs[roff[r]:roff[r + 1]]:
                aa.add(adL[x])
        o['altn'][e], o['alta'][e], o['altna'][e], o['altg'][e] = len(alt), '|'.join(sorted(aa)), len(aa), ag
        o['alto'][e] = bool(esa) and bool(aa - esa)
    for e in range(nE):
        fe, ce, Toe, ke = fL[e], cL[e], ToL[e], keL[e]
        tfe = tl[fe]
        mem = rb[J0[e]:J1[e]] + ru[V0[e]:V1[e]]
        NLr, SO, CA, U = [], [], [], []
        n_old = n_after = old_legs = mem_legs = 0
        for r in mem:
            member[r] = True
            mem_legs += nleg[r]
            if rtimed[r]:
                k = rk[r]
                if k <= fe - 1:
                    n_old += 1
                    old_legs += nleg[r]
                elif k <= ke:
                    NLr.append((k, r))
                else:
                    n_after += 1
            elif ramb[r]:
                cs = ck[coff[r]:coff[r + 1]]
                if cs[-1] <= fe - 1:
                    n_old += 1
                    old_legs += nleg[r]
                elif any(fe < k <= ke for k in cs):
                    CA.append(r)
                else:
                    n_after += 1
            else:
                if rbnd[r] and Toe < rlo[r] <= tfe and rhi[r] > tfe:
                    o['strad_rec'][e] = True              # v1's lower-endpoint rule would put it in the creation interval
                if rbnd[r] and settle_inside(rlo[r], rhi[r], Toe, tfe, low_end):
                    SO.append(r)
                    so_used[r] = True
                else:
                    U.append(r)
        o['old'][e] = n_old
        Tm = tbL[e] if zL[e] else tb1L[e]                  # the latest time a fill is certainly of this level's new life
        H1 = INF
        for r in mem:
            if rbnd[r] and rlo[r] > Toe and rhi[r] <= Tm and rhi[r] < H1:
                H1 = rhi[r]
        alt = [r for r in mem if (not rbnd[r]) or rlo[r] <= H1]
        rs_ = None
        if SO:
            SO.sort(key=lambda r: (rhi[r], r))
            rs_ = SO[0]
            E_ = [rs_] + [r for r in SO[1:] if rlo[r] <= rhi[rs_]]
            kst, timing = fe, 'settlement'
            later = [r for r in SO[1:] if r not in E_] + [r for _, r in sorted(NLr)]
        elif NLr:
            NLr.sort()
            kst, rs_ = NLr[0]
            E_, timing = [rs_], 'book_candidate'
            later = [r for _, r in NLr[1:]]
        if rs_ is None:
            if btL[e] != ZERO_B and tl[bL[e] - 1] < we and n_after > 0:
                o['st'][e], o['rs'][e] = ('unknown' if btL[e] == UNK_B else 'gap'), reasons[btL[e]]
            elif CA or U:
                o['st'][e], o['rs'][e] = 'unknown', 'untimed_record_only'
            else:
                o['rs'][e] = 'returned_to_zero' if zbwL[e] else 'window_end_without_fill'
            o['nun'][e] = mem_legs - old_legs
            alt_out(e, alt, set())
            continue
        CP = [r for r in CA if any(fe < k < kst for k in ck[coff[r]:coff[r + 1]])]
        tks = tl[kst] if kst >= 0 else tfe
        CP += [r for r in U if (rlo[r] if rbnd[r] else NEG) <= tks]
        Fset = E_ + CP
        codes, fa_addr, fa_hash = set(), set(), set()
        for r in Fset:
            for x in legs[roff[r]:roff[r + 1]]:
                codes.add(acL[x])
                fa_addr.add(adL[x])
                fa_hash.add(mkL[x])
        ambig = len(codes) >= 2
        unx = int(cumUX[kst, ce] - cumUX[fe + 1, ce]) if kst > fe + 1 else 0
        k_hi = ke if whole else min(bisect.bisect_left(tl, rhi[rs_]), F - 1, ke)
        sun = int(cumX[k_hi + 1, ce] - cumX[fe + 1, ce]) if k_hi >= fe + 1 else 0
        slife = kst - fe + 1
        altp = (slife + sun) > 0
        if cert:                                            # planted error: uniqueness among net decreases read as certainty
            altp, alt = False, list(Fset)
        o['fhi'][e], o['flo'][e] = rhi[rs_], rlo[rs_]
        o['nset'][e], o['one'][e] = len(Fset), len(Fset) >= 2 and len(codes) == 1
        o['timing'][e], o['bookms'][e], o['unx'][e] = timing, (tl[kst] if timing == 'book_candidate' else -1), unx
        o['kst'][e], o['ivlo'][e], o['ivhi'][e] = kst, tl[kst - 1], tl[kst]
        o['esa'][e], o['esh'][e] = '|'.join(sorted(fa_addr)), '|'.join(sorted(fa_hash))
        o['slife'][e], o['sun'][e], o['altp'][e] = slife, sun, altp
        alt_out(e, alt, fa_addr)
        if ambig:
            o['st'][e] = 'ambiguous_order'
            o['rs'][e] = 'untimed_record_could_be_first' if CP else ('overlapping_settlement_windows' if len(E_) >= 2 else 'sweep_several_makers')
            o['amb'][e] = o['esa'][e]
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
        o['st'][e], o['rs'][e] = FRONT, 'ok'
        lead = int(legs[roff[rs_]])
        o['lead'][e] = lead
        o['strad'][e] = bool(zL[e] and rhi[rs_] > tbL[e])
        n_att = 0
        for r in E_:
            for x in legs[roff[r]:roff[r + 1]]:
                ae.append(e); al.append(x); ar.append('earliest_set'); ak.append(0); ai.append(True); an.append(x == lead)
                n_att += 1
        beh = later + [r for r in CA if r not in CP and all(kst < k <= ke for k in ck[coff[r]:coff[r + 1]])]
        for j, r in enumerate(beh, start=1):
            for x in legs[roff[r]:roff[r + 1]]:
                same = (mkL[x] in fa_hash) and not same_beh
                ae.append(e); al.append(x); ar.append('later_same_hash' if same else 'later_other'); ak.append(j); ai.append(False); an.append(False)
                n_att += 1
                o['nbeh'][e] += 1
                if same:
                    o['nsame'][e] += 1
                else:
                    o['nother'][e] += 1
        o['nun'][e] = mem_legs - old_legs - n_att
    o['att'] = (np.asarray(ae, np.int64), np.asarray(al, np.int64), np.asarray(ar, dtype=object), np.asarray(ak, np.int64), np.asarray(ai, bool),
                np.asarray(an, bool))
    o['so_used'], o['member'] = so_used, member
    return o


def parity(o1, o2, nE):
    """v1 (_new_life, production rule) vs v2 on the same inputs: every level without a straddle record must agree on every v1 field (status
    front_identified -> front_candidate, timing book -> book_candidate) and on its attributed fills (front -> earliest_set, behind ->
    later_same_hash / later_other). Straddle levels are tabulated by transition."""
    aff = np.asarray(o2['strad_rec'], bool)
    ok = ~aff
    by_key, bad = {}, np.zeros(nE, bool)
    for k_ in PAR_KEYS:
        a, b_ = np.asarray(o1[k_]), np.asarray(o2[k_])
        if k_ == 'st':
            a = np.array([V1_TO_V2.get(str(x), str(x)) for x in a], dtype=object)
        elif k_ == 'timing':
            a = np.array(['book_candidate' if str(x) == 'book' else str(x) for x in a], dtype=object)
        d = np.asarray(a != b_, bool) & ok
        by_key[k_] = int(d.sum())
        bad |= d

    def tup(att, rmap):
        e_, x_, r_, k_, _, n_ = att
        return {(int(e), int(x), rmap(str(r)), int(kk), bool(nf)) for e, x, r, kk, nf in zip(e_, x_, r_, k_, n_) if ok[int(e)]}
    s1 = tup(o1['att'], lambda r: r)
    s2 = tup(o2['att'], lambda r: 'front' if r == 'earliest_set' else 'behind')
    att_bad = sorted({z[0] for z in s1 ^ s2})
    bad[att_bad] = True
    by_key['att'] = len(att_bad)
    tr = Counter(f'{o1["st"][e]}->{o2["st"][e]}' for e in np.flatnonzero(aff))
    return dict(n_levels=int(nE), n_affected=int(aff.sum()), n_violations=int(bad.sum()), by_key=by_key, affected_transitions=dict(tr),
                bad_e=np.flatnonzero(bad)[:10].tolist())


def _parity_empty():
    return dict(n_levels=0, n_affected=0, n_violations=0, by_key={k_: 0 for k_ in PAR_KEYS + ('att',)}, affected_transitions={}, bad_e=[])


def _fills_df(day, mid, col, ei, att, fl):
    """The attributed fills of front_candidate levels (earliest_set k = 0 / later k >= 1) -> fills.csv rows, the null's labels and the one
    designated front fill per level."""
    le_e, fix, role, kk, inset, nullf = att
    tot = int(len(fix))
    if not tot:
        return None, np.zeros(0, np.int8), np.zeros(0, bool)
    le = ei[le_e]
    df = pd.DataFrame(dict(day=np.full(tot, day, dtype=object), mid=np.full(tot, mid, np.int64), level_id=col['level_id'][le], side=col['side'][le],
                           tick=col['tick'][le], role=role, k=kk, sid=fl['sid'][fix], mk_idx=fl['mk_idx'][fix], mk_hash=fl['mk_hash'][fix],
                           addr=fl['addr'][fix], group=np.array(GROUPS, dtype=object)[fl['grp'][fix]], candidate_group=col['candidate_group'][le],
                           creation_context=col['creation_context'][le], q=fl['q'][fix], hi_ms=fl['hi'][fix], lo_ms=fl['lo'][fix], in_earliest_set=inset,
                           null_front=nullf, valid=fl['cohort'][fix], excl_reason=fl['reason'][fix], q_valid=fl['q_c'][fix],
                           **{f'v_{h}': fl[f'v_{h}'][fix] for h in HORIZONS}))
    ise = role == 'earliest_set'
    need(bool((fl['addr'][fix][ise] == col['candidate_addr'][le][ise]).all()), 'internal: an earliest_set fill is not its level candidate')
    need(int(nullf.sum()) == int(len(np.unique(le_e))), 'internal: not one designated front fill per front_candidate level')
    return df[list(FILL_COLS)], fl['grp'][fix].astype(np.int8), nullf.copy()


def _empty_core():
    return dict(levels=pd.DataFrame({c_: [] for c_ in LEVEL_COLS}), fills=pd.DataFrame({c_: [] for c_ in FILL_COLS}), lab=np.zeros(0, np.int8),
                front=np.zeros(0, bool), counts=Counter(), checks={}, parity=_parity_empty())


def market_core(day, mid, ws, we, bk, fl):
    """v1's market_core with section 5's corrections (pure: arrays in, tables out). The candidates, boundaries, members, record timing
    (QF._records), mechanism and speed are v1's lines; the new-life pass is _new_life_v2; v1's _new_life runs beside it for parity."""
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
    if F == 0:
        return _empty_core()
    P = QF._prints(t, cellA[g], fl['q'][g], lo[g], hi[g], F)
    U = np.zeros((F, NC), bool)
    if F > 1:
        with np.errstate(invalid='ignore'):
            U[1:] = (np.maximum(S[:-1] - S[1:], 0.) - P[1:]) > TOL
    pos = S > 0
    nanS = np.isnan(S)
    k0 = S == 0
    lpos = QF._acc_last(pos, F)
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
    col = {k_: None for k_ in LEVEL_COLS}
    col.update(day=np.full(nC, day, dtype=object), mid=np.full(nC, mid, np.int64), level_id=np.arange(nC, dtype=np.int64),
               side=np.where(cc // NT == 0, 'bid', 'ask').astype(object), tick=cc % NT, frame=ff, fid=bk['fid'][ff], t_open_ms=t[ff - 1],
               t_create_ms=t[ff], phase=np.where(t[ff] < ws, 'pre', 'live').astype(object), unit=np.where(fresh, 'E1', 'status_only').astype(object),
               status=np.where(isg, 'gap', 'unknown').astype(object), reason=np.where(isg, 'creation_across_gap', 'prev_cell_unknown').astype(object),
               size_created=S[ff, cc])
    col['v1_status'] = col['status'].copy()
    for k_ in STR_COLS:
        col[k_] = np.full(nC, '', dtype=object)
    for k_ in NEG1_COLS:
        col[k_] = np.full(nC, -1, np.int64)
    for k_ in ZERO_COLS:
        col[k_] = np.zeros(nC, np.int64)
    for k_ in BOOL_COLS:
        col[k_] = np.zeros(nC, bool)
    for k_ in FLOAT_COLS:
        col[k_] = np.full(nC, NAN)
    fr_rows, lab, front = [], np.zeros(0, np.int8), np.zeros(0, bool)
    acode = pd.factorize(pd.Series(fl['addr'], dtype=object))[0].astype(np.int64) if n else np.zeros(0, np.int64)
    R = QF._records(t, S, gap, F, fl, g, cellA, sidc, acode, base)          # v1 amendment 2: every record's book candidate(s)
    so1 = so2 = member = np.zeros(R['NR'], bool)
    par = _parity_empty()
    cnt['record_hi_mismatch'] = R['hi_mismatch']
    if nE:
        nbrk = QF._acc_next(~pos | gap[:, None], F)
        lnan = QF._acc_last(nanS, F)
        lgap = QF._acc_last(gap, F)
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
        u0 = np.searchsorted(keys_u, cK + off(To), 'right').astype(np.int64)
        nu = np.maximum(np.searchsorted(keys_u, cK + (KEY - 1), 'right').astype(np.int64) - u0, 0)
        k_we = int(np.searchsorted(t, we, 'left'))
        kend = np.minimum(np.where(zero, b, b - 1), k_we)
        J0 = np.searchsorted(R['rbk'], cK + off(To), 'right').astype(np.int64)
        J1 = np.maximum(np.searchsorted(R['rbk'], cK + off(T_last + MAX_LAG_MS), 'right').astype(np.int64), J0)
        V0 = np.searchsorted(R['ruk'], cK + off(To), 'right').astype(np.int64)
        V1 = np.maximum(np.searchsorted(R['ruk'], cK + (KEY - 1), 'right').astype(np.int64), V0)
        INC = np.zeros((F, NC), bool)
        with np.errstate(invalid='ignore'):
            INC[1:] = (S[1:] - S[:-1]) > TOL                 # an addition: never explained by a matched record
        cumX = R['cumUX'].copy()                             # + v1's decreases claimed by no record
        cumX[1:] += np.cumsum(INC, axis=0, dtype=np.int32)
        del INC
        A2 = _new_life_v2(R, nE, f, c, To, kend, b, btype, zero, zbw, tb, tb1, t, we, J0, J1, V0, V1, acode, fl['addr'], fl['mk_hash'], cumX, mut)
        del cumX
        A1 = QF._new_life(R, nE, f, c, To, kend, b, btype, zero, zbw, tb, t, we, J0, J1, V0, V1, acode, fl['addr'], '')
        par = parity(A1, A2, nE)
        par['bad_level_ids'] = [int(ei[e]) for e in par.pop('bad_e')]
        so1, so2, member = A1['so_used'], A2['so_used'], A2['member']
        # ---- creation context (v1's mechanism rule at frame f - 1) and the endpoint-proxy speed with its interval ----
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
        mech = QF._first([is_re, is_imp, is_deep, is_emp], ['re_establish', 'improve', 'deep', 'at_empty_touch'], 'other', nE)
        chg = np.zeros((2, F), bool)
        with np.errstate(invalid='ignore'):
            chg[:, 1:] = (best[:, 1:] != best[:, :-1]) | gap[None, 1:]
        lchg = QF._acc_last(chg.T, F).T
        gs = lchg[sd, f - 1].astype(np.int64)
        gsc = np.maximum(gs, 1)
        clean = (gs >= 1) & ~np.isnan(best[sd, gsc]) & ~np.isnan(best[sd, gsc - 1]) & ~gap[gsc]
        mre, mim = mech == 're_establish', mech == 'improve'
        mic = mim & clean
        speed = np.where(mre, dt_re.astype(np.float64), np.where(mic, (t[f - 1] - t[gsc]).astype(np.float64), NAN))
        sp_lo = np.where(mre, (t[f - 1] - t[p1]).astype(np.float64), np.where(mic, (t[f - 1] - t[gsc]).astype(np.float64), NAN))
        sp_hi = np.where(mre, (t[f] - t[pcl]).astype(np.float64), np.where(mic, (t[f] - t[gsc - 1]).astype(np.float64), NAN))
        # ---- E1 columns ----
        st = A2['st']
        isf = st == FRONT
        lead = A2['lead']
        okl = lead >= 0
        lc_ = np.maximum(lead, 0)

        def at(a, dv, dt=object):
            return np.where(okl, np.asarray(a)[lc_], dv) if n else np.full(nE, dv, dtype=dt)
        col['status'][ei], col['reason'][ei] = st, A2['rs']
        col['end_type'][ei] = np.where(zero & ~zbw, 'we', END_NAMES[btype])
        col['end_ms'][ei] = np.where(zbw, tb, np.where(zero, we, tb1))
        col['n_members'][ei], col['n_unbounded'][ei] = (i1 - i0) + nu, nu
        hg = np.array(GROUPS, dtype=object)[np.asarray(at(fl['grp'], 3, np.int64), np.int64)]
        col['candidate_addr'][ei] = np.where(isf, at(fl['addr'], ''), '')
        col['candidate_group'][ei] = np.where(isf, hg, '')
        col['candidate_sid'][ei] = np.where(isf, at(fl['sid'], ''), '')
        col['candidate_hash'][ei] = np.where(isf, at(fl['mk_hash'], ''), '')
        col['candidate_mk_idx'][ei] = np.where(isf, at(fl['mk_idx'], -1, np.int64), -1)
        col['earliest_set_hashes'][ei], col['timing_basis'][ei] = A2['esh'], A2['timing']
        col['candidate_lo_ms'][ei], col['candidate_hi_ms'][ei], col['candidate_book_ms'][ei] = A2['flo'], A2['fhi'], A2['bookms']
        col['candidate_interval_lo_ms'][ei], col['candidate_interval_hi_ms'][ei] = A2['ivlo'], A2['ivhi']
        col['straddles_end'][ei] = A2['strad']
        col['n_earliest_set'][ei], col['earliest_one_address'][ei], col['earliest_set_addresses'][ei] = A2['nset'], A2['one'], A2['esa']
        col['amb_gmask'][ei] = np.where(st == 'ambiguous_order', A2['gm'], 0)
        col['n_unexplained'][ei], col['n_old_life'][ei] = A2['unx'], A2['old']
        col['n_later_same_hash'][ei], col['n_later_other'][ei], col['n_unattributed'][ei] = A2['nsame'], A2['nother'], A2['nun']
        col['alt_netting_possible'][ei], col['alt_sites_life'][ei], col['alt_sites_unexplained'][ei] = A2['altp'], A2['slife'], A2['sun']
        col['alt_first_n_records'][ei], col['alt_first_addresses'][ei], col['alt_first_n_addresses'][ei] = A2['altn'], A2['alta'], A2['altna']
        col['alt_first_other_address'][ei], col['alt_gmask'][ei], col['straddle_record'][ei] = A2['alto'], A2['altg'], A2['strad_rec']
        col['v1_status'][ei], col['v1_timing'][ei] = A1['st'], A1['timing']
        col['creation_context'][ei] = mech
        for k_, v_ in (('is_improve', is_imp), ('is_re_establish', is_re), ('is_deep', is_deep), ('is_at_empty_touch', is_emp), ('touch_unknown', t_unk)):
            col[k_][ei] = v_
        col['best_prev_tick'][ei] = bp
        col['speed_ms'][ei], col['speed_lo_ms'][ei], col['speed_hi_ms'][ei] = speed, sp_lo, sp_hi
        cnt['straddling_fronts'] = int(np.asarray(A2['strad'], bool).sum())
        cnt['levels_old_life_excluded'] = int((np.asarray(A2['old']) > 0).sum())
        fdf, lab, front = _fills_df(day, mid, col, ei, A2['att'], fl)
        if fdf is not None:
            fr_rows.append(fdf)
    rst = np.where(R['rtimed'], 'book_candidate', np.where(R['ramb'], 'book_ambiguous', np.where(so2, 'settlement_only', 'book_unmatched')))
    for s_ in REC_ST:
        cnt[f'rec_{s_}'] = int((rst == s_).sum())
    cnt['rec_unbounded'] = int((~R['rbnd']).sum())
    cnt['rec_v1_settlement_only'] = int(np.asarray(so1, bool).sum())
    cnt['rec_straddle_untimed'] = int((np.asarray(so1, bool) & ~np.asarray(so2, bool)).sum())
    roff = R['roff']
    cnt['legs_member_of_fresh_level'] = int(sum(int(roff[r + 1] - roff[r]) for r in np.flatnonzero(member)))
    levels = pd.DataFrame({k_: col[k_] for k_ in LEVEL_COLS})
    if mut == 'drop_no_fill':                                # planted error: no_fill levels leave every denominator
        levels = levels[levels['status'] != 'no_fill'].reset_index(drop=True)
    fills = QF._cat(fr_rows, FILL_COLS)
    cnt['levels_rows'], cnt['fill_rows'] = int(len(levels)), int(len(fills))
    return dict(levels=levels, fills=fills, lab=lab, front=front, counts=cnt, checks=checks, parity=par)


def occupancy_rows(day, mid, M, pfm, cand_hashes):
    """Every top-3 filled order of the market (chain_post_fill + its post grade): the displayed size at its inferred posting-window lower
    endpoint post_lo_union_ms (chain_context.level_size: the frame in force, fresh <= 1 s) -> cell_occupied_at_window_lo /
    cell_empty_at_window_lo / cell_unknown / window_unbounded / cell_not_placeable, with the frame time and age; an endpoint observation
    conditional on that window, never an actual-post queue label."""
    if pfm is None or not len(pfm):
        return pd.DataFrame({c_: [] for c_ in OCC_COLS})
    nn = int(len(pfm))
    plo = pfm['post_lo_union_ms'].to_numpy(np.int64)
    ys = pfm['y_side'].to_numpy(dtype=object)
    tk = pfm['tick'].to_numpy(np.int64)
    okc = ((ys == 'bid') | (ys == 'ask')) & (tk >= 0) & (tk < NT)
    bnd = plo >= 0
    tq = np.where(bnd, plo, int(M.ws))
    ls = CX.level_size(M, tq, np.where(okc, ys, 'bid').astype(object), np.where(okc, tk, 0))
    lst, lsz = np.asarray(ls['status'], dtype=object), np.asarray(ls['size'], np.float64)
    lfr, lage = np.asarray(ls['frame'], np.int64), np.asarray(ls['age_ms'], np.int64)
    occ = QF._first([~bnd, ~okc, lst != 'ok', lsz > 0], ['window_unbounded', 'cell_not_placeable', 'cell_unknown', 'cell_occupied_at_window_lo'],
                    'cell_empty_at_window_lo', nn)
    tt = np.asarray(CX.frame_times(M), np.int64)
    fms = np.where(bnd & (lfr >= 0), tt[np.clip(lfr, 0, max(len(tt) - 1, 0))] if len(tt) else -1, -1)
    rk = pfm['rank'].to_numpy(np.int64)
    hs = pfm['mk_hash'].to_numpy(dtype=object)
    return pd.DataFrame(dict(day=np.full(nn, str(day), dtype=object), mid=np.full(nn, int(mid), np.int64), group=np.array(GROUPS, dtype=object)[rk - 1],
                             addr=pfm['addr'].to_numpy(dtype=object), mk_hash=hs, y_side=ys, tick=tk, n_fills=pfm['n_fills'].to_numpy(np.int64),
                             window_lo_ms=plo, post_hi_ms=pfm['post_hi_ms'].to_numpy(np.int64), window_status=pfm['window_status'].to_numpy(dtype=object),
                             post_grade=pfm['post_grade'].to_numpy(dtype=object), occupancy=occ,
                             size_at_window_lo=np.where(bnd & okc & (lst == 'ok'), lsz, NAN), frame_ms_at_window_lo=fms,
                             frame_age_ms=np.where(bnd & (lfr >= 0), lage, -1), lookup_status=np.where(bnd, lst, '').astype(object),
                             is_front_candidate_hash=np.array([h in cand_hashes for h in hs], bool)))[list(OCC_COLS)]


def market_pass(day, mid, M, fvm, pfm):
    """One market: v1's book and fills, the v2 core, the occupancy rows and v1's declared null draws -> a part."""
    bk = QF.market_book(M)
    fl, fcnt = QF.market_fills(M, fvm)
    core = market_core(str(day), int(mid), int(M.ws), int(M.we), bk, fl)
    fr = core['fills']
    fh = set(fr.loc[(fr['role'] == 'earliest_set').to_numpy(), 'mk_hash'].tolist()) if len(fr) else set()
    oc = occupancy_rows(day, mid, M, pfm, fh)
    nul = QF.null_market(day, mid, core['lab'], core['front'])
    cnt = Counter(fcnt)
    cnt.update(core['counts'])
    return dict(day=str(day), mid=int(mid), ws=int(M.ws), we=int(M.we), levels=core['levels'], fills=fr, occupancy=oc, null=nul, counts=cnt,
                checks=core['checks'], parity=core['parity'])


# ------------------------------------------------------------------ tables ------------------------------------------------------------------
def _sel(D, sc, d, m=None):
    if sc == 'pooled':
        return D
    msk = (D['day'] == d).to_numpy()
    if sc == 'market':
        msk = msk & (D['mid'] == m).to_numpy()
    return D[msk]


def shares_table(LV, FL, NL, scopes):
    """Front-win bounds per scope x group over ONE denominator (front_candidate + ambiguous_order levels): scenario_lower = the group's
    front candidates, scenario_upper = + the ambiguous levels whose address set holds the group (within the no-netting scenario); alt_lower /
    alt_upper = the levels whose masked-netting first-filler addresses all lie in / include the group; the fill share and p_hi over the
    attributed fills of front_candidate levels (v1's null); every status count beside."""
    rows = []
    for sc, d, m in scopes:
        keys = [k for k in NL if sc == 'pooled' or (k[0] == d and (sc == 'day' or k[1] == m))]
        Lm, Fm = _sel(LV, sc, d, m), _sel(FL, sc, d, m)
        E1 = Lm[(Lm['unit'] == 'E1').to_numpy()]
        so = Counter(Lm.loc[(Lm['unit'] == 'status_only').to_numpy(), 'status'].tolist())
        stc = Counter(E1['status'].tolist())
        nfc = stc[FRONT]
        FC = E1[(E1['status'] == FRONT).to_numpy()]
        AM = E1[(E1['status'] == 'ambiguous_order').to_numpy()]
        den = nfc + len(AM)
        amv = AM['amb_gmask'].to_numpy(np.int64)
        altg = np.r_[FC['alt_gmask'].to_numpy(np.int64), AM['alt_gmask'].to_numpy(np.int64)]
        qall = Fm['q'].to_numpy(np.float64) if len(Fm) else np.zeros(0)
        base = dict(scope=sc, day=d, mid=m if sc == 'market' else -1, n_markets=len(keys), n_candidates=int(len(Lm)),
                    n_status_only_gap=so['gap'], n_status_only_unknown=so['unknown'], n_fresh=int(len(E1)), **{f'n_{s}': stc[s] for s in STATUSES},
                    n_front_candidate_book=int((FC['timing_basis'] == 'book_candidate').sum()),
                    n_front_candidate_settlement=int((FC['timing_basis'] == 'settlement').sum()),
                    n_alt_netting_possible=int(E1['alt_netting_possible'].to_numpy(bool).sum()),
                    n_bound_alt_netting_possible=int(FC['alt_netting_possible'].to_numpy(bool).sum() + AM['alt_netting_possible'].to_numpy(bool).sum()),
                    n_front_candidate_alt_other_address=int(FC['alt_first_other_address'].to_numpy(bool).sum()))
        for gname in SCOPE_GROUPS:
            nfg = int(QF._gmask(FC['candidate_group'], gname).sum())
            nag = int(((amv & GBIT[gname]) != 0).sum())
            nf = int(len(Fm))
            gm = QF._gmask(Fm['group'], gname)
            nfgf = int(gm.sum())
            obs, nul, k = QF._null_sum(NL, keys, gname)
            need(obs == nfg and k == nfc, f'internal: null counts ({obs}, {k}) differ from the level table ({nfg}, {nfc}) in {sc} {d} {m}')
            ok_ = bool(k and den)
            ns = nul / den if ok_ else np.full(N_PERM, NAN)
            alo = int(((altg != 0) & ((altg & ~GBIT[gname]) == 0)).sum())
            ahi = int(((altg & GBIT[gname]) != 0).sum())
            r = dict(base, group=gname, n_front_group=nfg, n_ambiguous_with_group=nag, bound_denominator=den,
                     scenario_lower=nfg / den if den else NAN, scenario_upper=(nfg + nag) / den if den else NAN, n_alt_lower=alo, n_alt_upper=ahi,
                     alt_lower=alo / den if den else NAN, alt_upper=ahi / den if den else NAN,
                     share_of_front_candidates=nfg / nfc if nfc else NAN, share_of_fresh=nfg / len(E1) if len(E1) else NAN, n_fills=nf,
                     n_fills_group=nfgf, fill_share=nfgf / nf if nf else NAN,
                     fill_share_q=float(qall[gm].sum()) / float(qall.sum()) if float(qall.sum()) > 0 else NAN,
                     null_mean=float(np.mean(ns)) if ok_ else NAN, null_p05=float(np.quantile(ns, 0.05)) if ok_ else NAN,
                     null_p50=float(np.quantile(ns, 0.5)) if ok_ else NAN, null_p95=float(np.quantile(ns, 0.95)) if ok_ else NAN,
                     p_hi=QF.p_hi_of(obs, nul) if k else NAN, n_perm=N_PERM)
            r['front_minus_fill'] = r['scenario_lower'] - r['fill_share'] if (den and nf) else NAN
            rows.append(r)
    return pd.DataFrame(rows)


def mechanism_table(LV, pscopes):
    """The anonymous cell-creation context: among each group's front candidates (with the flags and the masked-netting other-address
    count) and among every fresh level by status (denominators include no_fill / unknown / gap levels)."""
    rows = []
    E1 = LV[(LV['unit'] == 'E1').to_numpy()]
    for sc, d in pscopes:
        Es = _sel(E1, sc, d)
        FC = Es[(Es['status'] == FRONT).to_numpy()]
        for gname in ('all',) + SCOPE_GROUPS:
            Fg = FC if gname == 'all' else FC[QF._gmask(FC['candidate_group'], gname)]
            nn = int(len(Fg))
            for ctx in CONTEXTS:
                k = int((Fg['creation_context'] == ctx).sum())
                rows.append(dict(table='candidate_creation_context', scope=sc, day=d, group=gname, status=FRONT, category=ctx, n=k, denom=nn,
                                 share=k / nn if nn else NAN))
            for fg in FLAGS + ('alt_first_other_address',):
                k = int(Fg[fg].to_numpy(dtype=bool).sum()) if nn else 0
                rows.append(dict(table='candidate_flags', scope=sc, day=d, group=gname, status=FRONT, category=fg, n=k, denom=nn,
                                 share=k / nn if nn else NAN))
        for s_ in STATUSES + ('all_fresh',):
            Ss = Es if s_ == 'all_fresh' else Es[(Es['status'] == s_).to_numpy()]
            nn = int(len(Ss))
            for ctx in CONTEXTS:
                k = int((Ss['creation_context'] == ctx).sum())
                rows.append(dict(table='fresh_creation_context', scope=sc, day=d, group='any', status=s_, category=ctx, n=k, denom=nn,
                                 share=k / nn if nn else NAN))
    return pd.DataFrame(rows)


def speed_table(LV, pscopes):
    """The cell-event endpoint proxy (re_establish / improve) with its lower / upper interval: n and quantiles of each column."""
    rows = []
    E1 = LV[(LV['unit'] == 'E1').to_numpy()]
    for sc, d in pscopes:
        Es = _sel(E1, sc, d)
        for ctx in ('re_establish', 'improve'):
            Em = Es[(Es['creation_context'] == ctx).to_numpy()]
            FC = Em[(Em['status'] == FRONT).to_numpy()]
            for gname in ('all_fresh', 'all_candidates') + SCOPE_GROUPS:
                X = Em if gname == 'all_fresh' else (FC if gname == 'all_candidates' else FC[QF._gmask(FC['candidate_group'], gname)])
                rows.append(dict(scope=sc, day=d, creation_context=ctx, group=gname, n_levels=int(len(X)), **QF._qd(X['speed_ms'], 'speed_ms'),
                                 **QF._qd(X['speed_lo_ms'], 'speed_lo_ms'), **QF._qd(X['speed_hi_ms'], 'speed_hi_ms')))
    return pd.DataFrame(rows)


def econ_table(FL, pscopes):
    """GROSS markouts (v1's _econ: c/sh = 100 sum v / sum q over valid rows) per role earliest_set / later_other / later_same_hash, by the
    level's candidate group (and creation context) and by the filling maker's group, with all / valid rows and shares."""
    X0 = FL[['day', 'role', 'group', 'candidate_group', 'creation_context', 'q', 'excl_reason'] + [f'v_{h}' for h in HORIZONS]].assign(
        cohort=FL['valid'].to_numpy(dtype=bool), q_c=FL['q_valid'].to_numpy(np.float64))
    rows = []
    for sc, d in pscopes:
        Fs = _sel(X0, sc, d).reset_index(drop=True)
        rv, cv = Fs['role'].to_numpy(dtype=object), Fs['creation_context'].to_numpy(dtype=object)
        for by, gcol in (('candidate', 'candidate_group'), ('filler', 'group')):
            for gname in ('all',) + SCOPE_GROUPS:
                gm = np.ones(len(Fs), bool) if gname == 'all' else QF._gmask(Fs[gcol], gname)
                for ctx in (('all',) + CONTEXTS if by == 'candidate' else ('all',)):
                    cm = gm if ctx == 'all' else gm & (cv == ctx)
                    for role in ROLES:
                        o = QF._econ(Fs[cm & (rv == role)])
                        rows.append(dict(scope=sc, day=d, by=by, group=gname, creation_context=ctx, role=role, all_rows=o['n_fills'],
                                         valid_rows=o['cohort_rows'], valid_row_share=o['cohort_rows'] / o['n_fills'] if o['n_fills'] else NAN,
                                         all_shares=o['shares'], valid_shares=o['cohort_shares'],
                                         valid_share_share=o['cohort_shares'] / o['shares'] if o['shares'] > 0 else NAN,
                                         v_1s_c_sh=o['v_1s_c_sh'], v_10s_c_sh=o['v_10s_c_sh'], v_60s_c_sh=o['v_60s_c_sh'], excluded=o['excluded']))
    return pd.DataFrame(rows)


def occupancy_summary(OC, pscopes):
    """Counts per scope x group x occupancy status with the denominator (every top-3 filled order) and the post-grade split."""
    out = []
    grades = sorted(set(OC['post_grade'].astype(str).tolist())) if len(OC) else []
    for sc, d in pscopes:
        Os = _sel(OC, sc, d)
        for gname in ('rank1', 'rank2', 'rank3', 'top3'):
            O = Os[QF._gmask(Os['group'], gname)]
            nn = int(len(O))
            for s_ in OCC_ST:
                X = O[(O['occupancy'] == s_).to_numpy()]
                out.append(dict(scope=sc, day=d, group=gname, occupancy=s_, n=int(len(X)), denom=nn, share=len(X) / nn if nn else NAN,
                                n_front_candidate_hash=int(X['is_front_candidate_hash'].to_numpy(dtype=bool).sum()) if len(X) else 0,
                                by_grade={gr: int((X['post_grade'].astype(str) == gr).sum()) for gr in grades}))
    return out


def readings(LV, SH, markets):
    """Section 5.6 and the creation-context mix, computed per rank1 and top3 pooled (verdicts from predicates evaluated here)."""
    E1 = LV[(LV['unit'] == 'E1').to_numpy()]
    FC = E1[(E1['status'] == FRONT).to_numpy()]
    MK = SH[(SH['scope'] == 'market').to_numpy()]
    T3 = MK[(MK['group'] == 'top3').to_numpy()]
    sup = [(str(r['day']), int(r['mid'])) for r in T3.to_dict('records') if int(r['n_front_candidate']) >= MIN_FRONTS]
    n_sup = len(sup)
    out = []
    for scope in READ_SCOPES:
        pr_ = SH[((SH['scope'] == 'pooled') & (SH['group'] == scope)).to_numpy()].iloc[0].to_dict()
        lower, upper, fill, p = pr_['scenario_lower'], pr_['scenario_upper'], pr_['fill_share'], pr_['p_hi']
        den, nf, n_alt = int(pr_['bound_denominator']), int(pr_['n_fills']), int(pr_['n_bound_alt_netting_possible'])
        Mg = {(str(r['day']), int(r['mid'])): r for r in MK[(MK['group'] == scope).to_numpy()].to_dict('records')}
        n_gt = sum(1 for key in sup if np.isfinite(Mg[key]['front_minus_fill']) and Mg[key]['front_minus_fill'] > 0)
        fin = bool(np.isfinite(lower) and np.isfinite(fill))
        pr = dict(scenario_lower_gt_fill_share=bool(fin and lower > fill), p_hi_le_0_05=bool(np.isfinite(p) and p <= P_MAX),
                  supported_markets_ge_5=n_sup >= MIN_MARKETS, market_share_ge_2_3=bool(n_sup and SHARE_DEN * n_gt >= SHARE_NUM * n_sup),
                  no_alt_netting=n_alt == 0)
        v = 'insufficient_support' if (n_sup < MIN_MARKETS or not (den and nf)) else ('wins_front_more_than_fills' if all(pr.values()) else 'unidentified')
        out.append(dict(scope=scope, reading='front_win_scenario_bound', verdict=v, predicates=pr,
                        beside=dict(fill_share_within_scenario_bound=bool(fin and np.isfinite(upper) and lower <= fill <= upper),
                                    fill_share_within_alt_bound=bool(np.isfinite(pr_['alt_lower']) and np.isfinite(fill)
                                                                     and pr_['alt_lower'] <= fill <= pr_['alt_upper'])),
                        values=dict(scenario_lower=lower, scenario_upper=upper, alt_lower=pr_['alt_lower'], alt_upper=pr_['alt_upper'], fill_share=fill,
                                    fill_share_q=pr_['fill_share_q'], n_fresh=int(pr_['n_fresh']), n_front_candidate=int(pr_['n_front_candidate']),
                                    n_ambiguous_order=int(pr_['n_ambiguous_order']), n_unexplained_decrease=int(pr_['n_unexplained_decrease']),
                                    n_no_fill=int(pr_['n_no_fill']), n_unknown=int(pr_['n_unknown']), n_gap=int(pr_['n_gap']), bound_denominator=den,
                                    n_front_group=int(pr_['n_front_group']), n_ambiguous_with_group=int(pr_['n_ambiguous_with_group']),
                                    n_alt_lower=int(pr_['n_alt_lower']), n_alt_upper=int(pr_['n_alt_upper']), n_fills=nf,
                                    n_fills_group=int(pr_['n_fills_group']), n_bound_alt_netting_possible=n_alt, p_hi=p, n_perm=N_PERM,
                                    n_supported_markets=n_sup, n_markets_lower_gt_fill=n_gt),
                        description='[scenario_lower, scenario_upper] is a bound within the declared reconstruction scenario (no-netting book '
                                    'candidates, one-address earliest sets), not a certified real-world bound; alt_* = the masked-netting '
                                    'first-filler sets over the same levels; fill share and p_hi over the attributed fills of front_candidate '
                                    'levels (a conditional label shuffle after selection, not an unconditional chance of winning a front)'))
        Fg = FC[QF._gmask(FC['candidate_group'], scope)]
        tot = int(len(Fg))
        cn = {ctx: int((Fg['creation_context'] == ctx).sum()) for ctx in CONTEXTS}
        topn = max(cn.values()) if tot else 0
        tops = [ctx for ctx in CONTEXTS if tot and cn[ctx] == topn]
        pr2 = dict(group_candidates_ge_5=tot >= MIN_WINS, supported_markets_ge_5=n_sup >= MIN_MARKETS, unique_largest=len(tops) == 1)
        v2 = ('insufficient_support' if not (pr2['group_candidates_ge_5'] and pr2['supported_markets_ge_5'])
              else (tops[0] if pr2['unique_largest'] else 'unidentified'))
        out.append(dict(scope=scope, reading='largest_creation_context', verdict=v2, predicates=pr2,
                        beside=dict(largest_is_majority=bool(tot and 2 * topn > tot)),
                        values=dict(n_front_candidates=tot, counts=cn, shares={ctx: (cn[ctx] / tot if tot else NAN) for ctx in CONTEXTS},
                                    n_supported_markets=n_sup),
                        description='the anonymous cell-creation context (frame f - 1) associated with a candidate first filler; not the '
                                    "filler's posting action (the eventual first filler is not necessarily the creator)"))
    return out


def analyse(kind, days, markets, parts, cnt):
    """Parts -> (tables, claims, digest); every aggregate is checked against the saved keyed rows (refused on a mismatch)."""
    LV = QF._cat([p['levels'] for p in parts], LEVEL_COLS)
    FL = QF._cat([p['fills'] for p in parts], FILL_COLS)
    OC = QF._cat([p['occupancy'] for p in parts], OCC_COLS)
    NL = {(p['day'], p['mid']): p['null'] for p in parts}
    scopes = [('pooled', 'all', -1)] + [('day', d, -1) for d in days] + [('market', d, m) for d, m in markets]
    pscopes = [('pooled', 'all')] + [('day', d) for d in days]
    SH = shares_table(LV, FL, NL, scopes)
    E1 = LV[(LV['unit'] == 'E1').to_numpy()]
    CD = E1[list(CAND_COLS)].reset_index(drop=True)
    tables = {'levels.csv': LV, 'candidates.csv': CD, 'fills.csv': FL, 'shares.csv': SH, 'mechanism.csv': mechanism_table(LV, pscopes),
              'speed.csv': speed_table(LV, pscopes), 'economics.csv': econ_table(FL, pscopes), 'occupancy.csv': OC}
    rd = readings(LV, SH, markets)
    FC = E1[(E1['status'] == FRONT).to_numpy()]
    WS = E1[(E1['timing_basis'] != '').to_numpy()]
    n_sup = int(rd[0]['values']['n_supported_markets']) if rd else 0
    # ---- reconciliation of the aggregates with the saved keyed rows (computed; a mismatch refuses) ----
    EC = tables['economics.csv']
    pool = SH[((SH['scope'] == 'pooled') & (SH['group'] == 'top3')).to_numpy()].iloc[0]
    ec_all = EC[((EC['scope'] == 'pooled') & (EC['by'] == 'candidate') & (EC['group'] == 'all') & (EC['creation_context'] == 'all')).to_numpy()]
    rec = dict(candidates_keyed_unique=bool(pd.MultiIndex.from_frame(CD[['day', 'mid', 'level_id']]).is_unique
                                            and pd.MultiIndex.from_frame(CD[['day', 'mid', 'side', 'tick', 'frame']]).is_unique) if len(CD) else True,
               candidates_eq_fresh=int(len(CD)) == int(pool['n_fresh']),
               statuses_add_up=int(sum(int(pool[f'n_{s}']) for s in STATUSES)) == int(pool['n_fresh']),
               economics_rows_eq_fills=all(int(ec_all.loc[(ec_all['role'] == r_).to_numpy(), 'all_rows'].iloc[0]) == int((FL['role'] == r_).sum())
                                           for r_ in ROLES),
               economics_valid_shares_eq_fills=all(abs(float(ec_all.loc[(ec_all['role'] == r_).to_numpy(), 'valid_shares'].iloc[0])
                                                       - float(FL.loc[((FL['role'] == r_) & FL['valid'].astype(bool)).to_numpy(), 'q_valid'].sum())) < 1e-6
                                                   for r_ in ROLES),
               one_earliest_fill_per_candidate=int(FL['null_front'].astype(bool).sum()) == int(len(FC)),
               levels_later_eq_fills=int(FC['n_later_same_hash'].sum()) == int((FL['role'] == 'later_same_hash').sum())
               and int(FC['n_later_other'].sum()) == int((FL['role'] == 'later_other').sum()))
    for k_, v_ in rec.items():
        need(bool(v_), f'reconciliation failed: {k_} (refused)')
    pt = dict(n_levels=0, n_affected=0, n_violations=0, by_key=Counter(), affected_transitions=Counter(), bad_levels=[])
    for p in parts:
        q_ = p['parity']
        for k_ in ('n_levels', 'n_affected', 'n_violations'):
            pt[k_] += int(q_[k_])
        pt['by_key'].update(q_['by_key'])
        pt['affected_transitions'].update(q_['affected_transitions'])
        pt['bad_levels'] += [f"{p['day']}_{p['mid']}_{lid}" for lid in q_.get('bad_level_ids', [])][:10]
    pt['by_key'], pt['affected_transitions'] = dict(pt['by_key']), dict(pt['affected_transitions'])
    pt['bad_levels'] = pt['bad_levels'][:20]
    occ = occupancy_summary(OC, pscopes)
    alt_by = {s: dict(n=int((E1['status'] == s).sum()), alt_netting_possible=int(E1.loc[(E1['status'] == s).to_numpy(), 'alt_netting_possible'].astype(bool).sum()),
                      alt_first_other_address=int(E1.loc[(E1['status'] == s).to_numpy(), 'alt_first_other_address'].astype(bool).sum())) for s in STATUSES}
    nfc = int(len(FC))
    pop = dict(markets=len(markets), candidates_rows=int(len(LV)), candidates_after_window_end=int(cnt.get('candidates_after_window_end', 0)),
               status_only=dict(Counter(LV.loc[(LV['unit'] == 'status_only').to_numpy(), 'status'].tolist())), fresh_levels=int(len(E1)),
               fresh_by_status={s: int((E1['status'] == s).sum()) for s in STATUSES}, fresh_by_reason=dict(Counter(E1['reason'].tolist())),
               front_candidate_by_basis={b_: int((FC['timing_basis'] == b_).sum()) for b_ in BASES},
               alt=dict(levels_with_earliest_record=int(len(WS)), alt_netting_possible=int(WS['alt_netting_possible'].astype(bool).sum()),
                        front_candidates=nfc, front_candidate_alt_netting_possible=int(FC['alt_netting_possible'].astype(bool).sum()),
                        front_candidate_alt_netting_share=float(FC['alt_netting_possible'].astype(bool).mean()) if nfc else None,
                        front_candidate_alt_first_other_address=int(FC['alt_first_other_address'].astype(bool).sum()),
                        front_candidate_alt_first_other_share=float(FC['alt_first_other_address'].astype(bool).mean()) if nfc else None,
                        by_status=alt_by),
               record_timing={s_: int(cnt.get(f'rec_{s_}', 0)) for s_ in REC_ST}, records_unbounded_never_timed=int(cnt.get('rec_unbounded', 0)),
               straddle=dict(v1_settlement_only_records=int(cnt.get('rec_v1_settlement_only', 0)),
                             v1_settlement_records_untimed_in_v2=int(cnt.get('rec_straddle_untimed', 0)),
                             levels_with_straddle_record=int(E1['straddle_record'].astype(bool).sum())),
               attributed_fills={r_: int((FL['role'] == r_).sum()) for r_ in ROLES},
               attributed_shares={r_: float(FL.loc[(FL['role'] == r_).to_numpy(), 'q'].sum()) for r_ in ROLES},
               levels_old_life_excluded=int((E1['n_old_life'].to_numpy(np.int64) > 0).sum()) if len(E1) else 0,
               legs=dict(grid=int(cnt.get('legs_grid', 0)), member_of_a_fresh_level=int(cnt.get('legs_member_of_fresh_level', 0)), attributed=int(len(FL))),
               occupancy_orders=int(len(OC)), supported_markets=n_sup)
    labels = dict(
        status='front_candidate = the one-address earliest set under the no-netting premise; never certified (review finding 1)',
        timing='book_candidate = the unique net decrease equal to the record quantity and claimed by no other record: a conditional timing candidate; '
               'an anonymous cancellation, replenishment or their netting inside a frame interval is not ruled out',
        alt='alt_netting_possible: intervals where a masked cancellation / replenishment could hide (creation interval through the candidate '
            'interval, plus unexplained changes up to the candidate record hi); alt_first_addresses: every first-filler address the hard '
            'windows and the level life allow (conservative, may over-include)',
        roles='earliest_set = the earliest record(s) under the premise; later_same_hash = a later fill of an order hash of the earliest set (the '
              'same order: no queue loss shown); later_other = later fills of other hashes; "does the front pay?" withdrawn (contract 5.3)',
        economics='GROSS markouts from settle_ms; valid = chain_post_fill PNL cohort (status ok, not a self-trade, st_res valid, 1 s / 10 s / '
                  '60 s marks ok); inherited settlement-clock values, not values at an identified private match or posting decision; censoring '
                  'and price / time composition are not matched',
        occupancy='cell_occupied_at_window_lo / cell_empty_at_window_lo = the displayed size at the inferred posting-window lower endpoint, '
                  'conditional on that window, with the post grade; never an actual-post queue label; population = top-3 orders that later filled',
        mechanism='creation_context = the anonymous cell-creation context associated with a candidate first filler',
        speed='speed_ms = the cell-event endpoint proxy with speed_lo_ms / speed_hi_ms; not a measured maker latency; endpoint ownership not '
              'assigned; no faster_than_others verdict',
        front_win='a bound within the declared reconstruction scenario; unidentified unless no counted level carries alternative histories',
        days='four consumed days: point estimates, no intervals (G = 4 < 5); QUEUE-FRONT v1 amendments 1-2 were adaptive consumed-day development')
    claims = dict(version=VERSION, kind=kind, contract=dict(file=QF._rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, section='5'), review=REVIEW,
                  v1=dict(version=QF.VERSION, contract_commit=QF.CONTRACT_COMMIT, amendments=QF.AMENDMENT_COMMIT), days=list(days),
                  markets=[f'{d}_{m}' for d, m in markets], population=pop, parity=pt, reconciliation=rec, readings=rd, occupancy=occ,
                  economics_pooled=EC[((EC['scope'] == 'pooled') & (EC['creation_context'] == 'all')
                                       & EC['group'].isin(['all', 'rank1', 'top3'])).to_numpy()].to_dict('records'),
                  withdrawn=dict(front_pays='queue-effect inference withdrawn (contract 5.3): economics are descriptive by role',
                                 faster_than_others='not re-issued: the speed is a cell-event endpoint proxy (contract 5.5)',
                                 certified_front='no level is certified: every holder is a front_candidate (contract 5.1)'),
                  null=dict(design="v1's: within each market the maker labels of every attributed fill of the front_candidate levels permuted; the "
                                   'designated earliest fills keep their positions (a conditional label shuffle after selection)', n_perm=N_PERM, seed=SEED),
                  rules=dict(min_front_candidates_per_supported_market=MIN_FRONTS, min_supported_markets=MIN_MARKETS, min_group_candidates=MIN_WINS,
                             market_share=f'{SHARE_NUM}/{SHARE_DEN}', p_max=P_MAX, gap_ms=GAP_MS, max_lag_ms=MAX_LAG_MS, tol_sh=TOL, reest_ms=REEST_MS),
                  multiplicity=dict(readings=2, scopes=len(READ_SCOPES), all_listed=True, selected=None), labels=labels, descriptive_only=True,
                  intervals=None, fill_to_cell={k: int(v) for k, v in sorted(cnt.items()) if k.startswith('cellcheck_') or k in QF.FILL_JOIN_KEYS})
    return tables, claims, digest(claims, tables)


def digest(claims, tables):
    """The pilot / run print: the key numbers (no verdict that the tables and claims do not carry)."""
    EC = tables['economics.csv']
    eco = {}
    if len(EC):
        Rr = EC[((EC['scope'] == 'pooled') & (EC['by'] == 'candidate') & (EC['creation_context'] == 'all') & EC['group'].isin(['all', 'rank1', 'top3'])).to_numpy()]
        eco = {f"{r['group']}:{r['role']}": dict(all_rows=r['all_rows'], valid_rows=r['valid_rows'], valid_shares=round(float(r['valid_shares']), 2),
                                                valid_share_share=r['valid_share_share'], v_10s_c_sh=r['v_10s_c_sh'], v_60s_c_sh=r['v_60s_c_sh'])
               for r in Rr.to_dict('records')}
    occ = {f"{r['group']}:{r['occupancy']}": dict(n=r['n'], denom=r['denom'], by_grade=r['by_grade'])
           for r in claims['occupancy'] if r['scope'] == 'pooled' and r['group'] == 'top3'}
    SH = tables['shares.csv']
    pool = SH[(SH['scope'] == 'pooled').to_numpy()] if len(SH) else SH
    sh = {r['group']: dict(scenario=[r['scenario_lower'], r['scenario_upper']], alt=[r['alt_lower'], r['alt_upper']], den=r['bound_denominator'],
                           fill_share=r['fill_share'], n_fills=r['n_fills'], p_hi=r['p_hi']) for r in pool.to_dict('records')}
    return dict(population=claims['population'], parity=claims['parity'], reconciliation=claims['reconciliation'], shares_pooled=sh, economics=eco,
                occupancy_top3=occ, readings={f"{r['scope']}:{r['reading']}": r['verdict'] for r in claims['readings']})


# ------------------------------------------------------------------ receipt, write-once, run / pilot ------------------------------------------------------------------
def receipt(kind, days, markets, hashes, cnt, stages, t0, asof0, outs, git, extra=None):
    srcs = {QF._rel(f_): QF._sha(f_) for f_ in TOOL_FILES + DEP_FILES if Path(f_).exists()}
    rec = dict(version=VERSION, kind=kind, contract=dict(file=QF._rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, section='5',
                                                         sha256=srcs.get(QF._rel(CONTRACT_FILE))),
               review=REVIEW, v1=dict(file=QF._rel(HERE / 'queue_front.py'), version=QF.VERSION, sha256=srcs.get(QF._rel(HERE / 'queue_front.py'))),
               module=str(Path(__file__).resolve()), argv=list(sys.argv), run=C.RUN, mutant=os.environ.get(MUT_ENV, ''), asof_start=asof0,
               asof_end=QF._now(), days=list(days), markets=[f'{d}_{m}' for d, m in markets], addresses=list(TOP3), inputs=dict(sorted(hashes.items())),
               sources=srcs, outputs=outs, git=git, counts=dict(sorted((str(k), int(v)) for k, v in cnt.items())),
               constants=dict(NT=NT, GAP_MS=GAP_MS, MAX_LAG_MS=MAX_LAG_MS, TOL=TOL, REEST_MS=REEST_MS, SEED=SEED, N_PERM=N_PERM, MIN_FRONTS=MIN_FRONTS,
                              MIN_MARKETS=MIN_MARKETS, MIN_WINS=MIN_WINS, SHARE=f'{SHARE_NUM}/{SHARE_DEN}', P_MAX=P_MAX, FV_SHA256=FV_SHA256,
                              PROJ_MAX_S=PROJ_MAX_S, STATUSES=STATUSES, REC_ST=REC_ST, ROLES=ROLES, OCC_ST=OCC_ST, MUTANTS=MUTANTS, PILOT_RUN=PILOT_RUN),
               cpu_s=dict(by_stage=dict(stages), process_total=round(time.process_time(), 3)), wall_s=round(time.time() - t0, 2),
               vmhwm_mib=round(QF.hwm(), 1), peak_rss='VmHWM of this process', guard_mib=C.GUARD_MIB)
    rec.update(extra or {})
    return rec


def run_core(kind, days, markets, parts, out, hashes, cnt, stages, tick, git, asof0, t0, extra_fn=None, require_parity=False):
    """Shared tail of the run / pilot (and the selftest's end-to-end): tables, claims, write-once (v1's write_out)."""
    for p in parts:
        for k_, v_ in p['checks'].items():
            cnt[f'cellcheck_{k_}'] += int(v_)
        cnt.update(p['counts'])
    tables, claims, dg = analyse(kind, days, markets, parts, cnt)
    tick('tables')
    QF.guard('tables')
    par = claims['parity']
    if require_parity:
        need(par['n_violations'] == 0, f"{par['n_violations']} levels differ from v1 outside the straddle fix (refused: parity)")

    def rec_fn(outs):
        tick('write')
        ex = dict(extra_fn(stages) if extra_fn else {})
        ex['parity'] = dict(n_levels=par['n_levels'], n_affected=par['n_affected'], n_violations=par['n_violations'])
        return receipt(kind, days, markets, hashes, cnt, stages, t0, asof0, outs, git, ex)
    rec = QF.write_out(out, tables, claims, rec_fn)
    return rec, claims, dg


def preflight_projection(path, hashes):
    """The full run launches only from a recorded v2 pilot whose projection is <= 120 CPU s and whose v1 parity held."""
    path = Path(path)
    need(path.is_file(), f'no recorded pilot projection: {path} is missing (run the pilot first; refused before launch)')
    J = json.loads(path.read_text())
    pr = (J.get('projection') or {}).get('full_run_cpu_s')
    need(J.get('kind') == 'pilot' and J.get('version') == VERSION and isinstance(pr, (int, float)) and np.isfinite(pr),
         f'{path.name}: not a {VERSION} pilot receipt with a finite projection.full_run_cpu_s (refused before launch)')
    need(float(pr) <= PROJ_MAX_S, f'the recorded pilot projects {float(pr):.1f} CPU s for the full run > {PROJ_MAX_S:.0f} s (refused before launch)')
    nv = (J.get('parity') or {}).get('n_violations')
    need(nv == 0, f'the recorded pilot carries {nv} v1-parity violations (refused before launch)')
    hashes[QF._rel(path)] = QF._sha(path)
    return dict(pilot_receipt=QF._rel(path), full_run_cpu_s=float(pr), limit_s=PROJ_MAX_S)


def execute(kind, days, out, inp=None, pilot_receipt=None, require_freeze=True, log=print, loader=None, book=None, pf_dir=None, rg_dir=None,
            fv_sha=FV_SHA256):
    """kind 'run' (every measurement day -> v2/chains/queue_front_v2; freeze + pilot projection required) or 'pilot' (its day, out given).
    loader / book / pf_dir / rg_dir / fv_sha are injected only by the selftest (production: chain_context.load_market, the BOOK manifest,
    v2/chains/post_fill, v2/chains/correction_v1, the accepted fills_valued binding)."""
    c_imp = time.process_time()
    t0, asof0 = time.time(), QF._now()
    need(mutant() == '', f"{MUT_ENV}={os.environ.get(MUT_ENV)!r} is set: production never runs a planted error (refused before any input is read)")
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
    git = QF._git([f_ for f_ in TOOL_FILES if Path(f_).exists()])
    pre = {}
    if kind == 'run':
        need(days == tuple(C.MEASURE_DAYS), 'the full run covers every measurement day (refused)')
        if require_freeze:
            need(all(Path(f_).exists() for f_ in TOOL_FILES) and bool(git.get('unmodified')),
                 'freeze: queue_front_v2.py, its selftest, the contract, queue_front.py and chain_context.py must be tracked and unmodified (refused)')
        pre = preflight_projection(Path(pilot_receipt) if pilot_receipt else PT['pilot_receipt'], hashes)
    inp = Path(inp) if inp else C.OUT
    t3 = CX.top3(inp)
    need(tuple(a.lower() for a in t3) == TOP3_L, 'chain_context.top3 differs from the module top 3 (refused)')
    hashes[QF._rel(inp / 'econ' / 'identities.json')] = QF._sha(inp / 'econ' / 'identities.json')
    bm = CX.book_markets() if book is None else book
    need(all(d in bm for d in days), f'days {days} are not all BOOK manifest days (refused)')
    markets = [(d, int(m)) for d in days for m in sorted(int(x) for x in bm[d])]
    tick('preflight')
    pfd = PT['post_fill'] if pf_dir is None else Path(pf_dir)
    PF, pfinfo = QF.read_post_fill(pfd, markets, hashes, fv_sha)
    tick('post_fill')
    RG, rginfo = read_regrade(PT['regrade'] if rg_dir is None else rg_dir, markets, hashes, hashes[QF._rel(pfd / 'orders.csv')])
    PF = join_grades(PF, RG)
    tick('regrade')
    fvp = inp / 'econ' / 'fills_valued.parquet'
    FV = QF.read_fv_makers(fvp, markets, cnt)
    fv_mid = FV['mid'].to_numpy(np.int64) if len(FV) else np.zeros(0, np.int64)
    pf_key = list(zip(PF['day'].to_numpy(dtype=object), PF['mid'].to_numpy(np.int64))) if len(PF) else []
    tick('fills_valued')
    loader = QF._load_default if loader is None else loader
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
        QF.guard(f'market {d} {m}')
    fk = str(fvp.resolve())
    fsha = hc.get(fk) or QF._sha(fvp)
    hashes['econ/fills_valued.parquet'] = fsha
    need(fsha == fv_sha, f'fills_valued sha256 {fsha[:12]} differs from the accepted binding {str(fv_sha)[:12]} (refused)')
    need(pfinfo['fills_valued'] == fsha, "fills_valued differs from chain_post_fill's receipt (refused: mismatched artifacts)")
    cnt['post_fill_rows'], cnt['post_fill_rows_run'] = pfinfo['rows'], pfinfo['rows_run']
    cnt['regrade_rows'], cnt['regrade_rows_run'] = rginfo['rows'], rginfo['rows_run']
    tick('bindings')
    n_mk = len(markets)

    def extra_fn(stg):
        ex = dict(preflight=pre, post_fill=pfinfo, regrade=rginfo)
        if kind == 'pilot':
            r_mk = N_BOOK / max(1, n_mk)
            fixed = sum(stg.get(k_, 0.) for k_ in ('imports', 'preflight', 'post_fill', 'regrade', 'bindings'))
            scal = sum(stg.get(k_, 0.) for k_ in ('fills_valued', 'context_load', 'market', 'tables', 'write'))
            proj = fixed + scal * r_mk
            ex['projection'] = dict(full_run_cpu_s=round(proj, 2), pilot_cpu_s=round(time.process_time(), 3), limit_s=PROJ_MAX_S,
                                    launch_allowed=bool(proj <= PROJ_MAX_S),
                                    rule=f'fixed (imports + preflight + post_fill + regrade + bindings) {fixed:.2f} + (fills_valued + context_load + '
                                         f'market + tables + write) {scal:.2f} x {N_BOOK}/{n_mk}; the first load carries the one-off fills_valued '
                                         f'sha256 (scaled: conservative)')
        return ex
    rec, claims, dg = run_core(kind, days, markets, parts, out, hashes, cnt, stages, tick, git, asof0, t0, extra_fn=extra_fn,
                               require_parity=(kind == 'run'))
    log(json.dumps(QF._clean(dict(out=str(out), digest=dg, projection=rec.get('projection'), cpu_s=rec['cpu_s'], vmhwm_mib=rec['vmhwm_mib'])),
                   indent=1, sort_keys=True, default=str))
    return rec


def main(argv):
    a = list(argv[1:])
    usage = 'usage: MT_RUN=v2 queue_front_v2.py run | MT_RUN=v2 queue_front_v2.py run --pilot <day> [--out <dir>]'
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
