"""chains_v3.py - SIX COMPLETE CHAINS v3 (P-2026-005 MAKER-TRACE; POSTING-OPPORTUNITY-AND-CHAIN v1, section 4; session aea5bfb9 builder,
2026-11-23). Built EXACTLY to predictfun/reviews/maker_trace_20261122/posting_chain_contract.md section 4 (frozen 8e09f3391) = review
3f640ad9b's next priority item 4 (reviews/automatic/20261123_take_queue_review/report.md). USER goal: the top makers' ACTION CHAINS and their
logic. SAVED OUTPUTS ONLY: no market, book, lifecycle, queue or FIFO is recomputed; every number is a saved keyed row or a sum over saved keyed
rows reconciled to the saved aggregate. DESCRIPTIVE on consumed days (Oct 24-27; these six chains lie on Oct 24): no intent, no identified
private action, no model skill, no profitability claim; filled orders are outcome-selected (unfilled orders are invisible).
CASES  the same 6 predetermined episodes as v2/chains/correction_v1/paths.md: per rank, the first two episodes in order_estimator/cases.csv
       row order of first appearance (chain_correct_post.select_paths, re-implemented); the paths.md selection is parsed and must be equal.
LINKS  per chain in order (contract 4 items 1-6; item 4 in two rows, 4a ending and 4b repost):
  1  pre_state  the section-2 features at the seconds of the post window, from the sibling v2/chains/posting_opportunity/rows.parquet
                (bound by its receipt); absent -> PENDING (draft) / refusal (run). Second k = (ws + 1000k, ws + 1000(k + 1)]; a post at x lies
                in k(x) = ceil((x - ws) / 1000) - 1; the row of second k is the state at t = ws + 1000k (pre-anchor observations).
  2  post       the union window [post_lo_union, post_hi] with its correction_v1 grade (post_regrade_orders.csv) and per-scenario lower ends;
                side and tick (observed in the fills); SIZE BOUNDS: the filled size is a LOWER bound and the submitted size is never set to
                it (unknown; EX / PL / EC feasible per scenario with size_lo / size_hi from the order_ledger size_end rows; prior picks are
                prior-driven estimates); same-hash taker legs (a crossing at arrival) counted.
  3  queue      the order's fills grouped BY HASH (repeated same-hash fills = ONE order: its first fill and later fills of the same hash);
                the S_FIFO queue class and ahead bounds (post_fill orders.csv; order_estimator contract 4.4: S_NOPRIO has no queue); the
                fresh-level lives compatible with the first fill (hi > t_open_ms and x_lo_hard <= the level's last instant) and the
                hash's candidate rows from the sibling v2/chains/queue_front_v2 levels / candidates when present, else v1
                v2/chains/queue_front levels.csv / fronts.csv marked 'v1 labels, see section 5' (with v1 labels a masked cancellation /
                replenishment is always kept as an alternative; a v1 'behind' row of a hash in the level's front set is later_same_hash).
  4a ending     EX / PL / EC per scenario (order_ledger size_end feasibility; union = post_fill union_histories), the cancel-compatible
                views (pulls.csv; must equal the PL-feasible views among S_FIFO / S_NOPRIO), the saved END logic rows, the census grade
                (correction_v1 census_graded.csv vocabulary) with its rank-day count.
  4b repost     the successor (post_fill next_kind / next_link_status / next_hash) and every saved child / parent link hypothesis
                (order_ledger link rows): 'unrelated' stays feasible, so never an identified repost.
  5  takes      every take record of the address in the market with settle_ms in the episode window [t0, t1] (take_logic_v2
                events.parquet): its direction at the anchor per clock (dir_D0 primary; dir_D1000, dir_pend), variant classes and O4
                status; the selected take case of cases.csv marked.
  6  value      the episode's lot legs (econ fills_valued: status ok, not a self-trade, settle_ms in [t0, t1]: order_estimator.episode_econ)
                and the case order's own maker rows: 10 s / 60 s / END views (valid = st_h ok and a finite v_h) with valid / all rows and
                shares, gross, net (- fee_fill of the valid rows), c/sh on valid shares, the common cohort (valid at all three) and the
                resolution view beside; completed turnover = closes.parquet closes with an episode leg as entry or close (inside / closed
                later / closing earlier inventory); residual = rq valued to END and to resolution; the address-market closes accounting
                (closes_markets.parquet) and the matched-gap interval accounting (matched_gap_r1 per_market, carry primary, zero beside).
READING per link: alternatives = the retained histories that change the explanation; ONE-HISTORY (1), TIED (>= 2), NO-EVENT (no take record),
       PENDING (sibling output absent). A TIED link is never rendered as identified; no link asserts an identified private action
       (posting time, submitted size, queue position, cancel, repost, intent). Per chain: the links whose retained histories agree, the
       hidden action that changes the explanation at each TIED link, and the first branching link.
CHECKS recorded predicates (claims.json; a run refuses on any failure, a draft records it): every input bound to its producing receipt
       (sha256); the selection equals paths.md; episode legs = episode_summary ep_*; own maker rows = post_fill orders.csv; closes per lot
       leg = mq; the carry row = episode_summary mg_*; census counts = the saved keyed rows; pulls views = PL-feasible scenarios; ledger
       union = post_fill union. The RENDERING VERIFIER refuses in both modes before any write: a TIED link rendered otherwise, same-hash
       fills split into several orders, a submitted size set, an IDENTIFIED token.
OUTPUTS write-once (a partial directory renamed at the end): run -> v2/chains/chains_v3/ (requires both sibling receipts with their files
       bound), run --draft -> v2/chains/chains_v3_draft/ (sibling links PENDING / v1 labels): chains.md, chain_links.csv (one row per link),
       claims.json (computed predicates only), receipt.json (sha256 of sources, inputs, bindings and outputs; CPU by stage; VmHWM).
GUARDS MT_RUN=v2; MT_CH3_MUT (the selftest's planted errors) refused outside the selftest; the output refused if it exists (before any
       input); run refuses unless this tool, its selftest and the contract are tracked and unmodified (the freeze); VmHWM self-guard
       pf_mt_common.GUARD_MIB (never raised); CPU self-guard CPU_CAP_S.
Usage (run_light, MT_RUN=v2): chains_v3.py run --draft | chains_v3.py run        selftest: chains_v3_selftest.py"""
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

sys.dont_write_bytecode = True                                   # no __pycache__ side effect in the receipt tree

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pf_mt_common as C  # noqa: E402

need, OK = C.need, C.STATUS_OK
NAN, INF = float('nan'), float('inf')
VERSION = 'chains_v3'
CONTRACT_FILE = C.RECEIPT_V1 / 'posting_chain_contract.md'
CONTRACT_COMMIT = '8e09f3391'
SELFTEST_FILE = HERE / 'chains_v3_selftest.py'
TOOL_FILES = (Path(__file__).resolve(), SELFTEST_FILE, CONTRACT_FILE)
DEP_FILES = (RESEARCH / 'pf_mt_common.py',)

# ------------------------------------------------------------------ declared constants (the contract's; none fitted) ------------------------------------------------------------------
SCEN = ('S_FIFO', 'S_FIFO_LO', 'S_NOPRIO')
PULL_VIEWS = ('S_FIFO', 'S_NOPRIO')                              # pulls.csv views (chain_pull)
HYP = ('EX', 'PL', 'EC')                                         # exhausted / pulled / end-censored (order_ledger size_end)
LATE_MS, EARLY_MS = 1_000, 10_000                                # the estimator's width classes
TOL_SH = 0.01
RES_VALID = ('ok', 'tie')                                        # pf_mt_econ.RES_VALID
HORIZONS = ('10s', '60s', 'end')                                 # the contract's 10 s / 60 s / END views
CLOCKS = ('D0', 'D1000', 'pend')                                 # take_logic_v2 clocks (D0 primary)
DIR_DETERMINATE = ('opposing', 'same')
FEATURES = ('spread', 'depth_touch_own', 'depth_touch_opp', 'depth_2_own', 'btc_5s_aligned', 'fair_gap', 'flow_5s', 'time_left')
PO_TIME_KEYS = ('t_ms', 'k', 't')                                # posting_opportunity second key, first present wins
PO_STATUS_COLS = ('status_D0', 'status', 'book_status', 'row_status', 'state_status')
CREATION_TYPES = ('improve', 're_establish', 'deep', 'at_empty_touch')
PO_OPTIONAL = ('E_any', 'N_any', 'improve_population', 'n_straddle', 'n_gap', 'n_unknown_cell') + tuple(f'keys_{t}' for t in ('any',) + CREATION_TYPES)
LINKS = (('1', 'pre_state'), ('2', 'post'), ('3', 'queue'), ('4a', 'ending'), ('4b', 'repost'), ('5', 'takes'), ('6', 'value'))
TITLE = dict(pre_state='PRE-STATE', post='POST', queue='QUEUE', ending='ENDING', repost='REPOST', takes='TAKES', value='VALUE')
READINGS = ('ONE-HISTORY', 'TIED', 'NO-EVENT', 'PENDING')
G0, G1, G2, G3, G4 = ('G0_unbounded_or_invalid', 'G1_conflict', 'G2_bound_constrained', 'G3_constrained_status_flagged', 'G4_scenario_only')
GRADES = (('observed_state',) + tuple(f'post_window:{g}' for g in (G0, G1, G2, G3, G4))
          + ('scenario_queue_history', 'cancel_compatible_ending', 'scenario_dependent_ending', 'unresolved_alternatives',
             'scenario_dependent_link', 'observed_execution', 'recorded_fill_accounting', 'pending'))
TIME_ALTS = {(G2, 'le_1s'): ('posted_le_1s_before_first_fill',), (G2, 'le_10s'): ('posted_le_10s_before_first_fill',),
             (G2, 'gt_10s'): ('early_post_rest_gt_10s', 'late_post_rest_le_10s'), G3: ('early_post', 'late_post'),
             G4: ('late_post_in_scenario_window', 'earlier_post_masked_by_replenishment'),
             G1: ('posting_time_unresolved_early', 'posting_time_unresolved_late'), G0: ('unbounded_early', 'unbounded_late')}
WINDOW_BASIS = {G0: 'none: window unbounded or order invalid', G1: 'void: conflict flags',
                G2: 'lifecycle scenario-free bound (conditional on H1 / H2 / capacity and the maximum match lag)',
                G3: 'lifecycle scenario-free bound with a flagged status (not certified)', G4: 'scenario union estimate only'}
QPOS = dict(front_at_acceptance=('front_at_acceptance',), front_reached_after_clearing=('front_reached_after_clearing',),
            front_at_acceptance_or_reached=('front_at_acceptance', 'front_reached_after_clearing'),
            behind_at_first_fill=('behind_at_first_fill',), unidentified=('front_at_first_fill', 'behind_at_first_fill'),
            conflicting=('front_at_first_fill', 'behind_at_first_fill'))
LINK_SUBSET = dict(tied='tied', sequential_identified='sequential_under_scenarios', overlap_identified='overlap_certain', clock_refused='clock_refused')
MUTANTS = ('tied_as_identified', 'split_same_hash', 'submitted_from_filled')   # planted errors: the selftest only
CPU_CAP_S = 8.
REL_TOL, ABS_TOL = 1e-9, 1e-6

# ---- inputs: key -> (root R = C.RECEIPT / D = C.OUT, path, binding (receipt, section, key or key suffix)) ----
REC_POST = 'chains/correction_v1/receipt_post.json'
REC_TL2 = 'chains/take_logic_v2/receipt.json'
INPUTS = dict(
    cases=('R', 'order_estimator/cases.csv', (REC_POST, 'inputs', 'cases')),
    episode_summary=('R', 'order_estimator/episode_summary.csv', (REC_POST, 'inputs', 'episode_summary')),
    order_ledger=('R', 'order_estimator/order_ledger.csv', (REC_POST, 'inputs', 'order_ledger')),
    episodes=('R', 'order_estimator/episodes.csv', (REC_POST, 'inputs', 'episodes')),
    paths_md=('R', 'chains/correction_v1/paths.md', (REC_POST, 'outputs', 'paths.md')),
    post_regrade_orders=('R', 'chains/correction_v1/post_regrade_orders.csv', (REC_POST, 'outputs', 'post_regrade_orders.csv')),
    episodes_post_regraded=('R', 'chains/correction_v1/episodes_post_regraded.csv', (REC_POST, 'outputs', 'episodes_post_regraded.csv')),
    census_graded=('R', 'chains/correction_v1/census_graded.csv', (REC_POST, 'outputs', 'census_graded.csv')),
    post_orders=('R', 'chains/post_fill/orders.csv', ('chains/post_fill/receipt.json', 'outputs', 'orders.csv')),
    pulls=('R', 'chains/pull/pulls.csv', ('chains/pull/receipt.json', 'outputs', 'pulls.csv')),
    events=('R', 'chains/take_logic_v2/events.parquet', (REC_TL2, 'outputs', 'events.parquet')),
    qf1_levels=('R', 'chains/queue_front/levels.csv', ('chains/queue_front/receipt.json', 'outputs', 'levels.csv')),
    qf1_fronts=('R', 'chains/queue_front/fronts.csv', ('chains/queue_front/receipt.json', 'outputs', 'fronts.csv')),
    closes=('R', 'closes/closes.parquet', ('closes/closes_receipt.json', 'outputs', 'closes.parquet')),
    closes_markets=('R', 'closes/closes_markets.parquet', ('closes/closes_receipt.json', 'outputs', 'closes_markets.parquet')),
    matched_gap=('R', 'matched_gap_r1/per_market.csv', ('matched_gap_r1/receipt.json', 'outputs', 'per_market.csv')),
    fills_valued=('D', 'econ/fills_valued.parquet', (REC_TL2, 'inputs', 'econ/fills_valued.parquet')))
LIFE_BIND = (REC_TL2, 'inputs')                                  # key suffix life/<day>_<mid>_fills.parquet
NEED = dict(
    cases=('kind', 'day', 'mid', 'addr', 'key', 'rank', 'case_key', 'episode_id', 'y_side', 'tick', 'n_fills', 'cum', 'size_status', 'union',
           'ub', 'P_cmd', 'P_drop', 'P_lot', 'P_clip', 'prior_sensitive', 'front_class', 'ahead_lo_fifo', 'ahead_hi_fifo',
           'ahead_at_acceptance_lo', 'ahead_at_acceptance_hi', 'status', 's', 'q', 'p_avg', 'settle_ms', 'pos_before', 'pos_after', 'cls',
           'cls_simultaneous', 'cls_certain', 'timing_sensitive', 'burst_cert', 'burst_pos'),
    episode_summary=('episode_id', 'day', 'mid', 'addr', 'rank', 't0', 't1', 'n_cases', 'case_keys', 'ep_n_legs', 'ep_n_maker', 'ep_n_taker',
                     'ep_sh', 'ep_v_end', 'ep_n_v_end_na', 'ep_fee', 'ep_net_end', 'ep_net_res', 'ep_mq', 'ep_rq', 'ep_pos_in', 'ep_pos_out',
                     'ep_outcome', 'ep_n_excluded', 'mg_status', 'mg_close_net', 'mg_resid', 'mg_total', 'mg_close_net_c_sh'),
    order_ledger=('case_key', 'row_kind', 'hypothesis', 'role', 'other_hash', 'kind', 'd_ticks', 'status', 'gap_lo_ms', 'gap_hi_ms',
                  'other_first_hi_ms', 'other_cum', 'feasible_union', 'feasible_S_FIFO', 'feasible_S_FIFO_LO', 'feasible_S_NOPRIO', 'size_lo',
                  'size_hi', 'window_lo_ms', 'window_hi_ms', 'remaining_lo', 'remaining_hi'),
    episodes=('episode_id', 'case_key', 'link', 'mechanism', 'identification_status', 'conditional_status', 'evidence_grade', 'timing_sensitive'),
    post_regrade_orders=('day', 'mid', 'rank', 'addr', 'mk_hash', 'y_side', 'tick', 'n_fills', 'cum_shares', 'first_hi_ms', 'post_lo_union_ms',
                         'post_lo_S_FIFO_ms', 'post_lo_S_FIFO_LO_ms', 'post_lo_S_NOPRIO_ms', 'post_hi_ms', 'window_width_ms', 'width_bin',
                         'life_post_class', 'status_packet', 'g1_flags_matched', 'grade', 'window_basis'),
    episodes_post_regraded=('episode_id', 'case_key', 'link', 'mechanism', 'identification_status', 'identification_status_corrected',
                            'evidence_grade', 'evidence_grade_corrected'),
    census_graded=('source', 'action_type', 'scope', 'day', 'subset', 'n', 'n_action_total', 'evidence_grade'),
    post_orders=('day', 'mid', 'rank', 'addr', 'mk_hash', 'y_side', 'tick', 'n_fills', 'cum_shares', 'first_hi_ms', 'last_hi_ms', 'ws_ms',
                 'we_ms', 'post_lo_union_ms', 'post_hi_ms', 'queue_class', 'ahead_acc_min', 'ahead_acc_max', 'ahead_lo_fifo', 'ahead_hi_fifo',
                 'lo_clock_queue_differs', 'touch_first_ms', 'touch_at_start', 'nomask_class', 'nomask_status', 'n_later_fills', 'n_tied_first',
                 'n_front_after_partial', 'size_status', 'union_histories', 'next_kind', 'next_link_status', 'n_children', 'next_hash',
                 'next_gap_lo_ms', 'next_gap_hi_ms', 'next_d_ticks', 'mq_sh', 'rq_sh', 'n_own_rows', 'pnl_rows', 'pnl_shares', 'net_res',
                 'fee_fill', 'pnl_excluded_rows', 'v_1s', 'v_10s', 'v_60s', 'status'),
    pulls=('day', 'mid', 'scope', 'addr', 'mk_hash', 'view', 'removal_set', 'removal_lo_ms', 'removal_hi_ms', 'removal_method', 'removal_class',
           'x_size_ub', 'end_histories', 'identification', 'pull_identified', 'window_status', 'width_ms', 'd_sensitive', 'most_supported',
           'support_grade', 'status'),
    qf1_levels=('day', 'mid', 'level_id', 'side', 'tick', 't_open_ms', 't_create_ms', 'unit', 'status', 'reason', 'end_type', 'end_ms',
                'size_created', 'n_members', 'holder', 'holder_group', 'front_hash', 'n_behind', 'n_unattributed', 'front_timing', 'mechanism'),
    qf1_fronts=('day', 'mid', 'level_id', 'role', 'k', 'sid', 'mk_idx', 'mk_hash', 'addr', 'q', 'hi_ms', 'lo_ms'),
    matched_gap=('day', 'm', 'party', 'view', 'primary', 'status', 'n_rows_in', 'n_excl_in', 'n_closes', 'close_sh', 'close_net', 'carry', 'resid',
                 'fee_unclosed', 'self_gross', 'total', 'close_net_c_sh', 'exposure', 'reading'))
EV_READ = ('day', 'mid', 'rank', 'addr', 'sid', 's', 'dq', 'p_avg', 'x_lo_hard_ms', 'hi_ms', 'settle_ms', 'status', 'take_class', 'cls_rec', 'cls_sim',
           'cls_cert_0', 'pop_status', 't_anchor_ms', 'o4_status', 'q_D0', 'dir_D0', 'q_D1000', 'dir_D1000', 'q_pend', 'dir_pend')
FV_READ = ('day', 'mid', 'sid', 'mk_idx', 'px_rank', 'role', 'addr', 'hash', 's', 'q', 'p', 'self_trade', 'settle_ms', 'V', 'st_res', 'gross_res',
           'st_1s', 'v_1s', 'st_10s', 'v_10s', 'st_60s', 'v_60s', 'm_end', 'st_end', 'v_end', 'fee_fill', 'net_res', 'mq', 'rq', 'status')
CL_READ = ('day', 'mid', 'addr', 'entry_sid', 'entry_mk_idx', 'entry_hash', 'entry_role', 'entry_settle_ms', 'close_sid', 'close_mk_idx',
           'close_hash', 'close_role', 'close_settle_ms', 'role_pair', 'sh', 'pnl', 'net', 'hold_ms', 'dust', 'res_ok')
CM_READ = ('day', 'mid', 'addr', 'n_rows_ok', 'n_rows_excl', 'n_closes', 'n_dust', 'close_sh', 'close_pnl', 'close_fee_fill', 'resid_sh', 'resid_end',
           'resid_res', 'net_pos', 'max_abs_pos', 'res_ok', 'end_ok', 'status')
LIFE_READ = ('mid', 'sid', 'mk_hash', 'mk_idx', 'y_side', 'y_ptick', 'shares', 'hi_ms', 'lo_ms', 'x_lo_hard_ms', 'status')

# ---- the sibling outputs (posting_opportunity sections 1-3; queue_front_v2 section 5): files, required and optional columns ----
#      (interfaces read from the siblings' declared outputs: posting_opportunity.rows_frame - <feature>_bin_D0 / status_D0 / t_ms, D1000 beside;
#       queue_front_v2 OUTPUTS - levels.csv, candidates.csv keyed per level by candidate_hash / earliest_set_hashes, fills.csv roles
#       earliest_set / later_same_hash / later_other, occupancy.csv per order; optional files are used only when present AND bound)
SIBLINGS = dict(
    posting_opportunity=dict(dir='chains/posting_opportunity', files=dict(rows=('rows.parquet',)), optional={}, need=dict(rows=('mid', 'side'))),
    queue_front_v2=dict(dir='chains/queue_front_v2', files=dict(levels=('levels.csv',), candidates=('candidates.csv',)),
                        optional=dict(fills=('fills.csv',), occupancy=('occupancy.csv',)),
                        need=dict(levels=('day', 'mid', 'level_id', 'side', 'tick', 't_open_ms', 'end_ms', 'status'),
                                  candidates=('day', 'mid', 'level_id', 'candidate_hash', 'earliest_set_hashes'),
                                  fills=('day', 'mid', 'level_id', 'role', 'mk_hash'), occupancy=('day', 'mid', 'mk_hash', 'occupancy'))))
QF2_SHOW = dict(levels=('t_create_ms', 'reason', 'end_type', 'creation_context', 'mechanism', 'alt_netting_possible', 'alt_first_other_address',
                        'n_members', 'timing_basis'),
                candidates=('status', 'candidate_addr', 'candidate_group', 'timing_basis', 'alt_netting_possible', 'alt_first_other_address'),
                fills=('k', 'sid', 'mk_idx', 'in_earliest_set', 'valid'),
                occupancy=('window_lo_ms', 'post_grade', 'size_at_window_lo', 'frame_ms_at_window_lo', 'lookup_status', 'is_front_candidate_hash'))
LINK_COLS = ('chain', 'rank', 'episode_id', 'day', 'mid', 'addr', 'order_hash', 'item', 'link', 'status', 'source', 'evidence_grade', 'census',
             'reading', 'n_histories', 'alternatives', 'histories_agree', 'hidden_action', 'identified_private_action', 'observed_public',
             'submitted_size', 'size_lower_bound', 'n_records', 'n_fills', 'n_orders_rendered', 'pending_reason', 'summary')
OUT_FILES = ('chains.md', 'chain_links.csv', 'claims.json', 'receipt.json')


# ------------------------------------------------------------------ small helpers ------------------------------------------------------------------
def _mut():
    return os.environ.get('MT_CH3_MUT', '')


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def hwm():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmHWM:'):
                    return int(line.split()[1]) / 1024.
    except OSError:
        pass
    return 0.


def guard(where):
    h = hwm()
    need(h < C.GUARD_MIB, f'memory self-guard: VmHWM {h:.0f} MiB >= {C.GUARD_MIB:.0f} MiB at {where}')
    return h


def cpu_guard(where, c0=0.):
    c = time.process_time() - c0
    need(c < CPU_CAP_S, f'CPU self-guard: {c:.2f} s >= {CPU_CAP_S} s at {where}')
    return c


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(C.REPO))
    except ValueError:
        return str(p)


def _clean(o):
    """JSON-safe copy: NaN / inf -> None, numpy scalars -> Python, keys -> str, sets -> sorted lists."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return sorted(_clean(v) for v in o)
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def _json_bytes(obj):
    return (json.dumps(_clean(obj), indent=1, sort_keys=True, allow_nan=False) + '\n').encode()


def _git(files):
    """HEAD and whether the files are tracked and unmodified (the run's freeze test reads 'unmodified')."""
    try:
        g = ['git', '-C', str(C.REPO)]
        head = subprocess.run(g + ['rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30).stdout.strip() or None
        rel = [_rel(f) for f in files]
        tracked = subprocess.run(g + ['ls-files', '--error-unmatch', '--'] + rel, capture_output=True, text=True, timeout=30).returncode == 0
        st = subprocess.run(g + ['status', '--porcelain', '--'] + rel, capture_output=True, text=True, timeout=30)
        return dict(head=head, files=rel, tracked=tracked, unmodified=bool(tracked and st.returncode == 0 and not st.stdout.strip()))
    except (OSError, ValueError, subprocess.SubprocessError):
        return dict(head=None, files=None, tracked=None, unmodified=False)


def _isna(x):
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _s(x):
    """a cell -> Python str ('' when missing)."""
    if isinstance(x, str):
        return x
    if _isna(x):
        return ''
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))
    return str(x)


def _sarr(col):
    return np.array([_s(v) for v in col], object)


def _fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v if np.isfinite(v) else NAN


def _int(x, default=-1):
    v = _fin(x)
    return default if np.isnan(v) else int(round(v))


def _b(x):
    """a saved boolean cell -> True / False / None (missing); anything else refuses."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if _isna(x):
        return None
    if isinstance(x, str):
        if x in ('True', 'true'):
            return True
        if x in ('False', 'false'):
            return False
        need(x == '', f'not a boolean cell: {x!r}')
        return None
    if isinstance(x, (int, float, np.integer, np.floating)) and float(x) in (0., 1.):
        return bool(x)
    raise ValueError(f'not a boolean cell: {x!r}')


def _fmt(x, nd=2):
    v = _fin(x)
    if np.isnan(v):
        return 'na'
    if np.isinf(v):
        return 'inf'
    return f'{v:.{nd}f}' if nd else f'{int(round(v))}'


def _sg(x, nd=2):
    v = _fin(x)
    return 'na' if np.isnan(v) else f'{v:+.{nd}f}'


def _rel_s(x, ws):
    v = _fin(x)
    return 'na' if (np.isnan(v) or v < 0) else f'ws{(v - ws) / 1000.:+.3f}s'


def _h12(h):
    return _s(h)[:12]


def _pred(mask):
    m = np.asarray(list(mask) if not isinstance(mask, np.ndarray) else mask, bool)
    return dict(n=int(len(m)), n_true=int(m.sum()), holds=bool(m.all()))


def _close(a, b, tol=REL_TOL, atol=ABS_TOL):
    a, b = _fin(a), _fin(b)
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    return abs(a - b) <= atol + tol * max(1., abs(a), abs(b))


def sec_of(x, ws):
    """the market second k whose interval (ws + 1000k, ws + 1000(k + 1)] holds the instant x."""
    d = int(x) - int(ws)
    return -((-d) // 1000) - 1


# ------------------------------------------------------------------ readers ------------------------------------------------------------------
def _undict(t):
    return pa.table({c: (t.column(c).cast(t.column(c).type.value_type) if pa.types.is_dictionary(t.column(c).type) else t.column(c))
                     for c in t.column_names})


def read_csv(p, key, cols):
    p = Path(p)
    need(p.is_file(), f'input {key} is missing: {_rel(p)} (refused)')
    have = list(pd.read_csv(p, nrows=0).columns)
    miss = [c for c in cols if c not in have]
    need(not miss, f'input {key}: required columns missing {miss} (refused)')
    dt = {c: str for c in ('day',) if c in cols}
    return pd.read_csv(p, usecols=list(cols), dtype=dt, float_precision='round_trip', low_memory=False)


def read_pq(p, key, cols):
    p = Path(p)
    need(p.is_file(), f'input {key} is missing: {_rel(p)} (refused)')
    names = pq.ParquetFile(p).schema_arrow.names
    miss = [c for c in cols if c not in names]
    need(not miss, f'input {key}: required columns missing {miss} (refused)')
    return _undict(pq.read_table(p, columns=list(cols))).to_pandas()


def _stat_val(v):
    return v.decode() if isinstance(v, bytes) else v


def read_groups(p, key, cols, mids, addrs_l, cnt, day=None):
    """a large parquet through its row-group statistics of mid (and day): only the chain markets' groups are read, then the rows of the
    chain markets and addresses are kept."""
    p = Path(p)
    need(p.is_file(), f'input {key} is missing: {_rel(p)} (refused)')
    pf = pq.ParquetFile(p)
    md = pf.metadata
    names = pf.schema_arrow.names
    miss = [c for c in cols if c not in names]
    need(not miss, f'input {key}: required columns missing {miss} (refused)')
    paths = [md.row_group(0).column(k).path_in_schema for k in range(md.num_columns)] if md.num_row_groups else []
    jm, jd = (paths.index('mid') if 'mid' in paths else None), (paths.index('day') if 'day' in paths else None)
    vm, va = pa.array(sorted(mids), pa.int64()), pa.array(sorted(addrs_l), pa.string())
    parts = []
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        sm = rg.column(jm).statistics if jm is not None else None
        if sm is not None and sm.has_min_max and not any(int(sm.min) <= m <= int(sm.max) for m in mids):
            cnt[f'{key}_groups_pruned'] += 1
            continue
        sd = rg.column(jd).statistics if (jd is not None and day is not None) else None
        if sd is not None and sd.has_min_max and not (str(_stat_val(sd.min)) <= day <= str(_stat_val(sd.max))):
            cnt[f'{key}_groups_pruned'] += 1
            continue
        t = _undict(pf.read_row_group(i, columns=list(cols)))
        cnt[f'{key}_groups_read'] += 1
        cnt[f'{key}_rows_scanned'] += t.num_rows
        m = pc.and_(pc.is_in(pc.cast(t.column('mid'), pa.int64()), value_set=vm),
                    pc.is_in(pc.utf8_lower(pc.cast(t.column('addr'), pa.string())), value_set=va))
        if day is not None:
            m = pc.and_(m, pc.equal(pc.cast(t.column('day'), pa.string()), day))
        m = pc.fill_null(m, False)
        if pc.any(m).as_py():
            parts.append(t.filter(m))
    df = pa.concat_tables(parts).to_pandas() if parts else pd.DataFrame({c: [] for c in cols})
    df['addr_l'] = [str(a).lower() for a in df['addr']]
    cnt[f'{key}_rows_kept'] += int(len(df))
    return df


def bound_sha(rec, section, key):
    """the sha256 a producing receipt records for a file: rec[section][key] (a string or a dict with 'sha256'); a key not present
    matches one key ending with '/' + key."""
    sec = rec.get(section) if isinstance(rec, dict) else None
    if not isinstance(sec, dict):
        return None
    v = sec.get(key)
    if v is None:
        hits = [k for k in sec if isinstance(k, str) and k.endswith('/' + key)]
        if len(hits) != 1:
            return None
        v = sec[hits[0]]
    if isinstance(v, dict):
        v = v.get('sha256')
    return v.lower() if isinstance(v, str) and re.fullmatch(r'[0-9a-fA-F]{64}', v) else None


class Inputs:
    """every file read: path, sha256, bytes, rows, and the sha256 its producing receipt records (binding)."""

    def __init__(self, rroot):
        self.rroot, self.info, self._rec = Path(rroot), {}, {}

    def receipt(self, rel):
        if rel not in self._rec:
            p = self.rroot / rel
            try:
                self._rec[rel] = (json.loads(p.read_text()), _sha(p), _rel(p))
            except (OSError, ValueError):
                self._rec[rel] = (None, None, _rel(p))
        return self._rec[rel]

    def add(self, key, p, rows, binding):
        p = Path(p)
        sha = _sha(p)
        rec_rel, section, bkey = binding
        rec, rsha, rpath = self.receipt(rec_rel)
        want = bound_sha(rec, section, bkey) if rec is not None else None
        st = 'receipt_missing' if rec is None else ('not_listed' if want is None else ('ok' if want == sha else 'mismatch'))
        self.info[key] = dict(path=_rel(p), sha256=sha, bytes=p.stat().st_size, rows=int(rows),
                              binding=dict(receipt=rpath, receipt_sha256=rsha, section=section, key=bkey, recorded=want, status=st))
        return st


def sibling_state(rroot, name):
    """a sibling output directory -> status ok (its receipt lists every file read with the same sha256) / absent / incomplete /
    unbound / mismatch / unreadable, with the reason."""
    sp = SIBLINGS[name]
    d = Path(rroot) / sp['dir']
    st = dict(name=name, dir=_rel(d), status='absent', reason='', receipt=None, files={})
    rp = d / 'receipt.json'
    if not rp.is_file():
        st['reason'] = f'{name} receipt absent ({_rel(rp)})'
        return st
    try:
        rec = json.loads(rp.read_text())
    except (OSError, ValueError) as e:
        st.update(status='unreadable', reason=f'{name} receipt unreadable: {type(e).__name__}')
        return st
    st['receipt'] = dict(path=_rel(rp), sha256=_sha(rp))
    st['absent_optional'], st['notes'] = [], []
    opt = sp.get('optional', {})
    for k, names in list(sp['files'].items()) + list(opt.items()):
        p = next((d / n for n in names if (d / n).is_file()), None)
        if p is None and k in opt:
            st['absent_optional'].append(k)
            continue
        if p is None:
            st.update(status='incomplete', reason=f'{name}: none of {list(names)} present')
            return st
        want, got = bound_sha(rec, 'outputs', p.name), _sha(p)
        if want is None:
            st.update(status='unbound', reason=f'{name} receipt does not list {p.name} with a sha256')
            return st
        if want != got:
            st.update(status='mismatch', reason=f'{name}/{p.name} sha256 {got[:12]} differs from its receipt {want[:12]}')
            return st
        st['files'][k] = dict(path=str(p), sha256=got, bytes=p.stat().st_size)
    st['status'] = 'ok'
    return st


def _read_any(p, cols=None):
    p = Path(p)
    if p.suffix == '.parquet':
        t = pq.read_table(p, columns=list(cols) if cols else None)
        return _undict(t).to_pandas()
    have = list(pd.read_csv(p, nrows=0).columns)
    use = [c for c in cols if c in have] if cols else None
    return pd.read_csv(p, usecols=use, dtype={'day': str} if 'day' in have else None, float_precision='round_trip', low_memory=False)


def po_columns(have):
    """posting_opportunity: each section-2 feature's D0 column (primary: <f>_bin_D0, <f>_D0, else unsuffixed) and D1000 column (the whole state
    lagged one second, a sensitivity), the row-status column and the second key; None when absent."""
    d0 = {f: next((c for c in (f'{f}_bin_D0', f'{f}_D0', f'{f}_bin', f) if c in have), None) for f in FEATURES}
    d1 = {f: next((c for c in (f'{f}_bin_D1000', f'{f}_D1000') if c in have), None) for f in FEATURES}
    return d0, d1, next((c for c in PO_STATUS_COLS if c in have), None), next((c for c in PO_TIME_KEYS if c in have), None)


def load_sibling(st, mids):
    """read an ok sibling's files (chain markets only) and check its columns: a required file's mismatch becomes the sibling's status
    (schema_mismatch); an optional file with missing columns is not used (noted)."""
    if st['status'] != 'ok':
        return None
    sp = SIBLINGS[st['name']]
    out = {}
    for k, f in list(st['files'].items()):
        p = Path(f['path'])
        have = (pq.ParquetFile(p).schema_arrow.names if p.suffix == '.parquet' else list(pd.read_csv(p, nrows=0).columns))
        miss = [c for c in sp['need'][k] if c not in have]
        if st['name'] == 'posting_opportunity':
            d0, d1, stc, tk = po_columns(have)
            miss += ([] if tk else [f'one of {list(PO_TIME_KEYS)}']) + [f_ for f_ in FEATURES if d0[f_] is None]
            use = list(dict.fromkeys(list(sp['need'][k]) + [c for c in [tk, stc] + list(d0.values()) + list(d1.values()) if c]
                                     + ['day'] * ('day' in have) + [c for c in PO_OPTIONAL if c in have]))
            out['cols'] = dict(d0=d0, d1=d1, status=stc, time=tk)
        else:
            use = list(dict.fromkeys(list(sp['need'][k]) + [c for c in QF2_SHOW[k] if c in have]))
        if miss and k in sp.get('optional', {}):
            st['notes'].append(f'{p.name}: required columns missing {miss} (not used)')
            del st['files'][k]
            continue
        if miss:
            st.update(status='schema_mismatch', reason=f"{st['name']}/{p.name}: required columns missing {miss}")
            return None
        df = _read_any(p, use)
        df = df[np.isin(np.array([_int(x) for x in df['mid']], np.int64), np.array(sorted(mids), np.int64))].reset_index(drop=True)
        out[k] = df
        f['rows_used'] = int(len(df))
    return out


# ------------------------------------------------------------------ selection (contract 4: no new selection) ------------------------------------------------------------------
PATH_RE = re.compile(r'^## Path (\d+) - rank (\d+): episode (\S+)\s*$')


def select_chains(CS):
    """per rank, the first two episodes in cases.csv row order of first appearance (chain_correct_post.select_paths)."""
    sel = []
    for r in (1, 2, 3):
        seen = []
        for e, rk in zip(CS['episode_id'], CS['rank']):
            if _int(rk) == r and _s(e) not in seen:
                seen.append(_s(e))
                if len(seen) == 2:
                    break
        sel += [(r, e) for e in seen]
    return sel


def paths_md_selection(text):
    return [(int(m.group(2)), m.group(3)) for m in (PATH_RE.match(x) for x in text.splitlines()) if m]


def chain_bases(sel, CS, ES):
    cs = CS.to_dict('records')
    es = {_s(r['episode_id']): r for r in ES.to_dict('records')}
    out = []
    for k, (r, eid) in enumerate(sel, 1):
        need(eid in es, f'episode {eid} is not in episode_summary.csv (refused)')
        e = es[eid]
        rows = [(i, c) for i, c in enumerate(cs) if _s(c['episode_id']) == eid]
        od = [c for _, c in rows if _s(c['kind']) == 'order']
        need(len(od) == 1, f'episode {eid}: {len(od)} order cases (exactly one is declared per chain; refused)')
        o = od[0]
        day, mid, addr = _s(e['day']), _int(e['mid']), _s(e['addr']).lower()
        need(_int(o['rank']) == r and _s(o['addr']).lower() == addr and _s(o['day']) == day and _int(o['mid']) == mid,
             f'episode {eid}: its order case differs from episode_summary in rank / address / day / market (refused)')
        out.append(dict(chain=k, rank=int(r), episode_id=eid, day=day, mid=mid, addr=addr, t0=_int(e['t0']), t1=_int(e['t1']),
                        case_row=rows[0][0] + 1, order_case=o, order_hash=_s(o['key']).lower(),
                        take_cases=[c for _, c in rows if _s(c['kind']) == 'take'], es=e))
    return out


# ------------------------------------------------------------------ link builders ------------------------------------------------------------------
def new_link(ch, item, link, **kw):
    L = dict(chain=ch['chain'], rank=ch['rank'], episode_id=ch['episode_id'], day=ch['day'], mid=ch['mid'], addr=ch['addr'],
             order_hash=ch['order_hash'], item=item, link=link, status='filled', source='', evidence_grade='', census='', reading='',
             n_histories=0, alternatives=[], histories_agree=False, hidden_action='', identified_private_action=False, observed_public=False,
             submitted_size=NAN, size_lower_bound=NAN, n_records=0, n_fills=0, n_orders_rendered=0, pending_reason='', summary='', lines=[],
             checks={})
    L.update(kw)
    return L


def pending_link(ch, item, link, source, reason):
    return new_link(ch, item, link, status='pending', source=source, evidence_grade='pending', pending_reason=reason,
                    summary=f'PENDING: {reason}', hidden_action='not evaluated (pending)')


def census_rows(CEN, rank, day, types, source='chains/take/actions.csv'):
    m = ((CEN['scope'].to_numpy(object) == f'rank{rank}') & (CEN['day'].to_numpy(object) == day)
         & np.isin(CEN['action_type'].to_numpy(object), list(types)) & (CEN['source'].to_numpy(object) == source))
    return CEN[m]


def link_pre_state(ch, p, t, PO, sib):
    """contract 4 item 1 (sections 1-2 features at the post window seconds)."""
    src = 'chains/posting_opportunity/rows.parquet'
    if PO is None:
        return pending_link(ch, '1', 'pre_state', src, sib['reason'] or f"posting_opportunity status {sib['status']}")
    R, cols = PO['rows'], PO['cols']
    ws, we = _int(p['ws_ms']), _int(p['we_ms'])
    n_sec = max(0, (we - ws) // 1000)
    lo, hi = _int(t['post_lo_union_ms']), _int(t['post_hi_ms'])
    k_hi = sec_of(hi, ws)
    k_lo = sec_of(lo, ws) if lo >= 0 else 0
    tk = cols['time']
    tv = np.array([_fin(x) for x in R[tk]], float)
    ms_key = tk == 't_ms' or (tk == 't' and np.nanmax(np.r_[tv, 0.]) > 1e11)   # absolute venue ms, else a second index from ws
    kk = (tv - ws) / 1000. if ms_key else tv
    aligned = np.isfinite(kk) & (np.abs(kk - np.round(kk)) < 1e-9)
    kk = np.where(aligned, np.round(kk), -10 ** 9).astype(np.int64)
    side = _sarr(R['side'])
    lo_c, hi_c = max(k_lo, 0), min(k_hi, n_sec - 1)
    sel = (np.array([_int(x) for x in R['mid']], np.int64) == ch['mid']) & (side == _s(p['y_side'])) & (kk >= lo_c) & (kk <= hi_c)
    if 'day' in R.columns:
        sel &= _sarr(R['day']) == ch['day']
    W = R[sel]
    n_out = max(0, min(k_hi, -1) - k_lo + 1) + max(0, k_hi - max(k_lo, n_sec) + 1)
    fcol = cols['d0']
    vals = {f: Counter(_sarr(W[fcol[f]])) for f in FEATURES}
    scols = [cols['status']] if cols['status'] else []
    stv = Counter(x for c in scols for x in _sarr(W[c]) if x not in ('', OK))
    recs = W.to_dict('records')
    vecs = sorted(set('; '.join(f'{f}={_s(r[fcol[f]])}' for f in FEATURES) for r in recs))
    d1 = cols['d1']
    vecs1 = (sorted(set('; '.join(f'{f}={_s(r[d1[f]])}' for f in FEATURES) for r in recs)) if all(d1[f] for f in FEATURES) else None)
    cell = []                                                     # visible fresh creations AT the order's cell inside the window (section 1 keys)
    if 'keys_any' in W.columns:
        for r in recs:
            for key in (x for x in _s(r['keys_any']).split(';') if x):
                parts = key.split('|')
                if len(parts) >= 3 and _int(parts[0]) == ch['mid'] and _int(parts[2]) == _int(p['tick']):
                    tys = [ty for ty in CREATION_TYPES if f'keys_{ty}' in W.columns and key in _s(r[f'keys_{ty}']).split(';')]
                    cell.append(f"{key} ({'+'.join(tys) or 'type unknown'})")
    n_cre = int(pd.to_numeric(W['N_any'], errors='coerce').fillna(0).sum()) if 'N_any' in W.columns else None
    n_expect = max(0, hi_c - lo_c + 1)
    alts = list(vecs) + [f'status:{k}' for k in sorted(stv)]
    missing = n_expect - int(len(W))
    if missing > 0 or n_out > 0:
        alts.append(f'state_not_saved_for_{missing + n_out}_seconds')
    if not alts:
        alts = ['state_not_saved_for_the_window', 'any_state']
    change = [f for f in FEATURES if len(vals[f]) > 1]
    hidden = '' if len(alts) == 1 else (
        f"the posting second inside k = {k_lo}..{k_hi} (ws{k_lo:+d}..ws{k_hi + 1:+d} s): "
        + (f"features {', '.join(change)} change across it" if change else 'the state is not saved (or carries a status) for some seconds'))
    lines = [f"  - window seconds k = {k_lo}..{k_hi} ({n_expect} in the market window, {n_out} outside; {len(W)} saved rows at side "
             f"{_s(p['y_side'])}; time key {tk}); post window [{_rel_s(lo, ws)}, {_rel_s(hi, ws)}]",
             f"  - features (D0 = the state at t; columns {fcol['spread']}, ...): "
             + '; '.join(f"{f} {dict(sorted(vals[f].items()))}" for f in FEATURES),
             f"  - row statuses ({cols['status'] or 'no status column'}): {dict(sorted(stv.items())) if stv else 'none'}",
             ('  - D1000 (the whole state lagged one second): ' + (f'{len(vecs1)} distinct state vector(s) vs {len(vecs)} at D0'
                                                                   + (' - the same set' if vecs1 == vecs else ' - a different set (timing-sensitive)'))
              if vecs1 is not None else '  - D1000: not saved'),
             f"  - visible fresh creations on this side in the window: {n_cre if n_cre is not None else 'not saved'}; at the order's cell "
             f"(tick {_int(p['tick'])}): {'; '.join(cell) if cell else 'none'} (an observed market creation; the order is at most a candidate "
             "creator, section 3)"]
    return new_link(ch, '1', 'pre_state', source=src, evidence_grade='observed_state', alternatives=alts, n_histories=len(alts),
                    hidden_action=hidden, observed_public=True, n_records=int(len(W)),
                    summary=f"{len(W)} saved seconds; features constant: {', '.join(f for f in FEATURES if len(vals[f]) == 1) or 'none'}",
                    lines=lines)


def size_fields(filled, hyps):
    """the post's size bounds: the filled size is a LOWER bound; the submitted size stays unknown (NaN). -> (submitted, lower, upper, text)."""
    feas = [h for h in HYP if hyps.get(h, {}).get('union')]
    ups = [hyps[h]['size_hi'] for h in feas]
    upper = max(ups) if ups and all(np.isfinite(u) for u in ups) else INF
    submitted = NAN
    if _mut() == 'submitted_from_filled':                         # planted error: the filled lower bound reported as the submitted size
        submitted = float(filled)
    parts = []
    for h in feas:
        lo_, hi_ = hyps[h]['size_lo'], hyps[h]['size_hi']
        if h == 'EX':
            parts.append(f'EX (exhausted: = {_fmt(lo_)} sh, equal to the filled size only under EX)')
        else:
            parts.append(f"{h} ({'pulled' if h == 'PL' else 'end-censored'}: [{_fmt(lo_)}, {_fmt(hi_) if np.isfinite(hi_) else 'inf'}] sh)")
    stxt = (f"submitted size {_fmt(submitted)} sh" if np.isfinite(submitted) else 'submitted size unknown')
    text = (f"  - size: filled {_fmt(filled)} sh = a LOWER bound (observed executions); {stxt}; feasible "
            + ('; '.join(parts) if parts else 'none saved') + f"; bounds [{_fmt(filled)}, {_fmt(upper) if np.isfinite(upper) else 'inf'}] sh")
    return submitted, float(filled), upper, text


def ledger_hyps(led_rows):
    out = {}
    for r in led_rows:
        if _s(r['row_kind']) != 'size_end':
            continue
        h = _s(r['hypothesis'])
        out[h] = dict(union=bool(_b(r['feasible_union'])), **{s: bool(_b(r[f'feasible_{s}'])) for s in SCEN},
                      size_lo=_fin(r['size_lo']), size_hi=_fin(r['size_hi']) if np.isfinite(_fin(r['size_hi'])) else INF,
                      window_lo=_int(r['window_lo_ms']), window_hi=_int(r['window_hi_ms']),
                      remaining_lo=_fin(r['remaining_lo']), remaining_hi=_fin(r['remaining_hi']))
    return out


def link_post(ch, p, t, led_rows, epr_rows, CEN, T, crossing):
    o = ch['order_case']
    ws = _int(p['ws_ms'])
    g, b = _s(t['grade']), _s(t['width_bin'])
    need(g in (G0, G1, G2, G3, G4), f"chain {ch['chain']}: unknown post grade {g!r} (refused)")
    talts = TIME_ALTS.get((g, b)) or TIME_ALTS.get(g)
    need(talts is not None, f"chain {ch['chain']}: no declared time alternatives for {g} / {b} (refused)")
    hyps = ledger_hyps(led_rows)
    salts = [h for h in HYP if hyps.get(h, {}).get('union')]
    if not salts:
        salts = ['EX', 'PL']                                      # no saved size_end row: both kept (conservative)
    submitted, lower, upper, stext = size_fields(_fin(p['cum_shares']), hyps)
    alts = [f'{a} + {s}' for a in talts for s in salts]
    hid = []
    if len(talts) > 1:
        hid.append(f"the posting instant inside [{_rel_s(t['post_lo_union_ms'], ws)}, {_rel_s(t['post_hi_ms'], ws)}] "
                   f"({_fmt(t['window_width_ms'], 0)} ms; {WINDOW_BASIS[g]})")
    if len(salts) > 1:
        hid.append('the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)' + (' or end-censored (EC)' if 'EC' in salts else ''))
    # census: this grade among the rank-day's first-filled orders (post_new_level + post_same_level), reconciled to post_regrade_orders
    cr = census_rows(CEN, ch['rank'], ch['day'], ('post_new_level', 'post_same_level'))
    n_g = int(pd.to_numeric(cr.loc[cr['subset'].to_numpy(object) == g, 'n']).sum())
    tot = int(sum(_int(x.iloc[0]) for _, x in cr.groupby('action_type')['n_action_total']))
    mt = (T['_rank'].to_numpy(np.int64) == ch['rank']) & (T['day'].to_numpy(object) == ch['day'])
    chk = dict(census_post_grade_count_equals_orders=bool(n_g == int((mt & (T['grade'].to_numpy(object) == g)).sum())),
               census_post_total_equals_orders=bool(tot == int(mt.sum())))
    q = [x for x in epr_rows if _s(x['link']) == 'POST' and _s(x['mechanism']) == 'queue_priority_before_demand']
    pri = '; '.join(f"{k} {_s(o[k])}" for k in ('P_cmd', 'P_drop', 'P_lot', 'P_clip'))
    lines = [f"  - order {_h12(ch['order_hash'])}: {_s(p['y_side'])} at {_int(p['tick'])}c (side and tick observed in its fills); "
             f"first fill <= {_rel_s(p['first_hi_ms'], ws)}",
             f"  - window [{_rel_s(t['post_lo_union_ms'], ws)}, {_rel_s(t['post_hi_ms'], ws)}], width {_fmt(t['window_width_ms'], 0)} ms ({b}); "
             f"per-scenario lower ends S_FIFO {_rel_s(t['post_lo_S_FIFO_ms'], ws)}, S_FIFO_LO {_rel_s(t['post_lo_S_FIFO_LO_ms'], ws)}, "
             f"S_NOPRIO {_rel_s(t['post_lo_S_NOPRIO_ms'], ws)}; lifecycle post_class {_s(t['life_post_class'])}; packet status "
             f"{_s(t['status_packet'])} -> grade {g} ({WINDOW_BASIS[g]})",
             stext,
             f"  - size status {_s(o['size_status'])}; prior picks (prior-driven estimates, not evidence): {pri}",
             f"  - same-hash taker legs (a crossing at arrival): {crossing[0]} leg(s), {_fmt(crossing[1])} sh",
             f"  - saved POST logic row: queue_priority_before_demand "
             + (f"{_s(q[0]['identification_status'])} -> {_s(q[0]['identification_status_corrected'])} (grade {_s(q[0]['evidence_grade'])} -> "
                f"{_s(q[0]['evidence_grade_corrected'])})" if q else 'none saved')]
    return new_link(ch, '2', 'post', source='correction_v1/post_regrade_orders.csv; post_fill/orders.csv; order_ledger.csv; cases.csv; census_graded.csv',
                    evidence_grade=f'post_window:{g}', census=f"rank{ch['rank']} {ch['day']} {g}: {n_g} of {tot} first-filled orders",
                    alternatives=alts, n_histories=len(alts), hidden_action='; '.join(hid), submitted_size=submitted, size_lower_bound=lower,
                    summary=f"{_s(p['y_side'])} {_int(p['tick'])}c, {g} {b}, size >= {_fmt(lower)} sh ({'|'.join(salts)})", lines=lines,
                    checks=chk)


def compatible_levels(LV, ch, side, tick, fill, we):
    """fresh-level lives at the cell compatible with a fill: t_open_ms < hi and (x_lo_hard unbounded or x_lo_hard <= the level's last
    instant: min(end_ms, we) at a 'zero' end (or a bounded end without end_type), else we)."""
    if LV is None or len(LV) == 0:
        return []
    m = ((_sarr(LV['day']) == ch['day']) & (np.array([_int(x) for x in LV['mid']]) == ch['mid']) & (_sarr(LV['side']) == side)
         & (np.array([_int(x) for x in LV['tick']]) == tick))
    out = []
    for r in LV[m].to_dict('records'):
        t_open, end_ms = _int(r['t_open_ms']), _int(r['end_ms'])
        et = _s(r.get('end_type', 'zero' if end_ms >= 0 else ''))
        t_last = min(end_ms, we) if (end_ms >= 0 and et in ('zero', '')) else we
        if t_open < fill['hi'] and (fill['xlo'] < 0 or fill['xlo'] <= t_last):
            out.append(r)
    return sorted(out, key=lambda r: (_int(r['t_open_ms']), _s(r['level_id'])))


def render_orders(fills, ws):
    """fills (one dict per fill: hash, sid, mk_idx, shares, hi, xlo, status) -> the queue block's order lines: ONE line per distinct hash
    (repeated same-hash fills are the same order), its fills below it in window order."""
    groups = {}
    for f in fills:
        groups.setdefault(f['hash'], []).append(f)
    if _mut() == 'split_same_hash':                               # planted error: every fill rendered as its own order
        groups = {}
        for i, f in enumerate(fills):
            groups[(f['hash'], i)] = [f]
    lines = []
    for key, fs in groups.items():
        h = key[0] if isinstance(key, tuple) else key
        lines.append(f"  - order {_h12(h)}: {len(fs)} fill(s), {_fmt(sum(f['shares'] for f in fs))} sh"
                     + (' (one order: its first fill and later fills of the same hash)' if len(fs) > 1 else ''))
        for i, f in enumerate(fs, 1):
            lines.append(f"    - fill {i}: {_fmt(f['shares'])} sh, hard window [{_rel_s(f['xlo'], ws)}, {_rel_s(f['hi'], ws)}], sid {f['sid'][:13]} "
                         f"mk_idx {f['mk_idx']}, status {f['status']} - " + ('first fill' if i == 1 else 'later fill of the same hash (the same order)'))
    return lines, len(groups)


def link_queue(ch, p, fills, QF1, QF2, sib):
    ws, we = _int(p['ws_ms']), _int(p['we_ms'])
    side, tick = _s(p['y_side']), _int(p['tick'])
    v2 = QF2 is not None
    LV = QF2['levels'] if v2 else QF1['levels']
    olines, n_orders = render_orders(fills, ws)
    qc = _s(p['queue_class'])
    pos = QPOS.get(qc, ('front_at_first_fill', 'behind_at_first_fill'))
    f1 = fills[0] if fills else None
    lv = compatible_levels(LV, ch, side, tick, f1, we) if f1 else []
    lvt = [f"level {_s(r['level_id'])} ({_s(r['status'])})" for r in lv] or ['no_saved_fresh_level_life']
    marker = '' if v2 else ' [v1 labels, see section 5]'
    if v2:                                                        # only an explicit False excludes the masked-netting history
        netting = any(_s(r['alt_netting_possible']).strip().lower() not in ('false', '0') for r in lv) if (lv and 'alt_netting_possible' in LV.columns) else bool(lv)
    else:
        netting = True                                            # v1 labels: a masked cancellation / replenishment is never excluded
    alts = [f'{a} @ {b_}' + (' + masked_netting_possible' if netting else '') for a in pos for b_ in lvt]
    if netting:
        alts += [f'{a} @ {b_} + as_saved' for a in pos for b_ in lvt]
    # candidate / front rows of this hash (v2 candidates, else v1 fronts.csv with the section-5.3 same-hash relabel)
    h = ch['order_hash']
    occ = []
    if v2:
        CA = QF2['candidates']
        cm = (_sarr(CA['day']) == ch['day']) & (np.array([_int(x) for x in CA['mid']]) == ch['mid'])
        crow = []
        for r in CA[cm].to_dict('records'):
            es_ = {x.strip().lower() for x in re.split(r'[|;,]', _s(r['earliest_set_hashes'])) if x.strip()}
            ch_ = _s(r['candidate_hash']).lower()
            if h == ch_ or h in es_:
                crow.append(f"front-candidate set at level {_s(r['level_id'])} ({'the candidate hash' if h == ch_ else 'an earliest-set member'}"
                            + ''.join(f"; {c} {_s(r[c])}" for c in QF2_SHOW['candidates'] if c in r) + ')')
        FLv = QF2.get('fills')
        if FLv is not None:
            fmv = (_sarr(FLv['mk_hash']) == h) & (_sarr(FLv['day']) == ch['day']) & (np.array([_int(x) for x in FLv['mid']]) == ch['mid'])
            crow += [f"{_s(r['role'])} at level {_s(r['level_id'])}" + (f" k={_int(r['k'])}" if 'k' in r else '') for r in FLv[fmv].to_dict('records')]
        OC = QF2.get('occupancy')
        if OC is not None:
            om = (_sarr(OC['mk_hash']) == h) & (_sarr(OC['day']) == ch['day']) & (np.array([_int(x) for x in OC['mid']]) == ch['mid'])
            occ = [f"endpoint occupancy {_s(r['occupancy'])}" + (f" at {_rel_s(r['window_lo_ms'], ws)}" if 'window_lo_ms' in r else '')
                   + ''.join(f"; {c} {_s(r[c])}" for c in ('size_at_window_lo', 'post_grade', 'lookup_status') if c in r)
                   + ' (an endpoint observation, never an actual-post queue label)' for r in OC[om].to_dict('records')]
    else:
        FR = QF1['fronts']
        fm = (_sarr(FR['mk_hash']) == h) & (_sarr(FR['day']) == ch['day']) & (np.array([_int(x) for x in FR['mid']]) == ch['mid'])
        crow = []
        for r in FR[fm].to_dict('records'):
            lvl = _s(r['level_id'])
            fh = set(_sarr(FR.loc[(_sarr(FR['level_id']) == lvl) & (np.array([_int(x) for x in FR['mid']]) == ch['mid'])
                                  & (_sarr(FR['role']) == 'front'), 'mk_hash']))
            role = _s(r['role'])
            lab = 'later_same_hash' if (role == 'behind' and h in fh) else ('front_candidate' if role == 'front' else 'later_other')
            crow.append(f"v1 {role} k={_int(r['k'])} at level {lvl} -> {lab}")
    hid = []
    if len(pos) > 1:
        hid.append(f"whether orders were ahead at the first fill (S_FIFO ahead [{_fmt(p['ahead_lo_fifo'])}, {_fmt(p['ahead_hi_fifo'])}] sh at the "
                   f"first fill; [{_fmt(p['ahead_acc_min'])}, {_fmt(p['ahead_acc_max'])}] sh at acceptance)")
    if len(lvt) > 1:
        hid.append('which fresh-level life the first fill matched in (' + ', '.join(lvt) + ')')
    if netting:
        hid.append('a cancellation / replenishment masked inside a frame interval (net book change; section 5 finding 1)')
    lines = olines + [
        f"  - S_FIFO queue class {qc} -> positions {'|'.join(pos)}; ahead at acceptance [{_fmt(p['ahead_acc_min'])}, {_fmt(p['ahead_acc_max'])}] sh, "
        f"at the first fill [{_fmt(p['ahead_lo_fifo'])}, {_fmt(p['ahead_hi_fifo'])}] sh; tied first fills {_int(p['n_tied_first'])}, fronts after a "
        f"partial {_int(p['n_front_after_partial'])}; lo-clock differs {_s(p['lo_clock_queue_differs'])}; S_NOPRIO has no queue (price-time "
        f"premise stated, not tested)",
        f"  - fresh-level lives compatible with the first fill{marker}: "
        + ('; '.join(f"level {_s(r['level_id'])} created {_rel_s(r.get('t_create_ms', r['t_open_ms']), ws)}, status {_s(r['status'])}"
                     + (f"/{_s(r['reason'])}" if 'reason' in r else '')
                     + (f", creation context {_s(r['creation_context'])}" if 'creation_context' in r else
                        (f", mechanism {_s(r['mechanism'])}" if 'mechanism' in r else ''))
                     + ''.join(f", {c} {_s(r[c])}" for c in ('alt_netting_possible', 'alt_first_other_address', 'timing_basis') if c in r)
                     + (f", members {_int(r['n_members'])}" if 'n_members' in r else '') for r in lv) if lv else 'none'),
        f"  - {'candidate / fill rows (queue_front_v2)' if v2 else 'v1 fronts.csv rows'} of this hash{marker}: {'; '.join(crow) if crow else 'none'}"]
    if v2:
        lines.append(f"  - endpoint occupancy (queue_front_v2 occupancy.csv): {'; '.join(occ) if occ else ('none saved' if QF2.get('occupancy') is not None else 'file absent')}")
    return new_link(ch, '3', 'queue', status='filled' if v2 else 'filled_v1_labels',
                    source=('queue_front_v2/levels+candidates' if v2 else 'queue_front/levels.csv+fronts.csv (v1 labels, see section 5)')
                    + '; post_fill/orders.csv; life fills', evidence_grade='scenario_queue_history', alternatives=alts, n_histories=len(alts),
                    hidden_action='; '.join(hid), n_fills=len(fills), n_orders_rendered=n_orders,
                    pending_reason='' if v2 else (sib['reason'] or 'queue_front_v2 not ok'),
                    summary=f"class {qc}; {len(lv)} compatible level(s){marker}", lines=lines)


def link_ending(ch, p, led_rows, pu, ep_rows, CEN, PU_all):
    ws = _int(p['ws_ms'])
    hyps = ledger_hyps(led_rows)
    union = [h for h in HYP if h in set(_s(p['union_histories']).split('|'))]
    per = {s: [h for h in HYP if hyps.get(h, {}).get(s)] for s in SCEN}
    views = sorted(pu)
    pl_views = [v for v in PULL_VIEWS if 'PL' in per[v]]
    chk = dict(pulls_views_equal_pl_feasible_views=bool(views == sorted(pl_views)),
               ledger_union_equals_post_fill_union=bool([h for h in HYP if hyps.get(h, {}).get('union')] == union))
    grade = ('unresolved_alternatives' if 'EC' in union else ('cancel_compatible_ending' if views else 'scenario_dependent_ending'))
    cen = []
    for v in views:
        cr = census_rows(CEN, ch['rank'], ch['day'], (f'pull_possible:{v}:cancel_possible',))
        n_c = int(pd.to_numeric(cr['n']).sum()) if len(cr) else 0
        m = ((PU_all['scope'].to_numpy(object) == f"rank{ch['rank']}") & (PU_all['day'].to_numpy(object) == ch['day'])
             & (PU_all['view'].to_numpy(object) == v))
        chk[f'census_cancel_possible_{v}_equals_pulls_rows'] = bool(n_c == int(m.sum()))
        cen.append(f'pull_possible:{v}:cancel_possible {n_c}')
    alts = list(union) or ['EX', 'PL']
    hid = ''
    if len(alts) > 1:
        r_ = hyps.get('PL', {})
        hid = (f"an unobserved cancel of the remainder (PL: withdrawn [{_fmt(r_.get('remaining_lo'))}, "
               f"{_fmt(r_.get('remaining_hi')) if np.isfinite(_fin(r_.get('remaining_hi'))) else 'inf'}] sh in "
               f"[{_rel_s(r_.get('window_lo'), ws)}, {_rel_s(r_.get('window_hi'), ws)}]) vs exhaustion by the last fill (EX): if exhausted "
               "there was no pull decision; if pulled, the END context is a candidate trigger only")
    ends = [x for x in ep_rows if _s(x['link']) == 'END']
    lines = [f"  - feasible end histories (union) {'|'.join(union) or 'none'}; per scenario "
             + ', '.join(f"{s} {{{','.join(per[s])}}}" for s in SCEN),
             f"  - cancel-compatible views (pulls.csv): {', '.join(views) or 'none'}"
             + ''.join(f"; {v}: removal_class {_s(pu[v]['removal_class'])}, removal set {_s(pu[v]['removal_set'])}, window "
                       f"[{_rel_s(pu[v]['removal_lo_ms'], ws)}, {_rel_s(pu[v]['removal_hi_ms'], ws)}] ({_s(pu[v]['window_status'])}), "
                       f"identification {_s(pu[v]['identification'])}, pull_identified {_b(pu[v]['pull_identified'])}, most supported "
                       f"{_s(pu[v]['most_supported'])} ({_s(pu[v]['support_grade'])})" for v in views),
             '  - saved END logic rows: ' + ('; '.join(f"{_s(x['mechanism'])} {_s(x['identification_status'])}"
                                                      + (f" (window condition {_s(x['conditional_status'])})" if _s(x['conditional_status']) else '')
                                                      for x in ends) or 'none')]
    return new_link(ch, '4a', 'ending', source='order_ledger.csv size_end; post_fill/orders.csv; pull/pulls.csv; episodes.csv; census_graded.csv',
                    evidence_grade=grade, census=f"rank{ch['rank']} {ch['day']} " + ('; '.join(cen) if cen else 'not cancel-compatible in S_FIFO / S_NOPRIO'),
                    alternatives=alts, n_histories=len(alts), hidden_action=hid,
                    summary=f"{'|'.join(union)}; cancel-compatible {', '.join(views) or 'none'}", lines=lines, checks=chk)


def link_repost(ch, p, led_rows, CEN, P_all):
    nk, ns, nh = _s(p['next_kind']), _s(p['next_link_status']), _s(p['next_hash']).lower()
    lk = [r for r in led_rows if _s(r['row_kind']) == 'link']
    kids = [r for r in lk if _s(r['role']) == 'child']
    pars = [r for r in lk if _s(r['role']) == 'parent']
    every_unrel = all('unrelated' in _s(r['hypothesis']).split('|') for r in lk)
    if nk == 'none':
        alts, grade, sub = ['pull_without_repost', 'unfilled_repost', 'no_pull'], 'unresolved_alternatives', 'none'
        hid = 'whether the order was pulled without a repost, reposted with an order that never filled, or exhausted (invisible in fills)'
    else:
        need(ns in LINK_SUBSET, f"chain {ch['chain']}: unknown successor link status {ns!r} (refused)")
        sub = LINK_SUBSET[ns]
        first = [r for r in kids if _s(r['other_hash']).lower() == nh]
        alts = (_s(first[0]['hypothesis']).split('|') if first else ['replace', 'replenish', 'unrelated'])
        grade = 'scenario_dependent_link'
        hid = (f"whether the successor {_h12(nh)} was a new submission replacing this order (replace), a top-up (replenish), an order already "
               "resting (overlap) or unrelated: a sequential timing never identifies a repost or its intent")
    cr = census_rows(CEN, ch['rank'], ch['day'], (f'successor_{nk}',), source='chains/post_fill/orders.csv')
    crs = cr[cr['subset'].to_numpy(object) == sub]
    n_c = int(pd.to_numeric(crs['n']).sum()) if len(crs) else 0
    tot = _int(cr['n_action_total'].iloc[0]) if len(cr) else 0
    m = ((P_all['_rank'].to_numpy(np.int64) == ch['rank']) & (P_all['day'].to_numpy(object) == ch['day'])
         & (_sarr(P_all['next_kind']) == nk))
    m_s = m & (_sarr(P_all['next_link_status']) == (ns if nk != 'none' else 'none'))
    chk = dict(unrelated_in_every_saved_link_hypothesis=bool(every_unrel), census_successor_count_equals_orders=bool(n_c == int(m_s.sum())),
               census_successor_total_equals_orders=bool(tot == int(m.sum())))

    def _lk(r):
        return (f"{_s(r['kind'])} {_h12(r['other_hash'])} ({_s(r['status'])}; gap [{_fmt(r['gap_lo_ms'], 0)}, {_fmt(r['gap_hi_ms'], 0)}] ms; "
                f"{_s(r['hypothesis'])})")
    lines = [(f"  - successor: none ({'no later first-filled same-side order: an unfilled successor or none (invisible in fills)'})" if nk == 'none' else
              f"  - successor: {nk} to {_h12(nh)} (first of {_int(p['n_children'])} filled successor(s)); link status {ns} -> {sub}; gap "
              f"[{_fmt(p['next_gap_lo_ms'], 0)}, {_fmt(p['next_gap_hi_ms'], 0)}] ms; d_ticks {_int(p['next_d_ticks'], 0)}; identified repost: no"),
             '  - saved child links: ' + ('; '.join(_lk(r) for r in kids) or 'none'),
             '  - saved parent links (was this post itself a repost?): ' + ('; '.join(_lk(r) for r in pars) or 'none')
             + ' - never identified (unrelated stays feasible)']
    return new_link(ch, '4b', 'repost', source='post_fill/orders.csv; order_ledger.csv links; census_graded.csv', evidence_grade=grade,
                    census=f"rank{ch['rank']} {ch['day']} successor_{nk} {sub}: {n_c} of {tot}", alternatives=alts, n_histories=len(alts),
                    hidden_action=hid, summary=f"{nk} {sub}" + (f" to {_h12(nh)}" if nk != 'none' else ''), lines=lines, checks=chk)


def link_takes(ch, EV, ep_rows, ws):
    e = EV[(EV['day'].to_numpy(object) == ch['day']) & (np.array([_int(x) for x in EV['mid']]) == ch['mid'])
           & (EV['addr_l'].to_numpy(object) == ch['addr'])]
    st = np.array([_int(x) for x in e['settle_ms']], np.int64)
    e = e[(st >= ch['t0']) & (st <= ch['t1'])].copy()
    e['_st'] = [_int(x) for x in e['settle_ms']]
    e = e.sort_values(['_st', 'sid'], kind='stable')
    case_sids = {_s(c['key']) for c in ch['take_cases']}
    recs = e.to_dict('records')
    vec = {c: tuple(_s(r[f'dir_{c}']) for r in recs) for c in CLOCKS}
    n_clock = len(set(vec.values()))
    some = sum(_s(r['o4_status']) == 'reduces_some' for r in recs)
    undet = Counter(_s(r['dir_D0']) for r in recs if _s(r['dir_D0']) not in DIR_DETERMINATE)
    dis = [r for r in recs if len({_s(r[f'dir_{c}']) for c in CLOCKS}) > 1]
    if not recs:
        alts = ['no_take_record_in_the_window']
    else:
        uniq = []                                                 # one history per distinct direction vector over the three clocks
        for c in CLOCKS:
            if all(vec[c] != vec[u] for u in uniq):
                uniq.append(c)
        alts = [f"{'/'.join(c2 for c2 in CLOCKS if vec[c2] == vec[c])}: {','.join(vec[c])}" for c in uniq]
        if some:
            alts = [f'{a} + variants_reduce' for a in alts] + [f'{a} + variants_not_reduce' for a in alts]
    hid = []
    if n_clock > 1:
        hid.append(f"the match instants of own fills around the anchor: D0 / D1000 / pend give different directions for {len(dis)} record(s) "
                   f"({', '.join(_s(r['sid'])[:13] for r in dis[:4])})")
    if some:
        hid.append(f'an excluded or unobserved leg: the saved variants disagree on reducing |q| for {some} record(s)')
    lines = []
    for r in recs:
        tg = ' [the selected take case]' if _s(r['sid']) in case_sids else ''
        lines.append(f"  - take {_s(r['sid'])[:13]}: YES {'buy' if _int(r['s']) > 0 else 'sell'} {_fmt(abs(_fin(r['dq'])))} sh @ {_fmt(r['p_avg'], 3)}, "
                     f"settle {_rel_s(r['settle_ms'], ws)}, anchor {_rel_s(r['t_anchor_ms'], ws)} ({_s(r['pop_status'])}); q_D0 {_sg(r['q_D0'])} -> "
                     + ', '.join(f"{c} {_s(r[f'dir_{c}'])}" for c in CLOCKS)
                     + f"; class {_s(r['take_class'])} (rec {_s(r['cls_rec'])} / cert {_s(r['cls_cert_0'])} / sim {_s(r['cls_sim'])}); O4 "
                     f"{_s(r['o4_status'])}; status {_s(r['status'])}{tg}")
    miss = sorted(case_sids - {_s(r['sid']) for r in recs})
    tl = [x for x in ep_rows if _s(x['link']) == 'TAKE']
    if tl:
        lines.append('  - saved TAKE logic rows (selected case): ' + '; '.join(f"{_s(x['mechanism'])} {_s(x['identification_status'])}" for x in tl))
    if miss:
        lines.append(f"  - selected take case(s) not among the window's records: {', '.join(m_[:13] for m_ in miss)}")
    d0 = Counter(_s(r['dir_D0']) for r in recs)
    return new_link(ch, '5', 'takes', source='take_logic_v2/events.parquet; cases.csv; episodes.csv', evidence_grade='observed_execution',
                    alternatives=alts, n_histories=len(alts), hidden_action='; '.join(hid), observed_public=True, n_records=len(recs),
                    summary=(f"{len(recs)} take record(s); D0 {dict(sorted(d0.items()))}; clock-disagreeing {len(dis)}; undetermined {dict(undet)}"
                             if recs else 'no take record in the episode window'), lines=lines,
                    checks=dict(selected_take_cases_in_window=bool(not miss)))


def views(E):
    """10 s / 60 s / END views (valid = st_h ok and a finite v_h), the common cohort (valid at all three) and the resolution view."""
    q = E['q'].to_numpy(float)
    ff = E['fee_fill'].to_numpy(float)
    out, okc = {}, np.ones(len(E), bool)
    for h in HORIZONS:
        st = _sarr(E[f'st_{h}'])
        v = E[f'v_{h}'].to_numpy(float)
        ok = (st == OK) & np.isfinite(v)
        okc &= ok
        g = float(v[ok].sum())
        out[h] = dict(n_valid=int(ok.sum()), n_all=int(len(E)), sh_valid=float(q[ok].sum()), sh_all=float(q.sum()), gross=g,
                      net=g - float(ff[ok].sum()), c_sh=(100. * g / q[ok].sum()) if q[ok].sum() > 0 else NAN,
                      invalid=dict(sorted(Counter(st[~ok]).items())))
    out['common'] = dict(n_valid=int(okc.sum()), sh_valid=float(q[okc].sum()),
                         **{f'c_sh_{h}': (100. * float(E[f'v_{h}'].to_numpy(float)[okc].sum()) / q[okc].sum()) if q[okc].sum() > 0 else NAN
                            for h in HORIZONS})
    rs = np.isin(_sarr(E['st_res']), list(RES_VALID)) & np.isfinite(E['net_res'].to_numpy(float))
    out['res'] = dict(n_valid=int(rs.sum()), n_all=int(len(E)), sh_valid=float(q[rs].sum()), net=float(E['net_res'].to_numpy(float)[rs].sum()),
                      c_sh=(100. * float(E['net_res'].to_numpy(float)[rs].sum()) / q[rs].sum()) if q[rs].sum() > 0 else NAN)
    return out


def _vline(lab, V):
    return (f"  - {lab}: " + '; '.join(f"{h.upper() if h == 'end' else h} valid {V[h]['n_valid']}/{V[h]['n_all']} rows, {_fmt(V[h]['sh_valid'])}/"
                                       f"{_fmt(V[h]['sh_all'])} sh, gross {_sg(V[h]['gross'])} $, net {_sg(V[h]['net'])} $, {_sg(V[h]['c_sh'])} c/sh"
                                       + (f" (invalid {V[h]['invalid']})" if V[h]['invalid'] else '') for h in HORIZONS)
            + f"; common cohort {V['common']['n_valid']} rows: " + ', '.join(f"{h} {_sg(V['common'][f'c_sh_{h}'])}" for h in HORIZONS)
            + f" c/sh; resolution (net_res) valid {V['res']['n_valid']}/{V['res']['n_all']}: {_sg(V['res']['net'])} $ ({_sg(V['res']['c_sh'])} c/sh)")


def _cl_sum(X):
    nd = ~X['dust'].to_numpy(bool) if len(X) else np.zeros(0, bool)
    hm = X['hold_ms'].to_numpy(float)[nd] if len(X) else np.zeros(0)
    return dict(n=int(nd.sum()), n_dust=int((~nd).sum()), sh=float(X['sh'].to_numpy(float).sum()) if len(X) else 0.,
                pnl=float(X['pnl'].to_numpy(float).sum()) if len(X) else 0., net=float(X['net'].to_numpy(float).sum()) if len(X) else 0.,
                hold_p50_s=float(np.median(hm)) / 1000. if len(hm) else NAN)


def _cltxt(lab, s):
    return (f"{lab} {s['n']} close(s) (+{s['n_dust']} dust), {_fmt(s['sh'])} sh, pnl {_sg(s['pnl'])} $, net {_sg(s['net'])} $, "
            f"hold p50 {_fmt(s['hold_p50_s'], 1)} s")


def link_value(ch, p, FV, CL, CM, MG):
    es, h, ws = ch['es'], ch['order_hash'], _int(p['ws_ms'])
    L = FV[(np.array([_int(x) for x in FV['mid']]) == ch['mid']) & (FV['addr_l'].to_numpy(object) == ch['addr'])]
    stt = _sarr(L['status'])
    lot = (stt == OK) & ~np.array([bool(_b(x)) for x in L['self_trade']], bool)
    sm = np.array([_int(x) for x in L['settle_ms']], np.int64)
    win = (sm >= ch['t0']) & (sm <= ch['t1'])
    E = L[lot & win]
    role = _sarr(E['role'])
    V = views(E)
    Vm, Vt = views(E[role == 'maker']), views(E[role == 'taker'])
    ve = E['v_end'].to_numpy(float)
    rec = dict(
        ep_n_legs=(len(E), es['ep_n_legs']), ep_n_maker=(int((role == 'maker').sum()), es['ep_n_maker']), ep_n_taker=(int((role == 'taker').sum()), es['ep_n_taker']),
        ep_sh=(float(E['q'].to_numpy(float).sum()), es['ep_sh']), ep_v_end=(float(np.nansum(ve)), es['ep_v_end']),
        ep_n_v_end_na=(int(np.isnan(ve).sum()), es['ep_n_v_end_na']), ep_fee=(float(E['fee_fill'].to_numpy(float).sum()), es['ep_fee']),
        ep_net_res=(float(np.nansum(E['net_res'].to_numpy(float))), es['ep_net_res']), ep_mq=(float(E['mq'].to_numpy(float).sum()), es['ep_mq']),
        ep_rq=(float(E['rq'].to_numpy(float).sum()), es['ep_rq']), ep_n_excluded=(int((~lot & win & (stt != OK)).sum()), es['ep_n_excluded']))
    chk = {f'episode_{k}_reconciles': bool(_close(a, b)) for k, (a, b) in rec.items()}
    chk['episode_legs_on_the_chain_day'] = bool((_sarr(L['day']) == ch['day']).all())
    # the case order's own maker rows (post_fill's definition: every status) and its PNL cohort
    own = L[(_sarr(L['role']) == 'maker') & (_sarr(L['hash']) == h)]
    ook = _sarr(own['status']) == OK
    oself = np.array([bool(_b(x)) for x in own['self_trade']], bool)
    coh = ook & ~oself & np.isin(_sarr(own['st_res']), list(RES_VALID))
    for hz in ('1s', '10s', '60s'):
        coh &= _sarr(own[f'st_{hz}']) == OK
    prec = dict(n_own_rows=(len(own), p['n_own_rows']), mq_sh=(float(own['mq'].to_numpy(float)[ook].sum()), p['mq_sh']),
                rq_sh=(float(own['rq'].to_numpy(float)[ook].sum()), p['rq_sh']), pnl_rows=(int(coh.sum()), p['pnl_rows']),
                pnl_shares=(float(own['q'].to_numpy(float)[coh].sum()), p['pnl_shares']),
                net_res=(float(own['net_res'].to_numpy(float)[coh].sum()), p['net_res']),
                fee_fill=(float(own['fee_fill'].to_numpy(float)[coh].sum()), p['fee_fill']),
                **{f'v_{hz}': (float(own[f'v_{hz}'].to_numpy(float)[coh].sum()), p[f'v_{hz}']) for hz in ('1s', '10s', '60s')})
    chk.update({f'own_{k}_reconciles_post_fill': bool(_close(a, b)) for k, (a, b) in prec.items()})
    Vo = views(own[ook & ~oself])
    # completed turnover (closes.parquet) and residual (rq)
    X = CL[(np.array([_int(x) for x in CL['mid']]) == ch['mid']) & (CL['addr_l'].to_numpy(object) == ch['addr'])]
    ek = list(zip(_sarr(X['entry_sid']), [_int(x) for x in X['entry_mk_idx']], _sarr(X['entry_role'])))
    ck = list(zip(_sarr(X['close_sid']), [_int(x) for x in X['close_mk_idx']], _sarr(X['close_role'])))
    keys = set(zip(_sarr(E['sid']), [_int(x) for x in E['mk_idx']], _sarr(E['role'])))
    e_in = np.array([k in keys for k in ek], bool)
    c_in = np.array([k in keys for k in ck], bool)
    att = Counter()
    for k, c, s_ in zip(ek, ck, X['sh'].to_numpy(float)):
        att[k] += s_
        att[c] += s_
    leg_ok = [abs(att.get(k, 0.) - m_) <= ABS_TOL + REL_TOL * max(1., abs(m_)) for k, m_ in
              zip(zip(_sarr(E['sid']), [_int(x) for x in E['mk_idx']], _sarr(E['role'])), E['mq'].to_numpy(float))]
    chk['closes_per_episode_leg_equal_mq'] = bool(all(leg_ok))
    inside, later, earlier = _cl_sum(X[e_in & c_in]), _cl_sum(X[e_in & ~c_in]), _cl_sum(X[~e_in & c_in])
    om = (((_sarr(X['entry_hash']) == h) & (_sarr(X['entry_role']) == 'maker')) | ((_sarr(X['close_hash']) == h) & (_sarr(X['close_role']) == 'maker')))
    oc = _cl_sum(X[om])
    rq = E['rq'].to_numpy(float)
    sgn, pp = E['s'].to_numpy(float), E['p'].to_numpy(float)
    me, VV = E['m_end'].to_numpy(float), E['V'].to_numpy(float)
    okr = np.isin(_sarr(E['st_res']), list(RES_VALID)) & np.isfinite(VV)
    r_end = float((sgn * rq * (me - pp))[np.isfinite(me)].sum())
    r_res = float((sgn * rq * (VV - pp))[okr].sum())
    # address-market closes accounting and the matched-gap interval accounting
    cm = CM[(np.array([_int(x) for x in CM['mid']]) == ch['mid']) & (CM['addr_l'].to_numpy(object) == ch['addr'])
            & (CM['day'].to_numpy(object) == ch['day'])]
    mg = MG[(MG['day'].to_numpy(object) == ch['day']) & (np.array([_int(x) for x in MG['m']]) == ch['mid'])
            & (_sarr(MG['party']) == f"rank{ch['rank']}")]
    mgv = {_s(r['view']): r for r in mg.to_dict('records')}
    cr = mgv.get('carry')
    chk['matched_gap_carry_row_reconciles_episode_summary'] = bool(
        cr is not None and _s(cr['status']) == _s(es['mg_status']) and _close(cr['close_net'], es['mg_close_net'])
        and _close(cr['resid'], es['mg_resid']) and _close(cr['total'], es['mg_total']))
    chk['closes_markets_row_present'] = bool(len(cm) == 1)
    same = [np.sign(V[hz]['gross']) for hz in HORIZONS if V[hz]['n_valid'] > 0]
    lines = [f"  - episode lot legs (status ok, not self-trades, settle_ms in [{_rel_s(ch['t0'], ws)}, {_rel_s(ch['t1'], ws)}]): {len(E)} "
             f"(maker {int((role == 'maker').sum())}, taker {int((role == 'taker').sum())}), {_fmt(E['q'].to_numpy(float).sum())} sh; excluded rows in "
             f"the window {rec['ep_n_excluded'][0]}; matched (mq) {_fmt(rec['ep_mq'][0])} sh, residual (rq) {_fmt(rec['ep_rq'][0])} sh",
             _vline('episode, all legs', V), _vline('episode, maker legs', Vm), _vline('episode, taker legs', Vt),
             '  - completed turnover (closes.parquet, closes.py FIFO): ' + '; '.join((_cltxt('entries closed inside the episode', inside),
                                                                                       _cltxt('entries closed after it', later),
                                                                                       _cltxt('closes of earlier inventory', earlier))),
             f"  - residual of the episode legs: {_fmt(rq.sum())} sh unclosed; value to END {_sg(r_end)} $ (m_end finite for "
             f"{int(np.isfinite(me).sum())}/{len(E)} legs), to resolution {_sg(r_res)} $ ({int(okr.sum())}/{len(E)} resolved)",
             _vline(f"case order {_h12(h)} own maker rows ({len(own)} rows, {int(ook.sum())} ok)", Vo)
             + f"; mq {_fmt(prec['mq_sh'][0])} sh, rq {_fmt(prec['rq_sh'][0])} sh; " + _cltxt('its closes', oc),
             ('  - address in this market (closes_markets): ' + (
                 f"{_int(cm['n_closes'].iloc[0])} closes, {_fmt(cm['close_sh'].iloc[0])} sh, close net "
                 f"{_sg(_fin(cm['close_pnl'].iloc[0]) - _fin(cm['close_fee_fill'].iloc[0]))} $; residual {_fmt(cm['resid_sh'].iloc[0])} sh, to END "
                 f"{_sg(cm['resid_end'].iloc[0])} $, to resolution {_sg(cm['resid_res'].iloc[0])} $; status {_s(cm['status'].iloc[0])}"
                 if len(cm) == 1 else 'no row')),
             '  - matched-gap interval accounting (matched_gap_r1): ' + ('; '.join(
                 f"{v} ({'primary' if v == 'carry' else 'beside'}) status {_s(r['status'])}: close net {_sg(r['close_net'])} $ on "
                 f"{_fmt(r['close_sh'])} sh ({_sg(r['close_net_c_sh'])} c/sh), carry {_sg(r['carry'])}, resid {_sg(r['resid'])}, total {_sg(r['total'])} $; "
                 f"reading {_s(r['reading'])}" for v, r in sorted(mgv.items())) or 'no row')]
    return new_link(ch, '6', 'value', source='econ/fills_valued.parquet; closes/closes.parquet; closes/closes_markets.parquet; matched_gap_r1/per_market.csv; '
                    'episode_summary.csv; post_fill/orders.csv', evidence_grade='recorded_fill_accounting',
                    alternatives=['recorded_fills_under_every_retained_history'], n_histories=1, observed_public=True, n_records=len(E),
                    hidden_action='', summary=(f"END net {_sg(V['end']['net'])} $ ({V['end']['n_valid']}/{V['end']['n_all']} valid); 10s/60s/END "
                                               f"same sign {len(set(same)) <= 1}"), lines=lines, checks=chk)


# ------------------------------------------------------------------ reading / rendering / verification ------------------------------------------------------------------
def expected_token(L):
    """the declared reading of a link, from its retained histories alone (the verifier's rule, never the renderer's)."""
    if L['status'] == 'pending':
        return 'PENDING'
    if L['link'] == 'takes' and int(L['n_records']) == 0:
        return 'NO-EVENT'
    return 'TIED' if int(L['n_histories']) >= 2 else 'ONE-HISTORY'


def reading_token(L):
    if _mut() == 'tied_as_identified' and L['status'] != 'pending' and int(L['n_histories']) >= 2:
        return 'IDENTIFIED'                                       # planted error: a tied link rendered as identified (its first history)
    return expected_token(L)


def render_link(L):
    tok = L['reading']
    n = int(L['n_histories'])
    head = (f"- **{L['item']} {TITLE[L['link']]}** — **{tok}** ({n} retained histor{'y' if n == 1 else 'ies'}) · grade `{L['evidence_grade']}` · "
            f"{L['status']}" + (f" · census: {L['census']}" if L['census'] else ''))
    out = [head]
    if tok == 'PENDING':
        out.append(f"  - pending: {L['pending_reason']}")
        return out
    if tok == 'IDENTIFIED':
        out.append(f"  - identified: {L['alternatives'][0]}")
    elif tok == 'TIED':
        out.append(f"  - retained histories: {' | '.join(L['alternatives'])}")
        out.append(f"  - hidden action that changes the explanation: {L['hidden_action']}")
    elif tok == 'NO-EVENT':
        out.append('  - no take record of this address in the episode window: every retained history agrees')
    else:
        out.append(f"  - every retained history agrees: {L['alternatives'][0]}")
    return out + L['lines']


def agreement(ls):
    ag = [x['link'] for x in ls if x['status'] != 'pending' and x['histories_agree']]
    tied = [x for x in ls if x['status'] != 'pending' and not x['histories_agree']]
    pend = [x['link'] for x in ls if x['status'] == 'pending']
    return dict(agree=ag, tied=[x['link'] for x in tied], pending=pend, first_branch=tied[0]['link'] if tied else '',
                hidden={x['link']: x['hidden_action'] for x in tied})


def chains_md(mode, chains, links, sib, asof):
    by = {}
    for L in links:
        by.setdefault(L['chain'], []).append(L)
    out = [f"# Six complete chains v3 (posting_chain_contract.md section 4) - {mode.upper()}", '',
           f"As of {asof}. Mode **{mode}**. Selection (predetermined, not outcome-selected): per rank, the first two episodes in "
           "`order_estimator/cases.csv` row order of first appearance = `correction_v1/paths.md`. Saved outputs only; descriptive; one consumed "
           "day (Oct 24) of four; filled orders are outcome-selected; no identified intent or private action. Times are venue ms relative to "
           "the market window start ws.", '',
           'Readings: **ONE-HISTORY** = every retained history agrees; **TIED** = at least two retained histories change the explanation (the '
           'hidden action is named); **NO-EVENT** = no take record; **PENDING** = a sibling output is absent (draft). Grades: '
           + ', '.join(f'`{g}`' for g in GRADES) + '.', '',
           'Siblings: ' + '; '.join(f"{nm} **{s['status']}**" + (f" ({s['reason']})" if s['reason'] else '') for nm, s in sorted(sib.items()))
           + '. Without queue_front_v2 the queue links carry v1 labels (see section 5); without posting_opportunity the pre-state links are PENDING.', '',
           '| chain | rank | episode | ' + ' | '.join(TITLE[lk] for _, lk in LINKS) + ' | first branch |',
           '|---|---|---|' + '---|' * len(LINKS) + '---|']
    for ch in chains:
        ls = by[ch['chain']]
        rd = {x['link']: x['reading'] for x in ls}
        out.append(f"| {ch['chain']} | {ch['rank']} | {ch['episode_id']} | " + ' | '.join(rd.get(lk, '') for _, lk in LINKS)
                   + f" | {agreement(ls)['first_branch'] or 'none'} |")
    out.append('')
    for ch in chains:
        ls = by[ch['chain']]
        out += [f"## Chain {ch['chain']} - rank {ch['rank']}: episode {ch['episode_id']}", '',
                f"day {ch['day']}, market {ch['mid']}, address {ch['addr']}; case order {_h12(ch['order_hash'])}; cases.csv data row {ch['case_row']}; "
                f"episode window [{_rel_s(ch['t0'], ch['ws'])}, {_rel_s(ch['t1'], ch['ws'])}]", '']
        for L in ls:
            out += render_link(L)
        a = agreement(ls)
        out += [f"- **AGREEMENT** histories agree on: {', '.join(a['agree']) or 'none'}; TIED: {', '.join(a['tied']) or 'none'}; pending: "
                f"{', '.join(a['pending']) or 'none'}; first branching link: {a['first_branch'] or 'none'}"]
        out += [f"  - {k}: {v}" for k, v in a['hidden'].items()]
        out.append('')
    return '\n'.join(out) + '\n'


MARK_RE = re.compile(r'^- \*\*(?P<item>\d[ab]?) (?P<title>[A-Z-]+)\*\* — \*\*(?P<tok>[A-Z-]+)\*\* \((?P<n>\d+) retained histor(?:y|ies)\)')
CHAIN_RE = re.compile(r'^## Chain (?P<k>\d+) ')
ORDER_RE = re.compile(r'^  - order (?P<h>0x[0-9a-f]+): (?P<k>\d+) fill\(s\)')


def parse_md(md):
    blocks, cur_chain, cur = {}, None, None
    for line in md.splitlines():
        m = CHAIN_RE.match(line)
        if m:
            cur_chain, cur = int(m['k']), None
            continue
        m = MARK_RE.match(line)
        if m and cur_chain is not None:
            cur = (cur_chain, m['title'])
            blocks[cur] = dict(token=m['tok'], n=int(m['n']), lines=[])
            continue
        if line.startswith('#') or line.startswith('- **'):
            cur = None
        if cur is not None:
            blocks[cur]['lines'].append(line)
    return blocks


def verify(links, md, facts):
    """the rendering verifier (contract properties, never the renderer's code): a TIED link (>= 2 retained histories) renders TIED and
    nothing renders IDENTIFIED; repeated same-hash fills render as ONE order; the submitted size is never set (never the filled lower
    bound); every link rendered once. -> violations."""
    v = []
    B = parse_md(md)
    if '**IDENTIFIED**' in md or re.search(r'identified repost: yes', md):
        v.append('an IDENTIFIED token / identified repost in the rendering')
    for L in links:
        key = (int(L['chain']), TITLE[L['link']])
        b = B.get(key)
        if b is None:
            v.append(f'chain {key[0]} {key[1]}: not rendered')
            continue
        want = expected_token(L)
        if b['token'] != want or L['reading'] != want or b['n'] != int(L['n_histories']):
            v.append(f"chain {key[0]} {key[1]}: rendered {b['token']} / recorded {L['reading']} with {L['n_histories']} retained histories "
                     f"(declared {want}: a TIED link is never rendered as identified)")
        if L['link'] == 'queue' and L['status'] != 'pending':
            f = facts[int(L['chain'])]
            od = [ORDER_RE.match(x) for x in b['lines']]
            od = [m for m in od if m]
            if len(od) != f['n_distinct_hashes'] or sum(int(m['k']) for m in od) != f['n_fills'] or int(L['n_orders_rendered']) != f['n_distinct_hashes']:
                v.append(f"chain {key[0]} QUEUE: {len(od)} order line(s) for {f['n_fills']} fill(s) of {f['n_distinct_hashes']} hash(es) "
                         '(repeated same-hash fills must render as one order)')
        if L['link'] == 'post' and L['status'] != 'pending':
            f = facts[int(L['chain'])]
            sl = [x for x in b['lines'] if x.startswith('  - size:')]
            if (len(sl) != 1 or 'submitted size unknown' not in sl[0] or re.search(r'submitted size [0-9]', sl[0])
                    or np.isfinite(_fin(L['submitted_size']))):
                v.append(f"chain {key[0]} POST: a submitted size is set ({_fmt(L['submitted_size'])} sh; filled lower bound {_fmt(f['filled'])} sh)")
    return v


# ------------------------------------------------------------------ claims ------------------------------------------------------------------
def make_claims(mode, chains, links, sel, sel_md, checks, bind, sib, violations):
    L = links
    n_t = Counter(x['reading'] for x in L)
    pend = [f"chain {x['chain']} {x['link']}: {x['pending_reason']}" for x in L if x['status'] == 'pending']
    by_link = {lk: dict(Counter(x['reading'] for x in L if x['link'] == lk)) for _, lk in LINKS}
    claims = dict(
        version=VERSION, mode=mode, contract=f'posting_chain_contract.md section 4 ({CONTRACT_COMMIT})',
        population=dict(unit='six predetermined episodes (correction_v1 paths.md selection), one row per link',
                        n_chains=len(chains), n_links=len(L), days=sorted({c['day'] for c in chains}),
                        episodes=[dict(chain=c['chain'], rank=c['rank'], episode_id=c['episode_id']) for c in chains]),
        predicates=dict(
            six_chains=bool(len(chains) == 6),
            selection_equals_correction_v1_paths_md=bool(sel == sel_md),
            every_chain_has_every_link=bool(all(sorted(x['link'] for x in L if x['chain'] == c['chain']) == sorted(lk for _, lk in LINKS) for c in chains)),
            every_link_graded=bool(all(x['evidence_grade'] in GRADES for x in L)),
            every_link_has_a_declared_reading=bool(all(x['reading'] in READINGS for x in L)),
            no_link_asserts_an_identified_private_action=bool(not any(x['identified_private_action'] for x in L)),
            tied_links_never_rendered_identified=bool(all(x['reading'] == 'TIED' for x in L if x['status'] != 'pending' and int(x['n_histories']) >= 2)),
            every_tied_link_names_its_hidden_action=bool(all(x['hidden_action'] for x in L if x['reading'] == 'TIED')),
            submitted_size_never_set=bool(all(not np.isfinite(_fin(x['submitted_size'])) for x in L if x['link'] == 'post')),
            same_hash_fills_render_one_order=bool(all(int(x['n_orders_rendered']) == 1 for x in L if x['link'] == 'queue' and x['status'] != 'pending')),
            rendering_verifier_clean=bool(not violations),
            no_pending_link=bool(not pend)),
        readings=dict(n_by_reading=dict(n_t), by_link=by_link,
                      by_grade=dict(Counter(x['evidence_grade'] for x in L))),
        agreement={str(c['chain']): agreement([x for x in L if x['chain'] == c['chain']]) for c in chains},
        pending=pend, siblings={k: dict(status=v['status'], reason=v['reason']) for k, v in sib.items()},
        queue_labels={str(x['chain']): x['status'] for x in L if x['link'] == 'queue'},
        reconciliation={k: v for k, v in checks.items()},
        reconciliation_all_hold=bool(all(v['holds'] for v in checks.values())),
        bindings={k: v['binding']['status'] for k, v in bind.items()},
        bindings_all_ok=bool(all(v['binding']['status'] == 'ok' for v in bind.values())),
        not_measured='no model skill, policy, intent, validation or profitability: an accounting of saved public records and labels')
    return claims


# ------------------------------------------------------------------ build / run ------------------------------------------------------------------
def write_out(out, files):
    """write-once: a partial directory (exclusive) renamed to out at the end; refused if out exists; the partial removed on a failure."""
    out = Path(out)
    need(not out.exists(), f'write-once: {out} exists (refused)')
    tmp = out.parent / f'.{out.name}.partial_{os.getpid()}'
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp.mkdir()
    try:
        for nm, b in files:
            with open(tmp / nm, 'xb') as f:
                f.write(b)
        need(not out.exists(), f'write-once: {out} appeared during the run (refused)')
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def run(mode='draft', rroot=None, droot=None, out=None, require_freeze=True, selftest=False, log=print):
    t0, asof0 = time.time(), _now()
    c_start = time.process_time()
    stages, last = {}, [c_start]

    def stage(nm):
        c = time.process_time()
        stages[nm] = round(c - last[0], 4)
        last[0] = c
        guard(nm)
        cpu_guard(nm, c_start)                                    # this run's CPU (the selftest runs many in one process)
    need(mode in ('draft', 'run'), f'unknown mode {mode!r}')
    need(C.RUN == 'v2', f'MT_RUN must be v2 (is {C.RUN!r})')
    need(selftest or not _mut(), f'MT_CH3_MUT={_mut()!r} is set: a planted mutant never runs outside the selftest (refused)')
    need(_mut() in ('',) + MUTANTS, f'unknown MT_CH3_MUT {_mut()!r} (refused)')
    rroot = Path(rroot) if rroot else C.RECEIPT
    droot = Path(droot) if droot else C.OUT
    out = Path(out) if out else rroot / 'chains' / ('chains_v3' if mode == 'run' else 'chains_v3_draft')
    need(not out.exists(), f'write-once: {out} exists (refused before any input is read)')
    sib = {nm: sibling_state(rroot, nm) for nm in SIBLINGS}
    if mode == 'run':
        bad = {nm: s['reason'] for nm, s in sib.items() if s['status'] != 'ok'}
        need(not bad, f'run requires the sibling outputs bound by their receipts: {bad} (refused; use run --draft)')
    git = _git(TOOL_FILES)
    if mode == 'run' and require_freeze:
        need(bool(git.get('unmodified')), 'freeze: the tool, its selftest and the contract must be tracked and unmodified before a run (refused)')
    stage('preflight')
    IN, cnt = Inputs(rroot), Counter()

    def path(k):
        root, rel, _ = INPUTS[k]
        return (rroot if root == 'R' else droot) / rel
    D = {}
    for k in ('cases', 'episode_summary'):
        D[k] = read_csv(path(k), k, NEED[k])
        IN.add(k, path(k), len(D[k]), INPUTS[k][2])
    pm = path('paths_md')
    need(pm.is_file(), f'input paths_md is missing: {_rel(pm)} (refused)')
    pmd = pm.read_text()
    IN.add('paths_md', pm, pmd.count('\n'), INPUTS['paths_md'][2])
    sel, sel_md = select_chains(D['cases']), paths_md_selection(pmd)
    need(len(sel) == 6, f'the predetermined selection has {len(sel)} episodes, not 6 (refused)')
    chains = chain_bases(sel, D['cases'], D['episode_summary'])
    days = sorted({c['day'] for c in chains})
    need(all(d in C.MEASURE_DAYS for d in days), f'chain days {days} are not all measurement days (refused before any other path is built)')
    mids = sorted({c['mid'] for c in chains})
    addrs = sorted({c['addr'] for c in chains})
    hashes = {(c['day'], c['mid'], c['order_hash']) for c in chains}
    for k in ('order_ledger', 'episodes', 'post_regrade_orders', 'episodes_post_regraded', 'census_graded', 'post_orders', 'pulls', 'qf1_levels',
              'qf1_fronts', 'matched_gap'):
        D[k] = read_csv(path(k), k, NEED[k])
        IN.add(k, path(k), len(D[k]), INPUTS[k][2])
    for k, mc in (('qf1_levels', 'mid'), ('qf1_fronts', 'mid'), ('matched_gap', 'm')):    # the chain markets only (rows counted in the receipt)
        D[k] = D[k][np.isin(np.array([_int(x) for x in D[k][mc]], np.int64), np.array(mids, np.int64))].reset_index(drop=True)
    for k in ('post_regrade_orders', 'post_orders'):
        D[k]['_rank'] = [_int(x) for x in D[k]['rank']]
    stage('csv_inputs')
    D['events'] = read_pq(path('events'), 'events', EV_READ)
    IN.add('events', path('events'), len(D['events']), INPUTS['events'][2])
    D['events'] = D['events'][np.isin(np.array([_int(x) for x in D['events']['mid']], np.int64), np.array(mids, np.int64))].reset_index(drop=True)
    D['events']['addr_l'] = [str(a).lower() for a in D['events']['addr']]
    D['closes_markets'] = read_pq(path('closes_markets'), 'closes_markets', CM_READ)
    D['closes_markets']['addr_l'] = [str(a).lower() for a in D['closes_markets']['addr']]
    IN.add('closes_markets', path('closes_markets'), len(D['closes_markets']), INPUTS['closes_markets'][2])
    need(len(days) == 1, f'chain days {days}: the closes reader is declared for one day (refused)')
    D['closes'] = read_groups(path('closes'), 'closes', CL_READ, mids, addrs, cnt, day=days[0])
    IN.add('closes', path('closes'), len(D['closes']), INPUTS['closes'][2])
    stage('closes')
    D['fv'] = read_groups(path('fills_valued'), 'fills_valued', FV_READ, mids, addrs, cnt)
    IN.add('fills_valued', path('fills_valued'), len(D['fv']), INPUTS['fills_valued'][2])
    stage('fills_valued')
    life = {}
    for d, m in sorted({(c['day'], c['mid']) for c in chains}):
        lp = droot / 'life' / f'{d}_{m}_fills.parquet'
        t = read_pq(lp, f'life/{d}_{m}', LIFE_READ)
        hs = {h for (dd, mm, h) in hashes if dd == d and mm == m}
        life[(d, m)] = t[np.isin(np.array([_s(x).lower() for x in t['mk_hash']], object), sorted(hs))].copy()
        IN.add(f'life/{d}_{m}_fills.parquet', lp, len(life[(d, m)]), LIFE_BIND + (f'life/{d}_{m}_fills.parquet',))
    PO = load_sibling(sib['posting_opportunity'], mids)
    QF2 = load_sibling(sib['queue_front_v2'], mids)
    if mode == 'run':
        bad = {nm: s['reason'] for nm, s in sib.items() if s['status'] != 'ok'}
        need(not bad, f'run requires the sibling outputs with their declared columns: {bad} (refused)')
    stage('small_inputs')
    # ---- indices ----
    P, T = D['post_orders'], D['post_regrade_orders']
    for X in (P, T):
        X['_key'] = list(zip(_sarr(X['day']), [_int(x) for x in X['mid']], [_s(h).lower() for h in X['mk_hash']]))
    pmap = {k: r for k, r in zip(P['_key'], P.to_dict('records')) if k in hashes}
    tmap = {k: r for k, r in zip(T['_key'], T.to_dict('records')) if k in hashes}
    led, ep, epr = {}, {}, {}
    cks = {c['order_case']['case_key'] for c in chains} | {_s(t_['case_key']) for c in chains for t_ in c['take_cases']}
    for src, dst in ((D['order_ledger'], led), (D['episodes'], ep), (D['episodes_post_regraded'], epr)):
        for r in src[src['case_key'].astype(str).isin(cks)].to_dict('records'):
            dst.setdefault(_s(r['case_key']), []).append(r)
    PU = D['pulls']
    pu = {}
    for r in PU.to_dict('records'):
        k = (_s(r['day']), _int(r['mid']), _s(r['mk_hash']).lower())
        if k in hashes:
            need(_s(r['view']) not in pu.get(k, {}), f'pulls.csv: a (order, view) row repeats for {k} (refused)')
            pu.setdefault(k, {})[_s(r['view'])] = r
    QF1 = dict(levels=D['qf1_levels'], fronts=D['qf1_fronts'])
    # ---- links ----
    links, facts, checks = [], {}, {}
    for ch in chains:
        k = (ch['day'], ch['mid'], ch['order_hash'])
        need(k in pmap and k in tmap, f"chain {ch['chain']}: order {_h12(k[2])} is not in post_fill / post_regrade_orders (refused)")
        p, t = pmap[k], tmap[k]
        ch['ws'] = _int(p['ws_ms'])
        oc = ch['order_case']
        lf = life[(ch['day'], ch['mid'])]
        lf = lf[np.array([_s(x).lower() for x in lf['mk_hash']], object) == ch['order_hash']]
        fills = sorted((dict(hash=_s(r['mk_hash']).lower(), sid=_s(r['sid']), mk_idx=_int(r['mk_idx']), shares=_fin(r['shares']), hi=_int(r['hi_ms']),
                             xlo=_int(r['x_lo_hard_ms']), status=_s(r['status'])) for r in lf.to_dict('records')),
                       key=lambda f: (f['hi'], f['xlo'], f['sid'], f['mk_idx']))
        facts[ch['chain']] = dict(n_fills=len(fills), n_distinct_hashes=len({f['hash'] for f in fills}), filled=_fin(p['cum_shares']))
        fv = D['fv']
        cross = fv[(np.array([_int(x) for x in fv['mid']]) == ch['mid']) & (fv['addr_l'].to_numpy(object) == ch['addr'])
                   & (_sarr(fv['role']) == 'taker') & (_sarr(fv['hash']) == ch['order_hash'])]
        led_rows, ck_ = led.get(oc['case_key'], []), oc['case_key']
        ep_rows = ep.get(ck_, []) + [x for t_ in ch['take_cases'] for x in ep.get(_s(t_['case_key']), [])]
        ls = [link_pre_state(ch, p, t, PO, sib['posting_opportunity']),
              link_post(ch, p, t, led_rows, epr.get(ck_, []), D['census_graded'], T, (int(len(cross)), float(cross['q'].to_numpy(float).sum()))),
              link_queue(ch, p, fills, QF1, QF2, sib['queue_front_v2']),
              link_ending(ch, p, led_rows, pu.get(k, {}), ep_rows, D['census_graded'], PU),
              link_repost(ch, p, led_rows, D['census_graded'], P),
              link_takes(ch, D['events'], ep_rows, ch['ws']),
              link_value(ch, p, D['fv'], D['closes'], D['closes_markets'], D['matched_gap'])]
        fc = dict(life_fills_count_equals_post_fill=bool(len(fills) == _int(p['n_fills'])),
                  life_fills_shares_equal_post_fill=bool(_close(sum(f['shares'] for f in fills), p['cum_shares'])),
                  life_fills_at_the_order_cell=bool((_sarr(lf['y_side']) == _s(p['y_side'])).all()
                                                    and (np.array([_int(x) for x in lf['y_ptick']]) == _int(p['tick'])).all()))
        for L in ls:
            L['reading'] = reading_token(L)
            L['histories_agree'] = bool(L['status'] != 'pending' and int(L['n_histories']) == 1)
            for kk, vv in L['checks'].items():
                checks.setdefault(kk, []).append(vv)
        for kk, vv in fc.items():
            checks.setdefault(kk, []).append(vv)
        links += ls
    stage('links')
    checks = {k: _pred(v) for k, v in sorted(checks.items())}
    asof = _now()
    md = chains_md(mode, chains, links, sib, asof)
    violations = verify(links, md, facts)
    need(not violations, f'rendering check failed (refused before any write): {violations[:4]}')
    claims = make_claims(mode, chains, links, sel, sel_md, checks, IN.info, sib, violations)
    if mode == 'run':
        failed = [k for k, v in claims['predicates'].items() if not v] + ([] if claims['reconciliation_all_hold'] else ['reconciliation'])
        failed += [] if claims['bindings_all_ok'] else ['bindings']
        need(not failed, f'a run refuses on failed predicates {failed} (refused before any write)')
    LK = pd.DataFrame([{c: ('|'.join(L[c]) if c == 'alternatives' else L[c]) for c in LINK_COLS} for L in links], columns=list(LINK_COLS))
    files = [('chains.md', md.encode()), ('chain_links.csv', LK.to_csv(index=False).encode()), ('claims.json', _json_bytes(claims))]
    stage('render')
    srcs = {_rel(f): (_sha(f) if Path(f).is_file() else None) for f in TOOL_FILES + DEP_FILES}
    sib_rec = {nm: dict(status=s['status'], reason=s['reason'], receipt=s['receipt'], absent_optional=s.get('absent_optional', []),
                        notes=s.get('notes', []),
                        files={k: dict(path=_rel(f['path']), sha256=f['sha256'], bytes=f['bytes'], rows_used=f.get('rows_used'))
                               for k, f in s['files'].items()}) for nm, s in sib.items()}
    rec = dict(version=VERSION, mode=mode, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT,
                                                         sha256=_sha(CONTRACT_FILE) if CONTRACT_FILE.is_file() else None),
               module=_rel(Path(__file__)), run=C.RUN, asof_start=asof0, asof_end=_now(), argv=list(sys.argv), git=git, sources=srcs,
               inputs=IN.info, siblings=sib_rec, counts=dict(cnt), out=_rel(out),
               outputs={nm: dict(sha256=hashlib.sha256(b).hexdigest(), bytes=len(b)) for nm, b in files},
               rows=dict(chain_links=len(LK), chains=len(chains)),
               constants=dict(HYP=list(HYP), SCEN=list(SCEN), HORIZONS=list(HORIZONS), CLOCKS=list(CLOCKS), FEATURES=list(FEATURES),
                              TIME_ALTS={str(k): list(v) for k, v in TIME_ALTS.items()}, QPOS={k: list(v) for k, v in QPOS.items()},
                              GRADES=list(GRADES), READINGS=list(READINGS), REL_TOL=REL_TOL, ABS_TOL=ABS_TOL, CPU_CAP_S=CPU_CAP_S),
               cpu_s=dict(by_stage=stages, process_total=round(time.process_time(), 3), cap=CPU_CAP_S), wall_s=round(time.time() - t0, 2),
               vmhwm_mib=round(hwm(), 1), guard_mib=C.GUARD_MIB, mutant=_mut() or None)
    write_out(out, files + [('receipt.json', _json_bytes(rec))])
    log(json.dumps(_clean(dict(out=str(out), mode=mode, readings=claims['readings']['n_by_reading'], predicates=claims['predicates'],
                               reconciliation_all_hold=claims['reconciliation_all_hold'],
                               failed_reconciliation=[k for k, v in claims['reconciliation'].items() if not v['holds']],
                               bindings=claims['bindings'], siblings=claims['siblings'], pending=len(claims['pending']),
                               cpu_s=rec['cpu_s'], vmhwm_mib=rec['vmhwm_mib'], counts=rec['counts'])), indent=1, sort_keys=True))
    return dict(out=out, claims=claims, receipt=rec, links=links, md=md)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['run', '--draft']:
        run(mode='draft')
    elif a == ['run']:
        run(mode='run')
    else:
        raise SystemExit('usage: MT_RUN=v2 chains_v3.py run --draft | run')
