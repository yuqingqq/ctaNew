"""continuation_econ.py - CANDIDATE CONTINUATION ECONOMICS (P-2026-005 MAKER-TRACE; COMPETING-TRANSITION-AND-CONTINUATION v1, section 5; session
aea5bfb9 builder, 2026-11-23). Built EXACTLY to predictfun/reviews/maker_trace_20261122/competing_transition_contract.md section 5 and its acceptance
items (frozen 9909beacc) = review a91f6bb52's finding 4 and next-priority item 5 (reviews/automatic/20261123_touch_response_review/report.md).
USER goal: the top makers' ACTION CHAINS and their logic (post / pull / reprice / take). SAVED EVIDENCE ONLY: no tape, book, market, lifecycle,
queue or FIFO is recomputed; every number is a saved keyed row or a sum over saved keyed rows. DESCRIPTIVE on four consumed days (Oct 24-27): no
intent, no identified private action, no model skill, no profitability claim; filled orders are outcome-selected (unfilled orders are invisible).
LINKS  the candidate response links = touch_transition episodes.parquet response rows with identity candidate_constrained / candidate_scenario
       (the identity rows; 89 declared), each matched to the continuations.csv 'order' row of its (day, mid, cand_hash) (response_keys must be
       bijective); the six chains = chains_v4 chain_links.csv (one order_hash per chain). Owner / grade / unresolved paths are carried from the
       saved cells: a response link's owner is a CANDIDATE (the eventual first filler), never the identified creator; a chain's order is the
       address's own recorded maker order with its post grade and its TIED links.
ORDERS one row per DISTINCT (day, mid, order hash) over both link sets: a repeated link adds no fill, close or take key. Legs = the order's
       fills_valued maker rows of every status (chains_v3.read_groups: the scope markets' row groups, the needed columns, the scope addresses;
       then the order hashes). CLEAN COHORT = status ok and not a self-trade (closes.py LOTS = chains_v3.link_value): only clean legs enter a
       value (views, closes, residual); every other leg is counted by its status.
EXACT  closes.parquet (closes.py FIFO over the top-10 population named by closes_receipt.json) rows keyed by FULL leg keys (sid, mk_idx, role):
       an ENTRY close has a clean leg of the order as its entry leg (the order's completed close); a CLOSING-EARLIER close has it as its close
       leg (earlier inventory: kept apart, never the order's completed close). 'available' = the address is in the closes population and every
       clean leg's attributed close shares (entry + close) equal its fills_valued mq; otherwise 'unavailable' with the reason, and only the
       neutral transition, the residual (econ's per-leg rq) and the views are kept.
WINDOWS (READING, one clock: both ends are settlement-id times) entry closes by closes.parquet hold_ms = close settle_ms - entry settle_ms:
       le10s (<= 10,000 ms), le60s (<= 60,000 ms), leEND (close settle_ms <= END) - cumulative, all bounded by END - and afterEND (close settle_ms
       > END) = the later closes. Fees: closes.py's prorated fee_fill of both legs (net = pnl - fees). RESIDUAL kept apart: rq of the clean legs
       valued to END (s rq (m_end - p), m_end finite) and to resolution (s rq (V - p), st_res ok / tie), its prorated fee fee_fill rq / q.
END    the market END = we, recovered from the saved episodes as state_ms_D0 + rint(1000 time_left_s_D0) (one value per market, else -1 =
       unavailable; touch_transition.state: time_left = (we - x) / 1000); post_fill we_ms of the market's orders must agree (predicate). The
       views' END (econ m_END: the last frame <= we - 2 s) is a different instant and stays the views' own.
TAKES  top-3 addresses only (take_logic_v2 events.parquet: every take record of the address in the market). Per LINK from its anchor A (READING:
       response = creation_hi = resp_hi_ms, touch_transition's t_c; chain = post_hi_ms, the latest possible post): status before_anchor (hi <= A) /
       straddles_anchor (x_lo_hard <= A < hi) / unbounded_lower (x_lo_hard < 0 < hi - A) / inside_END (x_lo_hard > A, hi <= END) / straddles_END
       (x_lo_hard < END < hi) / after_END (x_lo_hard >= END, hi > END); windows 10s / 60s (hi <= A + h) and END over inside_END only; direction
       dir_D0 relative to the RECONSTRUCTED anchor inventory (take_logic_v2; D1000 / pend beside as clock disagreement) and O4 status. A take is
       NEVER an exit by itself: the FIFO entry closes of the order whose close leg is a window take's taker leg are counted beside.
GUARDS (finding 4) the END upper bound on takes and closes; the clean cohort for every value; the response-context join requires the creation
       tick == resp_tick ("this cell restored the touch") and the side: a response link compares the order's fill tick rint(100 p) with the
       episode's resp_tick; a chain compares its cell tick with the resp_tick of every creation chains_v4 matched to a response episode (READING:
       chain_links.csv stores no creation -> episode pair, so the rendered chains.md creation lines, bound by sha256, are parsed and reconciled to
       the link's census counts and to the episode's (resp_lo_ms, resp_hi_ms]); otherwise 'contextual_only' (outside the post window always).
DEDUP  pooled tables run over DISTINCT orders, legs, close keys (day, mid, addr, entry key, close key) and take keys (day, mid, addr, sid); a
       per-link sum is never pooled; an account-level total (closes_markets / matched_gap / address P&L) is never read or attached.
VERIFY independent of the builders, on the raw saved rows (a run refuses on any violation): valued legs clean; counted closes and takes inside the
       END bound and their windows; pooled closes / takes = their distinct-key recomputation; every restoring context tick-equal; per-leg close
       attribution = mq. Mutants MT_CE_MUT (selftest only): no_end_bound, no_close_dedup, no_cohort, no_tick_guard.
OUTPUTS run -> write-once v2/chains/continuation_econ/ (refused if it exists, before any input; a partial directory renamed at the end):
       links_econ.csv (per link), orders_econ.csv (per distinct order), chains_econ.md (the six chains), claims.json (computed predicates and
       numbers only), receipt.json (sha256 of sources / inputs / outputs, CPU by stage, VmHWM). run --dry writes nothing and prints the
       reconciliation.
GUARDS MT_RUN=v2; run refuses unless this tool, its selftest, the contract and chains_v3.py are tracked and unmodified (the freeze), every input is
       bound to its producing receipt (sha256) and every acceptance predicate / verifier check holds; VmHWM self-guard pf_mt_common.GUARD_MIB
       (1,450 MiB, never raised; the 1.1 GB economic parquet is read by row group, columns and candidate keys only); CPU self-guard CPU_CAP_S.
Usage (run_light, MT_RUN=v2): continuation_econ.py run --dry | continuation_econ.py run        selftest: continuation_econ_selftest.py"""
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True                                   # no __pycache__ side effect in the receipt tree

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import chains_v3 as CH3  # noqa: E402   (frozen: imported, never edited)

C = CH3.C
need, OK = C.need, C.STATUS_OK
NAN = float('nan')
_s, _fin, _int, _sarr, _pred, _rel, _close, _sha = CH3._s, CH3._fin, CH3._int, CH3._sarr, CH3._pred, CH3._rel, CH3._close, CH3._sha
_fmt, _sg, _h12, _rel_s, _json_bytes = CH3._fmt, CH3._sg, CH3._h12, CH3._rel_s, CH3._json_bytes
VERSION = 'continuation_econ_v1'
CONTRACT_FILE = C.RECEIPT_V1 / 'competing_transition_contract.md'
CONTRACT_COMMIT = '9909beacc'
REVIEW = dict(commit='a91f6bb52', file='predictfun/reviews/automatic/20261123_touch_response_review/report.md', items='finding 4; next priority 5')
REVIEW_FILE = C.REPO / REVIEW['file']
SELFTEST_FILE = HERE / 'continuation_econ_selftest.py'
TOOL_FILES = (Path(__file__).resolve(), SELFTEST_FILE, CONTRACT_FILE)
FREEZE_FILES = TOOL_FILES + (HERE / 'chains_v3.py',)
DEP_FILES = (HERE / 'chains_v3.py', RESEARCH / 'pf_mt_common.py', REVIEW_FILE)

# ------------------------------------------------------------------ declared constants (the contract's; none fitted) ------------------------------------------------------------------
EXPECT = dict(n_links=89, n_chains=6)                            # contract section 5: the 89 candidate response links and the six chains
LINKED = ('candidate_constrained', 'candidate_scenario')
TOP3G = ('rank1', 'rank2', 'rank3')
BUCKETS = ('le10s', 'le60s', 'leEND', 'afterEND')
HOLD_MS = dict(le10s=10_000, le60s=60_000)
TAKE_W = ('10s', '60s', 'END')
TAKE_H = dict(zip(TAKE_W, (10_000, 60_000, None)))
TAKE_ST = ('before_anchor', 'straddles_anchor', 'unbounded_lower', 'inside_END', 'straddles_END', 'after_END', 'end_unavailable')
DIR_LAB = ('opposing', 'same', 'q0', 'q_unknown', 'side_unknown', 'not_in_panel')   # take_logic_v2.DIR_LAB (its receipt's constants must agree)
O4_ST = ('reduces_all', 'reduces_some', 'reduces_none', 'unavailable')               # take_logic_v2.O4_ST (idem)
RES_VALID = tuple(CH3.RES_VALID)
CTX = ('restoring_cell', 'contextual_only', 'no_response_context')
CHAIN_ITEMS = (('1', 'pre_state'), ('1b', 'transition'), ('2', 'post'), ('3', 'queue'), ('4a', 'ending'), ('4b', 'repost'), ('5', 'takes'), ('6', 'value'))
SCOPES = ('response_all', 'response_top3', 'response_other', 'chains', 'union')
MUT_ENV = 'MT_CE_MUT'
MUTANTS = ('no_end_bound', 'no_close_dedup', 'no_cohort', 'no_tick_guard')            # planted errors: the selftest only
CPU_CAP_S = 12.
REL_TOL, ABS_TOL = CH3.REL_TOL, CH3.ABS_TOL
TICK_TOL = 1e-6
MD_TOL = 0.005 + 1e-9                                            # chains.md renders dollars / shares to 2 decimals

# ---- inputs: key -> (root R = C.RECEIPT / D = C.OUT, path, binding (receipt, section, key)) ----
REC_TT = 'chains/touch_transition/receipt.json'
REC_CH4 = 'chains/chains_v4/receipt.json'
REC_TL2 = 'chains/take_logic_v2/receipt.json'
REC_CL = 'closes/closes_receipt.json'
INPUTS = dict(
    continuations=('R', 'chains/touch_transition/continuations.csv', (REC_TT, 'outputs', 'continuations.csv')),
    episodes=('R', 'chains/touch_transition/episodes.parquet', (REC_TT, 'outputs', 'episodes.parquet')),
    chain_links=('R', 'chains/chains_v4/chain_links.csv', (REC_CH4, 'outputs', 'chain_links.csv')),
    chains_md=('R', 'chains/chains_v4/chains.md', (REC_CH4, 'outputs', 'chains.md')),
    post_regrade_orders=('R', 'chains/correction_v1/post_regrade_orders.csv', ('chains/correction_v1/receipt_post.json', 'outputs', 'post_regrade_orders.csv')),
    post_orders=('R', 'chains/post_fill/orders.csv', ('chains/post_fill/receipt.json', 'outputs', 'orders.csv')),
    events=('R', 'chains/take_logic_v2/events.parquet', (REC_TL2, 'outputs', 'events.parquet')),
    closes=('R', 'closes/closes.parquet', (REC_CL, 'outputs', 'closes.parquet')),
    fills_valued=('D', 'econ/fills_valued.parquet', (REC_TT, 'inputs', 'econ/fills_valued.parquet')))
CO_READ = ('table', 'day', 'mid', 'cand_hash', 'unit_sid', 'cand_addr', 'cand_group', 'n_responses_linked', 'response_keys', 'creation_hi_ms',
           'fv_rows', 'fv_rows_status_ok', 'turnover_mq_sh', 'residual_rq_sh', 'v10s_net', 'v60s_net', 'vend_net', 'res_net', 'takes_status',
           'takes_unbounded_after', 'takes_end', 'takes_10s', 'takes_60s')
EP_READ = ('day', 'mid', 'episode_id', 'side', 'delta_ticks', 'end_type', 'end_reason', 'exposure_ms', 'no_frame_between', 'resp_type', 'resp_lo_ms',
           'resp_hi_ms', 'state_status_D0', 'state_ms_D0', 'time_left_s_D0', 'cand_addr', 'cand_group', 'cand_hash', 'grade', 'identity',
           'identity_reason', 'alt_netting', 'alt_first_other', 'alt_first_addresses', 'pre_state_link', 'po_parity', 'win_unbounded', 'win_contains',
           'resp_tick', 'resp_size_shown')
CL4_READ = ('chain', 'rank', 'episode_id', 'day', 'mid', 'addr', 'order_hash', 'item', 'link', 'status', 'evidence_grade', 'census', 'reading',
            'n_histories', 'alternatives', 'hidden_action', 'summary', 'post_grade', 'alt_netting_state', 'candidate_grades')
PR_READ = ('day', 'mid', 'mk_hash', 'y_side', 'tick', 'post_lo_union_ms', 'post_hi_ms', 'grade')
PF_READ = ('day', 'mid', 'mk_hash', 'ws_ms', 'we_ms', 'n_own_rows', 'mq_sh', 'rq_sh')
EV_READ = ('day', 'mid', 'addr', 'sid', 's', 'dq', 'x_lo_hard_ms', 'hi_ms', 'status', 'o4_status', 'dir_D0', 'dir_D1000', 'dir_pend')
FV_READ = ('day', 'mid', 'sid', 'mk_idx', 'role', 'addr', 'hash', 'y_side', 's', 'q', 'p', 'self_trade', 'settle_ms', 'V', 'st_res', 'st_10s', 'v_10s',
           'st_60s', 'v_60s', 'm_end', 'st_end', 'v_end', 'fee_fill', 'net_res', 'mq', 'rq', 'status')
XC_READ = ('day', 'mid', 'addr', 'entry_sid', 'entry_mk_idx', 'entry_role', 'close_sid', 'close_mk_idx', 'close_role', 'close_settle_ms', 'sh', 'pnl',
           'fee_fill_entry', 'fee_fill_close', 'net', 'hold_ms', 'dust')
ALLOWED_INPUTS = tuple(INPUTS)                                   # no account-level table (closes_markets / matched_gap) is ever read
CLOSE_F = ('n', 'n_dust', 'sh', 'pnl', 'fee', 'net')
LINK_COLS = ('link_id', 'kind', 'chain', 'ep_key', 'day', 'mid', 'side', 'order_hash', 'addr', 'group', 'top3', 'closes_rank', 'owner_basis', 'identity',
             'grade', 'identity_reason', 'pre_state_link', 'alt_netting', 'alt_first_other', 'alt_first_addresses', 'po_parity', 'win_contains',
             'win_unbounded', 'post_grade', 'chain_readings', 'tied_links', 'unresolved_paths', 'resp_type', 'delta_ticks', 'exposure_ms', 'end_reason',
             'no_frame_between', 'resp_lo_ms', 'resp_hi_ms', 'resp_tick', 'resp_size_shown', 'order_side', 'order_tick', 'ctx_status', 'ctx_reasons',
             'ctx_n', 'ctx_detail', 'anchor_ms', 'anchor_basis', 'end_ms', 'exact_link', 'exact_reason', 'n_links_of_order', 'takes_status',
             'takes_subsequent', 'takes_before_anchor', 'takes_straddles_anchor', 'takes_unbounded_lower', 'takes_inside_END', 'takes_straddles_END',
             'takes_after_END', 'takes_end_unavailable', 'takes_excluded_by_END')
LINK_COLS = LINK_COLS + tuple(f'takes_{w}_{f}' for w in TAKE_W for f in ('n', 'abs_dq', 'dir_D0', 'o4', 'clock_disagree', 'fifo_closing', 'fifo_closing_sh')) + (
    'saved_takes_end', 'saved_takes_10s', 'saved_takes_60s', 'saved_takes_unbounded', 'saved_takes_reproduced')
ORDER_COLS = ('day', 'mid', 'order_hash', 'addr', 'group', 'top3', 'closes_rank', 'n_links', 'link_ids', 'n_response_links', 'n_chain_links', 'end_ms',
              'side', 'tick', 'tick_status', 'n_fv_rows', 'n_clean', 'n_excl_status', 'excl_statuses', 'n_excl_self', 'clean_sh', 'n_legs_after_END',
              'first_settle_ms', 'last_settle_ms', 'mq_sh', 'rq_sh', 'fee_fill', 'exact_link', 'exact_reason', 'attributed_sh', 'attribution_ok',
              'fees_decompose_ok', 'n_close_on_excluded_leg') + tuple(f'close_{b}_{f}' for b in BUCKETS for f in CLOSE_F) + (
    'close_leEND_taker_sh', 'close_leEND_taker_net', 'close_leEND_maker_sh', 'close_leEND_maker_net') + tuple(f'earlier_{f}' for f in CLOSE_F) + (
    'shared_close_keys', 'resid_sh', 'resid_val_end', 'resid_n_end_ok', 'resid_val_res', 'resid_n_res_ok', 'resid_fee', 'resid_net_res', 'v10s_n_valid',
    'v10s_gross', 'v10s_net', 'v60s_n_valid', 'v60s_gross', 'v60s_net', 'vend_n_valid', 'vend_gross', 'vend_net', 'common_n', 'res_n_valid', 'res_net',
    'saved_fv_rows', 'saved_mq_sh', 'saved_rq_sh', 'saved_equal_legs', 'saved_equal_views')
OUT_FILES = ('links_econ.csv', 'orders_econ.csv', 'chains_econ.md', 'claims.json', 'receipt.json')
RULES = dict(
    links='identity rows = episodes.parquet end_type response with identity candidate_constrained / candidate_scenario; continuations.csv order '
          'rows keyed (day, mid, cand_hash) with response_keys = those rows; chains = chains_v4 chain_links.csv order_hash',
    cohort='clean = fills_valued status ok and not a self-trade; only clean legs are valued',
    exact='closes.parquet entry / close leg keys (sid, mk_idx, role); available = top-10 closes population and attributed = mq per clean leg',
    windows='entry closes by hold_ms <= 10 s / <= 60 s with close settle_ms <= END (cumulative), leEND = close settle_ms <= END, afterEND = later',
    residual='rq of clean legs: to END s rq (m_end - p), to resolution s rq (V - p) (st_res ok / tie), fee fee_fill rq / q',
    end='END = we = state_ms_D0 + rint(1000 time_left_s_D0) of the market episodes (one value)',
    takes='top-3 only; anchor = resp_hi_ms (response) / post_hi_ms (chain); inside_END = x_lo_hard > A and hi <= END; windows hi <= A + 10 s / 60 s',
    tick_guard='restoring_cell iff the order cell tick == resp_tick and the side agrees (chains: and the creation inside / straddling the post window)',
    dedup='pooled over distinct orders / legs / close keys / take keys; never a per-link sum; no account-level P&L')

CHAIN_HDR = re.compile(r'^## Chain (?P<k>[0-9]+) - rank (?P<r>[0-9]+): episode (?P<e>\S+)\s*$')
CRE_RE = re.compile(r'^  - level (?P<lv>\S+) created in \(ws(?P<a>[+-][0-9]+\.[0-9]{3})s, ws(?P<b>[+-][0-9]+\.[0-9]{3})s\] \(status (?P<st>[^,)]*)'
                    r'(?:, context (?P<cx>[^)]*))?\) - (?P<pos>inside|straddles|before|after) the post window; census pair: (?P<cls>[^;]*); '
                    r'(?:response episode (?P<eid>[0-9]+): (?P<eside>bid|ask) retreat at |(?P<estat>no_response_episode|ambiguous_response_episode))')
CLOSES_RE = re.compile(r'its closes (?P<n>[0-9]+) close\(s\) \(\+(?P<nd>[0-9]+) dust\), (?P<sh>-?[0-9]+\.[0-9]+) sh, pnl (?P<pnl>[+-][0-9]+\.[0-9]+|na) \$, '
                       r'net (?P<net>[+-][0-9]+\.[0-9]+|na) \$')
CENSUS_N = re.compile(r'^(?P<n>[0-9]+) compatible creation\(s\)')
CENSUS_E = re.compile(r'response episodes (?P<ne>[0-9]+)$')
NO_CRE = 'no compatible creation saved'


# ------------------------------------------------------------------ small helpers ------------------------------------------------------------------
def _mut():
    m = os.environ.get(MUT_ENV, '')
    need(m == '' or m in MUTANTS, f'{MUT_ENV}={m!r} is not a declared mutant {MUTANTS} (refused)')
    return m


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _l(x):
    return _s(x).lower()


def cpu_guard(where, c0):
    c = time.process_time() - c0
    need(c < CPU_CAP_S, f'CPU self-guard: {c:.2f} s >= {CPU_CAP_S} s at {where}')
    return c


def _bools(col):
    """a saved boolean column -> numpy bool (missing -> False: closes.py's fill_null(False) for self_trade)."""
    return np.array([bool(CH3._b(x)) for x in col], bool)


def _cnt(vals):
    return dict(sorted(Counter(str(v) for v in vals).items()))


def _cstr(d):
    return ';'.join(f'{k}:{v}' for k, v in d.items())


def read_csv_str(p, key, cols):
    """a saved CSV read as strings (no NA conversion; numbers parsed later by float(), exact round trip)."""
    p = Path(p)
    need(p.is_file(), f'input {key} is missing: {_rel(p)} (refused)')
    have = list(pd.read_csv(p, nrows=0).columns)
    miss = [c for c in cols if c not in have]
    need(not miss, f'input {key}: required columns missing {miss} (refused: interface)')
    return pd.read_csv(p, usecols=list(cols), dtype=str, keep_default_na=False)


def csum(keys, CR):
    """sums over close keys (closes.py DUST: a dust close stays in every sum and is never counted)."""
    n = nd = 0
    sh = pnl = fee = net = 0.
    for k in keys:
        r = CR[k]
        if r['dust']:
            nd += 1
        else:
            n += 1
        sh, pnl, fee, net = sh + r['sh'], pnl + r['pnl'], fee + r['fee'], net + r['net']
    return dict(n=n, n_dust=nd, sh=sh, pnl=pnl, fee=fee, net=net)


def residual(GU):
    """the clean legs' unclosed shares rq valued to END and to resolution, with the prorated fee (closes.py RESID)."""
    if not len(GU):
        return dict(sh=0., val_end=0., n_end_ok=0, val_res=0., n_res_ok=0, fee=0., net_res=0., n=0)
    s, q, p = GU['s'].to_numpy(np.float64), GU['q'].to_numpy(np.float64), GU['p'].to_numpy(np.float64)
    rq = GU['rq'].to_numpy(np.float64)
    rq = np.where(np.isfinite(rq), rq, 0.)
    me, V, ff = GU['m_end'].to_numpy(np.float64), GU['V'].to_numpy(np.float64), GU['fee_fill'].to_numpy(np.float64)
    oke = np.isfinite(me) & np.isfinite(p)
    okr = np.isin(_sarr(GU['st_res']), list(RES_VALID)) & np.isfinite(V) & np.isfinite(p)
    fr = np.where((q > 0) & np.isfinite(ff), ff * rq / np.where(q > 0, q, 1.), 0.)
    ve = np.where(oke, s * rq * (np.where(oke, me, 0.) - p), 0.)
    vr = np.where(okr, s * rq * (np.where(okr, V, 0.) - p), 0.)
    return dict(sh=float(rq.sum()), val_end=float(ve.sum()), n_end_ok=int(oke.sum()), val_res=float(vr.sum()), n_res_ok=int(okr.sum()),
                fee=float(fr.sum()), net_res=float(vr.sum() - fr[okr].sum()), n=int(len(GU)))


def vflat(GU):
    """chains_v3.views (10 s / 60 s / END gross markouts, net = gross - fee_fill of the valid rows, resolution net_res) flattened."""
    V = CH3.views(GU)
    out = {}
    for h, lab in (('10s', 'v10s'), ('60s', 'v60s'), ('end', 'vend')):
        out[f'{lab}_n_valid'], out[f'{lab}_gross'], out[f'{lab}_net'] = V[h]['n_valid'], V[h]['gross'], V[h]['net']
    out['common_n'] = V['common']['n_valid']
    out['res_n_valid'], out['res_net'] = V['res']['n_valid'], V['res']['net']
    return out


# ------------------------------------------------------------------ chains.md (the rendered creation -> episode pairs and the order's closes) ------------------------------------------------------------------
def parse_chains_md(md):
    """chains_v4 chains.md -> per chain: the creation lines of its 1b TRANSITION block (level, interval, position, attached response episode)
    and the VALUE block's 'its closes' figures of the case order. An unparsable creation line is kept with parsed False."""
    out, k, blk = {}, None, False
    for line in md.splitlines():
        m = CHAIN_HDR.match(line)
        if m:
            k, blk = int(m.group('k')), False
            out[k] = dict(creations=[], closes=None, header=m.groupdict())
            continue
        if k is None:
            continue
        if line.startswith('- **'):
            blk = line.startswith('- **1b TRANSITION**')
            continue
        if blk and line.startswith('  - level '):
            mm = CRE_RE.match(line)
            out[k]['creations'].append(dict(parsed=mm is not None, **(mm.groupdict() if mm else {})))
        if out[k]['closes'] is None and 'own maker rows' in line:
            mc = CLOSES_RE.search(line)
            if mc:
                out[k]['closes'] = mc.groupdict()
    return out


# ------------------------------------------------------------------ links ------------------------------------------------------------------
def market_ends(EP):
    """{(day, mid): END ms}: we = state_ms_D0 + rint(1000 time_left_s_D0) over the market's episodes with state_status_D0 ok (-1 when absent or
    not one value)."""
    d_ = _sarr(EP['day'])
    m_ = np.array([_int(x) for x in EP['mid']], np.int64)
    x = np.array([_int(v) for v in EP['state_ms_D0']], np.int64)
    tl = EP['time_left_s_D0'].to_numpy(np.float64)
    ok = (_sarr(EP['state_status_D0']) == 'ok') & (x >= 0) & np.isfinite(tl)
    e = np.full(len(EP), -1, np.int64)
    e[ok] = x[ok] + np.rint(tl[ok] * 1000.).astype(np.int64)
    vals = {}
    for i in np.flatnonzero(ok).tolist():
        vals.setdefault((d_[i], int(m_[i])), set()).add(int(e[i]))
    return {k: (next(iter(v)) if len(v) == 1 else -1) for k, v in vals.items()}, {k: len(v) for k, v in vals.items()}


def response_links(EP):
    et, idn = _sarr(EP['end_type']), _sarr(EP['identity'])
    R = EP[(et == 'response') & np.isin(idn, list(LINKED))]
    out = []
    for r in R.to_dict('records'):
        d, m, e = _s(r['day']), _int(r['mid']), _int(r['episode_id'])
        out.append(dict(kind='response', link_id=f'resp:{d}_{m}_{e}', ep_key=f'{d}_{m}_{e}', chain='', day=d, mid=m, eid=e,
                        order=(d, m, _l(r['cand_hash'])), addr=_l(r['cand_addr']), group=_s(r['cand_group']), side=_s(r['side']),
                        resp_tick=_int(r['resp_tick']), anchor=_int(r['resp_hi_ms']), anchor_basis='creation_hi = resp_hi_ms', ep=r))
    return out


def continuations_check(CO, links):
    """continuations.csv 'order' rows <-> the identity rows: one row per (day, mid, cand_hash), never split, its response_keys = its links."""
    rows, dup = {}, 0
    for r in CO.to_dict('records'):
        k = (_s(r['day']), _int(r['mid']), _l(r['cand_hash']))
        dup += int(k in rows)
        rows[k] = r
    by = {}
    for L in links:
        by.setdefault(L['order'], []).append(L)
    keys = {k: {x for x in _s(r['response_keys']).split(';') if x} for k, r in rows.items()}
    chk = dict(
        continuations_order_rows_unique=[dup == 0],
        continuations_not_split=[_s(r['unit_sid']) == '' for r in rows.values()],
        continuations_orders_equal_identity_orders=[set(rows) == set(by)],
        continuations_response_keys_equal_identity_rows=[keys.get(k) == {L['ep_key'] for L in v} for k, v in by.items()],
        continuations_n_linked_equal=[k in rows and _int(rows[k]['n_responses_linked']) == len(v) for k, v in by.items()],
        continuations_addr_group_equal=[k in rows and _l(rows[k]['cand_addr']) == L['addr'] and _s(rows[k]['cand_group']) == L['group']
                                        for k, v in by.items() for L in v])
    return rows, chk


def chain_links(CL4, PR, PF, md, EPK, mut):
    """the six chains: order, cell, post window / grade (post_regrade_orders), ws / we (post_fill), the carried link readings and the guarded
    response contexts of the 1b TRANSITION creations."""
    P = parse_chains_md(md)
    by = {}
    for r in CL4.to_dict('records'):
        by.setdefault(_int(r['chain']), []).append(r)
    prk = {(_s(r['day']), _int(r['mid']), _l(r['mk_hash'])): r for r in PR.to_dict('records')}
    pfk = {(_s(r['day']), _int(r['mid']), _l(r['mk_hash'])): r for r in PF.to_dict('records')}
    out = []
    for k in sorted(by):
        rows = by[k]

        def one(f):
            return sorted({f(x) for x in rows})
        ds, ms, hs, ads = one(lambda x: _s(x['day'])), one(lambda x: _int(x['mid'])), one(lambda x: _l(x['order_hash'])), one(lambda x: _l(x['addr']))
        rks, eps = one(lambda x: _int(x['rank'])), one(lambda x: _s(x['episode_id']))
        need(all(len(v) == 1 for v in (ds, ms, hs, ads, rks, eps)), f'chain {k}: its link rows disagree on the order (refused)')
        d, m, h, a, rk = ds[0], ms[0], hs[0], ads[0], rks[0]
        pr, pf = prk.get((d, m, h)), pfk.get((d, m, h))
        need(pr is not None and pf is not None, f'chain {k}: order {h[:12]} is not in post_regrade_orders.csv / post_fill orders.csv (refused)')
        items = {(_s(x['item']), _s(x['link'])): x for x in rows}
        side, tick, ws = _s(pr['y_side']), _int(pr['tick']), _int(pf['ws_ms'])
        tr = items.get(('1b', 'transition'))
        cen = _s(tr['census']) if tr is not None else ''
        mn, me = CENSUS_N.match(cen), CENSUS_E.search(cen)
        n_cen = int(mn.group('n')) if mn else (0 if cen == NO_CRE else -1)
        n_cen_r = int(me.group('ne')) if me else (0 if cen == NO_CRE else -1)
        pm = P.get(k, dict(creations=[], closes=None, header=None))
        ctxs = []
        for c in pm['creations']:
            if not c['parsed'] or c.get('eid') is None:
                continue
            eid = int(c['eid'])
            a_ms, b_ms = ws + int(round(float(c['a']) * 1000.)), ws + int(round(float(c['b']) * 1000.))
            er = EPK.get((d, m, eid))
            rs = []
            if er is None:
                rs.append('episode_missing')
            else:
                if _s(er['end_type']) != 'response':
                    rs.append('not_a_response')
                if _s(er['side']) != side or c['eside'] != side:
                    rs.append('side_mismatch')
                if (_int(er['resp_lo_ms']), _int(er['resp_hi_ms'])) != (a_ms, b_ms):
                    rs.append('interval_mismatch')
                if _int(er['resp_tick']) != tick and mut != 'no_tick_guard':     # the guard (the planted error drops it: chains_v4's join)
                    rs.append('tick_mismatch')
            if c['pos'] not in ('inside', 'straddles'):
                rs.append('outside_post_window')
            ctxs.append(dict(level=c['lv'], eid=eid, pos=c['pos'], a=a_ms, b=b_ms, cell_tick=tick,
                             resp_tick=_int(er['resp_tick']) if er is not None else -1, resp_type=_s(er['resp_type']) if er is not None else '',
                             status=CTX[0] if not rs else CTX[1], reasons=rs))
        out.append(dict(kind='chain', link_id=f'chain:{k}', chain=k, ep_key='', day=d, mid=m, eid=-1, order=(d, m, h), addr=a, group=f'rank{rk}',
                        rank=rk, episode=eps[0], side=side, cell_tick=tick, anchor=_int(pr['post_hi_ms']), anchor_basis='post_hi_ms (latest possible post)',
                        post_lo=_int(pr['post_lo_union_ms']), post_grade=_s(pr['grade']), ws=ws, we_pf=_int(pf['we_ms']), pf=pf, rows=rows, items=items,
                        contexts=ctxs, md_closes=pm['closes'], md_header=pm['header'],
                        parse=dict(n_lines=len(pm['creations']), n_parsed=sum(1 for c in pm['creations'] if c['parsed']),
                                   n_resp=sum(1 for c in pm['creations'] if c['parsed'] and c.get('eid') is not None), n_census=n_cen, n_census_resp=n_cen_r)))
    return out


def make_orders(links):
    O = {}
    for L in links:
        o = O.get(L['order'])
        if o is None:
            d, m, h = L['order']
            o = O[L['order']] = dict(key=L['order'], day=d, mid=m, hash=h, addrs=[], groups=[], links=[], kinds=[])
        o['addrs'].append(L['addr'])
        o['groups'].append(L['group'])
        o['links'].append(L['link_id'])
        o['kinds'].append(L['kind'])
    for o in O.values():
        o['addr'], o['group'] = o['addrs'][0], o['groups'][0]
    return O


# ------------------------------------------------------------------ orders: legs, exact closes, residual, views ------------------------------------------------------------------
def attach_legs(O, FV, mut):
    """per order: its fills_valued maker rows (every status), the clean cohort (the planted error no_cohort values every leg), leg keys,
    the cell tick rint(100 p) and side."""
    idx = {}
    if len(FV):
        role = _sarr(FV['role'])
        hk = list(zip(_sarr(FV['day']), [_int(x) for x in FV['mid']], [_l(x) for x in FV['hash']]))
        for i in np.flatnonzero(role == 'maker').tolist():
            idx.setdefault(hk[i], []).append(i)
    for k, o in O.items():
        ii = idx.get(k, [])
        G = (FV.iloc[ii] if ii else FV.iloc[0:0]).reset_index(drop=True)
        st = _sarr(G['status'])
        sf = _bools(G['self_trade'])
        clean = (st == OK) & ~sf
        use = np.ones(len(G), bool) if mut == 'no_cohort' else clean
        t100 = 100. * G['p'].to_numpy(np.float64)
        tk = np.rint(t100)
        on = np.isfinite(t100) & (np.abs(t100 - tk) <= TICK_TOL)
        ticks = sorted({int(x) for x in tk[on]})
        sides = sorted(set(_sarr(G['y_side']).tolist()))
        tst = 'no_legs' if not len(G) else ('off_grid' if not bool(on.all()) else ('ok' if len(ticks) == 1 else 'not_unique'))
        o.update(G=G, keys=[(_s(a), _int(b), 'maker') for a, b in zip(G['sid'], G['mk_idx'])], clean=clean, use=use, n_rows=int(len(G)),
                 n_clean=int(clean.sum()), n_excl_status=int((st != OK).sum()), n_excl_self=int(((st == OK) & sf).sum()),
                 excl_statuses=_cnt(st[st != OK].tolist()), tick=ticks[0] if tst == 'ok' else -1, tick_status=tst,
                 side=sides[0] if len(sides) == 1 else '', fv_addrs=sorted({_l(x) for x in G['addr']}))


def close_rows(XC):
    """closes.parquet rows -> {close key: values} and {(day, mid, addr): [close keys]}; a close key twice refuses (a duplicated close)."""
    CR, grp = {}, {}
    if not len(XC):
        return CR, grp
    d_ = _sarr(XC['day'])
    m_ = [_int(x) for x in XC['mid']]
    a_ = [_l(x) for x in XC['addr']]
    es, em, er = _sarr(XC['entry_sid']), [_int(x) for x in XC['entry_mk_idx']], _sarr(XC['entry_role'])
    cs, cm, cr = _sarr(XC['close_sid']), [_int(x) for x in XC['close_mk_idx']], _sarr(XC['close_role'])
    sh, pnl, net = (XC[c].to_numpy(np.float64) for c in ('sh', 'pnl', 'net'))
    ffe, ffc = XC['fee_fill_entry'].to_numpy(np.float64), XC['fee_fill_close'].to_numpy(np.float64)
    cset = np.array([_int(x) for x in XC['close_settle_ms']], np.int64)
    hold = np.array([_int(x) for x in XC['hold_ms']], np.int64)
    dust = _bools(XC['dust'])
    for i in range(len(XC)):
        g = (d_[i], m_[i], a_[i])
        ek, ck = (es[i], em[i], er[i]), (cs[i], cm[i], cr[i])
        k = g + ek + ck
        need(k not in CR, f'closes.parquet: a close key twice ({d_[i]} {m_[i]} entry {es[i][:13]}/{em[i]} close {cs[i][:13]}/{cm[i]}) '
                          '(refused: a duplicated close)')
        CR[k] = dict(g=g, ek=ek, ck=ck, sh=float(sh[i]), pnl=float(pnl[i]), ffe=float(ffe[i]), ffc=float(ffc[i]), fee=float(ffe[i] + ffc[i]),
                     net=float(net[i]), cset=int(cset[i]), hold=int(hold[i]), dust=bool(dust[i]), close_role=cr[i], close_sid=cs[i])
        grp.setdefault(g, []).append(k)
    return CR, grp


def order_econ(o, CR, grp, top10, mut):
    """the order's valued cohort, residual, views and (top-10 population) its exact entry / closing-earlier closes by window."""
    G, use = o['G'], o['use']
    GU = G[use].reset_index(drop=True)
    o['valued'] = [(o['day'], o['mid']) + kk for kk, u in zip(o['keys'], use) if u]
    E_ = o['end']
    st_ms = G['settle_ms'].to_numpy(np.int64)[use] if len(G) else np.zeros(0, np.int64)
    o['n_legs_after_END'] = int((st_ms > E_).sum()) if E_ >= 0 else -1
    o['first_settle_ms'] = int(st_ms.min()) if len(st_ms) else -1
    o['last_settle_ms'] = int(st_ms.max()) if len(st_ms) else -1
    o['clean_sh'] = float(np.nansum(G['q'].to_numpy(np.float64)[o['clean']])) if len(G) else 0.
    for c, f in (('mq', 'mq_sh'), ('rq', 'rq_sh'), ('fee_fill', 'fee_fill')):
        o[f] = float(np.nansum(GU[c].to_numpy(np.float64))) if len(GU) else 0.
    o['resid'] = residual(GU)
    o['views'] = vflat(GU)
    o['GU'] = GU
    o['closes_rank'] = int(top10.get(o['addr'], -1))
    o.update(close_keys={b: [] for b in BUCKETS}, earlier_keys=[], taker_close={}, attributed_sh=NAN, attribution_ok=None, fees_ok=None,
             n_close_on_excl=0, att_legs=[])
    if o['closes_rank'] < 0:
        o['exact'], o['exact_reason'] = 'unavailable', 'address_outside_closes_population'
        return
    if not use.any():
        o['exact'], o['exact_reason'] = 'unavailable', 'no_clean_legs'
        return
    Kc = {kk for kk, u in zip(o['keys'], use) if u}
    Kx = {kk for kk, u in zip(o['keys'], use) if not u}
    att = Counter()
    for k in grp.get((o['day'], o['mid'], o['addr']), []):
        r = CR[k]
        o['n_close_on_excl'] += int(r['ek'] in Kx or r['ck'] in Kx)
        if r['ek'] in Kc:
            att[r['ek']] += r['sh']
            inend = True if mut == 'no_end_bound' else (E_ >= 0 and r['cset'] <= E_)
            if inend:
                o['close_keys']['leEND'].append(k)
                for b, hm in HOLD_MS.items():
                    if r['hold'] <= hm:
                        o['close_keys'][b].append(k)
            else:
                o['close_keys']['afterEND'].append(k)
            if r['close_role'] == 'taker':
                o['taker_close'].setdefault(r['close_sid'], []).append(k)
        if r['ck'] in Kc:
            att[r['ck']] += r['sh']
            o['earlier_keys'].append(k)
    mq = GU['mq'].to_numpy(np.float64)
    legs = [kk for kk, u in zip(o['keys'], use) if u]
    oks = [bool(np.isfinite(m_) and abs(att.get(kk, 0.) - m_) <= ABS_TOL + REL_TOL * max(1., abs(m_))) for kk, m_ in zip(legs, mq)]
    o['att_legs'] = oks
    o['attributed_sh'] = float(sum(att.values()))
    o['attribution_ok'] = bool(all(oks))
    q, ff, rq = GU['q'].to_numpy(np.float64), GU['fee_fill'].to_numpy(np.float64), GU['rq'].to_numpy(np.float64)
    f_res = float(np.where(q > 0, ff * np.where(np.isfinite(rq), rq, 0.) / np.where(q > 0, q, 1.), 0.).sum())
    f_e = sum(CR[k]['ffe'] for k in o['close_keys']['leEND'] + o['close_keys']['afterEND'])
    f_c = sum(CR[k]['ffc'] for k in o['earlier_keys'])
    o['fees_ok'] = bool(_close(f_e + f_c + f_res, float(np.nansum(ff))))
    if o['attribution_ok']:
        o['exact'], o['exact_reason'] = 'available', ''
    else:
        o['exact'], o['exact_reason'] = 'unavailable', 'attribution_differs_from_mq'


# ------------------------------------------------------------------ takes (per link) ------------------------------------------------------------------
def link_takes(L, EVg, END, o, mut):
    """the take records of the link's address in its market against the link's anchor and the market END."""
    A = int(L['anchor'])
    xl, hi = EVg['x_lo_hard_ms'].to_numpy(np.int64), EVg['hi_ms'].to_numpy(np.int64)
    c0 = [(xl < 0) & (hi > A), xl < 0, (xl <= A) & (hi > A), xl <= A]
    s0 = ['unbounded_lower', 'before_anchor', 'straddles_anchor', 'before_anchor']
    if END >= 0:
        st = np.select(c0 + [hi <= END, xl >= END], s0 + ['inside_END', 'after_END'], 'straddles_END')
    else:
        st = np.select(c0, s0, 'end_unavailable')
    st = np.asarray(st, dtype=object)
    sub = xl > A
    inside = sub.copy() if mut == 'no_end_bound' else (st == 'inside_END')
    sids = _sarr(EVg['sid'])
    keys = [(L['day'], L['mid'], L['addr'], x) for x in sids]
    dq = np.abs(EVg['dq'].to_numpy(np.float64))
    dq = np.where(np.isfinite(dq), dq, 0.)
    d0, d1, dp, o4 = _sarr(EVg['dir_D0']), _sarr(EVg['dir_D1000']), _sarr(EVg['dir_pend']), _sarr(EVg['o4_status'])
    T = dict(status=_cnt(st.tolist()), n_subsequent=int(sub.sum()), keys={}, fifo_keys={},
             sub_status={keys[i]: str(st[i]) for i in np.flatnonzero(sub).tolist()},
             attr_of={keys[i]: (str(d0[i]), float(dq[i])) for i in np.flatnonzero(inside).tolist()},
             raw=dict(n_end=int(sub.sum()), n_10s=int((sub & (hi <= A + 10_000)).sum()), n_60s=int((sub & (hi <= A + 60_000)).sum()),
                      n_unbounded=int(((xl < 0) & (hi > A)).sum())))
    for w in TAKE_W:
        msk = inside & (hi <= A + TAKE_H[w]) if TAKE_H[w] is not None else inside
        ii = np.flatnonzero(msk).tolist()
        fk = [ck for i in ii for ck in o['taker_close'].get(sids[i], [])]
        T[w] = dict(n=len(ii), abs_dq=float(dq[ii].sum()) if ii else 0., dir_D0=_cnt(d0[ii].tolist()), o4=_cnt(o4[ii].tolist()),
                    clock_disagree=int(sum(1 for i in ii if not (d0[i] == d1[i] == dp[i]))),
                    fifo_closing=int(sum(1 for i in ii if sids[i] in o['taker_close'])), fifo_closing_sh=float(sum(o['_CR'][ck]['sh'] for ck in fk)))
        T['keys'][w] = [keys[i] for i in ii]
        T['fifo_keys'][w] = fk
    return T


# ------------------------------------------------------------------ pooled (distinct keys) ------------------------------------------------------------------
def scope_links(sc, links):
    if sc == 'union':
        return list(links)
    if sc == 'chains':
        return [L for L in links if L['kind'] == 'chain']
    R = [L for L in links if L['kind'] == 'response']
    if sc == 'response_top3':
        return [L for L in R if L['group'] in TOP3G]
    if sc == 'response_other':
        return [L for L in R if L['group'] not in TOP3G]
    return R


def pooled(sc, links, O, CR, mut):
    """one scope over its DISTINCT orders (planted error no_close_dedup: its closes pooled per link)."""
    SL = scope_links(sc, links)
    ks = list(dict.fromkeys(L['order'] for L in SL))
    seq = [L['order'] for L in SL] if mut == 'no_close_dedup' else ks
    av = [k for k in ks if O[k]['exact'] == 'available']
    un = [k for k in ks if O[k]['exact'] != 'available']
    ck, ek_ = {b: [] for b in BUCKETS}, []
    for k in seq:
        if O[k]['exact'] != 'available':
            continue
        for b in BUCKETS:
            ck[b] += O[k]['close_keys'][b]
        ek_ += O[k]['earlier_keys']
    if mut != 'no_close_dedup':
        ck = {b: list(dict.fromkeys(v)) for b, v in ck.items()}
        ek_ = list(dict.fromkeys(ek_))
    ent = set(ck['leEND']) | set(ck['afterEND'])

    def rsum(kk):
        out = dict(sh=0., val_end=0., n_end_ok=0, val_res=0., n_res_ok=0, fee=0., net_res=0., n=0)
        for k in kk:
            for f in out:
                out[f] += O[k]['resid'][f]
        return out
    GU = [O[k]['GU'] for k in ks if len(O[k]['GU'])]
    V = vflat(pd.concat(GU, ignore_index=True)) if GU else vflat(next(iter(O.values()))['GU'].iloc[0:0])
    ev = [L for L in SL if L['takes_status'] == 'evaluated']
    win = {w: [] for w in TAKE_W}
    fifo = {w: [] for w in TAKE_W}
    subs, attr = {}, {}
    for L in ev:
        for w in TAKE_W:
            win[w] += L['T']['keys'][w]
            fifo[w] += L['T']['fifo_keys'][w]
        subs.update(L['T']['sub_status'])
        attr.update(L['T']['attr_of'])                            # a take record's direction / size do not depend on the link
    tk = {}
    for w in TAKE_W:
        dk = list(dict.fromkeys(win[w]))
        fk = list(dict.fromkeys(fifo[w]))
        tk[w] = dict(n_distinct=len(dk), order_sum=len(win[w]), overlap=len(win[w]) - len(dk), keys=dk, n_fifo_close_keys=len(fk),
                     fifo_closing_sh=float(sum(CR[k]['sh'] for k in fk)), dir_D0=_cnt(attr[k][0] for k in dk),
                     abs_dq=float(sum(attr[k][1] for k in dk)))
    return dict(
        n_links=len(SL), n_response_links=sum(1 for L in SL if L['kind'] == 'response'), n_chain_links=sum(1 for L in SL if L['kind'] == 'chain'),
        n_orders=len(ks), n_orders_top3=sum(1 for k in ks if O[k]['group'] in TOP3G), n_available=len(av), n_unavailable=len(un),
        unavailable_reasons=_cnt(O[k]['exact_reason'] for k in un),
        legs=dict(n_rows=sum(O[k]['n_rows'] for k in ks), n_clean=sum(O[k]['n_clean'] for k in ks), n_excl_status=sum(O[k]['n_excl_status'] for k in ks),
                  n_excl_self=sum(O[k]['n_excl_self'] for k in ks), clean_sh=float(sum(O[k]['clean_sh'] for k in ks))),
        closes={b: csum(ck[b], CR) for b in BUCKETS}, close_keys=ck,
        closes_leEND_by_close_role={r_: csum([k for k in ck['leEND'] if CR[k]['close_role'] == r_], CR) for r_ in ('taker', 'maker')},
        earlier=csum(ek_, CR), earlier_keys=ek_, shared_close_keys=len(ent & set(ek_)), touching_distinct=len(ent | set(ek_)),
        residual_available=rsum(av), residual_unavailable=rsum(un), mq_sh_unavailable=float(sum(O[k]['mq_sh'] for k in un)),
        views=V, takes=dict(n_links_evaluated=len(ev), n_links_not_evaluated=len(SL) - len(ev), subsequent_distinct=len(subs),
                            end_status_distinct=_cnt(subs.values()), windows=tk))


# ------------------------------------------------------------------ verification (raw rows, independent of the builders) ------------------------------------------------------------------
def verify(links, O, P, raw):
    """properties stated on the records against the raw saved rows (the declared rules re-applied, never the builder functions)."""
    viol, chk = [], {}
    leg = raw['leg']                                              # (day, mid, sid, mk_idx, role) -> (status, self, mq)
    bad = [k for o in O.values() for k in o['valued'] if not (k in leg and leg[k][0] == OK and not leg[k][1])]
    chk['valued_legs_clean'] = not bad
    if bad:
        viol.append(f'valued_legs_clean: {len(bad)} valued leg(s) not status ok / non-self, e.g. {bad[0][2][:13]}/{bad[0][3]}')
    CRr, END = raw['close'], raw['end']
    badc = []
    for o in O.values():
        E_ = END.get((o['day'], o['mid']), -1)
        for b, ks in o['close_keys'].items():
            for k in ks:
                r = CRr[k]
                ok = E_ >= 0 and (r['cset'] > E_ if b == 'afterEND' else r['cset'] <= E_)
                if b in HOLD_MS:
                    ok = ok and r['hold'] <= HOLD_MS[b]
                if not ok:
                    badc.append((o['hash'][:12], b))
    chk['closes_within_end_bound'] = not badc
    if badc:
        viol.append(f'closes_within_end_bound: {len(badc)} counted close(s) outside their window / END, e.g. {badc[0]}')
    TK = raw['take']
    badt = []
    for L in links:
        if L['takes_status'] != 'evaluated':
            continue
        A, E_ = int(L['anchor']), END.get((L['day'], L['mid']), -1)
        for w, ks in L['T']['keys'].items():
            for k in ks:
                xl, hi = TK[k]
                ok = xl > A and E_ >= 0 and hi <= E_ and (TAKE_H[w] is None or hi <= A + TAKE_H[w])
                if not ok:
                    badt.append((L['link_id'], w, k[3][:13]))
    chk['takes_within_end_bound'] = not badt
    if badt:
        viol.append(f'takes_within_end_bound: {len(badt)} counted take(s) outside the anchor / END / window, e.g. {badt[0]}')
    # pooled closes / takes = the distinct-key recomputation per scope (the declared rules on the raw rows)
    badp = []
    for sc, Pv in P.items():
        SL = scope_links(sc, links)
        ords = {L['order'] for L in SL}
        clav = {(d, m) + kk for (d, m, h) in ords if raw['avail'].get((d, m, h)) for kk in raw['clean_keys'].get((d, m, h), [])}
        exp = {b: [0, 0., 0.] for b in BUCKETS}
        for lk in clav:                                           # every raw close whose ENTRY leg is a clean leg of an available scope order
            for k in raw['by_entry'].get(lk, []):
                r = CRr[k]
                E_ = END.get((r['g'][0], r['g'][1]), -1)
                ine = E_ >= 0 and r['cset'] <= E_
                for b in BUCKETS:
                    hit = (not ine) if b == 'afterEND' else (ine and (b == 'leEND' or r['hold'] <= HOLD_MS[b]))
                    if hit:
                        exp[b][0] += int(not r['dust'])
                        exp[b][1] += r['sh']
                        exp[b][2] += r['net']
        for b in BUCKETS:
            got = Pv['closes'][b]
            if not (got['n'] == exp[b][0] and _close(got['sh'], exp[b][1]) and _close(got['net'], exp[b][2])):
                badp.append((sc, b, got['n'], exp[b][0]))
        ev = [L for L in SL if L['takes_status'] == 'evaluated']
        for w in TAKE_W:
            want = set()
            for L in ev:
                A, E_ = int(L['anchor']), END.get((L['day'], L['mid']), -1)
                for k in raw['takes_of'].get((L['day'], L['mid'], L['addr']), []):
                    xl, hi = TK[k]
                    if xl > A and E_ >= 0 and hi <= E_ and (TAKE_H[w] is None or hi <= A + TAKE_H[w]):
                        want.add(k)
            if Pv['takes']['windows'][w]['n_distinct'] != len(want) or set(Pv['takes']['windows'][w]['keys']) != want:
                badp.append((sc, f'takes_{w}', Pv['takes']['windows'][w]['n_distinct'], len(want)))
    chk['pooled_closes_distinct'] = not [x for x in badp if not str(x[1]).startswith('takes_')]
    chk['pooled_takes_distinct'] = not [x for x in badp if str(x[1]).startswith('takes_')]
    if badp:
        viol.append(f'pooled_distinct: {len(badp)} pooled cell(s) differ from the distinct-key recomputation, e.g. {badp[0]}')
    badk = []
    for L in links:
        if L['kind'] == 'response' and L['ctx_status'] == CTX[0]:
            t = raw['order_ticks'].get(L['order'], set())
            e = raw['ep'].get((L['day'], L['mid'], L['eid']))
            if not (e is not None and t == {e['resp_tick']} and raw['order_sides'].get(L['order']) == {e['side']}):
                badk.append(L['link_id'])
        if L['kind'] == 'chain':
            for c in L['contexts']:
                if c['status'] != CTX[0]:
                    continue
                e = raw['ep'].get((L['day'], L['mid'], c['eid']))
                cell = raw['cells'].get(L['order'])
                if not (e is not None and cell is not None and e['resp_tick'] == cell[1] and e['side'] == cell[0]):
                    badk.append(f"{L['link_id']}:e{c['eid']}")
    chk['restoring_context_tick_equal'] = not badk
    if badk:
        viol.append(f'restoring_context_tick_equal: {len(badk)} restoring context(s) whose cell tick differs from resp_tick, e.g. {badk[0]}')
    bada = []
    for o in O.values():
        if o['exact'] != 'available':
            continue
        cks = set(raw['clean_keys'].get(o['key'], []))
        att = Counter()
        for k in raw['by_g'].get((o['day'], o['mid'], o['addr']), []):
            r = CRr[k]
            for kk in (r['ek'], r['ck']):
                if kk in cks:
                    att[kk] += r['sh']
        if not cks:
            bada.append((o['hash'][:12], 'no clean leg'))
        for kk in cks:
            m_ = leg[(o['day'], o['mid']) + kk][2]
            if not (np.isfinite(m_) and abs(att.get(kk, 0.) - m_) <= ABS_TOL + REL_TOL * max(1., abs(m_))):
                bada.append((o['hash'][:12], kk[0][:13]))
    chk['attribution_equals_mq'] = not bada
    if bada:
        viol.append(f'attribution_equals_mq: {len(bada)} clean leg(s) whose attributed close shares differ from mq, e.g. {bada[0]}')
    return viol, chk


def raw_tables(FV, CR, EP, EV, links, O, top10):
    """the raw saved rows the verifier re-reads (dictionaries keyed by the declared keys)."""
    leg, ck, ticks, sides = {}, {}, {}, {}
    if len(FV):
        d_, m_, sid = _sarr(FV['day']), [_int(x) for x in FV['mid']], _sarr(FV['sid'])
        mk, ro, st = [_int(x) for x in FV['mk_idx']], _sarr(FV['role']), _sarr(FV['status'])
        sf, mq, hs = _bools(FV['self_trade']), FV['mq'].to_numpy(np.float64), [_l(x) for x in FV['hash']]
        p, ys = FV['p'].to_numpy(np.float64), _sarr(FV['y_side'])
        for i in range(len(FV)):
            leg[(d_[i], m_[i], sid[i], mk[i], ro[i])] = (st[i], bool(sf[i]), float(mq[i]))
            if ro[i] != 'maker':
                continue
            ok_ = (d_[i], m_[i], hs[i])
            if ok_ not in O:
                continue
            ticks.setdefault(ok_, set()).add(int(np.rint(100. * p[i])) if np.isfinite(p[i]) else -999)
            sides.setdefault(ok_, set()).add(ys[i])
            if st[i] == OK and not sf[i]:
                ck.setdefault(ok_, []).append((sid[i], mk[i], 'maker'))
    epk = {}
    for r in EP.to_dict('records'):
        epk[(_s(r['day']), _int(r['mid']), _int(r['episode_id']))] = dict(resp_tick=_int(r['resp_tick']), side=_s(r['side']))
    tk, tof = {}, {}
    for r in EV.to_dict('records'):
        k = (_s(r['day']), _int(r['mid']), _l(r['addr']), _s(r['sid']))
        tk[k] = (_int(r['x_lo_hard_ms']), _int(r['hi_ms']))
        tof.setdefault(k[:3], []).append(k)
    cells = {L['order']: (L['side'], L['cell_tick']) for L in links if L['kind'] == 'chain'}
    by_g, by_entry = {}, {}
    for k, r in CR.items():
        by_g.setdefault(r['g'], []).append(k)
        by_entry.setdefault((r['g'][0], r['g'][1]) + r['ek'], []).append(k)
    avail = {}
    for k, o in O.items():                                       # the population and attribution rule re-applied on raw rows
        cks = set(ck.get(k, []))
        a_ok = top10.get(o['addr'], -1) >= 0 and bool(cks)
        if a_ok:
            att = Counter()
            for kk in by_g.get((o['day'], o['mid'], o['addr']), []):
                r = CR[kk]
                for x in (r['ek'], r['ck']):
                    if x in cks:
                        att[x] += r['sh']
            mqs = [leg[(o['day'], o['mid']) + x][2] for x in cks]
            a_ok = all(np.isfinite(m_) and abs(att.get(x, 0.) - m_) <= ABS_TOL + REL_TOL * max(1., abs(m_)) for x, m_ in zip(cks, mqs))
        avail[k] = bool(a_ok)
    return dict(leg=leg, clean_keys=ck, order_ticks=ticks, order_sides=sides, ep=epk, take=tk, takes_of=tof, cells=cells, close=CR, avail=avail,
                by_g=by_g, by_entry=by_entry)


# ------------------------------------------------------------------ rendering ------------------------------------------------------------------
def _cl(lab, s):
    return f"{lab} {s['n']} (+{s['n_dust']} dust), {_fmt(s['sh'])} sh, pnl {_sg(s['pnl'])} $, fees {_fmt(s['fee'], 3)} $, net {_sg(s['net'])} $"


def chains_md(mode, asof, links, O, P, claims):
    CHn = [L for L in links if L['kind'] == 'chain']
    pc = P['chains']
    out = [f"# Candidate continuation economics - the six chains (competing_transition_contract.md section 5) - {mode.upper()}", '',
           f"As of {asof}. Saved evidence only (no tape, book, lifecycle, queue or FIFO recomputed); consumed days; descriptive; filled orders are "
           "outcome-selected; no identified intent or private action; no profitability claim. Exact legs = closes.parquet FIFO rows keyed by full "
           "leg keys, clean cohort (status ok, non-self). Entry-close windows by hold time on the settlement clock (le10s / le60s cumulative, leEND = "
           "close settle <= END, afterEND = later); closing-earlier closes and the residual (to END / resolution) kept apart; END = we (market window "
           "end). Takes counted from the latest possible post (post_hi) with the END upper bound; a take is not an exit unless its taker leg "
           "FIFO-closes the order's leg. Response context requires the creation tick == resp_tick (else contextual only). Account-level P&L is "
           "never attached.", '',
           f"Pooled over the {pc['n_orders']} distinct chain orders (exact available {pc['n_available']}): entry closes "
           + '; '.join(_cl(b, pc['closes'][b]) for b in BUCKETS) + f"; closing earlier {_cl('', pc['earlier']).strip()}; residual "
           f"{_fmt(pc['residual_available']['sh'])} sh, to END {_sg(pc['residual_available']['val_end'])} $, to resolution "
           f"{_sg(pc['residual_available']['val_res'])} $ (fee {_fmt(pc['residual_available']['fee'], 3)} $).", '']
    for L in CHn:
        o = O[L['order']]
        others = [x for x in o['links'] if x != L['link_id']]
        out += [f"## Chain {L['chain']} - rank {L['rank']}: episode {L['episode']}", '',
                f"order {_h12(o['hash'])} ({L['side']} {L['cell_tick']}c), day {L['day']}, market {L['mid']}, address {L['addr']}; post grade "
                f"{L['post_grade']}; post window [{_rel_s(L['post_lo'], L['ws']) if L['post_lo'] >= 0 else 'unbounded'}, {_rel_s(L['anchor'], L['ws'])}]; "
                f"END {_rel_s(o['end'], L['ws'])}; other links of this order: {', '.join(others) or 'none'}", '',
                '| item | link | reading | histories | grade | summary |', '|---|---|---|---|---|---|']
        for it, ln in CHAIN_ITEMS:
            r = L['items'].get((it, ln))
            if r is None:
                out.append(f'| {it} | {ln} | MISSING | | | |')
                continue
            out.append(f"| {it} | {ln} | {_s(r['reading'])} | {_s(r['n_histories'])} | {_s(r['evidence_grade'])} | {_s(r['summary']).replace('|', '/')[:140]} |")
        out.append('')
        tied = [(it, ln, L['items'][(it, ln)]) for it, ln in CHAIN_ITEMS if (it, ln) in L['items'] and _s(L['items'][(it, ln)]['reading']) == 'TIED']
        out.append(f"- **Unresolved (TIED) links** ({len(tied)}): " + ('; '.join(f"{ln}: {_s(r['alternatives'])}" for _, ln, r in tied) or 'none'))
        if L['contexts']:
            for c in L['contexts']:
                out.append(f"- **Response context** level {c['level']} ({c['pos']} the post window), episode {c['eid']} ({c['resp_type'] or 'kind not saved'}): "
                           f"resp_tick {c['resp_tick']} vs cell {c['cell_tick']} -> **{c['status']}**" + (f" ({', '.join(c['reasons'])})" if c['reasons'] else ''))
        else:
            out.append(f"- **Response context**: none attached by chains_v4 ({L['parse']['n_lines']} creation line(s), {L['parse']['n_census_resp']} response "
                       "episode(s) in its census) -> no_response_context")
        if o['exact'] == 'available':
            out.append(f"- **Exact-leg continuation** (available; legs {o['n_clean']} clean of {o['n_rows']}, excluded status {o['n_excl_status']} / self "
                       f"{o['n_excl_self']}; {_fmt(o['clean_sh'])} sh, mq {_fmt(o['mq_sh'])}, rq {_fmt(o['rq_sh'])}): entry closes "
                       + '; '.join(_cl(b, o['csum'][b]) for b in BUCKETS)
                       + f"; leEND by closing leg: taker {_fmt(o['by_role']['taker']['sh'])} sh net {_sg(o['by_role']['taker']['net'])} $, maker "
                       f"{_fmt(o['by_role']['maker']['sh'])} sh net {_sg(o['by_role']['maker']['net'])} $; {_cl('closing earlier inventory (kept apart)', o['esum'])}")
        else:
            out.append(f"- **Exact-leg continuation**: unavailable ({o['exact_reason']}); legs {o['n_clean']} clean of {o['n_rows']}")
        rs, vw = o['resid'], o['views']
        out.append(f"- **Residual** (kept apart): {_fmt(rs['sh'])} sh unclosed; to END {_sg(rs['val_end'])} $ ({rs['n_end_ok']}/{rs['n']} legs with m_end), to "
                   f"resolution {_sg(rs['val_res'])} $ ({rs['n_res_ok']}/{rs['n']} resolved), prorated fee {_fmt(rs['fee'], 3)} $; fees of the clean legs "
                   f"{_fmt(o['fee_fill'], 3)} $")
        out.append(f"- **Views** (clean cohort, gross markouts beside): 10s gross {_sg(vw['v10s_gross'])} $ ({vw['v10s_n_valid']} valid), 60s "
                   f"{_sg(vw['v60s_gross'])} $ ({vw['v60s_n_valid']}), END {_sg(vw['vend_gross'])} $ ({vw['vend_n_valid']}); resolution net "
                   f"{_sg(vw['res_net'])} $ ({vw['res_n_valid']} valid)")
        if L['takes_status'] == 'evaluated':
            T = L['T']
            out.append(f"- **Takes of the address after the latest possible post** (anchor {_rel_s(L['anchor'], L['ws'])}): statuses {_cstr(T['status'])}; "
                       + '; '.join(f"{w} {T[w]['n']} take(s), |dq| {_fmt(T[w]['abs_dq'])} sh, dir_D0 {_cstr(T[w]['dir_D0']) or 'none'}, O4 "
                                   f"{_cstr(T[w]['o4']) or 'none'}, clock-disagreeing {T[w]['clock_disagree']}, FIFO-closing this order "
                                   f"{T[w]['fifo_closing']} ({_fmt(T[w]['fifo_closing_sh'])} sh)" for w in TAKE_W)
                       + ' - the other takes are not exits of this order')
        else:
            out.append('- **Takes**: not evaluated (not a top-3 address)')
        out.append('')
    out += ['## Predicates', '', f"acceptance all hold: {claims['acceptance_all_hold']}; verifier violations: {len(claims['verifier']['violations'])}", '']
    for k, v in claims['acceptance'].items():
        out.append(f"- {k}: {v['n_true']}/{v['n']} hold={v['holds']}")
    return '\n'.join(out) + '\n'


def _v(x):
    return NAN if x is None else x


def link_row(L, O):
    o = O[L['order']]
    r = {c: '' for c in LINK_COLS}
    r.update(link_id=L['link_id'], kind=L['kind'], chain=L['chain'], ep_key=L['ep_key'], day=L['day'], mid=L['mid'], side=L['side'], order_hash=o['hash'],
             addr=L['addr'], group=L['group'], top3=L['group'] in TOP3G, closes_rank=o['closes_rank'], order_side=o['side'], order_tick=o['tick'],
             ctx_status=L['ctx_status'], ctx_reasons='|'.join(L['ctx_reasons']), ctx_n=L['ctx_n'], ctx_detail=L['ctx_detail'], anchor_ms=L['anchor'],
             anchor_basis=L['anchor_basis'], end_ms=o['end'], exact_link=o['exact'], exact_reason=o['exact_reason'], n_links_of_order=len(o['links']),
             takes_status=L['takes_status'], unresolved_paths='|'.join(L['unresolved']))
    if L['kind'] == 'response':
        e = L['ep']
        r.update(owner_basis='candidate_first_filler (never the identified creator)', identity=_s(e['identity']), grade=_s(e['grade']),
                 identity_reason=_s(e['identity_reason']), pre_state_link=_s(e['pre_state_link']), alt_netting=_s(e['alt_netting']),
                 alt_first_other=_s(e['alt_first_other']), alt_first_addresses=_s(e['alt_first_addresses']), po_parity=_s(e['po_parity']),
                 win_contains=bool(CH3._b(e['win_contains'])), win_unbounded=bool(CH3._b(e['win_unbounded'])), resp_type=_s(e['resp_type']),
                 delta_ticks=_fin(e['delta_ticks']), exposure_ms=_int(e['exposure_ms']), end_reason=_s(e['end_reason']),
                 no_frame_between=bool(CH3._b(e['no_frame_between'])), resp_lo_ms=_int(e['resp_lo_ms']), resp_hi_ms=_int(e['resp_hi_ms']),
                 resp_tick=_int(e['resp_tick']), resp_size_shown=_fin(e['resp_size_shown']))
    else:
        r.update(owner_basis='recorded maker order of the chain address', identity='recorded_order', grade=L['post_grade'], post_grade=L['post_grade'],
                 chain_readings=';'.join(f"{it}:{ln}:{_s(L['items'][(it, ln)]['reading'])}:{_s(L['items'][(it, ln)]['n_histories'])}"
                                         for it, ln in CHAIN_ITEMS if (it, ln) in L['items']),
                 tied_links='|'.join(ln for it, ln in CHAIN_ITEMS if (it, ln) in L['items'] and _s(L['items'][(it, ln)]['reading']) == 'TIED'))
    if L['takes_status'] == 'evaluated':
        T = L['T']
        st = T['status']
        r.update(takes_subsequent=T['n_subsequent'], **{f'takes_{s_}': int(st.get(s_, 0)) for s_ in TAKE_ST})
        r['takes_excluded_by_END'] = int(st.get('straddles_END', 0) + st.get('after_END', 0) + st.get('end_unavailable', 0))
        for w in TAKE_W:
            r.update({f'takes_{w}_n': T[w]['n'], f'takes_{w}_abs_dq': T[w]['abs_dq'], f'takes_{w}_dir_D0': _cstr(T[w]['dir_D0']),
                      f'takes_{w}_o4': _cstr(T[w]['o4']), f'takes_{w}_clock_disagree': T[w]['clock_disagree'],
                      f'takes_{w}_fifo_closing': T[w]['fifo_closing'], f'takes_{w}_fifo_closing_sh': T[w]['fifo_closing_sh']})
    if L.get('saved') is not None:
        r.update(L['saved'])
    return r


def order_row(o, links_by_order):
    r = {c: '' for c in ORDER_COLS}
    kinds = Counter(o['kinds'])
    r.update(day=o['day'], mid=o['mid'], order_hash=o['hash'], addr=o['addr'], group=o['group'], top3=o['group'] in TOP3G, closes_rank=o['closes_rank'],
             n_links=len(o['links']), link_ids='|'.join(o['links']), n_response_links=kinds.get('response', 0), n_chain_links=kinds.get('chain', 0),
             end_ms=o['end'], side=o['side'], tick=o['tick'], tick_status=o['tick_status'], n_fv_rows=o['n_rows'], n_clean=o['n_clean'],
             n_excl_status=o['n_excl_status'], excl_statuses=_cstr(o['excl_statuses']), n_excl_self=o['n_excl_self'], clean_sh=o['clean_sh'],
             n_legs_after_END=o['n_legs_after_END'], first_settle_ms=o['first_settle_ms'], last_settle_ms=o['last_settle_ms'], mq_sh=o['mq_sh'],
             rq_sh=o['rq_sh'], fee_fill=o['fee_fill'], exact_link=o['exact'], exact_reason=o['exact_reason'], attributed_sh=_v(o['attributed_sh']),
             attribution_ok=o['attribution_ok'] if o['attribution_ok'] is not None else '', fees_decompose_ok=o['fees_ok'] if o['fees_ok'] is not None else '',
             n_close_on_excluded_leg=o['n_close_on_excl'], shared_close_keys=o.get('shared', 0))
    av = o['exact'] == 'available'
    for b in BUCKETS:
        for f in CLOSE_F:
            r[f'close_{b}_{f}'] = o['csum'][b][f] if av else NAN
    for rl in ('taker', 'maker'):
        r[f'close_leEND_{rl}_sh'] = o['by_role'][rl]['sh'] if av else NAN
        r[f'close_leEND_{rl}_net'] = o['by_role'][rl]['net'] if av else NAN
    for f in CLOSE_F:
        r[f'earlier_{f}'] = o['esum'][f] if av else NAN
    rs = o['resid']
    r.update(resid_sh=rs['sh'], resid_val_end=rs['val_end'], resid_n_end_ok=rs['n_end_ok'], resid_val_res=rs['val_res'], resid_n_res_ok=rs['n_res_ok'],
             resid_fee=rs['fee'], resid_net_res=rs['net_res'])
    r.update({k: v for k, v in o['views'].items() if k in r})
    sv = o.get('saved') or {}
    r.update(saved_fv_rows=sv.get('fv_rows', ''), saved_mq_sh=sv.get('mq_sh', ''), saved_rq_sh=sv.get('rq_sh', ''), saved_equal_legs=sv.get('equal_legs', ''),
             saved_equal_views=sv.get('equal_views', ''))
    return r


# ------------------------------------------------------------------ run ------------------------------------------------------------------
def run(mode='dry', rroot=None, droot=None, out=None, require_freeze=True, selftest=False, expect=None, git_probe=True, log=print):
    t0, asof0, c_start = time.time(), _now(), time.process_time()
    stages, last = {}, [c_start]

    def stage(nm):
        c = time.process_time()
        stages[nm] = round(c - last[0], 4)
        last[0] = c
        CH3.guard(nm)
        cpu_guard(nm, c_start)
    need(mode in ('dry', 'run'), f'unknown mode {mode!r}')
    need(C.RUN == 'v2', f'MT_RUN must be v2 (is {C.RUN!r})')
    mut = _mut()
    need(selftest or not mut, f'{MUT_ENV}={mut!r} is set: a planted mutant never runs outside the selftest (refused)')
    need(selftest or (require_freeze and git_probe), 'require_freeze=False / git_probe=False are the selftest only (refused)')
    exp = dict(EXPECT)
    exp.update(expect or {})
    rroot = Path(rroot) if rroot else C.RECEIPT
    droot = Path(droot) if droot else C.OUT
    out = Path(out) if out else rroot / 'chains' / 'continuation_econ'
    if mode == 'run':
        need(not out.exists(), f'write-once: {out} exists (refused before any input is read)')
    git = CH3._git(FREEZE_FILES) if git_probe else dict(head=None, files=None, tracked=None, unmodified=None, probe='skipped (selftest)')
    if mode == 'run' and require_freeze:
        need(bool(git.get('unmodified')), 'freeze: the tool, its selftest, the contract and chains_v3.py must be tracked and unmodified before a run (refused)')
    stage('preflight')
    IN, cnt = CH3.Inputs(rroot), Counter()

    def path(k):
        root, rel, _ = INPUTS[k]
        return (rroot if root == 'R' else droot) / rel
    CO = read_csv_str(path('continuations'), 'continuations', CO_READ)
    IN.add('continuations', path('continuations'), len(CO), INPUTS['continuations'][2])
    CO = CO[(CO['table'] == 'order').to_numpy()].reset_index(drop=True)
    EP = CH3.read_pq(path('episodes'), 'episodes', EP_READ)
    IN.add('episodes', path('episodes'), len(EP), INPUTS['episodes'][2])
    CL4 = read_csv_str(path('chain_links'), 'chain_links', CL4_READ)
    IN.add('chain_links', path('chain_links'), len(CL4), INPUTS['chain_links'][2])
    pm = path('chains_md')
    need(pm.is_file(), f'input chains_md is missing: {_rel(pm)} (refused)')
    md = pm.read_text()
    IN.add('chains_md', pm, md.count('\n'), INPUTS['chains_md'][2])
    PR = read_csv_str(path('post_regrade_orders'), 'post_regrade_orders', PR_READ)
    IN.add('post_regrade_orders', path('post_regrade_orders'), len(PR), INPUTS['post_regrade_orders'][2])
    PF = read_csv_str(path('post_orders'), 'post_orders', PF_READ)
    IN.add('post_orders', path('post_orders'), len(PF), INPUTS['post_orders'][2])
    EV = CH3.read_pq(path('events'), 'events', EV_READ)
    IN.add('events', path('events'), len(EV), INPUTS['events'][2])
    stage('small_inputs')
    # ---- links and orders ----
    EPK = {(_s(r['day']), _int(r['mid']), _int(r['episode_id'])): r for r in EP.to_dict('records')}
    RL = response_links(EP)
    COk, cchk = continuations_check(CO, RL)
    CHL = chain_links(CL4, PR, PF, md, EPK, mut)
    links = RL + CHL
    O = make_orders(links)
    ENDS, n_end_vals = market_ends(EP)
    for o in O.values():
        o['end'] = int(ENDS.get((o['day'], o['mid']), -1))
    mids = sorted({o['mid'] for o in O.values()})
    addrs = sorted({o['addr'] for o in O.values()})
    FV = CH3.read_groups(path('fills_valued'), 'fills_valued', FV_READ, mids, addrs, cnt) if O else pd.DataFrame({c: [] for c in FV_READ + ('addr_l',)})
    IN.add('fills_valued', path('fills_valued'), len(FV), INPUTS['fills_valued'][2])
    stage('fills_valued')
    attach_legs(O, FV, mut)
    rec_cl = IN.receipt(REC_CL)[0] or {}
    top10 = {_l(a.get('addr')): _int(a.get('rank')) for a in (rec_cl.get('addresses') or []) if isinstance(a, dict)}
    pop = [o for o in O.values() if o['addr'] in top10]
    XC = (CH3.read_groups(path('closes'), 'closes', XC_READ, sorted({o['mid'] for o in pop}), sorted({o['addr'] for o in pop}), cnt) if pop
          else pd.DataFrame({c: [] for c in XC_READ + ('addr_l',)}))
    IN.add('closes', path('closes'), len(XC), INPUTS['closes'][2])
    CR, grp = close_rows(XC)
    stage('closes')
    for o in O.values():
        o['_CR'] = CR
        order_econ(o, CR, grp, top10, mut)
        o['csum'] = {b: csum(o['close_keys'][b], CR) for b in BUCKETS}
        o['by_role'] = {rl: csum([k for k in o['close_keys']['leEND'] if CR[k]['close_role'] == rl], CR) for rl in ('taker', 'maker')}
        o['esum'] = csum(o['earlier_keys'], CR)
    owner = {}
    for o in O.values():                                         # closes touching two candidate orders (an entry of one, a close of another)
        for k in o['close_keys']['leEND'] + o['close_keys']['afterEND'] + o['earlier_keys']:
            owner.setdefault(k, set()).add(o['key'])
    for o in O.values():
        o['shared'] = sum(1 for k in set(o['close_keys']['leEND'] + o['close_keys']['afterEND'] + o['earlier_keys']) if len(owner[k]) > 1)
    # ---- per link: guards, takes, saved reproduction ----
    rec_tl2 = IN.receipt(REC_TL2)[0] or {}
    tl2_addrs = {_l(a) for a in (rec_tl2.get('addresses') or [])}
    evg = {}
    if len(EV):
        kk = list(zip(_sarr(EV['day']), [_int(x) for x in EV['mid']], [_l(x) for x in EV['addr']]))
        for i, k in enumerate(kk):
            evg.setdefault(k, []).append(i)
    for L in links:
        o = O[L['order']]
        if L['kind'] == 'response':
            rs = []
            if o['tick_status'] != 'ok':
                rs.append(f"order_tick_{o['tick_status']}")
            elif o['tick'] != L['resp_tick']:
                rs.append('tick_mismatch')
            if o['side'] != L['side']:
                rs.append('side_mismatch')
            if mut == 'no_tick_guard':                           # planted error: the join by frame interval alone (chains_v4's)
                rs = [x for x in rs if x != 'tick_mismatch' and not x.startswith('order_tick_')]
            L.update(ctx_status=CTX[0] if not rs else CTX[1], ctx_reasons=rs, ctx_n=1,
                     ctx_detail=f"order {o['side']} {o['tick']}c vs response {L['side']} {L['resp_tick']}c")
            e = L['ep']
            u = []
            if _s(e['identity']) == 'candidate_scenario':
                u.append('candidate_scenario_grade')
            if _s(e['pre_state_link']) != 'episode_state_G2_conditional':
                u.append(f"pre_state:{_s(e['pre_state_link']) or 'none'}")
            for f in ('alt_netting', 'alt_first_other'):
                if _s(e[f]) != 'not_possible':
                    u.append(f"{f}:{_s(e[f]) or 'missing'}")
            if _s(e['po_parity']) != 'equal':
                u.append(f"po_parity:{_s(e['po_parity']) or 'missing'}")
            L['unresolved'] = u
        else:
            cs = L['contexts']
            st_ = [c['status'] for c in cs]
            L.update(ctx_status=CTX[0] if CTX[0] in st_ else (CTX[1] if cs else CTX[2]), ctx_n=len(cs),
                     ctx_reasons=sorted({x for c in cs for x in c['reasons']}),
                     ctx_detail='; '.join(f"level {c['level']} {c['pos']} e{c['eid']} resp {c['resp_tick']}c vs cell {c['cell_tick']}c: {c['status']}" for c in cs))
            L['unresolved'] = [f"{ln}:{_s(L['items'][(it, ln)]['n_histories'])}" for it, ln in CHAIN_ITEMS
                               if (it, ln) in L['items'] and _s(L['items'][(it, ln)]['reading']) == 'TIED']
        if L['addr'] in tl2_addrs:
            ii = evg.get((L['day'], L['mid'], L['addr']), [])
            L['takes_status'] = 'evaluated'
            L['T'] = link_takes(L, EV.iloc[ii].reset_index(drop=True) if ii else EV.iloc[0:0], o['end'], o, mut)
        else:
            L['takes_status'], L['T'] = 'not_evaluated', None
        L['saved'] = None
        if L['kind'] == 'response' and L['order'] in COk:
            cr = COk[L['order']]
            if _s(cr['takes_status']) == 'evaluated' and L['T'] is not None:
                raw_ = L['T']['raw']
                sv = dict(saved_takes_end=_int(cr['takes_end']), saved_takes_10s=_int(cr['takes_10s']), saved_takes_60s=_int(cr['takes_60s']),
                          saved_takes_unbounded=_int(cr['takes_unbounded_after']))
                sv['saved_takes_reproduced'] = bool(sv['saved_takes_end'] == raw_['n_end'] and sv['saved_takes_10s'] == raw_['n_10s']
                                                    and sv['saved_takes_60s'] == raw_['n_60s'] and sv['saved_takes_unbounded'] == raw_['n_unbounded'])
                L['saved'] = sv
            elif _s(cr['takes_status']) == 'evaluated' or L['T'] is not None:
                L['saved'] = dict(saved_takes_reproduced=False)
    for o in O.values():                                         # the saved continuation row of a response-linked order
        cr = COk.get(o['key'])
        if cr is None:
            o['saved'] = None
            continue
        G = o['G']
        okr = _sarr(G['status']) == OK
        eq_legs = bool(o['n_rows'] == _int(cr['fv_rows']) and int(okr.sum()) == _int(cr['fv_rows_status_ok'])
                       and _close(float(np.nansum(G['mq'].to_numpy(np.float64)[okr])), _fin(cr['turnover_mq_sh']))
                       and _close(float(np.nansum(G['rq'].to_numpy(np.float64)[okr])), _fin(cr['residual_rq_sh'])))
        vw = o['views']
        eq_v = (bool(all(_close(vw[a], _fin(cr[b])) for a, b in (('v10s_net', 'v10s_net'), ('v60s_net', 'v60s_net'), ('vend_net', 'vend_net'),
                                                                ('res_net', 'res_net'))))
                if o['n_clean'] == o['n_rows'] else 'not_compared_unclean_legs')
        o['saved'] = dict(fv_rows=_int(cr['fv_rows']), mq_sh=_fin(cr['turnover_mq_sh']), rq_sh=_fin(cr['residual_rq_sh']), equal_legs=eq_legs, equal_views=eq_v)
    P = {sc: pooled(sc, links, O, CR, mut) for sc in SCOPES}
    stage('compute')
    # ---- verification and predicates ----
    raw = raw_tables(FV, CR, EP, EV, links, O, top10)
    raw['end'] = ENDS
    viol, vchk = verify(links, O, P, raw)
    IN_rec = {k: IN.receipt(v)[0] or {} for k, v in (('tt', REC_TT), ('ch4', REC_CH4), ('tl2', REC_TL2), ('cl', REC_CL))}
    sha = {k: v['sha256'] for k, v in IN.info.items()}
    bs = CH3.bound_sha
    cons = (rec_tl2.get('constants') or {})
    mk_scope = {(o['day'], o['mid']) for o in O.values()}
    pf_we = [(d, m, _int(w)) for d, m, w in zip(_sarr(PF['day']), [_int(x) for x in PF['mid']], PF['we_ms']) if (d, m) in mk_scope]
    acc = dict(cchk)
    acc.update(
        inputs_bound=[v['binding']['status'] == 'ok' for v in IN.info.values()],
        inputs_only_declared=[k in ALLOWED_INPUTS for k in IN.info],
        closes_built_on_this_fills_valued=[bs(IN_rec['cl'], 'inputs', 'fills_valued') == sha['fills_valued']],
        take_logic_v2_on_this_fills_valued=[bs(IN_rec['tl2'], 'inputs', 'econ/fills_valued.parquet') == sha['fills_valued']],
        touch_transition_read_this_events=[bs(IN_rec['tt'], 'inputs', 'chains/take_logic_v2/events.parquet') == sha['events']],
        chains_v4_read_this_fills_valued_and_closes=[bs(IN_rec['ch4'], 'inputs', 'fills_valued') == sha['fills_valued'],
                                                     bs(IN_rec['ch4'], 'inputs', 'closes') == sha['closes']],
        take_labels_equal_take_logic_v2=[tuple(cons.get('DIR_LAB') or ()) == DIR_LAB, tuple(cons.get('O4_ST') or ()) == O4_ST],
        response_links_as_declared=[len(RL) == int(exp['n_links'])],
        chains_as_declared=[len(CHL) == int(exp['n_chains'])],
        chain_has_every_link=[all((it, ln) in L['items'] for it, ln in CHAIN_ITEMS) and len(L['rows']) == len(CHAIN_ITEMS) for L in CHL],
        response_link_has_candidate=[bool(L['order'][2]) and bool(L['addr']) and L['group'] in TOP3G + ('other',) for L in RL],
        order_addr_consistent=[len(set(o['addrs'])) == 1 and o['fv_addrs'] in ([], [o['addr']]) for o in O.values()],
        order_group_consistent=[len(set(o['groups'])) == 1 for o in O.values()],
        order_has_fill_rows=[o['n_rows'] > 0 for o in O.values()],
        legs_equal_saved_continuations=[o['saved']['equal_legs'] for o in O.values() if o['saved'] is not None],
        views_equal_saved_where_all_clean=[o['saved']['equal_views'] for o in O.values() if o['saved'] is not None and o['saved']['equal_views'] != 'not_compared_unclean_legs'],
        chain_legs_equal_post_fill=[L_o['n_rows'] == _int(L['pf']['n_own_rows'])
                                    and _close(float(np.nansum(L_o['G']['mq'].to_numpy(np.float64)[_sarr(L_o['G']['status']) == OK])), _fin(L['pf']['mq_sh']))
                                    and _close(float(np.nansum(L_o['G']['rq'].to_numpy(np.float64)[_sarr(L_o['G']['status']) == OK])), _fin(L['pf']['rq_sh']))
                                    for L in CHL for L_o in [O[L['order']]]],
        leg_keys_unique=[len(set(kk)) == len(kk) for kk in [[(o['day'], o['mid']) + x for o in O.values() for x in o['keys']]]],
        order_tick_on_grid_unique=[o['tick_status'] == 'ok' for o in O.values()],
        end_recovered=[o['end'] >= 0 for o in O.values()],
        end_equals_post_fill_we=[w == ENDS.get((d, m), -2) for d, m, w in pf_we],
        attribution_equals_mq=[o['attribution_ok'] for o in O.values() if o['attribution_ok'] is not None],
        no_close_on_excluded_leg=[o['n_close_on_excl'] == 0 for o in O.values()],
        fees_decompose=[o['fees_ok'] for o in O.values() if o['fees_ok'] is not None],
        saved_take_counts_reproduced=[L['saved']['saved_takes_reproduced'] for L in RL if L['saved'] is not None],
        chain_creation_lines_parse=[L['parse']['n_parsed'] == L['parse']['n_lines'] for L in CHL],
        chain_creations_equal_census=[L['parse']['n_lines'] == L['parse']['n_census'] for L in CHL],
        chain_responses_equal_census=[L['parse']['n_resp'] == L['parse']['n_census_resp'] for L in CHL],
        chain_context_intervals_match_episodes=[not ({'interval_mismatch', 'episode_missing', 'not_a_response', 'side_mismatch'} & set(c['reasons']))
                                                for L in CHL for c in L['contexts']],
        chain_its_closes_reconcile=[_md_closes_ok(L, O[L['order']], CR) for L in CHL],
        **{f'verifier_{k}': [v] for k, v in vchk.items()})
    tc = []
    for o in O.values():
        if o['addr'] in tl2_addrs and o['exact'] == 'available':
            have = {k[3] for k in raw['takes_of'].get((o['day'], o['mid'], o['addr']), [])}
            tc += [s_ in have for s_ in o['taker_close']]
    diag = dict(taker_closes_are_recorded_takes=tc)
    acc = {k: _pred(v) for k, v in acc.items()}
    diag = {k: _pred(v) for k, v in diag.items()}
    acc_ok = bool(all(v['holds'] for v in acc.values())) and not viol
    claims = make_claims(mode, links, RL, CHL, O, P, acc, diag, viol, vchk, IN, n_end_vals, exp, acc_ok)
    if mode == 'run':
        failed = [k for k, v in acc.items() if not v['holds']]
        need(not failed and not viol, f'a run refuses on failed predicates {failed} / verifier violations {viol[:3]} (refused before any write)')
    asof = _now()
    LK = pd.DataFrame([link_row(L, O) for L in links], columns=list(LINK_COLS))
    OR = pd.DataFrame([order_row(o, None) for o in O.values()], columns=list(ORDER_COLS))
    md_out = chains_md(mode, asof, links, O, P, claims)
    files = [('links_econ.csv', LK.to_csv(index=False).encode()), ('orders_econ.csv', OR.to_csv(index=False).encode()),
             ('chains_econ.md', md_out.encode()), ('claims.json', _json_bytes(claims))]
    stage('render')
    srcs = {_rel(f): (_sha(f) if Path(f).is_file() else None) for f in TOOL_FILES + DEP_FILES}
    rec = dict(version=VERSION, mode=mode, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, sha256=_sha(CONTRACT_FILE) if CONTRACT_FILE.is_file() else None),
               review=REVIEW, module=_rel(Path(__file__)), run=C.RUN, asof_start=asof0, asof_end=_now(), argv=list(sys.argv), git=git, sources=srcs,
               inputs=IN.info, counts=dict(cnt), out=_rel(out), rules=RULES,
               outputs={nm: dict(sha256=hashlib.sha256(b).hexdigest(), bytes=len(b)) for nm, b in files},
               rows=dict(links_econ=len(LK), orders_econ=len(OR)),
               constants=dict(EXPECT=exp, BUCKETS=list(BUCKETS), HOLD_MS=HOLD_MS, TAKE_W=list(TAKE_W), TAKE_ST=list(TAKE_ST), DIR_LAB=list(DIR_LAB),
                              O4_ST=list(O4_ST), RES_VALID=list(RES_VALID), CTX=list(CTX), SCOPES=list(SCOPES), REL_TOL=REL_TOL, ABS_TOL=ABS_TOL,
                              TICK_TOL=TICK_TOL, MD_TOL=MD_TOL, CPU_CAP_S=CPU_CAP_S),
               cpu_s=dict(by_stage=stages, process_total=round(time.process_time(), 3), run_total=round(time.process_time() - c_start, 3), cap=CPU_CAP_S),
               wall_s=round(time.time() - t0, 2), vmhwm_mib=round(CH3.hwm(), 1), guard_mib=C.GUARD_MIB, mutant=mut or None)
    if mode == 'run':
        CH3.write_out(out, files + [('receipt.json', _json_bytes(rec))])
    dg = digest(claims, rec)
    log(json.dumps(CH3._clean(dg), indent=1, sort_keys=True))
    return dict(out=out, claims=claims, receipt=rec, links=links, orders=O, pooled=P, violations=viol, checks=vchk, md=md_out, files=dict(files), digest=dg)


def _md_closes_ok(L, o, CR):
    """chains_v4's rendered 'its closes' of the case order (entry or close leg of the hash) = this order's distinct entry + closing-earlier closes."""
    mc = L['md_closes']
    if mc is None or o['exact'] != 'available':
        return False
    s = csum(list(dict.fromkeys(o['close_keys']['leEND'] + o['close_keys']['afterEND'] + o['earlier_keys'])), CR)
    return bool(int(mc['n']) == s['n'] and int(mc['nd']) == s['n_dust'] and abs(float(mc['sh']) - s['sh']) <= MD_TOL
                and mc['pnl'] != 'na' and abs(float(mc['pnl']) - s['pnl']) <= MD_TOL and mc['net'] != 'na' and abs(float(mc['net']) - s['net']) <= MD_TOL)


def _pool_view(Pv):
    """the pooled numbers of a scope without the key lists."""
    t = Pv['takes']
    return dict({k: v for k, v in Pv.items() if k not in ('close_keys', 'earlier_keys', 'takes')},
                takes=dict({k: v for k, v in t.items() if k != 'windows'},
                           windows={w: {k: v for k, v in x.items() if k != 'keys'} for w, x in t['windows'].items()}))


def make_claims(mode, links, RL, CHL, O, P, acc, diag, viol, vchk, IN, n_end_vals, exp, acc_ok):
    rk = [O[k] for k in dict.fromkeys(L['order'] for L in RL)]
    tl = [L for L in links if L['takes_status'] == 'evaluated']
    guards = dict(
        end_bound=dict(takes_excluded_by_END_link_sum=sum(int(L['T']['status'].get('straddles_END', 0) + L['T']['status'].get('after_END', 0)
                                                               + L['T']['status'].get('end_unavailable', 0)) for L in tl),
                       takes_after_END_link_sum=sum(int(L['T']['status'].get('after_END', 0)) for L in tl),
                       takes_straddles_END_link_sum=sum(int(L['T']['status'].get('straddles_END', 0)) for L in tl),
                       closes_afterEND_distinct=P['union']['closes']['afterEND'], legs_after_END=sum(max(o['n_legs_after_END'], 0) for o in O.values()),
                       markets_end_values=dict(sorted(Counter(int(v) for v in n_end_vals.values()).items()))),
        cohort=dict(legs_excluded_status=sum(o['n_excl_status'] for o in O.values()), legs_excluded_self=sum(o['n_excl_self'] for o in O.values()),
                    orders_with_excluded_legs=sum(1 for o in O.values() if o['n_clean'] < o['n_rows'])),
        tick=dict(response_links_contextual_only=sum(1 for L in RL if L['ctx_status'] == CTX[1]),
                  response_link_reasons=_cnt(x for L in RL for x in L['ctx_reasons']),
                  chain_contexts=_cnt(c['status'] for L in CHL for c in L['contexts']),
                  chain_context_reasons=_cnt(x for L in CHL for c in L['contexts'] for x in c['reasons']),
                  chains_by_status=_cnt(L['ctx_status'] for L in CHL)),
        dedup=dict(orders_with_repeated_links=sum(1 for o in O.values() if len(o['links']) > 1), n_links=len(links), n_orders=len(O),
                   shared_close_keys_union=P['union']['shared_close_keys'],
                   take_overlap_union={w: P['union']['takes']['windows'][w]['overlap'] for w in TAKE_W}))
    return dict(
        version=VERSION, mode=mode, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, section='5'), review=REVIEW,
        population=dict(n_response_links=len(RL), n_chain_links=len(CHL), n_links=len(links), n_orders=len(O), n_orders_response=len(rk),
                        n_orders_chain=len({L['order'] for L in CHL}), n_orders_both=len({L['order'] for L in CHL} & {L['order'] for L in RL}),
                        response_orders_by_group=_cnt(o['group'] for o in rk), response_links_by_group=_cnt(L['group'] for L in RL),
                        response_links_by_identity=_cnt(_s(L['ep']['identity']) for L in RL), expect=exp),
        exact={sc: dict(available=P[sc]['n_available'], unavailable=P[sc]['n_unavailable'], reasons=P[sc]['unavailable_reasons']) for sc in SCOPES},
        pooled={sc: _pool_view(P[sc]) for sc in SCOPES}, guards=guards,
        acceptance=acc, acceptance_all_hold=acc_ok, diagnostics=diag, verifier=dict(checks=vchk, violations=viol),
        bindings={k: v['binding']['status'] for k, v in IN.info.items()},
        not_measured='no model skill, intent, identified private action, policy, validation or profitability: an accounting of saved public records')


def digest(claims, rec):
    """the dry-run reconciliation print (computed values only)."""
    pv = claims['pooled']

    def cl(sc):
        p = pv[sc]
        ra, ru = p['residual_available'], p['residual_unavailable']
        return dict(orders=p['n_orders'], orders_top3=p['n_orders_top3'], links=p['n_links'], available=p['n_available'], unavailable=p['n_unavailable'],
                    unavailable_reasons=p['unavailable_reasons'], legs=p['legs'], mq_sh_unavailable=round(p['mq_sh_unavailable'], 2),
                    closes={b: dict(n=p['closes'][b]['n'], n_dust=p['closes'][b]['n_dust'], sh=round(p['closes'][b]['sh'], 2),
                                    fee=round(p['closes'][b]['fee'], 3), net=round(p['closes'][b]['net'], 2)) for b in BUCKETS},
                    leEND_by_closing_leg={r_: dict(n=x['n'], sh=round(x['sh'], 2), net=round(x['net'], 2)) for r_, x in p['closes_leEND_by_close_role'].items()},
                    earlier=dict(n=p['earlier']['n'], sh=round(p['earlier']['sh'], 2), net=round(p['earlier']['net'], 2)),
                    shared_close_keys=p['shared_close_keys'],
                    residual_available=dict(sh=round(ra['sh'], 2), to_end=round(ra['val_end'], 2), end_cov=f"{ra['n_end_ok']}/{ra['n']}",
                                            to_res=round(ra['val_res'], 2), res_cov=f"{ra['n_res_ok']}/{ra['n']}", fee=round(ra['fee'], 3)),
                    residual_unavailable=dict(sh=round(ru['sh'], 2), to_end=round(ru['val_end'], 2), to_res=round(ru['val_res'], 2),
                                              res_cov=f"{ru['n_res_ok']}/{ru['n']}", fee=round(ru['fee'], 3)),
                    views={k: (round(v, 2) if isinstance(v, float) else v) for k, v in p['views'].items()},
                    takes=dict(links_eval=p['takes']['n_links_evaluated'], subsequent=p['takes']['subsequent_distinct'],
                               end_status=p['takes']['end_status_distinct'],
                               windows={w: dict(n=x['n_distinct'], order_sum=x['order_sum'], abs_dq=round(x['abs_dq'], 2), dir_D0=x['dir_D0'],
                                                fifo_close_keys=x['n_fifo_close_keys'], fifo_sh=round(x['fifo_closing_sh'], 2))
                                        for w, x in p['takes']['windows'].items()}))
    return dict(mode=claims['mode'], population=claims['population'], scopes={sc: cl(sc) for sc in SCOPES}, guards=claims['guards'],
                acceptance_all_hold=claims['acceptance_all_hold'], acceptance_failed=[k for k, v in claims['acceptance'].items() if not v['holds']],
                diagnostics={k: v for k, v in claims['diagnostics'].items()}, violations=claims['verifier']['violations'], bindings=claims['bindings'],
                cpu_s=rec['cpu_s'], vmhwm_mib=rec['vmhwm_mib'], counts=rec['counts'], git_unmodified=rec['git'].get('unmodified'))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['run', '--dry']:
        run(mode='dry')
    elif a == ['run']:
        run(mode='run')
    else:
        raise SystemExit('usage: MT_RUN=v2 continuation_econ.py run --dry | run')
