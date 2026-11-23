"""touch_transition.py - TOUCH-TRANSITION-AND-RESPONSE v1, sections 1-4 (P-2026-005 MAKER-TRACE; session aea5bfb9 builder, 2026-11-23).
Built EXACTLY to predictfun/reviews/maker_trace_20261122/touch_response_contract.md sections 1-4 (frozen before computation, commit 668e560db)
= the ONE next priority of review 01a72b4e2 (reviews/automatic/20261123_posting_chain_review/report.md, findings 1-2, next-priority items 1-5).
USER goal: the top makers' ACTION CHAINS and their logic - after the touch retreats, clears or moves, what observable state precedes the
replacement or improvement, which maker histories are compatible with it, and how that quote's chain develops. DESCRIPTIVE on four consumed
days (Oct 24-27, the 19 BOOK markets; G = 4 < 5: point estimates, no intervals); p is a within-stratum label-permutation tail fraction, not
independent-day confidence; no fit, no fair-value model, no threshold search, no replay, no model skill, no profitability claim.
S1 CENSUS every pair of adjacent usable depth frames (f - 1, f) of a market (queue_front.market_book: chain_context's usable frames, best
  ticks and gap flags), interval (upd(f-1), upd(f)]. Census population = pairs wholly inside the window (ws <= upd(f-1), upd(f) <= we); every
  other pair is counted as outside_window. CLASS (mutually exclusive, from best bid b / best ask a in f - 1 vs f, YES ticks; the first that
  applies): gap (upd(f) - upd(f-1) > 1,000 ms) | unknown (a best unknown (NaN) or a side empty (+-inf) in f - 1 or f; READING: a crossed book
  b >= a is unknown too, sub-label crossed) | unchanged (b and a unchanged; sizes may change) | two_side (both changed: both_up / both_down /
  widen / narrow) | retreat_bid (b down, a unchanged) | improve_bid (b up, a unchanged) | retreat_ask (a up, b unchanged) | improve_ask (a down,
  b unchanged). Only the NET change between the two frames is used: a same-frame retreat-and-refill is classed by its net change, never split
  into an invented order. CROSS-TABLE the census x posting_opportunity's saved cell events (events.csv = every queue_front fresh-cell candidate:
  improve / re_establish (a print-compatible reappearance) / deep / at_empty_touch / touch_unknown, and status-only creations) by frame pair
  (each row's frame, interval and fid checked against the frames; a mismatch is a status). RECONCILIATION reconcile.csv: every saved creation
  row -> its frame pair, class or an explicit unsupported status (outside_census_window / frame_out_of_range / time_mismatch / gap / unknown),
  the v1 anchor mask (rows.parquet improve_population of the row's second: anchor_excluded = an 'event' improve creation outside the v1 improve
  population) and its episode role; the improve creations (11,804 in the full run) and the anchor-excluded ones (8,821) by class / status.
S2 EPISODES one per retreat_<side> pair at f in the census population: exposure from upd(f); step delta = |best(f-1) - best(f)| on that side
  (bins 1 / 2 / >= 3 ticks); horizon H = upd(f) + 1,000 ms. The pairs after f are read in order until the first that ends it: (a) response -
  that side's best moves toward the spread: restoring (back to best(f-1)) / partial (between) / overshoot (better than best(f-1)); (b)
  censored_other - any other transition (the other side, two_side, a further same-side retreat); (c) censored_gap - a gap / unknown pair (or
  the book end); (d) no_response_1s - no transition observed by H. A transition is observed at its frame: response / censored_other end at
  upd(g) with bounds [upd(g-1), upd(g)] (no_frame_between: no frame between upd(f) and the response, timed only to that interval); an unknown
  pair ends the observed exposure at upd(g-1); a gap pair straddling H censors at upd(g-1) (the span is unobserved). READING (contract silent):
  the window end we censors when it comes before H (censored_gap, reason window_end). exposure_ms = end - upd(f); episodes never overlap.
  STATE strictly from frame f - 1 and data <= upd(f-1) (D0): spread (ticks, posting_opportunity bins), own-side touch depth (shares at the
  retreated side's best in f - 1, posting_opportunity bins), btc_5s_aligned (chain_take.Ctx.btc r_5s at upd(f-1), aligned +1 bid / -1 ask, 1.0
  bp), fair_gap (the EXISTING p_fair at upd(f-1), read exactly as posting_opportunity reads it, vs the post-retreat best best(f): bid 100 p_fair
  - b(f), ask a(f) - 100 p_fair; posting_opportunity bins), time_left (take_logic bins), side. LAG = the whole state read at frame f - 2
  (lag_missing when f - 2 does not exist or (f - 2, f - 1) is a gap: out of every LAG ratio). HAZARD = responses / exposure-seconds;
  Mantel-Haenszel person-time ratios: btc with vs flat, against vs flat, fair_gap > 1 vs <= 0, depth <= 50 vs > 200, time left <= 30 s vs
  (60, 240] s; strata market x side x delta bin x time-left bin (the time-left contrast omits its own bin). NULL 500 within-stratum permutations
  of the exposure label across episodes (only strata holding both labels can change), Generator(PCG64(SeedSequence([20261123, variant,
  contrast]))), p_hi / p_lo = (1 + #) / 501 (take_logic.tail_counts). Market consistency = the within-market MH ratio; supported market = >= 30
  exposed and >= 30 reference exposure-seconds and >= 1 response (READING: posting_opportunity's person-time unit); verdicts =
  posting_opportunity.verdict (supported / supported_lower / not_supported / insufficient_support); timing_sensitive when the LAG verdict
  differs (a state-lag sensitivity, not a reaction delay). Also: the termination census, the time-to-response distribution of responses (with
  bounds) and the restoring / partial / overshoot mix. The two_side population is a SEPARATE descriptive table (sub-label x state at f - 1),
  never a forced sequential response.
S3 OWNERS (separate layer). Candidate owner of a response's new best cell = queue_front_v2's candidate first filler of that fresh cell
  (candidates.csv, bound by its receipt) graded EXACTLY as posting_opportunity section 3 (posting_opportunity.identity on the cell: the candidate
  hash's lifecycle posting window containing the creation interval, the post_regrade grade: G2 -> candidate_constrained, else candidate_scenario;
  otherwise owner_unknown - in every denominator, never a maker WAIT; the eventual filler is never stated as the creator); parity with the saved
  posting_opportunity identity of the same cell recorded. queue_front_v2's masked-netting alternatives are carried (alt_netting / alt first-filler
  addresses: possible / not_possible / not_evaluated). pre_state_link = episode_state_G2_conditional only for a G2 window lying inside
  (upd(f-1), upd(g)], else unresolved_outside_window (G0 / G1 / G3 / G4 / no regrade row: an earlier or conflicted posting stays possible).
  CONTINUATIONS keyed rows, one per DISTINCT candidate order (its hash: repeated same-hash fills = ONE order): price / shown size / posting-time
  bounds; same-hash fills (lifecycle fills table); withdrawal / exhaustion and successor alternatives (chain_post_fill orders.csv union_histories
  / next_kind / next_link_status; top-3 orders only, else not_evaluated); subsequent recorded takes of the address (take_logic_v2 events.parquet:
  hard window starting after the creation, within 10 s / 60 s / to the end; top 3 only); the existing 10 s / 60 s / END views (fills_valued maker
  rows through chains_v3.views; shared valid cohort = valid at all three) and completed turnover / residual (fills_valued mq / rq of the order's
  own status-ok legs: the existing accounting). Tables pool every supported linked case with each order once (attribution overlap = orders
  linked to several responses; a take after several linked creations of one address counts ONCE in the pooled tables, its order-sum and
  overlap beside); no whole-book close value is summed per candidate creation. The termination census carries the class of the ending pair.
S4 GATE (touch_transition_selftest.py): eight fixtures, each with a planted error MT_TT_MUT caught by a named check: anchor_mask (1),
  split_two_side (2), drop_alternatives (3), conflict_identified (4), split_same_hash (5), missing_as_unchanged (6), no_horizon (7), state_leak (8).
OUTPUTS write-once (take_logic_v2.write_out: a partial directory renamed at the end): census.csv, reconcile.csv, episodes.parquet, ratios.csv,
  consistency.csv, two_side.csv, continuations.csv, claims.json (computed predicates / verdicts only), receipt.json (sha256 of sources, inputs and
  outputs; CPU by stage; VmHWM). run -> v2/chains/touch_transition; run --pilot <day> --out <dir> -> a pilot (prints the full-run projection).
GUARDS MT_RUN=v2; measurement days only; MT_TT_MUT and the reused tools' planted errors refused in production; every saved input bound to its
  producing receipt (sha256) and to the accepted fills_valued binding; the full run refuses unless this tool, its selftest, the contract and the
  reused tools are tracked and unmodified (the freeze) and the recorded pilot projection is <= 120 CPU s; VmHWM self-guard 1,450 MiB
  (pf_mt_common.GUARD_MIB, never raised); CPU self-guard.
Usage (run_light, MT_RUN=v2): touch_transition.py run | touch_transition.py run --pilot 20261024 --out <dir>    selftest: touch_transition_selftest.py"""
import json
import os
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
import chain_context as CX  # noqa: E402
import chain_take as CT  # noqa: E402
import queue_front as QF  # noqa: E402
import queue_front_v2 as QF2  # noqa: E402
import posting_opportunity as PO  # noqa: E402
import take_logic as TL  # noqa: E402
import take_logic_v2 as TL2  # noqa: E402
import chains_v3 as CH3  # noqa: E402

need = C.need
NAN, INF = float('nan'), float('inf')
VERSION = 'touch_transition_v1'
CONTRACT_FILE = C.RECEIPT_V1 / 'touch_response_contract.md'
CONTRACT_COMMIT = '668e560db'
REVIEW_FILE = C.REPO / 'predictfun' / 'reviews' / 'automatic' / '20261123_posting_chain_review' / 'report.md'
REVIEW_COMMIT = '01a72b4e2'
TOOL_FILES = (HERE / 'touch_transition.py', HERE / 'touch_transition_selftest.py', CONTRACT_FILE, HERE / 'queue_front.py', HERE / 'queue_front_v2.py',
              HERE / 'posting_opportunity.py', HERE / 'chain_context.py', HERE / 'chain_take.py', HERE / 'take_logic.py', HERE / 'take_logic_v2.py',
              HERE / 'chains_v3.py')
DEP_FILES = (RESEARCH / 'pf_mt_common.py', RESEARCH / 'pf_mt_econ.py', RESEARCH / 'pf_mt_lifecycle.py', RESEARCH / 'pf_mt_decision.py', REVIEW_FILE,
             PO.CONTRACT_FILE, QF.CONTRACT_FILE, TL.CONTRACT_FILE, TL2.CONTRACT_FILE, HERE / 'closes.py')
TOP3, TOP3_L = tuple(PO.TOP3), tuple(PO.TOP3_L)
N_BOOK = 19
PILOT_DAY = '20261024'
PILOT_RUN = 'r1'
FV_SHA256 = QF.FV_SHA256

# ------------------------------------------------------------------ declared constants (the contract's; none fitted) ------------------------------------------------------------------
NT = int(QF.NT)
GAP_MS = int(QF.GAP_MS)                                      # 1,000 ms
HORIZON_MS = 1_000                                           # the response horizon
NO_HORIZON_MS = 10 ** 12                                     # the no_horizon planted error only
CLASSES = ('unchanged', 'retreat_bid', 'retreat_ask', 'improve_bid', 'improve_ask', 'two_side', 'gap', 'unknown')
UNCH, RET_B, RET_A, IMP_B, IMP_A, TWO, GAP, UNK = range(8)
CENSUS_CLASSES = CLASSES + ('outside_window',)
TWO_SUB = ('both_up', 'both_down', 'widen', 'narrow')
UNK_SUB = ('unknown_touch', 'empty_side', 'crossed')
END_TYPES = ('response', 'censored_other', 'censored_gap', 'no_response_1s')
E_RESP, E_OTHER, E_GAP, E_NORESP = range(4)
END_REASONS = ('response', 'other_side', 'two_side', 'same_side_retreat', 'gap', 'unknown_touch', 'empty_side', 'crossed', 'book_end', 'window_end',
               'horizon')
RT_LAB = ('restoring', 'partial', 'overshoot')
DELTA_LAB = ('1', '2', '>=3')
SIDES = ('bid', 'ask')
VARIANTS = ('D0', 'LAG')
VDESC = dict(D0='state at frame f - 1 (data <= upd(f-1))', LAG='the whole state read at frame f - 2 (data <= upd(f-2))')
CONTRASTS = (('btc_with', 'btc_5s with', 'btc_5s flat'), ('btc_against', 'btc_5s against', 'btc_5s flat'),
             ('fair_gap', 'fair_gap > 1 c', 'fair_gap <= 0 c'), ('depth_own', 'own touch depth <= 50 sh', 'own touch depth > 200 sh'),
             ('time_left', 'time left <= 30 s', 'time left in (60,240] s'))
CN_NAMES = tuple(c[0] for c in CONTRASTS)
SEED, N_PERM, PERM_CHUNK = 20261123, 500, 100
MIN_EVENTS, MIN_MARKETS, MIN_MKT_S = PO.MIN_EVENTS, PO.MIN_MARKETS, PO.MIN_MKT_S      # 30 responses, 5 markets, 30 exposure-seconds
C_MAX, TIE = PO.C_MAX, PO.TIE
SHARE_NUM, SHARE_DEN = PO.SHARE_NUM, PO.SHARE_DEN
THR_5S = float(PO.THR_5S)
SPREAD_LAB, DEPTH_LAB, FG_LAB, TL_LAB = PO.SPREAD_LAB, PO.DEPTH_LAB, PO.FG_LAB, tuple(TL.TL_LAB)
BTC_LAB = PO.BTC_LAB                                          # with / against / flat / missing
BTC_DIR = ('up', 'down', 'flat', 'missing')                   # align_cat with s = +1 (two_side table)
VERDICTS = PO.VERDICTS
LINKED = ('candidate_constrained', 'candidate_scenario')
IDENT = PO.IDENT
G2 = PO.G2
PSL = ('episode_state_G2_conditional', 'unresolved_outside_window', 'owner_unknown')
ALT_ST = ('possible', 'not_possible', 'not_evaluated')
GROUPS = tuple(QF.GROUPS)
TAKE_H = (('10s', 10_000), ('60s', 60_000))
TAKE_KEYS = ('10s', '60s', 'end')
VIEW_H = tuple(CH3.HORIZONS)                                  # 10s, 60s, end
MUT_ENV = 'MT_TT_MUT'
MUTANTS = ('anchor_mask', 'split_two_side', 'drop_alternatives', 'conflict_identified', 'split_same_hash', 'missing_as_unchanged', 'no_horizon',
           'state_leak')
FOREIGN_MUT = ('MT_PO_MUT', 'MT_QF_MUT', 'MT_QF2_MUT', 'MT_TL_MUT', 'MT_TL2_MUT', 'MT_CH3_MUT')
CPU_CAP_RUN_S, CPU_CAP_PILOT_S = 150., 25.
PROJ_MAX_S = 120.
REVIEW_COUNTS = dict(improve_events=11_804, anchor_excluded=8_821, improve_event_seconds=4_484, anchor_excluded_event_seconds=3_664)
FRAMES = ('episodes.parquet',)
CSV_FILES = ('census.csv', 'reconcile.csv', 'ratios.csv', 'consistency.csv', 'two_side.csv', 'continuations.csv')
OUT_FILES = FRAMES + CSV_FILES + ('claims.json', 'receipt.json')
# ---- saved inputs: columns read (each file bound to its producing receipt) ----
EV_READ = ('key', 'day', 'mid', 'side', 'tick', 'frame', 'fid', 't_open_ms', 't_create_ms', 'row_k', 'status', 'is_improve', 'is_re_establish',
           'is_deep', 'is_at_empty_touch', 'touch_unknown', 'qf2_status', 'cand_addr', 'cand_hash', 'win_lo_ms', 'win_hi_ms', 'win_contains', 'grade',
           'identity', 'identity_reason')
EV_INT = ('mid', 'tick', 'frame', 'fid', 't_open_ms', 't_create_ms', 'row_k', 'win_lo_ms', 'win_hi_ms')
EV_BOOL = ('is_improve', 'is_re_establish', 'is_deep', 'is_at_empty_touch', 'touch_unknown', 'win_contains')
RW_READ = ('day', 'mid', 'side', 'k', 'improve_population', 'E_improve')
CAND_READ = ('day', 'mid', 'level_id', 'side', 'tick', 'frame', 't_open_ms', 't_create_ms', 'status', 'candidate_addr', 'candidate_hash',
             'timing_basis', 'alt_netting_possible', 'alt_first_addresses', 'alt_first_n_addresses', 'alt_first_other_address',
             'earliest_set_addresses', 'n_earliest_set')
CAND_INT = ('mid', 'level_id', 'tick', 'frame', 't_open_ms', 't_create_ms', 'alt_first_n_addresses', 'n_earliest_set')
CAND_BOOL = ('alt_netting_possible', 'alt_first_other_address')
PF_READ = ('day', 'mid', 'rank', 'addr', 'mk_hash', 'y_side', 'tick', 'n_fills', 'cum_shares', 'first_hi_ms', 'last_hi_ms', 'post_lo_union_ms',
           'post_hi_ms', 'size_status', 'union_histories', 'next_kind', 'next_link_status', 'n_children', 'next_hash', 'mq_sh', 'rq_sh', 'status')
PF_INT = ('mid', 'rank', 'tick')
PF_FLOAT = ('n_fills', 'cum_shares', 'first_hi_ms', 'last_hi_ms', 'post_lo_union_ms', 'post_hi_ms', 'n_children', 'mq_sh', 'rq_sh')
TK_READ = ('day', 'mid', 'addr', 'sid', 's', 'dq', 'x_lo_hard_ms', 'hi_ms', 'settle_ms', 'status', 'dir_D0')
FVV_READ = ('day', 'mid', 'sid', 'mk_idx', 'role', 'addr', 'hash', 'q', 'self_trade', 'settle_ms', 'st_res', 'st_10s', 'v_10s', 'st_60s', 'v_60s',
            'st_end', 'v_end', 'fee_fill', 'net_res', 'mq', 'rq', 'status')
Q_EMPTY_COLS = ('side', 'tick', 'frame', 't_open_ms', 't_create_ms', 'status', 'holder', 'front_hash', 'timing_basis', 'alt_netting_possible',
                'alt_first_addresses', 'alt_first_n_addresses', 'alt_first_other_address', 'earliest_set_addresses', 'n_earliest_set')
PAIR_CNT = ('n_cre', 'n_cre_improve', 'n_cre_re_establish', 'n_cre_deep', 'n_cre_at_empty_touch', 'n_cre_type_unknown', 'n_cre_bid', 'n_cre_ask',
            'n_cre_status_only', 'n_improve_event', 'n_improve_anchor_excluded')
CEN_SUM = PAIR_CNT + ('n_episode_starts', 'n_response_ends')
REC_COLS = ('key', 'day', 'mid', 'side', 'tick', 'frame', 'fid', 't_open_ms', 't_create_ms', 'po_status', 'is_improve', 'is_re_establish', 'is_deep',
            'is_at_empty_touch', 'touch_unknown', 'row_k', 'anchor_population', 'anchor_excluded', 'pair_status', 'cls', 'sublabel', 'recon_status',
            'episode_role', 'episode_id')
OWN_STR = ('qf2_status', 'cand_addr', 'cand_group', 'cand_hash', 'grade', 'identity', 'identity_reason', 'inv_relation', 'alt_netting',
           'alt_first_addresses', 'alt_first_other', 'earliest_set_addresses', 'pre_state_link', 'po_parity', 'po_identity')
OWN_INT = ('win_lo_ms', 'win_hi_ms', 'alt_first_n_addresses', 'n_earliest_set', 'resp_tick', 'resp_fid')
OWN_BOOL = ('win_unbounded', 'win_contains')
STATE_F = ('st', 'frame', 'ms', 'spread', 'sp_c', 'd_own', 'd_c', 'r5', 'btc_c', 'pf', 'fg', 'fg_c', 'tl', 'tl_c')


# ------------------------------------------------------------------ small helpers ------------------------------------------------------------------
def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


_rel, _sha, _clean, hwm, guard, _git = PO._rel, PO._sha, PO._clean, PO.hwm, PO.guard, PO._git
_gt1, _lt1 = PO._gt1, PO._lt1
bound_sha = CH3.bound_sha


def paths():
    """Run-time paths on the MT_RUN roots (v2: C.RECEIPT = the receipt's v2/ directory)."""
    ch = C.RECEIPT / 'chains'
    return dict(out=ch / 'touch_transition', pilot_receipt=ch / f'touch_transition_pilot_{PILOT_DAY}_{PILOT_RUN}' / 'receipt.json',
                po=ch / 'posting_opportunity', qf2=ch / 'queue_front_v2', regrade=ch / 'correction_v1', post_fill=ch / 'post_fill',
                tl2=ch / 'take_logic_v2')


def pilot_out(day):
    return C.RECEIPT / 'chains' / f'touch_transition_pilot_{day}_{PILOT_RUN}'


def mutant():
    """The selftest's planted-error switch MT_TT_MUT, read at call time; production refuses any value."""
    m = os.environ.get(MUT_ENV, '')
    need(m == '' or m in MUTANTS, f'{MUT_ENV}={m!r} is not a declared mutant {MUTANTS} (refused)')
    return m


def _lab(labs, codes):
    """Integer codes -> labels ('' for a negative code)."""
    c = np.asarray(codes, np.int64)
    out = np.full(len(c), '', dtype=object)
    ok = c >= 0
    out[ok] = np.asarray(labs, dtype=object)[c[ok]]
    return out


def _qd(x, pre):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    out = {f'{pre}_n': int(len(x)), f'{pre}_mean': float(x.mean()) if len(x) else NAN}
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        out[f'{pre}_p{int(round(q * 100)):02d}'] = float(np.quantile(x, q)) if len(x) else NAN
    return out


def read_csv_typed(p, cols, ints=(), floats=(), bools=(), what=''):
    """A saved CSV read as strings (no NA conversion), then typed: integer columns must be whole numbers, booleans 'True' / 'False'."""
    p = Path(p)
    hdr = list(pd.read_csv(p, nrows=0).columns)
    miss = [c for c in cols if c not in hdr]
    need(not miss, f'{what or p.name} lacks {miss} (refused: interface)')
    D = pd.read_csv(p, usecols=list(cols), dtype=str, keep_default_na=False)
    for c in ints:
        v = pd.to_numeric(D[c], errors='coerce')
        need(bool(v.notna().all()), f'{what or p.name}: column {c} holds non-numeric values (refused)')
        D[c] = v.astype(np.int64)
    for c in floats:
        D[c] = pd.to_numeric(D[c], errors='coerce').astype(np.float64)
    for c in bools:
        need(bool(D[c].isin(['True', 'False']).all()), f'{what or p.name}: column {c} is not boolean (refused)')
        D[c] = (D[c] == 'True').to_numpy(bool)
    return D[list(cols)]


# ------------------------------------------------------------------ saved inputs (each bound to its producing receipt) ------------------------------------------------------------------
def read_po(pdir, markets, hashes, fv_sha):
    """posting_opportunity's events.csv + rows.parquet, both bound to its receipt (sha256), its fills_valued the accepted binding -> per-market
    frames and the queue_front_v2 levels.csv sha256 its identity layer read."""
    pdir = Path(pdir)
    ev, rw, rp = pdir / 'events.csv', pdir / 'rows.parquet', pdir / 'receipt.json'
    need(ev.is_file() and rw.is_file() and rp.is_file(), f'posting_opportunity output incomplete: {pdir} (refused)')
    R = json.loads(rp.read_text())
    hashes[_rel(rp)] = _sha(rp)
    need(R.get('version') == PO.VERSION, f'{rp}: not a {PO.VERSION} receipt (refused)')
    for nm, p in (('events.csv', ev), ('rows.parquet', rw)):
        sh = _sha(p)
        hashes[_rel(p)] = sh
        need(bound_sha(R, 'outputs', nm) == sh, f"posting_opportunity {nm} differs from its receipt's sha256 (refused: mismatched artifacts)")
    need(bound_sha(R, 'inputs', 'econ/fills_valued.parquet') == fv_sha, 'posting_opportunity read another fills_valued (refused: mismatched artifacts)')
    lvs = ((R.get('dependency') or {}).get('queue_front_v2') or {}).get('levels_sha256')
    E = read_csv_typed(ev, EV_READ, ints=EV_INT, bools=EV_BOOL, what='posting_opportunity events.csv')
    need(bool(E['key'].is_unique), 'posting_opportunity events.csv: a cell-event key twice (refused)')
    names = pq.ParquetFile(rw).schema_arrow.names
    miss = [c for c in RW_READ if c not in names]
    need(not miss, f'posting_opportunity rows.parquet lacks {miss} (refused: interface)')
    W = pq.read_table(rw, columns=list(RW_READ)).to_pandas()
    W['day'] = W['day'].astype(str)
    W['side'] = W['side'].astype(str)
    keep = [(str(d), int(m)) for d, m in markets]
    EVd = {k: E[(E['day'] == k[0]).to_numpy() & (E['mid'] == k[1]).to_numpy()].reset_index(drop=True) for k in keep}
    RWd = {k: W[(W['day'] == k[0]).to_numpy() & (W['mid'].to_numpy(np.int64) == k[1])].reset_index(drop=True) for k in keep}
    return EVd, RWd, dict(dir=_rel(pdir), kind=R.get('kind'), events_rows=int(len(E)), rows_rows=int(len(W)),
                          events_run=int(sum(len(v) for v in EVd.values())), rows_run=int(sum(len(v) for v in RWd.values())), levels_sha256=lvs)


def read_qf2(qdir, markets, hashes, fv_sha, levels_sha):
    """queue_front_v2's candidates.csv (keyed per E1 level) bound to its receipt; the receipt's levels.csv must be the one posting_opportunity's
    identity read; (day, mid, side, tick, frame) unique -> per-market frames renamed to posting_opportunity's adapter (holder, front_hash)."""
    qdir = Path(qdir)
    cp, rp = qdir / 'candidates.csv', qdir / 'receipt.json'
    need(cp.is_file() and rp.is_file(), f'queue_front_v2 output incomplete: {qdir} (refused)')
    R = json.loads(rp.read_text())
    hashes[_rel(rp)] = _sha(rp)
    need(R.get('version') == QF2.VERSION, f'{rp}: not a {QF2.VERSION} receipt (refused)')
    sc = _sha(cp)
    hashes[_rel(cp)] = sc
    need(bound_sha(R, 'outputs', 'candidates.csv') == sc, "queue_front_v2 candidates.csv differs from its receipt's sha256 (refused: mismatched artifacts)")
    need(bound_sha(R, 'inputs', 'econ/fills_valued.parquet') == fv_sha, 'queue_front_v2 read another fills_valued (refused: mismatched artifacts)')
    ls = bound_sha(R, 'outputs', 'levels.csv')
    need(levels_sha is not None and ls == levels_sha, "queue_front_v2's levels.csv is not the one posting_opportunity's identity layer read "
                                                      '(refused: mismatched artifacts)')
    D = read_csv_typed(cp, CAND_READ, ints=CAND_INT, bools=CAND_BOOL, what='queue_front_v2 candidates.csv')
    need(bool(pd.MultiIndex.from_frame(D[['day', 'mid', 'side', 'tick', 'frame']]).is_unique), 'queue_front_v2 candidates.csv: a cell frame twice (refused)')
    D = D.rename(columns=PO.QF2_RENAME)
    keep = [(str(d), int(m)) for d, m in markets]
    out = {k: D[(D['day'] == k[0]).to_numpy() & (D['mid'] == k[1]).to_numpy()].reset_index(drop=True) for k in keep}
    return out, dict(dir=_rel(qdir), rows=int(len(D)), rows_run=int(sum(len(v) for v in out.values())), candidates_sha256=sc, levels_sha256=ls,
                     front_candidates_run=int(sum(int((v['status'] == QF2.FRONT).sum()) for v in out.values())))


def read_post_fill(pdir, markets, hashes, fv_sha):
    """chain_post_fill's orders.csv (every top-3 filled order: size / ending histories, successor) bound to its receipt and row count."""
    pdir = Path(pdir)
    po_, pr = pdir / 'orders.csv', pdir / 'receipt.json'
    need(po_.is_file() and pr.is_file(), f'chain_post_fill output incomplete: {pdir} (refused)')
    R = json.loads(pr.read_text())
    hashes[_rel(pr)] = _sha(pr)
    so = _sha(po_)
    hashes[_rel(po_)] = so
    need(bound_sha(R, 'outputs', 'orders.csv') == so, "chain_post_fill orders.csv differs from its receipt's sha256 (refused: mismatched artifacts)")
    need(bound_sha(R, 'inputs', 'econ/fills_valued.parquet') == fv_sha, 'chain_post_fill read another fills_valued (refused: mismatched artifacts)')
    D = read_csv_typed(po_, PF_READ, ints=PF_INT, floats=PF_FLOAT, what='chain_post_fill orders.csv')
    need(len(D) == int((R.get('rows') or {}).get('orders', -1)), "chain_post_fill orders.csv rows differ from its receipt (refused)")
    need(bool(pd.MultiIndex.from_frame(D[['day', 'mid', 'mk_hash']]).is_unique), 'chain_post_fill orders.csv: an order twice (refused)')
    keep = {(str(d), int(m)) for d, m in markets}
    sel = np.array([(str(d), int(m)) in keep for d, m in zip(D['day'].tolist(), D['mid'].tolist())], bool)
    D = D[sel].reset_index(drop=True)
    D['addr_l'] = D['addr'].str.lower()
    return D, dict(dir=_rel(pdir), rows_run=int(len(D)))


def read_takes(tdir, markets, hashes, fv_sha):
    """take_logic_v2's events.parquet (every recorded take record of the top 3) bound to its receipt."""
    tdir = Path(tdir)
    ep, rp = tdir / 'events.parquet', tdir / 'receipt.json'
    need(ep.is_file() and rp.is_file(), f'take_logic_v2 output incomplete: {tdir} (refused)')
    R = json.loads(rp.read_text())
    hashes[_rel(rp)] = _sha(rp)
    need(R.get('version') == TL2.VERSION, f'{rp}: not a {TL2.VERSION} receipt (refused)')
    sh = _sha(ep)
    hashes[_rel(ep)] = sh
    need(bound_sha(R, 'outputs', 'events.parquet') == sh, "take_logic_v2 events.parquet differs from its receipt's sha256 (refused: mismatched artifacts)")
    need(bound_sha(R, 'inputs', 'econ/fills_valued.parquet') == fv_sha, 'take_logic_v2 read another fills_valued (refused: mismatched artifacts)')
    names = pq.ParquetFile(ep).schema_arrow.names
    miss = [c for c in TK_READ if c not in names]
    need(not miss, f'take_logic_v2 events.parquet lacks {miss} (refused: interface)')
    T = CH3._undict(pq.read_table(ep, columns=list(TK_READ))).to_pandas()
    for c in ('day', 'addr', 'sid', 'status', 'dir_D0'):
        T[c] = T[c].astype(str)
    for c in ('mid', 's', 'x_lo_hard_ms', 'hi_ms', 'settle_ms'):
        T[c] = T[c].astype(np.int64)
    keep = {(str(d), int(m)) for d, m in markets}
    sel = np.array([(d, int(m)) in keep for d, m in zip(T['day'].tolist(), T['mid'].tolist())], bool)
    T = T[sel].reset_index(drop=True)
    T['addr_l'] = T['addr'].str.lower()
    return T, dict(dir=_rel(tdir), rows_run=int(len(T)))


def read_fv(fvp, LG, cnt):
    """The fills_valued maker rows of the candidate orders' legs (chains_v3.read_groups: only the markets' row groups; then the legs' (mid, sid,
    mk_idx) keys with role maker)."""
    if LG is None or not len(LG):
        return pd.DataFrame({c: [] for c in FVV_READ + ('addr_l',)})
    D = CH3.read_groups(fvp, 'fills_valued', FVV_READ, sorted({int(m) for m in LG['mid']}), sorted(set(LG['addr_l'])), cnt)
    if not len(D):
        return D
    D = D[(D['role'].astype(str) == 'maker').to_numpy()].reset_index(drop=True)
    kl = pd.MultiIndex.from_arrays([LG['mid'].to_numpy(np.int64), LG['sid'].astype(str).to_numpy(dtype=object), LG['mk_idx'].to_numpy(np.int64)])
    kd = pd.MultiIndex.from_arrays([D['mid'].to_numpy(np.int64), D['sid'].astype(str).to_numpy(dtype=object), D['mk_idx'].to_numpy(np.int64)])
    D = D[kd.isin(kl)].reset_index(drop=True)
    need(bool(pd.MultiIndex.from_arrays([D['mid'].to_numpy(np.int64), D['sid'].astype(str).to_numpy(dtype=object),
                                         D['mk_idx'].to_numpy(np.int64)]).is_unique), 'fills_valued: a maker leg twice (refused)')
    cnt['fv_view_rows'] += int(len(D))
    return D


# ------------------------------------------------------------------ section 1: classes ------------------------------------------------------------------
def classify(t, best, gapf):
    """Per frame f >= 1 the class of the pair (f - 1, f) from the NET change of the best bid / ask (first that applies: gap, unknown, unchanged,
    two_side, retreat / improve per side); index 0 has no pair (-1). MT_TT_MUT=missing_as_unchanged / split_two_side are planted errors."""
    mut = mutant()
    t = np.asarray(t, np.int64)
    F = int(len(t))
    B = np.asarray(best, np.float64).reshape(2, F).copy()
    cls = np.full(F, -1, np.int8)
    sub = np.full(F, '', dtype=object)
    db, da = np.full(F, NAN), np.full(F, NAN)
    if mut == 'missing_as_unchanged' and F:          # planted error: an unknown touch carried forward as the last known best
        B = pd.DataFrame(B.T).ffill().to_numpy(np.float64).T.copy()
    if F < 2:
        return dict(cls=cls, sub=sub, db=db, da=da, B=B)
    b0, a0, b1, a1 = B[0, :-1], B[1, :-1], B[0, 1:], B[1, 1:]
    g = np.asarray(gapf, bool)[1:]
    nan_ = np.isnan(b0) | np.isnan(a0) | np.isnan(b1) | np.isnan(a1)
    fin = np.isfinite(b0) & np.isfinite(a0) & np.isfinite(b1) & np.isfinite(a1)
    with np.errstate(invalid='ignore'):
        crossed = fin & ((b0 >= a0) | (b1 >= a1))
        d_b = np.where(fin, b1 - b0, 0.)
        d_a = np.where(fin, a1 - a0, 0.)
    bch, ach = d_b != 0, d_a != 0
    unk = ~fin | crossed
    c = np.select([g, unk, ~bch & ~ach, bch & ach, bch & (d_b < 0), bch & (d_b > 0), ach & (d_a > 0), ach & (d_a < 0)],
                  [GAP, UNK, UNCH, TWO, RET_B, IMP_B, RET_A, IMP_A], -2).astype(np.int8)
    need(bool((c >= 0).all()), 'internal: a frame pair without a class')
    s2 = np.select([(d_b > 0) & (d_a > 0), (d_b < 0) & (d_a < 0), (d_b < 0) & (d_a > 0), (d_b > 0) & (d_a < 0)], list(TWO_SUB), '')
    su = np.select([nan_, ~fin, crossed], list(UNK_SUB), '')
    if mut == 'split_two_side':                       # planted error: a simultaneous change read as a retreat first (sequential)
        two = c == TWO
        c = np.where(two & (d_b < 0), RET_B, np.where(two & (d_a > 0), RET_A, c)).astype(np.int8)
    cls[1:] = c
    sub[1:] = np.where(c == TWO, s2, np.where(c == UNK, su, '')).astype(object)
    db[1:] = np.where(fin, d_b, NAN)
    da[1:] = np.where(fin, d_a, NAN)
    return dict(cls=cls, sub=sub, db=db, da=da, B=B)


# ------------------------------------------------------------------ section 2: episodes and states ------------------------------------------------------------------
def _ep_empty():
    z, zf, zo = np.zeros(0, np.int64), np.zeros(0), np.zeros(0, dtype=object)
    return dict(n=0, f=z, s=z, g=z, et=np.zeros(0, np.int8), reason=zo, t0=z, tprev=z, t_end=z, expo=z, elo=z, t_read=z, end_class=z, after=z,
                bp=zf, br=zf, bo=zf, bg=zf, rt=z, delta=zf, dbin=z, nfb=np.zeros(0, bool), resp_lo=z, resp_hi=z)


def episodes(t, cl, inpop, ws, we):
    """One episode per retreat_<side> pair at f in the census population; read the later pairs in order until the first that ends it
    (contract section 2; times by frame observation). MT_TT_MUT=anchor_mask / no_horizon are planted errors."""
    mut = mutant()
    t = np.asarray(t, np.int64)
    F = int(len(t))
    cls, B = cl['cls'], cl['B']
    fr = np.flatnonzero(np.asarray(inpop, bool) & ((cls == RET_B) | (cls == RET_A))).astype(np.int64)
    if mut == 'anchor_mask' and len(fr):              # planted error: the v1 anchor mask (an episode in a second whose anchor spread is 1 dropped)
        ak = int(ws) + 1000 * np.floor_divide(t[fr - 1] - int(ws), 1000)
        ja = np.searchsorted(t, ak, 'right') - 1
        jc_ = np.maximum(ja, 0)
        with np.errstate(invalid='ignore'):
            sp1 = (ja >= 0) & ((B[1, jc_] - B[0, jc_]) == 1.)
        fr = fr[~sp1]
    n = int(len(fr))
    if n == 0:
        return _ep_empty()
    s = (cls[fr] == RET_A).astype(np.int64)
    tr = cls != UNCH
    tr[0] = False
    nxt = QF._acc_next(tr, F).astype(np.int64)
    g = np.where(fr + 1 < F, nxt[np.minimum(fr + 1, F - 1)], F).astype(np.int64)
    t0 = t[fr]
    Hf = t0 + (NO_HORIZON_MS if mut == 'no_horizon' else HORIZON_MS)
    He = np.minimum(Hf, int(we))
    jh = np.searchsorted(t, He, 'right').astype(np.int64)
    gc, jc, jp = np.minimum(g, F - 1), np.minimum(jh, F - 1), np.maximum(jh - 1, 0)
    A = (g < F) & (g < jh)                            # a transition observed at a frame at or before the horizon
    cg = np.where(g < F, cls[gc], -1).astype(np.int64)
    s_imp, s_ret = np.where(s == 0, IMP_B, IMP_A), np.where(s == 0, RET_B, RET_A)
    resp = A & (cg == s_imp)
    agap = A & ((cg == UNK) | (cg == GAP))
    aoth = A & ~resp & ~agap
    exact = ~A & (t[jp] == He)
    bend = ~A & ~exact & (jh >= F)
    sgap = ~A & ~exact & ~bend & (cls[jc] == GAP)
    hz = ~A & ~exact & ~bend & ~sgap
    eh = exact | hz
    win = He < Hf
    et = np.full(n, -1, np.int8)
    et[resp] = E_RESP
    et[aoth] = E_OTHER
    et[agap | bend | sgap | (eh & win)] = E_GAP
    et[eh & ~win] = E_NORESP
    need(bool((et >= 0).all()), 'internal: an episode without an end type')
    reason = np.full(n, '', dtype=object)
    reason[resp] = 'response'
    mu = A & (cg == UNK)
    reason[mu] = cl['sub'][gc[mu]]
    reason[A & (cg == GAP)] = 'gap'
    reason[aoth & (cg == TWO)] = 'two_side'
    reason[aoth & (cg == s_ret)] = 'same_side_retreat'
    reason[aoth & (cg != TWO) & (cg != s_ret)] = 'other_side'
    reason[eh & ~win] = 'horizon'
    reason[eh & win] = 'window_end'
    reason[bend] = 'book_end'
    reason[sgap] = 'gap'
    t_end = np.full(n, -1, np.int64)
    ao = resp | aoth
    t_end[ao] = t[gc[ao]]
    t_end[agap] = t[gc[agap] - 1]
    t_end[eh] = He[eh]
    t_end[bend] = t[F - 1]
    t_end[sgap] = t[jp[sgap]]
    expo = t_end - t0
    elo = expo.copy()
    elo[ao] = t[gc[ao] - 1] - t0[ao]
    t_read = np.full(n, -1, np.int64)
    t_read[A] = t[gc[A]]
    t_read[exact] = t[jp[exact]]
    m_ = hz | sgap
    t_read[m_] = t[jc[m_]]
    t_read[bend] = t[F - 1]
    bp, br, bo = B[s, fr - 1], B[s, fr], B[1 - s, fr]
    bg = np.where(resp, B[s, gc], NAN)
    with np.errstate(invalid='ignore'):
        rest = resp & (bg == bp)
        over = resp & np.where(s == 0, bg > bp, bg < bp)
        part = resp & ~rest & ~over
        okp = np.where(s == 0, (bg > br) & (bg < bp), (bg < br) & (bg > bp))
    need(bool(okp[part].all()), 'internal: a partial response outside (best(f), best(f-1))')
    rt = np.select([rest, part, over], [0, 1, 2], -1).astype(np.int64)
    delta = np.abs(bp - br)
    need(bool(np.isfinite(delta).all() and (delta >= 1).all()), 'internal: a retreat without a finite step')
    dbin = np.select([delta < 1.5, delta < 2.5], [0, 1], 2).astype(np.int64)
    return dict(n=n, f=fr, s=s, g=np.where(A, g, -1).astype(np.int64), et=et, reason=reason, t0=t0, tprev=t[fr - 1], t_end=t_end, expo=expo, elo=elo,
                t_read=t_read, end_class=np.where(A, cg, -1).astype(np.int64), after=np.where(hz, cls[jc], -1).astype(np.int64), bp=bp, br=br, bo=bo,
                bg=bg, rt=rt, delta=delta, dbin=dbin, nfb=resp & (g == fr + 1), resp_lo=np.where(resp, t[gc - 1], -1).astype(np.int64),
                resp_hi=np.where(resp, t[gc], -1).astype(np.int64))


def state(M, ctx, t, B, bsz, asz, fr, s, bpost, we):
    """The frozen state read at usable frame fr (-1 = missing) and its time x = upd(fr): spread, own-side touch depth, btc_5s_aligned, the
    existing p_fair vs the post-retreat best bpost (fair_gap), time left; BTC through chain_take.Ctx.btc at x (posting_opportunity's reads)."""
    fr = np.asarray(fr, np.int64)
    s = np.asarray(s, np.int64)
    n = int(len(fr))
    if n == 0:
        z, zf = np.zeros(0, np.int64), np.zeros(0)
        return dict(st=np.zeros(0, np.int8), frame=z, ms=z, spread=zf, sp_c=np.zeros(0, np.int8), d_own=zf, d_c=np.zeros(0, np.int8), r5=zf,
                    btc_c=np.zeros(0, np.int8), pf=zf, fg=zf, fg_c=np.zeros(0, np.int8), tl=zf, tl_c=np.zeros(0, np.int8))
    ok = fr >= 0
    fc = np.maximum(fr, 0)
    x = np.where(ok, t[fc], -1).astype(np.int64)
    b, a = np.where(ok, B[0, fc], NAN), np.where(ok, B[1, fc], NAN)
    with np.errstate(invalid='ignore'):
        sp = np.where(np.isfinite(b) & np.isfinite(a) & (a > b), a - b, NAN)
    own = np.where(s == 0, b, a)
    d_own = np.where(ok & np.isfinite(own), np.where(s == 0, bsz[fc], asz[fc]), NAN)
    xq = np.where(ok, x, int(M.ws)).astype(np.int64)
    Bt = ctx.btc(M, xq)
    okb = ok & (np.asarray(Bt['code']) == 0)
    r5 = np.where(okb, np.asarray(Bt['r_5s'], float), NAN)
    pf = np.where(okb, np.asarray(Bt['pf'], float), NAN)
    bc = TL.align_codes(np.where(s == 0, 1., -1.), r5, THR_5S)
    bpost = np.asarray(bpost, np.float64)
    with np.errstate(invalid='ignore'):
        fg = np.where(np.isfinite(pf) & np.isfinite(bpost), np.where(s == 0, 100. * pf - bpost, bpost - 100. * pf), NAN)
    tl = np.where(ok, (int(we) - xq) / 1000., NAN)
    tlc = np.zeros(n, np.int8)
    if ok.any():
        tlc[ok] = TL.tl_codes(tl[ok])
    return dict(st=np.where(ok, 0, 1).astype(np.int8), frame=np.where(ok, fr, -1), ms=x, spread=sp, sp_c=PO.spread_codes(sp), d_own=d_own,
                d_c=PO.depth_codes(d_own), r5=r5, btc_c=bc, pf=pf, fg=fg, fg_c=PO.fg_codes(fg), tl=tl, tl_c=tlc)


# ------------------------------------------------------------------ section 3: owners ------------------------------------------------------------------
def _q_empty():
    return pd.DataFrame({c: pd.Series(dtype=object) for c in Q_EMPTY_COLS})


def owner_links(day, mid, M, ep, t, fidv, S, Q2m, RGm, EVm, cnt):
    """Per episode (responses only; '' / -1 elsewhere): the new best cell's queue_front_v2 candidate, graded by posting_opportunity.identity
    (the frozen section-3 code, run on the cell with candidates.csv as its source), the masked-netting alternatives, the pre-state link and the
    parity with the saved posting_opportunity identity of the same cell. MT_TT_MUT=drop_alternatives / conflict_identified are planted errors."""
    mut = mutant()
    n = int(ep['n'])
    O = {k: np.full(n, '', dtype=object) for k in OWN_STR}
    for k in OWN_INT:
        O[k] = np.full(n, -1, np.int64)
    for k in OWN_BOOL:
        O[k] = np.zeros(n, bool)
    O['resp_size_shown'] = np.full(n, NAN)
    ri = np.flatnonzero(ep['et'] == E_RESP)
    m = int(len(ri))
    if m == 0:
        return O
    s, g = ep['s'][ri], ep['g'][ri]
    tick = np.rint(ep['bg'][ri]).astype(np.int64)
    a, b = t[g - 1], t[g]
    O['resp_tick'][ri], O['resp_fid'][ri] = tick, fidv[g]
    O['resp_size_shown'][ri] = S[g, s * NT + tick]
    Q = Q2m if (Q2m is not None and len(Q2m)) else _q_empty()
    LV = pd.DataFrame(dict(frame=g))
    CR = dict(e1=np.ones(m, bool), side=s, tick=tick, a=a, b=b, status=np.full(m, 'event', dtype=object))
    FB = dict(st_v1=np.full(m, '', dtype=object), timing_v1=np.full(m, '', dtype=object), st_v2=np.full(m, '', dtype=object),
              lead_v2=np.full(m, -1, np.int64), timing_v2=np.full(m, '', dtype=object))
    c2 = Counter()
    o = PO.identity(day, mid, M, dict(addr=np.zeros(0, dtype=object)), LV, CR, FB, RGm or {}, Q, c2)
    cnt['owner_cells'] += m
    cnt['owner_cells_unmatched_in_candidates'] += int(c2.get('qf2_e1_unmatched', 0))
    cnt['owner_top3_without_regrade_row'] += int(c2.get('candidate_top3_without_regrade_row', 0))
    for k in ('qf2_status', 'cand_addr', 'cand_group', 'cand_hash', 'grade', 'identity', 'identity_reason', 'inv_relation'):
        O[k][ri] = np.asarray(o[k], dtype=object)
    O['win_lo_ms'][ri], O['win_hi_ms'][ri] = np.asarray(o['win_lo_ms'], np.int64), np.asarray(o['win_hi_ms'], np.int64)
    O['win_unbounded'][ri], O['win_contains'][ri] = np.asarray(o['win_unbounded'], bool), np.asarray(o['win_contains'], bool)
    qk = {(str(r['side']), int(r['tick']), int(r['frame'])): r for r in Q.to_dict('records')} if len(Q) else {}
    for j, i in enumerate(ri.tolist()):
        r = qk.get((SIDES[int(s[j])], int(tick[j]), int(g[j])))
        tb = '' if r is None else str(r.get('timing_basis', ''))
        if r is None or tb in ('', 'nan'):                # no earliest record: the alternatives were never evaluated (never False)
            O['alt_netting'][i] = O['alt_first_other'][i] = 'not_evaluated'
            continue
        O['alt_netting'][i] = 'possible' if bool(r['alt_netting_possible']) else 'not_possible'
        O['alt_first_other'][i] = 'possible' if bool(r['alt_first_other_address']) else 'not_possible'
        O['alt_first_addresses'][i] = str(r['alt_first_addresses'])
        O['alt_first_n_addresses'][i] = int(r['alt_first_n_addresses'])
        O['earliest_set_addresses'][i] = str(r['earliest_set_addresses'])
        O['n_earliest_set'][i] = int(r['n_earliest_set'])
    lk = np.isin(O['identity'], LINKED)
    if mut == 'drop_alternatives':                      # planted error: the candidate read as the only history (masked netting dropped)
        O['alt_netting'][lk] = 'not_possible'
        O['alt_first_other'][lk] = 'not_possible'
        O['alt_first_addresses'][lk] = O['cand_addr'][lk]
        O['alt_first_n_addresses'][lk] = 1
    inside = lk & (O['win_lo_ms'] >= 0) & (O['win_lo_ms'] > ep['tprev']) & (O['win_hi_ms'] <= ep['resp_hi'])
    g2 = O['grade'] == G2
    if mut == 'conflict_identified':                   # planted error: a window inside the episode identifies the pre-state whatever its grade
        g2 = np.ones(n, bool)
    isr = ep['et'] == E_RESP
    O['pre_state_link'] = np.where(~isr, '', np.where(~lk, 'owner_unknown', np.where(inside & g2, PSL[0], PSL[1]))).astype(object)
    if EVm is not None and len(EVm):
        ev = {k_: r for k_, r in zip(EVm['key'].tolist(), EVm[['identity', 'grade', 'cand_addr', 'cand_hash', 'win_contains']].to_dict('records'))}
        for j, i in enumerate(ri.tolist()):
            r = ev.get(f'{int(mid)}|{SIDES[int(s[j])]}|{int(tick[j])}|{int(fidv[g[j]])}')
            if r is None:
                O['po_parity'][i] = 'absent'
                continue
            O['po_identity'][i] = str(r['identity'])
            same = (str(r['identity']) == O['identity'][i] and str(r['grade']) == O['grade'][i] and str(r['cand_addr']).lower() == O['cand_addr'][i]
                    and str(r['cand_hash']) == O['cand_hash'][i] and bool(r['win_contains']) == bool(O['win_contains'][i]))
            O['po_parity'][i] = 'equal' if same else 'differs'
    else:
        O['po_parity'][ri] = 'absent'
    return O


# ------------------------------------------------------------------ section 1: the creation cross and the reconciliation rows ------------------------------------------------------------------
def creation_rows(day, mid, EVm, RWm, t, fidv, cl, inpop, ep, B):
    """posting_opportunity's saved creation rows of the market -> per-pair counts (index = frame f of the pair (f - 1, f)) and reconcile rows
    (frame pair, class or unsupported status, the v1 anchor mask of an 'event' row, the episode role)."""
    F = int(len(t))
    P = {c: np.zeros(max(F, 1), np.int64) for c in PAIR_CNT}
    if EVm is None or not len(EVm):
        return P, pd.DataFrame({c: [] for c in REC_COLS})
    E = EVm
    n = int(len(E))
    fr = E['frame'].to_numpy(np.int64)
    okr = (fr >= 1) & (fr < F)
    fc = np.clip(fr, 1, max(F - 1, 1))
    if F >= 2:
        tm = okr & (E['t_open_ms'].to_numpy(np.int64) == t[fc - 1]) & (E['t_create_ms'].to_numpy(np.int64) == t[fc]) & (E['fid'].to_numpy(np.int64) == fidv[fc])
        inw = inpop[fc]
        cc = np.where(tm, cl['cls'][fc], -1).astype(np.int64)
        sb = np.where(tm, cl['sub'][fc], '').astype(object)
    else:
        tm, inw, cc, sb = np.zeros(n, bool), np.zeros(n, bool), np.full(n, -1, np.int64), np.full(n, '', dtype=object)
    pst = np.select([~okr, ~tm, ~inw], ['frame_out_of_range', 'time_mismatch', 'outside_census_window'], 'in_census').astype(object)
    pos = E['status'].to_numpy(dtype=object)
    fl = {k: E[k].to_numpy(bool) for k in ('is_improve', 'is_re_establish', 'is_deep', 'is_at_empty_touch', 'touch_unknown')}
    sd = np.where(E['side'].to_numpy(dtype=object) == 'bid', 0, 1).astype(np.int64)
    tk = E['tick'].to_numpy(np.int64)
    rk = E['row_k'].to_numpy(np.int64)
    ip = np.full(n, -1, np.int64)
    if RWm is not None and len(RWm):
        K = int(RWm['k'].max()) + 1
        tab = np.full((2, K), -1, np.int64)
        tab[np.where(RWm['side'].to_numpy(dtype=object) == 'bid', 0, 1), RWm['k'].to_numpy(np.int64)] = RWm['improve_population'].to_numpy(bool).astype(np.int64)
        evr = (pos == 'event') & (rk >= 0) & (rk < K)
        ip[evr] = tab[sd[evr], rk[evr]]
    is_ev = pos == 'event'
    imp_ev = is_ev & fl['is_improve']
    excl = imp_ev & (ip == 0)
    so = np.isin(pos, ('gap', 'unknown_cell'))
    rmap = {(int(s_), int(g_)): int(i) for i, (s_, g_, e_) in enumerate(zip(ep['s'].tolist(), ep['g'].tolist(), ep['et'].tolist())) if e_ == E_RESP}
    role = np.full(n, 'no_pair', dtype=object)
    eid = np.full(n, -1, np.int64)
    for i in np.flatnonzero(tm).tolist():
        j = rmap.get((int(sd[i]), int(fr[i])))
        if j is None:
            role[i] = 'not_a_response'
            continue
        eid[i] = j
        role[i] = 'response_new_best' if float(tk[i]) == float(B[sd[i], fr[i]]) else 'response_other_cell'
    cls_lab = _lab(CLASSES, cc)
    rs = np.where(pst != 'in_census', 'unsupported:' + pst,                           # object arrays: elementwise str concatenation
                  np.where(cc == GAP, 'unsupported:gap', np.where(cc == UNK, 'unsupported:unknown:' + sb, 'class:' + cls_lab))).astype(object)
    v = tm
    for c, w in (('n_cre', np.ones(n, bool)), ('n_cre_improve', fl['is_improve']), ('n_cre_re_establish', fl['is_re_establish']),
                 ('n_cre_deep', fl['is_deep']), ('n_cre_at_empty_touch', fl['is_at_empty_touch']), ('n_cre_type_unknown', fl['touch_unknown']),
                 ('n_cre_bid', sd == 0), ('n_cre_ask', sd == 1), ('n_cre_status_only', so), ('n_improve_event', imp_ev), ('n_improve_anchor_excluded', excl)):
        sel = v & w
        if sel.any():
            P[c] = np.bincount(fr[sel], minlength=max(F, 1)).astype(np.int64)
    RC = pd.DataFrame(dict(key=E['key'].to_numpy(dtype=object), day=np.full(n, str(day), dtype=object), mid=np.full(n, int(mid), np.int64),
                           side=E['side'].to_numpy(dtype=object), tick=tk, frame=fr, fid=E['fid'].to_numpy(np.int64),
                           t_open_ms=E['t_open_ms'].to_numpy(np.int64), t_create_ms=E['t_create_ms'].to_numpy(np.int64), po_status=pos,
                           is_improve=fl['is_improve'], is_re_establish=fl['is_re_establish'], is_deep=fl['is_deep'],
                           is_at_empty_touch=fl['is_at_empty_touch'], touch_unknown=fl['touch_unknown'], row_k=rk,
                           anchor_population=np.select([ip == 1, ip == 0], ['in', 'excluded'], 'not_applicable').astype(object), anchor_excluded=excl,
                           pair_status=pst, cls=cls_lab, sublabel=sb, recon_status=rs, episode_role=role, episode_id=eid))
    return P, RC


# ------------------------------------------------------------------ the two_side population (descriptive) ------------------------------------------------------------------
def two_side_rows(day, mid, M, ctx, t, B, bsz, asz, cl, i2, we, P):
    """Every two_side pair in the census population with the state at f - 1 (spread, BTC 5 s direction, time left, both touch depths) and the
    creations of the pair: a separate descriptive population, never a sequential response."""
    i2 = np.asarray(i2, np.int64)
    n = int(len(i2))
    cols = ('day', 'mid', 'f', 'lo_ms', 'hi_ms', 'sublabel', 'spread_bin', 'btc_5s_dir', 'time_left_bin', 'depth_bid_bin', 'depth_ask_bin',
            'move_bid_ticks', 'move_ask_ticks', 'n_cre', 'n_cre_improve', 'n_cre_deep')
    if n == 0:
        return pd.DataFrame({c: [] for c in cols})
    fp = i2 - 1
    b, a = B[0, fp], B[1, fp]
    with np.errstate(invalid='ignore'):
        sp = np.where(np.isfinite(b) & np.isfinite(a) & (a > b), a - b, NAN)
    Bt = ctx.btc(M, t[fp])
    okb = np.asarray(Bt['code']) == 0
    r5 = np.where(okb, np.asarray(Bt['r_5s'], float), NAN)
    bd = TL.align_codes(np.ones(n), r5, THR_5S)
    tlc = TL.tl_codes((int(we) - t[fp]) / 1000.)
    mb, ma = np.abs(cl['db'][i2]), np.abs(cl['da'][i2])

    def mv(x):
        return np.array(DELTA_LAB, dtype=object)[np.select([x < 1.5, x < 2.5], [0, 1], 2)]
    return pd.DataFrame(dict(day=np.full(n, str(day), dtype=object), mid=np.full(n, int(mid), np.int64), f=i2, lo_ms=t[fp], hi_ms=t[i2],
                             sublabel=cl['sub'][i2], spread_bin=_lab(SPREAD_LAB, PO.spread_codes(sp)), btc_5s_dir=_lab(BTC_DIR, bd),
                             time_left_bin=_lab(TL_LAB, tlc), depth_bid_bin=_lab(DEPTH_LAB, PO.depth_codes(bsz[fp])),
                             depth_ask_bin=_lab(DEPTH_LAB, PO.depth_codes(asz[fp])), move_bid_ticks=mv(mb), move_ask_ticks=mv(ma),
                             n_cre=P['n_cre'][i2], n_cre_improve=P['n_cre_improve'][i2], n_cre_deep=P['n_cre_deep'][i2]))[list(cols)]


# ------------------------------------------------------------------ one market ------------------------------------------------------------------
def market_pass(day, mid, M, ctx, EVm=None, RWm=None, Q2m=None, RGm=None):
    """One market: the census pairs (section 1), the response episodes with their D0 / LAG states (section 2), the two_side pairs, the creation
    cross and reconcile rows, the owner layer of the responses (section 3) and the same-hash legs of their candidate orders -> a part."""
    mut = mutant()
    day, mid = str(day), int(mid)
    ws, we = int(M.ws), int(M.we)
    cnt = Counter()
    bk = QF.market_book(M)
    t, gapf, F, fidv = np.asarray(bk['t'], np.int64), np.asarray(bk['gap'], bool), int(bk['F']), np.asarray(bk['fid'], np.int64)
    Dz = M.depth['best']
    bsz, asz = np.asarray(Dz['bsz'], np.float64), np.asarray(Dz['asz'], np.float64)
    cl = classify(t, bk['best'], gapf)
    B = cl['B']
    inpop = np.zeros(F, bool)
    if F >= 2:
        inpop[1:] = (t[:-1] >= ws) & (t[1:] <= we)
    ep = episodes(t, cl, inpop, ws, we)
    n = int(ep['n'])
    f, s = ep['f'], ep['s']
    bpost = B[s, f] if n else np.zeros(0)
    fd0 = f.copy() if mut == 'state_leak' else f - 1    # planted error state_leak: the state read at the post-transition frame f
    lagok = (f - 2 >= 0) & ~gapf[np.maximum(f - 1, 0)] if n else np.zeros(0, bool)
    flg = np.where(lagok, f - 2, -1).astype(np.int64)
    ST = {'D0': state(M, ctx, t, B, bsz, asz, fd0, s, bpost, we), 'LAG': state(M, ctx, t, B, bsz, asz, flg, s, bpost, we)}
    OW = owner_links(day, mid, M, ep, t, fidv, bk['S'], Q2m, RGm, EVm, cnt)
    P, RC = creation_rows(day, mid, EVm, RWm, t, fidv, cl, inpop, ep, B)
    TSr = two_side_rows(day, mid, M, ctx, t, B, bsz, asz, cl, np.flatnonzero(inpop & (cl['cls'] == TWO)), we, P)
    k = np.arange(1, F, dtype=np.int64)
    eps = np.bincount(f, minlength=max(F, 1)) if n else np.zeros(max(F, 1), np.int64)
    rg = ep['g'][ep['et'] == E_RESP]
    rse = np.bincount(rg, minlength=max(F, 1)) if len(rg) else np.zeros(max(F, 1), np.int64)
    pairs = pd.DataFrame(dict(day=np.full(len(k), day, dtype=object), mid=np.full(len(k), mid, np.int64), f=k, lo_ms=t[k - 1], hi_ms=t[k],
                              dt_ms=t[k] - t[k - 1], cls=cl['cls'][k].astype(np.int64), sub=cl['sub'][k], in_window=inpop[k],
                              **{c: P[c][k] for c in PAIR_CNT}, n_episode_starts=eps[k], n_response_ends=rse[k]))
    lk = np.isin(OW['identity'], LINKED)
    hs = sorted({str(h) for h in OW['cand_hash'][lk] if str(h)})
    LG = pd.DataFrame({c: [] for c in ('day', 'mid', 'mk_hash', 'addr_l', 'sid', 'mk_idx', 'shares', 'hi_ms', 'x_lo_hard_ms', 'y_side', 'y_ptick')})
    if hs:
        L = M.life_fills
        X = L[np.isin(L['mk_hash'].astype(str).to_numpy(dtype=object), np.array(hs, dtype=object))]
        adr = {str(h): str(a) for h, a in zip(OW['cand_hash'][lk].tolist(), OW['cand_addr'][lk].tolist())}
        LG = pd.DataFrame(dict(day=np.full(len(X), day, dtype=object), mid=np.full(len(X), mid, np.int64), mk_hash=X['mk_hash'].astype(str).to_numpy(dtype=object),
                               addr_l=np.array([adr[str(h)] for h in X['mk_hash'].astype(str)], dtype=object), sid=X['sid'].astype(str).to_numpy(dtype=object),
                               mk_idx=X['mk_idx'].to_numpy(np.int64), shares=X['shares'].to_numpy(np.float64), hi_ms=X['hi_ms'].to_numpy(np.int64),
                               x_lo_hard_ms=X['x_lo_hard_ms'].to_numpy(np.int64), y_side=X['y_side'].astype(str).to_numpy(dtype=object),
                               y_ptick=X['y_ptick'].to_numpy(np.int64)))
        cnt['candidate_orders'] += len(hs)
        cnt['candidate_orders_without_legs'] += int(len(set(hs) - set(LG['mk_hash'].tolist())))
    cnt['frames'] += F
    cnt['pairs'] += int(len(k))
    cnt['pairs_in_window'] += int(inpop.sum())
    cnt['episodes'] += n
    cnt['creation_rows'] += int(len(RC))
    if len(RC):
        cnt['creation_rows_time_mismatch'] += int((RC['pair_status'] == 'time_mismatch').sum())
        cnt['creation_rows_frame_out_of_range'] += int((RC['pair_status'] == 'frame_out_of_range').sum())
    return dict(day=day, mid=mid, ws=ws, we=we, F=F, pairs=pairs, ep=ep, ST=ST, OW=OW, two_side=TSr, reconcile=RC, legs=LG, counts=cnt)


# ------------------------------------------------------------------ assembled frames ------------------------------------------------------------------
def episodes_frame(parts):
    """episodes.parquet: one row per episode (keys, the defining transition, end, exposure and bounds, D0 / LAG states, the owner layer)."""
    fr_ = []
    for p in parts:
        ep, ST, OW = p['ep'], p['ST'], p['OW']
        n = int(ep['n'])
        d = dict(day=np.full(n, p['day'], dtype=object), mid=np.full(n, p['mid'], np.int64), episode_id=np.arange(n, dtype=np.int64),
                 side=_lab(SIDES, ep['s']), frame=ep['f'], t_prev_ms=ep['tprev'], t_start_ms=ep['t0'], best_prev=ep['bp'], best_retreat=ep['br'],
                 other_best=ep['bo'], delta_ticks=ep['delta'], delta_bin=_lab(DELTA_LAB, ep['dbin']), end_type=_lab(END_TYPES, ep['et']),
                 end_reason=ep['reason'].astype(object), end_frame=ep['g'], end_class=_lab(CLASSES, ep['end_class']), t_end_ms=ep['t_end'],
                 exposure_ms=ep['expo'], exposure_lo_ms=ep['elo'], exposure_hi_ms=ep['expo'], no_frame_between=ep['nfb'], resp_type=_lab(RT_LAB, ep['rt']),
                 resp_best=ep['bg'], resp_lo_ms=ep['resp_lo'], resp_hi_ms=ep['resp_hi'], after_horizon_class=_lab(CLASSES, ep['after']),
                 t_read_max_ms=ep['t_read'])
        for v in VARIANTS:
            X = ST[v]
            d.update({f'state_status_{v}': np.where(X['st'] == 0, 'ok', 'lag_missing').astype(object), f'state_frame_{v}': X['frame'],
                      f'state_ms_{v}': X['ms'], f'spread_{v}': X['spread'], f'spread_bin_{v}': _lab(SPREAD_LAB, X['sp_c']), f'depth_own_{v}': X['d_own'],
                      f'depth_own_bin_{v}': _lab(DEPTH_LAB, X['d_c']), f'btc_r_5s_bp_{v}': X['r5'], f'btc_5s_aligned_{v}': _lab(BTC_LAB, X['btc_c']),
                      f'p_fair_{v}': X['pf'], f'fair_gap_c_{v}': X['fg'], f'fair_gap_bin_{v}': _lab(FG_LAB, X['fg_c']), f'time_left_s_{v}': X['tl'],
                      f'time_left_bin_{v}': np.where(X['st'] == 0, _lab(TL_LAB, X['tl_c']), '').astype(object)})
        for k in OWN_STR:
            d[k] = OW[k].astype(object)
        for k in OWN_INT:
            d[k] = OW[k].astype(np.int64)
        for k in OWN_BOOL:
            d[k] = OW[k].astype(bool)
        d['resp_size_shown'] = OW['resp_size_shown']
        fr_.append(pd.DataFrame(d))
    return pd.concat(fr_, ignore_index=True) if fr_ else pd.DataFrame()


def ratio_inputs(parts):
    """The episode arrays of every part with its market index (strata market x side x delta bin x time-left bin)."""
    E = {k: [] for k in ('mi', 'side', 'dbin', 'resp', 'expo_ms')}
    for v in VARIANTS:
        for k in ('st', 'btc_c', 'fg_c', 'd_c', 'tl_c'):
            E[f'{k}_{v}'] = []
    for i, p in enumerate(parts):
        ep = p['ep']
        n = int(ep['n'])
        E['mi'].append(np.full(n, i, np.int64))
        E['side'].append(ep['s'].astype(np.int64))
        E['dbin'].append(ep['dbin'].astype(np.int64))
        E['resp'].append((ep['et'] == E_RESP).astype(np.float64))
        E['expo_ms'].append(ep['expo'].astype(np.float64))
        for v in VARIANTS:
            for k in ('st', 'btc_c', 'fg_c', 'd_c', 'tl_c'):
                E[f'{k}_{v}'].append(np.asarray(p['ST'][v][k]).astype(np.int64))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in E.items()}


# ------------------------------------------------------------------ section 2: hazard ratios, permutation null, market consistency ------------------------------------------------------------------
def contrast_code(E, v, cn, n_mk):
    """Exposure group (1 exposed / 0 reference / -1 out) and stratum of contrast cn at variant v -> (grp, strat, S, per)."""
    st = np.asarray(E['st_' + v], np.int64)
    bc, fg, dc, tl = (np.asarray(E[f'{x}_{v}'], np.int64) for x in ('btc_c', 'fg_c', 'd_c', 'tl_c'))
    if cn == 'btc_with':
        grp = np.select([bc == 0, bc == 2], [1, 0], -1)
    elif cn == 'btc_against':
        grp = np.select([bc == 1, bc == 2], [1, 0], -1)
    elif cn == 'fair_gap':
        grp = np.select([(fg == 2) | (fg == 3), fg == 0], [1, 0], -1)
    elif cn == 'depth_own':
        grp = np.select([(dc == 0) | (dc == 1), dc == 3], [1, 0], -1)
    elif cn == 'time_left':
        grp = np.select([tl >= 5, (tl == 2) | (tl == 3)], [1, 0], -1)
    else:
        raise ValueError(f'unknown contrast {cn!r}')
    grp = np.where(st == 0, grp, -1).astype(np.int64)
    base = (np.asarray(E['mi'], np.int64) * 2 + np.asarray(E['side'], np.int64)) * 3 + np.asarray(E['dbin'], np.int64)
    if cn == 'time_left':
        strat, S, per = base, 6 * n_mk, 6
    else:
        strat, S, per = base * 7 + tl, 42 * n_mk, 42
    need(bool(((strat >= 0) & (strat < S))[grp >= 0].all()), f'{cn}: a stratum outside its range (refused)')
    return grp, strat.astype(np.int64), int(S), int(per)


def mh_rows(a, b, T1, T0):
    """Mantel-Haenszel person-time RR per row of (R, S) arrays (each row's strata with T1 > 0 and T0 > 0); +inf exposed events without reference
    weight, NaN none (take_logic.mh_rr's conventions; person-time may differ per row)."""
    a, b, T1, T0 = (np.atleast_2d(np.asarray(x, np.float64)) for x in (a, b, T1, T0))
    inf_ = (T1 > 0) & (T0 > 0)
    Ts = np.where(inf_, T1 + T0, 1.)
    num = np.where(inf_, a * T0 / Ts, 0.).sum(1)
    den = np.where(inf_, b * T1 / Ts, 0.).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        rr = np.where(den > 0, num / np.where(den > 0, den, 1.), np.where(num > 0, INF, NAN))
    return rr, num, den


def perm_labels(st, g1, rng, R):
    """R draws of the exposure labels g1 permuted within the strata st (uniform within each stratum): random keys sorted inside the stratum
    blocks receive the block's labels -> (R, m) bool."""
    st = np.asarray(st, np.int64)
    m = int(len(st))
    lab = np.asarray(g1, bool)[np.argsort(st, kind='stable')]
    U = rng.random((R, m))
    order = np.argsort(st.astype(np.float64)[None, :] + U, axis=1)
    L = np.zeros((R, m), bool)
    np.put_along_axis(L, order, np.broadcast_to(lab, (R, m)), axis=1)
    return L


def perm_null(grp, strat, ev, T, S, rng, n_perm=N_PERM, chunk=PERM_CHUNK):
    """The declared null: the exposure label permuted across the episodes of each stratum (strata holding both labels; the others cannot change),
    n_perm draws in chunks from one Generator stream -> null MH ratios and the permutation info."""
    sel = np.asarray(grp) >= 0
    st, g1, e, tt = np.asarray(strat)[sel], np.asarray(grp)[sel] == 1, np.asarray(ev, np.float64)[sel], np.asarray(T, np.float64)[sel]
    n1, n0 = np.bincount(st[g1], minlength=S), np.bincount(st[~g1], minlength=S)
    mixed = (n1 > 0) & (n0 > 0)
    k = mixed[st]
    st, g1, e, tt = st[k], g1[k], e[k], tt[k]
    m = int(len(st))
    info = dict(n_mixed_strata=int(mixed.sum()), n_permuted=m)
    out = np.full(n_perm, NAN)
    if m == 0:
        return out, info
    Es, Tt = np.bincount(st, weights=e, minlength=S), np.bincount(st, weights=tt, minlength=S)
    for r0 in range(0, n_perm, chunk):
        R = min(chunk, n_perm - r0)
        L = perm_labels(st, g1, rng, R)
        rid = np.arange(R, dtype=np.int64)[:, None] * S + st[None, :]
        a = np.bincount(rid[L], weights=np.broadcast_to(e, (R, m))[L], minlength=R * S).reshape(R, S)
        T1 = np.bincount(rid[L], weights=np.broadcast_to(tt, (R, m))[L], minlength=R * S).reshape(R, S)
        out[r0:r0 + R] = mh_rows(a, Es[None, :] - a, T1, Tt[None, :] - T1)[0]
    return out, info


def _same(x, y):
    x, y = float(x), float(y)
    return (np.isnan(x) and np.isnan(y)) or x == y or abs(x - y) <= 1e-12 * max(abs(x), abs(y), 1e-300)


def ratio_block(E, vi, v, ci, cn, exl, rfl, markets, days):
    """One (variant, contrast): the pooled MH ratio of response hazards (responses / exposure-seconds), its within-stratum permutation null, the
    within-market MH / crude ratio per market, day readings and posting_opportunity's verdict."""
    n_mk = len(markets)
    grp, strat, S, per = contrast_code(E, v, cn, n_mk)
    ev, T = np.asarray(E['resp'], np.float64), np.asarray(E['expo_ms'], np.float64)
    g1, g0 = grp == 1, grp == 0
    a = np.bincount(strat[g1], weights=ev[g1], minlength=S)
    b = np.bincount(strat[g0], weights=ev[g0], minlength=S)
    T1 = np.bincount(strat[g1], weights=T[g1], minlength=S)
    T0 = np.bincount(strat[g0], weights=T[g0], minlength=S)
    o = float(mh_rows(a, b, T1, T0)[0][0])
    need(_same(o, TL.mh_rr(a[None, :], b[None, :], T1, T0)[0][0]), f'{v} {cn}: the MH ratio differs from take_logic.mh_rr (refused)')
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, vi, ci])))
    nl, pinfo = perm_null(grp, strat, ev, T, S, rng)
    c_hi, c_lo = TL.tail_counts(o, nl)
    inf_s = (T1 > 0) & (T0 > 0)
    n_ev = int(round(float((a + b)[inf_s].sum())))
    Ts = np.where(inf_s, T1 + T0, 1.)
    num_s, den_s = np.where(inf_s, a * T0 / Ts, 0.), np.where(inf_s, b * T1 / Ts, 0.)
    mof = np.arange(S) // per
    num_m, den_m = np.bincount(mof, weights=num_s, minlength=n_mk), np.bincount(mof, weights=den_s, minlength=n_mk)
    a_m, b_m = np.bincount(mof, weights=a, minlength=n_mk), np.bincount(mof, weights=b, minlength=n_mk)
    T1_m, T0_m = np.bincount(mof, weights=T1, minlength=n_mk) / 1000., np.bincount(mof, weights=T0, minlength=n_mk) / 1000.
    with np.errstate(divide='ignore', invalid='ignore'):
        rr_m = np.where(den_m > 0, num_m / np.where(den_m > 0, den_m, 1.), np.where(num_m > 0, INF, NAN))
        cr_m = np.where((b_m > 0) & (T1_m > 0) & (T0_m > 0), (a_m / np.where(T1_m > 0, T1_m, 1.)) / (b_m / np.where(T0_m > 0, T0_m, 1.)),
                        np.where((a_m > 0) & (T1_m > 0), INF, NAN))
    sup = (a_m + b_m >= 1) & (T1_m >= MIN_MKT_S) & (T0_m >= MIN_MKT_S)
    gt1, lt1, cgt1 = sup & _gt1(rr_m), sup & _lt1(rr_m), sup & _gt1(cr_m)
    n_sup, n_gt1, n_lt1 = int(sup.sum()), int(gt1.sum()), int(lt1.sum())
    a_t, b_t, T1t, T0t = float(a.sum()), float(b.sum()), float(T1.sum()) / 1000., float(T0.sum()) / 1000.
    crude = (a_t / T1t) / (b_t / T0t) if (T1t > 0 and T0t > 0 and b_t > 0) else (INF if (a_t > 0 and T1t > 0) else NAN)
    fin = nl[np.isfinite(nl)]
    qn = np.quantile(fin, [.05, .5, .95]) if len(fin) else [NAN] * 3
    vr = PO.verdict(o, c_hi, c_lo, n_ev, n_sup, n_gt1, n_lt1)
    sig_hi, sig_lo = c_hi is not None and int(c_hi) <= C_MAX, c_lo is not None and int(c_lo) <= C_MAX
    st = np.asarray(E['st_' + v], np.int64)
    row = dict(variant=v, contrast=cn, exposed=exl, reference=rfl, state=VDESC[v],
               strata='market x side x delta bin' if cn == 'time_left' else 'market x side x delta bin x time-left bin',
               population='retreat episodes with the variant state available', RR_MH=o, RR_crude=float(crude), a_exposed=a_t, T_exposed_s=T1t,
               b_reference=b_t, T_reference_s=T0t, n_ep_exposed=int(g1.sum()), n_ep_reference=int(g0.sum()), n_strata=int(S),
               n_strata_informative=int(inf_s.sum()), n_events_informative=n_ev, ep_state_missing=int((st != 0).sum()),
               ep_other_bins=int(((st == 0) & (grp < 0)).sum()), null_n=int(len(nl)), null_n_nan=int(np.isnan(nl).sum()),
               null_n_inf=int(np.isinf(nl).sum()), null_mean=float(fin.mean()) if len(fin) else NAN, null_p05=float(qn[0]), null_p50=float(qn[1]),
               null_p95=float(qn[2]), null_mixed_strata=pinfo['n_mixed_strata'], null_permuted_episodes=pinfo['n_permuted'],
               c_hi=-1 if c_hi is None else int(c_hi), c_lo=-1 if c_lo is None else int(c_lo), p_hi=NAN if c_hi is None else c_hi / (N_PERM + 1),
               p_lo=NAN if c_lo is None else c_lo / (N_PERM + 1), n_markets_supported=n_sup, n_markets_rr_gt1=n_gt1, n_markets_rr_lt1=n_lt1,
               share_rr_gt1=(n_gt1 / n_sup) if n_sup else NAN, share_rr_lt1=(n_lt1 / n_sup) if n_sup else NAN, n_markets_crude_gt1=int(cgt1.sum()),
               inconsistent_hi=bool(sig_hi and _gt1(o) and not (SHARE_DEN * n_gt1 >= SHARE_NUM * n_sup)),
               inconsistent_lo=bool(sig_lo and _lt1(o) and not (SHARE_DEN * n_lt1 >= SHARE_NUM * n_sup)), verdict_variant=vr, verdict='')
    cons = []
    for mi, (dd, mm) in enumerate(markets):
        cons.append(dict(row='market', variant=v, contrast=cn, day=dd, mid=mm, a=float(a_m[mi]), T1_s=float(T1_m[mi]), b=float(b_m[mi]),
                         T0_s=float(T0_m[mi]), rr_mh=float(rr_m[mi]), rr_crude=float(cr_m[mi]), supported=bool(sup[mi]), rr_gt1=bool(gt1[mi]),
                         rr_lt1=bool(lt1[mi]), crude_gt1=bool(cgt1[mi])))
    mday = np.array([days.index(dd) for dd, _ in markets], np.int64)
    for di, dd in enumerate(days):
        md = mday == di
        sm = md[mof]
        nm_, dn_ = float(num_s[sm].sum()), float(den_s[sm].sum())
        ns, ng, nl_ = int(sup[md].sum()), int(gt1[md].sum()), int(lt1[md].sum())
        cons.append(dict(row='day', variant=v, contrast=cn, day=dd, mid=-1, a=float(a_m[md].sum()), T1_s=float(T1_m[md].sum()), b=float(b_m[md].sum()),
                         T0_s=float(T0_m[md].sum()), rr_mh=nm_ / dn_ if dn_ > 0 else (INF if nm_ > 0 else NAN), n_supported=ns, n_rr_gt1=ng,
                         n_rr_lt1=nl_, share_gt1=(ng / ns) if ns else NAN))
    cons.append(dict(row='day', variant=v, contrast=cn, day='all', mid=-1, a=a_t, T1_s=T1t, b=b_t, T0_s=T0t, rr_mh=o, n_supported=n_sup,
                     n_rr_gt1=n_gt1, n_rr_lt1=n_lt1, share_gt1=(n_gt1 / n_sup) if n_sup else NAN))
    return row, cons


def ratio_tables(E, markets, days):
    RK, cons = {}, []
    for vi, v in enumerate(VARIANTS):
        for ci, (cn, exl, rfl) in enumerate(CONTRASTS):
            r, c_ = ratio_block(E, vi, v, ci, cn, exl, rfl, markets, days)
            RK[(v, cn)] = r
            cons.extend(c_)
    FIN = {}
    for cn in CN_NAMES:
        v0, v1 = RK[('D0', cn)]['verdict_variant'], RK[('LAG', cn)]['verdict_variant']
        FIN[cn] = v0 if v0 == v1 else 'timing_sensitive'
        RK[('D0', cn)]['verdict'] = RK[('LAG', cn)]['verdict'] = FIN[cn]
    return RK, cons, FIN


# ------------------------------------------------------------------ section 1 tables ------------------------------------------------------------------
def census_table(PR, days, markets):
    """census.csv: per scope (pooled / day / market) x class (+ the two_side and unknown sub-labels): frame pairs and the creations they hold."""
    lab = np.where(PR['in_window'].to_numpy(bool), np.array(CLASSES, dtype=object)[PR['cls'].to_numpy(np.int64)], 'outside_window') if len(PR) else np.zeros(0, object)
    X = PR.assign(census_class=lab)
    rows = []
    scopes = [('pooled', 'all', -1, X)] + [('day', d, -1, X[(X['day'] == d).to_numpy()]) for d in days] + \
             [('market', d, m, X[(X['day'] == d).to_numpy() & (X['mid'] == m).to_numpy()]) for d, m in markets]
    for sc, d, m, Y in scopes:
        cc = Y['census_class'].to_numpy(dtype=object)
        sb = Y['sub'].to_numpy(dtype=object)
        dt = Y['dt_ms'].to_numpy(np.int64)
        V = {c: Y[c].to_numpy(np.int64) for c in CEN_SUM}
        for c_ in CENSUS_CLASSES:
            subs = [('all', cc == c_)]
            if c_ == 'two_side':
                subs += [(x, (cc == c_) & (sb == x)) for x in TWO_SUB]
            elif c_ == 'unknown':
                subs += [(x, (cc == c_) & (sb == x)) for x in UNK_SUB]
            for sl, mk in subs:
                r = dict(scope=sc, day=d, mid=m, cls=c_, sublabel=sl, n_pairs=int(mk.sum()), n_pairs_zero_interval=int((mk & (dt == 0)).sum()),
                         dt_ms_sum=int(dt[mk].sum()), dt_ms_p50=float(np.median(dt[mk])) if mk.any() else NAN,
                         n_pairs_with_creation=int((mk & (V['n_cre'] > 0)).sum()))
                for c in CEN_SUM:
                    r[c] = int(V[c][mk].sum())
                rows.append(r)
    return pd.DataFrame(rows)


def two_side_table(TS, days):
    """two_side.csv: per scope x sub-label, the pair count and creations, then each state feature's distribution (shares within the sub-label)."""
    feats = ('spread_bin', 'btc_5s_dir', 'time_left_bin', 'depth_bid_bin', 'depth_ask_bin', 'move_bid_ticks', 'move_ask_ticks')
    rows = []
    for sc, d in [('pooled', 'all')] + [('day', x) for x in days]:
        X = TS if sc == 'pooled' else TS[(TS['day'] == d).to_numpy()]
        for sl in TWO_SUB + ('all',):
            Y = X if sl == 'all' else X[(X['sublabel'] == sl).to_numpy()]
            nY = int(len(Y))
            rows.append(dict(scope=sc, day=d, sublabel=sl, feature='total', value='', n_pairs=nY, share=1. if nY else NAN, n_cre=int(Y['n_cre'].sum()),
                             n_cre_improve=int(Y['n_cre_improve'].sum()), n_cre_deep=int(Y['n_cre_deep'].sum())))
            for ft in feats:
                vals = sorted(set(TS[ft].astype(str).tolist())) if len(TS) else []
                for val in vals:
                    Z = Y[(Y[ft].astype(str) == val).to_numpy()]
                    rows.append(dict(scope=sc, day=d, sublabel=sl, feature=ft, value=val, n_pairs=int(len(Z)), share=len(Z) / nY if nY else NAN,
                                     n_cre=int(Z['n_cre'].sum()), n_cre_improve=int(Z['n_cre_improve'].sum()), n_cre_deep=int(Z['n_cre_deep'].sum())))
    return pd.DataFrame(rows, columns=['scope', 'day', 'sublabel', 'feature', 'value', 'n_pairs', 'share', 'n_cre', 'n_cre_improve', 'n_cre_deep'])


# ------------------------------------------------------------------ section 3 tables: continuations ------------------------------------------------------------------
def flat_views(Efv):
    """chains_v3.views (10 s / 60 s / END, the shared valid cohort, the resolution view) flattened into columns."""
    V = CH3.views(Efv)
    out = {}
    for h in VIEW_H:
        for k in ('n_valid', 'n_all', 'sh_valid', 'sh_all', 'gross', 'net', 'c_sh'):
            out[f'v{h}_{k}'] = V[h][k]
    out['common_n'], out['common_sh'] = V['common']['n_valid'], V['common']['sh_valid']
    for h in VIEW_H:
        out[f'common_c_sh_{h}'] = V['common'][f'c_sh_{h}']
    out['res_n_valid'], out['res_sh_valid'], out['res_net'], out['res_c_sh'] = V['res']['n_valid'], V['res']['sh_valid'], V['res']['net'], V['res']['c_sh']
    ok = Efv['status'].astype(str).to_numpy(dtype=object) == 'ok' if len(Efv) else np.zeros(0, bool)
    out['turnover_mq_sh'] = float(Efv['mq'].to_numpy(np.float64)[ok].sum()) if len(Efv) else 0.
    out['residual_rq_sh'] = float(Efv['rq'].to_numpy(np.float64)[ok].sum()) if len(Efv) else 0.
    out['fv_rows'], out['fv_rows_status_ok'] = int(len(Efv)), int(ok.sum())
    return out


def _fv_of(FV, lg):
    if FV is None or not len(FV) or not len(lg):
        return FV.iloc[0:0] if FV is not None else pd.DataFrame({c: [] for c in FVV_READ})
    kl = pd.MultiIndex.from_arrays([lg['mid'].to_numpy(np.int64), lg['sid'].astype(str).to_numpy(dtype=object), lg['mk_idx'].to_numpy(np.int64)])
    kd = pd.MultiIndex.from_arrays([FV['mid'].to_numpy(np.int64), FV['sid'].astype(str).to_numpy(dtype=object), FV['mk_idx'].to_numpy(np.int64)])
    return FV[kd.isin(kl)]


def continuations(EPD, LG, CI):
    """continuations.csv: table 'order' = one row per DISTINCT candidate order linked to a response (MT_TT_MUT=split_same_hash: one per fill
    record, the planted error); table 'summary' = per scope (all / rank1-3 / top3 / other) owner counts over every response (unknown in the
    denominator) and the pooled continuation of the scope's distinct orders (each leg once)."""
    mut = mutant()
    PF, TK, FV = CI.get('PF'), CI.get('TK'), CI.get('FV')
    R = EPD[(EPD['end_type'] == 'response').to_numpy()] if len(EPD) else EPD
    L = R[R['identity'].isin(LINKED).to_numpy()] if len(R) else R
    rows, orders = [], []
    pfk = {}
    if PF is not None and len(PF):
        pfk = {(str(d), int(m), str(h)): r for d, m, h, r in zip(PF['day'].tolist(), PF['mid'].tolist(), PF['mk_hash'].tolist(), PF.to_dict('records'))}
    if len(L):
        for (d, m, h), X in L.groupby(['day', 'mid', 'cand_hash'], sort=True):
            lg = LG[(LG['day'] == d).to_numpy() & (LG['mid'] == m).to_numpy() & (LG['mk_hash'] == h).to_numpy()] if len(LG) else LG
            units = [('', lg)]
            if mut == 'split_same_hash' and len(lg):     # planted error: each later same-hash fill record counted as another order
                units = [(sid, lg[(lg['sid'] == sid).to_numpy()]) for sid in sorted(set(lg['sid'].tolist()))]
            addr = str(X['cand_addr'].iloc[0])
            grp = str(X['cand_group'].iloc[0])
            t_c = int(X['resp_hi_ms'].min())
            t_o = int(X['resp_lo_ms'].min())
            for usid, lu in units:
                Efv = _fv_of(FV, lu)
                r = dict(table='order', day=str(d), mid=int(m), cand_hash=str(h), unit_sid=usid, cand_addr=addr, cand_group=grp,
                         n_responses_linked=int(len(X)), response_keys=';'.join(f'{d}_{m}_{int(e)}' for e in X['episode_id'].tolist()),
                         identities=';'.join(sorted(set(X['identity'].astype(str)))), grades=';'.join(sorted(set(X['grade'].astype(str)))),
                         pre_state_links=';'.join(sorted(set(X['pre_state_link'].astype(str)))), side=';'.join(sorted(set(X['side'].astype(str)))),
                         price_ticks=';'.join(str(int(x)) for x in sorted(set(X['resp_tick'].tolist()))),
                         shown_size_at_creation=';'.join(f'{float(x):g}' for x in X['resp_size_shown'].tolist()),
                         creation_lo_ms=t_o, creation_hi_ms=t_c, win_lo_ms=int(X['win_lo_ms'].min()), win_hi_ms=int(X['win_hi_ms'].max()),
                         alt_netting=';'.join(sorted(set(X['alt_netting'].astype(str)))), alt_first_other=';'.join(sorted(set(X['alt_first_other'].astype(str)))),
                         alt_first_addresses='|'.join(sorted({a for x in X['alt_first_addresses'].astype(str) for a in x.split('|') if a})),
                         n_legs=int(len(lu)), n_records=int(lu['sid'].nunique()) if len(lu) else 0, filled_shares=float(lu['shares'].sum()) if len(lu) else 0.,
                         first_fill_hi_ms=int(lu['hi_ms'].min()) if len(lu) else -1, last_fill_hi_ms=int(lu['hi_ms'].max()) if len(lu) else -1,
                         n_legs_hi_before_creation=int((lu['hi_ms'].to_numpy(np.int64) <= t_o).sum()) if len(lu) else 0)
                r.update(flat_views(Efv))
                pr = pfk.get((str(d), int(m), str(h)))
                if pr is None:
                    r.update(successor_status='not_evaluated' if addr not in TOP3_L else 'top3_order_absent', size_status='', union_histories='',
                             next_kind='', next_link_status='', next_hash='', n_children=NAN, pf_n_fills=NAN, pf_mq_sh=NAN, pf_rq_sh=NAN)
                else:
                    r.update(successor_status='evaluated', size_status=str(pr['size_status']), union_histories=str(pr['union_histories']),
                             next_kind=str(pr['next_kind']), next_link_status=str(pr['next_link_status']), next_hash=str(pr['next_hash']),
                             n_children=float(pr['n_children']), pf_n_fills=float(pr['n_fills']), pf_mq_sh=float(pr['mq_sh']), pf_rq_sh=float(pr['rq_sh']))
                tkeys = {hn: set() for hn in TAKE_KEYS}           # take record keys (day|mid|addr|sid): the pooled tables count each take ONCE
                if addr in TOP3_L and TK is not None:
                    T_ = TK[(TK['day'] == str(d)).to_numpy() & (TK['mid'] == int(m)).to_numpy() & (TK['addr_l'] == addr).to_numpy()]
                    xl, hi = T_['x_lo_hard_ms'].to_numpy(np.int64), T_['hi_ms'].to_numpy(np.int64)
                    kk = [f'{d}|{int(m)}|{addr}|{x}' for x in T_['sid'].astype(str).tolist()]
                    aft = xl > t_c
                    r.update(takes_status='evaluated', takes_unbounded_after=int(((xl < 0) & (hi > t_c)).sum()), takes_end=int(aft.sum()),
                             takes_end_abs_dq=float(np.abs(T_['dq'].to_numpy(np.float64))[aft].sum()),
                             takes_end_dir_D0=';'.join(f'{k}:{v}' for k, v in sorted(Counter(T_['dir_D0'].astype(str).to_numpy(dtype=object)[aft].tolist()).items())))
                    tkeys['end'].update(kk[j] for j in np.flatnonzero(aft).tolist())
                    for hn, hm in TAKE_H:
                        sel_ = aft & (hi <= t_c + hm)
                        r[f'takes_{hn}'] = int(sel_.sum())
                        tkeys[hn].update(kk[j] for j in np.flatnonzero(sel_).tolist())
                else:
                    r.update(takes_status='not_evaluated', takes_unbounded_after=-1, takes_end=-1, takes_end_abs_dq=NAN, takes_end_dir_D0='')
                    for hn, _ in TAKE_H:
                        r[f'takes_{hn}'] = -1
                rows.append(r)
                orders.append((r, Efv, tkeys))
    OR = pd.DataFrame(rows)
    SU = []
    n_all = int(len(R))
    for sc in ('all',) + GROUPS + ('top3',):
        Rs = R if sc == 'all' else (R[R['cand_group'].isin(GROUPS[:3]).to_numpy()] if sc == 'top3' else R[(R['cand_group'] == sc).to_numpy()])
        if sc != 'all':
            Rs = Rs[Rs['identity'].isin(LINKED).to_numpy()]
        osel = [(r, e) for r, e, _ in orders if sc == 'all' or (r['cand_group'] in GROUPS[:3] if sc == 'top3' else r['cand_group'] == sc)]
        tsel = [tk for r, _, tk in orders if sc == 'all' or (r['cand_group'] in GROUPS[:3] if sc == 'top3' else r['cand_group'] == sc)]
        met = dict(n_responses=int(len(Rs)), n_responses_all=n_all, share_of_all_responses=len(Rs) / n_all if n_all else NAN)
        for idn in IDENT:
            met[f'n_{idn}'] = int((Rs['identity'] == idn).sum()) if len(Rs) else 0
        for ps in PSL:
            met[f'pre_state_link:{ps}'] = int((Rs['pre_state_link'] == ps).sum()) if len(Rs) else 0
        for k_ in ('alt_netting', 'alt_first_other'):
            for a_ in ALT_ST:
                met[f'{k_}:{a_}'] = int(((Rs['identity'].isin(LINKED)) & (Rs[k_] == a_)).sum()) if len(Rs) else 0
        met['n_orders'] = int(len(osel))
        met['n_orders_linked_to_several_responses'] = int(sum(1 for r, _ in osel if r['n_responses_linked'] >= 2))
        met['n_responses_in_shared_orders'] = int(sum(r['n_responses_linked'] for r, _ in osel if r['n_responses_linked'] >= 2))
        met['n_legs'] = int(sum(r['n_legs'] for r, _ in osel))
        met['filled_shares'] = float(sum(r['filled_shares'] for r, _ in osel))
        Ep = pd.concat([e for _, e in osel], ignore_index=True) if osel else (FV.iloc[0:0] if FV is not None else pd.DataFrame({c: [] for c in FVV_READ}))
        met.update({f'pooled_{k}': v for k, v in flat_views(Ep).items()})
        for kk in ('next_kind', 'successor_status', 'takes_status'):
            for val, c_ in sorted(Counter(str(r[kk]) for r, _ in osel).items()):
                met[f'{kk}:{val}'] = int(c_)
        for hn in TAKE_KEYS:                                # attribution overlap: one take after several linked creations counts once
            osum = int(sum(max(int(r[f'takes_{hn}']), 0) for r, _ in osel))
            dist = len(set().union(*[tk[hn] for tk in tsel])) if tsel else 0
            met[f'takes_{hn}_distinct'], met[f'takes_{hn}_order_sum'], met[f'takes_{hn}_overlap'] = dist, osum, osum - dist
        for k_, v_ in met.items():
            SU.append(dict(table='summary', scope=sc, metric=k_, value=v_))
    return OR, pd.DataFrame(SU, columns=['table', 'scope', 'metric', 'value'])


# ------------------------------------------------------------------ claims ------------------------------------------------------------------
def recon_claims(kind, PR, RC, EPD, RW_all):
    """Computed reconciliation predicates and counts (section 1): every improve creation (and every anchor-excluded one) -> one frame pair with
    its class or an explicit unsupported status; the anchor-mask recount from rows.parquet."""
    imp = RC[(RC['po_status'] == 'event').to_numpy() & RC['is_improve'].to_numpy(bool)] if len(RC) else RC
    exc = imp[imp['anchor_excluded'].to_numpy(bool)] if len(imp) else imp
    ok_cls = True
    if len(imp):
        sd = imp['side'].to_numpy(dtype=object)
        cl_ = imp['cls'].to_numpy(dtype=object)
        inc = imp['pair_status'].to_numpy(dtype=object) == 'in_census'
        allowed = np.where(sd == 'bid', 'improve_bid', 'improve_ask')
        ok_cls = bool((~inc | (cl_ == allowed) | (cl_ == 'two_side') | (cl_ == 'unknown')).all())
    pr_imp = int(PR['n_improve_event'][PR['in_window'].to_numpy(bool)].sum()) if len(PR) else 0
    out = dict(creation_rows=int(len(RC)), improve_events=int(len(imp)), anchor_excluded=int(len(exc)),
               improve_by_status=dict(Counter(imp['recon_status'].astype(str))) if len(imp) else {},
               improve_by_role=dict(Counter(imp['episode_role'].astype(str))) if len(imp) else {},
               anchor_excluded_by_status=dict(Counter(exc['recon_status'].astype(str))) if len(exc) else {},
               anchor_excluded_by_role=dict(Counter(exc['episode_role'].astype(str))) if len(exc) else {},
               anchor_in_by_status=dict(Counter(imp.loc[(imp['anchor_population'] == 'in').to_numpy(), 'recon_status'].astype(str))) if len(imp) else {},
               improve_rows_without_anchor_row=int((imp['anchor_population'] == 'not_applicable').sum()) if len(imp) else 0)
    out['predicates'] = dict(every_improve_event_one_row=bool(imp['key'].is_unique) if len(imp) else True,
                             every_improve_event_has_pair_or_status=bool((imp['recon_status'].astype(str).str.len() > 0).all()) if len(imp) else True,
                             improve_classes_consistent=ok_cls,
                             improve_events_in_census_equal_pair_counts=pr_imp == int((imp['pair_status'] == 'in_census').sum()) if len(imp) else pr_imp == 0,
                             anchor_excluded_subset_of_improve=bool(exc['is_improve'].all()) if len(exc) else True,
                             response_new_best_rows_le_responses=int((RC['episode_role'] == 'response_new_best').sum()) <= int((EPD['end_type'] == 'response').sum())
                             if len(RC) and len(EPD) else True)
    out['predicates']['all'] = all(bool(v) for v in out['predicates'].values())
    if RW_all is not None and len(RW_all):
        ei = RW_all['E_improve'].to_numpy(np.int64) > 0
        out['anchor_mask_recount'] = dict(improve_event_seconds=int(ei.sum()), excluded_event_seconds=int((ei & ~RW_all['improve_population'].to_numpy(bool)).sum()))
    if kind == 'run':
        out['review_counts'] = dict(improve_events_eq_11804=out['improve_events'] == REVIEW_COUNTS['improve_events'],
                                    anchor_excluded_eq_8821=out['anchor_excluded'] == REVIEW_COUNTS['anchor_excluded'],
                                    event_seconds_eq_4484=(out.get('anchor_mask_recount') or {}).get('improve_event_seconds') == REVIEW_COUNTS['improve_event_seconds'],
                                    excluded_seconds_eq_3664=(out.get('anchor_mask_recount') or {}).get('excluded_event_seconds') == REVIEW_COUNTS['anchor_excluded_event_seconds'])
    return out


def episode_claims(EPD):
    """The termination census, the response-type mix and the time-to-response distribution (computed from the episode rows)."""
    n = int(len(EPD))
    if n == 0:
        return dict(n_episodes=0, termination={}, predicates=dict(all=True))
    et, rs = EPD['end_type'].astype(str), EPD['end_reason'].astype(str)
    ex, lo, hi = (EPD[c].to_numpy(np.int64) for c in ('exposure_ms', 'exposure_lo_ms', 'exposure_hi_ms'))
    term = {}
    ec = np.array([x or 'none' for x in EPD['end_class'].astype(str).tolist()], dtype=object)     # the class of the pair that ended it
    for e_ in END_TYPES:
        m = (et == e_).to_numpy()
        term[e_] = dict(n=int(m.sum()), exposure_s=float(ex[m].sum()) / 1000., by_reason=dict(Counter(rs[m].tolist())),
                        by_end_class=dict(Counter(ec[m].tolist())))
    R = EPD[(et == 'response').to_numpy()]
    mix = {}
    for sc, X in [('all', R)] + [(f'side:{x}', R[(R['side'] == x).to_numpy()]) for x in SIDES] + [(f'delta:{x}', R[(R['delta_bin'] == x).to_numpy()]) for x in DELTA_LAB]:
        mix[sc] = {x: int((X['resp_type'] == x).sum()) for x in RT_LAB}
    ttr = _qd(R['exposure_ms'].to_numpy(np.float64), 'exposure_ms')
    ttr.update(_qd(R['exposure_lo_ms'].to_numpy(np.float64), 'lower_bound_ms'))
    ttr['no_frame_between'] = int(R['no_frame_between'].astype(bool).sum())
    pred = dict(one_end_type_each=int(sum(v['n'] for v in term.values())) == n, exposure_nonnegative=bool((ex >= 0).all()),
                bounds_ordered=bool(((lo <= hi) & (lo >= 0)).all()), exposure_le_horizon=bool((ex <= HORIZON_MS).all()),
                responses_typed=bool(R['resp_type'].isin(RT_LAB).all()) if len(R) else True)
    pred['all'] = all(bool(v) for v in pred.values())
    return dict(n_episodes=n, by_side={x: int((EPD['side'] == x).sum()) for x in SIDES}, by_delta={x: int((EPD['delta_bin'] == x).sum()) for x in DELTA_LAB},
                termination=term, exposure_s_total=float(ex.sum()) / 1000., responses=int(len(R)),
                hazard_per_s_crude=(len(R) / (float(ex.sum()) / 1000.)) if ex.sum() > 0 else NAN, response_mix=mix, time_to_response=ttr,
                after_horizon_class_of_no_response=dict(Counter(EPD.loc[(et == 'no_response_1s').to_numpy(), 'after_horizon_class'].astype(str).tolist())),
                state_status_LAG=dict(Counter(EPD['state_status_LAG'].astype(str).tolist())), predicates=pred)


def owner_claims(EPD, SU):
    R = EPD[(EPD['end_type'] == 'response').to_numpy()] if len(EPD) else EPD
    lk = R[R['identity'].isin(LINKED).to_numpy()] if len(R) else R
    al = SU[(SU['scope'] == 'all').to_numpy()] if len(SU) else SU
    return dict(responses=int(len(R)), by_identity={x: int((R['identity'] == x).sum()) for x in IDENT} if len(R) else {},
                linked_by_group={g: int((lk['cand_group'] == g).sum()) for g in GROUPS} if len(lk) else {},
                linked_by_grade=dict(Counter(lk['grade'].astype(str).tolist())) if len(lk) else {},
                unknown_reasons=dict(Counter(R.loc[(R['identity'] == 'owner_unknown').to_numpy(), 'identity_reason'].astype(str).tolist())) if len(R) else {},
                pre_state_link=dict(Counter(R['pre_state_link'].astype(str).tolist())) if len(R) else {},
                alt_netting_linked=dict(Counter(lk['alt_netting'].astype(str).tolist())) if len(lk) else {},
                alt_first_other_linked=dict(Counter(lk['alt_first_other'].astype(str).tolist())) if len(lk) else {},
                po_parity=dict(Counter(R['po_parity'].astype(str).tolist())) if len(R) else {},
                summary_all={str(r['metric']): r['value'] for r in al.to_dict('records')},
                reading='owner_unknown stays in every denominator and is never a maker WAIT; a candidate link carries its grade and is never the '
                        'identified creator; continuations are recorded rows of the candidate order, not an identified maker policy')


def make_claims(kind, days, markets, PR, RC, EPD, TS, RK, FIN, SU, OR, recon, epc, cnt, dependency):
    cen = {}
    if len(PR):
        lab = np.where(PR['in_window'].to_numpy(bool), np.array(CLASSES, dtype=object)[PR['cls'].to_numpy(np.int64)], 'outside_window')
        cen = {c: int((lab == c).sum()) for c in CENSUS_CLASSES}
    ratios = []
    for cn in CN_NAMES:
        ent = dict(contrast=cn, verdict=FIN[cn])
        for v in VARIANTS:
            r = RK[(v, cn)]
            ent[v] = dict(RR_MH=r['RR_MH'], RR_crude=r['RR_crude'], p_hi=r['p_hi'], p_lo=r['p_lo'], n_events_informative=r['n_events_informative'],
                          T_exposed_s=r['T_exposed_s'], T_reference_s=r['T_reference_s'], n_markets_supported=r['n_markets_supported'],
                          n_markets_rr_gt1=r['n_markets_rr_gt1'], n_markets_rr_lt1=r['n_markets_rr_lt1'], verdict=r['verdict_variant'],
                          exposed=r['exposed'], reference=r['reference'],
                          predicates=dict(rr_gt_1=bool(_gt1(r['RR_MH'])), rr_lt_1=bool(_lt1(r['RR_MH'])), p_hi_le_0_05=bool(0 <= r['c_hi'] <= C_MAX),
                                          p_lo_le_0_05=bool(0 <= r['c_lo'] <= C_MAX), supported_markets_ge_5=bool(r['n_markets_supported'] >= MIN_MARKETS),
                                          events_ge_30=bool(r['n_events_informative'] >= MIN_EVENTS),
                                          inconsistent_hi=r['inconsistent_hi'], inconsistent_lo=r['inconsistent_lo']))
        ent['flips_at_LAG'] = ent['D0']['verdict'] != ent['LAG']['verdict']
        ratios.append(ent)
    lists = {vd: [e['contrast'] for e in ratios if e['verdict'] == vd] for vd in VERDICTS}
    ts = {}
    if len(TS):
        ts = dict(pairs=int(len(TS)), by_sublabel={x: int((TS['sublabel'] == x).sum()) for x in TWO_SUB}, improve_creations=int(TS['n_cre_improve'].sum()))
    return dict(version=VERSION, kind=kind, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, sections='1-4'),
                review=dict(file=_rel(REVIEW_FILE), commit=REVIEW_COMMIT), days=list(days), markets=[f'{d}_{m}' for d, m in markets],
                census=dict(pooled_by_class=cen, pairs=int(len(PR)), zero_interval_pairs=int((PR['dt_ms'] == 0).sum()) if len(PR) else 0,
                            population='pairs wholly inside [ws, we]; others outside_window'),
                reconciliation=recon, episodes=epc, ratios=ratios,
                verdict_lists=dict(supported_associations=lists['supported'], supported_lower_contradicting_the_exposure=lists['supported_lower'],
                                   not_supported=lists['not_supported'], insufficient_support=lists['insufficient_support'],
                                   timing_sensitive=lists['timing_sensitive']),
                two_side=ts, owners=owner_claims(EPD, SU), continuation_orders=int(len(OR)),
                null=dict(design='within-stratum permutation of the exposure label across episodes (strata holding both labels)', n_perm=N_PERM, seed=SEED,
                          generator='Generator(PCG64(SeedSequence([20261123, variant index (D0 0 / LAG 1), contrast index])))',
                          p='(1 + #{null >= / <= obs}) / 501; ties within 1e-12 relative; NaN replicates on both sides (take_logic.tail_counts)'),
                rules=dict(verdict='posting_opportunity.verdict: insufficient_support (< 5 supported markets or < 30 responses in informative strata) | '
                                   'supported (RR > 1, p_hi <= 0.05, RR > 1 in >= 2/3 of supported markets) | supported_lower (the mirror) | '
                                   'not_supported; timing_sensitive when the LAG verdict differs',
                           market='within-market MH over its strata; supported = >= 30 exposed and >= 30 reference exposure-seconds and >= 1 response',
                           horizon_ms=HORIZON_MS, gap_ms=GAP_MS, strata='market x side x delta bin (1 / 2 / >= 3) x time-left bin; time-left contrast: '
                           'market x side x delta bin', lag='LAG = the whole state at frame f - 2'),
                multiplicity=dict(contrasts=len(CN_NAMES), variants=len(VARIANTS), all_listed=True, selected=None),
                wording=dict(transitions='native frame pairs, net change only; never an invented intra-frame order',
                             episodes='an observed spread opening and the next observed touch transition; not an identified maker reaction time',
                             p_fair='the existing trailing BTC reference, not a maker belief',
                             re_establish='a recent cell reappearance with print-compatible clearing',
                             owners='a candidate link with its grade; owner_unknown in every denominator; never the identified creator',
                             days='four consumed days: point estimates, no intervals (G = 4 < 5)'),
                counts={k: int(v) for k, v in sorted(cnt.items())}, dependency=dependency, descriptive_only=True, intervals=None)


def digest(claims):
    """The pilot / run print: the key numbers (no verdict the tables do not carry)."""
    rec = claims['reconciliation']
    ep = claims['episodes']
    return dict(census=claims['census']['pooled_by_class'],
                reconciliation=dict(improve_events=rec['improve_events'], improve_by_status=rec['improve_by_status'], improve_by_role=rec['improve_by_role'],
                                    anchor_excluded=rec['anchor_excluded'], anchor_excluded_by_status=rec['anchor_excluded_by_status'],
                                    anchor_excluded_by_role=rec['anchor_excluded_by_role'], predicates=rec['predicates']['all']),
                episodes=dict(n=ep['n_episodes'], by_end_type={k: v['n'] for k, v in (ep.get('termination') or {}).items()},
                              by_reason={k: v['by_reason'] for k, v in (ep.get('termination') or {}).items()},
                              by_end_class={k: v['by_end_class'] for k, v in (ep.get('termination') or {}).items()},
                              response_mix=(ep.get('response_mix') or {}).get('all'),
                              ttr_median_ms=(ep.get('time_to_response') or {}).get('exposure_ms_p50'),
                              ttr_lower_median_ms=(ep.get('time_to_response') or {}).get('lower_bound_ms_p50'), predicates=(ep.get('predicates') or {}).get('all')),
                ratios={e['contrast']: dict(RR_D0=e['D0']['RR_MH'], p_hi=e['D0']['p_hi'], p_lo=e['D0']['p_lo'], n_ev=e['D0']['n_events_informative'],
                                            markets=f"{e['D0']['n_markets_rr_gt1']}>1 {e['D0']['n_markets_rr_lt1']}<1 /{e['D0']['n_markets_supported']}",
                                            RR_LAG=e['LAG']['RR_MH'], verdict=e['verdict']) for e in claims['ratios']},
                two_side=claims['two_side'], owners={k: claims['owners'][k] for k in ('responses', 'by_identity', 'linked_by_group', 'linked_by_grade',
                                                                                   'pre_state_link', 'po_parity')},
                continuation_orders=claims['continuation_orders'])


# ------------------------------------------------------------------ analysis ------------------------------------------------------------------
def analyse(kind, days, parts, cnt, CI, RW_all=None, dependency=None):
    """Parts -> (frames, tables, claims, digest)."""
    need(len(parts) > 0, 'no market part (refused)')
    markets = [(p['day'], p['mid']) for p in parts]
    need(len(set(markets)) == len(markets), 'a market twice in one analysis (refused)')
    days = sorted({d for d, _ in markets})
    EPD = episodes_frame(parts)
    E = ratio_inputs(parts)
    PR = pd.concat([p['pairs'] for p in parts], ignore_index=True)
    RC = pd.concat([p['reconcile'] for p in parts], ignore_index=True)
    TS = pd.concat([p['two_side'] for p in parts], ignore_index=True)
    LG = pd.concat([p['legs'] for p in parts], ignore_index=True)
    CEN = census_table(PR, days, markets)
    RK, cons, FIN = ratio_tables(E, markets, days)
    TST = two_side_table(TS, days)
    OR, SU = continuations(EPD, LG, CI)
    recon = recon_claims(kind, PR, RC, EPD, RW_all)
    epc = episode_claims(EPD)
    claims = make_claims(kind, days, markets, PR, RC, EPD, TS, RK, FIN, SU, OR, recon, epc, cnt, dependency)
    CONT = pd.concat([OR, SU], ignore_index=True) if len(OR) else SU
    tables = {'census.csv': CEN, 'reconcile.csv': RC, 'ratios.csv': pd.DataFrame([RK[(v, cn)] for cn in CN_NAMES for v in VARIANTS]),
              'consistency.csv': pd.DataFrame(cons), 'two_side.csv': TST, 'continuations.csv': CONT}
    return {'episodes.parquet': EPD}, tables, claims, digest(claims)


# ------------------------------------------------------------------ receipt, write-once ------------------------------------------------------------------
def receipt(kind, days, hashes, cnt, stages, t0, asof0, outs, git, extra=None):
    srcs = {_rel(f): _sha(f) for f in TOOL_FILES + DEP_FILES if Path(f).exists()}
    rec = dict(version=VERSION, kind=kind, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, sha256=srcs.get(_rel(CONTRACT_FILE))),
               review=dict(file=_rel(REVIEW_FILE), commit=REVIEW_COMMIT, sha256=srcs.get(_rel(REVIEW_FILE))), module=str(Path(__file__).resolve()),
               argv=list(sys.argv), run=C.RUN, mutant=os.environ.get(MUT_ENV, ''), asof_start=asof0, asof_end=_now(), days=list(days),
               addresses=list(TOP3), inputs=dict(sorted(hashes.items())), sources=srcs, outputs=outs, git=git,
               counts=dict(sorted((str(k), int(v)) for k, v in cnt.items())),
               constants=dict(HORIZON_MS=HORIZON_MS, GAP_MS=GAP_MS, CLASSES=CLASSES, TWO_SUB=TWO_SUB, UNK_SUB=UNK_SUB, END_TYPES=END_TYPES,
                              END_REASONS=END_REASONS, RT_LAB=RT_LAB, DELTA_LAB=DELTA_LAB, VARIANTS=VARIANTS, CONTRASTS=CONTRASTS, SEED=SEED, N_PERM=N_PERM,
                              MIN_EVENTS=MIN_EVENTS, MIN_MARKETS=MIN_MARKETS, MIN_MKT_S=MIN_MKT_S, C_MAX=C_MAX, TIE=TIE,
                              SHARE=f'{SHARE_NUM}/{SHARE_DEN}', THR_5S=THR_5S, G2=G2, PSL=PSL, MUTANTS=MUTANTS, PROJ_MAX_S=PROJ_MAX_S, FV_SHA256=FV_SHA256,
                              PILOT_RUN=PILOT_RUN, REVIEW_COUNTS=REVIEW_COUNTS),
               cpu_s=dict(by_stage=dict(stages), process_total=round(time.process_time(), 3)), wall_s=round(time.time() - t0, 2),
               vmhwm_mib=round(hwm(), 1), peak_rss='VmHWM of this process', guard_mib=C.GUARD_MIB)
    rec.update(extra or {})
    return rec


def run_core(kind, days, parts, out, hashes, cnt, stages, tick, git, asof0, t0, CI, RW_all=None, dependency=None, extra_fn=None):
    """Shared tail of the run / pilot (and the selftest's end-to-end): tables, ratios, nulls, continuations, claims, write-once."""
    for p in parts:
        cnt.update(p['counts'])
    frames, tables, claims, dg = analyse(kind, days, parts, cnt, CI, RW_all, dependency)
    tick('tables')
    guard('tables')

    def rec_fn(outs):
        tick('write')
        ex = dict(dependency=dependency)
        ex.update(extra_fn(stages) if extra_fn else {})
        return receipt(kind, days, hashes, cnt, stages, t0, asof0, outs, git, ex)
    rec = TL2.write_out(out, frames, tables, claims, rec_fn)
    return rec, claims, dg


def preflight_projection(path, hashes):
    """The full run launches only below the recorded pilot projection limit (<= 120 CPU s)."""
    path = Path(path)
    need(path.is_file(), f'no recorded pilot projection: {path} is missing (run the pilot first; refused before launch)')
    J = json.loads(path.read_text())
    pr = (J.get('projection') or {}).get('full_run_cpu_s')
    need(J.get('kind') == 'pilot' and J.get('version') == VERSION and isinstance(pr, (int, float)) and np.isfinite(pr),
         f'{path.name}: not a {VERSION} pilot receipt with a finite projection.full_run_cpu_s (refused before launch)')
    need(float(pr) <= PROJ_MAX_S, f'the recorded pilot projects {float(pr):.1f} CPU s for the full run > {PROJ_MAX_S:.0f} s (refused before launch)')
    hashes[_rel(path)] = _sha(path)
    return dict(pilot_receipt=_rel(path), full_run_cpu_s=float(pr), limit_s=PROJ_MAX_S)


def _load_default(day, mid, root, hc):
    return CX.load_market(day, int(mid), TOP3, root=root, hash_cache=hc)


def execute(kind, days, out, inp=None, pilot_receipt=None, require_freeze=True, log=print, loader=None, book=None, dirs=None, fv_sha=FV_SHA256):
    """kind 'run' (every measurement day -> v2/chains/touch_transition; freeze + recorded pilot projection required) or 'pilot' (its day, out
    given). loader / book / dirs / fv_sha are injected only by the selftest."""
    c_imp = time.process_time()
    t0, asof0 = time.time(), _now()
    need(mutant() == '', f'{MUT_ENV}={os.environ.get(MUT_ENV)!r} is set: production never runs a planted error (refused before any input is read)')
    for ev_ in FOREIGN_MUT:
        need(os.environ.get(ev_, '') == '', f"{ev_}={os.environ.get(ev_)!r} is set: a reused tool's planted error would enter this run (refused before "
                                            'any input is read)')
    need(C.RUN == 'v2', f'MT_RUN must be v2 (is {C.RUN!r}; refused)')
    need(kind in ('run', 'pilot'), f'kind {kind!r} (refused)')
    days = tuple(str(d) for d in days)
    need(len(days) >= 1 and len(set(days)) == len(days) and all(d in C.MEASURE_DAYS for d in days),
         'days must be distinct measurement days (refused before any path is built)')
    out = Path(out)
    need(not out.exists(), f'write-once: {out} exists (refused before any input is read)')
    need(out.parent.is_dir(), f'the output parent directory is missing: {out.parent} (refused)')
    PT = dict(paths())
    PT.update(dirs or {})
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
                 'freeze: touch_transition.py, its selftest, the contract and the reused tools must be tracked and unmodified (refused)')
        pre = preflight_projection(Path(pilot_receipt) if pilot_receipt else PT['pilot_receipt'], hashes)
    inp = Path(inp) if inp else C.OUT
    t3 = CX.top3(inp)
    need(tuple(a.lower() for a in t3) == TOP3_L, 'chain_context.top3 differs from the module top 3 (refused)')
    hashes[_rel(inp / 'econ' / 'identities.json')] = _sha(inp / 'econ' / 'identities.json')
    bm = CX.book_markets() if book is None else book
    need(all(d in bm for d in days), f'days {days} are not all BOOK manifest days (refused)')
    markets = [(d, int(m)) for d in days for m in sorted(int(x) for x in bm[d])]
    tick('preflight')
    EV, RW, poinfo = read_po(PT['po'], markets, hashes, fv_sha)
    Q2, q2info = read_qf2(PT['qf2'], markets, hashes, fv_sha, poinfo['levels_sha256'])
    RG, rginfo = PO.read_regrade(PT['regrade'], markets, hashes)
    PF, pfinfo = read_post_fill(PT['post_fill'], markets, hashes, fv_sha)
    TK, tkinfo = read_takes(PT['tl2'], markets, hashes, fv_sha)
    dependency = dict(posting_opportunity=poinfo, queue_front_v2=q2info, regrade=rginfo, post_fill=pfinfo, take_logic_v2=tkinfo,
                      owner_grading='posting_opportunity.identity (frozen section-3 code) on each response cell, queue_front_v2 candidates.csv as source')
    tick('inputs')
    loader = _load_default if loader is None else loader
    ctx = CT.make_ctx(CX)
    hc, parts = {}, []
    for d, m in markets:
        M = loader(d, m, inp, hc)
        need(str(M.day) == d and int(M.mid) == m and bool(M.book), f'loaded market {M.day} {M.mid} is not the BOOK market {d} {m} (refused)')
        for k_, v_ in (M.inputs or {}).items():
            if isinstance(v_, dict) and v_.get('sha256'):
                hashes.setdefault(k_, v_['sha256'])
        tick('context_load')
        parts.append(market_pass(d, m, M, ctx, EV.get((d, m)), RW.get((d, m)), Q2.get((d, m)), RG.get((d, m), {})))
        del M
        cnt['markets'] += 1
        tick('market')
        guard(f'market {d} {m}')
    fvp = inp / 'econ' / 'fills_valued.parquet'
    fsha = hc.get(str(fvp.resolve())) or _sha(fvp)
    hashes['econ/fills_valued.parquet'] = fsha
    need(fsha == fv_sha, f'fills_valued sha256 {fsha[:12]} differs from the accepted binding {str(fv_sha)[:12]} (refused)')
    tick('bindings')
    LG = pd.concat([p['legs'] for p in parts], ignore_index=True)
    FV = read_fv(fvp, LG, cnt)
    tick('fills_valued')
    RW_all = pd.concat([RW[k] for k in markets], ignore_index=True) if RW else None
    n_mk = len(markets)
    if kind == 'run':
        bad = sum(int(p['counts'].get('creation_rows_time_mismatch', 0)) + int(p['counts'].get('creation_rows_frame_out_of_range', 0)) for p in parts)
        need(bad == 0, f'{bad} saved creation rows do not sit on the loaded frames (refused: mismatched artifacts)')
        par = sum(int((p['OW']['po_parity'] == 'differs').sum()) for p in parts)
        need(par == 0, f'{par} response cells graded unlike the saved posting_opportunity identity (refused: mismatched artifacts)')

    def extra_fn(stg):
        ex = dict(preflight=pre)
        if kind == 'pilot':
            r_mk = N_BOOK / max(1, n_mk)
            fixed = sum(stg.get(k, 0.) for k in ('imports', 'preflight', 'inputs', 'bindings'))
            scal = sum(stg.get(k, 0.) for k in ('context_load', 'market', 'fills_valued', 'tables', 'write'))
            proj = fixed + scal * r_mk
            ex['projection'] = dict(full_run_cpu_s=round(proj, 2), pilot_cpu_s=round(time.process_time(), 3), limit_s=PROJ_MAX_S,
                                    launch_allowed=bool(proj <= PROJ_MAX_S),
                                    rule=f'fixed (imports + preflight + inputs + bindings) {fixed:.2f} + (context_load + market + fills_valued + tables + '
                                         f'write) {scal:.2f} x {N_BOOK}/{n_mk}; the first load carries the one-off input sha256s (scaled: conservative)')
        return ex
    rec, claims, dg = run_core(kind, days, parts, out, hashes, cnt, stages, tick, git, asof0, t0, dict(PF=PF, TK=TK, FV=FV), RW_all, dependency, extra_fn)
    log(json.dumps(_clean(dict(out=str(out), digest=dg, projection=rec.get('projection'), cpu_s=rec['cpu_s'], vmhwm_mib=rec['vmhwm_mib'])),
                   indent=1, sort_keys=True, default=str))
    return rec


def main(argv):
    a = list(argv[1:])
    usage = 'usage: MT_RUN=v2 touch_transition.py run | MT_RUN=v2 touch_transition.py run --pilot <day> [--out <dir>]'
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
