"""MAKER-TRACE-RECONSTRUCTION-v1 economics (P-2026-005; contract.md section 6; session aea5bfb9 development, 2026-11-22). USER: "we figure
out how top makers make money first". Every fill of every address - its maker legs AND its taker legs - valued in YES space (Up = indexSet
1 = YES): s = +1 buys YES / -1 sells YES, q shares, p the leg's YES price (pf_mt_common.yes_space), V = 1 if Up won, 0 if Down.
  P&L to resolution gross_res = s*q*(V - p), net_res = gross_res - fee_res;  spread at the fill v_spr = s*q*(m_minus - p);
  markouts v_<h> = s*q*(m(t + h) - p), h = 1 s / 10 s / 60 s;  v_end = s*q*(m_END - p).
RULES (declared before any outcome is read; synthetic proof in pf_mt_econ_selftest.py)
 TIME   A record is ONE settlement: its maker legs and its taker share one fill time t. BOOK markets: t = the latest lifecycle lower end lo_ms
        over the record's legs, taken only from fills rows with status ok (t_status lo; the frame before the fill evidence, so m_minus is
        pre-trade; a flagged fills row gives hi_fill_status). Without a bounded lo t = settle_ms
        (the UUIDv7 creation time, an upper bound; first_recv_ns is the collector's clock and is never mixed into a venue-clock lookup -
        records received before their settle_ms are counted as clock-skew evidence), t_status hi_lo_unbounded / hi_window_inverted /
        hi_missing_life_fill / hi_no_lifecycle (every non-BOOK FILL market: the frame at settle_ms may already show the trade; segment dim
        'tclock' splits lo / hi). t (t_ms) anchors ONLY the spread m_minus; markouts, beyond_end and after_end always start from settle_ms
        (the upper bound) on every market; win_ms = settle_ms - lo. v_spr_hi = the spread at settle_ms for the same fills (on BOOK the
        receipt reports how often that frame already shows the trade: at / after the lifecycle evidence frame f_after).
 MID    m(x) = (best_bid + best_ask) / 2 of the last tob frame (ordered upd_ms, recv_ns) with upd_ms <= x in that market. Status, first that
        applies: no_frame, frame_status (a bad_frame / mid_mismatch / out_of_order / off_grid flag; gap / dup / new_conn / same_ms_changed
        frames are full snapshots and usable), one_sided, crossed (bid >= ask), stale (x - upd_ms > 1,000 ms), else ok. END = the last frame
        <= we - 2 s; a markout with settle + h > we - 2 s is beyond_end; the END markout of a fill with settle > we - 2 s is after_end.
        A value is NaN unless its status is ok; sums use ok rows only and every status is counted. ID day (REST legs only): id_day_no_book.
 TAKER  valued leg by leg at each maker's YES price, side opposite to the maker, at the record's t: every leg nets to zero across its two
        addresses (checked per market). The record's own average price (taker.price = priceExecuted) is reconciled against the legs as a
        diagnostic only: px_gap per row (record's taker average YES price - share-weighted leg YES price), counted with its USD at stake.
 FEES   recorded amounts on every leg (maker fees too). COLLATERAL = USD cash. SHARES = shares of the payer's own outcome token LOST: at
        resolution valued exactly (V for a YES token, 1 - V for a NO token: fee_res, so P&L to resolution is the cash identity), and at the
        fill price p_own (fee_fill = the conversion at the fill price, reported beside the mark-based horizons). Stored as the fee's YES-space
        position fee_dy and cash fee_dc: the fee at a YES value x is -(fee_dc + fee_dy * x). A record's taker fee is allocated to its legs pro
        rata to shares (the record sum is exact).
 FIFO   lots per address x market on gross fills, in (settle_ms, sid, px_rank, mk_idx, maker before taker) order; px_rank = price priority
        inside the record (makers[] is not the fill order: a taker buying YES takes the lowest ask first, a seller the highest bid; mk_idx
        only breaks equal prices, where the pairing cannot change a value). FIFO across sign flips pairs unit j
        bought with unit j sold, so mq / rq (matched / residual shares per fill) come from cumulative sums: round trip rt_pnl = -sum(s*p*mq)
        (V-free); residual resid_res = sum(s*rq*(V - p)) = resid_drift (sum(s*rq*(m_END - p))) + resid_resol ((V - m_END) * net_pos).
        CHECKED (refuses on failure): gross = rt_pnl + resid_res, resid_res = drift + resol, net = gross - fee_res, net_pos = buys - sells,
        and per market the sum over addresses of gross / net_pos / every mark value = 0 (tolerance 1e-9 + 1e-12 x notional). Fee shares
        are not lots (fees are their own line). A self-trade (maker == taker) is kept out of the lots: its two rows cancel within the
        address (mq = rq = 0; its gross, fees and shares are still counted).
 STATUS A leg pair (its maker and taker rows) is valued or excluded together, and an excluded leg excludes its whole record
        (record_leg_excluded). Source statuses are '+'-joined flags ('ok' only when clean); any flag excludes (src_status, raw string kept).
        A market with the boundary_censored flag on any of its legs, records or its markets_<day> row is excluded whole (boundary_censored,
        st_res censored): its registry outcome is blanked on load and never used (it can come from a sealed hour). Excluded rows stay in
        fills_valued with status and the source status (src_status), and are counted.
 SEGMENTS market time by legs.t_rel_s (pre / w000_060 .. w240_300 / post); price |p - 0.5| in 0.1 buckets with extremes p <= 0.10 / p >= 0.90
        (micro-unit integers); kind regular / mint (both Bid, other outcome) / merge (both Ask); role maker / taker; tclock lo / hi; on BOOK
        markets the classes of the maker's hash from its S_FIFO lifecycle row: repeated (n_fills >= 2) / single; exhausted (removal set
        exactly {exhausted}) / exhausted_or_other / not_exhausted; front (ahead_hi == 0) / behind (> 0) / unknown (NaN).
 IDENTITIES ex ante from the ID day only: top 10 by maker shares (each member's maker share of volume recorded) and top 10 by MAKER-leg
        net P&L to resolution among maker-majority addresses (sh_mk > sh_tk) with resolved maker shares (an address cannot rank on its taker
        legs); ties by address ascending; same-day top 10s per measurement day (same rules) are labelled descriptive. Comparison groups per set and day: in / field (the day's other makers) / nonmaker (closes the zero sum). Days are
        point estimates; intervals need >= 5 complete UTC days (rule 8), computed from the measured day count.
 REWARDS per-maker rewards are unobservable. Inventory of the reward fields per market (markets_<day>); per maker on BOOK markets only a
        LOWER BOUND of demonstrated eligible presence: the S_NOPRIO proven presence [persist_lo_ms, persist_hi_ms] (class observed /
        constrained) of orders with size_lb >= shareThreshold, cut where the order's proven remaining (its fills whose window starts after
        t) falls below shareThreshold, clipped to the reward schedule window, counted only while a fresh (<= 1 s) ok two-sided frame has
        |p - mid| < spreadThreshold (strict: the boundary is not demonstrated). Union per side; both = the intersection. Per market
        (rewards_market): the pool from the distinct rewards_variants schedule entries (units unverified), and on BOOK markets the depth
        UPPER bound per side (shown shares at levels strictly within spreadThreshold of the frame mid x frame duration <= 1 s, usable
        two-sided frames in the window; also only levels >= shareThreshold) with each identity set's proven lower bound.
 REBATE scenario reb25 (advertised, unverified): a maker leg earns 0.25 x its counterparty taker fee, valued like fees; never in net_res.
 TIES   V = 0.5 when start_price == end_price (st_res tie, used everywhere V is); a market with registry up_index != 1 is excluded (up_index).
OUTPUTS OUT/econ/ (write-once; built in a partial directory renamed at the end): fills_valued.parquet, maker_market.parquet, maker_day.parquet,
segments.parquet, rewards_market.parquet, identities.json, econ.json (receipt: input sha256s, rules, status counts, check results,
diagnostics, memory).
Not a strategy: no imitation, fit, tradeable claim or profitability verdict. Usage (run_light): pf_mt_econ.py run [--chunk N]"""
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_mt_common as C  # noqa: E402

need, OK = C.need, C.STATUS_OK
HORIZONS = (('1s', 1_000), ('10s', 10_000), ('60s', 60_000))
VALS = ('spr',) + tuple(h for h, _ in HORIZONS) + ('end',)
STALE_MS, END_GUARD_MS, U = 1_000, 2_000, 1_000_000
TOL_ABS, TOL_REL, TOL_AMT, TOL_PX = 1e-9, 1e-12, 1e-9, 1.5e-6
CHUNK, TOP_K, G_MIN = 16, 10, 5            # 16 markets per chunk: measured peak 932 vs 1044 MiB (32) at equal CPU
BOOK_MANIFEST = C.REPO / 'predictfun/reviews/ladder_async_fast_20261122/manifest.json'
FEE_TYPES = ('SHARES', 'COLLATERAL')

ROW_ST = ('ok', 'no_record', 'no_market', 'boundary_censored', 'up_index', 'src_status', 'bad_leg', 'no_taker_signer', 'record_mismatch',
          'amount_mismatch', 'leg_count_mismatch', 'yes_space_mismatch', 'side_mismatch', 'bad_mix', 'fee_type_unknown', 'record_leg_excluded')
MARK_ST = ('ok', 'no_frame', 'frame_status', 'one_sided', 'crossed', 'stale', 'beyond_end', 'after_end', 'id_day_no_book')
T_ST = ('lo', 'hi_lo_unbounded', 'hi_fill_status', 'hi_window_inverted', 'hi_missing_life_fill', 'hi_no_lifecycle')
RES_ST = ('ok', 'no_resolution', 'censored', 'tie', 'up_index')
RES_VALID = ('ok', 'tie')                                   # resolved: V = 1 / 0, or 0.5 when the start and end prices are equal
CENSOR = 'boundary_censored'
BAD_FRAME = frozenset(('bad_frame', 'mid_mismatch', 'out_of_order', 'off_grid'))   # the only frame flags that make a snapshot unusable
#   (gap / dup / new_conn / same_ms_changed frames are full snapshots and usable)
REB = 0.25                                                  # rebate scenario: share of the counterparty taker fee (advertised, unverified)


def flags(s):
    """A status string as its set of '+'-joined flags (extractor convention: 'ok' only when clean)."""
    return set(str(s).split('+'))


def flagged(col, flag):
    """Per row: does the status column carry `flag`? (evaluated once per distinct status value; missing = no)."""
    s = pd.Series(np.asarray(col, dtype=object), dtype=object).fillna('')
    return s.map({x: flag in flags(x) for x in s.unique()}).to_numpy(bool)


def flagged_any(col, flagset):
    """Per row: does the status column carry any flag of `flagset`?"""
    s = pd.Series(np.asarray(col, dtype=object), dtype=object).fillna('')
    return s.map({x: bool(flags(x) & flagset) for x in s.unique()}).to_numpy(bool)


KIND = ('regular', 'mint', 'merge', 'bad_mix')
TSEG = ('pre', 'w000_060', 'w060_120', 'w120_180', 'w180_240', 'w240_300', 'post', 'unknown')
PSEG = ('d_0.0_0.1', 'd_0.1_0.2', 'd_0.2_0.3', 'd_0.3_0.4', 'p_le_0.10', 'p_ge_0.90')
NA_LIFE = ('na_taker', 'na_not_book', 'life_not_ok')
LIFE_REP = ('repeated', 'single') + NA_LIFE
LIFE_EXH = ('exhausted', 'exhausted_or_other', 'not_exhausted') + NA_LIFE
LIFE_FRONT = ('front', 'behind', 'unknown') + NA_LIFE
LIFE_ST = ('ok', 'na_taker', 'na_not_book', 'missing_life', 'life_mismatch', 'nfills_mismatch', 'bad_removal_set', 'orders_status')
DEPTH = ('d0', 'd1', 'd2', 'd3p', 'inside', 'unknown')    # ticks from the pre-trade best on the maker's side (inside: better than shown)
SWEEP = ('sweep', 'single_price')
CPB = ('cp_top', 'cp_other', 'na_id_day')
BOOST = ('boosted', 'not_boosted', 'unknown')
SELFB = ('self', 'not_self')
SEG_DIMS = ('all', 'tseg', 'pseg', 'kind', 'tclock', 'life_rep', 'life_exh', 'life_front', 'depth', 'sweep', 'cp', 'boost', 'self')
BUCKETS = tuple(dict.fromkeys(('all',) + TSEG + PSEG + KIND + ('lo', 'hi') + LIFE_REP + LIFE_EXH + LIFE_FRONT + DEPTH + SWEEP + CPB + BOOST + SELFB))
DIM_COL = dict(depth='dpt', sweep='sweep', cp='cp', boost='boost')
RM_SETS = ('exante_shares', 'exante_pnl', 'sameday_shares', 'sameday_pnl')
B_IX = {b: i for i, b in enumerate(BUCKETS)}

# ---- inputs: the columns read (the full pf_mt_common schema is checked in every file footer) ----
LEG_COLS = ['day', 'mid', 'sid', 'settle_ms', 'first_recv_ns', 'mk_idx', 'mk_hash', 'mk_signer', 'mk_quote', 'mk_index_set', 'mk_price', 'shares',
            'mk_fee', 'mk_fee_type', 'y_side', 'y_price', 'y_ptick', 'tk_hash', 'tk_signer', 'tk_y_side', 't_rel_s', 'status']
LEG_CATS = ('day', 'mk_signer', 'mk_quote', 'mk_fee_type', 'y_side', 'tk_signer', 'tk_y_side', 'status')   # low cardinality; sid / hashes stay
#   arrow-backed strings for the day (a day-wide dictionary of ~450k values costs ~175 MiB of load peak, measured) and are categorised per chunk
REC_COLS = ['day', 'mid', 'sid', 'settle_ms', 'first_recv_ns', 'n_makers', 'amount_filled', 'tk_hash', 'tk_signer', 'tk_quote', 'tk_index_set',
            'tk_price', 'tk_amount', 'tk_fee', 'tk_fee_type', 'tk_y_side', 'status']
REC_CATS = ('day', 'tk_quote', 'tk_fee_type', 'tk_y_side', 'status')
TOB_COLS = ['mid', 'upd_ms', 'recv_ns', 'n_bid', 'n_ask', 'best_bid', 'best_ask', 'status']
ORD_COLS = ['scenario', 'mid', 'mk_hash', 'mk_signer', 'y_side', 'y_ptick', 'n_fills', 'size_lb', 'persist_lo_ms', 'persist_hi_ms', 'persist_class',
            'ahead_hi', 'removal_set', 'status']
LF_COLS = ['mid', 'sid', 'mk_hash', 'mk_idx', 'shares', 'hi_ms', 'lo_ms', 'f_after', 'status']
DFR_COLS = ['fid', 'upd_ms', 'recv_ns', 'status']
DLV_COLS = ['fid', 'side', 'ptick', 'shares', 'status']
REG_SCHEMA = dict(mid='int64', ws='num', we='num', outcome='str', up_index='int64', start_price='num', end_price='num')
REG_OPT = ('fee_bps', 'spread_thr', 'share_thr', 'decimal_precision')
MKT_SCHEMA = dict(mid='int64')
MKT_FIELDS = dict(fee_bps=('feeRateBps', 'fee_rate_bps', 'fee_bps'), maker_fee_bps=('makerFeeBps', 'maker_fee_bps'),
                  spread_thr=('spreadThreshold', 'spread_threshold', 'spread_thr'), share_thr=('shareThreshold', 'share_threshold', 'share_thr'),
                  boosted=('isBoosted', 'is_boosted'), boost_lo=('boostStartsAt', 'boost_starts_at'), boost_hi=('boostEndsAt', 'boost_ends_at'),
                  rewards=('rewards', 'rewards_json'), rewards_variants=('rewards_variants',))

# ---- outputs: exact columns and dtypes (written by put(); a mismatch refuses) ----
FV = dict(day='str', mid='int64', sid='str', mk_idx='int64', px_rank='int64', role='str', addr='str', hash='str', cp_addr='str', kind='str',
          own_index_set='int64', y_side='str', s='int64', q='float64', p='float64', px_gap='float64', self_trade='bool', settle_ms='int64', t_ms='int64',
          win_ms='int64', t_status='str', t_rel_s='float64',
          tseg='str', pseg='str', V='float64', st_res='str', gross_res='float64', m_minus='float64', age_ms='int64', spr_w='float64', st_spr='str',
          v_spr='float64', st_spr_hi='str', v_spr_hi='float64', **{k: t for h, _ in HORIZONS for k, t in ((f'st_{h}', 'str'), (f'v_{h}', 'float64'))},
          m_end='float64', st_end='str', v_end='float64', fee='float64', fee_type='str', fee_dy='float64', fee_dc='float64', fee_fill='float64',
          fee_res='float64', net_res='float64', reb25_fill='float64', reb25_res='float64', mq='float64', rq='float64', life_rep='str', life_exh='str',
          life_front='str', life_status='str', dpt='str', sweep='str', cp='str', boost='str', src_status='str', status='str')
MET = ['n', 'sh', 'notional', 'gross_res', 'sh_res', 'fee_res', 'fee_fill', 'net_res', 'n_res_na', 'reb25_res', 'reb25_fill'] + \
    [f'{k}_{h}' for h in VALS for k in ('v', 'sh')]
MM_SUM = ['n_mk', 'sh_mk', 'n_tk', 'sh_tk', 'notional', 'sh_buy', 'sh_sell', 'gross_res_mk', 'gross_res_tk', 'sh_res', 'sh_res_mk', 'fee_res', 'fee_fill',
          'net_res', 'net_res_mk', 'reb25_res', 'reb25_fill', 'n_res_na'] + [f'{k}_{h}_{r}' for h in VALS for r in ('mk', 'tk') for k in ('v', 'sh')]
FIFO = ['rt_sh', 'rt_pnl', 'net_pos', 'resid_res', 'resid_drift', 'resid_resol']
REW = ['rew_bid_ms', 'rew_ask_ms', 'rew_both_ms', 'rew_n_orders']
INTS = {'n', 'n_mk', 'n_tk', 'n_res_na', 'n_excl', 'n_addr', 'n_markets', 'n_markets_mk', 'n_mkt_res_na', 'n_mkt_end_na', 'rew_n_orders',
        'rank_sh', 'rank_pnl'}
_ty = lambda cols: {c: ('int64' if c in INTS else 'float64') for c in cols}
MM = dict(day='str', mid='int64', addr='str', population='str', book='bool', res_status='str', end_status='str', is_maker='bool', **_ty(MM_SUM),
          **_ty(FIFO), fifo_status='str', **_ty(REW), rew_status='str', n_excl='int64', status='str')
MD = dict(day='str', addr='str', population='str', n_markets='int64', n_markets_mk='int64', **_ty(MM_SUM), **_ty(FIFO), rt_pnl_res='float64',
          n_mkt_res_na='int64', n_mkt_end_na='int64', **_ty(REW), n_excl='int64', rank_sh='int64', rank_pnl='int64', in_exante_sh='bool',
          in_exante_pnl='bool', in_sameday_sh='bool', in_sameday_pnl='bool', status='str')
SEG = dict(day='str', id_set='str', label='str', grp='str', role='str', dim='str', bucket='str', n_addr='int64', **_ty(MET), ci='str', status='str')
RM = dict(day='str', mid='int64', book='bool', table_status='str', inv_status='str', n_variants='int64', pool_units='float64', spread_thr='float64',
          share_thr='float64', boosted='str', depth_status='str', n_frames_used='int64', ub_bid_shs='float64', ub_ask_shs='float64', ub_bid_shs_thr='float64',
          ub_ask_shs_thr='float64', **{f'lb_{k}_ms_{s_}': 'float64' for s_ in RM_SETS for k in ('bid', 'ask', 'both')}, status='str')
ARROW = {'str': pa.string(), 'int64': pa.int64(), 'float64': pa.float64(), 'bool': pa.bool_()}
RULES = dict(  # the docstring's declared rules as receipt fields
    time='spread anchor t_ms: BOOK = max lifecycle lo_ms over the record legs from status-ok fills rows (lo); else settle_ms (UUIDv7 creation, '
         'upper bound); markouts, beyond_end and after_end always from settle_ms; win_ms = settle_ms - lo; collector receipt never mixed',
    spread_hi='v_spr_hi = the spread at settle_ms for the same fills (on BOOK the hi-clock frame may already show the trade: share reported)',
    resolution='V = 1 Up / 0 Down; 0.5 when start_price == end_price (tie, applied before any use); a market whose registry up_index != 1 is '
               'excluded (up_index)',
    frames='a frame is unusable only with a bad_frame / mid_mismatch / out_of_order / off_grid flag (gap / dup / new_conn / same_ms_changed usable)',
    rebate='scenario reb25 (advertised, unverified): a maker leg earns 0.25 x the counterparty taker fee allocated to it, valued like fees '
           '(reb25_res at resolution, reb25_fill at the fill price); never in net_res (0 % = current)',
    censor="a market flagged boundary_censored in legs / records / markets status is excluded whole; its registry outcome is blanked on load",
    status="source statuses are '+'-joined flags; any flag other than ok excludes the leg pair and its record (counted, raw string kept)",
    mid='(best_bid + best_ask) / 2 of the last tob frame (upd_ms, recv_ns order) with upd_ms <= x; no_frame / frame_status / one_sided / crossed / stale > 1000 ms',
    end='END = last frame <= we - 2000 ms; t + h > END -> beyond_end; t > END -> after_end for the END markout',
    taker='valued per maker leg at that leg YES price, opposite side; record average price (priceExecuted) reconciled as a diagnostic only (px_gap)',
    fee='COLLATERAL = USD; SHARES = own-token shares lost: fee_res = shares x token value at resolution (V or 1 - V), fee_fill = shares x fill price; '
        'taker fee allocated pro rata to leg shares',
    fifo='per address x market on gross fills in (settle_ms, sid, price priority px_rank, mk_idx, maker first) order; unit j bought pairs unit j '
         'sold; self-trades not lots',
    rewards='per-maker rewards are unidentifiable. Lower bound per address: S_NOPRIO proven presence (observed / constrained) of size_lb >= '
            'shareThreshold orders, cut at the proven-remaining threshold, schedule window, fresh ok two-sided frames with |p - mid| < '
            'spreadThreshold (strict). Upper bound per BOOK market side: shown shares at depth levels strictly within spreadThreshold of the '
            'frame mid x frame duration (to the next frame, <= 1 s) over usable two-sided frames in the window (and with >= shareThreshold '
            'shares per level). Pool = sum hourlyRate x window overlap over the distinct rewards_variants schedule entries (units unverified)',
    identities='ex ante from the ID day: top 10 by maker shares (maker share of volume recorded); top 10 by MAKER-leg net P&L to resolution '
               'among maker-majority addresses (sh_mk > sh_tk) with resolved maker shares; ties by address ascending; same rules same-day',
    segments='dims tseg / pseg / kind / tclock / life_* / depth (ticks from the pre-trade best on the maker side at the m_minus frame) / sweep '
             '(record spans >= 2 YES prices) / cp (counterparty in an ex-ante set) / boost (isBoosted) / self (self-trade)')


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def hwm():
    """This process's never-reset high-water RSS (VmHWM, MiB)."""
    return int(re.search(r'VmHWM:\s+(\d+)', Path('/proc/self/status').read_text()).group(1)) / 1024.


def trim():
    """Return freed heap pages to the OS (arrow's pool and glibc), so later peaks do not stack on freed-but-held memory."""
    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except AttributeError:
        pass
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


def guard(lim, where):
    h = hwm()
    need(h < lim, f'memory self-guard: VmHWM {h:.0f} MiB >= {lim:.0f} MiB at {where}')
    return h


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _obj(x):
    return np.asarray(x, dtype=object)


def _isin(a, vals):
    return pd.Series(a.to_numpy() if isinstance(a, pd.Series) else np.asarray(a)).isin(list(vals)).to_numpy()


def _scalar(x):
    """A table cell as a JSON-able scalar (None for missing)."""
    if isinstance(x, (dict, list, str, bool)) or x is None:
        return x
    if hasattr(x, 'item'):
        x = x.item()
    return None if (isinstance(x, float) and np.isnan(x)) or x is pd.NA else x


def cat(codes, vocab):
    return pd.Categorical.from_codes(np.asarray(codes, dtype=np.int16), categories=list(vocab))


def _type_ok(t, want):
    if pa.types.is_dictionary(t):
        t = t.value_type
    return dict(str=pa.types.is_string(t) or pa.types.is_large_string(t), int64=pa.types.is_int64(t), float64=pa.types.is_float64(t),
                num=pa.types.is_int64(t) or pa.types.is_float64(t), bool=pa.types.is_boolean(t)).get(want, False)


def load(path, kind, cols, cats=(), schema=None, optional=(), hashes=None):
    """Read `cols` (+ the `optional` columns present) of a parquet input after checking every required column / dtype of its schema
    (pf_mt_common.SCHEMAS[kind] or `schema`) in the footer. Records the file's sha256 in `hashes`."""
    path = Path(path)
    need(path.is_file(), f'{kind}: input missing: {path.parent.name}/{path.name}')
    want, sch = (schema if schema is not None else C.SCHEMAS[kind]), pq.read_schema(path)
    for c, d in want.items():
        need(c in sch.names, f'{kind}: required column {c} missing ({path.name})')
        need(_type_ok(sch.field(c).type, d), f'{kind}: column {c} is {sch.field(c).type}, the schema says {d} ({path.name})')
    cols = list(cols) + [c for c in optional if c in sch.names and c not in cols]
    tb = pq.read_table(path, columns=cols, read_dictionary=[c for c in cats if c in cols])
    df = tb.to_pandas(split_blocks=True, self_destruct=True)       # frees each arrow column as it is converted (the load peak)
    del tb
    trim()
    for c in cats:
        if c in df and not isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype('category')
    if hashes is not None:
        hashes[f'{path.parent.name}/{path.name}'] = dict(sha256=sha256(path), bytes=path.stat().st_size, rows=len(df))
    return df


def load_book(path=BOOK_MANIFEST):
    """The contract's BOOK population: manifest.json 'complete' (day -> market ids)."""
    return {d: sorted(int(m) for m in v) for d, v in json.loads(Path(path).read_text())['complete'].items()}


def registry(path, hashes, censored=()):
    """step1 registry -> mid, ws_ms, we_ms, end_ms (we - 2 s), V, censored, upbad, tie, optional thresholds. V = 1 Up / 0 Down, 0.5 when
    start_price == end_price (the venue resolves 50-50; applied here, before anything uses V), NaN otherwise. A boundary-censored market's
    outcome is blanked on load (it can come from a sealed hour); a market whose up_index != 1 is flagged upbad and its outcome blanked
    (Up is not indexSet 1 there); the raw outcome column is not kept."""
    R = load(path, 'registry', list(REG_SCHEMA), schema=REG_SCHEMA, optional=REG_OPT, hashes=hashes)
    need(R.mid.is_unique, f'registry: market id not unique ({Path(path).parent.name})')
    R['censored'] = R.mid.isin(set(int(m) for m in censored)).to_numpy()
    R['upbad'] = R.up_index.to_numpy() != 1
    blank = R.censored.to_numpy() | R.upbad.to_numpy()
    o = np.where(blank, None, _obj(R.outcome))
    sp, ep = R.start_price.to_numpy(float), R.end_price.to_numpy(float)
    R['tie'] = ~blank & np.isfinite(sp) & np.isfinite(ep) & (sp == ep)
    R = R.drop(columns=['outcome'])
    R['V'] = np.where(R.tie.to_numpy(), 0.5, np.where(o == 'Up', 1., np.where(o == 'Down', 0., np.nan)))
    R['ws_ms'] = np.rint(R.ws.to_numpy(float) * 1000).astype(np.int64)
    R['we_ms'] = np.rint(R.we.to_numpy(float) * 1000).astype(np.int64)
    R['end_ms'] = R.we_ms - END_GUARD_MS
    return R


def frames(T):
    """tob (one day) -> frame arrays sorted by (mid, upd_ms, recv_ns) with the composite search key and each frame's own MARK_ST code
    (sorted array by array, never a full-table copy)."""
    o = np.lexsort((T.recv_ns.to_numpy(np.int64), T.upd_ms.to_numpy(np.int64), T.mid.to_numpy(np.int64)))
    mid, t = T.mid.to_numpy(np.int64)[o], T.upd_ms.to_numpy(np.int64)[o]
    need(len(t) == 0 or (t.min() >= 0 and t.max() < 1 << 42), 'tob: upd_ms outside the epoch-ms range')
    bb, ba = T.best_bid.to_numpy(float)[o], T.best_ask.to_numpy(float)[o]
    two = (T.n_bid.to_numpy()[o] > 0) & (T.n_ask.to_numpy()[o] > 0) & np.isfinite(bb) & np.isfinite(ba)
    st = np.zeros(len(T), np.int8)
    st[flagged_any(T.status, BAD_FRAME)[o]] = MARK_ST.index('frame_status')    # gap / dup / new_conn / same_ms_changed stay usable
    st[(st == 0) & ~two] = MARK_ST.index('one_sided')
    st[(st == 0) & (bb >= ba)] = MARK_ST.index('crossed')
    um = np.unique(mid)
    code = np.searchsorted(um, mid)
    return dict(um=um, code=code, t=t, m=np.where(st == 0, (bb + ba) / 2, np.nan), st=st, key=(code.astype(np.int64) << 42) | t, bb=bb, ba=ba,
                lo=np.searchsorted(code, np.arange(len(um))), hi=np.searchsorted(code, np.arange(len(um)), 'right'),
                counts={MARK_ST[k]: int(v) for k, v in zip(*np.unique(st, return_counts=True))},
                tob_status={str(k): int(v) for k, v in T.status.value_counts().items() if v})


def mid_at(Fr, mid, x, ret_j=False):
    """Each query (market, venue ms x) -> the last frame with upd_ms <= x in that market: (mid price or NaN, age ms or -1, MARK_ST code[,
    frame index or -1])."""
    n = len(x)
    m, age, st = np.full(n, np.nan), np.full(n, -1, np.int64), np.full(n, MARK_ST.index('no_frame'), np.int8)
    if Fr is None or n == 0 or len(Fr['t']) == 0:
        return (m, age, st, np.full(n, -1, np.int64)) if ret_j else (m, age, st)
    um = Fr['um']
    c = np.searchsorted(um, mid)
    has = (c < len(um)) & (um[np.minimum(c, len(um) - 1)] == mid) & (x >= 0)
    need((x[has] < 1 << 42).all(), 'a query time outside the epoch-ms range')
    i = np.searchsorted(Fr['key'], (c.astype(np.int64) << 42) | np.clip(x, 0, (1 << 42) - 1), 'right') - 1
    j = np.maximum(i, 0)
    ok = has & (i >= 0) & (Fr['code'][j] == c)
    age[ok] = x[ok] - Fr['t'][j[ok]]
    st[ok] = Fr['st'][j[ok]]
    st[ok & (st == 0) & (age > STALE_MS)] = MARK_ST.index('stale')
    sel = ok & (st == 0)
    m[sel] = Fr['m'][j[sel]]
    return (m, age, st, np.where(ok, j, -1)) if ret_j else (m, age, st)


def _index_of(keys, queries):
    """Position of each query in the unique string `keys`, -1 if absent (categorical queries resolved through their categories)."""
    kx = pd.Index(_obj(keys))
    if isinstance(queries.dtype, pd.CategoricalDtype):
        pos = kx.get_indexer(_obj(queries.cat.categories))
        c = queries.cat.codes.to_numpy()
        return np.where(c >= 0, pos[np.maximum(c, 0)], -1)
    return kx.get_indexer(_obj(queries))


def yes_space_many(quote, iset, price):
    """pf_mt_common.yes_space on every unique (quote, indexSet, price) -> per-row expected (YES side, YES price); invalid -> (None, NaN)."""
    df = pd.DataFrame({'q': _obj(quote), 'i': np.asarray(iset, dtype=np.int64), 'p': np.asarray(price, dtype=float)})
    u = df.drop_duplicates(ignore_index=True)
    es, ep = [], []
    for q_, i_, p_ in u.itertuples(index=False, name=None):
        try:
            a, b = C.yes_space(q_, int(i_), float(p_))
        except (ValueError, TypeError):
            a, b = None, np.nan
        es.append(a)
        ep.append(b)
    u['es'], u['ep'] = pd.Series(es, dtype=object), np.asarray(ep, dtype=float)
    j = df.merge(u, on=['q', 'i', 'p'], how='left', validate='many_to_one')
    return _obj(j.es), j.ep.to_numpy(float)


def fee_pos(fee, ftype, own):
    """A fee as the payer's YES-space position change dy and cash change dc (FEES): COLLATERAL f -> dc = -f; SHARES f of the YES token ->
    dy = -f; SHARES f of the NO token (= 1 cash - 1 YES each) -> dy = +f, dc = -f. Fee at a YES value x = -(dc + dy * x)."""
    ft = _obj(ftype)
    sh, co = (ft == 'SHARES') & (fee != 0), (ft == 'COLLATERAL') & (fee != 0)
    dy = np.where(sh, np.where(own == 1, -fee, np.where(own == 2, fee, np.nan)), 0.)
    dc = np.where(co, -fee, np.where(sh, np.where(own == 2, -fee, np.where(own == 1, 0., np.nan)), 0.))
    return dy, dc


def build_rows(d, population, L, R, reg, Fr, lf, book_mids, depth=None, top_set=frozenset(), boost=None):
    """Legs of one chunk of markets -> fill rows: per leg one maker row and one taker row (same q, YES price and t, opposite sides), with
    statuses, time, fees, resolution value and marks (RULES). depth: {BOOK mid: (sorted depth-frame upd_ms, {fid: position})} for the
    spread-clock diagnostic; top_set: the ex-ante identity members (segment cp); boost: {mid: isBoosted}. Returns (F, per-market END
    {mid: (m_end, code)}, diagnostics)."""
    n = len(L)
    if len(R) == 0:
        R = pd.DataFrame({c: pd.Series([None], dtype=object) for c in REC_COLS}).assign(sid='\x00none', mid=-1, settle_ms=-1, first_recv_ns=-1,
                                                                                       n_makers=-1, amount_filled=np.nan, tk_amount=np.nan,
                                                                                       tk_price=np.nan, tk_fee=0., tk_index_set=-1)
    nR = len(R)
    mid = L.mid.to_numpy(np.int64)
    ri = _index_of(R.sid, L.sid)
    has = ri >= 0
    rj = np.maximum(ri, 0)
    Ro = {c: (R[c].to_numpy() if pd.api.types.is_numeric_dtype(R[c]) else _obj(R[c]))[rj] for c in REC_COLS if c != 'day'}
    st = np.zeros(n, np.int8)

    def flag(mask, name):
        st[(st == 0) & mask] = ROW_ST.index(name)

    ri_m = pd.Index(reg.mid.to_numpy(np.int64)).get_indexer(mid)
    inreg = ri_m >= 0
    rm = np.maximum(ri_m, 0)
    q, p = L.shares.to_numpy(float), L.y_price.to_numpy(float)
    ys, mkq, mki = _obj(L.y_side), _obj(L.mk_quote), L.mk_index_set.to_numpy(np.int64)
    tkq = np.where(has, Ro['tk_quote'], None)
    tki = np.where(has, Ro['tk_index_set'].astype(np.int64), -1)
    lst, rst = _obj(L.status), np.where(has, Ro['status'], OK)
    src = np.where(lst != OK, lst, np.where(rst != OK, rst, OK))
    cen = np.where(inreg, reg.censored.to_numpy(bool)[rm], False)
    upb = np.where(inreg, reg.upbad.to_numpy(bool)[rm], False)
    tie = np.where(inreg, reg.tie.to_numpy(bool)[rm], False)
    flag(~has, 'no_record')
    flag(~inreg, 'no_market')
    flag(cen, CENSOR)
    flag(upb, 'up_index')
    flag(src != OK, 'src_status')
    good = np.isfinite(q) & (q > 0) & np.isfinite(p) & (p > 0) & (p < 1) & _isin(ys, ('bid', 'ask')) & _isin(mkq, ('Bid', 'Ask')) & _isin(mki, (1, 2))
    flag(~good | (has & ~(_isin(tkq, ('Bid', 'Ask')) & _isin(tki, (1, 2)))), 'bad_leg')
    tks = _obj(L.tk_signer)
    flag(pd.isna(tks) | (tks == ''), 'no_taker_signer')
    flag(has & ((L.settle_ms.to_numpy() != Ro['settle_ms']) | (mid != Ro['mid']) | (tks != Ro['tk_signer']) | (_obj(L.tk_hash) != Ro['tk_hash'])
                | (_obj(L.tk_y_side) != Ro['tk_y_side'])), 'record_mismatch')
    hv = has & np.isfinite(q)
    Q = np.bincount(ri[hv], weights=q[hv], minlength=nR)
    cnt = np.bincount(ri[has], minlength=nR)
    af, ta = R.amount_filled.to_numpy(float), R.tk_amount.to_numpy(float)
    tol = TOL_AMT * np.maximum(1., np.abs(af))
    flag(has & ~((np.abs(Q - af) <= tol) & (np.abs(ta - af) <= tol))[rj], 'amount_mismatch')
    flag(has & (cnt != R.n_makers.to_numpy())[rj], 'leg_count_mismatch')
    es, ep = yes_space_many(mkq, mki, L.mk_price.to_numpy(float))
    flag((es != ys) | ~(np.abs(ep - np.round(p, 6)) <= 1e-9), 'yes_space_mismatch')
    ts_, tp_ = yes_space_many(tkq, tki, Ro['tk_price'].astype(float))
    flag(has & (ts_ != Ro['tk_y_side']), 'yes_space_mismatch')
    flag(has & (Ro['tk_y_side'] == ys), 'side_mismatch')
    kind = np.where(mki == tki, 0, np.where((mkq == 'Bid') & (tkq == 'Bid'), 1, np.where((mkq == 'Ask') & (tkq == 'Ask'), 2, 3)))
    kind[(mki == tki) & (mkq == tkq)] = 3
    flag(has & (kind == 3), 'bad_mix')
    mkf, mkft = L.mk_fee.to_numpy(float), _obj(L.mk_fee_type)
    tkf, tkft = np.where(has, Ro['tk_fee'].astype(float), 0.), np.where(has, Ro['tk_fee_type'], None)
    flag(~(np.isfinite(mkf) & np.isfinite(tkf)) | ~((mkf == 0) | _isin(mkft, FEE_TYPES)) | ~((tkf == 0) | _isin(tkft, FEE_TYPES)), 'fee_type_unknown')
    rk = np.where(has, ri, nR + np.arange(n))
    flag(pd.Series(st != 0).groupby(rk).transform('max').to_numpy(bool), 'record_leg_excluded')
    # price priority inside a record (makers[] is not the fill order): a taker buying YES takes the lowest ask first, a taker selling YES the
    # highest bid first; equal prices keep mk_idx order (no economic difference)
    s_mk = np.where(ys == 'bid', 1, -1).astype(np.int64)
    px_rank = pd.DataFrame({'rk': rk, 'pk': -s_mk * np.nan_to_num(p), 'k': L.mk_idx.to_numpy(np.int64)}).sort_values(
        ['rk', 'pk', 'k'], kind='mergesort').groupby('rk').cumcount().sort_index().to_numpy(np.int64)
    Qp = np.bincount(ri[hv], weights=(q * p)[hv], minlength=nR)
    px_gap = np.where(has & (Q[rj] > 0), tp_ - Qp[rj] / np.where(Q[rj] > 0, Q[rj], 1.), np.nan)   # record: taker avg YES price - legs' avg
    # ---- time: one t per record ----
    settle = np.where(has, Ro['settle_ms'], L.settle_ms.to_numpy()).astype(np.int64)
    frn = np.where(has, Ro['first_recv_ns'], L.first_recv_ns.to_numpy()).astype(np.int64)
    hi = settle
    in_book = _isin(mid, book_mids)
    lo_leg, hi_lf, has_lf, lf_ok = np.full(n, -1, np.int64), np.full(n, -1, np.int64), np.zeros(n, bool), np.zeros(n, bool)
    fa_leg = np.full(n, -1, np.int64)
    diag = dict(lf_rows=0, lf_matched=0, lf_hi_mismatch=0, lf_status_not_ok=0, legs_recv_before_settle=int((frn // 1_000_000 < settle).sum()))
    if lf is not None and len(lf):
        need(not lf.duplicated(['sid', 'mk_idx']).any(), 'lifecycle fills: duplicate (sid, mk_idx)')
        J = pd.DataFrame({'sid': _obj(L.sid), 'mk_idx': L.mk_idx.to_numpy(np.int64)}).merge(
            pd.DataFrame({'sid': _obj(lf.sid), 'mk_idx': lf.mk_idx.to_numpy(np.int64), 'lo': lf.lo_ms.to_numpy(np.int64), 'hi': lf.hi_ms.to_numpy(np.int64),
                          'fa': lf.f_after.to_numpy(np.int64), 'st': _obj(lf.status)}), on=['sid', 'mk_idx'], how='left', validate='many_to_one')
        has_lf = J.lo.notna().to_numpy() & in_book
        fa_leg = np.where(has_lf, J.fa.fillna(-1).to_numpy(), -1).astype(np.int64)
        lo_raw = J.lo.fillna(-1).to_numpy().astype(np.int64)
        lf_ok = has_lf & (_obj(J.st) == OK)                       # a lifecycle lower end is used only from a clean fills row
        withheld = has_lf & ~lf_ok & (lo_raw >= 0)                # a bounded lo withheld because its row is flagged
        lo_leg = np.where(lf_ok, lo_raw, -1).astype(np.int64)
        hi_lf = np.where(has_lf, J.hi.fillna(-1).to_numpy(), -1).astype(np.int64)
        diag.update(lf_rows=int(len(lf)), lf_matched=int(has_lf.sum()), lf_hi_mismatch=int((has_lf & (hi_lf != hi)).sum()),
                    lf_status_not_ok=int((has_lf & ~lf_ok).sum()))
    else:
        withheld = np.zeros(n, bool)
    G = pd.DataFrame({'rk': rk, 'lo': lo_leg, 'lf': has_lf, 'bad': withheld, 'hi': hi}).groupby('rk')
    s_lo, any_lf, hi_r = G.lo.transform('max').to_numpy(), G.lf.transform('max').to_numpy(bool), G.hi.transform('min').to_numpy()
    any_bad = G.bad.transform('max').to_numpy(bool)
    ts = np.full(n, T_ST.index('hi_no_lifecycle'), np.int8)
    ts[in_book & ~any_lf] = T_ST.index('hi_missing_life_fill')
    ts[in_book & any_lf & (s_lo < 0)] = T_ST.index('hi_lo_unbounded')
    ts[in_book & any_lf & (s_lo < 0) & any_bad] = T_ST.index('hi_fill_status')
    ts[in_book & any_lf & (s_lo >= 0) & (s_lo > hi_r)] = T_ST.index('hi_window_inverted')
    lo_ok = in_book & any_lf & (s_lo >= 0) & (s_lo <= hi_r)
    ts[lo_ok] = T_ST.index('lo')
    t = np.where(lo_ok, s_lo, hi_r).astype(np.int64)                    # spread anchor (m_minus): lo on BOOK, else settle_ms
    win_ms = np.where(lo_ok, settle - s_lo, -1).astype(np.int64)        # the fill window settle_ms - lo (-1: no lower end)
    # ---- resolution and marks: the spread at t; v_spr_hi, markouts, beyond_end and after_end from settle_ms (the upper bound) ----
    V = np.where(inreg, reg.V.to_numpy(float)[rm], np.nan)
    end_ms = np.where(inreg, reg.end_ms.to_numpy(np.int64)[rm], -1)
    um, ui, inv = np.unique(mid, return_index=True, return_inverse=True)
    j0 = np.full(n, -1, np.int64)
    if population == 'ID':
        nb = MARK_ST.index('id_day_no_book')
        m0, age0, st0 = np.full(n, np.nan), np.full(n, -1, np.int64), np.full(n, nb, np.int8)
        m0h, st0h = np.full(n, np.nan), np.full(n, nb, np.int8)
        mh = {h: (np.full(n, np.nan), np.full(n, nb, np.int8)) for h, _ in HORIZONS}
        me_u, se_u = np.full(len(um), np.nan), np.full(len(um), nb, np.int8)
        st_end = np.full(n, nb, np.int8)
    else:
        m0, age0, st0, j0 = mid_at(Fr, mid, t, ret_j=True)
        m0h, _, st0h = mid_at(Fr, mid, settle)
        mh = {}
        for h, dh in HORIZONS:
            a, _, b = mid_at(Fr, mid, settle + dh)
            b[settle + dh > end_ms] = MARK_ST.index('beyond_end')
            a[b != 0] = np.nan
            mh[h] = (a, b)
        me_u, _, se_u = mid_at(Fr, um, end_ms[ui])
        st_end = np.where(settle > end_ms, MARK_ST.index('after_end'), se_u[inv]).astype(np.int8)
    m_end = me_u[inv]
    # spread width and the fill's depth at the m_minus frame (ticks from the best on the maker's side; inside = better than the shown best)
    okj = (st0 == 0) & (j0 >= 0)
    jj = np.maximum(j0, 0)
    bbj = np.where(okj, Fr['bb'][jj] if Fr is not None and len(Fr['t']) else np.nan, np.nan)
    baj = np.where(okj, Fr['ba'][jj] if Fr is not None and len(Fr['t']) else np.nan, np.nan)
    spr_w = np.where(okj, baj - bbj, np.nan)
    tpu = np.where(p > 0, np.rint(L.y_ptick.to_numpy(float) / np.where(p > 0, p, 1.)), np.nan)
    dtk = np.rint(np.where(ys == 'ask', p - baj, bbj - p) * tpu)
    dpt = np.where(~okj | ~np.isfinite(dtk), DEPTH.index('unknown'), np.where(dtk < 0, DEPTH.index('inside'), np.minimum(np.nan_to_num(dtk), 3)))
    npx = pd.Series(np.round(p, 6)).groupby(rk).transform('nunique').to_numpy()
    sweep = np.where(npx >= 2, 0, 1)
    bmap = boost or {}
    bst = np.array([BOOST.index('unknown') if bmap.get(int(m_)) is None else (0 if bmap.get(int(m_)) else 1) for m_ in mid], np.int16) if n else \
        np.zeros(0, np.int16)
    # spread-clock diagnostic (BOOK): does the last depth frame <= settle_ms already show the trade (at / after the lifecycle evidence frame)?
    if depth and in_book.any():
        mb = in_book & (st == 0) & lf_ok & (fa_leg >= 0)
        Rb = pd.DataFrame({'rk': rk[mb], 'fa': fa_leg[mb], 'mid': mid[mb], 'settle': settle[mb]}).groupby('rk').agg(
            fa=('fa', 'min'), mid=('mid', 'first'), settle=('settle', 'first'))
        n_ev, n_show, n_miss = 0, 0, 0
        for fa_, m_, se_ in Rb.itertuples(index=False, name=None):
            D_ = depth.get(int(m_))
            if D_ is None or int(fa_) not in D_[1]:
                n_miss += 1
                continue
            n_ev += 1
            n_show += int(np.searchsorted(D_[0], se_, 'right') - 1 >= D_[1][int(fa_)])
        nb_ok = int(pd.Series(rk[in_book & (st == 0)]).nunique())
        diag.update(clock_book_records=nb_ok, clock_with_evidence=n_ev, clock_hi_shows_trade=n_show, clock_evidence_frame_missing=n_miss,
                    clock_no_evidence=nb_ok - n_ev - n_miss)
    end_mkt = {int(m_): (float(a_), MARK_ST[int(b_)]) for m_, a_, b_ in zip(um, me_u, se_u)}
    # ---- rows: maker block then taker block ----
    s, qq, pp, VV = np.r_[s_mk, -s_mk], np.r_[q, q], np.r_[p, p], np.r_[V, V]
    own = np.r_[mki, tki]
    Qr = np.where(has, Q[rj], np.nan)
    fee = np.r_[mkf, np.where(has & (Qr > 0), tkf * q / np.where(Qr > 0, Qr, 1.), 0.)]
    ftype = np.r_[mkft, tkft]
    dy, dc = fee_pos(fee, ftype, own)
    gross = s * qq * (VV - pp)
    fee_fill = -(dc + dy * pp)
    fee_res = -(dc + np.where(dy != 0, dy * VV, 0.))
    reb_res, reb_fill = np.r_[REB * fee_res[n:], np.zeros(n)], np.r_[REB * fee_fill[n:], np.zeros(n)]   # maker rows: 25 % of the leg's taker fee

    def val(mx, sx):
        m2, s2 = np.r_[mx, mx], np.r_[sx, sx]
        return np.where(s2 == 0, s * qq * (m2 - pp), np.nan), s2

    tr = L.t_rel_s.to_numpy(float)
    tseg = np.full(n, TSEG.index('unknown'), np.int16)
    fin = np.isfinite(tr)
    tseg[fin & (tr < 0)] = TSEG.index('pre')
    tseg[fin & (tr >= 300)] = TSEG.index('post')
    w = fin & (tr >= 0) & (tr < 300)
    tseg[w] = 1 + np.floor(tr[w] / 60).astype(np.int16)
    pu = np.rint(np.nan_to_num(p) * U).astype(np.int64)
    pseg = np.where(pu <= U // 10, 4, np.where(pu >= 9 * U // 10, 5, np.minimum(np.abs(pu - U // 2) // (U // 10), 3)))
    selft = _obj(L.mk_signer) == tks
    uc = pd.api.types.union_categoricals
    sc = L.sid.cat.codes.to_numpy()
    F = pd.DataFrame({'day': cat(np.zeros(2 * n), (d,)), 'mid': np.r_[mid, mid], 'sid': pd.Categorical.from_codes(np.r_[sc, sc], L.sid.cat.categories),
                      'mk_idx': np.r_[L.mk_idx.to_numpy(np.int64), L.mk_idx.to_numpy(np.int64)], 'px_rank': np.r_[px_rank, px_rank],
                      'role': cat(np.r_[np.zeros(n), np.ones(n)], ('maker', 'taker')), 'addr': uc([L.mk_signer, L.tk_signer]),
                      'hash': uc([L.mk_hash, L.tk_hash]), 'cp_addr': uc([L.tk_signer, L.mk_signer]), 'kind': cat(np.r_[kind, kind], KIND),
                      'own_index_set': own.astype(np.int64), 'y_side': cat(np.where(s == 1, 0, 1), ('bid', 'ask')), 's': s, 'q': qq, 'p': pp,
                      'px_gap': np.r_[px_gap, px_gap], 'self_trade': np.r_[selft, selft].astype(bool), 'settle_ms': np.r_[settle, settle], 't_ms': np.r_[t, t],
                      'win_ms': np.r_[win_ms, win_ms], 't_status': cat(np.r_[ts, ts], T_ST), 't_rel_s': np.r_[tr, tr], 'tseg': cat(np.r_[tseg, tseg], TSEG),
                      'pseg': cat(np.r_[pseg, pseg], PSEG), 'V': VV,
                      'st_res': cat(np.where(np.r_[cen, cen], 2, np.where(np.r_[upb, upb], 4, np.where(np.r_[tie, tie], 3, np.where(np.isfinite(VV), 0, 1)))),
                                    RES_ST),
                      'gross_res': gross})
    F['m_minus'], F['age_ms'], F['spr_w'] = np.r_[m0, m0], np.r_[age0, age0], np.r_[spr_w, spr_w]
    v_, s_ = val(m0, st0)
    F['st_spr'], F['v_spr'] = cat(s_, MARK_ST), v_
    v_, s_ = val(m0h, st0h)
    F['st_spr_hi'], F['v_spr_hi'] = cat(s_, MARK_ST), v_
    for h, _ in HORIZONS:
        v_, s_ = val(*mh[h])
        F[f'st_{h}'], F[f'v_{h}'] = cat(s_, MARK_ST), v_
    F['m_end'] = np.r_[m_end, m_end]
    v_, s_ = val(m_end, st_end)
    F['st_end'], F['v_end'] = cat(s_, MARK_ST), v_
    F['fee'], F['fee_type'], F['fee_dy'], F['fee_dc'] = fee, pd.Categorical(ftype), dy, dc
    F['fee_fill'], F['fee_res'], F['net_res'] = fee_fill, fee_res, gross - fee_res
    F['reb25_fill'], F['reb25_res'] = reb_fill, reb_res
    F['dpt'], F['sweep'], F['boost'] = cat(np.r_[dpt, dpt], DEPTH), cat(np.r_[sweep, sweep], SWEEP), cat(np.r_[bst, bst], BOOST)
    F['cp'] = cat(np.full(2 * n, CPB.index('na_id_day')) if population == 'ID' else np.where(_isin(_obj(F.cp_addr), top_set), 0, 1), CPB)
    F['src_status'] = pd.Categorical(np.r_[src, src])
    F['status'] = cat(np.r_[st, st], ROW_ST)
    F['_ptick'] = np.r_[L.y_ptick.to_numpy(np.int64), L.y_ptick.to_numpy(np.int64)]
    # ---- record-level diagnostics (taker average price vs its legs; fee formula) ----
    ok_leg = st == 0
    rec = pd.DataFrame({'rk': ri[ok_leg], 'q': q[ok_leg], 'qp': q[ok_leg] * p[ok_leg], 'qm': q[ok_leg] * np.minimum(p[ok_leg], 1 - p[ok_leg]),
                        'tp': tp_[ok_leg], 'f': tkf[ok_leg], 'ft': tkft[ok_leg], 'own': tki[ok_leg], 'mid': mid[ok_leg]}).groupby('rk').agg(
        q=('q', 'sum'), qp=('qp', 'sum'), qm=('qm', 'sum'), tp=('tp', 'first'), f=('f', 'first'), ft=('ft', 'first'), own=('own', 'first'), mid=('mid', 'first'))
    gap = rec.tp - rec.qp / rec.q
    big = gap.abs() > TOL_PX
    diag.update(records=int(len(rec)), tk_px_gap_n=int(big.sum()), tk_px_gap_max=float(gap.abs().max()) if len(rec) else 0.,
                tk_px_gap_usd=float((gap.abs() * rec.q)[big].sum()))      # |taker avg - legs avg| x shares: what pricing at priceExecuted would move
    diag['_fee'] = rec.assign(pavg=rec.qp / rec.q)[['mid', 'q', 'qm', 'pavg', 'f', 'ft', 'own']]
    return F, end_mkt, diag


def fifo(F):
    """FIFO lots per address x market over the valid rows (FIFO): mq = shares closed against the opposite side, rq = q - mq. Self-trade
    rows are not lots (mq = rq = 0); excluded rows get NaN."""
    ok, selft = (F.status == OK).to_numpy(), F.self_trade.to_numpy(bool)
    v = ok & ~selft
    mq, rq = np.full(len(F), np.nan), np.full(len(F), np.nan)
    mq[ok & selft], rq[ok & selft] = 0., 0.
    ix = np.flatnonzero(v)
    if len(ix):
        ac = F.addr.cat.codes.to_numpy()[ix].astype(np.int64)
        need(ac.max() < 1 << 21, 'too many addresses in one chunk for the FIFO key')
        g = pd.factorize((F.mid.to_numpy()[ix] << 21) | ac)[0]
        s, q = F.s.to_numpy()[ix], F.q.to_numpy()[ix]
        cats = _obj(F.sid.cat.categories)
        sid_rank = np.empty(len(cats), np.int64)
        sid_rank[np.argsort(cats.astype(str), kind='mergesort')] = np.arange(len(cats))
        o = np.lexsort((F.role.cat.codes.to_numpy()[ix], F.mk_idx.to_numpy()[ix], F.px_rank.to_numpy()[ix], sid_rank[F.sid.cat.codes.to_numpy()[ix]],
                        F.settle_ms.to_numpy()[ix], s, g))
        gs, ss, qs = g[o], s[o], q[o]
        blk = np.cumsum(np.r_[True, (gs[1:] != gs[:-1]) | (ss[1:] != ss[:-1])]) - 1
        cq = pd.Series(qs).groupby(blk).cumsum().to_numpy()
        tb, ts_ = np.zeros(g.max() + 1), np.zeros(g.max() + 1)
        np.add.at(tb, gs[ss == 1], qs[ss == 1])
        np.add.at(ts_, gs[ss == -1], qs[ss == -1])
        X = np.minimum(tb, ts_)[gs]
        own_tot = np.where(ss == 1, tb[gs], ts_[gs])                 # the same floats as X: the smaller side is matched in full
        m_ = np.where(own_tot <= X, qs, np.clip(np.minimum(cq, X) - (cq - qs), 0., qs))
        mq[ix[o]], rq[ix[o]] = m_, qs - m_
    F['mq'], F['rq'] = mq, rq


def life_join(F, Lc, orders, book_mids):
    """Lifecycle classes of the maker rows of BOOK markets from the S_FIFO row of their hash (SEGMENTS); other rows get na_* buckets.
    life_status: missing_life (no row), life_mismatch (signer / side / tick differ: H1), nfills_mismatch (n_fills != legs of the hash),
    bad_removal_set (a token outside pf_mt_common.REMOVAL), orders_status (row status not ok) -> buckets life_not_ok."""
    n = len(F)
    mk = (F.role == 'maker').to_numpy()
    bk = _isin(F.mid.to_numpy(), book_mids)
    rep, exh, fr, lst = (np.full(n, k, np.int16) for k in (LIFE_REP.index('na_taker'), LIFE_EXH.index('na_taker'), LIFE_FRONT.index('na_taker'),
                                                          LIFE_ST.index('na_taker')))
    sel = mk & ~bk
    rep[sel], exh[sel], fr[sel], lst[sel] = LIFE_REP.index('na_not_book'), LIFE_EXH.index('na_not_book'), LIFE_FRONT.index('na_not_book'), LIFE_ST.index('na_not_book')
    ix = np.flatnonzero(mk & bk)
    if len(ix):
        O = orders if orders is not None else pd.DataFrame(columns=ORD_COLS)
        O = O[_obj(O.scenario) == 'S_FIFO']
        need(not O.duplicated(['mid', 'mk_hash']).any(), 'lifecycle orders: duplicate (mid, mk_hash) in scenario S_FIFO')
        O = pd.DataFrame({'mid': O.mid.to_numpy(np.int64), 'h': _obj(O.mk_hash), 'o_signer': _obj(O.mk_signer), 'o_side': _obj(O.y_side),
                          'o_tick': O.y_ptick.to_numpy(np.int64), 'o_nf': O.n_fills.to_numpy(np.int64), 'o_rs': _obj(O.removal_set),
                          'o_ah': O.ahead_hi.to_numpy(float), 'o_st': _obj(O.status)})
        nl = pd.DataFrame({'mid': Lc.mid.to_numpy(np.int64), 'h': _obj(Lc.mk_hash)}).value_counts().rename('nl').reset_index()
        K = pd.DataFrame({'mid': F.mid.to_numpy()[ix], 'h': _obj(F.hash)[ix]}).merge(O, on=['mid', 'h'], how='left', validate='many_to_one').merge(
            nl, on=['mid', 'h'], how='left', validate='many_to_one')
        found = K.o_nf.notna().to_numpy()
        toks = [set(re.findall(r'[a-z_]+', x)) if isinstance(x, str) else set() for x in K.o_rs]
        badrs = np.array([bool(t_ - set(C.REMOVAL)) for t_ in toks], bool)
        mism = found & ((K.o_signer.to_numpy() != _obj(F.addr)[ix]) | (K.o_side.to_numpy() != _obj(F.y_side)[ix]) | (K.o_tick.to_numpy() != F._ptick.to_numpy()[ix]))
        code = np.where(~found, LIFE_ST.index('missing_life'), np.where(mism, LIFE_ST.index('life_mismatch'),
                        np.where(found & (K.o_nf.to_numpy() != K.nl.to_numpy()), LIFE_ST.index('nfills_mismatch'),
                                 np.where(badrs, LIFE_ST.index('bad_removal_set'), np.where(K.o_st.to_numpy() != OK, LIFE_ST.index('orders_status'), 0)))))
        okl = code == 0
        e = np.array([2 if 'exhausted' not in t_ else (0 if t_ == {'exhausted'} else 1) for t_ in toks], np.int16)
        ah = K.o_ah.to_numpy(float)
        rep[ix] = np.where(okl, np.where(K.o_nf.to_numpy() >= 2, 0, 1), LIFE_REP.index('life_not_ok'))
        exh[ix] = np.where(okl, e, LIFE_EXH.index('life_not_ok'))
        fr[ix] = np.where(okl, np.where(ah == 0, 0, np.where(ah > 0, 1, 2)), LIFE_FRONT.index('life_not_ok'))
        lst[ix] = code
    F['life_rep'], F['life_exh'], F['life_front'], F['life_status'] = cat(rep, LIFE_REP), cat(exh, LIFE_EXH), cat(fr, LIFE_FRONT), cat(lst, LIFE_ST)


def contrib(F):
    """Per-row additive contributions: valid rows only (status ok); resolution terms only where st_res is ok, each mark value and its
    shares only where its status is ok."""
    v = (F.status == OK).to_numpy()
    q = np.where(v, F.q.to_numpy(), 0.)
    res = v & _isin(F.st_res, RES_VALID)
    c = {'n': v.astype(np.int64), 'sh': q, 'notional': q * F.p.to_numpy(), 'gross_res': np.where(res, F.gross_res.to_numpy(), 0.),
         'sh_res': np.where(res, q, 0.), 'fee_res': np.where(res, F.fee_res.to_numpy(), 0.), 'fee_fill': np.where(v, F.fee_fill.to_numpy(), 0.),
         'net_res': np.where(res, F.net_res.to_numpy(), 0.), 'n_res_na': (v & ~res).astype(np.int64),
         'reb25_res': np.where(res, F.reb25_res.to_numpy(), 0.), 'reb25_fill': np.where(v, F.reb25_fill.to_numpy(), 0.)}
    for h in VALS:
        o = v & (F[f'st_{h}'] == OK).to_numpy()
        c[f'v_{h}'], c[f'sh_{h}'] = np.where(o, F[f'v_{h}'].to_numpy(), 0.), np.where(o, q, 0.)
    return pd.DataFrame(c), v


def market_table(F, d, population, book_mids, end_mkt, reg):
    """fill rows of one chunk -> maker_market rows (day, mid, addr): sums by role, FIFO decomposition and statuses."""
    c, v = contrib(F)
    mk = (F.role == 'maker').to_numpy()
    s, p, V = F.s.to_numpy(), F.p.to_numpy(), F.V.to_numpy()
    mq, rq = np.where(v, np.nan_to_num(F.mq.to_numpy()), 0.), np.where(v, np.nan_to_num(F.rq.to_numpy()), 0.)
    mid = F.mid.to_numpy()
    me = pd.Series({k: a for k, (a, b) in end_mkt.items()})
    end_ok = pd.Series({k: b == OK for k, (a, b) in end_mkt.items()})
    m_end = me.reindex(mid).to_numpy(float)
    X = {'n_mk': c.n * mk, 'sh_mk': c.sh * mk, 'n_tk': c.n * ~mk, 'sh_tk': c.sh * ~mk, 'notional': c.notional, 'sh_buy': c.sh * (s == 1),
         'sh_sell': c.sh * (s == -1), 'gross_res_mk': c.gross_res * mk, 'gross_res_tk': c.gross_res * ~mk, 'sh_res': c.sh_res, 'sh_res_mk': c.sh_res * mk,
         'fee_res': c.fee_res, 'fee_fill': c.fee_fill, 'net_res': c.net_res, 'net_res_mk': c.net_res * mk, 'reb25_res': c.reb25_res,
         'reb25_fill': c.reb25_fill, 'n_res_na': c.n_res_na}
    for h in VALS:
        for r, msk in (('mk', mk), ('tk', ~mk)):
            X[f'v_{h}_{r}'], X[f'sh_{h}_{r}'] = c[f'v_{h}'] * msk, c[f'sh_{h}'] * msk
    X.update(rt_sh=np.where(s == 1, mq, 0.), rt_pnl=-s * p * mq, net_pos=s * rq, resid_res=np.where(np.isfinite(V), s * rq * (V - p), 0.),
             resid_drift=np.where(np.isfinite(m_end), s * rq * (m_end - p), 0.), n_excl=(~v).astype(np.int64))
    X = pd.DataFrame(X)
    X['mid'], X['addr'] = mid, _obj(F.addr)
    M = X.groupby(['mid', 'addr'], sort=True, dropna=False).sum().reset_index()
    rg = reg.set_index('mid')
    Vm = rg.V.reindex(M.mid).to_numpy(float)
    res_ok = np.isfinite(Vm)
    eok = end_ok.reindex(M.mid).fillna(False).to_numpy(bool)
    em = me.reindex(M.mid).to_numpy(float)
    cen_m = rg.censored.reindex(M.mid).fillna(False).to_numpy(bool)
    upb_m = rg.upbad.reindex(M.mid).fillna(False).to_numpy(bool)
    tie_m = rg.tie.reindex(M.mid).fillna(False).to_numpy(bool)
    M['res_status'] = np.where(cen_m, 'censored', np.where(upb_m, 'up_index', np.where(tie_m, 'tie', np.where(res_ok, OK, 'no_resolution'))))
    M['end_status'] = [end_mkt.get(int(m_), (np.nan, 'no_market'))[1] for m_ in M.mid]
    M['resid_res'] = np.where(res_ok, M.resid_res, np.nan)
    M['resid_drift'] = np.where(eok, M.resid_drift, np.nan)
    M['resid_resol'] = np.where(res_ok & eok, (Vm - em) * M.net_pos, np.nan)
    M['fifo_status'] = np.where(cen_m, 'censored', np.where(upb_m, 'up_index', np.where(~res_ok, 'no_resolution', np.where(~eok, 'end_unavailable', OK))))
    M['day'], M['population'] = d, population
    M['book'] = _isin(M.mid.to_numpy(), book_mids)
    M['is_maker'] = M.n_mk > 0
    for k in REW:
        M[k] = 0 if k in INTS else 0.
    M['rew_status'] = np.where(M.book, 'pending', 'not_book')
    M['status'] = np.where(M.n_mk + M.n_tk > 0, OK, 'all_rows_excluded')
    for k in INTS & set(M.columns):
        M[k] = M[k].astype(np.int64)
    return M[list(MM)]


def check_market(M):
    """Conservation (per address x market) and zero-sum closure (per market) -> max deviations; refuses on any violation."""
    tol = TOL_ABS + TOL_REL * M.notional.abs().to_numpy()
    gross = (M.gross_res_mk + M.gross_res_tk).to_numpy()
    r = (M.res_status == OK).to_numpy()
    e = r & (M.end_status == OK).to_numpy()
    devs = {}

    def chk(name, dev, mask, t_):
        dv = np.abs(np.asarray(dev, float))[mask]
        devs[name] = float(dv.max()) if len(dv) else 0.
        need(bool(np.all(dv <= t_[mask])), f'conservation: {name} violated (max deviation {devs[name]:.3g})')

    chk('gross_eq_rt_plus_resid', gross - (M.rt_pnl + M.resid_res).to_numpy(), r, tol)
    chk('resid_eq_drift_plus_resol', (M.resid_res - (M.resid_drift + M.resid_resol)).to_numpy(), e, tol)
    chk('net_eq_gross_minus_fee', M.net_res.to_numpy() - (gross - M.fee_res.to_numpy()), r, tol)
    chk('net_pos_eq_buys_minus_sells', (M.net_pos - (M.sh_buy - M.sh_sell)).to_numpy(), np.ones(len(M), bool), tol)
    zcols = ['gross_res_mk', 'gross_res_tk', 'net_pos', 'sh_buy', 'sh_sell', 'notional'] + [f'v_{h}_{r_}' for h in VALS for r_ in ('mk', 'tk')]
    Z = M.groupby('mid')[zcols].sum()
    zt = TOL_ABS + TOL_REL * Z.notional.to_numpy()
    allm = np.ones(len(Z), bool)
    for name, x in [('zero_sum_gross', Z.gross_res_mk + Z.gross_res_tk), ('zero_sum_net_pos', Z.net_pos), ('zero_sum_shares', Z.sh_buy - Z.sh_sell)] + \
            [(f'zero_sum_v_{h}', Z[f'v_{h}_mk'] + Z[f'v_{h}_tk']) for h in VALS]:
        dv = np.abs(x.to_numpy(float))
        devs[name] = float(dv.max()) if len(dv) else 0.
        need(bool(np.all(dv[allm] <= zt)), f'zero-sum: {name} violated in a market (max deviation {devs[name]:.3g})')
    return devs


SKEY = ['a', 'role', 'dim', 'bucket']


def day_codes(cat_series, index):
    """Day-level integer codes (positions in `index`) of a categorical's values; -1 for missing."""
    pos = index.get_indexer(_obj(cat_series.cat.categories))
    c = cat_series.cat.codes.to_numpy()
    return np.where(c >= 0, pos[np.maximum(c, 0)], -1).astype(np.int64)


def seg_partial(F, c, v, a_codes):
    """Per-address partial sums of the valid rows for every SEGMENTS dimension, keyed by integers (a = day address code, role 0 maker /
    1 taker, dim = SEG_DIMS index, bucket = BUCKETS index): compact, so a day's running sum stays small."""
    base = c[v].reset_index(drop=True)
    base['a'], base['role'] = a_codes[v], F.role.cat.codes.to_numpy()[v].astype(np.int64)
    out = []
    for di, dim in enumerate(SEG_DIMS):
        if dim == 'all':
            b = np.zeros(int(v.sum()), np.int64)
        elif dim == 'tclock':
            b = np.where(_obj(F.t_status)[v] == 'lo', B_IX['lo'], B_IX['hi'])
        elif dim == 'self':
            b = np.where(F.self_trade.to_numpy(bool)[v], B_IX['self'], B_IX['not_self'])
        else:
            col = DIM_COL.get(dim, dim)
            b = np.array([B_IX[x] for x in F[col].cat.categories], np.int64)[F[col].cat.codes.to_numpy()[v]]
        out.append(base.assign(dim=di, bucket=b).groupby(SKEY, sort=False).sum().reset_index())
    return pd.concat(out, ignore_index=True)


def fold(run_, part, keys):
    """Running sum: (running table, new partial) -> one row per key."""
    if run_ is None:
        return part.groupby(keys, sort=False).sum().reset_index()
    return pd.concat([run_, part], ignore_index=True).groupby(keys, sort=False).sum().reset_index()


def presence(O, lfm, Fr, mid, p_of, thr_sh, thr_sp, w_lo, w_hi):
    """REWARDS lower bound for one BOOK market -> {addr: dict(bid=[intervals], ask=[intervals], n=orders with time)}."""
    out = {}
    c = int(np.searchsorted(Fr['um'], mid)) if Fr is not None and len(Fr['um']) else 0
    if Fr is None or c >= len(Fr['um']) or Fr['um'][c] != mid:
        return out
    t, m, stf = (Fr[k][Fr['lo'][c]:Fr['hi'][c]] for k in ('t', 'm', 'st'))
    thr_u = int(round(thr_sp * U))
    O = O[(_obj(O.scenario) == 'S_NOPRIO') & _isin(O.persist_class, (C.OBSERVED, C.CONSTRAINED)) & (O.size_lb.to_numpy(float) >= thr_sh)
          & (_obj(O.status) == OK) & (O.persist_lo_ms.to_numpy() >= 0) & (O.persist_hi_ms.to_numpy() > O.persist_lo_ms.to_numpy())]
    for o in O.itertuples(index=False):
        f = lfm[(_obj(lfm.mk_hash) == o.mk_hash) & (lfm.lo_ms.to_numpy() >= 0) & (_obj(lfm.status) == OK)].sort_values('lo_ms', ascending=False,
                                                                                                                   kind='mergesort')
        cum = np.cumsum(f.shares.to_numpy(float))
        k = np.flatnonzero(cum >= thr_sh)
        if not len(k) or o.mk_hash not in p_of:
            continue
        a, b = max(int(o.persist_lo_ms), w_lo), min(int(o.persist_hi_ms), int(f.lo_ms.to_numpy()[k[0]]), w_hi)
        if b <= a:
            continue
        pu = int(round(p_of[o.mk_hash] * U))
        i0, i1 = max(int(np.searchsorted(t, a, 'right')) - 1, 0), int(np.searchsorted(t, b, 'left'))
        j = np.arange(i0, max(i1, i0))
        inb = (stf[j] == 0) & (np.abs(pu - np.rint(np.nan_to_num(m[j], nan=-1.) * U)) < thr_u)      # strict; non-ok frames never count
        e = np.minimum(t[j] + STALE_MS, np.where(j + 1 < len(t), t[np.minimum(j + 1, len(t) - 1)], t[j] + STALE_MS))
        lo_, hi_ = np.maximum(a, t[j]), np.minimum(b, e)
        sel = inb & (hi_ > lo_)
        iv = list(zip(lo_[sel].tolist(), hi_[sel].tolist()))
        if iv:
            D = out.setdefault(o.mk_signer, dict(bid=[], ask=[], n=0))
            D.setdefault(o.y_side, []).extend(iv)
            D['n'] += 1
    return out


def union(iv):
    u = []
    for a, b in sorted(iv):
        if u and a <= u[-1][1]:
            u[-1][1] = max(u[-1][1], b)
        else:
            u.append([a, b])
    return u


def measure(u):
    return sum(b - a for a, b in u)


def overlap(x, y):
    i = j = tot = 0
    while i < len(x) and j < len(y):
        lo, hi = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        tot += max(0, hi - lo)
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return tot


def depth_bounds(fr, lv, thr_sp, thr_sh, tpu, w_lo, w_hi):
    """REWARDS upper bound of one BOOK market, per side: over usable (BAD_FRAME rule) two-sided depth frames inside [w_lo, w_hi), the shown
    shares at levels strictly within spreadThreshold of the frame's own mid x the frame's duration (to the next frame, at most 1 s), and the
    variant counting only levels with >= shareThreshold shares. Share-seconds; an upper bound because shown depth includes everyone."""
    out = dict(n_frames_used=0, ub_bid_shs=0., ub_ask_shs=0., ub_bid_shs_thr=0., ub_ask_shs_thr=0., n_level_rows_other_side=0, n_level_rows_not_ok=0)
    if fr is None or not len(fr) or not np.isfinite(tpu):
        return out
    lok = _obj(lv.status) == OK                        # only level rows with status ok count (for the frame's best and for the bound)
    out['n_level_rows_not_ok'] = int((~lok).sum())
    lv = lv[lok]
    o = np.lexsort((fr.recv_ns.to_numpy(np.int64), fr.upd_ms.to_numpy(np.int64)))
    fid, t = fr.fid.to_numpy(np.int64)[o], fr.upd_ms.to_numpy(np.int64)[o]
    use = ~flagged_any(fr.status, BAD_FRAME)[o]
    e = np.minimum(np.r_[t[1:], t[-1] + STALE_MS], t + STALE_MS)
    dur = np.maximum(np.minimum(e, w_hi) - np.maximum(t, w_lo), 0) / 1000.
    sd = _obj(lv.side)
    side = np.where(_isin(sd, ('bid', 'bids', 'b')), 'bid', np.where(_isin(sd, ('ask', 'asks', 'a')), 'ask', 'other'))
    pt, sh, lf_ = lv.ptick.to_numpy(np.int64), lv.shares.to_numpy(float), lv.fid.to_numpy(np.int64)
    big = 1 << 40
    best = pd.DataFrame({'fid': lf_, 'bb': np.where(side == 'bid', pt, -1), 'ba': np.where(side == 'ask', pt, big)}).groupby('fid').agg(
        bb=('bb', 'max'), ba=('ba', 'min'))
    Fx = pd.DataFrame({'fid': fid, 'dur': dur, 'use': use}).drop_duplicates('fid').set_index('fid').join(best, how='left')
    Fx = Fx[Fx.use & (Fx.dur > 0) & (Fx.bb >= 0) & (Fx.ba < big) & (Fx.bb < Fx.ba)]
    J = pd.DataFrame({'fid': lf_, 'side': side, 'pt': pt, 'sh': sh}).join(Fx[['dur', 'bb', 'ba']], on='fid', how='inner')
    el = (np.abs(2 * J.pt.to_numpy() - (J.bb.to_numpy() + J.ba.to_numpy())) < np.rint(2 * thr_sp * tpu)) & (J.side.to_numpy() != 'other')
    w_ = J.sh.to_numpy() * J.dur.to_numpy() * el
    thr_ok = J.sh.to_numpy() >= thr_sh
    for s_ in ('bid', 'ask'):
        m_ = J.side.to_numpy() == s_
        out[f'ub_{s_}_shs'], out[f'ub_{s_}_shs_thr'] = float(w_[m_].sum()), float(w_[m_ & thr_ok].sum())
    out.update(n_frames_used=int(len(Fx)), n_level_rows_other_side=int((side == 'other').sum()))
    return out


def rewards_rows(d, reg, rew_inv, bm, cens, dfr, dlv, L, rew_by_mkt, sets):
    """econ/rewards_market rows of one FILL day, one per market (registry + markets table): the reward pool (units unverified; none for a
    censored or up_index market); on BOOK markets also the depth upper bound per side (depth_bounds) and each identity set's proven lower
    bound (the sum of its members' per-address lower bounds). Per-maker rewards are unidentifiable: neither bound is a maker's share."""
    inv = {r['mid']: r for r in rew_inv} if isinstance(rew_inv, list) else {}
    rg = reg.set_index('mid')
    rows = []
    for m_ in sorted({int(x) for x in reg.mid} | set(inv)):
        r = inv.get(m_, {})
        inr = m_ in rg.index
        st_ = CENSOR if m_ in cens else ('up_index' if inr and bool(rg.upbad[m_]) else OK)
        thr_sp = r.get('spread_thr', rg.spread_thr[m_] if inr and 'spread_thr' in rg.columns else None)
        thr_sh = r.get('share_thr', rg.share_thr[m_] if inr and 'share_thr' in rg.columns else None)
        b = r.get('boosted')
        row = dict(day=d, mid=m_, book=m_ in bm, table_status=r.get('table_status'),
                   inv_status=r.get('status', rew_inv if isinstance(rew_inv, str) else 'markets_row_missing'),
                   n_variants=int((r.get('reward_variants') or {}).get('n', 0)), pool_units=r.get('pool_units') if st_ == OK else None,
                   spread_thr=thr_sp, share_thr=thr_sh, boosted='unknown' if b is None else ('boosted' if b else 'not_boosted'),
                   depth_status='not_book', n_frames_used=0, status=st_)
        if m_ in bm:
            if st_ != OK:
                row['depth_status'] = st_
            elif thr_sp is None or thr_sh is None or pd.isna(thr_sp) or pd.isna(thr_sh) or not inr:
                row['depth_status'] = 'thresholds_missing'
            else:
                x = L[L.mid == m_]
                yp, yt = x.y_price.to_numpy(float), x.y_ptick.to_numpy(float)
                tpu = float(np.median(np.rint(yt[yp > 0] / yp[yp > 0]))) if (yp > 0).any() else np.nan
                bd = depth_bounds(dfr.get(m_), dlv.get(m_), float(thr_sp), float(thr_sh), tpu, int(r.get('sched_lo_ms') or rg.ws_ms[m_]),
                                  int(r.get('sched_hi_ms') or rg.we_ms[m_]))
                row.update(depth_status=OK, **{k: v for k, v in bd.items() if k in RM})
                rb = rew_by_mkt.get(m_, {})
                for s_ in RM_SETS:
                    mem = set(sets[s_][1]) if s_ in sets else set()
                    for i, k in enumerate(('bid', 'ask', 'both')):
                        row[f'lb_{k}_ms_{s_}'] = float(sum(v[i] for a, v in rb.items() if a in mem))
        rows.append(row)
    return pd.DataFrame(rows, columns=list(RM)).astype({'n_frames_used': np.int64, 'n_variants': np.int64})


def _parse_ts(x):
    import datetime as dt
    return dt.datetime.fromisoformat(str(x).replace('Z', '+00:00')).timestamp()


def rewards_inventory(mk, reg):
    """markets_<day> -> per market: fee / reward fields as published (first row; n_versions = distinct rows), the rewards schedule entries
    overlapping the window and their pool (sum hourlyRate x overlap hours; units as published, not asserted USD), registry cross-check."""
    out, cols = [], {k: next((c for c in names if c in mk.columns), None) for k, names in MKT_FIELDS.items()}
    rg = reg.set_index('mid')
    for mid, G in mk.groupby('mid', sort=True):
        r = dict(mid=int(mid), n_rows=int(len(G)), n_versions=int(len(G.astype(str).drop_duplicates())), source_cols={k: v for k, v in cols.items() if v})
        miss = [k for k, v in cols.items() if v is None]
        for k, cname in cols.items():
            if cname is not None:
                r[k] = _scalar(G[cname].iloc[0])
        st = 'ok' if not miss else 'missing_fields:' + ','.join(miss)
        r['table_status'] = _scalar(G.status.iloc[0]) if 'status' in G.columns else None
        rv = r.pop('rewards_variants', None)
        if isinstance(rv, str):
            try:
                rv = json.loads(rv)
            except ValueError:
                rv, st = None, 'rewards_variants_unparsed'
        variants = []
        if isinstance(rv, (list, dict)):            # extractor shape: [{first_recv_ns, rewards: {current, schedule}}] (rewards unwrapped)
            for v in (rv if isinstance(rv, list) else [rv]):
                v = json.loads(v) if isinstance(v, str) else v
                w = v.get('rewards', v) if isinstance(v, dict) else None
                w = json.loads(w) if isinstance(w, str) else w
                if isinstance(w, dict):
                    variants.append(dict(first_recv_ns=v.get('first_recv_ns'), current=w.get('current'), schedule=w.get('schedule') or []))
            cur = sorted({json.dumps(x['current'], sort_keys=True) for x in variants if x['current'] is not None})
            fr_ns = [x['first_recv_ns'] for x in variants if x['first_recv_ns'] is not None]
            r['reward_variants'] = dict(n=len(variants), n_current_non_null=len(cur), current=[c[:200] for c in cur[:3]],
                                        n_distinct_schedules=len({json.dumps(x['schedule'], sort_keys=True) for x in variants}),
                                        first_recv_ns=[min(fr_ns), max(fr_ns)] if fr_ns else None)
        rw = r.get('rewards')
        if isinstance(rw, str):
            try:
                rw = json.loads(rw)
            except ValueError:
                rw, st = None, 'rewards_unparsed'
        r['rewards'] = None
        # schedule entries: the distinct entries over the rewards_variants (else the rewards object's), each counted once
        raw = [e for x in variants for e in x['schedule']] if variants else ((rw.get('schedule') or []) if isinstance(rw, dict) else [])
        r['pool_source'] = 'rewards_variants' if variants else ('rewards' if isinstance(rw, dict) else None)
        if r['pool_source'] and int(mid) in rg.index:
            ws, we = float(rg.ws[int(mid)]), float(rg.we[int(mid)])
            ent, seen = [], set()
            for e in raw:
                try:
                    key = (float(e['hourlyRate']), str(e['startsAt']), str(e['endsAt']))
                    if key in seen:
                        continue
                    seen.add(key)
                    a, b = _parse_ts(e['startsAt']), _parse_ts(e['endsAt'])
                    ent.append(dict(rate=key[0], lo=a, hi=b, overlap_s=max(0., min(b, we) - max(a, ws))))
                except (KeyError, TypeError, ValueError):
                    st = 'rewards_entry_unparsed'
            r['schedule'] = ent
            r['pool_units'] = float(sum(e['rate'] * e['overlap_s'] / 3600. for e in ent)) if ent else None
            r['current'] = None if not isinstance(rw, dict) or rw.get('current') is None else str(rw.get('current'))
            win = [e for e in ent if e['overlap_s'] > 0]
            r['sched_lo_ms'] = int(round(min(e['lo'] for e in win) * 1000)) if win else None
            r['sched_hi_ms'] = int(round(max(e['hi'] for e in win) * 1000)) if win else None
        elif r['pool_source'] is None and st == 'ok':
            st = 'no_rewards'
        if int(mid) in rg.index:
            r['registry_match'] = {k: bool(np.isclose(float(r[k]), float(rg[k2][int(mid)]))) for k, k2 in (('fee_bps', 'fee_bps'), ('spread_thr', 'spread_thr'),
                                                                                                            ('share_thr', 'share_thr'))
                                   if r.get(k) is not None and k2 in rg.columns and pd.notna(rg[k2][int(mid)])}
        r['status'] = st
        out.append(r)
    return out


def fee_diag(fr_, reg):
    """Fee-formula counts of one chunk's valid records (diagnostic only): formula = feeRateBps/1e4 x sum(min(p, 1-p) x q); a COLLATERAL fee
    matches in USD; a SHARES fee matches either as USD at the fill price (shares x own price) or as an amount equal to the formula."""
    z = dict(n_fee_records=0, coll_n=0, coll_match=0, sh_n=0, sh_match_usd=0, sh_match_amount=0)
    if not len(fr_):
        return z
    rate = reg.set_index('mid').fee_bps.reindex(fr_.mid).to_numpy(float) / 1e4 if 'fee_bps' in reg else np.full(len(fr_), np.nan)
    form = rate * fr_.qm.to_numpy(float)
    pown = np.where(fr_.own.to_numpy() == 1, fr_.pavg.to_numpy(float), 1 - fr_.pavg.to_numpy(float))
    fv, ft = fr_.f.to_numpy(float), _obj(fr_.ft)
    ok_ = fv > 0
    near = lambda a: np.abs(a / np.where(form > 0, form, np.nan) - 1) <= 0.01
    co, sh = ok_ & (ft == 'COLLATERAL'), ok_ & (ft == 'SHARES')
    out = dict(n_fee_records=int(ok_.sum()), coll_n=int(co.sum()), coll_match=int((co & near(fv)).sum()), sh_n=int(sh.sum()),
               sh_match_usd=int((sh & near(fv * pown)).sum()), sh_match_amount=int((sh & near(fv)).sum()))
    qm = fr_.qm.to_numpy(float)
    for key, m_, x in (('coll', co, fv), ('sh_usd_at_fill', sh, fv * pown), ('sh_amount', sh, fv)):   # implied rate = fee / sum(min(p,1-p) q)
        r = np.round(x[m_ & (qm > 0)] / qm[m_ & (qm > 0)], 4)
        for v_, c_ in zip(*np.unique(r, return_counts=True)):
            out[f'hist|{key}|{v_:.4f}'] = int(c_)
    return out


def top_k(D, mode, k=TOP_K):
    """Deterministic top-k of a day table, ties by address ascending. mode 'shares': makers (sh_mk > 0) by maker shares; mode 'pnl':
    maker-majority addresses (sh_mk > sh_tk) with resolved maker shares (sh_res_mk > 0) by MAKER-leg net P&L to resolution (net_res_mk), so
    an address cannot rank on its taker legs. Each entry carries its maker share of volume."""
    col = 'sh_mk' if mode == 'shares' else 'net_res_mk'
    c = (D[D.sh_mk > 0] if mode == 'shares' else D[(D.sh_mk > D.sh_tk) & (D.sh_res_mk > 0)]).copy()
    c['addr'] = c.addr.astype(str)
    c = c.sort_values([col, 'addr'], ascending=[False, True], kind='mergesort')
    tie = bool(len(c) > k and c[col].iloc[k - 1] == c[col].iloc[k])
    return [dict(rank=i + 1, addr=a, value=float(x), sh_mk=float(sm), sh_tk=float(st_), mk_share=float(sm / (sm + st_)) if sm + st_ > 0 else None,
                 net_res_mk=float(nm), net_res=float(nr), n_res_na=int(na))
            for i, (a, x, sm, st_, nm, nr, na) in enumerate(zip(c.addr[:k], c[col][:k], c.sh_mk[:k], c.sh_tk[:k], c.net_res_mk[:k], c.net_res[:k],
                                                               c.n_res_na[:k]))], dict(n_candidates=int(len(c)), tie_at_cut=tie)


DSUM = MM_SUM + FIFO + ['rt_pnl_res'] + REW + ['n_excl', 'n_markets', 'n_markets_mk', 'n_mkt_res_na', 'n_mkt_end_na']


def day_partial(M):
    """maker_market rows of one chunk -> per-address partial day sums (FIFO residual parts over the markets where they exist, their
    missing markets counted; rows without an address stay in maker_market only)."""
    X = M[M.addr.notna()]
    P = X[MM_SUM + REW + ['n_excl']].copy()
    for c in FIFO:
        P[c] = X[c].fillna(0.)
    P['rt_pnl_res'] = np.where(_isin(X.res_status, RES_VALID), X.rt_pnl.to_numpy(float), 0.)   # round trips of resolved markets only
    P['n_markets'], P['n_markets_mk'] = 1, X.is_maker.astype(np.int64)
    P['n_mkt_res_na'], P['n_mkt_end_na'] = X.resid_res.isna().astype(np.int64), X.resid_drift.isna().astype(np.int64)
    P['addr'] = _obj(X.addr)
    return P[['addr'] + DSUM]


def day_table(Drun, d, population):
    """The day's running per-address sums -> maker_day rows (ranks among the day's makers; identity flags set by the caller)."""
    D = (Drun if Drun is not None else pd.DataFrame(columns=['addr'] + DSUM)).sort_values('addr', kind='mergesort').reset_index(drop=True)
    D['day'], D['population'], D['status'] = d, population, np.where(D.n_mk + D.n_tk > 0, OK, 'all_rows_excluded')
    for mode, rk in (('shares', 'rank_sh'), ('pnl', 'rank_pnl')):
        lst, _ = top_k(D, mode, k=len(D) + 1)
        D[rk] = D.addr.map({x['addr']: x['rank'] for x in lst}).fillna(-1).astype(np.int64)
    for k in ('in_exante_sh', 'in_exante_pnl', 'in_sameday_sh', 'in_sameday_pnl'):
        D[k] = False
    for k in INTS & set(D.columns):
        D[k] = D[k].astype(np.int64)
    return D


def day_segments(P, D, sets, d, ci):
    """Per-address partials of one day -> segment rows per identity set x group (in / field / nonmaker) x role x dim x bucket; checks the
    zero-sum closure of every (set, dim): the groups and roles sum to zero."""
    P = P.groupby(['addr', 'role', 'dim', 'bucket'], sort=True).sum().reset_index()
    makers = set(D.addr[D.sh_mk > 0])
    rows, devs = [], {}
    for name, (label, members) in sets.items():
        a = pd.Series(P.addr)
        grp = np.where(a.isin(set(members)), 'in', np.where(a.isin(makers), 'field', 'nonmaker'))
        g = P.assign(grp=grp).groupby(['grp', 'role', 'dim', 'bucket'], sort=True)
        S = g[MET].sum()
        S['n_addr'] = g.addr.nunique()
        S = S.reset_index()
        S['day'], S['id_set'], S['label'], S['ci'], S['status'] = d, name, label, ci, OK
        Z = S.groupby('dim')[['notional', 'gross_res'] + [f'v_{h}' for h in VALS]].sum()
        zt = TOL_ABS + TOL_REL * Z.notional.to_numpy()
        for col in ['gross_res'] + [f'v_{h}' for h in VALS]:
            dv = np.abs(Z[col].to_numpy())
            devs[col] = max(devs.get(col, 0.), float(dv.max()) if len(dv) else 0.)
            need(bool(np.all(dv <= zt)), f'segment zero-sum: {name} {col} does not close over the groups')
        rows.append(S)
    S = pd.concat(rows, ignore_index=True)
    for k in INTS & set(S.columns):
        S[k] = S[k].astype(np.int64)
    return S[list(SEG)], devs


def compare_rows(S):
    """identities.json point estimates: dim 'all' segment rows with per-share rates (cents / share over the shares each value covers)."""
    X = S[S.dim == 'all']
    out = []
    for r in X.itertuples(index=False):
        e = dict(day=r.day, id_set=r.id_set, label=r.label, grp=r.grp, role=r.role, n_addr=int(r.n_addr), n=int(r.n), sh=float(r.sh),
                 gross_res=float(r.gross_res), fee_res=float(r.fee_res), net_res=float(r.net_res), fee_fill=float(r.fee_fill), ci=r.ci)
        e['c_per_sh_res'] = 100. * r.gross_res / r.sh_res if r.sh_res > 0 else None
        e['c_per_sh_net'] = 100. * r.net_res / r.sh_res if r.sh_res > 0 else None
        for h in VALS:
            sh_ = getattr(r, f'sh_{h}')
            e[f'c_per_sh_{h}'] = 100. * getattr(r, f'v_{h}') / sh_ if sh_ > 0 else None
        out.append(e)
    return out


def _arrow_schema(spec):
    return pa.schema([(c, ARROW[t]) for c, t in spec.items()])


def _to_arrow(df, spec):
    need(list(df.columns) == list(spec), f'output columns differ from the declared schema: {sorted(set(df.columns) ^ set(spec))}')
    cols = {}
    for c, t in spec.items():
        x = df[c]
        if t == 'str':
            cols[c] = pa.array(_obj(x.astype(object)), type=pa.string(), from_pandas=True)
        else:
            need(not (t in ('int64', 'bool') and x.isna().any()), f'output column {c} ({t}) has missing values')
            cols[c] = pa.array(np.asarray(x, dtype={'int64': np.int64, 'bool': bool, 'float64': float}[t]), type=ARROW[t])
    return pa.table(cols, schema=_arrow_schema(spec))


def put(df, spec, path):
    need(not Path(path).exists(), f'output exists (write-once): {Path(path).name}')
    pq.write_table(_to_arrow(df, spec), path)


def run_day(d, inp, step1, part, book, sets_ex, n_meas, guard_mib, chunk, log=print):
    """One day (the ID day or a measurement day), meant to run in its own process so its VmHWM is one day's (contract: process per
    hour / market). Writes part/{fills_valued, maker_market, maker_day[, segments, rewards_market]}.parquet and part/day.json; refuses
    on any check."""
    need(d in C.ALLOWED_DAYS, f'day {d!r} is not a consumed day (refused before any path is built)')
    inp, step1, part = Path(inp), Path(step1), Path(part)
    part.mkdir(parents=True)
    population = 'ID' if d == C.ID_DAY else 'FILL'
    need(population == 'ID' or sets_ex is not None, f'{d}: the ex-ante identity sets are needed before a measurement day')
    ci = f'none_G{n_meas}_lt_{G_MIN}' if n_meas < G_MIN else f'eligible_G{n_meas}'
    bm = [int(m) for m in book.get(d, [])]
    hashes, counts, checks, diags, mem, ident, rew_inv = {}, {}, {}, {}, {}, {}, None
    W = pq.ParquetWriter(part / 'fills_valued.parquet', _arrow_schema(FV))
    MW = pq.ParquetWriter(part / 'maker_market.parquet', _arrow_schema(MM))
    L = load(inp / f'legs_{d}.parquet', 'legs', LEG_COLS, LEG_CATS, hashes=hashes)
    R = load(inp / f'records_{d}.parquet', 'records', REC_COLS, REC_CATS, hashes=hashes)
    need(bool((_obj(L.day) == d).all()) and bool((_obj(R.day) == d).all()), f'{d}: a legs / records day column differs from the file day')
    need(not L.duplicated(['sid', 'mk_idx']).any(), f'{d}: duplicate leg key (sid, mk_idx) in legs')
    need(R.sid.is_unique, f'{d}: records sid (settlementId) is not unique')
    Fr, lf, orders, mk = None, None, None, None
    mp = inp / f'markets_{d}.parquet'
    if population == 'FILL' and mp.is_file():
        mk = load(mp, 'markets', ['mid'], schema=MKT_SCHEMA, hashes=hashes, optional=('status',) + tuple(c for nm in MKT_FIELDS.values() for c in nm))
    # boundary-censored markets (any legs / records / markets status flag): excluded whole, outcome never read
    cens = set(L.mid[flagged(L.status, CENSOR)].tolist()) | set(R.mid[flagged(R.status, CENSOR)].tolist())
    if mk is not None and 'status' in mk.columns:
        cens |= set(mk.mid[flagged(mk.status, CENSOR)].tolist())
    cens = {int(m) for m in cens}
    reg = registry(step1 / d / 'registry.parquet', hashes, censored=cens)
    addrs = set(_obj(L.mk_signer.cat.categories)) | set(_obj(L.tk_signer.cat.categories)) | set(_obj(R.tk_signer.dropna().unique()))
    need(len({str(a).lower() for a in addrs}) == len(addrs), f'{d}: address case collision (two spellings of one address)')
    if population == 'FILL':
        Fr = frames(load(inp / f'tob_{d}.parquet', 'tob', TOB_COLS, ('status',), hashes=hashes))
        diags['frames'] = dict(n=int(len(Fr['t'])), own_status=Fr['counts'], tob_status=Fr['tob_status'])
        lfs, ods = [], []
        for m_ in bm:
            lfs.append(load(inp / 'life' / f'{d}_{m_}_fills.parquet', 'fills', LF_COLS, hashes=hashes))
            ods.append(load(inp / 'life' / f'{d}_{m_}_orders.parquet', 'orders', ORD_COLS, hashes=hashes))
            need(bool((lfs[-1].mid == m_).all()) and bool((ods[-1].mid == m_).all()), f'{d} {m_}: a lifecycle table holds another market')
        lf = pd.concat(lfs, ignore_index=True) if lfs else None
        orders = pd.concat(ods, ignore_index=True) if ods else None
        if orders is not None:
            diags['lifecycle'] = dict(orders_status={str(k): int(v) for k, v in orders.status.value_counts().items()},
                                      fills_status={str(k): int(v) for k, v in lf.status.value_counts().items()})
        rew_inv = rewards_inventory(mk, reg) if mk is not None else 'markets_table_missing'
    dfr, dlv, depth_d = {}, {}, {}
    for m_ in (bm if population == 'FILL' else []):                     # BOOK depth (spread-clock diagnostic, rewards upper bound)
        dfr[m_] = load(inp / 'depth' / f'{d}_{m_}_frames.parquet', 'frames', ['mid'] + DFR_COLS, ('status',), hashes=hashes)
        dlv[m_] = load(inp / 'depth' / f'{d}_{m_}_levels.parquet', 'levels', ['mid'] + DLV_COLS, ('side',), hashes=hashes)
        need(bool((dfr[m_].mid == m_).all()) and bool((dlv[m_].mid == m_).all()), f'{d} {m_}: a depth table holds another market')
        o_ = np.lexsort((dfr[m_].recv_ns.to_numpy(np.int64), dfr[m_].upd_ms.to_numpy(np.int64)))
        depth_d[m_] = (dfr[m_].upd_ms.to_numpy(np.int64)[o_], {int(x): i for i, x in enumerate(dfr[m_].fid.to_numpy(np.int64)[o_])})
    bcol = next((c for c in MKT_FIELDS['boosted'] if mk is not None and c in mk.columns), None)
    boost_map = {} if bcol is None else {int(m): (None if pd.isna(b) else bool(b)) for m, b in mk.drop_duplicates('mid')[['mid', bcol]].itertuples(index=False)}
    top_set = frozenset() if population == 'ID' else frozenset(sets_ex['exante_shares'][1]) | frozenset(sets_ex['exante_pnl'][1])
    rew_by_mkt = {}
    mids = np.unique(L.mid.to_numpy(np.int64))
    day_addrs = pd.Index(sorted(set(_obj(L.mk_signer.cat.categories)) | set(_obj(L.tk_signer.cat.categories))))
    Drun, Prun, fcnt, n_fv, n_mm = None, None, {}, 0, 0
    dd = dict(lf_rows=0, lf_matched=0, lf_hi_mismatch=0, lf_status_not_ok=0, records=0, tk_px_gap_n=0, tk_px_gap_max=0., tk_px_gap_usd=0.,
              legs_recv_before_settle=0)
    CLK = ('clock_book_records', 'clock_with_evidence', 'clock_hi_shows_trade', 'clock_evidence_frame_missing', 'clock_no_evidence')
    clk = dict({k: 0 for k in CLK}, rows_both_ok=0, sh=0., v_spr_lo=0., v_spr_hi=0.)
    tr_ = L.t_rel_s.to_numpy(float)
    dd.update(pre_legs=int((tr_ < 0).sum()), pre_legs_before_300s=int((tr_ < -300).sum()), t_rel_min_s=float(np.nanmin(tr_)) if len(tr_) else None,
              censored_markets=sorted(cens), censored_legs=int(L.mid.isin(cens).sum()), registry_markets=int(len(reg)), markets_with_legs=int(len(mids)),
              legs=int(len(L)), records_in=int(len(R)), registry_markets_without_legs=int((~reg.mid.isin(set(mids.tolist()))).sum()),
              legs_market_not_in_registry=int((~L.mid.isin(set(reg.mid.tolist()))).sum()), records_without_legs=int((~R.sid.isin(L.sid.unique())).sum()))
    inv_d = {r['mid']: r for r in rew_inv} if isinstance(rew_inv, list) else {}
    rr = reg.set_index('mid')
    n_rew = 0
    for k0 in range(0, len(mids), chunk):
        cm = mids[k0:k0 + chunk]
        bmc = [m for m in bm if m in set(cm.tolist())]
        Lc = L[L.mid.isin(cm)].reset_index(drop=True)
        Rc = R[R.mid.isin(cm) | R.sid.isin(set(_obj(Lc.sid)))].reset_index(drop=True)
        lfc = lf[lf.mid.isin(cm)] if lf is not None else None
        oc = orders[orders.mid.isin(cm)] if orders is not None else None
        for c_ in ('sid', 'mk_signer', 'tk_signer', 'mk_hash', 'tk_hash'):          # sid / hashes become categorical per chunk only
            Lc[c_] = Lc[c_].cat.remove_unused_categories() if isinstance(Lc[c_].dtype, pd.CategoricalDtype) else Lc[c_].astype('category')
        F, end_mkt, dg = build_rows(d, population, Lc, Rc, reg, Fr, lfc, bmc, depth=depth_d, top_set=top_set, boost=boost_map)
        for k in ('lf_rows', 'lf_matched', 'lf_hi_mismatch', 'lf_status_not_ok', 'records', 'tk_px_gap_n', 'tk_px_gap_usd', 'legs_recv_before_settle'):
            dd[k] += dg[k]
        for k in CLK:
            clk[k] += dg.get(k, 0)
        if bmc:                                    # spread at lo vs at settle_ms, same BOOK maker fills, both statuses ok
            mb_ = ((F.role == 'maker').to_numpy() & _isin(F.mid.to_numpy(), bmc) & (F.status == OK).to_numpy() & (F.st_spr == OK).to_numpy()
                   & (F.st_spr_hi == OK).to_numpy())
            clk['rows_both_ok'] += int(mb_.sum())
            clk['sh'] += float(F.q.to_numpy()[mb_].sum())
            clk['v_spr_lo'] += float(F.v_spr.to_numpy()[mb_].sum())
            clk['v_spr_hi'] += float(F.v_spr_hi.to_numpy()[mb_].sum())
        dd['tk_px_gap_max'] = max(dd['tk_px_gap_max'], dg['tk_px_gap_max'])
        for k, v_ in fee_diag(dg['_fee'], reg).items():
            fcnt[k] = fcnt.get(k, 0) + v_
        fifo(F)
        life_join(F, Lc, oc, bmc)
        M = market_table(F, d, population, bm, end_mkt, reg)
        for k, v_ in check_market(M).items():
            checks[k] = max(checks.get(k, 0.), v_)
        for m_ in bmc:                                                                # REWARDS lower bound on the chunk's BOOK markets
            inv = inv_d.get(int(m_), {})
            thr_sh = inv.get('share_thr', rr.share_thr.get(m_) if 'share_thr' in rr else None)
            thr_sp = inv.get('spread_thr', rr.spread_thr.get(m_) if 'spread_thr' in rr else None)
            w_lo, w_hi = inv.get('sched_lo_ms') or int(rr.ws_ms[m_]), inv.get('sched_hi_ms') or int(rr.we_ms[m_])
            if thr_sh is None or thr_sp is None or pd.isna(thr_sh) or pd.isna(thr_sp):
                M.loc[M.mid == m_, 'rew_status'] = 'thresholds_missing'
                continue
            mkr = F[(F.mid == m_) & (F.role == 'maker')]
            pres = presence(oc[oc.mid == m_], lfc[lfc.mid == m_], Fr, int(m_), dict(zip(_obj(mkr.hash), mkr.p.to_numpy())), float(thr_sh), float(thr_sp),
                            int(w_lo), int(w_hi))
            sel = (M.mid == m_).to_numpy()
            M.loc[sel, 'rew_status'] = OK
            rew_by_mkt[int(m_)] = {}
            for a_, D_ in pres.items():
                ub, ua = union(D_['bid']), union(D_['ask'])
                j = sel & (M.addr.to_numpy() == a_)
                M.loc[j, 'rew_bid_ms'], M.loc[j, 'rew_ask_ms'], M.loc[j, 'rew_both_ms'], M.loc[j, 'rew_n_orders'] = measure(ub), measure(ua), overlap(ub, ua), D_['n']
                rew_by_mkt[int(m_)][a_] = (measure(ub), measure(ua), overlap(ub, ua))
                n_rew += 1
        MW.write_table(_to_arrow(M[list(MM)], MM))
        n_mm += len(M)
        Drun = fold(Drun, day_partial(M), ['addr'])
        c_, v_ = contrib(F)
        Prun = fold(Prun, seg_partial(F, c_, v_, day_codes(F.addr, day_addrs)), SKEY)
        for col in ('status', 't_status', 'st_res', 'st_spr', 'st_end', 'life_status', 'kind', 'src_status') + tuple(f'st_{h}' for h, _ in HORIZONS):
            cc = counts.setdefault(col, {})
            for k, v2 in F[col].value_counts(dropna=False).items():
                if v2:
                    cc[str(k)] = cc.get(str(k), 0) + int(v2)
        W.write_table(_to_arrow(F[list(FV)], FV))
        n_fv += len(F)
        del F, Lc, Rc, c_, v_, M
        trim()
        mem[f'{d}/{k0}'] = guard(guard_mib, f'{d} chunk {k0 // chunk}')
    W.close()
    MW.close()
    D = day_table(Drun, d, population)
    dd['fee_formula'] = dict(rule='formula = feeRateBps/1e4 * sum(min(p, 1-p) * q) over the record legs; match = within 1 %',
                             n_fee_records=fcnt.get('n_fee_records', 0), collateral=dict(n=fcnt.get('coll_n', 0), n_match_usd=fcnt.get('coll_match', 0)),
                             shares=dict(n=fcnt.get('sh_n', 0), n_match_usd_at_fill_price=fcnt.get('sh_match_usd', 0),
                                         n_match_amount_as_formula=fcnt.get('sh_match_amount', 0)),
                             implied_rate_top={key: dict(sorted(((k.split('|')[2], v_) for k, v_ in fcnt.items() if k.startswith(f'hist|{key}|')),
                                                                key=lambda kv: (-kv[1], kv[0]))[:10]) for key in ('coll', 'sh_usd_at_fill', 'sh_amount')})
    l1, i1 = top_k(D, 'shares')
    l2, i2 = top_k(D, 'pnl')
    n_seg, n_rm = 0, 0
    if population == 'ID':
        ident = dict(day=d, top10_shares=l1, top10_shares_info=i1, top10_pnl=l2, top10_pnl_info=i2,
                     rule='top 10 by maker shares (mk_share = maker share of volume) / by MAKER-leg net P&L to resolution among maker-majority '
                          'addresses (sh_mk > sh_tk) with resolved maker shares; ties by address ascending')
        sets = {k: tuple(v) for k, v in dict(exante_shares=('ex_ante', [x['addr'] for x in l1]), exante_pnl=('ex_ante', [x['addr'] for x in l2])).items()}
    else:
        ident = dict(label='descriptive', top10_shares=l1, top10_shares_info=i1, top10_pnl=l2, top10_pnl_info=i2)
        sets = dict({k: tuple(v) for k, v in sets_ex.items()}, sameday_shares=('descriptive', [x['addr'] for x in l1]),
                    sameday_pnl=('descriptive', [x['addr'] for x in l2]))
    for k, (_, mem_) in sets.items():
        D[{'exante_shares': 'in_exante_sh', 'exante_pnl': 'in_exante_pnl', 'sameday_shares': 'in_sameday_sh', 'sameday_pnl': 'in_sameday_pnl'}[k]] = D.addr.isin(set(mem_))
    if population == 'FILL':
        P = Prun if Prun is not None else pd.DataFrame(columns=SKEY + MET)
        P = P.assign(addr=_obj(day_addrs)[P.a.to_numpy(np.int64)], role=np.array(['maker', 'taker'], dtype=object)[P.role.to_numpy(np.int64)],
                     dim=np.array(SEG_DIMS, dtype=object)[P.dim.to_numpy(np.int64)],
                     bucket=np.array(BUCKETS, dtype=object)[P.bucket.to_numpy(np.int64)]).drop(columns=['a'])
        S, zd = day_segments(P, D, sets, d, ci)
        for k, v_ in zd.items():
            checks[f'segment_zero_sum_{k}'] = v_
        put(S, SEG, part / 'segments.parquet')
        n_seg = len(S)
        dd['rewards_presence_rows'] = n_rew
        dd['spread_clock'] = dict(clk, share_hi_shows_trade=(clk['clock_hi_shows_trade'] / clk['clock_with_evidence']) if clk['clock_with_evidence'] else None,
                                  rule='BOOK records (status ok, clean fills row, f_after known): the last depth frame <= settle_ms is at or after the '
                                       'evidence frame f_after (record = earliest f_after of its legs); v_spr_lo / v_spr_hi over the same maker fills')
        RMr = rewards_rows(d, reg, rew_inv, bm, cens, dfr, dlv, L, rew_by_mkt, sets)
        put(RMr, RM, part / 'rewards_market.parquet')
        n_rm = len(RMr)
    put(D[list(MD)], MD, part / 'maker_day.parquet')
    log(f'[econ] {d} {population} legs {len(L)} records {len(R)} markets {len(mids)} maker_market {n_mm} addresses {len(D)} hwm {hwm():.0f} MiB',
        flush=True)
    mem[d] = guard(guard_mib, f'{d} end')
    J = dict(day=d, population=population, hashes=hashes, counts=counts, checks=checks, diagnostics=dict(dd, **diags), rewards_inventory=rew_inv,
             ident=ident, sets=sets if population == 'ID' else None,
             rows=dict(fills_valued=n_fv, maker_market=n_mm, maker_day=len(D), segments=n_seg, rewards_market=n_rm),
             memory=dict(hwm_mib=hwm(), by_stage=mem))
    (part / 'day.json').write_text(json.dumps(J, indent=1, sort_keys=True, default=str) + '\n')


def _stream(parts, dest, spec):
    """Concatenate parquet parts (declared schema) row group by row group into one file (bounded memory) -> rows."""
    w, n = pq.ParquetWriter(dest, _arrow_schema(spec)), 0
    try:
        for p_ in parts:
            pf = pq.ParquetFile(p_)
            need(pf.schema_arrow.equals(_arrow_schema(spec)), f'part {p_.parent.name}/{p_.name}: schema differs from the declared one')
            for i in range(pf.num_row_groups):
                t_ = pf.read_row_group(i)
                w.write_table(t_)
                n += t_.num_rows
    finally:
        w.close()
    return n


def run(inp=None, out=None, step1=None, book=None, days=None, guard_mib=C.GUARD_MIB, chunk=CHUNK, log=print):
    """The econ build (RULES): one worker process per day (ID day first: its identity sets feed the measurement days), then a
    bounded-memory merge into OUT/econ (write-once; built in a partial directory renamed at the end). Refuses before any filesystem
    access on a day outside the consumed days; a worker's refusal is re-raised with its reason."""
    asof0, t0 = _now(), time.time()
    days = tuple(days) if days is not None else (C.ID_DAY,) + C.MEASURE_DAYS
    for d in days:
        need(d in C.ALLOWED_DAYS, f'day {d!r} is not a consumed day (refused before any path is built)')
    need(len(days) >= 2 and days[0] == C.ID_DAY and all(d in C.MEASURE_DAYS for d in days[1:]) and len(set(days)) == len(days),
         'days must be the ID day followed by distinct measurement days')
    inp = Path(inp) if inp is not None else C.OUT
    step1 = Path(step1) if step1 is not None else C.STEP1
    out = Path(out) if out is not None else inp / 'econ'
    need(not out.exists(), f'the econ output exists (write-once): {out.name}')
    book_src = 'argument'
    if book is None:
        book, book_src = load_book(), str(BOOK_MANIFEST.relative_to(C.REPO))
    for d in book:
        need(d in C.MEASURE_DAYS, f'BOOK day {d!r} is not a measurement day')
    book = {d: sorted(int(m) for m in v) for d, v in book.items()}
    n_meas = len(days) - 1
    tmp = out.parent / f'.{out.name}.partial_{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}_{os.getpid()}'
    need(not tmp.exists(), 'the partial output directory exists')
    (tmp / 'parts').mkdir(parents=True)
    try:
        Js, sets_ex = {}, None
        for d in days:
            a_ = tmp / 'parts' / f'{d}.args.json'
            a_.write_text(json.dumps(dict(d=d, inp=str(inp), step1=str(step1), part=str(tmp / 'parts' / d), book=book, sets_ex=sets_ex, n_meas=n_meas,
                                          guard_mib=guard_mib, chunk=chunk)))
            p = subprocess.run([sys.executable, str(HERE / 'pf_mt_econ.py'), 'day', str(a_)], capture_output=True, text=True)
            for line in p.stdout.splitlines():
                log(line, flush=True)
            if p.returncode != 0:
                err = [x for x in p.stderr.strip().splitlines() if x.strip()]
                why = next((x for x in reversed(err) if 'Error' in x), err[-1] if err else f'rc {p.returncode}')
                need(False, f'{d}: day worker refused: {why}')
            Js[d] = json.loads((tmp / 'parts' / d / 'day.json').read_text())
            if d == C.ID_DAY:
                sets_ex = Js[d]['sets']
        n_fv = _stream([tmp / 'parts' / d / 'fills_valued.parquet' for d in days], tmp / 'fills_valued.parquet', FV)
        n_mm = _stream([tmp / 'parts' / d / 'maker_market.parquet' for d in days], tmp / 'maker_market.parquet', MM)
        MD_ = pd.concat([pd.read_parquet(tmp / 'parts' / d / 'maker_day.parquet') for d in days], ignore_index=True)
        ad = set(MD_.addr.dropna().tolist())
        need(len({a.lower() for a in ad}) == len(ad), 'address case collision across days (two spellings of one address)')
        put(MD_[list(MD)], MD, tmp / 'maker_day.parquet')
        segp = [tmp / 'parts' / d / 'segments.parquet' for d in days[1:]]
        SEGa = pd.concat([pd.read_parquet(p_) for p_ in segp], ignore_index=True) if segp else pd.DataFrame(columns=list(SEG))
        put(SEGa[list(SEG)], SEG, tmp / 'segments.parquet')
        rmp = [tmp / 'parts' / d / 'rewards_market.parquet' for d in days[1:]]
        RMa = pd.concat([pd.read_parquet(p_) for p_ in rmp], ignore_index=True) if rmp else pd.DataFrame(columns=list(RM))
        put(RMa[list(RM)], RM, tmp / 'rewards_market.parquet')
        ci = f'none_G{n_meas}_lt_{G_MIN}' if n_meas < G_MIN else f'eligible_G{n_meas}'
        ident = dict(ex_ante=Js[C.ID_DAY]['ident'], same_day={d: Js[d]['ident'] for d in days[1:]}, n_days=n_meas, g_min=G_MIN,
                     intervals=bool(n_meas >= G_MIN), ci=ci, compare=compare_rows(SEGa),
                     overlap={'exante_shares&exante_pnl': len(set(sets_ex['exante_shares'][1]) & set(sets_ex['exante_pnl'][1]))},
                     groups='in = the set; field = the day\'s other makers; nonmaker = addresses without a maker leg that day (closes the zero sum)')
        (tmp / 'identities.json').write_text(json.dumps(ident, indent=1, sort_keys=True, default=str) + '\n')
        checks = {}
        for J in Js.values():
            for k, v_ in J['checks'].items():
                checks[k] = max(checks.get(k, 0.), v_)
        rec = dict(version=C.VERSION, module='pf_mt_econ', asof_start=asof0, asof_end=_now(), wall_s=round(time.time() - t0, 1), days=list(days),
                   book=book, book_source=book_src, inputs={k: v for J in Js.values() for k, v in J['hashes'].items()},
                   sources={n_: sha256(HERE / n_) for n_ in ('pf_mt_econ.py', 'pf_mt_common.py', 'pf_mt_econ_selftest.py') if (HERE / n_).is_file()},
                   rows=dict(fills_valued=n_fv, maker_market=n_mm, maker_day=len(MD_), segments=len(SEGa), rewards_market=len(RMa)),
                   counts={d: J['counts'] for d, J in Js.items()},
                   checks={k: dict(max_dev=v_, tol=f'{TOL_ABS} + {TOL_REL} x notional', ok=True) for k, v_ in checks.items()},
                   diagnostics={d: J['diagnostics'] for d, J in Js.items()},
                   rewards=dict(inventory={d: J['rewards_inventory'] for d, J in Js.items() if J['population'] == 'FILL'},
                                presence_rule='REWARDS (lower bound of demonstrated eligible presence)',
                                per_maker='unidentifiable: rewards are not observable per maker; rewards_market gives the pool (units unverified), '
                                          'the depth upper bound per side and each identity set\'s proven lower bound only'),
                   rules=dict(stale_ms=STALE_MS, end_guard_ms=END_GUARD_MS, horizons_ms=dict(HORIZONS), chunk=chunk, top_k=TOP_K, g_min=G_MIN,
                              tolerance=dict(abs=TOL_ABS, rel_notional=TOL_REL), **RULES),
                   memory=dict(guard_mib=guard_mib, process='one worker per day', worker_hwm_mib={d: J['memory']['hwm_mib'] for d, J in Js.items()},
                               by_stage={k: v for J in Js.values() for k, v in J['memory']['by_stage'].items()}, parent_hwm_mib=hwm()),
                   slot=os.environ.get('PF_SLOT'), argv=sys.argv)
        (tmp / 'econ.json').write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + '\n')
        guard(guard_mib, 'merge')
        shutil.rmtree(tmp / 'parts')
        need(not out.exists(), f'the econ output appeared during the run (write-once): {out.name}')
        os.rename(tmp, out)
        return rec
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['day']:                     # worker: one day in its own process (spawned by run)
        A = json.loads(Path(a[1]).read_text())
        run_day(A['d'], A['inp'], A['step1'], A['part'], A['book'], A['sets_ex'], A['n_meas'], A['guard_mib'], A['chunk'])
    else:
        need(a[:1] == ['run'], 'usage: pf_mt_econ.py run [--chunk N]')
        ch = int(a[a.index('--chunk') + 1]) if '--chunk' in a else CHUNK
        r = run(chunk=ch)
        print(json.dumps(dict(rows=r['rows'], checks={k: v['max_dev'] for k, v in r['checks'].items()}, memory=r['memory']['worker_hwm_mib']), indent=1))
