"""COMPLETED CLOSES (P-2026-005 MAKER-TRACE; decision_contract.md Amendment 2, '1. Completed closes' = review 567f741d9's next priority item 2;
session aea5bfb9 development, 2026-11-23). Computed question, per address and day: is completed turnover good and then lost through residual
exposure, or is the completed turnover itself poor? DESCRIPTIVE ONLY, on consumed days (Oct 24-27: four point estimates, no intervals, G = 4 < 5);
no strategy, imitation, tradeable or profitability claim; it measures no model skill. Every rule below is declared before any data run; synthetic
proof (hand-computed closes, reconciliation and write-once refusals): closes_selftest.py.
POPULATION econ/identities.json ex_ante.top10_shares: ranks 1-3 = the contract's top 3 (else refuse), all 10 = the benchmark. Their econ/fills_valued
       rows (maker AND taker legs) on pf_mt_common.MEASURE_DAYS, one day at a time: row groups chosen by the footer statistics of `day` (a row
       group spanning days is filtered on its day column; one whose range leaves the consumed days refuses before it is read); other addresses
       never leave their row group. Completeness: per address x market x day the ok and excluded row counts equal econ/maker_market (n_mk + n_tk,
       n_excl), so no row is lost by the row-group choice.
LOTS   status ok rows with self-trades out (as pf_mt_econ.fifo; a self-trade keeps its gross, fees and counts), per address x market x day.
ORDER  econ's FIFO order (settle_ms, sid lexicographic, px_rank, mk_idx, maker before taker) and its per-side cumulative shares (the same float
       sums); unit j bought pairs unit j sold (= chronological FIFO across sign flips). A completed close = one interval of the merged cumulative
       boundaries inside [0, X], X = min(bought, sold): its buy leg, sell leg and shares. ENTRY = the earlier leg in the order, CLOSE = the later.
       Holding time = settle_ms(close) - settle_ms(entry): the FIFO clock (UUIDv7 settlement-ID creation, an upper bound of each match; t_ms is
       never used). A leg = (day, sid, mk_idx, role) with its hash, side, settle_ms and YES price.
VALUES YES space. P&L = shares x (sell price - buy price) (V-free). Fees of both legs prorated by shares (leg fee x shares / leg q), at the fill
       price (fee_fill: PRIMARY, V-free) and at resolution (fee_res: sensitivity): net = P&L - fee_fill(entry + close), net_feeres = P&L -
       fee_res(entry + close).
DUST   a close below DUST_SH shares (the rounding of two cumulative sums that are equal in exact arithmetic) stays in every sum, flagged dust; it is
       never counted as a close and never enters a holding-time quantile.
RESID  per lot row rq = q - its attributed shares. Per address x market: residual shares (sum rq); value to END sum s rq (m_end - p) (the market's
       END mid; none without one), to resolution sum s rq (V - p), END -> resolution (V - m_end) x net position; their prorated fees; max
       |position| along the chronological path of the lots (recorded fills only; self-trades out).
RECONCILE per address x market x day against econ/maker_market; any failure refuses (REL_TOL relative with a 1-unit floor unless stated): rows
       (above); every matched share attributed once (per lot row attributed = econ mq and q - attributed = econ rq within SH_TOL_ABS + SH_TOL_REL x
       the group's traded shares; self-trade rows mq = rq = 0); close P&L = rt_pnl and close shares = rt_sh; close P&L + residual value to
       resolution + self-trade gross = econ gross (gross_res_mk + gross_res_tk, resolved markets) = the fills_valued sum; residual value = resid_res,
       to END = resid_drift; net position = net_pos; fees = fee_fill / fee_res; econ net_res = close net + residual value - residual fee_fill -
       self fee_fill + (fee_fill - fee_res over all rows) + self gross; prorated fees sum to the rows' fees; close shares = X; residual shares =
       |net position|.
SUMMARY closes_summary: scope (each of the 10 addresses, top 3, top 10) x day (+ all) x role pair (maker->maker, maker->taker, taker->maker,
       taker->taker, + all), closes in resolved markets (the others counted): count, shares, gross / net / net_feeres per share, share-weighted
       holding-time quantiles (the smallest holding time whose cumulative share weight reaches q; no interpolation) and the count median (lower).
       residual_summary: scope x day over the resolved markets (the others counted): turnover beside residual drift to END, END -> resolution,
       residual value, fee-token drift, self-trades, econ net, the identity deviation and the recorded inventory excursion (max |position|).
COHORTS identical cohorts: lot rows with ok 1s, 10s, 60s and END marks AND a resolution (ok / tie), per scope (rank 1, 2, 3, top 3, top 10) x role
       x depth (econ's dpt: ticks from the pre-trade best on the maker leg's side; a taker row carries the depth of the maker leg it took; without
       dpt in the file, FALLBACK_DEPTH from 100 s_maker (m_minus - p) cents, st_spr ok) x day (+ all). Gross at 1s / 10s / 60s / END = v_h, at
       resolution gross_res; net = gross - fee_fill at the mark horizons and gross - fee_res at resolution. Per share on the cohort's own shares AND
       on the common denominator = every resolved lot share of that scope and day (all roles and depths); beside them the cell's own resolved
       shares and their resolution value (the cohort's coverage).
ANSWER per scope (the 10 addresses, top 3, top 10) and day (+ all), resolved markets: reading = no_rows | no_resolved_markets | no_closes |
       turnover_itself_poor (close net <= EPS_USD) | good_turnover_lost_to_residual (close net > EPS_USD and residual value to resolution <
       -EPS_USD) | good_turnover_residual_not_negative, with n; the same reading with fees at resolution (close net_feeres; residual value - its
       fee_res) is the sensitivity, fee_convention_sensitive when they differ; tallies over the address-days of the top 3 and of ranks 4-10.
OUTPUTS <out> = pf_mt_common.RECEIPT / 'closes' (write-once; built in a partial directory renamed at the end): closes.parquet (one row per
       completed close), closes_markets.parquet (address x market x day accounting with the econ values), closes_summary.csv, residual_summary.csv,
       cohorts.csv, answer.json, closes_receipt.json (inputs, sources, git, row groups, counts, checks, memory, CPU).
GUARDS days: measurement days only (refused before any path is built); the output must not exist (checked before any input); VmHWM self-guard
       pf_mt_common.GUARD_MIB (never raised); CPU self-guard CPU_CAP_S; the CLI refuses unless MT_RUN=v2 (amendment 2: the v2 roots).
Usage (run_light): MT_RUN=v2 closes.py run          selftest: closes_selftest.py"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
sys.path.insert(0, str(RESEARCH))
import pf_mt_common as C  # noqa: E402
import pf_mt_econ as E  # noqa: E402

need, OK = C.need, C.STATUS_OK
VERSION = 'closes_v1'
CONTRACT = 'predictfun/reviews/maker_trace_20261122/decision_contract.md (Amendment 2, 1. Completed closes)'
REVIEW = 'predictfun/reviews/automatic/20261123_maker_trace_review/report.md (567f741d9), next priority item 2'

# ------------------------------------------------------------------ declared constants (none fitted) ------------------------------------------------------------------
TOP3 = ('0x6Da6Cb464F92AE7aD4Ec3d239c81719Cb1D0Ae03', '0x7d32c568a9bEA1714036972601c51F4902C67091',
        '0x57FbC891E89184DA071AE9D97E745531cd7F718a')            # decision_contract.md population table, ranks 1-3
NTOP = 10
ROLES = ('maker', 'taker')
PAIRS = ('maker->maker', 'maker->taker', 'taker->maker', 'taker->taker')   # index 2 x entry role + close role
MARKS = ('1s', '10s', '60s', 'end')
HZ = MARKS + ('res',)
RES_VALID = tuple(E.RES_VALID)                                  # ('ok', 'tie'): V = 1 / 0, or 0.5 on a tie
DEPTH = tuple(E.DEPTH)
FALLBACK_DEPTH = ('mm_inside', 'mm_0_1c', 'mm_1_2c', 'mm_2_3c', 'mm_3c_plus', 'unknown')
DUST_SH = 1e-6
REL_TOL = 1e-6
SH_TOL_ABS, SH_TOL_REL = 1e-9, 1e-12
EPS_USD = 1e-9
QUANTS = (0.10, 0.25, 0.50, 0.75, 0.90)
CPU_CAP_S = 150.
KEY_MUL = 16                                                     # group key = mid x 16 + address index (< 10)
SCOPE_MEMBERS = dict({f'rank{i + 1}': (i,) for i in range(NTOP)}, top3=(0, 1, 2), top10=tuple(range(NTOP)))
COHORT_SCOPES = ('rank1', 'rank2', 'rank3', 'top3', 'top10')
READINGS = ('no_rows', 'no_resolved_markets', 'no_closes', 'turnover_itself_poor', 'good_turnover_lost_to_residual',
            'good_turnover_residual_not_negative')

# ---- inputs: the columns read (dtypes checked against pf_mt_econ.FV / MM in the file footers) ----
FV_NEED = ('day', 'mid', 'sid', 'mk_idx', 'px_rank', 'role', 'addr', 'hash', 's', 'q', 'p', 'self_trade', 'settle_ms', 'V', 'st_res', 'st_1s', 'v_1s',
           'st_10s', 'v_10s', 'st_60s', 'v_60s', 'm_end', 'st_end', 'v_end', 'fee_fill', 'fee_res', 'gross_res', 'mq', 'rq', 'status')
FV_FALLBACK = ('m_minus', 'st_spr')
DICT_COLS = frozenset(('day', 'role', 'addr', 'status', 'st_res', 'st_1s', 'st_10s', 'st_60s', 'st_end', 'dpt', 'st_spr'))
MM_NEED = ('day', 'mid', 'addr', 'gross_res_mk', 'gross_res_tk', 'rt_pnl', 'rt_sh', 'resid_res', 'resid_drift', 'net_pos', 'fee_res', 'fee_fill',
           'net_res', 'res_status', 'n_mk', 'n_tk', 'n_excl')
MM_DICT = frozenset(('day', 'addr', 'res_status'))

# ---- outputs: exact columns and dtypes ----
CL = dict(day='str', mid='int64', addr='str', rank='int64', entry_sid='str', entry_mk_idx='int64', entry_hash='str', entry_role='str',
          entry_side='str', entry_settle_ms='int64', entry_p='float64', close_sid='str', close_mk_idx='int64', close_hash='str', close_role='str',
          close_settle_ms='int64', close_p='float64', role_pair='str', sh='float64', pnl='float64', fee_fill_entry='float64', fee_fill_close='float64',
          fee_res_entry='float64', fee_res_close='float64', net='float64', net_feeres='float64', hold_ms='int64', dust='bool', res_ok='bool')
MK_INT = ('n_rows_ok', 'n_rows_self', 'n_rows_excl', 'n_closes', 'n_dust')
MK_VAL = ('sh_buy', 'sh_sell', 'net_pos', 'max_abs_pos', 'close_sh', 'close_pnl', 'close_fee_fill', 'close_fee_res', 'resid_sh', 'resid_end',
          'resid_resol', 'resid_res', 'resid_fee_fill', 'resid_fee_res', 'self_sh', 'self_gross', 'self_fee_fill', 'self_fee_res', 'fee_fill_tot',
          'fee_res_tot', 'gross_rows', 'V', 'm_end', 'econ_gross', 'econ_rt_pnl', 'econ_net', 'dev_gross', 'dev_net')
MKS = dict(day='str', mid='int64', addr='str', rank='int64', **{k: 'int64' for k in MK_INT}, **{k: 'float64' for k in MK_VAL}, res_ok='bool',
           end_ok='bool', status='str')
CK = ('n_cohort', 'sh_cohort', 'g_1s', 'g_10s', 'g_60s', 'g_end', 'g_res', 'ff_cohort', 'fr_cohort', 'n_res_all', 'sh_res_all', 'g_res_all', 'fr_res_all')
RULES = dict(
    population='econ/identities.json ex_ante.top10_shares (ranks 1-3 = the contract top 3, all 10 = the benchmark); their fills_valued rows (maker '
               'and taker legs) on the measurement days, read per day through the row-group statistics of day; ok / excluded counts = maker_market',
    lots='status ok rows, self-trades out (pf_mt_econ.fifo); per address x market x day',
    order='econ FIFO order (settle_ms, sid, px_rank, mk_idx, maker first) and its per-side cumulative shares; unit j bought pairs unit j sold; a '
          'close = one interval of the merged boundaries in [0, min(bought, sold)]; entry = the earlier leg, close = the later; hold = close settle_ms '
          '- entry settle_ms (UUIDv7 creation, the FIFO clock)',
    values='P&L = shares x (sell - buy price), YES space, V-free; leg fees prorated by shares at the fill price (fee_fill, primary) and at '
           'resolution (fee_res, sensitivity)',
    dust=f'closes below {DUST_SH:g} shares stay in every sum, are flagged dust, never counted nor in holding-time quantiles',
    residual='rq = q - attributed; value to END sum s rq (m_end - p), to resolution sum s rq (V - p), END->resolution (V - m_end) x net position; '
             'prorated fees; max |position| along the chronological lot path (self-trades out)',
    reconcile=f'vs econ maker_market per address x market x day, refusing: rows; attributed = mq / rq (tol {SH_TOL_ABS:g} + {SH_TOL_REL:g} x traded '
              f'shares); rt_pnl, rt_sh; gross identity; gross from rows; resid_res, resid_drift, net_pos, fee_fill, fee_res; net identity; fee '
              f'proration; close shares = X; residual shares = |net position| (tolerance {REL_TOL:g} relative, floor 1)',
    summary='closes in resolved markets (others counted); share-weighted holding-time quantiles (smallest hold reaching q of the shares) and '
            'the lower count median; residual_summary over resolved markets',
    cohorts='lot rows with ok 1s / 10s / 60s / END marks and a resolution; scope x role x depth x day; net = gross - fee_fill at marks, gross - '
            'fee_res at resolution; per share on own cohort shares and on the common denominator (all resolved lot shares of the scope and day)',
    answer=f'turnover_itself_poor: close net (fee_fill) <= {EPS_USD:g}; good_turnover_lost_to_residual: close net > {EPS_USD:g} and residual value '
           f'to resolution < -{EPS_USD:g}; else good_turnover_residual_not_negative; no_rows / no_resolved_markets / no_closes first; '
           f'sensitivity: fees at resolution (close net_feeres; residual value - its fee_res)')


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _clean(o):
    """A JSON-safe copy (NaN / inf -> None, numpy scalars -> Python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _json(path, obj):
    Path(path).write_text(json.dumps(_clean(obj), indent=1, sort_keys=True, allow_nan=False) + '\n')


# ------------------------------------------------------------------ arrow helpers ------------------------------------------------------------------
def _str(col):
    return pc.cast(col, pa.string()) if pa.types.is_dictionary(col.type) else col


def _np(col):
    return col.to_numpy() if isinstance(col, pa.ChunkedArray) else col.to_numpy(zero_copy_only=False)


def _eq(col, val):
    return _np(pc.fill_null(pc.equal(col, val), False)).astype(bool)


def _isin(col, vals):
    return _np(pc.fill_null(pc.is_in(col, value_set=pa.array(list(vals), pa.string())), False)).astype(bool)


def _codes(col, vocab, what):
    ix = pc.index_in(col, value_set=pa.array(list(vocab), pa.string()))
    need(ix.null_count == 0, f'{what}: a value outside {list(vocab)}')
    return _np(ix).astype(np.int64)


def _mask(col, values):
    """Per row of a (dictionary or plain) string column: is its value in `values` (exact; nulls False)? -> numpy bool."""
    vs = pa.array(sorted(values), pa.string())
    parts = []
    for ch in col.chunks:
        if pa.types.is_dictionary(ch.type):
            dm = _np(pc.fill_null(pc.is_in(pc.cast(ch.dictionary, pa.string()), value_set=vs), False)).astype(bool)
            ix = _np(pc.fill_null(ch.indices, 0)).astype(np.int64)
            m = dm[ix] if len(dm) else np.zeros(len(ch), bool)
            parts.append(m & _np(ch.is_valid()).astype(bool))
        else:
            parts.append(_np(pc.fill_null(pc.is_in(ch, value_set=vs), False)).astype(bool))
    return np.concatenate(parts) if parts else np.zeros(0, bool)


def _spelling(col, addrs, kind):
    """Refuse when a value equals a top-10 address only case-insensitively (two spellings of one address)."""
    low, exact = pa.array(sorted({a.lower() for a in addrs}), pa.string()), pa.array(sorted(addrs), pa.string())
    for ch in col.chunks:
        vals = pc.cast(ch.dictionary, pa.string()) if pa.types.is_dictionary(ch.type) else pc.unique(ch)
        bad = pc.and_(pc.fill_null(pc.is_in(pc.utf8_lower(vals), value_set=low), False),
                      pc.invert(pc.fill_null(pc.is_in(vals, value_set=exact), False)))
        need(not bool(pc.any(bad).as_py()), f'{kind}: a top-10 address appears under another spelling (refused)')


def check_schema(path, cols, spec, kind):
    need(Path(path).is_file(), f'{kind}: input missing: {Path(path).name}')
    sch = pq.read_schema(path)
    for c in cols:
        need(c in sch.names, f'{kind}: required column {c} missing ({Path(path).name})')
        need(E._type_ok(sch.field(c).type, spec[c]), f'{kind}: column {c} is {sch.field(c).type}, the econ schema says {spec[c]}')
    return sch


def load_top10(path):
    """econ/identities.json ex_ante.top10_shares -> the 10 addresses by rank; ranks 1-3 must equal the contract's top 3."""
    need(Path(path).is_file(), f'identities: input missing: {Path(path).name}')
    ea = json.loads(Path(path).read_text()).get('ex_ante') or {}
    need(ea.get('day') == C.ID_DAY, f'identities: ex_ante is not from the ID day {C.ID_DAY}')
    lst = ea.get('top10_shares') or []
    need(len(lst) == NTOP and [x.get('rank') for x in lst] == list(range(1, NTOP + 1)), 'identities: ex_ante.top10_shares is not ranks 1-10')
    addrs = tuple(str(x.get('addr')) for x in lst)
    need(len({a.lower() for a in addrs}) == NTOP, 'identities: two top-10 entries are one address')
    need(addrs[:3] == TOP3, 'identities: ex_ante ranks 1-3 differ from the contract top 3 (decision_contract.md population table; refused)')
    return addrs


def day_groups(pf, days, kind):
    """Row groups per measurement day from the footer statistics of `day` -> ({day: [(index, mixed)]}, info). A row group whose day range leaves
    the consumed days refuses before it is read; one without statistics is classified from its own day column (read alone)."""
    md = pf.metadata
    out, info = {d: [] for d in days}, dict(row_groups=md.num_row_groups, no_stats=0, skipped=0, mixed=0, empty=0)
    if md.num_row_groups == 0:
        return out, info
    j = next(k for k in range(md.num_columns) if md.row_group(0).column(k).path_in_schema == 'day')
    lo_ok, hi_ok = min(C.ALLOWED_DAYS), max(C.ALLOWED_DAYS)
    for i in range(md.num_row_groups):
        if md.row_group(i).num_rows == 0:
            info['empty'] += 1
            continue
        st = md.row_group(i).column(j).statistics
        if st is not None and st.has_min_max:
            a, b = (x.decode() if isinstance(x, bytes) else str(x) for x in (st.min, st.max))
        else:
            info['no_stats'] += 1
            u = pc.unique(_str(pf.read_row_group(i, columns=['day']).column('day'))).to_pylist()
            need(len(u) > 0 and all(isinstance(x, str) for x in u), f'{kind}: row group {i} has a missing day (refused)')
            a, b = min(u), max(u)
        need(len(a) == 8 and len(b) == 8 and lo_ok <= a and b <= hi_ok, f'{kind}: row group {i} holds a day outside the consumed days (refused; never read)')
        hit = [d for d in days if a <= d <= b]
        info['skipped'] += int(not hit)
        info['mixed'] += int(bool(hit) and a != b)
        for d in hit:
            out[d].append((i, a != b))
    return out, info


def read_rows(pf, groups, d, cols, dict_cols, addrs, cnt, kind):
    """The top-10 rows of day d from its row groups (other addresses never leave the group) -> arrow Table of plain columns, or None."""
    parts = []
    for i, mixed in groups:
        t = pf.read_row_group(i, columns=list(cols))
        cnt[f'{kind}_rows_scanned'] = cnt.get(f'{kind}_rows_scanned', 0) + t.num_rows
        a = t.column('addr')
        _spelling(a, addrs, kind)
        m = _mask(a, addrs)
        if mixed:
            m &= _mask(t.column('day'), (d,))
        if m.any():
            f = t.filter(pa.array(m))
            parts.append(pa.table({c: _str(f.column(c)) if c in dict_cols else f.column(c) for c in cols}))
        del t, a
    if not parts:
        return None
    T = pa.concat_tables(parts)
    need(bool(pc.all(pc.equal(T.column('day'), d)).as_py()), f'{kind}: a row read for {d} carries another day')
    return T


# ------------------------------------------------------------------ one day ------------------------------------------------------------------
def _group_const(G, x, m, nG, what):
    """A per-row value that must be constant inside each address x market group (ok rows; NaN-aware) -> per-group value (NaN if none)."""
    out = np.full(nG, np.nan)
    if not m.any():
        return out
    agg = pd.DataFrame({'g': G[m], 'x': x[m]}).groupby('g')['x'].agg(['min', 'max', 'count', 'size'])
    cons = ((agg['count'] == agg['size']) & (agg['min'] == agg['max'])) | (agg['count'] == 0)
    need(bool(cons.all()), f'fills_valued: {what} differs inside one address x market')
    out[agg.index.to_numpy(np.int64)] = agg['min'].to_numpy(float)
    return out


def arrays(T, addrs, has_dpt):
    """One day's top-10 rows -> numpy arrays, codes and the address x market groups; refuses on a value econ never writes on an ok row."""
    n = T.num_rows
    A = dict(n=n, a=_codes(T.column('addr'), addrs, 'fills_valued address'), role=_codes(T.column('role'), ROLES, 'fills_valued role'))
    for k, c in (('mid', 'mid'), ('mki', 'mk_idx'), ('pxr', 'px_rank'), ('s', 's'), ('settle', 'settle_ms')):
        A[k] = _np(T.column(c)).astype(np.int64)
    for c in ('q', 'p', 'V', 'm_end', 'fee_fill', 'fee_res', 'gross_res', 'mq', 'rq') + tuple(f'v_{h}' for h in MARKS):
        A[c] = _np(T.column(c)).astype(np.float64)
    A['self'] = _np(pc.fill_null(T.column('self_trade'), False)).astype(bool)
    A['ok'] = _eq(T.column('status'), OK)
    A['res_valid'] = _isin(T.column('st_res'), RES_VALID)
    for h in MARKS:
        A[f'ok_{h}'] = _eq(T.column(f'st_{h}'), OK)
    ok = A['ok']
    need(bool(np.isin(A['s'][ok], (-1, 1)).all()), 'fills_valued: s outside {-1, +1} on an ok row')
    need(bool((np.isfinite(A['q']) & (A['q'] > 0))[ok].all()), 'fills_valued: q not a positive number on an ok row')
    need(bool((np.isfinite(A['p']) & (A['p'] > 0) & (A['p'] < 1))[ok].all()), 'fills_valued: p outside (0, 1) on an ok row')
    need(bool((np.isfinite(A['V']) == A['res_valid'])[ok].all()), 'fills_valued: V is finite exactly where st_res is ok / tie (violated on an ok row)')
    need(bool(np.isfinite(A['fee_fill'])[ok].all()), 'fills_valued: fee_fill missing on an ok row')
    rv = ok & A['res_valid']
    need(bool((np.isfinite(A['fee_res']) & np.isfinite(A['gross_res']))[rv].all()), 'fills_valued: fee_res / gross_res missing on a resolved ok row')
    need(bool((np.isfinite(A['mq']) & np.isfinite(A['rq']))[ok].all()), 'fills_valued: mq / rq missing on an ok row')
    for h in MARKS:
        need(bool(np.isfinite(A[f'v_{h}'])[ok & A[f'ok_{h}']].all()), f'fills_valued: v_{h} missing where st_{h} is ok')
    if has_dpt:
        A['depth'] = _codes(T.column('dpt'), DEPTH, 'fills_valued dpt')
    else:                                     # fallback: cents from the pre-trade mid on the maker leg's side
        mm = _np(T.column('m_minus')).astype(np.float64)
        s_mk = np.where(A['role'] == 0, A['s'], -A['s'])
        x = np.round(100. * s_mk * (mm - A['p']), 9)
        okd = _eq(T.column('st_spr'), OK) & np.isfinite(x)
        A['depth'] = np.where(~okd, 5, np.where(x < 0, 0, np.where(x < 1, 1, np.where(x < 2, 2, np.where(x < 3, 3, 4))))).astype(np.int64)
    sid = T.column('sid').combine_chunks()
    u = pc.unique(sid)
    rk = np.empty(len(u), np.int64)
    rk[_np(pc.sort_indices(u)).astype(np.int64)] = np.arange(len(u))
    A['sidr'] = rk[_np(pc.index_in(sid, value_set=u)).astype(np.int64)]
    A['sid'], A['hash'] = sid, T.column('hash').combine_chunks()
    need(bool((A['mid'] >= 0).all()) and (n == 0 or int(A['mid'].max()) < (1 << 58)), 'fills_valued: mid out of range')
    uk, G = np.unique(A['mid'] * KEY_MUL + A['a'], return_inverse=True)
    A['uk'], A['G'], A['nG'] = uk, G.reshape(-1).astype(np.int64), len(uk)
    vc = pc.value_counts(T.column('status')).to_pylist()
    A['status_counts'] = {str(x['values']): int(x['counts']) for x in vc}
    return A


def lots(A):
    """FIFO completed closes of one day (ORDER) -> B / S (day rows of the buy / sell leg), E / K (entry / close leg), sh, att (attributed
    shares per day row), X and traded shares per group, maxabs (max |position| per group)."""
    n, nG, G = A['n'], A['nG'], A['G']
    iv = np.flatnonzero(A['ok'] & ~A['self'])
    nl = len(iv)
    e0 = np.zeros(0, np.int64)
    out = dict(att=np.zeros(n), X=np.zeros(nG), traded=np.zeros(nG), maxabs=np.zeros(nG), B=e0, S=e0, E=e0, K=e0, sh=np.zeros(0), n_lots=nl)
    if nl == 0:
        return out
    g, s, q = G[iv], A['s'][iv], A['q'][iv]
    keys = (A['role'][iv], A['mki'][iv], A['pxr'][iv], A['sidr'][iv], A['settle'][iv])
    o2 = np.lexsort(keys + (s, g))                       # pf_mt_econ.fifo: group, side, then the FIFO order
    gs, ss, qs = g[o2], s[o2], q[o2]
    blk = np.cumsum(np.r_[True, (gs[1:] != gs[:-1]) | (ss[1:] != ss[:-1])]) - 1
    cq = pd.Series(qs).groupby(blk).cumsum().to_numpy()
    tb, ts_ = np.zeros(nG), np.zeros(nG)
    np.add.at(tb, gs[ss == 1], qs[ss == 1])
    np.add.at(ts_, gs[ss == -1], qs[ss == -1])
    X = np.minimum(tb, ts_)
    typ = (ss == -1).astype(np.int8)                     # 0 = a buy boundary, 1 = a sell boundary (a buy first on equal values)
    e = np.lexsort((typ, cq, gs))
    ev, et, eg, er = cq[e], typ[e], gs[e], o2[e]         # er: position in the lot arrays
    ne = len(e)
    prev = np.r_[0., ev[:-1]]
    prev[np.r_[True, eg[1:] != eg[:-1]]] = 0.
    Xg = X[eg]
    ln = np.minimum(ev, Xg) - np.minimum(prev, Xg)
    kj = np.flatnonzero(ln > 0)
    pos = np.arange(ne)
    nb = np.minimum.accumulate(np.where(et == 0, pos, ne)[::-1])[::-1]     # the first buy / sell boundary at or after each position
    ns = np.minimum.accumulate(np.where(et == 1, pos, ne)[::-1])[::-1]
    need(bool((nb[kj] < ne).all() and (ns[kj] < ne).all()), 'internal: a close without a buy or a sell leg')
    need(bool((eg[nb[kj]] == eg[kj]).all() and (eg[ns[kj]] == eg[kj]).all()), 'internal: a close paired across address x market groups')
    bl, sl, sh = er[nb[kj]], er[ns[kj]], ln[kj]
    att_l = np.bincount(bl, weights=sh, minlength=nl) + np.bincount(sl, weights=sh, minlength=nl)
    o = np.lexsort(keys + (g,))                          # chronological order of the lots, both sides
    cr = np.empty(nl, np.int64)
    cr[o] = np.arange(nl)
    first = cr[bl] < cr[sl]
    el, kl = np.where(first, bl, sl), np.where(first, sl, bl)
    path = pd.Series((s * q)[o]).groupby(g[o]).cumsum().to_numpy()
    maxabs = np.zeros(nG)
    np.maximum.at(maxabs, g[o], np.abs(path))
    out['att'][iv] = att_l
    out.update(B=iv[bl], S=iv[sl], E=iv[el], K=iv[kl], sh=sh, X=X, traded=tb + ts_, maxabs=maxabs)
    return out


def close_values(A, L):
    """Per close: P&L (YES space, V-free), prorated fees of both legs, holding time, role pair, dust."""
    B, S, E_, K, sh = L['B'], L['S'], L['E'], L['K'], L['sh']
    q, ff, fr = A['q'], A['fee_fill'], A['fee_res']
    V = dict(pnl=sh * (A['p'][S] - A['p'][B]), ffe=ff[E_] * sh / q[E_], ffc=ff[K] * sh / q[K], fre=fr[E_] * sh / q[E_], frc=fr[K] * sh / q[K],
             hold=A['settle'][K] - A['settle'][E_], rp=2 * A['role'][E_] + A['role'][K], dust=sh < DUST_SH)
    need(bool((V['hold'] >= 0).all()), 'internal: a close before its entry')
    need(bool((A['s'][B] == 1).all() and (A['s'][S] == -1).all()), 'internal: a close leg on the wrong side')
    return V


def groups_table(A, L, CV):
    """Per address x market group of the day: rows, lots, closes, residual, self-trades, fees and the recorded inventory excursion."""
    nG, G = A['nG'], A['G']
    ok = A['ok']
    lot, sr = ok & ~A['self'], ok & A['self']
    q, s, p = A['q'], A['s'], A['p']
    with np.errstate(divide='ignore', invalid='ignore'):
        rq = np.where(lot, q - L['att'], 0.)
        frac = np.where(lot, rq / q, 0.)

        def bs(m, w):
            return np.bincount(G[m], weights=w[m], minlength=nG)

        def bc(m):
            return np.bincount(G[m], minlength=nG).astype(np.int64)

        Vg = _group_const(G, A['V'], ok, nG, 'V (resolution)')
        Mg = _group_const(G, A['m_end'], ok, nG, 'm_end (END mid)')
        res_ok, end_ok = np.isfinite(Vg), np.isfinite(Mg)
        rn = lambda x: np.where(res_ok, x, np.nan)          # noqa: E731
        cg = G[L['E']]
        dust = CV['dust']
        P = dict(key=A['uk'], mid=A['uk'] // KEY_MUL, a=A['uk'] % KEY_MUL, n_rows_ok=bc(ok), n_rows_self=bc(sr), n_rows_excl=bc(~ok),
                 n_closes=np.bincount(cg[~dust], minlength=nG).astype(np.int64), n_dust=np.bincount(cg[dust], minlength=nG).astype(np.int64),
                 sh_buy=bs(lot, q * (s == 1)), sh_sell=bs(lot, q * (s == -1)), net_pos=bs(lot, s * q), max_abs_pos=L['maxabs'],
                 close_sh=np.bincount(cg, weights=L['sh'], minlength=nG), close_pnl=np.bincount(cg, weights=CV['pnl'], minlength=nG),
                 close_fee_fill=np.bincount(cg, weights=CV['ffe'] + CV['ffc'], minlength=nG),
                 close_fee_res=rn(np.bincount(cg, weights=CV['fre'] + CV['frc'], minlength=nG)),
                 resid_sh=bs(lot, rq), resid_res=rn(bs(lot, s * rq * (A['V'] - p))),
                 resid_end=np.where(end_ok, bs(lot, s * rq * (A['m_end'] - p)), np.nan),
                 resid_fee_fill=bs(lot, A['fee_fill'] * frac), resid_fee_res=rn(bs(lot, A['fee_res'] * frac)),
                 self_sh=bs(sr, q * (A['role'] == 0)), self_gross=rn(bs(sr, A['gross_res'])), self_fee_fill=bs(sr, A['fee_fill']),
                 self_fee_res=rn(bs(sr, A['fee_res'])), fee_fill_tot=bs(ok, A['fee_fill']), fee_res_tot=rn(bs(ok, A['fee_res'])),
                 gross_rows=rn(bs(ok, A['gross_res'])), V=Vg, m_end=Mg, res_ok=res_ok, end_ok=end_ok, X=L['X'], traded=L['traded'])
        P['resid_resol'] = (Vg - Mg) * P['net_pos']
    return P


def econ_day(T, addrs):
    """econ/maker_market rows of the day's top 10 -> arrays keyed like the groups (mid x 16 + address index)."""
    if T is None:
        return dict(n=0)
    a, mid = _codes(T.column('addr'), addrs, 'maker_market address'), _np(T.column('mid')).astype(np.int64)
    f = lambda c: _np(T.column(c)).astype(np.float64)    # noqa: E731
    return dict(n=T.num_rows, key=mid * KEY_MUL + a, gross=f('gross_res_mk') + f('gross_res_tk'), rt_pnl=f('rt_pnl'), rt_sh=f('rt_sh'),
                resid_res=f('resid_res'), resid_drift=f('resid_drift'), net_pos=f('net_pos'), fee_res=f('fee_res'), fee_fill=f('fee_fill'),
                net_res=f('net_res'), res_ok=_isin(T.column('res_status'), RES_VALID),
                n_ok=_np(T.column('n_mk')).astype(np.int64) + _np(T.column('n_tk')).astype(np.int64), n_excl=_np(T.column('n_excl')).astype(np.int64))


def reconcile(d, P, A, L, Mm, devs):
    """RECONCILE (see the module rules) -> the econ arrays aligned to the groups; refuses on the first failing check, with its reason."""
    nG = len(P['key'])
    need(Mm['n'] == nG, f'reconcile rows: {d}: {nG} fills_valued address x market groups vs {Mm["n"]} maker_market rows of the top 10')
    if nG == 0:
        return {}
    o = np.argsort(Mm['key'], kind='stable')
    M = {k: (v[o] if isinstance(v, np.ndarray) else v) for k, v in Mm.items()}
    need(bool((np.diff(M['key']) > 0).all()), f'reconcile rows: {d}: a duplicate maker_market (market, address) row')
    need(bool((M['key'] == P['key']).all()), f'reconcile rows: {d}: the maker_market (market, address) rows differ from the fills_valued groups')
    bad = (M['n_ok'] != P['n_rows_ok']) | (M['n_excl'] != P['n_rows_excl'])
    need(not bad.any(), f'reconcile rows: {d}: {int(bad.sum())} address x market groups whose ok / excluded row counts differ from maker_market '
                        f'(n_mk + n_tk, n_excl)')
    lot, sr, G = A['ok'] & ~A['self'], A['ok'] & A['self'], A['G']
    tol = SH_TOL_ABS + SH_TOL_REL * L['traded'][G]
    dev = np.maximum(np.abs(L['att'] - A['mq']), np.abs(A['q'] - L['att'] - A['rq']))
    devs['attributed'] = max(devs.get('attributed', 0.), float(dev[lot].max()) if lot.any() else 0.)
    bad = lot & ~(dev <= tol)
    need(not bad.any(), f'reconcile attributed: {d}: {int(bad.sum())} lot rows whose attributed shares differ from econ mq / rq (every matched '
                        f'share attributed once; max deviation {devs["attributed"]:.3g} shares)')
    need(bool(((A['mq'] == 0) & (A['rq'] == 0))[sr].all()), f'reconcile self: {d}: a self-trade row with econ mq / rq other than 0')
    has = P['n_rows_ok'] > 0
    res = has & P['res_ok']
    need(bool((P['res_ok'] == M['res_ok'])[has].all()), f'reconcile res_status: {d}: a resolution in fills_valued where maker_market has none, or the reverse')

    def chk(name, ours, theirs, mask):
        ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
        o_, t_ = ours[mask], theirs[mask]
        dv = np.where(np.isnan(o_) & np.isnan(t_), 0., np.abs(o_ - t_))
        ok_ = dv <= REL_TOL * np.maximum(1., np.abs(np.nan_to_num(t_)))
        fin = dv[np.isfinite(dv)]
        devs[name] = max(devs.get(name, 0.), float(fin.max()) if len(fin) else 0.)
        if not ok_.all():
            j = np.flatnonzero(mask)[np.flatnonzero(~ok_)[0]]
            need(False, f'reconcile {name}: {d}: {int((~ok_).sum())} of {int(mask.sum())} address x market rows exceed {REL_TOL:g} relative (first: '
                        f'mid {int(P["mid"][j])} rank {int(P["a"][j]) + 1}: ours {ours[j]!r} vs {theirs[j]!r})')

    chk('rt_pnl', P['close_pnl'], M['rt_pnl'], has)
    chk('rt_sh', P['close_sh'], M['rt_sh'], has)
    chk('gross_identity', P['close_pnl'] + P['resid_res'] + P['self_gross'], M['gross'], res)
    chk('gross_rows', P['gross_rows'], M['gross'], res)
    chk('resid_res', P['resid_res'], M['resid_res'], res)
    chk('resid_end', P['resid_end'], M['resid_drift'], has)
    chk('net_pos', P['net_pos'], M['net_pos'], has)
    chk('fee_fill', P['fee_fill_tot'], M['fee_fill'], has)
    chk('fee_res', P['fee_res_tot'], M['fee_res'], res)
    chk('net_identity', P['close_pnl'] - P['close_fee_fill'] + P['resid_res'] - P['resid_fee_fill'] - P['self_fee_fill'] +
        (P['fee_fill_tot'] - P['fee_res_tot']) + P['self_gross'], M['net_res'], res)
    chk('fee_fill_attribution', P['close_fee_fill'] + P['resid_fee_fill'] + P['self_fee_fill'], P['fee_fill_tot'], has)
    chk('fee_res_attribution', P['close_fee_res'] + P['resid_fee_res'] + P['self_fee_res'], P['fee_res_tot'], res)
    chk('close_sh_eq_cap', P['close_sh'], P['X'], has)
    chk('resid_sh_eq_abs_net_pos', P['resid_sh'], np.abs(P['net_pos']), has)
    return M


def closes_table(d, A, L, CV, res_close, addrs):
    """closes.parquet rows of the day (schema CL)."""
    E_, K = L['E'], L['K']
    n = len(E_)
    tk = lambda arr, ix: pc.take(arr, pa.array(np.asarray(ix, np.int64), pa.int64()))   # noqa: E731
    roles, sides, pairs = pa.array(list(ROLES), pa.string()), pa.array(['buy', 'sell'], pa.string()), pa.array(list(PAIRS), pa.string())
    cols = dict(day=pa.array(np.full(n, d, dtype=object), pa.string()), mid=A['mid'][E_], addr=tk(pa.array(list(addrs), pa.string()), A['a'][E_]),
                rank=A['a'][E_] + 1, entry_sid=tk(A['sid'], E_), entry_mk_idx=A['mki'][E_], entry_hash=tk(A['hash'], E_),
                entry_role=tk(roles, A['role'][E_]), entry_side=tk(sides, (A['s'][E_] == -1).astype(np.int64)), entry_settle_ms=A['settle'][E_],
                entry_p=A['p'][E_], close_sid=tk(A['sid'], K), close_mk_idx=A['mki'][K], close_hash=tk(A['hash'], K), close_role=tk(roles, A['role'][K]),
                close_settle_ms=A['settle'][K], close_p=A['p'][K], role_pair=tk(pairs, CV['rp']), sh=L['sh'], pnl=CV['pnl'], fee_fill_entry=CV['ffe'],
                fee_fill_close=CV['ffc'], fee_res_entry=CV['fre'], fee_res_close=CV['frc'], net=CV['pnl'] - CV['ffe'] - CV['ffc'],
                net_feeres=CV['pnl'] - CV['fre'] - CV['frc'], hold_ms=CV['hold'], dust=CV['dust'], res_ok=res_close)
    return pa.table(cols, schema=E._arrow_schema(CL))


def market_frame(d, P, M, addrs):
    """closes_markets rows of the day (schema MKS)."""
    has = P['n_rows_ok'] > 0
    res = has & P['res_ok']
    ours_gross = P['close_pnl'] + P['resid_res'] + P['self_gross']
    ours_net = ours_gross - P['fee_res_tot']
    X = {'day': np.full(len(P['key']), d, dtype=object), 'mid': P['mid'], 'addr': np.asarray(addrs, dtype=object)[P['a']], 'rank': P['a'] + 1}
    X.update({k: P[k] for k in MK_INT})
    X.update({k: P[k] for k in MK_VAL if k in P})
    X.update(econ_gross=np.where(res, M['gross'], np.nan), econ_rt_pnl=np.asarray(M['rt_pnl'], float), econ_net=np.where(res, M['net_res'], np.nan),
             dev_gross=np.where(res, ours_gross - M['gross'], np.nan), dev_net=np.where(res, ours_net - M['net_res'], np.nan),
             res_ok=P['res_ok'] & has, end_ok=P['end_ok'] & has, status=np.where(has, OK, 'all_rows_excluded').astype(object))
    return pd.DataFrame({k: X[k] for k in MKS})


def cohort_sums(A, nd):
    """Identical-cohort sums of the day per (address, role, depth) cell -> array [10, 2, nd, len(CK)]."""
    lot = A['ok'] & ~A['self']
    res = lot & A['res_valid']
    coh = res & A['ok_1s'] & A['ok_10s'] & A['ok_60s'] & A['ok_end']
    cell = (A['a'] * 2 + A['role']) * nd + A['depth']
    nc = NTOP * 2 * nd
    out = np.zeros((nc, len(CK)))

    def put(k, m, w=None):
        out[:, CK.index(k)] = np.bincount(cell[m], weights=None if w is None else w[m], minlength=nc)

    put('n_cohort', coh)
    put('sh_cohort', coh, A['q'])
    for h in MARKS:
        put(f'g_{h}', coh, A[f'v_{h}'])
    put('g_res', coh, A['gross_res'])
    put('ff_cohort', coh, A['fee_fill'])
    put('fr_cohort', coh, A['fee_res'])
    put('n_res_all', res)
    put('sh_res_all', res, A['q'])
    put('g_res_all', res, A['gross_res'])
    put('fr_res_all', res, A['fee_res'])
    return out.reshape(NTOP, 2, nd, len(CK))


# ------------------------------------------------------------------ summaries ------------------------------------------------------------------
def _scope_id(sc, addrs):
    return (int(sc[4:]), addrs[int(sc[4:]) - 1]) if sc.startswith('rank') else (None, '')


def _pc(x, den):
    return 100. * x / den if den > 0 else np.nan


def closes_summary(CA, days, addrs):
    """closes_summary rows: scope x day (+ all) x role pair (+ all), closes in resolved markets."""
    srt = np.argsort(CA['hold'], kind='stable')
    H = {k: v[srt] for k, v in CA.items()}
    rows = []
    for sc, mem in SCOPE_MEMBERS.items():
        rank, addr = _scope_id(sc, addrs)
        ms = np.isin(H['a'], mem)
        for di, d in enumerate(days + ('all',)):
            md = ms if d == 'all' else ms & (H['d'] == di)
            for pi, pr in enumerate(PAIRS + ('all',)):
                m0 = md if pr == 'all' else md & (H['rp'] == pi)
                m, mna = m0 & H['res'], m0 & ~H['res']
                nd_ = m & ~H['dust']
                sh, pnl, ff, fr = (float(H[k][m].sum()) for k in ('sh', 'pnl', 'ff', 'fr'))
                r = dict(scope=sc, rank=rank, addr=addr, day=d, role_pair=pr, n_closes=int(nd_.sum()), n_dust=int((m & H['dust']).sum()), sh=sh,
                         gross=pnl, fee_fill=ff, fee_res=fr, net=pnl - ff, net_feeres=pnl - fr, c_gross_sh=_pc(pnl, sh), c_net_sh=_pc(pnl - ff, sh),
                         c_net_feeres_sh=_pc(pnl - fr, sh), n_closes_res_na=int((mna & ~H['dust']).sum()), sh_res_na=float(H['sh'][mna].sum()))
                hs, w = H['hold'][nd_], H['sh'][nd_]
                if len(hs):
                    cw = np.cumsum(w)
                    for qq in QUANTS:
                        r[f'hold_p{int(round(qq * 100))}_s'] = float(hs[min(int(np.searchsorted(cw, qq * cw[-1], 'left')), len(hs) - 1)]) / 1000.
                    r['hold_med_count_s'] = float(hs[(len(hs) - 1) // 2]) / 1000.
                    r['hold_mean_s'] = float((hs * w).sum() / w.sum()) / 1000.
                else:
                    for qq in QUANTS:
                        r[f'hold_p{int(round(qq * 100))}_s'] = np.nan
                    r['hold_med_count_s'] = r['hold_mean_s'] = np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def scope_day(MK, mem, d):
    """Aggregates of one scope and day over its markets: counts on every market, values over the resolved markets."""
    x = MK[MK['rank'].isin([i + 1 for i in mem]).to_numpy() & ((MK['day'] == d).to_numpy() if d != 'all' else True)]
    v = lambda c: x[c].to_numpy()   # noqa: E731
    okm = v('n_rows_ok') > 0
    rs = okm & v('res_ok')
    en = rs & v('end_ok')
    s_ = lambda c, m: float(v(c)[m].sum())   # noqa: E731
    R = dict(n_markets=int(okm.sum()), n_markets_res=int(rs.sum()), n_markets_res_na=int((okm & ~rs).sum()), n_markets_excl_only=int((~okm).sum()),
             n_markets_end=int(en.sum()), n_markets_end_na=int((rs & ~en).sum()), n_rows_ok=int(v('n_rows_ok')[rs].sum()),
             n_rows_self=int(v('n_rows_self')[rs].sum()), n_rows_excl=int(v('n_rows_excl').sum()), sh_traded=s_('sh_buy', rs) + s_('sh_sell', rs),
             n_closes=int(v('n_closes')[rs].sum()), n_dust=int(v('n_dust')[rs].sum()), close_sh=s_('close_sh', rs), close_gross=s_('close_pnl', rs),
             close_fee_fill=s_('close_fee_fill', rs), close_fee_res=s_('close_fee_res', rs), resid_sh=s_('resid_sh', rs),
             resid_end=s_('resid_end', en) if en.any() else np.nan, resid_resol=s_('resid_resol', en) if en.any() else np.nan,
             resid_res=s_('resid_res', rs), resid_fee_fill=s_('resid_fee_fill', rs), resid_fee_res=s_('resid_fee_res', rs),
             fee_drift=s_('fee_fill_tot', rs) - s_('fee_res_tot', rs), self_sh=s_('self_sh', rs), self_gross=s_('self_gross', rs),
             self_fee_fill=s_('self_fee_fill', rs), self_fee_res=s_('self_fee_res', rs), econ_gross=s_('econ_gross', rs), econ_net=s_('econ_net', rs),
             max_abs_pos_max=float(v('max_abs_pos')[rs].max()) if rs.any() else np.nan,
             max_abs_pos_p50=float(np.median(v('max_abs_pos')[rs])) if rs.any() else np.nan,
             max_abs_pos_mean=float(v('max_abs_pos')[rs].mean()) if rs.any() else np.nan,
             end_abs_pos_max=float(np.abs(v('net_pos')[rs]).max()) if rs.any() else np.nan)
    R['close_net'] = R['close_gross'] - R['close_fee_fill']
    R['close_net_feeres'] = R['close_gross'] - R['close_fee_res']
    R['c_close_net_sh'], R['c_close_net_feeres_sh'] = _pc(R['close_net'], R['close_sh']), _pc(R['close_net_feeres'], R['close_sh'])
    R['share_closed'] = 2. * R['close_sh'] / R['sh_traded'] if R['sh_traded'] > 0 else np.nan
    R['identity'] = R['close_net'] + R['resid_res'] - R['resid_fee_fill'] - R['self_fee_fill'] + R['fee_drift'] + R['self_gross']
    R['identity_dev'] = R['identity'] - R['econ_net']
    return R


def reading(R, alt=False):
    """The ANSWER predicate of one scope-day aggregate (alt: fees valued at resolution)."""
    if R['n_markets'] == 0:
        return 'no_rows'
    if R['n_markets_res'] == 0:
        return 'no_resolved_markets'
    if R['n_closes'] == 0:
        return 'no_closes'
    net = R['close_net_feeres'] if alt else R['close_net']
    resid = R['resid_res'] - R['resid_fee_res'] if alt else R['resid_res']
    if not net > EPS_USD:
        return 'turnover_itself_poor'
    if resid < -EPS_USD:
        return 'good_turnover_lost_to_residual'
    return 'good_turnover_residual_not_negative'


def summaries(MK, days, addrs):
    """residual_summary rows and answer entries (one per scope x day, + all days)."""
    rs, ans = [], []
    for sc, mem in SCOPE_MEMBERS.items():
        rank, addr = _scope_id(sc, addrs)
        for d in days + ('all',):
            R = scope_day(MK, mem, d)
            rs.append(dict(scope=sc, rank=rank, addr=addr, day=d, **R))
            a1, a2 = reading(R), reading(R, alt=True)
            ans.append(dict(scope=sc, rank=rank, addr=addr, day=d, reading=a1, reading_fee_res=a2, fee_convention_sensitive=a1 != a2,
                            n_markets=R['n_markets'], n_markets_res=R['n_markets_res'], n_markets_res_na=R['n_markets_res_na'], n_closes=R['n_closes'],
                            n_dust=R['n_dust'], close_sh=R['close_sh'], close_gross=R['close_gross'], close_net=R['close_net'],
                            c_close_net_sh=R['c_close_net_sh'], close_net_feeres=R['close_net_feeres'], c_close_net_feeres_sh=R['c_close_net_feeres_sh'],
                            resid_sh=R['resid_sh'], resid_res=R['resid_res'], resid_res_net_feeres=R['resid_res'] - R['resid_fee_res'],
                            resid_end=R['resid_end'], n_markets_end=R['n_markets_end'], econ_net=R['econ_net'], max_abs_pos_max=R['max_abs_pos_max']))
    return pd.DataFrame(rs), ans


def tally(ans, days, scopes):
    out = {k: 0 for k in READINGS}
    alt = {k: 0 for k in READINGS}
    n_sens = 0
    for e in ans:
        if e['scope'] in scopes and e['day'] in days:
            out[e['reading']] += 1
            alt[e['reading_fee_res']] += 1
            n_sens += int(e['fee_convention_sensitive'])
    return dict(n_address_days=int(sum(out.values())), reading=out, reading_fee_res=alt, n_fee_convention_sensitive=n_sens)


def cohorts_frame(CS, days, depth_vocab, depth_source, addrs):
    """cohorts.csv rows: scope x day (+ all) x role (+ all) x depth (+ all)."""
    rows = []
    ix = {k: i for i, k in enumerate(CK)}
    for sc in COHORT_SCOPES:
        mem = list(SCOPE_MEMBERS[sc])
        rank, addr = _scope_id(sc, addrs)
        for di, d in enumerate(days + ('all',)):
            Xs = (CS[:, mem] if d == 'all' else CS[di:di + 1, mem]).sum(axis=(0, 1))       # [role, depth, k]
            common = float(Xs[:, :, ix['sh_res_all']].sum())
            for ri, role in enumerate(ROLES + ('all',)):
                Xr = Xs.sum(axis=0) if role == 'all' else Xs[ri]
                for dj, dep in enumerate(depth_vocab + ('all',)):
                    v = Xr.sum(axis=0) if dep == 'all' else Xr[dj]
                    sh, sh_all = float(v[ix['sh_cohort']]), float(v[ix['sh_res_all']])
                    r = dict(scope=sc, rank=rank, addr=addr, day=d, role=role, depth=dep, depth_source=depth_source, n_cohort=int(v[ix['n_cohort']]),
                             sh_cohort=sh, n_res_all=int(v[ix['n_res_all']]), sh_res_all=sh_all, coverage=sh / sh_all if sh_all > 0 else np.nan,
                             sh_common=common)
                    g = {h: float(v[ix[f'g_{h}']]) for h in HZ}
                    net = {h: g[h] - float(v[ix['ff_cohort']]) for h in MARKS}
                    net['res'] = g['res'] - float(v[ix['fr_cohort']])
                    for h in HZ:
                        r[f'gross_{h}'], r[f'net_{h}'] = g[h], net[h]
                    for h in HZ:
                        r[f'c_gross_{h}_own'], r[f'c_net_{h}_own'] = _pc(g[h], sh), _pc(net[h], sh)
                    for h in HZ:
                        r[f'c_gross_{h}_common'], r[f'c_net_{h}_common'] = _pc(g[h], common), _pc(net[h], common)
                    ga, na = float(v[ix['g_res_all']]), float(v[ix['g_res_all']] - v[ix['fr_res_all']])
                    r.update(gross_res_all=ga, net_res_all=na, c_gross_res_all=_pc(ga, sh_all), c_net_res_all=_pc(na, sh_all))
                    rows.append(r)
    return pd.DataFrame(rows)


def _git(files):
    """HEAD and whether the tool files are tracked and unmodified (recorded for the freeze; never decides anything here)."""
    try:
        g = ['git', '-C', str(C.REPO)]
        head = subprocess.run(g + ['rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30).stdout.strip() or None
        rel = [str(Path(f).resolve().relative_to(C.REPO)) for f in files]
        tracked = subprocess.run(g + ['ls-files', '--error-unmatch', '--'] + rel, capture_output=True, text=True, timeout=30).returncode == 0
        st = subprocess.run(g + ['status', '--porcelain', '--'] + rel, capture_output=True, text=True, timeout=30)
        return dict(head=head, tool_files=rel, tracked=tracked, unmodified=bool(tracked and st.returncode == 0 and not st.stdout.strip()))
    except (OSError, ValueError, subprocess.SubprocessError):
        return dict(head=None, tool_files=None, tracked=None, unmodified=None)


# ------------------------------------------------------------------ run ------------------------------------------------------------------
def run(inp=None, out=None, days=None, guard_mib=C.GUARD_MIB, decision_receipt=None, log=print):
    """The closes build (module rules): preflight, one day at a time (read, FIFO, reconcile, write), summaries; write-once output."""
    t0, c0, asof0 = time.time(), time.process_time(), _now()
    days = tuple(days) if days is not None else tuple(C.MEASURE_DAYS)
    need(len(days) >= 1 and len(set(days)) == len(days) and all(d in C.MEASURE_DAYS for d in days),
         'days must be distinct measurement days (pf_mt_common.MEASURE_DAYS; refused before any path is built)')
    days = tuple(sorted(days))
    out = Path(out) if out is not None else C.RECEIPT / 'closes'
    need(not out.exists(), f'the closes output exists (write-once): {out}')
    need(out.parent.is_dir(), f'the receipt directory is missing: {out.parent}')
    inp = Path(inp) if inp is not None else C.OUT
    IN = dict(fills_valued=inp / 'econ' / 'fills_valued.parquet', maker_market=inp / 'econ' / 'maker_market.parquet',
              identities=inp / 'econ' / 'identities.json')
    addrs = load_top10(IN['identities'])
    sch = check_schema(IN['fills_valued'], FV_NEED, E.FV, 'fills_valued')
    has_dpt = 'dpt' in sch.names
    if has_dpt:
        check_schema(IN['fills_valued'], ('dpt',), E.FV, 'fills_valued')
    else:
        check_schema(IN['fills_valued'], FV_FALLBACK, E.FV, 'fills_valued')
    check_schema(IN['maker_market'], MM_NEED, E.MM, 'maker_market')
    stat0 = {k: (p.stat().st_size, p.stat().st_mtime_ns) for k, p in IN.items()}
    fv_cols = FV_NEED + (('dpt',) if has_dpt else FV_FALLBACK)
    pf = pq.ParquetFile(IN['fills_valued'], read_dictionary=[c for c in fv_cols if c in DICT_COLS])
    mmf = pq.ParquetFile(IN['maker_market'], read_dictionary=sorted(MM_DICT))
    fv_groups, fv_info = day_groups(pf, days, 'fills_valued')
    mm_groups, mm_info = day_groups(mmf, days, 'maker_market')
    depth_vocab, depth_source = (DEPTH, 'dpt') if has_dpt else (FALLBACK_DEPTH, 'm_minus_fallback')
    nd = len(depth_vocab)
    tmp = out.parent / f'.{out.name}.partial_{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}_{os.getpid()}'
    need(not tmp.exists(), 'the partial output directory exists')
    tmp.mkdir()
    W = None
    try:
        W = pq.ParquetWriter(tmp / 'closes.parquet', E._arrow_schema(CL), compression='zstd')
        CS = np.zeros((len(days), NTOP, 2, nd, len(CK)))
        CA, MKs, devs, cnt, mem, cpu = [], [], {}, {}, {}, {}
        for di, d in enumerate(days):
            c_ = {}
            T = read_rows(pf, fv_groups[d], d, fv_cols, DICT_COLS, addrs, c_, 'fills_valued')
            Tm = read_rows(mmf, mm_groups[d], d, MM_NEED, MM_DICT, addrs, c_, 'maker_market')
            Mm = econ_day(Tm, addrs)
            del Tm
            c_.update(row_groups_fv=len(fv_groups[d]), row_groups_fv_mixed=sum(int(m) for _, m in fv_groups[d]), row_groups_mm=len(mm_groups[d]))
            if T is None:
                need(Mm['n'] == 0, f'reconcile rows: {d}: maker_market holds {Mm["n"]} top-10 rows, fills_valued none')
                cnt[d] = dict(c_, rows=0)
                continue
            A = arrays(T, addrs, has_dpt)
            del T
            L = lots(A)
            CV = close_values(A, L)
            P = groups_table(A, L, CV)
            M = reconcile(d, P, A, L, Mm, devs)
            res_close = P['res_ok'][A['G'][L['E']]] if len(L['E']) else np.zeros(0, bool)
            W.write_table(closes_table(d, A, L, CV, res_close, addrs))
            CA.append(dict(a=A['a'][L['E']].astype(np.int8), d=np.full(len(L['E']), di, np.int8), rp=CV['rp'].astype(np.int8), sh=L['sh'],
                           pnl=CV['pnl'], ff=CV['ffe'] + CV['ffc'], fr=CV['fre'] + CV['frc'], hold=CV['hold'], dust=CV['dust'], res=res_close))
            MKs.append(market_frame(d, P, M, addrs))
            CS[di] = cohort_sums(A, nd)
            cnt[d] = dict(c_, rows=A['n'], rows_ok=int(A['ok'].sum()), rows_self=int((A['ok'] & A['self']).sum()), rows_excl=int((~A['ok']).sum()),
                          lots=L['n_lots'], groups=A['nG'], closes=int((~CV['dust']).sum()), dust=int(CV['dust'].sum()),
                          status=A['status_counts'], depth={depth_vocab[k]: int(v_) for k, v_ in zip(*np.unique(A['depth'], return_counts=True))})
            log(f'[closes] {d} rows {A["n"]} lots {L["n_lots"]} groups {A["nG"]} closes {cnt[d]["closes"]} dust {cnt[d]["dust"]} hwm {E.hwm():.0f} MiB',
                flush=True)
            del A, L, CV, P, M
            E.trim()
            mem[d] = E.guard(guard_mib, d)
            cpu[d] = round(time.process_time() - c0, 2)
            need(cpu[d] < CPU_CAP_S, f'CPU self-guard: {cpu[d]:.0f} s >= {CPU_CAP_S:.0f} s after {d}')
        W.close()
        W = None
        CA = {k: np.concatenate([x[k] for x in CA]) for k in ('a', 'd', 'rp', 'sh', 'pnl', 'ff', 'fr', 'hold', 'dust', 'res')} if CA else \
            dict(a=np.zeros(0, np.int8), d=np.zeros(0, np.int8), rp=np.zeros(0, np.int8), sh=np.zeros(0), pnl=np.zeros(0), ff=np.zeros(0),
                 fr=np.zeros(0), hold=np.zeros(0, np.int64), dust=np.zeros(0, bool), res=np.zeros(0, bool))
        MK = pd.concat(MKs, ignore_index=True) if MKs else pd.DataFrame({k: pd.Series(dtype=object if t == 'str' else t) for k, t in MKS.items()})
        pq.write_table(E._to_arrow(MK[list(MKS)].reset_index(drop=True), MKS), tmp / 'closes_markets.parquet', compression='zstd')
        CSUM = closes_summary(CA, days, addrs)
        CSUM.to_csv(tmp / 'closes_summary.csv', index=False)
        RS, ans = summaries(MK, days, addrs)
        RS.to_csv(tmp / 'residual_summary.csv', index=False)
        COH = cohorts_frame(CS, days, depth_vocab, depth_source, addrs)
        COH.to_csv(tmp / 'cohorts.csv', index=False)
        tal = dict(top3=tally(ans, days, ('rank1', 'rank2', 'rank3')), ranks4_10=tally(ans, days, tuple(f'rank{i}' for i in range(4, NTOP + 1))),
                   top10=tally(ans, days, tuple(f'rank{i}' for i in range(1, NTOP + 1))),
                   top3_pooled_days=tally(ans, ('all',), ('rank1', 'rank2', 'rank3')))
        n_meas = len(days)
        ANSWER = dict(question='is completed turnover good and then lost through residual exposure, or is the completed turnover itself poor?',
                      contract=CONTRACT, review=REVIEW, rule=RULES['answer'], population='resolved markets of each scope and day (others counted)',
                      clocks='closes on the FIFO clock (settle_ms = UUIDv7 creation, an upper bound of each match); residual to END / resolution',
                      fees='primary: fee_fill (fill price, V-free); sensitivity: fee_res (resolution)', eps_usd=EPS_USD, days=list(days),
                      ci=f'none_G{n_meas}_lt_5' if n_meas < 5 else f'eligible_G{n_meas}', descriptive='consumed days; no strategy or profitability claim',
                      entries=ans, tally=tal)
        _json(tmp / 'answer.json', ANSWER)
        for k, p in IN.items():
            need((p.stat().st_size, p.stat().st_mtime_ns) == stat0[k], f'input {k} changed during the run (refused)')
        inputs = {k: dict(path=str(p), sha256=_sha(p), bytes=p.stat().st_size) for k, p in IN.items()}
        dec = Path(decision_receipt) if decision_receipt is not None else C.RECEIPT / 'decision' / 'decision.json'
        same = None
        if dec.is_file():
            try:
                same = json.loads(dec.read_text())['inputs']['fv']['sha256'] == inputs['fills_valued']['sha256']
            except (KeyError, TypeError, ValueError):
                same = None
        hwm = E.guard(guard_mib, 'summaries')
        outputs = {p.name: dict(sha256=_sha(p), bytes=p.stat().st_size) for p in sorted(tmp.iterdir())}
        for name, n_ in (('closes.parquet', int(len(CA['sh']))), ('closes_markets.parquet', len(MK)), ('closes_summary.csv', len(CSUM)),
                         ('residual_summary.csv', len(RS)), ('cohorts.csv', len(COH)), ('answer.json', len(ans))):
            outputs[name]['rows'] = n_
        srcs = [HERE / 'closes.py', HERE / 'closes_selftest.py']
        rec = dict(version=VERSION, module='tools/closes.py', contract=CONTRACT, review=REVIEW, run=C.RUN, asof_start=asof0, asof_end=_now(),
                   wall_s=round(time.time() - t0, 1), cpu_s=dict(by_day=cpu, total=round(time.process_time() - c0, 2), cap=CPU_CAP_S), days=list(days),
                   addresses=[dict(rank=i + 1, addr=a, top3=i < 3) for i, a in enumerate(addrs)], inputs=inputs,
                   same_fills_valued_as_decision=dict(receipt=str(dec), same=same),
                   sources={p.name: _sha(p) for p in srcs + [RESEARCH / 'pf_mt_common.py', RESEARCH / 'pf_mt_econ.py'] if p.is_file()},
                   git=_git([p for p in srcs if p.is_file()]), row_groups=dict(fills_valued=fv_info, maker_market=mm_info), depth_source=depth_source,
                   depth_vocab=list(depth_vocab), counts=cnt,
                   checks={k: dict(max_dev=v_, ok=True, tol=(f'{SH_TOL_ABS:g} + {SH_TOL_REL:g} x traded shares' if k == 'attributed' else
                                                             f'{REL_TOL:g} relative, floor 1')) for k, v_ in sorted(devs.items())},
                   constants=dict(dust_sh=DUST_SH, rel_tol=REL_TOL, sh_tol_abs=SH_TOL_ABS, sh_tol_rel=SH_TOL_REL, eps_usd=EPS_USD, quants=list(QUANTS),
                                  cpu_cap_s=CPU_CAP_S, top3=list(TOP3)),
                   rules=RULES, answer_tally=tal, outputs=outputs,
                   memory=dict(guard_mib=guard_mib, vmhwm_mib=hwm, by_day=mem, process='one process; one day at a time'),
                   not_measured='no model skill, latent policy or profitability for us: an accounting of recorded fills',
                   slot=os.environ.get('PF_SLOT'), argv=sys.argv)
        _json(tmp / 'closes_receipt.json', rec)
        need(not out.exists(), f'the closes output appeared during the run (write-once): {out}')
        os.rename(tmp, out)
        return rec
    except BaseException:
        if W is not None:
            try:
                W.close()
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(tmp, ignore_errors=True)
        raise


if __name__ == '__main__':
    a = sys.argv[1:]
    need(a == ['run'], 'usage: MT_RUN=v2 closes.py run')
    need(C.RUN == 'v2', f'the closes run reads the v2 roots (decision_contract.md amendment 2): set MT_RUN=v2 (got {C.RUN!r})')
    r = run()
    print(json.dumps(_clean(dict(outputs={k: v.get('rows') for k, v in r['outputs'].items()}, checks={k: v['max_dev'] for k, v in r['checks'].items()},
                                 tally=r['answer_tally'], vmhwm_mib=r['memory']['vmhwm_mib'], cpu_s=r['cpu_s']['total'])), indent=1))
