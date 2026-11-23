"""chain_context.py - THE ONE DEFINITION OF MARKET CONTEXT for the maker decision-chain tools (P-2026-005 MAKER-TRACE; 2026-11-23;
order_estimator_contract.md section 7.0; plans automatic/20261123_maker_decision_timing / _logic / _order_estimator). Descriptive only, on
consumed days, from saved v2 artifacts; no trading, no fit, no new data.

API NOTE (stable names - code against them). M = load_market(day, mid, addrs) loads ONE market (day 20261024..20261027; the 19 BOOK markets,
which carry depth + lifecycle; MT_RUN=v2 -> pf_mt_common.OUT) from saved artifacts: ws / we (venue ms), the pf_mt_btc 100 ms BTC grid on
Binance E_ms through pf_mt_decision.load_btc UNCHANGED (the day plus the previous consumed day's last hour as the pre-ws pad when that file
exists; never a day outside Oct 23-27), tob frames, depth frames / levels through pf_mt_lifecycle._book (usable frames, unknown cells NaN),
the lifecycle fills / orders tables, and the addresses' fills_valued legs (every status kept; positions use status-ok legs) joined to their
lifecycle fill window [x_lo_hard_ms, hi_ms] by (sid, mk_idx). Every query takes int venue-ms instants t (a scalar becomes a length-1 array),
returns a dict of numpy arrays, reads ONLY data at or before t, and carries a status (a value is NaN unless its status is ok):
btc_ret(M, t) = pf_mt_decision.btc_feats unchanged: r_1s / r_5s / r_30s (bps log returns of grid row floor(t / 100) <= t against t - h; also
keyed r_1000 / r_5000 / r_30000), r_ws, sig, p_fair, tau, code / status (ok | btc_missing | btc_invalid | after_we), ok, plus 'cause' (stale /
gap / no_data / missing grid rows behind a non-ok code) and 'counts' - missing or stale is never fresh; book(M, t) = the last USABLE depth frame
with upd_ms <= t: bid_tick / ask_tick (float cents; -inf / +inf = empty side, NaN = unknown cell above the best known level), bid_sz / ask_sz,
spread (ticks), mid_c (cents), mid, age_ms, stale (> 1,000 ms), status, bid_st / ask_st, ok (off BOOK: the tob econ MID rule; book_tob() is
always that rule); level_size(M, t, side, tick) -> size / status at that cell of the same frame; touch_distance(M, t, side, tick) -> ticks
(+ behind, 0 at, - inside; an empty own side is inside), cat at | inside | behind | unknown, status; position(M, addr, t, clock='hi') -> recorded
(status-ok legs with settle_ms < t), certain (hi_ms < t), possible (x_lo_hard_ms < t), pending (= possible - certain: x_lo_hard_ms < t <=
hi_ms) and pos (the declared clock: 'hi' | 'lo' | 'settle'), signed YES shares - RECORDED VENUE inventory, never client-known inventory (a
fill reaches the maker only through a private notice whose delay is unidentified); pending(M, addr, t) adds buy / sell shares and the
feasible range [pos_min, pos_max]; time_left(M, t) = we - t (ms), tau(M, t) in s; phase(M, t) -> ('pre' | 'live' | 'post', window minute);
over_window(M, fn, lo, hi, step_ms=100, D_ms=0) evaluates fn(M, t) -> values or (values, ok) at lo - D, every 100 ms grid instant strictly
inside, and hi -> min / max / first / last / cats / n_missing / timing_sensitive / status (over_window_profiles: D in {0, 250, 1000});
matched_times(M, addr, n, seed=20261123, phase_of, mask_fn, exclude_windows) -> deterministic Generator(PCG64) control instants in the same
market and phase (pool pre = [ws - 300 s, ws), live = [ws, we)) where mask_fn holds over the width-matched window (n per target; pass
n = draws_per_target(<targets of the address-day>) for >= 200 pooled draws). Also: null_summary, MOVE_THR / align_cat / move_cat /
book_change, pos_bin / age_bin / width_bin, feat_* factories for over_window, near_fill / resting population masks, level_series,
level_at_frames, frame_times, book_tob, btc_state, position_path, pending_legs, top3(), book_markets(). Selftest: chain_context_selftest.py.
Smoke (run_light, MT_RUN=v2): chain_context.py smoke <day> <mid>.

DETAILS
 Clocks never mixed: BTC on Binance E_ms (grid row g carries the last update with E_ms <= g), book on venue upd_ms, fills on their hard
   window [x_lo_hard_ms, hi_ms] (hi = min(settle_ms, first receipt); x_lo_hard = hi - 2,000 ms, -1 = unbounded: a receipt-hi fill) and
   settle_ms (UUIDv7 creation, an upper bound of the match). Venue state at t is not what a maker received at t: shift t by a declared delay
   profile D for a sensitivity, never as an identified private latency.
 Look-ahead: every feature at t reads grid rows <= t, frames with upd_ms <= t and fills whose window starts before t. pending() and the
   recorded/certain split use records created after t (reconstruction evidence about the venue state at t, not decision information).
   Population masks (near_fill with after_ms > 0, resting) use later fills by design: they define control populations, never features.
 Statuses are carried, never dropped: non-ok fills_valued legs stay in M.excluded[addr] and flag position status
   'excluded_leg_possible' from their window start on; legs without a lifecycle window get the settle-based window (win_status
   'no_life_window', counted).
 Inputs are bound by bytes / mtime (and sha256 when a hash_cache dict is passed: one hash per file per run); every count lands in M.counts,
   integrity checks in M.checks (tob vs depth frame sets must be equal, else refused).
"""
import datetime as dt
import hashlib
import json
import sys
import time
from collections import Counter, OrderedDict
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
import pf_mt_econ as E  # noqa: E402
import pf_mt_lifecycle as LC  # noqa: E402
import pf_mt_decision as DEC  # noqa: E402

pa.set_cpu_count(1)
pa.set_io_thread_count(1)

need, Forbidden = C.need, C.Forbidden
VERSION = 'chain_context_v1'

# ------------------------------------------------------------------ declared constants (none fitted; shared with the frozen builders) ------------------------------------------------------------------
SEED = DEC.SEED                                  # 20261123
TOP3 = DEC.TOP3                                  # decision_contract.md population table, ranks 1-3
STEP_MS = DEC.STEP                               # 100 ms: BTC grid and instant grid
PRE_MS = DEC.PRE_MS                              # 300 s: the pre-ws phase pool [ws - 300 s, ws)
PAD_MS = 60_000                                  # span pad (sig_60 needs 60 s of grid before the first instant)
HORIZONS = tuple(k * STEP_MS for _, k in DEC.R_STEPS)   # (1000, 5000, 30000) ms
BOOK_STALE_MS = E.STALE_MS                       # 1,000 ms
DEPTH_GAP_MS = LC.GAP_MS                         # 1,000 ms
MAX_LAG_MS = C.MAX_LAG_MS                        # 2,000 ms (x_lo_hard_ms = hi - MAX_LAG_MS)
NT = LC.NT                                       # 101 ticks (0..100 cents)
MIN_POOLED = 200                                 # >= 200 control draws pooled per address-day (CLAUDE.md rule 6)
D_PROFILES = (0, 250, 1000)                      # declared submission-delay profiles (ms), shown separately, never identified
MOVE_THR = {'r_1s': 0.5, 'r_5s': 1.0, 'r_30s': 2.0}     # bp: |r| >= thr is a move (contract 7.0)
BOOK_MOVE_TICKS, BOOK_MOVE_MS = 1.0, 5_000      # a mid / opposite-touch change >= 1 tick over 5 s
POS_EDGES, POS_LABELS = (50., 200., 1000.), ('[0,50)', '[50,200)', '[200,1000)', '>=1000')
AGE_EDGES, AGE_LABELS = (1_000, 10_000), ('le_1s', 'le_10s', 'gt_10s')
WIDTH_EDGES, WIDTH_LABELS = (50, 250, 1_000), ('le_50', 'le_250', 'le_1000', 'gt_1000')
LOAD_DAYS = tuple(C.MEASURE_DAYS)                # 20261024..20261027
TOB_PRE_NONBOOK_MS = PRE_MS + 5_000              # off BOOK only: tob frames before ws - 305 s are not loaded (pf_mt_decision rule)
I64MIN = int(np.iinfo(np.int64).min)
CLOCKS = ('hi', 'lo', 'settle')
BTC_CODE = np.array(DEC.ROW_ST[:4], dtype=object)       # ok, btc_missing, btc_invalid, after_we
BTC_FILE_ST = ('ok', 'stale', 'gap', 'no_data')          # pf_mt_btc.STATUS; dense code 4 = no grid row
BTC_ROW_NAME = {1: 'stale', 2: 'gap', 3: 'no_data', 4: 'missing'}
MARK = np.array(E.MARK_ST, dtype=object)                 # pf_mt_econ MID statuses (book_tob)
BOOK_MANIFEST = E.BOOK_MANIFEST
FV_READ = ('day', 'mid', 'sid', 'mk_idx', 'px_rank', 'role', 'addr', 'hash', 'cp_addr', 'kind', 'y_side', 's', 'q', 'p', 'self_trade',
           'settle_ms', 't_ms', 't_status', 'fee', 'fee_type', 'fee_dy', 'status', 'src_status')
TOB_READ = ('mid', 'upd_ms', 'recv_ns', 'conn', 'n_bid', 'n_ask', 'best_bid', 'best_bid_sh', 'best_ask', 'best_ask_sh', 'status')
LIFE_REQ = dict(sid='str', mk_idx='int64', mk_hash='str', y_side='str', y_ptick='int64', shares='float64', hi_ms='int64', lo_ms='int64',
                status='str', x_lo_hard_ms='int64')
ORDERS_REQ = dict(scenario='str', mk_hash='str', mk_signer='str', y_side='str', y_ptick='int64', persist_lo_ms='int64', persist_hi_ms='int64',
                  status='str')
_BTC_CACHE = OrderedDict()                       # (root, day) -> loaded BTC day (at most 2 kept)


# ------------------------------------------------------------------ small helpers ------------------------------------------------------------------
def day_ms(day):
    need(isinstance(day, str) and len(day) == 8 and day.isdigit(), f'bad day {day!r}')
    return int(dt.datetime.strptime(day, '%Y%m%d').replace(tzinfo=dt.timezone.utc).timestamp()) * 1000


def _guard(day, allowed, what):
    """Refuses BEFORE any path is built."""
    if not (isinstance(day, str) and day in allowed):
        raise Forbidden(f'{what}: day {day!r} is not allowed here (allowed: {", ".join(allowed)}); refused before any path was built')


def _t(t):
    a = np.asarray(t)
    if a.ndim == 0:
        a = a.reshape(1)
    need(a.ndim == 1, 'instants must be a scalar or a 1-d array')
    if a.dtype.kind == 'f':
        need(bool(np.isfinite(a).all()) and bool((a == np.round(a)).all()), 'float instants must be whole ms')
    else:
        need(a.dtype.kind in 'iu' or len(a) == 0, f'instants must be integers (ms), got {a.dtype}')
    return a.astype(np.int64)


def _akey(a):
    need(isinstance(a, str) and len(a) == 42 and a[:2].lower() == '0x', f'bad address {a!r}')
    return a.lower()


def _addrs(addrs):
    A = tuple(_akey(a) for a in ([addrs] if isinstance(addrs, str) else addrs))
    need(len(A) == len(set(A)), 'an address is listed twice')
    return A


def _side(side, n):
    """'bid' / 'ask' (or 0 / 1), scalar or length-n array -> int64 codes 0 bid / 1 ask."""
    if np.ndim(side) == 0:
        v = side.item() if hasattr(side, 'item') else side
        c = {'bid': 0, 'ask': 1}.get(v) if isinstance(v, str) else (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else None)
        need(c in (0, 1), "side must be 'bid' / 'ask' (or 0 / 1)")
        return np.full(n, c, np.int64)
    s = np.asarray(side)
    need(s.shape == (n,), 'side: one value per instant')
    if s.dtype.kind in 'iu':
        need(bool(np.isin(s, (0, 1)).all()), "side must be 'bid' / 'ask' (or 0 / 1)")
        return s.astype(np.int64)
    s = s.astype(str)
    out = np.where(s == 'bid', 0, np.where(s == 'ask', 1, -1)).astype(np.int64)
    need(bool((out >= 0).all()), "side must be 'bid' / 'ask' (or 0 / 1)")
    return out


def _ints(x, n):
    return np.broadcast_to(np.asarray(x, dtype=np.int64), (n,)).copy()


def _sel(conds, names, default, n):
    """First condition that holds wins (object array of names)."""
    out = np.full(n, default, dtype=object)
    for c, nm in reversed(list(zip(conds, names))):
        out[np.broadcast_to(np.asarray(c, bool), (n,))] = nm
    return out


def _str(col):
    return pc.cast(col, pa.string()) if pa.types.is_dictionary(col.type) else col


def _check_cols(path, want):
    path = Path(path)
    need(path.is_file(), f'input missing: {path.parent.name}/{path.name}')
    sch = pq.read_schema(path)
    for c, ty in want.items():
        need(c in sch.names, f'{path.name}: required column {c} missing')
        need(E._type_ok(sch.field(c).type, ty), f'{path.name}: column {c} is {sch.field(c).type}, expected {ty}')


def _rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(C.REPO))
    except ValueError:
        return str(p)


def _note(inputs, path, hash_cache):
    path = Path(path)
    st = path.stat()
    rec = dict(bytes=int(st.st_size), mtime_ns=int(st.st_mtime_ns))
    if path.is_symlink():
        rec['resolved'] = _rel(path.resolve())
    if hash_cache is not None:
        k = str(path.resolve())
        if k not in hash_cache:
            hash_cache[k] = E.sha256(path)
        rec['sha256'] = hash_cache[k]
    inputs[_rel(path)] = rec


def _groups(pf, preds):
    """Row groups whose footer statistics may hold a row satisfying every (column, lo, hi); a group without statistics is kept."""
    md = pf.metadata
    leaf = {md.schema.column(i).name: i for i in range(md.num_columns)}
    keep = []
    for r in range(md.num_row_groups):
        rg, ok = md.row_group(r), True
        for col, lo, hi in preds:
            st = rg.column(leaf[col]).statistics
            if st is None or not st.has_min_max:
                continue
            try:
                if st.max < lo or st.min > hi:
                    ok = False
                    break
            except TypeError:
                continue
        if ok:
            keep.append(r)
    return keep, md.num_row_groups


def _empty(spec):
    dt_ = {'str': object, 'int64': np.int64, 'float64': np.float64, 'bool': bool, 'num': np.float64}
    return pd.DataFrame({c: pd.Series(dtype=dt_[t]) for c, t in spec.items()})


class Market:
    """One market's saved public context (see the module API note). Attributes: day, mid, ws, we (venue ms), market_status, book, addrs
    (lowercase), btc (dict: BT = pf_mt_decision.load_btc output, code = dense grid status codes), tob (sorted arrays), depth (None off
    BOOK; t_all / fid_all / upos / B = pf_mt_lifecycle._book / best), legs[addr] (status-ok legs with windows, path order), excluded[addr]
    (non-ok legs), pos[addr] (position paths per clock), life_fills, orders (lifecycle tables, all signers), raw_tob, span, checks, counts,
    inputs, cost."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f'Market({self.day} {self.mid} book={self.book} addrs={len(self.addrs)})'

    def summary(self):
        D = self.depth
        bt = self.btc['BT']
        return dict(version=VERSION, day=self.day, mid=self.mid, ws_ms=self.ws, we_ms=self.we, market_status=self.market_status, book=self.book,
                    span_ms=list(self.span), btc=dict(grid_lo_ms=int(bt['G0'] * STEP_MS), grid_hi_ms=int((bt['G0'] + bt['n'] - 1) * STEP_MS),
                                                       n=int(bt['n']), span_codes=_span_codes(self)),
                    tob=dict(frames=int(self.tob['n']), codes={MARK[k]: int(v) for k, v in zip(*np.unique(self.tob['code'], return_counts=True))}
                             if self.tob['n'] else {}),
                    depth=None if D is None else dict(frames=int(D['n_all']), usable=int(D['B']['F']), excluded=D['B']['excluded'],
                                                       unknown_cells=int(D['B']['unknown_cells']), gaps=int(D['B']['n_gaps']),
                                                       span_ms=D['B']['span_ms']),
                    legs={a: dict(ok=int(len(self.legs[a])), maker=int((np.asarray(self.legs[a]['role'], object) == 'maker').sum()),
                                  taker=int((np.asarray(self.legs[a]['role'], object) == 'taker').sum()),
                                  excluded=dict(Counter(str(x) for x in np.asarray(self.excluded[a]['status'], object))),
                                  no_life_window=int((~self.legs[a]['has_life'].to_numpy(bool)).sum()) if len(self.legs[a]) else 0,
                                  lo_unbounded=int((self.legs[a]['lo_hard_ms'].to_numpy(np.int64) < 0).sum()) if len(self.legs[a]) else 0)
                          for a in self.addrs},
                    checks=self.checks, counts={k: int(v) for k, v in self.counts.items()}, cost=self.cost)


def _span_codes(M):
    X, bt = M.btc, M.btc['BT']
    a = max(0, M.span[0] // STEP_MS - bt['G0'])
    b = min(bt['n'], M.span[1] // STEP_MS - bt['G0'] + 1)
    c = X['code'][a:b] if b > a else np.zeros(0, np.int8)
    names = {0: 'ok', **BTC_ROW_NAME}
    out = {names[int(k)]: int(v) for k, v in zip(*np.unique(c, return_counts=True))}
    out['outside_grid'] = int((M.span[1] - M.span[0]) // STEP_MS + 1 - len(c))
    return out


# ------------------------------------------------------------------ population helpers ------------------------------------------------------------------
def default_root():
    need(C.RUN != 'v1', 'chain_context reads the v2 roots: run with MT_RUN=v2 (pf_mt_common.OUT -> data/predict_fun/research/maker_trace_v2_20261123)')
    return C.OUT


def top3(root=None):
    """Ranks 1-3 of econ/identities.json ex_ante.top10_shares; must equal pf_mt_decision.TOP3 (else refused)."""
    root = Path(default_root() if root is None else root)
    J = json.loads((root / 'econ' / 'identities.json').read_text())
    rows = sorted(J['ex_ante']['top10_shares'], key=lambda r: int(r['rank']))[:3]
    got = tuple(str(r['addr']) for r in rows)
    need([int(r['rank']) for r in rows] == [1, 2, 3] and tuple(a.lower() for a in got) == tuple(a.lower() for a in TOP3),
         f'identities.json ranks 1-3 {got} differ from pf_mt_decision.TOP3 (refused)')
    return got


def book_markets(path=None):
    """The 19 BOOK markets {day: [mid, ...]} (manifest 'complete'); with the default manifest exactly 19 on measurement days, else refused."""
    bm = E.load_book(BOOK_MANIFEST if path is None else Path(path))
    if path is None:
        need(sum(len(v) for v in bm.values()) == 19 and set(bm) <= set(LOAD_DAYS), 'BOOK manifest is not 19 markets on measurement days (refused)')
    return bm


def market_ws(M):
    return M.ws


def market_we(M):
    return M.we


# ------------------------------------------------------------------ loading ------------------------------------------------------------------
def _day_paths(root, day, mid):
    _guard(day, LOAD_DAYS, 'chain_context paths')
    root = Path(root)
    return dict(markets=root / f'markets_{day}.parquet', tob=root / f'tob_{day}.parquet', fv=root / 'econ' / 'fills_valued.parquet',
                frames=root / 'depth' / f'{day}_{mid}_frames.parquet', levels=root / 'depth' / f'{day}_{mid}_levels.parquet',
                life_fills=root / 'life' / f'{day}_{mid}_fills.parquet', life_orders=root / 'life' / f'{day}_{mid}_orders.parquet')


def _read_market_row(path, mid):
    _check_cols(path, dict(mid='int64', ws='num', we='num', status='str'))
    T = pq.read_table(path, columns=['mid', 'ws', 'we', 'status']).to_pandas()
    r = T[T['mid'].to_numpy(np.int64) == mid]
    need(len(r) == 1, f'{Path(path).name}: market {mid} has {len(r)} rows')
    return int(round(float(r['ws'].iat[0]) * 1000)), int(round(float(r['we'].iat[0]) * 1000)), str(r['status'].iat[0])


def _read_tob(path, mid, lo, hi, cnt):
    """tob rows of one market with upd_ms in [lo, hi], reading only row groups whose mid / upd_ms statistics allow it."""
    _check_cols(path, {c: C.SCHEMAS['tob'][c] for c in TOB_READ})
    pf = pq.ParquetFile(path)
    groups, ng = _groups(pf, [('mid', mid, mid), ('upd_ms', lo, hi)])
    cnt['tob_row_groups_read'] += len(groups)
    cnt['tob_row_groups_total'] += ng
    if not groups:
        return _empty({c: C.SCHEMAS['tob'][c] for c in TOB_READ})
    T = pf.read_row_groups(groups, columns=list(TOB_READ))
    cnt['tob_rows_scanned'] += T.num_rows
    u = T.column('upd_ms')
    m = pc.and_(pc.equal(T.column('mid'), mid), pc.and_(pc.greater_equal(u, lo), pc.less_equal(u, hi)))
    T = T.filter(pc.fill_null(m, False))
    cnt['tob_rows_kept'] += T.num_rows
    return T.to_pandas()


def _read_fv(path, day, mid, A, cnt):
    """fills_valued rows of the addresses in one market (every status), read through the row-group mid statistics; every row of the
    market must carry its day (contract section 1), else refused."""
    _check_cols(path, {c: E.FV[c] for c in FV_READ})
    pf = pq.ParquetFile(path)
    groups, ng = _groups(pf, [('mid', mid, mid)])
    cnt['fv_row_groups_read'] += len(groups)
    cnt['fv_row_groups_total'] += ng
    if not groups:
        return _empty({c: E.FV[c] for c in FV_READ})
    T = pf.read_row_groups(groups, columns=list(FV_READ))
    cnt['fv_rows_scanned'] += T.num_rows
    T = T.filter(pc.fill_null(pc.equal(T.column('mid'), mid), False))
    cnt['fv_rows_market'] += T.num_rows
    days = set(pc.unique(_str(T.column('day'))).to_pylist())
    need(days <= {day}, f'fills_valued: market {mid} has rows on day(s) other than its own {day} (refused)')
    low = pc.utf8_lower(_str(T.column('addr')))
    T = T.filter(pc.fill_null(pc.is_in(low, value_set=pa.array(list(A))), False))
    cnt['fv_rows_kept'] += T.num_rows
    return T.to_pandas()


def _btc_status_codes(btc_dir, day, BT):
    """Dense grid status codes aligned with BT (0 usable / 1 stale / 2 gap / 3 no_data / 4 no row), from the same files load_btc read;
    code == 0 must equal BT['ok'] exactly (load_btc's usable rule), else refused."""
    G0, n = int(BT['G0']), int(BT['n'])
    code = np.full(n, 4, np.int8)
    prev = DEC.prev_day(day)
    for dd in ((prev, day) if prev in DEC.BTC_DAYS else (day,)):
        p = DEC.btc_path(btc_dir, dd)
        if not p.is_file():
            continue
        T = pq.read_table(p, columns=['g_ms', 'mid', 'age_ms', 'status'])
        g = T.column('g_ms').to_numpy()
        idx = g // STEP_MS - G0
        k = (idx >= 0) & (idx < n)
        if not k.any():
            continue
        sc = pc.fill_null(pc.index_in(_str(T.column('status')), value_set=pa.array(list(BTC_FILE_ST))), 9).to_numpy().astype(np.int8)
        need(not bool((sc[k] == 9).any()), f'btc/{p.name}: a status outside {BTC_FILE_ST} (refused)')
        age = T.column('age_ms').to_numpy().astype(np.float64)
        mid = T.column('mid').to_numpy().astype(np.float64)
        okr = sc == 0
        sc[okr & (age > DEC.BTC_MAX_AGE_MS)] = 1
        sc[okr & ((age < 0) | ~np.isfinite(mid) | ~(mid > 0))] = 3
        code[idx[k]] = sc[k]
    need(np.array_equal(code == 0, np.asarray(BT['ok'], bool)), 'BTC grid status codes disagree with pf_mt_decision.load_btc usable rows (refused)')
    return code


def _btc_day(root, day, cnt, hash_cache):
    key = (str(Path(root).resolve()), day)
    if key in _BTC_CACHE:
        _BTC_CACHE.move_to_end(key)
        cnt['btc_cache_hit'] += 1
        return _BTC_CACHE[key]
    btc_dir = Path(root) / 'btc'
    c2 = Counter()
    BT = DEC.load_btc(btc_dir, day, c2)                     # unchanged builder (allow-list, padding, usable rule, overlap rule)
    prev = DEC.prev_day(day)
    files = [p for p in (DEC.btc_path(btc_dir, dd) for dd in ((prev, day) if prev in DEC.BTC_DAYS else (day,))) if p.is_file()]
    X = dict(BT=BT, code=_btc_status_codes(btc_dir, day, BT), day=day, load_counts=dict(c2), files=files, cum=None)
    _BTC_CACHE[key] = X
    while len(_BTC_CACHE) > 2:
        _BTC_CACHE.popitem(last=False)
    return X


def join_windows(fv, life):
    """fills_valued rows + the lifecycle window of their record leg by (sid, mk_idx) (all legs of one record share it; a taker row carries
    its maker leg's mk_idx): hi_ms, lo_hard_ms (= x_lo_hard_ms; -1 unbounded), lo_ms (evidence refinement, -1 none), win_status (the
    lifecycle fills status, or 'no_life_window' -> hi = settle_ms, lo_hard = settle_ms - MAX_LAG_MS), has_life."""
    fv = fv.reset_index(drop=True).copy()
    n = len(fv)
    hi, loh, lev = np.full(n, -1, np.int64), np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    wst, has = np.full(n, 'no_life_window', dtype=object), np.zeros(n, bool)
    if n and life is not None and len(life):
        need(not life.duplicated(['sid', 'mk_idx']).any(), 'lifecycle fills: duplicate (sid, mk_idx) key (refused)')
        kl = pd.MultiIndex.from_arrays([np.asarray(life['sid'], dtype=object), life['mk_idx'].to_numpy(np.int64)])
        kf = pd.MultiIndex.from_arrays([np.asarray(fv['sid'], dtype=object), fv['mk_idx'].to_numpy(np.int64)])
        pos = kl.get_indexer(kf)
        has = pos >= 0
        pp = pos[has]
        hi[has] = life['hi_ms'].to_numpy(np.int64)[pp]
        loh[has] = life['x_lo_hard_ms'].to_numpy(np.int64)[pp]
        lev[has] = life['lo_ms'].to_numpy(np.int64)[pp]
        wst[has] = np.asarray(life['status'], dtype=object)[pp]
    if n:
        se = fv['settle_ms'].to_numpy(np.int64)
        nh = ~has
        hi[nh] = se[nh]
        loh[nh] = np.where(se[nh] > 0, se[nh] - MAX_LAG_MS, -1)
    fv['hi_ms'], fv['lo_hard_ms'], fv['lo_ms'], fv['win_status'], fv['has_life'] = hi, loh, lev, wst, has
    return fv


def load_market(day, mid, addrs, root=None, *, book_manifest=None, hash_cache=None, guard_mib=C.GUARD_MIB, require_book=True):
    """Load ONE market's saved context (module API note). day must be a measurement day (refused before any path is built); off the BOOK
    list the call is refused unless require_book=False (then no depth / lifecycle: book() falls back to the tob econ MID rule and legs get
    settle-based windows). hash_cache: a dict shared across calls -> sha256 of every input once per run (fills_valued is ~1 GB)."""
    _guard(day, LOAD_DAYS, 'load_market')
    cpu0, w0 = time.process_time(), time.time()
    root = Path(default_root() if root is None else root)
    mid = int(mid)
    A = _addrs(addrs)
    P = _day_paths(root, day, mid)
    cnt, inputs = Counter(), {}
    ws, we, mstatus = _read_market_row(P['markets'], mid)
    _note(inputs, P['markets'], hash_cache)
    is_book = mid in book_markets(book_manifest).get(day, [])
    _note(inputs, BOOK_MANIFEST if book_manifest is None else Path(book_manifest), hash_cache)
    need(is_book or not require_book, f'market {day} {mid} is not a BOOK market (refused; require_book=False for a tob-only context)')
    frames = levels = life = orders = None
    if is_book:
        for k in ('frames', 'levels', 'life_fills', 'life_orders'):
            need(P[k].is_file(), f'BOOK market {mid}: missing {P[k].parent.name}/{P[k].name} (refused)')
        _check_cols(P['frames'], C.SCHEMAS['frames'])
        _check_cols(P['levels'], C.SCHEMAS['levels'])
        _check_cols(P['life_fills'], LIFE_REQ)
        _check_cols(P['life_orders'], ORDERS_REQ)
        frames = pq.read_table(P['frames'], columns=list(C.SCHEMAS['frames'])).to_pandas()
        levels = pq.read_table(P['levels'], columns=list(C.SCHEMAS['levels'])).to_pandas()
        life = pq.read_table(P['life_fills']).to_pandas()
        orders = pq.read_table(P['life_orders']).to_pandas()
        for k in ('frames', 'levels', 'life_fills', 'life_orders'):
            _note(inputs, P[k], hash_cache)
    if frames is not None and len(frames):
        lo_t, hi_t = int(frames['upd_ms'].min()), int(frames['upd_ms'].max())
    else:
        lo_t, hi_t = ws - TOB_PRE_NONBOOK_MS, we + PAD_MS
    tob = _read_tob(P['tob'], mid, lo_t, hi_t, cnt)
    _note(inputs, P['tob'], hash_cache)
    fv = _read_fv(P['fv'], day, mid, A, cnt)
    _note(inputs, P['fv'], hash_cache)
    legs = join_windows(fv, life)
    bt = _btc_day(root, day, cnt, hash_cache)
    for p in bt['files']:
        _note(inputs, p, hash_cache)
    M = build_market(day, mid, ws, we, A, btc=bt, market_status=mstatus, book=is_book, tob=tob, frames=frames, levels=levels, legs=legs,
                     life_fills=life, orders=orders, inputs=inputs, counts=cnt)
    M.cost = dict(cpu_s=round(time.process_time() - cpu0, 3), wall_s=round(time.time() - w0, 3), vmhwm_mib=round(LC.vmhwm_mib(), 1))
    LC.guard(f'chain_context.load_market {day} {mid}', guard_mib)
    return M


# ------------------------------------------------------------------ construction (pure) ------------------------------------------------------------------
def _tob_arrays(tob, mid):
    """tob frames sorted (upd_ms, recv_ns) with pf_mt_econ.frames' per-frame code (0 ok / 2 frame_status / 3 one_sided / 4 crossed)."""
    if tob is None or len(tob) == 0:
        z = np.zeros(0)
        return dict(n=0, t=np.zeros(0, np.int64), recv=np.zeros(0, np.int64), bb=z, ba=z, bs=z, as_=z, nb=np.zeros(0, np.int64),
                    na=np.zeros(0, np.int64), bad=np.zeros(0, bool), bpres=np.zeros(0, bool), apres=np.zeros(0, bool),
                    crossed=np.zeros(0, bool), code=np.zeros(0, np.int8), status=np.zeros(0, object))
    need(bool((tob['mid'].to_numpy(np.int64) == mid).all()), f'tob rows of a market other than {mid}')
    t, rv = tob['upd_ms'].to_numpy(np.int64), tob['recv_ns'].to_numpy(np.int64)
    need(t.min() >= 0 and t.max() < 1 << 42, 'tob: upd_ms outside the epoch-ms range')
    o = np.lexsort((rv, t))
    bb, ba = tob['best_bid'].to_numpy(float)[o], tob['best_ask'].to_numpy(float)[o]
    nb, na = tob['n_bid'].to_numpy(np.int64)[o], tob['n_ask'].to_numpy(np.int64)[o]
    bad = E.flagged_any(tob['status'], E.BAD_FRAME)[o]
    bpres, apres = (nb > 0) & np.isfinite(bb), (na > 0) & np.isfinite(ba)
    code = np.zeros(len(t), np.int8)
    code[bad] = 2
    code[(code == 0) & ~(bpres & apres)] = 3
    code[(code == 0) & (bb >= ba)] = 4
    return dict(n=len(t), t=t[o], recv=rv[o], bb=bb, ba=ba, bs=tob['best_bid_sh'].to_numpy(float)[o], as_=tob['best_ask_sh'].to_numpy(float)[o],
                nb=nb, na=na, bad=bad, bpres=bpres, apres=apres, crossed=~bad & bpres & apres & (bb >= ba), code=code,
                status=np.asarray(tob['status'], dtype=object)[o])


def _depth_best(B):
    """Per usable frame: best bid / ask tick (float; -inf / +inf empty side; NaN when an unknown cell lies above the best known bid / below
    the best known ask) and the shares there (0 for an empty side)."""
    F = B['F']
    if F == 0:
        z = np.zeros(0)
        return dict(bid=z, ask=z, bsz=z, asz=z)
    S, out = B['shown'], {}
    for side, nm in ((0, 'bid'), (1, 'ask')):
        X = S[:, side, :]
        unk = np.isnan(X)
        pos = np.nan_to_num(X, nan=0.0) > 0
        anyp, anyu = pos.any(1), unk.any(1)
        if side == 0:
            kb = np.where(anyp, NT - 1 - np.argmax(pos[:, ::-1], 1), -1)
            ku = np.where(anyu, NT - 1 - np.argmax(unk[:, ::-1], 1), -1)
            unknown = ku > kb
            tick = np.where(unknown, np.nan, np.where(kb >= 0, kb.astype(float), -np.inf))
        else:
            kb = np.where(anyp, np.argmax(pos, 1), NT)
            ku = np.where(anyu, np.argmax(unk, 1), NT)
            unknown = ku < kb
            tick = np.where(unknown, np.nan, np.where(kb < NT, kb.astype(float), np.inf))
        sz = X[np.arange(F), np.clip(kb, 0, NT - 1)]
        out[nm] = tick
        out[nm[0] + 'sz'] = np.where(np.isfinite(tick), sz, np.where(np.isinf(tick), 0.0, np.nan))
    return out


def _depth(frames, levels, mid, ws, we):
    fr = frames.reset_index(drop=True)
    lv = (levels if levels is not None else _empty(C.SCHEMAS['levels'])).reset_index(drop=True)
    B = LC._book(fr, lv, mid, ws, we)                      # unchanged lifecycle semantics
    t_all, f_all, r_all = fr['upd_ms'].to_numpy(np.int64), fr['fid'].to_numpy(np.int64), fr['recv_ns'].to_numpy(np.int64)
    o = np.lexsort((f_all, t_all))
    t_all, f_all, r_all = t_all[o], f_all[o], r_all[o]
    upos = pd.Index(B['fid']).get_indexer(f_all) if B['F'] else np.full(len(f_all), -1)
    return dict(n_all=len(t_all), t_all=t_all, fid_all=f_all, recv_all=r_all, upos=np.asarray(upos, np.int64), B=B, best=_depth_best(B))


def _tob_depth_checks(TB, DP):
    """The tob rows and the depth frames are the same WS frames (pf_mt_extract writes both per frame): equal (upd_ms, recv_ns) multisets,
    else refused. Diagnostic: tob best ticks against the usable depth frame's best (unknown cells skipped)."""
    o = np.lexsort((DP['recv_all'], DP['t_all']))
    need(DP['n_all'] == TB['n'] and np.array_equal(DP['t_all'][o], TB['t']) and np.array_equal(DP['recv_all'][o], TB['recv']),
         f'tob frames ({TB["n"]}) and depth frames ({DP["n_all"]}) of the market differ (refused: mismatched artifacts)')
    u = DP['upos'][o]
    ok = u >= 0
    out = dict(tob_depth_frames_equal=True, depth_usable=int(ok.sum()), depth_unusable=int((~ok).sum()))
    for nm, px, pres in (('bid', TB['bb'], TB['bpres']), ('ask', TB['ba'], TB['apres'])):
        tt = np.where(pres, np.rint(px * 100), -np.inf if nm == 'bid' else np.inf)[ok]
        dd = DP['best'][nm][u[ok]] if ok.any() else np.zeros(0)
        cmp = ~np.isnan(dd) & ~TB['bad'][ok]
        out[f'best_{nm}_compared'] = int(cmp.sum())
        out[f'best_{nm}_mismatch'] = int((cmp & (tt != dd)).sum())
    return out


def _paths(s, q, hi, lo, se, xlo):
    sq = s * q
    lo_eff = np.where(lo >= 0, lo, I64MIN)
    P = {}
    for clock, tau in (('hi', hi), ('lo', lo_eff), ('settle', se)):
        o = np.argsort(tau, kind='stable')
        P[clock] = dict(tau=tau[o], cum=np.r_[0., np.cumsum(sq[o])], buy=np.r_[0., np.cumsum(np.where(s[o] > 0, q[o], 0.))],
                        sell=np.r_[0., np.cumsum(np.where(s[o] < 0, q[o], 0.))], order=o)
    P['unb_hi'] = np.sort(hi[lo < 0])
    P['ex_lo'] = np.sort(np.where(xlo >= 0, xlo, I64MIN))
    return P


def _legs(legs, A, cnt, checks):
    LEG, EX, POS = {}, {}, {}
    if legs is None:
        legs = join_windows(_empty({c: E.FV[c] for c in FV_READ}), None)
    low = np.array([str(x).lower() for x in np.asarray(legs['addr'], dtype=object)], dtype=object) if len(legs) else np.zeros(0, object)
    st = np.asarray(legs['status'], dtype=object) if len(legs) else np.zeros(0, object)
    for a in A:
        sel = low == a
        sp = set(str(x) for x in np.asarray(legs['addr'], dtype=object)[sel]) if sel.any() else set()
        need(len(sp) <= 1, f'address {a} appears under two spellings {sorted(sp)} (refused)')
        La = legs[sel]
        okm = st[sel] == 'ok'
        G = La[okm].sort_values(['hi_ms', 'sid', 'px_rank', 'mk_idx', 'role'], kind='mergesort').reset_index(drop=True)
        X = La[~okm].reset_index(drop=True)
        s, q = G['s'].to_numpy(np.int64), G['q'].to_numpy(np.float64)
        hi, lo, se = G['hi_ms'].to_numpy(np.int64), G['lo_hard_ms'].to_numpy(np.int64), G['settle_ms'].to_numpy(np.int64)
        need(bool(np.isin(s, (-1, 1)).all()) and bool(np.isfinite(q).all()) and bool((q > 0).all()),
             f'{a}: a status-ok leg with s outside +-1 or a non-positive / non-finite q (refused)')
        need(bool((hi > 0).all()), f'{a}: a status-ok leg without a positive hi_ms (refused)')
        need(bool(((lo < 0) | (lo <= hi)).all()), f'{a}: a fill window with x_lo_hard_ms > hi_ms (refused)')
        cnt[f'legs_ok:{a}'] += len(G)
        cnt[f'legs_excluded:{a}'] += len(X)
        cnt[f'legs_no_life_window:{a}'] += int((~G['has_life'].to_numpy(bool)).sum())
        cnt[f'legs_lo_unbounded:{a}'] += int((lo < 0).sum())
        if len(G):
            checks[f'hi_after_settle:{a}'] = int((hi > se).sum())
        LEG[a], EX[a] = G, X
        POS[a] = _paths(s, q, hi, lo, se, X['lo_hard_ms'].to_numpy(np.int64) if len(X) else np.zeros(0, np.int64))
    return LEG, EX, POS


def market_span(ws, we, TB=None, DP=None, legs=None):
    """The market's evaluation span [min(ws - 300 s, first frame, first own window start) - 60 s, max(we, last frame, last own hi) + 60 s]."""
    lo, hi = ws - PRE_MS, we
    if DP is not None and DP['B']['F']:
        lo, hi = min(lo, int(DP['B']['t'][0])), max(hi, int(DP['B']['t'][-1]))
    elif TB is not None and TB['n']:
        lo, hi = min(lo, int(TB['t'][0])), max(hi, int(TB['t'][-1]))
    for G in (legs or {}).values():
        if len(G):
            h, l_ = G['hi_ms'].to_numpy(np.int64), G['lo_hard_ms'].to_numpy(np.int64)
            lo, hi = min(lo, int(np.where(l_ >= 0, l_, h).min())), max(hi, int(h.max()))
    return lo - PAD_MS, hi + PAD_MS


def build_market(day, mid, ws_ms, we_ms, addrs, *, btc, market_status='ok', book=False, tob=None, frames=None, levels=None, legs=None,
                 life_fills=None, orders=None, inputs=None, counts=None):
    """Pure constructor (load_market's second half). btc = the loaded BTC day dict (BT = pf_mt_decision.load_btc output, code = dense grid
    status codes); legs = join_windows(fills_valued rows, lifecycle fills)."""
    cnt = Counter(counts or {})
    mid, ws, we = int(mid), int(ws_ms), int(we_ms)
    need(we > ws, 'we must be after ws')
    need(isinstance(btc, dict) and 'BT' in btc and 'code' in btc and len(btc['code']) == btc['BT']['n'], 'btc: a loaded BTC day dict is required')
    A = _addrs(addrs)
    TB = _tob_arrays(tob, mid)
    DP, checks = None, {}
    if frames is not None:
        DP = _depth(frames, levels, mid, ws, we)
        checks.update(_tob_depth_checks(TB, DP))
    LEG, EX, POS = _legs(legs, A, cnt, checks)
    span = market_span(ws, we, TB, DP, LEG)
    bt = btc['BT']
    checks['btc_grid_covers_span'] = bool(bt['n'] > 0 and bt['G0'] * STEP_MS <= span[0] and (bt['G0'] + bt['n'] - 1) * STEP_MS >= span[1] - PAD_MS)
    return Market(version=VERSION, day=day, mid=mid, ws=ws, we=we, market_status=str(market_status), book=bool(book), addrs=A, btc=btc, span=span,
                  tob=TB, depth=DP, legs=LEG, excluded=EX, pos=POS, life_fills=life_fills, orders=orders, raw_tob=tob, checks=checks,
                  counts=cnt, inputs=dict(inputs or {}), cost={})


# ------------------------------------------------------------------ BTC ------------------------------------------------------------------
def _btc_cum(X):
    if X['cum'] is None:
        X['cum'] = {c: np.r_[0, np.cumsum(X['code'] == c)].astype(np.int64) for c in (1, 2, 3, 4)}
    return X['cum']


def btc_ret(M, t, horizons=HORIZONS):
    """pf_mt_decision.btc_feats UNCHANGED at t (grid row i = floor(t / 100) <= t; every grid point of [i - 600, i] and ws when t >= ws must be
    usable, else btc_invalid; outside the loaded grid btc_missing; t >= we after_we): r_1s / r_5s / r_30s (also r_1000 / r_5000 / r_30000),
    r_ws, sig, z, p_fair, mid, tau (s), code, status, ok. Another horizon h (ms) -> r_<h> = 1e4 ln(mid_i / mid_j), j = floor((t - h) / 100),
    NaN unless ok and every grid point of [j, i] is usable. cause: why a code is not ok (the worst grid row status in the window: missing >
    no_data > gap > stale; ws_<status>; outside_grid; after_we). Values are NaN unless ok."""
    t = _t(t)
    X = M.btc
    BT = X['BT']
    F = DEC.btc_feats(BT, t, M.ws, M.we)
    code = np.asarray(F['code'], np.int8)
    ok = code == 0
    out = dict(t=t, code=code, status=BTC_CODE[code], ok=ok, r_ws=F['r_ws'], sig=F['sig'], z=F['z'], p_fair=F['pf'], mid=F['mid'], tau=F['tau'])
    for nm, k in DEC.R_STEPS:
        out[nm] = F[nm]
        out[f'r_{k * STEP_MS}'] = F[nm]
    for h in horizons:
        h = int(h)
        need(h > 0, 'horizons must be positive ms')
        if f'r_{h}' in out:
            continue
        i = t // STEP_MS - BT['G0']
        j = (t - h) // STEP_MS - BT['G0']
        r = np.full(len(t), np.nan)
        sel = ok & (j >= 0)
        if sel.any():
            ii, jj = i[sel], j[sel]
            full = (BT['okc'][ii + 1] - BT['okc'][jj]) == (ii - jj + 1)
            v = 1e4 * (BT['lm'][ii] - BT['lm'][jj])
            r[np.flatnonzero(sel)[full]] = v[full]
        out[f'r_{h}'] = r
    out['cause'] = _btc_cause(M, t, code)
    out['counts'] = dict(status=dict(Counter(out['status'].tolist())), cause=dict(Counter(out['cause'].tolist())))
    return out


def _btc_cause(M, t, code):
    X, BT = M.btc, M.btc['BT']
    cause = np.full(len(t), 'ok', dtype=object)
    cause[code == 1] = 'outside_grid'
    cause[code == 3] = 'after_we'
    sel = np.flatnonzero(code == 2)
    if len(sel):
        cum = _btc_cum(X)
        i = t[sel] // STEP_MS - BT['G0']
        a = i - DEC.SIG_STEPS
        worst = np.zeros(len(sel), np.int8)
        for c in (1, 2, 3, 4):
            worst[(cum[c][i + 1] - cum[c][a]) > 0] = c
        nm = np.array(['', 'stale', 'gap', 'no_data', 'missing'], dtype=object)[worst]
        w0 = worst == 0
        if w0.any():
            iw = M.ws // STEP_MS - BT['G0']
            wc = int(X['code'][iw]) if 0 <= iw < BT['n'] else 4
            nm[w0] = 'ws_' + BTC_ROW_NAME.get(wc, 'ok')
        cause[sel] = nm
    return cause


def btc_state(M, t):
    """The grid row floor(t / 100) <= t: mid (NaN unless usable) and its status ok / stale / gap / no_data / missing (a single-row status,
    not btc_feats' 60 s window rule)."""
    t = _t(t)
    X, BT = M.btc, M.btc['BT']
    i = t // STEP_MS - BT['G0']
    inr = (i >= 0) & (i < BT['n'])
    ic = np.clip(i, 0, max(BT['n'] - 1, 0))
    c = np.where(inr, X['code'][ic] if BT['n'] else 4, 4).astype(np.int8)
    mid = np.where(c == 0, np.exp(BT['lm'][ic]) if BT['n'] else np.nan, np.nan)
    return dict(t=t, mid=mid, status=np.array(['ok', 'stale', 'gap', 'no_data', 'missing'], dtype=object)[c], ok=c == 0)


# ------------------------------------------------------------------ book ------------------------------------------------------------------
def book(M, t, strict=True):
    """Contract 7.0: the last USABLE depth frame (pf_mt_lifecycle._book) with upd_ms <= t. bid_tick / ask_tick: float cents (-inf / +inf =
    empty side; NaN = unknown: an unknown cell above the best known bid / below the best known ask, a stale frame or none); bid_sz /
    ask_sz (0 for an empty side); spread (ticks; inf with an empty side), mid_c (cents) and mid (price) when both touches are levels;
    age_ms (-1 none), stale (age > 1,000 ms), frame (usable frame index), upd_ms. Per side bid_st / ask_st: no_frame / stale / unknown /
    empty / ok. status: no_frame / stale / unknown_touch / empty / one_sided / crossed / ok; ok = fresh, both touches known (a level or
    empty), not crossed. strict=False keeps a stale frame's values (status still stale). Off BOOK: the tob econ MID rule (source 'tob')."""
    t = _t(t)
    if M.depth is None:
        return _book_from_tob(M, t, strict)
    n, D = len(t), M.depth
    F, tt = D['B']['F'], D['B']['t']
    j = np.searchsorted(tt, t, 'right') - 1 if F else np.full(n, -1, np.int64)
    has = j >= 0
    jc = np.maximum(j, 0)
    age = np.where(has, t - (tt[jc] if F else 0), -1)
    stale = has & (age > BOOK_STALE_MS)
    use = (has & ~stale) if strict else has
    out = dict(t=t, age_ms=age, stale=stale, frame=np.where(has, j, -1), upd_ms=np.where(has, tt[jc] if F else -1, -1), source='depth')
    raw = {}
    for nm in ('bid', 'ask'):
        r = D['best'][nm][jc] if F else np.full(n, np.nan)
        sz = D['best'][nm[0] + 'sz'][jc] if F else np.full(n, np.nan)
        raw[nm] = np.where(has, r, np.nan)
        out[f'{nm}_tick'] = np.where(use, r, np.nan)
        out[f'{nm}_sz'] = np.where(use, sz, np.nan)
        out[f'{nm}_st'] = _sel([~has, stale, np.isnan(raw[nm]), np.isinf(raw[nm])], ['no_frame', 'stale', 'unknown', 'empty'], 'ok', n)
    return _book_finish(out, raw, has, stale, n)


def _book_finish(out, raw, has, stale, n):
    bst, ast = out['bid_st'], out['ask_st']
    crossed = np.isfinite(raw['bid']) & np.isfinite(raw['ask']) & (raw['bid'] >= raw['ask'])
    out['status'] = _sel([~has, stale, (bst == 'unknown') | (ast == 'unknown'), (bst == 'empty') & (ast == 'empty'),
                          (bst == 'empty') | (ast == 'empty'), crossed],
                         ['no_frame', 'stale', 'unknown_touch', 'empty', 'one_sided', 'crossed'], 'ok', n)
    out['ok'] = np.isin(out['status'], ['ok', 'one_sided', 'empty'])
    b, a = out['bid_tick'], out['ask_tick']
    with np.errstate(invalid='ignore'):
        out['spread'] = a - b
    out['spread_ticks'] = out['spread']
    both = np.isfinite(b) & np.isfinite(a)
    out['mid_c'] = np.where(both, (b + a) / 2, np.nan)
    out['mid'] = out['mid_c'] / 100.
    return out


def _book_from_tob(M, t, strict):
    """Off BOOK: the last tob frame (upd_ms, recv_ns order) with upd_ms <= t, pf_mt_econ's flags (a BAD_FRAME frame -> both sides unknown)."""
    n, T = len(t), M.tob
    j = np.searchsorted(T['t'], t, 'right') - 1 if T['n'] else np.full(n, -1, np.int64)
    has = (j >= 0) & (t >= 0)
    jc = np.maximum(j, 0)
    age = np.where(has, t - (T['t'][jc] if T['n'] else 0), -1)
    stale = has & (age > BOOK_STALE_MS)
    use = (has & ~stale) if strict else has
    bad = has & (T['bad'][jc] if T['n'] else False)
    out = dict(t=t, age_ms=age, stale=stale, frame=np.where(has, j, -1), upd_ms=np.where(has, T['t'][jc] if T['n'] else -1, -1), source='tob')
    raw = {}
    for nm, px, sz, pres, inf in (('bid', 'bb', 'bs', 'bpres', -np.inf), ('ask', 'ba', 'as_', 'apres', np.inf)):
        if T['n']:
            r = np.where(T[pres][jc], np.rint(T[px][jc] * 100), inf)
            s_ = np.where(T[pres][jc], T[sz][jc], 0.0)
        else:
            r = s_ = np.full(n, np.nan)
        r = np.where(bad, np.nan, r)
        raw[nm] = np.where(has, r, np.nan)
        out[f'{nm}_tick'] = np.where(use, r, np.nan)
        out[f'{nm}_sz'] = np.where(use & ~bad, s_, np.nan)
        out[f'{nm}_st'] = _sel([~has, stale, np.isnan(raw[nm]), np.isinf(raw[nm])], ['no_frame', 'stale', 'unknown', 'empty'], 'ok', n)
    return _book_finish(out, raw, has, stale, n)


def book_tob(M, t):
    """pf_mt_econ MID rule exactly (E.frames / E.mid_at): the last tob frame (upd_ms, recv_ns) with upd_ms <= t; status no_frame /
    frame_status / one_sided / crossed / stale (> 1,000 ms) / ok; mid NaN unless ok; best prices / sizes of the frame unless it is flagged."""
    t = _t(t)
    n, T = len(t), M.tob
    j = np.searchsorted(T['t'], t, 'right') - 1 if T['n'] else np.full(n, -1, np.int64)
    has = (j >= 0) & (t >= 0)
    jc = np.maximum(j, 0)
    age = np.where(has, t - (T['t'][jc] if T['n'] else 0), -1)
    code = np.where(has, T['code'][jc] if T['n'] else 1, 1).astype(np.int8)
    code[(code == 0) & (age > BOOK_STALE_MS)] = 5
    out = dict(t=t, code=code, status=MARK[code], age_ms=age, frame=np.where(has, j, -1))
    if T['n']:
        good = has & ~T['bad'][jc]
        out['mid'] = np.where(code == 0, (T['bb'][jc] + T['ba'][jc]) / 2, np.nan)
        for k, src in (('best_bid', 'bb'), ('best_ask', 'ba'), ('bid_sz', 'bs'), ('ask_sz', 'as_')):
            out[k] = np.where(good, T[src][jc], np.nan)
    else:
        for k in ('mid', 'best_bid', 'best_ask', 'bid_sz', 'ask_sz'):
            out[k] = np.full(n, np.nan)
    return out


def frame_times(M):
    """upd_ms of the usable depth frames in (upd_ms, fid) order (index = the frame of book() / level_at_frames); off BOOK the tob frames."""
    return (M.depth['B']['t'] if M.depth is not None else M.tob['t']).copy()


def level_at_frames(M, f, side, tick):
    """Shown shares at (side, tick) in usable depth frames f (indices into frame_times); NaN = unknown cell / no such frame."""
    f = _t(f)
    n = len(f)
    if M.depth is None or not M.depth['B']['F']:
        return np.full(n, np.nan)
    B = M.depth['B']
    sd, tk = _side(side, n), _ints(tick, n)
    okf = (f >= 0) & (f < B['F']) & (tk >= 0) & (tk < NT)
    v = B['shown'][np.clip(f, 0, B['F'] - 1), sd, np.clip(tk, 0, NT - 1)]
    return np.where(okf, v, np.nan)


def level_size(M, t, side, tick, strict=True):
    """Depth at (side, tick) in the frame book() uses (the last usable frame <= t): size (shares; 0 = absent from a usable snapshot), status
    no_depth / no_frame / bad_tick / stale / unknown_cell / ok, age_ms, frame. NaN unless ok (strict=False keeps a stale frame's value)."""
    t = _t(t)
    n = len(t)
    sd, tk = _side(side, n), _ints(tick, n)
    if M.depth is None:
        return dict(size=np.full(n, np.nan), status=np.full(n, 'no_depth', dtype=object), age_ms=np.full(n, -1, np.int64), frame=np.full(n, -1, np.int64))
    B = M.depth['B']
    F = B['F']
    j = np.searchsorted(B['t'], t, 'right') - 1 if F else np.full(n, -1, np.int64)
    has = j >= 0
    jc = np.maximum(j, 0)
    age = np.where(has, t - (B['t'][jc] if F else 0), -1)
    tok = (tk >= 0) & (tk < NT)
    raw = B['shown'][jc, sd, np.clip(tk, 0, NT - 1)] if F else np.full(n, np.nan)
    stale = has & (age > DEPTH_GAP_MS)
    st = _sel([~has, ~tok, stale, np.isnan(raw)], ['no_frame', 'bad_tick', 'stale', 'unknown_cell'], 'ok', n)
    use = (st == 'ok') if strict else (has & tok & ~np.isnan(raw))
    return dict(size=np.where(use, raw, np.nan), status=st, age_ms=age, frame=np.where(has, j, -1))


def level_series(M, side, tick, lo_ms, hi_ms):
    """Every usable depth frame from the one in force at lo through hi: upd_ms, fid, frame, in_force_at_lo, size (NaN unknown cell), status,
    dt_prev_ms (to the previous usable frame; -1 first), gap_before (> 1,000 ms: an unknown span, lifecycle rule). Reads frames <= hi only."""
    cols = ['upd_ms', 'fid', 'frame', 'in_force_at_lo', 'size', 'status', 'dt_prev_ms', 'gap_before']
    if M.depth is None or not M.depth['B']['F']:
        return pd.DataFrame({c: [] for c in cols})
    B = M.depth['B']
    lo, hi, tk = int(lo_ms), int(hi_ms), int(tick)
    need(hi >= lo, 'level_series: hi < lo')
    sd = int(_side(side, 1)[0])
    j0 = max(int(np.searchsorted(B['t'], lo, 'right')) - 1, 0)
    j1 = int(np.searchsorted(B['t'], hi, 'right')) - 1
    if j1 < j0:
        return pd.DataFrame({c: [] for c in cols})
    J = np.arange(j0, j1 + 1)
    tok = 0 <= tk < NT
    v = B['shown'][J, sd, min(max(tk, 0), NT - 1)] if tok else np.full(len(J), np.nan)
    st = np.where(not tok, 'bad_tick', np.where(np.isnan(v), 'unknown_cell', 'ok')).astype(object)
    dtp = np.where(J > 0, B['t'][J] - B['t'][np.maximum(J - 1, 0)], -1)
    return pd.DataFrame(dict(upd_ms=B['t'][J], fid=B['fid'][J], frame=J, in_force_at_lo=B['t'][J] < lo, size=np.where(st == 'ok', v, np.nan),
                             status=st, dt_prev_ms=dtp, gap_before=dtp > DEPTH_GAP_MS))


def touch_distance(M, t, side, tick, strict=True):
    """Ticks from the same-side touch of book(): bid d = best_bid - tick, ask d = tick - best_ask (+k behind, 0 at, -k inside; an empty own
    side gives -inf = inside); cat at / inside / behind / unknown; status = the side's status (book bid_st / ask_st); book_status beside."""
    t = _t(t)
    n = len(t)
    sd, tk = _side(side, n), _ints(tick, n).astype(float)
    b = book(M, t, strict)
    best = np.where(sd == 0, b['bid_tick'], b['ask_tick'])
    with np.errstate(invalid='ignore'):
        d = np.where(sd == 0, best - tk, tk - best)
    cat = _sel([np.isnan(d), d == 0, d > 0], ['unknown', 'at', 'behind'], 'inside', n)
    return dict(t=t, ticks=d, cat=cat, status=np.where(sd == 0, b['bid_st'], b['ask_st']).astype(object), book_status=b['status'], best_tick=best)


def book_change(M, t, side, lag_ms=BOOK_MOVE_MS):
    """Contract 7.0 book move over [t - lag, t]: d_mid_c (cents), d_opp (the opposite touch's change, ticks; bid order -> the ask, ask order
    -> the bid), mid_cat moved / flat / unknown (|d_mid_c| >= 1 tick), opp_cat toward / away / flat / unknown (toward = the opposite touch
    moved >= 1 tick toward the order's side), ok (both books ok)."""
    t = _t(t)
    n = len(t)
    sd = _side(side, n)
    b1, b0 = book(M, t), book(M, t - int(lag_ms))
    with np.errstate(invalid='ignore'):
        dm = b1['mid_c'] - b0['mid_c']
        opp1 = np.where(sd == 0, b1['ask_tick'], b1['bid_tick'])
        opp0 = np.where(sd == 0, b0['ask_tick'], b0['bid_tick'])
        dop = opp1 - opp0
        tw = np.where(sd == 0, -dop, dop)                    # > 0 = toward the order's side
    fin = np.isfinite(dop)
    return dict(t=t, d_mid_c=dm, d_opp=np.where(fin, dop, np.nan), ok=b1['ok'] & b0['ok'],
                mid_cat=_sel([~np.isfinite(dm), np.abs(dm) >= BOOK_MOVE_TICKS], ['unknown', 'moved'], 'flat', n),
                opp_cat=_sel([~fin, tw >= BOOK_MOVE_TICKS, tw <= -BOOK_MOVE_TICKS], ['unknown', 'toward', 'away'], 'flat', n))


# ------------------------------------------------------------------ inventory ------------------------------------------------------------------
def _pos(M, addr):
    a = _akey(addr)
    need(a in M.pos, f'address {addr} was not loaded into this market (pass it to load_market)')
    return a, M.pos[a]


def position(M, addr, t, clock='hi'):
    """RECORDED venue YES position (signed shares, fees excluded; self-trade rows cancel) from the address's status-ok legs strictly
    before t: recorded = settle_ms < t (contract 7.0), certain = hi_ms < t, possible = x_lo_hard_ms < t (unbounded lo counts), pending =
    possible - certain (legs with x_lo_hard_ms < t <= hi_ms); pos / n / last_ms on the declared clock ('hi' | 'lo' | 'settle'; last_ms -1
    none or unbounded); status 'excluded_leg_possible' once a non-ok leg's window has started (its fill is not in the path), else ok.
    RECORDED vs CLIENT-KNOWN: these are reconstructions of the venue inventory; what the maker's client knew at t also depends on private
    fill-notice delays (unidentified, not our collector lag). A declared notice delay D is a sensitivity: position(M, addr, t - D)."""
    need(clock in CLOCKS, f'clock must be one of {CLOCKS}')
    a, P = _pos(M, addr)
    t = _t(t)
    out, J = dict(t=t), {}
    for c, nm in (('settle', 'recorded'), ('hi', 'certain'), ('lo', 'possible')):
        J[c] = np.searchsorted(P[c]['tau'], t, 'left')
        out[nm] = P[c]['cum'][J[c]]
    out['pending'] = np.where(J['lo'] > J['hi'], out['possible'] - out['certain'], 0.0)
    j, tau = J[clock], P[clock]['tau']
    out['pos'], out['n'] = P[clock]['cum'][j], j
    last = np.where(j > 0, tau[np.maximum(j - 1, 0)], -1) if len(tau) else np.full(len(t), -1, np.int64)
    out['last_ms'] = np.where(last == I64MIN, -1, last)
    out['status'] = np.where(np.searchsorted(P['ex_lo'], t, 'left') > 0, 'excluded_leg_possible', 'ok').astype(object)
    return out


def pending(M, addr, t):
    """Own status-ok legs whose window [x_lo_hard_ms, hi_ms] holds t with x_lo_hard_ms < t <= hi_ms (matched by t or not yet: uncertain):
    n, buy_q, sell_q (shares), net (= position()['pending']), certain (hi_ms < t), pos_min = certain - sell_q, pos_max = certain + buy_q
    (the feasible recorded position at t), n_lo_unbounded (pending legs without a lower end). Reconstruction-side: built from records
    created after t; a maker could not have known them at t."""
    a, P = _pos(M, addr)
    t = _t(t)
    jh, jl = np.searchsorted(P['hi']['tau'], t, 'left'), np.searchsorted(P['lo']['tau'], t, 'left')
    n = jl - jh
    buy = np.where(n > 0, P['lo']['buy'][jl] - P['hi']['buy'][jh], 0.0)
    sell = np.where(n > 0, P['lo']['sell'][jl] - P['hi']['sell'][jh], 0.0)
    cert = P['hi']['cum'][jh]
    net = np.where(n > 0, P['lo']['cum'][jl] - cert, 0.0)
    return dict(t=t, n=n, buy_q=buy, sell_q=sell, net=net, certain=cert, pos_min=cert - sell, pos_max=cert + buy,
                n_lo_unbounded=len(P['unb_hi']) - np.searchsorted(P['unb_hi'], t, 'left'))


def pending_legs(M, addr, t0):
    """The legs (rows of M.legs[addr]) pending at one instant t0 (x_lo_hard_ms < t0 <= hi_ms)."""
    a, _ = _pos(M, addr)
    G = M.legs[a]
    t0 = int(_t(t0)[0])
    lo, hi = G['lo_hard_ms'].to_numpy(np.int64), G['hi_ms'].to_numpy(np.int64)
    return G[(np.where(lo >= 0, lo, I64MIN) < t0) & (t0 <= hi)].reset_index(drop=True)


def position_path(M, addr, clock='hi'):
    """The recorded YES-position path on a clock: one row per status-ok leg in clock order (t = its clock time; -1 = unbounded lo)."""
    need(clock in CLOCKS, f'clock must be one of {CLOCKS}')
    a, P = _pos(M, addr)
    G, p = M.legs[a], P[clock]
    o = p['order']
    tau = np.where(p['tau'] == I64MIN, -1, p['tau'])
    return pd.DataFrame(dict(t=tau, sid=np.asarray(G['sid'], dtype=object)[o], role=np.asarray(G['role'], dtype=object)[o],
                             s=G['s'].to_numpy(np.int64)[o], q=G['q'].to_numpy(float)[o], pos=p['cum'][1:]))


def near_fill(M, addr, t, before_ms=0, after_ms=0, role=None, clock='hi'):
    """POPULATION mask (uses later fills when after_ms > 0; never a feature): does an own status-ok leg (role filter) have its clock time
    in [t - before_ms, t + after_ms]?"""
    need(clock in ('hi', 'settle'), "near_fill clock must be 'hi' or 'settle'")
    a, _ = _pos(M, addr)
    G = M.legs[a]
    t = _t(t)
    sel = np.ones(len(G), bool) if role is None else (np.asarray(G['role'], dtype=object) == role)
    tau = np.sort(G['hi_ms' if clock == 'hi' else 'settle_ms'].to_numpy(np.int64)[sel])
    return np.searchsorted(tau, t + int(after_ms), 'right') > np.searchsorted(tau, t - int(before_ms), 'left')


def resting(M, mk_hash, t, scenario='S_NOPRIO'):
    """POPULATION mask (built from the order's later fills; never a feature): the lifecycle's certain presence span persist_lo_ms <= t <=
    persist_hi_ms ([first hi, last x_lo_hard], equal in every scenario row) -> (bool array, status ok / no_orders_table / no_order /
    ambiguous_order / no_persistence)."""
    t = _t(t)
    z = np.zeros(len(t), bool)
    O = M.orders
    if O is None:
        return z, 'no_orders_table'
    r = O[(np.asarray(O['scenario'], dtype=object) == scenario) & (np.asarray(O['mk_hash'], dtype=object) == mk_hash)]
    if len(r) != 1:
        return z, ('no_order' if not len(r) else 'ambiguous_order')
    lo, hi = int(r['persist_lo_ms'].iat[0]), int(r['persist_hi_ms'].iat[0])
    if lo < 0 or hi < lo:
        return z, 'no_persistence'
    return (t >= lo) & (t <= hi), 'ok'


# ------------------------------------------------------------------ time, phase ------------------------------------------------------------------
def time_left(M, t):
    """we - t in ms."""
    return M.we - _t(t)


def tau(M, t):
    """(we - t) / 1000 in s (contract 7.0)."""
    return (M.we - _t(t)) / 1000.


def phase(M, t, post=True):
    """('pre' t < ws | 'live' ws <= t < we | 'post' t >= we (contract phases: pass post=False to label t >= we 'live'), window minute
    floor((t - ws) / 60 s): -5..-1 before ws, 0..4 live)."""
    t = _t(t)
    lab = _sel([t < M.ws, (t >= M.we) & bool(post)], ['pre', 'post'], 'live', len(t))
    return lab, np.floor_divide(t - M.ws, 60_000)


# ------------------------------------------------------------------ window evaluation ------------------------------------------------------------------
def over_window(M, fn, lo_ms, hi_ms, step_ms=STEP_MS, D_ms=0, mode='auto', tol=0.0, max_points=2_000_000, max_row_points=200_000):
    """A scalar feature over feasible windows [lo - D, hi] (one row per window): fn(M, t) -> values or (values, ok) evaluated at lo - D,
    every step_ms grid instant strictly inside, and hi. mode 'sign' (float values: categories neg / zero / pos with |v| <= tol zero; NaN
    missing, +-inf valid) or 'category' (exact values; None / NaN missing); 'auto' = sign for float values. Returns min / max (numeric
    values), first (at lo - D) / last (at hi), cats ('|'-joined set over the valid instants), n_cats, n_points, n_valid, n_missing,
    timing_sensitive (n_cats > 1), status invalid_window (lo < 0 or hi < lo) / too_long / missing / timing_sensitive / partially_missing /
    stable. Each instant reads only data <= itself; the window as a whole reads data <= hi."""
    lo, hi = np.broadcast_arrays(_t(lo_ms), _t(hi_ms))
    lo, hi = lo.astype(np.int64).copy(), hi.astype(np.int64).copy()
    n, step, D = len(lo), int(step_ms), int(D_ms)
    need(step > 0 and D >= 0 and mode in ('auto', 'sign', 'category'), 'over_window: step_ms > 0, D_ms >= 0, mode auto | sign | category')
    bad = (lo < 0) | (hi < lo)
    lo = np.where(bad, lo, lo - D)
    g1 = (lo // step + 1) * step
    g2 = ((hi - 1) // step) * step
    inner = np.where(bad | (hi <= lo), 0, np.maximum(0, (g2 - g1) // step + 1))
    m = np.where(bad, 0, np.where(hi == lo, 1, inner + 2)).astype(np.int64)
    too = m > max_row_points
    m[too] = 0
    R = dict(min=np.full(n, np.nan), max=np.full(n, np.nan), first=np.full(n, None, dtype=object), last=np.full(n, None, dtype=object),
             cats=np.full(n, '', dtype=object), n_cats=np.zeros(n, np.int64), n_points=m.copy(), n_valid=np.zeros(n, np.int64),
             status=np.full(n, '', dtype=object), lo_eval=lo, hi_eval=hi, D_ms=D)
    R['status'][bad] = 'invalid_window'
    R['status'][too] = 'too_long'
    rows = np.flatnonzero(m > 0)
    used = set()
    i0 = 0
    while i0 < len(rows):
        cs = np.cumsum(m[rows[i0:]])
        k = max(1, int(np.searchsorted(cs, max_points, 'right')))
        Rr = rows[i0:i0 + k]
        i0 += k
        mm = m[Rr]
        tot = int(mm.sum())
        off = np.r_[0, np.cumsum(mm)[:-1]].astype(np.int64)
        rid = np.repeat(np.arange(len(Rr)), mm)
        kk = np.arange(tot) - off[rid]
        tt = g1[Rr][rid] + (kk - 1) * step
        tt = np.where(kk == 0, lo[Rr][rid], tt)
        tt = np.where((kk == (mm - 1)[rid]) & (kk > 0), hi[Rr][rid], tt)
        res = fn(M, tt)
        vals, okm = (res[0], np.asarray(res[1], bool)) if isinstance(res, tuple) else (res, None)
        vals = np.asarray(vals)
        need(vals.shape == (tot,), 'over_window: fn must return one value per instant')
        md = mode if mode != 'auto' else ('sign' if vals.dtype.kind == 'f' else 'category')
        used.add(md)
        if md == 'sign':
            v = vals.astype(np.float64)
            valid = ~np.isnan(v)
            code = np.where(v > tol, 2, np.where(v < -tol, 0, 1))
            names = np.array(['neg', 'zero', 'pos'], dtype=object)
        else:
            if vals.dtype.kind == 'f':
                valid = ~np.isnan(vals)
            elif vals.dtype == object:
                valid = ~pd.isna(vals)
            else:
                valid = np.ones(tot, bool)
            code = np.full(tot, -1, np.int64)
            names = np.zeros(0, dtype=object)
            if valid.any():
                u, inv = np.unique(vals[valid], return_inverse=True)
                code[valid] = inv
                names = np.array([str(x) for x in u], dtype=object)
        if okm is not None:
            need(okm.shape == (tot,), 'over_window: the ok mask must have one value per instant')
            valid = valid & okm
        nv = np.bincount(rid[valid], minlength=len(Rr))
        K = max(len(names), 1)
        pairs = np.unique(rid[valid].astype(np.int64) * K + code[valid])
        prow, pcode = pairs // K, pairs % K
        nc = np.bincount(prow, minlength=len(Rr))
        starts = np.searchsorted(prow, np.arange(len(Rr)), 'left')
        ends = np.searchsorted(prow, np.arange(len(Rr)), 'right')
        R['n_valid'][Rr], R['n_cats'][Rr] = nv, nc
        for r_ in np.flatnonzero(nc):
            R['cats'][Rr[r_]] = '|'.join(names[pcode[starts[r_]:ends[r_]]])
        if vals.dtype.kind in 'fiub':
            vf = np.where(valid, vals.astype(np.float64), np.nan)
            with np.errstate(invalid='ignore'):
                R['min'][Rr] = np.fmin.reduceat(vf, off)
                R['max'][Rr] = np.fmax.reduceat(vf, off)
        last = off + mm - 1
        for dst, ix in (('first', off), ('last', last)):
            col = np.empty(len(Rr), dtype=object)
            col[:] = [vals[i].item() if hasattr(vals[i], 'item') else vals[i] for i in ix]
            col[~valid[ix]] = None
            R[dst][Rr] = col
        st = _sel([nv == 0, nc > 1, nv < mm], ['missing', 'timing_sensitive', 'partially_missing'], 'stable', len(Rr))
        R['status'][Rr] = st
    R['n_missing'] = R['n_points'] - R['n_valid']
    R['timing_sensitive'] = R['n_cats'] > 1
    R['mode'] = '|'.join(sorted(used)) if used else ''
    return R


def over_window_profiles(M, fn, lo_ms, hi_ms, D=D_PROFILES, **kw):
    """over_window for each declared submission-delay profile D (lo shifted down by D) -> {D: result}; shown separately, never identified."""
    return {int(d): over_window(M, fn, lo_ms, hi_ms, D_ms=int(d), **kw) for d in D}


# ------------------------------------------------------------------ controls ------------------------------------------------------------------
def draws_per_target(n_targets, min_pooled=MIN_POOLED):
    """Draws per target so that n_targets x draws >= min_pooled (pass the address-day's target count)."""
    need(int(n_targets) > 0, 'draws_per_target: no targets')
    return max(1, -(-int(min_pooled) // int(n_targets)))


def _int_of(s):
    return int.from_bytes(hashlib.blake2b(str(s).encode(), digest_size=8).digest(), 'big')


def matched_times(M, addr, n=None, seed=SEED, phase_of=None, mask_fn=None, exclude_windows=None, *, width_ms=0, step_ms=STEP_MS,
                  match='phase', pre_ms=PRE_MS, within=None, fallback_mask_fn=None, key='', min_pooled=MIN_POOLED):
    """Control instants c (window [c, c + width_ms]) in the SAME market and phase as each target: pool pre = [ws - pre_ms, ws), live =
    [ws, we) (match='minute': also the target's window minute), intersected with within=(lo, hi) when given (e.g. the address's active
    span); candidates on the step_ms grid whose whole window lies in the pool, where mask_fn(M, grid) holds at every grid instant of the
    window (mask_fn defines the population and may use later data; features at the controls must still be causal) and which overlap no
    exclude_windows [a, b]. An empty pool -> fallback_mask_fn (status 'fallback', counted) -> else 'no_eligible'. phase_of: target times
    (ms) or phase labels. Draws: n per target (None -> draws_per_target(n_targets, min_pooled)), uniform with replacement, numpy
    Generator(PCG64(SeedSequence([seed, day, mid, addr, key, target time, occurrence]))): deterministic, independent of call order and
    of the other targets. Returns t (targets x n; -1 none), status ok / fallback / no_eligible / no_pool, n_eligible, phase, minute, pooled
    (valid draws), under_min_pooled (pooled < min_pooled: pool across the address-day before testing)."""
    a = _akey(addr)
    need(phase_of is not None, 'matched_times: phase_of = the target times (ms) or phase labels')
    need(match in ('phase', 'minute'), "match must be 'phase' or 'minute'")
    po = np.asarray(phase_of)
    if po.ndim == 0:
        po = po.reshape(1)
    if po.dtype.kind in 'iuf':
        tt = _t(po)
        lab, minute = phase(M, tt)
    else:
        need(match == 'phase', "minute matching needs target times")
        tt, lab, minute = None, po.astype(object), np.zeros(len(po), np.int64)
    nt = len(lab)
    n = draws_per_target(nt, min_pooled) if n is None else int(n)
    need(n > 0, 'matched_times: n must be positive')
    width, step = int(width_ms), int(step_ms)
    need(width >= 0 and step > 0, 'matched_times: width_ms >= 0 and step_ms > 0')
    EXW = np.zeros((0, 2), np.int64) if exclude_windows is None else np.asarray(exclude_windows, np.int64).reshape(-1, 2)
    need(bool((EXW[:, 1] >= EXW[:, 0]).all()), 'exclude_windows must be [a, b] with b >= a')
    oe = np.argsort(EXW[:, 0], kind='stable')
    ea, epm = EXW[oe, 0], (np.maximum.accumulate(EXW[oe, 1]) if len(oe) else np.zeros(0, np.int64))
    cache = {}

    def pool(lb, mi, fn):
        if lb == 'pre':
            P0, P1 = M.ws - int(pre_ms), M.ws
        elif lb == 'live':
            P0, P1 = M.ws, M.we
        else:
            return None
        if match == 'minute':
            P0, P1 = max(P0, M.ws + 60_000 * mi), min(P1, M.ws + 60_000 * (mi + 1))
        if within is not None:
            P0, P1 = max(P0, int(within[0])), min(P1, int(within[1]) + 1)
        if P1 <= P0:
            return np.zeros(0, np.int64)
        g = P0 + step * np.arange((P1 - P0 + step - 1) // step, dtype=np.int64)
        mk = np.ones(len(g), bool) if fn is None else np.asarray(fn(M, g), bool)
        need(mk.shape == g.shape, 'mask_fn must return one bool per grid instant')
        w = width // step
        kk = np.arange(len(g))
        inside = g + width < P1
        badc = np.r_[0, np.cumsum(~mk)]
        allm = (badc[np.minimum(kk + w, len(g) - 1) + 1] - badc[kk]) == 0
        el = inside & allm
        if len(ea):
            kx = np.searchsorted(ea, g + width, 'right')
            el &= ~((kx > 0) & (epm[np.maximum(kx - 1, 0)] >= g))
        return g[el]

    T = np.full((nt, n), -1, np.int64)
    st, ne = np.full(nt, 'no_pool', dtype=object), np.zeros(nt, np.int64)
    occ, ai, ki = Counter(), int(a[2:], 16), _int_of(key)
    for i in range(nt):
        lb, mi = str(lab[i]), int(minute[i]) if match == 'minute' else 0
        if lb not in ('pre', 'live'):
            continue
        if (lb, mi, 0) not in cache:
            cache[(lb, mi, 0)] = pool(lb, mi, mask_fn)
        el, s_ = cache[(lb, mi, 0)], 'ok'
        if not len(el) and fallback_mask_fn is not None:
            if (lb, mi, 1) not in cache:
                cache[(lb, mi, 1)] = pool(lb, mi, fallback_mask_fn)
            el, s_ = cache[(lb, mi, 1)], 'fallback'
        ne[i] = len(el)
        if not len(el):
            st[i] = 'no_eligible'
            continue
        tk = int(tt[i]) if tt is not None else i
        c = occ[tk]
        occ[tk] += 1
        ss = np.random.SeedSequence([int(seed), int(M.day), int(M.mid), ai, ki, tk & ((1 << 64) - 1), c])
        T[i] = el[np.random.Generator(np.random.PCG64(ss)).integers(0, len(el), size=n)]
        st[i] = s_
    pooled = int((T >= 0).sum())
    return dict(t=T, status=st, n_eligible=ne, phase=lab, minute=minute, pooled=pooled, under_min_pooled=pooled < int(min_pooled), n=n,
                seed=int(seed), key=str(key), match=match, width_ms=width, step_ms=step, pre_ms=int(pre_ms))


def null_summary(obs, null):
    """Observed value against a control null (contract section 7): n, mean, p05 / p50 / p95, p_upper = (1 + #null >= obs) / (1 + n),
    p_lower = (1 + #null <= obs) / (1 + n); NaN draws dropped (counted in n_nan)."""
    x = np.asarray(null, np.float64).ravel()
    fin = np.isfinite(x)
    x = x[fin]
    nn = int(len(x))
    base = dict(obs=float(obs) if obs is not None else float('nan'), n=nn, n_nan=int((~fin).sum()))
    if nn == 0 or obs is None or not np.isfinite(obs):
        return dict(base, mean=float('nan'), p05=float('nan'), p50=float('nan'), p95=float('nan'), p_upper=float('nan'), p_lower=float('nan'))
    q = np.quantile(x, [0.05, 0.5, 0.95])
    return dict(base, mean=float(x.mean()), p05=float(q[0]), p50=float(q[1]), p95=float(q[2]), p_upper=(1 + int((x >= obs).sum())) / (1 + nn),
                p_lower=(1 + int((x <= obs).sum())) / (1 + nn))


# ------------------------------------------------------------------ declared moves and bins (contract 7.0) ------------------------------------------------------------------
def move_cat(r, thr):
    """up (r >= thr) / down (r <= -thr) / flat / unknown (NaN)."""
    r = np.atleast_1d(np.asarray(r, np.float64))
    return _sel([np.isnan(r), r >= thr, r <= -thr], ['unknown', 'up', 'down'], 'flat', len(r))


def align_cat(s, r, thr):
    """with (s x r >= thr) / against (s x r <= -thr) / flat / unknown; s = +1 YES bid / buy, -1 ask / sell."""
    r = np.atleast_1d(np.asarray(r, np.float64))
    sr = np.broadcast_to(np.asarray(s, np.float64), r.shape) * r
    return _sel([np.isnan(sr), sr >= thr, sr <= -thr], ['unknown', 'with', 'against'], 'flat', len(r))


def pos_bin(pos):
    """|position| bins [0,50) / [50,200) / [200,1000) / >=1000 shares / unknown (NaN)."""
    a = np.abs(np.atleast_1d(np.asarray(pos, np.float64)))
    return _sel([np.isnan(a), a < POS_EDGES[0], a < POS_EDGES[1], a < POS_EDGES[2]], ['unknown'] + list(POS_LABELS[:3]), POS_LABELS[3], len(a))


def age_bin(age_ms):
    """Resting age bins on an upper bound: le_1s / le_10s / gt_10s / unbounded (negative or NaN)."""
    a = np.atleast_1d(np.asarray(age_ms, np.float64))
    return _sel([np.isnan(a) | (a < 0), a <= AGE_EDGES[0], a <= AGE_EDGES[1]], ['unbounded', AGE_LABELS[0], AGE_LABELS[1]], AGE_LABELS[2], len(a))


def width_bin(w_ms):
    """Window width bins: le_50 / le_250 / le_1000 / gt_1000 / unbounded (negative or NaN)."""
    w = np.atleast_1d(np.asarray(w_ms, np.float64))
    return _sel([np.isnan(w) | (w < 0), w <= WIDTH_EDGES[0], w <= WIDTH_EDGES[1], w <= WIDTH_EDGES[2]],
                ['unbounded'] + list(WIDTH_LABELS[:3]), WIDTH_LABELS[3], len(w))


# ------------------------------------------------------------------ feature factories for over_window ------------------------------------------------------------------
def feat_btc(name='r_1s'):
    """btc_ret field (r_1s / r_5s / r_30s / r_ws / sig / p_fair, or r_<h ms>) with btc_feats' ok as the validity mask."""
    return lambda M, t: _feat_btc(M, t, name)


def _feat_btc(M, t, name):
    r = btc_ret(M, t, () if name in ('r_1s', 'r_5s', 'r_30s', 'r_ws', 'sig', 'p_fair') else (int(name[2:]),))
    return r[name], r['ok']


def feat_btc_align(name, s):
    """BTC move aligned with side s (+1 / -1) as a category (with / against / flat), thresholds MOVE_THR."""
    return lambda M, t: (lambda r: (align_cat(s, r[name], MOVE_THR[name]), r['ok']))(btc_ret(M, t))


def feat_book(field='spread'):
    return lambda M, t: (lambda b: (b[field], b['ok']))(book(M, t))


def feat_touch(side, tick, what='cat'):
    need(what in ('cat', 'ticks'), "feat_touch what = 'cat' | 'ticks'")
    return lambda M, t: (lambda d: (d[what], d['status'] == 'ok'))(touch_distance(M, t, side, tick))


def feat_level(side, tick):
    return lambda M, t: (lambda d: (d['size'], d['status'] == 'ok'))(level_size(M, t, side, tick))


def feat_position(addr, which='certain'):
    need(which in ('recorded', 'certain', 'possible', 'pending'), 'feat_position which = recorded | certain | possible | pending')
    return lambda M, t: (lambda p: (p[which], np.ones(len(p['t']), bool)))(position(M, addr, t))


# ------------------------------------------------------------------ smoke (one real market) ------------------------------------------------------------------
def _rss_mib():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024.
    return 0.


def _smoke(day, mid):
    cpu_imp = time.process_time()
    h0 = LC.vmhwm_mib()
    addrs = top3()
    cpu0, w0 = time.process_time(), time.time()
    M = load_market(day, int(mid), addrs)
    load = dict(cpu_s=round(time.process_time() - cpu0, 3), wall_s=round(time.time() - w0, 3), vmhwm_mib_before=round(h0, 1),
                vmhwm_mib_after=round(LC.vmhwm_mib(), 1), rss_mib_after=round(_rss_mib(), 1), cpu_imports_s=round(cpu_imp, 3))
    cpu1 = time.process_time()
    g = np.arange(M.ws - PRE_MS, M.we, 1000, dtype=np.int64)
    R = btc_ret(M, g)
    Bk = book(M, g)
    feats = dict(grid_1s=int(len(g)), btc=R['counts'], book_status=dict(Counter(Bk['status'].tolist())),
                 bid_st=dict(Counter(Bk['bid_st'].tolist())), ask_st=dict(Counter(Bk['ask_st'].tolist())))
    per = {}
    for a in M.addrs:
        G = M.legs[a]
        p, pe = position(M, a, g), pending(M, a, g)
        mk = G[np.asarray(G['role'], dtype=object) == 'maker']
        d = dict(ok_legs=int(len(G)), maker=int(len(mk)), excluded=dict(Counter(str(x) for x in np.asarray(M.excluded[a]['status'], dtype=object))),
                 certain_min=float(p['certain'].min()), certain_max=float(p['certain'].max()), recorded_eq_certain=int((p['recorded'] == p['certain']).sum()),
                 pending_points=int((pe['n'] > 0).sum()), pending_n_max=int(pe['n'].max()), pos_status=dict(Counter(p['status'].tolist())))
        if len(mk):
            hi = mk['hi_ms'].to_numpy(np.int64)
            sd = np.asarray(mk['y_side'], dtype=object)
            tk = np.rint(mk['p'].to_numpy(float) * 100).astype(np.int64)
            td = touch_distance(M, hi, sd, tk)
            ls = level_size(M, hi - 1, sd, tk)
            ow = over_window(M, feat_btc('r_1s'), np.where(mk['lo_hard_ms'].to_numpy(np.int64) >= 0, mk['lo_hard_ms'].to_numpy(np.int64), hi), hi)
            d.update(touch_at_hi=dict(Counter(td['cat'].tolist())), level_before_hi=dict(Counter(ls['status'].tolist())),
                     r1s_over_fill_window=dict(Counter(ow['status'].tolist())))
        tk_rows = G[np.asarray(G['role'], dtype=object) == 'taker']
        if len(tk_rows):
            rec = tk_rows.drop_duplicates('sid')
            tgt = rec['hi_ms'].to_numpy(np.int64)
            lo_r = rec['lo_hard_ms'].to_numpy(np.int64)
            mt = matched_times(M, a, None, phase_of=tgt, mask_fn=lambda M_, t_, a_=a: ~near_fill(M_, a_, t_, 2000, 2000, role='taker'),
                               exclude_windows=np.c_[np.where(lo_r >= 0, lo_r, tgt), tgt], width_ms=2000, key='smoke_take')
            d.update(take_records=int(len(rec)), controls=dict(status=dict(Counter(mt['status'].tolist())), pooled=mt['pooled'], n=mt['n']))
        per[a] = d
    feats['addresses'] = per
    rng = np.random.Generator(np.random.PCG64(SEED))
    x = rng.integers(int(M.tob['t'][0]) - 2000, int(M.tob['t'][-1]) + 2000, size=3000) if M.tob['n'] else np.zeros(0, np.int64)
    Fr = E.frames(M.raw_tob)
    em, ea, es = E.mid_at(Fr, np.full(len(x), M.mid, np.int64), x)
    bt = book_tob(M, x)
    parity = dict(n=int(len(x)), mid_mismatch=int((~((em == bt['mid']) | (np.isnan(em) & np.isnan(bt['mid'])))).sum()),
                  age_mismatch=int((ea != bt['age_ms']).sum()), status_mismatch=int((MARK[es] != bt['status']).sum()))
    out = dict(version=VERSION, day=day, mid=int(mid), load=load, summary=M.summary(), features=feats, parity_book_tob_vs_econ=parity,
               feature_cpu_s=round(time.process_time() - cpu1, 3), total_cpu_s=round(time.process_time(), 3), vmhwm_mib_end=round(LC.vmhwm_mib(), 1))
    print(json.dumps(out, indent=1, default=str))
    need(parity['mid_mismatch'] == 0 and parity['age_mismatch'] == 0 and parity['status_mismatch'] == 0, 'book_tob differs from pf_mt_econ.mid_at')
    return 0


def main(argv):
    if len(argv) == 4 and argv[1] == 'smoke':
        return _smoke(argv[2], argv[3])
    if len(argv) == 2 and argv[1] == 'api':
        print(__doc__)
        return 0
    raise SystemExit('usage: chain_context.py smoke <day> <mid> | api')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
