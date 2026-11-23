"""chain_context_selftest.py - synthetic proof for chain_context.py (P-2026-005 MAKER-TRACE; order_estimator_contract.md section 7.0).
Writes synthetic v2-layout roots in a temp dir (synthetic days 20261023 / 20261024 only; no real data file is opened except the accepted
BOOK manifest and econ/identities.json of the v2 root) and runs the PRODUCTION path on them: load_market -> pf_mt_decision.load_btc /
btc_feats, pf_mt_lifecycle._book, row-group reads, the (sid, mk_idx) window join. Exact hand-computed expectations: BTC returns / sig /
p_fair / statuses / causes and the 60 s pad when the previous day's grid exists; the book (contract rule: last USABLE frame, empty side
+-inf, unknown touch), the tob econ rule against pf_mt_econ.mid_at, depth cells, touch distances, book moves; positions with pending,
unbounded, excluded, self-trade and no-lifecycle legs on the three clocks; phase / time; over_window (sign flip -> timing_sensitive,
category flip, endpoint, partial, invalid, D profiles, chunking); matched_times (determinism, seed / key, stability, phase / minute /
within / mask / width / exclusions, fallback, >= 200 pooled); moves / bins / null_summary. A look-ahead falsifier: every data family after
t* perturbed (BTC rows, frames and levels, legs certainly after t*) changes no output at t <= t*; planted +150 ms / +5 s leaks must be
caught (positive controls) and the outputs after t* must change. Refusals assert their reasons. Usage (run_light, PF_SLOT=claude3,
MT_RUN=v2): chain_context_selftest.py"""
import copy
import datetime as dt
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chain_context as CX  # noqa: E402

C, E, LC, DEC = CX.C, CX.E, CX.LC, CX.DEC
NAN, INF = float('nan'), float('inf')
RES = []

DAY, PREV = '20261024', '20261023'
NEXT_DAY = (dt.datetime.strptime(C.MEASURE_DAYS[-1], '%Y%m%d') + dt.timedelta(days=1)).strftime('%Y%m%d')   # never printed / opened
D0 = CX.day_ms(DAY)
WS, WE = D0 + 3_600_000, D0 + 3_900_000
MID, MID2, MID3 = 900001, 900002, 900003
WS3 = D0 + 120_000
A = '0xA1' + 'a1' * 19
AL = A.lower()
B = '0x' + 'b2' * 20
CC = '0x' + 'c3' * 20
Z = '0x' + 'd4' * 20
TSTAR = WS + 25_037
NG = 864_000
PA = {'str': pa.string(), 'int64': pa.int64(), 'float64': pa.float64(), 'bool': pa.bool_()}
TOBS = dict(C.SCHEMAS['tob'])
FRS = dict(C.SCHEMAS['frames'], x_los_kind='str')
LVS = dict(C.SCHEMAS['levels'], x_price='float64')
FVS = {c: E.FV[c] for c in CX.FV_READ}
LFS = dict(C.SCHEMAS['fills'], x_lo_hard_ms='int64')
ORS = dict(CX.ORDERS_REQ, mid='int64')
MKS = dict(mid='int64', ws='int64', we='int64', status='str')


# ------------------------------------------------------------------ harness ------------------------------------------------------------------
def check(name, cond, detail=''):
    ok = bool(cond)
    RES.append((name, ok))
    if not ok:
        print(f'FAIL {name}: {detail}'[:700], flush=True)


def raises(name, fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(name, needle in str(e), f'{type(e).__name__}: {str(e)[:300]}')
        return
    check(name, False, 'did not raise')


def forbidden(name, fn):
    """Must raise pf_mt_common.Forbidden; the message (it names the refused day) is never printed."""
    try:
        fn()
    except C.Forbidden:
        check(name, True)
        return
    except Exception as e:  # noqa: BLE001
        check(name, False, f'{type(e).__name__} instead of Forbidden')
        return
    check(name, False, 'did not raise')


def close(a, b, tol=1e-6):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    with np.errstate(invalid='ignore'):
        return a.shape == b.shape and bool(np.all((np.isnan(a) & np.isnan(b)) | (a == b) | (np.abs(a - b) <= tol)))


def eqo(a, b):
    a, b = list(np.asarray(a, dtype=object)), list(b)
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def same(a, b):
    """Exact equality of nested outputs (dicts, tuples, arrays, frames); NaN == NaN, None == None."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, pd.DataFrame):
        return isinstance(b, pd.DataFrame) and list(a.columns) == list(b.columns) and len(a) == len(b) and \
            all(same(a[c].to_numpy(), b[c].to_numpy()) for c in a.columns)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype.kind in 'fc' and b.dtype.kind in 'fc':
            return bool(np.array_equal(a, b, equal_nan=True))
        if a.dtype == object or b.dtype == object:
            return all(_eq1(x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist()))
        return bool(np.array_equal(a, b))
    return _eq1(a, b)


def _eq1(x, y):
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, float) and isinstance(y, float) and x != x and y != y:
        return True
    return bool(x == y)


# ------------------------------------------------------------------ synthetic inputs ------------------------------------------------------------------
def f_bps(k):
    """BTC log-price path in bps of 1e-4 (relative grid index k = (g - WS) / 100): flat far before, +0.5 bp per 100 ms to k = 1000, then
    -0.5 bp per 100 ms to k = 3000, flat after."""
    k = np.asarray(k, np.float64)
    return np.where(k < -5000, -2500., np.where(k <= 1000, 0.5 * k, np.where(k <= 3000, 500. - 0.5 * (k - 1000), -500.)))


def btc_day(d0, const=False):
    g = d0 + 100 * np.arange(NG, dtype=np.int64)
    f = np.full(NG, -2500.) if const else f_bps((g - WS) // 100)
    X = dict(g=g, mid=70000. * np.exp(f * 1e-4), age=np.full(NG, 50, np.int64), code=np.zeros(NG, np.int8))
    if not const:
        o = (WS - d0) // 100
        X['code'][o + 2000], X['age'][o + 2000] = 1, 1500                     # stale
        X['code'][o + np.arange(2100, 2105)] = 2                              # gap
        for k in [2200] + list(range(2250, 2255)):                            # no_data
            X['code'][o + k], X['age'][o + k], X['mid'][o + k] = 3, -1, NAN
    return X


def write_btc(path, X):
    st = pc.take(pa.array(list(CX.BTC_FILE_ST)), pa.array(X['code'].astype(np.int64)))
    pq.write_table(pa.table(dict(g_ms=pa.array(X['g']), bid=pa.array(X['mid'] - 0.05), ask=pa.array(X['mid'] + 0.05), mid=pa.array(X['mid']),
                                 n_upd=pa.array(np.ones(NG, np.int64)), age_ms=pa.array(X['age']), status=st)), path, compression='zstd')


def frames_spec():
    """(rel ms to WS, bids [(tick, sh)], asks, tob status, depth status, extra level rows (side, tick, sh, level status, x_price))."""
    F = [(-250_000, [(40, 100.), (39, 20.), (38, 5.)], [(45, 50.), (46, 10.)], 'ok', 'ok', []),
         (-100_000, [(41, 80.), (40, 20.), (39, 5.)], [(45, 40.), (46, 10.)], 'ok', 'ok', []),
         (-100_000, [(42, 60.), (41, 20.), (40, 5.)], [(45, 40.), (46, 10.)], 'same_ms_changed', 'same_ms_changed', []),
         (1_000, [(43, 30.), (42, 20.), (41, 5.)], [], 'ok', 'ok', []),
         (2_000, [(50, 10.), (49, 20.), (48, 5.)], [(49, 10.), (50, 10.)], 'ok', 'ok', []),
         (3_000, [(44, 25.), (43, 20.), (42, 5.)], [(46, 35.), (47, 10.)], 'bad_frame', 'bad_frame', []),
         (4_000, [(44, 25.), (43, 20.), (42, 5.)], [(46, 35.), (47, 10.)], 'ok', 'ok', []),
         (10_000, [(44, 25.), (43, 20.), (42, 5.)], [(47, 20.), (48, 10.)], 'off_grid', 'off_grid', [('ask', 46, 3., 'off_grid', 0.4605)]),
         (10_500, [(45, 5.), (44, 25.), (43, 20.)], [(47, 20.), (48, 10.)], 'new_conn', 'new_conn+dup_level', [('bid', 43, 7., 'ok', NAN)]),
         (20_000, [], [], 'ok', 'ok', []),
         (20_500, [(44, 50.), (43, 20.)], [(46, 50.), (47, 10.)], 'ok', 'ok', [])]
    for rel in range(21_000, 302_001, 500):
        if rel < 100_000:
            F.append((rel, [(44, 50.), (43, 20.)], [(46, 50.), (47, 10.)], 'ok', 'ok', []))
        else:
            F.append((rel, [(45, 30.), (44, 50.)], [(47, 40.), (48, 10.)], 'ok', 'ok', []))
    return F


def frame_rows(spec, mid, base, conn='c1'):
    tob, fr, lv = [], [], []
    for fid, (rel, bids, asks, tst, dst, extra) in enumerate(spec):
        upd = base + rel
        recv = (upd + 120) * 1_000_000 + fid
        rb = [(k / 100., s) for k, s in bids] + [(xp if xp == xp else k / 100., s) for sd, k, s, _, xp in extra if sd == 'bid']
        ra = [(k / 100., s) for k, s in asks] + [(xp if xp == xp else k / 100., s) for sd, k, s, _, xp in extra if sd == 'ask']
        bb, ba = (max(rb) if rb else None), (min(ra) if ra else None)
        tob.append(dict(day=DAY, mid=mid, upd_ms=upd, recv_ns=recv, conn=conn, order_count=-1, n_bid=len(rb), n_ask=len(ra),
                        best_bid=bb[0] if bb else NAN, best_bid_sh=bb[1] if bb else 0., best_ask=ba[0] if ba else NAN,
                        best_ask_sh=ba[1] if ba else 0., pend_bid_usd=0., pend_ask_usd=0., status=tst))
        fr.append(dict(mid=mid, fid=fid, upd_ms=upd, recv_ns=recv, conn=conn, order_count=-1, pend_bids='[]', pend_asks='[]', los_id='',
                       los_side='', los_price=NAN, los_outcome='', status=dst, x_los_kind=''))
        for side, L in (('bid', bids), ('ask', asks)):
            lv += [dict(mid=mid, fid=fid, side=side, ptick=k, shares=s, status='ok', x_price=NAN) for k, s in L]
        lv += [dict(mid=mid, fid=fid, side=sd, ptick=k, shares=s, status=st, x_price=xp) for sd, k, s, st, xp in extra]
    return tob, fr, lv


def fv_row(sid, mk_idx, role, addr, s, q, p, settle, status='ok', self_trade=False, mid=MID, day=DAY):
    return dict(day=day, mid=mid, sid=sid, mk_idx=mk_idx, px_rank=0, role=role, addr=addr, hash=f'h_{sid}_{role}', cp_addr=Z, kind='regular',
                y_side='bid' if (s > 0) == (role == 'maker') else 'ask', s=s, q=q, p=p, self_trade=self_trade, settle_ms=settle, t_ms=settle,
                t_status='lo', fee=0., fee_type='COLLATERAL', fee_dy=0., status=status, src_status='ok')


def life_row(sid, mk_idx, hi, lo_hard, lo=-1, status='ok', shares=1.):
    return dict(mid=MID, sid=sid, mk_hash=f'mk_{sid}_{mk_idx}', mk_idx=mk_idx, y_side='bid', y_ptick=44, shares=shares, hi_ms=hi, lo_ms=lo,
                t_method='pend_inc', f_before=-1, f_after=-1, status=status, x_lo_hard_ms=lo_hard)


def base_inputs():
    tob, fr, lv = frame_rows(frames_spec(), MID, WS)
    t2, _, _ = frame_rows([(r, [(30, 5.)], [(35, 5.)], 'ok', 'ok', []) for r in range(-250_000, 302_001, 5_000)], MID2, WS, conn='c2')
    t3, _, _ = frame_rows([(r, [(40, 5.)], [(45, 5.)], 'ok', 'ok', []) for r in range(10_000, 400_001, 5_000)], MID3, D0, conn='c3')
    t2e, _, _ = frame_rows([(r, [(30, 5.)], [(35, 5.)], 'ok', 'ok', []) for r in range(3_600_000, 3_700_001, 5_000)], MID2, WS, conn='c2')
    fv = [fv_row('S1', 0, 'maker', A, 1, 10., .40, WS + 10_005), fv_row('S2', 0, 'taker', A, -1, 4., .46, WS + 20_000),
          fv_row('S2', 0, 'maker', Z, 1, 4., .46, WS + 20_000), fv_row('S3', 1, 'maker', A, 1, 5., .44, WS + 30_000),
          fv_row('S4', 0, 'maker', A, -1, 7., .47, WS + 40_000, status='record_leg_excluded'),
          fv_row('S5', 0, 'maker', A, 1, 3., .44, WS + 50_000, self_trade=True), fv_row('S5', 0, 'taker', A, -1, 3., .44, WS + 50_000, self_trade=True),
          fv_row('S6', 0, 'maker', A, -1, 1., .47, WS + 52_000), fv_row('S7', 0, 'maker', A, 1, 2., .44, WS + 60_000),
          fv_row('S8', 0, 'maker', B, 1, 9., .44, WS + 15_000), fv_row('S20', 0, 'maker', A, 1, 99., .5, WS + 12_000, mid=MID2)]
    life = [life_row('S1', 0, WS + 10_000, WS + 8_000, WS + 9_500), life_row('S2', 0, WS + 20_000, WS + 18_000, WS + 19_000),
            life_row('S3', 0, WS + 30_000, WS + 28_000), life_row('S3', 1, WS + 30_000, WS + 28_000, -1, 'fill_lo_unbounded'),
            life_row('S4', 0, WS + 40_000, WS + 38_000), life_row('S5', 0, WS + 50_000, WS + 48_000),
            life_row('S7', 0, WS + 60_000, -1, -1, 'hi_from_receipt'), life_row('S8', 0, WS + 15_000, WS + 13_000)]
    orders = [dict(scenario=s, mid=MID, mk_hash=h, mk_signer=AL, y_side='bid', y_ptick=44, persist_lo_ms=lo, persist_hi_ms=hi, status='ok')
              for s in ('S_FIFO', 'S_FIFO_LO', 'S_NOPRIO') for h, lo, hi in (('H1', -1, -1), ('H3', WS + 30_000, WS + 58_000))]
    return dict(spec=frames_spec(), tob_rg0=t3, tob_rg1=sorted(tob + t2, key=lambda r: r['recv_ns']), tob_rg2=t2e, frames=fr, levels=lv,
                fv_rg0=[fv_row('S90', 0, 'maker', A, 1, 1., .5, D0 - 50_000, mid=800001, day=PREV)], fv_rg1=fv,
                fv_rg2=[fv_row('S91', 0, 'maker', A, 1, 1., .5, D0 + 86_500_000, mid=900010, day='20261025')], life=life, orders=orders,
                markets=[dict(mid=MID, ws=WS // 1000, we=WE // 1000, status='ok'), dict(mid=MID2, ws=WS // 1000, we=WE // 1000, status='ok'),
                         dict(mid=MID3, ws=WS3 // 1000, we=WS3 // 1000 + 300, status='ok')],
                identities=dict(ex_ante=dict(top10_shares=[dict(addr=a, rank=i + 1) for i, a in enumerate(CX.TOP3)])), btc={DAY: btc_day(D0)})


def table(rows, spec):
    cols = {}
    for c, ty in spec.items():
        v = [r[c] for r in rows]
        if ty == 'str':
            cols[c] = pa.array([str(x) for x in v], type=pa.string())
        else:
            cols[c] = pa.array(np.asarray(v, dtype={'int64': np.int64, 'float64': np.float64, 'bool': bool}[ty]), type=PA[ty])
    return pa.table(cols)


def write_pq(path, tables):
    w = pq.ParquetWriter(path, tables[0].schema)
    for t in tables:
        w.write_table(t)
    w.close()


def write_root(root, inp):
    for d in ('btc', 'econ', 'depth', 'life'):
        (root / d).mkdir(parents=True, exist_ok=True)
    for d, X in inp['btc'].items():
        write_btc(root / 'btc' / f'btc_{d}.parquet', X)
    write_pq(root / f'markets_{DAY}.parquet', [table(inp['markets'], MKS)])
    write_pq(root / f'tob_{DAY}.parquet', [table(inp[k], TOBS) for k in ('tob_rg0', 'tob_rg1', 'tob_rg2')])
    write_pq(root / 'depth' / f'{DAY}_{MID}_frames.parquet', [table(inp['frames'], FRS)])
    write_pq(root / 'depth' / f'{DAY}_{MID}_levels.parquet', [table(inp['levels'], LVS)])
    write_pq(root / 'econ' / 'fills_valued.parquet', [table(inp[k], FVS) for k in ('fv_rg0', 'fv_rg1', 'fv_rg2')])
    write_pq(root / 'life' / f'{DAY}_{MID}_fills.parquet', [table(inp['life'], LFS)])
    write_pq(root / 'life' / f'{DAY}_{MID}_orders.parquet', [table(inp['orders'], ORS)])
    (root / 'econ' / 'identities.json').write_text(json.dumps(inp['identities']))
    (root / 'manifest.json').write_text(json.dumps(dict(complete={DAY: [MID]})))


def perturb(inp, ts, seed=7):
    """Everything strictly after ts changes: BTC rows g > ts (values and statuses), frames / levels with upd_ms > ts (prices, sizes, 30 %
    dropped, two added at ts + 50 / ts + 120), legs whose window starts at or after ts (certainly after: sizes, sides, statuses, one
    record removed, one added). Nothing at or before ts is touched."""
    rng = np.random.Generator(np.random.PCG64(seed))
    out = copy.deepcopy({k: v for k, v in inp.items() if k != 'btc'})
    X = {k: v.copy() for k, v in inp['btc'][DAY].items()}
    j = np.flatnonzero(X['g'] > ts)
    X['mid'][j] = 70000. * np.exp(rng.uniform(-0.03, 0.03, len(j)))
    c = rng.choice(4, size=len(j), p=[0.85, 0.05, 0.05, 0.05]).astype(np.int8)
    X['code'][j] = c
    X['age'][j] = np.where(c == 0, rng.integers(0, 1001, len(j)), np.where(c == 1, 1500, np.where(c == 2, 50, -1)))
    X['mid'][j[c == 3]] = NAN
    out['btc'] = {DAY: X}
    new = []
    for rel, bids, asks, tst, dst, extra in inp['spec']:
        if WS + rel > ts:
            if rng.random() < 0.3:
                continue
            sh = int(rng.integers(-2, 3))
            bids, asks = [(k + sh, s * 1.7) for k, s in bids], [(k + sh, s * 0.6) for k, s in asks]
        new.append((rel, bids, asks, tst, dst, extra))
    new += [(ts - WS + 50, [(30, 1.)], [(70, 1.)], 'ok', 'ok', []), (ts - WS + 120, [(20, 2.)], [(80, 2.)], 'ok', 'ok', [])]
    new.sort(key=lambda x: x[0])
    tob, fr, lv = frame_rows(new, MID, WS)
    out['spec'], out['frames'], out['levels'] = new, fr, lv
    out['tob_rg1'] = sorted(tob + [r for r in inp['tob_rg1'] if r['mid'] != MID], key=lambda r: r['recv_ns'])
    lo_of = {(r['sid'], r['mk_idx']): r['x_lo_hard_ms'] for r in inp['life']}
    fv = []
    for r in out['fv_rg1']:
        lo = lo_of.get((r['sid'], r['mk_idx']), r['settle_ms'] - CX.MAX_LAG_MS)
        if r['sid'] in ('S3', 'S4', 'S5', 'S6'):
            assert lo >= ts, 'the perturbation would touch a leg that may precede t*'
        if r['sid'] == 'S5':
            continue
        if r['sid'] == 'S3' and r['addr'] == A:
            r['q'] = 50.
        if r['sid'] == 'S4':
            r['status'] = 'ok'
        if r['sid'] == 'S6':
            r['s'], r['y_side'] = 1, 'bid'
        fv.append(r)
    fv.append(fv_row('S10', 0, 'maker', A, -1, 100., .47, WS + 70_000))
    out['fv_rg1'] = fv
    out['life'] = [r for r in out['life'] if r['sid'] != 'S5'] + [life_row('S10', 0, WS + 70_000, WS + 68_000)]
    return out


# ------------------------------------------------------------------ the tests ------------------------------------------------------------------
def test_guards():
    nowhere = Path('/nonexistent_chain_context_selftest_root')
    forbidden('load_market refuses the ID day before any path', lambda: CX.load_market(PREV, MID, [A], root=nowhere))
    forbidden('load_market refuses the day after the last measure day', lambda: CX.load_market(NEXT_DAY, MID, [A], root=nowhere))
    forbidden('chain_context paths refuse a non-measure day', lambda: CX._day_paths(nowhere, NEXT_DAY, MID))
    forbidden('pf_mt_decision.btc_path refuses the day after (unchanged guard)', lambda: DEC.btc_path(nowhere, NEXT_DAY))
    bm = CX.book_markets()
    check('book_markets: the accepted manifest holds 19 BOOK markets on measurement days', sum(len(v) for v in bm.values()) == 19
          and set(bm) <= set(C.MEASURE_DAYS))
    check('top3 = pf_mt_decision.TOP3 (v2 identities)', tuple(a.lower() for a in CX.top3()) == tuple(a.lower() for a in CX.TOP3))


def test_load(M0, inp):
    c = M0.counts
    check('tob read: 1 of 3 row groups (mid statistics prune the others)', c['tob_row_groups_read'] == 1 and c['tob_row_groups_total'] == 3, dict(c))
    check('fills_valued read: 1 of 3 row groups', c['fv_row_groups_read'] == 1 and c['fv_row_groups_total'] == 3, dict(c))
    check('fills_valued rows kept: the 3 addresses in this market only (9)', c['fv_rows_kept'] == 9, c['fv_rows_kept'])
    check('BOOK market with depth', M0.book and M0.depth is not None and M0.market_status == 'ok')
    check('tob and depth frame sets equal', M0.checks.get('tob_depth_frames_equal') is True, M0.checks)
    check('tob best ticks agree with the usable depth frames', M0.checks['best_bid_mismatch'] == 0 and M0.checks['best_ask_mismatch'] == 0
          and M0.checks['best_bid_compared'] > 500, M0.checks)
    check('lifecycle exclusions: one bad_frame, one empty book in the window', M0.depth['B']['excluded'] == {'bad_frame': 1, 'empty_book_in_window': 1},
          M0.depth['B']['excluded'])
    check('depth unusable frames counted (2)', M0.checks['depth_unusable'] == 2, M0.checks)
    S = M0.summary()
    la = S['legs'][AL]
    check('legs: 7 ok, 1 excluded (record_leg_excluded), 1 without lifecycle window, 1 unbounded lo', la['ok'] == 7 and la['excluded'] ==
          {'record_leg_excluded': 1} and la['no_life_window'] == 1 and la['lo_unbounded'] == 1, la)
    check('legs of B and C', S['legs'][B.lower()]['ok'] == 1 and S['legs'][CC]['ok'] == 0, S['legs'])
    check('summary is JSON', isinstance(json.dumps(S, default=str), str))
    check('every input bound by sha256 (hash_cache)', len(M0.inputs) >= 9 and all('sha256' in v for v in M0.inputs.values()), list(M0.inputs))
    check('BTC grid covers the span with its 60 s pad', M0.checks['btc_grid_covers_span'] is True, M0.span)
    check('no fill hi after its settle_ms', M0.checks.get(f'hi_after_settle:{AL}') == 0, M0.checks)
    check('span = [ws - 300 s - 60 s, last frame + 60 s]', M0.span == (WS - 360_000, WS + 362_000), M0.span)


def test_btc(M0, R0, Q):
    ta, tb, tf, tc, td, te, tg, th, ti = WS - 99_950, WS + 10_050, WS + 110_000, WS + 210_200, WS + 226_000, WS + 205_000, WE, D0 + 30_000, WS + 290_000
    R = CX.btc_ret(M0, [ta, tb, tf, tc, td, te, tg, th, ti], horizons=(1000, 5000, 30000, 2000, 120_000))
    check('btc status per instant', eqo(R['status'], ['ok', 'ok', 'ok', 'btc_invalid', 'btc_invalid', 'btc_invalid', 'after_we', 'btc_missing', 'ok']),
          R['status'])
    check('btc cause: gap / no_data / stale / after_we / outside_grid', eqo(R['cause'], ['ok', 'ok', 'ok', 'gap', 'no_data', 'stale', 'after_we',
                                                                                            'outside_grid', 'ok']), R['cause'])
    check('r_1s hand values', close(R['r_1s'], [5, 5, -5, NAN, NAN, NAN, NAN, NAN, -5]), R['r_1s'])
    check('r_5s hand values', close(R['r_5s'], [25, 25, -25, NAN, NAN, NAN, NAN, NAN, -25]), R['r_5s'])
    check('r_30s hand values', close(R['r_30s'], [150, 150, 50, NAN, NAN, NAN, NAN, NAN, -150]), R['r_30s'])
    check('r_ws hand values (0 before ws)', close(R['r_ws'], [0, 50, 450, NAN, NAN, NAN, NAN, NAN, -450]), R['r_ws'])
    check('r_2000 (another horizon) hand values', close(R['r_2000'], [10, 10, -10, NAN, NAN, NAN, NAN, NAN, -10]), R['r_2000'])
    check('r_120000: NaN when a grid row in [j, i] is not usable', close(R['r_120000'], [600, 600, 500, NAN, NAN, NAN, NAN, NAN, NAN]), R['r_120000'])
    s25 = np.sqrt(2.5)
    check('sig = sqrt(600 x 0.5^2 / 60) bps / sqrt(s)', close(R['sig'][[0, 1, 2, 8]], [s25] * 4, 1e-9), R['sig'])
    zb, zf, zi = 50 / (s25 * np.sqrt(289.95)), 450 / (s25 * np.sqrt(190.)), -450 / (s25 * np.sqrt(10.))
    check('p_fair = Phi(r_ws / (sig sqrt(tau))); 0.5 before ws', close(R['p_fair'][[0, 1, 2, 8]], [0.5, ndtr(zb), ndtr(zf), ndtr(zi)], 1e-9), R['p_fair'])
    check('tau in s', close(R['tau'][[1, 2, 8]], [289.95, 190., 10.]))
    check('ms aliases are the same arrays', R['r_1000'] is R['r_1s'] and R['r_30000'] is R['r_30s'])
    check('status counts', R['counts']['status'] == {'ok': 4, 'btc_invalid': 3, 'after_we': 1, 'btc_missing': 1}, R['counts'])
    st = CX.btc_state(M0, [WS + 200_000, WS + 210_000, WS + 220_000, WS + 10_000, WS + 225_200])
    check('btc_state single-row statuses', eqo(st['status'], ['stale', 'gap', 'no_data', 'ok', 'no_data']), st['status'])
    check('btc_state mid', close(st['mid'][3], 70000. * np.exp(0.005), 1e-6) and np.isnan(st['mid'][0]))
    BT2 = DEC.load_btc(R0 / 'btc', DAY, Counter())
    F2 = DEC.btc_feats(BT2, Q, WS, WE)
    Rq = CX.btc_ret(M0, Q)
    check('btc_ret = pf_mt_decision.btc_feats on a fresh load_btc (every key, every instant)', all(
        same(np.asarray(F2[k]), np.asarray(Rq[m])) for k, m in (('code', 'code'), ('r_1s', 'r_1s'), ('r_5s', 'r_5s'), ('r_30s', 'r_30s'),
                                                                  ('r_ws', 'r_ws'), ('sig', 'sig'), ('pf', 'p_fair'), ('tau', 'tau'))))
    okc = Rq['status'] == 'ok'
    check('cause ok exactly where the code is ok; a named cause otherwise', bool((Rq['cause'][okc] == 'ok').all()) and
          not bool((Rq['cause'][~okc] == 'ok').any()))


def test_book(M0):
    t = np.array([WS - 300_000, WS - 250_000, WS - 249_000, WS - 248_999, WS - 100_000, WS + 1_000, WS + 2_000, WS + 3_000, WS + 3_500,
                  WS + 4_000, WS + 10_000, WS + 10_500, WS + 20_000, WS + 20_500])
    b = CX.book(M0, t)
    check('book status (contract rule: last usable frame)', eqo(b['status'], ['no_frame', 'ok', 'ok', 'stale', 'ok', 'one_sided', 'crossed', 'crossed',
                                                                               'stale', 'ok', 'unknown_touch', 'ok', 'stale', 'ok']), b['status'])
    check('book bid_tick', close(b['bid_tick'], [NAN, 40, 40, NAN, 42, 43, 50, 50, NAN, 44, 44, 45, NAN, 44]), b['bid_tick'])
    check('book ask_tick (+inf empty side, NaN unknown cell below the best)', close(b['ask_tick'], [NAN, 45, 45, NAN, 45, INF, 49, 49, NAN, 46, NAN,
                                                                                                    47, NAN, 46]), b['ask_tick'])
    check('book sizes', close(b['bid_sz'], [NAN, 100, 100, NAN, 60, 30, 10, 10, NAN, 25, 25, 5, NAN, 50]) and
          close(b['ask_sz'], [NAN, 50, 50, NAN, 40, 0, 10, 10, NAN, 35, NAN, 20, NAN, 50]), (b['bid_sz'], b['ask_sz']))
    check('book spread / mid_c', close(b['spread'], [NAN, 5, 5, NAN, 3, INF, -1, -1, NAN, 2, NAN, 2, NAN, 2]) and
          close(b['mid_c'], [NAN, 42.5, 42.5, NAN, 43.5, NAN, 49.5, 49.5, NAN, 45, NAN, 46, NAN, 45]), (b['spread'], b['mid_c']))
    check('book ok flags', eqo(b['ok'], [False, True, True, False, True, True, False, False, False, True, False, True, False, True]), b['ok'])
    check('per-side statuses', b['ask_st'][5] == 'empty' and b['ask_st'][10] == 'unknown' and b['bid_st'][10] == 'ok' and b['bid_st'][3] == 'stale'
          and b['bid_st'][0] == 'no_frame', (b['bid_st'], b['ask_st']))
    check('a bad frame is skipped: t = ws + 3 s reads the ws + 2 s frame (age 1000)', b['age_ms'][7] == 1000 and b['frame'][7] == b['frame'][6]
          and b['upd_ms'][7] == WS + 2_000)
    lo = CX.book(M0, t, strict=False)
    check('strict=False keeps a stale frame values', lo['bid_tick'][3] == 40 and lo['status'][3] == 'stale' and lo['bid_tick'][12] == 45
          and lo['ask_tick'][12] == 47 and not lo['ok'][12])
    check('book source depth', b['source'] == 'depth')
    k = CX.book_tob(M0, t)
    check('book_tob = econ MID rule (last frame, flagged frames unusable)', eqo(k['status'], ['no_frame', 'ok', 'ok', 'stale', 'ok', 'one_sided',
                                                                                              'crossed', 'frame_status', 'frame_status', 'ok',
                                                                                              'frame_status', 'ok', 'one_sided', 'ok']), k['status'])
    check('book_tob mid', close(k['mid'][[1, 4, 9, 11]], [0.425, 0.435, 0.45, 0.46], 1e-12), k['mid'])
    Fr = E.frames(M0.raw_tob)
    x = np.arange(WS - 300_000, WS + 310_000, 23, dtype=np.int64)
    em, ea, es = E.mid_at(Fr, np.full(len(x), MID, np.int64), x)
    kk = CX.book_tob(M0, x)
    check('book_tob = pf_mt_econ.mid_at (mid, age, status) at 26k instants', same(em, kk['mid']) and same(ea, kk['age_ms']) and
          same(CX.MARK[es], kk['status']))
    c = CX.book_change(M0, [WS + 100_200, WS + 50_000], 'bid')
    c2 = CX.book_change(M0, [WS + 100_200], 'ask')
    check('book_change: mid +1 tick moved; ask up = away from a bid; bid up = toward an ask', close(c['d_mid_c'], [1, 0]) and
          eqo(c['mid_cat'], ['moved', 'flat']) and eqo(c['opp_cat'], ['away', 'flat']) and eqo(c2['opp_cat'], ['toward']) and bool(c['ok'].all()),
          (c, c2))


def test_levels(M0):
    v = CX.level_size(M0, [WS - 250_000] * 5, ['bid', 'bid', 'bid', 'ask', 'ask'], [39, 38, 37, 46, 45])
    check('level sizes (0 = absent from a usable snapshot)', close(v['size'], [20, 5, 0, 10, 50]) and eqo(v['status'], ['ok'] * 5), v)
    check('level from the last usable frame (bad frame skipped)', close(CX.level_size(M0, WS + 3_000, 'bid', 49)['size'], [20]))
    v = CX.level_size(M0, [WS + 10_000] * 2, 'ask', [46, 47])
    check('off_grid level -> unknown cell, others known', close(v['size'], [NAN, 20]) and eqo(v['status'], ['unknown_cell', 'ok']), v)
    v = CX.level_size(M0, [WS + 10_500] * 3, 'bid', [43, 45, 44])
    check('dup_level cell unknown', close(v['size'], [NAN, 5, 25]) and eqo(v['status'], ['unknown_cell', 'ok', 'ok']), v)
    v, v2 = CX.level_size(M0, WS + 12_000, 'bid', 45), CX.level_size(M0, WS + 12_000, 'bid', 45, strict=False)
    check('stale level (age 1500)', np.isnan(v['size'][0]) and v['status'][0] == 'stale' and v2['size'][0] == 5)
    check('no frame / bad tick', CX.level_size(M0, WS - 300_000, 'bid', 40)['status'][0] == 'no_frame' and
          CX.level_size(M0, WS, 'bid', 150)['status'][0] == 'bad_tick')
    s = CX.level_series(M0, 'bid', 44, WS + 3_000, WS + 21_000)
    check('level_series: usable frames from the one in force at lo', list(s['upd_ms']) == [WS + 2_000, WS + 4_000, WS + 10_000, WS + 10_500,
                                                                                         WS + 20_500, WS + 21_000]
          and close(s['size'], [0, 25, 25, 25, 50, 50]) and list(s['gap_before']) == [False, True, True, False, True, False]
          and list(s['in_force_at_lo']) == [True, False, False, False, False, False], s)
    check('level_at_frames = level_series sizes', close(CX.level_at_frames(M0, s['frame'].to_numpy(), 'bid', 44), s['size']))
    check('frame_times = usable depth frames', len(CX.frame_times(M0)) == M0.depth['B']['F'] and bool((np.diff(CX.frame_times(M0)) >= 0).all()))
    d = CX.touch_distance(M0, [WS - 250_000] * 6, ['bid', 'bid', 'bid', 'ask', 'ask', 'ask'], [40, 38, 42, 45, 47, 44])
    check('touch distance AT / BEHIND / INSIDE both sides', close(d['ticks'], [0, 2, -2, 0, 2, -1]) and
          eqo(d['cat'], ['at', 'behind', 'inside', 'at', 'behind', 'inside']), d)
    d = CX.touch_distance(M0, [WS + 1_000, WS + 1_000, WS + 10_000, WS + 3_500], ['ask', 'bid', 'ask', 'bid'], [45, 43, 47, 50])
    check('touch: empty own side is inside (-inf); unknown touch; stale', close(d['ticks'], [-INF, 0, NAN, NAN]) and
          eqo(d['cat'], ['inside', 'at', 'unknown', 'unknown']) and eqo(d['status'], ['empty', 'ok', 'unknown', 'stale']), d)


def test_positions(M0):
    t = WS + np.array([5_000, 9_000, 10_000, 10_001, 10_003, 29_000, 38_000, 38_001, 51_000, 53_000, 60_001])
    p = CX.position(M0, A, t)
    check('certain (hi < t)', close(p['certain'], [0, 0, 0, 10, 10, 6, 11, 11, 11, 10, 12], 1e-12), p['certain'])
    check('recorded (settle_ms < t)', close(p['recorded'], [0, 0, 0, 0, 0, 6, 11, 11, 11, 10, 12], 1e-12), p['recorded'])
    check('possible (x_lo_hard < t; unbounded counts)', close(p['possible'], [2, 12, 12, 12, 12, 13, 13, 13, 12, 12, 12], 1e-12), p['possible'])
    check('pending = possible - certain', close(p['pending'], [2, 12, 12, 2, 2, 7, 2, 2, 1, 2, 0], 1e-12), p['pending'])
    check('pos = the declared clock (hi)', close(p['pos'], p['certain'], 0))
    check('excluded leg flags from its window start on', eqo(p['status'], ['ok'] * 7 + ['excluded_leg_possible'] * 4), p['status'])
    check('n and last_ms on the hi clock', p['n'][9] == 6 and p['last_ms'][9] == WS + 52_000 and p['last_ms'][0] == -1)
    pl = CX.position(M0, A, t, clock='lo')
    check('lo clock: pos = possible; unbounded last_ms -1', close(pl['pos'], p['possible'], 0) and pl['last_ms'][0] == -1)
    check('settle clock: pos = recorded', close(CX.position(M0, A, t, clock='settle')['pos'], p['recorded'], 0))
    q = CX.pending(M0, A, t)
    check('pending n', list(q['n']) == [1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 0], q['n'])
    check('pending buy / sell', close(q['buy_q'], [2, 12, 12, 2, 2, 7, 2, 2, 2, 2, 0]) and close(q['sell_q'], [0] * 8 + [1, 0, 0]), q)
    check('feasible range [pos_min, pos_max]', close(q['pos_min'], [0, 0, 0, 10, 10, 6, 11, 11, 10, 10, 12]) and
          close(q['pos_max'], [2, 12, 12, 12, 12, 13, 13, 13, 13, 12, 12]), q)
    check('pending net = position pending (exact identity)', same(q['net'], p['pending']))
    check('pending legs without a lower end', list(q['n_lo_unbounded']) == [1] * 10 + [0], q['n_lo_unbounded'])
    check('pending_legs at one instant', sorted(CX.pending_legs(M0, A, WS + 51_000)['sid'].tolist()) == ['S6', 'S7'])
    pp = CX.position_path(M0, A)
    check('position path (hi clock): 7 legs, final 12, ordered', len(pp) == 7 and close(pp['pos'].iloc[-1], 12) and bool((np.diff(pp['t']) >= 0).all()), pp)
    check('B and C positions', close(CX.position(M0, B, WS + 16_000)['certain'], [9]) and close(CX.position(M0, CC, WS + 16_000)['certain'], [0]))
    check('near_fill (population mask)', list(CX.near_fill(M0, A, [WS + 19_000, WS + 17_000, WS + 50_000], 0, 2000, role='taker')) == [True, False, True])
    r, s = CX.resting(M0, 'H3', [WS + 29_999, WS + 30_000, WS + 58_000, WS + 58_001])
    check('resting: certain presence span', s == 'ok' and list(r) == [False, True, True, False])
    check('resting statuses', CX.resting(M0, 'H1', WS)[1] == 'no_persistence' and CX.resting(M0, 'HX', WS)[1] == 'no_order')
    raises('an unloaded address refuses', lambda: CX.position(M0, Z, WS), 'was not loaded')


def test_time(M0):
    lab, mi = CX.phase(M0, [WS - 1, WS, WS + 59_999, WS + 60_000, WE - 1, WE])
    check('phase labels and window minute', eqo(lab, ['pre', 'live', 'live', 'live', 'live', 'post']) and list(mi) == [-1, 0, 0, 1, 4, 5], (lab, mi))
    check('phase post=False (contract two phases)', CX.phase(M0, [WE], post=False)[0][0] == 'live')
    check('time_left (ms) / tau (s)', CX.time_left(M0, WE - 1_000)[0] == 1_000 and CX.tau(M0, WE - 1_000)[0] == 1.0)
    check('market_ws / market_we', CX.market_ws(M0) == WS and CX.market_we(M0) == WE)


def test_over_window(M0):
    r1 = CX.feat_btc('r_1s')
    w = CX.over_window(M0, r1, WS - 100_000, WS - 90_000)
    check('stable window', w['status'][0] == 'stable' and w['n_points'][0] == 101 and w['n_valid'][0] == 101 and close(w['min'], [5]) and
          close(w['max'], [5]) and w['cats'][0] == 'pos' and abs(w['first'][0] - 5) < 1e-6 and abs(w['last'][0] - 5) < 1e-6, w)
    w = CX.over_window(M0, r1, WS + 99_000, WS + 102_000, tol=1e-9)
    check('sign flip inside the window -> timing_sensitive', bool(w['timing_sensitive'][0]) and w['status'][0] == 'timing_sensitive' and
          w['cats'][0] == 'neg|zero|pos' and close(w['min'], [-5]) and close(w['max'], [5]) and w['n_points'][0] == 31, w)
    w = CX.over_window(M0, r1, WS + 199_000, WS + 201_000)
    check('partially missing (BTC invalid after a stale row)', w['status'][0] == 'partially_missing' and w['n_points'][0] == 21 and
          w['n_valid'][0] == 10 and w['n_missing'][0] == 11 and w['cats'][0] == 'neg', w)
    w = CX.over_window(M0, CX.feat_touch('bid', 44), WS + 99_000, WS + 101_000)
    check('category flip (touch at -> behind) -> timing_sensitive', w['status'][0] == 'timing_sensitive' and w['cats'][0] == 'at|behind' and
          w['first'][0] == 'at' and w['last'][0] == 'behind', w)
    w = CX.over_window(M0, r1, [WS + 5, -1], [WS, WS + 10])
    check('invalid windows are a status', eqo(w['status'], ['invalid_window', 'invalid_window']) and list(w['n_points']) == [0, 0])
    w = CX.over_window(M0, lambda M, t: np.where(t >= WS + 1_250, 1.0, -1.0), WS + 1_000, WS + 1_250)
    check('hi off the grid is evaluated (lo, inner grid, hi)', w['n_points'][0] == 4 and w['first'][0] == -1.0 and w['last'][0] == 1.0 and
          bool(w['timing_sensitive'][0]), w)
    w = CX.over_window(M0, lambda M, t: (np.ones(len(t)), np.zeros(len(t), bool)), WS, WS + 1_000)
    check('all instants masked -> missing', w['status'][0] == 'missing' and w['n_valid'][0] == 0)
    lo = WS + np.arange(0, 200_000, 7_000)
    a, b = CX.over_window(M0, r1, lo, lo + 3_000), CX.over_window(M0, r1, lo, lo + 3_000, max_points=50)
    check('chunked evaluation = one pass', same(a, b))
    w0 = CX.over_window(M0, r1, WS + 99_000, WS + 102_000, D_ms=250)
    P = CX.over_window_profiles(M0, r1, WS + 99_000, WS + 102_000)
    check('D profile shifts lo down', w0['lo_eval'][0] == WS + 98_750 and w0['n_points'][0] == 34 and sorted(P) == [0, 250, 1000] and
          P[1000]['lo_eval'][0] == WS + 98_000)
    w = CX.over_window(M0, lambda M, t: CX.phase(M, t)[0], WS - 2_000, WS + 2_000)
    check('string categories', w['cats'][0] == 'live|pre' and w['first'][0] == 'pre' and w['last'][0] == 'live' and w['mode'] == 'category', w)
    al = CX.over_window(M0, CX.feat_btc_align('r_5s', 1), WS + 10_000, WS + 11_000)
    check('aligned BTC category (with)', al['cats'][0] == 'with' and al['status'][0] == 'stable', al)


def test_matched(M0):
    tgt = np.array([WS - 100_000, WS + 50_000, WS + 200_000, WE + 1_000])
    mask = lambda M, t: (t // 1000) % 2 == 0  # noqa: E731
    ex = [[WS + 10_000, WS + 20_000]]
    kw = dict(phase_of=tgt, mask_fn=mask, exclude_windows=ex, width_ms=300, key='k1')
    r1 = CX.matched_times(M0, A, 50, **kw)
    check('statuses (post target has no pool)', eqo(r1['status'], ['ok', 'ok', 'ok', 'no_pool']) and r1['t'].shape == (4, 50) and
          bool((r1['t'][3] == -1).all()), r1['status'])
    pre, live = r1['t'][0], r1['t'][1:3].ravel()
    check('pre controls inside [ws - 300 s, ws) with the whole window', bool(((pre >= WS - 300_000) & (pre + 300 < WS)).all()))
    check('live controls inside [ws, we) with the whole window', bool(((live >= WS) & (live + 300 < WE)).all()))
    allc = r1['t'][:3].ravel()
    check('mask holds over the width-matched window', all(bool((((allc + 100 * k) // 1000) % 2 == 0).all()) for k in range(4)))
    check('exclusion windows respected', bool(((allc > WS + 20_000) | (allc + 300 < WS + 10_000)).all()))
    check('controls on the 100 ms grid', bool((allc % 100 == 0).all()))
    check('deterministic (same call -> same draws)', same(r1['t'], CX.matched_times(M0, A, 50, **kw)['t']))
    check('seed changes the draws', not same(r1['t'], CX.matched_times(M0, A, 50, seed=CX.SEED + 1, **kw)['t']))
    check('key changes the draws', not same(r1['t'], CX.matched_times(M0, A, 50, **dict(kw, key='k2'))['t']))
    r3 = CX.matched_times(M0, A, 50, **dict(kw, phase_of=np.r_[tgt, WS + 150_000]))
    check('draws independent of the other targets', same(r3['t'][:4], r1['t']))
    check('pooled count and the under-200 flag', r1['pooled'] == 150 and r1['under_min_pooled'] is True)
    rd = CX.matched_times(M0, A, None, **dict(kw, phase_of=tgt[:3]))
    check('draws_per_target -> >= 200 pooled', CX.draws_per_target(3) == 67 and rd['n'] == 67 and rd['pooled'] == 201 and not rd['under_min_pooled'])
    rl = CX.matched_times(M0, A, 10, phase_of=['pre', 'live', 'post'])
    check('phase labels as targets', eqo(rl['status'], ['ok', 'ok', 'no_pool']) and bool((rl['t'][0] < WS).all()) and bool((rl['t'][1] >= WS).all()))
    rm = CX.matched_times(M0, A, 30, phase_of=[WS + 130_000], match='minute')
    check('minute matching', bool(((rm['t'] >= WS + 120_000) & (rm['t'] < WS + 180_000)).all()) and rm['minute'][0] == 2)
    rw = CX.matched_times(M0, A, 30, phase_of=[WS + 70_000], within=(WS + 60_000, WS + 90_000))
    check('within (active span)', bool(((rw['t'] >= WS + 60_000) & (rw['t'] <= WS + 90_000)).all()))
    never, always = (lambda M, t: np.zeros(len(t), bool)), (lambda M, t: np.ones(len(t), bool))
    rn = CX.matched_times(M0, A, 5, phase_of=[WS + 1_000], mask_fn=never)
    rf = CX.matched_times(M0, A, 5, phase_of=[WS + 1_000], mask_fn=never, fallback_mask_fn=always)
    check('empty stratum -> no_eligible; declared fallback -> fallback (counted)', rn['status'][0] == 'no_eligible' and rn['n_eligible'][0] == 0
          and rf['status'][0] == 'fallback' and rf['n_eligible'][0] == 3000 and bool((rf['t'] >= WS).all()))


def test_helpers():
    check('pos_bin', eqo(CX.pos_bin([0, 49.9, 50, 199, 200, 999, 1000, -1500, NAN]), ['[0,50)', '[0,50)', '[50,200)', '[50,200)', '[200,1000)',
                                                                                     '[200,1000)', '>=1000', '>=1000', 'unknown']))
    check('age_bin', eqo(CX.age_bin([0, 1000, 1001, 10000, 10001, -1, NAN]), ['le_1s', 'le_1s', 'le_10s', 'le_10s', 'gt_10s', 'unbounded', 'unbounded']))
    check('width_bin', eqo(CX.width_bin([50, 51, 250, 251, 1000, 1001, -1]), ['le_50', 'le_250', 'le_250', 'le_1000', 'le_1000', 'gt_1000', 'unbounded']))
    check('align_cat', eqo(CX.align_cat(1, [0.5, 0.49, -0.5, NAN], 0.5), ['with', 'flat', 'against', 'unknown']) and
          eqo(CX.align_cat(-1, [0.5], 0.5), ['against']) and eqo(CX.move_cat([2., -2., 1.], 2.), ['up', 'down', 'flat']))
    ns = CX.null_summary(3.0, [1, 2, 3, 4, NAN])
    check('null_summary', ns['n'] == 4 and ns['n_nan'] == 1 and ns['mean'] == 2.5 and abs(ns['p_upper'] - 0.6) < 1e-12 and abs(ns['p_lower'] - 0.8) < 1e-12, ns)
    check('declared constants', CX.SEED == 20261123 and CX.MIN_POOLED == 200 and CX.D_PROFILES == (0, 250, 1000) and CX.HORIZONS == (1000, 5000, 30000)
          and CX.MOVE_THR == {'r_1s': 0.5, 'r_5s': 1.0, 'r_30s': 2.0})


def test_pad_and_nonbook(R0, R2):
    man = R0 / 'manifest.json'
    raises('a non-BOOK market refuses unless require_book=False', lambda: CX.load_market(DAY, MID3, [A], root=R0, book_manifest=man), 'not a BOOK market')
    Ma = CX.load_market(DAY, MID3, [A], root=R0, book_manifest=man, require_book=False)
    r = CX.btc_ret(Ma, D0 + 20_000)
    check('no previous-day grid: pre-ws instants are btc_missing (pad unavailable)', r['status'][0] == 'btc_missing' and r['cause'][0] == 'outside_grid', r)
    b = CX.book(Ma, [D0 + 20_000, D0 + 22_000])
    check('off BOOK: tob fallback book', b['source'] == 'tob' and close(b['bid_tick'], [40, NAN]) and close(b['ask_tick'], [45, NAN]) and
          eqo(b['status'], ['ok', 'stale']), b)
    check('off BOOK: no depth', Ma.depth is None and CX.level_size(Ma, D0 + 20_000, 'bid', 40)['status'][0] == 'no_depth')
    Mb = CX.load_market(DAY, MID3, [A], root=R2, book_manifest=man, require_book=False)
    r = CX.btc_ret(Mb, D0 + 20_000)
    check('previous-day grid present: the pad makes the same instant ok (r_1s = 0 across midnight)', r['status'][0] == 'ok' and
          close(r['r_1s'], [0.], 0) and Mb.btc['BT']['G0'] * 100 == D0 - 3_600_000, r)
    check('the previous-day file is an input', any(k.endswith(f'btc_{PREV}.parquet') for k in Mb.inputs))


def test_refusals(M0, R0, R3, inp):
    fr = pq.read_table(R0 / 'depth' / f'{DAY}_{MID}_frames.parquet').to_pandas()
    lv = pq.read_table(R0 / 'depth' / f'{DAY}_{MID}_levels.parquet').to_pandas()
    legs = CX.join_windows(pd.DataFrame(inp['fv_rg1'])[list(CX.FV_READ)], pd.DataFrame(inp['life']))
    legs = legs[legs['mid'] == MID].reset_index(drop=True)
    kw = dict(btc=M0.btc, tob=M0.raw_tob, frames=fr, levels=lv, life_fills=None, orders=None)
    Mx = CX.build_market(DAY, MID, WS, WE, [A, B], legs=legs, **kw)
    check('build_market on the joined legs = load_market positions', same(CX.position(Mx, A, WS + np.arange(0, 70_000, 777)),
                                                                           CX.position(M0, A, WS + np.arange(0, 70_000, 777))))
    raises('tob / depth frame sets differ -> refused', lambda: CX.build_market(DAY, MID, WS, WE, [A], legs=legs, **dict(kw, frames=fr.iloc[1:],
                                                                                levels=lv[lv['fid'] != 0])), 'differ')
    bad = legs.copy()
    bad.loc[bad['sid'] == 'S1', 's'] = 0
    raises('an ok leg with s = 0 -> refused', lambda: CX.build_market(DAY, MID, WS, WE, [A], legs=bad, **kw), 's outside +-1')
    two = pd.concat([legs, legs[legs['sid'] == 'S1'].assign(addr=AL, sid='S1b')], ignore_index=True)
    raises('two spellings of one address -> refused', lambda: CX.build_market(DAY, MID, WS, WE, [A], legs=two, **kw), 'two spellings')
    raises('no BTC day -> refused', lambda: CX.build_market(DAY, MID, WS, WE, [A], legs=legs, **dict(kw, btc=None)), 'a loaded BTC day dict is required')
    raises('a market row on another day in fills_valued -> refused', lambda: CX.load_market(DAY, MID, [A], root=R3, book_manifest=R0 / 'manifest.json'),
           'other than its own')
    raises('identities ranks differ from TOP3 -> refused', lambda: CX.top3(R3), 'differ from pf_mt_decision.TOP3')
    raises('an address listed twice -> refused', lambda: CX.load_market(DAY, MID, [A, AL], root=R0, book_manifest=R0 / 'manifest.json'), 'listed twice')


def falsifier_fns():
    r1, sp, tc, lz, ps, al = (CX.feat_btc('r_1s'), CX.feat_book('spread'), CX.feat_touch('bid', 44), CX.feat_level('bid', 44),
                              CX.feat_position(A, 'certain'), CX.feat_btc_align('r_5s', 1))
    F = [('btc_ret', lambda M, t: CX.btc_ret(M, t, horizons=(1000, 5000, 30000, 2000, 120_000))), ('btc_state', CX.btc_state),
         ('book', CX.book), ('book_loose', lambda M, t: CX.book(M, t, strict=False)), ('book_tob', CX.book_tob),
         ('level_bid44', lambda M, t: CX.level_size(M, t, 'bid', 44)), ('level_ask46_loose', lambda M, t: CX.level_size(M, t, 'ask', 46, strict=False)),
         ('touch_bid44', lambda M, t: CX.touch_distance(M, t, 'bid', 44)), ('touch_ask46', lambda M, t: CX.touch_distance(M, t, 'ask', 46)),
         ('book_change', lambda M, t: CX.book_change(M, t, 'bid')),
         ('pos_hi', lambda M, t: CX.position(M, A, t, 'hi')), ('pos_lo', lambda M, t: CX.position(M, A, t, 'lo')),
         ('pos_settle', lambda M, t: CX.position(M, A, t, 'settle')), ('pending', lambda M, t: CX.pending(M, A, t)),
         ('pos_B', lambda M, t: CX.position(M, B, t)), ('time_left', CX.time_left), ('tau', CX.tau), ('phase', CX.phase)]
    for nm, f in (('r1', r1), ('spread', sp), ('touch', tc), ('level', lz), ('pos', ps), ('align', al)):
        F.append((f'over_window_{nm}', lambda M, t, f=f: CX.over_window(M, f, t[::40] - 3_000, t[::40])))
        F.append((f'over_window_{nm}_D1000', lambda M, t, f=f: CX.over_window(M, f, t[::40] - 3_000, t[::40], D_ms=1000)))
    F.append(('level_series', lambda M, t: CX.level_series(M, 'bid', 44, WS + 3_000, int(t.max()))))
    F.append(('pending_legs', lambda M, t: [CX.pending_legs(M, A, x)[['sid', 'role', 's', 'q', 'hi_ms', 'lo_hard_ms']] for x in t[::500]]))
    return F


def test_falsifier(M0, M1, Q):
    for nm, f in falsifier_fns():
        check(f'look-ahead: {nm} at t <= t* unchanged by perturbing everything after t*', same(f(M0, Q), f(M1, Q)))
    leaks = [('btc_ret +150 ms', lambda M, t: CX.btc_ret(M, t + 150)), ('book +150 ms', lambda M, t: CX.book(M, t + 150)),
             ('position +5 s', lambda M, t: CX.position(M, A, t + 5_000))]
    for nm, f in leaks:
        check(f'falsifier fires on a planted leak ({nm})', not same(f(M0, Q), f(M1, Q)))
    Qa = TSTAR + np.array([200, 1_000, 5_000, 30_000, 60_000])
    for nm, f in (('btc_ret', CX.btc_ret), ('book', CX.book), ('position', lambda M, t: CX.position(M, A, t))):
        check(f'after t* the outputs change ({nm}: the perturbation is real)', not same(f(M0, Qa), f(M1, Qa)))


def main():
    cpu0, w0 = time.process_time(), time.time()
    tmp = Path(tempfile.mkdtemp(prefix='chain_ctx_selftest_'))
    try:
        test_guards()
        inp = base_inputs()
        R0, R1, R2, R3 = tmp / 'r0', tmp / 'r1', tmp / 'r2', tmp / 'r3'
        write_root(R0, inp)
        HC = {}
        M0 = CX.load_market(DAY, MID, [A, B, CC], root=R0, book_manifest=R0 / 'manifest.json', hash_cache=HC)
        Q = np.unique(np.r_[np.arange(WS - 300_000, TSTAR + 1, 37), TSTAR, [WS + r for r, *_ in frames_spec() if WS + r <= TSTAR],
                            WS + np.array([8_000, 9_500, 10_000, 10_001, 18_000, 20_000, 20_001])]).astype(np.int64)
        test_load(M0, inp)
        test_btc(M0, R0, Q)
        test_book(M0)
        test_levels(M0)
        test_positions(M0)
        test_time(M0)
        test_over_window(M0)
        test_matched(M0)
        test_helpers()
        R2.mkdir()
        (R2 / 'btc').mkdir()
        write_btc(R2 / 'btc' / f'btc_{PREV}.parquet', btc_day(CX.day_ms(PREV), const=True))
        (R2 / 'btc' / f'btc_{DAY}.parquet').symlink_to(R0 / 'btc' / f'btc_{DAY}.parquet')
        for nm in (f'markets_{DAY}.parquet', f'tob_{DAY}.parquet', 'econ'):
            (R2 / nm).symlink_to(R0 / nm)
        test_pad_and_nonbook(R0, R2)
        R3.mkdir()
        for nm in (f'markets_{DAY}.parquet', f'tob_{DAY}.parquet', 'depth', 'life', 'btc'):
            (R3 / nm).symlink_to(R0 / nm)
        (R3 / 'econ').mkdir()
        badfv = [dict(r) for r in inp['fv_rg1']] + [fv_row('S99', 0, 'maker', A, 1, 1., .5, WS + 1_000, day='20261025')]
        write_pq(R3 / 'econ' / 'fills_valued.parquet', [table(inp['fv_rg0'], FVS), table(badfv, FVS), table(inp['fv_rg2'], FVS)])
        (R3 / 'econ' / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(top10_shares=[dict(addr=a, rank=3 - i) for i, a in enumerate(CX.TOP3)]))))
        test_refusals(M0, R0, R3, inp)
        write_root(R1, perturb(inp, TSTAR))
        M1 = CX.load_market(DAY, MID, [A, B, CC], root=R1, book_manifest=R0 / 'manifest.json')
        test_falsifier(M0, M1, Q)
        cost = dict(load_cpu_s=M0.cost['cpu_s'], load_wall_s=M0.cost['wall_s'])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RES)
    print(json.dumps(dict(version=CX.VERSION, passed=n_ok, total=len(RES), failed=[n for n, ok in RES if not ok], cpu_s=round(time.process_time() - cpu0, 2),
                          wall_s=round(time.time() - w0, 2), vmhwm_mib=round(LC.vmhwm_mib(), 1), synthetic_load=cost), indent=1))
    print(f'chain_context selftest: {n_ok}/{len(RES)} passed')
    return 0 if n_ok == len(RES) else 1


if __name__ == '__main__':
    sys.exit(main())
