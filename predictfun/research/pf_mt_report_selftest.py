"""Selftest of pf_mt_report (MAKER-TRACE-RECONSTRUCTION-v1 report; session aea5bfb9, 2026-11-22). SYNTHETIC data only, written to a
temporary directory in the real schemas (pf_mt_common.SCHEMAS, pf_mt_extract.MARKETS, pf_mt_econ FV / MM / MD / SEG); every expected number
below is computed by hand in its comment (never by the module). Runs the whole CLI path (run -> every writer -> publication) on a main
world and a variant, then refusals with the reason asserted. World (YES space; times = ms offsets from the market's ws):
 sets   exante_sh = A (rank 1), B (2), CC (3); exante_pnl = CC, B; same day: shares {B, A}, P&L {G}; coverage field = G; T takes only.
 BOOK   D1 20261024: 101, 102; D2 20261025: 201, 202; D3 20261026: 301; D4 20261027: none (401 / 402 are FILL markets); GT 801 (Nov 8).
 R1     T1 / T2 (den, violations, certain) per market below: pooled T1 190 / 2 / 1 -> INSUFFICIENT; T2 250 / 2 / 1 -> 0.008 supported.
 R2     one constrained exhausted-only removal defect per scenario -> defective; (removal, constrained) demoted.
 variant T2 of 801 has 3 violations (pooled 5 / 250 = 0.02 -> contradicted -> coverage scenario S_NOPRIO); no GT defect (no demotion).
 additions  D4 market 401: ten A-maker / B-taker 1-sh legs (a one-sided member pair) and one 4-sh A self-trade; END mids per market (MEND) with
        net positions (NPV) so resid_resol = NP (V - m_END); ranks per day (RANKS); declared econ columns reb25_res / reb25_fill / rt_pnl_res,
        segment dims depth / sweep / cp / boost / self, rewards_market.parquet; truth v2 with per-run clock bands, size.ub, defect_rows.
Usage (run_light): pf_mt_report_selftest.py"""
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_mt_common as C  # noqa: E402
import pf_mt_econ as E  # noqa: E402
import pf_mt_extract as X  # noqa: E402
import pf_mt_report as R  # noqa: E402

need = C.need
N_OK = [0]
ID, GT = C.ID_DAY, C.GT_DAY
D1, D2, D3, D4 = C.MEASURE_DAYS
NAN = float('nan')
A, B, CC, G, T = '0x' + 'a' * 40, '0x' + 'b' * 40, '0x' + 'c' * 40, '0x' + '1' * 40, '0x' + 'e' * 40
OWN = '0xba14f115ccf30e976f7639594fdee7b36c85ced8'
SC = tuple(C.SCENARIOS)
U, OB, K, S = C.UNRESOLVED, C.OBSERVED, C.CONSTRAINED, C.SENSITIVE
EXH, CE, CEE = 'exhausted', 'cancelled_or_expired|exhausted', 'cancelled_or_expired|end_censored|exhausted'
BOOK = {D1: [101, 102], D2: [201, 202], D3: [301], D4: []}
WS = {101: (D1, 1), 102: (D1, 2), 201: (D2, 1), 202: (D2, 2), 203: (D2, 3), 301: (D3, 1), 401: (D4, 1), 402: (D4, 2), 801: (GT, 1), 901: (ID, 1)}
# hash: (mid, signer, side, tick, fills [(sid, mk_idx, shares, hi, lo)], classes (post, persist, queue, removal, size),
#        post_lo, removal (set, hi), ahead (lo, hi), size_exact)  - a LIST = per scenario (S_FIFO, S_FIFO_LO, S_NOPRIO), else all equal
ORD = {
    'h101a': (101, A, 'bid', 50, [('s1011', 0, 10., 10000, 9000), ('s1012', 0, 10., 20000, 19500), ('s1013', 1, 10., 30000, 29000)],
              (K, OB, S, K, OB), 5000, (EXH, 30000), [(0., 3.), (0., 5.), (NAN, NAN)], NAN),
    'h101b': (101, B, 'ask', 52, [('s1014', 0, 5., 40000, -1)], (U, U, U, U, OB), -1, (CEE, -1), (NAN, NAN), NAN),
    'h101g': (101, G, 'bid', 49, [('s1013', 0, 4., 30000, 29000)], (S, U, S, S, OB), [20000, 20000, -1], [(EXH, 30000), (EXH, 30000), (CEE, -1)],
              [(10., 10.), (10., 12.), (NAN, NAN)], NAN),
    'h101g2': (101, G, 'bid', 48, [('s1015', 0, 2., 50000, 49000)], (U, U, U, U, OB), -1, (CEE, -1), (NAN, NAN), NAN),
    'h102a': (102, A, 'ask', 60, [('s1021', 0, 7., 10000, 9000), ('s1022', 0, 3., 15000, 14000)], (K, OB, U, S, K), 2000,
              [(CE, 20000), (CE, 21000), (CEE, -1)], (NAN, NAN), 10.),
    'h102a2': (102, A, 'ask', 61, [('s1026', 0, 2., 16000, 15000)], (U, U, U, S, OB), -1, [(CE, 17000), (CE, 17500), (CEE, -1)], (NAN, NAN), NAN),
    'h102a3': (102, A, 'bid', 57, [('s1027', 0, 1., 17000, 16000)], (K, U, U, S, OB), 1000, [(EXH, 17000), (EXH, 17000), (CE, 18000)], (NAN, NAN), NAN),
    'h102a4': (102, A, 'bid', 56, [('s1028', 0, 1., 18000, 17000), ('s1029', 0, 1., 19000, 18000)], (K, OB, U, K, OB), 500, (EXH, 19000), (NAN, NAN), NAN),
    'h102b': (102, B, 'bid', 58, [('s1023', 0, 2., 11000, 10000), ('s1024', 0, 2., 12000, 11000), ('s1025', 0, 1., 13000, 12000)],
              (U, OB, S, U, OB), -1, (CEE, -1), [(0., 0.), (0., 1.), (NAN, NAN)], NAN),
    'h102g': (102, G, 'ask', 62, [('s1030', 0, 1., 21000, 20000)], (U, U, U, U, OB), -1, (CEE, -1), (NAN, NAN), NAN),
    'h201b': (201, B, 'ask', 47, [('s2011', 0, 3., 10000, 9000), ('s2012', 0, 3., 11000, 10000), ('s2013', 0, 1., 12000, 11000)],
              (U, OB, U, U, OB), -1, (CEE, -1), (NAN, NAN), NAN),
    'h202a': (202, A, 'bid', 45, [('s2021', 0, 1., 10000, 9000), ('s2022', 0, 1., 20000, 19000)], (K, OB, S, K, OB), 3000, (EXH, 20000),
              [(0., 5.), (0., 6.), (NAN, NAN)], NAN),
    'h202a2': (202, A, 'bid', 45, [('s2023', 0, 1., 30000, 29000)], (U, U, U, U, OB), -1, (CEE, -1), (NAN, NAN), NAN),
    'h202g': (202, G, 'ask', 46, [('s2024', 0, 1., 25000, 24000)], (U, U, U, U, OB), -1, (CEE, -1), (NAN, NAN), NAN),
    'h301c': (301, CC, 'bid', 30, [('s3011', 0, 4., 10000, 9000)], (K, U, U, K, K), 2000, (EXH, 10000), (NAN, NAN), 4.),
    'h301g': (301, G, 'ask', 31, [('s3012', 0, 6., 20000, 19000)], (U, U, U, U, OB), -1, (CEE, -1), (NAN, NAN), NAN),
    'h801o': (801, OWN, 'bid', 50, [('s8011', 0, 5., 10000, 9000)], (K, U, U, K, OB), 1000, (EXH, 10000), (NAN, NAN), NAN),
}
EXTRA_LEGS = {'s4011': (401, A, 'bid', 40, 'h401a', 1.), 's9011': (901, A, 'ask', 50, 'h901a', 2.),     # non-lifecycle legs (FILL / ID markets)
              **{f's40{i}': (401, A, 'bid', 41, 'h401a2', 1.) for i in range(12, 22)},              # ten A-maker / B-taker legs: a one-sided pair
              's4022': (401, A, 'ask', 42, 'h401a3', 4.),                                          # an A self-trade (4 sh)
              's2031': (203, G, 'bid', 50, 'h203g', 1.)}                                           # market 203 resolves as a TIE (V = 0.5)
TAKER = {'s1015': A, 's1030': B, 's2024': B, 's4022': A, **{f's40{i}': B for i in range(12, 22)}}   # else T
MEND = {101: .75, 201: .5, 301: 1., 401: .25, 203: .5}                                # END mids of the maker_market markets (else 0.6)
VMK, STRES = {203: .5}, {203: 'tie'}                                                  # V (else 1) and st_res (else ok) per market
NPV = {A: [2., 2., 2., 2.], B: [-1., 0., 1., 0.], CC: [0.] * 4, G: [1.] * 4, T: [-2., -3., -4., -3.]}   # net positions per day
RANKS = {ID: dict(sh={A: 1, B: 2, CC: 3, G: 4}, pnl={CC: 1, B: 2, A: 3, G: 4}), D1: dict(sh={A: 1, B: 2, G: 3, CC: 12}, pnl={A: 1, G: 2, B: 3, CC: 4}),
         D2: dict(sh={A: 1, B: 2, CC: 3, G: 4}, pnl={A: 1, B: 2, CC: 3, G: 5}), D3: dict(sh={B: 1, A: 2, CC: 3, G: 4}, pnl={A: 1, B: 2, G: 3, CC: 15}),
         D4: dict(sh={A: 1, B: 2, CC: 3, G: 4}, pnl={B: 1, A: 2, CC: 3, G: 4})}
FFR = {('s1021', 0): (1, 2), ('s1022', 0): (3, 4), ('s1023', 0): (2, 3), ('s2011', 0): (1, 2)}   # (f_before, f_after) fids, else -1
NETS = {'h101a': [1., 2., -.5], 'h101b': [-4.], 'h101g': [.25], 'h101g2': [.5], 'h102a': [3., -1.], 'h102a2': [0.], 'h102a3': [0.], 'h102a4': [0., 0.],
        'h102b': [.125, .125, .125], 'h102g': [0.], 'h201b': [5., 0., 100.], 'h202a': [1., 1.], 'h202a2': [-.75], 'h202g': [0.], 'h301c': [-3.],
        'h301g': [0.], 'h401a': [0.], 'h901a': [0.], 'h401a2': [.25] * 10, 'h401a3': [0.], 'h203g': [0.]}
BAD_FV = {('s2013', 0)}                                                                 # fills_valued status src_status (its 100.0 is excluded)
PTV = {101: ((60, 1, 0), (100, 1, 1)), 102: ((40, 0, 0), (50, 0, 0)), 201: ((50, 1, 1), (60, 1, 0)), 202: ((0, 0, 0), (0, 0, 0)),
       301: ((30, 0, 0), (20, 0, 0)), 801: ((10, 0, 0), (20, 0, 0))}
LINKS = [(101, G, 'bid', 'reprice', 'h101g', 'h101g2', 49, 48, 30000, 50000), (102, A, 'ask', 'reprice', 'h102a', 'h102a2', 60, 61, 15000, 16000),
         (102, A, 'bid', 'reprice', 'h102a3', 'h102a4', 57, 56, 17000, 18000), (202, A, 'bid', 'refill', 'h202a', 'h202a2', 45, 45, 20000, 30000)]
# frames (fid, offset) and tob rows (offset, recv extra ns, best bid, bid sh, best ask, ask sh) of the timeline markets; levels (fid, side, tick, sh, status)
FRAMES = {102: [0, 9000, 10000, 14000, 15000, 30000], 201: [0, 9000, 10000], 301: [0]}
TOBV = {102: [(0, 0, .55, 10., .60, 5.), (9000, 0, .56, 4., .60, 7.), (9000, 5, .565, 4., .60, 7.), (10000, 0, .58, 4., .61, 2.), (14000, 0, .58, 2., .60, 3.),
              (15000, 0, .58, 2., .61, 2.), (30000, 0, .57, 1., .62, 1.)],
        201: [(0, 0, .40, 1., .47, 6.), (9000, 0, .40, 1., .47, 6.), (10000, 0, .40, 1., .48, 1.)]}
LEVELS = {102: [(1, 'ask', 60, 7., 'ok'), (1, 'bid', 58, 4., 'ok'), (2, 'bid', 58, 4., 'ok'), (3, 'ask', 60, 3., 'off_grid'), (3, 'bid', 58, 2., 'ok'),
                (4, 'ask', 61, 2., 'ok')], 201: [(1, 'ask', 47, 6., 'ok')]}
# maker_market per address: (n_mk, sh_mk, n_tk, sh_tk, notional, gross_mk[4], gross_tk[4], fee, rt[4], drift, v_spr_mk, v_spr_tk)
MMV = {A: (3, 30., 0, 0., 15., [4., 2., 3., 1.], [0.] * 4, 0., [1.] * 4, .5, .5, 0.),
       B: (2, 10., 0, 0., 5., [2.] * 4, [0.] * 4, .5, [2., .5, 1., 1.5], 0., .25, 0.),
       CC: (1, 4., 0, 0., 2., [-1., 1., -1., 1.], [0.] * 4, 0., [0.] * 4, 0., 0., 0.),
       G: (1, 6., 0, 0., 3., [1., -2., 1., 1.], [0.] * 4, 0., [.5] * 4, 0., .25, 0.),
       T: (0, 0., 7, 50., 25., [0.] * 4, [-6., -3., -5., -5.], .25, [0.] * 4, 0., 0., -1.)}
MKT = {D1: 101, D2: 201, D3: 301, D4: 401}
SETS = dict(exante_shares=('ex_ante', {d: {A, B, CC} for d in C.MEASURE_DAYS}), exante_pnl=('ex_ante', {d: {CC, B} for d in C.MEASURE_DAYS}),
            sameday_shares=('descriptive', {d: {A, B} for d in C.MEASURE_DAYS}), sameday_pnl=('descriptive', {d: {G} for d in C.MEASURE_DAYS}))
PA = {'str': pa.string(), 'int64': pa.int64(), 'float64': pa.float64(), 'bool': pa.bool_()}
DEF = {'str': '', 'int64': -1, 'float64': NAN, 'bool': False}


def ok(cond, msg):
    need(bool(cond), f'selftest assertion failed: {msg}')
    N_OK[0] += 1


def close(a, b, msg, tol=1e-9):
    ok(a is not None and not (isinstance(a, float) and math.isnan(a)) and abs(float(a) - float(b)) <= tol, f'{msg}: {a} != {b}')


def refuses(fn, frag, msg):
    try:
        fn()
    except ValueError as e:
        ok(frag in str(e), f'{msg}: refused for another reason: {e}')
        return
    ok(False, f'{msg}: not refused')


def per(v):
    return dict(zip(SC, v)) if isinstance(v, list) else {s: v for s in SC}


def dms(d):
    return int(dt.datetime.strptime(d, '%Y%m%d').replace(tzinfo=dt.timezone.utc).timestamp()) * 1000


def ws_ms(m):
    return dms(WS[m][0]) + WS[m][1] * 3_600_000


def write(path, spec, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {c: [r.get(c, 'ok' if c == 'status' else DEF[t]) for r in rows] for c, t in spec.items()}
    pq.write_table(pa.table({c: pa.array(v, type=PA[spec[c]]) for c, v in cols.items()}), path)


def legs_all():
    """Every maker leg: (day, mid, sid, mk_idx, hash, signer, side, tick, shares, hi_off, lo_off)."""
    L = []
    for h, (m, sg, sd, tk, fills, *_r) in ORD.items():
        L += [(WS[m][0], m, sid, ix, h, sg, sd, tk, sh, hi, lo) for sid, ix, sh, hi, lo in fills]
    L += [(WS[m][0], m, sid, 0, h, sg, sd, tk, sh, 5000, 4000) for sid, (m, sg, sd, tk, h, sh) in EXTRA_LEGS.items()]
    return sorted(L, key=lambda x: (x[0], x[1], x[2], x[3]))


def world(root, contradicted=False, gt_defect=True):
    OUT = root / 'OUT'
    L = legs_all()
    pos = {}
    for x in L:
        pos.setdefault(x[4], []).append(x)
    # ---- extract tables + receipts + inventories ----
    for d in R.DAYS:
        role = 'ID' if d == ID else ('GT' if d == GT else 'FILL')
        Ld = [x for x in L if x[0] == d]
        sids = sorted({x[2] for x in Ld})
        recs = [dict(day=d, mid=next(x[1] for x in Ld if x[2] == s), sid=s, settle_ms=ws_ms(next(x[1] for x in Ld if x[2] == s)), n_makers=1,
                     tk_hash=f'tk_{s}', tk_signer=TAKER.get(s, T), status='ok') for s in sids]
        if d == D1:   # two flagged records (no legs): amount reconciliation 16 / 17, taker price 16 / 17
            recs += [dict(day=d, mid=101, sid='sX1', status='recon_mk_sum'), dict(day=d, mid=101, sid='sX2', status='conflict+recon_price')]
        legs = [dict(day=d, mid=x[1], sid=x[2], mk_idx=x[3], mk_hash=x[4], mk_signer=x[5], y_side=x[6], y_ptick=x[7], shares=x[8],
                     status='off_grid+boundary_censored' if x[2] == 's1014' else 'ok') for x in Ld]
        mids = sorted({x[1] for x in Ld} | ({402} if d == D4 else set()))
        mk = [dict(day=d, mid=m, slug=f'btc-updown-5m-{ws_ms(m) // 1000}', ws=ws_ms(m) // 1000, we=ws_ms(m) // 1000 + 300, depth=int(m in sum(BOOK.values(), []) or m == 801),
                   status='boundary_censored' if m == 402 else 'ok') for m in mids]
        tob = []
        if role != 'ID':
            for m in mids:
                for off, rx, bb, bbs, ba, bas in TOBV.get(m, [(0, 0, .45, 1., .55, 1.)]):
                    tob.append(dict(day=d, mid=m, upd_ms=ws_ms(m) + off, recv_ns=(ws_ms(m) + off + 120) * 1_000_000 + rx, conn='c1', best_bid=bb,
                                    best_bid_sh=bbs, best_ask=ba, best_ask_sh=bas, status='ok'))
        for k, rows, spec in (('records', recs, C.SCHEMAS['records']), ('legs', legs, C.SCHEMAS['legs']), ('markets', mk, X.MARKETS)) + \
                ((('tob', tob, C.SCHEMAS['tob']),) if role != 'ID' else ()):
            write(OUT / f'{k}_{d}.parquet', spec, rows)
        depth = sorted(m for m in mids if m in sum(BOOK.values(), []) or m == 801)
        for m in depth:
            fr = [dict(mid=m, fid=i, upd_ms=ws_ms(m) + off, recv_ns=(ws_ms(m) + off) * 1_000_000, status='ok') for i, off in enumerate(FRAMES.get(m, [0]))]
            write(OUT / 'depth' / f'{d}_{m}_frames.parquet', C.SCHEMAS['frames'], fr)
            write(OUT / 'depth' / f'{d}_{m}_levels.parquet', C.SCHEMAS['levels'],
                  [dict(mid=m, fid=f, side=s, ptick=p, shares=sh, status=st) for f, s, p, sh, st in LEVELS.get(m, [])])
        cnt = lambda rows: dict(Counter(r.get('status', 'ok') for r in rows))  # noqa: E731
        sc = dict(records=cnt(recs), legs=cnt(legs), markets=cnt(mk), frames={'ok': 1}, levels={})
        counts = dict(records=len(recs), legs=len(legs), markets=len(mk))
        if role != 'ID':
            sc['tob'], counts['tob'] = cnt(tob), len(tob)
        (OUT / f'extract_{d}.json').write_text(json.dumps(dict(day=d, role=role, asof='2026-11-22T10:00:00+00:00', counts=counts, status_counts=sc,
                                                                population=dict(n_markets=len(mids), depth_mids=depth if role != 'ID' else []),
                                                                envelope=dict(margin_ms=60000, lag_p999_ms=4321.5, rule='text'))))
        inv = dict(day=d, objects={}, leaves={}, counters={}, rest=dict(ids={}, clocks_and_lags_ms={}), ws={})
        if d == D1:
            inv = dict(day=d, objects={'record': [dict(keys=['amountFilled', 'executedAt', 'makers'], n=100)], 'maker.fee': [dict(keys=['amount', 'type'], n=50)]},
                       leaves={'record.executedAt': dict(types={'str': 100}, formats={'iso_frac3_Z': 100})},
                       counters={'executedAt_after_seconds': {'.000Z': 98, '.500Z': 2},
                                 'makers_yes_price_order (multi-maker records)': {'single_price': 10, 'best_first': 2, 'worst_first': 6, 'neither': 2},
                                 'maker_fee': {'legs': 50, 'nonzero': 0}, 'maker_fee_type': {'SHARES': 50},
                                 'taker_fee_type | taker Bid/indexSet 1/YES bid': {'SHARES': 30, 'COLLATERAL': 20},
                                 'priceExecuted_vs_maker_avg': {'exact': 99, 'differs': 1}},
                       rest=dict(ids=dict(unique_keys=100, order_hashes=dict(taker_hashes=90)),
                                 clocks_and_lags_ms={'settle_minus_exec_ms': dict(p0=-2000., p50=-800., p100=0., n=100)},
                                 receipt_lag_ms=dict(p50=900., n=100)), ws={'recv_minus_upd_ms': dict(p50=126., n=1000)})
        (OUT / f'inventory_{d}.json').write_text(json.dumps(inv))
    # ---- lifecycle ----
    for d in R.LIFE_DAYS:
        for m in (BOOK.get(d) or ([801] if d == GT else [])):
            O, F = [], []
            for h, (mid, sg, sd, tk, fills, cl, plo, rm, ah, sx) in ORD.items():
                if mid != m:
                    continue
                w = ws_ms(m)
                his, los = [w + f[3] for f in fills], [w + f[4] for f in fills if f[4] >= 0]
                n, cum = len(fills), sum(f[2] for f in fills)
                p_lo, p_hi = (min(his), max(los)) if (n >= 2 and los and max(los) >= min(his)) else (-1, -1)
                for s in SC:
                    pl, (rs, rh), (al, ahh) = per(plo)[s], per(rm)[s], per(ah)[s]
                    O.append(dict(scenario=s, mid=m, mk_hash=h, mk_signer=sg, y_side=sd, y_ptick=tk, n_fills=n, cum_shares=cum, first_hi_ms=min(his),
                                  last_hi_ms=max(his), size_lb=cum, size_exact=sx, size_class=cl[4], post_lo_ms=w + pl if pl >= 0 else -1,
                                  post_hi_ms=min(his), post_class=cl[0], post_method='capacity' if pl >= 0 else 'none', persist_lo_ms=p_lo, persist_hi_ms=p_hi,
                                  persist_class=cl[1], ahead_lo=al, ahead_hi=ahh, queue_class=cl[2], removal_set=rs, removal_lo_ms=max(los) if los else -1,
                                  removal_hi_ms=w + rh if rh >= 0 else -1, removal_class=cl[3], removal_method='fifo_price' if rs == EXH else 'end_open'))
                for sid, ix, sh, hi, lo in fills:
                    fb, fa = FFR.get((sid, ix), (-1, -1))
                    F.append(dict(mid=m, sid=sid, mk_hash=h, mk_idx=ix, y_side=sd, y_ptick=tk, shares=sh, hi_ms=w + hi, lo_ms=w + lo if lo >= 0 else -1,
                                  t_method='pend_inc' if lo >= 0 else 'none', f_before=fb, f_after=fa, status='ok' if lo >= 0 else 'fill_lo_unbounded'))
            lk = [dict(mid=mm, mk_signer=sg, y_side=sd, kind=kd, prev_hash=ph, next_hash=nh, prev_ptick=pt, next_ptick=nt, prev_last_hi_ms=ws_ms(mm) + a,
                       next_first_hi_ms=ws_ms(mm) + b, gap_ms=b - a, x_latest=True, label='descriptive', status='ok')
                  for mm, sg, sd, kd, ph, nh, pt, nt, a, b in LINKS if mm == m]
            tag = f'{d}_{m}'
            write(OUT / 'life' / f'{tag}_orders.parquet', C.SCHEMAS['orders'], O)
            write(OUT / 'life' / f'{tag}_fills.parquet', C.SCHEMAS['fills'], F)
            write(OUT / 'life' / f'{tag}_x_links.parquet', dict(mid='int64', mk_signer='str', y_side='str', kind='str', prev_hash='str', next_hash='str',
                                                                prev_ptick='int64', next_ptick='int64', prev_last_hi_ms='int64', next_first_hi_ms='int64',
                                                                gap_ms='int64', x_latest='bool', label='str', status='str'), lk)
            o1 = [o for o in O if o['scenario'] == 'S_FIFO']
            cov = {f: {c: dict(orders=sum(o[col] == c for o in o1), executions=sum(o['n_fills'] for o in o1 if o[col] == c),
                               shares=round(sum(o['cum_shares'] for o in o1 if o[col] == c), 6)) for c in C.CLASSES} for f, col in R.FIELDS.items()}
            t1, t2 = PTV[m]
            if contradicted and m == 801:
                t2 = (20, 3, 0)
            rt = lambda v, n: round(v / n, 6) if n else None  # noqa: E731
            pt = {s: dict(T1_price=dict(makers=t1[0], violations=t1[1], rate=rt(t1[1], t1[0]), violations_certain=t1[2], rate_certain=rt(t1[2], t1[0]), ties=0),
                          T2_time=dict(groups=t2[0], violations=t2[1], rate=rt(t2[1], t2[0]), violations_certain=t2[2], rate_certain=rt(t2[2], t2[0]),
                                       survivor_groups=1, no_survivor_groups=0, groups_tie=0, groups_invalid=0)) for s in ('S_FIFO', 'S_FIFO_LO')}
            (OUT / 'life' / f'{tag}.json').write_text(json.dumps(dict(mid=m, priority_tests=pt, coverage=cov,
                                                                      inputs=dict(legs=len(F), frames=len(FRAMES.get(m, [0])), frames_excluded=0, unknown_cells=1))))
    # ---- econ: fills_valued ----
    fv = []
    for d_, m, sid, ix, h, sg, sd, tk, sh, hi, lo in L:
        if d_ == GT:
            continue
        net = NETS[h][[x[2] for x in pos[h]].index(sid)]
        st = 'src_status' if (sid, ix) in BAD_FV else 'ok'
        s_ = 1 if sd == 'bid' else -1
        base = dict(day=d_, mid=m, sid=sid, mk_idx=ix, px_rank=0, kind='regular', own_index_set=1, q=sh, p=tk / 100., settle_ms=ws_ms(m) + hi,
                    t_ms=ws_ms(m) + (lo if lo >= 0 else hi), t_status='lo' if lo >= 0 else 'hi_lo_unbounded', t_rel_s=hi / 1000., tseg='w000_060',
                    pseg='d_0.0_0.1', V=VMK.get(m, 1.), st_res=STRES.get(m, 'ok'), st_spr='ok', st_1s='ok', st_10s='ok',
                    st_60s='beyond_end' if sid == 's1030' else 'ok', st_end='ok',
                    m_end=MEND.get(m, .6), fee=0., fee_type='SHARES', fee_dy=0., fee_dc=0., fee_fill=0., fee_res=0., src_status=st, status=st,
                    self_trade=sg == TAKER.get(sid, T))
        fv.append(dict(base, role='maker', addr=sg, hash=h, cp_addr=TAKER.get(sid, T), y_side=sd, s=s_, gross_res=net, net_res=net, v_spr=net / 2,
                       life_rep='na_not_book', life_exh='na_not_book', life_front='na_not_book', life_status='ok'))
        fv.append(dict(base, role='taker', addr=TAKER.get(sid, T), hash=f'tk_{sid}', cp_addr=sg, y_side='ask' if sd == 'bid' else 'bid', s=-s_,
                       gross_res=-net, net_res=-net, v_spr=-net / 2, life_rep='na_taker', life_exh='na_taker', life_front='na_taker', life_status='na_taker'))
    write(OUT / 'econ' / 'fills_valued.parquet', E.FV, fv)
    counts = {}
    for r in fv:
        for c in R.FV_ST:
            counts.setdefault(r['day'], {}).setdefault(c, Counter())[str(r.get(c, ''))] += 1
    # ---- econ: maker_market / maker_day / segments ----
    mm = []
    for k, d in enumerate(C.MEASURE_DAYS):
        for a, (nmk, smk, ntk, stk, notl, gmk, gtk, fee, rt_, drift, vmk, vtk) in MMV.items():
            g = gmk[k] + gtk[k]
            npos, resol = NPV[a][k], NPV[a][k] * (1. - MEND[MKT[d]])                    # resid_resol = net_pos (V - m_END), V = 1
            r = dict(day=d, mid=MKT[d], addr=a, population='FILL', book=MKT[d] in sum(BOOK.values(), []), res_status='ok', end_status='ok',
                     is_maker=nmk > 0, n_mk=nmk, sh_mk=smk, n_tk=ntk, sh_tk=stk, notional=notl, sh_buy=smk + stk, sh_sell=0., gross_res_mk=gmk[k],
                     gross_res_tk=gtk[k], sh_res=smk + stk, fee_res=fee, fee_fill=fee, net_res=g - fee, n_res_na=0, rt_sh=1., rt_pnl=rt_[k], net_pos=npos,
                     resid_res=g - rt_[k], resid_drift=g - rt_[k] - resol, resid_resol=resol, fifo_status='ok', rew_status='not_book', n_excl=0, status='ok',
                     reb25_res=smk / 100., reb25_fill=smk / 50.)
            for h in E.VALS:
                for role, v, sh in (('mk', vmk if h != 'end' else gmk[k], smk), ('tk', vtk if h != 'end' else gtk[k], stk)):
                    r[f'v_{h}_{role}'], r[f'sh_{h}_{role}'] = v, sh
            mm.append(r)
    r = dict(mm[0], day=D4, mid=402, book=False, res_status='no_resolution', fifo_status='no_resolution', n_mk=1, sh_mk=2., n_tk=0, sh_tk=0., notional=1.,
             gross_res_mk=0., gross_res_tk=0., sh_res=0., fee_res=0., fee_fill=0., net_res=0., n_res_na=1, rt_pnl=.75, resid_res=NAN,
             resid_drift=NAN, resid_resol=NAN, net_pos=0., sh_buy=2., addr=A, reb25_res=.02, reb25_fill=.04)
    for h in E.VALS:
        r[f'v_{h}_mk'], r[f'sh_{h}_mk'], r[f'v_{h}_tk'], r[f'sh_{h}_tk'] = 0., 2., 0., 0.
    mm.append(r)
    # T in the TIE market 203 (D2): V = 0.5 = m_END: gross 0.5, round trip 0.25, residual 0.25 = drift 0.25 + resolution 1 x (0.5 - 0.5)
    r = dict(mm[0], day=D2, mid=203, addr=T, book=False, res_status='tie', end_status='ok', fifo_status='ok', is_maker=False, n_mk=0, sh_mk=0., n_tk=1,
             sh_tk=2., notional=1., sh_buy=2., sh_sell=0., gross_res_mk=0., gross_res_tk=.5, sh_res=2., fee_res=0., fee_fill=0., net_res=.5, n_res_na=0,
             rt_sh=1., rt_pnl=.25, net_pos=1., resid_res=.25, resid_drift=.25, resid_resol=0., reb25_res=0., reb25_fill=0.)
    for h in E.VALS:
        r[f'v_{h}_mk'], r[f'sh_{h}_mk'], r[f'v_{h}_tk'], r[f'sh_{h}_tk'] = 0., 0., (.5 if h == 'end' else 0.), 2.
    mm.append(r)
    for r in mm:
        for k_ in E.REW:
            r[k_] = 0 if k_ in E.INTS else 0.
    write(OUT / 'econ' / 'maker_market.parquet', dict(E.MM, **{c: 'float64' for c in R.ECON_NEW_MM}), mm)
    md = []
    for d in (ID,) + C.MEASURE_DAYS:
        for a in (A, B, CC, G, T):
            x = [r for r in mm if r['day'] == d and r['addr'] == a]
            md.append(dict({c: 0. if E.MD[c] == 'float64' else 0 for c in E.MD if E.MD[c] in ('float64', 'int64')}, day=d, addr=a,
                           population='ID' if d == ID else 'FILL', sh_mk=sum(r['sh_mk'] for r in x) if d != ID else (0. if a == T else 1.),
                           net_res=sum(r['net_res'] for r in x), rt_pnl_res=sum(r['rt_pnl'] for r in x if r['res_status'] in ('ok', 'tie')),
                           in_exante_sh=a in (A, B, CC), in_exante_pnl=a in (CC, B),
                           in_sameday_sh=d != ID and a in (A, B), in_sameday_pnl=d != ID and a == G,
                           rank_sh=RANKS[d]['sh'].get(a, -1), rank_pnl=RANKS[d]['pnl'].get(a, -1)))
    write(OUT / 'econ' / 'maker_day.parquet', E.MD, md)
    seg = []
    for d in C.MEASURE_DAYS:
        rows = [r for r in mm if r['day'] == d]
        for en, (lab, mem) in SETS.items():
            acc = {}
            for r in rows:
                grp = 'in' if r['addr'] in mem[d] else ('field' if r['addr'] in (A, B, CC, G) else 'nonmaker')
                role = 'maker' if r['n_mk'] > 0 else 'taker'
                rr = 'mk' if role == 'maker' else 'tk'
                v = dict(n=r[f'n_{rr}'], sh=r[f'sh_{rr}'], notional=r['notional'], gross_res=r[f'gross_res_{rr}'], sh_res=r['sh_res'], fee_res=r['fee_res'],
                         fee_fill=r['fee_fill'], net_res=r['net_res'], n_res_na=r['n_res_na'],
                         **{f'{k_}_{h}': r[f'{k_}_{h}_{rr}'] for h in E.VALS for k_ in ('v', 'sh')})
                for dim, bucket in (('all', 'all'), ('tseg', 'w000_060' if r['addr'] in (A, T) else 'w060_120'), ('life_exh', 'na_not_book'),
                                    *((dm, 'x') for dm in R.SEG_NEW_DIMS)):
                    a_ = acc.setdefault((grp, role, dim, bucket), dict(addrs=set(), **{c: 0. for c in E.MET}))
                    a_['addrs'].add(r['addr'])
                    for c in E.MET:
                        a_[c] += v[c] if c in v else float(np.nan_to_num(r.get(c, 0.)))
            for (grp, role, dim, bucket), a_ in acc.items():
                seg.append(dict(day=d, id_set=en, label=lab, grp=grp, role=role, dim=dim, bucket=bucket, n_addr=len(a_.pop('addrs')),
                                **{c: (int(a_[c]) if c in E.INTS else a_[c]) for c in E.MET}, ci='none_G4_lt_5', status='ok'))
    write(OUT / 'econ' / 'segments.parquet', E.SEG, seg)
    top = lambda L_: [dict(rank=i + 1, addr=a, value=1.) for i, a in enumerate(L_)]  # noqa: E731
    (OUT / 'econ' / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(top10_shares=top([A, B, CC]), top10_pnl=top([CC, B])),
                                                                  same_day={d: dict(top10_shares=top([B, A]), top10_pnl=top([G])) for d in C.MEASURE_DAYS})))
    write(OUT / 'econ' / 'rewards_market.parquet', dict(day='str', mid='int64', pool='float64', elig_sh_s_ub='float64', set_sh_s_lb='float64'),
          [dict(day=D1, mid=101, pool=5., elig_sh_s_ub=900., set_sh_s_lb=100.), dict(day=D1, mid=102, pool=5., elig_sh_s_ub=300., set_sh_s_lb=0.)])
    (OUT / 'econ' / 'econ.json').write_text(json.dumps(dict(book=BOOK, asof_end='2026-11-22T11:00:00Z',
                                                            counts={d: {c: dict(v) for c, v in cc.items()} for d, cc in counts.items()},
                                                            diagnostics={d: dict(legs=sum(1 for x in L if x[0] == d), censored_markets=[], lf_rows=3) for d in counts})))
    # ---- truth scores ----
    cls0 = lambda: {c: dict(n=0, failed=0, defects=0, sensitive_misses=0) for c in C.CLASSES}  # noqa: E731

    def scen():
        post = cls0()
        post[K] = dict(n=2, inside=2, failed=0, defects=0, sensitive_misses=0)
        post[S] = dict(n=4, inside=2, disjoint=2, failed=2, defects=0, sensitive_misses=2, gap_ms_failed=dict(min=5., max=9.))
        exo = cls0()
        exo[K] = dict(n=3, truly_filled=2 if gt_defect else 3, not_filled=1 if gt_defect else 0, failed=int(gt_defect), defects=int(gt_defect), sensitive_misses=0)
        ub = cls0()
        ub[K] = dict(n=2, holds=2, failed=0, defects=0, sensitive_misses=0)
        return dict(coverage=dict(orders=dict(n=1, covered=1, rate=1.)), identity=dict(n=1, signer=0, defects=0), posting=post,
                    size=dict(lb=cls0(), exact=cls0(), ub=ub, exact_stated_on_partial_truth=0),
                    removal=dict(exhausted_only=exo, membership=cls0(), window=cls0()),
                    persistence=cls0(), queue=dict(own_front_claims=cls0(), other_makers_front_claims=cls0(), ahead_hi_negative=0),
                    defects_total=int(gt_defect), sensitive_misses_total=2)
    fw = cls0()
    fw[OB] = dict(n=5, inside=5, failed=0, defects=0, sensitive_misses=0)
    prem = ['one constant bot-to-venue offset per run', 'NOT ESTABLISHED: the lifecycle venue book / UUID / collector clocks equal the venue status clock']
    S_ = dict(scored_utc='2026-11-22T11:30:00+00:00', clock_band=dict(per_run=dict(v6=dict(d_lo_ms=1., d_hi_ms=2., premises=prem),
                                                                                  basis=dict(d_lo_ms=3., d_hi_ms=4., premises=prem))),
              population=dict(scored_own_orders=7, scored_legs=9, scored_shares=20., truth_rows=50, lifecycle_markets=1),
              fills=dict(executions=dict(n=5, covered=5), window=fw, defects=0), scenarios={s: scen() for s in SC},
              defects_total=dict(fills=0, **{s: int(gt_defect) for s in SC}),
              defect_rows=[dict(scenario=s, category='removal_exhausted', field='removal_exhausted', mid=801, mk_hash='h801o', cls=K,
                                detail='exhausted-only, truly cancelled') for s in SC] if gt_defect else [],
              unique_defects=dict(n=1, by_field={'removal_exhausted': 1}) if gt_defect else dict(n=0, by_field={}))
    (OUT / 'truth' / 'v2').mkdir(parents=True, exist_ok=True)
    (OUT / 'truth' / 'v2' / 'scores.json').write_text(json.dumps(S_))
    return OUT


def csv(dest, name):
    return pd.read_csv(dest / f'{name}.csv', float_precision='round_trip', keep_default_na=True, dtype={'day': str})


def one(df, msg, **kv):
    m = np.ones(len(df), bool)
    for k, v in kv.items():
        m &= (df[k].astype(str) == str(v)).to_numpy()
    ok(m.sum() == 1, f'{msg}: {int(m.sum())} rows match {kv}')
    return df[m].iloc[0]


def quiet(*a, **k):
    pass


def main():
    hk = lambda h: hashlib.sha256(h.encode()).hexdigest()  # noqa: E731
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        OUT = world(td / 'w1')
        dest = td / 'w1' / 'receipt' / 'outputs'
        RD = R.run(OUT, dest, log=quiet)
        J = json.loads((dest / 'readings.json').read_text())
        ok(sorted(p.name for p in dest.iterdir()) == sorted(list(RD['outputs']) + ['readings.json']), 'outputs: every file published, nothing else')
        ok(not list(dest.parent.glob('.outputs.partial_*')), 'no partial directory left')
        for fn, v in J['outputs'].items():
            ok(v['sha256'] == hashlib.sha256((dest / fn).read_bytes()).hexdigest(), f'outputs sha256 of {fn}')
        ok('econ/fills_valued.parquet' in J['inputs'] and len(J['inputs']['truth/v2/scores.json']['sha256']) == 64, 'input sha256s recorded (truth v2)')
        ok(not any('own_orders' in k for k in J['inputs']), 'the truth table itself is never read')
        # ---- R1: T1 = 60+40+50+0+30+10 = 190 < 200 -> INSUFFICIENT (2 violations, 1 certain); T2 = 100+50+60+0+20+20 = 250, 2 -> 0.008 supported ----
        r1 = J['R1']
        ok((r1['T1']['den'], r1['T1']['violations'], r1['T1']['violations_certain'], r1['T1']['reading']) == (190, 2, 1, 'INSUFFICIENT'), f'R1 T1 {r1["T1"]}')
        ok((r1['T2']['den'], r1['T2']['violations'], r1['T2']['violations_certain'], r1['T2']['reading']) == (250, 2, 1, 'supported'), f'R1 T2 {r1["T2"]}')
        close(r1['T2']['rate'], 0.008, 'R1 T2 rate')
        close(r1['T2']['rate_certain'], 0.004, 'R1 T2 window-certain rate (1 / 250)')
        ok(r1['coverage_scenario'] == 'S_FIFO' and r1['prio_scenarios_equal'] and r1['n_markets'] == dict(BOOK=5, GT=1), f'R1 coverage scenario {r1}')
        P = csv(dest, 'priority_tests')
        x = one(P, 'pooled BOOK T1', scope='pooled_BOOK', scenario='S_FIFO', test='T1', version='settle_order')
        ok((x.den, x.violations, x.reading) == (180, 2, 'INSUFFICIENT'), 'pooled BOOK T1 = 180 / 2')
        x = one(P, 'primary T2', scope='pooled_ALL', scenario='S_FIFO', test='T2', version='settle_order')
        ok(x.role == 'primary' and x.n_markets == 6, 'primary row flagged, 6 markets')
        x = one(P, 'market 202 T1', scope='market', mid=202, scenario='S_FIFO', test='T1', version='settle_order')
        ok(x.den == 0 and pd.isna(x.rate), 'a zero denominator has no rate')
        for den, v, want in ((199, 0, 'INSUFFICIENT'), (0, 0, 'INSUFFICIENT'), (200, 2, 'supported'), (200, 3, 'contradicted'), (1000, 10, 'supported')):
            ok(R.r1_reading(den, v) == want, f'r1_reading({den}, {v}) == {want}')
        # ---- R2: one constrained exhausted-only defect per scenario -> defective, (removal, constrained) demoted ----
        r2 = J['R2']
        ok(r2['reading'] == 'defective' and r2['defects_total'] == 3 and r2['defects_by_scope'] == dict(fills=0, S_FIFO=1, S_FIFO_LO=1, S_NOPRIO=1), f'R2 {r2}')
        ok(r2['defective_pairs'] == [dict(field='removal', cls='constrained', r3_column='removal_class', demoted=True)], f'R2 pairs {r2["defective_pairs"]}')
        ok('NOT ESTABLISHED' in r2['clock_premise'] and 'outcome-selected' in r2['caveat'] and r2['n']['scored_own_orders'] == 7, 'R2 premise / caveat / n')
        ok(r2['sensitive_misses']['S_FIFO'] == 2 and r2['sensitive_n']['S_FIFO'] == 4, 'R2 sensitive misses 2 of 4')
        GS = csv(dest, 'gt_scores')
        x = one(GS, 'gt removal row', scenario='S_FIFO', field='removal', test='exhausted_only', cls='constrained')
        ok((x.n, x.failed, x.defects, x.reading) == (3, 1, 1, 'defective') and 'NOT ESTABLISHED' in x.clock_premise and 'outcome-selected' in x.caveat, 'gt row')
        x = one(GS, 'gt posting sensitive', scenario='S_FIFO', field='post', test='posting', cls='assumption_sensitive')
        ok((x.n, x.failed, x.sensitive_misses) == (4, 2, 2), 'gt posting sensitive misses')
        close(x.sensitive_miss_rate, 0.5, 'sensitive miss rate 2 / 4')
        # ---- R3 coverage: exante_sh = {A, B, CC}, D1 (101 + 102), both sides, removal ----
        # orders h101a (A, constrained -> demoted, 3 exec, 30 sh), h101b (B, unres, 1, 5), h102a (A, sens, 2, 10), h102a2 (sens, 1, 2), h102a3 (sens, 1, 1),
        # h102a4 (constrained -> demoted, 2, 2), h102b (B, unres, 3, 5): den 7 / 13 / 55; sens 5 / 9 / 45; unres 2 / 4 / 10; raw constrained 2 / 5 / 32
        CV = csv(dest, 'coverage')
        x = one(CV, 'cov sens', id_set='exante_sh', day=D1, side='both', field='removal', cls='assumption_sensitive')
        ok((x.n_orders, x.n_exec, x.den_orders, x.den_exec, x.demoted_in_orders, x.demoted_in_exec) == (5, 9, 7, 13, 2, 5), 'coverage sens counts')
        close(x.shares, 45., 'cov sens shares')
        close(x.den_shares, 55., 'cov den shares')
        close(x.frac_orders, 5 / 7, 'cov sens frac orders')
        close(x.frac_exec, 9 / 13, 'cov sens frac exec')
        close(x.frac_shares, 45 / 55, 'cov sens frac shares')
        ok(str(x.demoted) == 'False', 'the receiving class is not itself demoted')
        x = one(CV, 'cov constrained', id_set='exante_sh', day=D1, side='both', field='removal', cls='constrained')
        ok((x.n_orders, x.raw_n_orders, x.raw_n_exec, str(x.demoted)) == (0, 2, 5, 'True'), 'constrained removal demoted (0 effective, 2 raw)')
        close(x.raw_shares, 32., 'raw constrained shares 30 + 2')
        x = one(CV, 'cov unres', id_set='exante_sh', day=D1, side='both', field='removal', cls='unresolved')
        ok((x.n_orders, x.n_exec) == (2, 4), 'unresolved removal 2 / 4')
        # post, bid side: h101a (constrained, 3, 30), h102a3 (constrained, 1, 1), h102a4 (constrained, 2, 2), h102b (unres, 3, 5) -> constrained 33 / 38 sh
        x = one(CV, 'cov post bid', id_set='exante_sh', day=D1, side='bid', field='post', cls='constrained')
        ok((x.n_orders, x.n_exec, x.den_orders, x.den_exec, str(x.demoted)) == (3, 6, 4, 9, 'False'), 'coverage post bid')
        close(x.frac_shares, 33 / 38, 'post bid frac shares')
        # field = {G} on D1: h101g (post sens, 1, 4), h101g2 (unres, 1, 2), h102g (unres, 1, 1)
        x = one(CV, 'cov field', id_set='field', day=D1, side='both', field='post', cls='assumption_sensitive')
        ok((x.n_orders, x.den_orders, x.den_exec) == (1, 3, 3), 'field post sens 1 of 3')
        # pooled exante_sh removal: den 11 / 20 / 69; sens 7 / 12 / 51 (D1 5 + h202a + h301c); unres 4 / 8 / 18; raw constrained 4 (h101a h102a4 h202a h301c)
        x = one(CV, 'cov pooled', id_set='exante_sh', day='pooled', side='both', field='removal', cls='assumption_sensitive')
        ok((x.n_orders, x.n_exec, x.den_orders, x.den_exec) == (7, 12, 11, 20), 'pooled removal sens')
        close(x.shares, 51., 'pooled sens shares')
        ok(one(CV, 'cov pooled constrained', id_set='exante_sh', day='pooled', side='both', field='removal', cls='constrained').raw_n_orders == 4, 'raw 4')
        r3 = J['R3']['per_set']['exante_sh']
        ok((r3['n_orders'], r3['n_exec']) == (11, 20) and r3['fractions']['removal']['constrained']['demoted'], 'R3 readings summary')
        close(r3['fractions']['removal']['assumption_sensitive']['orders'], 7 / 11, 'R3 summary fraction')
        ok(J['R3']['demoted'] == dict(removal=['constrained']), 'R3 demoted list')
        x = one(CV, 'cov all', id_set='all', day='pooled', side='both', field='size', cls='observed')
        ok(x.den_orders == 16, 'all set: 16 BOOK orders')
        # ---- windows: exante_sh D1 S_FIFO post widths 5000 (h101a), 8000 (h102a), 16000 (h102a3), 17500 (h102a4); 3 open lower ends ----
        W = csv(dest, 'windows')
        x = one(W, 'post window', id_set='exante_sh', day=D1, scenario='S_FIFO', metric='post_window_ms')
        ok((x.n_orders, x.n_bounded, x.n_lo_open, x.n_hi_open, str(x.primary)) == (7, 4, 3, 0, 'True'), 'post window counts')
        close(x.p0, 5000., 'post p0')
        close(x.p50, 12000., 'post p50 = (8000 + 16000) / 2')
        close(x.p100, 17500., 'post p100')
        close(x['avg'], 11625., 'post mean 46500 / 4')
        # removal widths 1000 (h101a: 30000 - 29000), 6000 (h102a), 2000 (h102a2), 1000 (h102a3), 1000 (h102a4); h101b both open; h102b hi open
        x = one(W, 'removal window', id_set='exante_sh', day=D1, scenario='S_FIFO', metric='removal_window_ms')
        ok((x.n_bounded, x.n_both_open, x.n_hi_open) == (5, 1, 1), 'removal window counts')
        close(x.p50, 1000., 'removal p50')
        close(x['avg'], 2200., 'removal mean 11000 / 5')
        ok(not {'mean', 'rank', 'size', 'count', 'sum', 'min', 'max', 'first', 'last', 'pop'} & set().union(
            *(set(pd.read_csv(p, nrows=0).columns) for p in dest.glob('*.csv'))), 'no output column is named like a pandas method')
        x = one(W, 'ahead', id_set='exante_sh', day=D1, scenario='S_FIFO', metric='ahead_range_sh')
        ok((x.n_bounded, x.n_both_open) == (2, 5), 'ahead bounded on h101a (3) and h102b (0)')
        close(x.p100, 3., 'ahead max range 3 sh')
        x = one(W, 'persist', id_set='exante_sh', day=D1, scenario='S_FIFO', metric='persist_span_ms')
        ok(x.n_bounded == 4, 'persistence spans of h101a, h102a, h102a4, h102b')
        close(x.p100, 19000., 'persist span h101a 29000 - 10000')
        ok(str(one(W, 'nopro', id_set='exante_sh', day=D1, scenario='S_NOPRIO', metric='post_window_ms').primary) == 'False', 'S_NOPRIO not primary')
        RS = csv(dest, 'removal_sets')
        got = {r.removal_set: r.n_orders for r in RS[(RS.id_set == 'exante_sh') & (RS.day == D1) & (RS.scenario == 'S_FIFO')].itertuples()}
        ok(got == {EXH: 3, CEE: 2, CE: 2}, f'removal sets S_FIFO {got}')
        got = {r.removal_set: r.n_orders for r in RS[(RS.id_set == 'exante_sh') & (RS.day == D1) & (RS.scenario == 'S_NOPRIO')].itertuples()}
        ok(got == {EXH: 2, CEE: 4, CE: 1}, f'removal sets S_NOPRIO {got}')
        LKd = csv(dest, 'links')
        x = one(LKd, 'links reprice', id_set='exante_sh', day=D1, kind='reprice')
        ok((x.n_links, x.n_ok, x.den_orders) == (2, 2, 7), 'exante_sh D1: 2 reprices of 7 orders')
        close(x.gap_ms_p50, 1000., 'reprice gap')
        x = one(LKd, 'links refill', id_set='exante_sh', day=D2, kind='refill')
        ok(x.n_links == 1, 'one refill on D2')
        close(x.gap_ms_p0, 10000., 'refill gap 30000 - 20000')
        ok(one(LKd, 'links field', id_set='field', day=D1, kind='reprice').n_links == 1, 'G reprices once')
        # ---- R4 econ_identity_day: exante_sh in on D1 = A + B + CC: n_mk 6, sh 44, gross 4 + 2 - 1 = 5, fee 0.5, net 4.5, rt 1 + 2 + 0 = 3 ----
        ED = csv(dest, 'econ_identity_day')
        x = one(ED, 'econ in D1', id_set='exante_sh', grp='in', day=D1)
        ok((x.n_mk, x.n_addr) == (6, 3), 'econ in D1 counts')
        for c, v in (('sh_mk', 44.), ('gross_res', 5.), ('fee_res', .5), ('net_res', 4.5), ('share_matched', .6), ('v_spr', .75),
                     ('c_sh_spr', 100 * .75 / 44), ('c_sh_net', 100 * 4.5 / 44), ('reb25_res', .44), ('reb25_fill', .88),
                     # resolution part NP x (1 - 0.75): A 2 x .25 + B -1 x .25 = 0.25; drift = residual (3 + 0 - 1) - 0.25 = 1.75
                     ('resid_resol', .25), ('resid_drift', 1.75), ('rt_pnl_res', 3.)):
            close(x[c], v, f'econ in D1 {c}')
        ok(str(x.sign_consistent) == 'True' and x.n_days == 4 and x.ci == 'none (4 < 5 days)', 'econ in predicates')
        # pooled: gross 5 + 5 + 4 + 4 = 18; rt 3 + 1.5 + 2 + 2.5 = 9 (A's unresolved 402 row 0.75 excluded); net 4.5 + 4.5 + 3.5 + 3.5 = 16
        x = one(ED, 'econ in pooled', id_set='exante_sh', grp='in', day='pooled')
        for c, v in (('gross_res', 18.), ('rt_pnl_res', 9.), ('share_matched', .5), ('net_res', 16.),
                     ('rt_pnl_unresolved_mkts', .75), ('reb25_res', (44 * 4 + 2) / 100)):
            close(x[c], v, f'econ pooled {c}')
        ok(x.n_rows_res_ok == 12 and x.n_rows == 13, 'res-ok rows 12 of 13 (the no_resolution row)')
        ok(str(one(ED, 'field', id_set='exante_sh', grp='field', day=D2).sign_consistent) == 'False', 'field G (1, -2, 1, 1) not sign-consistent')
        x = one(ED, 'nonmaker', id_set='exante_sh', grp='nonmaker', day=D3)
        ok(str(x.sign_consistent) == 'True' and x.sign == '-', 'nonmaker T always negative')
        close(x.net_res, -5.25, 'nonmaker D3 net -5 - 0.25')
        x = one(ED, 'pnl field', id_set='exante_pnl', grp='field', day=D2)
        ok(str(x.sign_consistent) == 'False' and x.sign == 'mixed_or_zero', 'exante_pnl field D2 net 2 - 2 = 0 is not a strict sign')
        close(x.net_res, 0., 'exante_pnl field D2 net')
        r4 = J['R4']['per_set']
        ok(r4['exante_pnl']['in']['sign_consistent'] and r4['exante_pnl']['in']['net_res_by_day'] == {D1: .5, D2: 2.5, D3: .5, D4: 2.5}, 'R4 exante_pnl in')
        close(r4['exante_sh']['in']['share_matched_pooled'], .5, 'R4 share matched pooled')
        ok(J['R4']['ci'] == 'none (4 < 5 days)', 'R4 ci text')
        SG = csv(dest, 'econ_segments')
        x = one(SG, 'seg tseg', id_set='exante_sh', grp='in', day=D1, role='maker', dim='tseg', bucket='w000_060')
        close(x.share_net, 4 / 4.5, 'segment share of the set net (A 4 of 4.5)')
        close(x.dim_share_sum, 1., 'tseg shares sum to 1')
        x = one(SG, 'seg pooled', id_set='exante_sh', grp='in', day='pooled', role='maker', dim='all', bucket='all')
        close(x.net_res, 16., 'pooled segment net')
        # ---- timelines: D1 -> 102 (maker legs of A / B: 101 = 4, 102 = 9); D2 tie 3 / 3 -> 201 (202 would win if B's taker leg counted); D3 -> 301 (0) ----
        TL = J['timelines']
        ok(TL[D1]['mid'] == 102 and TL[D1]['candidates'] == {'101': 4, '102': 9}, f'timeline D1 {TL[D1]}')
        ok(TL[D2]['mid'] == 201 and TL[D2]['candidates'] == {'201': 3, '202': 3}, f'timeline D2 tie -> smallest mid {TL[D2]}')
        ok(TL[D3]['mid'] == 301 and TL[D3]['rows'] == 0 and TL[D4]['mid'] is None, 'timeline D3 empty, D4 none')
        T1 = csv(dest, f'timeline_{D1}_102')
        ok(len(T1) == 10 and (T1.role == 'taker').sum() == 1, 'D1 timeline: 9 maker rows + B taker row')
        w = ws_ms(102)
        x = one(T1, 's1021', sid='s1021')
        ok(x.addr == A and x.rank_exante_sh == 1 and x.ptick == 60 and x.n_fills == 2 and x.removal_set == CE and x.scenario == 'S_FIFO', 's1021 order cols')
        ok(x.tob_pre_upd_ms == w + 9000 and x.tob_post_upd_ms == w + 10000, 's1021 tob frames (lo 9000 / hi 10000)')
        close(x.tob_pre_best_bid, .565, 'tob at lo: the later receipt of two 9000 frames')
        close(x.tob_post_best_ask, .61, 'tob at hi')
        ok(x.shown_before == 7. and x.shown_before_status == 'ok' and x.shown_after == 0. and x.shown_after_status == 'absent_0', 's1021 depth')
        x = one(T1, 's1022', sid='s1022')
        ok(x.shown_before == 3. and x.shown_before_status == 'cell_status', 's1022 depth cell with a non-ok level')
        x = one(T1, 's1023', sid='s1023')
        ok(x.shown_before == 4. and x.shown_after == 2. and x.rest_side == 'bid', 's1023 depth')
        x = one(T1, 's1030', sid='s1030')
        ok(x.role == 'taker' and x.addr == B and x.rest_side == 'ask' and x.ptick == 62 and pd.isna(x.n_fills) and x.shown_before_status == 'no_frame',
           's1030 taker row: resting side ask, no order columns')
        ok(T1.sid.iloc[0] == 's1021', 'timeline sorted by hi')
        ok(len(csv(dest, f'timeline_{D2}_201')) == 3 and len(csv(dest, f'timeline_{D3}_301')) == 0, 'D2 / D3 timelines')
        x = one(T1, 'removal demoted in timeline', sid='s1028')
        ok(x.removal_class == S and x.raw_removal_class == K and str(x.removal_demoted) == 'True', 'timeline carries the demotion')
        # ---- examples: D1 A: most-filled h102a / h102a4 (2 fills, sha order), then h102a2 (2 explanations), then h102a3 (single); B: h102b; D2 B: h201b ----
        EX = csv(dest, 'examples')
        a1 = EX[(EX.day == D1) & (EX.addr == A)]
        mf = sorted(['h102a', 'h102a4'], key=hk)
        ok(list(a1.rule) == ['most_filled_1', 'most_filled_2', 'removal_set_ge_2_explanations', 'single_fill'] and list(a1.mk_hash) == mf + ['h102a2', 'h102a3'],
           f'examples A D1 {list(a1.mk_hash)}')
        ok(list(EX[(EX.day == D1) & (EX.addr == B)].mk_hash) == ['h102b'] and list(EX[EX.day == D2].mk_hash) == ['h201b'] and len(EX) == 6, 'examples B / D2')
        ok(one(EX, 'sha col', mk_hash='h102b').sha256_hash == hk('h102b'), 'examples carry sha256(hash)')
        # ---- unresolved_top: all 16 BOOK orders qualify (constrained removal demoted); |net| 5 (h201b: 5 + 0, its 100 row excluded), 4, 3, 2.5, 2 / 2 ... ----
        UT = csv(dest, 'unresolved_top')
        ok(J['unresolved_top'] == dict(n_candidates=16, n_listed=16, rule=J['unresolved_top']['rule']), f'unresolved counts {J["unresolved_top"]}')
        tie = sorted(['h102a', 'h202a'], key=hk)
        ok(list(UT.mk_hash[:7]) == ['h201b', 'h101b', 'h301c', 'h101a'] + tie + ['h202a2'], f'unresolved order {list(UT.mk_hash[:7])}')
        x = one(UT, 'h201b', mk_hash='h201b')
        ok((x.n_fv, x.n_fv_ok) == (3, 2), 'h201b: the src_status row is counted, not summed')
        close(x.net_res, 5., 'h201b net')
        x = one(UT, 'h101b', mk_hash='h101b')
        ok('post_lo open (no capacity / fifo_through bound) in S_FIFO/S_FIFO_LO/S_NOPRIO' in x.missing_post and 'removal_hi open' in x.missing_removal
           and 'removal_lo open' in x.missing_removal, f'h101b missing evidence {x.missing_post} | {x.missing_removal}')
        x = one(UT, 'h101a', mk_hash='h101a')
        ok(x.missing_post == '-' and 'class demoted by a GT defect (removal / constrained)' in x.missing_removal and str(x.removal_demoted) == 'True', 'h101a')
        x = one(UT, 'h102a', mk_hash='h102a')
        ok('removal_hi open (end_open: no bound to the last usable frame) in S_NOPRIO' in x.missing_removal and 'removal_set differs' in x.missing_removal, 'h102a')
        # ---- inventory (D1) ----
        IV = csv(dest, 'inventory')
        x = one(IV, 'amount', day=D1, section='predicate', item='amount_reconciles (taker.amount == amountFilled == sum makers.amount, ints)')
        ok((x.n, x.den, x.value) == (16, 17, 'False'), 'amount reconciliation 16 / 17')
        x = one(IV, 'exec', day=D1, section='predicate', item='executedAt_whole_second (fraction digits all zero)')
        ok((x.n, x.den, x.value) == (98, 100, 'False'), 'executedAt whole-second 98 / 100')
        x = one(IV, 'makers order', day=D1, section='predicate', item='makers_listing_best_first (multi-price records)')
        ok((x.n, x.den) == (2, 10), 'best-first 2 of 10 multi-price records')
        ok(one(IV, 'monotone', day=D1, section='predicate', item='makers_listing_monotone_all (no neither)').value == 'False', 'makers[] not monotone')
        x = one(IV, 'fee', day=D1, section='predicate', item='maker_fee_all_zero')
        ok((x.value, x.n, x.den) == ('True', 50, 50), 'maker fee all zero')
        ok(one(IV, 'tk fee', day=D1, section='predicate', item='taker_fee_types').value == 'COLLATERAL|SHARES', 'taker fee types')
        close(one(IV, 'lag', day=D1, section='quantile', item='rest.clocks_and_lags_ms.settle_minus_exec_ms', key='p50').value, -800., 'lag quantile')
        ok(one(IV, 'keys', day=D1, section='source_keys', item='maker.fee').key == 'amount|type', 'exact source keys')
        ok(one(IV, 'env', day=D1, section='envelope', item='margin_ms').value == '60000', 'envelope margin')
        ok(one(IV, 'na', day=D2, section='predicate', item='maker_fee_all_zero').value == 'not_evaluable', 'an empty counter is not_evaluable, never true')
        # ---- statuses ----
        ST = csv(dest, 'statuses')
        x = one(ST, 'legs ok', day=D1, table='legs', column='status', kind='status', value='ok')
        ok((x.n, x.den) == (15, 16), 'D1 legs ok 15 / 16')
        ok(one(ST, 'legs flag', day=D1, table='legs', column='status', kind='flag', value='boundary_censored').n == 1, 'boundary flag counted')
        ok(one(ST, 'fv', day=D2, table='econ_fills_valued', column='status', kind='status', value='src_status').n == 2, 'econ excluded leg pair counted')
        ok(one(ST, 'mk', day=D4, table='markets', column='status', kind='status', value='boundary_censored').n == 1, 'markets status')
        ok(one(ST, 'life', day=D1, table='life_fills', column='status', kind='flag', value='fill_lo_unbounded').n == 1, 'lifecycle fills status (s1014)')
        ok(one(ST, 'life in', day=D1, table='life_inputs', column='unknown_cells', kind='count').n == 2, 'lifecycle inputs summed over 2 markets')
        ok(one(ST, 'mm', day=D4, table='econ_maker_market', column='res_status', kind='status', value='no_resolution').n == 1, 'maker_market status')
        # ---- econ_address_day: A on D1: 30 maker sh of the day's 50 (A 30, B 10, CC 4, G 6); pooled 30 x 4 + 2 (402) = 122 of 202 ----
        AD = csv(dest, 'econ_address_day')
        x = one(AD, 'A D1', addr=A, day=D1)
        ok((x.rank_exante_sh, x.rank_exante_pnl, x.rank_sh_day, x.rank_pnl_day, str(x.in_exante_pnl)) == (1, -1, 1, 1, 'False'), 'A D1 ranks')
        for c, v in (('sh_mk', 30.), ('maker_frac', 1.), ('share_of_day_maker_volume', .6), ('net_res', 4.), ('reb25_res', .3), ('self_trade_sh', 0.),
                     ('resid_resol', .5), ('rt_pnl_res', 1.)):
            close(x[c], v, f'A D1 {c}')
        x = one(AD, 'A pooled', addr=A, day='pooled')
        close(x.share_of_day_maker_volume, 122 / 202, 'A pooled share of maker volume')
        close(x.self_trade_sh, 4., 'A self-trade 4 sh (s4022, counted once)')
        close(x.net_res, 10., 'A pooled net 4 + 2 + 3 + 1')
        ok(str(x.sign_consistent) == 'True' and x.n_days == 4, 'A sign-consistent over 4 days')
        ok(str(one(AD, 'CC', addr=CC, day='pooled').sign_consistent) == 'False', 'CC (-1, 1, -1, 1) not sign-consistent')
        ok(set(AD.addr) == {A, B, CC} and len(AD) == 15, 'one row per member per day + pooled (3 x 5)')
        # ---- counterparty: D4 A makes / buys 10 x 1 sh against B (one-sided both ways); A vs field on D1: maker 45 / taker 2 (0.957), buys 33 / sells 14 ----
        CPd = csv(dest, 'counterparty')
        x = one(CPd, 'A-B', day=D4, addr=A, cp=B)
        ok((x.n_fills, x.n_ok, x.role_direction, x.side_direction) == (10, 10, 'one_sided', 'one_sided'), 'A-B one-sided pair')
        for c, v in (('shares', 10.), ('sh_mk', 10.), ('sh_buy', 10.), ('net_res', 2.5), ('share_of_addr_shares', 10 / 15)):
            close(x[c], v, f'A-B {c}')
        x = one(CPd, 'B-A', day=D4, addr=B, cp=A)
        ok(x.sh_tk == 10. and x.sh_sell == 10. and x.role_direction == 'one_sided', 'B-A mirror')
        close(x.net_res, -2.5, 'B-A net')
        ok(one(CPd, 'A field D4', day=D4, addr=A, cp='field').role_direction == 'insufficient (n_ok < 10)', 'one fill is insufficient')
        ok(one(CPd, 'A self', day=D4, addr=A, cp='self').shares == 4., 'self-trade row: the leg once')
        x = one(CPd, 'A field D1', day=D1, addr=A, cp='field')
        ok((x.n_ok, x.role_direction, x.side_direction) == (10, 'one_sided', 'two_sided'), f'A vs field D1 {x.role_direction} / {x.side_direction}')
        close(x.shares, 47., 'A vs field D1 shares 45 + 2')
        ok(J['counterparty']['pairs_checked'] == 2 and len(J['counterparty']['one_sided_member_pairs_pooled']) == 2, 'pairs mirrored, 2 one-sided')
        # ---- resolution_null (exante_sh, in): D1 NP 1 at m_END .75: S = .25, P(S* < S) + P(S* = S) / 2 = .25 + .375; D3 m_END 1: S* = 0 = S -> 0.5;
        #      pooled S = .25 + 1 + 0 + 1.5 = 2.75 = max(S*) -> 1 - .09375 / 2 = 0.953 ----
        RN = csv(dest, 'resolution_null')
        x = one(RN, 'null D1', id_set='exante_sh', grp='in', day=D1)
        ok(x.n_markets == 1 and x.n_draws == 1000 and abs(x.percentile - .625) < .05 and abs(x.null_mean) < .1, f'null D1 pct {x.percentile}')
        close(x.actual_resid_resol, .25, 'null D1 actual')
        close(x.recomputed, .25, 'null D1 recomputed from fills_valued')
        x = one(RN, 'null D3', id_set='exante_sh', grp='in', day=D3)
        close(x.percentile, .5, 'degenerate m_END = 1 -> every draw ties')
        ok(abs(one(RN, 'null D2', id_set='exante_sh', grp='in', day=D2).percentile - .75) < .05, 'null D2 pct ~ 0.75')
        x = one(RN, 'null pooled', id_set='exante_sh', grp='in', day='pooled')
        ok(x.n_markets == 4 and abs(x.percentile - .953125) < .03, f'pooled null pct {x.percentile}')
        close(x.actual_resid_resol, 2.75, 'pooled actual')
        close(one(RN, 'null T', id_set='exante_sh', grp='nonmaker', day=D1).actual_resid_resol, -.5, 'nonmaker NP -2 x .25')
        ok(J['R4']['resolution_null_rule']['n_draws'] == 1000 and J['R4']['resolution_null_rule']['seed'] == R.SEED, 'null declared in the receipt')
        # ---- rank persistence ----
        RP = csv(dest, 'rank_persistence')
        x = one(RP, 'CC sh', kind='member_summary', addr=CC, metric='sh')
        ok((x.n_days_top10, x.id_rank) == (3, 3), 'CC shares: top 10 on 3 of 4 days (rank 12 on D1)')
        ok(one(RP, 'CC pnl', kind='member_summary', addr=CC, metric='pnl').n_days_top10 == 3, 'CC P&L: rank 15 on D3')
        close(one(RP, 'set sh', kind='set_summary', addr='exante_sh', metric='sh', day=D1).frac_in_top10, 2 / 3, 'exante_sh D1 retention')
        close(one(RP, 'set pnl', kind='set_summary', addr='exante_pnl', metric='pnl', day=D3).frac_in_top10, .5, 'exante_pnl D3 retention')
        close(J['rank_persistence']['frac_in_top10_by_day']['exante_sh/sh'][D1], 2 / 3, 'rank readings')
        # ---- rebate beside trading, rewards statement, segment R1 flags, GT unique defects ----
        x = one(csv(dest, 'rebate_scenario'), 'rebate', id_set='exante_sh', grp='in', day=D1)
        ok(abs(x.reb25_res - .44) < 1e-9 and abs(x.reb25_fill - .88) < 1e-9 and abs(x.net_res - 4.5) < 1e-9, 'rebate scenario beside net')
        RW = csv(dest, 'rewards_market')
        ok(len(RW) == 2 and RW.statement.str.contains('unidentifiable').all() and J['rewards']['rows'] == 2, 'rewards table + statement')
        x = one(SG, 'life_exh flag', id_set='exante_sh', grp='in', day=D1, role='maker', dim='life_exh', bucket='na_not_book')
        ok(str(x.s_fifo_dependent) == 'True' and str(x.r1_contradicted) == 'False' and x.r1_readings == 'T1=INSUFFICIENT; T2=supported', 'segment R1 flag')
        ok(str(one(SG, 'depth', id_set='exante_sh', grp='in', day=D1, role='maker', dim='depth', bucket='x').s_fifo_dependent) == 'False', 'depth dim')
        ok(J['R2']['unique_defects']['total'] == 1 and J['R2']['unique_defects']['by_field'] == {'removal_exhausted': 1}
           and J['R2']['unique_defects']['by_r3_field'] == dict(removal=1), 'one unique (order, field) defect behind 3 scenario rows')
        # ---- the TIE market 203 (V = 0.5): resolved (RES_VALID) -> T on D2: gross -3 + 0.5 = -2.5, round trip 0 + 0.25 (an '== ok' rule breaks the identity) ----
        x = one(ED, 'tie', id_set='exante_sh', grp='nonmaker', day=D2)
        ok(x.n_rows_res_ok == 2, 'the tie row counts as resolved')
        close(x.gross_res, -2.5, 'nonmaker D2 gross with the tie market')
        close(x.rt_pnl_res, .25, 'round trip on resolved markets includes the tie')
        close(x.share_matched, -.1, 'share matched .25 / -2.5')
        x = one(RN, 'tie null', id_set='exante_sh', grp='nonmaker', day=D2)
        ok(x.n_tie_markets == 1 and x.n_markets == 2, 'null counts the tie market')
        close(x.actual_resid_resol, -1.5, 'nonmaker D2 resolution part: -3 x .5 + 1 x (.5 - .5)')
        ok(one(ST, 'tie st', day=D2, table='econ_fills_valued', column='st_res', kind='status', value='tie').n == 2, 'tie rows counted')
        close(J['R4']['checks']['rt_pnl_res_vs_maker_day_max_abs_dev'], 0., 'maker_day rt_pnl_res reproduced')
        ok(one(GS, 'unique row', scenario='unique_across_scenarios', field='removal', cls='constrained').defects == 1, 'unique row in gt_scores')
        ok(one(GS, 'size ub', scenario='S_FIFO', field='size', test='size_ub', cls='constrained').n == 2, 'size ub scores carried')
        # ---- refusals (reason asserted) ----
        refuses(lambda: R.run(OUT, dest, log=quiet), 'write-once: outputs/ exists', 'an existing outputs/ dir')

        def mutated(tag, fn):
            o = td / tag / 'OUT'
            shutil.copytree(OUT, o)
            fn(o)
            return o, td / tag / 'receipt' / 'outputs'
        o, ds = mutated('m1', lambda o: (o / 'life' / f'{D2}_202_orders.parquet').unlink())
        refuses(lambda: R.run(o, ds, log=quiet), f'input missing: lifecycle orders {D2} 202', 'a missing input table')
        ok(not ds.exists() and not list(ds.parent.glob('.outputs.partial_*')), 'a refused run publishes nothing and leaves no partial dir')

        def bad_class(o):
            p = o / 'life' / f'{D1}_101_orders.parquet'
            t = pq.read_table(p).to_pandas()
            t.loc[t.mk_hash == 'h101b', 'queue_class'] = 'observedX'
            write(p, C.SCHEMAS['orders'], t.to_dict('records'))
        o, ds = mutated('m2', bad_class)
        refuses(lambda: R.run(o, ds, log=quiet), "unknown class ['observedX'] in queue_class", 'a class outside CLASSES')

        def bad_counts(o):
            p = o / f'extract_{D3}.json'
            j = json.loads(p.read_text())
            j['status_counts']['legs'] = {'ok': j['status_counts']['legs']['ok'] + 1}
            p.write_text(json.dumps(j))
        o, ds = mutated('m3', bad_counts)
        refuses(lambda: R.run(o, ds, log=quiet), f'{D3} legs: status counts', 'a status count that differs from the extract receipt')

        def bad_gt(o):
            p = o / 'truth' / 'v2' / 'scores.json'
            j = json.loads(p.read_text())
            j['scenarios']['S_FIFO']['defects_total'] = 0
            p.write_text(json.dumps(j))
        o, ds = mutated('m4', bad_gt)
        refuses(lambda: R.run(o, ds, log=quiet), 'GT scores S_FIFO: per-class defects sum 1 != defects_total 0', 'GT totals that do not recompute')

        def bad_seg(o):
            p = o / 'econ' / 'segments.parquet'
            t = pq.read_table(p).to_pandas()
            t.loc[(t.day == D2) & (t.dim == 'all') & (t.grp == 'in') & (t.id_set == 'exante_pnl'), 'net_res'] += 0.5
            write(p, E.SEG, t.to_dict('records'))
        o, ds = mutated('m5', bad_seg)
        refuses(lambda: R.run(o, ds, log=quiet), 'mismatched econ artifacts', 'econ segments that do not reconcile with maker_market')

        def bad_fv(o):
            p = o / 'econ' / 'econ.json'
            j = json.loads(p.read_text())
            j['counts'][D1]['status']['ok'] += 1
            p.write_text(json.dumps(j))
        o, ds = mutated('m6', bad_fv)
        refuses(lambda: R.run(o, ds, log=quiet), f'fills_valued {D1} status: counts', 'fills_valued counts that differ from econ.json')

        def no_reb(o):
            p = o / 'econ' / 'maker_market.parquet'
            t = pq.read_table(p).to_pandas()
            write(p, {c: v for c, v in dict(E.MM, **{c: 'float64' for c in R.ECON_NEW_MM}).items() if c != 'reb25_res'}, t.to_dict('records'))
        o, ds = mutated('m7', no_reb)
        refuses(lambda: R.run(o, ds, log=quiet), 'reb25_res', 'a declared econ column that is absent (fail closed, never guessed)')

        def no_ub(o):
            p = o / 'truth' / 'v2' / 'scores.json'
            j = json.loads(p.read_text())
            del j['scenarios']['S_FIFO']['size']['ub']
            p.write_text(json.dumps(j))
        o, ds = mutated('m8', no_ub)
        refuses(lambda: R.run(o, ds, log=quiet), 'GT scores S_FIFO: declared block size.ub absent', 'a declared GT block that is absent')

        def bad_rt(o):
            p = o / 'econ' / 'maker_day.parquet'
            t = pq.read_table(p).to_pandas()
            t.loc[(t.day == D2) & (t.addr == T), 'rt_pnl_res'] = 0.      # drops the tie market's round trip: econ's value no longer reproduces
            write(p, E.MD, t.to_dict('records'))
        o, ds = mutated('m9', bad_rt)
        refuses(lambda: R.run(o, ds, log=quiet), 'maker_day rt_pnl_res differs from the maker_market round trip', 'an rt_pnl_res that does not reproduce')
        # ---- variant: R1 contradicted (T2 5 / 250 = 0.02) -> S_NOPRIO coverage scenario; no GT defect -> nothing demoted ----
        OUT2 = world(td / 'w2', contradicted=True, gt_defect=False)
        dest2 = td / 'w2' / 'receipt' / 'outputs'
        R.run(OUT2, dest2, log=quiet)
        J2 = json.loads((dest2 / 'readings.json').read_text())
        ok(J2['R1']['T2']['reading'] == 'contradicted' and J2['R1']['coverage_scenario'] == 'S_NOPRIO' and J2['R1']['any_contradicted'], 'variant R1')
        close(J2['R1']['T2']['rate'], .02, 'variant T2 rate 5 / 250')
        ok(J2['R2']['reading'] == 'no_defect' and J2['R2']['defective_pairs'] == [] and J2['R3']['demoted'] == {}, 'variant R2 no defect')
        W2 = csv(dest2, 'windows')
        ok(str(one(W2, 'v primary', id_set='exante_sh', day=D1, scenario='S_NOPRIO', metric='post_window_ms').primary) == 'True', 'S_NOPRIO primary')
        CV2 = csv(dest2, 'coverage')
        x = one(CV2, 'v cov', id_set='exante_sh', day=D1, side='both', field='removal', cls='constrained')
        ok((x.n_orders, x.raw_n_orders, str(x.demoted)) == (2, 2, 'False'), 'no demotion: constrained stays constrained')
        # unresolved: h101a, h102a4, h202a, h301c (post and removal constrained) drop out -> 12 candidates
        ok(J2['unresolved_top']['n_candidates'] == 12, f'variant unresolved candidates {J2["unresolved_top"]}')
        UT2 = csv(dest2, 'unresolved_top')
        ok(not set(UT2.mk_hash) & {'h101a', 'h102a4', 'h202a', 'h301c'} and list(UT2.mk_hash[:3]) == ['h201b', 'h101b', 'h102a'], 'variant ranking')
        # examples under S_NOPRIO: h102a2 and h102a3 both have >= 2 explanations -> the smaller sha256 is the multi pick, the other the single-fill pick
        EX2 = csv(dest2, 'examples')
        a1 = EX2[(EX2.day == D1) & (EX2.addr == A)]
        two = sorted(['h102a2', 'h102a3'], key=hk)
        ok(list(a1.mk_hash) == mf + two and list(a1.rule)[2:] == ['removal_set_ge_2_explanations', 'single_fill'], f'variant examples {list(a1.mk_hash)}')
        ok(set(csv(dest2, f'timeline_{D1}_102').scenario.dropna()) == {'S_NOPRIO'}, 'variant timeline order columns from S_NOPRIO')
        x = one(csv(dest2, 'econ_segments'), 'v life_exh', id_set='exante_sh', grp='in', day=D1, role='maker', dim='life_exh', bucket='na_not_book')
        ok(str(x.r1_contradicted) == 'True' and x.r1_readings == 'T1=INSUFFICIENT; T2=contradicted', 'variant: S_FIFO-dependent segments flagged')
        ok(J2['R2']['unique_defects']['total'] == 0, 'variant: no unique defect')
    print(json.dumps(dict(selftest='pf_mt_report', assertions_passed=N_OK[0], result='PASS')), flush=True)


if __name__ == '__main__':
    main()
