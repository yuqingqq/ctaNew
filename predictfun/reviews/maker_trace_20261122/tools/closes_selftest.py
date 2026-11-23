"""Selftest of tools/closes.py (MAKER-TRACE decision contract amendment 2, item 1; session aea5bfb9, 2026-11-23). SYNTHETIC data only, in a
temporary directory (the tool's git queries read the repository, nothing else outside it is touched). The econ inputs are built from hand-written
legs by pf_mt_econ's own fifo (mq / rq) and market_table (maker_market), so the tool is reconciled against econ's real code; every expected close,
sum and reading below is computed by hand (comments), never by the tool.
World (YES space; V = 1 Up / 0 Down; ms after 01:00Z; fills_valued row groups: ID day | D1 101-103 | D1 104-106 + D2 201 (mixed) | D2 202-203):
 ID 9001: never read (its row group lies outside the measurement days).
 D1 101 rank 1 (V 1, END .80): maker buy 10 @.40, taker sell 4 @.45 (fee .02 C), a self-trade 2 @.47 (fee .01 C), maker sell 10 @.50 (10 s mark
    stale), taker buy 8 @.42 (fee .1 YES shares), an excluded leg -> 4 m->t +.20, 6 m->m +.60, 4 m->t +.32 (fees .021 / .05); residual long 4 @.42
    (END +1.52, resolution +2.32); econ gross 3.44 = 1.12 + 2.32, net 3.31; max |position| 10.
 D1 102 rank 2 (V 0): taker sell 5 @.60, taker buy 3 @.55, maker buy 6 @.50, maker sell 4 @.52 -> 3 t->t +.15, 2 t->m +.20, 4 m->m +.08; flat.
 D1 103 rank 3 (V 0, END .30): buy 10 @.40, sell 6 @.45 -> 6 m->m +.30; residual 4 (END -.40, resolution -1.60): good turnover lost to residual.
 D1 104 rank 5 (V 1): taker buy 10 @.50 (fee .4 YES shares = .2 at the fill price, .4 at resolution), maker sell 10 @.53 -> net +.10 / -.10.
 D1 105 rank 4 (V 1, END stale): one buy 5 @.30 -> no close; residual +3.50, END n/a.
 D1 106 rank 6 (V 1): buys .1 / .2, sells .3 / 1.0, buy 1.0 -> 3 closes + 1 dust close (0.1 + 0.2 != 0.3 in floats).
 D2 201 rank 3 (V 0): taker buy 5 @.50, taker sell 5 @.49 (fees .05 + .05) -> 5 t->t -.05, net -.15: turnover itself poor.
 D2 202 rank 1 (V 1, END .70): equal-ms maker buys zz-b 5 @.30 / aa-a 5 @.35 (sid order: aa-a first), taker sell 7 @.40 (fee .07 C), one taker
    record mm-c buying 3 @.36 (px_rank 0, mk_idx 1, fee .03 sh) + 2 @.38 (px_rank 1, mk_idx 0, fee .02 sh), maker sell 6 @.45 -> 5 m->t +.25,
    2 m->t +.20, 3 m->m +.45, 3 t->m +.27; residual 2 @.38 (END +.64, resolution +1.24); econ gross 2.41, net 2.29.
 D2 203 rank 2 (no resolution): 4 m->m +.08 in closes.parquet only.
Refusals (each asserts its reason; no partial directory may remain): gross reconciliation, attribution, row completeness, write-once (output checked
before any input), the top 3, a non-consumed day in the row-group statistics, the days argument, a second spelling of a top-10 address, the
memory guard, a second run onto a finished output. The depth fallback (no dpt column) runs on the four default measurement days.
Usage (run_light): closes_selftest.py"""
import json
import math
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import closes as X  # noqa: E402

C, E = X.C, X.E
D0, D1, D2 = C.ID_DAY, '20261024', '20261025'
B0 = {D0: 1792717200000, D1: 1792803600000, D2: 1792890000000}          # 01:00Z of each day, ms
NAN = float('nan')
R = X.TOP3 + tuple('0x' + format(i, '040x') for i in range(4, 11))      # ex-ante ranks 1-10 (4-10 synthetic)
FA = tuple('0xf' + format(i, '039x') for i in range(1, 10))            # field addresses (never read by the tool)
r1, r2, r3, r4, r5, r6, r7 = R[:7]
f1, f2, f3, f4, f5, f6 = FA[:6]
# market -> (day, V, END mid, st_res, END status)
MKT = {9001: (D0, 1., NAN, 'ok', 'id_day_no_book'), 101: (D1, 1., .80, 'ok', 'ok'), 102: (D1, 0., .10, 'ok', 'ok'), 103: (D1, 0., .30, 'ok', 'ok'),
       104: (D1, 1., .90, 'ok', 'ok'), 105: (D1, 1., NAN, 'ok', 'stale'), 106: (D1, 1., .50, 'ok', 'ok'), 201: (D2, 0., .20, 'ok', 'ok'),
       202: (D2, 1., .70, 'ok', 'ok'), 203: (D2, NAN, .50, 'no_resolution', 'ok')}
N_DUST = int((.1 + .2) != .3)                                         # 1 in IEEE doubles


def leg(mid, sid, t, maker, taker, s_mk, q, p, mk_idx=0, px=0, mfee=(0., 'COLLATERAL'), tfee=(0., 'COLLATERAL'), status='ok', marks=None, dpt='d0',
        mm=None):
    """One maker leg: s_mk = +1 the maker buys YES (its taker sells), -1 the maker sells; marks = (m_1s, m_10s, m_60s), None = stale."""
    return dict(mid=mid, sid=sid, t=t, maker=maker, taker=taker, s_mk=s_mk, q=q, p=p, mk_idx=mk_idx, px=px, mfee=mfee, tfee=tfee, status=status,
                marks=marks, dpt=dpt, mm=mm)


LEGS = [
    leg(9001, 'id-0001', 1000, r1, f1, +1, 1., .50), leg(9001, 'id-0002', 2000, r7, f2, -1, 1., .50),
    leg(101, 'a1-0001', 1000, r1, f1, +1, 10., .40, marks=(.41, .42, .43), dpt='d0', mm=.405),
    leg(101, 'a1-0002', 2000, f2, r1, +1, 4., .45, tfee=(.02, 'COLLATERAL'), marks=(.44, .46, .45), dpt='d0'),
    leg(101, 'a1-0003', 2500, r1, r1, +1, 2., .47, tfee=(.01, 'COLLATERAL'), dpt='d0'),                       # self-trade
    leg(101, 'a1-0004', 3000, r1, f3, -1, 10., .50, marks=(.49, None, .52), dpt='d1'),                         # 10 s mark stale
    leg(101, 'a1-0005', 4000, f4, r1, -1, 8., .42, tfee=(.1, 'SHARES'), marks=(.43, .44, .40), dpt='d2', mm=.41),
    leg(101, 'a1-0006', 5000, r1, f5, +1, 3., .30, status='src_status'),                                        # excluded
    leg(102, 'b2-0001', 100, f1, r2, +1, 5., .60), leg(102, 'b2-0002', 200, f2, r2, -1, 3., .55),
    leg(102, 'b2-0003', 300, r2, f3, +1, 6., .50), leg(102, 'b2-0004', 400, r2, f4, -1, 4., .52),
    leg(103, 'c3-0001', 1000, r3, f1, +1, 10., .40), leg(103, 'c3-0002', 2000, r3, f2, -1, 6., .45),
    leg(104, 'e5-0001', 1000, f1, r5, -1, 10., .50, tfee=(.4, 'SHARES')), leg(104, 'e5-0002', 2000, r5, f2, -1, 10., .53),
    leg(105, 'g4-0001', 1000, r4, f1, +1, 5., .30),
    leg(106, 'h6-0001', 100, r6, f1, +1, .1, .40), leg(106, 'h6-0002', 200, r6, f2, +1, .2, .41), leg(106, 'h6-0003', 300, f3, r6, +1, .3, .45),
    leg(106, 'h6-0004', 400, r6, f4, -1, 1., .50), leg(106, 'h6-0005', 500, f5, r6, -1, 1., .46),
    leg(201, 'p3-0001', 1000, f1, r3, -1, 5., .50, tfee=(.05, 'COLLATERAL')), leg(201, 'p3-0002', 2000, f2, r3, +1, 5., .49, tfee=(.05, 'COLLATERAL')),
    leg(202, 'zz-b', 1000, r1, f1, +1, 5., .30), leg(202, 'aa-a', 1000, r1, f2, +1, 5., .35),
    leg(202, 'kk-s', 2000, f3, r1, +1, 7., .40, tfee=(.07, 'COLLATERAL')),
    leg(202, 'mm-c', 3000, f4, r1, -1, 2., .38, mk_idx=0, px=1, tfee=(.02, 'SHARES')),
    leg(202, 'mm-c', 3000, f5, r1, -1, 3., .36, mk_idx=1, px=0, tfee=(.03, 'SHARES')),
    leg(202, 'pp-d', 4000, r1, f6, -1, 6., .45),
    leg(203, 'u2-0001', 1000, r2, f1, +1, 4., .50), leg(203, 'u2-0002', 2000, r2, f2, -1, 4., .52),
]
# hand-computed closes: key (day, mid, rank, entry sid, entry mk_idx, entry role, close sid, close mk_idx, close role) ->
#   shares, P&L = shares x (sell - buy), fee_fill entry / close, fee_res entry / close (prorated by shares), hold ms, entry side
EXP = [
    (D1, 101, 1, 'a1-0001', 0, 'maker', 'a1-0002', 0, 'taker', 4., .20, 0., .02, 0., .02, 1000, 'buy'),        # 4 x (.45 - .40)
    (D1, 101, 1, 'a1-0001', 0, 'maker', 'a1-0004', 0, 'maker', 6., .60, 0., 0., 0., 0., 2000, 'buy'),          # 6 x (.50 - .40)
    (D1, 101, 1, 'a1-0004', 0, 'maker', 'a1-0005', 0, 'taker', 4., .32, 0., .021, 0., .05, 1000, 'sell'),      # 4 x (.50 - .42); .042 / .1 x 4/8
    (D1, 102, 2, 'b2-0001', 0, 'taker', 'b2-0002', 0, 'taker', 3., .15, 0., 0., 0., 0., 100, 'sell'),
    (D1, 102, 2, 'b2-0001', 0, 'taker', 'b2-0003', 0, 'maker', 2., .20, 0., 0., 0., 0., 200, 'sell'),
    (D1, 102, 2, 'b2-0003', 0, 'maker', 'b2-0004', 0, 'maker', 4., .08, 0., 0., 0., 0., 100, 'buy'),
    (D1, 103, 3, 'c3-0001', 0, 'maker', 'c3-0002', 0, 'maker', 6., .30, 0., 0., 0., 0., 1000, 'buy'),
    (D1, 104, 5, 'e5-0001', 0, 'taker', 'e5-0002', 0, 'maker', 10., .30, .2, 0., .4, 0., 1000, 'buy'),
    (D1, 106, 6, 'h6-0001', 0, 'maker', 'h6-0003', 0, 'taker', .1, .005, 0., 0., 0., 0., 200, 'buy'),
    (D1, 106, 6, 'h6-0002', 0, 'maker', 'h6-0003', 0, 'taker', .2, .008, 0., 0., 0., 0., 100, 'buy'),
    (D1, 106, 6, 'h6-0004', 0, 'maker', 'h6-0005', 0, 'taker', 1., .04, 0., 0., 0., 0., 100, 'sell'),
    (D2, 201, 3, 'p3-0001', 0, 'taker', 'p3-0002', 0, 'taker', 5., -.05, .05, .05, .05, .05, 1000, 'buy'),
    (D2, 202, 1, 'aa-a', 0, 'maker', 'kk-s', 0, 'taker', 5., .25, 0., .05, 0., .05, 1000, 'buy'),             # .07 x 5/7
    (D2, 202, 1, 'zz-b', 0, 'maker', 'kk-s', 0, 'taker', 2., .20, 0., .02, 0., .02, 1000, 'buy'),             # .07 x 2/7
    (D2, 202, 1, 'zz-b', 0, 'maker', 'pp-d', 0, 'maker', 3., .45, 0., 0., 0., 0., 3000, 'buy'),
    (D2, 202, 1, 'mm-c', 1, 'taker', 'pp-d', 0, 'maker', 3., .27, .0108, 0., .03, 0., 1000, 'buy'),           # .03 sh x .36 at the fill
    (D2, 203, 2, 'u2-0001', 0, 'maker', 'u2-0002', 0, 'maker', 4., .08, 0., 0., 0., 0., 1000, 'buy'),
]
DUST = (D1, 106, 6, 'h6-0002', 0, 'maker', 'h6-0004', 0, 'maker')
OUTS = sorted(['answer.json', 'closes.parquet', 'closes_markets.parquet', 'closes_receipt.json', 'closes_summary.csv', 'cohorts.csv',
               'residual_summary.csv'])
RES = dict(n=0, fail=[])


def check(name, cond, detail=''):
    RES['n'] += 1
    if not cond:
        RES['fail'].append(f'{name} {detail}'.strip())
        print(f'FAIL {name} {detail}', flush=True)


def near(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol


def block(name, fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        RES['fail'].append(f'{name}: uncaught {type(e).__name__}: {e}')
        print(f'FAIL {name}: uncaught {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        return None


def expect(name, fn, substr, parent=None):
    try:
        fn()
    except ValueError as e:
        check(name, substr in str(e), f'(message: {e})')
    except Exception as e:  # noqa: BLE001
        check(name, False, f'(raised {type(e).__name__}: {e})')
    else:
        check(name, False, '(did not refuse)')
    if parent is not None:
        check(f'{name}: no partial directory left', not list(Path(parent).glob('.closes.partial_*')))


def pick(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k].astype(str) == str(v)).to_numpy()
    sub = df[m]
    X.need(len(sub) == 1, f'selftest: {len(sub)} rows match {kw}')
    return sub.iloc[0]


# ------------------------------------------------------------------ the synthetic world ------------------------------------------------------------------
def rows_df():
    """Both rows (maker, taker) of every leg in the pf_mt_econ.FV schema, with econ's value rules (gross, marks, fees at fill / resolution)."""
    out = []
    for g in LEGS:
        day, V, m_end, st_res, end_st = MKT[g['mid']]
        idd = day == D0
        settle = B0[day] + g['t']
        marks = g['marks'] or (g['p'],) * 3
        mm = g['mm'] if g['mm'] is not None else g['p']
        for role in ('maker', 'taker'):
            mk = role == 'maker'
            addr, cp = (g['maker'], g['taker']) if mk else (g['taker'], g['maker'])
            s = g['s_mk'] if mk else -g['s_mk']
            amt, ft = g['mfee'] if mk else g['tfee']
            dy, dc = E.fee_pos(np.array([amt]), np.array([ft], dtype=object), np.array([1]))
            dy, dc = float(dy[0]), float(dc[0])
            q, p = g['q'], g['p']
            gross = s * q * (V - p)
            fee_res = -(dc + (dy * V if dy != 0 else 0.))
            r = dict(day=day, mid=g['mid'], sid=g['sid'], mk_idx=g['mk_idx'], px_rank=g['px'], role=role, addr=addr,
                     hash=f"mh_{g['sid']}_{g['mk_idx']}" if mk else f"th_{g['sid']}", cp_addr=cp, kind='regular', own_index_set=1,
                     y_side='bid' if s == 1 else 'ask', s=s, q=q, p=p, px_gap=NAN, self_trade=g['maker'] == g['taker'], settle_ms=settle, t_ms=settle,
                     win_ms=-1, t_status='hi_no_lifecycle', t_rel_s=10., tseg='w000_060', pseg='d_0.0_0.1', V=V, st_res=st_res, gross_res=gross,
                     m_minus=NAN if idd else mm, age_ms=0, spr_w=.01, st_spr='id_day_no_book' if idd else 'ok', v_spr=NAN if idd else s * q * (mm - p))
            r.update(st_spr_hi=r['st_spr'], v_spr_hi=r['v_spr'])
            for h, m in zip(('1s', '10s', '60s'), marks):
                okm = (not idd) and m is not None
                r[f'st_{h}'] = 'ok' if okm else ('id_day_no_book' if idd else 'stale')
                r[f'v_{h}'] = s * q * (m - p) if okm else NAN
            oke = (not idd) and end_st == 'ok'
            r.update(m_end=m_end if oke else NAN, st_end=end_st, v_end=s * q * (m_end - p) if oke else NAN, fee=amt, fee_type=ft, fee_dy=dy, fee_dc=dc,
                     fee_fill=-(dc + dy * p), fee_res=fee_res, net_res=gross - fee_res, reb25_fill=0., reb25_res=0., mq=NAN, rq=NAN,
                     life_rep='na_not_book' if mk else 'na_taker', life_exh='na_not_book' if mk else 'na_taker',
                     life_front='na_not_book' if mk else 'na_taker', life_status='na_not_book' if mk else 'na_taker', dpt=g['dpt'],
                     sweep='single_price', cp='na_id_day' if idd else 'cp_other', boost='unknown', src_status=g['status'], status=g['status'])
            out.append(r)
    F = pd.DataFrame(out)
    X.need(set(F.columns) == set(E.FV), f'selftest world: columns differ from pf_mt_econ.FV: {sorted(set(F.columns) ^ set(E.FV))}')
    return F[list(E.FV)]


def econ_world(F):
    """pf_mt_econ.fifo (mq / rq) and market_table (maker_market) per day, and econ's own conservation check on the synthetic world."""
    parts, mms = [], []
    for d in (D0, D1, D2):
        Fd = F[(F['day'] == d).to_numpy()].copy().reset_index(drop=True)
        for c in ('addr', 'sid'):
            Fd[c] = pd.Categorical(Fd[c].astype(object))
        Fd['role'] = pd.Categorical(Fd['role'].astype(object), categories=['maker', 'taker'])
        E.fifo(Fd)
        mids = sorted(int(m) for m in set(Fd['mid'].tolist()))
        reg = pd.DataFrame({'mid': np.array(mids, np.int64), 'V': [MKT[m][1] for m in mids], 'censored': False, 'upbad': False, 'tie': False})
        end = {m: (MKT[m][2], MKT[m][4]) for m in mids}
        M = E.market_table(Fd, d, 'ID' if d == D0 else 'FILL', [], end, reg)
        E.check_market(M)
        parts.append(Fd.astype({'addr': object, 'sid': object, 'role': object}))
        mms.append(M)
    return pd.concat(parts, ignore_index=True)[list(E.FV)], pd.concat(mms, ignore_index=True)


def world(root, F, MM, ranks=R, spec=None, extra=None):
    """econ/{fills_valued, maker_market}.parquet + identities.json under root, and the receipt parent root/rcpt."""
    ec = root / 'econ'
    ec.mkdir(parents=True)
    spec = spec or E.FV
    w = pq.ParquetWriter(ec / 'fills_valued.parquet', E._arrow_schema(spec))
    for m in (F['day'] == D0, F['mid'].isin([101, 102, 103]), F['mid'].isin([104, 105, 106, 201]), F['mid'].isin([202, 203])):
        w.write_table(E._to_arrow(F.loc[m.to_numpy(), list(spec)].reset_index(drop=True), spec))
    if extra is not None:
        w.write_table(E._to_arrow(extra[list(spec)].reset_index(drop=True), spec))
    w.close()
    w = pq.ParquetWriter(ec / 'maker_market.parquet', E._arrow_schema(E.MM))
    for d in (D0, D1, D2):
        w.write_table(E._to_arrow(MM.loc[(MM['day'] == d).to_numpy(), list(E.MM)].reset_index(drop=True), E.MM))
    w.close()
    (ec / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(day=D0, top10_shares=[dict(rank=i + 1, addr=a) for i, a in enumerate(ranks)]))))
    (root / 'rcpt').mkdir()
    return root


def run_tool(root, days=(D1, D2), **kw):
    return X.run(inp=root, out=root / 'rcpt' / 'closes', days=days, decision_receipt=root / 'no_decision.json', log=lambda *a, **k: None, **kw)


# ------------------------------------------------------------------ checks ------------------------------------------------------------------
def chk_world(F, MM):
    a = pick(MM, day=D1, mid=101, addr=r1)
    check('world: econ D1 101 rank 1 gross 3.44, rt 1.12, net 3.31, 6 ok + 1 excluded rows', near(a['gross_res_mk'] + a['gross_res_tk'], 3.44)
          and near(a['rt_pnl'], 1.12) and near(a['net_res'], 3.31) and int(a['n_mk'] + a['n_tk']) == 6 and int(a['n_excl']) == 1)
    b = pick(MM, day=D2, mid=202, addr=r1)
    check('world: econ D2 202 rank 1 gross 2.41, rt 1.17, residual 1.24, drift .64', near(b['gross_res_mk'] + b['gross_res_tk'], 2.41)
          and near(b['rt_pnl'], 1.17) and near(b['resid_res'], 1.24) and near(b['resid_drift'], .64))
    check('world: econ D1 103 rank 3 residual -1.6', near(pick(MM, day=D1, mid=103, addr=r3)['resid_res'], -1.6))
    s = F[(F['sid'] == 'a1-0003').to_numpy()]
    check('world: the self-trade rows carry mq = rq = 0', len(s) == 2 and bool((s['mq'] == 0).all() and (s['rq'] == 0).all()))


def chk_closes(out):
    T = pq.read_table(out / 'closes.parquet').to_pandas()
    check('closes: schema', list(pq.read_schema(out / 'closes.parquet').names) == list(X.CL))
    check('closes: row count', len(T) == len(EXP) + N_DUST, f'({len(T)})')
    got = {(str(r['day']), int(r['mid']), int(r['rank']), str(r['entry_sid']), int(r['entry_mk_idx']), str(r['entry_role']), str(r['close_sid']),
            int(r['close_mk_idx']), str(r['close_role'])): r for _, r in T.iterrows()}
    check('closes: keys unique', len(got) == len(T))
    for e in EXP:
        k, (sh, pnl, ffe, ffc, fre, frc, hold, side) = e[:9], e[9:]
        r = got.get(k)
        if r is None:
            check(f'closes: {k} present', False)
            continue
        ok = (near(r['sh'], sh) and near(r['pnl'], pnl) and near(r['fee_fill_entry'], ffe) and near(r['fee_fill_close'], ffc)
              and near(r['fee_res_entry'], fre) and near(r['fee_res_close'], frc) and int(r['hold_ms']) == hold and str(r['entry_side']) == side
              and not bool(r['dust']) and str(r['role_pair']) == f'{e[5]}->{e[8]}' and near(r['net'], pnl - ffe - ffc)
              and near(r['net_feeres'], pnl - fre - frc) and bool(r['res_ok']) == (e[1] != 203))
        check(f'closes: {k}', ok, f"(got sh {r['sh']!r} pnl {r['pnl']!r} fees {r['fee_fill_entry']!r} {r['fee_fill_close']!r} {r['fee_res_entry']!r} "
                                  f"{r['fee_res_close']!r} hold {r['hold_ms']} side {r['entry_side']} pair {r['role_pair']} dust {r['dust']})")
    if N_DUST:
        r = got.get(DUST)
        check('closes: the dust close h6-0002 -> h6-0004 is flagged', r is not None and bool(r['dust']) and float(r['sh']) < X.DUST_SH)
    extra = set(got) - {e[:9] for e in EXP} - ({DUST} if N_DUST else set())
    check('closes: no unexpected close', not extra, str(extra))
    r = got[(D1, 101, 1, 'a1-0001', 0, 'maker', 'a1-0002', 0, 'taker')]
    check('closes: each leg keeps its own hash, price and FIFO clock', str(r['entry_hash']) == 'mh_a1-0001_0' and str(r['close_hash']) == 'th_a1-0002'
          and near(r['entry_p'], .40) and near(r['close_p'], .45) and int(r['entry_settle_ms']) == B0[D1] + 1000
          and int(r['close_settle_ms']) == B0[D1] + 2000 and str(r['addr']) == r1)
    r = got[(D2, 202, 1, 'mm-c', 1, 'taker', 'pp-d', 0, 'maker')]
    check('closes: price priority inside record mm-c (px_rank 0 = mk_idx 1 @.36 closes first)', str(r['entry_hash']) == 'th_mm-c' and near(r['entry_p'], .36))


def chk_markets(out):
    M = pq.read_table(out / 'closes_markets.parquet').to_pandas()
    check('markets: 9 address x market x day rows', len(M) == 9, f'({len(M)})')
    a = pick(M, day=D1, mid=101, rank=1)
    for k, v in dict(n_rows_ok=6, n_rows_self=2, n_rows_excl=1, n_closes=3, n_dust=0, close_sh=14., close_pnl=1.12, close_fee_fill=.041,
                     close_fee_res=.07, resid_sh=4., resid_res=2.32, resid_end=1.52, resid_resol=.8, resid_fee_fill=.021, resid_fee_res=.05, self_sh=2.,
                     self_gross=0., self_fee_fill=.01, self_fee_res=.01, fee_fill_tot=.072, fee_res_tot=.13, net_pos=4., max_abs_pos=10.,
                     econ_gross=3.44, econ_net=3.31, econ_rt_pnl=1.12, dev_gross=0., dev_net=0.).items():
        check(f'markets: D1 101 rank 1 {k}', near(a[k], v), f'(got {a[k]!r}, hand {v!r})')
    b = pick(M, day=D2, mid=202, rank=1)
    for k, v in dict(n_closes=4, close_sh=13., close_pnl=1.17, close_fee_fill=.0808, close_fee_res=.10, resid_sh=2., resid_res=1.24, resid_end=.64,
                     resid_resol=.6, resid_fee_fill=.0076, resid_fee_res=.02, net_pos=2., max_abs_pos=10., econ_gross=2.41, econ_net=2.29).items():
        check(f'markets: D2 202 rank 1 {k}', near(b[k], v), f'(got {b[k]!r}, hand {v!r})')
    c = pick(M, day=D1, mid=105, rank=4)
    check('markets: D1 105 rank 4 (no END mark): residual 3.5, END n/a', near(c['resid_res'], 3.5) and math.isnan(float(c['resid_end']))
          and not bool(c['end_ok']) and bool(c['res_ok']) and int(c['n_closes']) == 0)
    u = pick(M, day=D2, mid=203, rank=2)
    check('markets: D2 203 rank 2 (no resolution): V-free close kept, resolution values blank', near(u['close_pnl'], .08) and not bool(u['res_ok'])
          and math.isnan(float(u['resid_res'])) and math.isnan(float(u['econ_gross'])) and near(u['econ_rt_pnl'], .08))
    r = M[M['res_ok'].to_numpy(bool)]
    check('markets: every resolved market reconciles (|dev_gross|, |dev_net| <= 1e-9)', len(r) == 8 and bool((r['dev_gross'].abs() <= 1e-9).all())
          and bool((r['dev_net'].abs() <= 1e-9).all()))
    d = pick(M, day=D1, mid=106, rank=6)
    check('markets: D1 106 rank 6: 3 closes + the dust, flat', int(d['n_closes']) == 3 and int(d['n_dust']) == N_DUST and abs(float(d['resid_sh'])) < 1e-12
          and near(d['close_pnl'], .053) and near(d['close_sh'], 1.3))


def chk_summary(out):
    S = pd.read_csv(out / 'closes_summary.csv', float_precision='round_trip', dtype={'day': str, 'scope': str, 'role_pair': str})
    check('summary: rows = 12 scopes x 3 days x 5 pairs', len(S) == 180, f'({len(S)})')
    a = pick(S, scope='rank1', day=D1, role_pair='all')
    for k, v in dict(n_closes=3, n_dust=0, sh=14., gross=1.12, fee_fill=.041, fee_res=.07, net=1.079, net_feeres=1.05, c_net_sh=100 * 1.079 / 14,
                     hold_p10_s=1., hold_p25_s=1., hold_p50_s=1., hold_p75_s=2., hold_p90_s=2., hold_med_count_s=1., hold_mean_s=20 / 14).items():
        check(f'summary: rank1 D1 all {k}', near(a[k], v), f'(got {a[k]!r}, hand {v!r})')
    b = pick(S, scope='rank1', day=D1, role_pair='maker->taker')
    check('summary: rank1 D1 maker->taker', int(b['n_closes']) == 2 and near(b['sh'], 8.) and near(b['gross'], .52) and near(b['net'], .479))
    b = pick(S, scope='rank1', day=D1, role_pair='maker->maker')
    check('summary: rank1 D1 maker->maker', int(b['n_closes']) == 1 and near(b['sh'], 6.) and near(b['gross'], .60))
    b = pick(S, scope='top3', day='all', role_pair='taker->taker')
    check('summary: top3 all taker->taker (rank 2 +.15; rank 3 -.05 with .10 fees)', int(b['n_closes']) == 2 and near(b['sh'], 8.)
          and near(b['gross'], .10) and near(b['fee_fill'], .10) and near(b['net'], 0.))
    b = pick(S, scope='top3', day='all', role_pair='taker->maker')
    check('summary: top3 all taker->maker (rank 2 +.20; rank 1 +.27)', int(b['n_closes']) == 2 and near(b['sh'], 5.) and near(b['gross'], .47)
          and near(b['fee_fill'], .0108))
    b = pick(S, scope='rank2', day=D2, role_pair='all')
    check('summary: rank2 D2: the unresolved close counted, not summed', int(b['n_closes']) == 0 and int(b['n_closes_res_na']) == 1
          and near(b['sh_res_na'], 4.) and near(b['sh'], 0.))
    b = pick(S, scope='rank6', day=D1, role_pair='all')
    check('summary: rank6 D1: 3 closes, the dust not counted', int(b['n_closes']) == 3 and int(b['n_dust']) == N_DUST and near(b['sh'], 1.3)
          and near(b['gross'], .053))
    b = pick(S, scope='top10', day='all', role_pair='all')
    check('summary: top10 all: 16 resolved closes, 1 unresolved', int(b['n_closes']) == 16 and int(b['n_closes_res_na']) == 1)
    b = pick(S, scope='rank1', day='all', role_pair='all')
    check('summary: rank1 all days', int(b['n_closes']) == 7 and near(b['sh'], 27.) and near(b['gross'], 2.29) and near(b['fee_fill'], .1218))


def chk_residual(out):
    S = pd.read_csv(out / 'residual_summary.csv', float_precision='round_trip', dtype={'day': str, 'scope': str})
    check('residual: rows = 12 scopes x 3 days', len(S) == 36, f'({len(S)})')
    a = pick(S, scope='rank1', day=D1)
    for k, v in dict(n_markets=1, n_markets_res=1, n_rows_excl=1, close_sh=14., close_net=1.079, resid_sh=4., resid_end=1.52, resid_resol=.8,
                     resid_res=2.32, max_abs_pos_max=10., self_sh=2., self_fee_fill=.01, econ_net=3.31, identity_dev=0., fee_drift=-.058).items():
        check(f'residual: rank1 D1 {k}', near(a[k], v), f'(got {a[k]!r}, hand {v!r})')
    b = pick(S, scope='rank4', day=D1)
    check('residual: rank4 D1 (no END mark): END n/a, residual 3.5', near(b['resid_res'], 3.5) and int(b['n_markets_end']) == 0
          and int(b['n_markets_end_na']) == 1 and math.isnan(float(b['resid_end'])))
    b = pick(S, scope='rank3', day=D1)
    check('residual: rank3 D1: -1.6 = END -.4 + resolution -1.2', near(b['resid_res'], -1.6) and near(b['resid_end'], -.4) and near(b['resid_resol'], -1.2))
    b = pick(S, scope='rank2', day=D2)
    check('residual: rank2 D2: the unresolved market counted', int(b['n_markets']) == 1 and int(b['n_markets_res']) == 0 and int(b['n_markets_res_na']) == 1)
    b = pick(S, scope='rank1', day=D2)
    check('residual: rank1 D2', near(b['resid_res'], 1.24) and near(b['resid_end'], .64) and near(b['resid_resol'], .6) and near(b['econ_net'], 2.29)
          and near(b['identity_dev'], 0.))
    check('residual: every identity deviation <= 1e-9', bool((S['identity_dev'].abs() <= 1e-9).all()))


def chk_answer(out):
    J = json.loads((out / 'answer.json').read_text())
    A = {(e['scope'], e['day']): e for e in J['entries']}
    nn, lost, poor = 'good_turnover_residual_not_negative', 'good_turnover_lost_to_residual', 'turnover_itself_poor'
    for k, v in {('rank1', D1): nn, ('rank1', D2): nn, ('rank2', D1): nn, ('rank2', D2): 'no_resolved_markets', ('rank2', 'all'): nn,
                 ('rank3', D1): lost, ('rank3', D2): poor, ('rank3', 'all'): lost, ('rank4', D1): 'no_closes', ('rank4', D2): 'no_rows',
                 ('rank5', D1): nn, ('rank6', D1): nn, ('rank7', D1): 'no_rows', ('rank7', 'all'): 'no_rows', ('top3', D1): nn}.items():
        check(f'answer: {k} reading', A[k]['reading'] == v, f"(got {A[k]['reading']})")
    e = A[('rank5', D1)]
    check('answer: rank5 D1 flips with the fee convention', e['reading_fee_res'] == poor and e['fee_convention_sensitive'] is True
          and near(e['close_net'], .10) and near(e['close_net_feeres'], -.10))
    check('answer: rank1 D1 not fee-sensitive', A[('rank1', D1)]['fee_convention_sensitive'] is False)
    e = A[('rank3', D1)]
    check('answer: rank3 D1 numbers with n', near(e['close_net'], .30) and near(e['resid_res'], -1.6) and e['n_closes'] == 1 and near(e['close_sh'], 6.)
          and near(e['c_close_net_sh'], 5.) and e['n_markets_res'] == 1)
    t = J['tally']['top3']
    check('answer: top-3 tally over the six address-days', t['n_address_days'] == 6 and t['reading'] == dict(
        no_rows=0, no_resolved_markets=1, no_closes=0, turnover_itself_poor=1, good_turnover_lost_to_residual=1, good_turnover_residual_not_negative=3), str(t))
    check('answer: ci and days', J['ci'] == 'none_G2_lt_5' and J['days'] == [D1, D2])


def chk_cohorts(out):
    K = pd.read_csv(out / 'cohorts.csv', float_precision='round_trip', dtype={'day': str, 'scope': str, 'role': str, 'depth': str})
    check('cohorts: rows = 5 scopes x 3 days x 3 roles x 7 depths', len(K) == 315, f'({len(K)})')
    a = pick(K, scope='rank1', day=D1, role='maker', depth='d0')
    for k, v in dict(n_cohort=1, sh_cohort=10., gross_1s=.1, gross_10s=.2, gross_60s=.3, gross_end=4., gross_res=6., c_gross_1s_own=1.,
                     c_gross_10s_own=2., c_gross_60s_own=3., c_gross_end_own=40., c_gross_res_own=60., c_net_res_own=60., sh_common=32.,
                     c_gross_1s_common=100 * .1 / 32).items():
        check(f'cohorts: rank1 D1 maker d0 {k}', near(a[k], v), f'(got {a[k]!r}, hand {v!r})')
    a = pick(K, scope='rank1', day=D1, role='maker', depth='d1')
    check('cohorts: rank1 D1 maker d1 (10 s mark stale): outside the cohort, inside the resolved shares', int(a['n_cohort']) == 0
          and near(a['sh_cohort'], 0.) and math.isnan(float(a['c_gross_1s_own'])) and int(a['n_res_all']) == 1 and near(a['sh_res_all'], 10.)
          and near(a['c_gross_res_all'], -50.))
    a = pick(K, scope='rank1', day=D1, role='maker', depth='all')
    check('cohorts: rank1 D1 maker all: coverage .5', int(a['n_cohort']) == 1 and near(a['sh_res_all'], 20.) and near(a['coverage'], .5)
          and near(a['gross_res_all'], 1.) and near(a['c_gross_res_all'], 5.))
    a = pick(K, scope='rank1', day=D1, role='taker', depth='all')
    for k, v in dict(n_cohort=2, sh_cohort=12., gross_1s=.12, gross_10s=.12, gross_60s=-.16, gross_end=1.64, gross_res=2.44, net_1s=.058, net_10s=.058,
                     net_60s=-.222, net_end=1.578, net_res=2.32, c_net_1s_own=100 * .058 / 12, c_net_res_own=100 * 2.32 / 12,
                     c_gross_1s_common=100 * .12 / 32, c_net_res_common=100 * 2.32 / 32).items():
        check(f'cohorts: rank1 D1 taker all {k}', near(a[k], v), f'(got {a[k]!r}, hand {v!r})')
    a = pick(K, scope='rank1', day=D1, role='all', depth='all')
    check('cohorts: rank1 D1 all all', int(a['n_cohort']) == 3 and near(a['sh_cohort'], 22.) and near(a['net_1s'], .158)
          and near(a['c_net_1s_own'], 100 * .158 / 22))
    check('cohorts: top3 D1 common denominator 32 + 18 + 16', near(pick(K, scope='top3', day=D1, role='all', depth='all')['sh_common'], 66.))
    check('cohorts: depth source dpt', bool((K['depth_source'] == 'dpt').all()))


def chk_receipt(rec, F, out):
    names = {'attributed', 'rt_pnl', 'rt_sh', 'gross_identity', 'gross_rows', 'resid_res', 'resid_end', 'net_pos', 'fee_fill', 'fee_res', 'net_identity',
             'fee_fill_attribution', 'fee_res_attribution', 'close_sh_eq_cap', 'resid_sh_eq_abs_net_pos'}
    ch = rec['checks']
    check('receipt: every reconciliation check ran and passed within 1e-9', set(ch) == names and all(v['ok'] and v['max_dev'] <= 1e-9 for v in ch.values()),
          str({k: v['max_dev'] for k, v in ch.items()}))
    c1, c2 = rec['counts'][D1], rec['counts'][D2]
    n1, n2, n3 = (int(F['mid'].isin(m).sum()) for m in ([101, 102, 103], [104, 105, 106, 201], [202, 203]))
    check('receipt: D1 counts', c1['rows'] == 21 and c1['rows_self'] == 2 and c1['rows_excl'] == 1 and c1['closes'] == 11 and c1['dust'] == N_DUST, str(c1))
    check('receipt: D2 counts', c2['rows'] == 10 and c2['closes'] == 6 and c2['dust'] == 0, str(c2))
    check('receipt: the ID-day row group never read; the mixed one read for both days', c1['fills_valued_rows_scanned'] == n1 + n2
          and c2['fills_valued_rows_scanned'] == n2 + n3 and c1['row_groups_fv_mixed'] == 1 and c2['row_groups_fv_mixed'] == 1)
    check('receipt: row-group statistics', rec['row_groups']['fills_valued'] == dict(row_groups=4, no_stats=0, skipped=1, mixed=1, empty=0)
          and rec['row_groups']['maker_market']['skipped'] == 1, str(rec['row_groups']))
    check('receipt: output rows', rec['outputs']['closes.parquet']['rows'] == len(EXP) + N_DUST and rec['outputs']['closes_markets.parquet']['rows'] == 9)
    check('receipt: inputs hashed, depth source, top 3, no decision receipt', all(len(v['sha256']) == 64 for v in rec['inputs'].values())
          and rec['depth_source'] == 'dpt' and rec['addresses'][0]['addr'] == X.TOP3[0] and rec['same_fills_valued_as_decision']['same'] is None)
    check('receipt: memory under the guard', 0 < rec['memory']['vmhwm_mib'] < C.GUARD_MIB)
    J = json.loads((out / 'closes_receipt.json').read_text())
    check('receipt: the written receipt is strict JSON with the tally', J['answer_tally']['top3']['n_address_days'] == 6 and 'git' in J)


def chk_second(base):
    out = base / 'rcpt' / 'closes'
    before = (out / 'closes_receipt.json').read_bytes()
    expect('refuse: a second run onto the finished output', lambda: run_tool(base), 'write-once', base / 'rcpt')
    check('refuse: the finished output unchanged', (out / 'closes_receipt.json').read_bytes() == before and sorted(p.name for p in out.iterdir()) == OUTS)


def chk_fallback(tmp, F, MM):
    root = world(tmp / 'fb', F, MM, spec={k: v for k, v in E.FV.items() if k != 'dpt'})
    rec = run_tool(root, days=None)
    out = root / 'rcpt' / 'closes'
    K = pd.read_csv(out / 'cohorts.csv', float_precision='round_trip', dtype={'day': str, 'scope': str, 'role': str, 'depth': str})
    check('fallback: depth source', rec['depth_source'] == 'm_minus_fallback' and bool((K['depth_source'] == 'm_minus_fallback').all()))
    a = pick(K, scope='rank1', day=D1, role='maker', depth='mm_0_1c')
    check('fallback: maker buy .40, mid .405 -> [0, 1) c', int(a['n_cohort']) == 1 and near(a['sh_cohort'], 10.) and near(a['sh_res_all'], 20.))
    a = pick(K, scope='rank1', day=D1, role='taker', depth='mm_1_2c')
    check('fallback: taker buy .42 from an ask, mid .41 -> [1, 2) c', int(a['n_cohort']) == 1 and near(a['sh_cohort'], 8.))
    a = pick(K, scope='rank1', day=D1, role='taker', depth='mm_0_1c')
    check('fallback: taker sell at the mid -> [0, 1) c', int(a['n_cohort']) == 1 and near(a['sh_cohort'], 4.))
    check('fallback: four default measurement days, two empty', rec['days'] == list(C.MEASURE_DAYS) and rec['counts']['20261026']['rows'] == 0
          and rec['counts']['20261027']['rows'] == 0 and rec['outputs']['closes.parquet']['rows'] == len(EXP) + N_DUST)
    check('fallback: ci with four days', json.loads((out / 'answer.json').read_text())['ci'] == 'none_G4_lt_5')


def refusal_cases(tmp, F, MM):
    sel = ((MM['day'] == D1) & (MM['mid'] == 101) & (MM['addr'] == r1)).to_numpy()

    def gross():
        M1 = MM.copy()
        M1.loc[sel, 'gross_res_mk'] = M1.loc[sel, 'gross_res_mk'] + .01
        root = world(tmp / 'r_gross', F, M1)
        expect('refuse: gross reconciliation (maker_market gross + 0.01)', lambda: run_tool(root), 'reconcile gross_identity', root / 'rcpt')
        check('refuse: gross reconciliation writes no output', not (root / 'rcpt' / 'closes').exists())

    def attributed():
        F2 = F.copy()
        s2 = ((F2['day'] == D1) & (F2['sid'] == 'a1-0001') & (F2['role'] == 'maker')).to_numpy()
        F2.loc[s2, 'mq'] = F2.loc[s2, 'mq'] - .5
        root = world(tmp / 'r_attr', F2, MM)
        expect('refuse: attribution (econ mq - 0.5 on one lot row)', lambda: run_tool(root), 'reconcile attributed', root / 'rcpt')

    def rows():
        M3 = MM.copy()
        M3.loc[sel, 'n_excl'] = M3.loc[sel, 'n_excl'] + 1
        root = world(tmp / 'r_rows', F, M3)
        expect('refuse: row completeness (maker_market n_excl + 1)', lambda: run_tool(root), 'reconcile rows', root / 'rcpt')

    def write_once():
        root = world(tmp / 'r_wo', F, MM)
        out = root / 'rcpt' / 'closes'
        out.mkdir()
        (out / 'sentinel').write_text('keep')
        (root / 'econ' / 'fills_valued.parquet').unlink()
        expect('refuse: write-once, checked before any input (fills_valued missing)', lambda: run_tool(root), 'write-once', root / 'rcpt')
        check('refuse: write-once leaves the existing output untouched', (out / 'sentinel').read_text() == 'keep'
              and sorted(p.name for p in out.iterdir()) == ['sentinel'])

    def top3():
        root = world(tmp / 'r_top3', F, MM, ranks=(R[1], R[0]) + R[2:])
        expect('refuse: ex-ante ranks 1-3 differ from the contract top 3', lambda: run_tool(root), 'contract top 3', root / 'rcpt')

    def bad_day():
        ex = F[(F['mid'] == 104).to_numpy()].copy()
        ex['day'] = '20261099'
        root = world(tmp / 'r_day', F, MM, extra=ex)
        expect('refuse: a row group whose day statistics leave the consumed days', lambda: run_tool(root), 'outside the consumed days', root / 'rcpt')

    def days_arg():
        nowhere = tmp / 'nowhere'
        expect('refuse: the ID day as a measurement day, before any path', lambda: X.run(inp=nowhere, out=nowhere / 'closes', days=(C.ID_DAY,),
                                                                                         log=lambda *a, **k: None), 'measurement days')
        expect('refuse: a repeated day', lambda: X.run(inp=nowhere, out=nowhere / 'closes', days=(D1, D1), log=lambda *a, **k: None), 'measurement days')
        check('refuse: nothing created for a refused days argument', not nowhere.exists())

    def spelling():
        F8 = F.copy()
        s8 = ((F8['day'] == D1) & (F8['sid'] == 'a1-0006') & (F8['role'] == 'maker')).to_numpy()
        F8.loc[s8, 'addr'] = r1.lower()
        root = world(tmp / 'r_spell', F8, MM)
        expect('refuse: a top-10 address under a second spelling', lambda: run_tool(root), 'another spelling', root / 'rcpt')

    def memory():
        root = world(tmp / 'r_mem', F, MM)
        expect('refuse: the memory self-guard fires', lambda: run_tool(root, guard_mib=1.), 'memory self-guard', root / 'rcpt')
        check('refuse: the memory guard writes no output', not (root / 'rcpt' / 'closes').exists())

    for name, fn in (('refuse gross', gross), ('refuse attributed', attributed), ('refuse rows', rows), ('refuse write-once', write_once),
                     ('refuse top3', top3), ('refuse day', bad_day), ('refuse days arg', days_arg), ('refuse spelling', spelling),
                     ('refuse memory', memory)):
        block(name, fn)


def main():
    c0 = time.process_time()
    tmp = Path(tempfile.mkdtemp(prefix='mt_closes_selftest_'))
    try:
        W = block('world build', lambda: econ_world(rows_df()))
        if W is not None:
            F, MM = W
            block('world', chk_world, F, MM)
            base = world(tmp / 'base', F, MM)
            rec = block('base run', run_tool, base)
            if rec is not None:
                out = base / 'rcpt' / 'closes'
                check('base: the outputs, no partial directory left', sorted(p.name for p in out.iterdir()) == OUTS
                      and not list((base / 'rcpt').glob('.closes.partial_*')))
                for name, fn in (('closes', chk_closes), ('markets', chk_markets), ('summary', chk_summary), ('residual', chk_residual),
                                 ('answer', chk_answer), ('cohorts', chk_cohorts)):
                    block(name, fn, out)
                block('receipt', chk_receipt, rec, F, out)
                block('second run', chk_second, base)
            block('fallback', chk_fallback, tmp, F, MM)
            refusal_cases(tmp, F, MM)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_fail = len(RES['fail'])
    print(json.dumps(dict(result='PASS' if not n_fail else 'FAIL', checks=RES['n'], failed=n_fail, failures=RES['fail'][:60],
                          cpu_s=round(time.process_time() - c0, 2)), indent=1), flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
