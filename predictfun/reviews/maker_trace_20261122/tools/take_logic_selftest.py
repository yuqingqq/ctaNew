"""take_logic_selftest.py - synthetic proof of take_logic.py (P-2026-005 MAKER-TRACE, TAKE-LOGIC v1; take_logic_contract.md fixtures 1-6).
The REAL chain_context functions (position / pending / btc_ret = pf_mt_decision.btc_feats / book_tob / time_left / align_cat) run on synthetic
markets built with chain_context.build_market + join_windows (settle-based windows); every expectation is planted or hand-computed.
Fixtures: (1) a planted take-profit maker (reduce hazard 0.25 per second at a_5s with, 0.05 flat, 0 against; 8 markets, long and short) reads
take_profit supported and stop_loss not; (2) a planted stop-loss maker (0.25 against, 0.05 flat, 0 with) reads stop_loss supported; (3) a
random-time maker (hazard 0.08 whatever the state, reduce or add at random) reads no mechanism supported on the fixture seed, while
stop_loss / take_profit / directional are testable (>= 5 supported markets, >= 30 events: the negative control could fire); (4) causality:
moving BTC after t*, moving the arrival times of every later fill (and the later book frames, plus a new later fill) leaves every D0 / D1000
state at t <= t* unchanged while states after t* change (positive control); (5) the circular shift keeps every pair's event count in every
replicate, stays inside its pair, is a bijection, deterministic and independent of the other pairs; (6) the MT_TL_MUT mutants flip_align /
leak_1s / active_span, flipped in-process, are each caught by a named check (F1 / F2, F4, POP) that evaluates False without raising. Also: bin
edges, the state definitions (hi <= t, fills in (t - 5 s, t], pending / excluded flags, edge sign and units, BTC wiring), event assignment at
the window edges (an event-second counts once), the population check, the Mantel-Haenszel RR on a hand-computed 2-stratum toy (25/22, also
through the per-instant path), tail counts and the verdict rule, the refusals (each asserting its reason) and every writer end to end.
Usage (run_light, MT_RUN=v2): take_logic_selftest.py"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import take_logic as TL  # noqa: E402

CX, C, X, E = TL.CX, TL.C, TL.X, TL.CT.E
NAN = float('nan')
RES = []
A1, A2, A3 = TL.TOP3
OTHER = '0x' + 'ab' * 20
DAYS = ('20261024', '20261025')
SIG_BP = 0.21                                    # BTC 100 ms step sd (bp): 5 s sd ~1.5 bp -> with / against / flat ~ 1/4 / 1/4 / 1/2
NG = 8_001                                       # grid rows: [ws - 400 s, we + 100 s]
FV_DT = {c: {'str': object, 'int64': np.int64, 'float64': np.float64, 'bool': bool}[E.FV[c]] for c in CX.FV_READ}
TAKE_COLS = ('day', 'mid', 'rank', 'addr', 'sid', 's', 'dq', 'p_avg', 'x_lo_hard_ms', 'hi_ms', 'settle_ms', 'status', 'take_class')
HAZ = dict(tp=(0.25, 0.0, 0.05), sl=(0.0, 0.25, 0.05))     # reduce hazard per second by a_5s category (with, against, flat)
KIND_ID = dict(tp=1, sl=2, random=3)
VERDICTS = ('supported', 'inconsistent', 'insufficient_support', 'not_supported', 'timing_sensitive')
EXPECT_CATCH = dict(flip_align=('F1', 'F2'), leak_1s=('F4',), active_span=('POP',))


def check(name, ok, info=''):
    RES.append((name, bool(ok), str(info)[:600]))


def expect(name, fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(name, needle in str(e), f'{type(e).__name__}: {e}')
        return
    check(name, False, 'did not raise')


def section(name, fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        check(f'{name}: section crashed', False, f'{type(e).__name__}: {e} | {traceback.format_exc()[-1500:]}')
        return None


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == 'f' or b.dtype.kind == 'f':
        return bool(np.array_equal(a.astype(float), b.astype(float), equal_nan=True))
    return bool(np.array_equal(a, b))


# ------------------------------------------------------------------ synthetic markets ------------------------------------------------------------------
def bt_from_lm(G0, lm):
    """pf_mt_decision.load_btc's dense layout from a log-mid path (every row usable)."""
    n = len(lm)
    ok = np.isfinite(lm)
    ret = np.zeros(n)
    ret[1:] = 1e4 * (lm[1:] - lm[:-1])
    ret[~np.isfinite(ret)] = 0.
    csq = np.zeros(n + 1)
    csq[1:] = np.cumsum(ret * ret)
    okc = np.zeros(n + 1, np.int64)
    okc[1:] = np.cumsum(ok)
    return dict(G0=int(G0), n=int(n), ok=ok, lm=lm, csq=csq, okc=okc)


class SM:
    """One synthetic market: a 100 ms BTC grid over [ws - 400 s, we + 100 s], tob frames every 500 ms (constant touches unless shifted),
    fills_valued legs of the top 3 (no lifecycle table: settle-based windows [settle - 2 s, settle])."""

    def __init__(self, day, mid, ws, rng, bid=0.45, ask=0.55):
        self.day, self.mid, self.ws, self.we = str(day), int(mid), int(ws), int(ws) + 300_000
        self.G0 = (self.ws - 400_000) // 100
        inc = rng.normal(0., SIG_BP * 1e-4, NG)
        inc[0] = 0.
        self.lm = np.log(70_000.) + np.cumsum(inc)
        self.bid, self.ask, self.fv, self.n = float(bid), float(ask), [], 0

    def row(self, addr, settle, s, q, p, role='maker', status='ok'):
        self.n += 1
        sid = f'{self.mid}-{self.n:05d}'
        return dict(day=self.day, mid=self.mid, sid=sid, mk_idx=0, px_rank=0, role=role, addr=addr, hash=f'0xh{self.mid}{self.n:05d}', cp_addr=OTHER,
                    kind='regular', y_side='bid' if (s > 0) == (role == 'maker') else 'ask', s=int(s), q=float(q), p=float(p), self_trade=False,
                    settle_ms=int(settle), t_ms=int(settle), t_status='lo', fee=0., fee_type='COLLATERAL', fee_dy=0., status=status, src_status=status)

    def fill(self, *a, **k):
        r = self.row(*a, **k)
        self.fv.append(r)
        return r['sid']

    def r5(self, t):
        i = t // 100 - self.G0
        return 1e4 * (self.lm[i] - self.lm[i - 50])

    def tob(self, shift_after=None):
        t = np.arange(self.ws - 310_000, self.we + 2_001, 500, dtype=np.int64)
        n = len(t)
        bid, ask = np.full(n, self.bid), np.full(n, self.ask)
        if shift_after is not None:
            m = t > shift_after
            bid[m] -= 0.03
            ask[m] -= 0.02
        return pd.DataFrame(dict(mid=np.full(n, self.mid, np.int64), upd_ms=t, recv_ns=t * 1_000_000 + 7, conn=['c0'] * n, n_bid=np.ones(n, np.int64),
                                 n_ask=np.ones(n, np.int64), best_bid=bid, best_bid_sh=np.full(n, 50.), best_ask=ask, best_ask_sh=np.full(n, 50.),
                                 status=['ok'] * n))

    def build(self, lm=None, fv=None, tob=None):
        rows = self.fv if fv is None else fv
        fvdf = pd.DataFrame(rows, columns=list(CX.FV_READ)).astype(FV_DT) if rows else pd.DataFrame({c: pd.Series(dtype=FV_DT[c]) for c in CX.FV_READ})
        legs = CX.join_windows(fvdf, None)
        BT = bt_from_lm(self.G0, self.lm if lm is None else lm)
        btc = dict(BT=BT, code=np.where(BT['ok'], 0, 2).astype(np.int8), day=self.day, load_counts={}, files=[], cum=None)
        return CX.build_market(self.day, self.mid, self.ws, self.we, TL.TOP3, btc=btc, book=False, tob=self.tob() if tob is None else tob, legs=legs)


class World:
    """n_mk synthetic markets over two days. Rank 1 holds a pre-window inventory (long on even markets, short on odd) and takes 4 sh per
    planted event, x_lo_hard in (t, t + 1000]: kind tp / sl -> a reduce with the per-second hazard HAZ[kind][a_5s category at t] (the tool's
    own state: q from legs with hi <= t, r_5s of the grid row at t); random -> hazard 0.08 whatever the state, reduce or add at random. Rank 2
    enters at 0 and is filled 30 sh at ws + 50 s (never takes); rank 3 has no leg (enters at 0)."""

    def __init__(self, kind, seed, n_mk=8):
        rng = np.random.default_rng(seed)
        self.kind, self.sms, rows = kind, [], []
        rnd = kind == 'random'
        for j in range(n_mk):
            day = DAYS[j % 2]
            ws = CX.day_ms(day) + 3_600_000 * (2 + j)
            sm = SM(day, 970_000 + 100 * KIND_ID[kind] + j, ws, rng, bid=0.30 if rnd else 0.45, ask=0.70 if rnd else 0.55)
            side = 1 if j % 2 == 0 else -1
            q0 = (120., 300., 700.)[j % 3] if rnd else 400.
            sm.fill(A1, ws - 100_000, side, q0, 0.50)
            sm.fill(A2, ws + 50_000, 1, 30., 0.50)
            path = [(ws - 100_000, side * q0)]
            for k in range(300):
                t = ws + 1_000 * k
                q = sum(dq for h, dq in path if h <= t)
                sg = 1 if q > 1e-6 else (-1 if q < -1e-6 else 0)
                if sg == 0:
                    continue
                a = sg * sm.r5(t)
                cat = 0 if a >= 1.0 else (1 if a <= -1.0 else 2)
                h = 0.08 if rnd else HAZ[kind][cat]
                if rng.random() >= h:
                    continue
                xlo = t + 1 + int(rng.integers(0, 1_000))
                hi = xlo + 2_000
                add = bool(rnd and rng.random() < 0.5)
                s = sg if add else -sg
                p = 0.50 + 0.01 * s
                sid = sm.fill(A1, hi, s, 4., p, role='taker')
                path.append((hi, s * 4.))
                rows.append(dict(day=day, mid=sm.mid, rank=1, addr=A1, sid=sid, s=int(s), dq=s * 4., p_avg=p, x_lo_hard_ms=xlo, hi_ms=hi, settle_ms=hi,
                                 status='ok', take_class='add' if add else 'reduce_partial'))
            self.sms.append(sm)
        self.takes = pd.DataFrame(rows, columns=list(TAKE_COLS))
        self.markets = [(sm.day, sm.mid, sm.build()) for sm in self.sms]


def panel(w, k=None):
    mk = w.markets if k is None else w.markets[:k]
    P = TL.assemble([TL.market_part(d, m, M) for d, m, M in mk])
    T, EV = TL.events(P, TL.prepare_takes(w.takes))
    return P, T, EV


def mech(w):
    P, T, EV = panel(w)
    R = TL.analyse(P, T, EV, full=False)
    return {(m['mechanism'], m['scope']): m for m in R['mechanisms']}


def _brief(V, names):
    return {f'{k[0]}:{k[1]}': (V[k]['verdict'], [V[k][v]['verdict'] for v in TL.VARIANTS], round(float(V[k]['D0']['RR_MH']), 3), V[k]['D0']['p_hi'],
                               f"{V[k]['D0']['n_markets_rr_gt1']}/{V[k]['D0']['n_markets_supported']}", V[k]['D0']['n_events_informative'])
            for k in V if k[0] in names}


def f1(w):
    V = mech(w)
    ok = all(V[('take_profit', sc)]['verdict'] == 'supported' and V[('stop_loss', sc)]['verdict'] != 'supported' for sc in TL.M2_SCOPES)
    return ok, _brief(V, ('take_profit', 'stop_loss'))


def f2(w):
    V = mech(w)
    return all(V[('stop_loss', sc)]['verdict'] == 'supported' for sc in TL.M2_SCOPES), _brief(V, ('take_profit', 'stop_loss'))


def pop(w):
    P = TL.assemble([TL.market_part(d, m, M) for d, m, M in w.markets])
    pc = TL.population_checks(P, 3 * len(w.markets))
    r2 = [pm for pm in P['pairs'] if pm['rank'] == 2]
    r2ok = all(pm['n'] == pm['K'] and bool(np.all(P['q0'][pm['base']:pm['base'] + 50] == 0.))
               and bool(np.all(P['q0'][pm['base'] + 50:pm['base'] + pm['n']] == 30.)) for pm in r2)
    r3ok = all(pm['n_legs'] == 0 and pm['n'] == pm['K'] for pm in P['pairs'] if pm['rank'] == 3)
    ok = bool(pc['complete']) and r2ok and r3ok and P['N'] == 3 * len(w.markets) * 300
    return ok, dict(pc, rank2_enters_at_zero=r2ok, rank3_no_fill_full_span=r3ok)


# ------------------------------------------------------------------ fixture 4: causality world ------------------------------------------------------------------
CAUSAL = TL.MS_FIELDS + ('q0', 'q1', 'fl0', 'fl1')
CODES = ('qb', 'sg', 'sc', 'tl', 'f5b', 'bst', 'cat_1s', 'cat_5s', 'cat_30s', 'a5', 'dose', 'edge', 'edgeb')


def f4_world(seed=41):
    rng = np.random.default_rng(seed)
    day = DAYS[0]
    ws = CX.day_ms(day) + 3_600_000 * 14
    sm = SM(day, 979_001, ws, rng)
    ts = ws + 150_000
    sm.fill(A1, ws - 100_000, 1, 400., 0.50)
    for dt_, s in ((-3_000, 1), (400, -1), (2_500, 1), (7_000, -1), (40_000, 1)):
        sm.fill(A1, ts + dt_, s, 6., 0.50)
    sm.fill(A1, ts + 1_500, -1, 4., 0.49, role='taker')
    sm.fill(A2, ts - 800, 1, 9., 0.50)
    sm.fill(A2, ts + 300, -1, 9., 0.50)                                  # window [ts - 1.7 s, ts + 0.3 s]: pending at ts
    sm.fill(A3, ts + 10_000, 1, 5., 0.50, status='record_leg_excluded')   # excluded leg: ledger-conditional from ts + 8 s
    return sm, ts


def f4_perturbed(sm, ts, seed=43):
    """BTC rows after ts replaced, every leg arriving after ts moved to another arrival after ts, the book after ts shifted, one new fill."""
    rng = np.random.default_rng(seed)
    lm = sm.lm.copy()
    i0 = ts // 100 - sm.G0 + 1
    lm[i0:] = lm[i0:] + np.cumsum(rng.normal(0., 0.6e-4, len(lm) - i0))
    fv = []
    for r in sm.fv:
        r = dict(r)
        if r['settle_ms'] > ts:
            r['settle_ms'] = r['t_ms'] = ts + 1 + ((r['settle_ms'] - ts) * 7) % 9_000
        fv.append(r)
    fv.append(sm.row(A1, ts + 600, 1, 11., 0.50))
    return sm.build(lm=lm, fv=fv, tob=sm.tob(shift_after=ts))


def f4(sm, ts, M0, M1):
    P0 = TL.assemble([TL.market_part(sm.day, sm.mid, M0)])
    P1 = TL.assemble([TL.market_part(sm.day, sm.mid, M1)])
    if P0['N'] != P1['N'] or not np.array_equal(P0['t'], P1['t']):
        return False, dict(reason='instant sets differ', n0=P0['N'], n1=P1['N'])
    bef = P0['t'] <= ts
    aft = ~bef
    bad = [f for f in CAUSAL if not same(P0[f][bef], P1[f][bef])]
    for v in ('D0', 'D1000'):
        c0, c1 = TL.variant_codes(P0, v), TL.variant_codes(P1, v)
        bad += [f'{v}:{f}' for f in CODES if not same(c0[f][bef], c1[f][bef])]
    moved = dict(btc=not same(P0['r_5s0'][aft], P1['r_5s0'][aft]), inventory=not same(P0['q0'][aft], P1['q0'][aft]),
                 book=not same(P0['bid0'][aft], P1['bid0'][aft]))
    return (not bad) and all(moved.values()), dict(bad=bad, after_changed=moved, n_before=int(bef.sum()))


# ------------------------------------------------------------------ tests ------------------------------------------------------------------
def test_bins():
    q = np.array([0., 1e-7, -1e-7, 0.5, 50., 50.0001, 200., 200.5, 500., 500.1, -600.])
    check('bins: |q| 0 | (0,50] | (50,200] | (200,500] | >500 at the edges', TL.qbin(q).tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4], TL.qbin(q))
    check('bins: sign long / short / flat (|q| <= 1e-6 flat)', TL.qsign(q).tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, -1], TL.qsign(q))
    tl = np.array([301., 300., 240.5, 240., 120., 60.5, 60., 30., 10.5, 10., 0.])
    check('bins: time left >300 | (240,300] | (120,240] | (60,120] | (30,60] | (10,30] | [0,10]', TL.tl_codes(tl).tolist() == [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6],
          TL.tl_codes(tl))
    a = np.array([-6., -5., -4.9, -2., -1.9, -1., -0.99, 0., 0.99, 1., 1.99, 2., 4.99, 5., NAN])
    check('bins: a_5s dose <=-5 | (-5,-2] | (-2,-1] | (-1,1) | [1,2) | [2,5) | >=5 | missing', TL.dose_codes(a).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7],
          TL.dose_codes(a))
    check('bins: own fills 0 | 1-2 | >=3', TL.f5_codes(np.array([0, 1, 2, 3, 7])).tolist() == [0, 1, 1, 2, 2])
    sg = np.array([1, 1, 1, -1, -1, 0, 1])
    r5 = np.array([1.0, -1.0, 0.5, 1.0, -1.2, 3.0, NAN])
    check('bins: a_5s = sign(q) r_5s with (>= 1 bp) / against (<= -1 bp) / flat / missing via chain_context.align_cat',
          TL.align_codes(sg, r5, TL.THR['r_5s']).tolist() == [0, 1, 2, 1, 0, 2, 3], TL.align_codes(sg, r5, TL.THR['r_5s']))
    check('bins: MOVE_THR is chain_context 0.5 / 1 / 2 bp', TL.THR == {'r_1s': 0.5, 'r_5s': 1.0, 'r_30s': 2.0}, TL.THR)
    sg = np.array([1, 1, 1, 1, 1, -1, 0, 1, 1])
    pf = np.array([.475, .46, .452, .44, .42, .58, .5, .44, .44])
    kok = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1], bool)
    bok = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0], bool)
    e, eb = TL.edge_values(sg, np.full(9, .45), np.full(9, .55), pf, kok, bok)
    check('bins: unwind edge long 100 (bid - p_fair) / short 100 (p_fair - ask) cents and its bins (missing book / BTC, no inventory)',
          eb.tolist() == [0, 1, 2, 3, 4, 4, 6, 5, 5] and abs(e[0] + 2.5) < 1e-9 and abs(e[5] - 3.0) < 1e-9 and np.isnan(e[6:]).all(), (e, eb))


def test_mh():
    rr = TL.mh_rr([[2., 3.]], [[1., 2.]], [10., 30.], [20., 10.])[0][0]
    check('MH toy: (2*20/30 + 3*10/40) / (1*10/30 + 2*30/40) = 25/22 (crude 1.25 differs)', abs(rr - 25. / 22.) < 1e-12, rr)
    code = np.r_[np.full(10, 1), np.full(20, 0), np.full(30, 3), np.full(10, 2), np.full(5, -1)].astype(np.int64)
    ev = np.array([0, 1, 10, 30, 31, 32, 60, 61, 70])
    pt, obs = TL.stratum_counts(code, 2, 2, ev)
    rr2 = TL.mh_rr(obs[:, 1][None, :], obs[:, 0][None, :], pt[:, 1], pt[:, 0])[0][0]
    check('MH toy through the per-instant path (stratum_counts: person-time 10/20 + 30/10, events 2/1 + 3/2, one excluded) = 25/22',
          pt.tolist() == [[20., 10.], [10., 30.]] and obs.tolist() == [[1., 2.], [2., 3.]] and abs(rr2 - 25. / 22.) < 1e-12, (pt, obs, rr2))
    r3 = TL.mh_rr([[2., 0.], [0., 0.], [1., 1.]], [[0., 0.], [0., 0.], [1., 0.]], [10., 0.], [20., 5.])[0]
    check('MH: exposed events without reference events -> inf; no events -> NaN; a stratum without exposed person-time is ignored',
          np.isinf(r3[0]) and np.isnan(r3[1]) and abs(r3[2] - 2.0) < 1e-12, r3)
    c = TL.tail_counts(1.5, np.array([1., 2., NAN, 1.5]))
    check('null: p counts 1 + #null >= obs (ties count) with a NaN replicate counted on both sides', c == (4, 4), c)
    check('null: an undefined observation has no p', TL.tail_counts(NAN, np.array([1.])) == (None, None))
    check('null: an infinite observation counts only infinite or NaN replicates above it', TL.tail_counts(float('inf'), np.array([1., float('inf'), NAN])) == (3, 4))
    cases = [((2.0, 1, 40, 6, 5), 'supported'), ((2.0, 25, 40, 6, 4), 'supported'), ((2.0, 26, 40, 6, 6), 'not_supported'),
             ((2.0, 1, 40, 6, 3), 'inconsistent'), ((2.0, 1, 29, 6, 6), 'insufficient_support'), ((2.0, 1, 40, 4, 4), 'insufficient_support'),
             ((0.8, 1, 40, 6, 6), 'not_supported'), ((NAN, None, 0, 6, 0), 'insufficient_support'), ((float('inf'), 1, 40, 5, 5), 'supported')]
    got = [TL.verdict(*a) for a, _ in cases]
    check('verdict rule: supported / inconsistent / insufficient (< 5 markets or < 30 events) / not_supported at the p = 25/501 and 2/3 edges',
          got == [v for _, v in cases], got)
    check('end of market: a supported RR_end with share flattened < 0.5 reads not_supported, an undefined share insufficient_support',
          (TL.end_combine('supported', 0.49), TL.end_combine('supported', 0.5), TL.end_combine('not_supported', 0.9), TL.end_combine('supported', NAN))
          == ('not_supported', 'supported', 'not_supported', 'insufficient_support'))
    check('declared constants: 500 shifts, seed 20261123, p_hi <= 0.05 <=> c_hi <= 25', (TL.N_SHIFT, TL.SEED, TL.C_HI_MAX) == (500, 20261123, 25))


def test_states(sm, ts, M0):
    P = TL.assemble([TL.market_part(sm.day, sm.mid, M0)])
    pm = {p['rank']: p for p in P['pairs']}

    def at(rank, t):
        return pm[rank]['base'] + (t - sm.ws) // 1_000
    i = at(1, ts - 3_000)
    check('state: q(t) counts a leg with hi == t (hi <= t) and not one second before', P['q0'][i] == 406. and P['q0'][i - 1] == 400.,
          (P['q0'][i - 1], P['q0'][i]))
    check('state: D1000 reads the state one second earlier', P['q1'][i + 1] == 406. and P['q1'][i] == 400.)
    check('state: q(t) after a maker sell and a take (+6 - 6 - 4 = 396)', P['q0'][at(1, ts + 2_000)] == 396., P['q0'][at(1, ts + 2_000)])
    fl = [int(P['fl0'][at(1, ts + d)]) for d in (-4_000, -3_000, 2_000, 3_000)]
    check('state: own maker fills with hi in (t - 5 s, t] (a take leg never counts; hi = t - 5 s excluded)', fl == [0, 1, 1, 2], fl)
    j = at(2, ts)
    check('state: inv_timing_sensitive when a pending leg (window holds t) moves |q| to another bin / sign; q pend = possible',
          bool(P['ts'][j]) and P['q0'][j] == 9. and P['qp'][j] == 0. and not bool(P['ts'][at(2, ts - 5_000)]), (P['ts'][j], P['q0'][j], P['qp'][j]))
    check('state: inv_ledger_conditional from the excluded leg window start (ts + 8 s), never before; the pair stays at q = 0',
          (not bool(P['lg'][at(3, ts + 7_000)])) and bool(P['lg'][at(3, ts + 8_000)]) and not bool(np.any(P['q0'][pm[3]['base']:pm[3]['base'] + 300]))
          and pm[3]['n_legs'] == 0)
    C0 = TL.variant_codes(P, 'D0')
    pf = float(CX.btc_ret(M0, np.array([ts], np.int64))['p_fair'][0])
    check('state: edge (cents) of a long = 100 (best bid - p_fair) from book_tob and btc_ret; q = 0 -> no_inventory',
          abs(C0['edge'][at(1, ts)] - 100. * (0.45 - pf)) < 1e-9 and C0['edgeb'][at(3, ts)] == 6, (C0['edge'][at(1, ts)], pf))
    check('state: r_5s at t is btc_ret on the grid row floor(t / 100) (the planted r5)', abs(P['r_5s0'][at(1, ts)] - sm.r5(ts)) < 1e-9)
    b = pm[1]['base']
    check('state: time left 300 s at ws -> (240,300]; 1 s at we - 1 s -> [0,10]; D1000 at ws -> 301 s -> >300',
          (P['tl0'][b], P['tl0'][b + 299], P['tl1'][b]) == (300., 1., 301.) and TL.variant_codes(P, 'D1000')['tl'][b] == 0)


def test_assign(w):
    P = TL.assemble([TL.market_part(d, m, M) for d, m, M in w.markets[:1]])
    d, m, _ = w.markets[0]
    pm = P['pairs'][0]
    ws, we = pm['ws'], pm['we']
    xs = [ws, ws + 1, ws + 1_000, ws + 1_001, we, we + 1, -1]
    T = pd.DataFrame(dict(day=[d] * 7, mid=[m] * 7, rank=[1] * 7, addr=[A1] * 7, s=[-1] * 7, dq=[-4.] * 7, p_avg=[.49] * 7, x_lo_hard_ms=xs,
                          take_class=['reduce_partial'] * 6 + ['add']))
    T2, EV = TL.events(P, TL.prepare_takes(T))
    st = list(T2['pop_status'])
    kk = [int(g - pm['base']) if g >= 0 else -1 for g in T2['gi']]
    check('events: x_lo_hard = ws -> before_population; ws + 1 and ws + 1000 -> k 0; ws + 1001 -> 1; we -> 299; we + 1 -> after; -1 -> unbounded',
          st == ['before_population', 'assigned', 'assigned', 'assigned', 'assigned', 'after_population', 'unbounded_window'] and kk == [-1, 0, 0, 1, 299, -1, -1],
          (st, kk))
    check('events: an event-second counts once, its takes beside it (two reduce takes at k 0 -> 1 event-second, 2 takes)',
          EV['E_reduce'][pm['base']] == 1 and EV['N_reduce'][pm['base']] == 2 and EV['N_any'].sum() == 4 and EV['E_any'].sum() == 3)


def test_shift(w):
    parts = [TL.market_part(d, m, M) for d, m, M in w.markets]
    P = TL.assemble(parts)
    T, EV = TL.events(P, TL.prepare_takes(w.takes))
    npair = len(P['pairs'])
    ok_cnt = ok_in = ok_bij = ok_null = True
    tot = 0
    for t_ in TL.TYPES_ANY:
        idx = np.flatnonzero(EV['E_' + t_])
        tot += len(idx)
        SH = TL.shifted(P, idx)
        pr = P['pair'][SH]
        base = np.bincount(P['pair'][idx], minlength=npair)
        ok_in &= bool(np.all(pr == P['pair'][idx][None, :]))
        ok_cnt &= all(np.array_equal(np.bincount(pr[r], minlength=npair), base) for r in range(TL.N_SHIFT))
        if SH.shape[1] > 1:
            srt = np.sort(SH, axis=1)
            ok_bij &= bool(np.all(srt[:, 1:] != srt[:, :-1]))
        nul = TL.null_counts(np.zeros(P['N'], np.int64), 1, 1, SH)
        ok_null &= bool(np.all(nul.reshape(TL.N_SHIFT) == len(idx)))
    check("F5: every shift replicate keeps each pair's event-second count per type", ok_cnt and tot > 0, tot)
    check('F5: a shifted event-second never leaves its own pair', ok_in)
    check('F5: a shift is a bijection on the pair (no two event-seconds collide)', ok_bij)
    check('F5: the null bincount keeps the event count of every replicate', ok_null)
    check('F5: offsets lie in [0, instant count) per pair', bool(np.all((P['U'] >= 0) & (P['U'] < P['pair_n'][:, None]))))
    P2 = TL.assemble(parts)
    n0 = len(parts[0]['pairs'])
    P3 = TL.assemble(parts[1:])
    check("F5: offsets are deterministic (seed 20261123) and a pair's offsets do not depend on the other pairs",
          np.array_equal(P['U'], P2['U']) and np.array_equal(P3['U'], P['U'][n0:]) and not np.array_equal(P['U'][0], P['U'][1]))


def mutant_suite(w1, w2, f4a, w_pop):
    out = {}
    for nm, fn in (('F1', lambda: f1(w1)), ('F2', lambda: f2(w2)), ('F4', lambda: f4(*f4a)), ('POP', lambda: pop(w_pop))):
        try:
            ok, info = fn()
            out[nm] = (bool(ok), False, info)
        except Exception as e:  # noqa: BLE001
            out[nm] = (False, True, f'{type(e).__name__}: {e}')
    return out


def test_mutants(w1, w2, f4a):
    """The unmutated baseline of F1 / F2 / F4 / POP is the main section's checks (each must pass there)."""
    for m in TL.MUTANTS:
        os.environ['MT_TL_MUT'] = m
        try:
            res = mutant_suite(w1, w2, f4a, w1)
        finally:
            os.environ.pop('MT_TL_MUT', None)
        failed = [k for k, (ok, _, _) in res.items() if not ok]
        caught = [k for k in EXPECT_CATCH[m] if k in failed and not res[k][1]]
        check(f'mutant {m}: caught by {"/".join(EXPECT_CATCH[m])} evaluating False without raising (failed: {failed})', bool(caught),
              {k: res[k][2] for k in failed})
    check('mutants: MT_TL_MUT unset after the suite', 'MT_TL_MUT' not in os.environ)


def test_refusals(tmp, w):
    d = tmp / 'exists'
    d.mkdir()
    expect('refuse: write-once on an existing output directory', lambda: TL.write_out(d, {}, {}, lambda o: {}), 'write-once')

    def boom(o):
        raise RuntimeError('boom in the receipt writer')
    try:
        TL.write_out(tmp / 'fails', {'a.csv': pd.DataFrame({'x': [1]})}, {}, boom)
    except RuntimeError:
        pass
    check('write-once: a failed write leaves no output and no partial directory', not (tmp / 'fails').exists() and
          not any(p.name.startswith('.fails.partial') for p in tmp.iterdir()))
    for m, needle in (('flip_align', 'MT_TL_MUT'), ('bogus', 'not a declared mutant')):
        os.environ['MT_TL_MUT'] = m
        try:
            expect(f'refuse: production with MT_TL_MUT={m} before any input is read', lambda: TL.execute('pilot', (TL.PILOT_DAY,), tmp / 'never'), needle)
        finally:
            os.environ.pop('MT_TL_MUT', None)
    check('refuse: nothing written by a refused production call', not (tmp / 'never').exists())
    expect('refuse: a pilot day outside the measurement days (before any path is built)', lambda: TL.execute('pilot', (C.ID_DAY,), tmp / 'never'),
           'measurement days')
    g = X._git([f for f in TL.TOOL_FILES if Path(f).exists()])
    needle = 'freeze' if not g.get('unmodified') else 'no recorded pilot projection'
    expect(f'refuse: the full run before any input ({needle})', lambda: TL.execute('run', C.MEASURE_DAYS, tmp / 'run_out', pilot_receipt=tmp / 'absent.json'),
           needle)
    expect('refuse: the full run without a recorded pilot projection', lambda: TL.preflight_projection(tmp / 'absent.json', {}), 'no recorded pilot projection')
    p61 = tmp / 'p61.json'
    p61.write_text(json.dumps(dict(kind='pilot', projection=dict(full_run_cpu_s=61.0))))
    expect('refuse: a recorded pilot projection > 60 CPU s', lambda: TL.preflight_projection(p61, {}), '> 60 s (refused before launch)')
    p30 = tmp / 'p30.json'
    p30.write_text(json.dumps(dict(kind='pilot', projection=dict(full_run_cpu_s=30.0))))
    h = {}
    r = TL.preflight_projection(p30, h)
    check('preflight: a 30 s projection passes and the pilot receipt is hashed as an input', r['full_run_cpu_s'] == 30.0 and len(h) == 1)
    bad = w.takes.copy()
    bad.loc[0, 'take_class'] = 'bogus_class'
    expect('refuse: an unknown take class', lambda: TL.prepare_takes(bad), 'unknown take class')
    bad2 = w.takes.copy()
    bad2.loc[0, 'addr'] = A2
    expect("refuse: a take whose address is not its rank's", lambda: TL.prepare_takes(bad2), "differs from its rank's top-3 address")
    cls = [c for c, v in TL.EXPECT_CLS.items() for _ in range(v)]
    S = pd.DataFrame([dict(scope='pooled', day='all', **{f'cls_{c}': TL.EXPECT_CLS[c] for c in TL.CT.CLS})])
    r = TL.check_reproduction(pd.DataFrame({TL.CLASS_COL: cls}), S)
    check('reproduction: the dispatch counts (2065 / 1185 / 838 / 38 / 8 / 3 / 10) reproduce', r['reproduces'] and len(cls) == TL.N_TAKES_ALL)
    cls2 = list(cls)
    cls2[cls2.index('from_flat')] = 'add'
    expect('refuse: a class column that does not reproduce the summary counts', lambda: TL.check_reproduction(pd.DataFrame({TL.CLASS_COL: cls2}), S),
           'does not reproduce the summary counts')
    S2 = S.copy()
    S2['cls_add'] = 1184
    expect('refuse: summary.csv pooled / all disagrees with the class column', lambda: TL.check_reproduction(pd.DataFrame({TL.CLASS_COL: cls}), S2),
           'does not reproduce summary.csv')


def test_e2e(tmp, w, V3):
    parts = [TL.market_part(d, m, M) for d, m, M in w.markets]
    out = tmp / 'e2e_out'
    stages = {}
    rec, claims, R = TL.run_core('selftest', DAYS, parts, TL.prepare_takes(w.takes), out, {}, Counter(), stages, lambda nm: None, {}, 'asof', time.time(),
                                 3 * len(parts))
    names = sorted(p.name for p in out.iterdir())
    check('e2e: run_core writes exactly the nine contract outputs', names == sorted(TL.OUT_FILES), names)
    J = json.loads((out / 'claims.json').read_text())
    check('e2e: claims.json carries 7 mechanisms x 2 scopes, each with a final verdict and D0 / D1000 / pend verdicts',
          len(J['mechanisms']) == 14 and all(m['verdict'] in VERDICTS and all(m[v]['verdict'] in VERDICTS for v in TL.VARIANTS) for m in J['mechanisms']))
    check('e2e: the written verdicts equal the verdict path', all(V3[(m['mechanism'], m['scope'])]['verdict'] == m['verdict'] for m in J['mechanisms']))
    Rj = json.loads((out / 'receipt.json').read_text())
    check('e2e: receipt.json hashes every other output (sha256 = the file) and the sources', all(Rj['outputs'][f]['sha256'] == X._sha(out / f)
          for f in TL.OUT_FILES if f != 'receipt.json') and len(Rj['sources']) >= 9 and Rj['mutant'] == '')
    sizes = {f: len(pd.read_csv(out / f)) for f in TL.CSV_FILES}
    check('e2e: every csv parses with rows', all(v > 0 for v in sizes.values()), sizes)
    rt = pd.read_csv(out / 'ratios.csv')
    check('e2e: ratios.csv = 10 ratios x 2 types x 2 scopes x 3 variants; verdicts only on the 7 mechanism ratios',
          len(rt) == 120 and int((rt['mechanism'].fillna('') != '').sum()) == 7 * 2 * 3, len(rt))
    hz = pd.read_csv(out / 'hazard_cells.csv')
    m1 = hz[(hz['scope'] == 'pooled') & (hz['factor'] == 'qbin')]
    check('e2e: M1 marginals add up to the maker-seconds and the event-seconds of the panel',
          int(m1['maker_seconds'].sum()) == claims['maker_seconds']['pooled'] and int(m1['event_seconds_any'].sum()) == claims['events']['any']['event_seconds'])
    pt = pd.read_csv(out / 'patterns.csv')
    check('e2e: patterns.csv holds P1-P8 and the status rows for every scope', len(pt) == (8 + 3) * 4, len(pt))
    expect('e2e: a second write into the written directory is refused', lambda: TL.write_out(out, {}, {}, lambda o: {}), 'write-once')


def main():
    t0 = time.process_time()
    check('selftest environment: MT_RUN=v2 and no MT_TL_MUT', C.RUN == 'v2' and 'MT_TL_MUT' not in os.environ, C.RUN)
    tmp = Path(tempfile.mkdtemp(prefix='take_logic_selftest_'))
    try:
        section('bins', test_bins)
        section('mh', test_mh)
        w1 = World('tp', 101)
        w2 = World('sl', 202)
        w3 = World('random', 303)
        smf, ts = f4_world()
        M0, M1 = smf.build(), f4_perturbed(smf, ts)
        check('fixtures: planted takes (tp, sl, random)', min(len(w1.takes), len(w2.takes), len(w3.takes)) >= 150,
              (len(w1.takes), len(w2.takes), len(w3.takes)))
        r = section('F1', f1, w1)
        if r:
            check('F1: a planted take-profit maker reads take_profit supported and stop_loss not (rank 1 and pooled)', r[0], r[1])
        r = section('F2', f2, w2)
        if r:
            check('F2: a planted stop-loss maker reads stop_loss supported (rank 1 and pooled)', r[0], r[1])
        V3 = section('F3', mech, w3)
        if V3:
            sup = [f'{k[0]}:{k[1]}' for k, m in V3.items() if m['verdict'] == 'supported']
            check('F3: a random-time maker reads no mechanism supported on the fixture seed', not sup and len(V3) == 14, _brief(V3, [k[0] for k in V3]))
            live = all(V3[(mc, sc)]['D0']['predicates']['supported_markets_ge_5'] and V3[(mc, sc)]['D0']['predicates']['events_ge_30']
                       for mc in ('stop_loss', 'take_profit', 'directional') for sc in TL.M2_SCOPES)
            check('F3 is not vacuous: stop_loss / take_profit / directional have >= 5 supported markets and >= 30 events at D0', live)
        r = section('POP', pop, w1)
        if r:
            check('population: every pair x market instant (ws .. we - 1 s), the no-fill pair and the late-filled pair enter at q = 0', r[0], r[1])
        r = section('F4', f4, smf, ts, M0, M1)
        if r:
            check('F4 causality: later BTC / later fill arrivals / later book leave every D0 and D1000 state at t <= t* unchanged; after t* they change',
                  r[0], r[1])
        section('states', test_states, smf, ts, M0)
        section('assign', test_assign, w1)
        section('F5', test_shift, w3)
        section('mutants', test_mutants, w1, w2, (smf, ts, M0, M1))
        section('refusals', test_refusals, tmp, w1)
        if V3:
            section('e2e', test_e2e, tmp, w3, V3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop('MT_TL_MUT', None)
    fails = [r for r in RES if not r[1]]
    for name, ok, info in RES:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  | {info}'))
    print(json.dumps(dict(checks=len(RES), failed=len(fails), cpu_s=round(time.process_time() - t0, 3), process_cpu_s=round(time.process_time(), 3),
                          vmhwm_mib=round(TL.CT.hwm(), 1))))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
