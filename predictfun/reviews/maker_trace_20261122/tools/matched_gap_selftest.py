"""Selftest of tools/matched_gap.py (MATCHED-MARKET-TURNOVER-GAP-v1, contract matched_gap_contract.md; session aea5bfb9, 2026-11-23). SYNTHETIC
data only, in a temporary directory (MG_SELFTEST_TMP or the system default; the tool's git queries read the repository, nothing else outside it is
touched). The econ inputs are built from hand-written legs by pf_mt_econ's own fifo (mq / rq) and market_table (maker_market), so the tool is
reconciled against econ's real code; our records carry the engine arithmetic (pf_branch_engine.Engine.value, its source lines checked); every
expected number below is computed by hand (comments), never by the tool.
World (YES space, USD; ms after each market's ws; END mark = our mark / 100; theirs END = the same mark):
 M1 101 (D1; our start ws + 1000, END ws + 298,000, mark 60 c, opening mark .35 = the tob frame AT start; a frame 1 ms later is ignored):
   ours  buy 10 @40 c (+1,000 ms), sell 4 @45 c (+2,000), sell 3 @50 c AT END (+297,000 = end_rel) -> closes 4 (+.20) + 3 (+.30) = .50 on 7 sh;
         resid long 3 @.40 -> +.60; total 1.10 = recorded value 110 c (engine: cash -70, position 3).
   rank1 before start: maker buy 6 @.30 (-5,000), taker sell 2 @.32 fee .01 (-2,000), maker buy 1 @.31 at start - 1 ms -> carried lots 4 (acq
         -5,000) + 1 (acq +999), C0 5, re-based at .35. Inside: taker sell 3 @.38 AT start (fee .03), maker sell 5 @.40, a self-trade 2 @.45
         (taker fee .02), taker buy 5 @.42 (fee .05), an excluded maker buy 10 @.30 (src_status), maker sell 1 @.58 AT END; after END maker sell
         2 @.59 (outside). CARRY: closes carried 3 m->t +.09 - .03, carried 1 m->m +.05, carried 1 m->m +.05, within short 3 m->t -.06 - .03,
         within long 1 t->m +.16 - .01 -> close net .22 on 9 sh (carried 5 sh, net .16); resid long 1 @.42 +.18, unclosed fee .01, self fee
         .02 -> total .37 = direct -.78 - .10 + 5 x (.60 - .35); max |position| 5; actions 4: nonflat 4, opposing 4, reduce 3, overshoot 1.
         ZERO: closes t->t 3 -.12 - .06, m->t 2 -.04 - .02 -> -.24 on 5 sh; resid short 3 @.40 -.60 and 1 @.58 -.02 -> total -.88 = -.78 - .10.
   rank4 maker buy 2 @.50, maker sell 2 @.55 -> +.10 (in top10 only).
   rank5 (r1 fix, the first run's defect): maker sell 3 @.45 BEFORE start and no fill inside -> carry_only: its short 3 enters top10's carry
         view unclosed at the opening basis (carry 3 x -(.60 - .35) = -.75); top10 carry total .37 + .10 - .75 = -.28 vs zero -.78: the views
         differ by exactly carried_pos_recorded (5 - 3 = 2) x (.60 - .35) = .50 (cross-view); rank 5 alone stays no_activity.
 M2 102: ours buy 2 @50 / sell 2 @49 (-.02); rank1 +.04 (2 sh); rank2 +.09 (3 sh). M3 103: ours -.02; rank1 maker buy 10 @.50, taker sell 10
   @.45 -> -.50. M4 201 (D2, mark 40): ours buy 2 @50 / sell 1 @49 (-.01) + resid 1 @.50 -> -.10 (total -.11); rank1 -.02. M5 202: ours +.02;
   rank1 -.01. M6 203: our record missing_mark -> no_interval. Markets 104 (a non-BOOK rank1 fill, filtered by mid) and 150 (its own row group,
   pruned by the mid statistics) are never accounted.
 Readings (carry): rank1 M1 exposure_gap (dT .28 < dE .42), M2 our_turnover_loses, M3 both_adverse (dT .48 > dE 0), M4 exposure_gap (both
   negative, dE -.10 > dT .01), M5 mixed_unsupported -> 5 markets, supported; pooled o .47 / 13 > 0 > t -.27 / 24: mixed_unsupported. rank2:
   1 market (< 5): every reading mixed_unsupported (raw our_turnover_loses). rank3: none. top3 / top10: the rank1 pattern.
Refusals (each asserts its reason; no partial directory may remain; the output is never created): the recorded value (one cent; one ulp), the
accepted arms.csv value, maker_market completeness, econ mq / rq, a BOOK market's row under another day, path_info vs ident, our fills out of
order, a manifest day outside the measurement days, the population size, write-once, the freeze guard (when the tool is not committed); the
cross-view falsifier (identical views with a carried position and END != opening mark) and the planted first-run defect (_MUT_EXCLUDE_CARRY_ONLY:
carry_only lots left out) refused by the party identity.
Usage (run_light): matched_gap_selftest.py"""
import gzip
import hashlib
import inspect
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matched_gap as MG  # noqa: E402

X, C, E = MG.X, MG.C, MG.E
D1, D2 = '20261024', '20261025'
B0 = {D1: 1792803600000, D2: 1792890000000}                            # 01:00Z, ms
NAN = float('nan')
W_, K_ = MG.WITHIN, MG.CARRIED
R = X.TOP3 + tuple('0x' + format(i, '040x') for i in range(4, 11))      # ex-ante ranks 1-10 (4-10 synthetic)
FA = tuple('0xf' + format(i, '039x') for i in range(1, 10))            # field addresses (never read by the tool)
r1, r2, r3, r4, r5 = R[:5]
f1, f2, f3, f4, f5, f6, f7 = FA[:7]
WS = {101: B0[D1], 102: B0[D1] + 300000, 103: B0[D1] + 600000, 104: B0[D1] + 900000, 150: B0[D1] + 1200000,
      201: B0[D2], 202: B0[D2] + 300000, 203: B0[D2] + 600000}
DAY = {m: (D1 if m < 200 else D2) for m in WS}
BOOK = {D1: [101, 102, 103], D2: [201, 202, 203]}
PARTS = dict(pilot={D1: [101, 102], D2: [201]}, rest={D1: [103], D2: [202, 203]})
ENDM = {101: .60, 102: .50, 103: .50, 104: .50, 150: .50, 201: .40, 202: .50, 203: .50}
OURS = {101: dict(wait=1000, mark=60., fills=[[1000., 0, 0, 10., 40., 0], [2000., 1, 1, 4., 45., 1], [297000., 2, 1, 3., 50., 2]]),
        102: dict(wait=0, mark=50., fills=[[1000., 0, 0, 2., 50., 0], [2000., 1, 1, 2., 49., 1]]),
        103: dict(wait=0, mark=50., fills=[[1000., 0, 0, 2., 50., 0], [2000., 1, 1, 2., 49., 1]]),
        201: dict(wait=0, mark=40., fills=[[1000., 0, 0, 2., 50., 0], [2000., 1, 1, 1., 49., 1]]),
        202: dict(wait=0, mark=50., fills=[[1000., 0, 0, 1., 50., 0], [2000., 1, 1, 1., 52., 1]]),
        203: dict(wait=0, mark=None, status='missing_mark', fills=[[1000., 0, 0, 1., 50., 0]])}


def leg(mid, sid, dt, maker, taker, s_mk, q, p, mfee=0., tfee=0., status='ok'):
    """One maker leg (both rows): s_mk = +1 the maker buys YES (its taker sells), -1 the maker sells; fees in USD (COLLATERAL)."""
    return dict(mid=mid, sid=sid, t=WS[mid] + dt, maker=maker, taker=taker, s_mk=s_mk, q=q, p=p, mfee=mfee, tfee=tfee, status=status)


LEGS = [
    leg(101, 'm1-a', -5000, r1, f1, +1, 6., .30), leg(101, 'm1-b', -2000, f2, r1, +1, 2., .32, tfee=.01), leg(101, 'm1-c', 999, r1, f3, +1, 1., .31),
    leg(101, 'm1-d', 1000, f4, r1, +1, 3., .38, tfee=.03), leg(101, 'm1-e', 10000, r1, f5, -1, 5., .40),
    leg(101, 'm1-f', 20000, r1, r1, +1, 2., .45, tfee=.02), leg(101, 'm1-g', 30000, f6, r1, -1, 5., .42, tfee=.05),
    leg(101, 'm1-x', 40000, r1, f7, +1, 10., .30, status='src_status'), leg(101, 'm1-h', 298000, r1, f1, -1, 1., .58),
    leg(101, 'm1-i', 298001, r1, f2, -1, 2., .59),
    leg(101, 'm1-j', 50000, r4, f3, +1, 2., .50), leg(101, 'm1-k', 60000, r4, f4, -1, 2., .55),
    leg(101, 'm1-z', -3000, r5, f6, -1, 3., .45),                                                              # rank 5: carry_only
    leg(102, 'm2-a', 5000, r1, f1, +1, 2., .50), leg(102, 'm2-b', 6000, r1, f2, -1, 2., .52),
    leg(102, 'm2-c', 7000, r2, f3, +1, 3., .40), leg(102, 'm2-d', 8000, r2, f4, -1, 3., .43),
    leg(103, 'm3-a', 5000, r1, f1, +1, 10., .50), leg(103, 'm3-b', 6000, f2, r1, +1, 10., .45),
    leg(104, 'n4-a', 5000, r1, f1, +1, 1., .50), leg(150, 'n5-a', 5000, r2, f1, +1, 1., .50),
    leg(201, 'm4-a', 5000, r1, f1, +1, 2., .50), leg(201, 'm4-b', 6000, r1, f2, -1, 2., .49),
    leg(202, 'm5-a', 5000, r1, f1, +1, 1., .50), leg(202, 'm5-b', 6000, r1, f2, -1, 1., .49),
    leg(203, 'm6-a', -1000, r1, f1, +1, 1., .50), leg(203, 'm6-b', 5000, r1, f2, -1, 1., .55),
]
FRAMES = ([(101, WS[101] + 500, .33, .35), (101, WS[101] + 1000, .34, .36), (101, WS[101] + 1001, .50, .52), (101, WS[101] + 297500, .59, .61)] +
          [x for m in (102, 103, 104, 201, 202, 203) for x in ((m, WS[m] - 100, .49, .51), (m, WS[m] + 297900, ENDM[m] - .01, ENDM[m] + .01))])
RES = dict(n=0, fail=[])


def check(name, cond, detail=''):
    RES['n'] += 1
    if not cond:
        RES['fail'].append(f'{name} {detail}'.strip())
        print(f'FAIL {name} {detail}', flush=True)


def near(a, b, tol=1e-9):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
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


def expect(name, fn, substr, out=None):
    """fn must refuse with a ValueError naming `substr`; with out: no output and no partial directory remain."""
    try:
        fn()
    except ValueError as e:
        check(name, substr in str(e), f'(message: {e})')
    except Exception as e:  # noqa: BLE001
        check(name, False, f'(raised {type(e).__name__}: {e})')
    else:
        check(name, False, '(did not refuse)')
    if out is not None:
        check(f'{name}: no output, no partial directory', not Path(out).exists() and not list(Path(out).parent.glob(f'.{Path(out).name}.partial_*')))


def vals(name, row, exp, tol=1e-9):
    for k, v in exp.items():
        got = row[k]
        ok = (str(got) == str(v)) if isinstance(v, (str, bool)) else near(got, v, tol)
        check(f'{name}: {k}', ok, f'(got {got!r}, want {v!r})')


def pick(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k].astype(object).map(str) == str(v)).to_numpy(bool)
    sub = df[m]
    if len(sub) != 1:
        raise ValueError(f'selftest: {len(sub)} rows match {kw}')
    return sub.iloc[0]


def engine_ref(fills, mark):
    """pf_branch_engine.Engine.value's arithmetic on a fills list (the engine's source lines are checked to be MG.ENGINE_LINES)."""
    sg = lambda s: 1. if s == 0 else -1.  # noqa: E731
    cash = -sum(sg(s) * q * p for _, _, s, q, p, _ in fills); pos = sum(sg(s) * q for _, _, s, q, _, _ in fills)  # noqa: E702
    return cash, pos, cash + pos * mark


def L_(t, s, q, p=.5, role=0, fee=0., rec=0):
    return (float(t), int(s), float(q), float(p), int(role), float(fee), int(rec), W_)


# ------------------------------------------------------------------ unit checks (no files) ------------------------------------------------------------------
def t_boundary():
    got = MG.side_of([999., 1000., 3000., 5000., 5001.], 1000., 5000.).tolist()
    check("boundary '<': before start -1, start and END inside 0, after END +1", got == [-1, 0, 0, 0, 1], f'(got {got})')


def t_readings():
    cases = [((-1., 1., 0., 0.), 'our_turnover_loses'), ((-1., 1., 1., 5.), 'our_turnover_loses'), ((1., 1., 1., 2.), 'exposure_gap'),
             ((1., 1., 2., 1.), 'mixed_unsupported'), ((1., 1., 2., -2.), 'mixed_unsupported'), ((-1., -1., 1., 2.), 'exposure_gap'),
             ((-1., -1., 2., 1.), 'both_adverse'), ((-1., -1., 1., -1.), 'both_adverse'), ((0., 0., 0., 1.), 'exposure_gap'),
             ((0., 0., 0., 0.), 'mixed_unsupported'), ((1., -1., 0., 5.), 'mixed_unsupported'), ((-1., 0., 0., 5.), 'mixed_unsupported'),
             ((0., -1., 0., 5.), 'mixed_unsupported'), ((NAN, 1., 0., 5.), 'mixed_unsupported'), ((-1., NAN, 0., 5.), 'mixed_unsupported')]
    for (o, t, dT, dE), want in cases:
        got = MG.reading_of(o, t, dT, dE)
        check(f'reading o={o} t={t} dT={dT} dE={dE} -> {want}', got == want, f'(got {got})')


def _met(cn, cs, nc=1, carry=0., resid=0.):
    return dict(close_net=cn, close_sh=cs, n_closes=nc, carry=carry, resid=resid, fee_unclosed=0., self_gross=0., self_fee=0., total=cn + carry + resid)


def t_support():
    for nb in (4, 5):
        keys = [(D1, 1000 + i) for i in range(6)]
        res = {k: {p: {v: dict(status='ok' if (p == 'ours' or (p == 'rank1' and i < nb)) else 'no_activity',
                               met=_met(-1., 2.) if p == 'ours' else (_met(3., 4.) if (p == 'rank1' and i < nb) else None)) for v in MG.VIEWS}
                   for p in MG.PARTIES} for i, k in enumerate(keys)}
        RD = MG.readings(res, keys)
        rd = RD[('rank1', 'carry')]
        want = 'our_turnover_loses' if nb >= 5 else 'mixed_unsupported'
        check(f'support rule: {nb} both-active markets -> supported {nb >= 5}', rd['supported'] == (nb >= 5) and len(rd['both']) == nb)
        check(f'support rule: {nb} markets -> every reading {want}, raw our_turnover_loses',
              all(c['reading'] == want and c['reading_raw'] == 'our_turnover_loses' for c in rd['per'].values()))
        pl = rd['pooled']
        check(f'support rule: {nb} markets -> pooled {want}; o -50 c/sh, t +75 c/sh, dT {-4. * nb}', pl['reading'] == want and near(pl['o_c_sh'], -50.)
              and near(pl['t_c_sh'], 75.) and near(pl['d_turnover'], -4. * nb), f'({pl})')
        check(f'support rule: {nb} markets -> rank2 none, not supported', RD[('rank2', 'zero')]['both'] == [] and not RD[('rank2', 'zero')]['supported'])


def t_paths():
    r = MG.paths([L_(1, 1, 5, rec=1), L_(2, -1, 3, rec=2), L_(2, -1, 4, rec=3)], 0.)
    check('paths: strict-earlier position before (two simultaneous sells both see +5)',
          (r['n_actions'], r['n_nonflat'], r['n_opposing'], r['n_abs_reduce'], r['n_overshoot'], r['n_simultaneous']) == (3, 2, 2, 2, 0, 2)
          and near(r['max_abs_pos'], 5.) and near(r['end_pos'], -2.), f'({r})')
    r = MG.paths([L_(1, 1, 2, rec=7), L_(1, 1, 3, rec=7)], 0.)
    check('paths: the legs of one record are one action', r['n_actions'] == 1 and near(r['max_abs_pos'], 5.), f'({r})')
    r = MG.paths([L_(1, 1, 2, rec=1), L_(2, -1, 5, rec=2)], 0.)
    check('paths: opposing beyond flat = overshoot, not a reduction', (r['n_opposing'], r['n_abs_reduce'], r['n_overshoot']) == (1, 0, 1), f'({r})')
    r = MG.paths([L_(1, -1, 3, rec=1)], 5.)
    check('paths: from a carried position', (r['n_nonflat'], r['n_opposing'], r['n_abs_reduce']) == (1, 1, 1) and near(r['max_abs_pos'], 5.), f'({r})')


def t_hold():
    h = MG.hold_stats([6000., 15000., 9001., 20000., 268000.], [3., 1., 1., 3., 1.], [False] * 5)
    vals('hold quantiles (share-weighted, smallest reaching q)', h, dict(hold_p10_s=6., hold_p25_s=6., hold_p50_s=15., hold_p75_s=20., hold_p90_s=268.,
                                                                       hold_mean_s=370001. / 9. / 1000., hold_med_count_s=15.))
    h = MG.hold_stats([1000., 5000.], [1e-9, 1.], [True, False])
    check('hold quantiles: a dust close never enters', near(h['hold_p10_s'], 5.) and near(h['hold_med_count_s'], 5.), f'({h})')


def t_engine():
    src = (MG.RESEARCH / 'pf_branch_engine.py').read_text()
    check('engine source: Engine.value holds the declared lines verbatim', all(ln in src for ln in MG.ENGINE_LINES))
    mine = inspect.getsource(MG.engine_value)
    check('tool engine_value: the same lines on the record fills', all(ln.replace('self.fills', 'fills') in mine for ln in MG.ENGINE_LINES))
    fills = [[1., 0, 0, .1, 51., 0], [2., 1, 1, .2, 49., 1], [3., 0, 0, .3, 50., 2], [4., 2, 1, 7.607843137254902, 52., 3]]
    check('engine value: tool = reference, bit for bit', MG.engine_value(fills, 50.5) == engine_ref(fills, 50.5))


def t_identity():
    devs = {}
    ok = block('identity: 5e-7 on 1 USD passes', MG.check_identity, 1., 1. + 5e-7, 'x', devs)
    check('identity: 5e-7 on 1 USD passes', ok is not None and near(devs.get('identity'), 5e-7, 1e-12))
    expect('identity falsifier: 2e-6 on 1 USD refused', lambda: MG.check_identity(1., 1. + 2e-6, 'x', {}), 'identity')
    expect('identity falsifier: 2e-3 on 1,000 USD refused', lambda: MG.check_identity(1000., 1000.002, 'x', {}), 'identity')
    expect('identity falsifier: NaN refused', lambda: MG.check_identity(NAN, 1., 'x', {}), 'identity')


def t_cross_view():
    expect('cross-view falsifier: identical views with carried_pos 2 and END .60 != open .35 refused',
           lambda: MG.check_cross_view(1., 1., 2., .60, .35, 'x', {}, {}), 'cross-view')
    expect('cross-view falsifier: a gap other than carried_pos x (END - open) refused', lambda: MG.check_cross_view(1.6, 1., 2., .60, .35, 'x', {}, {}),
           'cross-view')
    devs, cnt = {}, {}
    ok1 = block('cross-view: carry - zero = 2 x .25 passes', MG.check_cross_view, 1.5, 1., 2., .60, .35, 'x', devs, cnt)
    ok2 = block('cross-view: identical views with END = opening mark pass (correct, not a defect)', MG.check_cross_view, 1., 1., 2., .50, .50, 'x', devs, cnt)
    ok3 = block('cross-view: no carried position, opening mark missing passes', MG.check_cross_view, 1., 1., 0., .60, NAN, 'x', devs, cnt)
    check('cross-view: the three valid cases pass and are counted', None not in (ok1, ok2, ok3) and cnt.get('cross_view_checks') == 3, f'({cnt})')


def t_fifo():
    expect('fifo: legs out of order refused', lambda: MG.fifo([L_(2, 1, 1), L_(1, -1, 1)]), 'not chronological')
    expect('fifo: a within leg as a carried lot refused', lambda: MG.fifo([L_(1, 1, 1)], init=[L_(0, 1, 1)]), 'carried lots must precede')
    expect('fifo: carried lots on both sides refused', lambda: MG.fifo([], init=[(0., 1, 1., .5, 0, 0., 0, K_), (0., -1, 1., .5, 0, 0., 0, K_)]), 'both sides')


def t_hand():
    """The M1 rank-1 case through fifo + account directly (no files), both views, and a carried lot left open."""
    M, om = .60, .35
    pre = [L_(-5000, 1, 6, .30, 0, 0., 1), L_(-2000, -1, 2, .32, 1, .01, 2), L_(999, 1, 1, .31, 0, 0., 3)]
    inn = [L_(1000, -1, 3, .38, 1, .03, 4), L_(10000, -1, 5, .40, 0, 0., 5), L_(30000, 1, 5, .42, 1, .05, 7), L_(298000, -1, 1, .58, 0, 0., 9)]
    selfr = [(1, 2., .45, 0.), (-1, 2., .45, .02)]
    fp = MG.fifo(pre)
    it = fp['items']
    lots = [(it[i][0], it[i][1], fp['rem'][i], om, it[i][4], 0., it[i][6], K_) for i in fp['book']]
    check('hand: carried lots = the pre-start FIFO remainder at the opening mark, actual times and roles',
          lots == [(-5000., 1, 4., .35, 0, 0., 1, K_), (999., 1, 1., .35, 0, 0., 3, K_)], f'({lots})')
    cnt = {}
    met, cl = MG.account(MG.fifo(inn, lots), inn, selfr, M, om, 5., 5., 'hand carry', {}, cnt)
    vals('hand carry', met, dict(close_sh=9., close_pnl=.29, close_fee=.07, close_net=.22, carried_close_sh=5., carried_close_net=.16, resid=.18,
                                 fee_unclosed=.01, carry=0., unclosed_sh=1., unclosed_carried_sh=0., self_gross=0., self_fee=.02, total=.37, direct=.37,
                                 max_abs_pos=5., end_pos=1., n_actions=4, n_nonflat=4, n_opposing=4, n_abs_reduce=3, n_overshoot=1, n_simultaneous=0,
                                 buy_sh=5., sell_sh=9., n_closes=5, n_dust=0))
    got = [(round(a, 9), round(b, 9), round(c, 9), int(d), int(e), int(f)) for a, b, c, d, e, f in
           zip(cl['sh'], cl['pnl'], cl['fee'], cl['hold'], cl['rp'], cl['kind'])]
    want = [(3., .09, .03, 6000, 1, K_), (1., .05, 0., 15000, 0, K_), (1., .05, 0., 9001, 0, K_), (3., -.06, .03, 20000, 1, W_), (1., .16, .01, 268000, 2, W_)]
    check('hand carry: the five closes (shares, P&L, fees, hold, role pair, entry kind)', got == want, f'({got})')
    met0, cl0 = MG.account(MG.fifo(inn), inn, selfr, M, om, None, 0., 'hand zero', {}, cnt)
    vals('hand zero', met0, dict(close_sh=5., close_pnl=-.16, close_fee=.08, close_net=-.24, resid=-.62, fee_unclosed=0., carry=0., total=-.88,
                                 direct=-.88, max_abs_pos=8., end_pos=-4., n_actions=4, n_nonflat=3, n_opposing=1, n_abs_reduce=1, n_overshoot=0))
    got0 = [(round(a, 9), round(b, 9), round(c, 9), int(d), int(e)) for a, b, c, d, e in zip(cl0['sh'], cl0['pnl'], cl0['fee'], cl0['hold'], cl0['rp'])]
    check('hand zero: the two closes', got0 == [(3., -.12, .06, 29000, 3), (2., -.04, .02, 20000, 1)], f'({got0})')
    one = inn[:1]
    m1, _ = MG.account(MG.fifo(one, lots), one, [], M, om, 5., 5., 'hand open carry', {}, cnt)
    vals('hand: carried lots left open -> carry at the opening basis', m1, dict(close_net=.06, carry=.50, unclosed_carried_sh=2., total=.56, direct=.56))
    check('hand: one identity check per accounting', cnt.get('identity_checks') == 3, f'({cnt})')


def t_our_market():
    sp = OURS[101]
    row = our_row(101, sp)
    pinfo, rec = row['path_info'], row[MG.ARM]
    mk = MG.our_market(D1, 101, pinfo, rec)
    check('our_market: M1 ok, exact, start / END / mark', mk['status'] == 'ok' and mk['value_exact'] is True and mk['start'] == WS[101] + 1000.
          and mk['end'] == WS[101] + 298000. and near(mk['end_mark'], .60) and near(mk['value_c'], 110.), f'({ {k: mk[k] for k in ("status", "start", "end")} })')
    check('our_market: legs (t_abs = start + t, s, q, p / 100, maker, fee 0, trade row)',
          mk['legs'] == [(WS[101] + 2000., 1, 10., .40, 0, 0., 0, W_), (WS[101] + 3000., -1, 4., .45, 0, 0., 1, W_), (WS[101] + 298000., -1, 3., .50, 0, 0., 2, W_)],
          f'({mk["legs"]})')
    r2_ = json.loads(json.dumps(rec))
    r2_['value'] = math.nextafter(r2_['value'], math.inf)
    expect('our_market: a recorded value one ulp off refused', lambda: MG.our_market(D1, 101, pinfo, r2_), 'do not reproduce exactly')
    r3_ = json.loads(json.dumps(rec))
    r3_['fills'].append([297001., 3, 1, 1., 50., 3])
    r3_['cash'], r3_['position'], r3_['value'] = engine_ref(r3_['fills'], r3_['mark'])
    expect('our_market: a fill after its own END refused', lambda: MG.our_market(D1, 101, pinfo, r3_), 'outside its own episode')
    r4_ = json.loads(json.dumps(rec))
    r4_['fills'][0][2] = True
    expect('our_market: a boolean side refused', lambda: MG.our_market(D1, 101, pinfo, r4_), 'malformed fill')
    expect('our_market: path_info differing from the ident refused', lambda: MG.our_market(D1, 101, dict(pinfo, t_start=pinfo['t_start'] + 50.), rec), 'path_info')
    r5_ = json.loads(json.dumps(rec))
    r5_['mark'] = 100.
    check('our_market: a mark outside (0, 100) -> no_interval', MG.our_market(D1, 101, pinfo, r5_)['status'] == 'no_interval')
    row6 = our_row(203, OURS[203])
    m6 = MG.our_market(D2, 203, row6['path_info'], row6[MG.ARM])
    check('our_market: a missing_mark record -> no_interval', m6['status'] == 'no_interval' and m6['reason'] == 'record_status_missing_mark', f'({m6["reason"]})')


# ------------------------------------------------------------------ the synthetic world ------------------------------------------------------------------
def our_row(m, spec):
    d, ws, w = DAY[m], WS[m], spec.get('wait', 0)
    start, er = float(ws + w), 298000. - w
    fills = [list(f) for f in spec['fills']]
    st = spec.get('status', 'ok')
    rec = dict(status=st, fills=fills, mark=spec['mark'], end=er, settings=dict(contract='branch_ladder_async_v1', end_ms=er), closed={}, orders=[],
               ident=dict(config='LADDER_REST_INVENTORY_v1', day=d, end_abs_ms=start + er, end_rel=er, instrument_id=m, opening_ms=float(ws),
                          mark_source='QP.build_path supported-end mark', start_abs_ms=start, tape_id='synthetic'))
    if st == 'ok':
        rec['cash'], rec['position'], rec['value'] = engine_ref(fills, spec['mark'])
    else:
        rec['cash'] = rec['position'] = rec['value'] = None
    return {MG.ARM: rec, 'FRONT_ASYNC_FAST10': dict(status='ok'), 'day': d, 'm': m,
            'path_info': dict(t_start=start, end_rel=er, ws=float(ws), opening_wait_ms=float(w), boundary_marks={})}


def fv_rows(legs):
    """Both rows (maker, taker) of every leg in the pf_mt_econ.FV schema (V = 1, every mark ok, fees COLLATERAL = fee_fill)."""
    out = []
    for g in legs:
        V, me = 1., ENDM[g['mid']]
        for role in ('maker', 'taker'):
            mk = role == 'maker'
            addr, cp = (g['maker'], g['taker']) if mk else (g['taker'], g['maker'])
            s = g['s_mk'] if mk else -g['s_mk']
            fee = g['mfee'] if mk else g['tfee']
            q, p = g['q'], g['p']
            gross = s * q * (V - p)
            lf = 'na_not_book' if mk else 'na_taker'
            out.append(dict(day=DAY[g['mid']], mid=g['mid'], sid=g['sid'], mk_idx=0, px_rank=0, role=role, addr=addr, hash=f"{role[0]}h_{g['sid']}",
                            cp_addr=cp, kind='regular', own_index_set=1, y_side='bid' if s == 1 else 'ask', s=s, q=q, p=p, px_gap=NAN,
                            self_trade=g['maker'] == g['taker'], settle_ms=g['t'], t_ms=g['t'], win_ms=-1, t_status='hi_no_lifecycle',
                            t_rel_s=(g['t'] - WS[g['mid']]) / 1000., tseg='w000_060', pseg='d_0.0_0.1', V=V, st_res='ok', gross_res=gross, m_minus=p,
                            age_ms=0, spr_w=.01, st_spr='ok', v_spr=0., st_spr_hi='ok', v_spr_hi=0., st_1s='ok', v_1s=0., st_10s='ok', v_10s=0.,
                            st_60s='ok', v_60s=0., m_end=me, st_end='ok', v_end=s * q * (me - p), fee=fee, fee_type='COLLATERAL', fee_dy=0., fee_dc=-fee,
                            fee_fill=fee, fee_res=fee, net_res=gross - fee, reb25_fill=0., reb25_res=0., mq=NAN, rq=NAN, life_rep=lf, life_exh=lf,
                            life_front=lf, life_status=lf, dpt='d0', sweep='single_price', cp='cp_other', boost='unknown', src_status=g['status'],
                            status=g['status']))
    F = pd.DataFrame(out)
    X.need(set(F.columns) == set(E.FV), f'selftest world: columns differ from pf_mt_econ.FV: {sorted(set(F.columns) ^ set(E.FV))}')
    return F[list(E.FV)]


def econ_world(F):
    """pf_mt_econ.fifo (mq / rq) and market_table (maker_market) per day, with econ's own conservation check."""
    parts, mms = [], []
    for d in (D1, D2):
        Fd = F[(F['day'] == d).to_numpy()].copy().reset_index(drop=True)
        for c in ('addr', 'sid'):
            Fd[c] = pd.Categorical(Fd[c].astype(object))
        Fd['role'] = pd.Categorical(Fd['role'].astype(object), categories=['maker', 'taker'])
        E.fifo(Fd)
        mids = sorted(int(m) for m in set(Fd['mid'].tolist()))
        reg = pd.DataFrame({'mid': np.array(mids, np.int64), 'V': np.ones(len(mids)), 'censored': np.zeros(len(mids), bool),
                            'upbad': np.zeros(len(mids), bool), 'tie': np.zeros(len(mids), bool)})
        M = E.market_table(Fd, d, 'FILL', [], {m: (ENDM[m], 'ok') for m in mids}, reg)
        E.check_market(M)
        parts.append(Fd.astype({'addr': object, 'sid': object, 'role': object}))
        mms.append(M)
    return pd.concat(parts, ignore_index=True)[list(E.FV)], pd.concat(mms, ignore_index=True)


def gz_bytes(rows):
    buf = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=buf, mtime=0) as g:
        g.write(b''.join((json.dumps(r, sort_keys=True) + '\n').encode() for r in rows))
    return buf.getvalue()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def world(root, F, MM, ours=None, row_mod=None, arms_mod=None, frames=None, fv_extra=None, manifest=None):
    """Every input of the tool under root (manifest, our day records + accepted measurement, econ, tob / markets) -> the tool's paths."""
    lad, ec, v2 = root / 'ladder', root / 'econ', root / 'v2'
    for p in (lad / 'pilot', lad / 'rest', lad / 'measure', ec, v2, root / 'rcpt'):
        p.mkdir(parents=True)
    man = manifest or dict(days=[D1, D2], complete=BOOK, parts=PARTS)
    (root / 'manifest.json').write_text(json.dumps(man, indent=1, sort_keys=True) + '\n')
    msha = sha((root / 'manifest.json').read_bytes())
    specs, values = ours or OURS, {}
    for part, pm in PARTS.items():
        for d, mem in pm.items():
            rows = []
            for m in mem:
                row = our_row(m, specs[m])
                if row_mod and m in row_mod:
                    row_mod[m](row)
                rows.append(row)
                values[(d, m)] = row[MG.ARM].get('value')
            b = gz_bytes(rows)
            (lad / part / f'{d}.jsonl.gz').write_bytes(b)
            arms = (['PARITY_BASELINE'] if part == 'pilot' else []) + [MG.ARM, 'FRONT_ASYNC_FAST10']
            (lad / part / f'{d}.json').write_text(json.dumps(dict(version=MG.DAY_VERSION, day=d, part=part, members=mem, arms=arms, status='DAY_DONE',
                                                                  sha256=dict(gz=sha(b), npz='0' * 64), manifest_sha256=msha), indent=1) + '\n')
    A = []
    for (d, m), v in sorted(values.items()):
        vv = NAN if v is None else float(v) + (arms_mod or {}).get((d, m), 0.)
        A.append(dict(day=d, m=m, arm=MG.ARM, both_exact=True, value=vv))
        A.append(dict(day=d, m=m, arm='FRONT_ASYNC_FAST10', both_exact=True, value=vv + 1000.))
    pd.DataFrame(A).to_csv(lad / 'measure' / 'arms.csv', index=False)
    (lad / 'measure' / 'audit.json').write_text(json.dumps(dict(outputs={'arms.csv': sha((lad / 'measure' / 'arms.csv').read_bytes())})) + '\n')
    w = pq.ParquetWriter(ec / 'fills_valued.parquet', E._arrow_schema(E.FV))
    for ms in ([101, 102, 103, 104], [150], [201, 202, 203]):
        w.write_table(E._to_arrow(F.loc[F['mid'].isin(ms).to_numpy(), list(E.FV)].reset_index(drop=True), E.FV))
    if fv_extra is not None:
        w.write_table(E._to_arrow(fv_extra[list(E.FV)].reset_index(drop=True), E.FV))
    w.close()
    w = pq.ParquetWriter(ec / 'maker_market.parquet', E._arrow_schema(E.MM))
    for d in (D1, D2):
        w.write_table(E._to_arrow(MM.loc[(MM['day'] == d).to_numpy(), list(E.MM)].reset_index(drop=True), E.MM))
    w.close()
    (ec / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(day=C.ID_DAY, top10_shares=[dict(rank=i + 1, addr=a) for i, a in enumerate(R)]))))
    spec = C.SCHEMAS['tob']
    for d in (D1, D2):
        tr = [dict(day=d, mid=m, upd_ms=t, recv_ns=t * 1000000 + 5, conn='c1', order_count=2, n_bid=1, n_ask=1, best_bid=b_, best_bid_sh=10., best_ask=a_,
                   best_ask_sh=10., pend_bid_usd=0., pend_ask_usd=0., status='ok') for m, t, b_, a_ in (FRAMES if frames is None else frames) if DAY[m] == d]
        pq.write_table(E._to_arrow(pd.DataFrame(tr, columns=list(spec)), spec), v2 / f'tob_{d}.parquet')
        mk = pd.DataFrame([dict(day=d, mid=m, ws=WS[m] // 1000, we=WS[m] // 1000 + 300, status='ok') for m in sorted(WS) if DAY[m] == d])
        pq.write_table(pa.Table.from_pandas(mk, preserve_index=False), v2 / f'markets_{d}.parquet')
    return dict(manifest=root / 'manifest.json', ladder=lad, econ=ec, root=v2, out=root / 'rcpt' / 'matched_gap')


# ------------------------------------------------------------------ world and end-to-end checks ------------------------------------------------------------------
def chk_world(F, MM):
    a = F[((F['sid'] == 'm1-i') & (F['addr'] == r1)).to_numpy()].iloc[0]
    check('world: econ FIFO leaves 1 of the post-END sell open (mq 1, rq 1)', near(a['mq'], 1.) and near(a['rq'], 1.))
    b = MM[((MM['mid'] == 101) & (MM['addr'] == r1)).to_numpy()].iloc[0]
    check('world: econ maker_market M1 rank 1: 10 ok rows, 1 excluded, net -1, fee .11', int(b['n_mk'] + b['n_tk']) == 10 and int(b['n_excl']) == 1
          and near(b['net_pos'], -1.) and near(b['fee_fill'], .11))


def chk_main(r, P):
    out = P['out']
    check('e2e: the four outputs, no partial directory', sorted(p.name for p in out.iterdir()) == ['answer.json', 'closes_by_role.csv', 'per_market.csv',
                                                                                                  'receipt.json']
          and not list(out.parent.glob('.matched_gap.partial_*')))
    PM = pd.read_csv(out / 'per_market.csv', dtype={'day': str})
    check('e2e: per_market rows = 6 markets x 6 parties x 2 views', len(PM) == 72 and list(PM.columns) == MG.PM_COLS, f'({len(PM)})')
    g = lambda m, p, v: pick(PM, day=DAY[m], m=m, party=p, view=v)  # noqa: E731
    for v in MG.VIEWS:
        vals(f'ours M1 {v}', g(101, 'ours', v), dict(status='ok', close_net=.50, close_sh=7., n_closes=2, resid=.60, carry=0., total=1.10, direct=1.10,
                                                    value_recorded_c=110., max_abs_pos=10., n_actions=3, n_nonflat=2, n_opposing=2, n_abs_reduce=2,
                                                    n_overshoot=0, hold_p10_s=1., hold_p50_s=1., hold_p75_s=296., hold_p90_s=296.,
                                                    hold_mean_s=892000. / 7. / 1000., hold_med_count_s=1., close_net_c_sh=50. / 7., open_mark=.35,
                                                    open_status='ok', open_age_ms=0, end_mark=.60, start_ms=WS[101] + 1000., end_ms=WS[101] + 298000.,
                                                    end_mark_minus_tob_c=0.))
        check(f'ours M1 {v}: value exact', str(g(101, 'ours', v)['value_exact']) == 'True')
    vals('rank1 M1 carry', g(101, 'rank1', 'carry'), dict(
        status='ok', close_sh=9., close_net=.22, close_fee=.07, carried_close_sh=5., carried_close_net=.16, resid=.18, carry=0., fee_unclosed=.01,
        self_fee=.02, self_gross=0., total=.37, direct=.37, exposure=.18, carried_pos_recorded=5., carried_lots_recorded=2,
        carried_first_ms=WS[101] - 5000., n_rows_pre=3, n_rows_in=4, n_rows_post=1, n_self_in=2, n_excl_pre=0, n_excl_in=1, n_excl_post=0,
        max_abs_pos=5., end_pos=1., n_actions=4, n_nonflat=4, n_opposing=4, n_abs_reduce=3, n_overshoot=1, hold_p10_s=6., hold_p50_s=15.,
        hold_p75_s=20., hold_p90_s=268., hold_mean_s=370001. / 9. / 1000., close_net_c_sh=22. / 9., both_active='True', reading='exposure_gap',
        reading_raw='exposure_gap', d_turnover=.28, d_exposure=.42, d_fee_unclosed=.01, d_self=.02, d_total=.73, o_c_sh=50. / 7., t_c_sh=22. / 9.))
    vals('rank1 M1 zero', g(101, 'rank1', 'zero'), dict(status='ok', close_sh=5., close_net=-.24, close_fee=.08, resid=-.62, fee_unclosed=0., total=-.88,
                                                        direct=-.88, max_abs_pos=8., end_pos=-4., n_opposing=1, n_abs_reduce=1, hold_p50_s=29.,
                                                        hold_mean_s=25.4, reading='mixed_unsupported', reading_raw='mixed_unsupported',
                                                        carried_pos_recorded=5.))
    vals('top10 M1 carry (rank 5 carry_only enters)', g(101, 'top10', 'carry'), dict(
        status='ok', close_sh=11., close_net=.32, carry=-.75, resid=.18, unclosed_carried_sh=3., carried_close_sh=5., total=-.28, direct=-.28,
        carried_pos_recorded=2., carried_pos_in=2., carried_lots_recorded=3, n_members_ok=2, n_members_carry_only=1, n_members_no_activity=7,
        max_abs_pos=5., party_identity_dev=0., reading='exposure_gap', d_turnover=.18, d_exposure=1.17, d_total=1.38))
    vals('top10 M1 zero (carried lots ignored)', g(101, 'top10', 'zero'), dict(
        status='ok', close_net=-.14, carry=0., unclosed_carried_sh=0., total=-.78, carried_pos_recorded=2., carried_pos_in=0., n_members_ok=2,
        n_members_carry_only=0, n_members_no_activity=8))
    tc, tz = g(101, 'top10', 'carry')['total'], g(101, 'top10', 'zero')['total']
    check('top10 M1: carry and zero views differ by carried_pos 2 x (END .60 - open .35)', tc != tz and near(tc - tz, .50), f'({tc!r} vs {tz!r})')
    vals('rank1 M1 carried position in the views', g(101, 'rank1', 'carry'), dict(carried_pos_in=5.))
    vals('rank1 M1 zero carried position', g(101, 'rank1', 'zero'), dict(carried_pos_in=0.))
    vals('rank5 alone is not a party; top3 M1 (ranks 1-3) unchanged', g(101, 'top3', 'carry'), dict(carried_pos_recorded=5., total=.37,
                                                                                                     n_members_carry_only=0))
    vals('top3 M1 carry', g(101, 'top3', 'carry'), dict(status='ok', close_net=.22, total=.37, n_members_ok=1, n_members_no_activity=2))
    vals('rank2 M1 carry', g(101, 'rank2', 'carry'), dict(status='no_activity', both_active='False', pair_status='not_both_active:ours_ok|theirs_no_activity'))
    vals('top3 M2 carry', g(102, 'top3', 'carry'), dict(close_net=.13, close_sh=5., reading='our_turnover_loses', n_members_ok=2))
    vals('rank2 M2 carry', g(102, 'rank2', 'carry'), dict(close_net=.09, reading='mixed_unsupported', reading_raw='our_turnover_loses'))
    vals('rank1 M3 carry', g(103, 'rank1', 'carry'), dict(close_net=-.50, reading='both_adverse', d_turnover=.48, d_exposure=0.))
    vals('rank1 M4 carry', g(201, 'rank1', 'carry'), dict(close_net=-.02, reading='exposure_gap', d_exposure=-.10, d_turnover=.01))
    vals('ours M4 carry', g(201, 'ours', 'carry'), dict(close_net=-.01, resid=-.10, total=-.11, value_recorded_c=-11.))
    vals('rank1 M5 carry', g(202, 'rank1', 'carry'), dict(reading='mixed_unsupported', reading_raw='mixed_unsupported'))
    vals('ours M6', g(203, 'ours', 'carry'), dict(status='no_interval', market_status='no_interval', market_reason='record_status_missing_mark'))
    vals('top10 M6', g(203, 'top10', 'zero'), dict(status='no_interval', pair_status='not_both_active:ours_no_interval|theirs_no_interval'))
    A = json.loads((out / 'answer.json').read_text())
    rc = A['readings']['rank1']['carry']
    cn = {k: (v['n'], v['markets']) for k, v in rc['counts'].items()}
    check('answer rank1 carry: counts with markets', cn == {'exposure_gap': (2, [f'{D1}:101', f'{D2}:201']), 'our_turnover_loses': (1, [f'{D1}:102']),
                                                           'both_adverse': (1, [f'{D1}:103']), 'mixed_unsupported': (1, [f'{D2}:202'])}, f'({cn})')
    check('answer rank1 carry: 5 of 6 markets, supported, denominators 13 / 24 sh, market mix 3 + 2',
          rc['n_markets_both'] == 5 and rc['n_markets'] == 6 and rc['supported'] is True and near(rc['our_close_sh'], 13.) and near(rc['their_close_sh'], 24.)
          and rc['market_mix'] == {D1: 3, D2: 2} and rc['counts']['exposure_gap']['n_markets_both'] == 5
          and near(rc['counts']['exposure_gap']['our_close_sh'], 8.) and near(rc['counts']['exposure_gap']['their_close_sh'], 11.), f'({rc["market_mix"]})')
    pl = rc['pooled_secondary']
    vals('answer rank1 carry pooled', pl, dict(reading='mixed_unsupported', o_c_sh=47. / 13., t_c_sh=-27. / 24., d_turnover=.74, d_exposure=.32, d_total=1.09))
    cz = {k: v['n'] for k, v in A['readings']['rank1']['zero']['counts'].items()}
    check('answer rank1 zero: counts', cz == {'our_turnover_loses': 1, 'exposure_gap': 1, 'both_adverse': 1, 'mixed_unsupported': 2}, f'({cz})')
    r2c = A['readings']['rank2']['carry']
    check('answer rank2 carry: 1 market, unsupported, forced mixed (raw our_turnover_loses)', r2c['n_markets_both'] == 1 and r2c['supported'] is False
          and r2c['counts']['mixed_unsupported']['n'] == 1 and r2c['counts_raw_before_support_rule']['our_turnover_loses'] == 1
          and r2c['pooled_secondary']['reading'] == 'mixed_unsupported')
    check('answer rank3 carry: no market both active', A['readings']['rank3']['carry']['n_markets_both'] == 0
          and all(v['n'] == 0 for v in A['readings']['rank3']['carry']['counts'].values()))
    for p in ('top3', 'top10'):
        cp_ = {k: v['n'] for k, v in A['readings'][p]['carry']['counts'].items()}
        check(f'answer {p} carry: the rank1 pattern', cp_ == {'exposure_gap': 2, 'our_turnover_loses': 1, 'both_adverse': 1, 'mixed_unsupported': 1}, f'({cp_})')
    check('answer statuses: rank2 carry ok 1, no_activity 4, no_interval 1', A['statuses']['rank2']['carry'] == {'ok': 1, 'no_activity': 4, 'no_interval': 1},
          f'({A["statuses"]["rank2"]["carry"]})')
    bd = A['by_day_secondary']['rank1']['carry'][D1]
    check('answer by day (rank1 carry D1): 3 markets, our close net .46, their -.24', bd['n_markets_both'] == 3 and near(bd['our_close_net'], .46)
          and near(bd['their_close_net'], -.24), f'({bd})')
    check('answer: point estimates on two days, no interval', A['ci'] == 'none_G2_lt_5' and A['primary_view'] == 'carry')
    CR = pd.read_csv(out / 'closes_by_role.csv')
    h = lambda **kw: pick(CR, **kw)  # noqa: E731
    vals('closes rank1 carry active carried m->t', h(party='rank1', view='carry', scope='active', entry_kind='carried', role_pair='maker->taker'),
         dict(n_markets=5, n_closes=1, sh=3., pnl=.09, fee=.03, net=.06))
    vals('closes rank1 carry active carried m->m', h(party='rank1', view='carry', scope='active', entry_kind='carried', role_pair='maker->maker'),
         dict(n_closes=2, sh=2., net=.10))
    vals('closes rank1 carry active within m->t', h(party='rank1', view='carry', scope='active', entry_kind='within', role_pair='maker->taker'),
         dict(n_closes=2, sh=13., pnl=-.56, fee=.03, net=-.59))
    vals('closes rank1 carry active within t->m', h(party='rank1', view='carry', scope='active', entry_kind='within', role_pair='taker->maker'),
         dict(n_closes=1, sh=1., net=.15, hold_p50_s=268.))
    vals('closes rank1 carry active all', h(party='rank1', view='carry', scope='active', entry_kind='all', role_pair='all'),
         dict(n_closes=9, sh=24., net=-.27, c_net_sh=-27. / 24.))
    vals('closes ours carry both_active_rank2', h(party='ours', view='carry', scope='both_active_rank2', entry_kind='all', role_pair='all'),
         dict(n_markets=1, sh=2., net=-.02))
    vals('closes ours zero active', h(party='ours', view='zero', scope='active', entry_kind='all', role_pair='all'), dict(n_markets=5, sh=13., net=.47))
    rec = json.loads((out / 'receipt.json').read_text())
    cnt = rec['counts']
    check('receipt: identity checks 20 (ours 5 + theirs 14 + rank5 carry_only 1), econ FIFO rows 21, completeness groups 9',
          cnt.get('identity_checks') == 20 and cnt.get('econ_fifo_rows') == 21 and cnt.get('completeness_groups') == 9, f'({cnt})')
    check('receipt: party identity 32, cross-view 21, carry_only members 1; 3 rows with a carried position, all 3 with differing views',
          cnt.get('party_identity_checks') == 32 and cnt.get('cross_view_checks') == 21 and cnt.get('member_carry_only') == 1
          and cnt.get('rows_carried_pos') == 3 and cnt.get('rows_carried_pos_views_differ') == 3 and cnt.get('rows_carried_pos_end_eq_open', 0) == 0,
          f'({cnt})')
    check('receipt: the party and cross-view deviations recorded', 'party_identity' in rec['checks'] and 'cross_view' in rec['checks']
          and rec['checks']['cross_view']['max_dev'] < 1e-9 and rec['out'] == str(out))
    check('receipt: fills_valued row groups read 2, the market-150 group pruned unread', cnt.get('fills_valued_groups_read') == 2
          and cnt.get('fills_valued_groups_pruned') == 1, f'({cnt})')
    check('receipt: excluded rows counted by status, our no_interval counted', cnt.get('excluded_rows:src_status') == 1 and cnt.get('ours_no_interval') == 1)
    check('receipt: value exact on 5 of 5 interval markets; the accepted values agree', rec['checks']['value_exact'] == dict(markets=5, of=5, ok=True)
          and rec['checks']['accepted_value_c']['max_dev'] == 0.)
    check('receipt: input hashes of every opened input', sorted(rec['inputs']) == sorted(
        ['manifest', 'econ/identities.json', 'econ/fills_valued.parquet', 'econ/maker_market.parquet', 'ours/measure/audit.json', 'ours/measure/arms.csv']
        + [f'ours/{pt}/{d}.{x}' for pt, pm in PARTS.items() for d in pm for x in ('json', 'jsonl.gz')]
        + [f'v2/{k}_{d}.parquet' for d in (D1, D2) for k in ('tob', 'markets')]), f'({sorted(rec["inputs"])})')
    check('receipt: outputs hashed with rows, CPU and memory recorded', rec['outputs']['per_market.csv']['rows'] == 72 and rec['cpu_s']['total'] >= 0
          and rec['memory']['vmhwm_mib'] > 0 and rec['memory']['guard_mib'] == C.GUARD_MIB)


def chk_rehearsal(P):
    r = MG.build(P, write=False)
    s = MG.rehearsal_summary(r)
    txt = json.dumps(X._clean(s))
    check('rehearsal: no write', sorted(p.name for p in P['out'].iterdir()) == ['answer.json', 'closes_by_role.csv', 'per_market.csv', 'receipt.json'])
    check('rehearsal summary: format facts', s['facts']['fills'] == 11 and s['facts']['fills_at_END'] == 1 and s['facts']['fills_at_start'] == 0
          and s['facts']['value_exact'] == [True] * 5 and s['facts']['open_status'][0] == 'ok', f'({s["facts"]})')
    banned = ('close_net', 'close_pnl', 'resid', 'pnl', 'exposure', 'reading', 'our_turnover_loses', 'exposure_gap', 'both_adverse', 'mixed_unsupported',
              'd_turnover', 'pooled')
    check('rehearsal summary: no P&L and no reading printed', all(w not in txt for w in banned), f'({[w for w in banned if w in txt]})')


def refusals(tmp, F, MM):
    n = [0]

    def fresh(**kw):
        n[0] += 1
        return world(tmp / f'w{n[0]:02d}', F, MM, **kw)

    P = fresh(row_mod={102: lambda row: row[MG.ARM].update(value=row[MG.ARM]['value'] + .01)})
    expect('refuse: a recorded value one cent off', lambda: MG.build(P, write=True), 'do not reproduce exactly', P['out'])
    P = fresh(row_mod={101: lambda row: row[MG.ARM].update(value=math.nextafter(row[MG.ARM]['value'], math.inf))})
    expect('refuse: a recorded value one ulp off', lambda: MG.build(P, write=True), 'do not reproduce exactly', P['out'])
    P = fresh(arms_mod={(D1, 102): 1e-6})
    expect('refuse: the accepted arms.csv value differs', lambda: MG.build(P, write=True), 'differs from the accepted arms.csv value', P['out'])
    MM2 = MM.copy()
    MM2.loc[((MM2['mid'] == 101) & (MM2['addr'] == r1)).to_numpy(), 'n_mk'] += 1
    P = world(tmp / 'w_mm', F, MM2)
    expect('refuse: a row missing against maker_market', lambda: MG.build(P, write=True), 'completeness', P['out'])
    F2 = F.copy()
    i = F2.index[((F2['sid'] == 'm2-a') & (F2['addr'] == r1)).to_numpy()][0]
    F2.loc[i, 'mq'] = F2.loc[i, 'mq'] - .5
    F2.loc[i, 'rq'] = F2.loc[i, 'rq'] + .5
    P = world(tmp / 'w_mq', F2, MM)
    expect('refuse: econ mq / rq not reproduced by the FIFO', lambda: MG.build(P, write=True), 'econ FIFO check', P['out'])
    ex = F[(F['sid'] == 'm2-a').to_numpy()].copy()
    ex['day'] = D2
    P = fresh(fv_extra=ex)
    expect("refuse: a BOOK market's row under another day", lambda: MG.build(P, write=True), 'carries another day', P['out'])
    P = fresh(row_mod={103: lambda row: row['path_info'].update(t_start=row['path_info']['t_start'] + 50.)})
    expect('refuse: path_info differing from the ident', lambda: MG.build(P, write=True), 'path_info', P['out'])
    P = fresh(ours={**OURS, 102: dict(OURS[102], fills=[OURS[102]['fills'][1], OURS[102]['fills'][0]])})
    expect('refuse: our fills out of chronological order', lambda: MG.build(P, write=True), 'not chronological', P['out'])
    P = fresh(manifest=dict(days=[C.ID_DAY], complete={C.ID_DAY: [101]}, parts=dict(pilot={C.ID_DAY: [101]}, rest={C.ID_DAY: []})))
    expect('refuse: a manifest day outside the measurement days, before any other path', lambda: MG.build(P, write=True), 'measurement days', P['out'])
    P = fresh()
    MG._MUT_EXCLUDE_CARRY_ONLY = True
    try:
        expect('refuse: the planted first-run defect (carry_only lots left out of the carry view) caught by the party identity',
               lambda: MG.build(P, write=True), 'identity: party top10', P['out'])
    finally:
        MG._MUT_EXCLUDE_CARRY_ONLY = False
    P = fresh()
    expect('refuse: the population size (contract 19)', lambda: MG.build(P, write=True, n_expected=MG.N_BOOK), 'the population holds 6 markets', P['out'])
    g = X._git(list(MG.TOOL_FILES))
    if g.get('tracked') and g.get('unmodified'):
        check('freeze guard: the committed tool passes', True)
    else:
        expect('refuse: the freeze guard (tool not committed)', lambda: MG.build(P, write=True, require_frozen=True), 'freeze', P['out'])
    fr = [x for x in FRAMES if not (x[0] == 101 and x[1] <= WS[101] + 1000)]
    P = fresh(frames=fr)
    r = block('no opening mark: build', lambda: MG.build(P, write=False))
    if r is not None:
        PMx = r['per_market']
        rw = PMx[((PMx['m'] == 101) & (PMx['party'] == 'rank1')).to_numpy()]
        st = dict(zip(rw['view'], rw['status']))
        check('no opening mark: rank1 M1 carry no_opening_mark, zero ok', st == {'carry': 'no_opening_mark', 'zero': 'ok'}, f'({st})')
        tw = PMx[((PMx['m'] == 101) & (PMx['party'] == 'top10') & (PMx['view'] == 'carry')).to_numpy()].iloc[0]
        check('no opening mark: top10 M1 carry no_opening_mark (rank1 active and rank5 carry_only lack it; rank4 ok)', tw['status'] == 'no_opening_mark'
              and int(tw['n_members_ok']) == 1 and int(tw['n_members_no_opening_mark']) == 2, f'({tw["n_members_ok"]}, {tw["n_members_no_opening_mark"]})')
        a = r['answer']
        check('no opening mark: rank1 carry 4 markets (unsupported), zero 5 (supported)', a['readings']['rank1']['carry']['n_markets_both'] == 4
              and a['readings']['rank1']['carry']['supported'] is False and a['readings']['rank1']['zero']['supported'] is True
              and a['statuses']['rank1']['carry'] == {'no_opening_mark': 1, 'ok': 4, 'no_interval': 1}, f'({a["statuses"]["rank1"]["carry"]})')
        check('no opening mark: 18 identity checks', r['receipt']['counts'].get('identity_checks') == 18)


def main():
    c0 = time.process_time()
    tmp = Path(tempfile.mkdtemp(prefix='mg_selftest_', dir=os.environ.get('MG_SELFTEST_TMP') or None))
    try:
        for nm, fn in (('boundary', t_boundary), ('readings', t_readings), ('support', t_support), ('paths', t_paths), ('hold', t_hold),
                       ('engine', t_engine), ('identity', t_identity), ('cross_view', t_cross_view), ('fifo', t_fifo), ('hand', t_hand),
                       ('our_market', t_our_market)):
            block(f'unit {nm}', fn)
        W = block('world: econ', lambda: econ_world(fv_rows(LEGS)))
        if W is not None:
            F, MM = W
            block('world: checks', chk_world, F, MM)
            P = block('e2e: world', world, tmp / 'w_main', F, MM)
            if P is not None:
                r = block('e2e: build', lambda: MG.build(P, write=True))
                if r is not None:
                    block('e2e: outputs', chk_main, r, P)
                    block('e2e: rehearsal', chk_rehearsal, P)
                    expect('refuse: write-once (a second run onto the finished output)', lambda: MG.build(P, write=True), 'write-once')
                    check('write-once: the finished output untouched, no partial', sorted(p.name for p in P['out'].iterdir()) ==
                          ['answer.json', 'closes_by_role.csv', 'per_market.csv', 'receipt.json'] and not list(P['out'].parent.glob('.matched_gap.partial_*')))
            block('refusals', refusals, tmp, F, MM)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res = dict(result='PASS' if not RES['fail'] else 'FAIL', checks=RES['n'], failed=len(RES['fail']), failures=RES['fail'],
               cpu_s=round(time.process_time() - c0, 2), vmhwm_mib=E.hwm())
    print(json.dumps(res, indent=1))
    sys.exit(0 if not RES['fail'] else 1)


if __name__ == '__main__':
    main()
