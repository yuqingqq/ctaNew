"""Source-only engine tests. No market data, held-out days, or saved replays are read."""
import ast
import bisect
import collections
import functools
import heapq
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from predictfun.research import pf_view_check as view

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ('pf_step4_live.py', 'pf_step4_ladder.py')


@functools.lru_cache(maxsize=2)
def compiled(file):
    tree = ast.parse((ROOT / file).read_text())
    return compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)],
                              type_ignores=[]), str(ROOT / file), 'exec')


def engine(file):
    ns = dict(np=np, pd=pd, bisect=bisect, collections=collections, heapq=heapq, itertools=itertools,
              math=math, SIZE=5., V2=True, SPLIT=True, L_O=22., L_FILL=100., L_PL=40., L_CA=22.,
              L_ANS=44., LEGACY_INTENT=False, QACT=False, G2V={}, PRED={}, PRED250={}, PREDVB={},
              PREDVA={}, FILLPRIO='price', OWNCROSS='on', LEGORDER='price', REJRESET='on',
              POLICYVIEW='known', CK=SimpleNamespace(L_PENDCANCEL=95.),
              LINES={'zero': [0.] * 4, 'inf': [float('inf')] * 4, 'test': [0.] * 4})
    exec(compiled(file), ns)
    ns['reg'] = pd.DataFrame([dict(mid=1, ws=0., outcome='Up')]).set_index('mid')
    return ns


def frame(t, bid=50, ask=51, bs=100., az=100.):
    f = dict(t_ms=float(t))
    for s in 'ba':
        for i in range(1, 6):
            f[f'{s}{i}_p'] = -1
            f[f'{s}{i}_s'] = 0.
    f.update(b1_p=bid, b1_s=bs, a1_p=ask, a1_s=az)
    return f


def leg(t, side, pt, sh):
    return dict(t_ms=float(t), side=side, ptick=pt, shares=float(sh), how=1, exec_s=0)


def run(ns, trades=(), frames=None, known_times=None, rules=None, size=10., policy='JOIN'):
    ns['LT_BY'] = {1: pd.DataFrame(trades)} if trades else {}
    r = ns['run_window'](1, frames or [frame(-1)], known_times or [0.],
                         rules or {'XNA', 'G5', 'S1'}, policy, 'mid', size,
                         cap_state=dict(realised=0., halted=False))
    assert r['n_fills'] == len(r['fills'])
    assert np.isclose(sum(f[4] for f in r['fills']), r['shares'])
    assert np.isclose(sum(f[7] for f in r['fills']), r['pnl'])
    assert np.isclose(r['mtm_stop'] + r['carry_stop'], r['pnl'])
    assert all(0 < f[4] <= f[11] + 1e-9 for f in r['fills'])
    return r


def neutral_inputs(ns, side=0):
    ns['KB'] = np.arange(-2000., 4000., 100.)
    ns['MB'] = np.full(60, 100.)
    ns['BC'] = np.zeros(60, dtype=np.int8)
    ns['BCV'] = {}
    ns['_FLIP_CACHE'] = {}
    for si, key in enumerate(('PREDVB', 'PREDVA')):
        ns[key] = {1: (np.arange(0., 2001., 50.), np.full(41, .01 if si == side else -1.))}


def budget_case(ns, n=18, spacing=19., side=0, delay=100., amount=5.):
    neutral_inputs(ns, side)
    ns['L_FILL'] = delay
    fill_time = (n - 1) * spacing + 20.
    tk = np.r_[np.arange(n) * spacing, fill_time + np.array([5., 10., 15., 20.])]
    frames = [frame(t - 15., 50 + (i % 2), 80, 10., 10.) for i, t in enumerate(tk)]
    for f, src in zip(frames[-4:], np.linspace((n - 1) * spacing - 14., fill_time - 1., 4)):
        f['t_ms'] = src
    for f in frames:
        f.update(b2_p=f['b1_p'] - 1, b2_s=10., a2_p=81, a2_s=10.)
    if side:
        for f in frames:
            for i in range(1, 6):
                bp, ap = f[f'b{i}_p'], f[f'a{i}_p']
                bq, aq = f[f'b{i}_s'], f[f'a{i}_s']
                f[f'b{i}_p'], f[f'a{i}_p'] = (100 - ap if ap >= 0 else -1), (100 - bp if bp >= 0 else -1)
                f[f'b{i}_s'], f[f'a{i}_s'] = aq, bq
    rules = {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'V0', 'GS', 'XNA', 'XMIN1', 'XV', 'S60'}
    return frames, tk.tolist(), rules, leg(fill_time, side, 51 if side else 49, amount), fill_time + delay


def requests_before(result, boundary):
    return [tuple(q) for q in result['requests'] if q[0] < boundary]


@pytest.mark.parametrize('file', ENGINES)
def test_price_priority_and_falsifier(file):
    ns = engine(file)
    frames = [frame(-1, 49, 52, 0., 0.), frame(10, 50, 52, 0., 0.)]
    assert run(ns, [leg(60, 0, 48, 5)], frames, [0., 10.], size=5.)['fills'][0][3] == 50
    ns['FILLPRIO'] = 'old'
    assert run(ns, [leg(60, 0, 48, 5)], frames, [0., 10.], size=5.)['fills'][0][3] == 49


@pytest.mark.parametrize('file', ENGINES)
def test_atomic_sweep_and_controls(file):
    ns = engine(file)
    cases = ([leg(60, 0, 50, 100), leg(60, 0, 49, 5)],
             [leg(59, 0, 50, 100), leg(60, 0, 49, 5)], [leg(60, 0, 50, 105)],
             [leg(60, 1, 51, 100), leg(60, 1, 52, 5)])
    for trades in cases:
        assert run(ns, trades)['shares'] == 5.
    ns['LEGORDER'] = 'old'
    assert run(ns, cases[0])['shares'] == 10.


@pytest.mark.parametrize('file', ENGINES)
def test_postonly_retry_and_falsifier(file):
    ns = engine(file)
    frames = [frame(-1, 50, 52, 0., 0.), frame(30, 49, 50, 0., 0.), frame(60, 50, 52, 0., 0.)]
    assert run(ns, [leg(160, 0, 49, 5)], frames, [0., 100., 100.], size=5.)['n_fills'] == 1
    ns['REJRESET'] = 'off'
    assert run(ns, [leg(160, 0, 49, 5)], frames, [0., 100., 100.], size=5.)['n_fills'] == 0


@pytest.mark.parametrize('file', ENGINES)
def test_own_cross_and_falsifier(file):
    ns = engine(file)
    frames = [frame(-1, 49, 51, 10., 10.), frame(10, 50, 53, 10., 1.)]
    fixed = run(ns, (), frames, [0., 10.], size=5., policy='MICRO')
    assert fixed['live_ctr'].get('postonly_refused_own', 0) > 0
    assert fixed['live_ctr'].get('own_cross', 0) == 0
    ns['OWNCROSS'] = 'off'
    assert run(ns, (), frames, [0., 10.], size=5., policy='MICRO')['live_ctr'].get('own_cross', 0) > 0


@pytest.mark.parametrize('file', ENGINES)
@pytest.mark.parametrize('stale', [False, True])
def test_strict_guard(file, stale):
    ns = engine(file)
    if stale:
        for key in ('PREDVB', 'PREDVA'):
            ns[key] = {1: (np.array([0., 10.]), np.array([-1., -1.]))}
    r = run(ns, [leg(160, 0, 48, 5)], [frame(-1, 50, 52, 0., 0.)], [0.],
            {'XNA', 'G5', 'S1', 'V0', 'GS'}, 5.)
    assert r['n_fills'] == 0


def test_rung_knowledge_boundary_and_falsifier():
    ns = engine('pf_step4_ladder.py')
    neutral_inputs(ns)
    frames = [frame(-100, 50, 52, 10., 10.), frame(45, 49, 52, 10., 10.), frame(85, 49, 52, 10., 10.)]
    for f in frames:
        f.update(b2_p=f['b1_p'] - 1, b2_s=10., a2_p=53, a2_s=10.)
    rules = {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'V0', 'GS', 'LD2', 'XNA', 'XMIN1', 'XV', 'S60'}
    filled = run(ns, [leg(90, 0, 49, 15)], frames, [0., 60., 100.], rules, 'AV:5:25:0.06', 'MICRO')
    quiet = run(ns, (), frames, [0., 60., 100.], rules, 'AV:5:25:0.06', 'MICRO')
    assert filled['n_fills'] > 0
    assert requests_before(filled, 190.) == requests_before(quiet, 190.)
    ns['POLICYVIEW'] = 'physical'
    bad = run(ns, [leg(90, 0, 49, 15)], frames, [0., 60., 100.], rules, 'AV:5:25:0.06', 'MICRO')
    assert requests_before(bad, 190.) != requests_before(quiet, 190.)


@pytest.mark.parametrize('file', ENGINES)
def test_throttle_boundary_and_falsifier(file):
    ns = engine(file)
    frames, tk, rules, trade, cut = budget_case(ns)
    hit = run(ns, [trade], frames, tk, rules, 'AV:5:25:0.06', 'MICRO')
    quiet = run(ns, (), frames, tk, rules, 'AV:5:25:0.06', 'MICRO')
    assert hit['n_fills'] > 0
    assert requests_before(hit, cut) == requests_before(quiet, cut)
    ns['POLICYVIEW'] = 'physical'
    old_hit = run(ns, [trade], frames, tk, rules, 'AV:5:25:0.06', 'MICRO')
    old_quiet = run(ns, (), frames, tk, rules, 'AV:5:25:0.06', 'MICRO')
    assert requests_before(old_hit, cut) != requests_before(old_quiet, cut)


@pytest.mark.parametrize('file', ENGINES)
@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('delay', [20., 100., 1000.])
@pytest.mark.parametrize('n,spacing', [(12, 15.), (16, 19.), (18, 19.), (20, 25.)])
@pytest.mark.parametrize('amount', [1., 5.])
def test_paired_request_causality(file, side, delay, n, spacing, amount):
    ns = engine(file)
    frames, tk, rules, trade, cut = budget_case(ns, n, spacing, side, delay, amount)
    hit = run(ns, [trade], frames, tk, rules, 'AV:5:25:0.06', 'MICRO')
    quiet = run(ns, (), frames, tk, rules, 'AV:5:25:0.06', 'MICRO')
    assert requests_before(hit, cut) == requests_before(quiet, cut)


@pytest.mark.parametrize('file', ENGINES)
@pytest.mark.parametrize('cap', [5., 15., 100.])
@pytest.mark.parametrize('sequential', [False, True])
def test_policy_loss_cap_causality(file, cap, sequential):
    ns = engine(file)
    frames, tk, rules, trade, cut = budget_case(ns)
    rules = rules | {f'XCAP{cap:g}'} | ({'XS0'} if sequential else set())
    hit = run(ns, [trade], frames, tk, rules, 'AV:5:25:0.06', 'MICRO')
    quiet = run(ns, (), frames, tk, rules, 'AV:5:25:0.06', 'MICRO')
    assert requests_before(hit, cut) == requests_before(quiet, cut)


@pytest.mark.parametrize('side', [0, 1])
def test_live_ladder_parity_without_ladder_tokens(side):
    runs = []
    for file in ENGINES:
        ns = engine(file)
        frames, tk, rules, trade, cut = budget_case(ns, side=side)
        runs.append(run(ns, [trade], frames, tk, rules, 'AV:5:25:0.06', 'MICRO'))
    a, b = runs
    assert [tuple(q[:3]) for q in a['requests']] == [tuple(q[:3]) for q in b['requests']]
    assert [tuple(f[:13]) for f in a['fills']] == [tuple(f[:13]) for f in b['fills']]
    for key in ('pnl', 'net_end', 'shares', 'mtm_stop', 'carry_stop'):
        assert np.isclose(a[key], b[key])


@pytest.mark.parametrize('file', ENGINES)
def test_known_bad_split_amend_refused(file):
    with pytest.raises(SystemExit, match='REFUSED'):
        run(engine(file), rules={'G5', 'S1'})


def write_run(root, name, *, early=True, arm='arm', mid=1, lanes=True):
    base = root / 'step4' / name / 'synthetic'
    for part in ('fills', 'requests', 'win'):
        (base / part).mkdir(parents=True, exist_ok=True)
    f = pd.DataFrame([dict(mid=mid, lane=0, side=0, zombie=False, t=.05)]) if early else pd.DataFrame(columns=['mid', 'lane', 'side', 'zombie', 't'])
    q = pd.DataFrame([dict(mid=mid, lane=0, side=0, kind='place', t_ms=60.)])
    if not lanes:
        f = f.drop(columns='lane'); q = q.drop(columns='lane')
    w = pd.DataFrame([dict(mid=mid, ws=0.)])
    stem = view.arm_stem(arm)
    f.to_parquet(base / 'fills' / stem, index=False)
    q.to_parquet(base / 'requests' / stem, index=False)
    w.to_parquet(base / 'win' / stem, index=False)
    (base / 'summary.json').write_text(json.dumps(dict(day='synthetic', arms={arm: dict(fills=len(f), windows=1, live=dict(requests={'place': 1}))})))
    return base


@pytest.mark.parametrize('lanes', [True, False])
def test_checker_positive_and_bad_control(tmp_path, monkeypatch, lanes):
    monkeypatch.setattr(view.io, 'OUT', str(tmp_path))
    write_run(tmp_path, 'physical', lanes=lanes)
    write_run(tmp_path, 'known', early=False, lanes=lanes)
    write_run(tmp_path, 'bad', lanes=lanes)
    assert view.main(['physical', 'known', 'synthetic']) == 0
    assert view.main(['physical', 'bad', 'synthetic']) == 1


@pytest.mark.parametrize('damage', ['missing', 'empty', 'requests', 'win', 'extra', 'manifest', 'arm', 'cohort', 'time', 'window', 'lane', 'count'])
def test_checker_refuses_incomplete_or_wrong_artifact(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(view.io, 'OUT', str(tmp_path))
    write_run(tmp_path, 'physical')
    if damage == 'missing':
        assert view.main(['physical', 'absent', 'synthetic']) == 2
        return
    b = write_run(tmp_path, 'known', early=False, arm='other' if damage == 'arm' else 'arm', mid=2 if damage == 'cohort' else 1)
    if damage == 'empty':
        (b / 'summary.json').write_text('{"day":"synthetic","arms":{}}')
    elif damage in ('requests', 'win'):
        (b / damage / 'arm.parquet').unlink()
    elif damage == 'extra':
        (b / 'fills' / 'extra.parquet').write_bytes(b'not a parquet')
    elif damage == 'manifest':
        (b / 'summary.json').write_text('{"day":"wrong","arms":{}}')
    elif damage in ('time', 'window', 'lane'):
        p = b / 'requests' / 'arm.parquet'; q = pd.read_parquet(p)
        q.loc[0, {'time': 't_ms', 'window': 'mid', 'lane': 'lane'}[damage]] = {'time': np.nan, 'window': 99, 'lane': 9}[damage]
        q.to_parquet(p, index=False)
    elif damage == 'count':
        p = b / 'summary.json'; s = json.loads(p.read_text()); s['arms']['arm']['fills'] = 99; p.write_text(json.dumps(s))
    assert view.main(['physical', 'known', 'synthetic']) == 2


@pytest.mark.parametrize('delay', ['0', '-1', 'nan', 'inf'])
def test_checker_refuses_bad_delay(delay):
    assert view.main(['anything', 'anything', 'synthetic', delay]) == 2
