"""Synthetic lifecycle checks for bounded paired-order quote retention."""
import ast
import json
import numpy as np
import pytest

from predictfun.research import pf_ladder_continuity_build as build
from predictfun.research.tests.test_ladder_pairs import pair_engine
from predictfun.research.tests.test_ladder_pairs_v3_arm import RULES, SIZE
from predictfun.research.tests.test_simulation_causality import frame, leg, run


def continuity_engine():
    source = build.generate()
    assert source == build.DST.read_text()
    ns = pair_engine()
    tree = ast.parse(source)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]), str(build.DST), 'exec'), ns)
    return ns


def case(side=0, changes=((0., .055), (100., .045), (150., .055)), trades=(), frames=None,
         known_times=None, enabled=True, parent=False, delay=10., stop=240, ro=60):
    ns = pair_engine() if parent else continuity_engine()
    ns['KB'] = np.arange(-2000., 302001., 50.)
    ns['MB'] = np.full(len(ns['KB']), 100.)
    ns['BC'] = np.zeros(len(ns['KB']), dtype=np.int8)
    ns['BCV'] = {}; ns['_FLIP_CACHE'] = {}; ns['L_FILL'] = delay
    stamps = np.unique(np.r_[np.arange(0., 300001., 50.), [t for t, _ in changes]])
    scores = np.array([[v for t0, v in changes if t0 <= t][-1] for t in stamps])
    ns['PREDVB'] = {1: (stamps, scores if side == 0 else np.full(len(stamps), -1.))}
    ns['PREDVA'] = {1: (stamps, scores if side == 1 else np.full(len(stamps), -1.))}
    if frames is None:
        f = frame(-1, 50, 52, 100., 100.)
        f.update(b2_p=49, a2_p=53, b2_s=100., a2_s=100.)
        frames = [f]
    if side:
        frames = [dict(f) for f in frames]
        for f in frames:
            for i in range(1, 6):
                bp, ap = f[f'b{i}_p'], f[f'a{i}_p']
                bs, az = f[f'b{i}_s'], f[f'a{i}_s']
                f[f'b{i}_p'], f[f'a{i}_p'] = (100-ap if ap >= 0 else -1), (100-bp if bp >= 0 else -1)
                f[f'b{i}_s'], f[f'a{i}_s'] = az, bs
    rules = (RULES - {'S240', 'RO60'}) | {'LZ', f'S{stop}', f'RO{ro}'}
    if enabled: rules.add('LC50')
    native, retention = [], []
    ns['_LZ_TRACE'] = native.append; ns['_LC_TRACE'] = retention.append
    legs = [leg(t, side, 100-price if side else price, qty) for t, price, qty in trades]
    result = run(ns, legs, frames, known_times or [0.], rules, SIZE, policy='MICRO')
    return result, native, retention


def touching(events, side, kind):
    return [e for e in events if e['kind'] == kind and e['lane'] % 4 == side]


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('trades', [(), ((60., 45, 2.),), ((110., 45, 5.),)])
def test_no_token_exact_parent_outputs(side, trades):
    a, at, _ = case(side, trades=trades, enabled=False)
    b, bt, _ = case(side, trades=trades, enabled=False, parent=True)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert at == bt


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('reopen', [140., 150., 151.])
def test_deadline_order_and_no_extension(side, reopen):
    _, events, holds = case(side, changes=((0., .055), (100., .045), (reopen, .055)))
    cancelled = [e['t'] for e in touching(events, side, 'cancel') if e['t'] <= 151]
    assert cancelled == ([] if reopen <= 150 else [150.])
    opened = next(h for h in holds if h['kind'] == 'open')
    closed = next(h for h in holds if h['kind'] == 'close')
    assert opened['deadline'] == 150. and closed['t'] == min(reopen, 150.)


@pytest.mark.parametrize('side', [0, 1])
def test_partial_fills_never_replenished_while_off_or_same_target_reopens(side):
    _, events, holds = case(side, trades=((60., 45, 2.), (120., 45, 1.)))
    assert [(e['t'], e['size']) for e in touching(events, side, 'place') if e['t'] < 200] == [(0., 5.)]
    assert next(h for h in holds if h['kind'] == 'open')['remaining'] == 3.


@pytest.mark.parametrize('side', [0, 1])
def test_pair_shrink_retains_partial_front_without_topup(side):
    _, events, holds = case(side, changes=((0., .055), (50., .1), (100., .045), (151., .055)), trades=((80., 45, 2.),))
    assert [(e['t'], e['size']) for e in touching(events, side, 'place') if e['t'] < 150] == [(0., 5.), (50., 20.)]
    opened = next(h for h in holds if h['kind'] == 'open')
    assert opened['total'] == 3. and opened['remaining'] == 3.
    assert any(e['kind'] == 'cancel' and e['lane'] == side+4 and e['t'] == 100 for e in events)


@pytest.mark.parametrize('side', [0, 1])
def test_full_fill_during_grace_never_reposts_while_off(side):
    _, events, holds = case(side, changes=((0., .055), (100., .045), (200., .055)), trades=((110., 45, 5.),))
    assert not [e for e in touching(events, side, 'place') if 100 <= e['t'] < 200]
    assert next(h for h in holds if h['kind'] == 'close')['t'] == 120.


@pytest.mark.parametrize('side', [0, 1])
def test_lower_line_probe_cancels_without_original_decision(side):
    _, events, holds = case(side, changes=((0., .055), (100., .045), (125., .02), (200., .055)))
    assert next(h for h in holds if h['kind'] == 'close')['reason'] == 'lower_line'
    assert [e['t'] for e in touching(events, side, 'cancel') if e['t'] <= 150] == [125.]
    assert not [e for e in events if e['kind'] == 'D' and e['t'] == 125.]


def late_cap_case(side=0):
    frames = []
    for t, bid in [(-1., 51), (9., 48)]:
        # Keep the old51-tick create post-only eligible at its40ms activation.
        # Its pending cancellation lasts to95ms, so a60ms fill is observable120ms.
        f = frame(t, bid, 53, 100., 100.)
        f.update(b2_p=bid-1, a2_p=54, b2_s=100., a2_s=100.)
        frames.append(f)
    return case(side, changes=((0., .1), (10., .055), (100., .045), (200., .055)),
                trades=((60., 49, 10.),), frames=frames, known_times=[0., 10.], delay=60.)


@pytest.mark.parametrize('side', [0, 1])
def test_cancelled_order_fill_hits_cap_and_preempts_hold(side):
    _, events, holds = late_cap_case(side)
    assert next(h for h in holds if h['kind'] == 'open')['t'] == 100.
    closed = next(h for h in holds if h['kind'] == 'close')
    assert closed['t'] == 120. and closed['reason'] == 'price_size_or_hard_guard'
    assert any(e['t'] == 120. for e in touching(events, side, 'cancel'))
    assert not [e for e in events if e['kind'] == 'D' and e['t'] == 120.]


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('stop,ro,boundary', [(240, 60, 60000.), (1, 60, 1000.)])
def test_phase_boundary_preempts_hold(side, stop, ro, boundary):
    _, events, holds = case(side, changes=((0., -1.), (boundary-100., .055), (boundary-25., .045)), stop=stop, ro=ro)
    assert next(h for h in holds if h['kind'] == 'close')['t'] == boundary
    assert any(e['t'] == boundary for e in touching(events, side, 'cancel'))


@pytest.mark.parametrize('side', [0, 1])
def test_pending_order_is_not_eligible(side):
    _, _, holds = case(side, changes=((0., .055), (20., .045)))
    assert not [h for h in holds if h['kind'] == 'open']


@pytest.mark.parametrize('side', [0, 1])
def test_stale_timer_does_not_end_later_hold(side):
    _, _, holds = case(side, changes=((0., .055), (100., .045), (120., .055), (130., .045), (200., .055)))
    assert [h['t'] for h in holds if h['kind'] == 'close'] == [120., 180.]


@pytest.mark.parametrize('side', [0, 1])
def test_price_move_preempts_hold(side):
    frames = []
    for t, bid in [(-1., 50), (124., 49)]:
        f = frame(t, bid, bid+2, 100., 100.)
        f.update(b2_p=bid-1, a2_p=bid+3, b2_s=100., a2_s=100.)
        frames.append(f)
    _, events, holds = case(side, frames=frames, known_times=[0., 125.])
    assert next(h for h in holds if h['kind'] == 'close')['t'] == 125.
    assert any(e['t'] == 125. for e in touching(events, side, 'cancel'))


def test_positive_without_token_and_unsupported_configuration_refusal():
    _, a, holds = case()
    _, b, _ = case(enabled=False)
    assert holds and a != b
    ns = continuity_engine()
    with pytest.raises(SystemExit, match='LC50 requires'):
        run(ns, rules={'LC50', 'LZ', 'XNA'})
