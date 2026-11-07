"""Synthetic lifecycle/queue tests; no recorded market data is read."""
import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

from predictfun.research.pf_incremental_size import SizeOrder, plan_size
from predictfun.research.pf_ladder_fwd_build import FROZEN, SRC
from predictfun.research.pf_ladder_incremental_build import INVENTORY_PATCHES, PATCHES
from predictfun.research.tests.test_simulation_causality import (
    engine, frame, leg, run, neutral_inputs, requests_before, budget_case,
)


def order(slot, qty, price=50, sent=None, sent_to_venue=True):
    return SizeOrder(slot, price, qty, float(slot if sent is None else sent), sent_to_venue)


def incremental_engine():
    # Compile the exact generated functions without importing a replay driver.
    source = open(SRC).read()
    assert hashlib.sha256(source.encode()).hexdigest() == FROZEN
    for old, new in INVENTORY_PATCHES + PATCHES:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    actual = Path(SRC).with_name('pf_step4_ladder_incremental.py').read_text()
    assert actual.split('\n', 2)[2] == source
    tree = ast.parse(actual)
    ns = engine('pf_step4_ladder.py')
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)],
                            type_ignores=[]), 'pf_step4_ladder_incremental.py', 'exec'), ns)
    return ns


def scenario(side=0, changes=((0, 5), (100, 25)), trades=(), extra=(), incremental=True, frames=None, known_times=None):
    ns = incremental_engine()
    neutral_inputs(ns, side)
    times = np.unique(np.r_[np.arange(0., 1101., 50.), [t for t, _ in changes]])
    values = []
    for t in times:
        sz = [q for when, q in changes if when <= t][-1]
        values.append(-1. if sz == 0 else .01 if sz == 5 else .1)
    ns['PREDVB' if side == 0 else 'PREDVA'] = {1: (times, np.array(values))}
    rules = {'XNA', 'G5', 'S1', 'V0', 'GS'} | set(extra)
    if incremental:
        rules.add('INCSIZE')
    f = frame(-1, 50, 52, 0., 0.)
    f.update(b2_p=49, a2_p=53, b2_s=0., a2_s=0.)
    return run(ns, trades, frames or [f], known_times or [0.], rules, 'AV:5:25:0.06')


def test_planner_adds_delta_and_keeps_queue():
    base = order(0, 5)
    p = plan_size(50, 25, [base], [base])
    assert p.keep == (0,) and p.cancel == () and p.add == 20
    add = order(2, 20)
    p = plan_size(50, 5, [base, add], [base, add])
    assert p.keep == (0,) and p.cancel == (2,) and p.add == 0 and p.waiting


def test_planner_partial_fills_cancels_and_unsent_intentions():
    base = order(0, 3)
    assert plan_size(50, 25, [base], [base]).add == 22
    cancel_pending = order(2, 20)
    assert plan_size(50, 25, [base], [base, cancel_pending]).add == 2
    unsent = order(4, 22, sent_to_venue=False)
    p = plan_size(50, 25, [base, unsent], [base])
    assert p.add == 0 and p.keep == (0, 4)
    p = plan_size(50, 5, [base, unsent], [base])
    assert p.cancel == (4,) and p.add == 2 and not p.waiting


def test_planner_large_original_and_minimum():
    big = order(0, 25)
    p = plan_size(50, 5, [big], [big])
    assert p.cancel == (0,) and p.add == 0 and p.waiting
    assert plan_size(50, 5, [], []).add == 5
    p = plan_size(50, 5, [order(0, 4)], [order(0, 4)], minimum=2)
    assert p.keep == (0,) and p.add == 0 and p.below_minimum


def test_planner_cancels_newest_first_and_pulls_every_tranche():
    orders = [order(4, 5), order(0, 5), order(2, 5)]
    assert plan_size(50, 5, orders, orders).cancel == (4, 2)
    assert plan_size(None, 0, orders, orders).cancel == (4, 2, 0)
    assert plan_size(49, 15, orders, orders).cancel == (4, 2, 0)


@pytest.mark.parametrize('target,minimum', [(float('nan'), 0), (-1, 0), (5, -1), (5, float('inf'))])
def test_planner_refuses_invalid_quantities(target, minimum):
    with pytest.raises(ValueError):
        plan_size(50, target, [], [], minimum)


def test_planner_refuses_duplicate_slots():
    with pytest.raises(ValueError, match='duplicate'):
        plan_size(50, 25, [order(0, 5), order(0, 5)], [])


def test_planner_refuses_missing_exposure_reservation():
    with pytest.raises(ValueError, match='exposure reservations'):
        plan_size(50, 25, [order(0, 5)], [])
    with pytest.raises(ValueError, match='duplicate reserved'):
        plan_size(50, 25, [], [order(0, 5), order(0, 5)])


@pytest.mark.parametrize('side', [0, 1])
def test_increase_retains_original_and_new_order_is_only_delta(side):
    r = scenario(side, trades=[leg(150, side, 49 if side == 0 else 53, 25)])
    f = [x for x in r['fills'] if x[1] == .15]
    assert [x[4] for x in f] == [5, 20]
    assert [x[11] for x in f] == [5, 20]
    assert [x[15] for x in f] == [110, 10]  # original order's activation time survives
    assert not [x for x in r['requests'] if x[0] < 150 and x[2] == 'cancel']


@pytest.mark.parametrize('side', [0, 1])
def test_unchanged_target_never_trigger_control(side):
    inc = scenario(side, changes=((0, 5),))
    old = scenario(side, changes=((0, 5),), incremental=False)
    assert inc['requests'] == old['requests']
    assert inc['fills'] == old['fills']


@pytest.mark.parametrize('side', [0, 1])
def test_downsize_cancels_addition_and_baseline_is_positive_control(side):
    args = dict(side=side, changes=((0, 5), (100, 25), (200, 5)),
                trades=[leg(250, side, 49 if side == 0 else 53, 25)])
    inc, old = scenario(**args), scenario(**args, incremental=False)
    assert inc['fills'][0][4] == old['fills'][0][4] == 5
    assert inc['fills'][0][15] == 210
    assert old['fills'][0][15] == 10  # known-bad queue-preservation control: full replacement
    assert len([q for q in inc['requests'] if q[0] < 250 and q[2] == 'place']) == 2
    assert len([q for q in old['requests'] if q[0] < 250 and q[2] == 'place']) == 3


@pytest.mark.parametrize('side', [0, 1])
def test_partial_fill_is_not_topped_up_until_target_changes(side):
    trades = [leg(60, side, 49 if side == 0 else 53, 2), leg(250, side, 49 if side == 0 else 53, 25)]
    r = scenario(side, changes=((0, 5), (200, 25)), trades=trades)
    assert [x[0] for x in r['requests'] if x[2] == 'place' and x[0] < 250] == [0, 200]
    assert [x[11] for x in r['fills'] if x[1] == .25] == [5, 22]
    assert sum(x[4] for x in r['fills'] if x[1] == .25) == 25


@pytest.mark.parametrize('side', [0, 1])
def test_pending_create_counts_and_cancel_ack_releases_capacity(side):
    r = scenario(side, changes=((0, 5), (10, 25), (20, 5), (30, 25)))
    places = [x[0] for x in r['requests'] if x[2] == 'place' and x[0] < 200]
    assert places == [0, 10, 105]  # cancelled create stays reserved through send+95ms
    assert [x[0] for x in r['requests'] if x[2] == 'cancel' and x[0] < 200] == [20]


@pytest.mark.parametrize('side', [0, 1])
def test_downsize_indivisible_original_waits_for_cancel_ack(side):
    r = scenario(side, changes=((0, 25), (100, 5)))
    assert [x[0] for x in r['requests'] if x[2] == 'place' and x[0] < 200] == [0, 122]


@pytest.mark.parametrize('side', [0, 1])
def test_adverse_guard_pulls_all_tranches(side):
    r = scenario(side, changes=((0, 5), (100, 25), (200, 0)),
                 trades=[leg(250, side, 49 if side == 0 else 53, 100)])
    cancels = [x for x in r['requests'] if x[2] == 'cancel' and x[0] == 200]
    assert len(cancels) == 2
    assert cancels[0][3] > cancels[1][3]
    assert r['n_fills'] == 0


@pytest.mark.parametrize('side', [0, 1])
def test_newer_cancel_pending_order_does_not_jump_old_queue(side):
    r = scenario(side, changes=((0, 5), (100, 25), (150, 5)),
                 trades=[leg(160, side, 49 if side == 0 else 53, 5)])
    f = r['fills'][0]
    assert f[11] == 5 and f[15] == 120 and not f[12]


def test_additional_order_joins_behind_new_external_queue():
    frames = [frame(-1, 50, 52, 0., 0.), frame(100, 50, 52, 10., 0.)]
    r = scenario(trades=[leg(150, 0, 50, 5), leg(160, 0, 50, 10), leg(170, 0, 50, 1)],
                 frames=frames, known_times=[0., 100.])
    assert [(f[1], f[4], f[11]) for f in r['fills']] == [(.15, 5, 5), (.17, 1, 20)]


@pytest.mark.parametrize('side', [0, 1])
def test_ladder_tranches_share_side_budget(side):
    r = scenario(side, extra={'LD2', 'CAP10'}, trades=[leg(150, side, 48 if side == 0 else 54, 100)])
    assert sum(f[4] for f in r['fills'] if f[1] == .15) == 10
    assert {f[13] % 4 for f in r['fills'] if f[1] == .15} == {side, side + 2}


def test_minimum_notional_does_not_destroy_small_retained_order():
    r = scenario(changes=((0, 5), (200, 25), (300, 5)), extra={'XMIN1'},
                 trades=[leg(60, 0, 49, 1), leg(400, 0, 49, 100)])
    # Remaining base is four shares. The additional one share is below $1;
    # preserve the old four rather than cancelling it to make a five-share order.
    assert [f[4] for f in r['fills'] if f[1] == .4] == [4]
    assert r['live_ctr']['inc_below_minimum'] > 0


@pytest.mark.parametrize('side', [0, 1])
def test_share_cap_includes_all_tranches(side):
    r = scenario(side, extra={'CAP10'}, trades=[leg(150, side, 49 if side == 0 else 53, 25)])
    assert sum(x[4] for x in r['fills'] if x[1] == .15) == 10


@pytest.mark.parametrize('side', [0, 1])
def test_loss_budget_includes_original_and_addition(side):
    r = scenario(side, extra={'XCAP5'}, trades=[leg(150, side, 49 if side == 0 else 53, 25)])
    assert sum(x[4] for x in r['fills'] if x[1] == .15) == 5
    assert r['refused'] > 0


def test_reduce_only_reserves_existing_reducing_tranches():
    ns = incremental_engine()
    neutral_inputs(ns)
    times = np.arange(0., 2101., 50.)
    ns['PREDVB'] = {1: (times, np.where(times < 200, .1, -1.))}
    ns['PREDVA'] = {1: (times, np.where(times < 900, -1., np.where(times < 950, .01, .1)))}
    r = run(ns, [leg(60, 0, 49, 20), leg(1100, 1, 53, 100)], [frame(-1, 50, 52, 0., 0.)], [0.],
            {'INCSIZE', 'XNA', 'G5', 'S2', 'RO1', 'V0', 'GS'}, 'AV:5:25:0.06')
    assert r['net_end'] == 0
    assert [f[11] for f in r['fills'] if f[2] == 1] == [5, 15]
    assert [q[0] for q in r['requests'] if q[1] == 1 and q[2] == 'place' and q[0] >= 1000] == [1022]


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('delay', [20., 100., 1000.])
def test_incremental_requests_do_not_see_unnotified_fills(side, delay):
    ns = incremental_engine()
    frames, tk, rules, trade, cut = budget_case(ns, side=side, delay=delay)
    hit = run(ns, [trade], frames, tk, rules | {'INCSIZE'}, 'AV:5:25:0.06', 'MICRO')
    quiet = run(ns, (), frames, tk, rules | {'INCSIZE'}, 'AV:5:25:0.06', 'MICRO')
    assert hit['n_fills'] > 0
    assert requests_before(hit, cut) == requests_before(quiet, cut)


@pytest.mark.parametrize('rules', [{'XS0'}, {'HY10'}, {'XL100A'}])
def test_unsupported_lifecycle_modes_refused(rules):
    with pytest.raises(SystemExit, match='REFUSED: INCSIZE'):
        scenario(extra=rules)


@pytest.mark.parametrize('side', [0, 1])
def test_disabled_variant_equals_frozen_engine(side):
    out = []
    for ns in [engine('pf_step4_ladder.py'), incremental_engine()]:
        frames, tk, rules, trade, _ = budget_case(ns, side=side)
        out.append(run(ns, [trade], frames, tk, rules, 'AV:5:25:0.06', 'MICRO'))
    a, b = out
    assert a['requests'] == b['requests']
    assert a['fills'] == b['fills']
    for k in ('pnl', 'net_end', 'shares', 'mtm_stop', 'carry_stop', 'live_ctr', 'withheld'):
        assert a[k] == b[k]


def test_builder_refuses_wrong_frozen_source(tmp_path, monkeypatch):
    from predictfun.research import pf_ladder_fwd_build as builder
    wrong = tmp_path / 'source.py'
    wrong.write_text(Path(SRC).read_text() + '\n# known-bad source identity\n')
    monkeypatch.setattr(builder, 'SRC', str(wrong))
    with pytest.raises(SystemExit, match='REFUSED:.*not the frozen engine'):
        builder.generate([], str(tmp_path / 'out.py'), 'negative control')
    assert not (tmp_path / 'out.py').exists()
