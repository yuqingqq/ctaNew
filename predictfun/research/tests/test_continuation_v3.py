"""Regression for predecision trades with the same timestamp as the action."""
import ast
import inspect
import heapq
import pytest
from predictfun.research import pf_continuation_v3 as V
V.install()
from predictfun.research import pf_continuation_engine as E
from predictfun.research import pf_continuation_build as B
from predictfun.research.tests import test_continuation as T
T.source=E.source;T.install=B.install;T.equal_fills=B.equal_fills


def boundary_scenario():
    text=inspect.getsource(T.scenario).replace('leg(5160,0,49,2)','leg(5150,0,49,2)')
    ns=dict(T.__dict__);exec(compile(text,'coincident_trade_test','exec'),ns)
    return ns['scenario']()


def test_equal_clock_trade_is_prefix_not_future():
    manager,_,_=boundary_scenario();a=manager.paired[0]
    assert a['decision']==5150
    assert a['flow_2000']==3.  # Exclude2 at5150; the6500 trade is above the old bid.
    assert a['keep_2000_q']==3.


def test_pruned_heap_regression_retained():
    old=[x for x in [1,7,2,9,10,3,5,11,12,13,14,4,6] if x<8];new=list(old)
    assert [heapq.heappop(old) for _ in range(len(old))]!=list(range(1,8))
    heapq.heapify(new)
    assert [heapq.heappop(new) for _ in range(len(new))]==list(range(1,8))


def test_uncorrected_boundary_is_refused():
    old=V.builder_source().replace('actual=[f for f in rawfills if t<f[0]<t+h]', 'actual=[f for f in rawfills if t<=f[0]<t+h]')
    exec(compile(old,'planted_boundary_error','exec'),B.__dict__);T.install=B.install
    try:
        with pytest.raises(ValueError,match='KEEP fork'):boundary_scenario()
    finally:
        exec(compile(V.builder_source(),'restored_builder','exec'),B.__dict__);T.install=B.install

test_transform_and_aliases=T.test_transform_and_aliases
test_same_engine_forks_and_current_touch=T.test_same_engine_forks_and_current_touch
test_impossible_cancel_latency_is_caught=T.test_impossible_cancel_latency_is_caught
test_fill_reconciliation_refuses_corruption=T.test_fill_reconciliation_refuses_corruption
test_joint_profiles_calibration_and_refusals=T.test_joint_profiles_calibration_and_refusals
