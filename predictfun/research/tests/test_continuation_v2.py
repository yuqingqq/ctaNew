"""Heap-pruning regression plus the full synthetic fork controls."""
import heapq
from predictfun.research import pf_continuation_v2 as V
V.install()
from predictfun.research import pf_continuation_engine as E
from predictfun.research.tests import test_continuation as T
T.source=E.source


def test_pruned_heap_requires_heapify():
    original=[1,7,2,9,10,3,5,11,12,13,14,4,6]
    assert all(original[(i-1)//2]<=original[i] for i in range(1,len(original)))
    broken=[x for x in original if x<8];good=list(broken)
    bad_order=[heapq.heappop(broken) for _ in range(len(broken))]
    assert bad_order!=sorted(bad_order)
    heapq.heapify(good)
    order=[heapq.heappop(good) for _ in range(len(good))]
    assert order==[1,2,3,4,5,6,7]

test_transform_and_aliases=T.test_transform_and_aliases
test_same_engine_forks_and_current_touch=T.test_same_engine_forks_and_current_touch
test_impossible_cancel_latency_is_caught=T.test_impossible_cancel_latency_is_caught
test_fill_reconciliation_refuses_corruption=T.test_fill_reconciliation_refuses_corruption
test_joint_profiles_calibration_and_refusals=T.test_joint_profiles_calibration_and_refusals
