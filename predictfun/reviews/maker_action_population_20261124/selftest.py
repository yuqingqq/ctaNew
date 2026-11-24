import common as B
from map_conditions import indices,category
import numpy as np
assert B.budget(100,100,True,{'maker'})=='exhausted'
assert B.budget(100,30,True,{'maker'})=='positive_remainder_not_a_pull'
assert B.budget(100,30,False,{'maker'})=='incomplete_recorded_history'
assert B.budget(100,30,True,{'maker','taker'})=='mixed_or_taker_role'
try:B.budget(100,101,True,{'maker'});raise RuntimeError('overspend accepted')
except AssertionError:pass
assert B.relation(10,20,21,30,True)=='sequential_after_under_bounds'
assert B.relation(10,20,20,30,True)=='ordering_ambiguous'
assert B.relation(10,20,1,9,True)=='successor_before_ending_possible_overlap'
assert B.relation('','',1,9,False)=='unavailable'
assert indices(999,2000,0,300000,True)==[0,1,2]
assert indices(-1,2000,0,300000,True)==[] and indices(100,300000,0,300000,True)==[]
assert indices(100,200,0,300000,False)==[]
assert category(np.array([float('nan')]),str)=='unknown'
assert category(np.array([-2,2]),lambda x:'against' if x<=-1 else 'with')=='against|with'
print('14 positive / refusal / clock groups passed')
