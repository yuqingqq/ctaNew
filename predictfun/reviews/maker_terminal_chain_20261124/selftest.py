"""Material selection/accounting/state controls; no market data."""
import copy
import json
import build as B


def bad(fn,message):
    try:fn()
    except ValueError as e:assert message in str(e),(message,str(e))
    else:raise AssertionError('accepted '+message)


rows=[dict(day='20261024',mid='1',mk_hash='0x'+str(i).zfill(64),addr='0xaa',view='S_FIFO') for i in range(12)]
selected,cells=B.choose(rows,{'0xaa':1})
assert selected==B.choose(rows[::-1],{'0xaa':1})[0]
assert len(selected)==8 and len(cells)==12 and cells[0]['population']==12
assert B.choose(rows+[dict(rows[0],view='S_NOPRIO')],{'0xaa':1})[0]==selected
bad(lambda:B.choose(rows+[rows[0]],{'0xaa':1}),'duplicate order')
bad(lambda:B.choose([dict(rows[0],day='20261110')],{'0xaa':1}),'unconsumed day')
assert B.classify(10**20,10**20)=='exhausted_by_observed_fills'
assert B.classify(10**20,10**20-1)=='unresolved_remainder'
assert B.classify(10**20,10**20-1,True)=='explicit_onchain_cancel'
assert B.classify(10**20,10**20,True)=='exhausted_by_observed_fills'
bad(lambda:B.classify(10**20,10**20+1),'making budget')
enc=lambda c,r:'0x'+c.to_bytes(32,'big').hex()+r.to_bytes(32,'big').hex()
assert B.state_grade(enc(0,7),10,3)==('onchain_remaining_at_block_end',0,7)
assert B.state_grade(enc(0,2),10,3)==('onchain_remaining_at_block_end',0,2)
assert B.state_grade(enc(1,7),10,3)[0]=='closed_cause_not_identified'
assert B.state_grade(enc(1,0),10,10)[0]=='closed_cause_not_identified'
assert B.state_grade(None,10,3)[0]=='state_schema_unavailable'
bad(lambda:B.state_grade(enc(0,8),10,3),'state budget discrepancy')
bad(lambda:B.state_grade(enc(0,0),10,10),'positive control')
bad(lambda:B.state_grade(enc(2,0),10,3),'state budget discrepancy')
r=dict(successor='reprice',successor_hash='0xab',successor_before_rhi='True')
assert B.successor_grade(r,'exhausted_by_observed_fills')=='overlap_possible'
assert B.successor_grade(dict(r,successor_before_rhi='False'),'exhausted_by_observed_fills')=='later_order_after_exhaustion_command_unidentified'
assert B.successor_grade(r,'unresolved_remainder')=='overlap_possible'
assert B.successor_grade(dict(r,successor='none'),'unresolved_remainder')=='no_recorded_successor'
B.A.selftest()
print(json.dumps({'material_controls':'PASS','selection_permutation':True,'integer_precision':True,'historical_state':True,'successor_ambiguity':True}))
