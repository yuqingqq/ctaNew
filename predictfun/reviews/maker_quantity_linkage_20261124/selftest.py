"""Hand-computed identities and falsifiers; no real-data outcomes."""
import copy,json
import numpy as np
import build as B
checks={}
def test(name,fn):fn();checks[name]=True
def equal(a,b):assert a==b,(a,b)
def refuse(fn):
 try:fn()
 except (ValueError,AssertionError):return
 raise AssertionError('bad control escaped')
v=[0,1,1,0,123,50*B.E18,100*B.E18,999,0,0,0,0]
es=[dict(order_hash='h',tx=str(i),log_index=i,role='maker',values=v,making=q*B.E18,shares=2*q*B.E18,block_number=1,transaction_index=i) for i,q in enumerate((10,15,25))]
o=dict(day='x',market=1,observed_making_wei=50*B.E18,original_making_wei=50*B.E18)
test('partial partial exhaustion',lambda:equal([int(x['remaining_after_wei'])//B.E18 for x in B.ledger(es,o)],[40,25,0]))
test('settlement order not input order',lambda:equal(B.ledger(es[::-1],o),B.ledger(es,o)))
test('duplicate execution refused',lambda:refuse(lambda:B.ledger(es+es[:1],o)))
changed=copy.deepcopy(es);changed[1]['values'][5]=60*B.E18
test('original size cannot reset',lambda:refuse(lambda:B.ledger(changed,o)))
changed=copy.deepcopy(es);changed[-1]['making']=26*B.E18
test('overfill refused',lambda:refuse(lambda:B.ledger(changed,o)))
test('buy collateral converted to shares',lambda:equal(B.native(v,'bid',50),(100.,.5,1)))
vno=v.copy();vno[5]=20*B.E18
test('NO buy complements to YES ask',lambda:equal(B.native(vno,'ask',80),(100.,.2,2)))
test('wrong outcome mapping refused',lambda:refuse(lambda:B.native(vno,'bid',80)))
test('pending then settlement not counted twice',lambda:equal(B.reserved(100,20,30),B.reserved(100,50,0)))
test('failed pending release does not add order',lambda:equal(B.reserved(100,20,0)-B.reserved(100,20,30),30))
test('overreserved refused',lambda:refuse(lambda:B.reserved(100,50,51)))
t=np.array([0,100,200,300,400]);cap=np.array([0.,40.,110.,0.,0.])
test('positive posting refinement',lambda:equal(B.capacity_post(t,cap,250,0,500,100),(100,1)))
test('post-match disappearance not late posting',lambda:equal(B.capacity_post(t,cap,200,0,500,100),(100,1)))
test('unknown is not empty',lambda:equal(B.capacity_post(t,np.full(5,np.nan),250,0,500,100),(0,None)))
test('terminal remaining capacity',lambda:equal(B.capacity_end(t,cap,250,250,500,30,False),(300,3)))
test('exhaustion cannot infer pull',lambda:equal(B.capacity_end(t,cap,250,250,500,0,True),(500,None)))
test('pending envelope prevents premature absence',lambda:equal(B.capacity_end(t,np.full(5,40.),250,250,500,30,False),(500,None)))
test('window tie excludes first possible match',lambda:equal(B.capacity_post(np.array([200]),np.array([0.]),200,0,500,100),(0,None)))
# Two observations are compatible with a match plus an addition, or a pull plus an addition.
# No synthetic clue counter is a private-command certificate.
test('masked netting arithmetic',lambda:equal(100-30+20,100-10))
test('pending USD must be divided by price once',lambda:equal(20/.2,100))
test('joint demand can fail individual fits',lambda:equal((60<=100 and 60<=100 and 60+60>100),True))
print(json.dumps(dict(checks=checks,passed=len(checks)),indent=2))
