"""Bounded meaningful controls of the hypothesis instrument, before real measurement."""
import numpy as np
import build as B

r=np.zeros(300);r[15]=2;r[20]=-2
bid,ok=B.signal(r,'bid');ask,_=B.signal(r,'ask')
assert np.flatnonzero(bid).tolist()==[20] and not ask.any()
rev,_=B.signal(-r,'ask');assert np.array_equal(bid,rev)
saved=bid[:21].copy();future=r.copy();future[21:]=100;assert np.array_equal(B.signal(future,'bid')[0][:21],saved)
missing=r.copy();missing[18]=np.nan;assert not B.signal(missing,'bid')[0][20]
assert not B.signal(np.full(300,-2.),'bid')[0].any(), 'adverse is not reversal'
assert B.classify(22000,24000,0,[20],ok,0)[0]=='all_times_compatible'
assert B.classify(19000,21000,0,[20],ok,0)[0]=='possible_only'
assert B.classify(26000,27000,0,[20],ok,0)[0]=='no_match'
assert B.classify(26000,27000,0,[20],ok,1000)[0]=='possible_only'
assert B.classify(22000,24000,0,[20],ok,0,21000)[0]=='no_match'
assert B.classify(22000,24000,0,[20],ok,0,float('nan'))[0]=='post_bound_unknown'
assert B.classify(9000,10000,0,[20],ok,0)[0]=='context_or_boundary_unknown'
assert B.classify(22000,24000,0,[],ok,0)[0]=='no_match'
assert B.classify(21000,29000,0,[20,24],ok,0)[0]=='all_times_compatible'
assert B.classify(21000,31000,0,[20,26],ok,0)[0]=='possible_only'
g={('20261024',1,'bid'):{'events':bid,'valid':ok},('20261024',1,'ask'):{'events':ask,'valid':ok}}
base=dict(day='20261024',mid=1,maker='maker',order_hash='h',rank=1,side='bid',post_grade='G2_bound_constrained',post_hi=18000,lo=22000,hi=24000,ws=0)
cases=[dict(base,group='conditional_withdrawal'),dict(base,order_hash='e',group='exhausted'),dict(base,order_hash='u',group='unresolved')]
a=B.evaluate(cases,g)
assert len(a)==24 and all(not x['private_pull_observed'] for x in a)
assert all(x['status']=='unresolved_ending' for x in a if x['group']=='unresolved')
assert any(x['matched'] for x in a if x['group']=='exhausted'), 'matching BTC does not turn exhaustion into cancellation'
assert not any(x['matched'] for x in a if x['control']=='wrong_side')
rng=np.random.default_rng(B.SEED)
for _ in range(50):
 z=B.shift_events(bid,rng)
 assert z.sum()==bid.sum() and not z[:5].any()
 for lo,hi in [(5,60),(60,120),(120,180),(180,240),(240,300)]:assert z[lo:hi].sum()==bid[lo:hi].sum()
# The candidate and exactly-one-failed-component controls of the reading rule.
def fixture(rate=.9,wrong=.1,null=.2,n=20):
 ag=[];dr=[]
 for delay in (0,1000):
  for day in ['all','20261024','20261025','20261026','20261027']:
   for control,x in [('correct_side',rate),('wrong_side',wrong)]:ag.append(dict(group='conditional_withdrawal',delay_ms=delay,mode='post_compatible',control=control,day=day,evaluable=n,rate=x))
   for d in range(500):dr.append(dict(draw=d,delay_ms=delay,day=day,rate=null))
 return B.readings(ag,dr)['reading']
assert fixture()=='timing_alignment_enriched'
assert fixture(rate=.1)=='hypothesis_not_demonstrated'
assert fixture(wrong=.9)=='hypothesis_not_demonstrated'
assert fixture(null=.9)=='hypothesis_not_demonstrated'
assert fixture(n=2)=='insufficient_coverage'
import check as C
for row in a:
 case=next(c for c in cases if c['order_hash']==row['order_hash'])
 side=case['side'] if row['control']=='correct_side' else 'ask'
 st,hits=C.independent_status(case,g[(case['day'],case['mid'],side)],row['delay_ms'],row['mode'],row['control'])
 assert st==row['status'] and '|'.join(map(str,hits))==row['reversal_times_ms']
print('PASS: causal transition, wrong side, gaps, response/ordering/boundary, competing outcomes, null counts, positive/negative readings.')
