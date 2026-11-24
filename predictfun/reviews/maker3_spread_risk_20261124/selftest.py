import copy
import numpy as np
import pandas as pd
import core as C
import risk as K
def fill(i,s,p,mid,spread,q=10,fee=0):
 return dict(id=str(i),sid=str(i),mk_idx=0,role='maker',s=s,price=p,q=q,fee_c=fee,t=float(i),hi_ms=i*1000,lo_ms=i*1000-100,
  book={clock:dict(mid=mid,spread=spread) for clock in C.CLOCKS})
f=[fill(1,1,45,50,10),fill(2,-1,55,50,10)];cl,lots,v=C.fifo(f,50);assert v['net_c']==100 and cl[0]['by_clock']['pre']['drift_c']==0
assert cl[0]['by_clock']['pre']['kind']=='both_wide'
f2=[fill(1,1,49,50,2),fill(2,-1,61,60,2)];cl,_,v=C.fifo(f2,50);assert v['net_c']==120 and cl[0]['by_clock']['pre']['drift_c']==100 and cl[0]['by_clock']['pre']['kind']=='neither_wide'
f3=[fill(1,-1,55,50,10,q=15,fee=3),fill(2,1,45,50,10,q=10,fee=2)];cl,lots,v=C.fifo(f3,50);assert v['completed_c']==96 and v['position']==-5 and v['net_c']==120
bad=copy.deepcopy(f);bad[0]['book']['pre']['mid']=None;assert C.fifo(bad,50)[0][0]['by_clock']['pre']['kind']=='unknown'
states=K.basis_states(f,[999,1000,1999,2000]);assert states[0][0]==0 and states[1][0]==10 and states[2][0]==10 and states[3][0]==0
g=[[i,50+i*.5] for i in range(61)];z=C.trend(g);assert z[0]['move_c']==15 and z[0]['efficiency']==1 and z[0]['qualifies']
g[10][1]=None;assert all(v['start_s']>10 for v in C.trend(g))
# Raw side must be reclassified at the shifted destination (not outcome shifting).
p=pd.DataFrame(dict(day=['x']*300,mid=[1]*300,k=range(300)));q=np.r_[np.ones(150),-np.ones(150)];inds={('x',1):np.arange(300)};raw={('x',1):[(0,-1),(0,-1)]}
assert K.outcome(inds,raw,q).sum()==1 and K.outcome(inds,raw,q,[150]).sum()==0
assert np.roll(K.outcome(inds,raw,q),150).sum()!=K.outcome(inds,raw,q,[150]).sum()
code=np.array([0]*4+[1]*4);x=np.array([1,1,0,0]*2);d=K.design(code,2,x,np.ones(8,bool));y=x.astype(bool)
assert abs(y@d['weights']-1)<1e-9 and abs((1-y)@d['weights']+1)<1e-9
assert C.band(None)=='unknown' and C.band(4)=='2-4' and C.band(4.1)=='>4'
b=C.Book(pd.DataFrame(dict(upd_ms=[0,2000],recv_ns=[0,2e9],best_bid=[.49,.48],best_ask=[.51,.52],status=['ok','ok'])))
assert b.at(-1)['status']=='no_book' and b.at(1500)['status']=='stale' and b.at(1999)['frame']==0
assert abs(sum(b.occupancy(0,3000).values())-3)<1e-8
print('PASS FIFO spread vs drift, short/fee/residual, missing mark, basis clock, trend, raw-side-shift falsifier, matched RD, spread boundaries, stale/future books')
