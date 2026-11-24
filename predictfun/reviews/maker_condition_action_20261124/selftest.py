"""Small tests for descriptive joint counts, timing and common-support matching."""
import numpy as np
import pandas as pd
import build as B
import check as C

def raises(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('known-bad input accepted')

def main():
    rows=[]
    for mid,burst,ys in [(1,'0',[0,0]),(1,'>=3',[1,0]),(2,'0',[1,1]),(2,'>=3',[1,1]),(3,'>=3',[1])]:
        for y in ys: rows.append(dict(day='20261024',mid=mid,rank=1,k=len(rows),inventory='(0,50]',btc='flat',time_left='(60,240]',prior_fills=burst,q_nonzero=True,O1=y,O2=0,O3=y,O4=y))
    x=pd.DataFrame(rows);j=B.joint(x);s=B.matched(x)
    C.recount_cells(x,j);C.recount_matched(x,s)
    r=s[(s.day=='all')&(s.outcome=='O1')].iloc[0]
    assert r.exposed_all==5 and r.exposed_common==4 and r.strata==2
    assert abs(r.rd-.25)<1e-12
    assert B.matched(x.sample(frac=1,random_state=17)).equals(s)
    bad=j.copy();bad.loc[0,'O1']+=1;raises(lambda:C.recount_cells(x,bad))
    bad=s.copy();bad.loc[0,'rd']+=.1;raises(lambda:C.recount_matched(x,bad))
    # State fields use only precomputed prior state; modifying a next-second outcome cannot change them.
    n=3;p={k:np.zeros(n) for k in B.TL.MS_FIELDS+B.TL.PS_FIELDS}
    for d in ['0','1']:
        p['q'+d]=np.array([10.,-10.,0.]);p['r_5s'+d]=np.array([2.,2.,2.]);p['tl'+d]=np.array([30.,60.,240.])
        p['bok'+d]=p['kok'+d]=np.ones(n,bool)
    p['qp']=np.array([10.,-10.,0.]);p['fl0']=np.array([0,2,3]);p['fl1']=np.array([3,0,0]);p['flp']=p['fl0']
    p.update(day=['20261024']*n,mid=[1]*n,rank=[1]*n,pair=[0]*n,k=np.arange(n),t_ms=np.arange(n)*1000,clean=[True]*n,O3=[0]*n,O4_reduces_all=[0]*n)
    for v in B.TL.VARIANTS:p['O1_'+v]=[0]*n;p['O2_'+v]=[0]*n
    p=pd.DataFrame(p);a=B.state_frame(p,'D0','full');b=B.state_frame(p.assign(O3=[1]*n),'D0','full')
    assert a[B.STATES].equals(b[B.STATES]);assert a.btc.tolist()==['with','against','flat']
    assert a.prior_fills.tolist()==['0','1-2','>=3']
    assert B.state_frame(p,'D1000','full').prior_fills.tolist()==['>=3','0','0']
    raises(lambda:B.state_frame(p.assign(q0=np.nan),'D0','full'))
    print('PASS: exact cells, common support, missing stratum, shuffle invariance, falsifiers, pre-state isolation, lag, flat/unknown state')

if __name__=='__main__': main()
