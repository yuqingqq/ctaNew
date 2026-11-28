"""Independent checks of recorded rows and the local rule, including a staggered-receipt falsifier."""
import json
import numpy as np
import pandas as pd
from run import HERE, DATA, DAYS, KEYS
from local import build

def clock_check():
    f=np.full((7,22),-1.)
    f[:,0]=[0,20,150,200,310,370,540]
    f[:,1]=[0,50,200,270,360,450,600]
    for s in (0,1):
        a=2+10*s
        f[:,a]=[48+s*4]*7
        f[:,a+1]=[47+s*6]*7
        f[:,a+5]=[150,150,80,70,25,15,70]
        f[:,a+6]=[200,200,60,50,100,100,110]
    c=np.array([1,51,109,257,312,357,421,453,515,535,589,646])
    mask=np.array([0,2,1,2,1,2,1,0,2,3,0,3])
    grid=np.arange(0,700,50)
    pubs={48:[360],52:[270,600],47:[200],53:[450]}
    def scalar(t,s,p,tape,pub):
        ii=np.flatnonzero(c<=t)
        if not len(ii) or t-c[ii[-1]]>300 or not mask[ii[-1]]&(1<<s):
            return None
        if not any(t-300 < r <= t for r in pub[p]):
            return None
        depths=[]
        for at in [t-300,t]:
            ix=np.flatnonzero(tape[:,1]<=at)
            if not len(ix) or at-tape[ix[-1],1]>300:
                return None
            row=tape[ix[-1]]
            if not 0<=row[2]<row[12]<=100:
                return None
            o=2+10*s
            book=dict(zip(row[o:o+5],row[o+5:o+10]))
            depths.append(book.get(p,0))
        return depths[-1] if depths[-1]<depths[0] else None
    tab,_=build(f,0,700,c,mask,grid,pubs)
    expected={}
    for t in np.unique(np.r_[c,grid,f[:,1]]):
        for s in (0,1):
            for p in pubs:
                val=scalar(t,s,p,f,pubs)
                if val is not None:
                    expected[t,s,p]=val
    actual={(t,s,p):v for (t,s),d in tab.items() for p,v in d.items()}
    assert expected and expected==actual
    changed=dict(pubs)
    changed[52]=[200,540]  # venue times substituted for the delivered notices
    wrong,_=build(f,0,700,c,mask,grid,changed)
    assert wrong!=tab
    print('scalar clock fixture; confirmed rows',len(actual),'early-publication corruption detected')

if __name__=='__main__':
    clock_check()
    f=pd.read_csv(HERE/'paired_markets.csv',float_precision='round_trip')
    inactive=f[['grid_pulls','offgrid_pulls','waits']].sum(axis=1)==0
    for k in KEYS:
        np.testing.assert_allclose(f.loc[inactive,'alert_'+k],f.loc[inactive,'current_'+k],atol=1e-6,rtol=0)
    assert (f.offgrid_one_of_two<=f.offgrid_pulls).all()
    r=json.loads((HERE/'results.json').read_text())
    assert r['pooled']['confirmed']['n']==len(f)==218
    print('markets without an extra action',inactive.sum(),'with unchanged economics and orders')
    from run import BASE
    boundary=0
    for day in DAYS:
        mids=set(f.loc[f.day==int(day),'mid'])
        directory=BASE/'step1'/day
        reg=pd.read_parquet(directory/'registry.parquet',columns=['mid','ws']).set_index('mid')
        for file in sorted((directory/'books').glob('part_*.parquet')):
            books=pd.read_parquet(file,columns=['mid','t_ms','b1_p','a1_p'],filters=[('mid','in',list(mids))])
            elapsed=books.t_ms-books.mid.map(reg.ws)*1000
            boundary+=int((elapsed.between(-30000,298000)&(books.b1_p<books.a1_p)&
                           (books.b1_p>=0)&(books.a1_p<=100)&
                           ((books.b1_p==0)|(books.a1_p==100))).sum())
    print('source frames with an unquotable 0/100 best in eligible market windows',boundary)
