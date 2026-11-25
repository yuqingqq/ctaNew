"""Realized spread on FIFO-matched shares of the retained buffered straight-line arm.

Descriptive accounting only. Equal weighting per matched share. Execution mid is the
last valid book observation strictly before the fill on the venue clock (max age
1 s; never skip over an invalid observation). Split signed holding movement at
min(entry+1s,exit). Sensitivity uses observations at least 50 ms earlier than
each reference time, applying the same strict lookup. No new policy or fit.
"""
from collections import Counter, defaultdict, deque
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'smooth_unwind_20261125'))
import smooth as S

R=S.R


def book_mid(frames,clock,t):
    j=np.searchsorted(clock,t,'left')-1
    if j<0 or t-clock[j]>1000:
        return None
    bid,ask=frames[j,[2,12]]
    return (bid+ask)/2 if 0<=bid<ask<=100 else None


def inventory_pairs(fills):
    lots=deque()
    for f in fills:
        left=f[3]
        while left>1e-9 and lots and lots[0][0][2]!=f[2]:
            opened,remaining=lots[0]
            q=min(left,remaining)
            yield opened,f,q
            left-=q
            if remaining-q<1e-9:
                lots.popleft()
            else:
                lots[0]=(opened,remaining-q)
        if left>1e-9:
            lots.append((f,left))


def tranche(opened,closed,q,mid,lag):
    s=1-2*opened[2]
    a,b=opened[4],closed[4]
    net=s*(b-a)
    r=Counter(shares=q,net=q*net)
    r['good_sh' if net>0 else 'bad_sh' if net<0 else 'flat_sh']=q
    t0,t1=opened[0],closed[0]
    m0,m1,me=(mid(t-lag) for t in (t0,min(t0+1000,t1),t1))
    if m0 is None or me is None:
        r.update(missing_legs_sh=q,missing_legs_net=q*net)
        return r
    r.update(valued=q,opening=q*s*(m0-a),drift=q*s*(me-m0),closing=q*s*(b-me),valued_net=q*net)
    assert math.isclose(r['opening']+r['drift']+r['closing'],r['valued_net'],abs_tol=1e-8)
    if m1 is None:
        r['missing_split_sh']=q
    else:
        r.update(split_sh=q,early=q*s*(m1-m0),late=q*s*(me-m1),split_drift=q*s*(me-m0))
    return r


def fill_kind(f,trades):
    t,_,s,_,price,i=f
    tt,ss,p,_=trades[i]
    assert t==tt and s==ss
    if price==p:
        return 'same_price'
    assert (1-2*s)*(price-p)>0
    return 'through_price'


def one(path,info,ident,flags,model,payout,ref):
    b=S.Branch(path,info['end_rel'],ident)
    pol=S.Smooth(b,flags,model,'linear',{},[])
    for k in range(R.LB.n_ticks(b)):
        R.LB.step(b,k,pol)
    R.LB.finish(b)
    fills=b.E.fills
    fifo=R.X.fifo(fills,100*payout)
    _,pos,settle=b.E.value(100*payout)
    checks={'closed':fifo['matched_shares'],'turnover':fifo['completed_c']/100,'residual':fifo['residual_c']/100,
            'settle':settle/100,'end_pos':pos,'shares':sum(f[3] for f in fills),'posts':len(b.E.o),
            'all_pulls':sum(o['req'] is not None for o in b.E.o)}
    assert all(abs(v-ref[k])<1e-6 for k,v in checks.items()),ident
    frames=b.E.f; clock=frames[:,0]
    from functools import lru_cache
    @lru_cache(None)
    def mid(t):
        return book_mid(frames,clock,t)
    groups=defaultdict(Counter)
    for opened,closed,q in inventory_pairs(fills):
        kind=fill_kind(opened,b.E.tr)
        for lag in (0,50):
            values=tranche(opened,closed,q,mid,lag)
            groups[lag,'all'].update(values)
            groups[lag,kind].update(values)
    for lag in (0,50):
        total=groups[lag,'all']
        assert abs(total['shares']-ref['closed'])<1e-6 and abs(total['net']/100-ref['turnover'])<1e-6
        validate(total)
    return [dict(mid=ident['instrument_id'],lag_ms=lag,entry=kind,**v) for (lag,kind),v in groups.items()]


def validate(v):
    for result,components in [
        ('net',('missing_legs_net','valued_net')),('valued_net',('opening','drift','closing')),
        ('shares',('valued','missing_legs_sh')),('valued',('split_sh','missing_split_sh')),
        ('shares',('good_sh','flat_sh','bad_sh')),('split_drift',('early','late'))]:
        assert math.isclose(v[result],sum(v[x] for x in components),abs_tol=1e-5)


def run():
    day=R.EVAL
    ref=pd.read_csv(S.HERE/'markets.csv').query("arm=='linear'").set_index('mid')
    census=S.read(R.CENSUS)['rows']
    idx={r['mid']:r for r in census if r['day']==day and r['status']=='complete'}
    fit=S.read(R.HERE/'fit_v2.json')
    assert set(ref.index)==set(idx) and len(ref)==70 and R.sha(R.CENSUS)==fit['census_sha']
    src=R.MM.KP.Tape(); off=src.offset(day); edges=R.MM.TR.T.P.frozen_edges()
    reg=R.MM.registry(day).set_index('mid')
    rg=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
    rows=[]
    for hour,group in reg.groupby('hour'):
        ids=[m for m in group.index if m in idx]
        if not ids: continue
        for m,st,path,info in R.MM.worlds(day,ids,src,off,reg):
            assert st=='complete'
            ident=R.MM.ident_of(day,m,1000.*reg.loc[m,'ws'],info,path)
            assert ident['tape_id']==idx[m]['tape_id']
            flags=R.triggers(R.MM.make_cache(info['t_start'],info['end_rel']),edges)
            rows.extend(one(path,info,ident,flags,fit['cells'],R.VM.V.payout_of(rg.loc[m]),ref.loc[m]))
        print('reconciled hour',hour,flush=True)
    df=pd.DataFrame(rows).fillna(0)
    assert not df.duplicated(['mid','lag_ms','entry']).any()
    for lag in (0,50):
        g=df[(df.lag_ms==lag)&(df.entry=='all')]
        assert set(g.mid)==set(ref.index)
    df.to_csv(HERE/'markets.csv',mode='x',index=False)
    pooled=df.drop(columns='mid').groupby(['lag_ms','entry']).sum().reset_index()
    R.write(HERE/'result.json',dict(rows=pooled.to_dict('records'),markets=len(ref),status=dict(Counter(r['status'] for r in census if r['day']==day)),
            source=R.sha(Path(__file__)),reference=R.sha(S.HERE/'markets.csv'),market_hash=R.sha(HERE/'markets.csv')))


def test():
    f=lambda t,s,q,p:(t,0,s,q,p,0)
    # Long partially closed, crossed through flat, then the resulting short closed.
    fills=[f(0,0,10,48),f(400,1,3,51),f(1200,1,10,46),f(1500,0,3,41)]
    pairs=list(inventory_pairs(fills))
    assert [q for _,_,q in pairs]==[3,7,3]
    get={0:50,400:50,1000:47,1200:47,1500:42}.get
    g=Counter()
    for a,b,q in pairs:g.update(tranche(a,b,q,get,0))
    validate(g)
    assert g['net']==10
    assert (g['opening'],g['closing'],g['early'],g['late'])==(17,-1,-6,0)
    assert g['good_sh']==6 and g['bad_sh']==7
    no=tranche(*pairs[0],lambda t:None,0)
    assert no['missing_legs_sh']==3 and no['net']==9 and no['valued']==0
    m=np.zeros((2,22)); m[:,0]=[0,300];m[:,2]=[48,75];m[:,12]=[50,77]
    assert book_mid(m,m[:,0],300)==49 and book_mid(m,m[:,0],301)==76 and book_mid(m,m[:,0],1500) is None
    broken=g.copy();broken['opening']+=1
    try: validate(broken)
    except AssertionError: pass
    else: raise ValueError('spread corruption passed')
    print('synthetic accounting and midpoint checks passed')


if __name__=='__main__':
    {'run':run,'test':test}[sys.argv[1]]()
