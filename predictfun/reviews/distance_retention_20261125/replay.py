"""One paired development ablation, KEEP when distance alone would pull. See protocol.md."""
import json
import math
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'churn_fill_timing_20261125'))
import audit as A
S,R,C=A.S,A.R,A.C


def keep_distance(flags):
    return flags['distance'] and not flags['no_book'] and not flags['cross']


class Branch(S.Branch):
    def __init__(self,path,end,ident,enabled):
        super().__init__(path,end,ident)
        self.keep_far=enabled
        self.retained={}

    def structural(self,g):
        if not self.keep_far:
            return super().structural(g)
        frame=self.E.f[self.E.known]
        for side,ids in self.unclosed.items():
            best,opp=frame[2+side*10],frame[2+(1-side)*10]
            for gid in ids:
                o=self.E.o[gid]
                if o['req'] is not None or g<=o['submit']:
                    continue
                behind=(1-2*side)*(best-o['price'])
                flags=dict(no_book=bool(best<0),distance=bool(behind>R.BE.HOLD+o['rung']),
                           cross=bool(o['price']>=opp if side==0 else o['price']<=opp))
                if keep_distance(flags):
                    self.retained.setdefault(gid,g)
                    continue
                if any(flags.values()):
                    self.E.request(gid,g,'structural')
                    self.struct_log.append(dict(t=float(g),gid=gid,side=side,rung=float(o['rung']),price=float(o['price']),
                      best=float(best),opp=float(opp),behind=float(behind),primary=next(k for k in R.LB.STRUCT_FLAGS if flags[k]),
                      mode=self.mode,effect=float(o['remove']),ack=float(o['remove']+R.BE.CANCEL_ACK),
                      received=float(frame[1]),frame=int(self.E.known),**flags))


def simulate(path,info,ident,on,model,payout,enabled):
    b=Branch(path,info['end_rel'],ident,enabled)
    pol=S.Smooth(b,on,model,'linear',{},[])
    both=0
    for k in range(R.LB.n_ticks(b)):
        R.LB.step(b,k,pol)
        g=k*R.BE.STEP
        both+=all(any(not b.E.o[i]['refused'] and b.E.o[i]['act']<=g<b.E.o[i]['remove'] and
                      b.E.o[i]['rem']>R.EPS for i in b.unclosed[s]) for s in (0,1))
    R.LB.finish(b)
    fills=b.E.fills
    fifo=R.X.fifo(fills,100*payout)
    _,pos,value=b.E.value(100*payout)
    reasons,flags,_=A.requests(b,pol)
    counts=Counter(reasons.values())
    assert not enabled or not any(v=='distance' for v in flags.values())
    r=dict(settle=value/100,residual=fifo['residual_c']/100,turnover=fifo['completed_c']/100,
           closed=fifo['matched_shares'],end_pos=pos,abs_end=abs(pos),shares=sum(f[3] for f in fills),
           both=both/R.LB.n_ticks(b),all_pulls=len(reasons),posts=len(b.E.o),
           pressure_pulls=counts['pressure'],inventory_pulls=counts['inventory'],
           distance_only=sum(v=='distance' for v in flags.values()),unique_retained=len(b.retained))
    for kind in ('same_price','through_price'):
        r[kind+'_usd']=r[kind+'_sh']=0.
    for f,z,q in C.inventory_pairs(fills):
        kind=C.fill_kind(f,b.E.tr)
        r[kind+'_sh']+=q
        r[kind+'_usd']+=(1-2*f[2])*(z[4]-f[4])*q/100
    validate(r)
    return r


def validate(r):
    for x,y in [(r['settle'],r['turnover']+r['residual']),(r['shares'],2*r['closed']+r['abs_end']),
                (r['closed'],r['same_price_sh']+r['through_price_sh']),(r['turnover'],r['same_price_usd']+r['through_price_usd'])]:
        assert math.isclose(x,y,abs_tol=1e-6)


def run():
    day=R.EVAL
    ref_file=S.HERE/'markets.csv'
    ref=pd.read_csv(ref_file).query("arm=='linear'").set_index('mid')
    prior=pd.read_csv(A.HERE/'requests.csv').query("flags=='distance'").set_index('mid')
    cref=pd.read_csv(C.HERE/'markets.csv').query("lag_ms==0 and entry!='all'").set_index(['mid','entry'])
    fit=S.read(R.HERE/'fit_v2.json')
    idx={r['mid']:r for r in S.read(R.CENSUS)['rows'] if r['day']==day and r['status']=='complete'}
    assert set(idx)==set(ref.index) and len(ref)==70 and R.sha(R.CENSUS)==fit['census_sha']
    src=R.MM.KP.Tape(); off=src.offset(day); edges=R.MM.TR.T.P.frozen_edges()
    reg=R.MM.registry(day).set_index('mid'); rg=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
    rows=[]
    for hour,g in reg.groupby('hour'):
        ids=[m for m in g.index if m in idx]
        if not ids: continue
        for m,st,path,info in R.MM.worlds(day,ids,src,off,reg):
            ident=R.MM.ident_of(day,m,1000.*reg.loc[m,'ws'],info,path)
            assert st=='complete' and ident['tape_id']==idx[m]['tape_id']
            flags=R.triggers(R.MM.make_cache(info['t_start'],info['end_rel']),edges)
            for arm in ('reference','retain'):
                r=simulate(path,info,ident,flags,fit['cells'],R.VM.V.payout_of(rg.loc[m]),arm=='retain')
                if arm=='reference':
                    assert all(abs(r[k]-ref.loc[m,k])<1e-6 for k in set(r)&set(ref.columns))
                    assert r['distance_only']==(prior.loc[m,'requests'] if m in prior.index else 0)
                    for kind in ('same_price','through_price'):
                        v=cref.loc[m,kind]
                        assert abs(v['net']/100-r[kind+'_usd'])<1e-6 and abs(v['shares']-r[kind+'_sh'])<1e-6
                rows.append(dict(mid=m,arm=arm,**r))
        print('paired hour',hour,flush=True)
    df=pd.DataFrame(rows)
    assert len(df)==140 and not df.duplicated(['mid','arm']).any() and set(df.mid)==set(idx)
    df.to_csv(HERE/'markets.csv',mode='x',index=False)
    agg={}
    for arm,g in df.groupby('arm'):
        agg[arm]={k:float(g[k].sum()) for k in g.select_dtypes('number') if k not in ('mid','end_pos','abs_end','both')}
        agg[arm].update(c_per_matched=100*g.turnover.sum()/g.closed.sum(),mean_abs_end=g.abs_end.mean(),both=g.both.mean())
    a,b=agg['reference'],agg['retain']
    delta=df.pivot(index='mid',columns='arm',values='settle').eval('retain-reference')
    competitive=(b['settle']>a['settle'] and b['turnover']>a['turnover'] and b['c_per_matched']>=a['c_per_matched']
                 and b['mean_abs_end']<=a['mean_abs_end']+10)
    reading='reject'
    if competitive:
        reading='positive_development' if min(b['settle'],b['turnover'])>0 else 'reduces_loss_only'
    R.write(HERE/'result.json',dict(arms=agg,reading=reading,better=int((delta>1e-6).sum()),
            worse=int((delta< -1e-6).sum()),equal=int((delta.abs()<=1e-6).sum()),delta=delta.sum(),
            delta_without_biggest_gain=delta.sum()-delta.max(),market_count=len(delta),
            complete_minutes=sum(r['end_rel'] for r in idx.values())/60000,source=R.sha(Path(__file__)),
            market_hash=R.sha(HERE/'markets.csv'),ref_hash=R.sha(ref_file)))


def test():
    from itertools import product
    for missing,distant,cross in product([False,True],repeat=3):
        assert keep_distance(dict(no_book=missing,distance=distant,cross=cross))==((missing,distant,cross)==(False,True,False))
    # A real structural callback on a two-sided received frame with a distant bid and crossing ask.
    from types import SimpleNamespace
    for enabled in (True,False):
        b=Branch.__new__(Branch); b.keep_far=enabled; b.retained={}; b.mode=None; b.struct_log=[]
        b.unclosed={0:[0],1:[1]}; frame=np.zeros((1,22)); frame[0,2]=50; frame[0,12]=51
        orders=[dict(price=p,submit=0,req=None,rung=0) for p in (45,49)]
        def request(i,t,why):
            orders[i].update(req=t,remove=t+37)
        b.E=SimpleNamespace(f=frame,known=0,o=orders,request=request)
        b.structural(100.)
        assert (orders[0]['req'] is None)==enabled and orders[1]['req']==100 and bool(b.retained)==enabled
    row=dict(settle=2,turnover=1,residual=1,shares=9,closed=4,abs_end=1,same_price_sh=3,through_price_sh=1,same_price_usd=2,through_price_usd=-1)
    validate(row)
    try: validate(dict(row,through_price_usd=0))
    except AssertionError: pass
    else: raise AssertionError('false turnover passed')
    print('all flag combinations, live branch callbacks, accounting and falsifier passed')


if __name__=='__main__':
    {'run':run,'test':test}[sys.argv[1]]()
