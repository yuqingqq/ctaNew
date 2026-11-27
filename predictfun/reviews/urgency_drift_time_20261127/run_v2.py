"""Replay an expiry cushion driven by BTC distance from the opening mid."""
import json
import math
import runpy
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
previous = runpy.run_path(str(HERE.parent/'urgency_soft128_20261127'/'run.py'))
ref, common = previous['ref'], previous['taper_test']
R, U, parent = (ref[k] for k in ('R','U','prior'))
Base = parent.Veto
DAYS = ('20261024','20261025','20261027')
POLICIES = ('time_only','btc_time')


def curves(times, opening, z):
    left = (opening + 300000 - times)/1000
    w = np.clip((150-left)/120,0,1)
    proximity = np.where(np.isfinite(z),np.clip(1.5-np.abs(z),0,1),0.5)
    return 150-11*w,150-22*w*proximity


class ConditionalSize(Base):
    def __init__(self,*args,limits):
        super().__init__(*args)
        self.admission_curve = limits
        self.activity = Counter()

    def slot(self,view,rung,price):
        opened,first = self.opened,self.first_side
        proposal = super().slot(view,rung,price)
        action,size,_ = proposal
        if action!='POST':
            return proposal
        side = view['side']
        signed = (1-2*side)*view['position']
        reserved = view['A'] if side else view['B']
        available = math.floor(self.admission_curve[self.b.k]-signed-reserved+R.EPS)
        if size<=available:
            return proposal
        native_price = 100-price if side else price
        if available<1 or available*native_price<100-R.EPS:
            self.proposals.pop()
            self.opened,self.first_side=opened,first
            self.activity['wait']+=1
            return 'WAIT',None,'conditional_room'
        g,s,capacity,level,_,unit = self.proposals[-1]
        self.proposals[-1] = (g,s,capacity,level,max(0.,signed+reserved+available),unit)
        self.activity['clip']+=1
        return 'POST',available,'conditional_size'


def fixtures():
    prev = previous['fixture']
    prev()
    left = np.array([200,90,90,90,90,90,30])
    times = 300000-left*1000
    tm,bt = curves(times,0,np.array([0,0,1,2,-2,np.nan,0]))
    assert np.allclose(tm,[150,144.5,144.5,144.5,144.5,144.5,139])
    assert np.allclose(bt,[150,139,144.5,150,150,144.5,128])
    def post(pol,v,r,p):
        pol.opened,pol.first_side=1,v['side']
        pol.proposals.append((0,v['side'],80,0,150,.5))
        return 'POST',50,'post'
    for side in (0,1):
        for signed,reserve,idx,expected in [(80,20,1,39),(80,20,3,50),(-80,20,1,50),(120,8,6,0)]:
            obj = ConditionalSize.__new__(ConditionalSize)
            obj.opened,obj.first_side,obj.proposals,obj.activity=None,None,[],Counter()
            obj.admission_curve=bt
            obj.b=type('Branch',(),{'k':idx})()
            view=dict(side=side,position=(1-2*side)*signed,A=reserve if side else 80,B=80 if side else reserve)
            with patch.object(Base,'slot',post):
                act,q,_=obj.slot(view,0,50)
            assert (act,q)==(('POST',expected) if expected else ('WAIT',None))
            assert len(obj.proposals)==int(expected>0) and (obj.opened is None)==(expected==0)


def simulate(arm,args,limits):
    policies=[]
    def build(*policy_args):
        pol=ConditionalSize(*policy_args,limits=limits)
        policies.append(pol)
        return pol
    with patch.object(parent,'Veto',build):
        result=ref['simulate'](arm,*args)
    pol,=policies
    return dict(result,clip=pol.activity['clip'],wait=pol.activity['wait'])


def score():
    panels=[]
    baseline=[]
    by_date=[]
    for day in DAYS:
        base=common['baseline'](day)
        baseline.append(base.assign(policy='current'))
        rows=pd.read_csv(HERE/f'markets_{day}.csv',float_precision='round_trip',dtype={'day':str})
        for label in POLICIES:
            common['validate'](rows[rows.policy==label],base)
        by_date.append(rows)
    for (arm,policy),g in pd.concat(baseline+by_date).groupby(['arm','policy']):
        for day,h in [('pooled',g),*g.groupby('day')]:
            panels.append(dict(day=day,arm=arm,policy=policy,**common['metrics'](h)))
    scores=pd.DataFrame(panels)
    scores.to_csv(HERE/'comparison.csv',index=False)
    p=scores[scores.day=='pooled'].set_index(['arm','policy'])
    tests={a:dict(benchmark=common['decision'](p.loc[a,'current'],p.loc[a,'btc_time']),
                  beats_time=bool(p.loc[(a,'btc_time'),'settle']>p.loc[(a,'time_only'),'settle'])) for a in ref['ARMS']}
    x=dict(tests=tests,passes=all(v['benchmark']=='passes' and v['beats_time'] for v in tests.values()),
           sha256={f.name:R.sha(f) for f in sorted(HERE.glob('*.csv'))})
    (HERE/'result.json').write_text(json.dumps(x,indent=2)+'\n')
    print(p.to_string(),tests,flush=True)


def run():
    fixtures()
    btc_dir=R.X.DATA/'maker_trace_v2_20261123/btc'
    for line in (HERE/'btc.sha256').read_text().splitlines():
        sha,name=line.split()
        assert R.sha(btc_dir/Path(name).name)==sha
    fit=U.S.read(R.HERE/'fit_v2.json')['cells']
    bind=U.S.read(ref['S']['L']['E'].REPLICATION/'result.json')
    assert R.sha(R.HERE/'fit_v2.json')==bind['fit_sha'] and R.sha(R.CENSUS)==bind['census_sha']
    tape=R.MM.KP.Tape()
    edges=R.MM.TR.T.P.frozen_edges()
    for day in DAYS:
        base=common['baseline'](day)
        lookup=base.set_index(['mid','arm'])
        ids=set(base.mid)
        reg=R.MM.registry(day).set_index('mid')
        payout=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
        btc=R.BTC.load_btc(btc_dir,day,Counter())
        rows=[]
        for _,h in reg.groupby('hour'):
            batch=sorted(set(h.index)&ids)
            if not batch:
                continue
            for m,status,path,info in R.MM.worlds(day,batch,tape,tape.offset(day),reg):
                ws=1000.*reg.at[m,'ws']
                ident=R.MM.ident_of(day,m,ws,info,path)
                assert status=='complete' and ident['tape_id']==lookup.at[(m,'nominal'),'tape_id']
                flags,z=U.S.signal_data(btc,info,ws,edges)
                times=info['t_start']+R.BE.STEP*np.arange(len(z))
                limits=curves(times,ws,z)
                ramp=(times>ws+150000)&(times<ws+270000)
                args=(path,info,ident,flags,fit,R.VM.V.payout_of(payout.loc[m]))
                for arm in ref['ARMS']:
                    def parity(metrics):
                        for k,v in metrics.items():
                            if k in lookup:
                                assert abs(v-lookup.at[(m,arm),k])<1e-6,(m,arm,k)
                    if m==min(ids):
                        parity(ref['simulate'](arm,*args))
                    for policy,lim in zip(POLICIES,limits):
                        row=simulate(arm,args,lim)
                        if row['clip']+row['wait']==0:
                            parity(row)
                        rows.append(dict(day=day,mid=m,tape_id=ident['tape_id'],arm=arm,policy=policy,
                                         ramp_ticks=int(ramp.sum()),ramp_missing=int((ramp&~np.isfinite(z)).sum()),**row))
        frame=pd.DataFrame(rows)
        for policy in POLICIES:
            common['validate'](frame[frame.policy==policy],base)
        frame.to_csv(HERE/f'markets_{day}.csv',index=False)
        print(day,len(ids),'paired markets reconciled',flush=True)
    score()


if __name__=='__main__':
    run()
