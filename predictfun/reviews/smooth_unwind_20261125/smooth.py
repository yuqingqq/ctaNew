"""Paired smooth-unwind experiment. See protocol.md for the pre-outcome design."""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'early_unwind_20261125'))
import early_unwind as E

R, Branch = E.R, E.Branch
OUT = R.X.DATA/'smooth_unwind_20261125'
ARMS = ('stair','stair_buffer','linear','adaptive')
TRAIN_TIMES = range(150,39,-10)
read = E.read


def signal_data(btc,info,ws,edges):
    n=int(np.ceil(info['end_rel']/R.BE.STEP))
    t=info['t_start']+R.BE.STEP*np.arange(n)
    z=R.BTC.btc_feats(btc,np.floor(t-1000).astype(np.int64),np.full(n,int(ws)),np.full(n,int(ws)+300000))['z']
    z[t-1000<ws]=np.nan
    return R.triggers(R.MM.make_cache(info['t_start'],info['end_rel']),edges),z


def state_vector(left,s,z,flow,blocked):
    return [left/150, (1-2*s)*max(-3.,min(3.,float(z))) if math.isfinite(z) else 0.,
            min(3.,math.log1p(max(0.,flow)/220)),float(blocked)]


class Liquidate:
    """Offer a virtual 440-share holding on its exit side; no replenishment of stock."""
    def __init__(self,side,on,b):
        self.side,self.on,self.b=side,on,b

    def live(self,v,o):
        return ('PULL','pressure') if self.on[v['side']][self.b.k] else ('KEEP','keep')

    def slot(self,v,rung,price):
        s=v['side']
        if s!=self.side or self.on[s][self.b.k] or not (0<=v['t']-v['received']<=1000 and 0<=v['bid']<v['ask']<=100):
            return 'WAIT',None,'exit_unavailable'
        closed=(1-2*s)*v['position']
        reserved=v['A'] if s else v['B']
        q=math.floor(min(220,440-closed-reserved)+R.EPS)
        if q<=0 or q*(price if s==0 else 100-price)<100-R.EPS:
            return 'WAIT',None,'exit_min'
        return 'POST',q,'exit'


def probe(path,info,ident,on,k,exit_side):
    b=Branch(path,info['end_rel'],ident)
    policy=Liquidate(exit_side,on,b)
    stop=k+int(15000/R.BE.STEP)
    assert stop*R.BE.STEP < b.end
    for tick in range(k,stop):
        R.LB.step(b,tick,policy)
    b.E.advance(stop*R.BE.STEP)
    assert all(f[2]==exit_side and k*R.BE.STEP+58<=f[0]<=stop*R.BE.STEP for f in b.E.fills)
    qty=sum(f[3] for f in b.E.fills)
    assert qty<=440+R.EPS
    return qty


def training_rows(path,info,ident,flags,z,expected):
    b=Branch(path,info['end_rel'],ident)
    teacher=R.Policy(b,*flags,'B',{})
    expiry=ident['opening_ms']+300000-ident['start_abs_ms']
    anchors={int(math.ceil((expiry-left*1000)/R.BE.STEP)) for left in TRAIN_TIMES}
    assert len(anchors)==len(TRAIN_TIMES) and min(anchors)>=0
    rows=[]
    for k in range(R.LB.n_ticks(b)):
        R.LB.step(b,k,teacher)
        if k in anchors:
            left=(expiry-k*R.BE.STEP)/1000
            for s in (0,1):
                x=state_vector(left,s,z[k],b.flow(1-s,k*R.BE.STEP),flags[0][1-s][k])
                rows.append(dict(side=s,k=k,x=x,closed=probe(path,info,ident,flags[0],k,1-s)))
    R.LB.finish(b)
    assert abs(sum(f[3] for f in b.E.fills)-expected['shares'])<1e-5
    return rows


def fit():
    original=read(R.HERE/'fit_v2.json')
    tape_index=read(R.CENSUS)['rows']
    assert R.sha(R.CENSUS)==original['census_sha']
    src=R.MM.KP.Tape()
    edges=R.MM.TR.T.P.frozen_edges()
    all_rows=[]
    for day in R.TRAIN:
        reg=R.MM.registry(day).set_index('mid')
        btc=R.BTC.load_btc(R.X.DATA/'maker_trace_v2_20261123/btc',day,Counter())
        off=src.offset(day)
        index={r['mid']:r for r in tape_index if r['day']==day}
        for _,group in reg.groupby('hour'):
            mids=[m for m in group.index if f'{day}_{m}.json' in original['inputs']]
            if not mids: continue
            for m,st,path,info in R.MM.worlds(day,mids,src,off,reg):
                assert st=='complete' and index[m]['tape_id']==R.BE.tape_id(path['frames'],path['trades'],path['mark'])
                name=f'{day}_{m}.json'
                saved=R.OUT/'train'/name
                assert R.sha(saved)==original['inputs'][name]
                ws=1000.*reg.loc[m,'ws']
                ident=R.MM.ident_of(day,m,ws,info,path)
                flags,z=signal_data(btc,info,ws,edges)
                rows=training_rows(path,info,ident,flags,z,read(saved)['arms']['B']['metrics'])
                all_rows.extend(dict(day=day,mid=m,**row) for row in rows)
        print('fit probes',day,len(all_rows),flush=True)
    assert {f"{r['day']}_{r['mid']}.json" for r in all_rows}==set(original['inputs'])
    assert len(all_rows)==len(original['inputs'])*2*len(TRAIN_TIMES)
    models={}
    for side in (0,1):
        rows=[r for r in all_rows if r['side']==side]
        x=np.array([r['x'] for r in rows])
        y=np.log1p([r['closed']/15 for r in rows])
        mean=x.mean(axis=0); scale=np.maximum(x.std(axis=0),.1); xn=(x-mean)/scale
        w=np.linalg.solve(xn.T@xn+20*np.eye(x.shape[1]),xn.T@(y-y.mean()))
        models[str(side)]=dict(mean=mean.tolist(),scale=scale.tolist(),coef=w.tolist(),intercept=float(y.mean()),
                               probes=len(rows),zero=sum(r['closed']<=R.EPS for r in rows))
    R.write(OUT/'probe_rows.json',all_rows)
    R.write(HERE/'fit.json',dict(model=models,inputs=original['inputs'],raw_sha=R.sha(OUT/'probe_rows.json'),
                                  source=R.sha(Path(__file__))))


def curve(left,ratio=1.):
    u=min(1.,max(0.,(left-30)/120))
    return 1000*u*ratio/(1-u+u*ratio)


class Smooth(E.EarlyUnwind):
    def __init__(self,b,flags,old_fit,arm,models,z):
        super().__init__(b,flags,old_fit)
        self.arm,self.models,self.z=arm,models,z
        self.active_side=0
        self.ratio_stats=[0,0.,0.]  # side reviews during ramp, sum ratio, sum |ratio-1|
        self.tick_ratio={}

    def limit(self,g):
        left=(self.b.ident['opening_ms']+300000-self.b.ident['start_abs_ms']-g)/1000
        if self.arm.startswith('stair'):
            return E.band(left*1000)
        if self.arm=='linear' or left<=30 or left>=150:
            return curve(left)
        s=self.active_side
        if self.tick_ratio.get(s,(None,None))[0]!=g:
            m=self.models[str(s)]
            x=state_vector(left,s,self.z[self.b.k],self.b.flow(1-s,g),self.on[1-s][self.b.k])
            # Compare the same side at the same time, centering the three market inputs only.
            logratio=sum(m['coef'][j]*(x[j]-m['mean'][j])/m['scale'][j] for j in (1,2,3))
            ratio=math.exp(max(-math.log(2),min(math.log(2),logratio)))
            self.tick_ratio[s]=(g,ratio)
            self.ratio_stats[0]+=1
            self.ratio_stats[1]+=ratio
            self.ratio_stats[2]+=abs(ratio-1)
        return curve(left,self.tick_ratio[s][1])

    def review(self,g,s):
        self.active_side=s
        limit=self.limit(g)
        self.cap=min(self.b.cap,limit+(22 if self.arm!='stair' and limit>0 else 0))
        E.Policy.review(self,g,s)

    def slot(self,v,rung,price):
        self.active_side=v['side']
        return super().slot(v,rung,price)


def close_stats(fills):
    pos=0.; closed=0.; events=0
    for _,_,s,q,_,_ in fills:
        direction=1-2*s
        reducing=min(q,max(0.,-direction*pos))
        closed+=reducing
        events+=reducing>R.EPS
        pos+=direction*q
    return closed,events


def simulate(path,info,ident,flags,z,old_fit,models,payout,arm):
    b=Branch(path,info['end_rel'],ident)
    pol=Smooth(b,flags,old_fit,arm,models,z)
    both=0
    for k in range(R.LB.n_ticks(b)):
        R.LB.step(b,k,pol)
        g=k*R.BE.STEP
        both+=all(any(not b.E.o[i]['refused'] and b.E.o[i]['act']<=g<b.E.o[i]['remove'] and
                      b.E.o[i]['rem']>R.EPS for i in b.unclosed[s]) for s in (0,1))
    R.LB.finish(b)
    ts=np.array([f[0] for f in b.E.fills])
    pos=np.r_[0.,np.cumsum([(1-2*f[2])*f[3] for f in b.E.fills])]
    expiry=ident['opening_ms']+300000-ident['start_abs_ms']
    stamps={f'abs_{sec}s':abs(pos[np.searchsorted(ts,expiry-sec*1000,'right')]) for sec in E.TIMES}
    _,end,settle=b.E.value(100*payout)
    fifo=R.X.fifo(b.E.fills,100*payout)
    closed,events=close_stats(b.E.fills)
    assert abs(closed-fifo['matched_shares'])<1e-5 and abs(settle-fifo['completed_c']-fifo['residual_c'])<1e-5
    assert abs(end-pos[-1])<1e-6 and max(abs(pos))<=1000+R.EPS
    assert all(it>=0 and b.E.o[i]['act']<=t<b.E.o[i]['remove'] for t,i,s,q,p,it in b.E.fills)
    assert len(pol.proposals)==len(b.E.o)
    return dict(settle=settle/100,turnover=fifo['completed_c']/100,residual=fifo['residual_c']/100,
                closed=closed,close_events=events,shares=sum(f[3] for f in b.E.fills),end_pos=end,abs_end=abs(end),
                peak=max(abs(pos)),posts=len(b.E.o),inventory_pulls=len(pol.trims),both=both/R.LB.n_ticks(b),
                pressure_pulls=sum(c[7:]==('PULL','pressure') for c in b.cmd_log),all_pulls=sum(o['req'] is not None for o in b.E.o),
                ratio_n=pol.ratio_stats[0],ratio_abs=pol.ratio_stats[2],**stamps)


def evaluate():
    import subprocess
    f=HERE/'fit.json'
    assert subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(HERE.parents[2]))])==f.read_bytes()
    fit=read(f)
    assert fit['source']==R.sha(Path(__file__)) and fit['raw_sha']==R.sha(OUT/'probe_rows.json')
    src=R.MM.KP.Tape(); day=R.EVAL
    ref=pd.read_csv(E.HERE/'markets.csv').query("arm=='scheduled'").set_index('mid')
    old=read(R.HERE/'fit_v2.json')
    assert R.sha(R.CENSUS)==old['census_sha'] and fit['inputs']==old['inputs']
    idx={r['mid']:r for r in read(R.CENSUS)['rows'] if r['day']==day and r['status']=='complete'}
    assert set(idx)==set(ref.index) and len(ref)==70
    reg=R.MM.registry(day).set_index('mid')
    rg=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
    btc=R.BTC.load_btc(R.X.DATA/'maker_trace_v2_20261123/btc',day,Counter())
    off=src.offset(day); edges=R.MM.TR.T.P.frozen_edges()
    for hour,group in reg.groupby('hour'):
        mids=[m for m in group.index if m in idx and not (OUT/'eval'/f'{m}.json').exists()]
        if not mids: continue
        for m,st,path,info in R.MM.worlds(day,mids,src,off,reg):
            assert st=='complete' and idx[m]['tape_id']==R.BE.tape_id(path['frames'],path['trades'],path['mark'])
            ws=1000.*reg.loc[m,'ws']; ident=R.MM.ident_of(day,m,ws,info,path)
            flags,z=signal_data(btc,info,ws,edges)
            arms={a:simulate(path,info,ident,flags,z,old['cells'],fit['model'],R.VM.V.payout_of(rg.loc[m]),a) for a in ARMS}
            assert all(abs(arms['stair'][k]-ref.loc[m,k])<1e-6 for k in arms['stair'].keys()&set(ref.columns))
            R.write(OUT/'eval'/f'{m}.json',dict(mid=m,arms=arms,source=R.sha(Path(__file__)),fit_sha=R.sha(f)))
        print('paired hour',hour,flush=True)


def score():
    ref=pd.read_csv(E.HERE/'markets.csv').query("arm=='scheduled'").set_index('mid')
    rows=[]
    for path in sorted((OUT/'eval').glob('*.json')):
        d=read(path)
        assert set(d['arms'])==set(ARMS) and d['source']==R.sha(Path(__file__)) and d['fit_sha']==R.sha(HERE/'fit.json')
        rows.extend(dict(mid=d['mid'],arm=a,**v) for a,v in d['arms'].items())
    df=pd.DataFrame(rows)
    assert len(df)==len(ARMS)*len(ref) and not df.duplicated(['mid','arm']).any() and set(df.mid)==set(ref.index)
    assert np.allclose(2*df.closed+df.abs_end,df.shares) and np.allclose(df.turnover+df.residual,df.settle)
    sums={}
    for arm,g in df.groupby('arm'):
        r={k:float(g[k].sum()) for k in ('settle','turnover','residual','shares','closed','close_events','posts','inventory_pulls','all_pulls')}
        r.update({k:float(g[k].mean()) for k in ['abs_end','both']+[f'abs_{s}s' for s in E.TIMES]})
        r.update(cents_per_closed=100*r['turnover']/r['closed'],within_ten=int((g.abs_end<=10).sum()),
                 mean_abs_multiplier_change=float(g.ratio_abs.sum()/g.ratio_n.sum()) if g.ratio_n.sum() else None)
        sums[arm]=r
    comparisons={}
    for a,b in zip(ARMS[1:],ARMS[:-1]):
        diff=df.pivot(index='mid',columns='arm',values='settle')
        delta=diff[a]-diff[b]
        v,w=sums[a],sums[b]
        comparisons[a+'-'+b]=dict(delta=float(delta.sum()),better=int((delta>1e-6).sum()),worse=int((delta< -1e-6).sum()),
            equal=int((abs(delta)<=1e-6).sum()),comparable=v['abs_end']<=w['abs_end']+10,
            gate=v['closed']>w['closed'] and v['turnover']>w['turnover'] and v['settle']>w['settle'] and v['abs_end']<=w['abs_end']+10)
    df.to_csv(HERE/'markets.csv',index=False)
    R.write(HERE/'result.json',dict(n=len(ref),status=read(E.HERE/'result.json')['status'],arms=sums,comparisons=comparisons))
    print(comparisons,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['fit','evaluate','score'])
    {'fit':fit,'evaluate':evaluate,'score':score}[parser.parse_args().stage]()
