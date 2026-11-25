"""Passive expiry inventory schedule on the matched resting-budget replay."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'resting_inventory_20261125'
sys.path.insert(0, str(PARENT))
from policy import Branch, Policy, R
from experiment import read

OUT = R.X.DATA / 'early_unwind_20261125'
TIMES = (150, 120, 90, 60, 30, 10, 5)


def band(left_ms):
    return 200 * max(0, min(5, math.ceil(left_ms / 30000) - 1))


class EarlyUnwind(Policy):
    def limit(self, g):
        return band(self.b.ident['opening_ms'] + 300000 - self.b.ident['start_abs_ms'] - g)

    def review(self, g, s):
        # The inherited resting-order selection uses the policy cap. Engine hard cap stays 1,000.
        self.cap = self.limit(g)
        super().review(g,s)

    def slot(self, v, rung, price):
        s = v['side']
        signed = (1-2*s)*v['position']
        reserved = v['B'] if s==0 else v['A']
        available = math.floor(self.limit(v['t']) - signed - reserved + R.EPS)
        native_price = price if s==0 else 100-price
        if available<1 or available*native_price<100-R.EPS:
            return 'WAIT', None, 'expiry_room'
        action,q,reason = super().slot(v,rung,price)
        if action=='POST' and q>available:
            # Preserve the parent's proposal record with the actual submitted size.
            g,side,capacity,level,_,unit = self.proposals[-1]
            self.proposals[-1] = (g,side,capacity,level,max(0.,signed+reserved+available),unit)
            return 'POST', available, 'expiry_size'
        return action,q,reason


def simulate(path, info, ident, flags, model, payout, arm):
    b=Branch(path,info['end_rel'],ident)
    policy=(EarlyUnwind if arm=='scheduled' else Policy)(b,flags,model)
    both=0
    for k in range(R.LB.n_ticks(b)):
        R.LB.step(b,k,policy)
        g=k*R.BE.STEP
        both += all(any(not b.E.o[i]['refused'] and b.E.o[i]['act']<=g<b.E.o[i]['remove']
                        and b.E.o[i]['rem']>R.EPS for i in b.unclosed[s]) for s in (0,1))
    R.LB.finish(b)
    fills=b.E.fills
    ts=np.array([f[0] for f in fills])
    signed=np.array([(1-2*f[2])*f[3] for f in fills])
    pos=np.r_[0.,np.cumsum(signed)]
    expiry=ident['opening_ms']+300000-ident['start_abs_ms']
    assert abs(expiry-b.end-2000)<R.EPS
    stamps={f'abs_{sec}s':abs(pos[np.searchsorted(ts,expiry-sec*1000,side='right')]) for sec in TIMES}
    fifo=R.X.fifo(fills,100*payout)
    _,end,settle=b.E.value(100*payout)
    assert abs(pos[-1]-end)<1e-6 and abs(settle-fifo['completed_c']-fifo['residual_c'])<1e-5
    assert max(abs(pos))<=1000+R.EPS and len(policy.proposals)==len(b.E.o)
    assert all(it>=0 and b.E.o[i]['act']<=t<b.E.o[i]['remove'] for t,i,s,q,p,it in fills)
    return dict(settle=settle/100,turnover=fifo['completed_c']/100,residual=fifo['residual_c']/100,
                end_pos=end,peak=max(abs(pos)),shares=sum(abs(signed)),both=both/R.LB.n_ticks(b),
                posts=len(b.E.o),all_pulls=sum(o['req'] is not None for o in b.E.o),
                inventory_pulls=len(policy.trims),pressure_pulls=sum(c[7:]==('PULL','pressure') for c in b.cmd_log),
                abs_end=abs(end),**stamps)


def run():
    src=R.MM.KP.Tape()
    day=R.EVAL
    ref=pd.read_csv(PARENT/'markets.csv').query("arm=='new'").set_index('mid')
    result=read(R.HERE/'results_v2.json')
    fit=read(R.HERE/'fit_v2.json')
    assert result['fit_sha']==R.sha(R.HERE/'fit_v2.json') and fit['census_sha']==R.sha(R.CENSUS)
    idx={r['mid']:r for r in read(R.CENSUS)['rows'] if r['day']==day and r['status']=='complete'}
    assert set(idx)==set(ref.index) and len(ref)==70
    reg=R.MM.registry(day).set_index('mid')
    rg=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
    off=src.offset(day)
    edges=R.MM.TR.T.P.frozen_edges()
    OUT.mkdir(parents=True,exist_ok=True)
    for hour,group in reg.groupby('hour'):
        mids=[m for m in group.index if m in idx and not (OUT/f'{m}.json').exists()]
        if not mids: continue
        for m,st,path,info in R.MM.worlds(day,mids,src,off,reg):
            assert st=='complete' and idx[m]['tape_id']==R.BE.tape_id(path['frames'],path['trades'],path['mark'])
            ident=R.MM.ident_of(day,m,1000.*reg.loc[m,'ws'],info,path)
            flags=R.triggers(R.MM.make_cache(info['t_start'],info['end_rel']),edges)
            pair={a:simulate(path,info,ident,flags,fit['cells'],R.VM.V.payout_of(rg.loc[m]),a) for a in ('prior','scheduled')}
            for k in pair['prior'].keys() & set(ref.columns):
                assert abs(pair['prior'][k]-ref.loc[m,k])<1e-6,(m,k)
            R.write(OUT/f'{m}.json',dict(mid=m,pair=pair,source=R.sha(Path(__file__))))
        print('paired hour',hour,flush=True)


def score():
    prior=read(PARENT/'result.json')
    ref=pd.read_csv(PARENT/'markets.csv').query("arm=='new'").set_index('mid')
    rows=[]
    for f in sorted(OUT.glob('*.json')):
        d=read(f)
        assert d['source']==R.sha(Path(__file__)) and set(d['pair'])=={'prior','scheduled'}
        for a,row in d['pair'].items():
            if a=='prior':
                assert all(abs(row[k]-ref.loc[d['mid'],k])<1e-6 for k in row.keys() & set(ref.columns))
            rows.append(dict(mid=d['mid'],arm=a,**row))
    df=pd.DataFrame(rows)
    assert set(df.mid)==set(ref.index) and not df.duplicated(['mid','arm']).any()
    assert np.allclose(df.settle,df.turnover+df.residual)
    sums={}
    for a,g in df.groupby('arm'):
        sums[a]={k:float(g[k].sum()) for k in ('settle','turnover','residual','shares','posts','all_pulls','inventory_pulls','pressure_pulls')}
        sums[a].update({k:float(g[k].mean()) for k in ['both','abs_end']+[f'abs_{sec}s' for sec in TIMES]})
        sums[a]['median_peak']=float(g.peak.median())
        sums[a]['flat_at_end']=int((g.abs_end<1e-6).sum())
    delta=df.pivot(index='mid',columns='arm',values='settle').eval('scheduled-prior')
    df.to_csv(HERE/'markets.csv',index=False)
    R.write(HERE/'result.json',dict(n=len(delta),status=prior['status'],arms=sums,better=int((delta>1e-6).sum()),
           worse=int((delta< -1e-6).sum()),equal=int((abs(delta)<=1e-6).sum()),lower_inventory=sums['scheduled']['abs_end']<sums['prior']['abs_end'],
           improves=sums['scheduled']['settle']>sums['prior']['settle'],profitable=sums['scheduled']['settle']>0))
    print(read(HERE/'result.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['run','score'])
    (run if parser.parse_args().stage=='run' else score)()
