"""Score whole markets equally, with UTC days resampled jointly for uncertainty."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from diagnose import DAYS, HERE, load_day, metrics

OUT=HERE.parents[2]/'data/predict_fun/research/pressure_market_sharpe_20261128'
ARMS=('current','candidate')


def write(name,content):
    (HERE/name).write_text(json.dumps(content,indent=2,allow_nan=False)+'\n')


def history():
    reviews=HERE.parent
    path=reviews/'event_pressure_all_dates_20261127'
    receipt=json.loads((path/'result.json').read_text())
    panels=[]
    for day,digest in receipt['hashes'].items():
        p=path/f'paired_{day}.csv'
        if not p.exists():
            p=reviews/'event_pressure_cancel_20261127'/p.name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
        f=pd.read_csv(p,float_precision='round_trip')
        assert len(f)==receipt['coverage'][day]['complete'] and sum(receipt['coverage'][day].values())==288
        panels.append(f)
    earlier=pd.concat(panels)
    recent=pd.concat([load_day(d) for d in DAYS])
    combined=pd.concat([earlier,recent])
    assert not combined.duplicated(['day','mid']).any()
    result={}
    for block,f in [('previous_14',earlier),('recent_8',recent),('all_22',combined),
                    *[(str(d),p) for d,p in combined.groupby('day')]]:
        result[block]={a:metrics(f[a+'_settle']) for a in ('grid','event')}
    write('historical_sharpe.json',result)


def score(g):
    r={}
    for a in ARMS:
        x=metrics(g[a+'_settle'])
        x.update({k:float(g[a+'_'+k].sum()) for k in ('gross_gain','gross_loss','turnover','residual','closed','shares','posts','requests','age_share_seconds','abs_end','1s_net','1s_shares')})
        x['mean_abs_end']=x['abs_end']/len(g)
        x['hold_seconds']=x['age_share_seconds']/x['closed']
        x['mark_1s_c_per_share']=100*x['1s_net']/x['1s_shares']
        r[a]=x
    delta=g.candidate_settle-g.current_settle
    r.update(delta=float(delta.sum()),better_markets=int((delta>1e-7).sum()),worse_markets=int((delta< -1e-7).sum()))
    return r


def run():
    panels=[]
    lot_rows=[]
    hashes={}
    for d in DAYS:
        f=pd.read_csv(OUT/d/'comparison.csv',float_precision='round_trip')
        archive=load_day(d).set_index('mid').loc[f.mid]
        assert f.mid.is_unique and len(f)==len(load_day(d)) and list(f.tape_id)==list(archive.tape_id)
        for k in ('settle','turnover','residual','shares','closed','gross_gain','gross_loss','abs_end','age_share_seconds'):
            np.testing.assert_allclose(f['current_'+k],archive['event_'+k],rtol=0,atol=1e-6)
        l=pd.read_csv(OUT/d/'attribution.csv',float_precision='round_trip').fillna(0)
        for a in ARMS:
            sums=l[l.arm==a].groupby('mid').sum(numeric_only=True).reindex(f.mid,fill_value=0)
            np.testing.assert_allclose(sums.pnl,f[a+'_turnover'],rtol=0,atol=1e-6)
            np.testing.assert_allclose(sums.shares,f[a+'_closed'],rtol=0,atol=1e-6)
        l=l.merge(f[['mid','current_settle','candidate_settle']],on='mid',validate='many_to_one')
        l['losing_market']=np.where(l.arm=='candidate',l.candidate_settle,l.current_settle)<-1e-7
        panels.append(f)
        lot_rows.append(l.drop(columns=['mid','current_settle','candidate_settle']))
        hashes[d]={p:hashlib.sha256((OUT/d/p).read_bytes()).hexdigest() for p in ('attribution.csv','comparison.csv')}
    f=pd.concat(panels,ignore_index=True)
    assert not f.duplicated(['day','mid']).any()
    lots=pd.concat(lot_rows).fillna(0)
    lots.drop(columns='day').groupby(['arm','birth','sign','losing_market']).sum(numeric_only=True).to_csv(HERE/'fifo_attribution.csv')
    by_day={str(d):score(g) for d,g in f.groupby('day')}
    stages={'early':score(f[f.day<20261124]),'late':score(f[f.day>=20261124]),'pooled':score(f)}
    best=max(by_day,key=lambda d:by_day[d]['delta'])
    without=stages['pooled']['delta']-by_day[best]['delta']
    gates={k: v['delta']>0 and v['candidate']['sharpe']>v['current']['sharpe'] for k,v in stages.items()}
    gates.update(worst_decile=stages['pooled']['candidate']['worst_decile']>stages['pooled']['current']['worst_decile'],
                 without_best=without>0)
    rng=np.random.default_rng(20261128)
    sums=[]
    for d in DAYS:
        x=f.loc[f.day==int(d),[a+'_settle' for a in ARMS]].to_numpy()
        sums.append([np.repeat(len(x),2),x.sum(axis=0),(x*x).sum(axis=0)])
    n,s,ss=np.array(sums)[rng.integers(0,len(DAYS),size=(4096,len(DAYS)))].sum(axis=1).transpose(1,0,2)
    sr=(s/n)/np.sqrt((ss-s*s/n)/(n-1))
    ci=np.quantile(sr[:,1]-sr[:,0],[.025,.975])
    write('result.json',dict(stages=stages,by_day=by_day,hashes=hashes,advance=all(gates.values()),gates=gates,
          delta_ex_best=without,most_improved_day=best,sharpe_delta_day_bootstrap_95=ci.tolist(),
          vetoed_proposals=int(f.veto.sum()),clipped_proposals=int(f.clipped.sum())))
    cols=['day','mid','tape_id']
    for a in ARMS:
        cols += [a+'_'+k for k in ('settle','turnover','residual','shares','1s_net','1s_shares')]
    f[cols].to_csv(HERE/'paired_markets.csv',index=False)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='history':
        history()
    else:
        run()
