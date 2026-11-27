"""Earlier, larger near-strike cap; same market paths and eligibility as the modest pilot."""
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'urgency_drift_time_20261127'
pilot = runpy.run_path(str(PRIOR / 'run_v2.py'))
ref, common = pilot['ref'], pilot['common']
R, U = pilot['R'], pilot['U']
DAYS, POLICIES = pilot['DAYS'], pilot['POLICIES']


def curves(left, z):
    far = np.minimum(150., 1000*np.clip((left-30)/120,0,1))
    near = np.interp(left,[30.,90.,210.],[0.,100.,150.])
    proximity = np.where(np.isfinite(z),np.clip(1.5-np.abs(z),0,1),0.5)
    return far + 0.5*(near-far), far + proximity*(near-far)


def fixture():
    pilot['fixtures']()
    left = np.array([240,210,150,90,60,48,42,39,30,5])
    far = [150,150,150,150,150,150,100,75,0,0]
    near = [150,150,125,100,50,30,20,15,0,0]
    mid,close = curves(left,np.zeros(len(left)))
    assert np.allclose(close,near) and np.allclose(mid,(np.array(far)+near)/2)
    for z,expected in ((1.5,far),(-1.5,far),(np.nan,mid),(1,mid),(-.5,near)):
        assert np.allclose(curves(left,np.full(len(left),z))[1],expected)
    assert not np.allclose(curves(left,np.zeros(len(left)))[1],far)
    # The generic admission overlay gives every attributable closing share room at any limit >= 0.
    for reserved in (0,20,50):
        assert max(0,80-reserved)<=int(0-(-80)-reserved)


def read_prior(day):
    path = PRIOR/f'markets_{day}.csv'
    assert R.sha(path)==json.loads((PRIOR/'result.json').read_text())['sha256'][path.name]
    return pd.read_csv(path,float_precision='round_trip',dtype={'day':str})


def summarize():
    frames = []
    for day in DAYS:
        base = common['baseline'](day)
        saved = pd.read_csv(HERE/f'markets_{day}.csv',float_precision='round_trip',dtype={'day':str})
        mild = read_prior(day).query("policy=='btc_time'").copy()
        for group in (mild, *(saved[saved.policy==p] for p in POLICIES)):
            common['validate'](group,base)
        frames.extend((base.assign(policy='current'),mild.assign(policy='mild_btc'),saved))
    rows = []
    for (arm,policy),g in pd.concat(frames).groupby(['arm','policy']):
        for day,h in [('pooled',g),*g.groupby('day')]:
            h = h.rename(columns={"clip":"clip_count","wait":"wait_count"})
            metrics = common['metrics'](h)
            if policy!='current':
                clips,waits = int(h.clip_count.sum()),int(h.wait_count.sum())
                metrics.update(clips=clips,waits=waits,proposals=int(h.posts.sum())+waits,
                               affected_markets=int(((h.clip_count+h.wait_count)>0).sum()))
            rows.append(dict(arm=arm,policy=policy,day=day,**metrics))
    table = pd.DataFrame(rows)
    table.to_csv(HERE/'comparison.csv',index=False)
    pool = table[table.day=='pooled'].set_index(['arm','policy'])
    tests={}
    for arm in ref['ARMS']:
        a=pool.loc[arm,'btc_time']
        tests[arm] = dict(benchmark=common['decision'](pool.loc[arm,'current'],a),
                          beats_time=bool(a.settle>pool.loc[(arm,'time_only'),'settle']))
    result = dict(tests=tests,passes=all(v['benchmark']=='passes' and v['beats_time'] for v in tests.values()),
                  sha256={p.name:R.sha(p) for p in sorted(HERE.glob('*.csv'))})
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(pool.to_string(),tests,flush=True)


def run():
    fixture()
    btc_dir=R.X.DATA/'maker_trace_v2_20261123/btc'
    for line in (PRIOR/'btc.sha256').read_text().splitlines():
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
        prior=read_prior(day).set_index(['mid','arm','policy'])
        reg=R.MM.registry(day).set_index('mid')
        payout=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
        btc=R.BTC.load_btc(btc_dir,day,Counter())
        off=tape.offset(day)
        rows=[]
        for _,hour in reg.groupby('hour'):
            batch=sorted(set(hour.index)&ids)
            if not batch:
                continue
            for m,status,path,info in R.MM.worlds(day,batch,tape,off,reg):
                ws=1000.*reg.at[m,'ws']
                ident=R.MM.ident_of(day,m,ws,info,path)
                assert status=='complete' and ident['tape_id']==lookup.at[(m,'nominal'),'tape_id']
                flags,z=U.S.signal_data(btc,info,ws,edges)
                times=info['t_start']+R.BE.STEP*np.arange(len(z))
                left=(ws+300000-times)/1000
                limits=curves(left,z)
                active=(left<210)&(left>30)
                args=(path,info,ident,flags,fit,R.VM.V.payout_of(payout.loc[m]))
                for arm in ref['ARMS']:
                    def parity(row,record):
                        for k,v in row.items():
                            if k in record:
                                assert abs(v-record[k])<1e-6,(m,arm,k)
                    if m==min(ids):
                        parity(ref['simulate'](arm,*args),lookup.loc[m,arm])
                        mild=pilot['curves'](times,ws,z)[1]
                        parity(pilot['simulate'](arm,args,mild),prior.loc[m,arm,'btc_time'])
                    for policy,lim in zip(POLICIES,limits):
                        row=pilot['simulate'](arm,args,lim)
                        if row['clip']+row['wait']==0:
                            parity(row,lookup.loc[m,arm])
                        rows.append(dict(day=day,mid=m,tape_id=ident['tape_id'],arm=arm,policy=policy,
                                         active_ticks=int(active.sum()),missing_ticks=int((active&~np.isfinite(z)).sum()),**row))
        frame=pd.DataFrame(rows)
        for policy in POLICIES:
            common['validate'](frame[frame.policy==policy],base)
        frame.to_csv(HERE/f'markets_{day}.csv',index=False)
        print(day,len(ids),'markets reconciled',flush=True)
    summarize()


if __name__=='__main__':
    run()
