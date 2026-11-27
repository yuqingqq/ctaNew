"""One lower time-dependent far cap against the previously frozen near-tight policy."""
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent/'urgency_early_strong_20261127'
old = runpy.run_path(str(PRIOR/'run.py'))
pilot,ref,common = (old[k] for k in ('pilot','ref','common'))
R,U,DAYS = (old[k] for k in ('R','U','DAYS'))


def limit(left,z):
    far_time, old_limit = old['curves'](left,z)
    near = np.interp(left,[30.,90.,210.],[0.,100.,150.])
    p = np.clip(1.5-np.abs(z),0,1)
    return np.where(np.isfinite(z),far_time+p*(near-far_time),old_limit)


def fixture():
    old['fixture']()
    left=np.array([240.,210.,150.,90.,60.,48.,42.,39.,30.,0.])
    far=np.array([150,150,137.5,125,100,90,60,45,0,0])
    near=np.array([150,150,125,100,50,30,20,15,0,0])
    for z,expected in ((2,far),(-2,far),(.5,near),(-.5,near),(1,(far+near)/2),(np.nan,far)):
        states=np.full(len(left),z)
        actual=limit(left,states)
        assert np.allclose(actual,expected)
        assert (actual<=old['curves'](left,states)[1]+1e-8).all()
    assert limit(np.array([90]),np.array([2]))[0]!=150


def reference(day):
    path=PRIOR/f'markets_{day}.csv'
    assert R.sha(path)==json.loads((PRIOR/'result.json').read_text())['sha256'][path.name]
    return pd.read_csv(path,float_precision='round_trip',dtype={'day':str})


def score():
    frames=[]
    for day in DAYS:
        prev=reference(day)
        base=prev[prev.policy=='btc_time'].assign(policy='previous_btc')
        new=pd.read_csv(HERE/f'markets_{day}.csv',float_precision='round_trip',dtype={'day':str})
        common['validate'](new,base)
        frames.extend((base,prev[prev.policy=='time_only'],new))
    rows=[]
    for (arm,policy),group in pd.concat(frames).groupby(['arm','policy']):
        for day,h in [('pooled',group),*group.groupby('day')]:
            rows.append(dict(day=day,arm=arm,policy=policy,**common['metrics'](h),
                             clips=int(h['clip'].sum()),waits=int(h.wait.sum()),
                             proposals=int((h.posts+h.wait).sum())))
    table=pd.DataFrame(rows)
    table.to_csv(HERE/'comparison.csv',index=False)
    p=table[table.day=='pooled'].set_index(['arm','policy'])
    tests={a:common['decision'](p.loc[a,'previous_btc'],p.loc[a,'lower_far']) for a in ref['ARMS']}
    receipt=dict(tests=tests,passes=all(x=='passes' for x in tests.values()),
                 sha256={p.name:R.sha(p) for p in sorted(HERE.glob('*.csv'))})
    (HERE/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(p.to_string(),tests,flush=True)


def run():
    fixture()
    btc_dir=R.X.DATA/'maker_trace_v2_20261123/btc'
    for line in (old['PRIOR']/'btc.sha256').read_text().splitlines():
        sha,name=line.split()
        assert R.sha(btc_dir/Path(name).name)==sha
    fit=U.S.read(R.HERE/'fit_v2.json')['cells']
    bind=U.S.read(ref['S']['L']['E'].REPLICATION/'result.json')
    assert R.sha(R.HERE/'fit_v2.json')==bind['fit_sha'] and R.sha(R.CENSUS)==bind['census_sha']
    tape=R.MM.KP.Tape()
    edges=R.MM.TR.T.P.frozen_edges()
    for day in DAYS:
        records=reference(day)
        prev=records[records.policy=='btc_time']
        lookup=prev.set_index(['mid','arm'])
        ids=set(prev.mid)
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
                left=(ws+300000-info['t_start']-R.BE.STEP*np.arange(len(z)))/1000
                _,old_lim=old['curves'](left,z)
                new_lim=limit(left,z)
                tighter=new_lim<old_lim-1e-8
                assert (new_lim<=old_lim+1e-8).all()
                assert not (tighter&(~np.isfinite(z)|(np.abs(z)<=.5))).any()
                args=(path,info,ident,flags,fit,R.VM.V.payout_of(payout.loc[m]))
                for arm in ref['ARMS']:
                    def parity(row):
                        for k,v in row.items():
                            assert abs(v-lookup.at[(m,arm),k])<1e-6,(m,arm,k)
                    if m==min(ids):
                        parity(pilot['simulate'](arm,args,old_lim))
                    row=pilot['simulate'](arm,args,new_lim)
                    if not tighter.any():
                        parity(row)
                    rows.append(dict(day=day,mid=m,tape_id=ident['tape_id'],arm=arm,policy='lower_far',
                                     active_ticks=int(((left<210)&(left>30)).sum()),tighter_ticks=int(tighter.sum()),
                                     far_tighter_ticks=int((tighter&(np.abs(z)>=1.5)).sum()),**row))
        frame=pd.DataFrame(rows)
        common['validate'](frame,prev)
        frame.to_csv(HERE/f'markets_{day}.csv',index=False)
        print(day,len(ids),'paired markets reconciled',flush=True)
    score()


if __name__=='__main__':
    run()
