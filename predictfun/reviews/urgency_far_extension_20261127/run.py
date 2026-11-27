"""Fixed-curve extension, five previously studied dates; pilot shown separately."""
from collections import Counter
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from grid import DAYS, OUT, CONTROL

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'urgency_far_taper_20261127'
far=runpy.run_path(str(PRIOR/'run.py'))
ref,common,old,pilot,R,U=(far[k] for k in ('ref','common','old','pilot','R','U'))
OD=ref['OD']
POLICIES=('previous_btc','lower_far')


def btc_for(day):
    D=R.BTC
    allowed=[day]
    if D.prev_day(day) in DAYS:
        allowed.insert(0,D.prev_day(day))
    for name in allowed:
        record=json.loads((HERE/f'grid_{name}.json').read_text())
        assert record['sha256']==R.sha(OUT/f'btc_{name}.parquet')
    with patch.multiple(D,BTC_DAYS=tuple(allowed),PAD_LO_MS=D.day_ms(allowed[0]),BTC_END_MS=D.day_ms(day)+86400000):
        return D.load_btc(OUT,day,Counter())


def run_day(day):
    assert day in DAYS
    assert json.loads((HERE/f'grid_{CONTROL}.json').read_text())['control_parity']
    far['fixture']()
    fit=U.S.read(R.HERE/'fit_v2.json')['cells']
    bind=U.S.read(ref['S']['L']['E'].REPLICATION/'result.json')
    assert R.sha(R.HERE/'fit_v2.json')==bind['fit_sha'] and R.sha(R.CENSUS)==bind['census_sha']
    assert R.sha(OD.OUT3/'bn_changes.parquet')==U.S.read(OD.RCPT/'bn_events.json')['out_sha256']
    assert R.sha(OD.OUT3/'summary.json')=='f1b45f6555c707654acf337359e98d900b67cf4c1277312dfdd4d82c0b46c460'
    tape=R.MM.KP.Tape()
    edges=R.MM.TR.T.P.frozen_edges()
    btc=btc_for(day)
    base=common['baseline'](day)
    lookup=base.set_index(['mid','arm'])
    ids=set(base.mid)
    reg=R.MM.registry(day).set_index('mid')
    payout=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
    off=tape.offset(day)
    seg=OD.SEG_OF[day]
    rows=[]
    for _,hour in reg.groupby('hour'):
        batch=sorted(set(hour.index)&ids)
        if not batch:
            continue
        for m,status,path,info in R.MM.worlds(day,batch,tape,off,reg):
            ws=1000.*reg.at[m,'ws']
            ident=R.MM.ident_of(day,m,ws,info,path)
            assert status=='complete' and ident['tape_id']==lookup.at[(m,'nominal'),'tape_id']
            assert OD.SEG_BOUNDS[seg][0]<=ws<ws+300000<OD.SEG_BOUNDS[seg][1]
            with patch.object(R.MM,'make_cache',lambda t0,end:OD.make_cache3(t0,end,seg)):
                flags,z=U.S.signal_data(btc,info,ws,edges)
            left=(ws+300000-info['t_start']-R.BE.STEP*np.arange(len(z)))/1000
            _,prior=old['curves'](left,z)
            lower=far['limit'](left,z)
            active=(left>30)&(left<210)
            tighter=lower<prior-1e-8
            assert (lower<=prior+1e-8).all() and not (tighter&(~np.isfinite(z)|(np.abs(z)<=.5))).any()
            args=(path,info,ident,flags,fit,R.VM.V.payout_of(payout.loc[m]))
            for arm in ref['ARMS']:
                def parity(numbers,record):
                    for k,v in numbers.items():
                        if k in record:
                            assert abs(v-record[k])<1e-6,(day,m,arm,k)
                if m==min(ids):
                    parity(ref['simulate'](arm,*args),lookup.loc[(m,arm)])
                prev_result=None
                for name,lim in zip(POLICIES,(prior,lower)):
                    row=pilot['simulate'](arm,args,lim)
                    if row['clip']+row['wait']==0:
                        parity(row,lookup.loc[(m,arm)])
                    if name=='previous_btc':
                        prev_result=row
                    elif not tighter.any():
                        parity(row,prev_result)
                    rows.append(dict(day=day,mid=m,tape_id=ident['tape_id'],arm=arm,policy=name,
                                     active_ticks=int(active.sum()),missing_ticks=int((active&~np.isfinite(z)).sum()),
                                     tighter_ticks=int(tighter.sum()),far_tighter_ticks=int((tighter&(np.abs(z)>=1.5)).sum()),**row))
    frame=pd.DataFrame(rows)
    for name in POLICIES:
        common['validate'](frame[frame.policy==name],base)
    frame.to_csv(HERE/f'markets_{day}.csv',index=False)
    print(day,len(ids),'market paths paired under both timing assumptions',flush=True)


def score():
    frames=[]
    for day in (*far['DAYS'],*DAYS):
        base=common['baseline'](day)
        if day in DAYS:
            rows=pd.read_csv(HERE/f'markets_{day}.csv',float_precision='round_trip',dtype={'day':str})
        else:
            p=PRIOR/f'markets_{day}.csv'
            assert R.sha(p)==json.loads((PRIOR/'result.json').read_text())['sha256'][p.name]
            lower=pd.read_csv(p,float_precision='round_trip',dtype={'day':str})
            previous=far['reference'](day).query("policy=='btc_time'").assign(policy='previous_btc')
            rows=pd.concat((lower,previous))
        for name in POLICIES:
            common['validate'](rows[rows.policy==name],base)
        frames.extend([base.assign(policy='original'),rows])
    panels=[]
    for (arm,name),h in pd.concat(frames).groupby(['arm','policy']):
        for label,f in [('pilot3',h[~h.day.isin(DAYS)]),('extension5',h[h.day.isin(DAYS)]),('all8',h),*h.groupby('day')]:
            panels.append(dict(arm=arm,policy=name,period=label,**common['metrics'](f)))
    comparison=pd.DataFrame(panels)
    comparison.to_csv(HERE/'comparison.csv',index=False)
    tests={}
    for label in ('extension5','all8','pilot3'):
        p=comparison[comparison.period==label].set_index(['arm','policy'])
        tests[label]={a:common['decision'](p.loc[a,'previous_btc'],p.loc[a,'lower_far']) for a in ref['ARMS']}
        print(label,p[['settle','residual','turnover','mean_end','tail_end','tail_loss','positive_days']].to_string(),flush=True)
    result=dict(decisions=tests,extension_passes=all(v=='passes' for v in tests['extension5'].values()),
                sha256={p.name:R.sha(p) for p in sorted([*HERE.glob('*.csv'),*HERE.glob('grid_*.json')])})
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    if sys.argv[1]=='score':
        score()
    else:
        run_day(sys.argv[1])
