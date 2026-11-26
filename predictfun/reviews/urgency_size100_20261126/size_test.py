"""Matched sensitivity of the frozen passive urgency policy to maximum order size."""
import json
import math
import sys
from collections import Counter
from pathlib import Path
from runpy import run_path
from subprocess import check_output
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
L=run_path(str(HERE.parent/'urgency_later_dates_20261126/replay.py'))
U,R,OD=L['U'],L['R'],L['OD']
DAYS=('20261023','20261024','20261025','20261026','20261027',*L['DAYS'])
FIELDS=L['FIELDS']
ARCHIVE='8c4023528'
NATIVE_SIZE=R.max_new


def size_bound(unit):
    def propose(signed, reserved, capacity, downside):
        return min(unit,NATIVE_SIZE(signed,reserved,capacity,downside))
    return propose


def archived(day):
    if day=='20261026':
        folder=HERE.parent/'inventory_urgency_20261125'
    elif day=='20261027':
        folder=HERE.parent/'urgency_replication_20261125'
    else:
        folder=(HERE.parent/'urgency_earlier_dates_20261126' if day<'20261026' else HERE.parent/'urgency_later_dates_20261126')
    path=folder/('markets.csv' if day in ('20261026','20261027') else f'markets_{day}.csv')
    assert path.read_bytes()==check_output(['git','show',f'{ARCHIVE}:{path.relative_to(HERE.parents[2])}'],cwd=HERE.parents[2])
    frame=pd.read_csv(path,float_precision='round_trip').query("arm=='urgency'").set_index('mid')
    assert frame.index.is_unique
    return frame


def order_check(orders,unit):
    for o in orders:
        if o['size']<1 or o['size']>unit or o['size']!=int(o['size']):
            raise ValueError('invalid submitted quantity')


def simulate(path,info,ident,flags,model,payout,unit):
    assert unit in (100,220) and payout in (0,.5,1)
    branch=U.UrgencyBranch(path,info['end_rel'],ident)
    branch.q0=unit
    policy=U.S.Smooth(branch,flags,model,'linear',{},[])
    assert policy.q0==unit and policy.Qside==440 and branch.cap==1000 and branch.levels==2
    # Confined to the serial replay in this process. The proposal size is bounded
    # before the existing order minimum, inventory reservations and POST bookkeeping.
    with patch.object(R,'max_new',size_bound(unit)):
        for tick in range(R.LB.n_ticks(branch)):
            R.LB.step(branch,tick,policy)
        R.LB.finish(branch)
    assert R.max_new is NATIVE_SIZE
    fills=branch.E.fills
    orders=branch.E.o
    order_check(orders,unit)
    assert len(policy.proposals)==len(orders)
    pos=np.r_[0.,np.cumsum([(1-2*f[2])*f[3] for f in fills])]
    assert max(abs(pos))<=1000+1e-6
    _,end,settle=branch.E.value(100*payout)
    fifo=R.X.fifo(fills,100*payout)
    result=dict(settle=settle/100,turnover=fifo['completed_c']/100,residual=fifo['residual_c']/100,
                closed=fifo['matched_shares'],end_pos=end,abs_end=abs(end),shares=sum(f[3] for f in fills),
                posts=len(orders),all_pulls=sum(o['req'] is not None for o in orders))
    assert abs(end-pos[-1])<1e-6 and abs(result['shares']-2*result['closed']-abs(end))<1e-6
    assert abs(result['settle']-result['turnover']-result['residual'])<1e-6
    return result


def test():
    size=size_bound(100)
    assert size(-200,0,1000,.5)==100 and NATIVE_SIZE(-200,0,1000,.5)==220
    assert size(980,5,5000,.3)==15 and size(300,100,0,.5)==0 and size(-200,430,1000,.5)==10
    order_check([dict(size=100),dict(size=4)],100)
    try:
        order_check([dict(size=101)],100)
    except ValueError:
        return
    raise AssertionError('oversize order accepted')


def compare(df):
    arms={}
    for n,g in df.groupby('size'):
        assert not g.duplicated(['day','mid']).any()
        assert ((g.settle-g.turnover-g.residual).abs()<1e-5).all()
        assert ((g.shares-2*g.closed-g.abs_end).abs()<1e-5).all()
        arms[str(n)]={key:float(g[key].sum()) for key in ('settle','turnover','residual','closed','posts','all_pulls')}
        arms[str(n)].update(n=len(g),mean_abs_end=float(g.abs_end.mean()),p90_end=float(g.abs_end.quantile(.9)),
                            cents_per_matched=float(100*g.turnover.sum()/g.closed.sum()))
    x=df.pivot(index=['day','mid'],columns='size',values='settle')
    assert x.notna().all().all()
    diff=x[100]-x[220]
    return dict(arms=arms,delta=float(diff.sum()),without_best=float(diff.sum()-diff.max()),
                better=int((diff>1e-6).sum()),worse=int((diff< -1e-6).sum()),
                positive_100=arms['100']['settle']>0)


def run():
    test()
    prior=U.S.read(L['E'].REPLICATION/'result.json')
    assert R.sha(R.HERE/'fit_v2.json')==prior['fit_sha'] and R.sha(R.CENSUS)==prior['census_sha']
    assert R.sha(L['E'].PRIOR/'urgency.py')==prior['controller_sha']
    assert R.sha(OD.OUT3/'bn_changes.parquet')==U.S.read(OD.RCPT/'bn_events.json')['out_sha256']
    fit=U.S.read(R.HERE/'fit_v2.json')['cells']
    raw=U.S.read(R.CENSUS)['rows']
    assert R.sha(OD.OUT3/'summary.json')=='f1b45f6555c707654acf337359e98d900b67cf4c1277312dfdd4d82c0b46c460'
    alt=U.S.read(OD.OUT3/'summary.json')['census']
    src=R.MM.KP.Tape()
    edges=R.MM.TR.T.P.frozen_edges()
    days={}
    for day in DAYS:
        old=archived(day)
        seg=OD.SEG_OF.get(day)
        census=[r for r in (alt if seg else raw) if r['day']==day]
        reg=R.MM.registry(day).set_index('mid')
        assert len(reg)==len(census)==288 and set(reg.index)=={r['mid'] for r in census}
        ids={r['mid'] for r in census if r['status']=='complete'}
        assert ids==set(old.index)
        expected={r['mid']:r['tape_id'] for r in census if r['status']=='complete'} if not seg else old.tape_id.to_dict()
        rg=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
        off=src.offset(day)
        rows=[]
        seen=set()
        for hour,group in reg.groupby('hour'):
            batch=sorted(ids & set(group.index))
            if not batch:
                continue
            for m,status,path,info in R.MM.worlds(day,batch,src,off,reg):
                assert status=='complete' and m in batch and m not in seen
                seen.add(m)
                ws=1000.*reg.at[m,'ws']
                ident=R.MM.ident_of(day,m,ws,info,path)
                assert expected[m]==ident['tape_id']
                if seg:
                    assert OD.SEG_BOUNDS[seg][0]<=ws<ws+300000<OD.SEG_BOUNDS[seg][1]
                    cache=OD.make_cache3(info['t_start'],info['end_rel'],seg)
                else:
                    cache=R.MM.make_cache(info['t_start'],info['end_rel'])
                args=path,info,ident,R.triggers(cache,edges),fit,R.VM.V.payout_of(rg.loc[m])
                base=old.loc[m,list(FIELDS)].to_dict()
                if m==min(ids):
                    control=simulate(*args,220)
                    assert all(abs(control[k]-base[k])<1e-6 for k in FIELDS),(day,m)
                candidate=simulate(*args,100)
                for n,arm in ((100,candidate),(220,base)):
                    rows.append(dict(day=day,mid=m,size=n,**arm))
        assert seen==ids
        df=pd.DataFrame(rows)
        daily=compare(df)
        daily['coverage']=dict(Counter(r['status'] for r in census))
        days[day]=daily
        df.to_csv(HERE/f'markets_{day}.csv',index=False,mode='x')
        print(day,len(seen),'verified',flush=True)
    all_rows=pd.concat([pd.read_csv(HERE/f'markets_{d}.csv',dtype={'day':str},float_precision='round_trip') for d in DAYS])
    totals=compare(all_rows)
    later=compare(all_rows[all_rows.day.isin(L['DAYS'])])
    result=dict(days=days,all_dates=totals,four_later_dates=later,
                profitable_days=sum(r['positive_100'] for r in days.values()),
                higher_pnl_days=sum(r['delta']>0 for r in days.values()))
    with (HERE/'result.json').open('x') as f:
        json.dump(result,f,allow_nan=False,indent=2)
        f.write('\n')


if __name__=='__main__':
    {'test':test,'run':run}[sys.argv[1]]()
