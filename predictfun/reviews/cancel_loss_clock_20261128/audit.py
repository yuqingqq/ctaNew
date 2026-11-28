"""Reconciled observational clock attribution on the two losing dates of the recent eight."""
from collections import Counter, defaultdict
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pressure_market_sharpe_20261128'))
from diagnose import load_day
from trial import DATA
DAYS=('20261115','20261125')


def clock_status(o,t,warnings):
    assert o['act']<=t<o['remove']
    if o['req'] is not None and o['req']<t:
        return 'cancel_in_flight'
    j=warnings.searchsorted(o['act'],side='left')
    if j<len(warnings) and warnings[j]<t:
        return 'warning_too_late' if warnings[j]+37>t else 'warning_with_lead'
    return 'no_prior_live_warning'


def cases():
    w=np.array([100.])
    o=dict(act=58.,req=None,remove=500.)
    assert clock_status(o,100,w)=='no_prior_live_warning'
    assert clock_status(o,136,w)=='warning_too_late' and clock_status(o,137,w)=='warning_with_lead'
    assert clock_status(dict(o,req=100,remove=137),101,w)=='cancel_in_flight'
    assert clock_status(dict(o,req=100),100,w)=='no_prior_live_warning'
    try:
        clock_status(dict(o,remove=137),137,w)
    except AssertionError:
        pass
    else:
        raise AssertionError('counted a removed order')


def analyze(ev,args,feed,b,row,market,day):
    path,info,ident,flags,fit,payout=args
    frames=path['frames']
    rc=frames[:,1]
    def mid(t,strict):
        i=rc.searchsorted(t,side='left' if strict else 'right')-1
        if i>=0 and t-rc[i]<=1000 and 0<=frames[i,2]<frames[i,12]<=100:
            return (frames[i,2]+frames[i,12])/2
    warnings=[np.sort(np.r_[np.flatnonzero(flags[0][s])*ev.STEP,[t for t,side in feed if side==s]]) for s in (0,1)]
    tags={}
    fill_groups=defaultdict(Counter)
    for f in b.E.fills:
        t,gid,side,q,price,it=f
        order=b.E.o[gid]
        key=clock_status(order,t,warnings[side])
        before=mid(t,True)
        s=1-2*side
        tick=path['trades'][it,2]
        assert s*(price-tick)>=0
        quote='unknown' if before is None else ('wrong_side_of_mid' if s*(before-price)<0 else 'at_or_favorable_to_mid')
        tags[(gid,it)]=(key,quote,'same' if price==tick else 'through')
        v=fill_groups[key]
        v['shares']+=q
        at=t+1000
        i=rc.searchsorted(at,side='right')-1
        if at<=info['end_rel'] and i>=0 and 0<=frames[i,2]<frames[i,12]<=100:
            value=s*((frames[i,2]+frames[i,12])/2-price)*q/100
            v['valued']+=q
            v['gross_bad_1s']+=min(0,value)
            v['gross_good_1s']+=max(0,value)
    assert abs(sum(v['shares'] for v in fill_groups.values())-row['shares'])<1e-6
    assert abs(fill_groups['cancel_in_flight']['shares']-row['post_request_shares'])<1e-6
    pairs=defaultdict(Counter)
    for a,z,q in ev.H.U.C.inventory_pairs(b.E.fills):
        ka,price,trade=tags[(a[1],a[5])]
        kz=tags[(z[1],z[5])][0]
        s=1-2*a[2]
        net=s*(z[4]-a[4])*q/100
        v=pairs[ka,kz,price,trade,'losing' if net<0 else 'other']
        v.update(shares=q,net=net,bad=min(0,net),good=max(0,net))
        m0,m1,mz=(mid(t,False) for t in (a[0],min(a[0]+1000,z[0]),z[0]))
        if None in (m0,m1,mz):
            v.update(missing_shares=q,missing_value=net)
            continue
        v.update(entry=s*(m0-a[4])*q/100,first_second=s*(m1-m0)*q/100,
                 later=s*(mz-m1)*q/100,exit=s*(z[4]-mz)*q/100)
    for k,v in pairs.items():
        assert abs(sum(v[x] for x in ('entry','first_second','later','exit','missing_value'))-v['net'])<1e-6
    for k,ref in (('bad','gross_loss'),('good','gross_gain'),('net','turnover'),('shares','closed')):
        assert abs(sum(v[k] for v in pairs.values())-row[ref])<1e-6
    ids=dict(day=day,mid=market)
    return ([dict(**ids,opening=k[0],closing=k[1],prefill=k[2],trade=k[3],sign=k[4],**v) for k,v in pairs.items()],
            [dict(**ids,status=k,**v) for k,v in fill_groups.items()])


def run(day):
    cases()
    spec=spec_from_file_location('event_for_clock', HERE.parent/'event_pressure_cancel_20261127/run.py')
    ev=module_from_spec(spec)
    spec.loader.exec_module(ev)
    arch=load_day(day).set_index('tape_id')
    base=DATA/('event_pressure_nov14_18' if day<'20261124' else 'event_pressure_nov24_26')
    dest=DATA/'cancel_loss_clock_20261128'/day
    dest.mkdir(exist_ok=True,parents=True)
    ev.H.OUT,ev.H.BTC,ev.PRIOR,ev.HERE,ev.DAYS=base,base/'btc',base/day,dest,DAYS
    simulate=ev.simulate
    finish=ev.R.LB.finish
    seen=Counter()
    pairs,marks=[],[]
    def measured(args,limits,feed):
        tape=args[2]['tape_id']
        seen[tape]+=1
        if seen[tape]==1:
            assert not feed
            return simulate(args,limits,feed)
        branches=[]
        def observe(branch):
            branches.append(branch)
            return finish(branch)
        with patch.object(ev.R.LB,'finish',observe):
            row,fills=simulate(args,limits,feed)
        branch,=branches
        table,fg=analyze(ev,args,feed,branch,row,int(arch.at[tape,'mid']),day)
        pairs.extend(table)
        marks.extend(fg)
        assert abs(sum(r.get('gross_bad_1s',0) for r in fg)-arch.at[tape,'event_marks_adverse'])<1e-6
        assert abs(sum(r.get('gross_good_1s',0) for r in fg)-arch.at[tape,'event_marks_favorable'])<1e-6
        return row,fills
    with patch.object(ev,'simulate',measured):
        ev.one_day(day)
    f=pd.read_csv(dest/f'paired_{day}.csv',float_precision='round_trip').set_index('tape_id').loc[arch.index]
    assert set(seen)==set(arch.index) and all(n==2 for n in seen.values()) and list(f.columns)==list(arch.columns)
    np.testing.assert_allclose(f.to_numpy(dtype=float),arch.to_numpy(dtype=float),rtol=0,atol=1e-6)
    for name,rows in [('pairs',pairs),('fills',marks)]:
        pd.DataFrame(rows).fillna(0).to_csv(dest/(name+'.csv'),index=False)
    print('paired ledger and attribution reproduced', day,len(f),flush=True)

if __name__=='__main__':
    assert sys.argv[1] in DAYS
    run(sys.argv[1])
