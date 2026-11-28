"""One predeclared entry change, with a full feedback replay and original-event parity."""
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch
import importlib.util
import json
import math
import sys

import numpy as np
import pandas as pd
from diagnose import DAYS, HERE, load_day

ROOT = HERE.parents[2]
DATA = ROOT/'data/predict_fun/research'
OUT = DATA/'pressure_market_sharpe_20261128'


def size_at_touch(view, rung, size, price):
    if rung != 0 or view['ask']-view['bid'] != 1:
        return size
    side = view['side']
    signed = (1-2*side)*view['position']
    free = math.floor(-signed-view['A' if side else 'B']+1e-9)
    if free >= size:
        return size
    native = 100-price if side else price
    return free if free >= 1 and free*native >= 100 else 0


def fixtures():
    v = dict(side=0, position=-30., B=10, A=25, bid=49., ask=50.)
    assert size_at_touch(v,0,50,49)==20
    assert size_at_touch(dict(v,position=30),0,50,49)==0
    assert size_at_touch(dict(v,side=1,position=40),0,50,50)==15
    assert size_at_touch(dict(v,side=1,position=25),0,50,50)==0
    assert size_at_touch(dict(v,position=-11),0,50,49)==0
    assert size_at_touch(v,0,5,49)==5 and size_at_touch(v,1,50,48)==50
    assert size_at_touch(dict(v,ask=51),0,50,49)==50


def attribution(ev, path, fills, births, ledger):
    groups = defaultdict(Counter)
    frames = path['frames']
    clock = frames[:,1]
    def mid(t):
        i = clock.searchsorted(t,side='right')-1
        if i >= 0 and t-clock[i] <= 1000 and 0 <= frames[i,2] < frames[i,12] <= 100:
            return (frames[i,2]+frames[i,12])/2
    for a,b,q in ev.H.U.C.inventory_pairs(fills):
        s=1-2*a[2]
        pnl=s*(b[4]-a[4])*q/100
        group = groups[(births[a[1]],'loss' if pnl<0 else 'gain' if pnl>0 else 'flat')]
        group.update(shares=q,pnl=pnl,share_seconds=q*(b[0]-a[0])/1000)
        m0,m1=mid(a[0]),mid(b[0])
        if m0 is None or m1 is None:
            group.update(missing_shares=q,missing_pnl=pnl)
            continue
        group.update(open_edge=s*(m0-a[4])*q/100,hold_move=s*(m1-m0)*q/100,close_edge=s*(b[4]-m1)*q/100)
    total=Counter()
    for group in groups.values():
        assert abs(sum(group[k] for k in ('open_edge','hold_move','close_edge','missing_pnl'))-group['pnl'])<1e-6
        total.update(group)
    assert abs(total['pnl']-ledger['turnover'])<1e-6 and abs(total['shares']-ledger['closed'])<1e-6
    return groups


def run(day):
    fixtures()
    path=HERE.parent/'event_pressure_cancel_20261127/run.py'
    spec=importlib.util.spec_from_file_location('replay_event_for_spread',path)
    ev=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    archive=load_day(day).set_index('tape_id')
    base=DATA/('event_pressure_nov14_18' if day<'20261124' else 'event_pressure_nov24_26')
    dest=OUT/day
    dest.mkdir(parents=True,exist_ok=True)
    ev.H.OUT,ev.H.BTC,ev.PRIOR,ev.HERE,ev.DAYS=base,base/'btc',base/day,dest,DAYS
    simulate=ev.simulate
    cls=ev.H.pilot['ConditionalSize']
    parent=cls.slot
    passes=Counter()
    comparisons=[]
    pairs=[]

    def replay(args,limits,feed):
        path,info,ident,flags,fit,payout=args
        tape=ident['tape_id']
        passes[tape]+=1
        if passes[tape]==1:
            assert not feed
            return simulate(args,limits,feed)
        assert passes[tape]==2
        market=int(archive.at[tape,'mid'])
        row=dict(day=day,mid=market,tape_id=tape)
        for arm in ('current','candidate'):
            births={}
            stats=Counter()
            def slot(pol,v,r,p):
                opened,first=pol.opened,pol.first_side
                proposal=parent(pol,v,r,p)
                action,size,_=proposal
                if action!='POST':
                    return proposal
                allowed=size_at_touch(v,r,size,p)
                if arm=='candidate' and allowed!=size:
                    if allowed==0:
                        pol.proposals.pop()
                        pol.opened,pol.first_side=opened,first
                        stats['veto']+=1
                        return 'WAIT',None,'min_spread_intake'
                    g,s,cap,level,_,unit=pol.proposals[-1]
                    reserved=v['A' if s else 'B']
                    pol.proposals[-1]=(g,s,cap,level,max(0.,(1-2*s)*v['position']+reserved+allowed),unit)
                    proposal=('POST',allowed,'min_spread_close')
                    stats['clip']+=1
                label='one_tick_touch' if r==0 and v['ask']-v['bid']==1 else 'other'
                births[len(pol.b.E.o)]=label
                return proposal
            with patch.object(cls,'slot',slot):
                ledger,fills=simulate(args,limits,feed)
            assert abs(ledger['turnover']+ledger['residual']-ledger['settle'])<1e-6
            assert abs(ledger['gross_gain']+ledger['gross_loss']-ledger['turnover'])<1e-6
            exp=ev.exposures(fills,path)
            marks=ev.valued(exp,path,info['end_rel'],payout)
            assert abs(marks['payout_win']+marks['payout_loss']-ledger['settle'])<1e-6
            row.update({arm+'_'+k:x for k,x in ledger.items()})
            row[arm+'_1s_net']=marks['favorable']+marks['adverse']
            row[arm+'_1s_shares']=marks['shares']-marks['censored']
            for (birth,sign),parts in attribution(ev,path,fills,births,ledger).items():
                pairs.append(dict(day=day,mid=market,arm=arm,birth=birth,sign=sign,**parts))
            if arm=='current':
                original=(ledger,fills)
            else:
                row.update(veto=stats['veto'],clipped=stats['clip'])
        comparisons.append(row)
        return original

    with patch.object(ev,'simulate',replay):
        ev.one_day(day)
    assert all(x==2 for x in passes.values()) and set(passes)==set(archive.index)
    check=pd.read_csv(dest/f'paired_{day}.csv',float_precision='round_trip').set_index('tape_id').loc[archive.index]
    assert list(check.columns)==list(archive.columns)
    np.testing.assert_allclose(check.to_numpy(dtype=float),archive.to_numpy(dtype=float),rtol=0,atol=1e-6)
    pd.DataFrame(comparisons).to_csv(dest/'comparison.csv',index=False)
    pd.DataFrame(pairs).fillna(0).to_csv(dest/'attribution.csv',index=False)
    print(day,len(comparisons),'full-panel parity and FIFO bridge passed',flush=True)


if __name__=='__main__':
    if sys.argv[1]=='fixture':
        fixtures()
    else:
        assert sys.argv[1] in DAYS
        run(sys.argv[1])
