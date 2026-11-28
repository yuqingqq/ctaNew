"""Overlay a received, directional alert on a complete independent feedback branch."""
from collections import Counter, defaultdict
from importlib.util import spec_from_file_location, module_from_spec
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd
from book_signal import DATA, DAYS, DELAY, FRESH, HERE, OUT
sys.path.insert(0,str(HERE.parent/'pressure_market_sharpe_20261128'))
from diagnose import load_day


def flags_at(clock,mask,at):
    j=clock.searchsorted(at,'right')-1
    return np.where((j>=0)&(at-clock[j]<=FRESH),mask[j],0)


def fixture():
    c=np.array([104,130,179])
    m=np.array([2,0,1])
    np.testing.assert_array_equal(flags_at(c,m,np.array([103,104,129,130,178,179,479,480])),[0,2,2,0,0,1,1,0])


def diagnostics(ev,branch,path,order_ids,births):
    next_by_price=defaultdict(list)
    for o in branch.E.o:
        if not o['refused']:
            next_by_price[o['side'],o['price']].append(o)
    stats=Counter()
    for gid,(when,ahead) in order_ids.items():
        o=branch.E.o[gid]
        assert o['req']==when and o['remove']==when+37
        nxt=next((n for n in next_by_price[o['side'],o['price']] if n['submit']>when),None)
        if nxt is not None:
            stats['reentries']+=1
            stats['ahead_before']+=ahead
            stats['ahead_after']+=births[nxt['gid']]
            stats['reentry_ms']+=nxt['act']-when
    return stats


def feasible(ev,oldbranch,fills,path,gridbits,off):
    # Read the original live KEEP log and valid off-grid opportunities on that side.
    grid_keep={}
    for cmd in oldbranch.cmd_log:
        t,side,kind,gid=cmd[:4]
        if kind=='live' and cmd[7]=='KEEP' and gridbits[int(t//ev.STEP)]&(1<<side):
            grid_keep.setdefault(gid,t)
    fr=path['frames']
    ts=[np.array([t for t,s in off if s==side]) for side in (0,1)]
    for side,times in enumerate(ts):
        jj=np.maximum(0,fr[:,1].searchsorted(times,'right')-1)
        valid=np.array([ev.R.LB.book_ok_side(fr[j],side) for j in jj],dtype=bool)
        ts[side]=times[valid]
    opportunities={}
    for o in oldbranch.E.o:
        times=ts[o['side']]
        j=times.searchsorted(o['act'],'left')
        first=min(grid_keep.get(o['gid'],np.inf),times[j] if j<len(times) else np.inf)
        if o['req'] is None or first<o['req']:
            opportunities[o['gid']]=first
    selected=[f for f in fills if opportunities.get(f[1],np.inf)+37<=f[0]]
    return ev.exposures(selected,path)


def run(day):
    fixture()
    spec=spec_from_file_location('event_guard_control',HERE.parent/'event_pressure_cancel_20261127/run.py')
    ev=module_from_spec(spec)
    spec.loader.exec_module(ev)
    base=DATA/'event_pressure_nov24_26'
    dest=OUT/day
    ev.H.OUT,ev.H.BTC,ev.PRIOR,ev.HERE,ev.DAYS=base,base/'btc',base/day,dest,DAYS
    archive=load_day(day).set_index('tape_id')
    inputs=np.load(dest/'alerts.npz')
    clock,mask=inputs['clock'],inputs['mask']
    assert DELAY==ev.R.MM.B_IN and (np.diff(clock)>0).all()
    real=ev.simulate
    cls=ev.H.pilot['ConditionalSize']
    eng=ev.R.VM.VMEngine
    place,finish,cancel,live,slot=eng.add9,ev.R.LB.finish,ev.cancel_live,cls.live,cls.slot
    seen=Counter()
    rows=[]

    def compare(args,limits,feed):
        path,info,ident,_,_,payout=args
        tape=ident['tape_id']
        seen[tape]+=1
        if seen[tape]==1:
            assert not feed
            return real(args,limits,feed)
        assert seen[tape]==2
        start,end=info['t_start'],info['end_rel']
        ticks=ev.STEP*np.arange(len(limits))
        gridbits=flags_at(clock,mask,start+ticks)
        l,r=clock.searchsorted([start,start+end],'left')
        times,mm=clock[l:r]-start,mask[l:r]
        baseline=set(feed)
        alerts={(t,int(s)) for s in (0,1) for t in times[((mm&(1<<s))!=0)&(times%ev.STEP!=0)]}
        extra=alerts-baseline
        events=sorted(baseline|alerts)
        branches=[]
        def observe(b):
            branches.append(b)
            return finish(b)
        with patch.object(ev.R.LB,'finish',observe):
            old,oldfills=real(args,limits,feed)
        oldbranch,=branches
        pointer=feasible(ev,oldbranch,oldfills,path,gridbits,extra)
        branches.clear()
        intervention={}
        births={}
        count=Counter()
        def new_cancel(b,t,s):
            pulled=cancel(b,t,s)
            if (t,s) in extra:
                for gid,_,_ in pulled:
                    intervention[gid]=(t,b.E.o[gid]['ahead'])
                    count['offgrid_pulls']+=1
            return pulled
        def new_live(pol,v,o):
            result=live(pol,v,o)
            if result[0]=='KEEP' and gridbits[pol.b.k]&(1<<v['side']):
                gid=o['gid']
                intervention[gid]=(v['t'],pol.b.E.o[gid]['ahead'])
                count['grid_pulls']+=1
                return 'PULL','book_alert'
            return result
        def new_slot(pol,v,rung,price):
            opened,first=pol.opened,pol.first_side
            result=slot(pol,v,rung,price)
            if result[0]=='POST' and gridbits[pol.b.k]&(1<<v['side']):
                pol.proposals.pop()
                pol.opened,pol.first_side=opened,first
                count['waits']+=1
                return 'WAIT',None,'book_alert'
            return result
        def new_order(e,*args,**kwargs):
            gid=place(e,*args,**kwargs)
            births[gid]=e.o[gid]['ahead']
            return gid
        with patch.object(cls,'live',new_live),patch.object(cls,'slot',new_slot),patch.object(ev,'cancel_live',new_cancel),patch.object(eng,'add9',new_order),patch.object(ev.R.LB,'finish',observe):
            new,newfills=real(args,limits,events)
        newbranch,=branches
        assert len(births)==len(newbranch.E.o) and len(intervention)==count['grid_pulls']+count['offgrid_pulls']
        count.update(diagnostics(ev,newbranch,path,intervention,births))
        a,b=ev.exposures(oldfills,path),ev.exposures(newfills,path)
        row=dict(day=day,mid=int(archive.at[tape,'mid']),tape_id=tape)
        for k in ('grid_pulls','offgrid_pulls','waits','reentries','ahead_before','ahead_after','reentry_ms'):
            row[k]=count[k]
        for name,val in [('current',old),('alert',new)]:
            assert abs(val['settle']-val['turnover']-val['residual'])<1e-6
            row.update({name+'_'+k:val[k] for k in ('settle','turnover','residual','closed','shares','posts','requests','abs_end','reposts')})
        for name,exp in [('baseline',a),('candidate',b),('lost',a-b),('added',b-a),('timely_baseline',pointer)]:
            row.update({name+'_'+k:v for k,v in ev.valued(exp,path,end,payout).items()})
        def value(name):
            return row[name+'_payout_win']+row[name+'_payout_loss']
        assert abs(value('candidate')-new['settle'])<1e-6 and abs(value('baseline')-old['settle'])<1e-6
        assert abs(value('added')-value('lost')-new['settle']+old['settle'])<1e-6
        rows.append(row)
        return old,oldfills

    with patch.object(ev,'simulate',compare):
        ev.one_day(day)
    restored=pd.read_csv(dest/f'paired_{day}.csv',float_precision='round_trip').set_index('tape_id').loc[archive.index]
    assert list(restored.columns)==list(archive.columns) and len(rows)==len(archive) and set(seen)==set(archive.index)
    np.testing.assert_allclose(restored.to_numpy(float),archive.to_numpy(float),atol=1e-6,rtol=0)
    pd.DataFrame(rows).to_csv(dest/'markets.csv',index=False)
    print(day,len(rows),'paired markets and archived controls verified',flush=True)

if __name__=='__main__':
    assert sys.argv[1] in DAYS
    run(sys.argv[1])
