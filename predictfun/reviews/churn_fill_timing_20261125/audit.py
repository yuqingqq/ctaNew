"""Unmodified retained baseline on 70 complete, consumed 20261026 markets.
Account for each first cancellation, including fully matched but unnotified at request;
partition FIFO-matched opening shares by public print type and decision timing.
No intervention, fitting or selection test. A prior feasible KEEP is opportunity only.
"""
from collections import Counter,defaultdict
from pathlib import Path
import math
import sys

import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cycle_spread_20261125'))
import cycles as C

S,R=C.S,C.R


def timing(t,o,keep,why,step,cancel,pending):
    assert o['act']<=t<o['remove']
    if o['req'] is not None and o['req']<t:
        return 'in_flight_'+why
    # Ties at a grid execute before its commands. At a cancel effect timestamp
    # a removed order cannot fill. Earlier valid KEEP implies no prior request.
    if keep is not None and keep+cancel<=t:
        return 'timely_live_keep'
    if t < max(o['submit']+step+cancel,o['submit']+pending):
        return 'before_pending_effect'
    if t < math.ceil(o['act']/step)*step+cancel:
        return 'pending_only'
    return 'no_timely_valid_keep'


def requests(branch,policy):
    reasons={}
    flags={}
    for x in branch.struct_log:
        gid=x['gid']
        assert gid not in reasons and branch.E.o[gid]['req']==x['t']
        assert x['primary']==next(k for k in ('no_book','distance','cross') if x[k])
        reasons[gid]='struct_'+x['primary']
        flags[gid]='+'.join(k for k in ('no_book','distance','cross') if x[k])
    for t,gid,*_ in policy.trims:
        assert gid not in reasons and branch.E.o[gid]['req']==t
        reasons[gid]='inventory'
    first={}
    for t,side,kind,gid,_,_,_,action,why in branch.cmd_log:
        if kind!='live':
            continue
        if action=='KEEP' and why=='keep':
            first.setdefault(gid,t)
        if action=='PULL' and why=='pressure':
            assert gid not in reasons and branch.E.o[gid]['req']==t
            reasons[gid]='pressure'
    assert set(reasons)=={i for i,o in enumerate(branch.E.o) if o['req'] is not None}
    return reasons,flags,first


def collect(b,policy,ref,payout,mid):
    fills=b.E.fills
    by_order=defaultdict(list)
    for f in fills:
        by_order[f[1]].append(f)
    reasons,flags,first=requests(b,policy)
    pulls=defaultdict(Counter)
    cycles=defaultdict(Counter)
    for gid,why in reasons.items():
        o=b.E.o[gid]; t=o['req']; qty=sum(f[3] for f in by_order[gid] if f[0]<=t)
        rows=pulls[why,flags.get(gid,'')]
        rows['requests']+=1
        if o['refused']:
            rows['venue_refused']+=1
        elif qty>=o['size']-1e-8:
            rows['fully_filled_before_request']+=1
        elif t<o['act']:
            rows['post_pending']+=1
        else:
            rows['live_remainder']+=1
        rows['post_request_shares']+=sum(f[3] for f in by_order[gid] if f[0]>t)
        assert all(f[0]<o['remove'] for f in by_order[gid])
    for f,z,q in C.inventory_pairs(fills):
        t,gid,side,_,price,_=f
        kind=C.fill_kind(f,b.E.tr)
        status=timing(t,b.E.o[gid],first.get(gid),reasons.get(gid,'none'),R.BE.STEP,b.prof['cancel'],b.prof['pending'])
        v=(1-2*side)*(z[4]-price)*q/100
        x=cycles[kind,status]
        x['shares']+=q
        x['gains_usd']+=max(0.,v)
        x['losses_usd']+=min(0.,v)
        x['net_usd']+=v
    assert len(reasons)==ref['all_pulls'] and len(b.E.o)==ref['posts']
    assert len(policy.trims)==ref['inventory_pulls'] and list(reasons.values()).count('pressure')==ref['pressure_pulls']
    assert abs(sum(x['shares'] for x in cycles.values())-ref['closed'])<1e-6
    assert abs(sum(x['net_usd'] for x in cycles.values())-ref['turnover'])<1e-6
    assert abs(b.E.value(100*payout)[2]/100-ref['settle'])<1e-6
    assert abs(sum(f[3] for f in fills)-ref['shares'])<1e-6
    return ([dict(mid=mid,reason=k,flags=f,**v) for (k,f),v in pulls.items()],
            [dict(mid=mid,kind=k,status=s,**v) for (k,s),v in cycles.items()])


def run():
    day=R.EVAL
    ref=pd.read_csv(S.HERE/'markets.csv').query("arm=='linear'").set_index('mid')
    cref=pd.read_csv(C.HERE/'markets.csv').query("lag_ms==0 and entry!='all'").set_index(['mid','entry'])
    fit=S.read(R.HERE/'fit_v2.json')
    idx={r['mid']:r['tape_id'] for r in S.read(R.CENSUS)['rows'] if r['day']==day and r['status']=='complete'}
    assert R.sha(R.CENSUS)==fit['census_sha'] and set(idx)==set(ref.index) and len(ref)==70
    reg=R.MM.registry(day).set_index('mid'); rg=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
    src=R.MM.KP.Tape(); off=src.offset(day); edges=R.MM.TR.T.P.frozen_edges()
    rows=[]; detail=[]
    for hour,group in reg.groupby('hour'):
        ids=[m for m in group.index if m in idx]
        if not ids: continue
        for m,st,path,info in R.MM.worlds(day,ids,src,off,reg):
            ident=R.MM.ident_of(day,m,1000.*reg.loc[m,'ws'],info,path)
            assert st=='complete' and ident['tape_id']==idx[m]
            on=R.triggers(R.MM.make_cache(info['t_start'],info['end_rel']),edges)
            b=S.Branch(path,info['end_rel'],ident)
            pol=S.Smooth(b,on,fit['cells'],'linear',{},[])
            for k in range(R.LB.n_ticks(b)):
                R.LB.step(b,k,pol)
            R.LB.finish(b)
            a,d=collect(b,pol,ref.loc[m],R.VM.V.payout_of(rg.loc[m]),m)
            rows.extend(a); detail.extend(d)
        print('reconciled hour',hour,flush=True)
    a=pd.DataFrame(rows).fillna(0); d=pd.DataFrame(detail).fillna(0)
    assert set(a.mid)==set(d.mid)==set(ref.index)
    by=d.groupby(['mid','kind']).sum(numeric_only=True)
    assert set(by.index)==set(cref.index)
    for k,r in by.iterrows():
        assert abs(r.net_usd-cref.loc[k,'net']/100)<1e-6 and abs(r.shares-cref.loc[k,'shares'])<1e-6
    a.to_csv(HERE/'requests.csv',mode='x',index=False)
    d.to_csv(HERE/'cycles.csv',mode='x',index=False)
    R.write(HERE/'result.json',dict(markets=len(ref),source=R.sha(Path(__file__)),
            requests=a.drop(columns='mid').groupby(['reason','flags']).sum().reset_index().to_dict('records'),
            cycles=d.drop(columns='mid').groupby(['kind','status']).sum().reset_index().to_dict('records')))


def test():
    o=dict(submit=0,act=58,req=150,remove=187)
    cases=[(58,None,'before_pending_effect'),(95,None,'pending_only'),(136,100,'pending_only'),
           (137,100,'timely_live_keep'),(150,100,'timely_live_keep'),(151,100,'in_flight_pressure'),
           (149,None,'no_timely_valid_keep')]
    for t,keep,status in cases:
        assert timing(t,o,keep,'pressure',50,37,95)==status
    try: timing(187,o,100,'pressure',50,37,95)
    except AssertionError: pass
    else: raise ValueError('impossible fill at removal passed')
    print('six boundaries, missing review and impossible-fill control passed')


if __name__=='__main__':
    {'run':run,'test':test}[sys.argv[1]]()
