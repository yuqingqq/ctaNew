"""Paired selective confirmation replay on accepted tapes, with no venue-side inputs to decisions."""
from collections import Counter, defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch
import json
import sys
import numpy as np
import pandas as pd
import local

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=HERE.parent/'book_alert_replay_20261128'
sys.path.insert(0,str(OLD))
from book_signal import DATA, DAYS, OUT, digest
sys.path.insert(0,str(HERE.parent/'pressure_market_sharpe_20261128'))
from diagnose import load_day, metrics
spec=spec_from_file_location('event_control', HERE.parent/'event_pressure_cancel_20261127/run.py')
ev=module_from_spec(spec)
spec.loader.exec_module(ev)
BASE=DATA/'event_pressure_nov24_26'
KEYS=('settle','turnover','residual','closed','shares','posts','requests','abs_end','reposts')
HASHES={ROOT/p:h for h,p in (line.split() for line in (HERE/'inputs.sha256').read_text().splitlines())}


def publications(day, mids):
    p=DATA/'step1'/day/'pend_inc.parquet'
    assert digest(p)==HASHES[p]
    inc=pd.read_parquet(p,columns=['mid','t_ms','recv_ns','ptick','inc'],filters=[('mid','in',mids)])
    assert inc.inc.gt(0).all()
    keys=['mid','t_ms','recv_ns']
    books=pd.concat([pd.read_parquet(f,columns=keys,filters=[('mid','in',mids)])
                     for f in sorted((BASE/'step1'/day/'books').glob('part_*.parquet'))],ignore_index=True)
    unique=books.loc[~books.duplicated(keys[:2],keep=False)]
    joined=inc.merge(unique,on=keys,how='inner',validate='many_to_one')
    print(day,'positive increments',len(inc),'unique receipt joins',len(joined),flush=True)
    return {mid:g[['t_ms','ptick']].drop_duplicates().to_numpy() for mid,g in joined.groupby('mid')}


def get_local(fr,start,rows):
    by_price=defaultdict(list)
    if len(rows):
        src=fr[:,0]
        rel=rows[:,0]-start
        ix=src.searchsorted(rel)
        keep=(ix>0)&(ix<len(fr))&(src[np.minimum(ix,len(fr)-1)]==rel)
        for j,price in zip(ix[keep],rows[keep,1]):
            by_price[int(price)].append(fr[j,1])
    return by_price


def timely(branch, fills, per_price, feed, ev):
    # Neutral-path exposure diagnostic, after original decisions and notices are known.
    kept={(c[0],c[3]) for c in branch.cmd_log if c[2]=='live' and c[7]=='KEEP' and c[8]!='fallback'}
    groups=defaultdict(list)
    for f in fills:
        groups[f[1]].append(f)
    selected=[]
    for gid, ff in groups.items():
        o=branch.E.o[gid]
        hit=per_price.get((o['side'],o['price']))
        if hit is None:
            continue
        times,sizes=hit
        lower=times.searchsorted(o['act'])
        stop=min(ff[-1][0]-37, o['req'] if o['req'] is not None else np.inf)
        for j in range(lower,times.searchsorted(stop,'right')):
            t=times[j]
            if (t,o['side']) in feed or (o['req'] is not None and t>=o['req']):
                continue
            if t%50==0 and (t,gid) not in kept:
                continue
            known=sum(f[3] for f in ff if f[0]+ev.R.BE.FILL_NOTICE<=t)
            if sizes[j]<=o['size']-known:
                selected.extend(f for f in ff if t+37<=f[0])
                break
    return selected


def run(day):
    local.fixture()
    assert day in DAYS
    ref=load_day(day).set_index('mid')
    archive=OLD/'paired_markets.csv'
    assert digest(archive)==HASHES[archive] and digest(OLD/'model.json')==HASHES[OLD/'model.json']
    published=pd.read_csv(archive,float_precision='round_trip').query('day==@day_num',local_dict={'day_num':int(day)}).set_index('mid')
    assert set(ref.index)==set(published.index)
    pub=publications(day,list(ref.index))
    cache=OUT/day/'alerts.npz'
    assert digest(cache)==HASHES[cache]
    with np.load(cache) as alerts:
        clock,bits=alerts['clock'],alerts['mask']
    regdir=BASE/'step1'/day
    prior=BASE/day
    binding=json.loads((prior/'inputs.json').read_text())
    for rel,sha in binding.items():
        if day in rel or rel==f'btc/btc_{ev.H.prev_day(day)}.parquet':
            assert digest(BASE/rel)==sha
    R,H=ev.R,ev.H
    H.OUT,H.BTC=BASE,BASE/'btc'
    reg=pd.read_parquet(regdir/'registry.parquet').set_index('mid')
    reg['hour']=(reg.ws.astype('int64')%86400)//3600
    pressure=pd.read_parquet(BASE/f'pressure_{day}.parquet')
    k,px=pressure.k_ms.to_numpy(),pressure.mid.to_numpy()
    received=k+R.MM.B_IN
    np.testing.assert_array_equal(clock[clock.searchsorted(received)],received)
    edges=R.MM.TR.T.P.frozen_edges()
    fit=H.U.S.read(R.HERE/'fit_v2.json')['cells']
    start_day=ev.source.day_ms(day)
    with patch.multiple(R.BTC,BTC_DAYS=(H.prev_day(day),day),PAD_LO_MS=start_day-3600000,BTC_END_MS=start_day+86400000):
        btc=R.BTC.load_btc(H.BTC,day,Counter())
    cls=H.pilot['ConditionalSize']
    live,slot,cancel,finish=cls.live,cls.slot,ev.cancel_live,R.LB.finish
    dest=DATA/HERE.name/day
    dest.mkdir(parents=True,exist_ok=True)
    census=pd.read_csv(prior/f'census_{day}.csv').set_index('mid')
    out=[]
    for m,status,path,info in H.worlds(day,regdir,reg.sort_values('ws'),False):
        assert status==census.at[m,'status']
        if status!='complete':
            continue
        ws=reg.at[m,'ws']*1000.
        ident=R.MM.ident_of(day,m,ws,info,path)
        assert ident['tape_id']==ref.at[m,'tape_id']==published.at[m,'tape_id']
        with patch.object(R.MM,'bn',lambda:(k,px)):
            flags,z=H.U.S.signal_data(btc,info,ws,edges)
        start,end=info['t_start'],info['end_rel']
        grid=ev.STEP*np.arange(len(z))
        _,limits=H.strong['curves']((ws+300000-start-grid)/1000,z)
        payout=R.VM.V.payout_of(reg.loc[m])
        path=dict(path,frames=np.asarray(path['frames']),trades=np.asarray(path['trades']))
        args=(path,info,ident,flags,fit,payout)
        feed=ev.events(clock,received,px,start,end,edges,flags)
        base_events=set(feed)
        by_price=get_local(path['frames'],start,pub.get(m,[]))
        table,by_order=local.build(path['frames'],start,end,clock,bits,grid,by_price)
        extra={(t,s) for t,s in table if t%50!=0}-base_events
        branches=[]
        def observe(branch):
            branches.append(branch)
            return finish(branch)
        with patch.object(R.LB,'finish',observe):
            old,oldfills=ev.simulate(args,limits,feed)
        for key in KEYS:
            assert abs(old[key]-ref.at[m,'event_'+key])<1e-6 and abs(old[key]-published.at[m,'current_'+key])<1e-6
        tagged=timely(branches.pop(),oldfills,by_order,base_events,ev)
        counts=Counter()
        def new_cancel(b,t,s):
            if (t,s) in base_events:
                return cancel(b,t,s)
            sizes=table[t,s]
            if not any(b.E.o[i]['price'] in sizes and b.E.o[i]['req'] is None and b.E.o[i]['act']<=t for i in b.unclosed[s]):
                return []
            b.E.advance(t)
            b.notices(t)
            view=b.side_view(t,s)
            pulled=[]
            current=[o for o in view['orders'] if o['state']=='live']
            if view['book_ok']:
                for o in current:
                    if sizes.get(o['price'],np.inf)<=o['size']-o['known_filled']:
                        b.E.request(o['gid'],t,'event_pressure')
                        assert b.E.o[o['gid']]['remove']==t+37
                        pulled.append((o['gid'],t,s))
            counts['offgrid_pulls']+=len(pulled)
            counts['offgrid_one_of_two']+=int(len(pulled)==1 and len(current)==2)
            return pulled
        def new_live(pol,v,o):
            result=live(pol,v,o)
            sizes=table.get((v['t'],v['side']),{})
            if result[0]=='KEEP' and sizes.get(o['price'],np.inf)<=o['size']-o['known_filled']:
                counts['grid_pulls']+=1
                return 'PULL','local_book_confirm'
            return result
        def new_slot(pol,v,rung,price):
            opened,first=pol.opened,pol.first_side
            result=slot(pol,v,rung,price)
            if result[0]=='POST' and table.get((v['t'],v['side']),{}).get(price,np.inf)<=result[1]:
                pol.proposals.pop()
                pol.opened,pol.first_side=opened,first
                counts['waits']+=1
                return 'WAIT',None,'local_book_confirm'
            return result
        with patch.object(cls,'live',new_live),patch.object(cls,'slot',new_slot),patch.object(ev,'cancel_live',new_cancel):
            new,newfills=ev.simulate(args,limits,sorted(base_events|extra))
        row=dict(day=day,mid=int(m),tape_id=ident['tape_id'],public_price_updates=sum(map(len,by_price.values())))
        for tag in ['offgrid_pulls','offgrid_one_of_two','grid_pulls','waits']:
            row[tag]=counts[tag]
        a,b=ev.exposures(oldfills,path),ev.exposures(newfills,path)
        for tag,d in [('baseline',a),('candidate',b),('lost',a-b),('added',b-a),('timely_baseline',ev.exposures(tagged,path))]:
            row.update({tag+'_'+key:v for key,v in ev.valued(d,path,end,payout).items()})
        for tag,metric in [('current',old),('alert',new)]:
            row.update({tag+'_'+key:metric[key] for key in KEYS})
        out.append(row)
        if len(out)%20==0:
            print(day,'paired markets',len(out),flush=True)
    assert set(ref.index)=={r['mid'] for r in out} and len(out)==len(ref)
    pd.DataFrame(out).to_csv(dest/'paired.csv',index=False)
    print(day,'complete',len(out),flush=True)

if __name__=='__main__':
    for day in sys.argv[1:]:
        run(day)
