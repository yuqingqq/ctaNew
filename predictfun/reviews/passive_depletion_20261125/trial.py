"""Single-candidate passive depletion replay; protocol.md specifies the comparison."""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent/'smooth_unwind_20261125'))
import smooth as S

R = S.R
OUT = R.X.DATA/'passive_depletion_v2_20261125'
HOUR = 3600000
EPS = 1e-7


@lru_cache(maxsize=2)
def raw_hour(h):
    date = pd.Timestamp(h*HOUR, unit='ms', tz='UTC')
    file = ROOT/'data/mm_hf/raw/bookTicker/BTCUSDT'/date.strftime('%Y%m%d_%H.csv.gz')
    if not file.exists():
        return None
    table = pd.read_csv(file, header=None, usecols=[0,4,5,6,7])
    clock = table[0].to_numpy(dtype=np.int64)
    # Keep receipt chronology. Never sort by exchange timestamp.
    assert np.all(np.diff(clock)>=0)
    return clock, table.iloc[:,1:].to_numpy(), R.sha(file)


def window(start, end):
    parts, hashes = [], {}
    for h in range(int((start-2500)//HOUR), int((end+2000)//HOUR)+1):
        raw = raw_hour(h)
        if raw is None:
            continue
        t,q,digest = raw
        lo,hi = np.searchsorted(t, [(start-1000)*1e6,(end+1)*1e6])
        if hi>lo:
            parts.append((t[lo:hi],q[lo:hi]))
            hashes[str(h)] = digest
    if not parts:
        return np.array([]),np.empty((0,4)),hashes
    t = np.concatenate([p[0] for p in parts])
    order = np.argsort(t,kind='stable')
    return t[order]/1e6,np.concatenate([p[1] for p in parts])[order],hashes


def signal(t, book, grid):
    out = np.full((2,len(grid)),-1,dtype=np.int8)
    if not len(t):
        return out
    end = np.searchsorted(t, grid,'right')-1
    base = np.searchsorted(t,grid-300,'right')-1
    a,b = np.maximum(base,0),np.maximum(end,0)
    valid = np.all(np.isfinite(book) & (book>0),axis=1) & (book[:,0]<book[:,2])
    bad = np.r_[0,np.cumsum(~valid)]
    gap = np.cumsum(np.r_[False,np.diff(t)>300])
    for side in (0,1):
        price,qty = book[:,2*side],book[:,2*side+1]
        epoch = np.cumsum(np.r_[False,price[1:]!=price[:-1]])
        ok = (base>=0) & (grid-300-t[a]<=300) & (grid-t[b]<=300) & (bad[b+1]==bad[a]) & (gap[b]==gap[a])
        out[side,ok] = ((epoch[a]==epoch[b]) & (qty[b]<=0.5*qty[a]))[ok]
    return out


def closing_room(v,gid=None,retiring=()):
    room = max(0.,(2*v['side']-1)*v['position'])
    # Retiring reservations first, then standing orders in submission order.
    for o in sorted(v['orders'],key=lambda o:(o['state']!='cancel_pending' and o['gid'] not in retiring,o['submit'],o['gid'])):
        if o['gid']==gid:
            break
        room -= o['size']-o['known_filled']
    return max(0.,room)


class Guard(S.Smooth):
    def __init__(self,b,flags,model,dep,enabled,mid):
        super().__init__(b,flags,model,'linear',{},[])
        self.dep,self.enabled = dep,enabled
        self.count = Counter()
        self.flagged,self.control,self.seen = {},{},Counter()
        self.rng = random.Random(mid)
        self.age_ms = 0.

    def observe(self,v,gid,rung,price,q,kind,flag,o=None):
        if self.enabled or flag<0:
            return
        # One control time per generation selected uniformly among eligible OFF callbacks.
        if not flag:
            self.seen[gid]+=1
            if self.rng.randrange(self.seen[gid]):
                return
        elif gid in self.flagged:
            return
        s,g = v['side'],v['t']
        if o is None:
            ahead = self.b.E.shown(self.b.E.known,s,price)
        else:
            ahead = self.b.E.o[gid]['ahead']  # venue-side matching diagnostic only
        qb = -1 if not math.isfinite(ahead) else 0 if ahead<=EPS else 1 if ahead<=q else 2
        key = (kind,s,int(rung),int((self.b.end-g)//60000),qb,int(q>110))
        row = (g,key)
        (self.flagged if flag else self.control)[gid] = row

    def live(self,v,o):
        action,why = super().live(v,o)
        f = int(self.dep[v['side'],self.b.k])
        self.count['live_reviews']+=1
        if action!='KEEP':
            self.count['old_pull_and_on']+=int(f==1)
            return action,why
        q=o['size']-o['known_filled']
        retiring={x['gid'] for x in v['orders'] if self.b.E.o[x['gid']]['req'] is not None}
        room=closing_room(v,o['gid'],retiring)
        if room>EPS:
            if f==1:
                self.count['protected_live_mixed' if room<q-EPS else 'protected_live_reduce']+=1
            return action,why
        self.count['live_eligible']+=1
        self.count['live_unknown']+=int(f<0)
        self.observe(v,o['gid'],o['rung'],o['price'],q,'live',f,o)
        if f==1:
            self.count['live_flagged_ticks']+=1
            if self.enabled:
                self.age_ms += v['t']-o['act']
                return 'PULL','depletion'
        return action,why

    def slot(self,v,rung,price):
        prior_open = self.opened,self.first_side
        action,q,why = super().slot(v,rung,price)
        f=int(self.dep[v['side'], self.b.k])
        if action!='POST':
            self.count['old_slot_pressure_and_on']+=int(why=='pressure' and f==1)
            return action,q,why
        room=closing_room(v)
        if room>=q-EPS:
            self.count['protected_slot_reduce']+=int(f==1)
            return action,q,why
        self.count['slot_eligible']+=1
        self.count['slot_unknown']+=int(f<0)
        if room<=EPS:
            self.observe(v,len(self.b.E.o),rung,price,q,'slot',f)
        if f!=1:
            return action,q,why
        self.count['slot_flagged_ticks']+=1
        reducing=math.floor(room+EPS)
        if room>EPS and reducing*(price if v['side']==0 else 100-price)<100-EPS:
            self.count['mixed_min_exempt']+=1
            return action,q,why
        if not self.enabled:
            return action,q,why
        if room<=EPS:
            self.opened,self.first_side = prior_open
            self.proposals.pop()
            self.count['withheld']+=1
            return 'WAIT',None,'depletion'
        g,s,capacity,level,_,unit=self.proposals[-1]
        reserved=v['A'] if s else v['B']
        self.proposals[-1]=(g,s,capacity,level,max(0.,(1-2*s)*v['position']+reserved+reducing),unit)
        self.count['clipped']+=1
        return action,reducing,'depletion_size'


def labels(branch,states,payout):
    groups=defaultdict(list)
    for t,i,s,q,p,_ in branch.E.fills:
        groups[i].append((t,s,q,p))
    frames=branch.E.f
    rows=[]
    for gid,(g,key) in states.items():
        o=branch.E.o[gid]
        live=key[0]=='live'
        effect=g+(branch.prof['cancel'] if live else branch.prof['place'])
        assert live or o['submit']==g
        remaining_end=min(branch.end,o['remove'])
        if o['rem']<=EPS and groups[gid]:
            remaining_end=min(remaining_end,groups[gid][-1][0])
        row=Counter()
        row['remaining_rest_ms']=max(0.,remaining_end-effect) if live and not o['refused'] else 0.
        for t,s,q,p in groups[gid]:
            if t<g or (live and t==g):
                row['already_matched_shares']+=q
                continue
            if t<effect:
                row['inflight_shares']+=q
                continue
            row['post_effect_shares']+=q
            amt=(1-2*s)*q*(100*payout-p)/100
            row['settle']+=amt
            row['positive_settle']+=max(0.,amt)
            row['negative_settle']+=min(0.,amt)
            for h in (1,5,30):
                target=t+h*1000
                j=np.searchsorted(frames[:,0],target,'right')-1
                if target>branch.end or j<0 or target-frames[j,0]>1000 or not (0<=frames[j,2]<frames[j,12]<=100):
                    row[f'censored_sh_{h}']+=q
                    continue
                mid=(frames[j,2]+frames[j,12])/2
                value=(1-2*s)*q*(mid-p)/100
                row[f'valid_sh_{h}']+=q
                row[f'value_{h}']+=value
                row[f'positive_{h}']+=max(0.,value)
                row[f'negative_{h}']+=min(0.,value)
        rows.append((gid,key,dict(row)))
    return rows


def simulate(path,info,ident,flags,model,payout,dep,enabled):
    b=S.Branch(path,info['end_rel'],ident)
    pol=Guard(b,flags,model,dep,enabled,ident['instrument_id'])
    both=0
    for k in range(R.LB.n_ticks(b)):
        R.LB.step(b,k,pol)
        g=k*R.BE.STEP
        both+=all(any(not b.E.o[i]['refused'] and b.E.o[i]['act']<=g<b.E.o[i]['remove']
                      and b.E.o[i]['rem']>R.EPS for i in b.unclosed[s]) for s in (0,1))
    R.LB.finish(b)
    fifo=R.X.fifo(b.E.fills,100*payout)
    _,pos,settle=b.E.value(100*payout)
    closed,events=S.close_stats(b.E.fills)
    shares=sum(f[3] for f in b.E.fills)
    assert abs(2*closed+abs(pos)-shares)<EPS and abs(settle-fifo['completed_c']-fifo['residual_c'])<EPS
    assert len(pol.proposals)==len(b.E.o)
    assert abs(closed-fifo['matched_shares'])<EPS
    assert all(b.E.o[gid]['act']<=t<b.E.o[gid]['remove'] for t,gid,*_ in b.E.fills)
    result=dict(settle=settle/100,turnover=fifo['completed_c']/100,residual=fifo['residual_c']/100,
                closed=closed,close_events=events,shares=shares,end_pos=pos,abs_end=abs(pos),posts=len(b.E.o),
                both=both/R.LB.n_ticks(b),all_pulls=sum(o['req'] is not None for o in b.E.o),
                pressure_pulls=sum(c[7:]==('PULL','pressure') for c in b.cmd_log),inventory_pulls=len(pol.trims))
    diagnostics=dict(counts=dict(pol.count),age_ms=pol.age_ms)
    if not enabled:
        diagnostics.update(flagged=labels(b,pol.flagged,payout),controls=labels(b,pol.control,payout))
    return result,diagnostics


def replay():
    day=R.EVAL
    src=R.MM.KP.Tape()
    ref=pd.read_csv(S.HERE/'markets.csv').query("arm=='linear'").set_index('mid')
    model=S.read(R.HERE/'fit_v2.json')
    assert R.sha(R.CENSUS)==model['census_sha']
    idx={r['mid']:r for r in S.read(R.CENSUS)['rows'] if r['day']==day and r['status']=='complete'}
    assert set(idx)==set(ref.index) and len(idx)==70
    reg=R.MM.registry(day).set_index('mid')
    rg=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
    off=src.offset(day)
    edges=R.MM.TR.T.P.frozen_edges()
    for h,group in reg.groupby('hour'):
        ids=[m for m in group.index if m in idx]
        if not ids:
            continue
        for m,st,path,info in R.MM.worlds(day,ids,src,off,reg):
            ident=R.MM.ident_of(day,m,1000.*reg.loc[m,'ws'],info,path)
            assert st=='complete' and idx[m]['tape_id']==ident['tape_id']
            t0=info['t_start']
            t,book,hashes=window(t0,t0+info['end_rel'])
            ticks=t0+R.BE.STEP*np.arange(math.ceil(info['end_rel']/R.BE.STEP))
            dep=signal(t,book,ticks)
            flags=R.triggers(R.MM.make_cache(t0,info['end_rel']),edges)
            arms,diag={},{}
            for arm in ('reference','guard'):
                arms[arm],diag[arm]=simulate(path,info,ident,flags,model['cells'],R.VM.V.payout_of(rg.loc[m]),dep,arm=='guard')
                if arm=='reference':
                    assert all(abs(v-ref.loc[m,k])<1e-6 for k,v in arms[arm].items()),m
            R.write(OUT/f'{m}.json',dict(mid=m,arms=arms,diag=diag,hashes=hashes,tape=ident['tape_id'],
                    source=R.sha(Path(__file__)),unknown=(dep<0).sum(axis=1).tolist(),ticks=len(ticks)))
        print('completed UTC hour',h,flush=True)


def add_values(rows):
    total=Counter()
    for _,_,values in rows:
        total.update(values)
    return total


def score():
    ref=pd.read_csv(S.HERE/'markets.csv').query("arm=='linear'").set_index('mid')
    census={r['mid']:r for r in S.read(R.CENSUS)['rows'] if r['day']==R.EVAL and r['status']=='complete'}
    rows=[]; gross=Counter(); matched=Counter(); counts=defaultdict(Counter)
    draws=np.zeros(500)
    rng=random.Random(20261125)
    nflag=nmatch=unknown=ticks=0
    age=0.
    inputs={}
    files=sorted(OUT.glob('*.json'))
    assert set(int(f.stem) for f in files)==set(ref.index)==set(census)
    for f in files:
        d=S.read(f); m=d['mid']
        assert d['source']==R.sha(Path(__file__)) and int(f.stem)==m and d['tape']==census[m]['tape_id']
        assert all(abs(v-ref.loc[m,k])<1e-6 for k,v in d['arms']['reference'].items())
        for h,sha in d['hashes'].items():
            assert inputs.setdefault(h,sha)==sha
        unknown+=sum(d['unknown']); ticks+=2*d['ticks']
        for arm,metrics in d['arms'].items():
            rows.append(dict(mid=m,arm=arm,**metrics))
            counts[arm].update(d['diag'][arm]['counts'])
        age+=d['diag']['guard']['age_ms']
        flagged=d['diag']['reference']['flagged']
        controls=d['diag']['reference']['controls']
        gross.update(add_values(flagged)); nflag+=len(flagged)
        flagged_by=defaultdict(list); controls_by=defaultdict(list)
        for row in flagged: flagged_by[tuple(row[1])].append(row)
        for row in controls: controls_by[tuple(row[1])].append(row)
        for key,fs in sorted(flagged_by.items()):
            gids={r[0] for r in fs}
            cs=[r[2].get('settle',0) for r in controls_by[key] if r[0] not in gids]
            n=min(len(cs),len(fs))
            if n:
                matched.update(add_values(rng.sample(fs,n)))
                for j in range(len(draws)):
                    draws[j]+=sum(rng.sample(cs,n))
                nmatch+=n
    df=pd.DataFrame(rows)
    df.to_csv(HERE/'markets.csv',index=False)
    sums={}
    for arm,g in df.groupby('arm'):
        sums[arm]={k:float(g[k].sum()) for k in ['settle','turnover','residual','closed','posts','all_pulls','shares','close_events']}
        sums[arm].update(cents_per_matched=100*g.turnover.sum()/g.closed.sum(),mean_end_abs=g.abs_end.mean(),both=g.both.mean())
    d=df.pivot(index='mid',columns='arm',values='settle')
    delta=d.guard-d.reference
    a,b=sums['guard'],sums['reference']
    gate=a['settle']>0 and a['turnover']>b['turnover']
    R.write(HERE/'results.json',dict(arms=sums,market_count=len(d),better=int((delta>1e-6).sum()),worse=int((delta< -1e-6).sum()),
        equal=int((delta.abs()<=1e-6).sum()),delta=float(delta.sum()),without_largest_gain=float(delta.sum()-delta.max()),
        largest_gain_mid=int(delta.idxmax()),without_largest_absolute=float(delta.sum()-delta.loc[delta.abs().idxmax()]),
        decision='follow_up' if gate else 'stop',counts=counts,age_at_pressure_depletion_pull_ms_sum=age,
        status=S.read(S.HERE/'result.json')['status'],signal_unknown_side_ticks=unknown,side_ticks=ticks,
        unique_first_flags=nflag,flagged_reference_exposure=gross,matched_n=nmatch,matched_exposure=matched,
        null_settlement_draw_mean=draws.mean(),null_p05=np.percentile(draws,5),percentile=(draws<=matched['settle']).mean(),
        raw_source_hashes=inputs,source=R.sha(Path(__file__))))
    print('scored',len(d),gate,flush=True)


def test():
    t=np.array([0.,180.,300.,350.,400.])
    book=np.array([[100,10,101,20],[100,10,101,20],[100,4,101,20],[100,10,101,8],[100,10,102,5]],float)
    observed=signal(t,book,np.array([0.,300.,350.,400.]))
    assert observed.tolist()==[[-1,1,0,0],[-1,0,1,0]]
    # A receipt following the decision can never change its flag.
    assert np.array_equal(observed[:,:3],signal(t[:-1],book[:-1],np.array([0.,300.,350.])))
    bad=book.copy(); bad[1,0]=99
    assert signal(t,bad,np.array([300.]))[0,0]==0
    assert signal(t,book,np.array([1200.]))[0,0]==-1
    orders=[dict(gid=i,submit=i,size=q,known_filled=0,state=k) for i,(q,k) in enumerate([(70,'live'),(30,'live'),(20,'cancel_pending')])]
    v=dict(side=1,position=60,orders=orders)
    assert [closing_room(v,i) for i in [0,1,None]]==[40,0,0]
    assert closing_room(dict(v,position=-60),0)==0
    # A locally requested cancellation before post activation still retires first.
    pending=[dict(o,state='post_pending') if o['gid']==2 else o for o in orders]
    assert closing_room(dict(v,orders=pending),0,{2})==40 and closing_room(dict(v,orders=pending),0)==60
    # Treatment must override a purely adding ask and keep a mixed ask; parent pressure wins.
    from types import SimpleNamespace
    bb=SimpleNamespace(k=0,E=SimpleNamespace(o=[dict(req=None),dict(req=None),dict(req=0)]))
    p=Guard.__new__(Guard)
    p.b=bb; p.arm='C'; p.on=[[False],[False]]; p.dep=np.array([[0],[1]]); p.enabled=True
    p.count=Counter(); p.age_ms=0
    v.update(t=50)
    for o in orders: o.update(rung=0,price=60,act=10)
    assert p.live(v,orders[0])[0]=='KEEP' and p.live(v,orders[1])==('PULL','depletion')
    p.on[1][0]=True
    assert p.live(v,orders[1])==('PULL','pressure')
    print('synthetic clock, direction, precedence, reservations and causal checks passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['test','replay','score'])
    {'test':test,'replay':replay,'score':score}[parser.parse_args().stage]()
