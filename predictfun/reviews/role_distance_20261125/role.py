"""Frozen role-specific distance rule. Protocol and economic gate in protocol.md."""
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import math
import sys
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'distance_retention_20261125'
spec=spec_from_file_location('previous_distance_comparison',BASE/'replay.py')
D=module_from_spec(spec)
spec.loader.exec_module(D)
S,R,C=D.S,D.R,D.C


def closing_qty(b,side,gid,g):
    """Client-known unallocated opposing inventory, closing reservations in priority order."""
    pos=b.reserve(g)[0]
    room=max(0.,(2*side-1)*pos)
    ids=b.unclosed[side]
    for other in sorted(ids,key=lambda i:(b.E.o[i]['req'] is None,b.E.o[i]['submit'],i)):
        remaining=max(0.,b.E.o[other]['size']-b.known_filled(other,g))
        if other==gid:
            return min(room,remaining),remaining
        room=max(0.,room-remaining)
    raise ValueError('generation absent from known reservations')


class RoleBranch(S.Branch):
    def __init__(self,path,end,ident):
        super().__init__(path,end,ident)
        self.retained=set()
        self.released={}

    def structural(self,g):
        frame=self.E.f[self.E.known]
        for side,ids in self.unclosed.items():
            best,opp=frame[2+side*10],frame[2+(1-side)*10]
            for gid in ids:
                o=self.E.o[gid]
                if o['req'] is not None or g<=o['submit']:
                    continue
                behind=(1-2*side)*(best-o['price'])
                flags=dict(no_book=bool(best<0),distance=bool(behind>R.BE.HOLD+o['rung']),
                           cross=bool(o['price']>=opp if side==0 else o['price']<=opp))
                if D.keep_distance(flags):
                    close,remaining=closing_qty(self,side,gid,g)
                    if close<=R.EPS:
                        self.retained.add(gid)
                        continue
                    self.released[gid]='mixed' if close<remaining-R.EPS else 'closing'
                if any(flags.values()):
                    self.E.request(gid,g,'structural')
                    self.struct_log.append(dict(t=float(g),gid=gid,side=side,rung=float(o['rung']),price=float(o['price']),
                         best=float(best),opp=float(opp),behind=float(behind),primary=next(k for k in R.LB.STRUCT_FLAGS if flags[k]),
                         mode=self.mode,effect=float(o['remove']),ack=float(o['remove']+R.BE.CANCEL_ACK),
                         received=float(frame[1]),frame=int(self.E.known),**flags))


def replacements(events,orders,released):
    """Uniquely attributable vacancies only. Multiple available same-side vacancies are ambiguous."""
    desired=[set(),set()]
    pending=[None,None]
    status={}
    links={}
    for t,kind,i,_ in events:
        side=orders[i]['side']
        active=desired[side]
        old=pending[side]
        if kind=='submit':
            if old is not None:
                assert len(active)==1
                status[old]='reposted'
                links[old]=i
                pending[side]=None
            assert len(active)<2 and i not in active
            active.add(i)
        elif i in active and (kind.endswith('_request') or kind.startswith('closed_')):
            if old is not None:
                status[old]='ambiguous'
                pending[side]=None
            if i in released and kind=='structural_request':
                status[i]='ambiguous' if len(active)<2 else 'no_repost_by_end'
                if len(active)==2:
                    pending[side]=i
            active.remove(i)
    assert set(status)==set(released) and len(set(links.values()))==len(links)
    return status,links


def one(path,info,ident,on,model,payout):
    b=RoleBranch(path,info['end_rel'],ident)
    pol=S.Smooth(b,on,model,'linear',{},[])
    both=0
    for k in range(R.LB.n_ticks(b)):
        R.LB.step(b,k,pol)
        g=k*R.BE.STEP
        both+=all(any(not b.E.o[i]['refused'] and b.E.o[i]['act']<=g<b.E.o[i]['remove'] and
                      b.E.o[i]['rem']>R.EPS for i in b.unclosed[s]) for s in (0,1))
    R.LB.finish(b)
    fills=b.E.fills
    fifo=R.X.fifo(fills,100*payout)
    _,pos,value=b.E.value(100*payout)
    reasons,flags,_=D.A.requests(b,pol)
    assert {i for i,f in flags.items() if f=='distance'}==set(b.released)
    status,links=replacements(b.E.events,b.E.o,b.released)
    successors=set(links.values())
    closes=Counter()
    r=dict(settle=value/100,residual=fifo['residual_c']/100,turnover=fifo['completed_c']/100,closed=fifo['matched_shares'],
           end_pos=pos,abs_end=abs(pos),shares=sum(f[3] for f in fills),both=both/R.LB.n_ticks(b),posts=len(b.E.o),
           all_pulls=len(reasons),pressure_pulls=list(reasons.values()).count('pressure'),inventory_pulls=len(pol.trims),
           distance_only=len(b.released),unique_retained=len(b.retained))
    for kind in ('same_price','through_price'):
        r[kind+'_sh']=r[kind+'_usd']=0.
    for opened,z,q in C.inventory_pairs(fills):
        kind=C.fill_kind(opened,b.E.tr)
        r[kind+'_sh']+=q
        r[kind+'_usd']+=(1-2*opened[2])*(z[4]-opened[4])*q/100
        if z[1] in successors:
            closes[z[1]]+=q
    D.validate(r)
    counts=Counter(status.values())
    trace=dict(mixed=sum(x=='mixed' for x in b.released.values()),released=len(b.released),unique_reposts=counts['reposted'],
          ambiguous=counts['ambiguous'],no_repost=counts['no_repost_by_end'],accepted=sum(not b.E.o[j]['refused'] for j in successors),
          closer=sum((1-2*b.E.o[i]['side'])*(b.E.o[j]['price']-b.E.o[i]['price'])>0 for i,j in links.items()),
          closing_successors=len(closes),closed_sh=sum(closes.values()))
    assert trace['closing_successors']<=trace['accepted']<=len(successors)
    assert sum(counts.values())==len(b.released)
    return r,trace


def gate(ref,role):
    return dict(positive_settlement=role['settle']>0,positive_turnover=role['turnover']>0,
                improved_settlement=role['settle']>ref['settle'],improved_turnover=role['turnover']>ref['turnover'],
                within_inventory_limit=role['mean_abs_end']<=ref['mean_abs_end']+10)


def run():
    day=R.EVAL
    previous=S.read(BASE/'result.json')
    assert R.sha(BASE/'markets.csv')==previous['market_hash']=='91af28474a8dd436320231c6ec2849ff235208c0ebe66d8580e18ebb2f017848'
    old=pd.read_csv(BASE/'markets.csv')
    idx={r['mid']:r['tape_id'] for r in S.read(R.CENSUS)['rows'] if r['day']==day and r['status']=='complete'}
    assert len(idx)==70 and set(idx)==set(old.mid) and len(old)==140 and not old.duplicated(['mid','arm']).any()
    ref=old.set_index(['mid','arm'])
    fit=S.read(R.HERE/'fit_v2.json')
    assert R.sha(R.CENSUS)==fit['census_sha']
    src=R.MM.KP.Tape(); off=src.offset(day); edges=R.MM.TR.T.P.frozen_edges()
    reg=R.MM.registry(day).set_index('mid'); rg=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
    rows=[]; traces=[]; parity=[]
    for hour,group in reg.groupby('hour'):
        ids=[m for m in group.index if m in idx]
        if not ids: continue
        for m,st,path,info in R.MM.worlds(day,ids,src,off,reg):
            ident=R.MM.ident_of(day,m,1000.*reg.loc[m,'ws'],info,path)
            assert st=='complete' and ident['tape_id']==idx[m]
            on=R.triggers(R.MM.make_cache(info['t_start'],info['end_rel']),edges)
            args=(path,info,ident,on,fit['cells'],R.VM.V.payout_of(rg.loc[m]))
            if m==min(ids):
                for arm in ('reference','retain'):
                    c=D.simulate(*args,arm=='retain')
                    assert set(c)==set(ref.columns) and all(abs(v-ref.loc[(m,arm),k])<1e-6 for k,v in c.items()),(m,arm)
                parity.append(m)
            r,z=one(*args)
            rows.append(dict(mid=m,arm='role',**r)); traces.append(dict(mid=m,**z))
        print('completed hour',hour,flush=True)
    df=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
    assert len(df)==210 and set(df.query("arm=='role'").mid)==set(idx) and not df.duplicated(['mid','arm']).any()
    df.to_csv(HERE/'markets.csv',mode='x',index=False)
    tr=pd.DataFrame(traces); tr.to_csv(HERE/'replacement.csv',mode='x',index=False)
    arms={}
    for arm,g in df.groupby('arm'):
        arms[arm]={k:float(g[k].sum()) for k in g.select_dtypes('number') if k not in ('mid','abs_end','end_pos','both')}
        arms[arm].update(mean_abs_end=g.abs_end.mean(),both=g.both.mean(),within_ten=int((g.abs_end<=10).sum()),
                         positive_markets=int((g.settle>0).sum()),c_per_matched=100*g.turnover.sum()/g.closed.sum())
    for arm in ('reference','retain'):
        assert all(math.isclose(arms[arm][k],v,abs_tol=1e-6) for k,v in previous['arms'][arm].items())
    tests=gate(arms['reference'],arms['role'])
    contrasts={}
    p=df.pivot(index='mid',columns='arm',values='settle')
    for arm in ('reference','retain'):
        delta=p.role-p[arm]
        contrasts[arm]=dict(delta=delta.sum(),better=int((delta>1e-6).sum()),worse=int((delta< -1e-6).sum()),
                           equal=int((delta.abs()<=1e-6).sum()),excluding_largest_gain=delta.sum()-delta.max())
    R.write(HERE/'result.json',dict(arms=arms,criteria=tests,pass_all=all(tests.values()),paired=contrasts,
          ceiling=arms['reference']['mean_abs_end']+10,trace=tr.drop(columns='mid').sum().to_dict(),
          source=R.sha(Path(__file__)),controls=previous['market_hash'],parity=parity,market_hash=R.sha(HERE/'markets.csv'),
          replacement_hash=R.sha(HERE/'replacement.csv')))


def test():
    from types import SimpleNamespace
    import numpy as np
    for s in (0,1):
        other=1-s
        b=RoleBranch.__new__(RoleBranch); b.released={}; b.retained=set(); b.struct_log=[]; b.mode=None
        b.unclosed={s:[0,1,2],other:[]}
        frame=np.zeros((1,22)); frame[0,2]=50; frame[0,12]=51
        orders=[dict(price=61 if s else 40,submit=i*5,req=20 if i==2 else None,rung=0,size=q,side=s) for i,q in enumerate([70,70,30])]
        def request(i,t,why): orders[i].update(req=t,remove=t+37)
        b.E=SimpleNamespace(f=frame,known=0,o=orders,request=request)
        b.known_filled=lambda i,t:10 if i==2 else 0
        b.reserve=lambda g:((2*s-1)*60,0.,0.)
        assert closing_qty(b,s,0,100)==(40,70) and closing_qty(b,s,1,100)==(0,70)
        b.structural(100)
        assert b.released=={0:'mixed'} and b.retained=={1} and orders[0]['req']==100
        orders[1]['price']=48 if s else 54
        b.structural(150)
        assert orders[1]['req']==150 and b.struct_log[-1]['cross']
    os=[dict(side=0)]*7
    seq=[('submit',0),('submit',1),('structural_request',0),('submit',2),('closed_removed',0),
         ('structural_request',1),('policy_request',2),('submit',3),('structural_request',3),
         ('submit',4),('submit',5),('structural_request',5)]
    states,links=replacements([(t,k,i,{}) for t,(k,i) in enumerate(seq)],os,{0,1,3,5})
    assert links=={0:2} and states=={0:'reposted',1:'ambiguous',3:'ambiguous',5:'no_repost_by_end'}
    good=dict(settle=10,turnover=12,mean_abs_end=20)
    ref=dict(settle=-1,turnover=-2,mean_abs_end=10)
    assert all(gate(ref,good).values()) and not all(gate(ref,dict(good,mean_abs_end=21)).values())
    print('bid/ask role, pending and partial reservations, crossing, unique vacancy and gate controls passed')


if __name__=='__main__':
    {'test':test,'run':run}[sys.argv[1]]()
