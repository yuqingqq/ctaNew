"""Single-candidate urgency replay on the consumed October 26 passive cohort."""
from collections import Counter
from pathlib import Path
import json
import math
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'role_distance_20261125'
sys.path.insert(0,str(PREV))
import role as P
D,R,S,C=P.D,P.R,P.S,P.C
TIMES=(150,120,90,60,30,10,2)
ARMS=('retain','role','urgency')


def over_budget(position,left):
    return abs(position)>S.curve(left)+R.EPS


class UrgencyBranch(P.RoleBranch):
    def structural(self,g):
        left=(self.ident['opening_ms']+300000-self.ident['start_abs_ms']-g)/1000
        if over_budget(self.reserve(g)[0],left):
            return super().structural(g)
        # Under budget, match the prior unconditional retention branch for this tick.
        # Its order tracker is a dict, whereas RoleBranch tracks a set of generation IDs.
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
                    self.retained.add(gid)
                elif any(flags.values()):
                    self.E.request(gid,g,'structural')
                    self.struct_log.append(dict(t=float(g),gid=gid,side=side,rung=float(o['rung']),price=float(o['price']),
                            best=float(best),opp=float(opp),behind=float(behind),primary=next(k for k in R.LB.STRUCT_FLAGS if flags[k]),
                            mode=self.mode,effect=float(o['remove']),ack=float(o['remove']+R.BE.CANCEL_ACK),
                            received=float(frame[1]),frame=int(self.E.known),**flags))


def one(path,info,ident,on,model,payout,arm):
    if arm=='retain':
        b=D.Branch(path,info['end_rel'],ident,True)
    else:
        cls=P.RoleBranch if arm=='role' else UrgencyBranch
        b=cls(path,info['end_rel'],ident)
    pol=S.Smooth(b,on,model,'linear',{},[])
    expiry=ident['opening_ms']+300000-ident['start_abs_ms']
    both=0; over=0; over_live=0; over_pressure=0
    for k in range(R.LB.n_ticks(b)):
        R.LB.step(b,k,pol)
        g=k*R.BE.STEP
        live=[any(not b.E.o[i]['refused'] and b.E.o[i]['act']<=g<b.E.o[i]['remove'] and
                  b.E.o[i]['rem']>R.EPS for i in b.unclosed[s]) for s in (0,1)]
        both+=all(live)
        pos=b.reserve(g)[0]
        if over_budget(pos,(expiry-g)/1000):
            s=int(pos>0)
            over+=1; over_live+=live[s]; over_pressure+=bool(on[0][s][k])
    R.LB.finish(b)
    fills=b.E.fills
    fifo=R.X.fifo(fills,100*payout)
    _,pos,value=b.E.value(100*payout)
    reasons,flags,_=D.A.requests(b,pol)
    counts=Counter(reasons.values())
    r=dict(settle=value/100,residual=fifo['residual_c']/100,turnover=fifo['completed_c']/100,closed=fifo['matched_shares'],
        end_pos=pos,abs_end=abs(pos),shares=sum(f[3] for f in fills),both=both/R.LB.n_ticks(b),posts=len(b.E.o),
        all_pulls=len(reasons),pressure_pulls=counts['pressure'],inventory_pulls=counts['inventory'],
        distance_only=sum(f=='distance' for f in flags.values()),unique_retained=len(b.retained))
    for kind in ('same_price','through_price'):
        r[kind+'_usd']=r[kind+'_sh']=0.
    for f,z,q in C.inventory_pairs(fills):
        kind=C.fill_kind(f,b.E.tr)
        r[kind+'_sh']+=q; r[kind+'_usd']+=(1-2*f[2])*(z[4]-f[4])*q/100
    D.validate(r)
    ts=np.array([f[0] for f in fills])
    inventory=np.r_[0.,np.cumsum([(1-2*f[2])*f[3] for f in fills])]
    assert abs(inventory[-1]-pos)<1e-6 and max(abs(inventory))<=1000+1e-6
    assert abs(expiry-b.end-2000)<1e-6
    stats={f'abs_{sec}s':abs(inventory[np.searchsorted(ts,expiry-sec*1000,'right')]) for sec in TIMES}
    stats.update(peak=float(max(abs(inventory))),over_s=over*R.BE.STEP/1000,over_live_s=over_live*R.BE.STEP/1000,
                 over_pressure_s=over_pressure*R.BE.STEP/1000)
    assert abs(stats['abs_2s']-r['abs_end'])<1e-6
    if arm=='urgency':
        assert set(b.released)=={i for i,f in flags.items() if f=='distance'}
        # Independently reconstruct client inventory from notices due at each request.
        for gid in b.released:
            req=b.E.o[gid]['req']
            j=np.searchsorted(ts, req-R.BE.FILL_NOTICE,'right')
            known=inventory[j]
            assert abs(known)>S.curve((expiry-req)/1000)+R.EPS and ((known>0)==(b.E.o[gid]['side']==1))
    return r,stats


def score(markets,stats,previous,source):
    groups={}
    for arm,g in markets.groupby('arm'):
        row={k:float(g[k].sum()) for k in ['settle','residual','turnover','closed','shares','posts','all_pulls','distance_only',
              'pressure_pulls','inventory_pulls','same_price_sh','through_price_sh','same_price_usd','through_price_usd']}
        row.update(c_per_matched=float(100*g.turnover.sum()/g.closed.sum()),mean_abs_end=float(g.abs_end.mean()),
                   p90_abs_end=float(g.abs_end.quantile(.9)),max_abs_end=float(g.abs_end.max()),within_ten=int((g.abs_end<=10).sum()),
                   positive_markets=int((g.settle>0).sum()),both=float(g.both.mean()))
        groups[arm]=row
    for arm in ('reference','role','retain'):
        for k in groups[arm].keys()&previous['arms'][arm].keys():
            assert math.isclose(groups[arm][k],previous['arms'][arm][k],abs_tol=1e-6)
    p=markets.pivot(index='mid',columns='arm',values='settle')
    comparisons={}
    for base in ('reference','retain','role'):
        v=p.urgency-p[base]
        comparisons[base]=dict(delta=float(v.sum()),better=int((v>1e-6).sum()),worse=int((v< -1e-6).sum()),
             equal=int((v.abs()<=1e-6).sum()),without_largest_gain=float(v.sum()-v.max()))
    tests=P.gate(groups['reference'],groups['urgency'])
    diagnostics=stats.drop(columns='mid').groupby('arm').mean().to_dict('index')
    return dict(groups=groups,criteria=tests,pass_all=all(tests.values()),comparisons=comparisons,mean_by_market=diagnostics,
                markets=len(p),source=source,gate_shares=groups['reference']['mean_abs_end']+10)


def run():
    day=R.EVAL
    previous=S.read(PREV/'result_v2.json')
    assert R.sha(PREV/'markets.csv')==previous['market_hash']=='0edc466b48a966afd82c0f99f4a6624aea70bfa53dcbaec640e109ceb93ed812'
    old=pd.read_csv(PREV/'markets.csv',float_precision='round_trip')
    control=old.set_index(['mid','arm'])
    idx={r['mid']:r['tape_id'] for r in S.read(R.CENSUS)['rows'] if r['day']==day and r['status']=='complete'}
    assert len(idx)==70 and set(idx)==set(old.mid) and len(control)==210 and control.index.is_unique
    fit=S.read(R.HERE/'fit_v2.json')
    assert R.sha(R.CENSUS)==fit['census_sha']
    reg=R.MM.registry(day).set_index('mid'); rg=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
    src=R.MM.KP.Tape(); off=src.offset(day); edges=R.MM.TR.T.P.frozen_edges()
    rows=[]; samples=[]
    for hour,group in reg.groupby('hour'):
        ids=[m for m in group.index if m in idx]
        if not ids: continue
        for m,st,path,info in R.MM.worlds(day,ids,src,off,reg):
            ident=R.MM.ident_of(day,m,1000.*reg.loc[m,'ws'],info,path)
            assert st=='complete' and ident['tape_id']==idx[m]
            on=R.triggers(R.MM.make_cache(info['t_start'],info['end_rel']),edges)
            for arm in ARMS:
                r,s=one(path,info,ident,on,fit['cells'],R.VM.V.payout_of(rg.loc[m]),arm)
                if arm=='urgency': rows.append(dict(mid=m,arm=arm,**r))
                else:
                    assert set(r)==set(control.columns) and all(abs(v-control.loc[(m,arm),k])<1e-6 for k,v in r.items()),(m,arm)
                samples.append(dict(mid=m,arm=arm,**s))
        print('reconciled opening hour',hour,flush=True)
    markets=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
    stats=pd.DataFrame(samples)
    assert len(markets)==280 and len(stats)==210 and not markets.duplicated(['mid','arm']).any() and not stats.duplicated(['mid','arm']).any()
    markets.to_csv(HERE/'markets.csv',index=False,mode='x')
    stats.to_csv(HERE/'inventory.csv',index=False,mode='x')
    out=score(markets,stats,previous,R.sha(Path(__file__)))
    out.update(controls=previous['market_hash'],markets_sha=R.sha(HERE/'markets.csv'),inventory_sha=R.sha(HERE/'inventory.csv'))
    # Serialize before publishing so a failure cannot produce a partial receipt.
    encoded=json.dumps(out,allow_nan=False,indent=2)
    with (HERE/'result.json').open('x') as f: f.write(encoded+'\n')


def test():
    from types import SimpleNamespace
    assert [S.curve(t) for t in (180,150,90,30,10)]==[1000,1000,500,0,0]
    assert not over_budget(-500,90) and over_budget(501,90) and over_budget(-501,90)
    assert not over_budget(0,10) and over_budget(.5,10)
    for side in (0,1):
        for amt,release in [(490,False),(500,False),(501,True)]:
            b=UrgencyBranch.__new__(UrgencyBranch); b.released={}; b.retained=set(); b.struct_log=[]; b.mode=None
            b.ident=dict(opening_ms=0,start_abs_ms=0); b.unclosed={side:[0,1,2],1-side:[]}
            f=np.zeros((1,22)); f[0,2]=50; f[0,12]=51
            orders=[dict(price=60 if side else 40,submit=i,req=150000 if i==2 else None,rung=0,size=q,side=side)
                    for i,q in enumerate((220,220,400))]
            def request(i,t,why): orders[i].update(req=t,remove=t+37)
            b.E=SimpleNamespace(f=f,known=0,o=orders,request=request)
            b.known_filled=lambda i,t:20 if i==2 else 0
            b.reserve=lambda t:((2*side-1)*amt,0,0)
            b.structural(210000)
            assert (orders[0]['req']==210000)==release and orders[1]['req'] is None
            orders[1]['price']=48 if side else 54
            b.structural(210050)
            assert orders[1]['req']==210050 and b.struct_log[-1]['cross']
    print('known long/short holdings, equality, reserve precedence, closing eligibility, cross protection and time endpoints passed')


if __name__=='__main__':
    {'test':test,'run':run}[sys.argv[1]]()
