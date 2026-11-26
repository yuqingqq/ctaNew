"""Closed-loop timing sensitivity of passive urgency at Tokyo's share cap."""
import json
import math
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'tokyo_execution_comparison_20261126'))
import cap as prior

LIB, R, U = prior.LIB, prior.R, prior.U
S, OD = LIB['S'], LIB['OD']
DAYS = ('20261023','20261024','20261025','20261027','20261031','20261101','20261102','20261103')
ARMS = ('nominal','short')


class ShortEngine(R.VM.VMEngine):
    """Change only eligibility inside venue matching; notices still use the original remove time."""
    def __init__(self,*args):
        super().__init__(*args)
        self.retiring={}

    def request(self,gid,g,why):
        super().request(gid,g,why)
        self.retiring[gid]=self.o[gid]['remove']

    def advance(self,grid):
        for gid in self.retiring:
            self.o[gid]['remove']=self.o[gid]['req']
        try:
            super().advance(grid)
        finally:
            for gid,remove in list(self.retiring.items()):
                self.o[gid]['remove']=remove
                if grid>=remove:
                    del self.retiring[gid]


def fixture():
    LIB['fixture']()
    frame=np.array([[0.,0.,40,39,38,37,36,0,0,0,0,0,60,61,62,63,64,0,0,0,0,0]])
    for side in [0,1]:
        tick=40 if side==0 else 60
        for req in (50.,100.):
            path=dict(frames=frame,trades=np.array([[t,side,tick,1.] for t in (59.,99.,100.,101.,136.,137.)]))
            for engine in (R.VM.VMEngine,ShortEngine):
                b=engine(path,300.,200.,R.VM.VM_PROFILE,'L')
                gid=b.add_order(side,tick,0.,50.,0)
                b.advance(req); b.request(gid,req,'policy')
                remove=b.o[gid]['remove']
                expected_remove=95. if req==50 else 137.
                assert remove==expected_remove
                b.advance(300.)
                assert b.o[gid]['remove']==expected_remove
                expected=([59.] if req==50 else [59.,99.,100.,101.,136.]) if engine is R.VM.VMEngine else ([] if req==50 else [59.,99.,100.])
                assert [f[0] for f in b.fills]==expected


def simulate(arm,path,info,ident,flags,fit,payout):
    b=U.UrgencyBranch(path,info['end_rel'],ident)
    if arm=='short':
        assert not b.E.o and b.E.it==0
        b.E=ShortEngine(path,info['end_rel'],200.,b.prof,'L')
    b.q0=50
    pol=prior.Veto(b,flags,fit,'linear',{},[])
    with patch.object(R,'max_new',S['size_bound'](50)):
        for k in range(R.LB.n_ticks(b)):
            R.LB.step(b,k,pol)
            inv,bids,asks=b.reserve(k*R.BE.STEP)
            assert inv+bids<=150+1e-6 and -inv+asks<=150+1e-6
        R.LB.finish(b)
    fills=b.E.fills
    pos=np.r_[0.,np.cumsum([(1-2*s)*q for _,_,s,q,_,_ in fills])]
    assert max(abs(pos))<=150+1e-6 and len(pol.proposals)==len(b.E.o)
    mark=100*payout
    result=LIB['ledger'](fills,mark)
    _,end,val=b.E.value(mark)
    assert abs(end-pos[-1])<1e-6 and abs(val/100-result['turnover']-result['residual'])<1e-6
    assert abs(result['shares']-2*result['closed']-abs(end))<1e-6
    pending=sum(q for t,i,_,q,_,_ in fills if b.E.o[i]['req'] is not None and t>b.E.o[i]['req'])
    assert arm!='short' or pending==0
    result.update(abs_end=abs(end),settle=val/100,posts=len(b.E.o),blocked_slots=pol.blocked,
                  peak_position=max(abs(pos)),age_share_seconds=sum((b_[0]-a[0])*q/1000 for a,b_,q in U.C.inventory_pairs(fills)),
                  post_request_shares=pending)
    return result


def totals(g):
    r={k:float(g[k].sum()) for k in ['settle','turnover','residual','closed','shares','post_request_shares']}
    r.update(markets=len(g),cents_per_matched=100*r['turnover']/r['closed'] if r['closed'] else None,
             mean_abs_end=float(g.abs_end.mean()),max_abs_end=float(g.abs_end.max()),without_best_market=r['settle']-float(g.settle.max()),
             zero_fill_markets=int((g.shares==0).sum()))
    return r


def verdict(arms):
    if any(min(r['settle'],r['without_best_market'],r['without_best_day'])<=0 for r in arms.values()):
        return 'stop'
    if all(r['positive_days']>len(DAYS)/2 for r in arms.values()):
        return 'consider_prospective_validation'
    return 'inconclusive'


def run():
    fixture()
    gate={'a':{'settle':5,'without_best_market':3,'without_best_day':2,'positive_days':5}}
    assert verdict(gate)=='consider_prospective_validation'
    assert verdict({'a':dict(gate['a'],without_best_day=-1)})=='stop'
    assert verdict({'a':dict(gate['a'],positive_days=4)})=='inconclusive'
    fit=U.S.read(R.HERE/'fit_v2.json')['cells']
    bind=U.S.read(S['L']['E'].REPLICATION/'result.json')
    assert R.sha(R.HERE/'fit_v2.json')==bind['fit_sha'] and R.sha(R.CENSUS)==bind['census_sha']
    assert R.sha(S['L']['E'].PRIOR/'urgency.py')==bind['controller_sha']
    assert R.sha(OD.OUT3/'bn_changes.parquet')==U.S.read(OD.RCPT/'bn_events.json')['out_sha256']
    assert R.sha(OD.OUT3/'summary.json')=='f1b45f6555c707654acf337359e98d900b67cf4c1277312dfdd4d82c0b46c460'
    regimes=U.S.read(R.CENSUS)['rows'],U.S.read(OD.OUT3/'summary.json')['census']
    tape=R.MM.KP.Tape()
    edges=R.MM.TR.T.P.frozen_edges()
    baseline=prior.HERE/'cap_markets.csv'
    assert R.sha(baseline)==U.S.read(prior.HERE/'verification.json')['sha256']['cap_markets.csv']
    control=pd.read_csv(baseline,float_precision='round_trip').query('cap==150').set_index('mid')
    coverage={}
    for day in ('20261026',*DAYS):
        seg=OD.SEG_OF.get(day)
        census=[r for r in regimes[int(bool(seg))] if r['day']==day]
        reg=R.MM.registry(day).set_index('mid')
        assert len(reg)==len(census)==288 and set(reg.index)=={r['mid'] for r in census}
        ids={r['mid'] for r in census if r['status']=='complete'}
        archived=S['archived'](day)
        assert ids==set(archived.index)
        expected=archived.tape_id.to_dict() if seg else {r['mid']:r['tape_id'] for r in census if r['status']=='complete'}
        coverage[day]=dict(Counter(r['status'] for r in census))
        ref=pd.read_csv(HERE.parent/'urgency_size50_20261126'/f'markets_{day}.csv').query('size==50')
        assert len(ref)==len(ids) and set(ref.mid)==ids
        if day=='20261026': assert ids==set(control.index)
        off=tape.offset(day)
        payouts=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
        seen=set(); rows=[]
        for _,group in reg.groupby('hour'):
            batch=sorted(ids & set(group.index))
            if not batch: continue
            for m,status,path,info in R.MM.worlds(day,batch,tape,off,reg):
                assert status=='complete' and m not in seen and m in batch
                seen.add(m)
                ws=1000.*reg.at[m,'ws']
                ident=R.MM.ident_of(day,m,ws,info,path)
                assert ident['tape_id']==expected[m]
                if seg:
                    assert OD.SEG_BOUNDS[seg][0]<=ws<ws+300000<OD.SEG_BOUNDS[seg][1]
                    cache=OD.make_cache3(info['t_start'],info['end_rel'],seg)
                else:
                    cache=R.MM.make_cache(info['t_start'],info['end_rel'])
                flags=R.triggers(cache,edges)
                payout=R.VM.V.payout_of(payouts.loc[m])
                for arm in (('nominal',) if day=='20261026' else ARMS):
                    res=simulate(arm,path,info,ident,flags,fit,payout)
                    if day=='20261026':
                        for k,val in res.items():
                            if k!='post_request_shares':
                                assert math.isclose(val,control.at[m,k],rel_tol=0,abs_tol=1e-6),(m,k,val,control.at[m,k])
                    rows.append(dict(day=day,mid=m,tape_id=expected[m],arm=arm,**res))
        assert seen==ids
        pd.DataFrame(rows).to_csv(HERE/f'markets_{day}.csv',index=False)
        print(day,len(seen),'complete',flush=True)
    df=pd.concat([pd.read_csv(HERE/f'markets_{d}.csv',float_precision='round_trip') for d in DAYS])
    assert len(df)==2*619 and not df.duplicated(['day','mid','arm']).any()
    daily=[]; arms={}
    for arm,g in df.groupby('arm'):
        r=totals(g)
        days=g.groupby('day').settle.sum()
        r.update(positive_days=int((days>0).sum()),without_best_day=r['settle']-float(days.max()))
        arms[arm]=r
        daily.extend(dict(day=d,arm=arm,**totals(h)) for d,h in g.groupby('day'))
    pd.DataFrame(daily).to_csv(HERE/'daily.csv',index=False)
    receipt=dict(arms=arms,screen=verdict(arms),coverage=coverage)
    (HERE/'result.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(arms),receipt['screen'],flush=True)


if __name__=='__main__':
    run()
