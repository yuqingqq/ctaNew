from collections import defaultdict
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'predictfun/research'))
import pf_timing_case_audit as T
ARCH=Path('/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108')
STEP=ARCH/'research/step1/20261108'
KEYS=['mid','exec_s','first_recv_ns','side','ptick','shares','price_executed','taker_index','taker_quote']


def matches(price,amount,inc):
    value=price*amount/100
    return abs(value-inc)<=np.maximum(.01,.011*value)


def window(t,lag,submit,ack,cancel,done):
    if t<submit: return 'before_submit'
    if t-lag>=done: return 'after_cancel_ack'
    if t-lag>=ack and t<cancel: return 'inside_inner'
    return 'overlaps_boundary'


def fixture():
    assert matches(50,10,5) and not matches(50,10,6)
    assert window(100,50,30,50,110,150)=='inside_inner'
    assert window(200,50,30,50,110,150)=='after_cancel_ack'
    assert window(199,50,30,50,110,150)=='overlaps_boundary'
    k=('synthetic',)
    assert len(T.pairs(np.array([101]),[k],np.array([100]),[k])[0])==1
    assert len(T.pairs(np.array([101,102]),[k,k],np.array([100]),[k])[0])==0


def main():
    fixture()
    core=pd.concat([pd.read_parquet(ARCH/f'live_vs_replay/core_orders_{r}.parquet').assign(run=r) for r in ('basis','v6')]).set_index('cid',verify_integrity=True)
    legs=pd.read_parquet(STEP/'legs_timed2.parquet')
    inc=pd.read_parquet(STEP/'pend_inc.parquet')
    known=pd.read_parquet(ROOT/'data/predict_fun/research/fill_clock_20261109/clock_cases.parquet')
    assert len(known)==99
    raw=pd.read_csv(HERE.parent/'tokyo_closing_trace_20261126/events.csv',float_precision='round_trip')
    jobs=[]
    for x in raw[raw.wallet_filled==0].itertuples():
        leg=legs.loc[x.leg]
        assert leg.mid==x.market and leg.ptick==x.trade_tick and abs(leg.shares-x.public_leg_qty)<1e-8
        jobs.append(dict(cohort='disputed',market=x.market,cid=x.cid,leg=x.leg,q=x.closed))
    for x in known.itertuples():
        k=legs[legs.mid==x.mid]
        for c in KEYS[1:]:
            k=k[k[c]==getattr(x,c)]
        # Step 1 encodes signer as an integer. Require a unique public row on
        # all eight shared identity fields from the wallet-clock ledger.
        jobs.append(dict(cohort='wallet',market=x.mid,cid=x.cid,leg=int(k.index[0]) if len(k)==1 else -1,q=x.shares))
    assert abs(sum(x['q'] for x in jobs if x['cohort']=='disputed')-119.48615384615368)<1e-7
    # Candidate uniqueness is checked against the full market, before looking at the wallet lifecycle.
    wins=defaultdict(list)
    for j in jobs:
        j['status']='public_leg_ambiguous_or_missing'
        if j['leg']<0: continue
        l=legs.loc[j['leg']]; m=j['market']; s=l.exec_s*1000; p=l.ptick
        block=inc[(inc.mid==m)&(inc.ptick==p)&(inc.t_ms>=s-1500)&(inc.t_ms<s+1500)]
        cand=block[matches(p,l.shares,block.inc)]
        if len(cand)!=1:
            j['status']='no_increment' if cand.empty else 'several_increments'
            continue
        row=cand.iloc[0]; u=row.t_ms
        rev=legs[(legs.mid==m)&(legs.ptick==p)&(legs.exec_s*1000<=u+1500)&(legs.exec_s*1000>u-1500)]
        if list(rev[matches(p,rev.shares,row.inc)].index)!=[j['leg']]:
            j['status']='increment_has_other_possible_leg'
            continue
        j.update(status='needs_frame',u=float(u),inc=float(row.inc),pend_side=int(row.side),ptick=int(p))
        wins[m].append((u-12000,u+12000))
    hours=sorted({h for ws in wins.values() for a,b in ws for h in T.hours_between(a,b)})
    rf,_,_=T.research_frames(hours,wins)
    markets_by_run={r:{j['market'] for j in jobs if core.loc[j['cid'],'run']==r} for r in ('basis','v6')}
    pairs={}
    for r,markets in markets_by_run.items():
        ws={m:wins[m] for m in markets if m in wins}
        if not ws: continue
        bf,_,_=T.input_frames(T.INPUTS[r],ws,[])
        for m in ws:
            F=rf.get(m); bot=bf.get(m,[])
            if F is None or not bot: continue
            paired,_=T.pairs(np.array([x[0] for x in bot]),[x[1] for x in bot],F.u.to_numpy(),list(F.key))
            pairs[m]={int(x.iu):float(x.t_bot) for x in paired.itertuples()}
    for j in jobs:
        if j['status']!='needs_frame': continue
        m=j['market']; u=j['u']; F=rf.get(m); twin=pairs.get(m,{})
        hits=[]; key=(j['pend_side'],j['ptick'])
        if F is not None:
            for i in np.flatnonzero(F.u.to_numpy()==u):
                if i==0: continue
                delta=F.pend.iloc[i].get(key,0)-F.pend.iloc[i-1].get(key,0)
                if abs(delta-j['inc'])<1e-6: hits.append(i)
        if len(hits)!=1 or hits[0] not in twin:
            j['status']='no_unique_paired_increment_frame'; continue
        t=twin[hits[0]]; o=core.loc[j['cid']]
        cancel=o.t_cancel if pd.notna(o.t_cancel) else np.inf
        done=o.t_CancelDone if pd.notna(o.t_CancelDone) else np.inf
        j['status']='paired'; j['receipt_ms']=t
        j['upper']='before_submit' if t<o.t_req else ('by_cancel_request' if np.isfinite(cancel) and t<=cancel else 'no_cancel_request' if not np.isfinite(cancel) else 'after_request_unresolved')
        for lag in (50,250,500):
            j[str(lag)]=window(t,lag,o.t_req,o.t_Accepted,cancel,done)
    d=pd.DataFrame(jobs)
    d.to_csv(HERE/'rows.csv',index=False)
    for cohort,g in d.groupby('cohort'):
        print(cohort,len(g),g.cid.nunique(),g.q.sum())
        for col in ('status','upper','50','250','500'):
            print(col,g.groupby(col).q.sum().round(3).to_dict())
if __name__=='__main__': main()
