"""Fixed-order execution calibration. Run on the Codex light slot."""
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCH = Path('/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108')
STEP = ARCH/'research/step1/20261108'
DATA = ROOT/'data/predict_fun/research'
OUT = DATA/HERE.name
LIB = run_path(str(HERE.parent/'urgency_size50_20261126/replay.py'))
R, U = LIB['R'], LIB['U']
WORLDS = ('primary','short','long','clock_lo','clock_hi','timed_only')


def trade_table(starts):
    df = pd.read_parquet(STEP/'legs_timed2.parquet',columns=['mid','side','ptick','shares','t_ms','t_e2','exec_s','how'])
    tk = pd.read_parquet(STEP/'legs_tk.parquet',columns=['tk']).tk.to_numpy()
    assert len(df)==len(tk)
    grp=np.where(tk>=0,tk,-1-np.arange(len(tk)))
    book=df.t_ms.where(df.how>0).groupby(grp).transform('min')
    chain=df.t_e2.where(df.how==0).groupby(grp).transform('min')
    a=book.fillna(chain)
    start=df.mid.map(starts)
    sample=(a>=start)&(a<start+60000)
    off=(a-df.exec_s*1000)[sample].median()
    df['imputed']=a.isna()
    df['t']=a.fillna(df.exec_s.groupby(grp).transform('min')*1000+off)-50-start
    df['priority']=np.where(df.side==0,-df.ptick,df.ptick)
    return df[df.t.between(-5000,300000)].sort_values(['t','side','priority'],kind='stable')


def lifetime(o,world,lo,hi):
    sub=o.t_req
    act=min(max(sub+58,sub),max(sub,o.t_Accepted))
    req=o.t_cancel
    ack=o.t_CancelDone
    rem=(max(req+37,sub+95 if req<act else req) if np.isfinite(req) else np.inf)
    if np.isfinite(ack):
        rem=min(rem,ack)
    if world=='short':
        act,rem=o.t_Accepted,req if np.isfinite(req) else rem
    elif world=='long':
        act,rem=sub,ack if np.isfinite(ack) else rem
    shift={'clock_lo':lo,'clock_hi':hi}.get(world,0.)
    return sub+shift,act+shift,rem+shift


def compare_overlap(actual,pred):
    overlap=np.minimum(actual,pred).sum()
    return dict(actual=float(sum(actual)),predicted=float(sum(pred)),same_order=float(overlap),
                excess=float(sum(pred)-overlap),missed=float(sum(actual)-overlap))


def fixture():
    assert compare_overlap([5,0],[0,5])['same_order']==0
    assert compare_overlap([5,0],[3,5])['same_order']==3
    LIB['fixture']()


def market_stats(fills,mark,clock,mid,world,market):
    # Fill tuples are sorted by their own ledger clock. Signs and order keys are YES-normalized.
    row=LIB['ledger'](fills,mark)
    close=Counter()
    age=0.
    for a,b,q in U.C.inventory_pairs(fills):
        age+=(b[0]-a[0])*q/1000
        close[b[1]]+=q
    row.update(age_share_seconds=age,market=market,world=world,settle=row['turnover']+row['residual'])
    for h in (1,5,30):
        value=qty=0.
        for t,_,s,q,p,_ in fills:
            target=t+h*1000
            j=np.searchsorted(clock,target,'right')-1
            if j>=0 and target<=300000 and 0<=target-clock[j]<=1000 and np.isfinite(mid[j]):
                value+=q*(1-2*s)*(mid[j]-p)
                qty+=q
        row[f'mo{h}_c']=value
        row[f'mo{h}_shares']=qty
    assert abs(sum(close.values())-row['closed'])<1e-6
    return row,close


def run():
    fixture()
    OUT.mkdir(parents=True,exist_ok=True)
    windows=json.loads((ARCH/'scope.json').read_text())['windows']
    starts={int(w['market']):w['ws']*1000 for w in windows}
    runs={int(w['market']):w['run'] for w in windows}
    assert len(starts)==len(windows)==220
    orders=pd.concat([pd.read_parquet(ARCH/f'live_vs_replay/core_orders_{r}.parquet').assign(run=r) for r in set(runs.values())])
    orders=orders[orders.market.map(runs)==orders.run]
    eligible=orders.t_Accepted.notna() & orders.refused.isna()
    assert orders[~eligible].filled.sum()==0
    orders=orders[eligible].copy()
    assert orders.cid.is_unique and orders.align_ok.all()
    trades=trade_table(starts)
    wallet=pd.read_parquet(DATA/'fill_clock_20261109/clock_cases.parquet')
    joined=wallet.merge(orders[['cid','market','lane','tick','run']],on=['run','cid'],validate='many_to_one')
    assert len(joined)==len(wallet)==99 and ((joined.mid==joined.market)&(joined.side==joined.lane%2)&(joined.ptick==joined.tick)).all()
    truth=wallet.groupby('cid').shares.sum().reindex(orders.cid,fill_value=0)
    assert np.allclose(orders.filled,truth,atol=1e-5)
    bands=pd.read_parquet(DATA/'maker_trace_v2_20261123/truth/v2/own_orders.parquet',columns=['run','d_lo_ms','d_hi_ms']).drop_duplicates().set_index('run',verify_integrity=True)
    registry=pd.read_parquet(STEP/'registry.parquet').set_index('mid')
    ck=json.loads((STEP/'clock_rl_d2.json').read_text())
    col=[f'{s}{i}_{v}' for s in 'ba' for v in 'ps' for i in range(1,6)]
    # Keep frames by market; hourly partitions are read once.
    parts={}
    for file in sorted((STEP/'books').glob('part_*.parquet')):
        df=pd.read_parquet(file,columns=['mid','t_ms','recv_ns',*col])
        dt=df.t_ms-df.mid.map(starts)
        for m,g in df[dt.between(-5000,301000)].groupby('mid'):
            parts.setdefault(m,[]).append(g)
    metrics=[]; per_order=[]; counts=Counter()
    for m,start in starts.items():
        reg=registry.loc[m]
        mark=100*R.VM.V.payout_of(reg)
        assert reg.maker_fee_bps==0
        b=pd.concat(parts.pop(m)).sort_values('t_ms',kind='stable')
        venue=b.t_ms.to_numpy()-start
        known=np.maximum.accumulate(np.maximum(venue+ck['pf_min'],b.recv_ns.to_numpy()/1e6-start-ck['shift_ms'])+ck['d_dec'])
        f=np.column_stack([venue,known,b[col].to_numpy()])
        mid=np.where((f[:,2]>=0)&(f[:,12]>f[:,2]),(f[:,2]+f[:,12])/2,np.nan)
        t=trades[trades.mid==m]
        own=orders[orders.market==m].sort_values(['t_req','cid'])
        w=wallet[wallet.mid==m].sort_values(['first_positive_notice','trade_id'])
        wf=[(x.first_positive_notice-start,x.cid,x.side,x.shares,x.ptick,i) for i,x in enumerate(w.itertuples())]
        stats,ac=market_stats(wf,mark,known,mid,'wallet_notice',m)
        metrics.append(stats)
        # Eligible wallet anchors support a paired timing diagnostic; clock-ambiguous legs stay in full wallet ledger.
        subset=w[w.leg_clock_status.isin(['unique_leg','identical_clock_candidates']) & w.Z.notna()]
        for label,times in [('wallet_anchor_subset',subset.Z-50),('wallet_notice_subset',subset.first_positive_notice)]:
            tmp=sorted([(tm-start,x.cid,x.side,x.shares,x.ptick,i) for i,(tm,x) in enumerate(zip(times,subset.itertuples()))])
            metrics.append(market_stats(tmp,mark,known,mid,label,m)[0])
        lo,hi=bands.loc[runs[m]]
        for world in WORLDS:
            tt=t[~t.imputed] if world=='timed_only' else t
            eng=R.VM.VMEngine(dict(frames=f,trades=tt[['t','side','ptick','shares']].to_numpy()),300000,200,R.VM.VM_PROFILE,'L')
            ids=[]; status={}
            for x in own.itertuples():
                sub,act,rem=lifetime(x,world,lo,hi)
                sub-=start; act-=start; rem=min(rem-start,300000.)
                if act<f[0,0] or rem<=act:
                    status[x.cid]='missing_prior_frame' if act<f[0,0] else 'empty_lifetime'
                    continue
                gid=eng.add9([x.lane%2,0,x.tick,x.size,0,act,rem,sub,1])
                assert gid==len(ids)
                ids.append(x.cid)
                status[x.cid]='refused' if eng.o[gid]['refused'] else 'rested'
            eng.advance(300000.)
            fills=[(tm,ids[g],s,q,p,i) for tm,g,s,q,p,i in eng.fills]
            pred=Counter()
            for _,cid,_,q,_,_ in fills: pred[cid]+=q
            stats,pc=market_stats(fills,mark,known,mid,world,m)
            metrics.append(stats)
            for x in own.itertuples():
                per_order.append((world,m,x.cid,status[x.cid],x.filled,pred[x.cid],ac[x.cid],pc[x.cid]))
        counts.update(markets=1,untimed_legs=int(t.imputed.sum()),public_legs=len(t))
    cols=['world','market','cid','status','actual','pred','actual_close','pred_close']
    od=pd.DataFrame(per_order,columns=cols)
    od.to_parquet(OUT/'orders.parquet',index=False)
    md=pd.DataFrame(metrics)
    md.to_csv(HERE/'fixed_markets.csv',index=False)
    print(json.dumps(dict(counts)))
    out={}
    for world,g in od.groupby('world'):
        stats=md[md.world==world].sum(numeric_only=True).to_dict()
        out[world]=dict(shares=compare_overlap(g.actual,g.pred),closing=compare_overlap(g.actual_close,g.pred_close),
                        any_both=int(((g.actual>0)&(g.pred>0)).sum()),any_pred=int((g.pred>0).sum()),any_real=int((g.actual>0).sum()),
                        closing_on_unfilled=float(g.loc[g.actual==0,'pred_close'].sum()),statuses=g.status.value_counts().to_dict())
    for world,g in md.groupby('world'):
        x=g.sum(numeric_only=True)
        out.setdefault(world,{})['economics']=dict(turnover=x.turnover,residual=x.residual,settle=x.settle,closed=x.closed,
            hold_seconds=x.age_share_seconds/x.closed if x.closed else None,shares=x.shares,
            **{f'mo{h}':[x[f'mo{h}_c']/x[f'mo{h}_shares'],x[f'mo{h}_shares']] for h in (1,5,30)})
    (HERE/'fixed.json').write_text(json.dumps(dict(metrics=out,counts=dict(counts),orders=len(orders)),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    run()
