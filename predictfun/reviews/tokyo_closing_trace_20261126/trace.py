"""Trace the previously reported closing-share mismatches on consumed November 8.
Freeze the cases by original primary CID and classify the public legs in each case.
Report other primary closes on wallet-filled IDs as a descriptive comparator.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'tokyo_execution_comparison_20261126'
sys.path.insert(0,str(PRIOR))
import compare as C


def timing(t,accept,request):
    if t<accept and t>=request:
        return 'before_ack_and_after_cancel_request'
    if t<accept: return 'before_acceptance_notice'
    if t>=request: return 'at_or_after_cancel_request'
    return 'after_ack_before_cancel_request'


def classify(side,quote,trade):
    if quote==trade: return 'same_price'
    if (1-2*side)*(quote-trade)>0: return 'through_price'
    raise ValueError('public price could not execute order')


def test():
    assert timing(2,2,3)=='after_ack_before_cancel_request'
    assert timing(3,2,3)=='at_or_after_cancel_request'
    assert timing(1,2,1)=='before_ack_and_after_cancel_request'
    assert classify(1,44,45)=='through_price' and classify(0,44,44)=='same_price'
    try: classify(0,44,45)
    except ValueError: return
    raise AssertionError('wrong-side price accepted')


def check_close(a,b):
    if abs(a-b)>1e-6: raise ValueError(f'quantity mismatch {a} {b}')


def run():
    test()
    sha=json.loads((PRIOR/'verification.json').read_text())['sha256']['orders.parquet']
    op=C.OUT/'orders.parquet'
    assert hashlib.sha256(op.read_bytes()).hexdigest()==sha
    old=pd.read_parquet(op)
    ref=old[old.world=='primary'].set_index('cid',verify_integrity=True)
    targets=ref[ref.pred_close>1e-9]
    windows=json.loads((C.ARCH/'scope.json').read_text())['windows']
    starts={int(w['market']):1000*w['ws'] for w in windows}
    runs={int(w['market']):w['run'] for w in windows}
    needed=set(targets.market)
    trades=C.trade_table(starts)
    bands=pd.read_parquet(C.DATA/'maker_trace_v2_20261123/truth/v2/own_orders.parquet',
                          columns=['run','d_lo_ms','d_hi_ms']).drop_duplicates().set_index('run',verify_integrity=True)
    core=pd.concat([pd.read_parquet(C.ARCH/f'live_vs_replay/core_orders_{r}.parquet').assign(run=r) for r in bands.index])
    core=core[core.t_Accepted.notna() & core.refused.isna() & (core.market.map(runs)==core.run)]
    ck=json.loads((C.STEP/'clock_rl_d2.json').read_text())
    col=[f'{s}{i}_{v}' for s in 'ba' for v in 'ps' for i in range(1,6)]
    frames={}
    for file in sorted((C.STEP/'books').glob('part_*.parquet')):
        df=pd.read_parquet(file,columns=['mid','t_ms','recv_ns',*col],filters=[('mid','in',sorted(needed))])
        for m,g in df[(df.t_ms-df.mid.map(starts)).between(-5000,301000)].groupby('mid'):
            frames.setdefault(m,[]).append(g)
    rows=[]
    for m in sorted(needed):
        start=starts[m]
        b=pd.concat(frames[m]).sort_values('t_ms',kind='stable')
        tf=b.t_ms.to_numpy()-start
        known=np.maximum.accumulate(np.maximum(tf+ck['pf_min'],b.recv_ns.to_numpy()/1e6-start-ck['shift_ms'])+ck['d_dec'])
        f=np.column_stack([tf,known,b[col].to_numpy()])
        tt=trades[trades.mid==m]
        lo,hi=bands.loc[runs[m]]
        eng=C.R.VM.VMEngine(dict(frames=f,trades=tt[['t','side','ptick','shares']].to_numpy()),300000,200,C.R.VM.VM_PROFILE,'L')
        meta=[]
        for x in core[core.market==m].sort_values(['t_req','cid']).itertuples():
            sub,act,rem=C.lifetime(x,'primary',lo,hi)
            act-=start; rem=min(rem-start,300000.)
            if act<f[0,0] or rem<=act: continue
            s=x.lane%2
            gid=eng.add9([s,0,x.tick,x.size,0,act,rem,sub-start,1])
            assert gid==len(meta)
            meta.append((x,eng.shown(eng.o[gid]['ci'],s,x.tick),act))
        eng.advance(300000.)
        pos=0.
        recon=Counter()
        for t,gid,s,q,p,i in eng.fills:
            closed=min(q,max(0.,-(1-2*s)*pos))
            pos+=(1-2*s)*q
            if closed<=1e-9: continue
            x,ahead,act=meta[gid]
            recon[x.cid]+=closed
            tr=tt.iloc[i]
            assert t==tr.t and s==tr.side
            req=x.t_cancel-start if np.isfinite(x.t_cancel) else np.inf
            acc=x.t_Accepted-start
            j=np.searchsorted(tf,act,'right')-1
            rows.append(dict(market=m,cid=x.cid,leg=int(tt.index[i]),side=s,price=p,trade_tick=tr.ptick,
                             fill_time_ms=t,from_ack_ms=t-acc,from_cancel_ms=t-req,initial_frame_age_ms=act-tf[j],
                             initial_ahead=ahead,public_leg_qty=tr.shares,fill_qty=q,closed=closed,
                             wallet_filled=x.filled,kind=classify(s,p,tr.ptick),timing=timing(t,acc,req),
                             inside_for_all_offsets=t>=acc+hi and t<req+lo,
                             imputed=bool(tr.imputed)))
        for cid,row in ref[ref.market==m].iterrows(): check_close(recon[cid],row.pred_close)
    df=pd.DataFrame(rows)
    assert set(df.cid)==set(targets.index) and not df.duplicated(['market','cid','leg']).any()
    focus=df[df.wallet_filled==0]
    expected=json.loads((PRIOR/'fixed.json').read_text())['metrics']['primary']['closing_on_unfilled']
    check_close(focus.closed.sum(),expected)
    output={}
    for name,pop in [('zero_wallet',focus),('positive_wallet',df[df.wallet_filled>0])]:
        g=targets.loc[pop.cid.unique()]
        out=dict(markets=pop.market.nunique(),orders=len(g),legs=len(pop),closed=pop.closed.sum(),
                 by_price=pop.groupby('kind').closed.sum().to_dict(),by_timing=pop.groupby('timing').closed.sum().to_dict(),
                 robust_clock=pop.loc[pop.inside_for_all_offsets,'closed'].sum(),
                 imputed=pop.loc[pop.imputed,'closed'].sum(),
                 at_price_initial_queue_zero=pop.loc[(pop.kind=='same_price') & (pop.initial_ahead==0),'closed'].sum(),
                 at_price_queue_unknown=pop.loc[(pop.kind=='same_price') & pop.initial_ahead.isna(),'closed'].sum())
        out['same_id_retained_closing']={}
        for w,v in old[old.cid.isin(g.index)].groupby('world'):
            qty=v.set_index('cid').pred_close.reindex(g.index)
            assert qty.notna().all()
            out['same_id_retained_closing'][w]=float(np.minimum(qty,g.pred_close).sum())
        output[name]=out
    df.to_csv(HERE/'events.csv',index=False)
    (HERE/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output))


if __name__=='__main__': run()
