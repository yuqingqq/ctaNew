"""Cost-aware inventory shadow diagnostic, not a changed-state MM replay.

Each neutral action starts with a hypothetical 25-share Up OR Down position.
Model chooses hold or hedge using the known quote; the hedge is valued 40ms
later against visible depth. Each anchor/side is a separate counterfactual,
never summed as independent trades. No simulated PnL is advertised as live.
"""
import datetime as dt
import numpy as np
import pandas as pd
from . import pf_clock_rl as CK
from .pf_sizing_study import DAYS, OLD, DATA, DEST, sha, write
from .pf_settlement_study import MODELS, paired_inference

Q=25.

def decide(p,side,bid,ask,rate):
    px=(100-bid)/100 if side==0 else ask/100
    locked=1-px-rate*min(px,1-px)
    held=p if side==0 else 1-p
    return held<locked

def controls():
    assert decide(.2,0,49,51,.02)
    assert not decide(.8,0,49,51,.02)
    assert not decide(.5,0,49,51,.02)
    assert decide(.8,1,49,51,.02)
    return dict(bad_up_hedged=True,good_up_kept=True,market_spread_fee_respected=True,symmetric=True)

def main():
    checked=controls();pr=pd.read_parquet(DEST/'settlement_predictions.parquet');rows=[];sources={}
    for day in DAYS[3:]:
        s1=OLD/'step1'/day
        files=sorted((s1/'books').glob('*.parquet'))
        cols=['mid','t_ms','recv_ns','b1_p','a1_p','b1_s','a1_s']
        books=pd.concat([pd.read_parquet(p,columns=cols) for p in files],ignore_index=True)
        books.sort_values(['mid','t_ms','recv_ns'],kind='stable',inplace=True)
        groups={m:g for m,g in books.groupby('mid',sort=False)}
        sources[day]={str(p):sha(p) for p in files}
        shift=CK.load_shift(str(s1))
        for mid,gp in pr[pr.day==day].groupby('mid'):
            book=groups.get(mid)
            if book is None:raise ValueError('missing declared market book')
            tv=book.t_ms.to_numpy();kn=np.maximum.accumulate(CK.pf_known(tv,book.recv_ns.to_numpy(),shift))
            for r in gp.itertuples():
                i=np.searchsorted(kn,r.tau,side='right')-1;j=np.searchsorted(tv,r.tau+40,side='right')-1
                if i<0 or j<0:raise ValueError('no current/arrival book for scored action')
                known=book.iloc[i];future=book.iloc[j]
                if abs((known.b1_p+known.a1_p)/2-r.pf_mid)>1e-6:raise ValueError('decision book/feature timestamp mismatch')
                if not np.isfinite(r.fee_bps) or not 0<=r.fee_bps<=1000:raise ValueError('invalid actual market fee')
                rate=r.fee_bps/10000
                for side in [0,1]:
                    payout=r.y if side==0 else 1-r.y
                    px=(100-future.b1_p)/100 if side==0 else future.a1_p/100
                    depth=future.b1_s if side==0 else future.a1_s
                    status='eligible'
                    if not (0<future.b1_p<future.a1_p<100):status='arrival_book_invalid'
                    elif depth<Q:status='arrival_depth_insufficient'
                    elif Q*px<1:status='minimum_notional'
                    elif r.tau-kn[i]>5000:status='stale_known_book'
                    locked=1-px-rate*min(px,1-px)
                    for m in MODELS:
                        hedge=decide(getattr(r,m),side,known.b1_p,known.a1_p,rate)
                        executed=hedge and status=='eligible'
                        rows.append(dict(day=day,mid=mid,anchor=r.t_rel,side=side,model=m,status=status,
                          wanted_hedge=hedge,executable=executed,delta_vs_hold=Q*(locked-payout) if executed else 0.))
        del books,groups
    a=pd.DataFrame(rows);summaries={}
    for m,g in a.groupby('model'):
        summaries[m]=dict(actions=len(g),wanted_hedges=int(g.wanted_hedge.sum()),executable_hedges=int(g.executable.sum()),
          statuses=g.status.value_counts().to_dict(),mean_dollars_per_25share_action=float(g.delta_vs_hold.mean()),
          per_day=g.groupby('day').delta_vs_hold.mean().to_dict(),
          per_anchor={str(k):dict(actions=len(v),mean_delta=float(v.delta_vs_hold.mean()),hedges=int(v.executable.sum())) for k,v in g.groupby('anchor')})
    delta=np.array([[summaries[m]['per_day'][d]-summaries['market_mid']['per_day'][d] for m in MODELS[1:]] for d in DAYS[3:]])
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),controls=checked,n_days=6,n_markets=int(a.mid.nunique()),
      hypothetical_inventory=Q,models=summaries,paired_inference=dict(zip(MODELS[1:],paired_inference(delta))),
      input_sha=sources,code_sha=sha(__file__),interpretation='neutral hypothetical inventory shadow; not actual strategy PnL or executable liquidity guarantee',
      notional='25 shares per side, separately; mean opportunity value, never sum anchors into a trading return',
      forward_data_read=False)
    a.to_parquet(DEST/'inventory_shadow_actions.parquet',index=False)
    write(DEST/'inventory_shadow_report.json',result)
    print({m:{k:v for k,v in s.items() if k in ['wanted_hedges','executable_hedges','mean_dollars_per_25share_action']} for m,s in summaries.items()})

if __name__=='__main__':main()
