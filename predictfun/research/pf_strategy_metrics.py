"""Common gross fill accounting for the Predict strategy scorecard.

Marks use the modeled execution clock, not the unshifted trade anchor.
Settlement is hold-to-resolution after first-minute quoting stops. Nothing in
this module supplies labels, marks or outcomes to a policy.
"""
from collections import Counter
import numpy as np
import pandas as pd
import pf_stop_ev_latency as L

HORIZONS = (100, 500, 1000, 5000)


def close(a, b, name):
    L.require(abs(float(a)-float(b)) <= 1e-6 + 1e-10*abs(float(b)), name)


def lookup(times, mids, query, before=False):
    ix = np.searchsorted(times, query, side='left' if before else 'right')-1
    return np.where(ix>=0, mids[np.maximum(ix,0)], np.nan), np.where(ix>=0, query-times[np.maximum(ix,0)], np.nan)


def union(intervals):
    out=[]
    for start,end in sorted((a,b) for a,b in intervals if b>a):
        if out and start<=out[-1][1]:out[-1]=(out[-1][0],max(end,out[-1][1]))
        else:out.append((start,end))
    return out


def duration(intervals):return sum(b-a for a,b in union(intervals))


def reconcile(v):
    close(v['spread_capture_usd']+v['mid_drift_1000_usd'],v['markout_1000_usd'],'spread/drift/markout identity')
    close(v['cash_usd']+v['terminal_position']*v['terminal_mid_cents']/100,v['terminal_pnl_usd'],'cash/inventory identity')
    close(v['terminal_pnl_usd']+v['settlement_carry_usd'],v['settlement_pnl_usd'],'settlement carry identity')
    close(v['cash_usd']+v['terminal_position']*v['settlement_yes'],v['settlement_pnl_usd'],'binary settlement identity')
    close(v['young_terminal_usd']+v['older_terminal_usd'],v['terminal_pnl_usd'],'age attribution identity')
    close(v['ws_terminal_usd']+v['e2_terminal_usd'],v['terminal_pnl_usd'],'source attribution identity')


def account(W,r,src,im,shift,day,case):
    tf=np.asarray(W[0]['tf'],float);bb=W[0]['P5'][:,0];ba=W[1]['P5'][:,0]
    good=(bb>=0)&(ba>bb);times=tf[good];mids=(bb[good]+ba[good])/2
    L.require(len(times)>0 and np.all(np.diff(times)>=0),'invalid valuation book')
    term=L.M.CL.portfolio_terminal(r,*W[0]['tf_mff'])
    outcome=float(W[0]['yup']);L.require(outcome in (0.,1.),'unknown settlement outcome')
    rows=[];events=[]
    for (s,d),fills in r['fills'].items():
        for seq,f in enumerate(fills):
            i,p,q,through,age,phase,ahead=f
            anchor=float(W[s]['anc'][i]);t=anchor-shift;events.append((s,d,t,p,q))
            rows.append((day,int(W[0]['m']),case,s,d,seq,t,anchor,float(p),float(q),bool(through),float(age),phase=='pend',bool(src[s][i]),bool(im[s][i])))
    L.require(Counter(events)==Counter(r['fl']),'fill clock/identity mismatch')
    cols=['day','m','case','side','rung','fill_seq','fill_ms','anchor_ms','price_cents','shares','through','age_ms','cancel_pending','ws_timed','imputed']
    f=pd.DataFrame(rows,columns=cols);q=f.shares.to_numpy(float);sg=np.where(f.side.to_numpy()==0,1.,-1.);p=f.price_cents.to_numpy(float);t=f.fill_ms.to_numpy(float)
    pre,preage=lookup(times,mids,t,before=True)
    L.require(np.isfinite(pre).all(),'missing pre-fill mid: no silent exclusion')
    f['pre_mid_cents']=pre;f['pre_mid_age_ms']=preage
    f['spread_capture_usd']=q*sg*(pre-p)/100
    v=dict(fills=len(f),shares=float(q.sum()),buy_shares=float(q[sg==1].sum()),sell_shares=float(q[sg==-1].sum()),
        spread_capture_usd=float(f.spread_capture_usd.sum()),pre_mid_stale_1000ms_shares=float(q[preage>1000].sum()),
        cash_usd=float(term['cash']),terminal_position=float(term['position']),terminal_mid_cents=float(term['mark']),
        settlement_yes=outcome,terminal_ms=float(term['time']),terminal_pnl_usd=float(term['pnl']),
        settlement_pnl_usd=float(term['cash']+term['position']*outcome),
        settlement_carry_usd=float(term['position']*(outcome-term['mark']/100)),
        orders=sum(len(o) for o in r['orders'].values()),pulls=int(sum(r['stat']['pulls'])),
        rejected_orders=sum(int(o['ref']) for z in r['order_states'].values() for o in z),
        filled_orders=sum(int(bool(o['fl'])) for z in r['order_states'].values() for o in z),
        max_abs_position=float(r['stat']['maxpos']),terminal_abs_position=abs(float(term['position'])))
    for h in HORIZONS:
        mark,age=lookup(times,mids,t+h);L.require(np.isfinite(mark).all(),'missing future mid: no silent exclusion')
        f[f'markout_{h}_usd']=q*sg*(mark-p)/100;f[f'mid_drift_{h}_usd']=q*sg*(mark-pre)/100
        v[f'markout_{h}_usd']=float(f[f'markout_{h}_usd'].sum());v[f'mid_drift_{h}_usd']=float(f[f'mid_drift_{h}_usd'].sum())
        v[f'mark_{h}_stale_1000ms_shares']=float(q[age>1000].sum())
    f['terminal_pnl_usd']=q*sg*(term['mark']-p)/100;f['settlement_pnl_usd']=q*sg*(outcome-p/100)
    f['terminal_mid_cents']=float(term['mark']);f['settlement_yes']=outcome
    close(f.terminal_pnl_usd.sum(),term['pnl'],'fill/portfolio terminal parity')
    close(f.settlement_pnl_usd.sum(),v['settlement_pnl_usd'],'fill/portfolio settlement parity')
    masks=dict(young=f.age_ms<100,older=f.age_ms>=100,age100_500=(f.age_ms>=100)&(f.age_ms<500),age500=f.age_ms>=500,
        ws=f.ws_timed.astype(bool),e2=~f.ws_timed.astype(bool),pending=f.cancel_pending.astype(bool),imputed=f.imputed.astype(bool),through=f.through.astype(bool))
    for label,mask in masks.items():
        v[f'{label}_terminal_usd']=float(f.loc[mask,'terminal_pnl_usd'].sum());v[f'{label}_shares']=float(f.loc[mask,'shares'].sum())
    # An overlap-safe availability statistic, distinct from the old sum of rung exposure.
    start=float(W[0]['tg'][0]);end=float(W[0]['te']);side={s:union([(max(start,a),min(end,b)) for (ss,d),oo in r['orders'].items() if ss==s for a,b,_,ref in oo if not ref]) for s in (0,1)}
    a,b,ab=duration(side[0]),duration(side[1]),duration(side[0]+side[1]);length=end-start
    v.update(quote_any_ms=ab,quote_both_ms=a+b-ab,quote_bid_ms=a,quote_ask_ms=b,quote_observed_ms=length)
    L.require(0<=v['quote_both_ms']<=v['quote_any_ms']<=length+1e-6,'quote availability bounds')
    ordered=sorted(r['fl'],key=lambda x:x[2]);last=start;position=0.;area=0.
    for s,d,t,p,q in ordered:
        L.require(start<=t<=term['time'],'fill outside inventory integration interval')
        area+=abs(position)*(t-last);position+=q if s==0 else -q;last=t
    area+=abs(position)*(term['time']-last)
    close(position,term['position'],'integrated position parity')
    v.update(abs_inventory_share_ms=area,inventory_observed_ms=term['time']-start)
    reconcile(v);return v,f


def selftest():
    pre,_=lookup(np.array([0.,10.,20.]),np.array([50.,51.,52.]),np.array([10.]),before=True)
    post,_=lookup(np.array([0.,10.,20.]),np.array([50.,51.,52.]),np.array([10.]))
    L.require(pre[0]==50 and post[0]==51,'strict pre-fill lookup control')
    L.require(duration([(0,3),(2,5),(9,10)])==6,'overlap-safe availability control')
    v=dict(spread_capture_usd=2.,mid_drift_1000_usd=-3.,markout_1000_usd=-1.,cash_usd=-4.,terminal_position=10.,terminal_mid_cents=50.,terminal_pnl_usd=1.,settlement_carry_usd=5.,settlement_pnl_usd=6.,settlement_yes=1.,young_terminal_usd=-1.,older_terminal_usd=2.,ws_terminal_usd=3.,e2_terminal_usd=-2.)
    reconcile(v);caught=0
    for key in ['mid_drift_1000_usd','terminal_position','settlement_carry_usd','settlement_pnl_usd','older_terminal_usd','e2_terminal_usd']:
        bad=dict(v);bad[key]+=1
        try:reconcile(bad)
        except ValueError:caught+=1
    L.require(caught==6,'corrupted metric not refused')
    # Buy 10 at 40c and sell 4 at 60c: Up=4.4, Down=-1.6, independent of trade ordering.
    for outcome,expected in [(1.,4.4),(0.,-1.6)]:
        close(10*(outcome-.4)-4*(outcome-.6),expected,'bid/ask settlement control')
    return dict(strict_pre_fill=True,overlap_union=True,accounting_positive_control=True,corruptions_refused=caught,binary_bid_ask=True)
