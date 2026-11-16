"""Minute-level execution accounting with explicit expiry censoring."""
from collections import Counter
import pf_stop_ev_minutes_inputs as I
S,np,pd=I.S,I.np,I.pd
L,Q=S.L,S.Q
HORIZONS=(100,500,1000,5000)


def book(W):
    b=W[0]['P5'][:,0];a=W[1]['P5'][:,0]
    valid=(b>=0)&(a>b)
    return np.asarray(W[0]['tf'])[valid],(b[valid]+a[valid])/2,b[valid],a[valid]


def value_fills(f,W):
    f=f.copy();ws=float(W[0]['ws']);expiry=ws+300000.
    t=f.fill_ms.to_numpy(float);q=f.shares.to_numpy(float)
    p=f.price_cents.to_numpy(float);sign=1-2*f.side.to_numpy()
    times,mids,bids,asks=book(W)
    L.require(len(times)>0,'no valid valuation book')
    f['minute']=np.floor((t-ws)/60000).astype(int)+1
    L.require(((t>=ws)&(t<expiry)).all(),'fill outside contract')
    pre,age=Q.lookup(times,mids,t,before=True)
    f['pre_mid_cents']=pre;f['pre_mid_age_ms']=age
    j=np.searchsorted(times,t,side='left')-1;ic=np.maximum(j,0)
    depth=np.where(sign==1,bids[ic]-p,p-asks[ic]);depth=np.where(j>=0,depth,np.nan)
    f['depth_cents']=depth
    f['depth_group']=np.select([~np.isfinite(depth),depth<0,depth==0,depth<=5],['unknown','inside','at_best','one_to_five'],default='over_five')
    f['spread_capture_usd']=q*sign*(pre-p)/100
    for h in HORIZONS:
        mark,markage=Q.lookup(times,mids,t+h)
        mature=t+h<=expiry
        status=np.where(~mature,'expiry_censored',np.where(~np.isfinite(mark),'missing_book','ok'))
        mark=np.where(status=='ok',mark,np.nan)
        f[f'markout_{h}_status']=status
        f[f'markout_{h}_usd']=q*sign*(mark-p)/100
        f[f'mid_drift_{h}_usd']=q*sign*(mark-pre)/100
        f[f'mark_{h}_age_ms']=np.where(status=='ok',markage,np.nan)
    terminal,termage=Q.lookup(times,mids,np.array([expiry]))
    L.require(np.isfinite(terminal[0]),'terminal mark missing')
    y=float(W[0]['yup']);L.require(y in (0.,1.),'unknown outcome')
    f['terminal_pnl_usd']=q*sign*(terminal[0]-p)/100
    f['settlement_pnl_usd']=q*sign*(y-p/100)
    f['settlement_yes']=y;f['terminal_mid_cents']=terminal[0]
    f['terminal_mid_age_ms']=termage[0]
    return f


def engine_fills(W,r,D,day):
    rows=[];events=[]
    for (s,d),fills in r['fills'].items():
        g=D['LT'].iloc[D['LI'].get((int(W[0]['m']),s),np.zeros(0,int))]
        g=g.iloc[np.lexsort(((-g.ptick.to_numpy() if s==0 else g.ptick.to_numpy()),g.anc.to_numpy()))]
        L.require(np.array_equal(g.anc.to_numpy(),W[s]['anc']),'engine tape index alignment')
        for seq,(i,p,q,through,age,phase,ahead) in enumerate(fills):
            row=g.iloc[i];t=float(row.anc)-50.
            events.append((s,d,t,p,q))
            rows.append(dict(day=day,m=int(W[0]['m']),case='STOP_EV',side=s,rung=d,fill_seq=seq,
                fill_ms=t,anchor_ms=float(row.anc),price_cents=float(p),shares=float(q),
                row_id=int(row.row_id),signer=-1,ws_timed=bool(row.ws_timed),imputed=bool(row.imputed),
                age_ms=float(age),cancel_pending=phase=='pend',through=bool(through)))
    L.require(Counter(events)==Counter(r['fl']),'modeled fill event identity')
    columns=['day','m','case','side','rung','fill_seq','fill_ms','anchor_ms','price_cents','shares',
        'row_id','signer','ws_timed','imputed','age_ms','cancel_pending','through']
    return pd.DataFrame(rows,columns=columns)


def maker_fills(W,D,day,groups):
    m=int(W[0]['m']);g=D['LT'];g=g[g.mid==m].copy()
    t=g.anc.to_numpy(float)-50.;mask=(t>=W[0]['ws'])&(t<W[0]['ws']+300000.)
    g=g.loc[mask].copy();g['case']=g.signer.map(groups).fillna('OTHER_MAKERS')
    return pd.DataFrame(dict(day=day,m=m,case=g.case.to_numpy(),side=g.side.to_numpy(),
        rung=-1,fill_seq=g.row_id.to_numpy(),fill_ms=g.anc.to_numpy()-50.,anchor_ms=g.anc.to_numpy(),
        price_cents=g.ptick.to_numpy(float),shares=g.shares.to_numpy(float),row_id=g.row_id.to_numpy(),
        signer=g.signer.to_numpy(),ws_timed=g.ws_timed.to_numpy(),imputed=g.imputed.to_numpy(),
        age_ms=np.nan,cancel_pending=False,through=False))


def summarize_fills(f):
    q=f.shares.to_numpy(float)
    out=dict(fills=len(f),shares=float(q.sum()),buy_shares=float(f.loc[f.side==0,'shares'].sum()),
        sell_shares=float(f.loc[f.side==1,'shares'].sum()))
    for key in ['spread_capture_usd','terminal_pnl_usd','settlement_pnl_usd']:
        ok=np.isfinite(f[key]);out[key]=float(f.loc[ok,key].sum())
        out[key.replace('_usd','_covered_shares')]=float(f.loc[ok,'shares'].sum())
    out['pre_mid_stale_shares']=float(f.loc[f.pre_mid_age_ms>1000,'shares'].sum())
    for h in HORIZONS:
        status=f[f'markout_{h}_status'];ok=status=='ok'
        for state in ['ok','expiry_censored','missing_book']:
            out[f'markout_{h}_{state}_fills']=int((status==state).sum())
            out[f'markout_{h}_{state}_shares']=float(f.loc[status==state,'shares'].sum())
        out[f'markout_{h}_usd']=float(f.loc[ok,f'markout_{h}_usd'].sum())
        out[f'mid_drift_{h}_usd']=float(f.loc[ok,f'mid_drift_{h}_usd'].sum())
        out[f'markout_{h}_negative_usd']=float(f.loc[ok & (f[f'markout_{h}_usd']<0),f'markout_{h}_usd'].sum())
        out[f'markout_{h}_stale_shares']=float(f.loc[ok & (f[f'mark_{h}_age_ms']>1000),'shares'].sum())
    for key,mask in [('ws',f.ws_timed),('e2',~f.ws_timed),('imputed',f.imputed),
                     ('young',f.age_ms<100),('pending',f.cancel_pending)]:
        out[f'{key}_shares']=float(f.loc[mask,'shares'].sum())
    return out


def minute_inventory(W,r):
    ws=W[0]['ws'];times,mids,_,_=book(W);fills=sorted(r['fl'],key=lambda x:x[2])
    result=[];pos=cash=last_pnl=0.;i=0
    for minute in range(1,6):
        lo=ws+(minute-1)*60000;hi=ws+minute*60000
        area=0.;last=lo;maximum=abs(pos)
        while i<len(fills) and fills[i][2]<hi:
            s,d,t,p,q=fills[i];L.require(t>=lo,'inventory chronology')
            area+=abs(pos)*(t-last);pos+=(1-2*s)*q;cash-=(1-2*s)*q*p/100
            maximum=max(maximum,abs(pos));last=t;i+=1
        area+=abs(pos)*(hi-last)
        mark,_=Q.lookup(times,mids,np.array([hi]));pnl=cash+pos*mark[0]/100
        intervals={s:Q.union([(max(lo,a),min(hi,b)) for (ss,d),orders in r['orders'].items()
            if ss==s for a,b,_,ref in orders if not ref]) for s in (0,1)}
        both=Q.duration(intervals[0])+Q.duration(intervals[1])-Q.duration(intervals[0]+intervals[1])
        result.append(dict(minute=minute,position_end=pos,abs_position_end=abs(pos),cash_end_usd=cash,
            pnl_at_minute_end_usd=pnl,incremental_marked_pnl_usd=pnl-last_pnl,
            abs_inventory_share_ms=area,max_abs_inventory=maximum,
            quote_any_ms=Q.duration(intervals[0]+intervals[1]),quote_both_ms=both,observed_ms=60000.,
            orders=sum(lo<=o['ts']<hi for z in r['order_states'].values() for o in z),
            guard_pulls=sum(lo<=e[0]<hi and e[3]=='cancel:trig' for e in r['log'])))
        last_pnl=pnl
    L.require(i==len(fills),'inventory unaccounted fills')
    term=S.M.CL.portfolio_terminal(r,*W[0]['tf_mff'])
    Q.close(pos,term['position'],'inventory end position');Q.close(cash,term['cash'],'inventory cash')
    Q.close(last_pnl,term['pnl'],'minute/terminal PnL')
    Q.close(sum(z['guard_pulls'] for z in result),sum(r['stat']['pulls']),'guard pull partition')
    Q.close(max(z['max_abs_inventory'] for z in result),r['stat']['maxpos'],'max inventory partition')
    return result


def rank_makers(frame):
    eligible=(frame.t_rel>=0)&(frame.t_rel<240)
    order=frame.loc[eligible].groupby('signer').shares.sum().sort_values(ascending=False,kind='stable')
    return {int(s):('TOP5' if n<5 else 'OTHER_MAKERS') for n,s in enumerate(order.index)}


def selftest():
    checks=S.selftest()
    W={0:dict(ws=0.,yup=1.,tf=np.array([0.,100.,200.,59999.,60000.,299999.,300000.]),
        P5=np.array([[49],[50],[48],[51],[52],[55],[56]])),
       1:dict(P5=np.array([[51],[52],[50],[53],[54],[57],[58]]))}
    f=pd.DataFrame(dict(fill_ms=[100.,60000.,299900.],side=[0,1,0],shares=[10.,4.,3.],price_cents=[49.,54.,55.],
        ws_timed=True,imputed=False,age_ms=200.,cancel_pending=False))
    z=value_fills(f,W)
    L.require(z.minute.tolist()==[1,2,5],'minute bin boundary')
    Q.close(z.spread_capture_usd.iloc[0],.1,'strict before execution edge')
    Q.close(z.spread_capture_usd.iloc[1],.08,'ask execution edge sign')
    L.require(z.markout_1000_status.iloc[-1]=='expiry_censored' and np.isnan(z.markout_1000_usd.iloc[-1]),'expiry censor control')
    L.require(z.markout_100_status.iloc[-1]=='ok','exact expiry horizon allowed')
    for row in z.itertuples():
        for h in HORIZONS:
            if getattr(row,f'markout_{h}_status')=='ok':Q.close(row.spread_capture_usd+getattr(row,f'mid_drift_{h}_usd'),getattr(row,f'markout_{h}_usd'),'markout identity')
    bad=f.copy();bad.loc[0,'fill_ms']=-1;caught=False
    try:value_fills(bad,W)
    except ValueError:caught=True
    L.require(caught,'out-of-contract fill refused')
    fixture=pd.DataFrame(dict(signer=list(range(6))+[5],shares=[10.,9.,8.,7.,6.,1.,999.],t_rel=[0.]*6+[250.]))
    ranks=rank_makers(fixture)
    L.require(ranks[5]=='OTHER_MAKERS' and ranks[4]=='TOP5','ranking uses declared interval')
    wrong=fixture.assign(t_rel=0.)
    L.require(rank_makers(wrong)[5]=='TOP5','wrong ranking interval positive control')
    checks.update(minute_boundary=True,expiry_censor=True,outside_contract_refused=True,ranking_interval=True)
    return checks
