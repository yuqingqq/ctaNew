"""Zero modeled feed-delay inputs for the frozen STOP-EV latency experiment.

PF frames/pending increments known at venue stamps, Binance ticker/trade/depth
known at event E. Relay keeps the established receipt-minus-72ms clock (no
extra decision delay). The pre-window sigma context is fixed. 50ms sampling
remains: this is zero modeled transport/processing delay, not continuous
reaction or knowledge of events absent from the recorded tape.
"""
from pathlib import Path
import collections
import math
import numpy as np
import pandas as pd
from scipy.special import ndtr
import pf_stop_ev_play as P
import pf_stop_model_x5 as X5
import pf_io as IO

M,K=P.M,P.K
DEPTH=[c.removeprefix('adv_') for c in X5.BN_DEPTH]


def require(ok,msg):
    if not bool(ok):raise ValueError(msg)


def hours(day):
    return [h for h in IO.hours_of(day,extra_before=1) if K.part_allowed(f'part_{h}.parquet',day)]


def at(t,v,g):
    i=np.searchsorted(t,g,side='right')-1
    return np.where(i>=0,v[np.maximum(i,0)],np.nan)


def sums(t,v,g,h):
    c=np.r_[0.,np.cumsum(v)];return c[np.searchsorted(t,g,side='right')]-c[np.searchsorted(t,g-h,side='right')]


def exchange_inputs(day,grid,track):
    """Stream raw hourly files; only cached decision times receive outputs."""
    hs=hours(day);n=len(grid);out={c:np.full(n,np.nan) for c in K.G32 if c.startswith('bn_')}
    out['bn_mid']=np.full(n,np.nan)
    for c in DEPTH:out[c]=np.full(n,np.nan)
    ref=[]
    for h in hs:
        track(Path(IO.REF_RUN)/'reference_prices'/f'{h}.jsonl.gz')
        ref.extend((r[0]/1e6-72.,float(r[2])) for r in IO.ref_rows(h) if not r[3])
    ref=np.asarray(sorted(ref),float).reshape(-1,2);rv=np.full(len(ref),np.nan)
    carry=None;tcarry=None;dprev=None
    st=collections.Counter()
    for h in hs:
        p=Path(IO.BN_DIR)/f'{h}.csv.gz';track(p)
        if not p.exists():st['ticker_missing']+=1;continue
        d=pd.read_csv(p,header=None,names=['recv','E','T','u','bid','bq','ask','aq'],usecols=['E','u','bid','bq','ask','aq'])
        d=d.apply(pd.to_numeric,errors='coerce').dropna();d=d[(d.bid>0)&(d.ask>d.bid)].drop_duplicates('u',keep='first')
        o=np.argsort(d.E.to_numpy(),kind='stable');t=d.E.to_numpy(float)[o]
        mid=((d.bid+d.ask)/2).to_numpy()[o];imb=((d.bq-d.aq)/(d.bq+d.aq)).to_numpy()[o];del d
        start=carry[0][-1] if carry is not None else -np.inf
        if carry is not None:t=np.r_[carry[0],t];mid=np.r_[carry[1],mid];imb=np.r_[carry[2],imb]
        sel=np.flatnonzero((grid>start)&(grid>=t[0])&(grid<=t[-1]));g=grid[sel]
        out['bn_mid'][sel]=at(t,mid,g);out['bn_imb'][sel]=at(t,imb,g);out['bn_age'][sel]=g-at(t,t,g)
        for hh in [25,50,100,250,500,1000,3000]:out[f'bn_r{hh}'][sel]=1e4*np.log(at(t,mid,g)/at(t,mid,g-hh))
        if len(ref):
            ri=np.flatnonzero(np.isnan(rv)&(ref[:,0]>=t[0])&(ref[:,0]<=t[-1]));rv[ri]=ref[ri,1]/at(t,mid,ref[ri,0])
        tp=Path(IO.ROOT)/'data/mm_hf/raw/trade/BTCUSDT'/f'{h}.csv.gz';track(tp)
        if tp.exists():
            td=pd.read_csv(tp,header=None,names=['recv','E','T','id','px','qty','bm'],usecols=['E','qty','bm'])
            td=td.apply(pd.to_numeric,errors='coerce').dropna();o=np.argsort(td.E.to_numpy(),kind='stable')
            tt=td.E.to_numpy(float)[o];sq=(np.where(td.bm==0,1.,-1.)*td.qty.to_numpy())[o];del td
        else:tt=np.zeros(0);sq=np.zeros(0);st['trade_missing']+=1
        if tcarry is not None:tt=np.r_[tcarry[0],tt];sq=np.r_[tcarry[1],sq]
        if len(tt):
            for hh in [250,1000,5000]:out[f'bn_flow{hh}'][sel]=sums(tt,sq,g,hh)
            out['bn_ntr1000'][sel]=np.searchsorted(tt,g,side='right')-np.searchsorted(tt,g-1000,side='right')
            keep=tt>=tt[-1]-10000;tcarry=(tt[keep],sq[keep])
        dh=([dprev] if dprev else [])+[h]
        for hh in dh:track(Path(X5.DEPTH_DIR)/f'{hh}.csv.gz')
        depth,ds=X5.depth_arrays(dh,0.)
        if depth is not None:
            z=X5.depth_feats(depth,g)
            for c in DEPTH:out[c][sel]=z[c]
        else:st['depth_missing_hours']+=1
        dprev=h;keep=t>=t[-1]-10000;carry=(t[keep],mid[keep],imb[keep])
        st['hours']+=1;IO.mem_check('zero latency raw hour')
        print(f'zero inputs {day} hour {h} rows {len(sel)}',flush=True)
    ok=np.isfinite(rv);rt=ref[ok,0];ratio=rv[ok]
    out['basis']=M.relay_basis(rt,ratio,grid)
    return out,dict(st)


def predict_inputs(b,tau,pg):
    """Predict step-5 quantities, with the knowledge clock equal to venue time."""
    tv=b.t_ms.to_numpy(float);bb=b.b1_p.to_numpy();ba=b.a1_p.to_numpy();bs=b.b1_s.to_numpy();ass=b.a1_s.to_numpy()
    valid=(bb>=0)&(ba>bb);mid=np.where(valid,(bb+ba)/2.,np.nan);mf=pd.Series(mid).ffill().to_numpy()
    n=len(tau);i=np.searchsorted(tv,tau,side='right')-1;ic=np.maximum(i,0);vv=(i>=0)&valid[ic]
    def v(x):return np.where(vv,x[ic],np.nan)
    z=dict(pf_spread=v(ba-bb),pf_mid=v(mid),pf_micro_dev=v((bb*ass+ba*bs)/np.maximum(bs+ass,1e-9)-mid),
        pf_imb1=v((bs-ass)/np.maximum(bs+ass,1e-9)),pf_lbs=v(np.log1p(bs)),pf_las=v(np.log1p(ass)),
        pf_age=tau-at(tv,tv,tau),pf_nfr1000=np.searchsorted(tv,tau,side='right')-np.searchsorted(tv,tau-1000,side='right'))
    sb=sum(np.where(b[f'b{j}_p']>=0,b[f'b{j}_s'],0.) for j in [1,2,3]);sa=sum(np.where(b[f'a{j}_p']>=0,b[f'a{j}_s'],0.) for j in [1,2,3])
    z['pf_imb3']=v((sb-sa)/np.maximum(sb+sa,1e-9))
    for h in [250,1000,5000]:z[f'pf_dm{h}']=at(tv,mf,tau)-at(tv,mf,tau-h)
    for c in ['pf_pend_bid1000','pf_pend_ask1000','pf_pend_tot5000']:z[c]=np.zeros(n)
    if pg is not None and len(pg):
        pg=pg.sort_values(['t_ms','recv_ns'],kind='stable');t=pg.t_ms.to_numpy(float);inc=pg.inc.to_numpy(float)
        z['pf_pend_bid1000']=sums(t,np.where(pg.side==0,inc,0.),tau,1000)
        z['pf_pend_ask1000']=sums(t,np.where(pg.side==1,inc,0.),tau,1000)
        z['pf_pend_tot5000']=sums(t,inc,tau,5000)
    return z


def build(day,D,track):
    ordered=sorted(D['ROWS'],key=D['W0'].get);grid=np.concatenate([D['ROWS'][m].tau.to_numpy(float) for m in ordered])
    ex,st=exchange_inputs(day,grid,track)
    pp=Path(K.IO)/'step1'/day/'pend_inc.parquet';track(pp);pend=pd.read_parquet(pp)
    pb={int(m):g for m,g in pend.groupby('mid',sort=False)}
    rp=Path(K.IO)/'step1'/day/'registry.parquet';track(rp);reg=pd.read_parquet(rp,columns=['mid','start_price'])
    starts=dict(zip(reg.mid,reg.start_price));pos=0;out={}
    for m in ordered:
        r=D['ROWS'][m].copy();n=len(r);sl=slice(pos,pos+n);pos+=n;t=r.tau.to_numpy(float)
        if m not in D['BI']:continue
        b=D['B'].iloc[D['BI'][m]].sort_values('t_ms',kind='stable');z=predict_inputs(b,t,pb.get(m))
        for c in K.G32:
            if c in ex:r[c]=ex[c][sl]
            if c in z:r[c]=z[c]
        sigma=r.sigma.to_numpy(float);tr=300.-r.t_rel.to_numpy(float)
        fair=100.*ndtr(np.log(ex['bn_mid'][sl]*ex['basis'][sl]/float(starts[m]))/(sigma*np.sqrt(np.maximum(tr,1e-3))))
        r['gap']=fair-r.pf_mid.to_numpy(float)
        r['d_lbs250']=r.pf_lbs-r.pf_lbs.shift(5);r['d_las250']=r.pf_las-r.pf_las.shift(5);r['d_bnimb250']=r.bn_imb-r.bn_imb.shift(5)
        lt=np.log1p(r.bn_ntr1000);r['act_rel']=lt-lt.rolling(600,min_periods=20).mean()
        out[m]=dict(rows=r,pb=(t.tolist(),fair,tr,sigma,r.gap.to_numpy(float)),
            depth={c:ex[c][sl] for c in DEPTH})
    return out,st


def window(W,z):
    W2={};r=z['rows'];t=W[0]['tg'];jj=np.searchsorted(r.tau.to_numpy(),t)
    require(np.array_equal(r.tau.to_numpy()[jj],t),'zero input grid mismatch')
    for s,w in W.items():
        kf=w['tf'];jk=np.searchsorted(kf,t,side='right')-1;ic=np.maximum(jk,0);own=w['P5'][:,0];opp=w['opp']
        ok=(jk>=0)&(own[ic]>=0)&(opp[ic]>=0)&((opp[ic]>own[ic]) if s==0 else (own[ic]>opp[ic]))
        ex=np.stack([np.asarray(w['vkl']),r['d_lbs250' if s==0 else 'd_las250'].to_numpy()[jj],
            r['d_las250' if s==0 else 'd_lbs250'].to_numpy()[jj],r.act_rel.to_numpy()[jj],r.d_bnimb250.to_numpy()[jj]],axis=1)
        W2[s]=dict(w,kf=kf,jk=jk,best=np.where(ok,own[ic],-1).tolist(),bopp=np.where(ok,opp[ic],-1).tolist(),
            X=np.column_stack([r[K.G32].to_numpy()[jj],ex]),cache={})
    scale,_=M.pair_scales(W2);po=X5.pair_ofi(W2);out={}
    for s,w in W2.items():
        _,valid,rs,_=M.QR.runs(w['tf'],w['P5'],w['opp'],s);rso=M.run_starts(np.asarray(w['opp'],'int64'),valid)
        ks=np.flatnonzero((w['jk']>=0)&(np.asarray(w['best'])>=0));nanp=np.full(w['n'],np.nan)
        vals=[M.all_feats(w,int(k),s,rs,rso,nanp,z['pb'],bool(w['vkl'][k]==M.MASK_V)) for k in ks]
        raw=pd.DataFrame(np.asarray(vals,np.float32),columns=M.FEAT);raw['scale_w30']=scale[ks]
        x5=X5.side_inputs(po,{c:z['depth'][c][jj][ks] for c in DEPTH},scale,s,ks)
        for c,v in x5.items():raw[c]=v
        x=M.model_matrix(raw,'W30',P.V.F71);ix=np.full(w['n'],-1,int);ix[ks]=np.arange(len(ks))
        out[s]=(ix,x,scale[ks],x[:,P.V.F71.index('pf_mid')])
    return W2,out


def selftest():
    t=np.array([0.,25.,100.]);v=np.array([1.,2.,3.]);g=np.array([50.])
    require(at(t,v,g)[0]==2.,'future lookup')
    require(sums(t,v,g,30)[0]==2.,'rolling event boundary')
    vv=v.copy();vv[-1]=999.;require(np.array_equal(at(t,v,g),at(t,vv,g)),'future perturbation')
    # The bound must reveal an event earlier than a delayed-clock lookup.
    require(at(t+40,v,g)[0]==1. and at(t,v,g)[0]==2.,'zero clock positive control')
    return dict(event_lookup=True,rolling_boundary=True,future_invariant=True,zero_clock_control=True)
