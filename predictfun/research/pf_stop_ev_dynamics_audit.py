"""Input dynamics, scales and tails audit of fixed STOP-EV models.

Nine consumed dates only; no fits, policy changes, collector tape or protected
dates. Current 83 inputs + 22 predefined descriptors from stored causal 50 ms
grid rows. New history descriptors preserve the preceding trajectory, but
cannot reconstruct intra-grid events or distinguish executions/cancellations.
Fixed-price depth follows the focal order's price, not a changing best level.
Grid gaps / missing history / outside visible book are NaN + counted status.

All features get distribution/scale profiles, train-range and tail checks.
Five later consumed days use each fold's reserved-day 1st percentile of value
per known remaining share to define severe losses. No losses trimmed. Report
univariate severe-vs-other-filled discrimination for existing flow/queue inputs
and all new descriptors; matched AUC conditions on day/side/rung/edge/P(fill)
and saved-score quintiles from each earlier calibration day, plus hour as a
finer sensitivity. These are information diagnostics, not learned model
improvement, confidence statements or an independent validation sample.

Tests: fixed-price round-trip depletion with equal endpoints; future-row
perturbation invariance; grid-gap refusal; volume-scaling equivariance;
matched oracle and tied signal; saved prediction identity corruptions refused.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from pf_stop_ev_generalization_review import validate,selftest as parity_controls

ROOT=Path('/home/yuqing/ctaNew/data/predict_fun/research')
MODEL=ROOT/'stop_ev_model_20261115';RAW=ROOT/'stop_model_20261114'
DEV=('20261023','20261024','20261025','20261026')
TEST=('20261027','20261031','20261101','20261102','20261103');DAYS=DEV+TEST
KEY=['day','m','side','rung','generation','decision_ms'];RK=['m','side','k']
FLOW=['adv_ofi250','adv_ofi1000','adv_bnd_ofi250','adv_bnd_ofi1000']
CONTEXT=['log_w30','qty_over_w30','bn_flow_relative30','pf_ofi_acceleration','bn_flow_acceleration']
PATH=['pf_ofi_peak1s','pf_ofi_absolute_mean1s','pf_adverse_fraction1s','bn_flow_peak1s','bn_ofi_peak1s','joint_adverse_fraction1s']
LEVEL=[f'level_{name}{h}' for h in (250,1000) for name in ['depletion','refill','peak_drop','net_depletion']]
EXTRA=CONTEXT+PATH+LEVEL+['depth_ahead_over_w30','level_depletion_over_order_and_shown','log_pseudocount_effect']
BASE=['adv_ofi250','adv_ofi1000','adv_bnd_ofi250','adv_bnd_ofi1000','adv_bn_flow250','adv_bn_flow1000','adv_bn_flow5000',
      'own_d250','opp_d250','adv_dd3_250','adv_dd3_1000','act_rel','bn_ntr1000','o_q_rel','o_q_frac','o_edge','dfv250','sigma']
MATCH=['day','side','rung','edge_band','pfill_band','model_band']


def require(ok,msg):
    if not bool(ok):raise ValueError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def guard():require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2048*1024,'RSS cap')


def stats(x):
    x=np.asarray(x,float);v=x[np.isfinite(x)];a=np.abs(v)
    if not len(v):return dict(n=len(x),finite=0,missing=int(np.isnan(x).sum()),inf=int(np.isinf(x).sum()))
    q=np.quantile(v,[0,.001,.01,.5,.99,.999,1]);aq=np.quantile(a,[.5,.99,.999,1])
    return dict(n=len(x),finite=len(v),missing=int(np.isnan(x).sum()),inf=int(np.isinf(x).sum()),zero=int((v==0).sum()),
        quantiles=dict(zip(['min','p001','p01','p50','p99','p999','max'],map(float,q))),
        abs_quantiles=dict(zip(['p50','p99','p999','max'],map(float,aq))),
        max_over_abs_p99=float(aq[-1]/aq[1]) if aq[1]>0 else None,mean=float(v.mean()),std=float(v.std()))


def histories(r,d,flow_scale):
    require(not r.duplicated(RK).any(),'duplicate history key')
    lag=np.arange(20,-1,-1);n=len(d)
    idx=pd.MultiIndex.from_frame(r[RK])
    want=pd.MultiIndex.from_arrays([np.repeat(d.m.to_numpy(),21),np.repeat(d.side.to_numpy(),21),
        (d.k.to_numpy()[:,None]-lag).ravel()])
    ix=idx.get_indexer(want).reshape(n,21);safe=np.maximum(ix,0)
    valid=(ix>=0)&(np.abs(r.t.to_numpy()[safe]-(d.decision_ms.to_numpy()[:,None]-lag*50))<.001)
    require(valid[:,-1].all(),'current action feature time/key mismatch')
    scale=r.scale_w30.to_numpy(float)[safe[:,-1]];require(np.isfinite(scale).all() and (scale>0).all(),'bad W30 scale')
    def get(c):return np.where(valid,r[c].to_numpy(float)[safe],np.nan)
    q=d.x_known_remaining.to_numpy(float);require(np.isfinite(q).all() and (q>0).all(),'invalid remaining')
    hist_scale=get('scale_w30');pf=get('adv_ofi250')*hist_scale/scale[:,None]
    p1=get('adv_ofi1000');bn=get('adv_bn_flow250');bnd=get('adv_bnd_ofi250')
    full=valid.all(axis=1);out={}
    out['log_w30']=np.log(scale);out['qty_over_w30']=q/scale
    fs=np.asarray(flow_scale,float)[safe[:,-1]]
    out['bn_flow_relative30']=np.divide(bn[:,-1],fs,out=np.full(n,np.nan),where=np.isfinite(fs)&(fs>0))
    out['pf_ofi_acceleration']=(4*pf[:,-1]-p1[:,-1])/3
    out['bn_flow_acceleration']=(4*bn[:,-1]-get('adv_bn_flow1000')[:,-1])/3
    def path(name,a,fn):
        ok=full&np.isfinite(a).all(axis=1);out[name]=np.where(ok,fn(a),np.nan)
    path('pf_ofi_peak1s',pf,lambda a:np.max(a,axis=1))
    path('pf_ofi_absolute_mean1s',pf,lambda a:np.mean(np.abs(a),axis=1))
    path('pf_adverse_fraction1s',pf,lambda a:np.mean(a>0,axis=1))
    path('bn_flow_peak1s',bn,lambda a:np.max(a,axis=1))
    path('bn_ofi_peak1s',bnd,lambda a:np.max(a,axis=1))
    joint=np.where(np.isfinite(pf)&np.isfinite(bnd),((pf>0)&(bnd>0)).astype(float),np.nan)
    path('joint_adverse_fraction1s',joint,lambda a:a.mean(axis=1))
    best=get('best');gaps=np.stack([np.zeros_like(best)]+[get(f'g{i}') for i in range(2,6)],axis=2)
    size=np.stack([get(f's{i}') for i in range(1,6)],axis=2)
    sg=np.where(d.side.to_numpy()==0,1.,-1.);price=d.o_price.to_numpy(float)
    px=best[:,:,None]-sg[:,None,None]*gaps;levels=(gaps>=0)&np.isfinite(size)&(size>=0)
    visible=valid&~((gaps[:,:,4]>=0)&(sg[:,None]*(price[:,None]-px[:,:,4])<0))
    shown=np.where(visible,np.where(levels&(px==price[:,None,None]),size,0.).sum(axis=2),np.nan)
    ahead=np.where(visible,np.where(levels&(sg[:,None,None]*(px-price[:,None,None])>0),size,0.).sum(axis=2),np.nan)
    for h in (250,1000):
        a=shown[:,-(h//50+1):];ok=np.isfinite(a).all(axis=1);delta=np.diff(a,axis=1)
        out[f'level_depletion{h}']=np.where(ok,np.maximum(-delta,0).sum(axis=1)/scale,np.nan)
        out[f'level_refill{h}']=np.where(ok,np.maximum(delta,0).sum(axis=1)/scale,np.nan)
        out[f'level_peak_drop{h}']=np.where(ok,np.maximum(-delta,0).max(axis=1)/scale,np.nan)
        out[f'level_net_depletion{h}']=np.where(ok,(a[:,0]-a[:,-1])/scale,np.nan)
    out['depth_ahead_over_w30']=ahead[:,-1]/scale
    out['level_depletion_over_order_and_shown']=out['level_depletion1000']*scale/(q+shown[:,-1])
    # Current own_l1 comes from log1p(size), so subtracting log(scale) is NOT
    # precisely log(size/scale). Quantify that pseudocount effect separately.
    own=get('s1')[:,-1]
    out['log_pseudocount_effect']=np.where(own>0,np.log1p(1/np.maximum(own,1e-300)),np.nan)
    require(set(out)==set(EXTRA),'descriptor contract')
    status=dict(actions=n,history_250_grid_complete=int(valid[:,-6:].all(axis=1).sum()),history_1000_grid_complete=int(full.sum()),
        fixed_price_visible_now=int(visible[:,-1].sum()),fixed_price_visible_history250=int(np.isfinite(shown[:,-6:]).all(axis=1).sum()),
        fixed_price_visible_history1000=int(np.isfinite(shown).all(axis=1).sum()),bn_flow_scale_valid=int((np.isfinite(fs)&(fs>0)).sum()))
    return pd.DataFrame(out),status


def net_flow_scale(r):
    out=np.full(len(r),np.nan)
    for ix in r.groupby(['m','side'],sort=False).indices.values():
        ix=np.asarray(ix);t=r.t.to_numpy()[ix]
        require((np.diff(t)>0).all(),'nonmonotone history')
        ser=pd.Series(np.abs(r.adv_bn_flow250.to_numpy()[ix]),index=pd.to_datetime(t,unit='ms'))
        out[ix]=ser.rolling('30s',min_periods=20).median().to_numpy()
    return out


def auc(y,x):
    y=np.asarray(y,bool);x=np.asarray(x,float);p=int(y.sum());n=len(y)-p
    return float((rankdata(x)[y].sum()-p*(p+1)/2)/(p*n)) if p and n else None


def discriminate(d,c,hour=False):
    x=d[c].to_numpy(float);ok=np.isfinite(x)&(d.filled_quantity.to_numpy()>0)
    z=d.iloc[np.flatnonzero(ok)].copy();pairs=conc=used=groups=0
    for ix in z.groupby(MATCH+(['hour'] if hour else []),dropna=False,sort=False).indices.values():
        g=z.iloc[ix];y=g.severe.to_numpy(bool);p=int(y.sum());n=len(g)-p
        if p and n:pairs+=p*n;conc+=auc(y,g[c])*p*n;used+=len(g);groups+=1
    return dict(n=len(z),severe=int(z.severe.sum()),raw_auc=auc(z.severe,z[c]),
        matched_auc=float(conc/pairs) if pairs else None,matched_n=used,pairs=pairs,groups=groups)


def population(d):
    y=d.value_cents.to_numpy();loss=np.maximum(-y,0)
    return dict(n=len(d),filled=int((d.filled_quantity>0).sum()),severe=int(d.severe.sum()),
        value_cents=float(y.sum()),gross_loss_cents=float(loss.sum()),severe_loss_cents=float(loss[d.severe].sum()),
        raw_kept_severe=int((d.severe&(d.pred>=0)).sum()))


def tail_mass(y):
    y=np.asarray(y,float);order=np.argsort(np.abs(y));den=np.square(y).sum();loss=np.maximum(-y,0)
    return {str(q):dict(n=int(np.ceil(q*len(y))),squared_value_share=float(np.square(y[order[-int(np.ceil(q*len(y))):]]).sum()/den) if den else None,
        gross_loss_share=float(loss[order[-int(np.ceil(q*len(y))):]].sum()/loss.sum()) if loss.sum() else None) for q in [.001,.01]}


def error_mass(d):
    y=d.value_cents.to_numpy();f=d.filled_quantity.to_numpy()>0;order=np.argsort(np.abs(y));out={}
    ev=(y-d.pred.to_numpy())**2;vh=np.where(f,(y-d.v_fill.to_numpy())**2,0.)
    for q in [.001,.01]:
        ix=order[-int(np.ceil(q*len(d))):]
        out[str(q)]=dict(n=len(ix),filled=int(f[ix].sum()),ev_squared_error_share=float(ev[ix].sum()/ev.sum()),
            filled_value_head_squared_error_share=float(vh[ix].sum()/vh.sum()))
    return out


def selftest():
    c=parity_controls();n=22;r=pd.DataFrame(dict(m=np.ones(n,int),side=np.zeros(n,int),k=np.arange(n),t=np.arange(n)*50.,
        scale_w30=np.full(n,100.),best=np.full(n,50.),adv_bn_flow250=np.zeros(n),adv_bn_flow1000=np.zeros(n)))
    for k in FLOW:r[k]=0.
    for i in range(1,6):r[f's{i}']=100.
    for i in range(2,6):r[f'g{i}']=i-1
    d=pd.DataFrame(dict(m=[1],side=[0],k=[20],decision_ms=[1000.],o_price=[50.],x_known_remaining=[30.]))
    r.loc[10,'s1']=20.;a,_=histories(r,d,np.ones(n))
    require(a.level_depletion1000.iloc[0]==.8 and a.level_refill1000.iloc[0]==.8 and a.level_net_depletion1000.iloc[0]==0,'round trip erased')
    ask=r.copy();ask['side']=1;da=d.copy();da['side']=1;b,_=histories(ask,da,np.ones(n))
    require(a.level_depletion1000.equals(b.level_depletion1000),'ask mirror')
    moving=r.copy();moving.loc[:9,'best']=51.;moving.loc[:9,'s1']=999.;b,_=histories(moving,d,np.ones(n))
    require(a.level_depletion1000.equals(b.level_depletion1000),'followed moving best rather than fixed price')
    fut=r.copy();fut.loc[21,'s1']=99999.;b,_=histories(fut,d,np.ones(n));require(a.equals(b),'future read')
    gap=r.drop(index=10).reset_index(drop=True);b,st=histories(gap,d,np.ones(len(gap)))
    require(np.isnan(b.level_depletion1000.iloc[0]) and st['history_1000_grid_complete']==0,'gap not refused')
    scaled=r.copy()
    for k in ['scale_w30']+[f's{i}' for i in range(1,6)]:scaled[k]*=10
    dd=d.copy();dd.x_known_remaining*=10;b,_=histories(scaled,dd,np.ones(n))
    require(a.level_depletion1000.equals(b.level_depletion1000) and a.qty_over_w30.equals(b.qty_over_w30),'scale equivariance')
    toy=pd.DataFrame({k:[0]*4 for k in MATCH+['hour']});toy['filled_quantity']=1.;toy['severe']=[False,False,True,True];toy['x']=[0,0,1,1]
    require(discriminate(toy,'x')['matched_auc']==1.,'matched oracle');toy['x']=1
    require(discriminate(toy,'x')['matched_auc']==.5,'matched null control')
    bad=r.copy();bad.loc[20,'scale_w30']=0
    try:histories(bad,d,np.ones(n))
    except ValueError:c['zero_scale_refused']=True
    else:raise ValueError('zero scale accepted')
    c.update(roundtrip_depletion_detected=True,ask_mirror=True,fixed_price_not_moving_best=True,
        future_history_invariant=True,gap_refused=True,scale_equivariance=True,matched_oracle=True)
    return c


def run(out):
    require(not out.exists(),'output exists');out.mkdir(parents=True);controls=selftest();hashes={};panels={};build={};profiles={}
    def read(p,cols=None):
        h=sha(p);require(str(p) not in hashes or hashes[str(p)]==h,'input changed');hashes[str(p)]=h
        return pd.read_parquet(p,columns=cols) if p.suffix=='.parquet' else json.loads(p.read_text())
    F=read(MODEL/f'fit2_{TEST[0]}.json')['features'];require(len(F)==83,'feature contract')
    rawcols=RK+['t','scale_w30','best','adv_bn_flow250','adv_bn_flow1000']+[f's{i}' for i in range(1,6)]+[f'g{i}' for i in range(2,6)]
    for day in DAYS:
        all_d=read(MODEL/f'ev_x2_{day}.parquet');d=all_d[all_d.cert_est].reset_index(drop=True)
        r=read(RAW/f'rows_{day}.parquet',rawcols);x=read(RAW/f'rows_{day}_x5.parquet',RK+FLOW)
        require(r[RK].equals(x[RK]),'x5 alignment')
        for c in FLOW:r[c]=x[c].to_numpy()
        extra,status=histories(r,d,net_flow_scale(r))
        for c in EXTRA:d[c]=extra[c].to_numpy()
        panels[day]=d;build[day]=dict(**status,original_actions=len(all_d),certified_actions=len(d),
            source_status=read(MODEL/f'ev_x_{day}.json')['status'],certificate=read(MODEL/f'ev_x2_{day}.json')['checks'])
        profiles[day]={c:stats(d[c]) for c in F+EXTRA}
        (out/f'profile_{day}.json').write_text(json.dumps(profiles[day],indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(day=day,history=status)),flush=True)
        del all_d,r,x,extra;gc.collect();guard()
    folds={};tests=[]
    for day in TEST:
        fit=read(MODEL/f'fit2_{day}.json');tr=fit['train_days'];cal=fit['cal'];require(max(tr)<cal<day,'chronology')
        require(fit['features']==F,'fold feature mismatch')
        for dd in tr+[cal,day]:require(hashes[str(MODEL/f'ev_x2_{dd}.parquet')]==fit['inputs_sha256'][dd],'fit input hash')
        train=pd.concat([panels[k] for k in tr],ignore_index=True);d=panels[day].copy();cald=panels[cal]
        require(len(train)==fit['n_train'],'training count')
        for dd in [cal,day]:
            orig=read(MODEL/f'ev_x2_{dd}.parquet');p=read(MODEL/f'pred2_{day}_{dd}.parquet');validate(orig,p)
            if dd==cal:pc=p[p.cert_est].pred.to_numpy()
            else:
                p=p[p.cert_est].reset_index(drop=True);d['pred']=p.pred;d['p_fill']=p.p_fill;d['v_fill']=p.v_fill
        cutoff=float(np.quantile(cald.value_cents/cald.x_known_remaining,.01));d['severe']=(d.value_cents/d.x_known_remaining<=cutoff)
        d['edge_band']=np.floor(d.o_edge).astype(int);d['pfill_band']=np.searchsorted([.05,.1,.2,.4],d.p_fill)
        d['model_band']=np.searchsorted(np.unique(np.quantile(pc,[.2,.4,.6,.8])),d.pred)
        extreme=np.zeros(len(d),bool);outside=np.zeros(len(d),bool);drift={}
        for c in F:
            a=train[c].to_numpy(float);a=a[np.isfinite(a)];b=d[c].to_numpy(float);ok=np.isfinite(b)
            if not len(a):continue
            lo,hi=np.quantile(a,[.001,.999]);ex=ok&((b<lo)|(b>hi));oo=ok&((b<a.min())|(b>a.max()))
            extreme|=ex;outside|=oo
            drift[c]=dict(train_n=len(a),test_finite=int(ok.sum()),train_q001=float(lo),train_q999=float(hi),
                test_extreme_n=int(ex.sum()),test_outside_range_n=int(oo.sum()),test_missing=int((~ok).sum()),
                severe_extreme_n=int((ex&d.severe).sum()),raw_kept_severe_extreme_n=int((ex&d.severe&(d.pred>=0)).sum()))
        d['any_feature_extreme']=extreme;d['any_feature_outside_range']=outside
        groups={'all':np.ones(len(d),bool),'filled':d.filled_quantity.to_numpy()>0,'good':d.value_cents.to_numpy()>0,
            'severe':d.severe.to_numpy(),'raw_kept_good':(d.value_cents>0)&(d.pred>=0),'raw_kept_severe':d.severe&(d.pred>=0)}
        health={}
        for name,mask in groups.items():
            g=d[mask];health[name]=dict(n=len(g),any_extreme=int(g.any_feature_extreme.sum()),outside_range=int(g.any_feature_outside_range.sum()),
                fair_missing=int(g.dfv250.isna().sum()),queue_missing=int(g.o_q_frac.isna().sum()),
                level_history_missing=int(g.level_depletion1000.isna().sum()),flow_history_missing=int(g.pf_ofi_peak1s.isna().sum()),
                log_scale=stats(g.log_w30),bn_flow250=stats(g.adv_bn_flow250),bn_flow_relative30=stats(g.bn_flow_relative30),
                ofi250=stats(g.adv_ofi250),age=stats(g.o_age))
        rec=dict(day=day,cal_day=cal,train_days=tr,train_n=len(train),cal_n=len(cald),tail_cutoff_cents_per_remaining_share=cutoff,
            population=population(d),health=health,input_drift=drift,
            label_tail_mass=dict(train=tail_mass(train.value_cents),test=tail_mass(d.value_cents)),
            test_error_tail_mass=error_mass(d),
            discriminators={c:discriminate(d,c) for c in BASE+EXTRA})
        folds[day]=rec;tests.append(d);(out/f'fold_{day}.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(day=day,population=rec['population'],tail_mass=rec['label_tail_mass'],health={k:{f:v for f,v in z.items() if not isinstance(v,dict)} for k,z in health.items()})),flush=True)
        del train;gc.collect();guard()
    d=pd.concat(tests,ignore_index=True);pooled={}
    for c in BASE+EXTRA:
        pooled[c]=dict(primary=discriminate(d,c),with_hour=discriminate(d,c,True),strict=discriminate(d[d.cert_ws].reset_index(drop=True),c))
    health={}
    for name,mask in [('all',np.ones(len(d),bool)),('good',d.value_cents>0),('severe',d.severe),('raw_kept_good',(d.value_cents>0)&(d.pred>=0)),('raw_kept_severe',d.severe&(d.pred>=0))]:
        g=d[mask];health[name]=dict(n=len(g),any_extreme=int(g.any_feature_extreme.sum()),outside_range=int(g.any_feature_outside_range.sum()),
            fair_missing=int(g.dfv250.isna().sum()),queue_missing=int(g.o_q_frac.isna().sum()),
            level_history_missing=int(g.level_depletion1000.isna().sum()),flow_history_missing=int(g.pf_ofi_peak1s.isna().sum()))
    d[KEY+['value_cents','filled_quantity','pred','severe','o_edge','o_q_frac','any_feature_extreme','any_feature_outside_range']+EXTRA].to_parquet(out/'action_diagnostics.parquet',index=False)
    for p,h in hashes.items():require(sha(p)==h,'input changed during audit')
    result=dict(asof=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        controls=controls,input_sha256=hashes,build=build,folds=folds,features=F,extra=EXTRA,pooled=pooled,health=health,
        population=population(d),strict_population=population(d[d.cert_ws]),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        interpretation='Descriptive dynamics/scale/tail audit on consumed data. No new model, no raw collector rebuild, no promoted features or policy; univariate associations do not prove incremental model skill.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(asof=result['asof'],population=result['population'],health=health)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--selftest',action='store_true');a=p.parse_args()
    if a.selftest:print(json.dumps(selftest()))
    else:require(a.out is not None,'--out required');run(a.out)
