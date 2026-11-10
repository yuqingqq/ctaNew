"""Price-change timing and chronological incremental prediction; not a strategy backtest."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from .pf_pm_leadlag_data import DEST,REVIEW,DAYS,LAGS,PHASES,sha,write,check


def change(x,k=1):
    if k<=0:raise ValueError('nonpositive change horizon')
    y=np.full(len(x),np.nan);y[k:]=x[k:]-x[:-k];return y


def pairs(x,y,lag):
    if len(x)!=len(y) or abs(lag)>=len(x):raise ValueError('invalid lag pair')
    return (x[:len(x)-lag],y[lag:]) if lag>=0 else (x[-lag:],y[:len(y)+lag])


def moments(x,y):
    ok=np.isfinite(x)&np.isfinite(y);a=x[ok];b=y[ok];n=len(a)
    if n<3:return np.array([n,0.,0.,0.])
    a=a-a.mean();b=b-b.mean()
    return np.array([n,np.dot(a,a),np.dot(b,b),np.dot(a,b)])


def correlations(m):
    den=np.sqrt(m[...,1]*m[...,2]);return np.divide(m[...,3],den,out=np.zeros_like(den),where=den>0)


def lag_stats(x,y):
    return np.array([moments(*pairs(x,y,int(l))) for l in LAGS])


def directional(c):
    return np.array([c[np.flatnonzero(LAGS==l)[0]]-c[np.flatnonzero(LAGS==-l)[0]] for l in LAGS[LAGS>0]])


def inputs():
    b=json.loads((DEST/'build.json').read_text());grids=[]
    for s in b['statuses']:
        if s['status']!='built':continue
        if sha(s['grid'])!=b['grids'][s['grid']]:raise ValueError('grid changed')
        grids.append((s,pd.read_parquet(s['grid'])))
    return b,grids


def valid_mid(g,venue,clock,age):
    a=g[f'{venue}_{clock}_mid'].to_numpy().copy()
    ok=(g[f'{venue}_{clock}_age'].to_numpy()>=0)&(g[f'{venue}_{clock}_age'].to_numpy()<=age)
    a[~ok]=np.nan;return a


def analyze_correlations(grids):
    rows=[];daily=[];eligible=[];null_segments=[]
    for clock in ('recv','source'):
        for age in (1000,5000):
            for pi,(lo,hi) in enumerate(PHASES):
                day_m={};counts={};eligible_counts={}
                for s,g in grids:
                    d=s['day'];x=change(valid_mid(g,'pm',clock,age));y=change(valid_mid(g,'pf',clock,age))
                    ix=(g.rel.to_numpy()>=lo)&(g.rel.to_numpy()<hi);x=x[ix];y=y[ix]
                    m=lag_stats(x,y);day_m[d]=day_m.get(d,np.zeros_like(m))+m;counts[d]=counts.get(d,0)+1
                    eligible_counts[d]=eligible_counts.get(d,0)+int((np.isfinite(x)&np.isfinite(y)).sum())
                    if clock=='recv' and age==1000:null_segments.append((pi,x,y))
                total=np.sum(list(day_m.values()),axis=0);c=correlations(total)
                for d,m in day_m.items():
                    daily.append(dict(day=d,clock=clock,max_age_ms=age,phase=[lo,hi],windows=counts[d],
                        correlations=correlations(m).tolist(),pair_counts=m[:,0].astype(int).tolist()))
                rows.append(dict(clock=clock,max_age_ms=age,phase=[lo,hi],windows=sum(counts.values()),
                    lag_ms=(LAGS*100).tolist(),correlations=c.tolist(),pair_counts=total[:,0].astype(int).tolist(),
                    peak_positive_correlation_lag_ms=int(LAGS[np.argmax(c)]*100),peak_correlation=float(c.max()),
                    positive_lag_asymmetry=directional(c).tolist()))
                eligible.append(dict(clock=clock,max_age_ms=age,phase=[lo,hi],expected_grid=int(sum(counts.values())*(hi-lo)*10),
                    paired_valid_changes=sum(eligible_counts.values()),per_day=eligible_counts))
    return rows,daily,eligible,null_segments


def shifted_null(segments,observed,draws=500):
    if draws<200:raise ValueError('undersampled null')
    rng=np.random.default_rng(20261110);maxima=[]
    for b in range(draws):
        mm=np.zeros((4,len(LAGS),4))
        for phase,x,y in segments:
            shift=int(rng.integers(100,len(x)-99));xx=np.roll(x,shift)
            mm[phase]+=lag_stats(xx,y)
        ds=np.array([directional(correlations(m)) for m in mm]);maxima.append(float(abs(ds).max()))
        if b%100==99:print('NULL',b+1,flush=True)
    maxima=np.array(maxima);out=[]
    for r in observed:
        if (r['clock'],r['max_age_ms'])!=('recv',1000):continue
        a=np.array(r['positive_lag_asymmetry'])
        out.append(dict(phase=r['phase'],positive_lag_ms=(LAGS[LAGS>0]*100).tolist(),asymmetry=a.tolist(),
            family_p=[float((1+(maxima>=abs(v)).sum())/(draws+1)) for v in a]))
    return dict(draws=draws,seed=20261110,family=28,null_max_p95=float(np.quantile(maxima,.95)),
        null_max_p99=float(np.quantile(maxima,.99)),results=out,
        interpretation='Timing alignment diagnostic; no confidence interval, independent-day validation, venue-causality or trading claim.')


def feature_table(grids,h):
    parts=[];status=[]
    for s,g in grids:
        pf=valid_mid(g,'pf','recv',1000);pm=valid_mid(g,'pm','recv',1000);bn=g.bn.to_numpy();t=g.rel.to_numpy()
        pr=[change(pf,k) for k in (2,10,50)];br=[change(np.log(bn),k)*10000 for k in (3,10,50)]
        sensitivity=np.clip(pf/100*(1-pf/100),0,1)/np.sqrt(np.maximum(300-t,1))
        base=np.column_stack(pr+br+[v*sensitivity for v in br]+[pf,g.pf_recv_spread.to_numpy(),t/300,
            (t>=60).astype(float),(t>=180).astype(float),(t>=240).astype(float)])
        extra=np.column_stack([change(pm,k) for k in (2,10,50)])
        k=int(round(h*10));target=np.full(len(pf),np.nan);target[:-k]=pf[k:]-pf[:-k]
        use=np.arange(50,2950,10);ok=np.isfinite(base[use]).all(1)&np.isfinite(extra[use]).all(1)&np.isfinite(target[use])
        status.append(dict(day=s['day'],ws=s['ws'],horizon_s=h,scheduled=len(use),eligible=int(ok.sum()),excluded_missing_or_stale=int((~ok).sum())))
        idx=use[ok];z=pd.DataFrame(base[idx],columns=[f'b{i}' for i in range(base.shape[1])])
        for i in range(3):z[f'p{i}']=extra[idx,i]
        z['day']=s['day'];z['ws']=s['ws'];z['t']=t[idx];z['target']=target[idx];parts.append(z)
    return pd.concat(parts,ignore_index=True),status


def predict(train,test,columns):
    if train.empty or test.empty or train.day.max()>=test.day.min():raise ValueError('noncausal split')
    a=train[columns].to_numpy();b=test[columns].to_numpy();mu=a.mean(0);sd=a.std(0);sd[sd<1e-10]=1
    a=(a-mu)/sd;b=(b-mu)/sd;mean=train.target.mean();y=train.target.to_numpy()-mean
    coef=np.linalg.solve(a.T@a+np.eye(a.shape[1]),a.T@y)
    return b@coef+mean


def prediction_study(grids):
    results=[];statuses=[];predictions=[]
    for h in (.5,1.,5.):
        a,ss=feature_table(grids,h);statuses+=ss;base=[c for c in a if c.startswith('b')];extra=base+['p0','p1','p2']
        for day in DAYS[3:]:
            tr=a[a.day<day];te=a[a.day==day].copy()
            if len(tr)<200 or len(te)<50:
                results.append(dict(horizon_s=h,day=day,status='insufficient_rows',n=len(te)));continue
            te['base']=predict(tr,te,base);te['plus_pm']=predict(tr,te,extra);te['horizon_s']=h
            y=te.target.to_numpy();b=te.base.to_numpy();p=te.plus_pm.to_numpy()
            bm=float(np.mean((y-b)**2));em=float(np.mean((y-p)**2))
            results.append(dict(horizon_s=h,day=day,status='scored',n=len(te),windows=int(te.ws.nunique()),train_days=sorted(tr.day.unique()),
                baseline_mse=bm,plus_pm_mse=em,zero_mse=float(np.mean(y*y)),improvement_percent=100*(bm-em)/bm if bm else None,
                baseline_correlation=float(correlations(moments(b,y))),plus_pm_correlation=float(correlations(moments(p,y)))))
            predictions.append(te[['day','ws','t','target','base','plus_pm','horizon_s']])
    pd.concat(predictions,ignore_index=True).to_parquet(DEST/'predictions.parquet',index=False)
    return results,statuses


def main():
    declaration=check();build,grids=inputs()
    rows,daily,eligible,segments=analyze_correlations(grids)
    null=shifted_null(segments,rows);prediction,pred_status=prediction_study(grids)
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=declaration,build_sha=sha(DEST/'build.json'),
        population_status=build['statuses'],correlations=rows,daily_correlations=daily,coverage=eligible,
        null=null,prediction=prediction,prediction_status=pred_status,predictions_sha=sha(DEST/'predictions.parquet'),
        complete_calendar_days_observed=0,live_changes=False,protected_prices_read=False,
        limitations=['Scheduled-window sample, not a complete-day census; no intervals or independent validation.',
        'Different contractual settlement definitions and strikes; price gaps are not executable arbitrage.',
        'Quote tick sizes, batching and different settlement sensitivity can create apparent timing differences.',
        'Local receipt ordering includes network/collector latency; source timestamps are different venue clocks.',
        'Incremental chronological model performance is exploratory; no fees, queue fills, own latency, or PnL evaluated.'])
    write(DEST/'report.json',result);write(REVIEW/'receipt.json',result)
    print('DONE',len(grids),'matched windows',flush=True)


if __name__=='__main__':main()
