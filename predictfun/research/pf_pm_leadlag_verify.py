"""Independent artifact arithmetic/target verification for the sampled lead-lag study."""
import datetime as dt
import json
import math
import subprocess
import numpy as np
import pandas as pd
from .pf_pm_leadlag_data import DEST,REVIEW,HERE,sha,write


def equal(actual,expected,label):
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)) or not math.isclose(float(actual),float(expected),rel_tol=1e-8,abs_tol=1e-9):
        raise ValueError('saved artifact mismatch: '+label)


def independent_correlation(grids,clock,age,phase,lag):
    xs=[];ys=[];n=0
    for g in grids:
        arrays=[]
        for venue in ('pm','pf'):
            p=g[f'{venue}_{clock}_mid'].to_numpy().copy();a=g[f'{venue}_{clock}_age'].to_numpy()
            p[(a<0)|(a>age)|~np.isfinite(a)]=np.nan
            d=np.r_[np.nan,np.diff(p)];arrays.append(d[(g.rel>=phase[0])&(g.rel<phase[1])])
        x,y=arrays
        i=np.arange(len(x));j=i+lag;inside=(j>=0)&(j<len(y));i=i[inside];j=j[inside]
        valid=np.isfinite(x[i])&np.isfinite(y[j]);a=x[i[valid]];b=y[j[valid]];n+=len(a)
        if len(a)>=3:xs.append(a-a.mean());ys.append(b-b.mean())
    a=np.concatenate(xs);b=np.concatenate(ys)
    corr=0. if not np.std(a) or not np.std(b) else float(np.corrcoef(a,b)[0,1])
    return corr,n


def controls():
    x=np.arange(20,dtype=float);g=pd.DataFrame(dict(rel=np.arange(20)*.1,
        pm_recv_mid=x,pf_recv_mid=np.r_[0.,0.,x[:-2]],pm_recv_age=0.,pf_recv_age=0.))
    c,n=independent_correlation([g],'recv',1000,[.5,2],2)
    # Constant increments have zero variance; a nonconstant planted path must be recovered.
    g.pm_recv_mid=x*x;g.pf_recv_mid=np.r_[0.,0.,(x*x)[:-2]]
    c,n=independent_correlation([g],'recv',1000,[.5,2],2);equal(c,1,'planted lag')
    refused=0
    for a,b in ((1.,2.),(float('nan'),1.)):
        try:equal(a,b,'corrupt control')
        except ValueError:refused+=1
    if refused!=2:raise ValueError('verification controls failed')
    return dict(planted_lag_recovered=True,known_bad_refusals=refused)


def main():
    ctl=controls();path=DEST/'report.json';r=json.loads(path.read_text());b=json.loads((DEST/'build.json').read_text())
    if sha(DEST/'build.json')!=r['build_sha']:raise ValueError('build changed')
    for rel,digest in r['declaration']['source_sha'].items():
        if sha(HERE.parents[1]/rel)!=digest:raise ValueError('source changed')
    grids={}
    for s in b['statuses']:
        if s['status']!='built':continue
        if sha(s['grid'])!=b['grids'][s['grid']]:raise ValueError('grid changed')
        grids[s['ws']]=pd.read_parquet(s['grid'])
    count=0
    for row in r['correlations']:
        for lag,c,n in zip(row['lag_ms'],row['correlations'],row['pair_counts']):
            expected,pop=independent_correlation(list(grids.values()),row['clock'],row['max_age_ms'],row['phase'],lag//100)
            equal(c,expected,'correlation');equal(n,pop,'pair count');count+=1
        k=int(np.argmax(row['correlations']));equal(row['peak_positive_correlation_lag_ms'],row['lag_ms'][k],'peak lag')
    if sha(DEST/'predictions.parquet')!=r['predictions_sha']:raise ValueError('predictions changed')
    p=pd.read_parquet(DEST/'predictions.parquet')
    if p.duplicated(['ws','t','horizon_s']).any():raise ValueError('duplicate prediction observation')
    for (ws,h),g in p.groupby(['ws','horizon_s']):
        a=grids[ws].pf_recv_mid.to_numpy().copy();age=grids[ws].pf_recv_age.to_numpy();a[(age<0)|(age>1000)]=np.nan
        i=np.rint(g.t.to_numpy()*10).astype(int);k=int(round(h*10));target=a[i+k]-a[i]
        if not np.allclose(g.target,target,rtol=0,atol=1e-10):raise ValueError('future target mismatch')
    for x in r['prediction']:
        if x['status']!='scored':continue
        g=p[(p.day==x['day'])&(p.horizon_s==x['horizon_s'])]
        if max(x['train_days'])>=x['day']:raise ValueError('chronological split mismatch')
        equal(x['n'],len(g),'prediction population')
        for col,key in [('base','baseline_mse'),('plus_pm','plus_pm_mse')]:equal(x[key],np.mean((g.target-g[col])**2),key)
        equal(x['improvement_percent'],100*(x['baseline_mse']-x['plus_pm_mse'])/x['baseline_mse'],'MSE change')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        verifier_sha=sha(__file__),report_sha=sha(path),controls=ctl,matched_windows=len(grids),correlations_verified=count,
        prediction_rows_verified=len(p),targets_and_MSE_verified=True,
        limitation='Arithmetic verification does not establish venue-clock synchronization or tradable performance.')
    write(DEST/'verification.json',out);write(REVIEW/'verification.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
