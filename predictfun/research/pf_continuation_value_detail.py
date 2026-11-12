"""Declared component calibration diagnostics; no policy selection or new fitting."""
import datetime as dt
import json
import subprocess
import sys
from . import pf_continuation_spec as S
from .pf_continuation_v3 import configure
configure()


def declare():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.WORK,text=True).strip();rel='predictfun/research/pf_continuation_value_detail.py'
    if subprocess.check_output(['git','show',commit+':'+rel],cwd=S.WORK)!=(S.WORK/rel).read_bytes():raise ValueError('commit diagnostic first')
    S.write('value_detail_declaration.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=commit,source_sha=S.sha(__file__),
        experiment_declaration_sha=S.sha(S.OUT/'declaration.json'),design='Before any full-day build/fit readout: report 10s gain/loss/risk means and errors for all existing families, paired day intervals for queue calibrated versus raw/market calibrated/training baseline, fixed score bins (-inf,-5,0,5,inf) in cents, and KEEP/PULL value decomposition on each declared zero-threshold kept subset. Positive aggregation and changed-sign refusal controls. No new fit, candidate, threshold or population.'))


def components(a,family):
    result={}
    for b in ('keep','pull'):
        pre=f'{family}_{b}_10000_'
        result[b+'_gain']=a[pre+'gain'].to_numpy();result[b+'_loss']=a[pre+'loss'].to_numpy()
        result[b+'_risk']=a[pre+'risk_pos'].to_numpy()-a[pre+'risk_neg'].to_numpy()
    result['score']=result['keep_gain']-result['keep_loss']-result['keep_risk']-result['pull_gain']+result['pull_loss']+result['pull_risk']
    return result


def run():
    import numpy as np
    import pandas as pd
    from .pf_fill_value_report import interval
    from .pf_continuation_report import FAMILIES
    d=json.loads((S.OUT/'value_detail_declaration.json').read_text())
    if S.sha(__file__)!=d['source_sha'] or S.sha(S.OUT/'declaration.json')!=d['experiment_declaration_sha']:raise ValueError('declaration/source identity')
    # Demonstrate the reporting algebra and a known-bad changed loss sign.
    row={}
    for b in ('keep','pull'):
        for k,v in [('gain',4 if b=='keep' else 1),('loss',2 if b=='keep' else 3),('risk_pos',1),('risk_neg',0)]:row[f'test_{b}_10000_{k}']=[v]
    control=components(pd.DataFrame(row),'test')
    if control['score'][0]!=4:raise ValueError('component positive control')
    bad=control['keep_gain']+control['keep_loss']-control['keep_risk']-control['pull_gain']+control['pull_loss']+control['pull_risk']
    if np.array_equal(control['score'],bad):raise ValueError('wrong-sign falsifier failed')
    result=json.loads((S.OUT/'result.json').read_text());p=S.OUT/'predictions.parquet'
    if S.sha(p)!=result['predictions_sha']:raise ValueError('prediction identity')
    a=pd.read_parquet(p);truth={f'{b}_{k}':a[f'{b}_10000_{k}'].to_numpy() for b in ('keep','pull') for k in ('gain','loss','risk')}
    truth['score']=a.keep_10000_value.to_numpy()-a.pull_10000_value.to_numpy()
    metrics=[];kept=[];bins=[];errors={}
    for family in FAMILIES:
        z=components(a,family);np.testing.assert_allclose(z['score'],a[family+'_score_10000'],atol=1e-8,rtol=0)
        for key,y in truth.items():
            x=z[key];errors[family,key]=(x-y)**2
            metrics.append(dict(family=family,component=key,n=len(a),predicted_mean_c=float(x.mean()),actual_mean_c=float(y.mean()),mse=float(errors[family,key].mean())))
        if family=='baseline':continue
        mask=z['score']>=0
        kept.append(dict(family=family,n=int(mask.sum()),predicted={k:float(v[mask].sum()) for k,v in z.items()},actual={k:float(v[mask].sum()) for k,v in truth.items()}))
        group=pd.cut(z['score'],[-np.inf,-5,0,5,np.inf],right=False)
        for label in group.categories:
            ix=group==label
            if ix.any():bins.append(dict(family=family,bin=str(label),n=int(ix.sum()),predicted_c=float(z['score'][ix].mean()),actual_c=float(truth['score'][ix].mean())))
    diffs=[]
    for base in ('baseline','market_cal','queue_raw'):
        for key in truth:diffs.append(dict(model='queue_cal',baseline=base,component=key,**interval(a,errors['queue_cal',key]-errors[base,key])))
    S.write('value_detail.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',n=len(a),days=S.TEST,validation=False,
        declaration_sha=S.sha(S.OUT/'value_detail_declaration.json'),result_sha=S.sha(S.OUT/'result.json'),metrics=metrics,kept=kept,bins=bins,differences=diffs,
        scope='Positive/negative parts of branch aggregate markout and inventory-square cost, on the same sampled neutral actions. Overlapping local utility, not portfolio PnL. No new model or policy selection.'))

if __name__=='__main__':declare() if sys.argv[1]=='declare' else run()
