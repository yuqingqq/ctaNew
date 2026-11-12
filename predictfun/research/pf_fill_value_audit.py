"""Supplementary physical-support audit; never changes fitted forecasts."""
import datetime as dt
import json
import subprocess
from pathlib import Path
from . import pf_fill_value_spec as S


def declare():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.WORK,text=True).strip()
    p=str(Path(__file__).resolve().relative_to(S.WORK))
    if subprocess.check_output(['git','show',commit+':'+p],cwd=S.WORK)!=Path(__file__).read_bytes():raise ValueError('commit audit first')
    S.write('audit_declaration.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=commit,
        source_path=p,source_sha=S.sha(__file__),original_declaration_sha=S.sha(S.OUT/'declaration.json'),
        scope='Additional necessary physical-support checks declared while fitting, before inspecting numerical model results. No model/threshold change, no new candidate. Report rather than repair incoherent component forecasts.',
        bounds='At price p in [0,100], buy gain <= (100-p)*E[Q], loss <= p*E[Q]; sell reverses the bounds. Also E[gain]+E[loss] <= 100*E[Q]. Zero expected quantity requires zero expected absolute markout. These are necessary bounds, not a sufficient joint-distribution or calibration proof.',
        controls='Valid forecast accepted; planted zero-quantity positive-value and excessive adverse-value forecasts flagged and refused by support gate. Saved input/source/model/prediction identities unchanged.'))


def violations(a,prefix):
    import numpy as np
    price=a.order_price.to_numpy();q=a[prefix+'_quantity'].to_numpy()
    gain=a[prefix+'_gain'].to_numpy();loss=a[prefix+'_loss'].to_numpy()
    if not (np.isfinite(price).all() and ((price>=0)&(price<=100)).all()):raise ValueError('price support')
    gmax=np.where(a.side.eq(0),100-price,price)*q
    lmax=np.where(a.side.eq(0),price,100-price)*q
    return dict(zero_quantity_nonzero_value=(q<=1e-12)&((gain+loss)>1e-8),
                gain_over_price_bound=gain>gmax+1e-8,loss_over_price_bound=loss>lmax+1e-8,
                absolute_mark_over_bound=gain+loss>100*q+1e-8)


def support_gate(flags):
    if any(v.any() for v in flags.values()):raise ValueError('incoherent fill/value support')


def run():
    import numpy as np
    import pandas as pd
    S.check();d=json.loads((S.OUT/'audit_declaration.json').read_text())
    if S.sha(__file__)!=d['source_sha']:raise ValueError('audit source changed')
    a=pd.DataFrame(dict(side=[0.,1.],order_price=[40.,60.],x_quantity=[2.,2.],x_gain=[1.,1.],x_loss=[1.,1.]))
    support_gate(violations(a,'x'))
    for bad in (a.assign(x_quantity=0.),a.assign(x_loss=1000.)):
        try:support_gate(violations(bad,'x'))
        except ValueError:pass
        else:raise ValueError('known-bad support accepted')
    fit=json.loads((S.OUT/'fit.json').read_text());result=json.loads((S.OUT/'result.json').read_text())
    if S.sha(S.OUT/'predictions.parquet')!=fit['predictions_sha'] or result['fit_sha']!=S.sha(S.OUT/'fit.json'):raise ValueError('report/predictions identity')
    for f in fit['folds']:
        for p,h in f['models_sha'].items():
            if S.sha(S.OUT/'models'/p)!=h:raise ValueError('model identity')
    a=pd.read_parquet(S.OUT/'predictions.parquet');rows=[]
    for family in ('baseline','market','queue'):
        for branch in ('keep','reset'):
            prefix=family+'_'+branch;flags=violations(a,prefix)
            union=np.logical_or.reduce(list(flags.values()))
            rows.append(dict(family=family,branch=branch,n=len(a),counts={k:int(v.sum()) for k,v in flags.items()},
                             any_violation=int(union.sum()),necessary_support_pass=not bool(union.any()),
                             sum_predicted_absolute_value_c_on_violations=float((a[prefix+'_gain']+a[prefix+'_loss'])[union].sum())))
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='ARTIFACT_INTEGRITY_PASS',
             n=len(a),days=S.TEST,source_commit=d['source_commit'],audit_declaration_sha=S.sha(S.OUT/'audit_declaration.json'),
             result_sha=S.sha(S.OUT/'result.json'),predictions_sha=fit['predictions_sha'],models_verified=fit['models'],
             forecast_support=rows,all_forecasts_meet_necessary_support=all(x['necessary_support_pass'] for x in rows),
             controls='Valid physical forecast accepted; planted impossible value forecasts detected and refused by support gate.',
             meaning='Integrity pass does not imply calibrated or jointly coherent forecasts. Necessary-support failures are recorded without altering the frozen predictions; original results remain as measured.')
    S.write('audit.json',out)
    print('FILL_VALUE_AUDIT_SAVED',flush=True)


if __name__=='__main__':
    import sys
    declare() if sys.argv[1]=='declare' else run()
